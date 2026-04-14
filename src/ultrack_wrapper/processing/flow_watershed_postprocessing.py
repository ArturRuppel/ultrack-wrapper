"""Post-processing for flow-watershed cell segmentation.

Includes: morphological smoothing, boundary smoothing, and hole-filling
based on cellpose probability.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.measure import find_contours
from scipy.interpolate import UnivariateSpline


def _resample_contour(contour: np.ndarray, num_points: int = 100) -> np.ndarray:
    """Resample a contour to a fixed number of points."""
    if len(contour) < 2:
        return contour
    # Parameterize by arc length
    diffs = np.diff(contour, axis=0)
    dists = np.sqrt((diffs**2).sum(axis=1))
    cumsum = np.concatenate([[0], np.cumsum(dists)])
    # Resample
    new_s = np.linspace(0, cumsum[-1], num_points)
    resampled = np.column_stack([
        np.interp(new_s, cumsum, contour[:, 0]),
        np.interp(new_s, cumsum, contour[:, 1]),
    ])
    return resampled


def _smooth_contour_spline(contour: np.ndarray, smoothness: float = 0.5) -> np.ndarray:
    """Smooth a contour using B-spline interpolation.

    Parameters
    ----------
    contour : np.ndarray
        (N, 2) array of (row, col) points
    smoothness : float
        Smoothing factor (0-1). Higher = smoother.

    Returns
    -------
    np.ndarray
        Smoothed contour
    """
    if len(contour) < 4:
        return contour

    # Close the contour for smoothness
    contour_closed = np.vstack([contour, contour[0:1]])

    # Parameterize by index
    t = np.arange(len(contour_closed))

    try:
        # Fit splines with smoothing
        s_val = smoothness * len(contour) ** 2
        spl_row = UnivariateSpline(t, contour_closed[:, 0], s=s_val, k=min(3, len(contour_closed) - 1))
        spl_col = UnivariateSpline(t, contour_closed[:, 1], s=s_val, k=min(3, len(contour_closed) - 1))

        # Evaluate on denser grid
        t_smooth = np.linspace(0, len(contour) - 1, len(contour) * 2)
        smoothed = np.column_stack([spl_row(t_smooth), spl_col(t_smooth)])
        return smoothed
    except Exception:
        return contour


def morphological_smoothing(
    labels: np.ndarray,
    opening_radius: int = 1,
    closing_radius: int = 1,
) -> np.ndarray:
    """Apply opening then closing to each cell label.

    Parameters
    ----------
    labels : np.ndarray
        Label image (H, W)
    opening_radius : int
        Radius for binary_opening (removes small noise)
    closing_radius : int
        Radius for binary_closing (fills small holes)

    Returns
    -------
    np.ndarray
        Smoothed labels
    """
    result = np.zeros_like(labels)

    for label_id in np.unique(labels):
        if label_id == 0:
            continue

        mask = labels == label_id

        # Opening: remove small noise spikes
        if opening_radius > 0:
            mask = ndimage.binary_opening(mask, iterations=opening_radius)

        # Closing: fill small holes
        if closing_radius > 0:
            mask = ndimage.binary_closing(mask, iterations=closing_radius)

        result[mask] = label_id

    return result


def boundary_smoothing(
    labels: np.ndarray,
    smoothness: float = 0.5,
) -> np.ndarray:
    """Smooth cell boundaries using contour smoothing.

    Parameters
    ----------
    labels : np.ndarray
        Label image (H, W)
    smoothness : float
        Smoothing factor (0-1). Higher = smoother.

    Returns
    -------
    np.ndarray
        Labels with smoothed boundaries
    """
    H, W = labels.shape
    result = np.zeros_like(labels)

    for label_id in np.unique(labels):
        if label_id == 0:
            continue

        mask = (labels == label_id).astype(np.uint8)

        # Find contours
        contours = find_contours(mask, level=0.5)

        if not contours:
            result[mask.astype(bool)] = label_id
            continue

        # Use largest contour (external boundary)
        contour = contours[0]

        # Smooth the contour
        if len(contour) > 3:
            smoothed_contour = _smooth_contour_spline(contour, smoothness=smoothness)

            # Re-render the smoothed contour back to a mask
            from skimage.draw import polygon
            try:
                rows = np.clip(smoothed_contour[:, 0].astype(int), 0, H - 1)
                cols = np.clip(smoothed_contour[:, 1].astype(int), 0, W - 1)
                rr, cc = polygon(rows, cols, shape=(H, W))
                result[rr, cc] = label_id
            except Exception:
                # Fallback to original mask if rendering fails
                result[mask.astype(bool)] = label_id
        else:
            result[mask.astype(bool)] = label_id

    return result


def trim_low_probability_boundaries(
    labels: np.ndarray,
    cellpose_prob: np.ndarray,
    prob_threshold: float = 0.5,
) -> np.ndarray:
    """Remove boundary pixels with low cellpose probability.

    For each cell, finds boundary pixels that touch background and removes them
    if their cellpose probability is below the threshold.

    Parameters
    ----------
    labels : np.ndarray
        Label image (H, W)
    cellpose_prob : np.ndarray
        Cellpose probability map (H, W), values in [0, 1]
    prob_threshold : float
        Probability threshold (0-1). Pixels below this at boundaries are removed.

    Returns
    -------
    np.ndarray
        Labels with low-probability boundary pixels removed
    """
    result = labels.copy()
    background = labels == 0

    for label_id in np.unique(labels):
        if label_id == 0:
            continue

        cell_mask = labels == label_id

        # Dilate to find boundary pixels (dilated - original)
        dilated = ndimage.binary_dilation(cell_mask)
        boundary = dilated & ~cell_mask & background

        if not boundary.any():
            continue

        # Find current boundary pixels of the cell
        eroded = ndimage.binary_erosion(cell_mask)
        current_boundary = cell_mask & ~eroded

        # Check cellpose probability at boundary
        boundary_points = np.argwhere(current_boundary)
        for point in boundary_points:
            if cellpose_prob[point[0], point[1]] < prob_threshold:
                # Remove this boundary pixel
                result[point[0], point[1]] = 0

    return result


def postprocess_flow_watershed(
    labels: np.ndarray,
    cellpose_prob: np.ndarray | None = None,
    opening_radius: int = 1,
    closing_radius: int = 1,
    boundary_smoothness: float = 0.5,
    fill_holes_threshold: float = 0.5,
) -> np.ndarray:
    """Complete post-processing pipeline for flow-watershed segmentation.

    Parameters
    ----------
    labels : np.ndarray
        Raw flow-watershed labels (H, W)
    cellpose_prob : np.ndarray, optional
        Cellpose probability map (H, W) for boundary trimming
    opening_radius : int
        Morphological opening radius (removes noise)
    closing_radius : int
        Morphological closing radius (fills holes)
    boundary_smoothness : float
        Boundary smoothing factor (0-1)
    fill_holes_threshold : float
        Cellpose probability threshold for removing low-confidence boundary pixels

    Returns
    -------
    np.ndarray
        Post-processed labels
    """
    result = labels.copy()

    # 1. Morphological smoothing (opening then closing per cell to fill holes)
    if opening_radius > 0 or closing_radius > 0:
        result = morphological_smoothing(result, opening_radius, closing_radius)

    # 2. Boundary smoothing (contour-based smoothing)
    if boundary_smoothness > 0:
        result = boundary_smoothing(result, smoothness=boundary_smoothness)

    # 3. Trim low-probability boundaries
    # Remove boundary pixels where cellpose prob is below threshold
    if cellpose_prob is not None and fill_holes_threshold > 0:
        result = trim_low_probability_boundaries(result, cellpose_prob, fill_holes_threshold)

    return result.astype(np.int32)
