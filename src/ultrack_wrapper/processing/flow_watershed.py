"""Flow-guided watershed cell segmentation using iterative expansion."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def flow_guided_watershed_iterative(
    nuclear_labels: np.ndarray,
    flow_field: np.ndarray,
    cellpose_prob: np.ndarray | None = None,
    flow_scale: float = 1.0,
    cellpose_prob_threshold: float = 0.0,
    flow_smoothing_sigma: float = 0.0,
    max_iterations: int = 50,
    uniform_growth_rate: float = 0.2,
) -> np.ndarray:
    """
    Flow-guided watershed using iterative expansion approach.

    Expands boundaries from nuclear seeds with velocity modulated by flow field.
    Flow points toward cell centers, so expansion is allowed where -flow points outward.

    Parameters
    ----------
    nuclear_labels : np.ndarray
        Integer label map of segmented nuclei (shape: H, W).
    flow_field : np.ndarray
        2D vector field from cellpose pointing toward cell centers (shape: H, W, 2).
    cellpose_prob : np.ndarray, optional
        Confidence map from cellpose (shape: H, W).
    flow_scale : float
        Scaling factor for flow influence on expansion probability (0.0-3.0).
    cellpose_prob_threshold : float
        Mask out regions with probability below this value.
    flow_smoothing_sigma : float
        Gaussian smoothing of flow field.
    max_iterations : int
        Maximum number of expansion iterations (default 50).
    uniform_growth_rate : float
        Baseline expansion probability when flow is neutral/inward (0.0-1.0, default 0.2).

    Returns
    -------
    np.ndarray
        Integer label map with expanded cell boundaries.
    """
    H, W = nuclear_labels.shape

    # Smooth flow field if requested
    if flow_smoothing_sigma > 0:
        from scipy.ndimage import gaussian_filter
        flow_field = np.stack([
            gaussian_filter(flow_field[..., 0], sigma=flow_smoothing_sigma),
            gaussian_filter(flow_field[..., 1], sigma=flow_smoothing_sigma),
        ], axis=-1)

    # Compute nuclear centroids
    centroids = {}
    for label_id in np.unique(nuclear_labels):
        if label_id > 0:
            points = np.argwhere(nuclear_labels == label_id)
            if len(points) > 0:
                centroids[label_id] = points.mean(axis=0)

    # Apply probability mask if provided
    prob_mask = None
    if cellpose_prob is not None:
        prob_mask = cellpose_prob >= cellpose_prob_threshold

    # Start with nuclei
    result = nuclear_labels.copy().astype(np.int32)

    # Iterative expansion using morphological dilation with flow-guided direction
    for iteration in range(max_iterations):
        unassigned = result == 0

        if not unassigned.any():
            break

        # Dilate each label by one pixel where flow allows expansion
        for label_id in np.unique(result[result > 0]):
            if label_id not in centroids:
                continue

            labeled = result == label_id
            dilated = ndimage.binary_dilation(labeled)
            boundary_candidates = dilated & unassigned

            if not boundary_candidates.any():
                continue

            # For boundary candidates, compute expansion direction (radial outward from center)
            center = centroids[label_id]
            boundary_points = np.argwhere(boundary_candidates)

            # Compute radial direction from center to each boundary pixel (y, x)
            radial_dirs = boundary_points - center  # Shape: (N, 2)
            radial_norms = np.linalg.norm(radial_dirs, axis=1, keepdims=True)
            # Avoid division by zero
            radial_dirs = np.where(radial_norms > 1e-6, radial_dirs / radial_norms, 0)

            # Negative flow (points outward since flow points inward to center)
            neg_flow = -flow_field[boundary_points[:, 0], boundary_points[:, 1]]

            # Dot product: radial_dir · (-flow)
            # Positive = flow points outward (good direction to expand)
            flow_alignment = np.sum(radial_dirs * neg_flow, axis=1)

            # Blend uniform growth with flow-guided expansion
            # uniform_growth_rate: baseline probability (e.g., 0.2)
            # When flow_alignment is positive, boost the probability
            flow_boost = np.clip(flow_alignment * flow_scale, 0, 1)
            expand_prob = uniform_growth_rate + (1.0 - uniform_growth_rate) * flow_boost

            # Apply probability mask
            if prob_mask is not None:
                expand_prob = np.where(prob_mask[boundary_points[:, 0], boundary_points[:, 1]],
                                      expand_prob, 0.0)

            # Stochastic expansion: expand where random < expand_prob
            expand_mask = np.random.rand(len(boundary_points)) < expand_prob
            expand_coords = boundary_points[expand_mask]

            if len(expand_coords) > 0:
                result[expand_coords[:, 0], expand_coords[:, 1]] = label_id

    return result


def flow_guided_watershed(
    nuclear_labels: np.ndarray,
    flow_field: np.ndarray,
    cellpose_prob: np.ndarray | None = None,
    flow_scale: float = 1.0,
    cellpose_prob_threshold: float = 0.0,
    flow_smoothing_sigma: float = 0.0,
    method: str = "iterative",
    max_iterations: int = 50,
    uniform_growth_rate: float = 0.2,
) -> np.ndarray:
    """
    Flow-guided watershed segmentation using individual cell growth and merging.

    Expands each nucleus independently with flow-modulated velocity, then merges
    overlapping regions intelligently to handle cell boundary noise.

    Parameters
    ----------
    nuclear_labels : np.ndarray
        Integer label map of segmented nuclei (shape: H, W).
    flow_field : np.ndarray
        2D vector field from cellpose (shape: H, W, 2).
    cellpose_prob : np.ndarray, optional
        Confidence map from cellpose (shape: H, W).
    flow_scale : float
        Scaling factor for flow influence on expansion probability (0.0-3.0).
    cellpose_prob_threshold : float
        Threshold for cellpose probability mask.
    flow_smoothing_sigma : float
        Gaussian smoothing of flow field.
    method : str
        "iterative" (default and only method - uses individual cell growth and merging).
    max_iterations : int
        Maximum iterations per cell (default: 50).
    uniform_growth_rate : float
        Baseline expansion probability for uniform growth (0.0-1.0, default: 0.2).

    Returns
    -------
    np.ndarray
        Integer label map with expanded cell boundaries.
    """
    if method != "iterative":
        raise ValueError(f"Only 'iterative' method is supported. Got: {method}")

    return flow_guided_watershed_iterative(
        nuclear_labels,
        flow_field,
        cellpose_prob,
        flow_scale,
        cellpose_prob_threshold,
        flow_smoothing_sigma,
        max_iterations,
        uniform_growth_rate,
    )
