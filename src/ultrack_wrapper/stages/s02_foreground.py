"""s02 — Foreground detection from Cellpose flow outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.special import expit
from skimage.exposure import equalize_adapthist
from skimage.filters import median, threshold_otsu, threshold_triangle
from skimage.morphology import ball, binary_opening, binary_closing, binary_dilation, remove_small_objects

from ultrack_wrapper._config import ForegroundConfig


def load_prob_map(prob_path: str | Path) -> np.ndarray:
    """Load a cell probability map.

    Parameters
    ----------
    prob_path : path to ``t*_prob.tif`` — shape ``(Z, Y, X)``

    Returns
    -------
    prob : np.ndarray, shape ``(Z, Y, X)``, float32
    """
    prob = tifffile.imread(str(prob_path)).astype(np.float32)
    pmin, pmax = prob.min(), prob.max()
    if pmax - pmin < 1e-8:
        return np.zeros_like(prob)
    return (prob - pmin) / (pmax - pmin)


def apply_blur(mag: np.ndarray, cfg: ForegroundConfig) -> np.ndarray:
    """Apply median and/or Gaussian blur to magnitude volume."""
    if not cfg.median_filter and not cfg.gaussian_filter:
        return mag
    mag = mag.copy()
    if cfg.median_filter:
        footprint = np.ones((2 * cfg.median_radius + 1,) * 2)
        for z in range(mag.shape[0]):
            mag[z] = median(mag[z], footprint=footprint)
    if cfg.gaussian_filter:
        mag = ndi.gaussian_filter(mag, sigma=(0, cfg.gaussian_sigma, cfg.gaussian_sigma))
    return mag


def apply_clahe(mag: np.ndarray, cfg: ForegroundConfig) -> np.ndarray:
    """Apply CLAHE to magnitude volume per Z-slice.

    Returns a float32 array in [0, 1] range.
    """
    if not cfg.clahe:
        return mag
    kernel_size = cfg.clahe_kernel_size if cfg.clahe_kernel_size > 0 else None
    # Normalize to [0, 1] — equalize_adapthist requires this for float input
    mag_min, mag_max = mag.min(), mag.max()
    if mag_max - mag_min < 1e-8:
        return mag.copy()
    mag_norm = (mag - mag_min) / (mag_max - mag_min)

    out = np.empty_like(mag, dtype=np.float32)
    for z in range(mag.shape[0]):
        out[z] = equalize_adapthist(
            mag_norm[z], clip_limit=cfg.clahe_clip_limit, kernel_size=kernel_size,
        ).astype(np.float32)
    return out


def threshold_magnitude(
    mag: np.ndarray,
    method: str = "fixed",
    threshold: float = 1.0,
    sigmoid_center: float = 1.0,
    sigmoid_steepness: float = 3.0,
) -> np.ndarray:
    """Threshold a flow-magnitude volume into a binary mask.

    Parameters
    ----------
    mag : (Z, Y, X) float32
    method : "fixed", "otsu", "triangle", or "sigmoid"
    threshold : used only when *method* is "fixed"
    sigmoid_center : threshold center for sigmoid method
    sigmoid_steepness : transition steepness for sigmoid method

    Returns
    -------
    mask : (Z, Y, X) bool
    """
    if method == "sigmoid":
        return expit((mag - sigmoid_center) * sigmoid_steepness) > 0.5
    elif method == "otsu":
        t = threshold_otsu(mag)
    elif method == "triangle":
        t = threshold_triangle(mag)
    else:
        t = threshold
    return mag > t


def postprocess(mask: np.ndarray, cfg: ForegroundConfig) -> np.ndarray:
    """Apply post-processing to a binary mask.

    Parameters
    ----------
    mask : (Z, Y, X) bool or uint8
    cfg : ForegroundConfig with post-processing parameters

    Returns
    -------
    mask : (Z, Y, X) uint8 (0/1)
    """
    mask = mask.astype(bool)

    if cfg.fill_holes:
        if cfg.fill_holes_max_size > 0:
            filled = ndi.binary_fill_holes(mask)
            holes = filled & ~mask
            labeled_holes, n = ndi.label(holes)
            for i in range(1, n + 1):
                if ndi.sum(holes, labeled_holes, i) <= cfg.fill_holes_max_size:
                    mask[labeled_holes == i] = True
        else:
            mask = ndi.binary_fill_holes(mask)

    if cfg.morpho_op == "opening":
        mask = binary_opening(mask, footprint=ball(cfg.morpho_radius))
    elif cfg.morpho_op == "closing":
        mask = binary_closing(mask, footprint=ball(cfg.morpho_radius))

    if cfg.remove_small and cfg.remove_small_min_size > 0:
        mask = remove_small_objects(mask, min_size=cfg.remove_small_min_size)

    if cfg.distance_filter:
        # Compute distance transform per Z-slice (XY only)
        seeds = np.zeros_like(mask, dtype=bool)
        for z in range(mask.shape[0]):
            dt_slice = ndi.distance_transform_edt(mask[z])
            seeds[z] = dt_slice >= cfg.distance_filter_min_radius
        if seeds.any():
            # Dilate seeds back to the original mask boundary (reconstruction)
            prev = seeds
            while True:
                expanded = binary_dilation(prev, footprint=ball(1)) & mask
                if np.array_equal(expanded, prev):
                    break
                prev = expanded
            mask = prev

    if cfg.area_filter:
        labeled, n = ndi.label(mask)
        if n > 0:
            areas = ndi.sum(mask, labeled, range(1, n + 1))
            keep = np.zeros(n + 1, dtype=bool)
            for i, area in enumerate(areas, start=1):
                if cfg.area_filter_min <= area <= cfg.area_filter_max:
                    keep[i] = True
            mask = keep[labeled]

    return mask.astype(np.uint8)


def compute_foreground_single(
    prob_path: str | Path,
    cfg: ForegroundConfig,
) -> np.ndarray:
    """Compute foreground mask for a single timepoint.

    Returns
    -------
    mask : (Z, Y, X) uint8
    """
    prob = load_prob_map(prob_path)
    prob = apply_blur(prob, cfg)
    prob = apply_clahe(prob, cfg)
    mask = threshold_magnitude(prob, cfg.method, cfg.threshold, cfg.sigmoid_center, cfg.sigmoid_steepness)
    return postprocess(mask, cfg)


def compute_foreground_from_mag(
    mag: np.ndarray,
    cfg: ForegroundConfig,
) -> np.ndarray:
    """Compute foreground mask from a pre-loaded magnitude volume.

    Used by the widget for interactive preview (avoids re-reading the file).

    Returns
    -------
    mask : (Z, Y, X) uint8
    """
    mag = apply_blur(mag, cfg)
    mag = apply_clahe(mag, cfg)
    mask = threshold_magnitude(mag, cfg.method, cfg.threshold, cfg.sigmoid_center, cfg.sigmoid_steepness)
    return postprocess(mask, cfg)


def discover_prob_files(input_dir: str | Path) -> list[Path]:
    """Return sorted list of ``t*_prob.tif`` files in *input_dir*."""
    return sorted(Path(input_dir).glob("t*_prob.tif"))


def run(
    input_dir: str | Path,
    output_dir: str | Path,
    cfg: ForegroundConfig,
    overwrite: bool = False,
) -> Generator[tuple[int, int, str], None, None]:
    """Process all timepoints, writing per-frame TIFFs and a stacked volume.

    Yields ``(done, total, status_label)`` for progress reporting.
    """
    prob_files = discover_prob_files(input_dir)
    total = len(prob_files)
    if total == 0:
        yield (0, 0, "No t*_prob.tif files found")
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    frames: list[np.ndarray] = []
    for i, prob_path in enumerate(prob_files):
        t_str = prob_path.name.split("_")[0]  # e.g. "t000"
        out_path = out / f"{t_str}_foreground.tif"

        if out_path.exists() and not overwrite:
            frame = tifffile.imread(str(out_path))
        else:
            frame = compute_foreground_single(prob_path, cfg)
            tifffile.imwrite(str(out_path), frame, compression="zlib")

        frames.append(frame)
        yield (i + 1, total, f"{t_str}")

    # Write stacked volume (T, Z, Y, X)
    stacked = np.stack(frames, axis=0)
    tifffile.imwrite(str(out / "foreground.tif"), stacked, compression="zlib")
    yield (total, total, "Done")
