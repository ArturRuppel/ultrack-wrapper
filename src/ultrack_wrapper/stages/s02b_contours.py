"""s02b — Contour (edge) map for Ultrack from Cellpose probability maps.

Ultrack's ``segment()`` requires both a **foreground** mask and a **contours**
array.  Contours encode boundary strength: high at cell boundaries, low inside
cells.  The watershed hierarchy uses this to propose segmentation hypotheses at
multiple scales.

Three methods (validated in ``test_contour_approaches_v2.py``):

1. **Prob-map** — ``1 - sigmoid(prob)``, optionally Gaussian-smoothed.
   Fast, continuous, no segmentation needed.
2. **Watershed UCM** — run watershed at multiple Gaussian blur scales on the
   sigmoidised prob map, average the boundaries → soft UCM.
3. **Combined** — weighted blend of (1) and (2).
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.special import expit
from skimage.feature import peak_local_max
from skimage.segmentation import find_boundaries, watershed

from ultrack_wrapper._config import ContoursConfig


# ── Helpers ─────────────────────────────────────────────────────────────────


def load_prob_map(path: str | Path) -> np.ndarray:
    """Load raw Cellpose probability map as float32 (no normalisation)."""
    return tifffile.imread(str(path)).astype(np.float32)


# ── Approach B: continuous prob-map contour ─────────────────────────────────


def contours_probmap(
    prob_map: np.ndarray,
    sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """``1 - sigmoid(prob)`` as a continuous contour map.

    Returns
    -------
    contours : (Z, Y, X) float32 in [0, 1].  1 = boundary, 0 = cell centre.
    fg : (Z, Y, X) float32 — sigmoid foreground.
    """
    fg = expit(prob_map).astype(np.float32)
    contour = 1.0 - fg

    if sigma > 0:
        contour = ndi.gaussian_filter(contour, sigma=sigma).astype(np.float32)
        cmin, cmax = contour.min(), contour.max()
        if cmax > cmin:
            contour = (contour - cmin) / (cmax - cmin)

    return contour, fg


# ── Approach C: multi-scale watershed UCM ───────────────────────────────────


def _watershed_at_scale(
    prob_sigmoid: np.ndarray,
    blur_sigma: float,
    min_seed_dist: int = 5,
    fg_thresh: float = 0.3,
) -> np.ndarray:
    """Watershed on the prob map blurred at a given scale."""
    if blur_sigma > 0:
        landscape = ndi.gaussian_filter(prob_sigmoid, sigma=blur_sigma)
    else:
        landscape = prob_sigmoid.copy()

    fg_mask = prob_sigmoid > fg_thresh

    coords = peak_local_max(
        landscape, min_distance=min_seed_dist,
        labels=fg_mask.astype(int), exclude_border=False,
    )
    if len(coords) == 0:
        return np.zeros(prob_sigmoid.shape, dtype=np.uint16)

    seed_mask = np.zeros(prob_sigmoid.shape, dtype=bool)
    seed_mask[tuple(coords.T)] = True
    seed_mask = ndi.binary_dilation(seed_mask, iterations=1)
    seed_labels, n_seeds = ndi.label(seed_mask)

    if n_seeds == 0:
        return np.zeros(prob_sigmoid.shape, dtype=np.uint16)

    inverted = 1.0 - prob_sigmoid
    labels = watershed(inverted, markers=seed_labels, mask=fg_mask)
    return labels.astype(np.uint16)


def contours_watershed(
    prob_map: np.ndarray,
    blur_sigmas: list[float] | None = None,
    sigma_contour: float = 0.5,
    min_seed_dist: int = 5,
    fg_thresh: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Multi-scale watershed UCM.

    Returns
    -------
    ucm : (Z, Y, X) float32 in [0, 1]
    fg : (Z, Y, X) float32 — sigmoid foreground
    """
    if blur_sigmas is None:
        blur_sigmas = [0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0]

    fg = expit(prob_map).astype(np.float32)

    contour_sum = np.zeros(prob_map.shape, dtype=np.float32)
    n_valid = 0

    for s in blur_sigmas:
        labels = _watershed_at_scale(fg, blur_sigma=s, min_seed_dist=min_seed_dist,
                                     fg_thresh=fg_thresh)
        if labels.max() == 0:
            continue
        contour_sum += find_boundaries(labels, mode="outer").astype(np.float32)
        n_valid += 1

    if n_valid == 0:
        return np.zeros_like(prob_map, dtype=np.float32), fg

    ucm = contour_sum / n_valid

    if sigma_contour > 0:
        ucm = ndi.gaussian_filter(ucm, sigma=sigma_contour).astype(np.float32)
        ucm_max = ucm.max()
        if ucm_max > 0:
            ucm /= ucm_max

    return ucm, fg


# ── Approach D: combined ────────────────────────────────────────────────────


def contours_combined(
    prob_map: np.ndarray,
    w_prob: float = 0.4,
    w_ws: float = 0.6,
    sigma: float = 1.0,
    blur_sigmas: list[float] | None = None,
    min_seed_dist: int = 5,
    fg_thresh: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted blend of prob-map and watershed contours.

    Returns
    -------
    ucm : (Z, Y, X) float32 in [0, 1]
    fg : (Z, Y, X) float32
    """
    contour_prob, fg = contours_probmap(prob_map, sigma=sigma)
    contour_ws, _ = contours_watershed(
        prob_map, blur_sigmas=blur_sigmas, sigma_contour=sigma,
        min_seed_dist=min_seed_dist, fg_thresh=fg_thresh,
    )

    ucm = w_prob * contour_prob + w_ws * contour_ws
    ucm_max = ucm.max()
    if ucm_max > 0:
        ucm /= ucm_max

    return ucm, fg


# ── Single-timepoint dispatch ───────────────────────────────────────────────


def compute_contours_single(
    prob_path: str | Path,
    cfg: ContoursConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute contour map + foreground for one timepoint.

    Returns
    -------
    contours : (Z, Y, X) float32 in [0, 1]
    fg : (Z, Y, X) float32
    """
    prob_map = load_prob_map(prob_path)

    if cfg.method == "watershed":
        return contours_watershed(
            prob_map, sigma_contour=cfg.smooth_sigma,
            min_seed_dist=cfg.min_seed_dist, fg_thresh=cfg.fg_thresh,
        )
    elif cfg.method == "combined":
        return contours_combined(
            prob_map, w_prob=cfg.w_prob, w_ws=cfg.w_ws,
            sigma=cfg.smooth_sigma, min_seed_dist=cfg.min_seed_dist,
            fg_thresh=cfg.fg_thresh,
        )
    else:  # "probmap"
        return contours_probmap(prob_map, sigma=cfg.smooth_sigma)


def compute_contours_from_array(
    prob_map: np.ndarray,
    cfg: ContoursConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute contours from a pre-loaded prob map (for interactive preview).

    Returns
    -------
    contours : (Z, Y, X) float32 in [0, 1]
    fg : (Z, Y, X) float32
    """
    if cfg.method == "watershed":
        return contours_watershed(
            prob_map, sigma_contour=cfg.smooth_sigma,
            min_seed_dist=cfg.min_seed_dist, fg_thresh=cfg.fg_thresh,
        )
    elif cfg.method == "combined":
        return contours_combined(
            prob_map, w_prob=cfg.w_prob, w_ws=cfg.w_ws,
            sigma=cfg.smooth_sigma, min_seed_dist=cfg.min_seed_dist,
            fg_thresh=cfg.fg_thresh,
        )
    else:
        return contours_probmap(prob_map, sigma=cfg.smooth_sigma)


# ── File discovery ──────────────────────────────────────────────────────────


def discover_prob_files(input_dir: str | Path) -> list[Path]:
    """Return sorted list of ``t*_prob.tif`` files in *input_dir*."""
    return sorted(Path(input_dir).glob("t*_prob.tif"))


# ── Batch run ───────────────────────────────────────────────────────────────


def run(
    input_dir: str | Path,
    output_dir: str | Path,
    cfg: ContoursConfig,
    overwrite: bool = False,
) -> Generator[tuple[int, int, str], None, None]:
    """Process all timepoints and write a single stacked ``contours.tif``.

    Yields ``(done, total, status_label)`` for progress reporting.
    If ``contours.tif`` already exists and *overwrite* is ``False``, the run
    is skipped immediately.
    """
    out = Path(output_dir)
    out_path = out / "contours.tif"

    if out_path.exists() and not overwrite:
        yield (0, 1, "contours.tif already exists, skipping")
        return

    prob_files = discover_prob_files(input_dir)
    total = len(prob_files)
    if total == 0:
        yield (0, 0, "No t*_prob.tif files found")
        return

    out.mkdir(parents=True, exist_ok=True)

    frames: list[np.ndarray] = []
    for i, prob_path in enumerate(prob_files):
        t_str = prob_path.name.split("_")[0]
        contours, _fg = compute_contours_single(prob_path, cfg)
        frames.append(contours)
        yield (i + 1, total, t_str)

    stacked = np.stack(frames, axis=0)
    tifffile.imwrite(str(out_path), stacked, compression="zlib")
    yield (total, total, "Done")


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        description="s02b — compute contour maps from Cellpose prob maps",
    )
    parser.add_argument("--input-dir", required=True,
                        help="Directory containing t*_prob.tif files")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write contour TIFFs")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to JSON file with ContoursConfig fields (optional)",
    )
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing per-timepoint files")
    args = parser.parse_args()

    cfg_dict: dict = {}
    if args.config:
        cfg_dict = json.loads(Path(args.config).read_text())
    cfg = ContoursConfig(**cfg_dict)

    for done, total, label in run(args.input_dir, args.output_dir, cfg,
                                   overwrite=args.overwrite):
        print(f"[{done}/{total}] {label}", flush=True)

    sys.exit(0)
