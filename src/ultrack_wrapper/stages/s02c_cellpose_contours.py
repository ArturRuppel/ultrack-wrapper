"""s02c — Cellpose-native contours from flow fields and probability maps.

Uses cellpose.dynamics.compute_masks to generate label maps from flow (dP) and
probability maps, then ultrack.utils.labels_to_contours to derive foreground and
contour maps.

This is an alternative to s02/s02b that leverages cellpose's own segmentation
logic rather than custom thresholding and watershed.

Usage
-----
    python -m ultrack_wrapper.stages.s02c_cellpose_contours \\
        --input-dir /path/to/cellpose_output \\
        --output-dir /path/to/output \\
        --config /tmp/cfg.json \\
        [--overwrite]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Generator

import numpy as np
import tifffile
from scipy import ndimage as ndi

from ultrack_wrapper._config import CellposeContoursConfig


# ── File discovery ──────────────────────────────────────────────────────────


def discover_dp_files(input_dir: str | Path) -> list[Path]:
    """Return sorted list of ``t*_dp.tif`` files in *input_dir*."""
    return sorted(Path(input_dir).glob("t*_dp.tif"))


def discover_prob_files(input_dir: str | Path) -> list[Path]:
    """Return sorted list of ``t*_prob.tif`` files in *input_dir*."""
    return sorted(Path(input_dir).glob("t*_prob.tif"))


# ── Core computation ────────────────────────────────────────────────────────


def compute_labels_single(
    dp_path: str | Path,
    prob_path: str | Path,
    cfg: CellposeContoursConfig,
) -> np.ndarray:
    """Compute label map using cellpose.dynamics.compute_masks.

    Parameters
    ----------
    dp_path : path to t*_dp.tif (flow field)
    prob_path : path to t*_prob.tif (cell probability map)
    cfg : CellposeContoursConfig

    Returns
    -------
    labels : np.ndarray, uint32
        Labeled segmentation; 0 = background, 1+ = cell IDs.
    """
    from cellpose.dynamics import compute_masks
    import torch

    dp = tifffile.imread(str(dp_path)).astype(np.float32)
    prob = tifffile.imread(str(prob_path)).astype(np.float32)

    try:
        device = torch.device(cfg.device)
        masks = compute_masks(
            dp,
            prob,
            cellprob_threshold=cfg.cellprob_threshold,
            do_3D=cfg.do_3D,
            device=device,
        )
    except Exception as e:
        print(f"  [warn] compute_masks failed: {e}", flush=True)
        masks = np.zeros(prob.shape, dtype=np.uint16)

    return masks.astype(np.uint32)


def compute_contours_from_labels(
    labels: np.ndarray,
    smooth_sigma: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert label map to foreground and contour maps via ultrack's labels_to_contours.

    Parameters
    ----------
    labels : (Z, Y, X) or (Y, X) uint32
        Label array from cellpose.dynamics.compute_masks
    smooth_sigma : float
        Gaussian sigma for smoothing the contour map (0 = no smoothing)

    Returns
    -------
    foreground : np.ndarray, float32 in [0, 1]
        Foreground probability (1 = cell interior, 0 = background)
    contours : np.ndarray, float32 in [0, 1]
        Contour map (1 = strong boundary, 0 = cell interior)
    """
    from ultrack.utils import labels_to_contours

    # labels_to_contours expects a list of label arrays
    fg, ucm = labels_to_contours([labels])
    fg = np.asarray(fg, dtype=np.float32)
    ucm = np.asarray(ucm, dtype=np.float32)

    # Optional smoothing of the contour map
    if smooth_sigma > 0:
        ucm = ndi.gaussian_filter(ucm, sigma=smooth_sigma)
        ucm_max = ucm.max()
        if ucm_max > 0:
            ucm = ucm / ucm_max

    return fg, ucm


def compute_single(
    dp_path: str | Path,
    prob_path: str | Path,
    cfg: CellposeContoursConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute labels, foreground, and contours for a single timepoint.

    Returns
    -------
    labels : np.ndarray, uint32
    foreground : np.ndarray, float32
    contours : np.ndarray, float32
    """
    labels = compute_labels_single(dp_path, prob_path, cfg)
    foreground, contours = compute_contours_from_labels(labels, smooth_sigma=cfg.smooth_sigma)
    return labels, foreground, contours


def compute_single_from_arrays(
    dp: np.ndarray,
    prob: np.ndarray,
    cfg: CellposeContoursConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute labels, foreground, and contours from pre-loaded arrays.

    Parameters
    ----------
    dp : np.ndarray, float32
        Flow field (dP)
    prob : np.ndarray, float32
        Probability map
    cfg : CellposeContoursConfig

    Returns
    -------
    labels : np.ndarray, uint32
    foreground : np.ndarray, float32
    contours : np.ndarray, float32
    """
    from cellpose.dynamics import compute_masks
    import torch

    try:
        device = torch.device(cfg.device)
        masks = compute_masks(
            dp,
            prob,
            cellprob_threshold=cfg.cellprob_threshold,
            do_3D=cfg.do_3D,
            device=device,
        )
    except Exception as e:
        print(f"[warn] compute_masks failed: {e}", flush=True)
        masks = np.zeros(prob.shape, dtype=np.uint16)

    labels = masks.astype(np.uint32)
    foreground, contours = compute_contours_from_labels(labels, smooth_sigma=cfg.smooth_sigma)
    return labels, foreground, contours


# ── Batch run ───────────────────────────────────────────────────────────────


def run(
    input_dir: str | Path,
    output_dir: str | Path,
    cfg: CellposeContoursConfig,
    overwrite: bool = False,
) -> Generator[tuple[int, int, str], None, None]:
    """Process all timepoints and write foreground.tif and contours.tif.

    Yields ``(done, total, status_label)`` for progress reporting.
    If both output files already exist and *overwrite* is False, skips immediately.
    """
    inp = Path(input_dir)
    out = Path(output_dir)
    fg_path = out / "foreground.tif"
    ct_path = out / "contours.tif"

    # Early exit if outputs exist
    if not overwrite and fg_path.exists() and ct_path.exists():
        yield (0, 1, "foreground.tif and contours.tif already exist, skipping")
        return

    # Discover files
    dp_files = discover_dp_files(inp)
    prob_files = discover_prob_files(inp)

    if not dp_files or not prob_files:
        yield (0, 0, "No t*_dp.tif or t*_prob.tif files found")
        return

    if len(dp_files) != len(prob_files):
        yield (0, 0, f"Mismatch: {len(dp_files)} dp files vs {len(prob_files)} prob files")
        return

    total = len(dp_files)
    out.mkdir(parents=True, exist_ok=True)

    fg_frames: list[np.ndarray] = []
    ct_frames: list[np.ndarray] = []

    for i, (dp_path, prob_path) in enumerate(zip(dp_files, prob_files)):
        t_str = dp_path.name.split("_")[0]  # e.g., "t000"
        try:
            _, fg, ct = compute_single(dp_path, prob_path, cfg)
            fg_frames.append(fg)
            ct_frames.append(ct)
        except Exception as e:
            print(f"  {t_str}: error: {e}", flush=True)
            spatial = tifffile.imread(str(prob_path)).shape
            fg_frames.append(np.zeros(spatial, dtype=np.float32))
            ct_frames.append(np.zeros(spatial, dtype=np.float32))

        yield (i + 1, total, t_str)

    fg_stack = np.stack(fg_frames, axis=0)
    ct_stack = np.stack(ct_frames, axis=0)

    tifffile.imwrite(str(fg_path), fg_stack, compression="zlib")
    tifffile.imwrite(str(ct_path), ct_stack, compression="zlib")

    yield (total, total, "Done")


# ── CLI entry point ─────────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="s02c — compute contours from Cellpose flow fields and probability maps",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing t*_dp.tif and t*_prob.tif files",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write foreground.tif and contours.tif",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to JSON file with CellposeContoursConfig fields (optional)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    args = parser.parse_args()

    cfg_dict: dict = {}
    if args.config:
        cfg_dict = json.loads(Path(args.config).read_text())
    cfg = CellposeContoursConfig(**cfg_dict)

    for done, total, label in run(args.input_dir, args.output_dir, cfg, overwrite=args.overwrite):
        print(f"[{done}/{total}] {label}", flush=True)

    sys.exit(0)
