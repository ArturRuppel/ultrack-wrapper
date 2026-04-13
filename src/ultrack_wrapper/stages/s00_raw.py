"""
s00 — Raw data export from NDTiff to per-timepoint TIFFs.

Outputs (per position)
----------------------
  0_raw/nucleus_3d_t<TTT>.tif   (Z, H, W)       uint16  — one per timepoint
  0_raw/cell_zavg.tif           (T, C=2, H, W)  uint16  — all timepoints
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import tifffile
from ndtiff import Dataset
from skimage.transform import downscale_local_mean

from ultrack_wrapper._config import DatasetConfig
from ultrack_wrapper._paths import cell_zavg_path, nucleus_3d_path, raw_dir

# Channel indices (0-based) in the NDTiff dataset
# Dataset ChNames: ['CSUTRANS', 'CSU405 ', 'CSU488', 'CSU561']
_CH_405 = 1  # CSU405  — nuclear marker (NLS-mCherry)
_CH_488 = 2  # CSU488  — membrane marker


# ── Helpers ──────────────────────────────────────────────────────────────────


def _read_z_stack(
    ds: Dataset, position: int, time: int, channel: int, z_indices: list[int]
) -> np.ndarray:
    """Return a (Z, H, W) uint16 array for one (pos, time, channel)."""
    slices = []
    for z in z_indices:
        img = ds.read_image(position=position, time=time, channel=channel, z=z)
        if img is None:
            img = np.zeros((ds.image_height, ds.image_width), dtype=np.uint16)
        slices.append(img)
    return np.stack(slices, axis=0)


def _xy_avg(arr: np.ndarray, factor: int) -> np.ndarray:
    """Block-average XY by *factor*. Accepts (Z, H, W) or (H, W); returns uint16."""
    if factor <= 1:
        return arr.astype(np.uint16)
    if arr.ndim == 3:
        downsampled = downscale_local_mean(arr, (1, factor, factor))
    else:
        downsampled = downscale_local_mean(arr, (factor, factor))
    return downsampled.astype(np.uint16)


# ── Export: nucleus ──────────────────────────────────────────────────────────


def _export_nucleus(
    ds: Dataset,
    config: DatasetConfig,
    pos: int,
    time_list: list[int],
    z_indices: list[int],
    overwrite: bool,
) -> Generator[tuple[int, int, str], None, None]:
    """Export 405-channel full Z-stack, one TIFF per timepoint."""
    out_dir = raw_dir(config.root_dir, pos)
    out_dir.mkdir(parents=True, exist_ok=True)
    xy_factor = config.xy_downsample
    total = len(time_list)

    for i, t in enumerate(time_list):
        out_path = nucleus_3d_path(config.root_dir, pos, t)
        if out_path.exists() and not overwrite:
            yield (i + 1, total, "nucleus")
            continue

        volume = _read_z_stack(ds, pos, t, _CH_405, z_indices)
        volume = _xy_avg(volume, xy_factor)

        tifffile.imwrite(
            str(out_path),
            volume,
            compression="zlib",
            metadata={"axes": "ZYX"},
        )
        yield (i + 1, total, "nucleus")


# ── Export: cell ─────────────────────────────────────────────────────────────


def _export_cell(
    ds: Dataset,
    config: DatasetConfig,
    pos: int,
    time_list: list[int],
    z_indices: list[int],
    overwrite: bool,
) -> Generator[tuple[int, int, str], None, None]:
    """Export two-channel (405, 488) Z-mean → single (T, C=2, H, W) stack."""
    out_path = cell_zavg_path(config.root_dir, pos)
    if out_path.exists() and not overwrite:
        return

    out_dir = raw_dir(config.root_dir, pos)
    out_dir.mkdir(parents=True, exist_ok=True)

    xy_factor = config.xy_downsample
    h_out = math.ceil(ds.image_height / xy_factor)
    w_out = math.ceil(ds.image_width / xy_factor)
    channels = [_CH_405, _CH_488]
    n_t = len(time_list)

    stack = np.zeros((n_t, 2, h_out, w_out), dtype=np.uint16)

    for ti, t in enumerate(time_list):
        for ci, ch in enumerate(channels):
            volume = _read_z_stack(ds, pos, t, ch, z_indices)
            projected = volume.mean(axis=0).astype(np.uint16)
            projected = _xy_avg(projected, xy_factor)
            stack[ti, ci] = projected
        yield (ti + 1, n_t, "cell")

    tifffile.imwrite(
        str(out_path),
        stack,
        compression="zlib",
        metadata={"axes": "TCYX"},
    )


# ── Public API ───────────────────────────────────────────────────────────────


def run(
    config: DatasetConfig,
    pos: int,
    overwrite: bool = False,
) -> Generator[tuple[int, int, str], None, None]:
    """
    Export raw NDTiff data for one position.

    Yields (done, total, label) tuples for progress reporting.
    """
    ds = Dataset(config.ndtiff_path)

    axes = ds.axes
    all_times = sorted(axes.get("time", [0]))
    z_indices = sorted(axes.get("z", [0]))

    time_list = config.timepoints if config.timepoints is not None else all_times

    yield from _export_nucleus(ds, config, pos, time_list, z_indices, overwrite)
    yield from _export_cell(ds, config, pos, time_list, z_indices, overwrite)

    # Write run_params.json
    out_dir = raw_dir(config.root_dir, pos)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_params = {
        "stage": "raw",
        "pos": pos,
        "xy_downsample": config.xy_downsample,
        "timepoints": time_list,
        "z_indices": z_indices,
        "ndtiff_path": config.ndtiff_path,
    }
    (out_dir / "run_params.json").write_text(
        json.dumps(run_params, indent=2), encoding="utf-8"
    )
