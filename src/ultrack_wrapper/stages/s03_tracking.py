"""s03 — Ultrack tracking (segment + link + solve)."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import tifffile

from ultrack.config import MainConfig
from ultrack.core.main import track
from ultrack.core.export.tracks_layer import to_tracks_layer
from ultrack.core.export.ctc import to_ctc

from ultrack_wrapper._config import TrackingConfig


def _build_ultrack_config(
    cfg: TrackingConfig,
    working_dir: str | Path,
) -> MainConfig:
    """Translate our TrackingConfig into an Ultrack MainConfig."""
    return MainConfig(
        data={
            "working_dir": str(working_dir),
        },
        segmentation={
            "min_area": cfg.min_area,
            "max_area": cfg.max_area,
            "min_frontier": cfg.min_frontier,
            "threshold": cfg.threshold,
            "ws_hierarchy": cfg.ws_hierarchy,
            "anisotropy_penalization": cfg.anisotropy_penalization,
            "n_workers": cfg.n_workers,
        },
        linking={
            "max_distance": cfg.max_distance,
            "max_neighbors": cfg.max_neighbors,
            "distance_weight": cfg.distance_weight,
            "n_workers": cfg.n_workers,
        },
        tracking={
            "appear_weight": cfg.appear_weight,
            "disappear_weight": cfg.disappear_weight,
            "division_weight": cfg.division_weight,
            "link_function": cfg.link_function,
            "power": cfg.power,
            "bias": cfg.bias,
            "solution_gap": cfg.solution_gap,
            "time_limit": cfg.time_limit,
            "window_size": cfg.window_size if cfg.window_size > 0 else None,
        },
    )


def load_stack(path: str | Path) -> np.ndarray:
    """Load a TIFF stack (T, Z, Y, X) or (T, Y, X)."""
    return tifffile.imread(str(path)).astype(np.float32)


def export_tracked_labels(
    working_dir: str | Path,
    cfg: TrackingConfig,
    output_path: str | Path,
) -> None:
    """Export tracked segmentation labels as ``tracked_labels.tif``.

    Each voxel in the output volume is labelled with its track ID (uint32).
    The output shape matches the segmented volume: (T, [Z,] Y, X).

    First tries the native ``ultrack.core.export.labels.to_labels`` API; falls
    back to reconstructing from a CTC export if that function is unavailable.
    """
    wd = Path(working_dir)
    ultrack_cfg = _build_ultrack_config(cfg, wd)
    output_path = Path(output_path)

    # Attempt native labels export
    try:
        from ultrack.core.export.labels import to_labels  # type: ignore[import]

        labels = to_labels(ultrack_cfg)
        if hasattr(labels, "compute"):  # dask array
            labels = labels.compute()
        tifffile.imwrite(str(output_path), labels.astype(np.uint32), compression="zlib")
        return
    except (ImportError, Exception):
        pass

    # Fallback: reconstruct from CTC export
    tmpdir = Path(tempfile.mkdtemp(prefix="ultrack_labels_"))
    try:
        to_ctc(tmpdir, ultrack_cfg, overwrite=True)
        # Search recursively — to_ctc may write to TRA/, RES/, or flat
        mask_files = sorted(tmpdir.rglob("mask*.tif"))
        if not mask_files:
            mask_files = sorted(tmpdir.rglob("man_track*.tif"))
        if not mask_files:
            mask_files = sorted(tmpdir.rglob("*.tif"))
        if mask_files:
            frames = [tifffile.imread(str(f)) for f in mask_files]
            stacked = np.stack(frames, axis=0)
            tifffile.imwrite(str(output_path), stacked.astype(np.uint32), compression="zlib")
        else:
            raise RuntimeError("CTC export produced no mask files.")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def get_labels_layer(working_dir: str | Path) -> np.ndarray:
    """Load ``tracked_labels.tif`` from *working_dir* for napari visualisation.

    Returns
    -------
    np.ndarray
        Integer label array shaped (T, [Z,] Y, X).

    Raises
    ------
    FileNotFoundError
        If ``tracked_labels.tif`` does not exist in *working_dir*.
    """
    labels_path = Path(working_dir) / "tracked_labels.tif"
    if not labels_path.exists():
        raise FileNotFoundError(f"tracked_labels.tif not found in {working_dir}")
    return tifffile.imread(str(labels_path))


def run(
    foreground_path: str | Path,
    contours_path: str | Path,
    working_dir: str | Path,
    cfg: TrackingConfig,
) -> Generator[tuple[int, int, str], None, None]:
    """Run Ultrack tracking pipeline.

    Yields ``(step, total_steps, status_label)`` for progress reporting.
    """
    total = 6

    fg_path = Path(foreground_path)
    if not fg_path.exists():
        raise FileNotFoundError(
            f"Foreground stack not found: {fg_path}\n"
            "Run the Foreground stage first."
        )
    ct_path = Path(contours_path)
    if not ct_path.exists():
        raise FileNotFoundError(
            f"Contours stack not found: {ct_path}\n"
            "Run the Contours stage first."
        )

    yield (0, total, "Loading foreground stack...")
    foreground = load_stack(fg_path)
    yield (1, total, "Loading contours stack...")
    contours = load_stack(ct_path)
    yield (2, total, "Building Ultrack config...")

    wd = Path(working_dir)
    wd.mkdir(parents=True, exist_ok=True)

    ultrack_cfg = _build_ultrack_config(cfg, wd)
    overwrite = cfg.overwrite if cfg.overwrite != "none" else "none"

    yield (3, total, "Running tracking (segment + link + solve)...")

    track(
        ultrack_cfg,
        foreground=foreground,
        contours=contours,
        overwrite=overwrite,
    )

    yield (4, total, "Exporting tracks CSV...")

    tracks_df, graph = to_tracks_layer(ultrack_cfg)
    tracks_df.to_csv(str(wd / "tracks.csv"), index=True)

    yield (5, total, "Exporting tracked labels...")

    export_tracked_labels(wd, cfg, wd / "tracked_labels.tif")

    yield (total, total, "Done")


def export_ctc(
    working_dir: str | Path,
    output_dir: str | Path,
    cfg: TrackingConfig,
) -> None:
    """Export tracking results to CTC format."""
    wd = Path(working_dir)
    ultrack_cfg = _build_ultrack_config(cfg, wd)
    to_ctc(Path(output_dir), ultrack_cfg, overwrite=True)


def get_tracks_layer(
    working_dir: str | Path,
    cfg: TrackingConfig,
):
    """Load tracks layer data for napari visualization.

    Returns
    -------
    tracks_df : pd.DataFrame
        Columns: track_id, t, (z), y, x
    graph : dict
        Lineage graph mapping track_id -> [parent_track_id]
    """
    wd = Path(working_dir)
    ultrack_cfg = _build_ultrack_config(cfg, wd)
    tracks_df, graph = to_tracks_layer(ultrack_cfg)

    # napari add_tracks requires exactly 4 or 5 columns: track_id, t, [z], y, x
    spatial_cols = [c for c in tracks_df.columns if c in {"track_id", "t", "z", "y", "x"}]
    return tracks_df[spatial_cols], graph


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        description="s03 — run Ultrack tracking pipeline from the command line",
    )
    parser.add_argument("--foreground", required=True, help="Path to foreground.tif stack")
    parser.add_argument("--contours", required=True, help="Path to contours.tif stack")
    parser.add_argument("--working-dir", required=True, help="Ultrack working/output directory")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to JSON file with TrackingConfig fields (optional)",
    )
    # Flat overrides for the most common params
    parser.add_argument("--overwrite", default="all",
                        choices=["all", "links", "solutions", "none"])
    args = parser.parse_args()

    cfg_dict: dict = {}
    if args.config:
        cfg_dict = json.loads(Path(args.config).read_text())
    cfg_dict.setdefault("overwrite", args.overwrite)
    cfg = TrackingConfig(**cfg_dict)

    for step, total, label in run(args.foreground, args.contours, args.working_dir, cfg):
        print(f"[{step}/{total}] {label}", flush=True)

    sys.exit(0)
