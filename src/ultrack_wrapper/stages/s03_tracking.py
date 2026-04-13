"""s03 — Ultrack tracking (segment + link + solve)."""

from __future__ import annotations

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


def run(
    foreground_path: str | Path,
    contours_path: str | Path,
    working_dir: str | Path,
    cfg: TrackingConfig,
) -> Generator[tuple[int, int, str], None, None]:
    """Run Ultrack tracking pipeline.

    Yields ``(step, total_steps, status_label)`` for progress reporting.
    """
    total = 5
    yield (0, total, "Loading foreground stack...")

    foreground = load_stack(foreground_path)
    yield (1, total, "Loading contours stack...")

    contours = load_stack(contours_path)
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

    yield (4, total, "Exporting tracks...")

    # Export tracks layer data
    tracks_df, graph = to_tracks_layer(ultrack_cfg)
    tracks_df.to_csv(str(wd / "tracks.csv"), index=True)

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
    return to_tracks_layer(ultrack_cfg)
