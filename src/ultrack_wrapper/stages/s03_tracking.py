"""s03 — Ultrack tracking: segment, link, and solve as independent stages."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import tifffile

from ultrack.config import MainConfig
from ultrack.core.database import clear_all_data
from ultrack.core.linking.processing import link
from ultrack.core.linking.utils import clear_linking_data
from ultrack.core.segmentation.processing import segment
from ultrack.core.solve.processing import solve
from ultrack.core.solve.sqltracking import SQLTracking
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
    """Export tracked segmentation labels as ``tracked_labels.tif``."""
    wd = Path(working_dir)
    ultrack_cfg = _build_ultrack_config(cfg, wd)
    output_path = Path(output_path)

    try:
        from ultrack.core.export.labels import to_labels  # type: ignore[import]

        labels = to_labels(ultrack_cfg)
        if hasattr(labels, "compute"):
            labels = labels.compute()
        tifffile.imwrite(str(output_path), labels.astype(np.uint32), compression="zlib")
        return
    except (ImportError, Exception):
        pass

    tmpdir = Path(tempfile.mkdtemp(prefix="ultrack_labels_"))
    try:
        to_ctc(tmpdir, ultrack_cfg, overwrite=True)
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
    """Load ``tracked_labels.tif`` from *working_dir* for napari visualisation."""
    labels_path = Path(working_dir) / "tracked_labels.tif"
    if not labels_path.exists():
        raise FileNotFoundError(f"tracked_labels.tif not found in {working_dir}")
    return tifffile.imread(str(labels_path))


# ── Independent stage runners ─────────────────────────────────────────────────

def run_segmentation(
    foreground_path: str | Path,
    contours_path: str | Path,
    working_dir: str | Path,
    cfg: TrackingConfig,
    overwrite: bool = True,
) -> Generator[tuple[int, int, str], None, None]:
    """Run only the segmentation (add_nodes) step.

    Yields ``(step, total_steps, status_label)``.
    """
    total = 4
    fg_path = Path(foreground_path)
    ct_path = Path(contours_path)

    if not fg_path.exists():
        raise FileNotFoundError(f"Foreground stack not found: {fg_path}\nRun the Foreground stage first.")
    if not ct_path.exists():
        raise FileNotFoundError(f"Contours stack not found: {ct_path}\nRun the Contours stage first.")

    yield (0, total, "Loading stacks…")
    foreground = load_stack(fg_path)
    contours = load_stack(ct_path)

    wd = Path(working_dir)
    wd.mkdir(parents=True, exist_ok=True)
    ultrack_cfg = _build_ultrack_config(cfg, wd)

    if overwrite:
        yield (1, total, "Clearing existing segmentation from DB…")
        clear_all_data(ultrack_cfg.data_config.database_path)
    else:
        yield (1, total, "Skipping DB clear (overwrite=False)…")

    yield (2, total, "Running segmentation (add nodes)…")
    segment(foreground, contours, ultrack_cfg)

    yield (total, total, "Segmentation done.")


def run_linking(
    working_dir: str | Path,
    cfg: TrackingConfig,
    overwrite: bool = True,
) -> Generator[tuple[int, int, str], None, None]:
    """Run only the linking (add_edges) step.

    Yields ``(step, total_steps, status_label)``.
    """
    total = 3
    wd = Path(working_dir)
    ultrack_cfg = _build_ultrack_config(cfg, wd)

    if overwrite:
        yield (0, total, "Clearing existing links from DB…")
        clear_linking_data(ultrack_cfg.data_config.database_path)
    else:
        yield (0, total, "Skipping DB clear (overwrite=False)…")

    yield (1, total, "Running linking (add edges)…")
    link(ultrack_cfg)

    yield (total, total, "Linking done.")


def run_solve(
    working_dir: str | Path,
    cfg: TrackingConfig,
    overwrite: bool = True,
) -> Generator[tuple[int, int, str], None, None]:
    """Run only the solve (ILP) step and export results.

    Yields ``(step, total_steps, status_label)``.
    """
    total = 5
    wd = Path(working_dir)
    ultrack_cfg = _build_ultrack_config(cfg, wd)

    if overwrite:
        yield (0, total, "Clearing existing solution from DB…")
        SQLTracking.clear_solution_from_database(ultrack_cfg.data_config.database_path)
    else:
        yield (0, total, "Skipping DB clear (overwrite=False)…")

    yield (1, total, "Running ILP solve…")
    solve(ultrack_cfg)

    yield (2, total, "Exporting tracks CSV…")
    tracks_df, _ = to_tracks_layer(ultrack_cfg)
    tracks_df.to_csv(str(wd / "tracks.csv"), index=True)

    yield (3, total, "Exporting tracked labels…")
    export_tracked_labels(wd, cfg, wd / "tracked_labels.tif")

    yield (total, total, "Solve done.")


def run(
    foreground_path: str | Path,
    contours_path: str | Path,
    working_dir: str | Path,
    cfg: TrackingConfig,
) -> Generator[tuple[int, int, str], None, None]:
    """Run the full Ultrack pipeline (segment → link → solve).

    Respects ``cfg.overwrite_segmentation``, ``cfg.overwrite_linking``,
    and ``cfg.overwrite_solve``.

    Yields ``(step, total_steps, status_label)``.
    """
    total = 12

    # Segmentation
    for step, sub_total, label in run_segmentation(
        foreground_path, contours_path, working_dir, cfg,
        overwrite=cfg.overwrite_segmentation,
    ):
        yield (int(step / max(sub_total, 1) * 4), total, f"[Seg] {label}")

    # Linking
    for step, sub_total, label in run_linking(
        working_dir, cfg, overwrite=cfg.overwrite_linking,
    ):
        yield (4 + int(step / max(sub_total, 1) * 3), total, f"[Link] {label}")

    # Solve + export
    for step, sub_total, label in run_solve(
        working_dir, cfg, overwrite=cfg.overwrite_solve,
    ):
        yield (7 + int(step / max(sub_total, 1) * 5), total, f"[Solve] {label}")

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
    """Load tracks layer data for napari visualization."""
    wd = Path(working_dir)
    ultrack_cfg = _build_ultrack_config(cfg, wd)
    tracks_df, graph = to_tracks_layer(ultrack_cfg)
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
    parser.add_argument("--stage", default="all",
                        choices=["all", "segmentation", "linking", "solve"],
                        help="Which stage to run (default: all)")
    parser.add_argument("--foreground", default=None, help="Path to foreground.tif stack")
    parser.add_argument("--contours", default=None, help="Path to contours.tif stack")
    parser.add_argument("--working-dir", required=True, help="Ultrack working/output directory")
    parser.add_argument("--config", default=None, help="Path to JSON file with TrackingConfig fields")
    parser.add_argument("--overwrite-segmentation", action="store_true", default=True)
    parser.add_argument("--overwrite-linking", action="store_true", default=True)
    parser.add_argument("--overwrite-solve", action="store_true", default=True)
    args = parser.parse_args()

    # Validate required args for each stage
    if args.stage in ("all", "segmentation"):
        if not args.foreground or not args.contours:
            parser.error("--foreground and --contours are required for --stage all/segmentation")

    cfg_dict: dict = {}
    if args.config:
        cfg_dict = json.loads(Path(args.config).read_text())
    cfg_dict.setdefault("overwrite_segmentation", args.overwrite_segmentation)
    cfg_dict.setdefault("overwrite_linking", args.overwrite_linking)
    cfg_dict.setdefault("overwrite_solve", args.overwrite_solve)
    cfg = TrackingConfig(**cfg_dict)

    if args.stage == "all":
        gen = run(args.foreground, args.contours, args.working_dir, cfg)
    elif args.stage == "segmentation":
        gen = run_segmentation(args.foreground, args.contours, args.working_dir, cfg, overwrite=cfg.overwrite_segmentation)
    elif args.stage == "linking":
        gen = run_linking(args.working_dir, cfg, overwrite=cfg.overwrite_linking)
    elif args.stage == "solve":
        gen = run_solve(args.working_dir, cfg, overwrite=cfg.overwrite_solve)
    else:
        parser.error(f"Unknown stage: {args.stage}")

    for step, total, label in gen:
        print(f"[{step}/{total}] {label}", flush=True)

    sys.exit(0)
