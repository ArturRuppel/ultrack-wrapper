#!/home/aruppel/miniconda3/envs/cellflow/bin/python
"""Parameter sweep for ultrack solve step.

Runs multiple solve operations in parallel with different parameter combinations
and evaluates the results. The solve step is single-threaded, so multiple processes
can run in parallel efficiently.

Usage:
    # Run a sweep
    ./sweep_solve_params.py \
        --working-dir /path/to/database \
        --param power 2.0 3.0 4.0 5.0 \
        --param division_weight -0.001 -0.01 -0.1 \
        --output-dir ./sweep_results

    # Save sweep config for reuse
    ./sweep_solve_params.py \
        --working-dir /path/to/database \
        --param power 2.0 3.0 4.0 \
        --save-sweep my_sweep.json

    # Load and run saved sweep
    ./sweep_solve_params.py \
        --working-dir /path/to/database \
        --load-sweep my_sweep.json \
        --output-dir ./sweep_results
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import binary_fill_holes, gaussian_filter
from skimage.morphology import binary_closing, disk as sk_disk

from ultrack_wrapper._config import TrackingConfig


# ── 2D Projection ─────────────────────────────────────────────────────────────

def _process_frame(
    frame: np.ndarray,
    fill_holes: bool = True,
    closing_radius: int = 2,
    smooth_sigma: float = 0.8,
    smooth_thresh: float = 0.5,
) -> np.ndarray:
    """Project (Z, Y, X) uint16 volume → (Y, X) uint16 using nearest-to-midplane.

    Projection rule: for XY columns where multiple labels overlap in Z, the label
    whose Z centroid is nearest to the stack midplane wins.
    """
    Z, Y, X = frame.shape
    mid_z = (Z - 1) / 2.0

    unique_labels = np.unique(frame)
    unique_labels = unique_labels[unique_labels > 0]

    if len(unique_labels) == 0:
        return np.zeros((Y, X), dtype=np.uint16)

    # Step 1: Z centroid per label
    z_idx = np.arange(Z, dtype=np.float32)
    centroid_z: dict[int, float] = {}
    for lbl in unique_labels:
        per_z = (frame == lbl).sum(axis=(1, 2)).astype(np.float32)  # (Z,)
        total = per_z.sum()
        centroid_z[lbl] = float((per_z * z_idx).sum() / total) if total > 0 else mid_z

    # Step 2: sort farthest → nearest (so nearest paints last → wins)
    paint_order = sorted(unique_labels, key=lambda l: -abs(centroid_z[l] - mid_z))

    # Step 3: initial projection
    proj = np.zeros((Y, X), dtype=np.uint16)
    for lbl in paint_order:
        footprint = (frame == lbl).any(axis=0)
        proj[footprint] = lbl

    # Step 4: postprocess each label mask
    closing_selem = sk_disk(closing_radius) if closing_radius > 0 else None
    processed: dict[int, np.ndarray] = {}

    for lbl in unique_labels:
        m = proj == lbl

        if fill_holes:
            m = binary_fill_holes(m)

        if closing_selem is not None:
            m = binary_closing(m, closing_selem)

        if smooth_sigma > 0:
            mf = gaussian_filter(m.astype(np.float32), sigma=smooth_sigma)
            m = mf >= smooth_thresh

        processed[lbl] = m

    # Step 5: re-composite with same farthest-first order so nearest wins
    result = np.zeros((Y, X), dtype=np.uint16)
    for lbl in paint_order:
        result[processed[lbl]] = lbl

    return result


def project_tracked_labels_2d(
    tracked_labels_3d_path: Path,
    output_path: Path,
    fill_holes: bool = True,
    closing_radius: int = 2,
    smooth_sigma: float = 0.8,
    smooth_thresh: float = 0.5,
) -> None:
    """Project 3D tracked labels to 2D and save."""
    labels_3d = tifffile.imread(str(tracked_labels_3d_path)).astype(np.uint16)

    if labels_3d.ndim != 4:
        raise ValueError(f"Expected (T, Z, H, W), got {labels_3d.shape}")

    T, Z, Y, X = labels_3d.shape
    proj_stack = []

    for t in range(T):
        proj = _process_frame(
            labels_3d[t],
            fill_holes=fill_holes,
            closing_radius=closing_radius,
            smooth_sigma=smooth_sigma,
            smooth_thresh=smooth_thresh,
        )
        proj_stack.append(proj)

    result = np.stack(proj_stack).astype(np.uint16)
    tifffile.imwrite(
        str(output_path),
        result,
        compression="zlib",
        metadata={"axes": "TYX"},
    )


# ── Sweep result tracking ─────────────────────────────────────────────────────

@dataclass
class SweepResult:
    """Result from a single solve run."""
    param_set: dict[str, float | str]
    run_index: int
    run_dir: Path
    success: bool
    error_message: str | None = None
    n_tracks: int = 0
    n_track_points: int = 0
    n_divisions: int = 0
    execution_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


def run_single_solve(
    run_dir: Path,
    working_dir: Path,
    base_config: TrackingConfig,
    param_set: dict[str, Any],
    run_index: int,
) -> SweepResult:
    """Run a single solve with given parameters.

    Uses a subdirectory for each run with a copy of the database.
    Skips if results already exist.
    """
    import time
    start_time = time.time()

    result = SweepResult(
        param_set=param_set,
        run_index=run_index,
        run_dir=run_dir,
        success=False,
    )

    # Check if results already exist
    tracked_labels = run_dir / "tracked_labels.tif"
    tracks_csv = run_dir / "tracks.csv"
    if tracked_labels.exists() and tracks_csv.exists():
        # Results already exist, load metrics and skip
        try:
            tracks_df = pd.read_csv(tracks_csv)
            result.n_tracks = len(tracks_df["track_id"].unique())
            result.n_track_points = len(tracks_df)
            result.success = True
            result.execution_time = 0.0  # Mark as skipped with 0 time
            return result
        except Exception:
            pass  # If we can't read, continue with solve

    try:
        # Create run directory
        run_dir.mkdir(parents=True, exist_ok=True)

        # Copy database files and metadata
        for db_file in working_dir.glob("*.db"):
            shutil.copy2(db_file, run_dir / db_file.name)
        for metadata_file in working_dir.glob("metadata.*"):
            shutil.copy2(metadata_file, run_dir / metadata_file.name)

        # Build config with parameters for this run
        cfg_dict = base_config.model_dump()
        cfg_dict.update(param_set)
        cfg = TrackingConfig(**cfg_dict)

        # Save config to run directory
        config_file = run_dir / "config.json"
        config_file.write_text(json.dumps(cfg.model_dump(), indent=2))

        # Run solve via subprocess
        cellflow_python = "/home/aruppel/miniconda3/envs/cellflow/bin/python"
        cmd = [
            cellflow_python, "-m", "ultrack_wrapper.stages.s03_tracking",
            "--stage", "solve",
            "--working-dir", str(run_dir),
            "--config", str(config_file),
            "--overwrite-solve",
        ]

        result_proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )

        if result_proc.returncode == 0:
            result.success = True

            # Extract metrics from results
            try:
                tracks_csv = run_dir / "tracks.csv"
                if tracks_csv.exists():
                    tracks_df = pd.read_csv(tracks_csv)
                    result.n_tracks = len(tracks_df["track_id"].unique())
                    result.n_track_points = len(tracks_df)
            except Exception:
                pass

            # Generate 2D projection
            try:
                tracked_labels_3d = run_dir / "tracked_labels.tif"
                if tracked_labels_3d.exists():
                    proj_output = run_dir / "tracked_labels_proj2d.tif"
                    project_tracked_labels_2d(tracked_labels_3d, proj_output)
            except Exception as e:
                pass  # 2D projection is optional

        else:
            result.error_message = result_proc.stderr[:500]

        result.execution_time = time.time() - start_time

    except subprocess.TimeoutExpired:
        result.error_message = "Solve timeout (>1 hour)"
        result.execution_time = time.time() - start_time
    except Exception as e:
        result.error_message = str(e)
        result.execution_time = time.time() - start_time

    return result


# ── Parameter grid ───────────────────────────────────────────────────────────

def create_parameter_grid(
    param_ranges: dict[str, list[float | str]],
) -> list[dict[str, float | str]]:
    """Create all combinations of parameters from ranges."""
    import itertools

    keys = list(param_ranges.keys())
    values = list(param_ranges.values())

    param_sets = []
    for combination in itertools.product(*values):
        param_set = dict(zip(keys, combination))
        param_sets.append(param_set)

    return param_sets


def param_set_to_dir_name(param_set: dict[str, Any]) -> str:
    """Convert parameter set to directory name."""
    parts = []
    for k, v in sorted(param_set.items()):
        if isinstance(v, float):
            parts.append(f"{k}_{v:.6f}".rstrip("0").rstrip("."))
        else:
            parts.append(f"{k}_{v}")
    return "_".join(parts)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Parameter sweep for ultrack solve step",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a sweep
  ./sweep_solve_params.py \\
    --working-dir /path/to/database \\
    --param power 2.0 3.0 4.0 5.0 \\
    --param division_weight -0.001 -0.01 -0.1

  # Save sweep configuration for reuse
  ./sweep_solve_params.py \\
    --working-dir /path/to/database \\
    --param power 2.0 3.0 4.0 \\
    --save-sweep my_sweep.json

  # Load and run saved sweep
  ./sweep_solve_params.py \\
    --working-dir /path/to/database \\
    --load-sweep my_sweep.json \\
    --output-dir ./sweep_results
        """,
    )

    parser.add_argument(
        "--working-dir",
        help="Path to ultrack database directory",
    )
    parser.add_argument(
        "--output-dir",
        default="./sweep_results",
        help="Directory to save sweep results (default: ./sweep_results)",
    )
    parser.add_argument(
        "--config",
        help="Path to JSON file with base TrackingConfig",
    )
    parser.add_argument(
        "--param",
        nargs="+",
        action="append",
        metavar=("NAME", "VALUE"),
        help="Parameter sweep (e.g. --param power 2.0 3.0 4.0). Can be used multiple times.",
    )
    parser.add_argument(
        "--save-sweep",
        metavar="FILE",
        help="Save parameter grid to JSON file (don't run)",
    )
    parser.add_argument(
        "--load-sweep",
        metavar="FILE",
        help="Load parameter grid from JSON file (instead of --param)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Number of parallel jobs (default: CPU count)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        metavar="N",
        help="Process N remaining parameter sets and exit (finds incomplete runs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print parameter combinations without running",
    )

    args = parser.parse_args()

    # Load or build parameter ranges
    param_ranges: dict[str, list[float | str]] = {}

    if args.load_sweep:
        # Load from file
        sweep_file = Path(args.load_sweep)
        if not sweep_file.exists():
            parser.error(f"Sweep file not found: {sweep_file}")
        sweep_data = json.loads(sweep_file.read_text())
        param_ranges = sweep_data.get("param_ranges", {})
        if not param_ranges:
            parser.error(f"No param_ranges in {sweep_file}")
    elif args.param:
        # Parse from CLI
        for param_group in args.param:
            param_name = param_group[0]
            param_values = []
            for val in param_group[1:]:
                try:
                    param_values.append(float(val))
                except ValueError:
                    param_values.append(val)
            param_ranges[param_name] = param_values
    else:
        parser.error("Must provide --param, --load-sweep, or --save-sweep")

    # Handle --save-sweep
    if args.save_sweep:
        sweep_file = Path(args.save_sweep)
        sweep_data = {
            "param_ranges": param_ranges,
            "n_combinations": len(create_parameter_grid(param_ranges)),
        }
        sweep_file.write_text(json.dumps(sweep_data, indent=2))
        print(f"Saved sweep config to: {sweep_file}")
        print(f"  Parameter ranges: {param_ranges}")
        print(f"  Total combinations: {sweep_data['n_combinations']}")
        return

    # Validate required args for running
    if not args.working_dir:
        parser.error("--working-dir is required for running (not needed with --save-sweep)")

    # Load base config
    base_cfg_dict: dict = {}
    if args.config:
        base_cfg_dict = json.loads(Path(args.config).read_text())
    base_config = TrackingConfig(**base_cfg_dict)

    # Create parameter grid
    param_sets = create_parameter_grid(param_ranges)
    n_combinations = len(param_sets)

    # Handle --batch mode: find and process only remaining incomplete runs
    if args.batch:
        output_dir = Path(args.output_dir)
        results_csv = output_dir / "results.csv"

        if results_csv.exists():
            # Load existing results and find completed parameter sets
            existing_results = pd.read_csv(results_csv)
            completed_params = set()
            for _, row in existing_results.iterrows():
                # Reconstruct param dict from row
                param_dict = {}
                for param_name in param_ranges.keys():
                    if param_name in row.index:
                        param_dict[param_name] = row[param_name]
                completed_params.add(json.dumps(param_dict, sort_keys=True))

            # Filter to only incomplete param sets
            incomplete_param_sets = []
            for param_set in param_sets:
                param_json = json.dumps(param_set, sort_keys=True)
                if param_json not in completed_params:
                    incomplete_param_sets.append(param_set)

            param_sets = incomplete_param_sets
            n_combinations = len(param_sets)

            print(f"Batch mode: Found {len(incomplete_param_sets)} incomplete parameter sets (out of {len(existing_results)})")
            if n_combinations == 0:
                print("All parameter sets already processed!")
                return
        else:
            print(f"Batch mode: No existing results found. Processing first {args.batch} of {n_combinations}...")

        # Limit to batch size
        param_sets = param_sets[:args.batch]
        n_combinations = len(param_sets)

    print(f"Parameter sweep configuration:")
    print(f"  Working directory: {args.working_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Parameter ranges:")
    for name, values in param_ranges.items():
        print(f"    {name}: {values}")
    print(f"  Parameter sets to process: {n_combinations}")
    print()

    if args.dry_run:
        print("Dry run - parameter combinations:")
        for i, param_set in enumerate(param_sets, 1):
            dir_name = param_set_to_dir_name(param_set)
            print(f"  {i}. {dir_name}")
        return

    # Create output directory and logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_file = output_dir / "sweep.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    # Save sweep configuration
    config_file = output_dir / "sweep_config.json"
    config_file.write_text(json.dumps({
        "working_dir": str(args.working_dir),
        "param_ranges": param_ranges,
        "n_combinations": n_combinations,
        "base_config": json.loads(base_config.model_dump_json()),
    }, indent=2))

    logger.info(f"Starting parameter sweep with {n_combinations} combinations")
    logger.info(f"Results directory: {output_dir}")

    working_dir = Path(args.working_dir)
    n_jobs = args.n_jobs or cpu_count()

    # Run parallel solves
    results = []
    with Pool(processes=n_jobs) as pool:
        tasks = [
            (
                output_dir / param_set_to_dir_name(param_set),
                working_dir,
                base_config,
                param_set,
                run_index,
            )
            for run_index, param_set in enumerate(param_sets)
        ]

        for i, result in enumerate(pool.starmap(run_single_solve, tasks), 1):
            results.append(result)
            status = "✓" if result.success else "✗"
            time_str = f"({result.execution_time:.1f}s)" if result.execution_time > 0 else "(skipped)"
            logger.info(
                f"[{i}/{n_combinations}] {status} {result.run_dir.name} "
                f"{time_str} "
                f"tracks={result.n_tracks}, points={result.n_track_points}"
            )

    logger.info("=" * 80)

    # Save results to CSV
    results_data = []
    for result in results:
        row = {
            "run_index": result.run_index,
            "run_dir": result.run_dir.name,
            "success": result.success,
            "execution_time_s": result.execution_time,
            "n_tracks": result.n_tracks,
            "n_track_points": result.n_track_points,
            "n_divisions": result.n_divisions,
            "error": result.error_message or "",
        }
        # Add parameter columns
        row.update(result.param_set)
        results_data.append(row)

    results_df = pd.DataFrame(results_data)
    results_csv = output_dir / "results.csv"
    results_df.to_csv(results_csv, index=False)
    logger.info(f"Results saved to: {results_csv}")

    # Print summary statistics
    successful = results_df[results_df["success"]]
    if len(successful) > 0:
        logger.info("")
        logger.info("Summary (successful runs):")
        logger.info(f"  Total successful: {len(successful)}/{len(results_df)}")
        logger.info(f"  Average tracks: {successful['n_tracks'].mean():.1f}")
        logger.info(f"  Average track points: {successful['n_track_points'].mean():.1f}")
        logger.info(f"  Average execution time: {successful['execution_time_s'].mean():.1f}s")
        logger.info("")

        # Find best by track count
        best_by_tracks = successful.nlargest(3, "n_tracks")[
            list(param_ranges.keys()) + ["n_tracks", "n_divisions"]
        ]
        logger.info("Top 3 by track count:")
        for line in best_by_tracks.to_string(index=False).split("\n"):
            logger.info(f"  {line}")
    else:
        logger.error("No successful runs!")

    logger.info(f"Log file: {log_file}")


if __name__ == "__main__":
    main()
