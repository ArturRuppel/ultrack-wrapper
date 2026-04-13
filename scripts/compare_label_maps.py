#!/home/aruppel/miniconda3/envs/cellflow/bin/python
"""Compare tracked label maps to detect merges, splits, and missed cells.

Compares tracked labels between parameter sets to identify:
- Merged tracks (two separate tracks in one run become one in another)
- Split tracks (one track becomes two)
- Missed cells (cells detected in some runs but not others)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import label as ndimage_label


def load_tracked_labels(label_path: Path) -> np.ndarray:
    """Load tracked labels (T, Z, Y, X) and project to 2D."""
    if not label_path.exists():
        return None
    labels_3d = tifffile.imread(str(label_path))
    # Project to 2D using nearest-to-midplane (already done if using proj2d)
    return labels_3d


def get_2d_projection(labels_3d: np.ndarray) -> np.ndarray:
    """Project 3D labels to 2D using nearest-to-midplane."""
    if labels_3d.ndim == 4:
        # Use proj2d if available, otherwise project
        T, Z, Y, X = labels_3d.shape
        mid_z = (Z - 1) / 2.0

        proj_stack = []
        for t in range(T):
            frame = labels_3d[t]
            unique_labels = np.unique(frame)
            unique_labels = unique_labels[unique_labels > 0]

            if len(unique_labels) == 0:
                proj_stack.append(np.zeros((Y, X), dtype=np.uint16))
                continue

            # Simple nearest-to-midplane
            z_idx = np.arange(Z, dtype=np.float32)
            centroid_z = {}
            for lbl in unique_labels:
                per_z = (frame == lbl).sum(axis=(1, 2)).astype(np.float32)
                total = per_z.sum()
                centroid_z[lbl] = float((per_z * z_idx).sum() / total) if total > 0 else mid_z

            # Paint by distance to midplane
            paint_order = sorted(unique_labels, key=lambda l: -abs(centroid_z[l] - mid_z))
            proj = np.zeros((Y, X), dtype=np.uint16)
            for lbl in paint_order:
                footprint = (frame == lbl).any(axis=0)
                proj[footprint] = lbl

            proj_stack.append(proj)

        return np.stack(proj_stack).astype(np.uint16)
    return labels_3d.astype(np.uint16)


def detect_cells_per_timepoint(labels: np.ndarray) -> list[set]:
    """For each timepoint, return set of cell IDs present."""
    if labels.ndim == 3:  # (T, Y, X)
        T = labels.shape[0]
    else:
        raise ValueError(f"Expected 3D array, got {labels.shape}")

    cells_per_t = []
    for t in range(T):
        unique_labels = set(np.unique(labels[t]))
        unique_labels.discard(0)  # Remove background
        cells_per_t.append(unique_labels)

    return cells_per_t


def analyze_cell_variation(all_labels: list[np.ndarray], run_names: list[str]) -> dict[str, Any]:
    """Analyze which cells are detected across different runs."""
    print("Analyzing cell detection variation...")
    print()

    # Get cells per timepoint for each run
    all_cells = []
    for labels, run_name in zip(all_labels, run_names):
        cells_per_t = detect_cells_per_timepoint(labels)
        all_cells.append((run_name, cells_per_t))

    # Analyze variation per timepoint
    n_timepoints = len(all_cells[0][1])
    timepoint_stats = []

    for t in range(n_timepoints):
        # Get union of all cells at this timepoint
        all_cells_at_t = set()
        cells_per_run = []
        for run_name, cells_per_t in all_cells:
            cells_t = cells_per_t[t]
            all_cells_at_t.update(cells_t)
            cells_per_run.append(cells_t)

        # Count how many runs detect each cell
        detection_counts = {}
        for cell_id in all_cells_at_t:
            count = sum(1 for cells_t in cells_per_run if cell_id in cells_t)
            detection_counts[cell_id] = count

        # Statistics
        if detection_counts:
            counts = list(detection_counts.values())
            n_all_runs = sum(1 for c in counts if c == len(cells_per_run))
            n_missed = sum(1 for c in counts if c == 1)  # Detected in only 1 run

            timepoint_stats.append({
                "t": t,
                "n_cells": len(all_cells_at_t),
                "n_all_runs": n_all_runs,
                "n_missed": n_missed,
                "consistency": n_all_runs / len(all_cells_at_t) if all_cells_at_t else 0,
            })

    stats_df = pd.DataFrame(timepoint_stats)

    print("Cell Detection Consistency Across Runs")
    print("=" * 80)
    print()
    print("Summary Statistics:")
    print(f"  Average cells per timepoint: {stats_df['n_cells'].mean():.0f}")
    print(f"  Average detected in all runs: {stats_df['n_all_runs'].mean():.1f} ({stats_df['n_all_runs'].mean() / stats_df['n_cells'].mean() * 100:.1f}%)")
    print(f"  Average missed cells (1 run only): {stats_df['n_missed'].mean():.1f}")
    print(f"  Average consistency: {stats_df['consistency'].mean():.3f}")
    print()

    # Show timepoints with worst consistency
    worst_t = stats_df.nsmallest(3, 'consistency')
    if len(worst_t) > 0:
        print("Worst consistency timepoints:")
        for _, row in worst_t.iterrows():
            print(f"  t={int(row['t'])}: {row['n_all_runs']:.0f}/{row['n_cells']:.0f} cells ({row['consistency']:.1%})")
        print()

    # Show timepoints with most missed cells
    most_missed_t = stats_df.nlargest(3, 'n_missed')
    if len(most_missed_t) > 0:
        print("Most missed cells (detected in only 1 run):")
        for _, row in most_missed_t.iterrows():
            print(f"  t={int(row['t'])}: {int(row['n_missed'])} missed cells")
        print()

    return stats_df


def main():
    parser = argparse.ArgumentParser(
        description="Compare label maps across runs to detect merges/splits/missed cells",
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Path to sweep results directory",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_csv = results_dir / "results.csv"

    if not results_csv.exists():
        print(f"Error: {results_csv} not found")
        return 1

    # Load results
    results_df = pd.read_csv(results_csv)
    results_df = results_df[results_df["success"]]  # Only successful runs

    if len(results_df) == 0:
        print("No successful runs found")
        return 1

    print("=" * 80)
    print("COMPARING TRACKED LABELS ACROSS RUNS")
    print("=" * 80)
    print()

    # Load label maps
    run_names = []
    all_labels = []

    for _, row in results_df.iterrows():
        run_dir = results_dir / row["run_dir"]
        tracked_labels_3d_path = run_dir / "tracked_labels.tif"
        tracked_labels_2d_path = run_dir / "tracked_labels_proj2d.tif"

        # Prefer 2D projection if available
        if tracked_labels_2d_path.exists():
            labels = tifffile.imread(str(tracked_labels_2d_path))
        elif tracked_labels_3d_path.exists():
            labels_3d = tifffile.imread(str(tracked_labels_3d_path))
            labels = get_2d_projection(labels_3d)
        else:
            print(f"Warning: No label file found in {run_dir}")
            continue

        run_names.append(row["run_dir"])
        all_labels.append(labels)

    if not all_labels:
        print("No label maps could be loaded")
        return 1

    print(f"Loaded {len(all_labels)} label maps")
    print(f"  Shape: {all_labels[0].shape}")
    print()

    # Analyze cell detection
    cell_stats = analyze_cell_variation(all_labels, run_names)

    # Save analysis
    analysis_csv = results_dir / "cell_detection_analysis.csv"
    cell_stats.to_csv(analysis_csv, index=False)
    print(f"Analysis saved to: {analysis_csv}")
    print()

    # Key findings
    print("=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print()

    consistency = cell_stats["consistency"].mean()
    if consistency > 0.95:
        print("✓ EXCELLENT: Very consistent cell detection across parameters")
    elif consistency > 0.85:
        print("⚠ GOOD: Mostly consistent, but some variation")
    else:
        print("✗ POOR: Significant variation in cell detection")

    missed_avg = cell_stats["n_missed"].mean()
    if missed_avg < 10:
        print(f"✓ LOW missed cells: {missed_avg:.1f} per timepoint")
    elif missed_avg < 50:
        print(f"⚠ MODERATE missed cells: {missed_avg:.1f} per timepoint")
    else:
        print(f"✗ HIGH missed cells: {missed_avg:.1f} per timepoint")

    print()
    print("Interpretation:")
    print("  - Consistency < 0.95: Parameter choice significantly affects which cells are detected")
    print("  - Missed cells: Cells detected in some runs but not others (potentially merged/split)")
    print("  - Goal: Find parameters that maximize consistency and minimize missed cells")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
