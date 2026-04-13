#!/home/aruppel/miniconda3/envs/cellflow/bin/python
"""Detect track merges and splits between parameter sets.

A merge: one cell in Run A becomes part of a larger region in Run B (split into multiple in B)
A split: one cell in Run A is subdivided (merged into different cells in Run B)

This reveals whether parameter changes cause cell fragmentation or aggregation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import label as ndimage_label
from scipy.ndimage import find_objects


def load_2d_labels(label_path: Path) -> np.ndarray | None:
    """Load 2D projected labels."""
    if label_path.exists():
        return tifffile.imread(str(label_path)).astype(np.uint16)
    return None


def analyze_label_overlaps(labels1: np.ndarray, labels2: np.ndarray, t: int) -> dict[str, Any]:
    """Analyze overlaps between two label frames.

    Returns counts of:
    - 1-to-1 matches (same cell detected)
    - 1-to-many (cell split: one cell becomes multiple)
    - many-to-1 (cell merged: multiple cells become one)
    """
    frame1 = labels1[t]
    frame2 = labels2[t]

    if frame1.size == 0 or frame2.size == 0:
        return {"perfect": 0, "split": 0, "merged": 0, "confused": 0}

    # Get unique labels
    ids1 = set(np.unique(frame1))
    ids1.discard(0)
    ids2 = set(np.unique(frame2))
    ids2.discard(0)

    results = {"perfect": 0, "split": 0, "merged": 0, "confused": 0}

    # For each cell in frame1, see how many cells it overlaps with in frame2
    for cell_id in ids1:
        mask1 = frame1 == cell_id
        overlapping_ids = set(np.unique(frame2[mask1]))
        overlapping_ids.discard(0)

        if len(overlapping_ids) == 0:
            # Cell disappeared
            results["confused"] += 1
        elif len(overlapping_ids) == 1:
            # Perfect 1-to-1 or merged into larger
            # Check if it's a complete match
            cell_id2 = list(overlapping_ids)[0]
            mask2 = frame2 == cell_id2
            if np.array_equal(mask1, mask2):
                results["perfect"] += 1
            else:
                results["merged"] += 1  # Part of a larger cell
        else:
            # Split into multiple cells
            results["split"] += 1

    return results


def compare_two_runs(
    labels1: np.ndarray,
    labels2: np.ndarray,
    run1_name: str,
    run2_name: str,
) -> dict[str, Any]:
    """Compare two label maps and return merge/split statistics."""
    if labels1.shape != labels2.shape:
        raise ValueError(f"Shape mismatch: {labels1.shape} vs {labels2.shape}")

    T = labels1.shape[0]
    all_results = []

    for t in range(T):
        overlap_stats = analyze_label_overlaps(labels1, labels2, t)
        overlap_stats["t"] = t
        all_results.append(overlap_stats)

    stats_df = pd.DataFrame(all_results)

    summary = {
        "run1": run1_name,
        "run2": run2_name,
        "perfect_matches": stats_df["perfect"].mean(),
        "avg_splits": stats_df["split"].mean(),
        "avg_merged": stats_df["merged"].mean(),
        "avg_confused": stats_df["confused"].mean(),
        "split_rate": (stats_df["split"].sum() / stats_df["split"].sum() + stats_df["perfect"].sum() + stats_df["merged"].sum() + 1e-6) * 100,
        "merge_rate": (stats_df["merged"].sum() / stats_df["split"].sum() + stats_df["perfect"].sum() + stats_df["merged"].sum() + 1e-6) * 100,
    }

    return summary, stats_df


def main():
    parser = argparse.ArgumentParser(
        description="Detect merges and splits between parameter sets",
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
    results_df = results_df[results_df["success"]]

    if len(results_df) < 2:
        print("Need at least 2 runs to compare")
        return 1

    print("=" * 80)
    print("ANALYZING MERGES AND SPLITS")
    print("=" * 80)
    print()

    # Load all label maps
    all_labels = {}
    run_names = []

    for _, row in results_df.iterrows():
        run_dir = results_dir / row["run_dir"]
        label_path = run_dir / "tracked_labels_proj2d.tif"

        if label_path.exists():
            labels = load_2d_labels(label_path)
            if labels is not None:
                run_names.append(row["run_dir"])
                all_labels[row["run_dir"]] = labels

    if len(all_labels) < 2:
        print("Could not load enough label maps")
        return 1

    print(f"Comparing {len(all_labels)} runs")
    print()

    # Compare all pairs
    comparisons = []

    # Compare each run to a reference (first one with lowest power)
    reference_run = sorted(all_labels.keys())[0]
    reference_labels = all_labels[reference_run]

    print(f"Reference run: {reference_run}")
    print()
    print("Comparison Results:")
    print("-" * 80)
    print(f"{'Run Pair':<60} {'Splits':<10} {'Merges':<10}")
    print("-" * 80)

    for run_name in sorted(all_labels.keys()):
        if run_name == reference_run:
            continue

        labels = all_labels[run_name]
        summary, _ = compare_two_runs(reference_labels, labels, reference_run, run_name)
        comparisons.append(summary)

        print(
            f"{reference_run} → {run_name:<38} "
            f"{summary['avg_splits']:>8.1f}  {summary['avg_merged']:>8.1f}"
        )

    comparison_df = pd.DataFrame(comparisons)

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    print("When comparing runs with different parameters:")
    print()
    print(f"  Average splits per timepoint:  {comparison_df['avg_splits'].mean():.1f}")
    print(f"  Average merges per timepoint:  {comparison_df['avg_merged'].mean():.1f}")
    print(f"  Average confused per timepoint: {comparison_df['avg_confused'].mean():.1f}")
    print()

    print("Interpretation:")
    print("-" * 80)
    print()

    splits = comparison_df['avg_splits'].mean()
    merges = comparison_df['avg_merged'].mean()

    if splits > merges:
        print("✗ FRAGMENTATION: Parameters are causing cell SPLITTING")
        print("   Higher power values fragment cells into more pieces")
        print("   This is VERY BAD - creates false tracks")
    elif merges > splits:
        print("✗ OVER-MERGING: Parameters are causing cell MERGING")
        print("   Higher division_weight is merging separate cells together")
        print("   This is VERY BAD - loses track of individual cells")
    else:
        print("⚠ BALANCED: Roughly equal splits and merges")

    print()
    print("Recommendation:")
    print("-" * 80)
    if splits > merges * 2:
        print("• Lower power value to reduce fragmentation")
        print("• Try power=1.0 or 1.5")
    elif merges > splits * 2:
        print("• Lower division_weight to reduce merging")
        print("• Try division_weight=-0.0005 or -0.001")
    else:
        print("• Look at biological accuracy (are the merged/split cells correct?)")
        print("• Consider intermediate parameter values")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
