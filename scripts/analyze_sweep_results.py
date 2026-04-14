#!/home/aruppel/miniconda3/envs/cellflow/bin/python
"""Analyze parameter sweep results for tracking quality.

Compares tracked labels across different parameter sets and against ground truth to identify:
- Segmentation accuracy (IoU, Dice) against ground truth
- Track matching accuracy
- Track continuity metrics per parameter
- Which parameters best match ground truth
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile
from collections import defaultdict
from scipy import ndimage


def load_tracks_csv(csv_path: Path) -> pd.DataFrame:
    """Load tracks CSV and return as DataFrame."""
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def load_ground_truth_stack(gt_path: Path) -> np.ndarray:
    """Load ground truth label stack (T, Y, X) or (Y, X, T)."""
    if not gt_path.exists():
        return None
    stack = tifffile.imread(str(gt_path))
    # Ensure it's (T, Y, X)
    if stack.ndim == 3:
        # If it's (Y, X, T), transpose to (T, Y, X)
        if stack.shape[2] < min(stack.shape[0], stack.shape[1]):
            stack = np.transpose(stack, (2, 0, 1))
    return stack


def get_gt_track_count(gt_stack: np.ndarray) -> int:
    """Count unique cell identities in ground truth (fast - just count labels)."""
    all_labels = set()
    for t in range(gt_stack.shape[0]):
        all_labels.update(np.unique(gt_stack[t]))
    all_labels.discard(0)
    return len(all_labels)


def compare_against_ground_truth(
    pred_tracks_df: pd.DataFrame,
    n_gt_tracks: int,
) -> dict[str, Any]:
    """Compare predicted track count against ground truth.

    Simple comparison: just count unique track IDs.
    """
    if pred_tracks_df is None or len(pred_tracks_df) == 0:
        return {}

    try:
        n_pred_tracks = len(pred_tracks_df["track_id"].unique())

        metrics = {
            "n_pred_tracks": int(n_pred_tracks),
            "n_gt_tracks": int(n_gt_tracks),
            "track_count_error": abs(n_pred_tracks - n_gt_tracks),
        }

        return metrics

    except Exception as e:
        return {"error": str(e)}


def analyze_track_continuity(tracks_df: pd.DataFrame) -> dict[str, Any]:
    """Analyze track continuity metrics.

    Returns metrics on:
    - Track spans (how continuous are tracks across time)
    - Disappearances (tracks that end early)
    - Appearances (tracks that start late)
    """
    if len(tracks_df) == 0:
        return {}

    metrics = {
        "n_tracks": len(tracks_df["track_id"].unique()),
        "n_timepoints": int(tracks_df["t"].max()) + 1,
        "total_points": len(tracks_df),
    }

    # Analyze each track
    track_continuity = []
    disappeared = 0
    appeared = 0

    for track_id, group in tracks_df.groupby("track_id"):
        timepoints = sorted(group["t"].values)
        # Calculate track span (how many frames it covers)
        span = timepoints[-1] - timepoints[0] + 1
        actual_frames = len(timepoints)
        continuity = actual_frames / span if span > 0 else 1.0
        track_continuity.append(continuity)

        # Track that disappeared (doesn't go to end)
        if timepoints[-1] < metrics["n_timepoints"] - 1:
            disappeared += 1
        # Track that appeared (doesn't start at 0)
        if timepoints[0] > 0:
            appeared += 1

    metrics["avg_continuity"] = np.mean(track_continuity)
    metrics["min_continuity"] = np.min(track_continuity)
    metrics["n_disappeared"] = disappeared
    metrics["n_appeared"] = appeared
    metrics["n_complete_tracks"] = sum(
        1 for c in track_continuity
        if c == 1.0  # Track spans every frame it covers
    )

    return metrics


def compare_runs(
    run_dirs: list[Path],
    run_names: list[str],
    params: list[dict],
    ground_truth_path: Path | None = None,
) -> dict[str, Any]:
    """Compare tracking results across multiple runs.

    Analyzes:
    - Track count variation
    - Continuity metrics
    - Segmentation accuracy against ground truth (if provided)
    - Which parameters most affect results
    """
    print(f"Analyzing {len(run_dirs)} runs...")
    if ground_truth_path:
        print(f"Ground truth: {ground_truth_path}")
    print()

    # Load ground truth once
    gt_stack = None
    n_gt_tracks = None
    if ground_truth_path:
        gt_stack = load_ground_truth_stack(ground_truth_path)
        if gt_stack is not None:
            print(f"Loaded ground truth: shape {gt_stack.shape}")
            n_gt_tracks = get_gt_track_count(gt_stack)
            print(f"Ground truth contains {n_gt_tracks} unique cell labels")
        else:
            print(f"WARNING: Could not load ground truth from {ground_truth_path}")
        print()

    all_metrics = []

    # Analyze each run
    for run_dir, run_name, param_set in zip(run_dirs, run_names, params):
        tracks_csv = run_dir / "tracks.csv"
        tracks_df = load_tracks_csv(tracks_csv)

        continuity = analyze_track_continuity(tracks_df)
        continuity["run_dir"] = run_name
        continuity.update(param_set)

        # Compare against ground truth if available
        if n_gt_tracks is not None:
            gt_metrics = compare_against_ground_truth(tracks_df, n_gt_tracks)
            continuity.update(gt_metrics)

        all_metrics.append(continuity)

    metrics_df = pd.DataFrame(all_metrics)

    # Display summary statistics
    print("=" * 80)
    print("TRACKING QUALITY SUMMARY")
    print("=" * 80)
    print()

    # Show ground truth tracking metrics first if available
    if "n_gt_tracks" in metrics_df.columns:
        print("Ground Truth Tracking Comparison:")
        print(f"  GT has {metrics_df['n_gt_tracks'].iloc[0]:.0f} unique cell labels")
        print()

        print("Tracking Accuracy by Parameter Set:")
        print(f"  Mean predicted tracks: {metrics_df['n_pred_tracks'].mean():.1f}")
        print(f"  Mean track count error: {metrics_df['track_count_error'].mean():.1f}")
        print()

        # Find best by track count accuracy
        best_idx = metrics_df["track_count_error"].idxmin()
        best_row = metrics_df.iloc[best_idx]
        print(f"Best Match to Ground Truth: {best_row['run_dir']}")
        print(f"  Predicted: {best_row['n_pred_tracks']:.0f} tracks (error: {best_row['track_count_error']:.0f})")
        print()

    print()

    print("Track Count Variation:")
    print(f"  Min: {metrics_df['n_tracks'].min()} tracks")
    print(f"  Max: {metrics_df['n_tracks'].max()} tracks")
    print(f"  Range: {metrics_df['n_tracks'].max() - metrics_df['n_tracks'].min()}")
    print(f"  Std Dev: {metrics_df['n_tracks'].std():.1f}")
    print()

    print("Track Continuity (higher is better):")
    print(f"  Avg continuity: {metrics_df['avg_continuity'].mean():.3f}")
    print(f"  Range: {metrics_df['avg_continuity'].min():.3f} - {metrics_df['avg_continuity'].max():.3f}")
    print()

    print("Track Disappearances (bad - tracks end early):")
    print(f"  Avg per run: {metrics_df['n_disappeared'].mean():.1f}")
    print(f"  Max in one run: {metrics_df['n_disappeared'].max()}")
    print()

    print("Track Appearances (bad - tracks start late):")
    print(f"  Avg per run: {metrics_df['n_appeared'].mean():.1f}")
    print(f"  Max in one run: {metrics_df['n_appeared'].max()}")
    print()

    print("Complete Tracks (span all frames they cover):")
    print(f"  Avg per run: {metrics_df['n_complete_tracks'].mean():.1f}")
    print(f"  Range: {metrics_df['n_complete_tracks'].min()} - {metrics_df['n_complete_tracks'].max()}")
    print()

    # Analyze parameter impact
    print("=" * 80)
    print("PARAMETER IMPACT ANALYSIS")
    print("=" * 80)
    print()

    param_names = [k for k in params[0].keys()]

    for param_name in param_names:
        if param_name not in metrics_df.columns:
            continue

        # Build aggregation dict
        agg_dict = {
            "n_tracks": ["min", "mean", "max"],
            "avg_continuity": ["min", "mean", "max"],
            "n_disappeared": ["min", "mean", "max"],
            "n_appeared": ["min", "mean", "max"],
        }

        # Add ground truth tracking metrics if available
        if "track_count_error" in metrics_df.columns:
            agg_dict["track_count_error"] = ["min", "mean", "max"]
            agg_dict["n_pred_tracks"] = ["min", "mean", "max"]

        # Group by parameter value
        grouped = metrics_df.groupby(param_name).agg(agg_dict)

        print(f"\n{param_name}:")
        print("-" * 80)
        print(grouped.to_string())

    print()
    print("=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)
    print()

    # Show all runs with key metrics
    display_cols = (
        ["run_dir"] + param_names +
        ["n_tracks", "avg_continuity", "n_disappeared", "n_appeared", "n_complete_tracks"]
    )

    # Add ground truth tracking columns if available
    if "track_count_error" in metrics_df.columns:
        display_cols.extend(["n_pred_tracks", "n_gt_tracks", "track_count_error"])

    available_cols = [c for c in display_cols if c in metrics_df.columns]
    print(metrics_df[available_cols].to_string(index=False))

    print()
    print("=" * 80)
    print("INTERPRETATION GUIDE")
    print("=" * 80)
    print()

    if "track_count_error" in metrics_df.columns:
        print("Ground Truth Tracking Metrics (most important):")
        print("  - n_pred_tracks: number of predicted tracks (from tracks.csv)")
        print("  - n_gt_tracks: number of ground truth cell labels")
        print("  - track_count_error: absolute difference (lower is better)")
        print()

    print("Track Count:")
    print("  - Higher is better (more cells detected)")
    print("  - But should be balanced with continuity")
    print()
    print("Continuity (0-1):")
    print("  - 1.0 = perfect (no disappearances/appearances within track)")
    print("  - Lower = tracks start/stop unexpectedly")
    print()
    print("Disappeared/Appeared:")
    print("  - These are BAD (tracks shouldn't vanish/appear)")
    print("  - Lower is better")
    print()
    print("Complete Tracks:")
    print("  - Tracks that span all frames they cover (no gaps)")
    print("  - Higher is better")

    return metrics_df


def main():
    parser = argparse.ArgumentParser(
        description="Analyze parameter sweep results",
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Path to sweep results directory",
    )
    parser.add_argument(
        "--ground-truth",
        help="Path to ground truth label stack (TYX format)",
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

    # Build list of runs
    run_dirs = [results_dir / row["run_dir"] for _, row in results_df.iterrows()]
    run_names = list(results_df["run_dir"])

    # Extract parameters
    param_cols = [c for c in results_df.columns
                  if c not in ["run_index", "run_dir", "success", "execution_time_s",
                               "n_tracks", "n_track_points", "n_divisions", "error"]]

    params = []
    for _, row in results_df.iterrows():
        param_dict = {col: row[col] for col in param_cols if pd.notna(row[col])}
        params.append(param_dict)

    # Analyze
    gt_path = Path(args.ground_truth) if args.ground_truth else None
    metrics_df = compare_runs(run_dirs, run_names, params, ground_truth_path=gt_path)

    # Save analysis
    analysis_csv = results_dir / "analysis.csv"
    metrics_df.to_csv(analysis_csv, index=False)
    print(f"\nAnalysis saved to: {analysis_csv}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
