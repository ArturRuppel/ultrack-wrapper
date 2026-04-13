#!/home/aruppel/miniconda3/envs/cellflow/bin/python
"""Analyze parameter sweep results for tracking quality.

Compares tracked labels across different parameter sets to identify:
- Track merges (very bad - two tracks become one)
- Track splits (very bad - one track becomes two)
- Disappeared/appeared tracks (bad - tracks start/end unexpectedly)
- Continuity metrics per parameter
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


def load_tracks_csv(csv_path: Path) -> pd.DataFrame:
    """Load tracks CSV and return as DataFrame."""
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


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
) -> dict[str, Any]:
    """Compare tracking results across multiple runs.

    Analyzes:
    - Track count variation
    - Continuity metrics
    - Which parameters most affect results
    """
    print(f"Analyzing {len(run_dirs)} runs...")
    print()

    all_metrics = []

    # Analyze each run
    for run_dir, run_name, param_set in zip(run_dirs, run_names, params):
        tracks_csv = run_dir / "tracks.csv"
        tracks_df = load_tracks_csv(tracks_csv)

        continuity = analyze_track_continuity(tracks_df)
        continuity["run_dir"] = run_name
        continuity.update(param_set)

        all_metrics.append(continuity)

    metrics_df = pd.DataFrame(all_metrics)

    # Display summary statistics
    print("=" * 80)
    print("TRACKING QUALITY SUMMARY")
    print("=" * 80)
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

        # Group by parameter value
        grouped = metrics_df.groupby(param_name).agg({
            "n_tracks": ["min", "mean", "max"],
            "avg_continuity": ["min", "mean", "max"],
            "n_disappeared": ["min", "mean", "max"],
            "n_appeared": ["min", "mean", "max"],
        })

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
    available_cols = [c for c in display_cols if c in metrics_df.columns]
    print(metrics_df[available_cols].to_string(index=False))

    print()
    print("=" * 80)
    print("INTERPRETATION GUIDE")
    print("=" * 80)
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
    metrics_df = compare_runs(run_dirs, run_names, params)

    # Save analysis
    analysis_csv = results_dir / "analysis.csv"
    metrics_df.to_csv(analysis_csv, index=False)
    print(f"\nAnalysis saved to: {analysis_csv}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
