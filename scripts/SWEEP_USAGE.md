# Ultrack Solve Parameter Sweep

Script to run parallel parameter sweeps for the ultrack solve (ILP) step.

## Overview

The solve step in ultrack is single-threaded and computationally expensive. This script runs multiple solves in parallel with different parameter combinations, allowing you to efficiently explore the parameter space.

**Key features:**
- ✅ Parallel execution (multiple solves at once)
- ✅ Per-run subdirectories (organized results)
- ✅ Progress tracking (logging to file)
- ✅ Save/load sweep configs (reuse parameter sets)
- ✅ 2D projections (nearest-to-midplane algorithm)
- ✅ Database copying (no conflicts between runs)

## Environment

The script uses the **cellflow** conda environment automatically (via shebang).

Run the script directly:
```bash
./scripts/sweep_solve_params.py --param power 2.0 3.0 4.0
```

Or explicitly:
```bash
/home/aruppel/miniconda3/envs/cellflow/bin/python scripts/sweep_solve_params.py --param power 2.0 3.0 4.0
```

## Available Parameters

See `src/ultrack_wrapper/_config.py::TrackingConfig` for full details:

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `appear_weight` | -0.001 | negative values | Weight for track appearance |
| `disappear_weight` | -0.001 | negative values | Weight for track disappearance |
| `division_weight` | -0.001 | negative values | Weight for cell divisions |
| `link_function` | "power" | "power", "identity" | Cost function form |
| `power` | 4.0 | 1.0-10.0 | Exponent for power link function |
| `bias` | 0.0 | any | Bias term in cost |
| `solution_gap` | 0.001 | 0.0-0.1 | Optimality gap tolerance |
| `time_limit` | 36000 | seconds | Max solver runtime (0=unlimited) |
| `window_size` | 0 | 0+ | Window for sliding window solve (0=all at once) |

## Basic Usage

### Quick sweep (2 parameters)

```bash
./scripts/sweep_solve_params.py \
  --working-dir /home/aruppel/Data/2026-04-01_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk/pos00/2_ultrack \
  --param power 2.0 3.0 4.0 5.0 \
  --param division_weight -0.001 -0.01 -0.1 \
  --output-dir ./sweep_results_power_div \
  --n-jobs 4
```

This runs 4 × 3 = 12 combinations with 4 parallel processes.

### Single parameter sweep

```bash
./scripts/sweep_solve_params.py \
  --working-dir /path/to/database \
  --param solution_gap 0.001 0.005 0.01 0.05 \
  --output-dir ./sweep_results_gap
```

### Three-parameter sweep

```bash
./scripts/sweep_solve_params.py \
  --working-dir /path/to/database \
  --param power 2.0 3.0 4.0 \
  --param division_weight -0.001 -0.01 -0.1 \
  --param appear_weight -0.001 -0.005 -0.01 \
  --output-dir ./sweep_results_3param
```

This runs 3 × 3 × 3 = 27 combinations.

### Control parallelism

```bash
# Use 8 parallel jobs
./scripts/sweep_solve_params.py \
  --working-dir /path/to/database \
  --param power 2.0 3.0 4.0 \
  --n-jobs 8

# Use all available CPU cores (default)
./scripts/sweep_solve_params.py \
  --working-dir /path/to/database \
  --param power 2.0 3.0 4.0
```

### Dry run (preview without executing)

```bash
./scripts/sweep_solve_params.py \
  --working-dir /path/to/database \
  --param power 2.0 3.0 4.0 \
  --param division_weight -0.001 -0.01 \
  --dry-run
```

Shows what would run and the directory names.

## Save and Load Sweep Configs

### Save a sweep configuration

```bash
./scripts/sweep_solve_params.py \
  --param power 2.0 3.0 4.0 5.0 \
  --param division_weight -0.001 -0.01 -0.1 \
  --save-sweep my_sweep.json
```

Creates `my_sweep.json`:
```json
{
  "param_ranges": {
    "power": [2.0, 3.0, 4.0, 5.0],
    "division_weight": [-0.001, -0.01, -0.1]
  },
  "n_combinations": 12
}
```

### Load and run a saved sweep

```bash
./scripts/sweep_solve_params.py \
  --working-dir /path/to/database \
  --load-sweep my_sweep.json \
  --output-dir ./sweep_results
```

Or preview it:
```bash
./scripts/sweep_solve_params.py \
  --working-dir /path/to/database \
  --load-sweep my_sweep.json \
  --dry-run
```

## Output Structure

Results go to `--output-dir` (default `./sweep_results`):

```
sweep_results/
├── sweep.log                              # Full execution log
├── sweep_config.json                      # Sweep metadata
├── results.csv                            # Summary of all runs
├── division_weight_-0.001_power_2/        # Run 1 subdirectory
│   ├── data.db                            # Copy of database
│   ├── config.json                        # Parameters used
│   ├── tracks.csv                         # Tracked segments
│   ├── tracked_labels.tif                 # 3D tracked labels (T, Z, Y, X)
│   └── tracked_labels_proj2d.tif          # 2D projection (T, Y, X)
├── division_weight_-0.01_power_2/         # Run 2 subdirectory
│   ├── ...
│   └── tracked_labels_proj2d.tif
└── ...
```

**Key features:**
- Each run gets its own subdirectory with a clean name
- Database is copied into each directory (isolated runs)
- Both 3D and 2D label stacks are saved
- 2D projections use nearest-to-midplane algorithm (same as cellpose_segmentation)

### `results.csv` Format

```csv
run_index,run_dir,success,execution_time_s,n_tracks,n_track_points,n_divisions,power,division_weight,error
0,division_weight_-0.001_power_2,True,45.2,127,4523,12,2.0,-0.001,
1,division_weight_-0.01_power_2,True,43.8,129,4612,14,2.0,-0.01,
2,division_weight_-0.001_power_3,True,48.1,131,4701,13,3.0,-0.001,
...
```

## Progress Tracking

All progress is logged to `sweep.log` in the output directory:

```bash
# Monitor progress in real-time
tail -f ./sweep_results/sweep.log

# Or check when done
cat ./sweep_results/sweep.log
```

Example log output:
```
2026-04-13 21:15:30,123 [INFO] Starting parameter sweep with 12 combinations
2026-04-13 21:15:30,456 [INFO] Results directory: ./sweep_results
2026-04-13 21:15:45,789 [INFO] [1/12] ✓ division_weight_-0.001_power_2 (45.2s) tracks=127, points=4523
2026-04-13 21:16:30,123 [INFO] [2/12] ✓ division_weight_-0.01_power_2 (43.8s) tracks=129, points=4612
2026-04-13 21:17:15,456 [INFO] [3/12] ✓ division_weight_-0.001_power_3 (48.1s) tracks=131, points=4701
...
2026-04-13 21:22:30,123 [INFO] Summary (successful runs):
2026-04-13 21:22:30,123 [INFO]   Total successful: 12/12
2026-04-13 21:22:30,123 [INFO]   Average tracks: 128.5
2026-04-13 21:22:30,123 [INFO]   Average track points: 4612.3
2026-04-13 21:22:30,123 [INFO]   Average execution time: 45.3s
```

## 2D Projections

Each run automatically generates a 2D projection using the **nearest-to-midplane** algorithm:

1. For each Z-stack frame, compute the Z centroid of each label
2. Label whose centroid is closest to the Z midplane wins conflicts
3. Apply per-label post-processing:
   - Hole filling
   - Morphological closing
   - Gaussian smoothing + re-thresholding
4. Re-composite so nearest-to-midplane label still wins

This is the same algorithm used in `cellpose_segmentation/stages/s04_nucleus_proj2d.py`.

Output: `tracked_labels_proj2d.tif` in each run subdirectory (T, Y, X) format.

## Analyzing Results

### Quick summary

```bash
# Print the log summary
tail -20 ./sweep_results/sweep.log
```

### Load results in Python

```python
import pandas as pd

df = pd.read_csv('./sweep_results/results.csv')

# Filter successful runs
df_ok = df[df['success']]

# Top 5 by track count
print(df_ok.nlargest(5, 'n_tracks')[['power', 'division_weight', 'n_tracks', 'n_divisions']])

# Average metrics
print(f"Average execution time: {df_ok['execution_time_s'].mean():.1f}s")
print(f"Average tracks: {df_ok['n_tracks'].mean():.0f}")
```

### Visualize results

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./sweep_results/results.csv')
df_ok = df[df['success']]

# Heatmap: n_tracks by parameters
pivot = df_ok.pivot_table(
    values='n_tracks',
    index='division_weight',
    columns='power'
)
pivot.plot(kind='bar')
plt.xlabel('Division Weight')
plt.ylabel('Number of Tracks')
plt.title('Track Count by Parameters')
plt.show()

# Execution time vs track count
plt.scatter(df_ok['execution_time_s'], df_ok['n_tracks'])
plt.xlabel('Execution Time (s)')
plt.ylabel('Track Count')
plt.show()
```

## Tips

- **Start broad, refine later**: Do a coarse sweep first, then fine-tune around the best region
- **Monitor progress**: `tail -f sweep.log` to watch in real-time
- **Reuse configs**: Save parameter sets with `--save-sweep` for reproducibility
- **Check subfolders**: Each run directory has full results + 2D/3D projections
- **Memory usage**: Each parallel job copies the database. For large DBs, reduce `--n-jobs`
- **Database state**: Original database is never modified; all work is in temp copies

## Example Workflow

```bash
# 1. Do a coarse sweep
./scripts/sweep_solve_params.py \
  --working-dir /path/to/database \
  --param power 2.0 4.0 6.0 8.0 \
  --param division_weight -0.001 -0.01 -0.1 \
  --output-dir ./sweep_coarse \
  --n-jobs 4

# 2. Monitor progress
tail -f ./sweep_coarse/sweep.log

# 3. Analyze results
cat ./sweep_coarse/results.csv

# 4. Fine sweep around best region
./scripts/sweep_solve_params.py \
  --working-dir /path/to/database \
  --param power 3.0 3.5 4.0 4.5 5.0 \
  --param division_weight -0.005 -0.007 -0.01 -0.015 \
  --output-dir ./sweep_fine \
  --n-jobs 6

# 5. Compare best runs and visualize
ls sweep_fine/division_weight_*_power_4*/tracked_labels_proj2d.tif
```

## Troubleshooting

- **Module not found**: Make sure you're in the ultrack_wrapper project directory
- **Permission denied**: `chmod +x scripts/sweep_solve_params.py`
- **Out of memory**: Reduce `--n-jobs` (fewer parallel processes)
- **Timeout errors**: Each solve has a 1-hour timeout. For longer solves, edit the script or reduce complexity
- **2D projection missing**: Check `sweep.log` for errors. 3D labels might have unusual shape/format
