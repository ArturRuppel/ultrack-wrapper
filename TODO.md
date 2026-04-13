# TODO

## Redesign: Unified Foreground + Contours + Tracking Widget

Merge the three separate widgets (`_widget_foreground.py`, `_widget_contours.py`,
`_widget_tracking.py`) into a single cohesive widget that lives in its own
sub-package. The widget should present the three stages as collapsible sections
or sub-tabs within one panel, not as three independent tabs.

### Structural changes

- Create a new folder `src/ultrack_wrapper/widgets/` to house all widget code.
  Move (and rename as appropriate):
  - `_widget_foreground.py`  →  `widgets/foreground.py`
  - `_widget_contours.py`    →  `widgets/contours.py`
  - `_widget_tracking.py`    →  `widgets/tracking.py`
  - Add `widgets/__init__.py` that exports the new unified widget class.
- Create `widgets/ultrack_widget.py` — the top-level `QWidget` that composes
  the three stage panels (foreground, contours, tracking) into one docked widget.
- Update `widget.py` (the main `QTabWidget`) to import from `widgets/` instead
  of the old flat `_widget_*.py` files.
- Remove the now-redundant flat `_widget_foreground.py`, `_widget_contours.py`,
  `_widget_tracking.py` files once the new layout is wired up.

### No per-frame intermediate saves

- Remove the existing per-timepoint intermediate file output for foreground and
  contours (currently written as `foreground.tif` / `contours.tif` stacks at
  the end of a full run; make sure no frame-by-frame `.tif` files are written
  during preview or partial runs).
- The final outputs of foreground and contours stages should remain single
  multi-timepoint stacks (T, Y, X), written only when "Run" is clicked.

### Run modes — intermediate steps separately or all-at-once

Each stage section (Foreground, Contours, Tracking) should have its own **Run**
button so stages can be executed independently. In addition, add a top-level
**Run All** button that chains all three stages in order. Suggested layout:

```
[Run Foreground]   [Run Contours]   [Run Tracking]
                   [Run All]
```

The "Run All" flow should be:
1. Foreground → check overwrite → run if needed
2. Contours   → check overwrite → run if needed
3. Tracking   → run (always, or add its own overwrite check)

### Overwrite checkbox

- Each stage section should have its own **Overwrite** checkbox (already present
  in foreground and contours; add one for tracking to replace/complement the
  existing `overwrite` combo-box).
- When **Overwrite** is unchecked and the output file for that stage already
  exists on disk, skip re-running that stage (both for per-stage runs and when
  invoked via "Run All").
- Display a short status message when a stage is skipped, e.g.
  `"Skipping foreground — output exists (overwrite unchecked)"`.

### Run in terminal button

Add a **Run in Terminal** button next to each stage's **Run** button (and
optionally one for "Run All"). Clicking it should:

1. Serialize the current widget parameters to a temporary JSON config file (or
   pass them as CLI arguments).
2. Build a command such as:
   ```
   python -m ultrack_wrapper.stages.s02_foreground \
     --input-dir /path/to/1a_cellpose_nucleus \
     --output-dir /path/to/2_foreground \
     --config /tmp/ultrack_fg_config.json \
     --overwrite
   ```
3. Launch the command in a new OS terminal using the existing
   `runners/terminal.py` infrastructure:
   - Linux:  `gnome-terminal -- bash -c "CMD; exec bash"`
   - macOS:  `osascript -e 'tell app "Terminal" to do script "CMD"'`
   - Windows: `start cmd /k CMD`
4. Each stage script (`s02_foreground.py`, `s02b_contours.py`, `s03_tracking.py`)
   needs a `__main__` entry point (argparse + `--config` JSON or individual
   `--param` flags) so it can be invoked from the command line.

This feature depends on the stage modules having CLI entry points — implement
those first before wiring up the button.

### Tracking output: tracked_labels.tif

- After tracking completes, export a `tracked_labels.tif` (T, Z, Y, X or T, Y, X)
  where each voxel is labelled with its track ID — in addition to the existing
  tracks CSV/zarr output.
- Load **both** outputs into the napari viewer:
  - The tracks layer (existing behaviour).
  - The `tracked_labels.tif` as a `Labels` layer.
- This loading should happen in **two places**:
  1. When the user clicks the **Load** button for the tracking stage (load
     pre-existing results from disk without re-running).
  2. Automatically upon successful completion of a tracking run (either via
     "Run Tracking" or "Run All").

### Implementation order

1. Add `__main__` CLI entry points to `s02_foreground.py`, `s02b_contours.py`,
   and `s03_tracking.py`.
2. Create `src/ultrack_wrapper/widgets/` sub-package with `__init__.py`.
3. Port and refactor the three widget classes into `widgets/foreground.py`,
   `widgets/contours.py`, `widgets/tracking.py`.
4. Build `widgets/ultrack_widget.py` — the unified composite widget with
   per-stage run buttons, a "Run All" button, overwrite checkboxes, and
   "Run in Terminal" buttons.
5. Update `widget.py` (main `QTabWidget`) to use the new composite widget.
6. Delete the old `_widget_*.py` files.
7. Test the full flow: Preview → Run Foreground → Run Contours → Run Tracking →
   Run All (with and without overwrite).
