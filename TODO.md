# TODO

## Widget layout redesign: collapsible sections with header checkboxes

Every stage section (Foreground, Contours, Segmentation, Linking, Solve,
Results) should be a **collapsible panel** whose header row contains:

- A **enable/disable checkbox** (or overwrite checkbox for cached stages) on
  the left of the title — controls whether "Run All" will execute this step.
- The **section title** label.
- A **collapse/expand toggle** (arrow or +/−) on the right.

The config parameters live *inside* the collapsible body. The checkbox is
**never** inside the collapsed area — it stays visible at all times so the
user can toggle a step without having to expand it first.

Sketch of one section header:

```
[✓] ▶  Segmentation                                    [▲]
```

When collapsed only the header row is shown. When expanded the parameter
widgets appear below it.

---

## Split the Tracking section into three independent pipeline stages

The current Tracking section runs `add_nodes` (segmentation candidates),
`add_edges` (linking), and `track` (ILP solve) as one monolithic step. These
should become **three separate collapsible sections**, each with its own
**Run** button and **enable/disable checkbox** in the header:

### New sections

| Section | Wraps | Cached in DB |
|---|---|---|
| **Segmentation** | `ultrack add_nodes` — detects candidates from foreground/contours | `nodes` table |
| **Linking** | `ultrack add_edges` — scores candidate links between frames | `edges` table |
| **Solve** | `ultrack track` (ILP only) — selects optimal solution | `solution` table |

Each section gets:
- A collapsible config panel (its relevant parameter knobs).
- A **Run** button that executes only that step.
- A header **enable/disable checkbox**: when unchecked, "Run All" skips the
  step and reuses the cached DB result.

### Motivation

Tuning solver weights (`appear_weight`, `disappear_weight`, `division_weight`,
`bias`, `solution_gap`) does not require re-segmenting or re-linking. Keeping
those cached cuts iteration time significantly on large datasets. The checkbox
makes the intent explicit: "I want to reuse the existing candidates/links and
only re-solve."

### Config split

Current `TrackingConfig` fields should be distributed across the three sections:

- **Segmentation config**: `min_area`, `max_area`, `min_frontier`, `threshold`,
  `ws_hierarchy`, `anisotropy_penalization`, `n_workers`
- **Linking config**: `max_distance`, `max_neighbors`, `distance_weight`,
  `n_workers`
- **Solve config**: `appear_weight`, `disappear_weight`, `division_weight`,
  `link_function`, `power`, `bias`, `solution_gap`, `time_limit`,
  `window_size`

### Implementation notes

- Replace the existing `TrackingConfig.overwrite` string enum with three `bool`
  fields: `overwrite_segmentation`, `overwrite_linking`, `overwrite_solve` —
  driven by the three header checkboxes.
- Split `s03_tracking.py:run()` into three functions: `run_segmentation()`,
  `run_linking()`, `run_solve()`, each callable independently.
- "Run All" at the top level chains all enabled sections in order.

---

## Database inspection tab in the Tracking section

Add an **"Inspect DB"** sub-tab inside the Tracking area (alongside the
existing Results tab). The goal is to let the user interrogate the ultrack
SQLite database interactively from within napari.

### What the DB contains that is not yet visible

| Data | Currently exposed |
|---|---|
| Final track centroids + lineage | Yes (Tracks layer via Load Results) |
| Final voxel segmentation | Yes (tracked_labels.tif via Load Results) |
| **All segmentation candidates** (including rejected ones) | No |
| **All candidate links** (scored hypotheses) | No |
| **Division events** specifically | No |
| **Per-segment ILP scores / weights** | No |

### "Inspect DB" tab contents

- **Load candidates** button: reads all candidate centroids from the DB, adds
  them as a `Points` layer coloured by selection status (chosen vs. rejected).
- **Load links** button: reads all candidate links, adds them as a `Vectors`
  layer with opacity/thickness scaled by link score.
- **Load divisions** button: filters solution links for division events and
  adds them as a distinct `Points` layer.
- **Colour by** combo-box: re-colours the Points layer by a chosen per-candidate
  scalar (area, ILP weight, link score, …).

### Implementation notes

- Query the DB via `sqlalchemy` or `sqlite3` — ultrack uses tables `nodes`,
  `edges`, and `solution` (verify exact names from ultrack source).
- All DB reads should run in a `thread_worker`.
- Use `_get_paths()` + `MainConfig.data.working_dir` to locate `data.db`.
