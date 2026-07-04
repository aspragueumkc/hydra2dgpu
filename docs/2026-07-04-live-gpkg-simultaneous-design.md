# Simultaneous Live + GPKG Results Viewing

**Date:** 2026-07-04
**Status:** Approved design, pending implementation

## Problem

The results viewer cannot display live (in-progress simulation) and GPKG
(baked/finalized) results at the same time. Three root causes:

1. **Live data is global, not per-run.** `_live_times`, `_live_h`,
   `_live_line_profile` are flat arrays/dicts on `SWE2DResultsData`, keyed
   by `line_id` only. There is no `run_id` association. When the profile
   viewer iterates enabled runs and tries the live path, it returns the
   same live data for ALL runs — not just the live run.

2. **Live data persists after finalize.** The live arrays are never cleared
   when a run finishes (only at the start of the next run). This causes
   stale-data bugs and prevents the run from reading its own baked GPKG
   results.

3. **Overlay has no explicit run selection.** The overlay uses whichever
   run happens to be first in the enabled list. The user cannot select
   which run drives the depth/velocity overlay, independent of plot
   visibility.

## Design

### Approach: Track `_live_run_id` + clear on finalize + overlay radio-select

There is only ever one live run at a time. A `_live_run_id` field on
`SWE2DResultsData` is the simplest correct way to associate live arrays
with their owning run.

### Section 1: Live data identity (`_live_run_id`)

**Changes to `swe2d/results/data.py`:**

- Add `_live_run_id: str = ""` field.
- `set_live_snapshot_timesteps(timesteps, t_sec, run_id="")` accepts a
  `run_id` parameter and stores it in `_live_run_id`.
- `clear_live_snapshots()` clears `_live_run_id = ""`.

**Changes to `swe2d/services/gpkg_persistence_service.py`:**

- `load_baked_line_profile(source, run_id, line_id, t_sec)`: when `source`
  is a data object (not a string), only return live profile data if
  `getattr(d, '_live_run_id', '') == run_id`. Otherwise return `{}`,
  triggering the caller's GPKG fallback.
- `load_baked_line_timeseries(source, run_id, line_id)`: same filter.

**Changes to `swe2d/workbench/controllers/run_controller.py`:**

- When calling `set_live_snapshot_timesteps` from the reporter callback,
  pass the current `run_id`.

**Effect:** During a run, only the live run's RunRecord gets live data.
All other enabled runs read from their own GPKG paths. Plots show all
enabled runs correctly — live run from live arrays, GPKG runs from GPKG.

### Section 2: Clear live data on finalize

**Changes to `swe2d/runtime/run_finalizer.py`:**

- After all GPKG baking is complete (mesh snapshots, line TS+profiles,
  coupling, run log), call `results_data.clear_live_snapshots()`.
- The RunRecord (with `gpkg_path` set during run setup) remains in
  `_run_records`. After clearing, the run reads from its GPKG path like
  any other.

**Effect:** The transition from live to GPKG is seamless. The user sees
the same run in the runs list, same color, same `run_id` — it just starts
reading from GPKG instead of live arrays. No visual disruption.

**Edge case — user scrubbing during finalize:** The finalizer runs in the
same thread as the UI. If the user is mid-scrub when finalize clears live
data, the next `refresh()` reads from GPKG instead. The timesteps match
since we just baked them. A brief flicker is acceptable — better than
stale data.

### Section 3: Overlay radio-select

**Changes to `swe2d/results/data.py`:**

- Add `_overlay_selected_key: str = ""` field (keyed by `RunRecord.key`,
  which is `gpkg_path::run_id`).
- Default: empty string — overlay falls back to first enabled run
  (backward compatible).
- `save_data_state()` / `restore_data_state()`: persist
  `_overlay_selected_key`.
- When a run is removed from the runs list, clear `_overlay_selected_key`
  if it matched that run's key.

**Changes to `swe2d/workbench/controllers/overlay_controller.py`:**

- `sync_high_perf_overlay_data()`: use the overlay-selected run instead
  of `first_enabled_record()`. If the selected run's `run_id` matches
  `_live_run_id`, use live arrays. Otherwise use that run's GPKG path.
- `load_mesh_snapshot_for_overlay()`: same — use overlay-selected run,
  not `enabled_overlay_targets()[0]`.

**Changes to `swe2d/workbench/views/results_controls.py`:**

- Each run row in the runs list gets a small overlay-select indicator
  (e.g., a "map" icon or bold text) showing which run is overlay-active.
- Clicking the indicator/row selects that run for overlay (radio-select
  behavior — only one run active at a time).
- The existing checkbox continues to control plot visibility independently.
- When a new live run starts, auto-select it for overlay.

**Effect:** User can have 5 runs enabled for plot comparison, but click
any one of them to drive the depth/velocity overlay on the map. During a
live run, the live run is auto-selected for overlay (but the user can
switch to a GPKG run if they want).

### Section 4: Data flow

```
DURING RUN
==========
reporter → set_live_snapshot_timesteps(snaps, run_id=X)
  _live_run_id = "X"
  _live_times, _live_h, etc. populated
  populate_live_line_metrics() fills _live_line_profile

Plots: iterate enabled runs
  Run X (live): _live_run_id matches → live arrays
  Run Y (GPKG): _live_run_id ≠ Y → GPKG fallback

Overlay: _overlay_selected_key → Run X (auto-selected)
  _live_run_id == X → live arrays

ON FINALIZE
===========
bake mesh/line/coupling to GPKG
clear_live_snapshots()
  _live_run_id = ""
  _live_times = empty
Run X now reads from GPKG like any other run

Plots: iterate enabled runs
  Run X: _live_run_id empty → GPKG fallback
  Run Y: _live_run_id empty → GPKG fallback

Overlay: _overlay_selected_key → Run X (unchanged)
  _live_run_id empty → GPKG path
```

### Edge cases

1. **User starts a second run while viewing results from run 1**:
   `clear_live_snapshots()` is called at run start (already exists at
   `run_controller.py:334`). Live data from run 1 is cleared. Run 1
   continues reading from GPKG. New live data populates for run 2.

2. **Overlay-selected run is removed from runs list**:
   `_overlay_selected_key` is cleared, overlay falls back to first
   enabled run.

3. **No runs enabled**: overlay shows nothing (already handled). Plots
   show "No data" (already handled).

4. **Live run toggled off via checkbox**: plot hides the live run's line.
   Overlay still shows it if it's the overlay-selected run. These are
   independent controls.

## Files to change

| File | Change |
|------|--------|
| `swe2d/results/data.py` | Add `_live_run_id`, `_overlay_selected_key`; update `set_live_snapshot_timesteps`, `clear_live_snapshots`, state persistence |
| `swe2d/services/gpkg_persistence_service.py` | Filter live path by `_live_run_id` in `load_baked_line_profile` and `load_baked_line_timeseries` |
| `swe2d/workbench/controllers/run_controller.py` | Pass `run_id` to `set_live_snapshot_timesteps` |
| `swe2d/runtime/run_finalizer.py` | Call `clear_live_snapshots()` after baking |
| `swe2d/workbench/controllers/overlay_controller.py` | Use `_overlay_selected_key` for overlay run selection |
| `swe2d/workbench/views/results_controls.py` | Add overlay-select UI to runs list |
| `swe2d/runtime/runtime_reporting.py` | Pass `run_id` to `set_live_snapshot_timesteps` |

## Testing strategy

- Unit test: `load_baked_line_profile(data, run_id=X, ...)` returns live
  data when `_live_run_id == X`, returns `{}` when `_live_run_id != X`.
- Unit test: `clear_live_snapshots()` clears `_live_run_id`.
- Unit test: overlay-selected run is used by `sync_high_perf_overlay_data`.
- Integration test: simulate finalize → verify live data cleared and GPKG
  fallback works.
- Integration test: two runs enabled, live run_id tracking ensures each
  gets correct data source.
