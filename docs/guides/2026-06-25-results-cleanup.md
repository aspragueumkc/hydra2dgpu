# Results Architecture Cleanup Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean up the results handling architecture — move scattered state from dialog attributes into `SWE2DResultsData`, fix MVP violations in `run_finalizer.py`, remove dead code, and preserve multi-run functionality.

**Architecture:** `SWE2DResultsData` is the central data object for all results state. All UI components (overlay controller, viewer, temporal dock, results panel) read from it instead of reaching into dialog attributes. The `run_finalizer.py` calls presenter methods instead of directly manipulating UI.

**Tech Stack:** Python, PyQt5, numpy, matplotlib, SQLite/GPKG

**Constraints:**
- Multi-run plotting MUST continue to work (checkboxes, `get_enabled_run_records()`, color-coded series)
- In-memory snapshots during a run MUST be accessible to the overlay controller
- GPKG-backed results after a run MUST use the same API as in-memory snapshots

---

### Task 1: Move overlay arrays from dialog attrs to SWE2DResultsData

The overlay geometry arrays are stashed as dialog attributes (`_high_perf_overlay_cell_x`, `_high_perf_overlay_cell_y`, etc.) and accessed via `getattr(view, ...)`. Move them into `SWE2DResultsData` so there's one source of truth.

**Files:**
- Modify: `swe2d/results/data.py`
- Modify: `swe2d/workbench/controllers/overlay_controller.py`

- [ ] **Step 1: Add overlay arrays to SWE2DResultsData**

In `swe2d/results/data.py`, add to `SWE2DResultsData.__init__()` after the coupling records section (after line 62):

```python
        # Overlay geometry arrays (populated by overlay controller)
        self.overlay_cell_x: Optional[np.ndarray] = None
        self.overlay_cell_y: Optional[np.ndarray] = None
        self.overlay_cell_bed: Optional[np.ndarray] = None
        self.overlay_node_x: Optional[np.ndarray] = None
        self.overlay_node_y: Optional[np.ndarray] = None
        self.overlay_cell_nodes: Optional[np.ndarray] = None
        self.overlay_tri_to_cell: Optional[np.ndarray] = None
```

- [ ] **Step 2: Update overlay_controller.py to read/write from SWE2DResultsData**

In `swe2d/workbench/controllers/overlay_controller.py`, find all `getattr(view, "_high_perf_overlay_*")` and `setattr(view, "_high_perf_overlay_*")` patterns. Replace them with reads/writes to `self._data.overlay_*` (where `self._data` is the `SWE2DResultsData` instance).

Example pattern — replace:
```python
cell_x = getattr(view, "_high_perf_overlay_cell_x", None)
```
with:
```python
cell_x = self._data.overlay_cell_x
```

And replace:
```python
setattr(view, "_high_perf_overlay_cell_x", cell_x)
```
with:
```python
self._data.overlay_cell_x = cell_x
```

- [ ] **Step 3: Verify overlay still works**

Run the workbench, load a mesh, run a simulation, verify the overlay renders correctly.

---

### Task 2: Move in-memory snapshots from dialog attrs to SWE2DResultsData

During a run, snapshot data lives in `dialog._snapshot_timesteps`, `dialog._line_snapshot_rows`, `dialog._line_snapshot_profile_rows`, `dialog._coupling_snapshot_rows`. Move these into `SWE2DResultsData` so the data path is unified.

**Files:**
- Modify: `swe2d/results/data.py`
- Modify: `swe2d/runtime/runtime_reporting.py`
- Modify: `swe2d/workbench/controllers/overlay_controller.py`
- Modify: `swe2d/workbench/views/studio_results_panel.py`

- [ ] **Step 1: Add snapshot storage to SWE2DResultsData**

In `swe2d/results/data.py`, add to `SWE2DResultsData.__init__()`:

```python
        # In-memory snapshots during a live run
        self._live_snapshot_timesteps: list = []
        self._live_line_snapshot_rows: list = []
        self._live_line_profile_rows: list = []
        self._live_coupling_snapshot_rows: list = []
```

Add public methods:

```python
    def clear_live_snapshots(self) -> None:
        """Clear all in-memory snapshot data (called at run start)."""
        self._live_snapshot_timesteps = []
        self._live_line_snapshot_rows = []
        self._live_line_profile_rows = []
        self._live_coupling_snapshot_rows = []

    def append_live_snapshot(self, t_s: float, h: np.ndarray, hu: np.ndarray, hv: np.ndarray) -> None:
        """Append a mesh snapshot (called by runtime_reporting each output step)."""
        self._live_snapshot_timesteps.append((t_s, h, hu, hv))

    def append_line_snapshot(self, row: dict) -> None:
        """Append a line sampling row."""
        self._live_line_snapshot_rows.append(row)

    def append_line_profile_snapshot(self, row: dict) -> None:
        """Append a line profile row."""
        self._live_line_profile_rows.append(row)

    def append_coupling_snapshot(self, row: dict) -> None:
        """Append a coupling snapshot row."""
        self._live_coupling_snapshot_rows.append(row)

    def get_live_snapshot_timesteps(self) -> list:
        """Return in-memory snapshot timesteps."""
        return self._live_snapshot_timesteps

    def get_live_line_snapshot_rows(self) -> list:
        """Return in-memory line snapshot rows."""
        return self._live_line_snapshot_rows

    def get_live_line_profile_rows(self) -> list:
        """Return in-memory line profile rows."""
        return self._live_line_profile_rows

    def get_live_coupling_snapshot_rows(self) -> list:
        """Return in-memory coupling snapshot rows."""
        return self._live_coupling_snapshot_rows
```

- [ ] **Step 2: Update runtime_reporting.py to write to SWE2DResultsData**

In `swe2d/runtime/runtime_reporting.py`, find where `dialog._snapshot_timesteps`, `dialog._line_snapshot_rows`, etc. are appended. Replace with calls to `self._results_data.append_live_snapshot()`, `self._results_data.append_line_snapshot()`, etc.

The `runtime_reporting` object needs a reference to `SWE2DResultsData`. Add a `results_data` parameter to its constructor or set it as an attribute.

- [ ] **Step 3: Update overlay_controller.py to read from SWE2DResultsData**

In `swe2d/workbench/controllers/overlay_controller.py`, find where `view._snapshot_timesteps` is read. Replace with `self._data.get_live_snapshot_timesteps()`.

- [ ] **Step 4: Update studio_results_panel.py to read from SWE2DResultsData**

In `swe2d/workbench/views/studio_results_panel.py`, find where `dialog._snapshot_timesteps` is read. Replace with `dialog._results_data.get_live_snapshot_timesteps()` (or similar accessor).

- [ ] **Step 5: Verify live overlay still works**

Run a simulation, verify the overlay updates in real-time.

---

### Task 3: Expose coupling records via public API

`SWE2DResultsData._coupling_records` is accessed as a private attribute by the Structure and Network plot tabs. Expose it via public methods.

**Files:**
- Modify: `swe2d/results/data.py`
- Modify: `swe2d/workbench/views/studio_viewer.py`
- Modify: `swe2d/workbench/views/results_controls.py`

- [ ] **Step 1: Add public accessor to SWE2DResultsData**

In `swe2d/results/data.py`, add:

```python
    def get_coupling_records(self) -> list:
        """Return coupling records for the active run."""
        return list(self._coupling_records)

    def get_coupling_run_id(self) -> str:
        """Return the run ID for the current coupling data."""
        return self._coupling_run_id
```

- [ ] **Step 2: Update consumers to use public API**

Search for `_coupling_records` in the codebase. Replace direct access with `get_coupling_records()`:

```python
# Before:
records = data._coupling_records

# After:
records = data.get_coupling_records()
```

- [ ] **Step 3: Verify coupling plots still work**

Load results with structure coupling data, verify Structure and Network tabs render correctly.

---

### Task 4: Fix run_finalizer.py MVP violations

`SWE2DRunFinalizer` takes `ui` (the dialog) and calls `self._ui._log()`, `self._ui._sync_high_perf_overlay_data()`, `self._ui._refresh_plot()`, etc. This violates MVP — the runtime layer should not reach into the UI.

**Files:**
- Modify: `swe2d/runtime/run_finalizer.py`
- Modify: `swe2d/workbench/controllers/run_controller.py`

- [ ] **Step 1: Define a presenter protocol for finalization**

In `swe2d/runtime/run_finalizer.py`, define a protocol (or use a simple callback pattern):

```python
from typing import Protocol

class RunFinalizationView(Protocol):
    def log_message(self, msg: str) -> None: ...
    def get_line_results_storage_path(self) -> str: ...
    def sync_overlay_data(self) -> None: ...
    def refresh_plot(self) -> None: ...
    def persist_conservation_forensics(self, gpkg_path: str) -> None: ...
```

- [ ] **Step 2: Update SWE2DRunFinalizer to use the protocol**

Change `SWE2DRunFinalizer.__init__()` to accept `view: RunFinalizationView` instead of `ui`. Replace all `self._ui._log(...)` with `self._view.log_message(...)`, etc.

- [ ] **Step 3: Implement the protocol in the run controller**

In `swe2d/workbench/controllers/run_controller.py`, create a small adapter class that implements `RunFinalizationView` and delegates to the dialog:

```python
class _FinalizationAdapter:
    def __init__(self, dialog):
        self._dialog = dialog

    def log_message(self, msg: str) -> None:
        self._dialog._log(msg)

    def get_line_results_storage_path(self) -> str:
        return self._dialog._current_line_results_storage_path()

    def sync_overlay_data(self) -> None:
        self._dialog._sync_high_perf_overlay_data()

    def refresh_plot(self) -> None:
        self._dialog._refresh_plot()

    def persist_conservation_forensics(self, gpkg_path: str) -> None:
        self._dialog._persist_conservation_forensics_to_geopackage(gpkg_path)
```

Pass `_FinalizationAdapter(dialog)` to `SWE2DRunFinalizer` instead of `dialog` directly.

- [ ] **Step 4: Verify finalization still works**

Run a simulation to completion, verify results are persisted to GPKG and the UI updates correctly.

---

### Task 5: Remove dead code (overlay_service.py)

`swe2d/results/overlay_service.py` contains Python overlay functions (`compute_velocity_magnitude`, `compute_wse`, `compute_froude`, `prepare_coupling_timeseries`) that are unused — the actual overlay rendering is done by the C++ `hydra_overlay` native extension.

**Files:**
- Delete: `swe2d/results/overlay_service.py`
- Verify: no imports of `overlay_service` remain

- [ ] **Step 1: Verify overlay_service.py is unused**

```bash
grep -r "overlay_service" swe2d/ --include="*.py" | grep -v "__pycache__"
```

Expected: no results (or only the file itself).

- [ ] **Step 2: Delete the file**

```bash
rm swe2d/results/overlay_service.py
```

- [ ] **Step 3: Verify no import errors**

```bash
python -c "from swe2d.results import data, queries, run_service, timestep_service, animation"
```

Expected: no import errors.

---

### Task 6: Unify data path (in-memory vs GPKG)

The overlay controller branches on `_overlay_data_from_gpkg` to decide whether to read from in-memory snapshots or load from GPKG. Unify this so the same code path is used regardless of data source.

**Files:**
- Modify: `swe2d/results/data.py`
- Modify: `swe2d/workbench/controllers/overlay_controller.py`
- Modify: `swe2d/workbench/views/studio_results_panel.py`

- [ ] **Step 1: Add data source flag to SWE2DResultsData**

In `swe2d/results/data.py`, add:

```python
        # Data source: "live" during run, "gpkg" after persistence
        self._data_source: str = "none"  # "none", "live", "gpkg"

    @property
    def data_source(self) -> str:
        """Return current data source: 'none', 'live', or 'gpkg'."""
        return self._data_source

    def set_data_source(self, source: str) -> None:
        """Set data source flag."""
        self._data_source = source
```

- [ ] **Step 2: Add unified snapshot accessor**

In `swe2d/results/data.py`, add a method that returns snapshot data regardless of source:

```python
    def get_snapshot_at_time(self, t_sec: float) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Return (h, hu, hv) at the given time, from either live or GPKG source.

        Returns None if no data is available.
        """
        if self._data_source == "live":
            # Find closest snapshot in live data
            snapshots = self._live_snapshot_timesteps
            if not snapshots:
                return None
            best = min(snapshots, key=lambda s: abs(s[0] - t_sec))
            return (best[1], best[2], best[3])
        elif self._data_source == "gpkg":
            # Load from GPKG via overlay controller
            # This is handled by overlay_controller.load_mesh_snapshot_for_overlay()
            return None  # Caller should use overlay controller
        return None
```

- [ ] **Step 3: Update overlay_controller.py to use unified interface**

In `swe2d/workbench/controllers/overlay_controller.py`, replace the `_overlay_data_from_gpkg` branching logic with a call to `self._data.get_snapshot_at_time()` or the overlay controller's own unified method.

The key change: instead of checking `_overlay_data_from_gpkg` and then branching between in-memory and GPKG reads, the overlay controller always goes through `SWE2DResultsData`. For live data, `SWE2DResultsData` returns from `_live_snapshot_timesteps`. For GPKG data, `SWE2DResultsData` returns from its cached GPKG data (or triggers a load).

- [ ] **Step 4: Update studio_results_panel.py to set data source**

In `swe2d/workbench/views/studio_results_panel.py`, where `_overlay_data_from_gpkg` is set, replace with `data.set_data_source("gpkg")`. Where live snapshots are fed, replace with `data.set_data_source("live")`.

- [ ] **Step 5: Verify both paths work**

1. Run a simulation → verify live overlay works
2. Close and reopen results → verify GPKG overlay works
3. Enable multiple runs → verify multi-run plots still work

---

### Task 7: Clean up studio_results_panel.py stateless functions

The functions in `studio_results_panel.py` are "stateless" (take `dialog` as first arg) but reach deep into the dialog to manipulate scattered state. After Tasks 1-6, the state is centralized in `SWE2DResultsData`. Clean up the functions to use the centralized data.

**Files:**
- Modify: `swe2d/workbench/views/studio_results_panel.py`

- [ ] **Step 1: Update `on_results_panel_timestep_changed()` to use SWE2DResultsData**

Replace direct dialog attribute access with `SWE2DResultsData` method calls:

```python
# Before:
snapshots = dialog._snapshot_timesteps
overlay_data = getattr(dialog, "_overlay_data_from_gpkg", False)

# After:
data = dialog._results_data
snapshots = data.get_live_snapshot_timesteps()
data_source = data.data_source
```

- [ ] **Step 2: Update `auto_load_results_panel()` to use SWE2DResultsData**

Replace direct dialog attribute access with `SWE2DResultsData` method calls.

- [ ] **Step 3: Update `on_results_refresh()` to use SWE2DResultsData**

Replace direct dialog attribute access with `SWE2DResultsData` method calls.

- [ ] **Step 4: Verify all results panel operations work**

Test: load results, switch timesteps, enable/disable runs, refresh plots.

---

### Task 8: Final verification and cleanup

- [ ] **Step 1: Run full test suite**

```bash
python -m pytest tests/ -v --timeout=120
```

- [ ] **Step 2: Verify multi-run plotting**

1. Load multiple GPKG results
2. Enable 2+ runs via checkboxes
3. Verify time series and profile plots show all enabled runs with distinct colors

- [ ] **Step 3: Verify live overlay**

1. Run a simulation
2. Verify overlay updates in real-time
3. Verify temporal dock slider works

- [ ] **Step 4: Verify GPKG overlay**

1. Load persisted results
2. Verify overlay renders correctly
3. Verify temporal dock slider works

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: unify results state in SWE2DResultsData, fix MVP violations

- Move overlay arrays from dialog attrs to SWE2DResultsData
- Move in-memory snapshots from dialog attrs to SWE2DResultsData
- Expose coupling records via public API
- Fix run_finalizer.py MVP violations (protocol pattern)
- Remove dead overlay_service.py
- Unify in-memory vs GPKG data path
- Clean up studio_results_panel.py to use centralized data
- Multi-run plotting preserved"
```
