# Simultaneous Live + GPKG Results Viewing — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow live (in-progress) and GPKG (baked) results to be viewed simultaneously in plots and overlay, with per-run overlay selection.

**Architecture:** Track `_live_run_id` on `SWE2DResultsData` to associate live arrays with their owning run. Clear live data on finalize so runs transition to GPKG seamlessly. Add `_overlay_selected_key` for independent overlay run selection.

**Tech Stack:** Python 3.12, PyQt5, pyqtgraph, numpy, sqlite3, QGIS

**Spec:** `docs/2026-07-04-live-gpkg-simultaneous-design.md`

---

## File Map

| File | Responsibility |
|------|----------------|
| `swe2d/results/data.py` | Add `_live_run_id`, `_overlay_selected_key`; update `clear_live_snapshots`, `remove_runs`, state persistence |
| `swe2d/services/gpkg_persistence_service.py` | Filter live path by `_live_run_id` in `load_baked_line_profile` and `load_baked_line_timeseries` |
| `swe2d/workbench/controllers/run_controller.py` | Set `_live_run_id` and `_overlay_selected_key` when run starts |
| `swe2d/runtime/run_finalizer.py` | Call `clear_live_snapshots()` after baking |
| `swe2d/workbench/controllers/overlay_controller.py` | Use overlay-selected run for mesh sync and snapshot loading |
| `swe2d/workbench/views/results_controls.py` | Add overlay-select indicator to runs list; double-click to select |
| `swe2d/workbench/views/studio_viewer_profile_pg.py` | Remove the explicit live→GPKG fallback (now handled by run_id filter in load functions) |
| `tests/test_live_run_id.py` | New test file for `_live_run_id` filtering |
| `tests/test_clear_on_finalize.py` | New test file for finalize clearing live data |
| `tests/test_overlay_selected.py` | New test file for overlay radio-select |

---

### Task 1: Add `_live_run_id` field and set it from run_controller

**Files:**
- Modify: `swe2d/results/data.py:85` (add field after `_live_times`)
- Modify: `swe2d/results/data.py:139` (update `clear_live_snapshots`)
- Modify: `swe2d/workbench/controllers/run_controller.py:380-389` (set `_live_run_id` when reseeding RunRecord)
- Test: `tests/test_live_run_id.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for _live_run_id tracking on SWE2DResultsData."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestLiveRunId(unittest.TestCase):
    def test_live_run_id_starts_empty(self):
        from swe2d.results.data import SWE2DResultsData
        data = SWE2DResultsData()
        self.assertEqual(data._live_run_id, "")

    def test_clear_live_snapshots_clears_live_run_id(self):
        from swe2d.results.data import SWE2DResultsData
        data = SWE2DResultsData()
        data._live_run_id = "run_123"
        data._live_times = np.array([0.0, 10.0])
        data.clear_live_snapshots()
        self.assertEqual(data._live_run_id, "")
        self.assertEqual(data._live_times.size, 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `mamba run -n qgis_stable python -m pytest tests/test_live_run_id.py -x --tb=short`
Expected: FAIL with `AttributeError: 'SWE2DResultsData' object has no attribute '_live_run_id'`

- [ ] **Step 3: Add `_live_run_id` field to `__init__`**

In `swe2d/results/data.py`, after the `_live_times` line (line 85), add both fields (the accessor methods for `_overlay_selected_key` come in Task 5):

```python
        self._live_times: np.ndarray = np.empty(0, dtype=np.float64)
        self._live_run_id: str = ""
        self._overlay_selected_key: str = ""
```

- [ ] **Step 4: Clear `_live_run_id` in `clear_live_snapshots`**

In `swe2d/results/data.py`, inside `clear_live_snapshots` (line 139), add after `self._live_times = np.empty(0, ...)`:

```python
        self._live_times = np.empty(0, dtype=np.float64)
        self._live_run_id = ""
```

- [ ] **Step 5: Run test to verify it passes**

Run: `mamba run -n qgis_stable python -m pytest tests/test_live_run_id.py -x --tb=short`
Expected: PASS (2 tests)

- [ ] **Step 6: Set `_live_run_id` from run_controller when reseeding RunRecord**

In `swe2d/workbench/controllers/run_controller.py`, in the re-seed block (around line 380-389), add after the `gpkg_path` assignment:

```python
                if _live_gpkg:
                    data._run_records[0].gpkg_path = _live_gpkg
                data._live_run_id = str(run_id)
                data._overlay_selected_key = str(data._run_records[0].key)
```

This sets `_live_run_id` when the run starts and auto-selects the live run for overlay.

- [ ] **Step 7: Commit**

```bash
git add swe2d/results/data.py swe2d/workbench/controllers/run_controller.py tests/test_live_run_id.py
git commit -m "feat(results): add _live_run_id tracking for live data identity"
```

---

### Task 2: Filter live path by `_live_run_id` in load functions

**Files:**
- Modify: `swe2d/services/gpkg_persistence_service.py:1008` (filter in `load_baked_line_profile`)
- Modify: `swe2d/services/gpkg_persistence_service.py` (filter in `load_baked_line_timeseries`)
- Test: `tests/test_live_run_id.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_live_run_id.py`:

```python
    def test_load_profile_live_filters_by_run_id(self):
        """load_baked_line_profile only returns live data when _live_run_id matches."""
        from swe2d.results.data import SWE2DResultsData
        from swe2d.services.gpkg_persistence_service import load_baked_line_profile

        data = SWE2DResultsData()
        data._live_run_id = "run_A"
        data._live_times = np.array([0.0, 10.0])
        data._live_line_profile[1] = {
            "station_m": np.array([0.0, 50.0, 100.0]),
            "wse_m": np.full((2, 3), 105.0),
            "bed_m": np.full((2, 3), 95.0),
            "depth_m": np.full((2, 3), 10.0),
            "velocity_ms": np.full((2, 3), 1.0),
            "flow_qn": np.full((2, 3), 50.0),
            "fr": np.full((2, 3), 0.3),
            "wet": np.ones((2, 3), dtype=np.int32),
        }

        # Matching run_id → returns live data
        result = load_baked_line_profile(data, "run_A", 1, 5.0)
        self.assertIn("station_m", result)

        # Non-matching run_id → returns empty (triggers GPKG fallback)
        result = load_baked_line_profile(data, "run_B", 1, 5.0)
        self.assertEqual(result, {})

    def test_load_timeseries_live_filters_by_run_id(self):
        """load_baked_line_timeseries only returns live data when _live_run_id matches."""
        from swe2d.results.data import SWE2DResultsData
        from swe2d.services.gpkg_persistence_service import load_baked_line_timeseries

        data = SWE2DResultsData()
        data._live_run_id = "run_A"
        data._live_times = np.array([0.0, 10.0])
        data._live_line_ts[1] = {
            "line_name": "test",
            "depth_m": np.array([1.0, 2.0]),
            "velocity_ms": np.array([0.5, 1.0]),
            "wse_m": np.array([100.0, 101.0]),
            "bed_m": np.array([95.0, 95.0]),
            "flow_cms": np.array([10.0, 20.0]),
            "wet_frac": np.array([1.0, 1.0]),
            "fr": np.array([0.3, 0.4]),
        }

        # Matching run_id → returns live data
        result = load_baked_line_timeseries(data, "run_A", 1)
        self.assertIn("depth_m", result)

        # Non-matching run_id → returns empty
        result = load_baked_line_timeseries(data, "run_B", 1)
        self.assertEqual(result, {})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `mamba run -n qgis_stable python -m pytest tests/test_live_run_id.py::TestLiveRunId::test_load_profile_live_filters_by_run_id tests/test_live_run_id.py::TestLiveRunId::test_load_timeseries_live_filters_by_run_id -x --tb=short`
Expected: FAIL — live data returned regardless of run_id mismatch

- [ ] **Step 3: Add run_id filter to `load_baked_line_profile`**

In `swe2d/services/gpkg_persistence_service.py`, inside `load_baked_line_profile`, at the start of the `if not isinstance(source, str):` block (around line 1008), add:

```python
        if not isinstance(source, str):
            d = source
            # Only return live data for the run that owns it.
            # Other runs fall through to GPKG.
            if getattr(d, '_live_run_id', '') != run_id:
                return {}
            raw = getattr(d, '_live_line_profile', {}).get(line_id)
```

- [ ] **Step 4: Add run_id filter to `load_baked_line_timeseries`**

Find `load_baked_line_timeseries` in the same file. It has a similar `if not isinstance(source, str):` live-data block. Add the same filter at the top of that block:

```python
        if not isinstance(source, str):
            d = source
            if getattr(d, '_live_run_id', '') != run_id:
                return {}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `mamba run -n qgis_stable python -m pytest tests/test_live_run_id.py -x --tb=short`
Expected: PASS (4 tests)

- [ ] **Step 6: Run broader tests for regressions**

Run: `mamba run -n qgis_stable python -m pytest tests/test_profile_fallback_diagnostic.py tests/test_gpkg_line_results_roundtrip.py tests/test_gpkg_coupling_roundtrip.py tests/test_coupling_snap_indexing.py -x --tb=short -k 'not test_data_overlay_properties'`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add swe2d/services/gpkg_persistence_service.py tests/test_live_run_id.py
git commit -m "feat(results): filter live data by _live_run_id in load functions"
```

---

### Task 3: Simplify profile viewer fallback (the run_id filter handles it now)

**Files:**
- Modify: `swe2d/workbench/views/studio_viewer_profile_pg.py:536-552` (refresh)
- Modify: `swe2d/workbench/views/studio_viewer_profile_pg.py:1007-1021` (_populate_table)

**Rationale:** With the `_live_run_id` filter inside `load_baked_line_profile`, the live path automatically returns `{}` for non-live runs. The explicit live-then-GPKG fallback we added in the previous fix is still correct and needed — the live path returns `{}` for GPKG-only runs (filtered by run_id), and the fallback reads from GPKG. No code change needed here. The existing fallback IS the correct pattern now.

- [ ] **Step 1: Verify the existing fallback works with run_id filtering**

Run: `mamba run -n qgis_stable python -m pytest tests/test_profile_fallback_diagnostic.py tests/test_live_run_id.py -x --tb=short`
Expected: PASS — the fallback still works because live returns `{}` for non-live runs, triggering GPKG

- [ ] **Step 2: Commit (if any changes were needed)**

If no changes needed, skip commit. If the test revealed issues, fix and commit:
```bash
git add swe2d/workbench/views/studio_viewer_profile_pg.py
git commit -m "refactor(profile): run_id filter replaces manual live/GPKG source selection"
```

---

### Task 4: Clear live data on finalize

**Files:**
- Modify: `swe2d/runtime/run_finalizer.py:444-449` (add clear call before return)
- Test: `tests/test_clear_on_finalize.py`

- [ ] **Step 1: Write the failing test**

```python
"""Test that finalize_and_persist clears live snapshots after baking."""
import os
import sys
import unittest
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestClearOnFinalize(unittest.TestCase):
    def test_finalize_clears_live_snapshots(self):
        """After finalize, _live_times and _live_run_id must be cleared."""
        from swe2d.runtime.run_finalizer import SWE2DRunFinalizer
        from swe2d.results.data import SWE2DResultsData

        data = SWE2DResultsData()
        data._live_run_id = "run_X"
        data._live_times = np.array([0.0, 10.0])
        data._live_h = np.zeros((2, 5))
        data._live_hu = np.zeros((2, 5))
        data._live_hv = np.zeros((2, 5))

        view = MagicMock()
        view.log_message = MagicMock()
        view.is_cancel_requested = MagicMock(return_value=False)
        view.get_line_results_storage_path = MagicMock(return_value="")
        view.refresh_plot = MagicMock()

        finalizer = SWE2DRunFinalizer(view)

        # Inspect the finalize_and_persist source — it must call
        # clear_live_snapshots on results_data before returning.
        import inspect
        src = inspect.getsource(finalizer.finalize_and_persist)
        self.assertIn(
            "clear_live_snapshots",
            src,
            "finalize_and_persist must call clear_live_snapshots() "
            "to transition the run from live to GPKG data.",
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `mamba run -n qgis_stable python -m pytest tests/test_clear_on_finalize.py -x --tb=short`
Expected: FAIL — `clear_live_snapshots` not in source

- [ ] **Step 3: Add `clear_live_snapshots()` call in finalizer**

In `swe2d/runtime/run_finalizer.py`, the `finalize_and_persist` method has access to `results_data` (via `self._view._results_data` or a parameter). Before the `return status` at the end (around line 449), add:

```python
        # Clear live snapshot data — the run is now fully baked to GPKG.
        # This transitions the run from live arrays to GPKG reads so
        # plots and overlay use the persisted data.  The RunRecord (with
        # gpkg_path set during run setup) remains for GPKG reads.
        _rd = getattr(self._view, "_results_data", None)
        if _rd is not None:
            _rd.clear_live_snapshots()

        return status
```

- [ ] **Step 4: Run test to verify it passes**

Run: `mamba run -n qgis_stable python -m pytest tests/test_clear_on_finalize.py -x --tb=short`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add swe2d/runtime/run_finalizer.py tests/test_clear_on_finalize.py
git commit -m "feat(finalize): clear live snapshots after baking to GPKG"
```

---

### Task 5: Add `_overlay_selected_key` field with persistence and cleanup

**Files:**
- Modify: `swe2d/results/data.py` (add field, accessor, state persistence, cleanup in remove_runs)
- Test: `tests/test_overlay_selected.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for _overlay_selected_key on SWE2DResultsData."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestOverlaySelected(unittest.TestCase):
    def test_overlay_selected_key_starts_empty(self):
        from swe2d.results.data import SWE2DResultsData
        data = SWE2DResultsData()
        self.assertEqual(data._overlay_selected_key, "")

    def test_remove_runs_clears_overlay_selected_if_matched(self):
        from swe2d.results.data import SWE2DResultsData
        from swe2d.results.run_service import RunRecord
        data = SWE2DResultsData()
        rec = RunRecord(
            run_id="r1", gpkg_path="/tmp/test.gpkg",
            color=(31, 119, 180), enabled=True,
        )
        data._run_records = [rec]
        data._overlay_selected_key = rec.key
        data.remove_runs({rec.key})
        self.assertEqual(data._overlay_selected_key, "")

    def test_remove_runs_preserves_overlay_selected_if_not_matched(self):
        from swe2d.results.data import SWE2DResultsData
        from swe2d.results.run_service import RunRecord
        data = SWE2DResultsData()
        rec1 = RunRecord(
            run_id="r1", gpkg_path="/tmp/a.gpkg",
            color=(31, 119, 180), enabled=True,
        )
        rec2 = RunRecord(
            run_id="r2", gpkg_path="/tmp/b.gpkg",
            color=(255, 127, 14), enabled=True,
        )
        data._run_records = [rec1, rec2]
        data._overlay_selected_key = rec2.key
        data.remove_runs({rec1.key})
        self.assertEqual(data._overlay_selected_key, rec2.key)

    def test_overlay_selected_run_returns_selected_or_first_enabled(self):
        from swe2d.results.data import SWE2DResultsData
        from swe2d.results.run_service import RunRecord
        data = SWE2DResultsData()
        rec1 = RunRecord(run_id="r1", gpkg_path="/tmp/a.gpkg", color=(1,2,3), enabled=True)
        rec2 = RunRecord(run_id="r2", gpkg_path="/tmp/b.gpkg", color=(4,5,6), enabled=True)
        data._run_records = [rec1, rec2]

        # No selection → first enabled
        self.assertIs(data.overlay_selected_run(), rec1)

        # Selection → selected run
        data._overlay_selected_key = rec2.key
        self.assertIs(data.overlay_selected_run(), rec2)

    def test_state_persists_overlay_selected_key(self):
        from swe2d.results.data import SWE2DResultsData
        data = SWE2DResultsData()
        data._overlay_selected_key = "path::run1"
        state = data.save_data_state()
        self.assertIn("overlay_selected_key", state)
        self.assertEqual(state["overlay_selected_key"], "path::run1")

        data2 = SWE2DResultsData()
        data2.restore_data_state(state)
        self.assertEqual(data2._overlay_selected_key, "path::run1")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `mamba run -n qgis_stable python -m pytest tests/test_overlay_selected.py -x --tb=short`
Expected: FAIL — `_overlay_selected_key` attribute doesn't exist

- [ ] **Step 3: Add `_overlay_selected_key` field and methods to `data.py`**

In `swe2d/results/data.py`:

**a) Add field in `__init__`** (after `_live_run_id`):

```python
        self._live_run_id: str = ""
        self._overlay_selected_key: str = ""
```

**b) Add `overlay_selected_run()` method** (in the "Public: overlay support" section, around line 860):

```python
    def overlay_selected_run(self):
        """Return the overlay-selected RunRecord, or first enabled if none selected."""
        if self._overlay_selected_key:
            for rec in self._run_records:
                if rec.key == self._overlay_selected_key:
                    return rec
        return self.first_enabled_record()

    def set_overlay_selected_key(self, key: str) -> None:
        """Set which run is selected for map overlay display."""
        self._overlay_selected_key = str(key or "")
```

**c) Clear in `remove_runs`** — in the `remove_runs` method (line 612), add after `self._selected_run_keys -= run_keys`:

```python
        self._selected_run_keys -= run_keys
        if self._overlay_selected_key in run_keys:
            self._overlay_selected_key = ""
```

**d) Persist in `save_data_state`** — add to the returned dict:

```python
            "overlay_selected_key": self._overlay_selected_key,
```

**e) Restore in `restore_data_state`** — add:

```python
        self._overlay_selected_key = str(state.get("overlay_selected_key", "") or "")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `mamba run -n qgis_stable python -m pytest tests/test_overlay_selected.py -x --tb=short`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add swe2d/results/data.py tests/test_overlay_selected.py
git commit -m "feat(results): add _overlay_selected_key for independent overlay run selection"
```

---

### Task 6: Use overlay-selected run in overlay_controller

**Files:**
- Modify: `swe2d/workbench/controllers/overlay_controller.py:107` (`sync_high_perf_overlay_data`)
- Modify: `swe2d/workbench/controllers/overlay_controller.py:626` (`load_mesh_snapshot_for_overlay`)
- Test: `tests/test_overlay_selected.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_overlay_selected.py`:

```python
    def test_sync_high_perf_uses_overlay_selected_run(self):
        """sync_high_perf_overlay_data must use the overlay-selected run,
        not just first_enabled_record()."""
        import inspect
        from swe2d.workbench.controllers.overlay_controller import OverlayController
        src = inspect.getsource(OverlayController.sync_high_perf_overlay_data)
        self.assertIn(
            "overlay_selected_run",
            src,
            "sync_high_perf_overlay_data must use overlay_selected_run() "
            "instead of first_enabled_record()",
        )

    def test_load_mesh_snapshot_uses_overlay_selected_run(self):
        """load_mesh_snapshot_for_overlay must use overlay-selected run."""
        import inspect
        from swe2d.workbench.controllers.overlay_controller import OverlayController
        src = inspect.getsource(OverlayController.load_mesh_snapshot_for_overlay)
        self.assertIn(
            "overlay_selected_run",
            src,
            "load_mesh_snapshot_for_overlay must use overlay_selected_run()",
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `mamba run -n qgis_stable python -m pytest tests/test_overlay_selected.py::TestOverlaySelected::test_sync_high_perf_uses_overlay_selected_run tests/test_overlay_selected.py::TestOverlaySelected::test_load_mesh_snapshot_uses_overlay_selected_run -x --tb=short`
Expected: FAIL — `overlay_selected_run` not in source

- [ ] **Step 3: Update `sync_high_perf_overlay_data` to use overlay-selected run**

In `swe2d/workbench/controllers/overlay_controller.py`, in `sync_high_perf_overlay_data` (around line 107), replace:

```python
                    rec = self._data.first_enabled_record()
```

with:

```python
                    rec = self._data.overlay_selected_run()
```

- [ ] **Step 4: Update `load_mesh_snapshot_for_overlay` to use overlay-selected run**

In the same file, in `load_mesh_snapshot_for_overlay` (around line 620-630), replace the `enabled_overlay_targets()[0]` lookup:

```python
        run_targets = data.enabled_overlay_targets()
        if not run_targets:
            return False
        gpkg, run_id = run_targets[0]
```

with:

```python
        rec = data.overlay_selected_run()
        if rec is None:
            return False
        gpkg = str(rec.gpkg_path or "")
        run_id = str(rec.run_id or "")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `mamba run -n qgis_stable python -m pytest tests/test_overlay_selected.py -x --tb=short`
Expected: PASS (7 tests)

- [ ] **Step 6: Commit**

```bash
git add swe2d/workbench/controllers/overlay_controller.py tests/test_overlay_selected.py
git commit -m "feat(overlay): use overlay-selected run for mesh sync and snapshot loading"
```

---

### Task 7: Add overlay-select UI to runs list (double-click)

**Files:**
- Modify: `swe2d/workbench/views/results_controls.py:412-462` (add double-click handler, visual indicator)
- Test: `tests/test_overlay_selected.py`

**Design decision:** Double-clicking a run selects it for overlay (simplest UI, no new widgets). The selected run gets a "[map]" prefix or bold font. The existing checkbox controls plot visibility independently.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_overlay_selected.py`:

```python
    def test_run_list_has_overlay_select_handler(self):
        """ResultsToolbox must wire a double-click handler for overlay selection."""
        import inspect
        from swe2d.workbench.views.results_controls import ResultsToolbox
        src = inspect.getsource(ResultsToolbox)
        self.assertIn(
            "_on_run_double_clicked",
            src,
            "ResultsToolbox must have a _on_run_double_clicked method "
            "for selecting the overlay run",
        )

    def test_rebuild_run_list_shows_overlay_indicator(self):
        """_rebuild_run_list must mark the overlay-selected run visually."""
        import inspect
        from swe2d.workbench.views.results_controls import ResultsToolbox
        src = inspect.getsource(ResultsToolbox._rebuild_run_list)
        self.assertIn(
            "overlay_selected_key",
            src,
            "_rebuild_run_list must check overlay_selected_key to mark "
            "the overlay-active run",
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `mamba run -n qgis_stable xvfb-run -a python -m pytest tests/test_overlay_selected.py::TestOverlaySelected::test_run_list_has_overlay_select_handler tests/test_overlay_selected.py::TestOverlaySelected::test_rebuild_run_list_shows_overlay_indicator -x --tb=short`
Expected: FAIL

- [ ] **Step 3: Add double-click handler and visual indicator**

In `swe2d/workbench/views/results_controls.py`:

**a) In `_build_runs_section`** (after `self.run_list.setItemDelegate(...)` at line 416), add double-click signal:

```python
        self.run_list.setItemDelegate(_SwatchDelegate(self.run_list))
        self.run_list.itemDoubleClicked.connect(self._on_run_double_clicked)
        layout.addWidget(self.run_list, 1)
```

**b) Add `_on_run_double_clicked` method** (after `_on_run_item_changed`, around line 472):

```python
    def _on_run_double_clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        """Double-click selects a run for map overlay display."""
        if self._data is None:
            return
        run_key = str(item.data(QtCore.Qt.ItemDataRole.UserRole) or "")
        if not run_key:
            return
        self._data.set_overlay_selected_key(run_key)
        self._rebuild_run_list()
        self.run_selection_changed.emit()
```

**c) Update `_rebuild_run_list`** to mark the overlay-selected run. Modify the `for rec in ...` loop:

```python
        for rec in self._data.get_run_records():
            overlay_marker = "\u25cf " if rec.key == self._data._overlay_selected_key else "   "
            item = QtWidgets.QListWidgetItem(f"{overlay_marker}{rec.display_label()}")
            item.setCheckState(
                QtCore.Qt.CheckState.Checked if rec.enabled
                else QtCore.Qt.CheckState.Unchecked)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, rec.key)
            item.setData(QtCore.Qt.ItemDataRole.UserRole + 1, rec.color)
            tip = f"Run: {rec.run_id}\nGPKG: {rec.gpkg_path}"
            if rec.key == self._data._overlay_selected_key:
                tip += "\n[Overlay active — double-click another run to switch]"
            else:
                tip += "\n[Double-click to set as overlay]"
            item.setToolTip(tip)
            self.run_list.addItem(item)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `mamba run -n qgis_stable xvfb-run -a python -m pytest tests/test_overlay_selected.py -x --tb=short`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add swe2d/workbench/views/results_controls.py tests/test_overlay_selected.py
git commit -m "feat(ui): double-click run to select for overlay; show indicator dot"
```

---

### Task 8: Full regression test + integration verification

**Files:**
- Test: all existing test files

- [ ] **Step 1: Purge pycache and run full results test suite**

```bash
find . -type d -name __pycache__ -exec rm -rf {} +
mamba run -n qgis_stable xvfb-run -a python -m pytest \
  tests/test_live_run_id.py \
  tests/test_clear_on_finalize.py \
  tests/test_overlay_selected.py \
  tests/test_profile_fallback_diagnostic.py \
  tests/test_overlay_mesh_gpkg_path.py \
  tests/test_results_panel.py \
  tests/test_results_path_audit_fixes.py \
  tests/test_temporal_controls_audit.py \
  tests/test_coupling_snap_indexing.py \
  tests/test_workbench_tab_views.py \
  tests/test_gpkg_coupling_roundtrip.py \
  tests/test_gpkg_line_results_roundtrip.py \
  tests/test_sample_line_metrics_profile.py \
  tests/test_studio_viewer_profile_no_undefined_t.py \
  tests/test_in_memory_results_render.py \
  --tb=short -k 'not test_data_overlay_properties'
```

Expected: All PASS

- [ ] **Step 2: Verify the run_id flow is end-to-end correct**

Inspect `run_controller.py` to confirm:
1. `_live_run_id` is set when the RunRecord is reseeded (Task 1, Step 6)
2. `_overlay_selected_key` is auto-set to the live run when it starts (Task 1, Step 6)
3. `clear_live_snapshots()` is called on finalize (Task 4)

- [ ] **Step 3: Final commit if any cleanup needed**

```bash
git add -A
git commit -m "test: full regression for simultaneous live+GPKG results"
```

---

## Self-Review Notes

**Spec coverage check:**
- Section 1 (_live_run_id): Tasks 1, 2 ✓
- Section 2 (clear on finalize): Task 4 ✓
- Section 3 (overlay radio-select): Tasks 5, 6, 7 ✓
- Section 4 (data flow): Verified by integration in Task 8 ✓
- Edge case 1 (second run): `clear_live_snapshots` at run start already exists + `_live_run_id` set in Task 1 ✓
- Edge case 2 (overlay-selected removed): Task 5 cleanup in `remove_runs` ✓
- Edge case 3 (no runs enabled): Already handled ✓
- Edge case 4 (live toggled off): Checkbox independent of overlay ✓

**Simplification from spec:** The spec says `set_live_snapshot_timesteps` should accept a `run_id` parameter. The plan instead sets `_live_run_id` directly from `run_controller.py` when the run starts. This is simpler — no signature changes to `set_live_snapshot_timesteps`, no changes to `runtime_reporting.py`, and the value is set once at run start rather than on every snapshot readback.
