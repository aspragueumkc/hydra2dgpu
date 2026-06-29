# In-Memory Results Viewing Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate GPKG writes during simulation runs — all viewer (overlay, plots) reads from in-memory snapshots. GPKG writes happen only at finalization via `run_finalizer.py`. Max results are GPKG-only.

**Architecture:** `SWE2DResultsData` holds all in-memory snapshot data (`_live_snapshot_timesteps`, `_live_line_snapshot_rows`, `_live_line_profile_rows`, `_live_coupling_snapshot_rows`). The plot render service reads from these when `data_source == "live"`, falling back to GPKG queries when `data_source == "gpkg"`. The snapshot manual-save button is removed — only `run_finalizer.py` persists to GPKG.

**Tech Stack:** Python, PyQt5, numpy, matplotlib, SQLite/GPKG

**File dependency map:**
- Task 1: `results_render_service.py` (modify), `queries.py` (add in-memory loader), `data.py` (verify accessors)
- Task 2: `run_controller.py` (modify), `studio_dialog.py` (modify), `gpkg_service.py` (verify)
- Task 3: `run_controller.py` (verify), `finalization_adapter.py` (verify), `gpkg_persistence_service.py` (verify)

**Parallel execution:** Tasks 1 and 2 are independent (touch no overlapping files). Task 3 is trivial (verification only) and can be done inline.

---

### Task 1: Add in-memory plot rendering path

The time series and profile plots read from GPKG via `queries.load_timeseries()` and `queries.load_profile()`. During a live run, `_live_line_snapshot_rows` and `_live_line_profile_rows` hold the same data in memory. Add an in-memory loader and modify the render service to use it when `data_source == "live"`.

**Files:**
- Modify: `swe2d/results/queries.py`
- Modify: `swe2d/workbench/services/results_render_service.py`
- Modify: `swe2d/results/data.py` (verify)
- Test: `tests/test_in_memory_results_render.py`

- [ ] **Step 1: Write failing tests for in-memory loading**

Create `tests/test_in_memory_results_render.py`:

```python
"""Tests for in-memory results rendering path."""
import numpy as np
from swe2d.results.data import SWE2DResultsData
from swe2d.results.queries import load_timeseries_from_live


def test_load_timeseries_from_live_empty():
    data = SWE2DResultsData()
    result = load_timeseries_from_live(data, "run_1", 0)
    assert result == {}


def test_load_timeseries_from_live_with_rows():
    data = SWE2DResultsData()
    data.append_line_snapshot({"t_s": 0.0, "line_id": 0, "depth_m": 1.0,
                               "velocity_ms": 0.5, "wse_m": 11.0,
                               "bed_m": 10.0, "flow_cms": 5.0,
                               "run_id": "run_1"})
    data.append_line_snapshot({"t_s": 1.0, "line_id": 0, "depth_m": 1.2,
                               "velocity_ms": 0.6, "wse_m": 11.2,
                               "bed_m": 10.0, "flow_cms": 6.0,
                               "run_id": "run_1"})
    result = load_timeseries_from_live(data, "run_1", 0)
    assert "t_s" in result
    assert len(result["t_s"]) == 2
    np.testing.assert_almost_equal(result["t_s"], [0.0, 1.0])


def test_render_timeseries_with_live_data():
    """Smoke test: render_timeseries accepts live data loader."""
    from swe2d.workbench.services.results_render_service import render_timeseries
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = SWE2DResultsData()
    data.append_line_snapshot({"t_s": 0.0, "line_id": 0, "depth_m": 1.0,
                               "velocity_ms": 0.5, "wse_m": 11.0,
                               "bed_m": 10.0, "flow_cms": 5.0, "run_id": "run_1"})
    from swe2d.results.run_service import RunRecord
    rec = RunRecord(run_id="run_1", gpkg_path="", color=(255,0,0),
                    enabled=True, has_profile=False, label="test")
    fig, ax = plt.subplots()
    try:
        render_timeseries(ax, [rec], 0, "flow_cms", "Flow", 0.0,
                          lambda r, lid, vk: load_timeseries_from_live(data, r.run_id, lid),
                          "m")
    finally:
        plt.close(fig)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_in_memory_results_render.py -v
```
Expected: FAIL — `load_timeseries_from_live` not defined.

- [ ] **Step 3: Add `load_timeseries_from_live` to queries.py**

In `swe2d/results/queries.py`, add:

```python
def load_timeseries_from_live(
    data: "SWE2DResultsData", run_id: str, line_id: int
) -> Dict[str, np.ndarray]:
    """Load time-series from in-memory snapshots during a live run.

    Returns the same dict shape as load_timeseries():
        ``t_s``, ``depth_m``, ``velocity_ms``, ``wse_m``, ``bed_m``, ``flow_cms``
    Each value is a 1-D float64 numpy array, sorted by *t_s*.

    Returns an empty dict if no matching data is found.
    """
    rows = data.get_live_line_snapshot_rows()
    if not rows or line_id < 0:
        return {}
    matched = [r for r in rows if int(r.get("line_id", -1)) == line_id
               and str(r.get("run_id", "")) == str(run_id)]
    if not matched:
        return {}
    matched.sort(key=lambda r: float(r.get("t_s", 0.0)))
    out: Dict[str, list] = {}
    keys = ["t_s", "depth_m", "velocity_ms", "wse_m", "bed_m", "flow_cms"]
    for k in keys:
        out[k] = []
    for r in matched:
        for k in keys:
            out[k].append(float(r.get(k, 0.0)))
    return {k: np.array(v, dtype=np.float64) for k, v in out.items()}


def load_profile_from_live(
    data: "SWE2DResultsData", run_id: str, line_id: int, t_sec: float
) -> Dict[str, np.ndarray]:
    """Load profile from in-memory snapshots during a live run.

    Returns the same dict shape as load_profile():
        ``dist_m``, ``wse_m``, ``bed_m``, ``depth_m``

    Returns an empty dict if no matching data is found.
    """
    rows = data.get_live_line_profile_rows()
    if not rows or line_id < 0:
        return {}
    matched = [r for r in rows if int(r.get("line_id", -1)) == line_id
               and str(r.get("run_id", "")) == str(run_id)]
    if not matched:
        return {}
    # Find closest timestep
    best = min(matched, key=lambda r: abs(float(r.get("t_s", 0.0)) - t_sec))
    raw = best.get("data", "")
    try:
        arr = np.frombuffer(raw.encode("latin1") if isinstance(raw, str) else raw,
                            dtype=np.float64).reshape(-1, 4)
        return {
            "dist_m": arr[:, 0],
            "wse_m":  arr[:, 1],
            "bed_m":  arr[:, 2],
            "depth_m": arr[:, 3],
        }
    except Exception:
        return {}
```

Import `SWE2DResultsData` at the top of queries.py (guarded to avoid circular import):
```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from swe2d.results.data import SWE2DResultsData
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_in_memory_results_render.py -v
```
Expected: PASS

- [ ] **Step 5: Modify `render_timeseries_on_figure` to use in-memory loader when live**

In `swe2d/workbench/services/results_render_service.py:387-419` (`render_timeseries_on_figure`), replace:

```python
    from swe2d.results.queries import load_timeseries as _load_ts
    ...
    render_timeseries(
        ...
        load_timeseries_fn=lambda rec, lid, vk: _load_ts(
            str(rec.gpkg_path), str(rec.run_id), int(lid)
        ),
        ...
    )
```

with:

```python
    from swe2d.results.queries import load_timeseries as _load_ts
    from swe2d.results.queries import load_timeseries_from_live as _load_ts_live
    data = result_data  # already passed as parameter
    is_live = data.data_source == "live"
    ...
    render_timeseries(
        ...
        load_timeseries_fn=lambda rec, lid, vk: (
            _load_ts_live(data, str(rec.run_id), int(lid))
            if is_live else
            _load_ts(str(rec.gpkg_path), str(rec.run_id), int(lid))
        ),
        ...
    )
```

- [ ] **Step 6: Same modification for `render_profile_on_figure`**

In `swe2d/workbench/services/results_render_service.py:421-449` (`render_profile_on_figure`), add the same live-data branching:

```python
    from swe2d.results.queries import load_profile as _load_prof
    from swe2d.results.queries import load_profile_from_live as _load_prof_live
    is_live = result_data.data_source == "live"
    ...
    load_profile_fn=lambda rec, lid, ts: (
        _load_prof_live(result_data, str(rec.run_id), int(lid), float(ts))
        if is_live else
        _load_prof(str(rec.gpkg_path), str(rec.run_id), int(lid), float(ts))
    ),
```

- [ ] **Step 7: Run full tests**

```bash
python -m pytest tests/test_in_memory_results_render.py tests/test_results_render_service.py -v
```
Expected: all PASS

- [ ] **Step 8: Commit**

```bash
git add swe2d/results/queries.py swe2d/workbench/services/results_render_service.py tests/test_in_memory_results_render.py
git commit -m "feat: add in-memory plot rendering path during live runs

- Add load_timeseries_from_live() and load_profile_from_live()
  to queries.py — read from SWE2DResultsData._live_* rows
- Modify render_timeseries_on_figure and render_profile_on_figure
  to use in-memory loader when data_source == 'live'
- Plots now work during live runs without GPKG persistence"
```

---

### Task 2: Remove snapshot GPKG writes during run

The snapshot button (`run_controller.py:1079`) writes to GPKG during the run. This should be removed — snapshots persist only at finalization.

**Files:**
- Modify: `swe2d/workbench/controllers/run_controller.py`
- Modify: `swe2d/workbench/studio_dialog.py`
- Delete or deprecate: `swe2d/workbench/studio_dialog.py:_persist_snapshot_to_gpkg` method

- [ ] **Step 1: Remove the `_persist_snapshot_to_gpkg` call from the snapshot button handler**

In `swe2d/workbench/controllers/run_controller.py:1075-1088`, change:

```python
        # GPKG persistence (mesh + coupling) — always attempt regardless of mesh data
        try:
            if gpkg_results_path:
                if _snapshots:
                    view._persist_snapshot_to_gpkg(gpkg_results_path, snap_run_id, accumulate=True)
                if _coupling_rows:
                    view._persist_coupling_results_to_geopackage(
                        gpkg_results_path, snap_run_id, _coupling_rows,
                        interval_s=0.0, accumulate=True,
                    )
                # Auto-load the snapshot result into the results panel
                view._auto_load_results_panel(gpkg_results_path, snap_run_id)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(view, "Snapshot", f"Snapshot failed:\n{exc}")
```

to:

```python
        # Auto-load the snapshot result into the results panel
        # (no GPKG write — snapshots stay in memory until run finalization)
        try:
            if gpkg_results_path:
                view._auto_load_results_panel(gpkg_results_path, snap_run_id)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(view, "Snapshot", f"Snapshot failed:\n{exc}")
```

- [ ] **Step 2: Mark `_persist_snapshot_to_gpkg` as deprecated**

In `swe2d/workbench/studio_dialog.py:1222`, add a deprecation note:

```python
    def _persist_snapshot_to_gpkg(self, gpkg_path: str, run_id: str, accumulate: bool = False) -> None:
        """[DEPRECATED] Snapshot persistence now deferred to run_finalizer.py."""
        logger.warning("_persist_snapshot_to_gpkg called but snapshots are in-memory only until finalization")
```

- [ ] **Step 3: Verify no other callers of `_persist_snapshot_to_gpkg`**

```bash
grep -rn "_persist_snapshot_to_gpkg" swe2d/ --include="*.py" | grep -v "def _persist"
```
Expected: only the deprecated method definition (caller removed in Step 1).

- [ ] **Step 4: Commit**

```bash
git add swe2d/workbench/controllers/run_controller.py swe2d/workbench/studio_dialog.py
git commit -m "refactor: remove snapshot GPKG writes during run

- Remove _persist_snapshot_to_gpkg call from manual snapshot handler
- Mark _persist_snapshot_to_gpkg as deprecated
- Snapshots now persist only at finalization via run_finalizer.py"
```

---

### Task 3: Verify max results are GPKG-only

Max results (per-cell max h, hu, hv across the entire simulation) should only be written to GPKG, never stored in memory.

- [ ] **Step 1: Verify max results are not stored in SWE2DResultsData**

```bash
grep -rn "max_results\|_live_max" swe2d/results/data.py
```
Expected: no results — `SWE2DResultsData` does not have max-results storage.

- [ ] **Step 2: Verify max results are written only to GPKG**

```bash
grep -n "persist_mesh_max_results_to_geopackage" swe2d/workbench/controllers/run_controller.py
```
This is at line 879-881 inside the `save_max_only` branch — called only at finalization via the run finalizer path. No in-memory max results path exists.

- [ ] **Step 3: Verify the finalization path persists max results to GPKG**

```bash
grep -n "persist_mesh_max_results_to_geopackage\|_persist_conservation_forensics\|max_results" swe2d/workbench/controllers/finalization_adapter.py
```

- [ ] **Step 4: Report findings**

If all checks pass, max results are already GPKG-only. No code changes needed.

If any in-memory max results storage is found, add a note to the plan.

---

### Task 4: Final verification

- [ ] **Step 1: Run the full test suite**

```bash
python -m pytest tests/ -v --timeout=120 2>&1 | tail -20
```

- [ ] **Step 2: Start a simulation run and verify:**

1. Overlay renders in real-time (reads from in-memory snapshots)
2. Time series plot shows data during the run (reads from in-memory line snapshots)
3. Profile plot shows data during the run (reads from in-memory profile snapshots)
4. No GPKG writes occur during the run (check file modification time)

- [ ] **Step 3: Finalize the run and verify:**

1. All results are persisted to GPKG
2. Overlay reads from GPKG (still works)
3. Time series and profile plots read from GPKG (still works)
4. Max results are present in GPKG

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat: in-memory results viewing during live runs

- Add in-memory plot rendering path (load_timeseries_from_live,
  load_profile_from_live)
- Remove snapshot GPKG writes during run (deferred to finalization)
- Max results remain GPKG-only (no change needed)
- All viewer operations read from memory during run, from GPKG
  after finalization"
```
