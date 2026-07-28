---
type: plan
status: complete
created: 2026-07-14
completed: 2026-07-25
---

# Drainage Link Profile Step 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist per-cell pipe geometry from the C++ readback and render the drainage-link profile with solid sloped invert/crown lines plus a velocity-colored water fill.

**Architecture:** Extend the pipe-cell live storage and GeoPackage persistence to carry `cell_invert`, `cell_width`, `cell_height`, and `cell_shape_type`, then update the profile viewer to draw solid invert/crown and segment the water-fill polygon by the selected fill metric using the existing colormap utilities.

**Tech Stack:** Python 3.12, PyQt5/PyQtGraph, NumPy, SQLite GeoPackage, SWE2D service/data layers.

---

## File Map

| File | Responsibility |
|---|---|
| `swe2d/results/data.py` | In-memory pipe-cell storage (`_live_pipe_cell`) and snapshot routing. |
| `swe2d/services/gpkg_persistence_service.py` | GeoPackage schema, persistence, and load for pipe-cell timeseries + geometry. |
| `swe2d/workbench/views/studio_viewer_profile_pg.py` | PyQtGraph profile viewer rendering for `drainage_link`. |
| `tests/test_pipe_cell_coupling_output.py` | Existing round-trip tests for pipe-cell persistence. |
| `tests/test_swe2d_gpu_drainage_network.py` | Existing drainage network tests; add geometry snapshot test. |

---

## Task 1: Store geometry fields in live pipe-cell storage

**Files:**
- Modify: `swe2d/results/data.py:172-184` (init_pipe_cell_storage)
- Modify: `swe2d/results/data.py:339-364` (append_pipe_cell_snapshot)

- [ ] **Step 1: Write the failing test**

Create a new test in `tests/test_swe2d_gpu_drainage_network.py` (or append to existing pipe-cell test):

```python
def test_append_pipe_cell_snapshot_stores_geometry():
    from swe2d.results.data import SWE2DResultsData
    rd = SWE2DResultsData()
    rd.init_pipe_cell_storage([("L1", 0, "depth")])
    rd.append_pipe_cell_snapshot({
        "link_id": "L1",
        "cell_sub_idx": 0,
        "metric": "depth",
        "t_s": 0.0,
        "value": 1.5,
        "cell_invert": 10.0,
        "cell_width": 2.0,
        "cell_height": 2.5,
        "cell_shape_type": 1,
    })
    d = rd._live_pipe_cell[("L1", 0, "depth")]
    assert d["cell_invert"] == 10.0
    assert d["cell_width"] == 2.0
    assert d["cell_height"] == 2.5
    assert d["cell_shape_type"] == 1
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
mamba run -n qgis_stable python3 -m pytest tests/test_swe2d_gpu_drainage_network.py::test_append_pipe_cell_snapshot_stores_geometry -v
```

Expected: FAIL with `KeyError: 'cell_height'` or assertion error on missing geometry fields.

- [ ] **Step 3: Initialize geometry fields in init_pipe_cell_storage**

Edit `swe2d/results/data.py:172-184` to include default geometry keys in each cell entry:

```python
def init_pipe_cell_storage(self, keys: List[Tuple[str, int, str]]) -> None:
    """Initialize empty pipe-cell storage for known (link_id, sub_idx, metric) keys."""
    self._live_pipe_cell.clear()
    for key in keys:
        link_id, cell_sub_idx, metric = key
        self._live_pipe_cell[key] = {
            "link_id": link_id,
            "cell_sub_idx": cell_sub_idx,
            "metric": metric,
            "times": [],
            "values": [],
            "cell_invert": 0.0,
            "cell_width": 1.0,
            "cell_height": 1.0,
            "cell_shape_type": 0,
        }
```

- [ ] **Step 4: Store all geometry fields on first snapshot write**

Edit `swe2d/results/data.py:358-364` in `append_pipe_cell_snapshot`:

```python
        # Store per-cell geometry on first write (same for all 4 metrics)
        if "cell_invert" not in d or d.get("cell_invert") is None:
            d["cell_invert"] = float(snapshot.get("cell_invert", 0.0))
            d["cell_width"] = float(snapshot.get("cell_width", 1.0))
            d["cell_height"] = float(snapshot.get("cell_height", cell_w))
            d["cell_shape_type"] = int(snapshot.get("cell_shape_type", 0))
```

- [ ] **Step 5: Run the test**

```bash
mamba run -n qgis_stable python3 -m pytest tests/test_swe2d_gpu_drainage_network.py::test_append_pipe_cell_snapshot_stores_geometry -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add swe2d/results/data.py tests/test_swe2d_gpu_drainage_network.py
git commit -m "feat: store full per-cell pipe geometry in live pipe-cell storage"
```

---

## Task 2: Persist and load pipe-cell geometry in the GeoPackage

**Files:**
- Modify: `swe2d/services/gpkg_persistence_service.py:684-692` (schema creation)
- Modify: `swe2d/services/gpkg_persistence_service.py:1036-1079` (persist)
- Modify: `swe2d/services/gpkg_persistence_service.py:1140-1185` (load)

- [ ] **Step 1: Write the failing round-trip test**

Add a test to `tests/test_pipe_cell_coupling_output.py`:

```python
def test_gpkg_pipe_cell_geometry_roundtrip(self):
    from swe2d.services.gpkg_persistence_service import (
        persist_baked_pipe_cell_ts, load_baked_pipe_cell_ts,
    )
    gpkg = os.path.join(self.temp_dir, "geo_roundtrip.gpkg")
    items = [
        {
            "link_id": "L1",
            "cell_sub_idx": 0,
            "metric": "depth",
            "times": np.array([0.0, 1.0], dtype=np.float64),
            "values": np.array([0.5, 0.7], dtype=np.float64),
            "cell_invert": 10.0,
            "cell_width": 2.0,
            "cell_height": 2.5,
            "cell_shape_type": 1,
        }
    ]
    persist_baked_pipe_cell_ts(gpkg, "run1", items, log_fn=None)
    conn = sqlite3.connect(gpkg)
    try:
        rows = list(load_baked_pipe_cell_ts(conn, "run1"))
    finally:
        conn.close()
    self.assertEqual(len(rows), 1)
    row = rows[0]
    self.assertEqual(row["cell_invert"], 10.0)
    self.assertEqual(row["cell_width"], 2.0)
    self.assertEqual(row["cell_height"], 2.5)
    self.assertEqual(row["cell_shape_type"], 1)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
mamba run -n qgis_stable python3 -m pytest tests/test_pipe_cell_coupling_output.py::TestPipeCellPersistence::test_gpkg_pipe_cell_geometry_roundtrip -v
```

Expected: FAIL because the schema and load function do not return geometry fields.

- [ ] **Step 3: Add geometry columns to the schema**

Edit `swe2d/services/gpkg_persistence_service.py:684-692` (inside `persist_baked_results`):

```python
                    CREATE TABLE IF NOT EXISTS swe2d_baked_pipe_cell_ts (
                        run_id TEXT,
                        link_id TEXT,
                        cell_sub_idx INTEGER,
                        metric TEXT,
                        n_timesteps INTEGER,
                        times_blob BLOB,
                        values_blob BLOB,
                        cell_invert REAL DEFAULT 0.0,
                        cell_width REAL DEFAULT 1.0,
                        cell_height REAL DEFAULT 1.0,
                        cell_shape_type INTEGER DEFAULT 0,
                        PRIMARY KEY (run_id, link_id, cell_sub_idx, metric))
```

Do the same for the standalone `persist_baked_pipe_cell_ts` schema at `swe2d/services/gpkg_persistence_service.py:1062-1070`.

- [ ] **Step 4: Persist geometry fields**

Edit the INSERT in `persist_baked_pipe_cell_ts` (around line 1076):

```python
                INSERT OR REPLACE INTO swe2d_baked_pipe_cell_ts
                    (run_id, link_id, cell_sub_idx, metric, n_timesteps,
                     times_blob, values_blob,
                     cell_invert, cell_width, cell_height, cell_shape_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
```

And pass the values from the item dict:

```python
                (
                    run_id, str(item["link_id"]), int(item["cell_sub_idx"]),
                    str(item["metric"]), int(len(item["times"])),
                    item["times"].astype(np.float64).tobytes(),
                    item["values"].astype(np.float64).tobytes(),
                    float(item.get("cell_invert", 0.0)),
                    float(item.get("cell_width", 1.0)),
                    float(item.get("cell_height", 1.0)),
                    int(item.get("cell_shape_type", 0)),
                )
```

Apply the same geometry handling in the `persist_baked_results` path at the first pipe-cell table creation (around line 698).

- [ ] **Step 5: Load geometry fields with backwards-compatible fallback**

Edit `load_baked_pipe_cell_ts` at `swe2d/services/gpkg_persistence_service.py:1140-1185`:

```python
    rows = conn.execute(
        "SELECT link_id, cell_sub_idx, metric, n_timesteps, times_blob, values_blob, "
        "cell_invert, cell_width, cell_height, cell_shape_type "
        "FROM swe2d_baked_pipe_cell_ts WHERE run_id=?",
        (run_id,),
    ).fetchall()
    out: List[Dict[str, Any]] = []
    for row in rows:
        link_id, cell_sub_idx, metric, n_timesteps, times_blob, values_blob = row[:6]
        geometry = row[6:]
        out.append({
            "link_id": str(link_id),
            "cell_sub_idx": int(cell_sub_idx),
            "metric": str(metric),
            "n_timesteps": int(n_timesteps),
            "times": (
                np.frombuffer(times_blob, dtype=np.float64).copy()
                if times_blob is not None else np.empty(0, dtype=np.float64)
            ),
            "values": (
                np.frombuffer(values_blob, dtype=np.float64).copy()
                if values_blob is not None else np.empty(0, dtype=np.float64)
            ),
            "cell_invert": float(geometry[0]) if len(geometry) > 0 and geometry[0] is not None else 0.0,
            "cell_width": float(geometry[1]) if len(geometry) > 1 and geometry[1] is not None else 1.0,
            "cell_height": float(geometry[2]) if len(geometry) > 2 and geometry[2] is not None else 1.0,
            "cell_shape_type": int(geometry[3]) if len(geometry) > 3 and geometry[3] is not None else 0,
        })
```

**Note:** If an old GeoPackage is missing the geometry columns, SQLite will raise an `OperationalError`. Wrap the query in a `try/except` and fall back to the legacy query without geometry columns; the geometry defaults above then apply.

- [ ] **Step 6: Run the round-trip test**

```bash
mamba run -n qgis_stable python3 -m pytest tests/test_pipe_cell_coupling_output.py::TestPipeCellPersistence::test_gpkg_pipe_cell_geometry_roundtrip -v
```

Expected: PASS.

- [ ] **Step 7: Run the existing pipe-cell tests**

```bash
mamba run -n qgis_stable python3 -m pytest tests/test_pipe_cell_coupling_output.py -v
```

Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add swe2d/services/gpkg_persistence_service.py tests/test_pipe_cell_coupling_output.py
git commit -m "feat: persist and load per-cell pipe geometry in GeoPackage"
```

---

## Task 3: Build pipe-cell items with geometry for finalization

**Files:**
- Modify: `swe2d/results/data.py:566-583` (build_pipe_cell_items)

- [ ] **Step 1: Verify build_pipe_cell_items includes geometry**

Edit `swe2d/results/data.py:566-583` to ensure each item dict includes:

```python
    def build_pipe_cell_items(self) -> List[Dict]:
        """Reconstruct list-of-dicts from _live_pipe_cell for persistence."""
        out: List[Dict] = []
        for (link_id, cell_sub_idx, metric), d in self._live_pipe_cell.items():
            out.append({
                "link_id": link_id,
                "cell_sub_idx": cell_sub_idx,
                "metric": metric,
                "times": np.asarray(d["times"], dtype=np.float64),
                "values": np.asarray(d["values"], dtype=np.float64),
                "cell_invert": float(d.get("cell_invert", 0.0)),
                "cell_width": float(d.get("cell_width", 1.0)),
                "cell_height": float(d.get("cell_height", 1.0)),
                "cell_shape_type": int(d.get("cell_shape_type", 0)),
            })
        return out
```

- [ ] **Step 2: Add a test**

Add to `tests/test_pipe_cell_coupling_output.py` or `tests/test_swe2d_gpu_drainage_network.py`:

```python
def test_build_pipe_cell_items_includes_geometry():
    from swe2d.results.data import SWE2DResultsData
    rd = SWE2DResultsData()
    rd.init_pipe_cell_storage([("L1", 0, "depth")])
    rd.append_pipe_cell_snapshot({
        "link_id": "L1",
        "cell_sub_idx": 0,
        "metric": "depth",
        "t_s": 0.0,
        "value": 1.0,
        "cell_invert": 5.0,
        "cell_width": 1.0,
        "cell_height": 1.5,
        "cell_shape_type": 1,
    })
    items = rd.build_pipe_cell_items()
    assert len(items) == 1
    assert items[0]["cell_invert"] == 5.0
    assert items[0]["cell_height"] == 1.5
    assert items[0]["cell_shape_type"] == 1
```

- [ ] **Step 3: Run the test**

```bash
mamba run -n qgis_stable python3 -m pytest tests/test_pipe_cell_coupling_output.py tests/test_swe2d_gpu_drainage_network.py -k "build_pipe_cell_items" -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add swe2d/results/data.py tests/test_swe2d_gpu_drainage_network.py
git commit -m "feat: include geometry in pipe-cell items for finalization"
```

---

## Task 4: Render drainage-link profile with solid sloped lines and velocity fill

**Files:**
- Modify: `swe2d/workbench/views/studio_viewer_profile_pg.py:708-804`
- Modify: `swe2d/workbench/views/studio_viewer_profile_pg.py:923-928` and related controls to enable Fill/Cmap for drainage_link mode

- [ ] **Step 1: Inspect the current drainage_link rendering block**

Read `swe2d/workbench/views/studio_viewer_profile_pg.py:708-804` to confirm the existing code before editing.

- [ ] **Step 2: Enable Fill/Cmap controls for drainage_link**

Edit the existing `_show_profile_controls` method at `swe2d/workbench/views/studio_viewer_profile_pg.py:1111-1129`:

```python
    def _show_profile_controls(self) -> None:
        """Show/hide profile-specific controls based on element type."""
        etype = str(self._etype_combo.currentData() or "line")
        is_profile = etype in ("line", "drainage_link")
        if self._fill_combo is not None:
            self._fill_combo.setVisible(is_profile)
        if self._cmap_combo is not None:
            self._cmap_combo.setVisible(is_profile)
        if self._show_struct_chk is not None:
            self._show_struct_chk.setVisible(is_profile)
        # Update axis labels for time-series mode
        if self._plot_widget is not None:
            u = _unit_labels()
            if is_profile:
                self._plot_widget.setLabel("bottom", f"Station ({u['len']})")
                self._plot_widget.setLabel("left", f"Elevation ({u['len']})")
            else:
                self._plot_widget.setLabel("bottom", f"Time ({_TIME_UNIT})")
                self._plot_widget.setLabel("left", "Value")
```

- [ ] **Step 3: Replace the drainage_link rendering block with geometry-aware, velocity-shaded fill**

Replace the block from `x_stations = np.linspace(...)` to `self._plot_widget.plotItem.autoRange()` (lines 760-804) with:

```python
            # Station axis: use actual per-cell station if available, else linear spacing
            x_stations = np.linspace(0.0, float(length_m), n_sub)

            # Read depth and fill metric at current time for each sub-cell
            fill_key = self._prof_fill_key
            cmap_name = self._prof_cmap
            use_fill_cmap = fill_key != "none" and fill_key is not None

            # Map line-profile fill names to pipe-cell metric names
            _DRAINAGE_FILL_MAP = {
                "velocity_ms": "velocity",
                "depth_m": "depth",
                "flow_qn": "flow",
            }
            drainage_fill_metric = _DRAINAGE_FILL_MAP.get(fill_key, fill_key)

            water_y = invert_y.copy()
            fill_y = np.full(n_sub, np.nan, dtype=np.float64)
            for k in sub_keys:
                depth_d = pipe_cell.get(k, {})
                t_arr = depth_d.get("times", [])
                v_arr = depth_d.get("values", [])
                if not t_arr or not v_arr:
                    continue
                t_np = np.asarray(t_arr, dtype=np.float64)
                v_np = np.asarray(v_arr, dtype=np.float64)
                i_nearest = int(np.argmin(np.abs(t_np - t_sec)))
                sub_idx = int(k[1]) if len(k) >= 2 else 0
                if 0 <= sub_idx < n_sub:
                    water_y[sub_idx] = invert_y[sub_idx] + float(v_np[i_nearest])

            if use_fill_cmap and drainage_fill_metric not in ("", "none"):
                for k in pipe_cell.keys():
                    if len(k) >= 2 and str(k[0]) == str(link_id) and k[2] == drainage_fill_metric:
                        fill_d = pipe_cell.get(k, {})
                        t_arr = fill_d.get("times", [])
                        v_arr = fill_d.get("values", [])
                        if not t_arr or not v_arr:
                            continue
                        t_np = np.asarray(t_arr, dtype=np.float64)
                        v_np = np.asarray(v_arr, dtype=np.float64)
                        i_nearest = int(np.argmin(np.abs(t_np - t_sec)))
                        sub_idx = int(k[1]) if len(k) >= 2 else 0
                        if 0 <= sub_idx < n_sub:
                            fill_y[sub_idx] = float(v_np[i_nearest])

            # Plot invert line (solid brown)
            invert_pen = pg.mkPen(color=QtGui.QColor(92, 64, 51), width=1.2)
            invert_plot = pg.PlotDataItem(x_stations, invert_y, pen=invert_pen, name="Invert")
            self._plot_widget.addItem(invert_plot)
            self._plot_items.append(invert_plot)

            # Plot crown line (solid dark grey)
            crown_pen = pg.mkPen(color=QtGui.QColor(64, 64, 64), width=1.2)
            crown_plot = pg.PlotDataItem(x_stations, crown_y, pen=crown_pen, name="Crown")
            self._plot_widget.addItem(crown_plot)
            self._plot_items.append(crown_plot)

            if use_fill_cmap and np.any(np.isfinite(fill_y)):
                # Segment-by-segment fill colored by the fill metric
                fill_mask = np.isfinite(water_y) & np.isfinite(invert_y) & np.isfinite(fill_y)
                seg_vals, seg_list = [], []
                for i in range(n_sub - 1):
                    if not (fill_mask[i] and fill_mask[i + 1]):
                        continue
                    vmid = 0.5 * (float(fill_y[i]) + float(fill_y[i + 1]))
                    seg_list.append(i)
                    seg_vals.append(vmid)
                if seg_vals:
                    sv = np.asarray(seg_vals, dtype=np.float64)
                    sv_min, sv_max = float(np.nanmin(sv)), float(np.nanmax(sv))
                    if sv_max <= sv_min:
                        sv_max = sv_min + 1.0
                    for idx, i in enumerate(seg_list):
                        vmid = seg_vals[idx]
                        t_norm = (vmid - sv_min) / (sv_max - sv_min)
                        rgb = _cmap_color(cmap_name, float(np.clip(t_norm, 0.0, 1.0)))
                        seg_bed = pg.PlotDataItem(
                            [float(x_stations[i]), float(x_stations[i + 1])],
                            [float(invert_y[i]), float(invert_y[i + 1])],
                        )
                        seg_wse = pg.PlotDataItem(
                            [float(x_stations[i]), float(x_stations[i + 1])],
                            [float(water_y[i]), float(water_y[i + 1])],
                        )
                        seg_fill = pg.FillBetweenItem(
                            curve1=seg_bed,
                            curve2=seg_wse,
                            brush=pg.mkBrush(QtGui.QColor(*rgb)),
                        )
                        self._plot_widget.addItem(seg_fill)
                        self._fill_items.append(seg_fill)
            else:
                # Uniform water fill when no metric selected
                water_plot = pg.PlotDataItem(x_stations, water_y)
                fill_item = pg.FillBetweenItem(
                    curve1=invert_plot,
                    curve2=water_plot,
                    brush=pg.mkBrush(QtGui.QColor(100, 149, 237, 96)),
                )
                self._plot_widget.addItem(fill_item)
                self._fill_items.append(fill_item)

            self._plot_widget.plotItem.autoRange()
```

- [ ] **Step 4: Set default fill for drainage_link to velocity**

In `_on_etype_changed`, when switching to `drainage_link`, set `_prof_fill_key` to `"velocity_ms"` if not already set:

```python
            if etype == "drainage_link":
                self._prof_fill_key = "velocity_ms"
                idx = self._fill_combo.findData("velocity_ms")
                if idx >= 0:
                    self._fill_combo.setCurrentIndex(idx)
```

- [ ] **Step 5: Run a viewer-focused test or ad-hoc script**

Since the viewer requires pyqtgraph, run an ad-hoc script that builds mock `_live_pipe_cell` and calls the rendering helper. If the render code is too embedded, create a small unit test for the data extraction part:

```python
def test_drainage_link_profile_data_extraction():
    from swe2d.results.data import SWE2DResultsData
    rd = SWE2DResultsData()
    rd.init_pipe_cell_storage([("L1", 0, "depth"), ("L1", 0, "velocity")])
    rd.append_pipe_cell_snapshot({
        "link_id": "L1", "cell_sub_idx": 0, "metric": "depth",
        "t_s": 0.0, "value": 1.0,
        "cell_invert": 10.0, "cell_width": 2.0, "cell_height": 2.0, "cell_shape_type": 0,
    })
    rd.append_pipe_cell_snapshot({
        "link_id": "L1", "cell_sub_idx": 0, "metric": "velocity",
        "t_s": 0.0, "value": 3.0,
    })
    # Verify geometry is stored
    d = rd._live_pipe_cell[("L1", 0, "depth")]
    assert d["cell_invert"] == 10.0
    assert d["cell_width"] == 2.0
```

- [ ] **Step 6: Run the workbench GUI import tests**

```bash
mamba run -n qgis_stable python3 -m pytest tests.test_workbench_imports -v
```

Expected: PASS (or same pre-existing failures unrelated to this change).

- [ ] **Step 7: Commit**

```bash
git add swe2d/workbench/views/studio_viewer_profile_pg.py
git commit -m "feat: render drainage-link profile with solid geometry and velocity fill"
```

---

## Task 5: Verify end-to-end with a real drainage run

**Files:**
- None (manual verification)

- [ ] **Step 1: Purge Python caches**

```bash
find . -type d -name __pycache__ -exec rm -rf {} +
```

- [ ] **Step 2: Run the drainage network test**

```bash
mamba run -n qgis_stable python3 -m pytest tests/test_swe2d_gpu_drainage_network.py -v -k "pipe_cell" --timeout=300
```

Expected: all new and existing pipe-cell tests PASS.

- [ ] **Step 3: Manual GUI check**

Open the Studio dialog, run a drainage simulation, open the Profile viewer, select Type = "Drainage Link", and confirm:
- Invert and crown are solid lines.
- Invert/crown slope along the link.
- The water-fill area is colored by velocity (default Fill = velocity).
- Changing the Colormap combo updates the shading.

- [ ] **Step 4: Commit any final tweaks**

```bash
git add -A
git commit -m "fix: final drainage-link profile verification tweaks"
```

---

## Self-Review Checklist

- [ ] Spec coverage: geometry persistence, solid lines, velocity fill, viewer controls all have tasks.
- [ ] Placeholder scan: no TBD/TODO/fill-in-later in the plan above.
- [ ] Type consistency: `cell_shape_type` is `int`; `cell_invert/cell_width/cell_height` are `float` everywhere.
- [ ] Backwards compatibility: `load_baked_pipe_cell_ts` handles old tables without geometry columns.

## Execution Choice

After this plan is saved, choose execution:
1. **Subagent-Driven** — fresh subagent per task (recommended).
2. **Inline Execution** — execute in this session with checkpoints.

**Decision required from user.**
