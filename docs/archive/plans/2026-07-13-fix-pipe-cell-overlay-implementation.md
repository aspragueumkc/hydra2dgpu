---
type: plan
status: complete
created: 2026-07-13
completed: 2026-07-25
---

# Fix Pipe-Cell + Overlay Field Implementation (with SI/USC Rain Units)

> **For agentic workers:** Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan phase-by-phase. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the broken pipe-cell + overlay-field implementation so it matches the original plan, passes tests, and honors the unit-handling convention: rainfall overlays display in **mm/hr (SI)** or **in/hr (USC)**; cumulative rain/excess/loss display in **mm (SI)** or **in (USC)**; variables have **no unit suffixes**.

**Architecture:** Keep the model/solver units unchanged (rainfall is still tracked in mm and rain rate in m/s internally). Apply unit-aware display conversions at the last moment in the overlay renderer and profile viewer. Wire Manning/CN from the 2D mesh spatial arrays, not drainage links. Fix the single broken attribute reference (`cc._dsoa` → `cc._drainage_soa`) that is silently disabling pipe-cell storage and overlay array wiring.

**Tech Stack:** Python 3.12+, C++20, CUDA, PyQt5, pybind11, NumPy, GeoPackage

---

## Context

The feature was originally planned in `docs/archive/plans/2026-07-13-drainage-pipe-cell-output-and-overlay-fields.md`. The audit in `docs/AUDIT_PIPE_CELL_OVERLAY_GAP.md` found the implementation is scaffolded but broken:

- `cc._dsoa` is referenced but the controller stores the SoA as `self._drainage_soa` → no pipe-cell data stored or persisted.
- Rain intensity conversion is `m/s * 3600` (m/hr) but label says mm/hr → 1000× too small.
- Manning/CN overlay arrays are copied from per-link `link_roughness_n` instead of 2D mesh cell arrays.
- `step_net_rainfall_mps(t_s, t_s, ...)` uses a zero-width interval → rate is zero.
- Pipe depth uses `cell_A / cell_width` instead of shape-aware geometry.
- Overlay GPKG only stores last snapshot, not full time series.
- No GPKG read-back path for either feature.
- Variables carry unit suffixes (`_mps`, `_mm`) which violate the unit-handling convention.

---

## Unit-Handling Spec (from this session)

| Quantity | Internal variable | Internal unit | SI display | USC display |
|----------|-------------------|---------------|------------|-------------|
| Rainfall rate | `overlay_cell_rainfall_rate` | m/s | mm/hr | in/hr |
| Cumulative rain | `overlay_cell_cumulative_rain` | mm | mm | in |
| Cumulative excess | `overlay_cell_cumulative_excess` | mm | mm | in |
| Cumulative loss | `overlay_cell_cumulative_loss` | mm | mm | in |
| Manning's n | `overlay_cell_mannings_n` | — | — | — |
| Curve Number | `overlay_cell_curve_number` | — | — | — |

Conversion factors (used at display time only):
- `m/s → mm/hr`: `× 3_600_000` (1000 mm/m × 3600 s/hr)
- `m/s → in/hr`: `× 3600 × 1000 / 25.4` (3600 s/hr, 1000 mm/m, 25.4 mm/in)
- `mm → in`: `÷ 25.4`

---

## Remediation Phases

### Phase 0: Rename Overlay Variables (No Unit Suffixes)

Update `swe2d/results/data.py`, `clear_live_snapshots()`, and all references across the codebase.

| Old name | New name |
|----------|----------|
| `overlay_cell_rain_rate_mps` | `overlay_cell_rainfall_rate` |
| `overlay_cell_cum_rain_mm` | `overlay_cell_cumulative_rain` |
| `overlay_cell_cum_excess_mm` | `overlay_cell_cumulative_excess` |
| `overlay_cell_cum_loss_mm` (computed in renderer) | `overlay_cell_cumulative_loss` (computed earlier, stored) |
| `overlay_cell_manning_n` | `overlay_cell_mannings_n` |
| `overlay_cell_cn` | `overlay_cell_curve_number` |

- [ ] **Task 0.1:** Rename in `swe2d/results/data.py:79-84`

- [ ] **Task 0.2:** Rename in `swe2d/results/data.py:clear_live_snapshots()` (~lines 136-140)

- [ ] **Task 0.3:** Rename in `swe2d/workbench/services/overlay_parameters_service.py:223-229`

- [ ] **Task 0.4:** Rename in `swe2d/workbench/services/non_gui_runtime_service.py:_copy_overlay_cell_data_from_coupling()` (~lines 170-220)

- [ ] **Task 0.5:** Rename in `swe2d/results/high_perf_viewer.py:618-664` and `:966-977`

- [ ] **Task 0.6:** Rename in `swe2d/workbench/views/results_controls.py` keys (currently `cum_rain`/`cum_excess`/`cum_loss`; align with new no-suffix convention: `cumulative_rain`, `cumulative_excess`, `cumulative_loss`). Keep display labels unit-aware in the controller, not hardcoded here.

- [ ] **Task 0.7:** Update tests that reference old names: `tests/test_overlay_rain_fields.py`, `tests/test_high_perf_viewer.py`

- [ ] **Task 0.8:** Run test subset to verify renames compile

  ```bash
  find . -type d -name __pycache__ -exec rm -rf {} +
  mamba run -n qgis_stable python3 -m unittest -v \
    tests.test_high_perf_viewer \
    tests.test_overlay_rain_fields
  ```

---

### Phase 1: Fix Critical Runtime Bugs (P0)

- [ ] **Task 1.1:** Fix the broken `_dsoa` reference in `swe2d/workbench/services/non_gui_runtime_service.py`

  Two occurrences:
  - Line 208: `dsoa = getattr(cc, "_dsoa", None)` → `dsoa = getattr(cc, "_drainage_soa", None)`
  - Line 550: `dsoa = getattr(cc, "_dsoa", None)` → `dsoa = getattr(cc, "_drainage_soa", None)`

  ```python
  # Before
  dsoa = getattr(cc, "_dsoa", None)
  # After
  dsoa = getattr(cc, "_drainage_soa", None)
  ```

- [ ] **Task 1.2:** Fix `if not link_lengths:` on NumPy array in `build_pipe_cell_keys()`

  ```python
  # Before
  if not link_lengths or not links:
      return keys
  # After
  if len(link_lengths) == 0 or not links:
      return keys
  ```

- [ ] **Task 1.3:** Fix zero-width rain interval in `_copy_overlay_cell_data_from_coupling()`

  The caller `_sample_coupling_object_metrics` receives `t_s` and `_h_s` (current water depth array, not time step). Add a `dt` parameter or use the last known step length.

  **Recommended approach:** Change `_sample_coupling_object_metrics` signature to accept `dt`:

  ```python
  # In non_gui_runtime_service.py
  def _sample_coupling_object_metrics(
      cc, t_s: float, dt: float, _h_s, _results_data=None
  ) -> list:
      ...
      if _results_data is not None:
          _copy_overlay_cell_data_from_coupling(cc, _results_data, t_s, dt)
      ...
  ```

  Then update the call to `step_net_rainfall_mps(t_s, t_s + dt, ...)`.

  Update all callers of `_sample_coupling_object_metrics` to pass `dt` (search with `grep`).

- [ ] **Task 1.4:** Fix rain intensity display conversion in `swe2d/results/high_perf_viewer.py:618-622`

  ```python
  elif mode == "rain_intensity":
      if overlay_cell_rainfall_rate is not None and overlay_cell_rainfall_rate.size == n:
          # rain_rate is in m/s (SI). Convert to display units.
          if length_unit_name == "ft":
              # m/s -> in/hr: * 3600 s/hr * 1000 mm/m / 25.4 mm/in
              vals = overlay_cell_rainfall_rate * 3600.0 * 1000.0 / 25.4
          else:
              # m/s -> mm/hr: * 3600 s/hr * 1000 mm/m
              vals = overlay_cell_rainfall_rate * 3_600_000.0
      else:
          vals = np.zeros(n, dtype=np.float64)
      vals[~wet_all] = np.nan
  ```

- [ ] **Task 1.5:** Fix cumulative rain/excess/loss display conversion in `high_perf_viewer.py`

  Cumulative values are stored in mm. Convert to inches only in USC.

  ```python
  elif mode == "cumulative_rain":
      if overlay_cell_cumulative_rain is not None and overlay_cell_cumulative_rain.size == n:
          vals = overlay_cell_cumulative_rain.copy()
          if length_unit_name == "ft":
              vals = vals / 25.4  # mm -> in
      else:
          vals = np.zeros(n, dtype=np.float64)
      vals[~wet_all] = np.nan
  ```

  Same for `cumulative_excess` and `cumulative_loss`.

- [ ] **Task 1.6:** Run tests

  ```bash
  mamba run -n qgis_stable python3 -m unittest -v \
    tests.test_high_perf_viewer \
    tests.test_overlay_rain_fields
  ```

---

### Phase 2: Wire Manning/CN from 2D Mesh (Not Drainage Links)

- [ ] **Task 2.1:** Identify where per-cell Manning/CN arrays are built

  Search for `build_spatial_manning_array_qgis` and `build_spatial_cn_array_qgis` in `swe2d/boundary_and_forcing/spatial_forcing_qgis_adapter.py` and any mesh metadata storage.

- [ ] **Task 2.2:** Store arrays on `results_data` at mesh-build time

  In `swe2d/workbench/controllers/overlay_controller.py` (or wherever the mesh is initialized for rendering), after the mesh/scalar arrays are built, store:

  ```python
  data.overlay_cell_mannings_n = np.asarray(manning_array, dtype=np.float64)
  data.overlay_cell_curve_number = np.asarray(cn_array, dtype=np.float64)
  ```

  If the arrays are only available on the GPU, provide a host readback or cache them from the initial build.

- [ ] **Task 2.3:** Remove the wrong fallback in `_copy_overlay_cell_data_from_coupling()`

  After Task 2.2, the overlay arrays should already be set. `_copy_overlay_cell_data_from_coupling` should only update the rain-related dynamic arrays. Remove the Manning/CN copying from `dsoa.link_roughness_n` and forcing `curve_number`.

- [ ] **Task 2.4:** Add regression test: `test_overlay_cell_manning_cn_from_mesh` asserts arrays are `(n_cells,)` and not equal to link roughness.

- [ ] **Task 2.5:** Run tests

  ```bash
  mamba run -n qgis_stable python3 -m unittest -v \
    tests.test_high_perf_viewer \
    tests.test_overlay_rain_fields
  ```

---

### Phase 3: Shape-Aware Pipe Depth + Profile Fix

- [ ] **Task 3.1:** Implement shape-aware depth in `swe2d/runtime/coupling.py:readback_coupling_state()`

  The C++ readback already returns `cell_shape_type`, `cell_width`, `cell_height`. Replace:

  ```python
  # Before
  out["cell_depth"] = cell_A / np.maximum(cell_width, 1e-9)
  # After
  cell_depth = np.empty(n_pipe_cells, dtype=np.float64)
  shape_type = np.asarray(cell_shape_type, dtype=np.int32)
  w = np.asarray(cell_width, dtype=np.float64)
  h = np.asarray(cell_height, dtype=np.float64)
  A = np.asarray(cell_A, dtype=np.float64)
  # Circular (0): A = (D²/4)*acos((D-2d)/D) - ((D-2d)/2)*sqrt(D*d - d²)
  circular = shape_type == 0
  if np.any(circular):
      D = w[circular]
      a = A[circular]
      # Use Newton or table lookup for d from A; for now use approximate d ≈ A/D is not correct.
      # ... insert correct circular depth-from-area formula or reuse pipe1d_lookup_geometry
      # ...
  # Rectangular (1): A = w * d  => d = A / w
  rect = shape_type == 1
  cell_depth[rect] = A[rect] / np.maximum(w[rect], 1e-12)
  # Elliptical (2): approximate or table lookup
  ...
  out["cell_depth"] = cell_depth
  ```

  If a Python helper already exists (e.g., `pipe1d_lookup_geometry`), use it instead of reimplementing.

- [ ] **Task 3.2:** Verify `cell_head = cell_invert + cell_depth` still holds

- [ ] **Task 3.3:** Add `cell_shape_type`, `cell_width`, `cell_height` to `readback_coupling_state()` return

  Needed for profile viewer to draw crown correctly.

- [ ] **Task 3.4:** Update `studio_viewer_profile_pg.py` crown line to use shape-aware geometry

  Circular: `crown = invert + diameter` (where `diameter = cell_width`)
  Rectangular: `crown = invert + height` (where `height = cell_height`)

- [ ] **Task 3.5:** Emit `_live_pipe_cell` in `simulation_worker.py:_on_snapshot_readback()`

  Currently only emits `_live_coupling`; add `_live_pipe_cell` so snapshot consumers see pipe-cell data.

- [ ] **Task 3.6:** Run pipe-specific tests

  ```bash
  mamba run -n qgis_stable python3 -m unittest -v \
    tests.test_coupling_integration \
    tests.test_pipe_cell_coupling_output
  ```

---

### Phase 4: GPKG Storage + Read-Back (Full Time Series)

#### Pipe-cell GPKG

- [ ] **Task 4.1:** Add `load_baked_pipe_cell_ts()` in `swe2d/services/gpkg_persistence_service.py`

  ```python
  def load_baked_pipe_cell_ts(
      conn: sqlite3.Connection,
      run_id: str,
  ) -> List[Dict[str, Any]]:
      rows = conn.execute(
          "SELECT link_id, cell_sub_idx, metric, n_timesteps, times_blob, values_blob "
          "FROM swe2d_baked_pipe_cell_ts WHERE run_id = ?",
          (run_id,),
      ).fetchall()
      items = []
      for link_id, cell_sub_idx, metric, n_ts, times_blob, values_blob in rows:
          items.append({
              "link_id": link_id,
              "cell_sub_idx": cell_sub_idx,
              "metric": metric,
              "n_timesteps": n_ts,
              "times": np.frombuffer(times_blob, dtype=np.float64),
              "values": np.frombuffer(values_blob, dtype=np.float64),
          })
      return items
  ```

- [ ] **Task 4.2:** Wire `load_baked_pipe_cell_ts()` into `swe2d/results/data.py` when loading a baked GPKG

  In `_load_coupling_for_first_enabled_run()` or a new loader, call `load_baked_pipe_cell_ts` and populate `self._live_pipe_cell`.

#### Overlay-field GPKG

- [ ] **Task 4.3:** Add `load_baked_overlay_fields()` in `gpkg_persistence_service.py`

  ```python
  def load_baked_overlay_fields(
      conn: sqlite3.Connection,
      run_id: str,
      n_cells: int,
  ) -> Dict[str, np.ndarray]:
      rows = conn.execute(
          "SELECT metric, n_timesteps, times_blob, values_blob "
          "FROM swe2d_baked_overlay_fields WHERE run_id = ?",
          (run_id,),
      ).fetchall()
      result = {}
      for metric, n_ts, times_blob, values_blob in rows:
          arr = np.frombuffer(values_blob, dtype=np.float64)
          if arr.size == n_ts * n_cells:
              result[metric] = arr.reshape(n_ts, n_cells)
          elif arr.size == n_cells:
              result[metric] = arr  # legacy single-snapshot fallback
      return result
  ```

- [ ] **Task 4.4:** Fix `build_overlay_field_items()` to store full time series

  In `swe2d/results/data.py`, accumulate per-timestep arrays into a list during the run, then build the flattened `(n_timesteps, n_cells)` row-major array at finalization.

  ```python
  # In data.py, add during __init__ or reset:
  self._overlay_field_history: Dict[str, List[np.ndarray]] = {}
  self._overlay_field_times: List[float] = []

  # In _copy_overlay_cell_data_from_coupling or append method:
  self._overlay_field_times.append(t_s)
  self._overlay_field_history.setdefault("rainfall_rate", []).append(
      np.asarray(self.overlay_cell_rainfall_rate)
  )
  ... same for cumulative_rain, cumulative_excess, cumulative_loss ...

  # In build_overlay_field_items():
  items = []
  for metric, hist in self._overlay_field_history.items():
      if not hist:
          continue
      arr = np.stack(hist, axis=0)  # (n_timesteps, n_cells)
      items.append({
          "metric": metric,
          "times": np.asarray(self._overlay_field_times, dtype=np.float64),
          "values": arr,
      })
  return items
  ```

- [ ] **Task 4.5:** Wire `load_baked_overlay_fields()` into `data.py` GPKG loading path

  Populate `overlay_cell_rainfall_rate`, `overlay_cell_cumulative_rain`, etc. with the latest snapshot when a GPKG is loaded, or support per-timestep playback if the renderer requests it.

- [ ] **Task 4.6:** Run GPKG tests

  ```bash
  mamba run -n qgis_stable python3 -m unittest -v \
    tests.test_gpkg_persistence \
    tests.test_pipe_cell_coupling_output
  ```

---

### Phase 5: Display Labels Are Unit-Aware

- [ ] **Task 5.1:** Move hardcoded labels out of `results_controls.py`

  The combo currently stores:
  ```python
  ("Rain Intensity (mm/hr)", "rain_intensity")
  ("Cumulative Rain (mm)", "cum_rain")
  ```

  Change to no-unit keys and generic display text, then update the label dynamically based on `length_unit_name` in the controller:
  ```python
  ("Rain Intensity", "rain_intensity")
  ("Cumulative Rain", "cumulative_rain")
  ```

- [ ] **Task 5.2:** Add unit-aware label helper in `swe2d/workbench/services/overlay_parameters_service.py`

  ```python
  def _label_for_overlay_mode(mode: str, length_unit_name: str) -> str:
      if mode == "rain_intensity":
          return "Rain Intensity (in/hr)" if length_unit_name == "ft" else "Rain Intensity (mm/hr)"
      if mode == "cumulative_rain":
          return "Cumulative Rain (in)" if length_unit_name == "ft" else "Cumulative Rain (mm)"
      if mode == "cumulative_excess":
          return "Cumulative Excess (in)" if length_unit_name == "ft" else "Cumulative Excess (mm)"
      if mode == "cumulative_loss":
          return "Cumulative Loss (in)" if length_unit_name == "ft" else "Cumulative Loss (mm)"
      ...
  ```

  Use it in `overlay_parameters_service.py` to set `legend_label`, and in `high_perf_viewer.py` fallback.

- [ ] **Task 5.3:** Add unit-aware labels to `studio_viewer_profile_pg.py` pipe-cell branch

  The station axis is in model units; label should be `"Station (m)"` or `"Station (ft)"` depending on `length_unit_name`.

- [ ] **Task 5.4:** Run test subset

  ```bash
  mamba run -n qgis_stable python3 -m unittest -v \
    tests.test_high_perf_viewer \
    tests.test_workbench_gui
  ```

---

### Phase 6: Tests

- [ ] **Task 6.1:** Fix broken test in `tests/test_swe2d_gpu_drainage_network.py:1073-1115`

  The test constructs `SWE2DCouplingController(cfg=MagicMock())` but `__init__` does not accept `cfg`. Either mock the required arguments or use the controller's actual construction path.

- [ ] **Task 6.2:** Add missing `test_pipe_cell_velocity_depth_head_flow_persisted` to `tests/test_gpkg_coupling_roundtrip.py`

  Run a short drainage simulation and assert `swe2d_baked_pipe_cell_ts` rows exist with correct shape.

- [ ] **Task 6.3:** Add missing `tests/test_overlay_rain_fields_gpkg.py`

  End-to-end: run a simulation with rain, persist, reload GPKG, assert `swe2d_baked_overlay_fields` rows decode and display conversion is correct.

- [ ] **Task 6.4:** Add unit-display tests

  - `test_rain_intensity_si_mm_per_hr` and `test_rain_intensity_usc_in_per_hr`
  - `test_cumulative_rain_usc_inches` and `test_cumulative_rain_si_mm`
  - `test_label_is_unit_aware`

- [ ] **Task 6.5:** Add pipe-cell shape-aware depth test

  - Circular pipe: given a known area and diameter, assert depth matches expected.
  - Rectangular pipe: given area and width, assert depth = A / width.

- [ ] **Task 6.6:** Run full test suite

  ```bash
  find . -type d -name __pycache__ -exec rm -rf {} +
  mamba run -n qgis_stable python3 -m unittest -v
  ```

---

### Phase 7: C++ Build + Verification Gate

- [ ] **Task 7.1:** Build C++/CUDA

  ```bash
  cd /home/aaron/QGIS_Plugins_dev/private-repo-hydra2dgpu/build
  cmake .. -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_BUILD_TYPE=Release
  cmake --build . -j$(nproc)
  ```

- [ ] **Task 7.2:** Architecture enforcement checks

  ```bash
  # No Qt in service layer
  ! grep -q 'from qgis\|from PyQt\|\.setEnabled\|\.setText\|\.setValue' \
      swe2d/runtime/ swe2d/boundary_and_forcing/ swe2d/workbench/services/*service*.py \
      && echo "PASS"
  ```

- [ ] **Task 7.3:** Final smoke test: run a short drainage + rain simulation and verify:
  - Overlay picker shows rain/Manning/CN fields
  - Rain intensity display value matches expected SI/USC unit
  - Pipe profile viewer draws a non-empty longitudinal profile
  - GPKG contains `swe2d_baked_pipe_cell_ts` and `swe2d_baked_overlay_fields` rows

---

## Verification Gate (after every phase)

```bash
find . -type d -name __pycache__ -exec rm -rf {} +
mamba run -n qgis_stable python3 -m unittest -v \
    tests.test_high_perf_viewer \
    tests.test_overlay_rain_fields \
    tests.test_coupling_integration \
    tests.test_pipe_cell_coupling_output
```

## Cross-Review Rule

Every code change produced by one subagent must be reviewed by a different subagent before the phase is marked complete.
