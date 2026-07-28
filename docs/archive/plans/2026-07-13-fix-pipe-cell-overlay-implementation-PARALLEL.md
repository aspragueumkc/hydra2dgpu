---
type: plan
status: complete
created: 2026-07-13
completed: 2026-07-25
---

# Parallel Execution Plan: Fix Pipe-Cell + Overlay Implementation

**Original plan:** `docs/archive/plans/2026-07-13-fix-pipe-cell-overlay-implementation.md`  
**Strategy:** Split into 5 independent file-owner paths that can run in parallel. No shared files between paths. The controller integrates all changes after every path reports done.

**Constraint:** No git worktrees used (per user request). Parallelism is achieved by assigning each subagent a non-overlapping set of files.

---

## Shared Naming Contract (all paths must use these exact names)

### Overlay attributes on `SWE2DResultsData`

| Old name | New name | Internal unit | Display unit (SI) | Display unit (USC) |
|----------|----------|---------------|-------------------|-------------------|
| `overlay_cell_rain_rate_mps` | `overlay_cell_rainfall_rate` | m/s | mm/hr | in/hr |
| `overlay_cell_cum_rain_mm` | `overlay_cell_cumulative_rain` | mm | mm | in |
| `overlay_cell_cum_excess_mm` | `overlay_cell_cumulative_excess` | mm | mm | in |
| (computed in renderer) | `overlay_cell_cumulative_loss` | mm | mm | in |
| `overlay_cell_manning_n` | `overlay_cell_mannings_n` | — | — | — |
| `overlay_cell_cn` | `overlay_cell_curve_number` | — | — | — |

### UI / renderer keys

| Old key | New key |
|---------|---------|
| `rain_intensity` | `rain_intensity` |
| `cum_rain` | `cumulative_rain` |
| `cum_excess` | `cumulative_excess` |
| `cum_loss` | `cumulative_loss` |
| `manning_n` | `mannings_n` |
| `curve_number` | `curve_number` |

### Display conversion

```python
if length_unit_name == "ft":
    # m/s → in/hr
    rainfall_rate_display = overlay_cell_rainfall_rate * 3600.0 * 1000.0 / 25.4
    cumulative_rain_display_inches = overlay_cell_cumulative_rain / 25.4
else:
    # m/s → mm/hr
    rainfall_rate_display = overlay_cell_rainfall_rate * 3_600_000.0
    cumulative_rain_display_mm = overlay_cell_cumulative_rain
```

### Pipe-cell keys

No rename needed. Continue using `cell_flow`, `cell_velocity`, `cell_depth`, `cell_head`, `cell_owner_link`, `cell_sub_idx`.

### Broken reference fix

Wherever the code reads `getattr(cc, "_dsoa", None)`, change to `getattr(cc, "_drainage_soa", None)`.

---

## Path 1: Overlay Runtime Wiring

**Owner file:** `swe2d/workbench/services/non_gui_runtime_service.py`

**Subagent type:** `python-pro` (coding)

**Tasks:**
- Rename all references to `overlay_cell_rain_rate_mps`, `overlay_cell_cum_rain_mm`, `overlay_cell_cum_excess_mm`, `overlay_cell_manning_n`, `overlay_cell_cn` to the new names.
- Fix `getattr(cc, "_dsoa", None)` → `getattr(cc, "_drainage_soa", None)` at line ~208 and line ~550.
- Fix `if not link_lengths:` → `if len(link_lengths) == 0:` in `build_pipe_cell_keys()`.
- Change `_sample_coupling_object_metrics(cc, t_s, _h_s, ...)` signature to accept `dt`.
- Fix the rain-rate call from `step_net_rainfall_mps(t_s, t_s, ...)` to `step_net_rainfall_mps(t_s, t_s + dt, ...)`.
- Update all callers of `_sample_coupling_object_metrics` to pass the correct `dt`.
- Remove the wrong Manning/CN fallback wiring in `_copy_overlay_cell_data_from_coupling` (Manning/CN will be set from mesh by Path 4).

**Verification:**

```bash
mamba run -n qgis_stable python3 -m py_compile swe2d/workbench/services/non_gui_runtime_service.py
mamba run -n qgis_stable python3 -m unittest -v tests.test_overlay_rain_fields
```

---

## Path 2: Overlay Display

**Owner files:** `swe2d/results/high_perf_viewer.py`, `swe2d/workbench/views/results_controls.py`, `swe2d/workbench/services/overlay_parameters_service.py`

**Subagent type:** `python-pro` (coding)

**Tasks:**
- Rename all references to old overlay attribute names to new names in all three files.
- In `high_perf_viewer.py` rain-intensity branch, use the correct conversion factors based on `length_unit_name`.
- In `high_perf_viewer.py` cumulative branches, convert `mm → in` only when `length_unit_name == "ft"`.
- Add a `cumulative_loss` branch that computes `overlay_cell_cumulative_rain - overlay_cell_cumulative_excess`.
- In `results_controls.py`, update picker keys to `cumulative_rain`, `cumulative_excess`, `cumulative_loss`, and update labels to unit-agnostic text ("Cumulative Rain", etc.).
- In `overlay_parameters_service.py`, add a `_label_for_overlay_mode(mode, length_unit_name)` helper and use it to set `legend_label` for the new modes.
- Update legend label fallbacks in `high_perf_viewer.py` to match.

**Verification:**

```bash
mamba run -n qgis_stable python3 -m py_compile swe2d/results/high_perf_viewer.py \
  swe2d/workbench/views/results_controls.py \
  swe2d/workbench/services/overlay_parameters_service.py
mamba run -n qgis_stable python3 -m unittest -v tests.test_high_perf_viewer
```

---

## Path 3: Pipe Depth + Profile

**Owner files:** `swe2d/runtime/coupling.py`, `swe2d/workbench/views/studio_viewer_profile_pg.py`, `swe2d/workbench/workers/simulation_worker.py`

**Subagent type:** `python-pro` (coding)

**Tasks:**
- In `coupling.py:readback_coupling_state()`, replace `cell_depth = cell_A / cell_width` with a shape-aware computation:
  - Circular: solve for depth from area and diameter.
  - Rectangular: `depth = A / width`.
  - Elliptical: use appropriate approximation or table lookup.
- Add `cell_shape_type`, `cell_width`, `cell_height` to the returned dict.
- In `studio_viewer_profile_pg.py`, update the crown line to use shape-aware geometry (`crown = invert + diameter` for circular, `invert + height` for rectangular/elliptical).
- In `simulation_worker.py`, update `_on_snapshot_readback` to emit `_live_pipe_cell` alongside `_live_coupling`.
- Rename any `_m`-suffixed pipe-cell internal variables if present (e.g., `influence_width_m` only if it appears in these files).

**Verification:**

```bash
mamba run -n qgis_stable python3 -m py_compile swe2d/runtime/coupling.py \
  swe2d/workbench/views/studio_viewer_profile_pg.py \
  swe2d/workbench/workers/simulation_worker.py
mamba run -n qgis_stable python3 -m unittest -v tests.test_coupling_integration
```

---

## Path 4: GPKG Read-Back + Full Time Series

**Owner files:** `swe2d/services/gpkg_persistence_service.py`, `swe2d/results/data.py`

**Subagent type:** `python-pro` (coding)

**Tasks:**
- Rename overlay attribute references in both files to the new names.
- Add `load_baked_pipe_cell_ts(conn, run_id)` in `gpkg_persistence_service.py`.
- Add `load_baked_overlay_fields(conn, run_id, n_cells)` in `gpkg_persistence_service.py`.
- In `data.py`:
  - Add `_overlay_field_history: Dict[str, List[np.ndarray]]` and `_overlay_field_times: List[float]`.
  - Append per-timestep arrays to history in the method that copies overlay data.
  - Update `build_overlay_field_items()` to build `(n_timesteps, n_cells)` row-major arrays from history.
  - Wire `load_baked_pipe_cell_ts()` and `load_baked_overlay_fields()` into the GPKG loading path.
- In `clear_live_snapshots()`, reset the overlay history.

**Verification:**

```bash
mamba run -n qgis_stable python3 -m py_compile swe2d/services/gpkg_persistence_service.py \
  swe2d/results/data.py
mamba run -n qgis_stable python3 -m unittest -v tests.test_gpkg_persistence \
  tests.test_pipe_cell_coupling_output
```

---

## Path 5: Manning/CN from 2D Mesh

**Owner file:** `swe2d/workbench/controllers/overlay_controller.py` (or wherever mesh/scalar arrays are initialized for rendering)

**Subagent type:** `python-pro` (coding)

**Tasks:**
- Find where the mesh is initialized for overlay rendering and where `data.overlay_cell_x`, `data.overlay_cell_y`, etc. are set.
- Locate the per-cell Manning and CN arrays from the spatial forcing adapter (likely in `data.mesh_metadata` or `spatial_forcing_qgis_adapter.py`).
- At mesh-build time, set:
  ```python
  data.overlay_cell_mannings_n = np.asarray(n_mann_cell_array, dtype=np.float64)
  data.overlay_cell_curve_number = np.asarray(cn_cell_array, dtype=np.float64)
  ```
- Ensure the arrays are `(n_cells,)` shape.
- If the arrays are not available at mesh-build time, provide a fallback that reads from `data.mesh_metadata`.

**Verification:**

```bash
mamba run -n qgis_stable python3 -m py_compile swe2d/workbench/controllers/overlay_controller.py
mamba run -n qgis_stable python3 -m unittest -v tests.test_high_perf_viewer
```

---

## Path 6: Tests (sequential, after paths 1–5 complete)

**Owner files:** `tests/test_swe2d_gpu_drainage_network.py`, `tests/test_gpkg_coupling_roundtrip.py`, `tests/test_overlay_rain_fields_gpkg.py`, plus updates to existing test files.

**Subagent type:** `test-automator`

**Tasks:**
- Fix `tests/test_swe2d_gpu_drainage_network.py:1073-1115` to construct `SWE2DCouplingController` correctly.
- Add `test_pipe_cell_velocity_depth_head_flow_persisted` to `tests/test_gpkg_coupling_roundtrip.py`.
- Create `tests/test_overlay_rain_fields_gpkg.py` with end-to-end GPKG reload test.
- Add unit-display tests for SI and USC rainfall conversions.
- Add shape-aware pipe depth test.

**Verification:**

```bash
find . -type d -name __pycache__ -exec rm -rf {} +
mamba run -n qgis_stable python3 -m unittest -v
```

---

## Integration Order

1. **Paths 1–5 run in parallel.** Each path edits only its owner files. No commits from subagents.
2. **Controller inspects all diffs.** If any file was edited by more than one path, resolve conflicts manually.
3. **Run target tests for each path.** If a path's tests fail, dispatch a fix subagent for that path only.
4. **Path 6 tests run.** If tests fail, fix and re-run.
5. **Build C++/CUDA.**
6. **Final commit.**

## Conflict Mitigation

- Each path has explicit file ownership. Subagents must not modify files outside their ownership.
- All paths use the shared naming contract above so renames stay consistent.
- Subagents do **not** commit. They report changed files and test results. The controller commits after integration.
- If two paths require a change in the same file, the controller resolves it by applying the more fundamental change first (e.g., Path 1 wins on `non_gui_runtime_service.py`).
