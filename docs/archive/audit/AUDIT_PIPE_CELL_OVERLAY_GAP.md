---
type: audit
status: complete
created: 2026-07-13
completed: 2026-07-25
---

# Implementation Gap Audit: Drainage Per-Pipe-Cell + Overlay Fields

**Plan:** `docs/archive/plans/2026-07-13-drainage-pipe-cell-output-and-overlay-fields.md`  
**Audit date:** 2026-07-13  
**Status:** Implementation scaffolded but **functionally broken at runtime**

---

## Executive Summary

The codebase contains the surface-level plumbing for both features (tables, schema docs, data-layer attributes, persistence branches, viewer branches), but a single critical bug — referencing `cc._dsoa` instead of the actual `cc._drainage_soa` attribute — causes pipe-cell storage to remain empty and corrupts overlay field wiring. Additionally, several correctness and completeness issues are present (rain-intensity conversion off by 1000×, non-shape-aware pipe depth, GPKG read-back missing, etc.).

**Root cause most responsible for "not working":** `swe2d/workbench/services/non_gui_runtime_service.py:208` and `:550` call `getattr(cc, "_dsoa", None)`, but the coupling controller stores the packed drainage SoA as `self._drainage_soa` (in `swe2d/runtime/coupling.py:702`). This makes `build_pipe_cell_keys()` always return an empty list, so `results_data.init_pipe_cell_storage()` does nothing, `append_pipe_cell_snapshot()` drops rows, `build_pipe_cell_items()` returns an empty list, and `persist_all_baked_results()` writes no pipe-cell data. The same `_dsoa` lookup also breaks the overlay Manning/CN fallback.

---

## Section 1 — Drainage Per-Pipe-Cell Outputs

| Plan item | Status | File / lines | Notes |
|-----------|--------|--------------|-------|
| New `swe2d_baked_pipe_cell_ts` table | ✅ IMPLEMENTED | `gpkg_persistence_service.py:31, 412, 621-647, 976-1029` | Schema matches plan |
| `cell_owner_link` / `sub_cells_per_link` mapping | ⚠️ PARTIAL | `coupling.py:1161-1168, 1205-1227` | Computed locally; not stored on controller/SoA |
| `readback_coupling_state()` returns `cell_velocity`, `cell_depth`, `cell_flow`, `cell_head` | ⚠️ PARTIAL | `coupling.py:1119-1124, 1173-1227` | Keys present, but `cell_depth = cell_A / cell_width` is **not shape-aware** |
| `_sample_coupling_object_metrics()` handles `drainage_cell` | ⚠️ PARTIAL | `non_gui_runtime_service.py:285-345` | Emits rows, but storage never initialized due to `_dsoa` bug |
| `build_pipe_cell_keys()` | ❌ BROKEN | `non_gui_runtime_service.py:540-568` | Uses `getattr(cc, "_dsoa", None)` instead of `cc._drainage_soa`; also `if not link_lengths:` invalid for NumPy arrays |
| `run_finalizer.py` passes `pipe_cell_items` | ✅ IMPLEMENTED | `run_finalizer.py:328-329, 339` | Passes list, but list is empty due to upstream bug |
| `results/data.py` accepts pipe-cell items | ✅ IMPLEMENTED | `data.py:95, 166-180, 307-324, 333-358, 541-556` | Fully wired, but `_live_pipe_cell` never populated |
| `studio_viewer_profile_pg.py` `drainage_link` / `drainage_node` branches | ⚠️ PARTIAL | `studio_viewer_profile_pg.py:177-182, 707-797, 812-877` | Branches exist; no data → falls into "No pipe-cell data available" path |
| Schema doc updated | ✅ IMPLEMENTED | `docs/RESULTS_GEOPACKAGE_SCHEMA.md:112-135, 257, 305-313` | Matches plan |
| `tests/test_gpkg_coupling_roundtrip.py` pipe-cell test | ❌ MISSING | — | File only has coupling tests |
| `tests/test_swe2d_gpu_drainage_network.py` readback test | ❌ BROKEN | `test_swe2d_gpu_drainage_network.py:1073-1115` | Constructs `SWE2DCouplingController(cfg=MagicMock())` but `__init__` does not accept `cfg` |
| `tests/test_coupling_integration.py` t=0 test | ✅ IMPLEMENTED | `test_coupling_integration.py:257-287` | Exercises `drainage_cell` branch |

### Critical bugs in Section 1

1. **Broken attribute reference `cc._dsoa`** → no pipe-cell data stored or persisted.
2. **Depth/head not shape-aware** — uses `cell_A / cell_width`; incorrect for circular/elliptical pipes.
3. **No GPKG read-back** — `data.py` `_load_coupling_for_first_enabled_run()` only loads `swe2d_baked_coupling`, not `swe2d_baked_pipe_cell_ts`.
4. **`simulation_worker.py` doesn't emit `_live_pipe_cell`** — `_on_snapshot_readback` only emits `_live_coupling`.

---

## Section 2 — Rain / Manning / Curve Number Overlay Fields

| Plan item | Status | File / lines | Notes |
|-----------|--------|--------------|-------|
| UI picker items | ✅ IMPLEMENTED | `results_controls.py:311-321` | Keys differ from plan: `cum_rain`/`cum_excess`/`cum_loss` instead of `cum_rain_mm`/`cum_excess_mm`/`cum_loss_mm` |
| Renderer `elif` branches | ✅ IMPLEMENTED | `high_perf_viewer.py:618-664` | `rain_intensity` conversion is **wrong** |
| Legend fallbacks | ✅ IMPLEMENTED | `high_perf_viewer.py:966-977` | Present, but no explicit override in `overlay_parameters_service.py` |
| `data.py` overlay attributes | ✅ IMPLEMENTED | `data.py:79-84` | All 5 attributes present |
| `overlay_parameters_service.py` passes arrays | ✅ IMPLEMENTED | `overlay_parameters_service.py:223-229` | Passes arrays, but no `legend_label` override for new modes |
| `non_gui_runtime_service.py` copies arrays | ⚠️ PARTIAL | `non_gui_runtime_service.py:170-220` | Rain copied; Manning/CN source is wrong; `step_net_rainfall_mps(t_s, t_s, ...)` has zero interval |
| `swe2d_baked_overlay_fields` table | ✅ IMPLEMENTED | `gpkg_persistence_service.py:651-671` | Table written, but builder stores only last snapshot |
| Schema doc updated | ✅ IMPLEMENTED | `docs/RESULTS_GEOPACKAGE_SCHEMA.md:137-155` | Matches plan |
| `tests/test_overlay_rain_fields.py` | ✅ IMPLEMENTED | — | Tests live-copy and GPKG round-trip |
| `tests/test_high_perf_viewer.py` | ✅ IMPLEMENTED | — | Tests Manning/CN/render |
| `tests/test_overlay_rain_fields_gpkg.py` | ❌ MISSING | — | End-to-end GPKG test not created |

### Critical bugs in Section 2

1. **Rain intensity conversion off by 1000×**: `m/s * 3600` gives `m/hr`, but label says `mm/hr`. Correct factor is `m/s * 3,600,000` for mm/hr, or `m/s * 3600 * 1000 / 25.4` for USC in/hr.
2. **Manning/CN source is wrong**: copied from `cc._dsoa.link_roughness_n` (per-link), not the per-cell `n_mann_cell` spatial array. Same `_dsoa` reference bug.
3. **Zero-width rain step interval**: `forcing.step_net_rainfall_mps(t_s, t_s, ...)` returns zero rate because the interval is zero.
4. **`build_overlay_field_items` stores only last snapshot**: GPKG `values_blob` is `(n_cells,)` instead of documented `(n_timesteps × n_cells)`.
5. **No GPKG read-back** for overlay fields when loading a saved run.
6. **Key name mismatch**: plan specified `cum_rain_mm`, `cum_excess_mm`, `cum_loss_mm`; implementation uses `cum_rain`, `cum_excess`, `cum_loss`. Internally consistent, but deviates from plan and data-attribute naming.

---

## Cross-Cutting Bug: `cc._dsoa` vs `cc._drainage_soa`

The same broken reference appears in both features:

- `non_gui_runtime_service.py:208` — `dsoa = getattr(cc, "_dsoa", None)`
- `non_gui_runtime_service.py:550` — `dsoa = getattr(cc, "_dsoa", None)`

Actual attribute is set at `swe2d/runtime/coupling.py:702`:

```python
self._drainage_soa = dsoa
```

Fixing this single reference will likely make both features advance to the next set of bugs.

---

## Recommendations (Prioritized)

### P0 — Fix the broken reference
- In `non_gui_runtime_service.py`, replace `getattr(cc, "_dsoa", None)` with `getattr(cc, "_drainage_soa", None)`.
- Fix `if not link_lengths:` → `if len(link_lengths) == 0:` on the NumPy array.

### P0 — Fix rain intensity conversion
- In `high_perf_viewer.py:619-622`, use `m/s * 3_600_000.0` for SI (mm/hr) and `m/s * 3600.0 * 1000.0 / 25.4` for USC (in/hr).

### P0 — Fix zero-width rain step interval
- In `_copy_overlay_cell_data_from_coupling`, pass `t_s, t_s + dt_used` (or the actual step interval) instead of `t_s, t_s`.

### P1 — Make pipe depth/head shape-aware
- In `coupling.py:1224`, replace `cell_A / cell_width` with a switch on `link_shape_type` (circular, rectangular, elliptical).

### P1 — Populate Manning/CN from the correct per-cell source
- At mesh-build time, capture `n_mann_cell` and `cn_cell` arrays from the spatial forcing adapter and store them on `results_data.overlay_cell_manning_n` / `overlay_cell_cn`.
- In `non_gui_runtime_service.py`, prefer those arrays over link roughness.

### P1 — Implement GPKG read-back for both features
- Add `load_baked_pipe_cell_ts()` and `load_baked_overlay_fields()` in `gpkg_persistence_service.py`.
- Wire them into `SWE2DResultsData` when loading a baked GPKG.

### P1 — Fix `build_overlay_field_items` to store full time series
- Either append per-timestep arrays during the run or build a `(n_timesteps, n_cells)` array; the GPKG schema expects the flattened row-major shape.

### P2 — Align field keys with plan (or update plan)
- Decide whether the UI/renderer keys should be `cum_rain` or `cum_rain_mm`. Update the other side consistently.

### P2 — Fix / add tests
- Fix `test_swe2d_gpu_drainage_network.py:1073-1115` controller construction.
- Add `test_gpkg_coupling_roundtrip.py` pipe-cell test.
- Add `tests/test_overlay_rain_fields_gpkg.py` end-to-end GPKG test.

---

## Plan vs. Implementation Matrix

| Feature | Planned | Implemented | Working |
|---------|---------|-------------|---------|
| `swe2d_baked_pipe_cell_ts` table | Yes | Yes | No rows written due to `_dsoa` bug |
| Pipe cell readback keys | Yes | Partial | Wrong depth formula |
| Pipe cell persistence | Yes | Yes | No-op upstream bug |
| Pipe profile viewer branch | Yes | Partial | Always empty |
| GPKG pipe-cell read-back | No explicit | No | Missing |
| Overlay UI picker | Yes | Yes | Yes (keys differ) |
| Overlay renderer branches | Yes | Yes | Rain intensity wrong |
| Overlay data attributes | Yes | Yes | Partially populated |
| Overlay GPKG table | Yes | Yes | Only stores last snapshot |
| Overlay GPKG read-back | No explicit | No | Missing |
| Tests | 6 planned | 3 exist, 1 broken | Insufficient |
