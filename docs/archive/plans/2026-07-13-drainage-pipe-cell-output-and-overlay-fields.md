---
type: plan
status: complete
created: 2026-07-13
completed: 2026-07-25
---

# Drainage Per-Pipe-Cell Persistence + Overlay Rain/CN/Manning Fields — Implementation Plan

> **For agentic workers:** Use subagent-driven-development skill or executing-plans to implement this plan task-by-task.

**Date:** 2026-07-13
**Status:** Draft
**Owner:** SWE2D Results & Visualization

---

## Scope

This plan covers two independent features sharing a common theme — extending the results pipeline to surface new per-element scalar fields:

**Section 1 — Drainage network outputs:** Persist every 1D pipe-cell's velocity, depth, flow, and head to the results GPKG, and wire a longitudinal pipe-profile view into the existing `PGProfileWidget` so users can see crown/invert geometry and water fill.

**Section 2 — Overlay scalar fields:** Add five new 2D-surface scalar fields to the high-performance canvas overlay picker: rain intensity, cumulative rain, cumulative excess, cumulative loss, Manning's *n*, and Curve Number.

The two sections are independent and may be implemented in any order.

---

## SECTION 1 — Drainage Per-Pipe-Cell Outputs and Profile Viewer

### 1. Problem

The existing coupling pipeline bakes per-node and per-link scalar time series (`drainage_node/depth`, `drainage_link/flow`, etc.) but provides no visibility into the 1D pipe's interior. Users cannot:

- See the velocity, depth, or head distribution along a pipe at a given time.
- View the pipe cross-section geometry (crown / invert) alongside the simulated water surface.
- Inspect per-cell flow through a culvert or long conduit.

### 2. Design

#### 2a. New GPKG table: `swe2d_baked_pipe_cell_ts`

No such table exists today. A new dedicated table is preferable to extending `swe2d_baked_coupling` because pipe-cell count can be 10–100× the node/link count (each link is subdivided into *n_sub* cells), which would otherwise multiply coupling row counts dramatically.

```sql
CREATE TABLE IF NOT EXISTS swe2d_baked_pipe_cell_ts (
    run_id        TEXT,
    link_id       TEXT,
    cell_sub_idx  INTEGER,   -- 0-based sub-cell index within the link
    metric        TEXT,       -- 'velocity' | 'depth' | 'flow' | 'head'
    n_timesteps   INTEGER,
    times_blob    BLOB,       -- float64[]  (length = n_timesteps)
    values_blob   BLOB,       -- float64[]  (length = n_timesteps)
    PRIMARY KEY (run_id, link_id, cell_sub_idx, metric)
);
```

**Why `link_id` + `cell_sub_idx` as key, not a flat `object_id`:** makes it trivial to reconstruct the full per-link longitudinal array without needing a separate cell-owner lookup. The total number of rows per run = `n_links × n_sub_per_link × n_metrics` (4: velocity, depth, flow, head). For a 100-link network with 10 sub-cells each, that's 4 000 rows — well within GPKG performance.

#### 2b. Per-cell identity — cell-owner mapping

The existing Python `SWE2DDrainageSoA` (`swe2d/runtime/coupling.py`) already stores `link_length`, `link_diameter`, and computes `sub_cells_per_link[]` during `readback_coupling_state()`. Extend that same computation to produce a flat `cell_owner_link` array of shape `(n_pipe_cells,)` — i.e. `cell_owner_link[c] = link_index` — and expose it alongside `cell_A` / `cell_Q` from the existing C++ readback.

Implementation path:
1. In `SWE2DCouplingController._build_dsoa()` or a new `SWE2DCouplingController._build_cell_owner_map()`, compute `cell_owner_link` Python-side using the same subdivision logic already in `readback_coupling_state()` (lines 1144–1153 of `coupling.py`).
2. Pass it to the C++ binding as an optional output array alongside the existing `n_pipe_cells` readback, or derive it in Python from the per-link subdivision offsets. No new CUDA kernel needed.

#### 2c. Snapshot pipeline — Python readback changes

`readback_coupling_state()` in `swe2d/runtime/coupling.py:1109–1201` currently calls the C++ binding and discards `cell_A` / `cell_Q`. Extend it to:

1. Capture the returned `cell_A` and `cell_Q` arrays from `swe2d_pipe1d_readback_node_state()`.
2. Derive per-cell `velocity = |Q| / max(ε, A)` and `depth` via the existing `pipe1d_lookup_geometry`-equivalent Python helper (shape-type switch on `dsoa.link_shape_type[link_idx]`).
3. Derive `head = cell_invert + depth` using `dsoa.link_diameter` or the shape-specific geometry.
4. Return them as new dict keys: `cell_velocity`, `cell_depth`, `cell_flow (= Q)`, `cell_head`.
5. `_sample_coupling_object_metrics()` in `swe2d/workbench/services/non_gui_runtime_service.py:170–380` — add a new `component="drainage_cell"` branch that emits one row per `(link_id, sub_idx, metric)` per snapshot. This is a loop over `n_pipe_cells × 4_metrics` per output interval; acceptable since pipe-cell counts are much smaller than 2D cell counts.

#### 2d. Persistence write — `persist_all_baked_results()`

In `swe2d/services/gpkg_persistence_service.py:419–625`, add a new `pipe_cell_items` argument (parallel to `coupling_items`) and a new `CREATE TABLE IF NOT EXISTS swe2d_baked_pipe_cell_ts` block. Write via `persist_baked_pipe_cell_ts()` (new, following the existing `persist_baked_coupling_batch` pattern).

The caller in `swe2d/runtime/run_finalizer.py` (approx lines 249–336) needs a new branch to pass `pipe_cell_items` downstream.

#### 2e. Profile viewer extension — longitudinal pipe cross-section in `PGProfileWidget`

The existing `swe2d/workbench/views/studio_viewer_profile_pg.py:184` (`PGProfileWidget`) already has `drainage_node` and `drainage_link` in `_ELEMENT_TYPES` (lines 160–165) but routes them through the generic time-series branch (`else:` at ~line 538). The extension adds a third branch:

```
if etype == "drainage_link":
    # Draw longitudinal pipe profile
elif etype == "drainage_node":
    # Draw node marker + time-series for head / depth / volume
else:
    # existing time-series or sample-line profile branch
```

**Drainage link longitudinal profile** (new branch):

- **Station axis:** `x = np.linspace(0.0, link.length, n_sub)` — cumulative station in meters.
- **Geometry data:** load from `SWE2DResultsData._live_coupling[(f"drainage_cell", f"{link_id}#{sub_idx}", "depth")]` (new storage keys added to `_live_coupling`). Per-link cell count `n_sub` is reconstructed from the snapshot by link ID using `dsoa.sub_cells_per_link`.
- **Crown line** (horizontal dashed, dark grey):
  - `crown_y = invert_elev + diameter` (circular) or `invert_elev + height` (rectangular/elliptical) — from `dsoa.link_diameter[]` or `dsoa.link_height[]`.
  - `pg.PlotDataItem(x, crown_y, pen=pg.mkPen(QColor(64,64,64), width=0.9, style=Qt.PenStyle.DashLine))`
- **Invert line** (solid, brown):
  - `invert_y = np.full_like(x, invert_elev)` — from `dsoa.link_invert_elev[]` (per-link constant unless a sloped pipe is modelled).
  - `pg.PlotDataItem(x, invert_y, pen=pg.mkPen(QColor(92,64,51), width=1.0))`
- **Water fill** (blue fill-between, same `pg.FillBetweenItem` idiom as the 2D surface profile):
  - `water_surface_y = invert_y + np.array(depth_vals)` — depth from new `drainage_cell/{link_id}/{sub_idx}/depth` time series at chosen `t_sec`.
  - `pg.FillBetweenItem(curve1=invert_plotdata, curve2=water_plotdata, brush=pg.mkBrush(QColor(100,149,237, 96)))`
- **Velocity-shading overlay** (optional fill mode): per-segment color using the identical loop pattern from `refresh()` lines 480+ and the existing `_cmap_color()` LUT (lines 128–136).
- **Node junction markers:** vertical dashed lines at `x=0` and `x=link.length` with node IDs as `pg.TextItem` annotations.

**Drainage node static info** (new branch):

- Render node head (`invert_elev + depth`) as a time-series line chart, and node storage volume (`depth × surface_area`) as a second series, using the existing time-series plotting path (which already handles arbitrary `metric` keys).
- No new geometry drawing needed; node is a 0-dimensional object.

#### 2f. GPKG schema update

- `docs/RESULTS_GEOPACKAGE_SCHEMA.md` — add `swe2d_baked_pipe_cell_ts` table declaration with column descriptions and an ER diagram entry.
- `docs/MODEL_GEOPACKAGE_SCHEMA.md` — no change (input model unchanged).

#### 2g. Test stubs

- `tests/test_gpkg_coupling_roundtrip.py` — add `test_pipe_cell_velocity_depth_head_flow_persisted()`: run a short simulation with a known drainage network, assert `swe2d_baked_pipe_cell_ts` rows exist with correct shape (`n_sub × n_timesteps`).
- `tests/test_swe2d_gpu_drainage_network.py` — add `test_readback_coupling_state_returns_cell_arrays()` asserting `cell_velocity`, `cell_depth`, `cell_flow`, `cell_head` keys are present in the return dict.
- `tests/test_coupling_integration.py` — add `test_drainage_cell_snapshot_at_t0_is_zero()` asserting all `drainage_cell/*/depth` values are 0 at the t=0 snapshot (confirming no spurious priming).

### 3. Key file changes summary

| File | Change | Lines |
|------|--------|-------|
| `swe2d/runtime/coupling.py` | Extend `readback_coupling_state()` to return `cell_velocity/depth/flow/head` dict keys | ~1109–1201 |
| `swe2d/runtime/coupling.py` | Add `cell_owner_link` / `sub_cells_per_link` computation | ~1144–1167 |
| `swe2d/services/gpkg_persistence_service.py` | New `persist_baked_pipe_cell_ts()` function | ~new |
| `swe2d/services/gpkg_persistence_service.py` | New `CREATE TABLE swe2d_baked_pipe_cell_ts` in `persist_all_baked_results()` | ~419–625 |
| `swe2d/workbench/services/non_gui_runtime_service.py` | Add `drainage_cell` branch in `_sample_coupling_object_metrics()` | ~170–380 |
| `swe2d/runtime/run_finalizer.py` | Pass `pipe_cell_items` to `persist_all_baked_results()` | ~249–336 |
| `swe2d/workbench/views/studio_viewer_profile_pg.py` | Add `drainage_link` longitudinal profile + `drainage_node` node branch in `refresh()` | ~406–580 |
| `swe2d/results/data.py` | Accept `pipe_cell_items` in `SWE2DResultsData`; extend `append_coupling_snapshot` or add `append_pipe_cell_snapshot` | ~132–149, ~263–274 |
| `docs/RESULTS_GEOPACKAGE_SCHEMA.md` | Document `swe2d_baked_pipe_cell_ts` table | ~new section |
| `tests/test_gpkg_coupling_roundtrip.py` | Add pipe-cell persistence test | ~new |
| `tests/test_swe2d_gpu_drainage_network.py` | Add cell-array readback test | ~new |
| `tests/test_coupling_integration.py` | Add t=0 cell-depth-zero test | ~new |

---

## SECTION 2 — Rain / Manning / Curve Number Overlay Fields

### 1. Problem

The high-performance canvas overlay (`swe2d/results/high_perf_viewer.py`) currently offers six scalar fields: `depth`, `speed`, `wse`, `froude`, `courant`, `shear_stress`. Users need to visualise:

- **Rainfall intensity** — where is it raining hardest right now?
- **Cumulative rainfall** — total precipitation received per cell.
- **Cumulative excess** — precipitation minus initial losses (what is available to run off).
- **Cumulative loss / infiltration** — how much water has been absorbed by the soil.
- **Manning's *n*** — spatial roughness distribution (why is flow slow in some areas?).
- **Curve Number (CN)** — soil/spatial runoff potential distribution.

None of these are currently renderable from the overlay picker.

### 2. Design

#### 2a. Extend the overlay field registry

**Registration point 1 — UI picker** (`swe2d/workbench/views/results_controls.py:326–360`):

```python
# Add to the field_combo items list:
("Rain Intensity (mm/hr)", "rain_intensity"),
("Cumulative Rain (mm)",   "cum_rain_mm"),
("Cumulative Excess (mm)", "cum_excess_mm"),
("Cumulative Loss (mm)",   "cum_loss_mm"),
("Manning's n (–)",        "manning_n"),
("Curve Number (–)",       "curve_number"),
```

**Registration point 2 — renderer if/elif chain** (`swe2d/results/high_perf_viewer.py:574–608`):

```python
elif mode == "rain_intensity":
    # Rate in m/s → convert to mm/hr for display
    # Source: data.overlay_cell_rain_rate_mps (new, see §2c)
    vals = data.overlay_cell_rain_rate_mps * 3600.0  # m/s → mm/hr
    vals[~wet_all] = np.nan
elif mode == "cum_rain_mm":
    vals = data.overlay_cell_cum_rain_mm.copy()
    vals[~wet_all] = np.nan
elif mode == "cum_excess_mm":
    vals = data.overlay_cell_cum_excess_mm.copy()
    vals[~wet_all] = np.nan
elif mode == "cum_loss_mm":
    vals = data.overlay_cell_cum_rain_mm - data.overlay_cell_cum_excess_mm
    vals[~wet_all] = np.nan
elif mode == "manning_n":
    vals = data.overlay_cell_manning_n.copy()
    vals[~wet_all] = np.nan
elif mode == "curve_number":
    vals = data.overlay_cell_cn.copy()
    vals[~wet_all] = np.nan
```

**Registration point 3 — legend label defaults** (`swe2d/results/high_perf_viewer.py:898–908`):

```python
elif mode == "rain_intensity":  label = "Rain Intensity (mm/hr)"
elif mode == "cum_rain_mm":     label = "Cumulative Rain (mm)"
elif mode == "cum_excess_mm":   label = "Cumulative Excess (mm)"
elif mode == "cum_loss_mm":     label = "Cumulative Loss (mm)"
elif mode == "manning_n":       label = "Manning's n (–)"
elif mode == "curve_number":    label = "Curve Number (–)"
```

**Registration point 4 — overlay parameter collector** (`swe2d/workbench/services/overlay_parameters_service.py:79–221`): add `manning_n` and `cn` keys to the kwargs dict passed to `render_unstructured_snapshot_image`, using the already-available `view._mannings_n` scalar (passed to `shear_stress` today) and the per-cell `data.overlay_cell_cn` array (new).

#### 2b. Unit convention

- **Rain intensity:** native model unit is `m/s`; display unit is `mm/hr`. Conversion factor `× 3600 × 1000 = × 3 600 000`. The renderer already handles `m/s` for speed/depth. Use `vals = rate_mps * 3600.0` for intensity; label override in `overlay_parameters_service.py` legend builder (lines 177–186) to say `mm/hr`.
- **Cumulative rain / excess / loss:** display unit is `mm` (matching the internal SCS-CN units already used in `rainfall_hydrology.py`). No unit conversion needed.
- **Manning's *n* and CN:** dimensionless; no conversion.

#### 2c. Data sources — attaching arrays to `SWE2DResultsData`

The snapshot tuple `(t_s, h, hu, hv)` is the canonical data carrier for the overlay. Rather than changing that tuple format, attach the new arrays as attributes on `SWE2DResultsData`:

```python
# In swe2d/results/data.py, add to __init__ or reset():
self.overlay_cell_rain_rate_mps  = np.empty(0, dtype=np.float64)  # shape (n_cells,) latest step rate
self.overlay_cell_cum_rain_mm    = np.empty(0, dtype=np.float64)  # shape (n_cells,)
self.overlay_cell_cum_excess_mm  = np.empty(0, dtype=np.float64)  # shape (n_cells,)
self.overlay_cell_manning_n       = np.empty(0, dtype=np.float64)  # shape (n_cells,)
self.overlay_cell_cn             = np.empty(0, dtype=np.float64)  # shape (n_cells,)
```

These arrays are populated in **two paths**:

**Path A — Live simulation** (in `swe2d/workbench/services/non_gui_runtime_service.py` around the `step_wall_t0` timing block at line 538, inside the per-step loop in `execute_run_timestep_loop`):
1. After each `coupling_controller.apply_native_device_sources()` + `backend.step()` cycle, the runtime has a `ThiessenRainCNForcing` instance accessible via `coupling_controller._rain_forcing` (or similar attribute).
2. Call `forcing.step_net_rainfall_mps(t, dt)` — it mutates `forcing._loss_calculator.cumulative_rain_mm` and `cumulative_excess_mm` in-place.
3. Copy those arrays to `results_data.overlay_cell_cum_rain_mm[...] = forcing._loss_calculator.cumulative_rain_mm` and `overlay_cell_cum_excess_mm[...] = forcing._loss_calculator.cumulative_excess_mm`.
4. Copy `forcing.step_net_rainfall_mps` return `rate_mps` to `overlay_cell_rain_rate_mps`.
5. For Manning's *n* and CN: these are static per-cell arrays uploaded at mesh build time. Store them once at `init_overlay_arrays()` time from the mesh metadata already in `data.mesh_metadata` (which holds `n_mann_cell` and `cn_cell` from the spatial forcing adapter).

**Path B — GPKG-baked results** (in `swe2d/results/data.py` `set_live_snapshot_timesteps()` at lines 171–219, or a new `set_baked_overlay_fields()`):
- At `set_live_snapshot_timesteps()` time, detect that a baked GPKG is being loaded. If `swe2d_baked_overlay` table exists (new, see §2d), read the per-timestep arrays from it.
- Otherwise: leave the arrays empty and show a "data not available" placeholder in the overlay legend.

#### 2d. GPKG table: `swe2d_baked_overlay_fields` (optional persistence)

To make baked results browsable with the same fields, add an optional table:

```sql
CREATE TABLE IF NOT EXISTS swe2d_baked_overlay_fields (
    run_id        TEXT,
    metric        TEXT,       -- 'rain_rate' | 'cum_rain' | 'cum_excess' | 'manning_n' | 'cn'
    n_timesteps   INTEGER,
    times_blob    BLOB,       -- float64[]
    values_blob   BLOB,       -- float64[]  (flattened n_timesteps × n_cells)
    PRIMARY KEY (run_id, metric)
);
```

Write this in `persist_all_baked_results()` alongside the mesh results. Read in `set_live_snapshot_timesteps()` when the GPKG path is detected.

> **Fallback:** if this table is absent (e.g. older GPKG), the overlay fields will be disabled with a tooltip "Rain/Manning/CN data not available for this run".

#### 2e. Manning's *n* and CN per-cell arrays — where they live

- **Manning's *n*:** `dev->d_n_mann_cell` on the GPU (`swe2d_gpu.cuh:128`), uploaded at mesh build time. On the Python side, `spatial_forcing_qgis_adapter.py:build_spatial_manning_array_qgis()` produces the `(n_cells,)` numpy array. Wire it into `data.overlay_cell_manning_n` at mesh-init time (same place as `data.overlay_cell_x` etc.).
- **Curve Number:** `dev->d_rain_cn` on the GPU (`swe2d_gpu.cuh:297`). Produced by `spatial_forcing_qgis_adapter.py:build_spatial_cn_array_qgis()`. Same wiring.
- The arrays are static (do not vary with time) — store once, reuse every frame.

#### 2f. Overlay parameter service wiring

`swe2d/workbench/services/overlay_parameters_service.py` (`collect_overlay_parameters()` at lines 79–221) currently passes `manning_n=float(view._mannings_n)` only for `shear_stress` mode. Extend to always include `manning_n=data.overlay_cell_manning_n` (when non-empty) and `cn=data.overlay_cell_cn` (when non-empty) so the renderer can access them.

#### 2g. Test stubs

- `tests/test_overlay_rain_fields.py` (new): instantiates a tiny 2D + rain run, asserts `overlay_cell_cum_rain_mm` increases over time, asserts `overlay_cell_rain_rate_mps` is non-negative.
- `tests/test_high_perf_viewer.py`: add `test_render_manning_n_field()` and `test_render_cn_field()` — render with `field_key="manning_n"` / `field_key="curve_number"` and assert no exception.
- `tests/test_overlay_rain_fields_gpkg.py` (new): end-to-end CLI run, persist results, reload GPKG, assert `swe2d_baked_overlay_fields` rows exist and decode correctly.

### 3. Key file changes summary

| File | Change | Lines |
|------|--------|-------|
| `swe2d/results/high_perf_viewer.py` | Add 6 `elif mode ==` branches in scalar field if/elif chain | ~574–608 |
| `swe2d/results/high_perf_viewer.py` | Add legend label fallbacks for 6 new fields | ~898–908 |
| `swe2d/workbench/views/results_controls.py` | Add 6 items to `field_combo` in `_populate_overlay_combos()` | ~326–360 |
| `swe2d/workbench/services/overlay_parameters_service.py` | Add `overlay_cell_manning_n`, `overlay_cell_cn` to kwargs in `collect_overlay_parameters()` | ~79–221 |
| `swe2d/results/data.py` | Add `overlay_cell_rain_rate_mps`, `overlay_cell_cum_rain_mm`, `overlay_cell_cum_excess_mm`, `overlay_cell_manning_n`, `overlay_cell_cn` attributes | ~__init__ |
| `swe2d/workbench/services/non_gui_runtime_service.py` | Copy rain/Manning/CN arrays into `results_data` after each step | ~538 |
| `swe2d/services/gpkg_persistence_service.py` | New `CREATE TABLE swe2d_baked_overlay_fields` in `persist_all_baked_results()` | ~419–625 |
| `docs/RESULTS_GEOPACKAGE_SCHEMA.md` | Document `swe2d_baked_overlay_fields` table | ~new section |
| `tests/test_overlay_rain_fields.py` | New: live rain overlay test | ~new |
| `tests/test_high_perf_viewer.py` | New: Manning / CN render tests | ~new |
| `tests/test_overlay_rain_fields_gpkg.py` | New: GPKG round-trip test for overlay fields | ~new |

---

## Open Questions

1. **Should per-cell pipe outputs be written to GPKG at every output interval or only at run finalization?** The current coupling path writes only at finalization (`persist_all_baked_results`). Following that pattern for pipe-cell data is consistent but means the GPKG won't have intermediate snapshots of pipe-cell data. If interactive pipe-profile playback from GPKG is needed, a ring-buffer readback similar to the 2D mesh path would be needed — deferred to a follow-on plan.

2. **Should the pipe longitudinal profile viewer support multi-link path tracing?** The architecture doc (`docs/archive/references/DRAINAGE_PROFILE_VIEWER_ARCH.md`) sketches a `LinkPathSegment` / `_trace_path()` for following a path through multiple links. The MVP scope covers single-link selection only.

3. **Rain overlay — live vs. GPKG data alignment:** in live mode, `overlay_cell_cum_rain_mm` updates every output-interval step. In GPKG playback mode, the baked `swe2d_baked_overlay_fields` table must be read at the same snapshot timesteps as the mesh results for alignment. Ensure `set_live_snapshot_timesteps()` handles both in one pass.

4. **Manning's n and CN arrays — should they be re-interpolated from the mesh at GPKG reload time?** These are static spatial fields, but the mesh cells may change between runs. The safe approach: store a copy in `swe2d_baked_overlay_fields` alongside the per-timestep rain data, rather than re-reading from the original spatial forcing layers.
