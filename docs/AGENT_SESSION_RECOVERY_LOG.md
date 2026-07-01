# Agent Session Recovery Log

## Session Start: Wed Jul 01 2026

## Goals
1. Fix bugs in pipe1d solver; establish GPU-only node state; rename drainage tests. ✅
2. Implement RAINFALL_SOURCE_OPTIMIZATION_PLAN (docs/RAINFALL_SOURCE_OPTIMIZATION_PLAN.md) ✅
3. Fix remaining 4 failing integration tests (drainage coupling)

## What Was Done

### Rainfall Source Optimization (NEW - Session 2)

**Phase 0: Data structures**
- Added `rain_update_interval_s = 60.0` to `SWE2DDeviceState` (re-evaluate SCS-CN rate every N seconds)
- Added `last_rain_update_time = -1.0` (scalar last update tick, host-owned)
- Added `d_rain_excess_at_last_update_mm` (device buffer, snapshot of cumulative excess at last update)
- Updated `swe2d_gpu_set_rain_cn_forcing` to allocate/deallocate this buffer and accept `rain_update_interval_s` parameter
- Updated `swe2d_solver_set_rain_cn_forcing` with new `rain_update_interval_s` parameter
- Updated Python binding `swe2d_solver_set_rain_cn_forcing` with new arg
- Updated `backend.py::set_rain_cn_forcing_native` with new `rain_update_interval_s` parameter

**Phase 1: New update kernel**
- `swe2d_update_rain_source_rate_kernel`: computes average SCS-CN excess rate over [t_prev, t_now]
  - Reads `prev_cum_excess_mm` snapshot to compute incremental excess
  - Writes to `cell_source_mps` with the average rate for the interval
  - Updates `cum_rain_mm` and `cum_excess_mm` cumulatively

**Phase 2: Integration**
- **swe2d_gpu_step** (main path): Replaced per-step `swe2d_build_rain_cn_source_kernel` call with interval-checked update:
  - First call ever (last_rain_update_time < 0): snapshot cum_excess → d_rain_excess_at_last_update_mm, set last_update = t_now (no rate applied yet)
  - If t_now - last_update >= interval: launch update kernel, snapshot cum_excess, update last_update
  - Otherwise: do nothing (cell_source_mps retains previous value)
- **swe2d_gpu_step** (chunked cooperative kernel path): Same interval-based update
- **RK functions (rk2, rk2_persistent_chunk, rk3, rk4, rk5)**: REMOVED all rain CN save/restore blocks and per-stage build kernel calls. Rain source is now constant between intervals; no cumulative state mutation during RK steps.
- **Breaking change**: Default `rain_update_interval_s=60.0`. Tests that expect per-step rain application will need to pass `rain_update_interval_s=0.0` to get the old behavior.

### Bug Fixes (Session 1)
1. **Per-iteration cudaMalloc**: Moved `d_A_new`/`d_Q_new` outside substep loop in `swe2d_pipe1d_step`.
2. **Uninitialized d_A_new/d_Q_new (CRITICAL)**: `d_A_new` and `d_Q_new` were freshly cudaMalloc'd each substep but NEVER initialized with current cell state. Fix: `cudaMemcpy p.d_A → d_A_new` at start of each substep.
3. **Boundary flux fix**: Changed from `Q_cell * dir` to `dH * sqrt(g*|dH|/L)` head-difference formula.
4. **node_invert nullptr**: Added `d_node_invert` to `Pipe1DDeviceState`; fixed both call sites in `swe2d_pipe1d_step`.

### GPU Node State Management (Session 1)
- Added `swe2d_pipe1d_accumulate_node_flux_kernel`: per-cell atomicAdd to `d_node_net_q`
- Added `swe2d_pipe1d_update_node_depth_kernel`: per-node `dh = dt * net_q / surface_area`
- Added `swe2d_pipe1d_node_mass_balance_host`: orchestrates zero → accumulate → update
- Added `swe2d_pipe1d_upload_node_depth` binding
- Added `swe2d_pipe1d_readback_node_state` binding

### coupling.py Updates (Session 1)
- `apply_native_device_sources` calls `swe2d_pipe1d_upload_node_depth` before step and `swe2d_pipe1d_readback_node_state` after
- Fixed `float(dsoa.max_cell_length) → int(dsoa.max_cell_length)` (pybind11 int32_t mismatch)
- Removed extra `nn` arg from `swe2d_pipe1d_upload_node_depth` call (binding derives n_nodes from shape)

### Tests
- Renamed `test_swe2d_drainage_structures.py` → `test_coupling_integration.py`
- Created `tests/test_swe2d_pipe1d.py` (7 tests, all pass)

## Test Results
- `tests/test_swe2d_pipe1d.py`: **7 passed** ✅
- `tests/test_coupling_integration.py`: **5 passed, 4 failed** (node depth not changing after coupling)

## Failing Tests (4) — Unresolved
All 4 failures show `node_n0_depth == 1.5` (initial value, unchanged) after `apply_native_device_sources` returns True:
1. `test_backend_cell_area_cache_and_source_callback`: h=3e-06 not 0.02 (max_rel_depth_increase cap issue)
2. `test_backend_gpu_run_combines_rain_and_drainage_sources`: h=3e-06 not 0.02
3. `test_face_flux_preloaded_with_drainage`: node depth unchanged
4. `test_gpu_persistent_path_with_drainage_and_culverts`: node depth unchanged

**Debug trace shows**:
- `apply_native_device_sources` returns True ✓
- `_gpu_node_depth` initialized correctly: [1.5, 0.8] ✓
- `static_args` has all keys ✓
- `dev_ptr` is valid non-zero ✓
- BUT: node depths remain at initial values after the call

**Hypothesis**: The `swe2d_pipe1d_step` runs but the node mass balance or the diffusion_wave flux computation isn't producing the expected flow change. Possible causes:
1. The node_surface_area=50 m² is very large compared to pipe area (0.19635 m²), so per-step depth change is tiny
2. The diffusion_wave solver needs multiple substeps to accumulate significant flow
3. The pipe1d mesh subdivision with max_cell_length=0 produces only 1 cell per link (no subdivision), which might affect connectivity

## Key Decisions
- Rain interval approach: re-evaluate SCS-CN rate every 60s, hold constant between updates
- First call ever: snapshot excess and set timer, no rate applied yet
- Breaking change: tests needing per-step rain must pass `rain_update_interval_s=0.0`
- node_depth lives on GPU; upload before step, readback after

## Relevant Files Changed (Rainfall Optimization)
- `cpp/src/swe2d_gpu.cuh`: Added rain interval fields to SWE2DDeviceState
- `cpp/src/swe2d_gpu.cu`: New kernel, interval-based update, removed RK save/restore
- `cpp/src/swe2d_solver.cpp`: Updated setter signature
- `cpp/src/swe2d_solver.hpp`: Updated declaration
- `cpp/src/swe2d_bindings.cpp`: Updated binding with new arg
- `swe2d/runtime/backend.py`: Updated set_rain_cn_forcing_native
