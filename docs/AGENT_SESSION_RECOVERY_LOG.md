# Agent Session Recovery Log

## Session: higher-order-coupling-issues

## Goal
Implement the full Phase 0-6 temporal scheme fix spec from `docs/TEMPORAL_SCHEME_FIX_SPEC.md`.

## What's Done

### Phase 0 — Value-set reconciliation (committed: a94ff6d)
- `swe2d/extensions/extension_models.py`: added `CLASSIC_RK4 = 4` to `TemporalScheme` enum
- `swe2d/workbench/views/model_tab_view.py`: inserted `("RK3 (SSP Shu-Osher, 3rd-order)", 3)` at combo index 2
- `swe2d/workbench/services/constants_service.py`: added RK3 entry to `TEMPORAL_ORDER_OPTIONS`
- `swe2d/workbench/services/run_service.py`: expanded `_VALID_TEMPORAL_SCHEMES` from `{0,1,2,3,4}` → `{1,2,3,4,5,6}`
- `cpp/src/swe2d_solver.cpp`: added `throw std::invalid_argument` in `swe2d_create` for `temporal_order ∉ {1,2}`

### Phase 1 — RK2 stale coupling fix (committed: 0116195)
**swe2d_gpu.cuh:**
- Added `d_rain_cn_scratch_h` and `d_rain_cn_scratch_ex` fields to `SWE2DDeviceState` struct
- Updated `swe2d_gpu_compute_coupling_full_on_device` declaration: added `bool graph_safe = false` param
- Added `swe2d_recompute_coupling_for_stage` declaration

**swe2d_gpu.cu:**
- `swe2d_gpu_compute_coupling_full_on_device`: added `graph_safe` param; sync now conditional on `!graph_safe`
- Added `swe2d_recompute_coupling_for_stage` wrapper after the above function
- `swe2d_gpu_alloc_rainfall`: allocate `d_rain_cn_scratch_h/ex`
- Deallocation: free `d_rain_cn_scratch_h/ex`
- `swe2d_gpu_step_rk2`: rain CN save/restore uses `d_rain_cn_scratch_h/ex` (not `d_h1/d_h2`)
- `swe2d_gpu_step_rk2_persistent_chunk`: same rain CN fix

**swe2d_solver.cpp:**
- Removed unused `use_rk2` boolean
- Dispatch: `temporal_order == 1` → `swe2d_gpu_step` direct; `temporal_order == 2` → `swe2d_gpu_step_rk2`; `default` → throw

## What's Next

### Phase 2 — Whitelist + kernel infrastructure
- Extend `time_integrator` whitelist in `swe2d_gpu.cu:4907-4910` to include `{2,3,4,5}`
- Allocate `d_k4_*` / `d_k6_*` buffers in device state
- Add `swe2d_blend_kernel`, `swe2d_rk3_combine_kernel`, `swe2d_rk4_combine_kernel` declarations
- Add `d_stage_*` BC snapshot helpers

### Phase 3 — RK3 step function
- Implement `swe2d_gpu_step_rk3` in `swe2d_gpu.cu`
- Rain CN: `save_rain_cn_to_scratch` (stage 1), `snapshot_edge_bc_to_stage_slot` (stage 2), `copy_stage_sources` (stage 3)
- Update dispatch in `swe2d_solver.cpp` for `case 3`

### Phase 4 — RK4 step function
- Implement `swe2d_gpu_step_rk4` using `d_h1/d_h2/d_h3` for k1/k2/k3 slopes
- Add `d_k4_*` buffers; compute k4 fresh in stage 4

### Phase 5 — RK5 step function
- Lock to Cash-Karp RK5(4) embedded
- k1/k3/k4/k6 stored; k2/k5 discarded
- Update dispatch for `case 4` and `case 5`

### Phase 6 — Python IMEX removal
- Remove `runtime_step_executor.py` IMEX path
- Remove dead IMEX wrappers in `backend.py`
- Remove `_IMEX_*` constants and `TemporalScheme` IMEX entries

## Key Context
- `temporal_order 4` and `5` both route to `swe2d_gpu_step_rk4` (algorithm identical; GUI differentiates)
- `d_h1/d_h2/d_h3` are reused across schemes (not new allocations)
- Per-stage graph capture (Strategy B) — each `swe2d_gpu_step` call captures independently
- C++ invariant: `throw` for `temporal_order ∉ {1,2}` in `swe2d_create` until Phases 3-5 land
