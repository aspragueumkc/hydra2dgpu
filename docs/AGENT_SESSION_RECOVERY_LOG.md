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

### Phase 2 — Whitelist + kernel infrastructure (committed: 661a8fe)
- `time_integrator` whitelist in `swe2d_gpu.cu:5030-5032` extended to `{2,3,4,5}`
- `d_k4_h/hu/hv` and `d_k6_h/hu/hv` allocated in init, freed in destroy
- `swe2d_rk3_combine_kernel` and `swe2d_rk4_combine_kernel` added

### Phase 3 — RK3 step function (committed: 6c56085)
- `swe2d_rk3_combine_kernel` at `swe2d_gpu.cu:2319`
- `swe2d_gpu_step_rk3` — textbook RK3 (3rd-order): Stage 1 dt/2→h1, Stage 2 dt/2 from h0→h2, Stage 3 dt from h0→h3, combine `(k1+2*k2+2*k3)/6`
- Dispatch: `case 3 → swe2d_gpu_step_rk3`

### Phase 4 — RK4 step function (IN PROGRESS — build OK)
**swe2d_gpu.cu:**
- `swe2d_rk4_stage3_prep_kernel` `__global__` added (line ~6816): `h += dt*k2` for Stage 3 intermediate
- `swe2d_gpu_step_rk4` rewritten with correct buffer management:
  - Stage 1: `k1 → d_k4`, `h1 → d_h1`
  - Stage 2: `k2 → d_k6`, `h2 → d_h2`, `hu2 → d_hu1`, `hv2 → d_hv1`
  - Stage 3: restore h0, `h += dt*k2` via `swe2d_rk4_stage3_prep_kernel`, then GPU step
  - Stage 4: restore from `d_h2`/`d_hu1`/`d_hv1` (NOT pointer arithmetic on `d_h2`)
- `swe2d_rk4_combine_kernel` bug fixed: added `hu2`/`hv2` params; `k4_hu = cell_hu - hu2`, `k4_hv = cell_hv - hv2` (was incorrectly using `h2[c]` for both)

**swe2d_gpu.cuh:**
- Added `swe2d_gpu_step_rk4` declaration

**swe2d_solver.cpp:**
- `else if (temporal_order == 4) → swe2d_gpu_step_rk4` dispatch added
- `swe2d_create` throw updated to allow `{1,2,3,4}`
- `!use_rk2` → `s->cfg.temporal_order == 1` (fused path only for Euler)

## What's Next

### Phase 5 — RK5 step function (Cash-Karp RK5(4))
- `swe2d_gpu_step_rk5` using existing `swe2d_rk5_graph_combine_kernel` at `swe2d_gpu.cu:2318`
- k1/k3/k4/k6 stored; k2/k5 discarded
- Dispatch: `temporal_order == 5 → swe2d_gpu_step_rk5`

### Phase 6 — Python IMEX removal
- Remove `runtime_step_executor.py` IMEX path
- Remove dead IMEX wrappers in `backend.py`
- Remove `_IMEX_*` constants and `TemporalScheme` IMEX entries

## Key Context
- `d_h2` is packed: `d_h2[c]`=h, `d_h2[c+n_cells]`=hu, `d_h2[c+2*n_cells]`=hv
- `d_hu1`/`d_hv1` store Stage 2 momentum (hu2, hv2) for use in Stage 4 restore
- `swe2d_rk5_graph_combine_kernel` already exists at `swe2d_gpu.cu:2318` with correct Cash-Karp weights
- `SWE2D_GRAPH_STAGE_SLOTS=6` + `d_stage_*` arrays pre-allocated but unused
