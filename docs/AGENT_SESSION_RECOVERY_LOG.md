# Agent Session Recovery Log

## Session Start: Wed Jul 01 2026

## Goal
Fix bugs in pipe1d solver; establish GPU-only node state; rename drainage tests.

## What Was Done

### Bug Fixes
1. **Per-iteration cudaMalloc (Bug 2)**: Moved `d_A_new`/`d_Q_new` outside substep loop in `swe2d_pipe1d_step`. Was allocating fresh each substep — fixed in `swe2d_gpu.cu:10363-10366`.

2. **Uninitialized d_A_new/d_Q_new (CRITICAL — found today)**: `d_A_new` and `d_Q_new` were freshly cudaMalloc'd each substep but NEVER initialized with current cell state. The fully_dynamic kernel read garbage (A=0, Q=0) on every substep. Fix: copy `p.d_A → d_A_new` and `p.d_Q → d_Q_new` via cudaMemcpy at start of each substep (`swe2d_gpu.cu:10370-10372`).

3. **Boundary flux fix**: Changed boundary formula from `Q_cell * dir` (zero when Q=0) to `dH * sqrt(g*|dH|/L)` (drives flow from head difference). Fixes diffusion_wave A updates from zero-Q boundary conditions (`swe2d_gpu.cu:9973-9975`).

4. **node_invert nullptr (Bug 4)**: Added `d_node_invert` to `Pipe1DDeviceState`; both call sites in `swe2d_pipe1d_step` now pass `p.d_node_invert` instead of `nullptr`.

### GPU Node State Management
- Added `swe2d_pipe1d_accumulate_node_flux_kernel`: per-cell atomicAdd to `d_node_net_q` (from_node=-Q, to_node=+Q)
- Added `swe2d_pipe1d_update_node_depth_kernel`: per-node `dh = dt * net_q / surface_area`
- Added `swe2d_pipe1d_node_mass_balance_host`: orchestrates zero → accumulate → update; called once after substep loop
- Added `swe2d_pipe1d_upload_node_depth` binding: H2D upload of node depths before each step
- Added `swe2d_pipe1d_readback_node_state` binding: D2H readback of node_depth, cell_A, cell_Q

### coupling.py Updates
- `apply_native_device_sources` now calls `swe2d_pipe1d_upload_node_depth` before each step and `swe2d_pipe1d_readback_node_state` after each step
- `_sync_gpu_state_back_to_drainage` now reads actual device values (not stale host array)

### Tests
- Renamed `test_swe2d_drainage_structures.py` → `test_coupling_integration.py`
- Created `tests/test_swe2d_pipe1d.py` (7 tests, all pass):
  - 2 mesh build tests (link topology, sub-division)
  - 5 step tests (diffusion_wave updates area, dry=no change, fully_dynamic updates Q+A, substeps decrease area, node depth upload changes area)
- Fixed `solver_mode=DrainageSolverMode.EGL` → `pipe_solver_mode="diffusion_wave"` in integration test files

## Test Results
- `tests/test_swe2d_pipe1d.py`: **7 passed**
- Integration tests (`test_coupling_integration.py`): **6 failed, 3 passed, 17 skipped**
  - The 6 failures are due to a PRE-EXISTING bug: `swe2d_build_pipe1d_mesh` is called with 16 arguments but the binding accepts only 15. This was never triggered before because `swe2d_pipe1d_step` binding was missing (so the early-return `if hasattr(native_mod, "swe2d_pipe1d_step"): return False` fired). My binding changes added the missing functions, exposing this latent bug.
  - NOT related to my pipe1d fixes.

## Key Decisions
- node_depth lives on GPU; Python coupling controller uploads before each step and reads back after
- Surface exchange stays in Python for now (future GPU task)
- fully_dynamic test uses `pipe_solver_mode="diffusion_wave"` (EGL) as default

## Relevant Files Changed
- `cpp/src/swe2d_gpu.cu`: Kernels + step function
- `cpp/src/swe2d_gpu.cuh`: Device state struct additions
- `cpp/src/swe2d_bindings.cpp`: New bindings for upload/readback
- `swe2d/runtime/coupling.py`: Upload/readback calls in apply_native_device_sources
- `tests/test_swe2d_pipe1d.py`: New pipe1d tests
- `tests/test_coupling_integration.py`: Renamed from test_swe2d_drainage_structures.py
