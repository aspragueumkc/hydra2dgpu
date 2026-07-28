---
type: plan
status: complete
created: 2026-07-20
completed: 2026-07-25
---

# Pipe1D Refactor — Phase 2.4/5 Cleanup (Delete legacy code paths)

**Date:** 2026-07-20  
**Scope:** Complete the refactor plan §4 Phase 2.4 + §4 Phase 5 deletions. After this, only the unified face-flux kernel remains; old HLLC/BC kernels and host wrappers are gone.

## Background

Per `docs/pipe1d_face_indexed_refactor_plan.md` §4 Phase 2.4 step 4 and Phase 5 step 4, the following legacy kernels and host wrappers are supposed to be **deleted** because their logic has been absorbed into `swe2d_unified_face_flux_kernel`:

1. `swe2d_pipe_face_flux_kernel` — old HLLC between pipe-end and 2D cell
2. `swe2d_drainage_pipe_end_bc_kernel` — old pipe-end BC updating node_depth
3. `swe2d_drainage_pipe_end_exchange_kernel` — old weir/orifice source-term exchange
4. `swe2d_drainage_inlet_exchange_kernel` — old inlet weir/orifice source-term exchange
5. `swe2d_drainage_outfall_exchange_kernel` — old outfall source-term exchange
6. `swe2d_pipe1d_outfall_bc_kernel` — old OUTFALL_BC (replaced by unified kernel class 1)

Plus their host wrappers and pybind bindings:

7. `swe2d_gpu_upload_pipe_face_coupling` — host wrapper
8. `swe2d_gpu_apply_pipe_face_flux` — host wrapper
9. `swe2d_gpu_readback_pipe_face_q` — host wrapper (legacy diagnostic)
10. `swe2d_gpu_apply_pipe_end_bc` — host wrapper
11. `swe2d_gpu_apply_coupling_drainage` — host wrapper

Plus matches in `swe2d_pipe1d_step`:
12. Calls to `swe2d_drainage_pipe_end_bc_kernel_host` from `swe2d_pipe1d_step`
13. Setup of `d_pipe_end_*` arrays in step
14. Node-head update paths

## Current state of repo

The plan was declared "Phase 2.5 complete" (commit `90df4b5`) with the comment "Old `swe2d_pipe1d_boundary_flux_kernel` deleted." But that's only ONE of the ~8 kernels marked for deletion. The other 7 still exist:

```
cpp/src/swe2d_gpu.cu:3347  __global__ void swe2d_pipe_face_flux_kernel(...)
cpp/src/swe2d_gpu.cu:4940  __global__ void swe2d_drainage_inlet_exchange_kernel(...)
cpp/src/swe2d_gpu.cu:5186  __global__ void swe2d_drainage_outfall_exchange_kernel(...)
cpp/src/swe2d_gpu.cu:8401  void swe2d_gpu_apply_pipe_end_bc(...)
cpp/src/swe2d_gpu.cu:8828  void swe2d_gpu_apply_coupling_drainage(...)
cpp/src/swe2d_gpu.cu:9215  void swe2d_gpu_upload_pipe_face_coupling(...)
cpp/src/swe2d_gpu.cu:9341  void swe2d_gpu_apply_pipe_face_flux(...)
cpp/src/swe2d_gpu.cu:9420  void swe2d_gpu_readback_pipe_face_q(...)
cpp/src/pipe1d.cu:4395     __global__ void swe2d_drainage_pipe_end_bc_kernel(...)
cpp/src/pipe1d.cu:4530     __global__ void swe2d_pipe1d_outfall_bc_kernel(...)
cpp/src/pipe1d.cu:4965     void swe2d_drainage_pipe_end_bc_kernel_host(...)
cpp/src/pipe1d.cu:4994     void swe2d_pipe1d_outfall_bc_kernel_host(...)
cpp/src/pipe1d.cu:5014     void swe2d_drainage_pipe_end_exchange_kernel_host(...)
```

Plus their pybind bindings (only the unique ones listed; old `swe2d_pipe1d_upload_node_depth`, `swe2d_pipe1d_init_area_from_depth`, `swe2d_pipe1d_readback_node_state`, `swe2d_pipe1d_upload_outfall_state`, `swe2d_pipe1d_upload_pipe_ends_and_junctions`, `swe2d_build_pipe1d_mesh` are also still exported).

Effects:
- `swe2d_pipe1d_step` (the main solver entry point) STILL calls `swe2d_drainage_pipe_end_bc_kernel_host` which sets up `d_pipe_end_*` arrays.
- `swe2d_pipe1d_step` ALSO runs the unified face kernel — but the unified kernel's SURFACE_2D_PIPE_END branch is **a no-op** because `face_owner_R = -1` (placeholder, never patched because `node_is_pipe_end` flag isn't passed via the binding).
- Tests still call `swe2d_gpu_upload_pipe_face_coupling` + `swe2d_gpu_apply_pipe_face_flux` (legacy path).
- The two code paths (unified kernel + legacy BC/exchange kernels) **race against each other** in the same step: each computes its own flux for the pipe-end → 2D coupling, the legacy kernel injects to `pipe_A` directly, and the unified kernel injects via fold+Godunov.

This is exactly why the F8 sign error manifested as both surface AND pipe losing mass — the test goes through legacy path (which is "wrong-direction sign" internally), so the bug isn't in the unified kernel.

## Phase 2.4+5 Cleanup Tasks

### Task 1 — Delete `swe2d_pipe_face_flux_kernel` and host wrappers

File: `cpp/src/swe2d_gpu.cu`

Delete:
- `__global__ void swe2d_pipe_face_flux_kernel(...)` (line 3347–3696)
- `void swe2d_gpu_upload_pipe_face_coupling(...)` (line 9215)
- `void swe2d_gpu_apply_pipe_face_flux(...)` (line 9341)
- `void swe2d_gpu_readback_pipe_face_q(...)` (line 9420)
- `void swe2d_gpu_readback_pipe_face_diag(...)` (line 9440)
- Remove all dispatch sites from `swe2d_gpu_compute_coupling_full_on_device` (lines 8700-8800 area)

File: `cpp/src/swe2d_gpu.cuh`

Delete declarations:
- `void swe2d_gpu_upload_pipe_face_coupling(...)`
- `void swe2d_gpu_apply_pipe_face_flux(...)`
- `void swe2d_gpu_readback_pipe_face_q(...)`
- `void swe2d_gpu_readback_pipe_face_diag(...)`
- `SWE2DDeviceState::pipe_ff_ws` struct fields (members of the legac