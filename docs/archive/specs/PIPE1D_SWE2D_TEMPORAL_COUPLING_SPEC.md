---
type: spec
status: active
created: 2026-07-22
---

# Pipe1D–SWE2D Temporal Coupling: Single-Exchange Architecture

**Date:** 2026-07-22
**Status:** Specification (synthesized from all prior plans)
**Supersedes:** `pipe_end_face_flux_plan.md` §3, `pipe1d_face_indexed_refactor_plan.md` §4.2.2, `pipe1d_phase_2_4_cleanup_plan.md`

---

## 1. Background

Tracing backward through the plan lineage:

| Date | Document | Relevant content |
|------|----------|-----------------|
| Jul 17 | `pipe_end_face_flux_plan.md` | Replace source-term pipe↔2D coupling with face-flux HLLC |
| Jul 19–20 | `pipe1d_face_indexed_refactor_plan.md` | Unified face-flux kernel in `pipe1d_step`, single exchange per timestep |
| Jul 20 | `pipe1d_phase_2_4_cleanup_plan.md` | Documents that `pipe_ff_ws` path and unified kernel co-exist, causing races |
| Jul 20 | `coupling_compat_shim_removal_2026-07-20.md` | Python-side binding renames; no structural change |
| Jul 21–22 | Agent-session investigation | Mass loss traced to `swe2d_gpu_step` re-evaluating exchange |

The original plan (Jul 17) and the face-indexed refactor plan (Jul 19–20) both specify a **single exchange evaluation per timestep** that updates both domains simultaneously. The implementation at HEAD (`8b00f31`) deviates: `swe2d_gpu_step` (lines 5719–5740) **zeros `d_ext_struct_flux_*` and re-evaluates** the unified face flux, discarding the pipe1d advance's contribution and applying only the smaller post-drain value to the 2D domain. This causes the 2D to receive only 1/3–1/2 of the pipe's outflow.

---

## 2. Architecture (Single-Exchange)

### 2.1 Per-timestep sequence

```
apply_native_device_sources(t, dt)
  │
  ├── swe2d_pipe1d_step(dev, dt, ..., solver_dev)
  │     │
  │     ├── Godunov stage 0
  │     │     ├── unified_face_flux(dev, dt, g, ..., solver_dev->d_h, ..., solver_dev->n_cells)
  │     │     │     ├── SURFACE_2D_PIPE_END face (class 3):
  │     │     │     │     direct_inject = true (n_cells_2d > 0)
  │     │     │     │     face_F_h[k] = 0 (skip fold)
  │     │     │     │     atomicAdd(&d_ext_struct_flux_h[2d_cell], fh)  ← 2D is credited
  │     │     │     │     atomicAdd(&d_A[pipe_cell], -fh * dt / L)     ← pipe is debited
  │     │     │     └── ...other face classes...
  │     │     └── godunov_update_kernel(...)  ← pipe internal dynamics
  │     │
  │     └── Godunov stage 1 (RK2 midpoint)
  │           ├── unified_face_flux(dev, dt, g, ..., solver_dev->d_h, ..., solver_dev->n_cells)
  │           │     └── same as above (ext_struct_flux accumulates across stages)
  │           └── godunov_update_kernel(...)
  │
  │   ← ext_struct_flux now holds cumulative exchange Q_total = Σ(Q_stage0, Q_stage1)
  │   ← pipe d_A has been corrected for exchange
  │
  └── swe2d_gpu_step(dev, t, dt, ...)   ← THE 2D SOLVER STEP
        │
        ├── [REMOVED] memset(d_ext_struct_flux, 0, ...)   ← WAS THE BUG
        ├── [REMOVED] swe2d_gpu_apply_unified_face_flux(...) ← WAS THE BUG
        │
        ├── Kernel 1: FLUX
        ├── Kernel 2: UPDATE
        │     └── reads d_ext_struct_flux_* → applies to h/hu/hv at each RK stage
        └── ...remaining kernels (CFL, diagnostics)
```

### 2.2 Single exchange rule

The exchange is evaluated **exactly once per pipe1d Godunov stage** (2 stages for RK2, which is the production path). The unified face kernel writes **both** `d_ext_struct_flux_*` (2D credit) and `d_A[pipe_cell]` (pipe debit) via atomicAdd in the same kernel call. The same `fh` value produces matched ± changes in both domains.

The 2D solver step **never re-evaluates** the exchange. It reads `d_ext_struct_flux_*` as written by the pipe1d advance and applies the constant source at each RK stage. The RK combine formula produces the correct total mass transfer (`dt * Q_total / area`) regardless of stage count:

| Scheme | Stages | Each applies | Combine | Net mass added to 2D cell |
|--------|--------|-------------|---------|--------------------------|
| Euler  | 1      | `dt·Q/A`   | identity | `dt·Q/A` |
| RK2    | 2      | `dt·Q/A`   | `0.5·(h0 + h2)` | `dt·Q/A` |
| RK3    | 3      | `dt·Q/A`   | `⅓·h0 + ⅔·h2 + ⅔·dt·Q/A` | `dt·Q/A` |
| RK4    | 4      | `dt·Q/A`   | `⅛·(7·h0 + h4) + 3·dt·Q/A` | `dt·Q/A` |
| RK5    | 6      | `dt·Q/A`   | standard CK45 | `dt·Q/A` |

### 2.3 `direct_inject` flag

The unified face kernel (line 2415 of `pipe1d.cu`) uses the `n_cells_2d` parameter to set `direct_inject`:

```cpp
const bool direct_inject = (n_cells_2d > 0);
```

When called from the pipe1d advance with `solver_dev` (n_cells_2d > 0):  
- `face_F_h[k] = 0.0` (skip fold pipeline to avoid double-count)  
- `atomicAdd(&d_ext_struct_flux_*[2d_cell], fh)` (2D credit)  
- `atomicAdd(&d_A[pipe_cell], -fh * dt / L)` (pipe debit)

When called WITHOUT `solver_dev` (n_cells_2d = 0, legacy/test paths):  
- `face_F_h[k] = fh` (feeds fold+godunov pipeline)  
- NO ext_struct_flux write (buffers may be nullptr)  
- NO d_A direct atomicAdd

### 2.4 What was the implementation gap

At HEAD (`8b00f31`), `swe2d_gpu_step` (lines 5729–5739) zeros `d_ext_struct_flux_*` and calls the unified face flux independently:

```cpp
// THIS IS THE BUG — lines 5729–5739 of swe2d_gpu.cu:
cudaMemsetAsync(dev->d_ext_struct_flux_h, 0, sz, dev->d_stream);
cudaMemsetAsync(dev->d_ext_struct_flux_hu, 0, sz, dev->d_stream);
cudaMemsetAsync(dev->d_ext_struct_flux_hv, 0, sz, dev->d_stream);
swe2d_gpu_apply_unified_face_flux(dev, dt, g, ...);
```

This re-evaluates the exchange using the **post-pipe1d** pipe state, producing a smaller Q than the pipe1d advance computed. The pipe1d's contribution is wiped. Net effect: 2D receives `Q_post_drain` instead of `Q_pre_drain`, losing 1/3–1/2 of the expected mass.

---

## 3. Changes Required

### 3.1 C++ pipe1d.cu — Accept solver_dev in Godunov step

Already implemented (and reverted) in prior session. Re-apply:
- `swe2d_pipe1d_godunov_step_internal` accepts optional `SWE2DDeviceState* solver_dev`
- When `solver_dev` is valid and has `n_cells > 0 && d_h && d_cell_zb`, pass 2D arrays + ext_struct_flux to `swe2d_gpu_apply_unified_face_flux`
- Otherwise pass nullptr (legacy path with `direct_inject = false`)

### 3.2 C++ pipe1d.cuh — Update declaration

Add `SWE2DDeviceState* solver_dev = nullptr` parameter to `swe2d_pipe1d_step`.

### 3.3 C++ swe2d_bindings.cpp — New binding + param

- Add `swe2d_get_solver_dev_ptr` binding that extracts `ps->solver->dev` as int64
- Add `solver_dev_ptr` to `swe2d_pipe1d_step` binding lambda

### 3.4 C++ swe2d_gpu.cu — Remove re-evaluation (THE KEY FIX)

Replace the block at lines 5719–5740 with a simple allocation guard:

```cpp
if (dev->pipe1d.n_faces > 0 && dev->pipe1d.d_flux_Q_scratch && dev->d_h && dev->d_cell_zb) {
    swe2d_gpu_alloc_ext_struct_flux(dev, n_cells);
}
```

No memset, no unified face flux call. The pipe1d advance already wrote the exchange.

### 3.5 Python swe2d/runtime/backend.py — Add getter

```python
def get_solver_dev_ptr(self) -> int:
    if self._solver_h is None:
        return 0
    return int(self._mod.swe2d_get_solver_dev_ptr(self._solver_h))
```

### 3.6 Python swe2d/runtime/coupling.py — Pass solver_dev_ptr

In `apply_native_device_sources`, before `swe2d_pipe1d_step`:
```python
_solver_dev_ptr = int(self._backend.get_solver_dev_ptr()) if self._backend else 0
```
Append `solver_dev_ptr=_solver_dev_ptr` to the call.

### 3.7 Python swe2d/runtime/coupling.py — build_coupling_controller

Accept `backend=None` parameter and forward to `SWE2DCouplingController(backend=backend, ...)`.

### 3.8 Python swe2d/workbench/workers/simulation_worker.py

Append `backend=backend` to `build_coupling_controller(...)` call.

---

## 4. Mass-Conservation Verification

The single-exchange architecture guarantees exact mass conservation because the same `fh` value is used in:
- `atomicAdd(&d_ext_struct_flux_h[2d_cell], +fh)` — 2D gains
- `atomicAdd(&d_A[pipe_cell], -fh * dt / L)` — pipe loses

The fold-and-Godunov pipeline handles the pipe's internal flux. The `d_ext_struct_flux_*` buffers are the communication channel to the 2D update kernel. No other path modifies these buffers during a timestep.

Mass balance check (per timestep):

```
pipe_mass_loss = Σ(|fh| > 0) fh · dt / L_pipe  (all SURFACE_2D faces)
2d_mass_gain   = Σ(d_ext_struct_flux_h) · dt     (all 2D cells that received)
|pipe_mass_loss - 2d_mass_gain| = 0  (identical atomicAdd values)
```

---

## 5. Out of Scope

- **Face class-4 (SURFACE_2D_INLET)** and class-5 (SURFACE_2D_JUNCTION_OVERFLOW) source-sink coupling — these paths write to `d_ext_struct_flux_h` independently and are not affected by this change. They also benefit from the removal of the zero+memset in `swe2d_gpu_step`.
- **`swe2d_gpu_alloc_ext_struct_flux` re-entrancy** — the function guards against double-allocation internally.
- **Non-production RK schemes** — RKF1, the dormant diffusion-wave path, and `solver_mode` variations are unchanged.

---

## 6. Spec Gap: Pipe-Side Momentum Debit (discovered 2026-07-22)

### 6.1 Gap description

The `direct_inject` path (added in this session) updates `d_A[pipe_cell]` via atomicAdd for mass conservation, but the corresponding **momentum** carried by the outflowing water was never subtracted from `d_Q[pipe_cell]`. The original spec (`pipe1d_face_indexed_refactor_plan.md` §4.2.2) only mentions the `d_A` atomicAdd — the `d_Q` debit was overlooked.

Without the `d_Q` debit:
- The Godunov update (line 2930: `flux_mom_div = -flux_mom[c] / L`) sees **no momentum flux** from the SURFACE_2D face because `face_F_Q[k]` was set to 0 (line 2417).
- The fold kernel at line 2675 (`atomicAdd(&cell_flux_mom[L], +face_F_Q[k])`) correctly accumulates `face_F_Q` for all face classes including SURFACE_2D — but the value was always 0.
- The pipe end cell's `d_Q` readback shows the pre-exchange momentum (Godunov computed from INTERIOR faces only), while `d_A` has been correctly reduced by the atomicAdd. The diagnostic `link_q` (read from `d_Q`) therefore reports a near-zero outflow, even though the overlay on the 2D side shows the correct flow.

### 6.2 Fix

Set `face_F_Q[k]` to the advective momentum flux of the outflowing water through the SURFACE_2D face:

```cpp
// pipe1d.cu:2417 (SURFACE_2D_PIPE_END face, class 3)
// uL_p = Q_p / A_p is already computed at line 2308
face_F_Q[k] = fh * uL_p;
```

The fold kernel (line 2675) accumulates this into `cell_flux_mom[L]`. The Godunov update (line 2930) then computes:

```
flux_mom_div = -flux_mom[c] / L
```

For outflow (`fh > 0`), `flux_mom[L]` gets `+fh * uL_p`, so `flux_mom_div = -fh * uL_p / L`. The momentum equation becomes:

```
Q_new = Q_curr + (flux_mom_div + src_gravity - g * A * Sf) * dt / (1 + gamma * dt)
```

The `-fh * uL_p / L * dt` term correctly reduces `d_Q` by the momentum that left with the outflow.

This fix does not affect the `face_F_h` value (still 0 for `direct_inject = true`) — the mass debit is already handled by the `d_A` atomicAdd. The fix is only to `face_F_Q`.

### 6.3 Effect on diagnostics

After the fix, `link_q` (read from `d_Q` via `readback_cell_state`) correctly reflects the post-exchange pipe momentum. The diagnostic `drainage_max_link_flow` will show the outflow magnitude. The 2D overlay was already correct and is unaffected.
