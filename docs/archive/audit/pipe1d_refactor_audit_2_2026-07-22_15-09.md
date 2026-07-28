---
type: audit
status: complete
created: 2026-07-22
completed: 2026-07-25
---

# Drainage Network Coupling Audit — 2026-07-22

**Investigation depth:** Full data flow from Python config → coupling.py → C++ mesh
build → unified face-flux kernel → 2D solver update. Trigger: user reports "zero
success getting drainage coupling working correctly" after the face-flux refactor.

---

## P0 — The coupling flow itself is broken (zero drainage→2D mass transfer)

No single "the bug." The coupling path has a **stacked failure chain** that
guarantees drainage↔2D coupling produces no (or unphysical) water exchange.

| # | Issue | Location | Impact |
|---|-------|----------|--------|
| 1 | **`d_ext_struct_flux_*` never zeroed before pipe1d accumulates into it** | `pipe1d.cu:2451-2453` `atomicAdd(&d_ext_struct_flux_h[R], fh)` — but `swe2d_gpu_apply_unified_face_flux` is called without first zeroing these buffers on the hot path. The only zeroing is inside `swe2d_gpu_compute_coupling_full_on_device` (culvert face-flux path). | Stale flux from previous steps accumulates. Even correctly computed `fh` values are wrong because the accumulator was not reset. |
| 2 | **`TestPipeEndExchange` completely skipped** | `tests/test_swe2d_gpu_drainage_network.py:908` — test gated on `hasattr(_MOD, "swe2d_gpu_apply_pipe_end_bc")` which was removed in the refactor. **Zero test coverage for SURFACE_2D_PIPE_END.** | No regression detection. If/when the path starts, nobody will notice it's wrong. |
| 3 | **Mass conservation never verified post-refactor** | The PIPE1D_AUDIT (2026-07-17) found −90.9% mass loss in 200 steps for diffusion, −73.3% for fully-dynamic. The refactor removed the `d_node_net_q`/`d_node_depth` arrays but **never re-validated conservation** with the unified face-flux kernel. | The original mass leak is almost certainly still present, just manifesting differently. |
| 4 | **`swe2d_gpu_compute_coupling_full_on_device` still compiled + exported** | `swe2d_bindings.cpp:1153` — the old source-term coupling binding still exists. If any code path calls it, it double-counts with the face-flux path. | Silent over-counting if both paths run. |

### Root cause chain

1. Face flux refactor → removed old BC kernels → no replacement test
2. `ext_struct_flux` buffers not zeroed on the pipe1d hot path
3. Original mass conservation bugs never confirmed fixed with new architecture

---

## P1 — Active physics defects (confirmed in current code)

| # | Issue | Location |
|---|-------|----------|
| 5 | **Wave speed `c_wave = sqrt(g·|ΔH|/L)` is dimensionally wrong** — units are m^0.5/s, not m/s. Uses head *difference* and sub-cell length instead of hydraulic depth `sqrt(g·A/T)`. | `pipe1d.cu` interior flux kernel |
| 6 | **Junction surcharge clamp destroys mass** — `swe2d_drainage_apply_delta_kernel` clamps `d = fmin(d, node_max_depth[i])` on every call. Junction surcharge above rim is silently deleted. | `swe2d_gpu.cu:4531` |
| 7 | **Outfall nodes are mass sinks** — Zero-storage outfalls force `node_depth = 0`, pipe water vanishes from system instead of wetting the 2D cell. | `swe2d_gpu.cu:9188-9200` (body retired but name still exported) |
| 8 | **First-substep boundary flux reads `cell_y == 0` as WSE** — interior face boundary branch reads cell_y without a zero-guard. First-substep cell_y is memset to 0, producing huge spurious c_wave and boundary flux. | `pipe1d.cu` interior face flux (finding #3 from original PIPE1D_AUDIT) |

---

## P2 — Code spaghetti / compat shims (actively hiding bugs)

| # | Issue | Location |
|---|-------|----------|
| 9 | **coupling.py readback still uses old binding name** — `swe2d_pipe1d_readback_cell_state` at line 1514 is the OLD API name. | `swe2d/runtime/coupling.py:1514` |
| 10 | **Unified face kernel passes face_nx/face_ny = 0 for SURFACE_2D_PIPE_END** — workaround uses `dir_k` as scalar normal, but HLLC then hardcodes `nx=dir_k, ny=0.0`, which is only correct for pipes aligned with the x-axis. | `pipe1d.cu:2313-2320` |
| 11 | **`swe2d_gpu_upload_outfall_free_bc_nodes` is a no-op** — body retired (comment at line 9194), but still called from coupling.py line 1752. Silently does nothing. | `swe2d_gpu.cu:9188-9200`, `coupling.py:1751-1754` |
| 12 | **`swe2d_pipe1d_step` ignores `solver_mode` parameter** — line 3688-3691: all three mode params are `(void)`-cast. UI values accepted but discarded. | `pipe1d.cu:3689-3691` |
| 13 | **`max_cell_length` truncated to int32** — `pipe1d.cuh:264` declares `int32_t`. Values like 0.5 m → 0, disabling subdivision. | `pipe1d.cuh:264` |
| 14 | **Elliptical `A_open` table formula inverted** — Returns complement of filled area. | `pipe1d.cu:559-577` |
| 15 | **`swe2d_gpu_fold_culvert_mass_to_source` still called** from coupling.py line 1850 in the non-face-flux fallback. If culvert face-flux is on, this path should never run — but the guard at line 1848 checks `self.culvert_face_flux_mode != "face_flux"` which may not match the device flag `dev->use_culvert_face_flux`. | `coupling.py:1848-1855` |

---

## P3 — Architecture regression from the face-flux refactor

| # | Issue |
|---|-------|
| 16 | **Compat shims were removed from coupling.py** (Phase 5a), but the C++ side was never verified end-to-end. The tests that would verify it are all skipped. |
| 17 | **Old bindings still compiled** — `swe2d_build_pipe1d_mesh`, `swe2d_pipe1d_upload_node_depth`, `swe2d_pipe1d_upload_pipe_ends_and_junctions`, `swe2d_pipe1d_upload_outfall_state`, etc. — all still exported from `swe2d_bindings.cpp`. Dead paths that could be accidentally invoked. |
| 18 | **No end-to-end test of coupling.py** — `tests/test_swe2d_gpu_drainage_network.py` calls the native module directly, never through `SWE2DCouplingController.apply_native_device_sources()`. So the actual production code path is never exercised by any automated test. |

---

## Recommended fix order

### Fix 1 — Zero ext_struct_flux before pipe1d step (P0#1)

In `swe2d_gpu_apply_unified_face_flux` in `pipe1d.cu`, add a `cudaMemsetAsync`
of `d_ext_struct_flux_h/hu/hv` to zero before the unified kernel runs. This
alone may fix the "no coupling" symptom.

### Fix 2 — Restore `TestPipeEndExchange` (P0#2)

Port the test in `tests/test_swe2d_gpu_drainage_network.py:908`:
- Remove the `hasattr` gate on the removed binding
- Call `swe2d_pipe1d_step` with a `solver_dev` pointer so SURFACE_2D_PIPE_END
  faces write to `d_ext_struct_flux_*`
- Verify water moves from pipe→2D cell

### Fix 3 — Verify mass conservation (P0#3)

Add a closed-system test: 2 pipe nodes, 1 link, 2 surface cells, run 200 steps.
Assert `Δ(pipe_volume + surface_volume) ≈ 0`.

### Fix 4 — Delete old bindings (P3#17)

Remove `swe2d_gpu_compute_coupling_full_on_device`,
`swe2d_build_pipe1d_mesh`, `swe2d_pipe1d_upload_node_depth`,
`swe2d_pipe1d_upload_pipe_ends_and_junctions`, `swe2d_pipe1d_upload_outfall_state`,
and `swe2d_gpu_upload_outfall_free_bc_nodes` from `swe2d_bindings.cpp`.

### Fix 5 — Fix c_wave (P1#5)

Replace `sqrt(g·|ΔH|/L)` with `sqrt(g·A/T)` (hydraulic depth) in the interior
pipe-face flux kernel.

### Fix 6 — Fix surcharge clamp (P1#6)

`swe2d_drainage_apply_delta_kernel` must not clamp surcharge volume. Overflow
should be the only exit path for volume above rim.

### Fix 7 — Fix outfall sink (P1#7)

Outfall nodes must pass pipe outflow into the 2D cell when coupled — use the
SURFACE_2D_PIPE_END face path instead of the zero-storage node_depth reset.

### Fix 8 — Add coupling.py end-to-end test (P3#18)

Add `tests/test_coupling_unified_mesh.py` that exercises
`SWE2DCouplingController.apply_native_device_sources()` with a real 2D solver
and a coupled pipe network. This is the only way to validate the production path.

---

## Files to modify (summary)

| File | What |
|------|------|
| `cpp/src/pipe1d.cu` | Zero ext_struct_flux before unified kernel; fix c_wave; add face_nx/ny for SURFACE_2D_PIPE_END |
| `cpp/src/swe2d_gpu.cu` | Fix surcharge clamp; fix outfall sink; delete retired functions |
| `cpp/src/swe2d_bindings.cpp` | Remove old binding exports |
| `swe2d/runtime/coupling.py` | Fix readback binding name; remove no-op outfall upload call |
| `tests/test_swe2d_gpu_drainage_network.py` | Port TestPipeEndExchange to new API; remove dead hasattr gates |
| `tests/test_coupling_unified_mesh.py` | New: end-to-end coupling.py test |

---

## P4 — Unit-system bugs in the drainage coupling path

### Finding U1 — CRITICAL: Unit inversion in `build_coupling_controller`

**File:** `swe2d/runtime/coupling.py:2316-2318, 2331`

```python
_ls = max(1.0e-6, float(length_scale_si_to_model))      # e.g., 0.3048 for USC
_si_m_per_model = 1.0 / _ls                               # 3.281 ← INVERTED!
_model_to_ft = _u.USC_FT_PER_SI_M * _si_m_per_model       # 10.76 ← should be 1.0
...
length_scale_si_to_model=_si_m_per_model,                  # passed to SWE2DCouplingController
```

`length_scale_si_to_model` is already `si_m_per_model` (0.3048 for feet). Line 2317
inverts it, producing `3.281`. This inverted value flows into `_u.configure()`
inside `SWE2DCouplingController.__init__`, setting `_u.gravity() ≈ 2.99 ft/s²`
instead of `32.17 ft/s²`. Every coupling calculation (inlet capture, pipe flow,
weir/orifice, culvert hydraulics) is dimensionally wrong.

**Impact:** Catastrophic for USC projects. Harmless for SI (1.0/1.0 = 1.0).

**Fix:** Remove the inversion. Use `_u` accessors directly:

```python
_ls = max(1.0e-6, float(length_scale_si_to_model))
_u.configure(_ls)
_model_to_ft = _u.model_to_ft()  # for SWE2DStructureModule (currently unused)
```

### Finding U2 — Medium: Default gravity values hardcoded to SI

| File | Line | Default | Should be |
|------|------|---------|-----------|
| `swe2d/extensions/extension_models.py` | 398 | `gravity: float = 9.81` | `_u.gravity()` (deferred) |
| `swe2d/workbench/services/pipe_network_service.py` | 103 | `gravity: float = 9.81` | `_u.gravity()` (deferred) |
| `swe2d/extensions/drainage_network.py` | 217 | `SI_GRAVITY` fallback (9.80665) | `_u.gravity()` fallback |

All callers in the production path override these with `_u.gravity()`, so these
defaults are latent traps for new code paths rather than active bugs.

### Finding U3 — Low: `backend.py` default arguments evaluated at import

**File:** `swe2d/runtime/backend.py:935-936`

```python
g:        float = _u.gravity(),   # evaluated at module import time
k_mann:   float = 1.0,            # SI Manning factor
```

`_u.gravity()` as a default argument is evaluated before `_u.configure()` is
called, so it returns the fallback `SI_GRAVITY = 9.80665`. For USC this is wrong,
but callers always override. `k_mann = 1.0` is the SI Manning factor — also always
overridden.

### Key C++ verification

The C++ side (`swe2d_gpu.cu`, `swe2d_gpu.cuh`) uses in-class member initializers
with SI defaults (9.81, 1.0, 3.28084) but all are **always overridden** via
`swe2d_gpu_set_k_mann`, `swe2d_gpu_preload_structure_params`, etc. before any
computation runs. The C++ structure-flows kernel correctly converts model units
to feet, computes in USC, and converts back. **No active C++ unit bugs found.**

---

## P5 — Structure/Drainage coupling entanglement

### Overview

The two paths share **three buffers** and **one control-flow function** but are
otherwise architecturally separated (separate workspace structs, separate
parameter arrays, class-based dispatch in the unified face kernel).

### S1 — SHARED_BUFFER: `d_ext_struct_flux_h/hu/hv` — the primary merge point

**Severity:** HIGH — both paths write to the same 3 buffers via `atomicAdd`.

| Writer | Domain | Kernel | File:Line |
|--------|--------|--------|-----------|
| Class 3 (SURFACE_2D_PIPE_END) | Drainage | `swe2d_unified_face_flux_kernel` | `pipe1d.cu:2451-2453` |
| Class 4 (SURFACE_2D_INLET) | Drainage | `swe2d_unified_face_flux_kernel` | `pipe1d.cu:2584-2586` |
| Class 5 (SURFACE_2D_JUNCTION_OVERFLOW) | Drainage | `swe2d_unified_face_flux_kernel` | `pipe1d.cu:2635` |
| Class 6 (CULVERT) * | Structure | `swe2d_unified_face_flux_kernel` | `pipe1d.cu:2665-2666` |
| Culvert face-flux kernel * | Structure | `swe2d_culvert_face_flux_kernel` | `swe2d_gpu.cu:7778` |

_* Class-6 never fires (see S3). The culvert face-flux kernel is only called from
the retired `compute_coupling_full_on_device`._

Both paths are **additive** (correct by design — drainage + structure
contributions should sum for the 2D cell). But this means:
- Neither path can be modified, disabled, or tested independently without risk
- The buffer must be zeroed exactly once per step before any path writes

### S2 — SHARED_BUFFER: `d_external_source_mps` — legacy merge for source-term path

**Severity:** MEDIUM — only used when non-culvert structures are present in the
non-face-flux fallback. The function that zeroes it (`compute_coupling_full_on_device`)
is not called from the hot path, risking stale accumulations.

### S3 — ORDER_DEPENDENCY: `apply_native_device_sources` control flow

**File:** `swe2d/runtime/coupling.py:1647-1919`

Execution order in the hot path:

```
1. _ensure_persistent_coupling_preloaded     → uploads structure params + cell_area
2. swe2d_pipe1d_step()                        → runs unified face kernel for class 3/4/5
3. _ensure_culvert_face_flux_preloaded        → uploads culvert face geometry
4. Set coupling dt
5. If non-face-flux: fold culvert mass
6. If face-flux: redistribute face flux
7. On-device redistribution of structure sources
```

**Critical gap: `swe2d_gpu_compute_coupling_full_on_device` is NOT called.**
The comment at line 1821 states it is "retired." This function was responsible for:
- Computing structure flows via `swe2d_compute_structure_flows_kernel`
- Running `swe2d_culvert_face_flux_kernel` for class-6 CULVERT faces
- Folding non-culvert structure sources into `d_external_source_mps`
- Zeroing `d_ext_struct_flux_*` and `d_external_source_mps`

**Consequence:** Class-6 CULVERT faces in the unified face kernel **never fire**
because `d_structure_flows` is always passed as `nullptr` (`pipe1d.cu:3467`).
The class-6 code returns early (`pipe1d.cu:2649`). If structures are in use with
face-flux mode enabled, their flows are not being applied to the 2D solver.

### S4 — CONTROL_FLOW: Unified face kernel class dispatch

**Severity:** LOW (cleanly separated).

The unified face kernel at `pipe1d.cu:1915` uses a two-pass design:
- **Pass 1** (line 3442): classes 3, 4, 5, 6 (SURFACE_2D_* + CULVERT) — writes to
  `d_ext_struct_flux_*`
- **Pass 2** (line 3489): classes 0, 1, 2 (INTERIOR, OUTFALL_BC, INLET_BC) — writes
  to `face_F_h`/`face_F_Q`

Classes are separated by `if (cls == N)` branches within each pass. No shared
mutable state between classes within a single thread. `cudaStreamSynchronize`
between passes prevents data races.

### S5 — POTENTIAL_DATA_DEPENDENCY: `d_ext_struct_flux_*` zeroing lifecycle

**Severity:** HIGH — the zeroing path is missing from the drainage-only hot path.

| Where zeroed | Called from hot path? |
|--------------|----------------------|
| `swe2d_gpu.cu:7690` (in `compute_coupling_full_on_device`) | **NO** — function retired |
| `swe2d_gpu.cu:8033` (in `apply_culvert_face_flux`) | **NO** — function not called |
| Nowhere in `apply_native_device_sources` | **The hot path does NOT zero** |

The buffer is allocated via `cudaMalloc` which initializes to zero on first
allocation, but on subsequent steps the previous step's values persist.

### S6 — MINOR: `sf_ws.d_cell_wse` allocated even for drainage-only

**Severity:** LOW.

`swe2d_gpu_preload_coupling_cell_area` in `coupling.py` allocates
`sf_ws.d_cell_wse` even when no structures are present. The drainage path uses
its own `drain_ws.d_cell_wse` instead. This is a harmless convenience allocation.

### S7 — POTENTIAL_BUG: structure flow computation is dead code

**Severity:** CRITICAL (if structures are used with face-flux).

Structure flows are computed by `swe2d_gpu_compute_coupling_full_on_device` which
is NOT called from the Python hot path. The only remaining structure application
path is:
- Non-culvert structures → `swe2d_coupling_structure_source_kernel` (inside the
  retired function → dead)
- Culvert face-flux → `swe2d_culvert_face_flux_kernel` (inside the retired
  function → dead)
- Class-6 in unified face kernel → never fires (`d_structure_flows` is null)

**If structures were working before the refactor, they stopped working when
`compute_coupling_full_on_device` was removed from the hot path.**

### Summary

| # | Vector | Severity | Detail |
|---|--------|----------|--------|
| S1 | `d_ext_struct_flux_*` | **HIGH** | Both paths `atomicAdd` into same 3 buffers. Correctly additive but neither path can be independently modified. |
| S2 | `d_external_source_mps` | **MEDIUM** | Shared legacy merge buffer. Zeroing function not called from hot path. |
| S3 | Control flow in `apply_native_device_sources` | **CRITICAL** | `compute_coupling_full_on_device` is "retired" — structure flows are never computed. Class-6 CULVERT faces never fire. |
| S4 | Unified kernel class dispatch | **LOW** | Cleanly separated by `if (cls == N)`. Two-pass sync is correct. |
| S5 | Buffer zeroing lifecycle | **HIGH** | `d_ext_struct_flux_*` never zeroed in the drainage-only hot path. |
| S6 | `sf_ws.d_cell_wse` for drainage-only | **LOW** | Convenience allocation, not a functional entanglement. |
| S7 | Structure flow computation dead | **CRITICAL** | The only code path that computes and applies structure flows is not called. |
