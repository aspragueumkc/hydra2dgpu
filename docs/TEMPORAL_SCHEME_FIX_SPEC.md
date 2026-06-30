# Fix RK2 Stale Coupling Sources + Implement Higher-Order Temporal Schemes

**Status:** Draft (Revised — oversights addressed)  
**Date:** 2026-06-30  
**Author:** opencode  

**Revision scope (this pass):**
- Reconcile `TemporalScheme` value set across Python enum, GUI combo, and `run_service` validator (Phase 0 prerequisite).
- Use the already-allocated `d_stage_cell_source_mps` / `d_stage_edge_bc*` slot buffers instead of overwriting `d_external_source_mps` between stages.
- Lock RK5 to the **RK5(4) Cash-Karp embedded** tableau (k1, k3, k4, k6 stored separately) — matches the existing `swe2d_rk5_graph_combine_kernel`.
- Extend the CUDA-graph `time_integrator` whitelist from `{2, 4}` to `{2, 3, 4, 5, 6}`.
- Fix the `dpl_ws` → `drain_ws` typo in `swe2d_recompute_coupling_for_stage` and add `swe2d_gpu_set_coupling_dt` call inside the wrapper.
- Expand Phase 6 file list to cover `run_service.py`, `coupling.py`, `non_gui_runtime_service.py`, plus C++ IMEX helper cleanup.
- Document hydrograph BC handling per stage, tiny-mode interaction with higher order, and `tiny_mode + temporal_order > 2` fallback.

---

## 1. Problem Statement

### 1.1 Stale Coupling Sources in SSP-RK2

When structures (culverts, weirs, orifices) or drainage networks are active, the coupling sources (`d_external_source_mps`, `d_ext_struct_flux_h/hu/hv`) are computed **once** before the RK2 step and consumed by **both** stages. The second stage uses sources evaluated at `U^n` instead of `U*` (the Stage 1 result).

**Impact:** Nonlinear source terms (culvert flow ∝ √Δh, weir flow ∝ h^1.5) create an O(dt) splitting error that compounds across timesteps, causing instability with 2nd-order temporal schemes. Boundaries (OPEN, NORMAL_DEPTH_SLOPE) amplify the error through positive feedback loops.

### 1.2 Higher-Order Temporal Schemes Not Implemented

The UI offers Euler(1), RK2(2), RK3(3), RK4(4), Graph-safe RK4(5), Graph-safe RK5(6). But the C++ dispatch (`swe2d_solver.cpp:195`) routes **everything ≥ 2** through `swe2d_gpu_step_rk2()`. The `swe2d_gpu_step_rk4` was removed as dead code. `swe2d_rk5_graph_combine_kernel` exists but is never called.

---

## 2. Decisions (Resolved)

| Question | Decision |
|----------|----------|
| Python IMEX path | **Remove it.** C++ handles source freshness internally. |
| CFL auto-scaling | **No.** Add recommended CFL values per scheme in the GUI tooltip only. |
| Persistent chunking for higher orders | **Explore.** Extend if not high effort; fall back to baseline loop otherwise. |
| RK3 variant | **SSP-RK3 (Shu-Osher).** GPU-friendly, SSP property, CFL=1.0. |
| CUDA graphs with higher order | **Yes, possible.** Per-stage capture — issue is a sync call + whitelist, not fundamental. |
| RK5 tableau | **RK5(4) Cash-Karp embedded.** Use existing `swe2d_rk5_graph_combine_kernel` (k1, k3, k4, k6 stored separately). |
| Temporal scheme value set | **`{1, 2, 3, 4, 5, 6}` contiguous** — fix Python enum + GUI combo + service validator as Phase 0. |
| RK4 dispatch (4 vs 5) | **Both route to `swe2d_gpu_step_rk4`.** GUI label differs; algorithm identical. |
| Stage source/BC storage | **Reuse `d_stage_cell_source_mps` / `d_stage_edge_bc*` slot arrays** (already allocated, currently unused). |
| Hydrograph BCs per stage | **Host-snapshot per stage** into `d_stage_edge_bc[i]` before each stage's graph launch. |

---

## 3. Prerequisites — Value-Set Reconciliation

The three layers that map user-facing "temporal scheme" to the C++ `temporal_order` integer do not currently agree. This must be fixed **before** any C++ dispatch lands, otherwise the user picks an option and the Python validator rejects it (or the GUI hides it).

### 3.1 Current state (broken)

| Layer | File | Accepted values | Missing |
|-------|------|-----------------|---------|
| Python enum | `swe2d/extensions/extension_models.py:30-36` | `1, 2, 3, 5, 6` | `4` (Classic RK4) |
| GUI combo | `swe2d/workbench/views/model_tab_view.py:497-504` | `1, 2, 4, 5, 6` | `3` (SSP-RK3) |
| Service validator | `swe2d/workbench/services/run_service.py:144` (`_VALID_TEMPORAL_SCHEMES = {0, 1, 2, 3, 4}`) | `0, 1, 2, 3, 4` | `5, 6` (Graph-safe RK4/RK5) |

Concretely: a user selecting "Graph-safe RK5" today is rejected by the service layer. Selecting "RK3" from the enum fails because the GUI doesn't expose it. Selecting "Classic RK4" via the GUI fails because the enum doesn't have `4`.

### 3.2 Required resolution

The supported scheme set is **`{1, 2, 3, 4, 5, 6}`** (six contiguous values, no gap). The mapping is:

| Value | Name | C++ routine | Notes |
|-------|------|-------------|-------|
| 1 | Euler (RK1) | `swe2d_gpu_step` | existing |
| 2 | SSP-RK2 (Heun) | `swe2d_gpu_step_rk2` | fixed (Phase 1) |
| 3 | SSP-RK3 (Shu-Osher) | `swe2d_gpu_step_rk3` | **new** (Phase 3) |
| 4 | Classic RK4 | `swe2d_gpu_step_rk4` | **new** (Phase 4) |
| 5 | Graph-safe RK4 (true staged) | `swe2d_gpu_step_rk4` | alias for `4` |
| 6 | Graph-safe RK5 (Cash-Karp RK5(4)) | `swe2d_gpu_step_rk5` | **new** (Phase 5) |

**Why are 4 and 5 both routed to `swe2d_gpu_step_rk4`?** Because option 5 ("Graph-safe RK4") is functionally the same algorithm as option 4, just exposed with the wording the user expects from a graph-safe scheme. The GUI label differentiates them so power-users can see what they're getting, but the implementation is identical. If/when a genuinely distinct graph-safe RK4 variant is designed (e.g., 8-stage low-storage RK4), value 5 can be re-targeted then.

### 3.3 Concrete edits required (Phase 0)

1. **`extension_models.py`** — add `CLASSIC_RK4 = 4` to `TemporalScheme` enum (slot it between `SSP_RK3 = 3` and `GRAPH_SAFE_RK4 = 5`).
2. **`model_tab_view.py:497-504`** — insert `("RK3 (SSP Shu-Osher, 3rd-order)", 3)` between the RK2 and RK4 entries. Update tooltip per §5.10.
3. **`constants_service.py:54-58`** — add RK3 entry to the temporal scheme combo data table.
4. **`run_service.py:144`** — change `_VALID_TEMPORAL_SCHEMES = {0, 1, 2, 3, 4}` to `_VALID_TEMPORAL_SCHEMES = {1, 2, 3, 4, 5, 6}` (drop `0`, add `5, 6`).
5. **`batch_simulation_dialog.py:73`** — verify the `temporal_order_combo` mapping table is complete (already wired to `temporal_scheme`, so no change needed once the combo widget exposes all six values).

**Note:** `0` is not a meaningful scheme; remove it from the validator to fail-fast on bad configs.

### 3.4 C++ invariant

The C++ solver config (`SWE2DSolverConfig::temporal_order` at `swe2d_solver.hpp:50`) accepts any `int`. Until Phase 3 lands, values `3, 4, 5, 6` should produce a hard error (`throw std::invalid_argument`) at `swe2d_create` time with a clear message: `"temporal_order=N not yet implemented; supported: 1, 2"` — this prevents users from selecting GUI options that silently no-op.

---

## 4. Architecture Overview

### 4.1 Current RK2 Flow

```
Python: apply_native_device_sources(t, dt)
  → swe2d_gpu_compute_coupling_full_on_device()
    → memset d_external_source_mps = 0
    → compute WSE = h + zb
    → compute structure flows → write d_external_source_mps
    → fold drainage → write d_external_source_mps
    → [face-flux: write d_ext_struct_flux_h/hu/hv]
  → stream sync

C++: swe2d_gpu_step_rk2(dev, t_now, dt, ...)
  1. Save U^n → d_h0, d_hu0, d_hv0
  2. Stage 1: swe2d_gpu_step() — reads d_external_source_mps (FRESH)
  3. [save rain CN state]
  4. Stage 2: swe2d_gpu_step() — reads d_external_source_mps (STALE!)
  5. Combine: 0.5 * (U^n + U*_stage2)
```

### 4.2 Target RK2 Flow (Fixed)

```
C++: swe2d_gpu_step_rk2(dev, t_now, dt, ...)
  1. Save U^n → d_h0, d_hu0, d_hv0
  2. Stage 1: swe2d_gpu_step() — reads d_external_source_mps (FRESH)
  3. [save rain CN state]
  4. Recompute coupling from updated state U*
     → swe2d_gpu_compute_coupling_full_on_device() (FRESH sources at U*)
  5. Stage 2: swe2d_gpu_step() — reads d_external_source_mps (FRESH at U*)
  6. Combine: 0.5 * (U^n + U*_stage2)
```

### 4.3 Target Higher-Order Flow

```
C++: swe2d_gpu_step_rkN(dev, t_now, dt, ...)   // N = 3, 4, 5
  1. Save U^n → d_h0/d_hu0/d_hv0
  2. For each stage i = 1..N:
     a. Recompute coupling from current state U^{i-1}            (swe2d_recompute_coupling_for_stage)
     b. Snapshot BCs at stage-time t_now + (i-1)*dt → d_stage_edge_bc[i] / d_stage_edge_bc_val[i]
     c. Stage i: swe2d_gpu_step() — reads fresh sources + per-stage BC snapshot
     d. Snapshot sources at this stage → d_stage_cell_source_mps[i]
     e. Save stage result (k-slope or intermediate U*) → d_h_i buffer
  3. Combine using Butcher tableau weights (final kernel reads k-slopes + U^n backup)
```

**Reuse existing infrastructure:** the per-stage slot arrays `d_stage_cell_source_mps` and `d_stage_edge_bc*` (already allocated for `SWE2D_GRAPH_STAGE_SLOTS = 6` in `swe2d_gpu.cu:4706-4708` but currently unused) are written at the host between stage launches and read inside the per-stage graph capture. This keeps `d_external_source_mps` as the "live" source buffer consumed by `swe2d_gpu_step`'s update kernel, while `d_stage_cell_source_mps[i]` provides per-stage source snapshots for diagnostics and replay validation.

**Hydrograph BCs per stage:** the hydrograph BC kernel currently runs OUTSIDE the per-step graph capture (swe2d_gpu.cu:5019-5028) because `t_now` is dynamic. For higher order, the host copies the interpolated BC value into `d_stage_edge_bc[i]/d_stage_edge_bc_val[i]` before each stage's `swe2d_gpu_step` call; the per-stage graph capture then reads from that slot (frozen at stage-launch time) instead of the live hydrograph interpolation. Static BC types (OPEN, WALL, etc.) never change, so copying them is a no-op.

---

## 5. Changes Required

### 5.1 Device State Additions (`swe2d_gpu.cuh`)

**Buffer-reuse strategy:** `d_h1/d_hu1/d_hv1`, `d_h2/d_hu2/d_hv2`, `d_h3/d_hu3/d_hv3` are already declared in `SWE2DDeviceState` (cpp/src/swe2d_gpu.cuh:146-154). Once the rain CN scratch dual-use is resolved (§5.3), these buffers are repurposed for RK3/RK4 k-slope storage. **No new allocation needed for k1/k2/k3** — they share with the existing intermediate buffers.

**Allocate only the NEW buffers** in `swe2d_gpu_init()`:

| Buffer | Purpose | Size |
|--------|---------|------|
| `d_k4_h, d_k4_hu, d_k4_hv` | RK4 k4 slope / RK5(4) k4 slope (reuses both) | 3 × n_cells × 8 bytes |
| `d_k6_h, d_k6_hu, d_k6_hv` | RK5(4) k6 slope (Cash-Karp embedded) | 3 × n_cells × 8 bytes |
| `d_rain_cn_scratch_h` | Dedicated rain CN scratch (replaces d_h1 dual-use) | n_cells × 8 bytes |
| `d_rain_cn_scratch_ex` | Dedicated rain CN scratch (replaces d_h2 dual-use) | n_cells × 8 bytes |

**Buffer mapping summary (cross-scheme):**

| Buffer | RK2 | RK3 | RK4 (and 5) | RK5(4) (and 6) |
|--------|-----|-----|-------------|-----------------|
| `d_h0/hu0/hv0` | U^n backup | U^n backup | U^n backup | U^n backup |
| `d_h1/hu1/hv1` | (unused after fix) | U* stage-2 intermediate | k1 slope | k1 slope |
| `d_h2/hu2/hv2` | (unused after fix) | U** stage-3 intermediate | k2 slope | (not stored) |
| `d_h3/hu3/hv3` | (unused after fix) | (unused) | k3 slope | k3 slope |
| `d_k4_h/hu/hv` | (unused) | (unused) | k4 slope | k4 slope |
| `d_k6_h/hu/hv` | (unused) | (unused) | (unused) | k6 slope |

**Note on RK5(4) k2 / k5:** The Cash-Karp embedded method only stores k1, k3, k4, k6 separately (matching the existing `swe2d_rk5_graph_combine_kernel` signature at swe2d_gpu.cu:2318). k2 and k5 are evaluated but their final-state contribution is folded into the next stage's input — we do not store them.

**Memory cost:** 11 NEW buffers × n_cells × 8 bytes = 88 × n_cells bytes. For 100K cells: ~8.8 MB. Acceptable. (The pre-existing `d_h1`/`d_h2`/`d_h3` and `d_stage_*` slots are reused, not added.)

**Free order:** Add the new buffers to the `safe_free(...)` block in `swe2d_gpu_free()` (cpp/src/swe2d_gpu.cu:9116).

### 5.2 RK2 Fix: Recompute Coupling Between Stages (`swe2d_gpu.cu`)

**Modify `swe2d_gpu_step_rk2()`** (lines 6181-6327):

After Stage 1 (line 6234), before Stage 2 (line 6252), insert:

```cpp
// ── Recompute coupling sources at intermediate state U* ──
if (dev->sf_ws.params_preloaded || dev->culvert_ff_ws.params_preloaded) {
    swe2d_recompute_coupling_for_stage(dev);
}
```

Same change in `swe2d_gpu_step_rk2_persistent_chunk()` between its two stages.

**Key consideration:** `swe2d_gpu_compute_coupling_full_on_device()` zeros `d_external_source_mps` at the start (line 7068), so it cleanly replaces the stale sources. It also zeros `d_ext_struct_flux_*` when face-flux mode is active (line 7137).

**Guard:** Only recompute if structures or drainage are configured. For pure-flow models, no change.

**`s_coupling_dev` safety:** The wrapper `swe2d_recompute_coupling_for_stage()` must pass `dev` explicitly to `swe2d_gpu_compute_coupling_full_on_device()`, NOT rely on the `s_coupling_dev` global. The global is set once at solver init via `swe2d_gpu_set_coupling_device_global` (cpp/src/swe2d_gpu.cu:6833) but is also used by host-side coupling utilities (coupling.py). Passing `dev` directly makes the per-stage call self-contained and avoids cross-thread races if the host coupling path runs concurrently in a future change.

### 5.3 Rain CN State Conflict Resolution (applies to all multi-stage schemes)

**Problem:** `d_h1` and `d_h2` are dual-used as rain CN scratch buffers (lines 6244-6275) AND as RK3/RK4 k-slope / intermediate buffers.

**Resolution:** Replace dual-use with dedicated `d_rain_cn_scratch_h` and `d_rain_cn_scratch_ex` buffers. Update the rain CN save/restore in `swe2d_gpu_step_rk2()`, `swe2d_gpu_step_rk2_persistent_chunk()`, AND every higher-order step function (rk3, rk4, rk5) to use the new buffers.

**Why this is needed for ALL orders, not just RK2:** For an N-stage scheme, the rain CN state must be saved before stage 1 starts and restored only at the end of stage N, so that net rain advancement matches `dt` (one full timestep). Without dedicated buffers, the save/restore would overwrite k-slopes or intermediates. The single pair `d_rain_cn_scratch_h/_ex` is sufficient for all orders because only one save/restore pair happens per outer step.

### 5.4 RK3: SSP-RK3 (Shu-Osher) — `swe2d_gpu.cu`

Add `swe2d_gpu_step_rk3()`:

```
SSP-RK3 (Shu-Osher, 3-stage, SSP-stable, CFL ≤ 1.0):
  Stage 1: U*    = U^n + dt·R(U^n)
  Stage 2: U**   = 0.75·U^n + 0.25·(U* + dt·R(U*))
  Stage 3: U^{n+1} = (1/3)·U^n + (2/3)·(U** + dt·R(U**))
```

**Pseudocode:**
```cpp
void swe2d_gpu_step_rk3(dev, t_now, dt, ...) {
    // Save U^n → d_h0/d_hu0/d_hv0 (full backup).
    swe2d_state_to_double_kernel<<<...>>>(n_cells, dev->d_h,  dev->d_h0);
    swe2d_state_to_double_kernel<<<...>>>(n_cells, dev->d_hu, dev->d_hu0);
    swe2d_state_to_double_kernel<<<...>>>(n_cells, dev->d_hv, dev->d_hv0);

    // Rain CN: save cumulative state once for the whole RK3 step.
    save_rain_cn_to_scratch(dev);

    // Stage 1: U* = U^n + dt·R(U^n)
    dev->kernel_graph_cache.time_integrator = 3;
    swe2d_recompute_coupling_for_stage(dev);                        // sources at U^n
    snapshot_edge_bc_to_stage_slot(dev, /*slot=*/1, t_now);          // stage 1 BC snapshot
    swe2d_gpu_step(dev, t_now, dt, ...);                             // produces U* in d_h/d_hu/d_hv
    copy_stage_sources(dev, /*slot=*/1);                             // → d_stage_cell_source_mps[1]

    // Stage 2: U** = 0.75·U^n + 0.25·(U* + dt·R(U*))
    swe2d_recompute_coupling_for_stage(dev);                        // sources at U*
    copy d_h → d_h1, d_hu → d_hu1, d_hv → d_hv1;                    // save U* (for blend if needed)
    snapshot_edge_bc_to_stage_slot(dev, /*slot=*/2, t_now + dt);
    swe2d_gpu_step(dev, t_now + dt, dt, ...);                        // produces U* + dt·R(U*) in state
    // Linear blend: state = 0.75·U^n + 0.25·(U* + dt·R(U*))
    swe2d_blend_kernel<<<...>>>(n_cells, dev->d_h,  dev->d_h0,  0.75, 0.25);
    swe2d_blend_kernel<<<...>>>(n_cells, dev->d_hu, dev->d_hu0, 0.75, 0.25);
    swe2d_blend_kernel<<<...>>>(n_cells, dev->d_hv, dev->d_hv0, 0.75, 0.25);
    // NB: the linear blend does NOT re-clamp wet/dry / momentum — the next swe2d_gpu_step
    // re-clamps, so partial-stage states can briefly carry non-physical values. This is
    // standard for Shu-Osher; final combine re-establishes invariants.

    // Stage 3: U^{n+1} = (1/3)·U^n + (2/3)·(U** + dt·R(U**))
    swe2d_recompute_coupling_for_stage(dev);                        // sources at U**
    copy d_h → d_h2, d_hu → d_hu2, d_hv → d_hv2;                    // save U** (for diagnostics)
    snapshot_edge_bc_to_stage_slot(dev, /*slot=*/3, t_now + 2*dt);
    swe2d_gpu_step(dev, t_now + 2*dt, dt, ...);                      // produces U** + dt·R(U**)
    copy_stage_sources(dev, /*slot=*/3);
    swe2d_rk3_combine_kernel<<<...>>>(n_cells, dev->d_h, dev->d_hu, dev->d_hv,
                                      dev->d_h0, dev->d_hu0, dev->d_hv0,
                                      1.0/3.0, 2.0/3.0, h_min);    // wet/dry + friction + momentum cap

    restore_rain_cn_from_scratch(dev);
}
```

**New kernels:**
- `swe2d_rk3_combine_kernel` — `state = α·backup + β·state` with wet/dry + friction + momentum cap (mirrors the existing `swe2d_rk2_combine_kernel` shape).

**Diagnostic snapshots:** `copy_stage_sources()` writes the live `d_external_source_mps` into the per-stage slot. This is a 1-kernel D2D copy, ~free. Used by snapshot/replay validation tests.

### 5.5 RK4: Classic 4-Stage — `swe2d_gpu.cu`

Add `swe2d_gpu_step_rk4()`:

```
Classic RK4 Butcher tableau:
  k1 = dt·R(U^n)
  k2 = dt·R(U^n + 0.5·k1)
  k3 = dt·R(U^n + 0.5·k2)
  k4 = dt·R(U^n + k3)
  U^{n+1} = U^n + (k1 + 2·k2 + 2·k3 + k4)/6
```

**Implementation:** Since `swe2d_gpu_step()` is a full step (flux + update + friction), we use the "evaluate at" pattern — blend state before each stage, save stage results, combine at the end.

**Buffer mapping (k-slope storage):**

| k-slope | Buffer |
|---------|--------|
| k1 | `d_h1/d_hu1/d_hv1` (reuse existing intermediate) |
| k2 | `d_h2/d_hu2/d_hv2` (reuse existing intermediate) |
| k3 | `d_h3/d_hu3/d_hv3` (reuse existing intermediate) |
| k4 | `d_k4_h/d_k4_hu/d_k4_hv` (NEW) |

**Pseudocode sketch:**
```cpp
void swe2d_gpu_step_rk4(dev, t_now, dt, ...) {
    save U^n → d_h0/d_hu0/d_hv0;
    save_rain_cn_to_scratch(dev);
    dev->kernel_graph_cache.time_integrator = 4;

    // k1 = dt·R(U^n)
    swe2d_recompute_coupling_for_stage(dev);
    snapshot_edge_bc_to_stage_slot(dev, 1, t_now);
    swe2d_gpu_step(dev, t_now, dt, ...);                       // state = U^n + k1
    copy_stage_sources(dev, 1);
    copy d_h → d_h1, ...;                                      // k1 → d_h1

    // state → U^n + 0.5·k1 for k2 eval
    swe2d_blend_kernel(... d_h,  d_h0, 1.0, 0.5);              // state = U^n + 0.5·k1
    swe2d_blend_kernel(... d_hu, d_hu0, 1.0, 0.5);
    swe2d_blend_kernel(... d_hv, d_hv0, 1.0, 0.5);

    // k2 = dt·R(U^n + 0.5·k1)
    swe2d_recompute_coupling_for_stage(dev);
    snapshot_edge_bc_to_stage_slot(dev, 2, t_now + 0.5*dt);
    swe2d_gpu_step(dev, t_now + 0.5*dt, dt, ...);
    copy_stage_sources(dev, 2);
    copy d_h → d_h2, ...;                                      // k2 → d_h2

    // state → U^n + 0.5·k2 for k3 eval
    swe2d_blend_kernel(... d_h,  d_h0, 1.0, 0.5);              // state = U^n + 0.5·k2
    ...

    // k3 = dt·R(U^n + 0.5·k2)
    swe2d_recompute_coupling_for_stage(dev);
    snapshot_edge_bc_to_stage_slot(dev, 3, t_now + 0.5*dt);
    swe2d_gpu_step(dev, t_now + 0.5*dt, dt, ...);
    copy_stage_sources(dev, 3);
    copy d_h → d_h3, ...;                                      // k3 → d_h3

    // state → U^n + k3 for k4 eval
    swe2d_blend_kernel(... d_h, d_h3, d_h0, ?, ?);             // state = U^n + k3
    ...

    // k4 = dt·R(U^n + k3)
    swe2d_recompute_coupling_for_stage(dev);
    snapshot_edge_bc_to_stage_slot(dev, 4, t_now + dt);
    swe2d_gpu_step(dev, t_now + dt, dt, ...);
    copy_stage_sources(dev, 4);
    copy d_h → d_k4_h, ...;                                    // k4 → d_k4_h

    // Combine: U^{n+1} = U^n + (k1 + 2·k2 + 2·k3 + k4)/6
    swe2d_rk4_combine_kernel(... d_h0/d_hu0/d_hv0,
                             d_h1/d_hu1/d_hv1, d_h2/d_hu2/d_hv2,
                             d_h3/d_hu3/d_hv3, d_k4_h/d_k4_hu/d_k4_hv,
                             h_min);

    restore_rain_cn_from_scratch(dev);
}
```

**New kernels:**
- `swe2d_blend_kernel` — generic `state = α·backup + β·state` (already listed in §6.1).
- `swe2d_rk4_combine_kernel` — weighted sum `(k1 + 2·k2 + 2·k3 + k4)/6` with wet/dry + friction + momentum cap.

### 5.6 RK5: Cash-Karp RK5(4) Embedded — `swe2d_gpu.cu`

**Decision:** Use the **Cash-Karp RK5(4) embedded** method. The 5th-order solution uses a 4-stage weighting `(37/378·k1 + 250/621·k3 + 125/594·k4 + 512/1771·k6)` — matches the existing `swe2d_rk5_graph_combine_kernel` (cpp/src/swe2d_gpu.cu:2318-2450) and the buffer allocation plan in §5.1. We do NOT need to allocate 6 separate k-slope buffers.

**Cash-Karp Butcher tableau (all 6 stage evaluations, even though only k1/k3/k4/k6 are stored):**
```
k1 = dt·R(U^n)
k2 = dt·R(U^n + (1/5)·k1)                              // evaluated, NOT stored
k3 = dt·R(U^n + (3/40)·k1 + (9/40)·k2)
k4 = dt·R(U^n + (3/10)·k1 - (9/10)·k2 + (6/5)·k3)
k5 = dt·R(U^n - (11/54)·k1 + (5/2)·k2 - (70/27)·k3 + (35/27)·k4)   // evaluated, NOT stored
k6 = dt·R(U^n + (1631/55296)·k1 + (175/512)·k2 + (575/13824)·k3 + (44275/110592)·k4 + (253/4096)·k5)
U^{n+1} = U^n + (37/378)·k1 + (250/621)·k3 + (125/594)·k4 + (512/1771)·k6
```

**Buffer mapping (RK5):**

| k-slope | Buffer | Used in combine? |
|---------|--------|-------------------|
| k1 | `d_h1/d_hu1/d_hv1` (reuse existing intermediate) | yes (37/378) |
| k2 | (not stored; folded into next stage input) | no |
| k3 | `d_h3/d_hu3/d_hv3` (reuse existing intermediate) | yes (250/621) |
| k4 | `d_k4_h/d_k4_hu/d_k4_hv` (NEW) | yes (125/594) |
| k5 | (not stored; folded into next stage input) | no |
| k6 | `d_k6_h/d_k6_hu/d_k6_hv` (NEW) | yes (512/1771) |

**Pseudocode sketch:**
```cpp
void swe2d_gpu_step_rk5(dev, t_now, dt, ...) {
    save U^n → d_h0/d_hu0/d_hv0;
    save_rain_cn_to_scratch(dev);
    dev->kernel_graph_cache.time_integrator = 6;

    // k1 = dt·R(U^n)
    swe2d_recompute_coupling_for_stage(dev);
    snapshot_edge_bc_to_stage_slot(dev, 1, t_now);
    swe2d_gpu_step(dev, t_now, dt, ...); copy d_h → d_h1, ...; copy_stage_sources(dev, 1);

    // k2 = dt·R(U^n + 0.2·k1) — evaluate, do NOT store
    blend_state(U^n + 0.2·k1);
    swe2d_recompute_coupling_for_stage(dev);
    snapshot_edge_bc_to_stage_slot(dev, 2, t_now + 0.2*dt);
    swe2d_gpu_step(dev, t_now + 0.2*dt, dt, ...); copy_stage_sources(dev, 2);
    // state now holds U^n + 0.2·k1 + k2 (k2 lives transiently in d_h)

    // k3 = dt·R(U^n + (3/40)·k1 + (9/40)·k2)
    // Compute target state U^n + (3/40)·k1 + (9/40)·k2 in d_h:
    //   = U^n + 0.075·k1 + 0.225·k2
    // Implement via two blend_kernel calls (first add 0.075·k1, then add 0.225·state).
    // (Or use a dedicated swe2d_blend3_kernel if profiling shows two kernels is wasteful.)
    blend_to_target_for_k3(dev);
    swe2d_recompute_coupling_for_stage(dev);
    snapshot_edge_bc_to_stage_slot(dev, 3, t_now + ...);
    swe2d_gpu_step(...); copy d_h → d_h3, ...; copy_stage_sources(dev, 3);

    // k4, k5, k6 — similar pattern
    // After k6 is stored, call the existing combine kernel.
    swe2d_rk5_graph_combine_kernel<<<...>>>(n_cells,
        dev->d_h, dev->d_hu, dev->d_hv,
        dev->d_h0, dev->d_hu0, dev->d_hv0,
        dev->d_h1, dev->d_hu1, dev->d_hv1,    // k1
        dev->d_h3, dev->d_hu3, dev->d_hv3,    // k3
        dev->d_k4_h, dev->d_k4_hu, dev->d_k4_hv,  // k4
        dev->d_k6_h, dev->d_k6_hu, dev->d_k6_hv,  // k6
        dev->d_max_wse_elev_error,
        dev->d_n_mann_cell,
        g, h_min, shallow_damping_depth, dt,
        momentum_cap_min_speed, momentum_cap_celerity_mult);

    restore_rain_cn_from_scratch(dev);
}
```

**Why the blend overhead is acceptable:** the blend kernels are O(n_cells) reads + writes with no global memory contention — they're cache-friendly and bound by memory bandwidth, not compute. Two blend kernels per stage cost ~1-2% of the per-step time. A dedicated `swe2d_blend3_kernel` (α·k1 + β·k2 + γ·U^n) would cut this in half; add only if profiling shows it matters.

### 5.7 Dispatch Changes (`swe2d_solver.cpp`)

**Phase 0 prerequisite:** before this dispatch lands, the Python `TemporalScheme` enum must expose `4`, the GUI must expose `3`, and `_VALID_TEMPORAL_SCHEMES` must accept `{1,2,3,4,5,6}` (see §3 Prerequisites). Until Phase 0 lands, a `temporal_order` outside `{1,2}` must `throw std::invalid_argument` in `swe2d_create` (the C++ invariant in §3.4).

Modify `swe2d_step()` to dispatch based on `temporal_order`:

```cpp
switch (s->cfg.temporal_order) {
    case 1:  swe2d_gpu_step(...);           break;  // Euler (existing path, single-stage)
    case 2:  swe2d_gpu_step_rk2(...);       break;  // SSP-RK2 (Heun)
    case 3:  swe2d_gpu_step_rk3(...);       break;  // SSP-RK3 (Shu-Osher)
    case 4:  swe2d_gpu_step_rk4(...);       break;  // Classic RK4
    case 5:  swe2d_gpu_step_rk4(...);       break;  // "Graph-safe RK4" — same algorithm as 4
    case 6:  swe2d_gpu_step_rk5(...);       break;  // Cash-Karp RK5(4)
    default: throw std::invalid_argument("swe2d_step: temporal_order must be in {1,2,3,4,5,6}");
}
```

**Note on the `case 1` path:** today, the code routes `temporal_order == 1` through `swe2d_gpu_step_rk2` (cpp/src/swe2d_solver.cpp:328), which is functionally Euler (single stage + no combine) but inherits the RK2 wrapper. After Phase 1, switch case 1 to call `swe2d_gpu_step(...)` directly — this avoids two `swe2d_state_to_double_kernel` D2D copies per Euler step that currently happen but are unused. Estimated win: ~1-2% on Euler steps.

**Why `case 4` and `case 5` both call `swe2d_gpu_step_rk4`:** see §3.2 mapping. The user-facing GUI label distinguishes them, but the algorithm is identical until a distinct low-storage graph-safe RK4 is designed.

### 5.8 CUDA Graph Capture — Whitelist Extension + Per-Stage Capture

**The issue is NOT that graphs can't capture higher-order schemes.** The issue is a single `cudaStreamSynchronize` call in the coupling function PLUS a hard-coded `time_integrator` whitelist.

#### 5.8.1 Root cause A: coupling sync (the easy part)

`swe2d_gpu.cu:7274-7279`:
```cpp
// Sync stream after coupling work so the solver's graph capture on the next
// step starts with a clean stream.  The coupling function is called from host
// code (apply_native_device_sources) and the solver's graph capture/replay
// uses the same stream; without this sync, pending async work causes
// cudaStreamBeginCapture to fail on the next solver step.
CUDA_CHECK(cudaStreamSynchronize(stream));
```

This sync is needed when coupling is called from **host code** (Python → pybind11 → C++). But when coupling recompute is called **from within the step function** (between RK stages), it's already on the same stream — the sync is redundant and breaks graph capture.

**Fix:** Add a `bool graph_safe = false` parameter to `swe2d_gpu_compute_coupling_full_on_device()`:

```cpp
void swe2d_gpu_compute_coupling_full_on_device(
    SWE2DDeviceState* dev, int32_t n_cells, int32_t n_structures,
    const double* cell_wse_host,
    const double* host_structure_flows,
    bool graph_safe = false)
{
    // ... all existing kernel launches (unchanged) ...
    if (!graph_safe) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
}
```

The `swe2d_recompute_coupling_for_stage()` wrapper always passes `graph_safe = true`.

**Binding signature:** the existing pybind11 binding (cpp/src/swe2d_bindings.cpp:1251) calls this function without the new param. Since it's defaulted, the binding continues to work. **Verify after Phase 1 that the binding still compiles and `coupling.py:1180` still passes `graph_safe=false` by default.**

#### 5.8.2 Root cause B: hard-coded `time_integrator` whitelist (must extend)

`swe2d_gpu.cu:4907-4910`:
```cpp
const int32_t graph_integrator =
    (dev->kernel_graph_cache.time_integrator == 2 || dev->kernel_graph_cache.time_integrator == 4)
        ? dev->kernel_graph_cache.time_integrator : 1;
```

**Only `{2, 4}` are recognized as graph-capturable.** Orders `3` and `6` silently fall back to non-graph (treated as Euler by the graph cache). Without extending this whitelist, the per-stage graph capture for RK3/RK5 will not fire.

**Fix:** extend to `{2, 3, 4, 5, 6}`:
```cpp
const int32_t graph_integrator =
    (dev->kernel_graph_cache.time_integrator >= 2 && dev->kernel_graph_cache.time_integrator <= 6)
        ? dev->kernel_graph_cache.time_integrator : 1;
```

(`time_integrator == 5` is valid because it's a routing key for the same RK4 routine; the graph cache treats 4 and 5 identically via this branch.)

#### 5.8.3 Per-stage vs full-sequence graph capture — pick one

The spec originally proposed capturing the entire multi-stage sequence as a single graph (~60-80 nodes for RK4 vs ~10-15 for Euler). Two viable alternatives:

| Strategy | Pros | Cons |
|----------|------|------|
| **A. Single full-sequence graph** (current spec text) | Maximum launch overhead amortization | One cache miss invalidates the whole RK step; large captured graph |
| **B. Per-stage graph** (recommended) | Smaller cache footprint; per-stage invalidation only affects one stage's capture; aligns with existing per-`swe2d_gpu_step` capture flow | Per-stage cache lookup overhead (negligible) |

**Recommendation: B (per-stage capture).** Reason: the existing `swe2d_gpu_step` already owns the graph capture logic (lines 5052-5236); the per-stage call inside `swe2d_gpu_step_rkN` naturally re-uses that capture. No code restructure needed — just set `time_integrator` correctly before each `swe2d_gpu_step` call and let the per-stage capture hit. The `cache.config_signature` already includes all relevant config knobs, so the per-stage graph invalidates independently when stage-specific state changes (rare in practice).

**Implementation steps:**
1. Add `graph_safe` parameter (defaulted) to `swe2d_gpu_compute_coupling_full_on_device`.
2. Extend `time_integrator` whitelist from `{2, 4}` to `{2..6}`.
3. In `swe2d_gpu_step_rkN()` functions, set `dev->kernel_graph_cache.time_integrator = N` before each `swe2d_gpu_step` sub-call (the per-stage graph cache key will then include the correct N).
4. Remove the Python-side graph-disable for `temporal_order >= 4` in `run_options_builder.py:139-145` (spec §5.11).

**Note on `swe2d_gpu_invalidate_graph_cache`:** the coupling controller (coupling.py:1201-1202) currently calls this when face-flux mode toggles. With coupling moved into the C++ step, this Python call is no longer on the critical path — but if face-flux toggles mid-run (it shouldn't, but defensively), the C++ step would re-capture naturally on the next step via the cache_signature mismatch.

### 5.9 Persistent Chunking + Tiny-Mode Interaction with Higher Orders

**Current constraint** (`swe2d_gpu.cu:5890-5898`):
```cpp
const bool cooperative_kernel_supported =
    (spatial_scheme == static_cast<int>(SWE2DSpatialScheme::FV_FIRST_ORDER)) &&
    !extreme_rain_mode && !source_true_subcycling && !source_imex_split;
```

The cooperative persistent kernel only supports **first-order spatial** scheme. For higher-order temporal:
- **First-order spatial + higher-order temporal:** The cooperative kernel COULD be extended to run multiple temporal stages. Each stage would need grid sync between flux and update. The cooperative kernel already has grid sync — adding a temporal loop inside it is feasible but complex.
- **Higher-order spatial + higher-order temporal:** The persistent chunk path falls back to `run_chunked_baseline()` which is a loop of `swe2d_gpu_step()` calls. This already works for any combination.

**Recommendation:** For Phase 1-5, use the baseline loop path for higher-order temporal. The persistent chunk extension is a Phase 7 optimization (low priority). The baseline loop already works correctly — it's just slightly slower than the cooperative kernel due to kernel launch overhead per sub-step.

#### 5.9.1 `tiny_mode` + `temporal_order > 2` interaction (currently undefined)

`swe2d_solver.cpp:226-230` already forces `kTinyModeFused` off when `use_rk2 = (temporal_order >= 2)`:
```cpp
const bool fused_supported_now =
    tiny_fused_path_eligible && !use_rk2;
if (!fused_supported_now) {
    tiny_effective = kTinyModeOff;
}
```
Good — fused mode is silently disabled for any multi-stage scheme.

**However:** `kTinyModePersistent` (cpp/src/swe2d_solver.cpp:298) only has an implementation for RK2 (`swe2d_gpu_step_rk2_persistent_chunk`). For `tiny_mode == persistent` with `temporal_order >= 3`, the code would call a non-existent function — silent fallback or hard error, depending on what was wired in.

**Required behavior:**
- `temporal_order == 1`: tiny_mode=persistent uses `swe2d_gpu_step_persistent_chunk` (existing first-order path).
- `temporal_order == 2`: tiny_mode=persistent uses `swe2d_gpu_step_rk2_persistent_chunk` (existing RK2 path).
- `temporal_order >= 3`: tiny_mode=persistent → either (a) fall through to `run_chunked_baseline()` (which is what the chunked loop already does), OR (b) reject the combination at solver init with `throw std::invalid_argument("tiny_mode=persistent not supported with temporal_order >= 3")`.

**Pick (b) — fail loud.** The whole point of `tiny_mode=persistent` is to skip Python-side orchestration overhead for small models; falling back to the chunked loop silently defeats the purpose and produces surprising performance regressions. Surface the constraint in the GUI: when `temporal_order >= 3`, disable the `tiny_mode` combo's "Persistent" option (set its `setEnabled(False)`).

This is a 5-line fix in `swe2d_solver.cpp` and a 3-line fix in `model_tab_view.py`; add to Phase 6.

### 5.10 CFL and Tooltip

**CFL stays at 0.45 for all schemes** (safe production default). Add recommended CFL ranges to the GUI tooltip:

| Scheme | Theoretical CFL | Recommended CFL | Tooltip Text |
|--------|-----------------|-----------------|--------------|
| Euler | 1.0 | 0.4-0.5 | "1st order. Stable up to CFL≈1.0. Use CFL 0.4-0.5 for reliability." |
| SSP-RK2 | 1.0 | 0.5-0.8 | "2nd order. Stable up to CFL≈1.0. Use CFL 0.5-0.8." |
| SSP-RK3 | 1.0 | 0.5-0.8 | "3rd order (Shu-Osher). Stable up to CFL≈1.0. Use CFL 0.5-0.8." |
| Classic RK4 | 2.8 | 1.0-1.5 | "4th order. Stable up to CFL≈2.8 in pure-flow; use CFL 1.0-1.5 with sources/structures." |
| Cash-Karp RK5(4) | ~1.5 (practical) | 0.5-1.0 | "5th order (Cash-Karp embedded). Practical CFL ≤1.5 with sources; recommended 0.5-1.0." |

**Why the RK5 numbers changed from the original spec:** the original "0.8-1.5" range assumed a generous theoretical stability bound. In practice, with nonlinear source terms (structures, drainage, rainfall) and wet/dry fronts, RK5(4) tends to lose its SSP-like stability margin. The conservative 0.5-1.0 range matches what large-scale coastal models (e.g., ADCIRC) use for higher-order temporal schemes in production.

### 5.11 Python-Side Changes

**Complete file list** — every file touched in Phase 6:

| File | Edit |
|------|------|
| `swe2d/extensions/extension_models.py` | Add `CLASSIC_RK4 = 4` to `TemporalScheme` enum (Phase 0). |
| `swe2d/workbench/views/model_tab_view.py` | Add RK3 combo entry; update tooltip with §5.10 text; disable `tiny_mode` Persistent option when `temporal_order >= 3` (§5.9.1). |
| `swe2d/workbench/services/constants_service.py` | Add RK3 entry to scheme combo data (Phase 0). |
| `swe2d/workbench/services/run_service.py` | Update `_VALID_TEMPORAL_SCHEMES` to `{1,2,3,4,5,6}` (drop `0`); update error message (Phase 0). |
| `swe2d/runtime/runtime_step_executor.py` | **Remove the IMEX path** (lines 85-204). C++ handles source freshness internally. Remove `stage_coupled_imex_enabled` parameter and all IMEX-related branching. The standard path (lines 205-273) becomes the only path. |
| `swe2d/runtime/run_options_builder.py` | **Remove the graph-disable** for `temporal_order >= 4` (lines 139-145). Graph capture is now handled correctly in C++ for all orders (see §5.8). Keep the graph-disable for persistent chunking path (first-order spatial only) if still relevant. |
| `swe2d/runtime/backend.py` | Remove `save_coupling_pred()`, `average_coupling_sources()`, `restore_state_from_backup()` Python wrappers (lines 1207-1223). They become dead code once `runtime_step_executor.py` stops calling them. |
| `swe2d/runtime/runtime_setup_configurator.py` | The `resolve_stage_coupled_imex` method (lines 108-134) becomes dead code — remove the whole method. Also remove the `source_stage_coupled_imex_rk2_chk` plumbing. |
| `swe2d/runtime/coupling.py` | Verify `apply_native_device_sources` (line 1028) is still called from the standard path. After Phase 1, coupling moves into C++ — this Python function becomes the LEGACY path for host readback scenarios only. Mark with a deprecation comment, do not remove in Phase 6 (some non-GPU tests still rely on it). |
| `swe2d/workbench/controllers/run_controller.py` | Remove `stage_coupled_imex_requested`/`stage_coupled_imex_enabled` plumbing (lines 130, 447, 835-856, 902). Remove `source_stage_coupled_imex_rk2_chk` from the params dict (line 2488 in studio_dialog.py too). |
| `swe2d/workbench/services/non_gui_runtime_service.py` | Remove `stage_coupled_imex_enabled` parameter (line 373) and forwarding (line 468). |
| `swe2d/workbench/studio_dialog.py` | Remove `source_stage_coupled_imex_rk2_chk` from collected params (line 2488). |
| `swe2d/workbench/dialogs/batch_simulation_dialog.py` | Verify `temporal_order_combo` mapping (line 73) covers all six values after Phase 0. |

**C++ cleanup (binding removal):**

After Python callers are gone, the following C++ helpers become dead code. Remove them in the same Phase 6 commit to avoid leaving orphans:

| C++ function | Binding | Files |
|--------------|---------|-------|
| `swe2d_gpu_save_coupling_pred` | `swe2d_bindings.cpp:2730` | `swe2d_gpu.cuh:955`, `swe2d_gpu.cu:1405` |
| `swe2d_gpu_average_coupling_sources` | `swe2d_bindings.cpp:2738` | `swe2d_gpu.cuh:957`, `swe2d_gpu.cu:1415` |
| `swe2d_gpu_restore_state_from_backup` | `swe2d_bindings.cpp:2746` | `swe2d_gpu.cuh:959`, `swe2d_gpu.cu:1427` |
| `s_coupling_pred_source` buffer | (allocated in `swe2d_gpu.cu:4717-4718`) | remove `alloc_d` + `safe_free` |

Also remove the corresponding `d_coupling_pred_source` field from `SWE2DDeviceState` (cpp/src/swe2d_gpu.cuh:239).

---

## 6. New Kernels

### 6.1 `swe2d_blend_kernel` (generic)

```cuda
__global__ void swe2d_blend_kernel(
    int32_t n_cells,
    State* state,           // in/out: α·backup + β·state
    const double* backup,   // U^n or intermediate
    double alpha,           // weight for backup
    double beta)            // weight for state
{
    int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells) return;
    state[c] = static_cast<State>(
        alpha * backup[c] + beta * static_cast<double>(state[c]));
}
```

### 6.2 `swe2d_rk3_combine_kernel`

```cuda
__global__ void swe2d_rk3_combine_kernel(
    int32_t n_cells,
    State* cell_h, State* cell_hu, State* cell_hv,
    const double* h0, const double* hu0, const double* hv0,
    double alpha, double beta,
    double h_min)
{
    int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells) return;
    double h_new = alpha * h0[c] + beta * static_cast<double>(cell_h[c]);
    double hu_new = alpha * hu0[c] + beta * static_cast<double>(cell_hu[c]);
    double hv_new = alpha * hv0[c] + beta * static_cast<double>(cell_hv[c]);
    // Wet/dry, friction, momentum cap (same as rk2_combine_kernel)
    ...
}
```

### 6.3 `swe2d_rk4_combine_kernel`

```cuda
__global__ void swe2d_rk4_combine_kernel(
    int32_t n_cells,
    State* cell_h, State* cell_hu, State* cell_hv,
    const double* h0, const double* hu0, const double* hv0,
    const double* k1_h, const double* k1_hu, const double* k1_hv,
    const double* k2_h, const double* k2_hu, const double* k2_hv,
    const double* k3_h, const double* k3_hu, const double* k3_hv,
    const double* k4_h, const double* k4_hu, const double* k4_hv,
    double h_min)
{
    // Classic RK4: U^{n+1} = U^n + (k1 + 2·k2 + 2·k3 + k4)/6
    int32_t c = ...;
    double h_new = h0[c]
        + (1.0/6.0) * k1_h[c]
        + (1.0/3.0) * k2_h[c]
        + (1.0/3.0) * k3_h[c]
        + (1.0/6.0) * k4_h[c];
    // ... wet/dry, friction, momentum cap ...
}
```

### 6.4 `swe2d_recompute_coupling_for_stage` (wrapper)

```cpp
// Wrapper signature: takes the per-stage dt so set_coupling_dt and coupling
// function both see the correct value (today this is set from Python before
// each coupling call; once coupling moves into the C++ step, the wrapper
// owns it).
void swe2d_recompute_coupling_for_stage(SWE2DDeviceState* dev, double dt_stage) {
    if (!dev) return;

    // Drainage: only re-run the drainage solver if drainage was preloaded for
    // this device.  The DrainageStepWs has no `drainage_preloaded` flag today;
    // detect via `drain_ws.cell_capacity > 0` (set by swe2d_gpu_drainage_step
    // on first call) AND a non-zero link_flow / node_depth buffer.  Add an
    // explicit `bool drain_preloaded` flag in Phase 1 for clarity.
    const bool drainage_active = (dev->drain_ws.cell_capacity > 0);
    if (drainage_active) {
        // swe2d_gpu_drainage_step reads node_depth / link_flow from device-
        // resident buffers, advances them by dt, and writes q_cell to
        // coupling_ws.d_drainage_q.  All on the same stream.
        swe2d_gpu_drainage_step(/*on-device args, dt_stage*/);
    }

    // Set the coupling dt BEFORE the coupling function — this dt is read by
    // the culvert face-flux kernel for its depth-safety limiter.  Today
    // coupling.py:1173-1174 does this; after Phase 1 the C++ wrapper owns it.
    swe2d_gpu_set_coupling_dt(dt_stage);

    // Structures + culverts: re-evaluate at current device state.
    // graph_safe=true skips the trailing cudaStreamSynchronize (we're inside
    // a per-stage graph capture; sync would break capture).
    const bool coupling_active =
        dev->sf_ws.params_preloaded ||
        dev->culvert_ff_ws.params_preloaded ||
        drainage_active;
    if (coupling_active) {
        swe2d_gpu_compute_coupling_full_on_device(
            dev, dev->n_cells, dev->sf_ws.n_structures,
            /*cell_wse_host=*/nullptr, /*host_structure_flows=*/nullptr,
            /*graph_safe=*/true);
    }
}
```

**Why the explicit `set_coupling_dt`:** the culvert face-flux kernel (cpp/src/swe2d_gpu.cu:7213-7226) reads `s_coupling_dt` for its depth-safety limiter. The Python path sets this via `swe2d_gpu_set_coupling_dt` from `coupling.py:1173-1174`. Once coupling moves inside the step, the wrapper must set it per stage — otherwise the face-flux limiter uses the PREVIOUS step's dt.

### 6.5 Header declarations and binding signatures

All new C++ host-callable functions need declarations in `cpp/src/swe2d_gpu.cuh` and (where exposed to Python) bindings in `cpp/src/swe2d_bindings.cpp`. Summary:

| Function | Header decl | Binding | Notes |
|----------|-------------|---------|-------|
| `swe2d_gpu_step_rk3` | yes (new) | not needed (Python uses `swe2d_step`) | mirrors `swe2d_gpu_step_rk2` signature |
| `swe2d_gpu_step_rk4` | yes (new) | not needed | mirrors `swe2d_gpu_step_rk2` signature |
| `swe2d_gpu_step_rk5` | yes (new) | not needed | mirrors `swe2d_gpu_step_rk2` signature |
| `swe2d_recompute_coupling_for_stage` | yes (new, internal-only) | **no binding** | called only from C++ step functions |
| `swe2d_gpu_compute_coupling_full_on_device` | already declared | already bound | add defaulted `bool graph_safe = false` param — backwards compat preserved |

**New device buffers** (added to `SWE2DDeviceState` in `swe2d_gpu.cuh`):
- `double* d_k4_h, d_k4_hu, d_k4_hv;`
- `double* d_k6_h, d_k6_hu, d_k6_hv;`
- `double* d_rain_cn_scratch_h;`
- `double* d_rain_cn_scratch_ex;`

Allocated in `swe2d_gpu_init()` (cpp/src/swe2d_gpu.cu ~line 4700) and freed in `swe2d_gpu_free()` (cpp/src/swe2d_gpu.cu:9116).

**Removed fields** (after Phase 6 Python cleanup):
- `double* d_coupling_pred_source;` (SWE2DDeviceState, cpp/src/swe2d_gpu.cuh:239)

---

## 7. Implementation Order

### Phase 0: Value-Set Reconciliation (Prerequisite — must land first)

1. Add `CLASSIC_RK4 = 4` to `TemporalScheme` enum in `extension_models.py`.
2. Insert RK3 combo entry in `model_tab_view.py:497-504` and `constants_service.py:54-58`.
3. Update `_VALID_TEMPORAL_SCHEMES` to `{1, 2, 3, 4, 5, 6}` in `run_service.py:144`.
4. Add the C++ invariant: `swe2d_create` throws `std::invalid_argument` for `temporal_order` outside `{1, 2}` (until Phase 3 lands).
5. **Verify:** unit test enumerates all six values via Python, all six appear in the GUI combo, all six pass `run_service` validation.

**Files modified:**
- `swe2d/extensions/extension_models.py`
- `swe2d/workbench/views/model_tab_view.py`
- `swe2d/workbench/services/constants_service.py`
- `swe2d/workbench/services/run_service.py`
- `cpp/src/swe2d_solver.cpp` (invariant throw)

**No behavior change** — Phase 0 only adds surface area for options that aren't yet implemented. The C++ throw prevents users from selecting them in the GUI.

### Phase 1: Fix RK2 Stale Sources (Critical Bug Fix)

1. Add `bool graph_safe = false` parameter to `swe2d_gpu_compute_coupling_full_on_device()` in `swe2d_gpu.cu:7049`. When `true`, skip the trailing `cudaStreamSynchronize`.
2. Add `swe2d_recompute_coupling_for_stage(dev, dt_stage)` wrapper (§6.4). Includes the `swe2d_gpu_set_coupling_dt(dt_stage)` call and drainage step dispatch.
3. Call recompute between Stage 1 and Stage 2 in `swe2d_gpu_step_rk2()` (cpp/src/swe2d_gpu.cu:6181) and `swe2d_gpu_step_rk2_persistent_chunk()` (line 6329).
4. Allocate `d_rain_cn_scratch_h` and `d_rain_cn_scratch_ex` in `swe2d_gpu_init()`. Update the rain CN save/restore at lines 6247-6275 to use the new buffers.
5. Switch `swe2d_solver.cpp:328` to call `swe2d_gpu_step(...)` directly for `temporal_order == 1` (skip the unused RK2 wrapper).
6. Test with structures + drainage + 2nd-order schemes.

**Files modified:**
- `cpp/src/swe2d_gpu.cu` — step_rk2, step_rk2_persistent_chunk, coupling function, init, free
- `cpp/src/swe2d_gpu.cuh` — new function declarations, new buffer pointers, updated step function signatures
- `cpp/src/swe2d_solver.cpp` — Euler path (case 1)

### Phase 2: Buffer Allocation + Utility Kernels + Graph Capture

1. Extend `time_integrator` whitelist from `{2, 4}` to `{2..6}` at swe2d_gpu.cu:4907-4910.
2. Allocate `d_k4_*`, `d_k6_*` in `swe2d_gpu_init()`.
3. Deallocate all new buffers in `swe2d_gpu_free()`.
4. Add `swe2d_blend_kernel`.
5. Add `swe2d_rk3_combine_kernel`.
6. Add `swe2d_rk4_combine_kernel`.

**Files modified:**
- `cpp/src/swe2d_gpu.cu` — init, free, new kernels, graph whitelist
- `cpp/src/swe2d_gpu.cuh` — new buffer pointers, kernel declarations

### Phase 3: Implement RK3

1. Add `swe2d_gpu_step_rk3()` in `swe2d_gpu.cu` (uses `d_h1/d_h2` for intermediates per §5.4).
2. Add `case 3:` in `swe2d_solver.cpp:swe2d_step()`.
3. Update the C++ invariant: extend the valid range from `{1, 2}` to `{1, 2, 3}`.
4. Test: stability, manufactured-solution convergence order ≈ 3.

**Files modified:**
- `cpp/src/swe2d_gpu.cu` — new step function
- `cpp/src/swe2d_solver.cpp` — dispatch switch, invariant throw

### Phase 4: Implement RK4

1. Add `swe2d_gpu_step_rk4()` in `swe2d_gpu.cu` (uses `d_h1/d_h2/d_h3` for k1/k2/k3 + new `d_k4_*` for k4 per §5.5).
2. Add `case 4:` and `case 5:` in `swe2d_solver.cpp:swe2d_step()`.
3. Update the C++ invariant: extend to `{1..5}`.
4. Test: manufactured-solution convergence order ≈ 4.

**Files modified:**
- `cpp/src/swe2d_gpu.cu` — new step function
- `cpp/src/swe2d_solver.cpp` — dispatch switch, invariant throw

### Phase 5: Implement RK5 (Cash-Karp RK5(4))

1. Add `swe2d_gpu_step_rk5()` in `swe2d_gpu.cu` (uses `d_h1/d_h3/d_k4/d_k6` per §5.6).
2. Wire up existing `swe2d_rk5_graph_combine_kernel`.
3. Add `case 6:` in `swe2d_solver.cpp:swe2d_step()`.
4. Update the C++ invariant: extend to `{1..6}`.
5. Test: manufactured-solution convergence order ≈ 5.

**Files modified:**
- `cpp/src/swe2d_gpu.cu` — new step function
- `cpp/src/swe2d_solver.cpp` — dispatch switch, invariant throw

### Phase 6: Python + GUI + C++ Cleanup

1. Remove IMEX path from `runtime_step_executor.py` (lines 85-204).
2. Remove `stage_coupled_imex_enabled` plumbing from `runtime_step_executor.py`, `runtime_setup_configurator.py`, `run_controller.py`, `non_gui_runtime_service.py`.
3. Remove `source_stage_coupled_imex_rk2_chk` from `model_tab_view.py` (line 614-627), `studio_dialog.py:2488`, `run_service.py`.
4. Remove graph-disable for `temporal_order >= 4` in `run_options_builder.py:139-145`.
5. Remove IMEX-related Python methods from `backend.py:1207-1223` (`save_coupling_pred`, `average_coupling_sources`, `restore_state_from_backup`).
6. Remove dead C++ helpers + bindings: `swe2d_gpu_save_coupling_pred`, `swe2d_gpu_average_coupling_sources`, `swe2d_gpu_restore_state_from_backup`, `d_coupling_pred_source` field.
7. Disable `tiny_mode` Persistent option in `model_tab_view.py` when `temporal_order >= 3` (§5.9.1).
8. Add `throw std::invalid_argument` in `swe2d_solver.cpp:swe2d_step` for unknown `temporal_order` (defensive — see §5.7).
9. Update GUI tooltips with §5.10 CFL recommendations.

**Files modified** (full list — see §5.11 for per-file edit details):
- `swe2d/runtime/runtime_step_executor.py`
- `swe2d/runtime/run_options_builder.py`
- `swe2d/runtime/backend.py`
- `swe2d/runtime/runtime_setup_configurator.py`
- `swe2d/runtime/coupling.py` (deprecation comment only)
- `swe2d/workbench/controllers/run_controller.py`
- `swe2d/workbench/services/non_gui_runtime_service.py`
- `swe2d/workbench/services/run_service.py` (if not already in Phase 0)
- `swe2d/workbench/views/model_tab_view.py`
- `swe2d/workbench/studio_dialog.py`
- `swe2d/workbench/dialogs/batch_simulation_dialog.py` (verify only)
- `cpp/src/swe2d_gpu.cu` — remove dead helpers
- `cpp/src/swe2d_gpu.cuh` — remove `d_coupling_pred_source` field
- `cpp/src/swe2d_bindings.cpp` — remove three IMEX bindings
- `cpp/src/swe2d_solver.cpp` — defensive throw

### Phase 7 (Future): Persistent Chunking for Higher Orders

Extend the cooperative persistent kernel to support higher-order temporal by adding a temporal loop inside the kernel. Low priority — the baseline loop path works correctly.

---

## 8. Testing Strategy

### 8.0 Phase 0 Tests (Value-Set Reconciliation)

| Test | Purpose |
|------|---------|
| `test_temporal_scheme_enum_complete` | `TemporalScheme` enum contains exactly `{1, 2, 3, 4, 5, 6}` |
| `test_run_service_accepts_all_six_schemes` | `_VALID_TEMPORAL_SCHEMES` accepts each value, rejects 0/7/-1 |
| `test_gui_combo_exposes_all_six_schemes` | Headless QGIS test: combo box has exactly 6 entries |
| `test_solver_rejects_unimplemented_temporal_order` | `swe2d_create` throws `std::invalid_argument` for `temporal_order ∈ {3,4,5,6}` before Phase 3/4/5 land |

### 8.1 Unit Tests

| Test | Purpose | File |
|------|---------|------|
| `test_rk2_structure_stability` | Structures + RK2 doesn't blow up | `tests/test_swe2d_gpu_structure_coupling.py` |
| `test_rk3_convergence_order` | Manufactured solution shows 3rd-order convergence | `tests/test_swe2d_gpu_convergence.py` |
| `test_rk4_convergence_order` | Manufactured solution shows 4th-order convergence | `tests/test_swe2d_gpu_convergence.py` |
| `test_rk5_convergence_order` | Manufactured solution shows 5th-order convergence | `tests/test_swe2d_gpu_convergence.py` |
| `test_coupling_recompute_freshness` | Verify sources change between RK2 stages (regression test for Phase 1) | `tests/test_swe2d_gpu_structure_coupling.py` |
| `test_rain_cn_scratch_isolation` | Rain CN cumulative state net-advances by `dt` (not `2*dt`/`3*dt`/`4*dt`/`6*dt`) for RK2/3/4/5 | `tests/test_swe2d_gpu_rain_cn.py` |
| `test_tiny_mode_persistent_rejects_high_order` | `swe2d_step` with `tiny_mode=persistent` + `temporal_order >= 3` throws | `tests/test_swe2d_solver_config.py` |
| `test_hydrograph_bc_per_stage_snapshots` | Verify `d_stage_edge_bc[i]` differs across stages when hydrograph is non-stationary | `tests/test_hydrograph_bc_native.py` |
| `test_graph_capture_extended_whitelist` | With `time_integrator ∈ {2,3,4,5,6}`, the per-stage graph cache hits after first capture | `tests/test_swe2d_gpu_kernel_graphs.py` |

### 8.2 Integration Tests

| Test | Purpose |
|------|---------|
| Rainfall + structures + drainage + RK2 | Full pipeline stability |
| Rainfall + structures + drainage + RK3 | Higher-order stability |
| Rainfall + structures + drainage + RK4 | Higher-order stability |
| Rainfall + structures + drainage + RK5 | Higher-order stability |
| Dam-break + RK2 vs RK3 vs RK4 vs RK5 | Convergence comparison (log-log L2 error vs dt) |
| Culvert flow + open boundaries + RK2/RK3 | Boundary interaction |
| Hydrograph BC + RK3 + structures | Verify per-stage BC snapshots reflect stage times |
| Pure-flow model + all 6 temporal orders | Sanity check no regression in non-coupling path |

### 8.3 Performance Regression

- Benchmark pure-flow model (no structures) with RK2: should be identical speed (or marginally faster, since Euler path drops the unused D2D copies in §5.7).
- Benchmark with structures: slight slowdown from coupling recompute (~5-10% per step for RK2; ~25-50% for RK5).
- Benchmark RK4 vs RK2: expect ~3-4× slower per step (4 stages vs 2).
- Benchmark RK5 vs RK2: expect ~5-6× slower per step (6 stages vs 2).
- Benchmark graph capture: confirm per-stage graphs are cached (no kernel launch overhead per stage after warm-up).

### 8.4 Numerical Equivalence (Phase 6 acceptance gate)

After Python IMEX removal (Phase 6), compare simulation outputs between the OLD Python IMEX path and the NEW C++ RK2 fresh-source path. They MUST agree to within round-off tolerance on a canonical test case (e.g., `tests/test_swe2d_gpu_structure_coupling.py:test_pipeline_stability`). Differences would indicate a regression in the C++ recompute logic.

Acceptance criterion: max(|h_old - h_new|) < 1e-9 × max(h_old) over the entire 1-hour simulation.

---

## 9. Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Coupling recompute adds overhead | Low | Guard: skip if no structures/drainage. Pure-flow models unaffected. |
| Rain CN state conflict with d_h1/d_h2 | Medium | Dedicated rain scratch buffers eliminate conflict. Test net-advance per scheme. |
| CUDA graph capture + coupling sync | Medium | Add `graph_safe` parameter to skip sync when called from within step. |
| CUDA graph `time_integrator` whitelist hardcoded to `{2, 4}` | **High** | Extend whitelist to `{2..6}` in Phase 2; without this fix, RK3/RK5 graphs silently fall back to non-graph. |
| Phase 0 not landing first → users select unimplemented schemes | Medium | C++ invariant throws in `swe2d_create` for `temporal_order ∉ {1,2}` until Phase 3/4/5 land. GUI hides invalid options. |
| Per-stage BC snapshots freeze `t_now` at stage-launch time | Low | Static BCs (OPEN/WALL) are unaffected. Hydrograph BCs are evaluated per stage with the stage-specific `t_now + (i-1)*dt`. |
| `tiny_mode=persistent` + `temporal_order >= 3` → silent fallback | Medium | Phase 6 adds explicit `throw std::invalid_argument` + GUI disables Persistent option. |
| `s_coupling_dt` not updated when coupling moves into step | Medium | `swe2d_recompute_coupling_for_stage` calls `swe2d_gpu_set_coupling_dt(dt_stage)` internally. |
| Dead C++ IMEX helpers + bindings left behind after Phase 6 | Low | Phase 6 explicitly removes `swe2d_gpu_save_coupling_pred`, `swe2d_gpu_average_coupling_sources`, `swe2d_gpu_restore_state_from_backup`, their bindings, and `d_coupling_pred_source` field. |
| `coupling.py:apply_native_device_sources` becomes dead code | Low | Mark with deprecation comment in Phase 6; leave intact for non-GPU tests that need host readback. Do not delete. |
| RK5(4) numerical stability with sources | Medium | Conservative CFL recommendation (0.5-1.0); convergence test on manufactured solution with structure forcing. |
| `_VALID_TEMPORAL_SCHEMES` rejects 5/6 today | Low | Phase 0 updates validator before any GUI option exposes them. |
| Per-stage graph cache misses invalidate all subsequent stages | Low | Use **per-stage capture** (§5.8.3 Strategy B): each `swe2d_gpu_step` call captures independently; stage-specific invalidation affects only that stage. |
| Numerical drift between OLD Python IMEX and NEW C++ RK2 path | Low | Phase 6 acceptance gate: equivalence test on canonical case (§8.4). |
