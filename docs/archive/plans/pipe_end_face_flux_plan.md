---
type: plan
status: superseded
created: 2026-07-17
completed: 2026-07-25
superseded_by: docs/PIPE1D_SWE2D_TEMPORAL_COUPLING_SPEC.md
---

# Pipe-end Face-Flux Coupling Plan

**Date:** 2026-07-17
**Status:** Proposed
**Goal:** Replace source-term-based weir/orifice coupling at the pipe-end / node / outfall interface with a true mass/momentum-conserving face-flux coupling, modelled on the existing culvert face-flux infrastructure.

---

## 1. Problem statement

The current pipe-end → 2D-surface coupling path is:

```
1D pipe solver   →   pipe-end weir/orifice kernel
                        (writes Q to d_pipe_end_q_2d)
                                  ↓
                          fold into d_external_source_mps  (Q / cell_area)
                                  ↓
                          2D SWE solver consumes as a depth-rate source
```

This is fundamentally a **source-term coupling**. Three consequences make it unsuitable for large flows:

1. **No momentum transfer.** The pipe-end outflow adds *depth* to the 2D cell but zero *velocity*. Water that should leave the pipe at several m/s arrives as stagnant fluid.
2. **CFL cap truncates large Q.** The 2D solver's `source_rate_cap` and `source_depth_step_cap` silently limit the source term. For 16 000 cfs into a 100 m² 2D cell the implied source rate is ≈5 m/s, which the solver caps at zero.
3. **Pressure-velocity coupling is one-sided.** The pipe solver sees the 2D WSE at the *start* of the timestep and locks in its Q for the whole `dt`. The 2D state changing during the step does not feed back into the pipe.

The culvert face-flux path already in the codebase (see `swe2d_culvert_face_flux_kernel`, swe2d_gpu.cu:3156) shows the right pattern: write a 3-component (h, hu, hv) flux to `d_ext_struct_flux_h/hu/hv` and let the 2D solver consume it via `d_ext_struct_flux_*` (set by `use_culvert_face_flux = true` in the device state). The pipe-end should use the same path.

---

## 2. Scope: every node type, not only pipe-end

The plan must apply to **every** node type that has a coupled 2D cell, not just `pipe_end`:

| `node_type`      | Current coupling                                            | Face-flux plan                                                                |
|------------------|-------------------------------------------------------------|-------------------------------------------------------------------------------|
| `inlet`          | grate/curb capture (HEC-22 formulae) → source term         | 2D → 1D face flux through the grate/curb opening                             |
| `pipe_end`       | weir/orifice → source term (broken)                          | 1D ↔ 2D face flux at the open end (HLLC between pipe-cell and 2D cell)         |
| `outfall`        | free-discharge kernel (depth reset to 0 each step)          | 1D → 2D face flux using rating curve as the *downstream* WSE, not 0            |
| `storage`        | mixed — acts as a manhole with surcharge                      | 1D ↔ 2D face flux with a virtual storage node on the 1D side                    |
| `junction`       | surcharge overflow (when WSE exceeds rim) → source term     | 1D ↔ 2D face flux with overflow limiting per SWMM manhole rules               |

All five node types share the same face-flux kernel; only the *boundary condition on the 1D side* differs. The 2D-side kernel is identical: read cell (h, hu, hv, zb), receive the pipe-side state (A, Q, h_p, S0), solve HLLC, write to `d_ext_struct_flux_*`.

### 2.1 Why unify now

Each node type currently has its own ad-hoc kernel (`swe2d_pipe_end_weir_orifice_kernel`, `swe2d_drainage_outfall_exchange_kernel`, `swe2d_drainage_inlet_exchange_kernel`, `swe2d_pipe1d_junction_overflow_kernel_host`). They all converge to the same destination (`d_external_source_mps`) but with different physics, different sign conventions, different avail limiters. Unifying them under one face-flux kernel eliminates the source-term coupling entirely.

---

## 3. Architecture

### 3.1 State at the coupling interface

For a node coupled to a 2D cell, define the *interface state vector*:

```
Left  (1D pipe side):  ρL = A_p,  (ρu)L = Q,  EL = invert + h_p       (depth above invert)
Right (2D surface):   ρR = h,    (ρu)R = hu,  ER = zb + h             (absolute WSE)
```

`A_p` is the pipe-cell area at the open-end sub-cell; `Q` is the volumetric flow; `h_p` is the depth above the pipe invert. On the 2D side, `h` is depth, `(hu, hv)` is momentum, `zb` is bed elevation. Pressures:

```
PL = 0.5 * g * ρL * T_L * h_p^2     (rectangular approximation; for slot use the Sjoberg slot-width)
PR = 0.5 * g * ρR * h^2              (standard 2D SWE)
```

Cell width on the left: use the pipe opening width `w_p` (slot_width in slot regime, full width otherwise).

### 3.2 HLLC at the 1D / 2D interface

The HLLC solver is structurally identical to `hllc_flux_cuda_local` (swe2d_gpu.cu:531) but with a non-trivial left-state (1D SWE with non-rectangular cross-section). The wave speeds:

```
sL = min(uL - cL, u_roe - c_roe)
sR = max(uR + cR, u_roe + c_roe)
sM = (PR - PL + ρR·uR·(sR - uR) - ρL·uL·(sL - uL))
       / (ρR·(sR - uR) - ρL·(sL - uL))
```

(All quantities in the 1D-normal-tangential frame; the pipe opening supplies a single normal direction, no tangential component is exchanged because the pipe cell is 1D.)

### 3.3 New device buffer

Add to `SWE2DDeviceState` (swe2d_gpu.cuh:548):

```cpp
struct PipeFaceCoupling {
    int32_t  n_faces = 0;
    int32_t* d_face_node_id  = nullptr;   // [n_faces] → pipe1d node index
    int32_t* d_face_cell_id  = nullptr;   // [n_faces] → 2D cell index
    int32_t* d_face_pipe_cell = nullptr;  // [n_faces] → open-end pipe sub-cell
    int32_t* d_face_node_type = nullptr;  // [n_faces] → 0=inlet, 1=pipe_end, 2=outfall, 3=storage, 4=junction
    double*  d_face_invert_elev = nullptr; // [n_faces] pipe invert
    double*  d_face_opening_w = nullptr;  // [n_faces] opening width
    double*  d_face_opening_area = nullptr;// [n_faces] opening area (full pipe)
    double*  d_face_k_in   = nullptr;      // [n_faces] entrance loss
    double*  d_face_k_out  = nullptr;      // [n_faces] exit loss
    // Outfall-only:
    double*  d_face_outfall_mode = nullptr; // [n_faces] 0=free, 1=fixed_wse, 2=rating
    double*  d_face_outfall_fixed_wse = nullptr;
    double*  d_face_outfall_rating_n = nullptr;
    double*  d_face_outfall_rating = nullptr;
    // Junction-only:
    double*  d_face_rim_elev = nullptr;   // manhole rim
    double*  d_face_node_surface_area = nullptr;
};
```

These arrays are populated by a single new upload function `swe2d_gpu_upload_pipe_face_coupling` that replaces the four separate exchange-param uploads.

### 3.4 New kernel: `swe2d_pipe_face_flux_kernel`

```cpp
__global__ void swe2d_pipe_face_flux_kernel(
    int32_t n_faces,
    const Pipe1DDeviceState* pipe1d,        // for A, Q, h_p, S0 arrays
    const int32_t* face_node_id,
    const int32_t* face_cell_id,
    const int32_t* face_pipe_cell,
    const int32_t* face_node_type,
    const double* face_invert_elev,
    const double* face_opening_w,
    const double* face_opening_area,
    const double* face_k_in,
    const double* face_k_out,
    const double* face_outfall_mode,
    const double* face_outfall_fixed_wse,
    const double* face_outfall_rating_n,
    const double* face_outfall_rating,
    const double* face_rim_elev,
    const double* face_node_surface_area,
    const State*  cell_h,
    const State*  cell_hu,
    const State*  cell_hv,
    const double* cell_zb,
    double gravity, double dt, double h_min,
    int32_t n_cells,
    double* ext_flux_h,
    double* ext_flux_hu,
    double* ext_flux_hv,
    double* d_pipe_end_q_2d   // diagnostic / legacy compat
);
```

Per face, the kernel:

1. Reads 1D state: `A = pipe1d->d_A[c_pipe]`, `Q = pipe1d->d_Q[c_pipe]`, `h_p = pipe1d->d_cell_h[c_pipe]`.
2. Reads 2D state: `h, hu, hv, zb`.
3. Builds left and right primitive states.
4. Dispatches on `node_type`:
   - **inlet** (`type == 0`): 2D → 1D only. The 2D side is the donor. Compute HLLC, write 3-component flux that DEPLETES the 2D cell. Avail-limited by 2D water depth.
   - **pipe_end** (`type == 1`): two-way. 1D is donor if `invert + h_p > zb + h`, else 2D. Compute HLLC, write 3-component flux with correct sign for donor → receiver.
   - **outfall** (`type == 2`): 1D → 2D only. The 2D-side WSE comes from the outfall mode:
     - `free`: downstream WSE = invert (free outfall below pipe end).
     - `fixed_wse`: downstream WSE = `fixed_wse` (tidal boundary).
     - `rating`: downstream WSE from rating table at the current 1D WSE.
   - **storage** (`type == 3`): treat the storage node as a small virtual 2D cell with WSE = `invert + node_depth`. Two-way HLLC between pipe-side and the storage node; the storage node is then coupled to its assigned 2D cell by a separate (existing) surcharge overflow path.
   - **junction** (`type == 4`): when `zb + h > rim_elev`, treat as a weir overflow from the 1D node to the 2D cell; else two-way HLLC with the storage node analogue of storage.
5. Computes HLLC flux: `fh, fhu, fhv`.
6. `atomicAdd(&ext_flux_h[donor], -fh)` etc. — opposite signs for donor/receiver.
7. (Diagnostic) `atomicAdd(&d_pipe_end_q_2d[receiver], fh)` so legacy readback paths still get a `Q`.

### 3.5 Integration with the 2D solver

After the kernel runs, set `dev->use_culvert_face_flux = true` *or* add a new flag `dev->use_pipe_face_flux` (cleaner). The 2D update kernel reads `d_ext_struct_flux_*` (swe2d_gpu.cu:2369) exactly as it does for culverts. No changes needed in the update kernel itself.

Disable the legacy path:
- `swe2d_drainage_inlet_exchange_kernel` — only runs the manhole surcharge accounting (`dw.d_node_delta`), not the inlet capture.
- `swe2d_pipe_end_weir_orifice_kernel` — removed from `swe2d_pipe1d_step`.
- `swe2d_drainage_outfall_exchange_kernel` — replaced by face-flux rating.
- `swe2d_fold_drainage_q_kernel`, `swe2d_fold_pipe_end_q_to_source_kernel`, `swe2d_fold_culvert_mass_to_source_kernel` — superseded.

### 3.6 Mass balance for the storage node case

`storage` and `junction` types have a `node_surface_area` (the manhole storage area). The face-flux kernel exchanges volume between the 1D `node_depth · node_surface_area` and the 2D cell. Two-way HLLC keeps total volume conserved exactly because both sides consume the same `fh`. The surcharge-overflow path (for when 2D WSE exceeds rim) becomes a separate "rim overflow" face that is *between* the storage node and the 2D cell — implemented as a second HLLC solve with the storage node as the left state and the 2D cell as the right state, with a weir-style head correction.

---

## 4. Outfall: keeping the SWMM-compatible rating semantics

The existing `swe2d_drainage_outfall_exchange_kernel` (swe2d_gpu.cu:4706) implements three modes:

- **free**: depth reset to 0 each step. Always out of the network.
- **fixed_wse**: tailwater fixed.
- **stage_discharge**: rating curve `(wse_m, Q_m3s)`.

The face-flux replacement must reproduce these. For free outfall, set the downstream (2D-side) WSE to `invert_elev` — the pipe end is open to air below the invert. For fixed_wse, use the configured elevation. For rating, look up Q from the rating curve at the current pipe WSE and use the resulting 2D WSE as the boundary state.

---

## 5. Plan steps

1. **Define `PipeFaceCoupling` in `swe2d_gpu.cuh`.** One struct holding all the per-face arrays above. Add to `SWE2DDeviceState`.

2. **Add `swe2d_gpu_upload_pipe_face_coupling`** (swe2d_gpu.cu) — host wrapper that takes the populated arrays and fills the device buffers. Replace four existing upload functions (`swe2d_pipe1d_upload_pipe_ends_and_junctions`, the various `swe2d_gpu_upload_drainage_exchange_params` calls).

3. **Add `swe2d_pipe_face_flux_kernel`** (swe2d_gpu.cu) — per the spec in §3.4. Reuse `hllc_flux_cuda_local` for the 2D-on-2D parts; add a helper `hlle_pipe_to_2d_flux` for the 1D/2D case.

4. **Add `swe2d_gpu_apply_pipe_face_flux`** host wrapper (swe2d_gpu.cu) — calls the kernel and sets `dev->use_pipe_face_flux = true`.

5. **Bind to Python** (swe2d_bindings.cpp:1338) — add `swe2d_gpu_upload_pipe_face_coupling`, `swe2d_gpu_apply_pipe_face_flux`.

6. **Wire into runtime** (swe2d/runtime/coupling.py:1697) — replace the source-term coupling at all five node-type paths with a single face-flux upload+launch. Drop the legacy inlet/outfall/pipe-end exchange kernels from the hot path.

7. **Build and run the existing pipe1d tests + GPU drainage tests.** All must pass. Specifically:
   - `test_swe2d_gpu_drainage_network.TestPipeEndExchange.test_wet_pipe_drains_into_dry_surface_cells` (currently passes) — must still pass.
   - `test_swe2d_gpu_drainage_network.TestPipeEndExchange.test_pipe_end_moves_water_downhill` (currently fails) — must now pass.
   - `test_swe2d_gpu_drainage_network.TestGPUInletCapture.*` (all pass) — must still pass.
   - `test_coupling_integration.test_real_pipe1d_readback_at_t0_is_zero` — update to allow `cell_depth == h_min` (already done in this session).

8. **Run the CLI replay** (reference/example_test_project/test_drainage_coupling1.json). Verify:
   - Node 2's coupled 2D cell receives water (currently `h_end == 0`).
   - Mass conservation: pipe volume loss = 2D volume gain over the run.
   - Cell momentum (hu, hv) reflects the inflow direction from the pipe end.

9. **Document** (docs/DRAINAGE_COUPLING_GUIDE.md) — describe the unified coupling path, the HLLC formulation, the outfall/storage/junction semantics.

---

## 6. Risks and rollbacks

- **Numerical stability of the 1D/2D HLLC.** The 1D side has weir/orifice-style flow area bounded by `opening_area`. Slot regime changes the pressure law. The HLLC pressure term for the slot (`P = 0.5·g·(w_slot·h_slot²)`) needs care. Risk: oscillations at the 1D/2D interface. Mitigation: limit flux magnitude per CFL on both sides, mirror the culvert kernel's `alpha · h_limit · A_donor / dt` cap.

- **Performance.** Five kernel calls per step (one per node type) replaced by one call. Net win.

- **Existing tests break.** Some weir/orifice-specific assertions in `TestGPUInletCapture` (e.g. `test_inlet_grate_weir` checking `Q = 3·P·H^1.5`) may need updates if the face-flux kernel produces slightly different Q at small H. Mitigation: compare to tolerance, update the magic constants only if the physics actually changed (e.g. the face-flux uses the donor cell's depth, which is exactly the H in the HEC-22 formula, so the values should match).

- **The pipe-end test data uses `node_type='outfall'`.** The face-flux path activates based on `node_type='pipe_end'` or `node_type='outfall'` (both). The GUI loader would need to set `pipe_end_id` for outfall nodes too. OR: the face-flux kernel treats outfall the same as pipe_end (both are open-ended exchanges with the 2D surface). Recommend: do not require node_type to be exactly 'pipe_end' — accept any node with a coupled 2D cell.

---

## 7. Why this fixes the original bug

The bug report (compression note 2026-07-17) said:
> "no outflow at node 2 ... pipe-end invert should match 2D cell bed - should fail loudly if mismatch"

The bug had two layers:

1. **The snap fix I added** (moving pipe_end_invert_elev up to the 2D cell bed) addresses *the terrain mismatch symptom*. After the snap, node 2's coupled 2D cell is *physically reachable* from the pipe end.

2. **The source-term coupling** is the actual reason no flow reaches the 2D cell: the `q_cap_pipe = A·dx/dt` cap was 0 because `init_area_from_depth(0.0)` set A=0. The `init_area_from_depth(self._h_min)` fix I added lets the cap be `0.1·dx/dt` (tiny but non-zero) so some flow gets out. But it's still source-term — a 5 m/s source rate into a 2D cell, capped to zero by the 2D solver's `source_rate_cap`.

Face-flux coupling removes both issues:
- The flux `fh` is in L³/T (volume rate) and is computed from BOTH states consistently.
- The 2D solver consumes it as a *flux at a face*, not as a *depth rate source*. No source caps apply.
- Continuity is exact: same `fh` enters the 1D side and leaves the 2D side.

---

## 8. Out of scope

- **Culvert replacement.** The existing culvert face-flux kernel already works for culverts. This plan does not change culvert handling.
- **SWMM compatibility at the BC level.** The face-flux path produces identical results to a hand-computed HLLC solve at the interface; it does not exactly reproduce SWMM's weir/orifice numbers at every test point. That is the intended trade-off — physics over fitting.
- **2D mesh modification.** No 2D mesh changes. The coupling cells are identified by `pipe_end_cell` and registered once.