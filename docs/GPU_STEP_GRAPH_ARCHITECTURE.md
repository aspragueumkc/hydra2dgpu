# GPU Step Graph & Graph Replay Architecture

> **Generated**: 2026-06-08  
> **Scope**: Complete GPU step-graph audit covering FVM step, time integrators, coupling orchestration, hydraulic structures, drainage networks, and all reconstruction/temporal/spatial discretization combinations.  
> **Source files**: `cpp/src/swe2d_gpu.cu`, `cpp/src/swe2d_gpu.cuh`, `cpp/src/swe2d_solver.cpp`, `swe2d/runtime/coupling.py`, `swe2d/extensions/structures.py`, `swe2d/extensions/drainage_network.py`

---

## 1. Overview

The SWE2D GPU solver has a layered step-graph architecture:

```
Python Orchestration (coupling.py)
    │
    ▼
C++ Dispatcher (swe2d_solver.cpp)
    │  ├── Choose temporal_order → RK method
    │  ├── Choose spatial_scheme → reconstruction type
    │  ├── Choose godunov_mode → rollout contract
    │  ├── Choose tiny_mode → fused/persistent paths
    │  └── Choose coupling_loop → CPU/GPU coupling
    │
    ▼
CUDA Kernel Graph (swe2d_gpu.cu)
    ├── Single-stage (swe2d_gpu_step) — capturable as CUDA graph
    ├── RK2 (swe2d_gpu_step_rk2) — capturable as CUDA graph per stage
    ├── RK4 composed (swe2d_gpu_step_rk4) — NOT graph-capturable (D→D copies)
    ├── RK4 graph-safe (swe2d_gpu_step_rk4_graph) — capturable as single graph
    ├── RK5 graph-safe (swe2d_gpu_step_rk5_graph) — capturable as single graph
    ├── Godunov rollouts — wrappers enforcing ≥ MUSCL-MinMod + shallow-front fallback
    └── Tiny-N paths — cooperative persistent kernels for small meshes
```

Each GPU timestep is a **kernel graph** — a sequence of CUDA kernel launches that is either launched individually or captured into a `cudaGraphExec_t` for amortized replay.

---

## 2. Base Euler Step (`swe2d_gpu_step`)

This is the fundamental single-step (forward Euler) GPU timestep. All higher-order time integrators compose it. The kernel sequence is:

### 2.1 Kernel Launch Order

```
┌─────────────────────────────────────────────────────────────────────┐
│                      swe2d_gpu_step (1 Euler step)                  │
├─────────────────────────────────────────────────────────────────────┤
│  PHASE 1: Active-set classification                                 │
│  ┌────────────────────────────────────────────────────────────      │
│  │ 1a. Copy d_was_active ← d_active   (if hysteresis enabled)       │
│  │ 1b. cudaMemsetAsync(d_n_wet, 0)                                  │
│  │ 1c. swe2d_classify_and_mark_kernel                               │
│  │    • For each cell: classify wet/dry per h>h_min                 │
│  │    • Check rain + external + struct_flux sources                 │
│  │    • Handle BC-forced cells                                      │
│  │    • Count wet cells into d_n_wet                                │
│  │ 1d. swe2d_degen_deactivate_kernel   (if degen_mode ∈ {1,3})     │
│  │    • Deactivate degenerate cells                                 │
│  │ 1e. swe2d_degen_sync_kernel          (if degen_mode == 3)        │
│  │    • Merge degenerate state to owner                             │
│  └────────────────────────────────────────────────────────────      │
│                                                                      │
│  PHASE 2: Boundary condition update                                 │
│  ┌────────────────────────────────────────────────────────────      │
│  │ 2a. swe2d_apply_hydrograph_bc_kernel  (if n_hg_edges > 0)       │
│  │    • Interpolate time-varying BCs at t_now                      │
│  │    • Write edge_bc[hg_edge], edge_bc_val[hg_edge]               │
│  └────────────────────────────────────────────────────────────      │
│                                                                      │
│  PHASE 3: Gradient pre-pass (if spatial_scheme >= 1)                │
│  ┌────────────────────────────────────────────────────────────      │
│  │ 3a. cudaMemsetAsync for 6 gradient arrays (d_grad_hx..hy,       │
│  │     d_grad_hux..uy, d_grad_hvx..vy)                             │
│  │ 3b. swe2d_gradient_kernel (Green-Gauss divergence theorem)      │
│  │    • Edge-parallel: for each edge, accumulate (q_c0+q_c1)/2     │
│  │      × n_hat × len to both cells' gradient accumulators         │
│  │      using atomicAdd                                             │
│  │    • Then divide by cell area per cell (CAS atomics)            │
│  │ 3c. swe2d_maybe_launch_lsq_gradient  (if scheme >= 5)          │
│  │    • Least-squares gradient (experimental, WENO3-like path)     │
│  └────────────────────────────────────────────────────────────      │
│                                                                      │
│  PHASE 4: Flux computation                                          │
│  ┌────────────────────────────────────────────────────────────      │
│  │ 4a. swe2d_flux_kernel (edge-parallel)                           │
│  │    • For each edge e between c0, c1:                            │
│  │      1. Read h, hu, hv, zb for c0, c1                          │
│  │      2. If c1 < 0 (boundary): ghost_state from BC type         │
│  │      3. hydrostatic_reconstruct_cuda_local                     │
│  │         └─ Surface-gradient method: η = h + zb → ∇η →          │
│  │            η_L = η_c0 + φ·∇η·(x_f - x_c0) → h = η - zb        │
│  │      4. hllc_flux_cuda_local (HLLC Riemann solver)            │
│  │      5. bed_slope_correction_cuda_local                       │
│  │      6. Write: flux_h[e], flux_hu[e], flux_hv[e],             │
│  │                flux_hu_r[e], flux_hv_r[e]                      │
│  └────────────────────────────────────────────────────────────      │
│                                                                      │
│  PHASE 5: Cell update                                               │
│  ┌────────────────────────────────────────────────────────────      │
│  │ 5a. cudaMemsetAsync(d_max_wse_elev_error, 0)                    │
│  │ 5b. swe2d_update_kernel (cell-parallel)                         │
│  │    • For each cell c: walk cell_edge_offsets[c]/ids[c]         │
│  │    • Accumulate fluxes (sign depends on edge orientation)      │
│  │    • Add: dt * inv_area (source + external_source)            │
│  │    • Apply depth_cap, max_rel_depth_increase                   │
│  │    • Apply shallow_damping (smooth cubic transition)           │
│  │    • Apply friction (semi-implicit, per cell)                  │
│  │    • Apply momentum cap                                        │
│  │    • Track max_wse_elev_error for diagnostic                   │
│  └────────────────────────────────────────────────────────────      │
│                                                                      │
│  PHASE 6: CFL reduction + diagnostics                              │
│  ┌────────────────────────────────────────────────────────────      │
│  │ 6a. cudaMemsetAsync(d_lambda_max, 0)                           │
│  │ 6b. swe2d_cfl_kernel (edge-parallel)                           │
│  │    • Compute max wave speed per edge, write block max         │
│  │ 6c. swe2d_cfl_reduce_blocks_kernel (single block)             │
│  │    • Reduce block maxima → d_lambda_max                       │
│  │ 6d. pack_diag_kernel (single thread)                          │
│  │    • Pack d_lambda_max, d_max_wse_elev_error, d_n_wet         │
│  │      into contiguous d_diag_packed buffer                     │
│  └────────────────────────────────────────────────────────────      │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 CUDA Graph Capture Scope

When `enable_kernel_graphs == true` and no per-edge debug flags are set, PHASES 1–6 are captured into a **single CUDA graph**. The capture scope is:

```
cudaStreamBeginCapture(stream)
    sweep: classify_and_mark → degen_deactivate → degen_sync → hg_bc → 
           gradient (memset + kernel) → flux → update → 
           cfl → cfl_reduce → pack_diag
cudaStreamEndCapture(stream)
```

The graph is cached per `(n_cells, n_edges, spatial_scheme, time_integrator, variant_key, config_signature)`.

### 2.3 Variant Key

```
variant_key = (has_hydrograph ? 1 : 0) | (needs_gradient ? 2 : 0)
```

- Bit 0: Hydrograph BCs present (affects hg_bc kernel launch)
- Bit 1: Gradients needed (scheme ≥ 1, enables gradient memset + kernel)

### 2.4 Config Signature

`swe2d_kernel_graph_signature()` hashes all scalar runtime parameters:
- `dt`, `g`, `h_min`, `cfl_lambda_cap`
- `max_inv_area`, `momentum_cap_min_speed`, `momentum_cap_celerity_mult`
- `depth_cap`, `max_rel_depth_increase`, `shallow_damping_depth`
- `extreme_rain_mode`, `source_cfl_beta`, `source_max_substeps`
- `source_rate_cap`, `source_depth_step_cap`
- `source_true_subcycling`, `source_imex_split`
- `enable_shallow_front_recon_fallback`, `front_flux_damping`
- `use_culvert_face_flux`

Any change in these parameters forces a graph re-capture on the next step.

---

## 3. Time Integrators

Each integrator is a C++ wrapper that calls `swe2d_gpu_step` (or its graph-safe equivalent) one or more times. The relationship is:

| `temporal_order` | C++ function called | graph-capturable? | `time_integrator` cache tag |
|:---------------:|---------------------|:-----------------:|:-------------------------:|
| 1 | `swe2d_gpu_step` | ✅ (single graph) | 1 |
| 2 | `swe2d_gpu_step_rk2` | ✅ (per stage) | 2 |
| 4 | `swe2d_gpu_step_rk4` | ❌ (D→D copies) | 4 |
| 5 | `swe2d_gpu_step_rk4_graph` | ✅ (single 4-stage) | 5 |
| ≥6 | `swe2d_gpu_step_rk5_graph` | ✅ (single 6-stage) | 6 |

### 3.1 RK2 (SSPRK2 Heun, `temporal_order=2`)

```
swe2d_gpu_step_rk2:
  1. Save U^0 → U_h0 (d_h0, d_hu0, d_hv0)    [D→D memcpy]
  2. swe2d_gpu_step(U^0, t, dt) → U^1         [graph-capturable]
  3. Save CN cumulative state                  [D→D memcpy, conditional]
  4. swe2d_gpu_step(U^1, t+dt, dt) → U^2      [graph-capturable]
  5. swe2d_rk2_combine_kernel: U_new = 0.5*(U^0 + U^2)
  6. Restore CN cumulative state               [D→D memcpy, conditional]
  7. CFL kernel + pack_diag
```

Each of the two `swe2d_gpu_step` calls can independently use CUDA graph replay if the cache matches.

### 3.2 RK4 Composed (`temporal_order=4`)

```
swe2d_gpu_step_rk4:
  1. Save U^0 → U_h0
  2. swe2d_gpu_step(U^0, t, dt/2) → U_mid1    [stage 1, k1 capture]
  3. save U_mid1 → U_h1 (k1)
  4. swe2d_rk4_capture_increment_kernel
  5. save midpoint → U_h3
  6. swe2d_gpu_step(U_mid1, t+dt/2, dt/2) → U_mid2  [stage 2, k2 capture]
  7. swe2d_rk4_capture_increment_kernel → U_h2 (k2)
  8. swe2d_rk4_build_stage_kernel → reconstruct stage
  9. swe2d_gpu_step(stage, t+dt/2, dt/2) → U_mid3   [stage 3, k3 capture]
  10. swe2d_rk4_shift_from_reference_kernel
  11. swe2d_gpu_step(stage, t+dt, dt/2) → U_final   [stage 4, k4 capture]
  12. swe2d_rk4_combine_kernel: y_new = y0 + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
  13. CFL kernel + pack_diag
```

**NOT graph-capturable** because of the intermixed D→D memcpy calls between stages. The per-stage `swe2d_gpu_step` calls can individually use graph replay.

### 3.3 RK4 Graph-Safe (`temporal_order=5`)

```
swe2d_gpu_step_rk4_graph:
  1. Save U^0 → U_h0   [D→D memcpy]
  2. Precompute stage forcing:
     - Copy d_edge_bc/d_edge_bc_val → 4 stage slots    [D→D memcpy × 8]
     - Apply hydrograph BCs at each stage time          [kernel × 4]
     - Compute stage rain rates                          [kernel × 5]
  3. Wet/dry classification (outside graph)
  4. CAPTURED GRAPH (single capture, time_integrator=5):
     For each stage s=0..3:
       a. compute_coupling()
          - Zero d_ext_struct_flux_*
          - swe2d_coupling_wse_from_state_kernel
          - swe2d_apply_enquiry_wse_kernel
          - swe2d_compute_structure_flows_kernel
          - swe2d_culvert_face_flux_kernel
          - swe2d_mask_culvert_source_kernel
       b. evaluate_rhs()
          - Gradient memset + kernel (if scheme≥1)
          - Flux memset + swe2d_flux_kernel
          - swe2d_rk4_rhs_collect_kernel
       c. stage_build_kernel (combine into next stage state)
     Final: swe2d_rk4_graph_combine_kernel
  5. Final rain CN update + CFL + pack_diag
```

**This is the most sophisticated graph-capturable path.** The entire 4-stage RK loop with per-stage structure recomputation (face-based culvert flux) is captured as a single `cudaGraphExec_t`.

### 3.4 RK5 Graph-Safe (`temporal_order ≥ 6`)

Same architecture as RK4 graph-safe but with 6 stages (Cash-Karp coefficients) and `time_integrator=6`. The `evaluate_rhs` lambda is identical. The stage build uses `swe2d_rk_multi_stage_build_kernel` with Butcher tableau coefficients.

---

## 4. Godunov Rollout Path

The Godunov mode enforces a stricter numerical contract:

| Property | Normal Path | Godunov Rollout |
|----------|-------------|-----------------|
| Min spatial scheme | User-selected (0-5) | **≥ 2** (MUSCL-MinMod minimum) |
| `enable_shallow_front_recon_fallback` | User-selected | **Always true** |
| RK method | User-selected (1-6) | RK2 minimum |

The wrappers are thin:

```cpp
void swe2d_gpu_step_godunov_rollout(...) {
    const int rollout_scheme = (spatial_scheme < 2) ? 2 : spatial_scheme;
    swe2d_gpu_step(dev, ..., rollout_scheme, ..., /*shallow_front=*/true, ...);
}

void swe2d_gpu_step_rk2_godunov_rollout(...) {
    const int rollout_scheme = (spatial_scheme < 2) ? 2 : spatial_scheme;
    swe2d_gpu_step_rk2(dev, ..., rollout_scheme, ..., /*shallow_front=*/true, ...);
}
```

The dispatch in `swe2d_solver.cpp` is:

```
if (use_godunov_rollout):
    if (use_rk2):              → swe2d_gpu_step_rk2_godunov_rollout()
    else (single-stage):       → swe2d_gpu_step_godunov_rollout()
```

**Key consequence**: Godunov rollout disables RK4 composed (order 4), RK4 graph (order 5), and RK5 graph (order ≥6) — they fall through to RK2. Graph-capturable RK4_graph/RK5_graph paths are NOT used in Godunov mode.

---

## 5. Tiny-N Optimized Paths

Three execution modes for small meshes where launch overhead dominates:

| `tiny_mode` | Description | When Active | Graph-capturable? |
|:-----------:|-------------|-------------|:-----------------:|
| 0 (off) | Standard path | Always available | ✅ |
| 1 (auto) | Automatic selection | `n_cells ≤ threshold` AND `n_edges ≤ threshold` | ❌ (fallback) |
| 2 (fused) | Fused kernels | Explicit | ❌ (uses fused kernels) |
| 3 (persistent) | Cooperative persistent kernel | Explicit, first-order only | ❌ (cooperative launch) |

### 5.1 Fused Mode (tiny_mode=2)

Supported only for single-stage (Euler, no RK2/4/5) non-hydrostatic paths. Merges edge-centric and cell-centric work into fewer kernel launches. Gradient remains separate.

### 5.2 Persistent Chunk Mode (tiny_mode=3)

Uses `swe2d_persistent_chunk_kernel_first_order` — a **cooperative grid** kernel that performs multiple substeps in a single launch:

```
cudaLaunchCooperativeKernel(swe2d_persistent_chunk_kernel_first_order, blocks, threads, args)

Internal loop (device-side, no host intervention):
  for sub = 0..chunk_substeps:
    • First-order flux computation (inlined HLLC)
    • Cell update (inlined accumulation + sources + friction + momentum cap)
    • grid.sync() between phases
```

**Constraints**:
- First-order spatial ONLY (`spatial_scheme == 0`)
- No extreme_rain_mode, source_true_subcycling, or source_imex_split
- Falls back to chunked `swe2d_gpu_step` calls if hardware doesn't support cooperative launch

---

## 6. CUDA Graph Capture/Replay Mechanism

### 6.1 Cache Structure

```cpp
struct KernelGraphCache {
    cudaGraph_t       graph;              // Captured graph template
    cudaGraphExec_t   exec;               // Executable instance
    int32_t           n_cells;            // Mesh size at capture
    int32_t           n_edges;            // Edge count at capture
    int32_t           spatial_scheme;     // Scheme at capture
    int32_t           time_integrator;    // RK order tag
    int32_t           variant_key;        // hg + gradient bits
    uint64_t          config_signature;   // Scalar parameter hash
    bool              is_valid;           // Ready for replay
};
```

### 6.2 Cache Lookup Flow

```
swe2d_gpu_step (or wrapper):
  if try_kernel_graph && cache_match:
    cudaGraphLaunch(exec, stream)
    return  ← zero kernel launch overhead
  else:
    cache.destroy()
    cudaStreamBeginCapture(stream)
    // execute all kernels
    cudaStreamEndCapture(stream)
    cudaGraphInstantiate(&exec, graph)
    cache = {...}
    cudaGraphLaunch(exec, stream)
```

### 6.3 Cache Invalidation Triggers

| Trigger | Detection | Action |
|---------|-----------|--------|
| Mesh size change | `n_cells` / `n_edges` mismatch | Re-capture |
| Spatial scheme change | `spatial_scheme` mismatch | Re-capture |
| Time integrator change | `time_integrator` mismatch | Re-capture |
| BC presence change | `variant_key` mismatch | Re-capture |
| Any scalar param change | `config_signature` mismatch | Re-capture |
| Debug flags enabled | `dbg_edge_flux` / `dbg_flux_summary` | Skip graph |

### 6.4 Graph-Replay-Capable Paths Summary

| Path | Graph-capturable? | Scope of Capture | Integrator Tag |
|------|:-----------------:|------------------|:--------------:|
| `swe2d_gpu_step` (Euler, order=1) | ✅ | Full classify→pack (6 phases) | 1 |
| `swe2d_gpu_step_rk2` (order=2) | ✅ per stage | Each `swe2d_gpu_step` call gains replay | 2 |
| `swe2d_gpu_step_rk4` (order=4) | ❌ | Composed, intermixed D→D copies | 4 |
| `swe2d_gpu_step_rk4_graph` (order=5) | ✅ single graph | All 4 stages + coupling + combine | 5 |
| `swe2d_gpu_step_rk5_graph` (order ≥ 6) | ✅ single graph | All 6 stages + coupling + combine | 6 |
| Godunov rollouts | ✅ (delegated) | Same as underlying Euler/RK2 | 1 or 2 |
| Tiny persistent chunk | ❌ | Cooperative kernel, not graph-compatible | N/A |
| Tiny fused | ❌ | Fused kernels, not graph-compatible | N/A |

### 6.5 Graph Replay Count Tracking

Two diagnostics track graph replay efficiency:
- `diag.gpu_graph_launches_step` — number of graph replays this step (0 or 1 for Euler, 0-2 for RK2)
- `diag.gpu_graph_launches_total` — cumulative count across all steps

---

## 7. Coupling Orchestration

### 7.1 Python Dispatch (`SWE2DCouplingController.compute_source_rates`)

```
compute_source_rates(t_s, dt_s, h):
  │
  ├─ coupling_loop == "cuda" AND native module available?
  │   │
  │   ├─ YES → _compute_source_rates_cuda(mod, t_s, dt_s, hh)
  │   │   │
  │   │   ├─ Coupling persistent path (apply_native_device_sources)?
  │   │   │   ✅ Struct only, no drainage, no bridges, CUDA loop
  │   │   │   → swe2d_gpu_compute_coupling_full_on_device()
  │   │   │     reads d_h/d_cell_zb on-device, computes structure flows
  │   │   │     → atomicAdd into d_external_source_mps
  │   │   │     → swe2d_gpu_redistribute_structure_sources_persistent()
  │   │   │     Returns True → source array already on device
  │   │   │
  │   │   └─ Fallback CUDA path:
  │   │       1. Drainage: swe2d_gpu_drainage_step() (GPU)
  │   │          OR drainage.surface_exchange_source_rate() (CPU)
  │   │       2. Structures: swe2d_gpu_compute_structure_flows() (GPU)
  │   │          OR _native_structure_flows() with host WSE readback
  │   │       3. Redistribution (CUDA kernel or Python fallback)
  │   │
  │   └─ NO → CPU path:
  │       ├─ drainage.solve_network_step(dt)
  │       ├─ drainage.surface_exchange_source_rate()
  │       ├─ _native_structure_flows(use_cuda=False) or structure_flows()
  │       ├─ _structure_source_rate_from_flows()
  │       └─ Redistribution (Python loop)
```

### 7.2 Persistent On-Device Fast Path

The fastest coupling path eliminates ALL host-device transfers:

```python
def apply_native_device_sources(self, t_s, dt_s) -> bool:
    if coupling_loop != "cuda":               return False
    if structures is None and drainage is None: return False
    if has_enabled_bridge_structures:          return False
    if not swe2d_gpu_compute_coupling_full_on_device: return False
    swe2d_gpu_compute_coupling_full_on_device(...)
    return True  # sources already on device
```

**Eligibility gates**:
1. `coupling_loop == "cuda"`
2. `structures is not None` OR `drainage is not None`
3. `_has_enabled_bridge_structures == False`
4. Native module has `swe2d_gpu_compute_coupling_full_on_device` binding

**Current limitation**: When `drainage is not None`, the persistent path returns `False` — drainage is not yet integrated into the on-device fast path.

### 7.3 Source Injection in Update Kernel

The `swe2d_update_kernel` receives coupling source arrays:

```cpp
swe2d_update_kernel<<<...>>>(...,
    dev->d_cell_source_mps,          // rainfall source [m/s]
    dev->d_external_source_mps,      // coupling source [m/s]
    dev->d_ext_struct_flux_h,        // face-based culvert flux h
    dev->d_ext_struct_flux_hu,       // face-based culvert flux hu
    dev->d_ext_struct_flux_hv        // face-based culvert flux hv
);
```

The update kernel applies sources in this order:
1. Flux accumulation (Riemann flux)
2. Rainfall source (`d_cell_source_mps`)
3. External coupling source (`d_external_source_mps`) — both drainage + structures
4. Face-based culvert flux (`d_ext_struct_flux_*`)
5. Depth capping, relative increase cap
6. Shallow damping (smooth cubic transition)
7. Semi-implicit friction
8. Momentum cap

---

## 8. Structure Coupling

### 8.1 Structure Types and Flow Models

| Type | Enum | Flow Equation |
|------|:----:|---------------|
| Culvert | 0 | FHWA HDS-5 inlet/outlet control, 58 codes. GPU: secant (mode 0) or lookup table (mode 1) |
| Weir | 1 | $Q = C_d B \sqrt{2g h^3}$ (submerged/unsubmerged) |
| Gate/Orifice | 2 | Orifice flow through opening |
| Bridge | 3 | Deck obstruction + underdeck loss-based damping |
| Pump | 4 | Fixed Q or variable per head |

### 8.2 GPU Structure Flow Computation

Two coupling modes for structures on GPU:

#### 8.2.1 Point-Source Injection (Legacy)

```
Host-side per timestep:
  1. Read d_h, d_cell_zb → host WSE array          [D→H memcpy]
  2. swe2d_gpu_compute_structure_flows()             [GPU kernel]
  3. Read d_structure_flow back                      [D→H memcpy]
  4. Python: structure_source_rate_from_flows()      [CPU]
  5. Upload source_rate → d_external_source_mps      [H→D memcpy]
```

#### 8.2.2 Persistent On-Device (Fast Path)

```
On first call:
  1. swe2d_gpu_preload_structure_params(...)        [H→D upload]
  2. swe2d_gpu_preload_coupling_cell_area(...)      [H→D upload]

Each timestep (zero host transfers):
  3. swe2d_gpu_compute_coupling_full_on_device()
     └─ reads d_h, d_cell_zb on-device
     └─ computes structure flows
     └─ atomicAdd → d_external_source_mps
```

#### 8.2.3 Face-Based Culvert Flux (Inside Graph Capture)

When `culvert_face_flux_mode == "face_flux"` and `use_culvert_face_flux == true`, structure coupling is embedded **inside the CUDA graph capture** for RK4_graph and RK5_graph paths:

```
compute_coupling() [before each evaluate_rhs inside the graph]:
  1. cudaMemsetAsync(d_ext_struct_flux_*, 0)
  2. swe2d_coupling_wse_from_state_kernel: WSE = h + zb
  3. swe2d_apply_enquiry_wse_kernel: total-energy driving head
  4. swe2d_compute_structure_flows_kernel: evaluate all structures
  5. swe2d_culvert_face_flux_kernel: apply as face fluxes
  6. swe2d_mask_culvert_source_kernel: zero structure flow for point-source path
```

This is the **only path where structure coupling is inside the graph capture**.

### 8.3 Bridge Stacked Coupling

Bridges use a separate stratified mesh approach:
1. `bridge_stacked_source_scale()` computes `opening_fraction × layer_scale`
2. `apply_bridge_stacked_phase3_source_weight()` distributes flow to corridors
3. Requires `bridge_cuda_coupling=True` (disabled by default)

### 8.4 Corridor Redistribution

Without redistribution, structure flows are point sources creating velocity jets:

```
Single-cell injection:
  source_rate[up_cell]   += +Q / area    [water removed]
  source_rate[dn_cell]   += -Q / area    [water added]

Redistribution:
  1. Reverse point injection
  2. For each corridor cell: source_rate[cell] += w_i * Q / area_i
```

---

## 9. Drainage Network Coupling

### 9.1 Network Topology Elements

| Element | Surface Connection | Flow Direction |
|---------|-------------------|----------------|
| Inlet | 2D cell → network node | Surface → drainage (capture) |
| Outfall | Network node → 2D cell | Drainage → surface (discharge) |
| Pipe End | Bidirectional 2D cell ↔ node | Both directions |

### 9.2 Drainage Solver Modes

| Mode | Name | Equation | Stability |
|:----:|------|----------|:---------:|
| 0 | EGL (Energy Grade Line) | $\Delta H = Q^2 \left[\frac{n^2 L}{A^2 R_h^{4/3}} + \frac{K_e+K_o}{2gA^2}\right]$ | Implicit, stable |
| 1 | Diffusion Wave | $Q = \frac{1}{n} A R_h^{2/3} \sqrt{S_w}$ | Explicit, CFL-limited |
| 2 | Dynamic Wave | $Q^{n+1} = \frac{Q^n + \Delta t \, g A \, \frac{\Delta H}{L}}{1 + \Delta t \, g n^2 \lvert Q^n \rvert / (A R_h^{4/3})}$ | Semi-implicit |

### 9.3 GPU Drainage Kernels

| Kernel | Function |
|--------|----------|
| `swe2d_drainage_node_update_kernel` | Continuity: $h^{n+1} = h + \Delta t \cdot Q_{net} / A_{surface}$ |
| `swe2d_drainage_pipe_end_qleave_kernel` | Accumulate link flows → node $Q_{leave}$ |
| `swe2d_drainage_pipe_end_bc_kernel` | Compute pipe-end effective WSE from 2D state with losses |
| `swe2d_drainage_pipe_end_exchange_kernel` | Exchange flow: 2D cell ↔ pipe-end node |
| `swe2d_drainage_apply_delta_kernel` | Apply per-cell source deltas → `d_external_source_mps` |

### 9.4 GPU Drainage Step (`swe2d_gpu_drainage_step`)

The full drainage step binding:
1. Reads `d_h`, `d_cell_zb` from device → computes WSE on-device
2. Runs all 5 drainage kernels
3. Returns `node_depth`, `link_flow`, `q_cell` arrays

**Current limitation**: The GPU drainage step is NOT integrated into the persistent on-device fast path. When `drainage is not None`, `apply_native_device_sources()` returns `False`.

---

## 10. Spatial Reconstruction Methods

### 10.1 Scheme Overview

| Scheme | Value | Reconstruction | Limiter | Accuracy |
|--------|:-----:|----------------|---------|:--------:|
| `FV_FIRST_ORDER` | 0 | None (piecewise-constant) | — | 1st |
| `FV_MUSCL_FAST` | 1 | Green-Gauss gradient | Superbee | 2nd |
| `FV_MUSCL_MINMOD` | 2 | Green-Gauss gradient | Minmod | 2nd |
| `FV_MUSCL_MC` | 3 | Green-Gauss gradient | Monotonized-Central | 2nd |
| `FV_MUSCL_VAN_LEER` | 4 | Green-Gauss gradient | Van Leer (smooth) | 2nd |
| `FV_WENO3_LIKE` | 5 | Green-Gauss + WENO blend | Nonlinear weights | ~2nd |

### 10.2 GPU Reconstruction Formula

```cpp
// hydrostatic_reconstruct_cuda_local()
// Step 1: Surface-gradient method (Zhou et al. 2001)
η_c0 = h_c0 + zb_c0
∇η_c0 = [grad_hx[c0] + grad_zb_x[c0], grad_hy[c0] + grad_zb_y[c0]]

// Step 2: Slope ratio r = (∇η · Δx_pair) / (η_c1 − η_c0)

// Step 3: Apply limiter φ(r)
double phi = limiter_function(r);

// Step 4: Extrapolate to actual face midpoint (geometrically correct)
η_L = η_c0 + phi · ∇η_c0 · (x_face − x_c0)
h_L = η_L − zb_c0

// Step 5: Pair-bounds clamping (safety)
h_L = clamp(h_L, min(h_c0, h_c1), max(h_c0, h_c1))
```

### 10.3 GPU vs CPU Reconstruction

| Aspect | GPU | CPU |
|--------|-----|-----|
| Extrapolation target | Actual face midpoint $\vec{x}_f$ | Pair midpoint |
| Formula | $q_f = q_0 + \phi \nabla q_0 \cdot (\vec{x}_f - \vec{x}_{c0})$ | $q_f = q_0 + \phi \cdot 0.5(q_1 - q_0)$ |
| Identical L/R states | No | **Yes** — causes neutral/unstable HLLC |

### 10.4 Gradient Computation

```cpp
swe2d_gradient_kernel:
  // Edge-parallel Green-Gauss divergence theorem
  for each edge e between c0, c1:
    face_val = 0.5 * (q_c0 + q_c1)
    atomicAdd(grad_qx[c0], face_val * edge_nx[e] * edge_len[e])
    atomicAdd(grad_qy[c0], face_val * edge_ny[e] * edge_len[e])
    atomicAdd(grad_qx[c1], -face_val * edge_nx[e] * edge_len[e])
    atomicAdd(grad_qy[c1], -face_val * edge_ny[e] * edge_len[e])

  // Per-cell division (CAS atomics)
  for each cell c:
    grad_qx[c] /= area[c];  grad_qy[c] /= area[c];
```

On non-orthogonal meshes, the face-average has $O(h)$ error because $\vec{d}_{c0\to f} + \vec{d}_{c1\to f} \neq 0$.

### 10.5 Scheme vs. Graph Replay Interaction

All schemes 1–5 share the same gradient-bit variant key. The `spatial_scheme` field further partitions the cache — so scheme 1 and scheme 4 have separate graph cache entries even though they share the gradient pass.

---

## 11. Cross-Component Interactions

### 11.1 Per-Timestep Data Flow

```
t = t_now
│
├─ (1) Determine dt via CFL computation
│
├─ (2) [Coupling] compute_source_rates(t, dt, h)
│     ├─ Drainage network → per-cell source [m/s]
│     ├─ Structures → per-cell source [m/s]
│     ├─ Redistribution → per-cell source [m/s]
│     └─ Returns total source_rate → d_external_source_mps
│
├─ (3) swe2d_gpu_step_*(t, dt, ...)
│     ├─ classify_and_mark (reads d_h, d_external_source_mps)
│     ├─ hydrograph BC update
│     ├─ gradient (if scheme ≥ 1)
│     ├─ flux (HLLC Riemann solver)
│     ├─ update (adds rain + external sources + struct face fluxes)
│     └─ CFL + diagnostics
│
└─ t += dt
```

### 11.2 Per-Stage Structure Coupling (RK4_graph / RK5_graph)

For graph-safe RK paths, structure coupling is recomputed **at each RK stage**:

```
Stage 0: compute_coupling() → evaluate_rhs() → build_stage(...)
Stage 1: compute_coupling() → evaluate_rhs() → build_stage(...)
Stage 2: compute_coupling() → evaluate_rhs() → build_stage(...)
Stage 3: compute_coupling() → evaluate_rhs()
Final:   rk4_graph_combine_kernel()
```

For Euler and RK2 paths, structure flows are computed **once per timestep** (outside the graph).

### 11.3 Drainage-Structure Source Separation

In `compute_source_rates()`:
1. Drainage source computed first → accumulated into `total[]`
2. Structure source computed next → accumulated into `total[]`
3. `total[]` returned → solver adds to `d_external_source_mps`

Both are summed into the same array. No ordering guarantee between drainage and structure within a cell.

### 11.4 Face-Flux Mask Interaction

When `use_culvert_face_flux == true`:
```
swe2d_compute_structure_flows_kernel  → d_structure_flow[n_struct]
swe2d_culvert_face_flux_kernel        → d_ext_struct_flux_h/hu/hv[n_cells]
swe2d_mask_culvert_source_kernel      → d_structure_flow[culvert_only] = 0
```

After masking, only non-culvert structure flows remain for the point-source path.

---

## 12. Execution Path Decision Tree

```
swe2d_step()
│
├─ Equation set?
│   ├─ NONHYDROSTATIC_2D → nonhydro_predictor_corrector()
│   └─ SWE2D (default) → continue
│
├─ Godunov mode?
│   ├─ godunov_mode != 0:
│   │   ├─ RK2 → swe2d_gpu_step_rk2_godunov_rollout()
│   │   │          └─ scheme ≥ 2, shallow_front=true
│   │   └─ Single-stage → swe2d_gpu_step_godunov_rollout()
│   │                     └─ scheme ≥ 2, shallow_front=true
│   │
│   └─ godunov_mode == 0 (normal):
│       │
│       ├─ temporal_order:
│       │   ├─ 1 (Euler):
│       │   │   ├─ tiny persistent? → swe2d_gpu_step_persistent_chunk()
│       │   │   │   └─ ONLY scheme=0, no extreme_rain
│       │   │   └─ standard → swe2d_gpu_step() [graph-capturable ✅]
│       │   │
│       │   ├─ 2 (RK2):
│       │   │   ├─ tiny persistent? → swe2d_gpu_step_rk2_persistent_chunk()
│       │   │   └─ standard → swe2d_gpu_step_rk2() [graph per stage ✅]
│       │   │
│       │   ├─ 4 (RK4 composed):
│       │   │   └─ swe2d_gpu_step_rk4() [NOT graph-capturable ❌]
│       │   │
│       │   ├─ 5 (RK4 graph-safe):
│       │   │   └─ swe2d_gpu_step_rk4_graph() [graph-capturable ✅]
│       │   │       └─ per-stage structure coupling ✅
│       │   │
│       │   └─ ≥6 (RK5 graph-safe):
│       │       └─ swe2d_gpu_step_rk5_graph() [graph-capturable ✅]
│       │           └─ per-stage structure coupling ✅
│       │
│       └─ Tiny mode auto: n_cells≤threshold → fused
│           └─ fused only for Euler (non-RK)
│
└─ Coupling (runs BEFORE step):
    ├─ coupling_loop="cuda":
    │   ├─ apply_native_device_sources() if eligible
    │   └─ else: _compute_source_rates_cuda()
    └─ coupling_loop="cpu": Python CPU path
```

---

## 13. Potential Issues and Known Problems

### 13.1 Graph Replay Issues

| Issue | Severity | Detail |
|-------|:--------:|--------|
| **Graph staleness from driver/kernel updates** | High | Graph captures kernel addresses at record time. CUDA driver/kernel updates invalidate silently. |
| **Config signature doesn't cover coupling state** | Medium | Scalar params hashed but NOT coupling arrays. Structure metadata changes between timesteps = stale graph. |
| **RK4 composed not graph-capturable** | Medium | D→D memcpy between stages is incompatible with graph capture. |
| **Graph capture overhead on first step** | Low | First step after config change incurs capture + instantiation overhead. |
| **Graph binary size** | Low | Can be 10s of MB for large meshes. |

### 13.2 Coupling Issues

| Issue | Severity | Detail |
|-------|:--------:|--------|
| **Persistent path silently skipped** | High | Returns `False` with no diagnostic when drainage/bridges active. |
| **Persistent path blocks drainage GPU** | High | `apply_native_device_sources()` returns `False` when drainage active, even though GPU drainage kernels exist. |
| **atomicAdd contention** | High | Multiple structures → same cell → serialized atomics. |
| **Redistribution fallback degrades silently** | Medium | CUDA kernel failure → Python fallback with only a log message, no exception. |
| **Bridge + non-bridge flow ordering** | Medium | `np.add.at` accumulation order matters for redistribution but is not explicit. |
| **Drainage/structures time-step mismatch** | Medium | Coupling uses 2D Δt. Drainage internal substepping opaque to GPU. |

### 13.3 Structure-Specific Issues

| Issue | Severity | Detail |
|-------|:--------:|--------|
| **Unit system sensitivity** | High | Culvert metadata in feet (HDS-5), cell data in model units. Multiple conversion points. |
| **Momentum cap heuristics** | Medium | Empirical parameters; wrong values fail to prevent jets or over-dampen. |
| **Bridge CUDA coupling feature-gated** | Medium | Requires `bridge_cuda_coupling=True`. Default False = host sync. |
| **Culvert outlet-control bypass** | Medium | Table mode (1) approximates outlet control, diverges from full energy solution. |

### 13.4 Drainage-Specific Issues

| Issue | Severity | Detail |
|-------|:--------:|--------|
| **Dynamic mode stiffness** | High | Semi-implicit update unstable for surcharged pipes with small Δt. |
| **Outfall zero-storage mass conservation** | Medium | Instantaneous sink → silent mass loss if receiving cell has insufficient capacity. |
| **Pipe-end loss coefficient defaults** | Medium | Defaults wrong for non-square-edged inlets. No warning. |
| **Node surface area default** | Low | Defaults to 50 m² — underestimates storage for large junction boxes. |

### 13.5 Spatial Scheme Issues

| Issue | Severity | Detail |
|-------|:--------:|--------|
| **Scheme 5 (WENO3-like) experimental** | Medium | 2-candidate blend, not true 3-stencil WENO3. |
| **Non-orthogonal accuracy degradation** | Medium | Green-Gauss gradient drops to O(h) on skewed meshes. |
| **Dry/wet threshold sensitivity** | Medium | Fixed depth threshold → oscillation between active/inactive. |
| **CPU reconstruction uses midpoint step** | Low | Explicitly removed from GPU for causing unstable behavior. |
| **No 2-ring stencil** | Low | 2nd-order spatial ceiling. True 3rd+ requires WENO5/LSQ + 2-ring. |

### 13.6 Tiny-N Path Issues

| Issue | Severity | Detail |
|-------|:--------:|--------|
| **Persistent kernel first-order only** | High | Only `swe2d_persistent_chunk_kernel_first_order` exists. Higher-order = fallback. |
| **Persistent kernel no RK integration** | High | Single-stage only. RK paths use chunked standard steps. |
| **Active edge compaction requires host sync** | Medium | Reads `d_n_active_edges` back to host, breaking all-device path. |
| **Fused mode not graph-capturable** | Low | Incompatible with graph capture of standard kernel sequence. |

### 13.7 Unit System Issues

| Issue | Severity | Detail |
|-------|:--------:|--------|
| **C++ culvert output conversion** | High | Computes in CFS, converts to model units via `model_to_ft`. Both directions must be correct. |
| **Python structure module always returns CMS** | Medium | `structures.py` returns CMS. Coupling controller must convert to model units. |
| **Diagnostics stored in model units** | Low | Runtime reporter assumes model units with `length_unit_name`. |

---

## 14. Key File References

| File | Role |
|------|------|
| `cpp/src/swe2d_gpu.cu` | All CUDA kernels and step functions |
| `cpp/src/swe2d_gpu.cuh` | `SWE2DDeviceState`, `KernelGraphCache`, function declarations |
| `cpp/src/swe2d_solver.cpp` | Main `swe2d_step()` dispatch |
| `cpp/src/swe2d_solver.hpp` | `SWE2DSolverConfig` with all runtime parameters |
| `cpp/src/swe2d_bindings.cpp` | Pybind11 exports (~40 GPU functions) |
| `swe2d/runtime/coupling.py` | `SWE2DCouplingController` — Python coupling orchestration |
| `swe2d/extensions/structures.py` | `SWE2DStructureModule` — Python structure computation |
| `swe2d/extensions/drainage_network.py` | `SWE2DUrbanDrainageModule` — Python drainage |
| `swe2d/extensions/extension_models.py` | `GodunovSolverMode`, config model classes |
| `swe2d/runtime/backend.py` | Native module loading |
| `docs/SOLVER_ORDER_AND_STENCIL.md` | Spatial/temporal order architecture |
| `docs/SWE2D_GPU_ARCHITECTURE_REPORT.md` | Comprehensive GPU architecture report |

---

## 15. Summary Decision Matrix: Path × Capabilities

| Path | Spatial Schemes | Temporal Order | Graph-Capturable | Structure Coupling | Drainage Coupling | Tiny-N | Godunov |
|------|:--------------:|:--------------:|:----------------:|:------------------:|:-----------------:|:------:|:-------:|
| `swe2d_gpu_step` | 0–5 | 1 (Euler) | ✅ | Via `d_ext_src_mps` | Via `d_ext_src_mps` | Fused | Wrapper |
| `swe2d_gpu_step_rk2` | 0–5 | 2 (SSPRK2) | ✅ per stage | Via `d_ext_src_mps` | Via `d_ext_src_mps` | Chunked | Wrapper |
| `swe2d_gpu_step_rk4` | 0–5 | 4 | ❌ | Via `d_ext_src_mps` | Via `d_ext_src_mps` | ❌ | ❌ |
| `swe2d_gpu_step_rk4_graph` | 0–5 | 5 | ✅ single | **Per-stage** in-graph | Via `d_ext_src_mps` | Chunked | ❌ |
| `swe2d_gpu_step_rk5_graph` | 0–5 | ≥6 | ✅ single | **Per-stage** in-graph | Via `d_ext_src_mps` | ❌ | ❌ |
| `persistent_chunk` | **0 only** | 1 (chunked) | ❌ | Via `d_ext_src_mps` | Via `d_ext_src_mps` | ✅ | ❌ |
| `godunov_rollout` | **≥2 enforced** | 1 or 2 | ✅ delegated | Via `d_ext_src_mps` | Via `d_ext_src_mps` | Chunked | N/A |
| `rk2_godunov_rollout` | **≥2 enforced** | 2 | ✅ delegated | Via `d_ext_src_mps` | Via `d_ext_src_mps` | Chunked+ | N/A |

---

*End of GPU Step Graph Architecture document.*
