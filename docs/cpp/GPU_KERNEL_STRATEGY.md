# GPU Kernel Strategy

## Design Philosophy

The GPU kernel path (`swe2d_gpu.cu` + `swe2d_gpu.cuh`) is the **primary execution path**. The CPU solver (`swe2d_solver.cpp`) is retained only for debugging and validation — it is not performance-optimised.

Key design choices:

1. **SoA layout everywhere** — All state arrays are Structure-of-Arrays (`double*` per field) for coalesced global memory access.

2. **Edge-centric flux computation** — Each thread processes one edge. Flux is computed from left/right reconstructed states, atomically accumulated to cell updates via `atomicAdd`.

3. **Graph caching** — `KernelGraphCache` captures the full kernel launch sequence into a CUDA graph for small-timestep scenarios. Replay avoids kernel launch overhead (~10-20µs per kernel → 50-100µs saved per step).

4. **Unified memory only for host-device transfer** — Host arrays use `std::vector`, device arrays use `cudaMalloc`. No managed memory in the hot path.

5. **Tiny-mode dispatch** — For meshes with ≤ 200 cells, fused kernel launches (mode 2) or persistent thread blocks (mode 3) reduce launch overhead.

## Kernel Launch Hierarchy

Each RK stage follows this sequence:

```
swe2d_gpu_step()
  ├── Boundary condition application
  │   └── swe2d_gpu_apply_bc_kernel()        — Per-edge: set ghost state
  ├── Gradient computation (MUSCL/WENO5 only)
  │   ├── swe2d_gpu_gradient_kernel()         — Per-cell: Green-Gauss or LSQ gradient
  │   └── swe2d_gpu_gradient_2ring_kernel()   — Per-cell: 2-ring LSQ (scheme 6)
  ├── Flux computation
  │   └── swe2d_gpu_flux_kernel()             — Per-edge: HLLC Riemann + reconstruction
  ├── Source terms
  │   ├── swe2d_gpu_bed_slope_kernel()        — Per-edge: hydrostatic bed slope
  │   └── swe2d_gpu_friction_kernel()          — Per-cell: semi-implicit friction
  ├── Update
  │   └── swe2d_gpu_update_kernel()            — Per-cell: conservative update
  └── Diagnostics (every N steps)
      └── swe2d_gpu_diag_kernel()              — Per-cell: min/max h, Courant, wet count
```

## Key GPU Kernels

### `swe2d_gpu_flux_kernel`

The core kernel. One thread per edge:

1. Load left/right cell states (h, hu, hv) and centroids
2. Reconstruct left/right interface states (first-order or MUSCL-limited gradient)
3. Apply well-balanced hydrostatic reconstruction (`swe2d_numerics.hpp` — `ReconstructedStates`)
4. Compute HLLC flux with two-wave-speed estimate
5. `atomicAdd` flux contribution to `d_h[d_c0]`, `d_hu[d_c0]`, `d_hv[d_c0]` (and similarly for `d_c1`)

### `swe2d_gpu_apply_bc_kernel`

One thread per boundary edge. Sets ghost-cell state based on BC type:

- **WALL**: Reflect normal velocity, copy tangential + depth
- **INFLOW_Q**: Set ghost depth = interior depth, set ghost velocity to match prescribed unit discharge
- **STAGE**: Set ghost WSE = prescribed stage, zero normal velocity gradient
- **OPEN**: Extrapolate all variables from interior (zero-gradient)
- **NORMAL_DEPTH**: Set ghost depth to prescribed value, copy velocity

### `swe2d_gpu_gradient_kernel`

One thread per cell (for schemes 1-4) or one thread per 2-ring neighbour (scheme 6):

- MUSCL schemes 1-4: Green-Gauss gradient via divergence theorem over cell edges
- WENO5 (scheme 6): Weighted least-squares gradient over 2-ring stencil, with WENO5 nonlinear weights for shock-sensitive reconstruction

### `swe2d_gpu_friction_kernel`

One thread per cell. Semi-implicit Manning friction:

```
u_new = u / (1 + dt * g * n² * |u| / h^(4/3))
```

This sub-stepping approach avoids the stability restriction from explicit friction treatment.

## Edge Reordering for GPU Coalescing

`swe2d_reorder_edges_for_gpu()` in `swe2d_mesh.cpp` reorders edge arrays so that edges sharing the same `c0` cell become contiguous. This ensures that adjacent warp threads in the flux kernel read from the same cell's state for `c0`, improving L1/L2 cache hit rates.

## CUDA Graph Cache

The `KernelGraphCache` struct in `swe2d_gpu.cuh` stores a captured graph + executable instance. Regeneration triggers when:

- Mesh size changes (different `n_cells` or `n_edges`)
- Spatial scheme changes
- Temporal scheme (RK order) changes
- Hydrograph or gradient presence changes

On graph replay, the runtime updates scalar parameters (boundary values, timestep, hydrograph times) via `cudaGraphExecKernelNodeSetParams` without recapturing.

## Tiny-Mode Dispatch

Three modes for small meshes (enabled via `swe2d_solver_set_tiny_mode`):

| Mode | Description | Best For |
|---|---|---|
| 0 | Standard per-kernel launch | > 200 cells |
| 1 | Fused kernels (single launch, all phases) | ≤ 200 cells, single-stage |
| 2 | Persistent thread blocks (kernel stays resident) | ≤ 200 cells, multi-stage RK |

## Device Memory Lifecycle

`SWE2DDeviceState` (`swe2d_gpu.cuh:187`) owns all device pointers. Allocation happens once during `swe2d_create_solver` after mesh upload. Deallocation happens in `swe2d_destroy`.

Mesh topology arrays are uploaded once and are read-only after init. State arrays (h, hu, hv) are updated every step. Gradient arrays and RK stage arrays are allocated on demand.

## Occupancy and Block Sizing

All kernels use 128-256 threads per block, chosen to maximise occupancy across sm_70–sm_90 architectures. The grid size is `ceil(n / block_size)` where `n` is the number of edges, cells, or boundary edges depending on the kernel.

---

## Related Documentation

- **[Documentation Index](../INDEX.md)** — All guides by audience
- **[Architecture](ARCHITECTURE.md)** — `SWE2DDeviceState`, build system
- **[Coupling Kernels](COUPLING_KERNELS.md)** — GPU coupling source kernels
- **[Solver Order & Stencil](../SOLVER_ORDER_AND_STENCIL.md)** — Why 2nd-order is the ceiling
- **[GPU Architecture Report](../SWE2D_GPU_ARCHITECTURE_REPORT.md)** — `KernelGraphCache`, `SWE2DDeviceState`
