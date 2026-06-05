# GPU Path Audit Report

> **Date:** 2026-06-04
> **GPU:** NVIDIA GeForce RTX 3080 (Ampere, sm_86, 10 GB VRAM)
> **CUDA:** 13.2, **Driver:** 595.71.05
> **Nsight Compute:** 2022.4.1 (--set full)
> **Mesh:** 80×40 rectangular grid (6,400 cells, ~19,200 edges)
> **Solver:** RK2 time integration, first-order spatial (FV_FIRST_ORDER)
> **Steps profiled:** 50

---

## Table of Contents

1. [Host-to-Device Transfers](#1-host-to-device-transfers)
2. [Coalesced Global Memory Access](#2-coalesced-global-memory-access)
3. [Shared Memory Utilization](#3-shared-memory-utilization)
4. [Occupancy and Register Usage](#4-occupancy-and-register-usage)
5. [Nsight Compute Profiling Results](#5-nsight-compute-profiling-results)
6. [Recommendations](#6-recommendations)

---

## 1. Host-to-Device Transfers

### 1.1 Initialization (One-Time)

| Transfer | Direction | Size | Frequency | Notes |
|----------|-----------|------|-----------|-------|
| Edge topology (c0, c1, n0, n1) | H→D | ~9 arrays × 19,200 elems | Once | Bulk, synchronous |
| Cell geometry (zb, area, inv_area) | H→D | ~4 arrays × 6,400 elems | Once | Bulk, synchronous |
| Cell centroids (cx, cy) | H→D | 2 arrays × 6,400 elems | Once | Bulk, synchronous |
| Manning's n | H→D | 1 array × 6,400 elems | Once | Bulk, synchronous |
| Initial state (h, hu, hv) | H→D | 3 arrays × 6,400 elems | Once | Bulk, synchronous |
| Cell-edge CSR (offsets, ids) | H→D | 2 arrays | Once | Bulk, synchronous |
| BC forced markings | H→D | 1 array × 6,400 elems | Once | Bulk, synchronous |
| Degenerate cell data | H→D | up to 3 arrays | Once | Conditional on degen_mode |
| **Total initial transfer** | H→D | **~4–5 MB** | **Once** | Acceptable |

**Assessment: ✅ Good.** Mesh topology and geometry are transferred once at init and remain resident on the GPU for the solver lifetime. No per-step re-upload of static data.

### 1.2 Per-Step Transfers (Hot Path)

| Transfer | Direction | Size | Frequency | Mechanism |
|----------|-----------|------|-----------|-----------|
| RK2 stage backup (h,hu,hv → h0,h1) | D→D | 3 × 6,400 × 8B = 154 KB | Each RK2 step | `cudaMemcpyAsync` on stream |
| Active-set hysteresis (d_active → d_was_active) | D→D | 6,400 × 4B = 26 KB | Each step | `cudaMemcpyAsync` on stream |
| Flux clearing | D→D | 5 × 19,200 × 8B = 768 KB | Each step | `cudaMemsetAsync` on stream |
| Gradient clearing (6 arrays) | D→D | 6 × 6,400 × 8B = 307 KB | When gradients needed | `cudaMemsetAsync` on stream |
| Diagnostics readback (packed 3 doubles) | D→H | **24 bytes** | Per step (optionally sync'd) | Single `cudaMemcpy` |
| Rain/CN source build | D→D | Varies | If rain enabled | On-device kernel |

**Assessment: ✅ Excellent.** The design keeps nearly all data on-device across steps.

- **Packed diagnostics** (`pack_diag_kernel`) consolidates λ_max, WSE error, and wet-count into a single 24-byte D→H transfer — a textbook optimization.
- **RK2 backup** is D→D (device-to-device), no PCIe traffic.
- **Flux clearing** uses `cudaMemsetAsync` — zero PCIe involvement.
- The `KernelGraphCache` can capture the entire step as a CUDA graph, eliminating kernel launch overhead entirely.

### 1.3 Coupling Workspace (Lazy Persistent Allocation)

The `CouplingWorkspace` and `StructureFlowWorkspace` structs in `SWE2DDeviceState` use persistent device buffers with **content hashing** (`inlet_data_hash`, `structure_data_hash`, `bridge_data_hash`). Data is re-uploaded only when the hash changes.

```cpp
// Content hashes for dirtiness tracking (skip re-upload if unchanged).
uint64_t inlet_data_hash = 0;
uint64_t structure_data_hash = 0;
uint64_t bridge_data_hash = 0;
```

**Assessment: ✅ Excellent.** Eliminates per-call `cudaMalloc`/`cudaFree` and redundant re-upload.

### 1.4 Areas for Improvement

| Issue | Location | Impact | Recommendation |
|-------|----------|--------|---------------|
| `swe2d_gpu_redistribute.cu` does per-call `cudaMalloc+Free+Memcpy` | Redistribution wrapper | PCIe traffic + allocation latency on every coupling step | Migrate to persistent buffers with content hashing (like `CouplingWorkspace`) |
| Debug-mode D→H transfers | `BACKWATER_SWE2D_DEBUG_GPU_INPUT/EDGE_FLUX/FLUX` | Up to 600 KB per debug read | Debug-only, acceptable |
| `swe2d_gpu_compute_dt_3d_patch` does a D→H sync per call | DT computation | ~8 bytes + sync overhead | Could use packed kernel, but frequency is low |

---

## 2. Coalesced Global Memory Access

### 2.1 Analysis by Kernel

#### `swe2d_flux_kernel` (1 thread per edge)

**Edge arrays (coalesced ✅):**
```
edge_c0[e], edge_c1[e], edge_nx[e], edge_ny[e], edge_len[e],
edge_mx[e], edge_my[e], edge_bc[e], edge_bc_val[e]
```
Each thread reads index `e = blockIdx.x * blockDim.x + threadIdx.x`, producing perfectly coalesced 128-byte cache-line accesses.

**Cell arrays (uncoalesced ⚠️):**
```
cell_h[c0], cell_hu[c0], cell_hv[c0], cell_zb[c0]   // c0 = edge_c0[e]
cell_h[c1], cell_hu[c1], cell_hv[c1], cell_zb[c1]   // c1 = edge_c1[e]
```
On an unstructured mesh, `c0` and `c1` are **not consecutive** for adjacent threads. This produces scattered (non-coalesced) reads.

**Mitigation:** Uses `__ldg()` (built-in read-only cache) for cell state reads:
```cpp
double hL  = __ldg(&cell_h[c0]);
```
This bypasses L1 and routes through the read-only (texture) cache, which handles scattered access patterns better.

**TVD reconstruction reads (further scattering):**
```cpp
grad_hx[c0], grad_hy[c0], ...  // cell-gradient reads
cell_cx[c0], cell_cy[c0]       // centroid reads
```
These are also scattered. With full higher-order reconstruction (MC/Van Leer/WENO3), each edge thread reads **~20 cell-centered quantities** through `c0`/`c1` indirection.

#### `swe2d_update_kernel` (1 thread per cell)

**Coalesced ✅:** Cell state arrays are read/written at index `c = blockIdx.x * blockDim.x + threadIdx.x`:
```cpp
cell_h[c], cell_hu[c], cell_hv[c]  // sequential per thread
```

**Scattered ⚠️:** Flux accumulation loops over the cell-edge CSR:
```cpp
for (int32_t k = s; k < e; ++k) {
    const int32_t edge = cell_edge_ids[k];
    fh += flux_h[edge];   // edge index is NOT consecutive
}
```
`flux_h[edge]` reads are scattered because each cell's incident edges are stored sparsely.

#### `swe2d_gradient_kernel` (1 thread per edge)

**Scattered atomics ⚠️:** Each edge atomically adds to `grad_hx[c0]` and `grad_hx[c1]` — fully scattered. The CAS-based `atomicAddDouble` adds further cost. However, this is fundamentally required for the Green-Gauss gradient on unstructured meshes.

#### `swe2d_cfl_kernel` (1 thread per edge)

**Same pattern as flux kernel:** Coalesced edge reads, scattered cell reads. Uses `cell_area` as well.

### 2.2 Coalescing Summary

| Kernel | Edge Reads | Cell Reads | Atomics | Overall Coalescing |
|--------|-----------|-----------|---------|--------------------|
| `swe2d_classify_and_mark_kernel` | N/A (cell-parallel) | ✅ Sequential | ✅ None | ✅ Good |
| `swe2d_gradient_kernel` | ✅ Coalesced | ⚠️ Scattered | ⚠️ CAS atomics | ⚠️ Fair |
| `swe2d_flux_kernel` | ✅ Coalesced | ⚠️ Scattered (__ldg) | None | ⚠️ Fair (mitigated) |
| `swe2d_update_kernel` | ⚠️ CSR-scattered | ✅ Sequential | None | ⚠️ Fair |
| `swe2d_cfl_kernel` | ✅ Coalesced | ⚠️ Scattered | None | ⚠️ Fair |
| `swe2d_rk2_combine_kernel` | N/A | ✅ Sequential | None | ✅ Good |

### 2.3 NCU Memory Throughput Confirmation

The profiling confirms sub-optimal memory throughput:

| Kernel | DRAM Throughput | L1/TEX Throughput | Memory Throughput |
|--------|----------------|-------------------|-------------------|
| `swe2d_flux_kernel` | **2.7%** | 1.8% | 2.7% |
| `swe2d_update_kernel` | **3.6%** | 3.7% | 3.6% |
| `swe2d_classify_and_mark_kernel` | **6.7%** | 10.3% | 6.7% |
| `swe2d_cfl_kernel` | **5.4%** | 3.6% | 5.4% |

The very low DRAM throughput (2–7% of peak) is expected for unstructured-mesh finite-volume codes — the scattered access pattern cannot achieve the theoretical 760 GB/s bandwidth of the RTX 3080. The `__ldg()` cache helps but does not eliminate the fundamental issue.

---

## 3. Shared Memory Utilization

### 3.1 Current Usage

| Kernel | Dynamic Shared Memory | Static Shared Memory | Purpose |
|--------|---------------------|---------------------|---------|
| `swe2d_classify_and_mark_kernel` | 1.0 KB/block (`BLOCK=256`) | 0 | Block-reduction for wet-cell count |
| `swe2d_cfl_kernel` | 2.05 KB/block | 0 | Block-reduction for max lambda |
| `swe2d_cfl_reduce_blocks_kernel` | 2.05 KB/block | 0 | Second-level reduction |
| `swe2d_flux_kernel` | **0** | 0 | **None** |
| `swe2d_update_kernel` | **0** | 0 | **None** |
| `swe2d_rk2_combine_kernel` | **0** | 0 | **None** |
| `swe2d_gradient_kernel` | **0** | 0 | **None** |

**Assessment: ⚠️ Underutilized.** Shared memory is only used for block-wide reductions (classify's wet count and CFL's max lambda). The three hottest kernels (`flux`, `update`, `gradient`) do **not** use shared memory at all.

### 3.2 Missed Opportunities

#### `swe2d_flux_kernel` — No shared memory tiling

Each edge reads 4 cell states (h, hu, hv, zb) from both endpoints c0 and c1. On the RTX 3080 (48 KB shared mem/SM, 256 threads/block), a tiled approach could:

1. **Load cell data into shared memory** in a pre-pass where blocks cooperatively load tile-sized chunks.
2. **Reuse cell data** across edges sharing the same cell.

However, implementing this is challenging on unstructured meshes because the edge→cell adjacency is irregular. An AoS→SoA layout conversion could help, but the scattered graph traversal makes tile-based reuse difficult without explicit coloring.

#### What would help most:

- **SoA (Structure of Arrays) layout for cell state**: Currently `d_h[c]`, `d_hu[c]`, `d_hv[c]` are separate arrays. If edge threads load all three for c0, the three reads hit three different cache lines. Packing into `float4` or interleaved structures could improve cache-line utilization.
- **Edge coloring / graph coloring**: Partition edges into sets where no two edges in the same warp share a cell, allowing safe shared-memory accumulation of fluxes. This would replace CSR-based flux accumulation (which uses per-edge storage) with shared-memory atomics or warp-shuffle reductions.

---

## 4. Occupancy and Register Usage

### 4.1 Nsight Compute Occupancy Results

| Kernel | Regs/Thread | Theor. Occupancy | Achieved Occupancy | Achieved Warps/SM | Primary Limiter |
|--------|-----------|-----------------|--------------------|--------------------|-----------------|
| `swe2d_flux_kernel` | **90** | 33.3% | **16.5%** | 7.9 | Registers (2 blocks/SM) |
| `swe2d_update_kernel` | **64** | 66.7% | **16.4%** | 7.9 | Registers (4 blocks/SM) |
| `swe2d_cfl_kernel` | **52** | 66.7% | **16.6%** | 8.0 | Registers (4 blocks/SM) |
| `swe2d_rk2_combine_kernel` | 24 | 100% | **14.9%** | 7.1 | Other (latency) |
| `swe2d_classify_and_mark_kernel` | 20 | 100% | **15.4%** | 7.4 | Other (latency) |
| `swe2d_cfl_reduce_blocks_kernel` | 16 | 100% | **13.5%** | 6.5 | Other (lightweight kernel) |

### 4.2 Register Pressure Analysis

**Critical findings:**

1. **`swe2d_flux_kernel` uses 90 registers/thread.** This is the most severe issue. With 90 registers, only **2 blocks** can run per SM on the RTX 3080 (limit: 65536 registers/SM ÷ 90 regs/thr ÷ 256 thr/blk = 2.8 → 2 blocks). This caps occupancy at 2 × 8 warps / 48 max = 33%.

2. **`swe2d_update_kernel` uses 64 registers/thread.** With 64 registers, 4 blocks can run per SM (65536 ÷ 64 ÷ 256 = 4), giving 32 warps/SM theoretical (67% occupancy).

3. **`swe2d_cfl_kernel` uses 52 registers/thread.** Also limited to 4 blocks/SM.

4. **Achieved occupancy is ~15-17% across all kernels**, far below theoretical. This suggests the kernels are **latency-bound** rather than occupancy-bound — they stall on memory waits and the scheduler cannot hide latency even with more warps.

The high register count in `swe2d_flux_kernel` comes from:
- TVD limiter lambda functions (Superbee/MinMod/MC/Van Leer)
- WENO3-like reconstruction inline code
- `ReconstructedStatesLocal` struct (6 doubles = 48 bytes of stack)
- `GhostStateLocal` struct (4 doubles)
- All the local variables for HLLC flux computation
- Momentum capping, bed slope correction, and front damping
- `__ldg()` intrinsic calls and math functions

### 4.3 Build Configuration

Current CMakeLists.txt sets:
```cmake
target_compile_options(${target_name} PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)
```

**Missing:** `--maxrregcount` / `-maxrregcount` flag. The compiler is free to use as many registers as needed.

---

## 5. Nsight Compute Profiling Results

### 5.1 Kernel Execution Profile

| Kernel | Duration (µs) | Grid Size | Block Size | % of Step Time |
|--------|--------------|-----------|------------|---------------|
| `swe2d_update_kernel` | 39.5 | 25 × 1 × 1 | 256 × 1 × 1 | **33.1%** |
| `swe2d_flux_kernel` | 31.3 | 75 × 1 × 1 | 256 × 1 × 1 | **26.2%** |
| `swe2d_cfl_kernel` | 13.1 | 38 × 1 × 1 | 256 × 1 × 1 | 11.0% |
| `swe2d_rk2_combine_kernel` | 11.6 | 25 × 1 × 1 | 256 × 1 × 1 | 9.7% |
| `swe2d_classify_and_mark_kernel` | 6.0 | 25 × 1 × 1 | 256 × 1 × 1 | 5.0% |
| `swe2d_cfl_reduce_blocks_kernel` | 4.2 | 1 × 1 × 1 | 256 × 1 × 1 | 3.5% |

**Total kernel time per step: ~106 µs** (first-order RK2, non-graph, 6,400 cells)

> With CUDA Graph optimization (`enable_kernel_graphs=true`), launch overhead is eliminated — 7 kernel launches → 1 graph launch. The graph capture path was observed in ncu output (the classify kernel signature matches the graph path).

### 5.2 Compute vs Memory Throughput

| Kernel | Compute (SM) % | Memory % | SM Active % | Characterization |
|--------|---------------|---------|------------|-----------------|
| `swe2d_flux_kernel` | **42.1%** | 2.7% | 51.9% | **Compute-heavy**, memory-light |
| `swe2d_update_kernel` | **22.4%** | 3.6% | 33.8% | Compute-medium, memory-light |
| `swe2d_cfl_kernel` | **33.8%** | 5.4% | 49.0% | Compute-heavy |
| `swe2d_classify_and_mark_kernel` | 2.8% | 6.7% | 26.7% | Latency-bound |
| `swe2d_rk2_combine_kernel` | 4.3% | 4.4% | 32.3% | Latency-bound |
| `swe2d_cfl_reduce_blocks_kernel` | 0.1% | 0.4% | 1.0% | Serial bottleneck |

### 5.3 Key Observations

1. **`swe2d_flux_kernel` is compute-bound at 42% SM throughput** — unusual for a finite-volume code, which is typically memory-bound. This is because the HLLC Riemann solver, TVD reconstruction, and momentum capping are compute-intensive, while the mesh is small enough to fit in cache.

2. **Extremely low DRAM throughput (2–7%)** is partly due to the small mesh (6,400 cells). Caching is effective, so most reads hit L1/L2.

3. **SM Active Cycles** range from 27–52%, meaning **48–73% of cycles are stalled** (waiting for memory, synchronization, or pipeline stalls).

4. **The `cfl_reduce_blocks_kernel` is a serial bottleneck** — only 1% SM active because it's a single-block reduction for 38 block-max values. This is a natural consequence of the two-level reduction design.

---

## 6. Recommendations

### Priority: High

#### H1. Add `__launch_bounds__` annotations to all kernels

The ncu profiling shows critical register pressure. Add `__launch_bounds__` to constrain register usage:

```cpp
// swe2d_flux_kernel — most critical: 90 regs → target 48 (doubles occupancy)
__global__ __launch_bounds__(256, 4)  // maxThreadsPerBlock=256, minBlocksPerMultiprocessor=4
void swe2d_flux_kernel(...)

// swe2d_update_kernel: 64 regs → target 48
__global__ __launch_bounds__(256, 4)
void swe2d_update_kernel(...)

// swe2d_cfl_kernel: 52 regs → target 48
__global__ __launch_bounds__(256, 4)
void swe2d_cfl_kernel(...)
```

This tells the compiler to limit register usage so at least 4 blocks can run per SM. The compiler will spill excess registers to local memory, but the increased occupancy should hide latency better.

#### H2. Add `--maxrregcount=48` (or similar) to CMakeLists.txt

```cmake
target_compile_options(${target_name} PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-maxrregcount=48>
)
```

Apply universally or per-kernel with `__launch_bounds__`.

#### H3. Refactor `swe2d_gpu_redistribute.cu` to use persistent buffers

Replace per-call `cudaMalloc`/`cudaMemcpy`/`cudaFree` with persistent workspace buffers in `SWE2DDeviceState`, tracking dirtiness via content hashes (same pattern as `CouplingWorkspace`).

### Priority: Medium

#### M1. Investigate `swe2d_flux_kernel` compute-bound behaviour

At 42% SM throughput with only 2.7% memory throughput, this kernel is unusually compute-bound for a finite-volume solver. Profile the breakdown:
- HLLC Riemann solver vs TVD reconstruction vs WENO3 vs momentum limiting
- Consider splitting into simpler sub-kernels for the first-order path (which doesn't need TVD gradients)

#### M2. Reduce TVD limiter overhead

The inline lambda functions (`tvd_reconstruct`, `weno3_like_reconstruct`) are defined per-edge inside `swe2d_flux_kernel`. These add significant register pressure. Options:
- Move them to `__device__` functions outside the kernel (reduces local variable count)
- Use `__noinline__` on the most complex paths to reduce register pressure

#### M3. Pack cell state for better cache utilization

Group the three conserved variables (h, hu, hv) plus zb into a struct-of-arrays layout or use `double4`/`double2` loads:

```cpp
struct CellState { double h, hu, hv, zb; };
CellState* d_cell;  // single pointer, adjacent threads read adjacent structs
```

This would improve cache-line utilization from 3+1 separate cache lines to 1 when accessing all four quantities for a cell.

### Priority: Low

#### L1. Reduce `swe2d_cfl_reduce_blocks_kernel` overhead

The second-level reduction kernel has 0.1% SM throughput — it's a tiny single-block kernel. For small meshes this is fine, but for large meshes, consider:
- Using warp-level reduction primitives
- Folding the final reduction into the main CFL kernel (write directly to `d_lambda_max` with atomicMax for the block result)

#### L2. Explore edge coloring for shared-memory flux accumulation

For very large meshes, a tiled approach with edge coloring could replace the CSR-based flux storage (which is currently 5 edge-length arrays = ~768 KB for 19,200 edges) with shared-memory resident accumulators. This would also eliminate the scattered reads of `flux_h[edge]` in the update kernel.

---

## Appendix A: Full NCU Metric Table

| Kernel | Duration (µs) | Regs | Dynam. Shmem | Theor. Occ | Achiev. Occ | AW/SM | SM Thr. | Mem Thr. | L1 Thr. | L2 Thr. | DRAM Thr. | SM Active |
|--------|:----------:|:----:|:----------:|:---------:|:----------:|:-----:|:------:|:-------:|:------:|:------:|:--------:|:--------:|
| `classify_and_mark_kernel` | 6.0 | 20 | 1.0 KB | 100.0% | 15.4% | 7.4 | 2.8% | 6.7% | 10.3% | 3.7% | 6.7% | 26.7% |
| `swe2d_flux_kernel` | 31.3 | **90** | 0 | 33.3% | 16.5% | 7.9 | **42.1%** | 2.7% | 1.8% | 2.6% | 2.7% | 51.9% |
| `swe2d_update_kernel` | 39.5 | **64** | 0 | 66.7% | 16.4% | 7.9 | **22.4%** | 3.6% | 3.7% | 3.0% | 3.6% | 33.8% |
| `swe2d_cfl_kernel` | 13.1 | 52 | 2.0 KB | 66.7% | 16.6% | 8.0 | **33.8%** | 5.4% | 3.6% | 3.5% | 5.4% | 49.0% |
| `swe2d_cfl_reduce_blocks_kernel` | 4.2 | 16 | 2.0 KB | 100.0% | 13.5% | 6.5 | 0.1% | 0.4% | 16.4% | 0.4% | 0.0% | 1.0% |
| `swe2d_rk2_combine_kernel` | 11.6 | 24 | 0 | 100.0% | 14.9% | 7.1 | 4.3% | 4.4% | 7.4% | 4.3% | 3.9% | 32.3% |

> **Legend:** AW/SM = Achieved Active Warps Per SM; SM Thr. = Compute (SM) Throughput; Mem Thr. = Memory Throughput; L1/L2 Thr. = L1/L2 Cache Throughput; DRAM Thr. = DRAM Throughput.

## Appendix B: GPU Specifications (RTX 3080)

| Property | Value |
|----------|-------|
| Compute Capability | 8.6 |
| SMs | 68 (GA102 cut to 68 active) |
| Cores/SM | 128 (64 FP32 + 64 INT32) |
| Max Warps/SM | 48 |
| Max Blocks/SM | 16 |
| Registers/SM | 65,536 |
| Shared Memory/SM | 48 KB (configurable up to 100 KB with extended mode) |
| L1 Cache | 128 KB per SM (unified with shared mem) |
| L2 Cache | 5 MB |
| VRAM | 10 GB GDDR6X |
| Memory Bandwidth | 760 GB/s |
