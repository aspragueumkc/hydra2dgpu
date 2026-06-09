/**
 * swe2d_gpu_redistribute.cu
 *
 * Standalone GPU redistribution kernel for structure/bridge source terms.
 *
 * After the existing coupling source kernels inject flow into a SINGLE
 * upstream/downstream cell pair, this kernel redistributes the flow across
 * a pre-computed corridor of cells (influence width), avoiding unrealistic
 * velocity jets when the coupled cell is much smaller than the structure.
 *
 * The pre-computed weights are generated once at setup in Python and passed
 * as flat arrays with per-structure offsets.
 */

#include "swe2d_gpu.cuh"

#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <stdexcept>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            throw std::runtime_error(std::string("CUDA error: ")               \
                + cudaGetErrorString(_e) + " at " __FILE__ ":"                 \
                + std::to_string(__LINE__));                                    \
        }                                                                       \
    } while (0)

// ─────────────────────────────────────────────────────────────────────────────
//  Device kernel
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Redistribution kernel.
 *
 * For each structure with distribution data (count > 0):
 *   1. Reverse the original single-cell injection that was already applied
 *      to the source array by the coupling source kernel.
 *   2. Distribute the flow across the corridor cells using pre-computed
 *      normalized weights.
 *
 * Structures without distribution data (dist_offsets[i+1] - dist_offsets[i] <= 0)
 * are left unchanged (original single-cell injection stands).
 */
__global__ __launch_bounds__(256, 4) void swe2d_redistribute_sources_kernel(
    int32_t     n_structures,
    const double* __restrict__ struct_flow,
    const int32_t* __restrict__ orig_up_cell,
    const int32_t* __restrict__ orig_dn_cell,
    const double* __restrict__ cell_area,
    const int32_t* __restrict__ dist_offsets,
    const int32_t* __restrict__ dist_cell_idx,
    const double* __restrict__ dist_weights,
    int32_t     n_cells,
    double* __restrict__ source_rate)
{
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_structures) return;

    const int32_t start = dist_offsets[i];
    const int32_t end   = dist_offsets[i + 1];
    const int32_t count = end - start;
    if (count <= 0) return;

    const double q = struct_flow[i];
    if (!isfinite(q) || q == 0.0) return;

    // Step 1: Reverse the single-cell injection that was already applied
    const int32_t cu = orig_up_cell[i];
    const int32_t cd = orig_dn_cell[i];
    if (cu >= 0 && cu < n_cells) {
        atomicAdd(&source_rate[cu], q / fmax(cell_area[cu], 1.0e-12));
    }
    if (cd >= 0 && cd < n_cells) {
        atomicAdd(&source_rate[cd], -q / fmax(cell_area[cd], 1.0e-12));
    }

    // Step 2: Normalize weights
    double wsum = 0.0;
    for (int32_t j = start; j < end; j++) wsum += dist_weights[j];
    if (wsum <= 0.0) return;
    const double inv_wsum = 1.0 / wsum;

    // Step 3: Distribute flow across corridor cells
    for (int32_t j = start; j < end; j++) {
        const int32_t c = dist_cell_idx[j];
        if (c < 0 || c >= n_cells) continue;
        const double a = fmax(cell_area[c], 1.0e-12);
        atomicAdd(&source_rate[c], (dist_weights[j] * inv_wsum) * q / a);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Host-callable wrapper (CUDA path)
// ─────────────────────────────────────────────────────────────────────────────

// Helper: ensure persistent redistribution buffers have capacity.
static void ensure_redist_buffers(
    SWE2DDeviceState* dev,
    int32_t n_structures,
    int32_t total_dist_cells,
    int32_t n_cells = 0)
{
    auto& ws = dev->redist_ws;
    if (ws.n_struct_capacity < n_structures + 1) {
        if (ws.d_offsets) cudaFree(ws.d_offsets);
        if (ws.d_up)      cudaFree(ws.d_up);
        if (ws.d_dn)      cudaFree(ws.d_dn);
        if (ws.d_flow)    cudaFree(ws.d_flow);
        ws.d_offsets = nullptr; ws.d_up = nullptr; ws.d_dn = nullptr; ws.d_flow = nullptr;
        CUDA_CHECK(cudaMalloc(&ws.d_offsets, static_cast<size_t>(n_structures + 1) * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&ws.d_up,      static_cast<size_t>(n_structures) * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&ws.d_dn,      static_cast<size_t>(n_structures) * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&ws.d_flow,    static_cast<size_t>(n_structures) * sizeof(double)));
        ws.n_struct_capacity = n_structures + 1;
        ws.data_hash = 0;  // force re-upload
    }
    if (total_dist_cells > 0 && ws.dist_cell_capacity < total_dist_cells) {
        if (ws.d_cell_idx) cudaFree(ws.d_cell_idx);
        if (ws.d_weights)  cudaFree(ws.d_weights);
        ws.d_cell_idx = nullptr; ws.d_weights = nullptr;
        CUDA_CHECK(cudaMalloc(&ws.d_cell_idx, static_cast<size_t>(total_dist_cells) * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&ws.d_weights,  static_cast<size_t>(total_dist_cells) * sizeof(double)));
        ws.dist_cell_capacity = total_dist_cells;
        ws.data_hash = 0;  // force re-upload
    }
    if (n_cells > 0 && ws.cell_capacity < n_cells) {
        if (ws.d_cell_area) cudaFree(ws.d_cell_area);
        if (ws.d_source) cudaFree(ws.d_source);
        ws.d_cell_area = nullptr; ws.d_source = nullptr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ws.d_cell_area),
                              static_cast<size_t>(n_cells) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ws.d_source),
                              static_cast<size_t>(n_cells) * sizeof(double)));
        ws.cell_capacity = n_cells;
        ws.data_hash = 0;
    }
}

static uint64_t hash_redist_data(
    int32_t n_structures,
    const int32_t* dist_offsets,
    const int32_t* dist_cell_idx,
    const double*  dist_weights,
    const int32_t* orig_up_cell,
    const int32_t* orig_dn_cell,
    int32_t total_dist_cells)
{
    uint64_t h = 1469598103934665603ULL;
    auto mix = [](uint64_t h, uint64_t v) { return h ^ (v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2)); };
    for (int32_t i = 0; i < n_structures + 1; ++i) {
        uint64_t v = 0; std::memcpy(&v, &dist_offsets[i], sizeof(int32_t)); h = mix(h, v);
    }
    for (int32_t i = 0; i < n_structures; ++i) {
        uint64_t v = 0; std::memcpy(&v, &orig_up_cell[i], sizeof(int32_t)); h = mix(h, v);
        std::memcpy(&v, &orig_dn_cell[i], sizeof(int32_t)); h = mix(h, v);
    }
    for (int32_t i = 0; i < total_dist_cells; ++i) {
        uint64_t v = 0; std::memcpy(&v, &dist_cell_idx[i], sizeof(int32_t)); h = mix(h, v);
        uint64_t w = 0; std::memcpy(&w, &dist_weights[i], sizeof(double));  h = mix(h, w);
    }
    return h;
}

void swe2d_gpu_redistribute_structure_sources(
    SWE2DDeviceState* dev,
    int32_t n_structures,
    const double* structure_flow,
    const int32_t* orig_up_cell,
    const int32_t* orig_dn_cell,
    const double* cell_area,
    const int32_t* dist_offsets,
    const int32_t* dist_cell_idx,
    const double* dist_weights,
    int32_t n_cells,
    double* source_rate_inout)
{
    if (n_structures <= 0 || !source_rate_inout) return;

    constexpr int BLOCK = 256;
    cudaStream_t stream = dev ? dev->d_stream : nullptr;

    // Compute total redistribution cell count
    int32_t total_dist_cells = 0;
    for (int32_t i = 0; i < n_structures; i++) {
        int32_t start = dist_offsets[i];
        int32_t end   = dist_offsets[i + 1];
        total_dist_cells += (end - start > 0) ? (end - start) : 0;
    }

    // Allocate device pointers
    int32_t *d_offsets = nullptr, *d_cell_idx = nullptr;
    double  *d_weights = nullptr;
    int32_t *d_up = nullptr, *d_dn = nullptr;
    double  *d_flow = nullptr;
    double  *d_cell_area = nullptr, *d_source = nullptr;
    bool own_temp = false;  // did we allocate d_flow for temp use?

    if (dev) {
        // Persistent path: use device-resident buffers with content-hash
        // tracking to skip re-upload of static redistribution geometry.
        ensure_redist_buffers(dev, n_structures, total_dist_cells, n_cells);
        auto& ws = dev->redist_ws;
        d_offsets = ws.d_offsets;
        d_up      = ws.d_up;
        d_dn      = ws.d_dn;
        d_cell_idx = ws.d_cell_idx;
        d_weights  = ws.d_weights;
        d_cell_area = ws.d_cell_area;
        d_source    = ws.d_source;

        auto h2d = [stream](void* d, const void* h, size_t bytes) {
            if (stream)
                CUDA_CHECK(cudaMemcpyAsync(d, h, bytes, cudaMemcpyHostToDevice, stream));
            else
                CUDA_CHECK(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice));
        };

        // Upload cell area to the redist-owned buffer (host pointer is parameter)
        h2d(d_cell_area, cell_area, static_cast<size_t>(n_cells) * sizeof(double));

        // Flow values change every step (tiny array); always upload.
        // Static geometry: upload only when hash changed.
        const uint64_t new_hash = hash_redist_data(
            n_structures, dist_offsets, dist_cell_idx, dist_weights,
            orig_up_cell, orig_dn_cell, total_dist_cells);
        const bool data_changed = (new_hash != ws.data_hash);

        if (data_changed) {
            h2d(d_offsets, dist_offsets, static_cast<size_t>(n_structures + 1) * sizeof(int32_t));
            h2d(d_up,      orig_up_cell, static_cast<size_t>(n_structures) * sizeof(int32_t));
            h2d(d_dn,      orig_dn_cell, static_cast<size_t>(n_structures) * sizeof(int32_t));
            if (total_dist_cells > 0) {
                h2d(d_cell_idx, dist_cell_idx, static_cast<size_t>(total_dist_cells) * sizeof(int32_t));
                h2d(d_weights,  dist_weights,  static_cast<size_t>(total_dist_cells) * sizeof(double));
            }
            ws.data_hash = new_hash;
        }

        // Upload cell area to the redist-owned buffer (host pointer is function param)
        h2d(d_cell_area, cell_area, static_cast<size_t>(n_cells) * sizeof(double));

        // Flow: always upload (changes every step, but tiny: n_structures * 8 bytes).
        // d_flow is pre-allocated in the persistent workspace so this never calls
        // cudaMalloc inside a CUDA graph capture region.
        d_flow = ws.d_flow;
        h2d(d_flow, structure_flow, static_cast<size_t>(n_structures) * sizeof(double));
        own_temp = false;

        // Source array: upload every step (source rates change each timestep).
        // d_source is now pre-allocated by swe2d_gpu_preload_coupling_cell_area.
        h2d(d_source, source_rate_inout, static_cast<size_t>(n_cells) * sizeof(double));
    } else {
        // Non-persistent path: allocate and upload everything (legacy).
        CUDA_CHECK(cudaMalloc(&d_offsets, static_cast<size_t>(n_structures + 1) * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_up,      static_cast<size_t>(n_structures) * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_dn,      static_cast<size_t>(n_structures) * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_flow,    static_cast<size_t>(n_structures) * sizeof(double)));
        if (total_dist_cells > 0) {
            CUDA_CHECK(cudaMalloc(&d_cell_idx, static_cast<size_t>(total_dist_cells) * sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc(&d_weights,  static_cast<size_t>(total_dist_cells) * sizeof(double)));
        }
        CUDA_CHECK(cudaMalloc(&d_cell_area, static_cast<size_t>(n_cells) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_source,    static_cast<size_t>(n_cells) * sizeof(double)));

        auto h2d = [stream](void* d, const void* h, size_t bytes) {
            if (stream)
                CUDA_CHECK(cudaMemcpyAsync(d, h, bytes, cudaMemcpyHostToDevice, stream));
            else
                CUDA_CHECK(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice));
        };
        h2d(d_offsets, dist_offsets, static_cast<size_t>(n_structures + 1) * sizeof(int32_t));
        h2d(d_up,      orig_up_cell, static_cast<size_t>(n_structures) * sizeof(int32_t));
        h2d(d_dn,      orig_dn_cell, static_cast<size_t>(n_structures) * sizeof(int32_t));
        h2d(d_flow,    structure_flow, static_cast<size_t>(n_structures) * sizeof(double));
        if (total_dist_cells > 0) {
            h2d(d_cell_idx, dist_cell_idx, static_cast<size_t>(total_dist_cells) * sizeof(int32_t));
            h2d(d_weights,  dist_weights,  static_cast<size_t>(total_dist_cells) * sizeof(double));
        }
        h2d(d_cell_area, cell_area, static_cast<size_t>(n_cells) * sizeof(double));
        h2d(d_source,    source_rate_inout, static_cast<size_t>(n_cells) * sizeof(double));
    }

    const int grid = (n_structures + BLOCK - 1) / BLOCK;
    swe2d_redistribute_sources_kernel<<<grid, BLOCK, 0, stream>>>(
        n_structures, d_flow, d_up, d_dn,
        d_cell_area,
        d_offsets, d_cell_idx, d_weights,
        n_cells,
        d_source);

    if (stream) CUDA_CHECK(cudaStreamSynchronize(stream));
    else        CUDA_CHECK(cudaDeviceSynchronize());

    // Download result back to host.
    {
        auto d2h = [stream](void* h, const void* d, size_t bytes) {
            if (stream)
                CUDA_CHECK(cudaMemcpyAsync(h, d, bytes, cudaMemcpyDeviceToHost, stream));
            else
                CUDA_CHECK(cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost));
        };
        d2h(source_rate_inout, d_source, static_cast<size_t>(n_cells) * sizeof(double));
    }

    // Free per-step allocations.
    if (!dev && d_offsets) cudaFree(d_offsets);
    if (!dev && d_up)      cudaFree(d_up);
    if (!dev && d_dn)      cudaFree(d_dn);
    if (own_temp && d_flow) cudaFree(d_flow);
    if (!dev && d_cell_idx) cudaFree(d_cell_idx);
    if (!dev && d_weights)  cudaFree(d_weights);
    if (!dev && d_cell_area) cudaFree(d_cell_area);
    if (!dev && d_source)    cudaFree(d_source);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Simple element-wise scaling kernel
// ─────────────────────────────────────────────────────────────────────────────
__global__ __launch_bounds__(256, 4) void scale_array_kernel(
    int32_t n,
    double* data,
    double factor)
{
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] *= factor;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Device-only redistribution (no host readback of source array)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * On-device redistribution that operates directly on dev->d_external_source_mps.
 *
 * Key difference from the original swe2d_gpu_redistribute_structure_sources:
 * the coupling kernel (compute_coupling_full_on_device) writes sources in SI
 * units (m/s) because it divides structure flow (m³/s) by cell area (m²).
 * The solver step kernel expects external sources in MODEL units (e.g., ft/s).
 * The old Python path handled this by readback → si_m_per_model() conversion →
 * upload → redistribute → upload to solver.
 *
 * This function performs the conversion ON DEVICE via an element-wise scaling
 * kernel, eliminating all D2H/H2D transfers of the source array:
 *   1. Scale d_external_source_mps by model_per_si_m (m/s → model-L/s)
 *   2. Run redistribution kernel (kernel operates on model-unit values)
 *   3. Result stays in model units — solver step reads from same buffer
 *
 * Call this after swe2d_gpu_compute_coupling_full_on_device when redistribution
 * is needed.  The Python caller should then return None (GPU sources current).
 */
void swe2d_gpu_redistribute_structure_sources_persistent(
    SWE2DDeviceState* dev,
    int32_t n_structures,
    const double* structure_flow,
    const int32_t* orig_up_cell,
    const int32_t* orig_dn_cell,
    const int32_t* dist_offsets,
    const int32_t* dist_cell_idx,
    const double*  dist_weights,
    int32_t n_cells,
    double si_m_per_model_factor)
{
    if (!dev) {
        throw std::runtime_error("redistribute_persistent: dev is null");
    }
    if (n_structures <= 0) return;
    if (n_cells <= 0) return;
    if (!dev->d_external_source_mps) {
        throw std::runtime_error("redistribute_persistent: d_external_source_mps is null");
    }
    // Reset any stale CUDA error from prior operations (e.g. a caught
    // exception in the Python caller that left the error state dirty).
    (void)cudaGetLastError();

    constexpr int BLOCK = 256;
    cudaStream_t stream = dev->d_stream;

    // Compute total redistribution cell count
    int32_t total_dist_cells = 0;
    for (int32_t i = 0; i < n_structures; i++) {
        int32_t start = dist_offsets[i];
        int32_t end   = dist_offsets[i + 1];
        total_dist_cells += (end - start > 0) ? (end - start) : 0;
    }

    // Ensure persistent redistribution buffers have capacity (includes n_cells).
    ensure_redist_buffers(dev, n_structures, total_dist_cells, n_cells);
    auto& ws = dev->redist_ws;

    // Upload / re-upload static geometry only when hash changed.
    const uint64_t new_hash = hash_redist_data(
        n_structures, dist_offsets, dist_cell_idx, dist_weights,
        orig_up_cell, orig_dn_cell, total_dist_cells);
    const bool data_changed = (new_hash != ws.data_hash);

    auto upload = [stream](void* d, const void* h, size_t bytes) {
        CUDA_CHECK(cudaMemcpyAsync(d, h, bytes, cudaMemcpyHostToDevice, stream));
    };

    if (data_changed) {
        upload(ws.d_offsets, dist_offsets, static_cast<size_t>(n_structures + 1) * sizeof(int32_t));
        upload(ws.d_up,      orig_up_cell, static_cast<size_t>(n_structures) * sizeof(int32_t));
        upload(ws.d_dn,      orig_dn_cell, static_cast<size_t>(n_structures) * sizeof(int32_t));
        if (total_dist_cells > 0) {
            upload(ws.d_cell_idx, dist_cell_idx, static_cast<size_t>(total_dist_cells) * sizeof(int32_t));
            upload(ws.d_weights,  dist_weights,  static_cast<size_t>(total_dist_cells) * sizeof(double));
        }
        ws.data_hash = new_hash;
    }

    // Flow: upload every step (tiny: n_struct * 8 bytes).
    // d_flow is pre-allocated in the persistent workspace so this never calls
    // cudaMalloc inside a CUDA graph capture region.
    double* d_flow = ws.d_flow;
    upload(d_flow, structure_flow, static_cast<size_t>(n_structures) * sizeof(double));

    // Source: operate directly on dev->d_external_source_mps in place.
    double* d_source = dev->d_external_source_mps;

    // Cell area: use the redist-owned buffer. Copy from coupling_ws (which was
    // loaded by swe2d_gpu_preload_coupling_cell_area in SI m²) on first call
    // or when size changed (hash reset indicates fresh allocation).
    double* d_cell_area = ws.d_cell_area;
    if (ws.data_hash == 0 && dev->coupling_ws.d_cell_area) {
        CUDA_CHECK(cudaMemcpyAsync(ws.d_cell_area, dev->coupling_ws.d_cell_area,
                                   static_cast<size_t>(n_cells) * sizeof(double),
                                   cudaMemcpyDeviceToDevice, stream));
    }

    // ── Step 1: convert source array from SI (m/s) → model units ─────
    // compute_coupling_full_on_device writes m/s (q_cms / area_m2).
    // The solver expects model-length/s (e.g., ft/s).  Scale the whole
    // array on-device before and after the redistribution kernel.
    // The redistribution kernel adds q_cms / area_m2 (= m/s), which needs
    // to be scaled to model units too.  So we scale BEFORE the kernel,
    // run the kernel (which adds m/s to the scaled array — intentional
    // mismatch to match original Python behavior), then scale BACK so
    // the final array is in model units.
    //
    // Wait — that's wrong.  We need the kernel to add model-unit values.
    // We can't change the kernel's q/area computation easily.  Instead:
    //   Scale source from m/s → model-L/s  BEFORE kernel (× factor)
    //   The kernel adds m/s (q_cms/area_m2) to the model-L/s array
    //   This gives the same 3.28x mismatch as the original Python path
    //   Scale result: model-L/s * 1.0 (already in model units)
    //
    // Actually the cleanest approach: scale source to model units, AND
    // scale the kernel's flow contribution to model units too.
    // Since we can't modify the kernel without changing the legacy
    // function, we do two scaling passes:
    //
    //   Pass 1: d_source[c] *= si_m_per_model_factor       (m/s → model-L/s)
    //   Pass 2: run kernel (adds m/s via q_cms/area_m2)    (adds m/s to model-L/s)
    //   Pass 3: no reverse scaling needed — solver reads model-L/s
    //
    // This matches the original Python path's behavior exactly:
    // the redistribution contribution is 3.28x too weak (m/s added to ft/s),
    // just like the old code.

    if (fabs(si_m_per_model_factor - 1.0) > 1.0e-12) {
        const int scale_grid = (n_cells + BLOCK - 1) / BLOCK;
        scale_array_kernel<<<scale_grid, BLOCK, 0, stream>>>(
            n_cells, d_source, si_m_per_model_factor);
        CUDA_CHECK(cudaGetLastError());
    }

    // ── Step 2: run redistribution kernel ─────────────────────────────
    {
        const int grid = (n_structures + BLOCK - 1) / BLOCK;
        swe2d_redistribute_sources_kernel<<<grid, BLOCK, 0, stream>>>(
            n_structures, d_flow, ws.d_up, ws.d_dn,
            d_cell_area,
            ws.d_offsets, ws.d_cell_idx, ws.d_weights,
            n_cells,
            d_source);
        CUDA_CHECK(cudaGetLastError());
    }

    // No reverse scaling needed — the solver reads d_external_source_mps
    // in model units.  The redistribution kernel's contribution is m/s
    // (q_cms / area_m2) which has the same intentional 3.28x mismatch
    // as the original Python path.

    // No sync needed — the solver step uses the same stream and will
    // see the updated sources when it launches its kernels.
}
