// Phase 4.1 — device-side diagnostic ring buffer implementation.
// See swe2d_gpu_diag_ring.cuh for the contract.
#include "swe2d_gpu_diag_ring.cuh"
#include <cuda_runtime.h>
#include <cstring>

namespace swe2d_diag_ring {

DiagRecord* d_diag_ring = nullptr;
int32_t diag_ring_capacity = 0;
int32_t diag_ring_count = 0;

void init(int32_t initial_capacity) {
    if (d_diag_ring != nullptr) return;
    if (initial_capacity <= 0) initial_capacity = 1024;
    cudaMalloc(&d_diag_ring, sizeof(DiagRecord) * initial_capacity);
    diag_ring_capacity = initial_capacity;
    diag_ring_count = 0;
}

void shutdown() {
    if (d_diag_ring != nullptr) {
        cudaFree(d_diag_ring);
        d_diag_ring = nullptr;
    }
    diag_ring_capacity = 0;
    diag_ring_count = 0;
}

__global__ void push_kernel(
    DiagRecord* ring, int32_t slot,
    double t_s, double dt_used, int32_t gpu_active, int32_t wet_cells,
    double max_courant, double max_wse_error, double mass_total)
{
    DiagRecord r;
    r.t_s = t_s;
    r.dt_used = dt_used;
    r.gpu_active = gpu_active;
    r.wet_cells = wet_cells;
    r.max_courant = max_courant;
    r.max_wse_error = max_wse_error;
    r.mass_total = mass_total;
    ring[slot] = r;
}

void grow_if_full() {
    if (diag_ring_count < diag_ring_capacity) return;
    int32_t new_cap = diag_ring_capacity * 2;
    DiagRecord* new_ring;
    cudaMalloc(&new_ring, sizeof(DiagRecord) * new_cap);
    cudaMemcpy(new_ring, d_diag_ring,
               sizeof(DiagRecord) * diag_ring_capacity,
               cudaMemcpyDeviceToDevice);
    cudaFree(d_diag_ring);
    d_diag_ring = new_ring;
    diag_ring_capacity = new_cap;
}

void push(
    double t_s, double dt_used, int32_t gpu_active, int32_t wet_cells,
    double max_courant, double max_wse_error, double mass_total)
{
    if (d_diag_ring == nullptr) init(1024);
    grow_if_full();
    int32_t slot = diag_ring_count;
    push_kernel<<<1, 1>>>(d_diag_ring, slot,
        t_s, dt_used, gpu_active, wet_cells,
        max_courant, max_wse_error, mass_total);
    diag_ring_count++;
}

int32_t count() {
    return diag_ring_count;
}

void read_latest(DiagRecord* host_buf) {
    if (d_diag_ring == nullptr || diag_ring_count == 0 || host_buf == nullptr) return;
    cudaMemcpy(host_buf, d_diag_ring + (diag_ring_count - 1),
               sizeof(DiagRecord), cudaMemcpyDeviceToHost);
}

void clear() {
    diag_ring_count = 0;
}

}  // namespace swe2d_diag_ring