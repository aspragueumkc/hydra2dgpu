#pragma once
// swe2d_gpu.cuh
// CUDA device state and host API declarations for the 2D SWE GPU path.
// Included only when BACKWATER_HAS_CUDA is defined.

#include "swe2d_mesh.hpp"
#include "swe2d_solver.hpp"   // SWE2DStepDiag

#include <cuda_runtime.h>
#include <cstdint>

// ─────────────────────────────────────────────────────────────────────────────
// Device memory pool for one solver instance
// ─────────────────────────────────────────────────────────────────────────────
struct SWE2DDeviceState {
    // Mesh topology (static after init, transferred once)
    int32_t* d_edge_c0     = nullptr;
    int32_t* d_edge_c1     = nullptr;
    int32_t* d_edge_n0     = nullptr;
    int32_t* d_edge_n1     = nullptr;
    double*  d_edge_nx     = nullptr;
    double*  d_edge_ny     = nullptr;
    double*  d_edge_len    = nullptr;
    int32_t* d_edge_bc     = nullptr;   // BCType stored as int32_t for CUDA compatibility
    double*  d_edge_bc_val = nullptr;

    double*  d_cell_zb     = nullptr;
    double*  d_cell_area   = nullptr;
    double*  d_cell_inv_area = nullptr;
    double*  d_n_mann_cell = nullptr;

    // Conserved state (updated each step)
    double*  d_h  = nullptr;
    double*  d_hu = nullptr;
    double*  d_hv = nullptr;

    // Flux accumulators (zeroed each step)
    double*  d_flux_h  = nullptr;
    double*  d_flux_hu = nullptr;
    double*  d_flux_hv = nullptr;

    // CFL workspace (device scalar)
    double*  d_lambda_max = nullptr;

    // Dimensions
    int32_t  n_cells = 0;
    int32_t  n_edges = 0;
};

// ─────────────────────────────────────────────────────────────────────────────
// Host API (callable from swe2d_solver.cpp)
// ─────────────────────────────────────────────────────────────────────────────

// Allocate device memory and transfer static mesh topology + initial state.
SWE2DDeviceState* swe2d_gpu_init(
    const SWE2DMesh& mesh,
    const double*    h0,
    const double*    hu0,
    const double*    hv0,
    const double*    n_mann_cell);

// Advance one timestep on GPU.  Writes diagnostics to *diag.
void swe2d_gpu_step(
    SWE2DDeviceState* dev,
    double dt,
    double g,
    double h_min,
    double cfl_factor,
    SWE2DStepDiag* diag);

// Copy current state from device to caller-supplied host arrays.
void swe2d_gpu_get_state(
    SWE2DDeviceState* dev,
    double* h_out,
    double* hu_out,
    double* hv_out);

// Free all device memory.
void swe2d_gpu_destroy(SWE2DDeviceState* dev);

// Query: returns true if a CUDA-capable device is available.
bool swe2d_gpu_available();
