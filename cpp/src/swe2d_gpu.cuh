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

    // Cell centroids (needed for gradient-based higher-order reconstruction)
    double*  d_cell_cx = nullptr;
    double*  d_cell_cy = nullptr;

    // Per-cell gradient arrays (Green-Gauss, for MC and Van Leer limiters)
    double*  d_grad_hx  = nullptr;   double*  d_grad_hy  = nullptr;
    double*  d_grad_hux = nullptr;   double*  d_grad_huy = nullptr;
    double*  d_grad_hvx = nullptr;   double*  d_grad_hvy = nullptr;

    // Conserved state (updated each step)
    double*  d_h  = nullptr;
    double*  d_hu = nullptr;
    double*  d_hv = nullptr;

    // RK2 backup state (U^n)
    double*  d_h0  = nullptr;
    double*  d_hu0 = nullptr;
    double*  d_hv0 = nullptr;

    // Flux accumulators (zeroed each step)
    double*  d_flux_h  = nullptr;
    double*  d_flux_hu = nullptr;
    double*  d_flux_hv = nullptr;

    // CFL workspace (device scalar)
    double*  d_lambda_max = nullptr;
    double*  d_max_wse_elev_error = nullptr;
    // Packed diagnostic buffer: [0]=lambda_max, [1]=max_wse_elev_error, [2]=(double)n_wet.
    // Filled on-device by pack_diag_kernel after each step; a single cudaMemcpy
    // of 24 bytes transfers all three values when sync_diagnostics is true.
    double*  d_diag_packed = nullptr;

    // Wet/dry active-set mask (updated at the start of every step).
    // d_active[c] = 1 if cell c is wet (h>h_min), adjacent to a wet cell,
    // or at a forced-inflow BC edge.  Used to skip gradient and update work
    // for fully-isolated dry cells.
    int32_t* d_active    = nullptr;   // n_cells
    int32_t* d_n_wet     = nullptr;   // device scalar: count of h>h_min cells
    int32_t* d_bc_forced = nullptr;   // n_cells: 1 if cell has forced-inflow BC
    // Hysteretic active set: stores d_active from the PREVIOUS step.
    // Passed to swe2d_classify_kernel so cells that were active last step and
    // still have h > 0 are kept active for one extra step, suppressing
    // rapid oscillatory activation/deactivation at wet/dry fronts.
    int32_t* d_was_active = nullptr;  // n_cells

    // Degenerate-cell handling (computed once at init; all null when degen_mode == 0).
    // degen_mode mirrors SWE2DSolverConfig::degen_mode.
    int32_t  degen_mode          = 0;
    int32_t* d_degen_mask        = nullptr;  // [n_cells]: 1 if cell_inv_area > max_inv_area
    double*  d_inv_area_repaired = nullptr;  // [n_cells]: neighbor-averaged inv_area (mode 2)
    int32_t* d_merge_owner       = nullptr;  // [n_cells]: merge-to cell index (mode 3), -1 if none

    // Persistent CUDA stream — all per-step kernel launches and async memsets
    // go on this stream.  Allows CPU-side work (BC updates, Python callbacks)
    // to overlap with GPU execution between steps.
    cudaStream_t d_stream = nullptr;

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
    const double*    n_mann_cell,
    int              degen_mode   = 0,
    double           max_inv_area = 1.0e6);

// Advance one timestep on GPU.  Writes diagnostics to *diag.
void swe2d_gpu_step(
    SWE2DDeviceState* dev,
    double dt,
    double g,
    double h_min,
    int spatial_scheme,
    double cfl_factor,
    double max_inv_area,
    double cfl_lambda_cap,
    double momentum_cap_min_speed,
    double momentum_cap_celerity_mult,
    double depth_cap,
    double max_rel_depth_increase,
    double shallow_damping_depth,
    bool sync_diagnostics,
    SWE2DStepDiag* diag,
    double front_flux_damping    = 0.5,
    bool   active_set_hysteresis = true);

// Advance one SSPRK2 (Heun) timestep fully on GPU.
void swe2d_gpu_step_rk2(
    SWE2DDeviceState* dev,
    double dt,
    double g,
    double h_min,
    int spatial_scheme,
    double cfl_factor,
    double max_inv_area,
    double cfl_lambda_cap,
    double momentum_cap_min_speed,
    double momentum_cap_celerity_mult,
    double depth_cap,
    double max_rel_depth_increase,
    double shallow_damping_depth,
    bool sync_diagnostics,
    SWE2DStepDiag* diag,
    double front_flux_damping    = 0.5,
    bool   active_set_hysteresis = true);

// Compute a CFL-limited dt from current device state without host-state sync.
double swe2d_gpu_compute_dt(
    SWE2DDeviceState* dev,
    double g,
    double h_min,
    double cfl_factor,
    double dt_max,
    double cfl_lambda_cap);

// Copy current state from device to caller-supplied host arrays.
void swe2d_gpu_get_state(
    SWE2DDeviceState* dev,
    double* h_out,
    double* hu_out,
    double* hv_out);

// Upload host state arrays into the current device solver state.
void swe2d_gpu_set_state(
    SWE2DDeviceState* dev,
    const double* h_in,
    const double* hu_in,
    const double* hv_in);

// Free all device memory.
void swe2d_gpu_destroy(SWE2DDeviceState* dev);

// Query: returns true if a CUDA-capable device is available.
bool swe2d_gpu_available();
