#pragma once
// swe2d_solver.hpp
// CPU solver for the 2D SWE on an unstructured triangular mesh.
// Uses OpenMP for parallelism when BACKWATER_HAS_OPENMP is defined.
//
// The solver owns its state arrays and an optional GPU device state.
// At runtime it selects CPU or GPU path based on availability and config.

#include "swe2d_mesh.hpp"
#include <cstdint>
#include <vector>

enum class SWE2DSpatialScheme : int {
    FV_FIRST_ORDER    = 0,
    FV_MUSCL_FAST     = 1,
    FV_MUSCL_MINMOD   = 2,
    FV_MUSCL_MC       = 3,   // Monotonized-Central limiter (gradient-based TVD)
    FV_MUSCL_VAN_LEER = 4,   // Van Leer smooth limiter (gradient-based TVD)
};

enum class SWE2DTurbulenceModel : int {
    NONE = 0,
    SMAGORINSKY = 1,
    K_EPSILON = 2,
    K_OMEGA_SST = 3,
};

enum class SWE2DBedFrictionModel : int {
    MANNING = 0,
    CHEZY = 1,
    DARCY_WEISBACH = 2,
    NIKURADSE = 3,
};

// Forward declaration of GPU state (defined in swe2d_gpu.cuh when CUDA present)
#ifdef BACKWATER_HAS_CUDA
struct SWE2DDeviceState;
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Solver configuration
// ─────────────────────────────────────────────────────────────────────────────
struct SWE2DSolverConfig {
    double  g        = 9.81;    // gravitational acceleration (m/s²)
    double  n_mann   = 0.035;   // Manning's n (global; m^{-1/3} s)
    double  h_min    = 1.0e-6;  // wet/dry threshold (m)
    double  cfl      = 0.45;    // CFL safety factor
    double  dt_max   = 10.0;    // maximum allowable timestep (s)
    double  dt_fixed = -1.0;    // if > 0, use this fixed dt (overrides CFL)
    int     temporal_order = 2; // 1 = Euler, 2 = SSPRK2 (Heun)
    int     spatial_scheme = static_cast<int>(SWE2DSpatialScheme::FV_FIRST_ORDER);
    int     turbulence_model = static_cast<int>(SWE2DTurbulenceModel::NONE);
    int     bed_friction_model = static_cast<int>(SWE2DBedFrictionModel::MANNING);
    bool    enable_rain_module = false;
    bool    enable_pipe_network_module = false;
    bool    enable_hydraulic_structures = false;
    bool    use_gpu  = true;    // attempt CUDA path; falls back to CPU
    int     n_threads = 0;      // 0 = auto (OMP_NUM_THREADS or hardware)
};

// ─────────────────────────────────────────────────────────────────────────────
// Per-step diagnostics
// ─────────────────────────────────────────────────────────────────────────────
struct SWE2DStepDiag {
    double   dt         = 0.0;
    int32_t  wet_cells  = 0;
    double   max_depth  = 0.0;
    double   min_depth  = 0.0;
    double   mass_total = 0.0;
    double   max_courant = 0.0;
    double   max_depth_residual = 0.0;
    double   max_wse_elev_error = 0.0;
    bool     gpu_active = false;
};

// ─────────────────────────────────────────────────────────────────────────────
// Solver handle
// ─────────────────────────────────────────────────────────────────────────────
struct SWE2DSolver {
    // ── Mesh reference (not owned) ────────────────────────────────────────────
    const SWE2DMesh* mesh = nullptr;

    // ── Conserved state (host) ───────────────────────────────────────────────
    std::vector<double> h;    // [n_cells] water depth (m)
    std::vector<double> hu;   // [n_cells] x-momentum (m²/s)
    std::vector<double> hv;   // [n_cells] y-momentum (m²/s)
    std::vector<double> n_mann_cell; // [n_cells] per-cell Manning n

    // ── Flux accumulators (host, reused each step) ───────────────────────────
    std::vector<double> dh;   // [n_cells] accumulated depth flux / area
    std::vector<double> dhu;  // [n_cells]
    std::vector<double> dhv;  // [n_cells]

    // ── Cell-neighbor connectivity (CSR) for higher-order reconstruction ────
    std::vector<int32_t> cell_nbr_offsets; // [n_cells + 1]
    std::vector<int32_t> cell_nbr_ids;     // [sum(n_nbr_cell)]

    // ── Config ───────────────────────────────────────────────────────────────
    SWE2DSolverConfig cfg;

    // ── Simulation time ──────────────────────────────────────────────────────
    double t = 0.0;

    // ── GPU state (null when CUDA unavailable or use_gpu=false) ─────────────
#ifdef BACKWATER_HAS_CUDA
    SWE2DDeviceState* dev = nullptr;
#endif
};

// ─────────────────────────────────────────────────────────────────────────────
// Lifecycle
// ─────────────────────────────────────────────────────────────────────────────

// Allocate solver.  Caller retains ownership of mesh; mesh must outlive solver.
// h0, hu0, hv0: initial condition arrays of length mesh.n_cells.
//   hu0 and hv0 may be nullptr (zero-initialised).
SWE2DSolver* swe2d_create(
    const SWE2DMesh& mesh,
    const double*    h0,
    const double*    hu0,   // nullable
    const double*    hv0,   // nullable
    const double*    n_mann_cell, // nullable
    const SWE2DSolverConfig& cfg);

// Advance one timestep.
// dt_request: desired timestep (s).  Actual dt may be smaller due to CFL constraint.
//   Pass dt_request <= 0 to use CFL-controlled timestep.
SWE2DStepDiag swe2d_step(SWE2DSolver* s, double dt_request);

// Copy current state out to caller-supplied arrays (length mesh.n_cells each).
void swe2d_get_state(const SWE2DSolver* s, double* h_out, double* hu_out, double* hv_out);

// Overwrite current state from caller-supplied arrays (length mesh.n_cells each).
void swe2d_set_state(SWE2DSolver* s, const double* h_in, const double* hu_in, const double* hv_in);

// Free all resources (including GPU device memory if allocated).
void swe2d_destroy(SWE2DSolver* s);

// ─────────────────────────────────────────────────────────────────────────────
// GPU availability query
// Returns true only when CUDA was compiled in AND a CUDA device is present.
// ─────────────────────────────────────────────────────────────────────────────
bool swe2d_gpu_available();

// ─────────────────────────────────────────────────────────────────────────────
// CPU solver step (always available, used as fallback)
// ─────────────────────────────────────────────────────────────────────────────
SWE2DStepDiag swe2d_step_cpu(SWE2DSolver* s, double dt);
