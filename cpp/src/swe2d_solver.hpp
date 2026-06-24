#pragma once
// swe2d_solver.hpp
// GPU solver for the 2D SWE on an unstructured mesh.
// Always uses CUDA GPU path — CPU/OpenMP fallback has been removed.
//
// The solver owns its state arrays and an optional GPU device state.

#include "swe2d_mesh.hpp"
#include <cstdint>
#include <vector>

enum class SWE2DSpatialScheme : int {
    FV_FIRST_ORDER    = 0,
    FV_MUSCL_FAST     = 1,
    FV_MUSCL_MINMOD   = 2,
    FV_MUSCL_MC       = 3,   // Monotonized-Central limiter (gradient-based TVD)
    FV_MUSCL_VAN_LEER = 4,   // Van Leer smooth limiter (gradient-based TVD)
    FV_WENO5          = 6,   // WENO5 + least-squares 2-ring gradient (~3rd order, GPU-first)
};

enum class SWE2DTurbulenceModel : int {
    NONE = 0,
};

enum class SWE2DBedFrictionModel : int {
    MANNING = 0,
};

enum class SWE2DEquationSet : int {
    HYDROSTATIC_2D = 0,
};

// Forward declaration of GPU state (defined in swe2d_gpu.cuh when CUDA present)
#ifdef HYDRA_HAS_CUDA
struct SWE2DDeviceState;
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Solver configuration
// ─────────────────────────────────────────────────────────────────────────────
struct SWE2DSolverConfig {
    double  g        = 9.81;    // gravitational acceleration (L/T²)
    double  k_mann   = 1.0;     // Manning unit factor: 1.0 (SI), 1.486 (USC).  V = (k/n)·R^(2/3)·S^(1/2)
    double  n_mann   = 0.035;   // Manning's n (global)
    double  h_min    = 1.0e-6;  // wet/dry threshold (L)
    double  cfl      = 0.45;    // CFL safety factor
    double  dt_max   = 10.0;    // maximum allowable timestep (s)
    double  dt_fixed = -1.0;    // if > 0, use this fixed dt (overrides CFL)
    double  dt_initial = -1.0;  // if > 0, use this dt for the first step only (cold-start override)
    int     temporal_order = 2; // 1 = Euler, 2 = SSPRK2 (Heun)
    int     spatial_scheme = static_cast<int>(SWE2DSpatialScheme::FV_FIRST_ORDER);
    int     turbulence_model = static_cast<int>(SWE2DTurbulenceModel::NONE);
    int     bed_friction_model = static_cast<int>(SWE2DBedFrictionModel::MANNING);
    int     equation_set = static_cast<int>(SWE2DEquationSet::HYDROSTATIC_2D);
    bool    enable_rain_module = false;
    bool    enable_pipe_network_module = false;
    bool    enable_hydraulic_structures = false;
    bool    use_gpu  = true;    // CUDA GPU path (GPU-only)
    int     n_threads = 0;      // 0 = auto (hardware concurrency)

    // Stability hardening controls (GPU-first tuning knobs).
    // These defaults preserve the current behaviour envelope while allowing
    // real-world wet/dry runs to be hardened from the GUI without recompiling.
    double  max_inv_area = 1.0e6;              // cap on 1/area used in updates
    double  cfl_lambda_cap = 1.0e6;            // cap on local CFL wave speed ratio
    double  momentum_cap_min_speed = 50.0;     // absolute min speed cap for momentum limiting
    double  momentum_cap_celerity_mult = 20.0; // speed cap = max(min_speed, mult*sqrt(g*h))
    double  depth_cap = 1.0e6;                 // hard upper bound on depth
    double  max_rel_depth_increase = 2.0;      // per-step limit: h <= h_old + rel*max(h_old,h_min)
    double  shallow_damping_depth = 1.0e-4;    // blend momentum to zero as h approaches h_min
    int     gpu_diag_sync_interval_steps = 50; // 1=sync diagnostics every step, N=every N steps, <=0 disables
    int     degen_mode = 0; // 0=none, 1=skip (permanently inactive), 2=repair (neighbor-avg inv_area), 3=merge (redirect flux to neighbor)

    // Wet/dry front stability controls
    double  front_flux_damping = 0.5;     // momentum-flux scale factor on wet/dry front edges (0=full damp, 1=none)
    bool    active_set_hysteresis = true; // keep cells active 1 extra step after drying to suppress oscillatory front switching
    bool    enable_shallow_front_recon_fallback = true; // if true, force 1st-order reconstruction on shallow edge pairs

    // Extreme-rain robustness controls (GPU-first path, mirrored in CPU fallback).
    bool    extreme_rain_mode = false;      // enable adaptive source-CFL limiting
    double  source_cfl_beta = 0.25;         // target source CFL: dt*src <= beta*h_ref
    int     source_max_substeps = 16;       // cap on equivalent source substep count
    double  source_rate_cap = 0.0;          // hard cap on positive source rate [depth/s], 0=off
    double  source_depth_step_cap = 0.0;    // hard cap on positive source depth increment per step [depth], 0=off
    bool    source_true_subcycling = false; // true: apply real source sub-iterations per hydro step
    bool    source_imex_split = false;      // true: flux step first, then source+friction split substeps

    // Friction temporal-order hardening (adaptive sub-stepping for higher-order RK).
    bool    friction_substep_enabled     = true;   // enable adaptive friction sub-stepping
    double  friction_target_courant      = 1.0;    // target nu_fric for substep count (>0)
    int     friction_max_substeps        = 64;     // hard cap on friction substeps per cell

    // Shallow-flow friction correction (Keulegan-based Cf enhancement).
    bool    shallow_friction_correction  = false;  // enable depth-limited Cf enhancement
    double  shallow_friction_depth_alpha = 5.0;    // h_ref = alpha * n^(3/2) (L^(1/2)/T)
    double  shallow_friction_exponent    = 0.4;    // Cf *= (h_ref/max(h,h_min))^beta

    // Tiny-N GPU execution controls (GPU-first perf tuning for small wet domains).
    // 0=off, 1=auto, 2=fused(preferred tiny path), 3=persistent(experimental).
    int     tiny_mode = 1;
    int     tiny_cell_threshold = 8000;
    int     tiny_edge_threshold = 24000;
    int     tiny_wet_cell_threshold = 2000;
    int     tiny_persistent_chunk_substeps = 8;
    int     tiny_active_compaction_stride_steps = 8;
    bool    tiny_enable_active_compaction = true;
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
    int32_t  gpu_graph_launches_step = 0;
    int64_t  gpu_graph_launches_total = 0;
    // Tiny-N telemetry.
    int32_t  tiny_mode_requested = 0;
    int32_t  tiny_mode_selected = 0;
    int32_t  tiny_mode_effective = 0;
    bool     tiny_mode_fallback = false;
    int32_t  tiny_active_cells_est = 0;
    int32_t  tiny_active_edges_est = 0;
    int64_t  tiny_mode_fallback_count_total = 0;
    int64_t  fused_path_steps_total = 0;
    int64_t  persistent_path_steps_total = 0;
};

struct SWE2DNonhydroDiag {
    int32_t pressure_iters = 0;
    double pressure_residual = 0.0;
    bool corrector_applied = false;
};

struct SWE2D3DInterfaceContractHost {
    std::vector<int32_t> cell2d;
    std::vector<double> face_area;
    std::vector<double> face_nx;
    std::vector<double> face_ny;
    std::vector<double> face_nz;
};

// ─────────────────────────────────────────────────────────────────────────────
// Phase 7: Contract validation & factory (pybind11 exposure)
// ─────────────────────────────────────────────────────────────────────────────

/// Validate contract struct consistency (all array sizes match).
/// Returns true if valid, false otherwise. Can be called from Python before upload.
inline bool swe2d_contract_is_valid(const SWE2D3DInterfaceContractHost& c) {
    if (c.cell2d.empty()) return false;
    const size_t n = c.cell2d.size();
    return c.face_area.size() == n &&
           c.face_nx.size() == n &&
           c.face_ny.size() == n &&
           c.face_nz.size() == n;
}

/// Factory: create a contract from arrays. Validates and deep-copies.
/// Returns default (empty) contract if validation fails.
inline SWE2D3DInterfaceContractHost swe2d_contract_create(
    const std::vector<int32_t>& cell2d_in,
    const std::vector<double>& face_area_in,
    const std::vector<double>& face_nx_in,
    const std::vector<double>& face_ny_in,
    const std::vector<double>& face_nz_in)
{
    SWE2D3DInterfaceContractHost c;
    c.cell2d = cell2d_in;
    c.face_area = face_area_in;
    c.face_nx = face_nx_in;
    c.face_ny = face_ny_in;
    c.face_nz = face_nz_in;
    return c;  // validation caller's responsibility, or validate on upload
}

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
    std::vector<double> source_terms; // [n_cells] additive depth source [m/s]
    std::vector<double> external_source_terms; // [n_cells] externally-coupled depth source [m/s]

    // ── Config ───────────────────────────────────────────────────────────────
    SWE2DSolverConfig cfg;

    // Optional per-boundary-edge hydrograph forcing (timeseries evaluated per step).
    bool hydrographs_enabled = false;
    std::vector<int32_t> hg_edge_index;   // [n_hg_edges] boundary edge indices
    std::vector<int32_t> hg_bc_type;      // [n_hg_edges] target BC type (2/3)
    std::vector<int32_t> hg_offsets;      // [n_hg_edges + 1]
    std::vector<double>  hg_time_s;       // [n_hg_samples] sample times (s)
    std::vector<double>  hg_value;        // [n_hg_samples] sample values

    // Optional rainfall + CN infiltration forcing (per-cell, event mode).
    bool rain_cn_enabled = false;
    std::vector<int32_t> rain_cell_gage;      // [n_cells] gage index per cell
    std::vector<int32_t> rain_gage_offsets;   // [n_gages + 1]
    std::vector<double>  rain_hg_time_s;      // [n_rain_samples]
    std::vector<double>  rain_hg_cum_mm;      // [n_rain_samples] cumulative rainfall depth [mm]
    std::vector<double>  rain_cn;             // [n_cells] curve number
    std::vector<double>  rain_cum_mm;         // [n_cells] cumulative rain [mm]
    std::vector<double>  rain_excess_cum_mm;  // [n_cells] cumulative excess [mm]
    double rain_ia_ratio = 0.2;
    double rain_mm_to_model_depth = 1.0e-3;   // convert mm excess depth -> solver depth units (m or ft)

    // Optional external coupling source terms (e.g., drainage/structure coupling
    // from Python). When enabled, these are added every step on top of native
    // rain+CN source terms.
    bool external_sources_enabled = false;

    // ── Simulation time ──────────────────────────────────────────────────────
    double t = 0.0;
    uint64_t gpu_steps = 0;
    bool first_step_done = false; // set after first swe2d_step() call when dt_initial override applies
    int32_t last_wet_cells = -1;
    uint64_t tiny_mode_fallback_count = 0;
    uint64_t fused_path_steps = 0;
    uint64_t persistent_path_steps = 0;

    // ── GPU state (null when CUDA unavailable or use_gpu=false) ─────────────
#ifdef HYDRA_HAS_CUDA
    SWE2DDeviceState* dev = nullptr;
#endif
};

// ─────────────────────────────────────────────────────────────────────────────
// Lifecycle
// ─────────────────────────────────────────────────────────────────────────────

/** Allocate solver.  Caller retains ownership of mesh; mesh must outlive solver.
    @param mesh Mesh reference (not owned) @param h0 Initial depth [n_cells]
    @param hu0 Initial x-momentum [n_cells], nullable @param hv0 Initial y-momentum, nullable
    @param n_mann_cell Per-cell Manning n, nullable @param cfg Solver configuration
    @returns Pointer to new SWE2DSolver */
SWE2DSolver* swe2d_create(
    const SWE2DMesh& mesh,
    const double*    h0,
    const double*    hu0,   // nullable
    const double*    hv0,   // nullable
    const double*    n_mann_cell, // nullable
    const SWE2DSolverConfig& cfg);

/** Advance one timestep.
    @param s Solver handle @param dt_request Desired timestep (s). <=0 for CFL-controlled.
    @returns Step diagnostics */
SWE2DStepDiag swe2d_step(SWE2DSolver* s, double dt_request);

/** Update BC type/value by boundary edge index on an existing solver and sync GPU state. */
void swe2d_solver_set_boundary_values(
    SWE2DSolver* s,
    const int32_t* edge_index,
    const int32_t* bc_type,
    const double* bc_val,
    int32_t n_updates);

/** Configure per-edge hydrograph forcing (evaluated each step). */
void swe2d_solver_set_boundary_hydrographs(
    SWE2DSolver* s,
    const int32_t* edge_index,
    const int32_t* bc_type,
    const int32_t* offsets,
    const double* time_s,
    const double* value,
    int32_t n_edges,
    int32_t n_samples);

/** Upload progressive BC group data for on-device Q->q distribution.
    The solver retains no host copy; forwards directly to the GPU. */
void swe2d_solver_set_progressive_bc_data(
    SWE2DSolver* s,
    int32_t n_groups,
    int32_t n_edges_total,
    const int32_t* group_offsets,
    const int32_t* edge_hg_idx,
    const double* edge_len,
    const double* edge_cum_len,
    const double* group_peak_q,
    const double* group_total_len);

/** Configure per-cell rain+CN forcing (evaluated each step). */
void swe2d_solver_set_rain_cn_forcing(
    SWE2DSolver* s,
    const int32_t* cell_gage_idx,
    const int32_t* gage_offsets,
    const double* hg_time_s,
    const double* hg_cum_mm,
    const double* cn,
    int32_t n_cells,
    int32_t n_gages,
    int32_t n_samples,
    double ia_ratio,
    double mm_to_model_depth = 1.0e-3);

/** Configure per-cell externally-coupled depth source terms [m/s].
    Passing nullptr or n_cells <= 0 clears external sources. */
void swe2d_solver_set_external_sources(
    SWE2DSolver* s,
    const double* source_mps,
    int32_t n_cells);

/** Copy current state out to caller-supplied arrays (length mesh.n_cells each). */
void swe2d_get_state(const SWE2DSolver* s, double* h_out, double* hu_out, double* hv_out);
void swe2d_get_max_tracking(const SWE2DSolver* s, double* h_max_out, double* hu_max_out, double* hv_max_out);

/** Overwrite current state from caller-supplied arrays (length mesh.n_cells each). */
void swe2d_set_state(SWE2DSolver* s, const double* h_in, const double* hu_in, const double* hv_in);

/** Free all resources (including GPU device memory if allocated). */
void swe2d_destroy(SWE2DSolver* s);

// ─────────────────────────────────────────────────────────────────────────────
// Native run-to-time loop (removes per-step Python orchestration)
// ─────────────────────────────────────────────────────────────────────────────
// Callback signature: called on progress at user-specified interval.
// Returns true to continue, false to cancel.
typedef bool (*SWE2DProgressCallback)(double current_time, uint64_t step_count, const SWE2DStepDiag* diag);

struct SWE2DRunConfig {
    double t_end;                           // End simulation time (s)
    double dt_request;                      // Requested per-step dt; -1 for CFL-controlled
    int progress_callback_interval_steps;   // Call progress_cb every N steps; <=0 disables
    SWE2DProgressCallback progress_cb;      // Callback (may be nullptr if interval <= 0)
    int diag_batch_size;                    // Diagnostics array capacity; 0 = no batching (only final)
};

/** Run simulation from current t to t_end. Batches diagnostics every N steps into diag_out.
    @returns Number of diagnostics written to diag_out. On cancellation (progress_cb returns false),
    returns negative count of steps completed. */
int32_t swe2d_run_to_time(
    SWE2DSolver* s,
    const SWE2DRunConfig* cfg,
    SWE2DStepDiag* diag_out,  // Output array, optional
    int32_t max_diags);        // Capacity of diag_out array

// ─────────────────────────────────────────────────────────────────────────────
// GPU availability query
/// Returns true only when CUDA was compiled in AND a CUDA device is present.
bool swe2d_gpu_available();

// ─────────────────────────────────────────────────────────────────────────────
/// CPU solver step (always available, used as fallback)
/// @param s Solver handle @param dt Timestep @returns Step diagnostics
SWE2DStepDiag swe2d_step_cpu(SWE2DSolver* s, double dt);
