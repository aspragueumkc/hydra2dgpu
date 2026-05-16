#pragma once
// swe2d_gpu.cuh
// CUDA device state and host API declarations for the 2D SWE GPU path.
// Included only when BACKWATER_HAS_CUDA is defined.

#include "swe2d_mesh.hpp"
#include "swe2d_solver.hpp"   // SWE2DStepDiag

#include <cuda_runtime.h>
#include <cstdint>

// ─────────────────────────────────────────────────────────────────────────────
// CUDA Graph cache for optimized kernel sequence replay
// ─────────────────────────────────────────────────────────────────────────────
struct KernelGraphCache {
    cudaGraph_t       graph = nullptr;       // Captured graph template
    cudaGraphExec_t   exec = nullptr;        // Executable instance for replay
    int32_t           n_cells = 0;           // Mesh size at capture time
    int32_t           n_edges = 0;           // Edge count at capture time
    int32_t           spatial_scheme = 0;    // Spatial scheme at capture
    int32_t           time_integrator = 0;   // RK order (2 or 4) at capture
    uint64_t          config_signature = 0;  // Scalar/runtime config signature
    bool              is_valid = false;      // True if graph can be replayed
    
    void destroy() {
        if (exec != nullptr) {
            cudaGraphExecDestroy(exec);
            exec = nullptr;
        }
        if (graph != nullptr) {
            cudaGraphDestroy(graph);
            graph = nullptr;
        }
        is_valid = false;
    }
};

enum class SWE2DPressureMatrixType : int {
    COLOCATED_COMPACT_STENCIL = 0,
    STAGGERED_VELOCITY = 1,
};

enum class SWE2DPreconditionerType : int {
    JACOBI = 0,
    BLOCK_JACOBI = 1,
    ILU0 = 2,
};

enum class SWE2DVelocityCorrectionMethod : int {
    DIVERGENCE_FREE = 0,
    ENERGY_STABLE = 1,
};

struct SWE2DNonhydroPcConfig {
    int pressure_max_iters = 100;
    double pressure_tol = 1.0e-5;
    double relax = 1.0;
    int matrix_type = static_cast<int>(SWE2DPressureMatrixType::COLOCATED_COMPACT_STENCIL);
    int preconditioner = static_cast<int>(SWE2DPreconditionerType::JACOBI);
    int velocity_correction = static_cast<int>(SWE2DVelocityCorrectionMethod::DIVERGENCE_FREE);
    bool use_adaptive_nh = false;
    double froude_activation_threshold = 0.5;
    double aspect_ratio_activation_threshold = 5.0;
};

struct SWE2DNonhydroPcDiag {
    int pressure_iters = 0;
    double pressure_residual = 0.0;
    bool corrector_applied = false;
};

struct SWE3DCartesianPatchDesc {
    int32_t nx = 0;
    int32_t ny = 0;
    int32_t nz = 0;
    double dx = 0.0;
    double dy = 0.0;
    double dz = 0.0;
    double origin_x = 0.0;
    double origin_y = 0.0;
    double origin_z = 0.0;
    bool single_phase_free_surface = true;
};

struct SWE3DCartesianPatchDeviceState {
    SWE3DCartesianPatchDesc desc;
    int64_t n_cells = 0;
    double* d_u = nullptr;
    double* d_v = nullptr;
    double* d_w = nullptr;
    double* d_p = nullptr;
    double* d_vof = nullptr;
    // 3D projection workspace (uncoupled path): pressure RHS + Jacobi scratch.
    double* d_p_rhs = nullptr;
    double* d_p_tmp = nullptr;
    unsigned long long* d_proj_residual_bits = nullptr;
    // Last projection diagnostics from swe2d_gpu_step_3d_single_phase_free_surface.
    int32_t last_projection_iters = 0;
    double last_projection_residual = -1.0;
    bool last_projection_converged = false;
};

struct SWE2D3DInterfaceContractDevice {
    int32_t n_faces = 0;
    int32_t* d_cell2d = nullptr;
    double* d_face_area = nullptr;
    double* d_face_nx = nullptr;
    double* d_face_ny = nullptr;
    double* d_face_nz = nullptr;
    double* d_flux_mass_2d_to_3d = nullptr;
    double* d_flux_momx_2d_to_3d = nullptr;
    double* d_flux_momy_2d_to_3d = nullptr;
    double* d_head_loss_3d_to2d = nullptr;
};

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
    double*  d_edge_mx     = nullptr;
    double*  d_edge_my     = nullptr;
    int32_t* d_edge_bc     = nullptr;   // BCType stored as int32_t for CUDA compatibility
    double*  d_edge_bc_val = nullptr;
    // Per-stage boundary forcing snapshots used by graph-safe higher-order schemes.
    // Layout is contiguous by stage: slot*swe_n_edges + edge.
    int32_t* d_stage_edge_bc = nullptr;
    double*  d_stage_edge_bc_val = nullptr;

    // Cell-to-edge CSR, used by the atomics-free unstructured kernels.
    int32_t* d_cell_edge_offsets = nullptr;  // [n_cells + 1]
    int32_t* d_cell_edge_ids     = nullptr;   // [sum(n_verts_cell)]

    // Per-edge hydrograph forcing (optional, evaluated on GPU each step).
    int32_t* d_hg_edge_index = nullptr;   // [n_hg_edges]
    int32_t* d_hg_bc_type = nullptr;      // [n_hg_edges]
    int32_t* d_hg_offsets = nullptr;      // [n_hg_edges+1]
    double*  d_hg_time_s = nullptr;       // [n_hg_samples]
    double*  d_hg_value = nullptr;        // [n_hg_samples]
    int32_t  n_hg_edges = 0;
    int32_t  n_hg_samples = 0;

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

    // RK4 intermediate stages (allocated on demand when temporal_order >= 4)
    // Used to store results from stages k1, k2, k3 during 4-stage integration.
    double*  d_h1  = nullptr;
    double*  d_hu1 = nullptr;
    double*  d_hv1 = nullptr;
    double*  d_h2  = nullptr;
    double*  d_hu2 = nullptr;
    double*  d_hv2 = nullptr;
    double*  d_h3  = nullptr;
    double*  d_hu3 = nullptr;
    double*  d_hv3 = nullptr;
    // k4 slope buffer for graph-safe true RK4 (temporal_order=5)
    double*  d_k4_h  = nullptr;
    double*  d_k4_hu = nullptr;
    double*  d_k4_hv = nullptr;
    // Extra slope buffers for graph-safe RK5 (temporal_order=6)
    double*  d_k5_h  = nullptr;
    double*  d_k5_hu = nullptr;
    double*  d_k5_hv = nullptr;
    double*  d_k6_h  = nullptr;
    double*  d_k6_hu = nullptr;
    double*  d_k6_hv = nullptr;

    // Flux accumulators (zeroed each step)
    double*  d_flux_h  = nullptr;
    double*  d_flux_hu = nullptr;
    double*  d_flux_hv = nullptr;
    double*  d_flux_hu_r = nullptr;
    double*  d_flux_hv_r = nullptr;

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

    // Rainfall + CN forcing (optional, evaluated on GPU each step).
    int32_t* d_cell_gage_idx      = nullptr; // [n_cells]
    int32_t* d_rain_hg_offsets    = nullptr; // [n_rain_gages+1]
    double*  d_rain_hg_time_s     = nullptr; // [n_rain_samples]
    double*  d_rain_hg_cum_mm     = nullptr; // [n_rain_samples]
    double*  d_rain_cn            = nullptr; // [n_cells]
    double*  d_rain_cum_mm        = nullptr; // [n_cells]
    double*  d_rain_excess_cum_mm = nullptr; // [n_cells]
    double*  d_cell_source_mps    = nullptr; // [n_cells]
    double*  d_external_source_mps = nullptr; // [n_cells]
    // Per-stage rain/source snapshots used by graph-safe higher-order schemes.
    // Layout is contiguous by stage: slot*n_cells + cell.
    double*  d_stage_cell_source_mps = nullptr;
    int32_t  n_rain_gages = 0;
    int32_t  n_rain_samples = 0;
    double   rain_ia_ratio = 0.2;
    double   rain_mm_to_model_depth = 1.0e-3;

    // Persistent CUDA stream — all per-step kernel launches and async memsets
    // go on this stream.  Allows CPU-side work (BC updates, Python callbacks)
    // to overlap with GPU execution between steps.
    cudaStream_t d_stream = nullptr;

    // Dimensions
    int32_t  n_cells = 0;
    int32_t  n_edges = 0;

    // CUDA Graph optimization for kernel sequence replay
    // Captures Flux → Update → CFL sequence to reduce launch overhead.
    KernelGraphCache kernel_graph_cache;
    bool             enable_kernel_graphs = false;
    uint64_t         graph_replay_count = 0;   // Diagnostics counter

    // Advanced-mode scaffolding handles.
    SWE3DCartesianPatchDeviceState* patch3d = nullptr;
    SWE2D3DInterfaceContractDevice* coupling_iface = nullptr;

    // Phase 5: Non-hydrostatic pressure workspace (GPU-only; initialized on demand)
    struct NonhydroPressureWorkspace {
        double* d_p = nullptr;
        double* d_p_rhs = nullptr;
        double* d_stencil_diag = nullptr;
        double* d_pcg_r = nullptr;
        double* d_pcg_p = nullptr;
        double* d_pcg_Ap = nullptr;
        double* d_pcg_z = nullptr;
        double* d_precond = nullptr;
        double* d_pcg_rr = nullptr;
        double* d_pcg_rrold = nullptr;
        double* d_pcg_pAp = nullptr;
        double* d_u_corr = nullptr;
        double* d_v_corr = nullptr;
        int matrix_type_last = -1;
        int preconditioner_last = -1;
        bool is_configured = false;
    } nh_workspace{};
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
    double t_now,
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
    bool extreme_rain_mode,
    double source_cfl_beta,
    int source_max_substeps,
    double source_rate_cap,
    double source_depth_step_cap,
    bool source_true_subcycling,
    bool source_imex_split,
    bool enable_shallow_front_recon_fallback,
    bool sync_diagnostics,
    SWE2DStepDiag* diag,
    double front_flux_damping    = 0.5,
    bool   active_set_hysteresis = true);

// Advance one SSPRK2 (Heun) timestep fully on GPU.
void swe2d_gpu_step_rk2(
    SWE2DDeviceState* dev,
    double t_now,
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
    bool extreme_rain_mode,
    double source_cfl_beta,
    int source_max_substeps,
    double source_rate_cap,
    double source_depth_step_cap,
    bool source_true_subcycling,
    bool source_imex_split,
    bool enable_shallow_front_recon_fallback,
    bool sync_diagnostics,
    SWE2DStepDiag* diag,
    double front_flux_damping    = 0.5,
    bool   active_set_hysteresis = true);

// Advance one Godunov rollout timestep on GPU.
// This path enforces the rollout numerics contract (minimum 2nd-order spatial
// reconstruction and shallow-front fallback hardening) while keeping the core
// CUDA kernels shared with the production path.
void swe2d_gpu_step_godunov_rollout(
    SWE2DDeviceState* dev,
    double t_now,
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
    bool extreme_rain_mode,
    double source_cfl_beta,
    int source_max_substeps,
    double source_rate_cap,
    double source_depth_step_cap,
    bool source_true_subcycling,
    bool source_imex_split,
    bool enable_shallow_front_recon_fallback,
    bool sync_diagnostics,
    SWE2DStepDiag* diag,
    double front_flux_damping    = 0.5,
    bool   active_set_hysteresis = true);

// Advance one SSPRK2 Godunov rollout timestep fully on GPU.
void swe2d_gpu_step_rk2_godunov_rollout(
    SWE2DDeviceState* dev,
    double t_now,
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
    bool extreme_rain_mode,
    double source_cfl_beta,
    int source_max_substeps,
    double source_rate_cap,
    double source_depth_step_cap,
    bool source_true_subcycling,
    bool source_imex_split,
    bool enable_shallow_front_recon_fallback,
    bool sync_diagnostics,
    SWE2DStepDiag* diag,
    double front_flux_damping    = 0.5,
    bool   active_set_hysteresis = true);

// Advance one classic RK4 (4th-order Runge-Kutta, composed) timestep fully on GPU.
// temporal_order=4. Composed method using 4×swe2d_gpu_step calls.
void swe2d_gpu_step_rk4(
    SWE2DDeviceState* dev,
    double t_now,
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
    bool extreme_rain_mode,
    double source_cfl_beta,
    int source_max_substeps,
    double source_rate_cap,
    double source_depth_step_cap,
    bool source_true_subcycling,
    bool source_imex_split,
    bool enable_shallow_front_recon_fallback,
    bool sync_diagnostics,
    SWE2DStepDiag* diag,
    double front_flux_damping    = 0.5,
    bool   active_set_hysteresis = true);

// Advance one graph-safe true RK4 timestep fully on GPU.
// temporal_order=5. Pure Butcher-tableau RK4 with separate L(U) evaluations per stage.
// Classify runs once per step (outside graph); gradient+flux+rhs_collect+stage_build
// across all 4 stages is captured as a single CUDA graph (time_integrator=5).
void swe2d_gpu_step_rk4_graph(
    SWE2DDeviceState* dev,
    double t_now,
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
    bool extreme_rain_mode,
    double source_cfl_beta,
    int source_max_substeps,
    double source_rate_cap,
    double source_depth_step_cap,
    bool source_true_subcycling,
    bool source_imex_split,
    bool enable_shallow_front_recon_fallback,
    bool sync_diagnostics,
    SWE2DStepDiag* diag,
    double front_flux_damping    = 0.5,
    bool   active_set_hysteresis = true);

// Advance one graph-safe RK5 timestep fully on GPU.
// temporal_order=6. Cash-Karp 5th-order explicit RK with stage forcing snapshots
// prepared outside the graph and consumed inside the captured stage sequence.
void swe2d_gpu_step_rk5_graph(
    SWE2DDeviceState* dev,
    double t_now,
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
    bool extreme_rain_mode,
    double source_cfl_beta,
    int source_max_substeps,
    double source_rate_cap,
    double source_depth_step_cap,
    bool source_true_subcycling,
    bool source_imex_split,
    bool enable_shallow_front_recon_fallback,
    bool sync_diagnostics,
    SWE2DStepDiag* diag,
    double front_flux_damping    = 0.5,
    bool   active_set_hysteresis = true);

// Advance one 3D single-phase free-surface scaffold timestep on GPU.
// This advances the 3D patch state and optionally applies 2D-3D exchange
// scaffolding when an interface contract is uploaded.
void swe2d_gpu_step_3d_single_phase_free_surface(
    SWE2DDeviceState* dev,
    double t_now,
    double dt,
    double g,
    bool enable_coupling_exchange,
    bool sync_diagnostics,
    SWE2DStepDiag* diag);

void swe2d_gpu_step_nonhydro_predictor_corrector(
    SWE2DDeviceState* dev,
    double t_now,
    double dt,
    double g,
    double h_min,
    int spatial_scheme,
    const SWE2DNonhydroPcConfig& nh_cfg,
    bool sync_diagnostics,
    SWE2DStepDiag* diag,
    SWE2DNonhydroPcDiag* nh_diag,
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

// Push updated boundary type/value arrays to device for selected edges.
void swe2d_gpu_update_boundary_values(
    SWE2DDeviceState* dev,
    const int32_t* edge_index,
    const int32_t* bc_type,
    const double* bc_val,
    int32_t n_updates);

// Upload per-edge hydrograph forcing arrays.
void swe2d_gpu_set_boundary_hydrographs(
    SWE2DDeviceState* dev,
    const int32_t* edge_index,
    const int32_t* bc_type,
    const int32_t* offsets,
    const double* time_s,
    const double* value,
    int32_t n_edges,
    int32_t n_samples);

// Upload per-cell rain+CN forcing arrays.
void swe2d_gpu_set_rain_cn_forcing(
    SWE2DDeviceState* dev,
    const int32_t* cell_gage_idx,
    const int32_t* gage_offsets,
    const double* hg_time_s,
    const double* hg_cum_mm,
    const double* cn,
    int32_t n_cells,
    int32_t n_gages,
    int32_t n_samples,
    double ia_ratio,
    double mm_to_model_depth);

// Upload per-cell external source terms [m/s] used by the GPU step update.
// Passing nullptr clears external sources on the device.
void swe2d_gpu_set_external_sources(
    SWE2DDeviceState* dev,
    const double* source_mps,
    int32_t n_cells);

// Headless coupling helper: compute per-cell depth-rate sources [m/s] from
// packed drainage/structure transfer arrays using CUDA kernels.
void swe2d_gpu_compute_coupling_sources(
    int32_t n_cells,
    const double* cell_area_m2,
    int32_t n_inlets,
    const int32_t* inlet_cell,
    const double* inlet_flow_cms,
    int32_t n_structures,
    const int32_t* structure_up_cell,
    const int32_t* structure_down_cell,
    const double* structure_flow_cms,
    double* source_rate_mps_out);

SWE3DCartesianPatchDeviceState* swe3d_cartesian_patch_alloc(
    const SWE3DCartesianPatchDesc& desc);

void swe3d_cartesian_patch_zero_state(
    SWE3DCartesianPatchDeviceState* patch,
    cudaStream_t stream);

void swe3d_cartesian_patch_release(
    SWE3DCartesianPatchDeviceState* patch);

void swe2d_gpu_set_2d3d_interface_contract(
    SWE2DDeviceState* dev,
    const int32_t* cell2d,
    const double* face_area,
    const double* face_nx,
    const double* face_ny,
    const double* face_nz,
    int32_t n_faces);

void swe2d_gpu_clear_2d3d_interface_contract(
    SWE2DDeviceState* dev);

// Phase 7: Contract API for pybind11 exposure
// Allocate and upload contract geometry + flux/head-loss buffers to GPU.
// Returns true on success; false on allocation failure.
// Pre-zeros flux and head-loss output buffers to avoid garbage values.
bool swe2d_gpu_contract_upload(
    SWE2DDeviceState* dev,
    const SWE2D3DInterfaceContractHost& contract);

// Free device-side contract buffers (flux, head-loss, etc).
// Safe to call even if contract was never uploaded (null check internally).
void swe2d_gpu_contract_free(
    SWE2DDeviceState* dev);

// Query: is a device contract currently uploaded?
bool swe2d_gpu_is_contract_uploaded(const SWE2DDeviceState* dev);

void swe2d_gpu_apply_2d3d_coupling_exchange_scaffold(
    SWE2DDeviceState* dev,
    double dt,
    bool one_way_2d_to_3d,
    SWE2DStepDiag* diag);

// Phase 5: Pressure workspace lifecycle
bool swe2d_gpu_allocate_pressure_workspace(
    SWE2DDeviceState* dev,
    int32_t n_cells);

void swe2d_gpu_deallocate_pressure_workspace(
    SWE2DDeviceState* dev);

// Phase 6: 2D-3D exchange kernel scaffold
__global__ void swe3d_exchange_kernel_skeleton(
    int32_t n_faces,
    const int32_t* d_cell2d,
    const double* d_h_2d,
    const double* d_hu_2d,
    const double* d_hv_2d,
    const double* d_cell_area_2d,
    const double* d_face_area,
    const double* d_face_nx,
    const double* d_face_ny,
    const double* d_face_nz,
    const double* d_u_3d,
    const double* d_p_3d,
    double* d_flux_mass_2d_to_3d,
    double* d_flux_momx_2d_to_3d,
    double* d_flux_momy_2d_to_3d,
    double* d_head_loss_3d_to2d,
    double g,
    double dt);

void swe2d_gpu_apply_2d3d_exchange_skeleton(
    SWE2DDeviceState* dev,
    double dt,
    double g,
    bool apply_head_loss_to_2d_rhs,
    SWE2DStepDiag* diag);

// ─────────────────────────────────────────────────────────────────────────────
// Phase 8A: Pressure RHS & Laplacian Stencil Kernels
// ─────────────────────────────────────────────────────────────────────────────

// Compute pressure right-hand side from velocity divergence.
// RHS_i = -(1/dt) * Σ_edges [ flux_out_normal_k ]
// Stores result in d_p_rhs (device workspace buffer).
bool swe2d_gpu_compute_pressure_rhs(
    SWE2DDeviceState* dev,
    double dt,
    double g);

// Matrix-free Laplacian evaluation and diagonal extraction.
// Computes (A*p) using 5-point stencil on triangular mesh.
// Also pre-computes Laplacian diagonal for Jacobi preconditioner.
// Returns true on success.
bool swe2d_gpu_laplacian_matrix_free(
    SWE2DDeviceState* dev);

// ─────────────────────────────────────────────────────────────────────────────

// Headless coupling helper: advance 1D drainage state by one step on GPU and
// return per-cell surface source flows [m3/s] (positive to 2D, negative from 2D).
// solver_mode: 0=EGL, 1=DIFFUSION, 2=DYNAMIC.
void swe2d_gpu_drainage_step(
    int32_t n_cells,
    int32_t n_nodes,
    int32_t n_links,
    int32_t n_inlets,
    int32_t n_outfalls,
    int32_t n_pipe_ends,
    const double* cell_wse,
    const double* cell_area,
    const double* node_invert_elev,
    const double* node_max_depth,
    const double* node_surface_area,
    const int32_t* link_from,
    const int32_t* link_to,
    const double* link_length,
    const double* link_roughness_n,
    const double* link_diameter,
    const double* link_max_flow,
    const int32_t* inlet_cell,
    const int32_t* inlet_node,
    const double* inlet_crest_elev,
    const double* inlet_width,
    const double* inlet_coefficient,
    const double* inlet_max_capture,
    const int32_t* outfall_cell,
    const int32_t* outfall_node,
    const double* outfall_invert_elev,
    const double* outfall_diameter,
    const double* outfall_coefficient,
    const double* outfall_max_flow,
    const int32_t* outfall_zero_storage,
    const int32_t* pipe_end_cell,
    const int32_t* pipe_end_node,
    const double* pipe_end_invert_elev,
    const double* pipe_end_diameter,
    const double* pipe_end_area,
    const double* pipe_end_inlet_loss_k,
    const double* pipe_end_outlet_loss_k,
    const double* cell_depth,
    const double* node_depth_in,
    const double* link_flow_in,
    double dt_s,
    double gravity,
    int32_t solver_mode,
    double head_deadband_m,
    double dynamic_flow_relaxation,
    double* node_depth_out,
    double* link_flow_out,
    double* q_cell_out,
    double* max_node_depth_out,
    double* max_link_flow_out,
    double* limiter_event_count_out,
    double* limiter_volume_m3_out);

// ─────────────────────────────────────────────────────────────────────────────
// 3D patch state observation and initialisation (validation / testing API)
// ─────────────────────────────────────────────────────────────────────────────

// Aggregate statistics over all cells in the 3D Cartesian patch.
// Fields are copied from device to host and reduced on the CPU — suitable
// for validation tests, not for inner-loop production use.
struct SWE3DPatchStats {
    int64_t n_cells = 0;
    // VoF
    double vof_min = 0.0;
    double vof_max = 0.0;
    double vof_sum = 0.0;       // total VoF (conserved if no source/sink)
    // Velocity RMS (sqrt(mean(u^2)))
    double u_rms = 0.0;
    double v_rms = 0.0;
    double w_rms = 0.0;
    // Pressure extrema
    double p_max_abs = 0.0;
    // Velocity divergence RMS over patch (from cell-centered finite differences)
    double divergence_rms = 0.0;
    // Last projection diagnostics
    int32_t projection_iters = 0;
    double projection_residual = -1.0;
    bool projection_converged = false;
    // Patch descriptor (for assertions)
    int32_t nx = 0;
    int32_t ny = 0;
    int32_t nz = 0;
    double dx = 0.0;
    double dy = 0.0;
    double dz = 0.0;
};

// Synchronise and collect stats from the 3D patch attached to dev.
// Throws if dev or dev->patch3d is null.
SWE3DPatchStats swe2d_gpu_get_3d_patch_stats(SWE2DDeviceState* dev);

// Upload a full per-cell VoF field (host → device, length must equal n_cells).
// Validates length against patch.  Throws on mismatch.
void swe2d_gpu_set_3d_patch_vof(
    SWE2DDeviceState* dev,
    const double*     vof_host,
    int64_t           n);

// Upload full per-cell velocity+pressure initial condition (all optional).
// Pass nullptr for any field to skip that field.  Length must equal n_cells.
void swe2d_gpu_set_3d_patch_state(
    SWE2DDeviceState* dev,
    const double* u_host,    // nullable
    const double* v_host,    // nullable
    const double* w_host,    // nullable
    const double* p_host,    // nullable
    const double* vof_host,  // nullable
    int64_t n);

// CUDA Graph optimization API (Suggestion 9)
// Enable graph capture on next step, and use replayed graphs on subsequent steps.
void swe2d_gpu_enable_kernel_graphs(SWE2DDeviceState* dev, bool enable);

// Manually destroy cached graph (called at cleanup or on config change).
void swe2d_gpu_destroy_kernel_graphs(SWE2DDeviceState* dev);

// Free all device memory.
void swe2d_gpu_destroy(SWE2DDeviceState* dev);

// Query: returns true if a CUDA-capable device is available.
bool swe2d_gpu_available();
