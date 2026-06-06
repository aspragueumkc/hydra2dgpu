#pragma once
// swe2d_gpu.cuh
// CUDA device state and host API declarations for the 2D SWE GPU path.
// Included only when HYDRA_HAS_CUDA is defined.

#include "swe2d_mesh.hpp"
#include "swe2d_solver.hpp"   // SWE2DStepDiag
#include "swe2d_units.cuh"

#include <cuda_runtime.h>
#include <cstdint>

// ─────────────────────────────────────────────────────────────────────────────
// CUDA Graph cache for optimized kernel sequence replay
// ─────────────────────────────────────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────────
// CUDA Graph cache for optimized kernel sequence replay
// ─────────────────────────────────────────────────────────────────────────────

struct KernelGraphCache {
    cudaGraph_t       graph = nullptr;       // Captured graph template
    cudaGraphExec_t   exec = nullptr;        // Executable instance for replay
    int32_t           n_cells = 0;           // Mesh size at capture time
    int32_t           n_edges = 0;           // Edge count at capture time
    int32_t           spatial_scheme = 0;    // Spatial scheme at capture
    int32_t           time_integrator = 0;   // RK order (2/4/5/6) at capture
    int32_t           variant_key = 0;       // Encodes has_hydrograph + need_gradient
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

enum class SWE3DPatchBoundaryFace : int32_t {
    XMIN = 0,
    XMAX = 1,
    YMIN = 2,
    YMAX = 3,
    ZMIN = 4,
    ZMAX = 5,
};

enum class SWE3DBoundaryMode : int32_t {
    WALL = 0,
    INFLOW = 1,
    OUTFLOW = 2,
    FREE_SURFACE = 3,
    INFLOW_FLOW_RATE = 4,
};

constexpr int32_t SWE3D_PATCH_FACE_COUNT = 6;

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
    // Vertical gravity sign convention in z-momentum source term.
    // Use -1.0 for z-up coordinates (gravity acts downward).
    double gravity_z_sign = -1.0;
    // Near-bed drag controls (Manning-equivalent quadratic drag on u/v).
    // If enable_bed_drag is true, drag is applied in the lowest bed_drag_layers
    // layers with coefficient g*n^2 / h_ref^(4/3).
    bool enable_bed_drag = true;
    double bed_manning_n = 0.03;
    double bed_drag_h_ref = 0.0; // <=0 uses dz as reference depth scale
    int32_t bed_drag_layers = 1;
    // Per-boundary-face BC mode and prescribed state.
    // Arrays index by SWE3DPatchBoundaryFace.
    int32_t bc_mode[SWE3D_PATCH_FACE_COUNT] = {
        static_cast<int32_t>(SWE3DBoundaryMode::WALL),
        static_cast<int32_t>(SWE3DBoundaryMode::WALL),
        static_cast<int32_t>(SWE3DBoundaryMode::WALL),
        static_cast<int32_t>(SWE3DBoundaryMode::WALL),
        static_cast<int32_t>(SWE3DBoundaryMode::WALL),
        static_cast<int32_t>(SWE3DBoundaryMode::WALL),
    };
    double bc_u[SWE3D_PATCH_FACE_COUNT] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double bc_v[SWE3D_PATCH_FACE_COUNT] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double bc_w[SWE3D_PATCH_FACE_COUNT] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double bc_vof[SWE3D_PATCH_FACE_COUNT] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    double bc_p[SWE3D_PATCH_FACE_COUNT] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    // Volumetric flow rate (m^3/s), consumed when mode is INFLOW_FLOW_RATE.
    double bc_q[SWE3D_PATCH_FACE_COUNT] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
};

struct SWE3DCartesianPatchDeviceState {
    SWE3DCartesianPatchDesc desc;
    int64_t n_cells = 0;
    double* d_u = nullptr;
    double* d_v = nullptr;
    double* d_w = nullptr;
    double* d_p = nullptr;
    double* d_vof = nullptr;
    double* d_vof_tmp = nullptr;
    double* d_vof_sum = nullptr;
    // Static geometry tensors for sub-grid solid occupancy and face openness.
    // Defaults are all-ones (fully open fluid cells/faces).
    double* d_phi = nullptr;  // cell fluid fraction [0..1]
    double* d_ax = nullptr;   // x-face open-area fraction [0..1]
    double* d_ay = nullptr;   // y-face open-area fraction [0..1]
    double* d_az = nullptr;   // z-face open-area fraction [0..1]
    // 3D projection workspace (uncoupled path): pressure RHS + Jacobi scratch.
    double* d_p_rhs = nullptr;
    double* d_p_tmp = nullptr;
    unsigned long long* d_proj_residual_bits = nullptr;
    // Per-face boundary open area reduction for volumetric inlet BC conversion.
    double* d_bc_face_open_area = nullptr; // SWE3D_PATCH_FACE_COUNT entries
    // Column-integrated liquid depth workspace (nx*ny entries) used for
    // hydrostatic head-gradient forcing in horizontal predictor components.
    double* d_column_depth = nullptr;
    // Last projection diagnostics from swe2d_gpu_step_3d_single_phase_free_surface.
    int32_t last_projection_iters = 0;
    double last_projection_residual = -1.0;
    bool last_projection_converged = false;
    int32_t last_vof_substeps = 1;
    // Active-set mask (1=solve this cell in predictor/projection/correction).
    uint8_t* d_active_mask = nullptr;
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

    // 2-ring cell stencil (CSR), used by the least-squares gradient (scheme 6).
    int32_t* d_cell_ring2_offsets   = nullptr;  // [n_cells + 1]
    int32_t* d_cell_ring2_ids       = nullptr;  // [sum(ring2_counts)]
    double*  d_cell_ring2_dcx       = nullptr;  // [sum(ring2_counts)]
    double*  d_cell_ring2_dcy       = nullptr;  // [sum(ring2_counts)]
    double*  d_cell_ring2_inv_dist2 = nullptr;  // [sum(ring2_counts)]
    int32_t  n_cell_ring2           = 0;        // length of ring2 id/Δ arrays

    // Per-edge hydrograph forcing (optional, evaluated on GPU each step).
    int32_t* d_hg_edge_index = nullptr;   // [n_hg_edges]
    int32_t* d_hg_bc_type = nullptr;      // [n_hg_edges]
    int32_t* d_hg_offsets = nullptr;      // [n_hg_edges+1]
    double*  d_hg_time_s = nullptr;       // [n_hg_samples]
    double*  d_hg_value = nullptr;        // [n_hg_samples]
    int32_t  n_hg_edges = 0;
    int32_t  n_hg_samples = 0;

    // Reusable upload buffers for per-step boundary value updates.
    // Capacity is in element count, not bytes.
    int32_t* d_bc_upd_edge = nullptr;
    int32_t* d_bc_upd_type = nullptr;
    double*  d_bc_upd_val = nullptr;
    int32_t  bc_upd_capacity = 0;

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
    // Two-level CFL reduction: block maxima are written here by swe2d_cfl_kernel,
    // then a lightweight second kernel reduces them to d_lambda_max.
    double*  d_cfl_block_max = nullptr;   // [grid_size] for CFL reduction
    int32_t  cfl_block_capacity = 0;      // allocated length of d_cfl_block_max
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

    // Optional active-edge compaction workspace for tiny persistent stepping.
    // d_active_edge_ids[k] stores edge indices selected from d_active mask.
    int32_t* d_active_edge_ids = nullptr; // n_edges
    int32_t* d_n_active_edges = nullptr;  // device scalar

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

    // Persistent coupling workspace: reused across coupling calls to eliminate
    // per-call cudaMalloc/cudaFree and H→D re-upload when data is unchanged.
    // Allocated lazily on first use; survives for the lifetime of the device state.
    struct CouplingWorkspace {
        int32_t  cell_capacity = 0;
        double*  d_cell_area = nullptr;
        double*  d_source = nullptr;
        int32_t  inlet_capacity = 0;
        int32_t* d_inlet_cell = nullptr;
        double*  d_inlet_q = nullptr;
        int32_t  structure_capacity = 0;
        int32_t* d_struct_up = nullptr;
        int32_t* d_struct_dn = nullptr;
        double*  d_struct_q = nullptr;
        int32_t  bridge_cell_capacity = 0;
        double*  d_bridge_cell_area = nullptr;
        double*  d_bridge_source = nullptr;
        int32_t  bridge_capacity = 0;
        int32_t* d_bridge_up = nullptr;
        int32_t* d_bridge_dn = nullptr;
        double*  d_bridge_q = nullptr;
        double*  d_bridge_ku = nullptr;
        double*  d_bridge_kd = nullptr;
        // Content hashes for dirtiness tracking (skip re-upload if unchanged).
        uint64_t inlet_data_hash = 0;
        uint64_t structure_data_hash = 0;
        uint64_t bridge_data_hash = 0;
    } coupling_ws{};

    // Persistent structure-flow workspace: caches all device buffers for the
    // 33-parameter structure flow kernel, eliminating per-step cudaMalloc churn.
    struct StructureFlowWorkspace {
        bool     params_preloaded = false;
        int32_t  n_structures = 0;
        int32_t  cell_capacity = 0;
        int32_t  struct_capacity = 0;
        double   gravity = 9.81;
        double   model_to_ft = 3.28084;
        double*  d_cell_wse = nullptr;
        double*  d_cell_bed = nullptr;
        int32_t* d_structure_type = nullptr;
        int32_t* d_upstream_cell = nullptr;
        int32_t* d_downstream_cell = nullptr;
        double*  d_crest_elev = nullptr;
        double*  d_width = nullptr;
        double*  d_height = nullptr;
        double*  d_diameter = nullptr;
        double*  d_length = nullptr;
        double*  d_roughness_n = nullptr;
        double*  d_coeff = nullptr;
        double*  d_cd = nullptr;
        double*  d_opening = nullptr;
        double*  d_q_pump = nullptr;
        double*  d_max_flow = nullptr;
        int32_t* d_culvert_code = nullptr;
        int32_t* d_culvert_shape = nullptr;
        double*  d_culvert_rise = nullptr;
        double*  d_culvert_span = nullptr;
        double*  d_culvert_area = nullptr;
        double*  d_culvert_barrels = nullptr;
        double*  d_culvert_slope = nullptr;
        double*  d_inlet_invert_elev = nullptr;
        double*  d_outlet_invert_elev = nullptr;
        double*  d_entrance_loss_k = nullptr;
        double*  d_exit_loss_k = nullptr;
        int32_t* d_embankment_enabled = nullptr;
        double*  d_embankment_crest_elev = nullptr;
        double*  d_embankment_overflow_width = nullptr;
        double*  d_embankment_weir_coeff = nullptr;
        double*  d_structure_flow = nullptr;
    } sf_ws{};

    // Persistent redistribution workspace: caches all redistribution
    // geometry arrays on-device, eliminating per-step cudaMalloc/free churn.
    // Static data (offsets, cell_idx, weights, up/down cells) is uploaded
    // once via content-hash tracking; flow values are re-uploaded each
    // step but are tiny (n_structures * 8 bytes).
    struct RedistWorkspace {
        int32_t  n_struct_capacity = 0;
        int32_t  dist_cell_capacity = 0;
        uint64_t data_hash = 0;
        int32_t* d_offsets = nullptr;    // [n_struct + 1]
        int32_t* d_cell_idx = nullptr;   // [total_dist_cells]
        double*  d_weights = nullptr;    // [total_dist_cells]
        int32_t* d_up = nullptr;         // [n_struct]
        int32_t* d_dn = nullptr;         // [n_struct]

        void destroy() {
            if (d_offsets) { cudaFree(d_offsets); d_offsets = nullptr; }
            if (d_cell_idx) { cudaFree(d_cell_idx); d_cell_idx = nullptr; }
            if (d_weights) { cudaFree(d_weights); d_weights = nullptr; }
            if (d_up) { cudaFree(d_up); d_up = nullptr; }
            if (d_dn) { cudaFree(d_dn); d_dn = nullptr; }
            n_struct_capacity = 0;
            dist_cell_capacity = 0;
            data_hash = 0;
        }
    } redist_ws{};

    // ── Face-based culvert coupling workspace ─────────────────────────
    // When culvert_face_flux_mode == "face_flux", culvert flows are applied
    // as proper FVM face fluxes (mass + momentum) instead of cell-center
    // source/sink terms.  This preserves strict mass conservation and
    // momentum balance.
    struct CulvertFaceFluxWorkspace {
        bool     params_preloaded = false;
        int32_t  n_culvert_faces = 0;
        int32_t  face_capacity = 0;
        int32_t  n_struct_flows_capacity = 0;

        // Culvert index into the full structure arrays (for reading Q_c)
        int32_t* d_culvert_struct_idx = nullptr;  // [n_culvert_faces]
        // Face geometry
        double*  d_face_nx = nullptr;              // [n_culvert_faces]
        double*  d_face_ny = nullptr;              // [n_culvert_faces]
        double*  d_face_width = nullptr;           // [n_culvert_faces]
        // Donor / receiver cell topology
        int32_t* d_donor_cell = nullptr;           // [n_culvert_faces]
        int32_t* d_receiver_cell = nullptr;        // [n_culvert_faces]
        // Invert elevation for depth limiting
        double*  d_invert_elev = nullptr;          // [n_culvert_faces]
        double*  d_depth_safety = nullptr;          // [n_culvert_faces]

        void destroy() {
            if (d_culvert_struct_idx) { cudaFree(d_culvert_struct_idx); d_culvert_struct_idx = nullptr; }
            if (d_face_nx) { cudaFree(d_face_nx); d_face_nx = nullptr; }
            if (d_face_ny) { cudaFree(d_face_ny); d_face_ny = nullptr; }
            if (d_face_width) { cudaFree(d_face_width); d_face_width = nullptr; }
            if (d_donor_cell) { cudaFree(d_donor_cell); d_donor_cell = nullptr; }
            if (d_receiver_cell) { cudaFree(d_receiver_cell); d_receiver_cell = nullptr; }
            if (d_invert_elev) { cudaFree(d_invert_elev); d_invert_elev = nullptr; }
            if (d_depth_safety) { cudaFree(d_depth_safety); d_depth_safety = nullptr; }
            n_culvert_faces = 0;
            face_capacity = 0;
            n_struct_flows_capacity = 0;
            params_preloaded = false;
        }
    } culvert_ff_ws{};

    // Per-cell external structure flux accumulators for face-based culvert coupling.
    // Written by swe2d_culvert_face_flux_kernel, consumed by swe2d_update_kernel.
    // Zeroed each step before the face-flux kernel runs.
    double*  d_ext_struct_flux_h  = nullptr;   // [n_cells] net mass flux (L²·L/T = L³/T)
    double*  d_ext_struct_flux_hu = nullptr;   // [n_cells] net x-momentum flux
    double*  d_ext_struct_flux_hv = nullptr;   // [n_cells] net y-momentum flux

    // Toggle: when true, swe2d_update_kernel reads d_ext_struct_flux_* instead
    // of applying external_source_mps for culvert mass transfers.
    bool     use_culvert_face_flux = false;
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

// Set the Manning unit-conversion factor in GPU constant memory.
// Call once after swe2d_gpu_init and before any step call.
//   k_mann = 1.0   for SI (meters)
//   k_mann = 1.486 for US Customary (feet)
void swe2d_gpu_set_k_mann(double k_mann);

// Set friction temporal-order hardening and shallow-correction params
// in GPU constant memory.  Call once after swe2d_gpu_init and before
// any step call.
void swe2d_gpu_set_friction_config(
    bool   substep_enabled,
    double target_courant,
    int    max_substeps,
    bool   shallow_correction,
    double depth_alpha,
    double exponent);

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

// Persistent cooperative-kernel chunk stepping for tiny-N runs.
// Executes chunk_substeps internal substeps with dt/chunk_substeps each.
// This path is currently constrained to first-order single-stage hydrostatic
// stepping and falls back to baseline stepping when unsupported.
void swe2d_gpu_step_persistent_chunk(
    SWE2DDeviceState* dev,
    double t_now,
    double dt,
    int chunk_substeps,
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
    bool enable_active_edge_compaction,
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

// Advance one SSPRK2 (Heun) timestep using persistent chunk stepping for
// each RK stage when supported.
void swe2d_gpu_step_rk2_persistent_chunk(
    SWE2DDeviceState* dev,
    double t_now,
    double dt,
    int chunk_substeps,
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
    bool enable_active_edge_compaction,
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

// Advance one graph-safe RK4 timestep using chunked persistent execution.
// Current implementation executes chunk_substeps RK4-graph substeps with
// dt/chunk_substeps each and syncs diagnostics on the final substep.
void swe2d_gpu_step_rk4_graph_persistent_chunk(
    SWE2DDeviceState* dev,
    double t_now,
    double dt,
    int chunk_substeps,
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
    int coupling_mode,
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

// Compute a CFL-limited dt from current 3D patch state.
// Uses velocity, directional face openness, and open-volume fraction to keep
// adaptive stepping stable in fractional cut-cells. Returns dt_max when patch
// is unavailable or effective CFL wave speed is near zero.
double swe2d_gpu_compute_dt_3d_patch(
    SWE2DDeviceState* dev,
    double g,
    double cfl_factor,
    double dt_max);

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
// When dev is non-null, uses persistent device buffers and dev->d_stream
// for async execution; falls back to synchronous static-cache path otherwise.
void swe2d_gpu_compute_coupling_sources(
    SWE2DDeviceState* dev,   // nullable: enables persistent workspace + async stream
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

// Bridge-specific source helper: apply a bridge loss law to structure flows
// before converting them to per-cell depth-rate sources [m/s].
void swe2d_gpu_compute_bridge_coupling_sources(
    SWE2DDeviceState* dev,   // nullable: enables persistent workspace + async stream
    int32_t n_cells,
    const double* cell_area_m2,
    int32_t n_bridges,
    const int32_t* bridge_up_cell,
    const int32_t* bridge_down_cell,
    const double* bridge_flow_cms,
    const double* bridge_loss_k_upstream,
    const double* bridge_loss_k_downstream,
    double bridge_opening_width_m,
    double dt_s,
    double* source_rate_mps_out);

// Redistribution: after coupling sources have been computed with single-cell
// injection, redistribute flow across a pre-computed corridor of cells for
// structures with influence_width > 0.  Pre-computed weights are stored as
// flat arrays with per-structure offsets.
void swe2d_gpu_redistribute_structure_sources(
    SWE2DDeviceState* dev,   // nullable
    int32_t n_structures,
    const double* structure_flow_cms,
    const int32_t* orig_up_cell,
    const int32_t* orig_dn_cell,
    const double* cell_area_m2,
    const int32_t* dist_offsets,
    const int32_t* dist_cell_idx,
    const double* dist_weights,
    int32_t n_cells,
    double* source_rate_mps_inout);

// On-device-only redistribution (no host readback of source array).
// Operates directly on dev->d_external_source_mps.  Call this after
// swe2d_gpu_compute_coupling_full_on_device, then skip the coupling
// readback  (return None from Python to keep GPU sources current).
void swe2d_gpu_redistribute_structure_sources_persistent(
    SWE2DDeviceState* dev,
    int32_t n_structures,
    const double* structure_flow_cms,
    const int32_t* orig_up_cell,
    const int32_t* orig_dn_cell,
    const int32_t* dist_offsets,
    const int32_t* dist_cell_idx,
    const double*  dist_weights,
    int32_t n_cells,
    double si_m_per_model_factor);

// Structure-flow helper: compute per-structure transfer flow [m^3/s] on CUDA.
void swe2d_gpu_compute_structure_flows(
    int32_t n_cells,
    int32_t n_structures,
    const double* cell_wse,
    const double* cell_bed,
    const int32_t* structure_type,
    const int32_t* upstream_cell,
    const int32_t* downstream_cell,
    const double* crest_elev,
    const double* width,
    const double* height,
    const double* diameter,
    const double* length,
    const double* roughness_n,
    const double* coeff,
    const double* cd,
    const double* opening,
    const double* q_pump,
    const double* max_flow,
    const int32_t* culvert_code,
    const int32_t* culvert_shape,
    const double* culvert_rise,
    const double* culvert_span,
    const double* culvert_area_m2,
    const double* culvert_barrels,
    const double* culvert_slope,
    const double* inlet_invert_elev,
    const double* outlet_invert_elev,
    const double* entrance_loss_k,
    const double* exit_loss_k,
    const int32_t* embankment_enabled,
    const double* embankment_crest_elev,
    const double* embankment_overflow_width,
    const double* embankment_weir_coeff,
    double gravity,
    double model_to_ft,
    double* structure_flow_out);

// Fused structure-flows + coupling-sources: runs both kernels on-device and
// returns per-cell source rates [m/s] without the intermediate H→D→H round trip.
void swe2d_gpu_compute_structure_and_coupling_sources(
    int32_t n_cells,
    const double* cell_area_m2,
    int32_t n_structures,
    const double* cell_wse,
    const double* cell_bed,
    const int32_t* structure_type,
    const int32_t* upstream_cell,
    const int32_t* downstream_cell,
    const double* crest_elev,
    const double* width,
    const double* height,
    const double* diameter,
    const double* length,
    const double* roughness_n,
    const double* coeff,
    const double* cd,
    const double* opening,
    const double* q_pump,
    const double* max_flow,
    const int32_t* culvert_code,
    const int32_t* culvert_shape,
    const double* culvert_rise,
    const double* culvert_span,
    const double* culvert_area_m2,
    const double* culvert_barrels,
    const double* culvert_slope,
    const double* inlet_invert_elev,
    const double* outlet_invert_elev,
    const double* entrance_loss_k,
    const double* exit_loss_k,
    const int32_t* embankment_enabled,
    const double* embankment_crest_elev,
    const double* embankment_overflow_width,
    const double* embankment_weir_coeff,
    double gravity,
    double model_to_ft,
    int32_t n_inlets,
    const int32_t* inlet_cell,
    const double* inlet_flow_cms,
    double* source_rate_mps_out);

// ── Persistent GPU coupling path ──
void swe2d_gpu_set_coupling_device_global(SWE2DDeviceState* dev);
void swe2d_gpu_preload_structure_params(
    SWE2DDeviceState* dev, int32_t n_structures,
    const int32_t* structure_type, const int32_t* upstream_cell, const int32_t* downstream_cell,
    const double* crest_elev, const double* width, const double* height,
    const double* diameter, const double* length, const double* roughness_n,
    const double* coeff, const double* cd, const double* opening,
    const double* q_pump, const double* max_flow,
    const int32_t* culvert_code, const int32_t* culvert_shape,
    const double* culvert_rise, const double* culvert_span, const double* culvert_area_m2,
    const double* culvert_barrels, const double* culvert_slope,
    const double* inlet_invert_elev, const double* outlet_invert_elev,
    const double* entrance_loss_k, const double* exit_loss_k,
    const int32_t* embankment_enabled, const double* embankment_crest_elev,
    const double* embankment_overflow_width, const double* embankment_weir_coeff,
    double gravity, double model_to_ft);
void swe2d_gpu_preload_coupling_cell_area(SWE2DDeviceState* dev, int32_t n_cells, const double* cell_area_m2);
void swe2d_gpu_compute_coupling_full_on_device(
    SWE2DDeviceState* dev, int32_t n_cells, int32_t n_structures, const double* cell_wse_host,
    int32_t n_inlets, const int32_t* inlet_cell, const double* inlet_flow_cms);
void swe2d_gpu_readback_coupling_sources(double* host_buf, int32_t n_cells);
void swe2d_gpu_readback_structure_flows(double* host_buf, int32_t n_structures);

// ── Face-based culvert flux coupling ───────────────────────────────────────
// Upload culvert face-flux geometry (face normals, widths, donor/receiver
// cells, invert elevations) to the GPU.  Called once when the mesh or
// structure configuration changes.
void swe2d_gpu_upload_culvert_face_flux_params(
    SWE2DDeviceState* dev,
    int32_t n_culvert_faces,
    const int32_t* culvert_struct_idx,
    const double*  face_nx,
    const double*  face_ny,
    const double*  face_width,
    const int32_t* donor_cell,
    const int32_t* receiver_cell,
    const double*  invert_elev,
    const double*  depth_safety,
    bool use_face_flux);

// Compute per-cell structure flows (Q_c) on device, then apply face-based
// culvert fluxes (mass + momentum) into d_ext_struct_flux_h/hu/hv and
// zero out culvert flows from the source-kernel path.
void swe2d_gpu_apply_culvert_face_flux(
    SWE2DDeviceState* dev,
    double dt,
    double h_min);

// Allocate and zero the per-cell external flux accumulators on device.
void swe2d_gpu_alloc_ext_struct_flux(SWE2DDeviceState* dev, int32_t n_cells);

// Read back per-cell external structure flux arrays from device (for debug).
void swe2d_gpu_readback_ext_struct_flux(
    double* host_h, double* host_hu, double* host_hv, int32_t n_cells);

// Set the coupling time step (used by face-flux depth limiter).
void swe2d_gpu_set_coupling_dt(double dt);

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
    int coupling_mode,
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
    // Last VoF transport substep count (CFL-limited).
    int32_t vof_transport_substeps = 1;
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

// Download full per-cell VoF field (device → host, length must equal n_cells).
// Validates length against patch. Throws on mismatch.
void swe2d_gpu_get_3d_patch_vof(
    SWE2DDeviceState* dev,
    double*           vof_host,
    int64_t           n);

// Download full per-cell velocity fields (device -> host, length == n_cells).
// Any output pointer may be nullptr to skip that component.
void swe2d_gpu_get_3d_patch_velocity(
    SWE2DDeviceState* dev,
    double*           u_host,
    double*           v_host,
    double*           w_host,
    int64_t           n);

// Download full per-cell pressure field (device -> host, length == n_cells).
void swe2d_gpu_get_3d_patch_pressure(
    SWE2DDeviceState* dev,
    double*           p_host,
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

// Upload per-cell/facet static geometry tensors for the 3D Cartesian patch.
// Pass nullptr for any field to keep its current device values.
void swe2d_gpu_set_3d_patch_geometry(
    SWE2DDeviceState* dev,
    const double* phi_host,  // nullable
    const double* ax_host,   // nullable
    const double* ay_host,   // nullable
    const double* az_host,   // nullable
    int64_t n);

// Update per-face boundary mode/state on an allocated 3D Cartesian patch.
// Face index uses SWE3DPatchBoundaryFace ordering [0..5].
void swe2d_gpu_set_3d_patch_face_bc(
    SWE2DDeviceState* dev,
    int32_t face,
    int32_t mode,
    double u,
    double v,
    double w,
    double q,
    double vof,
    double p);

// CUDA Graph optimization API (Suggestion 9)
// Enable graph capture on next step, and use replayed graphs on subsequent steps.
void swe2d_gpu_enable_kernel_graphs(SWE2DDeviceState* dev, bool enable);

// Manually destroy cached graph (called at cleanup or on config change).
void swe2d_gpu_destroy_kernel_graphs(SWE2DDeviceState* dev);

// Free all device memory.
void swe2d_gpu_destroy(SWE2DDeviceState* dev);

// Query: returns true if a CUDA-capable device is available.
bool swe2d_gpu_available();

// ─────────────────────────────────────────────────────────────────────────────
// Culvert lookup-table mode: pre-computed Q(headwater,tailwater) tables.
// When culvert_solver_mode=1, the kernel uses bilinear interpolation on these
// tables instead of the iterative secant solver, reducing per-culvert compute.
// ─────────────────────────────────────────────────────────────────────────────

struct CulvertLookupTableDesc {
    int32_t n_hw = 32;   // headwater axis points
    int32_t n_tw = 16;   // tailwater axis points
    // All tables packed into flat arrays; per-culvert offsets into these arrays.
    // Offset i points to a block of (n_hw * n_tw) doubles, stored row-major
    // (hw varies fastest).
    double* d_table_data = nullptr;     // [total_table_points] on device
    double* d_table_header = nullptr;   // [n_culverts * 6] header data:
                                        //   [n_hw, n_tw, hw_min, hw_max, tw_min, tw_max]
    int32_t n_culverts = 0;
    int32_t capacity = 0;
    bool uploaded = false;
};

// Host-side table generation: for each culvert, compute Q(hw,tw) on a grid
// using the secant outlet-control solver.  Returns packed arrays ready for
// cudaMemcpy.  Called once at solver init.
bool swe2d_gpu_build_culvert_tables(
    int32_t n_culverts,
    const int32_t* culvert_code,
    const int32_t* culvert_shape,
    const double* culvert_rise,
    const double* culvert_span,
    const double* culvert_diameter,
    const double* culvert_length,
    const double* culvert_roughness_n,
    const double* culvert_slope,
    const double* entrance_loss_k,
    const double* exit_loss_k,
    int32_t n_hw,
    int32_t n_tw,
    std::vector<double>& table_data_out,
    std::vector<double>& table_header_out);

// Upload pre-built tables to device.  Must be called after CUDA context is active.
void swe2d_gpu_upload_culvert_tables(
    CulvertLookupTableDesc& desc,
    const std::vector<double>& table_data,
    const std::vector<double>& table_header);

// Release table device memory.
void swe2d_gpu_release_culvert_tables(CulvertLookupTableDesc& desc);
