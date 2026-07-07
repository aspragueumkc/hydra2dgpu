#pragma once
// swe2d_gpu.cuh
// CUDA device state and host API declarations for the 2D SWE GPU path.
// Included only when HYDRA_HAS_CUDA is defined.

#include "swe2d_mesh.hpp"
#include "swe2d_solver.hpp"   // SWE2DStepDiag
#include "swe2d_units.cuh"

// State storage precision. The codebase assumes State == double (FP64); FP32 state
// breaks conservation and is not supported. Selective FP32 inside `swe2d_flux_kernel`
// is the chosen mixed-precision strategy (see reference/MIXED_PRECISION_GPU_PLAN.md).
using State = double;

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

// ─────────────────────────────────────────────────────────────────────────────
// Device memory pool for one solver instance
// ─────────────────────────────────────────────────────────────────────────────

/// Per-cell gradient AoS struct (hx, hy, hux, huy, hvx, hvy).
struct Grad { double hx, hy, hux, huy, hvx, hvy; };

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

    // Owned/peer split CSR for atomics-free update kernel — removes
    // edge_c0[edge] == c branch from the hot loop.
    int32_t* d_cell_owned_offsets = nullptr;   // [n_cells+1] CSR offsets for owned edges
    int32_t* d_cell_peer_offsets  = nullptr;   // [n_cells+1] CSR offsets for peer edges
    int32_t* d_cell_owned_ids     = nullptr;   // [sum(n_owned)] edge indices where c0==c
    int32_t* d_cell_peer_ids      = nullptr;   // [sum(n_peer)]  edge indices where c1==c

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

    // Progressive BC group data (for on-device Q→q distribution).
    // One block per group; each block iterates over its edges in sorted order
    // and determines the active set based on frac = |Q|/peak_q.
    int32_t  n_prog_groups = 0;
    int32_t  n_prog_edges_total = 0;
    int32_t* d_prog_group_offsets = nullptr; // [n_prog_groups + 1]
    int32_t* d_prog_edge_hg_idx = nullptr;   // [n_prog_edges_total] hg index
    double*  d_prog_edge_len = nullptr;      // [n_prog_edges_total]
    double*  d_prog_edge_cum_len = nullptr;  // [n_prog_edges_total] cumulative length
    double*  d_prog_group_peak_q = nullptr;  // [n_prog_groups]
    double*  d_prog_group_total_len = nullptr; // [n_prog_groups]

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

    Grad*  d_grad = nullptr;

    // Per-edge gradient scratch (atomics-free path).  Written by gradient
    // kernel, consumed by gradient-gather kernel.  Same layout as flux arrays.
    double*  d_grad_edge_hx  = nullptr;   double*  d_grad_edge_hy  = nullptr;
    double*  d_grad_edge_hux = nullptr;   double*  d_grad_edge_huy = nullptr;
    double*  d_grad_edge_hvx = nullptr;   double*  d_grad_edge_hvy = nullptr;

    // Conserved state (updated each step) — stored as State (float or double)
    State*  d_h  = nullptr;
    State*  d_hu = nullptr;
    State*  d_hv = nullptr;

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

    // Max-tracking arrays: per-cell maximum values across entire simulation.
    // Written in the update kernel after every step, read back at sim end.
    double*  d_max_h  = nullptr;   // [n_cells]
    double*  d_max_hu = nullptr;   // [n_cells]
    double*  d_max_hv = nullptr;   // [n_cells]

    // Snapshot ring buffer: device-resident history of (h,hu,hv) snapshots.
    // Written at each output interval via D2D copy — no D2H until requested.
    // Allocated on first store, grows geometrically.
    double*  d_snap_h     = nullptr;   // [snap_capacity * n_cells]
    double*  d_snap_hu    = nullptr;   // [snap_capacity * n_cells]
    double*  d_snap_hv    = nullptr;   // [snap_capacity * n_cells]
    double*  d_snap_times = nullptr;   // [snap_capacity]
    int32_t  snap_capacity = 0;
    int32_t  snap_count    = 0;

    // ── Line metrics ring buffer ──────────────────────────────────────
    // Uploaded once at configure time.
    int32_t* d_lm_station_offsets = nullptr;  // [n_lines+1] prefix sum of stations
    int32_t* d_lm_cell_idx = nullptr;          // [total_stations]
    double*  d_lm_weights = nullptr;           // [total_stations]
    double*  d_lm_normal_x = nullptr;          // [n_lines]
    double*  d_lm_normal_y = nullptr;          // [n_lines]
    double*  d_lm_station_m = nullptr;         // [total_stations] station distances
    int32_t  lm_n_lines = 0;
    int32_t  lm_total_stations = 0;
    double   lm_gravity = 9.81;
    double   lm_h_min = 1.0e-6;

    // Ring buffer (written by kernel at store_snapshot time).
    // Profile layout: [snap_count * total_stations * 6] for 6 profile fields.
    // TS layout:      [snap_count * n_lines * 7] for 7 TS fields.
    // Wet layout:     [snap_count * total_stations] (int32).
    double*  d_lm_profile = nullptr;   // [lm_capacity * total_stations * 6]
    double*  d_lm_ts      = nullptr;   // [lm_capacity * n_lines * 7]
    int32_t* d_lm_wet     = nullptr;   // [lm_capacity * total_stations]
    double*  d_lm_times   = nullptr;   // [lm_capacity]
    int32_t  lm_capacity = 0;
    int32_t  lm_count    = 0;

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
    double*  d_rain_cn_scratch_h  = nullptr; // [n_cells] dedicated CN save/restore scratch
    double*  d_rain_cn_scratch_ex  = nullptr; // [n_cells] dedicated CN save/restore scratch
    double*  d_cell_source_mps    = nullptr; // [n_cells]
    double*  d_external_source_mps = nullptr; // [n_cells]
    // Per-stage rain/source snapshots used by graph-safe higher-order schemes.
    // Layout is contiguous by stage: slot*n_cells + cell.
    double*  d_stage_cell_source_mps = nullptr;
    int32_t  n_rain_gages = 0;
    int32_t  n_rain_samples = 0;
    double   rain_ia_ratio = 0.2;
    double   rain_mm_to_model_depth = 1.0e-3;

    // Rainfall update interval: re-evaluate SCS-CN rate every N seconds.
    // Applied as a constant source rate between updates.
    double   rain_update_interval_s = 60.0;   // default 60 s
    double   last_rain_update_time = -1.0;   // scalar last update tick (host-owned)
    double*  d_rain_excess_at_last_update_mm = nullptr;  // [n_cells] snapshot at last update

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

    // Persistent coupling workspace: reused across coupling calls to eliminate
    // per-call cudaMalloc/cudaFree and H→D re-upload when data is unchanged.
    // Allocated lazily on first use; survives for the lifetime of the device state.
    struct CouplingWorkspace {
        int32_t  cell_capacity = 0;
        double*  d_cell_area = nullptr;
        double*  d_source = nullptr;
        int32_t  inlet_capacity = 0;  // drainage inlet structure SoA capacity
        double*  d_drainage_q = nullptr;  // device-resident q_cell for coupling
        double*  d_node_depth = nullptr;  // device-resident node depths
        double*  d_link_flow = nullptr;   // device-resident link flows
        int32_t  node_capacity = 0;
        int32_t  link_capacity = 0;
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
        double*  d_prev_structure_flow = nullptr;   // previous-step flows for secant hint
        // Diagnostic readback: [n_structures][8] culvert-specific metrics.
        // Populated by swe2d_compute_structure_flows_kernel when non-null.
        // Read back to Python only on snapshot/finalization, never per-step.
        double*  d_culvert_diagnostics = nullptr;
        int32_t  n_culvert_diag_capacity = 0;
    } sf_ws{};

    // Persistent redistribution workspace: caches all redistribution
    // geometry arrays on-device, eliminating per-step cudaMalloc/free churn.
    // Static data (offsets, cell_idx, weights, up/down cells) is uploaded
    // once via content-hash tracking; flow values are re-uploaded each
    // step but are tiny (n_structures * 8 bytes).
    struct RedistWorkspace {
        int32_t  n_struct_capacity = 0;
        int32_t  dist_cell_capacity = 0;
        int32_t  cell_capacity = 0;
        uint64_t data_hash = 0;
        int32_t* d_offsets = nullptr;    // [n_struct + 1]
        int32_t* d_cell_idx = nullptr;   // [total_dist_cells]
        double*  d_weights = nullptr;    // [total_dist_cells]
        int32_t* d_up = nullptr;         // [n_struct]
        int32_t* d_dn = nullptr;         // [n_struct]
        double*  d_flow = nullptr;       // [n_struct] per-step flows (pre-allocated)
        double*  d_cell_area = nullptr;  // [n_cells] area, owned by RedistWorkspace
        double*  d_source = nullptr;     // [n_cells] source, owned by RedistWorkspace

        void destroy() {
            if (d_offsets) { cudaFree(d_offsets); d_offsets = nullptr; }
            if (d_cell_idx) { cudaFree(d_cell_idx); d_cell_idx = nullptr; }
            if (d_weights) { cudaFree(d_weights); d_weights = nullptr; }
            if (d_up) { cudaFree(d_up); d_up = nullptr; }
            if (d_dn) { cudaFree(d_dn); d_dn = nullptr; }
            if (d_flow) { cudaFree(d_flow); d_flow = nullptr; }
            if (d_cell_area) { cudaFree(d_cell_area); d_cell_area = nullptr; }
            if (d_source) { cudaFree(d_source); d_source = nullptr; }
            n_struct_capacity = 0;
            dist_cell_capacity = 0;
            cell_capacity = 0;
            data_hash = 0;
        }
    } redist_ws{};

    // ── Face-flux redistribution workspace ───────────────────────────
    // Caches device buffers for the on-device face-flux redistribution
    // kernel (swe2d_redistribute_face_flux_kernel), eliminating PCIe
    // transfers and the Python host loop that was the performance bottleneck.
    struct FaceFluxRedistWorkspace {
        int32_t  n_face_capacity = 0;
        int32_t  dist_cell_capacity = 0;
        uint64_t data_hash = 0;
        int32_t* d_struct_idx  = nullptr;  // [n_faces]
        int32_t* d_donor_cell  = nullptr;  // [n_faces]
        int32_t* d_receiver_cell = nullptr; // [n_faces]
        int32_t* d_offsets     = nullptr;  // [n_structures + 1] (shared with redist_ws)
        int32_t* d_cell_idx    = nullptr;  // [total_dist_cells]
        double*  d_weights     = nullptr;  // [total_dist_cells]

        void destroy() {
            if (d_struct_idx) { cudaFree(d_struct_idx); d_struct_idx = nullptr; }
            if (d_donor_cell) { cudaFree(d_donor_cell); d_donor_cell = nullptr; }
            if (d_receiver_cell) { cudaFree(d_receiver_cell); d_receiver_cell = nullptr; }
            // d_offsets/cell_idx/weights are owned by redist_ws — do not free here
            n_face_capacity = 0;
            dist_cell_capacity = 0;
            data_hash = 0;
        }
    } face_flux_redist_ws{};

    // ── Persistent drainage step workspace ────────────────────────────
    // Caches device buffers for the pipe1d drainage step so the per-step
    // call avoids ~30 cudaMalloc + cudaFree + sync cudaMemcpy operations.
    // Static geometry (node/link/pipe end topology) is uploaded once;
    // only runtime state (node_depth, link_flow) changes per step.
    struct DrainageStepWs {
        int32_t cell_capacity = 0, node_capacity = 0, link_capacity = 0;
        int32_t inlet_capacity = 0, outfall_capacity = 0, pipe_end_capacity = 0;
        int32_t n_inlets = 0, n_outfalls = 0, n_nodes = 0;
        bool    exchange_loaded = false;
        double  *d_cell_area = nullptr, *d_cell_wse = nullptr, *d_cell_depth = nullptr;
        double  *d_node_inv = nullptr, *d_node_maxd = nullptr, *d_node_area = nullptr;
        double  *d_node_depth = nullptr, *d_node_net_q = nullptr, *d_node_delta = nullptr;
        int32_t *d_l_from = nullptr, *d_l_to = nullptr;
        double  *d_l_len = nullptr, *d_l_n = nullptr, *d_l_d = nullptr, *d_l_qmax = nullptr;
        double  *d_l_q_prev = nullptr, *d_l_q = nullptr;
        int32_t *d_i_cell = nullptr, *d_i_node = nullptr;
        double  *d_i_crest = nullptr, *d_i_width = nullptr, *d_i_cd = nullptr, *d_i_qmax = nullptr;
        int32_t *d_o_cell = nullptr, *d_o_node = nullptr;
        double  *d_o_invert = nullptr, *d_o_diameter = nullptr, *d_o_cd = nullptr, *d_o_qmax = nullptr;
        int32_t *d_o_zero_storage = nullptr;
        int32_t *d_p_cell = nullptr, *d_p_node = nullptr;
        double  *d_p_invert = nullptr, *d_p_diameter = nullptr, *d_p_area = nullptr;
        double  *d_p_kin = nullptr, *d_p_kout = nullptr;
        double  *d_p_depth_bc = nullptr, *d_p_node_area = nullptr;
        double  *d_node_qleave = nullptr;
        double  *d_q_cell = nullptr, *d_limiter_events = nullptr, *d_limiter_volume = nullptr;

        void destroy() {
            #define _DS_FREE(p) do { if (p) { cudaFree(p); p = nullptr; } } while(0)
            _DS_FREE(d_cell_area); _DS_FREE(d_cell_wse); _DS_FREE(d_cell_depth);
            _DS_FREE(d_node_inv); _DS_FREE(d_node_maxd); _DS_FREE(d_node_area);
            _DS_FREE(d_node_depth); _DS_FREE(d_node_net_q); _DS_FREE(d_node_delta);
            _DS_FREE(d_l_from); _DS_FREE(d_l_to); _DS_FREE(d_l_len);
            _DS_FREE(d_l_n); _DS_FREE(d_l_d); _DS_FREE(d_l_qmax);
            _DS_FREE(d_l_q_prev); _DS_FREE(d_l_q);
            _DS_FREE(d_i_cell); _DS_FREE(d_i_node); _DS_FREE(d_i_crest);
            _DS_FREE(d_i_width); _DS_FREE(d_i_cd); _DS_FREE(d_i_qmax);
            _DS_FREE(d_o_cell); _DS_FREE(d_o_node); _DS_FREE(d_o_invert);
            _DS_FREE(d_o_diameter); _DS_FREE(d_o_cd); _DS_FREE(d_o_qmax); _DS_FREE(d_o_zero_storage);
            _DS_FREE(d_p_cell); _DS_FREE(d_p_node); _DS_FREE(d_p_invert);
            _DS_FREE(d_p_diameter); _DS_FREE(d_p_area); _DS_FREE(d_p_kin); _DS_FREE(d_p_kout);
            _DS_FREE(d_p_depth_bc); _DS_FREE(d_p_node_area); _DS_FREE(d_node_qleave);
            _DS_FREE(d_q_cell); _DS_FREE(d_limiter_events); _DS_FREE(d_limiter_volume);
            #undef _DS_FREE
            cell_capacity = node_capacity = link_capacity = 0;
            inlet_capacity = outfall_capacity = pipe_end_capacity = 0;
            n_inlets = n_outfalls = n_nodes = 0;
            exchange_loaded = false;
        }
    } drain_ws{};

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
        // Donor-cell area for depth safety limiter
        double*  d_donor_cell_area = nullptr;      // [n_culvert_faces]
        // Enquiry cells for total-energy-based driving head.
        // These are cells offset from the face from which we sample WSE and
        // velocity to compute the driving head (WSE + v²/2g) for the culvert
        // solver — avoiding the local drawdown singularity at the face cell.
        int32_t* d_enquiry_up_cell = nullptr;      // [n_culvert_faces]
        int32_t* d_enquiry_dn_cell = nullptr;      // [n_culvert_faces]
        // Enquiry offset distance (in model units, default 0 = use face cell)
        double enquiry_offset = 0.0;

        void destroy() {
            if (d_culvert_struct_idx) { cudaFree(d_culvert_struct_idx); d_culvert_struct_idx = nullptr; }
            if (d_face_nx) { cudaFree(d_face_nx); d_face_nx = nullptr; }
            if (d_face_ny) { cudaFree(d_face_ny); d_face_ny = nullptr; }
            if (d_face_width) { cudaFree(d_face_width); d_face_width = nullptr; }
            if (d_donor_cell) { cudaFree(d_donor_cell); d_donor_cell = nullptr; }
            if (d_receiver_cell) { cudaFree(d_receiver_cell); d_receiver_cell = nullptr; }
            if (d_invert_elev) { cudaFree(d_invert_elev); d_invert_elev = nullptr; }
            if (d_depth_safety) { cudaFree(d_depth_safety); d_depth_safety = nullptr; }
            if (d_donor_cell_area) { cudaFree(d_donor_cell_area); d_donor_cell_area = nullptr; }
            if (d_enquiry_up_cell) { cudaFree(d_enquiry_up_cell); d_enquiry_up_cell = nullptr; }
            if (d_enquiry_dn_cell) { cudaFree(d_enquiry_dn_cell); d_enquiry_dn_cell = nullptr; }
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

    // ── 1D pipe network device state ─────────────────────────────────────────
    struct Pipe1DDeviceState {
        int32_t*  d_owned_offsets;  // [n_pipe_cells + 1]
        int32_t*  d_owned_ids;      // [n_owned_faces]
        int32_t*  d_peer_offsets;   // [n_pipe_cells + 1]
        int32_t*  d_peer_ids;       // [n_peers]  peer = DrainageNode index

        int32_t*  d_cell_neighbor_cell;  // [2 * n_pipe_cells] neighbor cell per interface, -1 if boundary
        double*   d_cell_interface_dir;  // [2 * n_pipe_cells] -1.0=inlet, +1.0=outlet
        int32_t*  d_cell_from_node;  // [n_pipe_cells] from-node index per pipe cell
        int32_t*  d_cell_to_node;    // [n_pipe_cells] to-node index per pipe cell

        double*   d_cell_length;    // [n_pipe_cells]
        double*   d_cell_area;      // [n_pipe_cells]
        double*   d_cell_perim;     // [n_pipe_cells]
        double*   d_cell_invert;    // [n_pipe_cells]
        double*   d_cell_n;        // [n_pipe_cells]
        double*   d_cell_link_k;   // [n_pipe_cells] k at boundary cells only (0 interior)
        double*   d_cell_link_area; // [n_pipe_cells] full pipe area at boundary cells (0 interior)

        double*   d_node_invert;    // [n_nodes] invert elevation at each node
        double*   d_node_depth;     // [n_nodes]
        double*   d_node_net_q;     // [n_nodes]
        double*   d_node_surface_area; // [n_nodes]

        double*   d_A;              // [n_pipe_cells]
        double*   d_Q;              // [n_pipe_cells]
        double*   d_A_prev;         // [n_pipe_cells]
        double*   d_Q_iter;         // [n_pipe_cells]

        int32_t   n_pipe_cells = 0;
        int32_t   n_nodes = 0;

        void destroy() {
            #define _P_FREE(p) do { if (p) { cudaFree(p); p = nullptr; } } while(0)
            _P_FREE(d_owned_offsets); _P_FREE(d_owned_ids);
            _P_FREE(d_peer_offsets); _P_FREE(d_peer_ids);
            _P_FREE(d_cell_neighbor_cell); _P_FREE(d_cell_interface_dir);
            _P_FREE(d_cell_from_node); _P_FREE(d_cell_to_node);
            _P_FREE(d_cell_length); _P_FREE(d_cell_area);
            _P_FREE(d_cell_perim); _P_FREE(d_cell_invert);
            _P_FREE(d_cell_n); _P_FREE(d_cell_link_k); _P_FREE(d_cell_link_area);
            _P_FREE(d_node_invert); _P_FREE(d_node_depth); _P_FREE(d_node_net_q); _P_FREE(d_node_surface_area);
            _P_FREE(d_A); _P_FREE(d_Q); _P_FREE(d_A_prev); _P_FREE(d_Q_iter);
            n_pipe_cells = 0; n_nodes = 0;
            #undef _P_FREE
        }
    } pipe1d{};
};

// ─────────────────────────────────────────────────────────────────────────────
// Host API (callable from swe2d_solver.cpp)
// ─────────────────────────────────────────────────────────────────────────────

/** Allocate device memory and transfer static mesh topology + initial state.
    @param mesh Mesh topology reference
    @param h0 Initial water depth [n_cells]
    @param hu0 Initial x-momentum [n_cells]
    @param hv0 Initial y-momentum [n_cells]
    @param n_mann_cell Manning's roughness per cell [n_cells]
    @param degen_mode Degenerate-cell handling mode (default 0)
    @param max_inv_area Maximum inverse-area threshold (default 1.0e6)
    @returns Pointer to initialized SWE2DDeviceState
    @host */
SWE2DDeviceState* swe2d_gpu_init(
    const SWE2DMesh& mesh,
    const double*    h0,
    const double*    hu0,
    const double*    hv0,
    const double*    n_mann_cell,
    int              degen_mode   = 0,
    double           max_inv_area = 1.0e6);

/** Set the Manning unit-conversion factor in GPU constant memory.
    Call once after swe2d_gpu_init and before any step call.
    @param k_mann 1.0 for SI (meters), 1.486 for US Customary (feet)
    @host */
void swe2d_gpu_set_k_mann(double k_mann);

/** Set friction temporal-order hardening and shallow-correction params in GPU constant memory.
    Call once after swe2d_gpu_init and before any step call.
    @param substep_enabled Enable friction substeping
    @param target_courant Target Courant number for friction
    @param max_substeps Maximum number of friction substeps
    @param shallow_correction Enable shallow-water depth correction
    @param depth_alpha Depth-correction alpha parameter
    @param exponent Friction exponent
    @host */
void swe2d_gpu_set_friction_config(
    bool   substep_enabled,
    double target_courant,
    int    max_substeps,
    bool   shallow_correction,
    double depth_alpha,
    double exponent);

/** Advance one timestep on GPU. Writes diagnostics to *diag.
    @param dev Device state pointer
    @param t_now Current simulation time
    @param dt Time step size
    @param g Gravitational acceleration
    @param h_min Minimum depth for wet/dry threshold
    @param spatial_scheme Spatial reconstruction scheme index
    @param cfl_factor CFL safety factor
    @param max_inv_area Maximum inverse area for degenerate-cell checks
    @param cfl_lambda_cap Capped characteristic speed for CFL
    @param momentum_cap_min_speed Minimum speed for momentum cap
    @param momentum_cap_celerity_mult Celerity multiplier for momentum cap
    @param depth_cap Maximum allowable depth cap
    @param max_rel_depth_increase Maximum relative depth increase per step
    @param shallow_damping_depth Depth threshold for shallow damping
    @param extreme_rain_mode Enable extreme-rain source treatment
    @param source_cfl_beta Beta parameter for source-term CFL
    @param source_max_substeps Maximum source subcycles
    @param source_rate_cap Maximum source rate cap
    @param source_depth_step_cap Per-step depth change cap from sources
    @param source_true_subcycling Enable true subcycling for sources
    @param source_imex_split Enable IMEX splitting for sources
    @param enable_shallow_front_recon_fallback Enable shallow-front reconstruction fallback
    @param sync_diagnostics Synchronize diagnostics to host after step
    @param diag Output diagnostics struct
    @param front_flux_damping Damping factor for front fluxes (default 0.5)
    @param active_set_hysteresis Enable active-set hysteresis (default true)
    @host */
/** GPU kernel: gather per-edge gradient contributions into per-cell arrays.
 *  Cell-parallel.  Sums the edge-scratch contributions for incident edges
 *  of each active cell.  No atomics — each edge contribution is written
 *  to a unique edge slot by swe2d_gradient_kernel_edge. */
void swe2d_gpu_gather_gradients(
    SWE2DDeviceState* dev,
    int32_t n_cells,
    Grad* d_grad);

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

/** Persistent cooperative-kernel chunk stepping for tiny-N runs.
    Executes chunk_substeps internal substeps with dt/chunk_substeps each.
    Constrained to first-order single-stage hydrostatic stepping; falls back to baseline when unsupported.
    @param dev Device state pointer
    @param t_now Current simulation time
    @param dt Time step size
    @param chunk_substeps Number of internal substeps
    @param g Gravitational acceleration
    @param h_min Minimum depth for wet/dry threshold
    @param spatial_scheme Spatial reconstruction scheme index
    @param cfl_factor CFL safety factor
    @param max_inv_area Maximum inverse area for degenerate-cell checks
    @param cfl_lambda_cap Capped characteristic speed for CFL
    @param momentum_cap_min_speed Minimum speed for momentum cap
    @param momentum_cap_celerity_mult Celerity multiplier for momentum cap
    @param depth_cap Maximum allowable depth cap
    @param max_rel_depth_increase Maximum relative depth increase per step
    @param shallow_damping_depth Depth threshold for shallow damping
    @param extreme_rain_mode Enable extreme-rain source treatment
    @param source_cfl_beta Beta parameter for source-term CFL
    @param source_max_substeps Maximum source subcycles
    @param source_rate_cap Maximum source rate cap
    @param source_depth_step_cap Per-step depth change cap from sources
    @param source_true_subcycling Enable true subcycling for sources
    @param source_imex_split Enable IMEX splitting for sources
    @param enable_shallow_front_recon_fallback Enable shallow-front reconstruction fallback
    @param enable_active_edge_compaction Enable active-edge compaction
    @param sync_diagnostics Synchronize diagnostics to host after step
    @param diag Output diagnostics struct
    @param front_flux_damping Damping factor for front fluxes (default 0.5)
    @param active_set_hysteresis Enable active-set hysteresis (default true)
    @host */
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

/** Advance one SSPRK2 (Heun) timestep fully on GPU.
    @param dev Device state pointer
    @param t_now Current simulation time
    @param dt Time step size
    @param g Gravitational acceleration
    @param h_min Minimum depth for wet/dry threshold
    @param spatial_scheme Spatial reconstruction scheme index
    @param cfl_factor CFL safety factor
    @param max_inv_area Maximum inverse area for degenerate-cell checks
    @param cfl_lambda_cap Capped characteristic speed for CFL
    @param momentum_cap_min_speed Minimum speed for momentum cap
    @param momentum_cap_celerity_mult Celerity multiplier for momentum cap
    @param depth_cap Maximum allowable depth cap
    @param max_rel_depth_increase Maximum relative depth increase per step
    @param shallow_damping_depth Depth threshold for shallow damping
    @param extreme_rain_mode Enable extreme-rain source treatment
    @param source_cfl_beta Beta parameter for source-term CFL
    @param source_max_substeps Maximum source subcycles
    @param source_rate_cap Maximum source rate cap
    @param source_depth_step_cap Per-step depth change cap from sources
    @param source_true_subcycling Enable true subcycling for sources
    @param source_imex_split Enable IMEX splitting for sources
    @param enable_shallow_front_recon_fallback Enable shallow-front reconstruction fallback
    @param sync_diagnostics Synchronize diagnostics to host after step
    @param diag Output diagnostics struct
    @param front_flux_damping Damping factor for front fluxes (default 0.5)
    @param active_set_hysteresis Enable active-set hysteresis (default true)
    @host */
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

/** Advance one SSPRK2 (Heun) timestep using persistent chunk stepping for each RK stage.
    @param dev Device state pointer
    @param t_now Current simulation time
    @param dt Time step size
    @param chunk_substeps Number of internal substeps
    @param g Gravitational acceleration
    @param h_min Minimum depth for wet/dry threshold
    @param spatial_scheme Spatial reconstruction scheme index
    @param cfl_factor CFL safety factor
    @param max_inv_area Maximum inverse area for degenerate-cell checks
    @param cfl_lambda_cap Capped characteristic speed for CFL
    @param momentum_cap_min_speed Minimum speed for momentum cap
    @param momentum_cap_celerity_mult Celerity multiplier for momentum cap
    @param depth_cap Maximum allowable depth cap
    @param max_rel_depth_increase Maximum relative depth increase per step
    @param shallow_damping_depth Depth threshold for shallow damping
    @param extreme_rain_mode Enable extreme-rain source treatment
    @param source_cfl_beta Beta parameter for source-term CFL
    @param source_max_substeps Maximum source subcycles
    @param source_rate_cap Maximum source rate cap
    @param source_depth_step_cap Per-step depth change cap from sources
    @param source_true_subcycling Enable true subcycling for sources
    @param source_imex_split Enable IMEX splitting for sources
    @param enable_shallow_front_recon_fallback Enable shallow-front reconstruction fallback
    @param enable_active_edge_compaction Enable active-edge compaction
    @param sync_diagnostics Synchronize diagnostics to host after step
    @param diag Output diagnostics struct
    @param front_flux_damping Damping factor for front fluxes (default 0.5)
    @param active_set_hysteresis Enable active-set hysteresis (default true)
    @host */
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

/** Three-stage SSP Shu-Osher RK3 (graph-safe).
    Uses d_k4_h/hu/hv and d_k6_h/hu/hv as slope scratch buffers.
    Rain CN save/restore uses dedicated d_rain_cn_scratch_h/ex.
    @host */
void swe2d_gpu_step_rk3(
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

/** Four-stage textbook RK4 (graph-safe).
    k1=h1-h0 (d_k4), k2=h2-h0 (d_k6), k3=h3-h0, k4=h4-h2.
    Stage 2 momentum saved to d_hu1/d_hv1; Stage 4 restore uses those.
    @host */
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

/** Six-stage Cash-Karp RK5(4) embedded (graph-safe).
    k1 in d_k4, k3/k6 in d_k6 (k3 overwritten by k4→k5→k6 chain),
    k4 in d_h1. y3/y4/y5 momentum preserved in d_hu1/d_hv1/d_h2.
    @host */
void swe2d_gpu_step_rk5(
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

/** Compute a CFL-limited dt from current device state without host-state sync.
    @param dev Device state pointer
    @param g Gravitational acceleration
    @param h_min Minimum depth for wet/dry threshold
    @param cfl_factor CFL safety factor
    @param dt_max Maximum allowable dt
    @param cfl_lambda_cap Capped characteristic speed for CFL
    @returns CFL-limited time step
    @host */
double swe2d_gpu_compute_dt(
    SWE2DDeviceState* dev,
    double g,
    double h_min,
    double cfl_factor,
    double dt_max,
    double cfl_lambda_cap);

/** Copy current state from device to caller-supplied host arrays.
    @param dev Device state pointer
    @param h_out Host output buffer for depth [n_cells]
    @param hu_out Host output buffer for x-momentum [n_cells]
    @param hv_out Host output buffer for y-momentum [n_cells]
    @host */
void swe2d_gpu_get_state(
    SWE2DDeviceState* dev,
    double* h_out,
    double* hu_out,
    double* hv_out);

/** Lightweight readback of just h (depth) from the coupling device state.
    Uses s_coupling_dev global — no device pointer needed.
    @param host_buf Host output buffer for depth [n_cells]
    @param n_cells Number of cells
    @host */
void swe2d_gpu_readback_h(double* host_buf, int32_t n_cells);

void swe2d_gpu_readback_max_tracking(
    SWE2DDeviceState* dev,
    double* h_max_out,
    double* hu_max_out,
    double* hv_max_out);

/** Upload host state arrays into the current device solver state.
    @param dev Device state pointer
    @param h_in Host input buffer for depth [n_cells]
    @param hu_in Host input buffer for x-momentum [n_cells]
    @param hv_in Host input buffer for y-momentum [n_cells]
    @host */
void swe2d_gpu_set_state(
    SWE2DDeviceState* dev,
    const double* h_in,
    const double* hu_in,
    const double* hv_in);

/** Push updated boundary type/value arrays to device for selected edges.
    @param dev Device state pointer
    @param edge_index Array of edge indices [n_updates]
    @param bc_type Array of boundary condition types [n_updates]
    @param bc_val Array of boundary condition values [n_updates]
    @param n_updates Number of edge updates
    @host */
void swe2d_gpu_update_boundary_values(
    SWE2DDeviceState* dev,
    const int32_t* edge_index,
    const int32_t* bc_type,
    const double* bc_val,
    int32_t n_updates);

/** Upload per-edge hydrograph forcing arrays.
    @param dev Device state pointer
    @param edge_index Edge indices [n_edges]
    @param bc_type BC types per edge [n_edges]
    @param offsets HG sample offsets [n_edges+1]
    @param time_s Sample time values [n_samples]
    @param value Sample flow/depth values [n_samples]
    @param n_edges Number of hydrograph edges
    @param n_samples Number of total samples
    @host */
void swe2d_gpu_set_boundary_hydrographs(
    SWE2DDeviceState* dev,
    const int32_t* edge_index,
    const int32_t* bc_type,
    const int32_t* offsets,
    const double* time_s,
    const double* value,
    int32_t n_edges,
    int32_t n_samples);

/** Upload progressive BC group metadata for on-device Q->q distribution.
    Uses one kernel block per group.
    @param dev Device state pointer
    @param n_groups Number of progressive BC groups
    @param n_edges_total Total edges across all groups
    @param group_offsets Per-group edge offsets [n_groups+1]
    @param edge_hg_idx HG index per edge [n_edges_total]
    @param edge_len Edge length [n_edges_total]
    @param edge_cum_len Cumulative edge length [n_edges_total]
    @param group_peak_q Peak Q per group [n_groups]
    @param group_total_len Total length per group [n_groups]
    @host */
void swe2d_gpu_set_progressive_bc_data(
    SWE2DDeviceState* dev,
    int32_t n_groups,
    int32_t n_edges_total,
    const int32_t* group_offsets,
    const int32_t* edge_hg_idx,
    const double* edge_len,
    const double* edge_cum_len,
    const double* group_peak_q,
    const double* group_total_len);

/** Upload per-cell rain+CN forcing arrays.
    @param dev Device state pointer
    @param cell_gage_idx Gage index per cell [n_cells]
    @param gage_offsets Gage sample offsets [n_gages+1]
    @param hg_time_s Sample time values [n_samples]
    @param hg_cum_mm Cumulative rainfall [n_samples] (mm)
    @param cn Curve number per cell [n_cells]
    @param n_cells Number of cells
    @param n_gages Number of rain gages
    @param n_samples Number of total samples
    @param ia_ratio Initial abstraction ratio (default 0.2)
    @param mm_to_model_depth Conversion factor mm to model depth units
    @param rain_update_interval_s SCS-CN re-evaluation interval in seconds (default 60.0)
    @host */
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
    double mm_to_model_depth,
    double rain_update_interval_s);

/** Upload per-cell external source terms [m/s] used by the GPU step update.
    Passing nullptr clears external sources on the device.
    @param dev Device state pointer
    @param source_mps Per-cell source rates [m/s], or nullptr to clear
    @param n_cells Number of cells
    @host */
void swe2d_gpu_set_external_sources(
    SWE2DDeviceState* dev,
    const double* source_mps,
    int32_t n_cells);

/** Compute per-cell depth-rate sources [m/s] from drainage/structure transfer arrays.
    When dev is non-null, uses persistent device buffers and async stream.
    @param dev Device state pointer (nullable, enables persistent workspace + async)
    @param n_cells Number of cells
    @param cell_area_m2 Cell area [n_cells] (m²)
    @param n_inlets Number of drainage inlets
    @param inlet_cell Inlet cell indices [n_inlets]
    @param inlet_flow_cms Inlet flow rates [n_inlets] (m³/s)
    @param n_structures Number of structures
    @param structure_up_cell Upstream cell per structure [n_structures]
    @param structure_down_cell Downstream cell per structure [n_structures]
    @param structure_flow_cms Structure flow rates [n_structures] (m³/s)
    @param source_rate_mps_out Output per-cell source rates [n_cells] (m/s)
    @host */
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

/** Apply bridge loss law to structure flows then convert to cell depth-rate sources [m/s].
    @param dev Device state pointer (nullable)
    @param n_cells Number of cells
    @param cell_area_m2 Cell area [n_cells] (m²)
    @param n_bridges Number of bridges
    @param bridge_up_cell Upstream cell per bridge [n_bridges]
    @param bridge_down_cell Downstream cell per bridge [n_bridges]
    @param bridge_flow_cms Bridge flow rates [n_bridges] (m³/s)
    @param bridge_loss_k_upstream Upstream loss coefficient [n_bridges]
    @param bridge_loss_k_downstream Downstream loss coefficient [n_bridges]
    @param bridge_opening_width_m Bridge opening width (m)
    @param dt_s Time step (s)
    @param source_rate_mps_out Output per-cell source rates [n_cells] (m/s)
    @host */
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

/** Redistribute structure sources across a pre-computed corridor of cells.
    For structures with influence_width > 0.
    @param dev Device state pointer (nullable)
    @param n_structures Number of structures
    @param structure_flow_cms Structure flow rates [n_structures] (m³/s)
    @param orig_up_cell Original upstream cell per structure [n_structures]
    @param orig_dn_cell Original downstream cell per structure [n_structures]
    @param cell_area_m2 Cell area [n_cells] (m²)
    @param dist_offsets Distribution offsets per structure [n_structures+1]
    @param dist_cell_idx Distribution cell indices [total_dist_cells]
    @param dist_weights Distribution weights [total_dist_cells]
    @param n_cells Number of cells
    @param source_rate_mps_inout Per-cell source rates in/out [n_cells] (m/s)
    @host */
/// Redistribute structure flow sources across multiple cells using precomputed weights. @host
void swe2d_gpu_redistribute_structure_sources(
    SWE2DDeviceState* dev,   // nullable: uses host path
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

/// On-device-only redistribution operating directly on dev->d_external_source_mps.
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

/** Compute per-structure transfer flow [m^3/s] on CUDA. @host */
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

/** Fused structure-flows + coupling-sources kernel: avoids H->D->H round trip. @host */
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

// ── Snapshot ring buffer ──
/// Allocate or grow the device snapshot ring buffer to hold at least `min_cap` snapshots. @host
void swe2d_gpu_ensure_snapshot_buf(SWE2DDeviceState* dev, int32_t min_cap);
/// Copy current h/hu/hv into the next snapshot slot. @host
void swe2d_gpu_store_snapshot(SWE2DDeviceState* dev, double t_s);
/// Read all accumulated snapshots back to host.  Returns {t_s, h, hu, hv} arrays. @host
/// h shape (snap_count, n_cells), t_s shape (snap_count,).  Caller must free via cudaFreeHost.
void swe2d_gpu_read_snapshots(SWE2DDeviceState* dev,
                               double** out_t_s, double** out_h, double** out_hu, double** out_hv,
                               int32_t* out_count, int32_t* out_n_cells);
/// Free snapshot ring buffer. @host
void swe2d_gpu_free_snapshot_buf(SWE2DDeviceState* dev);

// ── Line metrics ring buffer ──
/// Upload line sampling map and allocate ring buffer.  Call once after swe2d_gpu_init.
/// station_offsets [n_lines+1] prefix sum, cell_idx/weights [total_stations],
/// normal_x/y [n_lines], station_m [total_stations], gravity/h_min are scalars,
/// max_snapshots ring buffer capacity (will grow geometrically like mesh snapshots).
void swe2d_gpu_configure_line_sampling(
    SWE2DDeviceState* dev,
    int32_t           n_lines,
    const int32_t*    station_offsets,
    const int32_t*    cell_idx,
    const double*     weights,
    const double*     normal_x,
    const double*     normal_y,
    const double*     station_m,
    double            gravity,
    double            h_min);

/// Kernel (called from swe2d_gpu_store_snapshot): compute line metrics for current
/// solver state and write to the ring buffer at slot snap_count.
void swe2d_gpu_store_line_metrics(SWE2DDeviceState* dev, double t_s);

/// Read all accumulated line metrics back to host (profile + TS + wet flags).
/// Returns pinned host memory freed by caller via cudaFreeHost.
void swe2d_gpu_read_line_metrics(SWE2DDeviceState* dev,
                                  double** out_times,
                                  double** out_profile,   // [lm_count * total_stations * 6]
                                  double** out_ts,        // [lm_count * n_lines * 7]
                                  int32_t** out_wet,      // [lm_count * total_stations]
                                  double** out_station_m, // [total_stations]
                                  int32_t** out_station_offsets, // [n_lines + 1]
                                  int32_t* out_count,
                                  int32_t* out_n_lines,
                                  int32_t* out_total_stations);

/// Free line metrics ring buffer + sample map. @host
void swe2d_gpu_free_line_metrics(SWE2DDeviceState* dev);

// ── Persistent GPU coupling path ──
/// Set the global coupling device pointer for persistent GPU coupling. @host
void swe2d_gpu_set_coupling_device_global(SWE2DDeviceState* dev);
/// Pre-load structure parameters into persistent device workspace. @host
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
/// Pre-load coupling cell areas into persistent device workspace. @host
void swe2d_gpu_preload_coupling_cell_area(SWE2DDeviceState* dev, int32_t n_cells, const double* cell_area_m2);
/// Compute coupling sources entirely on-device (no host readback). @host
void swe2d_gpu_compute_coupling_full_on_device(
    SWE2DDeviceState* dev, int32_t n_cells, int32_t n_structures, const double* cell_wse_host,
    const double* host_structure_flows = nullptr,
    bool graph_safe = false);
void swe2d_recompute_coupling_for_stage(SWE2DDeviceState* dev, int32_t n_cells,
                                         int32_t n_structures, const double* cell_wse_host,
                                         const double* host_structure_flows, double dt_stage);
/// Upload drainage exchange parameters (inlets, outfalls, node geometry). @host
void swe2d_gpu_upload_drainage_exchange_params(
    SWE2DDeviceState* dev,
    int32_t n_nodes, int32_t n_inlets, int32_t n_outfalls,
    const int32_t* inlet_cell, const int32_t* inlet_node,
    const double* inlet_crest, const double* inlet_width,
    const double* inlet_cd, const double* inlet_qmax,
    const int32_t* outfall_cell, const int32_t* outfall_node,
    const double* outfall_invert, const double* outfall_diameter,
    const double* outfall_cd, const double* outfall_qmax,
    const int32_t* outfall_zero_storage,
    const double* node_max_depth);
/// Ensure drainage Q buffer is allocated in device workspace. @host
void swe2d_gpu_ensure_drainage_q_buf(SWE2DDeviceState* dev, int32_t n_cells);

/** Accumulate host-provided source rates into d_external_source_mps on-device.
 *  Uploads host_src to a persistent staging buffer, then adds element-by-element
 *  to d_external_source_mps via a GPU kernel.  No D2H readback.
 */
void swe2d_gpu_accumulate_external_source(
    SWE2DDeviceState* dev,
    const double* host_src,
    int32_t n_cells);
/// Read back coupling sources from device to host. @host
void swe2d_gpu_readback_coupling_sources(double* host_buf, int32_t n_cells);
/// Read back structure flows from device to host. @host
void swe2d_gpu_readback_structure_flows(double* host_buf, int32_t n_structures);
/// Upload structure flows from host to device. @host
void swe2d_gpu_upload_structure_flows(const double* host_buf, int32_t n_structures);
/// Read back coupling WSE from device to host. @host
void swe2d_gpu_readback_coupling_wse(double* host_buf, int32_t n_cells);

/** Upload culvert face-flux geometry to the GPU. Called once on mesh/structure change.
    @param dev Device state pointer
    @param n_culvert_faces Number of culvert faces
    @param culvert_struct_idx Culvert structure index per face [n_culvert_faces]
    @param face_nx Face normal x-component [n_culvert_faces]
    @param face_ny Face normal y-component [n_culvert_faces]
    @param face_width Face width [n_culvert_faces]
    @param donor_cell Donor cell index per face [n_culvert_faces]
    @param receiver_cell Receiver cell index per face [n_culvert_faces]
    @param invert_elev Invert elevation per face [n_culvert_faces]
    @param depth_safety Depth safety factor per face [n_culvert_faces]
    @param donor_cell_area Donor cell area [n_culvert_faces]
    @param enquiry_up_cell Enquiry upstream cell (nullable) [n_culvert_faces]
    @param enquiry_dn_cell Enquiry downstream cell (nullable) [n_culvert_faces]
    @param use_face_flux Toggle for face-based flux mode
    @host */
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
    const double*  donor_cell_area,
    const int32_t* enquiry_up_cell,   // nullable: enquiry cells for WSE sampling
    const int32_t* enquiry_dn_cell,   // nullable: enquiry cells for WSE sampling
    bool use_face_flux);

/** Compute structure flows on device then apply face-based culvert fluxes.
    @param dev Device state pointer
    @param dt Time step (s)
    @param h_min Minimum depth threshold
    @host */
void swe2d_gpu_apply_culvert_face_flux(
    SWE2DDeviceState* dev,
    double dt,
    double h_min);

/// Allocate and zero the per-cell external flux accumulators on device. @host
void swe2d_gpu_alloc_ext_struct_flux(SWE2DDeviceState* dev, int32_t n_cells);
/// Fold culvert face-flux mass into d_external_source_mps for subcycling. @host
void swe2d_gpu_fold_culvert_mass_to_source(SWE2DDeviceState* dev, int32_t n_cells);
/// Read back per-cell external structure flux arrays from device (for debug). @host
void swe2d_gpu_readback_ext_struct_flux(
    double* host_h, double* host_hu, double* host_hv, int32_t n_cells);

/// Upload host mass flux to device d_ext_struct_flux_h (for redistribution). @host
void swe2d_gpu_upload_ext_struct_flux_h(const double* host_h, int32_t n_cells);
/// On-device face-flux redistribution (no PCIe readback). Operates on
/// d_ext_struct_flux_h in place using pre-loaded redistribution geometry. @host
void swe2d_gpu_redistribute_face_flux(
    SWE2DDeviceState* dev,
    int32_t n_faces,
    const int32_t* struct_idx,
    const int32_t* donor_cell,
    const int32_t* receiver_cell,
    const int32_t* dist_offsets,
    const int32_t* dist_cell_idx,
    const double*  dist_weights,
    int32_t n_cells);
/// Set the coupling time step (used by face-flux depth limiter). @host
void swe2d_gpu_set_coupling_dt(double dt);

/** Build 1D pipe network CSR topology and allocate device buffers.
    @param n_links Number of links
    @param link_from_node From node index [n_links]
    @param link_to_node To node index [n_links]
    @param link_length Geometric length [n_links]
    @param link_diameter Pipe diameter [n_links]
    @param link_roughness_n Manning's n [n_links]
    @param link_inlet_loss_k Inlet minor loss K [n_links]
    @param link_outlet_loss_k Outlet minor loss K [n_links]
    @param node_invert_elev Node invert elevation [n_nodes]
    @param node_surface_area Node surface area [n_nodes]
    @param node_max_depth Node max depth [n_nodes]
    @param link_invert_in Inlet invert [n_links]
    @param link_invert_out Outlet invert [n_links]
    @param max_cell_length Max sub-cell length (0=no subdivision)
    @param dev Output pipe1d state
    @host */
void swe2d_build_pipe1d_mesh(
    int32_t               n_links,
    const int32_t*        link_from_node,
    const int32_t*        link_to_node,
    const double*         link_length,
    const double*         link_diameter,
    const double*         link_roughness_n,
    const double*         link_inlet_loss_k,
    const double*         link_outlet_loss_k,
    const double*         node_invert_elev,
    const double*         node_surface_area,
    const double*         node_max_depth,
    const double*         link_invert_in,
    const double*         link_invert_out,
    int32_t               max_cell_length,
    SWE2DDeviceState::Pipe1DDeviceState* dev);

/** GPU kernel: compute HLLE flux for 1D pipe network.
    @param n_cells Number of pipe cells
    @param owned_offsets CSR offsets [n_cells+1]
    @param owned_ids Interface IDs [n_owned_faces]
    @param neighbor_cell Neighbor cell per interface [2*n_cells], -1 if boundary
    @param interface_dir Interface direction [2*n_cells]: -1=inlet, +1=outlet
    @param cell_from_node From-node per cell [n_cells]
    @param cell_to_node To-node per cell [n_cells]
    @param cell_invert Cell midpoint invert [n_cells]
    @param cell_perim Pipe perimeter [n_cells]
    @param cell_k_loss Minor loss K at boundary cells [n_cells] (0 interior)
    @param cell_A Current area [n_cells]
    @param cell_Q Current discharge [n_cells]
    @param node_invert Node invert elevation [n_nodes]
    @param node_depth Node depth [n_nodes]
    @param cell_length Cell length [n_cells]
    @param flux_Q_out Net flux OUT of each cell [n_cells]
    @param g Gravitational acceleration
    @host */
void swe2d_pipe1d_flux_kernel_host(
    int32_t               n_cells,
    const int32_t*        owned_offsets,
    const int32_t*        owned_ids,
    const int32_t*        neighbor_cell,
    const double*         interface_dir,
    const int32_t*        cell_from_node,
    const int32_t*        cell_to_node,
    const double*         cell_invert,
    const double*         cell_perim,
    const double*         cell_A,
    const double*         cell_Q,
    const double*         node_invert,
    const double*         node_depth,
    const double*         cell_length,
    double*               flux_Q_out,
    double                g);

/** GPU kernel: explicit diffusion-wave update for 1D pipe network.
    @param n_cells Number of pipe cells
    @param cell_length Cell length [n_cells]
    @param cell_area_full Full pipe cross-section area [n_cells]
    @param cell_perim Pipe perimeter [n_cells]
    @param cell_n Manning's n [n_cells]
    @param cell_A Current area [n_cells]
    @param cell_Q Current discharge [n_cells]
    @param flux_Q Net flux OUT of cell [n_cells]
    @param dt Time step
    @param g Gravitational acceleration
    @param cell_A_new Updated area [n_cells]
    @param cell_Q_new Updated discharge [n_cells]
    @host */
void swe2d_pipe1d_diffusion_wave_kernel_host(
    int32_t               n_cells,
    const double*         cell_length,
    const double*         cell_area_full,
    const double*         cell_perim,
    const double*         cell_n,
    const double*         cell_k_loss,
    const double*         cell_A,
    const double*         cell_Q,
    const double*         flux_Q,
    double                dt,
    double                g,
    double*               cell_A_new,
    double*               cell_Q_new);

/** GPU kernel: semi-implicit fully-dynamic update for 1D pipe network.
    @param n_cells Number of pipe cells
    @param n_iters Number of Picard iterations
    @param relaxation Relaxation factor (0-1)
    @param owned_offsets CSR offsets [n_cells+1]
    @param owned_ids Interface IDs [n_owned_faces]
    @param neighbor_cell Neighbor cell per interface [2*n_cells]
    @param interface_dir Interface direction [2*n_cells]
    @param cell_from_node From-node per cell [n_cells]
    @param cell_to_node To-node per cell [n_cells]
    @param cell_length Cell length [n_cells]
    @param cell_area_full Full pipe cross-section area [n_cells]
    @param cell_perim Pipe perimeter [n_cells]
    @param cell_n Manning's n [n_cells]
    @param cell_k_loss Minor loss K at boundary cells [n_cells] (0 interior)
    @param node_invert Node invert elevation [n_nodes]
    @param node_depth Node depth [n_nodes]
    @param cell_A_prev Area from previous coupling step [n_cells]
    @param cell_Q_prev Discharge from previous step [n_cells]
    @param cell_A_iter Area iteration buffer [n_cells]
    @param cell_Q_iter Discharge iteration buffer [n_cells]
    @param dt Time step
    @param g Gravitational acceleration
    @host */
void swe2d_pipe1d_fully_dynamic_kernel_host(
    int32_t               n_cells,
    int32_t               n_iters,
    double                relaxation,
    const int32_t*        owned_offsets,
    const int32_t*        owned_ids,
    const int32_t*        neighbor_cell,
    const double*         interface_dir,
    const int32_t*        cell_from_node,
    const int32_t*        cell_to_node,
    const double*         cell_length,
    const double*         cell_area_full,
    const double*         cell_perim,
    const double*         cell_n,
    const double*         cell_k_loss,
    const double*         node_invert,
    const double*         node_depth,
    const double*         cell_A_prev,
    const double*         cell_Q_prev,
    double*               cell_A_iter,
    double*               cell_Q_iter,
    double                dt,
    double                g);

/** Host wrapper: advance 1D pipe network one coupling step.
    Orchestrates flux kernel + update kernel (diffusion or fully dynamic) in sequence.
    @param dev Device state pointer (with pipe1d state initialized)
    @param dt Coupling timestep for this substep
    @param solver_mode "diffusion_wave" or "fully_dynamic"
    @param coupling_substeps Number of substeps within this coupling step
    @param implicit_iters Number of Picard iterations per substep (for fully_dynamic)
    @param relaxation Relaxation factor for Picard iteration (0-1)
    @param g Gravitational acceleration
    @host */
void swe2d_pipe1d_step(
    SWE2DDeviceState* dev,
    double            dt,
    const char*       solver_mode,
    int32_t           coupling_substeps,
    int32_t           implicit_iters,
    double            relaxation,
    double            g);

/** Upload node depths from host to device (called before each pipe step).
    @param dev Device state pointer
    @param host_node_depth Host array of node depths [n_nodes]
    @param n_nodes Number of nodes
    @host */
void swe2d_pipe1d_upload_node_depth(
    SWE2DDeviceState* dev,
    const double*     host_node_depth,
    int32_t           n_nodes);

/** Readback pipe1d node/cell state for diagnostics and tests.
    @param dev Device state pointer
    @param host_node_depth Output host buffer for node depths [n_nodes] (may be nullptr)
    @param host_cell_A Output host buffer for cell areas [n_cells] (may be nullptr)
    @param host_cell_Q Output host buffer for cell flows [n_cells] (may be nullptr)
    @param n_nodes Expected number of nodes
    @param n_cells Expected number of pipe cells
    @host */
void swe2d_pipe1d_readback_node_state(
    SWE2DDeviceState* dev,
    double*           host_node_depth,
    double*           host_cell_A,
    double*           host_cell_Q,
    int32_t           n_nodes,
    int32_t           n_cells);

/** Enable graph capture on next step, use replayed graphs on subsequent steps.
    @param dev Device state pointer
    @param enable True to enable kernel graph capture/replay
    @host */
void swe2d_gpu_enable_kernel_graphs(SWE2DDeviceState* dev, bool enable);

/** Manually destroy cached kernel graph (at cleanup or on config change).
    @param dev Device state pointer
    @host */
void swe2d_gpu_destroy_kernel_graphs(SWE2DDeviceState* dev);

/** Invalidate cached kernel graph so the solver re-captures on the next step.
    Unlike destroy (which frees resources), invalidation sets a flag so the
    next swe2d_gpu_step call detects a cache miss and re-captures.
    Use when coupling changes dev->use_culvert_face_flux mid-run.
    @param dev Device state pointer
    @host */
void swe2d_gpu_invalidate_graph_cache(SWE2DDeviceState* dev);

/** Free all device memory.
    @param dev Device state pointer
    @host */
void swe2d_gpu_destroy(SWE2DDeviceState* dev);

/// Query: returns true if a CUDA-capable device is available. @host
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

/** Host-side table generation: compute Q(hw,tw) on a grid using secant outlet-control solver.
    Returns packed arrays ready for cudaMemcpy. Called once at solver init.
    @returns true on success @host */
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
    double model_to_ft,
    int32_t n_hw,
    int32_t n_tw,
    std::vector<double>& table_data_out,
    std::vector<double>& table_header_out);

/** Upload pre-built culvert lookup tables to device. Must be called after CUDA context active. @host */
void swe2d_gpu_upload_culvert_tables(
    CulvertLookupTableDesc& desc,
    const std::vector<double>& table_data,
    const std::vector<double>& table_header);

/// Release culvert lookup table device memory. @host
void swe2d_gpu_release_culvert_tables(CulvertLookupTableDesc& desc);
