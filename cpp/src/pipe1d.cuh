#pragma once
// pipe1d.cuh
// 1D pipe network device state and host API declarations for the GPU solver.
// Split from swe2d_gpu.cuh — mechanical refactoring, no behavior changes.

#include <cstdint>
#include <cuda_runtime.h>
#include <vector>

// ── 1D pipe network device state ─────────────────────────────────────────
struct Pipe1DDeviceState {
    int32_t*  d_owned_offsets;  // [n_pipe_cells + 1]
    int32_t*  d_owned_ids;      // [n_owned_faces]
    int32_t*  d_peer_offsets;   // [n_pipe_cells + 1]
    int32_t*  d_peer_ids;       // [n_peers]  peer = DrainageNode index

    // d_cell_neighbor_cell / d_cell_interface_dir removed (Phase 2.1 — replaced by face mesh)
    double*   d_slope_A;         // [n_pipe_cells] minmod-limited A (area) gradient for MUSCL reconstruction
    double*   d_slope_Q;         // [n_pipe_cells] minmod-limited Q (flow) gradient for MUSCL reconstruction
    int32_t*  d_cell_from_node;  // [n_cells_all] per-cell from-node tag; used by godunov update
                                 //  for boundary/interior face detection (>= n_nodes means interior).
                                 //  Retained for the godunov update kernel's boundary-vs-interior
                                 //  check (Phase 2.3 derivation of eta_left/eta_right from neighbours).
    int32_t*  d_cell_to_node;    // [n_cells_all] per-cell to-node tag (same rationale as from_node).

    double*   d_cell_length;    // [n_pipe_cells] sub-cell length (for volume/continuity)
    double*   d_cell_link_length; // [n_pipe_cells] total geometric length of the owning link (for head gradient)
    double*   d_cell_area;      // [n_pipe_cells]
    double*   d_cell_perim;     // [n_pipe_cells]
    double*   d_cell_invert;    // [n_pipe_cells]
    double*   d_cell_n;        // [n_pipe_cells]
    double*   d_cell_link_k;   // [n_pipe_cells] k at boundary cells only (0 interior)
    double*   d_cell_link_k_in;  // [n_pipe_cells] link-level entrance loss coeff (all cells of link)
    double*   d_cell_link_k_out; // [n_pipe_cells] link-level exit loss coeff (all cells of link)
    double*   d_cell_link_area; // [n_pipe_cells] full pipe area at boundary cells (0 interior)
    int32_t*  d_cell_owner_link; // [n_pipe_cells] which link each sub-cell belongs to
    int32_t*  d_cell_sub_idx;    // [n_pipe_cells] sub-cell index within its link
    double*   d_cell_S0;         // [n_pipe_cells] conduit slope (link_invert_in - link_invert_out) / L_link
    int32_t*  d_cell_is_end;     // [n_pipe_cells] 1 if cell is at link end (s == N-1), else 0 (SPEC §2.6)

    int32_t*  d_cell_shape_type; // [n_pipe_cells] 0=circular 1=rect 2=ellipse
    double*   d_cell_width;      // [n_pipe_cells]
    double*   d_cell_height;     // [n_pipe_cells] cross-section height (rise/vertical dimension)
                                  // Alias: cell_rise. NOT storage-cell depth (that's cell_h).
    double*   d_cell_tables;     // [n_pipe_cells × 2 × PIPE1D_TABLE_N] flattened P_ratio + T_ratio

    // SPEC §2.2 — Per-cell state (updated by diffusion/kernels)
    double*   d_cell_y;          // [n_pipe_cells] water surface elevation (m abs)
    double*   d_cell_q;          // [n_pipe_cells] cell_Q / cell_A (sign-aware velocity)
    double*   d_cell_fr;         // [n_pipe_cells] Froude number |cell_q| / sqrt(g * R_h)
    double*   d_cell_h;          // [n_pipe_cells] flow depth (m, invert-relative)
    double*   d_cell_slot_width; // [n_pipe_cells] Preissmann slot width (0 for open-channel)

    // d_node_crown removed (Phase 2.1 — manhole pipe-crown lives on d_cell_crown)
// d_vnode_H/Q/to_link/idx + n_vnodes removed (Phase 2.1 — virtual nodes retired; interior faces use d_face_owner_R)
// d_slope_H RESTORED (Phase 2.1 docs said "retired" but the kernel still
//   writes to it; the kernel is a pure writer — values are not currently
//   consumed by the Godunov update, but the kernel needs a valid pointer
//   so the d_slope_H buffer is allocated + zeroed in mesh build).
// d_node_invert / d_node_depth / d_node_net_q / d_node_surface_area / d_node_max_depth removed
//   (Phase 2.1 — per-cell state replaces per-node state; manhole cells carry their own invert/depth/etc.)
// d_node_is_boundary / d_node_is_outfall / d_node_is_inlet / d_node_is_pipe_end removed
//   (Phase 2.1 — face_class[] replaces node flag arrays)

    double*   d_A;              // [n_pipe_cells]
    double*   d_Q;              // [n_pipe_cells]
    double*   d_A_prev;         // [n_pipe_cells]
    double*   d_Q_iter;         // [n_pipe_cells]
    double*   d_A_start_save = nullptr; // [n_cells_all] RK2 start-of-step save
    double*   d_Q_start_save = nullptr; // [n_cells_all]
    int32_t   n_start_save_capacity = 0; // tracked capacity of d_A_start_save / d_Q_start_save

    // Persistent scratch buffers for swe2d_pipe1d_step (avoid per-step cudaMalloc/cudaFree)
    double*   d_flux_Q_scratch = nullptr;   // [n_cells_all] per-stage flux accumulator
    double*   d_flux_mom_scratch = nullptr; // [n_cells_all] per-stage momentum flux accumulator
    double*   d_A_new_scratch = nullptr;    // [n_cells_all] godunov output buffer
    double*   d_Q_new_scratch = nullptr;    // [n_cells_all] godunov output buffer

    // Phase 2.1 — Per-cell metadata arrays (sized [n_cells_all])
    int32_t*  d_cell_class = nullptr;        // [n_cells_all] 0=PIPE, 1=MANHOLE, 2=INLET
    double*   d_cell_crown = nullptr;        // [n_cells_all] pipe-crown at manhole (audit F9), 0 for pipe cells
    double*   d_cell_rim = nullptr;          // [n_cells_all] manhole rim elev, 0 for pipe cells
    double*   d_cell_surface_area = nullptr; // [n_cells_all] true horizontal area, 0 for pipe cells
    double*   d_cell_max_depth = nullptr;    // [n_cells_all] max depth before surcharge, +inf for pipe cells

    int32_t   n_pipe_cells = 0;
    int32_t   n_nodes = 0;
    int32_t   n_cells_all = 0;    // = n_pipe_cells + n_manhole_cells + n_inlet_cells
    double    slot_cfl_dt = 1e12; // min CFL-safe dt when slot is active (pre-computed at mesh build)
    int32_t   n_manhole_cells = 0;
    int32_t   n_inlet_cells = 0;

    // d_outfall_mode / d_outfall_fixed_wse / d_outfall_rating / d_outfall_rating_n /
//   d_outfall_tabular / d_outfall_tabular_n / d_outfall_link_idx removed
//   (Phase 2.1 — outfall state now lives on d_ghost_outfall_* per-face SoA; the per-node
//    arrays were read by the now-deleted swe2d_pipe1d_outfall_bc_kernel)

    // d_pipe_end_* / d_n_pipe_ends / d_pipe_end_*_capacity / d_pipe_end_A_open_table /
//   d_pipe_end_depth_bc / d_pipe_end_node_area / n_pipe_ends_capacity removed
//   (Phase 2.1 — pipe-end geometry now lives on per-face arrays; SURFACE_2D_PIPE_END
//    class in the unified face mesh replaces the legacy pipe-end path)
// d_junction_node / d_n_junctions removed
//   (Phase 2.1 — junctions are now MANHOLE_CELL with SURFACE_2D_JUNCTION_OVERFLOW faces)

    // d_debug_bc_face_count / d_debug_int_face_count / d_debug_cell_q_count / d_debug_timestep_marker /
//   d_timestep_counter removed (Phase 2.1 — transient diagnostics retired)
// d_junction_2d_cell / d_junction_overflow_diam / d_junction_overflow_coeff / d_junction_max_overflow
//   removed (Phase 2.1 — class-5 SURFACE_2D_JUNCTION_OVERFLOW faces carry the overflow attributes
//   via face_rim_elev / face_width / etc. on the per-face arrays)
// d_node_rim removed (Phase 2.1 — manhole rim lives on d_cell_rim)

    // --- Phase 2.1 — Per-face arrays (unified face mesh) ---
    int32_t   n_faces = 0;
    int32_t*  d_face_owner_L = nullptr;           // [n_faces]
    int32_t*  d_face_owner_R = nullptr;           // [n_faces]
    int32_t*  d_face_class = nullptr;             // [n_faces] 0=INTERIOR (only class built this phase)
    int32_t*  d_face_solve_mode = nullptr;        // [n_faces] 0=Riemann
    double*   d_face_dir = nullptr;               // [n_faces] ±1.0
    double*   d_face_F_h = nullptr;               // [n_faces] mass flux scratch
    double*   d_face_F_Q = nullptr;               // [n_faces] 1D momentum flux scratch
    double*   d_face_invert = nullptr;            // [n_faces]
    double*   d_face_nx = nullptr;                // [n_faces]
    double*   d_face_ny = nullptr;                // [n_faces]
    double*   d_face_width = nullptr;             // [n_faces]
    double*   d_face_area = nullptr;              // [n_faces]
    double*   d_face_k_in = nullptr;              // [n_faces]
    double*   d_face_k_out = nullptr;             // [n_faces]
    double*   d_face_depth_safety = nullptr;      // [n_faces]
    double*   d_face_rim_elev = nullptr;          // [n_faces]
    double*   d_face_node_surface_area = nullptr; // [n_faces]
    int32_t*  d_face_ghost_idx = nullptr;         // [n_faces] -1 = no ghost

    // Host-side face data cache — uploaded during mesh build, re-uploaded by
    // the face flux kernel if the device copy is found corrupted (CUDA memory
    // pool aliasing between 2D solver graph and pipe1d allocations).
    std::vector<int32_t> h_face_class_cache;
    std::vector<int32_t> h_face_owner_L_cache;
    std::vector<int32_t> h_face_owner_R_cache;

    // --- Phase F3 — HEC-22 inlet capture geometry per SURFACE_2D_INLET face (class 4) ---
    int32_t  n_inlet_capture_faces = 0;
    int32_t* d_face_inlet_type = nullptr;       // 0=GRATE 1=CURB 2=SLOTTED 3=COMBO
    double*  d_face_inlet_grate_len = nullptr;
    double*  d_face_inlet_grate_wid = nullptr;
    double*  d_face_inlet_grate_open = nullptr; // fraction open
    double*  d_face_inlet_curb_len = nullptr;
    double*  d_face_inlet_curb_ht = nullptr;
    double*  d_face_inlet_curb_throat = nullptr;
    double*  d_face_inlet_slot_len = nullptr;
    double*  d_face_inlet_slot_wid = nullptr;
    double*  d_face_inlet_crest = nullptr;      // grate elevation (m abs)
    double*  d_face_inlet_cd = nullptr;         // discharge coefficient
    double*  d_face_inlet_qmax = nullptr;       // max capture rate cap (m³/s)

    // --- Phase F3 — Pre-computed structure flows for class-6 CULVERT faces ---
    double*   d_structure_flows = nullptr;        // [n_structures] m³/s per structure

    // Private CUDA stream + memory pool for pipe1d.  Isolates pipe1d
    // allocations from the global pool shared with the 2D solver, preventing
    // cross-thread free/realloc aliasing that corrupts device arrays.
    cudaStream_t d_stream = nullptr;

    // --- Phase 2.4 — Ghost-state SoA arrays (per-face-class BC data) ---
    int32_t   n_outfall_faces = 0;
    int32_t*  d_ghost_outfall_mode = nullptr;         // [n_outfall_faces] 0=FREE,1=NORMAL_DEPTH,2=FIXED_WSE,3=RATING,4=TABULAR
    double*   d_ghost_outfall_fixed_wse = nullptr;    // [n_outfall_faces] fixed WSE for FIXED_WSE mode
    double*   d_ghost_outfall_rating = nullptr;       // [n_outfall_faces * MAX_RATING_POINTS * 2] SoA: [wse0..wse31, Q0..Q31]
    int32_t*  d_ghost_outfall_rating_n = nullptr;     // [n_outfall_faces] number of rating points per face
    double*   d_ghost_outfall_tabular = nullptr;     // [n_outfall_faces * MAX_TABULAR_POINTS * 2] SoA: [t0..t31, wse0..wse31]
    int32_t*  d_ghost_outfall_tabular_n = nullptr;   // [n_outfall_faces] number of tabular points per face
    double*   d_ghost_outfall_link_S0 = nullptr;      // [n_outfall_faces] normal-depth bed slope
    int32_t*  d_ghost_outfall_node_idx = nullptr;     // [n_outfall_faces] network node index for wse update

    int32_t   n_inlet_bc_faces = 0;
    double*   d_ghost_inlet_Q = nullptr;              // [n_inlet_bc_faces] prescribed flow per step

    int32_t   n_storage_pipe_faces = 0;     // storage→pipe INTERIOR faces
    int32_t   n_culvert_faces = 0;
    int32_t*  d_ghost_culvert_struct_idx = nullptr;   // [n_culvert_faces] index into d_structure_flows

    void destroy() {
        #define _P_FREE(p) do { if (p) { cudaFree(p); p = nullptr; } } while(0)
        _P_FREE(d_cell_from_node); _P_FREE(d_cell_to_node);
        _P_FREE(d_slope_A); _P_FREE(d_slope_Q);
        _P_FREE(d_cell_length); _P_FREE(d_cell_link_length); _P_FREE(d_cell_area);
        _P_FREE(d_cell_perim); _P_FREE(d_cell_invert);
        _P_FREE(d_cell_n); _P_FREE(d_cell_link_k); _P_FREE(d_cell_link_k_in); _P_FREE(d_cell_link_k_out); _P_FREE(d_cell_link_area);
        _P_FREE(d_cell_owner_link); _P_FREE(d_cell_sub_idx); _P_FREE(d_cell_S0); _P_FREE(d_cell_is_end);
        _P_FREE(d_cell_shape_type); _P_FREE(d_cell_width); _P_FREE(d_cell_height); _P_FREE(d_cell_tables);
        _P_FREE(d_cell_y); _P_FREE(d_cell_q); _P_FREE(d_cell_fr); _P_FREE(d_cell_h); _P_FREE(d_cell_slot_width);
        _P_FREE(d_A); _P_FREE(d_Q); _P_FREE(d_A_prev); _P_FREE(d_Q_iter); _P_FREE(d_A_start_save); _P_FREE(d_Q_start_save);
        n_start_save_capacity = 0;
        _P_FREE(d_flux_Q_scratch); _P_FREE(d_flux_mom_scratch); _P_FREE(d_A_new_scratch); _P_FREE(d_Q_new_scratch);
        _P_FREE(d_cell_class); _P_FREE(d_cell_crown); _P_FREE(d_cell_rim);
        _P_FREE(d_cell_surface_area); _P_FREE(d_cell_max_depth);
        n_cells_all = 0; n_manhole_cells = 0; n_inlet_cells = 0; slot_cfl_dt = 1e12;
        _P_FREE(d_face_owner_L); _P_FREE(d_face_owner_R);
        _P_FREE(d_face_class); _P_FREE(d_face_solve_mode);
        _P_FREE(d_face_dir); _P_FREE(d_face_F_h); _P_FREE(d_face_F_Q);
        _P_FREE(d_face_invert); _P_FREE(d_face_nx); _P_FREE(d_face_ny);
        _P_FREE(d_face_width); _P_FREE(d_face_area);
        _P_FREE(d_face_k_in); _P_FREE(d_face_k_out);
        _P_FREE(d_face_depth_safety); _P_FREE(d_face_rim_elev);
        _P_FREE(d_face_node_surface_area); _P_FREE(d_face_ghost_idx);
        _P_FREE(d_face_inlet_type);
        _P_FREE(d_face_inlet_grate_len); _P_FREE(d_face_inlet_grate_wid); _P_FREE(d_face_inlet_grate_open);
        _P_FREE(d_face_inlet_curb_len); _P_FREE(d_face_inlet_curb_ht); _P_FREE(d_face_inlet_curb_throat);
        _P_FREE(d_face_inlet_slot_len); _P_FREE(d_face_inlet_slot_wid);
        _P_FREE(d_face_inlet_crest); _P_FREE(d_face_inlet_cd); _P_FREE(d_face_inlet_qmax);
        n_faces = 0;
        _P_FREE(d_structure_flows);
        // Phase 2.4 — ghost-state SoA cleanup
        _P_FREE(d_ghost_outfall_mode); _P_FREE(d_ghost_outfall_fixed_wse);
        _P_FREE(d_ghost_outfall_rating); _P_FREE(d_ghost_outfall_rating_n);
        _P_FREE(d_ghost_outfall_tabular); _P_FREE(d_ghost_outfall_tabular_n);
        _P_FREE(d_ghost_outfall_link_S0);
        _P_FREE(d_ghost_outfall_node_idx);
        n_outfall_faces = 0;
        _P_FREE(d_ghost_inlet_Q);
        n_inlet_bc_faces = 0;
        _P_FREE(d_ghost_culvert_struct_idx);
        n_storage_pipe_faces = 0;
        n_culvert_faces = 0;
        n_pipe_cells = 0; n_nodes = 0;
        if (d_stream) { cudaStreamDestroy(d_stream); d_stream = nullptr; }
        #undef _P_FREE
    }
};

// ── Kernel declarations ────────────────────────────────────────────────────

/** Mark nodes that have inlet assignments. Defined in pipe1d.cu. @global
    @note This kernel is deprecated — node_is_inlet is derived from inlet_node
          in the binding layer and only used during mesh build. The device
          array d_node_is_inlet was removed in Phase 2.1 (face_class[] replaces
          node flag arrays). This kernel is kept only for API compatibility. */
__global__ void swe2d_mark_inlet_nodes_kernel(
    int32_t n_inlets,
    const int32_t* __restrict__ inlet_node,
    int32_t n_nodes,
    int32_t* __restrict__ node_is_inlet);

// ── Host API declarations ────────────────────────────────────────────────

// Forward declaration of SWE2DDeviceState for functions that take it as a pointer.
struct SWE2DDeviceState;

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
    @param node_inlet_loss_k Per-node inlet loss K [n_nodes] (optional, may be null)
    @param node_outlet_loss_k Per-node outlet loss K [n_nodes] (optional, may be null)
    @param node_surface_area Node surface area [n_nodes]
    @param node_max_depth Node max depth [n_nodes]
    @param link_invert_in Inlet invert [n_links]
    @param link_invert_out Outlet invert [n_links]
    @param max_cell_length Max sub-cell length (0=no subdivision)
    @param link_shape_type Per-link shape codes (optional, nullptr=all circular)
    @param link_width Per-link width (optional)
    @param link_height Per-link height (optional)
    @param dev Output pipe1d state
    @param n_manholes Number of manhole cells (Phase 2.1, default 0)
    @param manhole_node Manhole network node indices [n_manholes]
    @param manhole_cell_length Manhole cell length [n_manholes]
    @param manhole_cell_width Manhole cell width [n_manholes]
    @param manhole_cell_height Manhole cell height (rim - invert) [n_manholes]
    @param n_inlets Number of inlet cells (Phase 2.1, default 0)
    @param inlet_node Inlet network node indices [n_inlets]
    @param inlet_cell_length Inlet cell length [n_inlets]
    @param inlet_cell_width Inlet cell width [n_inlets]
    @param inlet_cell_height Inlet cell height [n_inlets]
    @param node_is_outfall Outfall flag per node [n_nodes] (optional, Phase 2.4)
    @param node_is_inlet Inlet flag per node [n_nodes] (optional, Phase 2.4)
    @param node_is_pipe_end Pipe-end (2D-coupled) flag per node [n_nodes] (optional, Phase 2.4)
    @param n_cells_2d Number of 2D SWE cells (for SURFACE_2D face owner_R, Phase 2.4)
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
    const double*         node_inlet_loss_k,     // per-node inlet loss (optional, may be null)
    const double*         node_outlet_loss_k,    // per-node outlet loss (optional, may be null)
    const double*         node_surface_area,
    const double*         node_max_depth,
    const double*         link_invert_in,
    const double*         link_invert_out,
    const double*         link_max_cell_length,
    double                g_slot_cfl,
    const int32_t*        link_shape_type,
    const double*         link_width,
    const double*         link_height,
    Pipe1DDeviceState*    dev,
    int32_t               n_manholes = 0,
    const int32_t*        manhole_node = nullptr,
    const double*         manhole_cell_length = nullptr,
    const double*         manhole_cell_width = nullptr,
    const double*         manhole_cell_height = nullptr,
    int32_t               n_inlets = 0,
    const int32_t*        inlet_node = nullptr,
    const double*         inlet_cell_length = nullptr,
    const double*         inlet_cell_width = nullptr,
    const double*         inlet_cell_height = nullptr,
    const int32_t*        node_is_outfall = nullptr,
    const int32_t*        node_is_inlet = nullptr,
    const int32_t*        node_is_pipe_end = nullptr,
    int32_t               n_cells_2d = 0,
    // ── Phase F3 — HEC-22 inlet capture (SURFACE_2D_INLET class 4) ──
    int32_t               n_inlet_capture_faces = 0,
    const int32_t*        inlet_face_node = nullptr,
    const int32_t*        inlet_face_2d_cell = nullptr,
    const int32_t*        inlet_face_type = nullptr,
    const double*         inlet_face_grate_len = nullptr,
    const double*         inlet_face_grate_wid = nullptr,
    const double*         inlet_face_grate_open = nullptr,
    const double*         inlet_face_curb_len = nullptr,
    const double*         inlet_face_curb_ht = nullptr,
    const double*         inlet_face_curb_throat = nullptr,
    const double*         inlet_face_slot_len = nullptr,
    const double*         inlet_face_slot_wid = nullptr,
    const double*         inlet_face_crest = nullptr,
    const double*         inlet_face_cd = nullptr,
    const double*         inlet_face_qmax = nullptr,
    // ── Phase F3 — Culvert faces (class 6) ──
    int32_t               n_culvert_faces = 0,
    const int32_t*        culvert_face_donor_2d_cell = nullptr,
    const int32_t*        culvert_face_receiver_2d_cell = nullptr,
    const int32_t*        culvert_struct_idx = nullptr,
    int32_t               n_structures = 0,
    const double*         structure_flows = nullptr);

// DEPRECATED Phase 2.2 — Replaced by swe2d_unified_face_flux_kernel (face-based).
// void swe2d_pipe1d_flux_kernel_host(...) — removed.

// REMOVED Phase 2.1 — swe2d_pipe1d_diffusion_wave_kernel_host (dormant, replaced by
//   swe2d_pipe1d_godunov_update_kernel in unified face-mesh path).
// REMOVED Phase 2.1 — swe2d_pipe1d_fully_dynamic_kernel_host (dormant, same rationale).

/** Host wrapper: advance 1D pipe network one coupling step.
    Orchestrates flux kernel + update kernel (diffusion or fully dynamic) in sequence.
    @param dev Device state pointer (with pipe1d state initialized)
    @param dt Coupling timestep for this substep
    @param solver_mode "diffusion_wave" or "fully_dynamic"
    @param coupling_substeps Number of substeps within this coupling step
    @param implicit_iters Number of Picard iterations per substep (for fully_dynamic)
    @param relaxation Relaxation factor for Picard iteration (0-1)
    @param g Gravitational acceleration
    @param k_mann Manning unit conversion factor (1.0 for SI, 1.486 for USC)
    @param h_min Minimum flow depth
    @param surcharge_method Surcharge method (0=none, 1=SLOT)
    @param theta Implicit factor for piezometric-head gradient in the Casulli-style
                  semi-implicit momentum update (1.0 = first-order backward-Euler;
                  0.5 = Crank-Nicolson).  Default 1.0.
    @param omega_min Floor for the friction coefficient gamma in the implicit
                     (1 + gamma*dt) denominator; corresponds to OMEGA_MIN in
                     swe2d_xsect_constants.h.  Default 1e-6.
    @param friction_method 0=NONE, 1=SUBSTEPPING (2D-style), 2=ALPHA_BOOST.
                     Controls which semi-implicit friction treatment is applied
                     in the godunov update kernel.  Default 0.
    @host */
void swe2d_pipe1d_step(
    SWE2DDeviceState* dev,
    double            dt,
    const char*       solver_mode,
    int32_t           coupling_substeps,
    int32_t           implicit_iters,
    double            relaxation,
    double            g,
    double            k_mann,
    double            h_min,
    int32_t           surcharge_method,
    double            theta            = 1.0,
    double            omega_min        = 1e-6,
    int32_t           friction_method  = 0,
    int32_t           recon_method     = 0,
    int32_t           time_integrator  = 1,
    double            friction_alpha   = 0.01,
    SWE2DDeviceState* solver_dev      = nullptr);

// REMOVED Phase 2.1:
//   swe2d_pipe1d_upload_node_depth          (per-node depth retired)
//   swe2d_pipe1d_upload_pipe_ends_and_junctions  (pipe-end geometry on per-face arrays now)
//   swe2d_pipe1d_upload_outfall_state       (outfall state on per-face d_ghost_outfall_* now)
//   swe2d_pipe1d_upload_junction_overflow_state  (class-5 face handles overflow now)
//   swe2d_pipe1d_upload_node_rim            (rim lives on d_cell_rim now)
//   swe2d_pipe1d_readback_node_state        (replaced by swe2d_pipe1d_readback_cell_state)
//   swe2d_junction_bc_kernel_host           (clamping moved into godunov update kernel)
//   swe2d_pipe1d_junction_overflow_kernel_host  (class-5 face handles overflow now)
//   swe2d_pipe_end_weir_orifice_kernel_host (pipe-end exchange is SURFACE_2D_PIPE_END face class)
//   swe2d_mark_inlet_nodes_kernel           (face_class[] replaces node flag arrays)
//   swe2d_fold_pipe_end_q_to_source_kernel  (unified face kernel writes ext_struct_flux directly)

/** Initialize cell area from per-cell depth (d_cell_h) for ALL cell classes.
    Reads d_cell_h, computes A(h) per shape_type, writes to d_A, d_A_prev,
    and d_cell_y = d_cell_invert + d_cell_h.
    @param dev Pipe1D device state (uses n_cells_all, d_cell_h, geometry arrays)
    @param h_min Wet/dry depth floor (m) */
void swe2d_pipe1d_init_cell_area(Pipe1DDeviceState* dev, double h_min);

/** Fold junction overflow Q directly into 2D cell depth h. @global */
__global__ void swe2d_fold_junction_overflow_to_h_kernel(
    int32_t n_cells,
    const double* __restrict__ d_Q,
    const double* __restrict__ d_cell_area,
    double dt,
    double* __restrict__ d_h);
