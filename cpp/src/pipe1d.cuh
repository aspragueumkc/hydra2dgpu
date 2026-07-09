#pragma once
// pipe1d.cuh
// 1D pipe network device state and host API declarations for the GPU solver.
// Split from swe2d_gpu.cuh — mechanical refactoring, no behavior changes.

#include <cstdint>
#include <cuda_runtime.h>

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

    int32_t*  d_cell_shape_type; // [n_pipe_cells] 0=circular 1=rect 2=ellipse
    double*   d_cell_width;      // [n_pipe_cells]
    double*   d_cell_height;     // [n_pipe_cells]
    double*   d_cell_tables;     // [n_pipe_cells × 2 × PIPE1D_TABLE_N] flattened P_ratio + T_ratio

    double*   d_node_invert;    // [n_nodes] invert elevation at each node
    double*   d_node_depth;     // [n_nodes]
    double*   d_node_net_q;     // [n_nodes]
    double*   d_node_surface_area; // [n_nodes]
    double*   d_node_max_depth;    // [n_nodes] crown elevation

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
        _P_FREE(d_cell_shape_type); _P_FREE(d_cell_width); _P_FREE(d_cell_height); _P_FREE(d_cell_tables);
        _P_FREE(d_node_invert); _P_FREE(d_node_depth); _P_FREE(d_node_net_q); _P_FREE(d_node_surface_area);
        _P_FREE(d_node_max_depth);
        _P_FREE(d_A); _P_FREE(d_Q); _P_FREE(d_A_prev); _P_FREE(d_Q_iter);
        n_pipe_cells = 0; n_nodes = 0;
        #undef _P_FREE
    }
};

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
    @param node_surface_area Node surface area [n_nodes]
    @param node_max_depth Node max depth [n_nodes]
    @param link_invert_in Inlet invert [n_links]
    @param link_invert_out Outlet invert [n_links]
    @param max_cell_length Max sub-cell length (0=no subdivision)
    @param link_shape_type Per-link shape codes (optional, nullptr=all circular)
    @param link_width Per-link width (optional)
    @param link_height Per-link height (optional)
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
    const int32_t*        link_shape_type,
    const double*         link_width,
    const double*         link_height,
    Pipe1DDeviceState*    dev);

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
    const double*         node_max_depth,
    const double*         cell_length,
    const double*         cell_height,
    double*               flux_Q_out,
    double                g,
    const double*         cell_tables,
    int32_t               table_N,
    int32_t               volume_decomposition);

/** GPU kernel: explicit diffusion-wave update for 1D pipe network.
    @param n_cells Number of pipe cells
    @param cell_length Cell length [n_cells]
    @param cell_area_full Full pipe cross-section area [n_cells]
    @param cell_perim Pipe perimeter [n_cells]
    @param cell_n Manning's n [n_cells]
    @param cell_k_loss Minor loss K at boundary cells [n_cells] (0 interior)
    @param cell_from_node From-node index per cell [n_cells]
    @param cell_to_node To-node index per cell [n_cells]
    @param node_invert Node invert elevation [n_nodes]
    @param node_depth Node water depth [n_nodes]
    @param cell_A Current area [n_cells]
    @param cell_Q Current discharge [n_cells]
    @param flux_Q Net flux OUT of cell [n_cells]
    @param dt Time step
    @param g Gravitational acceleration
    @param cell_A_new Updated area [n_cells]
    @param cell_Q_new Updated discharge [n_cells]
    @param cell_tables Geometry lookup tables [n_cells * 2 * table_N]
    @param table_N Number of entries per table
    @host */
void swe2d_pipe1d_diffusion_wave_kernel_host(
    int32_t               n_cells,
    const double*         cell_length,
    const double*         cell_area_full,
    const double*         cell_perim,
    const double*         cell_n,
    const double*         cell_k_loss,
    const int32_t*        cell_from_node,
    const int32_t*        cell_to_node,
    const double*         node_invert,
    const double*         node_depth,
    const double*         cell_A,
    const double*         cell_Q,
    const double*         flux_Q,
    double                dt,
    double                g,
    double*               cell_A_new,
    double*               cell_Q_new,
    const double*         cell_tables,
    int32_t               table_N);

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

/** Initialize pipe cell area from uploaded node depths. @host */
void swe2d_pipe1d_init_area_from_depth(Pipe1DDeviceState* dev);

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

/** Host wrapper: pipe-end boundary condition kernel.
    Sets node depth from surface WSE using pipe exit loss coefficient.
    @host */
void swe2d_drainage_pipe_end_bc_kernel_host(
    int32_t n_pipe_ends, int32_t n_cells,
    const int32_t* pipe_end_cell, const int32_t* pipe_end_node,
    const double* pipe_end_invert, const double* pipe_end_diameter,
    const double* pipe_end_area,
    const double* pipe_end_kin, const double* pipe_end_kout,
    const double* cell_wse, const double* node_invert,
    const double* node_surface_area, const double* node_qleave,
    double gravity,
    double* node_depth, double* pipe_end_depth_bc, double* pipe_end_node_area);

/** Host wrapper: pipe-end exchange kernel.
    Applies net node discharge from pipe solver as surface cell source/sink.
    @host */
void swe2d_drainage_pipe_end_exchange_kernel_host(
    int32_t n_pipe_ends, int32_t n_cells,
    const int32_t* pipe_end_cell, const int32_t* pipe_end_node,
    const double* pipe_end_node_area,
    const double* cell_area, const double* cell_depth,
    const double* node_net_q,
    double dt_s,
    double* q_cell,
    double* limiter_event_count, double* limiter_volume_m3,
    const int32_t* pipe_end_enable_overflow,
    const double*  pipe_end_max_overflow_rate);
