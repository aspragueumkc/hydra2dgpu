#pragma once
// swe2d_mesh.hpp
// Unstructured polygon mesh for the 2D SWE solver.
// All arrays use Structure-of-Arrays (SoA) layout for GPU coalescing and CPU SIMD.
//
// Coordinate system: standard (x east, y north). All distances in metres.
// Cell orientation: counter-clockwise (CCW) node ordering enforced by the builder.

#include <cstdint>
#include <vector>
#include <string>

// ─────────────────────────────────────────────────────────────────────────────
// Boundary condition types
// ─────────────────────────────────────────────────────────────────────────────
enum class BCType : int32_t {
    INTERIOR = 0,   // Interior edge — not a boundary
    WALL     = 1,   // Solid wall: zero normal flux
    INFLOW_Q = 2,   // Prescribed unit discharge normal to edge (m²/s)
    STAGE    = 3,   // Prescribed water-surface elevation (m)
    OPEN     = 4,   // Riemann outflow / zero-gradient
    REFLECT  = 5,   // Reflecting: reverse normal velocity component
    NORMAL_DEPTH = 6, // Prescribed boundary depth h (m)
    NORMAL_DEPTH_SLOPE = 7 // Friction-slope normal depth (bc_val = Sf)
};

// ─────────────────────────────────────────────────────────────────────────────
// Mesh structure
// ─────────────────────────────────────────────────────────────────────────────
// Ownership: SWE2DMesh owns all arrays via std::vector<>.
// When passing to CUDA, raw device pointers are managed separately by
// SWE2DDeviceState (swe2d_gpu.cuh).
struct SWE2DMesh {
    // ── Nodes ────────────────────────────────────────────────────────────────
    int32_t              n_nodes = 0;
    std::vector<double>  node_x;      // [n_nodes] easting (m)
    std::vector<double>  node_y;      // [n_nodes] northing (m)
    std::vector<double>  node_z;      // [n_nodes] bed elevation (m)

    // ── Cells (polygons) ─────────────────────────────────────────────────────
    int32_t              n_cells = 0;
    std::vector<int32_t> cell_face_offsets; // [n_cells + 1], CSR offsets into cell_face_nodes
    std::vector<int32_t> cell_face_nodes;   // [sum(n_verts_cell)] node indices for each cell ring
    std::vector<int32_t> cell_edge_offsets;  // [n_cells + 1], CSR offsets into cell_edge_ids
    std::vector<int32_t> cell_edge_ids;      // [sum(n_verts_cell)] edge indices for each cell ring
    std::vector<double>  cell_cx;     // [n_cells] centroid x (m)
    std::vector<double>  cell_cy;     // [n_cells] centroid y (m)
    std::vector<double>  cell_area;   // [n_cells] area (m²)
    std::vector<double>  cell_zb;     // [n_cells] bed elevation at centroid (m)

    // ── Edges ────────────────────────────────────────────────────────────────
    // edge_c0 is always the cell that "owns" the edge normal direction.
    // For interior edges edge_c1 >= 0; for boundary edges edge_c1 == -1.
    int32_t              n_edges = 0;
    std::vector<int32_t> edge_c0;      // [n_edges] left cell index
    std::vector<int32_t> edge_c1;      // [n_edges] right cell index (-1 = boundary)
    std::vector<int32_t> edge_n0;      // [n_edges] first endpoint node
    std::vector<int32_t> edge_n1;      // [n_edges] second endpoint node
    std::vector<double>  edge_nx;      // [n_edges] outward unit normal x (c0→exterior)
    std::vector<double>  edge_ny;      // [n_edges] outward unit normal y
    std::vector<double>  edge_len;     // [n_edges] edge length (m)
    std::vector<BCType>  edge_bc;      // [n_edges] boundary condition type
    std::vector<double>  edge_bc_val;  // [n_edges] prescribed BC value (context-dependent)

    // ── Derived per-cell geometry (built once, used by solver) ───────────────
    // For each cell c, inverse area is stored for the update step.
    std::vector<double>  cell_inv_area; // [n_cells] 1/area (m⁻²)
};

// ─────────────────────────────────────────────────────────────────────────────
// Builder
// ─────────────────────────────────────────────────────────────────────────────

// Build a complete SWE2DMesh from raw Python-side arrays.
//
// Parameters:
//   node_x, node_y, node_z  : node coordinate arrays (length n_nodes)
//   cell_face_offsets        : CSR offsets for polygon cells, length n_cells+1
//   cell_face_nodes          : concatenated CCW node rings for all cells
//   n_nodes, n_cells         : counts
//   bc_edge_node0/1          : arrays of length n_bc_edges defining boundary edge endpoints
//   bc_edge_type             : BCType per boundary edge
//   bc_edge_val              : prescribed value per boundary edge
//   n_bc_edges               : number of explicitly specified boundary edges
//                              (any boundary edge not listed defaults to WALL)
//
// Returns: populated SWE2DMesh or throws std::runtime_error on validation failure.
SWE2DMesh swe2d_build_mesh_poly(
    const double*  node_x,
    const double*  node_y,
    const double*  node_z,
    int32_t        n_nodes,
    const int32_t* cell_face_offsets, // length n_cells + 1
    const int32_t* cell_face_nodes,   // length cell_face_offsets[n_cells]
    int32_t        n_cells,
    const int32_t* bc_edge_node0,  // length n_bc_edges (may be nullptr if n_bc_edges==0)
    const int32_t* bc_edge_node1,
    const int32_t* bc_edge_type,
    const double*  bc_edge_val,
    int32_t        n_bc_edges
);

// Reorder edges by (c0, c1) to improve GPU memory coalescing.
// Edges sharing the same c0 cell become contiguous, so that adjacent warp
// threads in the flux/gradient kernels read from the same cell for c0.
// All edge arrays are permuted in-place; cell_edge_ids are remapped.
// Safe to call on any mesh.  No-op for empty meshes.
void swe2d_reorder_edges_for_gpu(SWE2DMesh& mesh);

// Legacy triangle builder retained for backward compatibility.
SWE2DMesh swe2d_build_mesh(
    const double*  node_x,
    const double*  node_y,
    const double*  node_z,
    int32_t        n_nodes,
    const int32_t* cell_nodes,   // length 3 * n_cells
    int32_t        n_cells,
    const int32_t* bc_edge_node0,
    const int32_t* bc_edge_node1,
    const int32_t* bc_edge_type,
    const double*  bc_edge_val,
    int32_t        n_bc_edges
);

// Validate mesh consistency.  Returns empty string on success, error message on failure.
std::string swe2d_validate_mesh(const SWE2DMesh& mesh);
