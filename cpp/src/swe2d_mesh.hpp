#pragma once
// swe2d_mesh.hpp
// Unstructured polygon mesh for the 2D SWE solver.
// All arrays use Structure-of-Arrays (SoA) layout for GPU coalescing and CPU SIMD.
//
// Coordinate system: standard (x east, y north). All distances in metres.
// Cell orientation: counter-clockwise (CCW) node ordering enforced by the builder.

#include <cstddef>
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

    // ── Cell renumbering permutation (reverse Cuthill-McKee BFS) ──────────
    // cell_perm[c_new] = c_old.  Empty if not applied.
    std::vector<int32_t> cell_perm;

    // ── 2-ring cell stencil (CSR) — used by least-squares gradient (scheme 6) ─
    // For each cell, the set of unique neighbour cells reachable via 1 or 2
    // interior-edge traversals (excluding self).  Precomputed Δx/Δy to each
    // neighbour centroid and inverse-distance² weights feed the LSQ gradient.
    std::vector<int32_t> cell_ring2_offsets;   // [n_cells + 1] CSR offsets
    std::vector<int32_t> cell_ring2_ids;       // [sum(ring2_counts)] neighbour cell indices
    std::vector<double>  cell_ring2_dcx;        // [sum(ring2_counts)] Δx = cx[j]-cx[c]
    std::vector<double>  cell_ring2_dcy;        // [sum(ring2_counts)] Δy = cy[j]-cy[c]
    std::vector<double>  cell_ring2_inv_dist2;  // [sum(ring2_counts)] 1/|Δr|² weight

    // ── WENO3 face sub-stencil (scheme 6) ──────────────────────────────────
    // Per-face variable-length CSR tables for the three WENO sub-stencils.
    // S1 = {owner, neighbor} is stored as a flat [2*n_faces] pair array.
    std::vector<int32_t> face_stencil_S0_offsets;  // [n_faces + 1] prefix-sum
    std::vector<int32_t> face_stencil_S0_cells;    // variable-length upwind lobe cells
    std::vector<int32_t> face_stencil_S1;          // [2 * n_faces] = {owner, neighbor}
    std::vector<int32_t> face_stencil_S2_offsets;  // [n_faces + 1] prefix-sum
    std::vector<int32_t> face_stencil_S2_cells;    // variable-length downwind lobe cells

    // ── MP5 5-cell walk (scheme 8) ─────────────────────────────────────────
    std::vector<int32_t> face_stencil_5;           // [5 * n_faces] = {u2, u1, u, v, v1}
    std::vector<int32_t> face_mp5_case;            // [n_faces] case ∈ {1,2,3,4}

    // Convenience
    bool has_stencil_data() const { return !face_stencil_S0_offsets.empty(); }
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
/** Build a complete SWE2DMesh from raw Python-side arrays (polygon cells).
    @param node_x Node x coordinates [n_nodes] @param node_y Node y coordinates
    @param node_z Node bed elevations @param n_nodes Node count
    @param cell_face_offsets CSR offsets for polygon cells, length n_cells+1
    @param cell_face_nodes Concatenated CCW node rings for all cells
    @param n_cells Cell count @param bc_edge_node0 Boundary edge start nodes [n_bc_edges]
    @param bc_edge_node1 Boundary edge end nodes [n_bc_edges] @param bc_edge_type BC types
    @param bc_edge_val Prescribed BC values @param n_bc_edges Boundary edge count
    @returns Populated SWE2DMesh @throws std::runtime_error on validation failure */
SWE2DMesh swe2d_build_mesh_poly(
    const double*  node_x,
    const double*  node_y,
    const double*  node_z,
    int32_t        n_nodes,
    const int32_t* cell_face_offsets,
    const int32_t* cell_face_nodes,
    int32_t        n_cells,
    const int32_t* bc_edge_node0,
    const int32_t* bc_edge_node1,
    const int32_t* bc_edge_type,
    const double*  bc_edge_val,
    int32_t        n_bc_edges
);

/** Renumber cells via reverse Cuthill-McKee BFS for GPU cache locality.
    Cells connected by edges get nearby indices. Edge c0/c1 entries are
    remapped through the permutation.  Stores the permutation in mesh.cell_perm.
    Safe to call on any mesh.  No-op for empty meshes. */
void swe2d_renumber_cells_for_gpu(SWE2DMesh& mesh);

/** Reorder edges by (c0, c1) for GPU memory coalescing.
    Edges sharing the same c0 cell become contiguous for warp-coherent
    cell reads in flux/gradient kernels. All edge arrays permuted in-place;
    cell_edge_ids remapped. Safe to call on any mesh. No-op for empty meshes. */
void swe2d_reorder_edges_for_gpu(SWE2DMesh& mesh);

/** Build the 2-ring cell stencil (CSR) for least-squares gradient (scheme 6).
    For each cell, collects unique cells reachable via 1 or 2 interior-edge
    traversals (excluding self), and precomputes Dx, Dy, and inverse-dist^2
    to each neighbour centroid. */
void swe2d_build_cell_ring2(SWE2DMesh& mesh);

/// Validate mesh consistency. Returns empty string on success, error message on failure.
std::string swe2d_validate_mesh(const SWE2DMesh& mesh);

// ─────────────────────────────────────────────────────────────────────────────
// Mesh BLOB serialization (raw binary, no zlib)
// ─────────────────────────────────────────────────────────────────────────────

/** Serialize a fully-constructed SWE2DMesh into a byte buffer.
    Format: 8-byte count prefix + raw element bytes for each std::vector member,
    preceded by 3 int32 scalars (n_nodes, n_cells, n_edges).
    Returns a byte vector suitable for GPKG BLOB storage. */
std::vector<uint8_t> swe2d_serialize_mesh(const SWE2DMesh& mesh);

/** Deserialize a SWE2DMesh from a byte buffer produced by swe2d_serialize_mesh.
    @param data Pointer to the buffer @param size Buffer size in bytes
    @returns Fully populated SWE2DMesh (identical to the original)
    @throws std::runtime_error on invalid/corrupt data */
SWE2DMesh swe2d_deserialize_mesh(const uint8_t* data, size_t size);

/** Legacy triangle mesh builder retained for backward compatibility.
    Wraps swe2d_build_mesh_poly with 3-vertex-per-cell offsets. */
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


