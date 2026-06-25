// swe2d_mesh.cpp
// Unstructured polygon mesh builder and validator.

#include "swe2d_mesh.hpp"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cmath>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

/// Canonical edge key: (min_node, max_node) packed into a 64-bit int.
static inline int64_t edge_key(int32_t a, int32_t b) {
    int32_t lo = (a < b) ? a : b;
    int32_t hi = (a < b) ? b : a;
    return (static_cast<int64_t>(lo) << 32) | static_cast<int64_t>(hi);
}

/** Compute centroid and signed area of a simple polygon ring.
    @param node_x Node x coords @param node_y Node y coords
    @param ring Node index ring @param cx [out] Centroid x @param cy [out] Centroid y
    @returns Signed area; positive means CCW orientation */
static double polygon_centroid_area(
    const std::vector<double>& node_x,
    const std::vector<double>& node_y,
    const std::vector<int32_t>& ring,
    double& cx,
    double& cy)
{
    const size_t n = ring.size();
    if (n < 3) {
        cx = 0.0;
        cy = 0.0;
        return 0.0;
    }

    double twice_area = 0.0;
    double cx_num = 0.0;
    double cy_num = 0.0;
    for (size_t i = 0; i < n; ++i) {
        int32_t ia = ring[i];
        int32_t ib = ring[(i + 1) % n];
        double xa = node_x[ia];
        double ya = node_y[ia];
        double xb = node_x[ib];
        double yb = node_y[ib];
        double cross = xa * yb - xb * ya;
        twice_area += cross;
        cx_num += (xa + xb) * cross;
        cy_num += (ya + yb) * cross;
    }

    if (std::abs(twice_area) <= 1.0e-15) {
        cx = 0.0;
        cy = 0.0;
        return 0.0;
    }

    cx = cx_num / (3.0 * twice_area);
    cy = cy_num / (3.0 * twice_area);
    return 0.5 * twice_area;
}

/** Build a complete SWE2DMesh from raw polygon cell topology. */
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
    int32_t        n_bc_edges)
{
    SWE2DMesh mesh;
    mesh.n_nodes = n_nodes;
    mesh.n_cells = n_cells;

    if (n_nodes <= 0) {
        throw std::runtime_error("swe2d_build_mesh_poly: n_nodes must be > 0");
    }
    if (n_cells <= 0) {
        throw std::runtime_error("swe2d_build_mesh_poly: n_cells must be > 0");
    }
    if (cell_face_offsets == nullptr || cell_face_nodes == nullptr) {
        throw std::runtime_error("swe2d_build_mesh_poly: cell_face_offsets and cell_face_nodes are required");
    }

    // Copy node arrays.
    mesh.node_x.assign(node_x, node_x + n_nodes);
    mesh.node_y.assign(node_y, node_y + n_nodes);
    mesh.node_z.assign(node_z, node_z + n_nodes);

    // Copy polygon-cell topology (CSR).
    mesh.cell_face_offsets.assign(cell_face_offsets, cell_face_offsets + n_cells + 1);
    int32_t n_face_nodes = mesh.cell_face_offsets.back();
    if (n_face_nodes < 0) {
        throw std::runtime_error("swe2d_build_mesh_poly: invalid cell_face_offsets tail");
    }
    mesh.cell_face_nodes.assign(cell_face_nodes, cell_face_nodes + n_face_nodes);

    mesh.cell_cx.resize(n_cells);
    mesh.cell_cy.resize(n_cells);
    mesh.cell_area.resize(n_cells);
    mesh.cell_zb.resize(n_cells);
    mesh.cell_inv_area.resize(n_cells);
    mesh.cell_edge_offsets.resize(static_cast<size_t>(n_cells) + 1, 0);

    for (int32_t c = 0; c < n_cells; ++c) {
        int32_t s = mesh.cell_face_offsets[c];
        int32_t e = mesh.cell_face_offsets[c + 1];
        if (s < 0 || e < s || e > n_face_nodes) {
            std::ostringstream oss;
            oss << "swe2d_build_mesh_poly: invalid face offsets for cell " << c;
            throw std::runtime_error(oss.str());
        }
        if (e - s < 3) {
            std::ostringstream oss;
            oss << "swe2d_build_mesh_poly: cell " << c << " has fewer than 3 vertices";
            throw std::runtime_error(oss.str());
        }

        std::vector<int32_t> ring;
        ring.reserve(static_cast<size_t>(e - s));
        for (int32_t k = s; k < e; ++k) {
            int32_t nidx = mesh.cell_face_nodes[k];
            if (nidx < 0 || nidx >= n_nodes) {
                std::ostringstream oss;
                oss << "swe2d_build_mesh_poly: cell " << c << " has out-of-range node index " << nidx;
                throw std::runtime_error(oss.str());
            }
            ring.push_back(nidx);
        }

        double cx = 0.0;
        double cy = 0.0;
        double signed_area = polygon_centroid_area(mesh.node_x, mesh.node_y, ring, cx, cy);

        // Enforce CCW orientation for stable outward normals.
        if (signed_area < 0.0) {
            std::reverse(ring.begin(), ring.end());
            std::copy(ring.begin(), ring.end(), mesh.cell_face_nodes.begin() + s);
            signed_area = polygon_centroid_area(mesh.node_x, mesh.node_y, ring, cx, cy);
        }
        if (signed_area <= 0.0) {
            std::ostringstream oss;
            oss << "swe2d_build_mesh_poly: degenerate cell " << c << " has non-positive area";
            throw std::runtime_error(oss.str());
        }

        mesh.cell_cx[c] = cx;
        mesh.cell_cy[c] = cy;
        mesh.cell_area[c] = signed_area;
        mesh.cell_inv_area[c] = 1.0 / signed_area;

        // Bed elevation proxy at centroid: average vertex bed elevation.
        double zb_sum = 0.0;
        for (int32_t nidx : ring) {
            zb_sum += mesh.node_z[nidx];
        }
        mesh.cell_zb[c] = zb_sum / static_cast<double>(ring.size());
    }

    // Build explicit BC lookup: (min_n, max_n) -> (type, val).
    std::unordered_map<int64_t, std::pair<BCType, double>> bc_map;
    if (n_bc_edges > 0 && bc_edge_node0 != nullptr) {
        for (int32_t b = 0; b < n_bc_edges; ++b) {
            int64_t key = edge_key(bc_edge_node0[b], bc_edge_node1[b]);
            BCType bct = static_cast<BCType>(bc_edge_type[b]);
            double bcv = (bc_edge_val != nullptr) ? bc_edge_val[b] : 0.0;
            bc_map[key] = {bct, bcv};
        }
    }

    // Build edge connectivity from polygon rings.
    struct EdgeEntry {
        int32_t edge_idx;
    };
    std::unordered_map<int64_t, EdgeEntry> edge_map;
    edge_map.reserve(mesh.cell_face_nodes.size());

    mesh.edge_c0.reserve(mesh.cell_face_nodes.size());
    mesh.edge_c1.reserve(mesh.cell_face_nodes.size());
    mesh.edge_n0.reserve(mesh.cell_face_nodes.size());
    mesh.edge_n1.reserve(mesh.cell_face_nodes.size());
    mesh.edge_nx.reserve(mesh.cell_face_nodes.size());
    mesh.edge_ny.reserve(mesh.cell_face_nodes.size());
    mesh.edge_len.reserve(mesh.cell_face_nodes.size());
    mesh.edge_bc.reserve(mesh.cell_face_nodes.size());
    mesh.edge_bc_val.reserve(mesh.cell_face_nodes.size());
    mesh.cell_edge_ids.reserve(mesh.cell_face_nodes.size());

    int32_t n_edges = 0;
    for (int32_t c = 0; c < n_cells; ++c) {
        int32_t s = mesh.cell_face_offsets[c];
        int32_t e = mesh.cell_face_offsets[c + 1];
        int32_t nv = e - s;
        mesh.cell_edge_offsets[static_cast<size_t>(c)] = static_cast<int32_t>(mesh.cell_edge_ids.size());
        for (int32_t i = 0; i < nv; ++i) {
            int32_t na = mesh.cell_face_nodes[s + i];
            int32_t nb = mesh.cell_face_nodes[s + ((i + 1) % nv)];
            int64_t key = edge_key(na, nb);

            auto it = edge_map.find(key);
            if (it == edge_map.end()) {
                double dx = mesh.node_x[nb] - mesh.node_x[na];
                double dy = mesh.node_y[nb] - mesh.node_y[na];
                double len = std::sqrt(dx * dx + dy * dy);
                if (len <= 0.0) {
                    std::ostringstream oss;
                    oss << "swe2d_build_mesh_poly: zero-length edge between nodes " << na << " and " << nb;
                    throw std::runtime_error(oss.str());
                }

                // Outward normal for a CCW ring is clockwise rotation.
                double nx = dy / len;
                double ny = -dx / len;

                mesh.edge_c0.push_back(c);
                mesh.edge_c1.push_back(-1);
                mesh.edge_n0.push_back(na);
                mesh.edge_n1.push_back(nb);
                mesh.edge_nx.push_back(nx);
                mesh.edge_ny.push_back(ny);
                mesh.edge_len.push_back(len);
                mesh.edge_bc.push_back(BCType::INTERIOR);
                mesh.edge_bc_val.push_back(0.0);

                edge_map[key] = EdgeEntry{n_edges};
                mesh.cell_edge_ids.push_back(n_edges);
                ++n_edges;
            } else {
                int32_t eidx = it->second.edge_idx;
                if (mesh.edge_c1[eidx] != -1) {
                    std::ostringstream oss;
                    oss << "swe2d_build_mesh_poly: non-manifold edge between nodes " << na << " and " << nb;
                    throw std::runtime_error(oss.str());
                }
                mesh.edge_c1[eidx] = c;
                mesh.edge_bc[eidx] = BCType::INTERIOR;
                mesh.cell_edge_ids.push_back(eidx);
            }
        }
    }

    mesh.n_edges = n_edges;
    mesh.cell_edge_offsets[static_cast<size_t>(n_cells)] = static_cast<int32_t>(mesh.cell_edge_ids.size());

    // Classify boundary edges.
    for (int32_t e = 0; e < mesh.n_edges; ++e) {
        if (mesh.edge_c1[e] == -1) {
            int64_t key = edge_key(mesh.edge_n0[e], mesh.edge_n1[e]);
            auto it = bc_map.find(key);
            if (it != bc_map.end()) {
                mesh.edge_bc[e] = it->second.first;
                mesh.edge_bc_val[e] = it->second.second;
            } else {
                mesh.edge_bc[e] = BCType::WALL;
                mesh.edge_bc_val[e] = 0.0;
            }
        }
    }

    // Renumber cells for GPU cache locality (reverse Cuthill-McKee BFS).
    swe2d_renumber_cells_for_gpu(mesh);

    // Reorder edges by (c0, c1) for GPU memory coalescing.
    swe2d_reorder_edges_for_gpu(mesh);

    // Build the 2-ring cell stencil (CSR) used by the least-squares gradient
    // (spatial scheme 6 / FV_WENO5).  Neighbour cell indices are invariant under
    // the edge reordering above, so this may run before or after the reorder.
    swe2d_build_cell_ring2(mesh);

    return mesh;
}

/// Legacy triangle builder — builds 3-vertex offsets and delegates to swe2d_build_mesh_poly.
SWE2DMesh swe2d_build_mesh(
    const double*  node_x,
    const double*  node_y,
    const double*  node_z,
    int32_t        n_nodes,
    const int32_t* cell_nodes,
    int32_t        n_cells,
    const int32_t* bc_edge_node0,
    const int32_t* bc_edge_node1,
    const int32_t* bc_edge_type,
    const double*  bc_edge_val,
    int32_t        n_bc_edges)
{
    std::vector<int32_t> offsets(static_cast<size_t>(n_cells) + 1, 0);
    for (int32_t c = 0; c < n_cells; ++c) {
        offsets[static_cast<size_t>(c) + 1] = offsets[static_cast<size_t>(c)] + 3;
    }

    return swe2d_build_mesh_poly(
        node_x,
        node_y,
        node_z,
        n_nodes,
        offsets.data(),
        cell_nodes,
        n_cells,
        bc_edge_node0,
        bc_edge_node1,
        bc_edge_type,
        bc_edge_val,
        n_bc_edges);
}

// ─────────────────────────────────────────────────────────────────────────────
// Edge reordering for GPU memory coalescing
// ─────────────────────────────────────────────────────────────────────────────
/** Reorder edges by (c0, c1) for GPU memory coalescing.
    Builds permutation, applies to all edge arrays, remaps cell_edge_ids. */
void swe2d_reorder_edges_for_gpu(SWE2DMesh& mesh) {
    const int32_t n_edges = mesh.n_edges;
    if (n_edges <= 0) return;

    // 1. Build permutation: sort edges by (c0, c1).
    //    Boundary edges (c1 == -1) sort after all interior edges.
    std::vector<int32_t> perm(static_cast<size_t>(n_edges));
    std::iota(perm.begin(), perm.end(), 0);
    std::sort(perm.begin(), perm.end(), [&](int32_t a, int32_t b) {
        const int32_t c0a = mesh.edge_c0[a];
        const int32_t c0b = mesh.edge_c0[b];
        if (c0a != c0b) return c0a < c0b;
        // c1 == -1 (boundary) sorts after all interior edges
        const int32_t c1a = (mesh.edge_c1[a] >= 0) ? mesh.edge_c1[a] : INT32_MAX;
        const int32_t c1b = (mesh.edge_c1[b] >= 0) ? mesh.edge_c1[b] : INT32_MAX;
        return c1a < c1b;
    });

    // 2. Build inverse permutation: perm[i] == old_index, inv_perm[old] == new_index.
    std::vector<int32_t> inv_perm(static_cast<size_t>(n_edges));
    for (int32_t i = 0; i < n_edges; ++i)
        inv_perm[static_cast<size_t>(perm[i])] = i;

    // 3. Apply permutation to all edge arrays.
    auto perm_i32 = [&](std::vector<int32_t>& arr) {
        std::vector<int32_t> tmp(arr);
        for (int32_t i = 0; i < n_edges; ++i)
            arr[static_cast<size_t>(i)] = tmp[static_cast<size_t>(perm[i])];
    };
    auto perm_dbl = [&](std::vector<double>& arr) {
        std::vector<double> tmp(arr);
        for (int32_t i = 0; i < n_edges; ++i)
            arr[static_cast<size_t>(i)] = tmp[static_cast<size_t>(perm[i])];
    };
    auto perm_bc = [&](std::vector<BCType>& arr) {
        std::vector<BCType> tmp(arr);
        for (int32_t i = 0; i < n_edges; ++i)
            arr[static_cast<size_t>(i)] = tmp[static_cast<size_t>(perm[i])];
    };

    perm_i32(mesh.edge_c0);
    perm_i32(mesh.edge_c1);
    perm_i32(mesh.edge_n0);
    perm_i32(mesh.edge_n1);
    perm_dbl(mesh.edge_nx);
    perm_dbl(mesh.edge_ny);
    perm_dbl(mesh.edge_len);
    perm_bc(mesh.edge_bc);
    perm_dbl(mesh.edge_bc_val);

    // 4. Remap cell_edge_ids through the inverse permutation.
    for (auto& eidx : mesh.cell_edge_ids)
        eidx = inv_perm[static_cast<size_t>(eidx)];

    // cell_edge_offsets unchanged (each cell has the same number of incident edges).
}

// ─────────────────────────────────────────────────────────────────────────────
// Reverse Cuthill-McKee cell renumbering for GPU cache locality
// ─────────────────────────────────────────────────────────────────────────────
/** Renumber cells via BFS so connected cells have nearby indices.
    Edge c0/c1 entries are remapped through the permutation.
    The permutation is stored in mesh.cell_perm for later use by the solver. */
void swe2d_renumber_cells_for_gpu(SWE2DMesh& mesh) {
    const int32_t n_cells = mesh.n_cells;
    if (n_cells <= 1) return;

    // 1. Build CSR cell adjacency from edges.
    std::vector<int32_t> adj_offsets(static_cast<size_t>(n_cells) + 1, 0);
    for (int32_t e = 0; e < mesh.n_edges; ++e) {
        int32_t c0 = mesh.edge_c0[static_cast<size_t>(e)];
        int32_t c1 = mesh.edge_c1[static_cast<size_t>(e)];
        if (c0 >= 0 && c0 < n_cells) adj_offsets[static_cast<size_t>(c0)]++;
        if (c1 >= 0 && c1 < n_cells) adj_offsets[static_cast<size_t>(c1)]++;
    }
    for (int32_t c = 1; c <= n_cells; ++c)
        adj_offsets[static_cast<size_t>(c)] += adj_offsets[static_cast<size_t>(c - 1)];
    int32_t total_adj = adj_offsets[static_cast<size_t>(n_cells)];
    std::vector<int32_t> adj_ids(static_cast<size_t>(total_adj));
    std::vector<int32_t> adj_pos = adj_offsets;
    for (int32_t e = 0; e < mesh.n_edges; ++e) {
        int32_t c0 = mesh.edge_c0[static_cast<size_t>(e)];
        int32_t c1 = mesh.edge_c1[static_cast<size_t>(e)];
        size_t p;
        if (c0 >= 0 && c0 < n_cells) {
            p = static_cast<size_t>(adj_pos[static_cast<size_t>(c0)]++);
            adj_ids[p] = c1 >= 0 ? c1 : -1;
        }
        if (c1 >= 0 && c1 < n_cells) {
            p = static_cast<size_t>(adj_pos[static_cast<size_t>(c1)]++);
            adj_ids[p] = c0;
        }
    }

    // 2. Degree-sorted BFS (Cuthill-McKee).  Start from lowest-degree cell.
    std::vector<int32_t> degree(static_cast<size_t>(n_cells));
    for (int32_t c = 0; c < n_cells; ++c)
        degree[static_cast<size_t>(c)] = adj_offsets[static_cast<size_t>(c + 1)]
                                        - adj_offsets[static_cast<size_t>(c)];

    std::vector<bool> visited(static_cast<size_t>(n_cells), false);
    std::vector<int32_t> perm(static_cast<size_t>(n_cells));
    size_t perm_idx = 0;

    // Find starting cell (lowest degree among cells with neighbours).
    int32_t start = 0;
    int32_t min_deg = INT32_MAX;
    for (int32_t c = 0; c < n_cells; ++c) {
        int32_t d = degree[static_cast<size_t>(c)];
        if (d > 0 && d < min_deg) { min_deg = d; start = c; }
    }
    if (min_deg == INT32_MAX) return;  // no edges at all

    std::vector<int32_t> queue;
    queue.push_back(start);
    visited[static_cast<size_t>(start)] = true;

    while (!queue.empty()) {
        int32_t cur = queue.front();
        queue.erase(queue.begin());
        perm[perm_idx++] = cur;

        // Collect unvisited neighbours, sorted by degree (Cuthill-McKee ordering).
        size_t s = static_cast<size_t>(adj_offsets[static_cast<size_t>(cur)]);
        size_t e = static_cast<size_t>(adj_offsets[static_cast<size_t>(cur + 1)]);
        std::vector<int32_t> nb;
        for (size_t i = s; i < e; ++i) {
            int32_t n = adj_ids[i];
            if (n >= 0 && !visited[static_cast<size_t>(n)]) {
                visited[static_cast<size_t>(n)] = true;
                nb.push_back(n);
            }
        }
        std::sort(nb.begin(), nb.end(), [&](int32_t a, int32_t b) {
            return degree[static_cast<size_t>(a)] < degree[static_cast<size_t>(b)];
        });
        queue.insert(queue.end(), nb.begin(), nb.end());
    }

    // Handle disconnected components (islands).
    for (int32_t c = 0; c < n_cells; ++c) {
        if (!visited[static_cast<size_t>(c)])
            perm[perm_idx++] = c;
    }

    // 3. Reverse the order (Reverse Cuthill-McKee).
    std::reverse(perm.begin(), perm.end());

    // 4. Build inverse permutation: inv_perm[old_id] = new_id.
    std::vector<int32_t> inv_perm(static_cast<size_t>(n_cells));
    for (int32_t i = 0; i < n_cells; ++i)
        inv_perm[static_cast<size_t>(perm[static_cast<size_t>(i)])] = i;

    // 5. Apply inverse permutation to all cell-indexed arrays.
    auto perm_cell_i32 = [&](std::vector<int32_t>& arr) {
        if (arr.size() != static_cast<size_t>(n_cells)) return;
        std::vector<int32_t> tmp(arr);
        for (int32_t c = 0; c < n_cells; ++c)
            arr[static_cast<size_t>(c)] = tmp[static_cast<size_t>(perm[static_cast<size_t>(c)])];
    };
    auto perm_cell_dbl = [&](std::vector<double>& arr) {
        if (arr.size() != static_cast<size_t>(n_cells)) return;
        std::vector<double> tmp(arr);
        for (int32_t c = 0; c < n_cells; ++c)
            arr[static_cast<size_t>(c)] = tmp[static_cast<size_t>(perm[static_cast<size_t>(c)])];
    };

    perm_cell_dbl(mesh.cell_cx);
    perm_cell_dbl(mesh.cell_cy);
    perm_cell_dbl(mesh.cell_area);
    perm_cell_dbl(mesh.cell_zb);
    perm_cell_dbl(mesh.cell_inv_area);
    // cell_face_offsets, cell_edge_offsets are CSR arrays — must stay increasing.
    // cell_face_nodes, cell_edge_ids are indexed by the offsets — not permuted.

    // 6. Remap edge c0/c1 through the inverse permutation.
    for (int32_t e = 0; e < mesh.n_edges; ++e) {
        size_t ei = static_cast<size_t>(e);
        if (mesh.edge_c0[ei] >= 0)
            mesh.edge_c0[ei] = inv_perm[static_cast<size_t>(mesh.edge_c0[ei])];
        if (mesh.edge_c1[ei] >= 0)
            mesh.edge_c1[ei] = inv_perm[static_cast<size_t>(mesh.edge_c1[ei])];
    }

    // 7. Store permutation (solver needs it to reorder h0/hu/hv).
    mesh.cell_perm = std::move(perm);
}

// ─────────────────────────────────────────────────────────────────────────────
// 2-ring cell stencil builder (for least-squares gradient / FV_WENO5)
// ─────────────────────────────────────────────────────────────────────────────
/** Build the 2-ring cell stencil for least-squares gradient (FV_WENO5). */
void swe2d_build_cell_ring2(SWE2DMesh& mesh) {
    const int32_t n_cells = mesh.n_cells;
    mesh.cell_ring2_offsets.assign(static_cast<size_t>(n_cells) + 1, 0);
    mesh.cell_ring2_ids.clear();
    mesh.cell_ring2_dcx.clear();
    mesh.cell_ring2_dcy.clear();
    mesh.cell_ring2_inv_dist2.clear();
    if (n_cells <= 0) return;

    // Helper: the face-neighbour of cell `c` across edge `eidx` (or -1 if boundary).
    auto edge_peer = [&](int32_t eidx, int32_t c) -> int32_t {
        const int32_t a = mesh.edge_c0[static_cast<size_t>(eidx)];
        const int32_t b = mesh.edge_c1[static_cast<size_t>(eidx)];
        if (a == c) return b;
        if (b == c) return a;
        return -1;
    };

    std::vector<int32_t> ring2;  // scratch, reused per cell
    ring2.reserve(32);

    for (int32_t c0 = 0; c0 < n_cells; ++c0) {
        ring2.clear();
        const int32_t es = mesh.cell_edge_offsets[static_cast<size_t>(c0)];
        const int32_t ee = mesh.cell_edge_offsets[static_cast<size_t>(c0) + 1];
        for (int32_t k = es; k < ee; ++k) {
            const int32_t e1 = mesh.cell_edge_ids[static_cast<size_t>(k)];
            const int32_t peer = edge_peer(e1, c0);
            if (peer < 0) continue;
            ring2.push_back(peer);  // 1-hop

            // 2-hop: face-neighbours of the 1-hop peer.
            const int32_t ps = mesh.cell_edge_offsets[static_cast<size_t>(peer)];
            const int32_t pe = mesh.cell_edge_offsets[static_cast<size_t>(peer) + 1];
            for (int32_t k2 = ps; k2 < pe; ++k2) {
                const int32_t e2 = mesh.cell_edge_ids[static_cast<size_t>(k2)];
                const int32_t peer2 = edge_peer(e2, peer);
                if (peer2 >= 0 && peer2 != c0) {
                    ring2.push_back(peer2);
                }
            }
        }

        // Deduplicate (and sort for deterministic ordering).
        std::sort(ring2.begin(), ring2.end());
        ring2.erase(std::unique(ring2.begin(), ring2.end()), ring2.end());

        const double cx0 = mesh.cell_cx[static_cast<size_t>(c0)];
        const double cy0 = mesh.cell_cy[static_cast<size_t>(c0)];
        for (int32_t j : ring2) {
            const double dx = mesh.cell_cx[static_cast<size_t>(j)] - cx0;
            const double dy = mesh.cell_cy[static_cast<size_t>(j)] - cy0;
            const double d2 = dx * dx + dy * dy;
            const double w  = (d2 > 0.0) ? (1.0 / d2) : 0.0;
            mesh.cell_ring2_ids.push_back(j);
            mesh.cell_ring2_dcx.push_back(dx);
            mesh.cell_ring2_dcy.push_back(dy);
            mesh.cell_ring2_inv_dist2.push_back(w);
        }
        mesh.cell_ring2_offsets[static_cast<size_t>(c0) + 1] =
            static_cast<int32_t>(mesh.cell_ring2_ids.size());
    }
}

/** Validate mesh consistency: array sizes, cell areas, edge lengths, node ranges.
    @param mesh Mesh to validate @returns Empty string on success, error message on failure */
std::string swe2d_validate_mesh(const SWE2DMesh& mesh) {
    std::ostringstream err;

    if (mesh.n_nodes <= 0) {
        err << "mesh has no nodes; ";
    }
    if (mesh.n_cells <= 0) {
        err << "mesh has no cells; ";
    }
    if (mesh.n_edges <= 0) {
        err << "mesh has no edges; ";
    }

    auto check_sz = [&](const char* name, size_t actual, size_t expected) {
        if (actual != expected) {
            err << name << " size mismatch (got " << actual << " expected " << expected << "); ";
        }
    };
    check_sz("node_x", mesh.node_x.size(), static_cast<size_t>(mesh.n_nodes));
    check_sz("node_y", mesh.node_y.size(), static_cast<size_t>(mesh.n_nodes));
    check_sz("node_z", mesh.node_z.size(), static_cast<size_t>(mesh.n_nodes));
    check_sz("cell_face_offsets", mesh.cell_face_offsets.size(), static_cast<size_t>(mesh.n_cells + 1));
    check_sz("cell_edge_offsets", mesh.cell_edge_offsets.size(), static_cast<size_t>(mesh.n_cells + 1));
    check_sz("cell_area", mesh.cell_area.size(), static_cast<size_t>(mesh.n_cells));
    check_sz("cell_edge_ids", mesh.cell_edge_ids.size(), mesh.cell_face_nodes.size());
    check_sz("edge_c0", mesh.edge_c0.size(), static_cast<size_t>(mesh.n_edges));
    check_sz("edge_bc", mesh.edge_bc.size(), static_cast<size_t>(mesh.n_edges));

    if (!mesh.cell_face_offsets.empty()) {
        if (mesh.cell_face_offsets.front() != 0) {
            err << "cell_face_offsets[0] must be 0; ";
        }
        int32_t face_nodes_n = static_cast<int32_t>(mesh.cell_face_nodes.size());
        if (mesh.cell_face_offsets.back() != face_nodes_n) {
            err << "cell_face_offsets tail must match cell_face_nodes size; ";
        }
        for (int32_t c = 0; c < mesh.n_cells; ++c) {
            int32_t s = mesh.cell_face_offsets[c];
            int32_t e = mesh.cell_face_offsets[c + 1];
            if (s < 0 || e < s || e > face_nodes_n) {
                err << "invalid face offsets for cell " << c << "; ";
                break;
            }
            if (e - s < 3) {
                err << "cell " << c << " has fewer than 3 vertices; ";
                break;
            }
        }
    }

    if (!mesh.cell_edge_offsets.empty()) {
        if (mesh.cell_edge_offsets.front() != 0) {
            err << "cell_edge_offsets[0] must be 0; ";
        }
        int32_t edge_ids_n = static_cast<int32_t>(mesh.cell_edge_ids.size());
        if (mesh.cell_edge_offsets.back() != edge_ids_n) {
            err << "cell_edge_offsets tail must match cell_edge_ids size; ";
        }
    }

    for (int32_t c = 0; c < mesh.n_cells; ++c) {
        if (mesh.cell_area[c] <= 0.0) {
            err << "cell " << c << " has non-positive area; ";
            break;
        }
    }

    for (int32_t e = 0; e < mesh.n_edges; ++e) {
        if (mesh.edge_len[e] <= 0.0) {
            err << "edge " << e << " has non-positive length; ";
            break;
        }
    }

    for (int32_t c = 0; c < mesh.n_cells; ++c) {
        int32_t s = mesh.cell_face_offsets[c];
        int32_t e = mesh.cell_face_offsets[c + 1];
        for (int32_t k = s; k < e; ++k) {
            int32_t nidx = mesh.cell_face_nodes[k];
            if (nidx < 0 || nidx >= mesh.n_nodes) {
                err << "cell " << c << " has node index out of range (" << nidx << "); ";
            }
        }
    }

    return err.str();
}