// swe2d_mesh.cpp
// Unstructured polygon mesh builder and validator.

#include "swe2d_mesh.hpp"

#include <algorithm>
#include <cassert>
#include <climits>
#include <cstddef>
#include <deque>
#include <cstdint>
#include <cstring>
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

    // Build edge connectivity from polygon rings using sorted-vector dedup.
    // ponytail: sorted-vector is 3-5x faster than unordered_map for large meshes.
    struct EdgeCandidate {
        int64_t key;
        int32_t c, na, nb, pos;
    };
    std::vector<EdgeCandidate> candidates;
    candidates.reserve(mesh.cell_face_nodes.size());

    for (int32_t c = 0; c < n_cells; ++c) {
        int32_t s = mesh.cell_face_offsets[c];
        int32_t e = mesh.cell_face_offsets[c + 1];
        int32_t nv = e - s;
        for (int32_t i = 0; i < nv; ++i) {
            int32_t na = mesh.cell_face_nodes[s + i];
            int32_t nb = mesh.cell_face_nodes[s + ((i + 1) % nv)];
            candidates.push_back({edge_key(na, nb), c, na, nb, static_cast<int32_t>(candidates.size())});
        }
    }

    // Sort by key.  Same-key edges are adjacent after sort.
    std::sort(candidates.begin(), candidates.end(),
        [](const EdgeCandidate& a, const EdgeCandidate& b) { return a.key < b.key; });

    mesh.edge_c0.reserve(candidates.size());
    mesh.edge_c1.reserve(candidates.size());
    mesh.edge_n0.reserve(candidates.size());
    mesh.edge_n1.reserve(candidates.size());
    mesh.edge_nx.reserve(candidates.size());
    mesh.edge_ny.reserve(candidates.size());
    mesh.edge_len.reserve(candidates.size());
    mesh.edge_bc.reserve(candidates.size());
    mesh.edge_bc_val.reserve(candidates.size());
    mesh.cell_edge_ids.resize(candidates.size());

    int32_t n_edges = 0;
    for (size_t i = 0; i < candidates.size(); ++i) {
        auto& cur = candidates[i];
        bool first_occurrence = (i == 0 || cur.key != candidates[i - 1].key);

        if (first_occurrence) {
            double dx = mesh.node_x[cur.nb] - mesh.node_x[cur.na];
            double dy = mesh.node_y[cur.nb] - mesh.node_y[cur.na];
            double len = std::sqrt(dx * dx + dy * dy);
            if (len <= 0.0) {
                std::ostringstream oss;
                oss << "swe2d_build_mesh_poly: zero-length edge between nodes " << cur.na << " and " << cur.nb;
                throw std::runtime_error(oss.str());
            }
            double nx = dy / len;
            double ny = -dx / len;

            mesh.edge_c0.push_back(cur.c);
            mesh.edge_c1.push_back(-1);
            mesh.edge_n0.push_back(cur.na);
            mesh.edge_n1.push_back(cur.nb);
            mesh.edge_nx.push_back(nx);
            mesh.edge_ny.push_back(ny);
            mesh.edge_len.push_back(len);
            mesh.edge_bc.push_back(BCType::INTERIOR);
            mesh.edge_bc_val.push_back(0.0);
            mesh.cell_edge_ids[static_cast<size_t>(cur.pos)] = n_edges;
            ++n_edges;
        } else {
            int32_t eidx = n_edges - 1;  // the edge just created
            if (mesh.edge_c1[static_cast<size_t>(eidx)] != -1) {
                std::ostringstream oss;
                oss << "swe2d_build_mesh_poly: non-manifold edge between nodes " << cur.na << " and " << cur.nb;
                throw std::runtime_error(oss.str());
            }
            mesh.edge_c1[static_cast<size_t>(eidx)] = cur.c;
            mesh.edge_bc[static_cast<size_t>(eidx)] = BCType::INTERIOR;
            mesh.cell_edge_ids[static_cast<size_t>(cur.pos)] = eidx;
        }
    }

    mesh.n_edges = n_edges;
    // cell_edge_offsets[c] = cell_face_offsets[c] because each face node pair
    // creates exactly one candidate, and pos preserves the original face-walk order.
    for (int32_t c = 0; c <= n_cells; ++c)
        mesh.cell_edge_offsets[static_cast<size_t>(c)] = mesh.cell_face_offsets[static_cast<size_t>(c)];

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

    // Renumber cells via Reverse Cuthill-McKee for GPU cache locality.
    // Must run before edge reorder so edges sort by the new cell indices.
    swe2d_renumber_cells_for_gpu(mesh);

    // Reorder edges by (c0, c1) for GPU memory coalescing.
    swe2d_reorder_edges_for_gpu(mesh);

    // Build the 2-ring cell stencil (CSR) used by the least-squares gradient
    // (spatial scheme 6 / FV_WENO5).
    swe2d_build_cell_ring2(mesh);

    // Build WENO3 sub-stencil tables (S0, S1, S2) and MP5 5-cell walk table
    // for spatial scheme 6 (FV_WENO3) and scheme 8 (MP5) reconstructions.
    swe2d_build_face_substencil_tables(mesh);
    swe2d_build_face_stencil_5_table(mesh);

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
/** Renumber cells using Reverse Cuthill-McKee so connected cells have nearby
    indices.  Uses minimum-degree root, degree-sorted BFS levels, and final
    reversal to minimise adjacency bandwidth.
    Edge c0/c1 entries are remapped through the permutation.
    The permutation is stored in mesh.cell_perm for later use by the solver. */
void swe2d_renumber_cells_for_gpu(SWE2DMesh& mesh) {
    const int32_t n_cells = mesh.n_cells;
    if (n_cells <= 1) return;

    // Build CSR adjacency from edges and compute cell degrees.
    std::vector<int32_t> degree(static_cast<size_t>(n_cells), 0);
    std::vector<int32_t> adj_offsets(static_cast<size_t>(n_cells) + 1, 0);
    for (int32_t e = 0; e < mesh.n_edges; ++e) {
        int32_t c0 = mesh.edge_c0[static_cast<size_t>(e)];
        int32_t c1 = mesh.edge_c1[static_cast<size_t>(e)];
        if (c0 >= 0 && c0 < n_cells) { adj_offsets[static_cast<size_t>(c0)]++; degree[static_cast<size_t>(c0)]++; }
        if (c1 >= 0 && c1 < n_cells) { adj_offsets[static_cast<size_t>(c1)]++; degree[static_cast<size_t>(c1)]++; }
    }
    int32_t running = 0;
    for (int32_t c = 0; c < n_cells; ++c) {
        int32_t deg = adj_offsets[static_cast<size_t>(c)];
        adj_offsets[static_cast<size_t>(c)] = running;
        running += deg;
    }
    adj_offsets[static_cast<size_t>(n_cells)] = running;
    int32_t total_adj = running;

    std::vector<int32_t> adj_ids(static_cast<size_t>(total_adj));
    std::vector<int32_t> adj_pos(adj_offsets.begin(), adj_offsets.begin() + n_cells);
    for (int32_t e = 0; e < mesh.n_edges; ++e) {
        int32_t c0 = mesh.edge_c0[static_cast<size_t>(e)];
        int32_t c1 = mesh.edge_c1[static_cast<size_t>(e)];
        if (c0 >= 0 && c0 < n_cells) {
            size_t p = static_cast<size_t>(adj_pos[static_cast<size_t>(c0)]++);
            adj_ids[p] = c1 >= 0 ? c1 : -1;
        }
        if (c1 >= 0 && c1 < n_cells) {
            size_t p = static_cast<size_t>(adj_pos[static_cast<size_t>(c1)]++);
            adj_ids[p] = c0;
        }
    }

    // Find minimum-degree root (pseudo-peripheral proxy).
    int32_t root = 0;
    {
        int32_t min_deg = degree[0];
        for (int32_t c = 1; c < n_cells; ++c) {
            int32_t d = degree[static_cast<size_t>(c)];
            if (d < min_deg) { min_deg = d; root = c; }
        }
    }

    // RCMK: BFS from root, sort each level by degree ascending.
    std::vector<int32_t> perm(static_cast<size_t>(n_cells));
    std::vector<char> visited(static_cast<size_t>(n_cells), 0);
    size_t perm_idx = 0;

    std::deque<int32_t> queue;
    queue.push_back(root);
    visited[static_cast<size_t>(root)] = 1;

    std::vector<int32_t> level_scratch;
    while (!queue.empty()) {
        size_t level_size = queue.size();
        level_scratch.clear();
        level_scratch.reserve(level_size);

        for (size_t i = 0; i < level_size; ++i) {
            int32_t cur = queue.front();
            queue.pop_front();
            level_scratch.push_back(cur);

            size_t s = static_cast<size_t>(adj_offsets[static_cast<size_t>(cur)]);
            size_t e_bfs = static_cast<size_t>(adj_offsets[static_cast<size_t>(cur + 1)]);
            for (size_t j = s; j < e_bfs; ++j) {
                int32_t n = adj_ids[j];
                if (n >= 0 && n < n_cells && !visited[static_cast<size_t>(n)]) {
                    visited[static_cast<size_t>(n)] = 1;
                    queue.push_back(n);
                }
            }
        }

        // Sort by degree ascending (Cuthill-McKee ordering).
        std::sort(level_scratch.begin(), level_scratch.end(),
            [&](int32_t a, int32_t b) {
                return degree[static_cast<size_t>(a)] < degree[static_cast<size_t>(b)];
            });

        for (int32_t node : level_scratch)
            perm[perm_idx++] = node;
    }

    // Handle disconnected components (islands) — place after main component.
    size_t main_component_end = perm_idx;
    for (int32_t c = 0; c < n_cells; ++c) {
        if (!visited[static_cast<size_t>(c)])
            perm[perm_idx++] = c;
    }

    // Reverse main component ordering (the "Reverse" in RCMK).
    std::reverse(perm.begin(), perm.begin() + static_cast<std::ptrdiff_t>(main_component_end));

    // Build inverse permutation.
    std::vector<int32_t> inv_perm(static_cast<size_t>(n_cells));
    for (int32_t i = 0; i < n_cells; ++i)
        inv_perm[static_cast<size_t>(perm[static_cast<size_t>(i)])] = i;

    // Apply permutation to cell-indexed arrays.
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

    // Permute cell-edge CSR arrays to match the new cell order.
    {
        std::vector<int32_t> new_edge_ids;
        new_edge_ids.reserve(mesh.cell_edge_ids.size());
        std::vector<int32_t> new_offsets(static_cast<size_t>(n_cells) + 1);
        new_offsets[0] = 0;
        for (int32_t c = 0; c < n_cells; ++c) {
            int32_t old_c = perm[static_cast<size_t>(c)];
            int32_t s = mesh.cell_edge_offsets[static_cast<size_t>(old_c)];
            int32_t e = mesh.cell_edge_offsets[static_cast<size_t>(old_c) + 1];
            for (int32_t k = s; k < e; ++k)
                new_edge_ids.push_back(mesh.cell_edge_ids[static_cast<size_t>(k)]);
            new_offsets[static_cast<size_t>(c) + 1] = static_cast<int32_t>(new_edge_ids.size());
        }
        mesh.cell_edge_ids = std::move(new_edge_ids);
        mesh.cell_edge_offsets = std::move(new_offsets);
    }

    // Permute cell_face_nodes to match the new cell order.
    {
        std::vector<int32_t> new_face_nodes;
        new_face_nodes.reserve(mesh.cell_face_nodes.size());
        std::vector<int32_t> new_face_offsets(static_cast<size_t>(n_cells) + 1);
        new_face_offsets[0] = 0;
        for (int32_t c = 0; c < n_cells; ++c) {
            int32_t old_c = perm[static_cast<size_t>(c)];
            int32_t s = mesh.cell_face_offsets[static_cast<size_t>(old_c)];
            int32_t e = mesh.cell_face_offsets[static_cast<size_t>(old_c) + 1];
            for (int32_t k = s; k < e; ++k)
                new_face_nodes.push_back(mesh.cell_face_nodes[static_cast<size_t>(k)]);
            new_face_offsets[static_cast<size_t>(c) + 1] = static_cast<int32_t>(new_face_nodes.size());
        }
        mesh.cell_face_nodes = std::move(new_face_nodes);
        mesh.cell_face_offsets = std::move(new_face_offsets);
    }

    // Remap edge c0/c1.
    for (int32_t e = 0; e < mesh.n_edges; ++e) {
        int32_t old_c0 = mesh.edge_c0[static_cast<size_t>(e)];
        if (old_c0 >= 0) {
            mesh.edge_c0[static_cast<size_t>(e)] = inv_perm[static_cast<size_t>(old_c0)];
        }
        int32_t old_c1 = mesh.edge_c1[static_cast<size_t>(e)];
        if (old_c1 >= 0) {
            mesh.edge_c1[static_cast<size_t>(e)] = inv_perm[static_cast<size_t>(old_c1)];
        }
    }

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

    // Reusable scratch for per-cell neighbor collection.
    std::vector<int32_t> ring2;
    ring2.reserve(32);
    // Epoch-based visited flag avoids O(n) clear per cell.
    std::vector<int32_t> visited_seq(static_cast<size_t>(n_cells), 0);
    uint32_t epoch = 0;

    for (int32_t c0 = 0; c0 < n_cells; ++c0) {
        ring2.clear();
        ++epoch;
        const int32_t es = mesh.cell_edge_offsets[static_cast<size_t>(c0)];
        const int32_t ee = mesh.cell_edge_offsets[static_cast<size_t>(c0) + 1];
        for (int32_t k = es; k < ee; ++k) {
            const int32_t e1 = mesh.cell_edge_ids[static_cast<size_t>(k)];
            const int32_t peer = edge_peer(e1, c0);
            if (peer < 0) continue;
            if (visited_seq[static_cast<size_t>(peer)] != static_cast<int32_t>(epoch)) {
                visited_seq[static_cast<size_t>(peer)] = static_cast<int32_t>(epoch);
                ring2.push_back(peer);
            }

            // 2-hop: face-neighbours of the 1-hop peer.
            const int32_t ps = mesh.cell_edge_offsets[static_cast<size_t>(peer)];
            const int32_t pe = mesh.cell_edge_offsets[static_cast<size_t>(peer) + 1];
            for (int32_t k2 = ps; k2 < pe; ++k2) {
                const int32_t e2 = mesh.cell_edge_ids[static_cast<size_t>(k2)];
                const int32_t peer2 = edge_peer(e2, peer);
                if (peer2 >= 0 && peer2 != c0) {
                    if (visited_seq[static_cast<size_t>(peer2)] != static_cast<int32_t>(epoch)) {
                        visited_seq[static_cast<size_t>(peer2)] = static_cast<int32_t>(epoch);
                        ring2.push_back(peer2);
                    }
                }
            }
        }

        // ring2 is now deduplicated and emitted in deterministic edge-traversal order.

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

// ─────────────────────────────────────────────────────────────────────────────
// WENO3 face sub-stencil tables (S0, S1, S2) and MP5 5-cell walk table
// ─────────────────────────────────────────────────────────────────────────────

/** Build the WENO3 face sub-stencil tables S0, S1, S2 for scheme 6.
    For each face f:
      S1[2*f .. 2*f+1] = {owner, neighbor}
      S0 = all face-neighbors of owner, excluding the neighbor (upwind lobe)
      S2 = all face-neighbors of neighbor, excluding the owner (downwind lobe)
    Boundary faces (c1 == -1) produce an empty S2.  Uses CSR for S0 and S2.
    Requires cell_edge_offsets/cell_edge_ids and edge_c0/edge_c1 to be built. */
void swe2d_build_face_substencil_tables(SWE2DMesh& mesh) {
    const int32_t n_edges = mesh.n_edges;
    if (n_edges <= 0) return;

    // Helper: the face-neighbour of cell c across edge eidx (or -1 if boundary).
    auto edge_peer = [&](int32_t eidx, int32_t c) -> int32_t {
        const int32_t a = mesh.edge_c0[static_cast<size_t>(eidx)];
        const int32_t b = mesh.edge_c1[static_cast<size_t>(eidx)];
        if (a == c) return b;
        if (b == c) return a;
        return -1;
    };

    // S1: flat {owner, neighbor} per face.
    mesh.face_stencil_S1.resize(static_cast<size_t>(2) * n_edges);

    // First pass: count S0 and S2 entries per face.
    std::vector<int32_t> s0_cnt(static_cast<size_t>(n_edges), 0);
    std::vector<int32_t> s2_cnt(static_cast<size_t>(n_edges), 0);

    for (int32_t f = 0; f < n_edges; ++f) {
        const int32_t c0 = mesh.edge_c0[static_cast<size_t>(f)];
        const int32_t c1 = mesh.edge_c1[static_cast<size_t>(f)];

        mesh.face_stencil_S1[static_cast<size_t>(2) * f + 0] = c0;
        mesh.face_stencil_S1[static_cast<size_t>(2) * f + 1] = c1;

        // S0: neighbours of c0 excluding c1.
        {
            const int32_t es = mesh.cell_edge_offsets[static_cast<size_t>(c0)];
            const int32_t ee = mesh.cell_edge_offsets[static_cast<size_t>(c0) + 1];
            for (int32_t k = es; k < ee; ++k) {
                const int32_t eidx = mesh.cell_edge_ids[static_cast<size_t>(k)];
                const int32_t peer = edge_peer(eidx, c0);
                if (peer >= 0 && peer != c1) {
                    ++s0_cnt[static_cast<size_t>(f)];
                }
            }
        }

        // S2: neighbours of c1 excluding c0.
        if (c1 >= 0) {
            const int32_t cs = mesh.cell_edge_offsets[static_cast<size_t>(c1)];
            const int32_t ce = mesh.cell_edge_offsets[static_cast<size_t>(c1) + 1];
            for (int32_t k = cs; k < ce; ++k) {
                const int32_t eidx = mesh.cell_edge_ids[static_cast<size_t>(k)];
                const int32_t peer = edge_peer(eidx, c1);
                if (peer >= 0 && peer != c0) {
                    ++s2_cnt[static_cast<size_t>(f)];
                }
            }
        }
    }

    // Build CSR offsets from counts.
    mesh.face_stencil_S0_offsets.assign(static_cast<size_t>(n_edges) + 1, 0);
    mesh.face_stencil_S2_offsets.assign(static_cast<size_t>(n_edges) + 1, 0);
    for (int32_t f = 0; f < n_edges; ++f) {
        const size_t fu = static_cast<size_t>(f);
        mesh.face_stencil_S0_offsets[fu + 1] =
            mesh.face_stencil_S0_offsets[fu] + s0_cnt[fu];
        mesh.face_stencil_S2_offsets[fu + 1] =
            mesh.face_stencil_S2_offsets[fu] + s2_cnt[fu];
    }

    const int32_t total_s0 = mesh.face_stencil_S0_offsets[static_cast<size_t>(n_edges)];
    const int32_t total_s2 = mesh.face_stencil_S2_offsets[static_cast<size_t>(n_edges)];
    mesh.face_stencil_S0_cells.resize(static_cast<size_t>(total_s0));
    mesh.face_stencil_S2_cells.resize(static_cast<size_t>(total_s2));

    // Second pass: fill S0 and S2 cell arrays.
    // Write cursors initialised from offsets.
    std::vector<int32_t> s0_pos(s0_cnt);  // reuse counts as cursors
    std::vector<int32_t> s2_pos(s2_cnt);
    for (int32_t f = 0; f < n_edges; ++f) {
        s0_pos[static_cast<size_t>(f)] =
            mesh.face_stencil_S0_offsets[static_cast<size_t>(f)];
        s2_pos[static_cast<size_t>(f)] =
            mesh.face_stencil_S2_offsets[static_cast<size_t>(f)];
    }

    for (int32_t f = 0; f < n_edges; ++f) {
        const int32_t c0 = mesh.edge_c0[static_cast<size_t>(f)];
        const int32_t c1 = mesh.edge_c1[static_cast<size_t>(f)];

        // Fill S0.
        {
            size_t pos = static_cast<size_t>(s0_pos[static_cast<size_t>(f)]);
            const int32_t es = mesh.cell_edge_offsets[static_cast<size_t>(c0)];
            const int32_t ee = mesh.cell_edge_offsets[static_cast<size_t>(c0) + 1];
            for (int32_t k = es; k < ee; ++k) {
                const int32_t eidx = mesh.cell_edge_ids[static_cast<size_t>(k)];
                const int32_t peer = edge_peer(eidx, c0);
                if (peer >= 0 && peer != c1) {
                    mesh.face_stencil_S0_cells[pos++] = peer;
                }
            }
        }

        // Fill S2.
        if (c1 >= 0) {
            size_t pos = static_cast<size_t>(s2_pos[static_cast<size_t>(f)]);
            const int32_t cs = mesh.cell_edge_offsets[static_cast<size_t>(c1)];
            const int32_t ce = mesh.cell_edge_offsets[static_cast<size_t>(c1) + 1];
            for (int32_t k = cs; k < ce; ++k) {
                const int32_t eidx = mesh.cell_edge_ids[static_cast<size_t>(k)];
                const int32_t peer = edge_peer(eidx, c1);
                if (peer >= 0 && peer != c0) {
                    mesh.face_stencil_S2_cells[pos++] = peer;
                }
            }
        }
    }
}

/** Build the MP5 5-cell face-normal walk table for scheme 8.
    For each interior face f with owner c0 and neighbour c1:
      {u2, u1, u, v, v1} where
        u  = c0 (upwind cell)
        v  = c1 (downwind cell)
        u1 = first neighbour of c0 != c1
        u2 = first neighbour of u1 != c0
        v1 = first neighbour of c1 != c0
    For boundary faces (c1 == -1) all five positions are set to c0.
    face_mp5_case[f] = 1 for all faces (case re-evaluated at runtime in the kernel). */
void swe2d_build_face_stencil_5_table(SWE2DMesh& mesh) {
    const int32_t n_edges = mesh.n_edges;
    if (n_edges <= 0) return;

    mesh.face_stencil_5.resize(static_cast<size_t>(5) * n_edges);
    mesh.face_mp5_case.assign(static_cast<size_t>(n_edges), 1);

    // Helper: the face-neighbour of cell c across edge eidx (or -1 if boundary).
    auto edge_peer = [&](int32_t eidx, int32_t c) -> int32_t {
        const int32_t a = mesh.edge_c0[static_cast<size_t>(eidx)];
        const int32_t b = mesh.edge_c1[static_cast<size_t>(eidx)];
        if (a == c) return b;
        if (b == c) return a;
        return -1;
    };

    // Helper: return the neighbour of cell c that is not exclude_cell and whose
    // centroid has the largest signed projection onto the direction dir.
    // If no neighbour exists, return c itself.
    auto best_neighbor_in_direction = [&](int32_t c, int32_t exclude_cell,
                                        double dir_x, double dir_y) -> int32_t {
        const int32_t es = mesh.cell_edge_offsets[static_cast<size_t>(c)];
        const int32_t ee = mesh.cell_edge_offsets[static_cast<size_t>(c) + 1];
        int32_t best_peer = c;
        double best_proj = -1.0e300;
        const double cx = mesh.cell_cx[static_cast<size_t>(c)];
        const double cy = mesh.cell_cy[static_cast<size_t>(c)];
        for (int32_t k = es; k < ee; ++k) {
            const int32_t eidx = mesh.cell_edge_ids[static_cast<size_t>(k)];
            const int32_t peer = edge_peer(eidx, c);
            if (peer < 0 || peer == exclude_cell) continue;
            const double px = mesh.cell_cx[static_cast<size_t>(peer)] - cx;
            const double py = mesh.cell_cy[static_cast<size_t>(peer)] - cy;
            const double proj = px * dir_x + py * dir_y;
            if (proj > best_proj) {
                best_proj = proj;
                best_peer = peer;
            }
        }
        return best_peer;
    };

    for (int32_t f = 0; f < n_edges; ++f) {
        const int32_t c0 = mesh.edge_c0[static_cast<size_t>(f)];
        const int32_t c1 = mesh.edge_c1[static_cast<size_t>(f)];

        int32_t u2, u1, u, v, v1;

        if (c1 < 0) {
            // Boundary face: all five positions = c0.
            u2 = u1 = u = v = v1 = c0;
        } else {
            u  = c0;
            v  = c1;
            // Face normal points from c0 to c1.  Upwind from c0 is the direction
            // opposite to the normal, i.e. -normal.  Downwind from c1 is +normal.
            const double nx = mesh.edge_nx[static_cast<size_t>(f)];
            const double ny = mesh.edge_ny[static_cast<size_t>(f)];
            u1 = best_neighbor_in_direction(c0, c1, -nx, -ny);
            u2 = best_neighbor_in_direction(u1, c0, -nx, -ny);
            v1 = best_neighbor_in_direction(c1, c0,  nx,  ny);
        }

        const size_t base = static_cast<size_t>(5) * f;
        mesh.face_stencil_5[base + 0] = u2;
        mesh.face_stencil_5[base + 1] = u1;
        mesh.face_stencil_5[base + 2] = u;
        mesh.face_stencil_5[base + 3] = v;
        mesh.face_stencil_5[base + 4] = v1;
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

// ─────────────────────────────────────────────────────────────────────────────
// Mesh BLOB serialization (raw binary, no zlib)
// ─────────────────────────────────────────────────────────────────────────────

/// Helper: append a vector's element count (uint64) and raw bytes to a buffer.
template<typename T>
static void serialize_vector(std::vector<uint8_t>& buf, const std::vector<T>& vec) {
    uint64_t count = static_cast<uint64_t>(vec.size());
    const uint8_t* count_ptr = reinterpret_cast<const uint8_t*>(&count);
    buf.insert(buf.end(), count_ptr, count_ptr + sizeof(count));
    if (!vec.empty()) {
        const uint8_t* data_ptr = reinterpret_cast<const uint8_t*>(vec.data());
        buf.insert(buf.end(), data_ptr, data_ptr + vec.size() * sizeof(T));
    }
}

/// Helper: read a uint64 count and then read `count` elements of type T from the buffer.
/// Advances `pos` past the consumed bytes. Throws on overflow.
template<typename T>
static std::vector<T> deserialize_vector(const uint8_t* data, size_t size, size_t& pos) {
    if (pos + sizeof(uint64_t) > size) {
        throw std::runtime_error("swe2d_deserialize_mesh: truncated count prefix");
    }
    uint64_t count;
    std::memcpy(&count, data + pos, sizeof(uint64_t));
    pos += sizeof(uint64_t);
    const size_t elem_bytes = count * sizeof(T);
    if (elem_bytes > size - pos) {
        throw std::runtime_error("swe2d_deserialize_mesh: truncated vector data");
    }
    std::vector<T> vec(static_cast<size_t>(count));
    if (count > 0) {
        std::memcpy(vec.data(), data + pos, elem_bytes);
        pos += elem_bytes;
    }
    return vec;
}

std::vector<uint8_t> swe2d_serialize_mesh(const SWE2DMesh& mesh) {
    std::vector<uint8_t> buf;
    buf.reserve(256);

    // Scalars: n_nodes, n_cells, n_edges (each int32)
    auto append_i32 = [&](int32_t v) {
        auto p = reinterpret_cast<const uint8_t*>(&v);
        buf.insert(buf.end(), p, p + sizeof(int32_t));
    };
    append_i32(mesh.n_nodes);
    append_i32(mesh.n_cells);
    append_i32(mesh.n_edges);

    // All vectors in a fixed order (matching the deserialization order).
    serialize_vector(buf, mesh.node_x);
    serialize_vector(buf, mesh.node_y);
    serialize_vector(buf, mesh.node_z);

    serialize_vector(buf, mesh.cell_face_offsets);
    serialize_vector(buf, mesh.cell_face_nodes);
    serialize_vector(buf, mesh.cell_edge_offsets);
    serialize_vector(buf, mesh.cell_edge_ids);

    serialize_vector(buf, mesh.cell_cx);
    serialize_vector(buf, mesh.cell_cy);
    serialize_vector(buf, mesh.cell_area);
    serialize_vector(buf, mesh.cell_zb);
    serialize_vector(buf, mesh.cell_inv_area);

    serialize_vector(buf, mesh.edge_c0);
    serialize_vector(buf, mesh.edge_c1);
    serialize_vector(buf, mesh.edge_n0);
    serialize_vector(buf, mesh.edge_n1);
    serialize_vector(buf, mesh.edge_nx);
    serialize_vector(buf, mesh.edge_ny);
    serialize_vector(buf, mesh.edge_len);

    // edge_bc is stored as int32 (BCType enum underlying type)
    {
        uint64_t count = static_cast<uint64_t>(mesh.edge_bc.size());
        auto p = reinterpret_cast<const uint8_t*>(&count);
        buf.insert(buf.end(), p, p + sizeof(uint64_t));
        for (auto bc : mesh.edge_bc) {
            int32_t v = static_cast<int32_t>(bc);
            auto vp = reinterpret_cast<const uint8_t*>(&v);
            buf.insert(buf.end(), vp, vp + sizeof(int32_t));
        }
    }

    serialize_vector(buf, mesh.edge_bc_val);

    serialize_vector(buf, mesh.cell_perm);

    serialize_vector(buf, mesh.cell_ring2_offsets);
    serialize_vector(buf, mesh.cell_ring2_ids);
    serialize_vector(buf, mesh.cell_ring2_dcx);
    serialize_vector(buf, mesh.cell_ring2_dcy);
    serialize_vector(buf, mesh.cell_ring2_inv_dist2);

    // WENO3 face sub-stencil tables (scheme 6)
    serialize_vector(buf, mesh.face_stencil_S0_offsets);
    serialize_vector(buf, mesh.face_stencil_S0_cells);
    serialize_vector(buf, mesh.face_stencil_S1);
    serialize_vector(buf, mesh.face_stencil_S2_offsets);
    serialize_vector(buf, mesh.face_stencil_S2_cells);

    // MP5 5-cell walk table (scheme 8)
    serialize_vector(buf, mesh.face_stencil_5);
    serialize_vector(buf, mesh.face_mp5_case);

    return buf;
}

SWE2DMesh swe2d_deserialize_mesh(const uint8_t* data, size_t size) {
    size_t pos = 0;
    SWE2DMesh mesh;

    // Read scalars.
    auto read_i32 = [&]() -> int32_t {
        if (pos + sizeof(int32_t) > size) {
            throw std::runtime_error("swe2d_deserialize_mesh: truncated scalar");
        }
        int32_t v;
        std::memcpy(&v, data + pos, sizeof(int32_t));
        pos += sizeof(int32_t);
        return v;
    };

    mesh.n_nodes = read_i32();
    mesh.n_cells = read_i32();
    mesh.n_edges = read_i32();

    // Read vectors in the same order they were written.
    mesh.node_x            = deserialize_vector<double>(data, size, pos);
    mesh.node_y            = deserialize_vector<double>(data, size, pos);
    mesh.node_z            = deserialize_vector<double>(data, size, pos);

    mesh.cell_face_offsets = deserialize_vector<int32_t>(data, size, pos);
    mesh.cell_face_nodes   = deserialize_vector<int32_t>(data, size, pos);
    mesh.cell_edge_offsets = deserialize_vector<int32_t>(data, size, pos);
    mesh.cell_edge_ids     = deserialize_vector<int32_t>(data, size, pos);

    mesh.cell_cx           = deserialize_vector<double>(data, size, pos);
    mesh.cell_cy           = deserialize_vector<double>(data, size, pos);
    mesh.cell_area         = deserialize_vector<double>(data, size, pos);
    mesh.cell_zb           = deserialize_vector<double>(data, size, pos);
    mesh.cell_inv_area     = deserialize_vector<double>(data, size, pos);

    mesh.edge_c0           = deserialize_vector<int32_t>(data, size, pos);
    mesh.edge_c1           = deserialize_vector<int32_t>(data, size, pos);
    mesh.edge_n0           = deserialize_vector<int32_t>(data, size, pos);
    mesh.edge_n1           = deserialize_vector<int32_t>(data, size, pos);
    mesh.edge_nx           = deserialize_vector<double>(data, size, pos);
    mesh.edge_ny           = deserialize_vector<double>(data, size, pos);
    mesh.edge_len          = deserialize_vector<double>(data, size, pos);

    // edge_bc: stored as int32 values (BCType underlying type)
    {
        if (pos + sizeof(uint64_t) > size) {
            throw std::runtime_error("swe2d_deserialize_mesh: truncated edge_bc count");
        }
        uint64_t bc_count;
        std::memcpy(&bc_count, data + pos, sizeof(uint64_t));
        pos += sizeof(uint64_t);
        if (bc_count > 0) {
            const size_t bc_bytes = static_cast<size_t>(bc_count) * sizeof(int32_t);
            if (bc_bytes > size - pos) {
                throw std::runtime_error("swe2d_deserialize_mesh: truncated edge_bc data");
            }
            mesh.edge_bc.resize(static_cast<size_t>(bc_count));
            for (uint64_t i = 0; i < bc_count; ++i) {
                int32_t v;
                std::memcpy(&v, data + pos + i * sizeof(int32_t), sizeof(int32_t));
                mesh.edge_bc[static_cast<size_t>(i)] = static_cast<BCType>(v);
            }
            pos += bc_bytes;
        }
    }

    mesh.edge_bc_val       = deserialize_vector<double>(data, size, pos);

    mesh.cell_perm         = deserialize_vector<int32_t>(data, size, pos);

    mesh.cell_ring2_offsets = deserialize_vector<int32_t>(data, size, pos);
    mesh.cell_ring2_ids    = deserialize_vector<int32_t>(data, size, pos);
    mesh.cell_ring2_dcx    = deserialize_vector<double>(data, size, pos);
    mesh.cell_ring2_dcy    = deserialize_vector<double>(data, size, pos);
    mesh.cell_ring2_inv_dist2 = deserialize_vector<double>(data, size, pos);

    // WENO3 face sub-stencil tables (scheme 6)
    mesh.face_stencil_S0_offsets = deserialize_vector<int32_t>(data, size, pos);
    mesh.face_stencil_S0_cells   = deserialize_vector<int32_t>(data, size, pos);
    mesh.face_stencil_S1         = deserialize_vector<int32_t>(data, size, pos);
    mesh.face_stencil_S2_offsets = deserialize_vector<int32_t>(data, size, pos);
    mesh.face_stencil_S2_cells   = deserialize_vector<int32_t>(data, size, pos);

    // MP5 5-cell walk table (scheme 8)
    mesh.face_stencil_5          = deserialize_vector<int32_t>(data, size, pos);
    mesh.face_mp5_case           = deserialize_vector<int32_t>(data, size, pos);

    return mesh;
}