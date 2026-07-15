// pipe1d.cu
// 1D pipe network CUDA kernel implementations.
// Split from swe2d_gpu.cu — mechanical refactoring, no behavior changes.

#include "pipe1d.cuh"
#include "swe2d_gpu.cuh"

#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <unordered_map>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            throw std::runtime_error(std::string("CUDA error: ")               \
                + cudaGetErrorString(_e) + " at " __FILE__ ":"                 \
                + std::to_string(__LINE__));                                    \
        }                                                                       \
    } while (0)

// Pipe1D geometry table: number of sampling intervals for P(A)/A_full and T(A) lookup.
#define PIPE1D_TABLE_N 256

// ─────────────────────────────────────────────────────────────────────────────
// pipe1d_compute_table — host-side hydraulic geometry precomputation
//
// Builds two lookup tables of length PIPE1D_TABLE_N for a given cross-section:
//   P_ratio[i] = P(A⁻¹(A_ratio[i]·A_full)) / P_full
//   T_val[i]    = T(A⁻¹(A_ratio[i]·A_full))               [actual top width]
// where A_ratio[i] = (i + 0.5) / PIPE1D_TABLE_N  (midpoint sampling).
//
// table_out layout: [P_ratio[0..N-1], T_val[0..N-1]]
// ─────────────────────────────────────────────────────────────────────────────
static void pipe1d_compute_table(
    int shape_type, double width, double height,
    double& A_full, double& P_full,
    std::vector<double>& table_out)
{
    table_out.clear();
    table_out.resize(2 * PIPE1D_TABLE_N, 0.0);

    constexpr double EPS = 1e-12;

    for (int i = 0; i < PIPE1D_TABLE_N; ++i) {
        double A_ratio = (i + 0.5) / PIPE1D_TABLE_N;
        double A_target = A_ratio * A_full;
        double P_cur, T_cur;

        // ── Circular (shape_type == 0) ──
        if (shape_type == 0) {
            double D = width;
            double R = D * 0.5;

            if (i == 0) {
                A_full = M_PI * R * R;
                P_full = 2.0 * M_PI * R;
            }

            // Newton on circular segment: F(y) = R²·acos((R-y)/R) − (R-y)·√(2Ry−y²) − A_target
            double y = A_target / (2.0 * R); // initial guess (rectangular proxy)
            for (int iter = 0; iter < 20; ++iter) {
                double arg = (R - y) / R;
                arg = fmax(-1.0, fmin(1.0, arg));
                double phi = acos(arg);
                double T  = 2.0 * sqrt(fmax(0.0, 2.0 * R * y - y * y));
                double A_cur = R * R * phi - (R - y) * T * 0.5;
                double F = A_cur - A_target;
                if (fabs(F) < EPS * A_full) break;
                y -= F / T;
            }
            // Clamp y to valid range
            if (y < EPS * R) y = EPS * R;
            if (y > 2.0 * R - EPS * R) y = 2.0 * R - EPS * R;

            double arg = (R - y) / R;
            arg = fmax(-1.0, fmin(1.0, arg));
            double phi = acos(arg);                  // half central angle
            P_cur = 2.0 * R * phi;                   // P = R·θ with θ = 2φ
            T_cur = 2.0 * sqrt(fmax(0.0, 2.0 * R * y - y * y));
        }
        // ── Rectangular (shape_type == 1) ──
        else if (shape_type == 1) {
            double W = width;
            double H = height;

            if (i == 0) {
                A_full = W * H;
                P_full = 2.0 * (W + H);
            }

            double y = A_target / W;
            if (y < 0.0) y = 0.0;
            if (y > H)   y = H;

            if (y <= 0.0) {
                P_cur = 0.0;
                T_cur = 0.0;
            } else if (y >= H) {
                P_cur = P_full;
                T_cur = W;
            } else {
                P_cur = W + 2.0 * y;
                T_cur = W;
            }
        }
        // ── Elliptical (shape_type == 2) ──
        else {
            double a = width  * 0.5;  // semi-major axis
            double b = height * 0.5;  // semi-minor axis

            if (i == 0) {
                A_full = M_PI * a * b;
                // Ramanujan approximation for ellipse perimeter
                double h = (a - b) * (a - b) / ((a + b) * (a + b));
                P_full = M_PI * (a + b) * (1.0 + 3.0 * h / (10.0 + sqrt(4.0 - 3.0 * h)));
            }

            // Newton on elliptic segment
            double y = A_target / (2.0 * a); // initial guess (rectangular proxy)
            double phi = 0.0;                // half central angle of filled portion
            for (int iter = 0; iter < 20; ++iter) {
                double yr = y / b;
                double A_cur, T_val;

                if (yr <= 1.0) {
                    // Lower half
                    double arg = 1.0 - yr;
                    arg = fmax(-1.0, fmin(1.0, arg));
                    phi = acos(arg);
                    A_cur = a * b * (phi - 0.5 * sin(2.0 * phi));
                    T_val = 2.0 * a * sqrt(fmax(0.0, yr * (2.0 - yr)));
                } else {
                    // Upper half: compute via complement (empty portion at top)
                    double yr2 = 2.0 - yr;
                    double arg = 1.0 - yr2;
                    arg = fmax(-1.0, fmin(1.0, arg));
                    phi = acos(arg);                              // empty-segment half-angle
                    double A_seg = a * b * (phi - 0.5 * sin(2.0 * phi));
                    A_cur = A_full - A_seg;
                    T_val = 2.0 * a * sqrt(fmax(0.0, yr2 * (2.0 - yr2)));
                }

                double F = A_cur - A_target;
                if (fabs(F) < EPS * A_full) break;
                y -= F / T_val;
            }

            // Clamp y to valid range
            if (y < EPS * b) y = EPS * b;
            if (y > 2.0 * b - EPS * b) y = 2.0 * b - EPS * b;

            // Recompute final phi / T_cur from clamped y
            double yr = y / b;
            if (yr <= 1.0) {
                double arg = 1.0 - yr;
                arg = fmax(-1.0, fmin(1.0, arg));
                phi = acos(arg);
                T_cur = 2.0 * a * sqrt(fmax(0.0, yr * (2.0 - yr)));
            } else {
                double yr2 = 2.0 - yr;
                double arg = 1.0 - yr2;
                arg = fmax(-1.0, fmin(1.0, arg));
                phi = acos(arg);  // empty-segment half-angle
                phi = M_PI - phi; // filled-portion half-angle
                T_cur = 2.0 * a * sqrt(fmax(0.0, yr2 * (2.0 - yr2)));
            }
            // Linear interpolation by central angle (approximate, adequate)
            P_cur = P_full * phi / M_PI;
        }

        table_out[i]                     = P_cur / P_full;
        table_out[PIPE1D_TABLE_N + i]    = T_cur;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// swe2d_build_pipe1d_mesh
// ─────────────────────────────────────────────────────────────────────────────
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
    Pipe1DDeviceState*    dev)
{
    auto alloc_d = [](void** ptr, size_t bytes) {
        CUDA_CHECK(cudaMalloc(ptr, bytes));
    };
    auto copy_h2d_i = [](int32_t* dst, const int32_t* src, size_t n) {
        CUDA_CHECK(cudaMemcpy(dst, src, n * sizeof(int32_t), cudaMemcpyHostToDevice));
    };
    auto copy_h2d_d = [](double* dst, const double* src, size_t n) {
        CUDA_CHECK(cudaMemcpy(dst, src, n * sizeof(double), cudaMemcpyHostToDevice));
    };

    // Count sub-cells per link and find max node index
    std::vector<int32_t> sub_cells_per_link(n_links);
    int32_t max_node_idx = -1;
    int32_t total_pipe_cells = 0;
    for (int32_t i = 0; i < n_links; ++i) {
        const double L = link_length[i];
        int32_t n_sub = 1;
        if (max_cell_length > 0 && L > 0.0) {
            n_sub = static_cast<int32_t>(std::ceil(L / static_cast<double>(max_cell_length)));
            if (n_sub < 1) n_sub = 1;
        }
        sub_cells_per_link[i] = n_sub;
        total_pipe_cells += n_sub;
        if (link_from_node[i] > max_node_idx) max_node_idx = link_from_node[i];
        if (link_to_node[i] > max_node_idx) max_node_idx = link_to_node[i];
    }
    const int32_t n_nodes = max_node_idx + 1;
    dev->n_pipe_cells = total_pipe_cells;
    dev->n_nodes = n_nodes;

    // Geometry per sub-cell
    std::vector<double> cell_length(total_pipe_cells);
    std::vector<double> cell_area(total_pipe_cells);
    std::vector<double> cell_perim(total_pipe_cells);
    std::vector<double> cell_invert(total_pipe_cells);
    std::vector<double> cell_n(total_pipe_cells);
    std::vector<double> cell_link_k(total_pipe_cells);     // k at boundary cells, 0 interior
    std::vector<double> cell_link_area(total_pipe_cells);  // full pipe area at boundary cells, 0 interior
    std::vector<int32_t> cell_from_node(total_pipe_cells);
    std::vector<int32_t> cell_to_node(total_pipe_cells);
    std::vector<int32_t> cell_owner_link(total_pipe_cells);  // which link each sub-cell belongs to
    std::vector<int32_t> cell_sub_idx(total_pipe_cells);    // sub-cell index within its link

    // Cross-section shape + table data
    std::vector<int32_t> cell_shape_type(total_pipe_cells);
    std::vector<double> cell_width(total_pipe_cells);
    std::vector<double> cell_height(total_pipe_cells);
    std::vector<double> cell_tables(total_pipe_cells * 2 * PIPE1D_TABLE_N, 0.0);
    int32_t cell_idx = 0;

    // Deduplication: avoid recomputing tables for identical cross-sections
    struct XsectKey {
        int shape_type;
        double w, h;
        bool operator==(const XsectKey& o) const {
            return shape_type == o.shape_type && fabs(w - o.w) < 1e-9 && fabs(h - o.h) < 1e-9;
        }
    };
    struct XsectKeyHash {
        size_t operator()(const XsectKey& k) const {
            return std::hash<int>()(k.shape_type) ^ std::hash<double>()(k.w) ^ std::hash<double>()(k.h);
        }
    };
    std::unordered_map<XsectKey, std::vector<double>, XsectKeyHash> table_cache;

    for (int32_t i = 0; i < n_links; ++i) {
        const double L = static_cast<double>(link_length[i]);
        const double D = static_cast<double>(link_diameter[i]);
        const double n_val = static_cast<double>(link_roughness_n[i]);
        const double k_in = static_cast<double>(link_inlet_loss_k[i]);
        const double k_out = static_cast<double>(link_outlet_loss_k[i]);
        const double inv_in = static_cast<double>(link_invert_in[i]);
        const double inv_out = static_cast<double>(link_invert_out[i]);
        const int32_t n_sub = sub_cells_per_link[i];
        const double sub_len = L / static_cast<double>(n_sub);

        // Shape resolution: default to circular (type 0); width=height=D
        int stype = 0; double sw = D, sh = D;
        if (link_shape_type) {
            stype = link_shape_type[i];
            if (link_width)  sw = link_width[i];
            if (link_height) sh = link_height[i];
        }

        // Compute area and perimeter from the actual shape dimensions.
        // When diameter is 0 (box/rectangular shapes with no circular equivalent),
        // derive A/P from width/height instead.
        double A, P;
        if (D > 0.0) {
            A = M_PI * D * D / 4.0;
            P = M_PI * D;
        } else if (stype == 1 && sw > 0.0 && sh > 0.0) {
            // Rectangular / box
            A = sw * sh;
            P = 2.0 * (sw + sh);
        } else if (stype == 2 && sw > 0.0 && sh > 0.0) {
            // Elliptical: area = π * (w/2) * (h/2), perimeter ≈ Ramanujan
            A = M_PI * (sw / 2.0) * (sh / 2.0);
            const double a = sw / 2.0, b = sh / 2.0;
            const double h = ((a - b) * (a - b)) / ((a + b) * (a + b));
            P = M_PI * (a + b) * (1.0 + (3.0 * h) / (10.0 + std::sqrt(4.0 - 3.0 * h)));
        } else {
            // Fallback: treat as circular with diameter=sw
            A = M_PI * sw * sw / 4.0;
            P = M_PI * sw;
        }

        for (int32_t s = 0; s < n_sub; ++s) {
            const double frac = (static_cast<double>(s) + 0.5) / static_cast<double>(n_sub);
            cell_length[cell_idx] = sub_len;
            cell_area[cell_idx] = A;
            cell_perim[cell_idx] = P;
            cell_invert[cell_idx] = inv_in + frac * (inv_out - inv_in);
            cell_shape_type[cell_idx] = stype;
            cell_width[cell_idx] = sw;
            cell_height[cell_idx] = sh;
            cell_n[cell_idx] = n_val;
            cell_link_k[cell_idx] = (s == 0) ? k_in : (s == n_sub - 1) ? k_out : 0.0;
            cell_link_area[cell_idx] = (s == 0 || s == n_sub - 1) ? A : 0.0;
            cell_from_node[cell_idx] = link_from_node[i];
            cell_to_node[cell_idx] = link_to_node[i];
            cell_owner_link[cell_idx] = i;
            cell_sub_idx[cell_idx] = s;
            ++cell_idx;
        }
    }

    // Compute precomputed tables for each unique cross-section
    for (int c = 0; c < total_pipe_cells; ++c) {
        XsectKey key{cell_shape_type[c], cell_width[c], cell_height[c]};
        auto it = table_cache.find(key);
        if (it == table_cache.end()) {
            std::vector<double> tbl;
            double A_full_dummy, P_full_dummy;
            pipe1d_compute_table(key.shape_type, key.w, key.h, A_full_dummy, P_full_dummy, tbl);
            it = table_cache.emplace(key, std::move(tbl)).first;
        }
        std::memcpy(&cell_tables[c * 2 * PIPE1D_TABLE_N],
                    it->second.data(), 2 * PIPE1D_TABLE_N * sizeof(double));
    }

    // CSR peer topology: each pipe cell has 2 peers (from_node, to_node)
    std::vector<int32_t> peer_offsets(static_cast<size_t>(total_pipe_cells) + 1, 0);
    for (int32_t c = 0; c < total_pipe_cells; ++c) {
        peer_offsets[static_cast<size_t>(c + 1)] = 2; // each cell has exactly 2 peers
    }
    for (int32_t c = 1; c <= total_pipe_cells; ++c) {
        peer_offsets[static_cast<size_t>(c)] += peer_offsets[static_cast<size_t>(c - 1)];
    }
    const int32_t n_peers = peer_offsets[static_cast<size_t>(total_pipe_cells)];
    std::vector<int32_t> peer_ids(static_cast<size_t>(n_peers));
    std::vector<int32_t> peer_pos = peer_offsets;
    for (int32_t c = 0; c < total_pipe_cells; ++c) {
        const int32_t fn = cell_from_node[static_cast<size_t>(c)];
        const int32_t tn = cell_to_node[static_cast<size_t>(c)];
        peer_ids[static_cast<size_t>(peer_pos[static_cast<size_t>(c)]++)] = fn;
        peer_ids[static_cast<size_t>(peer_pos[static_cast<size_t>(c)]++)] = tn;
    }

    // CSR owned topology: each pipe cell owns exactly 2 interfaces (inlet, outlet)
    // Interface indices: cell i has inlet at 2*i, outlet at 2*i+1
    std::vector<int32_t> owned_offsets(static_cast<size_t>(total_pipe_cells) + 1, 0);
    for (int32_t c = 0; c < total_pipe_cells; ++c) {
        owned_offsets[static_cast<size_t>(c + 1)] = 2;
    }
    for (int32_t c = 1; c <= total_pipe_cells; ++c) {
        owned_offsets[static_cast<size_t>(c)] += owned_offsets[static_cast<size_t>(c - 1)];
    }
    const int32_t n_owned = owned_offsets[static_cast<size_t>(total_pipe_cells)];
    std::vector<int32_t> owned_ids(static_cast<size_t>(n_owned));
    std::vector<int32_t> neighbor_cell(static_cast<size_t>(n_owned));
    std::vector<double> interface_dir(static_cast<size_t>(n_owned));

    // Build neighbor lookup: for each cell, find inlet_neighbor and outlet_neighbor
    // outlet_neighbor: a cell whose from_node == this cell's to_node
    // inlet_neighbor:  a cell whose to_node   == this cell's from_node
    std::vector<int32_t> inlet_neighbor(total_pipe_cells, -1);
    std::vector<int32_t> outlet_neighbor(total_pipe_cells, -1);
    for (int32_t i = 0; i < total_pipe_cells; ++i) {
        const int32_t my_from = cell_from_node[static_cast<size_t>(i)];
        const int32_t my_to   = cell_to_node[static_cast<size_t>(i)];
        for (int32_t j = 0; j < total_pipe_cells; ++j) {
            if (i == j) continue;
            if (cell_from_node[static_cast<size_t>(j)] == my_to) {
                outlet_neighbor[i] = j;
            }
            if (cell_to_node[static_cast<size_t>(j)] == my_from) {
                inlet_neighbor[i] = j;
            }
        }
    }

    for (int32_t c = 0; c < total_pipe_cells; ++c) {
        owned_ids[static_cast<size_t>(2 * c)]     = 2 * c;     // inlet interface
        owned_ids[static_cast<size_t>(2 * c + 1)] = 2 * c + 1; // outlet interface
        neighbor_cell[static_cast<size_t>(2 * c)]     = inlet_neighbor[c];
        neighbor_cell[static_cast<size_t>(2 * c + 1)] = outlet_neighbor[c];
        interface_dir[static_cast<size_t>(2 * c)]     = -1.0;  // inlet
        interface_dir[static_cast<size_t>(2 * c + 1)] = +1.0;  // outlet
    }

    // Allocate device buffers
    alloc_d(reinterpret_cast<void**>(&dev->d_owned_offsets), static_cast<size_t>(total_pipe_cells + 1) * sizeof(int32_t));
    alloc_d(reinterpret_cast<void**>(&dev->d_owned_ids), static_cast<size_t>(n_owned) * sizeof(int32_t));
    alloc_d(reinterpret_cast<void**>(&dev->d_peer_offsets), static_cast<size_t>(total_pipe_cells + 1) * sizeof(int32_t));
    alloc_d(reinterpret_cast<void**>(&dev->d_peer_ids), static_cast<size_t>(n_peers) * sizeof(int32_t));
    alloc_d(reinterpret_cast<void**>(&dev->d_cell_neighbor_cell), static_cast<size_t>(n_owned) * sizeof(int32_t));
    alloc_d(reinterpret_cast<void**>(&dev->d_cell_interface_dir), static_cast<size_t>(n_owned) * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_cell_from_node), static_cast<size_t>(total_pipe_cells) * sizeof(int32_t));
    alloc_d(reinterpret_cast<void**>(&dev->d_cell_to_node), static_cast<size_t>(total_pipe_cells) * sizeof(int32_t));
    alloc_d(reinterpret_cast<void**>(&dev->d_cell_length), static_cast<size_t>(total_pipe_cells) * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_cell_area), static_cast<size_t>(total_pipe_cells) * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_cell_perim), static_cast<size_t>(total_pipe_cells) * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_cell_invert), static_cast<size_t>(total_pipe_cells) * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_cell_n), static_cast<size_t>(total_pipe_cells) * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_cell_link_k), static_cast<size_t>(total_pipe_cells) * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_cell_link_area), static_cast<size_t>(total_pipe_cells) * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_cell_shape_type), static_cast<size_t>(total_pipe_cells) * sizeof(int32_t));
    alloc_d(reinterpret_cast<void**>(&dev->d_cell_width), static_cast<size_t>(total_pipe_cells) * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_cell_owner_link), static_cast<size_t>(total_pipe_cells) * sizeof(int32_t));
    alloc_d(reinterpret_cast<void**>(&dev->d_cell_sub_idx), static_cast<size_t>(total_pipe_cells) * sizeof(int32_t));
    alloc_d(reinterpret_cast<void**>(&dev->d_cell_height), static_cast<size_t>(total_pipe_cells) * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_cell_tables), static_cast<size_t>(total_pipe_cells) * 2 * PIPE1D_TABLE_N * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_node_invert), static_cast<size_t>(n_nodes) * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_node_depth), static_cast<size_t>(n_nodes) * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_node_net_q), static_cast<size_t>(n_nodes) * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_node_surface_area), static_cast<size_t>(n_nodes) * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_node_max_depth), static_cast<size_t>(n_nodes) * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_node_is_boundary), static_cast<size_t>(n_nodes) * sizeof(int32_t));
    alloc_d(reinterpret_cast<void**>(&dev->d_node_is_outfall), static_cast<size_t>(n_nodes) * sizeof(int32_t));
    alloc_d(reinterpret_cast<void**>(&dev->d_node_is_inlet),  static_cast<size_t>(n_nodes) * sizeof(int32_t));
    alloc_d(reinterpret_cast<void**>(&dev->d_A), static_cast<size_t>(total_pipe_cells) * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_Q), static_cast<size_t>(total_pipe_cells) * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_A_prev), static_cast<size_t>(total_pipe_cells) * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_Q_iter), static_cast<size_t>(total_pipe_cells) * sizeof(double));

    // Copy data to device
    copy_h2d_i(dev->d_owned_offsets, owned_offsets.data(), static_cast<size_t>(total_pipe_cells) + 1);
    copy_h2d_i(dev->d_owned_ids, owned_ids.data(), static_cast<size_t>(n_owned));
    copy_h2d_i(dev->d_peer_offsets, peer_offsets.data(), static_cast<size_t>(total_pipe_cells) + 1);
    copy_h2d_i(dev->d_peer_ids, peer_ids.data(), static_cast<size_t>(n_peers));
    copy_h2d_i(dev->d_cell_neighbor_cell, neighbor_cell.data(), static_cast<size_t>(n_owned));
    copy_h2d_d(dev->d_cell_interface_dir, interface_dir.data(), static_cast<size_t>(n_owned));
    copy_h2d_i(dev->d_cell_from_node, cell_from_node.data(), static_cast<size_t>(total_pipe_cells));
    copy_h2d_i(dev->d_cell_to_node, cell_to_node.data(), static_cast<size_t>(total_pipe_cells));
    copy_h2d_d(dev->d_cell_length, cell_length.data(), static_cast<size_t>(total_pipe_cells));
    copy_h2d_d(dev->d_cell_area, cell_area.data(), static_cast<size_t>(total_pipe_cells));
    copy_h2d_d(dev->d_cell_perim, cell_perim.data(), static_cast<size_t>(total_pipe_cells));
    copy_h2d_d(dev->d_cell_invert, cell_invert.data(), static_cast<size_t>(total_pipe_cells));
    copy_h2d_d(dev->d_cell_n, cell_n.data(), static_cast<size_t>(total_pipe_cells));
    copy_h2d_d(dev->d_cell_link_k, cell_link_k.data(), static_cast<size_t>(total_pipe_cells));
    copy_h2d_d(dev->d_cell_link_area, cell_link_area.data(), static_cast<size_t>(total_pipe_cells));
    copy_h2d_i(dev->d_cell_shape_type, cell_shape_type.data(), static_cast<size_t>(total_pipe_cells));
    copy_h2d_d(dev->d_cell_width, cell_width.data(), static_cast<size_t>(total_pipe_cells));
    copy_h2d_i(dev->d_cell_owner_link, cell_owner_link.data(), static_cast<size_t>(total_pipe_cells));
    copy_h2d_i(dev->d_cell_sub_idx, cell_sub_idx.data(), static_cast<size_t>(total_pipe_cells));
    copy_h2d_d(dev->d_cell_height, cell_height.data(), static_cast<size_t>(total_pipe_cells));
    CUDA_CHECK(cudaMemcpy(dev->d_cell_tables, cell_tables.data(),
        static_cast<size_t>(total_pipe_cells) * 2 * PIPE1D_TABLE_N * sizeof(double),
        cudaMemcpyHostToDevice));

    // Upload node invert elevations
    copy_h2d_d(dev->d_node_invert, node_invert_elev, static_cast<size_t>(n_nodes));

    // Upload node surface areas (used by mass-balance kernel)
    copy_h2d_d(dev->d_node_surface_area, node_surface_area, static_cast<size_t>(n_nodes));
    copy_h2d_d(dev->d_node_max_depth, node_max_depth, static_cast<size_t>(n_nodes));

    // Initialize node depth to zero (caller uploads actual depths before each step)
    CUDA_CHECK(cudaMemset(dev->d_node_depth, 0, static_cast<size_t>(n_nodes) * sizeof(double)));
    // Initialize boundary flags to zero; exchange upload marks pipe-end/outfall nodes
    CUDA_CHECK(cudaMemset(dev->d_node_is_boundary, 0, static_cast<size_t>(n_nodes) * sizeof(int32_t)));
    // Initialize outfall flags to zero; caller uploads actual outfall flags
    CUDA_CHECK(cudaMemset(dev->d_node_is_outfall, 0, static_cast<size_t>(n_nodes) * sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(dev->d_node_is_inlet, 0,  static_cast<size_t>(n_nodes) * sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(dev->d_node_net_q, 0, static_cast<size_t>(n_nodes) * sizeof(double)));

    // Initialize pipe cell state: d_A = 0 (dry), d_Q = 0
    // Call swe2d_pipe1d_init_full() from Python if primed/full initial condition
    // is desired (e.g. for perpetual streams or pipes with baseflow).
    CUDA_CHECK(cudaMemset(dev->d_A, 0, static_cast<size_t>(total_pipe_cells) * sizeof(double)));
    CUDA_CHECK(cudaMemset(dev->d_Q, 0, static_cast<size_t>(total_pipe_cells) * sizeof(double)));
    CUDA_CHECK(cudaMemset(dev->d_A_prev, 0, static_cast<size_t>(total_pipe_cells) * sizeof(double)));
    CUDA_CHECK(cudaMemset(dev->d_Q_iter, 0, static_cast<size_t>(total_pipe_cells) * sizeof(double)));
}

/// Device-side lookup: given current area A, full area A_full, full perimeter P_full,
/// and a pointer to the per-cell table [2 * TABLE_N doubles], return wetted perimeter P
/// and top width T.
__device__ __forceinline__ void pipe1d_lookup_geometry(
    double A, double A_full, double P_full,
    const double* table, int table_N,
    double& P, double& T)
{
    if (A <= 0.0) {
        P = 0.0;
        T = 0.0;
        return;
    }
    double frac = A * (1.0 / fmax(1e-20, A_full));
    frac = fmin(1.0, fmax(0.0, frac));
    double f = frac * table_N;
    int idx = min(table_N - 2, max(0, int(f)));
    double t = f - idx;
    P = P_full * (table[idx] + t * (table[idx + 1] - table[idx]));
    T = table[table_N + idx] + t * (table[table_N + idx + 1] - table[table_N + idx]);
}

// ─────────────────────────────────────────────────────────────────────────────
// swe2d_pipe1d_flux_kernel
// One thread per pipe cell. Accumulates discharge at each owned face interface.
// ─────────────────────────────────────────────────────────────────────────────
__global__ __launch_bounds__(256, 1) void swe2d_pipe1d_flux_kernel(
    int32_t                     n_cells,
    const int32_t* __restrict__ owned_offsets,
    const int32_t* __restrict__ owned_ids,
    const int32_t* __restrict__ neighbor_cell,
    const double*  __restrict__ interface_dir,
    const int32_t* __restrict__ cell_from_node,
    const int32_t* __restrict__ cell_to_node,
    const double*  __restrict__ cell_invert,
    const double*  __restrict__ cell_perim,
    const double*  __restrict__ cell_area_full,
    const double*  __restrict__ cell_A,
    const double*  __restrict__ cell_Q,
    const double*  __restrict__ node_invert,
    const double*  __restrict__ node_depth,
    const double*  __restrict__ node_max_depth,
    const int32_t* __restrict__ node_is_inlet,
    const double*  __restrict__ cell_length,
    const double*  __restrict__ cell_height,
    double*                     flux_Q_out,
    double                      g,
    const double*  __restrict__ cell_tables,
    int32_t                     table_N,
    int32_t                     volume_decomposition)
{
    int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells) return;

    double total_flux = 0.0;
    const int32_t start = owned_offsets[c];
    const int32_t end   = owned_offsets[c + 1];

    double P_c, T_c;
    pipe1d_lookup_geometry(cell_A[c], cell_area_full[c], cell_perim[c],
        cell_tables + static_cast<int64_t>(c) * 2 * table_N, table_N, P_c, T_c);
    const double A_c = cell_A[c];
    double H_c = cell_invert[c] + A_c / fmax(1e-10, T_c);

    if (volume_decomposition && cell_height) {
        const int32_t fn = cell_from_node[c];
        const int32_t tn = cell_to_node[c];
        if (fn >= 0 && tn >= 0) {
            const double cell_crown = cell_invert[c] + cell_height[c];
            const double H_fn = node_invert[fn] + node_depth[fn];
            const double H_tn = node_invert[tn] + node_depth[tn];
            // Skip head averaging at inlet nodes — they have prescribed flow,
            // not real storage head.
            const bool fn_inlet = node_is_inlet && node_is_inlet[fn];
            const bool tn_inlet = node_is_inlet && node_is_inlet[tn];
            if (!fn_inlet && !tn_inlet
                && H_fn >= cell_crown - 1e-8 && H_tn >= cell_crown - 1e-8) {
                H_c = 0.5 * (H_fn + H_tn);
            }
        }
    }

    for (int32_t idx = start; idx < end; ++idx) {
        const int32_t k = owned_ids[idx];
        const double  dir = interface_dir[k];
        const int32_t nbr = neighbor_cell[k];

        double H_n, A_n, Q_n;
        // For inlet nodes, use cell head as "neutral" neighbor head —
        // zeros HLLE driving term and avoids head-driven flux into the
        // prescribed-flow BC. The inlet exchange kernel supplies the flow.
        const bool shared_is_inlet = node_is_inlet && (
            (nbr >= 0)
                ? ((dir > 0.0) ? (node_is_inlet[cell_to_node[c]])
                               : (node_is_inlet[cell_from_node[c]]))
                : ((dir > 0.0) ? (node_is_inlet[cell_to_node[c]])
                               : (node_is_inlet[cell_from_node[c]])));
        if (shared_is_inlet) {
            H_n = H_c;
            A_n = cell_A[c];
            Q_n = 0.0;
        } else if (nbr >= 0) {
            // Interior neighbor: head at shared node
            const int32_t from_n = cell_from_node[nbr];
            const int32_t to_n   = cell_to_node[nbr];
            if (dir > 0.0) {
                // Outlet of c (to_node[c]), neighbor shares this node via its from_node
                const int32_t shared_node = cell_to_node[c]; // == from_n
                H_n = node_invert[shared_node] + node_depth[shared_node];
            } else {
                // Inlet of c (from_node[c]), neighbor shares this node via its to_node
                const int32_t shared_node = cell_from_node[c]; // == to_n
                H_n = node_invert[shared_node] + node_depth[shared_node];
            }
            A_n = cell_A[nbr];
            Q_n = cell_Q[nbr];
        } else {
            // Boundary: use node head directly
            if (dir > 0.0) {
                const int32_t shared_node = cell_to_node[c];
                H_n = node_invert[shared_node] + node_depth[shared_node];
            } else {
                const int32_t shared_node = cell_from_node[c];
                H_n = node_invert[shared_node] + node_depth[shared_node];
            }
            A_n = cell_A[c];
            Q_n = cell_Q[c];
        }

        double F;
        if (nbr < 0) {
            // Boundary: skip flux when dry — no water in the pipe cell,
            // so there's nothing to drive across this interface.
            if (cell_A[c] <= 0.0) {
                F = 0.0;
            } else {
                // Boundary: use head-difference flux instead of Q_cell * dir.
                // This drives flow from the pipe cell when the node head differs,
                // enabling flow development from zero initial Q.
                // c_face is in m/s (wave speed), so F = dH*c_face is in m/s.
                // Multiply by full pipe area to get volumetric flux in m³/s.
            const double dH = H_c - H_n;
            const double c_face = sqrt(fmax(0.0, g * fabs(dH))) / fmax(1e-12, cell_length[c]);
            F = dH * c_face * cell_area_full[c];
            }
        } else {
            // HLLE flux
            const double c_wave = sqrt(g * fabs(H_c - H_n) / fmax(1e-12, cell_length[c]));
            F = 0.5 * (cell_Q[c] + Q_n - c_wave * (A_n - cell_A[c]));
        }

        // Accumulate: outlet (+), inlet (-)
        total_flux += dir * F;
    }
    flux_Q_out[c] = total_flux;
}

// ─────────────────────────────────────────────────────────────────────────────
// swe2d_pipe1d_diffusion_wave_kernel
// One thread per pipe cell. Explicit update using Manning's friction only.
// ─────────────────────────────────────────────────────────────────────────────
__global__ __launch_bounds__(256, 1) void swe2d_pipe1d_diffusion_wave_kernel(
    int32_t                     n_cells,
    const double*  __restrict__ cell_length,
    const double*  __restrict__ cell_area_full,
    const double*  __restrict__ cell_perim,
    const double*  __restrict__ cell_n,
    const double*  __restrict__ cell_k_loss,
    const int32_t* __restrict__ cell_from_node,
    const int32_t* __restrict__ cell_to_node,
    const double*  __restrict__ node_invert,
    const double*  __restrict__ node_depth,
    const double*  __restrict__ cell_A,
    const double*  __restrict__ cell_Q,
    const double*  __restrict__ flux_Q,
    double                      dt,
    double                      g,
    double*                     cell_A_new,
    double*                     cell_Q_new,
    const double*  __restrict__ cell_tables,
    int32_t                     table_N)
{
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cells) return;

    const double A_full = cell_area_full[i];
    const double A = cell_A[i];
    const double Q = cell_Q[i];
    const double L = cell_length[i];
    const double P_full = cell_perim[i];
    const double n = cell_n[i];
    const double k_loss = cell_k_loss[i];
    const double absQ = fabs(Q);

    // Head gradient across the cell: diffusion wave is driven by the balance
    // between the pressure gradient and friction/minor losses.  Without this
    // term the solver cannot self-start from Q = 0 because friction is zero
    // when Q is zero.
    const int32_t fn = cell_from_node[i];
    const int32_t tn = cell_to_node[i];
    const double H_from = (fn >= 0) ? node_invert[fn] + node_depth[fn] : 0.0;
    const double H_to   = (tn >= 0) ? node_invert[tn] + node_depth[tn] : 0.0;
    const double dHdx = (H_to - H_from) / fmax(1e-6, L);
    const double pressure_grad = -g * A * dHdx;

    // Wetted perimeter and top width from geometry table
    double P_c, T_c;
    pipe1d_lookup_geometry(A, A_full, P_full,
        cell_tables + static_cast<int64_t>(i) * 2 * table_N, table_N, P_c, T_c);

    // Hydraulic radius (current area / wetted perimeter from table)
    const double R = A / fmax(1e-10, P_c);
    const double R43 = pow(R, 4.0 / 3.0);

    // Implicit Manning friction (minor/expansion losses handled explicitly in
    // swe2d_pipe1d_accumulate_node_flux_kernel; do NOT double-apply here).
    const double cf = g * n * n / (A * R43 + 1e-12);
    const double denom = 1.0 + dt * cf * absQ;
    double Q_new = (Q + dt * pressure_grad) / denom;

    // Clamp Q to reasonable bounds
    const double Q_cap = 1e6;
    Q_new = fmax(-Q_cap, fmin(Q_cap, Q_new));

    // Area update from continuity: dA/dt = -flux_Q/L
    double A_new = A - dt * flux_Q[i] / L;
    // Clamp area: non-negative, cannot exceed full pipe area
    A_new = fmax(0.0, fmin(A_full, A_new));

    cell_A_new[i] = A_new;
    cell_Q_new[i] = Q_new;
}

// ─────────────────────────────────────────────────────────────────────────────
// swe2d_pipe1d_fully_dynamic_kernel
// Semi-implicit solver with pressure gradient term g*A*∂H/∂x. Picard iteration.
// One thread per pipe cell.
// ─────────────────────────────────────────────────────────────────────────────
__global__ __launch_bounds__(256, 1) void swe2d_pipe1d_fully_dynamic_kernel(
    int32_t                     n_cells,
    int32_t                     n_iters,
    double                      relaxation,
    const int32_t* __restrict__ owned_offsets,
    const int32_t* __restrict__ owned_ids,
    const int32_t* __restrict__ neighbor_cell,
    const double*  __restrict__ interface_dir,
    const int32_t* __restrict__ cell_from_node,
    const int32_t* __restrict__ cell_to_node,
    const double*  __restrict__ cell_length,
    const double*  __restrict__ cell_area_full,
    const double*  __restrict__ cell_perim,
    const double*  __restrict__ cell_n,
    const double*  __restrict__ cell_k_loss,
    const double*  __restrict__ node_invert,
    const double*  __restrict__ node_depth,
    const double*  __restrict__ cell_A_prev,
    const double*  __restrict__ cell_Q_prev,
    double*                     cell_A_iter,
    double*                     cell_Q_iter,
    double                      dt,
    double                      g,
    const double*  __restrict__ cell_tables,
    int32_t                     table_N)
{
    int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells) return;

    double A = cell_A_iter[c];
    double Q = cell_Q_iter[c];
    const double L = cell_length[c];
    const double P_full = cell_perim[c];
    const double n = cell_n[c];
    const double k_loss = cell_k_loss[c];
    const double A_full = cell_area_full[c];

    // Piezometric head gradient across this cell: dH/dx = (H_to - H_from) / L
    const int32_t fn = cell_from_node[c];
    const int32_t tn = cell_to_node[c];
    const double H_from = node_invert[fn] + node_depth[fn];
    const double H_to   = node_invert[tn] + node_depth[tn];
    const double dHdx = (H_to - H_from) / fmax(1e-6, L);

    // Wetted perimeter from geometry table
    double P_c, T_c;
    pipe1d_lookup_geometry(A, A_full, P_full,
        cell_tables + static_cast<int64_t>(c) * 2 * table_N, table_N, P_c, T_c);

    // Hydraulic radius for current area
    const double R = A / fmax(1e-10, P_c);
    const double R43 = pow(R, 4.0 / 3.0);
    const double absQ = fabs(Q);

    // Pressure gradient term: -g * A * dH/dx
    const double pressure_grad = -g * A * dHdx;

    // Implicit Manning friction + HEC-22 minor loss.
    const double cf = g * n * n / (A * R43 + 1e-12);
    const double cm = g * k_loss / (2.0 * A * A * L + 1e-12);
    const double denom = 1.0 + dt * cf * absQ + dt * cm * absQ;
    double Q_new = (Q + dt * pressure_grad) / denom;

    // Clamp Q
    const double Q_cap = 1e6;
    Q_new = fmax(-Q_cap, fmin(Q_cap, Q_new));

    // Relaxation with previous step (or previous iteration)
    if (relaxation > 0.0 && relaxation < 1.0) {
        Q_new = (1.0 - relaxation) * cell_Q_prev[c] + relaxation * Q_new;
    }

    // Area update from continuity: dA/dt = -flux_Q/L
    // For fully dynamic, we use the current Q for the flux
    // Net flux = Q_out - Q_in (Q_out = Q for this cell, Q_in from inlet neighbor)
    double Q_in = 0.0;
    const int32_t inlet_iface = 2 * c; // inlet interface index
    const int32_t inlet_nbr = neighbor_cell[inlet_iface];
    if (inlet_nbr >= 0) {
        Q_in = cell_Q_iter[inlet_nbr];
    }
    const double Q_net = Q - Q_in;
    double A_new = A - dt * Q_net / L;
    A_new = fmax(0.0, fmin(A_full, A_new));

    cell_A_iter[c] = A_new;
    cell_Q_iter[c] = Q_new;
}

// ─────────────────────────────────────────────────────────────────────────────
// Host wrappers for pipe1d kernels
// ─────────────────────────────────────────────────────────────────────────────
#define BLOCK 256

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
    const double*         cell_area_full,
    const double*         cell_A,
    const double*         cell_Q,
    const double*         node_invert,
    const double*         node_depth,
    const double*         node_max_depth,
    const int32_t*        node_is_inlet,
    const double*         cell_length,
    const double*         cell_height,
    double*               flux_Q_out,
    double                g,
    const double*         cell_tables,
    int32_t               table_N,
    int32_t               volume_decomposition)
{
    const int32_t n_blocks = (n_cells + BLOCK - 1) / BLOCK;
    swe2d_pipe1d_flux_kernel<<<n_blocks, BLOCK>>>(
        n_cells, owned_offsets, owned_ids, neighbor_cell, interface_dir,
        cell_from_node, cell_to_node, cell_invert, cell_perim,
        cell_area_full, cell_A, cell_Q, node_invert, node_depth,
        node_max_depth, node_is_inlet, cell_length, cell_height,
        flux_Q_out, g, cell_tables, table_N, volume_decomposition);
    CUDA_CHECK(cudaGetLastError());
}

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
    int32_t               table_N)
{
    const int32_t n_blocks = (n_cells + BLOCK - 1) / BLOCK;
    swe2d_pipe1d_diffusion_wave_kernel<<<n_blocks, BLOCK>>>(
        n_cells, cell_length, cell_area_full, cell_perim, cell_n, cell_k_loss,
        cell_from_node, cell_to_node, node_invert, node_depth,
        cell_A, cell_Q, flux_Q, dt, g, cell_A_new, cell_Q_new,
        cell_tables, table_N);
    CUDA_CHECK(cudaGetLastError());
}

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
    double                g,
    const double*         cell_tables,
    int32_t               table_N)
{
    const int32_t n_blocks = (n_cells + BLOCK - 1) / BLOCK;
    for (int32_t iter = 0; iter < n_iters; ++iter) {
        swe2d_pipe1d_fully_dynamic_kernel<<<n_blocks, BLOCK>>>(
            n_cells, n_iters, relaxation, owned_offsets, owned_ids,
            neighbor_cell, interface_dir, cell_from_node, cell_to_node,
            cell_length, cell_area_full, cell_perim, cell_n, cell_k_loss,
            node_invert, node_depth, cell_A_prev, cell_Q_prev,
            cell_A_iter, cell_Q_iter, dt, g, cell_tables, table_N);
        CUDA_CHECK(cudaGetLastError());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Node mass-balance kernels: update d_node_depth from pipe flows
// ─────────────────────────────────────────────────────────────────────────────

/** Accumulate net pipe flux into each node. 1 thread per cell.
 *  Q > 0 means flow from from_node to to_node.
 *  HEC-22 entrance/exit losses are applied at boundary cells:
 *    h_loss = k * V^2 / (2g) = k * Q^2 / (2 * g * A_actual^2)
 *  Loss always opposes motion — it reduces |Q| towards zero but can never
 *  reverse flow direction (loss is a sink, not a source). */
__global__ __launch_bounds__(256, 4) void swe2d_pipe1d_accumulate_node_flux_kernel(
    int32_t                     n_cells,
    const int32_t* __restrict__ cell_from_node,
    const int32_t* __restrict__ cell_to_node,
    const double*  __restrict__ cell_Q,
    const double*  __restrict__ cell_A,
    const double*  __restrict__ cell_link_k,
    double                      g,
    double*                     node_net_q)
{
    int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells) return;

    const double Q = cell_Q[c];
    const int32_t fn = cell_from_node[c];
    const int32_t tn = cell_to_node[c];

    const double k = cell_link_k[c];
    const double A_actual = cell_A[c];
    double Q_eff = Q;
    if (k > 0.0 && A_actual > 0.0) {
        const double loss_mag = k * Q * Q / (2.0 * g * A_actual * A_actual + 1e-12);
        if (Q > 0.0)
            Q_eff = fmax(0.0, Q - loss_mag);
        else
            Q_eff = fmin(0.0, Q + loss_mag);
    }

    // Q > 0: flow leaves from_node, arrives at to_node
    if (fn >= 0) atomicAdd(&node_net_q[fn], -Q_eff);
    if (tn >= 0) atomicAdd(&node_net_q[tn],  Q_eff);
}

/** Update node depth from accumulated net flux. 1 thread per node.
 *  Boundary nodes (pipe-ends, outfalls) are skipped — they have no storage;
 *  their depth is set by the BC kernel or outfall kernel, not by mass balance. */
__global__ __launch_bounds__(256, 4) void swe2d_pipe1d_update_node_depth_kernel(
    int32_t           n_nodes,
    const double* __restrict__ node_net_q,
    const double* __restrict__ node_surface_area,
    const int32_t* __restrict__ node_is_boundary,
    const double* __restrict__ node_max_depth,
    double*                     node_depth,
    double                     dt)
{
    int32_t n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= n_nodes) return;

    if (node_is_boundary && node_is_boundary[n]) return;

    const double area = fmax(1e-6, node_surface_area[n]);
    const double dh = dt * node_net_q[n] / area;
    double d = node_depth[n] + dh;
    d = fmax(0.0, fmin(d, node_max_depth[n]));
    node_depth[n] = d;
}

/** GPU kernel: mark nodes that have inlet assignments.
 *  1 thread per inlet.  Uses atomicExch so the final value is always 1
 *  (multiple inlets at the same node will all write the same value).
 *  @global */
__global__ void swe2d_mark_inlet_nodes_kernel(
    int32_t n_inlets,
    const int32_t* __restrict__ inlet_node,
    int32_t n_nodes,
    int32_t* __restrict__ node_is_inlet)
{
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_inlets) return;
    const int32_t n = inlet_node[i];
    if (n >= 0 && n < n_nodes) {
        atomicExch(&node_is_inlet[n], 1);
    }
}

// ── Host wrappers ───────────────────────────────────────────────────────────

static void swe2d_pipe1d_node_mass_balance_host(
    SWE2DDeviceState* dev, double dt, double g)
{
    if (!dev) return;
    auto& p = dev->pipe1d;
    const int32_t n_cells = p.n_pipe_cells;
    const int32_t n_nodes = p.n_nodes;
    if (n_cells <= 0 || n_nodes <= 0) return;
    cudaStream_t stream = dev->d_stream;

    // Zero net flux accumulator
    CUDA_CHECK(cudaMemsetAsync(p.d_node_net_q, 0,
        static_cast<size_t>(n_nodes) * sizeof(double), stream));

    // Accumulate: each cell atomicAdds its Q to from/to nodes
    {
        const int32_t grid = (n_cells + BLOCK - 1) / BLOCK;
        swe2d_pipe1d_accumulate_node_flux_kernel<<<grid, BLOCK, 0, stream>>>(
            n_cells, p.d_cell_from_node, p.d_cell_to_node,
            p.d_Q, p.d_A, p.d_cell_link_k,
            g, p.d_node_net_q);
        CUDA_CHECK(cudaGetLastError());
    }

    // Update node depth (skip boundary nodes — no storage)
    {
        const int32_t grid = (n_nodes + BLOCK - 1) / BLOCK;
        swe2d_pipe1d_update_node_depth_kernel<<<grid, BLOCK, 0, stream>>>(
            n_nodes, p.d_node_net_q, p.d_node_surface_area,
            p.d_node_is_boundary, p.d_node_max_depth, p.d_node_depth, dt);
        CUDA_CHECK(cudaGetLastError());
    }
}

// ── Upload node depths from host to device (called before each step) ────────

void swe2d_pipe1d_upload_node_depth(
    SWE2DDeviceState* dev,
    const double*     host_node_depth,
    int32_t           n_nodes)
{
    if (!dev || !host_node_depth || n_nodes <= 0) return;
    auto& p = dev->pipe1d;
    if (n_nodes != p.n_nodes) return;
    CUDA_CHECK(cudaMemcpy(p.d_node_depth, host_node_depth,
        static_cast<size_t>(n_nodes) * sizeof(double),
        cudaMemcpyHostToDevice));
}

// ── Initialize pipe cell area from uploaded node depths ────────────────────

void swe2d_pipe1d_init_area_from_depth(Pipe1DDeviceState* dev)
{
    int32_t nc = dev->n_pipe_cells;
    int32_t nn = dev->n_nodes;
    if (nc <= 0 || nn <= 0) return;

    std::vector<int32_t> cell_from(nc);
    std::vector<int32_t> cell_to(nc);
    std::vector<double> cell_area_full(nc);
    std::vector<double> cell_width(nc);
    std::vector<double> cell_height(nc);
    std::vector<int32_t> cell_shape(nc);
    std::vector<double> node_depth(nn);

    CUDA_CHECK(cudaMemcpy(cell_from.data(), dev->d_cell_from_node, nc * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cell_to.data(), dev->d_cell_to_node, nc * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cell_area_full.data(), dev->d_cell_area, nc * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cell_width.data(), dev->d_cell_width, nc * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cell_height.data(), dev->d_cell_height, nc * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cell_shape.data(), dev->d_cell_shape_type, nc * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(node_depth.data(), dev->d_node_depth, nn * sizeof(double), cudaMemcpyDeviceToHost));

    std::vector<double> init_A(nc, 0.0);
    constexpr double PIPE_DEPTH_MIN = 1.0e-4;
    for (int c = 0; c < nc; ++c) {
        int fn = cell_from[c];
        int tn = cell_to[c];
        double d_fn = (fn >= 0 && fn < nn) ? node_depth[fn] : 0.0;
        double d_tn = (tn >= 0 && tn < nn) ? node_depth[tn] : 0.0;
        double depth = 0.5 * (d_fn + d_tn);
        if (depth <= PIPE_DEPTH_MIN) { init_A[c] = 0.0; continue; }
        double A_full = cell_area_full[c];
        double full_depth = (cell_shape[c] == 0) ? cell_width[c] : cell_height[c];
        full_depth = fmax(1e-10, full_depth);
        double frac = fmin(1.0, depth / full_depth);
        init_A[c] = A_full * frac;
    }
    CUDA_CHECK(cudaMemcpy(dev->d_A, init_A.data(), nc * sizeof(double), cudaMemcpyHostToDevice));
}

void swe2d_pipe1d_init_full(Pipe1DDeviceState* dev)
{
    int32_t nc = dev->n_pipe_cells;
    if (nc <= 0) return;
    std::vector<double> full_A(nc);
    CUDA_CHECK(cudaMemcpy(full_A.data(), dev->d_cell_area, nc * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dev->d_A, full_A.data(), nc * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev->d_A_prev, full_A.data(), nc * sizeof(double), cudaMemcpyHostToDevice));
}


// ── Readback node state for diagnostics/tests ───────────────────────────────

void swe2d_pipe1d_readback_node_state(
    SWE2DDeviceState* dev,
    double*           host_node_depth,
    double*           host_cell_A,
    double*           host_cell_Q,
    double*           host_cell_width,
    double*           host_cell_height,
    int32_t*          host_cell_shape_type,
    double*           host_cell_invert,
    int32_t*          host_cell_owner_link,
    int32_t*          host_cell_sub_idx,
    int32_t           n_nodes,
    int32_t           n_cells)
{
    if (!dev) return;
    auto& p = dev->pipe1d;
    if (host_node_depth && n_nodes > 0 && n_nodes == p.n_nodes) {
        CUDA_CHECK(cudaMemcpy(host_node_depth, p.d_node_depth,
            static_cast<size_t>(n_nodes) * sizeof(double),
            cudaMemcpyDeviceToHost));
    }
    if (host_cell_A && n_cells > 0 && n_cells == p.n_pipe_cells) {
        CUDA_CHECK(cudaMemcpy(host_cell_A, p.d_A,
            static_cast<size_t>(n_cells) * sizeof(double),
            cudaMemcpyDeviceToHost));
    }
    if (host_cell_Q && n_cells > 0 && n_cells == p.n_pipe_cells) {
        CUDA_CHECK(cudaMemcpy(host_cell_Q, p.d_Q,
            static_cast<size_t>(n_cells) * sizeof(double),
            cudaMemcpyDeviceToHost));
    }
    if (host_cell_width && n_cells > 0 && n_cells == p.n_pipe_cells) {
        CUDA_CHECK(cudaMemcpy(host_cell_width, p.d_cell_width,
            static_cast<size_t>(n_cells) * sizeof(double),
            cudaMemcpyDeviceToHost));
    }
    if (host_cell_height && n_cells > 0 && n_cells == p.n_pipe_cells) {
        CUDA_CHECK(cudaMemcpy(host_cell_height, p.d_cell_height,
            static_cast<size_t>(n_cells) * sizeof(double),
            cudaMemcpyDeviceToHost));
    }
    if (host_cell_shape_type && n_cells > 0 && n_cells == p.n_pipe_cells) {
        CUDA_CHECK(cudaMemcpy(host_cell_shape_type, p.d_cell_shape_type,
            static_cast<size_t>(n_cells) * sizeof(int32_t),
            cudaMemcpyDeviceToHost));
    }
    if (host_cell_invert && n_cells > 0 && n_cells == p.n_pipe_cells) {
        CUDA_CHECK(cudaMemcpy(host_cell_invert, p.d_cell_invert,
            static_cast<size_t>(n_cells) * sizeof(double),
            cudaMemcpyDeviceToHost));
    }
    if (host_cell_owner_link && n_cells > 0 && n_cells == p.n_pipe_cells) {
        CUDA_CHECK(cudaMemcpy(host_cell_owner_link, p.d_cell_owner_link,
            static_cast<size_t>(n_cells) * sizeof(int32_t),
            cudaMemcpyDeviceToHost));
    }
    if (host_cell_sub_idx && n_cells > 0 && n_cells == p.n_pipe_cells) {
        CUDA_CHECK(cudaMemcpy(host_cell_sub_idx, p.d_cell_sub_idx,
            static_cast<size_t>(n_cells) * sizeof(int32_t),
            cudaMemcpyDeviceToHost));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// swe2d_pipe1d_step
// ─────────────────────────────────────────────────────────────────────────────
void swe2d_pipe1d_step(
    SWE2DDeviceState* dev,
    double            dt,
    const char*       solver_mode,
    int32_t           coupling_substeps,
    int32_t           implicit_iters,
    double            relaxation,
    double            g)
{
    if (!dev || !dev->pipe1d.d_A) return;

    auto& p = dev->pipe1d;
    const int32_t n_cells = p.n_pipe_cells;

    const double* d_cell_tables = p.d_cell_tables;
    const int32_t table_N = PIPE1D_TABLE_N;

    double* d_flux_Q = nullptr;
    CUDA_CHECK(cudaMalloc(&d_flux_Q, static_cast<size_t>(n_cells) * sizeof(double)));

    double* d_A_new = nullptr;
    double* d_Q_new = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A_new, static_cast<size_t>(n_cells) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Q_new, static_cast<size_t>(n_cells) * sizeof(double)));

    const double local_dt = dt / static_cast<double>(coupling_substeps);

    for (int32_t sub = 0; sub < coupling_substeps; ++sub) {
        // d_A_new/d_Q_new are freshly allocated each substep — initialize from current state
        CUDA_CHECK(cudaMemcpy(d_A_new, p.d_A, static_cast<size_t>(n_cells) * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_Q_new, p.d_Q, static_cast<size_t>(n_cells) * sizeof(double), cudaMemcpyDeviceToDevice));

        swe2d_pipe1d_flux_kernel_host(
            n_cells,
            p.d_owned_offsets, p.d_owned_ids,
            p.d_cell_neighbor_cell, p.d_cell_interface_dir,
            p.d_cell_from_node, p.d_cell_to_node,
            p.d_cell_invert, p.d_cell_perim,
            p.d_cell_area,
            p.d_A, p.d_Q,
            p.d_node_invert, p.d_node_depth,
            p.d_node_max_depth,
            p.d_node_is_inlet,
            p.d_cell_length, p.d_cell_height,
            d_flux_Q, g,
            d_cell_tables, table_N,
            1);

        if (std::strcmp(solver_mode, "fully_dynamic") == 0) {
            swe2d_pipe1d_fully_dynamic_kernel_host(
                n_cells,
                implicit_iters,
                relaxation,
                p.d_owned_offsets, p.d_owned_ids,
                p.d_cell_neighbor_cell, p.d_cell_interface_dir,
                p.d_cell_from_node, p.d_cell_to_node,
                p.d_cell_length, p.d_cell_area,
                p.d_cell_perim, p.d_cell_n,
                p.d_cell_link_k,
                p.d_node_invert, p.d_node_depth,
                p.d_A, p.d_Q,
                d_A_new, d_Q_new,
                local_dt, g,
                d_cell_tables, table_N);
        } else {
            swe2d_pipe1d_diffusion_wave_kernel_host(
                n_cells,
                p.d_cell_length, p.d_cell_area,
                p.d_cell_perim, p.d_cell_n,
                p.d_cell_link_k,
                p.d_cell_from_node, p.d_cell_to_node,
                p.d_node_invert, p.d_node_depth,
                p.d_A, p.d_Q, d_flux_Q,
                local_dt, g,
                d_A_new, d_Q_new,
                d_cell_tables, table_N);
        }

        CUDA_CHECK(cudaMemcpy(p.d_A, d_A_new, static_cast<size_t>(n_cells) * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(p.d_Q, d_Q_new, static_cast<size_t>(n_cells) * sizeof(double), cudaMemcpyDeviceToDevice));
    }

    // Update node depths from pipe flows (mass balance on device)
    swe2d_pipe1d_node_mass_balance_host(dev, dt, g);

    cudaFree(d_flux_Q);
    cudaFree(d_A_new);
    cudaFree(d_Q_new);
}

// ─────────────────────────────────────────────────────────────────────────────
// Drainage pipe-end kernels (not currently called — preserved from swe2d_gpu.cu)
// ─────────────────────────────────────────────────────────────────────────────

/// GPU kernel: apply pipe-end boundary condition (WSE coupling surface↔network).
/**
 * 1 thread per pipe end.  Computes effective depth boundary condition
 * for the drainage node based on surface WSE, node head, and loss
 * coefficients.  Used to couple 1D drainage network to 2D SWE cells.
 *
 * @global
 */
__global__ __launch_bounds__(256, 4) void swe2d_drainage_pipe_end_bc_kernel(
    int32_t n_pipe_ends,
    int32_t n_cells,
    const int32_t* __restrict__ pipe_end_cell,
    const int32_t* __restrict__ pipe_end_node,
    const double* __restrict__ pipe_end_invert_elev,
    const double* __restrict__ pipe_end_diameter,
    const double* __restrict__ pipe_end_area,
    const double* __restrict__ pipe_end_inlet_loss_k,
    const double* __restrict__ pipe_end_outlet_loss_k,
    const double* __restrict__ cell_wse,
    const double* __restrict__ cell_h,
    double h_min,
    const double* __restrict__ node_invert_elev,
    const double* __restrict__ node_surface_area,
    const double* __restrict__ node_qleave,
    double gravity,
    double* __restrict__ node_depth,
    double* __restrict__ pipe_end_depth_bc,
    double* __restrict__ pipe_end_node_area)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_pipe_ends) return;
    const int32_t c = pipe_end_cell[i];
    const int32_t n = pipe_end_node[i];
    if (c < 0 || c >= n_cells || n < 0) {
        pipe_end_depth_bc[i] = 0.0;
        pipe_end_node_area[i] = 1.0;
        return;
    }

    const double h_surface = cell_h ? cell_h[c] : 0.0;
    if (h_surface <= h_min) {
        pipe_end_depth_bc[i] = 0.0;
        pipe_end_node_area[i] = fmax(1.0, node_surface_area[n]);
        return;
    }

    const double invert = pipe_end_invert_elev[i];
    const double area_node = fmax(1.0, node_surface_area[n]);
    const double wse_surface = cell_wse[c];
    const double node_head = node_invert_elev[n] + fmax(0.0, node_depth[n]);

    double area_pipe = fmax(0.0, pipe_end_area[i]);
    if (area_pipe <= 0.0) {
        const double d_pipe = fmax(0.0, pipe_end_diameter[i]);
        area_pipe = (d_pipe > 0.0) ? (0.25 * M_PI * d_pipe * d_pipe) : 0.0;
    }

    const double q_leave = node_qleave ? node_qleave[n] : 0.0;
    // q_leave convention: positive = water leaving node (node→surface).
    // When q_leave > 0, flow is FROM network TO surface → use outlet loss (k_out).
    // When q_leave < 0, flow is FROM surface TO network → use inlet loss (k_in).
    bool flow_surface_to_network = false;
    if (fabs(q_leave) <= 1.0e-12) {
        flow_surface_to_network = (wse_surface >= node_head);
    } else {
        flow_surface_to_network = (q_leave <= 0.0);
    }

    const double k_in = fmax(0.0, pipe_end_inlet_loss_k[i]);
    const double k_out = fmax(0.0, pipe_end_outlet_loss_k[i]);
    const double k_use = flow_surface_to_network ? k_in : k_out;

    double h_loss = 0.0;
    if (area_pipe > 0.0) {
        const double vel = fabs(q_leave) / fmax(area_pipe, 1.0e-12);
        h_loss = k_use * vel * vel / (2.0 * fmax(gravity, 1.0e-9));
    }
    const double wse_eff = fmax(invert, wse_surface - h_loss);
    // Cap BC depth to the available water depth in the surface cell.
    // Without this cap, a cell whose bed is above the pipe invert would
    // create a phantom depth from WSE = bed + h even when h is near zero,
    // causing the pipe network to start "full" from non-existent water.
    const double d_bc = fmin(h_surface, fmax(0.0, wse_eff - invert));
    node_depth[n] = d_bc;
    pipe_end_depth_bc[i] = d_bc;
    pipe_end_node_area[i] = area_node;
}

/// GPU kernel: force free outfall node depth to zero.
/** 1 thread per node.  Outfall nodes are treated as free discharge boundaries:
 *  tailwater head equals the invert, so node_depth is forced to zero. */
__global__ __launch_bounds__(256, 4) void swe2d_outfall_free_bc_kernel(
    int32_t n_nodes,
    const int32_t* __restrict__ node_is_outfall,
    double* __restrict__ node_depth)
{
    int32_t n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= n_nodes) return;
    if (node_is_outfall && node_is_outfall[n]) {
        node_depth[n] = 0.0;
    }
}

/// GPU kernel: exchange flow between pipe end and 2D surface cell.
/**
 * 1 thread per pipe end.  Uses the net node discharge computed by the pipe
 * solver's mass balance (node_net_q) as the exchange flux.  Positive
 * node_net_q means flow is entering the node (pipe -> surface), so the
 * surface cell gains water; negative node_net_q means flow is leaving the
 * node (surface -> pipe), so the cell loses water.
 *
 * @global
 */
__global__ __launch_bounds__(256, 4) void swe2d_drainage_pipe_end_exchange_kernel(
    int32_t n_pipe_ends,
    int32_t n_cells,
    const int32_t* __restrict__ pipe_end_cell,
    const int32_t* __restrict__ pipe_end_node,
    const double* __restrict__ pipe_end_node_area,
    const double* __restrict__ cell_area,
    const double* __restrict__ cell_depth,
    const double* __restrict__ node_net_q,
    double dt_s,
    double* __restrict__ q_cell,
    double* __restrict__ limiter_event_count,
    double* __restrict__ limiter_volume_m3,
    const int32_t* __restrict__ pipe_end_enable_overflow,
    const double*  __restrict__ pipe_end_max_overflow_rate)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_pipe_ends) return;
    const int32_t c = pipe_end_cell[i];
    const int32_t n = pipe_end_node[i];
    if (c < 0 || c >= n_cells || n < 0) return;

    double q_net = node_net_q ? node_net_q[n] : 0.0;

    if (q_net > 0.0) {
        const int32_t overflow_enabled = pipe_end_enable_overflow ? pipe_end_enable_overflow[i] : 1;
        if (!overflow_enabled) {
            if (limiter_event_count) atomicAdd(limiter_event_count, 1.0);
            if (limiter_volume_m3) atomicAdd(limiter_volume_m3, q_net * dt_s);
            return;
        }
        if (pipe_end_max_overflow_rate && pipe_end_max_overflow_rate[i] > 0.0
            && q_net > pipe_end_max_overflow_rate[i]) {
            if (limiter_event_count) atomicAdd(limiter_event_count, 1.0);
            if (limiter_volume_m3)
                atomicAdd(limiter_volume_m3, (q_net - pipe_end_max_overflow_rate[i]) * dt_s);
            q_net = pipe_end_max_overflow_rate[i];
        }
        atomicAdd(&q_cell[c], q_net);
    } else if (q_net < 0.0) {
        // Surface -> pipe.  Limit by available surface water.
        double q_in = -q_net;
        if (cell_depth && cell_area) {
            const double avail_surface_vol = fmax(0.0, cell_depth[c]) * fmax(0.0, cell_area[c]);
            const double q_cap_surface = (dt_s > 0.0) ? (avail_surface_vol / dt_s) : 0.0;
            if (q_in > q_cap_surface) {
                if (limiter_event_count) atomicAdd(limiter_event_count, 1.0);
                if (limiter_volume_m3) atomicAdd(limiter_volume_m3, fmax(0.0, q_in - q_cap_surface) * dt_s);
                q_in = q_cap_surface;
            }
        }
        atomicAdd(&q_cell[c], -q_in);
    }
}

// ── Host wrappers for pipe-end exchange kernels ──────────────────────────

void swe2d_drainage_pipe_end_bc_kernel_host(
    int32_t n_pipe_ends, int32_t n_cells,
    const int32_t* pipe_end_cell, const int32_t* pipe_end_node,
    const double* pipe_end_invert, const double* pipe_end_diameter,
    const double* pipe_end_area,
    const double* pipe_end_kin, const double* pipe_end_kout,
    const double* cell_wse, const double* cell_h, double h_min,
    const double* node_invert,
    const double* node_surface_area, const double* node_qleave,
    double gravity,
    double* node_depth, double* pipe_end_depth_bc, double* pipe_end_node_area)
{
    if (n_pipe_ends <= 0) return;
    const int PE_BLOCK = 256;
    int grid = (n_pipe_ends + PE_BLOCK - 1) / PE_BLOCK;
    swe2d_drainage_pipe_end_bc_kernel<<<grid, PE_BLOCK>>>(
        n_pipe_ends, n_cells,
        pipe_end_cell, pipe_end_node, pipe_end_invert,
        pipe_end_diameter, pipe_end_area,
        pipe_end_kin, pipe_end_kout,
        cell_wse, cell_h, h_min,
        node_invert, node_surface_area, node_qleave,
        gravity, node_depth, pipe_end_depth_bc, pipe_end_node_area);
    CUDA_CHECK(cudaGetLastError());
}

void swe2d_outfall_free_bc_kernel_host(
    int32_t n_nodes,
    const int32_t* node_is_outfall,
    double* node_depth)
{
    if (n_nodes <= 0) return;
    const int BC_BLOCK = 256;
    int grid = (n_nodes + BC_BLOCK - 1) / BC_BLOCK;
    swe2d_outfall_free_bc_kernel<<<grid, BC_BLOCK>>>(n_nodes, node_is_outfall, node_depth);
    CUDA_CHECK(cudaGetLastError());
}

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
    const double*  pipe_end_max_overflow_rate)
{
    if (n_pipe_ends <= 0) return;
    const int PE_BLOCK = 256;
    int grid = (n_pipe_ends + PE_BLOCK - 1) / PE_BLOCK;
    swe2d_drainage_pipe_end_exchange_kernel<<<grid, PE_BLOCK>>>(
        n_pipe_ends, n_cells,
        pipe_end_cell, pipe_end_node,
        pipe_end_node_area,
        cell_area, cell_depth,
        node_net_q, dt_s,
        q_cell,
        limiter_event_count, limiter_volume_m3,
        pipe_end_enable_overflow,
        pipe_end_max_overflow_rate);
    CUDA_CHECK(cudaGetLastError());
}
