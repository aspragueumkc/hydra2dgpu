#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

struct Bounds {
    double xmin = 0.0;
    double ymin = 0.0;
    double xmax = 0.0;
    double ymax = 0.0;
};

struct Constraint {
    std::vector<std::array<double, 2>> ring;
    double target_size = 1.0;
    std::string cell_type = "triangular";
};

struct ChannelGuide {
    bool has_frame = false;
    bool has_centerline = false;
    std::vector<std::array<double, 2>> centerline;
    std::vector<double> station;
    std::vector<std::array<double, 2>> seg_tangent;
    std::vector<std::array<double, 2>> left_bank;
    std::vector<std::array<double, 2>> right_bank;
    bool has_bank_bounds = false;
    double left_n = 0.0;
    double right_n = 0.0;
};

inline std::string to_lower(const std::string& s) {
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return out;
}

inline bool is_quad_like(const std::string& cell_type) {
    const std::string ct = to_lower(cell_type);
    return ct == "quadrilateral" || ct == "cartesian" || ct == "channel_generator";
}

inline bool is_empty(const std::string& cell_type) {
    return to_lower(cell_type) == "empty";
}

Bounds ring_bounds(const std::vector<std::array<double, 2>>& ring) {
    Bounds b;
    if (ring.empty()) {
        return b;
    }
    b.xmin = b.xmax = ring[0][0];
    b.ymin = b.ymax = ring[0][1];
    for (const auto& p : ring) {
        b.xmin = std::min(b.xmin, p[0]);
        b.ymin = std::min(b.ymin, p[1]);
        b.xmax = std::max(b.xmax, p[0]);
        b.ymax = std::max(b.ymax, p[1]);
    }
    return b;
}

bool point_in_ring(double x, double y, const std::vector<std::array<double, 2>>& ring) {
    if (ring.size() < 3) {
        return false;
    }
    bool inside = false;
    size_t j = ring.size() - 1;
    for (size_t i = 0; i < ring.size(); ++i) {
        const double xi = ring[i][0], yi = ring[i][1];
        const double xj = ring[j][0], yj = ring[j][1];
        const bool intersects = ((yi > y) != (yj > y));
        if (intersects) {
            const double denom = (yj - yi);
            if (std::abs(denom) > 1.0e-20) {
                const double x_on_edge = xi + (y - yi) * (xj - xi) / denom;
                if (x < x_on_edge) {
                    inside = !inside;
                }
            }
        }
        j = i;
    }
    return inside;
}

inline double dot2(double ax, double ay, double bx, double by) {
    return ax * bx + ay * by;
}

inline double clamp01(double v) {
    if (v < 0.0) {
        return 0.0;
    }
    if (v > 1.0) {
        return 1.0;
    }
    return v;
}

bool build_centerline(ChannelGuide* g, const std::vector<std::array<double, 2>>& pts) {
    if (g == nullptr || pts.size() < 2) {
        return false;
    }
    g->centerline.clear();
    g->station.clear();
    g->seg_tangent.clear();

    g->centerline.push_back(pts[0]);
    for (size_t i = 1; i < pts.size(); ++i) {
        const double dx = pts[i][0] - g->centerline.back()[0];
        const double dy = pts[i][1] - g->centerline.back()[1];
        if (std::hypot(dx, dy) > 1.0e-9) {
            g->centerline.push_back(pts[i]);
        }
    }
    if (g->centerline.size() < 2) {
        return false;
    }

    g->station.resize(g->centerline.size(), 0.0);
    g->seg_tangent.resize(g->centerline.size() - 1, {1.0, 0.0});
    for (size_t i = 0; i + 1 < g->centerline.size(); ++i) {
        const double vx = g->centerline[i + 1][0] - g->centerline[i][0];
        const double vy = g->centerline[i + 1][1] - g->centerline[i][1];
        const double len = std::hypot(vx, vy);
        if (len <= 1.0e-12) {
            return false;
        }
        g->station[i + 1] = g->station[i] + len;
        g->seg_tangent[i] = {vx / len, vy / len};
    }
    g->has_centerline = true;
    g->has_frame = true;
    return true;
}

bool sample_frame_at_station(const ChannelGuide& g, double s, double* cx, double* cy, double* tx, double* ty) {
    if (!g.has_centerline || g.centerline.size() < 2 || g.station.size() < 2 || g.seg_tangent.empty()) {
        return false;
    }
    const double smax = g.station.back();
    const double sc = std::min(std::max(0.0, s), smax);

    size_t seg = 0;
    for (size_t i = 0; i + 1 < g.station.size(); ++i) {
        if (sc <= g.station[i + 1] + 1.0e-12) {
            seg = i;
            break;
        }
        seg = i;
    }

    const double s0 = g.station[seg];
    const double s1 = g.station[seg + 1];
    const double len = std::max(1.0e-12, s1 - s0);
    const double t = clamp01((sc - s0) / len);

    const auto& p0 = g.centerline[seg];
    const auto& p1 = g.centerline[seg + 1];
    if (cx) {
        *cx = p0[0] + t * (p1[0] - p0[0]);
    }
    if (cy) {
        *cy = p0[1] + t * (p1[1] - p0[1]);
    }
    if (tx) {
        *tx = g.seg_tangent[seg][0];
    }
    if (ty) {
        *ty = g.seg_tangent[seg][1];
    }
    return true;
}

bool project_point_to_centerline(const ChannelGuide& g, double x, double y, double* s_out, double* n_out) {
    if (!g.has_centerline || g.centerline.size() < 2 || g.station.size() < 2) {
        return false;
    }

    double best_d2 = std::numeric_limits<double>::infinity();
    double best_s = 0.0;
    double best_n = 0.0;

    for (size_t i = 0; i + 1 < g.centerline.size(); ++i) {
        const auto& a = g.centerline[i];
        const auto& b = g.centerline[i + 1];
        const double vx = b[0] - a[0];
        const double vy = b[1] - a[1];
        const double len2 = vx * vx + vy * vy;
        if (len2 <= 1.0e-18) {
            continue;
        }
        const double inv_len2 = 1.0 / len2;
        const double wx = x - a[0];
        const double wy = y - a[1];
        const double t = clamp01((wx * vx + wy * vy) * inv_len2);
        const double qx = a[0] + t * vx;
        const double qy = a[1] + t * vy;
        const double dx = x - qx;
        const double dy = y - qy;
        const double d2 = dx * dx + dy * dy;
        if (d2 < best_d2) {
            const double seg_len = std::sqrt(len2);
            const double tx = vx / seg_len;
            const double ty = vy / seg_len;
            const double nx = -ty;
            const double ny = tx;
            best_d2 = d2;
            best_s = g.station[i] + t * seg_len;
            best_n = dot2(dx, dy, nx, ny);
        }
    }

    if (!std::isfinite(best_d2)) {
        return false;
    }
    if (s_out) {
        *s_out = best_s;
    }
    if (n_out) {
        *n_out = best_n;
    }
    return true;
}

std::array<double, 2> unproject_sn(const ChannelGuide& g, double s, double n) {
    double cx = 0.0;
    double cy = 0.0;
    double tx = 1.0;
    double ty = 0.0;
    if (!sample_frame_at_station(g, s, &cx, &cy, &tx, &ty)) {
        return {cx + s, cy + n};
    }
    const double nx = -ty;
    const double ny = tx;
    return {cx + n * nx, cy + n * ny};
}

double mean_projected_n(const ChannelGuide& g, const std::vector<std::array<double, 2>>& pts) {
    if (pts.empty()) {
        return 0.0;
    }
    double acc = 0.0;
    int used = 0;
    for (const auto& p : pts) {
        double s = 0.0;
        double n = 0.0;
        if (project_point_to_centerline(g, p[0], p[1], &s, &n)) {
            acc += n;
            ++used;
        }
    }
    if (used <= 0) {
        return 0.0;
    }
    return acc / static_cast<double>(used);
}

double point_to_polyline_distance(double x, double y, const std::vector<std::array<double, 2>>& line) {
    if (line.size() < 2) {
        return std::numeric_limits<double>::infinity();
    }
    double best = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i + 1 < line.size(); ++i) {
        const auto& a = line[i];
        const auto& b = line[i + 1];
        const double vx = b[0] - a[0];
        const double vy = b[1] - a[1];
        const double len2 = vx * vx + vy * vy;
        if (len2 <= 1.0e-18) {
            continue;
        }
        const double t = clamp01(((x - a[0]) * vx + (y - a[1]) * vy) / len2);
        const double qx = a[0] + t * vx;
        const double qy = a[1] + t * vy;
        const double d = std::hypot(x - qx, y - qy);
        best = std::min(best, d);
    }
    return best;
}

bool closest_point_tangent_on_polyline(
    double x,
    double y,
    const std::vector<std::array<double, 2>>& line,
    double* out_dist,
    double* out_px,
    double* out_py,
    double* out_tx,
    double* out_ty) {
    if (line.size() < 2) {
        return false;
    }
    double best = std::numeric_limits<double>::infinity();
    double best_x = 0.0;
    double best_y = 0.0;
    double best_tx = 1.0;
    double best_ty = 0.0;
    bool found = false;

    for (size_t i = 0; i + 1 < line.size(); ++i) {
        const auto& a = line[i];
        const auto& b = line[i + 1];
        const double vx = b[0] - a[0];
        const double vy = b[1] - a[1];
        const double len2 = vx * vx + vy * vy;
        if (len2 <= 1.0e-18) {
            continue;
        }
        const double len = std::sqrt(len2);
        const double tx = vx / len;
        const double ty = vy / len;
        const double t = clamp01(((x - a[0]) * vx + (y - a[1]) * vy) / len2);
        const double qx = a[0] + t * vx;
        const double qy = a[1] + t * vy;
        const double d = std::hypot(x - qx, y - qy);
        if (d < best) {
            best = d;
            best_x = qx;
            best_y = qy;
            best_tx = tx;
            best_ty = ty;
            found = true;
        }
    }
    if (!found) {
        return false;
    }
    if (out_dist) {
        *out_dist = best;
    }
    if (out_px) {
        *out_px = best_x;
    }
    if (out_py) {
        *out_py = best_y;
    }
    if (out_tx) {
        *out_tx = best_tx;
    }
    if (out_ty) {
        *out_ty = best_ty;
    }
    return true;
}

inline bool tri_method_is(const std::string& tri_method, const char* name) {
    return to_lower(tri_method) == std::string(name);
}

inline double tri_twice_area(
    const std::array<double, 2>& a,
    const std::array<double, 2>& b,
    const std::array<double, 2>& c) {
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
}

inline bool is_valid_triangle(
    const std::array<double, 2>& a,
    const std::array<double, 2>& b,
    const std::array<double, 2>& c,
    double min_abs_twice_area = 1.0e-14) {
    return std::abs(tri_twice_area(a, b, c)) >= min_abs_twice_area;
}

bool closest_point_on_polyline(
    double x,
    double y,
    const std::vector<std::array<double, 2>>& line,
    double* out_dist,
    double* out_px,
    double* out_py) {
    if (line.size() < 2) {
        return false;
    }
    double best = std::numeric_limits<double>::infinity();
    double best_x = 0.0;
    double best_y = 0.0;
    bool found = false;

    for (size_t i = 0; i + 1 < line.size(); ++i) {
        const auto& a = line[i];
        const auto& b = line[i + 1];
        const double vx = b[0] - a[0];
        const double vy = b[1] - a[1];
        const double len2 = vx * vx + vy * vy;
        if (len2 <= 1.0e-18) {
            continue;
        }
        const double t = clamp01(((x - a[0]) * vx + (y - a[1]) * vy) / len2);
        const double qx = a[0] + t * vx;
        const double qy = a[1] + t * vy;
        const double d = std::hypot(x - qx, y - qy);
        if (d < best) {
            best = d;
            best_x = qx;
            best_y = qy;
            found = true;
        }
    }

    if (!found) {
        return false;
    }
    if (out_dist) {
        *out_dist = best;
    }
    if (out_px) {
        *out_px = best_x;
    }
    if (out_py) {
        *out_py = best_y;
    }
    return true;
}

bool closest_point_on_any_polyline(
    double x,
    double y,
    const std::vector<std::vector<std::array<double, 2>>>& lines,
    double* out_dist,
    double* out_px,
    double* out_py) {
    if (lines.empty()) {
        return false;
    }
    double best = std::numeric_limits<double>::infinity();
    double best_x = 0.0;
    double best_y = 0.0;
    bool found = false;
    for (const auto& line : lines) {
        double d = std::numeric_limits<double>::infinity();
        double px = 0.0;
        double py = 0.0;
        if (!closest_point_on_polyline(x, y, line, &d, &px, &py)) {
            continue;
        }
        if (d < best) {
            best = d;
            best_x = px;
            best_y = py;
            found = true;
        }
    }
    if (!found) {
        return false;
    }
    if (out_dist) {
        *out_dist = best;
    }
    if (out_px) {
        *out_px = best_x;
    }
    if (out_py) {
        *out_py = best_y;
    }
    return true;
}

inline bool point_in_region_with_holes(
    double x,
    double y,
    const std::vector<std::array<double, 2>>& ring,
    const std::vector<std::vector<std::array<double, 2>>>& holes) {
    if (!point_in_ring(x, y, ring)) {
        return false;
    }
    for (const auto& h : holes) {
        if (point_in_ring(x, y, h)) {
            return false;
        }
    }
    return true;
}

struct EdgeKey {
    int a = -1;
    int b = -1;

    bool operator==(const EdgeKey& other) const {
        return a == other.a && b == other.b;
    }
};

struct EdgeKeyHash {
    std::size_t operator()(const EdgeKey& e) const {
        std::size_t h1 = std::hash<int>{}(e.a);
        std::size_t h2 = std::hash<int>{}(e.b);
        return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6U) + (h1 >> 2U));
    }
};

struct QuantKey {
    std::int64_t xq = 0;
    std::int64_t yq = 0;

    bool operator==(const QuantKey& other) const {
        return xq == other.xq && yq == other.yq;
    }
};

struct QuantKeyHash {
    std::size_t operator()(const QuantKey& k) const {
        std::size_t h1 = std::hash<std::int64_t>{}(k.xq);
        std::size_t h2 = std::hash<std::int64_t>{}(k.yq);
        return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6U) + (h1 >> 2U));
    }
};

inline EdgeKey make_edge_key(int i, int j) {
    if (i < j) {
        return {i, j};
    }
    return {j, i};
}

void optimize_mesh_quality(
    std::vector<double>* node_x,
    std::vector<double>* node_y,
    const std::vector<int>& face_offsets,
    const std::vector<int>& face_nodes,
    const std::vector<std::vector<std::array<double, 2>>>& boundary_lines,
    double boundary_lock_tol,
    int sweeps,
    double relax,
    double max_move_factor) {
    if (node_x == nullptr || node_y == nullptr) {
        return;
    }
    std::vector<double>& x = *node_x;
    std::vector<double>& y = *node_y;
    const int n_nodes = static_cast<int>(x.size());
    if (n_nodes <= 0 || static_cast<int>(y.size()) != n_nodes) {
        return;
    }
    if (face_offsets.size() <= 1 || face_nodes.empty()) {
        return;
    }

    sweeps = std::max(0, sweeps);
    relax = std::min(1.0, std::max(0.0, relax));
    max_move_factor = std::min(1.0, std::max(0.01, max_move_factor));
    if (sweeps <= 0 || relax <= 0.0) {
        return;
    }

    std::vector<std::unordered_set<int>> nbr_set(static_cast<size_t>(n_nodes));
    std::unordered_map<EdgeKey, int, EdgeKeyHash> edge_count;
    edge_count.reserve(face_nodes.size());

    for (size_t f = 0; f + 1 < face_offsets.size(); ++f) {
        const int s = face_offsets[f];
        const int e = face_offsets[f + 1];
        if (e - s < 3) {
            continue;
        }
        for (int k = s; k < e; ++k) {
            const int i = face_nodes[k];
            const int j = face_nodes[(k + 1 < e) ? (k + 1) : s];
            if (i < 0 || j < 0 || i >= n_nodes || j >= n_nodes || i == j) {
                continue;
            }
            nbr_set[static_cast<size_t>(i)].insert(j);
            nbr_set[static_cast<size_t>(j)].insert(i);
            const EdgeKey ek = make_edge_key(i, j);
            auto it = edge_count.find(ek);
            if (it == edge_count.end()) {
                edge_count.emplace(ek, 1);
            } else {
                it->second += 1;
            }
        }
    }

    std::vector<char> locked(static_cast<size_t>(n_nodes), 0);
    for (const auto& kv : edge_count) {
        if (kv.second == 1) {
            locked[static_cast<size_t>(kv.first.a)] = 1;
            locked[static_cast<size_t>(kv.first.b)] = 1;
        }
    }

    const double lock_tol = std::max(0.0, boundary_lock_tol);
    if (!boundary_lines.empty() && lock_tol > 0.0) {
        for (int i = 0; i < n_nodes; ++i) {
            if (locked[static_cast<size_t>(i)]) {
                continue;
            }
            double d = std::numeric_limits<double>::infinity();
            if (!closest_point_on_any_polyline(x[i], y[i], boundary_lines, &d, nullptr, nullptr)) {
                continue;
            }
            if (d <= lock_tol) {
                locked[static_cast<size_t>(i)] = 1;
            }
        }
    }

    std::vector<double> new_x(static_cast<size_t>(n_nodes), 0.0);
    std::vector<double> new_y(static_cast<size_t>(n_nodes), 0.0);
    for (int sweep = 0; sweep < sweeps; ++sweep) {
        for (int i = 0; i < n_nodes; ++i) {
            if (locked[static_cast<size_t>(i)]) {
                new_x[static_cast<size_t>(i)] = x[i];
                new_y[static_cast<size_t>(i)] = y[i];
                continue;
            }

            const auto& nbrs = nbr_set[static_cast<size_t>(i)];
            if (nbrs.size() < 3) {
                new_x[static_cast<size_t>(i)] = x[i];
                new_y[static_cast<size_t>(i)] = y[i];
                continue;
            }

            double sx = 0.0;
            double sy = 0.0;
            double mean_len = 0.0;
            int used = 0;
            for (int j : nbrs) {
                sx += x[j];
                sy += y[j];
                mean_len += std::hypot(x[j] - x[i], y[j] - y[i]);
                ++used;
            }
            if (used <= 0) {
                new_x[static_cast<size_t>(i)] = x[i];
                new_y[static_cast<size_t>(i)] = y[i];
                continue;
            }
            const double tx = sx / static_cast<double>(used);
            const double ty = sy / static_cast<double>(used);
            const double max_move = max_move_factor * std::max(1.0e-9, mean_len / static_cast<double>(used));
            double dx = (tx - x[i]) * relax;
            double dy = (ty - y[i]) * relax;
            const double d = std::hypot(dx, dy);
            if (d > max_move) {
                const double s = max_move / std::max(1.0e-12, d);
                dx *= s;
                dy *= s;
            }
            new_x[static_cast<size_t>(i)] = x[i] + dx;
            new_y[static_cast<size_t>(i)] = y[i] + dy;
        }
        x.swap(new_x);
        y.swap(new_y);
    }
}

struct TriFace {
    int a = -1;
    int b = -1;
    int c = -1;
    int region_id = -1;
    double target_size = 1.0;
};

inline int tri_third_vertex(const TriFace& t, int u, int v) {
    if (t.a != u && t.a != v) {
        return t.a;
    }
    if (t.b != u && t.b != v) {
        return t.b;
    }
    if (t.c != u && t.c != v) {
        return t.c;
    }
    return -1;
}

inline bool segment_intersects_strict(
    const std::array<double, 2>& a,
    const std::array<double, 2>& b,
    const std::array<double, 2>& c,
    const std::array<double, 2>& d) {
    auto orient = [](const std::array<double, 2>& p, const std::array<double, 2>& q, const std::array<double, 2>& r) {
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]);
    };

    const double o1 = orient(a, b, c);
    const double o2 = orient(a, b, d);
    const double o3 = orient(c, d, a);
    const double o4 = orient(c, d, b);

    const double eps = 1.0e-14;
    const bool ab_straddle = ((o1 > eps && o2 < -eps) || (o1 < -eps && o2 > eps));
    const bool cd_straddle = ((o3 > eps && o4 < -eps) || (o3 < -eps && o4 > eps));
    return ab_straddle && cd_straddle;
}

inline void append_quad_as_tris(
    int n0,
    int n1,
    int n2,
    int n3,
    int region_id,
    double target_size,
    const std::vector<double>& node_x,
    const std::vector<double>& node_y,
    std::vector<TriFace>* tri_faces) {
    if (tri_faces == nullptr) {
        return;
    }
    if (n0 < 0 || n1 < 0 || n2 < 0 || n3 < 0) {
        return;
    }
    const int n_nodes = static_cast<int>(node_x.size());
    if (n0 >= n_nodes || n1 >= n_nodes || n2 >= n_nodes || n3 >= n_nodes) {
        return;
    }

    const std::array<double, 2> p0 = {node_x[static_cast<size_t>(n0)], node_y[static_cast<size_t>(n0)]};
    const std::array<double, 2> p1 = {node_x[static_cast<size_t>(n1)], node_y[static_cast<size_t>(n1)]};
    const std::array<double, 2> p2 = {node_x[static_cast<size_t>(n2)], node_y[static_cast<size_t>(n2)]};
    const std::array<double, 2> p3 = {node_x[static_cast<size_t>(n3)], node_y[static_cast<size_t>(n3)]};

    const double aA0 = std::abs(tri_twice_area(p0, p1, p2));
    const double aA1 = std::abs(tri_twice_area(p0, p2, p3));
    const double aB0 = std::abs(tri_twice_area(p0, p1, p3));
    const double aB1 = std::abs(tri_twice_area(p1, p2, p3));
    const bool split_a = std::min(aA0, aA1) >= std::min(aB0, aB1);

    if (split_a) {
        if (!is_valid_triangle(p0, p1, p2) || !is_valid_triangle(p0, p2, p3)) {
            return;
        }
        tri_faces->push_back(TriFace{n0, n1, n2, region_id, target_size});
        tri_faces->push_back(TriFace{n0, n2, n3, region_id, target_size});
    } else {
        if (!is_valid_triangle(p0, p1, p3) || !is_valid_triangle(p1, p2, p3)) {
            return;
        }
        tri_faces->push_back(TriFace{n0, n1, n3, region_id, target_size});
        tri_faces->push_back(TriFace{n1, n2, n3, region_id, target_size});
    }
}

std::unordered_map<EdgeKey, std::vector<int>, EdgeKeyHash> build_edge_to_triangles(const std::vector<TriFace>& tris) {
    std::unordered_map<EdgeKey, std::vector<int>, EdgeKeyHash> edge_to_tris;
    edge_to_tris.reserve(tris.size() * 2);
    for (int ti = 0; ti < static_cast<int>(tris.size()); ++ti) {
        const auto& t = tris[static_cast<size_t>(ti)];
        if (t.a < 0 || t.b < 0 || t.c < 0) {
            continue;
        }
        edge_to_tris[make_edge_key(t.a, t.b)].push_back(ti);
        edge_to_tris[make_edge_key(t.b, t.c)].push_back(ti);
        edge_to_tris[make_edge_key(t.c, t.a)].push_back(ti);
    }
    return edge_to_tris;
}

int find_nearest_node_for_point(
    double px,
    double py,
    const std::vector<double>& node_x,
    const std::vector<double>& node_y,
    double max_dist,
    std::unordered_map<QuantKey, int, QuantKeyHash>* quant_map,
    double quant_scale) {
    if (node_x.empty() || node_y.size() != node_x.size()) {
        return -1;
    }
    if (quant_map != nullptr) {
        const QuantKey k{static_cast<std::int64_t>(std::llround(px * quant_scale)), static_cast<std::int64_t>(std::llround(py * quant_scale))};
        auto it = quant_map->find(k);
        if (it != quant_map->end()) {
            return it->second;
        }
    }

    const double tol2 = max_dist * max_dist;
    int best = -1;
    double best_d2 = std::numeric_limits<double>::infinity();
    for (int i = 0; i < static_cast<int>(node_x.size()); ++i) {
        const double dx = node_x[static_cast<size_t>(i)] - px;
        const double dy = node_y[static_cast<size_t>(i)] - py;
        const double d2 = dx * dx + dy * dy;
        if (d2 < best_d2) {
            best_d2 = d2;
            best = i;
        }
    }
    if (best < 0 || best_d2 > tol2) {
        return -1;
    }
    return best;
}

std::vector<EdgeKey> collect_constrained_edges_from_boundaries(
    const std::vector<double>& node_x,
    const std::vector<double>& node_y,
    const std::vector<std::vector<std::array<double, 2>>>& boundary_lines,
    double snap_tol) {
    std::vector<EdgeKey> out;
    if (boundary_lines.empty() || node_x.empty()) {
        return out;
    }

    std::unordered_map<QuantKey, int, QuantKeyHash> qmap;
    qmap.reserve(node_x.size() * 2);
    const double quant_scale = 1.0e6;
    for (int i = 0; i < static_cast<int>(node_x.size()); ++i) {
        const QuantKey k{static_cast<std::int64_t>(std::llround(node_x[static_cast<size_t>(i)] * quant_scale)),
                         static_cast<std::int64_t>(std::llround(node_y[static_cast<size_t>(i)] * quant_scale))};
        qmap.emplace(k, i);
    }

    std::unordered_set<EdgeKey, EdgeKeyHash> uniq;
    const double step_len = std::max(1.0e-9, 0.75 * snap_tol);
    for (const auto& line : boundary_lines) {
        if (line.size() < 2) {
            continue;
        }
        for (size_t i = 0; i + 1 < line.size(); ++i) {
            const auto& p0 = line[i];
            const auto& p1 = line[i + 1];
            const int n0 = find_nearest_node_for_point(p0[0], p0[1], node_x, node_y, snap_tol, &qmap, quant_scale);
            const int n1 = find_nearest_node_for_point(p1[0], p1[1], node_x, node_y, snap_tol, &qmap, quant_scale);
            if (n0 < 0 || n1 < 0 || n0 == n1) {
                continue;
            }

            // Build a node chain along each constrained segment. This mirrors
            // conforming triangulation practice: split long constraints into
            // shorter recoverable pieces instead of recovering one long edge.
            int prev = n0;
            const double seg_len = std::hypot(p1[0] - p0[0], p1[1] - p0[1]);
            const int n_samples = std::max(0, static_cast<int>(std::floor(seg_len / step_len)) - 1);
            for (int s = 1; s <= n_samples; ++s) {
                const double t = static_cast<double>(s) / static_cast<double>(n_samples + 1);
                const double sx = (1.0 - t) * p0[0] + t * p1[0];
                const double sy = (1.0 - t) * p0[1] + t * p1[1];
                const int ns = find_nearest_node_for_point(sx, sy, node_x, node_y, snap_tol, &qmap, quant_scale);
                if (ns < 0 || ns == prev) {
                    continue;
                }
                uniq.insert(make_edge_key(prev, ns));
                prev = ns;
            }
            if (prev != n1) {
                uniq.insert(make_edge_key(prev, n1));
            }
        }
    }
    out.reserve(uniq.size());
    for (const auto& e : uniq) {
        out.push_back(e);
    }
    return out;
}

void recover_constrained_edges_by_flips(
    const std::vector<double>& node_x,
    const std::vector<double>& node_y,
    const std::vector<EdgeKey>& constrained_edges,
    std::vector<TriFace>* tri_faces,
    int max_flips_per_edge) {
    if (tri_faces == nullptr || tri_faces->empty() || constrained_edges.empty()) {
        return;
    }
    std::vector<TriFace>& tris = *tri_faces;
    max_flips_per_edge = std::max(1, max_flips_per_edge);

    for (const auto& ce : constrained_edges) {
        if (ce.a < 0 || ce.b < 0 || ce.a >= static_cast<int>(node_x.size()) || ce.b >= static_cast<int>(node_x.size()) || ce.a == ce.b) {
            continue;
        }
        int flips = 0;
        while (flips < max_flips_per_edge) {
            auto edge_to_tris = build_edge_to_triangles(tris);
            if (edge_to_tris.find(ce) != edge_to_tris.end()) {
                break;
            }

            const std::array<double, 2> a = {node_x[static_cast<size_t>(ce.a)], node_y[static_cast<size_t>(ce.a)]};
            const std::array<double, 2> b = {node_x[static_cast<size_t>(ce.b)], node_y[static_cast<size_t>(ce.b)]};

            struct FlipCandidate {
                EdgeKey e;
                int t0 = -1;
                int t1 = -1;
                double len2 = std::numeric_limits<double>::infinity();
            };
            std::vector<FlipCandidate> candidates;
            candidates.reserve(16);
            for (const auto& kv : edge_to_tris) {
                const EdgeKey e = kv.first;
                if (e.a == ce.a || e.a == ce.b || e.b == ce.a || e.b == ce.b) {
                    continue;
                }
                const auto& adj = kv.second;
                if (adj.size() != 2) {
                    continue;
                }
                const std::array<double, 2> c = {node_x[static_cast<size_t>(e.a)], node_y[static_cast<size_t>(e.a)]};
                const std::array<double, 2> d = {node_x[static_cast<size_t>(e.b)], node_y[static_cast<size_t>(e.b)]};
                if (!segment_intersects_strict(a, b, c, d)) {
                    continue;
                }
                const double ex = c[0] - d[0];
                const double ey = c[1] - d[1];
                const double len2 = ex * ex + ey * ey;
                candidates.push_back(FlipCandidate{e, adj[0], adj[1], len2});
            }

            if (candidates.empty()) {
                break;
            }
            std::sort(
                candidates.begin(),
                candidates.end(),
                [](const FlipCandidate& lhs, const FlipCandidate& rhs) {
                    return lhs.len2 < rhs.len2;
                });

            bool flipped = false;
            for (const auto& cand : candidates) {
                const EdgeKey cut_edge = cand.e;
                const int t0 = cand.t0;
                const int t1 = cand.t1;
                if (cut_edge.a < 0 || t0 < 0 || t1 < 0
                    || t0 >= static_cast<int>(tris.size())
                    || t1 >= static_cast<int>(tris.size())) {
                    continue;
                }

                const TriFace old0 = tris[static_cast<size_t>(t0)];
                const TriFace old1 = tris[static_cast<size_t>(t1)];
                const int p = tri_third_vertex(old0, cut_edge.a, cut_edge.b);
                const int q = tri_third_vertex(old1, cut_edge.a, cut_edge.b);
                if (p < 0 || q < 0 || p == q) {
                    continue;
                }

                const std::array<double, 2> pa = {node_x[static_cast<size_t>(p)], node_y[static_cast<size_t>(p)]};
                const std::array<double, 2> qa = {node_x[static_cast<size_t>(q)], node_y[static_cast<size_t>(q)]};
                const std::array<double, 2> ea = {node_x[static_cast<size_t>(cut_edge.a)], node_y[static_cast<size_t>(cut_edge.a)]};
                const std::array<double, 2> eb = {node_x[static_cast<size_t>(cut_edge.b)], node_y[static_cast<size_t>(cut_edge.b)]};
                if (!is_valid_triangle(pa, qa, ea) || !is_valid_triangle(pa, qa, eb)) {
                    continue;
                }

                TriFace nt0;
                nt0.a = p;
                nt0.b = q;
                nt0.c = cut_edge.a;
                nt0.region_id = old0.region_id;
                nt0.target_size = old0.target_size;

                TriFace nt1;
                nt1.a = q;
                nt1.b = p;
                nt1.c = cut_edge.b;
                nt1.region_id = old1.region_id;
                nt1.target_size = old1.target_size;

                tris[static_cast<size_t>(t0)] = nt0;
                tris[static_cast<size_t>(t1)] = nt1;
                ++flips;
                flipped = true;
                break;
            }

            if (!flipped) {
                break;
            }
        }
    }
}

std::vector<std::array<double, 2>> build_polyline_anchors(
    const std::vector<std::array<double, 2>>& line,
    double spacing) {
    std::vector<std::array<double, 2>> anchors;
    if (line.size() < 2) {
        return anchors;
    }

    const double ds = std::max(1.0e-9, spacing);
    anchors.push_back(line.front());
    double remaining = ds;

    for (size_t i = 0; i + 1 < line.size(); ++i) {
        const auto& a = line[i];
        const auto& b = line[i + 1];
        const double vx = b[0] - a[0];
        const double vy = b[1] - a[1];
        const double seg_len = std::hypot(vx, vy);
        if (seg_len <= 1.0e-12) {
            continue;
        }
        const double tx = vx / seg_len;
        const double ty = vy / seg_len;
        double walked = 0.0;
        while (walked + remaining <= seg_len + 1.0e-12) {
            walked += remaining;
            anchors.push_back({a[0] + walked * tx, a[1] + walked * ty});
            remaining = ds;
        }
        remaining = std::max(1.0e-9, remaining - (seg_len - walked));
    }

    const auto& last = line.back();
    if (anchors.empty() || std::hypot(anchors.back()[0] - last[0], anchors.back()[1] - last[1]) > 1.0e-9) {
        anchors.push_back(last);
    }
    return anchors;
}

void snap_point_to_bank_anchor(
    double* x,
    double* y,
    const std::vector<std::vector<std::array<double, 2>>>& bank_lines,
    const std::vector<std::vector<std::array<double, 2>>>& bank_anchors,
    double snap_tol) {
    if (x == nullptr || y == nullptr || bank_lines.empty() || bank_anchors.empty()) {
        return;
    }
    const double tol = std::max(0.0, snap_tol);
    if (tol <= 0.0) {
        return;
    }

    int best_line = -1;
    double best_dist = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < bank_lines.size(); ++i) {
        double d = std::numeric_limits<double>::infinity();
        if (!closest_point_on_polyline(*x, *y, bank_lines[i], &d, nullptr, nullptr)) {
            continue;
        }
        if (d < best_dist) {
            best_dist = d;
            best_line = static_cast<int>(i);
        }
    }

    if (best_line < 0 || best_dist > tol || best_line >= static_cast<int>(bank_anchors.size())) {
        return;
    }

    const auto& anchors = bank_anchors[static_cast<size_t>(best_line)];
    if (anchors.empty()) {
        return;
    }

    double best_anchor_d2 = std::numeric_limits<double>::infinity();
    std::array<double, 2> best_anchor = anchors.front();
    for (const auto& p : anchors) {
        const double dx = *x - p[0];
        const double dy = *y - p[1];
        const double d2 = dx * dx + dy * dy;
        if (d2 < best_anchor_d2) {
            best_anchor_d2 = d2;
            best_anchor = p;
        }
    }

    *x = best_anchor[0];
    *y = best_anchor[1];
}

double env_double(const char* name, double default_value) {
    if (name == nullptr) {
        return default_value;
    }
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') {
        return default_value;
    }
    try {
        return std::stod(std::string(raw));
    } catch (...) {
        return default_value;
    }
}

struct Key {
    std::int64_t xq = 0;
    std::int64_t yq = 0;

    bool operator==(const Key& other) const {
        return xq == other.xq && yq == other.yq;
    }
};

struct KeyHash {
    std::size_t operator()(const Key& k) const {
        std::size_t h1 = std::hash<std::int64_t>{}(k.xq);
        std::size_t h2 = std::hash<std::int64_t>{}(k.yq);
        return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6U) + (h1 >> 2U));
    }
};

class NodeRegistry {
public:
    explicit NodeRegistry(double quant_scale) : quant_scale_(std::max(1.0, quant_scale)) {}

    int get_or_create(double x, double y) {
        Key k{static_cast<std::int64_t>(std::llround(x * quant_scale_)),
              static_cast<std::int64_t>(std::llround(y * quant_scale_))};
        auto it = map_.find(k);
        if (it != map_.end()) {
            return it->second;
        }
        const int idx = static_cast<int>(x_.size());
        x_.push_back(x);
        y_.push_back(y);
        map_.emplace(k, idx);
        return idx;
    }

    const std::vector<double>& x() const { return x_; }
    const std::vector<double>& y() const { return y_; }

private:
    double quant_scale_;
    std::unordered_map<Key, int, KeyHash> map_;
    std::vector<double> x_;
    std::vector<double> y_;
};

}  // namespace

py::dict generate_hybrid_mesh(
    const std::vector<std::vector<std::array<double, 2>>>& region_rings,
    const std::vector<std::vector<std::vector<std::array<double, 2>>>>& region_holes,
    const std::vector<double>& region_target_sizes,
    const std::vector<std::string>& region_cell_types,
    const std::vector<int>& region_ids,
    const std::vector<std::vector<std::array<double, 2>>>& constraint_rings,
    const std::vector<double>& constraint_target_sizes,
    const std::vector<std::string>& constraint_cell_types,
    const std::vector<int>& arc_region_ids,
    const std::vector<std::string>& arc_roles,
    const std::vector<std::vector<std::array<double, 2>>>& arc_lines,
    const std::string& tri_meshing_method,
    double transition_width_factor,
    double transition_outer_factor,
    double overbank_grading_factor,
    double constrained_edge_snap_tol,
    int constrained_edge_max_flips,
    double region_conformance_band_factor,
    double arc_conformance_band_factor,
    bool strict_conformance_mode) {

    const size_t n_regions = region_rings.size();
    if (region_target_sizes.size() != n_regions || region_cell_types.size() != n_regions || region_ids.size() != n_regions) {
        throw std::invalid_argument("Region arrays must have matching lengths");
    }

    const size_t n_constraints = constraint_rings.size();
    if (constraint_target_sizes.size() != n_constraints || constraint_cell_types.size() != n_constraints) {
        throw std::invalid_argument("Constraint arrays must have matching lengths");
    }
    if (arc_region_ids.size() != arc_roles.size() || arc_region_ids.size() != arc_lines.size()) {
        throw std::invalid_argument("Arc arrays must have matching lengths");
    }

    std::vector<Constraint> constraints;
    constraints.reserve(n_constraints);
    for (size_t i = 0; i < n_constraints; ++i) {
        if (constraint_rings[i].size() < 3) {
            continue;
        }
        constraints.push_back(Constraint{constraint_rings[i], std::max(1.0e-9, constraint_target_sizes[i]), constraint_cell_types[i]});
    }

    std::unordered_map<int, std::vector<std::array<double, 2>>> centerlines;
    std::unordered_map<int, std::vector<std::array<double, 2>>> left_banks;
    std::unordered_map<int, std::vector<std::array<double, 2>>> right_banks;
    std::vector<std::vector<std::array<double, 2>>> all_arc_lines;
    std::vector<std::vector<std::array<double, 2>>> all_bank_lines;
    for (size_t i = 0; i < arc_region_ids.size(); ++i) {
        if (arc_lines[i].size() < 2) {
            continue;
        }
        all_arc_lines.push_back(arc_lines[i]);
        const int rid = arc_region_ids[i];
        const std::string role = to_lower(arc_roles[i]);
        if (role == "centerline") {
            centerlines[rid] = arc_lines[i];
        } else if (role == "left_bank") {
            left_banks[rid] = arc_lines[i];
            all_bank_lines.push_back(arc_lines[i]);
        } else if (role == "right_bank") {
            right_banks[rid] = arc_lines[i];
            all_bank_lines.push_back(arc_lines[i]);
        }
    }

    const std::string tri_method = to_lower(tri_meshing_method);
    const bool strict_conformance = env_double(
        "HYDRA_HYBRIDCPP_STRICT_CONFORMANCE",
        strict_conformance_mode ? 1.0 : 0.0) >= 0.5;
    const double region_conformance_band = std::max(
        0.0,
        env_double("HYDRA_HYBRIDCPP_REGION_CONFORMANCE_BAND_FACTOR", region_conformance_band_factor));
    const double arc_conformance_band = std::max(
        0.0,
        env_double("HYDRA_HYBRIDCPP_ARC_CONFORMANCE_BAND_FACTOR", arc_conformance_band_factor));

    double min_target_size = std::numeric_limits<double>::infinity();
    for (double s : region_target_sizes) {
        min_target_size = std::min(min_target_size, std::max(1.0e-9, s));
    }
    if (!std::isfinite(min_target_size)) {
        min_target_size = 1.0;
    }
    const double bank_anchor_spacing = std::max(1.0e-9, env_double("HYDRA_HYBRIDCPP_BANK_ANCHOR_SPACING", min_target_size));
    std::vector<std::vector<std::array<double, 2>>> all_bank_anchors;
    all_bank_anchors.reserve(all_bank_lines.size());
    for (const auto& line : all_bank_lines) {
        all_bank_anchors.push_back(build_polyline_anchors(line, bank_anchor_spacing));
    }

    std::vector<std::vector<std::array<double, 2>>> all_region_boundary_lines;
    all_region_boundary_lines.reserve(n_regions * 2);
    for (size_t r = 0; r < n_regions; ++r) {
        if (region_rings[r].size() >= 3) {
            std::vector<std::array<double, 2>> ln = region_rings[r];
            ln.push_back(region_rings[r].front());
            all_region_boundary_lines.push_back(std::move(ln));
        }
        if (r < region_holes.size()) {
            for (const auto& h : region_holes[r]) {
                if (h.size() < 3) {
                    continue;
                }
                std::vector<std::array<double, 2>> ln = h;
                ln.push_back(h.front());
                all_region_boundary_lines.push_back(std::move(ln));
            }
        }
    }

    const double region_anchor_spacing = std::max(1.0e-9, env_double("HYDRA_HYBRIDCPP_REGION_ANCHOR_SPACING", min_target_size));
    std::vector<std::vector<std::array<double, 2>>> all_region_boundary_anchors;
    all_region_boundary_anchors.reserve(all_region_boundary_lines.size());
    for (const auto& line : all_region_boundary_lines) {
        all_region_boundary_anchors.push_back(build_polyline_anchors(line, region_anchor_spacing));
    }

    const double arc_anchor_spacing = std::max(1.0e-9, env_double("HYDRA_HYBRIDCPP_ARC_ANCHOR_SPACING", 0.8 * min_target_size));
    std::vector<std::vector<std::array<double, 2>>> all_arc_anchors;
    all_arc_anchors.reserve(all_arc_lines.size());
    for (const auto& line : all_arc_lines) {
        all_arc_anchors.push_back(build_polyline_anchors(line, arc_anchor_spacing));
    }

    std::unordered_map<int, ChannelGuide> guides;
    guides.reserve(n_regions);
    for (size_t r = 0; r < n_regions; ++r) {
        const std::string region_type = to_lower(region_cell_types[r]);
        if (region_type != "channel_generator") {
            continue;
        }
        const int rid = region_ids[r];
        ChannelGuide g;

        const auto c_it = centerlines.find(rid);
        const auto l_it = left_banks.find(rid);
        const auto r_it = right_banks.find(rid);

        bool ok_centerline = false;
        if (c_it != centerlines.end()) {
            ok_centerline = build_centerline(&g, c_it->second);
        } else if (l_it != left_banks.end()) {
            ok_centerline = build_centerline(&g, l_it->second);
        } else if (r_it != right_banks.end()) {
            ok_centerline = build_centerline(&g, r_it->second);
        }

        if (!ok_centerline) {
            continue;
        }

        if (l_it != left_banks.end()) {
            g.left_bank = l_it->second;
        }
        if (r_it != right_banks.end()) {
            g.right_bank = r_it->second;
        }

        if (l_it != left_banks.end() && r_it != right_banks.end()) {
            double ln = mean_projected_n(g, l_it->second);
            double rn = mean_projected_n(g, r_it->second);
            if (ln < rn) {
                std::swap(ln, rn);
            }
            g.left_n = ln;
            g.right_n = rn;
            g.has_bank_bounds = (std::abs(g.left_n - g.right_n) > 1.0e-9);
        }
        guides[rid] = g;
    }

    NodeRegistry registry(1.0e6);
    std::vector<int> face_offsets{0};
    std::vector<int> face_nodes;
    std::vector<int> cell_nodes;
    std::vector<std::string> cell_types;
    std::vector<int> out_region_ids;
    std::vector<double> out_target_sizes;

    for (size_t r = 0; r < n_regions; ++r) {
        const auto& ring = region_rings[r];
        if (ring.size() < 3) {
            continue;
        }
        const std::string region_type = to_lower(region_cell_types[r]);
        if (is_empty(region_type)) {
            continue;
        }

        std::vector<std::vector<std::array<double, 2>>> holes;
        if (r < region_holes.size()) {
            holes = region_holes[r];
        }

        std::vector<std::vector<std::array<double, 2>>> local_boundary_lines;
        if (ring.size() >= 3) {
            std::vector<std::array<double, 2>> outer = ring;
            outer.push_back(ring.front());
            local_boundary_lines.push_back(std::move(outer));
        }
        for (const auto& h : holes) {
            if (h.size() < 3) {
                continue;
            }
            std::vector<std::array<double, 2>> hh = h;
            hh.push_back(h.front());
            local_boundary_lines.push_back(std::move(hh));
        }

        const double base_size = std::max(1.0e-9, region_target_sizes[r]);
        const bool is_channel = region_type == "channel_generator";
        const auto g_it = guides.find(region_ids[r]);
        const bool use_aligned_frame = is_channel && (g_it != guides.end()) && g_it->second.has_frame;

        double ax0 = 0.0;
        double ay0 = 0.0;
        double ax1 = 0.0;
        double ay1 = 0.0;

        if (use_aligned_frame) {
            const ChannelGuide& g = g_it->second;
            double s0 = 0.0;
            double n0 = 0.0;
            if (!project_point_to_centerline(g, ring[0][0], ring[0][1], &s0, &n0)) {
                const Bounds b = ring_bounds(ring);
                ax0 = b.xmin;
                ay0 = b.ymin;
                ax1 = b.xmax;
                ay1 = b.ymax;
            } else {
                ax0 = ax1 = s0;
                ay0 = ay1 = n0;
            }
            for (const auto& p : ring) {
                double sp = 0.0;
                double np = 0.0;
                if (!project_point_to_centerline(g, p[0], p[1], &sp, &np)) {
                    continue;
                }
                ax0 = std::min(ax0, sp);
                ay0 = std::min(ay0, np);
                ax1 = std::max(ax1, sp);
                ay1 = std::max(ay1, np);
            }
            if (g.has_bank_bounds) {
                ay0 = std::min(ay0, g.right_n);
                ay1 = std::max(ay1, g.left_n);
            }
        } else {
            const Bounds b = ring_bounds(ring);
            ax0 = b.xmin;
            ay0 = b.ymin;
            ax1 = b.xmax;
            ay1 = b.ymax;
        }

        const int nx = std::max(1, static_cast<int>(std::ceil((ax1 - ax0) / base_size)));
        const int ny = std::max(1, static_cast<int>(std::ceil((ay1 - ay0) / base_size)));
        const double dx = std::max(1.0e-9, (ax1 - ax0) / static_cast<double>(nx));
        const double dy = std::max(1.0e-9, (ay1 - ay0) / static_cast<double>(ny));

        const double trn_factor_env = env_double("HYDRA_HYBRIDCPP_TRANSITION_WIDTH_FACTOR", transition_width_factor);
        const double trn_factor = std::max(0.0, trn_factor_env);
        const double trn_outer_env = env_double("HYDRA_HYBRIDCPP_TRANSITION_OUTER_FACTOR", transition_outer_factor);
        const double trn_outer_factor = std::max(trn_factor + 1.0e-6, trn_outer_env);
        const double overbank_factor_env = env_double("HYDRA_HYBRIDCPP_OVERBANK_GRADING_FACTOR", overbank_grading_factor);
        const double overbank_factor = std::max(0.0, overbank_factor_env);
        const double snap_factor = std::max(0.0, env_double("HYDRA_HYBRIDCPP_BOUNDARY_SNAP_FACTOR", 0.55));

        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const double u0 = ax0 + static_cast<double>(i) * dx;
                const double v0 = ay0 + static_cast<double>(j) * dy;
                const double u1 = u0 + dx;
                const double v1 = v0 + dy;
                const double uc = u0 + 0.5 * dx;
                const double vc = v0 + 0.5 * dy;

                std::array<double, 2> p00;
                std::array<double, 2> p10;
                std::array<double, 2> p01;
                std::array<double, 2> p11;
                std::array<double, 2> pc;

                if (use_aligned_frame) {
                    const ChannelGuide& g = g_it->second;
                    p00 = unproject_sn(g, u0, v0);
                    p10 = unproject_sn(g, u1, v0);
                    p01 = unproject_sn(g, u0, v1);
                    p11 = unproject_sn(g, u1, v1);
                    pc = unproject_sn(g, uc, vc);
                } else {
                    p00 = {u0, v0};
                    p10 = {u1, v0};
                    p01 = {u0, v1};
                    p11 = {u1, v1};
                    pc = {uc, vc};
                }

                const double cx = pc[0];
                const double cy = pc[1];

                const bool centroid_valid = point_in_region_with_holes(cx, cy, ring, holes);
                std::array<std::array<double, 2>, 4> corners = {p00, p10, p11, p01};
                std::array<char, 4> corner_valid = {0, 0, 0, 0};
                int n_corner_valid = 0;
                for (int q = 0; q < 4; ++q) {
                    const bool ok = point_in_region_with_holes(corners[static_cast<size_t>(q)][0], corners[static_cast<size_t>(q)][1], ring, holes);
                    corner_valid[static_cast<size_t>(q)] = ok ? 1 : 0;
                    n_corner_valid += ok ? 1 : 0;
                }

                if (!centroid_valid && n_corner_valid <= 0) {
                    continue;
                }

                // Fill boundary strips: for partially overlapping cells, project outside
                // vertices to nearest local boundary to avoid boundary holes/gaps.
                if (!centroid_valid && n_corner_valid > 0 && !local_boundary_lines.empty()) {
                    for (int q = 0; q < 4; ++q) {
                        if (corner_valid[static_cast<size_t>(q)]) {
                            continue;
                        }
                        double d = std::numeric_limits<double>::infinity();
                        double px = corners[static_cast<size_t>(q)][0];
                        double py = corners[static_cast<size_t>(q)][1];
                        if (closest_point_on_any_polyline(px, py, local_boundary_lines, &d, &px, &py)) {
                            corners[static_cast<size_t>(q)][0] = px;
                            corners[static_cast<size_t>(q)][1] = py;
                        }
                    }
                    p00 = corners[0];
                    p10 = corners[1];
                    p11 = corners[2];
                    p01 = corners[3];
                }

                std::string local_type = region_type;
                double local_size = base_size;

                for (const auto& cst : constraints) {
                    if (!point_in_ring(cx, cy, cst.ring)) {
                        continue;
                    }
                    if (is_empty(cst.cell_type)) {
                        local_type = "empty";
                        break;
                    }
                    local_type = to_lower(cst.cell_type);
                    local_size = std::max(1.0e-9, cst.target_size);
                }

                if (is_empty(local_type)) {
                    continue;
                }

                double nearest_arc_dist = std::numeric_limits<double>::infinity();
                double nearest_arc_tx = 1.0;
                double nearest_arc_ty = 0.0;
                bool has_nearest_arc = false;
                if (!all_arc_lines.empty()) {
                    for (const auto& arc_line : all_arc_lines) {
                        double d = std::numeric_limits<double>::infinity();
                        double tx = 1.0;
                        double ty = 0.0;
                        if (!closest_point_tangent_on_polyline(cx, cy, arc_line, &d, nullptr, nullptr, &tx, &ty)) {
                            continue;
                        }
                        if (d < nearest_arc_dist) {
                            nearest_arc_dist = d;
                            nearest_arc_tx = tx;
                            nearest_arc_ty = ty;
                            has_nearest_arc = true;
                        }
                    }
                }

                const bool is_local_tri = to_lower(local_type) == "triangular";
                if (is_local_tri && has_nearest_arc) {
                    const double arc_refine_band = std::max(1.0e-9, env_double("HYDRA_HYBRIDCPP_TRI_ARC_REFINE_BAND", 2.5 * base_size));
                    if (nearest_arc_dist <= arc_refine_band) {
                        const double t = clamp01(nearest_arc_dist / arc_refine_band);
                        const double scale = 0.45 + 0.55 * t;
                        local_size = std::max(1.0e-9, std::min(local_size, base_size * scale));
                    }
                }

                // Enforce triangular strips near geometric constraints so face edges
                // can conform to region/arcs after constrained-edge recovery.
                if (is_quad_like(local_type)) {
                    bool near_region_boundary = false;
                    if (region_conformance_band > 0.0 && !local_boundary_lines.empty()) {
                        const double region_band = region_conformance_band * base_size;
                        if (region_band > 0.0) {
                            double dloc = std::numeric_limits<double>::infinity();
                            if (closest_point_on_any_polyline(cx, cy, local_boundary_lines, &dloc, nullptr, nullptr) && dloc <= region_band) {
                                near_region_boundary = true;
                            }
                        }
                    }

                    bool near_any_arc = false;
                    if (arc_conformance_band > 0.0 && has_nearest_arc) {
                        const double arc_band = arc_conformance_band * base_size;
                        near_any_arc = (arc_band > 0.0 && nearest_arc_dist <= arc_band);
                    }

                    if (near_region_boundary || near_any_arc) {
                        local_type = "triangular";
                        local_size = std::max(1.0e-9, std::min(local_size, base_size * 0.72));
                    }
                }

                if (use_aligned_frame && is_quad_like(local_type) && trn_factor > 0.0) {
                    const ChannelGuide& g = g_it->second;
                    const double inner_band = trn_factor * base_size;
                    const double outer_band = trn_outer_factor * base_size;
                    double dist_bank = std::numeric_limits<double>::infinity();
                    if (g.has_bank_bounds) {
                        const double dn_left = std::abs(vc - g.left_n);
                        const double dn_right = std::abs(vc - g.right_n);
                        dist_bank = std::min(dn_left, dn_right);
                    } else {
                        const double dn_edge = std::min(std::abs(vc - ay0), std::abs(ay1 - vc));
                        dist_bank = dn_edge;
                    }

                    if (dist_bank <= inner_band) {
                        local_type = "triangular";
                        local_size = std::max(1.0e-9, base_size * 0.58);
                    } else if (dist_bank <= outer_band) {
                        const double tau = clamp01((dist_bank - inner_band) / std::max(1.0e-9, outer_band - inner_band));
                        local_size = std::max(1.0e-9, base_size * (0.62 + 0.33 * tau));

                        const int h = (i * 73856093) ^ (j * 19349663) ^ (region_ids[r] * 83492791);
                        const double hh = static_cast<double>(h & 1023) / 1023.0;
                        const double tri_weight = std::max(0.0, 0.85 * (1.0 - tau));
                        if (hh < tri_weight) {
                            local_type = "triangular";
                        }
                    }
                }

                if (!is_channel && to_lower(local_type) == "triangular" && overbank_factor > 0.0 && !all_bank_lines.empty()) {
                    double dmin = std::numeric_limits<double>::infinity();
                    for (const auto& bank : all_bank_lines) {
                        dmin = std::min(dmin, point_to_polyline_distance(cx, cy, bank));
                    }
                    if (std::isfinite(dmin)) {
                        const double grade_band = overbank_factor * base_size;
                        if (grade_band > 0.0 && dmin <= grade_band) {
                            const double t = clamp01(dmin / grade_band);
                            const double scale = 0.72 + 0.58 * t;
                            local_size = std::max(1.0e-9, std::min(local_size, base_size * scale));
                        }
                    }
                }

                if (!all_bank_lines.empty() && snap_factor > 0.0) {
                    const double snap_tol = snap_factor * std::max(base_size, local_size);
                    snap_point_to_bank_anchor(&p00[0], &p00[1], all_bank_lines, all_bank_anchors, snap_tol);
                    snap_point_to_bank_anchor(&p10[0], &p10[1], all_bank_lines, all_bank_anchors, snap_tol);
                    snap_point_to_bank_anchor(&p01[0], &p01[1], all_bank_lines, all_bank_anchors, snap_tol);
                    snap_point_to_bank_anchor(&p11[0], &p11[1], all_bank_lines, all_bank_anchors, snap_tol);
                }

                if (is_local_tri && !all_arc_lines.empty()) {
                    const double tri_arc_snap_factor = std::max(0.0, env_double("HYDRA_HYBRIDCPP_TRI_ARC_SNAP_FACTOR", 0.35));
                    if (tri_arc_snap_factor > 0.0) {
                        const double tri_arc_snap_tol = tri_arc_snap_factor * std::max(base_size, local_size);
                        snap_point_to_bank_anchor(&p00[0], &p00[1], all_arc_lines, all_arc_anchors, tri_arc_snap_tol);
                        snap_point_to_bank_anchor(&p10[0], &p10[1], all_arc_lines, all_arc_anchors, tri_arc_snap_tol);
                        snap_point_to_bank_anchor(&p01[0], &p01[1], all_arc_lines, all_arc_anchors, tri_arc_snap_tol);
                        snap_point_to_bank_anchor(&p11[0], &p11[1], all_arc_lines, all_arc_anchors, tri_arc_snap_tol);
                    }
                }

                if (!all_region_boundary_lines.empty()) {
                    const double rb_snap_factor = std::max(0.0, env_double("HYDRA_HYBRIDCPP_REGION_BOUNDARY_SNAP_FACTOR", 0.40));
                    if (rb_snap_factor > 0.0) {
                        const double rb_snap_tol = rb_snap_factor * std::max(base_size, local_size);
                        snap_point_to_bank_anchor(&p00[0], &p00[1], all_region_boundary_lines, all_region_boundary_anchors, rb_snap_tol);
                        snap_point_to_bank_anchor(&p10[0], &p10[1], all_region_boundary_lines, all_region_boundary_anchors, rb_snap_tol);
                        snap_point_to_bank_anchor(&p01[0], &p01[1], all_region_boundary_lines, all_region_boundary_anchors, rb_snap_tol);
                        snap_point_to_bank_anchor(&p11[0], &p11[1], all_region_boundary_lines, all_region_boundary_anchors, rb_snap_tol);
                    }
                }

                const int n00 = registry.get_or_create(p00[0], p00[1]);
                const int n10 = registry.get_or_create(p10[0], p10[1]);
                const int n01 = registry.get_or_create(p01[0], p01[1]);
                const int n11 = registry.get_or_create(p11[0], p11[1]);

                if (is_quad_like(local_type)) {
                    if (!is_valid_triangle(p00, p10, p11) || !is_valid_triangle(p00, p11, p01)) {
                        continue;
                    }

                    face_nodes.push_back(n00);
                    face_nodes.push_back(n10);
                    face_nodes.push_back(n11);
                    face_nodes.push_back(n01);
                    face_offsets.push_back(static_cast<int>(face_nodes.size()));

                    cell_nodes.push_back(n00);
                    cell_nodes.push_back(n10);
                    cell_nodes.push_back(n11);
                    cell_nodes.push_back(n00);
                    cell_nodes.push_back(n11);
                    cell_nodes.push_back(n01);

                    cell_types.push_back(local_type);
                    out_region_ids.push_back(region_ids[r]);
                    out_target_sizes.push_back(local_size);
                } else {
                    bool split_a = ((i + j) % 2) == 0;
                    if (tri_method_is(tri_method, "direct_curvilinear_generation") && has_nearest_arc) {
                        const double d0x = p11[0] - p00[0];
                        const double d0y = p11[1] - p00[1];
                        const double d1x = p01[0] - p10[0];
                        const double d1y = p01[1] - p10[1];
                        const double a0 = std::abs(dot2(d0x, d0y, nearest_arc_tx, nearest_arc_ty));
                        const double a1 = std::abs(dot2(d1x, d1y, nearest_arc_tx, nearest_arc_ty));
                        split_a = (a0 >= a1);
                    } else if (tri_method_is(tri_method, "frontal_delaunay")) {
                        const double front_bias = std::abs((p10[0] - p00[0]) * (p01[1] - p00[1]) - (p10[1] - p00[1]) * (p01[0] - p00[0]));
                        split_a = (front_bias >= 1.0e-15) ? split_a : !split_a;
                    } else if (tri_method_is(tri_method, "advancing_front")) {
                        // Prefer the split that maximizes the minimum triangle area
                        // while aligning with nearest feature tangent when available.
                        const double aA0 = std::abs(tri_twice_area(p00, p10, p11));
                        const double aA1 = std::abs(tri_twice_area(p00, p11, p01));
                        const double aB0 = std::abs(tri_twice_area(p00, p10, p01));
                        const double aB1 = std::abs(tri_twice_area(p10, p11, p01));
                        const double minA = std::min(aA0, aA1);
                        const double minB = std::min(aB0, aB1);
                        split_a = (minA >= minB);

                        if (has_nearest_arc) {
                            const double d0x = p11[0] - p00[0];
                            const double d0y = p11[1] - p00[1];
                            const double d1x = p01[0] - p10[0];
                            const double d1y = p01[1] - p10[1];
                            const double a0 = std::abs(dot2(d0x, d0y, nearest_arc_tx, nearest_arc_ty));
                            const double a1 = std::abs(dot2(d1x, d1y, nearest_arc_tx, nearest_arc_ty));
                            if (std::abs(a0 - a1) > 1.0e-12) {
                                split_a = (a0 >= a1);
                            }
                        }
                    }

                    bool use_aniso_refine = false;
                    if (tri_method_is(tri_method, "anisotropic_delaunay_refinement") && has_nearest_arc) {
                        const double anis_band = std::max(1.0e-9, env_double("HYDRA_HYBRIDCPP_ANISO_ARC_BAND", 3.0 * base_size));
                        const double anis_ratio = 1.0 + std::max(0.0, (anis_band - nearest_arc_dist) / anis_band) * 2.0;
                        use_aniso_refine = anis_ratio >= 1.5;
                    }

                    if (use_aniso_refine) {
                        const double cxv = 0.25 * (p00[0] + p10[0] + p11[0] + p01[0]);
                        const double cyv = 0.25 * (p00[1] + p10[1] + p11[1] + p01[1]);
                        const double nx = -nearest_arc_ty;
                        const double ny = nearest_arc_tx;
                        const double offset_mag = 0.10 * std::min(dx, dy);
                        const int nc = registry.get_or_create(cxv + offset_mag * nx, cyv + offset_mag * ny);

                        const std::array<std::array<int, 3>, 4> tris = {{{n00, n10, nc}, {n10, n11, nc}, {n11, n01, nc}, {n01, n00, nc}}};
                        for (const auto& tri : tris) {
                            const std::array<double, 2> pa = {registry.x()[tri[0]], registry.y()[tri[0]]};
                            const std::array<double, 2> pb = {registry.x()[tri[1]], registry.y()[tri[1]]};
                            const std::array<double, 2> pcv = {registry.x()[tri[2]], registry.y()[tri[2]]};
                            if (!is_valid_triangle(pa, pb, pcv)) {
                                continue;
                            }
                            face_nodes.push_back(tri[0]);
                            face_nodes.push_back(tri[1]);
                            face_nodes.push_back(tri[2]);
                            face_offsets.push_back(static_cast<int>(face_nodes.size()));

                            cell_nodes.push_back(tri[0]);
                            cell_nodes.push_back(tri[1]);
                            cell_nodes.push_back(tri[2]);

                            cell_types.push_back("triangular");
                            out_region_ids.push_back(region_ids[r]);
                            out_target_sizes.push_back(std::max(1.0e-9, local_size * 0.8));
                        }
                    } else if (split_a) {
                        if (!is_valid_triangle(p00, p10, p11) || !is_valid_triangle(p00, p11, p01)) {
                            continue;
                        }
                        face_nodes.push_back(n00);
                        face_nodes.push_back(n10);
                        face_nodes.push_back(n11);
                        face_offsets.push_back(static_cast<int>(face_nodes.size()));

                        face_nodes.push_back(n00);
                        face_nodes.push_back(n11);
                        face_nodes.push_back(n01);
                        face_offsets.push_back(static_cast<int>(face_nodes.size()));

                        cell_nodes.push_back(n00);
                        cell_nodes.push_back(n10);
                        cell_nodes.push_back(n11);
                        cell_nodes.push_back(n00);
                        cell_nodes.push_back(n11);
                        cell_nodes.push_back(n01);

                        cell_types.push_back("triangular");
                        out_region_ids.push_back(region_ids[r]);
                        out_target_sizes.push_back(local_size);
                        cell_types.push_back("triangular");
                        out_region_ids.push_back(region_ids[r]);
                        out_target_sizes.push_back(local_size);
                    } else {
                        if (!is_valid_triangle(p00, p10, p01) || !is_valid_triangle(p10, p11, p01)) {
                            continue;
                        }
                        face_nodes.push_back(n00);
                        face_nodes.push_back(n10);
                        face_nodes.push_back(n01);
                        face_offsets.push_back(static_cast<int>(face_nodes.size()));

                        face_nodes.push_back(n10);
                        face_nodes.push_back(n11);
                        face_nodes.push_back(n01);
                        face_offsets.push_back(static_cast<int>(face_nodes.size()));

                        cell_nodes.push_back(n00);
                        cell_nodes.push_back(n10);
                        cell_nodes.push_back(n01);
                        cell_nodes.push_back(n10);
                        cell_nodes.push_back(n11);
                        cell_nodes.push_back(n01);

                        cell_types.push_back("triangular");
                        out_region_ids.push_back(region_ids[r]);
                        out_target_sizes.push_back(local_size);
                        cell_types.push_back("triangular");
                        out_region_ids.push_back(region_ids[r]);
                        out_target_sizes.push_back(local_size);
                    }
                }
            }
        }
    }

    if (face_offsets.size() <= 1) {
        throw std::runtime_error("Hybrid CPP backend produced no computational cells");
    }

    // Constrained-edge conformance recovery on triangular faces:
    // attempt edge-flip recovery so selected boundary segments appear as explicit mesh edges.
    std::vector<double> out_node_x = registry.x();
    std::vector<double> out_node_y = registry.y();

    std::vector<std::vector<std::array<double, 2>>> constrained_boundary_lines = all_region_boundary_lines;
    constrained_boundary_lines.insert(
        constrained_boundary_lines.end(),
        all_arc_lines.begin(),
        all_arc_lines.end());

    const double constrained_quad_split_band_factor = std::max(
        0.0,
        env_double("HYDRA_HYBRIDCPP_CONSTRAINED_QUAD_SPLIT_BAND_FACTOR", 0.50));
    const double constrained_quad_split_band = std::max(
        constrained_quad_split_band_factor,
        strict_conformance ? 0.90 : 0.0) * min_target_size;

    std::vector<TriFace> tri_faces;
    tri_faces.reserve(face_offsets.size());
    std::vector<std::array<int, 4>> non_tri_faces;
    std::vector<int> non_tri_nverts;
    std::vector<std::string> non_tri_types;
    std::vector<int> non_tri_region_ids;
    std::vector<double> non_tri_target_sizes;

    for (size_t fi = 0; fi + 1 < face_offsets.size(); ++fi) {
        const int s = face_offsets[fi];
        const int e = face_offsets[fi + 1];
        const int nv = e - s;
        if (nv == 3) {
            TriFace tf;
            tf.a = face_nodes[s + 0];
            tf.b = face_nodes[s + 1];
            tf.c = face_nodes[s + 2];
            tf.region_id = out_region_ids[fi];
            tf.target_size = out_target_sizes[fi];
            tri_faces.push_back(tf);
        } else if (nv >= 4) {
            std::array<int, 4> f4 = {-1, -1, -1, -1};
            for (int k = 0; k < std::min(4, nv); ++k) {
                f4[static_cast<size_t>(k)] = face_nodes[s + k];
            }

            bool split_for_conformance = false;
            if (nv == 4 && constrained_quad_split_band > 0.0 && !constrained_boundary_lines.empty()) {
                const int n0 = f4[0];
                const int n1 = f4[1];
                const int n2 = f4[2];
                const int n3 = f4[3];
                if (n0 >= 0 && n1 >= 0 && n2 >= 0 && n3 >= 0
                    && n0 < static_cast<int>(out_node_x.size())
                    && n1 < static_cast<int>(out_node_x.size())
                    && n2 < static_cast<int>(out_node_x.size())
                    && n3 < static_cast<int>(out_node_x.size())) {
                    const double cx = 0.25 * (
                        out_node_x[static_cast<size_t>(n0)] +
                        out_node_x[static_cast<size_t>(n1)] +
                        out_node_x[static_cast<size_t>(n2)] +
                        out_node_x[static_cast<size_t>(n3)]);
                    const double cy = 0.25 * (
                        out_node_y[static_cast<size_t>(n0)] +
                        out_node_y[static_cast<size_t>(n1)] +
                        out_node_y[static_cast<size_t>(n2)] +
                        out_node_y[static_cast<size_t>(n3)]);
                    double d = std::numeric_limits<double>::infinity();
                    if (closest_point_on_any_polyline(cx, cy, constrained_boundary_lines, &d, nullptr, nullptr)
                        && d <= constrained_quad_split_band) {
                        split_for_conformance = true;
                    }
                }
            }

            if (split_for_conformance && nv == 4) {
                append_quad_as_tris(
                    f4[0],
                    f4[1],
                    f4[2],
                    f4[3],
                    out_region_ids[fi],
                    out_target_sizes[fi],
                    out_node_x,
                    out_node_y,
                    &tri_faces);
                continue;
            }

            non_tri_faces.push_back(f4);
            non_tri_nverts.push_back(nv);
            non_tri_types.push_back(cell_types[fi]);
            non_tri_region_ids.push_back(out_region_ids[fi]);
            non_tri_target_sizes.push_back(out_target_sizes[fi]);
        }
    }
    const double constrained_snap_tol = std::max(
        1.0e-9,
        env_double(
            "HYDRA_HYBRIDCPP_CONSTRAINED_EDGE_SNAP_TOL",
            std::max(1.0e-9, constrained_edge_snap_tol)));
    const int constrained_max_flips = std::max(
        8,
        static_cast<int>(std::llround(env_double(
            "HYDRA_HYBRIDCPP_CONSTRAINED_EDGE_MAX_FLIPS",
            static_cast<double>(std::max(8, constrained_edge_max_flips))))));
    const std::vector<EdgeKey> constrained_edges =
        collect_constrained_edges_from_boundaries(out_node_x, out_node_y, constrained_boundary_lines, constrained_snap_tol);
    recover_constrained_edges_by_flips(out_node_x, out_node_y, constrained_edges, &tri_faces, constrained_max_flips);

    if (strict_conformance) {
        const double strict_snap_tol = std::max(1.35 * constrained_snap_tol, constrained_snap_tol + 1.0e-9);
        const int strict_max_flips = std::max(constrained_max_flips, static_cast<int>(std::llround(1.50 * constrained_max_flips)));
        const std::vector<EdgeKey> strict_edges =
            collect_constrained_edges_from_boundaries(out_node_x, out_node_y, constrained_boundary_lines, strict_snap_tol);
        recover_constrained_edges_by_flips(out_node_x, out_node_y, strict_edges, &tri_faces, strict_max_flips);
    }

    // Rebuild face arrays with non-tri faces preserved and recovered triangles appended.
    std::vector<int> rebuilt_face_offsets;
    std::vector<int> rebuilt_face_nodes;
    std::vector<std::string> rebuilt_cell_types;
    std::vector<int> rebuilt_region_ids;
    std::vector<double> rebuilt_target_sizes;
    rebuilt_face_offsets.push_back(0);

    for (size_t i = 0; i < non_tri_faces.size(); ++i) {
        const int nv = non_tri_nverts[i];
        for (int k = 0; k < nv && k < 4; ++k) {
            rebuilt_face_nodes.push_back(non_tri_faces[i][static_cast<size_t>(k)]);
        }
        rebuilt_face_offsets.push_back(static_cast<int>(rebuilt_face_nodes.size()));
        rebuilt_cell_types.push_back(non_tri_types[i]);
        rebuilt_region_ids.push_back(non_tri_region_ids[i]);
        rebuilt_target_sizes.push_back(non_tri_target_sizes[i]);
    }

    for (const auto& t : tri_faces) {
        if (t.a < 0 || t.b < 0 || t.c < 0) {
            continue;
        }
        const std::array<double, 2> pa = {out_node_x[static_cast<size_t>(t.a)], out_node_y[static_cast<size_t>(t.a)]};
        const std::array<double, 2> pb = {out_node_x[static_cast<size_t>(t.b)], out_node_y[static_cast<size_t>(t.b)]};
        const std::array<double, 2> pc = {out_node_x[static_cast<size_t>(t.c)], out_node_y[static_cast<size_t>(t.c)]};
        if (!is_valid_triangle(pa, pb, pc)) {
            continue;
        }
        rebuilt_face_nodes.push_back(t.a);
        rebuilt_face_nodes.push_back(t.b);
        rebuilt_face_nodes.push_back(t.c);
        rebuilt_face_offsets.push_back(static_cast<int>(rebuilt_face_nodes.size()));
        rebuilt_cell_types.push_back("triangular");
        rebuilt_region_ids.push_back(t.region_id);
        rebuilt_target_sizes.push_back(t.target_size);
    }

    if (rebuilt_face_offsets.size() <= 1) {
        throw std::runtime_error("Hybrid CPP constrained-edge recovery produced no faces");
    }

    // Rebuild plot triangles from recovered faces.
    std::vector<int> rebuilt_cell_nodes;
    rebuilt_cell_nodes.reserve(rebuilt_face_nodes.size() * 2);
    for (size_t fi = 0; fi + 1 < rebuilt_face_offsets.size(); ++fi) {
        const int s = rebuilt_face_offsets[fi];
        const int e = rebuilt_face_offsets[fi + 1];
        const int nv = e - s;
        if (nv == 3) {
            rebuilt_cell_nodes.push_back(rebuilt_face_nodes[s + 0]);
            rebuilt_cell_nodes.push_back(rebuilt_face_nodes[s + 1]);
            rebuilt_cell_nodes.push_back(rebuilt_face_nodes[s + 2]);
        } else if (nv == 4) {
            const int n0 = rebuilt_face_nodes[s + 0];
            const int n1 = rebuilt_face_nodes[s + 1];
            const int n2 = rebuilt_face_nodes[s + 2];
            const int n3 = rebuilt_face_nodes[s + 3];
            rebuilt_cell_nodes.push_back(n0);
            rebuilt_cell_nodes.push_back(n1);
            rebuilt_cell_nodes.push_back(n2);
            rebuilt_cell_nodes.push_back(n0);
            rebuilt_cell_nodes.push_back(n2);
            rebuilt_cell_nodes.push_back(n3);
        }
    }

    face_offsets.swap(rebuilt_face_offsets);
    face_nodes.swap(rebuilt_face_nodes);
    cell_nodes.swap(rebuilt_cell_nodes);
    cell_types.swap(rebuilt_cell_types);
    out_region_ids.swap(rebuilt_region_ids);
    out_target_sizes.swap(rebuilt_target_sizes);

    std::vector<std::vector<std::array<double, 2>>> quality_lock_lines = all_region_boundary_lines;
    quality_lock_lines.insert(quality_lock_lines.end(), all_arc_lines.begin(), all_arc_lines.end());
    const double quality_lock_tol_factor = std::max(
        0.0,
        env_double("HYDRA_HYBRIDCPP_QUALITY_LOCK_TOL_FACTOR", 0.10));
    const double quality_lock_tol = std::max(1.0e-9, quality_lock_tol_factor * min_target_size);
    const int quality_sweeps = std::max(0, static_cast<int>(std::llround(env_double("HYDRA_HYBRIDCPP_QUALITY_SWEEPS", 4.0))));
    const double quality_relax = env_double("HYDRA_HYBRIDCPP_QUALITY_RELAX", 0.35);
    const double quality_move = env_double("HYDRA_HYBRIDCPP_QUALITY_MAX_MOVE_FACTOR", 0.30);

    optimize_mesh_quality(
        &out_node_x,
        &out_node_y,
        face_offsets,
        face_nodes,
        quality_lock_lines,
        quality_lock_tol,
        quality_sweeps,
        quality_relax,
        quality_move);

    py::dict out;
    out["node_x"] = py::array_t<double>(out_node_x.size(), out_node_x.data());
    out["node_y"] = py::array_t<double>(out_node_y.size(), out_node_y.data());
    out["cell_nodes"] = py::array_t<int>(cell_nodes.size(), cell_nodes.data());
    out["cell_face_offsets"] = py::array_t<int>(face_offsets.size(), face_offsets.data());
    out["cell_face_nodes"] = py::array_t<int>(face_nodes.size(), face_nodes.data());
    out["cell_type"] = py::cast(cell_types);
    out["region_id"] = py::array_t<int>(out_region_ids.size(), out_region_ids.data());
    out["target_size"] = py::array_t<double>(out_target_sizes.size(), out_target_sizes.data());
    return out;
}

PYBIND11_MODULE(hydra_hybridmesh, m) {
    m.doc() = "Custom hybrid topology mesher for SWE2D (CPP backend)";
    m.def(
        "generate_hybrid_mesh",
        &generate_hybrid_mesh,
        py::arg("region_rings"),
        py::arg("region_holes"),
        py::arg("region_target_sizes"),
        py::arg("region_cell_types"),
        py::arg("region_ids"),
        py::arg("constraint_rings") = std::vector<std::vector<std::array<double, 2>>>{},
        py::arg("constraint_target_sizes") = std::vector<double>{},
        py::arg("constraint_cell_types") = std::vector<std::string>{},
        py::arg("arc_region_ids") = std::vector<int>{},
        py::arg("arc_roles") = std::vector<std::string>{},
        py::arg("arc_lines") = std::vector<std::vector<std::array<double, 2>>>{},
        py::arg("tri_meshing_method") = "frontal_delaunay",
        py::arg("transition_width_factor") = 1.25,
        py::arg("transition_outer_factor") = 2.5,
        py::arg("overbank_grading_factor") = 4.0,
        py::arg("constrained_edge_snap_tol") = 12.0,
        py::arg("constrained_edge_max_flips") = 128,
        py::arg("region_conformance_band_factor") = 0.55,
        py::arg("arc_conformance_band_factor") = 0.45,
        py::arg("strict_conformance_mode") = false,
        R"doc(
Generate a hybrid topology mesh in C++ for SWE2D.

- Quad-like cells for region/constraint types: cartesian, quadrilateral, channel_generator.
- Triangular cells for triangular regions/constraints.
- Runtime-selectable triangular strategies:
    - frontal_delaunay
    - direct_curvilinear_generation
    - anisotropic_delaunay_refinement
    - advancing_front
- Excludes cells outside region rings and inside region holes.
    - For channel_generator regions with centerline/bank arcs, uses centerline-curvilinear
      station/normal sweeping to align cells with local flow direction.
    - Builds graded inner/outer transition bands near banks to avoid abrupt quad/tri jumps.
- Applies overbank-side triangular size grading and arc-guided snapping/refinement.
- Applies boundary-aware fill/projection for partial boundary cells to reduce holes/gaps.
- Runs interior quality optimization sweeps with boundary/arc-locked nodes.
)doc");
}
