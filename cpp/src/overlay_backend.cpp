#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

bool has_valid_shape(const py::array_t<double>& arr) {
    return arr.ndim() == 1 && arr.size() > 0;
}

bool has_valid_shape_2d(const py::array_t<double>& arr) {
    return arr.ndim() == 2 && arr.shape(0) > 0 && arr.shape(1) > 0;
}

bool has_valid_shape_i32(const py::array_t<int32_t>& arr) {
    return arr.ndim() == 1 && arr.size() > 0;
}

py::dict rasterize_unstructured_accum(
    const py::array_t<double>& x,
    const py::array_t<double>& y,
    const py::array_t<double>& scalar,
    const py::object& u_obj,
    const py::object& v_obj,
    int width,
    int height,
    double xmin,
    double xmax,
    double ymin,
    double ymax
) {
    if (!has_valid_shape(x) || !has_valid_shape(y) || !has_valid_shape(scalar)) {
        throw std::runtime_error("x/y/scalar must be non-empty 1D float arrays");
    }
    if (width < 2 || height < 2) {
        throw std::runtime_error("width and height must be >= 2");
    }

    const bool have_uv = !u_obj.is_none() && !v_obj.is_none();
    py::array_t<double> u;
    py::array_t<double> v;
    if (have_uv) {
        u = py::cast<py::array_t<double>>(u_obj);
        v = py::cast<py::array_t<double>>(v_obj);
        if (!has_valid_shape(u) || !has_valid_shape(v)) {
            throw std::runtime_error("u/v must be non-empty 1D float arrays when provided");
        }
    }

    const auto bx = x.unchecked<1>();
    const auto by = y.unchecked<1>();
    const auto bs = scalar.unchecked<1>();

    const ssize_t n = std::min({bx.size(), by.size(), bs.size(), have_uv ? py::ssize_t(u.size()) : py::ssize_t(std::numeric_limits<ssize_t>::max()), have_uv ? py::ssize_t(v.size()) : py::ssize_t(std::numeric_limits<ssize_t>::max())});
    if (n <= 0) {
        throw std::runtime_error("No overlapping samples");
    }

    const double xspan = std::max(1.0e-12, xmax - xmin);
    const double yspan = std::max(1.0e-12, ymax - ymin);
    const double sx = static_cast<double>(width - 1) / xspan;
    const double sy = static_cast<double>(height - 1) / yspan;

    py::array_t<double> sum_scalar({height, width});
    py::array_t<double> count({height, width});
    auto sum2 = sum_scalar.mutable_unchecked<2>();
    auto cnt2 = count.mutable_unchecked<2>();

    for (int iy = 0; iy < height; ++iy) {
        for (int ix = 0; ix < width; ++ix) {
            sum2(iy, ix) = 0.0;
            cnt2(iy, ix) = 0.0;
        }
    }

    py::array_t<double> sum_u;
    py::array_t<double> sum_v;

    if (have_uv) {
        sum_u = py::array_t<double>({height, width});
        sum_v = py::array_t<double>({height, width});
        auto su2 = sum_u.mutable_unchecked<2>();
        auto sv2 = sum_v.mutable_unchecked<2>();
        auto bu = u.unchecked<1>();
        auto bv = v.unchecked<1>();

        for (int iy = 0; iy < height; ++iy) {
            for (int ix = 0; ix < width; ++ix) {
                su2(iy, ix) = 0.0;
                sv2(iy, ix) = 0.0;
            }
        }
        for (ssize_t i = 0; i < n; ++i) {
            const double xi = bx(i);
            const double yi = by(i);
            const double si = bs(i);
            if (!std::isfinite(xi) || !std::isfinite(yi) || !std::isfinite(si)) {
                continue;
            }

            int ix = static_cast<int>(std::llround((xi - xmin) * sx));
            int iy = static_cast<int>(std::llround((ymax - yi) * sy));
            ix = std::max(0, std::min(width - 1, ix));
            iy = std::max(0, std::min(height - 1, iy));

            sum2(iy, ix) += si;
            cnt2(iy, ix) += 1.0;

            const double ui = bu(i);
            const double vi = bv(i);
            if (std::isfinite(ui) && std::isfinite(vi)) {
                su2(iy, ix) += ui;
                sv2(iy, ix) += vi;
            }
        }
    } else {
        for (ssize_t i = 0; i < n; ++i) {
            const double xi = bx(i);
            const double yi = by(i);
            const double si = bs(i);
            if (!std::isfinite(xi) || !std::isfinite(yi) || !std::isfinite(si)) {
                continue;
            }

            int ix = static_cast<int>(std::llround((xi - xmin) * sx));
            int iy = static_cast<int>(std::llround((ymax - yi) * sy));
            ix = std::max(0, std::min(width - 1, ix));
            iy = std::max(0, std::min(height - 1, iy));

            sum2(iy, ix) += si;
            cnt2(iy, ix) += 1.0;
        }
    }

    py::dict out;
    out["sum_scalar"] = sum_scalar;
    out["count"] = count;
    if (have_uv) {
        out["sum_u"] = sum_u;
        out["sum_v"] = sum_v;
    } else {
        out["sum_u"] = py::array_t<double>({0});
        out["sum_v"] = py::array_t<double>({0});
    }
    return out;
}

py::dict rasterize_tri_mesh_accum(
    const py::array_t<double>& node_x,
    const py::array_t<double>& node_y,
    const py::array_t<int32_t>& tri_nodes,
    const py::array_t<double>& scalar_cell,
    const py::object& u_obj,
    const py::object& v_obj,
    int width,
    int height,
    double xmin,
    double xmax,
    double ymin,
    double ymax
) {
    if (!has_valid_shape(node_x) || !has_valid_shape(node_y) || !has_valid_shape_i32(tri_nodes) || !has_valid_shape(scalar_cell)) {
        throw std::runtime_error("node_x/node_y/tri_nodes/scalar_cell must be non-empty arrays");
    }
    if (width < 2 || height < 2) {
        throw std::runtime_error("width and height must be >= 2");
    }

    const auto nx = node_x.unchecked<1>();
    const auto ny = node_y.unchecked<1>();
    const auto tri = tri_nodes.unchecked<1>();
    const auto sc = scalar_cell.unchecked<1>();

    if (tri.size() % 3 != 0) {
        throw std::runtime_error("tri_nodes length must be divisible by 3");
    }

    const ssize_t n_nodes = std::min(nx.size(), ny.size());
    const ssize_t n_tri = tri.size() / 3;
    const ssize_t n_cell = std::min(n_tri, sc.size());
    if (n_cell <= 0 || n_nodes <= 0) {
        throw std::runtime_error("No overlapping triangles/cell values");
    }

    const bool have_uv = !u_obj.is_none() && !v_obj.is_none();
    py::array_t<double> u;
    py::array_t<double> v;
    if (have_uv) {
        u = py::cast<py::array_t<double>>(u_obj);
        v = py::cast<py::array_t<double>>(v_obj);
        if (!has_valid_shape(u) || !has_valid_shape(v)) {
            throw std::runtime_error("u/v must be non-empty 1D float arrays when provided");
        }
    }

    const double xspan = std::max(1.0e-12, xmax - xmin);
    const double yspan = std::max(1.0e-12, ymax - ymin);
    const double sx = static_cast<double>(width - 1) / xspan;
    const double sy = static_cast<double>(height - 1) / yspan;

    py::array_t<double> sum_scalar({height, width});
    py::array_t<double> count({height, width});
    auto sum2 = sum_scalar.mutable_unchecked<2>();
    auto cnt2 = count.mutable_unchecked<2>();

    for (int iy = 0; iy < height; ++iy) {
        for (int ix = 0; ix < width; ++ix) {
            sum2(iy, ix) = 0.0;
            cnt2(iy, ix) = 0.0;
        }
    }

    py::array_t<double> sum_u;
    py::array_t<double> sum_v;
    const double* bu_ptr = nullptr;
    const double* bv_ptr = nullptr;
    ssize_t uv_n = 0;
    double* su_ptr = nullptr;
    double* sv_ptr = nullptr;

    if (have_uv) {
        sum_u = py::array_t<double>({height, width});
        sum_v = py::array_t<double>({height, width});
        su_ptr = sum_u.mutable_data();
        sv_ptr = sum_v.mutable_data();
        bu_ptr = u.data();
        bv_ptr = v.data();
        uv_n = std::min(u.size(), v.size());
        std::fill(su_ptr, su_ptr + static_cast<ssize_t>(height) * static_cast<ssize_t>(width), 0.0);
        std::fill(sv_ptr, sv_ptr + static_cast<ssize_t>(height) * static_cast<ssize_t>(width), 0.0);
    }

    const double eps = 1.0e-12;

    for (ssize_t c = 0; c < n_cell; ++c) {
        const int32_t ia = tri(3 * c + 0);
        const int32_t ib = tri(3 * c + 1);
        const int32_t ic = tri(3 * c + 2);
        if (ia < 0 || ib < 0 || ic < 0 || ia >= n_nodes || ib >= n_nodes || ic >= n_nodes) {
            continue;
        }

        const double sval = sc(c);
        if (!std::isfinite(sval)) {
            continue;
        }

        const double ax = (nx(ia) - xmin) * sx;
        const double ay = (ymax - ny(ia)) * sy;
        const double bx = (nx(ib) - xmin) * sx;
        const double by = (ymax - ny(ib)) * sy;
        const double cx = (nx(ic) - xmin) * sx;
        const double cy = (ymax - ny(ic)) * sy;

        if (!std::isfinite(ax) || !std::isfinite(ay) || !std::isfinite(bx) || !std::isfinite(by) || !std::isfinite(cx) || !std::isfinite(cy)) {
            continue;
        }

        const double area2 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
        if (!std::isfinite(area2) || std::abs(area2) <= eps) {
            continue;
        }

        const int ix_min = std::max(0, static_cast<int>(std::floor(std::min({ax, bx, cx}) - 0.5)));
        const int ix_max = std::min(width - 1, static_cast<int>(std::ceil(std::max({ax, bx, cx}) - 0.5)));
        const int iy_min = std::max(0, static_cast<int>(std::floor(std::min({ay, by, cy}) - 0.5)));
        const int iy_max = std::min(height - 1, static_cast<int>(std::ceil(std::max({ay, by, cy}) - 0.5)));

        if (ix_min > ix_max || iy_min > iy_max) {
            continue;
        }

        const double ui = (have_uv && c < uv_n) ? bu_ptr[c] : 0.0;
        const double vi = (have_uv && c < uv_n) ? bv_ptr[c] : 0.0;
        const bool uv_ok = have_uv && std::isfinite(ui) && std::isfinite(vi);

        for (int iy = iy_min; iy <= iy_max; ++iy) {
            const double py = static_cast<double>(iy) + 0.5;
            for (int ix = ix_min; ix <= ix_max; ++ix) {
                const double px = static_cast<double>(ix) + 0.5;
                const double e0 = (bx - ax) * (py - ay) - (by - ay) * (px - ax);
                const double e1 = (cx - bx) * (py - by) - (cy - by) * (px - bx);
                const double e2 = (ax - cx) * (py - cy) - (ay - cy) * (px - cx);
                const bool inside_pos = (e0 >= -eps) && (e1 >= -eps) && (e2 >= -eps);
                const bool inside_neg = (e0 <= eps) && (e1 <= eps) && (e2 <= eps);
                if (!(inside_pos || inside_neg)) {
                    continue;
                }
                sum2(iy, ix) += sval;
                cnt2(iy, ix) += 1.0;
                if (uv_ok) {
                    const ssize_t p = static_cast<ssize_t>(iy) * static_cast<ssize_t>(width) + static_cast<ssize_t>(ix);
                    su_ptr[p] += ui;
                    sv_ptr[p] += vi;
                }
            }
        }
    }

    py::dict out;
    out["sum_scalar"] = sum_scalar;
    out["count"] = count;
    if (have_uv) {
        out["sum_u"] = sum_u;
        out["sum_v"] = sum_v;
    } else {
        out["sum_u"] = py::array_t<double>({0});
        out["sum_v"] = py::array_t<double>({0});
    }
    return out;
}

py::dict finalize_scalar_field(
    const py::array_t<double>& sum_scalar,
    const py::array_t<double>& count,
    int dilate_radius = 0
) {
    if (!has_valid_shape_2d(sum_scalar) || !has_valid_shape_2d(count)) {
        throw std::runtime_error("sum_scalar/count must be non-empty 2D float arrays");
    }
    if (sum_scalar.shape(0) != count.shape(0) || sum_scalar.shape(1) != count.shape(1)) {
        throw std::runtime_error("sum_scalar/count shape mismatch");
    }

    const int h = static_cast<int>(sum_scalar.shape(0));
    const int w = static_cast<int>(sum_scalar.shape(1));

    auto sum2 = sum_scalar.unchecked<2>();
    auto cnt2 = count.unchecked<2>();

    py::array_t<double> grid({h, w});
    py::array_t<uint8_t> known_mask({h, w});
    py::array_t<uint8_t> shell_mask({h, w});
    auto g2 = grid.mutable_unchecked<2>();
    auto k2 = known_mask.mutable_unchecked<2>();
    auto sh2 = shell_mask.mutable_unchecked<2>();

    std::size_t known_count = 0;
#ifdef HYDRA_HAS_OPENMP
#pragma omp parallel for reduction(+ : known_count) collapse(2)
#endif
    for (int iy = 0; iy < h; ++iy) {
        for (int ix = 0; ix < w; ++ix) {
            const double c = cnt2(iy, ix);
            if (c > 0.0) {
                g2(iy, ix) = sum2(iy, ix) / c;
                k2(iy, ix) = static_cast<uint8_t>(1);
                ++known_count;
            } else {
                g2(iy, ix) = std::numeric_limits<double>::quiet_NaN();
                k2(iy, ix) = static_cast<uint8_t>(0);
            }
            sh2(iy, ix) = static_cast<uint8_t>(0);
        }
    }

    if (known_count == 0) {
        py::dict out;
        out["grid"] = grid;
        out["known_mask"] = known_mask;
        out["shell_mask"] = shell_mask;
        return out;
    }

    int radius = dilate_radius;
    if (radius <= 0) {
        const double spacing = std::sqrt((static_cast<double>(w) * static_cast<double>(h)) / std::max(1.0, static_cast<double>(known_count)));
        radius = static_cast<int>(std::llround(0.55 * spacing));
        radius = std::max(1, std::min(4, radius));
    }

    std::vector<uint8_t> shell_seed(static_cast<std::size_t>(h) * static_cast<std::size_t>(w), 0);
    for (int iy = 0; iy < h; ++iy) {
        for (int ix = 0; ix < w; ++ix) {
            if (!k2(iy, ix)) {
                continue;
            }
            const int y0 = std::max(0, iy - radius);
            const int y1 = std::min(h - 1, iy + radius);
            const int x0 = std::max(0, ix - radius);
            const int x1 = std::min(w - 1, ix + radius);
            for (int yy = y0; yy <= y1; ++yy) {
                for (int xx = x0; xx <= x1; ++xx) {
                    shell_seed[static_cast<std::size_t>(yy) * static_cast<std::size_t>(w) + static_cast<std::size_t>(xx)] = static_cast<uint8_t>(1);
                }
            }
        }
    }

    std::vector<uint8_t> outside(static_cast<std::size_t>(h) * static_cast<std::size_t>(w), 0);
    std::deque<std::pair<int, int>> q;

    auto try_push = [&](int yy, int xx) {
        const std::size_t idx = static_cast<std::size_t>(yy) * static_cast<std::size_t>(w) + static_cast<std::size_t>(xx);
        if (!shell_seed[idx] && !outside[idx]) {
            outside[idx] = static_cast<uint8_t>(1);
            q.emplace_back(yy, xx);
        }
    };

    for (int x = 0; x < w; ++x) {
        try_push(0, x);
        try_push(h - 1, x);
    }
    for (int y = 0; y < h; ++y) {
        try_push(y, 0);
        try_push(y, w - 1);
    }

    while (!q.empty()) {
        const auto [y, x] = q.front();
        q.pop_front();
        if (y > 0) try_push(y - 1, x);
        if (y + 1 < h) try_push(y + 1, x);
        if (x > 0) try_push(y, x - 1);
        if (x + 1 < w) try_push(y, x + 1);
    }

    for (int iy = 0; iy < h; ++iy) {
        for (int ix = 0; ix < w; ++ix) {
            const std::size_t idx = static_cast<std::size_t>(iy) * static_cast<std::size_t>(w) + static_cast<std::size_t>(ix);
            sh2(iy, ix) = outside[idx] ? static_cast<uint8_t>(0) : static_cast<uint8_t>(1);
        }
    }

    py::dict out;
    out["grid"] = grid;
    out["known_mask"] = known_mask;
    out["shell_mask"] = shell_mask;
    return out;
}

py::array_t<double> nearest_fill(
    const py::array_t<double>& values,
    const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& known_mask
) {
    if (!has_valid_shape_2d(values) || known_mask.ndim() != 2 || known_mask.shape(0) != values.shape(0) || known_mask.shape(1) != values.shape(1)) {
        throw std::runtime_error("values/known_mask must be same-shape 2D arrays");
    }

    const int h = static_cast<int>(values.shape(0));
    const int w = static_cast<int>(values.shape(1));
    auto v2 = values.unchecked<2>();
    auto k2 = known_mask.unchecked<2>();

    py::array_t<double> out({h, w});
    auto o2 = out.mutable_unchecked<2>();

    std::vector<int> src_y(static_cast<std::size_t>(h) * static_cast<std::size_t>(w), -1);
    std::vector<int> src_x(static_cast<std::size_t>(h) * static_cast<std::size_t>(w), -1);
    std::vector<double> dist(static_cast<std::size_t>(h) * static_cast<std::size_t>(w), 1.0e30);

    for (int iy = 0; iy < h; ++iy) {
        for (int ix = 0; ix < w; ++ix) {
            o2(iy, ix) = v2(iy, ix);
            const std::size_t idx = static_cast<std::size_t>(iy) * static_cast<std::size_t>(w) + static_cast<std::size_t>(ix);
            if (k2(iy, ix)) {
                src_y[idx] = iy;
                src_x[idx] = ix;
                dist[idx] = 0.0;
            }
        }
    }

    for (int iy = 0; iy < h; ++iy) {
        for (int ix = 0; ix < w; ++ix) {
            const std::size_t idx = static_cast<std::size_t>(iy) * static_cast<std::size_t>(w) + static_cast<std::size_t>(ix);
            if (iy > 0) {
                const std::size_t nidx = static_cast<std::size_t>(iy - 1) * static_cast<std::size_t>(w) + static_cast<std::size_t>(ix);
                if (src_y[nidx] >= 0) {
                    const double cand = dist[nidx] + 1.0;
                    if (cand < dist[idx]) {
                        dist[idx] = cand;
                        src_y[idx] = src_y[nidx];
                        src_x[idx] = src_x[nidx];
                    }
                }
            }
            if (ix > 0) {
                const std::size_t nidx = static_cast<std::size_t>(iy) * static_cast<std::size_t>(w) + static_cast<std::size_t>(ix - 1);
                if (src_y[nidx] >= 0) {
                    const double cand = dist[nidx] + 1.0;
                    if (cand < dist[idx]) {
                        dist[idx] = cand;
                        src_y[idx] = src_y[nidx];
                        src_x[idx] = src_x[nidx];
                    }
                }
            }
        }
    }

    for (int iy = h - 1; iy >= 0; --iy) {
        for (int ix = w - 1; ix >= 0; --ix) {
            const std::size_t idx = static_cast<std::size_t>(iy) * static_cast<std::size_t>(w) + static_cast<std::size_t>(ix);
            if (iy + 1 < h) {
                const std::size_t nidx = static_cast<std::size_t>(iy + 1) * static_cast<std::size_t>(w) + static_cast<std::size_t>(ix);
                if (src_y[nidx] >= 0) {
                    const double cand = dist[nidx] + 1.0;
                    if (cand < dist[idx]) {
                        dist[idx] = cand;
                        src_y[idx] = src_y[nidx];
                        src_x[idx] = src_x[nidx];
                    }
                }
            }
            if (ix + 1 < w) {
                const std::size_t nidx = static_cast<std::size_t>(iy) * static_cast<std::size_t>(w) + static_cast<std::size_t>(ix + 1);
                if (src_y[nidx] >= 0) {
                    const double cand = dist[nidx] + 1.0;
                    if (cand < dist[idx]) {
                        dist[idx] = cand;
                        src_y[idx] = src_y[nidx];
                        src_x[idx] = src_x[nidx];
                    }
                }
            }
        }
    }

    for (int iy = 0; iy < h; ++iy) {
        for (int ix = 0; ix < w; ++ix) {
            if (k2(iy, ix)) {
                continue;
            }
            const std::size_t idx = static_cast<std::size_t>(iy) * static_cast<std::size_t>(w) + static_cast<std::size_t>(ix);
            const int sy = src_y[idx];
            const int sx = src_x[idx];
            if (sy >= 0 && sx >= 0) {
                o2(iy, ix) = o2(sy, sx);
            }
        }
    }

    return out;
}

py::dict advect_streamlines(
    const py::array_t<double>& u_grid,
    const py::array_t<double>& v_grid,
    const py::array_t<double>& speed_grid,
    const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& shell_mask,
    const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& seed_mask,
    int seed_count,
    int max_steps,
    double step_px,
    double min_speed,
    const std::string& backend = "auto"
) {
    if (!has_valid_shape_2d(u_grid) || !has_valid_shape_2d(v_grid) || !has_valid_shape_2d(speed_grid)) {
        throw std::runtime_error("u_grid/v_grid/speed_grid must be non-empty 2D float arrays");
    }
    if (u_grid.shape(0) != v_grid.shape(0) || u_grid.shape(1) != v_grid.shape(1) ||
        u_grid.shape(0) != speed_grid.shape(0) || u_grid.shape(1) != speed_grid.shape(1)) {
        throw std::runtime_error("u_grid/v_grid/speed_grid shape mismatch");
    }
    if (shell_mask.ndim() != 2 || seed_mask.ndim() != 2 ||
        shell_mask.shape(0) != u_grid.shape(0) || shell_mask.shape(1) != u_grid.shape(1) ||
        seed_mask.shape(0) != u_grid.shape(0) || seed_mask.shape(1) != u_grid.shape(1)) {
        throw std::runtime_error("shell_mask/seed_mask must match grid shape");
    }

    const int h = static_cast<int>(u_grid.shape(0));
    const int w = static_cast<int>(u_grid.shape(1));
    const int n_seed_target = std::max(1, seed_count);
    const int n_steps = std::max(2, max_steps);
    const int max_points = n_steps + 1;
    const double ds = std::max(1.0e-6, step_px);
    const double smin = std::max(0.0, min_speed);

    auto u2 = u_grid.unchecked<2>();
    auto v2 = v_grid.unchecked<2>();
    auto s2 = speed_grid.unchecked<2>();
    auto sh2 = shell_mask.unchecked<2>();
    auto sd2 = seed_mask.unchecked<2>();

    std::vector<std::pair<double, double>> candidates;
    candidates.reserve(static_cast<std::size_t>(h) * static_cast<std::size_t>(w) / 8u);
    for (int iy = 0; iy < h; ++iy) {
        for (int ix = 0; ix < w; ++ix) {
            if (!sd2(iy, ix) || !sh2(iy, ix)) {
                continue;
            }
            const double sp = s2(iy, ix);
            if (!std::isfinite(sp) || sp < smin) {
                continue;
            }
            candidates.emplace_back(static_cast<double>(ix), static_cast<double>(iy));
        }
    }

    std::vector<std::pair<double, double>> seeds;
    if (!candidates.empty()) {
        const int pick_step = std::max(1, static_cast<int>(candidates.size() / static_cast<std::size_t>(n_seed_target)));
        for (std::size_t i = 0; i < candidates.size() && static_cast<int>(seeds.size()) < n_seed_target; i += static_cast<std::size_t>(pick_step)) {
            seeds.push_back(candidates[i]);
        }
    }

    const int n_traces = static_cast<int>(seeds.size());
    py::array_t<int32_t> counts({n_traces});
    py::array_t<double> mean_speed({n_traces});
    py::array_t<double> length({n_traces});
    py::array_t<double> xy({n_traces, max_points, 2});

    auto c1 = counts.mutable_unchecked<1>();
    auto ms1 = mean_speed.mutable_unchecked<1>();
    auto ln1 = length.mutable_unchecked<1>();
    auto xy3 = xy.mutable_unchecked<3>();

#ifdef HYDRA_HAS_OPENMP
#pragma omp parallel for collapse(3)
#endif
    for (int it = 0; it < n_traces; ++it) {
        for (int ip = 0; ip < max_points; ++ip) {
            for (int k = 0; k < 2; ++k) {
                xy3(it, ip, k) = std::numeric_limits<double>::quiet_NaN();
            }
        }
    }

#ifdef HYDRA_HAS_OPENMP
#pragma omp parallel for
#endif
    for (int it = 0; it < n_traces; ++it) {
        double x = seeds[static_cast<std::size_t>(it)].first;
        double y = seeds[static_cast<std::size_t>(it)].second;

        int npt = 1;
        double speed_acc = 0.0;
        int speed_cnt = 0;
        double len_acc = 0.0;

        xy3(it, 0, 0) = x;
        xy3(it, 0, 1) = y;

        for (int st = 0; st < n_steps; ++st) {
            if (x < 0.0 || y < 0.0 || x >= static_cast<double>(w - 1) || y >= static_cast<double>(h - 1)) {
                break;
            }

            const int x0 = static_cast<int>(x);
            const int y0 = static_cast<int>(y);
            const double tx = x - static_cast<double>(x0);
            const double ty = y - static_cast<double>(y0);

            const double u00 = u2(y0, x0);
            const double u10 = u2(y0, x0 + 1);
            const double u01 = u2(y0 + 1, x0);
            const double u11 = u2(y0 + 1, x0 + 1);
            const double v00 = v2(y0, x0);
            const double v10 = v2(y0, x0 + 1);
            const double v01 = v2(y0 + 1, x0);
            const double v11 = v2(y0 + 1, x0 + 1);
            const double s00 = s2(y0, x0);
            const double s10 = s2(y0, x0 + 1);
            const double s01 = s2(y0 + 1, x0);
            const double s11 = s2(y0 + 1, x0 + 1);

            if (!std::isfinite(u00) || !std::isfinite(u10) || !std::isfinite(u01) || !std::isfinite(u11) ||
                !std::isfinite(v00) || !std::isfinite(v10) || !std::isfinite(v01) || !std::isfinite(v11) ||
                !std::isfinite(s00) || !std::isfinite(s10) || !std::isfinite(s01) || !std::isfinite(s11)) {
                break;
            }

            const double ui =
                (1.0 - tx) * (1.0 - ty) * u00 + tx * (1.0 - ty) * u10 + (1.0 - tx) * ty * u01 + tx * ty * u11;
            const double vi =
                (1.0 - tx) * (1.0 - ty) * v00 + tx * (1.0 - ty) * v10 + (1.0 - tx) * ty * v01 + tx * ty * v11;
            const double si =
                (1.0 - tx) * (1.0 - ty) * s00 + tx * (1.0 - ty) * s10 + (1.0 - tx) * ty * s01 + tx * ty * s11;

            if (!std::isfinite(ui) || !std::isfinite(vi) || !std::isfinite(si) || si < smin) {
                break;
            }

            const double dn = std::max(1.0e-8, std::hypot(ui, vi));
            const double dx = ds * (ui / dn);
            const double dy = -ds * (vi / dn);

            const double xn = x + dx;
            const double yn = y + dy;

            if (xn < 1.0 || yn < 1.0 || xn > static_cast<double>(w - 2) || yn > static_cast<double>(h - 2)) {
                break;
            }

            const int ixn = static_cast<int>(std::llround(xn));
            const int iyn = static_cast<int>(std::llround(yn));
            if (ixn < 0 || ixn >= w || iyn < 0 || iyn >= h || !sh2(iyn, ixn)) {
                break;
            }

            x = xn;
            y = yn;
            if (npt < max_points) {
                xy3(it, npt, 0) = x;
                xy3(it, npt, 1) = y;
            }
            ++npt;
            len_acc += std::hypot(dx, dy);
            speed_acc += si;
            ++speed_cnt;
        }

        c1(it) = static_cast<int32_t>(npt);
        ms1(it) = (speed_cnt > 0) ? (speed_acc / static_cast<double>(speed_cnt)) : 0.0;
        ln1(it) = len_acc;
    }

    std::string backend_used = "cpu_serial";
#ifdef HYDRA_HAS_OPENMP
    backend_used = "cpu_openmp";
#endif
#ifdef HYDRA_HAS_CUDA
    if (backend == "cuda") {
        backend_used = "cuda_api_cpu_fallback";
    } else if (backend == "auto") {
        backend_used = "cpu_openmp_cuda_api_ready";
    }
#endif

    py::dict out;
    out["counts"] = counts;
    out["mean_speed"] = mean_speed;
    out["length"] = length;
    out["xy"] = xy;
    out["backend_used"] = py::str(backend_used);
    return out;
}

} // namespace

PYBIND11_MODULE(hydra_overlay, m) {
    m.doc() = "Compiled high-performance overlay raster core (CPU now, CUDA/OpenGL-ready API surface)";

    m.def("rasterize_unstructured_accum", &rasterize_unstructured_accum,
          py::arg("x"), py::arg("y"), py::arg("scalar"),
          py::arg("u") = py::none(), py::arg("v") = py::none(),
          py::arg("width"), py::arg("height"),
          py::arg("xmin"), py::arg("xmax"), py::arg("ymin"), py::arg("ymax"),
          "Rasterize unstructured samples into per-pixel sums/counts in compiled code.");

        m.def("rasterize_tri_mesh_accum", &rasterize_tri_mesh_accum,
            py::arg("node_x"), py::arg("node_y"), py::arg("tri_nodes"), py::arg("scalar_cell"),
            py::arg("u") = py::none(), py::arg("v") = py::none(),
            py::arg("width"), py::arg("height"),
            py::arg("xmin"), py::arg("xmax"), py::arg("ymin"), py::arg("ymax"),
            "Rasterize triangulated cell fields into per-pixel sums/counts by filling triangle area in compiled code.");

        m.def("finalize_scalar_field", &finalize_scalar_field,
            py::arg("sum_scalar"), py::arg("count"), py::arg("dilate_radius") = 0,
            "Build shell mask and filled scalar grid from accumulated sums/counts.");

        m.def("nearest_fill", &nearest_fill,
            py::arg("values"), py::arg("known_mask"),
            "Nearest-like fill of unknown grid cells from known-mask seeds.");

        m.def("advect_streamlines", &advect_streamlines,
            py::arg("u_grid"), py::arg("v_grid"), py::arg("speed_grid"),
            py::arg("shell_mask"), py::arg("seed_mask"),
            py::arg("seed_count"), py::arg("max_steps"), py::arg("step_px"), py::arg("min_speed"),
            py::arg("backend") = "auto",
            "Advect streamline traces on a gridded velocity field (CPU OpenMP path, CUDA API-ready)."
        );

    m.def("capabilities", []() {
        py::dict out;
        out["backend"] = py::str("compiled_cpu");
#ifdef HYDRA_HAS_OPENMP
        out["openmp"] = py::bool_(true);
#else
        out["openmp"] = py::bool_(false);
#endif
#ifdef HYDRA_HAS_CUDA
        out["cuda"] = py::bool_(true);
#else
        out["cuda"] = py::bool_(false);
#endif
        out["opengl_interop"] = py::bool_(false);
        return out;
    });
}
