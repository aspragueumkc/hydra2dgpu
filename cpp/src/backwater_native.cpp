#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#if defined(BACKWATER_HAS_OPENMP)
#include <omp.h>
#endif

namespace py = pybind11;

namespace {

using Point = std::pair<double, double>;
int g_table_threads = 0;

double interp_linear(const double* x_values, const double* y_values, int n, double x) {
    if (n <= 0) {
        return 0.0;
    }
    if (x <= x_values[0]) {
        return y_values[0];
    }
    if (x >= x_values[n - 1]) {
        return y_values[n - 1];
    }

    int lo = 0;
    int hi = n - 1;
    while (lo < hi - 1) {
        const int mid = (lo + hi) / 2;
        if (x_values[mid] <= x) {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    const double x0 = x_values[lo];
    const double x1 = x_values[hi];
    if (x1 <= x0) {
        return y_values[lo];
    }
    const double frac = (x - x0) / (x1 - x0);
    return y_values[lo] + frac * (y_values[hi] - y_values[lo]);
}

double activation_factor(double stage, double activation_elev, double ramp_depth) {
    const double depth = stage - activation_elev;
    if (depth <= 0.0) {
        return 0.0;
    }
    if (ramp_depth <= 0.0) {
        return 1.0;
    }
    const double frac = depth / ramp_depth;
    if (frac < 0.0) {
        return 0.0;
    }
    if (frac > 1.0) {
        return 1.0;
    }
    return frac;
}

void submerged_trapezoids_area_perimeter_xy(
    const double* x_ptr,
    const double* z_ptr,
    int n,
    double wse,
    double& out_a,
    double& out_p,
    double& out_t) {
    out_a = 0.0;
    out_p = 0.0;
    out_t = 0.0;
    if (n < 2) {
        return;
    }

    double last_wet_x = 0.0;
    bool has_last_wet_x = false;
    double top_width = 0.0;

    for (int i = 0; i < n - 1; ++i) {
        const double x0 = x_ptr[i];
        const double z0 = z_ptr[i];
        const double x1 = x_ptr[i + 1];
        const double z1 = z_ptr[i + 1];

        const double y0 = std::max(0.0, wse - z0);
        const double y1 = std::max(0.0, wse - z1);
        const double dx = x1 - x0;
        const double dz = z1 - z0;

        const bool both_submerged = (y0 > 0.0 && y1 > 0.0);
        const bool one_submerged = ((y0 > 0.0) != (y1 > 0.0));

        if (both_submerged) {
            out_a += 0.5 * (y0 + y1) * std::abs(dx);
            out_p += std::hypot(dx, dz);
            if (!has_last_wet_x) {
                last_wet_x = x0;
                has_last_wet_x = true;
            }
            top_width = x1 - last_wet_x;
        } else if (one_submerged) {
            double xi = x0;
            double zi = z0;
            if (z1 != z0) {
                xi = x0 + (x1 - x0) * ((wse - z0) / (z1 - z0));
                zi = wse;
            }

            if (y0 > 0.0) {
                out_a += 0.5 * y0 * std::abs(xi - x0);
                out_p += std::hypot(xi - x0, zi - z0);
                if (!has_last_wet_x) {
                    last_wet_x = x0;
                    has_last_wet_x = true;
                }
                top_width = xi - last_wet_x;
                has_last_wet_x = false;
            } else {
                out_a += 0.5 * y1 * std::abs(x1 - xi);
                out_p += std::hypot(x1 - xi, z1 - zi);
                last_wet_x = xi;
                has_last_wet_x = true;
                top_width = x1 - last_wet_x;
            }
        }
    }

    out_t = std::max(0.0, top_width);
}

std::vector<Point> clip_polyline_by_x_points(const std::vector<Point>& poly, double x_min, double x_max) {
    if (poly.empty()) {
        return {};
    }

    std::vector<Point> clipped;
    for (size_t i = 0; i + 1 < poly.size(); ++i) {
        const double x0 = poly[i].first;
        const double z0 = poly[i].second;
        const double x1 = poly[i + 1].first;
        const double z1 = poly[i + 1].second;

        std::vector<Point> seg;
        seg.emplace_back(x0, z0);

        if ((x0 < x_min && x_min < x1) || (x1 < x_min && x_min < x0)) {
            const double zi = (x1 != x0) ? (z0 + (z1 - z0) * ((x_min - x0) / (x1 - x0))) : z0;
            seg.emplace_back(x_min, zi);
        }
        if ((x0 < x_max && x_max < x1) || (x1 < x_max && x_max < x0)) {
            const double zi = (x1 != x0) ? (z0 + (z1 - z0) * ((x_max - x0) / (x1 - x0))) : z0;
            seg.emplace_back(x_max, zi);
        }

        seg.emplace_back(x1, z1);

        std::sort(seg.begin(), seg.end(), [](const Point& a, const Point& b) {
            return a.first < b.first;
        });

        std::vector<Point> seg_filtered;
        for (const auto& p : seg) {
            if ((x_min - 1.0e-9) <= p.first && p.first <= (x_max + 1.0e-9)) {
                seg_filtered.push_back(p);
            }
        }
        if (seg_filtered.empty()) {
            continue;
        }

        if (clipped.empty()) {
            clipped.insert(clipped.end(), seg_filtered.begin(), seg_filtered.end());
        } else {
            const Point& tail = clipped.back();
            const Point& head = seg_filtered.front();
            if (std::abs(tail.first - head.first) <= 1.0e-9 && std::abs(tail.second - head.second) <= 1.0e-9) {
                clipped.insert(clipped.end(), seg_filtered.begin() + 1, seg_filtered.end());
            } else {
                clipped.insert(clipped.end(), seg_filtered.begin(), seg_filtered.end());
            }
        }
    }

    std::vector<Point> out;
    for (const auto& p : clipped) {
        if (out.empty()) {
            out.push_back(p);
            continue;
        }
        const Point& last = out.back();
        if (std::abs(last.first - p.first) > 1.0e-9 || std::abs(last.second - p.second) > 1.0e-9) {
            out.push_back(p);
        }
    }
    return out;
}

void subsection_hydraulics_points(
    const std::vector<Point>& geom,
    double wse,
    double n_val,
    double& out_a,
    double& out_t,
    double& out_k) {
    out_a = 0.0;
    out_t = 0.0;
    out_k = 0.0;
    if (geom.size() < 2) {
        return;
    }

    std::vector<double> x_values;
    std::vector<double> z_values;
    x_values.reserve(geom.size());
    z_values.reserve(geom.size());
    for (const auto& p : geom) {
        x_values.push_back(p.first);
        z_values.push_back(p.second);
    }

    double area = 0.0;
    double perim = 0.0;
    double top_width = 0.0;
    submerged_trapezoids_area_perimeter_xy(
        x_values.data(),
        z_values.data(),
        static_cast<int>(x_values.size()),
        wse,
        area,
        perim,
        top_width);

    out_a = std::max(0.0, area);
    out_t = std::max(0.0, top_width);
    if (area > 0.0 && perim > 0.0 && n_val > 0.0) {
        const double radius = area / perim;
        out_k = (1.49 / n_val) * area * std::pow(radius, 2.0 / 3.0);
    }
}

void subsection_hydraulics_xy(
    const py::buffer_info& x_buf,
    const py::buffer_info& z_buf,
    double wse,
    double n_val,
    double& out_a,
    double& out_t,
    double& out_k) {
    out_a = 0.0;
    out_t = 0.0;
    out_k = 0.0;

    if (x_buf.ndim != 1 || z_buf.ndim != 1) {
        throw std::runtime_error("subsection_hydraulics_xy expects 1D x/z arrays.");
    }
    const int n = static_cast<int>(x_buf.shape[0]);
    if (static_cast<int>(z_buf.shape[0]) != n || n < 2) {
        return;
    }

    const auto* x_ptr = static_cast<const double*>(x_buf.ptr);
    const auto* z_ptr = static_cast<const double*>(z_buf.ptr);

    double area = 0.0;
    double perim = 0.0;
    double top_width = 0.0;
    submerged_trapezoids_area_perimeter_xy(x_ptr, z_ptr, n, wse, area, perim, top_width);

    out_a = std::max(0.0, area);
    out_t = std::max(0.0, top_width);
    if (area > 0.0 && perim > 0.0 && n_val > 0.0) {
        const double radius = area / perim;
        out_k = (1.49 / n_val) * area * std::pow(radius, 2.0 / 3.0);
    }
}

std::vector<double> solve_dense_gaussian(std::vector<double> a, std::vector<double> b, int n) {
    for (int k = 0; k < n; ++k) {
        int pivot = k;
        double max_abs = std::abs(a[k * n + k]);
        for (int i = k + 1; i < n; ++i) {
            const double v = std::abs(a[i * n + k]);
            if (v > max_abs) {
                max_abs = v;
                pivot = i;
            }
        }

        if (max_abs <= 1.0e-14) {
            throw std::runtime_error("Singular matrix in native banded solve.");
        }

        if (pivot != k) {
            for (int j = k; j < n; ++j) {
                std::swap(a[k * n + j], a[pivot * n + j]);
            }
            std::swap(b[k], b[pivot]);
        }

        const double akk = a[k * n + k];
        for (int i = k + 1; i < n; ++i) {
            const double factor = a[i * n + k] / akk;
            a[i * n + k] = 0.0;
            for (int j = k + 1; j < n; ++j) {
                a[i * n + j] -= factor * a[k * n + j];
            }
            b[i] -= factor * b[k];
        }
    }

    std::vector<double> x(static_cast<size_t>(n), 0.0);
    for (int i = n - 1; i >= 0; --i) {
        double s = b[i];
        for (int j = i + 1; j < n; ++j) {
            s -= a[i * n + j] * x[j];
        }
        x[i] = s / a[i * n + i];
    }
    return x;
}

py::array_t<double> solve_banded_full(py::array_t<double, py::array::c_style | py::array::forcecast> ab,
                                      py::array_t<double, py::array::c_style | py::array::forcecast> rhs) {
    auto ab_buf = ab.request();
    auto rhs_buf = rhs.request();

    if (ab_buf.ndim != 2) {
        throw std::runtime_error("ab must be a 2D array with shape (5, n).");
    }
    if (rhs_buf.ndim != 1) {
        throw std::runtime_error("rhs must be a 1D array with shape (n,).");
    }

    const int bands = static_cast<int>(ab_buf.shape[0]);
    const int n = static_cast<int>(ab_buf.shape[1]);
    if (bands != 5) {
        throw std::runtime_error("ab must have exactly 5 band rows for l=u=2 storage.");
    }
    if (static_cast<int>(rhs_buf.shape[0]) != n) {
        throw std::runtime_error("rhs length must match ab.shape[1].");
    }

    const auto* ab_ptr = static_cast<const double*>(ab_buf.ptr);
    const auto* rhs_ptr = static_cast<const double*>(rhs_buf.ptr);

    std::vector<double> a_dense(static_cast<size_t>(n) * static_cast<size_t>(n), 0.0);
    std::vector<double> b(static_cast<size_t>(n), 0.0);

    for (int i = 0; i < n; ++i) {
        b[i] = rhs_ptr[i];
    }

    for (int diag_offset = -2; diag_offset <= 2; ++diag_offset) {
        const int k = diag_offset;
        const int row_start = std::max(0, -k);
        const int col_start = std::max(0, k);
        const int length = n - std::abs(k);
        const int ab_row = 2 - k;

        for (int i = 0; i < length; ++i) {
            const int row = row_start + i;
            const int col = col_start + i;
            a_dense[row * n + col] = ab_ptr[ab_row * n + col];
        }
    }

    auto x = solve_dense_gaussian(std::move(a_dense), std::move(b), n);

    py::array_t<double> out(n);
    auto out_buf = out.request();
    auto* out_ptr = static_cast<double*>(out_buf.ptr);
    for (int i = 0; i < n; ++i) {
        out_ptr[i] = x[i];
    }
    return out;
}

py::tuple assemble_system_core(
    py::array_t<double, py::array::c_style | py::array::forcecast> reach_lengths,
    py::array_t<double, py::array::c_style | py::array::forcecast> z_values,
    py::array_t<double, py::array::c_style | py::array::forcecast> q_values,
    py::array_t<double, py::array::c_style | py::array::forcecast> area_values,
    py::array_t<double, py::array::c_style | py::array::forcecast> conveyance_values,
    py::array_t<double, py::array::c_style | py::array::forcecast> top_width_values,
    py::array_t<double, py::array::c_style | py::array::forcecast> velocity_values,
    py::array_t<double, py::array::c_style | py::array::forcecast> alpha_values,
    py::array_t<double, py::array::c_style | py::array::forcecast> dkdz_values,
    double dt,
    double theta,
    double q_upstream_next,
    bool ds_is_stage,
    double ds_bc_value,
    double ds_bc_ramp_factor) {
    auto reach_buf = reach_lengths.request();
    auto z_buf = z_values.request();
    auto q_buf = q_values.request();
    auto area_buf = area_values.request();
    auto conveyance_buf = conveyance_values.request();
    auto width_buf = top_width_values.request();
    auto velocity_buf = velocity_values.request();
    auto alpha_buf = alpha_values.request();
    auto dkdz_buf = dkdz_values.request();

    if (reach_buf.ndim != 1 || z_buf.ndim != 1 || q_buf.ndim != 1 || area_buf.ndim != 1 ||
        conveyance_buf.ndim != 1 || width_buf.ndim != 1 || velocity_buf.ndim != 1 ||
        alpha_buf.ndim != 1 || dkdz_buf.ndim != 1) {
        throw std::runtime_error("assemble_system_core expects 1D arrays.");
    }

    const int n = static_cast<int>(z_buf.shape[0]);
    if (n < 2) {
        throw std::runtime_error("assemble_system_core requires at least two nodes.");
    }
    if (static_cast<int>(reach_buf.shape[0]) != n - 1 ||
        static_cast<int>(q_buf.shape[0]) != n ||
        static_cast<int>(area_buf.shape[0]) != n ||
        static_cast<int>(conveyance_buf.shape[0]) != n ||
        static_cast<int>(width_buf.shape[0]) != n ||
        static_cast<int>(velocity_buf.shape[0]) != n ||
        static_cast<int>(alpha_buf.shape[0]) != n ||
        static_cast<int>(dkdz_buf.shape[0]) != n) {
        throw std::runtime_error("assemble_system_core input array lengths are inconsistent.");
    }

    const auto* reach_ptr = static_cast<const double*>(reach_buf.ptr);
    const auto* z_ptr = static_cast<const double*>(z_buf.ptr);
    const auto* q_ptr = static_cast<const double*>(q_buf.ptr);
    const auto* area_ptr = static_cast<const double*>(area_buf.ptr);
    const auto* conveyance_ptr = static_cast<const double*>(conveyance_buf.ptr);
    const auto* width_ptr = static_cast<const double*>(width_buf.ptr);
    const auto* velocity_ptr = static_cast<const double*>(velocity_buf.ptr);
    const auto* alpha_ptr = static_cast<const double*>(alpha_buf.ptr);
    const auto* dkdz_ptr = static_cast<const double*>(dkdz_buf.ptr);

    const int size = 2 * n;
    py::array_t<double> ab({5, size});
    py::array_t<double> rhs(size);
    auto ab_buf = ab.request();
    auto rhs_out_buf = rhs.request();
    auto* ab_ptr = static_cast<double*>(ab_buf.ptr);
    auto* rhs_ptr = static_cast<double*>(rhs_out_buf.ptr);

    std::fill(ab_ptr, ab_ptr + (5 * size), 0.0);
    std::fill(rhs_ptr, rhs_ptr + size, 0.0);

    ab_ptr[1 * size + 1] = 1.0;
    rhs_ptr[0] = q_upstream_next - q_ptr[0];

    for (int r = 0; r < n - 1; ++r) {
        const double z_r = z_ptr[r];
        const double q_r = q_ptr[r];
        const double z_rp1 = z_ptr[r + 1];
        const double q_rp1 = q_ptr[r + 1];
        const double length = reach_ptr[r];

        const double area_r = area_ptr[r];
        const double area_rp1 = area_ptr[r + 1];
        const double conveyance_r = conveyance_ptr[r];
        const double conveyance_rp1 = conveyance_ptr[r + 1];
        const double width_r = width_ptr[r];
        const double width_rp1 = width_ptr[r + 1];
        const double velocity_r = velocity_ptr[r];
        const double velocity_rp1 = velocity_ptr[r + 1];
        const double alpha_r = alpha_ptr[r];
        const double alpha_rp1 = alpha_ptr[r + 1];
        const double dkdz_r = dkdz_ptr[r];
        const double dkdz_rp1 = dkdz_ptr[r + 1];

        const double sf_r = (conveyance_r > 0.0) ? (q_r * std::abs(q_r)) / (conveyance_r * conveyance_r) : 0.0;
        const double sf_rp1 = (conveyance_rp1 > 0.0) ? (q_rp1 * std::abs(q_rp1)) / (conveyance_rp1 * conveyance_rp1) : 0.0;
        const double sf_avg = 0.5 * (sf_r + sf_rp1);
        const double abar = 0.5 * (area_r + area_rp1);

        const double dSf_dQ_r = (conveyance_r > 0.0) ? (2.0 * std::abs(q_r)) / (conveyance_r * conveyance_r) : 0.0;
        const double dSf_dQ_rp1 = (conveyance_rp1 > 0.0) ? (2.0 * std::abs(q_rp1)) / (conveyance_rp1 * conveyance_rp1) : 0.0;
        const double dSf_dz_r = (conveyance_r > 0.0) ? (-2.0 * q_r * std::abs(q_r) * dkdz_r) / (conveyance_r * conveyance_r * conveyance_r) : 0.0;
        const double dSf_dz_rp1 = (conveyance_rp1 > 0.0) ? (-2.0 * q_rp1 * std::abs(q_rp1) * dkdz_rp1) / (conveyance_rp1 * conveyance_rp1 * conveyance_rp1) : 0.0;

        const int row_c = 2 * r + 1;
        const double CZ_r = width_r / (2.0 * dt);
        const double CQ_r = -theta / length;
        const double CZ_rp1 = width_rp1 / (2.0 * dt);
        const double CQ_rp1 = theta / length;
        const double CB = -(q_rp1 - q_r) / length;

        ab_ptr[3 * size + (2 * r)] += CZ_r;
        ab_ptr[2 * size + (2 * r + 1)] += CQ_r;
        ab_ptr[1 * size + (2 * r + 2)] += CZ_rp1;
        ab_ptr[0 * size + (2 * r + 3)] += CQ_rp1;
        rhs_ptr[row_c] += CB;

        const int row_m = 2 * r + 2;
        const double MQ_r = (1.0 / (2.0 * dt)) - theta * alpha_r * velocity_r / length + theta * 32.174 * abar * 0.5 * dSf_dQ_r;
        const double MZ_r = theta * 32.174 * abar * 0.5 * dSf_dz_r - 32.174 * abar * theta / length;
        const double MQ_rp1 = (1.0 / (2.0 * dt)) + theta * alpha_rp1 * velocity_rp1 / length + theta * 32.174 * abar * 0.5 * dSf_dQ_rp1;
        const double MZ_rp1 = theta * 32.174 * abar * 0.5 * dSf_dz_rp1 + 32.174 * abar * theta / length;
        const double MB = -(
            (alpha_rp1 * q_rp1 * velocity_rp1 - alpha_r * q_r * velocity_r) / length
            + 32.174 * abar * (z_rp1 - z_r) / length
            + 32.174 * abar * sf_avg
        );

        ab_ptr[4 * size + (2 * r)] += MZ_r;
        ab_ptr[3 * size + (2 * r + 1)] += MQ_r;
        ab_ptr[2 * size + (2 * r + 2)] += MZ_rp1;
        ab_ptr[1 * size + (2 * r + 3)] += MQ_rp1;
        rhs_ptr[row_m] += MB;
    }

    const int row_ds = size - 1;
    if (ds_is_stage) {
        ab_ptr[3 * size + (size - 2)] = 1.0;
        rhs_ptr[row_ds] = ds_bc_ramp_factor * (ds_bc_value - z_ptr[n - 1]);
    } else {
        const double slope = std::max(ds_bc_value, 1.0e-8);
        const double k_ds = std::max(0.0, conveyance_ptr[n - 1]);
        const double dKdz_ds = dkdz_ptr[n - 1];
        const double sqrt_slope = std::sqrt(slope);
        const double q_nd = k_ds * sqrt_slope;
        const double dQdz = dKdz_ds * sqrt_slope;
        ab_ptr[2 * size + (size - 1)] = 1.0;
        ab_ptr[3 * size + (size - 2)] -= dQdz;
        rhs_ptr[row_ds] = ds_bc_ramp_factor * (q_nd - q_ptr[n - 1]);
    }

    return py::make_tuple(ab, rhs);
}

double adaptive_damping_scale(
    py::array_t<double, py::array::c_style | py::array::forcecast> bed_elevations,
    py::array_t<double, py::array::c_style | py::array::forcecast> z_iter,
    py::array_t<double, py::array::c_style | py::array::forcecast> q_iter,
    py::array_t<double, py::array::c_style | py::array::forcecast> dz_raw,
    py::array_t<double, py::array::c_style | py::array::forcecast> dq_raw,
    double wetting_depth) {
    auto bed_buf = bed_elevations.request();
    auto z_buf = z_iter.request();
    auto q_buf = q_iter.request();
    auto dz_buf = dz_raw.request();
    auto dq_buf = dq_raw.request();

    if (bed_buf.ndim != 1 || z_buf.ndim != 1 || q_buf.ndim != 1 || dz_buf.ndim != 1 || dq_buf.ndim != 1) {
        throw std::runtime_error("adaptive_damping_scale expects 1D arrays.");
    }

    const int n = static_cast<int>(bed_buf.shape[0]);
    if (static_cast<int>(z_buf.shape[0]) != n ||
        static_cast<int>(q_buf.shape[0]) != n ||
        static_cast<int>(dz_buf.shape[0]) != n ||
        static_cast<int>(dq_buf.shape[0]) != n) {
        throw std::runtime_error("adaptive_damping_scale input array lengths are inconsistent.");
    }

    const auto* bed_ptr = static_cast<const double*>(bed_buf.ptr);
    const auto* z_ptr = static_cast<const double*>(z_buf.ptr);
    const auto* q_ptr = static_cast<const double*>(q_buf.ptr);
    const auto* dz_ptr = static_cast<const double*>(dz_buf.ptr);
    const auto* dq_ptr = static_cast<const double*>(dq_buf.ptr);

    double scale = 1.0;
    for (int i = 0; i < n; ++i) {
        const double depth = std::max(0.0, z_ptr[i] - bed_ptr[i]);
        const double max_dz = std::max(0.05, std::min(0.5, 0.5 * std::max(depth, wetting_depth)));
        const double dz_abs = std::abs(dz_ptr[i]);
        if (dz_abs > max_dz && dz_abs > 0.0) {
            scale = std::min(scale, max_dz / dz_abs);
        }

        const double q_ref = std::max(20.0, std::abs(q_ptr[i]));
        const double max_dq = 0.35 * q_ref + 10.0;
        const double dq_abs = std::abs(dq_ptr[i]);
        if (dq_abs > max_dq && dq_abs > 0.0) {
            scale = std::min(scale, max_dq / dq_abs);
        }
    }

    return std::max(0.05, std::min(1.0, scale));
}

py::tuple solve_table_state(
    double z,
    double q_total,
    py::array_t<double, py::array::c_style | py::array::forcecast> z_values,
    py::array_t<double, py::array::c_style | py::array::forcecast> a_lob_raw_series,
    py::array_t<double, py::array::c_style | py::array::forcecast> t_lob_raw_series,
    py::array_t<double, py::array::c_style | py::array::forcecast> k_lob_raw_series,
    py::array_t<double, py::array::c_style | py::array::forcecast> a_ch_series,
    py::array_t<double, py::array::c_style | py::array::forcecast> t_ch_series,
    py::array_t<double, py::array::c_style | py::array::forcecast> k_ch_series,
    py::array_t<double, py::array::c_style | py::array::forcecast> a_rob_raw_series,
    py::array_t<double, py::array::c_style | py::array::forcecast> t_rob_raw_series,
    py::array_t<double, py::array::c_style | py::array::forcecast> k_rob_raw_series,
    double left_activation_elev,
    double right_activation_elev,
    double ramp_depth) {
    auto z_buf = z_values.request();
    auto a_lob_buf = a_lob_raw_series.request();
    auto t_lob_buf = t_lob_raw_series.request();
    auto k_lob_buf = k_lob_raw_series.request();
    auto a_ch_buf = a_ch_series.request();
    auto t_ch_buf = t_ch_series.request();
    auto k_ch_buf = k_ch_series.request();
    auto a_rob_buf = a_rob_raw_series.request();
    auto t_rob_buf = t_rob_raw_series.request();
    auto k_rob_buf = k_rob_raw_series.request();

    if (z_buf.ndim != 1 || a_lob_buf.ndim != 1 || t_lob_buf.ndim != 1 || k_lob_buf.ndim != 1 ||
        a_ch_buf.ndim != 1 || t_ch_buf.ndim != 1 || k_ch_buf.ndim != 1 ||
        a_rob_buf.ndim != 1 || t_rob_buf.ndim != 1 || k_rob_buf.ndim != 1) {
        throw std::runtime_error("solve_table_state expects 1D arrays.");
    }

    const int n = static_cast<int>(z_buf.shape[0]);
    if (n <= 0) {
        throw std::runtime_error("solve_table_state requires non-empty stage table.");
    }

    const auto ensure_matching_length = [n](const py::buffer_info& info) {
        if (static_cast<int>(info.shape[0]) != n) {
            throw std::runtime_error("solve_table_state array lengths must match z_values.");
        }
    };
    ensure_matching_length(a_lob_buf);
    ensure_matching_length(t_lob_buf);
    ensure_matching_length(k_lob_buf);
    ensure_matching_length(a_ch_buf);
    ensure_matching_length(t_ch_buf);
    ensure_matching_length(k_ch_buf);
    ensure_matching_length(a_rob_buf);
    ensure_matching_length(t_rob_buf);
    ensure_matching_length(k_rob_buf);

    const auto* z_ptr = static_cast<const double*>(z_buf.ptr);
    const auto* a_lob_ptr = static_cast<const double*>(a_lob_buf.ptr);
    const auto* t_lob_ptr = static_cast<const double*>(t_lob_buf.ptr);
    const auto* k_lob_ptr = static_cast<const double*>(k_lob_buf.ptr);
    const auto* a_ch_ptr = static_cast<const double*>(a_ch_buf.ptr);
    const auto* t_ch_ptr = static_cast<const double*>(t_ch_buf.ptr);
    const auto* k_ch_ptr = static_cast<const double*>(k_ch_buf.ptr);
    const auto* a_rob_ptr = static_cast<const double*>(a_rob_buf.ptr);
    const auto* t_rob_ptr = static_cast<const double*>(t_rob_buf.ptr);
    const auto* k_rob_ptr = static_cast<const double*>(k_rob_buf.ptr);

    const double a_lob_raw = interp_linear(z_ptr, a_lob_ptr, n, z);
    const double t_lob_raw = interp_linear(z_ptr, t_lob_ptr, n, z);
    const double k_lob_raw = interp_linear(z_ptr, k_lob_ptr, n, z);
    const double a_ch = interp_linear(z_ptr, a_ch_ptr, n, z);
    const double t_ch = interp_linear(z_ptr, t_ch_ptr, n, z);
    const double k_ch = interp_linear(z_ptr, k_ch_ptr, n, z);
    const double a_rob_raw = interp_linear(z_ptr, a_rob_ptr, n, z);
    const double t_rob_raw = interp_linear(z_ptr, t_rob_ptr, n, z);
    const double k_rob_raw = interp_linear(z_ptr, k_rob_ptr, n, z);

    const double left_factor = activation_factor(z, left_activation_elev, ramp_depth);
    const double right_factor = activation_factor(z, right_activation_elev, ramp_depth);

    const double a_lob = left_factor * a_lob_raw;
    const double t_lob = left_factor * t_lob_raw;
    const double k_lob = left_factor * k_lob_raw;
    const double a_rob = right_factor * a_rob_raw;
    const double t_rob = right_factor * t_rob_raw;
    const double k_rob = right_factor * k_rob_raw;

    const double a_t = a_lob + a_ch + a_rob;
    double t_t = t_lob + t_ch + t_rob;
    const double k_t = k_lob + k_ch + k_rob;

    double q_lob = 0.0;
    double q_ch = 0.0;
    double q_rob = 0.0;
    if (k_t > 0.0) {
        q_lob = q_total * (k_lob / k_t);
        q_ch = q_total * (k_ch / k_t);
        q_rob = q_total * (k_rob / k_t);
    }

    const double v_t = a_t > 0.0 ? q_total / a_t : 0.0;

    double alpha_num = 0.0;
    if (k_lob > 0.0 && a_lob > 0.0) {
        alpha_num += (k_lob * k_lob * k_lob) / (a_lob * a_lob);
    }
    if (k_ch > 0.0 && a_ch > 0.0) {
        alpha_num += (k_ch * k_ch * k_ch) / (a_ch * a_ch);
    }
    if (k_rob > 0.0 && a_rob > 0.0) {
        alpha_num += (k_rob * k_rob * k_rob) / (a_rob * a_rob);
    }

    double alpha = 1.0;
    if (k_t > 0.0 && a_t > 0.0) {
        alpha = ((a_t * a_t) * alpha_num) / (k_t * k_t * k_t);
    }

    if (t_t <= 0.0) {
        const double dz_eps = 1.0e-6;
        const double a_total_hi = interp_linear(z_ptr, a_lob_ptr, n, z + dz_eps)
                                + interp_linear(z_ptr, a_ch_ptr, n, z + dz_eps)
                                + interp_linear(z_ptr, a_rob_ptr, n, z + dz_eps);
        const double a_total_lo = interp_linear(z_ptr, a_lob_ptr, n, z - dz_eps)
                                + interp_linear(z_ptr, a_ch_ptr, n, z - dz_eps)
                                + interp_linear(z_ptr, a_rob_ptr, n, z - dz_eps);
        t_t = (a_total_hi - a_total_lo) / (2.0 * dz_eps);
        if (t_t < 0.01) {
            t_t = 0.01;
        }
    }

    return py::make_tuple(
        a_lob, a_ch, a_rob,
        t_lob, t_ch, t_rob,
        k_lob, k_ch, k_rob,
        q_lob, q_ch, q_rob,
        a_t, t_t, k_t, v_t, alpha,
        left_factor, right_factor);
}

py::tuple run_one_timestep_unsteady_1d_cpp(
    py::array_t<double, py::array::c_style | py::array::forcecast> z_n_input,
    py::array_t<double, py::array::c_style | py::array::forcecast> q_n_input,
    py::array_t<double, py::array::c_style | py::array::forcecast> reach_lengths,
    py::array_t<double, py::array::c_style | py::array::forcecast> bed_elevations,
    py::array_t<double, py::array::c_style | py::array::forcecast> area_values,
    py::array_t<double, py::array::c_style | py::array::forcecast> conveyance_values,
    py::array_t<double, py::array::c_style | py::array::forcecast> top_width_values,
    py::array_t<double, py::array::c_style | py::array::forcecast> velocity_values,
    py::array_t<double, py::array::c_style | py::array::forcecast> alpha_values,
    py::array_t<double, py::array::c_style | py::array::forcecast> dkdz_values,
    double dt,
    double theta,
    double q_upstream_next,
    bool ds_is_stage,
    double ds_bc_value,
    double ds_bc_ramp_factor,
    int max_iter,
    double tol,
    double wetting_depth) {
    auto z_buf = z_n_input.request();
    auto q_buf = q_n_input.request();
    auto bed_buf = bed_elevations.request();

    if (z_buf.ndim != 1 || q_buf.ndim != 1 || bed_buf.ndim != 1) {
        throw std::runtime_error("run_one_timestep_unsteady_1d_cpp: all input arrays must be 1D.");
    }

    const int n = static_cast<int>(z_buf.shape[0]);
    if (static_cast<int>(q_buf.shape[0]) != n || static_cast<int>(bed_buf.shape[0]) != n) {
        throw std::runtime_error("run_one_timestep_unsteady_1d_cpp: input arrays must have matching length.");
    }

    const auto* z_ptr_in = static_cast<const double*>(z_buf.ptr);
    const auto* q_ptr_in = static_cast<const double*>(q_buf.ptr);
    const auto* bed_ptr = static_cast<const double*>(bed_buf.ptr);

    // Working copies for Newton iterations
    std::vector<double> z_iter(z_ptr_in, z_ptr_in + n);
    std::vector<double> q_iter(q_ptr_in, q_ptr_in + n);

    int executed_iters = 0;
    double max_update_error = 0.0;

    for (int inner = 0; inner < max_iter; ++inner) {
        executed_iters = inner + 1;

        // Assemble system using the existing native kernel
        auto result = assemble_system_core(
            reach_lengths,
            py::array_t<double>(static_cast<size_t>(n), z_iter.data()),
            py::array_t<double>(static_cast<size_t>(n), q_iter.data()),
            area_values,
            conveyance_values,
            top_width_values,
            velocity_values,
            alpha_values,
            dkdz_values,
            dt,
            theta,
            q_upstream_next,
            ds_is_stage,
            ds_bc_value,
            ds_bc_ramp_factor
        );
        auto ab = py::cast<py::array_t<double>>(result[0]);
        auto rhs_vec = py::cast<py::array_t<double>>(result[1]);

        // Solve banded system using the existing native kernel
        auto delta = solve_banded_full(ab, rhs_vec);

        auto delta_buf = delta.request();
        const auto* delta_ptr = static_cast<const double*>(delta_buf.ptr);

        std::vector<double> dz_raw(n);
        std::vector<double> dq_raw(n);
        for (int i = 0; i < n; ++i) {
            dz_raw[i] = delta_ptr[2 * i];
            dq_raw[i] = delta_ptr[2 * i + 1];
        }

        // Compute adaptive damping using the existing native kernel
        double damping = adaptive_damping_scale(
            bed_elevations,
            py::array_t<double>(static_cast<size_t>(n), z_iter.data()),
            py::array_t<double>(static_cast<size_t>(n), q_iter.data()),
            py::array_t<double>(static_cast<size_t>(n), dz_raw.data()),
            py::array_t<double>(static_cast<size_t>(n), dq_raw.data()),
            wetting_depth
        );

        // Apply updates
        max_update_error = 0.0;
        for (int i = 0; i < n; ++i) {
            const double dz_scaled = dz_raw[i] * damping;
            const double dq_scaled = dq_raw[i] * damping;
            z_iter[i] += dz_scaled;
            q_iter[i] += dq_scaled;
            max_update_error = std::max(max_update_error, std::abs(dz_scaled));
            max_update_error = std::max(max_update_error, std::abs(dq_scaled));
        }

        // Enforce minimum depth (wetting depth)
        for (int i = 0; i < n; ++i) {
            const double bed = bed_ptr[i];
            const double z_min = bed + std::max(1e-6, wetting_depth);
            if (z_iter[i] < z_min) {
                z_iter[i] = z_min;
            }
        }

        // Check convergence
        if (max_update_error < tol) {
            break;
        }
    }

    // Pack results
    py::array_t<double> z_out(n);
    py::array_t<double> q_out(n);
    auto z_out_buf = z_out.request();
    auto q_out_buf = q_out.request();
    auto* z_out_ptr = static_cast<double*>(z_out_buf.ptr);
    auto* q_out_ptr = static_cast<double*>(q_out_buf.ptr);

    for (int i = 0; i < n; ++i) {
        z_out_ptr[i] = z_iter[i];
        q_out_ptr[i] = q_iter[i];
    }

    return py::make_tuple(
        z_out, q_out,
        static_cast<int>(executed_iters),
        static_cast<double>(max_update_error),
        executed_iters < max_iter ? true : false  // converged flag
    );
}

py::tuple build_section_hydraulic_table_cpp(
    py::array_t<double, py::array::c_style | py::array::forcecast> lob_x,
    py::array_t<double, py::array::c_style | py::array::forcecast> lob_z,
    py::array_t<double, py::array::c_style | py::array::forcecast> ch_x,
    py::array_t<double, py::array::c_style | py::array::forcecast> ch_z,
    py::array_t<double, py::array::c_style | py::array::forcecast> rob_x,
    py::array_t<double, py::array::c_style | py::array::forcecast> rob_z,
    py::array_t<double, py::array::c_style | py::array::forcecast> z_values,
    double n_lob,
    double n_ch,
    double n_rob) {
    auto lob_x_buf = lob_x.request();
    auto lob_z_buf = lob_z.request();
    auto ch_x_buf = ch_x.request();
    auto ch_z_buf = ch_z.request();
    auto rob_x_buf = rob_x.request();
    auto rob_z_buf = rob_z.request();
    auto z_values_buf = z_values.request();

    if (z_values_buf.ndim != 1) {
        throw std::runtime_error("z_values must be 1D.");
    }

    const int n_points = static_cast<int>(z_values_buf.shape[0]);
    if (n_points <= 0) {
        throw std::runtime_error("z_values must be non-empty.");
    }

    const auto* z_ptr = static_cast<const double*>(z_values_buf.ptr);

    py::array_t<double> a_lob_raw(n_points);
    py::array_t<double> t_lob_raw(n_points);
    py::array_t<double> k_lob_raw(n_points);
    py::array_t<double> a_ch(n_points);
    py::array_t<double> t_ch(n_points);
    py::array_t<double> k_ch(n_points);
    py::array_t<double> a_rob_raw(n_points);
    py::array_t<double> t_rob_raw(n_points);
    py::array_t<double> k_rob_raw(n_points);
    py::array_t<double> k_total_raw(n_points);
    py::array_t<double> dk_dz_raw(n_points);

    auto a_lob_buf = a_lob_raw.request();
    auto t_lob_buf = t_lob_raw.request();
    auto k_lob_buf = k_lob_raw.request();
    auto a_ch_buf = a_ch.request();
    auto t_ch_buf = t_ch.request();
    auto k_ch_buf = k_ch.request();
    auto a_rob_buf = a_rob_raw.request();
    auto t_rob_buf = t_rob_raw.request();
    auto k_rob_buf = k_rob_raw.request();
    auto k_total_buf = k_total_raw.request();
    auto dk_dz_buf = dk_dz_raw.request();

    auto* a_lob_ptr = static_cast<double*>(a_lob_buf.ptr);
    auto* t_lob_ptr = static_cast<double*>(t_lob_buf.ptr);
    auto* k_lob_ptr = static_cast<double*>(k_lob_buf.ptr);
    auto* a_ch_ptr = static_cast<double*>(a_ch_buf.ptr);
    auto* t_ch_ptr = static_cast<double*>(t_ch_buf.ptr);
    auto* k_ch_ptr = static_cast<double*>(k_ch_buf.ptr);
    auto* a_rob_ptr = static_cast<double*>(a_rob_buf.ptr);
    auto* t_rob_ptr = static_cast<double*>(t_rob_buf.ptr);
    auto* k_rob_ptr = static_cast<double*>(k_rob_buf.ptr);
    auto* k_total_ptr = static_cast<double*>(k_total_buf.ptr);
    auto* dk_dz_ptr = static_cast<double*>(dk_dz_buf.ptr);

    #if defined(BACKWATER_HAS_OPENMP)
    if (g_table_threads > 0) {
        omp_set_num_threads(g_table_threads);
    }
    #pragma omp parallel for if(n_points >= 64)
    #endif
    for (int i = 0; i < n_points; ++i) {
        const double z_eval = z_ptr[i];

        subsection_hydraulics_xy(lob_x_buf, lob_z_buf, z_eval, n_lob, a_lob_ptr[i], t_lob_ptr[i], k_lob_ptr[i]);
        subsection_hydraulics_xy(ch_x_buf, ch_z_buf, z_eval, n_ch, a_ch_ptr[i], t_ch_ptr[i], k_ch_ptr[i]);
        subsection_hydraulics_xy(rob_x_buf, rob_z_buf, z_eval, n_rob, a_rob_ptr[i], t_rob_ptr[i], k_rob_ptr[i]);

        k_total_ptr[i] = k_lob_ptr[i] + k_ch_ptr[i] + k_rob_ptr[i];
    }

    if (n_points == 1) {
        dk_dz_ptr[0] = 0.0;
    } else if (n_points == 2) {
        const double dz = z_ptr[1] - z_ptr[0];
        const double g = (dz != 0.0) ? (k_total_ptr[1] - k_total_ptr[0]) / dz : 0.0;
        dk_dz_ptr[0] = g;
        dk_dz_ptr[1] = g;
    } else {
        for (int i = 1; i < n_points - 1; ++i) {
            const double dz = z_ptr[i + 1] - z_ptr[i - 1];
            dk_dz_ptr[i] = (dz != 0.0) ? (k_total_ptr[i + 1] - k_total_ptr[i - 1]) / dz : 0.0;
        }

        // Match numpy.gradient(..., edge_order=2) one-sided 3-point stencils.
        {
            const double h0 = z_ptr[1] - z_ptr[0];
            const double h1 = z_ptr[2] - z_ptr[1];
            if (h0 != 0.0 && h1 != 0.0 && (h0 + h1) != 0.0) {
                const double a = -(2.0 * h0 + h1) / (h0 * (h0 + h1));
                const double b = (h0 + h1) / (h0 * h1);
                const double c = -h0 / (h1 * (h0 + h1));
                dk_dz_ptr[0] = a * k_total_ptr[0] + b * k_total_ptr[1] + c * k_total_ptr[2];
            } else {
                dk_dz_ptr[0] = 0.0;
            }
        }
        {
            const int n = n_points;
            const double hm1 = z_ptr[n - 1] - z_ptr[n - 2];
            const double hm2 = z_ptr[n - 2] - z_ptr[n - 3];
            if (hm1 != 0.0 && hm2 != 0.0 && (hm1 + hm2) != 0.0) {
                const double a = hm1 / (hm2 * (hm1 + hm2));
                const double b = -(hm1 + hm2) / (hm1 * hm2);
                const double c = (2.0 * hm1 + hm2) / (hm1 * (hm1 + hm2));
                dk_dz_ptr[n - 1] = a * k_total_ptr[n - 3] + b * k_total_ptr[n - 2] + c * k_total_ptr[n - 1];
            } else {
                dk_dz_ptr[n - 1] = 0.0;
            }
        }
    }

    return py::make_tuple(
        a_lob_raw,
        t_lob_raw,
        k_lob_raw,
        a_ch,
        t_ch,
        k_ch,
        a_rob_raw,
        t_rob_raw,
        k_rob_raw,
        k_total_raw,
        dk_dz_raw
    );
}

py::tuple build_section_hydraulic_table_from_geometry_cpp(
    py::array_t<double, py::array::c_style | py::array::forcecast> geom_x,
    py::array_t<double, py::array::c_style | py::array::forcecast> geom_z,
    double left_bank_station,
    double right_bank_station,
    py::array_t<double, py::array::c_style | py::array::forcecast> z_values,
    double n_lob,
    double n_ch,
    double n_rob) {
    auto geom_x_buf = geom_x.request();
    auto geom_z_buf = geom_z.request();
    auto z_values_buf = z_values.request();

    if (geom_x_buf.ndim != 1 || geom_z_buf.ndim != 1 || z_values_buf.ndim != 1) {
        throw std::runtime_error("build_section_hydraulic_table_from_geometry_cpp expects 1D arrays.");
    }

    const int n_geom = static_cast<int>(geom_x_buf.shape[0]);
    const int n_points = static_cast<int>(z_values_buf.shape[0]);
    if (n_geom < 2 || static_cast<int>(geom_z_buf.shape[0]) != n_geom) {
        throw std::runtime_error("geometry arrays must have matching length >= 2.");
    }
    if (n_points <= 0) {
        throw std::runtime_error("z_values must be non-empty.");
    }

    const auto* gx_ptr = static_cast<const double*>(geom_x_buf.ptr);
    const auto* gz_ptr = static_cast<const double*>(geom_z_buf.ptr);
    const auto* z_ptr = static_cast<const double*>(z_values_buf.ptr);

    std::vector<Point> geom;
    geom.reserve(static_cast<size_t>(n_geom));
    for (int i = 0; i < n_geom; ++i) {
        geom.emplace_back(gx_ptr[i], gz_ptr[i]);
    }
    std::sort(geom.begin(), geom.end(), [](const Point& a, const Point& b) {
        return a.first < b.first;
    });

    const double min_x = geom.front().first;
    const double max_x = geom.back().first;

    const std::vector<Point> lob_geom = clip_polyline_by_x_points(geom, min_x, left_bank_station);
    const std::vector<Point> ch_geom = clip_polyline_by_x_points(geom, left_bank_station, right_bank_station);
    const std::vector<Point> rob_geom = clip_polyline_by_x_points(geom, right_bank_station, max_x);

    py::array_t<double> a_lob_raw(n_points);
    py::array_t<double> t_lob_raw(n_points);
    py::array_t<double> k_lob_raw(n_points);
    py::array_t<double> a_ch(n_points);
    py::array_t<double> t_ch(n_points);
    py::array_t<double> k_ch(n_points);
    py::array_t<double> a_rob_raw(n_points);
    py::array_t<double> t_rob_raw(n_points);
    py::array_t<double> k_rob_raw(n_points);
    py::array_t<double> k_total_raw(n_points);
    py::array_t<double> dk_dz_raw(n_points);

    auto a_lob_buf = a_lob_raw.request();
    auto t_lob_buf = t_lob_raw.request();
    auto k_lob_buf = k_lob_raw.request();
    auto a_ch_buf = a_ch.request();
    auto t_ch_buf = t_ch.request();
    auto k_ch_buf = k_ch.request();
    auto a_rob_buf = a_rob_raw.request();
    auto t_rob_buf = t_rob_raw.request();
    auto k_rob_buf = k_rob_raw.request();
    auto k_total_buf = k_total_raw.request();
    auto dk_dz_buf = dk_dz_raw.request();

    auto* a_lob_ptr = static_cast<double*>(a_lob_buf.ptr);
    auto* t_lob_ptr = static_cast<double*>(t_lob_buf.ptr);
    auto* k_lob_ptr = static_cast<double*>(k_lob_buf.ptr);
    auto* a_ch_ptr = static_cast<double*>(a_ch_buf.ptr);
    auto* t_ch_ptr = static_cast<double*>(t_ch_buf.ptr);
    auto* k_ch_ptr = static_cast<double*>(k_ch_buf.ptr);
    auto* a_rob_ptr = static_cast<double*>(a_rob_buf.ptr);
    auto* t_rob_ptr = static_cast<double*>(t_rob_buf.ptr);
    auto* k_rob_ptr = static_cast<double*>(k_rob_buf.ptr);
    auto* k_total_ptr = static_cast<double*>(k_total_buf.ptr);
    auto* dk_dz_ptr = static_cast<double*>(dk_dz_buf.ptr);

    #if defined(BACKWATER_HAS_OPENMP)
    if (g_table_threads > 0) {
        omp_set_num_threads(g_table_threads);
    }
    #pragma omp parallel for if(n_points >= 64)
    #endif
    for (int i = 0; i < n_points; ++i) {
        const double z_eval = z_ptr[i];

        subsection_hydraulics_points(lob_geom, z_eval, n_lob, a_lob_ptr[i], t_lob_ptr[i], k_lob_ptr[i]);
        subsection_hydraulics_points(ch_geom, z_eval, n_ch, a_ch_ptr[i], t_ch_ptr[i], k_ch_ptr[i]);
        subsection_hydraulics_points(rob_geom, z_eval, n_rob, a_rob_ptr[i], t_rob_ptr[i], k_rob_ptr[i]);

        k_total_ptr[i] = k_lob_ptr[i] + k_ch_ptr[i] + k_rob_ptr[i];
    }

    if (n_points == 1) {
        dk_dz_ptr[0] = 0.0;
    } else if (n_points == 2) {
        const double dz = z_ptr[1] - z_ptr[0];
        const double g = (dz != 0.0) ? (k_total_ptr[1] - k_total_ptr[0]) / dz : 0.0;
        dk_dz_ptr[0] = g;
        dk_dz_ptr[1] = g;
    } else {
        for (int i = 1; i < n_points - 1; ++i) {
            const double dz = z_ptr[i + 1] - z_ptr[i - 1];
            dk_dz_ptr[i] = (dz != 0.0) ? (k_total_ptr[i + 1] - k_total_ptr[i - 1]) / dz : 0.0;
        }

        {
            const double h0 = z_ptr[1] - z_ptr[0];
            const double h1 = z_ptr[2] - z_ptr[1];
            if (h0 != 0.0 && h1 != 0.0 && (h0 + h1) != 0.0) {
                const double a = -(2.0 * h0 + h1) / (h0 * (h0 + h1));
                const double b = (h0 + h1) / (h0 * h1);
                const double c = -h0 / (h1 * (h0 + h1));
                dk_dz_ptr[0] = a * k_total_ptr[0] + b * k_total_ptr[1] + c * k_total_ptr[2];
            } else {
                dk_dz_ptr[0] = 0.0;
            }
        }
        {
            const int n = n_points;
            const double hm1 = z_ptr[n - 1] - z_ptr[n - 2];
            const double hm2 = z_ptr[n - 2] - z_ptr[n - 3];
            if (hm1 != 0.0 && hm2 != 0.0 && (hm1 + hm2) != 0.0) {
                const double a = hm1 / (hm2 * (hm1 + hm2));
                const double b = -(hm1 + hm2) / (hm1 * hm2);
                const double c = (2.0 * hm1 + hm2) / (hm1 * (hm1 + hm2));
                dk_dz_ptr[n - 1] = a * k_total_ptr[n - 3] + b * k_total_ptr[n - 2] + c * k_total_ptr[n - 1];
            } else {
                dk_dz_ptr[n - 1] = 0.0;
            }
        }
    }

    return py::make_tuple(
        a_lob_raw,
        t_lob_raw,
        k_lob_raw,
        a_ch,
        t_ch,
        k_ch,
        a_rob_raw,
        t_rob_raw,
        k_rob_raw,
        k_total_raw,
        dk_dz_raw
    );
}

void configure_table_threads_cpp(int thread_count) {
    g_table_threads = std::max(0, thread_count);
#if defined(BACKWATER_HAS_OPENMP)
    if (g_table_threads > 0) {
        omp_set_num_threads(g_table_threads);
    }
#endif
}

int get_table_threads_cpp() {
    return g_table_threads;
}

}  // namespace

PYBIND11_MODULE(backwater_native, m) {
    m.doc() = "Native acceleration helpers for qgis-backwater-plugin";
    m.def("assemble_system_core", &assemble_system_core,
          py::arg("reach_lengths"), py::arg("z_values"), py::arg("q_values"),
          py::arg("area_values"), py::arg("conveyance_values"), py::arg("top_width_values"),
          py::arg("velocity_values"), py::arg("alpha_values"), py::arg("dkdz_values"),
          py::arg("dt"), py::arg("theta"), py::arg("q_upstream_next"),
          py::arg("ds_is_stage"), py::arg("ds_bc_value"), py::arg("ds_bc_ramp_factor"),
          "Assemble the unsteady solver banded matrix and RHS from precomputed node scalars.");
        m.def("adaptive_damping_scale", &adaptive_damping_scale,
            py::arg("bed_elevations"), py::arg("z_iter"), py::arg("q_iter"), py::arg("dz_raw"), py::arg("dq_raw"), py::arg("wetting_depth"),
            "Compute the scalar adaptive damping factor for one Newton update.");
    m.def("solve_banded_full", &solve_banded_full,
          py::arg("ab"), py::arg("rhs"),
          "Solve l=u=2 banded linear system from scipy-style 5-row storage.");
    m.def("solve_table_state", &solve_table_state,
          py::arg("z"), py::arg("q_total"), py::arg("z_values"),
          py::arg("a_lob_raw_series"), py::arg("t_lob_raw_series"), py::arg("k_lob_raw_series"),
          py::arg("a_ch_series"), py::arg("t_ch_series"), py::arg("k_ch_series"),
          py::arg("a_rob_raw_series"), py::arg("t_rob_raw_series"), py::arg("k_rob_raw_series"),
          py::arg("left_activation_elev"), py::arg("right_activation_elev"), py::arg("ramp_depth"),
          "Compute interpolated subsection hydraulic state from a precomputed stage table.");
    m.def("run_one_timestep_unsteady_1d_cpp", &run_one_timestep_unsteady_1d_cpp,
          py::arg("z_n_input"), py::arg("q_n_input"), py::arg("reach_lengths"),
          py::arg("bed_elevations"), py::arg("area_values"), py::arg("conveyance_values"),
          py::arg("top_width_values"), py::arg("velocity_values"), py::arg("alpha_values"),
          py::arg("dkdz_values"), py::arg("dt"), py::arg("theta"), py::arg("q_upstream_next"),
          py::arg("ds_is_stage"), py::arg("ds_bc_value"), py::arg("ds_bc_ramp_factor"),
          py::arg("max_iter"), py::arg("tol"), py::arg("wetting_depth"),
          "Run one complete Newton iteration loop for a single timestep.");
        m.def("build_section_hydraulic_table_cpp", &build_section_hydraulic_table_cpp,
            py::arg("lob_x"), py::arg("lob_z"), py::arg("ch_x"), py::arg("ch_z"),
            py::arg("rob_x"), py::arg("rob_z"), py::arg("z_values"),
            py::arg("n_lob"), py::arg("n_ch"), py::arg("n_rob"),
            "Build subsection hydraulic table arrays for one cross section from subsection geometry.");
            m.def("build_section_hydraulic_table_from_geometry_cpp", &build_section_hydraulic_table_from_geometry_cpp,
                py::arg("geom_x"), py::arg("geom_z"), py::arg("left_bank_station"), py::arg("right_bank_station"),
                py::arg("z_values"), py::arg("n_lob"), py::arg("n_ch"), py::arg("n_rob"),
                "Clip subsection geometry and build hydraulic table arrays for one cross section.");
                m.def("configure_table_threads_cpp", &configure_table_threads_cpp,
                    py::arg("thread_count"),
                    "Configure OpenMP thread count for native hydraulic-table kernels (0 uses runtime default).");
                m.def("get_table_threads_cpp", &get_table_threads_cpp,
                    "Get configured native hydraulic-table OpenMP thread count (0 means runtime default).");
}
