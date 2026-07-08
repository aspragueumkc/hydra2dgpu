// swe2d_bindings.cpp
// pybind11 module: backwater_swe2d
//
// Exposes the 2D SWE hybrid GPU/CPU solver to Python as an opaque capsule-based API.
// Python users interact through swe2d_backend.py which wraps this module.
//
// UNIT CONVENTION: The kernel receives geometry in model units (feet or meters).
// Weir, orifice, bridge, and pump formulas are unit-agnostic — they produce
// correct results in whatever units the inputs are in, as long as the gravity
// parameter matches.  Only the HDS-5 culvert tables require USC internally;
// the culvert path converts geometry to feet, computes in USC, then converts
// the result back to model units using the caller-supplied model_to_ft factor.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda_runtime_api.h>

#include "swe2d_mesh.hpp"
#include "swe2d_solver.hpp"
#include "swe2d_units.cuh"

#ifdef HYDRA_HAS_CUDA
#include "swe2d_gpu.cuh"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

// ─────────────────────────────────────────────────────────────────────────────
// Helper: require a C-contiguous numpy array of given dtype
// ─────────────────────────────────────────────────────────────────────────────
template <typename T>
static const T* require_array(
    const py::array_t<T, py::array::c_style | py::array::forcecast>& arr,
    py::ssize_t expected_size,
    const char* name)
{
    if (arr.size() != expected_size) {
        throw std::invalid_argument(
            std::string(name) + ": expected size " + std::to_string(expected_size)
            + " but got " + std::to_string(arr.size()));
    }
    return arr.data();
}

static inline double bw2d_clamp(double v, double lo, double hi)
{
    return std::max(lo, std::min(hi, v));
}

static inline double bw2d_circular_area(double diameter)
{
    const double d = std::max(0.0, diameter);
    return 0.25 * M_PI * d * d;
}

static inline double bw2d_circular_perimeter_full(double diameter)
{
    return M_PI * std::max(0.0, diameter);
}

static inline double bw2d_equiv_diameter_from_area(double area)
{
    const double a = std::max(0.0, area);
    return (a > 0.0) ? std::sqrt(4.0 * a / M_PI) : 0.0;
}

static inline double bw2d_pipe_manning_capacity_full(double diameter, double slope, double roughness_n)
{
    const double d = std::max(0.0, diameter);
    const double s = std::max(0.0, slope);
    const double n = std::max(1.0e-6, roughness_n);
    if (d <= 0.0 || s <= 0.0) return 0.0;
    const double area = bw2d_circular_area(d);
    const double wetted_perimeter = bw2d_circular_perimeter_full(d);
    if (wetted_perimeter <= 0.0) return 0.0;
    const double rh = area / wetted_perimeter;
    return (1.486 / n) * area * std::pow(rh, 2.0 / 3.0) * std::sqrt(s);
}

static inline double bw2d_rect_manning_capacity_full(double width, double height, double slope, double roughness_n)
{
    const double w = std::max(0.0, width);
    const double h = std::max(0.0, height);
    const double s = std::max(0.0, slope);
    const double n = std::max(1.0e-6, roughness_n);
    if (w <= 0.0 || h <= 0.0 || s <= 0.0) return 0.0;
    const double area = w * h;
    const double perim = 2.0 * (w + h);
    if (perim <= 0.0) return 0.0;
    const double rh = area / perim;
    return (1.486 / n) * area * std::pow(rh, 2.0 / 3.0) * std::sqrt(s);
}

static inline double bw2d_orifice_q(double head_up, double head_down, double area, double cd, double g)
{
    const double a = std::max(0.0, area);
    const double dh = head_up - head_down;
    if (a <= 0.0 || std::abs(dh) <= 1.0e-12) return 0.0;
    const double q = cd * a * std::sqrt(std::max(0.0, 2.0 * g * std::abs(dh)));
    return (dh >= 0.0) ? q : -q;
}

static inline double bw2d_weir_q(double upstream_wse, double downstream_wse, double crest_elev, double width, double coeff)
{
    const double b = std::max(0.0, width);
    if (b <= 0.0) return 0.0;
    const double hup = std::max(0.0, upstream_wse - crest_elev);
    const double hdn = std::max(0.0, downstream_wse - crest_elev);
    if (hup <= 0.0 && hdn <= 0.0) return 0.0;
    if (upstream_wse >= downstream_wse) {
        return coeff * b * std::pow(hup, 1.5);
    }
    return -coeff * b * std::pow(hdn, 1.5);
}

namespace {

constexpr int BW2D_FORM = 0;
constexpr int BW2D_K = 1;
constexpr int BW2D_M = 2;
constexpr int BW2D_C = 3;
constexpr int BW2D_Y = 4;
constexpr int BW2D_MAX_CULVERT_CODE = 57;
// BW2D_GRAVITY is provided by swe2d_units.cuh (USC gravity for HDS-5 culvert tables)
constexpr double BW2D_BIG = 1.0e20;

static const std::array<std::array<double, 5>, 58> BW2D_CULVERT_PARAMS = {{
    {{0.0, 0.0, 0.0, 0.0, 0.00}},
    {{1.0, 0.0098, 2.00, 0.0398, 0.67}}, {{1.0, 0.0018, 2.00, 0.0292, 0.74}}, {{1.0, 0.0045, 2.00, 0.0317, 0.69}},
    {{1.0, 0.0078, 2.00, 0.0379, 0.69}}, {{1.0, 0.0210, 1.33, 0.0463, 0.75}}, {{1.0, 0.0340, 1.50, 0.0553, 0.54}},
    {{1.0, 0.0018, 2.50, 0.0300, 0.74}}, {{1.0, 0.0018, 2.50, 0.0243, 0.83}},
    {{1.0, 0.026, 1.0, 0.0347, 0.81}}, {{1.0, 0.061, 0.75, 0.0400, 0.80}}, {{1.0, 0.061, 0.75, 0.0423, 0.82}},
    {{2.0, 0.510, 0.667, 0.0309, 0.80}}, {{2.0, 0.486, 0.667, 0.0249, 0.83}},
    {{2.0, 0.515, 0.667, 0.0375, 0.79}}, {{2.0, 0.495, 0.667, 0.0314, 0.82}}, {{2.0, 0.486, 0.667, 0.0252, 0.865}},
    {{2.0, 0.545, 0.667, 0.04505, 0.73}}, {{2.0, 0.533, 0.667, 0.0425, 0.705}}, {{2.0, 0.522, 0.667, 0.0402, 0.68}}, {{2.0, 0.498, 0.667, 0.0327, 0.75}},
    {{2.0, 0.497, 0.667, 0.0339, 0.803}}, {{2.0, 0.493, 0.667, 0.0361, 0.806}}, {{2.0, 0.495, 0.667, 0.0386, 0.71}},
    {{2.0, 0.497, 0.667, 0.0302, 0.835}}, {{2.0, 0.495, 0.667, 0.0252, 0.881}}, {{2.0, 0.493, 0.667, 0.0227, 0.887}},
    {{1.0, 0.0083, 2.00, 0.0379, 0.69}}, {{1.0, 0.0145, 1.75, 0.0419, 0.64}}, {{1.0, 0.0340, 1.50, 0.0496, 0.57}},
    {{1.0, 0.0100, 2.00, 0.0398, 0.67}}, {{1.0, 0.0018, 2.50, 0.0292, 0.74}}, {{1.0, 0.0045, 2.00, 0.0317, 0.69}},
    {{1.0, 0.0100, 2.00, 0.0398, 0.67}}, {{1.0, 0.0018, 2.50, 0.0292, 0.74}}, {{1.0, 0.0095, 2.00, 0.0317, 0.69}},
    {{1.0, 0.0083, 2.00, 0.0379, 0.69}}, {{1.0, 0.0300, 1.00, 0.0463, 0.75}}, {{1.0, 0.0340, 1.50, 0.0496, 0.57}},
    {{1.0, 0.0300, 1.50, 0.0496, 0.57}}, {{1.0, 0.0088, 2.00, 0.0368, 0.68}}, {{1.0, 0.0030, 2.00, 0.0269, 0.77}},
    {{1.0, 0.0300, 1.50, 0.0496, 0.57}}, {{1.0, 0.0088, 2.00, 0.0368, 0.68}}, {{1.0, 0.0030, 2.00, 0.0269, 0.77}},
    {{1.0, 0.0083, 2.00, 0.0379, 0.69}}, {{1.0, 0.0300, 1.00, 0.0463, 0.75}}, {{1.0, 0.0340, 1.50, 0.0496, 0.57}},
    {{2.0, 0.534, 0.555, 0.0196, 0.90}}, {{2.0, 0.519, 0.640, 0.0210, 0.90}},
    {{2.0, 0.536, 0.622, 0.0368, 0.83}}, {{2.0, 0.5035, 0.719, 0.0478, 0.80}}, {{2.0, 0.547, 0.800, 0.0598, 0.75}},
    {{2.0, 0.475, 0.667, 0.0179, 0.97}},
    {{2.0, 0.560, 0.667, 0.0446, 0.85}}, {{2.0, 0.560, 0.667, 0.0378, 0.87}},
    {{2.0, 0.500, 0.667, 0.0446, 0.65}}, {{2.0, 0.500, 0.667, 0.0378, 0.71}}
}};

struct Bw2dXsect {
    int code = 1;
    bool rectangular = false;
    double y_full = 0.0;
    double a_full = 0.0;
    double radius = 0.0;
    double width = 0.0;
};

struct Bw2dCulvert {
    double y_full = 0.0;
    double scf = 0.0;
    double d_q_d_h = 0.0;
    double q_critical = 0.0;
    double kk = 0.0;
    double mm = 0.0;
    double ad = 0.0;
    double h_plus = 0.0;
    const Bw2dXsect* xsect = nullptr;
};

static inline double bw2d_xsect_area(const Bw2dXsect& x, double y)
{
    if (x.rectangular) {
        const double yy = bw2d_clamp(y, 0.0, x.y_full);
        return x.width * yy;
    }
    const double yy = bw2d_clamp(y, 0.0, 2.0 * x.radius);
    if (yy <= 0.0) return 0.0;
    const double arg = bw2d_clamp((x.radius - yy) / x.radius, -1.0, 1.0);
    const double theta = 2.0 * std::acos(arg);
    return 0.5 * x.radius * x.radius * (theta - std::sin(theta));
}

static inline double bw2d_xsect_top_width(const Bw2dXsect& x, double y)
{
    if (x.rectangular) return (y > 0.0) ? x.width : 0.0;
    const double yy = bw2d_clamp(y, 0.0, 2.0 * x.radius);
    if (yy <= 0.0) return 0.0;
    return 2.0 * std::sqrt(std::max(0.0, 2.0 * x.radius * yy - yy * yy));
}

static inline double bw2d_xsect_wetted_perimeter(const Bw2dXsect& x, double y)
{
    if (x.rectangular) {
        const double yy = bw2d_clamp(y, 0.0, x.y_full);
        if (yy <= 0.0) return 0.0;
        return x.width + 2.0 * yy;
    }
    const double yy = bw2d_clamp(y, 0.0, 2.0 * x.radius);
    if (yy <= 0.0) return 0.0;
    const double arg = bw2d_clamp((x.radius - yy) / x.radius, -1.0, 1.0);
    const double theta = 2.0 * std::acos(arg);
    return x.radius * theta;
}

static inline double bw2d_xsect_hydraulic_radius(const Bw2dXsect& x, double y)
{
    const double area = bw2d_xsect_area(x, y);
    const double perimeter = bw2d_xsect_wetted_perimeter(x, y);
    if (area <= 0.0 || perimeter <= 0.0) return 0.0;
    return area / perimeter;
}

template <typename Func>
static bool bw2d_ridder(Func&& f, double a, double b, double tol, int max_iter, double& root)
{
    double fa = f(a);
    double fb = f(b);
    if (!std::isfinite(fa) || !std::isfinite(fb) || fa * fb > 0.0) return false;
    if (fa == 0.0) {
        root = a;
        return true;
    }
    if (fb == 0.0) {
        root = b;
        return true;
    }

    for (int it = 0; it < max_iter; ++it) {
        const double m = 0.5 * (a + b);
        const double fm = f(m);
        const double s_sq = fm * fm - fa * fb;
        if (s_sq <= 0.0 || !std::isfinite(s_sq)) {
            if (fa * fm < 0.0) {
                b = m;
                fb = fm;
            } else {
                a = m;
                fa = fm;
            }
            if (std::abs(b - a) < tol) {
                root = 0.5 * (a + b);
                return true;
            }
            continue;
        }
        const double s = std::sqrt(s_sq);
        const double sign = ((fa - fb) < 0.0) ? -1.0 : 1.0;
        const double x = m + ((m - a) * fm / s) * sign;
        const double fx = f(x);
        if (!std::isfinite(fx)) return false;
        if (std::abs(fx) < tol) {
            root = x;
            return true;
        }

        if (fm * fx < 0.0) {
            a = m;
            fa = fm;
            b = x;
            fb = fx;
        } else if (fa * fx < 0.0) {
            b = x;
            fb = fx;
        } else {
            a = x;
            fa = fx;
        }

        if (std::abs(b - a) < tol) {
            root = 0.5 * (a + b);
            return true;
        }
    }
    root = 0.5 * (a + b);
    return true;
}

static double bw2d_form1_eqn(double yc, Bw2dCulvert& culvert)
{
    const double ac = bw2d_xsect_area(*culvert.xsect, yc);
    const double wc = bw2d_xsect_top_width(*culvert.xsect, yc);
    const double yh = (wc > 0.0) ? (ac / wc) : 0.0;
    culvert.q_critical = ac * std::sqrt(BW2D_GRAVITY * yh);
    return culvert.h_plus - yc / culvert.y_full - yh / (2.0 * culvert.y_full)
        - culvert.kk * std::pow(culvert.q_critical / culvert.ad, culvert.mm);
}

static double bw2d_get_form1_flow(double h, Bw2dCulvert& culvert)
{
    culvert.h_plus = h / culvert.y_full + culvert.scf;
    double a = std::max(1.0e-6, 0.01 * h);
    double b = std::max(a * 1.01, h);
    auto f = [&](double yc) { return bw2d_form1_eqn(yc, culvert); };

    double fa = f(a);
    double fb = f(b);
    if (!(fa == 0.0 || fb == 0.0 || fa * fb < 0.0)) {
        for (int k = 1; k <= 40; ++k) {
            const double x = a + (b - a) * (static_cast<double>(k) / 41.0);
            const double fx = f(x);
            if (fa * fx < 0.0) {
                b = x;
                fb = fx;
                break;
            }
            if (fx * fb < 0.0) {
                a = x;
                fa = fx;
                break;
            }
        }
    }
    if (!(fa == 0.0 || fb == 0.0 || fa * fb < 0.0)) {
        for (int k = 0; k < 10; ++k) {
            b *= 2.0;
            fb = f(b);
            if (fa * fb < 0.0) break;
        }
    }

    double yc = 0.5 * (a + b);
    double root = yc;
    if (bw2d_ridder(f, a, b, 1.0e-3, 100, root)) {
        yc = root;
    }
    (void)bw2d_form1_eqn(yc, culvert);
    return culvert.q_critical;
}

static double bw2d_get_unsubmerged_flow(int code, double h, Bw2dCulvert& culvert)
{
    culvert.kk = BW2D_CULVERT_PARAMS[code][BW2D_K];
    culvert.mm = BW2D_CULVERT_PARAMS[code][BW2D_M];
    const double arg = h / culvert.y_full / culvert.kk;
    double q = 0.0;
    if (BW2D_CULVERT_PARAMS[code][BW2D_FORM] == 1.0) {
        q = bw2d_get_form1_flow(h, culvert);
    } else {
        q = culvert.ad * std::pow(arg, 1.0 / culvert.mm);
    }
    culvert.d_q_d_h = (q / std::max(h, 1.0e-12)) / culvert.mm;
    return q;
}

static double bw2d_get_submerged_flow(int code, double h, Bw2dCulvert& culvert)
{
    const double cc = BW2D_CULVERT_PARAMS[code][BW2D_C];
    const double yy = BW2D_CULVERT_PARAMS[code][BW2D_Y];
    const double arg = (h / culvert.y_full - yy + culvert.scf) / cc;
    if (arg <= 0.0) {
        culvert.d_q_d_h = 0.0;
        return BW2D_BIG;
    }
    const double q = std::sqrt(arg) * culvert.ad;
    culvert.d_q_d_h = 0.5 * q / arg / culvert.y_full / cc;
    return q;
}

static double bw2d_get_transition_flow(int code, double h, double h1, double h2, Bw2dCulvert& culvert)
{
    const double q1 = bw2d_get_unsubmerged_flow(code, h1, culvert);
    const double q2 = bw2d_get_submerged_flow(code, h2, culvert);
    const double q = q1 + (q2 - q1) * (h - h1) / (h2 - h1);
    culvert.d_q_d_h = (q2 - q1) / (h2 - h1);
    return q;
}

static double bw2d_inlet_controlled_flow(const Bw2dXsect& xsect, double slope, double h, double* d_q_d_h_out)
{
    const int code = bw2d_clamp(static_cast<double>(xsect.code), 1.0, static_cast<double>(BW2D_MAX_CULVERT_CODE));
    Bw2dCulvert culvert;
    culvert.y_full = xsect.y_full;
    culvert.ad = xsect.a_full * std::sqrt(std::max(1.0e-12, xsect.y_full));
    culvert.xsect = &xsect;

    if (code == 5 || code == 37 || code == 46) {
        culvert.scf = -7.0 * slope;
    } else {
        culvert.scf = 0.5 * slope;
    }

    const double y = std::max(0.0, h);
    const double y2 = culvert.y_full * (16.0 * BW2D_CULVERT_PARAMS[code][BW2D_C] + BW2D_CULVERT_PARAMS[code][BW2D_Y] - culvert.scf);
    double q = 0.0;
    if (y >= y2) {
        q = bw2d_get_submerged_flow(code, y, culvert);
    } else {
        const double y1 = 0.95 * culvert.y_full;
        if (y <= y1) {
            q = bw2d_get_unsubmerged_flow(code, y, culvert);
        } else {
            q = bw2d_get_transition_flow(code, y, y1, y2, culvert);
        }
    }
    if (d_q_d_h_out != nullptr) *d_q_d_h_out = culvert.d_q_d_h;
    return q;
}

static double bw2d_critical_depth(const Bw2dXsect& xsect, double q)
{
    if (q <= 0.0) return 0.0;
    if (xsect.rectangular) {
        const double q_unit = q / std::max(1.0e-12, xsect.width);
        return std::min(std::pow((q_unit * q_unit) / BW2D_GRAVITY, 1.0 / 3.0), xsect.y_full);
    }

    const double target = (q * q) / BW2D_GRAVITY;
    double lo = 1.0e-4 * xsect.y_full;
    double hi = xsect.y_full;
    auto f = [&](double y) {
        const double a = bw2d_xsect_area(xsect, y);
        const double t = bw2d_xsect_top_width(xsect, y);
        return (t > 0.0) ? (a * a * a / t - target) : std::numeric_limits<double>::infinity();
    };
    double flo = f(lo);
    double fhi = f(hi);
    if (fhi <= 0.0) return xsect.y_full;
    if (flo >= 0.0) return lo;

    for (int it = 0; it < 80; ++it) {
        const double mid = 0.5 * (lo + hi);
        const double fmid = f(mid);
        if (std::abs(fmid) < 1.0e-9 * std::max(target, 1.0) || (hi - lo) < 1.0e-7) {
            return mid;
        }
        if (flo * fmid <= 0.0) {
            hi = mid;
            fhi = fmid;
        } else {
            lo = mid;
            flo = fmid;
        }
    }
    return 0.5 * (lo + hi);
}

static inline double bw2d_velocity(const Bw2dXsect& xsect, double q, double depth)
{
    const double area = bw2d_xsect_area(xsect, depth);
    if (area <= 0.0) return 0.0;
    return q / area;
}

static inline double bw2d_specific_energy(const Bw2dXsect& xsect, double q, double depth)
{
    const double v = bw2d_velocity(xsect, q, depth);
    return depth + v * v / (2.0 * BW2D_GRAVITY);
}

static inline double bw2d_friction_slope(const Bw2dXsect& xsect, double q, double n_value, double depth)
{
    if (depth <= 0.0 || n_value <= 0.0) return 0.0;
    const double area = bw2d_xsect_area(xsect, depth);
    const double radius = bw2d_xsect_hydraulic_radius(xsect, depth);
    if (area <= 0.0 || radius <= 0.0) return 0.0;
    const double conveyance = (1.49 / n_value) * area * std::pow(radius, 2.0 / 3.0);
    if (conveyance <= 0.0) return 0.0;
    return std::pow(q / conveyance, 2.0);
}

static double bw2d_solve_supercritical_depth_for_energy(const Bw2dXsect& xsect, double q, double target_energy)
{
    if (q <= 0.0) return 0.0;
    const double dc = bw2d_critical_depth(xsect, q);
    const double eps = std::max(1.0e-6, 1.0e-6 * xsect.y_full);
    const double lo = eps;
    const double hi = std::max(eps, std::min(dc, xsect.y_full - eps));
    if (hi <= lo) return std::max(eps, std::min(dc, xsect.y_full - eps));

    auto residual = [&](double depth) {
        return bw2d_specific_energy(xsect, q, depth) - target_energy;
    };

    const int samples = 240;
    const double step = (hi - lo) / static_cast<double>(std::max(samples - 1, 1));
    double best_depth = lo;
    double best_res = residual(lo);
    double prev_depth = lo;
    double prev_res = best_res;
    bool found_bracket = false;
    double a = lo;
    double b = hi;

    for (int i = 1; i < samples; ++i) {
        const double depth = lo + i * step;
        const double res = residual(depth);
        if (std::abs(res) < std::abs(best_res)) {
            best_depth = depth;
            best_res = res;
        }
        if (prev_res == 0.0) return prev_depth;
        if (prev_res * res < 0.0) {
            found_bracket = true;
            a = prev_depth;
            b = depth;
            break;
        }
        prev_depth = depth;
        prev_res = res;
    }
    if (!found_bracket) return best_depth;

    double fa = residual(a);
    for (int it = 0; it < 80; ++it) {
        const double m = 0.5 * (a + b);
        const double fm = residual(m);
        if (std::abs(fm) < 1.0e-10 || std::abs(b - a) < eps) return m;
        if (std::abs(fm) < std::abs(best_res)) {
            best_depth = m;
            best_res = fm;
        }
        if (fa * fm <= 0.0) {
            b = m;
        } else {
            a = m;
            fa = fm;
        }
    }
    return best_depth;
}

} // namespace

#ifdef HYDRA_HAS_CUDA
static py::array_t<double> compute_structure_flows_cuda(
    py::array_t<double, py::array::c_style | py::array::forcecast> cell_wse,
    py::array_t<double, py::array::c_style | py::array::forcecast> cell_bed,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> structure_type,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> upstream_cell,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> downstream_cell,
    py::array_t<double, py::array::c_style | py::array::forcecast> crest_elev,
    py::array_t<double, py::array::c_style | py::array::forcecast> width,
    py::array_t<double, py::array::c_style | py::array::forcecast> height,
    py::array_t<double, py::array::c_style | py::array::forcecast> diameter,
    py::array_t<double, py::array::c_style | py::array::forcecast> length,
    py::array_t<double, py::array::c_style | py::array::forcecast> roughness_n,
    py::array_t<double, py::array::c_style | py::array::forcecast> coeff,
    py::array_t<double, py::array::c_style | py::array::forcecast> cd,
    py::array_t<double, py::array::c_style | py::array::forcecast> opening,
    py::array_t<double, py::array::c_style | py::array::forcecast> q_pump,
    py::array_t<double, py::array::c_style | py::array::forcecast> max_flow,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> culvert_code,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> culvert_shape,
    py::array_t<double, py::array::c_style | py::array::forcecast> culvert_rise,
    py::array_t<double, py::array::c_style | py::array::forcecast> culvert_span,
    py::array_t<double, py::array::c_style | py::array::forcecast> culvert_area,
    py::array_t<double, py::array::c_style | py::array::forcecast> culvert_barrels,
    py::array_t<double, py::array::c_style | py::array::forcecast> culvert_slope,
    py::array_t<double, py::array::c_style | py::array::forcecast> inlet_invert_elev,
    py::array_t<double, py::array::c_style | py::array::forcecast> outlet_invert_elev,
    py::array_t<double, py::array::c_style | py::array::forcecast> entrance_loss_k,
    py::array_t<double, py::array::c_style | py::array::forcecast> exit_loss_k,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> embankment_enabled,
    py::array_t<double, py::array::c_style | py::array::forcecast> embankment_crest_elev,
    py::array_t<double, py::array::c_style | py::array::forcecast> embankment_overflow_width,
    py::array_t<double, py::array::c_style | py::array::forcecast> embankment_weir_coeff,
    double gravity,
    double model_to_ft)
{
    const py::ssize_t n_cells = cell_wse.size();
    const py::ssize_t ns = structure_type.size();
    require_array(cell_bed, n_cells, "cell_bed");
    require_array(upstream_cell, ns, "upstream_cell");
    require_array(downstream_cell, ns, "downstream_cell");
    require_array(crest_elev, ns, "crest_elev");
    require_array(width, ns, "width");
    require_array(height, ns, "height");
    require_array(diameter, ns, "diameter");
    require_array(length, ns, "length");
    require_array(roughness_n, ns, "roughness_n");
    require_array(coeff, ns, "coeff");
    require_array(cd, ns, "cd");
    require_array(opening, ns, "opening");
    require_array(q_pump, ns, "q_pump");
    require_array(max_flow, ns, "max_flow");
    require_array(culvert_code, ns, "culvert_code");
    require_array(culvert_shape, ns, "culvert_shape");
    require_array(culvert_rise, ns, "culvert_rise");
    require_array(culvert_span, ns, "culvert_span");
    require_array(culvert_area, ns, "culvert_area");
    require_array(culvert_barrels, ns, "culvert_barrels");
    require_array(culvert_slope, ns, "culvert_slope");
    require_array(inlet_invert_elev, ns, "inlet_invert_elev");
    require_array(outlet_invert_elev, ns, "outlet_invert_elev");
    require_array(entrance_loss_k, ns, "entrance_loss_k");
    require_array(exit_loss_k, ns, "exit_loss_k");
    require_array(embankment_enabled, ns, "embankment_enabled");
    require_array(embankment_crest_elev, ns, "embankment_crest_elev");
    require_array(embankment_overflow_width, ns, "embankment_overflow_width");
    require_array(embankment_weir_coeff, ns, "embankment_weir_coeff");

    auto out = py::array_t<double>(ns);
    swe2d_gpu_compute_structure_flows(
        static_cast<int32_t>(n_cells),
        static_cast<int32_t>(ns),
        n_cells ? cell_wse.data() : nullptr,
        n_cells ? cell_bed.data() : nullptr,
        ns ? structure_type.data() : nullptr,
        ns ? upstream_cell.data() : nullptr,
        ns ? downstream_cell.data() : nullptr,
        ns ? crest_elev.data() : nullptr,
        ns ? width.data() : nullptr,
        ns ? height.data() : nullptr,
        ns ? diameter.data() : nullptr,
        ns ? length.data() : nullptr,
        ns ? roughness_n.data() : nullptr,
        ns ? coeff.data() : nullptr,
        ns ? cd.data() : nullptr,
        ns ? opening.data() : nullptr,
        ns ? q_pump.data() : nullptr,
        ns ? max_flow.data() : nullptr,
        ns ? culvert_code.data() : nullptr,
        ns ? culvert_shape.data() : nullptr,
        ns ? culvert_rise.data() : nullptr,
        ns ? culvert_span.data() : nullptr,
        ns ? culvert_area.data() : nullptr,
        ns ? culvert_barrels.data() : nullptr,
        ns ? culvert_slope.data() : nullptr,
        ns ? inlet_invert_elev.data() : nullptr,
        ns ? outlet_invert_elev.data() : nullptr,
        ns ? entrance_loss_k.data() : nullptr,
        ns ? exit_loss_k.data() : nullptr,
        ns ? embankment_enabled.data() : nullptr,
        ns ? embankment_crest_elev.data() : nullptr,
        ns ? embankment_overflow_width.data() : nullptr,
        ns ? embankment_weir_coeff.data() : nullptr,
        gravity,
        model_to_ft,
        ns ? out.mutable_data() : nullptr);
    return out;
}

// ── Culvert diagnostics readback ──────────────────────────────────────────

extern SWE2DDeviceState* s_coupling_dev;

static py::array_t<double> readback_culvert_diagnostics() {
    SWE2DDeviceState* dev = s_coupling_dev;
    if (!dev || !dev->sf_ws.d_culvert_diagnostics || dev->sf_ws.n_structures <= 0) {
        return py::array_t<double>(py::array::ShapeContainer({0, 8}));
    }
    const int32_t n = dev->sf_ws.n_structures;
    py::array_t<double> result({n, 8});
    auto buf = result.mutable_unchecked<2>();
    cudaMemcpy(
        buf.mutable_data(0, 0),
        dev->sf_ws.d_culvert_diagnostics,
        static_cast<size_t>(n) * 8 * sizeof(double),
        cudaMemcpyDeviceToHost);
    return result;
}

static void ensure_culvert_diagnostics_buffer() {
    SWE2DDeviceState* dev = s_coupling_dev;
    if (!dev || dev->sf_ws.n_structures <= 0) return;
    if (!dev->sf_ws.d_culvert_diagnostics) {
        cudaMalloc(&dev->sf_ws.d_culvert_diagnostics,
                   static_cast<size_t>(dev->sf_ws.n_structures) * 8 * sizeof(double));
        dev->sf_ws.n_culvert_diag_capacity = dev->sf_ws.n_structures;
        cudaMemset(dev->sf_ws.d_culvert_diagnostics, 0,
                   static_cast<size_t>(dev->sf_ws.n_structures) * 8 * sizeof(double));
    }
}
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Thin Python wrapper for SWE2DMesh (holds the mesh by value)
// ─────────────────────────────────────────────────────────────────────────────
struct PyMesh {
    SWE2DMesh mesh;
};

// ─────────────────────────────────────────────────────────────────────────────
// Thin Python wrapper for SWE2DSolver (holds the solver; mesh kept alive
// via shared_ptr to PyMesh to prevent use-after-free)
// ─────────────────────────────────────────────────────────────────────────────
struct PySolver {
    std::shared_ptr<PyMesh> mesh_owner;
    SWE2DSolver*            solver = nullptr;

    ~PySolver() {
        if (solver) {
            swe2d_destroy(solver);
            solver = nullptr;
        }
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Module definition
// ─────────────────────────────────────────────────────────────────────────────
#ifndef HYDRA_SWE2D_PY_MODULE_NAME
#define HYDRA_SWE2D_PY_MODULE_NAME hydra_swe2d
#endif

PYBIND11_MODULE(HYDRA_SWE2D_PY_MODULE_NAME, m) {
    m.doc() = "2D SWE hybrid GPU/CPU solver on unstructured polygon mesh";

    // ── GPU query ─────────────────────────────────────────────────────────────
    m.def("swe2d_gpu_available", &swe2d_gpu_available,
          "Return True if a CUDA-capable GPU is present and the GPU path was compiled.");

#ifdef HYDRA_HAS_CUDA
    m.def("swe2d_gpu_device_sync", []() {
        cudaDeviceSynchronize();
        cudaGetLastError();
    }, "Full device sync + error clear.  Call after coupling work before solver step.");
    m.def("swe2d_gpu_ensure_culvert_diagnostics", &ensure_culvert_diagnostics_buffer,
          "Allocate device-side culvert diagnostics buffer if not already present.");
    m.def("swe2d_gpu_readback_culvert_diagnostics", &readback_culvert_diagnostics,
          "Read back [n_structures][8] culvert diagnostic array (D2H). Returns empty if not allocated.");
    m.def("swe2d_get_coupling_dev_ptr", []() -> uintptr_t {
        return reinterpret_cast<uintptr_t>(s_coupling_dev);
    }, "Return the global coupling device pointer as uintptr.");
#else
    m.def("swe2d_gpu_device_sync", []() {
    }, "No-op: device sync not available without CUDA.");
#endif

    m.def("swe2d_gpu_compute_structure_flows",
        &compute_structure_flows_cuda,
        py::arg("cell_wse"),
        py::arg("cell_bed"),
        py::arg("structure_type"),
        py::arg("upstream_cell"),
        py::arg("downstream_cell"),
        py::arg("crest_elev"),
        py::arg("width"),
        py::arg("height"),
        py::arg("diameter"),
        py::arg("length"),
        py::arg("roughness_n"),
        py::arg("coeff"),
        py::arg("cd"),
        py::arg("opening"),
        py::arg("q_pump"),
        py::arg("max_flow"),
        py::arg("culvert_code"),
        py::arg("culvert_shape"),
        py::arg("culvert_rise"),
        py::arg("culvert_span"),
        py::arg("culvert_area"),
        py::arg("culvert_barrels"),
        py::arg("culvert_slope"),
        py::arg("inlet_invert_elev"),
        py::arg("outlet_invert_elev"),
        py::arg("entrance_loss_k"),
        py::arg("exit_loss_k"),
        py::arg("embankment_enabled"),
        py::arg("embankment_crest_elev"),
        py::arg("embankment_overflow_width"),
        py::arg("embankment_weir_coeff"),
        py::arg("gravity") = 9.81,
        py::arg("model_to_ft") = 3.28084,
        "Compute per-structure flow transfers in model units.\n"
        "Weir/orifice/bridge/pump formulas are unit-agnostic; use the correct\n"
        "gravity for your model units.  Culverts convert to ft internally for\n"
        "HDS-5 tables, then convert results back to model units via model_to_ft.");

#ifdef HYDRA_HAS_CUDA
    m.def("swe2d_gpu_compute_structure_and_coupling_sources",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> cell_area,
           py::array_t<double, py::array::c_style | py::array::forcecast> cell_wse,
           py::array_t<double, py::array::c_style | py::array::forcecast> cell_bed,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> structure_type,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> upstream_cell,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> downstream_cell,
           py::array_t<double, py::array::c_style | py::array::forcecast> crest_elev,
           py::array_t<double, py::array::c_style | py::array::forcecast> width,
           py::array_t<double, py::array::c_style | py::array::forcecast> height,
           py::array_t<double, py::array::c_style | py::array::forcecast> diameter,
           py::array_t<double, py::array::c_style | py::array::forcecast> length,
           py::array_t<double, py::array::c_style | py::array::forcecast> roughness_n,
           py::array_t<double, py::array::c_style | py::array::forcecast> coeff,
           py::array_t<double, py::array::c_style | py::array::forcecast> cd,
           py::array_t<double, py::array::c_style | py::array::forcecast> opening,
           py::array_t<double, py::array::c_style | py::array::forcecast> q_pump,
           py::array_t<double, py::array::c_style | py::array::forcecast> max_flow,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> culvert_code,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> culvert_shape,
           py::array_t<double, py::array::c_style | py::array::forcecast> culvert_rise,
           py::array_t<double, py::array::c_style | py::array::forcecast> culvert_span,
           py::array_t<double, py::array::c_style | py::array::forcecast> culvert_area,
           py::array_t<double, py::array::c_style | py::array::forcecast> culvert_barrels,
           py::array_t<double, py::array::c_style | py::array::forcecast> culvert_slope,
           py::array_t<double, py::array::c_style | py::array::forcecast> inlet_invert_elev,
           py::array_t<double, py::array::c_style | py::array::forcecast> outlet_invert_elev,
           py::array_t<double, py::array::c_style | py::array::forcecast> entrance_loss_k,
           py::array_t<double, py::array::c_style | py::array::forcecast> exit_loss_k,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> embankment_enabled,
           py::array_t<double, py::array::c_style | py::array::forcecast> embankment_crest_elev,
           py::array_t<double, py::array::c_style | py::array::forcecast> embankment_overflow_width,
           py::array_t<double, py::array::c_style | py::array::forcecast> embankment_weir_coeff,
           double gravity,
           double model_to_ft,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> inlet_cell,
           py::array_t<double, py::array::c_style | py::array::forcecast> inlet_flow)
           -> py::array_t<double>
        {
            const int32_t n_cells = static_cast<int32_t>(cell_area.size());
            const int32_t n_structures = static_cast<int32_t>(structure_type.size());
            auto out = py::array_t<double>(n_cells);
            swe2d_gpu_compute_structure_and_coupling_sources(
                n_cells,
                (n_cells > 0) ? cell_area.data() : nullptr,
                n_structures,
                (n_cells > 0) ? cell_wse.data() : nullptr,
                (n_cells > 0) ? cell_bed.data() : nullptr,
                (n_structures > 0) ? structure_type.data() : nullptr,
                (n_structures > 0) ? upstream_cell.data() : nullptr,
                (n_structures > 0) ? downstream_cell.data() : nullptr,
                (n_structures > 0) ? crest_elev.data() : nullptr,
                (n_structures > 0) ? width.data() : nullptr,
                (n_structures > 0) ? height.data() : nullptr,
                (n_structures > 0) ? diameter.data() : nullptr,
                (n_structures > 0) ? length.data() : nullptr,
                (n_structures > 0) ? roughness_n.data() : nullptr,
                (n_structures > 0) ? coeff.data() : nullptr,
                (n_structures > 0) ? cd.data() : nullptr,
                (n_structures > 0) ? opening.data() : nullptr,
                (n_structures > 0) ? q_pump.data() : nullptr,
                (n_structures > 0) ? max_flow.data() : nullptr,
                (n_structures > 0) ? culvert_code.data() : nullptr,
                (n_structures > 0) ? culvert_shape.data() : nullptr,
                (n_structures > 0) ? culvert_rise.data() : nullptr,
                (n_structures > 0) ? culvert_span.data() : nullptr,
                (n_structures > 0) ? culvert_area.data() : nullptr,
                (n_structures > 0) ? culvert_barrels.data() : nullptr,
                (n_structures > 0) ? culvert_slope.data() : nullptr,
                (n_structures > 0) ? inlet_invert_elev.data() : nullptr,
                (n_structures > 0) ? outlet_invert_elev.data() : nullptr,
                (n_structures > 0) ? entrance_loss_k.data() : nullptr,
                (n_structures > 0) ? exit_loss_k.data() : nullptr,
                (n_structures > 0) ? embankment_enabled.data() : nullptr,
                (n_structures > 0) ? embankment_crest_elev.data() : nullptr,
                (n_structures > 0) ? embankment_overflow_width.data() : nullptr,
                (n_structures > 0) ? embankment_weir_coeff.data() : nullptr,
                gravity,
                model_to_ft,
                static_cast<int32_t>(inlet_cell.size()),
                inlet_cell.size() > 0 ? inlet_cell.data() : nullptr,
                inlet_flow.size() > 0 ? inlet_flow.data() : nullptr,
                out.mutable_data());
            return out;
        },
        py::arg("cell_area"),
        py::arg("cell_wse"),
        py::arg("cell_bed"),
        py::arg("structure_type"),
        py::arg("upstream_cell"),
        py::arg("downstream_cell"),
        py::arg("crest_elev"),
        py::arg("width"),
        py::arg("height"),
        py::arg("diameter"),
        py::arg("length"),
        py::arg("roughness_n"),
        py::arg("coeff"),
        py::arg("cd"),
        py::arg("opening"),
        py::arg("q_pump"),
        py::arg("max_flow"),
        py::arg("culvert_code"),
        py::arg("culvert_shape"),
        py::arg("culvert_rise"),
        py::arg("culvert_span"),
        py::arg("culvert_area"),
        py::arg("culvert_barrels"),
        py::arg("culvert_slope"),
        py::arg("inlet_invert_elev"),
        py::arg("outlet_invert_elev"),
        py::arg("entrance_loss_k"),
        py::arg("exit_loss_k"),
        py::arg("embankment_enabled"),
        py::arg("embankment_crest_elev"),
        py::arg("embankment_overflow_width"),
        py::arg("embankment_weir_coeff"),
        py::arg("gravity") = 9.81,
        py::arg("model_to_ft") = 3.28084,
        py::arg("inlet_cell"),
        py::arg("inlet_flow"),
        "Fused CUDA helper: compute structure flows and coupling sources on-device, returning per-cell source rates [m/s].");

    // ── Persistent GPU coupling path ──
    m.def("swe2d_gpu_set_coupling_device_global",
        [](uintptr_t dev_ptr) { swe2d_gpu_set_coupling_device_global(reinterpret_cast<SWE2DDeviceState*>(dev_ptr)); },
        py::arg("dev_ptr"), "Set global device pointer for persistent coupling.");

    // ── Snapshot ring buffer bindings ──
    m.def("swe2d_gpu_store_snapshot",
        [](std::shared_ptr<PySolver>& ps, double t_s) {
            if (!ps || !ps->solver || !ps->solver->dev) return;
            swe2d_gpu_store_snapshot(ps->solver->dev, t_s);
        },
        py::arg("solver"), py::arg("t_s"),
        "Copy current h/hu/hv to the next snapshot slot on the device ring buffer.");

    m.def("swe2d_gpu_read_snapshots",
        [](std::shared_ptr<PySolver>& ps) -> py::dict {
            if (!ps || !ps->solver || !ps->solver->dev) return py::dict();
            SWE2DDeviceState* dev = ps->solver->dev;
            double *h_ts = nullptr, *h_h = nullptr, *h_hu = nullptr, *h_hv = nullptr;
            int32_t count = 0, n_cells = 0;
            swe2d_gpu_read_snapshots(dev, &h_ts, &h_h, &h_hu, &h_hv, &count, &n_cells);
            if (count <= 0 || n_cells <= 0) return py::dict();
            // Wrap pinned host memory as numpy arrays (no copy).
            auto capsule = py::capsule(h_ts, [](void* p) { cudaFreeHost(p); });
            auto cap_h  = py::capsule(h_h,  [](void* p) { cudaFreeHost(p); });
            auto cap_hu = py::capsule(h_hu, [](void* p) { cudaFreeHost(p); });
            auto cap_hv = py::capsule(h_hv, [](void* p) { cudaFreeHost(p); });
            py::dict d;
            d["t_s"] = py::array_t<double>({count}, {sizeof(double)}, h_ts, capsule);
            d["h"]   = py::array_t<double>({count, n_cells}, {static_cast<long>(n_cells) * sizeof(double), sizeof(double)}, h_h, cap_h);
            d["hu"]  = py::array_t<double>({count, n_cells}, {static_cast<long>(n_cells) * sizeof(double), sizeof(double)}, h_hu, cap_hu);
            d["hv"]  = py::array_t<double>({count, n_cells}, {static_cast<long>(n_cells) * sizeof(double), sizeof(double)}, h_hv, cap_hv);
            // NOTE: device ring buffer is NOT freed here — fetch is non-destructive.
            // Call swe2d_gpu_free_snapshot_buf explicitly when reset is needed
            // (e.g. before starting a new simulation).
            return d;
        },
        py::arg("solver"),
        "Read all accumulated snapshots as {t_s, h, hu, hv} dict. Does NOT reset buffer.");

    m.def("swe2d_gpu_free_snapshot_buf",
        [](std::shared_ptr<PySolver>& ps) {
            if (!ps || !ps->solver || !ps->solver->dev) return;
            swe2d_gpu_free_snapshot_buf(ps->solver->dev);
        },
        py::arg("solver"),
        "Free the snapshot ring buffer on device.");

    m.def("swe2d_gpu_snapshot_count",
        [](std::shared_ptr<PySolver>& ps) -> int {
            if (!ps || !ps->solver || !ps->solver->dev) return 0;
            return ps->solver->dev->snap_count;
        },
        py::arg("solver"),
        "Return number of snapshots currently in the device ring buffer.");

    // ── Line metrics ring buffer bindings ──
    m.def("swe2d_gpu_configure_line_sampling",
        [](std::shared_ptr<PySolver>& ps,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> station_offsets,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> cell_idx,
           py::array_t<double,  py::array::c_style | py::array::forcecast> weights,
           py::array_t<double,  py::array::c_style | py::array::forcecast> normal_x,
           py::array_t<double,  py::array::c_style | py::array::forcecast> normal_y,
           py::array_t<double,  py::array::c_style | py::array::forcecast> station_m,
           double gravity,
           double h_min) {
            if (!ps || !ps->solver || !ps->solver->dev)
                throw std::runtime_error("solver not initialized");
            SWE2DDeviceState* dev = ps->solver->dev;
            int32_t n_lines = static_cast<int32_t>(station_offsets.size()) - 1;
            swe2d_gpu_configure_line_sampling(
                dev, n_lines,
                station_offsets.data(), cell_idx.data(), weights.data(),
                normal_x.data(), normal_y.data(), station_m.data(),
                gravity, h_min);
        },
        py::arg("solver"), py::arg("station_offsets"), py::arg("cell_idx"),
        py::arg("weights"), py::arg("normal_x"), py::arg("normal_y"),
        py::arg("station_m"), py::arg("gravity"), py::arg("h_min"),
        "Upload line sampling map and allocate ring buffer on device.");

    m.def("swe2d_gpu_read_line_metrics",
        [](std::shared_ptr<PySolver>& ps) -> py::dict {
            if (!ps || !ps->solver || !ps->solver->dev) return py::dict();
            SWE2DDeviceState* dev = ps->solver->dev;
            double *h_times = nullptr, *h_profile = nullptr, *h_ts = nullptr;
            int32_t *h_wet = nullptr;
            double  *h_station_m       = nullptr;
            int32_t *h_station_offsets = nullptr;
            int32_t count = 0, n_lines = 0, total_stations = 0;
            swe2d_gpu_read_line_metrics(dev, &h_times, &h_profile, &h_ts,
                                         &h_wet, &h_station_m, &h_station_offsets,
                                         &count, &n_lines, &total_stations);
            if (count <= 0 || total_stations <= 0) return py::dict();
            auto cap_times   = py::capsule(h_times,   [](void* p) { cudaFreeHost(p); });
            auto cap_profile = py::capsule(h_profile, [](void* p) { cudaFreeHost(p); });
            auto cap_ts      = py::capsule(h_ts,      [](void* p) { cudaFreeHost(p); });
            auto cap_wet     = py::capsule(h_wet,     [](void* p) { cudaFreeHost(p); });
            auto cap_sm      = py::capsule(h_station_m,       [](void* p) { cudaFreeHost(p); });
            auto cap_so      = py::capsule(h_station_offsets, [](void* p) { cudaFreeHost(p); });

            py::dict d;
            d["t_s"] = py::array_t<double>({count}, {sizeof(double)}, h_times, cap_times);
            // 3D/2D arrays use the 3-arg ctor (shape, ptr, base) — auto-computes C-style strides
            d["profiles"] = py::array_t<double>({count, total_stations, 6}, h_profile, cap_profile);
            d["ts"]       = py::array_t<double>({count, n_lines, 7},         h_ts,      cap_ts);
            d["wet"]      = py::array_t<int32_t>({count, total_stations},     h_wet,     cap_wet);
            // Station map arrays (host copies — safe for Python access)
            d["station_m"] = py::array_t<double>(
                {total_stations}, {sizeof(double)},
                h_station_m, cap_sm);
            d["station_offsets"] = py::array_t<int32_t>(
                {n_lines + 1}, {sizeof(int32_t)},
                h_station_offsets, cap_so);
            d["n_lines"] = n_lines;
            d["total_stations"] = total_stations;
            return d;
        },
        py::arg("solver"),
        "Read all accumulated line metrics as {t_s, profiles, ts, wet, n_lines, total_stations, station_m, station_offsets}. Returns pinned memory.");

    m.def("swe2d_gpu_free_line_metrics",
        [](std::shared_ptr<PySolver>& ps) {
            if (!ps || !ps->solver || !ps->solver->dev) return;
            swe2d_gpu_free_line_metrics(ps->solver->dev);
        },
        py::arg("solver"),
        "Free line metrics ring buffer and sample map on device.");

    m.def("swe2d_gpu_device_memory_info",
        []() -> py::dict {
            size_t free_bytes = 0, total_bytes = 0;
            cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
            py::dict d;
            d["free_bytes"] = static_cast<uint64_t>(free_bytes);
            d["total_bytes"] = static_cast<uint64_t>(total_bytes);
            d["err"] = static_cast<int>(err);
            return d;
        },
        "Return {free_bytes, total_bytes, err} dict from cudaMemGetInfo.");

    // ── Test helpers (memory pressure simulation) ──
    // Returns opaque uintptr; caller must pass it back to free.
    m.def("swe2d_gpu_test_alloc",
        [](uint64_t bytes) -> uintptr_t {
            if (bytes == 0) return 0;
            void* ptr = nullptr;
            cudaError_t err = cudaMalloc(&ptr, static_cast<size_t>(bytes));
            if (err != cudaSuccess) {
                throw std::runtime_error(std::string("cudaMalloc failed: ") + cudaGetErrorString(err));
            }
            return reinterpret_cast<uintptr_t>(ptr);
        },
        py::arg("bytes"),
        "Allocate a device buffer of the given size (bytes) for memory-pressure testing. "
        "Returns an opaque uintptr; pass it to swe2d_gpu_test_free to release.");

    m.def("swe2d_gpu_test_free",
        [](uintptr_t ptr) {
            if (ptr) cudaFree(reinterpret_cast<void*>(ptr));
        },
        py::arg("ptr"),
        "Free a device buffer allocated by swe2d_gpu_test_alloc.");

    m.def("swe2d_gpu_preload_structure_params",
        [](py::array_t<int32_t, py::array::c_style|py::array::forcecast> structure_type,
           py::array_t<int32_t, py::array::c_style|py::array::forcecast> upstream_cell,
           py::array_t<int32_t, py::array::c_style|py::array::forcecast> downstream_cell,
           py::array_t<double, py::array::c_style|py::array::forcecast> crest_elev,
           py::array_t<double, py::array::c_style|py::array::forcecast> width,
           py::array_t<double, py::array::c_style|py::array::forcecast> height,
           py::array_t<double, py::array::c_style|py::array::forcecast> diameter,
           py::array_t<double, py::array::c_style|py::array::forcecast> length,
           py::array_t<double, py::array::c_style|py::array::forcecast> roughness_n,
           py::array_t<double, py::array::c_style|py::array::forcecast> coeff,
           py::array_t<double, py::array::c_style|py::array::forcecast> cd,
           py::array_t<double, py::array::c_style|py::array::forcecast> opening,
           py::array_t<double, py::array::c_style|py::array::forcecast> q_pump,
           py::array_t<double, py::array::c_style|py::array::forcecast> max_flow,
           py::array_t<int32_t, py::array::c_style|py::array::forcecast> culvert_code,
           py::array_t<int32_t, py::array::c_style|py::array::forcecast> culvert_shape,
           py::array_t<double, py::array::c_style|py::array::forcecast> culvert_rise,
           py::array_t<double, py::array::c_style|py::array::forcecast> culvert_span,
           py::array_t<double, py::array::c_style|py::array::forcecast> culvert_area,
           py::array_t<double, py::array::c_style|py::array::forcecast> culvert_barrels,
           py::array_t<double, py::array::c_style|py::array::forcecast> culvert_slope,
           py::array_t<double, py::array::c_style|py::array::forcecast> inlet_invert_elev,
           py::array_t<double, py::array::c_style|py::array::forcecast> outlet_invert_elev,
           py::array_t<double, py::array::c_style|py::array::forcecast> entrance_loss_k,
           py::array_t<double, py::array::c_style|py::array::forcecast> exit_loss_k,
           py::array_t<int32_t, py::array::c_style|py::array::forcecast> embankment_enabled,
           py::array_t<double, py::array::c_style|py::array::forcecast> embankment_crest_elev,
           py::array_t<double, py::array::c_style|py::array::forcecast> embankment_overflow_width,
           py::array_t<double, py::array::c_style|py::array::forcecast> embankment_weir_coeff,
           double gravity = 9.81,
           double model_to_ft = 3.28084)
        {
            int32_t n = static_cast<int32_t>(structure_type.size());
            swe2d_gpu_preload_structure_params(
                nullptr, n,
                n>0?structure_type.data():nullptr, n>0?upstream_cell.data():nullptr, n>0?downstream_cell.data():nullptr,
                n>0?crest_elev.data():nullptr, n>0?width.data():nullptr, n>0?height.data():nullptr,
                n>0?diameter.data():nullptr, n>0?length.data():nullptr, n>0?roughness_n.data():nullptr,
                n>0?coeff.data():nullptr, n>0?cd.data():nullptr, n>0?opening.data():nullptr,
                n>0?q_pump.data():nullptr, n>0?max_flow.data():nullptr,
                n>0?culvert_code.data():nullptr, n>0?culvert_shape.data():nullptr,
                n>0?culvert_rise.data():nullptr, n>0?culvert_span.data():nullptr, n>0?culvert_area.data():nullptr,
                n>0?culvert_barrels.data():nullptr, n>0?culvert_slope.data():nullptr,
                n>0?inlet_invert_elev.data():nullptr, n>0?outlet_invert_elev.data():nullptr,
                n>0?entrance_loss_k.data():nullptr, n>0?exit_loss_k.data():nullptr,
                n>0?embankment_enabled.data():nullptr, n>0?embankment_crest_elev.data():nullptr,
                n>0?embankment_overflow_width.data():nullptr, n>0?embankment_weir_coeff.data():nullptr,
                gravity, model_to_ft);
        }, "Preload structure params to GPU once.");

    m.def("swe2d_gpu_preload_coupling_cell_area",
        [](py::array_t<double, py::array::c_style|py::array::forcecast> cell_area) {
            swe2d_gpu_preload_coupling_cell_area(nullptr, static_cast<int32_t>(cell_area.size()), cell_area.data());
        }, py::arg("cell_area"), "Preload cell areas to GPU once.");

    m.def("swe2d_gpu_compute_coupling_full_on_device",
        [](py::object cell_wse_obj,
           int32_t n_structures,
           py::object host_flows_obj) {
            const double* cell_wse_ptr = nullptr;
            int32_t n_cells = 0;
            const double* host_flows_ptr = nullptr;
            if (!cell_wse_obj.is_none()) {
                auto cell_wse = cell_wse_obj.cast<py::array_t<double, py::array::c_style|py::array::forcecast>>();
                cell_wse_ptr = cell_wse.data();
                n_cells = static_cast<int32_t>(cell_wse.size());
            }
            if (!host_flows_obj.is_none()) {
                auto host_flows = host_flows_obj.cast<py::array_t<double, py::array::c_style|py::array::forcecast>>();
                host_flows_ptr = host_flows.data();
            }
            swe2d_gpu_compute_coupling_full_on_device(
                nullptr, n_cells, n_structures, cell_wse_ptr,
                host_flows_ptr);
        }, py::arg("cell_wse")=py::none(), py::arg("n_structures")=0,
           py::arg("host_structure_flows")=py::none(),
        "Run full coupling on-device using preloaded params. "
        "Pass cell_wse=None to compute WSE = h + zb on GPU. "
        "Pass host_structure_flows to override GPU-computed structure flows.");

    m.def("swe2d_gpu_ensure_drainage_q_buf",
        [](int32_t n_cells) {
            extern SWE2DDeviceState* s_coupling_dev;
            swe2d_gpu_ensure_drainage_q_buf(s_coupling_dev, n_cells);
        },
        "Allocate the persistent d_drainage_q buffer in the coupling workspace.");

    m.def("swe2d_gpu_upload_drainage_exchange_params",
        [](py::object inlet_cell_obj, py::object inlet_node_obj,
           py::object inlet_crest_obj, py::object inlet_width_obj,
           py::object inlet_cd_obj, py::object inlet_qmax_obj,
           py::object outfall_cell_obj, py::object outfall_node_obj,
           py::object outfall_invert_obj, py::object outfall_diameter_obj,
           py::object outfall_cd_obj, py::object outfall_qmax_obj,
           py::object outfall_zero_storage_obj,
           py::object node_max_depth_obj) {
            extern SWE2DDeviceState* s_coupling_dev;
            auto get_arr_i32 = [](py::object o, py::array_t<int32_t>& out) {
                if (!o.is_none()) out = o.cast<py::array_t<int32_t, py::array::c_style|py::array::forcecast>>();
            };
            auto get_arr_f64 = [](py::object o, py::array_t<double>& out) {
                if (!o.is_none()) out = o.cast<py::array_t<double, py::array::c_style|py::array::forcecast>>();
            };
            py::array_t<double> nmd;
            py::array_t<int32_t> ic, in_, oc, on_, ozs;
            py::array_t<double> icr, iw, icd, iq, oi_, od_, ocd, oq;
            get_arr_f64(node_max_depth_obj, nmd);
            int32_t n_nodes = static_cast<int32_t>(nmd.size());
            get_arr_i32(inlet_cell_obj, ic);
            int32_t n_inlets = static_cast<int32_t>(ic.size());
            get_arr_i32(inlet_node_obj, in_);
            get_arr_f64(inlet_crest_obj, icr);
            get_arr_f64(inlet_width_obj, iw);
            get_arr_f64(inlet_cd_obj, icd);
            get_arr_f64(inlet_qmax_obj, iq);
            get_arr_i32(outfall_cell_obj, oc);
            int32_t n_outfalls = static_cast<int32_t>(oc.size());
            get_arr_i32(outfall_node_obj, on_);
            get_arr_f64(outfall_invert_obj, oi_);
            get_arr_f64(outfall_diameter_obj, od_);
            get_arr_f64(outfall_cd_obj, ocd);
            get_arr_f64(outfall_qmax_obj, oq);
            get_arr_i32(outfall_zero_storage_obj, ozs);
            swe2d_gpu_upload_drainage_exchange_params(
                s_coupling_dev, n_nodes, n_inlets, n_outfalls,
                ic.data(), in_.data(), icr.data(), iw.data(),
                icd.data(), iq.data(),
                oc.data(), on_.data(), oi_.data(), od_.data(),
                ocd.data(), oq.data(), ozs.data(),
                nmd.data());
        },
        py::arg("inlet_cell")=py::none(), py::arg("inlet_node")=py::none(),
        py::arg("inlet_crest")=py::none(), py::arg("inlet_width")=py::none(),
        py::arg("inlet_cd")=py::none(), py::arg("inlet_qmax")=py::none(),
        py::arg("outfall_cell")=py::none(), py::arg("outfall_node")=py::none(),
        py::arg("outfall_invert")=py::none(), py::arg("outfall_diameter")=py::none(),
        py::arg("outfall_cd")=py::none(), py::arg("outfall_qmax")=py::none(),
        py::arg("outfall_zero_storage")=py::none(),
        py::arg("node_max_depth")=py::none(),
        "Upload drainage exchange parameters (inlets, outfalls, node max depth) to GPU.");

    m.def("swe2d_gpu_accumulate_external_source",
        [](std::shared_ptr<PySolver>& ps,
           py::array_t<double, py::array::c_style | py::array::forcecast> src)
        {
            if (!ps || !ps->solver || !ps->solver->dev) return;
            const int32_t nc = static_cast<int32_t>(src.size());
            if (nc != ps->solver->mesh->n_cells) {
                throw std::invalid_argument("source array length must match n_cells");
            }
            swe2d_gpu_accumulate_external_source(
                ps->solver->dev, src.data(), nc);
        },
        py::arg("solver"), py::arg("src"),
        "Accumulate host-provided source rates into d_external_source_mps on-device.\n"
        "No D2H readback — uploads to persistent staging buffer and adds via kernel.");

    m.def("swe2d_gpu_readback_coupling_sources",
        [](int32_t n_cells) -> py::array_t<double> {
            auto result = py::array_t<double>(n_cells);
            swe2d_gpu_readback_coupling_sources(result.mutable_data(), n_cells);
            return result;
        }, py::arg("n_cells"),
        "Read back coupling source rates [m/s] from device after on-device compute.");

    m.def("swe2d_gpu_readback_structure_flows",
        [](int32_t n_structures) -> py::array_t<double> {
            auto result = py::array_t<double>(n_structures);
            swe2d_gpu_readback_structure_flows(result.mutable_data(), n_structures);
            return result;
        }, py::arg("n_structures"),
        "Read back per-structure flow rates [m^3/s] from persistent GPU buffer.");

    m.def("swe2d_gpu_readback_h",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> h_out,
           int32_t n_cells) {
            swe2d_gpu_readback_h(h_out.mutable_data(), n_cells);
        }, py::arg("h_out"), py::arg("n_cells"),
        "Read back current depth array h from coupling device state (lightweight, no solver handle needed).");

    m.def("swe2d_gpu_upload_structure_flows",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> flows) {
            int32_t n = static_cast<int32_t>(flows.size());
            swe2d_gpu_upload_structure_flows(n > 0 ? flows.data() : nullptr, n);
        }, py::arg("flows"),
        "Upload per-structure flow rates [model-units] to the persistent GPU buffer.");

    m.def("swe2d_gpu_readback_coupling_wse",
        [](int32_t n_cells) -> py::array_t<double> {
            auto result = py::array_t<double>(n_cells);
            swe2d_gpu_readback_coupling_wse(result.mutable_data(), n_cells);
            return result;
        }, py::arg("n_cells"),
        "Read back coupling WSE array [model-units] from device.");

    // ── Face-based culvert flux coupling ──────────────────────────────────────
    m.def("swe2d_gpu_upload_culvert_face_flux_params",
        [](py::array_t<int32_t, py::array::c_style | py::array::forcecast> culvert_struct_idx,
           py::array_t<double, py::array::c_style | py::array::forcecast> face_nx,
           py::array_t<double, py::array::c_style | py::array::forcecast> face_ny,
           py::array_t<double, py::array::c_style | py::array::forcecast> face_width,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> donor_cell,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> receiver_cell,
           py::array_t<double, py::array::c_style | py::array::forcecast> invert_elev,
           py::array_t<double, py::array::c_style | py::array::forcecast> depth_safety,
           py::array_t<double, py::array::c_style | py::array::forcecast> donor_cell_area,
           bool use_face_flux,
           py::object enquiry_up_cell_obj,
           py::object enquiry_dn_cell_obj)
        {
            int32_t n = static_cast<int32_t>(culvert_struct_idx.size());
            const int32_t* enq_up_ptr = nullptr;
            const int32_t* enq_dn_ptr = nullptr;
            if (!enquiry_up_cell_obj.is_none()) {
                auto enq_up = enquiry_up_cell_obj.cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
                enq_up_ptr = enq_up.data();
            }
            if (!enquiry_dn_cell_obj.is_none()) {
                auto enq_dn = enquiry_dn_cell_obj.cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
                enq_dn_ptr = enq_dn.data();
            }
            swe2d_gpu_upload_culvert_face_flux_params(
                nullptr,
                n,
                n > 0 ? culvert_struct_idx.data() : nullptr,
                n > 0 ? face_nx.data() : nullptr,
                n > 0 ? face_ny.data() : nullptr,
                n > 0 ? face_width.data() : nullptr,
                n > 0 ? donor_cell.data() : nullptr,
                n > 0 ? receiver_cell.data() : nullptr,
                n > 0 ? invert_elev.data() : nullptr,
                n > 0 ? depth_safety.data() : nullptr,
                n > 0 ? donor_cell_area.data() : nullptr,
                enq_up_ptr,
                enq_dn_ptr,
                use_face_flux);
        },
        py::arg("culvert_struct_idx"),
        py::arg("face_nx"),
        py::arg("face_ny"),
        py::arg("face_width"),
        py::arg("donor_cell"),
        py::arg("receiver_cell"),
        py::arg("invert_elev"),
        py::arg("depth_safety"),
        py::arg("donor_cell_area"),
        py::arg("use_face_flux"),
        py::arg("enquiry_up_cell")=py::none(),
        py::arg("enquiry_dn_cell")=py::none(),
        "Upload culvert face-flux geometry to GPU for face-based coupling.");

    m.def("swe2d_gpu_apply_culvert_face_flux",
        [](double dt, double h_min)
        {
            swe2d_gpu_apply_culvert_face_flux(nullptr, dt, h_min);
        },
        py::arg("dt"),
        py::arg("h_min") = 1.0e-6,
        "Apply face-based culvert flux coupling on device (computes Q_c, "
        "builds face fluxes, masks culverts from source kernel).");

    m.def("swe2d_gpu_fold_culvert_mass_to_source",
        [](int32_t n_cells)
        {
            swe2d_gpu_fold_culvert_mass_to_source(nullptr, n_cells);
        },
        py::arg("n_cells"),
        "Fold culvert face-flux mass into d_external_source_mps for subcycling support.");

    m.def("swe2d_gpu_readback_ext_struct_flux",
        [](int32_t n_cells)
           -> std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>>
        {
            auto h  = py::array_t<double>(n_cells);
            auto hu = py::array_t<double>(n_cells);
            auto hv = py::array_t<double>(n_cells);
            swe2d_gpu_readback_ext_struct_flux(
                h.mutable_data(), hu.mutable_data(), hv.mutable_data(), n_cells);
            return std::make_tuple(h, hu, hv);
        },
        py::arg("n_cells"),
        "Read back per-cell external structure flux arrays (for debug).");

    m.def("swe2d_gpu_alloc_ext_struct_flux",
        [](int32_t n_cells)
        {
            swe2d_gpu_alloc_ext_struct_flux(nullptr, n_cells);
        },
        py::arg("n_cells"),
        "Allocate per-cell external structure flux accumulators on device.");

    m.def("swe2d_gpu_upload_ext_struct_flux_h",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> flux_h)
        {
            int32_t n = static_cast<int32_t>(flux_h.size());
            swe2d_gpu_upload_ext_struct_flux_h(n > 0 ? flux_h.data() : nullptr, n);
        },
        py::arg("flux_h"),
        "Upload redistributed mass flux to device d_ext_struct_flux_h.");

    m.def("swe2d_gpu_set_coupling_dt",
        [](double dt)
        {
            swe2d_gpu_set_coupling_dt(dt);
        },
        py::arg("dt"),
        "Set the coupling time step for the face-flux depth limiter.");

    // Culvert table mode: build pre-computed Q(hw,tw) lookup tables from culvert params.
    m.def("swe2d_gpu_build_culvert_tables",
        [](py::array_t<int32_t, py::array::c_style | py::array::forcecast> culvert_code,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> culvert_shape,
           py::array_t<double, py::array::c_style | py::array::forcecast> culvert_rise,
           py::array_t<double, py::array::c_style | py::array::forcecast> culvert_span,
           py::array_t<double, py::array::c_style | py::array::forcecast> culvert_diameter,
           py::array_t<double, py::array::c_style | py::array::forcecast> culvert_length,
           py::array_t<double, py::array::c_style | py::array::forcecast> culvert_roughness_n,
           py::array_t<double, py::array::c_style | py::array::forcecast> culvert_slope,
           py::array_t<double, py::array::c_style | py::array::forcecast> entrance_loss_k,
           py::array_t<double, py::array::c_style | py::array::forcecast> exit_loss_k,
           double model_to_ft,
           int32_t n_hw, int32_t n_tw)
           -> std::tuple<py::array_t<double>, py::array_t<double>>
        {
            int32_t n = static_cast<int32_t>(culvert_code.size());
            std::vector<double> table_data, table_header;
            bool ok = swe2d_gpu_build_culvert_tables(
                n,
                n > 0 ? culvert_code.data() : nullptr,
                n > 0 ? culvert_shape.data() : nullptr,
                n > 0 ? culvert_rise.data() : nullptr,
                n > 0 ? culvert_span.data() : nullptr,
                n > 0 ? culvert_diameter.data() : nullptr,
                n > 0 ? culvert_length.data() : nullptr,
                n > 0 ? culvert_roughness_n.data() : nullptr,
                n > 0 ? culvert_slope.data() : nullptr,
                n > 0 ? entrance_loss_k.data() : nullptr,
                n > 0 ? exit_loss_k.data() : nullptr,
                model_to_ft,
                n_hw, n_tw,
                table_data, table_header);
            if (!ok) {
                throw std::runtime_error("CUDA culvert table generation failed");
            }
            auto py_data = py::array_t<double>(table_data.size());
            auto py_header = py::array_t<double>(table_header.size());
            std::memcpy(py_data.mutable_data(), table_data.data(), table_data.size() * sizeof(double));
            std::memcpy(py_header.mutable_data(), table_header.data(), table_header.size() * sizeof(double));
            return std::make_tuple(py_data, py_header);
        },
        py::arg("culvert_code"),
        py::arg("culvert_shape"),
        py::arg("culvert_rise"),
        py::arg("culvert_span"),
        py::arg("culvert_diameter"),
        py::arg("culvert_length"),
        py::arg("culvert_roughness_n"),
        py::arg("culvert_slope"),
        py::arg("entrance_loss_k"),
        py::arg("exit_loss_k"),
        py::arg("model_to_ft"),
        py::arg("n_hw") = 32,
        py::arg("n_tw") = 16,
        "Build pre-computed culvert Q(hw,tw) lookup tables on GPU, returning (table_data, table_header).");

    // Set culvert solver mode: 0 = direct secant (default), 1 = table lookup.
    m.def("swe2d_gpu_set_culvert_solver_mode",
        [](int32_t mode,
           py::array_t<double, py::array::c_style | py::array::forcecast> table_data,
           py::array_t<double, py::array::c_style | py::array::forcecast> table_header,
           int32_t n_hw, int32_t n_tw) {
            // Use extern to access the static variables in swe2d_gpu.cu
            extern void swe2d_gpu_set_culvert_solver_mode_impl(
                int32_t mode, const double* data, const double* header,
                size_t data_sz, size_t header_sz, int32_t n_hw, int32_t n_tw);
            swe2d_gpu_set_culvert_solver_mode_impl(
                mode,
                table_data.size() > 0 ? table_data.data() : nullptr,
                table_header.size() > 0 ? table_header.data() : nullptr,
                static_cast<size_t>(table_data.size()),
                static_cast<size_t>(table_header.size()),
                n_hw, n_tw);
        },
        py::arg("mode"),
        py::arg("table_data") = py::array_t<double>(),
        py::arg("table_header") = py::array_t<double>(),
        py::arg("n_hw") = 32,
        py::arg("n_tw") = 16,
        "Set culvert solver mode. mode=0 for direct secant, mode=1 for table lookup.");
#endif

    m.def("swe2d_gpu_compute_coupling_sources",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> cell_area,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> inlet_cell,
           py::array_t<double, py::array::c_style | py::array::forcecast> inlet_flow,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> structure_up_cell,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> structure_down_cell,
           py::array_t<double, py::array::c_style | py::array::forcecast> structure_flow)
           -> py::array_t<double>
        {
            const int32_t n_cells = static_cast<int32_t>(cell_area.size());
            if (inlet_cell.size() != inlet_flow.size()) {
                throw std::invalid_argument("inlet_cell and inlet_flow must have the same length");
            }
            if (structure_up_cell.size() != structure_down_cell.size() ||
                structure_up_cell.size() != structure_flow.size()) {
                throw std::invalid_argument(
                    "structure_up_cell, structure_down_cell, and structure_flow must have the same length");
            }

            auto out = py::array_t<double>(n_cells);
            swe2d_gpu_compute_coupling_sources(
                nullptr,  // dev: nullptr uses static-cache fallback
                n_cells,
                (n_cells > 0) ? cell_area.data() : nullptr,
                static_cast<int32_t>(inlet_cell.size()),
                inlet_cell.size() ? inlet_cell.data() : nullptr,
                inlet_flow.size() ? inlet_flow.data() : nullptr,
                static_cast<int32_t>(structure_up_cell.size()),
                structure_up_cell.size() ? structure_up_cell.data() : nullptr,
                structure_down_cell.size() ? structure_down_cell.data() : nullptr,
                structure_flow.size() ? structure_flow.data() : nullptr,
                out.mutable_data());
            return out;
        },
        py::arg("cell_area"),
        py::arg("inlet_cell"),
        py::arg("inlet_flow"),
        py::arg("structure_up_cell"),
        py::arg("structure_down_cell"),
        py::arg("structure_flow"),
        "Headless CUDA helper: convert inlet/structure transfer flows to per-cell depth-rate sources [m/s].");

    m.def("swe2d_gpu_compute_bridge_coupling_sources",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> cell_area,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> bridge_up_cell,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> bridge_down_cell,
           py::array_t<double, py::array::c_style | py::array::forcecast> bridge_flow,
           py::array_t<double, py::array::c_style | py::array::forcecast> bridge_loss_k_upstream,
           py::array_t<double, py::array::c_style | py::array::forcecast> bridge_loss_k_downstream,
           double bridge_opening_width,
           double dt_s) -> py::array_t<double>
        {
            const int32_t n_cells = static_cast<int32_t>(cell_area.size());
            if (bridge_up_cell.size() != bridge_down_cell.size() ||
                bridge_up_cell.size() != bridge_flow.size() ||
                bridge_up_cell.size() != bridge_loss_k_upstream.size() ||
                bridge_up_cell.size() != bridge_loss_k_downstream.size()) {
                throw std::invalid_argument(
                    "bridge_up_cell, bridge_down_cell, bridge_flow, bridge_loss_k_upstream, and bridge_loss_k_downstream must have the same length");
            }

            auto out = py::array_t<double>(n_cells);
            swe2d_gpu_compute_bridge_coupling_sources(
                nullptr,  // dev: nullptr uses static-cache fallback
                n_cells,
                (n_cells > 0) ? cell_area.data() : nullptr,
                static_cast<int32_t>(bridge_up_cell.size()),
                bridge_up_cell.size() ? bridge_up_cell.data() : nullptr,
                bridge_down_cell.size() ? bridge_down_cell.data() : nullptr,
                bridge_flow.size() ? bridge_flow.data() : nullptr,
                bridge_loss_k_upstream.size() ? bridge_loss_k_upstream.data() : nullptr,
                bridge_loss_k_downstream.size() ? bridge_loss_k_downstream.data() : nullptr,
                bridge_opening_width,
                dt_s,
                out.mutable_data());
            return out;
        },
        py::arg("cell_area"),
        py::arg("bridge_up_cell"),
        py::arg("bridge_down_cell"),
        py::arg("bridge_flow"),
        py::arg("bridge_loss_k_upstream"),
        py::arg("bridge_loss_k_downstream"),
        py::arg("bridge_opening_width") = 1.0,
        py::arg("dt_s") = 1.0,
        "Headless CUDA helper: convert bridge transfer flows to per-cell depth-rate sources [m/s] with an empirical loss law.");

#ifdef HYDRA_HAS_CUDA
    m.def("swe2d_gpu_redistribute_structure_sources",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> source_rate_inout,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> dist_offsets,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> dist_cell_idx,
           py::array_t<double, py::array::c_style | py::array::forcecast> dist_weights,
           py::array_t<double, py::array::c_style | py::array::forcecast> struct_flow,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> orig_up_cell,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> orig_dn_cell,
           py::array_t<double, py::array::c_style | py::array::forcecast> cell_area) -> py::array_t<double>
        {
            auto src = source_rate_inout;
            const int32_t n_cells = static_cast<int32_t>(src.size());
            const int32_t n_struct = static_cast<int32_t>(struct_flow.size());

            // Build contiguous host arrays
            auto offsets_host = dist_offsets;
            auto cell_idx_host = dist_cell_idx;
            auto weights_host = dist_weights;
            auto flow_host = struct_flow;
            auto up_host = orig_up_cell;
            auto dn_host = orig_dn_cell;
            auto area_host = cell_area;

            extern SWE2DDeviceState* s_coupling_dev;
            swe2d_gpu_redistribute_structure_sources(
                s_coupling_dev,  // persistent device buffers when available
                n_struct,
                flow_host.data(),
                up_host.data(),
                dn_host.data(),
                area_host.data(),
                offsets_host.data(),
                cell_idx_host.data(),
                weights_host.data(),
                n_cells,
                src.mutable_data());
            return src;
        },
        py::arg("source_rate_inout"),
        py::arg("dist_offsets"),
        py::arg("dist_cell_idx"),
        py::arg("dist_weights"),
        py::arg("struct_flow"),
        py::arg("orig_up_cell"),
        py::arg("orig_dn_cell"),
        py::arg("cell_area"),
        "CUDA helper: redistribute single-cell structure sources across a pre-computed corridor of cells using influence-width weights.");

    m.def("swe2d_gpu_redistribute_structure_sources_persistent",
        [](py::array_t<int32_t, py::array::c_style | py::array::forcecast> dist_offsets,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> dist_cell_idx,
           py::array_t<double, py::array::c_style | py::array::forcecast> dist_weights,
           py::array_t<double, py::array::c_style | py::array::forcecast> struct_flow,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> orig_up_cell,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> orig_dn_cell,
           int32_t n_cells,
           double unit_to_si_factor) -> void
        {
            const int32_t n_struct = static_cast<int32_t>(struct_flow.size());

            auto offsets_host = dist_offsets;
            auto cell_idx_host = dist_cell_idx;
            auto weights_host = dist_weights;
            auto flow_host = struct_flow;
            auto up_host = orig_up_cell;
            auto dn_host = orig_dn_cell;

            extern SWE2DDeviceState* s_coupling_dev;
            if (!s_coupling_dev) {
                pybind11::set_error(PyExc_RuntimeError, "s_coupling_dev is null");
                return;
            }
            swe2d_gpu_redistribute_structure_sources_persistent(
                s_coupling_dev,
                n_struct,
                flow_host.data(),
                up_host.data(),
                dn_host.data(),
                offsets_host.data(),
                cell_idx_host.data(),
                weights_host.data(),
                n_cells,
                unit_to_si_factor);
        },
        py::arg("dist_offsets"),
        py::arg("dist_cell_idx"),
        py::arg("dist_weights"),
        py::arg("struct_flow"),
        py::arg("orig_up_cell"),
        py::arg("orig_dn_cell"),
        py::arg("n_cells"),
        py::arg("unit_to_si_factor"),
        "CUDA helper (device-only): redistribute single-cell structure sources "
        "directly on d_external_source_mps with no host readback.  Call after "
        "swe2d_gpu_compute_coupling_full_on_device then return None from Python "
        "to keep GPU sources current.");

    m.def("swe2d_gpu_redistribute_face_flux",
        [](py::array_t<int32_t, py::array::c_style | py::array::forcecast> struct_idx,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> donor_cell,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> receiver_cell,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> dist_offsets,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> dist_cell_idx,
           py::array_t<double, py::array::c_style | py::array::forcecast> dist_weights,
           int32_t n_cells) -> void
        {
            const int32_t n_faces = static_cast<int32_t>(struct_idx.size());

            extern SWE2DDeviceState* s_coupling_dev;
            if (!s_coupling_dev) {
                pybind11::set_error(PyExc_RuntimeError, "s_coupling_dev is null");
                return;
            }
            swe2d_gpu_redistribute_face_flux(
                s_coupling_dev,
                n_faces,
                struct_idx.data(),
                donor_cell.data(),
                receiver_cell.data(),
                dist_offsets.data(),
                dist_cell_idx.data(),
                dist_weights.data(),
                n_cells);
        },
        py::arg("struct_idx"),
        py::arg("donor_cell"),
        py::arg("receiver_cell"),
        py::arg("dist_offsets"),
        py::arg("dist_cell_idx"),
        py::arg("dist_weights"),
        py::arg("n_cells"),
        "CUDA helper (device-only): redistribute face-flux culvert mass "
        "directly on d_ext_struct_flux_h with no host readback.  Eliminates "
        "the PCIe transfers and Python loop that were the face-flux bottleneck.");
#endif

    // ── 1D pipe network ────────────────────────────────────────────────────────
    m.def("swe2d_build_pipe1d_mesh",
        [](int32_t n_links,
           py::array_t<int32_t, py::array::c_style|py::array::forcecast> link_from_node,
           py::array_t<int32_t, py::array::c_style|py::array::forcecast> link_to_node,
           py::array_t<double, py::array::c_style|py::array::forcecast> link_length,
           py::array_t<double, py::array::c_style|py::array::forcecast> link_diameter,
           py::array_t<double, py::array::c_style|py::array::forcecast> link_roughness_n,
           py::array_t<double, py::array::c_style|py::array::forcecast> link_inlet_loss_k,
           py::array_t<double, py::array::c_style|py::array::forcecast> link_outlet_loss_k,
           py::array_t<double, py::array::c_style|py::array::forcecast> node_invert_elev,
           py::array_t<double, py::array::c_style|py::array::forcecast> node_surface_area,
           py::array_t<double, py::array::c_style|py::array::forcecast> node_max_depth,
           py::array_t<double, py::array::c_style|py::array::forcecast> link_invert_in,
           py::array_t<double, py::array::c_style|py::array::forcecast> link_invert_out,
           int32_t max_cell_length,
           uintptr_t dev_ptr,
           // Optional shape arrays (default empty = all circular)
           py::array_t<int32_t, py::array::c_style|py::array::forcecast> link_shape_type =
               py::array_t<int32_t>(),
           py::array_t<double, py::array::c_style|py::array::forcecast> link_width =
               py::array_t<double>(),
           py::array_t<double, py::array::c_style|py::array::forcecast> link_height =
               py::array_t<double>()) -> void
        {
            auto* dev = reinterpret_cast<SWE2DDeviceState*>(dev_ptr);
            const int32_t* shape_ptr = nullptr;
            const double* w_ptr = nullptr;
            const double* h_ptr = nullptr;
            if (link_shape_type.size() > 0) shape_ptr = link_shape_type.data();
            if (link_width.size() > 0)      w_ptr   = link_width.data();
            if (link_height.size() > 0)     h_ptr   = link_height.data();
            swe2d_build_pipe1d_mesh(n_links,
                link_from_node.data(), link_to_node.data(),
                link_length.data(), link_diameter.data(), link_roughness_n.data(),
                link_inlet_loss_k.data(), link_outlet_loss_k.data(),
                node_invert_elev.data(), node_surface_area.data(), node_max_depth.data(),
                link_invert_in.data(), link_invert_out.data(),
                max_cell_length,
                shape_ptr, w_ptr, h_ptr,
                &dev->pipe1d);
        },
        py::arg("n_links"), py::arg("link_from_node"), py::arg("link_to_node"),
        py::arg("link_length"), py::arg("link_diameter"), py::arg("link_roughness_n"),
        py::arg("link_inlet_loss_k"), py::arg("link_outlet_loss_k"),
        py::arg("node_invert_elev"), py::arg("node_surface_area"), py::arg("node_max_depth"),
        py::arg("link_invert_in"), py::arg("link_invert_out"),
        py::arg("max_cell_length"), py::arg("dev_ptr"),
        py::arg("link_shape_type") = py::array_t<int32_t>(),
        py::arg("link_width") = py::array_t<double>(),
        py::arg("link_height") = py::array_t<double>(),
        "Build 1D pipe network CSR topology and allocate device buffers in pipe1d state.");

    m.def("swe2d_pipe1d_step",
        [](uintptr_t dev_ptr,
           double dt,
           std::string solver_mode,
           int32_t coupling_substeps,
           int32_t implicit_iters,
           double relaxation,
           double gravity) -> void
        {
            auto* dev = reinterpret_cast<SWE2DDeviceState*>(dev_ptr);
            swe2d_pipe1d_step(dev, dt, solver_mode.c_str(), coupling_substeps,
                              implicit_iters, relaxation, gravity);
        },
        py::arg("dev_ptr"),
        py::arg("dt"),
        py::arg("solver_mode"),
        py::arg("coupling_substeps"),
        py::arg("implicit_iters"),
        py::arg("relaxation"),
        py::arg("gravity"),
        "Advance 1D pipe network one coupling step using GPU kernels.");

    m.def("swe2d_pipe1d_upload_node_depth",
        [](uintptr_t dev_ptr,
           py::array_t<double, py::array::c_style|py::array::forcecast> node_depth) -> void
        {
            auto* dev = reinterpret_cast<SWE2DDeviceState*>(dev_ptr);
            auto info = node_depth.request();
            swe2d_pipe1d_upload_node_depth(dev, static_cast<const double*>(info.ptr),
                                           static_cast<int32_t>(info.shape[0]));
        },
        py::arg("dev_ptr"),
        py::arg("node_depth"),
        "Upload node depths from host to device before each pipe step.");

    m.def("swe2d_pipe1d_init_area_from_depth",
        [](uintptr_t dev_ptr) -> void {
            auto* dev = reinterpret_cast<SWE2DDeviceState*>(dev_ptr);
            swe2d_pipe1d_init_area_from_depth(&dev->pipe1d);
        },
        py::arg("dev_ptr"),
        "Initialize pipe cell area from uploaded node depths.");

    m.def("swe2d_pipe1d_readback_node_state",
        [](uintptr_t dev_ptr, int32_t n_nodes, int32_t n_cells) -> py::dict
        {
            auto* dev = reinterpret_cast<SWE2DDeviceState*>(dev_ptr);
            py::array_t<double> node_depth_arr(n_nodes);
            py::array_t<double> cell_A_arr(n_cells);
            py::array_t<double> cell_Q_arr(n_cells);
            swe2d_pipe1d_readback_node_state(
                dev,
                node_depth_arr.mutable_data(),
                cell_A_arr.mutable_data(),
                cell_Q_arr.mutable_data(),
                n_nodes, n_cells);
            py::dict d;
            d["node_depth"] = node_depth_arr;
            d["cell_A"] = cell_A_arr;
            d["cell_Q"] = cell_Q_arr;
            return d;
        },
        py::arg("dev_ptr"),
        py::arg("n_nodes"),
        py::arg("n_cells"),
        "Readback pipe1d node depths, cell areas, and cell flows for diagnostics/tests.");

    m.def("swe2d_solver_get_device_capsule",
        [](const std::shared_ptr<PySolver>& ps) -> py::object {
            if (!ps || !ps->solver) throw std::invalid_argument("null solver handle");
            if (!ps->solver->dev) throw std::runtime_error("GPU not initialized");
            void* dev_ptr = ps->solver->dev;
            return py::capsule(dev_ptr, "SWE2DDeviceState*",
                [](void*) { /* no-op: solver owns device lifetime */ });
        },
        py::arg("solver"),
        "Return a PyCapsule wrapping the solver's SWE2DDeviceState*.\n"
        "Pass the result to swe2d_gpu_enable_kernel_graphs / _destroy.");

    m.def("swe2d_gpu_enable_kernel_graphs",
        [](py::object dev_capsule, bool enable) {
            auto dev = static_cast<SWE2DDeviceState*>(PyCapsule_GetPointer(dev_capsule.ptr(), "SWE2DDeviceState*"));
            if (!dev) throw std::runtime_error("Invalid device pointer");
            swe2d_gpu_enable_kernel_graphs(dev, enable);
        },
        py::arg("dev"),
        py::arg("enable"),
        "Enable or disable CUDA graph optimization for kernel sequence replay.");

    m.def("swe2d_gpu_destroy_kernel_graphs",
        [](py::object dev_capsule) {
            auto dev = static_cast<SWE2DDeviceState*>(PyCapsule_GetPointer(dev_capsule.ptr(), "SWE2DDeviceState*"));
            if (!dev) throw std::runtime_error("Invalid device pointer");
            swe2d_gpu_destroy_kernel_graphs(dev);
        },
        py::arg("dev"),
        "Destroy cached CUDA graph resources for this solver instance.");

    m.def("swe2d_gpu_invalidate_graph_cache",
        []() {
            swe2d_gpu_invalidate_graph_cache(nullptr);
        },
        "Invalidate cached CUDA graph so the solver re-captures on the next step. "
        "Uses s_coupling_dev internally — call after coupling changes "
        "use_culvert_face_flux to force a cache miss on the next solver step.");


    // ── Mesh builder (legacy triangular triplets) ───────────────────────────
    // swe2d_build_mesh(node_x, node_y, node_z, cell_nodes,
    //                  bc_edge_node0, bc_edge_node1, bc_edge_type, bc_edge_val)
    // Returns an opaque PyMesh handle.
    m.def("swe2d_build_mesh",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> node_x,
           py::array_t<double, py::array::c_style | py::array::forcecast> node_y,
           py::array_t<double, py::array::c_style | py::array::forcecast> node_z,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> cell_nodes,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> bc_node0,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> bc_node1,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> bc_type,
           py::array_t<double, py::array::c_style | py::array::forcecast> bc_val)
           -> std::shared_ptr<PyMesh>
        {
            if (node_x.size() != node_y.size() || node_x.size() != node_z.size()) {
                throw std::invalid_argument("node_x, node_y, node_z must have the same length");
            }
            int32_t n_nodes = static_cast<int32_t>(node_x.size());
            if (cell_nodes.size() % 3 != 0) {
                throw std::invalid_argument("cell_nodes length must be a multiple of 3");
            }
            int32_t n_cells = static_cast<int32_t>(cell_nodes.size() / 3);

            int32_t n_bc = static_cast<int32_t>(bc_node0.size());
            if (bc_node1.size() != static_cast<size_t>(n_bc) ||
                bc_type.size()  != static_cast<size_t>(n_bc) ||
                bc_val.size()   != static_cast<size_t>(n_bc)) {
                throw std::invalid_argument(
                    "bc_node0, bc_node1, bc_type, bc_val must all have the same length");
            }

            auto pm = std::make_shared<PyMesh>();
            pm->mesh = swe2d_build_mesh(
                node_x.data(), node_y.data(), node_z.data(), n_nodes,
                cell_nodes.data(), n_cells,
                n_bc > 0 ? bc_node0.data() : nullptr,
                n_bc > 0 ? bc_node1.data() : nullptr,
                n_bc > 0 ? bc_type.data()  : nullptr,
                n_bc > 0 ? bc_val.data()   : nullptr,
                n_bc);

            std::string err = swe2d_validate_mesh(pm->mesh);
            if (!err.empty()) {
                throw std::runtime_error("Mesh validation failed: " + err);
            }

            return pm;
        },
        py::arg("node_x"), py::arg("node_y"), py::arg("node_z"),
        py::arg("cell_nodes"),
        py::arg("bc_edge_node0"), py::arg("bc_edge_node1"),
        py::arg("bc_edge_type"),  py::arg("bc_edge_val"),
        "Build an unstructured triangular mesh from node and element arrays.\n\n"
        "Parameters\n----------\n"
        "node_x, node_y, node_z : ndarray float64, shape (N,)\n"
        "    Node coordinates and bed elevations.\n"
        "cell_nodes : ndarray int32, shape (M*3,) or (M,3)\n"
        "    Counter-clockwise node triplets per cell.\n"
        "bc_edge_node0, bc_edge_node1 : ndarray int32, shape (E,)\n"
        "    Endpoint node indices for each boundary edge specification.\n"
        "bc_edge_type : ndarray int32, shape (E,)\n"
        "    BCType value per boundary edge (0=INTERIOR,1=WALL,2=INFLOW_Q,\n"
        "    3=STAGE,4=OPEN,5=REFLECT).\n"
        "bc_edge_val : ndarray float64, shape (E,)\n"
        "    Prescribed value per boundary edge.\n"
        "Returns\n-------\n"
        "PyMesh handle (opaque).\n");

    // ── Mesh builder (polygon CSR) ──────────────────────────────────────────
    // swe2d_build_mesh_poly(node_x, node_y, node_z,
    //                      cell_face_offsets, cell_face_nodes,
    //                      bc_edge_node0, bc_edge_node1, bc_edge_type, bc_edge_val)
    m.def("swe2d_build_mesh_poly",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> node_x,
           py::array_t<double, py::array::c_style | py::array::forcecast> node_y,
           py::array_t<double, py::array::c_style | py::array::forcecast> node_z,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> cell_face_offsets,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> cell_face_nodes,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> bc_node0,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> bc_node1,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> bc_type,
           py::array_t<double, py::array::c_style | py::array::forcecast> bc_val)
           -> std::shared_ptr<PyMesh>
        {
            if (node_x.size() != node_y.size() || node_x.size() != node_z.size()) {
                throw std::invalid_argument("node_x, node_y, node_z must have the same length");
            }
            if (cell_face_offsets.size() < 2) {
                throw std::invalid_argument("cell_face_offsets must have at least 2 entries");
            }

            int32_t n_nodes = static_cast<int32_t>(node_x.size());
            int32_t n_cells = static_cast<int32_t>(cell_face_offsets.size() - 1);
            int32_t n_face_nodes = static_cast<int32_t>(cell_face_nodes.size());

            int32_t n_bc = static_cast<int32_t>(bc_node0.size());
            if (bc_node1.size() != static_cast<size_t>(n_bc) ||
                bc_type.size()  != static_cast<size_t>(n_bc) ||
                bc_val.size()   != static_cast<size_t>(n_bc)) {
                throw std::invalid_argument(
                    "bc_node0, bc_node1, bc_type, bc_val must all have the same length");
            }

            int32_t tail = cell_face_offsets.data()[n_cells];
            if (tail != n_face_nodes) {
                throw std::invalid_argument(
                    "cell_face_offsets tail must equal len(cell_face_nodes)");
            }

            auto pm = std::make_shared<PyMesh>();
            pm->mesh = swe2d_build_mesh_poly(
                node_x.data(), node_y.data(), node_z.data(), n_nodes,
                cell_face_offsets.data(), cell_face_nodes.data(), n_cells,
                n_bc > 0 ? bc_node0.data() : nullptr,
                n_bc > 0 ? bc_node1.data() : nullptr,
                n_bc > 0 ? bc_type.data()  : nullptr,
                n_bc > 0 ? bc_val.data()   : nullptr,
                n_bc);

            std::string err = swe2d_validate_mesh(pm->mesh);
            if (!err.empty()) {
                throw std::runtime_error("Mesh validation failed: " + err);
            }

            return pm;
        },
        py::arg("node_x"), py::arg("node_y"), py::arg("node_z"),
        py::arg("cell_face_offsets"), py::arg("cell_face_nodes"),
        py::arg("bc_edge_node0"), py::arg("bc_edge_node1"),
        py::arg("bc_edge_type"), py::arg("bc_edge_val"),
        "Build an unstructured polygon mesh from node and CSR cell topology arrays.\n\n"
        "Parameters\n----------\n"
        "cell_face_offsets : ndarray int32, shape (M+1,)\n"
        "    CSR offsets into cell_face_nodes per cell.\n"
        "cell_face_nodes : ndarray int32, shape (K,)\n"
        "    Concatenated node rings for all polygon cells (CCW preferred).\n"
        "Returns\n-------\n"
        "PyMesh handle (opaque).\n");

    // ── Mesh info ─────────────────────────────────────────────────────────────
    m.def("swe2d_mesh_info",
        [](const std::shared_ptr<PyMesh>& pm) -> py::dict {
            if (!pm) throw std::invalid_argument("null mesh handle");
            py::dict d;
            d["n_nodes"] = pm->mesh.n_nodes;
            d["n_cells"] = pm->mesh.n_cells;
            d["n_edges"] = pm->mesh.n_edges;
            return d;
        },
        py::arg("mesh"),
        "Return dict with n_nodes, n_cells, n_edges.");

    // ── Mesh serialization (raw BLOB) ───────────────────────────────────────
    m.def("swe2d_serialize_mesh",
        [](const std::shared_ptr<PyMesh>& pm) -> py::bytes {
            if (!pm) throw std::invalid_argument("null mesh handle");
            auto blob = swe2d_serialize_mesh(pm->mesh);
            return py::bytes(reinterpret_cast<const char*>(blob.data()), blob.size());
        },
        py::arg("mesh"),
        "Serialize a fully-constructed SWE2DMesh to a raw byte BLOB suitable for\n"
        "GPKG storage.  Returns Python bytes object.\n"
        "Use swe2d_deserialize_mesh(blob) to restore.");

    m.def("swe2d_deserialize_mesh",
        [](py::bytes blob) -> std::shared_ptr<PyMesh> {
            std::string buf = static_cast<std::string>(blob);
            if (buf.size() < 12) {
                throw std::runtime_error("swe2d_deserialize_mesh: blob too small (< 12 bytes)");
            }
            auto pm = std::make_shared<PyMesh>();
            pm->mesh = swe2d_deserialize_mesh(
                reinterpret_cast<const uint8_t*>(buf.data()), buf.size());

            std::string err = swe2d_validate_mesh(pm->mesh);
            if (!err.empty()) {
                throw std::runtime_error("Deserialized mesh validation failed: " + err);
            }
            return pm;
        },
        py::arg("blob"),
        "Deserialize a SWE2DMesh from a raw byte BLOB produced by swe2d_serialize_mesh.\n"
        "Returns a fully-constructed PyMesh handle (identical to the original).");

    // ── Cell permutation (RCMK renumbering) ──────────────────────────────────
    m.def("swe2d_get_cell_perm",
        [](const std::shared_ptr<PyMesh>& pm) -> py::array_t<int32_t>
        {
            if (!pm) throw std::invalid_argument("null mesh handle");
            const auto& perm = pm->mesh.cell_perm;
            if (perm.empty()) {
                // No renumbering performed — return empty array.
                return py::array_t<int32_t>(0);
            }
            py::array_t<int32_t> arr(static_cast<py::ssize_t>(perm.size()));
            std::copy(perm.begin(), perm.end(), arr.mutable_data());
            return arr;
        },
        py::arg("mesh"),
        "Return cell_perm array where cell_perm[c_new] = c_old.  "
        "Empty if no renumbering was applied.");

    // ── Boundary edges + runtime BC updates ─────────────────────────────────
    m.def("swe2d_boundary_edges",
        [](const std::shared_ptr<PyMesh>& pm)
            -> std::tuple<py::array_t<int32_t>, py::array_t<int32_t>, py::array_t<int32_t>, py::array_t<int32_t>, py::array_t<double>, py::array_t<int32_t>>
        {
            if (!pm) throw std::invalid_argument("null mesh handle");

            std::vector<int32_t> edge_idx;
            std::vector<int32_t> n0;
            std::vector<int32_t> n1;
            std::vector<int32_t> bc_type;
            std::vector<double> bc_val;
            std::vector<int32_t> cell0;

            edge_idx.reserve(static_cast<size_t>(pm->mesh.n_edges));
            n0.reserve(static_cast<size_t>(pm->mesh.n_edges));
            n1.reserve(static_cast<size_t>(pm->mesh.n_edges));
            bc_type.reserve(static_cast<size_t>(pm->mesh.n_edges));
            bc_val.reserve(static_cast<size_t>(pm->mesh.n_edges));
            cell0.reserve(static_cast<size_t>(pm->mesh.n_edges));

            for (int32_t e = 0; e < pm->mesh.n_edges; ++e) {
                if (pm->mesh.edge_c1[e] != -1) continue;
                edge_idx.push_back(e);
                n0.push_back(pm->mesh.edge_n0[e]);
                n1.push_back(pm->mesh.edge_n1[e]);
                bc_type.push_back(static_cast<int32_t>(pm->mesh.edge_bc[e]));
                bc_val.push_back(pm->mesh.edge_bc_val[e]);
                cell0.push_back(pm->mesh.edge_c0[e]);
            }

            py::array_t<int32_t> edge_idx_arr(edge_idx.size());
            py::array_t<int32_t> n0_arr(n0.size());
            py::array_t<int32_t> n1_arr(n1.size());
            py::array_t<int32_t> bc_type_arr(bc_type.size());
            py::array_t<double> bc_val_arr(bc_val.size());
            py::array_t<int32_t> cell0_arr(cell0.size());
            std::copy(edge_idx.begin(), edge_idx.end(), edge_idx_arr.mutable_data());
            std::copy(n0.begin(), n0.end(), n0_arr.mutable_data());
            std::copy(n1.begin(), n1.end(), n1_arr.mutable_data());
            std::copy(bc_type.begin(), bc_type.end(), bc_type_arr.mutable_data());
            std::copy(bc_val.begin(), bc_val.end(), bc_val_arr.mutable_data());
            std::copy(cell0.begin(), cell0.end(), cell0_arr.mutable_data());

            return {edge_idx_arr, n0_arr, n1_arr, bc_type_arr, bc_val_arr, cell0_arr};
        },
        py::arg("mesh"),
        "Return boundary edge arrays: (edge_index, node0, node1, bc_type, bc_val).");

    m.def("swe2d_set_boundary_values",
        [](const std::shared_ptr<PyMesh>& pm,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> edge_index,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> bc_type,
           py::array_t<double, py::array::c_style | py::array::forcecast> bc_val) {
            if (!pm) throw std::invalid_argument("null mesh handle");
            if (edge_index.size() != bc_type.size() || edge_index.size() != bc_val.size()) {
                throw std::invalid_argument("edge_index, bc_type, bc_val must have same length");
            }
            for (py::ssize_t i = 0; i < edge_index.size(); ++i) {
                int32_t e = edge_index.data()[i];
                if (e < 0 || e >= pm->mesh.n_edges) {
                    throw std::invalid_argument("edge_index out of range");
                }
                if (pm->mesh.edge_c1[e] != -1) {
                    throw std::invalid_argument("edge_index refers to interior edge");
                }
                pm->mesh.edge_bc[e] = static_cast<BCType>(bc_type.data()[i]);
                pm->mesh.edge_bc_val[e] = bc_val.data()[i];
            }
        },
        py::arg("mesh"), py::arg("edge_index"), py::arg("bc_type"), py::arg("bc_val"),
        "Update boundary condition type/value for boundary edges by edge index.");

    m.def("swe2d_solver_set_boundary_values",
        [](const std::shared_ptr<PySolver>& ps,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> edge_index,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> bc_type,
           py::array_t<double, py::array::c_style | py::array::forcecast> bc_val) {
            if (!ps || !ps->solver) throw std::invalid_argument("null solver handle");
            if (edge_index.size() != bc_type.size() || edge_index.size() != bc_val.size()) {
                throw std::invalid_argument("edge_index, bc_type, bc_val must have same length");
            }
            swe2d_solver_set_boundary_values(ps->solver,
                                             edge_index.data(),
                                             bc_type.data(),
                                             bc_val.data(),
                                             static_cast<int32_t>(edge_index.size()));
        },
        py::arg("solver"), py::arg("edge_index"), py::arg("bc_type"), py::arg("bc_val"),
        "Update boundary condition values on an active solver and sync GPU arrays.");

    // ── Per-edge BC relaxation overrides ─────────────────────────────────────
    m.def("swe2d_solver_set_edge_bc_relax",
        [](std::shared_ptr<PySolver>& ps,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> edge_index,
           py::array_t<double, py::array::c_style | py::array::forcecast> relax) {
            if (!ps || !ps->solver || !ps->solver->dev)
                throw std::runtime_error("solver not initialized");
            if (edge_index.size() != relax.size())
                throw std::runtime_error("edge_index and relax must have the same length");
            if (!ps->solver->dev->d_edge_bc_relax)
                throw std::runtime_error("device relaxation array not allocated");
            const int32_t n = static_cast<int32_t>(edge_index.size());
            swe2d_gpu_set_edge_bc_relax(ps->solver->dev, edge_index.data(), relax.data(), n);
        },
        py::arg("solver"), py::arg("edge_index"), py::arg("relax"),
        "Upload per-edge relaxation overrides for boundary edges.");

    m.def("swe2d_solver_set_boundary_hydrographs",
        [](const std::shared_ptr<PySolver>& ps,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> edge_index,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> bc_type,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> offsets,
           py::array_t<double, py::array::c_style | py::array::forcecast> time_s,
           py::array_t<double, py::array::c_style | py::array::forcecast> value) {
            if (!ps || !ps->solver) throw std::invalid_argument("null solver handle");
            if (edge_index.size() != bc_type.size()) {
                throw std::invalid_argument("edge_index and bc_type must have same length");
            }
            if (offsets.size() != edge_index.size() + 1) {
                throw std::invalid_argument("offsets length must be n_edges + 1");
            }
            swe2d_solver_set_boundary_hydrographs(ps->solver,
                                                  edge_index.data(),
                                                  bc_type.data(),
                                                  offsets.data(),
                                                  time_s.data(),
                                                  value.data(),
                                                  static_cast<int32_t>(edge_index.size()),
                                                  static_cast<int32_t>(time_s.size()));
        },
        py::arg("solver"), py::arg("edge_index"), py::arg("bc_type"), py::arg("offsets"), py::arg("time_s"), py::arg("value"),
        "Register per-boundary-edge hydrograph timeseries on the solver.");

    m.def("swe2d_solver_set_progressive_bc_data",
        [](const std::shared_ptr<PySolver>& ps,
           int32_t n_groups,
           int32_t n_edges_total,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> group_offsets,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> edge_hg_idx,
           py::array_t<double, py::array::c_style | py::array::forcecast> edge_len,
           py::array_t<double, py::array::c_style | py::array::forcecast> edge_cum_len,
           py::array_t<double, py::array::c_style | py::array::forcecast> group_peak_q,
           py::array_t<double, py::array::c_style | py::array::forcecast> group_total_len) {
            if (!ps || !ps->solver) throw std::invalid_argument("null solver handle");
            swe2d_solver_set_progressive_bc_data(ps->solver,
                n_groups, n_edges_total,
                group_offsets.data(), edge_hg_idx.data(), edge_len.data(),
                edge_cum_len.data(), group_peak_q.data(), group_total_len.data());
        },
        py::arg("solver"), py::arg("n_groups"), py::arg("n_edges_total"),
        py::arg("group_offsets"), py::arg("edge_hg_idx"),
        py::arg("edge_len"), py::arg("edge_cum_len"),
        py::arg("group_peak_q"), py::arg("group_total_len"),
        "Upload progressive BC group data for on-device Q->q distribution.");

    m.def("swe2d_solver_set_rain_cn_forcing",
        [](const std::shared_ptr<PySolver>& ps,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> cell_gage_idx,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> gage_offsets,
           py::array_t<double, py::array::c_style | py::array::forcecast> hg_time_s,
           py::array_t<double, py::array::c_style | py::array::forcecast> hg_cum_mm,
           py::array_t<double, py::array::c_style | py::array::forcecast> cn,
           double ia_ratio,
           double mm_to_model_depth,
           double rain_update_interval_s) {
            if (!ps || !ps->solver) throw std::invalid_argument("null solver handle");
            if (cell_gage_idx.size() != cn.size()) {
                throw std::invalid_argument("cell_gage_idx and cn must have same length");
            }
            if (gage_offsets.size() < 2) {
                throw std::invalid_argument("gage_offsets must contain at least 2 entries");
            }
            if (hg_time_s.size() != hg_cum_mm.size()) {
                throw std::invalid_argument("hg_time_s and hg_cum_mm must have same length");
            }
            swe2d_solver_set_rain_cn_forcing(ps->solver,
                                             cell_gage_idx.data(),
                                             gage_offsets.data(),
                                             hg_time_s.data(),
                                             hg_cum_mm.data(),
                                             cn.data(),
                                             static_cast<int32_t>(cell_gage_idx.size()),
                                             static_cast<int32_t>(gage_offsets.size() - 1),
                                             static_cast<int32_t>(hg_time_s.size()),
                                             ia_ratio,
                                             mm_to_model_depth,
                                             rain_update_interval_s);
        },
        py::arg("solver"), py::arg("cell_gage_idx"), py::arg("gage_offsets"), py::arg("hg_time_s"), py::arg("hg_cum_mm"), py::arg("cn"), py::arg("ia_ratio") = 0.2, py::arg("mm_to_model_depth") = 1.0e-3, py::arg("rain_update_interval_s") = 60.0,
        "Register per-cell rain/CN forcing data on the solver.");

    m.def("swe2d_solver_set_external_sources",
        [](const std::shared_ptr<PySolver>& ps,
           py::object external_source_obj) {
            if (!ps || !ps->solver) throw std::invalid_argument("null solver handle");
            if (external_source_obj.is_none()) {
                swe2d_solver_set_external_sources(ps->solver, nullptr, 0);
                return;
            }
            auto src = external_source_obj.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
            const int32_t nc = ps->solver->mesh->n_cells;
            if (src.size() != static_cast<size_t>(nc)) {
                throw std::invalid_argument("external_source length must equal n_cells");
            }
            swe2d_solver_set_external_sources(ps->solver, src.data(), nc);
        },
        py::arg("solver"), py::arg("external_source") = py::none(),
        "Set per-cell external depth source rates [m/s] on solver (None clears).");

    // ── Solver creation ───────────────────────────────────────────────────────
    m.def("swe2d_create_solver",
        [](std::shared_ptr<PyMesh> pm,
           py::array_t<double, py::array::c_style | py::array::forcecast> h0,
           py::object hu0_obj,
           py::object hv0_obj,
           py::object n_mann_cell_obj,
           double g, double k_mann, double n_mann, double h_min,
           double cfl, double dt_max, double dt_fixed, double dt_initial,
                  double max_inv_area,
                  double cfl_lambda_cap,
                  double momentum_cap_min_speed,
                  double momentum_cap_celerity_mult,
                  double depth_cap,
                  double max_rel_depth_increase,
                   double shallow_damping_depth,
                   double source_cfl_beta,

                  int source_max_substeps,
                  double source_rate_cap,
                  double source_depth_step_cap,
                  bool source_true_subcycling,
                  bool source_imex_split,
                  bool enable_shallow_front_recon_fallback,
                  int gpu_diag_sync_interval_steps,
                  int tiny_mode,
                  int tiny_cell_threshold,
                  int tiny_edge_threshold,
                   int tiny_wet_cell_threshold,
                   bool use_gpu, int n_threads,

               int temporal_order,
               int spatial_scheme,
                int turbulence_model,
                int bed_friction_model,
                int equation_set,
                int godunov_mode,
              bool enable_rain_module,
              bool enable_pipe_network_module,
              bool enable_hydraulic_structures,
              int degen_mode,
               double front_flux_damping,
               double open_bc_relaxation,
               bool   active_set_hysteresis,
              bool   friction_substep_enabled,
              double friction_target_courant,
              int    friction_max_substeps,
              bool   shallow_friction_correction,
              double shallow_friction_depth_alpha,
              double shallow_friction_exponent)
           -> std::shared_ptr<PySolver>
        {
            if (!pm) throw std::invalid_argument("null mesh handle");
            int32_t nc = pm->mesh.n_cells;

            if (h0.size() != static_cast<size_t>(nc)) {
                throw std::invalid_argument("h0 length must equal n_cells");
            }

            const double* hu0_ptr = nullptr;
            const double* hv0_ptr = nullptr;
            const double* n_mann_cell_ptr = nullptr;
            py::array_t<double, py::array::c_style | py::array::forcecast> hu0_arr, hv0_arr;
            py::array_t<double, py::array::c_style | py::array::forcecast> n_mann_cell_arr;

            if (!hu0_obj.is_none()) {
                hu0_arr = hu0_obj.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
                if (hu0_arr.size() != static_cast<size_t>(nc))
                    throw std::invalid_argument("hu0 length must equal n_cells");
                hu0_ptr = hu0_arr.data();
            }
            if (!hv0_obj.is_none()) {
                hv0_arr = hv0_obj.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
                if (hv0_arr.size() != static_cast<size_t>(nc))
                    throw std::invalid_argument("hv0 length must equal n_cells");
                hv0_ptr = hv0_arr.data();
            }
            if (!n_mann_cell_obj.is_none()) {
                n_mann_cell_arr = n_mann_cell_obj.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
                if (n_mann_cell_arr.size() != static_cast<size_t>(nc))
                    throw std::invalid_argument("n_mann_cell length must equal n_cells");
                n_mann_cell_ptr = n_mann_cell_arr.data();
            }

            SWE2DSolverConfig cfg;
            cfg.g         = g;
            cfg.k_mann    = k_mann;
            cfg.n_mann    = n_mann;
            cfg.h_min     = h_min;
            cfg.cfl       = cfl;
            cfg.dt_max    = dt_max;
            cfg.dt_fixed  = dt_fixed;
            cfg.dt_initial = dt_initial;
            cfg.max_inv_area = max_inv_area;
            cfg.cfl_lambda_cap = cfl_lambda_cap;
            cfg.momentum_cap_min_speed = momentum_cap_min_speed;
            cfg.momentum_cap_celerity_mult = momentum_cap_celerity_mult;
            cfg.depth_cap = depth_cap;
            cfg.max_rel_depth_increase = max_rel_depth_increase;
            cfg.shallow_damping_depth = shallow_damping_depth;
            cfg.source_cfl_beta = source_cfl_beta;
            cfg.source_max_substeps = source_max_substeps;
            cfg.source_rate_cap = source_rate_cap;
            cfg.source_depth_step_cap = source_depth_step_cap;
            cfg.source_true_subcycling = source_true_subcycling;
            cfg.source_imex_split = source_imex_split;
            cfg.enable_shallow_front_recon_fallback = enable_shallow_front_recon_fallback;
            cfg.gpu_diag_sync_interval_steps = gpu_diag_sync_interval_steps;
            cfg.tiny_mode = tiny_mode;
            cfg.tiny_cell_threshold = tiny_cell_threshold;
            cfg.tiny_edge_threshold = tiny_edge_threshold;
            cfg.tiny_wet_cell_threshold = tiny_wet_cell_threshold;
            cfg.temporal_order = temporal_order;
            cfg.spatial_scheme = spatial_scheme;
            cfg.turbulence_model = turbulence_model;
            cfg.bed_friction_model = bed_friction_model;
            cfg.equation_set = equation_set;
            cfg.godunov_mode = godunov_mode;
            cfg.enable_rain_module = enable_rain_module;
            cfg.enable_pipe_network_module = enable_pipe_network_module;
            cfg.enable_hydraulic_structures = enable_hydraulic_structures;
            cfg.use_gpu   = use_gpu;
            cfg.n_threads = n_threads;
            cfg.degen_mode = degen_mode;
            cfg.front_flux_damping = front_flux_damping;
            cfg.open_bc_relaxation = open_bc_relaxation;
            cfg.active_set_hysteresis = active_set_hysteresis;
            cfg.friction_substep_enabled = friction_substep_enabled;
            cfg.friction_target_courant = friction_target_courant;
            cfg.friction_max_substeps = friction_max_substeps;
            cfg.shallow_friction_correction = shallow_friction_correction;
            cfg.shallow_friction_depth_alpha = shallow_friction_depth_alpha;
            cfg.shallow_friction_exponent = shallow_friction_exponent;

            auto ps = std::make_shared<PySolver>();
            ps->mesh_owner = pm;
            ps->solver = swe2d_create(pm->mesh, h0.data(), hu0_ptr, hv0_ptr, n_mann_cell_ptr, cfg);
            return ps;
        },
        py::arg("mesh"),
        py::arg("h0"),
        py::arg("hu0")      = py::none(),
        py::arg("hv0")      = py::none(),
        py::arg("n_mann_cell") = py::none(),
        py::arg("g")        = 9.81,
        py::arg("k_mann")   = 1.0,
        py::arg("n_mann")   = 0.035,
        py::arg("h_min")    = 1.0e-6,
        py::arg("cfl")      = 0.45,
        py::arg("dt_max")   = 10.0,
        py::arg("dt_fixed") = -1.0,
        py::arg("dt_initial") = -1.0,
        py::arg("max_inv_area") = 1.0e6,
        py::arg("cfl_lambda_cap") = 1.0e6,
        py::arg("momentum_cap_min_speed") = 50.0,
        py::arg("momentum_cap_celerity_mult") = 20.0,
        py::arg("depth_cap") = 1.0e6,
        py::arg("max_rel_depth_increase") = 2.0,
        py::arg("shallow_damping_depth") = 1.0e-4,
        py::arg("source_cfl_beta") = 0.25,
        py::arg("source_max_substeps") = 16,
        py::arg("source_rate_cap") = 0.0,
        py::arg("source_depth_step_cap") = 0.0,
        py::arg("source_true_subcycling") = false,
        py::arg("source_imex_split") = false,
        py::arg("enable_shallow_front_recon_fallback") = true,
        py::arg("gpu_diag_sync_interval_steps") = 50,
        py::arg("tiny_mode") = 1,
        py::arg("tiny_cell_threshold") = 8000,
        py::arg("tiny_edge_threshold") = 24000,
        py::arg("tiny_wet_cell_threshold") = 2000,
        py::arg("use_gpu")  = true,
        py::arg("n_threads") = 0,
        py::arg("temporal_order") = 2,
        py::arg("spatial_scheme") = 0,
        py::arg("turbulence_model") = 0,
        py::arg("bed_friction_model") = 0,
        py::arg("equation_set") = 0,
        py::arg("godunov_mode")  = 0,
        py::arg("enable_rain_module") = false,
        py::arg("enable_pipe_network_module") = false,
        py::arg("enable_hydraulic_structures") = false,
        py::arg("degen_mode") = 0,
        py::arg("front_flux_damping") = 0.5,
        py::arg("open_bc_relaxation") = 0.0,
        py::arg("active_set_hysteresis") = true,
        py::arg("friction_substep_enabled") = true,
        py::arg("friction_target_courant") = 1.0,
        py::arg("friction_max_substeps") = 64,
        py::arg("shallow_friction_correction") = false,
        py::arg("shallow_friction_depth_alpha") = 5.0,
        py::arg("shallow_friction_exponent") = 0.4,
        "Create a 2D SWE solver.\n\n"
        "Parameters\n----------\n"
        "mesh : PyMesh handle from swe2d_build_mesh\n"
        "h0   : ndarray float64, shape (M,) — initial water depth\n"
        "hu0  : ndarray float64, shape (M,) or None — initial x-momentum\n"
        "hv0  : ndarray float64, shape (M,) or None — initial y-momentum\n"
        "Returns PySolver handle.\n");

    // ── Step ──────────────────────────────────────────────────────────────────
    m.def("swe2d_step",
        [](std::shared_ptr<PySolver>& ps, double dt_request) -> py::dict
        {
            if (!ps || !ps->solver) throw std::invalid_argument("null solver handle");
            SWE2DStepDiag diag = swe2d_step(ps->solver, dt_request);
            py::dict d;
            d["dt"]         = diag.dt;
            d["wet_cells"]  = diag.wet_cells;
            d["max_depth"]  = diag.max_depth;
            d["min_depth"]  = diag.min_depth;
            d["mass_total"] = diag.mass_total;
            d["max_courant"] = diag.max_courant;
            d["max_depth_residual"] = diag.max_depth_residual;
            d["max_wse_elev_error"] = diag.max_wse_elev_error;
            d["gpu_active"] = diag.gpu_active;
            d["gpu_graph_launches_step"] = diag.gpu_graph_launches_step;
            d["gpu_graph_launches_total"] = diag.gpu_graph_launches_total;
            d["tiny_mode_requested"] = diag.tiny_mode_requested;
            d["tiny_mode_selected"] = diag.tiny_mode_selected;
            d["tiny_mode_effective"] = diag.tiny_mode_effective;
            d["tiny_mode_fallback"] = diag.tiny_mode_fallback;
            d["tiny_active_cells_est"] = diag.tiny_active_cells_est;
            d["tiny_active_edges_est"] = diag.tiny_active_edges_est;
            d["tiny_mode_fallback_count_total"] = diag.tiny_mode_fallback_count_total;
            d["fused_path_steps_total"] = diag.fused_path_steps_total;
            d["persistent_path_steps_total"] = diag.persistent_path_steps_total;
            return d;
        },
        py::arg("solver"), py::arg("dt_request") = -1.0,
        "Advance one timestep.  Returns diagnostics dict.");

    // ── Get state ─────────────────────────────────────────────────────────────
    m.def("swe2d_get_state",
        [](const std::shared_ptr<PySolver>& ps)
            -> std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>>
        {
            if (!ps || !ps->solver) throw std::invalid_argument("null solver handle");
            int32_t nc = ps->solver->mesh->n_cells;

            auto h_out  = py::array_t<double>(nc);
            auto hu_out = py::array_t<double>(nc);
            auto hv_out = py::array_t<double>(nc);

            // swe2d_get_state routes directly device→caller when GPU is active;
            // no host mirror update — state stays device-resident.
            swe2d_get_state(ps->solver,
                h_out.mutable_data(), hu_out.mutable_data(), hv_out.mutable_data());
            return {h_out, hu_out, hv_out};
        },
        py::arg("solver"),
        "Return current (h, hu, hv) state arrays.");

    // ── Set state ─────────────────────────────────────────────────────────────
    m.def("swe2d_set_state",
        [](std::shared_ptr<PySolver>& ps,
           py::array_t<double, py::array::c_style | py::array::forcecast> h_in,
           py::array_t<double, py::array::c_style | py::array::forcecast> hu_in,
           py::array_t<double, py::array::c_style | py::array::forcecast> hv_in)
        {
            if (!ps || !ps->solver) throw std::invalid_argument("null solver handle");
            const int32_t nc = ps->solver->mesh->n_cells;
            require_array(h_in, nc, "h_in");
            require_array(hu_in, nc, "hu_in");
            require_array(hv_in, nc, "hv_in");
            swe2d_set_state(ps->solver, h_in.data(), hu_in.data(), hv_in.data());
        },
        py::arg("solver"), py::arg("h_in"), py::arg("hu_in"), py::arg("hv_in"),
        "Overwrite current (h, hu, hv) solver state arrays.");

    // ── Get max tracking ───────────────────────────────────────────
    m.def("swe2d_get_max_tracking",
        [](const std::shared_ptr<PySolver>& ps)
            -> std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>>
        {
            if (!ps || !ps->solver) throw std::invalid_argument("null solver handle");
            int32_t nc = ps->solver->mesh->n_cells;
            auto h_out  = py::array_t<double>(nc);
            auto hu_out = py::array_t<double>(nc);
            auto hv_out = py::array_t<double>(nc);
            swe2d_get_max_tracking(ps->solver,
                h_out.mutable_data(), hu_out.mutable_data(), hv_out.mutable_data());
            return {h_out, hu_out, hv_out};
        },
        py::arg("solver"),
        "Return per-cell max (h, hu, hv) across entire simulation.");

    // ── Destroy ───────────────────────────────────────────────────────────────
    m.def("swe2d_destroy",
        [](std::shared_ptr<PySolver>& ps) {
            if (ps && ps->solver) {
                swe2d_destroy(ps->solver);
                ps->solver = nullptr;
            }
        },
        py::arg("solver"),
        "Explicitly free native solver resources (also called on GC).");

    // ── Native run-to-time loop ───────────────────────────────────────────────
    m.def("swe2d_run_to_time",
        [](std::shared_ptr<PySolver> ps,
           double t_end,
           double dt_request,
           int diag_batch_size) -> py::dict
        {
            if (!ps || !ps->solver) throw std::invalid_argument("null solver handle");

            // Run the native loop without Python callbacks (callbacks would require
            // a context pointer in the C interface, which we don't have).
            // For now, we batch diagnostics and return them after completion.
            std::vector<SWE2DStepDiag> diag_batch;
            if (diag_batch_size > 0) {
                diag_batch.reserve(diag_batch_size);
            }

            SWE2DRunConfig cfg;
            cfg.t_end = t_end;
            cfg.dt_request = dt_request;
            cfg.progress_callback_interval_steps = 0;  // No Python callbacks
            cfg.progress_cb = nullptr;
            cfg.diag_batch_size = diag_batch_size;
            cfg.progress_callback_interval_steps = 0;
            cfg.progress_cb = nullptr;

            // Allocate temp array for diagnostics if batching enabled.
            std::vector<SWE2DStepDiag> temp_diag_array;
            if (diag_batch_size > 0) {
                temp_diag_array.resize(diag_batch_size);
            }

            int32_t result = swe2d_run_to_time(
                ps->solver,
                &cfg,
                temp_diag_array.size() > 0 ? temp_diag_array.data() : nullptr,
                static_cast<int32_t>(temp_diag_array.size()));

            // Convert diagnostics to Python list.
            py::list diag_list;
            if (result > 0) {
                for (int32_t i = 0; i < result; ++i) {
                    const SWE2DStepDiag& d = temp_diag_array[i];
                    py::dict d_dict;
                    d_dict["dt"] = d.dt;
                    d_dict["wet_cells"] = static_cast<int32_t>(d.wet_cells);
                    d_dict["max_depth"] = d.max_depth;
                    d_dict["min_depth"] = d.min_depth;
                    d_dict["mass_total"] = d.mass_total;
                    d_dict["max_courant"] = d.max_courant;
                    d_dict["max_depth_residual"] = d.max_depth_residual;
                    d_dict["max_wse_elev_error"] = d.max_wse_elev_error;
                    d_dict["gpu_active"] = d.gpu_active;
                    d_dict["gpu_graph_launches_step"] = static_cast<int32_t>(d.gpu_graph_launches_step);
                    d_dict["tiny_mode_requested"] = d.tiny_mode_requested;
                    d_dict["tiny_mode_selected"] = d.tiny_mode_selected;
                    d_dict["tiny_mode_effective"] = d.tiny_mode_effective;
                    d_dict["tiny_mode_fallback"] = d.tiny_mode_fallback;
                    d_dict["tiny_active_cells_est"] = d.tiny_active_cells_est;
                    d_dict["tiny_active_edges_est"] = d.tiny_active_edges_est;
                    d_dict["tiny_mode_fallback_count_total"] = d.tiny_mode_fallback_count_total;
                    d_dict["fused_path_steps_total"] = d.fused_path_steps_total;
                    d_dict["persistent_path_steps_total"] = d.persistent_path_steps_total;
                    diag_list.append(d_dict);
                }
            }

            py::dict ret;
            ret["diags"] = diag_list;
            ret["steps_completed"] = static_cast<int32_t>(std::abs(result));
            ret["cancelled"] = (result < 0);
            ret["final_time"] = ps->solver->t;
            return ret;
        },
        py::arg("solver"),
        py::arg("t_end"),
        py::arg("dt_request") = -1.0,
        py::arg("diag_batch_size") = 0,
        "Run simulation natively from current time to t_end. Returns dict with 'diags', "
        "'steps_completed', 'cancelled', 'final_time'.");

    // ─────────────────────────────────────────────────────────────────────────────
    // Phase 7: 2D-3D interface contract API
    // ─────────────────────────────────────────────────────────────────────────────

    // Wrapper class for contract handle (Python GC owns lifetime)
    struct PyContractHandle {
        SWE2D3DInterfaceContractHost host_contract;
        // Device contract (if uploaded) is managed by solver, not this handle.
    };

    // Create contract from arrays (validates and deep-copies).
    m.def("swe2d_contract_create",
        [](py::array_t<int32_t, py::array::c_style | py::array::forcecast> cell2d,
           py::array_t<double, py::array::c_style | py::array::forcecast> face_area,
           py::array_t<double, py::array::c_style | py::array::forcecast> face_nx,
           py::array_t<double, py::array::c_style | py::array::forcecast> face_ny,
           py::array_t<double, py::array::c_style | py::array::forcecast> face_nz) 
            -> std::shared_ptr<PyContractHandle>
        {
            auto handle = std::make_shared<PyContractHandle>();
            
            // Copy arrays into host contract
            const int32_t* c2d_ptr = cell2d.data();
            const double* fa_ptr = face_area.data();
            const double* fnx_ptr = face_nx.data();
            const double* fny_ptr = face_ny.data();
            const double* fnz_ptr = face_nz.data();
            
            int32_t n = static_cast<int32_t>(cell2d.size());
            if (n <= 0 ||
                face_area.size() != n ||
                face_nx.size() != n ||
                face_ny.size() != n ||
                face_nz.size() != n) {
                throw std::invalid_argument(
                    "swe2d_contract_create: all arrays must have same length > 0");
            }
            
            handle->host_contract.cell2d.assign(c2d_ptr, c2d_ptr + n);
            handle->host_contract.face_area.assign(fa_ptr, fa_ptr + n);
            handle->host_contract.face_nx.assign(fnx_ptr, fnx_ptr + n);
            handle->host_contract.face_ny.assign(fny_ptr, fny_ptr + n);
            handle->host_contract.face_nz.assign(fnz_ptr, fnz_ptr + n);
            
            return handle;
        },
        py::arg("cell2d"), py::arg("face_area"), py::arg("face_nx"),
        py::arg("face_ny"), py::arg("face_nz"),
        "Create a 2D-3D interface contract from numpy arrays. Arrays must all have same length.");

    // Validate contract before upload.
    m.def("swe2d_contract_is_valid",
        [](const std::shared_ptr<PyContractHandle>& contract) -> bool
        {
            if (!contract) return false;
            return swe2d_contract_is_valid(contract->host_contract);
        },
        py::arg("contract"),
        "Validate contract consistency (all arrays same length, non-empty).");

    // ── PyMesh / PySolver as opaque Python types ──────────────────────────────
    py::class_<PyMesh, std::shared_ptr<PyMesh>>(m, "SWE2DMeshHandle")
        .def("__repr__", [](const PyMesh& pm) {
            return "<SWE2DMeshHandle nodes=" + std::to_string(pm.mesh.n_nodes)
                 + " cells=" + std::to_string(pm.mesh.n_cells)
                 + " edges=" + std::to_string(pm.mesh.n_edges) + ">";
        })
        // ── Python accessor properties for post-hoc line resampling ─────
        .def_property_readonly("node_x", [](const PyMesh& pm) {
            return py::array_t<double>(
                static_cast<py::ssize_t>(pm.mesh.node_x.size()),
                pm.mesh.node_x.data(),
                py::cast(pm));
        })
        .def_property_readonly("node_y", [](const PyMesh& pm) {
            return py::array_t<double>(
                static_cast<py::ssize_t>(pm.mesh.node_y.size()),
                pm.mesh.node_y.data(),
                py::cast(pm));
        })
        .def_property_readonly("node_z", [](const PyMesh& pm) {
            return py::array_t<double>(
                static_cast<py::ssize_t>(pm.mesh.node_z.size()),
                pm.mesh.node_z.data(),
                py::cast(pm));
        })
        .def_property_readonly("cell_face_offsets", [](const PyMesh& pm) -> py::object {
            if (pm.mesh.cell_face_offsets.empty()) return py::none();
            return py::array_t<int32_t>(
                static_cast<py::ssize_t>(pm.mesh.cell_face_offsets.size()),
                pm.mesh.cell_face_offsets.data(),
                py::cast(pm));
        })
        .def_property_readonly("cell_face_nodes", [](const PyMesh& pm) -> py::object {
            if (pm.mesh.cell_face_nodes.empty()) return py::none();
            return py::array_t<int32_t>(
                static_cast<py::ssize_t>(pm.mesh.cell_face_nodes.size()),
                pm.mesh.cell_face_nodes.data(),
                py::cast(pm));
        })
        .def_property_readonly("cell_nodes", [](const PyMesh& pm) -> py::object {
            // For triangular meshes (3 vertices per cell), return (M*3,) array.
            // For polygon meshes, return None (use cell_face_offsets + cell_face_nodes).
            const auto& offs = pm.mesh.cell_face_offsets;
            if (offs.empty() || pm.mesh.n_cells <= 0) return py::none();
            // Check if all cells have exactly 3 vertices (triangular mesh).
            bool all_tri = true;
            for (int32_t c = 0; c < pm.mesh.n_cells; ++c) {
                if (offs[static_cast<size_t>(c) + 1] - offs[static_cast<size_t>(c)] != 3) {
                    all_tri = false;
                    break;
                }
            }
            if (!all_tri) return py::none();
            return py::array_t<int32_t>(
                static_cast<py::ssize_t>(pm.mesh.cell_face_nodes.size()),
                pm.mesh.cell_face_nodes.data(),
                py::cast(pm));
        })
        .def_property_readonly("cell_zb", [](const PyMesh& pm) {
            return py::array_t<double>(
                static_cast<py::ssize_t>(pm.mesh.cell_zb.size()),
                pm.mesh.cell_zb.data(),
                py::cast(pm));
        })
        .def_property_readonly("cell_cx", [](const PyMesh& pm) {
            return py::array_t<double>(
                static_cast<py::ssize_t>(pm.mesh.cell_cx.size()),
                pm.mesh.cell_cx.data(),
                py::cast(pm));
        })
        .def_property_readonly("cell_cy", [](const PyMesh& pm) {
            return py::array_t<double>(
                static_cast<py::ssize_t>(pm.mesh.cell_cy.size()),
                pm.mesh.cell_cy.data(),
                py::cast(pm));
        })
        .def_property_readonly("cell_area", [](const PyMesh& pm) {
            return py::array_t<double>(
                static_cast<py::ssize_t>(pm.mesh.cell_area.size()),
                pm.mesh.cell_area.data(),
                py::cast(pm));
        })
        .def_property_readonly("cell_inv_area", [](const PyMesh& pm) {
            return py::array_t<double>(
                static_cast<py::ssize_t>(pm.mesh.cell_inv_area.size()),
                pm.mesh.cell_inv_area.data(),
                py::cast(pm));
        })
        .def_property_readonly("cell_perm", [](const PyMesh& pm) -> py::object {
            if (pm.mesh.cell_perm.empty()) return py::none();
            return py::array_t<int32_t>(
                static_cast<py::ssize_t>(pm.mesh.cell_perm.size()),
                pm.mesh.cell_perm.data(),
                py::cast(pm));
        })
        // BC edge arrays — needed by workbench for boundary condition overrides
        .def_property_readonly("edge_bc", [](const PyMesh& pm) -> py::object {
            if (pm.mesh.edge_bc.empty()) return py::none();
            return py::array_t<int32_t>(
                static_cast<py::ssize_t>(pm.mesh.edge_bc.size()),
                reinterpret_cast<const int32_t*>(pm.mesh.edge_bc.data()),
                py::cast(pm));
        })
        .def_property_readonly("edge_bc_val", [](const PyMesh& pm) -> py::object {
            if (pm.mesh.edge_bc_val.empty()) return py::none();
            return py::array_t<double>(
                static_cast<py::ssize_t>(pm.mesh.edge_bc_val.size()),
                pm.mesh.edge_bc_val.data(),
                py::cast(pm));
        })
        .def_property_readonly("edge_n0", [](const PyMesh& pm) -> py::object {
            if (pm.mesh.edge_n0.empty()) return py::none();
            return py::array_t<int32_t>(
                static_cast<py::ssize_t>(pm.mesh.edge_n0.size()),
                pm.mesh.edge_n0.data(),
                py::cast(pm));
        })
        .def_property_readonly("edge_n1", [](const PyMesh& pm) -> py::object {
            if (pm.mesh.edge_n1.empty()) return py::none();
            return py::array_t<int32_t>(
                static_cast<py::ssize_t>(pm.mesh.edge_n1.size()),
                pm.mesh.edge_n1.data(),
                py::cast(pm));
        });

    py::class_<PySolver, std::shared_ptr<PySolver>>(m, "SWE2DSolverHandle")
        .def("__repr__", [](const PySolver& ps) {
            return std::string("<SWE2DSolverHandle ") +
                   (ps.solver ? ("t=" + std::to_string(ps.solver->t)) : "destroyed") + ">";
        });

    py::class_<PyContractHandle, std::shared_ptr<PyContractHandle>>(m, "SWE2DContractHandle")
        .def("__repr__", [](const PyContractHandle& pc) {
            return "<SWE2DContractHandle n_faces=" + std::to_string(pc.host_contract.cell2d.size()) + ">";
        });

    // ── BCType constants ──────────────────────────────────────────────────────
    py::class_<BCType>(m, "BCType");
    m.attr("BC_INTERIOR") = py::int_(static_cast<int>(BCType::INTERIOR));
    m.attr("BC_WALL")     = py::int_(static_cast<int>(BCType::WALL));
    m.attr("BC_INFLOW_Q") = py::int_(static_cast<int>(BCType::INFLOW_Q));
    m.attr("BC_STAGE")    = py::int_(static_cast<int>(BCType::STAGE));
    m.attr("BC_OPEN")     = py::int_(static_cast<int>(BCType::OPEN));
    m.attr("BC_REFLECT")  = py::int_(static_cast<int>(BCType::REFLECT));
    m.attr("BC_NORMAL_DEPTH") = py::int_(static_cast<int>(BCType::NORMAL_DEPTH));
    m.attr("BC_NORMAL_DEPTH_SLOPE") = py::int_(static_cast<int>(BCType::NORMAL_DEPTH_SLOPE));
}
