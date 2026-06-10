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

static double bw2d_direct_step_upstream_energy(
    const Bw2dXsect& xsect,
    double q,
    double n_value,
    double slope,
    double length,
    double tailwater_depth,
    double* upstream_depth)
{
    if (q <= 0.0) {
        if (upstream_depth != nullptr) *upstream_depth = 0.0;
        return 0.0;
    }
    const double dc = bw2d_critical_depth(xsect, q);
    const double y_full = xsect.y_full;
    const double eps = std::max(1.0e-6, 1.0e-6 * y_full);
    const double y_ds = std::min(std::max(tailwater_depth, dc), y_full);
    const double step_depth = std::min(std::max(0.01, 0.02 * y_full), 0.05);

    if (y_ds >= y_full - eps) {
        const double sf_full = bw2d_friction_slope(xsect, q, n_value, y_full - eps);
        const double e_full = bw2d_specific_energy(xsect, q, y_full - eps);
        if (upstream_depth != nullptr) *upstream_depth = y_full - eps;
        return e_full + std::max(0.0, sf_full - slope) * length;
    }

    auto dx_to_depth = [&](double y_from, double e_from, double y_to, double* dx_out) {
        const double sf_from = bw2d_friction_slope(xsect, q, n_value, y_from);
        const double sf_to = bw2d_friction_slope(xsect, q, n_value, y_to);
        const double sf_avg = 0.5 * (sf_from + sf_to);
        const double denom = slope - sf_avg;
        if (std::abs(denom) < 1.0e-12) return false;
        const double e_to = bw2d_specific_energy(xsect, q, y_to);
        const double dx = (e_from - e_to) / denom;
        if (!std::isfinite(dx) || dx <= 0.0) return false;
        *dx_out = dx;
        return true;
    };

    double distance = 0.0;
    double y_cur = std::max(y_ds, eps);
    double e_cur = bw2d_specific_energy(xsect, q, y_cur);

    while (distance < length - 1.0e-8) {
        if (y_cur >= y_full - eps) {
            const double sf_full = bw2d_friction_slope(xsect, q, n_value, y_full - eps);
            const double rem = length - distance;
            if (upstream_depth != nullptr) *upstream_depth = y_full;
            return e_cur + std::max(0.0, sf_full - slope) * rem;
        }

        double dy = std::min(step_depth, y_full - y_cur);
        double dx = 0.0;
        bool have_step = false;
        double y_next = y_cur;
        for (int k = 0; k < 10; ++k) {
            const double y_try = std::min(y_cur + dy, y_full);
            if (dx_to_depth(y_cur, e_cur, y_try, &dx)) {
                have_step = true;
                y_next = y_try;
                break;
            }
            dy *= 0.5;
            if (dy <= eps) break;
        }

        if (!have_step) {
            const double y_super = bw2d_solve_supercritical_depth_for_energy(xsect, q, e_cur);
            if (upstream_depth != nullptr) *upstream_depth = y_super;
            return bw2d_specific_energy(xsect, q, y_super);
        }

        if (distance + dx >= length) {
            const double remaining = length - distance;
            auto g = [&](double y_target) {
                double dx_target = 0.0;
                if (!dx_to_depth(y_cur, e_cur, y_target, &dx_target)) {
                    return std::numeric_limits<double>::infinity();
                }
                return dx_target - remaining;
            };

            double a = y_cur;
            double b = y_next;
            double fa = g(a);
            double fb = g(b);
            double best_y = a;
            double best_err = std::abs(fa);
            for (int i = 1; i < 80; ++i) {
                const double ys = a + (b - a) * (static_cast<double>(i) / 79.0);
                const double err = g(ys);
                if (std::isfinite(err) && std::abs(err) < best_err) {
                    best_y = ys;
                    best_err = std::abs(err);
                }
                if (std::isfinite(fa) && std::isfinite(err) && fa * err <= 0.0) {
                    b = ys;
                    fb = err;
                    break;
                }
                fa = err;
                a = ys;
            }

            if (std::isfinite(fa) && std::isfinite(fb) && fa * fb <= 0.0) {
                double lo = a;
                double hi = b;
                double flo = g(lo);
                for (int it = 0; it < 60; ++it) {
                    const double mid = 0.5 * (lo + hi);
                    const double fmid = g(mid);
                    if (!std::isfinite(fmid)) break;
                    if (std::abs(fmid) < 1.0e-8 || std::abs(hi - lo) < eps) {
                        best_y = mid;
                        break;
                    }
                    if (flo * fmid <= 0.0) {
                        hi = mid;
                    } else {
                        lo = mid;
                        flo = fmid;
                    }
                }
            }

            if (upstream_depth != nullptr) *upstream_depth = best_y;
            return bw2d_specific_energy(xsect, q, best_y);
        }

        distance += dx;
        y_cur = y_next;
        e_cur = bw2d_specific_energy(xsect, q, y_cur);
    }

    if (upstream_depth != nullptr) *upstream_depth = y_cur;
    return e_cur;
}

static double bw2d_culvert_outlet_control_flow(
    const Bw2dXsect& xsect,
    double available_head_up,
    double tailwater_depth,
    double length,
    double slope,
    double roughness_n,
    double entrance_loss_k,
    double exit_loss_k,
    double q_hint)
{
    if (available_head_up <= 0.0) return 0.0;

    auto required_head = [&](double q) {
        if (q <= 0.0) return 0.0;
        double y_up = 0.0;
        const double e_up = bw2d_direct_step_upstream_energy(
            xsect,
            q,
            std::max(1.0e-6, roughness_n),
            std::max(1.0e-6, slope),
            std::max(1.0, length),
            std::max(0.0, tailwater_depth),
            &y_up);
        const double area = std::max(bw2d_xsect_area(xsect, bw2d_clamp(y_up, 1.0e-6, xsect.y_full)), 1.0e-9);
        const double vel = q / area;
        const double hv_loss = (std::max(0.0, entrance_loss_k) + std::max(0.0, exit_loss_k)) * vel * vel / (2.0 * BW2D_GRAVITY);
        return e_up + hv_loss;
    };

    // Illinois algorithm: secant with stalling-side damping.
    // Same robustness as bisection, converges in ~8-10 iterations.
    double q_lo = 0.0;
    double f_lo = -available_head_up;
    double q_hi = std::max(1.0, q_hint * 2.0);
    double f_hi = required_head(q_hi) - available_head_up;
    for (int br = 0; br < 12 && f_hi < 0.0; ++br) {
        q_lo = q_hi; f_lo = f_hi;
        q_hi *= 2.0;
        f_hi = required_head(q_hi) - available_head_up;
    }
    if (f_hi < 0.0) {
        return q_hi;
    }

    int side = 0;
    for (int it = 0; it < 16; ++it) {
        const double denom = f_hi - f_lo;
        if (std::abs(denom) < 1.0e-30) break;
        double q_mid = (q_lo * f_hi - q_hi * f_lo) / denom;
        if (q_mid <= q_lo || q_mid >= q_hi) {
            q_mid = 0.5 * (q_lo + q_hi);
        }
        const double f_mid = required_head(q_mid) - available_head_up;
        if (std::abs(f_mid) < 1.0e-8 * available_head_up) {
            return std::max(0.0, q_mid);
        }
        if (f_lo * f_mid < 0.0) {
            q_hi = q_mid; f_hi = f_mid;
            if (side == 1) f_lo *= 0.5;
            side = 1;
        } else {
            q_lo = q_mid; f_lo = f_mid;
            if (side == 0) f_hi *= 0.5;
            side = 0;
        }
    }
    return std::max(0.0, 0.5 * (q_lo + q_hi));
}

} // namespace

static py::array_t<double> compute_structure_flows_native(
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
    const auto* wse = cell_wse.data();
    const auto* stype = structure_type.data();
    const auto* up = upstream_cell.data();
    const auto* dn = downstream_cell.data();
    auto* qout = out.mutable_data();

    for (py::ssize_t i = 0; i < ns; ++i) {
        qout[i] = 0.0;
        const int32_t iu = up[i];
        const int32_t id = dn[i];
        if (iu < 0 || id < 0 || iu >= n_cells || id >= n_cells) continue;
        const double wu = wse[iu];
        const double wd = wse[id];
        const double crest = crest_elev.data()[i];
        const double qmax = std::isfinite(max_flow.data()[i]) ? std::max(0.0, max_flow.data()[i]) : -1.0;

        if (stype[i] == 1) {
            double q = bw2d_weir_q(wu, wd, crest, width.data()[i], coeff.data()[i]);
            if (qmax >= 0.0) q = bw2d_clamp(q, -qmax, qmax);
            qout[i] = q;
            continue;
        }
        if (stype[i] == 3) {
            const double area = std::max(0.0, opening.data()[i]) * std::max(0.0, width.data()[i]) * std::max(0.0, height.data()[i]);
            double q = bw2d_orifice_q(wu, wd, area, cd.data()[i], gravity);
            if (qmax >= 0.0) q = bw2d_clamp(q, -qmax, qmax);
            qout[i] = q;
            continue;
        }
        if (stype[i] == 4) {
            const double area = std::max(0.0, opening.data()[i]) * std::max(0.0, width.data()[i]) * std::max(0.0, height.data()[i]);
            const double loss_scale = std::max(1.0e-6, 1.0 + std::max(0.0, entrance_loss_k.data()[i]) + std::max(0.0, exit_loss_k.data()[i]));
            const double dh = wu - wd;
            if (area > 0.0 && std::abs(dh) > 1.0e-12) {
                double q = area * std::sqrt(std::max(0.0, 2.0 * gravity * std::abs(dh))) / loss_scale;
                if (qmax >= 0.0) q = std::min(q, qmax);
                qout[i] = (dh >= 0.0) ? q : -q;
            }
            continue;
        }
        if (stype[i] == 5) {
            double q = std::max(0.0, q_pump.data()[i]);
            if (qmax >= 0.0) q = std::min(q, qmax);
            qout[i] = (wu >= wd) ? q : -q;
            continue;
        }
        if (stype[i] != 2) continue;
        const double sign = (wu >= wd) ? 1.0 : -1.0;
        const double upstream_wse = (sign >= 0.0) ? wu : wd;
        const double downstream_wse = (sign >= 0.0) ? wd : wu;
        const double upstream_invert = (sign >= 0.0) ? inlet_invert_elev.data()[i] : outlet_invert_elev.data()[i];
        const double downstream_invert = (sign >= 0.0) ? outlet_invert_elev.data()[i] : inlet_invert_elev.data()[i];

        // HDS-5 culvert tables are hardcoded USC.  Convert geometry to feet.
        const double to_ft = std::max(1.0e-6, model_to_ft);
        const double available_head_up_ft = std::max(0.0, (upstream_wse - upstream_invert) * to_ft);
        const double tailwater_depth_ft = std::max(0.0, (downstream_wse - downstream_invert) * to_ft);
        const double len_ft = std::max(0.1, length.data()[i] * to_ft);
        double slope = culvert_slope.data()[i];
        if (!(slope > 0.0)) {
            slope = std::abs(upstream_invert - downstream_invert) / std::max(0.1, length.data()[i]);
        }
        slope = std::max(1.0e-6, slope);
        const double rise_ft = std::max(0.0, culvert_rise.data()[i] > 0.0 ? culvert_rise.data()[i] : std::max(height.data()[i], diameter.data()[i])) * to_ft;
        const double span_ft = std::max(0.0, culvert_span.data()[i] > 0.0 ? culvert_span.data()[i] : std::max(width.data()[i], rise_ft / to_ft)) * to_ft;
        const int code = static_cast<int>(bw2d_clamp(static_cast<double>(culvert_code.data()[i]), 1.0, static_cast<double>(BW2D_MAX_CULVERT_CODE)));

        Bw2dXsect xsect;
        xsect.code = code;
        xsect.rectangular = (culvert_shape.data()[i] == 1);
        if (xsect.rectangular) {
            xsect.width = std::max(1.0e-6, span_ft);
            xsect.y_full = std::max(1.0e-6, rise_ft);
            xsect.a_full = xsect.width * xsect.y_full;
        } else {
            const double dia_ft = std::max(1.0e-6, std::max(diameter.data()[i] * to_ft, rise_ft));
            xsect.radius = 0.5 * dia_ft;
            xsect.y_full = dia_ft;
            xsect.a_full = M_PI * xsect.radius * xsect.radius;
        }

        const double q_inlet = std::max(0.0, bw2d_inlet_controlled_flow(xsect, slope, std::max(0.0, available_head_up_ft), nullptr));

        double area_ft2 = std::max(0.0, culvert_area.data()[i]);
        if (area_ft2 <= 0.0 && std::max(diameter.data()[i], rise_ft / to_ft) > 0.0 && culvert_shape.data()[i] == 0) {
            area_ft2 = bw2d_circular_area(std::max(diameter.data()[i] * to_ft, rise_ft));
        }

        double q_orifice = 0.0;
        if (area_ft2 > 0.0) {
            q_orifice = std::abs(bw2d_orifice_q(available_head_up_ft, tailwater_depth_ft, area_ft2, cd.data()[i], BW2D_GRAVITY));
            if (qmax >= 0.0) q_orifice = std::min(q_orifice, qmax);
        }

        double q_manning_cap = 0.0;
        if (xsect.rectangular) {
            q_manning_cap = bw2d_rect_manning_capacity_full(xsect.width, xsect.y_full, slope, roughness_n.data()[i]);
        } else {
            const double dia_for_cap_ft = std::max(std::max(diameter.data()[i] * to_ft, rise_ft), bw2d_equiv_diameter_from_area(std::max(0.0, area_ft2)));
            if (dia_for_cap_ft > 0.0) {
                q_manning_cap = bw2d_pipe_manning_capacity_full(dia_for_cap_ft, slope, roughness_n.data()[i]);
            }
        }

        const double q_hint = std::max(q_inlet, std::max(q_orifice, q_manning_cap));
        const double q_outlet = bw2d_culvert_outlet_control_flow(
            xsect,
            std::max(0.0, available_head_up_ft),
            std::max(0.0, tailwater_depth_ft),
            std::max(0.1, len_ft),
            std::max(1.0e-6, slope),
            std::max(1.0e-6, roughness_n.data()[i]),
            entrance_loss_k.data()[i],
            exit_loss_k.data()[i],
            std::max(1.0, q_hint));

        double q = std::max(0.0, std::min(q_inlet, q_outlet > 0.0 ? q_outlet : q_inlet));
        if (q_orifice > 0.0) q = (q > 0.0) ? std::min(q, q_orifice) : q_orifice;
        if (q_manning_cap > 0.0) q = (q > 0.0) ? std::min(q, q_manning_cap) : q_manning_cap;

        if (embankment_enabled.data()[i] != 0) {
            const double q_emb = std::abs(
                bw2d_weir_q(
                    upstream_wse * to_ft,
                    downstream_wse * to_ft,
                    embankment_crest_elev.data()[i] * to_ft,
                    std::max(0.0, embankment_overflow_width.data()[i]) * to_ft,
                    std::max(1.0e-6, embankment_weir_coeff.data()[i])));
            q += q_emb;
        }

        q *= std::max(1.0, culvert_barrels.data()[i]);
        if (qmax >= 0.0) q = std::min(q, qmax);
        // Convert from CFS back to model units: ÷ to_ft³
        qout[i] = sign * q / (to_ft * to_ft * to_ft);
    }
    return out;
}

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
#else
    m.def("swe2d_gpu_device_sync", []() {
    }, "No-op: device sync not available without CUDA.");
#endif

    m.def("swe2d_gpu_compute_structure_flows",
        py::doc("Compute structure flow rates (weir/culvert/gate/bridge/pump) on device."),
#ifdef HYDRA_HAS_CUDA
        &compute_structure_flows_cuda,
#else
        &compute_structure_flows_native,
#endif
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

    m.def("swe2d_cpu_compute_structure_flows",
        &compute_structure_flows_native,
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
        "Compute per-structure flow transfers in model units (CPU fallback).\n"
        "Same unit convention as the GPU path.");

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
#else
    m.def("swe2d_gpu_compute_structure_and_coupling_sources",
        [](py::args) -> py::array_t<double> {
            throw std::runtime_error("CUDA path not compiled; swe2d_gpu_compute_structure_and_coupling_sources is unavailable.");
        },
        "Fused CUDA helper (unavailable without CUDA).");
#endif

#ifdef HYDRA_HAS_CUDA
    // ── Persistent GPU coupling path ──
    m.def("swe2d_gpu_set_coupling_device_global",
        [](uintptr_t dev_ptr) { swe2d_gpu_set_coupling_device_global(reinterpret_cast<SWE2DDeviceState*>(dev_ptr)); },
        py::arg("dev_ptr"), "Set global device pointer for persistent coupling.");

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
           py::array_t<int32_t, py::array::c_style|py::array::forcecast> inlet_cell,
           py::array_t<double, py::array::c_style|py::array::forcecast> inlet_flow,
           py::object host_flows_obj) {
            const double* cell_wse_ptr = nullptr;
            int32_t n_cells = 0;
            const double* host_flows_ptr = nullptr;
            int32_t n_host_flows = 0;
            if (!cell_wse_obj.is_none()) {
                auto cell_wse = cell_wse_obj.cast<py::array_t<double, py::array::c_style|py::array::forcecast>>();
                cell_wse_ptr = cell_wse.data();
                n_cells = static_cast<int32_t>(cell_wse.size());
            }
            if (!host_flows_obj.is_none()) {
                auto host_flows = host_flows_obj.cast<py::array_t<double, py::array::c_style|py::array::forcecast>>();
                host_flows_ptr = host_flows.data();
                n_host_flows = static_cast<int32_t>(host_flows.size());
            }
            swe2d_gpu_compute_coupling_full_on_device(
                nullptr, n_cells, n_structures, cell_wse_ptr,
                static_cast<int32_t>(inlet_cell.size()),
                inlet_cell.size()>0?inlet_cell.data():nullptr,
                inlet_flow.size()>0?inlet_flow.data():nullptr,
                host_flows_ptr);
        }, py::arg("cell_wse")=py::none(), py::arg("n_structures")=0,
           py::arg("inlet_cell")=py::array_t<int32_t>(),
           py::arg("inlet_flow")=py::array_t<double>(),
           py::arg("host_structure_flows")=py::none(),
        "Run full coupling on-device using preloaded params. "
        "Pass cell_wse=None to compute WSE = h + zb on GPU. "
        "Pass host_structure_flows to override GPU-computed structure flows.");

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

#ifdef HYDRA_HAS_CUDA
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
#endif

    m.def("swe2d_gpu_drainage_step",
        [](py::object cell_wse_obj,
           py::array_t<double, py::array::c_style | py::array::forcecast> cell_area,
           py::array_t<double, py::array::c_style | py::array::forcecast> node_invert_elev,
           py::array_t<double, py::array::c_style | py::array::forcecast> node_max_depth,
           py::array_t<double, py::array::c_style | py::array::forcecast> node_surface_area,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> link_from,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> link_to,
           py::array_t<double, py::array::c_style | py::array::forcecast> link_length,
           py::array_t<double, py::array::c_style | py::array::forcecast> link_roughness_n,
           py::array_t<double, py::array::c_style | py::array::forcecast> link_diameter,
           py::array_t<double, py::array::c_style | py::array::forcecast> link_max_flow,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> inlet_cell,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> inlet_node,
           py::array_t<double, py::array::c_style | py::array::forcecast> inlet_crest_elev,
           py::array_t<double, py::array::c_style | py::array::forcecast> inlet_width,
           py::array_t<double, py::array::c_style | py::array::forcecast> inlet_coefficient,
           py::array_t<double, py::array::c_style | py::array::forcecast> inlet_max_capture,
              py::array_t<int32_t, py::array::c_style | py::array::forcecast> outfall_cell,
              py::array_t<int32_t, py::array::c_style | py::array::forcecast> outfall_node,
              py::array_t<double, py::array::c_style | py::array::forcecast> outfall_invert_elev,
              py::array_t<double, py::array::c_style | py::array::forcecast> outfall_diameter,
              py::array_t<double, py::array::c_style | py::array::forcecast> outfall_coefficient,
              py::array_t<double, py::array::c_style | py::array::forcecast> outfall_max_flow,
              py::array_t<int32_t, py::array::c_style | py::array::forcecast> outfall_zero_storage,
                  py::array_t<int32_t, py::array::c_style | py::array::forcecast> pipe_end_cell,
                  py::array_t<int32_t, py::array::c_style | py::array::forcecast> pipe_end_node,
                  py::array_t<double, py::array::c_style | py::array::forcecast> pipe_end_invert_elev,
                  py::array_t<double, py::array::c_style | py::array::forcecast> pipe_end_diameter,
                  py::array_t<double, py::array::c_style | py::array::forcecast> pipe_end_area,
                  py::array_t<double, py::array::c_style | py::array::forcecast> pipe_end_inlet_loss_k,
                  py::array_t<double, py::array::c_style | py::array::forcecast> pipe_end_outlet_loss_k,
              py::object cell_depth_obj,
           py::array_t<double, py::array::c_style | py::array::forcecast> node_depth,
           py::array_t<double, py::array::c_style | py::array::forcecast> link_flow,
           double dt_s,
           double gravity,
              int32_t solver_mode,
              double head_deadband_m,
              double dynamic_flow_relaxation)
           -> py::tuple
        {
            // cell_wse and cell_depth accept None for on-device WSE computation
            const double* cell_wse_ptr = nullptr;
            const double* cell_depth_ptr = nullptr;
            int32_t n_cells = 0;

            if (cell_wse_obj.is_none()) {
                n_cells = static_cast<int32_t>(cell_area.size());
            } else {
                auto cell_wse = cell_wse_obj.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
                n_cells = static_cast<int32_t>(cell_wse.size());
                if (cell_area.size() != static_cast<size_t>(n_cells)) {
                    throw std::invalid_argument("cell_wse and cell_area must have same length");
                }
                cell_wse_ptr = cell_wse.data();
            }

            if (!cell_depth_obj.is_none()) {
                auto cell_depth_arr = cell_depth_obj.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
                if (cell_depth_arr.size() != static_cast<size_t>(n_cells)) {
                    throw std::invalid_argument("cell_depth length must match n_cells");
                }
                cell_depth_ptr = cell_depth_arr.data();
            }
            const int32_t n_nodes = static_cast<int32_t>(node_invert_elev.size());
            if (node_max_depth.size() != static_cast<size_t>(n_nodes) ||
                node_surface_area.size() != static_cast<size_t>(n_nodes) ||
                node_depth.size() != static_cast<size_t>(n_nodes)) {
                throw std::invalid_argument("node arrays must have consistent length");
            }
            const int32_t n_links = static_cast<int32_t>(link_from.size());
            if (link_to.size() != static_cast<size_t>(n_links) ||
                link_length.size() != static_cast<size_t>(n_links) ||
                link_roughness_n.size() != static_cast<size_t>(n_links) ||
                link_diameter.size() != static_cast<size_t>(n_links) ||
                link_max_flow.size() != static_cast<size_t>(n_links) ||
                link_flow.size() != static_cast<size_t>(n_links)) {
                throw std::invalid_argument("link arrays must have consistent length");
            }
            const int32_t n_inlets = static_cast<int32_t>(inlet_cell.size());
            if (inlet_node.size() != static_cast<size_t>(n_inlets) ||
                inlet_crest_elev.size() != static_cast<size_t>(n_inlets) ||
                inlet_width.size() != static_cast<size_t>(n_inlets) ||
                inlet_coefficient.size() != static_cast<size_t>(n_inlets) ||
                inlet_max_capture.size() != static_cast<size_t>(n_inlets)) {
                throw std::invalid_argument("inlet arrays must have consistent length");
            }
            const int32_t n_outfalls = static_cast<int32_t>(outfall_cell.size());
            if (outfall_node.size() != static_cast<size_t>(n_outfalls) ||
                outfall_invert_elev.size() != static_cast<size_t>(n_outfalls) ||
                outfall_diameter.size() != static_cast<size_t>(n_outfalls) ||
                outfall_coefficient.size() != static_cast<size_t>(n_outfalls) ||
                outfall_max_flow.size() != static_cast<size_t>(n_outfalls) ||
                outfall_zero_storage.size() != static_cast<size_t>(n_outfalls)) {
                throw std::invalid_argument("outfall arrays must have consistent length");
            }
            const int32_t n_pipe_ends = static_cast<int32_t>(pipe_end_cell.size());
            if (pipe_end_node.size() != static_cast<size_t>(n_pipe_ends) ||
                pipe_end_invert_elev.size() != static_cast<size_t>(n_pipe_ends) ||
                pipe_end_diameter.size() != static_cast<size_t>(n_pipe_ends) ||
                pipe_end_area.size() != static_cast<size_t>(n_pipe_ends) ||
                pipe_end_inlet_loss_k.size() != static_cast<size_t>(n_pipe_ends) ||
                pipe_end_outlet_loss_k.size() != static_cast<size_t>(n_pipe_ends)) {
                throw std::invalid_argument("pipe_end arrays must have consistent length");
            }

            auto node_depth_out = py::array_t<double>(n_nodes);
            auto link_flow_out = py::array_t<double>(n_links);
            auto q_cell_out = py::array_t<double>(n_cells);
            double max_node_depth = 0.0;
            double max_link_flow = 0.0;
            double limiter_events = 0.0;
            double limiter_volume_m3 = 0.0;

            swe2d_gpu_drainage_step(
                n_cells,
                n_nodes,
                n_links,
                n_inlets,
                n_outfalls,
                n_pipe_ends,
                cell_wse_ptr,
                cell_area.data(),
                node_invert_elev.data(),
                node_max_depth.data(),
                node_surface_area.data(),
                link_from.data(),
                link_to.data(),
                link_length.data(),
                link_roughness_n.data(),
                link_diameter.data(),
                link_max_flow.data(),
                inlet_cell.data(),
                inlet_node.data(),
                inlet_crest_elev.data(),
                inlet_width.data(),
                inlet_coefficient.data(),
                inlet_max_capture.data(),
                outfall_cell.data(),
                outfall_node.data(),
                outfall_invert_elev.data(),
                outfall_diameter.data(),
                outfall_coefficient.data(),
                outfall_max_flow.data(),
                outfall_zero_storage.data(),
                pipe_end_cell.data(),
                pipe_end_node.data(),
                pipe_end_invert_elev.data(),
                pipe_end_diameter.data(),
                pipe_end_area.data(),
                pipe_end_inlet_loss_k.data(),
                pipe_end_outlet_loss_k.data(),
                cell_depth_ptr,
                node_depth.data(),
                link_flow.data(),
                dt_s,
                gravity,
                solver_mode,
                head_deadband_m,
                dynamic_flow_relaxation,
                node_depth_out.mutable_data(),
                link_flow_out.mutable_data(),
                q_cell_out.mutable_data(),
                &max_node_depth,
                &max_link_flow,
                &limiter_events,
                &limiter_volume_m3);

            py::dict diag;
            diag["max_node_depth"] = max_node_depth;
            diag["max_link_flow"] = max_link_flow;
            diag["limiter_events"] = limiter_events;
            diag["limiter_volume_m3"] = limiter_volume_m3;
            return py::make_tuple(node_depth_out, link_flow_out, q_cell_out, diag);
        },
        py::arg("cell_wse"),
        py::arg("cell_area"),
        py::arg("node_invert_elev"),
        py::arg("node_max_depth"),
        py::arg("node_surface_area"),
        py::arg("link_from"),
        py::arg("link_to"),
        py::arg("link_length"),
        py::arg("link_roughness_n"),
        py::arg("link_diameter"),
        py::arg("link_max_flow"),
        py::arg("inlet_cell"),
        py::arg("inlet_node"),
        py::arg("inlet_crest_elev"),
        py::arg("inlet_width"),
        py::arg("inlet_coefficient"),
        py::arg("inlet_max_capture"),
        py::arg("outfall_cell"),
        py::arg("outfall_node"),
        py::arg("outfall_invert_elev"),
        py::arg("outfall_diameter"),
        py::arg("outfall_coefficient"),
        py::arg("outfall_max_flow"),
        py::arg("outfall_zero_storage"),
        py::arg("pipe_end_cell"),
        py::arg("pipe_end_node"),
        py::arg("pipe_end_invert_elev"),
        py::arg("pipe_end_diameter"),
        py::arg("pipe_end_area"),
        py::arg("pipe_end_inlet_loss_k"),
        py::arg("pipe_end_outlet_loss_k"),
        py::arg("cell_depth"),
        py::arg("node_depth"),
        py::arg("link_flow"),
        py::arg("dt_s"),
        py::arg("gravity"),
        py::arg("solver_mode"),
        py::arg("head_deadband") = 1.0e-3,
        py::arg("dynamic_flow_relaxation") = 1.0,
        "Headless CUDA helper: advance 1D drainage network one step (EGL/diffusion/dynamic).");

    m.def("swe2d_gpu_drainage_step_iterative",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> cell_bed,
           py::array_t<double, py::array::c_style | py::array::forcecast> cell_area,
           py::array_t<double, py::array::c_style | py::array::forcecast> node_invert_elev,
           py::array_t<double, py::array::c_style | py::array::forcecast> node_max_depth,
           py::array_t<double, py::array::c_style | py::array::forcecast> node_surface_area,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> link_from,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> link_to,
           py::array_t<double, py::array::c_style | py::array::forcecast> link_length,
           py::array_t<double, py::array::c_style | py::array::forcecast> link_roughness_n,
           py::array_t<double, py::array::c_style | py::array::forcecast> link_diameter,
           py::array_t<double, py::array::c_style | py::array::forcecast> link_max_flow,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> inlet_cell,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> inlet_node,
           py::array_t<double, py::array::c_style | py::array::forcecast> inlet_crest_elev,
           py::array_t<double, py::array::c_style | py::array::forcecast> inlet_width,
           py::array_t<double, py::array::c_style | py::array::forcecast> inlet_coefficient,
           py::array_t<double, py::array::c_style | py::array::forcecast> inlet_max_capture,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> outfall_cell,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> outfall_node,
           py::array_t<double, py::array::c_style | py::array::forcecast> outfall_invert_elev,
           py::array_t<double, py::array::c_style | py::array::forcecast> outfall_diameter,
           py::array_t<double, py::array::c_style | py::array::forcecast> outfall_coefficient,
           py::array_t<double, py::array::c_style | py::array::forcecast> outfall_max_flow,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> outfall_zero_storage,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> pipe_end_cell,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> pipe_end_node,
           py::array_t<double, py::array::c_style | py::array::forcecast> pipe_end_invert_elev,
           py::array_t<double, py::array::c_style | py::array::forcecast> pipe_end_diameter,
           py::array_t<double, py::array::c_style | py::array::forcecast> pipe_end_area,
           py::array_t<double, py::array::c_style | py::array::forcecast> pipe_end_inlet_loss_k,
           py::array_t<double, py::array::c_style | py::array::forcecast> pipe_end_outlet_loss_k,
           py::array_t<double, py::array::c_style | py::array::forcecast> cell_depth,
           py::array_t<double, py::array::c_style | py::array::forcecast> node_depth,
           py::array_t<double, py::array::c_style | py::array::forcecast> link_flow,
           double dt_s,
           double gravity,
           int32_t solver_mode,
           double head_deadband_m,
           double dynamic_flow_relaxation,
           int32_t n_substeps,
           int32_t implicit_iters,
           double coupling_relaxation)
           -> py::tuple
        {
            const int32_t n_cells = static_cast<int32_t>(cell_bed.size());
            if (cell_area.size() != static_cast<size_t>(n_cells) ||
                cell_depth.size() != static_cast<size_t>(n_cells)) {
                throw std::invalid_argument("cell_bed, cell_area, and cell_depth must have same length");
            }
            const int32_t n_nodes = static_cast<int32_t>(node_invert_elev.size());
            if (node_max_depth.size() != static_cast<size_t>(n_nodes) ||
                node_surface_area.size() != static_cast<size_t>(n_nodes) ||
                node_depth.size() != static_cast<size_t>(n_nodes)) {
                throw std::invalid_argument("node arrays must have consistent length");
            }
            const int32_t n_links = static_cast<int32_t>(link_from.size());
            if (link_to.size() != static_cast<size_t>(n_links) ||
                link_length.size() != static_cast<size_t>(n_links) ||
                link_roughness_n.size() != static_cast<size_t>(n_links) ||
                link_diameter.size() != static_cast<size_t>(n_links) ||
                link_max_flow.size() != static_cast<size_t>(n_links) ||
                link_flow.size() != static_cast<size_t>(n_links)) {
                throw std::invalid_argument("link arrays must have consistent length");
            }
            const int32_t n_inlets = static_cast<int32_t>(inlet_cell.size());
            if (inlet_node.size() != static_cast<size_t>(n_inlets) ||
                inlet_crest_elev.size() != static_cast<size_t>(n_inlets) ||
                inlet_width.size() != static_cast<size_t>(n_inlets) ||
                inlet_coefficient.size() != static_cast<size_t>(n_inlets) ||
                inlet_max_capture.size() != static_cast<size_t>(n_inlets)) {
                throw std::invalid_argument("inlet arrays must have consistent length");
            }
            const int32_t n_outfalls = static_cast<int32_t>(outfall_cell.size());
            if (outfall_node.size() != static_cast<size_t>(n_outfalls) ||
                outfall_invert_elev.size() != static_cast<size_t>(n_outfalls) ||
                outfall_diameter.size() != static_cast<size_t>(n_outfalls) ||
                outfall_coefficient.size() != static_cast<size_t>(n_outfalls) ||
                outfall_max_flow.size() != static_cast<size_t>(n_outfalls) ||
                outfall_zero_storage.size() != static_cast<size_t>(n_outfalls)) {
                throw std::invalid_argument("outfall arrays must have consistent length");
            }
            const int32_t n_pipe_ends = static_cast<int32_t>(pipe_end_cell.size());
            if (pipe_end_node.size() != static_cast<size_t>(n_pipe_ends) ||
                pipe_end_invert_elev.size() != static_cast<size_t>(n_pipe_ends) ||
                pipe_end_diameter.size() != static_cast<size_t>(n_pipe_ends) ||
                pipe_end_area.size() != static_cast<size_t>(n_pipe_ends) ||
                pipe_end_inlet_loss_k.size() != static_cast<size_t>(n_pipe_ends) ||
                pipe_end_outlet_loss_k.size() != static_cast<size_t>(n_pipe_ends)) {
                throw std::invalid_argument("pipe_end arrays must have consistent length");
            }

            const int32_t substeps = std::max<int32_t>(1, n_substeps);
            const int32_t iters = std::max<int32_t>(1, implicit_iters);
            const double relax = std::max(0.0, std::min(1.0, coupling_relaxation));
            const double dt_sub = dt_s / static_cast<double>(substeps);

            auto node_state_out = py::array_t<double>(n_nodes);
            auto link_state_out = py::array_t<double>(n_links);
            auto q_cell_out = py::array_t<double>(n_cells);

            std::vector<double> node_state(static_cast<size_t>(n_nodes));
            std::vector<double> link_state(static_cast<size_t>(n_links));
            std::vector<double> q_cell_acc(static_cast<size_t>(n_cells), 0.0);
            std::vector<double> q_cell_last(static_cast<size_t>(n_cells), 0.0);
            std::vector<double> hh_sub(static_cast<size_t>(n_cells));
            std::vector<double> hh_iter(static_cast<size_t>(n_cells));
            std::vector<double> hh_target(static_cast<size_t>(n_cells));
            std::vector<double> cell_wse(static_cast<size_t>(n_cells));
            std::vector<double> node_out(static_cast<size_t>(n_nodes));
            std::vector<double> link_out(static_cast<size_t>(n_links));
            std::vector<double> q_out(static_cast<size_t>(n_cells));

            std::memcpy(node_state.data(), node_depth.data(), sizeof(double) * static_cast<size_t>(n_nodes));
            std::memcpy(link_state.data(), link_flow.data(), sizeof(double) * static_cast<size_t>(n_links));
            std::memcpy(hh_sub.data(), cell_depth.data(), sizeof(double) * static_cast<size_t>(n_cells));

            // Fast path for fully inactive exchange states.
            const double tiny_h = std::max(1.0e-12, 0.1 * head_deadband_m);
            bool any_wet_surface = false;
            for (int32_t i = 0; i < n_cells; ++i) {
                if (hh_sub[static_cast<size_t>(i)] > tiny_h) {
                    any_wet_surface = true;
                    break;
                }
            }
            bool any_wet_nodes = false;
            for (int32_t i = 0; i < n_nodes; ++i) {
                if (node_state[static_cast<size_t>(i)] > tiny_h) {
                    any_wet_nodes = true;
                    break;
                }
            }
            bool any_link_flow = false;
            for (int32_t i = 0; i < n_links; ++i) {
                if (std::abs(link_state[static_cast<size_t>(i)]) > 1.0e-10) {
                    any_link_flow = true;
                    break;
                }
            }
            if (!any_wet_surface && !any_wet_nodes && !any_link_flow) {
                std::memcpy(node_state_out.mutable_data(), node_state.data(), sizeof(double) * static_cast<size_t>(n_nodes));
                std::memcpy(link_state_out.mutable_data(), link_state.data(), sizeof(double) * static_cast<size_t>(n_links));
                std::fill(q_cell_out.mutable_data(), q_cell_out.mutable_data() + n_cells, 0.0);
                py::dict diag;
                diag["max_node_depth"] = 0.0;
                diag["max_link_flow"] = 0.0;
                diag["limiter_events"] = 0.0;
                diag["limiter_volume_m3"] = 0.0;
                diag["substeps_used"] = 0.0;
                diag["implicit_iters_used"] = 0.0;
                diag["inactive_fastpath"] = 1.0;
                return py::make_tuple(node_state_out, link_state_out, q_cell_out, diag);
            }

            double max_node_depth = 0.0;
            double max_link_flow = 0.0;
            double limiter_events = 0.0;
            double limiter_volume_m3 = 0.0;

            int32_t substeps_used = 0;
            int32_t implicit_iters_used = 0;
            for (int32_t s = 0; s < substeps; ++s) {
                ++substeps_used;
                std::memcpy(hh_iter.data(), hh_sub.data(), sizeof(double) * static_cast<size_t>(n_cells));
                std::fill(q_cell_last.begin(), q_cell_last.end(), 0.0);
                double step_max_node_depth = 0.0;
                double step_max_link_flow = 0.0;
                double step_limiter_events = 0.0;
                double step_limiter_volume_m3 = 0.0;
                bool converged = false;

                for (int32_t it = 0; it < iters; ++it) {
                    ++implicit_iters_used;
                    for (int32_t i = 0; i < n_cells; ++i) {
                        cell_wse[static_cast<size_t>(i)] = cell_bed.data()[i] + hh_iter[static_cast<size_t>(i)];
                    }
                    swe2d_gpu_drainage_step(
                        n_cells,
                        n_nodes,
                        n_links,
                        n_inlets,
                        n_outfalls,
                        n_pipe_ends,
                        cell_wse.data(),
                        cell_area.data(),
                        node_invert_elev.data(),
                        node_max_depth.data(),
                        node_surface_area.data(),
                        link_from.data(),
                        link_to.data(),
                        link_length.data(),
                        link_roughness_n.data(),
                        link_diameter.data(),
                        link_max_flow.data(),
                        inlet_cell.data(),
                        inlet_node.data(),
                        inlet_crest_elev.data(),
                        inlet_width.data(),
                        inlet_coefficient.data(),
                        inlet_max_capture.data(),
                        outfall_cell.data(),
                        outfall_node.data(),
                        outfall_invert_elev.data(),
                        outfall_diameter.data(),
                        outfall_coefficient.data(),
                        outfall_max_flow.data(),
                        outfall_zero_storage.data(),
                        pipe_end_cell.data(),
                        pipe_end_node.data(),
                        pipe_end_invert_elev.data(),
                        pipe_end_diameter.data(),
                        pipe_end_area.data(),
                        pipe_end_inlet_loss_k.data(),
                        pipe_end_outlet_loss_k.data(),
                        hh_iter.data(),
                        node_state.data(),
                        link_state.data(),
                        dt_sub,
                        gravity,
                        solver_mode,
                        head_deadband_m,
                        dynamic_flow_relaxation,
                        node_out.data(),
                        link_out.data(),
                        q_out.data(),
                        &step_max_node_depth,
                        &step_max_link_flow,
                        &step_limiter_events,
                        &step_limiter_volume_m3);

                    std::memcpy(node_state.data(), node_out.data(), sizeof(double) * static_cast<size_t>(n_nodes));
                    std::memcpy(link_state.data(), link_out.data(), sizeof(double) * static_cast<size_t>(n_links));
                    std::memcpy(q_cell_last.data(), q_out.data(), sizeof(double) * static_cast<size_t>(n_cells));

                    double max_h_update = 0.0;
                    for (int32_t i = 0; i < n_cells; ++i) {
                        const double h_prev = hh_iter[static_cast<size_t>(i)];
                        const double h_old = hh_sub[static_cast<size_t>(i)];
                        const double dh = q_cell_last[static_cast<size_t>(i)] * dt_sub / std::max(cell_area.data()[i], 1.0e-12);
                        const double h_tgt = std::max(h_old - dh, 0.0);
                        hh_target[static_cast<size_t>(i)] = h_tgt;
                        hh_iter[static_cast<size_t>(i)] = (1.0 - relax) * h_prev + relax * h_tgt;
                        max_h_update = std::max(max_h_update, std::abs(hh_iter[static_cast<size_t>(i)] - h_prev));
                    }

                    if (max_h_update <= tiny_h) {
                        converged = true;
                        break;
                    }
                }

                for (int32_t i = 0; i < n_cells; ++i) {
                    q_cell_acc[static_cast<size_t>(i)] += q_cell_last[static_cast<size_t>(i)];
                    const double dh = q_cell_last[static_cast<size_t>(i)] * dt_sub / std::max(cell_area.data()[i], 1.0e-12);
                    hh_sub[static_cast<size_t>(i)] = std::max(hh_sub[static_cast<size_t>(i)] - dh, 0.0);
                }
                max_node_depth = std::max(max_node_depth, step_max_node_depth);
                max_link_flow = std::max(max_link_flow, step_max_link_flow);
                limiter_events += step_limiter_events;
                limiter_volume_m3 += step_limiter_volume_m3;

                if (converged) {
                    break;
                }
            }

            for (int32_t i = 0; i < n_cells; ++i) {
                q_cell_out.mutable_data()[i] = q_cell_acc[static_cast<size_t>(i)] / static_cast<double>(std::max(1, substeps_used));
            }

            std::memcpy(node_state_out.mutable_data(), node_state.data(), sizeof(double) * static_cast<size_t>(n_nodes));
            std::memcpy(link_state_out.mutable_data(), link_state.data(), sizeof(double) * static_cast<size_t>(n_links));

            py::dict diag;
            diag["max_node_depth"] = max_node_depth;
            diag["max_link_flow"] = max_link_flow;
            diag["limiter_events"] = limiter_events;
            diag["limiter_volume_m3"] = limiter_volume_m3;
            diag["substeps_used"] = static_cast<double>(substeps_used);
            diag["implicit_iters_used"] = static_cast<double>(implicit_iters_used);
            diag["inactive_fastpath"] = 0.0;
            return py::make_tuple(node_state_out, link_state_out, q_cell_out, diag);
        },
        py::arg("cell_bed"),
        py::arg("cell_area"),
        py::arg("node_invert_elev"),
        py::arg("node_max_depth"),
        py::arg("node_surface_area"),
        py::arg("link_from"),
        py::arg("link_to"),
        py::arg("link_length"),
        py::arg("link_roughness_n"),
        py::arg("link_diameter"),
        py::arg("link_max_flow"),
        py::arg("inlet_cell"),
        py::arg("inlet_node"),
        py::arg("inlet_crest_elev"),
        py::arg("inlet_width"),
        py::arg("inlet_coefficient"),
        py::arg("inlet_max_capture"),
        py::arg("outfall_cell"),
        py::arg("outfall_node"),
        py::arg("outfall_invert_elev"),
        py::arg("outfall_diameter"),
        py::arg("outfall_coefficient"),
        py::arg("outfall_max_flow"),
        py::arg("outfall_zero_storage"),
        py::arg("pipe_end_cell"),
        py::arg("pipe_end_node"),
        py::arg("pipe_end_invert_elev"),
        py::arg("pipe_end_diameter"),
        py::arg("pipe_end_area"),
        py::arg("pipe_end_inlet_loss_k"),
        py::arg("pipe_end_outlet_loss_k"),
        py::arg("cell_depth"),
        py::arg("node_depth"),
        py::arg("link_flow"),
        py::arg("dt_s"),
        py::arg("gravity"),
        py::arg("solver_mode"),
        py::arg("head_deadband") = 1.0e-3,
        py::arg("dynamic_flow_relaxation") = 1.0,
        py::arg("n_substeps") = 1,
        py::arg("implicit_iters") = 1,
        py::arg("coupling_relaxation") = 0.5,
        "Headless CUDA helper: advance drainage network with native substep/implicit loops in one call.");

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
#else
    m.def("swe2d_gpu_compute_coupling_sources",
        [](py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>) -> py::array_t<double>
        {
            throw std::runtime_error("CUDA path not compiled; swe2d_gpu_compute_coupling_sources is unavailable.");
        },
        py::arg("cell_area"),
        py::arg("inlet_cell"),
        py::arg("inlet_flow"),
        py::arg("structure_up_cell"),
        py::arg("structure_down_cell"),
        py::arg("structure_flow"));

    m.def("swe2d_gpu_compute_bridge_coupling_sources",
        [](py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           double,
           double) -> py::array_t<double>
        {
            throw std::runtime_error("CUDA path not compiled; swe2d_gpu_compute_bridge_coupling_sources is unavailable.");
        },
        py::arg("cell_area"),
        py::arg("bridge_up_cell"),
        py::arg("bridge_down_cell"),
        py::arg("bridge_flow"),
        py::arg("bridge_loss_k_upstream"),
        py::arg("bridge_loss_k_downstream"),
        py::arg("bridge_opening_width") = 1.0,
        py::arg("dt_s") = 1.0);

    m.def("swe2d_gpu_drainage_step",
        [](py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           double,
           double,
           int32_t,
           double,
           double) -> py::tuple
        {
            throw std::runtime_error("CUDA path not compiled; swe2d_gpu_drainage_step is unavailable.");
        });

    m.def("swe2d_gpu_drainage_step_iterative",
        [](py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           double,
           double,
           int32_t,
           double,
           double,
           int32_t,
           int32_t,
           double) -> py::tuple
        {
            throw std::runtime_error("CUDA path not compiled; swe2d_gpu_drainage_step_iterative is unavailable.");
        });

    m.def("swe2d_gpu_enable_kernel_graphs",
        [](py::object, bool) {
            throw std::runtime_error("CUDA path not compiled; swe2d_gpu_enable_kernel_graphs is unavailable.");
        });

    m.def("swe2d_gpu_destroy_kernel_graphs",
        [](py::object) {
            throw std::runtime_error("CUDA path not compiled; swe2d_gpu_destroy_kernel_graphs is unavailable.");
        });
#endif

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
           double mm_to_model_depth) {
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
                                             mm_to_model_depth);
        },
        py::arg("solver"), py::arg("cell_gage_idx"), py::arg("gage_offsets"), py::arg("hg_time_s"), py::arg("hg_cum_mm"), py::arg("cn"), py::arg("ia_ratio") = 0.2, py::arg("mm_to_model_depth") = 1.0e-3,
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

    // ── Predictor-corrector GPU helpers ──────────────────────────────────
    m.def("swe2d_solver_save_coupling_pred",
        [](const std::shared_ptr<PySolver>& ps) {
            if (!ps || !ps->solver) throw std::invalid_argument("null solver handle");
            if (!ps->solver->dev) throw std::runtime_error("GPU not initialized");
            swe2d_gpu_save_coupling_pred(ps->solver->dev);
        },
        "Save current coupling source to predictor buffer (GPU D2D copy).");

    m.def("swe2d_solver_average_coupling_sources",
        [](const std::shared_ptr<PySolver>& ps) {
            if (!ps || !ps->solver) throw std::invalid_argument("null solver handle");
            if (!ps->solver->dev) throw std::runtime_error("GPU not initialized");
            swe2d_gpu_average_coupling_sources(ps->solver->dev);
        },
        "Average predictor and corrector coupling sources on GPU: ext = 0.5*(pred + ext).");

    m.def("swe2d_solver_restore_state_from_backup",
        [](const std::shared_ptr<PySolver>& ps) {
            if (!ps || !ps->solver) throw std::invalid_argument("null solver handle");
            if (!ps->solver->dev) throw std::runtime_error("GPU not initialized");
            swe2d_gpu_restore_state_from_backup(ps->solver->dev);
        },
        "Restore state from backup (d_h0/d_hu0/d_hv0 → d_h/d_hu/d_hv) on GPU.");

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
                  bool extreme_rain_mode,
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
                  int tiny_persistent_chunk_substeps,
                  int tiny_active_compaction_stride_steps,
                  bool tiny_enable_active_compaction,
              bool use_gpu, int n_threads,
              int temporal_order,
              int spatial_scheme,
               int turbulence_model,
               int bed_friction_model,
               int equation_set,
              bool enable_rain_module,
              bool enable_pipe_network_module,
              bool enable_hydraulic_structures,
              int degen_mode,
              double front_flux_damping,
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
            cfg.extreme_rain_mode = extreme_rain_mode;
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
            cfg.tiny_persistent_chunk_substeps = tiny_persistent_chunk_substeps;
            cfg.tiny_active_compaction_stride_steps = tiny_active_compaction_stride_steps;
            cfg.tiny_enable_active_compaction = tiny_enable_active_compaction;
            cfg.temporal_order = temporal_order;
            cfg.spatial_scheme = spatial_scheme;
            cfg.turbulence_model = turbulence_model;
            cfg.bed_friction_model = bed_friction_model;
            cfg.equation_set = equation_set;
            cfg.enable_rain_module = enable_rain_module;
            cfg.enable_pipe_network_module = enable_pipe_network_module;
            cfg.enable_hydraulic_structures = enable_hydraulic_structures;
            cfg.use_gpu   = use_gpu;
            cfg.n_threads = n_threads;
            cfg.degen_mode = degen_mode;
            cfg.front_flux_damping = front_flux_damping;
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
        py::arg("extreme_rain_mode") = false,
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
        py::arg("tiny_persistent_chunk_substeps") = 8,
        py::arg("tiny_active_compaction_stride_steps") = 8,
        py::arg("tiny_enable_active_compaction") = true,
        py::arg("use_gpu")  = true,
        py::arg("n_threads") = 0,
        py::arg("temporal_order") = 2,
        py::arg("spatial_scheme") = 0,
        py::arg("turbulence_model") = 0,
        py::arg("bed_friction_model") = 0,
        py::arg("equation_set") = 0,
        py::arg("enable_rain_module") = false,
        py::arg("enable_pipe_network_module") = false,
        py::arg("enable_hydraulic_structures") = false,
        py::arg("degen_mode") = 0,
        py::arg("front_flux_damping") = 0.5,
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
            d["projection_retry_count"] = diag.projection_retry_count;
            d["projection_attempt_count"] = diag.projection_attempt_count;
            d["projection_retry_exhausted"] = diag.projection_retry_exhausted;
            d["projection_retry_enabled"] = diag.projection_retry_enabled;
            d["projection_retry_fail_fast"] = diag.projection_retry_fail_fast;
            d["projection_retry_dt_initial"] = diag.projection_retry_dt_initial;
            d["projection_retry_dt_floor"] = diag.projection_retry_dt_floor;
            d["projection_retry_dt_reduction"] = diag.projection_retry_dt_reduction;
            d["projection_retry_residual_target"] = diag.projection_retry_residual_target;
            d["projection_retry_residual_ratio"] = diag.projection_retry_residual_ratio;
            d["projection_retry_residual_ratio_max"] = diag.projection_retry_residual_ratio_max;
            d["projection_divergence_ratio"] = diag.projection_divergence_ratio;
            d["projection_divergence_ratio_max"] = diag.projection_divergence_ratio_max;
            d["projection_divergence_gate_enabled"] = diag.projection_divergence_gate_enabled;
            d["projection_divergence_ratio_target"] = diag.projection_divergence_ratio_target;
            d["projection_correction_scale_used"] = diag.projection_correction_scale_used;
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
                    d_dict["projection_retry_count"] = d.projection_retry_count;
                    d_dict["projection_attempt_count"] = d.projection_attempt_count;
                    d_dict["projection_retry_exhausted"] = d.projection_retry_exhausted;
                    d_dict["projection_retry_enabled"] = d.projection_retry_enabled;
                    d_dict["projection_retry_fail_fast"] = d.projection_retry_fail_fast;
                    d_dict["projection_retry_dt_initial"] = d.projection_retry_dt_initial;
                    d_dict["projection_retry_dt_floor"] = d.projection_retry_dt_floor;
                    d_dict["projection_retry_dt_reduction"] = d.projection_retry_dt_reduction;
                    d_dict["projection_retry_residual_target"] = d.projection_retry_residual_target;
                    d_dict["projection_retry_residual_ratio"] = d.projection_retry_residual_ratio;
                    d_dict["projection_retry_residual_ratio_max"] = d.projection_retry_residual_ratio_max;
                    d_dict["projection_divergence_ratio"] = d.projection_divergence_ratio;
                    d_dict["projection_divergence_ratio_max"] = d.projection_divergence_ratio_max;
                    d_dict["projection_divergence_gate_enabled"] = d.projection_divergence_gate_enabled;
                    d_dict["projection_divergence_ratio_target"] = d.projection_divergence_ratio_target;
                    d_dict["projection_correction_scale_used"] = d.projection_correction_scale_used;
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
