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
#include <tuple>
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

// Forward declarations
__global__ __launch_bounds__(256, 4) void swe2d_ext_flux_rk2_fixup_kernel(
    int32_t, double*, double*, double*, const double*, const double*);

// ════════════════════════════════════════════════════════════════════════════
// SPEC §2.4 — Cross-section geometry device functions
// SPEC §2.5 — Preissmann slot surcharge helpers
//
// Phase 1 of the pipe1d solver rewrite. These helpers are pure analytical
// forms of SWMM's xsect_getAofY / xsect_getWofY / xsect_getRofY (for the three
// shapes the current model supports: circular, rectangular, elliptical) plus
// the Sjoberg slot-width formula from SWMM's dwflow.c:583-587.
//
// Calling convention: depths `y` are measured from the link invert (m), NOT
// absolute water-surface elevation. Spec §2.3 states H = z + h(A, shape), so
// callers must pass y = H - invert. Geometry units are SI (m, m²).
// ════════════════════════════════════════════════════════════════════════════

// Shape codes and surcharge constants — shared via swe2d_xsect_constants.h.
#include "swe2d_xsect_constants.h"
using namespace swe2d;

// ── Cross-section parameter layout ─────────────────────────────────────────
// params is a small double[3] array:
//   CIRCULAR    : params[0] = D          (diameter, m)
//   RECTANGULAR : params[0] = b          (width, m),  params[1] = H (height)
//   ELLIPTICAL  : params[0] = 2a (full width),  params[1] = 2b (full height)
// ──────────────────────────────────────────────────────────────────────────

// SPEC §DIAG — Transient instrumentation global variables (revert after debug).
__device__ int32_t g_debug_cell_q_count = 0;       // godunov Q-threshold counter (legacy)
__device__ int64_t g_debug_timestep_counter = 0;     // incremented per pipe-step launch (legacy)

// SPEC §2.4 — yFull (full flow depth) of the cross-section.
__device__ __forceinline__ double xsect_yFull(int shape_type, const double params[3])
{
    if (shape_type == XSECT_CIRCULAR)    return params[0];
    if (shape_type == XSECT_RECTANGULAR) return params[1];
    /* ELLIPTICAL */                     return params[1];
}

// SPEC §2.4 — wMax (max top width) of the cross-section.
__device__ __forceinline__ double xsect_wMax(int shape_type, const double params[3])
{
    (void)shape_type;
    return params[0];
}

// SPEC §2.4 — A_full (full cross-section area).
__device__ __forceinline__ double xsect_aFull(int shape_type, const double params[3])
{
    if (shape_type == XSECT_CIRCULAR) {
        const double D = params[0];
        return 0.25 * M_PI * D * D;
    }
    if (shape_type == XSECT_RECTANGULAR) {
        return params[0] * params[1];
    }
    /* ELLIPTICAL */
    const double a_semi = 0.5 * params[0];
    const double b_semi = 0.5 * params[1];
    return M_PI * a_semi * b_semi;
}

// ════════════════════════════════════════════════════════════════════════════
// SPEC §2.4 — CIRCULAR PIPE analytical geometry
// ════════════════════════════════════════════════════════════════════════════

// SPEC §2.4 circular — flow area A(y).
__device__ __forceinline__ double xsect_getAofY_circular(double D, double y)
{
    if (y <= 0.0) return 0.0;
    const double D_safe = fmax(D, 1.0e-12);
    const double arg = 1.0 - 2.0 * y / D_safe;
    const double theta = 2.0 * acos(fmax(-1.0, fmin(1.0, arg)));
    return (D_safe * D_safe / 8.0) * (theta - sin(theta));
}

// SPEC §2.4 circular — top width T(y).
__device__ __forceinline__ double xsect_getWofY_circular(double D, double y)
{
    if (y <= 0.0) return 0.0;
    const double D_safe = fmax(D, 1.0e-12);
    if (y >= D_safe) return 0.0;
    const double arg = 1.0 - 2.0 * y / D_safe;
    const double theta = 2.0 * acos(fmax(-1.0, fmin(1.0, arg)));
    return D_safe * sin(theta * 0.5);
}

// SPEC §2.4 circular — hydraulic radius R_h(y) = A / P.
__device__ __forceinline__ double xsect_getRofY_circular(double D, double y)
{
    if (y <= 0.0) return 0.0;
    const double D_safe = fmax(D, 1.0e-12);
    const double arg = 1.0 - 2.0 * y / D_safe;
    const double theta = 2.0 * acos(fmax(-1.0, fmin(1.0, arg)));
    const double A = (D_safe * D_safe / 8.0) * (theta - sin(theta));
    const double P = D_safe * theta * 0.5;
    return A / fmax(1.0e-12, P);
}

// ════════════════════════════════════════════════════════════════════════════
// SPEC §2.4 — RECTANGULAR (closed) analytical geometry
// ════════════════════════════════════════════════════════════════════════════

// SPEC §2.4 rectangular — flow area A(y).
__device__ __forceinline__ double xsect_getAofY_rectangular(double b, double H, double y)
{
    if (y <= 0.0) return 0.0;
    return b * fmin(y, H);
}

// SPEC §2.4 rectangular — top width T(y).
__device__ __forceinline__ double xsect_getWofY_rectangular(double b, double H, double y)
{
    if (y <= 0.0 || y >= H) return 0.0;
    return b;
}

// SPEC §2.4 rectangular — hydraulic radius R_h(y) = A / P.
__device__ __forceinline__ double xsect_getRofY_rectangular(double b, double H, double y)
{
    if (y <= 0.0) return 0.0;
    const double yc = fmin(y, H);
    const double P = b + 2.0 * yc;
    return (b * yc) / fmax(1.0e-12, P);
}

// ════════════════════════════════════════════════════════════════════════════
// SPEC §2.4 — ELLIPTICAL analytical geometry
// ════════════════════════════════════════════════════════════════════════════

// SPEC §2.4 elliptical helper — central angle φ(y).
__device__ __forceinline__ double xsect_ellipse_phi(double b_semi, double y)
{
    if (y <= 0.0) return 0.0;
    const double b_safe = fmax(b_semi, 1.0e-12);
    if (y >= 2.0 * b_safe) return M_PI;
    const double arg = 1.0 - y / b_safe;
    return acos(fmax(-1.0, fmin(1.0, arg)));
}

// SPEC §2.4 elliptical — flow area A(y).
__device__ __forceinline__ double xsect_getAofY_elliptical(double a_semi, double b_semi, double y)
{
    if (y <= 0.0) return 0.0;
    const double phi = xsect_ellipse_phi(b_semi, y);
    return a_semi * b_semi * (phi - 0.5 * sin(2.0 * phi));
}

// SPEC §2.4 elliptical — top width T(y).
__device__ __forceinline__ double xsect_getWofY_elliptical(double a_semi, double b_semi, double y)
{
    if (y <= 0.0) return 0.0;
    if (y >= 2.0 * b_semi) return 0.0;
    const double phi = xsect_ellipse_phi(b_semi, y);
    return 2.0 * a_semi * sin(phi);
}

// SPEC §2.4 elliptical — hydraulic radius R_h(y) = A / P (approx).
__device__ __forceinline__ double xsect_getRofY_elliptical(double a_semi, double b_semi, double y)
{
    if (y <= 0.0) return 0.0;
    const double a_safe = fmax(a_semi, 1.0e-12);
    const double b_safe = fmax(b_semi, 1.0e-12);
    const double phi = xsect_ellipse_phi(b_safe, y);
    const double A = a_safe * b_safe * (phi - 0.5 * sin(2.0 * phi));
    const double h_param = ((a_safe - b_safe) * (a_safe - b_safe)) /
                           ((a_safe + b_safe) * (a_safe + b_safe));
    const double P_full = M_PI * (a_safe + b_safe) *
        (1.0 + 3.0 * h_param / (10.0 + sqrt(fmax(0.0, 4.0 - 3.0 * h_param))));
    const double P_partial = P_full * phi / M_PI;
    return A / fmax(1.0e-12, P_partial);
}

// ════════════════════════════════════════════════════════════════════════════
// SPEC §2.4 — Top-level dispatch (matches SWMM's xsect_getAofY/WofY/RofY).
// ════════════════════════════════════════════════════════════════════════════

// SPEC §2.4 — flow area A(y) for any of the three supported shapes.
__device__ __forceinline__ double xsect_getAofY(int shape_type, const double params[3], double y)
{
    if (shape_type == XSECT_CIRCULAR)
        return xsect_getAofY_circular(params[0], y);
    if (shape_type == XSECT_RECTANGULAR)
        return xsect_getAofY_rectangular(params[0], params[1], y);
    return xsect_getAofY_elliptical(0.5 * params[0], 0.5 * params[1], y);
}

// SPEC §2.4 — top width T(y) for any of the three supported shapes.
__device__ __forceinline__ double xsect_getWofY(int shape_type, const double params[3], double y)
{
    if (shape_type == XSECT_CIRCULAR)
        return xsect_getWofY_circular(params[0], y);
    if (shape_type == XSECT_RECTANGULAR)
        return xsect_getWofY_rectangular(params[0], params[1], y);
    return xsect_getWofY_elliptical(0.5 * params[0], 0.5 * params[1], y);
}

// SPEC §2.4 — hydraulic radius R_h(y) = A(y) / P(y) for any of the three.
__device__ __forceinline__ double xsect_getRofY(int shape_type, const double params[3], double y)
{
    if (shape_type == XSECT_CIRCULAR)
        return xsect_getRofY_circular(params[0], y);
    if (shape_type == XSECT_RECTANGULAR)
        return xsect_getRofY_rectangular(params[0], params[1], y);
    return xsect_getRofY_elliptical(0.5 * params[0], 0.5 * params[1], y);
}

// ════════════════════════════════════════════════════════════════════════════
// SPEC §2.5 — Preissmann slot width (Sjoberg formula).
//   slot_width(y, yFull, wMax):
//     if y > 1.78 * yFull:   return 0.01 * wMax            (cap)
//     else:                  return wMax * 0.5423 * exp(-(y/yFull)^2.4)
// Matches SWMM dwflow.c:583-587 (getSlotWidth, SLOT surcharge branch).
// ════════════════════════════════════════════════════════════════════════════

__device__ __forceinline__ double slot_width(double y, double yFull, double wMax)
{
    if (y <= 0.0 || yFull <= 0.0) return 0.0;
    const double yNorm = y / yFull;
    if (yNorm > 1.78) return 0.01 * wMax;
    return wMax * 0.5423 * exp(-pow(yNorm, 2.4));
}

// ════════════════════════════════════════════════════════════════════════════
// SPEC §2.5 — Pressurised cross-section area.
//   A_pressurised(y, wMax, surcharge_method):
//     if surcharge_method == SLOT and y > yFull:
//         return A_full + (y - yFull) * slot_width(y, yFull, wMax)
//     else:
//         return min(xsect_getAofY(shape_type, params, y), A_full)
//
// The explicit min(..., A_full) clamp on the non-SLOT branch matches the spec
// wording verbatim and is defensive against floating-point overshoot in the
// geometric functions (circular / elliptical saturate at A_full via acos
// clamping; rectangular saturates via fmin(y, H)).
//
// Matches SWMM dwflow.c:609-619 (getArea, SLOT surcharge branch).
// ════════════════════════════════════════════════════════════════════════════

__device__ __forceinline__ double xsect_getAofY_pressurised(
    int shape_type, const double params[3],
    double y, double wMax, int surcharge_method)
{
    if (y <= 0.0) return 0.0;
    const double yFull = xsect_yFull(shape_type, params);
    const double A_full = xsect_aFull(shape_type, params);
    if (y > yFull && surcharge_method == SURCHARGE_SLOT && wMax > 0.0 && yFull > 0.0) {
        // SPEC §2.5 — Pressurised SLOT branch:
        //   A_pressurised = A_full + (y - yFull) * slot_width(y, yFull, wMax)
        return A_full + (y - yFull) * slot_width(y, yFull, wMax);
    }
    // SPEC §2.5 — Open-channel (or non-SLOT surcharge) branch:
    //   A_pressurised = min(xsect_getAofY(y), A_full)
    return fmin(xsect_getAofY(shape_type, params, y), A_full);
}

// ════════════════════════════════════════════════════════════════════════════
// SPEC §2.4 — Inverse area-to-depth for the three supported shapes.
//
// For rectangular shapes the inverse is explicit: y = A / b (clamped at yFull).
// For circular and elliptical a bisection on [0, yFull] is used.
// ════════════════════════════════════════════════════════════════════════════

// SPEC §2.4 — Inverse: given A, find the open-channel depth y.
__device__ __forceinline__ double xsect_getAofY_inv(
    int shape_type, const double params[3], double A_target, double yFull)
{
    if (A_target <= 0.0) return 0.0;
    const double A_full = xsect_aFull(shape_type, params);
    if (A_target >= A_full) return yFull;

    // Rectangle has explicit closed-form inverse: y = A / b.
    if (shape_type == XSECT_RECTANGULAR) {
        const double b = params[0];
        if (b > 1.0e-12) return fmin(yFull, A_target / b);
        return 0.0;
    }

    // Circular or elliptical: bisection on the monotonic A(y) in [0, yFull].
    double y_lo = 0.0;
    double y_hi = yFull;
    for (int iter = 0; iter < 30; ++iter) {
        const double y_mid = 0.5 * (y_lo + y_hi);
        const double A_mid = xsect_getAofY(shape_type, params, y_mid);
        if (fabs(A_mid - A_target) <= 1.0e-12 * A_full) return y_mid;
        if (A_mid < A_target) y_lo = y_mid;
        else                  y_hi = y_mid;
    }
    return 0.5 * (y_lo + y_hi);
}

// SPEC §2.5 — Inverse for pressurised area (SLOT surcharge).
// Returns y >= yFull such that A_pressurised(y) ≈ A_target.
// Uses Newton iteration on dA/dy = slot_width(y).
__device__ __forceinline__ double xsect_getAofY_pressurised_inv(
    int shape_type, const double params[3],
    double A_target, double wMax, int surcharge_method)
{
    if (A_target <= 0.0) return 0.0;
    const double yFull = xsect_yFull(shape_type, params);
    const double A_full = xsect_aFull(shape_type, params);
    if (A_target <= A_full || surcharge_method != SURCHARGE_SLOT) {
        return xsect_getAofY_inv(shape_type, params, A_target, yFull);
    }

    // Pressurised: initial guess using 1% slot width
    const double sw_0 = fmax(0.01 * fmax(wMax, 1.0e-12), 1.0e-12);
    double y = yFull + (A_target - A_full) / sw_0;
    if (y <= yFull) y = yFull + 1.0e-6;

    for (int iter = 0; iter < 12; ++iter) {
        const double A_cur = xsect_getAofY_pressurised(shape_type, params, y, wMax, SURCHARGE_SLOT);
        const double sw = fmax(slot_width(y, yFull, wMax), 1.0e-12);
        y -= (A_cur - A_target) / sw;
        if (fabs(A_cur - A_target) < 1.0e-12 * A_full) break;
        if (y < yFull) y = yFull;
    }
    return fmax(yFull, y);
}

// ════════════════════════════════════════════════════════════════════════════
// End of SPEC §2.4 / §2.5 helpers — host code resumes below.
// ════════════════════════════════════════════════════════════════════════════

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
    table_out.resize(3 * PIPE1D_TABLE_N, 0.0);

    constexpr double EPS = 1e-12;

    for (int i = 0; i < PIPE1D_TABLE_N; ++i) {
        double A_ratio = (i + 0.5) / PIPE1D_TABLE_N;
        double A_target = A_ratio * A_full;
        double P_cur, T_cur, I1_cur = 0.0;

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
            // I₁ = A·(y − y_c) where y_c is centroid from invert
            // θ = central angle, sin_half = sin(θ/2)
            {
                const double theta_circ = 2.0 * phi;
                const double sin_half_circ = sin(phi);  // sin(θ/2)
                const double den_circ = theta_circ - sin(theta_circ);
                const double yc_from_center_circ = (fabs(den_circ) > 1e-14)
                    ? (4.0/3.0) * R * sin_half_circ * sin_half_circ * sin_half_circ / den_circ
                    : 0.0;
                const double yc_circ = R - yc_from_center_circ;
                const double A_cur_circ = R * R * phi - (R - y) * T_cur * 0.5;
                I1_cur = A_cur_circ * (y - yc_circ);
            }
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
            I1_cur = (y > 0.0) ? 0.5 * W * y * y : 0.0;
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
            // Numerical integration for I1 of elliptical segment
            I1_cur = 0.0;
            const int N_I1 = 200;
            const double dy_I1 = y / (double)N_I1;
            for (int j = 0; j < N_I1; ++j) {
                const double yj = (j + 0.5) * dy_I1;
                const double yrj = yj / b;
                const double Tj = 2.0 * a * sqrt(fmax(0.0, 1.0 - (yrj - 1.0) * (yrj - 1.0)));
                I1_cur += (y - yj) * Tj;
            }
            I1_cur *= dy_I1;
        }

        table_out[i]                     = P_cur / P_full;
        table_out[PIPE1D_TABLE_N + i]    = T_cur;
        table_out[2 * PIPE1D_TABLE_N + i] = I1_cur;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SPEC §2.9 — Per-pipe-end A_open(y_sub) lookup table.
//
// The weir/orifice exchange kernel needs the area of the pipe-end opening
// submerged below the tailwater elevation.  y_sub = tailwater_WSE - node_invert,
// clamped to [0, yFull].  Computing this on-the-fly for circular/elliptical
// shapes calls acos every step — instead we precompute a 1D table per pipe-end
// (deduplicated by XsectKey across pipe-ends that share geometry, exactly
// like the cell-level table in pipe1d_compute_table).
//
// table_out layout: [A_open[0..N-1]]  with  A_open[i] at  y_sub = (i+0.5)/N * yFull.
// ═══════════════════════════════════════════════════════════════════════════

// Hash for std::tuple<int, double, double> used to dedup A_open tables.
struct XsectKeyTripleHash {
    size_t operator()(const std::tuple<int, double, double>& k) const {
        return std::hash<int>()(std::get<0>(k))
             ^ std::hash<double>()(std::get<1>(k))
             ^ std::hash<double>()(std::get<2>(k));
    }
};

static void pipe1d_compute_pipe_end_A_open_table(
    int shape_type, double width, double height,
    double& yFull_out,
    std::vector<double>& table_out)
{
    table_out.clear();
    table_out.resize(PIPE1D_TABLE_N, 0.0);

    double yFull = 0.0;
    if (shape_type == 0) {
        yFull = width;  // circular: yFull = diameter
    } else if (shape_type == 1) {
        yFull = height;  // rectangular: yFull = rise
    } else {
        yFull = height;  // elliptical: yFull = rise
    }
    yFull_out = yFull;
    if (yFull <= 0.0) return;

    // Build via inlined host versions of the geometric formulas (the __device__
// helpers can't be called from host code).  For rectangular shapes,
    // A_open(y_sub) = width * y_sub is closed-form.  For circular and
    // elliptical, we evaluate the geometric functions at each midpoint once
    // at upload time.
    auto xsect_getAofY_local = [shape_type, width, height](double y) -> double {
        if (shape_type == 0) {
            // Circular: A = R²·acos((R-y)/R) - (R-y)·sqrt(2Ry - y²)
            const double R = 0.5 * width;
            const double y_clamped = fmax(0.0, fmin(y, 2.0 * R));
            const double arg = (R - y_clamped) / R;
            const double arg_c = fmax(-1.0, fmin(1.0, arg));
            const double inside = fmax(0.0, 2.0 * R * y_clamped - y_clamped * y_clamped);
            return R * R * acos(arg_c) - (R - y_clamped) * sqrt(inside);
        } else if (shape_type == 1) {
            // Rectangular: A = W * min(y, H)
            return width * fmax(0.0, fmin(y, height));
        }
        // Elliptical: A(y) = a·b·(acos(t) − t·sqrt(1−t²)), t = 1 − y/b.
        // Correct unified formula for 0 ≤ y ≤ 2b (audit F12).
        const double a = 0.5 * width;
        const double b = 0.5 * height;
        const double y_clamped = fmax(0.0, fmin(y, 2.0 * b));
        const double t = fmax(-1.0, fmin(1.0, 1.0 - y_clamped / b));
        return a * b * (acos(t) - t * sqrt(fmax(0.0, 1.0 - t * t)));
    };

    for (int i = 0; i < PIPE1D_TABLE_N; ++i) {
        const double y_sub = (i + 0.5) / PIPE1D_TABLE_N * yFull;
        table_out[i] = xsect_getAofY_local(y_sub);
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
    const double*         node_inlet_loss_k,     // per-node inlet loss (optional, may be null)
    const double*         node_outlet_loss_k,    // per-node outlet loss (optional, may be null)
    const double*         node_surface_area,
    const double*         node_max_depth,
    const double*         link_invert_in,
    const double*         link_invert_out,
    const double*        link_max_cell_length,
    double               g_slot_cfl,        // grav const for slot CFL pre-compute
    const int32_t*        link_shape_type,
    const double*         link_width,
    const double*         link_height,
    Pipe1DDeviceState*    dev,
    int32_t               n_manholes,
    const int32_t*        manhole_node,
    const double*         manhole_cell_length,
    const double*         manhole_cell_width,
    const double*         manhole_cell_height,
    int32_t               n_inlets,
    const int32_t*        inlet_node,
    const double*         inlet_cell_length,
    const double*         inlet_cell_width,
    const double*         inlet_cell_height,
    const int32_t*        node_is_outfall,
    const int32_t*        node_is_inlet,
    const int32_t*        node_is_pipe_end,
    int32_t               n_cells_2d,
    // ── Phase F3 — HEC-22 inlet capture geometry (SURFACE_2D_INLET faces) ──
    int32_t               n_inlet_capture_faces,
    const int32_t*        inlet_face_node,
    const int32_t*        inlet_face_2d_cell,
    const int32_t*        inlet_face_type,
    const double*         inlet_face_grate_len,
    const double*         inlet_face_grate_wid,
    const double*         inlet_face_grate_open,
    const double*         inlet_face_curb_len,
    const double*         inlet_face_curb_ht,
    const double*         inlet_face_curb_throat,
    const double*         inlet_face_slot_len,
    const double*         inlet_face_slot_wid,
    const double*         inlet_face_crest,
    const double*         inlet_face_cd,
    const double*         inlet_face_qmax,
    // ── Phase F3 — Culvert faces (class 6) ──
    int32_t               n_culvert_faces,
    const int32_t*        culvert_face_donor_2d_cell,
    const int32_t*        culvert_face_receiver_2d_cell,
    const int32_t*        culvert_struct_idx,
    int32_t               n_structures,
    const double*         structure_flows)
{
    // Create a private stream for all pipe1d operations.  Using cudaMallocAsync
    // on this stream creates a dedicated stream-ordered memory pool, isolating
    // pipe1d allocations from the global CUDA pool shared with the 2D solver.
    // This prevents cross-thread pool aliasing where cudaFree on one thread
    // returns a pipe1d address to the pool, and cudaMalloc on another thread
    // reuses it, corrupting the face/state arrays.
    if (!dev->d_stream) {
        CUDA_CHECK(cudaStreamCreate(&dev->d_stream));
    }
    cudaStream_t stream = dev->d_stream;

    auto alloc_d = [stream](void** ptr, size_t bytes) {
        // Use stream-ordered allocation so the resulting device pointer is
        // synchronised with subsequent operations on dev->d_stream.
        CUDA_CHECK(cudaMallocAsync(ptr, bytes, stream));
    };
    auto copy_h2d_i = [stream](int32_t* dst, const int32_t* src, size_t n) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, n * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    };
    auto copy_h2d_d = [stream](double* dst, const double* src, size_t n) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, n * sizeof(double), cudaMemcpyHostToDevice, stream));
    };

    // Count sub-cells per link and find max node index
    std::vector<int32_t> sub_cells_per_link(n_links);
    int32_t max_node_idx = -1;
    int32_t total_pipe_cells = 0;
    for (int32_t i = 0; i < n_links; ++i) {
        const double L = link_length[i];
        int32_t n_sub = 1;
        if (link_max_cell_length && link_max_cell_length[i] > 0.0 && L > 0.0) {
            n_sub = static_cast<int32_t>(std::ceil(L / link_max_cell_length[i]));
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
    dev->n_manhole_cells = n_manholes;
    dev->n_inlet_cells = n_inlets;
    dev->n_cells_all = total_pipe_cells + n_manholes + n_inlets;
    const int32_t n_cells_all = dev->n_cells_all;

    // Per-node entrance/exit loss coefficients (optional, may be null)
    std::vector<double> h_node_inlet_loss_k(n_nodes, 0.0);
    std::vector<double> h_node_outlet_loss_k(n_nodes, 0.0);
    if (node_inlet_loss_k) {
        std::memcpy(h_node_inlet_loss_k.data(), node_inlet_loss_k, n_nodes * sizeof(double));
    }
    if (node_outlet_loss_k) {
        std::memcpy(h_node_outlet_loss_k.data(), node_outlet_loss_k, n_nodes * sizeof(double));
    }

    // Allocate persistent scratch buffers for swe2d_pipe1d_step
    // (avoids per-step cudaMalloc/cudaFree which forces synchronous driver round-trips).
    // Zero them up front — swe2d_pipe1d_godunov_step_internal reads from
    // d_A_new_scratch / d_Q_new_scratch / d_flux_*_scratch on its very first
    // call.  cudaMalloc returns uninitialised memory, and a kernel reading
    // that garbage will dereference a garbage-derived index → "illegal
    // memory access" on the first FVM step (e.g. line 3021 of this file,
    // where the Godunov kernel is launched with these scratch args).
    alloc_d(reinterpret_cast<void**>(&dev->d_flux_Q_scratch), static_cast<size_t>(n_cells_all) * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_flux_mom_scratch), static_cast<size_t>(n_cells_all) * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_A_new_scratch), static_cast<size_t>(n_cells_all) * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_Q_new_scratch), static_cast<size_t>(n_cells_all) * sizeof(double));
    CUDA_CHECK(cudaMemset(dev->d_flux_Q_scratch,   0, static_cast<size_t>(n_cells_all) * sizeof(double)));
    CUDA_CHECK(cudaMemset(dev->d_flux_mom_scratch, 0, static_cast<size_t>(n_cells_all) * sizeof(double)));
    CUDA_CHECK(cudaMemset(dev->d_A_new_scratch,    0, static_cast<size_t>(n_cells_all) * sizeof(double)));
    CUDA_CHECK(cudaMemset(dev->d_Q_new_scratch,    0, static_cast<size_t>(n_cells_all) * sizeof(double)));

    // SPEC §2.1 — Virtual-node topology.
    // Sub-cells of a link i are internal to the link. V[0] = link.from (network),
    // V[N] = link.to (network), V[1..N-1] are virtual nodes that occupy
    // indices [n_nodes + offset, n_nodes + offset + N - 2).
    // Compute n_vnodes and per-link virtual-node offsets here so the cell-build
    // loop and array allocations can use them.
    int32_t n_vnodes = 0;
    std::vector<int32_t> vnode_offset_per_link(n_links, 0);
    {
        int32_t cursor = 0;
        for (int32_t i = 0; i < n_links; ++i) {
            vnode_offset_per_link[i] = cursor;
            cursor += std::max(0, sub_cells_per_link[i] - 1);
        }
    }
    // Phase 2.1: n_vnodes retired — virtual nodes no longer used as the
    // interior-face state. Interior faces now use the unified face mesh
    // (face_class=INTERIOR, with face_owner_R pointing at the downstream cell).
    // The cursor-based accounting above is still used inside this function
    // for cell_from_node / cell_to_node population (which the godunov update
    // kernel uses to distinguish boundary faces), but no device array is
    // allocated for virtual nodes any more.

    // Geometry per sub-cell (sized to n_cells_all to include manhole/inlet cells)
    std::vector<double> cell_length(n_cells_all);
    std::vector<double> cell_link_length(n_cells_all);
    std::vector<double> cell_area(n_cells_all);
    std::vector<double> cell_perim(n_cells_all);
    std::vector<double> cell_invert(n_cells_all);
    std::vector<double> cell_n(n_cells_all);
    std::vector<double> cell_link_k(n_cells_all);     // k at boundary cells, 0 interior
    std::vector<double> cell_link_k_in(n_cells_all);  // link-level k_in (all cells of link)
    std::vector<double> cell_link_k_out(n_cells_all); // link-level k_out (all cells of link)
    std::vector<double> cell_link_area(n_cells_all);  // full pipe area at boundary cells, 0 interior
    std::vector<int32_t> cell_from_node(n_cells_all);
    std::vector<int32_t> cell_to_node(n_cells_all);
    std::vector<int32_t> cell_owner_link(n_cells_all);  // which link each sub-cell belongs to
    std::vector<int32_t> cell_sub_idx(n_cells_all);    // sub-cell index within its link
    std::vector<double>  cell_S0(n_cells_all);         // SPEC §2.6 — conduit slope per cell
    std::vector<int32_t> cell_is_end(n_cells_all);     // SPEC §2.6 — 1 if cell is at link end (s == N-1)

    // Cross-section shape + table data
    std::vector<int32_t> cell_shape_type(n_cells_all);
    std::vector<double> cell_width(n_cells_all);
    std::vector<double> cell_height(n_cells_all);
    std::vector<double> cell_tables(static_cast<size_t>(n_cells_all) * 3 * PIPE1D_TABLE_N, 0.0);
    // SPEC §2.5 — Per-cell Preissmann slot width (Sjoberg formula's wMax input).
    // For PIPE cells: wMax = xsect_wMax(shape, params) = params[0] (D, b, or 2a for
    //   circular / rectangular / elliptical respectively). In the existing data
    //   layout cell_width[c] IS params[0] for all 3 supported shapes, so the slot
    //   width for pipe cells equals cell_width[c].
    // For MANHOLE/INLET cells: wMax = volume-equivalent rectangular width (W),
    //   which is exactly what cell_width[c] already holds in those loops.
    // Therefore: h_cell_slot_width[c] = cell_width[c] for all cell classes.
    std::vector<double> h_cell_slot_width(n_cells_all, 0.0);
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
        double A, P;
        if (stype == XSECT_RECTANGULAR && sw > 0.0 && sh > 0.0) {
            // Rectangular / box: A = w * h, P = 2(w + h)
            A = sw * sh;
            P = 2.0 * (sw + sh);
        } else if (stype == XSECT_ELLIPTICAL && sw > 0.0 && sh > 0.0) {
            // Elliptical: area = π * (w/2) * (h/2), perimeter ≈ Ramanujan
            A = M_PI * (sw / 2.0) * (sh / 2.0);
            const double a = sw / 2.0, b = sh / 2.0;
            const double h_ell = ((a - b) * (a - b)) / ((a + b) * (a + b));
            P = M_PI * (a + b) * (1.0 + (3.0 * h_ell) / (10.0 + std::sqrt(4.0 - 3.0 * h_ell)));
        } else {
            // Circular (or fallback): use diameter D, or width as diameter if D==0
            const double D_eff = (D > 0.0) ? D : sw;
            A = M_PI * D_eff * D_eff / 4.0;
            P = M_PI * D_eff;
        }

        for (int32_t s = 0; s < n_sub; ++s) {
            const double frac = (static_cast<double>(s) + 0.5) / static_cast<double>(n_sub);
            cell_length[cell_idx] = sub_len;
            cell_link_length[cell_idx] = L;
            cell_area[cell_idx] = A;
            cell_perim[cell_idx] = P;
            cell_invert[cell_idx] = inv_in + frac * (inv_out - inv_in);
            cell_shape_type[cell_idx] = stype;
            cell_width[cell_idx] = sw;
            cell_height[cell_idx] = sh;
            // SPEC §2.5 — Preissmann slot wMax for this pipe sub-cell. Maps
            //   circular:    D     (params[0])
            //   rectangular: b     (params[0])
            //   elliptical:  2a    (params[0])
            // All three cases collapse to cell_width = sw, so we set the slot
            // width directly from sw (the link's primary width).
            h_cell_slot_width[cell_idx] = sw;
            cell_n[cell_idx] = n_val;
            cell_link_k[cell_idx] = (s == 0) ? k_in : (s == n_sub - 1) ? k_out : 0.0;
            cell_link_k_in[cell_idx] = k_in;
            cell_link_k_out[cell_idx] = k_out;
            cell_link_area[cell_idx] = (s == 0 || s == n_sub - 1) ? A : 0.0;
            // SPEC §2.1 — sub-cell s of link i has cell_from = V[s], cell_to = V[s+1].
            // V[0] and V[N] are real network nodes (< n_nodes); V[1..N-1] are virtual
            // (indices in [n_nodes + offset, n_nodes + offset + N - 2)).
            const int32_t V_off = n_nodes + vnode_offset_per_link[i];
            cell_from_node[cell_idx] = (s == 0) ? link_from_node[i] : (V_off + (s - 1));
            cell_to_node[cell_idx]   = (s == n_sub - 1) ? link_to_node[i] : (V_off + s);
            cell_owner_link[cell_idx] = i;
            cell_sub_idx[cell_idx] = s;
            // SPEC §2.6 — conduit slope S0 per cell. Positive when downstream
            // invert is lower (gravity-driven flow direction). Used by the
            // wave kernels' regime override (SWMM checkNormalFlow) to compute
            // Manning's normal flow Q_n. Sub-cells of the same link share S0.
            const double L_safe = fmax(L, 1.0e-12);
            cell_S0[cell_idx] = (inv_in - inv_out) / L_safe;
            // SPEC §2.6 — end-cell flag. End cells (s == N-1) have a real
            // network node as `to_node`; only then can the downstream-outfall
            // condition be evaluated against node_is_outfall[].
            cell_is_end[cell_idx] = (s == n_sub - 1) ? 1 : 0;
            ++cell_idx;
        }
    }

    // Compute precomputed tables for each unique cross-section (pipe cells)
    for (int c = 0; c < total_pipe_cells; ++c) {
        XsectKey key{cell_shape_type[c], cell_width[c], cell_height[c]};
        auto it = table_cache.find(key);
        if (it == table_cache.end()) {
            std::vector<double> tbl;
            double A_full_dummy, P_full_dummy;
            pipe1d_compute_table(key.shape_type, key.w, key.h, A_full_dummy, P_full_dummy, tbl);
            it = table_cache.emplace(key, std::move(tbl)).first;
        }
        std::memcpy(&cell_tables[c * 3 * PIPE1D_TABLE_N],
                    it->second.data(), 3 * PIPE1D_TABLE_N * sizeof(double));
    }

    // ── Phase 2.1 — Populate manhole cell data (indices total_pipe_cells..total_pipe_cells+n_manholes-1) ──
    std::vector<int32_t> h_cell_class(n_cells_all, 0);         // default PIPE_CELL
    std::vector<double>  h_cell_crown(n_cells_all, 0.0);
    std::vector<double>  h_cell_rim(n_cells_all, 0.0);
    std::vector<double>  h_cell_surface_area(n_cells_all, 0.0);
    std::vector<double>  h_cell_max_depth(n_cells_all, 0.0);

    int32_t manhole_cell_cursor = total_pipe_cells;
    for (int32_t i = 0; i < n_manholes; ++i, ++manhole_cell_cursor) {
        const int32_t c = manhole_cell_cursor;
        const int32_t n = (manhole_node ? manhole_node[i] : -1);
        const double L = manhole_cell_length[i];
        const double W = manhole_cell_width[i];
        const double H = manhole_cell_height[i];

        h_cell_class[c] = 1;  // MANHOLE_CELL

        cell_length[c]       = L;
        cell_link_length[c]  = L;
        cell_width[c]        = W;
        cell_height[c]       = H;
        cell_area[c]         = W * H;  // A_full
        cell_perim[c]        = 2.0 * (W + H);
        cell_S0[c]           = 0.0;
        cell_n[c]            = 0.0;
        cell_owner_link[c]   = -1;
        cell_sub_idx[c]      = 0;
        cell_is_end[c]       = 0;
        cell_link_k[c]       = 0.0;
        cell_link_k_in[c]    = 0.0;
        cell_link_k_out[c]   = 0.0;
        cell_link_area[c]    = 0.0;
        cell_from_node[c]    = -1;
        cell_to_node[c]      = -1;
        cell_shape_type[c]   = XSECT_RECTANGULAR;

        // Crown = node_crown if the manhole corresponds to a network node (audit F9)
        // Phase 2.3 will compute crown from connected link heights;
        // for now copy node_crown if available.
        if (n >= 0 && n < n_nodes) {
            const double h_link = 0.0; // will be populated by Phase 2.3
            h_cell_crown[c] = 0.0;
            // Invert from the manhole's network node
            cell_invert[c] = node_invert_elev[n];
        } else {
            h_cell_crown[c] = 0.0;
            cell_invert[c] = 0.0;
        }

        h_cell_rim[c]          = (n >= 0 && n < n_nodes ? node_invert_elev[n] : 0.0) + H;
        h_cell_surface_area[c] = W * L; // true horizontal area
        h_cell_max_depth[c]    = H;
        // SPEC §2.5 — Preissmann slot wMax for manhole cell. The manhole is
        // treated as a volume-equivalent rectangle; W is already that width
        // and equals the slot's wMax.
        h_cell_slot_width[c]   = W;
    }

    // ── Phase 2.1 — Populate inlet cell data ──
    int32_t inlet_cell_cursor = manhole_cell_cursor;
    for (int32_t i = 0; i < n_inlets; ++i, ++inlet_cell_cursor) {
        const int32_t c = inlet_cell_cursor;
        const int32_t n = (inlet_node ? inlet_node[i] : -1);
        const double L = inlet_cell_length[i];
        const double W = inlet_cell_width[i];
        const double H = inlet_cell_height[i];

        h_cell_class[c] = 2;  // INLET_CELL

        cell_length[c]       = L;
        cell_link_length[c]  = L;
        cell_width[c]        = W;
        cell_height[c]       = H;
        cell_area[c]         = W * H;  // A_full
        cell_perim[c]        = 2.0 * (W + H);
        cell_S0[c]           = 0.0;
        cell_n[c]            = 0.0;
        cell_owner_link[c]   = -1;
        cell_sub_idx[c]      = 0;
        cell_is_end[c]       = 0;
        cell_link_k[c]       = 0.0;
        cell_link_k_in[c]    = 0.0;
        cell_link_k_out[c]   = 0.0;
        cell_link_area[c]    = 0.0;
        cell_from_node[c]    = -1;
        cell_to_node[c]      = -1;
        cell_shape_type[c]   = XSECT_RECTANGULAR;

        h_cell_crown[c] = 0.0;
        if (n >= 0 && n < n_nodes) {
            cell_invert[c] = node_invert_elev[n];
        } else {
            cell_invert[c] = 0.0;
        }
        h_cell_rim[c]          = (n >= 0 && n < n_nodes ? node_invert_elev[n] : 0.0) + H;
        h_cell_surface_area[c] = W * L;
        h_cell_max_depth[c]    = H;
        // SPEC §2.5 — Preissmann slot wMax for inlet cell. Same logic as
        // manhole: volume-equivalent rectangle → W is the slot's wMax.
        h_cell_slot_width[c]   = W;
    }

    // Compute precomputed tables for manhole/inlet cells
    for (int c = total_pipe_cells; c < n_cells_all; ++c) {
        XsectKey key{cell_shape_type[c], cell_width[c], cell_height[c]};
        auto it = table_cache.find(key);
        if (it == table_cache.end()) {
            std::vector<double> tbl;
            double A_full_dummy, P_full_dummy;
            pipe1d_compute_table(key.shape_type, key.w, key.h, A_full_dummy, P_full_dummy, tbl);
            it = table_cache.emplace(key, std::move(tbl)).first;
        }
        std::memcpy(&cell_tables[c * 3 * PIPE1D_TABLE_N],
                    it->second.data(), 3 * PIPE1D_TABLE_N * sizeof(double));
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
    // Per-cell arrays — sized [n_cells_all] (includes manhole/inlet cells)
    // d_cell_from_node / d_cell_to_node retained (Phase 2.3): godunov update
    // uses them to distinguish boundary faces (node_index < n_nodes) from
    // interior faces (vnode_index >= n_nodes) and derive eta_left/eta_right
    // from neighbouring cells at interior faces.
    alloc_d(reinterpret_cast<void**>(&dev->d_cell_from_node), static_cast<size_t>(n_cells_all) * sizeof(int32_t));
    alloc_d(reinterpret_cast<void**>(&dev->d_cell_to_node), static_cast<size_t>(n_cells_all) * sizeof(int32_t));
    alloc_d(reinterpret_cast<void**>(&dev->d_cell_length), static_cast<size_t>(n_cells_all) * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_cell_link_length), static_cast<size_t>(n_cells_all) * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_cell_area), static_cast<size_t>(n_cells_all) * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_cell_perim), static_cast<size_t>(n_cells_all) * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_cell_invert), static_cast<size_t>(n_cells_all) * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_cell_n), static_cast<size_t>(n_cells_all) * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_cell_link_k), static_cast<size_t>(n_cells_all) * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_cell_link_k_in), static_cast<size_t>(n_cells_all) * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_cell_link_k_out), static_cast<size_t>(n_cells_all) * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_cell_link_area), static_cast<size_t>(n_cells_all) * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_cell_shape_type), static_cast<size_t>(n_cells_all) * sizeof(int32_t));
    alloc_d(reinterpret_cast<void**>(&dev->d_cell_width), static_cast<size_t>(n_cells_all) * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_cell_owner_link), static_cast<size_t>(n_cells_all) * sizeof(int32_t));
    alloc_d(reinterpret_cast<void**>(&dev->d_cell_sub_idx), static_cast<size_t>(n_cells_all) * sizeof(int32_t));
    alloc_d(reinterpret_cast<void**>(&dev->d_cell_S0), static_cast<size_t>(n_cells_all) * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_cell_is_end), static_cast<size_t>(n_cells_all) * sizeof(int32_t));
    alloc_d(reinterpret_cast<void**>(&dev->d_cell_height), static_cast<size_t>(n_cells_all) * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_cell_tables), static_cast<size_t>(n_cells_all) * 3 * PIPE1D_TABLE_N * sizeof(double));
    // Phase 2.1 — Per-cell metadata arrays
    alloc_d(reinterpret_cast<void**>(&dev->d_cell_class), static_cast<size_t>(n_cells_all) * sizeof(int32_t));
    alloc_d(reinterpret_cast<void**>(&dev->d_cell_crown), static_cast<size_t>(n_cells_all) * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_cell_rim), static_cast<size_t>(n_cells_all) * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_cell_surface_area), static_cast<size_t>(n_cells_all) * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_cell_max_depth), static_cast<size_t>(n_cells_all) * sizeof(double));

    alloc_d(reinterpret_cast<void**>(&dev->d_A), static_cast<size_t>(n_cells_all) * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_Q), static_cast<size_t>(n_cells_all) * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_A_prev), static_cast<size_t>(n_cells_all) * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_Q_iter), static_cast<size_t>(n_cells_all) * sizeof(double));

    // RK2 stage-0 save buffers — allocated here so the first pipe1d_step
    // call has them ready (the lazy allocation inside godunov_step_internal
    // would re-allocate on the first call which is wasteful when the
    // mesh is built and the first step runs in quick succession).
    alloc_d(reinterpret_cast<void**>(&dev->d_A_start_save),
        static_cast<size_t>(n_cells_all) * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_Q_start_save),
        static_cast<size_t>(n_cells_all) * sizeof(double));
    dev->n_start_save_capacity = n_cells_all;

    alloc_d(reinterpret_cast<void**>(&dev->d_cell_y), static_cast<size_t>(n_cells_all) * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_cell_q), static_cast<size_t>(n_cells_all) * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_cell_fr), static_cast<size_t>(n_cells_all) * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_cell_h), static_cast<size_t>(n_cells_all) * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_cell_slot_width), static_cast<size_t>(n_cells_all) * sizeof(double));

    // MUSCL-minmod WSE-slope buffer (recon_method == 1 only).  Sized
    // [total_pipe_cells] because slope is only defined for pipe cells, not
    // MUSCL-minmod A and Q slope buffers (recon_method == 1)
    alloc_d(reinterpret_cast<void**>(&dev->d_slope_A),
        static_cast<size_t>(total_pipe_cells) * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_slope_Q),
        static_cast<size_t>(total_pipe_cells) * sizeof(double));
    CUDA_CHECK(cudaMemset(dev->d_slope_A, 0, static_cast<size_t>(total_pipe_cells) * sizeof(double)));
    CUDA_CHECK(cudaMemset(dev->d_slope_Q, 0, static_cast<size_t>(total_pipe_cells) * sizeof(double)));

    // Pre-compute slot CFL dt: the minimum CFL-safe timestep when the
    // Preissmann slot activates.  c_slot = sqrt(g * A_full / sw) where
    // sw = 0.01 * wMax (Sjoberg slot width).
    if (g_slot_cfl > 0.0) {
        constexpr double CFL = 0.5;
        dev->slot_cfl_dt = 1e12;
        for (int32_t c = 0; c < total_pipe_cells; ++c) {
            const double dx = fmax(cell_length[c], 1e-12);
            if (cell_shape_type[c] == 1) continue;
            const double sw = fmax(0.01 * cell_width[c], 1e-12);
            const double cs = sqrt(g_slot_cfl * cell_area[c] / sw);
            const double dt_cfl = CFL * dx / cs;
            if (dt_cfl < dev->slot_cfl_dt) dev->slot_cfl_dt = dt_cfl;
        }
    }

    // SPEC §DIAG — Instrumentation counters removed (Phase 2.1).

    // SPEC §2.2 — Virtual nodes retired (Phase 2.1). Interior faces use the unified
    // face mesh directly (face_class=INTERIOR with face_owner_R pointing at the
    // downstream cell). cell_from_node / cell_to_node are still populated on the
    // host (above) and uploaded for the godunov update kernel to detect
    // boundary vs interior faces per cell.

    // Copy data to device
    copy_h2d_i(dev->d_cell_from_node, cell_from_node.data(), static_cast<size_t>(n_cells_all));
    copy_h2d_i(dev->d_cell_to_node, cell_to_node.data(), static_cast<size_t>(n_cells_all));
    copy_h2d_d(dev->d_cell_length, cell_length.data(), static_cast<size_t>(n_cells_all));
    copy_h2d_d(dev->d_cell_link_length, cell_link_length.data(), static_cast<size_t>(n_cells_all));
    copy_h2d_d(dev->d_cell_area, cell_area.data(), static_cast<size_t>(n_cells_all));
    copy_h2d_d(dev->d_cell_perim, cell_perim.data(), static_cast<size_t>(n_cells_all));
    copy_h2d_d(dev->d_cell_invert, cell_invert.data(), static_cast<size_t>(n_cells_all));
    copy_h2d_d(dev->d_cell_n, cell_n.data(), static_cast<size_t>(n_cells_all));
    copy_h2d_d(dev->d_cell_link_k, cell_link_k.data(), static_cast<size_t>(n_cells_all));
    copy_h2d_d(dev->d_cell_link_k_in, cell_link_k_in.data(), static_cast<size_t>(n_cells_all));
    copy_h2d_d(dev->d_cell_link_k_out, cell_link_k_out.data(), static_cast<size_t>(n_cells_all));
    copy_h2d_d(dev->d_cell_link_area, cell_link_area.data(), static_cast<size_t>(n_cells_all));
    copy_h2d_i(dev->d_cell_shape_type, cell_shape_type.data(), static_cast<size_t>(n_cells_all));
    copy_h2d_d(dev->d_cell_width, cell_width.data(), static_cast<size_t>(n_cells_all));
    copy_h2d_i(dev->d_cell_owner_link, cell_owner_link.data(), static_cast<size_t>(n_cells_all));
    copy_h2d_i(dev->d_cell_sub_idx, cell_sub_idx.data(), static_cast<size_t>(n_cells_all));
    copy_h2d_d(dev->d_cell_S0, cell_S0.data(), static_cast<size_t>(n_cells_all));
    copy_h2d_i(dev->d_cell_is_end, cell_is_end.data(), static_cast<size_t>(n_cells_all));
    copy_h2d_d(dev->d_cell_height, cell_height.data(), static_cast<size_t>(n_cells_all));
    CUDA_CHECK(cudaMemcpy(dev->d_cell_tables, cell_tables.data(),
        static_cast<size_t>(n_cells_all) * 3 * PIPE1D_TABLE_N * sizeof(double),
        cudaMemcpyHostToDevice));

    // Phase 2.1 — Upload per-cell metadata
    copy_h2d_i(dev->d_cell_class, h_cell_class.data(), static_cast<size_t>(n_cells_all));
    copy_h2d_d(dev->d_cell_crown, h_cell_crown.data(), static_cast<size_t>(n_cells_all));
    copy_h2d_d(dev->d_cell_rim, h_cell_rim.data(), static_cast<size_t>(n_cells_all));
    copy_h2d_d(dev->d_cell_surface_area, h_cell_surface_area.data(), static_cast<size_t>(n_cells_all));
    copy_h2d_d(dev->d_cell_max_depth, h_cell_max_depth.data(), static_cast<size_t>(n_cells_all));
    // SPEC §2.5 — Upload per-cell Preissmann slot width. Replaces the prior
    // zero-memset (Gap G2): the slot surcharge branch in
    // xsect_getAofY_pressurised is gated on wMax > 0, so a zero slot width
    // disabled pressurised A for every cell. wMax is set per-cell-class:
    //   pipe cells    → link's primary width (D, b, or 2a depending on shape)
    //   manhole cells → manhole volume-equivalent width W
    //   inlet cells   → inlet volume-equivalent width W
    copy_h2d_d(dev->d_cell_slot_width, h_cell_slot_width.data(), static_cast<size_t>(n_cells_all));

    // SPEC §2.13.1 — Upload virtual-node → link mapping and per-face vnode index
    // (removed Phase 2.1): copy_h2d_i(dev->d_vnode_to_link, vnode_to_link_host.data(), static_cast<size_t>(n_vnodes));
    // (removed Phase 2.1): copy_h2d_i(dev->d_vnode_idx, vnode_idx_host.data(), static_cast<size_t>(2 * total_pipe_cells));

    // Upload node invert elevations
    // (removed Phase 2.1): copy_h2d_d(dev->d_node_invert, node_invert_elev, static_cast<size_t>(n_nodes));

    // Upload node surface areas (used by mass-balance kernel)
    // (removed Phase 2.1): copy_h2d_d(dev->d_node_surface_area, node_surface_area, static_cast<size_t>(n_nodes));
    // (removed Phase 2.1): copy_h2d_d(dev->d_node_max_depth, node_max_depth, static_cast<size_t>(n_nodes));

    // Audit F9: populate d_node_crown = node_invert + max connected link height.
    {
        std::vector<double> node_crown_v(n_nodes, 0.0);
        for (int32_t i = 0; i < n_links; ++i) {
            const double h_link = (link_shape_type && link_shape_type[i] != 0)
                ? (link_height ? link_height[i] : link_diameter[i])
                : link_diameter[i];
            const int32_t fn = link_from_node[i], tn = link_to_node[i];
            if (fn >= 0 && fn < n_nodes)
                node_crown_v[fn] = std::max(node_crown_v[fn], node_invert_elev[fn] + h_link);
            if (tn >= 0 && tn < n_nodes)
                node_crown_v[tn] = std::max(node_crown_v[tn], node_invert_elev[tn] + h_link);
        }
    // (removed Phase 2.1): copy_h2d_d(dev->d_node_crown, node_crown_v.data(), static_cast<size_t>(n_nodes));
    }

    // SPEC §2.10 — Populate d_node_rim from manhole cell data (invert + manhole_height).
    {
        std::vector<double> node_rim_v(n_nodes, 0.0);
        for (int32_t ii = 0; ii < n_manholes; ++ii) {
            const int32_t n = manhole_node[ii];
            if (n < 0 || n >= n_nodes) continue;
            const double H = manhole_cell_height[ii];
            node_rim_v[n] = node_invert_elev[n] + H;
        }
    // (removed Phase 2.1): copy_h2d_d(dev->d_node_rim, node_rim_v.data(), static_cast<size_t>(n_nodes));
    }

    // Initialize node depth to zero (caller uploads actual depths before each step)
    // d_node_depth and d_node_net_q are sized [n_node_slots] = [n_nodes + n_vnodes]
    // so virtual-node slots are zero-initialised (SPEC §2.1).
    // (removed Phase 2.1): CUDA_CHECK(cudaMemsetAsync(dev->d_node_depth, 0, static_cast<size_t>(n_node_slots) * sizeof(double), stream));
    // Initialize boundary flags to zero; exchange upload marks pipe-end/outfall nodes
    // (removed Phase 2.1): CUDA_CHECK(cudaMemsetAsync(dev->d_node_is_boundary, 0, static_cast<size_t>(n_nodes) * sizeof(int32_t), stream));
    // Initialize outfall flags to zero; caller uploads actual outfall flags
    // (removed Phase 2.1): CUDA_CHECK(cudaMemsetAsync(dev->d_node_is_outfall, 0, static_cast<size_t>(n_nodes) * sizeof(int32_t), stream));
    // (removed Phase 2.1): CUDA_CHECK(cudaMemsetAsync(dev->d_node_is_inlet, 0,  static_cast<size_t>(n_nodes) * sizeof(int32_t), stream));
    // (removed Phase 2.1): CUDA_CHECK(cudaMemsetAsync(dev->d_node_is_pipe_end, 0, static_cast<size_t>(n_nodes) * sizeof(int32_t), stream));
    // (removed Phase 2.1): CUDA_CHECK(cudaMemsetAsync(dev->d_node_net_q, 0, static_cast<size_t>(n_node_slots) * sizeof(double), stream));

    // Initialize pipe cell state: d_A = 0 (dry), d_Q = 0
    // Call swe2d_pipe1d_init_full() from Python if primed/full initial condition
    // is desired (e.g. for perpetual streams or pipes with baseflow).
    CUDA_CHECK(cudaMemsetAsync(dev->d_A, 0, static_cast<size_t>(n_cells_all) * sizeof(double), stream));
    CUDA_CHECK(cudaMemsetAsync(dev->d_Q, 0, static_cast<size_t>(n_cells_all) * sizeof(double), stream));
    CUDA_CHECK(cudaMemsetAsync(dev->d_A_prev, 0, static_cast<size_t>(n_cells_all) * sizeof(double), stream));
    CUDA_CHECK(cudaMemsetAsync(dev->d_Q_iter, 0, static_cast<size_t>(n_cells_all) * sizeof(double), stream));

    // SPEC §2.2 / §2.13 — Initialise derived per-cell arrays (diffusion-wave outputs)
    CUDA_CHECK(cudaMemsetAsync(dev->d_cell_y, 0, static_cast<size_t>(n_cells_all) * sizeof(double), stream));
    CUDA_CHECK(cudaMemsetAsync(dev->d_cell_q, 0, static_cast<size_t>(n_cells_all) * sizeof(double), stream));
    CUDA_CHECK(cudaMemsetAsync(dev->d_cell_fr, 0, static_cast<size_t>(n_cells_all) * sizeof(double), stream));
    CUDA_CHECK(cudaMemsetAsync(dev->d_cell_h, 0, static_cast<size_t>(n_cells_all) * sizeof(double), stream));
    // SPEC §2.5 — d_cell_slot_width is initialised ABOVE via copy_h2d_d from
    // h_cell_slot_width (the per-cell-class wMax values).  The previous
    // zero-memset here was Gap G2 — it disabled the slot surcharge branch
    // (gated on wMax > 0) for every cell.
    // (No memset for d_cell_slot_width — see h_cell_slot_width upload above.)

    // SPEC §2.8 — d_outfall_* arrays retired (Phase 2.1). Outfall state now lives on
    // the per-face d_ghost_outfall_* SoA arrays (allocated below in the face-mesh build).

    // ── Phase 2.4 — Build face mesh (INTERIOR + BC faces) ─────────────────────
    // Count interior pipe faces (between adjacent sub-cells within each link)
    int32_t n_interior_faces = 0;
    for (int32_t i = 0; i < n_links; ++i) {
        const int32_t n_sub = sub_cells_per_link[i];
        if (n_sub > 1) n_interior_faces += (n_sub - 1);
    }

    // Count BC faces from link-end nodes (regardless of classification — all links
    // have boundary faces at their ends for backward compatibility with the old
    // node-depth boundary condition).
    int32_t n_outfall_faces = 0;
    int32_t n_wall_faces = 0;
    int32_t n_inlet_bc_faces = 0;
    int32_t n_surface_pipe_end_faces = 0;

    // Build node→ghost mapping (for pre-step WSE update)
    std::vector<int32_t> ghost_node_idx;   // [n_outfall_faces] maps ghost index → node index

    {
        int32_t cell_cursor = 0;
        for (int32_t i = 0; i < n_links; ++i) {
            const int32_t n_sub = sub_cells_per_link[i];
            const int32_t first_cell = cell_cursor;
            const int32_t last_cell = cell_cursor + n_sub - 1;
            const int32_t from_node = link_from_node[i];
            const int32_t to_node = link_to_node[i];

            // Upstream end (from_node side)
            if (from_node >= 0 && from_node < n_nodes) {
                if (node_is_pipe_end && node_is_pipe_end[from_node]) {
                    ++n_surface_pipe_end_faces;
                } else if (node_is_inlet && node_is_inlet[from_node]) {
                    ++n_inlet_bc_faces;
                } else if (node_is_outfall && node_is_outfall[from_node]) {
                    ++n_outfall_faces;
                    ghost_node_idx.push_back(from_node);
                } else if (!node_is_outfall) {
                    // nullptr → backward compat: WALL_BC for all unclassified
                    ++n_wall_faces;
                } else {
                    // non-null but node not marked → WALL_BC
                    ++n_wall_faces;
                }
            }

            // Downstream end (to_node side)
            if (to_node >= 0 && to_node < n_nodes) {
                if (node_is_pipe_end && node_is_pipe_end[to_node]) {
                    ++n_surface_pipe_end_faces;
                } else if (node_is_inlet && node_is_inlet[to_node]) {
                    ++n_inlet_bc_faces;
                } else if (node_is_outfall && node_is_outfall[to_node]) {
                    ++n_outfall_faces;
                    ghost_node_idx.push_back(to_node);
                } else if (!node_is_outfall) {
                    // nullptr → backward compat: WALL_BC
                    ++n_wall_faces;
                } else {
                    // non-null but node not marked → WALL_BC
                    ++n_wall_faces;
                }
            }

            cell_cursor += n_sub;
        }
    }

    dev->n_outfall_faces = n_outfall_faces;
    dev->n_inlet_bc_faces = n_inlet_bc_faces;

    // Count SURFACE_2D_JUNCTION_OVERFLOW faces: one per manhole or inlet (class 5).
    // owner_R is set to -1 initially; swe2d_pipe1d_upload_junction_overflow_2d_cells
    // patches it from the actual 2D cell coupling.
    const int32_t n_overflow_faces = n_manholes + n_inlets;

    // Count storage→pipe INTERIOR faces: one per manhole/inlet that connects to a link.
    // Build a temporary node→storage-cell map from the raw node-id arrays.
    const int32_t first_manhole_cell = total_pipe_cells;
    const int32_t first_inlet_cell  = total_pipe_cells + n_manholes;
    std::vector<int32_t> node_to_storage(n_nodes, -1);
    for (int32_t mi = 0; mi < n_manholes; ++mi) {
        int32_t ni = manhole_node ? manhole_node[mi] : -1;
        if (ni >= 0 && ni < n_nodes) node_to_storage[ni] = first_manhole_cell + mi;
    }
    for (int32_t ii = 0; ii < n_inlets; ++ii) {
        int32_t ni = inlet_node ? inlet_node[ii] : -1;
        if (ni >= 0 && ni < n_nodes && node_to_storage[ni] < 0)
            node_to_storage[ni] = first_inlet_cell + ii;
    }
    int32_t n_storage_pipe_faces = 0;
    {
        int32_t cell_cursor_c = 0;
        for (int32_t i = 0; i < n_links; ++i) {
            const int32_t n_sub = sub_cells_per_link[i];
            const int32_t fn = link_from_node[i], tn = link_to_node[i];
            if (fn >= 0 && fn < n_nodes && node_to_storage[fn] >= 0) ++n_storage_pipe_faces;
            if (tn >= 0 && tn < n_nodes && node_to_storage[tn] >= 0) ++n_storage_pipe_faces;
            cell_cursor_c += n_sub;
        }
    }

    dev->n_storage_pipe_faces = n_storage_pipe_faces;

    const int32_t n_total_faces = n_interior_faces + n_outfall_faces + n_wall_faces
                                + n_inlet_bc_faces + n_surface_pipe_end_faces
                                + n_overflow_faces
                                + n_inlet_capture_faces + n_culvert_faces
                                + n_storage_pipe_faces;
    dev->n_faces = n_total_faces;

    // Allocate face arrays (sized for total faces)
    auto alloc_face = [&](auto** ptr, size_t elem_size, size_t count) {
        alloc_d(reinterpret_cast<void**>(ptr), elem_size * count);
    };
    alloc_face(&dev->d_face_owner_L, sizeof(int32_t), n_total_faces);
    alloc_face(&dev->d_face_owner_R, sizeof(int32_t), n_total_faces);
    alloc_face(&dev->d_face_class, sizeof(int32_t), n_total_faces);
    alloc_face(&dev->d_face_solve_mode, sizeof(int32_t), n_total_faces);
    alloc_face(&dev->d_face_dir, sizeof(double), n_total_faces);
    alloc_face(&dev->d_face_F_h, sizeof(double), n_total_faces);
    alloc_face(&dev->d_face_F_Q, sizeof(double), n_total_faces);
    alloc_face(&dev->d_face_invert, sizeof(double), n_total_faces);
    alloc_face(&dev->d_face_nx, sizeof(double), n_total_faces);
    alloc_face(&dev->d_face_ny, sizeof(double), n_total_faces);
    alloc_face(&dev->d_face_width, sizeof(double), n_total_faces);
    alloc_face(&dev->d_face_area, sizeof(double), n_total_faces);
    alloc_face(&dev->d_face_k_in, sizeof(double), n_total_faces);
    alloc_face(&dev->d_face_k_out, sizeof(double), n_total_faces);
    alloc_face(&dev->d_face_depth_safety, sizeof(double), n_total_faces);
    alloc_face(&dev->d_face_rim_elev, sizeof(double), n_total_faces);
    alloc_face(&dev->d_face_node_surface_area, sizeof(double), n_total_faces);
    alloc_face(&dev->d_face_ghost_idx, sizeof(int32_t), n_total_faces);

    // Build face host vectors (all types)
    std::vector<int32_t> face_owner_L(n_total_faces, -1);
    std::vector<int32_t> face_owner_R(n_total_faces, -1);
    std::vector<int32_t> face_class_v(n_total_faces, 0);
    std::vector<int32_t> face_solve_mode_v(n_total_faces, 0);
    std::vector<double>  face_dir_v(n_total_faces, 1.0);
    std::vector<double>  face_invert_v(n_total_faces, 0.0);
    std::vector<double>  face_width_v(n_total_faces, 0.0);
    std::vector<double>  face_area_v(n_total_faces, 0.0);
    std::vector<int32_t> face_ghost_idx_h(n_total_faces, -1);
    std::vector<double>  face_k_in_v(n_total_faces, 0.0);
    std::vector<double>  face_k_out_v(n_total_faces, 0.0);
    std::vector<double>  face_rim_elev_v(n_total_faces, 0.0);

    // ── Build INTERIOR faces (class 0) ──
    {
        int32_t face_idx = 0;
        int32_t face_cell_cursor = 0;
        for (int32_t i = 0; i < n_links; ++i) {
            const int32_t n_sub = sub_cells_per_link[i];
            for (int32_t s = 0; s < n_sub - 1; ++s) {
                const int32_t c = face_cell_cursor + s;
                face_owner_L[face_idx] = c;
                face_owner_R[face_idx] = c + 1;
                face_class_v[face_idx] = 0;   // INTERIOR
                face_solve_mode_v[face_idx] = 0; // Riemann
                face_dir_v[face_idx] = 1.0;
                face_invert_v[face_idx] = 0.5 * (cell_invert[static_cast<size_t>(c)]
                                                + cell_invert[static_cast<size_t>(c + 1)]);
                ++face_idx;
            }
            face_cell_cursor += n_sub;
        }
    }

    // ── Build BC faces (classes 1-5) unconditionally for all link ends ──
    {
        // Build reverse mapping: node_id → inlet index (for INLET_BC face owner_L)
        std::vector<int32_t> node_to_inlet_idx(n_nodes, -1);
        for (int32_t ii = 0; ii < n_inlets; ++ii) {
            const int32_t ni = inlet_node ? inlet_node[ii] : -1;
            if (ni >= 0 && ni < n_nodes) node_to_inlet_idx[ni] = ii;
        }
        // Build reverse mapping: node_id → manhole index (for overflow face owner_L)
        std::vector<int32_t> node_to_manhole_idx(n_nodes, -1);
        for (int32_t ii = 0; ii < n_manholes; ++ii) {
            const int32_t ni = manhole_node ? manhole_node[ii] : -1;
            if (ni >= 0 && ni < n_nodes) node_to_manhole_idx[ni] = ii;
        }

        int32_t face_idx = n_interior_faces;
        int32_t outfall_gi = 0;
        int32_t inlet_gi = 0;
        int32_t cell_cursor = 0;

        for (int32_t i = 0; i < n_links; ++i) {
            const int32_t n_sub = sub_cells_per_link[i];
            const int32_t first_cell = cell_cursor;
            const int32_t last_cell = cell_cursor + n_sub - 1;
            const int32_t from_node = link_from_node[i];
            const int32_t to_node = link_to_node[i];

            auto add_bc_face = [&](int32_t cell_idx, int32_t node_id, bool is_upstream) {
                if (node_id < 0 || node_id >= n_nodes) return;
                if (node_is_pipe_end && node_is_pipe_end[node_id]) {
                    // SURFACE_2D_PIPE_END (class 3) — owner_R = -1 placeholder;
                    // actual 2D cell index patched by upload_pipe_end_surface_faces.
                    face_owner_L[face_idx] = cell_idx;
                    face_owner_R[face_idx] = -1;
                    face_class_v[face_idx] = 3;
                    face_solve_mode_v[face_idx] = 0;
                    face_dir_v[face_idx] = is_upstream ? -1.0 : 1.0;
                    face_invert_v[face_idx] = cell_invert[cell_idx];
                    face_ghost_idx_h[face_idx] = -1;
                    ++face_idx;
                    face_k_in_v[face_idx - 1] = cell_link_k_in[cell_idx];
                    face_k_out_v[face_idx - 1] = cell_link_k_out[cell_idx];
                } else if (node_is_inlet && node_is_inlet[node_id]) {
                    // INLET_BC (class 2) — prescribed Q
                    // Attach face to the INLET_CELL (not the pipe end-cell)
                    const int32_t inlet_pos = node_to_inlet_idx[node_id];
                    const int32_t inlet_cell_index = (inlet_pos >= 0)
                        ? total_pipe_cells + n_manholes + inlet_pos
                        : cell_idx;
                    face_owner_L[face_idx] = inlet_cell_index;
                    face_owner_R[face_idx] = -1;
                    face_class_v[face_idx] = 2;
                    face_solve_mode_v[face_idx] = 0;
                    face_dir_v[face_idx] = is_upstream ? -1.0 : 1.0;
                    face_invert_v[face_idx] = cell_invert[cell_idx];
                    face_ghost_idx_h[face_idx] = inlet_gi;
                    ++inlet_gi;
                    ++face_idx;
                } else if (node_is_outfall && node_is_outfall[node_id]) {
                    // OUTFALL_BC (class 1) — ghost WSE updated from node_depth
                    // by swe2d_update_outfall_ghost_wse_kernel before each step.
                    face_owner_L[face_idx] = cell_idx;
                    face_owner_R[face_idx] = -1;
                    face_class_v[face_idx] = 1;
                    face_solve_mode_v[face_idx] = 0;
                    face_dir_v[face_idx] = is_upstream ? -1.0 : 1.0;
                    face_invert_v[face_idx] = cell_invert[cell_idx];
                    face_ghost_idx_h[face_idx] = outfall_gi;
                    ++outfall_gi;
                    ++face_idx;
                } else {
                    // WALL_BC (class 7) — pipe end with no explicit BC classification.
                    // Reflective ghost: zero net flux through the boundary.
                    face_owner_L[face_idx] = cell_idx;
                    face_owner_R[face_idx] = -1;
                    face_class_v[face_idx] = 7;
                    face_solve_mode_v[face_idx] = 0;
                    face_dir_v[face_idx] = is_upstream ? -1.0 : 1.0;
                    face_invert_v[face_idx] = cell_invert[cell_idx];
                    face_ghost_idx_h[face_idx] = -1;
                    ++face_idx;
                }
            };

            add_bc_face(first_cell, from_node, true);
            add_bc_face(last_cell, to_node, false);

            cell_cursor += n_sub;
        }

        // ── Build SURFACE_2D_JUNCTION_OVERFLOW faces (class 5) for manholes and inlets ──
        // owner_R = -1 initially; patched by upload_junction_overflow_state
        // from d_junction_2d_cell.  face_ghost_idx stores the network node index
        // so the upload function can find the matching junction.
        // Rim elevation from h_cell_rim[cell]; overflow triggers when
        // WSE exceeds rim.
        {
            int32_t oc_cursor = total_pipe_cells;  // first manhole cell index
            for (int32_t ii = 0; ii < n_manholes; ++ii, ++oc_cursor) {
                const int32_t n = manhole_node[ii];
                if (n < 0 || n >= n_nodes) continue;
                face_owner_L[face_idx] = oc_cursor;
                face_class_v[face_idx] = 5;
                face_solve_mode_v[face_idx] = 1;
                face_dir_v[face_idx] = 1.0;
                face_invert_v[face_idx] = cell_invert[oc_cursor];
                face_rim_elev_v[face_idx] = h_cell_rim[oc_cursor];
                face_width_v[face_idx] = fmax(cell_width[oc_cursor], 0.01);
                face_ghost_idx_h[face_idx] = n;
                ++face_idx;
            }
            // Inlet-cell overflow faces (same class 5 logic)
            int32_t ic_cursor = total_pipe_cells + n_manholes;
            for (int32_t ii = 0; ii < n_inlets; ++ii, ++ic_cursor) {
                const int32_t n = inlet_node[ii];
                if (n < 0 || n >= n_nodes) continue;
                face_owner_L[face_idx] = ic_cursor;
                face_class_v[face_idx] = 5;
                face_solve_mode_v[face_idx] = 1;
                face_dir_v[face_idx] = 1.0;
                face_invert_v[face_idx] = cell_invert[ic_cursor];
                face_rim_elev_v[face_idx] = h_cell_rim[ic_cursor];
                face_width_v[face_idx] = fmax(cell_width[ic_cursor], 0.01);
                face_ghost_idx_h[face_idx] = n;
                ++face_idx;
            }
        }

        // ── Phase F3 — Build SURFACE_2D_INLET faces (class 4) for HEC-22 capture ──
        // owner_L = INLET_CELL (sump storage), owner_R = coupled 2D surface cell.
        // Per-face HEC-22 geometry (grate/curb/slot, crest, cd, qmax) is uploaded
        // below into the d_face_inlet_* SoA, indexed by face_idx - first_inlet_capture_face.
        if (n_inlet_capture_faces > 0 && inlet_face_node && inlet_face_2d_cell) {
            const int32_t first_inlet_cell = total_pipe_cells + n_manholes;
            std::vector<int32_t> h_face_inlet_type(n_inlet_capture_faces, 0);
            std::vector<double>  h_face_inlet_grate_len(n_inlet_capture_faces, 0.0);
            std::vector<double>  h_face_inlet_grate_wid(n_inlet_capture_faces, 0.0);
            std::vector<double>  h_face_inlet_grate_open(n_inlet_capture_faces, 1.0);
            std::vector<double>  h_face_inlet_curb_len(n_inlet_capture_faces, 0.0);
            std::vector<double>  h_face_inlet_curb_ht(n_inlet_capture_faces, 0.0);
            std::vector<double>  h_face_inlet_curb_throat(n_inlet_capture_faces, 0.0);
            std::vector<double>  h_face_inlet_slot_len(n_inlet_capture_faces, 0.0);
            std::vector<double>  h_face_inlet_slot_wid(n_inlet_capture_faces, 0.0);
            std::vector<double>  h_face_inlet_crest(n_inlet_capture_faces, 0.0);
            std::vector<double>  h_face_inlet_cd(n_inlet_capture_faces, 0.5);
            std::vector<double>  h_face_inlet_qmax(n_inlet_capture_faces, 0.0);
            std::unordered_map<int32_t, int32_t> node_to_inlet_cell;
            for (int32_t ii = 0; ii < n_inlets; ++ii) {
                const int32_t ni = inlet_node ? inlet_node[ii] : -1;
                if (ni >= 0) node_to_inlet_cell[ni] = first_inlet_cell + ii;
            }
            for (int32_t ii = 0; ii < n_inlet_capture_faces; ++ii) {
                const int32_t ni = inlet_face_node[ii];
                const int32_t c2d = inlet_face_2d_cell[ii];
                auto it = node_to_inlet_cell.find(ni);
                if (it == node_to_inlet_cell.end() || c2d < 0) continue;
                face_owner_L[face_idx] = it->second;
                face_owner_R[face_idx] = c2d;
                face_class_v[face_idx] = 4;       // SURFACE_2D_INLET
                face_solve_mode_v[face_idx] = 1;  // source-sink (HEC-22 weir/orifice)
                face_dir_v[face_idx] = 1.0;
                face_invert_v[face_idx] = cell_invert[it->second];
                face_ghost_idx_h[face_idx] = ii;   // index into d_face_inlet_* SoA
                // face_width = grate length (or curb/slot length depending on type)
                // Used by the weir equation in swe2d_unified_face_flux_kernel class-4 branch.
                face_width_v[face_idx] = fmax(h_face_inlet_grate_len[ii], 0.01);
                // Copy per-face HEC-22 attrs (nullptr-safe)
                auto copy_opt = [](double& dst, const double* src, int32_t i, double fallback) {
                    dst = (src) ? src[i] : fallback;
                };
                h_face_inlet_type[ii]            = inlet_face_type            ? inlet_face_type[ii]            : 0;
                copy_opt(h_face_inlet_grate_len[ii],  inlet_face_grate_len,  ii, 0.0);
                copy_opt(h_face_inlet_grate_wid[ii],  inlet_face_grate_wid,  ii, 0.0);
                copy_opt(h_face_inlet_grate_open[ii], inlet_face_grate_open, ii, 1.0);
                copy_opt(h_face_inlet_curb_len[ii],   inlet_face_curb_len,   ii, 0.0);
                copy_opt(h_face_inlet_curb_ht[ii],    inlet_face_curb_ht,    ii, 0.0);
                copy_opt(h_face_inlet_curb_throat[ii],inlet_face_curb_throat,ii, 0.0);
                copy_opt(h_face_inlet_slot_len[ii],   inlet_face_slot_len,   ii, 0.0);
                copy_opt(h_face_inlet_slot_wid[ii],   inlet_face_slot_wid,   ii, 0.0);
                copy_opt(h_face_inlet_crest[ii],      inlet_face_crest,      ii, 0.0);
                copy_opt(h_face_inlet_cd[ii],         inlet_face_cd,         ii, 0.5);
                copy_opt(h_face_inlet_qmax[ii],       inlet_face_qmax,       ii, 0.0);
                ++face_idx;
            }
            // Allocate + upload HEC-22 SoA (sized [n_inlet_capture_faces])
            alloc_d(reinterpret_cast<void**>(&dev->d_face_inlet_type),
                    static_cast<size_t>(n_inlet_capture_faces) * sizeof(int32_t));
            alloc_d(reinterpret_cast<void**>(&dev->d_face_inlet_grate_len),
                    static_cast<size_t>(n_inlet_capture_faces) * sizeof(double));
            alloc_d(reinterpret_cast<void**>(&dev->d_face_inlet_grate_wid),
                    static_cast<size_t>(n_inlet_capture_faces) * sizeof(double));
            alloc_d(reinterpret_cast<void**>(&dev->d_face_inlet_grate_open),
                    static_cast<size_t>(n_inlet_capture_faces) * sizeof(double));
            alloc_d(reinterpret_cast<void**>(&dev->d_face_inlet_curb_len),
                    static_cast<size_t>(n_inlet_capture_faces) * sizeof(double));
            alloc_d(reinterpret_cast<void**>(&dev->d_face_inlet_curb_ht),
                    static_cast<size_t>(n_inlet_capture_faces) * sizeof(double));
            alloc_d(reinterpret_cast<void**>(&dev->d_face_inlet_curb_throat),
                    static_cast<size_t>(n_inlet_capture_faces) * sizeof(double));
            alloc_d(reinterpret_cast<void**>(&dev->d_face_inlet_slot_len),
                    static_cast<size_t>(n_inlet_capture_faces) * sizeof(double));
            alloc_d(reinterpret_cast<void**>(&dev->d_face_inlet_slot_wid),
                    static_cast<size_t>(n_inlet_capture_faces) * sizeof(double));
            alloc_d(reinterpret_cast<void**>(&dev->d_face_inlet_crest),
                    static_cast<size_t>(n_inlet_capture_faces) * sizeof(double));
            alloc_d(reinterpret_cast<void**>(&dev->d_face_inlet_cd),
                    static_cast<size_t>(n_inlet_capture_faces) * sizeof(double));
            alloc_d(reinterpret_cast<void**>(&dev->d_face_inlet_qmax),
                    static_cast<size_t>(n_inlet_capture_faces) * sizeof(double));
            dev->n_inlet_capture_faces = n_inlet_capture_faces;
            copy_h2d_i(dev->d_face_inlet_type, h_face_inlet_type.data(), static_cast<size_t>(n_inlet_capture_faces));
            copy_h2d_d(dev->d_face_inlet_grate_len, h_face_inlet_grate_len.data(), static_cast<size_t>(n_inlet_capture_faces));
            copy_h2d_d(dev->d_face_inlet_grate_wid, h_face_inlet_grate_wid.data(), static_cast<size_t>(n_inlet_capture_faces));
            copy_h2d_d(dev->d_face_inlet_grate_open, h_face_inlet_grate_open.data(), static_cast<size_t>(n_inlet_capture_faces));
            copy_h2d_d(dev->d_face_inlet_curb_len, h_face_inlet_curb_len.data(), static_cast<size_t>(n_inlet_capture_faces));
            copy_h2d_d(dev->d_face_inlet_curb_ht, h_face_inlet_curb_ht.data(), static_cast<size_t>(n_inlet_capture_faces));
            copy_h2d_d(dev->d_face_inlet_curb_throat, h_face_inlet_curb_throat.data(), static_cast<size_t>(n_inlet_capture_faces));
            copy_h2d_d(dev->d_face_inlet_slot_len, h_face_inlet_slot_len.data(), static_cast<size_t>(n_inlet_capture_faces));
            copy_h2d_d(dev->d_face_inlet_slot_wid, h_face_inlet_slot_wid.data(), static_cast<size_t>(n_inlet_capture_faces));
            copy_h2d_d(dev->d_face_inlet_crest, h_face_inlet_crest.data(), static_cast<size_t>(n_inlet_capture_faces));
            copy_h2d_d(dev->d_face_inlet_cd, h_face_inlet_cd.data(), static_cast<size_t>(n_inlet_capture_faces));
            copy_h2d_d(dev->d_face_inlet_qmax, h_face_inlet_qmax.data(), static_cast<size_t>(n_inlet_capture_faces));
        }

        // ── Storage→pipe INTERIOR faces (class 0, RIEMANN mode) ──────────
        // For each manhole / inlet cell at a drainage node, create an INTERIOR
        // face to the adjacent pipe sub-cell of the connected link so water can
        // flow between the storage cell and the pipe network.
        if (n_storage_pipe_faces > 0) {
            int32_t sc_cursor = 0;
            for (int32_t i = 0; i < n_links; ++i) {
                const int32_t n_sub = sub_cells_per_link[i];
                const int32_t fn = link_from_node[i], tn = link_to_node[i];
                const int32_t first = sc_cursor;
                const int32_t last  = sc_cursor + n_sub - 1;
                if (fn >= 0 && fn < n_nodes) {
                    int32_t sc = node_to_storage[fn];
                    if (sc >= 0) {
                        face_owner_L[face_idx] = sc;
                        face_owner_R[face_idx] = first;
                        face_class_v[face_idx] = 8;       // STORAGE_PIPE (weir/orifice)
                        face_solve_mode_v[face_idx] = 0;
                        face_dir_v[face_idx] = 1.0;
                        // Crest = pipe invert at the first pipe cell
                        face_invert_v[face_idx] = cell_invert[first];
                        face_width_v[face_idx] = fmax(cell_width[first], 0.01);
                        face_area_v[face_idx] = cell_area[first];
                        face_ghost_idx_h[face_idx] = -1;
                        // Loss coefficients from node-level (preferred) or link-level.
                        // face_k_in = entrance loss (Ke), face_k_out = exit loss (Kx).
                        // The weir/orifice coefficients Cw/Cd are fixed constants.
                        {
                            const double nk = (node_inlet_loss_k && fn < n_nodes)
                                            ? h_node_inlet_loss_k[fn] : 0.0;
                            face_k_in_v[face_idx] = (nk > 0.0) ? nk : cell_link_k_in[first];
                        }
                        {
                            const double nk = (node_outlet_loss_k && fn < n_nodes)
                                            ? h_node_outlet_loss_k[fn] : 0.0;
                            face_k_out_v[face_idx] = (nk > 0.0) ? nk : cell_link_k_out[first];
                        }
                        ++face_idx;
                    }
                }
                if (tn >= 0 && tn < n_nodes) {
                    int32_t sc = node_to_storage[tn];
                    if (sc >= 0) {
                        face_owner_L[face_idx] = sc;
                        face_owner_R[face_idx] = last;
                        face_class_v[face_idx] = 8;       // STORAGE_PIPE (weir/orifice)
                        face_solve_mode_v[face_idx] = 0;
                        face_dir_v[face_idx] = -1.0;
                        // Crest = pipe invert at the last pipe cell
                        face_invert_v[face_idx] = cell_invert[last];
                        face_width_v[face_idx] = fmax(cell_width[last], 0.01);
                        face_area_v[face_idx] = cell_area[last];
                        face_ghost_idx_h[face_idx] = -1;
                        // Loss coefficients from node-level (preferred) or link-level.
                        {
                            const double nk = (node_inlet_loss_k && tn < n_nodes)
                                            ? h_node_inlet_loss_k[tn] : 0.0;
                            face_k_in_v[face_idx] = (nk > 0.0) ? nk : cell_link_k_in[last];
                        }
                        {
                            const double nk = (node_outlet_loss_k && tn < n_nodes)
                                            ? h_node_outlet_loss_k[tn] : 0.0;
                            face_k_out_v[face_idx] = (nk > 0.0) ? nk : cell_link_k_out[last];
                        }
                        ++face_idx;
                    }
                }
                sc_cursor += n_sub;
            }
        }

        // ── Phase F3 — Build CULVERT faces (class 6) for 2D-to-2D structures ──
        // owner_L = donor 2D cell, owner_R = receiver 2D cell.
        // face_ghost_idx indexes into d_ghost_culvert_struct_idx → d_structure_flows.
        if (n_culvert_faces > 0 && culvert_face_donor_2d_cell
            && culvert_face_receiver_2d_cell && culvert_struct_idx) {
            std::vector<int32_t> h_ghost_culvert_struct_idx(n_culvert_faces, 0);
            for (int32_t ii = 0; ii < n_culvert_faces; ++ii) {
                const int32_t c_d = culvert_face_donor_2d_cell[ii];
                const int32_t c_r = culvert_face_receiver_2d_cell[ii];
                const int32_t s_i = culvert_struct_idx[ii];
                if (c_d < 0 || c_r < 0) continue;
                face_owner_L[face_idx] = c_d;
                face_owner_R[face_idx] = c_r;
                face_class_v[face_idx] = 6;       // CULVERT
                face_solve_mode_v[face_idx] = 0;  // structure-flow driven
                face_dir_v[face_idx] = 1.0;
                face_invert_v[face_idx] = 0.0;
                face_ghost_idx_h[face_idx] = ii;
                h_ghost_culvert_struct_idx[ii] = s_i;
                ++face_idx;
            }
            // Allocate + upload culvert SoA
            alloc_d(reinterpret_cast<void**>(&dev->d_ghost_culvert_struct_idx),
                    static_cast<size_t>(n_culvert_faces) * sizeof(int32_t));
            copy_h2d_i(dev->d_ghost_culvert_struct_idx,
                       h_ghost_culvert_struct_idx.data(),
                       static_cast<size_t>(n_culvert_faces));
            dev->n_culvert_faces = n_culvert_faces;
        }

        // ── Phase F3 — Allocate d_structure_flows for class 6 read ──
        if (n_structures > 0 && structure_flows) {
            alloc_d(reinterpret_cast<void**>(&dev->d_structure_flows),
                    static_cast<size_t>(n_structures) * sizeof(double));
            copy_h2d_d(dev->d_structure_flows, structure_flows, static_cast<size_t>(n_structures));
        } else {
            // Always allocate a 1-element scratch so the unified kernel doesn't crash
            // on a null d_structure_flows pointer for class 6.
            if (!dev->d_structure_flows) {
                alloc_d(reinterpret_cast<void**>(&dev->d_structure_flows), sizeof(double));
                CUDA_CHECK(cudaMemsetAsync(dev->d_structure_flows, 0, sizeof(double), stream));
            }
        }
    }

    // Upload face data
    copy_h2d_i(dev->d_face_owner_L, face_owner_L.data(), static_cast<size_t>(n_total_faces));
    copy_h2d_i(dev->d_face_owner_R, face_owner_R.data(), static_cast<size_t>(n_total_faces));
    copy_h2d_i(dev->d_face_class, face_class_v.data(), static_cast<size_t>(n_total_faces));

    // Cache host copies: the 2D solver on the GUI thread can corrupt device
    // face arrays via cross-thread CUDA pool aliasing (cudaFree on GUI thread
    // returns memory to shared pool, cudaMalloc on worker thread picks up the
    // same address, then 2D solver graph-replay writes to the stale address).
    // The face flux function checks the device copy against this cache and
    // re-uploads if corrupted.
    dev->h_face_class_cache.assign(face_class_v.begin(), face_class_v.end());
    dev->h_face_owner_L_cache.assign(face_owner_L.begin(), face_owner_L.end());
    dev->h_face_owner_R_cache.assign(face_owner_R.begin(), face_owner_R.end());
    copy_h2d_i(dev->d_face_solve_mode, face_solve_mode_v.data(), static_cast<size_t>(n_total_faces));
    copy_h2d_d(dev->d_face_dir, face_dir_v.data(), static_cast<size_t>(n_total_faces));
    copy_h2d_d(dev->d_face_invert, face_invert_v.data(), static_cast<size_t>(n_total_faces));

    // Zero scratch and default arrays (face_width/area/rim/surf populated below)
    CUDA_CHECK(cudaMemsetAsync(dev->d_face_F_h, 0, static_cast<size_t>(n_total_faces) * sizeof(double), stream));
    CUDA_CHECK(cudaMemsetAsync(dev->d_face_F_Q, 0, static_cast<size_t>(n_total_faces) * sizeof(double), stream));
    CUDA_CHECK(cudaMemsetAsync(dev->d_face_nx, 0, static_cast<size_t>(n_total_faces) * sizeof(double), stream));
    CUDA_CHECK(cudaMemsetAsync(dev->d_face_ny, 0, static_cast<size_t>(n_total_faces) * sizeof(double), stream));
    CUDA_CHECK(cudaMemsetAsync(dev->d_face_depth_safety, 0, static_cast<size_t>(n_total_faces) * sizeof(double), stream));
    CUDA_CHECK(cudaMemsetAsync(dev->d_face_width, 0, static_cast<size_t>(n_total_faces) * sizeof(double), stream));
    CUDA_CHECK(cudaMemsetAsync(dev->d_face_area, 0, static_cast<size_t>(n_total_faces) * sizeof(double), stream));
    CUDA_CHECK(cudaMemsetAsync(dev->d_face_rim_elev, 0, static_cast<size_t>(n_total_faces) * sizeof(double), stream));
    CUDA_CHECK(cudaMemsetAsync(dev->d_face_node_surface_area, 0, static_cast<size_t>(n_total_faces) * sizeof(double), stream));
    CUDA_CHECK(cudaMemsetAsync(dev->d_face_ghost_idx, -1, static_cast<size_t>(n_total_faces) * sizeof(int32_t), stream));

    // Overwrite face_width for faces that need a non-zero value (SURFACE_2D_INLET).
    // All other faces keep the memset zero — the unified kernel uses
    // face_width[k] as weir length for source-sink faces (class 4, 5).
    copy_h2d_d(dev->d_face_width, face_width_v.data(), static_cast<size_t>(n_total_faces));
    copy_h2d_d(dev->d_face_area, face_area_v.data(), static_cast<size_t>(n_total_faces));
    copy_h2d_d(dev->d_face_k_in, face_k_in_v.data(), static_cast<size_t>(n_total_faces));
    copy_h2d_d(dev->d_face_k_out, face_k_out_v.data(), static_cast<size_t>(n_total_faces));
    copy_h2d_d(dev->d_face_rim_elev, face_rim_elev_v.data(), static_cast<size_t>(n_total_faces));

    // Upload ghost state indices for BC faces
    if (n_outfall_faces > 0 || n_inlet_bc_faces > 0) {
        copy_h2d_i(dev->d_face_ghost_idx, face_ghost_idx_h.data(), static_cast<size_t>(n_total_faces));
    }

    // ── Phase 2.4 — Allocate ghost-state SoA arrays for BC faces ──
    if (n_outfall_faces > 0) {
        alloc_d(reinterpret_cast<void**>(&dev->d_ghost_outfall_mode),
                static_cast<size_t>(n_outfall_faces) * sizeof(int32_t));
        alloc_d(reinterpret_cast<void**>(&dev->d_ghost_outfall_fixed_wse),
                static_cast<size_t>(n_outfall_faces) * sizeof(double));
        alloc_d(reinterpret_cast<void**>(&dev->d_ghost_outfall_rating),
                static_cast<size_t>(n_outfall_faces) * MAX_RATING_POINTS * 2 * sizeof(double));
        alloc_d(reinterpret_cast<void**>(&dev->d_ghost_outfall_rating_n),
                static_cast<size_t>(n_outfall_faces) * sizeof(int32_t));
        alloc_d(reinterpret_cast<void**>(&dev->d_ghost_outfall_tabular),
                static_cast<size_t>(n_outfall_faces) * MAX_TABULAR_POINTS * 2 * sizeof(double));
        alloc_d(reinterpret_cast<void**>(&dev->d_ghost_outfall_tabular_n),
                static_cast<size_t>(n_outfall_faces) * sizeof(int32_t));
        alloc_d(reinterpret_cast<void**>(&dev->d_ghost_outfall_link_S0),
                static_cast<size_t>(n_outfall_faces) * sizeof(double));
        alloc_d(reinterpret_cast<void**>(&dev->d_ghost_outfall_node_idx),
                static_cast<size_t>(n_outfall_faces) * sizeof(int32_t));
        CUDA_CHECK(cudaMemsetAsync(dev->d_ghost_outfall_mode, 0,
                              static_cast<size_t>(n_outfall_faces) * sizeof(int32_t)));
        CUDA_CHECK(cudaMemsetAsync(dev->d_ghost_outfall_fixed_wse, 0,
                              static_cast<size_t>(n_outfall_faces) * sizeof(double)));
        CUDA_CHECK(cudaMemsetAsync(dev->d_ghost_outfall_rating, 0,
                              static_cast<size_t>(n_outfall_faces) * MAX_RATING_POINTS * 2 * sizeof(double)));
        CUDA_CHECK(cudaMemsetAsync(dev->d_ghost_outfall_rating_n, 0,
                              static_cast<size_t>(n_outfall_faces) * sizeof(int32_t)));
        CUDA_CHECK(cudaMemsetAsync(dev->d_ghost_outfall_tabular, 0,
                              static_cast<size_t>(n_outfall_faces) * MAX_TABULAR_POINTS * 2 * sizeof(double)));
        CUDA_CHECK(cudaMemsetAsync(dev->d_ghost_outfall_tabular_n, 0,
                              static_cast<size_t>(n_outfall_faces) * sizeof(int32_t)));
        CUDA_CHECK(cudaMemsetAsync(dev->d_ghost_outfall_link_S0, 0,
                              static_cast<size_t>(n_outfall_faces) * sizeof(double)));
        // Upload ghost→node mapping for backward-compat WSE update
        if (ghost_node_idx.size() >= static_cast<size_t>(n_outfall_faces)) {
            copy_h2d_i(dev->d_ghost_outfall_node_idx, ghost_node_idx.data(),
                       static_cast<size_t>(n_outfall_faces));
        }
        // Initialize OUTFALL_BC mode to OUTFALL_FIXED_WSE (2) so the unified
        // kernel uses d_ghost_outfall_fixed_wse[gi] as the ghost WSE. The
        // fixed_wse array is updated before each step by the ghost WSE
        // update kernel.
        std::vector<int32_t> outfall_mode_default(n_outfall_faces, 0);  // FREE
        copy_h2d_i(dev->d_ghost_outfall_mode, outfall_mode_default.data(),
                   static_cast<size_t>(n_outfall_faces));
    }
    if (n_inlet_bc_faces > 0) {
        alloc_d(reinterpret_cast<void**>(&dev->d_ghost_inlet_Q),
                static_cast<size_t>(n_inlet_bc_faces) * sizeof(double));
        CUDA_CHECK(cudaMemsetAsync(dev->d_ghost_inlet_Q, 0,
                              static_cast<size_t>(n_inlet_bc_faces) * sizeof(double)));
    }
    // CULVERT ghost-state SoA will be allocated when culvert geometry is available

    // Sync the pipe1d stream — all allocations, memsets, and memcpys above
    // are asynchronous on the private stream.  The caller expects the mesh
    // to be fully resident on the device when this function returns.
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

/// Device-side lookup: given current area A, full area A_full, full perimeter P_full,
/// and a pointer to the per-cell table [2 * TABLE_N doubles], return wetted perimeter P
/// and top width T.
__device__ __forceinline__ void pipe1d_lookup_geometry(
    double A, double A_full, double P_full,
    const double* table, int table_N,
    double& P, double& T, double& I1)
{
    if (A <= 0.0) {
        P = 0.0;
        T = 0.0;
        I1 = 0.0;
        return;
    }
    double frac = A * (1.0 / fmax(1e-20, A_full));
    frac = fmin(1.0, fmax(0.0, frac));
    double f = frac * table_N;
    int idx = min(table_N - 2, max(0, int(f)));
    double t = f - idx;
    P = P_full * (table[idx] + t * (table[idx + 1] - table[idx]));
    T = table[table_N + idx] + t * (table[table_N + idx + 1] - table[table_N + idx]);
    I1 = table[2 * table_N + idx] + t * (table[2 * table_N + idx + 1] - table[2 * table_N + idx]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper device functions for OUTFALL_BC face class (normal depth / rating WSE)
// ─────────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ double pipe1d_circular_normal_depth_bisect(
    double Q, double n_mann, double S0, double D, double h_min)
{
    if (D <= 0.0 || n_mann <= 0.0 || S0 <= 1e-12 || Q <= 0.0) return fmax(h_min, 0.0);
    double y_lo = fmax(h_min, 1e-9), y_hi = D;
    for (int it = 0; it < 50; ++it) {
        const double y = 0.5 * (y_lo + y_hi);
        const double Qc = (1.0 / n_mann) * xsect_getAofY_circular(D, y) *
                          pow(fmax(xsect_getRofY_circular(D, y), 1e-12), 2.0/3.0) * sqrt(S0);
        const double r = Qc - Q;
        if (fabs(r) < 1e-6 * Q) return y;
        if (r > 0.0) y_hi = y; else y_lo = y;
        if (y_hi - y_lo < 1e-9) return y;
    }
    return 0.5 * (y_lo + y_hi);
}

__device__ __forceinline__ double pipe1d_rating_curve_invert(
    double Q, const double* __restrict__ w, const double* __restrict__ qv, int n)
{
    if (n <= 0 || !w || !qv) return 0.0;
    if (n == 1 || Q <= qv[0]) return w[0];
    if (Q >= qv[n-1]) return w[n-1];
    double lo = w[0], hi = w[n-1];
    for (int it = 0; it < 50; ++it) {
        const double wm = 0.5 * (lo + hi);
        int ilo = 0, ihi = n - 1;
        while (ihi - ilo > 1) { int m = (ilo + ihi) >> 1; if (w[m] <= wm) ilo = m; else ihi = m; }
        const double t = (wm - w[ilo]) / fmax(1e-12, w[ihi] - w[ilo]);
        const double Qm = qv[ilo] + t * (qv[ihi] - qv[ilo]);
        const double r = Qm - Q;
        if (fabs(r) < 1e-6 * fmax(Q, 1e-9)) return wm;
        if (r > 0.0) hi = wm; else lo = wm;
        if (hi - lo < 1e-9) return wm;
    }
    return 0.5 * (lo + hi);
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 2.4 — Ghost WSE update kernel (backward-compat node-depth BC)
//
// One thread per outfall face.  For OUTFALL_BC faces with mode FIXED_WSE (2),
// sets d_ghost_outfall_fixed_wse[gi] = node_invert[node] + node_depth[node],
// so the downstream ghost WSE reflects the current node state.
// Called by the host wrapper before the unified face flux kernel.
// ─────────────────────────────────────────────────────────────────────────────
// Phase 2.3 — Update outfall ghost WSE from end-cell state (not node_depth)
// For OUTFALL_BC faces with mode FIXED_WSE (2), reads the end-cell's depth
// so the unified kernel sees the current boundary condition.
__global__ void swe2d_update_outfall_ghost_wse_kernel(
    int32_t                     n_outfall_faces,
    const int32_t* __restrict__ d_face_owner_L,
    double*                     d_ghost_outfall_fixed_wse,
    const double*  __restrict__ cell_h,
    const double*  __restrict__ cell_invert,
    int32_t                     n_cells_all)
{
    int32_t gi = blockIdx.x * blockDim.x + threadIdx.x;
    if (gi >= n_outfall_faces) return;
    if (!d_face_owner_L || !d_ghost_outfall_fixed_wse || !cell_h) return;
    const int32_t c = d_face_owner_L[gi];
    if (c < 0 || c >= n_cells_all) return;
    // Store the cell WSE (absolute elevation) so the OUTFALL_BC FIXED_WSE
    // handler can compute the ghost depth as y_R = ghost_wse - inv_L.
    double wse = cell_h[c] + (cell_invert ? cell_invert[c] : 0.0);
    d_ghost_outfall_fixed_wse[gi] = fmax(0.0, wse);
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 2.4 — Unified face-flux kernel (two-pass split for race fix)
//
// One thread per face. Dispatch on face_class[k]:
//   0 = INTERIOR        — HLLE flux between cells L and R
//   1 = OUTFALL_BC      — HLLE between cell L and ghost state
//   2 = INLET_BC        — HLLE between cell L and ghost prescribed-Q state
//   3 = SURFACE_2D_PIPE_END — HLLC between pipe cell L and 2D cell R
//   4 = SURFACE_2D_INLET    — Source-sink (weir/orifice, HEC-22)
//   5 = SURFACE_2D_JUNCTION_OVERFLOW — Source-sink (weir)
//   6 = CULVERT          — Structure-flow driven
//
// Solve modes:
//   0 = Riemann (HLLC) for classes 0,1,2,3
//   1 = Source-sink (weir/orifice) for classes 4,5
//
// For Riemann-mode faces, writes face_F_h[k] / face_F_Q[k] consumed by the
// fold kernel.  For SURFACE_2D_PIPE_END, also writes d_ext_struct_flux_*
// via atomicAdd.  For source-sink faces, writes face_F_h[k] (mass rate)
// and d_ext_struct_flux_* (2D side).
//
// === Two-pass split (race fix 2026-07-22) ===
// Pass 1 (pass=1): classes 3,4,5,6 — atomicAdd to d_A (class 3/4/5) and
//   d_ext_struct_flux_*.  This kernel DOES NOT read d_A (only the L-cell
//   state needed to compute the exchange flux).
// Pass 2 (pass=2): classes 0,1,2 — reads d_A (which is now consistent
//   post-pass-1) and computes face fluxes.  This kernel DOES NOT write
//   to d_A, so no race with pass-1 atomicAdds.
//
// Why split: in the original unified kernel, class-3/4/5 threads did
// atomicAdd(&d_A[L_end], ...) while class-0 INTERIOR threads read
// cell_A[L_end] non-atomically.  When an end cell (cell 0 or N-1) was
// L for both a SURFACE_2D face AND an adjacent class-0 face, the read
// saw a stale or fresh value depending on GPU scheduling — interior
// face flux F = 0.5*(Q_L + Q_R - c_wave*(A_R - A_L)) was computed with
// the wrong A gradient, so end-exchange mass didn't propagate into the
// pipe interior (Q stayed near zero).
// ─────────────────────────────────────────────────────────────────────────────
__global__ void swe2d_unified_face_flux_kernel(
    int32_t                     pass,
    int32_t                     n_faces,
    const int32_t* __restrict__ face_owner_L,
    const int32_t* __restrict__ face_owner_R,
    const int32_t* __restrict__ face_class,
    const int32_t* __restrict__ face_solve_mode,
    const double*  __restrict__ face_dir,
    double*                     face_F_h,
    double*                     face_F_Q,
    const double*  __restrict__ cell_A,
    const double*  __restrict__ cell_Q,
    const double*  __restrict__ cell_y,
    const double*  __restrict__ cell_invert,
    const double*  __restrict__ cell_length,
    const double*  __restrict__ cell_n,
    const double*  __restrict__ cell_width,
    const double*  __restrict__ cell_height,
    const double*  __restrict__ cell_area_full,
    const double*  __restrict__ cell_perim,
    const double*  __restrict__ cell_tables,
    const int32_t* __restrict__ cell_shape_type,
    int32_t                     n_cells_all,
    double                      dt,
    double                      g,
    int32_t                     table_N,
    // ── MUSCL-MC reconstruction slopes (nullptr = no reconstruction) ──
    const double*  __restrict__ d_slope_A,
    const double*  __restrict__ d_slope_Q,
    // ── 2D solver state (for SURFACE_2D_* faces) ──
    const double*  __restrict__ cell_h_2d,
    const double*  __restrict__ cell_hu_2d,
    const double*  __restrict__ cell_hv_2d,
    const double*  __restrict__ cell_zb_2d,
    double*                     d_ext_struct_flux_h,
    double*                     d_ext_struct_flux_hu,
    double*                     d_ext_struct_flux_hv,
    double*                     d_A,
    int32_t                     n_cells_2d,
    // ── Ghost-state SoA ──
    const int32_t* __restrict__ d_ghost_outfall_mode,
    const double*  __restrict__ d_ghost_outfall_fixed_wse,
    const double*  __restrict__ d_ghost_outfall_rating,
    const int32_t* __restrict__ d_ghost_outfall_rating_n,
    const double*  __restrict__ d_ghost_outfall_tabular,
    const int32_t* __restrict__ d_ghost_outfall_tabular_n,
    const double*  __restrict__ d_ghost_outfall_link_S0,
    int32_t                     n_outfall_faces,
    const double*  __restrict__ d_ghost_inlet_Q,
    int32_t                     n_inlet_bc_faces,
    const int32_t* __restrict__ d_ghost_culvert_struct_idx,
    const double*  __restrict__ d_structure_flows,
    int32_t                     n_culvert_faces,
    // ── Per-face attribute arrays ──
    const double*  __restrict__ face_rim_elev,
    const double*  __restrict__ face_node_surface_area,
    const int32_t* __restrict__ face_ghost_idx,
    const double*  __restrict__ face_invert,
    const double*  __restrict__ face_nx,
    const double*  __restrict__ face_ny,
    const double*  __restrict__ face_width,
    const double*  __restrict__ face_area,
    const double*  __restrict__ face_k_in,
    const double*  __restrict__ face_k_out,
    const double*  __restrict__ face_depth_safety,
    // ── HEC-22 inlet capture arrays (SURFACE_2D_INLET class 4) ──
    const int32_t* __restrict__ d_face_inlet_type,
    const double*  __restrict__ d_face_inlet_grate_len,
    const double*  __restrict__ d_face_inlet_grate_wid,
    const double*  __restrict__ d_face_inlet_grate_open,
    const double*  __restrict__ d_face_inlet_curb_len,
    const double*  __restrict__ d_face_inlet_curb_ht,
    const double*  __restrict__ d_face_inlet_slot_len,
    const double*  __restrict__ d_face_inlet_slot_wid,
    const double*  __restrict__ d_face_inlet_crest,
    const double*  __restrict__ d_face_inlet_cd,
    const double*  __restrict__ d_face_inlet_qmax,
    int32_t                     n_inlet_capture_faces,
    double                      current_time,
    double                      h_min)
{
    int32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_faces) return;

    const int32_t cls = face_class[k];
    const int32_t solve_mode = (face_solve_mode) ? face_solve_mode[k] : 0;
    const int32_t L = face_owner_L[k];
    const int32_t R = face_owner_R[k];
    const int32_t gi = (face_ghost_idx) ? face_ghost_idx[k] : -1;

    // Default-zero face_F_h / face_F_Q for faces THIS pass handles, so the
    // class branch can write the actual values.  We must not zero faces the
    // OTHER pass handles — pass-1 writes face_F_Q = fh*uL_p for class 3 in
    // direct_inject mode, and pass-2 must not clobber that.
    // Pass 1 handles class 3/4/5/6; pass 2 handles class 0/1/2/7/8.
    const bool this_pass_handles =
        (pass == 1 && (cls == 3 || cls == 4 || cls == 5 || cls == 6))
     || (pass == 2 && (cls == 0 || cls == 1 || cls == 2 || cls == 7 || cls == 8));
    if (this_pass_handles) {
        face_F_h[k] = 0.0;
        face_F_Q[k] = 0.0;
    }

    // ════════════════════════════════════════════════════════════════════════
    // RIEMANN MODE (solve_mode == 0)
    // ════════════════════════════════════════════════════════════════════════

    // ── INTERIOR face (class 0) — HLLC between pipe cells ────────────────────
    if (pass == 2 && cls == 0) {
        if (L < 0 || L >= n_cells_all || R < 0 || R >= n_cells_all) return;
        if (L == R) return;

        const double A_L = fmax(cell_A[L], 0.0);
        const double A_R = fmax(cell_A[R], 0.0);
        const double Q_L = cell_Q[L];
        const double Q_R = cell_Q[R];
        const double A_L_safe = fmax(A_L, 1.0e-12);
        const double A_R_safe = fmax(A_R, 1.0e-12);

        // MUSCL-MC reconstruction at the face (null d_slope_A/Q = no recon)
        double A_L_face = A_L, A_R_face = A_R, Q_L_face = Q_L, Q_R_face = Q_R;
        if (d_slope_A && d_slope_Q) {
            const double dx_L = fmax(cell_length[L], 1.0e-12);
            const double dx_R = fmax(cell_length[R], 1.0e-12);
            A_L_face = fmax(A_L + 0.5 * d_slope_A[L] * dx_L, 0.0);
            A_R_face = fmax(A_R - 0.5 * d_slope_A[R] * dx_R, 0.0);
            Q_L_face = Q_L + 0.5 * d_slope_Q[L] * dx_L;
            Q_R_face = Q_R - 0.5 * d_slope_Q[R] * dx_R;
        }
        const double A_L_face_safe = fmax(A_L_face, 1.0e-12);
        const double A_R_face_safe = fmax(A_R_face, 1.0e-12);

        double P_L, T_L, I1_L;
        pipe1d_lookup_geometry(A_L_safe, cell_area_full[L], cell_perim[L],
            cell_tables + static_cast<int64_t>(L) * 3 * table_N, table_N, P_L, T_L, I1_L);
        double P_R, T_R, I1_R;
        pipe1d_lookup_geometry(A_R_safe, cell_area_full[R], cell_perim[R],
            cell_tables + static_cast<int64_t>(R) * 3 * table_N, table_N, P_R, T_R, I1_R);

        const double T_L_safe = fmax(T_L, 1.0e-10);
        const double T_R_safe = fmax(T_R, 1.0e-10);
        const double hd_L = A_L_face_safe / T_L_safe;
        const double hd_R = A_R_face_safe / T_R_safe;
        const double v_L = Q_L_face / A_L_face_safe;
        const double v_R = Q_R_face / A_R_face_safe;

        // Normal direction: L→R = +1 (interior face convention matches mesh build)
        const double nx = 1.0;
        const double unL = v_L * nx;
        const double unR = v_R * nx;

        // Wave speeds
        const double c_L = sqrt(g * fmax(hd_L, h_min));
        const double c_R = sqrt(g * fmax(hd_R, h_min));

        const double sqrt_AL = sqrt(A_L_face_safe);
        const double sqrt_AR = sqrt(A_R_face_safe);
        const double denom = sqrt_AL + sqrt_AR;
        double u_roe = 0.0, c_roe = 0.0;
        if (denom > 1.0e-15) {
            u_roe = (sqrt_AL * unL + sqrt_AR * unR) / denom;
            c_roe = sqrt(0.5 * g * (hd_L + hd_R));
        }
        const double SL = fmin(unL - c_L, u_roe - c_roe);
        const double SR = fmax(unR + c_R, u_roe + c_roe);

        // Add slot surcharge contribution to I₁ when A exceeds A_full
        auto i1_with_slot = [&](double A, double A_full, double yFull, double wMax, double I1) -> double {
            if (A <= A_full) return I1;
            const double sw = fmax(0.01 * wMax, 1.0e-12);
            const double h_ex = (A - A_full) / sw;
            return I1 + A_full * h_ex + 0.5 * sw * h_ex * h_ex;
        };

        const double P_flux_L = g * i1_with_slot(A_L_face, cell_area_full[L], cell_height[L], cell_width[L], I1_L);
        const double P_flux_R = g * i1_with_slot(A_R_face, cell_area_full[R], cell_height[R], cell_width[R], I1_R);
        const double fA_L = A_L_face * unL;
        const double fQ_L = A_L_face * unL * v_L + P_flux_L * nx;
        const double fA_R = A_R_face * unR;
        const double fQ_R = A_R_face * unR * v_R + P_flux_R * nx;

        // F_max CFL limiter (volume-integrity cap, applied before HLLC branch).
        const double L_avg = 0.5 * (fmax(cell_length[L], 1.0e-12)
                                  + fmax(cell_length[R], 1.0e-12));
        const double A_min = fmin(A_L_face, A_R_face);
        const double F_max_val = (dt > 0.0) ? (A_min * L_avg / dt) : 1.0e10;

        // HLLC branch — uses face-reconstructed values throughout
        double F, F_Q;
        if (SL >= 0.0) {
            F = fA_L; F_Q = fQ_L;
        } else if (SR <= 0.0) {
            F = fA_R; F_Q = fQ_R;
        } else {
            const double numS = A_R_face * unR * (SR - unR)
                              - A_L_face * unL * (SL - unL)
                              + P_flux_L - P_flux_R;
            const double denS = A_R_face * (SR - unR) - A_L_face * (SL - unL);
            const double s_star = (fabs(denS) > 1.0e-15) ? (numS / denS) : 0.0;
            if (s_star >= 0.0) {
                const double coeff = A_L_face * (SL - unL) / (SL - s_star);
                const double hu_star = coeff * (v_L + (s_star - unL) * nx);
                F = fA_L + SL * (coeff - A_L_face);
                F_Q = fQ_L + SL * (hu_star - A_L_face * v_L);
            } else {
                const double coeff = A_R_face * (SR - unR) / (SR - s_star);
                const double hu_star = coeff * (v_R + (s_star - unR) * nx);
                F = fA_R + SR * (coeff - A_R_face);
                F_Q = fQ_R + SR * (hu_star - A_R_face * v_R);
            }
        }

        // Entrance/exit loss correction for storage→pipe faces
        // (pipe→pipe interior faces have k=0, so this is a no-op for them)
        double k_loss = 0.0;
        if (face_k_in)  k_loss = face_k_in[k];
        if (k_loss <= 0.0 && face_k_out) k_loss = face_k_out[k];
        if (k_loss > 0.0) {
            const double scale = 1.0 / sqrt(1.0 + k_loss);
            F  *= scale;
            F_Q *= scale;
        }

        // Apply volume-integrity cap
        if (F > F_max_val) F = F_max_val;
        if (F < -F_max_val) F = -F_max_val;
        if (F_Q > F_max_val) F_Q = F_max_val;
        if (F_Q < -F_max_val) F_Q = -F_max_val;

        face_F_h[k] = F;
        face_F_Q[k] = F_Q;
    }

        // ── OUTFALL_BC face (class 1) — Riemann mode, ghost from outfall SoA ──
    if (pass == 2 && cls == 1) {
        if (L < 0 || L >= n_cells_all) return;
        if (gi < 0 || gi >= n_outfall_faces) return;
        if (d_ghost_outfall_mode == nullptr) return;

        // Left cell state (pipe cell at outfall)
        const double A_L = fmax(cell_A[L], 0.0);
        const double Q_L = cell_Q[L];
        const double A_L_safe = fmax(A_L, 1.0e-12);

        double P_L, T_L, I1_L;
        pipe1d_lookup_geometry(A_L_safe, cell_area_full[L], cell_perim[L],
            cell_tables + static_cast<int64_t>(L) * 3 * table_N, table_N, P_L, T_L, I1_L);
        const double T_L_safe = fmax(T_L, 1.0e-10);

        // ── Build ghost (right) state from outfall mode ──
        const int32_t mode = d_ghost_outfall_mode[gi];
        const double inv_L = cell_invert[L];
        const double yFull_L = cell_height[L];
        const double A_full_L = cell_area_full[L];
        const double wMax_L = cell_width[L];

        double A_R, Q_R, y_R;
        // dir: sign convention for dry-node check (matches old boundary kernel)
        const double dir_k = (face_dir) ? face_dir[k] : 1.0;

        switch (mode) {
            case 0: // FREE — dry ghost (atmospheric pressure)
            default:
                A_R = 0.0;
                Q_R = 0.0;
                y_R = 0.0;
                break;

            case 1: { // NORMAL_DEPTH — compute from cell S0, n, width
                const double S0 = (d_ghost_outfall_link_S0) ? d_ghost_outfall_link_S0[gi] : 0.0;
                if (S0 > 1.0e-9 && A_L > 1.0e-12) {
                    y_R = pipe1d_circular_normal_depth_bisect(
                        fmax(fabs(Q_L), 1.0e-12), cell_n[L], S0, wMax_L, h_min);
                } else {
                    y_R = 0.0;
                }
                A_R = xsect_getAofY(cell_shape_type[L], (double[]){wMax_L, yFull_L, 0.0}, y_R);
                Q_R = A_R * (A_L_safe > 0.0 ? Q_L / A_L_safe : 0.0);
                break;
            }

            case 2: { // FIXED_WSE — stored value is absolute water-surface elevation
                const double ghost_wse = (d_ghost_outfall_fixed_wse) ? d_ghost_outfall_fixed_wse[gi] : 0.0;
                y_R = fmax(0.0, ghost_wse - inv_L);
                A_R = xsect_getAofY(cell_shape_type[L], (double[]){wMax_L, yFull_L, 0.0}, y_R);
                Q_R = A_R * (A_L_safe > 0.0 ? Q_L / A_L_safe : 0.0);
                break;
            }

            case 3: { // RATING_CURVE — interpolate Q→WSE from rating table
                const int32_t npts = (d_ghost_outfall_rating_n) ? d_ghost_outfall_rating_n[gi] : 0;
                if (npts > 0 && d_ghost_outfall_rating) {
                    const double* tw = d_ghost_outfall_rating + (size_t)gi * MAX_RATING_POINTS * 2;
                    const double* tq = tw + MAX_RATING_POINTS;
                    const double wse = pipe1d_rating_curve_invert(fabs(Q_L), tw, tq, npts);
                    y_R = fmax(0.0, wse - inv_L);
                } else {
                    y_R = 0.0;
                }
                A_R = xsect_getAofY(cell_shape_type[L], (double[]){wMax_L, yFull_L, 0.0}, y_R);
                Q_R = A_R * (A_L_safe > 0.0 ? Q_L / A_L_safe : 0.0);
                break;
            }

            case 4: { // TABULAR — interpolate time→WSE from tabular table
                const int32_t npts = (d_ghost_outfall_tabular_n) ? d_ghost_outfall_tabular_n[gi] : 0;
                if (npts > 0 && d_ghost_outfall_tabular) {
                    const double* tt = d_ghost_outfall_tabular + (size_t)gi * MAX_TABULAR_POINTS * 2;
                    const double* tw = tt + MAX_TABULAR_POINTS;
                    double wse = (npts == 1 || current_time <= tt[0]) ? tw[0] : tw[npts-1];
                    if (npts > 1 && current_time > tt[0] && current_time < tt[npts-1]) {
                        int lo = 0, hi = npts - 1;
                        while (hi - lo > 1) { int m = (lo + hi) >> 1; if (tt[m] <= current_time) lo = m; else hi = m; }
                        const double t = (current_time - tt[lo]) / fmax(1.0e-12, tt[hi] - tt[lo]);
                        wse = tw[lo] + t * (tw[hi] - tw[lo]);
                    }
                    y_R = fmax(0.0, wse - inv_L);
                } else {
                    y_R = 0.0;
                }
                A_R = xsect_getAofY(cell_shape_type[L], (double[]){wMax_L, yFull_L, 0.0}, y_R);
                Q_R = A_R * (A_L_safe > 0.0 ? Q_L / A_L_safe : 0.0);
                break;
            }
        }

        // Backward-compat dry-node / dry-cell protection (matches old boundary
        // kernel checks).  F > 0 means cell→ghost (outflow from cell) in the
        // unified kernel's L=cell,R=ghost convention, regardless of face_dir.
        const double A_floor_L = (h_min > 0.0)
            ? A_full_L * fmin(1.0, h_min / fmax(yFull_L, 1.0e-12))
            : 0.0;
        if (y_R <= 1.0e-12 || A_L <= A_floor_L + 1.0e-15) {
            const double A_R_safe_tmp = fmax(A_R, 1.0e-12);
            const double hd_L_tmp = A_L_safe / T_L_safe;
            const double hd_R_tmp = A_R_safe_tmp / T_L_safe;
            const double hd_open_tmp = A_full_L / T_L_safe;
            const double hd_eff_tmp = fmin(0.5 * (hd_L_tmp + hd_R_tmp), hd_open_tmp);
            const double c_wave_tmp = sqrt(g * fmax(hd_eff_tmp, 1.0e-12));
            const double F_tmp = 0.5 * (Q_L + Q_R - c_wave_tmp * (A_R - A_L));
            // F_tmp > 0 = outflow from cell (cell→ghost). Protect against
            // drawing water from a dry ghost (y_R ≈ 0) INTO the cell.
            if (y_R <= 1.0e-12 && F_tmp < 0.0) {
                face_F_h[k] = 0.0;
                face_F_Q[k] = 0.0;
                return;
            }
            // Protect against draining the pipe below A_floor (outflow when
            // cell is at minimum area).
            if (A_L <= A_floor_L + 1.0e-15 && F_tmp > 0.0) {
                face_F_h[k] = 0.0;
                face_F_Q[k] = 0.0;
                return;
            }
            // NOTE: no unconditional dry-ghost block here.  With HLLC the
            // Riemann solver correctly handles outflow into a dry ghost (FREE
            // outfall).  The inflow-direction check above (F_tmp < 0 when
            // y_R ≈ 0) is sufficient to prevent drawing water from nowhere.
        }

        const double A_R_safe = fmax(A_R, 1.0e-12);
        const double v_L = Q_L / A_L_safe;
        const double v_R = (A_R_safe > 0.0) ? Q_R / A_R_safe : 0.0;

        // Right-state geometry (same cross-section as pipe cell)
        double P_R_tmp, T_R_tmp, I1_R;
        pipe1d_lookup_geometry(A_R_safe, A_full_L, cell_perim[L],
            cell_tables + static_cast<int64_t>(L) * 3 * table_N, table_N,
            P_R_tmp, T_R_tmp, I1_R);
        const double T_R_safe = fmax(T_R_tmp, 1.0e-10);

        // ── HLLC Riemann solver between pipe cell (L) and ghost (R) ──
        const double nx = 1.0;
        const double unL = v_L * nx;
        const double unR = v_R * nx;

        const double hd_L = A_L_safe / T_L_safe;
        const double hd_R = A_R_safe / T_R_safe;
        const double c_L = sqrt(g * fmax(hd_L, h_min));
        const double c_R = sqrt(g * fmax(hd_R, h_min));

        const double sqrt_AL = sqrt(A_L_safe);
        const double sqrt_AR = sqrt(A_R_safe);
        const double denom = sqrt_AL + sqrt_AR;
        double u_roe = 0.0, c_roe = 0.0;
        if (denom > 1.0e-15) {
            u_roe = (sqrt_AL * unL + sqrt_AR * unR) / denom;
            c_roe = sqrt(0.5 * g * (hd_L + hd_R));
        }
        const double SL = fmin(unL - c_L, u_roe - c_roe);
        const double SR = fmax(unR + c_R, u_roe + c_roe);

        const double P_flux_L = g * I1_L;
        const double P_flux_R = g * I1_R;
        const double fA_L = A_L * unL;
        const double fQ_L = A_L * unL * v_L + P_flux_L * nx;
        const double fA_R = A_R_safe * unR;
        const double fQ_R = A_R_safe * unR * v_R + P_flux_R * nx;

        const double L_c = fmax(cell_length[L], 1.0e-12);
        const double F_max_val = (dt > 0.0) ? (A_L_safe * L_c / dt) : 1.0e10;

        double F, F_Q;
        if (SL >= 0.0) {
            F = fA_L; F_Q = fQ_L;
        } else if (SR <= 0.0) {
            F = fA_R; F_Q = fQ_R;
        } else {
            const double numS = A_R_safe * unR * (SR - unR)
                              - A_L * unL * (SL - unL)
                              + P_flux_L - P_flux_R;
            const double denS = A_R_safe * (SR - unR) - A_L * (SL - unL);
            const double s_star = (fabs(denS) > 1.0e-15) ? (numS / denS) : 0.0;
            if (s_star >= 0.0) {
                const double coeff = A_L * (SL - unL) / (SL - s_star);
                const double hu_star = coeff * (v_L + (s_star - unL) * nx);
                F = fA_L + SL * (coeff - A_L);
                F_Q = fQ_L + SL * (hu_star - A_L * v_L);
            } else {
                const double coeff = A_R_safe * (SR - unR) / (SR - s_star);
                const double hu_star = coeff * (v_R + (s_star - unR) * nx);
                F = fA_R + SR * (coeff - A_R_safe);
                F_Q = fQ_R + SR * (hu_star - A_R_safe * v_R);
            }
        }

        // Apply CFL cap (preserves conservation — same F to both sides)
        if (F > F_max_val) F = F_max_val;
        if (F < -F_max_val) F = -F_max_val;
        if (F_Q > F_max_val) F_Q = F_max_val;
        if (F_Q < -F_max_val) F_Q = -F_max_val;

        // Face_dir adjustment: see comment above.
        face_F_h[k] = F;
        face_F_Q[k] = dir_k * F_Q;

        // Pipe-side direct update (bypasses fold+godunov when caller is
        // swe2d_gpu_step).  Sign: F > 0 means pipe → ghost (outflow), so pipe
        // loses mass.  n_cells_2d>0 gate prevents double-counting on the
        // pipe1D godunov path where face_F_h[k] is consumed by fold+godunov.
        if (d_A && cell_length && cell_length[L] > 0.0 && n_cells_2d > 0) {
            atomicAdd(&d_A[L], -F * dt / fmax(cell_length[L], 1.0e-3));
        }
    }

    // ── WALL_BC face (class 7) — HLLC with reflective ghost ─────────────────
    // The reflective ghost gives A_R = A_L, Q_R = -Q_L, so the HLLC mass flux
    // F = 0 and the momentum flux F_Q = g·I₁ (hydrostatic pressure at wall).
    // This F_Q is NOT a bug — it is the wall reaction force that balances the
    // pressure from the interior face.  Without it the end cell receives an
    // unbalanced pressure force and develops unphysical flow.
    if (pass == 2 && cls == 7) {
        if (L < 0 || L >= n_cells_all) return;
        const double A_L = fmax(cell_A[L], 0.0);
        const double Q_L = cell_Q[L];
        const double A_L_safe = fmax(A_L, 1.0e-12);
        double P_L, T_L, I1_L;
        pipe1d_lookup_geometry(A_L_safe, cell_area_full[L], cell_perim[L],
            cell_tables + static_cast<int64_t>(L) * 3 * table_N, table_N, P_L, T_L, I1_L);
        auto i1s = [&](double A, double Af, double yF, double wM, double i1o) -> double {
            if (A <= Af) return i1o;
            const double sw = fmax(0.01 * wM, 1.0e-12);
            const double hx = (A - Af) / sw;
            return i1o + Af * hx + 0.5 * sw * hx * hx;
        };
        const double I1_L_eff = i1s(A_L, cell_area_full[L], cell_height[L], cell_width[L], I1_L);
        const double nx = 1.0;
        const double v_L = Q_L / A_L_safe;
        const double v_R = -v_L;
        const double unL = v_L * nx;
        const double unR = v_R * nx;
        const double T_L_safe = fmax(T_L, 1.0e-10);
        const double hd_L = A_L_safe / T_L_safe;
        const double c_L = sqrt(g * fmax(hd_L, h_min));
        const double sqrt_AL = sqrt(A_L_safe);
        const double denom = sqrt_AL + sqrt_AL;
        double u_roe = 0.0, c_roe = 0.0;
        if (denom > 1.0e-15) {
            u_roe = (sqrt_AL * unL + sqrt_AL * unR) / denom;
            c_roe = c_L;
        }
        const double SL = fmin(unL - c_L, u_roe - c_roe);
        const double SR = fmax(unR + c_L, u_roe + c_roe);
        const double PL = g * I1_L_eff;
        const double fA_L = A_L * unL;
        const double fQ_L = A_L * unL * v_L + PL * nx;
        const double A_R = A_L;
        const double A_R_safe = A_L_safe;
        const double fA_R = A_R * unR;
        const double fQ_R = A_R * unR * v_R + PL * nx;
        const double L_c = fmax(cell_length[L], 1.0e-12);
        const double F_max_val = (dt > 0.0) ? (A_L_safe * L_c / dt) : 1.0e10;
        double F, F_Q;
        if (SL >= 0.0) { F = fA_L; F_Q = fQ_L; }
        else if (SR <= 0.0) { F = fA_R; F_Q = fQ_R; }
        else {
            const double numS = A_R_safe * unR * (SR - unR) - A_L * unL * (SL - unL) + PL - PL;
            const double denS = A_R_safe * (SR - unR) - A_L * (SL - unL);
            const double s_star = (fabs(denS) > 1.0e-15) ? (numS / denS) : 0.0;
            if (s_star >= 0.0) {
                const double coeff = A_L * (SL - unL) / (SL - s_star);
                const double hu_star = coeff * (v_L + (s_star - unL) * nx);
                F = fA_L + SL * (coeff - A_L);
                F_Q = fQ_L + SL * (hu_star - A_L * v_L);
            } else {
                const double coeff = A_R_safe * (SR - unR) / (SR - s_star);
                const double hu_star = coeff * (v_R + (s_star - unR) * nx);
                F = fA_R + SR * (coeff - A_R_safe);
                F_Q = fQ_R + SR * (hu_star - A_R_safe * v_R);
            }
        }
        if (F > F_max_val) F = F_max_val; if (F < -F_max_val) F = -F_max_val;
        if (F_Q > F_max_val) F_Q = F_max_val; if (F_Q < -F_max_val) F_Q = -F_max_val;
        face_F_h[k] = F;
        // Multiply by face_dir so the pressure sign matches the wall orientation.
        // Left wall (face_dir=-1): F_Q_wall = -P balances interior face +P on cell 0.
        // Right wall (face_dir=+1): F_Q_wall = +P balances interior face -P on cell 9.
        const double dir_k_wall = (face_dir) ? face_dir[k] : 1.0;
        face_F_Q[k] = dir_k_wall * F_Q;
    }

    // ── INLET_BC face (class 2) — Riemann mode, ghost with prescribed Q ──
    if (pass == 2 && cls == 2) {
        if (L < 0 || L >= n_cells_all) return;
        if (gi < 0 || gi >= n_inlet_bc_faces) return;
        if (d_ghost_inlet_Q == nullptr) return;

        const double A_L = fmax(cell_A[L], 0.0);
        const double Q_L = cell_Q[L];
        const double A_L_safe = fmax(A_L, 1.0e-12);

        double P_L, T_L, I1_L;
        pipe1d_lookup_geometry(A_L_safe, cell_area_full[L], cell_perim[L],
            cell_tables + static_cast<int64_t>(L) * 3 * table_N, table_N, P_L, T_L, I1_L);
        const double T_L_safe = fmax(T_L, 1.0e-10);
        const double A_full_L = cell_area_full[L];

        // Prescribed Q (positive = flow INTO the system = R→L in our convention)
        // The ghost state has same A and y as L, but Q = prescribed value
        const double Q_prescribed = d_ghost_inlet_Q[gi];

        // Ghost right state: same geometry as L, Q = prescribed
        // Positive Q_prescribed means water enters L from ghost (R→L)
        // In HLLE convention (L→R positive): F = 0.5*(Q_L + Q_R - c*(A_R - A_L))
        // We want the ghost to deliver Q_prescribed when A_R ~ A_L
        // Set A_R = A_L, Q_R = -Q_prescribed (so that F gets the contribution)
        const double A_R = A_L;
        const double A_R_safe = A_L_safe;
        const double Q_R = -Q_prescribed;

        const double hd_L = A_L_safe / T_L_safe;
        const double hd_open = A_full_L / T_L_safe;
        const double hd_eff = fmin(hd_L, hd_open);
        const double c_wave = sqrt(g * fmax(hd_eff, 1.0e-12));

        // INLET_BC: inject the FULL prescribed flow (NOT the HLLE average).
        // F = (Q_L + Q_R)/2 - c_wave*(A_R-A_L)/2  → HLLE average gives -Q_prescribed/2
        // which under-delivers by 2×.  Instead, just inject Q_prescribed directly.
        // Negative F = flow into cell (R→L in L→R convention).
        double F = -Q_prescribed;

        // F_max clamp: use A_full (not A_L_safe) so a dry cell can receive flow.
        const double L_c = fmax(cell_length[L], 1.0e-12);
        const double F_max = (dt > 0.0)
            ? (fmax(A_L_safe, A_full_L) * L_c / dt)
            : fmax(fabs(F), 1.0);
        if (F > F_max) F = F_max;
        if (F < -F_max) F = -F_max;

        // Momentum flux: prescribed flow carries zero momentum (no directional head)
        face_F_h[k] = F;
        face_F_Q[k] = 0.0;
    }

    // ── STORAGE_PIPE face (class 8) — weir/orifice between storage cell and pipe cell ──
    // Storage cells (manhole/inlet, class 1/2) connect to pipe cells via this face.
    // Standard weir/orifice hydraulics replace the HLLC Riemann solver because
    // the cross-sectional geometry of a storage cell (rectangular, w×h) is
    // fundamentally different from a pipe cell (circular/elliptical), and the
    // HLLC solver's A-based primitive variables and CFL limiter cannot produce
    // correct fluxes across such dissimilar sections.
    //
    // Weir (free-surface, storage WSE ≤ storage rim):
    //   Q = Cw · W · H^1.5    (Cw stored in face_k_in, W = face_width)
    // Orifice (submerged, storage WSE > storage rim):
    //   Q = Cd · A · √(2g·H)   (Cd stored in face_k_out, A = face_area)
    // H = head above crest (pipe invert at the interface face).
    if (pass == 2 && cls == 8) {
        if (L < 0 || L >= n_cells_all) return;
        if (R < 0 || R >= n_cells_all) return;
        if (!face_invert) return;
        // L = storage cell, R = pipe cell
        const double y_L = cell_y[L];
        const double y_R = cell_y[R];
        const double A_L = fmax(cell_A[L], 0.0);
        const double A_R = fmax(cell_A[R], 0.0);
        const double crest = face_invert[k];
        // Fixed weir coefficient (SI, broad-crested).  Users configure
        // entrance/exit loss through face_k_in/face_k_out, not the weir Cw.
        // Broad-crested weir coefficient Cw from g: Cw = 1.84 * sqrt(g/9.81).
        // This gives Cw ≈ 1.84 in SI (g=9.81) and Cw ≈ 3.33 in USC (g=32.2).
        const double Cw = 1.84 * sqrt(g / 9.80665);
        const double Cd = 0.65;
        const double face_w = (face_width) ? fmax(face_width[k], 1.0e-12) : 1.0;
        const double face_a = (face_area) ? fmax(face_area[k], 1.0e-12) : 1.0;
        // Bidirectional weir/orifice between storage cell and pipe cell.
        // Compute both weir and orifice discharges and take the minimum —
        // the more restrictive equation governs.  This is simpler and more
        // correct than regime-switching on water-surface thresholds because
        // the transition between free-surface and pressurised is smooth.
        //   Weir:  Q = Cw · W · H_weir^1.5   (H_weir = max head above crest)
        //   Orifice: Q = Cd · A · sqrt(2g·|y_L−y_R|)
        const double crest_eff = fmin(cell_invert[L], cell_invert[R]);
        const double h_L_eff = y_L - crest_eff;
        const double h_R_eff = y_R - crest_eff;
        double Q_weir = 0.0, Q_orifice = 0.0;
        if (h_L_eff > 1.0e-12 || h_R_eff > 1.0e-12) {
            const double H_weir = fmax(h_L_eff, h_R_eff);
            Q_weir = Cw * face_w * pow(fmax(H_weir, 0.0), 1.5);
            if (y_R > y_L) Q_weir = -Q_weir;
        }
        if (h_L_eff > 1.0e-12 && h_R_eff > 1.0e-12) {
            const double head = fabs(y_L - y_R);
            Q_orifice = Cd * face_a * sqrt(2.0 * g * fmax(head, 1.0e-12));
            if (y_R > y_L) Q_orifice = -Q_orifice;
        }
        double Q_dir = (fabs(Q_weir) < fabs(Q_orifice)) ? Q_weir : Q_orifice;
        // Volume-integrity cap: don't extract more from a cell than it holds.
        if (Q_dir > 0.0 && cell_length[L] > 0.0) {
            const double max_Q = A_L * cell_length[L] / fmax(dt, 1.0e-12);
            Q_dir = fmin(Q_dir, max_Q);
        } else if (Q_dir < 0.0 && cell_length[R] > 0.0) {
            const double max_Q = A_R * cell_length[R] / fmax(dt, 1.0e-12);
            Q_dir = fmax(Q_dir, -max_Q);
        }
        // Entrance/exit loss scaling (face_k_in = Ke, face_k_out = Kx).
        if (fabs(Q_dir) > 0.0) {
            double k_loss = (face_k_in && face_k_in[k] > 0.0) ? face_k_in[k] : 0.0;
            if (k_loss <= 0.0 && face_k_out) k_loss = face_k_out[k];
            if (k_loss > 0.0) Q_dir *= 1.0 / sqrt(1.0 + k_loss);
        }
        face_F_h[k] = Q_dir;  // positive = storage→pipe, negative = pipe→storage
        face_F_Q[k] = 0.0;
    }

    // ── SURFACE_2D_PIPE_END face (class 3) — HLLC between pipe cell and 2D cell ──
    if (pass == 1 && cls == 3) {
        if (L < 0 || L >= n_cells_all) return;
        if (R < 0 || R >= n_cells_2d) return;
        if (!cell_h_2d || !cell_hu_2d || !cell_hv_2d || !cell_zb_2d) return;
        if (!d_ext_struct_flux_h || !d_ext_struct_flux_hu || !d_ext_struct_flux_hv) return;
        if (!face_invert || !face_nx || !face_ny || !face_width || !face_area) return;

        // ── 1D pipe cell state ──
        const double A_p = fmax(cell_A[L], 0.0);
        const double Q_p = cell_Q[L];
        const double h_p = (cell_y && cell_invert) ? fmax(cell_y[L] - cell_invert[L], 0.0) : 0.0;
        const double inv_p = face_invert[k];
        // Face arrays may be zero-initialised; fall back to cell geometry.
        const double w_p = (face_width && face_width[k] > 0.0)
                         ? face_width[k]
                         : fmax(cell_width[L], 0.01);
        const double A_full_p = (face_area && face_area[k] > 0.0)
                              ? face_area[k]
                              : fmax(cell_area_full[L], 0.01);
        // face_nx/face_ny are zero-initialised for SURFACE_2D_PIPE_END faces
        // (they are never set at mesh build).  Use face_dir (±1) as the
        // scalar outward-normal direction: +1 = out the downstream pipe end,
        // -1 = out the upstream pipe end.  This gives the correct sign for
        // all projections (unL, unR, fhuL, fhvL, ...).
        const double dir_k = (face_dir && face_dir[k] != 0.0) ? face_dir[k] : 1.0;
        const double nx = dir_k;
        const double ny = 0.0;
        if (w_p <= 0.0 || A_full_p <= 0.0) return;

        const double WSE_p = inv_p + fmax(h_p, 0.0);

        // ── Hydrostatic reconstruction (Audusse et al. 2004) ──
        // The HLLC Riemann solver uses hydraulic depth to compute pressure
        // gradients, but the pipe invert and 2D cell bed may differ.
        // Without correction, a pipe end whose invert is above the 2D bed
        // sees the 2D depth as a head equal to its own — ignoring the
        // elevation offset (e.g. inv_p=920, z2=910, both h=2m → HLLC thinks
        // heads match, but WSE_p=922 >> WSE_2d=912).
        // Fix: reconstruct depths at the interface (z_max = max bed) and
        // add the hydrostatic pressure correction to the momentum flux.
        const double z2 = cell_zb_2d[R];
        const double WSE_2d = z2 + cell_h_2d[R];
        const double z_max = fmax(inv_p, z2);
        const double hL_star = fmax(0.0, WSE_p - z_max);   // pipe water above max bed
        const double hR_star = fmax(0.0, WSE_2d - z_max);  // 2D water above max bed

        // ── I₁ lookup for pipe pressure term ──
        double P_p_tmp, T_p_tmp, I1_L;
        pipe1d_lookup_geometry(fmax(A_p, 0.0), A_full_p, fmax(cell_perim[L], 0.01),
            cell_tables + static_cast<int64_t>(L) * 3 * table_N, table_N,
            P_p_tmp, T_p_tmp, I1_L);

        // ── 2D cell state ──
        const double h2 = cell_h_2d[R];
        const double hu2 = cell_hu_2d[R];
        const double hv2 = cell_hv_2d[R];
        const double uR = (h2 > h_min) ? hu2 / h2 : 0.0;
        const double vR = (h2 > h_min) ? hv2 / h2 : 0.0;

        // ── Hydrostatic reconstruction: pipe primitive at interface ──
        // Reconstructed area and pressure: rectangular approximation (w_p × h),
        // consistent with the existing hydraulic-depth approximation.
        const double hL_safe = fmax(hL_star, h_min);
        const double A_p_star = w_p * hL_safe;        // rect area at interface depth
        const double PL_s = 0.5 * g * hL_safe * w_p * hL_safe;  // I1 = ½·w·h²
        const double uL_p = (A_p > 1.0e-12) ? Q_p / A_p : 0.0;
        const double cL_p = sqrt(g * hL_safe);
        const double unL = uL_p * nx;

        // ── 2D primitive (reconstructed at interface) ──
        const double hR_safe = fmax(hR_star, h_min);
        const double unR = uR * nx + vR * ny;
        const double cR = sqrt(g * hR_safe);

        // ── Wave speed estimates (Roe averages) ──
        const double sqrt_hL = sqrt(hL_safe);
        const double sqrt_hR = sqrt(hR_safe);
        const double denom = sqrt_hL + sqrt_hR;
        const double u_roe = (denom > 0.0) ? (sqrt_hL * unL + sqrt_hR * unR) / denom : 0.0;
        const double c_roe = (denom > 0.0) ? sqrt(0.5 * g * (hL_safe + hR_safe)) : 0.0;
        const double SL = fmin(unL - cL_p, u_roe - c_roe);
        const double SR = fmax(unR + cR, u_roe + c_roe);

        // ── Physical flux components (reconstructed) ──
        const double fhL  = A_p_star * unL;
        const double fhuL = A_p_star * unL * uL_p * nx + PL_s * nx;
        const double fhvL = A_p_star * unL * uL_p * ny + PL_s * ny;

        const double fhR  = hR_safe * unR;
        const double fhuR = hR_safe * unR * uR + 0.5 * g * hR_safe * hR_safe * nx;
        const double fhvR = hR_safe * unR * vR + 0.5 * g * hR_safe * hR_safe * ny;

        double fh = 0.0, fhu = 0.0, fhv = 0.0;
        double s_star = 0.0;

        if (SL >= 0.0) {
            fh = fhL; fhu = fhuL; fhv = fhvL;
        } else if (SR <= 0.0) {
            fh = fhR; fhu = fhuR; fhv = fhvR;
        } else {
            // HLLC star-state contact speed
            const double numS = hR_safe * unR * (SR - unR) - A_p_star * unL * (SL - unL)
                              + 0.5 * g * (A_p_star * hL_safe - hR_safe * hR_safe);
            const double denS = hR_safe * (SR - unR) - A_p_star * (SL - unL);
            s_star = (fabs(denS) > 1.0e-15) ? (numS / denS) : 0.0;
            if (s_star >= 0.0) {
                const double coeffL = A_p_star * (SL - unL) / (SL - s_star);
                const double hu_star_L = coeffL * (uL_p + (s_star - unL) * nx);
                const double hv_star_L = coeffL * (uL_p * (ny/(fabs(nx)+1.0e-15)) + (s_star - unL) * ny);
                fh  = fhL  + SL * (coeffL - A_p_star);
                fhu = fhuL + SL * (hu_star_L - A_p_star * uL_p);
                fhv = fhvL + SL * (hv_star_L - 0.0);
            } else {
                const double coeffR = hR_safe * (SR - unR) / (SR - s_star);
                const double hu_star_R = coeffR * (uR + (s_star - unR) * nx);
                const double hv_star_R = coeffR * (vR + (s_star - unR) * ny);
                fh  = fhR  + SR * (coeffR - hR_safe);
                fhu = fhuR + SR * (hu_star_R - hR_safe * uR);
                fhv = fhvR + SR * (hv_star_R - hR_safe * vR);
            }
        }

        // ── Hydrostatic pressure correction for bed step ──
        // fhu = F_HLLC(h*, u) + g*(I1_L - I1_L_star) + g/2*(hR_star² - h2²)
        {
            const double PL_orig = g * I1_L;
            const double PR_orig = 0.5 * g * h2 * h2;
            const double PR_star = 0.5 * g * hR_safe * hR_safe;
            const double corr_mom = (PL_orig - PL_s) + (PR_star - PR_orig);
            fhu += corr_mom * nx;
        }

        // ── Entrance/exit loss correction ──
        double k_loss = 0.0;
        if (face_k_in)  k_loss = face_k_in[k];
        if (k_loss <= 0.0 && face_k_out) k_loss = face_k_out[k];
        if (k_loss > 0.0) {
            const double scale = 1.0 / sqrt(1.0 + k_loss);
            fh  *= scale;
            fhu *= scale;
            fhv *= scale;
        }

        // ── CFL depth-safety cap ──
        const double alpha = (face_depth_safety && face_depth_safety[k] > 0.0)
                           ? face_depth_safety[k] : 1.0;
        const double Q_mag = fabs(fh);
        if (Q_mag > 0.0) {
            const double A_2d = fmax(cell_h_2d[R], 0.0);
            const double L_p = fmax(cell_length[L], 1.0e-12);
            const double V_p = A_p * L_p;
            const double V_2d = A_2d * 1.0;
            const double V_donor = (fh > 0.0) ? V_p : V_2d;
            const double Q_cap = alpha * V_donor / fmax(dt, 1.0e-12);
            if (Q_mag > Q_cap && Q_cap > 0.0) {
                const double scale = Q_cap / Q_mag;
                fh  *= scale;
                fhu *= scale;
                fhv *= scale;
                // Don't reassign Q_mag here, we use fh below
            }
            const double h_donor_mag = (fh > 0.0) ? h_p : A_2d;
            const double c_at_donor = sqrt(fmax(g * h_donor_mag, 1.0e-12));
            const double u_safe = fmax(50.0, 20.0 * c_at_donor);
            const double fhu_cap = alpha * V_donor / fmax(dt, 1.0e-12) * u_safe;
            const double fhu_mag = fmax(fabs(fhu), fabs(fhv));
            if (fhu_mag > fhu_cap && fhu_cap > 0.0) {
                const double mom_scale = fhu_cap / fhu_mag;
                fhu *= mom_scale;
                fhv *= mom_scale;
            }
        }

        // ── Write fluxes ──
        // face_F_h[k] is the mass flux L→R.  When called from swe2d_gpu_step
        // (n_cells_2d > 0), no fold+godunov runs after this kernel, so we
        // zero face_F_h and update the pipe side directly below.  When called
        // from swe2d_pipe1d_godunov_step, face_F_h feeds the fold pipeline
        // and we don't double-count.
        const bool direct_inject = (n_cells_2d > 0);
        face_F_h[k] = direct_inject ? 0.0 : fh;
        // Momentum flux: outflow carries the pipe cell's specific momentum.
        // The Godunov update (line 2930) reads flux_mom via the fold kernel
        // (line 2675) and subtracts it from Q.  When direct_inject is true
        // the fold kernel still accumulates face_F_Q — we rely on that.
        face_F_Q[k] = fh * uL_p;

        // 2D side: 2D cell R gains fh mass, fhu/fhv momentum
        atomicAdd(&d_ext_struct_flux_h[R], fh);
        atomicAdd(&d_ext_struct_flux_hu[R], fhu);
        atomicAdd(&d_ext_struct_flux_hv[R], fhv);

        // Pipe-side direct update (bypasses fold+godunov when caller is
        // swe2d_gpu_step).  Sign: fh < 0 means 2D→pipe, so pipe gains mass.
        if (direct_inject && d_A && cell_length && cell_length[L] > 0.0) {
            atomicAdd(&d_A[L], -fh * dt / fmax(cell_length[L], 1.0e-3));
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    // SOURCE-SINK MODE (solve_mode == 1) for classes 4,5
    // ════════════════════════════════════════════════════════════════════════

    // ── SURFACE_2D_INLET face (class 4) — Source-sink, HEC-22 weir/orifice ──
    if (pass == 1 && cls == 4) {
        if (L < 0 || L >= n_cells_all) return;
        if (R < 0 || R >= n_cells_2d) return;
        if (!cell_h_2d || !cell_zb_2d || !d_ext_struct_flux_h) return;

        // Inlet sump cell state
        const double A_L = fmax(cell_A[L], 0.0);
        const double y_L = (cell_y) ? cell_y[L] : cell_invert[L];
        const double WSE_inlet = y_L;

        // 2D cell state
        const double h2 = cell_h_2d[R];
        const double z2 = cell_zb_2d[R];
        const double WSE_2d = z2 + h2;
        const double hu2 = (cell_hu_2d) ? cell_hu_2d[R] : 0.0;
        const double hv2 = (cell_hv_2d) ? cell_hv_2d[R] : 0.0;

        // Head difference: positive when 2D > inlet (capture direction)
        const double dH = fmax(0.0, WSE_2d - WSE_inlet);

        // ════════════════════════════════════════════════════════════════════
        // HEC-22 inlet capture — type-specific weir/orifice equations
        // (HEC-22 §4-3, §4-4: grate, curb, slotted, combo inlets on-grade)
        // ════════════════════════════════════════════════════════════════════
        double Q = 0.0;

        // Look up per-face HEC-22 geometry via ghost index
        const int32_t ii = (face_ghost_idx && gi >= 0 && gi < n_inlet_capture_faces) ? gi : -1;
        if (ii >= 0 && d_face_inlet_type && d_face_inlet_qmax) {
            const int32_t type = d_face_inlet_type[ii];
            const double grate_len = d_face_inlet_grate_len ? fmax(d_face_inlet_grate_len[ii], 0.01) : 0.01;
            const double grate_wid = d_face_inlet_grate_wid ? fmax(d_face_inlet_grate_wid[ii], 0.01) : 0.01;
            const double open_frac = d_face_inlet_grate_open ? fmax(d_face_inlet_grate_open[ii], 0.01) : 1.0;
            const double curb_len  = d_face_inlet_curb_len ? fmax(d_face_inlet_curb_len[ii], 0.01) : 0.01;
            const double curb_ht   = d_face_inlet_curb_ht ? fmax(d_face_inlet_curb_ht[ii], 0.0) : 0.0;
            const double slot_len  = d_face_inlet_slot_len ? fmax(d_face_inlet_slot_len[ii], 0.01) : 0.01;
            const double slot_wid  = d_face_inlet_slot_wid ? fmax(d_face_inlet_slot_wid[ii], 0.01) : 0.01;
            const double crest     = d_face_inlet_crest ? d_face_inlet_crest[ii] : WSE_inlet;
            const double cd        = d_face_inlet_cd ? fmax(d_face_inlet_cd[ii], 0.01) : 0.67;
            const double qmax      = d_face_inlet_qmax[ii];

            // Depth of water above inlet crest (= grate elevation)
            const double d_crest = fmax(0.0, WSE_2d - crest);

            if (type == 0) {
                // ── GRATE inlet ──────────────────────────────────────────
                // Wet perimeter: 2*(L+W) for a rectangular grate
                const double P = 2.0 * (grate_len + grate_wid);
                // Open area: grate area * open fraction
                const double Ag = grate_len * grate_wid * open_frac;
                // Weir: Q_w = C_w * P * d_crest^1.5  (C_w ≈ 3.0 for broad-crested)
                const double Q_w = 3.0 * P * pow(d_crest, 1.5);
                // Orifice: Q_o = C_o * Ag * sqrt(2*g*d_crest)
                const double Q_o = cd * Ag * sqrt(fmax(2.0 * g * d_crest, 1.0e-12));
                // HEC-22 transition: min of weir and orifice
                Q = fmin(Q_w, Q_o);
            } else if (type == 1) {
                // ── CURB inlet ───────────────────────────────────────────
                // Weir: Q_w = C_w * (L + 1.8*W) * d_crest^1.5  (frontal + side)
                // For a straight curb opening, Q_w = C_w * curb_len * d_crest^1.5
                const double L_curb = fmax(curb_len, 0.01);
                const double Q_w = 3.0 * L_curb * pow(fmin(d_crest, fmax(curb_ht, 0.01)), 1.5);
                // Orifice (d_crest > 1.4 * curb_ht): Q_o = C_o * A_o * sqrt(2*g*d_crest)
                const double Ao = L_curb * fmax(curb_ht, 0.01);
                const double Q_o = cd * Ao * sqrt(fmax(2.0 * g * d_crest, 1.0e-12));
                Q = fmin(Q_w, Q_o);
            } else if (type == 2) {
                // ── SLOTTED inlet ────────────────────────────────────────
                const double As = slot_len * slot_wid;
                const double Q_w = 3.0 * 2.0 * (slot_len + slot_wid) * pow(d_crest, 1.5);
                const double Q_o = cd * As * sqrt(fmax(2.0 * g * d_crest, 1.0e-12));
                Q = fmin(Q_w, Q_o);
            } else if (type == 3) {
                // ── COMBO (grate + curb) ─────────────────────────────────
                const double P_g = 2.0 * (grate_len + grate_wid);
                const double Ag = grate_len * grate_wid * open_frac;
                const double Q_w_g = 3.0 * P_g * pow(d_crest, 1.5);
                const double Q_o_g = cd * Ag * sqrt(fmax(2.0 * g * d_crest, 1.0e-12));
                const double Q_grate = fmin(Q_w_g, Q_o_g);
                const double L_curb = fmax(curb_len, 0.01);
                const double Q_w_c = 3.0 * L_curb * pow(fmin(d_crest, fmax(curb_ht, 0.01)), 1.5);
                const double Ao = L_curb * fmax(curb_ht, 0.01);
                const double Q_o_c = cd * Ao * sqrt(fmax(2.0 * g * d_crest, 1.0e-12));
                Q = fmin(Q_w_g, Q_o_g) + fmin(Q_w_c, Q_o_c);
            } else {
                // ── Custom / unknown type — fallback weir ────────────────
                const double L_weir = (face_width) ? fmax(face_width[k], 0.01) : 0.01;
                Q = 1.7 * L_weir * pow(d_crest, 1.5);
            }

            // Clamp to qmax (if positive)
            if (isfinite(qmax) && qmax > 0.0) Q = fmin(Q, qmax);

            // Reverse (surcharge / relief): when inlet WSE > 2D WSE, capture
            // is zero; relief flow is handled by SURFACE_2D_JUNCTION_OVERFLOW
            // or a dedicated face class (not yet implemented).
        } else if (face_width) {
            // Fallback when no HEC-22 arrays available: simple weir placeholder
            const double L_weir = fmax(face_width[k], 0.01);
            Q = 1.7 * L_weir * pow(dH, 1.5);
        }

        // F_h convention: positive = L→R. Water flows R→L (2D→inlet), so F_h = -Q.
        const bool direct_inject_inlet = (n_cells_2d > 0);
        // When direct_inject is active (pipe1d advance with solver_dev), the
        // pipe-side mass is handled by the atomicAdd below — zero face_F_h so
        // the fold+Godunov pipeline doesn't double-count.  When direct_inject
        // is false (legacy path), face_F_h feeds the fold pipeline.
        face_F_h[k] = direct_inject_inlet ? 0.0 : -Q;
        face_F_Q[k] = 0.0;     // no 1D momentum for source-sink

        // 2D side: mass loss with proportional momentum extraction
        if (h2 > h_min && Q > 0.0) {
            const double dh_2d = Q * dt / fmax(h2, 1.0e-12);
            const double h2_safe = fmax(h2, h_min);
            const double u_2d = hu2 / h2_safe;
            const double v_2d = hv2 / h2_safe;
            atomicAdd(&d_ext_struct_flux_h[R], -Q);      // mass removal
            atomicAdd(&d_ext_struct_flux_hu[R], -u_2d * dh_2d);  // proportion. momentum
            atomicAdd(&d_ext_struct_flux_hv[R], -v_2d * dh_2d);
        }
        // Pipe-side direct update: Q > 0 means 2D → pipe (inlet capture), so pipe
        // gains mass.  Only fires when n_cells_2d > 0 (the direct_inject path).
        if (d_A && cell_length && cell_length[L] > 0.0 && Q > 0.0 && n_cells_2d > 0) {
            atomicAdd(&d_A[L], Q * dt / fmax(cell_length[L], 1.0e-3));
        }
        // No momentum added to inlet cell (vertical turn dissipates horizontal momentum).
    }

    // ── SURFACE_2D_JUNCTION_OVERFLOW face (class 5) — Source-sink, weir ──
    if (pass == 1 && cls == 5) {
        if (L < 0 || L >= n_cells_all) return;
        if (R < 0 || R >= n_cells_2d) return;
        if (!cell_h_2d || !cell_zb_2d || !d_ext_struct_flux_h) return;

        // Manhole cell state
        const double y_L = (cell_y) ? cell_y[L] : cell_invert[L];
        const double inv_L = cell_invert[L];
        const double WSE_manhole = y_L;

        // Rim elevation
        const double rim = (face_rim_elev && face_rim_elev[k] > inv_L)
                          ? face_rim_elev[k] : 1.0e30;

        // 2D cell state
        const double h2 = cell_h_2d[R];
        const double z2 = cell_zb_2d[R];
        const double WSE_2d = z2 + h2;

        // Overflow head: positive when manhole WSE exceeds rim
        const double overflow_head = fmax(0.0, WSE_manhole - rim);

        // Broad-crested weir equation: Q = C * L * H^1.5
        const double C_weir = 1.7;
        // Use face_width as the overflow width, or default to cell_width
        const double L_weir = (face_width) ? fmax(face_width[k], 0.01)
                             : fmax(cell_width[L], 0.01);
        const double Q = C_weir * L_weir * pow(overflow_head, 1.5);

        // Q > 0 means manhole → 2D (overflow). F_h convention: positive = L→R.
        // Water flows L→R (manhole→2D), so F_h = Q.
        // When direct_inject is active, zero face_F_h — the atomicAdd below
        // handles pipe-side mass.  When direct_inject is false, face_F_h
        // feeds the fold+Godunov pipeline.
        face_F_h[k] = (n_cells_2d > 0) ? 0.0 : Q;
        face_F_Q[k] = 0.0;  // no 1D momentum

        // 2D side gains mass with zero horizontal momentum (vertical landing)
        atomicAdd(&d_ext_struct_flux_h[R], Q);
        // No momentum added — overflow water lands vertically

        // Pipe-side direct update (manhole → 2D overflow).  Q > 0 means
        // manhole → 2D, so pipe loses mass.  n_cells_2d>0 gate ensures
        // this is the only path when direct_inject is active.
        if (d_A && cell_length && cell_length[L] > 0.0 && n_cells_2d > 0) {
            atomicAdd(&d_A[L], -Q * dt / fmax(cell_length[L], 1.0e-3));
        }
    }

    // ── CULVERT face (class 6) — Structure-flow driven ──
    if (pass == 1 && cls == 6) {
        if (gi < 0 || gi >= n_culvert_faces) return;
        if (!d_ghost_culvert_struct_idx || !d_structure_flows) return;

        // L = 2D donor cell, R = 2D receiver cell
        if (L < 0 || L >= n_cells_2d || R < 0 || R >= n_cells_2d) return;

        const int32_t si = d_ghost_culvert_struct_idx[gi];
        if (si < 0) return;

        const double Q_struct = d_structure_flows[si];  // precomputed by structure kernel

        // Q_struct positive = flow from L to R (donor to receiver)
        // F_h convention: positive = L→R, so F_h = Q_struct
        face_F_h[k] = Q_struct;
        face_F_Q[k] = 0.0;

        // Apply mass exchange directly to 2D cells
        atomicAdd(&d_ext_struct_flux_h[L], -Q_struct);  // donor loses
        atomicAdd(&d_ext_struct_flux_h[R], +Q_struct);  // receiver gains
        // No momentum extraction (structure flow dissipates horizontally)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 2.4 — Fold per-face fluxes into per-cell accumulators
//
// One thread per face. Reads face_F_h[k] (mass flux from L→R) and
// face_F_Q[k] (momentum flux from L→R), then atomicAdds the contribution
// into per-cell accumulators consumed by swe2d_pipe1d_godunov_update_kernel.
//
// Ghost faces (R < 0, e.g. OUTFALL_BC, INLET_BC): only accumulate into L.
// SURFACE_2D_* faces (where R is a 2D cell index): skip — 2D side is handled
// by direct atomicAdd to d_ext_struct_flux_* inside the unified face kernel.
// STORAGE_PIPE faces (class 8, weir/orifice): accumulate into both L and R
// (L = storage cell, R = pipe cell — both are pipe1D cells).
//
// Sign convention: cell_flux_h[c] = net outflow from cell c (positive = outflow),
// matching the Godunov kernel's A_new = A - dt * cell_flux_h[c] / L.
//   - Cell L loses mass flowing to R → cell_flux_h[L] += +face_F_h[k]
//   - Cell R gains mass from L    → cell_flux_h[R] += -face_F_h[k]
// ─────────────────────────────────────────────────────────────────────────────
__global__ void swe2d_fold_face_flux_to_cells(
    int32_t                     n_faces,
    const int32_t* __restrict__ face_owner_L,
    const int32_t* __restrict__ face_owner_R,
    const int32_t* __restrict__ face_class,
    const double*  __restrict__ face_F_h,
    const double*  __restrict__ face_F_Q,
    double*                     cell_flux_h,
    double*                     cell_flux_mom,
    int32_t                     n_cells_all)
{
    int32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_faces) return;

    const int32_t L = face_owner_L[k];
    const int32_t R = face_owner_R[k];
    if (L < 0 || L >= n_cells_all) return;

    const int32_t cls = (face_class) ? face_class[k] : 0;

    // Always accumulate into L (pipe1d cell)
    atomicAdd(&cell_flux_h[L], +face_F_h[k]);
    atomicAdd(&cell_flux_mom[L], +face_F_Q[k]);

    // For INTERIOR faces (class 0) and STORAGE_PIPE faces (class 8),
    // R is also a pipe1d cell — accumulate there too
    if ((cls == 0 || cls == 8) && R >= 0 && R < n_cells_all) {
        atomicAdd(&cell_flux_h[R], -face_F_h[k]);
        atomicAdd(&cell_flux_mom[R], -face_F_Q[k]);
    }
    // For all other face classes, the R side is either:
    //   - a ghost (-1, for OUTFALL_BC/INLET_BC): no R-side storage
    //   - a 2D cell index (for SURFACE_2D_*): handled by direct atomicAdd
    // In both cases, do NOT accumulate into cell_flux_h[R].
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 2.4 — Boundary flux kernel DELETED.  Replaced by OUTFALL_BC (class 1)
// and INLET_BC (class 2) face classes in swe2d_unified_face_flux_kernel.
// ─────────────────────────────────────────────────────────────────────────────
// pipe1d_area_from_depth — device helper for end-area from node depth
//
// Uses the same linearized depth-area relation as swe2d_pipe1d_init_area_from_depth:
//   full_depth = diameter for circular, height for rectangular/elliptical
//   A = A_full * min(1.0, depth / full_depth)
// ─────────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ double pipe1d_area_from_depth(
    int32_t shape_type,
    double width,
    double height,
    double A_full,
    double depth)
{
    if (depth <= 0.0 || A_full <= 0.0) return 0.0;
    const double full_depth = (shape_type == 0) ? width : height;
    if (full_depth <= 0.0) return 0.0;
    double frac = depth / full_depth;
    if (frac > 1.0) frac = 1.0;
    return A_full * frac;
}

// ─────────────────────────────────────────────────────────────────────────────
// swe2d_pipe1d_compute_slopes_kernel — MUSCL-minmod WSE slope limiter
//
// Computes the minmod-limited gradient of cell_H for each interior cell
// (i.e., a cell with both left and right neighbors in the same link).
// Boundary cells and cells at link boundaries get slope = 0.
//
// The slope is used by the flux kernel for MUSCL reconstruction at the
// virtual-node face: H_face = H_cell ± 0.5 * L_cell * d_slope_H.
// ─────────────────────────────────────────────────────────────────────────────
__global__ __launch_bounds__(256, 1) void swe2d_pipe1d_compute_slopes_kernel(
    int32_t                     n_cells,
    const double*  __restrict__ cell_H,
    const double*  __restrict__ cell_length,
    const int32_t* __restrict__ cell_owner_link,
    const int32_t* __restrict__ cell_sub_idx,
    double*                     d_slope_H)
{
    int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells) return;

    // Interior cell: has both left and right neighbors within the same link
    double slope = 0.0;
    if (c > 0 && c < n_cells - 1) {
        if (cell_owner_link[c-1] == cell_owner_link[c]
            && cell_owner_link[c+1] == cell_owner_link[c]) {
            const double H_c   = cell_H[c];
            const double H_prev = cell_H[c-1];
            const double H_next = cell_H[c+1];
            const double L_c   = fmax(cell_length[c], 1.0e-12);
            const double L_left  = 0.5 * (cell_length[c-1] + L_c);
            const double L_right = 0.5 * (L_c + cell_length[c+1]);
            const double dH_left  = (H_c - H_prev) / fmax(L_left, 1.0e-12);
            const double dH_right = (H_next - H_c) / fmax(L_right, 1.0e-12);
            // minmod: zero if slopes have opposite sign, else choose min magnitude
            if (dH_left * dH_right > 0.0) {
                slope = (fabs(dH_left) < fabs(dH_right)) ? dH_left : dH_right;
            }
        }
    }
    d_slope_H[c] = slope;
}

/** Host wrapper for swe2d_pipe1d_compute_slopes_kernel. */
void swe2d_pipe1d_compute_slopes_kernel_host(
    int32_t               n_cells,
    const double*         cell_H,
    const double*         cell_length,
    const int32_t*        cell_owner_link,
    const int32_t*        cell_sub_idx,
    double*               d_slope_H)
{
    if (!d_slope_H) {
        // Defensive: the kernel writes d_slope_H[c] for every c — passing
        // nullptr silently corrupted device memory in production.  Caller
        // may pass null to skip the kernel; otherwise the pointer must
        // reference at least n_cells doubles.
        return;
    }
    const int32_t n_blocks = (n_cells + 255) / 256;
    swe2d_pipe1d_compute_slopes_kernel<<<n_blocks, 256>>>(
        n_cells, cell_H, cell_length, cell_owner_link, cell_sub_idx, d_slope_H);
    CUDA_CHECK(cudaGetLastError());
}

// ── MUSCL slope kernel for A and Q ──────────────────────────────────
__global__ __launch_bounds__(256, 1) void swe2d_pipe1d_compute_AQ_slopes_kernel(
    int32_t                     n_cells,
    const double* __restrict__ cell_A,
    const double* __restrict__ cell_Q,
    const double* __restrict__ cell_length,
    const int32_t* __restrict__ cell_owner_link,
    const int32_t* __restrict__ cell_sub_idx,
    double*                     d_slope_A,
    double*                     d_slope_Q)
{
    int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells) return;

    double slope_A = 0.0, slope_Q = 0.0;
    if (c > 0 && c < n_cells - 1) {
        if (cell_owner_link[c-1] == cell_owner_link[c]
            && cell_owner_link[c+1] == cell_owner_link[c])
        {
            const double A_c = cell_A[c], A_prev = cell_A[c-1], A_next = cell_A[c+1];
            const double Q_c = cell_Q[c], Q_prev = cell_Q[c-1], Q_next = cell_Q[c+1];
            const double L_c = fmax(cell_length[c], 1.0e-12);
            const double L_left  = 0.5 * (cell_length[c-1] + L_c);
            const double L_right = 0.5 * (L_c + cell_length[c+1]);
            const double dA_left  = (A_c - A_prev) / fmax(L_left, 1.0e-12);
            const double dA_right = (A_next - A_c) / fmax(L_right, 1.0e-12);
            const double dQ_left  = (Q_c - Q_prev) / fmax(L_left, 1.0e-12);
            const double dQ_right = (Q_next - Q_c) / fmax(L_right, 1.0e-12);
            // minmod: zero if slopes have opposite sign, else choose min magnitude
            if (dA_left * dA_right > 0.0)
                slope_A = (fabs(dA_left) < fabs(dA_right)) ? dA_left : dA_right;
            if (dQ_left * dQ_right > 0.0)
                slope_Q = (fabs(dQ_left) < fabs(dQ_right)) ? dQ_left : dQ_right;
        }
    }
    d_slope_A[c] = slope_A;
    d_slope_Q[c] = slope_Q;
}

/** Host wrapper for swe2d_pipe1d_compute_AQ_slopes_kernel. */
void swe2d_pipe1d_compute_AQ_slopes_host(
    int32_t               n_cells,
    const double*         cell_A,
    const double*         cell_Q,
    const double*         cell_length,
    const int32_t*        cell_owner_link,
    const int32_t*        cell_sub_idx,
    double*               d_slope_A,
    double*               d_slope_Q)
{
    if (!d_slope_A || !d_slope_Q) return;
    const int32_t n_blocks = (n_cells + 255) / 256;
    swe2d_pipe1d_compute_AQ_slopes_kernel<<<n_blocks, 256>>>(
        n_cells, cell_A, cell_Q,
        cell_length, cell_owner_link, cell_sub_idx,
        d_slope_A, d_slope_Q);
    CUDA_CHECK(cudaGetLastError());
}

// ─────────────────────────────────────────────────────────────────────────────
// swe2d_pipe1d_godunov_update_kernel — explicit RK2 Godunov solver for
// the fully-dynamic pipe1d momentum equation.
//
// Phase 2.3: extended to handle MANHOLE_CELL (class 1) and INLET_CELL (class 2)
// alongside PIPE_CELL (class 0). For manhole/inlet cells:
//   - Continuity is computed identically (A update from flux accumulation).
//   - Momentum is set to zero (no net momentum in storage nodes).
//   - Area is clamped against cell_width × cell_max_depth (surcharge).
//   - Rim overflow threshold is checked (y > rim → marked for Phase 2.4 coupling).
//
// Replaces the old SWMM-style Picard-iteration scheme. Continuity is handled
// by the HLLE mass flux. Momentum is advanced by the HLLE momentum flux
// divergence plus explicit friction source.
// ─────────────────────────────────────────────────────────────────────────────
__global__ __launch_bounds__(256, 1) void swe2d_pipe1d_godunov_update_kernel(
    int32_t                     n_cells,
    const int32_t* __restrict__ cell_from_node,
    const int32_t* __restrict__ cell_to_node,
    const double*  __restrict__ cell_length,
    const double*  __restrict__ cell_invert,
    const double*  __restrict__ cell_n,
    const int32_t* __restrict__ cell_shape_type,
    const double*  __restrict__ cell_width,
    const double*  __restrict__ cell_height,
    const double*  __restrict__ cell_area_full,
    const double*  __restrict__ cell_perim,
    const double*  __restrict__ cell_A,
    const double*  __restrict__ cell_Q,
    const double*  __restrict__ flux_Q,
    const double*  __restrict__ flux_mom,
    int32_t                     n_nodes,
    double                      dt,
    double                      g,
    double                      k_mann,
    double                      h_min,
    int32_t                     surcharge_method,
    const double*  __restrict__ cell_S0,
    double*                     cell_A_new,
    double*                     cell_Q_new,
    double*                     cell_y,
    double*                     cell_q,
    double*                     cell_fr,
    double*                     cell_h,
    double*                     cell_slot_width,
    const double*  __restrict__ cell_tables,
    int32_t                     table_N,
    double* __restrict__        A_start_save,
    double* __restrict__        Q_start_save,
    int32_t                     stage,
    double                      theta,
    double                      omega_min,
    int32_t                     friction_method,
    double                      friction_alpha,
    int32_t                     time_integrator,
    // Phase 2.3 — per-cell metadata for manhole/inlet cells
    const int32_t* __restrict__ cell_class,
    const double*  __restrict__ cell_crown,
    const double*  __restrict__ cell_rim,
    const double*  __restrict__ cell_surface_area,
    const double*  __restrict__ cell_max_depth,
    int32_t                     n_cells_all)
    // Phase 2.1 — node_invert / node_depth / vnode_H parameters removed;
    //   eta_left/eta_right at cell endpoints are now derived from cell_h +
    //   cell_invert of c and its neighbours (interior faces), or c alone
    //   (boundary faces). The from_node / to_node / n_nodes fields are still
    //   needed to distinguish boundary vs interior faces within each cell.
{
    int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells_all) return;

    const double A_curr = fmax(1.0e-12, cell_A[c]);
    const double Q_curr = cell_Q[c];
    const double params[3] = {cell_width[c], cell_height[c], 0.0};
    const int32_t shape = cell_shape_type[c];
    const double A_full = cell_area_full[c];
    const double P_full = cell_perim[c];
    const double yFull = xsect_yFull(shape, params);
    const double wMax = xsect_wMax(shape, params);
    const double L = fmax(cell_length[c], 1.0e-12);
    const double n_val = fmax(cell_n[c], 1.0e-12);
    const double A_floor = xsect_getAofY(shape, params, h_min);

    // ── Cell class dispatch ──
    bool is_pipe = true;
    if (cell_class) {
        int32_t cls = cell_class[c];
        if (cls == 1 || cls == 2) is_pipe = false;  // MANHOLE or INLET
    }

    // ── End-face WSEs (H_up, H_dn) — computed from cell states ──
    // For interior faces: average of adjacent cell WSEs
    // For boundary faces: use cell's own WSE (boundary condition handled by face kernel)
    const int32_t from_node = cell_from_node[c];
    const int32_t to_node   = cell_to_node[c];
    const double cell_wse = cell_invert[c] + cell_h[c];
    
    double H_up, H_dn;
    if (from_node >= n_nodes) {
        // Interior face: upstream neighbor is cell c-1
        const int32_t up_cell = c - 1;
        if (up_cell >= 0 && up_cell < n_cells_all) {
            H_up = 0.5 * (cell_invert[up_cell] + cell_h[up_cell] + cell_wse);
        } else {
            H_up = cell_wse;  // fallback
        }
    } else {
        // Boundary face: use cell's own WSE
        H_up = cell_wse;
    }
    if (to_node >= n_nodes) {
        // Interior face: downstream neighbor is cell c+1
        const int32_t dn_cell = c + 1;
        if (dn_cell >= 0 && dn_cell < n_cells_all) {
            H_dn = 0.5 * (cell_wse + cell_invert[dn_cell] + cell_h[dn_cell]);
        } else {
            H_dn = cell_wse;  // fallback
        }
    } else {
        // Boundary face: use cell's own WSE
        H_dn = cell_wse;
    }

    // ── Geometry ──
    double P_c, T_c, I1_c;
    pipe1d_lookup_geometry(A_curr, A_full, P_full,
        cell_tables + static_cast<int64_t>(c) * 3 * table_N, table_N, P_c, T_c, I1_c);
    const double A_eff = fmax(A_curr, A_floor);
    const double T_safe = fmax(T_c, 1.0e-10);
    const double R_h = A_eff / fmax(1.0e-12, P_c);
    const double R = R_h;
    const double R43 = pow(R_h, 4.0 / 3.0);

    const double absQ = fabs(Q_curr);
    const double Sf = (R43 > 0.0 && A_eff > 0.0)
        ? (n_val * n_val) * absQ * Q_curr / (k_mann * k_mann * A_eff * A_eff * R43 + 1.0e-12)
        : 0.0;

    // Continuity mass flux (m³/s, signed: net outflow positive).
    const double flux_A = flux_Q[c];

    // ── Momentum: only pipe cells have meaningful momentum ──
    double Q_next = 0.0;
    double eta_left = 0.0, eta_right = 0.0, d_eta_n_val = 0.0;
    double gamma = 0.0;
    if (is_pipe) {
        // Momentum: finite-volume flux divergence + bed slope source + explicit
        // friction. The pressure gradient is carried implicitly by the face
        // momentum flux M = Q·u + 0.5·g·A²/T (Preissmann-slot safe: when the cell
        // is pressurised, T → slot_width and the pressure term stays finite but
        // stiff, giving the correct acoustic response). Bed slope must enter as
        // an explicit source because the pressure term uses depth-relative head.
        const double flux_mom_div = -flux_mom[c] / L;
        const double S0_cell = (cell_S0 != nullptr) ? cell_S0[c] : 0.0;
        const double src_gravity = g * A_eff * S0_cell;

        // eta_left/eta_right at cell endpoints — computed from cell WSEs
        // (Phase 2.3: replaced vnode_H and node_depth reads)
        if (from_node >= 0) {
            if (from_node >= n_nodes) {
                // Interior face: upstream neighbor is cell c-1
                const int32_t up_cell = c - 1;
                eta_left = (up_cell >= 0 && up_cell < n_cells_all)
                    ? 0.5 * (cell_invert[up_cell] + cell_h[up_cell] + cell_wse)
                    : cell_wse;
            } else {
                // Boundary face: use cell's own WSE
                eta_left = cell_wse;
            }
        }
        if (to_node >= 0) {
            if (to_node >= n_nodes) {
                // Interior face: downstream neighbor is cell c+1
                const int32_t dn_cell = c + 1;
                eta_right = (dn_cell >= 0 && dn_cell < n_cells_all)
                    ? 0.5 * (cell_wse + cell_invert[dn_cell] + cell_h[dn_cell])
                    : cell_wse;
            } else {
                // Boundary face: use cell's own WSE
                eta_right = cell_wse;
            }
        }
        d_eta_n_val = (eta_right - eta_left) / L;

        // Manning's friction coefficient γ (units: 1/s).  Matches 2D's
        // apply_friction_cuda_local (cpp/src/swe2d_gpu.cu:404) with h→R and
        // u→|Q|/A.
        const double gamma_nat = (R43 > 0.0 && A_eff > 0.0 && absQ > 1e-9)
            ? g * n_val * n_val * absQ / (k_mann * k_mann * A_eff * R * R43)
            : omega_min;

        // SPEC §Phase A stability hardening: the natural γ for a pipe with large
        // R (~1-5 ft) is fundamentally ~100-1000× weaker than the 2D solver's cf
        // at shallow depths (h~0.01 ft → h^(4/3)~0.002).  Manning's friction
        // vanishes as the pipe fills (γ ∝ Q/A), while the 2D solver's damping
        // grows as cells shallow (cf·|u| ∝ 1/h^(4/3)).  The pipe solver needs a
        // complementary term that stays strong at large Q / large A.
        //
        // The additive term is |Q|·friction_alpha/A_full — linear in Q,
        // independent of current fill, so the larger Q gets, the stronger the
        // damping.  This is NOT in the Casulli/Hu papers (they use implicit θ
        // on pressure to remove the need for extra damping), but it is the
        // pipe's analog of the 2D solver's cf·spd substepping — both ensure
        // the effective damping does not collapse at large conduit dimensions.
        const double gamma_stable = friction_alpha * absQ / fmax(A_full, 1.0e-12);
        gamma = gamma_nat + gamma_stable;

        // SPEC §Phase A fix-up: subtract ONLY the bed-slope source that
        // flux_mom_div doesn't include. flux_mom_div already carries the
        // integrated hydrostatic pressure gradient via 0.5·g·A²/T inside
        // M_c/M_n, so we must NOT add -g·A·dη here (Casulli-style) — it
        // would double-count the pressure gradient and destabilize pressurized
        // simulations with deep A-differences between neighbour cells (the
        // 16k-cfs runaway we saw at t=1.25 hr).  Phase B's tridiagonal
        // would put pressure back implicitly; until then the only safe
        // semi-implicit layer is friction.
        //
        // NOTE: -g·A·Sf is included in the explicit force because the friction
        // in the denominator alone is too weak for a pipe at realistic Q
        // (γ·dt ≈ 0.001 at Q=1000 cfs for a 10×5 box — see discussion in
        // docs/pipe1d_casulli_hu_plan.md §Phase A).  The combination of explicit
        // + implicit friction provides roughly 2·γ·dt damping per step.
        const double explicit_force = flux_mom_div
                                    + src_gravity
                                    - g * A_eff * Sf;

        // ── Friction substepping (mirrors 2D solver's apply_friction_cuda_local) ──
        // When the friction Courant number ν = γ·dt exceeds the target
        // (2.0 for pipe1d, which is tighter than the 2D solver's default of
        // 0.5–1.0), subdivide the friction update into n_sub smaller substeps
        // so the effective damping stays within numerical stability.  Without
        // this, γ·dt at dt=0.05 s for a large pipe can be O(1), which is
        // borderline for the explicit part of the momentum update.
        Q_next = Q_curr;
        {
            // Default: no substepping (nu <= target → single step).
            // For friction_method == 1 (SUBSTEPPING), subdivide when ν > target.
            // For friction_method == 0 (NONE) or 2 (ALPHA_BOOST), skip
            // substepping entirely (ALPHA_BOOST uses the alpha term instead).
            if (friction_method == 1) {
                constexpr int FRICTION_MAX_SUBSTEPS = 16;
                constexpr double FRICTION_TARGET_COURANT = 2.0;
                const double nu = gamma * dt;
                int n_sub = 1;
                if (nu > FRICTION_TARGET_COURANT) {
                    n_sub = static_cast<int>(ceil(nu / FRICTION_TARGET_COURANT));
                    if (n_sub > FRICTION_MAX_SUBSTEPS) n_sub = FRICTION_MAX_SUBSTEPS;
                }
                const double dt_sub = dt / static_cast<double>(n_sub);
                for (int k = 0; k < n_sub; ++k) {
                    const double absQk = fabs(Q_next);
                    const double gamma_sub = (R43 > 0.0 && A_eff > 0.0 && absQk > 1e-9)
                        ? g * n_val * n_val * absQk / (k_mann * k_mann * A_eff * R * R43)
                        : omega_min;
                    Q_next = (Q_next + dt_sub * explicit_force) / (1.0 + gamma_sub * dt_sub);
                }
            } else {
                // No substepping — single RK2 update with (1+γ·dt) denominator.
                const double denom = 1.0 + gamma * dt;
                Q_next = (Q_curr + dt * explicit_force) / denom;
            }
        }
    }

    // ── Continuity (common to all cell classes) ──
    double A_next;
    if (time_integrator == 0) {
        // RK1 (Forward Euler): single stage, directly write result.
        // No start-save needed.
        A_next = A_curr - dt * flux_A / L;
    } else if (stage == 0) {
        // RK2 stage 0: save start-of-step state, compute intermediate update
        A_start_save[c] = A_curr;
        Q_start_save[c] = Q_curr;
        A_next = A_curr - dt * flux_A / L;
    } else {
        // RK2 stage 1: average with saved start-of-step state
        const double A_mid = A_curr;
        A_next = 0.5 * (A_start_save[c] + A_mid - dt * flux_A / L);
    }

    // ── Surcharge clamp (cell-class specific) ──
    if (!is_pipe && cell_max_depth) {
        // MANHOLE/INLET: clamp against cell_width × cell_max_depth
        // For SLOT surcharge, allow unlimited depth (overflow handled by
        // SURFACE_2D_JUNCTION_OVERFLOW face or the old overflow kernel).
        // For non-SLOT, clamp at max_A (physical storage limit).
        if (surcharge_method == SURCHARGE_SLOT) {
            A_next = fmax(A_floor, A_next);
        } else {
            const double max_A = cell_width[c] * fmax(cell_max_depth[c], 0.0);
            A_next = fmax(A_floor, fmin(A_next, max_A));
        }
    } else {
        if (surcharge_method == SURCHARGE_SLOT) {
            A_next = fmax(A_floor, A_next);
        } else {
            A_next = fmax(A_floor, fmin(A_full, A_next));
        }
    }

    // ── CFL limiter (pipe cells only) ──
    if (is_pipe) {
        const double Q_cfl = A_eff * L / fmax(dt, 1.0e-12);
        if (Q_next >  Q_cfl) Q_next =  Q_cfl;
        if (Q_next < -Q_cfl) Q_next = -Q_cfl;
    }
    const double Q_cap = 1.0e6;
    Q_next = fmax(-Q_cap, fmin(Q_cap, Q_next));

    double h_new;
    // Non-pipe cells (manhole/inlet): keep the existing cell_h (from upload) as
    // the water depth.  The cross-sectional area A = W*min(h,H) is clamped at
    // A_full, so recomputing h from A/W would give at most H — losing any depth
    // above the structure height.  Use A - dt*flux/L for the next step's A.
    if (surcharge_method == SURCHARGE_SLOT && A_next > A_full) {
        h_new = xsect_getAofY_pressurised_inv(shape, params, A_next, wMax, SURCHARGE_SLOT);
    } else {
        h_new = xsect_getAofY_inv(shape, params, A_next, yFull);
    }
    const double y_new = cell_invert[c] + h_new;

    // ── Rim overflow check (manhole/inlet cells only) ──
    if (!is_pipe && cell_rim && cell_invert) {
        if (y_new > cell_rim[c] - cell_invert[c]) {
            // Overflow threshold exceeded — actual coupling via SURFACE_2D
            // face comes in Phase 2.4. For now Q is already 0.
            Q_next = 0.0;
        }
    }

    double q_new = (A_next > 1.0e-12) ? Q_next / A_next : 0.0;

    double P_new, T_new, I1_new;
    pipe1d_lookup_geometry(A_next, A_full, P_full,
        cell_tables + static_cast<int64_t>(c) * 3 * table_N, table_N, P_new, T_new, I1_new);
    double fr = fabs(q_new) / fmax(1.0e-12, sqrt(g * A_next / fmax(T_new, 1.0e-10)));

    double slot_w = 0.0;
    if (surcharge_method == SURCHARGE_SLOT && h_new > yFull) {
        slot_w = slot_width(h_new, yFull, wMax);
    }

    cell_A_new[c] = A_next;
    cell_Q_new[c] = Q_next;
    if (cell_y)          cell_y[c]          = y_new;
    if (cell_q)          cell_q[c]          = q_new;
    if (cell_fr)         cell_fr[c]         = fr;
    if (cell_h)          cell_h[c]          = h_new;
    if (cell_slot_width) cell_slot_width[c] = slot_w;
}

// Phase 2.4 — non-static so swe2d_gpu_step (swe2d_gpu.cu) can call it
// with real 2D arrays for SURFACE_2D_* class-3/4/5/6 dispatch.
void swe2d_gpu_apply_unified_face_flux(
    SWE2DDeviceState* dev,
    double dt,
    double g,
    double* d_flux_Q,
    double* d_flux_mom,
    cudaStream_t stream,
    const double* cell_h_2d = nullptr,
    const double* cell_hu_2d = nullptr,
    const double* cell_hv_2d = nullptr,
    const double* cell_zb_2d = nullptr,
    double* d_ext_struct_flux_h = nullptr,
    double* d_ext_struct_flux_hu = nullptr,
    double* d_ext_struct_flux_hv = nullptr,
    double* d_A_ptr = nullptr,
    int32_t n_cells_2d = 0,
    double current_time = 0.0,
    double h_min = 0.0,
    const double* d_slope_A = nullptr,
    const double* d_slope_Q = nullptr);

// ── Host wrapper for the Godunov RK2 scheme ──
static void swe2d_pipe1d_godunov_step_internal(
    SWE2DDeviceState* dev,
    double dt, double g, double k_mann, double h_min,
    int32_t surcharge_method,
    double* d_flux_Q, double* d_flux_mom, double* d_A_new, double* d_Q_new,
    double theta,
    double omega_min,
    int32_t friction_method,
    int32_t recon_method,
    int32_t time_integrator,
    double  friction_alpha,
    SWE2DDeviceState* solver_dev = nullptr)
{
    auto& p = dev->pipe1d;
    const int32_t n_cells_all = p.n_cells_all;
    const int32_t n_pipe_cells = p.n_pipe_cells;

    if (n_cells_all <= 0) return;

    // Zero the four pipe1d scratch buffers at the start of every godunov
    // step.  Initial allocation in mesh build also zeroes them, but this
    // safety net handles the (rare) case where a previous step was cancelled
    // mid-update and left non-zero garbage in the scratch space, which a
    // subsequent step would read and propagate into d_A/d_Q via the fold
    // kernel.
    {
        cudaStream_t _zero_stream = p.d_stream;
        CUDA_CHECK(cudaMemsetAsync(d_flux_Q,   0, static_cast<size_t>(n_cells_all) * sizeof(double), _zero_stream));
        CUDA_CHECK(cudaMemsetAsync(d_flux_mom, 0, static_cast<size_t>(n_cells_all) * sizeof(double), _zero_stream));
        CUDA_CHECK(cudaMemsetAsync(d_A_new,    0, static_cast<size_t>(n_cells_all) * sizeof(double), _zero_stream));
        CUDA_CHECK(cudaMemsetAsync(d_Q_new,    0, static_cast<size_t>(n_cells_all) * sizeof(double), _zero_stream));
        CUDA_CHECK(cudaStreamSynchronize(_zero_stream));
    }
    // Use the private pipe1d stream created during mesh build — all pipe1d
    // device memory was allocated via cudaMallocAsync on this stream, so all
    // kernel launches, memsets, and memcpys must stay on the same stream to
    // keep the memory pool consistent.
    cudaStream_t stream = p.d_stream;
    const int32_t n_blocks = (n_cells_all + 255) / 256;
    const int32_t n_blocks_pipe = (n_pipe_cells + 255) / 256;

    const double* d_cell_tables = p.d_cell_tables;
    const int32_t table_N = PIPE1D_TABLE_N;

    // MUSCL slopes for A and Q (used by unified face kernel for face reconstruction)
    if (recon_method == 1 && p.d_slope_A && p.d_slope_Q) {
        swe2d_pipe1d_compute_AQ_slopes_host(
            n_pipe_cells,
            p.d_A, p.d_Q,
            p.d_cell_length,
            p.d_cell_owner_link, p.d_cell_sub_idx,
            p.d_slope_A, p.d_slope_Q);
    }

    // Zero ext_struct_flux before stage 0 (ONCE per full step, not per stage).
    // The unified face flux stage 0 and stage 1 both atomicAdd to this buffer;
    // zeroing between stages discards stage 0's contribution and breaks mass
    // conservation between the pipe d_A (which accumulates both stages via
    // atomicAdd) and the 2D solver's downstream read.
    if (solver_dev && solver_dev->n_cells > 0
        && solver_dev->d_ext_struct_flux_h) {
        const size_t sz = static_cast<size_t>(solver_dev->n_cells) * sizeof(double);
        CUDA_CHECK(cudaMemsetAsync(solver_dev->d_ext_struct_flux_h,  0, sz, stream));
        CUDA_CHECK(cudaMemsetAsync(solver_dev->d_ext_struct_flux_hu, 0, sz, stream));
        CUDA_CHECK(cudaMemsetAsync(solver_dev->d_ext_struct_flux_hv, 0, sz, stream));
    }

    // Stage 0 flux: compute face fluxes via unified face-flux kernel + fold
    // (Phase 2.4 — all face classes, including BC faces).
    // When solver_dev is provided, pass real 2D arrays so SURFACE_2D faces
    // evaluate the exchange (direct_inject = true, writes to ext_struct_flux
    // and atomicAdds to pipe d_A).  The 2D solver step reads ext_struct_flux
    // and applies it to h/hu/hv — it does NOT re-evaluate.
    if (solver_dev && solver_dev->n_cells > 0
        && solver_dev->d_h && solver_dev->d_cell_zb) {
        swe2d_gpu_alloc_ext_struct_flux(solver_dev, solver_dev->n_cells);
        swe2d_gpu_apply_unified_face_flux(dev, dt, g, d_flux_Q, d_flux_mom, stream,
            solver_dev->d_h, solver_dev->d_hu, solver_dev->d_hv, solver_dev->d_cell_zb,
            solver_dev->d_ext_struct_flux_h,
            solver_dev->d_ext_struct_flux_hu,
            solver_dev->d_ext_struct_flux_hv,
            dev->pipe1d.d_A,
            solver_dev->n_cells, 0.0, h_min,
            p.d_slope_A, p.d_slope_Q);
    } else {
        swe2d_gpu_apply_unified_face_flux(dev, dt, g, d_flux_Q, d_flux_mom, stream,
            nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr,
            0, 0.0, h_min,
            p.d_slope_A, p.d_slope_Q);
    }

    // Stage 0 cell update (all cell classes)
    swe2d_pipe1d_godunov_update_kernel<<<n_blocks, 256, 0, stream>>>(
        n_cells_all,
        p.d_cell_from_node, p.d_cell_to_node,
        p.d_cell_length, p.d_cell_invert, p.d_cell_n,
        p.d_cell_shape_type, p.d_cell_width, p.d_cell_height,
        p.d_cell_area, p.d_cell_perim,
        p.d_A, p.d_Q,
        d_flux_Q, d_flux_mom,
        p.n_nodes, dt, g, k_mann, h_min, surcharge_method, p.d_cell_S0,
        d_A_new, d_Q_new,
        p.d_cell_y, p.d_cell_q, p.d_cell_fr, p.d_cell_h, p.d_cell_slot_width,
        d_cell_tables, table_N,
        p.d_A_start_save, p.d_Q_start_save, 0,
        theta, omega_min, friction_method,
        friction_alpha, time_integrator,
        // Phase 2.3 — per-cell metadata for manhole/inlet cells
        p.d_cell_class, p.d_cell_crown, p.d_cell_rim,
        p.d_cell_surface_area, p.d_cell_max_depth, p.n_cells_all);

    // RK1: copy result directly, skip stage 1 entirely.
    if (time_integrator == 0) {
        CUDA_CHECK(cudaMemcpyAsync(p.d_A, d_A_new,
            static_cast<size_t>(n_cells_all) * sizeof(double), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(p.d_Q, d_Q_new,
            static_cast<size_t>(n_cells_all) * sizeof(double), cudaMemcpyDeviceToDevice, stream));
        return;
    }

    { cudaError_t e = cudaStreamSynchronize(stream); if (e != cudaSuccess) throw std::runtime_error(std::string("CUDA sync error in pipe1d godunov step: ") + cudaGetErrorString(e)); }
    CUDA_CHECK(cudaMemcpyAsync(p.d_A, d_A_new,
        static_cast<size_t>(n_cells_all) * sizeof(double), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(p.d_Q, d_Q_new,
        static_cast<size_t>(n_cells_all) * sizeof(double), cudaMemcpyDeviceToDevice, stream));

    // Save stage 0 ext_struct_flux before stage 1 overwrites it.  The pipe1d
    // RK2 Godunov combine gives stage 1's direct_inject a 0.5 weight factor
    // (see swe2d_pipe1d_godunov_update_kernel stage==1 branch).  If we let
    // ext_struct_flux accumulate both stages raw (fh_0 + fh_1) the 2D solver
    // receives the full sum while the pipe only loses (fh_0 + 0.5*fh_1)*dt,
    // creating an O(fh_1*dt) conservation error per timestep.
    // Fix: after stage 1, blend ext = 0.5*(ext + saved) so the effective
    // flux matches the pipe's RK2-integrated direct_inject.
    if (solver_dev && solver_dev->n_cells > 0
        && solver_dev->d_ext_struct_flux_h) {
        // Guard: d_A_new/d_Q_new are sized for n_cells_all (pipe cells), which
        // may be far smaller than solver_dev->n_cells (2D mesh cells).  Copy
        // only the first n_cells_all entries — the fixup kernel below also runs
        // on the smaller tile.
        const size_t n_save = std::min(static_cast<size_t>(solver_dev->n_cells),
                                       static_cast<size_t>(p.n_cells_all));
        const size_t sz_save = n_save * sizeof(double);
        CUDA_CHECK(cudaMemcpyAsync(d_A_new, solver_dev->d_ext_struct_flux_h, sz_save,
            cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_Q_new, solver_dev->d_ext_struct_flux_hu, sz_save,
            cudaMemcpyDeviceToDevice, stream));
    }

    // Stage 1: re-run flux kernel with midpoint states, then correct.
    // Node_net_q accumulates stage 0 + stage 1 contributions (scaled by 0.5
    // after the substep loop completes, in the step function).
    // MUSCL slopes for A and Q (stage 1, from updated A/Q)
    if (recon_method == 1 && p.d_slope_A && p.d_slope_Q) {
        swe2d_pipe1d_compute_AQ_slopes_host(
            n_pipe_cells,
            p.d_A, p.d_Q,
            p.d_cell_length,
            p.d_cell_owner_link, p.d_cell_sub_idx,
            p.d_slope_A, p.d_slope_Q);
    }
    // Stage 1 flux: recompute face fluxes from midpoint states
    if (solver_dev && solver_dev->n_cells > 0
        && solver_dev->d_h && solver_dev->d_cell_zb) {
        swe2d_gpu_alloc_ext_struct_flux(solver_dev, solver_dev->n_cells);
        swe2d_gpu_apply_unified_face_flux(dev, dt, g, d_flux_Q, d_flux_mom, stream,
            solver_dev->d_h, solver_dev->d_hu, solver_dev->d_hv, solver_dev->d_cell_zb,
            solver_dev->d_ext_struct_flux_h,
            solver_dev->d_ext_struct_flux_hu,
            solver_dev->d_ext_struct_flux_hv,
            dev->pipe1d.d_A,
            solver_dev->n_cells, 0.0, h_min,
            p.d_slope_A, p.d_slope_Q);
    } else {
        swe2d_gpu_apply_unified_face_flux(dev, dt, g, d_flux_Q, d_flux_mom, stream,
            nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr,
            0, 0.0, h_min,
            p.d_slope_A, p.d_slope_Q);
    }

    // Fixup ext_struct_flux BEFORE the stage-1 Godunov update overwrites
    // d_A_new/d_Q_new (which still hold the stage-0 flux save from earlier).
    // Without this fixup the ext_flux accumulates fh_0 + fh_1 (both stages)
    // while the pipe's RK2 combine gives stage 1 only 0.5 weight, creating
    // an O(fh_1*dt) conservation error per timestep.
    //   corrected: ext_flux = 0.5*(ext_flux + saved) = fh_0 + 0.5*fh_1
    if (solver_dev && solver_dev->n_cells > 0
        && solver_dev->d_ext_struct_flux_h) {
        // Cap to pipe cell count — d_A_new/d_Q_new are sized for n_cells_all
        // which may be far smaller than solver_dev->n_cells (2D mesh cells).
        const int32_t n2d = std::min(static_cast<int32_t>(solver_dev->n_cells),
                                     static_cast<int32_t>(p.n_cells_all));
        const int f_grid = (n2d + 255) / 256;
        swe2d_ext_flux_rk2_fixup_kernel<<<f_grid, 256, 0, stream>>>(
            n2d, solver_dev->d_ext_struct_flux_h, solver_dev->d_ext_struct_flux_hu,
            solver_dev->d_ext_struct_flux_hv, d_A_new, d_Q_new);
    }

    swe2d_pipe1d_godunov_update_kernel<<<n_blocks, 256, 0, stream>>>(
        n_cells_all,
        p.d_cell_from_node, p.d_cell_to_node,
        p.d_cell_length, p.d_cell_invert, p.d_cell_n,
        p.d_cell_shape_type, p.d_cell_width, p.d_cell_height,
        p.d_cell_area, p.d_cell_perim,
        p.d_A, p.d_Q,
        d_flux_Q, d_flux_mom,
        p.n_nodes, dt, g, k_mann, h_min, surcharge_method, p.d_cell_S0,
        d_A_new, d_Q_new,
        p.d_cell_y, p.d_cell_q, p.d_cell_fr, p.d_cell_h, p.d_cell_slot_width,
        d_cell_tables, table_N,
        p.d_A_start_save, p.d_Q_start_save, 1,
        theta, omega_min, friction_method,
        friction_alpha, time_integrator,
        // Phase 2.3 — per-cell metadata for manhole/inlet cells
        p.d_cell_class, p.d_cell_crown, p.d_cell_rim,
        p.d_cell_surface_area, p.d_cell_max_depth, p.n_cells_all);

    // Finalise: copy the stage-1 RK2 result from d_A_new/d_Q_new back into the
    // persistent state arrays, ordered on the same stream as the stage-1
    // kernel. The caller (swe2d_pipe1d_step) used to do this with a synchronous
    // cudaMemcpy on the default stream, which races with the stage-1 kernel
    // launched on dev->d_stream — producing a torn read that left p.d_A at
    // zero while p.d_cell_h held the post-step depth. Doing the copy here on
    // the same stream preserves ordering.
    CUDA_CHECK(cudaMemcpyAsync(p.d_A, d_A_new,
        static_cast<size_t>(n_cells_all) * sizeof(double),
        cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(p.d_Q, d_Q_new,
        static_cast<size_t>(n_cells_all) * sizeof(double),
        cudaMemcpyDeviceToDevice, stream));
}

// ─────────────────────────────────────────────────────────────────────────────
// Host wrappers for pipe1d kernels
// ─────────────────────────────────────────────────────────────────────────────
#define BLOCK 256

// ── Phase 2.4 — Unified face-flux host wrapper ───────────────────────────
// Launches the unified face kernel for all faces, then folds per-face fluxes
// into per-cell accumulators.  Accepts 2D solver state for SURFACE_2D faces
// and ghost-state SoA arrays for BC faces.
void swe2d_gpu_apply_unified_face_flux(
    SWE2DDeviceState* dev,
    double dt,
    double g,
    double* d_flux_Q,
    double* d_flux_mom,
    cudaStream_t stream,
    const double* cell_h_2d,
    const double* cell_hu_2d,
    const double* cell_hv_2d,
    const double* cell_zb_2d,
    double* d_ext_struct_flux_h,
    double* d_ext_struct_flux_hu,
    double* d_ext_struct_flux_hv,
    double* d_A_ptr,
    int32_t n_cells_2d,
    double current_time,
    double h_min,
    const double* d_slope_A,
    const double* d_slope_Q)
{
    auto& p = dev->pipe1d;
    const int32_t n_faces = p.n_faces;
    if (n_faces <= 0) return;

    const int32_t grid = (n_faces + BLOCK - 1) / BLOCK;

    // Zero face scratch arrays (safety for any partially-written slots)
    CUDA_CHECK(cudaMemsetAsync(p.d_face_F_h, 0,
        static_cast<size_t>(n_faces) * sizeof(double), stream));
    CUDA_CHECK(cudaMemsetAsync(p.d_face_F_Q, 0,
        static_cast<size_t>(n_faces) * sizeof(double), stream));

    // Zero cell flux accumulators (per-stage reset — fold kernel uses atomicAdd).
    CUDA_CHECK(cudaMemsetAsync(d_flux_Q, 0,
        static_cast<size_t>(p.n_cells_all) * sizeof(double), stream));
    CUDA_CHECK(cudaMemsetAsync(d_flux_mom, 0,
        static_cast<size_t>(p.n_cells_all) * sizeof(double), stream));

    // ext_struct_flux zeroing moved to swe2d_pipe1d_godunov_step_internal
    // (done once per step, not once per RK2 stage).  Zeroing per-stage
    // wiped the stage-0 flux so the 2D solver only saw stage 1 — the pipe
    // d_A accumulated both stages via direct_inject atomicAdd, breaking
    // mass conservation by ~1× the stage-0 flux every step.

    // Re-validate face data against host cache — cross-thread CUDA pool
    // aliasing can corrupt the device arrays between mesh build and now.
    if (!p.h_face_class_cache.empty() && p.h_face_class_cache.size() >= (size_t)n_faces) {
        int32_t v; cudaMemcpy(&v, p.d_face_class, sizeof(int32_t), cudaMemcpyDeviceToHost);
        if (v != p.h_face_class_cache[0]) {
            cudaMemcpy(p.d_face_class, p.h_face_class_cache.data(),
                       n_faces * sizeof(int32_t), cudaMemcpyHostToDevice);
            cudaMemcpy(p.d_face_owner_L, p.h_face_owner_L_cache.data(),
                       n_faces * sizeof(int32_t), cudaMemcpyHostToDevice);
            cudaMemcpy(p.d_face_owner_R, p.h_face_owner_R_cache.data(),
                       n_faces * sizeof(int32_t), cudaMemcpyHostToDevice);
        }
    }

    // Use dev->d_A as the continuity injection target if d_A_ptr not provided
    double* d_A_inject = d_A_ptr ? d_A_ptr : p.d_A;

    // ── Update OUTFALL ghost WSE from node state (backward compat) ──
    // For OUTFALL_BC faces with FIXED_WSE mode (the default for unclassified
    // Phase 2.3 — Update outfall ghost WSE from end-cell state (not node_depth)
    if (p.n_outfall_faces > 0 && p.d_ghost_outfall_fixed_wse && p.d_cell_h) {
        const int32_t ghost_grid = (p.n_outfall_faces + BLOCK - 1) / BLOCK;
        swe2d_update_outfall_ghost_wse_kernel<<<ghost_grid, BLOCK, 0, stream>>>(
            p.n_outfall_faces,
            p.d_face_owner_L,
            p.d_ghost_outfall_fixed_wse,
            p.d_cell_h,
            p.d_cell_invert,
            p.n_cells_all);
        CUDA_CHECK(cudaGetLastError());
    }

    // Launch unified face kernel — two passes to avoid race between
    // pass-1 atomicAdd to d_A (SURFACE_2D classes 3/4/5) and pass-2
    // non-atomic read of d_A (class-0 INTERIOR).  See kernel header comment.
    swe2d_unified_face_flux_kernel<<<grid, BLOCK, 0, stream>>>(
        1,  // pass=1: classes 3/4/5/6 (SURFACE_2D atomicAdd to d_A + d_ext_struct_flux_*)
        n_faces,
        p.d_face_owner_L, p.d_face_owner_R,
        p.d_face_class, p.d_face_solve_mode, p.d_face_dir,
        p.d_face_F_h, p.d_face_F_Q,
        p.d_A, p.d_Q, p.d_cell_y,
        p.d_cell_invert, p.d_cell_length,
        p.d_cell_n,
        p.d_cell_width, p.d_cell_height,
        p.d_cell_area, p.d_cell_perim,
        p.d_cell_tables, p.d_cell_shape_type,
        p.n_cells_all, dt, g, PIPE1D_TABLE_N,
        // MUSCL-MC reconstruction slopes
        d_slope_A, d_slope_Q,
        // 2D solver state
        cell_h_2d, cell_hu_2d, cell_hv_2d, cell_zb_2d,
        d_ext_struct_flux_h, d_ext_struct_flux_hu, d_ext_struct_flux_hv,
        d_A_inject, n_cells_2d,
        // Ghost-state SoA
        p.d_ghost_outfall_mode, p.d_ghost_outfall_fixed_wse,
        p.d_ghost_outfall_rating, p.d_ghost_outfall_rating_n,
        p.d_ghost_outfall_tabular, p.d_ghost_outfall_tabular_n,
        p.d_ghost_outfall_link_S0,
        p.n_outfall_faces,
        p.d_ghost_inlet_Q, p.n_inlet_bc_faces,
        p.d_ghost_culvert_struct_idx,
        nullptr,  // d_structure_flows — passed by caller via future parameter
        p.n_culvert_faces,
        // Per-face attributes
        p.d_face_rim_elev, p.d_face_node_surface_area, p.d_face_ghost_idx,
        p.d_face_invert, p.d_face_nx, p.d_face_ny,
        p.d_face_width, p.d_face_area,
        p.d_face_k_in, p.d_face_k_out,
        p.d_face_depth_safety,
        // HEC-22 inlet capture arrays (may be null for meshes without inlets)
        p.d_face_inlet_type,
        p.d_face_inlet_grate_len, p.d_face_inlet_grate_wid,
        p.d_face_inlet_grate_open,
        p.d_face_inlet_curb_len, p.d_face_inlet_curb_ht,
        p.d_face_inlet_slot_len, p.d_face_inlet_slot_wid,
        p.d_face_inlet_crest, p.d_face_inlet_cd, p.d_face_inlet_qmax,
        p.n_inlet_capture_faces,
        current_time, h_min);
    // Stream sync ensures pass-1 atomicAdds to d_A are visible to pass-2 reads.
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Pass 2: classes 0/1/2 — reads d_A (now consistent post-pass-1) and
    // writes face_F_h/face_F_Q.  No race with pass-1's atomicAdds.
    swe2d_unified_face_flux_kernel<<<grid, BLOCK, 0, stream>>>(
        2,  // pass=2: classes 0/1/2 (pipe Riemann mode)
        n_faces,
        p.d_face_owner_L, p.d_face_owner_R,
        p.d_face_class, p.d_face_solve_mode, p.d_face_dir,
        p.d_face_F_h, p.d_face_F_Q,
        p.d_A, p.d_Q, p.d_cell_y,
        p.d_cell_invert, p.d_cell_length,
        p.d_cell_n,
        p.d_cell_width, p.d_cell_height,
        p.d_cell_area, p.d_cell_perim,
        p.d_cell_tables, p.d_cell_shape_type,
        p.n_cells_all, dt, g, PIPE1D_TABLE_N,
        // MUSCL-MC reconstruction slopes
        d_slope_A, d_slope_Q,
        // 2D solver state (unused for class 0/1/2 but passed for signature compat)
        cell_h_2d, cell_hu_2d, cell_hv_2d, cell_zb_2d,
        d_ext_struct_flux_h, d_ext_struct_flux_hu, d_ext_struct_flux_hv,
        d_A_inject, n_cells_2d,
        // Ghost-state SoA
        p.d_ghost_outfall_mode, p.d_ghost_outfall_fixed_wse,
        p.d_ghost_outfall_rating, p.d_ghost_outfall_rating_n,
        p.d_ghost_outfall_tabular, p.d_ghost_outfall_tabular_n,
        p.d_ghost_outfall_link_S0,
        p.n_outfall_faces,
        p.d_ghost_inlet_Q, p.n_inlet_bc_faces,
        p.d_ghost_culvert_struct_idx,
        nullptr,  // d_structure_flows — unused for pass 2
        p.n_culvert_faces,
        // Per-face attributes
        p.d_face_rim_elev, p.d_face_node_surface_area, p.d_face_ghost_idx,
        p.d_face_invert, p.d_face_nx, p.d_face_ny,
        p.d_face_width, p.d_face_area,
        p.d_face_k_in, p.d_face_k_out,
        p.d_face_depth_safety,
        // HEC-22 inlet capture arrays (may be null for meshes without inlets)
        p.d_face_inlet_type,
        p.d_face_inlet_grate_len, p.d_face_inlet_grate_wid,
        p.d_face_inlet_grate_open,
        p.d_face_inlet_curb_len, p.d_face_inlet_curb_ht,
        p.d_face_inlet_slot_len, p.d_face_inlet_slot_wid,
        p.d_face_inlet_crest, p.d_face_inlet_cd, p.d_face_inlet_qmax,
        p.n_inlet_capture_faces,
        current_time, h_min);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Fold per-face fluxes into per-cell accumulators (same grid)
    // Phase 2.4: fold kernel handles all classes; for BC faces (R < 0)
    // only accumulates into L.
    swe2d_fold_face_flux_to_cells<<<grid, BLOCK, 0, stream>>>(
        n_faces,
        p.d_face_owner_L, p.d_face_owner_R, p.d_face_class,
        p.d_face_F_h, p.d_face_F_Q,
        d_flux_Q, d_flux_mom,
        p.n_cells_all);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Fold OUTFALL_BC boundary fluxes into node_net_q for the node mass
    // balance kernel.  Only needed for OUTFALL_BC faces with backward-compat
    // node-depth coupling.
    if (p.n_outfall_faces > 0 && p.d_ghost_outfall_node_idx) {
        // Phase 2.1 — swe2d_fold_outfall_to_node_net_q retired. The unified
        // face kernel writes d_ext_struct_flux_h / d_ext_struct_flux_hu / d_ext_struct_flux_hv
        // directly for OUTFALL_BC faces; the SWE2D update kernel reads those buffers
        // and applies the mass/momentum exchange to the 2D cells. No fold into the
        // per-node d_node_net_q is needed any more.
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Node mass-balance kernels: update d_node_depth from pipe flows
// ─────────────────────────────────────────────────────────────────────────────

/** Update node depth from accumulated net flux. 1 thread per node.
 *  SPEC §2.7 — Only network nodes participate; virtual nodes inside a link
 *  have no storage and are not part of the mass balance.  The grid is sized
 *  to `n_nodes` (not n_node_slots), so vnodes are never visited.
 *  Boundary nodes (pipe-ends, outfalls) are skipped — they have no storage;
 *  their depth is set by the BC kernel or outfall kernel, not by mass balance.
 *  Non-boundary nodes are allowed to rise above node_max_depth so that surcharge
 *  volume is conserved (the pipe cell area is already capped at A_full, so the
 *  excess water is stored in the node). */

/** GPU kernel: mark nodes that have inlet assignments.
 *  1 thread per inlet.  Uses atomicExch so the final value is always 1
 *  (multiple inlets at the same node will all write the same value).
 *  @global */

// ── Init cell area from per-cell depth (Phase 2.5) ─────────────────────────
// For each cell in [0, n_cells_all), compute A(h) from d_cell_h and geometry.
// Pipe cells use the full shape-aware area; manhole/inlet cells are rectangular.
__global__ void swe2d_pipe1d_init_cell_y_kernel(
    int32_t n_cells,
    const double* __restrict__ cell_invert,
    const double* __restrict__ cell_h,
    double* __restrict__ cell_y)
{
    const int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < n_cells) cell_y[c] = cell_invert[c] + cell_h[c];
}

void swe2d_pipe1d_init_cell_area(Pipe1DDeviceState* dev, double h_min)
{
    int32_t nc = dev->n_cells_all;
    if (nc <= 0) return;
    std::vector<double> cell_h(static_cast<size_t>(nc));
    std::vector<double> cell_width(static_cast<size_t>(nc));
    std::vector<double> cell_height(static_cast<size_t>(nc));
    std::vector<double> cell_A_full(static_cast<size_t>(nc));
    std::vector<int32_t> cell_shape(static_cast<size_t>(nc));
    CUDA_CHECK(cudaMemcpy(cell_h.data(), dev->d_cell_h,
        static_cast<size_t>(nc) * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cell_width.data(), dev->d_cell_width,
        static_cast<size_t>(nc) * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cell_height.data(), dev->d_cell_height,
        static_cast<size_t>(nc) * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cell_A_full.data(), dev->d_cell_area,
        static_cast<size_t>(nc) * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cell_shape.data(), dev->d_cell_shape_type,
        static_cast<size_t>(nc) * sizeof(int32_t), cudaMemcpyDeviceToHost));

    std::vector<double> init_A(static_cast<size_t>(nc), 0.0);
    for (int32_t c = 0; c < nc; ++c) {
        const double h = fmax(cell_h[static_cast<size_t>(c)], h_min);
        const int32_t st = cell_shape[static_cast<size_t>(c)];
        if (st == XSECT_CIRCULAR) {
            const double D = fmax(cell_width[static_cast<size_t>(c)], 1.0e-12);
            const double R = 0.5 * D;
            const double hc = fmin(h, D);
            const double arg = fmax(-1.0, fmin(1.0, (R - hc) / R));
            init_A[static_cast<size_t>(c)] = R * R * acos(arg)
                - (R - hc) * sqrt(fmax(0.0, 2.0 * R * hc - hc * hc));
        } else if (st == XSECT_RECTANGULAR) {
            const double w = fmax(cell_width[static_cast<size_t>(c)], 0.0);
            const double H = fmax(cell_height[static_cast<size_t>(c)], 1.0e-12);
            init_A[static_cast<size_t>(c)] = w * fmin(h, H);
        } else if (st == XSECT_ELLIPTICAL) {
            // Matches device-side xsect_getAofY_elliptical exactly so the
            // host init and the kernel A(h) reading agree to round-off.
            //   phi = acos(1 - y / b_semi)   (central half-angle)
            //   A   = a · b · (phi - 0.5·sin(2·phi))
            // At y=H=2·b_semi: phi = acos(-1) = π, A = a·b·π.
            const double a_semi = 0.5 * fmax(cell_width[static_cast<size_t>(c)], 0.0);
            const double b_semi = 0.5 * fmax(cell_height[static_cast<size_t>(c)], 0.0);
            const double y_clamped = fmin(h, 2.0 * b_semi);
            if (y_clamped <= 0.0 || b_semi <= 0.0) {
                init_A[static_cast<size_t>(c)] = 0.0;
            } else {
                const double arg = fmax(-1.0, fmin(1.0, 1.0 - y_clamped / b_semi));
                const double phi = acos(arg);
                init_A[static_cast<size_t>(c)] = a_semi * b_semi * (phi - 0.5 * sin(2.0 * phi));
            }
        } else {
            const double w = fmax(cell_width[static_cast<size_t>(c)], 0.0);
            const double H = fmax(cell_height[static_cast<size_t>(c)], 1.0e-12);
            init_A[static_cast<size_t>(c)] = w * fmin(h, H);
        }
    }
    CUDA_CHECK(cudaMemcpy(dev->d_A, init_A.data(),
        static_cast<size_t>(nc) * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev->d_A_prev, init_A.data(),
        static_cast<size_t>(nc) * sizeof(double), cudaMemcpyHostToDevice));

    const int32_t grid = (nc + BLOCK - 1) / BLOCK;
    swe2d_pipe1d_init_cell_y_kernel<<<grid, BLOCK>>>(
        nc, dev->d_cell_invert, dev->d_cell_h, dev->d_cell_y);
    CUDA_CHECK(cudaGetLastError());
}


// ── Scale kernel (RK2 node_net_q correction) ──
__global__ void pipe1d_scale_double_kernel(
    int32_t n, double* __restrict__ d_arr, double scale)
{
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d_arr[i] *= scale;
}

// swe2d_pipe1d_step
// ─────────────────────────────────────────────────────────────────────────────
void swe2d_pipe1d_step(
    SWE2DDeviceState* dev,
    double            dt,
    const char*       solver_mode,
    int32_t           coupling_substeps,
    int32_t           implicit_iters,
    double            relaxation,
    double            g,
    double            k_mann,
    double            h_min,
    int32_t           surcharge_method,
    double            theta,
    double            omega_min,
    int32_t           friction_method,
    int32_t           recon_method     /* = 0 */,
    int32_t           time_integrator  /* = 1 */,
    double            friction_alpha   /* = 0.01 */,
    SWE2DDeviceState* solver_dev      /* = nullptr */)
{
    if (!dev || !dev->pipe1d.d_A) return;

    // The RK2 Godunov solver is the only production pipe1d path. The
    // `solver_mode`, `implicit_iters`, and `relaxation` parameters are retained
    // on the API for caller compatibility but are no longer branched on; silence
    // the resulting unused-parameter warnings.
    (void)solver_mode;
    (void)implicit_iters;
    (void)relaxation;

    auto& p = dev->pipe1d;

    // Surcharge CFL substepping: when the Preissmann slot activates, the
    // wave speed jumps to c = sqrt(g·A_full/sw) where sw = 0.01·D.  For
    // short cells this can be 10× the open-channel celerity, requiring
    // proportionally smaller dt.  The per-pipe-cell CFL-safe dt is
    // pre-computed at mesh build time in slot_cfl_dt.
    double local_dt = dt / static_cast<double>(coupling_substeps);
    if (surcharge_method == SURCHARGE_SLOT && local_dt > p.slot_cfl_dt) {
        const int32_t n_slot_sub = max(1, static_cast<int32_t>(
            ceil(local_dt / fmax(p.slot_cfl_dt, 1e-12))));
        local_dt /= static_cast<double>(n_slot_sub);
        for (int32_t sub = 0; sub < coupling_substeps * n_slot_sub; ++sub) {
            swe2d_pipe1d_godunov_step_internal(
                dev, local_dt, g, k_mann, h_min, surcharge_method,
                p.d_flux_Q_scratch, p.d_flux_mom_scratch,
                p.d_A_new_scratch, p.d_Q_new_scratch,
                theta, omega_min, friction_method,
                recon_method, time_integrator, friction_alpha,
                solver_dev);
        }
        if (dev) {
            CUDA_CHECK(cudaStreamSynchronize(p.d_stream));
        }
        return;
    }

    for (int32_t sub = 0; sub < coupling_substeps; ++sub) {
        swe2d_pipe1d_godunov_step_internal(
            dev, local_dt, g, k_mann, h_min, surcharge_method,
            p.d_flux_Q_scratch, p.d_flux_mom_scratch,
            p.d_A_new_scratch, p.d_Q_new_scratch,
            theta, omega_min, friction_method,
            recon_method, time_integrator, friction_alpha,
            solver_dev);
    }

    // Synchronize the default stream (where pipe1d kernels run) so that any
    // direct d_h writes (junction overflow kernel's atomicAdd to dev->d_h) are
    // visible to the caller's subsequent cudaMemcpy on the default stream.
    if (dev) {
        CUDA_CHECK(cudaStreamSynchronize(p.d_stream));
    }
}

/** Clamp the open-end pipe-cell area after the weir/orifice injection, and
 *  refresh cell_h / node_depth from the injected area so the next solver
 *  step sees a consistent (zero-gradient) boundary state instead of leaking
 *  mass through a stale node head.  1 thread per pipe-end.
 *  Lower bound 0 (the solver re-floors at A_floor); upper bound A_full
 *  unless the Preissmann slot carries surcharge volume.
 *  @global */

/** Fixup ext_struct_flux after pipe1d RK2 step.

 *  The pipe's RK2 Godunov combine gives stage 1's direct_inject a 0.5 weight:
 *    A_next = 0.5 * (A_start + A_mid - dt*flux_A/L)
 *  The ext_struct_flux accumulates both stages raw via atomicAdd, so the 2D
 *  solver receives fh_0 + fh_1 while the pipe only loses (fh_0+0.5*fh_1)*dt.
 *
 *  This kernel blends ext_flux with the stage-0 snapshot (saved_h/saved_hu/saved_hv)
 *  to match the pipe's effective integrated flux:
 *    ext_flux_out[c] = 0.5 * (ext_flux_in[c] + saved_h[c])
 *    (same for hu, hv)
 *
 *  1 thread per 2D cell.  @global */
__global__ __launch_bounds__(256, 4) void swe2d_ext_flux_rk2_fixup_kernel(
    int32_t n_cells,
    double* ext_flux_h,
    double* ext_flux_hu,
    double* ext_flux_hv,
    const double* __restrict__ saved_h,
    const double* __restrict__ saved_hu)
{
    int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells) return;
    if (ext_flux_h && saved_h) ext_flux_h[c] = 0.5 * (ext_flux_h[c] + saved_h[c]);
    if (ext_flux_hu && saved_hu) ext_flux_hu[c] = 0.5 * (ext_flux_hu[c] + saved_hu[c]);
}

/** GPU kernel: fold junction overflow Q (m³/s) directly into 2D cell depth h.
 *  For each cell c: h[c] += Q[c] * dt / area[c].
 *  This bypasses the external-source pipeline for tests/steps where the 2D
 *  solver does not advance (swe2d_pipe1d_step is called directly).
 *  1 thread per 2D mesh cell.  @global */
__global__ __launch_bounds__(256, 4) void swe2d_fold_junction_overflow_to_h_kernel(
    int32_t n_cells,
    const double* __restrict__ d_Q,
    const double* __restrict__ d_cell_area,
    double dt,
    double* __restrict__ d_h)
{
    int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells) return;
    if (!d_Q || !d_h || !d_cell_area) return;
    const double q = d_Q[c];
    if (!isfinite(q) || q == 0.0) return;
    atomicAdd(&d_h[c], q * dt / fmax(d_cell_area[c], 1.0e-12));
}
