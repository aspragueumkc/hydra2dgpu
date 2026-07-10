// swe2d_reconstruct.cu
// Spatial reconstruction kernels for advanced schemes.
//
// All kernels reconstruct WATER SURFACE (eta = h + zb) for C-property compliance,
// AND momentum (hu, hv) as conserved variables.  All three are reconstructed
// at each face so the HLLC solver gets consistent higher-order left/right states.
//
// The LSQ/Lagrange weights depend only on geometry (cell centroid positions
// relative to the face midpoint), so they are computed once and applied to
// all three variables — no triple cost.
//
// Contains:
//   barth_jespersen_kernel (scheme 5) — 2nd-order gradient limiter on eta
//   weno3_kernel           (scheme 6) — 3rd-order WENO on 1-ring sub-stencils
//   mp5_kernel             (scheme 8) — 4th–5th-order mapped MP on 5-cell walk
//   extract_hx_hy_kernel         — helper for scheme 5

#include "swe2d_gpu.cuh"
#include "swe2d_mesh.hpp"
#include "swe2d_solver.hpp"
#include "swe2d_units.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

// ═════════════════════════════════════════════════════════════════════════════
//  Device helpers
// ═════════════════════════════════════════════════════════════════════════════

static __device__ __forceinline__ double minmod(double a, double b)
{
    if (a * b <= 0.0) return 0.0;
    return (fabs(a) < fabs(b)) ? a : b;
}

/// Weighted LSQ plane fit for a sub-stencil, evaluated at a target point.
/// Computes the 3×3 normal-equations system once from geometry, then solves
/// for the constant term (face value) of THREE variables simultaneously.
///
/// q1 = eta = h + zb (computed internally for C-property compliance)
/// q2 = hu (passed directly)
/// q3 = hv (passed directly)
static __device__ __forceinline__ void lsq3_at_face(
    const double* __restrict__ h,
    const double* __restrict__ zb,
    const double* __restrict__ hu_arr,
    const double* __restrict__ hv_arr,
    const double* __restrict__ cx,
    const double* __restrict__ cy,
    double fx, double fy,
    const int* __restrict__ ids, int n,
    double& out_eta, double& out_hu, double& out_hv, double& out_var)
{
    if (n <= 0) { out_eta = out_hu = out_hv = 0.0; out_var = 1e6; return; }
    if (n == 1) {
        int id = ids[0];
        out_eta = h[id] + zb[id]; out_hu = hu_arr[id]; out_hv = hv_arr[id];
        out_var = 0.0; return;
    }

    double S1 = 0.0, Sx = 0.0, Sy = 0.0;
    double Sxx = 0.0, Sxy = 0.0, Syy = 0.0;
    double Se = 0.0, Su = 0.0, Sv = 0.0;
    double Sex = 0.0, Sey = 0.0, Sux = 0.0, Suy = 0.0, Svx = 0.0, Svy = 0.0;

    for (int k = 0; k < n; ++k) {
        int id = ids[k];
        double dx = cx[id] - fx;
        double dy = cy[id] - fy;
        double eta = h[id] + zb[id];
        double u = hu_arr[id];
        double v = hv_arr[id];

        S1  += 1.0;  Sx += dx;  Sy += dy;
        Sxx += dx*dx;  Sxy += dx*dy;  Syy += dy*dy;
        Se += eta;  Su += u;  Sv += v;
        Sex += eta*dx;  Sey += eta*dy;
        Sux += u*dx;  Suy += u*dy;
        Svx += v*dx;  Svy += v*dy;
    }

    double M00 = Sxx*Syy - Sxy*Sxy;
    double M01 = Sx*Syy - Sxy*Sy;
    double M02 = Sx*Sxy - Sxx*Sy;
    double det = S1*M00 - Sx*M01 + Sy*M02;

    if (fabs(det) < 1e-20) {
        double inv = 1.0 / S1;
        out_eta = Se * inv; out_hu = Su * inv; out_hv = Sv * inv;
    } else {
        double inv_det = 1.0 / det;
        out_eta = (Se*M00 - Sx*(Sex*Syy - Sxy*Sey) + Sy*(Sex*Sxy - Sxx*Sey)) * inv_det;
        out_hu  = (Su*M00 - Sx*(Sux*Syy - Sxy*Suy) + Sy*(Sux*Sxy - Sxx*Suy)) * inv_det;
        out_hv  = (Sv*M00 - Sx*(Svx*Syy - Sxy*Svy) + Sy*(Svx*Sxy - Sxx*Svy)) * inv_det;
    }

    // Smoothness from eta variance
    double mean = Se / S1;
    double res = 0.0;
    for (int k = 0; k < n; ++k) {
        double diff = (h[ids[k]] + zb[ids[k]]) - mean;
        res += diff * diff;
    }
    out_var = res;
}

/// Lagrange interpolation of 5 points at s = 0 for a single variable.
static __device__ __forceinline__ double lagrange5_at_zero(
    const double s[5], double f0, double f1, double f2, double f3, double f4)
{
    double f[5] = {f0, f1, f2, f3, f4};
    double result = 0.0;
    for (int k = 0; k < 5; ++k) {
        double L = 1.0;
        for (int j = 0; j < 5; ++j) {
            if (j == k) continue;
            double denom = s[k] - s[j];
            if (fabs(denom) < 1e-14) { L = 0.0; break; }
            L *= (-s[j]) / denom;
        }
        result += L * f[k];
    }
    return result;
}

// ═════════════════════════════════════════════════════════════════════════════
//  Extract hx/hy from Grad struct into flat arrays  (helper for scheme 5)
// ═════════════════════════════════════════════════════════════════════════════

__global__ void extract_hx_hy_kernel(
    const Grad* __restrict__ d_grad,
    double* __restrict__ hx_out,
    double* __restrict__ hy_out,
    int n_cells)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells) return;
    hx_out[c] = d_grad[c].hx;
    hy_out[c] = d_grad[c].hy;
}

// ═════════════════════════════════════════════════════════════════════════════
//  Barth-Jespersen limiter  (scheme 5)
//
//  One thread per cell.  Limits the ETA gradient so that extrapolated face
//  eta values do not exceed the min/max eta envelope of the cell and its
//  edge-neighbours.  Operates entirely on eta for C-property compliance.
//
//  The limited gradient feeds into the existing tvd_reconstruct lambda in
//  the flux kernel, which handles all 3 variables (eta, hu, hv) using
//  unlimited momentum gradients + limited eta gradient.
// ═════════════════════════════════════════════════════════════════════════════

__global__ void barth_jespersen_kernel(
    const double* __restrict__ h,
    const double* __restrict__ zb,
    const double* __restrict__ grad_x,
    const double* __restrict__ grad_y,
    const double* __restrict__ cell_cx,
    const double* __restrict__ cell_cy,
    const int* __restrict__ cell_edge_offsets,
    const int* __restrict__ cell_edge_ids,
    const int* __restrict__ edge_c0,
    const int* __restrict__ edge_c1,
    int n_cells,
    double* __restrict__ grad_x_lim,
    double* __restrict__ grad_y_lim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cells) return;

    double eta_i = h[i] + zb[i];
    double gx = grad_x[i];
    double gy = grad_y[i];
    double xi = cell_cx[i];
    double yi = cell_cy[i];

    double chi = 1.0;
    int start = cell_edge_offsets[i];
    int end   = cell_edge_offsets[i + 1];

    for (int k = start; k < end; ++k) {
        int edge = cell_edge_ids[k];
        int j = (edge_c0[edge] == i) ? edge_c1[edge] : edge_c0[edge];
        if (j < 0) continue;

        double eta_j = h[j] + zb[j];
        double dx = cell_cx[j] - xi;
        double dy = cell_cy[j] - yi;
        double eta_face = eta_i + gx * dx + gy * dy;

        double eta_min = fmin(eta_i, eta_j);
        double eta_max = fmax(eta_i, eta_j);

        double chi_k = 1.0;
        if (eta_face > eta_max) {
            double denom = eta_face - eta_i;
            if (fabs(denom) > 1e-15) chi_k = (eta_max - eta_i) / denom;
        } else if (eta_face < eta_min) {
            double denom = eta_i - eta_face;
            if (fabs(denom) > 1e-15) chi_k = (eta_i - eta_min) / denom;
        }
        chi = fmin(chi, chi_k);
    }

    chi = fmax(0.0, fmin(1.0, chi));
    grad_x_lim[i] = chi * gx;
    grad_y_lim[i] = chi * gy;
}

// ═════════════════════════════════════════════════════════════════════════════
//  WENO3  (scheme 6)
//
//  One thread per face.  Reconstructs all 3 conserved variables (eta, hu, hv)
//  at the face midpoint via 3-sub-stencil WENO blend.
//
//  Each sub-stencil candidate is a weighted LSQ plane fit (S0, S2) or linear
//  interpolation (S1), computed once and applied to all 3 variables.
//  The WENO nonlinear weights are based on eta smoothness and applied to all 3.
//
//  Output: eta_face, hu_face, hv_face — the reconstructed values at the face.
//  The flux kernel uses these directly as etaL_rec=etaR_rec, etc.
// ═════════════════════════════════════════════════════════════════════════════

__global__ void weno3_kernel(
    const double* __restrict__ h,
    const double* __restrict__ zb,
    const double* __restrict__ hu_arr,
    const double* __restrict__ hv_arr,
    const double* __restrict__ cell_cx,
    const double* __restrict__ cell_cy,
    const double* __restrict__ face_mid_x,
    const double* __restrict__ face_mid_y,
    const int* __restrict__ face_stencil_S0_offsets,
    const int* __restrict__ face_stencil_S0_cells,
    const int* __restrict__ face_stencil_S1,
    const int* __restrict__ face_stencil_S2_offsets,
    const int* __restrict__ face_stencil_S2_cells,
    int n_faces,
    double* __restrict__ eta_face,
    double* __restrict__ hu_face,
    double* __restrict__ hv_face)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n_faces) return;

    int i = face_stencil_S1[2 * f + 0];
    int j = face_stencil_S1[2 * f + 1];
    if (j < 0) {
        eta_face[f] = h[i] + zb[i];
        hu_face[f]  = hu_arr[i];
        hv_face[f]  = hv_arr[i];
        return;
    }

    double xf = face_mid_x[f];
    double yf = face_mid_y[f];

    double xi = cell_cx[i], yi = cell_cy[i];
    double xj = cell_cx[j], yj = cell_cy[j];
    double eta_i = h[i] + zb[i], eta_j = h[j] + zb[j];

    // ── S0: upwind lobe — LSQ plane fit for all 3 variables ─────────────
    int s0 = face_stencil_S0_offsets[f];
    int e0 = face_stencil_S0_offsets[f + 1];
    int n0 = e0 - s0;
    double q0_eta, q0_hu, q0_hv, beta_0;
    if (n0 >= 2) {
        lsq3_at_face(h, zb, hu_arr, hv_arr,
                     cell_cx, cell_cy, xf, yf,
                     face_stencil_S0_cells + s0, n0,
                     q0_eta, q0_hu, q0_hv, beta_0);
    } else {
        q0_eta = eta_i; q0_hu = hu_arr[i]; q0_hv = hv_arr[i]; beta_0 = 1e6;
    }

    // ── S1: central pair — linear interpolation ──────────────────────────
    double dx_ij = xj - xi, dy_ij = yj - yi;
    double dist_ij_sq = dx_ij * dx_ij + dy_ij * dy_ij;
    double t = (dist_ij_sq > 1e-24)
        ? ((xf - xi) * dx_ij + (yf - yi) * dy_ij) / dist_ij_sq : 0.5;
    t = fmax(0.0, fmin(1.0, t));
    double q1_eta = eta_i + t * (eta_j - eta_i);
    double q1_hu  = hu_arr[i] + t * (hu_arr[j] - hu_arr[i]);
    double q1_hv  = hv_arr[i] + t * (hv_arr[j] - hv_arr[i]);
    double beta_1 = (eta_i - eta_j) * (eta_i - eta_j);

    // ── S2: downwind lobe — LSQ plane fit for all 3 variables ───────────
    int s2 = face_stencil_S2_offsets[f];
    int e2 = face_stencil_S2_offsets[f + 1];
    int n2 = e2 - s2;
    double q2_eta, q2_hu, q2_hv, beta_2;
    if (n2 >= 2) {
        lsq3_at_face(h, zb, hu_arr, hv_arr,
                     cell_cx, cell_cy, xf, yf,
                     face_stencil_S2_cells + s2, n2,
                     q2_eta, q2_hu, q2_hv, beta_2);
    } else {
        q2_eta = eta_j; q2_hu = hu_arr[j]; q2_hv = hv_arr[j]; beta_2 = 1e6;
    }

    // ── Nonlinear WENO weights (computed from eta smoothness) ────────────
    double d_weights[3] = {0.1, 0.6, 0.3};
    double eps = 1e-6;
    double betas[3] = {beta_0, beta_1, beta_2};
    double alpha[3], alpha_sum = 0.0;
    for (int k = 0; k < 3; ++k) {
        alpha[k] = d_weights[k] / ((eps + betas[k]) * (eps + betas[k]));
        alpha_sum += alpha[k];
    }
    double w[3];
    for (int k = 0; k < 3; ++k) w[k] = alpha[k] / alpha_sum;

    // ── Weighted combination for all 3 variables ────────────────────────
    double rec_eta = w[0]*q0_eta + w[1]*q1_eta + w[2]*q2_eta;
    double rec_hu  = w[0]*q0_hu  + w[1]*q1_hu  + w[2]*q2_hu;
    double rec_hv  = w[0]*q0_hv  + w[1]*q1_hv  + w[2]*q2_hv;

    // Safety: clamp eta to local envelope
    double eta_min = fmin(eta_i, eta_j);
    double eta_max = fmax(eta_i, eta_j);
    rec_eta = fmax(eta_min, fmin(eta_max, rec_eta));

    eta_face[f] = rec_eta;
    hu_face[f]  = rec_hu;
    hv_face[f]  = rec_hv;
}

// ═════════════════════════════════════════════════════════════════════════════
//  MP5  (scheme 8) — Suresh-Huynh (1997) Mapped Monotonicity-Preserving
//
//  One thread per face.  Projects the 5-cell stencil onto the face normal,
//  computes high-order value via Lagrange interpolation, then applies the
//  runtime Suresh-Huynh limiter.  Reconstructs all 3 variables (eta, hu, hv).
// ═════════════════════════════════════════════════════════════════════════════

__global__ void mp5_kernel(
    const double* __restrict__ h,
    const double* __restrict__ zb,
    const double* __restrict__ hu_arr,
    const double* __restrict__ hv_arr,
    const double* __restrict__ cell_cx,
    const double* __restrict__ cell_cy,
    const double* __restrict__ face_mid_x,
    const double* __restrict__ face_mid_y,
    const double* __restrict__ face_nx,
    const double* __restrict__ face_ny,
    const int* __restrict__ face_stencil_5,
    int n_faces,
    double* __restrict__ eta_face,
    double* __restrict__ hu_face,
    double* __restrict__ hv_face)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n_faces) return;

    const int* st = &face_stencil_5[5 * f];

    double mx = face_mid_x[f], my = face_mid_y[f];
    double nx = face_nx[f], ny = face_ny[f];

    // Project centroids onto face normal → 1-D coordinates (face midpoint at s=0)
    double s[5];
    double eta[5], hu_v[5], hv_v[5];
    for (int k = 0; k < 5; ++k) {
        int id = st[k];
        s[k]   = (cell_cx[id] - mx) * nx + (cell_cy[id] - my) * ny;
        eta[k] = h[id] + zb[id];
        hu_v[k] = hu_arr[id];
        hv_v[k] = hv_arr[id];
    }

    // High-order Lagrange interpolation at s=0 for all 3 variables
    double fHO_eta = lagrange5_at_zero(s, eta[0], eta[1], eta[2], eta[3], eta[4]);
    double fHO_hu  = lagrange5_at_zero(s, hu_v[0], hu_v[1], hu_v[2], hu_v[3], hu_v[4]);
    double fHO_hv  = lagrange5_at_zero(s, hv_v[0], hv_v[1], hv_v[2], hv_v[3], hv_v[4]);

    // ── Monotonicity-preserving limiter on eta ───────────────────────────
    double f_min = fmin(fmin(eta[1], eta[2]), eta[3]);
    double f_max = fmax(fmax(eta[1], eta[2]), eta[3]);

    double fMP_eta, fMP_hu, fMP_hv;

    if (fHO_eta >= f_min && fHO_eta <= f_max) {
        // Smooth: accept high-order for all 3 variables
        fMP_eta = fHO_eta;
        fMP_hu  = fHO_hu;
        fMP_hv  = fHO_hv;
    } else {
        // Limiting needed on eta → compute blend factor and apply to all 3

        // TVD value with proper non-uniform spacing
        double s_u = s[2], s_v = s[3];
        double ds = s_v - s_u;
        double fTVD_eta;
        if (fabs(ds) < 1e-14) {
            fTVD_eta = 0.5 * (eta[2] + eta[3]);
        } else {
            double slope_r = (eta[3] - eta[2]) / (s[3] - s[2]);
            double slope_l = (eta[2] - eta[1]) / (s[2] - s[1]);
            double limited = minmod(slope_r, slope_l);
            fTVD_eta = eta[2] + limited * (0.0 - s_u);
        }

        // Check smooth extremum
        double d_ho = fHO_eta - eta[2];
        double d_r  = eta[3] - eta[2];
        double d_l  = eta[2] - eta[1];

        double blend;  // 1.0 = high order, 0.0 = TVD
        if (d_ho * d_r > 0.0 && d_ho * d_l > 0.0) {
            // Smooth extremum: allow controlled overshoot (up to 2× TVD)
            double d_tvd = fTVD_eta - eta[2];
            if (fabs(d_tvd) < 1e-20) {
                blend = 0.0;
            } else {
                double ratio = d_ho / d_tvd;
                double mapped = fmax(-2.0, fmin(2.0, ratio));
                blend = mapped / ratio;  // how much of fHO to keep
                blend = fmax(0.0, fmin(1.0, blend));
            }
        } else {
            // Discontinuity: pure TVD
            blend = 0.0;
        }

        // Apply blend to all 3 variables
        fMP_eta = (1.0 - blend) * fTVD_eta + blend * fHO_eta;
        // For hu/hv: compute TVD values similarly
        double fTVD_hu, fTVD_hv;
        if (fabs(ds) < 1e-14) {
            fTVD_hu = 0.5 * (hu_v[2] + hu_v[3]);
            fTVD_hv = 0.5 * (hv_v[2] + hv_v[3]);
        } else {
            double sl_hu_r = (hu_v[3] - hu_v[2]) / (s[3] - s[2]);
            double sl_hu_l = (hu_v[2] - hu_v[1]) / (s[2] - s[1]);
            double lim_hu = minmod(sl_hu_r, sl_hu_l);
            fTVD_hu = hu_v[2] + lim_hu * (0.0 - s_u);

            double sl_hv_r = (hv_v[3] - hv_v[2]) / (s[3] - s[2]);
            double sl_hv_l = (hv_v[2] - hv_v[1]) / (s[2] - s[1]);
            double lim_hv = minmod(sl_hv_r, sl_hv_l);
            fTVD_hv = hv_v[2] + lim_hv * (0.0 - s_u);
        }
        fMP_hu = (1.0 - blend) * fTVD_hu + blend * fHO_hu;
        fMP_hv = (1.0 - blend) * fTVD_hv + blend * fHO_hv;

        // Safety clip on eta
        fMP_eta = fmax(f_min - 1e-10, fmin(f_max + 1e-10, fMP_eta));
    }

    eta_face[f] = fMP_eta;
    hu_face[f]  = fMP_hu;
    hv_face[f]  = fMP_hv;
}
