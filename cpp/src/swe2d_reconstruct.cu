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

/// 1D linear reconstruction of q at s = 0 from two points (s_a, q_a), (s_b, q_b).
///
/// q(s) = q_a * (s - s_b)/(s_a - s_b) + q_b * (s - s_a)/(s_b - s_a)
/// At s = 0: q(0) = (q_a * s_b - q_b * s_a) / (s_b - s_a)
static __device__ __forceinline__ double weno3_linear2(
    double s_a, double q_a, double s_b, double q_b)
{
    double ds = s_b - s_a;
    if (fabs(ds) < 1.0e-14) return 0.5 * (q_a + q_b);
    return (q_a * s_b - q_b * s_a) / ds;
}

/// Jiang-Shu smoothness indicator for a 2-point linear stencil projected
/// onto the face-normal coordinate.
static __device__ __forceinline__ double weno3_beta2(
    double s_a, double q_a, double s_b, double q_b)
{
    double ds = s_b - s_a;
    if (fabs(ds) < 1.0e-14) return 0.0;
    double dq = q_b - q_a;
    double slope = dq / ds;
    return slope * slope;
}

/// Textbook 1D WENO3 reconstruction at a face (s = 0) using projected
/// 1-D coordinates.  Two 2-cell linear sub-stencils per side with the
/// standard Jiang-Shu optimal weights.
///
/// Left state (looking from c0 towards c1):
///   sub-stencils: {u, c0} and {c0, c1}
///   linear weights: d0 = 1/3, d1 = 2/3
///
/// Right state (looking from c1 towards c0):
///   sub-stencils: {c0, c1} and {c1, v}
///   linear weights: d0 = 2/3, d1 = 1/3
static __device__ __forceinline__ double weno3_reconstruct(
    double q_u, double q_c0, double q_c1, double q_v,
    double s_u, double s_c0, double s_c1, double s_v,
    bool is_left)
{
    double q0, q1, beta0, beta1, d0, d1;
    if (is_left) {
        q0    = weno3_linear2(s_u, q_u, s_c0, q_c0);
        q1    = weno3_linear2(s_c0, q_c0, s_c1, q_c1);
        beta0 = weno3_beta2(s_u, q_u, s_c0, q_c0);
        beta1 = weno3_beta2(s_c0, q_c0, s_c1, q_c1);
        d0 = 1.0 / 3.0;
        d1 = 2.0 / 3.0;
    } else {
        q0    = weno3_linear2(s_c0, q_c0, s_c1, q_c1);
        q1    = weno3_linear2(s_c1, q_c1, s_v, q_v);
        beta0 = weno3_beta2(s_c0, q_c0, s_c1, q_c1);
        beta1 = weno3_beta2(s_c1, q_c1, s_v, q_v);
        d0 = 2.0 / 3.0;
        d1 = 1.0 / 3.0;
    }

    const double eps = 1.0e-6;
    double alpha0 = d0 / ((eps + beta0) * (eps + beta0));
    double alpha1 = d1 / ((eps + beta1) * (eps + beta1));
    double asum = alpha0 + alpha1;
    if (asum <= 0.0) return q0;
    double w0 = alpha0 / asum;
    double w1 = alpha1 / asum;
    return w0 * q0 + w1 * q1;
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
//  WENO3  (scheme 6) — component-wise left/right reconstruction of conserved vars
//
//  One thread per face.  Projects the relevant cells onto the face normal and
//  performs a 1D WENO3 reconstruction separately for the left and right
//  Riemann states of each conserved variable (eta, hu, hv).  Two linear
//  sub-stencils per side with the standard Jiang-Shu smoothness indicators and
//  optimal weights (1/3, 2/3) for the left state and (2/3, 1/3) for the right.
//
//  Left state  uses {u, c0} and {c0, c1}, where u is the most upwind neighbour
//  of owner c0.  Right state uses {c0, c1} and {c1, v}, where v is the most
//  upwind neighbour of c1.  Boundary faces simply duplicate the owner cell.
//
//  Output: eta/hu/hv as distinct left/right values at each face, consumed by
//  the HLLC flux kernel.
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
    const double* __restrict__ face_nx,
    const double* __restrict__ face_ny,
    const int* __restrict__ face_stencil_S0_offsets,
    const int* __restrict__ face_stencil_S0_cells,
    const int* __restrict__ face_stencil_S1,
    const int* __restrict__ face_stencil_S2_offsets,
    const int* __restrict__ face_stencil_S2_cells,
    int n_faces,
    double h_min,
    double g,
    double* __restrict__ eta_face_L,
    double* __restrict__ eta_face_R,
    double* __restrict__ hu_face_L,
    double* __restrict__ hu_face_R,
    double* __restrict__ hv_face_L,
    double* __restrict__ hv_face_R)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n_faces) return;

    int c0 = face_stencil_S1[2 * f + 0];
    int c1 = face_stencil_S1[2 * f + 1];

    // Boundary face: left/right both see the owner cell (flux kernel applies
    // the physical boundary condition on the right side later).
    if (c1 < 0) {
        double eta = h[c0] + zb[c0];
        double u = (h[c0] > h_min) ? hu_arr[c0] / h[c0] : 0.0;
        double v = (h[c0] > h_min) ? hv_arr[c0] / h[c0] : 0.0;
        eta_face_L[f] = eta; eta_face_R[f] = eta;
        hu_face_L[f] = hu_arr[c0]; hu_face_R[f] = hu_arr[c0];
        hv_face_L[f] = hv_arr[c0]; hv_face_R[f] = hv_arr[c0];
        return;
    }

    double mx = face_mid_x[f];
    double my = face_mid_y[f];
    double nx = face_nx[f];
    double ny = face_ny[f];

    auto project = [&](int id) -> double {
        return (cell_cx[id] - mx) * nx + (cell_cy[id] - my) * ny;
    };

    double s_c0 = project(c0);
    double s_c1 = project(c1);

    // Select most upwind neighbour of c0 for the left state (minimum s).
    int s0_beg = face_stencil_S0_offsets[f];
    int s0_end = face_stencil_S0_offsets[f + 1];
    int id_u = -1;
    double s_u = s_c0;  // fallback: c0 itself
    for (int k = s0_beg; k < s0_end; ++k) {
        int id = face_stencil_S0_cells[k];
        double s = project(id);
        if (id_u < 0 || s < s_u) {
            id_u = id;
            s_u = s;
        }
    }
    if (id_u < 0) { id_u = c0; s_u = s_c0; }

    // Select most upwind neighbour of c1 for the right state (maximum s).
    int s2_beg = face_stencil_S2_offsets[f];
    int s2_end = face_stencil_S2_offsets[f + 1];
    int id_v = -1;
    double s_v = s_c1;  // fallback: c1 itself
    for (int k = s2_beg; k < s2_end; ++k) {
        int id = face_stencil_S2_cells[k];
        double s = project(id);
        if (id_v < 0 || s > s_v) {
            id_v = id;
            s_v = s;
        }
    }
    if (id_v < 0) { id_v = c1; s_v = s_c1; }

    // Degenerate stencil detection: on triangle meshes, S0/S2 may have only
    // 1 cell, or the selected upwind cell may be nearly coincident with c0/c1
    // (projected distance < 1% of c0-c1 spacing).  In these cases the WENO3
    // smoothness indicators become erratic and the nonlinear weights collapse
    // to a single sub-stencil, producing oscillatory reconstruction.  Fall
    // back to first-order upwind (copy cell values) when the stencil is
    // insufficient for a meaningful WENO3 reconstruction.
    const double ds_pair = fabs(s_c1 - s_c0);
    const bool degenerate_stencil = (ds_pair < 1.0e-14)
        || (fabs(s_u - s_c0) < 0.01 * ds_pair)
        || (fabs(s_v - s_c1) < 0.01 * ds_pair)
        || ((s0_end - s0_beg) < 1)
        || ((s2_end - s2_beg) < 1);

    // Conserved variables for the stencil cells: eta, hu, hv.
    double eta_u  = h[id_u]  + zb[id_u],  eta_c0 = h[c0] + zb[c0], eta_c1 = h[c1] + zb[c1], eta_v  = h[id_v]  + zb[id_v];
    double hu_u   = hu_arr[id_u], hu_c0  = hu_arr[c0], hu_c1  = hu_arr[c1], hu_v   = hu_arr[id_v];
    double hv_u   = hv_arr[id_u], hv_c0  = hv_arr[c0], hv_c1  = hv_arr[c1], hv_v   = hv_arr[id_v];

    double eta_L, eta_R, hu_L, hu_R, hv_L, hv_R;

    if (degenerate_stencil) {
        eta_L = eta_c0; eta_R = eta_c1;
        hu_L  = hu_c0;  hu_R  = hu_c1;
        hv_L  = hv_c0;  hv_R  = hv_c1;
    } else {
    // WENO3 left/right reconstruction of each conserved variable.
    eta_L = weno3_reconstruct(eta_u, eta_c0, eta_c1, eta_v, s_u, s_c0, s_c1, s_v, true);
    eta_R = weno3_reconstruct(eta_u, eta_c0, eta_c1, eta_v, s_u, s_c0, s_c1, s_v, false);
    hu_L  = weno3_reconstruct(hu_u,  hu_c0,  hu_c1,  hu_v,  s_u, s_c0, s_c1, s_v, true);
    hu_R  = weno3_reconstruct(hu_u,  hu_c0,  hu_c1,  hu_v,  s_u, s_c0, s_c1, s_v, false);
    hv_L  = weno3_reconstruct(hv_u,  hv_c0,  hv_c1,  hv_v,  s_u, s_c0, s_c1, s_v, true);
    hv_R  = weno3_reconstruct(hv_u,  hv_c0,  hv_c1,  hv_v,  s_u, s_c0, s_c1, s_v, false);
    }

    // Local pair monotonicity clip (same safeguard WENO5 uses).
    double eta_min = fmin(eta_c0, eta_c1);
    double eta_max = fmax(eta_c0, eta_c1);
    double hu_min  = fmin(hu_c0, hu_c1);
    double hu_max  = fmax(hu_c0, hu_c1);
    double hv_min  = fmin(hv_c0, hv_c1);
    double hv_max  = fmax(hv_c0, hv_c1);

    eta_L = fmax(eta_min, fmin(eta_max, eta_L));
    eta_R = fmax(eta_min, fmin(eta_max, eta_R));
    hu_L  = fmax(hu_min,  fmin(hu_max,  hu_L));
    hu_R  = fmax(hu_min,  fmin(hu_max,  hu_R));
    hv_L  = fmax(hv_min,  fmin(hv_max,  hv_L));
    hv_R  = fmax(hv_min,  fmin(hv_max,  hv_R));

    // Ensure reconstructed water surface is at least the local bed so that the
    // flux kernel's hydrostatic reconstruction sees non-negative depth.
    eta_L = fmax(eta_L, zb[c0]);
    eta_R = fmax(eta_R, zb[c1]);

    // Zero momentum on effectively dry reconstructed faces.
    double hL = fmax(0.0, eta_L - zb[c0]);
    double hR = fmax(0.0, eta_R - zb[c1]);
    if (hL <= h_min) { hu_L = 0.0; hv_L = 0.0; }
    if (hR <= h_min) { hu_R = 0.0; hv_R = 0.0; }

    // g is unused in component-wise reconstruction; silence unused parameter warning.
    (void)g;

    eta_face_L[f] = eta_L;
    eta_face_R[f] = eta_R;
    hu_face_L[f]  = hu_L;
    hu_face_R[f]  = hu_R;
    hv_face_L[f]  = hv_L;
    hv_face_R[f]  = hv_R;
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
    const Grad* __restrict__ d_grad,
    int n_faces,
    double* __restrict__ eta_face_L,
    double* __restrict__ eta_face_R,
    double* __restrict__ hu_face_L,
    double* __restrict__ hu_face_R,
    double* __restrict__ hv_face_L,
    double* __restrict__ hv_face_R)
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

    // Degenerate stencil check: if any two consecutive projected coordinates
    // are nearly coincident (< 1% of the c0-c1 spacing), the Lagrange
    // interpolation is ill-conditioned and the MP5 limiter can produce
    // oscillatory results.  Fall back to first-order upwind.
    const double ds_pair = fabs(s[3] - s[2]);
    bool degenerate_stencil = (ds_pair < 1.0e-14);
    if (!degenerate_stencil) {
        for (int k = 0; k < 4; ++k) {
            if (fabs(s[k + 1] - s[k]) < 0.01 * ds_pair) {
                degenerate_stencil = true;
                break;
            }
        }
    }

    if (degenerate_stencil) {
        int cL = st[2];
        int cR = st[3];
        eta_face_L[f] = eta[2]; eta_face_R[f] = eta[3];
        hu_face_L[f]  = hu_v[2]; hu_face_R[f]  = hu_v[3];
        hv_face_L[f]  = hv_v[2]; hv_face_R[f]  = hv_v[3];
        return;
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

    // ── Split centered MP5 value into L/R states ────────────────────────
    // MP5 produces a single centered value at the face.  On unstructured
    // meshes this creates identical L/R Riemann states, giving a purely
    // central flux with zero numerical dissipation through the HLLC solver.
    //
    // The split uses the GRADIENT DIFFERENCE between cL and cR to create
    // left/right asymmetry.  This avoids double-counting the extrapolation:
    // the centered fMP already encodes the mean gradient from the Lagrange
    // polynomial.  Using the absolute gradient projection (fMP + grad.dx)
    // adds a second extrapolation on top, which on triangle meshes with
    // poorly-conditioned LSQ gradients produces values at the pair-envelope
    // clamp boundary, creating a systematic one-sided flux bias that
    // amplifies into a feedback loop.
    //
    // The gradient-difference approach (fMP + 0.5*(gradL - gradR).dx) uses
    // only the curvature signal — the part of the gradient that the centered
    // Lagrange value does not already capture.  This restores upwind
    // dissipation through the HLLC wave-speed selection without overshooting.
    int cL = st[2];
    int cR = st[3];
    double dxL = mx - cell_cx[cL], dyL = my - cell_cy[cL];
    double dxR = mx - cell_cx[cR], dyR = my - cell_cy[cR];
    double dg_eta_x = d_grad[cL].hx - d_grad[cR].hx;
    double dg_eta_y = d_grad[cL].hy - d_grad[cR].hy;
    double dg_hu_x  = d_grad[cL].hux - d_grad[cR].hux;
    double dg_hu_y  = d_grad[cL].huy - d_grad[cR].huy;
    double dg_hv_x  = d_grad[cL].hvx - d_grad[cR].hvx;
    double dg_hv_y  = d_grad[cL].hvy - d_grad[cR].hvy;
    double half_gL_eta = 0.5 * (dg_eta_x * dxL + dg_eta_y * dyL);
    double half_gR_eta = 0.5 * (dg_eta_x * dxR + dg_eta_y * dyR);
    double half_gL_hu  = 0.5 * (dg_hu_x  * dxL + dg_hu_y  * dyL);
    double half_gR_hu  = 0.5 * (dg_hu_x  * dxR + dg_hu_y  * dyR);
    double half_gL_hv  = 0.5 * (dg_hv_x  * dxL + dg_hv_y  * dyL);
    double half_gR_hv  = 0.5 * (dg_hv_x  * dxR + dg_hv_y  * dyR);

    double rec_eta_L = fMP_eta + half_gL_eta;
    double rec_eta_R = fMP_eta + half_gR_eta;
    double rec_hu_L  = fMP_hu  + half_gL_hu;
    double rec_hu_R  = fMP_hu  + half_gR_hu;
    double rec_hv_L  = fMP_hv  + half_gL_hv;
    double rec_hv_R  = fMP_hv  + half_gR_hv;

    // Pair-envelope clamp (same safeguard WENO5 and WENO3 use).
    double eta_min = fmin(eta[2], eta[3]);
    double eta_max = fmax(eta[2], eta[3]);
    double hu_min  = fmin(hu_v[2], hu_v[3]);
    double hu_max  = fmax(hu_v[2], hu_v[3]);
    double hv_min  = fmin(hv_v[2], hv_v[3]);
    double hv_max  = fmax(hv_v[2], hv_v[3]);

    rec_eta_L = fmax(eta_min, fmin(eta_max, rec_eta_L));
    rec_eta_R = fmax(eta_min, fmin(eta_max, rec_eta_R));
    rec_hu_L  = fmax(hu_min,  fmin(hu_max,  rec_hu_L));
    rec_hu_R  = fmax(hu_min,  fmin(hu_max,  rec_hu_R));
    rec_hv_L  = fmax(hv_min,  fmin(hv_max,  rec_hv_L));
    rec_hv_R  = fmax(hv_min,  fmin(hv_max,  rec_hv_R));

    // Ensure reconstructed water surface is at least the local bed.
    rec_eta_L = fmax(rec_eta_L, zb[cL]);
    rec_eta_R = fmax(rec_eta_R, zb[cR]);

    eta_face_L[f] = rec_eta_L;
    eta_face_R[f] = rec_eta_R;
    hu_face_L[f]  = rec_hu_L;
    hu_face_R[f]  = rec_hu_R;
    hv_face_L[f]  = rec_hv_L;
    hv_face_R[f]  = rec_hv_R;
}
