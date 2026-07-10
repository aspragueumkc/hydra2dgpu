// swe2d_reconstruct.cu
// Spatial reconstruction kernels for advanced schemes.
// Contains: barth_jespersen_kernel (scheme 5), weno3_kernel (scheme 6),
//           mp5_kernel (scheme 8), and device helper functions.

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

/// Small residual for a set of cell values: sum of squared deviations from mean.
/// Used by weno3_kernel to estimate sub-stencil smoothness.
static __device__ double lsq2d_residual(
    const double* __restrict__ q,
    int n,
    const int* __restrict__ cell_ids)
{
    if (n <= 0) return 1e6;
    double mean = 0.0;
    for (int k = 0; k < n; ++k) {
        mean += q[cell_ids[k]];
    }
    mean /= (double)n;
    double res = 0.0;
    for (int k = 0; k < n; ++k) {
        double diff = q[cell_ids[k]] - mean;
        res += diff * diff;
    }
    return res;
}

/// Standard minmod function: returns zero if arguments have opposite sign,
/// otherwise returns the argument with the smaller magnitude.
static __device__ double minmod(double a, double b)
{
    if (a * b <= 0.0) return 0.0;
    return (fabs(a) < fabs(b)) ? a : b;
}

// ═════════════════════════════════════════════════════════════════════════════
//  Barth-Jespersen limiter  (scheme 5)
//  One thread per cell.  Applies a slope limiter using the 1-ring neighbor
//  envelope so that the extrapolated face values do not exceed the local
//  min/max of the cell and its neighbours.
// ═════════════════════════════════════════════════════════════════════════════

__global__ void barth_jespersen_kernel(
    const double* __restrict__ q,
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

    double qi  = q[i];
    double gx  = grad_x[i];
    double gy  = grad_y[i];
    double xi  = cell_cx[i];
    double yi  = cell_cy[i];

    double chi = 1.0;
    int start = cell_edge_offsets[i];
    int end   = cell_edge_offsets[i + 1];

    for (int k = start; k < end; ++k) {
        int edge = cell_edge_ids[k];
        int j = (edge_c0[edge] == i) ? edge_c1[edge] : edge_c0[edge];
        if (j < 0) continue;  // boundary edge, skip

        double qj = q[j];
        double xj = cell_cx[j];
        double yj = cell_cy[j];

        double dx = xj - xi;
        double dy = yj - yi;
        double q_face = qi + gx * dx + gy * dy;

        double q_min = fmin(qi, qj);
        double q_max = fmax(qi, qj);

        double chi_k = 1.0;
        if (q_face > q_max) {
            double denom = q_face - qi;
            if (fabs(denom) > 1e-15) {
                chi_k = (q_max - qi) / denom;
            }
        } else if (q_face < q_min) {
            double denom = qi - q_face;
            if (fabs(denom) > 1e-15) {
                chi_k = (qi - q_min) / denom;
            }
        }
        chi = fmin(chi, chi_k);
    }

    chi = fmax(0.0, fmin(1.0, chi));
    grad_x_lim[i] = chi * gx;
    grad_y_lim[i] = chi * gy;
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
//  WENO3  (scheme 6)
//  One thread per face.  Constructs three candidate face values from the
//  three WENO sub-stencils (S0=upwind lobe, S1=central pair, S2=downwind lobe),
//  then combines them via Hu-Shu (1999) nonlinear weights using sub-stencil
//  smoothness indicators.
// ═════════════════════════════════════════════════════════════════════════════

__global__ void weno3_kernel(
    const double* __restrict__ q,
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
    double* __restrict__ q_face_recon)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n_faces) return;

    // S1 = {owner, neighbour} — always 2 cells
    int i = face_stencil_S1[2 * f + 0];
    int j = face_stencil_S1[2 * f + 1];
    if (j < 0) {
        // Boundary face: no neighbour cell; use owner value directly.
        q_face_recon[f] = q[i];
        return;
    }

    double qi = q[i], qj = q[j];
    double xi = cell_cx[i], yi = cell_cy[i];
    double xj = cell_cx[j], yj = cell_cy[j];
    double xf = face_mid_x[f], yf = face_mid_y[f];

    // ── S0: upwind lobe (neighbours of owner i, excluding j) ──────────
    int s0 = face_stencil_S0_offsets[f];
    int e0 = face_stencil_S0_offsets[f + 1];
    int n0 = e0 - s0;
    double q_cand_0;
    double beta_0 = 1e6;
    if (n0 > 0) {
        double sum_q = 0.0;
        for (int k = s0; k < e0; ++k) {
            sum_q += q[face_stencil_S0_cells[k]];
        }
        q_cand_0 = sum_q / (double)n0;
        beta_0 = lsq2d_residual(q, n0, face_stencil_S0_cells + s0);
    } else {
        q_cand_0 = qi;
    }

    // ── S1: central pair — linear interpolation to face midpoint ──────
    double dist_face = sqrt((xf - xi) * (xf - xi) + (yf - yi) * (yf - yi));
    double dist_ij   = sqrt((xj - xi) * (xj - xi) + (yj - yi) * (yj - yi));
    double t = (dist_ij > 1e-12) ? (dist_face / dist_ij) : 0.5;
    double q_cand_1 = qi + t * (qj - qi);
    double beta_1 = (qi - qj) * (qi - qj);

    // ── S2: downwind lobe (neighbours of neighbour j, excluding i) ────
    int s2 = face_stencil_S2_offsets[f];
    int e2 = face_stencil_S2_offsets[f + 1];
    int n2 = e2 - s2;
    double q_cand_2;
    double beta_2 = 1e6;
    if (n2 > 0) {
        double sum_q = 0.0;
        for (int k = s2; k < e2; ++k) {
            sum_q += q[face_stencil_S2_cells[k]];
        }
        q_cand_2 = sum_q / (double)n2;
        beta_2 = lsq2d_residual(q, n2, face_stencil_S2_cells + s2);
    } else {
        q_cand_2 = qj;
    }

    // ── Nonlinear WENO3 weights (Hu-Shu 1999) ────────────────────────
    double d_weights[3] = {0.1, 0.6, 0.3};
    double eps = 1e-6;
    double alpha[3];
    double alpha_sum = 0.0;
    for (int k = 0; k < 3; ++k) {
        double beta_k;
        switch (k) {
            case 0: beta_k = beta_0; break;
            case 1: beta_k = beta_1; break;
            default: beta_k = beta_2; break;
        }
        alpha[k] = d_weights[k] / ((eps + beta_k) * (eps + beta_k));
        alpha_sum += alpha[k];
    }

    double q_recon = 0.0;
    double q_cands[3] = {q_cand_0, q_cand_1, q_cand_2};
    for (int k = 0; k < 3; ++k) {
        q_recon += (alpha[k] / alpha_sum) * q_cands[k];
    }
    q_face_recon[f] = q_recon;
}

// ═════════════════════════════════════════════════════════════════════════════
//  MP5  (scheme 8)
//  One thread per face.  Reads the 5-cell face-normal walk {u2,u1,u,v,v1},
//  computes a 4th-degree polynomial high-order face value, then applies the
//  mapped monotonicity-preserving (MP) limiter of Suresh-Huynh (1997) to
//  suppress oscillations at discontinuities.
// ═════════════════════════════════════════════════════════════════════════════

__global__ void mp5_kernel(
    const double* __restrict__ q,
    const double* __restrict__ cell_cx,
    const double* __restrict__ cell_cy,
    const double* __restrict__ face_mid_x,
    const double* __restrict__ face_mid_y,
    const int* __restrict__ face_stencil_5,
    const int* __restrict__ face_mp5_case,
    int n_faces,
    double* __restrict__ q_face_recon)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n_faces) return;

    const int* st = &face_stencil_5[5 * f];
    int u2 = st[0], u1 = st[1], u  = st[2];
    int v  = st[3], v1 = st[4];

    double fm2 = q[u2], fm1 = q[u1], f0 = q[u];
    double fp1 = q[v],  fp2 = q[v1];

    double xf = face_mid_x[f], yf = face_mid_y[f];
    double xu = cell_cx[u],    yu = cell_cy[u];
    double xv = cell_cx[v],    yv = cell_cy[v];

    double dist_uv = sqrt((xv - xu) * (xv - xu) + (yv - yu) * (yv - yu));
    double dist_uf = sqrt((xf - xu) * (xf - xu) + (yf - yu) * (yf - yu));
    double t = (dist_uv > 1e-12) ? (dist_uf / dist_uv) : 0.5;

    // 4th-degree polynomial high-order reconstruction
    // (Suresh-Huynh 1997, Eq. 3.2)
    double f4 = (1.0 / 60.0) * (
         2.0 * fm2 - 13.0 * fm1 + 47.0 * f0 + 27.0 * fp1 - 3.0 * fp2
    );

    // Linear interpolation to face midpoint
    double f_linear = f0 + t * (fp1 - f0);

    // TVD bound
    double f_tvd = f0 + 0.5 * minmod(fp1 - f0, f0 - fm1);

    // Clip envelope from the three-cell stencil {fm1, f0, fp1}
    double f_min = fmin(fmin(fm1, f0), fp1);
    double f_max = fmax(fmax(fm1, f0), fp1);

    int fcase = face_mp5_case[f];
    double f_mp5;

    switch (fcase) {
        case 1: {
            // Unconstrained: apply the high-order value, then clip
            f_mp5 = f4;
            f_mp5 = fmax(f_min, fmin(f_max, f_mp5));
            break;
        }
        case 2: {
            // Mapped compression toward f_linear
            double d_min = f_min - f_linear;
            double d_max = f_max - f_linear;
            double d4 = f4 - f_linear;
            if (d4 > 0.0) {
                f_mp5 = f_linear + fmax(d_min, d4 * 0.5);
            } else {
                f_mp5 = f_linear + fmin(d_max, d4 * 0.5);
            }
            f_mp5 = fmax(f_min, fmin(f_max, f_mp5));
            break;
        }
        case 3: {
            // Blend toward TVD bound
            f_mp5 = f_linear + 0.5 * (f_tvd - f_linear);
            break;
        }
        default: {
            // Fallback: linear is the safest
            f_mp5 = f_linear;
            break;
        }
    }

    q_face_recon[f] = f_mp5;
}
