// swe2d_gpu.cu
// CUDA kernel implementations for the 2D SWE hybrid solver.
//
// Three kernel launches per timestep:
//   1. swe2d_flux_kernel   — parallel over edges, writes flux accumulators
//   2. swe2d_update_kernel — parallel over cells, applies fluxes + friction
//   3. swe2d_cfl_kernel    — parallel over cells, block-reduce to find max lambda
//
// All numerical logic is shared with the CPU path via swe2d_numerics.hpp
// (the SWE2D_HOSTDEV macro marks kernels callable from both host and device).

#include "swe2d_gpu.cuh"
#include "swe2d_numerics.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace cg = cooperative_groups;

namespace {

bool swe2d_debug_enabled(const char* name) {
    const char* v = std::getenv(name);
    return (v && v[0] && v[0] != '0');
}

void dump_flux_summary(const char* tag,
                       const std::vector<double>& dh,
                       const std::vector<double>& dhu,
                       const std::vector<double>& dhv)
{
    if (dh.empty()) return;
    double s_h = 0.0, s_hu = 0.0, s_hv = 0.0;
    double m_h = 0.0, m_hu = 0.0, m_hv = 0.0;
    for (size_t i = 0; i < dh.size(); ++i) {
        const double ah = std::abs(dh[i]);
        const double au = std::abs(dhu[i]);
        const double av = std::abs(dhv[i]);
        s_h += dh[i];
        s_hu += dhu[i];
        s_hv += dhv[i];
        if (ah > m_h) m_h = ah;
        if (au > m_hu) m_hu = au;
        if (av > m_hv) m_hv = av;
    }
    std::fprintf(stderr,
                 "[SWE2D_DEBUG] %s flux summary: sum(dh)=%.9e sum(dhu)=%.9e sum(dhv)=%.9e max|dh|=%.9e max|dhu|=%.9e max|dhv|=%.9e\n",
                 tag, s_h, s_hu, s_hv, m_h, m_hu, m_hv);
    const size_t n_show = std::min<size_t>(dh.size(), 8);
    for (size_t i = 0; i < n_show; ++i) {
        std::fprintf(stderr,
                     "[SWE2D_DEBUG] %s flux cell[%zu]: dh=%.9e dhu=%.9e dhv=%.9e\n",
                     tag, i, dh[i], dhu[i], dhv[i]);
    }
}

__device__ inline double atomicMaxDouble(double* address, double val) {
    unsigned long long int* address_as_ull = reinterpret_cast<unsigned long long int*>(address);
    unsigned long long int old = *address_as_ull;

    while (true) {
        double old_val = __longlong_as_double(static_cast<long long int>(old));
        if (old_val >= val) {
            return old_val;
        }
        unsigned long long int assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        static_cast<unsigned long long int>(__double_as_longlong(val)));
        if (old == assumed) {
            return val;
        }
    }
}

__device__ inline void hllc_flux_cuda_local(
    double hL, double uL, double vL,
    double hR, double uR, double vR,
    double nx, double ny,
    double g, double h_min,
    double& fh, double& fhu, double& fhv)
{
    fh = 0.0;
    fhu = 0.0;
    fhv = 0.0;

    if (hL <= h_min && hR <= h_min) {
        return;
    }

    const double unL = uL * nx + vL * ny;
    const double unR = uR * nx + vR * ny;

    const double cL = (hL > 0.0) ? ::sqrt(g * hL) : 0.0;
    const double cR = (hR > 0.0) ? ::sqrt(g * hR) : 0.0;

    const double sqrt_hL = (hL > 0.0) ? ::sqrt(hL) : 0.0;
    const double sqrt_hR = (hR > 0.0) ? ::sqrt(hR) : 0.0;
    const double denom = sqrt_hL + sqrt_hR;

    const double u_roe = (denom > 0.0)
                       ? (sqrt_hL * unL + sqrt_hR * unR) / denom
                       : 0.0;
    const double c_roe = (denom > 0.0)
                       ? ::sqrt(0.5 * g * (hL + hR))
                       : 0.0;

    const double SL = ::fmin(unL - cL, u_roe - c_roe);
    const double SR = ::fmax(unR + cR, u_roe + c_roe);

    const double fhL  = hL * unL;
    const double fhuL = hL * uL * unL + 0.5 * g * hL * hL * nx;
    const double fhvL = hL * vL * unL + 0.5 * g * hL * hL * ny;

    const double fhR  = hR * unR;
    const double fhuR = hR * uR * unR + 0.5 * g * hR * hR * nx;
    const double fhvR = hR * vR * unR + 0.5 * g * hR * hR * ny;

    if (SL >= 0.0) {
        fh = fhL;
        fhu = fhuL;
        fhv = fhvL;
        return;
    }
    if (SR <= 0.0) {
        fh = fhR;
        fhu = fhuR;
        fhv = fhvR;
        return;
    }

    const double numS = hR * unR * (SR - unR) - hL * unL * (SL - unL)
                      + 0.5 * g * (hL * hL - hR * hR);
    const double denS = hR * (SR - unR) - hL * (SL - unL);
    const double S_star = (::fabs(denS) > 1.0e-15) ? (numS / denS) : 0.0;

    if (S_star >= 0.0) {
        const double coeff = hL * (SL - unL) / (SL - S_star);
        const double h_star_L  = coeff;
        const double hu_star_L = coeff * (uL + (S_star - unL) * nx);
        const double hv_star_L = coeff * (vL + (S_star - unL) * ny);

        const double dh  = h_star_L  - hL;
        const double dhu = hu_star_L - hL * uL;
        const double dhv = hv_star_L - hL * vL;

        fh  = fhL  + SL * dh;
        fhu = fhuL + SL * dhu;
        fhv = fhvL + SL * dhv;
    } else {
        const double coeff = hR * (SR - unR) / (SR - S_star);
        const double h_star_R  = coeff;
        const double hu_star_R = coeff * (uR + (S_star - unR) * nx);
        const double hv_star_R = coeff * (vR + (S_star - unR) * ny);

        const double dh  = h_star_R  - hR;
        const double dhu = hu_star_R - hR * uR;
        const double dhv = hv_star_R - hR * vR;

        fh  = fhR  + SR * dh;
        fhu = fhuR + SR * dhu;
        fhv = fhvR + SR * dhv;
    }
}

} // namespace

// ─────────────────────────────────────────────────────────────────────────────
// CUDA error checking
// ─────────────────────────────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            throw std::runtime_error(std::string("CUDA error: ")               \
                + cudaGetErrorString(_e) + " at " __FILE__ ":"                 \
                + std::to_string(__LINE__));                                    \
        }                                                                       \
    } while (0)

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 0 (optional): Green-Gauss gradient estimation — one thread per edge.
// Accumulates face-average * outward-normal * len / area into per-cell gradient
// arrays. Must be run with zeroed gradient arrays before the flux kernel when
// the scheme is FV_MUSCL_MC (3) or FV_MUSCL_VAN_LEER (4).
// ─────────────────────────────────────────────────────────────────────────────
__global__ void swe2d_gradient_kernel(
    int32_t        n_edges,
    const int32_t* edge_c0,
    const int32_t* edge_c1,
    const double*  edge_nx,
    const double*  edge_ny,
    const double*  edge_len,
    const double*  cell_h,
    const double*  cell_hu,
    const double*  cell_hv,
    const double*  cell_inv_area,
    double*        grad_hx,  double* grad_hy,
    double*        grad_hux, double* grad_huy,
    double*        grad_hvx, double* grad_hvy)
{
    int32_t e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_edges) return;

    int32_t c0 = edge_c0[e];
    int32_t c1 = edge_c1[e];

    // Skip degenerate cells
    if (cell_inv_area[c0] > 1.0e6) return;

    double nx  = edge_nx[e];
    double ny  = edge_ny[e];
    double len = edge_len[e];

    double h0  = cell_h[c0],  hu0 = cell_hu[c0], hv0 = cell_hv[c0];
    double h1, hu1, hv1;
    if (c1 >= 0 && cell_inv_area[c1] <= 1.0e6) {
        h1 = cell_h[c1]; hu1 = cell_hu[c1]; hv1 = cell_hv[c1];
    } else {
        // Boundary: use cell value (zero-gradient extrapolation for gradient)
        h1 = h0; hu1 = hu0; hv1 = hv0;
    }

    // Green-Gauss: contribution = q_face * n * len * inv_area
    double ia0 = cell_inv_area[c0];
    double qh  = 0.5 * (h0  + h1);
    double qhu = 0.5 * (hu0 + hu1);
    double qhv = 0.5 * (hv0 + hv1);

    atomicAdd(&grad_hx[c0],   qh  * nx * len * ia0);
    atomicAdd(&grad_hy[c0],   qh  * ny * len * ia0);
    atomicAdd(&grad_hux[c0],  qhu * nx * len * ia0);
    atomicAdd(&grad_huy[c0],  qhu * ny * len * ia0);
    atomicAdd(&grad_hvx[c0],  qhv * nx * len * ia0);
    atomicAdd(&grad_hvy[c0],  qhv * ny * len * ia0);

    // Contribution to c1: outward normal for c1 is the reverse of c0's normal
    if (c1 >= 0 && cell_inv_area[c1] <= 1.0e6) {
        double ia1 = cell_inv_area[c1];
        atomicAdd(&grad_hx[c1],  -qh  * nx * len * ia1);
        atomicAdd(&grad_hy[c1],  -qh  * ny * len * ia1);
        atomicAdd(&grad_hux[c1], -qhu * nx * len * ia1);
        atomicAdd(&grad_huy[c1], -qhu * ny * len * ia1);
        atomicAdd(&grad_hvx[c1], -qhv * nx * len * ia1);
        atomicAdd(&grad_hvy[c1], -qhv * ny * len * ia1);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1: Flux computation — one thread per edge
// Writes atomic increments into flux accumulators.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void swe2d_flux_kernel(
    int32_t        n_edges,
    const int32_t* edge_c0,
    const int32_t* edge_c1,
    const double*  edge_nx,
    const double*  edge_ny,
    const double*  edge_len,
    const int32_t* edge_bc,
    const double*  edge_bc_val,
    const double*  cell_h,
    const double*  cell_hu,
    const double*  cell_hv,
    const double*  cell_zb,
    const double*  cell_inv_area,
    // Cell centroids and gradients (used for MC and Van Leer limiters)
    const double*  cell_cx,
    const double*  cell_cy,
    const double*  grad_hx,  const double* grad_hy,
    const double*  grad_hux, const double* grad_huy,
    const double*  grad_hvx, const double* grad_hvy,
    double*        flux_h,    // [n_cells] accumulator
    double*        flux_hu,
    double*        flux_hv,
    double*        dbg_fh,
    double*        dbg_fhu,
    double*        dbg_fhv,
    int            spatial_scheme,
    double g, double h_min)
{
    int32_t e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_edges) return;

    int32_t c0 = edge_c0[e];
    int32_t c1 = edge_c1[e];
    
    // Skip flux TO degenerate cells (inv_area > 1e6)
    if (cell_inv_area[c0] > 1.0e6 || (c1 >= 0 && cell_inv_area[c1] > 1.0e6)) {
        return;
    }
    
    double  nx  = edge_nx[e];
    double  ny  = edge_ny[e];
    double  len = edge_len[e];

    double hL  = cell_h[c0],  huL = cell_hu[c0], hvL = cell_hv[c0];
    double zbL = cell_zb[c0];

    double hR, huR, hvR, zbR;
    if (c1 >= 0) {
        hR  = cell_h[c1]; huR = cell_hu[c1]; hvR = cell_hv[c1];
        zbR = cell_zb[c1];

        // GPU-only selectable higher-order reconstruction modes.
        // All schemes 1–4 use Green-Gauss gradient-based TVD reconstruction.
        // The pair-only midpoint approach (coefficient 0.5) was removed because it
        // produces identical face states on both sides of every edge, which cancels
        // the HLLC solver's upwind dissipation and causes neutral-to-unstable
        // behaviour on non-trivial meshes regardless of unit system.
        //
        // Limiter table:
        //   FV_MUSCL_FAST     (1) — Superbee:  most aggressive TVD, sharpest fronts
        //   FV_MUSCL_MINMOD   (2) — MinMod:    most conservative TVD, most stable
        //   FV_MUSCL_MC       (3) — MC:        balanced monotonized-central
        //   FV_MUSCL_VAN_LEER (4) — Van Leer:  smooth limiter, phi→2 as r→∞
        const int scheme_fast   = static_cast<int>(SWE2DSpatialScheme::FV_MUSCL_FAST);
        const int scheme_robust = static_cast<int>(SWE2DSpatialScheme::FV_MUSCL_MINMOD);
        const int scheme_mc     = static_cast<int>(SWE2DSpatialScheme::FV_MUSCL_MC);
        const int scheme_vl     = static_cast<int>(SWE2DSpatialScheme::FV_MUSCL_VAN_LEER);
        if (spatial_scheme >= scheme_fast && cell_cx != nullptr && grad_hx != nullptr) {
            const double dcx = cell_cx[c1] - cell_cx[c0];
            const double dcy = cell_cy[c1] - cell_cy[c0];
            constexpr double EPS = 1.0e-30;

            // Helper lambda: compute TVD limiter phi(r) and reconstruct one variable.
            // q0, q1 — cell-centre values; gx0/gy0, gx1/gy1 — GG gradient at c0, c1.
            auto tvd_reconstruct = [&](double q0, double q1,
                                       double gx0, double gy0,
                                       double gx1, double gy1,
                                       double& qL_out, double& qR_out) {
                const double dq = q1 - q0;   // downwind difference (c0→c1)

                // Slope ratio at c0: GG gradient projected onto c0→c1 / pair jump
                const double s0 = gx0 * dcx + gy0 * dcy;
                const double sign_dq = (dq >= 0.0) ? 1.0 : -1.0;
                const double r0 = s0 / (dq + sign_dq * EPS);

                // Slope ratio at c1 (looking back toward c0)
                const double s1 = -(gx1 * dcx + gy1 * dcy);
                const double r1 = s1 / (-dq + (-sign_dq) * EPS);

                double phi0, phi1;
                if (spatial_scheme == scheme_fast) {
                    // Superbee: most liberal TVD limiter (sharpest)
                    phi0 = fmax(0.0, fmax(fmin(2.0 * r0, 1.0), fmin(r0, 2.0)));
                    phi1 = fmax(0.0, fmax(fmin(2.0 * r1, 1.0), fmin(r1, 2.0)));
                } else if (spatial_scheme == scheme_robust) {
                    // MinMod: most conservative TVD limiter (most stable)
                    phi0 = fmax(0.0, fmin(r0, 1.0));
                    phi1 = fmax(0.0, fmin(r1, 1.0));
                } else if (spatial_scheme == scheme_mc) {
                    // MC (monotonized central): balanced
                    phi0 = fmax(0.0, fmin(fmin(2.0 * r0, 0.5 * (1.0 + r0)), 2.0));
                    phi1 = fmax(0.0, fmin(fmin(2.0 * r1, 0.5 * (1.0 + r1)), 2.0));
                } else {
                    // Van Leer: smooth, phi → 2 as r → ∞
                    phi0 = (r0 + fabs(r0)) / (1.0 + fabs(r0));
                    phi1 = (r1 + fabs(r1)) / (1.0 + fabs(r1));
                }

                qL_out = q0 + phi0 * 0.5 * dq;
                qR_out = q1 - phi1 * 0.5 * dq;
            };

            double hL_rec, hR_rec, huL_rec, huR_rec, hvL_rec, hvR_rec;
            tvd_reconstruct(hL,  hR,  grad_hx[c0],  grad_hy[c0],  grad_hx[c1],  grad_hy[c1],  hL_rec,  hR_rec);
            tvd_reconstruct(huL, huR, grad_hux[c0], grad_huy[c0], grad_hux[c1], grad_huy[c1], huL_rec, huR_rec);
            tvd_reconstruct(hvL, hvR, grad_hvx[c0], grad_hvy[c0], grad_hvx[c1], grad_hvy[c1], hvL_rec, hvR_rec);

            hL  = (hL_rec  > 0.0) ? hL_rec  : 0.0;
            hR  = (hR_rec  > 0.0) ? hR_rec  : 0.0;
            huL = huL_rec; huR = huR_rec;
            hvL = hvL_rec; hvR = hvR_rec;
        }

        // Keep reconstructed momentum physically bounded for shallow cells.
        const double hL_eff = (hL > h_min) ? hL : h_min;
        const double hR_eff = (hR > h_min) ? hR : h_min;
        const double u_cap_L = fmax(50.0, 20.0 * sqrt(g * hL_eff));
        const double u_cap_R = fmax(50.0, 20.0 * sqrt(g * hR_eff));
        const double hu_cap_L = hL_eff * u_cap_L;
        const double hv_cap_L = hL_eff * u_cap_L;
        const double hu_cap_R = hR_eff * u_cap_R;
        const double hv_cap_R = hR_eff * u_cap_R;
        huL = fmin(hu_cap_L, fmax(-hu_cap_L, huL));
        hvL = fmin(hv_cap_L, fmax(-hv_cap_L, hvL));
        huR = fmin(hu_cap_R, fmax(-hu_cap_R, huR));
        hvR = fmin(hv_cap_R, fmax(-hv_cap_R, hvR));
    } else {
        swe2d::GhostState gs = swe2d::make_ghost(
            hL, huL, hvL, zbL, nx, ny,
            edge_bc[e], edge_bc_val[e], h_min);
        hR  = gs.h; huR = gs.hu; hvR = gs.hv;
        zbR = gs.zb;
    }

// Reconstruct hydrostatic states then apply a CUDA-local HLLC flux.
    // This avoids relying on host/device overload resolution from the shared
    // header in this kernel hot-path.
    swe2d::ReconstructedStates rs = swe2d::hydrostatic_reconstruct(
        hL, huL, hvL, zbL, hR, huR, hvR, zbR, h_min);

    double flux_fh = 0.0, flux_fhu = 0.0, flux_fhv = 0.0;
    hllc_flux_cuda_local(
        rs.hL_star, rs.uL, rs.vL,
        rs.hR_star, rs.uR, rs.vR,
        nx, ny, g, h_min,
        flux_fh, flux_fhu, flux_fhv);

    double corr_hu = 0.0, corr_hv = 0.0;
    swe2d::bed_slope_correction(hL, rs.hL_star, nx, ny, g, corr_hu, corr_hv);

    double fh  = flux_fh  * len;
    double fhu = (flux_fhu + corr_hu) * len;
    double fhv = (flux_fhv + corr_hv) * len;

    if (dbg_fh) {
        dbg_fh[e] = fh;
        dbg_fhu[e] = fhu;
        dbg_fhv[e] = fhv;
    }

    // c0: flux leaves
    atomicAdd(&flux_h[c0],  -fh);
    atomicAdd(&flux_hu[c0], -fhu);
    atomicAdd(&flux_hv[c0], -fhv);

    if (c1 >= 0) {
        // c1: flux enters, with right-side bed-slope correction
        double corr_hu_r = 0.0, corr_hv_r = 0.0;
        // Same normal direction as c0 — see swe2d_solver.cpp for derivation.
        swe2d::bed_slope_correction(hR, rs.hR_star, nx, ny, g, corr_hu_r, corr_hv_r);
        double fhu_r = (flux_fhu + corr_hu_r) * len;
        double fhv_r = (flux_fhv + corr_hv_r) * len;
        atomicAdd(&flux_h[c1],  fh);
        atomicAdd(&flux_hu[c1], fhu_r);
        atomicAdd(&flux_hv[c1], fhv_r);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2: State update — one thread per cell
// ─────────────────────────────────────────────────────────────────────────────
__global__ void swe2d_update_kernel(
    int32_t       n_cells,
    double*       cell_h,
    double*       cell_hu,
    double*       cell_hv,
    const double* flux_h,
    const double* flux_hu,
    const double* flux_hv,
    const double* cell_inv_area,
    const double* cell_n_mann,
    double*       d_max_wse_elev_error,
    double dt, double g, double h_min)
{
    int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells) return;

    const double h_old = cell_h[c];

    double inv_a = cell_inv_area[c];

    // Clamp inv_area to prevent overflow: bound state update by max_inv_a * |flux|
    const double max_inv_a = 1.0e6;
    if (inv_a > max_inv_a) inv_a = max_inv_a;

    double fh = flux_h[c];
    double fhu = flux_hu[c];
    double fhv = flux_hv[c];
    if (!isfinite(fh)) fh = 0.0;
    if (!isfinite(fhu)) fhu = 0.0;
    if (!isfinite(fhv)) fhv = 0.0;

    cell_h[c]  += dt * fh  * inv_a;
    cell_hu[c] += dt * fhu * inv_a;
    cell_hv[c] += dt * fhv * inv_a;

    // Positivity enforcement
    if (cell_h[c] < 0.0) cell_h[c] = 0.0;
    if (cell_h[c] < h_min) {
        cell_hu[c] = 0.0;
        cell_hv[c] = 0.0;
    }

    // Manning friction (semi-implicit)
    double n_mann = cell_n_mann[c];
    swe2d::apply_friction(cell_h[c], cell_hu[c], cell_hv[c],
                          dt, n_mann, g, h_min);

    // Robustness guard: remove non-finite states and cap extreme momentum.
    if (!isfinite(cell_h[c]) || !isfinite(cell_hu[c]) || !isfinite(cell_hv[c])) {
        cell_h[c] = 0.0;
        cell_hu[c] = 0.0;
        cell_hv[c] = 0.0;
    } else if (cell_h[c] > h_min) {
        const double inv_h = 1.0 / cell_h[c];
        const double u = cell_hu[c] * inv_h;
        const double v = cell_hv[c] * inv_h;
        const double spd = sqrt(u * u + v * v);
        const double spd_cap = fmax(50.0, 20.0 * sqrt(g * cell_h[c]));
        if (isfinite(spd) && spd > spd_cap && spd > 0.0) {
            const double scale = spd_cap / spd;
            cell_hu[c] *= scale;
            cell_hv[c] *= scale;
        }
    }

    if (d_max_wse_elev_error) {
        const double wse_err = fabs(cell_h[c] - h_old);
        atomicMaxDouble(d_max_wse_elev_error, wse_err);
    }
}

__global__ void swe2d_rk2_combine_kernel(
    int32_t n_cells,
    double* cell_h,
    double* cell_hu,
    double* cell_hv,
    const double* h0,
    const double* hu0,
    const double* hv0,
    double* d_max_wse_elev_error,
    double h_min)
{
    int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells) return;

    const double h_new = 0.5 * (h0[c] + cell_h[c]);
    const double hu_new = 0.5 * (hu0[c] + cell_hu[c]);
    const double hv_new = 0.5 * (hv0[c] + cell_hv[c]);

    const double h_final = (h_new < 0.0) ? 0.0 : h_new;
    cell_h[c] = h_final;

    if (h_final < h_min) {
        cell_hu[c] = 0.0;
        cell_hv[c] = 0.0;
    } else {
        cell_hu[c] = hu_new;
        cell_hv[c] = hv_new;
    }

    if (d_max_wse_elev_error) {
        const double depth_res = fabs(h_final - h0[c]);
        atomicMaxDouble(d_max_wse_elev_error, depth_res);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 3: CFL reduction — one thread per cell, block-level max, then global
// ─────────────────────────────────────────────────────────────────────────────
__global__ void swe2d_cfl_kernel(
    int32_t       n_cells,
    const double* cell_h,
    const double* cell_hu,
    const double* cell_hv,
    const double* cell_area,
    double g, double h_min,
    double* d_lambda_max)
{
    extern __shared__ double sdata[];
    int32_t tid = threadIdx.x;
    int32_t c   = blockIdx.x * blockDim.x + tid;

    double lambda = 0.0;
    if (c < n_cells) {
        double h  = cell_h[c];
        double hu = cell_hu[c];
        double hv = cell_hv[c];
        double a  = cell_area[c];
        // Approximate cell size as sqrt(area)
        double dx = (a > 0.0) ? sqrt(a) : 1.0;
        double u  = swe2d::vel_u(hu, h, h_min);
        double v  = swe2d::vel_v(hv, h, h_min);
        double spd = sqrt(u * u + v * v) + swe2d::celerity(h, g);
        if (!isfinite(spd) || !isfinite(dx) || dx <= 0.0) {
            lambda = 0.0;
        } else {
            lambda = spd / dx;
            if (!isfinite(lambda)) lambda = 0.0;
            if (lambda > 1.0e6) lambda = 1.0e6;
        }
    }

    sdata[tid] = lambda;
    __syncthreads();

    // Block-level reduction (power-of-2 stride)
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (sdata[tid + stride] > sdata[tid])
                sdata[tid] = sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        // Atomic max via double (CUDA does not have native atomicMax for double;
        // use CAS loop)
        unsigned long long int* addr =
            reinterpret_cast<unsigned long long int*>(d_lambda_max);
        unsigned long long int old_bits = *addr;
        while (true) {
            double old_val = __longlong_as_double(static_cast<long long int>(old_bits));
            if (sdata[0] <= old_val) break;
            unsigned long long int new_bits = static_cast<unsigned long long int>(
                __double_as_longlong(sdata[0]));
            unsigned long long int assumed = old_bits;
            old_bits = atomicCAS(addr, assumed, new_bits);
            if (old_bits == assumed) break;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// swe2d_gpu_available
// ─────────────────────────────────────────────────────────────────────────────
bool swe2d_gpu_available() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// swe2d_gpu_init
// ─────────────────────────────────────────────────────────────────────────────
SWE2DDeviceState* swe2d_gpu_init(
    const SWE2DMesh& mesh,
    const double*    h0,
    const double*    hu0,
    const double*    hv0,
    const double*    n_mann_cell)
{
    auto* dev = new SWE2DDeviceState();
    dev->n_cells = mesh.n_cells;
    dev->n_edges = mesh.n_edges;

    size_t sz_cells = static_cast<size_t>(mesh.n_cells);
    size_t sz_edges = static_cast<size_t>(mesh.n_edges);

    // Helper lambdas for allocation + copy
    auto alloc_d = [](void** ptr, size_t bytes) {
        CUDA_CHECK(cudaMalloc(ptr, bytes));
    };
    auto copy_h2d_i = [](int32_t* dst, const int32_t* src, size_t n) {
        CUDA_CHECK(cudaMemcpy(dst, src, n * sizeof(int32_t), cudaMemcpyHostToDevice));
    };
    auto copy_h2d_d = [](double* dst, const double* src, size_t n) {
        CUDA_CHECK(cudaMemcpy(dst, src, n * sizeof(double), cudaMemcpyHostToDevice));
    };

    // Edge topology
    alloc_d(reinterpret_cast<void**>(&dev->d_edge_c0),     sz_edges * sizeof(int32_t));
    alloc_d(reinterpret_cast<void**>(&dev->d_edge_c1),     sz_edges * sizeof(int32_t));
    alloc_d(reinterpret_cast<void**>(&dev->d_edge_n0),     sz_edges * sizeof(int32_t));
    alloc_d(reinterpret_cast<void**>(&dev->d_edge_n1),     sz_edges * sizeof(int32_t));
    alloc_d(reinterpret_cast<void**>(&dev->d_edge_nx),     sz_edges * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_edge_ny),     sz_edges * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_edge_len),    sz_edges * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_edge_bc),     sz_edges * sizeof(int32_t));
    alloc_d(reinterpret_cast<void**>(&dev->d_edge_bc_val), sz_edges * sizeof(double));

    copy_h2d_i(dev->d_edge_c0, mesh.edge_c0.data(), sz_edges);
    copy_h2d_i(dev->d_edge_c1, mesh.edge_c1.data(), sz_edges);
    copy_h2d_i(dev->d_edge_n0, mesh.edge_n0.data(), sz_edges);
    copy_h2d_i(dev->d_edge_n1, mesh.edge_n1.data(), sz_edges);
    copy_h2d_d(dev->d_edge_nx,  mesh.edge_nx.data(),  sz_edges);
    copy_h2d_d(dev->d_edge_ny,  mesh.edge_ny.data(),  sz_edges);
    copy_h2d_d(dev->d_edge_len, mesh.edge_len.data(), sz_edges);
    // BCType → int32_t
    {
        std::vector<int32_t> bc_int(sz_edges);
        for (size_t i = 0; i < sz_edges; ++i)
            bc_int[i] = static_cast<int32_t>(mesh.edge_bc[i]);
        copy_h2d_i(dev->d_edge_bc, bc_int.data(), sz_edges);
    }
    copy_h2d_d(dev->d_edge_bc_val, mesh.edge_bc_val.data(), sz_edges);

    // Cell geometry
    alloc_d(reinterpret_cast<void**>(&dev->d_cell_zb),      sz_cells * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_cell_area),    sz_cells * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_cell_inv_area),sz_cells * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_n_mann_cell),  sz_cells * sizeof(double));
    copy_h2d_d(dev->d_cell_zb,       mesh.cell_zb.data(),       sz_cells);
    copy_h2d_d(dev->d_cell_area,     mesh.cell_area.data(),     sz_cells);
    copy_h2d_d(dev->d_cell_inv_area, mesh.cell_inv_area.data(), sz_cells);
    copy_h2d_d(dev->d_n_mann_cell,   n_mann_cell,               sz_cells);

    // Cell centroids (for gradient-based higher-order reconstruction)
    alloc_d(reinterpret_cast<void**>(&dev->d_cell_cx), sz_cells * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_cell_cy), sz_cells * sizeof(double));
    copy_h2d_d(dev->d_cell_cx, mesh.cell_cx.data(), sz_cells);
    copy_h2d_d(dev->d_cell_cy, mesh.cell_cy.data(), sz_cells);

    // Gradient arrays (zeroed; filled by swe2d_gradient_kernel each step for MC/VL)
    alloc_d(reinterpret_cast<void**>(&dev->d_grad_hx),  sz_cells * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_grad_hy),  sz_cells * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_grad_hux), sz_cells * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_grad_huy), sz_cells * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_grad_hvx), sz_cells * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_grad_hvy), sz_cells * sizeof(double));
    CUDA_CHECK(cudaMemset(dev->d_grad_hx,  0, sz_cells * sizeof(double)));
    CUDA_CHECK(cudaMemset(dev->d_grad_hy,  0, sz_cells * sizeof(double)));
    CUDA_CHECK(cudaMemset(dev->d_grad_hux, 0, sz_cells * sizeof(double)));
    CUDA_CHECK(cudaMemset(dev->d_grad_huy, 0, sz_cells * sizeof(double)));
    CUDA_CHECK(cudaMemset(dev->d_grad_hvx, 0, sz_cells * sizeof(double)));
    CUDA_CHECK(cudaMemset(dev->d_grad_hvy, 0, sz_cells * sizeof(double)));

    // State
    alloc_d(reinterpret_cast<void**>(&dev->d_h),  sz_cells * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_hu), sz_cells * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_hv), sz_cells * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_h0),  sz_cells * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_hu0), sz_cells * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_hv0), sz_cells * sizeof(double));
    copy_h2d_d(dev->d_h,  h0,                 sz_cells);
    copy_h2d_d(dev->d_hu, hu0 ? hu0 : h0,     sz_cells);  // reuse pointer; zeroed if same
    if (!hu0) CUDA_CHECK(cudaMemset(dev->d_hu, 0, sz_cells * sizeof(double)));
    copy_h2d_d(dev->d_hv, hv0 ? hv0 : h0,     sz_cells);
    if (!hv0) CUDA_CHECK(cudaMemset(dev->d_hv, 0, sz_cells * sizeof(double)));

    // Flux accumulators
    alloc_d(reinterpret_cast<void**>(&dev->d_flux_h),  sz_cells * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_flux_hu), sz_cells * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_flux_hv), sz_cells * sizeof(double));

    // CFL workspace
    alloc_d(reinterpret_cast<void**>(&dev->d_lambda_max), sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_max_wse_elev_error), sizeof(double));

    return dev;
}

// ─────────────────────────────────────────────────────────────────────────────
// swe2d_gpu_step
// ─────────────────────────────────────────────────────────────────────────────
void swe2d_gpu_step(
    SWE2DDeviceState* dev,
    double dt,
    double g,
    double h_min,
    int spatial_scheme,
    double /*cfl_factor*/,
    SWE2DStepDiag* diag)
{
    constexpr int BLOCK = 256;
    int32_t n_edges = dev->n_edges;
    int32_t n_cells = dev->n_cells;

    if (swe2d_debug_enabled("BACKWATER_SWE2D_DEBUG_GPU_INPUT")) {
        std::fprintf(stderr, "[SWE2D_DEBUG] GPU input: n_cells=%d n_edges=%d dt=%.9e g=%.9e h_min=%.9e\n",
                     static_cast<int>(n_cells), static_cast<int>(n_edges), dt, g, h_min);
        const size_t e_show = std::min<size_t>(static_cast<size_t>(n_edges), 8);
        std::vector<int32_t> e_c0(e_show), e_c1(e_show);
        std::vector<double> e_nx(e_show), e_ny(e_show), e_len(e_show);
        if (e_show > 0) {
            CUDA_CHECK(cudaMemcpy(e_c0.data(), dev->d_edge_c0, e_show * sizeof(int32_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(e_c1.data(), dev->d_edge_c1, e_show * sizeof(int32_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(e_nx.data(), dev->d_edge_nx, e_show * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(e_ny.data(), dev->d_edge_ny, e_show * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(e_len.data(), dev->d_edge_len, e_show * sizeof(double), cudaMemcpyDeviceToHost));
        }
        for (size_t i = 0; i < e_show; ++i) {
            std::fprintf(stderr,
                         "[SWE2D_DEBUG] GPU edge[%zu]: c0=%d c1=%d nx=%.9e ny=%.9e len=%.9e\n",
                         i,
                         static_cast<int>(e_c0[i]),
                         static_cast<int>(e_c1[i]),
                         e_nx[i], e_ny[i], e_len[i]);
        }

        const size_t c_show = std::min<size_t>(static_cast<size_t>(n_cells), 8);
        std::vector<double> h_in(c_show), hu_in(c_show), hv_in(c_show), zb_in(c_show);
        if (c_show > 0) {
            CUDA_CHECK(cudaMemcpy(h_in.data(), dev->d_h, c_show * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hu_in.data(), dev->d_hu, c_show * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hv_in.data(), dev->d_hv, c_show * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(zb_in.data(), dev->d_cell_zb, c_show * sizeof(double), cudaMemcpyDeviceToHost));
        }
        for (size_t i = 0; i < c_show; ++i) {
            std::fprintf(stderr,
                         "[SWE2D_DEBUG] GPU state[%zu]: h=%.9e hu=%.9e hv=%.9e zb=%.9e\n",
                         i, h_in[i], hu_in[i], hv_in[i], zb_in[i]);
        }
    }

    // Zero flux accumulators
    CUDA_CHECK(cudaMemset(dev->d_flux_h,  0, static_cast<size_t>(n_cells) * sizeof(double)));
    CUDA_CHECK(cudaMemset(dev->d_flux_hu, 0, static_cast<size_t>(n_cells) * sizeof(double)));
    CUDA_CHECK(cudaMemset(dev->d_flux_hv, 0, static_cast<size_t>(n_cells) * sizeof(double)));

    // Kernel 0 (gradient pre-pass): required for all 2nd-order schemes (1–4).
    // Schemes 1 & 2 also need per-cell gradients so that each side of a face
    // gets an independently-estimated slope → face states differ → the HLLC
    // retains its upwind dissipation (pair-only midpoint reconstruction gave
    // equal face states and a neutrally-stable central flux).
    const bool need_gradients = (spatial_scheme >= 1);
    if (need_gradients) {
        const size_t sz_c = static_cast<size_t>(n_cells) * sizeof(double);
        CUDA_CHECK(cudaMemset(dev->d_grad_hx,  0, sz_c));
        CUDA_CHECK(cudaMemset(dev->d_grad_hy,  0, sz_c));
        CUDA_CHECK(cudaMemset(dev->d_grad_hux, 0, sz_c));
        CUDA_CHECK(cudaMemset(dev->d_grad_huy, 0, sz_c));
        CUDA_CHECK(cudaMemset(dev->d_grad_hvx, 0, sz_c));
        CUDA_CHECK(cudaMemset(dev->d_grad_hvy, 0, sz_c));
        int g_grid = (n_edges + BLOCK - 1) / BLOCK;
        swe2d_gradient_kernel<<<g_grid, BLOCK>>>(
            n_edges,
            dev->d_edge_c0, dev->d_edge_c1,
            dev->d_edge_nx, dev->d_edge_ny, dev->d_edge_len,
            dev->d_h, dev->d_hu, dev->d_hv,
            dev->d_cell_inv_area,
            dev->d_grad_hx,  dev->d_grad_hy,
            dev->d_grad_hux, dev->d_grad_huy,
            dev->d_grad_hvx, dev->d_grad_hvy);
        CUDA_CHECK(cudaGetLastError());
    }

    // Kernel 1: Flux
    {
        const bool dbg_edge_flux = swe2d_debug_enabled("BACKWATER_SWE2D_DEBUG_GPU_EDGE_FLUX");
        double* d_dbg_fh = nullptr;
        double* d_dbg_fhu = nullptr;
        double* d_dbg_fhv = nullptr;
        if (dbg_edge_flux) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_dbg_fh),
                                  static_cast<size_t>(n_edges) * sizeof(double)));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_dbg_fhu),
                                  static_cast<size_t>(n_edges) * sizeof(double)));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_dbg_fhv),
                                  static_cast<size_t>(n_edges) * sizeof(double)));
            CUDA_CHECK(cudaMemset(d_dbg_fh, 0xA5, static_cast<size_t>(n_edges) * sizeof(double)));
            CUDA_CHECK(cudaMemset(d_dbg_fhu, 0xA5, static_cast<size_t>(n_edges) * sizeof(double)));
            CUDA_CHECK(cudaMemset(d_dbg_fhv, 0xA5, static_cast<size_t>(n_edges) * sizeof(double)));
        }
        int grid = (n_edges + BLOCK - 1) / BLOCK;
        swe2d_flux_kernel<<<grid, BLOCK>>>(
            n_edges,
            dev->d_edge_c0, dev->d_edge_c1,
            dev->d_edge_nx, dev->d_edge_ny, dev->d_edge_len,
            dev->d_edge_bc, dev->d_edge_bc_val,
            dev->d_h, dev->d_hu, dev->d_hv,
            dev->d_cell_zb,
            dev->d_cell_inv_area,
            dev->d_cell_cx, dev->d_cell_cy,
            dev->d_grad_hx,  dev->d_grad_hy,
            dev->d_grad_hux, dev->d_grad_huy,
            dev->d_grad_hvx, dev->d_grad_hvy,
            dev->d_flux_h, dev->d_flux_hu, dev->d_flux_hv,
            d_dbg_fh, d_dbg_fhu, d_dbg_fhv,
            spatial_scheme,
            g, h_min);
        CUDA_CHECK(cudaGetLastError());

        if (dbg_edge_flux) {
            std::vector<double> fh(static_cast<size_t>(n_edges));
            std::vector<double> fhu(static_cast<size_t>(n_edges));
            std::vector<double> fhv(static_cast<size_t>(n_edges));
            CUDA_CHECK(cudaMemcpy(fh.data(), d_dbg_fh,
                                  static_cast<size_t>(n_edges) * sizeof(double),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(fhu.data(), d_dbg_fhu,
                                  static_cast<size_t>(n_edges) * sizeof(double),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(fhv.data(), d_dbg_fhv,
                                  static_cast<size_t>(n_edges) * sizeof(double),
                                  cudaMemcpyDeviceToHost));
            const size_t n_show = std::min<size_t>(static_cast<size_t>(n_edges), 12);
            for (size_t i = 0; i < n_show; ++i) {
                std::fprintf(stderr,
                             "[SWE2D_DEBUG] GPU edge_flux[%zu]: fh=%.9e fhu=%.9e fhv=%.9e\n",
                             i, fh[i], fhu[i], fhv[i]);
            }
            CUDA_CHECK(cudaFree(d_dbg_fh));
            CUDA_CHECK(cudaFree(d_dbg_fhu));
            CUDA_CHECK(cudaFree(d_dbg_fhv));
        }
    }

    if (swe2d_debug_enabled("BACKWATER_SWE2D_DEBUG_GPU_FLUX")) {
        std::vector<double> h_flux(static_cast<size_t>(n_cells));
        std::vector<double> hu_flux(static_cast<size_t>(n_cells));
        std::vector<double> hv_flux(static_cast<size_t>(n_cells));
        CUDA_CHECK(cudaMemcpy(h_flux.data(), dev->d_flux_h,
                              static_cast<size_t>(n_cells) * sizeof(double),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hu_flux.data(), dev->d_flux_hu,
                              static_cast<size_t>(n_cells) * sizeof(double),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hv_flux.data(), dev->d_flux_hv,
                              static_cast<size_t>(n_cells) * sizeof(double),
                              cudaMemcpyDeviceToHost));
        dump_flux_summary("GPU", h_flux, hu_flux, hv_flux);
    }

    // Kernel 2: Update
    {
        CUDA_CHECK(cudaMemset(dev->d_max_wse_elev_error, 0, sizeof(double)));
        int grid = (n_cells + BLOCK - 1) / BLOCK;
        swe2d_update_kernel<<<grid, BLOCK>>>(
            n_cells,
            dev->d_h, dev->d_hu, dev->d_hv,
            dev->d_flux_h, dev->d_flux_hu, dev->d_flux_hv,
            dev->d_cell_inv_area, dev->d_n_mann_cell,
            dev->d_max_wse_elev_error,
            dt, g, h_min);
        CUDA_CHECK(cudaGetLastError());
    }

    // Kernel 3: CFL reduction for max Courant diagnostic
    CUDA_CHECK(cudaMemset(dev->d_lambda_max, 0, sizeof(double)));
    {
        int grid = (n_cells + BLOCK - 1) / BLOCK;
        swe2d_cfl_kernel<<<grid, BLOCK, BLOCK * static_cast<int>(sizeof(double))>>>(
            n_cells,
            dev->d_h, dev->d_hu, dev->d_hv,
            dev->d_cell_area,
            g, h_min,
            dev->d_lambda_max);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // Fill diagnostics (lightweight host-side pass; can be replaced by
    // a dedicated reduce kernel for large domains)
    if (diag) {
        double lambda_max = 0.0;
        double max_wse_elev_error = 0.0;
        CUDA_CHECK(cudaMemcpy(&lambda_max, dev->d_lambda_max, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&max_wse_elev_error, dev->d_max_wse_elev_error, sizeof(double), cudaMemcpyDeviceToHost));

        diag->dt         = dt;
        diag->gpu_active = true;
        // Wet cell count and mass require device-to-host or a reduce kernel.
        // For MVP, set to sentinel values indicating GPU path ran.
        diag->wet_cells  = -1;  // -1 = not computed on this path
        diag->max_depth  = -1.0;
        diag->min_depth  = -1.0;
        diag->mass_total = -1.0;
        diag->max_courant = dt * lambda_max;
        diag->max_depth_residual = max_wse_elev_error;
        diag->max_wse_elev_error = max_wse_elev_error;
    }
}

double swe2d_gpu_compute_dt(
    SWE2DDeviceState* dev,
    double g,
    double h_min,
    double cfl_factor,
    double dt_max)
{
    constexpr int BLOCK = 256;
    const int32_t n_cells = dev->n_cells;

    CUDA_CHECK(cudaMemset(dev->d_lambda_max, 0, sizeof(double)));
    int grid = (n_cells + BLOCK - 1) / BLOCK;
    swe2d_cfl_kernel<<<grid, BLOCK, BLOCK * static_cast<int>(sizeof(double))>>>(
        n_cells,
        dev->d_h, dev->d_hu, dev->d_hv,
        dev->d_cell_area,
        g, h_min,
        dev->d_lambda_max);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    double lambda_max = 0.0;
    CUDA_CHECK(cudaMemcpy(&lambda_max, dev->d_lambda_max, sizeof(double), cudaMemcpyDeviceToHost));

    if (lambda_max <= 0.0) {
        return dt_max;
    }
    const double dt = cfl_factor / lambda_max;
    return (dt < dt_max) ? dt : dt_max;
}

void swe2d_gpu_step_rk2(
    SWE2DDeviceState* dev,
    double dt,
    double g,
    double h_min,
    int spatial_scheme,
    double cfl_factor,
    SWE2DStepDiag* diag)
{
    constexpr int BLOCK = 256;
    const int32_t n_cells = dev->n_cells;
    const size_t sz = static_cast<size_t>(n_cells) * sizeof(double);

    CUDA_CHECK(cudaMemcpy(dev->d_h0, dev->d_h, sz, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(dev->d_hu0, dev->d_hu, sz, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(dev->d_hv0, dev->d_hv, sz, cudaMemcpyDeviceToDevice));

    SWE2DStepDiag tmp_diag;
    swe2d_gpu_step(dev, dt, g, h_min, spatial_scheme, cfl_factor, &tmp_diag);
    swe2d_gpu_step(dev, dt, g, h_min, spatial_scheme, cfl_factor, &tmp_diag);

    CUDA_CHECK(cudaMemset(dev->d_max_wse_elev_error, 0, sizeof(double)));
    int grid = (n_cells + BLOCK - 1) / BLOCK;
    swe2d_rk2_combine_kernel<<<grid, BLOCK>>>(
        n_cells,
        dev->d_h, dev->d_hu, dev->d_hv,
        dev->d_h0, dev->d_hu0, dev->d_hv0,
        dev->d_max_wse_elev_error,
        h_min);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemset(dev->d_lambda_max, 0, sizeof(double)));
    swe2d_cfl_kernel<<<grid, BLOCK, BLOCK * static_cast<int>(sizeof(double))>>>(
        n_cells,
        dev->d_h, dev->d_hu, dev->d_hv,
        dev->d_cell_area,
        g, h_min,
        dev->d_lambda_max);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    if (diag) {
        double lambda_max = 0.0;
        double max_depth_residual = 0.0;
        CUDA_CHECK(cudaMemcpy(&lambda_max, dev->d_lambda_max, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&max_depth_residual, dev->d_max_wse_elev_error, sizeof(double), cudaMemcpyDeviceToHost));

        diag->dt = dt;
        diag->gpu_active = true;
        diag->wet_cells = -1;
        diag->max_depth = -1.0;
        diag->min_depth = -1.0;
        diag->mass_total = -1.0;
        diag->max_courant = dt * lambda_max;
        diag->max_depth_residual = max_depth_residual;
        diag->max_wse_elev_error = max_depth_residual;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// swe2d_gpu_get_state
// ─────────────────────────────────────────────────────────────────────────────
void swe2d_gpu_get_state(
    SWE2DDeviceState* dev,
    double* h_out,
    double* hu_out,
    double* hv_out)
{
    size_t sz = static_cast<size_t>(dev->n_cells) * sizeof(double);
    if (h_out)  CUDA_CHECK(cudaMemcpy(h_out,  dev->d_h,  sz, cudaMemcpyDeviceToHost));
    if (hu_out) CUDA_CHECK(cudaMemcpy(hu_out, dev->d_hu, sz, cudaMemcpyDeviceToHost));
    if (hv_out) CUDA_CHECK(cudaMemcpy(hv_out, dev->d_hv, sz, cudaMemcpyDeviceToHost));
}

void swe2d_gpu_set_state(
    SWE2DDeviceState* dev,
    const double* h_in,
    const double* hu_in,
    const double* hv_in)
{
    size_t sz = static_cast<size_t>(dev->n_cells) * sizeof(double);
    if (h_in)  CUDA_CHECK(cudaMemcpy(dev->d_h,  h_in,  sz, cudaMemcpyHostToDevice));
    if (hu_in) CUDA_CHECK(cudaMemcpy(dev->d_hu, hu_in, sz, cudaMemcpyHostToDevice));
    if (hv_in) CUDA_CHECK(cudaMemcpy(dev->d_hv, hv_in, sz, cudaMemcpyHostToDevice));
}

// ─────────────────────────────────────────────────────────────────────────────
// swe2d_gpu_destroy
// ─────────────────────────────────────────────────────────────────────────────
void swe2d_gpu_destroy(SWE2DDeviceState* dev) {
    if (!dev) return;
    auto safe_free = [](void* ptr) { if (ptr) cudaFree(ptr); };

    safe_free(dev->d_edge_c0);    safe_free(dev->d_edge_c1);
    safe_free(dev->d_edge_n0);    safe_free(dev->d_edge_n1);
    safe_free(dev->d_edge_nx);    safe_free(dev->d_edge_ny);
    safe_free(dev->d_edge_len);   safe_free(dev->d_edge_bc);
    safe_free(dev->d_edge_bc_val);
    safe_free(dev->d_cell_zb);    safe_free(dev->d_cell_area);
    safe_free(dev->d_cell_inv_area);
    safe_free(dev->d_n_mann_cell);
    safe_free(dev->d_cell_cx);    safe_free(dev->d_cell_cy);
    safe_free(dev->d_grad_hx);    safe_free(dev->d_grad_hy);
    safe_free(dev->d_grad_hux);   safe_free(dev->d_grad_huy);
    safe_free(dev->d_grad_hvx);   safe_free(dev->d_grad_hvy);
    safe_free(dev->d_h);          safe_free(dev->d_hu);
    safe_free(dev->d_hv);
    safe_free(dev->d_h0);         safe_free(dev->d_hu0);
    safe_free(dev->d_hv0);
    safe_free(dev->d_flux_h);     safe_free(dev->d_flux_hu);
    safe_free(dev->d_flux_hv);
    safe_free(dev->d_lambda_max);
    safe_free(dev->d_max_wse_elev_error);
    delete dev;
}
