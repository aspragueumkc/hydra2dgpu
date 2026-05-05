// swe2d_solver.cpp
// CPU solver implementation for the 2D SWE on an unstructured triangular mesh.
// OpenMP parallelism is used for the flux and update loops when available.
//
// SWE2D project direction note:
// - Active numerics, validation, and performance work are GPU-first.
// - The CPU path is kept only as a maintenance/debug fallback so the plugin can
//   still run without CUDA and so algorithm experiments can be inspected more
//   easily in host code.
// - CPU/GPU parity is no longer a primary engineering objective here; do not
//   block GPU optimization work on keeping these paths numerically matched.

#include "swe2d_solver.hpp"
#include "swe2d_numerics.hpp"

#ifdef BACKWATER_HAS_CUDA
#  include "swe2d_gpu.cuh"
#endif

#ifdef BACKWATER_HAS_OPENMP
#  include <omp.h>
#endif

#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <limits>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <mutex>

namespace {

static double compute_lambda_max(const SWE2DSolver* s);

inline double clamp_double(double v, double lo, double hi) {
    return std::max(lo, std::min(v, hi));
}

// ─────────────────────────────────────────────────────────────────────────────
// Green-Gauss gradient — identical algorithm to swe2d_gradient_kernel (GPU).
// Must be called with pre-zeroed gx/gy vectors.
// Boundary edges (c1 < 0) contribute a face value equal to c0 (zero-gradient).
// ─────────────────────────────────────────────────────────────────────────────
void compute_gg_gradient_cpu(
    const SWE2DMesh& mesh,
    const std::vector<double>& q,
    std::vector<double>& gx,
    std::vector<double>& gy)
{
    const int32_t n_cells = mesh.n_cells;
    const int32_t n_edges = mesh.n_edges;
    gx.assign(static_cast<size_t>(n_cells), 0.0);
    gy.assign(static_cast<size_t>(n_cells), 0.0);

    for (int32_t e = 0; e < n_edges; ++e) {
        const int32_t c0 = mesh.edge_c0[e];
        const int32_t c1 = mesh.edge_c1[e];
        const double  nx  = mesh.edge_nx[e];
        const double  ny  = mesh.edge_ny[e];
        const double  len = mesh.edge_len[e];

        // Skip degenerate cells
        if (mesh.cell_inv_area[c0] > 1.0e6) continue;

        const double q0 = q[static_cast<size_t>(c0)];
        const double q1 = (c1 >= 0 && mesh.cell_inv_area[c1] <= 1.0e6)
                          ? q[static_cast<size_t>(c1)] : q0;
        const double qf = 0.5 * (q0 + q1);

        const double ia0 = mesh.cell_inv_area[c0];
        gx[static_cast<size_t>(c0)] += qf * nx * len * ia0;
        gy[static_cast<size_t>(c0)] += qf * ny * len * ia0;

        if (c1 >= 0 && mesh.cell_inv_area[c1] <= 1.0e6) {
            const double ia1 = mesh.cell_inv_area[c1];
            gx[static_cast<size_t>(c1)] -= qf * nx * len * ia1;
            gy[static_cast<size_t>(c1)] -= qf * ny * len * ia1;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TVD phi limiter — identical formulas to swe2d_flux_kernel (GPU).
// scheme: 1=Superbee, 2=MinMod, 3=MC, 4=VanLeer
// ─────────────────────────────────────────────────────────────────────────────
inline double phi_tvd_cpu(double r, int scheme)
{
    switch (scheme) {
        case 1:  // Superbee (most aggressive)
            return std::fmax(0.0, std::fmax(std::fmin(2.0 * r, 1.0),
                                            std::fmin(r, 2.0)));
        case 2:  // MinMod (most conservative)
            return std::fmax(0.0, std::fmin(r, 1.0));
        case 3:  // MC (monotonized central)
            return std::fmax(0.0, std::fmin(std::fmin(2.0 * r, 0.5 * (1.0 + r)),
                                            2.0));
        case 4:  // Van Leer (smooth)
            return (r + std::fabs(r)) / (1.0 + std::fabs(r));
        default:
            return 0.0;
    }
}

#ifdef BACKWATER_HAS_CUDA
void sync_gpu_state_to_host(SWE2DSolver* s) {
    if (!s || !s->dev) return;
    swe2d_gpu_get_state(s->dev, s->h.data(), s->hu.data(), s->hv.data());
}
#endif

bool swe2d_debug_enabled(const char* name) {
    const char* v = std::getenv(name);
    return (v && v[0] && v[0] != '0');
}

SWE2DStepDiag summarize_state(const SWE2DSolver* s, double dt, bool gpu_active, double max_depth_residual) {
    const SWE2DMesh& mesh = *s->mesh;
    const double h_min = s->cfg.h_min;

    SWE2DStepDiag diag;
    diag.dt = dt;
    diag.gpu_active = gpu_active;
    diag.max_depth = 0.0;
    diag.min_depth = std::numeric_limits<double>::max();
    diag.wet_cells = 0;
    diag.mass_total = 0.0;
    diag.max_depth_residual = max_depth_residual;
    diag.max_wse_elev_error = max_depth_residual;

    for (int32_t c = 0; c < mesh.n_cells; ++c) {
        const double h = s->h[c];
        if (h > h_min) {
            diag.wet_cells += 1;
            if (h > diag.max_depth) diag.max_depth = h;
            if (h < diag.min_depth) diag.min_depth = h;
        }
        diag.mass_total += h * mesh.cell_area[c];
    }

    if (diag.min_depth == std::numeric_limits<double>::max()) {
        diag.min_depth = 0.0;
    }

    diag.max_courant = dt * compute_lambda_max(s);
    return diag;
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

} // namespace

// ─────────────────────────────────────────────────────────────────────────────
// swe2d_gpu_available — CPU-only stub (overridden by swe2d_gpu.cu when compiled)
// ─────────────────────────────────────────────────────────────────────────────
#ifndef BACKWATER_HAS_CUDA
bool swe2d_gpu_available() { return false; }
#endif

// ─────────────────────────────────────────────────────────────────────────────
// swe2d_create
// ─────────────────────────────────────────────────────────────────────────────
SWE2DSolver* swe2d_create(
    const SWE2DMesh& mesh,
    const double*    h0,
    const double*    hu0,
    const double*    hv0,
    const double*    n_mann_cell,
    const SWE2DSolverConfig& cfg)
{
    if (h0 == nullptr) {
        throw std::invalid_argument("swe2d_create: h0 must not be null");
    }

    auto* s = new SWE2DSolver();
    s->mesh = &mesh;
    s->cfg  = cfg;
    s->t    = 0.0;

    int32_t n = mesh.n_cells;
    s->h.assign(h0, h0 + n);
    s->hu.assign(n, 0.0);
    s->hv.assign(n, 0.0);
    s->n_mann_cell.assign(n, cfg.n_mann);
    if (hu0) std::copy(hu0, hu0 + n, s->hu.begin());
    if (hv0) std::copy(hv0, hv0 + n, s->hv.begin());
    if (n_mann_cell) std::copy(n_mann_cell, n_mann_cell + n, s->n_mann_cell.begin());

    s->dh.assign(n, 0.0);
    s->dhu.assign(n, 0.0);
    s->dhv.assign(n, 0.0);

    #ifdef BACKWATER_HAS_OPENMP
    if (cfg.n_threads > 0) {
        omp_set_num_threads(cfg.n_threads);
    }
#endif

    // Initialise GPU state if requested and available
#ifdef BACKWATER_HAS_CUDA
    s->dev = nullptr;
    if (cfg.use_gpu && swe2d_gpu_available()) {
        s->dev = swe2d_gpu_init(mesh,
                                s->h.data(), s->hu.data(), s->hv.data(),
                                s->n_mann_cell.data(),
                                cfg.degen_mode,
                                cfg.max_inv_area);
    }
#endif

    return s;
}

// ─────────────────────────────────────────────────────────────────────────────
// swe2d_destroy
// ─────────────────────────────────────────────────────────────────────────────
void swe2d_destroy(SWE2DSolver* s) {
    if (!s) return;
#ifdef BACKWATER_HAS_CUDA
    if (s->dev) {
        swe2d_gpu_destroy(s->dev);
        s->dev = nullptr;
    }
#endif
    delete s;
}

// ─────────────────────────────────────────────────────────────────────────────
// swe2d_get_state
// ─────────────────────────────────────────────────────────────────────────────
void swe2d_get_state(const SWE2DSolver* s, double* h_out, double* hu_out, double* hv_out) {
    if (!s) return;
#ifdef BACKWATER_HAS_CUDA
    if (s->dev) {
        // GPU path: transfer directly device → caller; host mirror is not updated.
        // State remains device-resident between explicit snapshots.
        swe2d_gpu_get_state(s->dev, h_out, hu_out, hv_out);
        return;
    }
#endif
    if (h_out)  std::copy(s->h.begin(),  s->h.end(),  h_out);
    if (hu_out) std::copy(s->hu.begin(), s->hu.end(), hu_out);
    if (hv_out) std::copy(s->hv.begin(), s->hv.end(), hv_out);
}

void swe2d_set_state(SWE2DSolver* s, const double* h_in, const double* hu_in, const double* hv_in) {
    if (!s) return;
    const int32_t n = s->mesh->n_cells;
    if (h_in) {
        std::copy(h_in, h_in + n, s->h.begin());
    }
    if (hu_in) {
        std::copy(hu_in, hu_in + n, s->hu.begin());
    }
    if (hv_in) {
        std::copy(hv_in, hv_in + n, s->hv.begin());
    }

#ifdef BACKWATER_HAS_CUDA
    if (s->dev) {
        swe2d_gpu_set_state(s->dev, s->h.data(), s->hu.data(), s->hv.data());
    }
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// CFL timestep calculation — CPU pass over all edges
// ─────────────────────────────────────────────────────────────────────────────
namespace {
static double compute_lambda_max(const SWE2DSolver* s) {
    const SWE2DMesh& mesh = *s->mesh;
    const double g        = s->cfg.g;
    const double h_min    = s->cfg.h_min;

    double lambda_max = 0.0;

    #ifdef BACKWATER_HAS_OPENMP
    #pragma omp parallel for reduction(max:lambda_max) schedule(static)
    #endif
    for (int32_t e = 0; e < mesh.n_edges; ++e) {
        int32_t c0 = mesh.edge_c0[e];
        int32_t c1 = mesh.edge_c1[e];

        double hL  = s->h[c0],  huL = s->hu[c0], hvL = s->hv[c0];
        double aL  = mesh.cell_area[c0];
        double hR, huR, hvR, aR;

        if (c1 >= 0) {
            hR  = s->h[c1]; huR = s->hu[c1]; hvR = s->hv[c1];
            aR  = mesh.cell_area[c1];
        } else {
            // Boundary: use ghost with WALL (conservative estimate)
            hR = hL; huR = huL; hvR = hvL;
            aR = aL;
        }

        double lam = swe2d::edge_cfl_lambda(
            hL, huL, hvL, hR, huR, hvR,
            mesh.edge_nx[e], mesh.edge_ny[e], mesh.edge_len[e],
            aL, aR, g, h_min);

        if (lam > lambda_max) lambda_max = lam;
    }

    return lambda_max;
}
} // namespace

static double compute_cfl_dt(const SWE2DSolver* s) {
    const double cfl = s->cfg.cfl;
    double lambda_max = compute_lambda_max(s);

    if (lambda_max <= 0.0) return s->cfg.dt_max;
    double dt = cfl / lambda_max;
    return std::min(dt, s->cfg.dt_max);
}

// ─────────────────────────────────────────────────────────────────────────────
// swe2d_step_cpu — one explicit Euler timestep on CPU
// COMPATIBILITY FALLBACK ONLY.  This path is used when no CUDA device is
// present.  All active SWE2D numerical work (new schemes, optimisations,
// robustness fixes) targets the CUDA path in swe2d_gpu.cu.  Do not backport
// GPU-only improvements here unless explicitly required for the CPU fallback
// to remain runnable.
// ─────────────────────────────────────────────────────────────────────────────
SWE2DStepDiag swe2d_step_cpu(SWE2DSolver* s, double dt) {
    static std::once_flag s_cpu_warn;
    std::call_once(s_cpu_warn, []() {
        std::fprintf(stderr,
            "[SWE2D] WARNING: CPU fallback solver path is active. "
            "GPU path is strongly preferred for all production runs.\n");
    });
    const SWE2DMesh& mesh = *s->mesh;
    const double g        = s->cfg.g;
    const double h_min    = s->cfg.h_min;
    int32_t n_cells       = mesh.n_cells;
    int32_t n_edges       = mesh.n_edges;
    const int spatial_scheme = s->cfg.spatial_scheme;
    // All schemes >= 1 use Green-Gauss gradient + phi-TVD reconstruction,
    // consistent with the GPU kernel (swe2d_gpu.cu).  The old IDW gradient +
    // Barth-Jespersen path only covered schemes 1 and 2 and produced different
    // numerical results from the GPU, leading to instability on unstructured
    // meshes for scheme 1 (no limiter) and incorrect well-balancing for all.
    // Keep this CPU logic functionally sane, but performance tuning effort
    // belongs in the GPU path rather than here.
    const bool use_higher_order = (spatial_scheme >= 1);

    std::vector<double> ghx, ghy;
    std::vector<double> guhx, guhy;
    std::vector<double> gvhx, gvhy;

    if (use_higher_order) {
        compute_gg_gradient_cpu(mesh, s->h,  ghx,  ghy);
        compute_gg_gradient_cpu(mesh, s->hu, guhx, guhy);
        compute_gg_gradient_cpu(mesh, s->hv, gvhx, gvhy);
    }

    // For the surface-gradient method we need ∇η where η = h + zb.
    // Build per-cell η view and compute its GG gradient.
    std::vector<double> getax, getay;
    if (use_higher_order) {
        std::vector<double> eta(static_cast<size_t>(n_cells));
        for (int32_t c = 0; c < n_cells; ++c)
            eta[static_cast<size_t>(c)] = s->h[c] + mesh.cell_zb[c];
        compute_gg_gradient_cpu(mesh, eta, getax, getay);
    }

    // Zero flux accumulators
    #ifdef BACKWATER_HAS_OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int32_t c = 0; c < n_cells; ++c) {
        s->dh[c] = s->dhu[c] = s->dhv[c] = 0.0;
    }

    // ── Flux loop ─────────────────────────────────────────────────────────────
    // Note: parallel over edges with atomics is correct but has contention on
    // shared cells.  For MVP this is acceptable; a CSR-sorted edge-to-cell pass
    // can eliminate atomics in a follow-up optimisation.
    for (int32_t e = 0; e < n_edges; ++e) {
        int32_t c0 = mesh.edge_c0[e];
        int32_t c1 = mesh.edge_c1[e];
        
        // Skip flux TO degenerate cells to prevent overflow
        // (degenerate cells have inv_area > 1e6)
        if (mesh.cell_inv_area[c0] > 1.0e6 || (c1 >= 0 && mesh.cell_inv_area[c1] > 1.0e6)) {
            continue;
        }
        
        double  nx = mesh.edge_nx[e];
        double  ny = mesh.edge_ny[e];
        double  len = mesh.edge_len[e];

        double hL  = s->h[c0],  huL = s->hu[c0], hvL = s->hv[c0];
        double zbL = mesh.cell_zb[c0];

        double hR, huR, hvR, zbR;
        if (c1 >= 0) {
            hR  = s->h[c1]; huR = s->hu[c1]; hvR = s->hv[c1];
            zbR = mesh.cell_zb[c1];

            if (use_higher_order) {
                    // Green-Gauss + phi-TVD reconstruction — mirrors GPU kernel.
                    // Cell-to-cell vector (not face midpoint) is used for the
                    // slope ratio, exactly as in swe2d_flux_kernel.
                    constexpr double EPS = 1.0e-30;
                    const double dcx = mesh.cell_cx[c1] - mesh.cell_cx[c0];
                    const double dcy = mesh.cell_cy[c1] - mesh.cell_cy[c0];

                    auto tvd_rec = [&](double q0, double q1,
                                       double gx0, double gy0,
                                       double gx1, double gy1,
                                       double& qL_out, double& qR_out)
                    {
                        const double dq = q1 - q0;
                        const double s0 =  (gx0 * dcx + gy0 * dcy);
                        const double s1 = -(gx1 * dcx + gy1 * dcy);
                        const double sign_dq = (dq >= 0.0) ? 1.0 : -1.0;
                        const double r0 =  s0 / (dq + sign_dq * EPS);
                        const double r1 =  s1 / (-dq + (-sign_dq) * EPS);
                        const double phi0 = phi_tvd_cpu(r0, spatial_scheme);
                        const double phi1 = phi_tvd_cpu(r1, spatial_scheme);
                        qL_out = q0 + phi0 * 0.5 * dq;
                        qR_out = q1 - phi1 * 0.5 * dq;
                    };

                        // Surface-gradient method (Zhou et al. 2001):
                        // Reconstruct η = h + zb via ∇η, then convert back to h.
                        // For lake-at-rest: dη = 0 → phi = 0 → hL = h_c0, hR = h_c1
                        // → exact well-balancing independent of mesh irregularity.
                        {
                            const double etaL_c = hL + zbL;
                            const double etaR_c = hR + zbR;
                            double etaL_r, etaR_r;
                            tvd_rec(etaL_c, etaR_c,
                                    getax[c0], getay[c0], getax[c1], getay[c1],
                                    etaL_r, etaR_r);
                            hL = std::fmax(0.0, etaL_r - zbL);
                            hR = std::fmax(0.0, etaR_r - zbR);
                        }

                        // Momentum: reconstruct using ∇h (not ∇η — velocity/momentum
                        // is not directly affected by the bed elevation bias).
                        double huL_r, huR_r, hvL_r, hvR_r;
                        tvd_rec(huL, huR, guhx[c0], guhy[c0], guhx[c1], guhy[c1], huL_r, huR_r);
                        tvd_rec(hvL, hvR, gvhx[c0], gvhy[c0], gvhx[c1], gvhy[c1], hvL_r, hvR_r);
                        huL = huL_r; huR = huR_r;
                        hvL = hvL_r; hvR = hvR_r;

                        // Momentum cap (matches GPU update_kernel logic)
                        const double hL_eff = (hL > h_min) ? hL : h_min;
                        const double hR_eff = (hR > h_min) ? hR : h_min;
                        const double u_cap_L = std::fmax(50.0, 20.0 * std::sqrt(g * hL_eff));
                        const double u_cap_R = std::fmax(50.0, 20.0 * std::sqrt(g * hR_eff));
                        huL = clamp_double(huL, -hL_eff * u_cap_L, hL_eff * u_cap_L);
                        hvL = clamp_double(hvL, -hL_eff * u_cap_L, hL_eff * u_cap_L);
                        huR = clamp_double(huR, -hR_eff * u_cap_R, hR_eff * u_cap_R);
                        hvR = clamp_double(hvR, -hR_eff * u_cap_R, hR_eff * u_cap_R);

            }
        } else {
            // Boundary edge: construct ghost cell
            swe2d::GhostState gs = swe2d::make_ghost(
                hL, huL, hvL, zbL,
                nx, ny,
                static_cast<int>(mesh.edge_bc[e]),
                mesh.edge_bc_val[e],
                h_min);
            hR  = gs.h;  huR = gs.hu; hvR = gs.hv;
            zbR = gs.zb;
        }

        double zb_face = 0.0;
        swe2d::HLLCFlux flux = swe2d::edge_flux(
            hL, huL, hvL, zbL,
            hR, huR, hvR, zbR,
            nx, ny, g, h_min, zb_face);

        // Well-balanced bed-slope correction for momentum
        swe2d::ReconstructedStates rs = swe2d::hydrostatic_reconstruct(
            hL, huL, hvL, zbL, hR, huR, hvR, zbR, h_min);
        double corr_hu = 0.0, corr_hv = 0.0;
        swe2d::bed_slope_correction(hL, rs.hL_star, nx, ny, g, corr_hu, corr_hv);

        // Accumulate into c0 (negative: flux leaves c0)
        s->dh[c0]  -= flux.fh  * len;
        s->dhu[c0] -= (flux.fhu + corr_hu) * len;
        s->dhv[c0] -= (flux.fhv + corr_hv) * len;

        if (c1 >= 0) {
            // Flux enters c1 (positive, opposite sign)
            // Well-balanced correction for c1: use the SAME normal direction (nx, ny)
            // as c0.  The correction term bed_slope_correction(hR, hR*, nx, ny) produces
            // net_dhu[c1] += +0.5*g*hR^2*nx*len per edge, which sums to zero over c1's
            // closed polygon — matching the c0 balance.  Passing -nx,-ny would break
            // the lake-at-rest invariant.
            double corr_hu_r = 0.0, corr_hv_r = 0.0;
            swe2d::bed_slope_correction(hR, rs.hR_star, nx, ny, g, corr_hu_r, corr_hv_r);
            s->dh[c1]  += flux.fh  * len;
            s->dhu[c1] += (flux.fhu + corr_hu_r) * len;
            s->dhv[c1] += (flux.fhv + corr_hv_r) * len;
        }
    }

    // ── Update loop ───────────────────────────────────────────────────────────
    // OpenMP struct-member reductions require OMP 5.0 which is not universally
    // available.  Use local scalars and assign to the diag struct afterwards.
    int32_t r_wet_cells  = 0;
    double  r_max_depth  = 0.0;
    double  r_min_depth  = std::numeric_limits<double>::max();
    double  r_mass_total = 0.0;
    double  r_max_wse_elev_error = 0.0;

    #ifdef BACKWATER_HAS_OPENMP
    #pragma omp parallel for schedule(static) \
        reduction(+:r_wet_cells, r_mass_total) \
        reduction(max:r_max_depth) \
        reduction(min:r_min_depth) \
        reduction(max:r_max_wse_elev_error)
    #endif
    for (int32_t c = 0; c < n_cells; ++c) {
        const double h_old = s->h[c];
        double inv_a = mesh.cell_inv_area[c];
        double area  = mesh.cell_area[c];

        s->h[c]  += dt * s->dh[c]  * inv_a;
        s->hu[c] += dt * s->dhu[c] * inv_a;
        s->hv[c] += dt * s->dhv[c] * inv_a;

        // Positivity enforcement
        if (s->h[c] < 0.0) s->h[c] = 0.0;
        if (s->h[c] < h_min) {
            s->hu[c] = 0.0;
            s->hv[c] = 0.0;
        }

        // Manning friction (semi-implicit)
        double n_mann = s->n_mann_cell[c];
        swe2d::apply_friction(s->h[c], s->hu[c], s->hv[c],
                              dt, n_mann, g, h_min);

        const double wse_err = std::abs(s->h[c] - h_old);
        if (wse_err > r_max_wse_elev_error) {
            r_max_wse_elev_error = wse_err;
        }

        // Diagnostics
        double h_c = s->h[c];
        if (h_c > h_min) {
            r_wet_cells++;
            if (h_c > r_max_depth) r_max_depth = h_c;
            if (h_c < r_min_depth) r_min_depth = h_c;
        }
        r_mass_total += h_c * area;
    }

    if (swe2d_debug_enabled("BACKWATER_SWE2D_DEBUG_CPU_FLUX")) {
        dump_flux_summary("CPU", s->dh, s->dhu, s->dhv);
    }

    SWE2DStepDiag diag;
    diag.gpu_active  = false;
    diag.dt          = dt;
    diag.wet_cells   = r_wet_cells;
    diag.max_depth   = r_max_depth;
    diag.min_depth   = (r_min_depth == std::numeric_limits<double>::max()) ? 0.0 : r_min_depth;
    diag.mass_total  = r_mass_total;
    diag.max_courant = dt * compute_lambda_max(s);
    diag.max_depth_residual = r_max_wse_elev_error;
    diag.max_wse_elev_error = r_max_wse_elev_error;

    return diag;
}

// ─────────────────────────────────────────────────────────────────────────────
// swe2d_step — main dispatch (GPU preferred, CPU fallback)
// ─────────────────────────────────────────────────────────────────────────────
SWE2DStepDiag swe2d_step(SWE2DSolver* s, double dt_request) {
    if (!s) throw std::invalid_argument("swe2d_step: null solver");

    const bool use_rk2 = (s->cfg.temporal_order >= 2);
    const int diag_interval = s->cfg.gpu_diag_sync_interval_steps;
    const bool sync_diag_this_step =
        (diag_interval > 0) ? ((s->gpu_steps % static_cast<uint64_t>(diag_interval)) == 0u) : false;

#ifdef BACKWATER_HAS_CUDA
    if (s->dev) {
        double dt;
        if (s->cfg.dt_fixed > 0.0) {
            dt = s->cfg.dt_fixed;
        } else {
            const double dt_cfl = swe2d_gpu_compute_dt(
                s->dev,
                s->cfg.g,
                s->cfg.h_min,
                s->cfg.cfl,
                s->cfg.dt_max,
                s->cfg.cfl_lambda_cap);
            dt = (dt_request > 0.0) ? std::min(dt_request, dt_cfl) : dt_cfl;
        }

        if (!use_rk2) {
            SWE2DStepDiag diag;
            swe2d_gpu_step(s->dev, dt,
                           s->cfg.g, s->cfg.h_min,
                           s->cfg.spatial_scheme,
                           s->cfg.cfl,
                           s->cfg.max_inv_area,
                           s->cfg.cfl_lambda_cap,
                           s->cfg.momentum_cap_min_speed,
                           s->cfg.momentum_cap_celerity_mult,
                           s->cfg.depth_cap,
                           s->cfg.max_rel_depth_increase,
                           s->cfg.shallow_damping_depth,
                           sync_diag_this_step,
                           &diag,
                           s->cfg.front_flux_damping,
                           s->cfg.active_set_hysteresis);
            diag.gpu_active = true;
            s->t += dt;
            s->gpu_steps += 1;
            return diag;
        }

        SWE2DStepDiag diag;
        swe2d_gpu_step_rk2(s->dev, dt,
                           s->cfg.g, s->cfg.h_min,
                           s->cfg.spatial_scheme,
                           s->cfg.cfl,
                           s->cfg.max_inv_area,
                           s->cfg.cfl_lambda_cap,
                           s->cfg.momentum_cap_min_speed,
                           s->cfg.momentum_cap_celerity_mult,
                           s->cfg.depth_cap,
                           s->cfg.max_rel_depth_increase,
                           s->cfg.shallow_damping_depth,
                           sync_diag_this_step,
                           &diag,
                           s->cfg.front_flux_damping,
                           s->cfg.active_set_hysteresis);
        s->t += dt;
        s->gpu_steps += 1;
        return diag;
    }
#endif

    // Determine timestep (CPU path)
    double dt;
    if (s->cfg.dt_fixed > 0.0) {
        dt = s->cfg.dt_fixed;
    } else {
        double dt_cfl = compute_cfl_dt(s);
        dt = (dt_request > 0.0) ? std::min(dt_request, dt_cfl) : dt_cfl;
    }

    if (!use_rk2) {
        SWE2DStepDiag diag = swe2d_step_cpu(s, dt);
        s->t += dt;
        return diag;
    }

    std::vector<double> h0 = s->h;
    std::vector<double> hu0 = s->hu;
    std::vector<double> hv0 = s->hv;

    swe2d_step_cpu(s, dt);
    swe2d_step_cpu(s, dt);

    double max_depth_residual = 0.0;
    for (int32_t c = 0; c < s->mesh->n_cells; ++c) {
        s->h[c] = 0.5 * (h0[c] + s->h[c]);
        s->hu[c] = 0.5 * (hu0[c] + s->hu[c]);
        s->hv[c] = 0.5 * (hv0[c] + s->hv[c]);

        if (s->h[c] < 0.0) s->h[c] = 0.0;
        if (s->h[c] < s->cfg.h_min) {
            s->hu[c] = 0.0;
            s->hv[c] = 0.0;
        }

        const double depth_res = std::abs(s->h[c] - h0[c]);
        if (depth_res > max_depth_residual) {
            max_depth_residual = depth_res;
        }
    }

    SWE2DStepDiag diag = summarize_state(s, dt, false, max_depth_residual);
    s->t += dt;
    return diag;
}
