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
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <mutex>

namespace {

static double compute_lambda_max(const SWE2DSolver* s);

inline double clamp_double(double v, double lo, double hi) {
    return std::max(lo, std::min(v, hi));
}

inline double interp_series_clamped_host(
    const std::vector<double>& t,
    const std::vector<double>& v,
    int32_t start,
    int32_t end,
    double x)
{
    const int32_t n = end - start;
    if (n <= 0) return 0.0;
    if (n == 1) return v[static_cast<size_t>(start)];
    if (x <= t[static_cast<size_t>(start)]) return v[static_cast<size_t>(start)];
    if (x >= t[static_cast<size_t>(end - 1)]) return v[static_cast<size_t>(end - 1)];

    auto t0 = t.begin() + start;
    auto t1 = t.begin() + end;
    auto it = std::upper_bound(t0, t1, x);
    int32_t i1 = static_cast<int32_t>(it - t.begin());
    int32_t i0 = i1 - 1;
    const double tx0 = t[static_cast<size_t>(i0)];
    const double tx1 = t[static_cast<size_t>(i1)];
    const double y0 = v[static_cast<size_t>(i0)];
    const double y1 = v[static_cast<size_t>(i1)];
    const double a = (x - tx0) / std::max(tx1 - tx0, 1.0e-12);
    return y0 + a * (y1 - y0);
}

void apply_solver_boundary_hydrographs(SWE2DSolver* s, double t_now)
{
    if (!s || !s->hydrographs_enabled) return;
    SWE2DMesh& mesh = const_cast<SWE2DMesh&>(*s->mesh);
    const int32_t n = static_cast<int32_t>(s->hg_edge_index.size());
    for (int32_t i = 0; i < n; ++i) {
        const int32_t e = s->hg_edge_index[static_cast<size_t>(i)];
        if (e < 0 || e >= mesh.n_edges) continue;
        const int32_t off0 = s->hg_offsets[static_cast<size_t>(i)];
        const int32_t off1 = s->hg_offsets[static_cast<size_t>(i + 1)];
        const double val = interp_series_clamped_host(s->hg_time_s, s->hg_value, off0, off1, t_now);
        mesh.edge_bc[e] = static_cast<BCType>(s->hg_bc_type[static_cast<size_t>(i)]);
        mesh.edge_bc_val[e] = val;
    }
}

void build_solver_rain_cn_source(SWE2DSolver* s, double t0, double t1)
{
    if (!s) return;
    const int32_t n_cells = s->mesh->n_cells;
    if (static_cast<int32_t>(s->source_terms.size()) != n_cells) {
        s->source_terms.assign(static_cast<size_t>(n_cells), 0.0);
    }

    // Seed with externally-coupled source terms (if configured), then add
    // native rain/CN forcing on top.
    if (s->external_sources_enabled &&
        static_cast<int32_t>(s->external_source_terms.size()) == n_cells) {
        std::copy(
            s->external_source_terms.begin(),
            s->external_source_terms.end(),
            s->source_terms.begin());
    } else {
        std::fill(s->source_terms.begin(), s->source_terms.end(), 0.0);
    }

    if (!s->rain_cn_enabled || t1 <= t0) return;

    for (int32_t c = 0; c < n_cells; ++c) {
        const int32_t gidx = s->rain_cell_gage[static_cast<size_t>(c)];
        if (gidx < 0) continue;
        const int32_t off0 = s->rain_gage_offsets[static_cast<size_t>(gidx)];
        const int32_t off1 = s->rain_gage_offsets[static_cast<size_t>(gidx + 1)];
        if (off1 <= off0) continue;

        const double r0 = interp_series_clamped_host(s->rain_hg_time_s, s->rain_hg_cum_mm, off0, off1, t0);
        const double r1 = interp_series_clamped_host(s->rain_hg_time_s, s->rain_hg_cum_mm, off0, off1, t1);
        const double dr = std::max(0.0, r1 - r0);
        const double p = s->rain_cum_mm[static_cast<size_t>(c)] + dr;

        const double cn = std::min(100.0, std::max(1.0, s->rain_cn[static_cast<size_t>(c)]));
        const double s_mm = std::max((25400.0 / cn) - 254.0, 0.0);
        const double ia = s->rain_ia_ratio * s_mm;
        double pe = 0.0;
        if (p > ia) {
            const double num = (p - ia) * (p - ia);
            const double den = std::max(p + (1.0 - s->rain_ia_ratio) * s_mm, 1.0e-12);
            pe = num / den;
        }
        const double de = std::max(0.0, pe - s->rain_excess_cum_mm[static_cast<size_t>(c)]);
        s->rain_cum_mm[static_cast<size_t>(c)] = p;
        s->rain_excess_cum_mm[static_cast<size_t>(c)] = pe;
        s->source_terms[static_cast<size_t>(c)] += (de * s->rain_mm_to_model_depth) / (t1 - t0);
    }
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

bool swe2d_env_enabled(const char* name) {
    const char* v = std::getenv(name);
    if (!v || !v[0]) return false;
    const char c0 = static_cast<char>(std::tolower(static_cast<unsigned char>(v[0])));
    return !(c0 == '0' || c0 == 'f' || c0 == 'n');
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

    const bool advanced_mode_requested =
        (cfg.equation_set != static_cast<int>(SWE2DEquationSet::HYDROSTATIC_2D)) ||
        (cfg.coupling_mode != static_cast<int>(SWE2DThreeDCouplingMode::OFF));
    if (advanced_mode_requested && cfg.enforce_gpu_only_advanced_modes && !cfg.use_gpu) {
        throw std::invalid_argument(
            "swe2d_create: nonhydrostatic/coupled modes are GPU-only; set use_gpu=true");
    }

#ifdef BACKWATER_HAS_CUDA
    if (advanced_mode_requested && cfg.enforce_gpu_only_advanced_modes && !swe2d_gpu_available()) {
        throw std::runtime_error(
            "swe2d_create: advanced nonhydrostatic/coupled modes require CUDA-enabled runtime");
    }
#else
    if (advanced_mode_requested && cfg.enforce_gpu_only_advanced_modes) {
        throw std::runtime_error(
            "swe2d_create: advanced nonhydrostatic/coupled modes require CUDA build");
    }
#endif

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
    s->source_terms.assign(n, 0.0);
    s->external_source_terms.assign(n, 0.0);

    if (s->cfg.godunov_mode != 0) {
        s->cfg.temporal_order = std::max(s->cfg.temporal_order, 2);
        s->cfg.enable_shallow_front_recon_fallback = true;
    }

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
        if (s->dev) {
            const bool enable_cuda_graphs = swe2d_env_enabled("BACKWATER_ENABLE_CUDA_GRAPHS");
            swe2d_gpu_enable_kernel_graphs(s->dev, enable_cuda_graphs);
        }
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
    const double recon_fallback_depth = std::max(h_min, 0.5 * s->cfg.shallow_damping_depth);

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

            const bool shallow_pair = (hL < recon_fallback_depth) || (hR < recon_fallback_depth);
            const bool disable_higher_order = s->cfg.enable_shallow_front_recon_fallback && shallow_pair;
            if (use_higher_order && !disable_higher_order) {
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
                h_min,
                s->n_mann_cell[c0]);
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
        bool friction_applied_in_substeps = false;
        if (c < static_cast<int32_t>(s->source_terms.size())) {
            double src = s->source_terms[static_cast<size_t>(c)];
            int nsub = 1;
            if (src > 0.0) {
                if (s->cfg.source_rate_cap > 0.0 && src > s->cfg.source_rate_cap) {
                    src = s->cfg.source_rate_cap;
                }
                if (s->cfg.source_depth_step_cap > 0.0) {
                    const double src_step_cap = s->cfg.source_depth_step_cap / std::max(dt, 1.0e-12);
                    if (src > src_step_cap) src = src_step_cap;
                }
                if (s->cfg.extreme_rain_mode && s->cfg.source_cfl_beta > 0.0) {
                    const double h_ref = std::max(h_old, h_min);
                    const double dt_src = s->cfg.source_cfl_beta * h_ref / std::max(src, 1.0e-12);
                    if (dt_src < dt) {
                        nsub = std::max(1, static_cast<int>(std::ceil(dt / std::max(dt_src, 1.0e-12))));
                        if (s->cfg.source_max_substeps > 0) nsub = std::min(nsub, s->cfg.source_max_substeps);
                    }
                }
            }
            if (s->cfg.source_true_subcycling && nsub > 1 && src > 0.0) {
                const double dt_sub = dt / static_cast<double>(nsub);
                for (int k = 0; k < nsub; ++k) {
                    s->h[c] += dt_sub * src;
                    if (s->h[c] < 0.0) s->h[c] = 0.0;
                    if (s->cfg.source_imex_split && s->h[c] > h_min) {
                        double n_mann_sub = s->n_mann_cell[c];
                        swe2d::apply_friction(s->h[c], s->hu[c], s->hv[c],
                                              dt_sub, n_mann_sub, g, h_min);
                        friction_applied_in_substeps = true;
                    }
                }
            } else {
                if (s->cfg.extreme_rain_mode && nsub > 1 && src > 0.0) {
                    src *= (1.0 / static_cast<double>(nsub));
                }
                s->h[c] += dt * src;
            }
        }

        // Positivity enforcement
        if (s->h[c] < 0.0) s->h[c] = 0.0;
        if (s->h[c] < h_min) {
            s->hu[c] = 0.0;
            s->hv[c] = 0.0;
        }

        // Manning friction (semi-implicit)
        if (!friction_applied_in_substeps) {
            double n_mann = s->n_mann_cell[c];
            swe2d::apply_friction(s->h[c], s->hu[c], s->hv[c],
                                  dt, n_mann, g, h_min);
        }

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
    const double t_now = s->t;

    const bool use_nonhydrostatic_mode =
        (s->cfg.equation_set == static_cast<int>(SWE2DEquationSet::NONHYDROSTATIC_2D));
    const bool use_2d3d_coupling =
        (s->cfg.coupling_mode != static_cast<int>(SWE2DThreeDCouplingMode::OFF));
    if (use_nonhydrostatic_mode || use_2d3d_coupling) {
        throw std::runtime_error(
            "swe2d_step: nonhydrostatic/coupled GPU solver path is scaffolded but not implemented yet");
    }

    const bool use_godunov_rollout = (s->cfg.godunov_mode != 0);
    const bool use_rk4 = (s->cfg.temporal_order >= 4) && !use_godunov_rollout;
    const bool use_rk2 = ((s->cfg.temporal_order >= 2) || use_godunov_rollout) && !use_rk4;
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

        if (use_rk4) {
            SWE2DStepDiag diag;
            swe2d_gpu_step_rk4(s->dev, t_now, dt,
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
                               s->cfg.extreme_rain_mode,
                               s->cfg.source_cfl_beta,
                               s->cfg.source_max_substeps,
                               s->cfg.source_rate_cap,
                               s->cfg.source_depth_step_cap,
                               s->cfg.source_true_subcycling,
                               s->cfg.source_imex_split,
                               s->cfg.enable_shallow_front_recon_fallback,
                               sync_diag_this_step,
                               &diag,
                               s->cfg.front_flux_damping,
                               s->cfg.active_set_hysteresis);
            diag.gpu_active = true;
            s->t += dt;
            s->gpu_steps += 1;
            return diag;
        }

        if (!use_rk2) {
            SWE2DStepDiag diag;
            if (use_godunov_rollout) {
                swe2d_gpu_step_godunov_rollout(s->dev, t_now, dt,
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
                                               s->cfg.extreme_rain_mode,
                                               s->cfg.source_cfl_beta,
                                               s->cfg.source_max_substeps,
                                               s->cfg.source_rate_cap,
                                               s->cfg.source_depth_step_cap,
                                               s->cfg.source_true_subcycling,
                                               s->cfg.source_imex_split,
                                               s->cfg.enable_shallow_front_recon_fallback,
                                               sync_diag_this_step,
                                               &diag,
                                               s->cfg.front_flux_damping,
                                               s->cfg.active_set_hysteresis);
            } else {
                swe2d_gpu_step(s->dev, t_now, dt,
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
                               s->cfg.extreme_rain_mode,
                               s->cfg.source_cfl_beta,
                               s->cfg.source_max_substeps,
                               s->cfg.source_rate_cap,
                               s->cfg.source_depth_step_cap,
                               s->cfg.source_true_subcycling,
                               s->cfg.source_imex_split,
                               s->cfg.enable_shallow_front_recon_fallback,
                               sync_diag_this_step,
                               &diag,
                               s->cfg.front_flux_damping,
                               s->cfg.active_set_hysteresis);
            }
            diag.gpu_active = true;
            s->t += dt;
            s->gpu_steps += 1;
            return diag;
        }

        SWE2DStepDiag diag;
        if (use_godunov_rollout) {
            swe2d_gpu_step_rk2_godunov_rollout(s->dev, t_now, dt,
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
                                               s->cfg.extreme_rain_mode,
                                               s->cfg.source_cfl_beta,
                                               s->cfg.source_max_substeps,
                                               s->cfg.source_rate_cap,
                                               s->cfg.source_depth_step_cap,
                                               s->cfg.source_true_subcycling,
                                               s->cfg.source_imex_split,
                                               s->cfg.enable_shallow_front_recon_fallback,
                                               sync_diag_this_step,
                                               &diag,
                                               s->cfg.front_flux_damping,
                                               s->cfg.active_set_hysteresis);
        } else {
            swe2d_gpu_step_rk2(s->dev, t_now, dt,
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
                               s->cfg.extreme_rain_mode,
                               s->cfg.source_cfl_beta,
                               s->cfg.source_max_substeps,
                               s->cfg.source_rate_cap,
                               s->cfg.source_depth_step_cap,
                               s->cfg.source_true_subcycling,
                               s->cfg.source_imex_split,
                               s->cfg.enable_shallow_front_recon_fallback,
                               sync_diag_this_step,
                               &diag,
                               s->cfg.front_flux_damping,
                               s->cfg.active_set_hysteresis);
        }
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
        apply_solver_boundary_hydrographs(s, t_now);
        build_solver_rain_cn_source(s, t_now, t_now + dt);
        SWE2DStepDiag diag = swe2d_step_cpu(s, dt);
        s->t += dt;
        return diag;
    }

    std::vector<double> h0 = s->h;
    std::vector<double> hu0 = s->hu;
    std::vector<double> hv0 = s->hv;
    std::vector<double> rain_cum_stage1;
    std::vector<double> rain_excess_stage1;

    apply_solver_boundary_hydrographs(s, t_now);
    build_solver_rain_cn_source(s, t_now, t_now + dt);
    if (s->rain_cn_enabled) {
        rain_cum_stage1 = s->rain_cum_mm;
        rain_excess_stage1 = s->rain_excess_cum_mm;
    }
    swe2d_step_cpu(s, dt);
    apply_solver_boundary_hydrographs(s, t_now + dt);
    build_solver_rain_cn_source(s, t_now + dt, t_now + 2.0 * dt);
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

    // RK2 advances physical time by dt, so keep cumulative CN state at t+dt
    // (the second stage source evaluation should not commit cumulative totals).
    if (s->rain_cn_enabled &&
        rain_cum_stage1.size() == s->rain_cum_mm.size() &&
        rain_excess_stage1.size() == s->rain_excess_cum_mm.size()) {
        s->rain_cum_mm.swap(rain_cum_stage1);
        s->rain_excess_cum_mm.swap(rain_excess_stage1);
    }

    SWE2DStepDiag diag = summarize_state(s, dt, false, max_depth_residual);
    s->t += dt;
    return diag;
}

void swe2d_solver_set_boundary_values(
    SWE2DSolver* s,
    const int32_t* edge_index,
    const int32_t* bc_type,
    const double* bc_val,
    int32_t n_updates)
{
    if (!s || !edge_index || !bc_type || !bc_val || n_updates <= 0) return;
    SWE2DMesh& mesh = const_cast<SWE2DMesh&>(*s->mesh);
    for (int32_t i = 0; i < n_updates; ++i) {
        const int32_t e = edge_index[static_cast<size_t>(i)];
        if (e < 0 || e >= mesh.n_edges) continue;
        mesh.edge_bc[static_cast<size_t>(e)] = static_cast<BCType>(bc_type[static_cast<size_t>(i)]);
        mesh.edge_bc_val[static_cast<size_t>(e)] = bc_val[static_cast<size_t>(i)];
    }
#ifdef BACKWATER_HAS_CUDA
    if (s->dev) {
        swe2d_gpu_update_boundary_values(s->dev, edge_index, bc_type, bc_val, n_updates);
    }
#endif
}

void swe2d_solver_set_boundary_hydrographs(
    SWE2DSolver* s,
    const int32_t* edge_index,
    const int32_t* bc_type,
    const int32_t* offsets,
    const double* time_s,
    const double* value,
    int32_t n_edges,
    int32_t n_samples)
{
    if (!s) return;
    s->hg_edge_index.clear();
    s->hg_bc_type.clear();
    s->hg_offsets.clear();
    s->hg_time_s.clear();
    s->hg_value.clear();
    s->hydrographs_enabled = false;

    if (n_edges <= 0 || n_samples <= 0 || !edge_index || !bc_type || !offsets || !time_s || !value) {
#ifdef BACKWATER_HAS_CUDA
        if (s->dev) {
            swe2d_gpu_set_boundary_hydrographs(s->dev, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0);
        }
#endif
        return;
    }

    s->hg_edge_index.assign(edge_index, edge_index + n_edges);
    s->hg_bc_type.assign(bc_type, bc_type + n_edges);
    s->hg_offsets.assign(offsets, offsets + n_edges + 1);
    s->hg_time_s.assign(time_s, time_s + n_samples);
    s->hg_value.assign(value, value + n_samples);
    s->hydrographs_enabled = true;
#ifdef BACKWATER_HAS_CUDA
    if (s->dev) {
        swe2d_gpu_set_boundary_hydrographs(s->dev, edge_index, bc_type, offsets, time_s, value, n_edges, n_samples);
    }
#endif
}

void swe2d_solver_set_rain_cn_forcing(
    SWE2DSolver* s,
    const int32_t* cell_gage_idx,
    const int32_t* gage_offsets,
    const double* hg_time_s,
    const double* hg_cum_mm,
    const double* cn,
    int32_t n_cells,
    int32_t n_gages,
    int32_t n_samples,
    double ia_ratio,
    double mm_to_model_depth)
{
    if (!s) return;
    s->rain_cell_gage.clear();
    s->rain_gage_offsets.clear();
    s->rain_hg_time_s.clear();
    s->rain_hg_cum_mm.clear();
    s->rain_cn.clear();
    s->rain_cum_mm.clear();
    s->rain_excess_cum_mm.clear();
    s->rain_cn_enabled = false;
    s->rain_ia_ratio = ia_ratio;
    s->rain_mm_to_model_depth = (mm_to_model_depth > 0.0) ? mm_to_model_depth : 1.0e-3;

    if (n_cells <= 0 || n_gages <= 0 || n_samples <= 0 || !cell_gage_idx || !gage_offsets || !hg_time_s || !hg_cum_mm || !cn) {
#ifdef BACKWATER_HAS_CUDA
        if (s->dev) {
            swe2d_gpu_set_rain_cn_forcing(
                s->dev,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                0,
                0,
                0,
                   s->rain_ia_ratio,
                s->rain_mm_to_model_depth);
        }
#endif
        return;
    }

    s->rain_cell_gage.assign(cell_gage_idx, cell_gage_idx + n_cells);
    s->rain_gage_offsets.assign(gage_offsets, gage_offsets + n_gages + 1);
    s->rain_hg_time_s.assign(hg_time_s, hg_time_s + n_samples);
    s->rain_hg_cum_mm.assign(hg_cum_mm, hg_cum_mm + n_samples);
    s->rain_cn.assign(cn, cn + n_cells);
    s->rain_cum_mm.assign(static_cast<size_t>(n_cells), 0.0);
    s->rain_excess_cum_mm.assign(static_cast<size_t>(n_cells), 0.0);
    s->source_terms.assign(static_cast<size_t>(n_cells), 0.0);
    s->rain_cn_enabled = true;
#ifdef BACKWATER_HAS_CUDA
    if (s->dev) {
        swe2d_gpu_set_rain_cn_forcing(s->dev,
                                      cell_gage_idx,
                                      gage_offsets,
                                      hg_time_s,
                                      hg_cum_mm,
                                      cn,
                                      n_cells,
                                      n_gages,
                                      n_samples,
                                      ia_ratio,
                                      s->rain_mm_to_model_depth);
    }
#endif
}

void swe2d_solver_set_external_sources(
    SWE2DSolver* s,
    const double* source_mps,
    int32_t n_cells)
{
    if (!s) return;
    const int32_t nc = s->mesh->n_cells;
    s->external_sources_enabled = false;
    s->external_source_terms.assign(static_cast<size_t>(nc), 0.0);

    if (source_mps && n_cells == nc && n_cells > 0) {
        s->external_source_terms.assign(source_mps, source_mps + static_cast<size_t>(n_cells));
        s->external_sources_enabled = true;
    }

#ifdef BACKWATER_HAS_CUDA
    if (s->dev) {
        swe2d_gpu_set_external_sources(
            s->dev,
            s->external_sources_enabled ? s->external_source_terms.data() : nullptr,
            s->external_sources_enabled ? nc : 0);
    }
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// Native run-to-time loop
// ─────────────────────────────────────────────────────────────────────────────
int32_t swe2d_run_to_time(
    SWE2DSolver* s,
    const SWE2DRunConfig* cfg,
    SWE2DStepDiag* diag_out,
    int32_t max_diags)
{
    if (!s || !cfg) throw std::invalid_argument("swe2d_run_to_time: null solver or config");

    int32_t diag_count = 0;
    uint64_t step_count = 0;
    double t = s->t;
    const double t_end = cfg->t_end;
    const double dt_request = cfg->dt_request;
    const int progress_interval = cfg->progress_callback_interval_steps;
    SWE2DProgressCallback progress_cb = cfg->progress_cb;
    const int batch_size = cfg->diag_batch_size;

    while (t < t_end) {
        SWE2DStepDiag diag = swe2d_step(s, dt_request);
        t += diag.dt;
        ++step_count;

        // Store diagnostics if batching is enabled and we have space.
        if (batch_size > 0 && diag_count < max_diags) {
            diag_out[diag_count++] = diag;
        }

        // Call progress callback at specified interval.
        if (progress_interval > 0 && (step_count % static_cast<uint64_t>(progress_interval)) == 0u) {
            if (progress_cb != nullptr) {
                bool should_continue = progress_cb(t, step_count, &diag);
                if (!should_continue) {
                    // Cancellation requested; return negative count to signal partial run.
                    return -static_cast<int32_t>(step_count);
                }
            }
        }
    }

    return diag_count;
}

