// swe2d_gpu.cu
// CUDA kernel implementations for the 2D SWE hybrid solver.
//
// Three kernel launches per timestep:
//   1. swe2d_flux_kernel   — parallel over edges, writes flux accumulators
//   2. swe2d_update_kernel — parallel over cells, applies fluxes + friction
//   3. swe2d_cfl_kernel    — parallel over cells, block-reduce to find max lambda
//
// CUDA hot-path numerics are implemented locally in this translation unit to
// keep GPU optimization decoupled from the CPU fallback implementation.

#include "swe2d_gpu.cuh"
#include "swe2d_units.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

#include <cmath>
#include <cstring>
#include <stdexcept>

// Forward declarations for file-scope globals used by culvert/coupling paths.
extern int32_t s_culvert_solver_mode;
extern double* s_culvert_table_header;
extern double* s_culvert_table_data;
extern int32_t s_culvert_table_n_hw;
extern int32_t s_culvert_table_n_tw;
extern SWE2DDeviceState* s_coupling_dev;

// Manning unit-conversion constant stored in GPU constant memory.
// k_mann = 1.0   for SI units (meters)
// k_mann = 1.486 for US Customary units (feet)
__constant__ double c_k_mann = 1.0;

// Friction sub-stepping and shallow-correction constants.
// Set via swe2d_gpu_set_friction_config(); accessed by all kernels.
__constant__ int    c_friction_substep_enabled     = 1;
__constant__ double c_friction_target_courant      = 1.0;
__constant__ int    c_friction_max_substeps        = 64;
__constant__ int    c_shallow_friction_correction  = 0;
__constant__ double c_shallow_friction_depth_alpha = 5.0;
__constant__ double c_shallow_friction_exponent    = 0.4;

/// Host-side setter for the GPU constant-memory k_mann value.
/// Must be called before any step that uses friction.
void swe2d_gpu_set_k_mann(double k_mann) {
    cudaMemcpyToSymbol(c_k_mann, &k_mann, sizeof(double));
}

/// Host-side setter for GPU constant-memory friction configuration.
/// Must be called after swe2d_gpu_init() and before any step call.
void swe2d_gpu_set_friction_config(
    bool   substep_enabled,
    double target_courant,
    int    max_substeps,
    bool   shallow_correction,
    double depth_alpha,
    double exponent)
{
    int v;
    v = substep_enabled ? 1 : 0;
    cudaMemcpyToSymbol(c_friction_substep_enabled, &v, sizeof(int));
    cudaMemcpyToSymbol(c_friction_target_courant, &target_courant, sizeof(double));
    cudaMemcpyToSymbol(c_friction_max_substeps, &max_substeps, sizeof(int));
    v = shallow_correction ? 1 : 0;
    cudaMemcpyToSymbol(c_shallow_friction_correction, &v, sizeof(int));
    cudaMemcpyToSymbol(c_shallow_friction_depth_alpha, &depth_alpha, sizeof(double));
    cudaMemcpyToSymbol(c_shallow_friction_exponent, &exponent, sizeof(double));
}

#include <limits>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <mutex>

namespace cg = cooperative_groups;

namespace {

constexpr int SWE2D_GRAPH_STAGE_SLOTS = 6;

inline int32_t* swe2d_stage_edge_bc_slot(SWE2DDeviceState* dev, int slot) {
    return dev->d_stage_edge_bc + static_cast<size_t>(slot) * static_cast<size_t>(dev->n_edges);
}

inline double* swe2d_stage_edge_bc_val_slot(SWE2DDeviceState* dev, int slot) {
    return dev->d_stage_edge_bc_val + static_cast<size_t>(slot) * static_cast<size_t>(dev->n_edges);
}

inline double* swe2d_stage_source_slot(SWE2DDeviceState* dev, int slot) {
    return dev->d_stage_cell_source_mps + static_cast<size_t>(slot) * static_cast<size_t>(dev->n_cells);
}

struct GhostStateLocal {
    double h;
    double hu;
    double hv;
    double zb;
};

inline uint64_t swe2d_mix_u64(uint64_t h, uint64_t v) {
    h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    return h;
}

inline uint64_t swe2d_u64_from_double(double v) {
    uint64_t bits = 0;
    std::memcpy(&bits, &v, sizeof(double));
    return bits;
}

inline uint64_t swe2d_kernel_graph_signature(
    double dt,
    double g,
    double h_min,
    double cfl_lambda_cap,
    double max_inv_area,
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
    double front_flux_damping,
    bool use_culvert_face_flux = false)
{
    uint64_t h = 1469598103934665603ULL;
    h = swe2d_mix_u64(h, swe2d_u64_from_double(dt));
    h = swe2d_mix_u64(h, swe2d_u64_from_double(g));
    h = swe2d_mix_u64(h, swe2d_u64_from_double(h_min));
    h = swe2d_mix_u64(h, swe2d_u64_from_double(cfl_lambda_cap));
    h = swe2d_mix_u64(h, swe2d_u64_from_double(max_inv_area));
    h = swe2d_mix_u64(h, swe2d_u64_from_double(momentum_cap_min_speed));
    h = swe2d_mix_u64(h, swe2d_u64_from_double(momentum_cap_celerity_mult));
    h = swe2d_mix_u64(h, swe2d_u64_from_double(depth_cap));
    h = swe2d_mix_u64(h, swe2d_u64_from_double(max_rel_depth_increase));
    h = swe2d_mix_u64(h, swe2d_u64_from_double(shallow_damping_depth));
    h = swe2d_mix_u64(h, static_cast<uint64_t>(extreme_rain_mode ? 1 : 0));
    h = swe2d_mix_u64(h, swe2d_u64_from_double(source_cfl_beta));
    h = swe2d_mix_u64(h, static_cast<uint64_t>(source_max_substeps));
    h = swe2d_mix_u64(h, swe2d_u64_from_double(source_rate_cap));
    h = swe2d_mix_u64(h, swe2d_u64_from_double(source_depth_step_cap));
    h = swe2d_mix_u64(h, static_cast<uint64_t>(source_true_subcycling ? 1 : 0));
    h = swe2d_mix_u64(h, static_cast<uint64_t>(source_imex_split ? 1 : 0));
    h = swe2d_mix_u64(h, static_cast<uint64_t>(enable_shallow_front_recon_fallback ? 1 : 0));
    h = swe2d_mix_u64(h, swe2d_u64_from_double(front_flux_damping));
    h = swe2d_mix_u64(h, static_cast<uint64_t>(use_culvert_face_flux ? 1 : 0));
    return h;
}

struct ReconstructedStatesLocal {
    double hL_star;
    double uL;
    double vL;
    double hR_star;
    double uR;
    double vR;
    double zb_face;
};

__device__ __forceinline__ double vel_u_cuda_local(double hu, double h, double h_min) {
    return (h > h_min) ? (hu / h) : 0.0;
}

__device__ __forceinline__ double vel_v_cuda_local(double hv, double h, double h_min) {
    return (h > h_min) ? (hv / h) : 0.0;
}

__device__ __forceinline__ double celerity_cuda_local(double h, double g) {
    return (h > 0.0) ? ::sqrt(g * h) : 0.0;
}

__device__ __forceinline__ ReconstructedStatesLocal hydrostatic_reconstruct_cuda_local(
    double hL,  double huL, double hvL, double zbL,
    double hR,  double huR, double hvR, double zbR,
    double h_min)
{
    ReconstructedStatesLocal rs;
    const double etaL = hL + zbL;
    const double etaR = hR + zbR;
    rs.zb_face = (zbL > zbR) ? zbL : zbR;
    rs.hL_star = (etaL > rs.zb_face) ? (etaL - rs.zb_face) : 0.0;
    rs.hR_star = (etaR > rs.zb_face) ? (etaR - rs.zb_face) : 0.0;
    rs.uL = vel_u_cuda_local(huL, hL, h_min);
    rs.vL = vel_v_cuda_local(hvL, hL, h_min);
    rs.uR = vel_u_cuda_local(huR, hR, h_min);
    rs.vR = vel_v_cuda_local(hvR, hR, h_min);
    return rs;
}

__device__ __forceinline__ void bed_slope_correction_cuda_local(
    double hL, double hL_star,
    double nx, double ny, double g,
    double& corr_hu, double& corr_hv)
{
    const double dp = 0.5 * g * (hL_star * hL_star - hL * hL);
    corr_hu -= dp * nx;
    corr_hv -= dp * ny;
}

__device__ __forceinline__ GhostStateLocal make_ghost_cuda_local(
    double hI,  double huI, double hvI, double zbI,
    double nx,  double ny,
    int bc_type,
    double bc_val,
    double h_min,
    double n_mann)
{
    GhostStateLocal g{};
    g.zb = zbI;

    switch (bc_type) {
        case 1:
        case 5: {
            g.h = hI;
            const double un = huI * nx + hvI * ny;
            g.hu = huI - 2.0 * un * nx;
            g.hv = hvI - 2.0 * un * ny;
            break;
        }
        case 2:
            g.h = hI;
            g.hu = -bc_val * nx;
            g.hv = -bc_val * ny;
            break;
        case 3: {
            const double h_ghost = bc_val - zbI;
            g.h = (h_ghost > h_min) ? h_ghost : h_min;
            g.hu = huI;
            g.hv = hvI;
            break;
        }
        case 4:
            g.h = hI;
            g.hu = huI;
            g.hv = hvI;
            break;
        case 6:
            g.h = (bc_val > h_min) ? bc_val : h_min;
            g.hu = huI;
            g.hv = hvI;
            break;
        case 7: {
            const double sf = fmax(fabs(bc_val), 1.0e-8);
            const double qn = huI * nx + hvI * ny;
            const double qmag = fabs(qn);
            if (qmag <= 1.0e-12) {
                g.h = (hI > h_min) ? hI : h_min;
            } else {
                const double n_eff = fmax(fabs(n_mann), 1.0e-6);
                const double h_nd = pow((qmag * n_eff) / sqrt(sf), 3.0 / 5.0);
                g.h = (h_nd > h_min) ? h_nd : h_min;
            }
            g.hu = huI;
            g.hv = hvI;
            break;
        }
        default:
            g.h = hI;
            g.hu = huI;
            g.hv = hvI;
            break;
    }
    return g;
}

__device__ __forceinline__ void apply_friction_cuda_local(
    double& h, double& hu, double& hv,
    double dt, double n_mann, double g, double h_min)
{
    if (h <= h_min) {
        hu = 0.0;
        hv = 0.0;
        return;
    }
    const double k2 = c_k_mann * c_k_mann;
    // Regularize shallow-cell friction stiffness to avoid large Cf spikes
    // right above h_min at advancing wet/dry fronts.
    const double h_fric = fmax(h, 4.0 * h_min);
    const double h43 = ::pow(h_fric, 4.0 / 3.0);
    double cf = (h43 > 0.0) ? (g * n_mann * n_mann / (k2 * h43)) : 0.0;

    // Shallow-flow depth correction (Keulegan-based Cf enhancement).
    if (c_shallow_friction_correction != 0 && cf > 0.0) {
        const double h_ref = c_shallow_friction_depth_alpha * ::pow(n_mann, 1.5);
        if (h_fric < h_ref) {
            cf *= ::pow(h_ref / h_fric, c_shallow_friction_exponent);
        }
    }

    // Adaptive sub-stepping for temporal-order hardening.
    int n_sub = 1;
    if (c_friction_substep_enabled != 0 && c_friction_target_courant > 0.0) {
        const double u = hu / h;
        const double v = hv / h;
        const double spd = ::sqrt(u * u + v * v);
        const double nu_fric = dt * cf * spd;
        n_sub = max(1, min(c_friction_max_substeps,
                           static_cast<int>(::ceil(nu_fric / c_friction_target_courant))));
    }

    const double dt_sub = dt / static_cast<double>(n_sub);
    for (int k = 0; k < n_sub; ++k) {
        const double u_k = hu / h;
        const double v_k = hv / h;
        const double spd_k = ::sqrt(u_k * u_k + v_k * v_k);
        const double denom = 1.0 + dt_sub * cf * spd_k;
        hu /= denom;
        hv /= denom;
    }
}

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

// ─────────────────────────────────────────────────────────────────────────────
// Wet/dry classification and active-set marking
// ─────────────────────────────────────────────────────────────────────────────

// swe2d_classify_kernel: standalone classification (no mark-neighbor).
// Retained for backward compatibility with persistent-chunk / RK step functions.
__global__ __launch_bounds__(256, 4) void swe2d_classify_kernel(
    int32_t                     n_cells,
    const double*  __restrict__ d_h,
    const double*  __restrict__ d_cell_source_mps,
    const double*  __restrict__ d_external_source_mps,
    const double*  __restrict__ d_ext_struct_flux_h,  // nullable: face-based culvert flux
    const int32_t* __restrict__ d_bc_forced,
    int32_t*                    d_active,
    int32_t*                    d_n_wet,
    double                      h_min,
    const int32_t* __restrict__ d_was_active)
{
    extern __shared__ int32_t scount[];
    int32_t tid = threadIdx.x;
    int32_t c   = blockIdx.x * blockDim.x + tid;

    int32_t wet = 0;
    if (c < n_cells) {
        const int32_t forced = d_bc_forced  ? d_bc_forced[c]  : 0;
        const int32_t w      = (d_h[c] > h_min) ? 1 : 0;
        const double src_rain = d_cell_source_mps ? d_cell_source_mps[c] : 0.0;
        const double src_ext  = d_external_source_mps ? d_external_source_mps[c] : 0.0;
        // Both positive (incoming) and negative (outgoing) face flux keeps
        // the cell active — a culvert donor cell losing water must stay
        // active so the edge-flux Riemann solver communicates its drawdown
        // to upstream neighbors.  Otherwise the drawdown can't propagate.
        const double src_ff   = d_ext_struct_flux_h ? fabs(d_ext_struct_flux_h[c]) : 0.0;
        const double src      = src_rain + src_ext + src_ff;
        const int32_t src_on  = (isfinite(src) && src > 0.0) ? 1 : 0;
        const int32_t grace  = (d_was_active && d_was_active[c] && d_h[c] > 0.0) ? 1 : 0;
        d_active[c] = w | forced | grace | src_on;
        wet         = w;
    }

    scount[tid] = wet;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) scount[tid] += scount[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(d_n_wet, scount[0]);
}

// swe2d_mark_neighbor_kernel: retained for backward compatibility.
__global__ __launch_bounds__(256, 4) void swe2d_mark_neighbor_kernel(
    int32_t                     n_edges,
    const int32_t* __restrict__ edge_c0,
    const int32_t* __restrict__ edge_c1,
    int32_t*                    d_active)
{
    int32_t e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_edges) return;
    int32_t c0 = edge_c0[e];
    int32_t c1 = edge_c1[e];
    if (c1 < 0) return;
    if (d_active[c0] && !d_active[c1]) atomicOr(&d_active[c1], 1);
    if (d_active[c1] && !d_active[c0]) atomicOr(&d_active[c0], 1);
}

// swe2d_classify_and_mark_kernel: fused classify + 1-hop neighbor marking.
// One thread per cell for classification; after block-level sync, each block
// also processes a contiguous segment of edges to mark dry neighbors of active
// cells.  This eliminates a separate kernel launch for the mark-neighbor pass.
__global__ __launch_bounds__(256, 4) void swe2d_classify_and_mark_kernel(
    int32_t                     n_cells,
    int32_t                     n_edges,
    const double*  __restrict__ d_h,
    const double*  __restrict__ d_cell_source_mps,
    const double*  __restrict__ d_external_source_mps,
    const double*  __restrict__ d_ext_struct_flux_h,  // nullable: face-based culvert flux
    const int32_t* __restrict__ d_bc_forced,
    const int32_t* __restrict__ edge_c0,
    const int32_t* __restrict__ edge_c1,
    int32_t*                    d_active,
    int32_t*                    d_n_wet,
    double                      h_min,
    const int32_t* __restrict__ d_was_active)  // nullable: previous-step active set for 1-step hysteresis
{
    extern __shared__ int32_t scount[];
    int32_t tid = threadIdx.x;
    int32_t c   = blockIdx.x * blockDim.x + tid;

    // ── Pass 1: classify ──────────────────────────────────────────────────
    int32_t wet = 0;
    if (c < n_cells) {
        const int32_t forced = d_bc_forced  ? d_bc_forced[c]  : 0;
        const int32_t w      = (d_h[c] > h_min) ? 1 : 0;
        const double src_rain = d_cell_source_mps ? d_cell_source_mps[c] : 0.0;
        const double src_ext  = d_external_source_mps ? d_external_source_mps[c] : 0.0;
        // Both positive (incoming) and negative (outgoing) face flux keeps
        // the cell active — a culvert donor cell losing water must stay
        // active so the edge-flux Riemann solver communicates its drawdown
        // to upstream neighbors.  Otherwise the drawdown can't propagate.
        const double src_ff   = d_ext_struct_flux_h ? fabs(d_ext_struct_flux_h[c]) : 0.0;
        const double src      = src_rain + src_ext + src_ff;
        const int32_t src_on  = (isfinite(src) && src > 0.0) ? 1 : 0;
        // Hysteretic wetting: cells that were active last step and still carry
        // non-zero depth stay active for one additional step.
        const int32_t grace  = (d_was_active && d_was_active[c] && d_h[c] > 0.0) ? 1 : 0;
        d_active[c] = w | forced | grace | src_on;
        wet         = w;
    }

    scount[tid] = wet;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) scount[tid] += scount[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(d_n_wet, scount[0]);

    // ── Pass 2: mark neighbors (edge-parallel within block segment) ───────
    __syncthreads();
    const int32_t grid_sz = gridDim.x;
    const int32_t edges_per_block = (n_edges + grid_sz - 1) / grid_sz;
    const int32_t e_start = static_cast<int32_t>(blockIdx.x) * edges_per_block;
    const int32_t e_end   = (e_start + edges_per_block < n_edges) ? (e_start + edges_per_block) : n_edges;

    for (int32_t e = e_start + tid; e < e_end; e += blockDim.x) {
        int32_t c0 = edge_c0[e];
        int32_t c1 = edge_c1[e];
        if (c1 < 0) continue;   // boundary edge — no second interior cell
        int32_t a0 = d_active[c0];
        int32_t a1 = d_active[c1];
        if (a0 && !a1) atomicOr(&d_active[c1], 1);
        if (a1 && !a0) atomicOr(&d_active[c0], 1);
    }
}

// swe2d_degen_deactivate_kernel: one thread per cell.
// Modes 1 (skip) and 3 (merge): force degenerate cells permanently inactive
// after each classify pass so they never receive flux or get updated.
__global__ void swe2d_degen_deactivate_kernel(
    int32_t                     n_cells,
    const int32_t* __restrict__ d_degen_mask,
    int32_t*                    d_active)
{
    int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells || !d_degen_mask[c]) return;
    d_active[c] = 0;
}

__global__ void swe2d_collect_active_edges_kernel(
    int32_t n_edges,
    const int32_t* __restrict__ edge_c0,
    const int32_t* __restrict__ edge_c1,
    const int32_t* __restrict__ d_active,
    int32_t* __restrict__ d_active_edge_ids,
    int32_t* __restrict__ d_n_active_edges)
{
    int32_t e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_edges) return;

    const int32_t c0 = edge_c0[e];
    const int32_t c1 = edge_c1[e];
    bool edge_active = false;
    if (c1 >= 0) {
        edge_active = (d_active[c0] != 0) || (d_active[c1] != 0);
    } else {
        edge_active = (d_active[c0] != 0);
    }
    if (!edge_active) return;

    const int32_t out = atomicAdd(d_n_active_edges, 1);
    d_active_edge_ids[out] = e;
}

// swe2d_degen_sync_kernel: one thread per degenerate cell.
// Mode 3 (merge): copy owner state into each degenerate cell so that
// flux computation and higher-order reconstruction sees physically sane values.
__global__ void swe2d_degen_sync_kernel(
    int32_t                     n_cells,
    const int32_t* __restrict__ d_degen_mask,
    const int32_t* __restrict__ d_merge_owner,
    double*                     d_h,
    double*                     d_hu,
    double*                     d_hv)
{
    int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells || !d_degen_mask[c]) return;
    const int32_t owner = d_merge_owner[c];
    if (owner < 0 || owner >= n_cells) {
        d_h[c] = 0.0; d_hu[c] = 0.0; d_hv[c] = 0.0;
    } else {
        d_h[c]  = d_h[owner];
        d_hu[c] = d_hu[owner];
        d_hv[c] = d_hv[owner];
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

static inline void swe2d_ensure_cfl_block_workspace(
    SWE2DDeviceState* dev,
    int32_t n_blocks)
{
    if (!dev || n_blocks <= 0) return;
    if (!dev->d_cfl_block_max || dev->cfl_block_capacity < n_blocks) {
        if (dev->d_cfl_block_max) {
            CUDA_CHECK(cudaFree(dev->d_cfl_block_max));
            dev->d_cfl_block_max = nullptr;
        }
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev->d_cfl_block_max),
                              static_cast<size_t>(n_blocks) * sizeof(double)));
        dev->cfl_block_capacity = n_blocks;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Double-precision atomicAdd via CAS loop (used by edge-centric gradient kernel).
// ─────────────────────────────────────────────────────────────────────────────
__device__ inline void atomicAddDouble(double* address, double val) {
    unsigned long long int* addr_as_ull = reinterpret_cast<unsigned long long int*>(address);
    unsigned long long int old = *addr_as_ull;
    while (true) {
        double new_val = __longlong_as_double(static_cast<long long int>(old)) + val;
        unsigned long long int assumed = old;
        old = atomicCAS(addr_as_ull, assumed,
                        static_cast<unsigned long long int>(__double_as_longlong(new_val)));
        if (old == assumed) break;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 0 (optional): Green-Gauss gradient estimation — one thread per EDGE.
// Each thread computes the face-average * outward-normal * len contribution
// and atomically adds it to both incident cells' gradient accumulators.
// This halves the work compared to the cell-centric version (each edge was
// visited twice, once from each incident cell's perspective).
// ─────────────────────────────────────────────────────────────────────────────
__global__ __launch_bounds__(256, 4) void swe2d_gradient_kernel(
    int32_t                     n_edges,
    const int32_t* __restrict__ edge_c0,
    const int32_t* __restrict__ edge_c1,
    const double*  __restrict__ edge_nx,
    const double*  __restrict__ edge_ny,
    const double*  __restrict__ edge_len,
    const double*  __restrict__ cell_h,
    const double*  __restrict__ cell_zb,
    const double*  __restrict__ cell_hu,
    const double*  __restrict__ cell_hv,
    const double*  __restrict__ cell_inv_area,
    double                      max_inv_area,
    double*                     grad_hx,  double* grad_hy,
    double*                     grad_hux, double* grad_huy,
    double*                     grad_hvx, double* grad_hvy,
    const int32_t* __restrict__ d_active,
    const int32_t* __restrict__ d_degen_mask,
    int                         degen_mode)
{
    int32_t e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_edges) return;

    const int32_t c0 = edge_c0[e];
    const int32_t c1 = edge_c1[e];

    // Skip boundary edges and edges where both cells are inactive/degenerate.
    const bool c0_active = (!d_active || d_active[c0]) && (!d_degen_mask || !d_degen_mask[c0] || degen_mode == 2);
    const bool c1_valid  = (c1 >= 0);
    const bool c1_active = c1_valid && (!d_active || d_active[c1]) && (!d_degen_mask || !d_degen_mask[c1] || degen_mode == 2);
    if (!c0_active && !c1_active) return;

    const double nx = edge_nx[e];
    const double ny = edge_ny[e];
    const double len = edge_len[e];

    const double h_c0  = cell_h[c0];
    const double zb_c0 = cell_zb[c0];
    const double hu_c0 = cell_hu[c0];
    const double hv_c0 = cell_hv[c0];

    double h_c1  = h_c0;
    double zb_c1 = zb_c0;
    double hu_c1 = hu_c0;
    double hv_c1 = hv_c0;
    if (c1_valid) {
        h_c1  = cell_h[c1];
        zb_c1 = cell_zb[c1];
        hu_c1 = cell_hu[c1];
        hv_c1 = cell_hv[c1];
    }

    const double eta0 = h_c0 + zb_c0;
    const double eta1 = h_c1 + zb_c1;
    const double qh  = 0.5 * (eta0 + eta1);
    const double qhu = 0.5 * (hu_c0 + hu_c1);
    const double qhv = 0.5 * (hv_c0 + hv_c1);

    // Contribute to c0 (outward normal = +nx, +ny from c0's perspective).
    if (c0_active) {
        const double ia0 = fmin(fmax(cell_inv_area[c0], 1.0 / fmax(max_inv_area, 1.0)), fmax(max_inv_area, 1.0));
        const double w = len * ia0;
        // Using atomicAdd for double: not natively supported on all archs.
        // We use atomicCAS-based double atomicAdd (implemented below).
        atomicAddDouble(&grad_hx[c0],  qh  * nx * w);
        atomicAddDouble(&grad_hy[c0],  qh  * ny * w);
        atomicAddDouble(&grad_hux[c0], qhu * nx * w);
        atomicAddDouble(&grad_huy[c0], qhu * ny * w);
        atomicAddDouble(&grad_hvx[c0], qhv * nx * w);
        atomicAddDouble(&grad_hvy[c0], qhv * ny * w);
    }

    // Contribute to c1 (outward normal = -nx, -ny from c1's perspective).
    if (c1_active) {
        const double ia1 = fmin(fmax(cell_inv_area[c1], 1.0 / fmax(max_inv_area, 1.0)), fmax(max_inv_area, 1.0));
        const double w = len * ia1;
        atomicAddDouble(&grad_hx[c1],  qh  * -nx * w);
        atomicAddDouble(&grad_hy[c1],  qh  * -ny * w);
        atomicAddDouble(&grad_hux[c1], qhu * -nx * w);
        atomicAddDouble(&grad_huy[c1], qhu * -ny * w);
        atomicAddDouble(&grad_hvx[c1], qhv * -nx * w);
        atomicAddDouble(&grad_hvy[c1], qhv * -ny * w);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Least-squares (2-ring) gradient kernel — spatial scheme 6 (FV_WENO5).
//
// Cell-parallel (one thread per cell), no atomics: each thread solves a 2×2
// weighted normal-equations system over its 2-ring neighbour set and writes the
// gradient of η = h + z_b, hu, hv into the SAME grad_* arrays used by the
// Green-Gauss path.  This kernel is launched AFTER swe2d_gradient_kernel for
// scheme 6, so it overwrites the GG gradient for cells with a well-posed
// stencil.  Cells with fewer than 3 ring-2 neighbours return early WITHOUT
// writing, leaving the Green-Gauss gradient in place as an automatic fallback.
// ─────────────────────────────────────────────────────────────────────────────
__global__ __launch_bounds__(256, 4) void swe2d_lsq_gradient_kernel(
    int32_t                     n_cells,
    const int32_t* __restrict__ cell_ring2_offsets,
    const int32_t* __restrict__ cell_ring2_ids,
    const double*  __restrict__ cell_ring2_dcx,
    const double*  __restrict__ cell_ring2_dcy,
    const double*  __restrict__ cell_ring2_inv_dist2,
    const double*  __restrict__ cell_h,
    const double*  __restrict__ cell_zb,
    const double*  __restrict__ cell_hu,
    const double*  __restrict__ cell_hv,
    const int32_t* __restrict__ d_active,
    double*                     grad_hx,  double* grad_hy,
    double*                     grad_hux, double* grad_huy,
    double*                     grad_hvx, double* grad_hvy)
{
    int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells) return;
    if (d_active && !d_active[c]) return;

    const int32_t s = cell_ring2_offsets[c];
    const int32_t e = cell_ring2_offsets[c + 1];

    // Degenerate / under-determined stencil: keep the Green-Gauss gradient.
    if (e - s < 3) return;

    const double eta0 = cell_h[c] + cell_zb[c];
    const double hu0  = cell_hu[c];
    const double hv0  = cell_hv[c];

    // Weighted normal-equations accumulators for the 2×2 LSQ system.
    double a11 = 0.0, a12 = 0.0, a22 = 0.0;
    double b1_eta = 0.0, b2_eta = 0.0;
    double b1_hu  = 0.0, b2_hu  = 0.0;
    double b1_hv  = 0.0, b2_hv  = 0.0;

    for (int32_t k = s; k < e; ++k) {
        const int32_t j  = cell_ring2_ids[k];
        const double  dx = cell_ring2_dcx[k];
        const double  dy = cell_ring2_dcy[k];
        const double  w  = cell_ring2_inv_dist2[k];   // 1/|Δr|²

        const double d_eta = (cell_h[j] + cell_zb[j]) - eta0;
        const double d_hu  = cell_hu[j] - hu0;
        const double d_hv  = cell_hv[j] - hv0;

        const double wdx = w * dx;
        const double wdy = w * dy;
        a11 += wdx * dx;
        a12 += wdx * dy;
        a22 += wdy * dy;
        b1_eta += wdx * d_eta;  b2_eta += wdy * d_eta;
        b1_hu  += wdx * d_hu;   b2_hu  += wdy * d_hu;
        b1_hv  += wdx * d_hv;   b2_hv  += wdy * d_hv;
    }

    // Solve via Cramer's rule.  Near-singular systems fall back to GG.
    const double det = a11 * a22 - a12 * a12;
    if (fabs(det) <= 1.0e-30) return;
    const double inv_det = 1.0 / det;

    grad_hx[c]  = inv_det * (a22 * b1_eta - a12 * b2_eta);
    grad_hy[c]  = inv_det * (a11 * b2_eta - a12 * b1_eta);
    grad_hux[c] = inv_det * (a22 * b1_hu  - a12 * b2_hu);
    grad_huy[c] = inv_det * (a11 * b2_hu  - a12 * b1_hu);
    grad_hvx[c] = inv_det * (a22 * b1_hv  - a12 * b2_hv);
    grad_hvy[c] = inv_det * (a11 * b2_hv  - a12 * b1_hv);
}

// Launch the LSQ 2-ring gradient kernel for scheme 6, overwriting the
// Green-Gauss gradient that was just computed into dev->d_grad_*.  No-op for
// any other scheme or when the 2-ring stencil is unavailable.  Must be called
// inside any active CUDA-graph capture region (it issues onto dev->d_stream).
static inline void swe2d_maybe_launch_lsq_gradient(
    SWE2DDeviceState* dev, int spatial_scheme, int32_t n_cells, int block,
    const double* cell_h, const double* cell_hu, const double* cell_hv)
{
    if (spatial_scheme != static_cast<int>(SWE2DSpatialScheme::FV_WENO5)) return;
    if (dev->d_cell_ring2_offsets == nullptr) return;
    const int grid = (n_cells + block - 1) / block;
    swe2d_lsq_gradient_kernel<<<grid, block, 0, dev->d_stream>>>(
        n_cells,
        dev->d_cell_ring2_offsets, dev->d_cell_ring2_ids,
        dev->d_cell_ring2_dcx, dev->d_cell_ring2_dcy, dev->d_cell_ring2_inv_dist2,
        cell_h, dev->d_cell_zb, cell_hu, cell_hv,
        dev->d_active,
        dev->d_grad_hx,  dev->d_grad_hy,
        dev->d_grad_hux, dev->d_grad_huy,
        dev->d_grad_hvx, dev->d_grad_hvy);
}

__device__ __forceinline__ double interp_series_clamped_cuda(
    const double* __restrict__ t,
    const double* __restrict__ v,
    int32_t start,
    int32_t end,
    double x)
{
    const int32_t n = end - start;
    if (n <= 0) return 0.0;
    if (n == 1) return v[start];
    if (x <= t[start]) return v[start];
    if (x >= t[end - 1]) return v[end - 1];

    int32_t lo = start;
    int32_t hi = end - 1;
    while (hi - lo > 1) {
        const int32_t mid = (lo + hi) >> 1;
        if (x < t[mid]) hi = mid;
        else lo = mid;
    }
    const double t0 = t[lo];
    const double t1 = t[hi];
    const double y0 = v[lo];
    const double y1 = v[hi];
    const double a = (x - t0) / fmax(t1 - t0, 1.0e-12);
    return y0 + a * (y1 - y0);
}

__device__ __forceinline__ double interp_series_slope_clamped_cuda(
    const double* __restrict__ t,
    const double* __restrict__ v,
    int32_t start,
    int32_t end,
    double x)
{
    if (end - start <= 1) return 0.0;
    if (x <= t[start]) {
        const double dt = fmax(t[start + 1] - t[start], 1.0e-12);
        return (v[start + 1] - v[start]) / dt;
    }
    if (x >= t[end - 1]) {
        const double dt = fmax(t[end - 1] - t[end - 2], 1.0e-12);
        return (v[end - 1] - v[end - 2]) / dt;
    }

    int32_t lo = start;
    int32_t hi = end - 1;
    while (hi - lo > 1) {
        const int32_t mid = (lo + hi) >> 1;
        if (x < t[mid]) hi = mid;
        else lo = mid;
    }
    const double dt = fmax(t[hi] - t[lo], 1.0e-12);
    return (v[hi] - v[lo]) / dt;
}

__global__ void swe2d_apply_hydrograph_bc_kernel(
    int32_t n_hg_edges,
    const int32_t* __restrict__ hg_edge_index,
    const int32_t* __restrict__ hg_bc_type,
    const int32_t* __restrict__ hg_offsets,
    const double*  __restrict__ hg_time_s,
    const double*  __restrict__ hg_value,
    int32_t* __restrict__ edge_bc,
    double*  __restrict__ edge_bc_val,
    double t_now)
{
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_hg_edges) return;
    const int32_t e = hg_edge_index[i];
    const int32_t s = hg_offsets[i];
    const int32_t eoff = hg_offsets[i + 1];
    edge_bc[e] = hg_bc_type[i];
    edge_bc_val[e] = interp_series_clamped_cuda(hg_time_s, hg_value, s, eoff, t_now);
}

__global__ void swe2d_apply_boundary_updates_kernel(
    int32_t n,
    const int32_t* __restrict__ upd_edge,
    const int32_t* __restrict__ upd_type,
    const double*  __restrict__ upd_val,
    int32_t* __restrict__ edge_bc,
    double*  __restrict__ edge_bc_val)
{
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int32_t e = upd_edge[i];
    edge_bc[e] = upd_type[i];
    edge_bc_val[e] = upd_val[i];
}

__global__ void swe2d_build_rain_cn_source_kernel(
    int32_t n_cells,
    const int32_t* __restrict__ cell_gage_idx,
    const int32_t* __restrict__ hg_offsets,
    const double*  __restrict__ hg_time_s,
    const double*  __restrict__ hg_cum_mm,
    const double*  __restrict__ cn,
    double* __restrict__ cum_rain_mm,
    double* __restrict__ cum_excess_mm,
    double* __restrict__ cell_source_mps,
    double t0,
    double t1,
    double ia_ratio,
    double mm_to_model_depth)
{
    int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells) return;

    const int32_t gidx = cell_gage_idx[c];
    if (gidx < 0) {
        cell_source_mps[c] = 0.0;
        return;
    }
    const int32_t s = hg_offsets[gidx];
    const int32_t e = hg_offsets[gidx + 1];
    if (e <= s) {
        cell_source_mps[c] = 0.0;
        return;
    }

    const double r0 = interp_series_clamped_cuda(hg_time_s, hg_cum_mm, s, e, t0);
    const double r1 = interp_series_clamped_cuda(hg_time_s, hg_cum_mm, s, e, t1);
    const double dr = fmax(0.0, r1 - r0);
    const double p = cum_rain_mm[c] + dr;

    const double cn_c = fmin(100.0, fmax(1.0, cn[c]));
    const double s_mm = fmax((25400.0 / cn_c) - 254.0, 0.0);
    const double ia = ia_ratio * s_mm;
    double pe = 0.0;
    if (p > ia) {
        const double num = (p - ia) * (p - ia);
        const double den = fmax(p + (1.0 - ia_ratio) * s_mm, 1.0e-12);
        pe = num / den;
    }
    const double de = fmax(0.0, pe - cum_excess_mm[c]);

    cum_rain_mm[c] = p;
    cum_excess_mm[c] = pe;

    const double dt = fmax(t1 - t0, 1.0e-9);
    cell_source_mps[c] = (de * mm_to_model_depth) / dt;
}

__global__ void swe2d_eval_rain_cn_stage_rate_kernel(
    int32_t n_cells,
    const int32_t* __restrict__ cell_gage_idx,
    const int32_t* __restrict__ hg_offsets,
    const double*  __restrict__ hg_time_s,
    const double*  __restrict__ hg_cum_mm,
    const double*  __restrict__ cn,
    const double*  __restrict__ cum_rain_mm,
    double* __restrict__ cell_source_mps,
    double t_base,
    double t_stage,
    double ia_ratio,
    double mm_to_model_depth)
{
    int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells) return;

    const int32_t gidx = cell_gage_idx[c];
    if (gidx < 0) {
        cell_source_mps[c] = 0.0;
        return;
    }

    const int32_t s = hg_offsets[gidx];
    const int32_t e = hg_offsets[gidx + 1];
    if (e <= s) {
        cell_source_mps[c] = 0.0;
        return;
    }

    const double r_base = interp_series_clamped_cuda(hg_time_s, hg_cum_mm, s, e, t_base);
    const double r_stage = interp_series_clamped_cuda(hg_time_s, hg_cum_mm, s, e, t_stage);
    const double dr = fmax(0.0, r_stage - r_base);
    const double p = cum_rain_mm[c] + dr;

    const double cn_c = fmin(100.0, fmax(1.0, cn[c]));
    const double s_mm = fmax((25400.0 / cn_c) - 254.0, 0.0);
    const double ia = ia_ratio * s_mm;
    if (p <= ia) {
        cell_source_mps[c] = 0.0;
        return;
    }

    const double a = p - ia;
    const double b = fmax(p + (1.0 - ia_ratio) * s_mm, 1.0e-12);
    const double dpe_dp = a * (2.0 * b - a) / (b * b);
    const double rain_rate_mmps = fmax(0.0, interp_series_slope_clamped_cuda(hg_time_s, hg_cum_mm, s, e, t_stage));
    cell_source_mps[c] = fmax(0.0, dpe_dp * rain_rate_mmps * mm_to_model_depth);
}

__global__ void swe2d_stage_source_max_kernel(
    int32_t n_cells,
    int32_t n_slots,
    const double* __restrict__ stage_source,
    double* __restrict__ cell_source_mps)
{
    int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells) return;

    double src_max = 0.0;
    for (int32_t slot = 0; slot < n_slots; ++slot) {
        const double v = stage_source[static_cast<size_t>(slot) * static_cast<size_t>(n_cells) + static_cast<size_t>(c)];
        if (isfinite(v) && v > src_max) src_max = v;
    }
    cell_source_mps[c] = src_max;
}

__global__ void swe2d_rk_multi_stage_build_kernel(
    int32_t n_cells,
    double* dst_h,
    double* dst_hu,
    double* dst_hv,
    const double* base_h,
    const double* base_hu,
    const double* base_hv,
    const double* k1_h,
    const double* k1_hu,
    const double* k1_hv,
    double a1,
    const double* k2_h,
    const double* k2_hu,
    const double* k2_hv,
    double a2,
    const double* k3_h,
    const double* k3_hu,
    const double* k3_hv,
    double a3,
    const double* k4_h,
    const double* k4_hu,
    const double* k4_hv,
    double a4,
    const double* k5_h,
    const double* k5_hu,
    const double* k5_hv,
    double a5,
    const double* k6_h,
    const double* k6_hu,
    const double* k6_hv,
    double a6)
{
    int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells) return;

    double h = base_h[c];
    double hu = base_hu[c];
    double hv = base_hv[c];
    if (k1_h) { h += a1 * k1_h[c]; hu += a1 * k1_hu[c]; hv += a1 * k1_hv[c]; }
    if (k2_h) { h += a2 * k2_h[c]; hu += a2 * k2_hu[c]; hv += a2 * k2_hv[c]; }
    if (k3_h) { h += a3 * k3_h[c]; hu += a3 * k3_hu[c]; hv += a3 * k3_hv[c]; }
    if (k4_h) { h += a4 * k4_h[c]; hu += a4 * k4_hu[c]; hv += a4 * k4_hv[c]; }
    if (k5_h) { h += a5 * k5_h[c]; hu += a5 * k5_hu[c]; hv += a5 * k5_hv[c]; }
    if (k6_h) { h += a6 * k6_h[c]; hu += a6 * k6_hu[c]; hv += a6 * k6_hv[c]; }

    dst_h[c] = h;
    dst_hu[c] = hu;
    dst_hv[c] = hv;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1: Flux computation — one thread per edge.
// Writes one flux contribution per edge. The update kernel consumes the edge
// fluxes through the cell-edge CSR, removing the need for atomic accumulation.
// ─────────────────────────────────────────────────────────────────────────────
__global__ __launch_bounds__(256, 4) void swe2d_flux_kernel(
    int32_t                     n_edges,
    const int32_t* __restrict__ edge_c0,
    const int32_t* __restrict__ edge_c1,
    const double*  __restrict__ edge_nx,
    const double*  __restrict__ edge_ny,
    const double*  __restrict__ edge_len,
    const double*  __restrict__ edge_mx,
    const double*  __restrict__ edge_my,
    const int32_t* __restrict__ edge_bc,
    const double*  __restrict__ edge_bc_val,
    const double*  __restrict__ cell_h,
    const double*  __restrict__ cell_hu,
    const double*  __restrict__ cell_hv,
    const double*  __restrict__ cell_n_mann,
    const double*  __restrict__ cell_zb,
    const double*  __restrict__ cell_inv_area,
    // Cell centroids and gradients (used for MC and Van Leer limiters)
    const double*  __restrict__ cell_cx,
    const double*  __restrict__ cell_cy,
    const double*  __restrict__ grad_hx,  const double* __restrict__ grad_hy,
    const double*  __restrict__ grad_hux, const double* __restrict__ grad_huy,
    const double*  __restrict__ grad_hvx, const double* __restrict__ grad_hvy,
    double*                     flux_h,    // [n_cells] accumulator
    double*                     flux_hu,
    double*                     flux_hv,
    double*                     flux_hu_r,
    double*                     flux_hv_r,
    double*                     dbg_fh,
    double*                     dbg_fhu,
    double*                     dbg_fhv,
    int                         spatial_scheme,
    double g, double h_min,
    double                      max_inv_area,
    double                      momentum_cap_min_speed,
    double                      momentum_cap_celerity_mult,
    const int32_t* __restrict__ d_degen_mask,
    const int32_t* __restrict__ d_merge_owner,
    int                         degen_mode,
    const int32_t* __restrict__ d_active,            // nullable: wet/dry active-set mask
    double                      front_flux_damping,  // momentum-flux scale for wet/dry front edges
    double                      shallow_damping_depth,
    bool                        enable_shallow_front_recon_fallback)
{
    int32_t e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_edges) return;

    int32_t c0 = edge_c0[e];
    int32_t c1 = edge_c1[e];

    // Degenerate-cell handling at edge level.
    // Build per-endpoint degenerate flags using mask when available.
    const int32_t dm0 = d_degen_mask ? d_degen_mask[c0]
                                     : (cell_inv_area[c0] > max_inv_area ? 1 : 0);
    const int32_t dm1 = (c1 >= 0) ? (d_degen_mask ? d_degen_mask[c1]
                                                   : (cell_inv_area[c1] > max_inv_area ? 1 : 0))
                                   : 0;

    // Modes 0 and 1: skip edges involving degenerate cells entirely.
    if (degen_mode <= 1 && (dm0 || dm1)) return;

    // Mode 3: skip edge if a degenerate endpoint has no valid merge owner.
    if (degen_mode == 3) {
        if (dm0 && (d_merge_owner == nullptr || d_merge_owner[c0] < 0)) return;
        if (dm1 && (d_merge_owner == nullptr || d_merge_owner[c1] < 0)) return;
    }

    // Dry-edge early exit: skip edges fully outside the active set.
    // - Interior: both endpoint cells inactive.
    // - Boundary: boundary-adjacent cell inactive.
    // This avoids reconstruction and HLLC work for dry regions, which is
    // important when wetted-cell fraction is small.
    if (d_active) {
        if (c1 >= 0) {
            if (!d_active[c0] && !d_active[c1]) return;
        } else {
            if (!d_active[c0]) return;
        }
    }

    double  nx  = edge_nx[e];
    double  ny  = edge_ny[e];
    double  len = edge_len[e];

    // __ldg: forces L1 read-only (texture) cache path for irregular scatter-reads.
    double hL  = __ldg(&cell_h[c0]),  huL = __ldg(&cell_hu[c0]), hvL = __ldg(&cell_hv[c0]);
    double zbL = __ldg(&cell_zb[c0]);

    double hR, huR, hvR, zbR;
    if (c1 >= 0) {
        hR  = __ldg(&cell_h[c1]); huR = __ldg(&cell_hu[c1]); hvR = __ldg(&cell_hv[c1]);
        zbR = __ldg(&cell_zb[c1]);

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
        const int scheme_weno5  = static_cast<int>(SWE2DSpatialScheme::FV_WENO5);
        const double recon_fallback_depth = fmax(h_min, 0.5 * shallow_damping_depth);
        const bool shallow_pair = (hL < recon_fallback_depth) || (hR < recon_fallback_depth);
        const bool disable_higher_order = enable_shallow_front_recon_fallback && shallow_pair;
        if (!disable_higher_order && spatial_scheme >= scheme_fast && cell_cx != nullptr && grad_hx != nullptr) {
            const double fx = edge_mx[e];
            const double fy = edge_my[e];
            const double dcx = cell_cx[c1] - cell_cx[c0];
            const double dcy = cell_cy[c1] - cell_cy[c0];
            constexpr double EPS = 1.0e-30;

            // Helper lambda: compute TVD limiter phi(r), extrapolate each cell state
            // to the actual face midpoint, then clamp to the local cell-pair bounds.
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

                const double dxL = fx - cell_cx[c0];
                const double dyL = fy - cell_cy[c0];
                const double dxR = fx - cell_cx[c1];
                const double dyR = fy - cell_cy[c1];
                const double rawL = q0 + phi0 * (gx0 * dxL + gy0 * dyL);
                const double rawR = q1 + phi1 * (gx1 * dxR + gy1 * dyR);
                const double qmin = fmin(q0, q1);
                const double qmax = fmax(q0, q1);
                qL_out = fmin(qmax, fmax(qmin, rawL));
                qR_out = fmin(qmax, fmax(qmin, rawR));
            };

            // WENO5 helper on unstructured cell pairs (scheme 6):
            // The grad_* arrays here hold least-squares 2-ring gradients
            // (swe2d_lsq_gradient_kernel overwrote the Green-Gauss values for
            // degree>=3 cells; degenerate cells keep the GG fallback). Three
            // candidate face values per side are blended with nonlinear WENO
            // weights biased toward the high-order LSQ candidate, then clamped
            // to the local cell-pair bounds for monotonicity/well-balancing.
            auto weno5_reconstruct = [&](double q0, double q1,
                                         double gx0, double gy0,
                                         double gx1, double gy1,
                                         double& qL_out, double& qR_out) {
                const double dq = q1 - q0;
                const double dxL = fx - cell_cx[c0];
                const double dyL = fy - cell_cy[c0];
                const double dxR = fx - cell_cx[c1];
                const double dyR = fy - cell_cy[c1];

                // LSQ-extrapolated face slope projections (high-order candidate).
                const double sL = gx0 * dxL + gy0 * dyL;
                const double sR = gx1 * dxR + gy1 * dyR;

                // Van Leer TVD-limited LSQ slope using the pair jump as the
                // monotonicity reference (robust candidate).
                const double sign_dq = (dq >= 0.0) ? 1.0 : -1.0;
                const double s0_pair = gx0 * dcx + gy0 * dcy;
                const double r0 = s0_pair / (dq + sign_dq * EPS);
                const double s1_pair = -(gx1 * dcx + gy1 * dcy);
                const double r1 = s1_pair / (-dq + (-sign_dq) * EPS);
                const double phi0 = (r0 + fabs(r0)) / (1.0 + fabs(r0));
                const double phi1 = (r1 + fabs(r1)) / (1.0 + fabs(r1));

                // Three candidate face values per side.
                const double pL0 = q0 + 0.5 * dq;   // pair midpoint (low order)
                const double pL1 = q0 + sL;         // unlimited LSQ (high order)
                const double pL2 = q0 + phi0 * sL;  // TVD-limited LSQ (robust)
                const double pR0 = q1 - 0.5 * dq;
                const double pR1 = q1 + sR;
                const double pR2 = q1 + phi1 * sR;

                // Smoothness indicators (squared deviation from cell mean).
                const double scale = q0 * q0 + q1 * q1 + dq * dq;
                const double eps_weno = 1.0e-20 + 1.0e-6 * fmax(1.0, scale);
                const double betaL0 = (pL0 - q0) * (pL0 - q0);
                const double betaL1 = (pL1 - q0) * (pL1 - q0);
                const double betaL2 = (pL2 - q0) * (pL2 - q0);
                const double betaR0 = (pR0 - q1) * (pR0 - q1);
                const double betaR1 = (pR1 - q1) * (pR1 - q1);
                const double betaR2 = (pR2 - q1) * (pR2 - q1);

                // Linear (ideal) weights: bias toward high-order LSQ candidate.
                const double d0 = 0.10, d1 = 0.30, d2 = 0.60;
                const double aL0 = d0 / ((eps_weno + betaL0) * (eps_weno + betaL0));
                const double aL1 = d1 / ((eps_weno + betaL1) * (eps_weno + betaL1));
                const double aL2 = d2 / ((eps_weno + betaL2) * (eps_weno + betaL2));
                const double sumL = aL0 + aL1 + aL2;
                const double rawL = (sumL > 0.0)
                    ? (aL0 * pL0 + aL1 * pL1 + aL2 * pL2) / sumL
                    : pL0;

                const double aR0 = d0 / ((eps_weno + betaR0) * (eps_weno + betaR0));
                const double aR1 = d1 / ((eps_weno + betaR1) * (eps_weno + betaR1));
                const double aR2 = d2 / ((eps_weno + betaR2) * (eps_weno + betaR2));
                const double sumR = aR0 + aR1 + aR2;
                const double rawR = (sumR > 0.0)
                    ? (aR0 * pR0 + aR1 * pR1 + aR2 * pR2) / sumR
                    : pR0;

                const double qmin = fmin(q0, q1);
                const double qmax = fmax(q0, q1);
                qL_out = fmin(qmax, fmax(qmin, rawL));
                qR_out = fmin(qmax, fmax(qmin, rawR));
            };

            // Reconstruct free surface eta=h+zb using grad_h* (which stores
            // Green-Gauss gradients of eta from swe2d_gradient_kernel), then
            // convert back to depth for hydrostatic reconstruction.
            double etaL_rec, etaR_rec, huL_rec, huR_rec, hvL_rec, hvR_rec;
            const double etaL = hL + zbL;
            const double etaR = hR + zbR;
            if (spatial_scheme == scheme_weno5) {
                weno5_reconstruct(etaL, etaR, grad_hx[c0], grad_hy[c0], grad_hx[c1], grad_hy[c1], etaL_rec, etaR_rec);
                weno5_reconstruct(huL, huR, grad_hux[c0], grad_huy[c0], grad_hux[c1], grad_huy[c1], huL_rec, huR_rec);
                weno5_reconstruct(hvL, hvR, grad_hvx[c0], grad_hvy[c0], grad_hvx[c1], grad_hvy[c1], hvL_rec, hvR_rec);
            } else {
                tvd_reconstruct(etaL, etaR, grad_hx[c0], grad_hy[c0], grad_hx[c1], grad_hy[c1], etaL_rec, etaR_rec);
                tvd_reconstruct(huL, huR, grad_hux[c0], grad_huy[c0], grad_hux[c1], grad_huy[c1], huL_rec, huR_rec);
                tvd_reconstruct(hvL, hvR, grad_hvx[c0], grad_hvy[c0], grad_hvx[c1], grad_hvy[c1], hvL_rec, hvR_rec);
            }

            hL  = fmax(0.0, etaL_rec - zbL);
            hR  = fmax(0.0, etaR_rec - zbR);
            huL = huL_rec; huR = huR_rec;
            hvL = hvL_rec; hvR = hvR_rec;
        }

        // Keep reconstructed momentum physically bounded for shallow cells.
        const double hL_eff = (hL > h_min) ? hL : h_min;
        const double hR_eff = (hR > h_min) ? hR : h_min;
        const double u_cap_L = fmax(momentum_cap_min_speed,
                        momentum_cap_celerity_mult * sqrt(g * hL_eff));
        const double u_cap_R = fmax(momentum_cap_min_speed,
                        momentum_cap_celerity_mult * sqrt(g * hR_eff));
        const double hu_cap_L = hL_eff * u_cap_L;
        const double hv_cap_L = hL_eff * u_cap_L;
        const double hu_cap_R = hR_eff * u_cap_R;
        const double hv_cap_R = hR_eff * u_cap_R;
        huL = fmin(hu_cap_L, fmax(-hu_cap_L, huL));
        hvL = fmin(hv_cap_L, fmax(-hv_cap_L, hvL));
        huR = fmin(hu_cap_R, fmax(-hu_cap_R, huR));
        hvR = fmin(hv_cap_R, fmax(-hv_cap_R, hvR));
    } else {
        const double n_local = cell_n_mann ? cell_n_mann[c0] : 0.03;
        GhostStateLocal gs = make_ghost_cuda_local(
            hL, huL, hvL, zbL, nx, ny,
            edge_bc[e], edge_bc_val[e], h_min, n_local);
        hR  = gs.h; huR = gs.hu; hvR = gs.hv;
        zbR = gs.zb;
    }

// Reconstruct hydrostatic states then apply a CUDA-local HLLC flux.
    // This avoids relying on host/device overload resolution from the shared
    // header in this kernel hot-path.
    ReconstructedStatesLocal rs = hydrostatic_reconstruct_cuda_local(
        hL, huL, hvL, zbL, hR, huR, hvR, zbR, h_min);

    double flux_fh = 0.0, flux_fhu = 0.0, flux_fhv = 0.0;
    hllc_flux_cuda_local(
        rs.hL_star, rs.uL, rs.vL,
        rs.hR_star, rs.uR, rs.vR,
        nx, ny, g, h_min,
        flux_fh, flux_fhu, flux_fhv);

    double corr_hu = 0.0, corr_hv = 0.0;
    bed_slope_correction_cuda_local(hL, rs.hL_star, nx, ny, g, corr_hu, corr_hv);

    double fh  = flux_fh  * len;
    double fhu_l = (flux_fhu + corr_hu) * len;
    double fhv_l = (flux_fhv + corr_hv) * len;

    // Front-aware flux damping: at wet/dry front edges (exactly one side active),
    // scale the momentum component of the flux to suppress oscillations that grow
    // at the advancing front.  Mass flux (fh) is NOT scaled to preserve
    // mass conservation; only the momentum signal is attenuated.
    const bool is_wet_dry_front = d_active && (c1 >= 0) &&
                                  ((d_active[c0] != 0) != (d_active[c1] != 0));
    if (is_wet_dry_front && front_flux_damping < 1.0) {
        fhu_l *= front_flux_damping;
        fhv_l *= front_flux_damping;
    }

    if (dbg_fh) {
        dbg_fh[e] = fh;
        dbg_fhu[e] = fhu_l;
        dbg_fhv[e] = fhv_l;
    }

    double corr_hu_r = 0.0, corr_hv_r = 0.0;
    if (c1 >= 0) {
        // Same normal direction as c0 to preserve lake-at-rest balance.
        bed_slope_correction_cuda_local(hR, rs.hR_star, nx, ny, g, corr_hu_r, corr_hv_r);
    }

    double fhu_r = (flux_fhu + corr_hu_r) * len;
    double fhv_r = (flux_fhv + corr_hv_r) * len;
    if (is_wet_dry_front && front_flux_damping < 1.0) {
        fhu_r *= front_flux_damping;
        fhv_r *= front_flux_damping;
    }

    // Store the edge contribution once, with momentum terms for both sides.
    flux_h[e] = -fh;
    flux_hu[e] = -fhu_l;
    flux_hv[e] = -fhv_l;
    if (flux_hu_r) flux_hu_r[e] = fhu_r;
    if (flux_hv_r) flux_hv_r[e] = fhv_r;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2: State update — one thread per cell.
// Reduces the incident edge fluxes through the cell-edge CSR, removing the
// need for atomic cell accumulation in the flux kernel.
// ─────────────────────────────────────────────────────────────────────────────
__global__ __launch_bounds__(256, 4) void swe2d_update_kernel(
    int32_t                     n_cells,
    const int32_t* __restrict__ cell_edge_offsets,
    const int32_t* __restrict__ cell_edge_ids,
    const int32_t* __restrict__ edge_c0,
    const int32_t* __restrict__ edge_c1,
    double*                     cell_h,
    double*                     cell_hu,
    double*                     cell_hv,
    const double*  __restrict__ flux_h,
    const double*  __restrict__ flux_hu,
    const double*  __restrict__ flux_hv,
    const double*  __restrict__ flux_hu_r,
    const double*  __restrict__ flux_hv_r,
    const double*  __restrict__ cell_inv_area,
    const double*  __restrict__ cell_n_mann,
    double*                     d_max_wse_elev_error,
    double dt, double g, double h_min,
    double                      max_inv_area,
    double                      momentum_cap_min_speed,
    double                      momentum_cap_celerity_mult,
    double                      depth_cap,
    double                      max_rel_depth_increase,
    double                      shallow_damping_depth,
    bool                        extreme_rain_mode,
    double                      source_cfl_beta,
    int                         source_max_substeps,
    double                      source_rate_cap,
    double                      source_depth_step_cap,
    bool                        source_true_subcycling,
    bool                        source_imex_split,
    const int32_t* __restrict__ d_active,
    const int32_t* __restrict__ d_degen_mask,
    const double*  __restrict__ d_inv_area_repaired,
    int                         degen_mode,
    const double* __restrict__  cell_source_mps,
    const double* __restrict__  external_source_mps,
    const double* __restrict__  ext_struct_flux_h,   // nullable: face-based culvert mass flux
    const double* __restrict__  ext_struct_flux_hu,  // nullable: face-based culvert x-mom flux
    const double* __restrict__  ext_struct_flux_hv)  // nullable: face-based culvert y-mom flux
{
    int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells) return;

    // Skip fully isolated dry cells when there is no local source term.
    // If a positive rain/source term exists, allow a source-only wet-up update.
    // Also include face-based culvert flux (ext_struct_flux_h) which carries
    // mass from the face-flux path — without this, a dry downstream cell
    // receiving water through a culvert face would be skipped because
    // external_source_mps is 0 (culvert flow goes through ext_struct_flux_h).
    if (d_active && !d_active[c]) {
        double src =
            (cell_source_mps ? cell_source_mps[c] : 0.0) +
            (external_source_mps ? external_source_mps[c] : 0.0);
        if (ext_struct_flux_h) {
            src += fmax(0.0, ext_struct_flux_h[c]);  // only positive (incoming) flux
        }
        if (!(isfinite(src) && src > 0.0)) return;
    }

    // Modes 1 and 3: skip degenerate cells entirely (flux was dropped or redirected).
    if (d_degen_mask && d_degen_mask[c] && degen_mode != 2) return;

    const double h_old = cell_h[c];

    double inv_a;
    if (degen_mode == 2 && d_inv_area_repaired != nullptr && d_degen_mask && d_degen_mask[c]) {
        // Mode 2: use neighbor-averaged repaired inv_area to prevent CFL collapse.
        inv_a = d_inv_area_repaired[c];
    } else {
        inv_a = cell_inv_area[c];
        // Modes 0/1: clamp inv_area to prevent overflow in update.
        const double max_inv_a = fmax(max_inv_area, 1.0);
        if (inv_a > max_inv_a) inv_a = max_inv_a;
    }

    double fh = 0.0;
    double fhu = 0.0;
    double fhv = 0.0;
    const int32_t s = cell_edge_offsets[c];
    const int32_t e = cell_edge_offsets[c + 1];
    for (int32_t k = s; k < e; ++k) {
        const int32_t edge = cell_edge_ids[k];
        if (edge_c0[edge] == c) {
            fh  += flux_h[edge];
            fhu += flux_hu[edge];
            fhv += flux_hv[edge];
        } else {
            fh  -= flux_h[edge];
            fhu += flux_hu_r ? flux_hu_r[edge] : -flux_hu[edge];
            fhv += flux_hv_r ? flux_hv_r[edge] : -flux_hv[edge];
        }
    }

    if (!isfinite(fh)) fh = 0.0;
    if (!isfinite(fhu)) fhu = 0.0;
    if (!isfinite(fhv)) fhv = 0.0;

    double h_trial = cell_h[c] + dt * fh * inv_a;
    double src =
        (cell_source_mps ? cell_source_mps[c] : 0.0) +
        (external_source_mps ? external_source_mps[c] : 0.0);

    int nsub = 1;
    if (src > 0.0) {
        if (source_rate_cap > 0.0 && src > source_rate_cap) {
            src = source_rate_cap;
        }
        if (source_depth_step_cap > 0.0) {
            const double src_step_cap = source_depth_step_cap / fmax(dt, 1.0e-12);
            if (src > src_step_cap) src = src_step_cap;
        }
        if (extreme_rain_mode && source_cfl_beta > 0.0) {
            const double h_ref = fmax(h_old, h_min);
            const double dt_src = source_cfl_beta * h_ref / fmax(src, 1.0e-12);
            if (dt_src < dt) {
                nsub = max(1, static_cast<int>(ceil(dt / fmax(dt_src, 1.0e-12))));
                if (source_max_substeps > 0) nsub = min(nsub, source_max_substeps);
            }
        }
    }

    if (source_true_subcycling && nsub > 1 && src > 0.0) {
        const double dt_sub = dt / static_cast<double>(nsub);
        for (int k = 0; k < nsub; ++k) {
            h_trial += dt_sub * src;
            // Apply face-flux within subcycling so it's subject to the same
            // depth limiting as the source rate (prevents donor going negative).
            if (ext_struct_flux_h) {
                h_trial += dt_sub * ext_struct_flux_h[c] * inv_a;
            }
            if (h_trial < 0.0) h_trial = 0.0;
            if (source_imex_split && h_trial > h_min) {
                double n_mann = cell_n_mann[c];
                apply_friction_cuda_local(h_trial, cell_hu[c], cell_hv[c],
                                          dt_sub, n_mann, g, h_min);
            }
        }
    } else {
        if (extreme_rain_mode && nsub > 1 && src > 0.0) {
            src *= (1.0 / static_cast<double>(nsub));
        }
        h_trial += dt * src;
        // Apply face-flux mass in non-subcycling path
        if (ext_struct_flux_h) {
            h_trial += dt * ext_struct_flux_h[c] * inv_a;
        }
    }

    if (!isfinite(h_trial)) h_trial = 0.0;

    // Per-step depth growth limiter for wetting-front robustness.
    if (max_rel_depth_increase > 0.0) {
        const double h_ref = fmax(h_old, h_min);
        const double h_step_cap = h_old + max_rel_depth_increase * h_ref;
        if (h_trial > h_step_cap) h_trial = h_step_cap;
    }
    if (depth_cap > 0.0 && h_trial > depth_cap) h_trial = depth_cap;

    cell_h[c]  = h_trial;
    cell_hu[c] += dt * fhu * inv_a;
    cell_hv[c] += dt * fhv * inv_a;

    // Face-based structure flux: apply momentum from culvert face coupling.
    // Mass component was already applied inside source subcycling above.
    if (ext_struct_flux_hu) {
        cell_hu[c] += dt * ext_struct_flux_hu[c] * inv_a;
    }
    if (ext_struct_flux_hv) {
        cell_hv[c] += dt * ext_struct_flux_hv[c] * inv_a;
    }

    // Positivity enforcement
    if (cell_h[c] < 0.0) cell_h[c] = 0.0;
    if (cell_h[c] < h_min) {
        cell_hu[c] = 0.0;
        cell_hv[c] = 0.0;
    } else if (shallow_damping_depth > h_min && cell_h[c] < shallow_damping_depth) {
        // Smoothly damp momentum in shallow cells to stabilize moving wet/dry fronts.
        // Uses Hermite smoothstep (3t²−2t³) which is C¹ at both endpoints:
        //   scale=0 at h=h_min  (no abrupt momentum discontinuity)
        //   scale=1 at h=shallow_damping_depth (full momentum beyond threshold)
        const double t   = (cell_h[c] - h_min) / (shallow_damping_depth - h_min);
        const double t_s = fmin(1.0, fmax(0.0, t));
        const double scale = t_s * t_s * (3.0 - 2.0 * t_s);
        cell_hu[c] *= scale;
        cell_hv[c] *= scale;
    }

    // Manning friction (semi-implicit). In IMEX split + true-subcycling mode
    // friction has already been applied with dt_sub substeps.
    if (!(source_true_subcycling && source_imex_split && nsub > 1)) {
        double n_mann = cell_n_mann[c];
        apply_friction_cuda_local(cell_h[c], cell_hu[c], cell_hv[c],
                                  dt, n_mann, g, h_min);
    }

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
        const double spd_cap = fmax(momentum_cap_min_speed,
                                    momentum_cap_celerity_mult * sqrt(g * cell_h[c]));
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

__global__ __launch_bounds__(256, 4) void swe2d_rk2_combine_kernel(
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

__global__ __launch_bounds__(256, 4) void swe2d_rk5_graph_combine_kernel(
    int32_t                     n_cells,
    double*                     cell_h,
    double*                     cell_hu,
    double*                     cell_hv,
    const double*  __restrict__ h0,
    const double*  __restrict__ hu0,
    const double*  __restrict__ hv0,
    const double*  __restrict__ k1_h,
    const double*  __restrict__ k1_hu,
    const double*  __restrict__ k1_hv,
    const double*  __restrict__ k3_h,
    const double*  __restrict__ k3_hu,
    const double*  __restrict__ k3_hv,
    const double*  __restrict__ k4_h,
    const double*  __restrict__ k4_hu,
    const double*  __restrict__ k4_hv,
    const double*  __restrict__ k6_h,
    const double*  __restrict__ k6_hu,
    const double*  __restrict__ k6_hv,
    double*                     d_max_wse_elev_error,
    const double*  __restrict__ cell_n_mann,
    double                      g,
    double                      h_min,
    double                      shallow_damping_depth,
    double                      dt,
    double                      momentum_cap_min_speed,
    double                      momentum_cap_celerity_mult)
{
    int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells) return;

    double h_new = h0[c]
        + (37.0 / 378.0) * k1_h[c]
        + (250.0 / 621.0) * k3_h[c]
        + (125.0 / 594.0) * k4_h[c]
        + (512.0 / 1771.0) * k6_h[c];
    double hu_new = hu0[c]
        + (37.0 / 378.0) * k1_hu[c]
        + (250.0 / 621.0) * k3_hu[c]
        + (125.0 / 594.0) * k4_hu[c]
        + (512.0 / 1771.0) * k6_hu[c];
    double hv_new = hv0[c]
        + (37.0 / 378.0) * k1_hv[c]
        + (250.0 / 621.0) * k3_hv[c]
        + (125.0 / 594.0) * k4_hv[c]
        + (512.0 / 1771.0) * k6_hv[c];

    if (!isfinite(h_new))  h_new = 0.0;
    if (!isfinite(hu_new)) hu_new = 0.0;
    if (!isfinite(hv_new)) hv_new = 0.0;

    if (h_new < 0.0) h_new = 0.0;
    double h_final = h_new;

    if (h_final < h_min) {
        hu_new = 0.0;
        hv_new = 0.0;
    } else if (shallow_damping_depth > h_min && h_final < shallow_damping_depth) {
        const double t = (h_final - h_min) / (shallow_damping_depth - h_min);
        const double t_s = fmin(1.0, fmax(0.0, t));
        const double scale = t_s * t_s * (3.0 - 2.0 * t_s);
        hu_new *= scale;
        hv_new *= scale;
    }

    if (h_final >= h_min) {
        double n_mann = cell_n_mann[c];
        apply_friction_cuda_local(h_final, hu_new, hv_new, dt, n_mann, g, h_min);
    }

    if (h_final > h_min) {
        const double inv_h = 1.0 / h_final;
        const double u = hu_new * inv_h;
        const double v = hv_new * inv_h;
        const double spd = sqrt(u * u + v * v);
        const double spd_cap = fmax(momentum_cap_min_speed,
                                    momentum_cap_celerity_mult * sqrt(g * h_final));
        if (isfinite(spd) && spd > spd_cap && spd > 0.0) {
            const double scale = spd_cap / spd;
            hu_new *= scale;
            hv_new *= scale;
        }
    }

    cell_h[c] = h_final;
    cell_hu[c] = hu_new;
    cell_hv[c] = hv_new;

    if (d_max_wse_elev_error) {
        atomicMaxDouble(d_max_wse_elev_error, fabs(h_final - h0[c]));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 3: CFL reduction — one thread per edge, block-level max, then
// block maxima written to d_cfl_block_max.  A second lightweight kernel
// reduces d_cfl_block_max → d_lambda_max (single atomic per block instead
// of per-edge).
// ─────────────────────────────────────────────────────────────────────────────
__global__ __launch_bounds__(256, 4) void swe2d_cfl_kernel(
    int32_t                     n_edges,
    const int32_t* __restrict__ edge_c0,
    const int32_t* __restrict__ edge_c1,
    const double*  __restrict__ edge_nx,
    const double*  __restrict__ edge_ny,
    const double*  __restrict__ edge_len,
    const double*  __restrict__ cell_h,
    const double*  __restrict__ cell_hu,
    const double*  __restrict__ cell_hv,
    const double*  __restrict__ cell_area,
    double g, double h_min,
    double                      lambda_cap,
    double*                     d_cfl_block_max,
    const int32_t* __restrict__ d_degen_mask,
    int                         degen_mode)
{
    extern __shared__ double sdata[];
    int32_t tid = threadIdx.x;
    int32_t e   = blockIdx.x * blockDim.x + tid;

    double lambda = 0.0;
    if (e < n_edges) {
        const int32_t c0 = edge_c0[e];
        const int32_t c1 = edge_c1[e];

        const bool c0_degen = (d_degen_mask && d_degen_mask[c0] && degen_mode > 0);
        const bool c1_degen = (c1 >= 0 && d_degen_mask && d_degen_mask[c1] && degen_mode > 0);
        if (!(c0_degen || c1_degen)) {
            const double nx = edge_nx[e];
            const double ny = edge_ny[e];

            const double hL  = cell_h[c0];
            const double huL = cell_hu[c0];
            const double hvL = cell_hv[c0];

            double hR = hL;
            double huR = huL;
            double hvR = hvL;
            if (c1 >= 0) {
                hR = cell_h[c1];
                huR = cell_hu[c1];
                hvR = cell_hv[c1];
            }

            const double uL = vel_u_cuda_local(huL, hL, h_min);
            const double vL = vel_v_cuda_local(hvL, hL, h_min);
            const double uR = vel_u_cuda_local(huR, hR, h_min);
            const double vR = vel_v_cuda_local(hvR, hR, h_min);
            const double cL = celerity_cuda_local(hL, g);
            const double cR = celerity_cuda_local(hR, g);

            const double unL = uL * nx + vL * ny;
            const double unR = uR * nx + vR * ny;
            const double max_wave = fmax(fabs(unL) + cL, fabs(unR) + cR);

            const double aL = cell_area[c0];
            const double aR = (c1 >= 0) ? cell_area[c1] : aL;
            const double area_eff = (c1 >= 0 && aR > 0.0) ? fmin(aL, aR) : aL;
            const double len = edge_len[e];
            const double dx_eff = (len > 0.0 && area_eff > 0.0) ? (area_eff / len) : 1.0;

            lambda = (dx_eff > 0.0) ? (max_wave / dx_eff) : 0.0;
            if (!isfinite(lambda)) lambda = 0.0;
            const double lam_cap = fmax(lambda_cap, 1.0);
            if (lambda > lam_cap) lambda = lam_cap;
        } else {
            lambda = 0.0;
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
        d_cfl_block_max[blockIdx.x] = sdata[0];
    }
}

// Second-level reduction: one block reduces d_cfl_block_max → d_lambda_max.
__global__ __launch_bounds__(256, 4) void swe2d_cfl_reduce_blocks_kernel(
    int32_t n_blocks,
    const double* __restrict__ d_cfl_block_max,
    double* __restrict__ d_lambda_max)
{
    extern __shared__ double sdata[];
    int32_t tid = threadIdx.x;
    double val = 0.0;
    for (int32_t i = tid; i < n_blocks; i += blockDim.x) {
        double v = d_cfl_block_max[i];
        if (isfinite(v) && v > val) val = v;
    }
    sdata[tid] = val;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (sdata[tid + stride] > sdata[tid])
                sdata[tid] = sdata[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        // Atomic max via CAS loop (only one per block, so contention is minimal)
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
// pack_diag_kernel — pack two device scalars into a contiguous 2-double buffer
// so a single cudaMemcpy of 16 bytes transfers all diagnostic values.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void pack_diag_kernel(
    const double*  __restrict__ d_lambda_max,
    const double*  __restrict__ d_max_wse_elev_error,
    const int32_t* __restrict__ d_n_wet,
    double*        __restrict__ d_out)
{
    d_out[0] = d_lambda_max[0];
    d_out[1] = d_max_wse_elev_error[0];
    d_out[2] = d_n_wet ? static_cast<double>(d_n_wet[0]) : -1.0;
}

__global__ __launch_bounds__(256, 4) void swe2d_coupling_inlet_source_kernel(
    int32_t n_inlets,
    const int32_t* __restrict__ inlet_cell,
    const double* __restrict__ inlet_flow_cms,
    const double* __restrict__ cell_area,
    int32_t n_cells,
    double* __restrict__ source_rate)
{
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_inlets) return;
    const int32_t c = inlet_cell[i];
    if (c < 0 || c >= n_cells) return;
    const double q = inlet_flow_cms[i];
    if (!isfinite(q) || q == 0.0) return;
    const double area = fmax(cell_area[c], 1.0e-12);
    // Positive inlet capture removes water from the surface cell.
    atomicAdd(&source_rate[c], -q / area);
}

__global__ __launch_bounds__(256, 4) void swe2d_coupling_structure_source_kernel(
    int32_t n_structures,
    const int32_t* __restrict__ structure_up_cell,
    const int32_t* __restrict__ structure_down_cell,
    const double* __restrict__ structure_flow,
    const double* __restrict__ cell_area,
    int32_t n_cells,
    double* __restrict__ source_rate)
{
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_structures) return;
    const int32_t cu = structure_up_cell[i];
    const int32_t cd = structure_down_cell[i];
    if (cu < 0 || cu >= n_cells || cd < 0 || cd >= n_cells) return;
    const double q = structure_flow[i];
    if (!isfinite(q) || q == 0.0) return;

    const double au = fmax(cell_area[cu], 1.0e-12);
    const double ad = fmax(cell_area[cd], 1.0e-12);
    // Positive q transfers mass from upstream cell -> downstream cell.
    atomicAdd(&source_rate[cu], -q / au);
    atomicAdd(&source_rate[cd],  q / ad);
}

__global__ __launch_bounds__(256, 4) void swe2d_coupling_bridge_source_kernel(
    int32_t n_bridges,
    const int32_t* __restrict__ bridge_up_cell,
    const int32_t* __restrict__ bridge_down_cell,
    const double* __restrict__ bridge_flow_cms,
    const double* __restrict__ bridge_loss_k_upstream,
    const double* __restrict__ bridge_loss_k_downstream,
    const double* __restrict__ cell_area,
    int32_t n_cells,
    double bridge_opening_width_m,
    double dt_s,
    double* __restrict__ source_rate)
{
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_bridges) return;

    const int32_t cu = bridge_up_cell[i];
    const int32_t cd = bridge_down_cell[i];
    if (cu < 0 || cu >= n_cells || cd < 0 || cd >= n_cells) return;

    const double q = bridge_flow_cms[i];
    if (!isfinite(q) || q == 0.0) return;

    const double au = fmax(cell_area[cu], 1.0e-12);
    const double ad = fmax(cell_area[cd], 1.0e-12);
    const double area_proxy = fmax(0.5 * (au + ad), 1.0e-12);
    const double char_len = fmax(bridge_opening_width_m, sqrt(area_proxy));
    const double vel_proxy = fabs(q) / area_proxy;
    const double k_up = fmax(0.0, bridge_loss_k_upstream[i]);
    const double k_down = fmax(0.0, bridge_loss_k_downstream[i]);

    double q_eff = q;
    if (k_up > 0.0) {
        q_eff /= (1.0 + k_up * dt_s * vel_proxy / fmax(char_len, 1.0e-12));
    }
    if (k_down > 0.0) {
        const double vel_after_up = fabs(q_eff) / area_proxy;
        q_eff /= (1.0 + k_down * dt_s * vel_after_up / fmax(char_len, 1.0e-12));
    }

    // Positive q transfers mass from upstream cell -> downstream cell.
    atomicAdd(&source_rate[cu], -q_eff / au);
    atomicAdd(&source_rate[cd],  q_eff / ad);
}

// ─────────────────────────────────────────────────────────────────────────────
// Face-based culvert coupling kernel
//
// Converts pre-computed culvert discharge Q_c into a three-component FVM
// face flux (mass + momentum) and accumulates into per-cell external flux
// arrays.  This replaces the cell-center source/sink for culverts when
// culvert_face_flux_mode == "face_flux".
//
// Theory:
//   Structure face normal: n̂ = (x_d - x_u) / |x_d - x_u|
//   Mass flux:   F_h  = Q_c * L_s
//   x-momentum:  F_hu = Q_c * u_donor * L_s  +  0.5 * g * h_s² * n_x * L_s
//   y-momentum:  F_hv = Q_c * v_donor * L_s  +  0.5 * g * h_s² * n_y * L_s
//
// where u_donor,v_donor are donor-cell velocity and h_s = max(h_donor - z_invert, 0).
// ─────────────────────────────────────────────────────────────────────────────
__global__ __launch_bounds__(256, 4) void swe2d_culvert_face_flux_kernel(
    int32_t n_culvert_faces,
    const double*  __restrict__ structure_flow,       // [n_structures] Q_c in model units
    const int32_t* __restrict__ culvert_struct_idx,    // [n_culvert_faces]
    const double*  __restrict__ face_nx,               // [n_culvert_faces]
    const double*  __restrict__ face_ny,               // [n_culvert_faces]
    const double*  __restrict__ face_width,             // [n_culvert_faces] L_s
    const int32_t* __restrict__ donor_cell,            // [n_culvert_faces]
    const int32_t* __restrict__ receiver_cell,         // [n_culvert_faces]
    const double*  __restrict__ invert_elev,            // [n_culvert_faces]
    const double*  __restrict__ depth_safety,           // [n_culvert_faces]
    const double*  __restrict__ donor_cell_area,        // [n_culvert_faces]
    const double*  __restrict__ cell_h,                // [n_cells]
    const double*  __restrict__ cell_hu,               // [n_cells]
    const double*  __restrict__ cell_hv,               // [n_cells]
    const double*  __restrict__ cell_zb,               // [n_cells]
    double gravity,
    double dt,
    double h_min,
    int32_t n_cells,
    double* __restrict__ ext_flux_h,                   // [n_cells]
    double* __restrict__ ext_flux_hu,                  // [n_cells]
    double* __restrict__ ext_flux_hv)                  // [n_cells]
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_culvert_faces) return;

    const int32_t si = culvert_struct_idx[i];
    const double Q_c = structure_flow[si];
    if (!isfinite(Q_c) || Q_c == 0.0) return;

    const int32_t cu = donor_cell[i];
    const int32_t cd = receiver_cell[i];
    if (cu < 0 || cu >= n_cells || cd < 0 || cd >= n_cells) return;

    // Determine flow direction: positive Q_c = upstream→downstream
    const double sign = (Q_c >= 0.0) ? 1.0 : -1.0;
    const int32_t donor    = (sign >= 0.0) ? cu : cd;
    const int32_t receiver = (sign >= 0.0) ? cd : cu;
    if (donor < 0 || donor >= n_cells || receiver < 0 || receiver >= n_cells) return;

    const double h_donor = cell_h[donor];
    const double hu_donor = cell_hu[donor];
    const double hv_donor = cell_hv[donor];
    const double zb_donor = cell_zb[donor];

    // Invert elevation for depth limiting
    const double invert = invert_elev[i];
    const double wse_donor = h_donor + zb_donor;
    const double depth_above_invert = fmax(0.0, wse_donor - invert);

    // Face width and normal
    const double L_s = face_width[i];
    const double nx = face_nx[i];
    const double ny = face_ny[i];
    const double alpha = depth_safety[i];
    const double A_donor = fmax(donor_cell_area[i], 1.0e-12);

    // ── Handle dry donor: allow culvert to wet downstream even when
    //    upstream cell is dry (embankment wetting front).  The culvert
    //    has already computed Q_c based on available head; we pass it
    //    through as-is with zero momentum.
    if (h_donor <= h_min) {
        // Dry donor: apply full culvert discharge to receiver only
        // (no donor removal since there's nothing to remove).
        atomicAdd(&ext_flux_h[receiver], Q_c);
        // No momentum flux from a dry donor
        return;
    }

    const double inv_h = 1.0 / fmax(h_donor, h_min);
    const double u_donor = hu_donor * inv_h;
    const double v_donor = hv_donor * inv_h;

    // ── Depth limiter: prevent drying the donor cell ──
    // Q_max = alpha * h_limit * A_donor / dt.
    double Q_lim = Q_c;
    const double h_limit = fmax(depth_above_invert, h_donor);
    const double max_flux = alpha * h_limit * A_donor / fmax(dt, 1.0e-12);
    if (fabs(Q_c) > max_flux && max_flux > 0.0) {
        Q_lim = sign * max_flux;
    }

    // ── Hydrostatic pressure at the face ──
    const double h_s = fmax(depth_above_invert, 0.0);

    // ── Three-component flux ──
    const double fh  = Q_lim;
    const double fhu = Q_lim * u_donor + 0.5 * gravity * h_s * h_s * nx * L_s;
    const double fhv = Q_lim * v_donor + 0.5 * gravity * h_s * h_s * ny * L_s;

    // Accumulate into per-cell external flux arrays (opposite signs for donor/receiver)
    atomicAdd(&ext_flux_h[donor],    -fh);
    atomicAdd(&ext_flux_hu[donor],   -fhu);
    atomicAdd(&ext_flux_hv[donor],   -fhv);
    atomicAdd(&ext_flux_h[receiver],  fh);
    atomicAdd(&ext_flux_hu[receiver], fhu);
    atomicAdd(&ext_flux_hv[receiver], fhv);
}

// ─────────────────────────────────────────────────────────────────────────────
// Culvert flow masking kernel
//
// Zeroes out structure_flow[i] for all culvert indices so that the
// swe2d_coupling_structure_source_kernel skips them (avoids double-counting
// when face-based coupling is active).
// ─────────────────────────────────────────────────────────────────────────────
__global__ void swe2d_mask_culvert_source_kernel(
    int32_t n_culvert,
    const int32_t* __restrict__ culvert_indices,
    double* __restrict__ structure_flow)
{
    const int32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_culvert) return;
    structure_flow[culvert_indices[j]] = 0.0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Culvert face-flux mass → depth-rate folding kernel
//
// Converts per-cell culvert mass flux (ext_struct_flux_h) into a depth rate
// (Q / A_cell) and adds it to d_external_source_mps.  This allows the
// culvert mass contribution to benefit from the same subcycling, rate
// limiting, and IMEX stage coupling as other external sources.
//
// Must run AFTER the source kernel (which accumulates non-culvert terms)
// and BEFORE the update kernel consumes d_external_source_mps.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void swe2d_fold_culvert_mass_to_source_kernel(
    int32_t n_cells,
    const double* __restrict__ ext_struct_flux_h,   // [n_cells] mass flux (L³/T)
    const double* __restrict__ cell_area,            // [n_cells] cell area (L²)
    double* __restrict__ source_rate)                // [n_cells] depth rate (L/T), atomicAdd
{
    const int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells) return;
    const double fh = ext_struct_flux_h[c];
    if (!isfinite(fh) || fh == 0.0) return;
    const double inv_a = 1.0 / fmax(cell_area[c], 1.0e-12);
    atomicAdd(&source_rate[c], fh * inv_a);
}

// ─────────────────────────────────────────────────────────────────────────────
// Enquiry-cell WSE correction kernel
//
// For each culvert face with enquiry-cell support, overwrites the WSE at the
// face cell (donor/receiver) with the total energy (WSE + v²/2g) sampled at
// an offset enquiry cell.  This lets the structure-flows kernel use approach-
// flow energy rather than the locally-drawn-down WSE at the face, producing
// a correct driving head for the culvert hydraulic solver.
//
// Must run AFTER the WSE array is built from h+zb and BEFORE the structure
// flows kernel.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void swe2d_apply_enquiry_wse_kernel(
    int32_t n_culvert_faces,
    const int32_t* __restrict__ d_enquiry_up_cell,  // [n_faces] enquiry cell for upstream
    const int32_t* __restrict__ d_enquiry_dn_cell,  // [n_faces] enquiry cell for downstream
    const int32_t* __restrict__ d_donor_cell,        // [n_faces] face upstream cell
    const int32_t* __restrict__ d_receiver_cell,     // [n_faces] face downstream cell
    const double*  __restrict__ d_cell_wse,          // [n_cells] WSE at all cells
    const double*  __restrict__ d_cell_h,            // [n_cells] depth
    const double*  __restrict__ d_cell_hu,           // [n_cells] x-momentum
    const double*  __restrict__ d_cell_hv,           // [n_cells] y-momentum
    double  gravity,
    double  h_min,
    double* __restrict__ d_cell_wse_out)             // [n_cells] modified WSE output
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_culvert_faces) return;

    // ── Upstream side: use enquiry cell WSE + velocity head ──────────
    const int32_t fc_up = d_donor_cell[i];
    const int32_t enq_up = d_enquiry_up_cell[i];
    if (enq_up >= 0 && enq_up != fc_up) {
        double wse = d_cell_wse[enq_up];
        const double h_enq = d_cell_h[enq_up];
        if (h_enq > h_min) {
            const double u = d_cell_hu[enq_up] / h_enq;
            const double v = d_cell_hv[enq_up] / h_enq;
            wse += 0.5 * (u*u + v*v) / gravity;  // velocity head
        }
        d_cell_wse_out[fc_up] = wse;
    }

    // ── Downstream side: use enquiry cell WSE + velocity head ────────
    const int32_t fc_dn = d_receiver_cell[i];
    const int32_t enq_dn = d_enquiry_dn_cell[i];
    if (enq_dn >= 0 && enq_dn != fc_dn) {
        double wse = d_cell_wse[enq_dn];
        const double h_enq = d_cell_h[enq_dn];
        if (h_enq > h_min) {
            const double u = d_cell_hu[enq_dn] / h_enq;
            const double v = d_cell_hv[enq_dn] / h_enq;
            wse += 0.5 * (u*u + v*v) / gravity;  // velocity head
        }
        d_cell_wse_out[fc_dn] = wse;
    }
}

__device__ __forceinline__ double bw2d_clamp(double x, double lo, double hi)
{
    return fmin(hi, fmax(lo, x));
}

__device__ __forceinline__ double bw2d_weir_q(double hu, double hd, double crest, double width, double coeff)
{
    const double cu = hu - crest;
    const double cd = hd - crest;
    if (cu <= 0.0 && cd <= 0.0) return 0.0;
    if (cu >= cd) {
        if (cu <= 0.0) return 0.0;
        return coeff * width * pow(cu, 1.5);
    }
    if (cd <= 0.0) return 0.0;
    return -coeff * width * pow(cd, 1.5);
}

__device__ __forceinline__ double bw2d_orifice_q(double hu, double hd, double area, double cd, double g)
{
    const double dh = hu - hd;
    if (area <= 0.0 || fabs(dh) <= 1.0e-12) return 0.0;
    const double q = cd * area * sqrt(fmax(0.0, 2.0 * g * fabs(dh)));
    return (dh >= 0.0) ? q : -q;
}

__device__ __forceinline__ double bw2d_circular_area(double diameter_m)
{
    if (diameter_m <= 0.0) return 0.0;
    const double r = 0.5 * diameter_m;
    return M_PI * r * r;
}

__device__ __forceinline__ double bw2d_equiv_diameter_from_area(double area_m2)
{
    if (area_m2 <= 0.0) return 0.0;
    return sqrt(4.0 * area_m2 / M_PI);
}

__device__ __forceinline__ double bw2d_pipe_manning_capacity_full(double diameter_m, double slope, double n)
{
    if (diameter_m <= 0.0 || n <= 0.0 || slope <= 0.0) return 0.0;
    const double area = bw2d_circular_area(diameter_m);
    const double rh = diameter_m / 4.0;
    return (1.486 / n) * area * pow(rh, 2.0 / 3.0) * sqrt(slope);  // US Manning constant
}

__device__ __forceinline__ double bw2d_rect_manning_capacity_full(double width_ft, double height_ft, double slope, double n)
{
    if (width_ft <= 0.0 || height_ft <= 0.0 || n <= 0.0 || slope <= 0.0) return 0.0;
    const double area_ft2 = width_ft * height_ft;
    const double perim_ft = 2.0 * (width_ft + height_ft);
    if (perim_ft <= 0.0) return 0.0;
    const double rh_ft = area_ft2 / perim_ft;
    return (1.486 / n) * area_ft2 * pow(rh_ft, 2.0 / 3.0) * sqrt(slope);  // US Manning constant
}

struct swe2d_culvert_xsect_cuda {
    int code;
    int is_rect;
    double y_full_ft;
    double a_full_ft2;
    double width_ft;
    double radius_ft;
};

struct swe2d_culvert_state_cuda {
    double y_full;
    double scf;
    double d_q_d_h;
    double q_critical;
    double kk;
    double mm;
    double ad;
    double h_plus;
    swe2d_culvert_xsect_cuda xsect;
};

__device__ __constant__ double SWE2D_CULVERT_PARAMS_CUDA[58][5] = {
    {0.0, 0.0, 0.0, 0.0, 0.00},
    {1.0, 0.0098, 2.00, 0.0398, 0.67}, {1.0, 0.0018, 2.00, 0.0292, 0.74}, {1.0, 0.0045, 2.00, 0.0317, 0.69},
    {1.0, 0.0078, 2.00, 0.0379, 0.69}, {1.0, 0.0210, 1.33, 0.0463, 0.75}, {1.0, 0.0340, 1.50, 0.0553, 0.54},
    {1.0, 0.0018, 2.50, 0.0300, 0.74}, {1.0, 0.0018, 2.50, 0.0243, 0.83},
    {1.0, 0.026, 1.0, 0.0347, 0.81}, {1.0, 0.061, 0.75, 0.0400, 0.80}, {1.0, 0.061, 0.75, 0.0423, 0.82},
    {2.0, 0.510, 0.667, 0.0309, 0.80}, {2.0, 0.486, 0.667, 0.0249, 0.83},
    {2.0, 0.515, 0.667, 0.0375, 0.79}, {2.0, 0.495, 0.667, 0.0314, 0.82}, {2.0, 0.486, 0.667, 0.0252, 0.865},
    {2.0, 0.545, 0.667, 0.04505, 0.73}, {2.0, 0.533, 0.667, 0.0425, 0.705}, {2.0, 0.522, 0.667, 0.0402, 0.68}, {2.0, 0.498, 0.667, 0.0327, 0.75},
    {2.0, 0.497, 0.667, 0.0339, 0.803}, {2.0, 0.493, 0.667, 0.0361, 0.806}, {2.0, 0.495, 0.667, 0.0386, 0.71},
    {2.0, 0.497, 0.667, 0.0302, 0.835}, {2.0, 0.495, 0.667, 0.0252, 0.881}, {2.0, 0.493, 0.667, 0.0227, 0.887},
    {1.0, 0.0083, 2.00, 0.0379, 0.69}, {1.0, 0.0145, 1.75, 0.0419, 0.64}, {1.0, 0.0340, 1.50, 0.0496, 0.57},
    {1.0, 0.0100, 2.00, 0.0398, 0.67}, {1.0, 0.0018, 2.50, 0.0292, 0.74}, {1.0, 0.0045, 2.00, 0.0317, 0.69},
    {1.0, 0.0100, 2.00, 0.0398, 0.67}, {1.0, 0.0018, 2.50, 0.0292, 0.74}, {1.0, 0.0095, 2.00, 0.0317, 0.69},
    {1.0, 0.0083, 2.00, 0.0379, 0.69}, {1.0, 0.0300, 1.00, 0.0463, 0.75}, {1.0, 0.0340, 1.50, 0.0496, 0.57},
    {1.0, 0.0300, 1.50, 0.0496, 0.57}, {1.0, 0.0088, 2.00, 0.0368, 0.68}, {1.0, 0.0030, 2.00, 0.0269, 0.77},
    {1.0, 0.0300, 1.50, 0.0496, 0.57}, {1.0, 0.0088, 2.00, 0.0368, 0.68}, {1.0, 0.0030, 2.00, 0.0269, 0.77},
    {1.0, 0.0083, 2.00, 0.0379, 0.69}, {1.0, 0.0300, 1.00, 0.0463, 0.75}, {1.0, 0.0340, 1.50, 0.0496, 0.57},
    {2.0, 0.534, 0.555, 0.0196, 0.90}, {2.0, 0.519, 0.640, 0.0210, 0.90},
    {2.0, 0.536, 0.622, 0.0368, 0.83}, {2.0, 0.5035, 0.719, 0.0478, 0.80}, {2.0, 0.547, 0.800, 0.0598, 0.75},
    {2.0, 0.475, 0.667, 0.0179, 0.97},
    {2.0, 0.560, 0.667, 0.0446, 0.85}, {2.0, 0.560, 0.667, 0.0378, 0.87},
    {2.0, 0.500, 0.667, 0.0446, 0.65}, {2.0, 0.500, 0.667, 0.0378, 0.71}
};

__device__ __forceinline__ double swe2d_culvert_area_ft2_cuda(const swe2d_culvert_xsect_cuda& x, double y_ft)
{
    if (x.is_rect) {
        const double y = fmax(0.0, fmin(y_ft, x.y_full_ft));
        return x.width_ft * y;
    }
    const double y = fmax(0.0, fmin(y_ft, 2.0 * x.radius_ft));
    if (y <= 0.0) return 0.0;
    const double arg = fmax(-1.0, fmin(1.0, (x.radius_ft - y) / x.radius_ft));
    const double theta = 2.0 * acos(arg);
    return 0.5 * x.radius_ft * x.radius_ft * (theta - sin(theta));
}

__device__ __forceinline__ double swe2d_culvert_top_width_ft_cuda(const swe2d_culvert_xsect_cuda& x, double y_ft)
{
    if (x.is_rect) return (y_ft > 0.0) ? x.width_ft : 0.0;
    const double y = fmax(0.0, fmin(y_ft, 2.0 * x.radius_ft));
    if (y <= 0.0) return 0.0;
    return 2.0 * sqrt(fmax(0.0, 2.0 * x.radius_ft * y - y * y));
}

__device__ __forceinline__ double swe2d_culvert_wetted_perimeter_ft_cuda(const swe2d_culvert_xsect_cuda& x, double y_ft)
{
    if (x.is_rect) {
        const double y = fmax(0.0, fmin(y_ft, x.y_full_ft));
        if (y <= 0.0) return 0.0;
        return x.width_ft + 2.0 * y;
    }
    const double y = fmax(0.0, fmin(y_ft, 2.0 * x.radius_ft));
    if (y <= 0.0) return 0.0;
    const double arg = fmax(-1.0, fmin(1.0, (x.radius_ft - y) / x.radius_ft));
    const double theta = 2.0 * acos(arg);
    return x.radius_ft * theta;
}

__device__ __forceinline__ double swe2d_culvert_hydraulic_radius_ft_cuda(const swe2d_culvert_xsect_cuda& x, double y_ft)
{
    const double area = swe2d_culvert_area_ft2_cuda(x, y_ft);
    const double perim = swe2d_culvert_wetted_perimeter_ft_cuda(x, y_ft);
    if (area <= 0.0 || perim <= 0.0) return 0.0;
    return area / perim;
}

__device__ __forceinline__ double swe2d_culvert_form1_eqn_cuda(double yc, swe2d_culvert_state_cuda* c)
{
    const double ac = swe2d_culvert_area_ft2_cuda(c->xsect, yc);
    const double wc = swe2d_culvert_top_width_ft_cuda(c->xsect, yc);
    const double yh = (wc > 0.0) ? (ac / wc) : 0.0;
    c->q_critical = ac * sqrt(USC_GRAVITY * yh);
    return c->h_plus - yc / c->y_full - yh / (2.0 * c->y_full)
        - c->kk * pow(c->q_critical / c->ad, c->mm);
}

__device__ __forceinline__ double swe2d_culvert_get_form1_flow_cuda(double h, swe2d_culvert_state_cuda* c)
{
    c->h_plus = h / c->y_full + c->scf;
    double a = fmax(1.0e-6, 0.01 * h);
    double b = fmax(a * 1.01, h);
    double fa = swe2d_culvert_form1_eqn_cuda(a, c);
    double fb = swe2d_culvert_form1_eqn_cuda(b, c);
    if (!(fa == 0.0 || fb == 0.0 || fa * fb < 0.0)) {
        for (int k = 1; k <= 40; ++k) {
            const double x = a + (b - a) * (static_cast<double>(k) / 41.0);
            const double fx = swe2d_culvert_form1_eqn_cuda(x, c);
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
            fb = swe2d_culvert_form1_eqn_cuda(b, c);
            if (fa * fb < 0.0) break;
        }
    }

    double lo = a;
    double hi = b;
    double flo = swe2d_culvert_form1_eqn_cuda(lo, c);
    double fhi = swe2d_culvert_form1_eqn_cuda(hi, c);
    double yc = 0.5 * (lo + hi);
    if (flo * fhi < 0.0) {
        for (int it = 0; it < 100; ++it) {
            yc = 0.5 * (lo + hi);
            const double fm = swe2d_culvert_form1_eqn_cuda(yc, c);
            if (fabs(fm) < 1.0e-3 || fabs(hi - lo) < 1.0e-3) break;
            if (flo * fm <= 0.0) {
                hi = yc;
                fhi = fm;
            } else {
                lo = yc;
                flo = fm;
            }
        }
    }
    (void)swe2d_culvert_form1_eqn_cuda(yc, c);
    return c->q_critical;
}

__device__ __forceinline__ double swe2d_culvert_get_unsubmerged_flow_cuda(int code, double h, swe2d_culvert_state_cuda* c)
{
    c->kk = SWE2D_CULVERT_PARAMS_CUDA[code][1];
    c->mm = SWE2D_CULVERT_PARAMS_CUDA[code][2];
    const double arg = h / c->y_full / c->kk;
    double q = 0.0;
    if (SWE2D_CULVERT_PARAMS_CUDA[code][0] == 1.0) {
        q = swe2d_culvert_get_form1_flow_cuda(h, c);
    } else {
        q = c->ad * pow(arg, 1.0 / c->mm);
    }
    c->d_q_d_h = (q / fmax(h, 1.0e-12)) / c->mm;
    return q;
}

__device__ __forceinline__ double swe2d_culvert_get_submerged_flow_cuda(int code, double h, swe2d_culvert_state_cuda* c)
{
    const double cc = SWE2D_CULVERT_PARAMS_CUDA[code][3];
    const double yy = SWE2D_CULVERT_PARAMS_CUDA[code][4];
    const double arg = (h / c->y_full - yy + c->scf) / cc;
    if (arg <= 0.0) {
        c->d_q_d_h = 0.0;
        return 1.0e20;
    }
    const double q = sqrt(arg) * c->ad;
    c->d_q_d_h = 0.5 * q / arg / c->y_full / cc;
    return q;
}

__device__ __forceinline__ double swe2d_culvert_get_transition_flow_cuda(int code, double h, double h1, double h2, swe2d_culvert_state_cuda* c)
{
    const double q1 = swe2d_culvert_get_unsubmerged_flow_cuda(code, h1, c);
    const double q2 = swe2d_culvert_get_submerged_flow_cuda(code, h2, c);
    const double q = q1 + (q2 - q1) * (h - h1) / (h2 - h1);
    c->d_q_d_h = (q2 - q1) / (h2 - h1);
    return q;
}

__device__ __forceinline__ double swe2d_culvert_inlet_controlled_flow_cfs_cuda(const swe2d_culvert_xsect_cuda& xsect, double slope, double h_ft)
{
    const int code = max(1, min(57, xsect.code));
    swe2d_culvert_state_cuda c;
    c.y_full = xsect.y_full_ft;
    c.ad = xsect.a_full_ft2 * sqrt(fmax(1.0e-12, xsect.y_full_ft));
    c.xsect = xsect;
    c.scf = (code == 5 || code == 37 || code == 46) ? (-7.0 * slope) : (0.5 * slope);

    const double y = fmax(0.0, h_ft);
    const double y2 = c.y_full * (16.0 * SWE2D_CULVERT_PARAMS_CUDA[code][3] + SWE2D_CULVERT_PARAMS_CUDA[code][4] - c.scf);
    if (y >= y2) return swe2d_culvert_get_submerged_flow_cuda(code, y, &c);
    const double y1 = 0.95 * c.y_full;
    if (y <= y1) return swe2d_culvert_get_unsubmerged_flow_cuda(code, y, &c);
    return swe2d_culvert_get_transition_flow_cuda(code, y, y1, y2, &c);
}

__device__ __forceinline__ double swe2d_culvert_critical_depth_ft_cuda(const swe2d_culvert_xsect_cuda& xsect, double q_cfs)
{
    if (q_cfs <= 0.0) return 0.0;
    if (xsect.is_rect) {
        const double q_unit = q_cfs / fmax(1.0e-12, xsect.width_ft);
        return fmin(pow((q_unit * q_unit)  / USC_GRAVITY, 1.0 / 3.0), xsect.y_full_ft);
    }
    const double target = q_cfs * q_cfs  / USC_GRAVITY;
    double lo = 1.0e-4 * xsect.y_full_ft;
    double hi = xsect.y_full_ft;
    auto residual = [&](double y) {
        const double a = swe2d_culvert_area_ft2_cuda(xsect, y);
        const double t = swe2d_culvert_top_width_ft_cuda(xsect, y);
        return (t > 0.0) ? (a * a * a / t - target) : 1.0e20;
    };
    double flo = residual(lo);
    double fhi = residual(hi);
    if (fhi <= 0.0) return xsect.y_full_ft;
    if (flo >= 0.0) return lo;
    for (int i = 0; i < 80; ++i) {
        const double mid = 0.5 * (lo + hi);
        const double fm = residual(mid);
        if (fabs(fm) < 1.0e-9 * fmax(target, 1.0) || (hi - lo) < 1.0e-7) return mid;
        if (flo * fm <= 0.0) {
            hi = mid;
            fhi = fm;
        } else {
            lo = mid;
            flo = fm;
        }
    }
    return 0.5 * (lo + hi);
}

__device__ __forceinline__ double swe2d_culvert_specific_energy_ft_cuda(const swe2d_culvert_xsect_cuda& xsect, double q_cfs, double depth_ft)
{
    const double a = swe2d_culvert_area_ft2_cuda(xsect, depth_ft);
    const double v = (a > 0.0) ? (q_cfs / a) : 0.0;
    return depth_ft + v * v / (2.0 * USC_GRAVITY);
}

__device__ __forceinline__ double swe2d_culvert_friction_slope_cuda(const swe2d_culvert_xsect_cuda& xsect, double q_cfs, double n_value, double depth_ft)
{
    if (depth_ft <= 0.0 || n_value <= 0.0) return 0.0;
    const double area = swe2d_culvert_area_ft2_cuda(xsect, depth_ft);
    const double rh = swe2d_culvert_hydraulic_radius_ft_cuda(xsect, depth_ft);
    if (area <= 0.0 || rh <= 0.0) return 0.0;
    const double conveyance = (1.49 / n_value) * area * pow(rh, 2.0 / 3.0);
    if (conveyance <= 0.0) return 0.0;
    return pow(q_cfs / conveyance, 2.0);
}

__device__ __forceinline__ double swe2d_culvert_supercritical_depth_for_energy_cuda(const swe2d_culvert_xsect_cuda& xsect, double q_cfs, double target_energy)
{
    if (q_cfs <= 0.0) return 0.0;
    const double dc = swe2d_culvert_critical_depth_ft_cuda(xsect, q_cfs);
    const double eps = fmax(1.0e-6, 1.0e-6 * xsect.y_full_ft);
    const double lo = eps;
    const double hi = fmax(eps, fmin(dc, xsect.y_full_ft - eps));
    if (hi <= lo) return fmax(eps, fmin(dc, xsect.y_full_ft - eps));

    auto residual = [&](double d) { return swe2d_culvert_specific_energy_ft_cuda(xsect, q_cfs, d) - target_energy; };
    const int samples = 240;
    const double step = (hi - lo) / static_cast<double>(max(samples - 1, 1));
    double best_d = lo;
    double best_r = residual(lo);
    double prev_d = lo;
    double prev_r = best_r;
    bool have_bracket = false;
    double a = lo;
    double b = hi;
    for (int i = 1; i < samples; ++i) {
        const double d = lo + i * step;
        const double r = residual(d);
        if (fabs(r) < fabs(best_r)) {
            best_d = d;
            best_r = r;
        }
        if (prev_r * r < 0.0) {
            have_bracket = true;
            a = prev_d;
            b = d;
            break;
        }
        prev_d = d;
        prev_r = r;
    }
    if (!have_bracket) return best_d;
    double fa = residual(a);
    for (int i = 0; i < 80; ++i) {
        const double m = 0.5 * (a + b);
        const double fm = residual(m);
        if (fabs(fm) < 1.0e-10 || fabs(b - a) < 1.0e-6) return m;
        if (fa * fm <= 0.0) b = m;
        else {
            a = m;
            fa = fm;
        }
    }
    return 0.5 * (a + b);
}

__device__ __forceinline__ void swe2d_direct_step_culvert_upstream_energy_cuda(
    const swe2d_culvert_xsect_cuda& xsect,
    double q_cfs,
    double n_value,
    double slope,
    double length_ft,
    double tailwater_depth,
    double* e_upstream_ft,
    double* y_upstream_ft)
{
    if (q_cfs <= 0.0) {
        *e_upstream_ft = 0.0;
        *y_upstream_ft = 0.0;
        return;
    }

    const double dc = swe2d_culvert_critical_depth_ft_cuda(xsect, q_cfs);
    const double y_full = xsect.y_full_ft;
    const double eps = fmax(1.0e-6, 1.0e-6 * y_full);
    const double y_ds = fmin(fmax(tailwater_depth, dc), y_full);
    const double step_depth = fmin(fmax(0.01, 0.02 * y_full), 0.50);

    if (y_ds >= y_full - eps) {
        const double sf_full = swe2d_culvert_friction_slope_cuda(xsect, q_cfs, n_value, y_full - eps);
        const double e_full = swe2d_culvert_specific_energy_ft_cuda(xsect, q_cfs, y_full - eps);
        *e_upstream_ft = e_full + fmax(0.0, sf_full - slope) * length_ft;
        *y_upstream_ft = y_full - eps;
        return;
    }

    double distance = 0.0;
    double y_cur = fmax(y_ds, eps);
    double e_cur = swe2d_culvert_specific_energy_ft_cuda(xsect, q_cfs, y_cur);

    while (distance < length_ft - 1.0e-8) {
        if (y_cur >= y_full - eps) {
            const double sf_full = swe2d_culvert_friction_slope_cuda(xsect, q_cfs, n_value, y_full - eps);
            const double rem = length_ft - distance;
            *e_upstream_ft = e_cur + fmax(0.0, sf_full - slope) * rem;
            *y_upstream_ft = y_full;
            return;
        }

        double dy = fmin(step_depth, y_full - y_cur);
        bool have_step = false;
        double y_next = y_cur;
        double dx = 0.0;
        for (int k = 0; k < 10; ++k) {
            const double y_try = fmin(y_cur + dy, y_full);
            const double sf_from = swe2d_culvert_friction_slope_cuda(xsect, q_cfs, n_value, y_cur);
            const double sf_to = swe2d_culvert_friction_slope_cuda(xsect, q_cfs, n_value, y_try);
            const double sf_avg = 0.5 * (sf_from + sf_to);
            const double denom = slope - sf_avg;
            if (fabs(denom) >= 1.0e-12) {
                const double e_to = swe2d_culvert_specific_energy_ft_cuda(xsect, q_cfs, y_try);
                const double dx_try = (e_cur - e_to) / denom;
                if (isfinite(dx_try) && dx_try > 0.0) {
                    have_step = true;
                    y_next = y_try;
                    dx = dx_try;
                    break;
                }
            }
            dy *= 0.5;
            if (dy <= eps) break;
        }

        if (!have_step) {
            const double y_super = swe2d_culvert_supercritical_depth_for_energy_cuda(xsect, q_cfs, e_cur);
            *e_upstream_ft = swe2d_culvert_specific_energy_ft_cuda(xsect, q_cfs, y_super);
            *y_upstream_ft = y_super;
            return;
        }

        if (distance + dx >= length_ft) {
            const double remaining = length_ft - distance;
            // Newton-Raphson (via secant) to find y_next such that
            //   F(y) = E(y) - E_cur - remaining * (S0 - 0.5*(Sf_cur + Sf(y))) = 0.
            // Uses the direct-step equation directly as the residual, converging
            // in ~6 iterations instead of 80 bisection steps.
            const double sf_cur = swe2d_culvert_friction_slope_cuda(xsect, q_cfs, n_value, y_cur);
            auto residual = [&](double y) {
                const double sf_y = swe2d_culvert_friction_slope_cuda(xsect, q_cfs, n_value, y);
                const double sf_avg = 0.5 * (sf_cur + sf_y);
                const double e_y = swe2d_culvert_specific_energy_ft_cuda(xsect, q_cfs, y);
                return e_y - e_cur - remaining * (slope - sf_avg);
            };
            double a = y_cur;
            double b = y_next;
            double fa = residual(a);
            double fb = residual(b);
            // Ensure bracket
            for (int br = 0; br < 10 && fa * fb > 0.0; ++br) {
                if (fabs(fa) < fabs(fb)) {
                    a = fmax(eps, a - (b - a));
                    fa = residual(a);
                } else {
                    b = fmin(y_full, b + (b - a));
                    fb = residual(b);
                }
            }
            double y_best = (fabs(fa) < fabs(fb)) ? a : b;
            double f_best = (fabs(fa) < fabs(fb)) ? fa : fb;
            for (int iter = 0; iter < 12; ++iter) {
                if (fabs(fb - fa) < 1.0e-30) break;
                const double y_mid = b - fb * (b - a) / (fb - fa);
                const double fm = residual(y_mid);
                if (fabs(fm) < fabs(f_best)) { y_best = y_mid; f_best = fm; }
                if (fabs(fm) < 1.0e-10 || fabs(b - a) < eps) { y_best = y_mid; break; }
                a = b; fa = fb;
                b = y_mid; fb = fm;
            }
            *e_upstream_ft = swe2d_culvert_specific_energy_ft_cuda(xsect, q_cfs, y_best);
            *y_upstream_ft = y_best;
            return;
        }

        distance += dx;
        y_cur = y_next;
        e_cur = swe2d_culvert_specific_energy_ft_cuda(xsect, q_cfs, y_cur);
    }

    *e_upstream_ft = e_cur;
    *y_upstream_ft = y_cur;
}

__device__ __forceinline__ double swe2d_culvert_outlet_control_flow_cms_cuda(
    const swe2d_culvert_xsect_cuda& xsect,
    double available_head_up,
    double tailwater_depth,
    double length_ft,
    double slope,
    double roughness_n,
    double entrance_loss_k,
    double exit_loss_k,
    double q_hint)
{
    // NOTE: Despite the "_cms" name suffix, this function returns CFS
    // (cubic feet per second).  The name is retained for API compatibility;
    // the previous implementation erroneously divided by USC_FT3_PER_SI_M3
    // which caused a unit mismatch with q_inlet (also CFS) and the final
    // kernel conversion `sign * q / to_ft³`.  The CMS conversion is now
    // handled exclusively by the final `sign * q / (to_ft³)` in the caller.
    if (available_head_up <= 0.0) return 0.0;

    auto required_head_ft = [&](double q_cfs) {
        if (q_cfs <= 0.0) return 0.0;
        double e_up = 0.0;
        double y_up = 0.0;
        swe2d_direct_step_culvert_upstream_energy_cuda(
            xsect,
            q_cfs,
            fmax(1.0e-6, roughness_n),
            fmax(1.0e-6, slope),
            fmax(1.0, length_ft),
            fmax(0.0, tailwater_depth),
            &e_up,
            &y_up);
        const double area = fmax(1.0e-9, swe2d_culvert_area_ft2_cuda(xsect, fmax(1.0e-6, fmin(y_up, xsect.y_full_ft))));
        const double vel = q_cfs / area;
        const double hv = (fmax(0.0, entrance_loss_k) + fmax(0.0, exit_loss_k)) * vel * vel / (2.0 * USC_GRAVITY);
        return e_up + hv;
    };

    // Illinois algorithm: secant with stalling-side damping for guaranteed
    // convergence on monotonic F(Q) = required_head(Q) - available_head.
    // Converges in ~8-10 iterations vs 12+ for pure secant; handles flat
    // tails and near-zero-loss cases that cause pure secant to diverge.
    double q_lo = 0.0;
    double f_lo = -available_head_up;  // F(0) = -available_head
    double q_hi = fmax(1.0, q_hint * 2.0);
    double f_hi = required_head_ft(q_hi) - available_head_up;
    for (int br = 0; br < 12 && f_hi < 0.0; ++br) {
        q_lo = q_hi; f_lo = f_hi;
        q_hi *= 2.0;
        f_hi = required_head_ft(q_hi) - available_head_up;
    }
    if (f_hi < 0.0) {
        return q_hi;
    }

    // Illinois: track which side last moved so we halve the stalling f-value.
    int side = 0;  // 0 = lo was updated last, 1 = hi was updated last
    for (int iter = 0; iter < 12; ++iter) {
        const double denom = f_hi - f_lo;
        if (fabs(denom) < 1.0e-30) break;
        double q_mid = (q_lo * f_hi - q_hi * f_lo) / denom;  // secant step
        // Fall back to bisection if secant steps outside bracket
        if (q_mid <= q_lo || q_mid >= q_hi) {
            q_mid = 0.5 * (q_lo + q_hi);
        }
        const double f_mid = required_head_ft(q_mid) - available_head_up;
        if (fabs(f_mid) < 1.0e-8 * available_head_up) {
            return fmax(0.0, q_mid);
        }
        if (f_lo * f_mid < 0.0) {
            // Root is between lo and mid
            q_hi = q_mid; f_hi = f_mid;
            if (side == 1) f_lo *= 0.5;  // lo was stalling, halve it
            side = 1;
        } else {
            // Root is between mid and hi
            q_lo = q_mid; f_lo = f_mid;
            if (side == 0) f_hi *= 0.5;  // hi was stalling, halve it
            side = 0;
        }
    }
    return fmax(0.0, 0.5 * (q_lo + q_hi));
}

// Forward-declare table lookup for use inside the structure flow kernel.
__device__ double swe2d_culvert_table_lookup_cuda(
    int32_t ci, double hw_ft, double tw_ft,
    int32_t n_culverts, const double* d_table_header, const double* d_table_data,
    int32_t n_hw_global, int32_t n_tw_global);

__global__ __launch_bounds__(256, 4) void swe2d_compute_structure_flows_kernel(
    int32_t n_cells,
    int32_t n_structures,
    const double* __restrict__ cell_wse,
    const int32_t* __restrict__ structure_type,
    const int32_t* __restrict__ upstream_cell,
    const int32_t* __restrict__ downstream_cell,
    const double* __restrict__ crest_elev,
    const double* __restrict__ width,
    const double* __restrict__ height,
    const double* __restrict__ diameter,
    const double* __restrict__ length,
    const double* __restrict__ roughness_n,
    const double* __restrict__ coeff,
    const double* __restrict__ cd,
    const double* __restrict__ opening,
    const double* __restrict__ q_pump,
    const double* __restrict__ max_flow,
    const int32_t* __restrict__ culvert_code,
    const int32_t* __restrict__ culvert_shape,
    const double* __restrict__ culvert_rise,
    const double* __restrict__ culvert_span,
    const double* __restrict__ culvert_area,
    const double* __restrict__ culvert_barrels,
    const double* __restrict__ culvert_slope,
    const double* __restrict__ inlet_invert_elev,
    const double* __restrict__ outlet_invert_elev,
    const double* __restrict__ entrance_loss_k,
    const double* __restrict__ exit_loss_k,
    const int32_t* __restrict__ embankment_enabled,
    const double* __restrict__ embankment_crest_elev,
    const double* __restrict__ embankment_overflow_width,
    const double* __restrict__ embankment_weir_coeff,
    double gravity,
    double model_to_ft,
    double* __restrict__ structure_flow,
    const double* __restrict__ prev_structure_flow,
    int32_t culvert_solver_mode,
    const double* __restrict__ culvert_table_header,
    const double* __restrict__ culvert_table_data,
    int32_t culvert_table_n_hw,
    int32_t culvert_table_n_tw)
{
    const int32_t i = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n_structures) return;

    const int32_t iu = upstream_cell[i];
    const int32_t id = downstream_cell[i];
    if (iu < 0 || iu >= n_cells || id < 0 || id >= n_cells) return;

    const double wu = cell_wse[iu];
    const double wd = cell_wse[id];
    const double crest = crest_elev[i];
    const double qmax = (isfinite(max_flow[i]) ? fmax(0.0, max_flow[i]) : -1.0);

    if (structure_type[i] == 1) {
        double q = bw2d_weir_q(wu, wd, crest, width[i], coeff[i]);
        if (qmax >= 0.0) q = fmax(-qmax, fmin(q, qmax));
        structure_flow[i] = q;
        return;
    }
    if (structure_type[i] == 3) {
        const double area = fmax(0.0, opening[i]) * fmax(0.0, width[i]) * fmax(0.0, height[i]);
        double q = bw2d_orifice_q(wu, wd, area, cd[i], gravity);
        if (qmax >= 0.0) q = fmax(-qmax, fmin(q, qmax));
        structure_flow[i] = q;
        return;
    }
    if (structure_type[i] == 4) {
        const double area = fmax(0.0, opening[i]) * fmax(0.0, width[i]) * fmax(0.0, height[i]);
        const double loss_scale = fmax(1.0e-6, 1.0 + fmax(0.0, entrance_loss_k[i]) + fmax(0.0, exit_loss_k[i]));
        const double dh = wu - wd;
        if (area > 0.0 && fabs(dh) > 1.0e-12) {
            double q = area * sqrt(fmax(0.0, 2.0 * gravity * fabs(dh))) / loss_scale;
            if (qmax >= 0.0) q = fmin(q, qmax);
            structure_flow[i] = (dh >= 0.0) ? q : -q;
        }
        return;
    }
    if (structure_type[i] == 5) {
        double q = fmax(0.0, q_pump[i]);
        if (qmax >= 0.0) q = fmin(q, qmax);
        structure_flow[i] = (wu >= wd) ? q : -q;
        return;
    }
    if (structure_type[i] != 2) return;

    // ── Culvert path: convert model-unit inputs to feet for HDS-5 ──
    const double to_ft = fmax(1.0e-6, model_to_ft);
    const double sign = (wu >= wd) ? 1.0 : -1.0;
    const double upstream_wse = (sign >= 0.0) ? wu : wd;
    const double downstream_wse = (sign >= 0.0) ? wd : wu;
    const double upstream_invert = (sign >= 0.0) ? inlet_invert_elev[i] : outlet_invert_elev[i];
    const double downstream_invert = (sign >= 0.0) ? outlet_invert_elev[i] : inlet_invert_elev[i];
    const double available_head_up_ft = fmax(0.0, (upstream_wse - upstream_invert) * to_ft);
    const double tailwater_depth_ft = fmax(0.0, (downstream_wse - downstream_invert) * to_ft);
    const double len_ft = fmax(0.1, length[i] * to_ft);

    double slope = culvert_slope[i];
    if (!(slope > 0.0)) {
        slope = fabs((upstream_invert - downstream_invert) * to_ft) / len_ft;
    }
    slope = fmax(1.0e-6, slope);

    const double rise_ft = fmax(0.0, (culvert_rise[i] > 0.0 ? culvert_rise[i] : fmax(height[i], diameter[i])) * to_ft);
    const double span_ft = fmax(0.0, (culvert_span[i] > 0.0 ? culvert_span[i] : fmax(width[i], rise_ft / to_ft)) * to_ft);
    const int code = max(1, min(57, static_cast<int>(culvert_code[i])));

    swe2d_culvert_xsect_cuda xsect{};
    xsect.code = code;
    xsect.is_rect = (culvert_shape[i] == 1) ? 1 : 0;
    if (xsect.is_rect) {
        xsect.width_ft = fmax(1.0e-6, span_ft);
        xsect.y_full_ft = fmax(1.0e-6, rise_ft);
        xsect.a_full_ft2 = xsect.width_ft * xsect.y_full_ft;
        xsect.radius_ft = 0.0;
    } else {
        const double dia_ft = fmax(1.0e-6, fmax(diameter[i] * to_ft, rise_ft));
        xsect.radius_ft = 0.5 * dia_ft;
        xsect.y_full_ft = dia_ft;
        xsect.a_full_ft2 = M_PI * xsect.radius_ft * xsect.radius_ft;
        xsect.width_ft = 0.0;
    }

    const double q_inlet = fmax(0.0, swe2d_culvert_inlet_controlled_flow_cfs_cuda(xsect, slope, fmax(0.0, available_head_up_ft)));

    double area_ft2 = fmax(0.0, culvert_area[i] * to_ft * to_ft);
    if (area_ft2 <= 0.0 && fmax(diameter[i] * to_ft, rise_ft) > 0.0 && culvert_shape[i] == 0) {
        area_ft2 = bw2d_circular_area(fmax(diameter[i] * to_ft, rise_ft));
    }

    double q_orifice = 0.0;
    if (area_ft2 > 0.0) {
        q_orifice = fabs(bw2d_orifice_q(available_head_up_ft, tailwater_depth_ft, area_ft2, cd[i], USC_GRAVITY));
        if (qmax >= 0.0) q_orifice = fmin(q_orifice, qmax);
    }

    double q_manning_cap = 0.0;
    if (xsect.is_rect) {
        q_manning_cap = bw2d_rect_manning_capacity_full(xsect.width_ft, xsect.y_full_ft, slope, roughness_n[i]);
    } else {
        const double dia_for_cap_ft = fmax(fmax(diameter[i] * to_ft, rise_ft), bw2d_equiv_diameter_from_area(fmax(0.0, area_ft2)));
        if (dia_for_cap_ft > 0.0) {
            q_manning_cap = bw2d_pipe_manning_capacity_full(dia_for_cap_ft, slope, roughness_n[i]);
        }
    }

    // Use the previous step's flow as a secant-solver hint, falling back
    // to the computed max of inlet/orifice/Manning estimates.
    double q_hint = (prev_structure_flow) ? fabs(prev_structure_flow[i]) : 0.0;
    if (!(q_hint > 0.0 && isfinite(q_hint))) {
        q_hint = fmax(1.0, fmax(q_inlet, fmax(q_orifice, q_manning_cap)));
    }

    double q_outlet = 0.0;
    if (culvert_solver_mode == 1 && culvert_table_data && culvert_table_header) {
        // Table lookup: bilinear interpolation from pre-computed Q(hw,tw) grid.
        q_outlet = swe2d_culvert_table_lookup_cuda(
            i,  // culvert index (local to the table, matches upload order)
            fmax(0.0, available_head_up_ft),
            fmax(0.0, tailwater_depth_ft),
            n_structures,
            culvert_table_header,
            culvert_table_data,
            culvert_table_n_hw,
            culvert_table_n_tw);
    } else {
        // Direct secant solver (default)
        q_outlet = swe2d_culvert_outlet_control_flow_cms_cuda(
            xsect,
            fmax(0.0, available_head_up_ft),
            fmax(0.0, tailwater_depth_ft),
            fmax(0.1, len_ft),
            fmax(1.0e-6, slope),
            fmax(1.0e-6, roughness_n[i]),
            entrance_loss_k[i],
            exit_loss_k[i],
            q_hint);
    }

    double q = fmax(0.0, fmin(q_inlet, q_outlet > 0.0 ? q_outlet : q_inlet));
    if (q_orifice > 0.0) q = (q > 0.0) ? fmin(q, q_orifice) : q_orifice;
    if (q_manning_cap > 0.0) q = (q > 0.0) ? fmin(q, q_manning_cap) : q_manning_cap;

    if (embankment_enabled[i] != 0) {
        const double q_emb = fabs(bw2d_weir_q(
            upstream_wse * to_ft,
            downstream_wse * to_ft,
            embankment_crest_elev[i] * to_ft,
            fmax(0.0, embankment_overflow_width[i]) * to_ft,
            fmax(1.0e-6, embankment_weir_coeff[i])));
        q += q_emb;
    }

    q *= fmax(1.0, culvert_barrels[i]);
    if (qmax >= 0.0) q = fmin(q, qmax);
    // Convert from CFS back to model units: ÷ to_ft³
    structure_flow[i] = sign * q / (to_ft * to_ft * to_ft);

    // ── Diagnostic: print culvert params for first structure (index 0) ──
    // Enable by compiling with -DCULVERT_DIAG or uncommenting the #define below.
    // Device-side printf is SLOW — don't leave enabled in production runs.
    #ifdef CULVERT_DIAG
    if (i == 0) {
        printf("[CULVERT_DIAG] i=%d type=%d shape=%d code=%d\n",
               i, structure_type[i], culvert_shape[i], code);
        printf("[CULVERT_DIAG] wu=%.4f wd=%.4f invert_up=%.4f invert_dn=%.4f\n",
               wu, wd, upstream_invert, downstream_invert);
        printf("[CULVERT_DIAG] head_ft=%.4f tw_ft=%.4f len_ft=%.4f slope=%.6f\n",
               available_head_up_ft, tailwater_depth_ft, len_ft, slope);
        printf("[CULVERT_DIAG] span_ft=%.4f rise_ft=%.4f area_ft2=%.4f\n",
               span_ft, rise_ft, xsect.a_full_ft2);
        printf("[CULVERT_DIAG] q_inlet=%.4f q_outlet=%.4f q_orifice=%.4f q_manning=%.4f\n",
               q_inlet, q_outlet, q_orifice, q_manning_cap);
        printf("[CULVERT_DIAG] barrels=%.1f qmax=%.4f q_final_CFS=%.4f to_ft=%.6f\n",
               culvert_barrels[i], qmax, q, to_ft);
        printf("[CULVERT_DIAG] structure_flow=%.6f (model units)\n",
               structure_flow[i]);
    }
    #endif
}

__device__ __forceinline__ void swe2d_circular_section_cuda(
    double depth_m,
    double diameter_m,
    double* area,
    double* perimeter)
{
    const double D = fmax(1.0e-9, diameter_m);
    const double y = fmax(0.0, fmin(depth_m, D));
    if (y <= 0.0) {
        *area = 0.0;
        *perimeter = 0.0;
        return;
    }
    if (y >= D) {
        *area = 0.25 * M_PI * D * D;
        *perimeter = M_PI * D;
        return;
    }
    const double arg = fmax(-1.0, fmin(1.0, 1.0 - 2.0 * y / D));
    const double theta = 2.0 * acos(arg);
    *area = (D * D / 8.0) * (theta - sin(theta));
    *perimeter = 0.5 * D * theta;
}

__global__ __launch_bounds__(256, 4) void swe2d_drainage_link_kernel(
    int32_t n_links,
    const int32_t* __restrict__ link_from,
    const int32_t* __restrict__ link_to,
    const double* __restrict__ link_length,
    const double* __restrict__ link_roughness_n,
    const double* __restrict__ link_diameter,
    const double* __restrict__ link_max_flow,
    const double* __restrict__ node_invert_elev,
    const double* __restrict__ node_depth,
    const double* __restrict__ link_flow_prev,
    double dt_s,
    double gravity,
    int32_t solver_mode,
    double head_deadband_m,
    double dynamic_flow_relaxation,
    double* __restrict__ link_flow_out,
    double* __restrict__ node_net_q)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_links) return;
    const int32_t n0 = link_from[i];
    const int32_t n1 = link_to[i];
    if (n0 < 0 || n1 < 0) {
        link_flow_out[i] = 0.0;
        return;
    }

    const double d = fmax(0.0, link_diameter[i]);
    if (d <= 0.0) {
        link_flow_out[i] = 0.0;
        return;
    }
    const double L = fmax(1.0, link_length[i]);
    const double n_mann = fmax(1.0e-6, link_roughness_n[i]);
    const double h0 = node_invert_elev[n0] + fmax(0.0, node_depth[n0]);
    const double h1 = node_invert_elev[n1] + fmax(0.0, node_depth[n1]);
    const double dh_raw = h0 - h1;
    const double deadband = fmax(0.0, head_deadband_m);
    double dh = 0.0;
    if (fabs(dh_raw) > deadband) {
        dh = copysign(fabs(dh_raw) - deadband, dh_raw);
    }
    if (fabs(dh) <= 1.0e-12) {
        link_flow_out[i] = 0.0;
        return;
    }

    const double depth0 = fmax(0.0, fmin(h0 - node_invert_elev[n0], d));
    const double depth1 = fmax(0.0, fmin(h1 - node_invert_elev[n1], d));
    double area = 0.0;
    double perimeter = 0.0;

    if (solver_mode == 0) {
        const double crown0 = node_invert_elev[n0] + d;
        const double crown1 = node_invert_elev[n1] + d;
        if (h0 >= crown0 && h1 >= crown1) {
            area = 0.25 * M_PI * d * d;
            perimeter = M_PI * d;
        } else {
            swe2d_circular_section_cuda(0.5 * (depth0 + depth1), d, &area, &perimeter);
        }
    } else {
        swe2d_circular_section_cuda(0.5 * (depth0 + depth1), d, &area, &perimeter);
    }

    if (area <= 0.0 || perimeter <= 0.0) {
        link_flow_out[i] = 0.0;
        return;
    }
    const double r_h = area / perimeter;

    double q = 0.0;
    const double k2 = c_k_mann * c_k_mann;
    if (solver_mode == 0) {
        const double C_fric = (n_mann * n_mann * L) / (k2 * area * area * pow(r_h, 4.0 / 3.0));
        const double C_minor = (0.5 + 1.0) / (2.0 * fmax(gravity, 1.0e-6) * area * area);
        const double C_total = C_fric + C_minor;
        q = (C_total > 0.0) ? sqrt(fabs(dh) / C_total) : 0.0;
    } else if (solver_mode == 1) {
        const double s_w = fabs(dh) / L;
        q = (c_k_mann / n_mann) * area * pow(r_h, 2.0 / 3.0) * sqrt(s_w);
    } else {
        const double q_old = link_flow_prev ? link_flow_prev[i] : 0.0;
        const double pressure_accel = gravity * area * dh / L;
        double friction_denom = 0.0;
        if (fabs(q_old) > 0.0 && r_h > 0.0) {
            friction_denom = dt_s * gravity * n_mann * n_mann * fabs(q_old)
                / (k2 * area * pow(r_h, 4.0 / 3.0));
        }
        const double q_candidate = (q_old + dt_s * pressure_accel) / (1.0 + friction_denom);
        const double relax = fmin(1.0, fmax(0.0, dynamic_flow_relaxation));
        q = (1.0 - relax) * q_old + relax * q_candidate;
    }

    if (solver_mode != 2) {
        q = (dh >= 0.0) ? fabs(q) : -fabs(q);
    }

    const double q_cap = link_max_flow[i];
    if (isfinite(q_cap) && q_cap > 0.0) {
        q = fmax(-q_cap, fmin(q_cap, q));
    }

    link_flow_out[i] = q;
    atomicAdd(&node_net_q[n0], -q);
    atomicAdd(&node_net_q[n1],  q);
}

__global__ void swe2d_drainage_node_update_kernel(
    int32_t n_nodes,
    const double* __restrict__ node_max_depth,
    const double* __restrict__ node_surface_area,
    const double* __restrict__ node_net_q,
    const double* __restrict__ node_depth_in,
    double dt_s,
    double* __restrict__ node_depth_out)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_nodes) return;
    const double area = fmax(1.0, node_surface_area[i]);
    const double d0 = fmax(0.0, node_depth_in[i]);
    double d1 = d0 + dt_s * node_net_q[i] / area;
    d1 = fmax(0.0, fmin(node_max_depth[i], d1));
    node_depth_out[i] = d1;
}

__global__ void swe2d_drainage_pipe_end_qleave_kernel(
    int32_t n_links,
    const int32_t* __restrict__ link_from,
    const int32_t* __restrict__ link_to,
    const double* __restrict__ link_flow_prev,
    double* __restrict__ node_qleave)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_links) return;
    const int32_t n0 = link_from[i];
    const int32_t n1 = link_to[i];
    if (n0 < 0 || n1 < 0) return;
    const double q = link_flow_prev ? link_flow_prev[i] : 0.0;
    if (!isfinite(q) || q == 0.0) return;
    // Positive q is defined from link_from -> link_to.
    atomicAdd(&node_qleave[n0], q);
    atomicAdd(&node_qleave[n1], -q);
}

__global__ __launch_bounds__(256, 4) void swe2d_drainage_pipe_end_bc_kernel(
    int32_t n_pipe_ends,
    int32_t n_cells,
    const int32_t* __restrict__ pipe_end_cell,
    const int32_t* __restrict__ pipe_end_node,
    const double* __restrict__ pipe_end_invert_elev,
    const double* __restrict__ pipe_end_diameter,
    const double* __restrict__ pipe_end_area,
    const double* __restrict__ pipe_end_inlet_loss_k,
    const double* __restrict__ pipe_end_outlet_loss_k,
    const double* __restrict__ cell_wse,
    const double* __restrict__ node_invert_elev,
    const double* __restrict__ node_surface_area,
    const double* __restrict__ node_qleave,
    double gravity,
    double* __restrict__ node_depth,
    double* __restrict__ pipe_end_depth_bc,
    double* __restrict__ pipe_end_node_area)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_pipe_ends) return;
    const int32_t c = pipe_end_cell[i];
    const int32_t n = pipe_end_node[i];
    if (c < 0 || c >= n_cells || n < 0) {
        pipe_end_depth_bc[i] = 0.0;
        pipe_end_node_area[i] = 1.0;
        return;
    }

    const double invert = pipe_end_invert_elev[i];
    const double area_node = fmax(1.0, node_surface_area[n]);
    const double wse_surface = cell_wse[c];
    const double node_head = node_invert_elev[n] + fmax(0.0, node_depth[n]);

    double area_pipe = fmax(0.0, pipe_end_area[i]);
    if (area_pipe <= 0.0) {
        const double d_pipe = fmax(0.0, pipe_end_diameter[i]);
        area_pipe = (d_pipe > 0.0) ? (0.25 * M_PI * d_pipe * d_pipe) : 0.0;
    }

    const double q_leave = node_qleave ? node_qleave[n] : 0.0;
    bool flow_surface_to_network = false;
    if (fabs(q_leave) <= 1.0e-12) {
        flow_surface_to_network = (wse_surface >= node_head);
    } else {
        flow_surface_to_network = (q_leave >= 0.0);
    }

    const double k_in = fmax(0.0, pipe_end_inlet_loss_k[i]);
    const double k_out = fmax(0.0, pipe_end_outlet_loss_k[i]);
    const double k_use = flow_surface_to_network ? k_in : k_out;

    double h_loss = 0.0;
    if (area_pipe > 0.0) {
        const double vel = fabs(q_leave) / fmax(area_pipe, 1.0e-12);
        h_loss = k_use * vel * vel / (2.0 * fmax(gravity, 1.0e-9));
    }
    const double wse_eff = fmax(invert, wse_surface - h_loss);
    const double d_bc = fmax(0.0, wse_eff - invert);
    node_depth[n] = d_bc;
    pipe_end_depth_bc[i] = d_bc;
    pipe_end_node_area[i] = area_node;
}

__global__ __launch_bounds__(256, 4) void swe2d_drainage_pipe_end_exchange_kernel(
    int32_t n_pipe_ends,
    int32_t n_cells,
    const int32_t* __restrict__ pipe_end_cell,
    const int32_t* __restrict__ pipe_end_node,
    const double* __restrict__ pipe_end_depth_bc,
    const double* __restrict__ pipe_end_node_area,
    const double* __restrict__ cell_area,
    const double* __restrict__ cell_depth,
    const double* __restrict__ node_max_depth,
    double dt_s,
    const double* __restrict__ node_depth,
    double* __restrict__ q_cell,
    double* __restrict__ node_depth_write,
    double* __restrict__ limiter_event_count,
    double* __restrict__ limiter_volume_m3)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_pipe_ends) return;
    const int32_t c = pipe_end_cell[i];
    const int32_t n = pipe_end_node[i];
    if (c < 0 || c >= n_cells || n < 0) return;

    const double d_bc = fmax(0.0, pipe_end_depth_bc[i]);
    const double area_node = fmax(1.0, pipe_end_node_area[i]);
    const double d_after = fmax(0.0, node_depth[n]);
    const double delta_vol = (d_after - d_bc) * area_node;
    double q_net = (dt_s > 0.0) ? (delta_vol / dt_s) : 0.0;

    if (q_net > 0.0) {
        atomicAdd(&q_cell[c], q_net);
        return;
    }
    if (q_net >= 0.0) return;

    // Surface -> network sink, apply availability limiter.
    double q_in = -q_net;
    if (cell_depth && cell_area) {
        const double avail_surface_vol = fmax(0.0, cell_depth[c]) * fmax(0.0, cell_area[c]);
        const double q_cap_surface = (dt_s > 0.0) ? (avail_surface_vol / dt_s) : 0.0;
        if (q_in > q_cap_surface) {
            if (limiter_event_count) atomicAdd(limiter_event_count, 1.0);
            if (limiter_volume_m3) atomicAdd(limiter_volume_m3, fmax(0.0, q_in - q_cap_surface) * dt_s);
            q_in = q_cap_surface;
        }
    }
    atomicAdd(&q_cell[c], -q_in);

    const double d_reconciled = d_bc + q_in * dt_s / area_node;
    node_depth_write[n] = fmax(0.0, fmin(node_max_depth[n], d_reconciled));
}

__global__ __launch_bounds__(256, 4) void swe2d_drainage_inlet_exchange_kernel(
    int32_t n_inlets,
    int32_t n_cells,
    const int32_t* __restrict__ inlet_cell,
    const int32_t* __restrict__ inlet_node,
    const double* __restrict__ inlet_crest_elev,
    const double* __restrict__ inlet_width,
    const double* __restrict__ inlet_coefficient,
    const double* __restrict__ inlet_max_capture,
    const double* __restrict__ cell_wse,
    const double* __restrict__ cell_area,
    const double* __restrict__ cell_depth,
    const double* __restrict__ node_invert_elev,
    const double* __restrict__ node_max_depth,
    const double* __restrict__ node_depth,
    const double* __restrict__ node_surface_area,
    double dt_s,
    double gravity,
    double head_deadband_m,
    double* __restrict__ q_cell,
    double* __restrict__ node_depth_delta,
    double* __restrict__ limiter_event_count,
    double* __restrict__ limiter_volume_m3)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_inlets) return;
    const int32_t c = inlet_cell[i];
    const int32_t n = inlet_node[i];
    if (c < 0 || c >= n_cells || n < 0) return;

    const double wse_surface = cell_wse[c];
    const double wse_node = node_invert_elev[n] + fmax(0.0, node_depth[n]);
    const double crest = inlet_crest_elev[i];
    const double width = fmax(0.0, inlet_width[i]);
    const double cd = fmax(0.0, inlet_coefficient[i]);
    const double q_cap = inlet_max_capture[i];
    const double deadband = fmax(0.0, head_deadband_m);

    const double capture_head = fmax(0.0, wse_surface - fmax(wse_node, crest) - deadband);
    const double area_capture = width * fmax(0.01, capture_head);
    double q_capture = cd * area_capture * sqrt(fmax(0.0, 2.0 * gravity * capture_head));
    if (isfinite(q_cap) && q_cap > 0.0) q_capture = fmin(q_capture, q_cap);

    const double relief_head = fmax(0.0, wse_node - fmax(wse_surface, crest) - deadband);
    const double area_relief = width * fmax(0.01, relief_head);
    double q_relief = cd * area_relief * sqrt(fmax(0.0, 2.0 * gravity * relief_head));
    if (isfinite(q_cap) && q_cap > 0.0) q_relief = fmin(q_relief, q_cap);

    const double node_area = fmax(1.0, node_surface_area[n]);
    const double d_node = fmax(0.0, node_depth[n]);
    const double rem_node_storage = fmax(0.0, node_max_depth[n] - d_node) * node_area;
    const double avail_node_storage = fmax(0.0, d_node) * node_area;
    const double q_cap_node_in = (dt_s > 0.0) ? rem_node_storage / dt_s : 0.0;
    const double q_cap_node_out = (dt_s > 0.0) ? avail_node_storage / dt_s : 0.0;

    if (q_capture > q_cap_node_in) {
        if (limiter_event_count) atomicAdd(limiter_event_count, 1.0);
        if (limiter_volume_m3) atomicAdd(limiter_volume_m3, fmax(0.0, q_capture - q_cap_node_in) * dt_s);
        q_capture = q_cap_node_in;
    }

    if (cell_depth && cell_area) {
        const double avail_surface_vol = fmax(0.0, cell_depth[c]) * fmax(0.0, cell_area[c]);
        const double q_cap_surface = (dt_s > 0.0) ? avail_surface_vol / dt_s : 0.0;
        if (q_capture > q_cap_surface) {
            if (limiter_event_count) atomicAdd(limiter_event_count, 1.0);
            if (limiter_volume_m3) atomicAdd(limiter_volume_m3, fmax(0.0, q_capture - q_cap_surface) * dt_s);
            q_capture = q_cap_surface;
        }
    }

    if (q_relief > q_cap_node_out) {
        if (limiter_event_count) atomicAdd(limiter_event_count, 1.0);
        if (limiter_volume_m3) atomicAdd(limiter_volume_m3, fmax(0.0, q_relief - q_cap_node_out) * dt_s);
        q_relief = q_cap_node_out;
    }

    atomicAdd(&q_cell[c], q_relief - q_capture);
    atomicAdd(&node_depth_delta[n], dt_s * (q_capture - q_relief) / node_area);
}

__global__ __launch_bounds__(256, 4) void swe2d_drainage_outfall_exchange_kernel(
    int32_t n_outfalls,
    int32_t n_cells,
    const int32_t* __restrict__ outfall_cell,
    const int32_t* __restrict__ outfall_node,
    const double* __restrict__ outfall_invert_elev,
    const double* __restrict__ outfall_diameter,
    const double* __restrict__ outfall_coefficient,
    const double* __restrict__ outfall_max_flow,
    const int32_t* __restrict__ outfall_zero_storage,
    const double* __restrict__ cell_wse,
    const double* __restrict__ cell_area,
    const double* __restrict__ cell_depth,
    const double* __restrict__ node_max_depth,
    const double* __restrict__ node_depth,
    const double* __restrict__ node_surface_area,
    double dt_s,
    double gravity,
    double head_deadband_m,
    double* __restrict__ q_cell,
    double* __restrict__ node_depth_delta,
    double* __restrict__ node_depth_work,
    double* __restrict__ limiter_event_count,
    double* __restrict__ limiter_volume_m3)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_outfalls) return;
    const int32_t c = outfall_cell[i];
    const int32_t n = outfall_node[i];
    if (c < 0 || c >= n_cells || n < 0) return;

    const double d_pipe = fmax(0.0, outfall_diameter[i]);
    const double area_pipe = (d_pipe > 0.0) ? (0.25 * M_PI * d_pipe * d_pipe) : 0.0;
    if (area_pipe <= 0.0) return;

    const double invert = outfall_invert_elev[i];
    const double coeff = fmax(0.0, outfall_coefficient[i]);
    const double q_cap = outfall_max_flow[i];
    const bool zero_storage = (outfall_zero_storage[i] != 0);
    const double deadband = fmax(0.0, head_deadband_m);

    const double wse_surface = cell_wse[c];
    double d_node = 0.0;
    double node_area = 1.0;
    double wse_node = invert;
    if (!zero_storage) {
        d_node = fmax(0.0, node_depth[n]);
        node_area = fmax(1.0, node_surface_area[n]);
        wse_node = invert + d_node;
    } else {
        // Daylight outfall: no local storage bucket during exchange.
        node_depth_work[n] = 0.0;
    }

    if (wse_node > wse_surface + deadband && wse_node > invert) {
        // Surcharge: network discharges to surface.
        const double head = fmax(0.0, wse_node - wse_surface);
        double q_out = coeff * area_pipe * sqrt(fmax(0.0, 2.0 * gravity * head));
        if (isfinite(q_cap) && q_cap > 0.0) q_out = fmin(q_out, q_cap);
        q_out = fmax(0.0, q_out);

        if (!zero_storage) {
            const double avail_node_vol = d_node * node_area;
            const double q_cap_node = (dt_s > 0.0) ? (avail_node_vol / dt_s) : 0.0;
            if (q_out > q_cap_node) {
                if (limiter_event_count) atomicAdd(limiter_event_count, 1.0);
                if (limiter_volume_m3) atomicAdd(limiter_volume_m3, fmax(0.0, q_out - q_cap_node) * dt_s);
                q_out = q_cap_node;
            }
        }

        atomicAdd(&q_cell[c], q_out);
        if (!zero_storage) {
            atomicAdd(&node_depth_delta[n], -dt_s * q_out / node_area);
        }
    } else if (wse_surface > wse_node + deadband && wse_surface > invert) {
        // Backwater: surface drains into outfall node.
        const double head = fmax(0.0, wse_surface - wse_node);
        double q_in = coeff * area_pipe * sqrt(fmax(0.0, 2.0 * gravity * head));
        if (isfinite(q_cap) && q_cap > 0.0) q_in = fmin(q_in, q_cap);
        q_in = fmax(0.0, q_in);

        if (!zero_storage) {
            const double rem_node_vol = fmax(0.0, node_max_depth[n] - d_node) * node_area;
            const double q_cap_node = (dt_s > 0.0) ? (rem_node_vol / dt_s) : 0.0;
            if (q_in > q_cap_node) {
                if (limiter_event_count) atomicAdd(limiter_event_count, 1.0);
                if (limiter_volume_m3) atomicAdd(limiter_volume_m3, fmax(0.0, q_in - q_cap_node) * dt_s);
                q_in = q_cap_node;
            }
        }

        if (cell_depth && cell_area) {
            const double avail_surface_vol = fmax(0.0, cell_depth[c]) * fmax(0.0, cell_area[c]);
            const double q_cap_surface = (dt_s > 0.0) ? (avail_surface_vol / dt_s) : 0.0;
            if (q_in > q_cap_surface) {
                if (limiter_event_count) atomicAdd(limiter_event_count, 1.0);
                if (limiter_volume_m3) atomicAdd(limiter_volume_m3, fmax(0.0, q_in - q_cap_surface) * dt_s);
                q_in = q_cap_surface;
            }
        }

        atomicAdd(&q_cell[c], -q_in);
        if (!zero_storage) {
            atomicAdd(&node_depth_delta[n], dt_s * q_in / node_area);
        }
    }
}

__global__ void swe2d_drainage_apply_delta_kernel(
    int32_t n_nodes,
    const double* __restrict__ node_max_depth,
    const double* __restrict__ node_depth_delta,
    double* __restrict__ node_depth)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_nodes) return;
    double d = node_depth[i] + node_depth_delta[i];
    d = fmax(0.0, fmin(node_max_depth[i], d));
    node_depth[i] = d;
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
    const double*    n_mann_cell,
    int              degen_mode,
    double           max_inv_area)
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
    alloc_d(reinterpret_cast<void**>(&dev->d_edge_mx),     sz_edges * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_edge_my),     sz_edges * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_edge_bc),     sz_edges * sizeof(int32_t));
    alloc_d(reinterpret_cast<void**>(&dev->d_edge_bc_val), sz_edges * sizeof(double));

    copy_h2d_i(dev->d_edge_c0, mesh.edge_c0.data(), sz_edges);
    copy_h2d_i(dev->d_edge_c1, mesh.edge_c1.data(), sz_edges);
    copy_h2d_i(dev->d_edge_n0, mesh.edge_n0.data(), sz_edges);
    copy_h2d_i(dev->d_edge_n1, mesh.edge_n1.data(), sz_edges);
    copy_h2d_d(dev->d_edge_nx,  mesh.edge_nx.data(),  sz_edges);
    copy_h2d_d(dev->d_edge_ny,  mesh.edge_ny.data(),  sz_edges);
    copy_h2d_d(dev->d_edge_len, mesh.edge_len.data(), sz_edges);
    {
        std::vector<double> edge_mx(sz_edges), edge_my(sz_edges);
        for (size_t i = 0; i < sz_edges; ++i) {
            const int32_t n0 = mesh.edge_n0[i];
            const int32_t n1 = mesh.edge_n1[i];
            edge_mx[i] = 0.5 * (mesh.node_x[n0] + mesh.node_x[n1]);
            edge_my[i] = 0.5 * (mesh.node_y[n0] + mesh.node_y[n1]);
        }
        copy_h2d_d(dev->d_edge_mx, edge_mx.data(), sz_edges);
        copy_h2d_d(dev->d_edge_my, edge_my.data(), sz_edges);
    }
    // BCType → int32_t
    {
        std::vector<int32_t> bc_int(sz_edges);
        for (size_t i = 0; i < sz_edges; ++i)
            bc_int[i] = static_cast<int32_t>(mesh.edge_bc[i]);
        copy_h2d_i(dev->d_edge_bc, bc_int.data(), sz_edges);
    }
    copy_h2d_d(dev->d_edge_bc_val, mesh.edge_bc_val.data(), sz_edges);

    // Cell-to-edge CSR for atomics-free accumulation.
    alloc_d(reinterpret_cast<void**>(&dev->d_cell_edge_offsets), (sz_cells + 1) * sizeof(int32_t));
    alloc_d(reinterpret_cast<void**>(&dev->d_cell_edge_ids),     mesh.cell_edge_ids.size() * sizeof(int32_t));
    copy_h2d_i(dev->d_cell_edge_offsets, mesh.cell_edge_offsets.data(), sz_cells + 1);
    copy_h2d_i(dev->d_cell_edge_ids, mesh.cell_edge_ids.data(), mesh.cell_edge_ids.size());

    // 2-ring cell stencil CSR for the least-squares gradient (scheme 6).
    {
        const size_t n_ring2 = mesh.cell_ring2_ids.size();
        dev->n_cell_ring2 = static_cast<int32_t>(n_ring2);
        alloc_d(reinterpret_cast<void**>(&dev->d_cell_ring2_offsets), (sz_cells + 1) * sizeof(int32_t));
        copy_h2d_i(dev->d_cell_ring2_offsets, mesh.cell_ring2_offsets.data(), sz_cells + 1);
        if (n_ring2 > 0) {
            alloc_d(reinterpret_cast<void**>(&dev->d_cell_ring2_ids),       n_ring2 * sizeof(int32_t));
            alloc_d(reinterpret_cast<void**>(&dev->d_cell_ring2_dcx),       n_ring2 * sizeof(double));
            alloc_d(reinterpret_cast<void**>(&dev->d_cell_ring2_dcy),       n_ring2 * sizeof(double));
            alloc_d(reinterpret_cast<void**>(&dev->d_cell_ring2_inv_dist2), n_ring2 * sizeof(double));
            copy_h2d_i(dev->d_cell_ring2_ids,       mesh.cell_ring2_ids.data(),       n_ring2);
            copy_h2d_d(dev->d_cell_ring2_dcx,       mesh.cell_ring2_dcx.data(),       n_ring2);
            copy_h2d_d(dev->d_cell_ring2_dcy,       mesh.cell_ring2_dcy.data(),       n_ring2);
            copy_h2d_d(dev->d_cell_ring2_inv_dist2, mesh.cell_ring2_inv_dist2.data(), n_ring2);
        }
    }

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


    alloc_d(reinterpret_cast<void**>(&dev->d_cell_source_mps), sz_cells * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_stage_cell_source_mps), static_cast<size_t>(SWE2D_GRAPH_STAGE_SLOTS) * sz_cells * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_stage_edge_bc), static_cast<size_t>(SWE2D_GRAPH_STAGE_SLOTS) * sz_edges * sizeof(int32_t));
    alloc_d(reinterpret_cast<void**>(&dev->d_stage_edge_bc_val), static_cast<size_t>(SWE2D_GRAPH_STAGE_SLOTS) * sz_edges * sizeof(double));
    CUDA_CHECK(cudaMemset(dev->d_cell_source_mps, 0, sz_cells * sizeof(double)));
    CUDA_CHECK(cudaMemset(dev->d_stage_cell_source_mps, 0, static_cast<size_t>(SWE2D_GRAPH_STAGE_SLOTS) * sz_cells * sizeof(double)));

    // External coupling source buffer — allocated once, reused every step.
    alloc_d(reinterpret_cast<void**>(&dev->d_external_source_mps), sz_cells * sizeof(double));
    CUDA_CHECK(cudaMemset(dev->d_external_source_mps, 0, sz_cells * sizeof(double)));

    // Edge flux buffers (consumed by the cell-centric update kernel).
    alloc_d(reinterpret_cast<void**>(&dev->d_flux_h),    sz_edges * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_flux_hu),   sz_edges * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_flux_hv),   sz_edges * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_flux_hu_r), sz_edges * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_flux_hv_r), sz_edges * sizeof(double));

    // CFL workspace
    alloc_d(reinterpret_cast<void**>(&dev->d_lambda_max), sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_max_wse_elev_error), sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_diag_packed), 3 * sizeof(double));
    CUDA_CHECK(cudaMemset(dev->d_diag_packed, 0, 3 * sizeof(double)));
    // Two-level CFL reduction: block-max array sized for worst-case grid.
    // Reallocated if mesh grows; zero-sized for now, filled on first step.
    dev->d_cfl_block_max = nullptr;
    dev->cfl_block_capacity = 0;

    // Wet/dry active-set arrays
    alloc_d(reinterpret_cast<void**>(&dev->d_active),    sz_cells * sizeof(int32_t));
    alloc_d(reinterpret_cast<void**>(&dev->d_n_wet),     sizeof(int32_t));
    alloc_d(reinterpret_cast<void**>(&dev->d_bc_forced), sz_cells * sizeof(int32_t));
    alloc_d(reinterpret_cast<void**>(&dev->d_was_active), sz_cells * sizeof(int32_t));
    alloc_d(reinterpret_cast<void**>(&dev->d_active_edge_ids), sz_edges * sizeof(int32_t));
    alloc_d(reinterpret_cast<void**>(&dev->d_n_active_edges), sizeof(int32_t));
    CUDA_CHECK(cudaMemset(dev->d_active,    0, sz_cells * sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(dev->d_n_wet,     0, sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(dev->d_was_active, 0, sz_cells * sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(dev->d_n_active_edges, 0, sizeof(int32_t)));
    // Build bc_forced host-side: mark cells at forced-inflow BC edges (types 2, 3, 6)
    // so that even initially-dry inflow cells are included in the active set.
    {
        std::vector<int32_t> h_bcf(sz_cells, 0);
        for (size_t ei = 0; ei < sz_edges; ++ei) {
            if (mesh.edge_c1[ei] >= 0) continue;   // interior edge
            const int32_t bc = static_cast<int32_t>(mesh.edge_bc[ei]);
            if (bc == 2 || bc == 3 || bc == 6) {
                const int32_t c0 = mesh.edge_c0[ei];
                if (c0 >= 0 && c0 < static_cast<int32_t>(sz_cells))
                    h_bcf[static_cast<size_t>(c0)] = 1;
            }
        }
        copy_h2d_i(dev->d_bc_forced, h_bcf.data(), sz_cells);
    }

    // ── Degenerate-cell precompute (host-side, uploaded once) ────────────────
    dev->degen_mode = degen_mode;
    if (degen_mode > 0) {
        // Build degenerate mask: cell is degenerate if cell_inv_area > max_inv_area.
        std::vector<int32_t> h_degen(sz_cells, 0);
        for (size_t ci = 0; ci < sz_cells; ++ci) {
            if (mesh.cell_inv_area[ci] > max_inv_area)
                h_degen[ci] = 1;
        }
        alloc_d(reinterpret_cast<void**>(&dev->d_degen_mask), sz_cells * sizeof(int32_t));
        copy_h2d_i(dev->d_degen_mask, h_degen.data(), sz_cells);

        if (degen_mode == 1) {
            // Mode 1 (skip): zero initial state of degenerate cells so they start dry.
            std::vector<double> h_mod(h0, h0 + sz_cells);
            for (size_t ci = 0; ci < sz_cells; ++ci) {
                if (h_degen[ci]) h_mod[ci] = 0.0;
            }
            copy_h2d_d(dev->d_h, h_mod.data(), sz_cells);
        }

        if (degen_mode == 2) {
            // Mode 2 (repair): replace degenerate cell inv_area with neighbor-averaged value.
            std::vector<double> h_repair(sz_cells);
            for (size_t ci = 0; ci < sz_cells; ++ci)
                h_repair[ci] = mesh.cell_inv_area[ci];

            // Accumulate non-degenerate neighbor inv_area for each degenerate cell.
            std::vector<double> sum_ia(sz_cells, 0.0);
            std::vector<int32_t> cnt_ia(sz_cells, 0);
            for (size_t ei = 0; ei < sz_edges; ++ei) {
                int32_t c0e = mesh.edge_c0[ei];
                int32_t c1e = mesh.edge_c1[ei];
                if (c1e < 0) continue;
                if (h_degen[static_cast<size_t>(c0e)] && !h_degen[static_cast<size_t>(c1e)]) {
                    sum_ia[static_cast<size_t>(c0e)] += mesh.cell_inv_area[static_cast<size_t>(c1e)];
                    cnt_ia[static_cast<size_t>(c0e)]++;
                }
                if (h_degen[static_cast<size_t>(c1e)] && !h_degen[static_cast<size_t>(c0e)]) {
                    sum_ia[static_cast<size_t>(c1e)] += mesh.cell_inv_area[static_cast<size_t>(c0e)];
                    cnt_ia[static_cast<size_t>(c1e)]++;
                }
            }
            for (size_t ci = 0; ci < sz_cells; ++ci) {
                if (h_degen[ci] && cnt_ia[ci] > 0)
                    h_repair[ci] = sum_ia[ci] / static_cast<double>(cnt_ia[ci]);
                else if (h_degen[ci])
                    h_repair[ci] = max_inv_area;  // fallback: use cap
            }
            alloc_d(reinterpret_cast<void**>(&dev->d_inv_area_repaired), sz_cells * sizeof(double));
            copy_h2d_d(dev->d_inv_area_repaired, h_repair.data(), sz_cells);
        }

        if (degen_mode == 3) {
            // Mode 3 (merge): for each degenerate cell, find the non-degenerate neighbor
            // with the largest area as the merge owner.
            std::vector<int32_t> h_owner(sz_cells, -1);
            std::vector<double> best_area(sz_cells, -1.0);
            for (size_t ei = 0; ei < sz_edges; ++ei) {
                int32_t c0e = mesh.edge_c0[ei];
                int32_t c1e = mesh.edge_c1[ei];
                if (c1e < 0) continue;
                if (h_degen[static_cast<size_t>(c0e)] && !h_degen[static_cast<size_t>(c1e)]) {
                    double a = mesh.cell_area[static_cast<size_t>(c1e)];
                    if (a > best_area[static_cast<size_t>(c0e)]) {
                        best_area[static_cast<size_t>(c0e)] = a;
                        h_owner[static_cast<size_t>(c0e)] = c1e;
                    }
                }
                if (h_degen[static_cast<size_t>(c1e)] && !h_degen[static_cast<size_t>(c0e)]) {
                    double a = mesh.cell_area[static_cast<size_t>(c0e)];
                    if (a > best_area[static_cast<size_t>(c1e)]) {
                        best_area[static_cast<size_t>(c1e)] = a;
                        h_owner[static_cast<size_t>(c1e)] = c0e;
                    }
                }
            }
            alloc_d(reinterpret_cast<void**>(&dev->d_merge_owner), sz_cells * sizeof(int32_t));
            copy_h2d_i(dev->d_merge_owner, h_owner.data(), sz_cells);
        }
    }

    // Create a persistent non-default stream.  All per-step kernel launches and
    // async memsets go on this stream so that host-side work (BC updates, Python
    // callbacks) can proceed while the GPU finishes the previous step.
    CUDA_CHECK(cudaStreamCreate(&dev->d_stream));

    // Note: deferred for now to avoid allocation overhead for hydrostatic-only runs.

    s_coupling_dev = dev;
    return dev;
}

// ─────────────────────────────────────────────────────────────────────────────
// swe2d_gpu_step
// ─────────────────────────────────────────────────────────────────────────────
void swe2d_gpu_step(
    SWE2DDeviceState* dev,
    double t_now,
    double dt,
    double g,
    double h_min,
    int spatial_scheme,
    double /*cfl_factor*/,
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
    bool sync_diagnostics,
    SWE2DStepDiag* diag,
    double front_flux_damping,
    bool   active_set_hysteresis)
{
    constexpr int BLOCK = 256;
    int32_t n_edges = dev->n_edges;
    int32_t n_cells = dev->n_cells;
    const int32_t graph_integrator =
        (dev->kernel_graph_cache.time_integrator == 2 || dev->kernel_graph_cache.time_integrator == 4)
            ? dev->kernel_graph_cache.time_integrator
            : 1;

    // Clear any stale CUDA error left by prior host-side work (coupling,
    // face-flux uploads, diagnostics, etc.) before any solver GPU work begins.
    (void)cudaGetLastError();

    if (swe2d_debug_enabled("BACKWATER_SWE2D_DEBUG_GPU_INPUT")) {
        std::fprintf(stderr, "[SWE2D_DEBUG] GPU input: n_cells=%d n_edges=%d dt=%.9e g=%.9e h_min=%.9e\n",
                     static_cast<int>(n_cells), static_cast<int>(n_edges), dt, g, h_min);
        std::fprintf(stderr,
                     "[SWE2D_DEBUG] face-flux: use_culvert_face_flux=%d n_ff_faces=%d ff_preloaded=%d "
                     "d_ext_h=%p d_ext_hu=%p d_ext_hv=%p n_structures=%d sf_params_preloaded=%d "
                     "graph_replay_count=%lu\n",
                     static_cast<int>(dev->use_culvert_face_flux),
                     dev->culvert_ff_ws.params_preloaded ? static_cast<int>(dev->culvert_ff_ws.n_culvert_faces) : -1,
                     static_cast<int>(dev->culvert_ff_ws.params_preloaded),
                     static_cast<void*>(dev->d_ext_struct_flux_h),
                     static_cast<void*>(dev->d_ext_struct_flux_hu),
                     static_cast<void*>(dev->d_ext_struct_flux_hv),
                     static_cast<int>(dev->sf_ws.n_structures),
                     static_cast<int>(dev->sf_ws.params_preloaded),
                     static_cast<unsigned long>(dev->graph_replay_count));
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

    // Zero flux accumulators (async: pipelined with preceding GPU work on stream)
    const size_t sz_edges_d = static_cast<size_t>(n_edges) * sizeof(double);
    CUDA_CHECK(cudaMemsetAsync(dev->d_flux_h,  0, sz_edges_d, dev->d_stream));
    CUDA_CHECK(cudaMemsetAsync(dev->d_flux_hu, 0, sz_edges_d, dev->d_stream));
    CUDA_CHECK(cudaMemsetAsync(dev->d_flux_hv, 0, sz_edges_d, dev->d_stream));
    CUDA_CHECK(cudaMemsetAsync(dev->d_flux_hu_r, 0, sz_edges_d, dev->d_stream));
    CUDA_CHECK(cudaMemsetAsync(dev->d_flux_hv_r, 0, sz_edges_d, dev->d_stream));

    // Optional rain + CN forcing: build per-cell source term for this step.
    // This must run BEFORE classify so source-only dry cells are marked active
    // and can receive wet-up updates in this same step.
    if (dev->n_rain_samples > 0 && dev->d_cell_gage_idx && dev->d_rain_hg_offsets && dev->d_rain_hg_time_s && dev->d_rain_hg_cum_mm && dev->d_rain_cn) {
        const int r_grid = (n_cells + BLOCK - 1) / BLOCK;
        swe2d_build_rain_cn_source_kernel<<<r_grid, BLOCK, 0, dev->d_stream>>>(
            n_cells,
            dev->d_cell_gage_idx,
            dev->d_rain_hg_offsets,
            dev->d_rain_hg_time_s,
            dev->d_rain_hg_cum_mm,
            dev->d_rain_cn,
            dev->d_rain_cum_mm,
            dev->d_rain_excess_cum_mm,
            dev->d_cell_source_mps,
            t_now,
            t_now + dt,
            dev->rain_ia_ratio,
            dev->rain_mm_to_model_depth);
        CUDA_CHECK(cudaGetLastError());
    }

    // Hysteretic active set: save d_active BEFORE classify overwrites it.
    if (active_set_hysteresis && dev->d_was_active && dev->d_active && dev->d_n_wet) {
        CUDA_CHECK(cudaMemcpyAsync(dev->d_was_active, dev->d_active,
                                   static_cast<size_t>(n_cells) * sizeof(int32_t),
                                   cudaMemcpyDeviceToDevice,
                                   dev->d_stream));
    }

    const bool need_gradients = (spatial_scheme >= 1);
    const bool dbg_edge_flux = swe2d_debug_enabled("BACKWATER_SWE2D_DEBUG_GPU_EDGE_FLUX");
    const bool dbg_flux_summary = swe2d_debug_enabled("BACKWATER_SWE2D_DEBUG_GPU_FLUX");
    const bool try_kernel_graph = dev->enable_kernel_graphs && !dbg_edge_flux && !dbg_flux_summary;
    const bool has_hg = (dev->n_hg_edges > 0);
    // Variant key: bit 0 = has hydrograph BCs, bit 1 = needs gradient pre-pass.
    const int32_t graph_variant_key = (has_hg ? 1 : 0) | (need_gradients ? 2 : 0);
    const uint64_t graph_signature = swe2d_kernel_graph_signature(
        dt,
        g,
        h_min,
        cfl_lambda_cap,
        max_inv_area,
        momentum_cap_min_speed,
        momentum_cap_celerity_mult,
        depth_cap,
        max_rel_depth_increase,
        shallow_damping_depth,
        extreme_rain_mode,
        source_cfl_beta,
        source_max_substeps,
        source_rate_cap,
        source_depth_step_cap,
        source_true_subcycling,
        source_imex_split,
        enable_shallow_front_recon_fallback,
        front_flux_damping, dev->use_culvert_face_flux);

    bool used_graph_replay = false;
    if (try_kernel_graph) {
        auto& cache = dev->kernel_graph_cache;
        const bool cache_match =
            cache.is_valid &&
            cache.exec != nullptr &&
            cache.n_cells == n_cells &&
            cache.n_edges == n_edges &&
            cache.spatial_scheme == spatial_scheme &&
            cache.time_integrator == graph_integrator &&
            cache.variant_key == graph_variant_key &&
            cache.config_signature == graph_signature;

        if (cache_match) {
            CUDA_CHECK(cudaGraphLaunch(cache.exec, dev->d_stream));
            dev->graph_replay_count += 1;
            used_graph_replay = true;
        } else {
            cache.destroy();
            CUDA_CHECK(cudaStreamSynchronize(dev->d_stream));
            // Pre-allocate CFL workspace outside capture — cudaMalloc/cudaFree
            // are not permitted inside a graph capture region.
            int grid_cfl_pre = (n_edges + BLOCK - 1) / BLOCK;
            if (grid_cfl_pre > 0) {
                swe2d_ensure_cfl_block_workspace(dev, grid_cfl_pre);
            }
            cudaError_t cap_begin = cudaStreamBeginCapture(dev->d_stream, cudaStreamCaptureModeThreadLocal);
            if (cap_begin == cudaSuccess) {
                // ── Expanded graph: classify_and_mark → degen → hg_bc → gradient → flux → update → cfl → reduce → pack ──

                // classify_and_mark (fused, always runs)
                {
                    const int c_grid = (n_cells + BLOCK - 1) / BLOCK;
                    // was_active copy already done before graph section
                    CUDA_CHECK(cudaMemsetAsync(dev->d_n_wet, 0, sizeof(int32_t), dev->d_stream));
                    swe2d_classify_and_mark_kernel<<<c_grid, BLOCK, BLOCK * sizeof(int32_t), dev->d_stream>>>(
                        n_cells, n_edges,
                        dev->d_h, dev->d_cell_source_mps, dev->d_external_source_mps,
                        dev->d_ext_struct_flux_h,
                        dev->d_bc_forced,
                        dev->d_edge_c0, dev->d_edge_c1,
                        dev->d_active, dev->d_n_wet, h_min,
                        active_set_hysteresis ? dev->d_was_active : nullptr);
                }
                // degen_deactivate + optional degen_sync
                if ((dev->degen_mode == 1 || dev->degen_mode == 3) && dev->d_degen_mask) {
                    const int c_grid2 = (n_cells + BLOCK - 1) / BLOCK;
                    swe2d_degen_deactivate_kernel<<<c_grid2, BLOCK, 0, dev->d_stream>>>(
                        n_cells, dev->d_degen_mask, dev->d_active);
                    if (dev->degen_mode == 3 && dev->d_merge_owner) {
                        swe2d_degen_sync_kernel<<<c_grid2, BLOCK, 0, dev->d_stream>>>(
                            n_cells, dev->d_degen_mask, dev->d_merge_owner,
                            dev->d_h, dev->d_hu, dev->d_hv);
                    }
                }
                // hydrograph BCs
                if (has_hg) {
                    const int hg_grid = (dev->n_hg_edges + BLOCK - 1) / BLOCK;
                    swe2d_apply_hydrograph_bc_kernel<<<hg_grid, BLOCK, 0, dev->d_stream>>>(
                        dev->n_hg_edges,
                        dev->d_hg_edge_index, dev->d_hg_bc_type,
                        dev->d_hg_offsets, dev->d_hg_time_s, dev->d_hg_value,
                        dev->d_edge_bc, dev->d_edge_bc_val, t_now);
                }
                // gradient pre-pass
                if (need_gradients) {
                    const size_t sz_c = static_cast<size_t>(n_cells) * sizeof(double);
                    CUDA_CHECK(cudaMemsetAsync(dev->d_grad_hx,  0, sz_c, dev->d_stream));
                    CUDA_CHECK(cudaMemsetAsync(dev->d_grad_hy,  0, sz_c, dev->d_stream));
                    CUDA_CHECK(cudaMemsetAsync(dev->d_grad_hux, 0, sz_c, dev->d_stream));
                    CUDA_CHECK(cudaMemsetAsync(dev->d_grad_huy, 0, sz_c, dev->d_stream));
                    CUDA_CHECK(cudaMemsetAsync(dev->d_grad_hvx, 0, sz_c, dev->d_stream));
                    CUDA_CHECK(cudaMemsetAsync(dev->d_grad_hvy, 0, sz_c, dev->d_stream));
                    int g_grid = (n_edges + BLOCK - 1) / BLOCK;
                    swe2d_gradient_kernel<<<g_grid, BLOCK, 0, dev->d_stream>>>(
                        n_edges,
                        dev->d_edge_c0, dev->d_edge_c1,
                        dev->d_edge_nx, dev->d_edge_ny, dev->d_edge_len,
                        dev->d_h, dev->d_cell_zb, dev->d_hu, dev->d_hv,
                        dev->d_cell_inv_area,
                        max_inv_area,
                        dev->d_grad_hx,  dev->d_grad_hy,
                        dev->d_grad_hux, dev->d_grad_huy,
                        dev->d_grad_hvx, dev->d_grad_hvy,
                        dev->d_active,
                        dev->d_degen_mask,
                        dev->degen_mode);
                    swe2d_maybe_launch_lsq_gradient(dev, spatial_scheme, n_cells, BLOCK,
                                                    dev->d_h, dev->d_hu, dev->d_hv);
                }
                // flux
                int grid_flux = (n_edges + BLOCK - 1) / BLOCK;
                swe2d_flux_kernel<<<grid_flux, BLOCK, 0, dev->d_stream>>>(
                    n_edges,
                    dev->d_edge_c0, dev->d_edge_c1,
                    dev->d_edge_nx, dev->d_edge_ny, dev->d_edge_len,
                    dev->d_edge_mx, dev->d_edge_my,
                    dev->d_edge_bc, dev->d_edge_bc_val,
                    dev->d_h, dev->d_hu, dev->d_hv,
                    dev->d_n_mann_cell,
                    dev->d_cell_zb,
                    dev->d_cell_inv_area,
                    dev->d_cell_cx, dev->d_cell_cy,
                    dev->d_grad_hx,  dev->d_grad_hy,
                    dev->d_grad_hux, dev->d_grad_huy,
                    dev->d_grad_hvx, dev->d_grad_hvy,
                    dev->d_flux_h, dev->d_flux_hu, dev->d_flux_hv,
                    dev->d_flux_hu_r, dev->d_flux_hv_r,
                    nullptr, nullptr, nullptr,
                    spatial_scheme,
                    g, h_min,
                    max_inv_area,
                    momentum_cap_min_speed,
                    momentum_cap_celerity_mult,
                    dev->d_degen_mask, dev->d_merge_owner, dev->degen_mode,
                    dev->d_active, front_flux_damping, shallow_damping_depth,
                    enable_shallow_front_recon_fallback);
                // update
                CUDA_CHECK(cudaMemsetAsync(dev->d_max_wse_elev_error, 0, sizeof(double), dev->d_stream));
                int grid_update = (n_cells + BLOCK - 1) / BLOCK;
                swe2d_update_kernel<<<grid_update, BLOCK, 0, dev->d_stream>>>(
                    n_cells,
                    dev->d_cell_edge_offsets, dev->d_cell_edge_ids,
                    dev->d_edge_c0, dev->d_edge_c1,
                    dev->d_h, dev->d_hu, dev->d_hv,
                    dev->d_flux_h, dev->d_flux_hu, dev->d_flux_hv,
                    dev->d_flux_hu_r, dev->d_flux_hv_r,
                    dev->d_cell_inv_area, dev->d_n_mann_cell,
                    dev->d_max_wse_elev_error,
                    dt, g, h_min,
                    max_inv_area,
                    momentum_cap_min_speed,
                    momentum_cap_celerity_mult,
                    depth_cap,
                    max_rel_depth_increase,
                    shallow_damping_depth,
                    extreme_rain_mode,
                    source_cfl_beta,
                    source_max_substeps,
                    source_rate_cap,
                    source_depth_step_cap,
                    source_true_subcycling,
                    source_imex_split,
                    dev->d_active,
                    dev->d_degen_mask, dev->d_inv_area_repaired, dev->degen_mode,
                    (dev->n_rain_samples > 0) ? dev->d_cell_source_mps : nullptr,
                    dev->d_external_source_mps,
                    dev->use_culvert_face_flux ? dev->d_ext_struct_flux_h  : nullptr,
                    dev->use_culvert_face_flux ? dev->d_ext_struct_flux_hu : nullptr,
                    dev->use_culvert_face_flux ? dev->d_ext_struct_flux_hv : nullptr);
                // cfl + cfl_reduce (two-level)
                CUDA_CHECK(cudaMemsetAsync(dev->d_lambda_max, 0, sizeof(double), dev->d_stream));
                {
                    int grid_cfl = (n_edges + BLOCK - 1) / BLOCK;
                    if (grid_cfl > 0) {
                        // workspace pre-allocated before graph capture above
                        swe2d_cfl_kernel<<<grid_cfl, BLOCK, BLOCK * static_cast<int>(sizeof(double)), dev->d_stream>>>(
                            n_edges,
                            dev->d_edge_c0, dev->d_edge_c1,
                            dev->d_edge_nx, dev->d_edge_ny, dev->d_edge_len,
                            dev->d_h, dev->d_hu, dev->d_hv,
                            dev->d_cell_area,
                            g, h_min,
                            cfl_lambda_cap,
                            dev->d_cfl_block_max,
                            dev->d_degen_mask, dev->degen_mode);
                        swe2d_cfl_reduce_blocks_kernel<<<1, BLOCK, BLOCK * static_cast<int>(sizeof(double)), dev->d_stream>>>(
                            grid_cfl, dev->d_cfl_block_max, dev->d_lambda_max);
                    }
                }
                pack_diag_kernel<<<1, 1, 0, dev->d_stream>>>(
                    dev->d_lambda_max, dev->d_max_wse_elev_error, dev->d_n_wet, dev->d_diag_packed);

                // End capture, instantiate graph
                cudaGraph_t captured_graph = nullptr;
                cudaError_t cap_end = cudaStreamEndCapture(dev->d_stream, &captured_graph);
                if (cap_end == cudaSuccess && captured_graph != nullptr) {
                    cudaGraphExec_t graph_exec = nullptr;
                    if (cudaGraphInstantiate(&graph_exec, captured_graph, nullptr, nullptr, 0) == cudaSuccess) {
                        cache.graph = captured_graph;
                        cache.exec = graph_exec;
                        cache.n_cells = n_cells;
                        cache.n_edges = n_edges;
                        cache.spatial_scheme = spatial_scheme;
                        cache.time_integrator = graph_integrator;
                        cache.variant_key = graph_variant_key;
                        cache.config_signature = graph_signature;
                        cache.is_valid = true;
                        CUDA_CHECK(cudaGraphLaunch(cache.exec, dev->d_stream));
                        dev->graph_replay_count += 1;
                        used_graph_replay = true;
                    } else {
                        if (graph_exec != nullptr) cudaGraphExecDestroy(graph_exec);
                        cudaGraphDestroy(captured_graph);
                    }
                } else if (captured_graph != nullptr) {
                    cudaGraphDestroy(captured_graph);
                }
            }
        }
    }

    // Kernel 1: Flux
    if (!used_graph_replay) {
    // ── Pre-flux: classify_and_mark → degen → hg_bc → gradient ──────────
    if (dev->d_active && dev->d_n_wet) {
        CUDA_CHECK(cudaMemsetAsync(dev->d_n_wet, 0, sizeof(int32_t), dev->d_stream));
        const int c_grid = (n_cells + BLOCK - 1) / BLOCK;
        swe2d_classify_and_mark_kernel<<<c_grid, BLOCK, BLOCK * sizeof(int32_t), dev->d_stream>>>(
            n_cells, n_edges,
            dev->d_h, dev->d_cell_source_mps, dev->d_external_source_mps,
            dev->d_ext_struct_flux_h,
            dev->d_bc_forced,
            dev->d_edge_c0, dev->d_edge_c1,
            dev->d_active, dev->d_n_wet, h_min,
            active_set_hysteresis ? dev->d_was_active : nullptr);
        CUDA_CHECK(cudaGetLastError());
        if ((dev->degen_mode == 1 || dev->degen_mode == 3) && dev->d_degen_mask) {
            const int c_grid2 = (n_cells + BLOCK - 1) / BLOCK;
            swe2d_degen_deactivate_kernel<<<c_grid2, BLOCK, 0, dev->d_stream>>>(
                n_cells, dev->d_degen_mask, dev->d_active);
            CUDA_CHECK(cudaGetLastError());
        }
        if (dev->degen_mode == 3 && dev->d_degen_mask && dev->d_merge_owner) {
            const int c_grid2 = (n_cells + BLOCK - 1) / BLOCK;
            swe2d_degen_sync_kernel<<<c_grid2, BLOCK, 0, dev->d_stream>>>(
                n_cells, dev->d_degen_mask, dev->d_merge_owner,
                dev->d_h, dev->d_hu, dev->d_hv);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    if (dev->n_hg_edges > 0 && dev->d_hg_edge_index && dev->d_hg_offsets && dev->d_hg_time_s && dev->d_hg_value) {
        const int hg_grid = (dev->n_hg_edges + BLOCK - 1) / BLOCK;
        swe2d_apply_hydrograph_bc_kernel<<<hg_grid, BLOCK, 0, dev->d_stream>>>(
            dev->n_hg_edges, dev->d_hg_edge_index, dev->d_hg_bc_type,
            dev->d_hg_offsets, dev->d_hg_time_s, dev->d_hg_value,
            dev->d_edge_bc, dev->d_edge_bc_val, t_now);
        CUDA_CHECK(cudaGetLastError());
    }

    if (need_gradients) {
        const size_t sz_c = static_cast<size_t>(n_cells) * sizeof(double);
        CUDA_CHECK(cudaMemsetAsync(dev->d_grad_hx,  0, sz_c, dev->d_stream));
        CUDA_CHECK(cudaMemsetAsync(dev->d_grad_hy,  0, sz_c, dev->d_stream));
        CUDA_CHECK(cudaMemsetAsync(dev->d_grad_hux, 0, sz_c, dev->d_stream));
        CUDA_CHECK(cudaMemsetAsync(dev->d_grad_huy, 0, sz_c, dev->d_stream));
        CUDA_CHECK(cudaMemsetAsync(dev->d_grad_hvx, 0, sz_c, dev->d_stream));
        CUDA_CHECK(cudaMemsetAsync(dev->d_grad_hvy, 0, sz_c, dev->d_stream));
        int g_grid = (n_edges + BLOCK - 1) / BLOCK;
        swe2d_gradient_kernel<<<g_grid, BLOCK, 0, dev->d_stream>>>(
            n_edges, dev->d_edge_c0, dev->d_edge_c1,
            dev->d_edge_nx, dev->d_edge_ny, dev->d_edge_len,
            dev->d_h, dev->d_cell_zb, dev->d_hu, dev->d_hv,
            dev->d_cell_inv_area, max_inv_area,
            dev->d_grad_hx, dev->d_grad_hy,
            dev->d_grad_hux, dev->d_grad_huy,
            dev->d_grad_hvx, dev->d_grad_hvy,
            dev->d_active, dev->d_degen_mask, dev->degen_mode);
        CUDA_CHECK(cudaGetLastError());
        swe2d_maybe_launch_lsq_gradient(dev, spatial_scheme, n_cells, BLOCK,
                                        dev->d_h, dev->d_hu, dev->d_hv);
    }

    // ── Flux ────────────────────────────────────────────────────────────
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
            CUDA_CHECK(cudaMemsetAsync(d_dbg_fh, 0xA5, static_cast<size_t>(n_edges) * sizeof(double), dev->d_stream));
            CUDA_CHECK(cudaMemsetAsync(d_dbg_fhu, 0xA5, static_cast<size_t>(n_edges) * sizeof(double), dev->d_stream));
            CUDA_CHECK(cudaMemsetAsync(d_dbg_fhv, 0xA5, static_cast<size_t>(n_edges) * sizeof(double), dev->d_stream));
        }
        int grid = (n_edges + BLOCK - 1) / BLOCK;
        swe2d_flux_kernel<<<grid, BLOCK, 0, dev->d_stream>>>(
            n_edges,
            dev->d_edge_c0, dev->d_edge_c1,
            dev->d_edge_nx, dev->d_edge_ny, dev->d_edge_len,
            dev->d_edge_mx, dev->d_edge_my,
            dev->d_edge_bc, dev->d_edge_bc_val,
            dev->d_h, dev->d_hu, dev->d_hv,
            dev->d_n_mann_cell,
            dev->d_cell_zb,
            dev->d_cell_inv_area,
            dev->d_cell_cx, dev->d_cell_cy,
            dev->d_grad_hx,  dev->d_grad_hy,
            dev->d_grad_hux, dev->d_grad_huy,
            dev->d_grad_hvx, dev->d_grad_hvy,
            dev->d_flux_h, dev->d_flux_hu, dev->d_flux_hv,
            dev->d_flux_hu_r, dev->d_flux_hv_r,
            d_dbg_fh, d_dbg_fhu, d_dbg_fhv,
            spatial_scheme,
            g, h_min,
            max_inv_area,
            momentum_cap_min_speed,
            momentum_cap_celerity_mult,
            dev->d_degen_mask, dev->d_merge_owner, dev->degen_mode,
            dev->d_active, front_flux_damping, shallow_damping_depth,
            enable_shallow_front_recon_fallback);
        CUDA_CHECK(cudaGetLastError());

        if (dbg_edge_flux) {
            CUDA_CHECK(cudaStreamSynchronize(dev->d_stream));
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
        CUDA_CHECK(cudaStreamSynchronize(dev->d_stream));
        std::vector<double> h_flux(static_cast<size_t>(n_edges));
        std::vector<double> hu_flux(static_cast<size_t>(n_edges));
        std::vector<double> hv_flux(static_cast<size_t>(n_edges));
        CUDA_CHECK(cudaMemcpy(h_flux.data(), dev->d_flux_h,
                              static_cast<size_t>(n_edges) * sizeof(double),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hu_flux.data(), dev->d_flux_hu,
                              static_cast<size_t>(n_edges) * sizeof(double),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hv_flux.data(), dev->d_flux_hv,
                              static_cast<size_t>(n_edges) * sizeof(double),
                              cudaMemcpyDeviceToHost));
        dump_flux_summary("GPU", h_flux, hu_flux, hv_flux);
    }

    // Kernel 2: Update
    {
        CUDA_CHECK(cudaMemsetAsync(dev->d_max_wse_elev_error, 0, sizeof(double), dev->d_stream));
        int grid = (n_cells + BLOCK - 1) / BLOCK;
        swe2d_update_kernel<<<grid, BLOCK, 0, dev->d_stream>>>(
            n_cells,
            dev->d_cell_edge_offsets, dev->d_cell_edge_ids,
            dev->d_edge_c0, dev->d_edge_c1,
            dev->d_h, dev->d_hu, dev->d_hv,
            dev->d_flux_h, dev->d_flux_hu, dev->d_flux_hv,
            dev->d_flux_hu_r, dev->d_flux_hv_r,
            dev->d_cell_inv_area, dev->d_n_mann_cell,
            dev->d_max_wse_elev_error,
            dt, g, h_min,
            max_inv_area,
            momentum_cap_min_speed,
            momentum_cap_celerity_mult,
            depth_cap,
            max_rel_depth_increase,
            shallow_damping_depth,
            extreme_rain_mode,
            source_cfl_beta,
            source_max_substeps,
            source_rate_cap,
            source_depth_step_cap,
            source_true_subcycling,
            source_imex_split,
            dev->d_active,
            dev->d_degen_mask, dev->d_inv_area_repaired, dev->degen_mode,
            (dev->n_rain_samples > 0) ? dev->d_cell_source_mps : nullptr,
            dev->d_external_source_mps,
            dev->use_culvert_face_flux ? dev->d_ext_struct_flux_h  : nullptr,
            dev->use_culvert_face_flux ? dev->d_ext_struct_flux_hu : nullptr,
            dev->use_culvert_face_flux ? dev->d_ext_struct_flux_hv : nullptr);
        CUDA_CHECK(cudaGetLastError());
    }

    // Kernel 3: CFL reduction for max Courant diagnostic
    CUDA_CHECK(cudaMemsetAsync(dev->d_lambda_max, 0, sizeof(double), dev->d_stream));
    {
        int grid = (n_edges + BLOCK - 1) / BLOCK;
        if (grid > 0) {
            swe2d_ensure_cfl_block_workspace(dev, grid);
            swe2d_cfl_kernel<<<grid, BLOCK, BLOCK * static_cast<int>(sizeof(double)), dev->d_stream>>>(
                n_edges,
                dev->d_edge_c0, dev->d_edge_c1,
                dev->d_edge_nx, dev->d_edge_ny, dev->d_edge_len,
                dev->d_h, dev->d_hu, dev->d_hv,
                dev->d_cell_area,
                g, h_min,
                cfl_lambda_cap,
                dev->d_cfl_block_max,
                dev->d_degen_mask, dev->degen_mode);
            CUDA_CHECK(cudaGetLastError());
            swe2d_cfl_reduce_blocks_kernel<<<1, BLOCK, BLOCK * static_cast<int>(sizeof(double)), dev->d_stream>>>(
                grid, dev->d_cfl_block_max, dev->d_lambda_max);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    // Pack all three diagnostic scalars into contiguous buffer for single-transfer readback.
    pack_diag_kernel<<<1, 1, 0, dev->d_stream>>>(dev->d_lambda_max, dev->d_max_wse_elev_error, dev->d_n_wet, dev->d_diag_packed);
    CUDA_CHECK(cudaGetLastError());
    }

    // Fill diagnostics. For high-throughput loops this can skip host sync and
    // report sentinel values until a synchronized diagnostic sample is requested.
    if (diag) {
        diag->dt         = dt;
        diag->gpu_active = true;
        diag->wet_cells  = -1;
        diag->max_depth  = -1.0;
        diag->min_depth  = -1.0;
        diag->mass_total = -1.0;
        diag->max_courant = -1.0;
        diag->max_depth_residual = -1.0;
        diag->max_wse_elev_error = -1.0;
        diag->gpu_graph_launches_step = used_graph_replay ? 1 : 0;
        diag->gpu_graph_launches_total = static_cast<int64_t>(dev->graph_replay_count);

        if (sync_diagnostics) {
            CUDA_CHECK(cudaStreamSynchronize(dev->d_stream));
            double packed[3] = {0.0, 0.0, -1.0};
            CUDA_CHECK(cudaMemcpy(packed, dev->d_diag_packed, 3 * sizeof(double), cudaMemcpyDeviceToHost));
            diag->max_courant        = dt * packed[0];
            diag->max_depth_residual = packed[1];
            diag->max_wse_elev_error = packed[1];
            diag->wet_cells          = static_cast<int32_t>(packed[2]);
        }
    }
}

__global__ __launch_bounds__(256, 4) void swe2d_persistent_chunk_kernel_first_order(
    int32_t n_edges,
    int32_t n_active_edges,
    int32_t n_cells,
    int chunk_substeps,
    double dt_sub,
    double g,
    double h_min,
    double max_inv_area,
    double momentum_cap_min_speed,
    double momentum_cap_celerity_mult,
    double depth_cap,
    double max_rel_depth_increase,
    double shallow_damping_depth,
    double source_rate_cap,
    double source_depth_step_cap,
    const int32_t* __restrict__ edge_c0,
    const int32_t* __restrict__ edge_c1,
    const double*  __restrict__ edge_nx,
    const double*  __restrict__ edge_ny,
    const double*  __restrict__ edge_len,
    const int32_t* __restrict__ edge_bc,
    const double*  __restrict__ edge_bc_val,
    const int32_t* __restrict__ cell_edge_offsets,
    const int32_t* __restrict__ cell_edge_ids,
    const double*  __restrict__ cell_inv_area,
    const double*  __restrict__ cell_n_mann,
    const double*  __restrict__ cell_zb,
    double* __restrict__ cell_h,
    double* __restrict__ cell_hu,
    double* __restrict__ cell_hv,
    double* __restrict__ flux_h,
    double* __restrict__ flux_hu,
    double* __restrict__ flux_hv,
    double* __restrict__ flux_hu_r,
    double* __restrict__ flux_hv_r,
    const double* __restrict__ cell_source_mps,
    const double* __restrict__ external_source_mps,
    const int32_t* __restrict__ d_active,
    const int32_t* __restrict__ d_degen_mask,
    const int32_t* __restrict__ d_merge_owner,
    const double* __restrict__ d_inv_area_repaired,
    int degen_mode,
    const int32_t* __restrict__ active_edge_ids,
    double front_flux_damping)
{
    cg::grid_group grid = cg::this_grid();
    const int32_t tid = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);

    for (int sub = 0; sub < chunk_substeps; ++sub) {
        if (tid < n_active_edges) {
            const int32_t e = active_edge_ids ? active_edge_ids[tid] : tid;
            const int32_t c0 = edge_c0[e];
            const int32_t c1 = edge_c1[e];

            const int32_t dm0 = d_degen_mask ? d_degen_mask[c0]
                                             : (cell_inv_area[c0] > max_inv_area ? 1 : 0);
            const int32_t dm1 = (c1 >= 0) ? (d_degen_mask ? d_degen_mask[c1]
                                                           : (cell_inv_area[c1] > max_inv_area ? 1 : 0))
                                           : 0;
            if (degen_mode <= 1 && (dm0 || dm1)) {
                flux_h[e] = 0.0;
                flux_hu[e] = 0.0;
                flux_hv[e] = 0.0;
                flux_hu_r[e] = 0.0;
                flux_hv_r[e] = 0.0;
                continue;
            }
            if (degen_mode == 3) {
                if (dm0 && (d_merge_owner == nullptr || d_merge_owner[c0] < 0)) {
                    flux_h[e] = 0.0;
                    flux_hu[e] = 0.0;
                    flux_hv[e] = 0.0;
                    flux_hu_r[e] = 0.0;
                    flux_hv_r[e] = 0.0;
                    continue;
                }
                if (dm1 && (d_merge_owner == nullptr || d_merge_owner[c1] < 0)) {
                    flux_h[e] = 0.0;
                    flux_hu[e] = 0.0;
                    flux_hv[e] = 0.0;
                    flux_hu_r[e] = 0.0;
                    flux_hv_r[e] = 0.0;
                    continue;
                }
            }

            if (d_active && ((c1 >= 0 && !d_active[c0] && !d_active[c1]) ||
                             (c1 < 0 && !d_active[c0]))) {
                flux_h[e] = 0.0;
                flux_hu[e] = 0.0;
                flux_hv[e] = 0.0;
                flux_hu_r[e] = 0.0;
                flux_hv_r[e] = 0.0;
            } else {
                const double nx = edge_nx[e];
                const double ny = edge_ny[e];
                const double len = edge_len[e];

                const double hL = cell_h[c0];
                const double huL = cell_hu[c0];
                const double hvL = cell_hv[c0];
                const double zbL = cell_zb[c0];

                double hR = 0.0;
                double huR = 0.0;
                double hvR = 0.0;
                double zbR = 0.0;
                if (c1 >= 0) {
                    hR = cell_h[c1];
                    huR = cell_hu[c1];
                    hvR = cell_hv[c1];
                    zbR = cell_zb[c1];
                } else {
                    const double n_local = cell_n_mann ? cell_n_mann[c0] : 0.03;
                    GhostStateLocal gs = make_ghost_cuda_local(
                        hL, huL, hvL, zbL, nx, ny,
                        edge_bc[e], edge_bc_val[e], h_min, n_local);
                    hR = gs.h;
                    huR = gs.hu;
                    hvR = gs.hv;
                    zbR = gs.zb;
                }

                ReconstructedStatesLocal rs = hydrostatic_reconstruct_cuda_local(
                    hL, huL, hvL, zbL, hR, huR, hvR, zbR, h_min);

                double flux_fh = 0.0, flux_fhu = 0.0, flux_fhv = 0.0;
                hllc_flux_cuda_local(
                    rs.hL_star, rs.uL, rs.vL,
                    rs.hR_star, rs.uR, rs.vR,
                    nx, ny, g, h_min,
                    flux_fh, flux_fhu, flux_fhv);

                double corr_hu = 0.0, corr_hv = 0.0;
                bed_slope_correction_cuda_local(hL, rs.hL_star, nx, ny, g, corr_hu, corr_hv);

                double fh = flux_fh * len;
                double fhu_l = (flux_fhu + corr_hu) * len;
                double fhv_l = (flux_fhv + corr_hv) * len;

                const bool is_wet_dry_front = d_active && (c1 >= 0) &&
                                              ((d_active[c0] != 0) != (d_active[c1] != 0));
                if (is_wet_dry_front && front_flux_damping < 1.0) {
                    fhu_l *= front_flux_damping;
                    fhv_l *= front_flux_damping;
                }

                double corr_hu_r = 0.0, corr_hv_r = 0.0;
                if (c1 >= 0) {
                    bed_slope_correction_cuda_local(hR, rs.hR_star, nx, ny, g, corr_hu_r, corr_hv_r);
                }
                double fhu_r = (flux_fhu + corr_hu_r) * len;
                double fhv_r = (flux_fhv + corr_hv_r) * len;
                if (is_wet_dry_front && front_flux_damping < 1.0) {
                    fhu_r *= front_flux_damping;
                    fhv_r *= front_flux_damping;
                }

                flux_h[e] = -fh;
                flux_hu[e] = -fhu_l;
                flux_hv[e] = -fhv_l;
                flux_hu_r[e] = fhu_r;
                flux_hv_r[e] = fhv_r;
            }
        }

        grid.sync();

        if (tid < n_cells) {
            const int32_t c = tid;
            bool do_update = true;
            if (d_active && !d_active[c]) {
                const double src0 = (cell_source_mps ? cell_source_mps[c] : 0.0) +
                                    (external_source_mps ? external_source_mps[c] : 0.0);
                if (!(isfinite(src0) && src0 > 0.0)) {
                    do_update = false;
                }
            }

            if (do_update) {
                if (d_degen_mask && d_degen_mask[c] && degen_mode != 2) {
                    do_update = false;
                }
            }

            if (do_update) {
                const double h_old = cell_h[c];
                double inv_a;
                if (degen_mode == 2 && d_inv_area_repaired != nullptr && d_degen_mask && d_degen_mask[c]) {
                    inv_a = d_inv_area_repaired[c];
                } else {
                    inv_a = cell_inv_area[c];
                    const double max_inv_a = fmax(max_inv_area, 1.0);
                    if (inv_a > max_inv_a) inv_a = max_inv_a;
                }

                double fh = 0.0;
                double fhu = 0.0;
                double fhv = 0.0;
                const int32_t s = cell_edge_offsets[c];
                const int32_t e = cell_edge_offsets[c + 1];
                for (int32_t k = s; k < e; ++k) {
                    const int32_t edge = cell_edge_ids[k];
                    if (edge_c0[edge] == c) {
                        fh += flux_h[edge];
                        fhu += flux_hu[edge];
                        fhv += flux_hv[edge];
                    } else {
                        fh -= flux_h[edge];
                        fhu += flux_hu_r[edge];
                        fhv += flux_hv_r[edge];
                    }
                }

                double h_trial = cell_h[c] + dt_sub * fh * inv_a;
                double src = (cell_source_mps ? cell_source_mps[c] : 0.0) +
                             (external_source_mps ? external_source_mps[c] : 0.0);
                if (src > 0.0 && source_rate_cap > 0.0 && src > source_rate_cap) src = source_rate_cap;
                if (src > 0.0 && source_depth_step_cap > 0.0) {
                    const double src_step_cap = source_depth_step_cap / fmax(dt_sub, 1.0e-12);
                    if (src > src_step_cap) src = src_step_cap;
                }
                h_trial += dt_sub * src;

                if (max_rel_depth_increase > 0.0) {
                    const double h_ref = fmax(h_old, h_min);
                    const double h_step_cap = h_old + max_rel_depth_increase * h_ref;
                    if (h_trial > h_step_cap) h_trial = h_step_cap;
                }
                if (depth_cap > 0.0 && h_trial > depth_cap) h_trial = depth_cap;
                if (!isfinite(h_trial) || h_trial < 0.0) h_trial = 0.0;

                cell_h[c] = h_trial;
                cell_hu[c] += dt_sub * fhu * inv_a;
                cell_hv[c] += dt_sub * fhv * inv_a;

                if (cell_h[c] < h_min) {
                    cell_hu[c] = 0.0;
                    cell_hv[c] = 0.0;
                } else if (shallow_damping_depth > h_min && cell_h[c] < shallow_damping_depth) {
                    const double t = (cell_h[c] - h_min) / (shallow_damping_depth - h_min);
                    const double ts = fmin(1.0, fmax(0.0, t));
                    const double scale = ts * ts * (3.0 - 2.0 * ts);
                    cell_hu[c] *= scale;
                    cell_hv[c] *= scale;
                }

                double n_mann = cell_n_mann[c];
                apply_friction_cuda_local(cell_h[c], cell_hu[c], cell_hv[c], dt_sub, n_mann, g, h_min);

                if (!isfinite(cell_h[c]) || !isfinite(cell_hu[c]) || !isfinite(cell_hv[c])) {
                    cell_h[c] = 0.0;
                    cell_hu[c] = 0.0;
                    cell_hv[c] = 0.0;
                } else if (cell_h[c] > h_min) {
                    const double inv_h = 1.0 / cell_h[c];
                    const double u = cell_hu[c] * inv_h;
                    const double v = cell_hv[c] * inv_h;
                    const double spd = sqrt(u * u + v * v);
                    const double spd_cap = fmax(momentum_cap_min_speed,
                                                momentum_cap_celerity_mult * sqrt(g * cell_h[c]));
                    if (isfinite(spd) && spd > spd_cap && spd > 0.0) {
                        const double scale = spd_cap / spd;
                        cell_hu[c] *= scale;
                        cell_hv[c] *= scale;
                    }
                }
            }
        }

        grid.sync();
    }
}

void swe2d_gpu_step_persistent_chunk(
    SWE2DDeviceState* dev,
    double t_now,
    double dt,
    int chunk_substeps,
    double g,
    double h_min,
    int spatial_scheme,
    double cfl_factor,
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
    bool enable_active_edge_compaction,
    bool sync_diagnostics,
    SWE2DStepDiag* diag,
    double front_flux_damping,
    bool active_set_hysteresis)
{
    (void)t_now;
    auto run_chunked_baseline = [&](bool sync_final_diag) {
        if (!dev || chunk_substeps <= 1 || dt <= 0.0) {
            swe2d_gpu_step(
                dev, t_now, dt, g, h_min, spatial_scheme, cfl_factor,
                max_inv_area, cfl_lambda_cap, momentum_cap_min_speed, momentum_cap_celerity_mult,
                depth_cap, max_rel_depth_increase, shallow_damping_depth,
                extreme_rain_mode, source_cfl_beta, source_max_substeps,
                source_rate_cap, source_depth_step_cap, source_true_subcycling, source_imex_split,
                enable_shallow_front_recon_fallback, sync_final_diag, diag, front_flux_damping, active_set_hysteresis);
            return;
        }
        SWE2DStepDiag sub_diag{};
        const double dt_sub = dt / static_cast<double>(chunk_substeps);
        for (int sub = 0; sub < chunk_substeps; ++sub) {
            swe2d_gpu_step(
                dev,
                t_now + static_cast<double>(sub) * dt_sub,
                dt_sub,
                g,
                h_min,
                spatial_scheme,
                cfl_factor,
                max_inv_area,
                cfl_lambda_cap,
                momentum_cap_min_speed,
                momentum_cap_celerity_mult,
                depth_cap,
                max_rel_depth_increase,
                shallow_damping_depth,
                extreme_rain_mode,
                source_cfl_beta,
                source_max_substeps,
                source_rate_cap,
                source_depth_step_cap,
                source_true_subcycling,
                source_imex_split,
                enable_shallow_front_recon_fallback,
                sync_final_diag && (sub == (chunk_substeps - 1)),
                &sub_diag,
                front_flux_damping,
                active_set_hysteresis);
        }
        if (diag) {
            *diag = sub_diag;
            diag->dt = dt;
        }
    };

    if (!dev || chunk_substeps <= 1 || dt <= 0.0) {
        swe2d_gpu_step(
            dev, t_now, dt, g, h_min, spatial_scheme, cfl_factor,
            max_inv_area, cfl_lambda_cap, momentum_cap_min_speed, momentum_cap_celerity_mult,
            depth_cap, max_rel_depth_increase, shallow_damping_depth,
            extreme_rain_mode, source_cfl_beta, source_max_substeps,
            source_rate_cap, source_depth_step_cap, source_true_subcycling, source_imex_split,
            enable_shallow_front_recon_fallback, sync_diagnostics, diag, front_flux_damping, active_set_hysteresis);
        return;
    }

    // Cooperative persistent kernel currently accelerates first-order single-stage hydrostatic path.
    const bool cooperative_kernel_supported =
        (spatial_scheme == static_cast<int>(SWE2DSpatialScheme::FV_FIRST_ORDER)) &&
        !extreme_rain_mode &&
        !source_true_subcycling &&
        !source_imex_split;
    if (!cooperative_kernel_supported) {
        run_chunked_baseline(sync_diagnostics);
        return;
    }

    constexpr int BLOCK = 256;
    int32_t n_cells = dev->n_cells;
    int32_t n_edges = dev->n_edges;
    double dt_sub = dt / static_cast<double>(chunk_substeps);
    int32_t n_flux_edges = n_edges;
    const int32_t* d_flux_edge_ids = nullptr;

    if (dev->n_rain_samples > 0 && dev->d_cell_gage_idx && dev->d_rain_hg_offsets && dev->d_rain_hg_time_s && dev->d_rain_hg_cum_mm && dev->d_rain_cn) {
        const int r_grid = (n_cells + BLOCK - 1) / BLOCK;
        swe2d_build_rain_cn_source_kernel<<<r_grid, BLOCK, 0, dev->d_stream>>>(
            n_cells,
            dev->d_cell_gage_idx,
            dev->d_rain_hg_offsets,
            dev->d_rain_hg_time_s,
            dev->d_rain_hg_cum_mm,
            dev->d_rain_cn,
            dev->d_rain_cum_mm,
            dev->d_rain_excess_cum_mm,
            dev->d_cell_source_mps,
            t_now,
            t_now + dt,
            dev->rain_ia_ratio,
            dev->rain_mm_to_model_depth);
        CUDA_CHECK(cudaGetLastError());
    }

    if (dev->d_active && dev->d_n_wet) {
        if (active_set_hysteresis && dev->d_was_active) {
            CUDA_CHECK(cudaMemcpyAsync(dev->d_was_active, dev->d_active,
                                       static_cast<size_t>(n_cells) * sizeof(int32_t),
                                       cudaMemcpyDeviceToDevice,
                                       dev->d_stream));
        }
        CUDA_CHECK(cudaMemsetAsync(dev->d_n_wet, 0, sizeof(int32_t), dev->d_stream));
        const int c_grid = (n_cells + BLOCK - 1) / BLOCK;
        swe2d_classify_kernel<<<c_grid, BLOCK, BLOCK * sizeof(int32_t), dev->d_stream>>>(
            n_cells, dev->d_h, dev->d_cell_source_mps, dev->d_external_source_mps,
            dev->d_ext_struct_flux_h,
            dev->d_bc_forced,
            dev->d_active, dev->d_n_wet, h_min,
            active_set_hysteresis ? dev->d_was_active : nullptr);
        CUDA_CHECK(cudaGetLastError());
        const int e_grid = (n_edges + BLOCK - 1) / BLOCK;
        swe2d_mark_neighbor_kernel<<<e_grid, BLOCK, 0, dev->d_stream>>>(
            n_edges, dev->d_edge_c0, dev->d_edge_c1, dev->d_active);
        CUDA_CHECK(cudaGetLastError());
        if ((dev->degen_mode == 1 || dev->degen_mode == 3) && dev->d_degen_mask) {
            const int c_grid2 = (n_cells + BLOCK - 1) / BLOCK;
            swe2d_degen_deactivate_kernel<<<c_grid2, BLOCK, 0, dev->d_stream>>>(
                n_cells, dev->d_degen_mask, dev->d_active);
            CUDA_CHECK(cudaGetLastError());
        }
        if (dev->degen_mode == 3 && dev->d_degen_mask && dev->d_merge_owner) {
            const int c_grid2 = (n_cells + BLOCK - 1) / BLOCK;
            swe2d_degen_sync_kernel<<<c_grid2, BLOCK, 0, dev->d_stream>>>(
                n_cells, dev->d_degen_mask, dev->d_merge_owner,
                dev->d_h, dev->d_hu, dev->d_hv);
            CUDA_CHECK(cudaGetLastError());
        }

        if (enable_active_edge_compaction && dev->d_active_edge_ids && dev->d_n_active_edges) {
            CUDA_CHECK(cudaMemsetAsync(dev->d_n_active_edges, 0, sizeof(int32_t), dev->d_stream));
            const int e_grid2 = (n_edges + BLOCK - 1) / BLOCK;
            swe2d_collect_active_edges_kernel<<<e_grid2, BLOCK, 0, dev->d_stream>>>(
                n_edges,
                dev->d_edge_c0,
                dev->d_edge_c1,
                dev->d_active,
                dev->d_active_edge_ids,
                dev->d_n_active_edges);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(&n_flux_edges, dev->d_n_active_edges, sizeof(int32_t), cudaMemcpyDeviceToHost));
            if (n_flux_edges < 0) n_flux_edges = 0;
            if (n_flux_edges > n_edges) n_flux_edges = n_edges;
            d_flux_edge_ids = dev->d_active_edge_ids;
        }
    }

    if (dev->n_hg_edges > 0 && dev->d_hg_edge_index && dev->d_hg_offsets && dev->d_hg_time_s && dev->d_hg_value) {
        const int hg_grid = (dev->n_hg_edges + BLOCK - 1) / BLOCK;
        swe2d_apply_hydrograph_bc_kernel<<<hg_grid, BLOCK, 0, dev->d_stream>>>(
            dev->n_hg_edges,
            dev->d_hg_edge_index,
            dev->d_hg_bc_type,
            dev->d_hg_offsets,
            dev->d_hg_time_s,
            dev->d_hg_value,
            dev->d_edge_bc,
            dev->d_edge_bc_val,
            t_now);
        CUDA_CHECK(cudaGetLastError());
    }

    cudaDeviceProp prop{};
    int dev_id = 0;
    CUDA_CHECK(cudaGetDevice(&dev_id));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev_id));
    if (!prop.cooperativeLaunch) {
        run_chunked_baseline(sync_diagnostics);
        return;
    }

    int max_active_blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks_per_sm,
        swe2d_persistent_chunk_kernel_first_order,
        BLOCK,
        0));
    const int max_coop_blocks = max_active_blocks_per_sm * prop.multiProcessorCount;
    const int needed_blocks = (std::max(n_cells, n_flux_edges) + BLOCK - 1) / BLOCK;
    if (needed_blocks <= 0 || needed_blocks > max_coop_blocks) {
        run_chunked_baseline(sync_diagnostics);
        return;
    }

    void* args[] = {
        &n_edges,
        &n_flux_edges,
        &n_cells,
        &chunk_substeps,
        &dt_sub,
        &g,
        &h_min,
        &max_inv_area,
        &momentum_cap_min_speed,
        &momentum_cap_celerity_mult,
        &depth_cap,
        &max_rel_depth_increase,
        &shallow_damping_depth,
        &source_rate_cap,
        &source_depth_step_cap,
        &dev->d_edge_c0,
        &dev->d_edge_c1,
        &dev->d_edge_nx,
        &dev->d_edge_ny,
        &dev->d_edge_len,
        &dev->d_edge_bc,
        &dev->d_edge_bc_val,
        &dev->d_cell_edge_offsets,
        &dev->d_cell_edge_ids,
        &dev->d_cell_inv_area,
        &dev->d_n_mann_cell,
        &dev->d_cell_zb,
        &dev->d_h,
        &dev->d_hu,
        &dev->d_hv,
        &dev->d_flux_h,
        &dev->d_flux_hu,
        &dev->d_flux_hv,
        &dev->d_flux_hu_r,
        &dev->d_flux_hv_r,
        &dev->d_cell_source_mps,
        &dev->d_external_source_mps,
        &dev->d_active,
        &dev->d_degen_mask,
        &dev->d_merge_owner,
        &dev->d_inv_area_repaired,
        &dev->degen_mode,
        &d_flux_edge_ids,
        &front_flux_damping,
    };

    cudaError_t coop_err = cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(swe2d_persistent_chunk_kernel_first_order),
        needed_blocks,
        BLOCK,
        args,
        0,
        dev->d_stream);
    if (coop_err != cudaSuccess) {
        run_chunked_baseline(sync_diagnostics);
        return;
    }

    CUDA_CHECK(cudaMemsetAsync(dev->d_lambda_max, 0, sizeof(double), dev->d_stream));
    const int cfl_grid = (n_edges + BLOCK - 1) / BLOCK;
    swe2d_cfl_kernel<<<cfl_grid, BLOCK, BLOCK * static_cast<int>(sizeof(double)), dev->d_stream>>>(
        n_edges,
        dev->d_edge_c0, dev->d_edge_c1,
        dev->d_edge_nx, dev->d_edge_ny, dev->d_edge_len,
        dev->d_h, dev->d_hu, dev->d_hv,
        dev->d_cell_area,
        g, h_min,
        cfl_lambda_cap,
        dev->d_lambda_max,
        dev->d_degen_mask, dev->degen_mode);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemsetAsync(dev->d_max_wse_elev_error, 0, sizeof(double), dev->d_stream));
    pack_diag_kernel<<<1, 1, 0, dev->d_stream>>>(dev->d_lambda_max, dev->d_max_wse_elev_error, dev->d_n_wet, dev->d_diag_packed);
    CUDA_CHECK(cudaGetLastError());

    if (diag) {
        diag->dt = dt;
        diag->gpu_active = true;
        diag->wet_cells = -1;
        diag->max_depth = -1.0;
        diag->min_depth = -1.0;
        diag->mass_total = -1.0;
        diag->max_courant = -1.0;
        diag->max_depth_residual = -1.0;
        diag->max_wse_elev_error = -1.0;
        diag->gpu_graph_launches_step = 0;
        diag->gpu_graph_launches_total = static_cast<int64_t>(dev->graph_replay_count);
        if (sync_diagnostics) {
            CUDA_CHECK(cudaStreamSynchronize(dev->d_stream));
            double packed[3] = {0.0, 0.0, -1.0};
            CUDA_CHECK(cudaMemcpy(packed, dev->d_diag_packed, 3 * sizeof(double), cudaMemcpyDeviceToHost));
            diag->max_courant = dt * packed[0];
            diag->max_depth_residual = packed[1];
            diag->max_wse_elev_error = packed[1];
            diag->wet_cells = static_cast<int32_t>(packed[2]);
        }
    }
}

double swe2d_gpu_compute_dt(
    SWE2DDeviceState* dev,
    double g,
    double h_min,
    double cfl_factor,
    double dt_max,
    double cfl_lambda_cap)
{
    constexpr int BLOCK = 256;
    const int32_t n_edges = dev->n_edges;

    // Clear any stale CUDA error from prior operations so the CFL kernel
    // error check is not triggered by a latent error.
    (void)cudaGetLastError();

    CUDA_CHECK(cudaMemsetAsync(dev->d_lambda_max, 0, sizeof(double), dev->d_stream));
    int grid = (n_edges + BLOCK - 1) / BLOCK;
    if (grid > 0) {
        // Match the two-stage CFL reduction path used in the solver step kernels.
        swe2d_ensure_cfl_block_workspace(dev, grid);

        swe2d_cfl_kernel<<<grid, BLOCK, BLOCK * static_cast<int>(sizeof(double)), dev->d_stream>>>(
            n_edges,
            dev->d_edge_c0, dev->d_edge_c1,
            dev->d_edge_nx, dev->d_edge_ny, dev->d_edge_len,
            dev->d_h, dev->d_hu, dev->d_hv,
            dev->d_cell_area,
            g, h_min,
            cfl_lambda_cap,
            dev->d_cfl_block_max,
            dev->d_degen_mask, dev->degen_mode);
        CUDA_CHECK(cudaGetLastError());
        swe2d_cfl_reduce_blocks_kernel<<<1, BLOCK, BLOCK * static_cast<int>(sizeof(double)), dev->d_stream>>>(
            grid, dev->d_cfl_block_max, dev->d_lambda_max);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaStreamSynchronize(dev->d_stream));

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
    double t_now,
    double dt,
    double g,
    double h_min,
    int spatial_scheme,
    double cfl_factor,
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
    bool sync_diagnostics,
    SWE2DStepDiag* diag,
    double front_flux_damping,
    bool   active_set_hysteresis)
{
    constexpr int BLOCK = 256;
    const int32_t n_cells = dev->n_cells;
    const size_t sz = static_cast<size_t>(n_cells) * sizeof(double);
    const uint64_t graph_launches_before = dev->graph_replay_count;
    const int32_t prev_graph_integrator = dev->kernel_graph_cache.time_integrator;
    dev->kernel_graph_cache.time_integrator = 2;

    CUDA_CHECK(cudaMemcpyAsync(dev->d_h0, dev->d_h, sz, cudaMemcpyDeviceToDevice, dev->d_stream));
    CUDA_CHECK(cudaMemcpyAsync(dev->d_hu0, dev->d_hu, sz, cudaMemcpyDeviceToDevice, dev->d_stream));
    CUDA_CHECK(cudaMemcpyAsync(dev->d_hv0, dev->d_hv, sz, cudaMemcpyDeviceToDevice, dev->d_stream));

    SWE2DStepDiag tmp_diag;
    swe2d_gpu_step(dev, t_now, dt, g, h_min, spatial_scheme, cfl_factor,
                   max_inv_area, cfl_lambda_cap,
                   momentum_cap_min_speed, momentum_cap_celerity_mult,
                   depth_cap, max_rel_depth_increase, shallow_damping_depth,
                   extreme_rain_mode, source_cfl_beta, source_max_substeps,
                   source_rate_cap, source_depth_step_cap, source_true_subcycling, source_imex_split,
                   enable_shallow_front_recon_fallback,
                   false,
                   &tmp_diag,
                   front_flux_damping, active_set_hysteresis);

    // Save CN cumulative state at t+dt so RK2 source prediction in the second
    // stage does not commit cumulative rainfall/excess through t+2dt.
    const bool has_rain_cn_state = (
        dev->d_rain_cum_mm &&
        dev->d_rain_excess_cum_mm &&
        dev->d_h1 &&
        dev->d_h2
    );
    if (has_rain_cn_state) {
        CUDA_CHECK(cudaMemcpyAsync(dev->d_h1, dev->d_rain_cum_mm, sz, cudaMemcpyDeviceToDevice, dev->d_stream));
        CUDA_CHECK(cudaMemcpyAsync(dev->d_h2, dev->d_rain_excess_cum_mm, sz, cudaMemcpyDeviceToDevice, dev->d_stream));
    }

    swe2d_gpu_step(dev, t_now + dt, dt, g, h_min, spatial_scheme, cfl_factor,
                   max_inv_area, cfl_lambda_cap,
                   momentum_cap_min_speed, momentum_cap_celerity_mult,
                   depth_cap, max_rel_depth_increase, shallow_damping_depth,
                   extreme_rain_mode, source_cfl_beta, source_max_substeps,
                   source_rate_cap, source_depth_step_cap, source_true_subcycling, source_imex_split,
                   enable_shallow_front_recon_fallback,
                   false,
                   &tmp_diag,
                   front_flux_damping, active_set_hysteresis);

    CUDA_CHECK(cudaMemsetAsync(dev->d_max_wse_elev_error, 0, sizeof(double), dev->d_stream));
    int grid = (n_cells + BLOCK - 1) / BLOCK;
    swe2d_rk2_combine_kernel<<<grid, BLOCK, 0, dev->d_stream>>>(
        n_cells,
        dev->d_h, dev->d_hu, dev->d_hv,
        dev->d_h0, dev->d_hu0, dev->d_hv0,
        dev->d_max_wse_elev_error,
        h_min);
    CUDA_CHECK(cudaGetLastError());

    if (has_rain_cn_state) {
        CUDA_CHECK(cudaMemcpyAsync(dev->d_rain_cum_mm, dev->d_h1, sz, cudaMemcpyDeviceToDevice, dev->d_stream));
        CUDA_CHECK(cudaMemcpyAsync(dev->d_rain_excess_cum_mm, dev->d_h2, sz, cudaMemcpyDeviceToDevice, dev->d_stream));
    }

    CUDA_CHECK(cudaMemsetAsync(dev->d_lambda_max, 0, sizeof(double), dev->d_stream));
    const int edge_grid = (dev->n_edges + BLOCK - 1) / BLOCK;
    swe2d_cfl_kernel<<<edge_grid, BLOCK, BLOCK * static_cast<int>(sizeof(double)), dev->d_stream>>>(
        dev->n_edges,
        dev->d_edge_c0, dev->d_edge_c1,
        dev->d_edge_nx, dev->d_edge_ny, dev->d_edge_len,
        dev->d_h, dev->d_hu, dev->d_hv,
        dev->d_cell_area,
        g, h_min,
        cfl_lambda_cap,
        dev->d_lambda_max,
        dev->d_degen_mask, dev->degen_mode);
    CUDA_CHECK(cudaGetLastError());
    // Pack diagnostic scalars for single-transfer readback.
    pack_diag_kernel<<<1, 1, 0, dev->d_stream>>>(dev->d_lambda_max, dev->d_max_wse_elev_error, dev->d_n_wet, dev->d_diag_packed);
    CUDA_CHECK(cudaGetLastError());

    if (diag) {
        diag->dt = dt;
        diag->gpu_active = true;
        diag->wet_cells = -1;
        diag->max_depth = -1.0;
        diag->min_depth = -1.0;
        diag->mass_total = -1.0;
        diag->max_courant = -1.0;
        diag->max_depth_residual = -1.0;
        diag->max_wse_elev_error = -1.0;
        const uint64_t graph_launches_after = dev->graph_replay_count;
        diag->gpu_graph_launches_step = static_cast<int32_t>(graph_launches_after - graph_launches_before);
        diag->gpu_graph_launches_total = static_cast<int64_t>(graph_launches_after);

        if (sync_diagnostics) {
            CUDA_CHECK(cudaStreamSynchronize(dev->d_stream));
            double packed[3] = {0.0, 0.0, -1.0};
            CUDA_CHECK(cudaMemcpy(packed, dev->d_diag_packed, 3 * sizeof(double), cudaMemcpyDeviceToHost));
            diag->max_courant        = dt * packed[0];
            diag->max_depth_residual = packed[1];
            diag->max_wse_elev_error = packed[1];
            diag->wet_cells          = static_cast<int32_t>(packed[2]);
        }
    }

    dev->kernel_graph_cache.time_integrator = prev_graph_integrator;
}

void swe2d_gpu_step_rk2_persistent_chunk(
    SWE2DDeviceState* dev,
    double t_now,
    double dt,
    int chunk_substeps,
    double g,
    double h_min,
    int spatial_scheme,
    double cfl_factor,
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
    bool enable_active_edge_compaction,
    bool sync_diagnostics,
    SWE2DStepDiag* diag,
    double front_flux_damping,
    bool   active_set_hysteresis)
{
    if (!dev) throw std::invalid_argument("swe2d_gpu_step_rk2_persistent_chunk: null device");
    if (chunk_substeps <= 1) {
        swe2d_gpu_step_rk2(
            dev, t_now, dt, g, h_min, spatial_scheme, cfl_factor,
            max_inv_area, cfl_lambda_cap,
            momentum_cap_min_speed, momentum_cap_celerity_mult,
            depth_cap, max_rel_depth_increase, shallow_damping_depth,
            extreme_rain_mode, source_cfl_beta, source_max_substeps,
            source_rate_cap, source_depth_step_cap, source_true_subcycling, source_imex_split,
            enable_shallow_front_recon_fallback,
            sync_diagnostics,
            diag,
            front_flux_damping,
            active_set_hysteresis);
        return;
    }

    constexpr int BLOCK = 256;
    const int32_t n_cells = dev->n_cells;
    const size_t sz = static_cast<size_t>(n_cells) * sizeof(double);
    const uint64_t graph_launches_before = dev->graph_replay_count;
    const int32_t prev_graph_integrator = dev->kernel_graph_cache.time_integrator;
    dev->kernel_graph_cache.time_integrator = 2;

    CUDA_CHECK(cudaMemcpyAsync(dev->d_h0, dev->d_h, sz, cudaMemcpyDeviceToDevice, dev->d_stream));
    CUDA_CHECK(cudaMemcpyAsync(dev->d_hu0, dev->d_hu, sz, cudaMemcpyDeviceToDevice, dev->d_stream));
    CUDA_CHECK(cudaMemcpyAsync(dev->d_hv0, dev->d_hv, sz, cudaMemcpyDeviceToDevice, dev->d_stream));

    SWE2DStepDiag tmp_diag;
    swe2d_gpu_step_persistent_chunk(
        dev,
        t_now,
        dt,
        chunk_substeps,
        g,
        h_min,
        spatial_scheme,
        cfl_factor,
        max_inv_area,
        cfl_lambda_cap,
        momentum_cap_min_speed,
        momentum_cap_celerity_mult,
        depth_cap,
        max_rel_depth_increase,
        shallow_damping_depth,
        extreme_rain_mode,
        source_cfl_beta,
        source_max_substeps,
        source_rate_cap,
        source_depth_step_cap,
        source_true_subcycling,
        source_imex_split,
        enable_shallow_front_recon_fallback,
        enable_active_edge_compaction,
        false,
        &tmp_diag,
        front_flux_damping,
        active_set_hysteresis);

    const bool has_rain_cn_state = (
        dev->d_rain_cum_mm &&
        dev->d_rain_excess_cum_mm &&
        dev->d_h1 &&
        dev->d_h2
    );
    if (has_rain_cn_state) {
        CUDA_CHECK(cudaMemcpyAsync(dev->d_h1, dev->d_rain_cum_mm, sz, cudaMemcpyDeviceToDevice, dev->d_stream));
        CUDA_CHECK(cudaMemcpyAsync(dev->d_h2, dev->d_rain_excess_cum_mm, sz, cudaMemcpyDeviceToDevice, dev->d_stream));
    }

    swe2d_gpu_step_persistent_chunk(
        dev,
        t_now + dt,
        dt,
        chunk_substeps,
        g,
        h_min,
        spatial_scheme,
        cfl_factor,
        max_inv_area,
        cfl_lambda_cap,
        momentum_cap_min_speed,
        momentum_cap_celerity_mult,
        depth_cap,
        max_rel_depth_increase,
        shallow_damping_depth,
        extreme_rain_mode,
        source_cfl_beta,
        source_max_substeps,
        source_rate_cap,
        source_depth_step_cap,
        source_true_subcycling,
        source_imex_split,
        enable_shallow_front_recon_fallback,
        enable_active_edge_compaction,
        false,
        &tmp_diag,
        front_flux_damping,
        active_set_hysteresis);

    CUDA_CHECK(cudaMemsetAsync(dev->d_max_wse_elev_error, 0, sizeof(double), dev->d_stream));
    int grid = (n_cells + BLOCK - 1) / BLOCK;
    swe2d_rk2_combine_kernel<<<grid, BLOCK, 0, dev->d_stream>>>(
        n_cells,
        dev->d_h, dev->d_hu, dev->d_hv,
        dev->d_h0, dev->d_hu0, dev->d_hv0,
        dev->d_max_wse_elev_error,
        h_min);
    CUDA_CHECK(cudaGetLastError());

    if (has_rain_cn_state) {
        CUDA_CHECK(cudaMemcpyAsync(dev->d_rain_cum_mm, dev->d_h1, sz, cudaMemcpyDeviceToDevice, dev->d_stream));
        CUDA_CHECK(cudaMemcpyAsync(dev->d_rain_excess_cum_mm, dev->d_h2, sz, cudaMemcpyDeviceToDevice, dev->d_stream));
    }

    CUDA_CHECK(cudaMemsetAsync(dev->d_lambda_max, 0, sizeof(double), dev->d_stream));
    const int edge_grid = (dev->n_edges + BLOCK - 1) / BLOCK;
    swe2d_cfl_kernel<<<edge_grid, BLOCK, BLOCK * static_cast<int>(sizeof(double)), dev->d_stream>>>(
        dev->n_edges,
        dev->d_edge_c0, dev->d_edge_c1,
        dev->d_edge_nx, dev->d_edge_ny, dev->d_edge_len,
        dev->d_h, dev->d_hu, dev->d_hv,
        dev->d_cell_area,
        g, h_min,
        cfl_lambda_cap,
        dev->d_lambda_max,
        dev->d_degen_mask, dev->degen_mode);
    CUDA_CHECK(cudaGetLastError());
    pack_diag_kernel<<<1, 1, 0, dev->d_stream>>>(dev->d_lambda_max, dev->d_max_wse_elev_error, dev->d_n_wet, dev->d_diag_packed);
    CUDA_CHECK(cudaGetLastError());

    if (diag) {
        diag->dt = dt;
        diag->gpu_active = true;
        diag->wet_cells = -1;
        diag->max_depth = -1.0;
        diag->min_depth = -1.0;
        diag->mass_total = -1.0;
        diag->max_courant = -1.0;
        diag->max_depth_residual = -1.0;
        diag->max_wse_elev_error = -1.0;
        const uint64_t graph_launches_after = dev->graph_replay_count;
        diag->gpu_graph_launches_step = static_cast<int32_t>(graph_launches_after - graph_launches_before);
        diag->gpu_graph_launches_total = static_cast<int64_t>(graph_launches_after);

        if (sync_diagnostics) {
            CUDA_CHECK(cudaStreamSynchronize(dev->d_stream));
            double packed[3] = {0.0, 0.0, -1.0};
            CUDA_CHECK(cudaMemcpy(packed, dev->d_diag_packed, 3 * sizeof(double), cudaMemcpyDeviceToHost));
            diag->max_courant        = dt * packed[0];
            diag->max_depth_residual = packed[1];
            diag->max_wse_elev_error = packed[1];
            diag->wet_cells          = static_cast<int32_t>(packed[2]);
        }
    }

    dev->kernel_graph_cache.time_integrator = prev_graph_integrator;
}


// ─────────────────────────────────────────────────────────────────────────────
// swe2d_gpu_step_rk4 — removed (dead code)
// ─────────────────────────────────────────────────────────────────────────────

// ── Forward declarations for coupling kernels defined later in this file ──
__global__ void swe2d_coupling_wse_from_state_kernel(
    int32_t n_cells, const double* d_h, const double* d_cell_zb, double* d_cell_wse);


void swe2d_gpu_step_rk5_graph(
    SWE2DDeviceState* dev,
    double t_now,
    double dt,
    double g,
    double h_min,
    int spatial_scheme,
    double cfl_factor,
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
    bool sync_diagnostics,
    SWE2DStepDiag* diag,
    double front_flux_damping,
    bool   active_set_hysteresis)
{
    if (!dev) throw std::invalid_argument("swe2d_gpu_step_rk5_graph: null device");

    constexpr int BLOCK = 256;
    const int32_t n_cells = dev->n_cells;
    const int32_t n_edges = dev->n_edges;
    const size_t sz = static_cast<size_t>(n_cells) * sizeof(double);
    const size_t sz_edges = static_cast<size_t>(n_edges) * sizeof(double);
    const uint64_t graph_launches_before = dev->graph_replay_count;
    const int32_t prev_graph_integrator = dev->kernel_graph_cache.time_integrator;
    dev->kernel_graph_cache.time_integrator = 6;
    const double stage_c[6] = {0.0, 1.0 / 5.0, 3.0 / 10.0, 3.0 / 5.0, 1.0, 7.0 / 8.0};

    CUDA_CHECK(cudaMemcpyAsync(dev->d_h0,  dev->d_h,  sz, cudaMemcpyDeviceToDevice, dev->d_stream));
    CUDA_CHECK(cudaMemcpyAsync(dev->d_hu0, dev->d_hu, sz, cudaMemcpyDeviceToDevice, dev->d_stream));
    CUDA_CHECK(cudaMemcpyAsync(dev->d_hv0, dev->d_hv, sz, cudaMemcpyDeviceToDevice, dev->d_stream));

    auto precompute_stage_forcing = [&]() {
        for (int slot = 0; slot < 6; ++slot) {
            CUDA_CHECK(cudaMemcpyAsync(swe2d_stage_edge_bc_slot(dev, slot), dev->d_edge_bc,
                                       static_cast<size_t>(n_edges) * sizeof(int32_t),
                                       cudaMemcpyDeviceToDevice, dev->d_stream));
            CUDA_CHECK(cudaMemcpyAsync(swe2d_stage_edge_bc_val_slot(dev, slot), dev->d_edge_bc_val,
                                       sz_edges, cudaMemcpyDeviceToDevice, dev->d_stream));
        }

        if (dev->n_hg_edges > 0 && dev->d_hg_edge_index && dev->d_hg_offsets &&
            dev->d_hg_time_s && dev->d_hg_value) {
            const int hg_grid = (dev->n_hg_edges + BLOCK - 1) / BLOCK;
            for (int slot = 0; slot < 6; ++slot) {
                swe2d_apply_hydrograph_bc_kernel<<<hg_grid, BLOCK, 0, dev->d_stream>>>(
                    dev->n_hg_edges,
                    dev->d_hg_edge_index,
                    dev->d_hg_bc_type,
                    dev->d_hg_offsets,
                    dev->d_hg_time_s,
                    dev->d_hg_value,
                    swe2d_stage_edge_bc_slot(dev, slot),
                    swe2d_stage_edge_bc_val_slot(dev, slot),
                    t_now + stage_c[slot] * dt);
                CUDA_CHECK(cudaGetLastError());
            }
        }

        if (dev->n_rain_samples > 0 && dev->d_cell_gage_idx && dev->d_rain_hg_offsets &&
            dev->d_rain_hg_time_s && dev->d_rain_hg_cum_mm && dev->d_rain_cn) {
            const int r_grid = (n_cells + BLOCK - 1) / BLOCK;
            for (int slot = 0; slot < 6; ++slot) {
                swe2d_eval_rain_cn_stage_rate_kernel<<<r_grid, BLOCK, 0, dev->d_stream>>>(
                    n_cells,
                    dev->d_cell_gage_idx,
                    dev->d_rain_hg_offsets,
                    dev->d_rain_hg_time_s,
                    dev->d_rain_hg_cum_mm,
                    dev->d_rain_cn,
                    dev->d_rain_cum_mm,
                    swe2d_stage_source_slot(dev, slot),
                    t_now,
                    t_now + stage_c[slot] * dt,
                    dev->rain_ia_ratio,
                    dev->rain_mm_to_model_depth);
                CUDA_CHECK(cudaGetLastError());
            }
            swe2d_stage_source_max_kernel<<<r_grid, BLOCK, 0, dev->d_stream>>>(
                n_cells, 6, dev->d_stage_cell_source_mps, dev->d_cell_source_mps);
            CUDA_CHECK(cudaGetLastError());
        } else {
            CUDA_CHECK(cudaMemsetAsync(dev->d_stage_cell_source_mps, 0,
                                       static_cast<size_t>(6) * sz, dev->d_stream));
            CUDA_CHECK(cudaMemsetAsync(dev->d_cell_source_mps, 0, sz, dev->d_stream));
        }
    };

    precompute_stage_forcing();

    // Wet/dry classification based on U^0 (rk5_graph)
    if (dev->d_active && dev->d_n_wet) {
        if (active_set_hysteresis && dev->d_was_active) {
            CUDA_CHECK(cudaMemcpyAsync(dev->d_was_active, dev->d_active,
                                       static_cast<size_t>(n_cells) * sizeof(int32_t),
                                       cudaMemcpyDeviceToDevice,
                                       dev->d_stream));
        }
        CUDA_CHECK(cudaMemsetAsync(dev->d_n_wet, 0, sizeof(int32_t), dev->d_stream));
        const int c_grid = (n_cells + BLOCK - 1) / BLOCK;
        swe2d_classify_kernel<<<c_grid, BLOCK, BLOCK * sizeof(int32_t), dev->d_stream>>>(
            n_cells, dev->d_h, dev->d_cell_source_mps, dev->d_external_source_mps,
            dev->d_ext_struct_flux_h,
            dev->d_bc_forced,
            dev->d_active, dev->d_n_wet, h_min,
            active_set_hysteresis ? dev->d_was_active : nullptr);
        CUDA_CHECK(cudaGetLastError());

        const int e_grid = (n_edges + BLOCK - 1) / BLOCK;
        swe2d_mark_neighbor_kernel<<<e_grid, BLOCK, 0, dev->d_stream>>>(
            n_edges, dev->d_edge_c0, dev->d_edge_c1, dev->d_active);
        CUDA_CHECK(cudaGetLastError());

        if ((dev->degen_mode == 1 || dev->degen_mode == 3) && dev->d_degen_mask) {
            const int c_grid2 = (n_cells + BLOCK - 1) / BLOCK;
            swe2d_degen_deactivate_kernel<<<c_grid2, BLOCK, 0, dev->d_stream>>>(
                n_cells, dev->d_degen_mask, dev->d_active);
            CUDA_CHECK(cudaGetLastError());
        }

        if (dev->degen_mode == 3 && dev->d_degen_mask && dev->d_merge_owner) {
            const int c_grid2 = (n_cells + BLOCK - 1) / BLOCK;
            swe2d_degen_sync_kernel<<<c_grid2, BLOCK, 0, dev->d_stream>>>(
                n_cells, dev->d_degen_mask, dev->d_merge_owner,
                dev->d_h, dev->d_hu, dev->d_hv);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    const bool need_gradients = (spatial_scheme >= 1);
    const int grid_c = (n_cells + BLOCK - 1) / BLOCK;
    const int grid_e = (n_edges + BLOCK - 1) / BLOCK;

    auto evaluate_rhs = [&](const int32_t* edge_bc,
                            const double* edge_bc_val,
                            const double* stage_source,
                            double* k_h,
                            double* k_hu,
                            double* k_hv) {
        if (need_gradients) {
            CUDA_CHECK(cudaMemsetAsync(dev->d_grad_hx,  0, static_cast<size_t>(n_cells) * sizeof(double), dev->d_stream));
            CUDA_CHECK(cudaMemsetAsync(dev->d_grad_hy,  0, static_cast<size_t>(n_cells) * sizeof(double), dev->d_stream));
            CUDA_CHECK(cudaMemsetAsync(dev->d_grad_hux, 0, static_cast<size_t>(n_cells) * sizeof(double), dev->d_stream));
            CUDA_CHECK(cudaMemsetAsync(dev->d_grad_huy, 0, static_cast<size_t>(n_cells) * sizeof(double), dev->d_stream));
            CUDA_CHECK(cudaMemsetAsync(dev->d_grad_hvx, 0, static_cast<size_t>(n_cells) * sizeof(double), dev->d_stream));
            CUDA_CHECK(cudaMemsetAsync(dev->d_grad_hvy, 0, static_cast<size_t>(n_cells) * sizeof(double), dev->d_stream));
            swe2d_gradient_kernel<<<grid_e, BLOCK, 0, dev->d_stream>>>(
                n_edges,
                dev->d_edge_c0, dev->d_edge_c1,
                dev->d_edge_nx, dev->d_edge_ny, dev->d_edge_len,
                dev->d_h, dev->d_cell_zb, dev->d_hu, dev->d_hv,
                dev->d_cell_inv_area,
                max_inv_area,
                dev->d_grad_hx,  dev->d_grad_hy,
                dev->d_grad_hux, dev->d_grad_huy,
                dev->d_grad_hvx, dev->d_grad_hvy,
                dev->d_active,
                dev->d_degen_mask,
                dev->degen_mode);
            CUDA_CHECK(cudaGetLastError());
            swe2d_maybe_launch_lsq_gradient(dev, spatial_scheme, n_cells, BLOCK,
                                            dev->d_h, dev->d_hu, dev->d_hv);
            CUDA_CHECK(cudaGetLastError());
        }

        CUDA_CHECK(cudaMemsetAsync(dev->d_flux_h, 0, sz_edges, dev->d_stream));
        CUDA_CHECK(cudaMemsetAsync(dev->d_flux_hu, 0, sz_edges, dev->d_stream));
        CUDA_CHECK(cudaMemsetAsync(dev->d_flux_hv, 0, sz_edges, dev->d_stream));
        CUDA_CHECK(cudaMemsetAsync(dev->d_flux_hu_r, 0, sz_edges, dev->d_stream));
        CUDA_CHECK(cudaMemsetAsync(dev->d_flux_hv_r, 0, sz_edges, dev->d_stream));

        swe2d_flux_kernel<<<grid_e, BLOCK, 0, dev->d_stream>>>(
            n_edges,
            dev->d_edge_c0, dev->d_edge_c1,
            dev->d_edge_nx, dev->d_edge_ny, dev->d_edge_len,
            dev->d_edge_mx, dev->d_edge_my,
            edge_bc, edge_bc_val,
            dev->d_h, dev->d_hu, dev->d_hv,
            dev->d_n_mann_cell, dev->d_cell_zb,
            dev->d_cell_inv_area,
            dev->d_cell_cx, dev->d_cell_cy,
            dev->d_grad_hx, dev->d_grad_hy,
            dev->d_grad_hux, dev->d_grad_huy,
            dev->d_grad_hvx, dev->d_grad_hvy,
            dev->d_flux_h, dev->d_flux_hu, dev->d_flux_hv,
            dev->d_flux_hu_r, dev->d_flux_hv_r,
            nullptr, nullptr, nullptr,
            spatial_scheme,
            g,
            h_min,
            max_inv_area,
            momentum_cap_min_speed,
            momentum_cap_celerity_mult,
            dev->d_degen_mask,
            dev->d_merge_owner,
            dev->degen_mode,
            dev->d_active,
            front_flux_damping,
            shallow_damping_depth,
            enable_shallow_front_recon_fallback);
        CUDA_CHECK(cudaGetLastError());
    };

    // ── Per-stage structure-flow recomputation ────────────────────────
    // Recomputes culvert face fluxes from the current stage state (d_h).
    // Called before each evaluate_rhs so RK stages see structure coupling.
    auto compute_coupling = [&]() {
        if (!dev->use_culvert_face_flux
            || !dev->culvert_ff_ws.params_preloaded
            || dev->culvert_ff_ws.n_culvert_faces <= 0)
            return;

        auto& sf_ws = dev->sf_ws;
        auto& ff = dev->culvert_ff_ws;
        cudaStream_t stream = dev->d_stream;

        // 1. Zero external structure flux accumulators
        CUDA_CHECK(cudaMemsetAsync(dev->d_ext_struct_flux_h,  0, static_cast<size_t>(n_cells) * sizeof(double), stream));
        CUDA_CHECK(cudaMemsetAsync(dev->d_ext_struct_flux_hu, 0, static_cast<size_t>(n_cells) * sizeof(double), stream));
        CUDA_CHECK(cudaMemsetAsync(dev->d_ext_struct_flux_hv, 0, static_cast<size_t>(n_cells) * sizeof(double), stream));

        // 2. Compute WSE from current stage state (h + zb)
        if (sf_ws.cell_capacity >= n_cells && sf_ws.d_cell_wse) {
            int grid_wse = (n_cells + BLOCK - 1) / BLOCK;
            swe2d_coupling_wse_from_state_kernel<<<grid_wse, BLOCK, 0, stream>>>(
                n_cells, dev->d_h, dev->d_cell_zb, sf_ws.d_cell_wse);
            CUDA_CHECK(cudaGetLastError());
        }

        // 3. Apply enquiry-cell WSE correction (total-energy driving head)
        if (ff.d_enquiry_up_cell && ff.d_enquiry_dn_cell
            && dev->d_hu && dev->d_hv && dev->d_h) {
            int grid_enq = (ff.n_culvert_faces + BLOCK - 1) / BLOCK;
            swe2d_apply_enquiry_wse_kernel<<<grid_enq, BLOCK, 0, stream>>>(
                ff.n_culvert_faces,
                ff.d_enquiry_up_cell,
                ff.d_enquiry_dn_cell,
                ff.d_donor_cell,
                ff.d_receiver_cell,
                sf_ws.d_cell_wse,
                dev->d_h,
                dev->d_hu,
                dev->d_hv,
                sf_ws.gravity,
                1.0e-6,
                sf_ws.d_cell_wse);
            CUDA_CHECK(cudaGetLastError());
        }

        // 4. Recompute structure flows from fresh WSE (table or secant)
        if (sf_ws.n_structures > 0) {
            if (s_culvert_solver_mode == 1 && !s_culvert_table_data) {
                s_culvert_solver_mode = 0;
            }
            CUDA_CHECK(cudaMemsetAsync(sf_ws.d_structure_flow, 0,
                                       static_cast<size_t>(sf_ws.n_structures) * sizeof(double), stream));
            int grid_sf = (sf_ws.n_structures + BLOCK - 1) / BLOCK;
            swe2d_compute_structure_flows_kernel<<<grid_sf, BLOCK, 0, stream>>>(
                n_cells, sf_ws.n_structures, sf_ws.d_cell_wse,
                sf_ws.d_structure_type, sf_ws.d_upstream_cell, sf_ws.d_downstream_cell,
                sf_ws.d_crest_elev, sf_ws.d_width, sf_ws.d_height,
                sf_ws.d_diameter, sf_ws.d_length, sf_ws.d_roughness_n,
                sf_ws.d_coeff, sf_ws.d_cd, sf_ws.d_opening,
                sf_ws.d_q_pump, sf_ws.d_max_flow,
                sf_ws.d_culvert_code, sf_ws.d_culvert_shape,
                sf_ws.d_culvert_rise, sf_ws.d_culvert_span, sf_ws.d_culvert_area,
                sf_ws.d_culvert_barrels, sf_ws.d_culvert_slope,
                sf_ws.d_inlet_invert_elev, sf_ws.d_outlet_invert_elev,
                sf_ws.d_entrance_loss_k, sf_ws.d_exit_loss_k,
                sf_ws.d_embankment_enabled, sf_ws.d_embankment_crest_elev,
                sf_ws.d_embankment_overflow_width, sf_ws.d_embankment_weir_coeff,
                sf_ws.gravity, sf_ws.model_to_ft, sf_ws.d_structure_flow,
                sf_ws.d_prev_structure_flow,
                s_culvert_solver_mode, s_culvert_table_header, s_culvert_table_data,
                s_culvert_table_n_hw, s_culvert_table_n_tw);
            CUDA_CHECK(cudaGetLastError());
        }

        // 5. Face-flux kernel: apply fresh culvert flows to cells
        {
            int grid_ff = (ff.n_culvert_faces + BLOCK - 1) / BLOCK;
            swe2d_culvert_face_flux_kernel<<<grid_ff, BLOCK, 0, stream>>>(
                ff.n_culvert_faces,
                sf_ws.d_structure_flow,
                ff.d_culvert_struct_idx,
                ff.d_face_nx, ff.d_face_ny, ff.d_face_width,
                ff.d_donor_cell, ff.d_receiver_cell,
                ff.d_invert_elev, ff.d_depth_safety,
                ff.d_donor_cell_area,
                dev->d_h, dev->d_hu, dev->d_hv, dev->d_cell_zb,
                sf_ws.gravity, dt, 1.0e-6,
                n_cells,
                dev->d_ext_struct_flux_h, dev->d_ext_struct_flux_hu, dev->d_ext_struct_flux_hv);
            CUDA_CHECK(cudaGetLastError());
        }

        // 6. Mask culvert flows so source kernel skips them
        {
            int grid_m = (ff.n_culvert_faces + BLOCK - 1) / BLOCK;
            swe2d_mask_culvert_source_kernel<<<grid_m, BLOCK, 0, stream>>>(
                ff.n_culvert_faces, ff.d_culvert_struct_idx, sf_ws.d_structure_flow);
            CUDA_CHECK(cudaGetLastError());
        }
    };

    auto run_rk5_stages_and_combine = [&]() {
        compute_coupling();
        evaluate_rhs(swe2d_stage_edge_bc_slot(dev, 0), swe2d_stage_edge_bc_val_slot(dev, 0), swe2d_stage_source_slot(dev, 0),
                     dev->d_h1, dev->d_hu1, dev->d_hv1);

        swe2d_rk_multi_stage_build_kernel<<<grid_c, BLOCK, 0, dev->d_stream>>>(
            n_cells, dev->d_h, dev->d_hu, dev->d_hv,
            dev->d_h0, dev->d_hu0, dev->d_hv0,
            dev->d_h1, dev->d_hu1, dev->d_hv1, 1.0 / 5.0,
            nullptr, nullptr, nullptr, 0.0,
            nullptr, nullptr, nullptr, 0.0,
            nullptr, nullptr, nullptr, 0.0,
            nullptr, nullptr, nullptr, 0.0,
            nullptr, nullptr, nullptr, 0.0);
        CUDA_CHECK(cudaGetLastError());
        compute_coupling();
        evaluate_rhs(swe2d_stage_edge_bc_slot(dev, 1), swe2d_stage_edge_bc_val_slot(dev, 1), swe2d_stage_source_slot(dev, 1),
                     dev->d_h2, dev->d_hu2, dev->d_hv2);

        swe2d_rk_multi_stage_build_kernel<<<grid_c, BLOCK, 0, dev->d_stream>>>(
            n_cells, dev->d_h, dev->d_hu, dev->d_hv,
            dev->d_h0, dev->d_hu0, dev->d_hv0,
            dev->d_h1, dev->d_hu1, dev->d_hv1, 3.0 / 40.0,
            dev->d_h2, dev->d_hu2, dev->d_hv2, 9.0 / 40.0,
            nullptr, nullptr, nullptr, 0.0,
            nullptr, nullptr, nullptr, 0.0,
            nullptr, nullptr, nullptr, 0.0,
            nullptr, nullptr, nullptr, 0.0);
        CUDA_CHECK(cudaGetLastError());
        compute_coupling();
        evaluate_rhs(swe2d_stage_edge_bc_slot(dev, 2), swe2d_stage_edge_bc_val_slot(dev, 2), swe2d_stage_source_slot(dev, 2),
                     dev->d_h3, dev->d_hu3, dev->d_hv3);

        swe2d_rk_multi_stage_build_kernel<<<grid_c, BLOCK, 0, dev->d_stream>>>(
            n_cells, dev->d_h, dev->d_hu, dev->d_hv,
            dev->d_h0, dev->d_hu0, dev->d_hv0,
            dev->d_h1, dev->d_hu1, dev->d_hv1, 3.0 / 10.0,
            dev->d_h2, dev->d_hu2, dev->d_hv2, -9.0 / 10.0,
            dev->d_h3, dev->d_hu3, dev->d_hv3, 6.0 / 5.0,
            nullptr, nullptr, nullptr, 0.0,
            nullptr, nullptr, nullptr, 0.0,
            nullptr, nullptr, nullptr, 0.0);
        CUDA_CHECK(cudaGetLastError());
        compute_coupling();
        evaluate_rhs(swe2d_stage_edge_bc_slot(dev, 3), swe2d_stage_edge_bc_val_slot(dev, 3), swe2d_stage_source_slot(dev, 3),
                     dev->d_k4_h, dev->d_k4_hu, dev->d_k4_hv);

        swe2d_rk_multi_stage_build_kernel<<<grid_c, BLOCK, 0, dev->d_stream>>>(
            n_cells, dev->d_h, dev->d_hu, dev->d_hv,
            dev->d_h0, dev->d_hu0, dev->d_hv0,
            dev->d_h1, dev->d_hu1, dev->d_hv1, -11.0 / 54.0,
            dev->d_h2, dev->d_hu2, dev->d_hv2, 5.0 / 2.0,
            dev->d_h3, dev->d_hu3, dev->d_hv3, -70.0 / 27.0,
            dev->d_k4_h, dev->d_k4_hu, dev->d_k4_hv, 35.0 / 27.0,
            nullptr, nullptr, nullptr, 0.0,
            nullptr, nullptr, nullptr, 0.0);
        CUDA_CHECK(cudaGetLastError());
        compute_coupling();
        evaluate_rhs(swe2d_stage_edge_bc_slot(dev, 4), swe2d_stage_edge_bc_val_slot(dev, 4), swe2d_stage_source_slot(dev, 4),
                     dev->d_k5_h, dev->d_k5_hu, dev->d_k5_hv);

        swe2d_rk_multi_stage_build_kernel<<<grid_c, BLOCK, 0, dev->d_stream>>>(
            n_cells, dev->d_h, dev->d_hu, dev->d_hv,
            dev->d_h0, dev->d_hu0, dev->d_hv0,
            dev->d_h1, dev->d_hu1, dev->d_hv1, 1631.0 / 55296.0,
            dev->d_h2, dev->d_hu2, dev->d_hv2, 175.0 / 512.0,
            dev->d_h3, dev->d_hu3, dev->d_hv3, 575.0 / 13824.0,
            dev->d_k4_h, dev->d_k4_hu, dev->d_k4_hv, 44275.0 / 110592.0,
            dev->d_k5_h, dev->d_k5_hu, dev->d_k5_hv, 253.0 / 4096.0,
            nullptr, nullptr, nullptr, 0.0);
        CUDA_CHECK(cudaGetLastError());
        compute_coupling();
        evaluate_rhs(swe2d_stage_edge_bc_slot(dev, 5), swe2d_stage_edge_bc_val_slot(dev, 5), swe2d_stage_source_slot(dev, 5),
                     dev->d_k6_h, dev->d_k6_hu, dev->d_k6_hv);

        CUDA_CHECK(cudaMemsetAsync(dev->d_max_wse_elev_error, 0, sizeof(double), dev->d_stream));
        swe2d_rk5_graph_combine_kernel<<<grid_c, BLOCK, 0, dev->d_stream>>>(
            n_cells,
            dev->d_h, dev->d_hu, dev->d_hv,
            dev->d_h0, dev->d_hu0, dev->d_hv0,
            dev->d_h1, dev->d_hu1, dev->d_hv1,
            dev->d_h3, dev->d_hu3, dev->d_hv3,
            dev->d_k4_h, dev->d_k4_hu, dev->d_k4_hv,
            dev->d_k6_h, dev->d_k6_hu, dev->d_k6_hv,
            dev->d_max_wse_elev_error,
            dev->d_n_mann_cell,
            g, h_min, shallow_damping_depth, dt,
            momentum_cap_min_speed,
            momentum_cap_celerity_mult);
        CUDA_CHECK(cudaGetLastError());
    };

    const bool dbg_edge_flux = swe2d_debug_enabled("BACKWATER_SWE2D_DEBUG_GPU_EDGE_FLUX");
    const bool dbg_flux_summary = swe2d_debug_enabled("BACKWATER_SWE2D_DEBUG_GPU_FLUX");
    const bool try_kernel_graph = dev->enable_kernel_graphs && !dbg_edge_flux && !dbg_flux_summary;
    const uint64_t graph_signature = swe2d_kernel_graph_signature(
        dt, g, h_min, cfl_lambda_cap, max_inv_area,
        momentum_cap_min_speed, momentum_cap_celerity_mult,
        depth_cap, max_rel_depth_increase, shallow_damping_depth,
        extreme_rain_mode, source_cfl_beta, source_max_substeps,
        source_rate_cap, source_depth_step_cap,
        source_true_subcycling, source_imex_split,
        enable_shallow_front_recon_fallback,
        front_flux_damping, dev->use_culvert_face_flux);

    bool used_graph_replay = false;
    if (try_kernel_graph) {
        auto& cache = dev->kernel_graph_cache;
        const bool cache_match =
            cache.is_valid && cache.exec != nullptr &&
            cache.n_cells == n_cells && cache.n_edges == n_edges &&
            cache.spatial_scheme == spatial_scheme &&
            cache.time_integrator == 6 &&
            cache.config_signature == graph_signature;
        if (cache_match) {
            CUDA_CHECK(cudaGraphLaunch(cache.exec, dev->d_stream));
            dev->graph_replay_count += 1;
            used_graph_replay = true;
        } else {
            cache.destroy();
            CUDA_CHECK(cudaStreamSynchronize(dev->d_stream));
            cudaGraph_t graph = nullptr;
            cudaError_t cap_begin = cudaStreamBeginCapture(dev->d_stream, cudaStreamCaptureModeThreadLocal);
            if (cap_begin == cudaSuccess) {
                run_rk5_stages_and_combine();
                const cudaError_t cap_end = cudaStreamEndCapture(dev->d_stream, &graph);
                if (cap_end == cudaSuccess && graph != nullptr) {
                    cudaGraphExec_t exec = nullptr;
                    if (cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) == cudaSuccess) {
                        cache.graph = graph;
                        cache.exec = exec;
                        cache.n_cells = n_cells;
                        cache.n_edges = n_edges;
                        cache.spatial_scheme = spatial_scheme;
                        cache.time_integrator = 6;
                        cache.config_signature = graph_signature;
                        cache.is_valid = true;
                        CUDA_CHECK(cudaGraphLaunch(cache.exec, dev->d_stream));
                        dev->graph_replay_count += 1;
                        used_graph_replay = true;
                    } else {
                        if (exec) cudaGraphExecDestroy(exec);
                        cudaGraphDestroy(graph);
                    }
                } else if (graph != nullptr) {
                    cudaGraphDestroy(graph);
                }
            }
        }
    }

    if (!used_graph_replay) {
        run_rk5_stages_and_combine();
    }

    if (dev->n_rain_samples > 0 && dev->d_cell_gage_idx && dev->d_rain_hg_offsets &&
        dev->d_rain_hg_time_s && dev->d_rain_hg_cum_mm && dev->d_rain_cn) {
        const int r_grid = (n_cells + BLOCK - 1) / BLOCK;
        swe2d_build_rain_cn_source_kernel<<<r_grid, BLOCK, 0, dev->d_stream>>>(
            n_cells,
            dev->d_cell_gage_idx,
            dev->d_rain_hg_offsets,
            dev->d_rain_hg_time_s,
            dev->d_rain_hg_cum_mm,
            dev->d_rain_cn,
            dev->d_rain_cum_mm,
            dev->d_rain_excess_cum_mm,
            dev->d_cell_source_mps,
            t_now,
            t_now + dt,
            dev->rain_ia_ratio,
            dev->rain_mm_to_model_depth);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaMemsetAsync(dev->d_lambda_max, 0, sizeof(double), dev->d_stream));
    const int edge_grid = (n_edges + BLOCK - 1) / BLOCK;
    swe2d_cfl_kernel<<<edge_grid, BLOCK, BLOCK * static_cast<int>(sizeof(double)), dev->d_stream>>>(
        n_edges,
        dev->d_edge_c0, dev->d_edge_c1,
        dev->d_edge_nx, dev->d_edge_ny, dev->d_edge_len,
        dev->d_h, dev->d_hu, dev->d_hv,
        dev->d_cell_area,
        g, h_min,
        cfl_lambda_cap,
        dev->d_lambda_max,
        dev->d_degen_mask, dev->degen_mode);
    CUDA_CHECK(cudaGetLastError());

    pack_diag_kernel<<<1, 1, 0, dev->d_stream>>>(dev->d_lambda_max, dev->d_max_wse_elev_error, dev->d_n_wet, dev->d_diag_packed);
    CUDA_CHECK(cudaGetLastError());

    if (diag) {
        diag->dt = dt;
        diag->gpu_active = true;
        diag->wet_cells = -1;
        diag->max_depth = -1.0;
        diag->min_depth = -1.0;
        diag->mass_total = -1.0;
        diag->max_courant = -1.0;
        diag->max_depth_residual = -1.0;
        diag->max_wse_elev_error = -1.0;
        const uint64_t graph_launches_after = dev->graph_replay_count;
        diag->gpu_graph_launches_step = static_cast<int32_t>(graph_launches_after - graph_launches_before);
        diag->gpu_graph_launches_total = static_cast<int64_t>(graph_launches_after);
        if (sync_diagnostics) {
            CUDA_CHECK(cudaStreamSynchronize(dev->d_stream));
            double packed[3] = {0.0, 0.0, -1.0};
            CUDA_CHECK(cudaMemcpy(packed, dev->d_diag_packed, 3 * sizeof(double), cudaMemcpyDeviceToHost));
            diag->max_courant        = dt * packed[0];
            diag->max_depth_residual = packed[1];
            diag->max_wse_elev_error = packed[1];
            diag->wet_cells          = static_cast<int32_t>(packed[2]);
        }
    }

    dev->kernel_graph_cache.time_integrator = prev_graph_integrator;
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

void swe2d_gpu_readback_h(double* host_buf, int32_t n_cells)
{
    SWE2DDeviceState* dev = s_coupling_dev;
    if (!dev || !dev->d_h || n_cells <= 0) {
        if (host_buf && n_cells > 0)
            std::memset(host_buf, 0, static_cast<size_t>(n_cells) * sizeof(double));
        return;
    }
    if (n_cells > dev->n_cells)
        n_cells = dev->n_cells;
    CUDA_CHECK(cudaMemcpy(host_buf, dev->d_h,
                          static_cast<size_t>(n_cells) * sizeof(double),
                          cudaMemcpyDeviceToHost));
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

void swe2d_gpu_update_boundary_values(
    SWE2DDeviceState* dev,
    const int32_t* edge_index,
    const int32_t* bc_type,
    const double* bc_val,
    int32_t n_updates)
{
    if (!dev || n_updates <= 0 || !edge_index || !bc_type || !bc_val) return;
    if (dev->bc_upd_capacity < n_updates) {
        if (dev->d_bc_upd_edge) CUDA_CHECK(cudaFree(dev->d_bc_upd_edge));
        if (dev->d_bc_upd_type) CUDA_CHECK(cudaFree(dev->d_bc_upd_type));
        if (dev->d_bc_upd_val) CUDA_CHECK(cudaFree(dev->d_bc_upd_val));
        dev->d_bc_upd_edge = nullptr;
        dev->d_bc_upd_type = nullptr;
        dev->d_bc_upd_val = nullptr;

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev->d_bc_upd_edge), static_cast<size_t>(n_updates) * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev->d_bc_upd_type), static_cast<size_t>(n_updates) * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev->d_bc_upd_val), static_cast<size_t>(n_updates) * sizeof(double)));
        dev->bc_upd_capacity = n_updates;
    }

    CUDA_CHECK(cudaMemcpy(dev->d_bc_upd_edge, edge_index, static_cast<size_t>(n_updates) * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev->d_bc_upd_type, bc_type, static_cast<size_t>(n_updates) * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev->d_bc_upd_val, bc_val, static_cast<size_t>(n_updates) * sizeof(double), cudaMemcpyHostToDevice));
    constexpr int BLOCK = 256;
    const int grid = (n_updates + BLOCK - 1) / BLOCK;
    swe2d_apply_boundary_updates_kernel<<<grid, BLOCK, 0, dev->d_stream>>>(
        n_updates, dev->d_bc_upd_edge, dev->d_bc_upd_type, dev->d_bc_upd_val, dev->d_edge_bc, dev->d_edge_bc_val);
    CUDA_CHECK(cudaGetLastError());
}

void swe2d_gpu_set_boundary_hydrographs(
    SWE2DDeviceState* dev,
    const int32_t* edge_index,
    const int32_t* bc_type,
    const int32_t* offsets,
    const double* time_s,
    const double* value,
    int32_t n_edges,
    int32_t n_samples)
{
    if (!dev) return;
    if (dev->d_hg_edge_index) CUDA_CHECK(cudaFree(dev->d_hg_edge_index));
    if (dev->d_hg_bc_type) CUDA_CHECK(cudaFree(dev->d_hg_bc_type));
    if (dev->d_hg_offsets) CUDA_CHECK(cudaFree(dev->d_hg_offsets));
    if (dev->d_hg_time_s) CUDA_CHECK(cudaFree(dev->d_hg_time_s));
    if (dev->d_hg_value) CUDA_CHECK(cudaFree(dev->d_hg_value));
    dev->d_hg_edge_index = nullptr;
    dev->d_hg_bc_type = nullptr;
    dev->d_hg_offsets = nullptr;
    dev->d_hg_time_s = nullptr;
    dev->d_hg_value = nullptr;
    dev->n_hg_edges = 0;
    dev->n_hg_samples = 0;

    if (n_edges <= 0 || n_samples <= 0 || !edge_index || !bc_type || !offsets || !time_s || !value) return;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev->d_hg_edge_index), static_cast<size_t>(n_edges) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev->d_hg_bc_type), static_cast<size_t>(n_edges) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev->d_hg_offsets), static_cast<size_t>(n_edges + 1) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev->d_hg_time_s), static_cast<size_t>(n_samples) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev->d_hg_value), static_cast<size_t>(n_samples) * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dev->d_hg_edge_index, edge_index, static_cast<size_t>(n_edges) * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev->d_hg_bc_type, bc_type, static_cast<size_t>(n_edges) * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev->d_hg_offsets, offsets, static_cast<size_t>(n_edges + 1) * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev->d_hg_time_s, time_s, static_cast<size_t>(n_samples) * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev->d_hg_value, value, static_cast<size_t>(n_samples) * sizeof(double), cudaMemcpyHostToDevice));
    dev->n_hg_edges = n_edges;
    dev->n_hg_samples = n_samples;
}

void swe2d_gpu_set_rain_cn_forcing(
    SWE2DDeviceState* dev,
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
    if (!dev) return;
    if (dev->d_cell_gage_idx) CUDA_CHECK(cudaFree(dev->d_cell_gage_idx));
    if (dev->d_rain_hg_offsets) CUDA_CHECK(cudaFree(dev->d_rain_hg_offsets));
    if (dev->d_rain_hg_time_s) CUDA_CHECK(cudaFree(dev->d_rain_hg_time_s));
    if (dev->d_rain_hg_cum_mm) CUDA_CHECK(cudaFree(dev->d_rain_hg_cum_mm));
    if (dev->d_rain_cn) CUDA_CHECK(cudaFree(dev->d_rain_cn));
    if (dev->d_rain_cum_mm) CUDA_CHECK(cudaFree(dev->d_rain_cum_mm));
    if (dev->d_rain_excess_cum_mm) CUDA_CHECK(cudaFree(dev->d_rain_excess_cum_mm));
    dev->d_cell_gage_idx = nullptr;
    dev->d_rain_hg_offsets = nullptr;
    dev->d_rain_hg_time_s = nullptr;
    dev->d_rain_hg_cum_mm = nullptr;
    dev->d_rain_cn = nullptr;
    dev->d_rain_cum_mm = nullptr;
    dev->d_rain_excess_cum_mm = nullptr;
    dev->n_rain_gages = 0;
    dev->n_rain_samples = 0;
    dev->rain_ia_ratio = ia_ratio;
    dev->rain_mm_to_model_depth = (mm_to_model_depth > 0.0) ? mm_to_model_depth : 1.0e-3;

    if (!dev->d_cell_source_mps && dev->n_cells > 0) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev->d_cell_source_mps), static_cast<size_t>(dev->n_cells) * sizeof(double)));
    }
    if (dev->d_cell_source_mps && dev->n_cells > 0) {
        CUDA_CHECK(cudaMemset(dev->d_cell_source_mps, 0, static_cast<size_t>(dev->n_cells) * sizeof(double)));
    }

    if (n_cells <= 0 || n_gages <= 0 || n_samples <= 0 || !cell_gage_idx || !gage_offsets || !hg_time_s || !hg_cum_mm || !cn) return;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev->d_cell_gage_idx), static_cast<size_t>(n_cells) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev->d_rain_hg_offsets), static_cast<size_t>(n_gages + 1) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev->d_rain_hg_time_s), static_cast<size_t>(n_samples) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev->d_rain_hg_cum_mm), static_cast<size_t>(n_samples) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev->d_rain_cn), static_cast<size_t>(n_cells) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev->d_rain_cum_mm), static_cast<size_t>(n_cells) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev->d_rain_excess_cum_mm), static_cast<size_t>(n_cells) * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dev->d_cell_gage_idx, cell_gage_idx, static_cast<size_t>(n_cells) * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev->d_rain_hg_offsets, gage_offsets, static_cast<size_t>(n_gages + 1) * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev->d_rain_hg_time_s, hg_time_s, static_cast<size_t>(n_samples) * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev->d_rain_hg_cum_mm, hg_cum_mm, static_cast<size_t>(n_samples) * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev->d_rain_cn, cn, static_cast<size_t>(n_cells) * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dev->d_rain_cum_mm, 0, static_cast<size_t>(n_cells) * sizeof(double)));
    CUDA_CHECK(cudaMemset(dev->d_rain_excess_cum_mm, 0, static_cast<size_t>(n_cells) * sizeof(double)));
    CUDA_CHECK(cudaMemset(dev->d_cell_source_mps, 0, static_cast<size_t>(n_cells) * sizeof(double)));
    dev->n_rain_gages = n_gages;
    dev->n_rain_samples = n_samples;
}

void swe2d_gpu_set_external_sources(
    SWE2DDeviceState* dev,
    const double* source_mps,
    int32_t n_cells)
{
    if (!dev) return;
    if (!dev->d_external_source_mps || n_cells <= 0 || n_cells != dev->n_cells) {
        return;
    }

    if (source_mps) {
        CUDA_CHECK(cudaMemcpyAsync(
            dev->d_external_source_mps,
            source_mps,
            static_cast<size_t>(n_cells) * sizeof(double),
            cudaMemcpyHostToDevice,
            dev->d_stream));
    } else {
        CUDA_CHECK(cudaMemsetAsync(
            dev->d_external_source_mps,
            0,
            static_cast<size_t>(n_cells) * sizeof(double),
            dev->d_stream));
    }
}

// ── Persistent coupling globals ──
SWE2DDeviceState* s_coupling_dev = nullptr;

// ── Coupling dt for face-flux depth limiter ──
// Set by swe2d_gpu_set_coupling_dt before the full_on_device call;
// consumed by swe2d_culvert_face_flux_kernel for depth limiting.
static double s_coupling_dt = 0.0;

void swe2d_gpu_set_coupling_dt(double dt)
{
    s_coupling_dt = dt;
}

void swe2d_gpu_set_coupling_device_global(SWE2DDeviceState* dev) {
    s_coupling_dev = dev;
}

// ── Fused structure-flows + coupling-sources ──
void swe2d_gpu_compute_structure_and_coupling_sources(
    int32_t n_cells, const double* cell_area,
    int32_t n_structures,
    const double* cell_wse, const double* cell_bed,
    const int32_t* structure_type, const int32_t* upstream_cell, const int32_t* downstream_cell,
    const double* crest_elev, const double* width, const double* height,
    const double* diameter, const double* length, const double* roughness_n,
    const double* coeff, const double* cd, const double* opening,
    const double* q_pump, const double* max_flow,
    const int32_t* culvert_code, const int32_t* culvert_shape,
    const double* culvert_rise, const double* culvert_span, const double* culvert_area,
    const double* culvert_barrels, const double* culvert_slope,
    const double* inlet_invert_elev, const double* outlet_invert_elev,
    const double* entrance_loss_k, const double* exit_loss_k,
    const int32_t* embankment_enabled, const double* embankment_crest_elev,
    const double* embankment_overflow_width, const double* embankment_weir_coeff,
    double gravity,
    double model_to_ft,
    int32_t n_inlets, const int32_t* inlet_cell, const double* inlet_flow_cms,
    double* source_rate_out)
{
    if (!source_rate_out || n_cells <= 0) return;
    std::fill(source_rate_out, source_rate_out + static_cast<size_t>(n_cells), 0.0);
    if (n_structures > 0) {
        std::vector<double> sf(static_cast<size_t>(n_structures), 0.0);
        swe2d_gpu_compute_structure_flows(
            n_cells, n_structures, cell_wse, cell_bed,
            structure_type, upstream_cell, downstream_cell,
            crest_elev, width, height, diameter, length, roughness_n,
            coeff, cd, opening, q_pump, max_flow,
            culvert_code, culvert_shape, culvert_rise, culvert_span, culvert_area,
            culvert_barrels, culvert_slope, inlet_invert_elev, outlet_invert_elev,
            entrance_loss_k, exit_loss_k, embankment_enabled, embankment_crest_elev,
            embankment_overflow_width, embankment_weir_coeff, gravity, model_to_ft, sf.data());
        swe2d_gpu_compute_coupling_sources(
            nullptr, n_cells, cell_area, n_inlets, inlet_cell, inlet_flow_cms,
            n_structures, upstream_cell, downstream_cell, sf.data(), source_rate_out);
    }
}

// ── Persistent on-device coupling: preload and run ──
namespace {
template <typename T>
void sf_ensure_buf(T*& ptr, int32_t& cap, int32_t need) {
    if (!ptr || cap < need) {
        if (ptr) cudaFree(ptr);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), static_cast<size_t>(need) * sizeof(T)));
        cap = need;
    }
}
template <typename T>
void sf_upload_buf(const T* src, T* dst, int32_t n, cudaStream_t stream) {
    if (n > 0 && src && dst) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, static_cast<size_t>(n) * sizeof(T), cudaMemcpyHostToDevice, stream));
    }
}
}

void swe2d_gpu_preload_structure_params(
    SWE2DDeviceState* dev, int32_t n_structures,
    const int32_t* structure_type, const int32_t* upstream_cell, const int32_t* downstream_cell,
    const double* crest_elev, const double* width, const double* height,
    const double* diameter, const double* length, const double* roughness_n,
    const double* coeff, const double* cd, const double* opening,
    const double* q_pump, const double* max_flow,
    const int32_t* culvert_code, const int32_t* culvert_shape,
    const double* culvert_rise, const double* culvert_span, const double* culvert_area,
    const double* culvert_barrels, const double* culvert_slope,
    const double* inlet_invert_elev, const double* outlet_invert_elev,
    const double* entrance_loss_k, const double* exit_loss_k,
    const int32_t* embankment_enabled, const double* embankment_crest_elev,
    const double* embankment_overflow_width, const double* embankment_weir_coeff,
    double gravity, double model_to_ft)
{
    if (!dev) dev = s_coupling_dev;
    if (!dev) {
        if (n_structures <= 0) return;
        throw std::runtime_error("preload_structure_params: no GPU device state");
    }
    if (n_structures <= 0) return;
    auto& ws = dev->sf_ws;
    if (ws.params_preloaded && ws.n_structures == n_structures && ws.gravity == gravity && ws.model_to_ft == model_to_ft) return;
    cudaStream_t stream = dev->d_stream;

    sf_ensure_buf(ws.d_cell_wse, ws.cell_capacity, dev->n_cells);
    sf_ensure_buf(ws.d_structure_type, ws.struct_capacity, n_structures);
    sf_ensure_buf(ws.d_upstream_cell, ws.struct_capacity, n_structures);
    sf_ensure_buf(ws.d_downstream_cell, ws.struct_capacity, n_structures);
    sf_ensure_buf(ws.d_crest_elev, ws.struct_capacity, n_structures);
    sf_ensure_buf(ws.d_width, ws.struct_capacity, n_structures);
    sf_ensure_buf(ws.d_height, ws.struct_capacity, n_structures);
    sf_ensure_buf(ws.d_diameter, ws.struct_capacity, n_structures);
    sf_ensure_buf(ws.d_length, ws.struct_capacity, n_structures);
    sf_ensure_buf(ws.d_roughness_n, ws.struct_capacity, n_structures);
    sf_ensure_buf(ws.d_coeff, ws.struct_capacity, n_structures);
    sf_ensure_buf(ws.d_cd, ws.struct_capacity, n_structures);
    sf_ensure_buf(ws.d_opening, ws.struct_capacity, n_structures);
    sf_ensure_buf(ws.d_q_pump, ws.struct_capacity, n_structures);
    sf_ensure_buf(ws.d_max_flow, ws.struct_capacity, n_structures);
    sf_ensure_buf(ws.d_culvert_code, ws.struct_capacity, n_structures);
    sf_ensure_buf(ws.d_culvert_shape, ws.struct_capacity, n_structures);
    sf_ensure_buf(ws.d_culvert_rise, ws.struct_capacity, n_structures);
    sf_ensure_buf(ws.d_culvert_span, ws.struct_capacity, n_structures);
    sf_ensure_buf(ws.d_culvert_area, ws.struct_capacity, n_structures);
    sf_ensure_buf(ws.d_culvert_barrels, ws.struct_capacity, n_structures);
    sf_ensure_buf(ws.d_culvert_slope, ws.struct_capacity, n_structures);
    sf_ensure_buf(ws.d_inlet_invert_elev, ws.struct_capacity, n_structures);
    sf_ensure_buf(ws.d_outlet_invert_elev, ws.struct_capacity, n_structures);
    sf_ensure_buf(ws.d_entrance_loss_k, ws.struct_capacity, n_structures);
    sf_ensure_buf(ws.d_exit_loss_k, ws.struct_capacity, n_structures);
    sf_ensure_buf(ws.d_embankment_enabled, ws.struct_capacity, n_structures);
    sf_ensure_buf(ws.d_embankment_crest_elev, ws.struct_capacity, n_structures);
    sf_ensure_buf(ws.d_embankment_overflow_width, ws.struct_capacity, n_structures);
    sf_ensure_buf(ws.d_embankment_weir_coeff, ws.struct_capacity, n_structures);
    sf_ensure_buf(ws.d_structure_flow, ws.struct_capacity, n_structures);
    sf_ensure_buf(ws.d_prev_structure_flow, ws.struct_capacity, n_structures);

    sf_upload_buf(structure_type, ws.d_structure_type, n_structures, stream);
    sf_upload_buf(upstream_cell, ws.d_upstream_cell, n_structures, stream);
    sf_upload_buf(downstream_cell, ws.d_downstream_cell, n_structures, stream);
    sf_upload_buf(crest_elev, ws.d_crest_elev, n_structures, stream);
    sf_upload_buf(width, ws.d_width, n_structures, stream);
    sf_upload_buf(height, ws.d_height, n_structures, stream);
    sf_upload_buf(diameter, ws.d_diameter, n_structures, stream);
    sf_upload_buf(length, ws.d_length, n_structures, stream);
    sf_upload_buf(roughness_n, ws.d_roughness_n, n_structures, stream);
    sf_upload_buf(coeff, ws.d_coeff, n_structures, stream);
    sf_upload_buf(cd, ws.d_cd, n_structures, stream);
    sf_upload_buf(opening, ws.d_opening, n_structures, stream);
    sf_upload_buf(q_pump, ws.d_q_pump, n_structures, stream);
    sf_upload_buf(max_flow, ws.d_max_flow, n_structures, stream);
    sf_upload_buf(culvert_code, ws.d_culvert_code, n_structures, stream);
    sf_upload_buf(culvert_shape, ws.d_culvert_shape, n_structures, stream);
    sf_upload_buf(culvert_rise, ws.d_culvert_rise, n_structures, stream);
    sf_upload_buf(culvert_span, ws.d_culvert_span, n_structures, stream);
    sf_upload_buf(culvert_area, ws.d_culvert_area, n_structures, stream);
    sf_upload_buf(culvert_barrels, ws.d_culvert_barrels, n_structures, stream);
    sf_upload_buf(culvert_slope, ws.d_culvert_slope, n_structures, stream);
    sf_upload_buf(inlet_invert_elev, ws.d_inlet_invert_elev, n_structures, stream);
    sf_upload_buf(outlet_invert_elev, ws.d_outlet_invert_elev, n_structures, stream);
    sf_upload_buf(entrance_loss_k, ws.d_entrance_loss_k, n_structures, stream);
    sf_upload_buf(exit_loss_k, ws.d_exit_loss_k, n_structures, stream);
    sf_upload_buf(embankment_enabled, ws.d_embankment_enabled, n_structures, stream);
    sf_upload_buf(embankment_crest_elev, ws.d_embankment_crest_elev, n_structures, stream);
    sf_upload_buf(embankment_overflow_width, ws.d_embankment_overflow_width, n_structures, stream);
    sf_upload_buf(embankment_weir_coeff, ws.d_embankment_weir_coeff, n_structures, stream);

    ws.n_structures = n_structures;
    ws.gravity = gravity;
    ws.model_to_ft = model_to_ft;
    ws.params_preloaded = true;
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void swe2d_gpu_preload_coupling_cell_area(SWE2DDeviceState* dev, int32_t n_cells, const double* cell_area)
{
    if (!dev) dev = s_coupling_dev;
    if (!dev || n_cells <= 0 || !cell_area) {
        if (!dev && cell_area) throw std::runtime_error("preload_coupling_cell_area: no GPU device state");
        return;
    }
    auto& ws = dev->coupling_ws;
    if (ws.cell_capacity < n_cells) {
        if (ws.d_cell_area) cudaFree(ws.d_cell_area);
        if (ws.d_source) cudaFree(ws.d_source);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ws.d_cell_area), static_cast<size_t>(n_cells) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ws.d_source), static_cast<size_t>(n_cells) * sizeof(double)));
        ws.cell_capacity = n_cells;
    }
    CUDA_CHECK(cudaMemcpyAsync(ws.d_cell_area, cell_area, static_cast<size_t>(n_cells) * sizeof(double),
                               cudaMemcpyHostToDevice, dev->d_stream));
    CUDA_CHECK(cudaStreamSynchronize(dev->d_stream));
}

__global__ void swe2d_coupling_wse_from_state_kernel(
    int32_t n_cells,
    const double* __restrict__ d_h,
    const double* __restrict__ d_cell_zb,
    double* __restrict__ d_cell_wse)
{
    int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells) return;
    const double h = d_h ? d_h[c] : 0.0;
    const double zb = d_cell_zb ? d_cell_zb[c] : 0.0;
    d_cell_wse[c] = h + zb;
}

void swe2d_gpu_compute_coupling_full_on_device(
    SWE2DDeviceState* dev, int32_t n_cells, int32_t n_structures, const double* cell_wse_host,
    int32_t n_inlets, const int32_t* inlet_cell, const double* inlet_flow_cms,
    const double* host_structure_flows)  // if non-null, override computed structure flows
{
    if (!dev) dev = s_coupling_dev;
    if (!dev) throw std::runtime_error("compute_coupling_full_on_device: no GPU device state");
    if (n_cells <= 0) n_cells = dev->n_cells;
    if (n_cells <= 0 || !dev->d_external_source_mps) return;
    if (n_cells != dev->n_cells) {
        throw std::runtime_error("compute_coupling_full_on_device: n_cells mismatch");
    }
    auto& sf_ws = dev->sf_ws;
    auto& cpl_ws = dev->coupling_ws;
    cudaStream_t stream = dev->d_stream;
    constexpr int BLOCK = 256;

    CUDA_CHECK(cudaMemsetAsync(dev->d_external_source_mps, 0, static_cast<size_t>(n_cells) * sizeof(double), stream));

    if (sf_ws.cell_capacity >= n_cells && sf_ws.d_cell_wse) {
        if (cell_wse_host) {
            CUDA_CHECK(cudaMemcpyAsync(sf_ws.d_cell_wse, cell_wse_host, static_cast<size_t>(n_cells) * sizeof(double),
                                       cudaMemcpyHostToDevice, stream));
        } else if (dev->d_h && dev->d_cell_zb) {
            const int grid_cells = (n_cells + BLOCK - 1) / BLOCK;
            swe2d_coupling_wse_from_state_kernel<<<grid_cells, BLOCK, 0, stream>>>(
                n_cells, dev->d_h, dev->d_cell_zb, sf_ws.d_cell_wse);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    if (n_structures > 0 && sf_ws.params_preloaded) {
        // Save previous-step flows as secant-solver hint before zeroing.
        if (sf_ws.d_prev_structure_flow) {
            CUDA_CHECK(cudaMemcpyAsync(sf_ws.d_prev_structure_flow, sf_ws.d_structure_flow,
                                       static_cast<size_t>(n_structures) * sizeof(double),
                                       cudaMemcpyDeviceToDevice, stream));
        }
        CUDA_CHECK(cudaMemsetAsync(sf_ws.d_structure_flow, 0, static_cast<size_t>(n_structures) * sizeof(double), stream));
        if (host_structure_flows) {
            // Use host-provided flows (pre-computed in Python) — avoids
            // the GPU culvert solver which may return incorrect Q_c.
            CUDA_CHECK(cudaMemcpyAsync(sf_ws.d_structure_flow, host_structure_flows,
                                      static_cast<size_t>(n_structures) * sizeof(double),
                                      cudaMemcpyHostToDevice, stream));
        } else {
            int grid = (n_structures + BLOCK - 1) / BLOCK;
            swe2d_compute_structure_flows_kernel<<<grid, BLOCK, 0, stream>>>(
                n_cells, n_structures, sf_ws.d_cell_wse,
                sf_ws.d_structure_type, sf_ws.d_upstream_cell, sf_ws.d_downstream_cell,
                sf_ws.d_crest_elev, sf_ws.d_width, sf_ws.d_height,
                sf_ws.d_diameter, sf_ws.d_length, sf_ws.d_roughness_n,
                sf_ws.d_coeff, sf_ws.d_cd, sf_ws.d_opening,
                sf_ws.d_q_pump, sf_ws.d_max_flow,
                sf_ws.d_culvert_code, sf_ws.d_culvert_shape,
                sf_ws.d_culvert_rise, sf_ws.d_culvert_span, sf_ws.d_culvert_area,
                sf_ws.d_culvert_barrels, sf_ws.d_culvert_slope,
                sf_ws.d_inlet_invert_elev, sf_ws.d_outlet_invert_elev,
                sf_ws.d_entrance_loss_k, sf_ws.d_exit_loss_k,
                sf_ws.d_embankment_enabled, sf_ws.d_embankment_crest_elev,
                sf_ws.d_embankment_overflow_width, sf_ws.d_embankment_weir_coeff,
                sf_ws.gravity, sf_ws.model_to_ft, sf_ws.d_structure_flow,
                sf_ws.d_prev_structure_flow,
                s_culvert_solver_mode, s_culvert_table_header, s_culvert_table_data,
                s_culvert_table_n_hw, s_culvert_table_n_tw);
            CUDA_CHECK(cudaGetLastError());
        }

        // ── Face-based culvert flux: apply before source kernel ─────────
        // When face-flux mode is active, apply culvert face fluxes FIRST
        // and zero out culvert flows from d_structure_flow.  This way the
        // source kernel below only applies non-culvert structures (weirs,
        // orifices, pumps, bridges).
        if (dev->use_culvert_face_flux
            && dev->culvert_ff_ws.params_preloaded
            && dev->culvert_ff_ws.n_culvert_faces > 0) {
            auto& ff = dev->culvert_ff_ws;
            // Ensure external flux accumulators exist and zero them
            swe2d_gpu_alloc_ext_struct_flux(dev, n_cells);
            CUDA_CHECK(cudaMemsetAsync(dev->d_ext_struct_flux_h,  0, static_cast<size_t>(n_cells) * sizeof(double), stream));
            CUDA_CHECK(cudaMemsetAsync(dev->d_ext_struct_flux_hu, 0, static_cast<size_t>(n_cells) * sizeof(double), stream));
            CUDA_CHECK(cudaMemsetAsync(dev->d_ext_struct_flux_hv, 0, static_cast<size_t>(n_cells) * sizeof(double), stream));

            // Recompute WSE from current device state (h + zb) so structure
            // flows reflect the latest water levels, not stale host uploads.
            if (sf_ws.cell_capacity >= n_cells && sf_ws.d_cell_wse) {
                int grid_wse = (n_cells + BLOCK - 1) / BLOCK;
                swe2d_coupling_wse_from_state_kernel<<<grid_wse, BLOCK, 0, stream>>>(
                    n_cells, dev->d_h, dev->d_cell_zb, sf_ws.d_cell_wse);
                CUDA_CHECK(cudaGetLastError());
            }

            // ── Apply enquiry-cell WSE correction for total-energy driving head ──
            // Overwrite the WSE at face cells with the total energy (WSE + v²/2g)
            // sampled at offset enquiry cells.  This avoids the local drawdown
            // singularity that occurs when sampling directly at the face.
            if (ff.d_enquiry_up_cell && ff.d_enquiry_dn_cell
                && dev->d_hu && dev->d_hv && dev->d_h) {
                int grid_enq = (ff.n_culvert_faces + BLOCK - 1) / BLOCK;
                swe2d_apply_enquiry_wse_kernel<<<grid_enq, BLOCK, 0, stream>>>(
                    ff.n_culvert_faces,
                    ff.d_enquiry_up_cell,
                    ff.d_enquiry_dn_cell,
                    ff.d_donor_cell,
                    ff.d_receiver_cell,
                    sf_ws.d_cell_wse,
                    dev->d_h,
                    dev->d_hu,
                    dev->d_hv,
                    sf_ws.gravity,
                    1.0e-6,
                    sf_ws.d_cell_wse);
                CUDA_CHECK(cudaGetLastError());
            }

            // Recompute structure flows from fresh WSE
            // Ensure solver mode is correct: if table data is null but mode=1,
            // fall back to mode 0 (direct secant).  This handles the case where
            // table build failed during initialization.
            if (s_culvert_solver_mode == 1 && !s_culvert_table_data) {
                s_culvert_solver_mode = 0;
            }
            if (sf_ws.d_prev_structure_flow) {
                CUDA_CHECK(cudaMemcpyAsync(sf_ws.d_prev_structure_flow, sf_ws.d_structure_flow,
                                           static_cast<size_t>(n_structures) * sizeof(double),
                                           cudaMemcpyDeviceToDevice, stream));
            }
            CUDA_CHECK(cudaMemsetAsync(sf_ws.d_structure_flow, 0, static_cast<size_t>(n_structures) * sizeof(double), stream));
            {
                int grid_sf = (n_structures + BLOCK - 1) / BLOCK;
                swe2d_compute_structure_flows_kernel<<<grid_sf, BLOCK, 0, stream>>>(
                    n_cells, n_structures, sf_ws.d_cell_wse,
                    sf_ws.d_structure_type, sf_ws.d_upstream_cell, sf_ws.d_downstream_cell,
                    sf_ws.d_crest_elev, sf_ws.d_width, sf_ws.d_height,
                    sf_ws.d_diameter, sf_ws.d_length, sf_ws.d_roughness_n,
                    sf_ws.d_coeff, sf_ws.d_cd, sf_ws.d_opening,
                    sf_ws.d_q_pump, sf_ws.d_max_flow,
                    sf_ws.d_culvert_code, sf_ws.d_culvert_shape,
                    sf_ws.d_culvert_rise, sf_ws.d_culvert_span, sf_ws.d_culvert_area,
                    sf_ws.d_culvert_barrels, sf_ws.d_culvert_slope,
                    sf_ws.d_inlet_invert_elev, sf_ws.d_outlet_invert_elev,
                    sf_ws.d_entrance_loss_k, sf_ws.d_exit_loss_k,
                    sf_ws.d_embankment_enabled, sf_ws.d_embankment_crest_elev,
                    sf_ws.d_embankment_overflow_width, sf_ws.d_embankment_weir_coeff,
                    sf_ws.gravity, sf_ws.model_to_ft, sf_ws.d_structure_flow,
                    sf_ws.d_prev_structure_flow,
                    s_culvert_solver_mode, s_culvert_table_header, s_culvert_table_data,
                    s_culvert_table_n_hw, s_culvert_table_n_tw);
                CUDA_CHECK(cudaGetLastError());
            }

            // Face-flux kernel: reads fresh culvert flows BEFORE masking
            {
                int grid_ff = (ff.n_culvert_faces + BLOCK - 1) / BLOCK;
                swe2d_culvert_face_flux_kernel<<<grid_ff, BLOCK, 0, stream>>>(
                    ff.n_culvert_faces,
                    sf_ws.d_structure_flow,
                    ff.d_culvert_struct_idx,
                    ff.d_face_nx, ff.d_face_ny, ff.d_face_width,
                    ff.d_donor_cell, ff.d_receiver_cell,
                    ff.d_invert_elev, ff.d_depth_safety,
                    ff.d_donor_cell_area,
                    dev->d_h, dev->d_hu, dev->d_hv, dev->d_cell_zb,
                    sf_ws.gravity, s_coupling_dt, 1.0e-6,
                    n_cells,
                    dev->d_ext_struct_flux_h, dev->d_ext_struct_flux_hu, dev->d_ext_struct_flux_hv);
                CUDA_CHECK(cudaGetLastError());
            }
            // Mask culvert flows so source kernel skips them
            {
                int grid_m = (ff.n_culvert_faces + BLOCK - 1) / BLOCK;
                swe2d_mask_culvert_source_kernel<<<grid_m, BLOCK, 0, stream>>>(
                    ff.n_culvert_faces, ff.d_culvert_struct_idx, sf_ws.d_structure_flow);
                CUDA_CHECK(cudaGetLastError());
            }
        }

        // ── Source kernel: applies non-culvert structures ──────────────
        int grid_src = (n_structures + BLOCK - 1) / BLOCK;
        swe2d_coupling_structure_source_kernel<<<grid_src, BLOCK, 0, stream>>>(
            n_structures, sf_ws.d_upstream_cell, sf_ws.d_downstream_cell, sf_ws.d_structure_flow, cpl_ws.d_cell_area, n_cells,
            dev->d_external_source_mps);
        CUDA_CHECK(cudaGetLastError());

        // ── Fold culvert face-flux mass into external source ──────────
        // When face-flux mode is active, the face-flux kernel already adds
        // mass via ext_struct_flux_h which the update kernel reads directly.
        // The fold is only needed when face-flux is NOT active (fallback
        // path where culvert mass was folded by apply_native_device_sources
        // before the graph was captured).
        if (!dev->use_culvert_face_flux
            && dev->culvert_ff_ws.params_preloaded
            && dev->culvert_ff_ws.n_culvert_faces > 0
            && dev->d_ext_struct_flux_h) {
            int grid_fold = (n_cells + BLOCK - 1) / BLOCK;
            swe2d_fold_culvert_mass_to_source_kernel<<<grid_fold, BLOCK, 0, stream>>>(
                n_cells,
                dev->d_ext_struct_flux_h,
                cpl_ws.d_cell_area,
                dev->d_external_source_mps);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    if (n_inlets > 0 && inlet_cell && inlet_flow_cms) {
        sf_ensure_buf(cpl_ws.d_inlet_cell, cpl_ws.inlet_capacity, n_inlets);
        sf_ensure_buf(cpl_ws.d_inlet_q, cpl_ws.inlet_capacity, n_inlets);
        CUDA_CHECK(cudaMemcpyAsync(cpl_ws.d_inlet_cell, inlet_cell, static_cast<size_t>(n_inlets) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(cpl_ws.d_inlet_q, inlet_flow_cms, static_cast<size_t>(n_inlets) * sizeof(double), cudaMemcpyHostToDevice, stream));
        int grid = (n_inlets + BLOCK - 1) / BLOCK;
        swe2d_coupling_inlet_source_kernel<<<grid, BLOCK, 0, stream>>>(
            n_inlets, cpl_ws.d_inlet_cell, cpl_ws.d_inlet_q, cpl_ws.d_cell_area, n_cells, dev->d_external_source_mps);
        CUDA_CHECK(cudaGetLastError());
    }

    // Sync stream after coupling work so the solver's graph capture on the next
    // step starts with a clean stream.  The coupling function is called from host
    // code (apply_native_device_sources) and the solver's graph capture/replay
    // uses the same stream; without this sync, pending async work causes
    // cudaStreamBeginCapture to fail on the next solver step.
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void swe2d_gpu_readback_coupling_sources(double* host_buf, int32_t n_cells)
{
    SWE2DDeviceState* dev = s_coupling_dev;
    if (!dev || !dev->d_external_source_mps || n_cells <= 0) {
        if (host_buf && n_cells > 0) {
            std::memset(host_buf, 0, static_cast<size_t>(n_cells) * sizeof(double));
        }
        return;
    }
    CUDA_CHECK(cudaMemcpy(host_buf, dev->d_external_source_mps,
                          static_cast<size_t>(n_cells) * sizeof(double),
                          cudaMemcpyDeviceToHost));
}

void swe2d_gpu_readback_structure_flows(double* host_buf, int32_t n_structures)
{
    SWE2DDeviceState* dev = s_coupling_dev;
    if (!dev || !dev->sf_ws.d_structure_flow || n_structures <= 0) {
        if (host_buf && n_structures > 0) {
            std::memset(host_buf, 0, static_cast<size_t>(n_structures) * sizeof(double));
        }
        return;
    }
    CUDA_CHECK(cudaMemcpy(host_buf, dev->sf_ws.d_structure_flow,
                          static_cast<size_t>(n_structures) * sizeof(double),
                          cudaMemcpyDeviceToHost));
}

void swe2d_gpu_readback_coupling_wse(double* host_buf, int32_t n_cells)
{
    SWE2DDeviceState* dev = s_coupling_dev;
    if (!dev || !dev->sf_ws.d_cell_wse || n_cells <= 0) {
        if (host_buf && n_cells > 0) {
            std::memset(host_buf, 0, static_cast<size_t>(n_cells) * sizeof(double));
        }
        return;
    }
    CUDA_CHECK(cudaMemcpy(host_buf, dev->sf_ws.d_cell_wse,
                          static_cast<size_t>(n_cells) * sizeof(double),
                          cudaMemcpyDeviceToHost));
}

void swe2d_gpu_upload_structure_flows(const double* host_buf, int32_t n_structures)
{
    SWE2DDeviceState* dev = s_coupling_dev;
    if (!dev || !dev->sf_ws.d_structure_flow || n_structures <= 0 || !host_buf) return;
    CUDA_CHECK(cudaMemcpy(dev->sf_ws.d_structure_flow, host_buf,
                          static_cast<size_t>(n_structures) * sizeof(double),
                          cudaMemcpyHostToDevice));
}

// ─────────────────────────────────────────────────────────────────────────────
// Face-based culvert coupling: upload + orchestration
// ─────────────────────────────────────────────────────────────────────────────

static void sf_ensure_buf_ff_d(double*& ptr, int32_t& cap, int32_t needed)
{
    if (needed <= cap && ptr) return;
    if (ptr) cudaFree(ptr);
    CUDA_CHECK(cudaMalloc(&ptr, static_cast<size_t>(needed) * sizeof(double)));
    cap = needed;
}

static void sf_ensure_buf_ff_i(int32_t*& ptr, int32_t& cap, int32_t needed)
{
    if (needed <= cap && ptr) return;
    if (ptr) cudaFree(ptr);
    CUDA_CHECK(cudaMalloc(&ptr, static_cast<size_t>(needed) * sizeof(int32_t)));
    cap = needed;
}

void swe2d_gpu_alloc_ext_struct_flux(SWE2DDeviceState* dev, int32_t n_cells)
{
    if (!dev || n_cells <= 0) return;
    if (dev->d_ext_struct_flux_h && dev->n_cells == n_cells) return;
    // Free old if size changed
    if (dev->d_ext_struct_flux_h)  { cudaFree(dev->d_ext_struct_flux_h);  dev->d_ext_struct_flux_h  = nullptr; }
    if (dev->d_ext_struct_flux_hu) { cudaFree(dev->d_ext_struct_flux_hu); dev->d_ext_struct_flux_hu = nullptr; }
    if (dev->d_ext_struct_flux_hv) { cudaFree(dev->d_ext_struct_flux_hv); dev->d_ext_struct_flux_hv = nullptr; }
    CUDA_CHECK(cudaMalloc(&dev->d_ext_struct_flux_h,  static_cast<size_t>(n_cells) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dev->d_ext_struct_flux_hu, static_cast<size_t>(n_cells) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dev->d_ext_struct_flux_hv, static_cast<size_t>(n_cells) * sizeof(double)));
}

void swe2d_gpu_fold_culvert_mass_to_source(SWE2DDeviceState* dev, int32_t n_cells)
{
    if (!dev || n_cells <= 0) return;
    if (!dev->d_ext_struct_flux_h || !dev->d_external_source_mps) return;
    auto& cpl_ws = dev->coupling_ws;
    if (!cpl_ws.d_cell_area) return;
    constexpr int BLOCK = 256;
    int grid = (n_cells + BLOCK - 1) / BLOCK;
    swe2d_fold_culvert_mass_to_source_kernel<<<grid, BLOCK, 0, dev->d_stream>>>(
        n_cells,
        dev->d_ext_struct_flux_h,
        cpl_ws.d_cell_area,
        dev->d_external_source_mps);
    CUDA_CHECK(cudaGetLastError());
}

void swe2d_gpu_upload_culvert_face_flux_params(
    SWE2DDeviceState* dev,
    int32_t n_culvert_faces,
    const int32_t* culvert_struct_idx,
    const double*  face_nx,
    const double*  face_ny,
    const double*  face_width,
    const int32_t* donor_cell,
    const int32_t* receiver_cell,
    const double*  invert_elev,
    const double*  depth_safety,
    const double*  donor_cell_area,
    const int32_t* enquiry_up_cell,   // nullable: enquiry cells for WSE sampling
    const int32_t* enquiry_dn_cell,   // nullable: enquiry cells for WSE sampling
    bool use_face_flux)
{
    if (!dev) dev = s_coupling_dev;
    if (!dev) throw std::runtime_error("upload_culvert_face_flux_params: no GPU device state");
    auto& ff = dev->culvert_ff_ws;

    if (n_culvert_faces <= 0) {
        ff.n_culvert_faces = 0;
        ff.params_preloaded = false;
        dev->use_culvert_face_flux = false;
        return;
    }

    // Ensure ext_struct_flux arrays exist
    swe2d_gpu_alloc_ext_struct_flux(dev, dev->n_cells);

    // Upload culvert struct index
    sf_ensure_buf_ff_i(ff.d_culvert_struct_idx, ff.face_capacity, n_culvert_faces);
    CUDA_CHECK(cudaMemcpy(ff.d_culvert_struct_idx, culvert_struct_idx,
                          static_cast<size_t>(n_culvert_faces) * sizeof(int32_t),
                          cudaMemcpyHostToDevice));

    // Upload face geometry
    sf_ensure_buf_ff_d(ff.d_face_nx, ff.face_capacity, n_culvert_faces);
    sf_ensure_buf_ff_d(ff.d_face_ny, ff.face_capacity, n_culvert_faces);
    sf_ensure_buf_ff_d(ff.d_face_width, ff.face_capacity, n_culvert_faces);
    CUDA_CHECK(cudaMemcpy(ff.d_face_nx, face_nx,
                          static_cast<size_t>(n_culvert_faces) * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ff.d_face_ny, face_ny,
                          static_cast<size_t>(n_culvert_faces) * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ff.d_face_width, face_width,
                          static_cast<size_t>(n_culvert_faces) * sizeof(double), cudaMemcpyHostToDevice));

    // Upload cell topology
    sf_ensure_buf_ff_i(ff.d_donor_cell, ff.face_capacity, n_culvert_faces);
    sf_ensure_buf_ff_i(ff.d_receiver_cell, ff.face_capacity, n_culvert_faces);
    CUDA_CHECK(cudaMemcpy(ff.d_donor_cell, donor_cell,
                          static_cast<size_t>(n_culvert_faces) * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ff.d_receiver_cell, receiver_cell,
                          static_cast<size_t>(n_culvert_faces) * sizeof(int32_t), cudaMemcpyHostToDevice));

    // Upload invert elevation and depth safety
    sf_ensure_buf_ff_d(ff.d_invert_elev, ff.face_capacity, n_culvert_faces);
    sf_ensure_buf_ff_d(ff.d_depth_safety, ff.face_capacity, n_culvert_faces);
    CUDA_CHECK(cudaMemcpy(ff.d_invert_elev, invert_elev,
                          static_cast<size_t>(n_culvert_faces) * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ff.d_depth_safety, depth_safety,
                          static_cast<size_t>(n_culvert_faces) * sizeof(double), cudaMemcpyHostToDevice));

    // Upload donor-cell area for depth safety limiter
    sf_ensure_buf_ff_d(ff.d_donor_cell_area, ff.face_capacity, n_culvert_faces);
    CUDA_CHECK(cudaMemcpy(ff.d_donor_cell_area, donor_cell_area,
                          static_cast<size_t>(n_culvert_faces) * sizeof(double), cudaMemcpyHostToDevice));

    // Upload enquiry cell indices (nullable — if null, face cells are used,
    // so the kernel skips the WSE override and uses the face-cell WSE directly).
    if (enquiry_up_cell && enquiry_dn_cell) {
        sf_ensure_buf_ff_i(ff.d_enquiry_up_cell, ff.face_capacity, n_culvert_faces);
        sf_ensure_buf_ff_i(ff.d_enquiry_dn_cell, ff.face_capacity, n_culvert_faces);
        CUDA_CHECK(cudaMemcpy(ff.d_enquiry_up_cell, enquiry_up_cell,
                              static_cast<size_t>(n_culvert_faces) * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ff.d_enquiry_dn_cell, enquiry_dn_cell,
                              static_cast<size_t>(n_culvert_faces) * sizeof(int32_t), cudaMemcpyHostToDevice));
    }

    ff.n_culvert_faces = n_culvert_faces;
    ff.params_preloaded = true;
    dev->use_culvert_face_flux = use_face_flux;
}

void swe2d_gpu_apply_culvert_face_flux(
    SWE2DDeviceState* dev,
    double dt,
    double h_min)
{
    if (!dev || !dev->culvert_ff_ws.params_preloaded || dev->culvert_ff_ws.n_culvert_faces <= 0) return;

    auto& ff = dev->culvert_ff_ws;
    auto& sf = dev->sf_ws;
    cudaStream_t stream = dev->d_stream;
    constexpr int BLOCK = 256;

    // Ensure per-cell flux accumulators exist
    swe2d_gpu_alloc_ext_struct_flux(dev, dev->n_cells);

    // Zero the external flux accumulators
    CUDA_CHECK(cudaMemsetAsync(dev->d_ext_struct_flux_h,  0,
                               static_cast<size_t>(dev->n_cells) * sizeof(double), stream));
    CUDA_CHECK(cudaMemsetAsync(dev->d_ext_struct_flux_hu, 0,
                               static_cast<size_t>(dev->n_cells) * sizeof(double), stream));
    CUDA_CHECK(cudaMemsetAsync(dev->d_ext_struct_flux_hv, 0,
                               static_cast<size_t>(dev->n_cells) * sizeof(double), stream));

    // Ensure structure flows are computed (Q_c must be in sf.d_structure_flow)
    if (sf.d_structure_flow && sf.params_preloaded && sf.n_structures > 0) {
        if (sf.d_prev_structure_flow) {
            CUDA_CHECK(cudaMemcpyAsync(sf.d_prev_structure_flow, sf.d_structure_flow,
                                       static_cast<size_t>(sf.n_structures) * sizeof(double),
                                       cudaMemcpyDeviceToDevice, stream));
        }
        CUDA_CHECK(cudaMemsetAsync(sf.d_structure_flow, 0,
                                   static_cast<size_t>(sf.n_structures) * sizeof(double), stream));
        int grid_sf = (sf.n_structures + BLOCK - 1) / BLOCK;
        swe2d_compute_structure_flows_kernel<<<grid_sf, BLOCK, 0, stream>>>(
            dev->n_cells, sf.n_structures, sf.d_cell_wse,
            sf.d_structure_type, sf.d_upstream_cell, sf.d_downstream_cell,
            sf.d_crest_elev, sf.d_width, sf.d_height,
            sf.d_diameter, sf.d_length, sf.d_roughness_n,
            sf.d_coeff, sf.d_cd, sf.d_opening,
            sf.d_q_pump, sf.d_max_flow,
            sf.d_culvert_code, sf.d_culvert_shape,
            sf.d_culvert_rise, sf.d_culvert_span, sf.d_culvert_area,
            sf.d_culvert_barrels, sf.d_culvert_slope,
            sf.d_inlet_invert_elev, sf.d_outlet_invert_elev,
            sf.d_entrance_loss_k, sf.d_exit_loss_k,
            sf.d_embankment_enabled, sf.d_embankment_crest_elev,
            sf.d_embankment_overflow_width, sf.d_embankment_weir_coeff,
            sf.gravity, sf.model_to_ft, sf.d_structure_flow,
            sf.d_prev_structure_flow,
            s_culvert_solver_mode, s_culvert_table_header, s_culvert_table_data,
            s_culvert_table_n_hw, s_culvert_table_n_tw);
        CUDA_CHECK(cudaGetLastError());
    }

    // Launch face-flux kernel
    {
        int grid = (ff.n_culvert_faces + BLOCK - 1) / BLOCK;
        swe2d_culvert_face_flux_kernel<<<grid, BLOCK, 0, stream>>>(
            ff.n_culvert_faces,
            sf.d_structure_flow,
            ff.d_culvert_struct_idx,
            ff.d_face_nx,
            ff.d_face_ny,
            ff.d_face_width,
            ff.d_donor_cell,
            ff.d_receiver_cell,
            ff.d_invert_elev,
            ff.d_depth_safety,
            ff.d_donor_cell_area,
            dev->d_h,
            dev->d_hu,
            dev->d_hv,
            dev->d_cell_zb,
            sf.gravity,
            dt,
            h_min,
            dev->n_cells,
            dev->d_ext_struct_flux_h,
            dev->d_ext_struct_flux_hu,
            dev->d_ext_struct_flux_hv);
        CUDA_CHECK(cudaGetLastError());
    }

    // Mask culvert flows so the source-kernel skips them
    if (sf.d_structure_flow && ff.n_culvert_faces > 0) {
        int grid_m = (ff.n_culvert_faces + BLOCK - 1) / BLOCK;
        swe2d_mask_culvert_source_kernel<<<grid_m, BLOCK, 0, stream>>>(
            ff.n_culvert_faces,
            ff.d_culvert_struct_idx,
            sf.d_structure_flow);
        CUDA_CHECK(cudaGetLastError());
    }

    if (swe2d_debug_enabled("BACKWATER_SWE2D_SYNC_COUPLING")) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
}

void swe2d_gpu_readback_ext_struct_flux(
    double* host_h, double* host_hu, double* host_hv, int32_t n_cells)
{
    SWE2DDeviceState* dev = s_coupling_dev;
    if (!dev || n_cells <= 0) {
        if (host_h)  std::memset(host_h,  0, static_cast<size_t>(n_cells) * sizeof(double));
        if (host_hu) std::memset(host_hu, 0, static_cast<size_t>(n_cells) * sizeof(double));
        if (host_hv) std::memset(host_hv, 0, static_cast<size_t>(n_cells) * sizeof(double));
        return;
    }
    if (host_h && dev->d_ext_struct_flux_h)
        CUDA_CHECK(cudaMemcpy(host_h, dev->d_ext_struct_flux_h,
                              static_cast<size_t>(n_cells) * sizeof(double), cudaMemcpyDeviceToHost));
    if (host_hu && dev->d_ext_struct_flux_hu)
        CUDA_CHECK(cudaMemcpy(host_hu, dev->d_ext_struct_flux_hu,
                              static_cast<size_t>(n_cells) * sizeof(double), cudaMemcpyDeviceToHost));
    if (host_hv && dev->d_ext_struct_flux_hv)
        CUDA_CHECK(cudaMemcpy(host_hv, dev->d_ext_struct_flux_hv,
                              static_cast<size_t>(n_cells) * sizeof(double), cudaMemcpyDeviceToHost));
}

void swe2d_gpu_upload_ext_struct_flux_h(const double* host_h, int32_t n_cells)
{
    SWE2DDeviceState* dev = s_coupling_dev;
    if (!dev || !dev->d_ext_struct_flux_h || n_cells <= 0 || !host_h) return;
    CUDA_CHECK(cudaMemcpy(dev->d_ext_struct_flux_h, host_h,
                          static_cast<size_t>(n_cells) * sizeof(double),
                          cudaMemcpyHostToDevice));
}

// ── Culvert table-mode globals ──
int32_t s_culvert_solver_mode = 0;
double* s_culvert_table_header = nullptr;
double* s_culvert_table_data = nullptr;
int32_t s_culvert_table_n_hw = 32;
int32_t s_culvert_table_n_tw = 16;

void swe2d_gpu_compute_coupling_sources(
    SWE2DDeviceState* dev,
    int32_t n_cells,
    const double* cell_area,
    int32_t n_inlets,
    const int32_t* inlet_cell,
    const double* inlet_flow_cms,
    int32_t n_structures,
    const int32_t* structure_up_cell,
    const int32_t* structure_down_cell,
    const double* structure_flow,
    double* source_rate_out)
{
    if (!source_rate_out || n_cells <= 0) return;
    std::fill(source_rate_out, source_rate_out + static_cast<size_t>(n_cells), 0.0);
    if (n_inlets <= 0 && n_structures <= 0) return;

    constexpr int BLOCK = 256;
    const bool use_stream = (dev != nullptr);
    cudaStream_t stream = use_stream ? dev->d_stream : nullptr;

    // Use persistent workspace (dev->coupling_ws) or static-cache fallback.
    static SWE2DDeviceState::CouplingWorkspace s_fallback_ws{};
    auto& ws = use_stream ? dev->coupling_ws : s_fallback_ws;

    double* d_cell_area = nullptr;
    double* d_source = nullptr;
    int32_t* d_inlet_cell = nullptr;
    double* d_inlet_q = nullptr;
    int32_t* d_struct_up = nullptr;
    int32_t* d_struct_dn = nullptr;
    double* d_struct_q = nullptr;

    try {
        if (ws.cell_capacity < n_cells) {
            double* new_cell_area = nullptr;
            double* new_source = nullptr;
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&new_cell_area), static_cast<size_t>(n_cells) * sizeof(double)));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&new_source), static_cast<size_t>(n_cells) * sizeof(double)));
            if (ws.d_cell_area) cudaFree(ws.d_cell_area);
            if (ws.d_source) cudaFree(ws.d_source);
            ws.d_cell_area = new_cell_area;
            ws.d_source = new_source;
            ws.cell_capacity = n_cells;
        }
        d_cell_area = ws.d_cell_area;
        d_source = ws.d_source;

        if (n_inlets > 0 && inlet_cell && inlet_flow_cms) {
            if (ws.inlet_capacity < n_inlets) {
                int32_t* new_inlet_cell = nullptr;
                double* new_inlet_q = nullptr;
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&new_inlet_cell), static_cast<size_t>(n_inlets) * sizeof(int32_t)));
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&new_inlet_q), static_cast<size_t>(n_inlets) * sizeof(double)));
                if (ws.d_inlet_cell) cudaFree(ws.d_inlet_cell);
                if (ws.d_inlet_q) cudaFree(ws.d_inlet_q);
                ws.d_inlet_cell = new_inlet_cell;
                ws.d_inlet_q = new_inlet_q;
                ws.inlet_capacity = n_inlets;
            }
            d_inlet_cell = ws.d_inlet_cell;
            d_inlet_q = ws.d_inlet_q;
        }

        if (n_structures > 0 && structure_up_cell && structure_down_cell && structure_flow) {
            if (ws.structure_capacity < n_structures) {
                int32_t* new_struct_up = nullptr;
                int32_t* new_struct_dn = nullptr;
                double* new_struct_q = nullptr;
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&new_struct_up), static_cast<size_t>(n_structures) * sizeof(int32_t)));
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&new_struct_dn), static_cast<size_t>(n_structures) * sizeof(int32_t)));
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&new_struct_q), static_cast<size_t>(n_structures) * sizeof(double)));
                if (ws.d_struct_up) cudaFree(ws.d_struct_up);
                if (ws.d_struct_dn) cudaFree(ws.d_struct_dn);
                if (ws.d_struct_q) cudaFree(ws.d_struct_q);
                ws.d_struct_up = new_struct_up;
                ws.d_struct_dn = new_struct_dn;
                ws.d_struct_q = new_struct_q;
                ws.structure_capacity = n_structures;
            }
            d_struct_up = ws.d_struct_up;
            d_struct_dn = ws.d_struct_dn;
            d_struct_q = ws.d_struct_q;
        }

        // Upload and launch using device stream when available (async overlap).
        auto copy_h2d = [stream, use_stream](void* dst, const void* src, size_t bytes) {
            if (use_stream)
                CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream));
            else
                CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
        };
        auto copy_d2h = [stream, use_stream](void* dst, const void* src, size_t bytes) {
            if (use_stream)
                CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream));
            else
                CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
        };

        copy_h2d(d_cell_area, cell_area, static_cast<size_t>(n_cells) * sizeof(double));
        if (use_stream)
            CUDA_CHECK(cudaMemsetAsync(d_source, 0, static_cast<size_t>(n_cells) * sizeof(double), stream));
        else
            CUDA_CHECK(cudaMemset(d_source, 0, static_cast<size_t>(n_cells) * sizeof(double)));

        if (n_inlets > 0 && d_inlet_cell && d_inlet_q) {
            copy_h2d(d_inlet_cell, inlet_cell, static_cast<size_t>(n_inlets) * sizeof(int32_t));
            copy_h2d(d_inlet_q, inlet_flow_cms, static_cast<size_t>(n_inlets) * sizeof(double));
            const int grid = (n_inlets + BLOCK - 1) / BLOCK;
            swe2d_coupling_inlet_source_kernel<<<grid, BLOCK, 0, stream>>>(
                n_inlets, d_inlet_cell, d_inlet_q, d_cell_area, n_cells, d_source);
            CUDA_CHECK(cudaGetLastError());
        }

        if (n_structures > 0 && d_struct_up && d_struct_dn && d_struct_q) {
            copy_h2d(d_struct_up, structure_up_cell, static_cast<size_t>(n_structures) * sizeof(int32_t));
            copy_h2d(d_struct_dn, structure_down_cell, static_cast<size_t>(n_structures) * sizeof(int32_t));
            copy_h2d(d_struct_q, structure_flow, static_cast<size_t>(n_structures) * sizeof(double));
            const int grid = (n_structures + BLOCK - 1) / BLOCK;
            swe2d_coupling_structure_source_kernel<<<grid, BLOCK, 0, stream>>>(
                n_structures, d_struct_up, d_struct_dn, d_struct_q, d_cell_area, n_cells, d_source);
            CUDA_CHECK(cudaGetLastError());
        }

        copy_d2h(source_rate_out, d_source, static_cast<size_t>(n_cells) * sizeof(double));
        if (use_stream) CUDA_CHECK(cudaStreamSynchronize(stream));
    } catch (...) {
        throw;
    }
}

void swe2d_gpu_compute_bridge_coupling_sources(
    SWE2DDeviceState* dev,
    int32_t n_cells,
    const double* cell_area,
    int32_t n_bridges,
    const int32_t* bridge_up_cell,
    const int32_t* bridge_down_cell,
    const double* bridge_flow_cms,
    const double* bridge_loss_k_upstream,
    const double* bridge_loss_k_downstream,
    double bridge_opening_width_m,
    double dt_s,
    double* source_rate_out)
{
    if (!source_rate_out || n_cells <= 0) return;
    std::fill(source_rate_out, source_rate_out + static_cast<size_t>(n_cells), 0.0);
    if (n_bridges <= 0) return;

    constexpr int BLOCK = 256;
    const bool use_stream = (dev != nullptr);
    cudaStream_t stream = use_stream ? dev->d_stream : nullptr;

    static SWE2DDeviceState::CouplingWorkspace s_fallback_ws2{};
    auto& ws = use_stream ? dev->coupling_ws : s_fallback_ws2;

    double* d_cell_area = nullptr;
    double* d_source = nullptr;
    int32_t* d_bridge_up = nullptr;
    int32_t* d_bridge_dn = nullptr;
    double* d_bridge_q = nullptr;
    double* d_bridge_ku = nullptr;
    double* d_bridge_kd = nullptr;

    try {
        if (ws.bridge_cell_capacity < n_cells) {
            double* new_cell_area = nullptr;
            double* new_source = nullptr;
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&new_cell_area), static_cast<size_t>(n_cells) * sizeof(double)));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&new_source), static_cast<size_t>(n_cells) * sizeof(double)));
            if (ws.d_bridge_cell_area) cudaFree(ws.d_bridge_cell_area);
            if (ws.d_bridge_source) cudaFree(ws.d_bridge_source);
            ws.d_bridge_cell_area = new_cell_area;
            ws.d_bridge_source = new_source;
            ws.bridge_cell_capacity = n_cells;
        }
        if (ws.bridge_capacity < n_bridges) {
            int32_t* new_up = nullptr;
            int32_t* new_dn = nullptr;
            double* new_q = nullptr;
            double* new_ku = nullptr;
            double* new_kd = nullptr;
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&new_up), static_cast<size_t>(n_bridges) * sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&new_dn), static_cast<size_t>(n_bridges) * sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&new_q), static_cast<size_t>(n_bridges) * sizeof(double)));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&new_ku), static_cast<size_t>(n_bridges) * sizeof(double)));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&new_kd), static_cast<size_t>(n_bridges) * sizeof(double)));
            if (ws.d_bridge_up) cudaFree(ws.d_bridge_up);
            if (ws.d_bridge_dn) cudaFree(ws.d_bridge_dn);
            if (ws.d_bridge_q) cudaFree(ws.d_bridge_q);
            if (ws.d_bridge_ku) cudaFree(ws.d_bridge_ku);
            if (ws.d_bridge_kd) cudaFree(ws.d_bridge_kd);
            ws.d_bridge_up = new_up;
            ws.d_bridge_dn = new_dn;
            ws.d_bridge_q = new_q;
            ws.d_bridge_ku = new_ku;
            ws.d_bridge_kd = new_kd;
            ws.bridge_capacity = n_bridges;
        }

        d_cell_area = ws.d_bridge_cell_area;
        d_source = ws.d_bridge_source;
        d_bridge_up = ws.d_bridge_up;
        d_bridge_dn = ws.d_bridge_dn;
        d_bridge_q = ws.d_bridge_q;
        d_bridge_ku = ws.d_bridge_ku;
        d_bridge_kd = ws.d_bridge_kd;

        auto copy_h2d = [stream, use_stream](void* dst, const void* src, size_t bytes) {
            if (use_stream)
                CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream));
            else
                CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
        };
        auto copy_d2h = [stream, use_stream](void* dst, const void* src, size_t bytes) {
            if (use_stream)
                CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream));
            else
                CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
        };

        copy_h2d(d_cell_area, cell_area, static_cast<size_t>(n_cells) * sizeof(double));
        if (use_stream)
            CUDA_CHECK(cudaMemsetAsync(d_source, 0, static_cast<size_t>(n_cells) * sizeof(double), stream));
        else
            CUDA_CHECK(cudaMemset(d_source, 0, static_cast<size_t>(n_cells) * sizeof(double)));

        if (bridge_up_cell && bridge_down_cell && bridge_flow_cms && bridge_loss_k_upstream && bridge_loss_k_downstream) {
            copy_h2d(d_bridge_up, bridge_up_cell, static_cast<size_t>(n_bridges) * sizeof(int32_t));
            copy_h2d(d_bridge_dn, bridge_down_cell, static_cast<size_t>(n_bridges) * sizeof(int32_t));
            copy_h2d(d_bridge_q, bridge_flow_cms, static_cast<size_t>(n_bridges) * sizeof(double));
            copy_h2d(d_bridge_ku, bridge_loss_k_upstream, static_cast<size_t>(n_bridges) * sizeof(double));
            copy_h2d(d_bridge_kd, bridge_loss_k_downstream, static_cast<size_t>(n_bridges) * sizeof(double));
            const int grid = (n_bridges + BLOCK - 1) / BLOCK;
            swe2d_coupling_bridge_source_kernel<<<grid, BLOCK, 0, stream>>>(
                n_bridges, d_bridge_up, d_bridge_dn, d_bridge_q, d_bridge_ku, d_bridge_kd,
                d_cell_area, n_cells, bridge_opening_width_m, dt_s, d_source);
            CUDA_CHECK(cudaGetLastError());
        }

        copy_d2h(source_rate_out, d_source, static_cast<size_t>(n_cells) * sizeof(double));
        if (use_stream) CUDA_CHECK(cudaStreamSynchronize(stream));
    } catch (...) {
        throw;
    }
}

// ── Culvert table-mode globals defined above ──
// (see non-static definitions after set_external_sources)

void swe2d_gpu_set_culvert_solver_mode_impl(
    int32_t mode, const double* data, const double* header,
    size_t data_sz, size_t header_sz, int32_t n_hw, int32_t n_tw)
{
    s_culvert_solver_mode = mode;
    if (s_culvert_table_data) { cudaFree(s_culvert_table_data); s_culvert_table_data = nullptr; }
    if (s_culvert_table_header) { cudaFree(s_culvert_table_header); s_culvert_table_header = nullptr; }
    s_culvert_table_n_hw = n_hw;
    s_culvert_table_n_tw = n_tw;
    if (mode == 1 && data && header && data_sz > 0 && header_sz > 0) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s_culvert_table_data), data_sz * sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s_culvert_table_header), header_sz * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(s_culvert_table_data, data, data_sz * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(s_culvert_table_header, header, header_sz * sizeof(double), cudaMemcpyHostToDevice));
    }
}

void swe2d_gpu_compute_structure_flows(
    int32_t n_cells,
    int32_t n_structures,
    const double* cell_wse,
    const double* cell_bed,
    const int32_t* structure_type,
    const int32_t* upstream_cell,
    const int32_t* downstream_cell,
    const double* crest_elev,
    const double* width,
    const double* height,
    const double* diameter,
    const double* length,
    const double* roughness_n,
    const double* coeff,
    const double* cd,
    const double* opening,
    const double* q_pump,
    const double* max_flow,
    const int32_t* culvert_code,
    const int32_t* culvert_shape,
    const double* culvert_rise,
    const double* culvert_span,
    const double* culvert_area,
    const double* culvert_barrels,
    const double* culvert_slope,
    const double* inlet_invert_elev,
    const double* outlet_invert_elev,
    const double* entrance_loss_k,
    const double* exit_loss_k,
    const int32_t* embankment_enabled,
    const double* embankment_crest_elev,
    const double* embankment_overflow_width,
    const double* embankment_weir_coeff,
    double gravity,
    double model_to_ft,
    double* structure_flow_out)
{
    if (!structure_flow_out || n_structures <= 0 || n_cells <= 0) {
        return;
    }
    std::fill(structure_flow_out, structure_flow_out + static_cast<size_t>(n_structures), 0.0);

    constexpr int BLOCK = 256;
    // Persistent workspace: function-local static survives for process lifetime,
    // eliminating per-call cudaMalloc churn for the 33 device buffers.
    static int32_t s_cell_cap = 0, s_struct_cap = 0;
    static double* s_d_cell_wse = nullptr;
    static double* s_d_cell_bed = nullptr;
    static int32_t* s_d_structure_type = nullptr;
    static int32_t* s_d_upstream_cell = nullptr;
    static int32_t* s_d_downstream_cell = nullptr;
    static double* s_d_crest_elev = nullptr;
    static double* s_d_width = nullptr;
    static double* s_d_height = nullptr;
    static double* s_d_diameter = nullptr;
    static double* s_d_length = nullptr;
    static double* s_d_roughness_n = nullptr;
    static double* s_d_coeff = nullptr;
    static double* s_d_cd = nullptr;
    static double* s_d_opening = nullptr;
    static double* s_d_q_pump = nullptr;
    static double* s_d_max_flow = nullptr;
    static int32_t* s_d_culvert_code = nullptr;
    static int32_t* s_d_culvert_shape = nullptr;
    static double* s_d_culvert_rise = nullptr;
    static double* s_d_culvert_span = nullptr;
    static double* s_d_culvert_area = nullptr;
    static double* s_d_culvert_barrels = nullptr;
    static double* s_d_culvert_slope = nullptr;
    static double* s_d_inlet_invert_elev = nullptr;
    static double* s_d_outlet_invert_elev = nullptr;
    static double* s_d_entrance_loss_k = nullptr;
    static double* s_d_exit_loss_k = nullptr;
    static int32_t* s_d_embankment_enabled = nullptr;
    static double* s_d_embankment_crest_elev = nullptr;
    static double* s_d_embankment_overflow_width = nullptr;
    static double* s_d_embankment_weir_coeff = nullptr;
    static double* s_d_structure_flow = nullptr;

    // Culvert table-mode storage (file scope, accessed by set_culvert_solver_mode_impl
    // and the kernel launch in this function).
    // ── NOTE: these are declared at file scope, NOT inside this function ──

    // Grow buffers if needed (template-free explicit approach to avoid CUDA lambda issues)
    #define SF_ENSURE(ptr, cap, need, sz) do { \
        if (!(ptr) || (cap) < (need)) { \
            void* _np = nullptr; \
            CUDA_CHECK(cudaMalloc(&_np, static_cast<size_t>(need) * (sz))); \
            if (ptr) cudaFree(ptr); \
            ptr = static_cast<decltype(ptr)>(_np); \
            (cap) = (need); \
        } \
    } while(0)

    SF_ENSURE(s_d_cell_wse, s_cell_cap, n_cells, sizeof(double));
    SF_ENSURE(s_d_cell_bed, s_cell_cap, n_cells, sizeof(double));
    SF_ENSURE(s_d_structure_type, s_struct_cap, n_structures, sizeof(int32_t));
    SF_ENSURE(s_d_upstream_cell, s_struct_cap, n_structures, sizeof(int32_t));
    SF_ENSURE(s_d_downstream_cell, s_struct_cap, n_structures, sizeof(int32_t));
    SF_ENSURE(s_d_crest_elev, s_struct_cap, n_structures, sizeof(double));
    SF_ENSURE(s_d_width, s_struct_cap, n_structures, sizeof(double));
    SF_ENSURE(s_d_height, s_struct_cap, n_structures, sizeof(double));
    SF_ENSURE(s_d_diameter, s_struct_cap, n_structures, sizeof(double));
    SF_ENSURE(s_d_length, s_struct_cap, n_structures, sizeof(double));
    SF_ENSURE(s_d_roughness_n, s_struct_cap, n_structures, sizeof(double));
    SF_ENSURE(s_d_coeff, s_struct_cap, n_structures, sizeof(double));
    SF_ENSURE(s_d_cd, s_struct_cap, n_structures, sizeof(double));
    SF_ENSURE(s_d_opening, s_struct_cap, n_structures, sizeof(double));
    SF_ENSURE(s_d_q_pump, s_struct_cap, n_structures, sizeof(double));
    SF_ENSURE(s_d_max_flow, s_struct_cap, n_structures, sizeof(double));
    SF_ENSURE(s_d_culvert_code, s_struct_cap, n_structures, sizeof(int32_t));
    SF_ENSURE(s_d_culvert_shape, s_struct_cap, n_structures, sizeof(int32_t));
    SF_ENSURE(s_d_culvert_rise, s_struct_cap, n_structures, sizeof(double));
    SF_ENSURE(s_d_culvert_span, s_struct_cap, n_structures, sizeof(double));
    SF_ENSURE(s_d_culvert_area, s_struct_cap, n_structures, sizeof(double));
    SF_ENSURE(s_d_culvert_barrels, s_struct_cap, n_structures, sizeof(double));
    SF_ENSURE(s_d_culvert_slope, s_struct_cap, n_structures, sizeof(double));
    SF_ENSURE(s_d_inlet_invert_elev, s_struct_cap, n_structures, sizeof(double));
    SF_ENSURE(s_d_outlet_invert_elev, s_struct_cap, n_structures, sizeof(double));
    SF_ENSURE(s_d_entrance_loss_k, s_struct_cap, n_structures, sizeof(double));
    SF_ENSURE(s_d_exit_loss_k, s_struct_cap, n_structures, sizeof(double));
    SF_ENSURE(s_d_embankment_enabled, s_struct_cap, n_structures, sizeof(int32_t));
    SF_ENSURE(s_d_embankment_crest_elev, s_struct_cap, n_structures, sizeof(double));
    SF_ENSURE(s_d_embankment_overflow_width, s_struct_cap, n_structures, sizeof(double));
    SF_ENSURE(s_d_embankment_weir_coeff, s_struct_cap, n_structures, sizeof(double));
    SF_ENSURE(s_d_structure_flow, s_struct_cap, n_structures, sizeof(double));

    auto upload = [](auto* dst, const auto* src, size_t n) {
        (void)sizeof(dst);
        (void)sizeof(src);
        CUDA_CHECK(cudaMemcpy(dst, static_cast<const void*>(src), n * sizeof(std::decay_t<decltype(*src)>), cudaMemcpyHostToDevice));
    };
    try {
        if (cell_wse) upload(s_d_cell_wse, cell_wse, n_cells);
        if (structure_type) upload(s_d_structure_type, structure_type, n_structures);
        if (upstream_cell) upload(s_d_upstream_cell, upstream_cell, n_structures);
        if (downstream_cell) upload(s_d_downstream_cell, downstream_cell, n_structures);
        if (crest_elev) upload(s_d_crest_elev, crest_elev, n_structures);
        if (width) upload(s_d_width, width, n_structures);
        if (height) upload(s_d_height, height, n_structures);
        if (diameter) upload(s_d_diameter, diameter, n_structures);
        if (length) upload(s_d_length, length, n_structures);
        if (roughness_n) upload(s_d_roughness_n, roughness_n, n_structures);
        if (coeff) upload(s_d_coeff, coeff, n_structures);
        if (cd) upload(s_d_cd, cd, n_structures);
        if (opening) upload(s_d_opening, opening, n_structures);
        if (q_pump) upload(s_d_q_pump, q_pump, n_structures);
        if (max_flow) upload(s_d_max_flow, max_flow, n_structures);
        if (culvert_code) upload(s_d_culvert_code, culvert_code, n_structures);
        if (culvert_shape) upload(s_d_culvert_shape, culvert_shape, n_structures);
        if (culvert_rise) upload(s_d_culvert_rise, culvert_rise, n_structures);
        if (culvert_span) upload(s_d_culvert_span, culvert_span, n_structures);
        if (culvert_area) upload(s_d_culvert_area, culvert_area, n_structures);
        if (culvert_barrels) upload(s_d_culvert_barrels, culvert_barrels, n_structures);
        if (culvert_slope) upload(s_d_culvert_slope, culvert_slope, n_structures);
        if (inlet_invert_elev) upload(s_d_inlet_invert_elev, inlet_invert_elev, n_structures);
        if (outlet_invert_elev) upload(s_d_outlet_invert_elev, outlet_invert_elev, n_structures);
        if (entrance_loss_k) upload(s_d_entrance_loss_k, entrance_loss_k, n_structures);
        if (exit_loss_k) upload(s_d_exit_loss_k, exit_loss_k, n_structures);
        if (embankment_enabled) upload(s_d_embankment_enabled, embankment_enabled, n_structures);
        if (embankment_crest_elev) upload(s_d_embankment_crest_elev, embankment_crest_elev, n_structures);
        if (embankment_overflow_width) upload(s_d_embankment_overflow_width, embankment_overflow_width, n_structures);
        if (embankment_weir_coeff) upload(s_d_embankment_weir_coeff, embankment_weir_coeff, n_structures);

        CUDA_CHECK(cudaMemset(s_d_structure_flow, 0, static_cast<size_t>(n_structures) * sizeof(double)));

        const int grid = (n_structures + BLOCK - 1) / BLOCK;
        swe2d_compute_structure_flows_kernel<<<grid, BLOCK>>>(
            n_cells, n_structures,
            s_d_cell_wse, s_d_structure_type,
            s_d_upstream_cell, s_d_downstream_cell,
            s_d_crest_elev, s_d_width, s_d_height,
            s_d_diameter, s_d_length, s_d_roughness_n,
            s_d_coeff, s_d_cd, s_d_opening,
            s_d_q_pump, s_d_max_flow,
            s_d_culvert_code, s_d_culvert_shape,
            s_d_culvert_rise, s_d_culvert_span, s_d_culvert_area,
            s_d_culvert_barrels, s_d_culvert_slope,
            s_d_inlet_invert_elev, s_d_outlet_invert_elev,
            s_d_entrance_loss_k, s_d_exit_loss_k,
            s_d_embankment_enabled, s_d_embankment_crest_elev,
            s_d_embankment_overflow_width, s_d_embankment_weir_coeff,
            gravity, model_to_ft, s_d_structure_flow,
            nullptr,  // no prev-flow tracking in legacy path
            s_culvert_solver_mode,
            s_culvert_table_header, s_culvert_table_data,
            s_culvert_table_n_hw, s_culvert_table_n_tw);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(structure_flow_out, s_d_structure_flow,
                              static_cast<size_t>(n_structures) * sizeof(double),
                              cudaMemcpyDeviceToHost));
    } catch (...) {
        throw;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Culvert lookup-table mode: GPU kernel that fills per-culvert Q(hw,tw) tables
// using the existing device-side secant solver, then host helpers to upload.
// ─────────────────────────────────────────────────────────────────────────────

// Kernel: one thread per (culvert, hw_idx, tw_idx).  Fills d_table with
// Q_outlet for each headwater/tailwater grid point.
__global__ void swe2d_culvert_build_table_kernel(
    int32_t n_culverts,
    int32_t n_hw,
    int32_t n_tw,
    const int32_t* __restrict__ culvert_code,
    const int32_t* __restrict__ culvert_shape,
    const double* __restrict__ culvert_rise,
    const double* __restrict__ culvert_span,
    const double* __restrict__ culvert_diameter,
    const double* __restrict__ culvert_length,
    const double* __restrict__ culvert_roughness_n,
    const double* __restrict__ culvert_slope,
    const double* __restrict__ entrance_loss_k,
    const double* __restrict__ exit_loss_k,
    double* __restrict__ d_table_data,
    double* __restrict__ d_table_header)
{
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t total = n_culverts * n_hw * n_tw;
    if (idx >= total) return;

    int32_t ci = idx / (n_hw * n_tw);
    int32_t rem = idx % (n_hw * n_tw);
    int32_t hi = rem / n_tw;
    int32_t ti = rem % n_tw;

    if (ci >= n_culverts) return;

    // Build cross-section from culvert params
    const double rise = fmax(0.0, culvert_rise[ci]);
    const double span = fmax(0.0, (culvert_span[ci] > 0.0) ? culvert_span[ci] : fmax(rise, culvert_diameter[ci]));
    const int code = max(1, min(57, static_cast<int>(culvert_code[ci])));

    swe2d_culvert_xsect_cuda xsect{};
    xsect.code = code;
    xsect.is_rect = (culvert_shape[ci] == 1) ? 1 : 0;
    if (xsect.is_rect) {
        xsect.width_ft = fmax(1.0e-6, span);
        xsect.y_full_ft = fmax(1.0e-6, rise);
        xsect.a_full_ft2 = xsect.width_ft * xsect.y_full_ft;
        xsect.radius_ft = 0.0;
    } else {
        const double dia_ft = fmax(1.0e-6, fmax(culvert_diameter[ci], rise));
        xsect.radius_ft = 0.5 * dia_ft;
        xsect.y_full_ft = dia_ft;
        xsect.a_full_ft2 = M_PI * xsect.radius_ft * xsect.radius_ft;
        xsect.width_ft = 0.0;
    }

    const double y_full_ft = xsect.y_full_ft;
    const double hw_ft = fmax(0.0, (static_cast<double>(hi) / fmax(1.0, static_cast<double>(n_hw - 1))) * y_full_ft * 2.0);
    const double tw_ft = fmax(0.0, (static_cast<double>(ti) / fmax(1.0, static_cast<double>(n_tw - 1))) * y_full_ft);

    double slope = fmax(1.0e-6, culvert_slope[ci]);
    double len_ft = fmax(0.1, culvert_length[ci]);

    // Inlet control for hint
    const double q_inlet = fmax(0.0, swe2d_culvert_inlet_controlled_flow_cfs_cuda(xsect, slope, hw_ft));

    double orifice_cap = 0.0;
    if (culvert_diameter[ci] > 0.0) {
        const double a_orif = bw2d_circular_area(culvert_diameter[ci]);
        const double dh_orif = fmax(0.0, hw_ft - tw_ft);
        if (a_orif > 0.0 && dh_orif > 1.0e-12) {
            orifice_cap = a_orif * sqrt(2.0 * USC_GRAVITY * dh_orif);  // CFS
        }
    }
    const double q_hint = fmax(1.0, fmax(q_inlet, orifice_cap));

    double q_outlet_cfs = 0.0;
    if (hw_ft > 0.0) {
        q_outlet_cfs = swe2d_culvert_outlet_control_flow_cms_cuda(
            xsect, hw_ft, tw_ft, len_ft, slope,
            fmax(1.0e-6, culvert_roughness_n[ci]),
            entrance_loss_k[ci], exit_loss_k[ci], q_hint);
    }

    double q = fmax(0.0, fmin(q_inlet, (q_outlet_cfs > 0.0) ? q_outlet_cfs : q_inlet));
    if (orifice_cap > 0.0) q = (q > 0.0) ? fmin(q, orifice_cap) : orifice_cap;

    // Write result
    d_table_data[static_cast<size_t>(ci) * static_cast<size_t>(n_hw) * static_cast<size_t>(n_tw)
                  + static_cast<size_t>(hi) * static_cast<size_t>(n_tw) + static_cast<size_t>(ti)] = q;

    // Thread 0 of each culvert writes the header
    if (hi == 0 && ti == 0) {
        size_t off = static_cast<size_t>(ci) * 6;
        d_table_header[off + 0] = static_cast<double>(n_hw);
        d_table_header[off + 1] = static_cast<double>(n_tw);
        d_table_header[off + 2] = 0.0;                   // hw_min (always 0)
        d_table_header[off + 3] = y_full_ft * 2.0;       // hw_max
        d_table_header[off + 4] = 0.0;                   // tw_min
        d_table_header[off + 5] = y_full_ft;             // tw_max
    }
}

bool swe2d_gpu_build_culvert_tables(
    int32_t n_culverts,
    const int32_t* culvert_code,
    const int32_t* culvert_shape,
    const double* culvert_rise,
    const double* culvert_span,
    const double* culvert_diameter,
    const double* culvert_length,
    const double* culvert_roughness_n,
    const double* culvert_slope,
    const double* entrance_loss_k,
    const double* exit_loss_k,
    int32_t n_hw,
    int32_t n_tw,
    std::vector<double>& table_data_out,
    std::vector<double>& table_header_out)
{
    if (n_culverts <= 0) {
        table_data_out.clear();
        table_header_out.clear();
        return true;
    }

    constexpr int BLOCK = 256;
    int32_t n_hw_use = fmax(2, n_hw);
    int32_t n_tw_use = fmax(2, n_tw);
    int32_t total_points = n_culverts * n_hw_use * n_tw_use;
    int32_t total_cells = (total_points + BLOCK - 1) / BLOCK;

    // Allocate device buffers for the kernel
    int32_t* d_code = nullptr, *d_shape = nullptr;
    double* d_rise = nullptr, *d_span = nullptr, *d_diam = nullptr;
    double* d_len = nullptr, *d_n = nullptr, *d_slope = nullptr;
    double* d_ent_k = nullptr, *d_exit_k = nullptr;
    double* d_table_data = nullptr, *d_table_header = nullptr;

    auto alloc_copy = [](auto** dptr, const auto* hptr, size_t n) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(dptr), n * sizeof(std::decay_t<decltype(*hptr)>)));
        CUDA_CHECK(cudaMemcpy(*dptr, hptr, n * sizeof(std::decay_t<decltype(*hptr)>), cudaMemcpyHostToDevice));
    };

    try {
        alloc_copy(&d_code, culvert_code, n_culverts);
        alloc_copy(&d_shape, culvert_shape, n_culverts);
        alloc_copy(&d_rise, culvert_rise, n_culverts);
        alloc_copy(&d_span, culvert_span, n_culverts);
        alloc_copy(&d_diam, culvert_diameter, n_culverts);
        alloc_copy(&d_len, culvert_length, n_culverts);
        alloc_copy(&d_n, culvert_roughness_n, n_culverts);
        alloc_copy(&d_slope, culvert_slope, n_culverts);
        alloc_copy(&d_ent_k, entrance_loss_k, n_culverts);
        alloc_copy(&d_exit_k, exit_loss_k, n_culverts);

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_table_data),
                              static_cast<size_t>(total_points) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_table_header),
                              static_cast<size_t>(n_culverts) * 6 * sizeof(double)));

        swe2d_culvert_build_table_kernel<<<total_cells, BLOCK>>>(
            n_culverts, n_hw_use, n_tw_use,
            d_code, d_shape, d_rise, d_span, d_diam,
            d_len, d_n, d_slope, d_ent_k, d_exit_k,
            d_table_data, d_table_header);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        table_data_out.resize(static_cast<size_t>(total_points));
        table_header_out.resize(static_cast<size_t>(n_culverts) * 6);
        CUDA_CHECK(cudaMemcpy(table_data_out.data(), d_table_data,
                              static_cast<size_t>(total_points) * sizeof(double),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(table_header_out.data(), d_table_header,
                              static_cast<size_t>(n_culverts) * 6 * sizeof(double),
                              cudaMemcpyDeviceToHost));
    } catch (...) {
        if (d_code) cudaFree(d_code);
        if (d_shape) cudaFree(d_shape);
        if (d_rise) cudaFree(d_rise);
        if (d_span) cudaFree(d_span);
        if (d_diam) cudaFree(d_diam);
        if (d_len) cudaFree(d_len);
        if (d_n) cudaFree(d_n);
        if (d_slope) cudaFree(d_slope);
        if (d_ent_k) cudaFree(d_ent_k);
        if (d_exit_k) cudaFree(d_exit_k);
        if (d_table_data) cudaFree(d_table_data);
        if (d_table_header) cudaFree(d_table_header);
        return false;
    }

    if (d_code) cudaFree(d_code);
    if (d_shape) cudaFree(d_shape);
    if (d_rise) cudaFree(d_rise);
    if (d_span) cudaFree(d_span);
    if (d_diam) cudaFree(d_diam);
    if (d_len) cudaFree(d_len);
    if (d_n) cudaFree(d_n);
    if (d_slope) cudaFree(d_slope);
    if (d_ent_k) cudaFree(d_ent_k);
    if (d_exit_k) cudaFree(d_exit_k);
    if (d_table_data) cudaFree(d_table_data);
    if (d_table_header) cudaFree(d_table_header);
    return true;
}

void swe2d_gpu_upload_culvert_tables(
    CulvertLookupTableDesc& desc,
    const std::vector<double>& table_data,
    const std::vector<double>& table_header)
{
    if (desc.d_table_data) cudaFree(desc.d_table_data);
    if (desc.d_table_header) cudaFree(desc.d_table_header);
    desc.d_table_data = nullptr;
    desc.d_table_header = nullptr;
    desc.uploaded = false;

    if (table_data.empty() || table_header.empty()) return;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&desc.d_table_data),
                          table_data.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&desc.d_table_header),
                          table_header.size() * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(desc.d_table_data, table_data.data(),
                          table_data.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(desc.d_table_header, table_header.data(),
                          table_header.size() * sizeof(double), cudaMemcpyHostToDevice));
    desc.uploaded = true;
}

void swe2d_gpu_release_culvert_tables(CulvertLookupTableDesc& desc) {
    if (desc.d_table_data) { cudaFree(desc.d_table_data); desc.d_table_data = nullptr; }
    if (desc.d_table_header) { cudaFree(desc.d_table_header); desc.d_table_header = nullptr; }
    desc.uploaded = false;
}

// Device-side bilinear table lookup: given headwater (ft) and tailwater (ft),
// interpolate Q (cms) from the pre-computed table.
// Header layout per culvert: [n_hw, n_tw, hw_min, hw_max, tw_min, tw_max]
// Table data: row-major [hw × tw] doubles.
__device__ double swe2d_culvert_table_lookup_cuda(
    int32_t ci,
    double hw_ft,
    double tw_ft,
    int32_t n_culverts,
    const double* __restrict__ d_table_header,
    const double* __restrict__ d_table_data,
    int32_t n_hw_global,
    int32_t n_tw_global)
{
    const size_t hdr_off = static_cast<size_t>(ci) * 6;
    int32_t n_hw = fmax(2, static_cast<int32_t>(d_table_header[hdr_off + 0]));
    int32_t n_tw = fmax(2, static_cast<int32_t>(d_table_header[hdr_off + 1]));
    double hw_min = d_table_header[hdr_off + 2];
    double hw_max = d_table_header[hdr_off + 3];
    double tw_min = d_table_header[hdr_off + 4];
    double tw_max = d_table_header[hdr_off + 5];

    // Clamp to table bounds
    double hw = fmin(hw_max, fmax(hw_min, hw_ft));
    double tw = fmin(tw_max, fmax(tw_min, tw_ft));

    double hw_step = (hw_max - hw_min) / fmax(1.0, static_cast<double>(n_hw - 1));
    double tw_step = (tw_max - tw_min) / fmax(1.0, static_cast<double>(n_tw - 1));

    double hw_frac = (hw - hw_min) / fmax(1.0e-12, hw_step);
    double tw_frac = (tw - tw_min) / fmax(1.0e-12, tw_step);

    int32_t hi0 = static_cast<int32_t>(hw_frac);
    int32_t ti0 = static_cast<int32_t>(tw_frac);
    int32_t hi1 = fmin(hi0 + 1, n_hw - 1);
    int32_t ti1 = fmin(ti0 + 1, n_tw - 1);
    hi0 = fmax(0, fmin(hi0, n_hw - 1));
    ti0 = fmax(0, fmin(ti0, n_tw - 1));

    double hw_w1 = (hw - hw_min) / fmax(1.0e-12, hw_max - hw_min) - static_cast<double>(hi0) / fmax(1.0, static_cast<double>(n_hw - 1));
    hw_w1 = fmin(1.0, fmax(0.0, hw_w1 * static_cast<double>(n_hw - 1)));
    double tw_w1 = (tw - tw_min) / fmax(1.0e-12, tw_max - tw_min) - static_cast<double>(ti0) / fmax(1.0, static_cast<double>(n_tw - 1));
    tw_w1 = fmin(1.0, fmax(0.0, tw_w1 * static_cast<double>(n_tw - 1)));

    size_t data_base = static_cast<size_t>(ci) * static_cast<size_t>(n_hw) * static_cast<size_t>(n_tw);
    auto q = [&](int32_t h, int32_t t) -> double {
        return d_table_data[data_base + static_cast<size_t>(h) * static_cast<size_t>(n_tw) + static_cast<size_t>(t)];
    };

    double q00 = q(hi0, ti0);
    double q10 = q(hi1, ti0);
    double q01 = q(hi0, ti1);
    double q11 = q(hi1, ti1);

    double q0 = q00 + (q10 - q00) * hw_w1;
    double q1 = q01 + (q11 - q01) * hw_w1;
    return q0 + (q1 - q0) * tw_w1;
}

// ── Persistent workspace for drainage step buffers ─────────────────────
// Fills s_coupling_dev->drain_ws with device buffers sized for n_cells/
// n_nodes/n_links/n_inlets/n_outfalls/n_pipe_ends.  Returns true when
// the workspace is ready.  Idempotent: skips re-allocation when capacity
// already sufficient (but always marks static geo as needing re-upload
// on first allocation via the return flag).
static bool ensure_drainage_step_workspace(
    int32_t n_cells, int32_t n_nodes, int32_t n_links,
    int32_t n_inlets, int32_t n_outfalls, int32_t n_pipe_ends)
{
    SWE2DDeviceState* dev = s_coupling_dev;
    if (!dev) return false;
    auto& ws = dev->drain_ws;

    #define _DS_ENSURE(ptr, capacity, needed, type) \
        if ((capacity) < (needed)) { \
            if (ptr) cudaFree(ptr); \
            ptr = nullptr; \
            capacity = 0; \
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), \
                                  static_cast<size_t>(needed) * sizeof(type))); \
            capacity = (needed); \
        }

    _DS_ENSURE(ws.d_cell_area, ws.cell_capacity, n_cells, double);
    _DS_ENSURE(ws.d_cell_wse,   ws.cell_capacity, n_cells, double);
    _DS_ENSURE(ws.d_cell_depth, ws.cell_capacity, n_cells, double);
    _DS_ENSURE(ws.d_q_cell,    ws.cell_capacity, n_cells, double);

    _DS_ENSURE(ws.d_node_inv,     ws.node_capacity, n_nodes, double);
    _DS_ENSURE(ws.d_node_maxd,    ws.node_capacity, n_nodes, double);
    _DS_ENSURE(ws.d_node_area,    ws.node_capacity, n_nodes, double);
    _DS_ENSURE(ws.d_node_depth,   ws.node_capacity, n_nodes, double);
    _DS_ENSURE(ws.d_node_net_q,   ws.node_capacity, n_nodes, double);
    _DS_ENSURE(ws.d_node_delta,   ws.node_capacity, n_nodes, double);
    _DS_ENSURE(ws.d_node_qleave,  ws.node_capacity, n_nodes, double);

    _DS_ENSURE(ws.d_l_from,   ws.link_capacity, n_links, int32_t);
    _DS_ENSURE(ws.d_l_to,     ws.link_capacity, n_links, int32_t);
    _DS_ENSURE(ws.d_l_len,    ws.link_capacity, n_links, double);
    _DS_ENSURE(ws.d_l_n,      ws.link_capacity, n_links, double);
    _DS_ENSURE(ws.d_l_d,      ws.link_capacity, n_links, double);
    _DS_ENSURE(ws.d_l_qmax,   ws.link_capacity, n_links, double);
    _DS_ENSURE(ws.d_l_q_prev, ws.link_capacity, n_links, double);
    _DS_ENSURE(ws.d_l_q,      ws.link_capacity, n_links, double);

    _DS_ENSURE(ws.d_i_cell,   ws.inlet_capacity, n_inlets, int32_t);
    _DS_ENSURE(ws.d_i_node,   ws.inlet_capacity, n_inlets, int32_t);
    _DS_ENSURE(ws.d_i_crest,  ws.inlet_capacity, n_inlets, double);
    _DS_ENSURE(ws.d_i_width,  ws.inlet_capacity, n_inlets, double);
    _DS_ENSURE(ws.d_i_cd,     ws.inlet_capacity, n_inlets, double);
    _DS_ENSURE(ws.d_i_qmax,   ws.inlet_capacity, n_inlets, double);

    _DS_ENSURE(ws.d_o_cell,        ws.outfall_capacity, n_outfalls, int32_t);
    _DS_ENSURE(ws.d_o_node,        ws.outfall_capacity, n_outfalls, int32_t);
    _DS_ENSURE(ws.d_o_invert,      ws.outfall_capacity, n_outfalls, double);
    _DS_ENSURE(ws.d_o_diameter,    ws.outfall_capacity, n_outfalls, double);
    _DS_ENSURE(ws.d_o_cd,          ws.outfall_capacity, n_outfalls, double);
    _DS_ENSURE(ws.d_o_qmax,        ws.outfall_capacity, n_outfalls, double);
    _DS_ENSURE(ws.d_o_zero_storage, ws.outfall_capacity, n_outfalls, int32_t);

    _DS_ENSURE(ws.d_p_cell,         ws.pipe_end_capacity, n_pipe_ends, int32_t);
    _DS_ENSURE(ws.d_p_node,         ws.pipe_end_capacity, n_pipe_ends, int32_t);
    _DS_ENSURE(ws.d_p_invert,       ws.pipe_end_capacity, n_pipe_ends, double);
    _DS_ENSURE(ws.d_p_diameter,     ws.pipe_end_capacity, n_pipe_ends, double);
    _DS_ENSURE(ws.d_p_area,         ws.pipe_end_capacity, n_pipe_ends, double);
    _DS_ENSURE(ws.d_p_kin,          ws.pipe_end_capacity, n_pipe_ends, double);
    _DS_ENSURE(ws.d_p_kout,         ws.pipe_end_capacity, n_pipe_ends, double);
    _DS_ENSURE(ws.d_p_depth_bc,     ws.pipe_end_capacity, n_pipe_ends, double);
    _DS_ENSURE(ws.d_p_node_area,    ws.pipe_end_capacity, n_pipe_ends, double);

    _DS_ENSURE(ws.d_limiter_events, ws.cell_capacity, 1, double);
    _DS_ENSURE(ws.d_limiter_volume, ws.cell_capacity, 1, double);

    ws.cell_capacity = n_cells;
    ws.node_capacity = n_nodes;
    ws.link_capacity = n_links;
    ws.inlet_capacity = n_inlets;
    ws.outfall_capacity = n_outfalls;
    ws.pipe_end_capacity = n_pipe_ends;

    #undef _DS_ENSURE
    return true;
}

void swe2d_gpu_drainage_step(
    int32_t n_cells,
    int32_t n_nodes,
    int32_t n_links,
    int32_t n_inlets,
    int32_t n_outfalls,
    int32_t n_pipe_ends,
    const double* cell_wse,
    const double* cell_area,
    const double* node_invert_elev,
    const double* node_max_depth,
    const double* node_surface_area,
    const int32_t* link_from,
    const int32_t* link_to,
    const double* link_length,
    const double* link_roughness_n,
    const double* link_diameter,
    const double* link_max_flow,
    const int32_t* inlet_cell,
    const int32_t* inlet_node,
    const double* inlet_crest_elev,
    const double* inlet_width,
    const double* inlet_coefficient,
    const double* inlet_max_capture,
    const int32_t* outfall_cell,
    const int32_t* outfall_node,
    const double* outfall_invert_elev,
    const double* outfall_diameter,
    const double* outfall_coefficient,
    const double* outfall_max_flow,
    const int32_t* outfall_zero_storage,
    const int32_t* pipe_end_cell,
    const int32_t* pipe_end_node,
    const double* pipe_end_invert_elev,
    const double* pipe_end_diameter,
    const double* pipe_end_area,
    const double* pipe_end_inlet_loss_k,
    const double* pipe_end_outlet_loss_k,
    const double* cell_depth,
    const double* node_depth_in,
    const double* link_flow_in,
    double dt_s,
    double gravity,
    int32_t solver_mode,
    double head_deadband_m,
    double dynamic_flow_relaxation,
    double* node_depth_out,
    double* link_flow_out,
    double* q_cell_out,
    double* max_node_depth_out,
    double* max_link_flow_out,
    double* limiter_event_count_out,
    double* limiter_volume_m3_out)
{
    if (!cell_area || !node_invert_elev || !node_max_depth || !node_surface_area ||
        !link_from || !link_to || !link_length || !link_roughness_n || !link_diameter || !link_max_flow ||
        !inlet_cell || !inlet_node || !inlet_crest_elev || !inlet_width || !inlet_coefficient || !inlet_max_capture ||
        !outfall_cell || !outfall_node || !outfall_invert_elev || !outfall_diameter || !outfall_coefficient || !outfall_max_flow || !outfall_zero_storage ||
        !pipe_end_cell || !pipe_end_node || !pipe_end_invert_elev || !pipe_end_diameter || !pipe_end_area ||
        !pipe_end_inlet_loss_k || !pipe_end_outlet_loss_k ||
        !node_depth_in || !link_flow_in || !node_depth_out || !link_flow_out || !q_cell_out ||
        n_cells < 0 || n_nodes < 0 || n_links < 0 || n_inlets < 0 || n_outfalls < 0 || n_pipe_ends < 0) {
        throw std::invalid_argument("swe2d_gpu_drainage_step: invalid arguments");
    }

    if (n_cells == 0 || n_nodes == 0) {
        if (max_node_depth_out) *max_node_depth_out = 0.0;
        if (max_link_flow_out) *max_link_flow_out = 0.0;
        if (limiter_event_count_out) *limiter_event_count_out = 0.0;
        if (limiter_volume_m3_out) *limiter_volume_m3_out = 0.0;
        return;
    }

    constexpr int BLOCK = 256;
    double *d_cell_wse = nullptr, *d_cell_area = nullptr, *d_cell_depth = nullptr;
    double *d_node_inv = nullptr, *d_node_maxd = nullptr, *d_node_area = nullptr;
    double *d_node_depth = nullptr, *d_node_net_q = nullptr, *d_node_delta = nullptr;
    int32_t *d_l_from = nullptr, *d_l_to = nullptr;
    double *d_l_len = nullptr, *d_l_n = nullptr, *d_l_d = nullptr, *d_l_qmax = nullptr;
    double *d_l_q_prev = nullptr, *d_l_q = nullptr;
    int32_t *d_i_cell = nullptr, *d_i_node = nullptr;
    double *d_i_crest = nullptr, *d_i_width = nullptr, *d_i_cd = nullptr, *d_i_qmax = nullptr;
    int32_t *d_o_cell = nullptr, *d_o_node = nullptr, *d_o_zero_storage = nullptr;
    double *d_o_invert = nullptr, *d_o_diameter = nullptr, *d_o_cd = nullptr, *d_o_qmax = nullptr;
    int32_t *d_p_cell = nullptr, *d_p_node = nullptr;
    double *d_p_invert = nullptr, *d_p_diameter = nullptr, *d_p_area = nullptr;
    double *d_p_kin = nullptr, *d_p_kout = nullptr;
    double *d_p_depth_bc = nullptr, *d_p_node_area = nullptr, *d_node_qleave = nullptr;
    double *d_q_cell = nullptr;
    double *d_limiter_events = nullptr, *d_limiter_volume = nullptr;

    try {
        // Clear stale CUDA error from prior operations on this stream.
        (void)cudaGetLastError();

        // Ensure persistent device buffers are allocated.
        SWE2DDeviceState* ds = s_coupling_dev;
        cudaStream_t stream = ds ? ds->d_stream : nullptr;
        bool use_persistent = (ds != nullptr && ensure_drainage_step_workspace(
            n_cells, n_nodes, n_links, n_inlets, n_outfalls, n_pipe_ends));

        auto* dws = &ds->drain_ws;  // only valid when use_persistent=true

        // Helper: upload to persistent or temp buffer.
        double* dev_cell_area  = nullptr;
        double* dev_cell_wse   = nullptr;
        double* dev_cell_depth = nullptr;
        // ... (all device pointers assigned below from ws or allocated locally)

        // Assign or allocate all device pointers from persistent workspace
        // when available, otherwise fall back to the old malloc+sync path.
        #define _DRAIN_ALLOC(dst, ptr, n, type) do { \
            if (use_persistent) { dst = ptr; } \
            else { \
                dst = nullptr; \
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dst), \
                                      static_cast<size_t>(n) * sizeof(type))); \
            } \
        } while(0)

        #define _DRAIN_H2D(dst, src, n, type) do { \
            size_t _b = static_cast<size_t>(n) * sizeof(type); \
            if (use_persistent) { \
                CUDA_CHECK(cudaMemcpyAsync(dst, src, _b, cudaMemcpyHostToDevice, stream)); \
            } else { \
                CUDA_CHECK(cudaMemcpy(dst, src, _b, cudaMemcpyHostToDevice)); \
            } \
        } while(0)

        #define _DRAIN_D2D(dst, src, n, type) do { \
            size_t _b = static_cast<size_t>(n) * sizeof(type); \
            if (use_persistent) { \
                CUDA_CHECK(cudaMemcpyAsync(dst, src, _b, cudaMemcpyDeviceToDevice, stream)); \
            } else { \
                CUDA_CHECK(cudaMemcpy(dst, src, _b, cudaMemcpyDeviceToDevice)); \
            } \
        } while(0)

        #define _DRAIN_MEMSET(dst, n, type) do { \
            size_t _b = static_cast<size_t>(n) * sizeof(type); \
            if (use_persistent) { \
                CUDA_CHECK(cudaMemsetAsync(dst, 0, _b, stream)); \
            } else { \
                CUDA_CHECK(cudaMemset(dst, 0, _b)); \
            } \
        } while(0)

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_cell_wse), static_cast<size_t>(n_cells) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_cell_area), static_cast<size_t>(n_cells) * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_cell_area, cell_area, static_cast<size_t>(n_cells) * sizeof(double), cudaMemcpyHostToDevice));

        // When cell_wse is null, compute WSE on-device from s_coupling_dev state.
        // This eliminates the D2H readback of h + host-side WSE computation.
        if (cell_wse) {
            CUDA_CHECK(cudaMemcpy(d_cell_wse, cell_wse, static_cast<size_t>(n_cells) * sizeof(double), cudaMemcpyHostToDevice));
        } else {
            SWE2DDeviceState* ds = s_coupling_dev;
            if (ds && ds->d_h && ds->d_cell_zb) {
                int grid = (n_cells + BLOCK - 1) / BLOCK;
                swe2d_coupling_wse_from_state_kernel<<<grid, BLOCK, 0, 0>>>(
                    n_cells, ds->d_h, ds->d_cell_zb, d_cell_wse);
                CUDA_CHECK(cudaGetLastError());
            }
        }
        if (cell_depth) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_cell_depth), static_cast<size_t>(n_cells) * sizeof(double)));
            CUDA_CHECK(cudaMemcpy(d_cell_depth, cell_depth, static_cast<size_t>(n_cells) * sizeof(double), cudaMemcpyHostToDevice));
        } else {
            // Compute depth on-device from s_coupling_dev->d_h
            SWE2DDeviceState* ds = s_coupling_dev;
            if (ds && ds->d_h) {
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_cell_depth), static_cast<size_t>(n_cells) * sizeof(double)));
                CUDA_CHECK(cudaMemcpy(d_cell_depth, ds->d_h, static_cast<size_t>(n_cells) * sizeof(double), cudaMemcpyDeviceToDevice));
            }
        }

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_node_inv), static_cast<size_t>(n_nodes) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_node_maxd), static_cast<size_t>(n_nodes) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_node_area), static_cast<size_t>(n_nodes) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_node_depth), static_cast<size_t>(n_nodes) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_node_net_q), static_cast<size_t>(n_nodes) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_node_delta), static_cast<size_t>(n_nodes) * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_node_inv, node_invert_elev, static_cast<size_t>(n_nodes) * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_node_maxd, node_max_depth, static_cast<size_t>(n_nodes) * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_node_area, node_surface_area, static_cast<size_t>(n_nodes) * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_node_depth, node_depth_in, static_cast<size_t>(n_nodes) * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_node_net_q, 0, static_cast<size_t>(n_nodes) * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_node_delta, 0, static_cast<size_t>(n_nodes) * sizeof(double)));

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_l_from), static_cast<size_t>(n_links) * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_l_to), static_cast<size_t>(n_links) * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_l_len), static_cast<size_t>(n_links) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_l_n), static_cast<size_t>(n_links) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_l_d), static_cast<size_t>(n_links) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_l_qmax), static_cast<size_t>(n_links) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_l_q_prev), static_cast<size_t>(n_links) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_l_q), static_cast<size_t>(n_links) * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_l_from, link_from, static_cast<size_t>(n_links) * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_l_to, link_to, static_cast<size_t>(n_links) * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_l_len, link_length, static_cast<size_t>(n_links) * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_l_n, link_roughness_n, static_cast<size_t>(n_links) * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_l_d, link_diameter, static_cast<size_t>(n_links) * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_l_qmax, link_max_flow, static_cast<size_t>(n_links) * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_l_q_prev, link_flow_in, static_cast<size_t>(n_links) * sizeof(double), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_i_cell), static_cast<size_t>(n_inlets) * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_i_node), static_cast<size_t>(n_inlets) * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_i_crest), static_cast<size_t>(n_inlets) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_i_width), static_cast<size_t>(n_inlets) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_i_cd), static_cast<size_t>(n_inlets) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_i_qmax), static_cast<size_t>(n_inlets) * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_i_cell, inlet_cell, static_cast<size_t>(n_inlets) * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_i_node, inlet_node, static_cast<size_t>(n_inlets) * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_i_crest, inlet_crest_elev, static_cast<size_t>(n_inlets) * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_i_width, inlet_width, static_cast<size_t>(n_inlets) * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_i_cd, inlet_coefficient, static_cast<size_t>(n_inlets) * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_i_qmax, inlet_max_capture, static_cast<size_t>(n_inlets) * sizeof(double), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_o_cell), static_cast<size_t>(n_outfalls) * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_o_node), static_cast<size_t>(n_outfalls) * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_o_invert), static_cast<size_t>(n_outfalls) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_o_diameter), static_cast<size_t>(n_outfalls) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_o_cd), static_cast<size_t>(n_outfalls) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_o_qmax), static_cast<size_t>(n_outfalls) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_o_zero_storage), static_cast<size_t>(n_outfalls) * sizeof(int32_t)));
        CUDA_CHECK(cudaMemcpy(d_o_cell, outfall_cell, static_cast<size_t>(n_outfalls) * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_o_node, outfall_node, static_cast<size_t>(n_outfalls) * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_o_invert, outfall_invert_elev, static_cast<size_t>(n_outfalls) * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_o_diameter, outfall_diameter, static_cast<size_t>(n_outfalls) * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_o_cd, outfall_coefficient, static_cast<size_t>(n_outfalls) * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_o_qmax, outfall_max_flow, static_cast<size_t>(n_outfalls) * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_o_zero_storage, outfall_zero_storage, static_cast<size_t>(n_outfalls) * sizeof(int32_t), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_p_cell), static_cast<size_t>(n_pipe_ends) * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_p_node), static_cast<size_t>(n_pipe_ends) * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_p_invert), static_cast<size_t>(n_pipe_ends) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_p_diameter), static_cast<size_t>(n_pipe_ends) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_p_area), static_cast<size_t>(n_pipe_ends) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_p_kin), static_cast<size_t>(n_pipe_ends) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_p_kout), static_cast<size_t>(n_pipe_ends) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_p_depth_bc), static_cast<size_t>(n_pipe_ends) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_p_node_area), static_cast<size_t>(n_pipe_ends) * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_p_cell, pipe_end_cell, static_cast<size_t>(n_pipe_ends) * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_p_node, pipe_end_node, static_cast<size_t>(n_pipe_ends) * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_p_invert, pipe_end_invert_elev, static_cast<size_t>(n_pipe_ends) * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_p_diameter, pipe_end_diameter, static_cast<size_t>(n_pipe_ends) * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_p_area, pipe_end_area, static_cast<size_t>(n_pipe_ends) * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_p_kin, pipe_end_inlet_loss_k, static_cast<size_t>(n_pipe_ends) * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_p_kout, pipe_end_outlet_loss_k, static_cast<size_t>(n_pipe_ends) * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_node_qleave), static_cast<size_t>(n_nodes) * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_node_qleave, 0, static_cast<size_t>(n_nodes) * sizeof(double)));

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_q_cell), static_cast<size_t>(n_cells) * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_q_cell, 0, static_cast<size_t>(n_cells) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_limiter_events), sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_limiter_volume), sizeof(double)));
        CUDA_CHECK(cudaMemset(d_limiter_events, 0, sizeof(double)));
        CUDA_CHECK(cudaMemset(d_limiter_volume, 0, sizeof(double)));

        if (n_links > 0) {
            const int grid_links = (n_links + BLOCK - 1) / BLOCK;
            swe2d_drainage_pipe_end_qleave_kernel<<<grid_links, BLOCK>>>(
                n_links, d_l_from, d_l_to, d_l_q_prev, d_node_qleave);
            CUDA_CHECK(cudaGetLastError());
        }

        if (n_pipe_ends > 0) {
            const int grid_pipe_ends = (n_pipe_ends + BLOCK - 1) / BLOCK;
            swe2d_drainage_pipe_end_bc_kernel<<<grid_pipe_ends, BLOCK>>>(
                n_pipe_ends,
                n_cells,
                d_p_cell,
                d_p_node,
                d_p_invert,
                d_p_diameter,
                d_p_area,
                d_p_kin,
                d_p_kout,
                d_cell_wse,
                d_node_inv,
                d_node_area,
                d_node_qleave,
                gravity,
                d_node_depth,
                d_p_depth_bc,
                d_p_node_area);
            CUDA_CHECK(cudaGetLastError());
        }

        if (n_links > 0) {
            const int grid_links = (n_links + BLOCK - 1) / BLOCK;
            swe2d_drainage_link_kernel<<<grid_links, BLOCK>>>(
                n_links, d_l_from, d_l_to, d_l_len, d_l_n, d_l_d, d_l_qmax,
                d_node_inv, d_node_depth, d_l_q_prev, dt_s, gravity, solver_mode,
                head_deadband_m, dynamic_flow_relaxation,
                d_l_q, d_node_net_q);
            CUDA_CHECK(cudaGetLastError());
        }

        const int grid_nodes = (n_nodes + BLOCK - 1) / BLOCK;
        swe2d_drainage_node_update_kernel<<<grid_nodes, BLOCK>>>(
            n_nodes, d_node_maxd, d_node_area, d_node_net_q, d_node_depth, dt_s, d_node_depth);
        CUDA_CHECK(cudaGetLastError());

        if (n_pipe_ends > 0) {
            const int grid_pipe_ends = (n_pipe_ends + BLOCK - 1) / BLOCK;
            swe2d_drainage_pipe_end_exchange_kernel<<<grid_pipe_ends, BLOCK>>>(
                n_pipe_ends,
                n_cells,
                d_p_cell,
                d_p_node,
                d_p_depth_bc,
                d_p_node_area,
                d_cell_area,
                d_cell_depth,
                d_node_maxd,
                dt_s,
                d_node_depth,
                d_q_cell,
                d_node_depth,
                d_limiter_events,
                d_limiter_volume);
            CUDA_CHECK(cudaGetLastError());
        }

        if (n_inlets > 0) {
            const int grid_inlets = (n_inlets + BLOCK - 1) / BLOCK;
            swe2d_drainage_inlet_exchange_kernel<<<grid_inlets, BLOCK>>>(
                n_inlets, n_cells, d_i_cell, d_i_node, d_i_crest, d_i_width,
                d_i_cd, d_i_qmax, d_cell_wse, d_cell_area, d_cell_depth,
                d_node_inv, d_node_maxd, d_node_depth, d_node_area,
                dt_s, gravity, head_deadband_m, d_q_cell, d_node_delta,
                d_limiter_events, d_limiter_volume);
            CUDA_CHECK(cudaGetLastError());
            swe2d_drainage_apply_delta_kernel<<<grid_nodes, BLOCK>>>(
                n_nodes, d_node_maxd, d_node_delta, d_node_depth);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemset(d_node_delta, 0, static_cast<size_t>(n_nodes) * sizeof(double)));
        }

        if (n_outfalls > 0) {
            const int grid_outfalls = (n_outfalls + BLOCK - 1) / BLOCK;
            swe2d_drainage_outfall_exchange_kernel<<<grid_outfalls, BLOCK>>>(
                n_outfalls, n_cells,
                d_o_cell, d_o_node, d_o_invert, d_o_diameter, d_o_cd, d_o_qmax, d_o_zero_storage,
                d_cell_wse, d_cell_area, d_cell_depth,
                d_node_maxd, d_node_depth, d_node_area,
                dt_s, gravity, head_deadband_m,
                d_q_cell, d_node_delta, d_node_depth,
                d_limiter_events, d_limiter_volume);
            CUDA_CHECK(cudaGetLastError());
            swe2d_drainage_apply_delta_kernel<<<grid_nodes, BLOCK>>>(
                n_nodes, d_node_maxd, d_node_delta, d_node_depth);
            CUDA_CHECK(cudaGetLastError());
        }
        // Sync only the solver stream (not all device streams) so we don't
        // wait for unrelated GPU work from previous steps.
        if (s_coupling_dev && s_coupling_dev->d_stream) {
            CUDA_CHECK(cudaStreamSynchronize(s_coupling_dev->d_stream));
        } else {
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        CUDA_CHECK(cudaMemcpy(node_depth_out, d_node_depth, static_cast<size_t>(n_nodes) * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(link_flow_out, d_l_q, static_cast<size_t>(n_links) * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(q_cell_out, d_q_cell, static_cast<size_t>(n_cells) * sizeof(double), cudaMemcpyDeviceToHost));

        if (max_node_depth_out) {
            const auto it = std::max_element(node_depth_out, node_depth_out + static_cast<size_t>(n_nodes));
            *max_node_depth_out = (it != node_depth_out + static_cast<size_t>(n_nodes)) ? *it : 0.0;
        }
        if (max_link_flow_out) {
            double qmax = 0.0;
            for (int32_t i = 0; i < n_links; ++i) qmax = std::max(qmax, std::abs(link_flow_out[i]));
            *max_link_flow_out = qmax;
        }
        if (limiter_event_count_out) {
            CUDA_CHECK(cudaMemcpy(limiter_event_count_out, d_limiter_events, sizeof(double), cudaMemcpyDeviceToHost));
        }
        if (limiter_volume_m3_out) {
            CUDA_CHECK(cudaMemcpy(limiter_volume_m3_out, d_limiter_volume, sizeof(double), cudaMemcpyDeviceToHost));
        }
    } catch (...) {
        if (d_cell_wse) cudaFree(d_cell_wse);
        if (d_cell_area) cudaFree(d_cell_area);
        if (d_cell_depth) cudaFree(d_cell_depth);
        if (d_node_inv) cudaFree(d_node_inv);
        if (d_node_maxd) cudaFree(d_node_maxd);
        if (d_node_area) cudaFree(d_node_area);
        if (d_node_depth) cudaFree(d_node_depth);
        if (d_node_net_q) cudaFree(d_node_net_q);
        if (d_node_delta) cudaFree(d_node_delta);
        if (d_l_from) cudaFree(d_l_from);
        if (d_l_to) cudaFree(d_l_to);
        if (d_l_len) cudaFree(d_l_len);
        if (d_l_n) cudaFree(d_l_n);
        if (d_l_d) cudaFree(d_l_d);
        if (d_l_qmax) cudaFree(d_l_qmax);
        if (d_l_q_prev) cudaFree(d_l_q_prev);
        if (d_l_q) cudaFree(d_l_q);
        if (d_i_cell) cudaFree(d_i_cell);
        if (d_i_node) cudaFree(d_i_node);
        if (d_i_crest) cudaFree(d_i_crest);
        if (d_i_width) cudaFree(d_i_width);
        if (d_i_cd) cudaFree(d_i_cd);
        if (d_i_qmax) cudaFree(d_i_qmax);
        if (d_o_cell) cudaFree(d_o_cell);
        if (d_o_node) cudaFree(d_o_node);
        if (d_o_invert) cudaFree(d_o_invert);
        if (d_o_diameter) cudaFree(d_o_diameter);
        if (d_o_cd) cudaFree(d_o_cd);
        if (d_o_qmax) cudaFree(d_o_qmax);
        if (d_o_zero_storage) cudaFree(d_o_zero_storage);
        if (d_q_cell) cudaFree(d_q_cell);
        if (d_limiter_events) cudaFree(d_limiter_events);
        if (d_limiter_volume) cudaFree(d_limiter_volume);
        throw;
    }

    if (d_cell_wse) cudaFree(d_cell_wse);
    if (d_cell_area) cudaFree(d_cell_area);
    if (d_cell_depth) cudaFree(d_cell_depth);
    if (d_node_inv) cudaFree(d_node_inv);
    if (d_node_maxd) cudaFree(d_node_maxd);
    if (d_node_area) cudaFree(d_node_area);
    if (d_node_depth) cudaFree(d_node_depth);
    if (d_node_net_q) cudaFree(d_node_net_q);
    if (d_node_delta) cudaFree(d_node_delta);
    if (d_l_from) cudaFree(d_l_from);
    if (d_l_to) cudaFree(d_l_to);
    if (d_l_len) cudaFree(d_l_len);
    if (d_l_n) cudaFree(d_l_n);
    if (d_l_d) cudaFree(d_l_d);
    if (d_l_qmax) cudaFree(d_l_qmax);
    if (d_l_q_prev) cudaFree(d_l_q_prev);
    if (d_l_q) cudaFree(d_l_q);
    if (d_i_cell) cudaFree(d_i_cell);
    if (d_i_node) cudaFree(d_i_node);
    if (d_i_crest) cudaFree(d_i_crest);
    if (d_i_width) cudaFree(d_i_width);
    if (d_i_cd) cudaFree(d_i_cd);
    if (d_i_qmax) cudaFree(d_i_qmax);
    if (d_o_cell) cudaFree(d_o_cell);
    if (d_o_node) cudaFree(d_o_node);
    if (d_o_invert) cudaFree(d_o_invert);
    if (d_o_diameter) cudaFree(d_o_diameter);
    if (d_o_cd) cudaFree(d_o_cd);
    if (d_o_qmax) cudaFree(d_o_qmax);
    if (d_o_zero_storage) cudaFree(d_o_zero_storage);
    if (d_p_cell) cudaFree(d_p_cell);
    if (d_p_node) cudaFree(d_p_node);
    if (d_p_invert) cudaFree(d_p_invert);
    if (d_p_diameter) cudaFree(d_p_diameter);
    if (d_p_area) cudaFree(d_p_area);
    if (d_p_kin) cudaFree(d_p_kin);
    if (d_p_kout) cudaFree(d_p_kout);
    if (d_p_depth_bc) cudaFree(d_p_depth_bc);
    if (d_p_node_area) cudaFree(d_p_node_area);
    if (d_node_qleave) cudaFree(d_node_qleave);
    if (d_q_cell) cudaFree(d_q_cell);
    if (d_limiter_events) cudaFree(d_limiter_events);
    if (d_limiter_volume) cudaFree(d_limiter_volume);
}

// ─────────────────────────────────────────────────────────────────────────────
// Graph management (Suggestion 9)
// ─────────────────────────────────────────────────────────────────────────────
void swe2d_gpu_enable_kernel_graphs(SWE2DDeviceState* dev, bool enable) {
    if (!dev) return;
    // Destroy any existing graph if disabling or if we're re-enabling with different config
    if (!enable || (enable && dev->kernel_graph_cache.is_valid)) {
        dev->kernel_graph_cache.destroy();
    }
    dev->enable_kernel_graphs = enable;
}

void swe2d_gpu_destroy_kernel_graphs(SWE2DDeviceState* dev) {
    if (!dev) return;
    dev->kernel_graph_cache.destroy();
}

// ─────────────────────────────────────────────────────────────────────────────
// swe2d_gpu_destroy
// ─────────────────────────────────────────────────────────────────────────────
void swe2d_gpu_destroy(SWE2DDeviceState* dev) {
    if (!dev) return;
    auto safe_free = [](void* ptr) { if (ptr) cudaFree(ptr); };

    // Advanced-mode scaffolding was removed (dead code)

    safe_free(dev->d_edge_c0);    safe_free(dev->d_edge_c1);
    safe_free(dev->d_edge_n0);    safe_free(dev->d_edge_n1);
    safe_free(dev->d_edge_nx);    safe_free(dev->d_edge_ny);
    safe_free(dev->d_edge_len);   safe_free(dev->d_edge_bc);
    safe_free(dev->d_edge_mx);    safe_free(dev->d_edge_my);
    safe_free(dev->d_edge_bc_val);
    safe_free(dev->d_cell_edge_offsets);
    safe_free(dev->d_cell_edge_ids);
    safe_free(dev->d_cell_ring2_offsets);
    safe_free(dev->d_cell_ring2_ids);
    safe_free(dev->d_cell_ring2_dcx);
    safe_free(dev->d_cell_ring2_dcy);
    safe_free(dev->d_cell_ring2_inv_dist2);
    safe_free(dev->d_hg_edge_index);
    safe_free(dev->d_hg_bc_type);
    safe_free(dev->d_hg_offsets);
    safe_free(dev->d_hg_time_s);
    safe_free(dev->d_hg_value);
    safe_free(dev->d_bc_upd_edge);
    safe_free(dev->d_bc_upd_type);
    safe_free(dev->d_bc_upd_val);
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
    safe_free(dev->d_flux_hv);    safe_free(dev->d_flux_hu_r);
    safe_free(dev->d_flux_hv_r);
    safe_free(dev->d_lambda_max);
    safe_free(dev->d_max_wse_elev_error);
    safe_free(dev->d_diag_packed);
    safe_free(dev->d_cfl_block_max);
    safe_free(dev->d_active);
    safe_free(dev->d_n_wet);
    safe_free(dev->d_bc_forced);
    safe_free(dev->d_was_active);
    safe_free(dev->d_active_edge_ids);
    safe_free(dev->d_n_active_edges);
    safe_free(dev->d_degen_mask);
    safe_free(dev->d_inv_area_repaired);
    safe_free(dev->d_merge_owner);
    safe_free(dev->d_cell_gage_idx);
    safe_free(dev->d_rain_hg_offsets);
    safe_free(dev->d_rain_hg_time_s);
    safe_free(dev->d_rain_hg_cum_mm);
    safe_free(dev->d_rain_cn);
    safe_free(dev->d_rain_cum_mm);
    safe_free(dev->d_rain_excess_cum_mm);
    safe_free(dev->d_cell_source_mps);
    safe_free(dev->d_stage_cell_source_mps);
    safe_free(dev->d_stage_edge_bc);
    safe_free(dev->d_stage_edge_bc_val);
    safe_free(dev->d_external_source_mps);
    // Coupling workspace cleanup
    {
        auto& ws = dev->coupling_ws;
        safe_free(ws.d_cell_area); safe_free(ws.d_source);
        safe_free(ws.d_inlet_cell); safe_free(ws.d_inlet_q);
        safe_free(ws.d_struct_up); safe_free(ws.d_struct_dn); safe_free(ws.d_struct_q);
        safe_free(ws.d_bridge_cell_area); safe_free(ws.d_bridge_source);
        safe_free(ws.d_bridge_up); safe_free(ws.d_bridge_dn);
        safe_free(ws.d_bridge_q); safe_free(ws.d_bridge_ku); safe_free(ws.d_bridge_kd);
    }
    // Structure flow workspace cleanup
    {
        auto& ws = dev->sf_ws;
        safe_free(ws.d_cell_wse); safe_free(ws.d_cell_bed);
        safe_free(ws.d_structure_type); safe_free(ws.d_upstream_cell); safe_free(ws.d_downstream_cell);
        safe_free(ws.d_crest_elev); safe_free(ws.d_width); safe_free(ws.d_height);
        safe_free(ws.d_diameter); safe_free(ws.d_length); safe_free(ws.d_roughness_n);
        safe_free(ws.d_coeff); safe_free(ws.d_cd); safe_free(ws.d_opening);
        safe_free(ws.d_q_pump); safe_free(ws.d_max_flow);
        safe_free(ws.d_culvert_code); safe_free(ws.d_culvert_shape);
        safe_free(ws.d_culvert_rise); safe_free(ws.d_culvert_span);
        safe_free(ws.d_culvert_area); safe_free(ws.d_culvert_barrels); safe_free(ws.d_culvert_slope);
        safe_free(ws.d_inlet_invert_elev); safe_free(ws.d_outlet_invert_elev);
        safe_free(ws.d_entrance_loss_k); safe_free(ws.d_exit_loss_k);
        safe_free(ws.d_embankment_enabled); safe_free(ws.d_embankment_crest_elev);
        safe_free(ws.d_embankment_overflow_width); safe_free(ws.d_embankment_weir_coeff);
        safe_free(ws.d_structure_flow);
        safe_free(ws.d_prev_structure_flow);
    }
    // Redistribution workspace cleanup
    dev->redist_ws.destroy();
    // Culvert face-flux workspace cleanup
    dev->culvert_ff_ws.destroy();
    if (dev->d_ext_struct_flux_h)  { cudaFree(dev->d_ext_struct_flux_h);  dev->d_ext_struct_flux_h  = nullptr; }
    if (dev->d_ext_struct_flux_hu) { cudaFree(dev->d_ext_struct_flux_hu); dev->d_ext_struct_flux_hu = nullptr; }
    if (dev->d_ext_struct_flux_hv) { cudaFree(dev->d_ext_struct_flux_hv); dev->d_ext_struct_flux_hv = nullptr; }
    if (dev->d_stream) {
        cudaStreamSynchronize(dev->d_stream);
        cudaStreamDestroy(dev->d_stream);
        dev->d_stream = nullptr;
        // Clean up CUDA graph cache
        dev->kernel_graph_cache.destroy();
    }
    delete dev;
}
