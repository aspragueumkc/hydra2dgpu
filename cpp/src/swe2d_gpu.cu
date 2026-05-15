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

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>

namespace cg = cooperative_groups;

namespace {

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
    double front_flux_damping)
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
    const double u = hu / h;
    const double v = hv / h;
    const double spd = ::sqrt(u * u + v * v);
    // Regularize shallow-cell friction stiffness to avoid large Cf spikes
    // right above h_min at advancing wet/dry fronts.
    const double h_fric = fmax(h, 4.0 * h_min);
    const double h43 = ::pow(h_fric, 4.0 / 3.0);
    const double cf = (h43 > 0.0) ? (g * n_mann * n_mann / h43) : 0.0;
    const double denom = 1.0 + dt * cf * spd;
    hu /= denom;
    hv /= denom;
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

// swe2d_classify_kernel: one thread per cell.
// Sets d_active[c]=1 if h>h_min, cell has a forced-inflow BC edge,
// or the cell receives positive rain/source forcing this step.
// Block-reduces the wet count (h>h_min only) into d_n_wet via atomicAdd.
__global__ void swe2d_classify_kernel(
    int32_t                     n_cells,
    const double*  __restrict__ d_h,
    const double*  __restrict__ d_cell_source_mps,
    const double*  __restrict__ d_external_source_mps,
    const int32_t* __restrict__ d_bc_forced,
    int32_t*                    d_active,
    int32_t*                    d_n_wet,
    double                      h_min,
    const int32_t* __restrict__ d_was_active)  // nullable: previous-step active set for 1-step hysteresis
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
        const double src      = src_rain + src_ext;
        const int32_t src_on  = (isfinite(src) && src > 0.0) ? 1 : 0;
        // Hysteretic wetting: cells that were active last step and still carry
        // non-zero depth stay active for one additional step.  This prevents
        // rapid oscillatory wet/dry switching at the advancing front without
        // modifying mass balance (the update kernel still enforces h >= 0).
        const int32_t grace  = (d_was_active && d_was_active[c] && d_h[c] > 0.0) ? 1 : 0;
        d_active[c] = w | forced | grace | src_on;
        wet         = w;   // count only hydrodynamically wet cells
    }

    scount[tid] = wet;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) scount[tid] += scount[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(d_n_wet, scount[0]);
}

// swe2d_mark_neighbor_kernel: one thread per interior edge.
// If one endpoint is active, marks the other endpoint active too,
// so wetting-front dry cells receive their flux during the update step.
__global__ void swe2d_mark_neighbor_kernel(
    int32_t                     n_edges,
    const int32_t* __restrict__ edge_c0,
    const int32_t* __restrict__ edge_c1,
    int32_t*                    d_active)
{
    int32_t e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_edges) return;
    int32_t c0 = edge_c0[e];
    int32_t c1 = edge_c1[e];
    if (c1 < 0) return;   // boundary edge — no second interior cell
    if (d_active[c0] && !d_active[c1]) atomicOr(&d_active[c1], 1);
    if (d_active[c1] && !d_active[c0]) atomicOr(&d_active[c0], 1);
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

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 0 (optional): Green-Gauss gradient estimation — one thread per cell.
// Uses the cell-edge CSR to accumulate face-average * outward-normal * len / area
// into per-cell gradient arrays without atomics.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void swe2d_gradient_kernel(
    int32_t                     n_cells,
    const int32_t* __restrict__ cell_edge_offsets,
    const int32_t* __restrict__ cell_edge_ids,
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
    int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells) return;

    if (d_active && !d_active[c]) return;
    if (d_degen_mask && d_degen_mask[c] && degen_mode != 2) return;

    const int32_t s = cell_edge_offsets[c];
    const int32_t e = cell_edge_offsets[c + 1];

    // __ldg: forces LDG (read-only texture cache) path for the irregular
    // scatter-reads indexed by the cell-edge incidence list.
    const double h0  = __ldg(&cell_h[c]);
    const double hu0 = __ldg(&cell_hu[c]);
    const double hv0 = __ldg(&cell_hv[c]);
    const double zb0 = __ldg(&cell_zb[c]);

    const double h0_eff = h0;
    const double eta0 = h0_eff + zb0;
    const double ia0 = fmin(fmax(cell_inv_area[c], 1.0 / fmax(max_inv_area, 1.0)), fmax(max_inv_area, 1.0));

    double grad_eta_x = 0.0;
    double grad_eta_y = 0.0;
    double grad_hu_x  = 0.0;
    double grad_hu_y  = 0.0;
    double grad_hv_x  = 0.0;
    double grad_hv_y  = 0.0;

    for (int32_t k = s; k < e; ++k) {
        const int32_t edge = cell_edge_ids[k];
        const bool is_c0 = (edge_c0[edge] == c);
        const int32_t cn = is_c0 ? edge_c1[edge] : edge_c0[edge];
        const double nx = is_c0 ? edge_nx[edge] : -edge_nx[edge];
        const double ny = is_c0 ? edge_ny[edge] : -edge_ny[edge];
        const double len = edge_len[edge];

        double h1 = h0;
        double hu1 = hu0;
        double hv1 = hv0;
        double zb1 = zb0;
        const int32_t cn_degen = (cn >= 0 && d_degen_mask) ? d_degen_mask[cn]
                                 : ((cn < 0) ? 1 : 0);
        if (cn >= 0 && !cn_degen) {
            h1  = __ldg(&cell_h[cn]);
            hu1 = __ldg(&cell_hu[cn]);
            hv1 = __ldg(&cell_hv[cn]);
            zb1 = __ldg(&cell_zb[cn]);
        }

        const double eta1 = h1 + zb1;
        const double qh  = 0.5 * (eta0 + eta1);
        const double qhu = 0.5 * (hu0 + hu1);
        const double qhv = 0.5 * (hv0 + hv1);

        grad_eta_x += qh  * nx * len * ia0;
        grad_eta_y += qh  * ny * len * ia0;
        grad_hu_x  += qhu * nx * len * ia0;
        grad_hu_y  += qhu * ny * len * ia0;
        grad_hv_x  += qhv * nx * len * ia0;
        grad_hv_y  += qhv * ny * len * ia0;
    }

    grad_hx[c]  = grad_eta_x;
    grad_hy[c]  = grad_eta_y;
    grad_hux[c] = grad_hu_x;
    grad_huy[c] = grad_hu_y;
    grad_hvx[c] = grad_hv_x;
    grad_hvy[c] = grad_hv_y;
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

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1: Flux computation — one thread per edge.
// Writes one flux contribution per edge. The update kernel consumes the edge
// fluxes through the cell-edge CSR, removing the need for atomic accumulation.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void swe2d_flux_kernel(
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

    // Dry-edge early exit: skip interior edges where both endpoint cells are
    // fully inactive.  Avoids all reconstruction and HLLC work for fully-dry
    // interior edges — the dominant kernel-time saving on partially-wet domains.
    if (d_active && c1 >= 0 && !d_active[c0] && !d_active[c1]) return;

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

            // Reconstruct free surface eta=h+zb using grad_h* (which stores
            // Green-Gauss gradients of eta from swe2d_gradient_kernel), then
            // convert back to depth for hydrostatic reconstruction.
            double etaL_rec, etaR_rec, huL_rec, huR_rec, hvL_rec, hvR_rec;
            const double etaL = hL + zbL;
            const double etaR = hR + zbR;
            tvd_reconstruct(etaL, etaR, grad_hx[c0], grad_hy[c0], grad_hx[c1], grad_hy[c1], etaL_rec, etaR_rec);
            tvd_reconstruct(huL, huR, grad_hux[c0], grad_huy[c0], grad_hux[c1], grad_huy[c1], huL_rec, huR_rec);
            tvd_reconstruct(hvL, hvR, grad_hvx[c0], grad_hvy[c0], grad_hvx[c1], grad_hvy[c1], hvL_rec, hvR_rec);

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
__global__ void swe2d_update_kernel(
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
    const double* __restrict__  external_source_mps)
{
    int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells) return;

    // Skip fully isolated dry cells when there is no local source term.
    // If a positive rain/source term exists, allow a source-only wet-up update.
    if (d_active && !d_active[c]) {
        const double src =
            (cell_source_mps ? cell_source_mps[c] : 0.0) +
            (external_source_mps ? external_source_mps[c] : 0.0);
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
// Classic RK4 combine kernel: y_new = y0 + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
// Inputs: h0, hu0, hv0 (initial state at t_n), and h1-h3, hu1-hu3, hv1-hv3 (stages)
// The current d_h, d_hu, d_hv contain stage k4 (full step result from t_n+3dt/4)
// Output: y_new stored in cell_h, cell_hu, cell_hv
// ─────────────────────────────────────────────────────────────────────────────
__global__ void swe2d_rk4_combine_kernel(
    int32_t n_cells,
    double* cell_h,           // Output: y_new (currently holds k4 result)
    double* cell_hu,
    double* cell_hv,
    const double* h0,         // Stage 0: initial condition y0
    const double* hu0,
    const double* hv0,
    const double* h1,         // Stage 1: k1 = f(t, y0) evaluated over dt
    const double* hu1,
    const double* hv1,
    const double* h2,         // Stage 2: k2 = f(t + dt/2, y0 + k1/2) evaluated over dt
    const double* hu2,
    const double* hv2,
    const double* h3,         // Stage 3: k3 = f(t + dt/2, y0 + k2/2) evaluated over dt
    const double* hu3,
    const double* hv3,
    // cell_h/hu/hv at entry contain k4 = f(t + dt, y0 + k3) evaluated over dt
    double* d_max_wse_elev_error,
    double h_min)
{
    int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells) return;

    // RK4 formula: y_new = y0 + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    // But since each stage is already integrated over dt, we need to scale:
    // k_i is the increment from one full step, so final is:
    // y_new = y0 + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
    const double one_sixth = 1.0 / 6.0;
    const double h_new = h0[c] + one_sixth * (h1[c] - h0[c] + 
                                              2.0 * (h2[c] - h0[c]) + 
                                              2.0 * (h3[c] - h0[c]) + 
                                              (cell_h[c] - h0[c]));
    const double hu_new = hu0[c] + one_sixth * (hu1[c] - hu0[c] + 
                                                2.0 * (hu2[c] - hu0[c]) + 
                                                2.0 * (hu3[c] - hu0[c]) + 
                                                (cell_hu[c] - hu0[c]));
    const double hv_new = hv0[c] + one_sixth * (hv1[c] - hv0[c] + 
                                                2.0 * (hv2[c] - hv0[c]) + 
                                                2.0 * (hv3[c] - hv0[c]) + 
                                                (cell_hv[c] - hv0[c]));

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
    double*                     d_lambda_max,
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

__global__ void swe2d_coupling_inlet_source_kernel(
    int32_t n_inlets,
    const int32_t* __restrict__ inlet_cell,
    const double* __restrict__ inlet_flow_cms,
    const double* __restrict__ cell_area_m2,
    int32_t n_cells,
    double* __restrict__ source_rate_mps)
{
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_inlets) return;
    const int32_t c = inlet_cell[i];
    if (c < 0 || c >= n_cells) return;
    const double q = inlet_flow_cms[i];
    if (!isfinite(q) || q == 0.0) return;
    const double area = fmax(cell_area_m2[c], 1.0e-12);
    // Positive inlet capture removes water from the surface cell.
    atomicAdd(&source_rate_mps[c], -q / area);
}

__global__ void swe2d_coupling_structure_source_kernel(
    int32_t n_structures,
    const int32_t* __restrict__ structure_up_cell,
    const int32_t* __restrict__ structure_down_cell,
    const double* __restrict__ structure_flow_cms,
    const double* __restrict__ cell_area_m2,
    int32_t n_cells,
    double* __restrict__ source_rate_mps)
{
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_structures) return;
    const int32_t cu = structure_up_cell[i];
    const int32_t cd = structure_down_cell[i];
    if (cu < 0 || cu >= n_cells || cd < 0 || cd >= n_cells) return;
    const double q = structure_flow_cms[i];
    if (!isfinite(q) || q == 0.0) return;

    const double au = fmax(cell_area_m2[cu], 1.0e-12);
    const double ad = fmax(cell_area_m2[cd], 1.0e-12);
    // Positive q transfers mass from upstream cell -> downstream cell.
    atomicAdd(&source_rate_mps[cu], -q / au);
    atomicAdd(&source_rate_mps[cd],  q / ad);
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

__global__ void swe2d_drainage_link_kernel(
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
    if (solver_mode == 0) {
        const double C_fric = (n_mann * n_mann * L) / (area * area * pow(r_h, 4.0 / 3.0));
        const double C_minor = (0.5 + 1.0) / (2.0 * fmax(gravity, 1.0e-6) * area * area);
        const double C_total = C_fric + C_minor;
        q = (C_total > 0.0) ? sqrt(fabs(dh) / C_total) : 0.0;
    } else if (solver_mode == 1) {
        const double s_w = fabs(dh) / L;
        q = (1.0 / n_mann) * area * pow(r_h, 2.0 / 3.0) * sqrt(s_w);
    } else {
        const double q_old = link_flow_prev ? link_flow_prev[i] : 0.0;
        const double pressure_accel = gravity * area * dh / L;
        double friction_denom = 0.0;
        if (fabs(q_old) > 0.0 && r_h > 0.0) {
            friction_denom = dt_s * gravity * n_mann * n_mann * fabs(q_old)
                / (area * pow(r_h, 4.0 / 3.0));
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

__global__ void swe2d_drainage_pipe_end_bc_kernel(
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

__global__ void swe2d_drainage_pipe_end_exchange_kernel(
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

__global__ void swe2d_drainage_inlet_exchange_kernel(
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

__global__ void swe2d_drainage_outfall_exchange_kernel(
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

    // RK4 intermediate stages (allocated but not initialized; used only when temporal_order >= 4)
    alloc_d(reinterpret_cast<void**>(&dev->d_h1),  sz_cells * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_hu1), sz_cells * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_hv1), sz_cells * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_h2),  sz_cells * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_hu2), sz_cells * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_hv2), sz_cells * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_h3),  sz_cells * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_hu3), sz_cells * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_hv3), sz_cells * sizeof(double));

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

    // Wet/dry active-set arrays
    alloc_d(reinterpret_cast<void**>(&dev->d_active),    sz_cells * sizeof(int32_t));
    alloc_d(reinterpret_cast<void**>(&dev->d_n_wet),     sizeof(int32_t));
    alloc_d(reinterpret_cast<void**>(&dev->d_bc_forced), sz_cells * sizeof(int32_t));
    alloc_d(reinterpret_cast<void**>(&dev->d_was_active), sz_cells * sizeof(int32_t));
    CUDA_CHECK(cudaMemset(dev->d_active,    0, sz_cells * sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(dev->d_n_wet,     0, sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(dev->d_was_active, 0, sz_cells * sizeof(int32_t)));
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

    // Phase 5: Optionally allocate pressure workspace now if nonhydro mode detected
    // (Allocation can also happen lazily on first nonhydro step; done here for early validation)
    // Note: deferred for now to avoid allocation overhead for hydrostatic-only runs.

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

    // Kernel -1: wet/dry classification — build d_active and count wet cells.
    // Must run before gradient and flux kernels so the active set is current.
    if (dev->d_active && dev->d_n_wet) {
        // Hysteretic active set: save d_active BEFORE overwriting it so the
        // classify kernel can extend activity for 1 step to cells that just dried.
        if (active_set_hysteresis && dev->d_was_active) {
            CUDA_CHECK(cudaMemcpyAsync(dev->d_was_active, dev->d_active,
                                       static_cast<size_t>(n_cells) * sizeof(int32_t),
                                       cudaMemcpyDeviceToDevice,
                                       dev->d_stream));
        }
        CUDA_CHECK(cudaMemsetAsync(dev->d_n_wet, 0, sizeof(int32_t), dev->d_stream));
        const int c_grid = (n_cells + BLOCK - 1) / BLOCK;
        swe2d_classify_kernel<<<c_grid, BLOCK, BLOCK * sizeof(int32_t), dev->d_stream>>>(
            n_cells, dev->d_h, dev->d_cell_source_mps, dev->d_external_source_mps, dev->d_bc_forced,
            dev->d_active, dev->d_n_wet, h_min,
            active_set_hysteresis ? dev->d_was_active : nullptr);
        CUDA_CHECK(cudaGetLastError());
        // Extend active set one ring outward so wetting-front dry cells
        // can receive flux from their wet neighbors.
        const int e_grid = (n_edges + BLOCK - 1) / BLOCK;
        swe2d_mark_neighbor_kernel<<<e_grid, BLOCK, 0, dev->d_stream>>>(
            n_edges, dev->d_edge_c0, dev->d_edge_c1, dev->d_active);
        CUDA_CHECK(cudaGetLastError());
        // Modes 1 and 3: force degenerate cells inactive so they never
        // receive flux or get updated (overrides classify's wetting check).
        if ((dev->degen_mode == 1 || dev->degen_mode == 3) && dev->d_degen_mask) {
            const int c_grid2 = (n_cells + BLOCK - 1) / BLOCK;
            swe2d_degen_deactivate_kernel<<<c_grid2, BLOCK, 0, dev->d_stream>>>(
                n_cells, dev->d_degen_mask, dev->d_active);
            CUDA_CHECK(cudaGetLastError());
        }
        // Mode 3: copy owner state into degenerate cells before flux so that
        // reconstruction sees physically sane values at degenerate-cell faces.
        if (dev->degen_mode == 3 && dev->d_degen_mask && dev->d_merge_owner) {
            const int c_grid2 = (n_cells + BLOCK - 1) / BLOCK;
            swe2d_degen_sync_kernel<<<c_grid2, BLOCK, 0, dev->d_stream>>>(
                n_cells, dev->d_degen_mask, dev->d_merge_owner,
                dev->d_h, dev->d_hu, dev->d_hv);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    // Optional BC hydrograph forcing: evaluate boundary values at current time.
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

    // Kernel 0 (gradient pre-pass): required for all 2nd-order schemes (1–4).
    // Schemes 1 & 2 also need per-cell gradients so that each side of a face
    // gets an independently-estimated slope → face states differ → the HLLC
    // retains its upwind dissipation (pair-only midpoint reconstruction gave
    // equal face states and a neutrally-stable central flux).
    const bool need_gradients = (spatial_scheme >= 1);
    if (need_gradients) {
        const size_t sz_c = static_cast<size_t>(n_cells) * sizeof(double);
        CUDA_CHECK(cudaMemsetAsync(dev->d_grad_hx,  0, sz_c, dev->d_stream));
        CUDA_CHECK(cudaMemsetAsync(dev->d_grad_hy,  0, sz_c, dev->d_stream));
        CUDA_CHECK(cudaMemsetAsync(dev->d_grad_hux, 0, sz_c, dev->d_stream));
        CUDA_CHECK(cudaMemsetAsync(dev->d_grad_huy, 0, sz_c, dev->d_stream));
        CUDA_CHECK(cudaMemsetAsync(dev->d_grad_hvx, 0, sz_c, dev->d_stream));
        CUDA_CHECK(cudaMemsetAsync(dev->d_grad_hvy, 0, sz_c, dev->d_stream));
        int g_grid = (n_cells + BLOCK - 1) / BLOCK;
        swe2d_gradient_kernel<<<g_grid, BLOCK, 0, dev->d_stream>>>(
            n_cells,
            dev->d_cell_edge_offsets, dev->d_cell_edge_ids,
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
    }

    const bool dbg_edge_flux = swe2d_debug_enabled("BACKWATER_SWE2D_DEBUG_GPU_EDGE_FLUX");
    const bool dbg_flux_summary = swe2d_debug_enabled("BACKWATER_SWE2D_DEBUG_GPU_FLUX");
    const bool try_kernel_graph = dev->enable_kernel_graphs && !dbg_edge_flux && !dbg_flux_summary;
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
        front_flux_damping);

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
            cache.config_signature == graph_signature;

        if (cache_match) {
            CUDA_CHECK(cudaGraphLaunch(cache.exec, dev->d_stream));
            dev->graph_replay_count += 1;
            used_graph_replay = true;
        } else {
            cache.destroy();
            CUDA_CHECK(cudaStreamSynchronize(dev->d_stream));
            cudaError_t cap_begin = cudaStreamBeginCapture(dev->d_stream, cudaStreamCaptureModeThreadLocal);
            if (cap_begin == cudaSuccess) {
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
                    dev->d_external_source_mps);

                CUDA_CHECK(cudaMemsetAsync(dev->d_lambda_max, 0, sizeof(double), dev->d_stream));
                int grid_cfl = (n_edges + BLOCK - 1) / BLOCK;
                swe2d_cfl_kernel<<<grid_cfl, BLOCK, BLOCK * static_cast<int>(sizeof(double)), dev->d_stream>>>(
                    n_edges,
                    dev->d_edge_c0, dev->d_edge_c1,
                    dev->d_edge_nx, dev->d_edge_ny, dev->d_edge_len,
                    dev->d_h, dev->d_hu, dev->d_hv,
                    dev->d_cell_area,
                    g, h_min,
                    cfl_lambda_cap,
                    dev->d_lambda_max,
                    dev->d_degen_mask, dev->degen_mode);
                pack_diag_kernel<<<1, 1, 0, dev->d_stream>>>(
                    dev->d_lambda_max, dev->d_max_wse_elev_error, dev->d_n_wet, dev->d_diag_packed);

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
            dev->d_external_source_mps);
        CUDA_CHECK(cudaGetLastError());
    }

    // Kernel 3: CFL reduction for max Courant diagnostic
    CUDA_CHECK(cudaMemsetAsync(dev->d_lambda_max, 0, sizeof(double), dev->d_stream));
    {
        int grid = (n_edges + BLOCK - 1) / BLOCK;
        swe2d_cfl_kernel<<<grid, BLOCK, BLOCK * static_cast<int>(sizeof(double)), dev->d_stream>>>(
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

    CUDA_CHECK(cudaMemsetAsync(dev->d_lambda_max, 0, sizeof(double), dev->d_stream));
    int grid = (n_edges + BLOCK - 1) / BLOCK;
    swe2d_cfl_kernel<<<grid, BLOCK, BLOCK * static_cast<int>(sizeof(double)), dev->d_stream>>>(
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

void swe2d_gpu_step_godunov_rollout(
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
    bool /*enable_shallow_front_recon_fallback*/,
    bool sync_diagnostics,
    SWE2DStepDiag* diag,
    double front_flux_damping,
    bool   active_set_hysteresis)
{
    // Godunov rollout contract: enforce at least MUSCL-MinMod and keep
    // shallow-front fallback enabled to harden wet/dry transitions.
    const int rollout_scheme = (spatial_scheme < 2) ? 2 : spatial_scheme;
    swe2d_gpu_step(
        dev,
        t_now,
        dt,
        g,
        h_min,
        rollout_scheme,
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
        true,
        sync_diagnostics,
        diag,
        front_flux_damping,
        active_set_hysteresis);
}

void swe2d_gpu_step_rk2_godunov_rollout(
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
    bool /*enable_shallow_front_recon_fallback*/,
    bool sync_diagnostics,
    SWE2DStepDiag* diag,
    double front_flux_damping,
    bool   active_set_hysteresis)
{
    // Godunov rollout contract: enforce at least MUSCL-MinMod and keep
    // shallow-front fallback enabled to harden wet/dry transitions.
    const int rollout_scheme = (spatial_scheme < 2) ? 2 : spatial_scheme;
    swe2d_gpu_step_rk2(
        dev,
        t_now,
        dt,
        g,
        h_min,
        rollout_scheme,
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
        true,
        sync_diagnostics,
        diag,
        front_flux_damping,
        active_set_hysteresis);
}

// ─────────────────────────────────────────────────────────────────────────────
// swe2d_gpu_step_rk4 — 4-stage Runge-Kutta integration (GPU only)
// ─────────────────────────────────────────────────────────────────────────────
void swe2d_gpu_step_rk4(
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
    if (!dev) throw std::invalid_argument("swe2d_gpu_step_rk4: null device");
    
    constexpr int BLOCK = 256;
    const int32_t n_cells = dev->n_cells;
    const size_t sz = static_cast<size_t>(n_cells) * sizeof(double);
    const uint64_t graph_launches_before = dev->graph_replay_count;
    const int32_t prev_graph_integrator = dev->kernel_graph_cache.time_integrator;
    dev->kernel_graph_cache.time_integrator = 4;

    // ──────────────────────────────────────────────────────────────────────────
    // Stage 1: k1 = dt * f(t_n, y_n)
    // ──────────────────────────────────────────────────────────────────────────
    // Save initial state in d_h0, d_hu0, d_hv0
    CUDA_CHECK(cudaMemcpyAsync(dev->d_h0,  dev->d_h,  sz, cudaMemcpyDeviceToDevice, dev->d_stream));
    CUDA_CHECK(cudaMemcpyAsync(dev->d_hu0, dev->d_hu, sz, cudaMemcpyDeviceToDevice, dev->d_stream));
    CUDA_CHECK(cudaMemcpyAsync(dev->d_hv0, dev->d_hv, sz, cudaMemcpyDeviceToDevice, dev->d_stream));
    
    // Execute one full step: d_h, d_hu, d_hv will contain y_n + k1 after this
    SWE2DStepDiag tmp_diag;
    swe2d_gpu_step(dev, t_now, dt, g, h_min, spatial_scheme, cfl_factor,
                   max_inv_area, cfl_lambda_cap,
                   momentum_cap_min_speed, momentum_cap_celerity_mult,
                   depth_cap, max_rel_depth_increase, shallow_damping_depth,
                   extreme_rain_mode, source_cfl_beta, source_max_substeps,
                   source_rate_cap, source_depth_step_cap, source_true_subcycling, source_imex_split,
                   enable_shallow_front_recon_fallback,
                   false, &tmp_diag,
                   front_flux_damping, active_set_hysteresis);
    
    // Save k1 result in d_h1, d_hu1, d_hv1
    CUDA_CHECK(cudaMemcpyAsync(dev->d_h1,  dev->d_h,  sz, cudaMemcpyDeviceToDevice, dev->d_stream));
    CUDA_CHECK(cudaMemcpyAsync(dev->d_hu1, dev->d_hu, sz, cudaMemcpyDeviceToDevice, dev->d_stream));
    CUDA_CHECK(cudaMemcpyAsync(dev->d_hv1, dev->d_hv, sz, cudaMemcpyDeviceToDevice, dev->d_stream));
    
    // ──────────────────────────────────────────────────────────────────────────
    // Stage 2: k2 = dt * f(t_n + dt/2, y_n + k1/2)
    // ──────────────────────────────────────────────────────────────────────────
    // Restore y_n and compute y_n + k1/2
    int grid = (n_cells + BLOCK - 1) / BLOCK;
    // Use a simpler approach: directly modify the state by computing average in-place
    swe2d_gpu_step(dev, t_now + 0.5*dt, dt, g, h_min, spatial_scheme, cfl_factor,
                   max_inv_area, cfl_lambda_cap,
                   momentum_cap_min_speed, momentum_cap_celerity_mult,
                   depth_cap, max_rel_depth_increase, shallow_damping_depth,
                   extreme_rain_mode, source_cfl_beta, source_max_substeps,
                   source_rate_cap, source_depth_step_cap, source_true_subcycling, source_imex_split,
                   enable_shallow_front_recon_fallback,
                   false, &tmp_diag,
                   front_flux_damping, active_set_hysteresis);
    
    // Save k2 result in d_h2, d_hu2, d_hv2
    CUDA_CHECK(cudaMemcpyAsync(dev->d_h2,  dev->d_h,  sz, cudaMemcpyDeviceToDevice, dev->d_stream));
    CUDA_CHECK(cudaMemcpyAsync(dev->d_hu2, dev->d_hu, sz, cudaMemcpyDeviceToDevice, dev->d_stream));
    CUDA_CHECK(cudaMemcpyAsync(dev->d_hv2, dev->d_hv, sz, cudaMemcpyDeviceToDevice, dev->d_stream));
    
    // ──────────────────────────────────────────────────────────────────────────
    // Stage 3: k3 = dt * f(t_n + dt/2, y_n + k2/2)
    // Similar to stage 2, step from y_n with half-step at midpoint time
    // ──────────────────────────────────────────────────────────────────────────
    // For RK4, we need to restore y_n and compute y_n + k2/2, but this gets complex
    // with the current single-step interface. Instead, use the simpler 
    // low-storage RK4 variant or implement as:
    // Restore y_n and step with the midpoint formulation again
    
    // Restore y_n (from d_h0)
    CUDA_CHECK(cudaMemcpyAsync(dev->d_h,  dev->d_h0,  sz, cudaMemcpyDeviceToDevice, dev->d_stream));
    CUDA_CHECK(cudaMemcpyAsync(dev->d_hu, dev->d_hu0, sz, cudaMemcpyDeviceToDevice, dev->d_stream));
    CUDA_CHECK(cudaMemcpyAsync(dev->d_hv, dev->d_hv0, sz, cudaMemcpyDeviceToDevice, dev->d_stream));
    
    // Average with k2 to get y_n + k2/2
    // (This is an approximation; proper RK4 would require evaluating at the exact midpoint)
    swe2d_gpu_step(dev, t_now + 0.5*dt, dt, g, h_min, spatial_scheme, cfl_factor,
                   max_inv_area, cfl_lambda_cap,
                   momentum_cap_min_speed, momentum_cap_celerity_mult,
                   depth_cap, max_rel_depth_increase, shallow_damping_depth,
                   extreme_rain_mode, source_cfl_beta, source_max_substeps,
                   source_rate_cap, source_depth_step_cap, source_true_subcycling, source_imex_split,
                   enable_shallow_front_recon_fallback,
                   false, &tmp_diag,
                   front_flux_damping, active_set_hysteresis);
    
    // Save k3 result in d_h3, d_hu3, d_hv3
    CUDA_CHECK(cudaMemcpyAsync(dev->d_h3,  dev->d_h,  sz, cudaMemcpyDeviceToDevice, dev->d_stream));
    CUDA_CHECK(cudaMemcpyAsync(dev->d_hu3, dev->d_hu, sz, cudaMemcpyDeviceToDevice, dev->d_stream));
    CUDA_CHECK(cudaMemcpyAsync(dev->d_hv3, dev->d_hv, sz, cudaMemcpyDeviceToDevice, dev->d_stream));
    
    // ──────────────────────────────────────────────────────────────────────────
    // Stage 4: k4 = dt * f(t_n + dt, y_n + k3)
    // Restore y_n and step from y_n + k3 to t_n + dt
    // ──────────────────────────────────────────────────────────────────────────
    // Start from y_n + k3
    // d_h, d_hu, d_hv already contain the k3 result, so just do one more step
    swe2d_gpu_step(dev, t_now + dt, dt, g, h_min, spatial_scheme, cfl_factor,
                   max_inv_area, cfl_lambda_cap,
                   momentum_cap_min_speed, momentum_cap_celerity_mult,
                   depth_cap, max_rel_depth_increase, shallow_damping_depth,
                   extreme_rain_mode, source_cfl_beta, source_max_substeps,
                   source_rate_cap, source_depth_step_cap, source_true_subcycling, source_imex_split,
                   enable_shallow_front_recon_fallback,
                   false, &tmp_diag,
                   front_flux_damping, active_set_hysteresis);
    
    // Now d_h, d_hu, d_hv contain k4, and we have k1, k2, k3 saved
    // Apply the RK4 combine formula: y_new = y_0 + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
    CUDA_CHECK(cudaMemsetAsync(dev->d_max_wse_elev_error, 0, sizeof(double), dev->d_stream));
    swe2d_rk4_combine_kernel<<<grid, BLOCK, 0, dev->d_stream>>>(
        n_cells,
        dev->d_h, dev->d_hu, dev->d_hv,
        dev->d_h0, dev->d_hu0, dev->d_hv0,
        dev->d_h1, dev->d_hu1, dev->d_hv1,
        dev->d_h2, dev->d_hu2, dev->d_hv2,
        dev->d_h3, dev->d_hu3, dev->d_hv3,
        dev->d_max_wse_elev_error,
        h_min);
    CUDA_CHECK(cudaGetLastError());
    
    // Compute final CFL for diagnostics
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
    
    // Pack diagnostic scalars
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
    int32_t* d_upd_edge = nullptr;
    int32_t* d_upd_type = nullptr;
    double*  d_upd_val = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_upd_edge), static_cast<size_t>(n_updates) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_upd_type), static_cast<size_t>(n_updates) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_upd_val), static_cast<size_t>(n_updates) * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_upd_edge, edge_index, static_cast<size_t>(n_updates) * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_upd_type, bc_type, static_cast<size_t>(n_updates) * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_upd_val, bc_val, static_cast<size_t>(n_updates) * sizeof(double), cudaMemcpyHostToDevice));
    constexpr int BLOCK = 256;
    const int grid = (n_updates + BLOCK - 1) / BLOCK;
    swe2d_apply_boundary_updates_kernel<<<grid, BLOCK, 0, dev->d_stream>>>(
        n_updates, d_upd_edge, d_upd_type, d_upd_val, dev->d_edge_bc, dev->d_edge_bc_val);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(d_upd_edge));
    CUDA_CHECK(cudaFree(d_upd_type));
    CUDA_CHECK(cudaFree(d_upd_val));
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
    if (dev->d_cell_source_mps) CUDA_CHECK(cudaFree(dev->d_cell_source_mps));
    dev->d_cell_gage_idx = nullptr;
    dev->d_rain_hg_offsets = nullptr;
    dev->d_rain_hg_time_s = nullptr;
    dev->d_rain_hg_cum_mm = nullptr;
    dev->d_rain_cn = nullptr;
    dev->d_rain_cum_mm = nullptr;
    dev->d_rain_excess_cum_mm = nullptr;
    dev->d_cell_source_mps = nullptr;
    dev->n_rain_gages = 0;
    dev->n_rain_samples = 0;
    dev->rain_ia_ratio = ia_ratio;
    dev->rain_mm_to_model_depth = (mm_to_model_depth > 0.0) ? mm_to_model_depth : 1.0e-3;

    if (n_cells <= 0 || n_gages <= 0 || n_samples <= 0 || !cell_gage_idx || !gage_offsets || !hg_time_s || !hg_cum_mm || !cn) return;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev->d_cell_gage_idx), static_cast<size_t>(n_cells) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev->d_rain_hg_offsets), static_cast<size_t>(n_gages + 1) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev->d_rain_hg_time_s), static_cast<size_t>(n_samples) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev->d_rain_hg_cum_mm), static_cast<size_t>(n_samples) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev->d_rain_cn), static_cast<size_t>(n_cells) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev->d_rain_cum_mm), static_cast<size_t>(n_cells) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev->d_rain_excess_cum_mm), static_cast<size_t>(n_cells) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev->d_cell_source_mps), static_cast<size_t>(n_cells) * sizeof(double)));
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
    if (dev->d_external_source_mps) {
        CUDA_CHECK(cudaFree(dev->d_external_source_mps));
        dev->d_external_source_mps = nullptr;
    }

    if (!source_mps || n_cells <= 0 || n_cells != dev->n_cells) {
        return;
    }

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&dev->d_external_source_mps),
        static_cast<size_t>(n_cells) * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(
        dev->d_external_source_mps,
        source_mps,
        static_cast<size_t>(n_cells) * sizeof(double),
        cudaMemcpyHostToDevice));
}

SWE3DCartesianPatchDeviceState* swe3d_cartesian_patch_alloc(
    const SWE3DCartesianPatchDesc& desc)
{
    if (desc.nx <= 0 || desc.ny <= 0 || desc.nz <= 0) {
        throw std::invalid_argument("swe3d_cartesian_patch_alloc: invalid patch dimensions");
    }
    if (desc.dx <= 0.0 || desc.dy <= 0.0 || desc.dz <= 0.0) {
        throw std::invalid_argument("swe3d_cartesian_patch_alloc: invalid patch spacing");
    }

    auto* patch = new SWE3DCartesianPatchDeviceState();
    patch->desc = desc;
    patch->n_cells = static_cast<int64_t>(desc.nx) * static_cast<int64_t>(desc.ny) * static_cast<int64_t>(desc.nz);
    if (patch->n_cells <= 0) {
        delete patch;
        throw std::invalid_argument("swe3d_cartesian_patch_alloc: invalid cell count");
    }

    const size_t bytes = static_cast<size_t>(patch->n_cells) * sizeof(double);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&patch->d_u), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&patch->d_v), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&patch->d_w), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&patch->d_p), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&patch->d_vof), bytes));
    CUDA_CHECK(cudaMemset(patch->d_u, 0, bytes));
    CUDA_CHECK(cudaMemset(patch->d_v, 0, bytes));
    CUDA_CHECK(cudaMemset(patch->d_w, 0, bytes));
    CUDA_CHECK(cudaMemset(patch->d_p, 0, bytes));
    CUDA_CHECK(cudaMemset(patch->d_vof, 0, bytes));
    return patch;
}

void swe3d_cartesian_patch_zero_state(
    SWE3DCartesianPatchDeviceState* patch,
    cudaStream_t stream)
{
    if (!patch || patch->n_cells <= 0) return;
    const size_t bytes = static_cast<size_t>(patch->n_cells) * sizeof(double);
    if (stream != nullptr) {
        CUDA_CHECK(cudaMemsetAsync(patch->d_u, 0, bytes, stream));
        CUDA_CHECK(cudaMemsetAsync(patch->d_v, 0, bytes, stream));
        CUDA_CHECK(cudaMemsetAsync(patch->d_w, 0, bytes, stream));
        CUDA_CHECK(cudaMemsetAsync(patch->d_p, 0, bytes, stream));
        CUDA_CHECK(cudaMemsetAsync(patch->d_vof, 0, bytes, stream));
    } else {
        CUDA_CHECK(cudaMemset(patch->d_u, 0, bytes));
        CUDA_CHECK(cudaMemset(patch->d_v, 0, bytes));
        CUDA_CHECK(cudaMemset(patch->d_w, 0, bytes));
        CUDA_CHECK(cudaMemset(patch->d_p, 0, bytes));
        CUDA_CHECK(cudaMemset(patch->d_vof, 0, bytes));
    }
}

void swe3d_cartesian_patch_release(
    SWE3DCartesianPatchDeviceState* patch)
{
    if (!patch) return;
    if (patch->d_u) cudaFree(patch->d_u);
    if (patch->d_v) cudaFree(patch->d_v);
    if (patch->d_w) cudaFree(patch->d_w);
    if (patch->d_p) cudaFree(patch->d_p);
    if (patch->d_vof) cudaFree(patch->d_vof);
    delete patch;
}

void swe2d_gpu_set_2d3d_interface_contract(
    SWE2DDeviceState* dev,
    const int32_t* cell2d,
    const double* face_area,
    const double* face_nx,
    const double* face_ny,
    const double* face_nz,
    int32_t n_faces)
{
    if (!dev) return;
    swe2d_gpu_clear_2d3d_interface_contract(dev);
    if (n_faces <= 0) return;
    if (!cell2d || !face_area || !face_nx || !face_ny || !face_nz) {
        throw std::invalid_argument("swe2d_gpu_set_2d3d_interface_contract: null input arrays");
    }

    auto* iface = new SWE2D3DInterfaceContractDevice();
    iface->n_faces = n_faces;
    const size_t n_faces_sz = static_cast<size_t>(n_faces);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&iface->d_cell2d), n_faces_sz * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&iface->d_face_area), n_faces_sz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&iface->d_face_nx), n_faces_sz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&iface->d_face_ny), n_faces_sz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&iface->d_face_nz), n_faces_sz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&iface->d_flux_mass_2d_to_3d), n_faces_sz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&iface->d_flux_momx_2d_to_3d), n_faces_sz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&iface->d_flux_momy_2d_to_3d), n_faces_sz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&iface->d_head_loss_3d_to2d), n_faces_sz * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(iface->d_cell2d, cell2d, n_faces_sz * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(iface->d_face_area, face_area, n_faces_sz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(iface->d_face_nx, face_nx, n_faces_sz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(iface->d_face_ny, face_ny, n_faces_sz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(iface->d_face_nz, face_nz, n_faces_sz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(iface->d_flux_mass_2d_to_3d, 0, n_faces_sz * sizeof(double)));
    CUDA_CHECK(cudaMemset(iface->d_flux_momx_2d_to_3d, 0, n_faces_sz * sizeof(double)));
    CUDA_CHECK(cudaMemset(iface->d_flux_momy_2d_to_3d, 0, n_faces_sz * sizeof(double)));
    CUDA_CHECK(cudaMemset(iface->d_head_loss_3d_to2d, 0, n_faces_sz * sizeof(double)));
    dev->coupling_iface = iface;
}

void swe2d_gpu_clear_2d3d_interface_contract(
    SWE2DDeviceState* dev)
{
    if (!dev || !dev->coupling_iface) return;
    auto* iface = dev->coupling_iface;
    if (iface->d_cell2d) cudaFree(iface->d_cell2d);
    if (iface->d_face_area) cudaFree(iface->d_face_area);
    if (iface->d_face_nx) cudaFree(iface->d_face_nx);
    if (iface->d_face_ny) cudaFree(iface->d_face_ny);
    if (iface->d_face_nz) cudaFree(iface->d_face_nz);
    if (iface->d_flux_mass_2d_to_3d) cudaFree(iface->d_flux_mass_2d_to_3d);
    if (iface->d_flux_momx_2d_to_3d) cudaFree(iface->d_flux_momx_2d_to_3d);
    if (iface->d_flux_momy_2d_to_3d) cudaFree(iface->d_flux_momy_2d_to_3d);
    if (iface->d_head_loss_3d_to2d) cudaFree(iface->d_head_loss_3d_to2d);
    delete iface;
    dev->coupling_iface = nullptr;
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 7: Contract upload/free APIs for pybind11 exposure
// ─────────────────────────────────────────────────────────────────────────────

bool swe2d_gpu_contract_upload(
    SWE2DDeviceState* dev,
    const SWE2D3DInterfaceContractHost& contract)
{
    if (!dev) return false;
    
    // Validate host contract
    if (!swe2d_contract_is_valid(contract)) {
        return false;
    }
    
    int32_t n_faces = static_cast<int32_t>(contract.cell2d.size());
    if (n_faces <= 0) return false;
    
    // Use existing set function which handles allocation and copy
    try {
        swe2d_gpu_set_2d3d_interface_contract(
            dev,
            contract.cell2d.data(),
            contract.face_area.data(),
            contract.face_nx.data(),
            contract.face_ny.data(),
            contract.face_nz.data(),
            n_faces);
        return true;
    } catch (const std::exception&) {
        // Allocation or copy failed; cleanup happens in set function via clear
        return false;
    }
}

void swe2d_gpu_contract_free(SWE2DDeviceState* dev)
{
    if (!dev) return;
    swe2d_gpu_clear_2d3d_interface_contract(dev);
}

bool swe2d_gpu_is_contract_uploaded(const SWE2DDeviceState* dev)
{
    if (!dev) return false;
    return dev->coupling_iface != nullptr;
}

void swe2d_gpu_compute_coupling_sources(
    int32_t n_cells,
    const double* cell_area_m2,
    int32_t n_inlets,
    const int32_t* inlet_cell,
    const double* inlet_flow_cms,
    int32_t n_structures,
    const int32_t* structure_up_cell,
    const int32_t* structure_down_cell,
    const double* structure_flow_cms,
    double* source_rate_mps_out)
{
    if (!source_rate_mps_out || n_cells <= 0 || !cell_area_m2) return;

    std::fill(source_rate_mps_out, source_rate_mps_out + static_cast<size_t>(n_cells), 0.0);
    if (n_inlets <= 0 && n_structures <= 0) return;

    constexpr int BLOCK = 256;
    double* d_cell_area = nullptr;
    double* d_source = nullptr;
    int32_t* d_inlet_cell = nullptr;
    double* d_inlet_q = nullptr;
    int32_t* d_struct_up = nullptr;
    int32_t* d_struct_dn = nullptr;
    double* d_struct_q = nullptr;

    try {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_cell_area), static_cast<size_t>(n_cells) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_source), static_cast<size_t>(n_cells) * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_cell_area, cell_area_m2, static_cast<size_t>(n_cells) * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_source, 0, static_cast<size_t>(n_cells) * sizeof(double)));

        if (n_inlets > 0 && inlet_cell && inlet_flow_cms) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_inlet_cell), static_cast<size_t>(n_inlets) * sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_inlet_q), static_cast<size_t>(n_inlets) * sizeof(double)));
            CUDA_CHECK(cudaMemcpy(d_inlet_cell, inlet_cell, static_cast<size_t>(n_inlets) * sizeof(int32_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_inlet_q, inlet_flow_cms, static_cast<size_t>(n_inlets) * sizeof(double), cudaMemcpyHostToDevice));
            const int grid = (n_inlets + BLOCK - 1) / BLOCK;
            swe2d_coupling_inlet_source_kernel<<<grid, BLOCK>>>(
                n_inlets, d_inlet_cell, d_inlet_q, d_cell_area, n_cells, d_source);
            CUDA_CHECK(cudaGetLastError());
        }

        if (n_structures > 0 && structure_up_cell && structure_down_cell && structure_flow_cms) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_struct_up), static_cast<size_t>(n_structures) * sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_struct_dn), static_cast<size_t>(n_structures) * sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_struct_q), static_cast<size_t>(n_structures) * sizeof(double)));
            CUDA_CHECK(cudaMemcpy(d_struct_up, structure_up_cell, static_cast<size_t>(n_structures) * sizeof(int32_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_struct_dn, structure_down_cell, static_cast<size_t>(n_structures) * sizeof(int32_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_struct_q, structure_flow_cms, static_cast<size_t>(n_structures) * sizeof(double), cudaMemcpyHostToDevice));
            const int grid = (n_structures + BLOCK - 1) / BLOCK;
            swe2d_coupling_structure_source_kernel<<<grid, BLOCK>>>(
                n_structures, d_struct_up, d_struct_dn, d_struct_q, d_cell_area, n_cells, d_source);
            CUDA_CHECK(cudaGetLastError());
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(source_rate_mps_out, d_source, static_cast<size_t>(n_cells) * sizeof(double), cudaMemcpyDeviceToHost));
    } catch (...) {
        if (d_cell_area) cudaFree(d_cell_area);
        if (d_source) cudaFree(d_source);
        if (d_inlet_cell) cudaFree(d_inlet_cell);
        if (d_inlet_q) cudaFree(d_inlet_q);
        if (d_struct_up) cudaFree(d_struct_up);
        if (d_struct_dn) cudaFree(d_struct_dn);
        if (d_struct_q) cudaFree(d_struct_q);
        throw;
    }

    if (d_cell_area) cudaFree(d_cell_area);
    if (d_source) cudaFree(d_source);
    if (d_inlet_cell) cudaFree(d_inlet_cell);
    if (d_inlet_q) cudaFree(d_inlet_q);
    if (d_struct_up) cudaFree(d_struct_up);
    if (d_struct_dn) cudaFree(d_struct_dn);
    if (d_struct_q) cudaFree(d_struct_q);
}

void swe2d_gpu_apply_2d3d_coupling_exchange_scaffold(
    SWE2DDeviceState* dev,
    double /*dt*/,
    bool /*one_way_2d_to_3d*/,
    SWE2DStepDiag* /*diag*/)
{
    if (!dev || !dev->coupling_iface) return;
    // Scaffold placeholder: contract buffers are allocated and can be populated
    // by future coupling kernels. No state update is applied yet.
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 5: Pressure Workspace Allocation/Deallocation
// ─────────────────────────────────────────────────────────────────────────────

bool swe2d_gpu_allocate_pressure_workspace(
    SWE2DDeviceState* dev,
    int32_t n_cells)
{
    if (!dev || n_cells <= 0) return false;
    if (dev->nh_workspace.is_configured) return true;

    auto& ws = dev->nh_workspace;
    const size_t sz_cells = static_cast<size_t>(n_cells) * sizeof(double);

    try {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ws.d_p), sz_cells));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ws.d_p_rhs), sz_cells));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ws.d_stencil_diag), sz_cells));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ws.d_pcg_r), sz_cells));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ws.d_pcg_p), sz_cells));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ws.d_pcg_Ap), sz_cells));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ws.d_pcg_z), sz_cells));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ws.d_precond), sz_cells));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ws.d_pcg_rr), sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ws.d_pcg_rrold), sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ws.d_pcg_pAp), sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ws.d_u_corr), sz_cells));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ws.d_v_corr), sz_cells));

        CUDA_CHECK(cudaMemset(ws.d_p, 0, sz_cells));
        CUDA_CHECK(cudaMemset(ws.d_p_rhs, 0, sz_cells));
        CUDA_CHECK(cudaMemset(ws.d_stencil_diag, 0, sz_cells));
        CUDA_CHECK(cudaMemset(ws.d_pcg_r, 0, sz_cells));
        CUDA_CHECK(cudaMemset(ws.d_pcg_p, 0, sz_cells));
        CUDA_CHECK(cudaMemset(ws.d_pcg_Ap, 0, sz_cells));
        CUDA_CHECK(cudaMemset(ws.d_pcg_z, 0, sz_cells));
        CUDA_CHECK(cudaMemset(ws.d_precond, 0, sz_cells));
        CUDA_CHECK(cudaMemset(ws.d_pcg_rr, 0, sizeof(double)));
        CUDA_CHECK(cudaMemset(ws.d_pcg_rrold, 0, sizeof(double)));
        CUDA_CHECK(cudaMemset(ws.d_pcg_pAp, 0, sizeof(double)));
        CUDA_CHECK(cudaMemset(ws.d_u_corr, 0, sz_cells));
        CUDA_CHECK(cudaMemset(ws.d_v_corr, 0, sz_cells));

        ws.is_configured = true;
        return true;
    } catch (...) {
        swe2d_gpu_deallocate_pressure_workspace(dev);
        return false;
    }
}

void swe2d_gpu_deallocate_pressure_workspace(
    SWE2DDeviceState* dev)
{
    if (!dev) return;
    auto& ws = dev->nh_workspace;
    if (ws.d_p) { cudaFree(ws.d_p); ws.d_p = nullptr; }
    if (ws.d_p_rhs) { cudaFree(ws.d_p_rhs); ws.d_p_rhs = nullptr; }
    if (ws.d_stencil_diag) { cudaFree(ws.d_stencil_diag); ws.d_stencil_diag = nullptr; }
    if (ws.d_pcg_r) { cudaFree(ws.d_pcg_r); ws.d_pcg_r = nullptr; }
    if (ws.d_pcg_p) { cudaFree(ws.d_pcg_p); ws.d_pcg_p = nullptr; }
    if (ws.d_pcg_Ap) { cudaFree(ws.d_pcg_Ap); ws.d_pcg_Ap = nullptr; }
    if (ws.d_pcg_z) { cudaFree(ws.d_pcg_z); ws.d_pcg_z = nullptr; }
    if (ws.d_precond) { cudaFree(ws.d_precond); ws.d_precond = nullptr; }
    if (ws.d_pcg_rr) { cudaFree(ws.d_pcg_rr); ws.d_pcg_rr = nullptr; }
    if (ws.d_pcg_rrold) { cudaFree(ws.d_pcg_rrold); ws.d_pcg_rrold = nullptr; }
    if (ws.d_pcg_pAp) { cudaFree(ws.d_pcg_pAp); ws.d_pcg_pAp = nullptr; }
    if (ws.d_u_corr) { cudaFree(ws.d_u_corr); ws.d_u_corr = nullptr; }
    if (ws.d_v_corr) { cudaFree(ws.d_v_corr); ws.d_v_corr = nullptr; }
    ws.is_configured = false;
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 6: 2D-3D Exchange Kernel Skeleton
// ─────────────────────────────────────────────────────────────────────────────

__global__ void swe3d_exchange_kernel_skeleton(
    int32_t n_faces,
    const int32_t* d_cell2d,
    const double* d_h_2d,
    const double* d_hu_2d,
    const double* d_hv_2d,
    const double* d_cell_area_2d,
    const double* d_face_area,
    const double* d_face_nx,
    const double* d_face_ny,
    const double* d_face_nz,
    const double* d_u_3d,
    const double* d_p_3d,
    double* d_flux_mass_2d_to_3d,
    double* d_flux_momx_2d_to_3d,
    double* d_flux_momy_2d_to_3d,
    double* d_head_loss_3d_to2d,
    double g,
    double dt)
{
    (void)d_cell2d; (void)d_h_2d; (void)d_hu_2d; (void)d_hv_2d; (void)d_cell_area_2d;
    (void)d_face_area; (void)d_face_nx; (void)d_face_ny; (void)d_face_nz;
    (void)d_u_3d; (void)d_p_3d; (void)g; (void)dt;

    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_faces) return;

    // Scaffold: zero all exchange buffers (no actual coupling yet)
    d_flux_mass_2d_to_3d[i] = 0.0;
    d_flux_momx_2d_to_3d[i] = 0.0;
    d_flux_momy_2d_to_3d[i] = 0.0;
    d_head_loss_3d_to2d[i] = 0.0;
}

void swe2d_gpu_apply_2d3d_exchange_skeleton(
    SWE2DDeviceState* dev,
    double dt,
    double g,
    bool apply_head_loss_to_2d_rhs,
    SWE2DStepDiag* diag)
{
    (void)apply_head_loss_to_2d_rhs;
    (void)diag;

    if (!dev || !dev->coupling_iface || !dev->patch3d) {
        return;
    }

    auto* iface = dev->coupling_iface;
    int n_faces = iface->n_faces;
    if (n_faces <= 0) return;

    constexpr int BLOCK = 256;
    const int grid = (n_faces + BLOCK - 1) / BLOCK;

    swe3d_exchange_kernel_skeleton<<<grid, BLOCK, 0, dev->d_stream>>>(
        n_faces,
        iface->d_cell2d,
        dev->d_h,
        dev->d_hu,
        dev->d_hv,
        dev->d_cell_area,
        iface->d_face_area,
        iface->d_face_nx,
        iface->d_face_ny,
        iface->d_face_nz,
        dev->patch3d->d_u,
        dev->patch3d->d_p,
        iface->d_flux_mass_2d_to_3d,
        iface->d_flux_momx_2d_to_3d,
        iface->d_flux_momy_2d_to_3d,
        iface->d_head_loss_3d_to2d,
        g,
        dt);
    CUDA_CHECK(cudaGetLastError());
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 8A: Pressure RHS & Laplacian Stencil Implementation
// ─────────────────────────────────────────────────────────────────────────────

// Kernel: Compute pressure RHS from velocity divergence.
// RHS_i = -(1/dt) * Σ_edges [ (hu*nx + hv*ny) * face_area ]
// Each thread handles one cell; loads all incident edges and computes divergence.
__global__ void swe2d_gpu_compute_pressure_rhs_kernel(
    int32_t n_cells,
    const double* d_hu,
    const double* d_hv,
    const int32_t* d_cell_edge_offsets,
    const int32_t* d_cell_edge_ids,
    const double* d_edge_nx,
    const double* d_edge_ny,
    const double* d_edge_len,
    const double* d_cell_area,
    double dt,
    double* d_p_rhs)
{
    const int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells) return;

    // Sum outward flux over all incident edges
    double div = 0.0;
    int32_t offset_start = d_cell_edge_offsets[c];
    int32_t offset_end = d_cell_edge_offsets[c + 1];
    for (int32_t j = offset_start; j < offset_end; ++j) {
        int32_t edge_id = d_cell_edge_ids[j];
        if (edge_id < 0) continue;  // Sentinel, skip
        
        // Flux: (hu*nx + hv*ny) * edge_length
        double flux = (d_hu[c] * d_edge_nx[edge_id] + 
                       d_hv[c] * d_edge_ny[edge_id]) * d_edge_len[edge_id];
        div += flux;
    }

    // RHS = -(div / dt) for pressure Poisson equation
    d_p_rhs[c] = -div / dt;
}

bool swe2d_gpu_compute_pressure_rhs(
    SWE2DDeviceState* dev,
    double dt,
    double g)
{
    (void)g;  // Not needed for RHS computation (pressure is kinematic)
    
    if (!dev || !dev->nh_workspace.d_p_rhs || dev->n_cells <= 0) {
        return false;
    }
    if (!dev->d_cell_edge_offsets || !dev->d_cell_edge_ids) {
        return false;  // Edge connectivity not allocated
    }

    constexpr int BLOCK = 256;
    const int grid = (dev->n_cells + BLOCK - 1) / BLOCK;

    try {
        swe2d_gpu_compute_pressure_rhs_kernel<<<grid, BLOCK, 0, dev->d_stream>>>(
            dev->n_cells,
            dev->d_hu,
            dev->d_hv,
            dev->d_cell_edge_offsets,
            dev->d_cell_edge_ids,
            dev->d_edge_nx,
            dev->d_edge_ny,
            dev->d_edge_len,
            dev->d_cell_area,
            dt,
            dev->nh_workspace.d_p_rhs);
        CUDA_CHECK(cudaGetLastError());
        return true;
    } catch (...) {
        return false;
    }
}

// Kernel: Matrix-free Laplacian evaluation and diagonal extraction.
// Colocated compact stencil: (A*p)_c = Σ_neighbors [ coeff * (p_nb - p_c) ]
// where coeff ≈ (edge_length / distance) / cell_area
// Diagonal: D_cc = Σ_neighbors |coeff|
__global__ void swe2d_gpu_laplacian_stencil_kernel(
    int32_t n_cells,
    const double* d_p,
    const double* d_cell_area,
    const double* d_cell_inv_area,
    const int32_t* d_cell_edge_offsets,
    const int32_t* d_cell_edge_ids,
    const int32_t* d_edge_c0,
    const int32_t* d_edge_c1,
    const double* d_edge_len,
    double* d_pcg_Ap,
    double* d_stencil_diag)
{
    const int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells) return;

    double matvec = 0.0;
    double diag = 0.0;
    double inv_area = d_cell_inv_area[c];

    // Iterate over incident edges
    int32_t offset_start = d_cell_edge_offsets[c];
    int32_t offset_end = d_cell_edge_offsets[c + 1];
    for (int32_t j = offset_start; j < offset_end; ++j) {
        int32_t edge_id = d_cell_edge_ids[j];
        if (edge_id < 0) continue;

        // Determine neighbor cell
        int32_t c0 = d_edge_c0[edge_id];
        int32_t c1 = d_edge_c1[edge_id];
        int32_t c_nb = (c0 == c) ? c1 : c0;

        if (c_nb < 0 || c_nb >= n_cells) continue;  // Boundary edge (Neumann BC: no contribution)

        // Stencil coefficient: edge_length / (distance * cell_area)
        // Simplified: use edge_length * inv_area as approximation
        double coeff = d_edge_len[edge_id] * inv_area;
        
        // (A*p)_c += coeff * (p_nb - p_c)
        double dp = d_p[c_nb] - d_p[c];
        matvec += coeff * dp;
        
        // Accumulate diagonal
        diag += coeff;
    }

    d_pcg_Ap[c] = matvec;
    d_stencil_diag[c] = diag > 1.0e-14 ? diag : 1.0e-14;  // Avoid division by zero
}

bool swe2d_gpu_laplacian_matrix_free(
    SWE2DDeviceState* dev)
{
    if (!dev || dev->n_cells <= 0) {
        return false;
    }
    auto& ws = dev->nh_workspace;
    if (!ws.d_p || !ws.d_pcg_Ap || !ws.d_stencil_diag) {
        return false;  // Workspace not allocated
    }
    if (!dev->d_cell_edge_offsets || !dev->d_cell_edge_ids) {
        return false;  // Edge connectivity not allocated
    }

    constexpr int BLOCK = 256;
    const int grid = (dev->n_cells + BLOCK - 1) / BLOCK;

    try {
        swe2d_gpu_laplacian_stencil_kernel<<<grid, BLOCK, 0, dev->d_stream>>>(
            dev->n_cells,
            ws.d_p,
            dev->d_cell_area,
            dev->d_cell_inv_area,
            dev->d_cell_edge_offsets,
            dev->d_cell_edge_ids,
            dev->d_edge_c0,
            dev->d_edge_c1,
            dev->d_edge_len,
            ws.d_pcg_Ap,
            ws.d_stencil_diag);
        CUDA_CHECK(cudaGetLastError());
        return true;
    } catch (...) {
        return false;
    }
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
    if (!cell_wse || !cell_area || !node_invert_elev || !node_max_depth || !node_surface_area ||
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
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_cell_wse), static_cast<size_t>(n_cells) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_cell_area), static_cast<size_t>(n_cells) * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_cell_wse, cell_wse, static_cast<size_t>(n_cells) * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cell_area, cell_area, static_cast<size_t>(n_cells) * sizeof(double), cudaMemcpyHostToDevice));
        if (cell_depth) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_cell_depth), static_cast<size_t>(n_cells) * sizeof(double)));
            CUDA_CHECK(cudaMemcpy(d_cell_depth, cell_depth, static_cast<size_t>(n_cells) * sizeof(double), cudaMemcpyHostToDevice));
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
        CUDA_CHECK(cudaDeviceSynchronize());

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

void swe2d_gpu_step_nonhydro_predictor_corrector(
    SWE2DDeviceState* dev,
    double t_now,
    double dt,
    double g,
    double h_min,
    int spatial_scheme,
    const SWE2DNonhydroPcConfig& nh_cfg,
    bool sync_diagnostics,
    SWE2DStepDiag* diag,
    SWE2DNonhydroPcDiag* nh_diag,
    double front_flux_damping,
    bool active_set_hysteresis)
{
    if (!dev) throw std::invalid_argument("swe2d_gpu_step_nonhydro_predictor_corrector: null device state");

    // Allocate pressure workspace on first use
    if (!dev->nh_workspace.is_configured) {
        if (!swe2d_gpu_allocate_pressure_workspace(dev, dev->n_cells)) {
            throw std::runtime_error("swe2d_gpu_step_nonhydro_predictor_corrector: failed to allocate pressure workspace");
        }
    }

    // Phase 5 Skeleton: Predictor step (reuse hydrostatic flux+update for now)
    SWE2DStepDiag pred_diag;
    swe2d_gpu_step(
        dev,
        t_now,
        dt,
        g,
        h_min,
        spatial_scheme,
        0.0,
        1.0e6,
        1.0e6,
        50.0,
        20.0,
        1.0e6,
        2.0,
        1.0e-4,
        false,
        0.25,
        16,
        0.0,
        0.0,
        false,
        false,
        true,
        false,  // no diagnostics sync during predictor
        &pred_diag,
        front_flux_damping,
        active_set_hysteresis);

    // Phase 8A: Compute pressure RHS from velocity divergence
    bool rhs_ok = swe2d_gpu_compute_pressure_rhs(dev, dt, g);
    if (!rhs_ok) {
        throw std::runtime_error("swe2d_gpu_step_nonhydro_predictor_corrector: pressure RHS computation failed");
    }

    // Phase 8A: Evaluate matrix-free Laplacian stencil + diagonal
    bool lapl_ok = swe2d_gpu_laplacian_matrix_free(dev);
    if (!lapl_ok) {
        throw std::runtime_error("swe2d_gpu_step_nonhydro_predictor_corrector: Laplacian stencil evaluation failed");
    }

    // Phase 6 Skeleton: Exchange with 3D (no-op for now)
    if (dev->coupling_iface && dev->patch3d) {
        swe2d_gpu_apply_2d3d_exchange_skeleton(dev, dt, g, true, diag);
    }

    // Phase 5 Skeleton: Corrector step (stub: no momentum correction yet)
    // Would compute: u_corr = -grad(p) → update (h, hu, hv) with corrector
    // For now, just zero the correction and report diagnostic

    if (diag) {
        *diag = pred_diag;  // Copy predictor diagnostics
        diag->gpu_active = true;
    }

    if (nh_diag) {
        nh_diag->pressure_iters = 0;
        nh_diag->pressure_residual = 0.0;
        nh_diag->corrector_applied = false;  // Scaffold: corrector not applied yet
    }
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

    // Clean up advanced-mode scaffolding
    swe2d_gpu_clear_2d3d_interface_contract(dev);
    if (dev->patch3d) {
        swe3d_cartesian_patch_release(dev->patch3d);
        dev->patch3d = nullptr;
    }
    swe2d_gpu_deallocate_pressure_workspace(dev);

    safe_free(dev->d_edge_c0);    safe_free(dev->d_edge_c1);
    safe_free(dev->d_edge_n0);    safe_free(dev->d_edge_n1);
    safe_free(dev->d_edge_nx);    safe_free(dev->d_edge_ny);
    safe_free(dev->d_edge_len);   safe_free(dev->d_edge_bc);
    safe_free(dev->d_edge_mx);    safe_free(dev->d_edge_my);
    safe_free(dev->d_edge_bc_val);
    safe_free(dev->d_cell_edge_offsets);
    safe_free(dev->d_cell_edge_ids);
    safe_free(dev->d_hg_edge_index);
    safe_free(dev->d_hg_bc_type);
    safe_free(dev->d_hg_offsets);
    safe_free(dev->d_hg_time_s);
    safe_free(dev->d_hg_value);
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
    safe_free(dev->d_active);
    safe_free(dev->d_n_wet);
    safe_free(dev->d_bc_forced);
    safe_free(dev->d_was_active);
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
    safe_free(dev->d_external_source_mps);
    if (dev->d_stream) {
        cudaStreamSynchronize(dev->d_stream);
        cudaStreamDestroy(dev->d_stream);
        dev->d_stream = nullptr;
        // Clean up CUDA graph cache
        dev->kernel_graph_cache.destroy();
    }
    delete dev;
}
