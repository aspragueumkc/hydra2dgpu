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

// ── Module-scope persistent state (used by culvert and coupling paths) ──
extern int32_t s_culvert_solver_mode;
extern double* s_culvert_table_header;
extern double* s_culvert_table_data;
extern int32_t s_culvert_table_n_hw;
extern int32_t s_culvert_table_n_tw;
extern SWE2DDeviceState* s_coupling_dev;

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

enum class SWE3DAdaptiveDtMode : int32_t {
    ADVECTIVE_ONLY = 0,
    ADVECTIVE_PLUS_GRAVITY_WAVE = 1,
    ADVECTIVE_GRAVITY_PROJECTION = 2,
};

struct SWE3DRuntimeControls {
    int32_t adaptive_dt_mode = static_cast<int32_t>(SWE3DAdaptiveDtMode::ADVECTIVE_ONLY);
    double gravity_wave_cfl = 0.35;
    double velocity_soft_cap_cfl = 16.0;
    double closed_box_full_wet_dt_cfl = 0.16;
    int32_t projection_jacobi_max_iters = 4096;
    int32_t projection_residual_sample_iters = 1;
    double predictor_damping_coeff = 0.05;
    double free_surface_gauge_tolerance_pa = 3000.0;
    bool projection_reject_enable = false;
    bool projection_fail_fast = false;
    bool projection_divergence_gate_enable = false;
    double projection_divergence_ratio_target = 1.0;
    double projection_correction_scale_min = 1.5;
    double projection_correction_scale_max = 1.5;
    // Pressure-update residual target used by projection telemetry/retry checks.
    // 1e-2 keeps default diagnostics in a practical range for current GPU solves.
    double projection_residual_target = 1.0e-2;
    double projection_dt_reduction = 0.5;
    int32_t projection_max_retries = 3;
    double projection_min_dt_factor = 0.05;
    int32_t vof_max_substeps = 8;
    bool state_reject_enable = false;
    double state_vof_bounds_tol = 1.0e-5;
    double state_max_abs_velocity = 100.0;
    double state_max_abs_pressure = 1.0e6;
    double active_alpha_wet = 0.98;
    double active_alpha_gas = 1.0e-5;
    bool vof_transport_debug = false;
    int32_t outflow_policy = 0;
    double free_surface_vent_bias = 1.0;
    int32_t q_inflow_area_policy = 0;
    double open_bc_damping = 0.0;
    int32_t projection_boundary_policy = 0;
};

int32_t swe3d_env_int_bounded(const char* name, int32_t fallback, int32_t vmin, int32_t vmax) {
    const char* raw = std::getenv(name);
    if (!raw || !raw[0]) return fallback;
    char* end = nullptr;
    long parsed = std::strtol(raw, &end, 10);
    if (end == raw) return fallback;
    if (parsed < static_cast<long>(vmin)) parsed = static_cast<long>(vmin);
    if (parsed > static_cast<long>(vmax)) parsed = static_cast<long>(vmax);
    return static_cast<int32_t>(parsed);
}

double swe3d_env_double_bounded(const char* name, double fallback, double vmin, double vmax) {
    const char* raw = std::getenv(name);
    if (!raw || !raw[0]) return fallback;
    char* end = nullptr;
    const double parsed = std::strtod(raw, &end);
    if (end == raw || !std::isfinite(parsed)) return fallback;
    return fmin(vmax, fmax(vmin, parsed));
}

SWE3DRuntimeControls swe3d_load_runtime_controls() {
    SWE3DRuntimeControls cfg;
    cfg.adaptive_dt_mode = swe3d_env_int_bounded(
        "BACKWATER_SWE3D_ADAPTIVE_DT_MODE",
        cfg.adaptive_dt_mode,
        static_cast<int32_t>(SWE3DAdaptiveDtMode::ADVECTIVE_ONLY),
        static_cast<int32_t>(SWE3DAdaptiveDtMode::ADVECTIVE_GRAVITY_PROJECTION));
    cfg.gravity_wave_cfl = swe3d_env_double_bounded(
        "BACKWATER_SWE3D_GRAVITY_WAVE_CFL",
        cfg.gravity_wave_cfl,
        1.0e-4,
        10.0);
    cfg.velocity_soft_cap_cfl = swe3d_env_double_bounded(
        "BACKWATER_SWE3D_VELOCITY_SOFT_CAP_CFL",
        cfg.velocity_soft_cap_cfl,
        0.0,
        1.0e6);
    cfg.closed_box_full_wet_dt_cfl = swe3d_env_double_bounded(
        "BACKWATER_SWE3D_CLOSED_BOX_FULL_WET_DT_CFL",
        cfg.closed_box_full_wet_dt_cfl,
        1.0e-4,
        10.0);
    cfg.projection_jacobi_max_iters = swe3d_env_int_bounded(
        "BACKWATER_SWE3D_PROJECTION_JACOBI_MAX_ITERS",
        cfg.projection_jacobi_max_iters,
        8,
        65536);
    cfg.projection_residual_sample_iters = swe3d_env_int_bounded(
        "BACKWATER_SWE3D_PROJECTION_RESIDUAL_SAMPLE_ITERS",
        cfg.projection_residual_sample_iters,
        1,
        1024);
    cfg.predictor_damping_coeff = swe3d_env_double_bounded(
        "BACKWATER_SWE3D_PREDICTOR_DAMPING_COEFF",
        cfg.predictor_damping_coeff,
        0.0,
        1.0e6);
    cfg.free_surface_gauge_tolerance_pa = swe3d_env_double_bounded(
        "BACKWATER_SWE3D_FREE_SURFACE_GAUGE_TOLERANCE_PA",
        cfg.free_surface_gauge_tolerance_pa,
        0.0,
        1.0e9);
    cfg.projection_reject_enable =
        swe3d_env_int_bounded(
            "BACKWATER_SWE3D_PROJECTION_REJECT_ENABLE",
            cfg.projection_reject_enable ? 1 : 0,
            0,
            1) != 0;
    cfg.projection_fail_fast =
        swe3d_env_int_bounded(
            "BACKWATER_SWE3D_PROJECTION_FAIL_FAST",
            cfg.projection_fail_fast ? 1 : 0,
            0,
            1) != 0;
    cfg.projection_divergence_gate_enable =
        swe3d_env_int_bounded(
            "BACKWATER_SWE3D_PROJECTION_DIVERGENCE_GATE_ENABLE",
            cfg.projection_divergence_gate_enable ? 1 : 0,
            0,
            1) != 0;
    cfg.projection_divergence_ratio_target = swe3d_env_double_bounded(
        "BACKWATER_SWE3D_PROJECTION_DIVERGENCE_RATIO_TARGET",
        cfg.projection_divergence_ratio_target,
        1.0e-6,
        100.0);
    cfg.projection_correction_scale_min = swe3d_env_double_bounded(
        "BACKWATER_SWE3D_PROJECTION_CORRECTION_SCALE_MIN",
        cfg.projection_correction_scale_min,
        0.1,
        4.0);
    cfg.projection_correction_scale_max = swe3d_env_double_bounded(
        "BACKWATER_SWE3D_PROJECTION_CORRECTION_SCALE_MAX",
        cfg.projection_correction_scale_max,
        0.1,
        4.0);
    if (cfg.projection_correction_scale_max < cfg.projection_correction_scale_min) {
        cfg.projection_correction_scale_max = cfg.projection_correction_scale_min;
    }
    cfg.projection_residual_target = swe3d_env_double_bounded(
        "BACKWATER_SWE3D_PROJECTION_RESIDUAL_TARGET",
        cfg.projection_residual_target,
        1.0e-12,
        1.0);
    cfg.projection_dt_reduction = swe3d_env_double_bounded(
        "BACKWATER_SWE3D_PROJECTION_DT_REDUCTION",
        cfg.projection_dt_reduction,
        0.05,
        0.99);
    cfg.projection_max_retries = swe3d_env_int_bounded(
        "BACKWATER_SWE3D_PROJECTION_MAX_RETRIES",
        cfg.projection_max_retries,
        0,
        32);
    cfg.projection_min_dt_factor = swe3d_env_double_bounded(
        "BACKWATER_SWE3D_PROJECTION_MIN_DT_FACTOR",
        cfg.projection_min_dt_factor,
        1.0e-4,
        1.0);
    cfg.vof_max_substeps = swe3d_env_int_bounded(
        "BACKWATER_SWE3D_VOF_MAX_SUBSTEPS",
        cfg.vof_max_substeps,
        1,
        256);
    cfg.state_reject_enable =
        swe3d_env_int_bounded(
            "BACKWATER_SWE3D_STATE_REJECT_ENABLE",
            cfg.state_reject_enable ? 1 : 0,
            0,
            1) != 0;
    cfg.state_vof_bounds_tol = swe3d_env_double_bounded(
        "BACKWATER_SWE3D_STATE_VOF_BOUNDS_TOL",
        cfg.state_vof_bounds_tol,
        0.0,
        1.0e-1);
    cfg.state_max_abs_velocity = swe3d_env_double_bounded(
        "BACKWATER_SWE3D_STATE_MAX_ABS_VELOCITY",
        cfg.state_max_abs_velocity,
        1.0e-3,
        1.0e8);
    cfg.state_max_abs_pressure = swe3d_env_double_bounded(
        "BACKWATER_SWE3D_STATE_MAX_ABS_PRESSURE",
        cfg.state_max_abs_pressure,
        1.0e-3,
        1.0e12);
    cfg.active_alpha_wet = swe3d_env_double_bounded(
        "BACKWATER_SWE3D_ACTIVE_ALPHA_WET",
        cfg.active_alpha_wet,
        0.5,
        1.0);
    cfg.active_alpha_gas = swe3d_env_double_bounded(
        "BACKWATER_SWE3D_ACTIVE_ALPHA_GAS",
        cfg.active_alpha_gas,
        0.0,
        0.25);
    cfg.vof_transport_debug =
        swe3d_env_int_bounded(
            "BACKWATER_SWE3D_VOF_TRANSPORT_DEBUG",
            cfg.vof_transport_debug ? 1 : 0,
            0,
            1) != 0;
    cfg.outflow_policy = swe3d_env_int_bounded(
        "BACKWATER_SWE3D_OUTFLOW_POLICY",
        cfg.outflow_policy,
        0,
        1);
    cfg.free_surface_vent_bias = swe3d_env_double_bounded(
        "BACKWATER_SWE3D_FREE_SURFACE_VENT_BIAS",
        cfg.free_surface_vent_bias,
        -100.0,
        100.0);
    cfg.q_inflow_area_policy = swe3d_env_int_bounded(
        "BACKWATER_SWE3D_Q_INFLOW_AREA_POLICY",
        cfg.q_inflow_area_policy,
        0,
        1);
    cfg.open_bc_damping = swe3d_env_double_bounded(
        "BACKWATER_SWE3D_OPEN_BC_DAMPING",
        cfg.open_bc_damping,
        0.0,
        1.0);
    cfg.projection_boundary_policy = swe3d_env_int_bounded(
        "BACKWATER_SWE3D_PROJECTION_BOUNDARY_POLICY",
        cfg.projection_boundary_policy,
        0,
        1);
    return cfg;
}

SWE3DRuntimeControls swe3d_load_runtime_controls_cached() {
    // Cache parsed controls and refresh only when relevant env strings change.
    static std::mutex cache_mutex;
    static bool cache_ready = false;
    static uint64_t cache_sig = 0;
    static SWE3DRuntimeControls cache_cfg;

    auto mix_cstr = [](uint64_t h, const char* s) {
        const unsigned char* p = reinterpret_cast<const unsigned char*>(s ? s : "");
        while (*p) {
            h = swe2d_mix_u64(h, static_cast<uint64_t>(*p));
            ++p;
        }
        return h;
    };

    uint64_t sig = 1469598103934665603ULL;
    sig = mix_cstr(sig, std::getenv("BACKWATER_SWE3D_ADAPTIVE_DT_MODE"));
    sig = mix_cstr(sig, std::getenv("BACKWATER_SWE3D_GRAVITY_WAVE_CFL"));
    sig = mix_cstr(sig, std::getenv("BACKWATER_SWE3D_PROJECTION_JACOBI_MAX_ITERS"));
    sig = mix_cstr(sig, std::getenv("BACKWATER_SWE3D_PROJECTION_RESIDUAL_SAMPLE_ITERS"));
    sig = mix_cstr(sig, std::getenv("BACKWATER_SWE3D_PREDICTOR_DAMPING_COEFF"));
    sig = mix_cstr(sig, std::getenv("BACKWATER_SWE3D_FREE_SURFACE_GAUGE_TOLERANCE_PA"));
    sig = mix_cstr(sig, std::getenv("BACKWATER_SWE3D_PROJECTION_REJECT_ENABLE"));
    sig = mix_cstr(sig, std::getenv("BACKWATER_SWE3D_PROJECTION_FAIL_FAST"));
    sig = mix_cstr(sig, std::getenv("BACKWATER_SWE3D_PROJECTION_DIVERGENCE_GATE_ENABLE"));
    sig = mix_cstr(sig, std::getenv("BACKWATER_SWE3D_PROJECTION_DIVERGENCE_RATIO_TARGET"));
    sig = mix_cstr(sig, std::getenv("BACKWATER_SWE3D_PROJECTION_CORRECTION_SCALE_MIN"));
    sig = mix_cstr(sig, std::getenv("BACKWATER_SWE3D_PROJECTION_CORRECTION_SCALE_MAX"));
    sig = mix_cstr(sig, std::getenv("BACKWATER_SWE3D_PROJECTION_RESIDUAL_TARGET"));
    sig = mix_cstr(sig, std::getenv("BACKWATER_SWE3D_PROJECTION_DT_REDUCTION"));
    sig = mix_cstr(sig, std::getenv("BACKWATER_SWE3D_PROJECTION_MAX_RETRIES"));
    sig = mix_cstr(sig, std::getenv("BACKWATER_SWE3D_PROJECTION_MIN_DT_FACTOR"));
    sig = mix_cstr(sig, std::getenv("BACKWATER_SWE3D_VOF_MAX_SUBSTEPS"));
    sig = mix_cstr(sig, std::getenv("BACKWATER_SWE3D_STATE_REJECT_ENABLE"));
    sig = mix_cstr(sig, std::getenv("BACKWATER_SWE3D_STATE_VOF_BOUNDS_TOL"));
    sig = mix_cstr(sig, std::getenv("BACKWATER_SWE3D_STATE_MAX_ABS_VELOCITY"));
    sig = mix_cstr(sig, std::getenv("BACKWATER_SWE3D_STATE_MAX_ABS_PRESSURE"));
    sig = mix_cstr(sig, std::getenv("BACKWATER_SWE3D_ACTIVE_ALPHA_WET"));
    sig = mix_cstr(sig, std::getenv("BACKWATER_SWE3D_ACTIVE_ALPHA_GAS"));
    sig = mix_cstr(sig, std::getenv("BACKWATER_SWE3D_VOF_TRANSPORT_DEBUG"));
    sig = mix_cstr(sig, std::getenv("BACKWATER_SWE3D_OUTFLOW_POLICY"));
    sig = mix_cstr(sig, std::getenv("BACKWATER_SWE3D_FREE_SURFACE_VENT_BIAS"));
    sig = mix_cstr(sig, std::getenv("BACKWATER_SWE3D_Q_INFLOW_AREA_POLICY"));
    sig = mix_cstr(sig, std::getenv("BACKWATER_SWE3D_OPEN_BC_DAMPING"));
    sig = mix_cstr(sig, std::getenv("BACKWATER_SWE3D_PROJECTION_BOUNDARY_POLICY"));

    std::lock_guard<std::mutex> lock(cache_mutex);
    if (!cache_ready || sig != cache_sig) {
        cache_cfg = swe3d_load_runtime_controls();
        cache_sig = sig;
        cache_ready = true;
    }
    return cache_cfg;
}


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

// swe2d_classify_kernel: standalone classification (no mark-neighbor).
// Retained for backward compatibility with persistent-chunk / RK step functions.
__global__ void swe2d_classify_kernel(
    int32_t                     n_cells,
    const double*  __restrict__ d_h,
    const double*  __restrict__ d_cell_source_mps,
    const double*  __restrict__ d_external_source_mps,
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
        const double src      = src_rain + src_ext;
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
    if (c1 < 0) return;
    if (d_active[c0] && !d_active[c1]) atomicOr(&d_active[c1], 1);
    if (d_active[c1] && !d_active[c0]) atomicOr(&d_active[c0], 1);
}

// swe2d_classify_and_mark_kernel: fused classify + 1-hop neighbor marking.
// One thread per cell for classification; after block-level sync, each block
// also processes a contiguous segment of edges to mark dry neighbors of active
// cells.  This eliminates a separate kernel launch for the mark-neighbor pass.
__global__ void swe2d_classify_and_mark_kernel(
    int32_t                     n_cells,
    int32_t                     n_edges,
    const double*  __restrict__ d_h,
    const double*  __restrict__ d_cell_source_mps,
    const double*  __restrict__ d_external_source_mps,
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
        const double src      = src_rain + src_ext;
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
__global__ void swe2d_gradient_kernel(
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
        const int scheme_weno3  = static_cast<int>(SWE2DSpatialScheme::FV_WENO3_LIKE);
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

            // WENO3-like helper on unstructured cell pairs:
            // blend GG-extrapolated state with pair-midpoint state using
            // nonlinear smoothness weights, then enforce pair-bounds clamp.
            auto weno3_like_reconstruct = [&](double q0, double q1,
                                              double gx0, double gy0,
                                              double gx1, double gy1,
                                              double& qL_out, double& qR_out) {
                const double dq = q1 - q0;
                const double dxL = fx - cell_cx[c0];
                const double dyL = fy - cell_cy[c0];
                const double dxR = fx - cell_cx[c1];
                const double dyR = fy - cell_cy[c1];

                const double pL_grad = q0 + (gx0 * dxL + gy0 * dyL);
                const double pR_grad = q1 + (gx1 * dxR + gy1 * dyR);
                const double pL_mid  = q0 + 0.5 * dq;
                const double pR_mid  = q1 - 0.5 * dq;

                const double scale = q0 * q0 + q1 * q1 + dq * dq;
                const double eps_weno = 1.0e-20 + 1.0e-12 * fmax(1.0, scale);
                const double betaL0 = (pL_grad - q0) * (pL_grad - q0);
                const double betaL1 = dq * dq;
                const double betaR0 = (pR_grad - q1) * (pR_grad - q1);
                const double betaR1 = dq * dq;

                // Jump-aware linear weights:
                // favor GG reconstruction in smooth regions; reduce its weight
                // near strong local jumps for additional robustness.
                const double jump_ratio = fabs(dq) / (fabs(q0) + fabs(q1) + 1.0e-12);
                const double d0 = (jump_ratio < 0.25) ? (2.0 / 3.0) : 0.52;
                const double d1 = 1.0 - d0;

                const double aL0 = d0 / ((eps_weno + betaL0) * (eps_weno + betaL0));
                const double aL1 = d1 / ((eps_weno + betaL1) * (eps_weno + betaL1));
                const double sumL = aL0 + aL1;
                const double wL0 = (sumL > 0.0) ? (aL0 / sumL) : d0;
                const double wL1 = (sumL > 0.0) ? (aL1 / sumL) : d1;

                const double aR0 = d0 / ((eps_weno + betaR0) * (eps_weno + betaR0));
                const double aR1 = d1 / ((eps_weno + betaR1) * (eps_weno + betaR1));
                const double sumR = aR0 + aR1;
                const double wR0 = (sumR > 0.0) ? (aR0 / sumR) : d0;
                const double wR1 = (sumR > 0.0) ? (aR1 / sumR) : d1;

                const double rawL = wL0 * pL_grad + wL1 * pL_mid;
                const double rawR = wR0 * pR_grad + wR1 * pR_mid;
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
            if (spatial_scheme == scheme_weno3) {
                weno3_like_reconstruct(etaL, etaR, grad_hx[c0], grad_hy[c0], grad_hx[c1], grad_hy[c1], etaL_rec, etaR_rec);
                weno3_like_reconstruct(huL, huR, grad_hux[c0], grad_huy[c0], grad_hux[c1], grad_huy[c1], huL_rec, huR_rec);
                weno3_like_reconstruct(hvL, hvR, grad_hvx[c0], grad_hvy[c0], grad_hvx[c1], grad_hvy[c1], hvL_rec, hvR_rec);
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
// RK4 helper kernels.
// We keep the intermediate stages device-resident and only combine on the GPU:
//   1. capture half-step increments k_i = 2 * (y_half - y_stage)
//   2. build the next stage state from y0 and the captured increment
//   3. final combine y_{n+1} = y0 + (1/6) * (k1 + 2k2 + 2k3 + k4)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void swe2d_rk4_capture_increment_kernel(
    int32_t n_cells,
    double* dst_h,
    double* dst_hu,
    double* dst_hv,
    const double* stage_h,
    const double* stage_hu,
    const double* stage_hv,
    const double* base_h,
    const double* base_hu,
    const double* base_hv)
{
    int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells) return;

    dst_h[c]  = 2.0 * (stage_h[c]  - base_h[c]);
    dst_hu[c] = 2.0 * (stage_hu[c] - base_hu[c]);
    dst_hv[c] = 2.0 * (stage_hv[c] - base_hv[c]);
}

__global__ void swe2d_rk4_build_stage_kernel(
    int32_t n_cells,
    double* dst_h,
    double* dst_hu,
    double* dst_hv,
    const double* base_h,
    const double* base_hu,
    const double* base_hv,
    const double* inc_h,
    const double* inc_hu,
    const double* inc_hv,
    double inc_scale)
{
    int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells) return;

    dst_h[c]  = base_h[c]  + inc_scale * inc_h[c];
    dst_hu[c] = base_hu[c] + inc_scale * inc_hu[c];
    dst_hv[c] = base_hv[c] + inc_scale * inc_hv[c];
}

__global__ void swe2d_rk4_shift_from_reference_kernel(
    int32_t n_cells,
    double* dst_h,
    double* dst_hu,
    double* dst_hv,
    const double* base_h,
    const double* base_hu,
    const double* base_hv,
    const double* current_h,
    const double* current_hu,
    const double* current_hv,
    const double* ref_h,
    const double* ref_hu,
    const double* ref_hv,
    double scale)
{
    int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells) return;

    dst_h[c]  = base_h[c]  + scale * (current_h[c]  - ref_h[c]);
    dst_hu[c] = base_hu[c] + scale * (current_hu[c] - ref_hu[c]);
    dst_hv[c] = base_hv[c] + scale * (current_hv[c] - ref_hv[c]);
}

__global__ void swe2d_rk4_combine_kernel(
    int32_t n_cells,
    double* cell_h,           // Output: y_new (currently holds k4 result)
    double* cell_hu,
    double* cell_hv,
    const double* h0,         // Stage 0: initial condition y0
    const double* hu0,
    const double* hv0,
    const double* k1,         // Stage 1 increment over dt/2, scaled to full dt
    const double* hu1,
    const double* hv1,
    const double* k2,         // Stage 2 increment over dt/2, scaled to full dt
    const double* hu2,
    const double* hv2,
    const double* stage4_h,   // Stage 4 state: y0 + k3
    const double* hu3,
    const double* hv3,
    // cell_h/hu/hv at entry contain y4_half = stage4_h + 0.5 * k4
    double* d_max_wse_elev_error,
    double h_min)
{
    int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells) return;

    const double k3_h  = stage4_h[c]  - h0[c];
    const double k3_hu = hu3[c]       - hu0[c];
    const double k3_hv = hv3[c]       - hv0[c];
    const double k4_h  = 2.0 * (cell_h[c]  - stage4_h[c]);
    const double k4_hu = 2.0 * (cell_hu[c] - hu3[c]);
    const double k4_hv = 2.0 * (cell_hv[c] - hv3[c]);

    const double one_sixth = 1.0 / 6.0;
    const double h_new = h0[c] + one_sixth * (k1[c] + 2.0 * k2[c] + 2.0 * k3_h + k4_h);
    const double hu_new = hu0[c] + one_sixth * (hu1[c] + 2.0 * hu2[c] + 2.0 * k3_hu + k4_hu);
    const double hv_new = hv0[c] + one_sixth * (hv1[c] + 2.0 * hv2[c] + 2.0 * k3_hv + k4_hv);

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
// RK4 graph-safe helpers: pure L(U) collector and 4-stage Butcher-tableau combine.
// These support temporal_order=5 (swe2d_gpu_step_rk4_graph).
// ─────────────────────────────────────────────────────────────────────────────

// Accumulates the flux divergence + source into a k-vector (slope × dt).
// Does NOT update cell state, does NOT apply friction or positivity.
// Safe to capture in a CUDA graph: pointer addresses are fixed across steps.
__global__ void swe2d_rk4_rhs_collect_kernel(
    int32_t                     n_cells,
    const int32_t* __restrict__ cell_edge_offsets,
    const int32_t* __restrict__ cell_edge_ids,
    const int32_t* __restrict__ edge_c0,
    const int32_t* __restrict__ edge_c1,
    double*                     k_h,
    double*                     k_hu,
    double*                     k_hv,
    const double*  __restrict__ flux_h,
    const double*  __restrict__ flux_hu,
    const double*  __restrict__ flux_hv,
    const double*  __restrict__ flux_hu_r,
    const double*  __restrict__ flux_hv_r,
    const double*  __restrict__ cell_inv_area,
    const double*  __restrict__ cell_source_mps,
    const double*  __restrict__ external_source_mps,
    const int32_t* __restrict__ d_active,
    const int32_t* __restrict__ d_degen_mask,
    const double*  __restrict__ d_inv_area_repaired,
    int                         degen_mode,
    double                      max_inv_area,
    double                      dt)
{
    int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells) return;

    // Inactive cells contribute zero slope.
    if (d_active && !d_active[c]) {
        k_h[c] = 0.0; k_hu[c] = 0.0; k_hv[c] = 0.0;
        return;
    }
    // Degen modes 1 and 3: skip entirely (flux was dropped/redirected).
    if (d_degen_mask && d_degen_mask[c] && degen_mode != 2) {
        k_h[c] = 0.0; k_hu[c] = 0.0; k_hv[c] = 0.0;
        return;
    }

    double inv_a;
    if (degen_mode == 2 && d_inv_area_repaired && d_degen_mask && d_degen_mask[c]) {
        inv_a = d_inv_area_repaired[c];
    } else {
        inv_a = cell_inv_area[c];
        const double max_inv_a = fmax(max_inv_area, 1.0);
        if (inv_a > max_inv_a) inv_a = max_inv_a;
    }

    double fh = 0.0, fhu = 0.0, fhv = 0.0;
    const int32_t s = cell_edge_offsets[c];
    const int32_t e = cell_edge_offsets[c + 1];
    for (int32_t ki = s; ki < e; ++ki) {
        const int32_t edge = cell_edge_ids[ki];
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

    if (!isfinite(fh))  fh  = 0.0;
    if (!isfinite(fhu)) fhu = 0.0;
    if (!isfinite(fhv)) fhv = 0.0;

    const double src = ((cell_source_mps    ? cell_source_mps[c]    : 0.0) +
                        (external_source_mps ? external_source_mps[c] : 0.0));
    const double src_safe = isfinite(src) ? src : 0.0;

    k_h[c]  = dt * (fh  * inv_a + src_safe);
    k_hu[c] = dt *  fhu * inv_a;
    k_hv[c] = dt *  fhv * inv_a;
}

// Combines the four RK4 slopes with Butcher-tableau weights (1,2,2,1)/6,
// then applies positivity, shallow damping, Manning friction, momentum cap,
// and NaN guard on the final combined state.  Writes into cell_h/hu/hv.
__global__ void swe2d_rk4_graph_combine_kernel(
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
    const double*  __restrict__ k2_h,
    const double*  __restrict__ k2_hu,
    const double*  __restrict__ k2_hv,
    const double*  __restrict__ k3_h,
    const double*  __restrict__ k3_hu,
    const double*  __restrict__ k3_hv,
    const double*  __restrict__ k4_h,
    const double*  __restrict__ k4_hu,
    const double*  __restrict__ k4_hv,
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

    // Classic RK4 combination: y_new = y0 + (1/6)(k1 + 2k2 + 2k3 + k4)
    const double one_sixth = 1.0 / 6.0;
    double h_new  = h0[c]  + one_sixth * (k1_h[c]  + 2.0*k2_h[c]  + 2.0*k3_h[c]  + k4_h[c]);
    double hu_new = hu0[c] + one_sixth * (k1_hu[c] + 2.0*k2_hu[c] + 2.0*k3_hu[c] + k4_hu[c]);
    double hv_new = hv0[c] + one_sixth * (k1_hv[c] + 2.0*k2_hv[c] + 2.0*k3_hv[c] + k4_hv[c]);

    // NaN guard (before positivity so NaN doesn't propagate)
    if (!isfinite(h_new))  h_new  = 0.0;
    if (!isfinite(hu_new)) hu_new = 0.0;
    if (!isfinite(hv_new)) hv_new = 0.0;

    // Positivity
    if (h_new < 0.0) h_new = 0.0;
    double h_final = h_new;

    if (h_final < h_min) {
        hu_new = 0.0;
        hv_new = 0.0;
    } else if (shallow_damping_depth > h_min && h_final < shallow_damping_depth) {
        // Hermite smoothstep: damps momentum smoothly near the wet/dry front
        const double t   = (h_final - h_min) / (shallow_damping_depth - h_min);
        const double t_s = fmin(1.0, fmax(0.0, t));
        const double scale = t_s * t_s * (3.0 - 2.0 * t_s);
        hu_new *= scale;
        hv_new *= scale;
    }

    // Manning friction (semi-implicit) on the combined final state
    if (h_final >= h_min) {
        double n_mann = cell_n_mann[c];
        apply_friction_cuda_local(h_final, hu_new, hv_new, dt, n_mann, g, h_min);
    }

    // Momentum cap
    if (h_final > h_min) {
        const double inv_h = 1.0 / h_final;
        const double u = hu_new * inv_h;
        const double v = hv_new * inv_h;
        const double spd = sqrt(u*u + v*v);
        const double spd_cap = fmax(momentum_cap_min_speed,
                                    momentum_cap_celerity_mult * sqrt(g * h_final));
        if (isfinite(spd) && spd > spd_cap && spd > 0.0) {
            const double scale = spd_cap / spd;
            hu_new *= scale;
            hv_new *= scale;
        }
    }

    cell_h[c]  = h_final;
    cell_hu[c] = hu_new;
    cell_hv[c] = hv_new;

    if (d_max_wse_elev_error) {
        atomicMaxDouble(d_max_wse_elev_error, fabs(h_final - h0[c]));
    }
}

__global__ void swe2d_rk5_graph_combine_kernel(
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
__global__ void swe2d_cfl_reduce_blocks_kernel(
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

__global__ void swe2d_coupling_bridge_source_kernel(
    int32_t n_bridges,
    const int32_t* __restrict__ bridge_up_cell,
    const int32_t* __restrict__ bridge_down_cell,
    const double* __restrict__ bridge_flow_cms,
    const double* __restrict__ bridge_loss_k_upstream,
    const double* __restrict__ bridge_loss_k_downstream,
    const double* __restrict__ cell_area_m2,
    int32_t n_cells,
    double bridge_opening_width_m,
    double dt_s,
    double* __restrict__ source_rate_mps)
{
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_bridges) return;

    const int32_t cu = bridge_up_cell[i];
    const int32_t cd = bridge_down_cell[i];
    if (cu < 0 || cu >= n_cells || cd < 0 || cd >= n_cells) return;

    const double q = bridge_flow_cms[i];
    if (!isfinite(q) || q == 0.0) return;

    const double au = fmax(cell_area_m2[cu], 1.0e-12);
    const double ad = fmax(cell_area_m2[cd], 1.0e-12);
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
    atomicAdd(&source_rate_mps[cu], -q_eff / au);
    atomicAdd(&source_rate_mps[cd],  q_eff / ad);
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
    return (1.0 / n) * area * pow(rh, 2.0 / 3.0) * sqrt(slope);
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
    c->q_critical = ac * sqrt(32.2 * yh);
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
        return fmin(pow((q_unit * q_unit) / 32.2, 1.0 / 3.0), xsect.y_full_ft);
    }
    const double target = q_cfs * q_cfs / 32.2;
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
    return depth_ft + v * v / (2.0 * 32.2);
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
    double tailwater_depth_ft,
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
    const double y_ds = fmin(fmax(tailwater_depth_ft, dc), y_full);
    const double step_depth = fmin(fmax(0.01, 0.02 * y_full), 0.05);

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
    double available_head_up_ft,
    double tailwater_depth_ft,
    double length_ft,
    double slope_ftft,
    double roughness_n,
    double entrance_loss_k,
    double exit_loss_k,
    double q_hint_cfs)
{
    if (available_head_up_ft <= 0.0) return 0.0;

    auto required_head_ft = [&](double q_cfs) {
        if (q_cfs <= 0.0) return 0.0;
        double e_up = 0.0;
        double y_up = 0.0;
        swe2d_direct_step_culvert_upstream_energy_cuda(
            xsect,
            q_cfs,
            fmax(1.0e-6, roughness_n),
            fmax(1.0e-6, slope_ftft),
            fmax(1.0, length_ft),
            fmax(0.0, tailwater_depth_ft),
            &e_up,
            &y_up);
        const double area = fmax(1.0e-9, swe2d_culvert_area_ft2_cuda(xsect, fmax(1.0e-6, fmin(y_up, xsect.y_full_ft))));
        const double vel = q_cfs / area;
        const double hv = (fmax(0.0, entrance_loss_k) + fmax(0.0, exit_loss_k)) * vel * vel / (2.0 * 32.2);
        return e_up + hv;
    };

    // Secant method for F(Q) = required_head(Q) - available_head = 0.
    // Bracketing phase: find Q_lo where F < 0 and Q_hi where F > 0.
    double q_lo = 0.0;
    double f_lo = -available_head_up_ft;  // F(0) = -available_head
    double q_hi = fmax(1.0, q_hint_cfs * 2.0);
    double f_hi = required_head_ft(q_hi) - available_head_up_ft;
    for (int br = 0; br < 12 && f_hi < 0.0; ++br) {
        q_lo = q_hi; f_lo = f_hi;
        q_hi *= 2.0;
        f_hi = required_head_ft(q_hi) - available_head_up_ft;
    }
    if (f_hi < 0.0) {
        // Even after widening, available head exceeds required head at q_hi
        return q_hi / 35.31466672148859;
    }

    for (int iter = 0; iter < 12; ++iter) {
        if (fabs(f_hi - f_lo) < 1.0e-30) break;
        const double q_mid = q_hi - f_hi * (q_hi - q_lo) / (f_hi - f_lo);
        if (q_mid <= 0.0) break;
        const double f_mid = required_head_ft(q_mid) - available_head_up_ft;
        if (fabs(f_mid) < 1.0e-8 * available_head_up_ft) {
            return fmax(0.0, q_mid) / 35.31466672148859;
        }
        if (f_mid < 0.0) {
            q_lo = q_mid; f_lo = f_mid;
        } else {
            q_hi = q_mid; f_hi = f_mid;
        }
    }
    return fmax(0.0, 0.5 * (q_lo + q_hi)) / 35.31466672148859;
}

// Forward-declare table lookup for use inside the structure flow kernel.
__device__ double swe2d_culvert_table_lookup_cuda(
    int32_t ci, double hw_ft, double tw_ft,
    int32_t n_culverts, const double* d_table_header, const double* d_table_data,
    int32_t n_hw_global, int32_t n_tw_global);

__global__ void swe2d_compute_structure_flows_kernel(
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
    const double* __restrict__ culvert_area_m2,
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
    double gravity_mps2,
    double* __restrict__ structure_flow_cms,
    int32_t culvert_solver_mode,
    const double* __restrict__ culvert_table_header,
    const double* __restrict__ culvert_table_data,
    int32_t culvert_table_n_hw,
    int32_t culvert_table_n_tw)
{
    const int32_t i = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n_structures) return;

    structure_flow_cms[i] = 0.0;
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
        structure_flow_cms[i] = q;
        return;
    }
    if (structure_type[i] == 3) {
        const double area = fmax(0.0, opening[i]) * fmax(0.0, width[i]) * fmax(0.0, height[i]);
        double q = bw2d_orifice_q(wu, wd, area, cd[i], gravity_mps2);
        if (qmax >= 0.0) q = fmax(-qmax, fmin(q, qmax));
        structure_flow_cms[i] = q;
        return;
    }
    if (structure_type[i] == 4) {
        const double area = fmax(0.0, opening[i]) * fmax(0.0, width[i]) * fmax(0.0, height[i]);
        const double loss_scale = fmax(1.0e-6, 1.0 + fmax(0.0, entrance_loss_k[i]) + fmax(0.0, exit_loss_k[i]));
        const double dh = wu - wd;
        if (area > 0.0 && fabs(dh) > 1.0e-12) {
            double q = area * sqrt(fmax(0.0, 2.0 * gravity_mps2 * fabs(dh))) / loss_scale;
            if (qmax >= 0.0) q = fmin(q, qmax);
            structure_flow_cms[i] = (dh >= 0.0) ? q : -q;
        }
        return;
    }
    if (structure_type[i] == 5) {
        double q = fmax(0.0, q_pump[i]);
        if (qmax >= 0.0) q = fmin(q, qmax);
        structure_flow_cms[i] = (wu >= wd) ? q : -q;
        return;
    }
    if (structure_type[i] != 2) return;

    const double sign = (wu >= wd) ? 1.0 : -1.0;
    const double upstream_wse = (sign >= 0.0) ? wu : wd;
    const double downstream_wse = (sign >= 0.0) ? wd : wu;
    const double upstream_invert = (sign >= 0.0) ? inlet_invert_elev[i] : outlet_invert_elev[i];
    const double downstream_invert = (sign >= 0.0) ? outlet_invert_elev[i] : inlet_invert_elev[i];
    const double available_head_up = fmax(0.0, upstream_wse - upstream_invert);
    const double tailwater_depth = fmax(0.0, downstream_wse - downstream_invert);
    const double len = fmax(0.1, length[i]);

    double slope = culvert_slope[i];
    if (!(slope > 0.0)) {
        slope = fabs(upstream_invert - downstream_invert) / len;
    }
    slope = fmax(1.0e-6, slope);

    const double rise = fmax(0.0, (culvert_rise[i] > 0.0) ? culvert_rise[i] : fmax(height[i], diameter[i]));
    const double span = fmax(0.0, (culvert_span[i] > 0.0) ? culvert_span[i] : fmax(width[i], rise));
    const int code = max(1, min(57, static_cast<int>(culvert_code[i])));

    swe2d_culvert_xsect_cuda xsect{};
    xsect.code = code;
    xsect.is_rect = (culvert_shape[i] == 1) ? 1 : 0;
    if (xsect.is_rect) {
        xsect.width_ft = fmax(1.0e-6, span * 3.280839895013123);
        xsect.y_full_ft = fmax(1.0e-6, rise * 3.280839895013123);
        xsect.a_full_ft2 = xsect.width_ft * xsect.y_full_ft;
        xsect.radius_ft = 0.0;
    } else {
        const double dia_ft = fmax(1.0e-6, fmax(diameter[i], rise) * 3.280839895013123);
        xsect.radius_ft = 0.5 * dia_ft;
        xsect.y_full_ft = dia_ft;
        xsect.a_full_ft2 = M_PI * xsect.radius_ft * xsect.radius_ft;
        xsect.width_ft = 0.0;
    }

    const double q_inlet_cfs = fmax(0.0, swe2d_culvert_inlet_controlled_flow_cfs_cuda(xsect, slope, fmax(0.0, available_head_up * 3.280839895013123)));
    const double q_inlet_cms = q_inlet_cfs / 35.31466672148859;

    double area = fmax(0.0, culvert_area_m2[i]);
    if (area <= 0.0 && fmax(diameter[i], rise) > 0.0 && culvert_shape[i] == 0) {
        area = bw2d_circular_area(fmax(diameter[i], rise));
    }

    double q_orifice = 0.0;
    if (area > 0.0) {
        q_orifice = fabs(bw2d_orifice_q(available_head_up, tailwater_depth, area, cd[i], gravity_mps2));
        if (qmax >= 0.0) q_orifice = fmin(q_orifice, qmax);
    }

    double q_manning_cap = 0.0;
    const double dia_for_cap = fmax(fmax(diameter[i], rise), bw2d_equiv_diameter_from_area(fmax(0.0, area)));
    if (dia_for_cap > 0.0) {
        q_manning_cap = bw2d_pipe_manning_capacity_full(dia_for_cap, slope, roughness_n[i]);
    }

    const double q_hint_cfs = fmax(1.0, fmax(q_inlet_cfs, fmax(q_orifice, q_manning_cap) * 35.31466672148859));

    double q_outlet = 0.0;
    if (culvert_solver_mode == 1 && culvert_table_data && culvert_table_header) {
        // Table lookup: bilinear interpolation from pre-computed Q(hw,tw) grid.
        q_outlet = swe2d_culvert_table_lookup_cuda(
            i,  // culvert index (local to the table, matches upload order)
            fmax(0.0, available_head_up * 3.280839895013123),
            fmax(0.0, tailwater_depth * 3.280839895013123),
            n_structures,
            culvert_table_header,
            culvert_table_data,
            culvert_table_n_hw,
            culvert_table_n_tw);
    } else {
        // Direct secant solver (default)
        q_outlet = swe2d_culvert_outlet_control_flow_cms_cuda(
            xsect,
            fmax(0.0, available_head_up * 3.280839895013123),
            fmax(0.0, tailwater_depth * 3.280839895013123),
            fmax(0.1, len * 3.280839895013123),
            fmax(1.0e-6, slope),
            fmax(1.0e-6, roughness_n[i]),
            entrance_loss_k[i],
            exit_loss_k[i],
            q_hint_cfs);
    }

    double q = fmax(0.0, fmin(q_inlet_cms, (q_outlet > 0.0) ? q_outlet : q_inlet_cms));
    if (q_orifice > 0.0) q = (q > 0.0) ? fmin(q, q_orifice) : q_orifice;
    if (q_manning_cap > 0.0) q = (q > 0.0) ? fmin(q, q_manning_cap) : q_manning_cap;

    if (embankment_enabled[i] != 0) {
        const double q_emb = fabs(bw2d_weir_q(
            upstream_wse,
            downstream_wse,
            embankment_crest_elev[i],
            fmax(0.0, embankment_overflow_width[i]),
            fmax(1.0e-6, embankment_weir_coeff[i])));
        q += q_emb;
    }

    q *= fmax(1.0, culvert_barrels[i]);
    if (qmax >= 0.0) q = fmin(q, qmax);
    structure_flow_cms[i] = sign * q;
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
    // k4 slope buffer for true RK4 (temporal_order=5, graph-safe)
    alloc_d(reinterpret_cast<void**>(&dev->d_k4_h),  sz_cells * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_k4_hu), sz_cells * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_k4_hv), sz_cells * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_k5_h),  sz_cells * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_k5_hu), sz_cells * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_k5_hv), sz_cells * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_k6_h),  sz_cells * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_k6_hu), sz_cells * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_k6_hv), sz_cells * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_cell_source_mps), sz_cells * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_stage_cell_source_mps), static_cast<size_t>(SWE2D_GRAPH_STAGE_SLOTS) * sz_cells * sizeof(double));
    alloc_d(reinterpret_cast<void**>(&dev->d_stage_edge_bc), static_cast<size_t>(SWE2D_GRAPH_STAGE_SLOTS) * sz_edges * sizeof(int32_t));
    alloc_d(reinterpret_cast<void**>(&dev->d_stage_edge_bc_val), static_cast<size_t>(SWE2D_GRAPH_STAGE_SLOTS) * sz_edges * sizeof(double));
    CUDA_CHECK(cudaMemset(dev->d_cell_source_mps, 0, sz_cells * sizeof(double)));
    CUDA_CHECK(cudaMemset(dev->d_stage_cell_source_mps, 0, static_cast<size_t>(SWE2D_GRAPH_STAGE_SLOTS) * sz_cells * sizeof(double)));

    // External coupling source buffer (shared with on-device coupling path)
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

    // Phase 5: Optionally allocate pressure workspace now if nonhydro mode detected
    // (Allocation can also happen lazily on first nonhydro step; done here for early validation)
    // Note: deferred for now to avoid allocation overhead for hydrostatic-only runs.

    // Wire this device to the persistent coupling global so that Python-side
    // coupling functions can operate on-device without explicit dev pointers.
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
            cache.variant_key == graph_variant_key &&
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
                // ── Expanded graph: classify_and_mark → degen → hg_bc → gradient → flux → update → cfl → reduce → pack ──

                // classify_and_mark (fused, always runs)
                {
                    const int c_grid = (n_cells + BLOCK - 1) / BLOCK;
                    // was_active copy already done before graph section
                    CUDA_CHECK(cudaMemsetAsync(dev->d_n_wet, 0, sizeof(int32_t), dev->d_stream));
                    swe2d_classify_and_mark_kernel<<<c_grid, BLOCK, BLOCK * sizeof(int32_t), dev->d_stream>>>(
                        n_cells, n_edges,
                        dev->d_h, dev->d_cell_source_mps, dev->d_external_source_mps, dev->d_bc_forced,
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
                    dev->d_external_source_mps);
                // cfl + cfl_reduce (two-level)
                CUDA_CHECK(cudaMemsetAsync(dev->d_lambda_max, 0, sizeof(double), dev->d_stream));
                {
                    int grid_cfl = (n_edges + BLOCK - 1) / BLOCK;
                    {
                        static int32_t s_cfl_block_cap = 0;
                        static double* s_cfl_block_ptr = nullptr;
                        if (s_cfl_block_cap < grid_cfl) {
                            double* new_ptr = nullptr;
                            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&new_ptr),
                                                  static_cast<size_t>(grid_cfl) * sizeof(double)));
                            if (s_cfl_block_ptr) cudaFree(s_cfl_block_ptr);
                            s_cfl_block_ptr = new_ptr;
                            s_cfl_block_cap = grid_cfl;
                        }
                        dev->d_cfl_block_max = s_cfl_block_ptr;
                    }
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
            dev->d_h, dev->d_cell_source_mps, dev->d_external_source_mps, dev->d_bc_forced,
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
            dev->d_external_source_mps);
        CUDA_CHECK(cudaGetLastError());
    }

    // Kernel 3: CFL reduction for max Courant diagnostic
    CUDA_CHECK(cudaMemsetAsync(dev->d_lambda_max, 0, sizeof(double), dev->d_stream));
    {
        int grid = (n_edges + BLOCK - 1) / BLOCK;
        // Ensure cfl_block_max is sized for this grid
        {
            static int32_t s_cfl_block_cap2 = 0;
            static double* s_cfl_block_ptr2 = nullptr;
            if (s_cfl_block_cap2 < grid) {
                double* new_ptr = nullptr;
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&new_ptr),
                                      static_cast<size_t>(grid) * sizeof(double)));
                if (s_cfl_block_ptr2) cudaFree(s_cfl_block_ptr2);
                s_cfl_block_ptr2 = new_ptr;
                s_cfl_block_cap2 = grid;
            }
            dev->d_cfl_block_max = s_cfl_block_ptr2;
        }
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

__global__ void swe2d_persistent_chunk_kernel_first_order(
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
            n_cells, dev->d_h, dev->d_cell_source_mps, dev->d_external_source_mps, dev->d_bc_forced,
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

__global__ void swe3d_dt_wave_max_kernel(
    int64_t n_cells,
    const double* d_u,
    const double* d_v,
    const double* d_w,
    const double* d_vof,
    const double* d_phi,
    const double* d_ax,
    const double* d_ay,
    const double* d_az,
    double g,
    double dx,
    double dy,
    double dz,
    double gravity_wave_weight,
    int32_t adaptive_mode,
    unsigned long long* d_max_bits);

double swe2d_gpu_compute_dt_3d_patch(
    SWE2DDeviceState* dev,
    double g,
    double cfl_factor,
    double dt_max)
{
    if (!dev || !dev->patch3d) {
        return dt_max;
    }

    auto* patch = dev->patch3d;
    if (patch->n_cells <= 0) {
        return dt_max;
    }

    const auto& desc = patch->desc;
    const double dx = fabs(desc.dx);
    const double dy = fabs(desc.dy);
    const double dz = fabs(desc.dz);
    const double h_char = fmin(dx, fmin(dy, dz));
    if (!(dx > 0.0) || !(dy > 0.0) || !(dz > 0.0) ||
        !std::isfinite(dx) || !std::isfinite(dy) || !std::isfinite(dz) ||
        !(cfl_factor > 0.0)) {
        return dt_max;
    }

    const SWE3DRuntimeControls runtime_controls = swe3d_load_runtime_controls_cached();
    const int32_t adaptive_mode = std::max<int32_t>(
        static_cast<int32_t>(SWE3DAdaptiveDtMode::ADVECTIVE_ONLY),
        std::min<int32_t>(
            static_cast<int32_t>(SWE3DAdaptiveDtMode::ADVECTIVE_GRAVITY_PROJECTION),
            runtime_controls.adaptive_dt_mode));
    const double gravity_wave_weight = cfl_factor / fmax(runtime_controls.gravity_wave_cfl, 1.0e-8);

    constexpr int BLOCK = 256;
    const int64_t n_cells = patch->n_cells;
    const int grid = static_cast<int>((n_cells + static_cast<int64_t>(BLOCK) - 1) / static_cast<int64_t>(BLOCK));

    // Validate that 3D patch device pointers are allocated before kernel call
    if (!patch->d_u || !patch->d_v || !patch->d_w || !patch->d_vof || !patch->d_proj_residual_bits) {
        std::fprintf(stderr,
                     "[SWE3D_ERR] swe2d_gpu_compute_dt_3d_patch: uninitialized patch pointers (u=%p v=%p w=%p vof=%p bits=%p)\n",
                     static_cast<const void*>(patch->d_u),
                     static_cast<const void*>(patch->d_v),
                     static_cast<const void*>(patch->d_w),
                     static_cast<const void*>(patch->d_vof),
                     static_cast<const void*>(patch->d_proj_residual_bits));
        return dt_max;
    }

    CUDA_CHECK(cudaMemsetAsync(
        patch->d_proj_residual_bits,
        0,
        sizeof(unsigned long long),
        dev->d_stream));
    swe3d_dt_wave_max_kernel<<<grid, BLOCK, 0, dev->d_stream>>>(
        n_cells,
        patch->d_u,
        patch->d_v,
        patch->d_w,
        patch->d_vof,
        patch->d_phi,
        patch->d_ax,
        patch->d_ay,
        patch->d_az,
        g,
        dx,
        dy,
        dz,
        gravity_wave_weight,
        adaptive_mode,
        patch->d_proj_residual_bits);
    CUDA_CHECK(cudaGetLastError());

    unsigned long long max_bits = 0ULL;
    CUDA_CHECK(cudaMemcpyAsync(
        &max_bits,
        patch->d_proj_residual_bits,
        sizeof(unsigned long long),
        cudaMemcpyDeviceToHost,
        dev->d_stream));
    CUDA_CHECK(cudaStreamSynchronize(dev->d_stream));

    double wave_rate = 0.0;
    std::memcpy(&wave_rate, &max_bits, sizeof(double));
    if (!std::isfinite(wave_rate) || wave_rate <= 1.0e-12) {
        return dt_max;
    }

    double dt = cfl_factor / wave_rate;
    if (adaptive_mode >= static_cast<int32_t>(SWE3DAdaptiveDtMode::ADVECTIVE_GRAVITY_PROJECTION)) {
        const double target = fmax(runtime_controls.projection_residual_target, 1.0e-12);
        const double resid = patch->last_projection_residual;
        const bool projection_bad =
            !patch->last_projection_converged ||
            !std::isfinite(resid) ||
            resid > target;
        if (projection_bad) {
            const double ratio = target / fmax(std::isfinite(resid) ? resid : target * 100.0, target);
            const double scale = fmax(
                runtime_controls.projection_min_dt_factor,
                fmin(1.0, sqrt(fmax(0.0, ratio))));
            dt *= scale;
        }
    }

    if (!std::isfinite(dt) || dt <= 0.0) {
        return fmax(1.0e-9, fmin(dt_max, h_char * 1.0e-3));
    }
    return fmax(1.0e-9, fmin(dt, dt_max));
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
    const double dt_half = 0.5 * dt;

    // Save initial state in d_h0, d_hu0, d_hv0.
    CUDA_CHECK(cudaMemcpyAsync(dev->d_h0,  dev->d_h,  sz, cudaMemcpyDeviceToDevice, dev->d_stream));
    CUDA_CHECK(cudaMemcpyAsync(dev->d_hu0, dev->d_hu, sz, cudaMemcpyDeviceToDevice, dev->d_stream));
    CUDA_CHECK(cudaMemcpyAsync(dev->d_hv0, dev->d_hv, sz, cudaMemcpyDeviceToDevice, dev->d_stream));

    SWE2DStepDiag tmp_diag;
    int grid = (n_cells + BLOCK - 1) / BLOCK;

    // Stage 1: evaluate at t_n and capture k1.
    swe2d_gpu_step(dev, t_now, dt_half, g, h_min, spatial_scheme, cfl_factor,
                   max_inv_area, cfl_lambda_cap,
                   momentum_cap_min_speed, momentum_cap_celerity_mult,
                   depth_cap, max_rel_depth_increase, shallow_damping_depth,
                   extreme_rain_mode, source_cfl_beta, source_max_substeps,
                   source_rate_cap, source_depth_step_cap, source_true_subcycling, source_imex_split,
                   enable_shallow_front_recon_fallback,
                   false, &tmp_diag,
                   front_flux_damping, active_set_hysteresis);
    CUDA_CHECK(cudaMemcpyAsync(dev->d_h1,  dev->d_h,  sz, cudaMemcpyDeviceToDevice, dev->d_stream));
    CUDA_CHECK(cudaMemcpyAsync(dev->d_hu1, dev->d_hu, sz, cudaMemcpyDeviceToDevice, dev->d_stream));
    CUDA_CHECK(cudaMemcpyAsync(dev->d_hv1, dev->d_hv, sz, cudaMemcpyDeviceToDevice, dev->d_stream));
    swe2d_rk4_capture_increment_kernel<<<grid, BLOCK, 0, dev->d_stream>>>(
        n_cells,
        dev->d_h1, dev->d_hu1, dev->d_hv1,
        dev->d_h, dev->d_hu, dev->d_hv,
        dev->d_h0, dev->d_hu0, dev->d_hv0);
    CUDA_CHECK(cudaGetLastError());

    // Preserve the stage-1 midpoint state for the next increment capture.
    CUDA_CHECK(cudaMemcpyAsync(dev->d_h3,  dev->d_h,  sz, cudaMemcpyDeviceToDevice, dev->d_stream));
    CUDA_CHECK(cudaMemcpyAsync(dev->d_hu3, dev->d_hu, sz, cudaMemcpyDeviceToDevice, dev->d_stream));
    CUDA_CHECK(cudaMemcpyAsync(dev->d_hv3, dev->d_hv, sz, cudaMemcpyDeviceToDevice, dev->d_stream));

    // Stage 2: evaluate at t_n + dt/2 and capture k2.
    swe2d_gpu_step(dev, t_now + dt_half, dt_half, g, h_min, spatial_scheme, cfl_factor,
                   max_inv_area, cfl_lambda_cap,
                   momentum_cap_min_speed, momentum_cap_celerity_mult,
                   depth_cap, max_rel_depth_increase, shallow_damping_depth,
                   extreme_rain_mode, source_cfl_beta, source_max_substeps,
                   source_rate_cap, source_depth_step_cap, source_true_subcycling, source_imex_split,
                   enable_shallow_front_recon_fallback,
                   false, &tmp_diag,
                   front_flux_damping, active_set_hysteresis);
    swe2d_rk4_capture_increment_kernel<<<grid, BLOCK, 0, dev->d_stream>>>(
        n_cells,
        dev->d_h2, dev->d_hu2, dev->d_hv2,
        dev->d_h, dev->d_hu, dev->d_hv,
        dev->d_h3, dev->d_hu3, dev->d_hv3);
    CUDA_CHECK(cudaGetLastError());

    swe2d_rk4_build_stage_kernel<<<grid, BLOCK, 0, dev->d_stream>>>(
        n_cells,
        dev->d_h3, dev->d_hu3, dev->d_hv3,
        dev->d_h0, dev->d_hu0, dev->d_hv0,
        dev->d_h2, dev->d_hu2, dev->d_hv2,
        0.5);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(dev->d_h,  dev->d_h3,  sz, cudaMemcpyDeviceToDevice, dev->d_stream));
    CUDA_CHECK(cudaMemcpyAsync(dev->d_hu, dev->d_hu3, sz, cudaMemcpyDeviceToDevice, dev->d_stream));
    CUDA_CHECK(cudaMemcpyAsync(dev->d_hv, dev->d_hv3, sz, cudaMemcpyDeviceToDevice, dev->d_stream));

    // Stage 3: evaluate at t_n + dt/2 and capture k3.
    swe2d_gpu_step(dev, t_now + dt_half, dt_half, g, h_min, spatial_scheme, cfl_factor,
                   max_inv_area, cfl_lambda_cap,
                   momentum_cap_min_speed, momentum_cap_celerity_mult,
                   depth_cap, max_rel_depth_increase, shallow_damping_depth,
                   extreme_rain_mode, source_cfl_beta, source_max_substeps,
                   source_rate_cap, source_depth_step_cap, source_true_subcycling, source_imex_split,
                   enable_shallow_front_recon_fallback,
                   false, &tmp_diag,
                   front_flux_damping, active_set_hysteresis);
    swe2d_rk4_shift_from_reference_kernel<<<grid, BLOCK, 0, dev->d_stream>>>(
        n_cells,
        dev->d_h3, dev->d_hu3, dev->d_hv3,
        dev->d_h0, dev->d_hu0, dev->d_hv0,
        dev->d_h, dev->d_hu, dev->d_hv,
        dev->d_h3, dev->d_hu3, dev->d_hv3,
        0.5);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(dev->d_h,  dev->d_h3,  sz, cudaMemcpyDeviceToDevice, dev->d_stream));
    CUDA_CHECK(cudaMemcpyAsync(dev->d_hu, dev->d_hu3, sz, cudaMemcpyDeviceToDevice, dev->d_stream));
    CUDA_CHECK(cudaMemcpyAsync(dev->d_hv, dev->d_hv3, sz, cudaMemcpyDeviceToDevice, dev->d_stream));

    // Stage 4: evaluate at t_n + dt and capture k4.
    swe2d_gpu_step(dev, t_now + dt, dt_half, g, h_min, spatial_scheme, cfl_factor,
                   max_inv_area, cfl_lambda_cap,
                   momentum_cap_min_speed, momentum_cap_celerity_mult,
                   depth_cap, max_rel_depth_increase, shallow_damping_depth,
                   extreme_rain_mode, source_cfl_beta, source_max_substeps,
                   source_rate_cap, source_depth_step_cap, source_true_subcycling, source_imex_split,
                   enable_shallow_front_recon_fallback,
                   false, &tmp_diag,
                   front_flux_damping, active_set_hysteresis);

    // Final combine: y_new = y0 + (1/6) * (k1 + 2*k2 + 2*k3 + k4).
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
// swe2d_gpu_step_rk4_graph: Graph-safe true Butcher-tableau RK4
// ─────────────────────────────────────────────────────────────────────────────
// Temporal order = 5. Uses pure L(U) evaluations per stage, with full 4-stage
// sequence capturable as a single CUDA graph (time_integrator=5).
// Classify runs once per step outside the graph.
void swe2d_gpu_step_rk4_graph(
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
    if (!dev) throw std::invalid_argument("swe2d_gpu_step_rk4_graph: null device");

    constexpr int BLOCK = 256;
    const int32_t n_cells = dev->n_cells;
    const int32_t n_edges = dev->n_edges;
    const size_t sz = static_cast<size_t>(n_cells) * sizeof(double);
    const size_t sz_edges = static_cast<size_t>(n_edges) * sizeof(double);
    const uint64_t graph_launches_before = dev->graph_replay_count;
    const int32_t prev_graph_integrator = dev->kernel_graph_cache.time_integrator;
    dev->kernel_graph_cache.time_integrator = 5;
    const double stage_c[4] = {0.0, 0.5, 0.5, 1.0};

    // Save initial state as U^0
    CUDA_CHECK(cudaMemcpyAsync(dev->d_h0,  dev->d_h,  sz, cudaMemcpyDeviceToDevice, dev->d_stream));
    CUDA_CHECK(cudaMemcpyAsync(dev->d_hu0, dev->d_hu, sz, cudaMemcpyDeviceToDevice, dev->d_stream));
    CUDA_CHECK(cudaMemcpyAsync(dev->d_hv0, dev->d_hv, sz, cudaMemcpyDeviceToDevice, dev->d_stream));

    auto precompute_stage_forcing = [&]() {
        for (int slot = 0; slot < 4; ++slot) {
            CUDA_CHECK(cudaMemcpyAsync(swe2d_stage_edge_bc_slot(dev, slot), dev->d_edge_bc,
                                       static_cast<size_t>(n_edges) * sizeof(int32_t),
                                       cudaMemcpyDeviceToDevice, dev->d_stream));
            CUDA_CHECK(cudaMemcpyAsync(swe2d_stage_edge_bc_val_slot(dev, slot), dev->d_edge_bc_val,
                                       sz_edges, cudaMemcpyDeviceToDevice, dev->d_stream));
        }

        if (dev->n_hg_edges > 0 && dev->d_hg_edge_index && dev->d_hg_offsets &&
            dev->d_hg_time_s && dev->d_hg_value) {
            const int hg_grid = (dev->n_hg_edges + BLOCK - 1) / BLOCK;
            for (int slot = 0; slot < 4; ++slot) {
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
            for (int slot = 0; slot < 4; ++slot) {
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
                n_cells, 4, dev->d_stage_cell_source_mps, dev->d_cell_source_mps);
            CUDA_CHECK(cudaGetLastError());
        } else {
            CUDA_CHECK(cudaMemsetAsync(dev->d_stage_cell_source_mps, 0,
                                       static_cast<size_t>(4) * sz, dev->d_stream));
            CUDA_CHECK(cudaMemsetAsync(dev->d_cell_source_mps, 0, sz, dev->d_stream));
        }
    };

    precompute_stage_forcing();

    // Wet/dry classification based on U^0
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
            n_cells, dev->d_h, dev->d_cell_source_mps, dev->d_external_source_mps, dev->d_bc_forced,
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

        swe2d_rk4_rhs_collect_kernel<<<grid_c, BLOCK, 0, dev->d_stream>>>(
            n_cells,
            dev->d_cell_edge_offsets,
            dev->d_cell_edge_ids,
            dev->d_edge_c0,
            dev->d_edge_c1,
            k_h,
            k_hu,
            k_hv,
            dev->d_flux_h,
            dev->d_flux_hu,
            dev->d_flux_hv,
            dev->d_flux_hu_r,
            dev->d_flux_hv_r,
            dev->d_cell_inv_area,
            stage_source,
            dev->d_external_source_mps,
            dev->d_active,
            dev->d_degen_mask,
            dev->d_inv_area_repaired,
            dev->degen_mode,
            max_inv_area,
            dt);
        CUDA_CHECK(cudaGetLastError());
    };

    auto run_rk4_stages_and_combine = [&]() {
        evaluate_rhs(swe2d_stage_edge_bc_slot(dev, 0), swe2d_stage_edge_bc_val_slot(dev, 0), swe2d_stage_source_slot(dev, 0),
                     dev->d_h1, dev->d_hu1, dev->d_hv1);

        swe2d_rk4_build_stage_kernel<<<grid_c, BLOCK, 0, dev->d_stream>>>(
            n_cells,
            dev->d_h, dev->d_hu, dev->d_hv,
            dev->d_h0, dev->d_hu0, dev->d_hv0,
            dev->d_h1, dev->d_hu1, dev->d_hv1,
            0.5);
        CUDA_CHECK(cudaGetLastError());
        evaluate_rhs(swe2d_stage_edge_bc_slot(dev, 1), swe2d_stage_edge_bc_val_slot(dev, 1), swe2d_stage_source_slot(dev, 1),
                     dev->d_h2, dev->d_hu2, dev->d_hv2);

        swe2d_rk4_build_stage_kernel<<<grid_c, BLOCK, 0, dev->d_stream>>>(
            n_cells,
            dev->d_h, dev->d_hu, dev->d_hv,
            dev->d_h0, dev->d_hu0, dev->d_hv0,
            dev->d_h2, dev->d_hu2, dev->d_hv2,
            0.5);
        CUDA_CHECK(cudaGetLastError());
        evaluate_rhs(swe2d_stage_edge_bc_slot(dev, 2), swe2d_stage_edge_bc_val_slot(dev, 2), swe2d_stage_source_slot(dev, 2),
                     dev->d_h3, dev->d_hu3, dev->d_hv3);

        swe2d_rk4_build_stage_kernel<<<grid_c, BLOCK, 0, dev->d_stream>>>(
            n_cells,
            dev->d_h, dev->d_hu, dev->d_hv,
            dev->d_h0, dev->d_hu0, dev->d_hv0,
            dev->d_h3, dev->d_hu3, dev->d_hv3,
            1.0);
        CUDA_CHECK(cudaGetLastError());
        evaluate_rhs(swe2d_stage_edge_bc_slot(dev, 3), swe2d_stage_edge_bc_val_slot(dev, 3), swe2d_stage_source_slot(dev, 3),
                     dev->d_k4_h, dev->d_k4_hu, dev->d_k4_hv);

        CUDA_CHECK(cudaMemsetAsync(dev->d_max_wse_elev_error, 0, sizeof(double), dev->d_stream));
        swe2d_rk4_graph_combine_kernel<<<grid_c, BLOCK, 0, dev->d_stream>>>(
            n_cells,
            dev->d_h, dev->d_hu, dev->d_hv,
            dev->d_h0, dev->d_hu0, dev->d_hv0,
            dev->d_h1, dev->d_hu1, dev->d_hv1,
            dev->d_h2, dev->d_hu2, dev->d_hv2,
            dev->d_h3, dev->d_hu3, dev->d_hv3,
            dev->d_k4_h, dev->d_k4_hu, dev->d_k4_hv,
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
    if (swe2d_debug_enabled("BACKWATER_SWE2D_DEBUG_GPU_GRAPH")) {
        std::fprintf(stderr,
                     "[SWE2D_DEBUG] RK4_GRAPH integrator=5 eligible=%d (graphs=%d edge_dbg=%d flux_dbg=%d)\n",
                     static_cast<int>(try_kernel_graph),
                     static_cast<int>(dev->enable_kernel_graphs),
                     static_cast<int>(dbg_edge_flux),
                     static_cast<int>(dbg_flux_summary));
    }

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
            cache.time_integrator == 5 &&
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
                run_rk4_stages_and_combine();

                cudaGraph_t graph = nullptr;
                cudaError_t cap_end = cudaStreamEndCapture(dev->d_stream, &graph);
                if (cap_end == cudaSuccess && graph != nullptr) {
                    cudaGraphExec_t exec = nullptr;
                    if (cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) == cudaSuccess) {
                        cache.graph = graph;
                        cache.exec = exec;
                        cache.n_cells = n_cells;
                        cache.n_edges = n_edges;
                        cache.spatial_scheme = spatial_scheme;
                        cache.time_integrator = 5;
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
        if (swe2d_debug_enabled("BACKWATER_SWE2D_DEBUG_GPU_GRAPH")) {
            std::fprintf(stderr,
                         "[SWE2D_DEBUG] RK4_GRAPH integrator=5 executing non-graph path\\n");
        }
        run_rk4_stages_and_combine();
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

    // Compute final CFL for diagnostics
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

void swe2d_gpu_step_rk4_graph_persistent_chunk(
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
    bool sync_diagnostics,
    SWE2DStepDiag* diag,
    double front_flux_damping,
    bool   active_set_hysteresis)
{
    if (!dev) throw std::invalid_argument("swe2d_gpu_step_rk4_graph_persistent_chunk: null device");

    if (chunk_substeps <= 1) {
        swe2d_gpu_step_rk4_graph(
            dev, t_now, dt, g, h_min, spatial_scheme, cfl_factor,
            max_inv_area, cfl_lambda_cap, momentum_cap_min_speed, momentum_cap_celerity_mult,
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

    SWE2DStepDiag sub_diag{};
    const double dt_sub = dt / static_cast<double>(chunk_substeps);
    for (int sub = 0; sub < chunk_substeps; ++sub) {
        swe2d_gpu_step_rk4_graph(
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
            sync_diagnostics && (sub == (chunk_substeps - 1)),
            &sub_diag,
            front_flux_damping,
            active_set_hysteresis);
    }

    if (diag) {
        *diag = sub_diag;
        diag->dt = dt;
    }
}

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
            n_cells, dev->d_h, dev->d_cell_source_mps, dev->d_external_source_mps, dev->d_bc_forced,
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

        swe2d_rk4_rhs_collect_kernel<<<grid_c, BLOCK, 0, dev->d_stream>>>(
            n_cells,
            dev->d_cell_edge_offsets,
            dev->d_cell_edge_ids,
            dev->d_edge_c0,
            dev->d_edge_c1,
            k_h,
            k_hu,
            k_hv,
            dev->d_flux_h,
            dev->d_flux_hu,
            dev->d_flux_hv,
            dev->d_flux_hu_r,
            dev->d_flux_hv_r,
            dev->d_cell_inv_area,
            stage_source,
            dev->d_external_source_mps,
            dev->d_active,
            dev->d_degen_mask,
            dev->d_inv_area_repaired,
            dev->degen_mode,
            max_inv_area,
            dt);
        CUDA_CHECK(cudaGetLastError());
    };

    auto run_rk5_stages_and_combine = [&]() {
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
        front_flux_damping);

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

// ─────────────────────────────────────────────────────────────────────────────
// Persistent GPU coupling path: preload once, run on-device every step.
// Eliminates per-step H2D upload of static structure params and avoids
// intermediate D2H→H2D round trips for structure_flow_cms.
// ─────────────────────────────────────────────────────────────────────────────

namespace {

// Helper: ensure a device buffer in the SF workspace is large enough.
// Allocates or grows the buffer and zeros if newly allocated.
template <typename T>
void sf_ensure(SWE2DDeviceState::StructureFlowWorkspace& ws,
               T*& ptr, int32_t& cap, int32_t need) {
    if (cap < need) {
        if (ptr) cudaFree(ptr);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr),
                              static_cast<size_t>(need) * sizeof(T)));
        cap = need;
    }
}

// Upload a single array to an existing device buffer.
template <typename T>
void sf_upload(const T* src, T* dst, int32_t n, cudaStream_t stream) {
    if (n > 0 && src && dst) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src,
                                   static_cast<size_t>(n) * sizeof(T),
                                   cudaMemcpyHostToDevice, stream));
    }
}

} // namespace

void swe2d_gpu_preload_structure_params(
    SWE2DDeviceState* dev,
    int32_t n_structures,
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
    const double* culvert_area_m2,
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
    double gravity_mps2)
{
    if (!dev) dev = s_coupling_dev;
    if (!dev) {
        if (n_structures <= 0) return;  // no-op: no structures to preload
        // Throw so Python knows preload wasn't possible
        throw std::runtime_error(
            "swe2d_gpu_preload_structure_params: no GPU device state available. "
            "Ensure GPU solver init runs before coupling starts.");
    }
    if (n_structures <= 0) return;
    auto& ws = dev->sf_ws;
    cudaStream_t stream = dev->d_stream;

    // Early exit if params are already loaded with the same count and gravity.
    if (ws.params_preloaded && ws.n_structures == n_structures &&
        ws.gravity_mps2 == gravity_mps2) {
        return;
    }

    sf_ensure(ws, ws.d_cell_wse, ws.cell_capacity, dev->n_cells);
    sf_ensure(ws, ws.d_structure_type, ws.struct_capacity, n_structures);
    sf_ensure(ws, ws.d_upstream_cell, ws.struct_capacity, n_structures);
    sf_ensure(ws, ws.d_downstream_cell, ws.struct_capacity, n_structures);
    sf_ensure(ws, ws.d_crest_elev, ws.struct_capacity, n_structures);
    sf_ensure(ws, ws.d_width, ws.struct_capacity, n_structures);
    sf_ensure(ws, ws.d_height, ws.struct_capacity, n_structures);
    sf_ensure(ws, ws.d_diameter, ws.struct_capacity, n_structures);
    sf_ensure(ws, ws.d_length, ws.struct_capacity, n_structures);
    sf_ensure(ws, ws.d_roughness_n, ws.struct_capacity, n_structures);
    sf_ensure(ws, ws.d_coeff, ws.struct_capacity, n_structures);
    sf_ensure(ws, ws.d_cd, ws.struct_capacity, n_structures);
    sf_ensure(ws, ws.d_opening, ws.struct_capacity, n_structures);
    sf_ensure(ws, ws.d_q_pump, ws.struct_capacity, n_structures);
    sf_ensure(ws, ws.d_max_flow, ws.struct_capacity, n_structures);
    sf_ensure(ws, ws.d_culvert_code, ws.struct_capacity, n_structures);
    sf_ensure(ws, ws.d_culvert_shape, ws.struct_capacity, n_structures);
    sf_ensure(ws, ws.d_culvert_rise, ws.struct_capacity, n_structures);
    sf_ensure(ws, ws.d_culvert_span, ws.struct_capacity, n_structures);
    sf_ensure(ws, ws.d_culvert_area_m2, ws.struct_capacity, n_structures);
    sf_ensure(ws, ws.d_culvert_barrels, ws.struct_capacity, n_structures);
    sf_ensure(ws, ws.d_culvert_slope, ws.struct_capacity, n_structures);
    sf_ensure(ws, ws.d_inlet_invert_elev, ws.struct_capacity, n_structures);
    sf_ensure(ws, ws.d_outlet_invert_elev, ws.struct_capacity, n_structures);
    sf_ensure(ws, ws.d_entrance_loss_k, ws.struct_capacity, n_structures);
    sf_ensure(ws, ws.d_exit_loss_k, ws.struct_capacity, n_structures);
    sf_ensure(ws, ws.d_embankment_enabled, ws.struct_capacity, n_structures);
    sf_ensure(ws, ws.d_embankment_crest_elev, ws.struct_capacity, n_structures);
    sf_ensure(ws, ws.d_embankment_overflow_width, ws.struct_capacity, n_structures);
    sf_ensure(ws, ws.d_embankment_weir_coeff, ws.struct_capacity, n_structures);
    sf_ensure(ws, ws.d_structure_flow, ws.struct_capacity, n_structures);
    sf_ensure(ws, ws.d_cell_bed, ws.cell_capacity, dev->n_cells);

    // Upload all static structure arrays (these never change during a run).
    sf_upload(structure_type, ws.d_structure_type, n_structures, stream);
    sf_upload(upstream_cell, ws.d_upstream_cell, n_structures, stream);
    sf_upload(downstream_cell, ws.d_downstream_cell, n_structures, stream);
    sf_upload(crest_elev, ws.d_crest_elev, n_structures, stream);
    sf_upload(width, ws.d_width, n_structures, stream);
    sf_upload(height, ws.d_height, n_structures, stream);
    sf_upload(diameter, ws.d_diameter, n_structures, stream);
    sf_upload(length, ws.d_length, n_structures, stream);
    sf_upload(roughness_n, ws.d_roughness_n, n_structures, stream);
    sf_upload(coeff, ws.d_coeff, n_structures, stream);
    sf_upload(cd, ws.d_cd, n_structures, stream);
    sf_upload(opening, ws.d_opening, n_structures, stream);
    sf_upload(q_pump, ws.d_q_pump, n_structures, stream);
    sf_upload(max_flow, ws.d_max_flow, n_structures, stream);
    sf_upload(culvert_code, ws.d_culvert_code, n_structures, stream);
    sf_upload(culvert_shape, ws.d_culvert_shape, n_structures, stream);
    sf_upload(culvert_rise, ws.d_culvert_rise, n_structures, stream);
    sf_upload(culvert_span, ws.d_culvert_span, n_structures, stream);
    sf_upload(culvert_area_m2, ws.d_culvert_area_m2, n_structures, stream);
    sf_upload(culvert_barrels, ws.d_culvert_barrels, n_structures, stream);
    sf_upload(culvert_slope, ws.d_culvert_slope, n_structures, stream);
    sf_upload(inlet_invert_elev, ws.d_inlet_invert_elev, n_structures, stream);
    sf_upload(outlet_invert_elev, ws.d_outlet_invert_elev, n_structures, stream);
    sf_upload(entrance_loss_k, ws.d_entrance_loss_k, n_structures, stream);
    sf_upload(exit_loss_k, ws.d_exit_loss_k, n_structures, stream);
    sf_upload(embankment_enabled, ws.d_embankment_enabled, n_structures, stream);
    sf_upload(embankment_crest_elev, ws.d_embankment_crest_elev, n_structures, stream);
    sf_upload(embankment_overflow_width, ws.d_embankment_overflow_width, n_structures, stream);
    sf_upload(embankment_weir_coeff, ws.d_embankment_weir_coeff, n_structures, stream);

    // cell_bed is also static for coupling purposes
    if (dev->d_cell_zb) {
        CUDA_CHECK(cudaMemcpyAsync(ws.d_cell_bed, dev->d_cell_zb,
                                   static_cast<size_t>(dev->n_cells) * sizeof(double),
                                   cudaMemcpyDeviceToDevice, stream));
    }

    ws.n_structures = n_structures;
    ws.gravity_mps2 = gravity_mps2;
    ws.params_preloaded = true;
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void swe2d_gpu_preload_coupling_cell_area(
    SWE2DDeviceState* dev,
    int32_t n_cells,
    const double* cell_area_m2)
{
    if (!dev) dev = s_coupling_dev;
    if (!dev || n_cells <= 0 || !cell_area_m2) {
        if (!dev && cell_area_m2) {
            throw std::runtime_error(
                "swe2d_gpu_preload_coupling_cell_area: no GPU device state available.");
        }
        return;
    }
    auto& ws = dev->coupling_ws;

    if (ws.cell_capacity < n_cells) {
        if (ws.d_cell_area) cudaFree(ws.d_cell_area);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ws.d_cell_area),
                              static_cast<size_t>(n_cells) * sizeof(double)));
        ws.cell_capacity = n_cells;
    }

    CUDA_CHECK(cudaMemcpyAsync(ws.d_cell_area, cell_area_m2,
                               static_cast<size_t>(n_cells) * sizeof(double),
                               cudaMemcpyHostToDevice, dev->d_stream));
    CUDA_CHECK(cudaStreamSynchronize(dev->d_stream));
}

void swe2d_gpu_compute_coupling_full_on_device(
    SWE2DDeviceState* dev,
    int32_t n_cells,
    int32_t n_structures,
    const double* cell_wse_host,
    int32_t n_inlets,
    const int32_t* inlet_cell,
    const double* inlet_flow_cms)
{
    if (!dev) dev = s_coupling_dev;
    if (!dev) {
        // No device — this is a critical error for the persistent path.
        throw std::runtime_error(
            "swe2d_gpu_compute_coupling_full_on_device: no GPU device state. "
            "Preload must succeed before compute.");
    }
    if (n_cells <= 0 || !dev->d_external_source_mps) return;
    auto& sf_ws = dev->sf_ws;
    auto& cpl_ws = dev->coupling_ws;
    cudaStream_t stream = dev->d_stream;
    constexpr int BLOCK = 256;

    // Zero the external source buffer (will be filled by coupling kernels).
    CUDA_CHECK(cudaMemsetAsync(dev->d_external_source_mps, 0,
                               static_cast<size_t>(n_cells) * sizeof(double), stream));

    // ── Step 1: Upload cell_wse (the ONLY dynamic data per step) ──
    if (cell_wse_host && sf_ws.cell_capacity >= n_cells && sf_ws.d_cell_wse) {
        CUDA_CHECK(cudaMemcpyAsync(sf_ws.d_cell_wse, cell_wse_host,
                                   static_cast<size_t>(n_cells) * sizeof(double),
                                   cudaMemcpyHostToDevice, stream));
    }

    // ── Step 2: Upload inlet data if present ──
    if (n_inlets > 0 && inlet_cell && inlet_flow_cms) {
        if (cpl_ws.inlet_capacity < n_inlets) {
            if (cpl_ws.d_inlet_cell) cudaFree(cpl_ws.d_inlet_cell);
            if (cpl_ws.d_inlet_q) cudaFree(cpl_ws.d_inlet_q);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&cpl_ws.d_inlet_cell),
                                  static_cast<size_t>(n_inlets) * sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&cpl_ws.d_inlet_q),
                                  static_cast<size_t>(n_inlets) * sizeof(double)));
            cpl_ws.inlet_capacity = n_inlets;
        }
        CUDA_CHECK(cudaMemcpyAsync(cpl_ws.d_inlet_cell, inlet_cell,
                                   static_cast<size_t>(n_inlets) * sizeof(int32_t),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(cpl_ws.d_inlet_q, inlet_flow_cms,
                                   static_cast<size_t>(n_inlets) * sizeof(double),
                                   cudaMemcpyHostToDevice, stream));
    }

    // ── Step 3: Run structure-flows kernel from PRELOADED params ──
    if (n_structures > 0 && sf_ws.params_preloaded) {
        CUDA_CHECK(cudaMemsetAsync(sf_ws.d_structure_flow, 0,
                                   static_cast<size_t>(n_structures) * sizeof(double), stream));
        const int grid_sf = (n_structures + BLOCK - 1) / BLOCK;
        swe2d_compute_structure_flows_kernel<<<grid_sf, BLOCK, 0, stream>>>(
            n_cells, n_structures,
            sf_ws.d_cell_wse,
            sf_ws.d_structure_type,
            sf_ws.d_upstream_cell, sf_ws.d_downstream_cell,
            sf_ws.d_crest_elev, sf_ws.d_width, sf_ws.d_height,
            sf_ws.d_diameter, sf_ws.d_length, sf_ws.d_roughness_n,
            sf_ws.d_coeff, sf_ws.d_cd, sf_ws.d_opening,
            sf_ws.d_q_pump, sf_ws.d_max_flow,
            sf_ws.d_culvert_code, sf_ws.d_culvert_shape,
            sf_ws.d_culvert_rise, sf_ws.d_culvert_span, sf_ws.d_culvert_area_m2,
            sf_ws.d_culvert_barrels, sf_ws.d_culvert_slope,
            sf_ws.d_inlet_invert_elev, sf_ws.d_outlet_invert_elev,
            sf_ws.d_entrance_loss_k, sf_ws.d_exit_loss_k,
            sf_ws.d_embankment_enabled, sf_ws.d_embankment_crest_elev,
            sf_ws.d_embankment_overflow_width, sf_ws.d_embankment_weir_coeff,
            sf_ws.gravity_mps2, sf_ws.d_structure_flow,
            s_culvert_solver_mode,
            s_culvert_table_header, s_culvert_table_data,
            s_culvert_table_n_hw, s_culvert_table_n_tw);
        CUDA_CHECK(cudaGetLastError());
    }

    // ── Step 4: Apply structure source rates to cells ──
    if (n_structures > 0 && sf_ws.params_preloaded) {
        // Ensure coupling workspace has struct index buffers.
        if (cpl_ws.structure_capacity < n_structures) {
            if (cpl_ws.d_struct_up) cudaFree(cpl_ws.d_struct_up);
            if (cpl_ws.d_struct_dn) cudaFree(cpl_ws.d_struct_dn);
            if (cpl_ws.d_struct_q) cudaFree(cpl_ws.d_struct_q);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&cpl_ws.d_struct_up),
                                  static_cast<size_t>(n_structures) * sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&cpl_ws.d_struct_dn),
                                  static_cast<size_t>(n_structures) * sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&cpl_ws.d_struct_q),
                                  static_cast<size_t>(n_structures) * sizeof(double)));
            cpl_ws.structure_capacity = n_structures;
        }
        // Copy struct indices from sf_ws to coupling_ws (D2D, fast).
        CUDA_CHECK(cudaMemcpyAsync(cpl_ws.d_struct_up, sf_ws.d_upstream_cell,
                                   static_cast<size_t>(n_structures) * sizeof(int32_t),
                                   cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(cpl_ws.d_struct_dn, sf_ws.d_downstream_cell,
                                   static_cast<size_t>(n_structures) * sizeof(int32_t),
                                   cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(cpl_ws.d_struct_q, sf_ws.d_structure_flow,
                                   static_cast<size_t>(n_structures) * sizeof(double),
                                   cudaMemcpyDeviceToDevice, stream));

        const int grid_struct = (n_structures + BLOCK - 1) / BLOCK;
        swe2d_coupling_structure_source_kernel<<<grid_struct, BLOCK, 0, stream>>>(
            n_structures, cpl_ws.d_struct_up, cpl_ws.d_struct_dn,
            cpl_ws.d_struct_q, cpl_ws.d_cell_area, n_cells,
            dev->d_external_source_mps);
        CUDA_CHECK(cudaGetLastError());
    }

    // ── Step 5: Apply inlet source rates to cells ──
    if (n_inlets > 0 && cpl_ws.d_inlet_cell && cpl_ws.d_inlet_q) {
        if (!cpl_ws.d_cell_area) {
            // cell_area must have been preloaded via preload_coupling_cell_area.
            return;
        }
        const int grid_inlet = (n_inlets + BLOCK - 1) / BLOCK;
        swe2d_coupling_inlet_source_kernel<<<grid_inlet, BLOCK, 0, stream>>>(
            n_inlets, cpl_ws.d_inlet_cell, cpl_ws.d_inlet_q,
            cpl_ws.d_cell_area, n_cells, dev->d_external_source_mps);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

// ─────────────────────────────────────────────────────────────────────────────
// Fused structure-flows + coupling-sources (legacy host-callable path).
// Calls the existing compute_structure_flows then compute_coupling_sources,
// avoiding the intermediate D2H+H2D round trip by reusing device-side buffers.
// ─────────────────────────────────────────────────────────────────────────────
void swe2d_gpu_compute_structure_and_coupling_sources(
    int32_t n_cells,
    const double* cell_area_m2,
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
    const double* culvert_area_m2,
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
    double gravity_mps2,
    int32_t n_inlets,
    const int32_t* inlet_cell,
    const double* inlet_flow_cms,
    double* source_rate_mps_out)
{
    if (!source_rate_mps_out || n_cells <= 0) return;
    std::fill(source_rate_mps_out, source_rate_mps_out + static_cast<size_t>(n_cells), 0.0);

    // Step 1: compute structure flows using the existing host→device function.
    std::vector<double> struct_flows(static_cast<size_t>(n_structures > 0 ? n_structures : 0), 0.0);
    if (n_structures > 0) {
        swe2d_gpu_compute_structure_flows(
            n_cells, n_structures,
            cell_wse, cell_bed,
            structure_type, upstream_cell, downstream_cell,
            crest_elev, width, height, diameter, length, roughness_n,
            coeff, cd, opening, q_pump, max_flow,
            culvert_code, culvert_shape,
            culvert_rise, culvert_span, culvert_area_m2,
            culvert_barrels, culvert_slope,
            inlet_invert_elev, outlet_invert_elev,
            entrance_loss_k, exit_loss_k,
            embankment_enabled, embankment_crest_elev,
            embankment_overflow_width, embankment_weir_coeff,
            gravity_mps2,
            struct_flows.data());
    }

    // Step 2: convert structure flows + inlets into per-cell source rates.
    swe2d_gpu_compute_coupling_sources(
        nullptr,  // dev
        n_cells, cell_area_m2,
        n_inlets, inlet_cell, inlet_flow_cms,
        n_structures,
        n_structures > 0 ? upstream_cell : nullptr,
        n_structures > 0 ? downstream_cell : nullptr,
        n_structures > 0 ? struct_flows.data() : nullptr,
        source_rate_mps_out);
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
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&patch->d_vof_tmp), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&patch->d_vof_sum), sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&patch->d_p_rhs), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&patch->d_p_tmp), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&patch->d_proj_residual_bits), sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&patch->d_active_mask), static_cast<size_t>(patch->n_cells) * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemset(patch->d_u, 0, bytes));
    CUDA_CHECK(cudaMemset(patch->d_v, 0, bytes));
    CUDA_CHECK(cudaMemset(patch->d_w, 0, bytes));
    CUDA_CHECK(cudaMemset(patch->d_p, 0, bytes));
    CUDA_CHECK(cudaMemset(patch->d_vof, 0, bytes));
    CUDA_CHECK(cudaMemset(patch->d_vof_tmp, 0, bytes));
    CUDA_CHECK(cudaMemset(patch->d_vof_sum, 0, sizeof(double)));
    CUDA_CHECK(cudaMemset(patch->d_p_rhs, 0, bytes));
    CUDA_CHECK(cudaMemset(patch->d_p_tmp, 0, bytes));
    CUDA_CHECK(cudaMemset(patch->d_proj_residual_bits, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(patch->d_active_mask, 0, static_cast<size_t>(patch->n_cells) * sizeof(uint8_t)));
    patch->last_projection_iters = 0;
    patch->last_projection_residual = -1.0;
    patch->last_projection_converged = false;
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
        CUDA_CHECK(cudaMemsetAsync(patch->d_vof_tmp, 0, bytes, stream));
        CUDA_CHECK(cudaMemsetAsync(patch->d_vof_sum, 0, sizeof(double), stream));
        CUDA_CHECK(cudaMemsetAsync(patch->d_p_rhs, 0, bytes, stream));
        CUDA_CHECK(cudaMemsetAsync(patch->d_p_tmp, 0, bytes, stream));
        CUDA_CHECK(cudaMemsetAsync(patch->d_proj_residual_bits, 0, sizeof(unsigned long long), stream));
        CUDA_CHECK(cudaMemsetAsync(patch->d_active_mask, 0, static_cast<size_t>(patch->n_cells) * sizeof(uint8_t), stream));
    } else {
        CUDA_CHECK(cudaMemset(patch->d_u, 0, bytes));
        CUDA_CHECK(cudaMemset(patch->d_v, 0, bytes));
        CUDA_CHECK(cudaMemset(patch->d_w, 0, bytes));
        CUDA_CHECK(cudaMemset(patch->d_p, 0, bytes));
        CUDA_CHECK(cudaMemset(patch->d_vof, 0, bytes));
        CUDA_CHECK(cudaMemset(patch->d_vof_tmp, 0, bytes));
        CUDA_CHECK(cudaMemset(patch->d_vof_sum, 0, sizeof(double)));
        CUDA_CHECK(cudaMemset(patch->d_p_rhs, 0, bytes));
        CUDA_CHECK(cudaMemset(patch->d_p_tmp, 0, bytes));
        CUDA_CHECK(cudaMemset(patch->d_proj_residual_bits, 0, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(patch->d_active_mask, 0, static_cast<size_t>(patch->n_cells) * sizeof(uint8_t)));
    }
    patch->last_projection_iters = 0;
    patch->last_projection_residual = -1.0;
    patch->last_projection_converged = false;
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
    if (patch->d_vof_tmp) cudaFree(patch->d_vof_tmp);
    if (patch->d_vof_sum) cudaFree(patch->d_vof_sum);
    if (patch->d_phi) cudaFree(patch->d_phi);
    if (patch->d_ax) cudaFree(patch->d_ax);
    if (patch->d_ay) cudaFree(patch->d_ay);
    if (patch->d_az) cudaFree(patch->d_az);
    if (patch->d_p_rhs) cudaFree(patch->d_p_rhs);
    if (patch->d_p_tmp) cudaFree(patch->d_p_tmp);
    if (patch->d_proj_residual_bits) cudaFree(patch->d_proj_residual_bits);
    if (patch->d_active_mask) cudaFree(patch->d_active_mask);
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
    SWE2DDeviceState* dev,
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
    if (!source_rate_mps_out || n_cells <= 0) return;
    std::fill(source_rate_mps_out, source_rate_mps_out + static_cast<size_t>(n_cells), 0.0);
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

        if (n_structures > 0 && structure_up_cell && structure_down_cell && structure_flow_cms) {
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

        copy_h2d(d_cell_area, cell_area_m2, static_cast<size_t>(n_cells) * sizeof(double));
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
            copy_h2d(d_struct_q, structure_flow_cms, static_cast<size_t>(n_structures) * sizeof(double));
            const int grid = (n_structures + BLOCK - 1) / BLOCK;
            swe2d_coupling_structure_source_kernel<<<grid, BLOCK, 0, stream>>>(
                n_structures, d_struct_up, d_struct_dn, d_struct_q, d_cell_area, n_cells, d_source);
            CUDA_CHECK(cudaGetLastError());
        }

        copy_d2h(source_rate_mps_out, d_source, static_cast<size_t>(n_cells) * sizeof(double));
        if (use_stream) CUDA_CHECK(cudaStreamSynchronize(stream));
    } catch (...) {
        throw;
    }
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

void swe2d_gpu_compute_bridge_coupling_sources(
    SWE2DDeviceState* dev,
    int32_t n_cells,
    const double* cell_area_m2,
    int32_t n_bridges,
    const int32_t* bridge_up_cell,
    const int32_t* bridge_down_cell,
    const double* bridge_flow_cms,
    const double* bridge_loss_k_upstream,
    const double* bridge_loss_k_downstream,
    double bridge_opening_width_m,
    double dt_s,
    double* source_rate_mps_out)
{
    if (!source_rate_mps_out || n_cells <= 0) return;
    std::fill(source_rate_mps_out, source_rate_mps_out + static_cast<size_t>(n_cells), 0.0);
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

        copy_h2d(d_cell_area, cell_area_m2, static_cast<size_t>(n_cells) * sizeof(double));
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

        copy_d2h(source_rate_mps_out, d_source, static_cast<size_t>(n_cells) * sizeof(double));
        if (use_stream) CUDA_CHECK(cudaStreamSynchronize(stream));
    } catch (...) {
        throw;
    }
}

// ── Culvert table-mode globals (file scope, shared across functions) ──
int32_t s_culvert_solver_mode = 0;  // 0=direct, 1=table
double* s_culvert_table_header = nullptr;
double* s_culvert_table_data = nullptr;
int32_t s_culvert_table_n_hw = 32;
int32_t s_culvert_table_n_tw = 16;

// ── Persistent coupling global device pointer ──
SWE2DDeviceState* s_coupling_dev = nullptr;

void swe2d_gpu_set_coupling_device_global(SWE2DDeviceState* dev) {
    s_coupling_dev = dev;
}
SWE2DDeviceState* swe2d_gpu_get_coupling_device_global() {
    return s_coupling_dev;
}

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
    const double* culvert_area_m2,
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
    double gravity_mps2,
    double* structure_flow_cms_out)
{
    if (!structure_flow_cms_out || n_structures <= 0 || n_cells <= 0) {
        return;
    }
    std::fill(structure_flow_cms_out, structure_flow_cms_out + static_cast<size_t>(n_structures), 0.0);

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
    static double* s_d_culvert_area_m2 = nullptr;
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
        if ((cap) < (need)) { \
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
    SF_ENSURE(s_d_culvert_area_m2, s_struct_cap, n_structures, sizeof(double));
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
        if (culvert_area_m2) upload(s_d_culvert_area_m2, culvert_area_m2, n_structures);
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
            s_d_culvert_rise, s_d_culvert_span, s_d_culvert_area_m2,
            s_d_culvert_barrels, s_d_culvert_slope,
            s_d_inlet_invert_elev, s_d_outlet_invert_elev,
            s_d_entrance_loss_k, s_d_exit_loss_k,
            s_d_embankment_enabled, s_d_embankment_crest_elev,
            s_d_embankment_overflow_width, s_d_embankment_weir_coeff,
            gravity_mps2, s_d_structure_flow,
            s_culvert_solver_mode,
            s_culvert_table_header, s_culvert_table_data,
            s_culvert_table_n_hw, s_culvert_table_n_tw);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(structure_flow_cms_out, s_d_structure_flow,
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
        xsect.width_ft = fmax(1.0e-6, span * 3.280839895013123);
        xsect.y_full_ft = fmax(1.0e-6, rise * 3.280839895013123);
        xsect.a_full_ft2 = xsect.width_ft * xsect.y_full_ft;
        xsect.radius_ft = 0.0;
    } else {
        const double dia_ft = fmax(1.0e-6, fmax(culvert_diameter[ci], rise) * 3.280839895013123);
        xsect.radius_ft = 0.5 * dia_ft;
        xsect.y_full_ft = dia_ft;
        xsect.a_full_ft2 = M_PI * xsect.radius_ft * xsect.radius_ft;
        xsect.width_ft = 0.0;
    }

    const double y_full_ft = xsect.y_full_ft;
    const double hw_ft = fmax(0.0, (static_cast<double>(hi) / fmax(1.0, static_cast<double>(n_hw - 1))) * y_full_ft * 2.0);
    const double tw_ft = fmax(0.0, (static_cast<double>(ti) / fmax(1.0, static_cast<double>(n_tw - 1))) * y_full_ft);

    double slope = fmax(1.0e-6, culvert_slope[ci]);
    double len_ft = fmax(0.1, culvert_length[ci] * 3.280839895013123);

    // Inlet control for hint
    const double q_inlet_cfs = fmax(0.0, swe2d_culvert_inlet_controlled_flow_cfs_cuda(xsect, slope, hw_ft));
    const double q_inlet_cms = q_inlet_cfs / 35.31466672148859;

    double orifice_cap = 0.0;
    if (culvert_diameter[ci] > 0.0) {
        const double a_orif = bw2d_circular_area(culvert_diameter[ci]);
        const double dh_orif = fmax(0.0, hw_ft - tw_ft) / 3.280839895013123;
        if (a_orif > 0.0 && dh_orif > 1.0e-12) {
            orifice_cap = a_orif * sqrt(2.0 * 9.81 * dh_orif);
        }
    }
    const double q_hint_cfs = fmax(1.0, fmax(q_inlet_cfs, orifice_cap * 35.31466672148859));

    double q_outlet_cms = 0.0;
    if (hw_ft > 0.0) {
        q_outlet_cms = swe2d_culvert_outlet_control_flow_cms_cuda(
            xsect, hw_ft, tw_ft, len_ft, slope,
            fmax(1.0e-6, culvert_roughness_n[ci]),
            entrance_loss_k[ci], exit_loss_k[ci], q_hint_cfs);
    }

    double q = fmax(0.0, fmin(q_inlet_cms, (q_outlet_cms > 0.0) ? q_outlet_cms : q_inlet_cms));
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
    (void)d_u_3d;
    (void)d_p_3d;
    (void)g;
    (void)dt;

    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_faces) return;

    const int32_t c = d_cell2d ? d_cell2d[i] : -1;
    if (!d_h_2d || !d_hu_2d || !d_hv_2d || !d_cell_area_2d ||
        !d_face_area || !d_face_nx || !d_face_ny || !d_face_nz ||
        c < 0) {
        d_flux_mass_2d_to_3d[i] = 0.0;
        d_flux_momx_2d_to_3d[i] = 0.0;
        d_flux_momy_2d_to_3d[i] = 0.0;
        d_head_loss_3d_to2d[i] = 0.0;
        return;
    }

    const double h2d = fmax(0.0, d_h_2d[c]);
    const double hu2d = d_hu_2d[c];
    const double hv2d = d_hv_2d[c];
    const double nx = d_face_nx[i];
    const double ny = d_face_ny[i];
    const double nz = d_face_nz[i];
    const double n_norm = fmax(1.0e-12, sqrt(nx * nx + ny * ny + nz * nz));
    const double nxn = nx / n_norm;
    const double nyn = ny / n_norm;
    const double area = fmax(0.0, d_face_area[i]);

    double u2d = 0.0;
    double v2d = 0.0;
    if (h2d > 1.0e-9) {
        u2d = hu2d / h2d;
        v2d = hv2d / h2d;
    }
    const double qn2d = u2d * nxn + v2d * nyn;
    const double mass_flux = h2d * qn2d * area;

    d_flux_mass_2d_to_3d[i] = mass_flux;
    d_flux_momx_2d_to_3d[i] = mass_flux * u2d;
    d_flux_momy_2d_to_3d[i] = mass_flux * v2d;
    d_head_loss_3d_to2d[i] = 0.0;
}

__device__ __forceinline__ int32_t swe3d_face_from_contract_normal(
    double nx,
    double ny,
    double nz)
{
    const double ax = fabs(nx);
    const double ay = fabs(ny);
    const double az = fabs(nz);
    if (ax >= ay && ax >= az) {
        return (nx >= 0.0)
            ? static_cast<int32_t>(SWE3DPatchBoundaryFace::XMAX)
            : static_cast<int32_t>(SWE3DPatchBoundaryFace::XMIN);
    }
    if (ay >= ax && ay >= az) {
        return (ny >= 0.0)
            ? static_cast<int32_t>(SWE3DPatchBoundaryFace::YMAX)
            : static_cast<int32_t>(SWE3DPatchBoundaryFace::YMIN);
    }
    return (nz >= 0.0)
        ? static_cast<int32_t>(SWE3DPatchBoundaryFace::ZMAX)
        : static_cast<int32_t>(SWE3DPatchBoundaryFace::ZMIN);
}

__global__ void swe3d_reduce_contract_to_face_targets_kernel(
    int32_t n_faces,
    const int32_t* d_cell2d,
    const double* d_h_2d,
    const double* d_hu_2d,
    const double* d_hv_2d,
    const double* d_face_area,
    const double* d_face_nx,
    const double* d_face_ny,
    const double* d_face_nz,
    double patch_height,
    double* d_face_sum_area,
    double* d_face_sum_u,
    double* d_face_sum_v,
    double* d_face_sum_w,
    double* d_face_sum_vof)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_faces) return;
    if (!d_cell2d || !d_h_2d || !d_hu_2d || !d_hv_2d ||
        !d_face_area || !d_face_nx || !d_face_ny || !d_face_nz) {
        return;
    }

    const int32_t c = d_cell2d[i];
    if (c < 0) return;
    const double area = fmax(0.0, d_face_area[i]);
    if (area <= 0.0) return;

    const int32_t face = swe3d_face_from_contract_normal(d_face_nx[i], d_face_ny[i], d_face_nz[i]);
    const double h2d = fmax(0.0, d_h_2d[c]);
    double u2d = 0.0;
    double v2d = 0.0;
    if (h2d > 1.0e-9) {
        u2d = d_hu_2d[c] / h2d;
        v2d = d_hv_2d[c] / h2d;
    }
    const double vof_target = fmin(1.0, fmax(0.0, h2d / fmax(patch_height, 1.0e-9)));

    atomicAdd(&d_face_sum_area[face], area);
    atomicAdd(&d_face_sum_u[face], area * u2d);
    atomicAdd(&d_face_sum_v[face], area * v2d);
    atomicAdd(&d_face_sum_w[face], 0.0);
    atomicAdd(&d_face_sum_vof[face], area * vof_target);
}

__device__ __forceinline__ void swe3d_apply_face_target_blend(
    int32_t face,
    double alpha,
    double phi_cap,
    const double* d_face_sum_area,
    const double* d_face_sum_u,
    const double* d_face_sum_v,
    const double* d_face_sum_w,
    const double* d_face_sum_vof,
    double& u,
    double& v,
    double& w,
    double& vof)
{
    const double area = d_face_sum_area[face];
    if (area <= 1.0e-12) return;
    const double inv = 1.0 / area;
    const double u_t = d_face_sum_u[face] * inv;
    const double v_t = d_face_sum_v[face] * inv;
    const double w_t = d_face_sum_w[face] * inv;
    const double vof_t = fmin(phi_cap, fmax(0.0, d_face_sum_vof[face] * inv));

    u = (1.0 - alpha) * u + alpha * u_t;
    v = (1.0 - alpha) * v + alpha * v_t;
    w = (1.0 - alpha) * w + alpha * w_t;
    vof = (1.0 - alpha) * vof + alpha * vof_t;
}

__global__ void swe3d_apply_one_way_contract_forcing_kernel(
    SWE3DCartesianPatchDesc desc,
    int64_t n_cells,
    double dt,
    const double* d_phi,
    double* d_u,
    double* d_v,
    double* d_w,
    double* d_vof,
    const double* d_face_sum_area,
    const double* d_face_sum_u,
    const double* d_face_sum_v,
    const double* d_face_sum_w,
    const double* d_face_sum_vof)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
                      static_cast<int64_t>(threadIdx.x);
    if (i >= n_cells) return;

    const int32_t nx = desc.nx;
    const int32_t ny = desc.ny;
    const int32_t nz = desc.nz;
    const int32_t plane = nx * ny;
    const int32_t iz = static_cast<int32_t>(i / plane);
    const int32_t rem = static_cast<int32_t>(i - static_cast<int64_t>(iz) * plane);
    const int32_t iy = rem / nx;
    const int32_t ix = rem - iy * nx;

    const double phi_cap = d_phi ? fmin(1.0, fmax(0.0, d_phi[i])) : 1.0;
    if (phi_cap <= 1.0e-9) {
        d_u[i] = 0.0;
        d_v[i] = 0.0;
        d_w[i] = 0.0;
        d_vof[i] = 0.0;
        return;
    }

    const double alpha = fmin(1.0, fmax(0.0, dt / 0.2));
    double u = d_u[i];
    double v = d_v[i];
    double w = d_w[i];
    double vof = d_vof[i];

    if (ix == 0) {
        swe3d_apply_face_target_blend(
            static_cast<int32_t>(SWE3DPatchBoundaryFace::XMIN),
            alpha,
            phi_cap,
            d_face_sum_area,
            d_face_sum_u,
            d_face_sum_v,
            d_face_sum_w,
            d_face_sum_vof,
            u,
            v,
            w,
            vof);
    }
    if (ix + 1 == nx) {
        swe3d_apply_face_target_blend(
            static_cast<int32_t>(SWE3DPatchBoundaryFace::XMAX),
            alpha,
            phi_cap,
            d_face_sum_area,
            d_face_sum_u,
            d_face_sum_v,
            d_face_sum_w,
            d_face_sum_vof,
            u,
            v,
            w,
            vof);
    }
    if (iy == 0) {
        swe3d_apply_face_target_blend(
            static_cast<int32_t>(SWE3DPatchBoundaryFace::YMIN),
            alpha,
            phi_cap,
            d_face_sum_area,
            d_face_sum_u,
            d_face_sum_v,
            d_face_sum_w,
            d_face_sum_vof,
            u,
            v,
            w,
            vof);
    }
    if (iy + 1 == ny) {
        swe3d_apply_face_target_blend(
            static_cast<int32_t>(SWE3DPatchBoundaryFace::YMAX),
            alpha,
            phi_cap,
            d_face_sum_area,
            d_face_sum_u,
            d_face_sum_v,
            d_face_sum_w,
            d_face_sum_vof,
            u,
            v,
            w,
            vof);
    }
    if (iz == 0) {
        swe3d_apply_face_target_blend(
            static_cast<int32_t>(SWE3DPatchBoundaryFace::ZMIN),
            alpha,
            phi_cap,
            d_face_sum_area,
            d_face_sum_u,
            d_face_sum_v,
            d_face_sum_w,
            d_face_sum_vof,
            u,
            v,
            w,
            vof);
    }
    if (iz + 1 == nz) {
        swe3d_apply_face_target_blend(
            static_cast<int32_t>(SWE3DPatchBoundaryFace::ZMAX),
            alpha,
            phi_cap,
            d_face_sum_area,
            d_face_sum_u,
            d_face_sum_v,
            d_face_sum_w,
            d_face_sum_vof,
            u,
            v,
            w,
            vof);
    }

    if (!isfinite(u)) u = 0.0;
    if (!isfinite(v)) v = 0.0;
    if (!isfinite(w)) w = 0.0;
    if (!isfinite(vof)) vof = 0.0;
    d_u[i] = u;
    d_v[i] = v;
    d_w[i] = w;
    d_vof[i] = fmin(phi_cap, fmax(0.0, vof));
}

__global__ void swe3d_collect_boundary_feedback_flux_kernel(
    SWE3DCartesianPatchDesc desc,
    int64_t n_cells,
    const double* d_phi,
    const double* d_vof,
    const double* d_u,
    const double* d_v,
    const double* d_w,
    const double* d_p,
    double* d_face_sum_qn,
    double* d_face_sum_p,
    double* d_face_sum_wet_area)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
                      static_cast<int64_t>(threadIdx.x);
    if (i >= n_cells) return;

    const double phi = d_phi ? fmin(1.0, fmax(0.0, d_phi[i])) : 1.0;
    const double vof = d_vof ? fmin(1.0, fmax(0.0, d_vof[i])) : 1.0;
    const double wet = phi * vof;
    if (wet <= 1.0e-9) return;

    const int32_t nx = desc.nx;
    const int32_t ny = desc.ny;
    const int32_t nz = desc.nz;
    const int32_t plane = nx * ny;
    const int32_t iz = static_cast<int32_t>(i / plane);
    const int32_t rem = static_cast<int32_t>(i - static_cast<int64_t>(iz) * plane);
    const int32_t iy = rem / nx;
    const int32_t ix = rem - iy * nx;

    const double u = d_u[i];
    const double v = d_v[i];
    const double w = d_w[i];
    const double p = d_p[i];
    const double area_x = fmax(0.0, desc.dy * desc.dz);
    const double area_y = fmax(0.0, desc.dx * desc.dz);
    const double area_z = fmax(0.0, desc.dx * desc.dy);

    if (ix == 0) {
        const int32_t face = static_cast<int32_t>(SWE3DPatchBoundaryFace::XMIN);
        const double wet_area = wet * area_x;
        atomicAdd(&d_face_sum_qn[face], -u * wet_area);
        atomicAdd(&d_face_sum_p[face], p * wet_area);
        atomicAdd(&d_face_sum_wet_area[face], wet_area);
    }
    if (ix + 1 == nx) {
        const int32_t face = static_cast<int32_t>(SWE3DPatchBoundaryFace::XMAX);
        const double wet_area = wet * area_x;
        atomicAdd(&d_face_sum_qn[face], u * wet_area);
        atomicAdd(&d_face_sum_p[face], p * wet_area);
        atomicAdd(&d_face_sum_wet_area[face], wet_area);
    }
    if (iy == 0) {
        const int32_t face = static_cast<int32_t>(SWE3DPatchBoundaryFace::YMIN);
        const double wet_area = wet * area_y;
        atomicAdd(&d_face_sum_qn[face], -v * wet_area);
        atomicAdd(&d_face_sum_p[face], p * wet_area);
        atomicAdd(&d_face_sum_wet_area[face], wet_area);
    }
    if (iy + 1 == ny) {
        const int32_t face = static_cast<int32_t>(SWE3DPatchBoundaryFace::YMAX);
        const double wet_area = wet * area_y;
        atomicAdd(&d_face_sum_qn[face], v * wet_area);
        atomicAdd(&d_face_sum_p[face], p * wet_area);
        atomicAdd(&d_face_sum_wet_area[face], wet_area);
    }
    if (iz == 0) {
        const int32_t face = static_cast<int32_t>(SWE3DPatchBoundaryFace::ZMIN);
        const double wet_area = wet * area_z;
        atomicAdd(&d_face_sum_qn[face], -w * wet_area);
        atomicAdd(&d_face_sum_p[face], p * wet_area);
        atomicAdd(&d_face_sum_wet_area[face], wet_area);
    }
    if (iz + 1 == nz) {
        const int32_t face = static_cast<int32_t>(SWE3DPatchBoundaryFace::ZMAX);
        const double wet_area = wet * area_z;
        atomicAdd(&d_face_sum_qn[face], w * wet_area);
        atomicAdd(&d_face_sum_p[face], p * wet_area);
        atomicAdd(&d_face_sum_wet_area[face], wet_area);
    }
}

__global__ void swe2d_apply_two_way_contract_feedback_kernel(
    int32_t n_faces,
    const int32_t* d_cell2d,
    const double* d_face_area,
    const double* d_face_nx,
    const double* d_face_ny,
    const double* d_face_nz,
    const double* d_cell_area,
    const double* d_face_sum_qn,
    const double* d_face_sum_p,
    const double* d_face_sum_wet_area,
    double dt,
    double g,
    double* d_h,
    double* d_hu,
    double* d_hv,
    double* d_head_loss_3d_to2d)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_faces) return;
    if (!d_cell2d || !d_face_area || !d_face_nx || !d_face_ny || !d_face_nz ||
        !d_cell_area || !d_h || !d_hu || !d_hv || !d_head_loss_3d_to2d) {
        return;
    }

    const int32_t c = d_cell2d[i];
    if (c < 0) {
        d_head_loss_3d_to2d[i] = 0.0;
        return;
    }

    const double nx = d_face_nx[i];
    const double ny = d_face_ny[i];
    const double nz = d_face_nz[i];
    const int32_t face = swe3d_face_from_contract_normal(nx, ny, nz);
    const double wet_area = d_face_sum_wet_area[face];
    if (wet_area <= 1.0e-9) {
        d_head_loss_3d_to2d[i] = 0.0;
        return;
    }

    const double qn3d = d_face_sum_qn[face] / wet_area;
    const double p3d = d_face_sum_p[face] / wet_area;
    const double n_norm = fmax(1.0e-12, sqrt(nx * nx + ny * ny + nz * nz));
    const double nxn = nx / n_norm;
    const double nyn = ny / n_norm;
    const double area = fmax(0.0, d_face_area[i]);
    const double cell_area = fmax(1.0e-9, d_cell_area[c]);
    const double h = fmax(0.0, d_h[c]);

    double u2d = 0.0;
    double v2d = 0.0;
    if (h > 1.0e-9) {
        u2d = d_hu[c] / h;
        v2d = d_hv[c] / h;
    }
    const double qn2d = u2d * nxn + v2d * nyn;

    const double relax = 0.25;
    double dQ = (qn3d - qn2d) * area * relax;
    const double max_outflow_rate = h * cell_area / fmax(dt, 1.0e-9);
    if (dQ < -max_outflow_rate) {
        dQ = -max_outflow_rate;
    }

    const double dh = dt * dQ / cell_area;
    atomicAdd(&d_h[c], dh);

    // Flux-form momentum exchange uses donor-side advective momentum.
    double u_donor = 0.0;
    double v_donor = 0.0;
    if (dQ >= 0.0) {
        u_donor = qn3d * nxn;
        v_donor = qn3d * nyn;
    } else {
        u_donor = u2d;
        v_donor = v2d;
    }
    atomicAdd(&d_hu[c], dt * dQ * u_donor / cell_area);
    atomicAdd(&d_hv[c], dt * dQ * v_donor / cell_area);

    d_head_loss_3d_to2d[i] = fmax(0.0, p3d / fmax(g, 1.0e-9));
}

__global__ void swe2d_coupling_clamp_2d_state_kernel(
    int32_t n_cells,
    double h_min,
    double* d_h,
    double* d_hu,
    double* d_hv)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cells) return;

    double h = d_h[i];
    if (!isfinite(h) || h < 0.0) {
        h = 0.0;
    }
    d_h[i] = h;
    if (h < h_min) {
        d_hu[i] = 0.0;
        d_hv[i] = 0.0;
    }
}

void swe2d_gpu_apply_2d3d_exchange_skeleton(
    SWE2DDeviceState* dev,
    double dt,
    double g,
    int coupling_mode,
    bool apply_head_loss_to_2d_rhs,
    SWE2DStepDiag* diag)
{
    (void)apply_head_loss_to_2d_rhs;
    (void)diag;

    if (coupling_mode == static_cast<int>(SWE2DThreeDCouplingMode::OFF)) {
        return;
    }

    if (!dev || !dev->coupling_iface || !dev->patch3d) {
        return;
    }

    auto* iface = dev->coupling_iface;
    const int n_faces = iface->n_faces;
    if (n_faces <= 0) return;
    if (dt <= 0.0) return;

    constexpr int BLOCK = 256;
    const int grid_faces = (n_faces + BLOCK - 1) / BLOCK;
    const int grid_patch = static_cast<int>((dev->patch3d->n_cells + static_cast<int64_t>(BLOCK) - 1) / static_cast<int64_t>(BLOCK));
    const size_t face_bytes = static_cast<size_t>(SWE3D_PATCH_FACE_COUNT) * sizeof(double);

    double* d_face_sum_area = nullptr;
    double* d_face_sum_u = nullptr;
    double* d_face_sum_v = nullptr;
    double* d_face_sum_w = nullptr;
    double* d_face_sum_vof = nullptr;
    double* d_face_sum_qn = nullptr;
    double* d_face_sum_p = nullptr;
    double* d_face_sum_wet_area = nullptr;

    try {
        CUDA_CHECK(cudaMemsetAsync(iface->d_flux_mass_2d_to_3d, 0, static_cast<size_t>(n_faces) * sizeof(double), dev->d_stream));
        CUDA_CHECK(cudaMemsetAsync(iface->d_flux_momx_2d_to_3d, 0, static_cast<size_t>(n_faces) * sizeof(double), dev->d_stream));
        CUDA_CHECK(cudaMemsetAsync(iface->d_flux_momy_2d_to_3d, 0, static_cast<size_t>(n_faces) * sizeof(double), dev->d_stream));
        CUDA_CHECK(cudaMemsetAsync(iface->d_head_loss_3d_to2d, 0, static_cast<size_t>(n_faces) * sizeof(double), dev->d_stream));

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_face_sum_area), face_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_face_sum_u), face_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_face_sum_v), face_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_face_sum_w), face_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_face_sum_vof), face_bytes));
        CUDA_CHECK(cudaMemsetAsync(d_face_sum_area, 0, face_bytes, dev->d_stream));
        CUDA_CHECK(cudaMemsetAsync(d_face_sum_u, 0, face_bytes, dev->d_stream));
        CUDA_CHECK(cudaMemsetAsync(d_face_sum_v, 0, face_bytes, dev->d_stream));
        CUDA_CHECK(cudaMemsetAsync(d_face_sum_w, 0, face_bytes, dev->d_stream));
        CUDA_CHECK(cudaMemsetAsync(d_face_sum_vof, 0, face_bytes, dev->d_stream));

        swe3d_exchange_kernel_skeleton<<<grid_faces, BLOCK, 0, dev->d_stream>>>(
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

        const double patch_height = std::max(
            dev->patch3d->desc.dz * static_cast<double>(dev->patch3d->desc.nz),
            1.0e-9);

        swe3d_reduce_contract_to_face_targets_kernel<<<grid_faces, BLOCK, 0, dev->d_stream>>>(
            n_faces,
            iface->d_cell2d,
            dev->d_h,
            dev->d_hu,
            dev->d_hv,
            iface->d_face_area,
            iface->d_face_nx,
            iface->d_face_ny,
            iface->d_face_nz,
            patch_height,
            d_face_sum_area,
            d_face_sum_u,
            d_face_sum_v,
            d_face_sum_w,
            d_face_sum_vof);
        CUDA_CHECK(cudaGetLastError());

        swe3d_apply_one_way_contract_forcing_kernel<<<grid_patch, BLOCK, 0, dev->d_stream>>>(
            dev->patch3d->desc,
            dev->patch3d->n_cells,
            dt,
            dev->patch3d->d_phi,
            dev->patch3d->d_u,
            dev->patch3d->d_v,
            dev->patch3d->d_w,
            dev->patch3d->d_vof,
            d_face_sum_area,
            d_face_sum_u,
            d_face_sum_v,
            d_face_sum_w,
            d_face_sum_vof);
        CUDA_CHECK(cudaGetLastError());

        const bool two_way =
            coupling_mode == static_cast<int>(SWE2DThreeDCouplingMode::TWO_WAY_2D_3D);
        if (two_way) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_face_sum_qn), face_bytes));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_face_sum_p), face_bytes));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_face_sum_wet_area), face_bytes));
            CUDA_CHECK(cudaMemsetAsync(d_face_sum_qn, 0, face_bytes, dev->d_stream));
            CUDA_CHECK(cudaMemsetAsync(d_face_sum_p, 0, face_bytes, dev->d_stream));
            CUDA_CHECK(cudaMemsetAsync(d_face_sum_wet_area, 0, face_bytes, dev->d_stream));

            swe3d_collect_boundary_feedback_flux_kernel<<<grid_patch, BLOCK, 0, dev->d_stream>>>(
                dev->patch3d->desc,
                dev->patch3d->n_cells,
                dev->patch3d->d_phi,
                dev->patch3d->d_vof,
                dev->patch3d->d_u,
                dev->patch3d->d_v,
                dev->patch3d->d_w,
                dev->patch3d->d_p,
                d_face_sum_qn,
                d_face_sum_p,
                d_face_sum_wet_area);
            CUDA_CHECK(cudaGetLastError());

            swe2d_apply_two_way_contract_feedback_kernel<<<grid_faces, BLOCK, 0, dev->d_stream>>>(
                n_faces,
                iface->d_cell2d,
                iface->d_face_area,
                iface->d_face_nx,
                iface->d_face_ny,
                iface->d_face_nz,
                dev->d_cell_area,
                d_face_sum_qn,
                d_face_sum_p,
                d_face_sum_wet_area,
                dt,
                g,
                dev->d_h,
                dev->d_hu,
                dev->d_hv,
                iface->d_head_loss_3d_to2d);
            CUDA_CHECK(cudaGetLastError());

            const int grid_2d_cells = (dev->n_cells + BLOCK - 1) / BLOCK;
            swe2d_coupling_clamp_2d_state_kernel<<<grid_2d_cells, BLOCK, 0, dev->d_stream>>>(
                dev->n_cells,
                1.0e-6,
                dev->d_h,
                dev->d_hu,
                dev->d_hv);
            CUDA_CHECK(cudaGetLastError());
        }
    } catch (...) {
        if (d_face_sum_area) cudaFree(d_face_sum_area);
        if (d_face_sum_u) cudaFree(d_face_sum_u);
        if (d_face_sum_v) cudaFree(d_face_sum_v);
        if (d_face_sum_w) cudaFree(d_face_sum_w);
        if (d_face_sum_vof) cudaFree(d_face_sum_vof);
        if (d_face_sum_qn) cudaFree(d_face_sum_qn);
        if (d_face_sum_p) cudaFree(d_face_sum_p);
        if (d_face_sum_wet_area) cudaFree(d_face_sum_wet_area);
        throw;
    }

    if (d_face_sum_area) cudaFree(d_face_sum_area);
    if (d_face_sum_u) cudaFree(d_face_sum_u);
    if (d_face_sum_v) cudaFree(d_face_sum_v);
    if (d_face_sum_w) cudaFree(d_face_sum_w);
    if (d_face_sum_vof) cudaFree(d_face_sum_vof);
    if (d_face_sum_qn) cudaFree(d_face_sum_qn);
    if (d_face_sum_p) cudaFree(d_face_sum_p);
    if (d_face_sum_wet_area) cudaFree(d_face_sum_wet_area);
}

__device__ __forceinline__ int64_t swe3d_flat_idx(
    int32_t ix,
    int32_t iy,
    int32_t iz,
    int32_t nx,
    int32_t ny)
{
    return static_cast<int64_t>(iz) * static_cast<int64_t>(nx) * static_cast<int64_t>(ny) +
           static_cast<int64_t>(iy) * static_cast<int64_t>(nx) +
           static_cast<int64_t>(ix);
}

__device__ __forceinline__ double swe3d_clamp01(double v)
{
    return fmin(1.0, fmax(0.0, v));
}

__device__ __forceinline__ bool swe3d_is_inflow_mode(int32_t mode)
{
    return mode == static_cast<int32_t>(SWE3DBoundaryMode::INFLOW) ||
           mode == static_cast<int32_t>(SWE3DBoundaryMode::INFLOW_FLOW_RATE);
}

__device__ __forceinline__ bool swe3d_interface_band(double vof, double alpha_gas, double alpha_wet)
{
    return vof > alpha_gas && vof < alpha_wet;
}

__device__ __forceinline__ double swe3d_minmod(double a, double b)
{
    if (a * b <= 0.0) return 0.0;
    return (fabs(a) < fabs(b)) ? a : b;
}

__device__ __forceinline__ double swe3d_vof_cell_clamped(
    int64_t idx,
    const double* d_phi,
    const double* d_vof)
{
    const double phi = d_phi ? swe3d_clamp01(d_phi[idx]) : 1.0;
    return fmin(phi, fmax(0.0, d_vof[idx]));
}

__device__ __forceinline__ double swe3d_face_activity_weight(
    const double* d_vof,
    const uint8_t* d_active,
    int64_t ia,
    int64_t ib)
{
    const bool act_a = !d_active || d_active[ia] != 0;
    const bool act_b = !d_active || d_active[ib] != 0;
    if (!act_a && !act_b) return 0.0;

    if (!d_vof) {
        return (act_a || act_b) ? 1.0 : 0.0;
    }

    const double va = swe3d_clamp01(d_vof[ia]);
    const double vb = swe3d_clamp01(d_vof[ib]);
    double w = 0.5 * (va + vb);
    if (act_a != act_b) {
        w = fmax(w, 0.1);
    }
    return fmin(1.0, fmax(0.0, w));
}

__global__ void swe3d_build_active_mask_kernel(
    SWE3DCartesianPatchDesc desc,
    int64_t n_cells,
    const double* d_phi,
    const double* d_vof,
    double alpha_wet,
    double alpha_gas,
    uint8_t* d_active_mask)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
                      static_cast<int64_t>(threadIdx.x);
    if (i >= n_cells) return;

    const int32_t nx = desc.nx;
    const int32_t ny = desc.ny;
    const int32_t nz = desc.nz;
    const int32_t plane = nx * ny;
    const int32_t iz = static_cast<int32_t>(i / plane);
    const int32_t rem = static_cast<int32_t>(i - static_cast<int64_t>(iz) * plane);
    const int32_t iy = rem / nx;
    const int32_t ix = rem - iy * nx;

    const double phi = d_phi ? swe3d_clamp01(d_phi[i]) : 1.0;
    if (phi <= 1.0e-9) {
        d_active_mask[i] = 0;
        return;
    }

    const double vof_i = swe3d_vof_cell_clamped(i, d_phi, d_vof);
    bool active = (vof_i >= alpha_wet) || swe3d_interface_band(vof_i, alpha_gas, alpha_wet);

    auto neighbor_is_wet_or_interface = [&](int32_t x, int32_t y, int32_t z) -> bool {
        if (x < 0 || y < 0 || z < 0 || x >= nx || y >= ny || z >= nz) return false;
        const int64_t j = swe3d_flat_idx(x, y, z, nx, ny);
        const double phi_j = d_phi ? swe3d_clamp01(d_phi[j]) : 1.0;
        if (phi_j <= 1.0e-9) return false;
        const double vof_j = swe3d_vof_cell_clamped(j, d_phi, d_vof);
        return (vof_j > alpha_gas);
    };

    if (!active) {
        active =
            neighbor_is_wet_or_interface(ix - 1, iy, iz) ||
            neighbor_is_wet_or_interface(ix + 1, iy, iz) ||
            neighbor_is_wet_or_interface(ix, iy - 1, iz) ||
            neighbor_is_wet_or_interface(ix, iy + 1, iz) ||
            neighbor_is_wet_or_interface(ix, iy, iz - 1) ||
            neighbor_is_wet_or_interface(ix, iy, iz + 1);
    }

    if (!active) {
        if (ix == 0 && swe3d_is_inflow_mode(desc.bc_mode[static_cast<int32_t>(SWE3DPatchBoundaryFace::XMIN)])) active = true;
        if (ix + 1 == nx && swe3d_is_inflow_mode(desc.bc_mode[static_cast<int32_t>(SWE3DPatchBoundaryFace::XMAX)])) active = true;
        if (iy == 0 && swe3d_is_inflow_mode(desc.bc_mode[static_cast<int32_t>(SWE3DPatchBoundaryFace::YMIN)])) active = true;
        if (iy + 1 == ny && swe3d_is_inflow_mode(desc.bc_mode[static_cast<int32_t>(SWE3DPatchBoundaryFace::YMAX)])) active = true;
        if (iz == 0 && swe3d_is_inflow_mode(desc.bc_mode[static_cast<int32_t>(SWE3DPatchBoundaryFace::ZMIN)])) active = true;
        if (iz + 1 == nz && swe3d_is_inflow_mode(desc.bc_mode[static_cast<int32_t>(SWE3DPatchBoundaryFace::ZMAX)])) active = true;
    }

    d_active_mask[i] = active ? 1 : 0;
}

__device__ __forceinline__ double swe3d_patch_face_total_area(
    SWE3DCartesianPatchDesc desc,
    int32_t face)
{
    if (face == static_cast<int32_t>(SWE3DPatchBoundaryFace::XMIN) ||
        face == static_cast<int32_t>(SWE3DPatchBoundaryFace::XMAX)) {
        return fmax(1.0e-9, static_cast<double>(desc.ny) * static_cast<double>(desc.nz) * desc.dy * desc.dz);
    }
    if (face == static_cast<int32_t>(SWE3DPatchBoundaryFace::YMIN) ||
        face == static_cast<int32_t>(SWE3DPatchBoundaryFace::YMAX)) {
        return fmax(1.0e-9, static_cast<double>(desc.nx) * static_cast<double>(desc.nz) * desc.dx * desc.dz);
    }
    return fmax(1.0e-9, static_cast<double>(desc.nx) * static_cast<double>(desc.ny) * desc.dx * desc.dy);
}

__global__ void swe3d_reduce_face_effective_area_kernel(
    SWE3DCartesianPatchDesc desc,
    int64_t n_cells,
    const double* d_phi,
    const double* d_vof,
    const double* d_ax,
    const double* d_ay,
    const double* d_az,
    const uint8_t* d_active_mask,
    double* d_face_effective_area)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
                      static_cast<int64_t>(threadIdx.x);
    if (i >= n_cells || !d_face_effective_area) return;

    if (d_active_mask && d_active_mask[i] == 0) return;

    const int32_t nx = desc.nx;
    const int32_t ny = desc.ny;
    const int32_t nz = desc.nz;
    const int32_t plane = nx * ny;
    const int32_t iz = static_cast<int32_t>(i / plane);
    const int32_t rem = static_cast<int32_t>(i - static_cast<int64_t>(iz) * plane);
    const int32_t iy = rem / nx;
    const int32_t ix = rem - iy * nx;

    const double phi = d_phi ? swe3d_clamp01(d_phi[i]) : 1.0;
    if (phi <= 1.0e-9) return;

    const double vof = swe3d_vof_cell_clamped(i, d_phi, d_vof);
    const double wet_weight = fmin(1.0, fmax(0.0, vof / fmax(phi, 1.0e-6)));
    if (wet_weight <= 0.0) return;

    const double ax = d_ax ? swe3d_clamp01(d_ax[i]) : 1.0;
    const double ay = d_ay ? swe3d_clamp01(d_ay[i]) : 1.0;
    const double az = d_az ? swe3d_clamp01(d_az[i]) : 1.0;

    const double area_x = wet_weight * ax * fmax(desc.dy * desc.dz, 1.0e-12);
    const double area_y = wet_weight * ay * fmax(desc.dx * desc.dz, 1.0e-12);
    const double area_z = wet_weight * az * fmax(desc.dx * desc.dy, 1.0e-12);

    if (ix == 0) {
        atomicAdd(&d_face_effective_area[static_cast<int32_t>(SWE3DPatchBoundaryFace::XMIN)], area_x);
    }
    if (ix + 1 == nx) {
        atomicAdd(&d_face_effective_area[static_cast<int32_t>(SWE3DPatchBoundaryFace::XMAX)], area_x);
    }
    if (iy == 0) {
        atomicAdd(&d_face_effective_area[static_cast<int32_t>(SWE3DPatchBoundaryFace::YMIN)], area_y);
    }
    if (iy + 1 == ny) {
        atomicAdd(&d_face_effective_area[static_cast<int32_t>(SWE3DPatchBoundaryFace::YMAX)], area_y);
    }
    if (iz == 0) {
        atomicAdd(&d_face_effective_area[static_cast<int32_t>(SWE3DPatchBoundaryFace::ZMIN)], area_z);
    }
    if (iz + 1 == nz) {
        atomicAdd(&d_face_effective_area[static_cast<int32_t>(SWE3DPatchBoundaryFace::ZMAX)], area_z);
    }
}

__device__ __forceinline__ double swe3d_boundary_face_velocity_component(
    SWE3DCartesianPatchDesc desc,
    int32_t face,
    double u,
    double v,
    double w,
    const double* d_face_effective_area,
    int32_t q_inflow_area_policy,
    double free_surface_vent_bias)
{
    const int32_t mode = desc.bc_mode[face];
    const int32_t wall_mode = static_cast<int32_t>(SWE3DBoundaryMode::WALL);
    const int32_t inflow_mode = static_cast<int32_t>(SWE3DBoundaryMode::INFLOW);
    const int32_t outflow_mode = static_cast<int32_t>(SWE3DBoundaryMode::OUTFLOW);
    const int32_t free_surface_mode = static_cast<int32_t>(SWE3DBoundaryMode::FREE_SURFACE);
    const int32_t inflow_q_mode = static_cast<int32_t>(SWE3DBoundaryMode::INFLOW_FLOW_RATE);

    double inside = 0.0;
    if (face == static_cast<int32_t>(SWE3DPatchBoundaryFace::XMIN) ||
        face == static_cast<int32_t>(SWE3DPatchBoundaryFace::XMAX)) {
        inside = u;
    } else if (face == static_cast<int32_t>(SWE3DPatchBoundaryFace::YMIN) ||
               face == static_cast<int32_t>(SWE3DPatchBoundaryFace::YMAX)) {
        inside = v;
    } else {
        inside = w;
    }

    if (mode == wall_mode) {
        return 0.0;
    }
    if (mode == outflow_mode) {
        return inside;
    }
    if (mode == free_surface_mode) {
        if (face == static_cast<int32_t>(SWE3DPatchBoundaryFace::ZMAX)) {
            // Runtime-controlled vent bias keeps legacy default behavior while
            // allowing case-specific stability tuning.
            return fmax(inside, 0.0) + free_surface_vent_bias;
        }
        return inside;
    }
    if (mode == inflow_mode) {
        if (face == static_cast<int32_t>(SWE3DPatchBoundaryFace::XMIN) ||
            face == static_cast<int32_t>(SWE3DPatchBoundaryFace::XMAX)) {
            return desc.bc_u[face];
        }
        if (face == static_cast<int32_t>(SWE3DPatchBoundaryFace::YMIN) ||
            face == static_cast<int32_t>(SWE3DPatchBoundaryFace::YMAX)) {
            return desc.bc_v[face];
        }
        return desc.bc_w[face];
    }
    if (mode == inflow_q_mode) {
        const double q = desc.bc_q[face];
        double area = swe3d_patch_face_total_area(desc, face);
        if (q_inflow_area_policy != 0 && d_face_effective_area) {
            const double area_eff = d_face_effective_area[face];
            if (isfinite(area_eff) && area_eff > 1.0e-9) {
                area = area_eff;
            }
        }
        if (!isfinite(q) || area <= 1.0e-9) {
            return 0.0;
        }
        const double vn = fabs(q) / area;
        if (face == static_cast<int32_t>(SWE3DPatchBoundaryFace::XMIN) ||
            face == static_cast<int32_t>(SWE3DPatchBoundaryFace::YMIN) ||
            face == static_cast<int32_t>(SWE3DPatchBoundaryFace::ZMIN)) {
            return vn;
        }
        return -vn;
    }

    return inside;
}

__device__ __forceinline__ double swe3d_boundary_face_vof(
    SWE3DCartesianPatchDesc desc,
    int32_t face,
    double inside_vof)
{
    const int32_t mode = desc.bc_mode[face];
    if (mode == static_cast<int32_t>(SWE3DBoundaryMode::INFLOW) ||
        mode == static_cast<int32_t>(SWE3DBoundaryMode::INFLOW_FLOW_RATE)) {
        return swe3d_clamp01(desc.bc_vof[face]);
    }
    return swe3d_clamp01(inside_vof);
}

__global__ void swe3d_vof_transport_upwind_kernel(
    SWE3DCartesianPatchDesc desc,
    int64_t n_cells,
    int32_t high_order_enable,
    const double* d_face_effective_area,
    int32_t q_inflow_area_policy,
    double free_surface_vent_bias,
    double dt,
    const double* d_u,
    const double* d_v,
    const double* d_w,
    const double* d_phi,
    const double* d_ax,
    const double* d_ay,
    const double* d_az,
    const uint8_t* d_active_mask,
    double alpha_wet,
    double alpha_gas,
    const double* d_vof,
    double* d_vof_out)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
                      static_cast<int64_t>(threadIdx.x);
    if (i >= n_cells) return;

    const int32_t nx = desc.nx;
    const int32_t ny = desc.ny;
    const int32_t nz = desc.nz;
    const int32_t plane = nx * ny;
    const int32_t iz = static_cast<int32_t>(i / plane);
    const int32_t rem = static_cast<int32_t>(i - static_cast<int64_t>(iz) * plane);
    const int32_t iy = rem / nx;
    const int32_t ix = rem - iy * nx;

    const double dx = fmax(desc.dx, 1.0e-9);
    const double dy = fmax(desc.dy, 1.0e-9);
    const double dz = fmax(desc.dz, 1.0e-9);

    const double phi = d_phi ? swe3d_clamp01(d_phi[i]) : 1.0;
    if (phi <= 1.0e-9) {
        d_vof_out[i] = 0.0;
        return;
    }

    const bool active_i = !d_active_mask || d_active_mask[i] != 0;
    if (!active_i) {
        d_vof_out[i] = fmin(phi, fmax(0.0, d_vof[i]));
        return;
    }
    const double phi_eff = fmax(phi, 1.0e-6);

    const double ax_i = d_ax ? swe3d_clamp01(d_ax[i]) : 1.0;
    const double ay_i = d_ay ? swe3d_clamp01(d_ay[i]) : 1.0;
    const double az_i = d_az ? swe3d_clamp01(d_az[i]) : 1.0;

    const double u_i = d_u[i];
    const double v_i = d_v[i];
    const double w_i = d_w[i];
    const double vof_i = fmin(phi, fmax(0.0, d_vof[i]));
    const bool interface_i = swe3d_interface_band(vof_i, alpha_gas, alpha_wet);

    auto limited_reconstruct = [&](int64_t up_idx, int64_t minus_idx, int64_t plus_idx, double bias) -> double {
        const double up_phi = d_phi ? swe3d_clamp01(d_phi[up_idx]) : 1.0;
        const double q = swe3d_vof_cell_clamped(up_idx, d_phi, d_vof);
        const double qm = (minus_idx >= 0) ? swe3d_vof_cell_clamped(minus_idx, d_phi, d_vof) : q;
        const double qp = (plus_idx >= 0) ? swe3d_vof_cell_clamped(plus_idx, d_phi, d_vof) : q;
        const double slope = swe3d_minmod(q - qm, qp - q);
        return fmin(up_phi, fmax(0.0, q + bias * slope));
    };

    double fxm = 0.0;
    double fxp = 0.0;
    double fym = 0.0;
    double fyp = 0.0;
    double fzm = 0.0;
    double fzp = 0.0;

    // X- face
    if (ix > 0) {
        const int64_t im = swe3d_flat_idx(ix - 1, iy, iz, nx, ny);
        const double u_face = 0.5 * (u_i + d_u[im]);
        const double vof_m = fmin(d_phi ? swe3d_clamp01(d_phi[im]) : 1.0, fmax(0.0, d_vof[im]));
        const bool interface_m = swe3d_interface_band(vof_m, alpha_gas, alpha_wet);
        const bool high_order_face = (high_order_enable != 0) && (interface_i || interface_m);
        double vof_up = (u_face >= 0.0) ? vof_m : vof_i;
        if (high_order_face) {
            if (u_face >= 0.0) {
                const int64_t imm = (ix > 1) ? swe3d_flat_idx(ix - 2, iy, iz, nx, ny) : -1;
                vof_up = limited_reconstruct(im, imm, i, 0.5);
            } else {
                const int64_t ip = (ix + 1 < nx) ? swe3d_flat_idx(ix + 1, iy, iz, nx, ny) : -1;
                vof_up = limited_reconstruct(i, im, ip, -0.5);
            }
        }
        const double ax_face = fmin(ax_i, d_ax ? swe3d_clamp01(d_ax[im]) : 1.0);
        fxm = u_face * vof_up * ax_face;
    } else {
        const int32_t face = static_cast<int32_t>(SWE3DPatchBoundaryFace::XMIN);
        double u_face = swe3d_boundary_face_velocity_component(
            desc,
            face,
            u_i,
            v_i,
            w_i,
            d_face_effective_area,
            q_inflow_area_policy,
            free_surface_vent_bias);
        const double vof_bc = swe3d_boundary_face_vof(desc, face, vof_i);
        const double vof_up = (u_face >= 0.0) ? vof_bc : vof_i;
        fxm = u_face * vof_up * ax_i;
    }

    // X+ face
    if (ix + 1 < nx) {
        const int64_t ip = swe3d_flat_idx(ix + 1, iy, iz, nx, ny);
        const double u_face = 0.5 * (u_i + d_u[ip]);
        const double vof_p = fmin(d_phi ? swe3d_clamp01(d_phi[ip]) : 1.0, fmax(0.0, d_vof[ip]));
        const bool interface_p = swe3d_interface_band(vof_p, alpha_gas, alpha_wet);
        const bool high_order_face = (high_order_enable != 0) && (interface_i || interface_p);
        double vof_up = (u_face >= 0.0) ? vof_i : vof_p;
        if (high_order_face) {
            if (u_face >= 0.0) {
                const int64_t im = (ix > 0) ? swe3d_flat_idx(ix - 1, iy, iz, nx, ny) : -1;
                vof_up = limited_reconstruct(i, im, ip, 0.5);
            } else {
                const int64_t ipp = (ix + 2 < nx) ? swe3d_flat_idx(ix + 2, iy, iz, nx, ny) : -1;
                vof_up = limited_reconstruct(ip, i, ipp, -0.5);
            }
        }
        const double ax_face = fmin(ax_i, d_ax ? swe3d_clamp01(d_ax[ip]) : 1.0);
        fxp = u_face * vof_up * ax_face;
    } else {
        const int32_t face = static_cast<int32_t>(SWE3DPatchBoundaryFace::XMAX);
        double u_face = swe3d_boundary_face_velocity_component(
            desc,
            face,
            u_i,
            v_i,
            w_i,
            d_face_effective_area,
            q_inflow_area_policy,
            free_surface_vent_bias);
        const double vof_bc = swe3d_boundary_face_vof(desc, face, vof_i);
        const double vof_up = (u_face >= 0.0) ? vof_i : vof_bc;
        fxp = u_face * vof_up * ax_i;
    }

    // Y- face
    if (iy > 0) {
        const int64_t jm = swe3d_flat_idx(ix, iy - 1, iz, nx, ny);
        const double v_face = 0.5 * (v_i + d_v[jm]);
        const double vof_m = fmin(d_phi ? swe3d_clamp01(d_phi[jm]) : 1.0, fmax(0.0, d_vof[jm]));
        const bool interface_m = swe3d_interface_band(vof_m, alpha_gas, alpha_wet);
        const bool high_order_face = (high_order_enable != 0) && (interface_i || interface_m);
        double vof_up = (v_face >= 0.0) ? vof_m : vof_i;
        if (high_order_face) {
            if (v_face >= 0.0) {
                const int64_t jmm = (iy > 1) ? swe3d_flat_idx(ix, iy - 2, iz, nx, ny) : -1;
                vof_up = limited_reconstruct(jm, jmm, i, 0.5);
            } else {
                const int64_t jp = (iy + 1 < ny) ? swe3d_flat_idx(ix, iy + 1, iz, nx, ny) : -1;
                vof_up = limited_reconstruct(i, jm, jp, -0.5);
            }
        }
        const double ay_face = fmin(ay_i, d_ay ? swe3d_clamp01(d_ay[jm]) : 1.0);
        fym = v_face * vof_up * ay_face;
    } else {
        const int32_t face = static_cast<int32_t>(SWE3DPatchBoundaryFace::YMIN);
        double v_face = swe3d_boundary_face_velocity_component(
            desc,
            face,
            u_i,
            v_i,
            w_i,
            d_face_effective_area,
            q_inflow_area_policy,
            free_surface_vent_bias);
        const double vof_bc = swe3d_boundary_face_vof(desc, face, vof_i);
        const double vof_up = (v_face >= 0.0) ? vof_bc : vof_i;
        fym = v_face * vof_up * ay_i;
    }

    // Y+ face
    if (iy + 1 < ny) {
        const int64_t jp = swe3d_flat_idx(ix, iy + 1, iz, nx, ny);
        const double v_face = 0.5 * (v_i + d_v[jp]);
        const double vof_p = fmin(d_phi ? swe3d_clamp01(d_phi[jp]) : 1.0, fmax(0.0, d_vof[jp]));
        const bool interface_p = swe3d_interface_band(vof_p, alpha_gas, alpha_wet);
        const bool high_order_face = (high_order_enable != 0) && (interface_i || interface_p);
        double vof_up = (v_face >= 0.0) ? vof_i : vof_p;
        if (high_order_face) {
            if (v_face >= 0.0) {
                const int64_t jm = (iy > 0) ? swe3d_flat_idx(ix, iy - 1, iz, nx, ny) : -1;
                vof_up = limited_reconstruct(i, jm, jp, 0.5);
            } else {
                const int64_t jpp = (iy + 2 < ny) ? swe3d_flat_idx(ix, iy + 2, iz, nx, ny) : -1;
                vof_up = limited_reconstruct(jp, i, jpp, -0.5);
            }
        }
        const double ay_face = fmin(ay_i, d_ay ? swe3d_clamp01(d_ay[jp]) : 1.0);
        fyp = v_face * vof_up * ay_face;
    } else {
        const int32_t face = static_cast<int32_t>(SWE3DPatchBoundaryFace::YMAX);
        double v_face = swe3d_boundary_face_velocity_component(
            desc,
            face,
            u_i,
            v_i,
            w_i,
            d_face_effective_area,
            q_inflow_area_policy,
            free_surface_vent_bias);
        const double vof_bc = swe3d_boundary_face_vof(desc, face, vof_i);
        const double vof_up = (v_face >= 0.0) ? vof_i : vof_bc;
        fyp = v_face * vof_up * ay_i;
    }

    // Z- face
    if (iz > 0) {
        const int64_t km = swe3d_flat_idx(ix, iy, iz - 1, nx, ny);
        const double w_face = 0.5 * (w_i + d_w[km]);
        const double vof_m = fmin(d_phi ? swe3d_clamp01(d_phi[km]) : 1.0, fmax(0.0, d_vof[km]));
        const bool interface_m = swe3d_interface_band(vof_m, alpha_gas, alpha_wet);
        const bool high_order_face = (high_order_enable != 0) && (interface_i || interface_m);
        double vof_up = (w_face >= 0.0) ? vof_m : vof_i;
        if (high_order_face) {
            if (w_face >= 0.0) {
                const int64_t kmm = (iz > 1) ? swe3d_flat_idx(ix, iy, iz - 2, nx, ny) : -1;
                vof_up = limited_reconstruct(km, kmm, i, 0.5);
            } else {
                const int64_t kp = (iz + 1 < nz) ? swe3d_flat_idx(ix, iy, iz + 1, nx, ny) : -1;
                vof_up = limited_reconstruct(i, km, kp, -0.5);
            }
        }
        const double az_face = fmin(az_i, d_az ? swe3d_clamp01(d_az[km]) : 1.0);
        fzm = w_face * vof_up * az_face;
    } else {
        const int32_t face = static_cast<int32_t>(SWE3DPatchBoundaryFace::ZMIN);
        double w_face = swe3d_boundary_face_velocity_component(
            desc,
            face,
            u_i,
            v_i,
            w_i,
            d_face_effective_area,
            q_inflow_area_policy,
            free_surface_vent_bias);
        const double vof_bc = swe3d_boundary_face_vof(desc, face, vof_i);
        const double vof_up = (w_face >= 0.0) ? vof_bc : vof_i;
        fzm = w_face * vof_up * az_i;
    }

    // Z+ face
    if (iz + 1 < nz) {
        const int64_t kp = swe3d_flat_idx(ix, iy, iz + 1, nx, ny);
        const double w_face = 0.5 * (w_i + d_w[kp]);
        const double vof_p = fmin(d_phi ? swe3d_clamp01(d_phi[kp]) : 1.0, fmax(0.0, d_vof[kp]));
        const bool interface_p = swe3d_interface_band(vof_p, alpha_gas, alpha_wet);
        const bool high_order_face = (high_order_enable != 0) && (interface_i || interface_p);
        double vof_up = (w_face >= 0.0) ? vof_i : vof_p;
        if (high_order_face) {
            if (w_face >= 0.0) {
                const int64_t km = (iz > 0) ? swe3d_flat_idx(ix, iy, iz - 1, nx, ny) : -1;
                vof_up = limited_reconstruct(i, km, kp, 0.5);
            } else {
                const int64_t kpp = (iz + 2 < nz) ? swe3d_flat_idx(ix, iy, iz + 2, nx, ny) : -1;
                vof_up = limited_reconstruct(kp, i, kpp, -0.5);
            }
        }
        const double az_face = fmin(az_i, d_az ? swe3d_clamp01(d_az[kp]) : 1.0);
        fzp = w_face * vof_up * az_face;
    } else {
        const int32_t face = static_cast<int32_t>(SWE3DPatchBoundaryFace::ZMAX);
        double w_face = swe3d_boundary_face_velocity_component(
            desc,
            face,
            u_i,
            v_i,
            w_i,
            d_face_effective_area,
            q_inflow_area_policy,
            free_surface_vent_bias);
        const double vof_bc = swe3d_boundary_face_vof(desc, face, vof_i);
        const double vof_up = (w_face >= 0.0) ? vof_i : vof_bc;
        fzp = w_face * vof_up * az_i;
    }

    const double rhs = -((fxp - fxm) / dx + (fyp - fym) / dy + (fzp - fzm) / dz) / phi_eff;
    const double vof_new = fmin(phi, fmax(0.0, vof_i + dt * rhs));
    d_vof_out[i] = vof_new;
}

__global__ void swe3d_single_phase_predictor_kernel(
    int64_t n_cells,
    double* d_u,
    double* d_v,
    double* d_w,
    const double* d_phi,
    const double* d_ax,
    const double* d_ay,
    const double* d_az,
    const uint8_t* d_active_mask,
    double* d_vof,
    double dt,
    double predictor_damping_coeff)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
                      static_cast<int64_t>(threadIdx.x);
    if (i >= n_cells) return;

    const double phi = d_phi ? swe3d_clamp01(d_phi[i]) : 1.0;
    const double ax = d_ax ? swe3d_clamp01(d_ax[i]) : 1.0;
    const double ay = d_ay ? swe3d_clamp01(d_ay[i]) : 1.0;
    const double az = d_az ? swe3d_clamp01(d_az[i]) : 1.0;

    // Keep VoF bounded and clipped by local fluid fraction.
    const double vof = d_vof[i];
    d_vof[i] = fmin(phi, fmax(0.0, vof));

    if (phi <= 1.0e-9) {
        d_u[i] = 0.0;
        d_v[i] = 0.0;
        d_w[i] = 0.0;
        d_vof[i] = 0.0;
        return;
    }

    if (d_active_mask && d_active_mask[i] == 0) {
        d_u[i] = 0.0;
        d_v[i] = 0.0;
        d_w[i] = 0.0;
        return;
    }

    // Minimal diffusion-like predictor damping for robustness while projection matures.
    const double damp = 1.0 / (1.0 + predictor_damping_coeff * dt);
    d_u[i] *= damp * phi * ax;
    d_v[i] *= damp * phi * ay;
    d_w[i] *= damp * phi * az;
}

__global__ void swe3d_compute_pressure_rhs_kernel(
    int32_t nx,
    int32_t ny,
    int32_t nz,
    double dx,
    double dy,
    double dz,
    const double* d_u,
    const double* d_v,
    const double* d_w,
    const double* d_vof,
    const uint8_t* d_active_mask,
    double dt,
    double* d_rhs)
{
    const int64_t n_cells = static_cast<int64_t>(nx) * static_cast<int64_t>(ny) * static_cast<int64_t>(nz);
    const int64_t i = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
                      static_cast<int64_t>(threadIdx.x);
    if (i >= n_cells) return;

    const int32_t plane = nx * ny;
    const int32_t iz = static_cast<int32_t>(i / plane);
    const int32_t rem = static_cast<int32_t>(i - static_cast<int64_t>(iz) * plane);
    const int32_t iy = rem / nx;
    const int32_t ix = rem - iy * nx;

    if (d_active_mask && d_active_mask[i] == 0) {
        d_rhs[i] = 0.0;
        return;
    }

    const int32_t ixm = (ix > 0) ? (ix - 1) : ix;
    const int32_t ixp = (ix + 1 < nx) ? (ix + 1) : ix;
    const int32_t iym = (iy > 0) ? (iy - 1) : iy;
    const int32_t iyp = (iy + 1 < ny) ? (iy + 1) : iy;
    const int32_t izm = (iz > 0) ? (iz - 1) : iz;
    const int32_t izp = (iz + 1 < nz) ? (iz + 1) : iz;

    const int64_t id_xm = swe3d_flat_idx(ixm, iy, iz, nx, ny);
    const int64_t id_xp = swe3d_flat_idx(ixp, iy, iz, nx, ny);
    const int64_t id_ym = swe3d_flat_idx(ix, iym, iz, nx, ny);
    const int64_t id_yp = swe3d_flat_idx(ix, iyp, iz, nx, ny);
    const int64_t id_zm = swe3d_flat_idx(ix, iy, izm, nx, ny);
    const int64_t id_zp = swe3d_flat_idx(ix, iy, izp, nx, ny);

    const double wxp = (ix + 1 < nx) ? swe3d_face_activity_weight(d_vof, d_active_mask, i, id_xp) : 0.0;
    const double wxm = (ix > 0) ? swe3d_face_activity_weight(d_vof, d_active_mask, id_xm, i) : 0.0;
    const double wyp = (iy + 1 < ny) ? swe3d_face_activity_weight(d_vof, d_active_mask, i, id_yp) : 0.0;
    const double wym = (iy > 0) ? swe3d_face_activity_weight(d_vof, d_active_mask, id_ym, i) : 0.0;
    const double wzp = (iz + 1 < nz) ? swe3d_face_activity_weight(d_vof, d_active_mask, i, id_zp) : 0.0;
    const double wzm = (iz > 0) ? swe3d_face_activity_weight(d_vof, d_active_mask, id_zm, i) : 0.0;

    const double ux_p = (ix + 1 < nx) ? (0.5 * (d_u[i] + d_u[id_xp])) : d_u[i];
    const double ux_m = (ix > 0) ? (0.5 * (d_u[id_xm] + d_u[i])) : d_u[i];
    const double vy_p = (iy + 1 < ny) ? (0.5 * (d_v[i] + d_v[id_yp])) : d_v[i];
    const double vy_m = (iy > 0) ? (0.5 * (d_v[id_ym] + d_v[i])) : d_v[i];
    const double wz_p = (iz + 1 < nz) ? (0.5 * (d_w[i] + d_w[id_zp])) : d_w[i];
    const double wz_m = (iz > 0) ? (0.5 * (d_w[id_zm] + d_w[i])) : d_w[i];

    const double div_u =
        ((wxp * ux_p) - (wxm * ux_m)) / fmax(dx, 1.0e-9) +
        ((wyp * vy_p) - (wym * vy_m)) / fmax(dy, 1.0e-9) +
        ((wzp * wz_p) - (wzm * wz_m)) / fmax(dz, 1.0e-9);
    d_rhs[i] = div_u / dt;
}

__global__ void swe3d_pressure_jacobi_kernel(
    SWE3DCartesianPatchDesc desc,
    int32_t nx,
    int32_t ny,
    int32_t nz,
    double dx,
    double dy,
    double dz,
    double free_surface_gauge_tolerance_pa,
    const double* d_p,
    const double* d_vof,
    const uint8_t* d_active_mask,
    const double* d_rhs,
    double* d_p_out)
{
    const int64_t n_cells = static_cast<int64_t>(nx) * static_cast<int64_t>(ny) * static_cast<int64_t>(nz);
    const int64_t i = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
                      static_cast<int64_t>(threadIdx.x);
    if (i >= n_cells) return;

    const int32_t plane = nx * ny;
    const int32_t iz = static_cast<int32_t>(i / plane);
    const int32_t rem = static_cast<int32_t>(i - static_cast<int64_t>(iz) * plane);
    const int32_t iy = rem / nx;
    const int32_t ix = rem - iy * nx;

    const bool active_i = !d_active_mask || d_active_mask[i] != 0;
    if (!active_i) {
        d_p_out[i] = 0.0;
        return;
    }

    const int32_t ixm = (ix > 0) ? (ix - 1) : ix;
    const int32_t ixp = (ix + 1 < nx) ? (ix + 1) : ix;
    const int32_t iym = (iy > 0) ? (iy - 1) : iy;
    const int32_t iyp = (iy + 1 < ny) ? (iy + 1) : iy;
    const int32_t izm = (iz > 0) ? (iz - 1) : iz;
    const int32_t izp = (iz + 1 < nz) ? (iz + 1) : iz;

    const int64_t id_xm = swe3d_flat_idx(ixm, iy, iz, nx, ny);
    const int64_t id_xp = swe3d_flat_idx(ixp, iy, iz, nx, ny);
    const int64_t id_ym = swe3d_flat_idx(ix, iym, iz, nx, ny);
    const int64_t id_yp = swe3d_flat_idx(ix, iyp, iz, nx, ny);
    const int64_t id_zm = swe3d_flat_idx(ix, iy, izm, nx, ny);
    const int64_t id_zp = swe3d_flat_idx(ix, iy, izp, nx, ny);

    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);
    const double inv_dz2 = 1.0 / (dz * dz);

    const double wxm = (ix > 0) ? swe3d_face_activity_weight(d_vof, d_active_mask, id_xm, i) : 0.0;
    const double wxp = (ix + 1 < nx) ? swe3d_face_activity_weight(d_vof, d_active_mask, i, id_xp) : 0.0;
    const double wym = (iy > 0) ? swe3d_face_activity_weight(d_vof, d_active_mask, id_ym, i) : 0.0;
    const double wyp = (iy + 1 < ny) ? swe3d_face_activity_weight(d_vof, d_active_mask, i, id_yp) : 0.0;
    const double wzm = (iz > 0) ? swe3d_face_activity_weight(d_vof, d_active_mask, id_zm, i) : 0.0;
    const double wzp = (iz + 1 < nz) ? swe3d_face_activity_weight(d_vof, d_active_mask, i, id_zp) : 0.0;

    const double axm = wxm * inv_dx2;
    const double axp = wxp * inv_dx2;
    const double aym = wym * inv_dy2;
    const double ayp = wyp * inv_dy2;
    const double azm = wzm * inv_dz2;
    const double azp = wzp * inv_dz2;

    const double denom = axm + axp + aym + ayp + azm + azp;
    const double nbr_sum =
        axm * d_p[id_xm] +
        axp * d_p[id_xp] +
        aym * d_p[id_ym] +
        ayp * d_p[id_yp] +
        azm * d_p[id_zm] +
        azp * d_p[id_zp];

    const double p_jac = (denom > 1.0e-12) ? ((nbr_sum - d_rhs[i]) / denom) : d_p[i];

    const int32_t free_surface_mode = static_cast<int32_t>(SWE3DBoundaryMode::FREE_SURFACE);
    const int32_t zmax_face = static_cast<int32_t>(SWE3DPatchBoundaryFace::ZMAX);
    const bool zmax_is_free_surface =
        (desc.bc_mode[zmax_face] == free_surface_mode) && (iz + 1 == nz);
    if (zmax_is_free_surface) {
        const double p_target = desc.bc_p[zmax_face];
        const double p_tol = fmax(0.0, free_surface_gauge_tolerance_pa);
        if (p_tol <= 0.0) {
            d_p_out[i] = p_target;
            return;
        }
        d_p_out[i] = fmin(p_target + p_tol, fmax(p_target - p_tol, p_jac));
        return;
    }

    d_p_out[i] = p_jac;
}

__global__ void swe3d_projection_residual_max_kernel(
    int64_t n_cells,
    const double* d_p_old,
    const double* d_p_new,
    unsigned long long* d_residual_bits)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
                      static_cast<int64_t>(threadIdx.x);
    __shared__ double smax[256];
    double local_max = 0.0;
    if (i < n_cells) {
        const double p_old = d_p_old[i];
        const double p_new = d_p_new[i];
        const double scale = fmax(1.0, fmax(fabs(p_old), fabs(p_new)));
        local_max = fabs(p_new - p_old) / scale;
    }
    smax[threadIdx.x] = local_max;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            smax[threadIdx.x] = fmax(smax[threadIdx.x], smax[threadIdx.x + offset]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicMax(d_residual_bits, __double_as_longlong(smax[0]));
    }
}

__global__ void swe3d_velocity_absmax_kernel(
    int64_t n_cells,
    const double* d_u,
    const double* d_v,
    const double* d_w,
    unsigned long long* d_max_bits)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
                      static_cast<int64_t>(threadIdx.x);
    __shared__ double smax[256];
    double local_max = 0.0;
    if (i < n_cells) {
        local_max = fmax(fabs(d_u[i]), fmax(fabs(d_v[i]), fabs(d_w[i])));
    }
    smax[threadIdx.x] = local_max;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            smax[threadIdx.x] = fmax(smax[threadIdx.x], smax[threadIdx.x + offset]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicMax(d_max_bits, __double_as_longlong(smax[0]));
    }
}

__global__ void swe3d_vof_min_kernel(
    int64_t n_cells,
    const double* d_phi,
    const double* d_vof,
    unsigned long long* d_min_bits)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
                      static_cast<int64_t>(threadIdx.x);
    __shared__ double smin[256];
    double local_min = 1.0;
    if (i < n_cells) {
        local_min = swe3d_vof_cell_clamped(i, d_phi, d_vof);
    }
    smin[threadIdx.x] = local_min;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            smin[threadIdx.x] = fmin(smin[threadIdx.x], smin[threadIdx.x + offset]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicMin(d_min_bits, __double_as_longlong(smin[0]));
    }
}

__global__ void swe3d_vof_sum_kernel(
    int64_t n_cells,
    const double* d_phi,
    const double* d_vof,
    double* d_sum)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
                      static_cast<int64_t>(threadIdx.x);
    __shared__ double ssum[256];
    double local_sum = 0.0;
    if (i < n_cells) {
        local_sum = swe3d_vof_cell_clamped(i, d_phi, d_vof);
    }
    ssum[threadIdx.x] = local_sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            ssum[threadIdx.x] += ssum[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(d_sum, ssum[0]);
    }
}

__global__ void swe3d_sum_sq_kernel(
    int64_t n_cells,
    const double* d_data,
    double* d_sum_sq)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
                      static_cast<int64_t>(threadIdx.x);
    __shared__ double ssum[256];
    double local_sum = 0.0;
    if (i < n_cells) {
        const double v = d_data[i];
        if (isfinite(v)) {
            local_sum = v * v;
        }
    }
    ssum[threadIdx.x] = local_sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            ssum[threadIdx.x] += ssum[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(d_sum_sq, ssum[0]);
    }
}

__global__ void swe3d_velocity_soft_cap_kernel(
    int64_t n_cells,
    double max_abs_velocity,
    double* d_u,
    double* d_v,
    double* d_w)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
                      static_cast<int64_t>(threadIdx.x);
    if (i >= n_cells || !(max_abs_velocity > 0.0) || !isfinite(max_abs_velocity)) return;

    const double u = d_u[i];
    const double v = d_v[i];
    const double w = d_w[i];
    if (!isfinite(u) || !isfinite(v) || !isfinite(w)) return;

    const double peak = fmax(fabs(u), fmax(fabs(v), fabs(w)));
    if (!(peak > max_abs_velocity)) return;

    const double scale = max_abs_velocity / peak;
    d_u[i] = u * scale;
    d_v[i] = v * scale;
    d_w[i] = w * scale;
}

__global__ void swe3d_state_health_kernel(
    int64_t n_cells,
    const double* d_u,
    const double* d_v,
    const double* d_w,
    const double* d_p,
    const double* d_vof,
    double vof_bounds_tol,
    unsigned int* d_flags,
    unsigned long long* d_umax_bits,
    unsigned long long* d_pabs_bits)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
                      static_cast<int64_t>(threadIdx.x);
    if (i >= n_cells) return;

    const double u = d_u[i];
    const double v = d_v[i];
    const double w = d_w[i];
    const double p = d_p[i];
    const double vof = d_vof[i];

    const bool finite_state =
        isfinite(u) && isfinite(v) && isfinite(w) && isfinite(p) && isfinite(vof);
    if (!finite_state) {
        atomicOr(d_flags, 1u);
    }

    if (isfinite(vof)) {
        if (vof < -vof_bounds_tol || vof > 1.0 + vof_bounds_tol) {
            atomicOr(d_flags, 2u);
        }
    }

    if (isfinite(u) && isfinite(v) && isfinite(w)) {
        const double umax = fmax(fabs(u), fmax(fabs(v), fabs(w)));
        atomicMax(d_umax_bits, __double_as_longlong(umax));
    }

    if (isfinite(p)) {
        const double pabs = fabs(p);
        atomicMax(d_pabs_bits, __double_as_longlong(pabs));
    }
}

__global__ void swe3d_dt_wave_max_kernel(
    int64_t n_cells,
    const double* d_u,
    const double* d_v,
    const double* d_w,
    const double* d_vof,
    const double* d_phi,
    const double* d_ax,
    const double* d_ay,
    const double* d_az,
    double g,
    double dx,
    double dy,
    double dz,
    double gravity_wave_weight,
    int32_t adaptive_mode,
    unsigned long long* d_max_bits)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
                      static_cast<int64_t>(threadIdx.x);
    if (i >= n_cells) return;

    const double u = fabs(d_u[i]);
    const double v = fabs(d_v[i]);
    const double w = fabs(d_w[i]);

    const double phi_raw = d_phi ? d_phi[i] : 1.0;
    const double phi = fmax(0.05, fmin(1.0, phi_raw));
    const double ax = d_ax ? fmax(0.0, fmin(1.0, d_ax[i])) : 1.0;
    const double ay = d_ay ? fmax(0.0, fmin(1.0, d_ay[i])) : 1.0;
    const double az = d_az ? fmax(0.0, fmin(1.0, d_az[i])) : 1.0;

    const double lam_x = u * ax / fmax(dx * phi, 1.0e-9);
    const double lam_y = v * ay / fmax(dy * phi, 1.0e-9);
    const double lam_z = w * az / fmax(dz * phi, 1.0e-9);
    const double adv = lam_x + lam_y + lam_z;

    double grav = 0.0;
    if (adaptive_mode >= static_cast<int32_t>(SWE3DAdaptiveDtMode::ADVECTIVE_PLUS_GRAVITY_WAVE) &&
        g > 0.0 && dx > 0.0 && dy > 0.0 && dz > 0.0) {
        const double vof = d_vof ? d_vof[i] : 1.0;
        const double depth_eff = fmax(0.0, fmin(1.0, vof)) * phi * dz;
        const double wave_scale = fmax(fmin(dx, dy), 1.0e-9);
        grav = (sqrt(g * depth_eff) / wave_scale) * gravity_wave_weight;
    }

    const double c = adv + grav;
    atomicMax(d_max_bits, __double_as_longlong(c));
}

__global__ void swe3d_apply_wall_noslip_kernel(
    SWE3DCartesianPatchDesc desc,
    int64_t n_cells,
    double* d_u,
    double* d_v,
    double* d_w)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
                      static_cast<int64_t>(threadIdx.x);
    if (i >= n_cells) return;

    const int32_t nx = desc.nx;
    const int32_t ny = desc.ny;
    const int32_t nz = desc.nz;
    const int32_t plane = nx * ny;
    const int32_t iz = static_cast<int32_t>(i / plane);
    const int32_t rem = static_cast<int32_t>(i - static_cast<int64_t>(iz) * plane);
    const int32_t iy = rem / nx;
    const int32_t ix = rem - iy * nx;

    const int32_t wall_mode = static_cast<int32_t>(SWE3DBoundaryMode::WALL);
    bool no_slip = false;
    if (ix == 0 && desc.bc_mode[static_cast<int32_t>(SWE3DPatchBoundaryFace::XMIN)] == wall_mode) no_slip = true;
    if (ix + 1 == nx && desc.bc_mode[static_cast<int32_t>(SWE3DPatchBoundaryFace::XMAX)] == wall_mode) no_slip = true;
    if (iy == 0 && desc.bc_mode[static_cast<int32_t>(SWE3DPatchBoundaryFace::YMIN)] == wall_mode) no_slip = true;
    if (iy + 1 == ny && desc.bc_mode[static_cast<int32_t>(SWE3DPatchBoundaryFace::YMAX)] == wall_mode) no_slip = true;
    if (iz == 0 && desc.bc_mode[static_cast<int32_t>(SWE3DPatchBoundaryFace::ZMIN)] == wall_mode) no_slip = true;
    if (iz + 1 == nz && desc.bc_mode[static_cast<int32_t>(SWE3DPatchBoundaryFace::ZMAX)] == wall_mode) no_slip = true;

    if (no_slip) {
        d_u[i] = 0.0;
        d_v[i] = 0.0;
        d_w[i] = 0.0;
    }
}

__global__ void swe3d_apply_face_inflow_bc_kernel(
    SWE3DCartesianPatchDesc desc,
    int64_t n_cells,
    const double* d_face_effective_area,
    int32_t q_inflow_area_policy,
    double free_surface_vent_bias,
    const double* d_phi,
    const double* d_vof,
    double* d_u,
    double* d_v,
    double* d_w)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
                      static_cast<int64_t>(threadIdx.x);
    if (i >= n_cells) return;

    const int32_t nx = desc.nx;
    const int32_t ny = desc.ny;
    const int32_t nz = desc.nz;
    const int32_t plane = nx * ny;
    const int32_t iz = static_cast<int32_t>(i / plane);
    const int32_t rem = static_cast<int32_t>(i - static_cast<int64_t>(iz) * plane);
    const int32_t iy = rem / nx;
    const int32_t ix = rem - iy * nx;

    const int32_t wall_mode = static_cast<int32_t>(SWE3DBoundaryMode::WALL);
    const int32_t inflow_mode = static_cast<int32_t>(SWE3DBoundaryMode::INFLOW);
    const int32_t inflow_q_mode = static_cast<int32_t>(SWE3DBoundaryMode::INFLOW_FLOW_RATE);

    const double phi_i = d_phi ? swe3d_clamp01(d_phi[i]) : 1.0;
    const double vof_i = swe3d_vof_cell_clamped(i, d_phi, d_vof);

    // Keep no-slip dominant at edges/corners that touch any wall boundary.
    if (ix == 0 && desc.bc_mode[static_cast<int32_t>(SWE3DPatchBoundaryFace::XMIN)] == wall_mode) return;
    if (ix + 1 == nx && desc.bc_mode[static_cast<int32_t>(SWE3DPatchBoundaryFace::XMAX)] == wall_mode) return;
    if (iy == 0 && desc.bc_mode[static_cast<int32_t>(SWE3DPatchBoundaryFace::YMIN)] == wall_mode) return;
    if (iy + 1 == ny && desc.bc_mode[static_cast<int32_t>(SWE3DPatchBoundaryFace::YMAX)] == wall_mode) return;
    if (iz == 0 && desc.bc_mode[static_cast<int32_t>(SWE3DPatchBoundaryFace::ZMIN)] == wall_mode) return;
    if (iz + 1 == nz && desc.bc_mode[static_cast<int32_t>(SWE3DPatchBoundaryFace::ZMAX)] == wall_mode) return;

    auto apply_face = [&](int32_t face) {
        const int32_t mode = desc.bc_mode[face];
        if (mode != inflow_mode && mode != inflow_q_mode) {
            return;
        }

        const double u_cur = d_u[i];
        const double v_cur = d_v[i];
        const double w_cur = d_w[i];
        const double vn = swe3d_boundary_face_velocity_component(
            desc,
            face,
            u_cur,
            v_cur,
            w_cur,
            d_face_effective_area,
            q_inflow_area_policy,
            free_surface_vent_bias);
        const double vn_eff = vn;

        if (face == static_cast<int32_t>(SWE3DPatchBoundaryFace::XMIN) ||
            face == static_cast<int32_t>(SWE3DPatchBoundaryFace::XMAX)) {
            d_u[i] = vn_eff;
            if (mode == inflow_mode) {
                d_v[i] = desc.bc_v[face];
                d_w[i] = desc.bc_w[face];
            }
            return;
        }
        if (face == static_cast<int32_t>(SWE3DPatchBoundaryFace::YMIN) ||
            face == static_cast<int32_t>(SWE3DPatchBoundaryFace::YMAX)) {
            d_v[i] = vn_eff;
            if (mode == inflow_mode) {
                d_u[i] = desc.bc_u[face];
                d_w[i] = desc.bc_w[face];
            }
            return;
        }

        d_w[i] = vn_eff;
        if (mode == inflow_mode) {
            d_u[i] = desc.bc_u[face];
            d_v[i] = desc.bc_v[face];
        }
    };

    if (ix == 0) apply_face(static_cast<int32_t>(SWE3DPatchBoundaryFace::XMIN));
    if (ix + 1 == nx) apply_face(static_cast<int32_t>(SWE3DPatchBoundaryFace::XMAX));
    if (iy == 0) apply_face(static_cast<int32_t>(SWE3DPatchBoundaryFace::YMIN));
    if (iy + 1 == ny) apply_face(static_cast<int32_t>(SWE3DPatchBoundaryFace::YMAX));
    if (iz == 0) apply_face(static_cast<int32_t>(SWE3DPatchBoundaryFace::ZMIN));
    if (iz + 1 == nz) apply_face(static_cast<int32_t>(SWE3DPatchBoundaryFace::ZMAX));
}

__global__ void swe3d_velocity_correction_kernel(
    int32_t nx,
    int32_t ny,
    int32_t nz,
    double dx,
    double dy,
    double dz,
    const double* d_p,
    const double* d_vof,
    const uint8_t* d_active_mask,
    double dt,
    double projection_correction_scale,
    double* d_u,
    double* d_v,
    double* d_w)
{
    const int64_t n_cells = static_cast<int64_t>(nx) * static_cast<int64_t>(ny) * static_cast<int64_t>(nz);
    const int64_t i = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
                      static_cast<int64_t>(threadIdx.x);
    if (i >= n_cells) return;

    const int32_t plane = nx * ny;
    const int32_t iz = static_cast<int32_t>(i / plane);
    const int32_t rem = static_cast<int32_t>(i - static_cast<int64_t>(iz) * plane);
    const int32_t iy = rem / nx;
    const int32_t ix = rem - iy * nx;

    if (d_active_mask && d_active_mask[i] == 0) {
        d_u[i] = 0.0;
        d_v[i] = 0.0;
        d_w[i] = 0.0;
        return;
    }

    const int32_t ixm = (ix > 0) ? (ix - 1) : ix;
    const int32_t ixp = (ix + 1 < nx) ? (ix + 1) : ix;
    const int32_t iym = (iy > 0) ? (iy - 1) : iy;
    const int32_t iyp = (iy + 1 < ny) ? (iy + 1) : iy;
    const int32_t izm = (iz > 0) ? (iz - 1) : iz;
    const int32_t izp = (iz + 1 < nz) ? (iz + 1) : iz;

    const int64_t id_xm = swe3d_flat_idx(ixm, iy, iz, nx, ny);
    const int64_t id_xp = swe3d_flat_idx(ixp, iy, iz, nx, ny);
    const int64_t id_ym = swe3d_flat_idx(ix, iym, iz, nx, ny);
    const int64_t id_yp = swe3d_flat_idx(ix, iyp, iz, nx, ny);
    const int64_t id_zm = swe3d_flat_idx(ix, iy, izm, nx, ny);
    const int64_t id_zp = swe3d_flat_idx(ix, iy, izp, nx, ny);

    const double wxm = (ix > 0) ? swe3d_face_activity_weight(d_vof, d_active_mask, id_xm, i) : 0.0;
    const double wxp = (ix + 1 < nx) ? swe3d_face_activity_weight(d_vof, d_active_mask, i, id_xp) : 0.0;
    const double wym = (iy > 0) ? swe3d_face_activity_weight(d_vof, d_active_mask, id_ym, i) : 0.0;
    const double wyp = (iy + 1 < ny) ? swe3d_face_activity_weight(d_vof, d_active_mask, i, id_yp) : 0.0;
    const double wzm = (iz > 0) ? swe3d_face_activity_weight(d_vof, d_active_mask, id_zm, i) : 0.0;
    const double wzp = (iz + 1 < nz) ? swe3d_face_activity_weight(d_vof, d_active_mask, i, id_zp) : 0.0;

    const double denom_x = fmax(0.5 * (wxm + wxp), 1.0e-6) * dx;
    const double denom_y = fmax(0.5 * (wym + wyp), 1.0e-6) * dy;
    const double denom_z = fmax(0.5 * (wzm + wzp), 1.0e-6) * dz;

    const double dp_dx = ((wxp * (d_p[id_xp] - d_p[i])) - (wxm * (d_p[i] - d_p[id_xm]))) / denom_x;
    const double dp_dy = ((wyp * (d_p[id_yp] - d_p[i])) - (wym * (d_p[i] - d_p[id_ym]))) / denom_y;
    const double dp_dz = ((wzp * (d_p[id_zp] - d_p[i])) - (wzm * (d_p[i] - d_p[id_zm]))) / denom_z;

    const double dt_corr = projection_correction_scale * dt;
    d_u[i] -= dt_corr * dp_dx;
    d_v[i] -= dt_corr * dp_dy;
    d_w[i] -= dt_corr * dp_dz;
}

__global__ void swe3d_velocity_scale_kernel(
    int64_t n_cells,
    double scale,
    double* d_u,
    double* d_v,
    double* d_w)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
                      static_cast<int64_t>(threadIdx.x);
    if (i >= n_cells) return;
    d_u[i] *= scale;
    d_v[i] *= scale;
    d_w[i] *= scale;
}

void swe2d_gpu_step_3d_single_phase_free_surface(
    SWE2DDeviceState* dev,
    double /*t_now*/,
    double dt,
    double g,
    int coupling_mode,
    bool sync_diagnostics,
    SWE2DStepDiag* diag)
{
    if (!dev) throw std::invalid_argument("swe2d_gpu_step_3d_single_phase_free_surface: null device state");
    if (!dev->patch3d) {
        throw std::runtime_error(
            "swe2d_gpu_step_3d_single_phase_free_surface: 3D patch is not allocated");
    }

    auto* patch = dev->patch3d;
    if (patch->n_cells <= 0) {
        throw std::runtime_error(
            "swe2d_gpu_step_3d_single_phase_free_surface: invalid 3D patch size");
    }
    if (!patch->d_u || !patch->d_v || !patch->d_w || !patch->d_vof || !patch->d_vof_tmp ||
        !patch->d_p || !patch->d_p_rhs || !patch->d_p_tmp || !patch->d_proj_residual_bits) {
        throw std::runtime_error(
            "swe2d_gpu_step_3d_single_phase_free_surface: 3D patch device pointers are not allocated; "
            "this may indicate a memory allocation failure during patch initialization");
    }
    if (!(dt > 0.0) || !std::isfinite(dt)) {
        throw std::invalid_argument("swe2d_gpu_step_3d_single_phase_free_surface: non-positive dt");
    }

    const SWE3DRuntimeControls runtime_controls = swe3d_load_runtime_controls_cached();
    const int32_t vof_max_substeps_cap = std::max<int32_t>(1, runtime_controls.vof_max_substeps);
    const bool projection_reject_enabled =
        runtime_controls.projection_reject_enable &&
        runtime_controls.projection_max_retries > 0;
    const bool projection_fail_fast = runtime_controls.projection_fail_fast;
    const bool projection_divergence_gate_enable = runtime_controls.projection_divergence_gate_enable;
    const double projection_divergence_ratio_target = fmax(1.0e-6, runtime_controls.projection_divergence_ratio_target);
    const int32_t projection_max_retries = std::max<int32_t>(0, runtime_controls.projection_max_retries);
    const double projection_target = fmax(runtime_controls.projection_residual_target, 1.0e-12);
    const double projection_scale_min = fmax(0.1, runtime_controls.projection_correction_scale_min);
    const double projection_scale_max = fmax(projection_scale_min, runtime_controls.projection_correction_scale_max);
    const double projection_reduction = fmin(0.99, fmax(0.05, runtime_controls.projection_dt_reduction));
    const double projection_min_dt_factor = fmin(1.0, fmax(1.0e-4, runtime_controls.projection_min_dt_factor));
    const double projection_dt_floor = fmax(dt * projection_min_dt_factor, 1.0e-9);
    const double predictor_damping_coeff = fmax(0.0, runtime_controls.predictor_damping_coeff);
    const double free_surface_gauge_tolerance_pa = fmax(0.0, runtime_controls.free_surface_gauge_tolerance_pa);
    const bool state_reject_enabled = runtime_controls.state_reject_enable;
    const double state_vof_bounds_tol = fmax(0.0, runtime_controls.state_vof_bounds_tol);
    const double state_max_abs_velocity = fmax(1.0e-3, runtime_controls.state_max_abs_velocity);
    const double state_max_abs_pressure = fmax(1.0e-3, runtime_controls.state_max_abs_pressure);
    const double active_alpha_wet = fmin(1.0, fmax(0.5, runtime_controls.active_alpha_wet));
    const double active_alpha_gas = fmin(0.25, fmax(0.0, runtime_controls.active_alpha_gas));
    const bool vof_transport_debug = runtime_controls.vof_transport_debug;
    const int32_t q_inflow_area_policy = (runtime_controls.q_inflow_area_policy != 0) ? 1 : 0;
    const double free_surface_vent_bias = runtime_controls.free_surface_vent_bias;

    constexpr int BLOCK = 256;
    const int64_t n_cells = patch->n_cells;
    const int grid = static_cast<int>((n_cells + static_cast<int64_t>(BLOCK) - 1) / static_cast<int64_t>(BLOCK));
    const auto& desc = patch->desc;
    const double h_char = fmin(fabs(desc.dx), fmin(fabs(desc.dy), fabs(desc.dz)));
    const size_t patch_bytes = static_cast<size_t>(n_cells) * sizeof(double);
    const size_t bc_face_bytes = static_cast<size_t>(SWE3D_PATCH_FACE_COUNT) * sizeof(double);

    double* d_u_backup = nullptr;
    double* d_v_backup = nullptr;
    double* d_w_backup = nullptr;
    double* d_p_backup = nullptr;
    double* d_vof_backup = nullptr;
    bool backups_ready = false;

    unsigned int* d_state_flags = nullptr;
    unsigned long long* d_state_umax_bits = nullptr;
    unsigned long long* d_state_pabs_bits = nullptr;
    bool state_buffers_ready = false;
    double* d_face_effective_area = nullptr;

    auto free_backups = [&]() {
        if (d_u_backup) cudaFree(d_u_backup);
        if (d_v_backup) cudaFree(d_v_backup);
        if (d_w_backup) cudaFree(d_w_backup);
        if (d_p_backup) cudaFree(d_p_backup);
        if (d_vof_backup) cudaFree(d_vof_backup);
        d_u_backup = nullptr;
        d_v_backup = nullptr;
        d_w_backup = nullptr;
        d_p_backup = nullptr;
        d_vof_backup = nullptr;
        backups_ready = false;
    };

    auto free_state_buffers = [&]() {
        if (d_state_flags) cudaFree(d_state_flags);
        if (d_state_umax_bits) cudaFree(d_state_umax_bits);
        if (d_state_pabs_bits) cudaFree(d_state_pabs_bits);
        d_state_flags = nullptr;
        d_state_umax_bits = nullptr;
        d_state_pabs_bits = nullptr;
        state_buffers_ready = false;
    };

    auto free_face_effective_area = [&]() {
        if (d_face_effective_area) cudaFree(d_face_effective_area);
        d_face_effective_area = nullptr;
    };

    if (projection_reject_enabled) {
        try {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_u_backup), patch_bytes));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_v_backup), patch_bytes));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_w_backup), patch_bytes));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_p_backup), patch_bytes));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vof_backup), patch_bytes));

            CUDA_CHECK(cudaMemcpyAsync(d_u_backup, patch->d_u, patch_bytes, cudaMemcpyDeviceToDevice, dev->d_stream));
            CUDA_CHECK(cudaMemcpyAsync(d_v_backup, patch->d_v, patch_bytes, cudaMemcpyDeviceToDevice, dev->d_stream));
            CUDA_CHECK(cudaMemcpyAsync(d_w_backup, patch->d_w, patch_bytes, cudaMemcpyDeviceToDevice, dev->d_stream));
            CUDA_CHECK(cudaMemcpyAsync(d_p_backup, patch->d_p, patch_bytes, cudaMemcpyDeviceToDevice, dev->d_stream));
            CUDA_CHECK(cudaMemcpyAsync(d_vof_backup, patch->d_vof, patch_bytes, cudaMemcpyDeviceToDevice, dev->d_stream));
            backups_ready = true;
        } catch (...) {
            free_backups();
        }
    }

    if (state_reject_enabled) {
        try {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_state_flags), sizeof(unsigned int)));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_state_umax_bits), sizeof(unsigned long long)));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_state_pabs_bits), sizeof(unsigned long long)));
            state_buffers_ready = true;
        } catch (...) {
            free_state_buffers();
        }
    }

    if (q_inflow_area_policy != 0) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_face_effective_area), bc_face_bytes));
    }

    const bool projection_retry_path_active = projection_reject_enabled && backups_ready;
    const bool state_reject_path_active = state_reject_enabled && state_buffers_ready;

    auto restore_backups = [&]() {
        if (!backups_ready) return;
        CUDA_CHECK(cudaMemcpyAsync(patch->d_u, d_u_backup, patch_bytes, cudaMemcpyDeviceToDevice, dev->d_stream));
        CUDA_CHECK(cudaMemcpyAsync(patch->d_v, d_v_backup, patch_bytes, cudaMemcpyDeviceToDevice, dev->d_stream));
        CUDA_CHECK(cudaMemcpyAsync(patch->d_w, d_w_backup, patch_bytes, cudaMemcpyDeviceToDevice, dev->d_stream));
        CUDA_CHECK(cudaMemcpyAsync(patch->d_p, d_p_backup, patch_bytes, cudaMemcpyDeviceToDevice, dev->d_stream));
        CUDA_CHECK(cudaMemcpyAsync(patch->d_vof, d_vof_backup, patch_bytes, cudaMemcpyDeviceToDevice, dev->d_stream));
    };

    auto compute_umax = [&]() -> double {
        CUDA_CHECK(cudaMemsetAsync(
            patch->d_proj_residual_bits,
            0,
            sizeof(unsigned long long),
            dev->d_stream));
        swe3d_velocity_absmax_kernel<<<grid, BLOCK, 0, dev->d_stream>>>(
            n_cells,
            patch->d_u,
            patch->d_v,
            patch->d_w,
            patch->d_proj_residual_bits);
        CUDA_CHECK(cudaGetLastError());

        unsigned long long max_bits = 0ULL;
        CUDA_CHECK(cudaMemcpyAsync(
            &max_bits,
            patch->d_proj_residual_bits,
            sizeof(unsigned long long),
            cudaMemcpyDeviceToHost,
            dev->d_stream));
        CUDA_CHECK(cudaStreamSynchronize(dev->d_stream));

        double umax = 0.0;
        std::memcpy(&umax, &max_bits, sizeof(double));
        return std::isfinite(umax) ? umax : 0.0;
    };

    auto compute_vof_min = [&]() -> double {
        const double vof_init = 1.0;
        unsigned long long min_bits = 0ULL;
        std::memcpy(&min_bits, &vof_init, sizeof(double));
        CUDA_CHECK(cudaMemcpyAsync(
            patch->d_proj_residual_bits,
            &min_bits,
            sizeof(unsigned long long),
            cudaMemcpyHostToDevice,
            dev->d_stream));
        swe3d_vof_min_kernel<<<grid, BLOCK, 0, dev->d_stream>>>(
            n_cells,
            patch->d_phi,
            patch->d_vof,
            patch->d_proj_residual_bits);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(
            &min_bits,
            patch->d_proj_residual_bits,
            sizeof(unsigned long long),
            cudaMemcpyDeviceToHost,
            dev->d_stream));
        CUDA_CHECK(cudaStreamSynchronize(dev->d_stream));

        double vof_min = 0.0;
        std::memcpy(&vof_min, &min_bits, sizeof(double));
        return std::isfinite(vof_min) ? vof_min : 0.0;
    };

    auto compute_vof_sum = [&]() -> double {
        double sum = 0.0;
        CUDA_CHECK(cudaMemsetAsync(patch->d_vof_sum, 0, sizeof(double), dev->d_stream));
        swe3d_vof_sum_kernel<<<grid, BLOCK, 0, dev->d_stream>>>(
            n_cells,
            patch->d_phi,
            patch->d_vof,
            patch->d_vof_sum);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(&sum, patch->d_vof_sum, sizeof(double), cudaMemcpyDeviceToHost, dev->d_stream));
        CUDA_CHECK(cudaStreamSynchronize(dev->d_stream));
        return std::isfinite(sum) ? sum : 0.0;
    };

    auto compute_divergence_rms_from_rhs = [&](double dt_local) -> double {
        if (!(dt_local > 0.0) || !std::isfinite(dt_local)) {
            return 0.0;
        }
        double sum_sq = 0.0;
        CUDA_CHECK(cudaMemsetAsync(patch->d_vof_sum, 0, sizeof(double), dev->d_stream));
        swe3d_sum_sq_kernel<<<grid, BLOCK, 0, dev->d_stream>>>(
            n_cells,
            patch->d_p_rhs,
            patch->d_vof_sum);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(&sum_sq, patch->d_vof_sum, sizeof(double), cudaMemcpyDeviceToHost, dev->d_stream));
        CUDA_CHECK(cudaStreamSynchronize(dev->d_stream));
        if (!std::isfinite(sum_sq) || sum_sq <= 0.0) {
            return 0.0;
        }
        const double rhs_rms = std::sqrt(sum_sq / static_cast<double>(n_cells));
        const double div_rms = dt_local * rhs_rms;
        return std::isfinite(div_rms) ? div_rms : 0.0;
    };

    auto evaluate_state_health = [&](unsigned int& flags_out, double& umax_out, double& pabs_out) -> bool {
        flags_out = 0u;
        umax_out = 0.0;
        pabs_out = 0.0;

        if (!state_reject_path_active) {
            return true;
        }

        CUDA_CHECK(cudaMemsetAsync(d_state_flags, 0, sizeof(unsigned int), dev->d_stream));
        CUDA_CHECK(cudaMemsetAsync(d_state_umax_bits, 0, sizeof(unsigned long long), dev->d_stream));
        CUDA_CHECK(cudaMemsetAsync(d_state_pabs_bits, 0, sizeof(unsigned long long), dev->d_stream));

        swe3d_state_health_kernel<<<grid, BLOCK, 0, dev->d_stream>>>(
            n_cells,
            patch->d_u,
            patch->d_v,
            patch->d_w,
            patch->d_p,
            patch->d_vof,
            state_vof_bounds_tol,
            d_state_flags,
            d_state_umax_bits,
            d_state_pabs_bits);
        CUDA_CHECK(cudaGetLastError());

        unsigned int flags = 0u;
        unsigned long long umax_bits = 0ULL;
        unsigned long long pabs_bits = 0ULL;
        CUDA_CHECK(cudaMemcpyAsync(
            &flags,
            d_state_flags,
            sizeof(unsigned int),
            cudaMemcpyDeviceToHost,
            dev->d_stream));
        CUDA_CHECK(cudaMemcpyAsync(
            &umax_bits,
            d_state_umax_bits,
            sizeof(unsigned long long),
            cudaMemcpyDeviceToHost,
            dev->d_stream));
        CUDA_CHECK(cudaMemcpyAsync(
            &pabs_bits,
            d_state_pabs_bits,
            sizeof(unsigned long long),
            cudaMemcpyDeviceToHost,
            dev->d_stream));
        CUDA_CHECK(cudaStreamSynchronize(dev->d_stream));

        std::memcpy(&umax_out, &umax_bits, sizeof(double));
        std::memcpy(&pabs_out, &pabs_bits, sizeof(double));
        flags_out = flags;

        const bool finite_ok = (flags_out & 1u) == 0u;
        const bool vof_ok = (flags_out & 2u) == 0u;
        const bool vel_ok = std::isfinite(umax_out) && umax_out <= state_max_abs_velocity;
        const bool p_ok = std::isfinite(pabs_out) && pabs_out <= state_max_abs_pressure;
        return finite_ok && vof_ok && vel_ok && p_ok;
    };

    const int32_t wall_mode = static_cast<int32_t>(SWE3DBoundaryMode::WALL);
    const int32_t free_surface_mode = static_cast<int32_t>(SWE3DBoundaryMode::FREE_SURFACE);
    const int32_t outflow_mode = static_cast<int32_t>(SWE3DBoundaryMode::OUTFLOW);
    const int32_t inflow_mode = static_cast<int32_t>(SWE3DBoundaryMode::INFLOW);
    const int32_t inflow_q_mode = static_cast<int32_t>(SWE3DBoundaryMode::INFLOW_FLOW_RATE);

    bool transport_boundary_present = false;
    for (int32_t face = 0; face < SWE3D_PATCH_FACE_COUNT; ++face) {
        const int32_t mode = desc.bc_mode[face];
        if (mode == outflow_mode || mode == inflow_mode || mode == inflow_q_mode) {
            transport_boundary_present = true;
            break;
        }
    }

    const bool closed_box_faces =
        desc.bc_mode[static_cast<int32_t>(SWE3DPatchBoundaryFace::XMIN)] == wall_mode &&
        desc.bc_mode[static_cast<int32_t>(SWE3DPatchBoundaryFace::XMAX)] == wall_mode &&
        desc.bc_mode[static_cast<int32_t>(SWE3DPatchBoundaryFace::YMIN)] == wall_mode &&
        desc.bc_mode[static_cast<int32_t>(SWE3DPatchBoundaryFace::YMAX)] == wall_mode &&
        desc.bc_mode[static_cast<int32_t>(SWE3DPatchBoundaryFace::ZMIN)] == wall_mode &&
        (desc.bc_mode[static_cast<int32_t>(SWE3DPatchBoundaryFace::ZMAX)] == wall_mode ||
         desc.bc_mode[static_cast<int32_t>(SWE3DPatchBoundaryFace::ZMAX)] == free_surface_mode);
    const double vof_min_pre = compute_vof_min();
    const double umax_pre_step = compute_umax();
    constexpr double CLOSED_BOX_FULL_WET_UMAX_CAP = 5.0;
    // Full-wet closed boxes get the strongest damping/cap branch.
    const bool closed_box_full_wet_low_speed =
        closed_box_faces &&
        !transport_boundary_present &&
        vof_min_pre >= active_alpha_wet &&
        umax_pre_step <= CLOSED_BOX_FULL_WET_UMAX_CAP;
    // Partially wet closed boxes use lighter transport-only stabilization.
    const bool closed_box_partial_low_speed =
        closed_box_faces &&
        !transport_boundary_present &&
        vof_min_pre < active_alpha_wet &&
        umax_pre_step <= CLOSED_BOX_FULL_WET_UMAX_CAP;

    auto projection_scale_from_ratio = [&](double residual_ratio) {
        if (!std::isfinite(residual_ratio)) {
            return projection_scale_min;
        }
        const double health = 1.0 / (1.0 + fmax(0.0, residual_ratio));
        const double blended = projection_scale_min + (projection_scale_max - projection_scale_min) * health;
        return fmin(projection_scale_max, fmax(projection_scale_min, blended));
    };
    double projection_correction_scale_used = projection_scale_max;
    if (std::isfinite(patch->last_projection_residual)) {
        projection_correction_scale_used = projection_scale_from_ratio(
            patch->last_projection_residual / projection_target);
    }

    auto run_single_attempt = [&](double dt_attempt, double& cfl_estimate_out, double& divergence_ratio_out, double& divergence_ratio_max_out) {
        constexpr double JACOBI_TOL = 1.0e-6;
        constexpr double VOF_CFL = 0.45;
        const int jacobi_max_iters = std::max(8, runtime_controls.projection_jacobi_max_iters);
        const int jacobi_residual_sample_iters =
            std::max(1, runtime_controls.projection_residual_sample_iters);

        int vof_substeps = 1;
        const double umax_pre = compute_umax();
        if (umax_pre > 1.0e-12 && h_char > 1.0e-12) {
            const double dt_max_vof = VOF_CFL * h_char / umax_pre;
            if (dt_max_vof > 0.0 && dt_attempt > dt_max_vof) {
                vof_substeps = static_cast<int>(std::ceil(dt_attempt / dt_max_vof));
                vof_substeps = std::max(1, std::min(static_cast<int>(vof_max_substeps_cap), vof_substeps));
            }
        }
        patch->last_vof_substeps = vof_substeps;

        const double dt_sub = dt_attempt / static_cast<double>(vof_substeps);
        const double effective_predictor_damping_coeff =
            closed_box_full_wet_low_speed
            ? (8.0 * predictor_damping_coeff)
            : predictor_damping_coeff;
        double attempt_residual_max = 0.0;
        int attempt_iters_max = 0;
        bool attempt_converged = true;
        double attempt_divergence_ratio_last = 1.0;
        double attempt_divergence_ratio_max = 0.0;

        for (int sub = 0; sub < vof_substeps; ++sub) {
            swe3d_build_active_mask_kernel<<<grid, BLOCK, 0, dev->d_stream>>>(
                desc,
                n_cells,
                patch->d_phi,
                patch->d_vof,
                active_alpha_wet,
                active_alpha_gas,
                patch->d_active_mask);
            CUDA_CHECK(cudaGetLastError());

            if (d_face_effective_area) {
                CUDA_CHECK(cudaMemsetAsync(d_face_effective_area, 0, bc_face_bytes, dev->d_stream));
                swe3d_reduce_face_effective_area_kernel<<<grid, BLOCK, 0, dev->d_stream>>>(
                    desc,
                    n_cells,
                    patch->d_phi,
                    patch->d_vof,
                    patch->d_ax,
                    patch->d_ay,
                    patch->d_az,
                    patch->d_active_mask,
                    d_face_effective_area);
                CUDA_CHECK(cudaGetLastError());
            }

            swe3d_single_phase_predictor_kernel<<<grid, BLOCK, 0, dev->d_stream>>>(
                n_cells,
                patch->d_u,
                patch->d_v,
                patch->d_w,
                patch->d_phi,
                patch->d_ax,
                patch->d_ay,
                patch->d_az,
                patch->d_active_mask,
                patch->d_vof,
                dt_sub,
                effective_predictor_damping_coeff);
            CUDA_CHECK(cudaGetLastError());

            swe3d_apply_wall_noslip_kernel<<<grid, BLOCK, 0, dev->d_stream>>>(
                desc,
                n_cells,
                patch->d_u,
                patch->d_v,
                patch->d_w);
            CUDA_CHECK(cudaGetLastError());

            swe3d_apply_face_inflow_bc_kernel<<<grid, BLOCK, 0, dev->d_stream>>>(
                desc,
                n_cells,
                d_face_effective_area,
                q_inflow_area_policy,
                free_surface_vent_bias,
                patch->d_phi,
                patch->d_vof,
                patch->d_u,
                patch->d_v,
                patch->d_w);
            CUDA_CHECK(cudaGetLastError());

            swe3d_compute_pressure_rhs_kernel<<<grid, BLOCK, 0, dev->d_stream>>>(
                desc.nx,
                desc.ny,
                desc.nz,
                desc.dx,
                desc.dy,
                desc.dz,
                patch->d_u,
                patch->d_v,
                patch->d_w,
                patch->d_vof,
                patch->d_active_mask,
                dt_sub,
                patch->d_p_rhs);
            CUDA_CHECK(cudaGetLastError());
            const double divergence_rms_pre = compute_divergence_rms_from_rhs(dt_sub);

            double last_residual = std::numeric_limits<double>::infinity();
            int iter_count = 0;
            bool converged = false;
            for (int iter = 0; iter < jacobi_max_iters; ++iter) {
                swe3d_pressure_jacobi_kernel<<<grid, BLOCK, 0, dev->d_stream>>>(
                    desc,
                    desc.nx,
                    desc.ny,
                    desc.nz,
                    desc.dx,
                    desc.dy,
                    desc.dz,
                    free_surface_gauge_tolerance_pa,
                    patch->d_p,
                    patch->d_vof,
                    patch->d_active_mask,
                    patch->d_p_rhs,
                    patch->d_p_tmp);
                CUDA_CHECK(cudaGetLastError());

                const bool sample_residual =
                    ((iter + 1) % jacobi_residual_sample_iters == 0) ||
                    (iter + 1 == jacobi_max_iters);
                if (sample_residual) {
                    CUDA_CHECK(cudaMemsetAsync(
                        patch->d_proj_residual_bits,
                        0,
                        sizeof(unsigned long long),
                        dev->d_stream));
                    swe3d_projection_residual_max_kernel<<<grid, BLOCK, 0, dev->d_stream>>>(
                        n_cells,
                        patch->d_p,
                        patch->d_p_tmp,
                        patch->d_proj_residual_bits);
                    CUDA_CHECK(cudaGetLastError());

                    unsigned long long residual_bits = 0ULL;
                    CUDA_CHECK(cudaMemcpyAsync(
                        &residual_bits,
                        patch->d_proj_residual_bits,
                        sizeof(unsigned long long),
                        cudaMemcpyDeviceToHost,
                        dev->d_stream));
                    CUDA_CHECK(cudaStreamSynchronize(dev->d_stream));
                    std::memcpy(&last_residual, &residual_bits, sizeof(double));
                }

                std::swap(patch->d_p, patch->d_p_tmp);
                iter_count = iter + 1;
                if (sample_residual && last_residual <= JACOBI_TOL) {
                    converged = true;
                    break;
                }
            }

            swe3d_velocity_correction_kernel<<<grid, BLOCK, 0, dev->d_stream>>>(
                desc.nx,
                desc.ny,
                desc.nz,
                desc.dx,
                desc.dy,
                desc.dz,
                patch->d_p,
                patch->d_vof,
                patch->d_active_mask,
                dt_sub,
                projection_correction_scale_used,
                patch->d_u,
                patch->d_v,
                patch->d_w);
            CUDA_CHECK(cudaGetLastError());

            swe3d_apply_wall_noslip_kernel<<<grid, BLOCK, 0, dev->d_stream>>>(
                desc,
                n_cells,
                patch->d_u,
                patch->d_v,
                patch->d_w);
            CUDA_CHECK(cudaGetLastError());

            swe3d_apply_face_inflow_bc_kernel<<<grid, BLOCK, 0, dev->d_stream>>>(
                desc,
                n_cells,
                d_face_effective_area,
                q_inflow_area_policy,
                free_surface_vent_bias,
                patch->d_phi,
                patch->d_vof,
                patch->d_u,
                patch->d_v,
                patch->d_w);
            CUDA_CHECK(cudaGetLastError());

            // Re-evaluate divergence after projection correction for quality telemetry.
            swe3d_compute_pressure_rhs_kernel<<<grid, BLOCK, 0, dev->d_stream>>>(
                desc.nx,
                desc.ny,
                desc.nz,
                desc.dx,
                desc.dy,
                desc.dz,
                patch->d_u,
                patch->d_v,
                patch->d_w,
                patch->d_vof,
                patch->d_active_mask,
                dt_sub,
                patch->d_p_rhs);
            CUDA_CHECK(cudaGetLastError());
            const double divergence_rms_post = compute_divergence_rms_from_rhs(dt_sub);
            double divergence_ratio = 1.0;
            if (divergence_rms_pre > 1.0e-12) {
                divergence_ratio = divergence_rms_post / divergence_rms_pre;
            } else {
                divergence_ratio = (divergence_rms_post <= 1.0e-12) ? 0.0 : std::numeric_limits<double>::infinity();
            }
            attempt_divergence_ratio_last = divergence_ratio;
            if (std::isfinite(divergence_ratio)) {
                attempt_divergence_ratio_max = fmax(attempt_divergence_ratio_max, divergence_ratio);
            } else {
                attempt_divergence_ratio_max = std::numeric_limits<double>::infinity();
            }

            if (std::isfinite(last_residual) && last_residual > 1.0) {
                // Guardrail for severely under-resolved projection solves.
                if (vof_min_pre >= active_alpha_wet) {
                    swe3d_velocity_scale_kernel<<<grid, BLOCK, 0, dev->d_stream>>>(
                        n_cells,
                        0.4,
                        patch->d_u,
                        patch->d_v,
                        patch->d_w);
                    CUDA_CHECK(cudaGetLastError());
                }
            }

            if (closed_box_partial_low_speed &&
                std::isfinite(last_residual) &&
                last_residual > 5.0e-2) {
                // In partially wet closed boxes, damp post-projection velocity
                // spikes before transport to reduce non-conservative clamp churn.
                // This path is low-speed-only to avoid affecting obstruction/high-speed cases.
                const double scale = fmax(0.2, fmin(1.0, 5.0e-2 / last_residual));
                swe3d_velocity_scale_kernel<<<grid, BLOCK, 0, dev->d_stream>>>(
                    n_cells,
                    scale,
                    patch->d_u,
                    patch->d_v,
                    patch->d_w);
                CUDA_CHECK(cudaGetLastError());
            }

            const double vof_sum_before = vof_transport_debug ? compute_vof_sum() : 0.0;

            const bool open_boundary_projection_stress =
                transport_boundary_present &&
                std::isfinite(last_residual) &&
                (last_residual > fmax(5.0e-2, 10.0 * projection_target));

            // Disable high-order interface reconstruction in two narrow slices:
            // 1) low-speed partially wet closed boxes, and
            // 2) open-boundary runs under severe projection stress.
            const int32_t use_high_order_transport =
                (closed_box_partial_low_speed || open_boundary_projection_stress)
                    ? 0
                    : 1;

            swe3d_vof_transport_upwind_kernel<<<grid, BLOCK, 0, dev->d_stream>>>(
                desc,
                n_cells,
                use_high_order_transport,
                d_face_effective_area,
                q_inflow_area_policy,
                free_surface_vent_bias,
                dt_sub,
                patch->d_u,
                patch->d_v,
                patch->d_w,
                patch->d_phi,
                patch->d_ax,
                patch->d_ay,
                patch->d_az,
                patch->d_active_mask,
                active_alpha_wet,
                active_alpha_gas,
                patch->d_vof,
                patch->d_vof_tmp);
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaMemcpyAsync(
                patch->d_vof,
                patch->d_vof_tmp,
                patch_bytes,
                cudaMemcpyDeviceToDevice,
                dev->d_stream));

            if (vof_transport_debug) {
                CUDA_CHECK(cudaStreamSynchronize(dev->d_stream));
                const double vof_sum_after = compute_vof_sum();
                const double vof_abs_delta = vof_sum_after - vof_sum_before;
                const double vof_rel_delta = vof_abs_delta / fmax(1.0, fabs(vof_sum_before));
                std::fprintf(
                    stderr,
                    "[SWE3D_DEBUG] transport substep=%d/%d dt_sub=%.6e resid=%.6e vof_sum_before=%.9e vof_sum_after=%.9e vof_abs_delta=%.9e vof_rel_delta=%.9e\n",
                    sub + 1,
                    vof_substeps,
                    dt_sub,
                    last_residual,
                    vof_sum_before,
                    vof_sum_after,
                    vof_abs_delta,
                    vof_rel_delta);
            }

            attempt_residual_max = fmax(attempt_residual_max, std::isfinite(last_residual) ? last_residual : 1.0e30);
            attempt_iters_max = std::max(attempt_iters_max, iter_count);
            attempt_converged = attempt_converged && converged;
        }

        patch->last_projection_iters = attempt_iters_max;
        patch->last_projection_residual = attempt_residual_max;
        patch->last_projection_converged = attempt_converged;
        divergence_ratio_out = attempt_divergence_ratio_last;
        divergence_ratio_max_out = attempt_divergence_ratio_max;

        const double umax_post = compute_umax();
        cfl_estimate_out =
            (h_char > 1.0e-12 && umax_post > 0.0)
                ? (dt_attempt * umax_post / h_char)
                : 0.0;
    };

    double dt_attempt = dt;
    if (closed_box_full_wet_low_speed) {
        const double dt_closed_box = runtime_controls.closed_box_full_wet_dt_cfl * h_char / umax_pre_step;
        if (std::isfinite(dt_closed_box) && dt_closed_box > 0.0) {
            dt_attempt = fmax(1.0e-9, fmin(dt_attempt, dt_closed_box));
        }
    }
    double cfl_estimate = 0.0;
    int32_t attempt_index = 0;
    int32_t projection_attempt_count = 0;
    int32_t projection_retry_count = 0;
    bool projection_retry_exhausted = false;
    bool projection_retry_exhausted_due_state = false;
    double projection_residual_ratio = 0.0;
    double projection_residual_ratio_max = 0.0;
    double projection_divergence_ratio = 1.0;
    double projection_divergence_ratio_max = 0.0;
    unsigned int state_guard_flags = 0u;
    double state_guard_umax = 0.0;
    double state_guard_pabs = 0.0;
    bool state_guard_last_ok = true;
    try {
        while (true) {
            if (attempt_index > 0 && projection_retry_path_active) {
                restore_backups();
            }

            ++projection_attempt_count;
            run_single_attempt(
                dt_attempt,
                cfl_estimate,
                projection_divergence_ratio,
                projection_divergence_ratio_max);

            const bool projection_residual_finite = std::isfinite(patch->last_projection_residual);
            if (projection_residual_finite) {
                projection_residual_ratio = patch->last_projection_residual / projection_target;
            } else {
                projection_residual_ratio = std::numeric_limits<double>::infinity();
            }
            if (std::isfinite(projection_residual_ratio)) {
                projection_residual_ratio_max = fmax(projection_residual_ratio_max, projection_residual_ratio);
            } else {
                projection_residual_ratio_max = std::numeric_limits<double>::infinity();
            }
            projection_correction_scale_used = projection_scale_from_ratio(projection_residual_ratio);

            const bool projection_ok =
                projection_residual_finite &&
                projection_residual_ratio <= 1.0;
            const bool divergence_ratio_finite = std::isfinite(projection_divergence_ratio);
            const bool projection_divergence_ok =
                (!projection_divergence_gate_enable) ||
                (divergence_ratio_finite && projection_divergence_ratio <= projection_divergence_ratio_target);
            state_guard_last_ok = evaluate_state_health(state_guard_flags, state_guard_umax, state_guard_pabs);
            const bool attempt_ok = projection_ok && projection_divergence_ok && state_guard_last_ok;

            if (!projection_retry_path_active || attempt_ok) {
                break;
            }
            if (attempt_index >= projection_max_retries) {
                projection_retry_exhausted = true;
                projection_retry_exhausted_due_state = !state_guard_last_ok;
                break;
            }

            const double dt_next = fmax(projection_dt_floor, dt_attempt * projection_reduction);
            if (!(std::isfinite(dt_next) && dt_next < dt_attempt)) {
                projection_retry_exhausted = true;
                break;
            }

            dt_attempt = dt_next;
            ++attempt_index;
            ++projection_retry_count;
        }
    } catch (...) {
        free_face_effective_area();
        free_backups();
        free_state_buffers();
        throw;
    }

    free_face_effective_area();
    free_backups();
    free_state_buffers();

    if (projection_retry_exhausted) {
        const double resid = patch->last_projection_residual;
        const double ratio = projection_residual_ratio;
        if (projection_fail_fast) {
            if (projection_retry_exhausted_due_state) {
                throw std::runtime_error(
                    "swe2d_gpu_step_3d_single_phase_free_surface: state guard retry exhausted; "
                    "set BACKWATER_SWE3D_PROJECTION_FAIL_FAST=0 to continue with diagnostics");
            }
            throw std::runtime_error(
                "swe2d_gpu_step_3d_single_phase_free_surface: projection retry exhausted; "
                "set BACKWATER_SWE3D_PROJECTION_FAIL_FAST=0 to continue with diagnostics");
        }
        std::fprintf(
            stderr,
            "[SWE3D_WARN] projection retry exhausted: attempts=%d retries=%d dt_initial=%.6e dt_final=%.6e dt_floor=%.6e resid=%.6e target=%.6e ratio=%.6e ratio_max=%.6e\n",
            static_cast<int>(std::max<int32_t>(1, projection_attempt_count)),
            static_cast<int>(projection_retry_count),
            dt,
            dt_attempt,
            projection_dt_floor,
            resid,
            projection_target,
            ratio,
            projection_residual_ratio_max);
        if (projection_retry_exhausted_due_state) {
            std::fprintf(
                stderr,
                "[SWE3D_WARN] state guard triggered retry exhaustion: flags=0x%x umax=%.6e pabs_max=%.6e limits(umax<=%.6e, pabs<=%.6e, vof_tol=%.6e)\n",
                state_guard_flags,
                state_guard_umax,
                state_guard_pabs,
                state_max_abs_velocity,
                state_max_abs_pressure,
                state_vof_bounds_tol);
        }

        // Recovery path for diagnostics mode (fail-fast disabled): reduce
        // velocity magnitudes after retry exhaustion so the next outer step has
        // a chance to re-enter a stable regime instead of compounding blow-up.
        double emergency_umax = state_guard_umax;
        if (!std::isfinite(emergency_umax) || emergency_umax <= 0.0) {
            emergency_umax = compute_umax();
        }
        if (std::isfinite(emergency_umax) && emergency_umax > state_max_abs_velocity) {
            const double raw_scale = state_max_abs_velocity / emergency_umax;
            const double scale = fmax(0.05, fmin(1.0, raw_scale));
            swe3d_velocity_scale_kernel<<<grid, BLOCK, 0, dev->d_stream>>>(
                n_cells,
                scale,
                patch->d_u,
                patch->d_v,
                patch->d_w);
            CUDA_CHECK(cudaGetLastError());
            std::fprintf(
                stderr,
                "[SWE3D_WARN] emergency velocity attenuation applied after retry exhaustion: umax=%.6e limit=%.6e scale=%.6e\n",
                emergency_umax,
                state_max_abs_velocity,
                scale);
        }
    } else if (state_reject_enabled && !state_guard_last_ok && !projection_retry_path_active) {
        std::fprintf(
            stderr,
            "[SWE3D_WARN] state guard failed but retry path unavailable (backup alloc failed): flags=0x%x umax=%.6e pabs_max=%.6e\n",
            state_guard_flags,
            state_guard_umax,
            state_guard_pabs);
    }

    if (coupling_mode != static_cast<int>(SWE2DThreeDCouplingMode::OFF)) {
        if (!dev->coupling_iface) {
            throw std::runtime_error(
                "swe2d_gpu_step_3d_single_phase_free_surface: 2D-3D coupling requested but no interface contract is uploaded");
        }
        swe2d_gpu_apply_2d3d_exchange_skeleton(dev, dt_attempt, g, coupling_mode, true, diag);
    }

    if (diag) {
        diag->dt = dt_attempt;
        diag->gpu_active = true;
        diag->wet_cells = -1;
        diag->max_depth = -1.0;
        diag->min_depth = -1.0;
        diag->mass_total = -1.0;
        diag->max_courant = cfl_estimate;
        diag->max_depth_residual = -1.0;
        diag->max_wse_elev_error = -1.0;
        diag->gpu_graph_launches_step = 0;
        diag->gpu_graph_launches_total = static_cast<int64_t>(dev->graph_replay_count);
        diag->projection_retry_count = projection_retry_count;
        diag->projection_attempt_count = std::max<int32_t>(1, projection_attempt_count);
        diag->projection_retry_exhausted = projection_retry_exhausted;
        diag->projection_retry_enabled = projection_retry_path_active;
        diag->projection_retry_dt_initial = dt;
        diag->projection_retry_dt_floor = projection_dt_floor;
        diag->projection_retry_dt_reduction = projection_reduction;
        diag->projection_retry_residual_target = projection_target;
        diag->projection_retry_residual_ratio = projection_residual_ratio;
        diag->projection_retry_residual_ratio_max = projection_residual_ratio_max;
        diag->projection_divergence_ratio = projection_divergence_ratio;
        diag->projection_divergence_ratio_max = projection_divergence_ratio_max;
        diag->projection_divergence_gate_enabled = projection_divergence_gate_enable;
        diag->projection_divergence_ratio_target = projection_divergence_ratio_target;
        diag->projection_retry_fail_fast = projection_fail_fast;
        diag->projection_correction_scale_used = projection_correction_scale_used;
    }

    if (sync_diagnostics) {
        CUDA_CHECK(cudaStreamSynchronize(dev->d_stream));
    }
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

// Phase 8B: PCG kernels and host loop.

__global__ void swe2d_gpu_pcg_init_kernel(
    int32_t n_cells,
    const double* d_rhs,
    const double* d_diag,
    double* d_pressure,
    double* d_r,
    double* d_z,
    double* d_dir)
{
    const int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells) return;

    const double rhs = d_rhs[c];
    const double diag = d_diag[c] > 1.0e-14 ? d_diag[c] : 1.0e-14;
    d_pressure[c] = 0.0;
    d_r[c] = rhs;
    d_z[c] = rhs / diag;
    d_dir[c] = d_z[c];
}

__global__ void swe2d_gpu_dot_product_kernel(
    int32_t n_cells,
    const double* d_a,
    const double* d_b,
    double* d_out)
{
    const int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells) return;
    atomicAdd(d_out, d_a[c] * d_b[c]);
}

__global__ void swe2d_gpu_pcg_update_pr_kernel(
    int32_t n_cells,
    double alpha,
    const double* d_dir,
    const double* d_Ap,
    double* d_pressure,
    double* d_r,
    const double* d_diag,
    double* d_z)
{
    const int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells) return;

    d_pressure[c] += alpha * d_dir[c];
    d_r[c] -= alpha * d_Ap[c];
    const double diag = d_diag[c] > 1.0e-14 ? d_diag[c] : 1.0e-14;
    d_z[c] = d_r[c] / diag;
}

__global__ void swe2d_gpu_pcg_update_dir_kernel(
    int32_t n_cells,
    double beta,
    const double* d_z,
    double* d_dir)
{
    const int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells) return;
    d_dir[c] = d_z[c] + beta * d_dir[c];
}

static bool swe2d_gpu_dot_product(
    SWE2DDeviceState* dev,
    const double* d_a,
    const double* d_b,
    double* d_tmp_scalar,
    double* out)
{
    if (!dev || !d_a || !d_b || !d_tmp_scalar || !out) return false;
    const int32_t n_cells = dev->n_cells;
    if (n_cells <= 0) return false;

    constexpr int BLOCK = 256;
    const int grid = (n_cells + BLOCK - 1) / BLOCK;
    CUDA_CHECK(cudaMemsetAsync(d_tmp_scalar, 0, sizeof(double), dev->d_stream));
    swe2d_gpu_dot_product_kernel<<<grid, BLOCK, 0, dev->d_stream>>>(
        n_cells,
        d_a,
        d_b,
        d_tmp_scalar);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(out, d_tmp_scalar, sizeof(double), cudaMemcpyDeviceToHost, dev->d_stream));
    CUDA_CHECK(cudaStreamSynchronize(dev->d_stream));
    return true;
}

static bool swe2d_gpu_apply_laplacian_vec(
    SWE2DDeviceState* dev,
    const double* d_vec,
    double* d_out)
{
    if (!dev || !d_vec || !d_out || dev->n_cells <= 0) return false;
    if (!dev->d_cell_edge_offsets || !dev->d_cell_edge_ids) return false;

    constexpr int BLOCK = 256;
    const int grid = (dev->n_cells + BLOCK - 1) / BLOCK;
    swe2d_gpu_laplacian_stencil_kernel<<<grid, BLOCK, 0, dev->d_stream>>>(
        dev->n_cells,
        d_vec,
        dev->d_cell_area,
        dev->d_cell_inv_area,
        dev->d_cell_edge_offsets,
        dev->d_cell_edge_ids,
        dev->d_edge_c0,
        dev->d_edge_c1,
        dev->d_edge_len,
        d_out,
        dev->nh_workspace.d_stencil_diag);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

static bool swe2d_gpu_solve_pressure_pcg(
    SWE2DDeviceState* dev,
    const SWE2DNonhydroPcConfig& nh_cfg,
    SWE2DNonhydroPcDiag* nh_diag)
{
    if (!dev || dev->n_cells <= 0) return false;
    auto& ws = dev->nh_workspace;
    if (!ws.is_configured || !ws.d_p || !ws.d_p_rhs || !ws.d_stencil_diag) return false;
    if (!ws.d_pcg_r || !ws.d_pcg_p || !ws.d_pcg_Ap || !ws.d_pcg_z) return false;
    if (!ws.d_pcg_rr || !ws.d_pcg_rrold || !ws.d_pcg_pAp) return false;

    constexpr int BLOCK = 256;
    const int grid = (dev->n_cells + BLOCK - 1) / BLOCK;

    swe2d_gpu_pcg_init_kernel<<<grid, BLOCK, 0, dev->d_stream>>>(
        dev->n_cells,
        ws.d_p_rhs,
        ws.d_stencil_diag,
        ws.d_p,
        ws.d_pcg_r,
        ws.d_pcg_z,
        ws.d_pcg_p);
    CUDA_CHECK(cudaGetLastError());

    double rhs_rr = 0.0;
    if (!swe2d_gpu_dot_product(dev, ws.d_p_rhs, ws.d_p_rhs, ws.d_pcg_rr, &rhs_rr)) return false;
    const double rhs_norm = std::sqrt(std::max(rhs_rr, 0.0));

    double rr_old = 0.0;
    if (!swe2d_gpu_dot_product(dev, ws.d_pcg_r, ws.d_pcg_z, ws.d_pcg_rrold, &rr_old)) return false;
    rr_old = std::max(rr_old, 0.0);

    const int max_iter = std::max(1, nh_cfg.pressure_max_iters);
    const double tol_rel = nh_cfg.pressure_tol > 0.0 ? nh_cfg.pressure_tol : 1.0e-3;
    const double tol_abs = 1.0e-6;

    int iter_count = 0;
    double residual_norm = std::sqrt(rr_old);
    bool converged = (residual_norm <= tol_abs) || (rhs_norm > 0.0 && (residual_norm / rhs_norm) <= tol_rel);

    for (int iter = 0; iter < max_iter && !converged; ++iter) {
        if (!swe2d_gpu_apply_laplacian_vec(dev, ws.d_pcg_p, ws.d_pcg_Ap)) return false;

        double pAp = 0.0;
        if (!swe2d_gpu_dot_product(dev, ws.d_pcg_p, ws.d_pcg_Ap, ws.d_pcg_pAp, &pAp)) return false;
        if (std::abs(pAp) < 1.0e-20) {
            break;
        }

        const double alpha = rr_old / pAp;
        swe2d_gpu_pcg_update_pr_kernel<<<grid, BLOCK, 0, dev->d_stream>>>(
            dev->n_cells,
            alpha,
            ws.d_pcg_p,
            ws.d_pcg_Ap,
            ws.d_p,
            ws.d_pcg_r,
            ws.d_stencil_diag,
            ws.d_pcg_z);
        CUDA_CHECK(cudaGetLastError());

        double rr_new = 0.0;
        if (!swe2d_gpu_dot_product(dev, ws.d_pcg_r, ws.d_pcg_z, ws.d_pcg_rr, &rr_new)) return false;
        rr_new = std::max(rr_new, 0.0);
        residual_norm = std::sqrt(rr_new);
        iter_count = iter + 1;

        converged = (residual_norm <= tol_abs) || (rhs_norm > 0.0 && (residual_norm / rhs_norm) <= tol_rel);
        if (converged) {
            rr_old = rr_new;
            break;
        }

        if (rr_old <= 1.0e-30) {
            rr_old = rr_new;
            break;
        }
        const double beta = rr_new / rr_old;
        swe2d_gpu_pcg_update_dir_kernel<<<grid, BLOCK, 0, dev->d_stream>>>(
            dev->n_cells,
            beta,
            ws.d_pcg_z,
            ws.d_pcg_p);
        CUDA_CHECK(cudaGetLastError());
        rr_old = rr_new;
    }

    CUDA_CHECK(cudaStreamSynchronize(dev->d_stream));

    if (nh_diag) {
        nh_diag->pressure_iters = iter_count;
        nh_diag->pressure_residual = residual_norm;
    }
    return true;
}

// Phase 8C: Apply momentum correction from solved pressure field.
// Computes cell-centered pressure gradient and applies velocity correction:
//   du = -dt * grad(p)_x
//   dv = -dt * grad(p)_y
//   hu += h * du, hv += h * dv
__global__ void swe2d_gpu_momentum_corrector_kernel(
    int32_t n_cells,
    double dt,
    double g,
    double h_min,
    double relax,
    int velocity_correction_mode,
    const double* d_h,
    const double* d_p,
    const int32_t* d_cell_edge_offsets,
    const int32_t* d_cell_edge_ids,
    const int32_t* d_edge_c0,
    const int32_t* d_edge_c1,
    const double* d_edge_nx,
    const double* d_edge_ny,
    const double* d_edge_len,
    const double* d_cell_inv_area,
    double* d_u_corr,
    double* d_v_corr,
    double* d_hu,
    double* d_hv)
{
    const int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells) return;

    const double h = d_h[c];
    if (!(h > h_min) || !std::isfinite(h)) {
        d_u_corr[c] = 0.0;
        d_v_corr[c] = 0.0;
        return;
    }

    const double p_c = d_p[c];
    if (!std::isfinite(p_c)) {
        d_u_corr[c] = 0.0;
        d_v_corr[c] = 0.0;
        return;
    }

    double grad_px = 0.0;
    double grad_py = 0.0;
    const int32_t offset_start = d_cell_edge_offsets[c];
    const int32_t offset_end = d_cell_edge_offsets[c + 1];
    for (int32_t j = offset_start; j < offset_end; ++j) {
        const int32_t edge_id = d_cell_edge_ids[j];
        if (edge_id < 0) continue;

        const int32_t c0 = d_edge_c0[edge_id];
        const int32_t c1 = d_edge_c1[edge_id];
        int32_t c_nb = -1;
        double sign = 0.0;
        if (c0 == c) {
            c_nb = c1;
            sign = 1.0;
        } else if (c1 == c) {
            c_nb = c0;
            sign = -1.0;
        } else {
            continue;
        }
        if (c_nb < 0 || c_nb >= n_cells) {
            // Homogeneous Neumann pressure BC on boundary edges.
            continue;
        }

        const double p_nb = d_p[c_nb];
        if (!std::isfinite(p_nb)) continue;

        const double dp = p_nb - p_c;
        const double nx_out = sign * d_edge_nx[edge_id];
        const double ny_out = sign * d_edge_ny[edge_id];
        const double len = d_edge_len[edge_id];
        grad_px += dp * nx_out * len;
        grad_py += dp * ny_out * len;
    }

    const double inv_area = d_cell_inv_area[c];
    grad_px *= inv_area;
    grad_py *= inv_area;

    double du = -dt * grad_px;
    double dv = -dt * grad_py;

    // Optional extra damping for the energy-stable mode scaffold.
    if (velocity_correction_mode == static_cast<int>(SWE2DVelocityCorrectionMethod::ENERGY_STABLE)) {
        du *= 0.9;
        dv *= 0.9;
    }

    // CFL-style safety cap for correction velocity magnitude.
    const double celerity = sqrt(max(g * h, 0.0));
    const double cap = max(1.0e-6, 3.0 * celerity);
    const double mag = sqrt(du * du + dv * dv);
    if (mag > cap) {
        const double s = cap / mag;
        du *= s;
        dv *= s;
    }

    du *= relax;
    dv *= relax;

    if (!std::isfinite(du) || !std::isfinite(dv)) {
        du = 0.0;
        dv = 0.0;
    }

    d_u_corr[c] = du;
    d_v_corr[c] = dv;
    d_hu[c] += h * du;
    d_hv[c] += h * dv;
}

static bool swe2d_gpu_apply_momentum_correction(
    SWE2DDeviceState* dev,
    double dt,
    double g,
    double h_min,
    const SWE2DNonhydroPcConfig& nh_cfg)
{
    if (!dev || dev->n_cells <= 0) return false;
    auto& ws = dev->nh_workspace;
    if (!ws.is_configured || !ws.d_p || !ws.d_u_corr || !ws.d_v_corr) return false;
    if (!dev->d_h || !dev->d_hu || !dev->d_hv || !dev->d_cell_inv_area) return false;
    if (!dev->d_cell_edge_offsets || !dev->d_cell_edge_ids) return false;

    constexpr int BLOCK = 256;
    const int grid = (dev->n_cells + BLOCK - 1) / BLOCK;
    const double relax = std::max(0.0, std::min(1.0, nh_cfg.relax));

    swe2d_gpu_momentum_corrector_kernel<<<grid, BLOCK, 0, dev->d_stream>>>(
        dev->n_cells,
        dt,
        g,
        h_min,
        relax,
        nh_cfg.velocity_correction,
        dev->d_h,
        ws.d_p,
        dev->d_cell_edge_offsets,
        dev->d_cell_edge_ids,
        dev->d_edge_c0,
        dev->d_edge_c1,
        dev->d_edge_nx,
        dev->d_edge_ny,
        dev->d_edge_len,
        dev->d_cell_inv_area,
        ws.d_u_corr,
        ws.d_v_corr,
        dev->d_hu,
        dev->d_hv);
    CUDA_CHECK(cudaGetLastError());
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

    // Phase 8B: Solve pressure correction with PCG.
    SWE2DNonhydroPcDiag pcg_diag{};
    bool pcg_ok = swe2d_gpu_solve_pressure_pcg(dev, nh_cfg, &pcg_diag);
    if (!pcg_ok) {
        throw std::runtime_error("swe2d_gpu_step_nonhydro_predictor_corrector: PCG pressure solve failed");
    }

    // Phase 8C: Apply pressure-gradient momentum correction.
    bool corr_ok = swe2d_gpu_apply_momentum_correction(dev, dt, g, h_min, nh_cfg);
    if (!corr_ok) {
        throw std::runtime_error("swe2d_gpu_step_nonhydro_predictor_corrector: momentum correction failed");
    }

    // Phase 6 Skeleton: Exchange with 3D (no-op for now)
    if (dev->coupling_iface && dev->patch3d) {
        swe2d_gpu_apply_2d3d_exchange_skeleton(
            dev,
            dt,
            g,
            static_cast<int>(SWE2DThreeDCouplingMode::TWO_WAY_2D_3D),
            true,
            diag);
    }

    if (diag) {
        *diag = pred_diag;  // Copy predictor diagnostics
        diag->gpu_active = true;
    }

    if (nh_diag) {
        nh_diag->pressure_iters = pcg_diag.pressure_iters;
        nh_diag->pressure_residual = pcg_diag.pressure_residual;
        nh_diag->corrector_applied = true;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3D patch state observation and initialisation
// ─────────────────────────────────────────────────────────────────────────────

SWE3DPatchStats swe2d_gpu_get_3d_patch_stats(SWE2DDeviceState* dev)
{
    if (!dev || !dev->patch3d)
        throw std::runtime_error("swe2d_gpu_get_3d_patch_stats: no 3D patch allocated");

    const auto* patch = dev->patch3d;
    const int64_t n = patch->n_cells;
    if (n <= 0)
        throw std::runtime_error("swe2d_gpu_get_3d_patch_stats: empty patch");

    const size_t sz = static_cast<size_t>(n) * sizeof(double);

    // Synchronise so the patch step kernel has finished before we read.
    if (dev->d_stream)
        CUDA_CHECK(cudaStreamSynchronize(dev->d_stream));

    std::vector<double> h_u(n), h_v(n), h_w(n), h_p(n), h_vof(n);
    CUDA_CHECK(cudaMemcpy(h_u.data(),   patch->d_u,   sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_v.data(),   patch->d_v,   sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_w.data(),   patch->d_w,   sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_p.data(),   patch->d_p,   sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vof.data(), patch->d_vof, sz, cudaMemcpyDeviceToHost));

    SWE3DPatchStats s;
    s.n_cells = n;
    s.nx = patch->desc.nx;
    s.ny = patch->desc.ny;
    s.nz = patch->desc.nz;
    s.dx = patch->desc.dx;
    s.dy = patch->desc.dy;
    s.dz = patch->desc.dz;

    s.vof_min = h_vof[0];
    s.vof_max = h_vof[0];
    s.vof_sum = 0.0;
    double su2 = 0.0, sv2 = 0.0, sw2 = 0.0;
    double sdiv2 = 0.0;
    s.p_max_abs = 0.0;

    for (int64_t i = 0; i < n; ++i) {
        const double vf = h_vof[i];
        if (vf < s.vof_min) s.vof_min = vf;
        if (vf > s.vof_max) s.vof_max = vf;
        s.vof_sum += vf;
        su2 += h_u[i] * h_u[i];
        sv2 += h_v[i] * h_v[i];
        sw2 += h_w[i] * h_w[i];
        const double pabs = std::abs(h_p[i]);
        if (pabs > s.p_max_abs) s.p_max_abs = pabs;

        const int32_t nx = s.nx;
        const int32_t ny = s.ny;
        const int32_t nz = s.nz;
        const int32_t plane = nx * ny;
        const int32_t iz = static_cast<int32_t>(i / plane);
        const int32_t rem = static_cast<int32_t>(i - static_cast<int64_t>(iz) * plane);
        const int32_t iy = rem / nx;
        const int32_t ix = rem - iy * nx;

        const int32_t ixm = (ix > 0) ? (ix - 1) : ix;
        const int32_t ixp = (ix + 1 < nx) ? (ix + 1) : ix;
        const int32_t iym = (iy > 0) ? (iy - 1) : iy;
        const int32_t iyp = (iy + 1 < ny) ? (iy + 1) : iy;
        const int32_t izm = (iz > 0) ? (iz - 1) : iz;
        const int32_t izp = (iz + 1 < nz) ? (iz + 1) : iz;

        const auto hidx = [nx, ny](int32_t x, int32_t y, int32_t z) -> int64_t {
            return static_cast<int64_t>(z) * static_cast<int64_t>(nx) * static_cast<int64_t>(ny) +
                   static_cast<int64_t>(y) * static_cast<int64_t>(nx) +
                   static_cast<int64_t>(x);
        };
        const int64_t id_xm = hidx(ixm, iy, iz);
        const int64_t id_xp = hidx(ixp, iy, iz);
        const int64_t id_ym = hidx(ix, iym, iz);
        const int64_t id_yp = hidx(ix, iyp, iz);
        const int64_t id_zm = hidx(ix, iy, izm);
        const int64_t id_zp = hidx(ix, iy, izp);

        const double du_dx = (h_u[id_xp] - h_u[id_xm]) / (2.0 * s.dx);
        const double dv_dy = (h_v[id_yp] - h_v[id_ym]) / (2.0 * s.dy);
        const double dw_dz = (h_w[id_zp] - h_w[id_zm]) / (2.0 * s.dz);
        const double div_u = du_dx + dv_dy + dw_dz;
        sdiv2 += div_u * div_u;
    }

    const double inv_n = 1.0 / static_cast<double>(n);
    s.u_rms = std::sqrt(su2 * inv_n);
    s.v_rms = std::sqrt(sv2 * inv_n);
    s.w_rms = std::sqrt(sw2 * inv_n);
    s.divergence_rms = std::sqrt(sdiv2 * inv_n);
    s.projection_iters = patch->last_projection_iters;
    s.projection_residual = patch->last_projection_residual;
    s.projection_converged = patch->last_projection_converged;
    s.vof_transport_substeps = patch->last_vof_substeps;
    return s;
}

void swe2d_gpu_set_3d_patch_vof(
    SWE2DDeviceState* dev,
    const double*     vof_host,
    int64_t           n)
{
    if (!dev || !dev->patch3d)
        throw std::runtime_error("swe2d_gpu_set_3d_patch_vof: no 3D patch allocated");
    if (!vof_host)
        throw std::invalid_argument("swe2d_gpu_set_3d_patch_vof: null input");
    if (n != dev->patch3d->n_cells)
        throw std::invalid_argument("swe2d_gpu_set_3d_patch_vof: length mismatch");

    CUDA_CHECK(cudaMemcpyAsync(
        dev->patch3d->d_vof,
        vof_host,
        static_cast<size_t>(n) * sizeof(double),
        cudaMemcpyHostToDevice,
        dev->d_stream));
}

void swe2d_gpu_get_3d_patch_vof(
    SWE2DDeviceState* dev,
    double*           vof_host,
    int64_t           n)
{
    if (!dev || !dev->patch3d)
        throw std::runtime_error("swe2d_gpu_get_3d_patch_vof: no 3D patch allocated");
    if (!vof_host)
        throw std::invalid_argument("swe2d_gpu_get_3d_patch_vof: null output buffer");
    if (n != dev->patch3d->n_cells)
        throw std::invalid_argument("swe2d_gpu_get_3d_patch_vof: length mismatch");

    CUDA_CHECK(cudaMemcpyAsync(
        vof_host,
        dev->patch3d->d_vof,
        static_cast<size_t>(n) * sizeof(double),
        cudaMemcpyDeviceToHost,
        dev->d_stream));
    CUDA_CHECK(cudaStreamSynchronize(dev->d_stream));
}

void swe2d_gpu_get_3d_patch_velocity(
    SWE2DDeviceState* dev,
    double*           u_host,
    double*           v_host,
    double*           w_host,
    int64_t           n)
{
    if (!dev || !dev->patch3d)
        throw std::runtime_error("swe2d_gpu_get_3d_patch_velocity: no 3D patch allocated");
    if (n != dev->patch3d->n_cells)
        throw std::invalid_argument("swe2d_gpu_get_3d_patch_velocity: length mismatch");

    const size_t sz = static_cast<size_t>(n) * sizeof(double);
    if (u_host) {
        CUDA_CHECK(cudaMemcpyAsync(
            u_host,
            dev->patch3d->d_u,
            sz,
            cudaMemcpyDeviceToHost,
            dev->d_stream));
    }
    if (v_host) {
        CUDA_CHECK(cudaMemcpyAsync(
            v_host,
            dev->patch3d->d_v,
            sz,
            cudaMemcpyDeviceToHost,
            dev->d_stream));
    }
    if (w_host) {
        CUDA_CHECK(cudaMemcpyAsync(
            w_host,
            dev->patch3d->d_w,
            sz,
            cudaMemcpyDeviceToHost,
            dev->d_stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(dev->d_stream));
}

void swe2d_gpu_get_3d_patch_pressure(
    SWE2DDeviceState* dev,
    double*           p_host,
    int64_t           n)
{
    if (!dev || !dev->patch3d)
        throw std::runtime_error("swe2d_gpu_get_3d_patch_pressure: no 3D patch allocated");
    if (!p_host)
        throw std::invalid_argument("swe2d_gpu_get_3d_patch_pressure: null output buffer");
    if (n != dev->patch3d->n_cells)
        throw std::invalid_argument("swe2d_gpu_get_3d_patch_pressure: length mismatch");

    CUDA_CHECK(cudaMemcpyAsync(
        p_host,
        dev->patch3d->d_p,
        static_cast<size_t>(n) * sizeof(double),
        cudaMemcpyDeviceToHost,
        dev->d_stream));
    CUDA_CHECK(cudaStreamSynchronize(dev->d_stream));
}

void swe2d_gpu_set_3d_patch_state(
    SWE2DDeviceState* dev,
    const double* u_host,
    const double* v_host,
    const double* w_host,
    const double* p_host,
    const double* vof_host,
    int64_t n)
{
    if (!dev || !dev->patch3d)
        throw std::runtime_error("swe2d_gpu_set_3d_patch_state: no 3D patch allocated");
    if (n != dev->patch3d->n_cells)
        throw std::invalid_argument("swe2d_gpu_set_3d_patch_state: length mismatch");

    const size_t sz = static_cast<size_t>(n) * sizeof(double);
    cudaStream_t s = dev->d_stream;

    auto upload = [&](double* dst, const double* src) {
        if (src) CUDA_CHECK(cudaMemcpyAsync(dst, src, sz, cudaMemcpyHostToDevice, s));
    };
    upload(dev->patch3d->d_u,   u_host);
    upload(dev->patch3d->d_v,   v_host);
    upload(dev->patch3d->d_w,   w_host);
    upload(dev->patch3d->d_p,   p_host);
    upload(dev->patch3d->d_vof, vof_host);
}

void swe2d_gpu_set_3d_patch_geometry(
    SWE2DDeviceState* dev,
    const double* phi_host,
    const double* ax_host,
    const double* ay_host,
    const double* az_host,
    int64_t n)
{
    if (!dev || !dev->patch3d)
        throw std::runtime_error("swe2d_gpu_set_3d_patch_geometry: no 3D patch allocated");
    if (n != dev->patch3d->n_cells)
        throw std::invalid_argument("swe2d_gpu_set_3d_patch_geometry: length mismatch");

    const size_t sz = static_cast<size_t>(n) * sizeof(double);
    cudaStream_t s = dev->d_stream;

    auto ensure_alloc = [&](double*& dst, const char* name) {
        if (dst) return;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dst), sz));
        if (!dst) {
            throw std::runtime_error(std::string("swe2d_gpu_set_3d_patch_geometry: allocation failed for ") + name);
        }
    };

    auto upload = [&](double*& dst, const double* src, const char* name) {
        if (!src) return;
        ensure_alloc(dst, name);
        CUDA_CHECK(cudaMemcpyAsync(dst, src, sz, cudaMemcpyHostToDevice, s));
    };
    upload(dev->patch3d->d_phi, phi_host, "d_phi");
    upload(dev->patch3d->d_ax,  ax_host, "d_ax");
    upload(dev->patch3d->d_ay,  ay_host, "d_ay");
    upload(dev->patch3d->d_az,  az_host, "d_az");
}

void swe2d_gpu_set_3d_patch_face_bc(
    SWE2DDeviceState* dev,
    int32_t face,
    int32_t mode,
    double u,
    double v,
    double w,
    double q,
    double vof,
    double p)
{
    if (!dev || !dev->patch3d)
        throw std::runtime_error("swe2d_gpu_set_3d_patch_face_bc: no 3D patch allocated");

    if (face < 0 || face >= SWE3D_PATCH_FACE_COUNT)
        throw std::invalid_argument("swe2d_gpu_set_3d_patch_face_bc: face index out of range");

    const int32_t mode_min = static_cast<int32_t>(SWE3DBoundaryMode::WALL);
    const int32_t mode_max = static_cast<int32_t>(SWE3DBoundaryMode::INFLOW_FLOW_RATE);
    const int32_t m = std::max(mode_min, std::min(mode_max, mode));

    auto& desc = dev->patch3d->desc;
    desc.bc_mode[face] = m;
    desc.bc_u[face] = u;
    desc.bc_v[face] = v;
    desc.bc_w[face] = w;
    desc.bc_q[face] = q;
    desc.bc_vof[face] = fmin(1.0, fmax(0.0, vof));
    desc.bc_p[face] = p;
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
    safe_free(dev->d_h1);         safe_free(dev->d_hu1);
    safe_free(dev->d_hv1);
    safe_free(dev->d_h2);         safe_free(dev->d_hu2);
    safe_free(dev->d_hv2);
    safe_free(dev->d_h3);         safe_free(dev->d_hu3);
    safe_free(dev->d_hv3);
    safe_free(dev->d_k4_h);       safe_free(dev->d_k4_hu);
    safe_free(dev->d_k4_hv);
    safe_free(dev->d_k5_h);       safe_free(dev->d_k5_hu);
    safe_free(dev->d_k5_hv);
    safe_free(dev->d_k6_h);       safe_free(dev->d_k6_hu);
    safe_free(dev->d_k6_hv);
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
        safe_free(ws.d_culvert_area_m2); safe_free(ws.d_culvert_barrels); safe_free(ws.d_culvert_slope);
        safe_free(ws.d_inlet_invert_elev); safe_free(ws.d_outlet_invert_elev);
        safe_free(ws.d_entrance_loss_k); safe_free(ws.d_exit_loss_k);
        safe_free(ws.d_embankment_enabled); safe_free(ws.d_embankment_crest_elev);
        safe_free(ws.d_embankment_overflow_width); safe_free(ws.d_embankment_weir_coeff);
        safe_free(ws.d_structure_flow);
    }
    if (dev->d_stream) {
        cudaStreamSynchronize(dev->d_stream);
        cudaStreamDestroy(dev->d_stream);
        dev->d_stream = nullptr;
        // Clean up CUDA graph cache
        dev->kernel_graph_cache.destroy();
    }
    delete dev;
}
