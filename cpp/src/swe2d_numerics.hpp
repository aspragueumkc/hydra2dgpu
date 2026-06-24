#pragma once
// swe2d_numerics.hpp
// Core numerical kernels for the 2D SWE solver:
//   - HLLC Riemann solver (positivity-preserving)
//   - Well-balanced hydrostatic bed-slope reconstruction
//   - Manning friction source (semi-implicit)
//   - CFL timestep estimate per edge
//
// All functions are constexpr/inline-friendly and callable from both
// CPU (swe2d_solver.cpp) and CUDA device code (swe2d_gpu.cu via include).

#include <cmath>
#include <algorithm>

// Allow the same header to be included in CUDA device code
#ifdef __CUDACC__
#  define SWE2D_HOSTDEV __host__ __device__
#else
#  define SWE2D_HOSTDEV
#endif

namespace swe2d {

// ─────────────────────────────────────────────────────────────────────────────
// Physical constants
// ─────────────────────────────────────────────────────────────────────────────
static constexpr double G_DEFAULT   = 9.81;    // gravitational acceleration (m/s²)
static constexpr double H_MIN       = 1.0e-6;  // thin-film wet/dry threshold (m)
static constexpr double CFL_DEFAULT = 0.45;    // CFL safety factor (explicit, triangular)

// ─────────────────────────────────────────────────────────────────────────────
// State helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute x-velocity from momentum and depth.  Zero below h_min.
SWE2D_HOSTDEV inline double vel_u(double hu, double h, double h_min) {
    return (h > h_min) ? (hu / h) : 0.0;
}

/// Compute y-velocity from momentum and depth.
SWE2D_HOSTDEV inline double vel_v(double hv, double h, double h_min) {
    return (h > h_min) ? (hv / h) : 0.0;
}

/// Wave speed (celerity) = sqrt(g*h).  Zero for dry cells.
SWE2D_HOSTDEV inline double celerity(double h, double g) {
    return (h > 0.0) ? std::sqrt(g * h) : 0.0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Well-balanced hydrostatic reconstruction
// Decomposes the left/right interface states into starred depths that
// preserve lake-at-rest to machine precision.
// ─────────────────────────────────────────────────────────────────────────────
struct ReconstructedStates {
    double hL_star, uL, vL;   // left interface depth + velocity
    double hR_star, uR, vR;   // right interface depth + velocity
    double zb_face;            // maximum bed elevation at the interface
};

/** Well-balanced hydrostatic reconstruction of interface states.
    Decomposes left/right cell states into starred depths that preserve
    lake-at-rest to machine precision. @param hL Left depth
    @param huL Left x-momentum @param hvL Left y-momentum @param zbL Left bed
    @param hR Right depth @param huR Right x-momentum @param hvR Right y-momentum
    @param zbR Right bed @param h_min Wet/dry threshold
    @returns ReconstructedStates with starred depths, velocities, and face bed elev. */
SWE2D_HOSTDEV inline ReconstructedStates hydrostatic_reconstruct(
    double hL,  double huL, double hvL, double zbL,
    double hR,  double huR, double hvR, double zbR,
    double h_min)
{
    ReconstructedStates rs;

    // Water-surface elevations
    double etaL = hL + zbL;
    double etaR = hR + zbR;

    // Interface bed elevation: take the maximum to preserve positivity
    rs.zb_face = (zbL > zbR) ? zbL : zbR;

    // Starred depths: max(0, eta - zb_face)
    rs.hL_star = (etaL > rs.zb_face) ? (etaL - rs.zb_face) : 0.0;
    rs.hR_star = (etaR > rs.zb_face) ? (etaR - rs.zb_face) : 0.0;

    // Velocities from original depths (not starred) to keep momentum consistent
    rs.uL = vel_u(huL, hL, h_min);
    rs.vL = vel_v(hvL, hL, h_min);
    rs.uR = vel_u(huR, hR, h_min);
    rs.vR = vel_v(hvR, hR, h_min);

    return rs;
}

// ─────────────────────────────────────────────────────────────────────────────
// HLLC Riemann solver
// Computes normal fluxes (F_h, F_hu, F_hv) for the SWE system across an edge.
//
// Convention: normal (nx, ny) points from the left cell (c0) outward.
// Returned fluxes are positive when mass/momentum flows from left to right.
//
// References:
//   Toro (2001) "Shock Capturing Methods for Free-Surface Shallow Flows"
//   Einfeldt (1988) wave speed estimates for positivity preservation.
// ─────────────────────────────────────────────────────────────────────────────
struct HLLCFlux {
    double fh;   // mass flux
    double fhu;  // x-momentum flux
    double fhv;  // y-momentum flux
};

/** HLLC Riemann solver for the 2D shallow-water equations.
    Computes normal fluxes (mass, x-momentum, y-momentum) across an edge
    using the HLLC approximate Riemann solver with Einfeldt wave-speed bounds.
    Convention: normal (nx, ny) points from left cell outward. @param hL Left depth
    @param uL Left x-velocity @param vL Left y-velocity @param hR Right depth
    @param uR Right x-velocity @param vR Right y-velocity @param nx Edge normal x
    @param ny Edge normal y @param g Gravity @param h_min Wet/dry threshold
    @returns HLLCFlux containing mass and momentum fluxes. */
SWE2D_HOSTDEV inline HLLCFlux hllc_flux(
    double hL,  double uL, double vL,
    double hR,  double uR, double vR,
    double nx, double ny,
    double g,   double h_min)
{
    HLLCFlux flux{0.0, 0.0, 0.0};

    // Both sides dry → zero flux
    if (hL <= h_min && hR <= h_min) {
        return flux;
    }

    // Normal velocity components
    double unL = uL * nx + vL * ny;
    double unR = uR * nx + vR * ny;

    // Celerities
    double cL = celerity(hL, g);
    double cR = celerity(hR, g);

    // Roe-averaged estimates for HLLC contact speed
    double sqrt_hL = (hL > 0.0) ? std::sqrt(hL) : 0.0;
    double sqrt_hR = (hR > 0.0) ? std::sqrt(hR) : 0.0;
    double denom   = sqrt_hL + sqrt_hR;

    double u_roe = (denom > 0.0)
                   ? (sqrt_hL * unL + sqrt_hR * unR) / denom
                   : 0.0;
    double c_roe = (denom > 0.0)
                   ? std::sqrt(0.5 * g * (hL + hR))
                   : 0.0;

    // Einfeldt wave-speed bounds (Einfeldt 1988)
    double SL = std::min(unL - cL, u_roe - c_roe);
    double SR = std::max(unR + cR, u_roe + c_roe);

    // Supercritical Fr>1: pure upwind
    if (SL >= 0.0) {
        // Left state flux
        flux.fh  = hL * unL;
        flux.fhu = hL * uL * unL + 0.5 * g * hL * hL * nx;
        flux.fhv = hL * vL * unL + 0.5 * g * hL * hL * ny;
        return flux;
    }
    if (SR <= 0.0) {
        // Right state flux
        flux.fh  = hR * unR;
        flux.fhu = hR * uR * unR + 0.5 * g * hR * hR * nx;
        flux.fhv = hR * vR * unR + 0.5 * g * hR * hR * ny;
        return flux;
    }

    // HLLC contact speed (S_star)
    // S* = (hR*unR*(SR-unR) - hL*unL*(SL-unL) + 0.5g*(hL²-hR²)) /
    //      (hR*(SR-unR) - hL*(SL-unL))
    double numS  = hR * unR * (SR - unR) - hL * unL * (SL - unL)
                   + 0.5 * g * (hL * hL - hR * hR);
    double denS  = hR * (SR - unR) - hL * (SL - unL);
    double S_star = (std::abs(denS) > 1.0e-15) ? (numS / denS) : 0.0;

    // Left and right physical fluxes
    double fhL  = hL * unL;
    double fhuL = hL * uL * unL + 0.5 * g * hL * hL * nx;
    double fhvL = hL * vL * unL + 0.5 * g * hL * hL * ny;

    double fhR  = hR * unR;
    double fhuR = hR * uR * unR + 0.5 * g * hR * hR * nx;
    double fhvR = hR * vR * unR + 0.5 * g * hR * hR * ny;

    if (S_star >= 0.0) {
        // Left HLLC state
        double coeff = hL * (SL - unL) / (SL - S_star);
        double h_star_L  = coeff;
        double hu_star_L = coeff * (uL + (S_star - unL) * nx);
        double hv_star_L = coeff * (vL + (S_star - unL) * ny);

        double dh  = h_star_L  - hL;
        double dhu = hu_star_L - hL * uL;
        double dhv = hv_star_L - hL * vL;

        flux.fh  = fhL  + SL * dh;
        flux.fhu = fhuL + SL * dhu;
        flux.fhv = fhvL + SL * dhv;
    } else {
        // Right HLLC state
        double coeff = hR * (SR - unR) / (SR - S_star);
        double h_star_R  = coeff;
        double hu_star_R = coeff * (uR + (S_star - unR) * nx);
        double hv_star_R = coeff * (vR + (S_star - unR) * ny);

        double dh  = h_star_R  - hR;
        double dhu = hu_star_R - hR * uR;
        double dhv = hv_star_R - hR * vR;

        flux.fh  = fhR  + SR * dh;
        flux.fhu = fhuR + SR * dhu;
        flux.fhv = fhvR + SR * dhv;
    }

    return flux;
}

// ─────────────────────────────────────────────────────────────────────────────
// Combined: reconstruct + HLLC flux for a single edge
// Returns the HLLC flux and writes zb_face for the bed-slope correction.
// ─────────────────────────────────────────────────────────────────────────────
/** Combined hydrostatic reconstruction + HLLC flux for a single edge.
    Returns the HLLC flux and outputs the face bed elevation for the
    bed-slope correction. @param zb_face_out [out] Max bed elev at interface */
SWE2D_HOSTDEV inline HLLCFlux edge_flux(
    double hL,  double huL, double hvL, double zbL,
    double hR,  double huR, double hvR, double zbR,
    double nx, double ny,
    double g,   double h_min,
    double& zb_face_out)   // output: max(zbL, zbR)
{
    ReconstructedStates rs = hydrostatic_reconstruct(
        hL, huL, hvL, zbL,
        hR, huR, hvR, zbR,
        h_min);

    zb_face_out = rs.zb_face;

    return hllc_flux(
        rs.hL_star, rs.uL, rs.vL,
        rs.hR_star, rs.uR, rs.vR,
        nx, ny, g, h_min);
}

// ─────────────────────────────────────────────────────────────────────────────
// Well-balanced bed-slope pressure correction
// Adds the hydrostatic correction to the momentum flux accumulators for cell c0.
// This correction cancels with the pressure term in the flux to produce the
// exact lake-at-rest solution.
//
// correction_hu += -0.5 * g * (hL_star² - hL²) * nx
// correction_hv += -0.5 * g * (hL_star² - hL²) * ny
// ─────────────────────────────────────────────────────────────────────────────
/** Well-balanced bed-slope pressure correction for momentum.
    Adds hydrostatic correction to flux accumulators to cancel the pressure
    term and preserve lake-at-rest. @param hL Original depth @param hL_star
    Starred depth from hydrostatic reconstruction @param nx Edge normal x
    @param ny Edge normal y @param g Gravity @param corr_hu [in/out] hu correction
    @param corr_hv [in/out] hv correction */
SWE2D_HOSTDEV inline void bed_slope_correction(
    double hL, double hL_star,
    double nx, double ny, double g,
    double& corr_hu, double& corr_hv)
{
    double dp = 0.5 * g * (hL_star * hL_star - hL * hL);
    corr_hu -= dp * nx;
    corr_hv -= dp * ny;
}

// ─────────────────────────────────────────────────────────────────────────────
// Manning friction source — semi-implicit limiter
// Applied after flux update to prevent velocity reversal in shallow cells.
//
// The semi-implicit form solves:
//   u^{n+1} = u^n / (1 + dt * Cf * |u^n| / h)
//
// where Cf = g * n^2 / h^(4/3) is the linearised friction coefficient.
//
// Optional shallow-flow correction enhances Cf when the log boundary layer
// fills a significant fraction of the water column (Keulegan-based model):
//   Cf *= (h_ref / h_fric)^exponent   if h_fric < h_ref
//   h_ref = depth_alpha * n^(3/2)
// ─────────────────────────────────────────────────────────────────────────────
/** Semi-implicit Manning friction source term.
    Applies Manning friction after flux update to prevent velocity reversal
    in shallow cells. Uses semi-implicit form: u^{n+1}=u^n/(1+dt*Cf*|u|/h).
    Optional shallow-flow correction enhances Cf when the log boundary layer
    fills a significant fraction of the water column. @param h [in/out] Depth
    @param hu [in/out] x-momentum @param hv [in/out] y-momentum @param dt Timestep
    @param n_mann Manning's n @param g Gravity @param h_min Wet/dry threshold
    @param k_mann Manning unit factor @param shallow_correction Enable Cf enhancement
    @param depth_alpha Reference depth coefficient @param exponent Cf enhancement exponent */
SWE2D_HOSTDEV inline void apply_friction(
    double& h, double& hu, double& hv,
    double dt, double n_mann, double g, double h_min, double k_mann = 1.0,
    bool   shallow_correction = false,
    double depth_alpha = 5.0,
    double exponent = 0.4)
{
    if (h <= h_min) {
        hu = hv = 0.0;
        return;
    }
    double u    = hu / h;
    double v    = hv / h;
    double spd  = std::sqrt(u * u + v * v);
    // Regularize shallow-cell friction stiffness to avoid large Cf spikes
    // right above h_min at advancing wet/dry fronts.
    const double h_fric = std::max(h, 4.0 * h_min);
    // Cf = g * n² / (k² * h^(4/3))   where k = 1.0 (SI) or 1.486 (USC)
    double k2   = k_mann * k_mann;
    double h43  = std::pow(h_fric, 4.0 / 3.0);
    double Cf   = (h43 > 0.0) ? (g * n_mann * n_mann / (k2 * h43)) : 0.0;

    // Shallow-flow depth correction: enhance Cf when h < h_ref.
    if (shallow_correction && Cf > 0.0) {
        const double h_ref = depth_alpha * std::pow(n_mann, 1.5);
        if (h_fric < h_ref) {
            Cf *= std::pow(h_ref / h_fric, exponent);
        }
    }

    double denom = 1.0 + dt * Cf * spd;
    hu /= denom;
    hv /= denom;
}

/// Compute the number of friction sub-steps for adaptive temporal accuracy.
/// Returns 1 when sub-stepping is disabled or the friction Courant number
/// is below the target threshold.
/** @param h Depth @param hu x-momentum @param hv y-momentum @param dt Timestep
    @param n_mann Manning's n @param g Gravity @param h_min Wet/dry threshold
    @param k_mann Manning unit factor @param substep_enabled Enable sub-stepping
    @param target_courant Target friction Courant number
    @param max_substeps Maximum allowed substeps @returns Number of substeps */
SWE2D_HOSTDEV inline int friction_substep_count(
    double h, double hu, double hv,
    double dt, double n_mann, double g, double h_min, double k_mann,
    bool substep_enabled, double target_courant, int max_substeps)
{
    if (!substep_enabled || target_courant <= 0.0) return 1;
    if (h <= h_min) return 1;
    const double inv_h = 1.0 / h;
    const double u = hu * inv_h;
    const double v = hv * inv_h;
    const double spd = std::sqrt(u*u + v*v);
    const double k2 = k_mann * k_mann;
    const double h_fric = std::max(h, 4.0 * h_min);
    const double h43 = std::pow(h_fric, 4.0 / 3.0);
    const double Cf = (h43 > 0.0) ? (g * n_mann * n_mann / (k2 * h43)) : 0.0;
    const double nu_fric = dt * Cf * spd;
    const int n_sub = static_cast<int>(std::ceil(nu_fric / target_courant));
    return std::max(1, std::min(n_sub, max_substeps));
}

/** Apply Manning friction with adaptive sub-stepping for temporal-order hardening.
    Subdivides dt into N substeps when the friction Courant number exceeds
    the target threshold. */
/** @param h [in/out] Depth @param hu [in/out] x-momentum @param hv [in/out] y-momentum
    @param dt Timestep @param n_mann Manning's n @param g Gravity @param h_min Wet/dry
    @param k_mann Manning unit factor @param substep_enabled Enable sub-stepping
    @param target_courant Target friction Courant @param max_substeps Cap on substeps
    @param shallow_correction Enable Cf enhancement @param depth_alpha Ref depth coeff
    @param exponent Cf enhancement exponent */
SWE2D_HOSTDEV inline void apply_friction_substepped(
    double& h, double& hu, double& hv,
    double dt, double n_mann, double g, double h_min, double k_mann,
    bool substep_enabled, double target_courant, int max_substeps,
    bool shallow_correction = false,
    double depth_alpha = 5.0,
    double exponent = 0.4)
{
    if (h <= h_min) {
        hu = hv = 0.0;
        return;
    }

    const double k2 = k_mann * k_mann;
    const double h_fric = std::max(h, 4.0 * h_min);
    const double h43 = std::pow(h_fric, 4.0 / 3.0);
    double Cf = (h43 > 0.0) ? (g * n_mann * n_mann / (k2 * h43)) : 0.0;

    // Shallow-flow depth correction.
    if (shallow_correction && Cf > 0.0) {
        const double h_ref = depth_alpha * std::pow(n_mann, 1.5);
        if (h_fric < h_ref) {
            Cf *= std::pow(h_ref / h_fric, exponent);
        }
    }

    // Determine sub-step count.
    const double u0 = hu / h;
    const double v0 = hv / h;
    const double spd0 = std::sqrt(u0*u0 + v0*v0);
    const double nu_fric = dt * Cf * spd0;
    int n_sub = 1;
    if (substep_enabled && target_courant > 0.0) {
        n_sub = static_cast<int>(std::ceil(nu_fric / target_courant));
        n_sub = std::max(1, std::min(n_sub, max_substeps));
    }

    const double dt_sub = dt / static_cast<double>(n_sub);
    for (int k = 0; k < n_sub; ++k) {
        const double u_k = hu / h;
        const double v_k = hv / h;
        const double spd_k = std::sqrt(u_k*u_k + v_k*v_k);
        const double denom = 1.0 + dt_sub * Cf * spd_k;
        hu /= denom;
        hv /= denom;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CFL characteristic speed for a single edge
// Returns the maximum characteristic speed / effective cell size (s⁻¹).
// The global dt is dt = CFL_factor / max_over_all_edges(lambda).
// ─────────────────────────────────────────────────────────────────────────────
/** CFL characteristic speed per edge.
    Returns the maximum characteristic speed / effective cell size (s^{-1}).
    Global dt = CFL / max(lambda). @param hL Left depth @param huL Left x-momentum
    @param hvL Left y-momentum @param hR Right depth @param huR Right x-momentum
    @param hvR Right y-momentum @param nx Edge normal x @param ny Edge normal y
    @param edge_len Edge length @param cell_area_L Left cell area @param cell_area_R Right cell area
    @param g Gravity @param h_min Wet/dry threshold @returns Wave speed ratio (s^{-1}) */
SWE2D_HOSTDEV inline double edge_cfl_lambda(
    double hL, double huL, double hvL,
    double hR, double huR, double hvR,
    double nx, double ny, double edge_len,
    double cell_area_L, double cell_area_R,
    double g, double h_min)
{
    double uL  = vel_u(huL, hL, h_min);
    double vL  = vel_v(hvL, hL, h_min);
    double uR  = vel_u(huR, hR, h_min);
    double vR  = vel_v(hvR, hR, h_min);
    double cL  = celerity(hL, g);
    double cR  = celerity(hR, g);

    double unL = uL * nx + vL * ny;
    double unR = uR * nx + vR * ny;

    double max_wave = std::max(std::abs(unL) + cL, std::abs(unR) + cR);

    // Effective cell size proxy: min(area_L, area_R) / edge_len
    double area_eff = (cell_area_R > 0.0)
                      ? std::min(cell_area_L, cell_area_R)
                      : cell_area_L;
    double dx_eff = (edge_len > 0.0 && area_eff > 0.0)
                    ? (area_eff / edge_len)
                    : 1.0;

    return (dx_eff > 0.0) ? (max_wave / dx_eff) : 0.0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Ghost cell states for boundary edges
// Used by both CPU and GPU flux loops to handle BCs without branching on cell type.
// ─────────────────────────────────────────────────────────────────────────────
struct GhostState {
    double h, hu, hv, zb;
};

// BCType values passed as int to stay CUDA-compatible without enum class overhead
/** Construct a ghost cell state for boundary edges.
    Used by both CPU and GPU flux loops to handle BCs without branching on
    cell type. Supports WALL, INFLOW_Q, STAGE, OPEN, REFLECT, NORMAL_DEPTH,
    and NORMAL_DEPTH_SLOPE boundary types. @param hI Interior depth @param huI
    Interior x-momentum @param hvI Interior y-momentum @param zbI Interior bed
    @param nx Outward normal x @param ny Outward normal y @param bc_type
    Boundary condition type (1-7) @param bc_val Prescribed BC value
    @param h_min Wet/dry threshold @param n_mann Manning's n (for normal depth)
    @returns GhostState with ghost cell depth, momentum, and bed elevation. */
SWE2D_HOSTDEV inline GhostState make_ghost(
    double hI,  double huI, double hvI, double zbI,  // interior (c0) state
    double nx,  double ny,                            // outward normal
    int    bc_type,
    double bc_val,
    double h_min,
    double n_mann)
{
    GhostState g{};
    g.zb = zbI;  // ghost bed elevation = interior bed elevation

    switch (bc_type) {
        case 1: // WALL: reflect normal velocity
            g.h  = hI;
            {
                double un = huI * nx + hvI * ny;
                g.hu = huI - 2.0 * un * nx;
                g.hv = hvI - 2.0 * un * ny;
            }
            break;
        case 2: // INFLOW_Q: inward unit discharge q_in (m^2/s), positive = into domain.
            // The outward normal (nx, ny) points AWAY from the domain, so the ghost
            // momentum points inward: -bc_val*nx, -bc_val*ny.
            // For a left boundary (nx=-1): -bc_val*(-1) = +bc_val (rightward = into domain).
            g.h  = hI;
            g.hu = -bc_val * nx;
            g.hv = -bc_val * ny;
            break;
        case 3: // STAGE: prescribed WSE
            {
                double h_ghost = bc_val - zbI;
                g.h  = (h_ghost > h_min) ? h_ghost : h_min;
                g.hu = huI;
                g.hv = hvI;
            }
            break;
        case 4: // OPEN: zero-gradient outflow
            g.h  = hI;
            g.hu = huI;
            g.hv = hvI;
            break;
        case 5: // REFLECT: same as WALL
            g.h  = hI;
            {
                double un = huI * nx + hvI * ny;
                g.hu = huI - 2.0 * un * nx;
                g.hv = hvI - 2.0 * un * ny;
            }
            break;
        case 6: // NORMAL_DEPTH: prescribed depth h
            g.h  = (bc_val > h_min) ? bc_val : h_min;
            g.hu = huI;
            g.hv = hvI;
            break;
        case 7: { // NORMAL_DEPTH_SLOPE: friction-slope normal depth (bc_val = Sf)
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
        default: // INTERIOR or unknown — should not appear in boundary loop
            g.h  = hI;
            g.hu = huI;
            g.hv = hvI;
            break;
    }
    return g;
}

} // namespace swe2d
