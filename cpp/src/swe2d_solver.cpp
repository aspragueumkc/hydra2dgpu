// swe2d_solver.cpp
// GPU solver implementation for the 2D SWE on an unstructured mesh.
// GPU/CUDA only

#include "swe2d_solver.hpp"
#include "swe2d_numerics.hpp"

#ifdef HYDRA_HAS_CUDA
#  include "swe2d_gpu.cuh"
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

/// Check if an environment variable is enabled (non-zero, non-false, non-no).
bool swe2d_env_enabled(const char* name) {
    const char* v = std::getenv(name);
    if (!v || !v[0]) return false;
    const char c0 = static_cast<char>(std::tolower(static_cast<unsigned char>(v[0])));
    return !(c0 == '0' || c0 == 'f' || c0 == 'n');
}



// ─────────────────────────────────────────────────────────────────────────────
/** Allocate and initialise a solver instance. */
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
    if (cfg.temporal_order < 1 || cfg.temporal_order > 6) {
        throw std::invalid_argument(
            "swe2d_create: temporal_order must be in {1, 2, 3, 4, 5, 6}");
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

    // Apply cell renumbering permutation to solver state arrays.
    const auto& perm = mesh.cell_perm;
    if (perm.size() == static_cast<size_t>(n)) {
        std::vector<double> h_tmp(s->h);
        std::vector<double> hu_tmp(s->hu);
        std::vector<double> hv_tmp(s->hv);
        std::vector<double> n_tmp(s->n_mann_cell);
        for (int32_t c = 0; c < n; ++c) {
            size_t src = static_cast<size_t>(perm[static_cast<size_t>(c)]);
            s->h[static_cast<size_t>(c)] = h_tmp[src];
            s->hu[static_cast<size_t>(c)] = hu_tmp[src];
            s->hv[static_cast<size_t>(c)] = hv_tmp[src];
            s->n_mann_cell[static_cast<size_t>(c)] = n_tmp[src];
        }
    }

    s->external_source_terms.assign(n, 0.0);

    // Initialise GPU state if requested and available
#ifdef HYDRA_HAS_CUDA
    s->dev = nullptr;
    if (cfg.use_gpu && swe2d_gpu_available()) {
        s->dev = swe2d_gpu_init(mesh,
                                s->h.data(), s->hu.data(), s->hv.data(),
                                s->n_mann_cell.data(),
                                cfg.degen_mode,
                                cfg.max_inv_area,
                                cfg.open_bc_relaxation);
        if (s->dev) {
            const bool enable_cuda_graphs = swe2d_env_enabled("BACKWATER_ENABLE_CUDA_GRAPHS");
            swe2d_gpu_enable_kernel_graphs(s->dev, enable_cuda_graphs);

            // Set Manning unit-conversion factor for GPU constant memory.
            swe2d_gpu_set_k_mann(cfg.k_mann);

            // Set friction temporal-order hardening and shallow-correction
            // params in GPU constant memory.
            swe2d_gpu_set_friction_config(
                cfg.friction_substep_enabled,
                cfg.friction_target_courant,
                cfg.friction_max_substeps,
                cfg.shallow_friction_correction,
                cfg.shallow_friction_depth_alpha,
                cfg.shallow_friction_exponent);


        }
    }
#endif

    return s;
}

// ─────────────────────────────────────────────────────────────────────────────
/// Free solver resources, including GPU device memory if allocated.
void swe2d_destroy(SWE2DSolver* s) {
    if (!s) return;
#ifdef HYDRA_HAS_CUDA
    if (s->dev) {
        swe2d_gpu_destroy(s->dev);
        s->dev = nullptr;
    }
#endif
    delete s;
}

// ─────────────────────────────────────────────────────────────────────────────
/// Copy solver state to caller-supplied arrays (GPU or host path).
void swe2d_get_state(const SWE2DSolver* s, double* h_out, double* hu_out, double* hv_out) {
    if (!s) return;
#ifdef HYDRA_HAS_CUDA
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

void swe2d_get_max_tracking(const SWE2DSolver* s, double* h_max_out, double* hu_max_out, double* hv_max_out) {
    if (!s) return;
#ifdef HYDRA_HAS_CUDA
    if (s->dev) {
        swe2d_gpu_readback_max_tracking(s->dev, h_max_out, hu_max_out, hv_max_out);
        return;
    }
#endif
    throw std::runtime_error("swe2d_get_max_tracking requires a GPU device");
}

/** Overwrite solver state from caller-supplied arrays.
    Also syncs to GPU device if CUDA is active. */
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

#ifdef HYDRA_HAS_CUDA
    if (s->dev) {
        swe2d_gpu_set_state(s->dev, s->h.data(), s->hu.data(), s->hv.data());
    }
#endif
}





// ─────────────────────────────────────────────────────────────────────────────
/** Main timestep dispatch (GPU-only).
    Handles dt_initial override, dt_fixed, CFL-adaptive dt, RK2 temporal
    scheme, tiny-N mode selection, and persistent chunking. @param s Solver handle
    @param dt_request Desired timestep (s); <=0 for CFL-controlled
    @returns Step diagnostics */
SWE2DStepDiag swe2d_step(SWE2DSolver* s, double dt_request) {
    if (!s) throw std::invalid_argument("swe2d_step: null solver");
    const double t_now = s->t;

    constexpr int32_t kTinyModeOff = 0;
    constexpr int32_t kTinyModeAuto = 1;
    constexpr int32_t kTinyModeFused = 2;
    // kTinyModePersistent (3) was removed; map anything >= 3 to off.

    const int32_t tiny_requested_raw = s->cfg.tiny_mode;
    bool tiny_path_eligible = (s->dev != nullptr);
    const int32_t active_cells_est =
        (s->last_wet_cells > 0) ? std::min(s->last_wet_cells, s->mesh->n_cells) : s->mesh->n_cells;
    int32_t active_edges_est = s->mesh->n_edges;
    if (s->mesh->n_cells > 0 && active_cells_est < s->mesh->n_cells) {
        const double frac = static_cast<double>(active_cells_est) / static_cast<double>(s->mesh->n_cells);
        active_edges_est = static_cast<int32_t>(std::max(1.0, std::round(frac * static_cast<double>(s->mesh->n_edges))));
    }
    const bool tiny_geom_by_total =
        (s->mesh->n_cells <= std::max(1, s->cfg.tiny_cell_threshold)) &&
        (s->mesh->n_edges <= std::max(1, s->cfg.tiny_edge_threshold));
    const bool tiny_geom_by_wet =
        (active_cells_est <= std::max(1, s->cfg.tiny_wet_cell_threshold));
    const bool tiny_fused_path_eligible = tiny_path_eligible;
    int32_t tiny_selected = kTinyModeOff;
    if (tiny_requested_raw == kTinyModeAuto) {
        if (tiny_fused_path_eligible && (tiny_geom_by_total || tiny_geom_by_wet)) {
            tiny_selected = kTinyModeFused;
        }
    } else if (tiny_requested_raw == kTinyModeFused) {
        tiny_selected = kTinyModeFused;
    }
    int32_t tiny_effective = tiny_selected;
    if (tiny_selected == kTinyModeFused) {
        // Fused mode is currently supported for the single-stage GPU step path
        // (Euler / non-RK graph variants). Multi-stage RK and advanced modes
        // keep explicit fallback until dedicated tiny kernels are landed.
        const bool fused_supported_now =
            tiny_fused_path_eligible &&
            s->cfg.temporal_order == 1;
        if (!fused_supported_now) {
            tiny_effective = kTinyModeOff;
        }
    }
    const bool tiny_mode_unsupported =
        (tiny_requested_raw != kTinyModeOff && tiny_requested_raw != kTinyModeAuto && tiny_requested_raw != kTinyModeFused);
    const bool tiny_fallback = (tiny_selected != tiny_effective) || tiny_mode_unsupported;

    auto finalize_diag = [&](SWE2DStepDiag& diag) {
        if (diag.wet_cells >= 0) {
            s->last_wet_cells = diag.wet_cells;
        }
        if (tiny_fallback) {
            s->tiny_mode_fallback_count += 1;
        }
        if (tiny_effective == kTinyModeFused) {
            s->fused_path_steps += 1;
        }
        diag.tiny_mode_requested = tiny_requested_raw;
        diag.tiny_mode_selected = tiny_selected;
        diag.tiny_mode_effective = tiny_effective;
        diag.tiny_mode_fallback = tiny_fallback;
        diag.tiny_active_cells_est = active_cells_est;
        diag.tiny_active_edges_est = active_edges_est;
        diag.tiny_mode_fallback_count_total = static_cast<int64_t>(s->tiny_mode_fallback_count);
        diag.fused_path_steps_total = static_cast<int64_t>(s->fused_path_steps);
        diag.persistent_path_steps_total = static_cast<int64_t>(s->persistent_path_steps);
    };

    const int diag_interval = s->cfg.gpu_diag_sync_interval_steps;
    const bool sync_diag_this_step =
        (diag_interval > 0) ? ((s->gpu_steps % static_cast<uint64_t>(diag_interval)) == 0u) : false;

#ifdef HYDRA_HAS_CUDA
    if (s->dev) {
        double dt;
        // Initial dt override: use dt_initial for the first step only.
        // This is critical for cold-start stability with CFL adaptive stepping,
        // where lambda_max=0 on a dry domain causes compute_cfl_dt() to return dt_max.
        if (s->cfg.dt_initial > 0.0 && !s->first_step_done) {
            dt = s->cfg.dt_initial;
            s->first_step_done = true;
        } else if (s->cfg.dt_fixed > 0.0) {
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

        SWE2DStepDiag diag;
        if (s->cfg.temporal_order == 1) {
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
        } else if (s->cfg.temporal_order == 2) {
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

        } else if (s->cfg.temporal_order == 3) {
            swe2d_gpu_step_rk3(s->dev, t_now, dt,
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
        } else if (s->cfg.temporal_order == 4 || s->cfg.temporal_order == 5) {
            // order=4 -> classic RK4, order=5 -> graph-safe RK4 (same algorithm).
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
        } else if (s->cfg.temporal_order == 6) {
            // order=6 -> graph-safe Cash-Karp RK5(4).
            swe2d_gpu_step_rk5(s->dev, t_now, dt,
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
            throw std::invalid_argument(
                "swe2d_step: temporal_order must be in {1, 2, 3, 4, 5, 6}");
        }
        diag.gpu_active = true;
        finalize_diag(diag);
        s->t += dt;
        s->gpu_steps += 1;
        return diag;
    }
#endif

    throw std::runtime_error(
        "swe2d_step requires a GPU device; CPU path is not available in GPU-only builds");
}

/// Update boundary condition type/value for selected edges and sync GPU.
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
#ifdef HYDRA_HAS_CUDA
    if (s->dev) {
        swe2d_gpu_update_boundary_values(s->dev, edge_index, bc_type, bc_val, n_updates);
    }
#endif
}

/** Configure per-edge boundary hydrograph timeseries and upload to GPU. */
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
#ifdef HYDRA_HAS_CUDA
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
#ifdef HYDRA_HAS_CUDA
    if (s->dev) {
        swe2d_gpu_set_boundary_hydrographs(s->dev, edge_index, bc_type, offsets, time_s, value, n_edges, n_samples);
    }
#endif
}

/// Upload progressive BC group data for on-device Q->q distribution.
void swe2d_solver_set_progressive_bc_data(
    SWE2DSolver* s,
    int32_t n_groups,
    int32_t n_edges_total,
    const int32_t* group_offsets,
    const int32_t* edge_hg_idx,
    const double* edge_len,
    const double* edge_cum_len,
    const double* group_peak_q,
    const double* group_total_len)
{
    if (!s) return;
#ifdef HYDRA_HAS_CUDA
    if (s->dev) {
        swe2d_gpu_set_progressive_bc_data(s->dev, n_groups, n_edges_total, group_offsets,
                                          edge_hg_idx, edge_len, edge_cum_len,
                                          group_peak_q, group_total_len);
    }
#endif
}

/** Configure per-cell rain+CN infiltration forcing and upload to GPU. */
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
    double mm_to_model_depth,
    double rain_update_interval_s)
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
#ifdef HYDRA_HAS_CUDA
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
                s->rain_mm_to_model_depth,
                0.0);
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
    s->rain_cn_enabled = true;
#ifdef HYDRA_HAS_CUDA
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
                                      s->rain_mm_to_model_depth,
                                      rain_update_interval_s);
    }
#endif
}

/** Configure per-cell externally-coupled depth source terms [m/s] and sync GPU. */
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

#ifdef HYDRA_HAS_CUDA
    if (s->dev) {
        swe2d_gpu_set_external_sources(
            s->dev,
            s->external_sources_enabled ? s->external_source_terms.data() : nullptr,
            s->external_sources_enabled ? nc : 0);
    }
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
/** Native run-to-time loop (removes per-step Python orchestration).
    Runs simulation from current t to t_end, batching diagnostics.
    @param s Solver handle @param cfg Run configuration
    @param diag_out Output diagnostics array (optional) @param max_diags Capacity
    @returns Number of diagnostics written; negative count if cancelled */
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

