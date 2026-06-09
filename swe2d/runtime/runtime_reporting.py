#!/usr/bin/env python3
"""Runtime reporting seam for SWE2D workbench.

Phase 9 goal: extract post-step snapshot/progress/logging reporting from
`_on_run` into a focused helper module.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, Optional

import numpy as np


class SWE2DRuntimeReporter:
    """Handles per-step diagnostics, snapshot capture, and runtime logging."""

    def process_step(
        self,
        *,
        backend: Any,
        t_accum: float,
        dt_used: float,
        last_diag: Dict[str, Any],
        last_valid_cmax: float,
        last_valid_wse_res: float,
        sample_map: Any,
        cell_solver_z: Optional[np.ndarray],
        coupling_controller: Any,
        experimental_3d_runtime: bool,
        rain_src: Any,
        state_ms: float,
        ui_ms: float,
        step_wall_t0: float,
        step_ms: float,
        coupling_ms: float,
        source_ms: float,
        bc_ms: float,
        timing_totals_ms: Dict[str, float],
        timing_samples: int,
        i: int,
        run_duration_s: float,
        next_snap_t: float,
        next_line_snap_t: float,
        next_coupling_snap_t: float,
        output_interval_s: float,
        line_output_interval_s: float,
        process_events_interval_s: float,
        last_process_events_wall: float,
        h_min: float,
        length_unit_name: str,
        snapshot_timesteps: list,
        line_snapshot_rows: list,
        line_snapshot_profile_rows: list,
        coupling_snapshot_rows: list,
        get_3d_patch_stats_callback: Callable[[], Optional[Dict[str, object]]],
        get_3d_patch_vof_callback: Callable[[], Optional[np.ndarray]],
        get_3d_patch_velocity_callback: Optional[Callable[[], Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]]],
        physics_diag_enabled: bool = False,
        front_flux_damping_value: float = 1.0,
        zmax_bc_mode: Optional[int] = None,
        append_3d_patch_snapshot_callback: Callable[
            [float, Dict[str, object], np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]],
            None,
        ],
        sample_line_metrics_callback: Callable[..., Any],
        sample_coupling_object_metrics_callback: Callable[..., Any],
        process_events_callback: Callable[[], None],
        set_progress_callback: Callable[[int], None],
        log_callback: Callable[[str], None],
        perf_mode: bool = False,
    ) -> Dict[str, Any]:
        step_cmax = float(last_diag.get("max_courant", float("nan")))
        if np.isfinite(step_cmax) and step_cmax >= 0.0:
            last_valid_cmax = step_cmax
        step_wse_res = float(
            last_diag.get(
                "max_depth_residual",
                last_diag.get("max_wse_elev_error", float("nan")),
            )
        )
        if np.isfinite(step_wse_res) and step_wse_res >= 0.0:
            last_valid_wse_res = step_wse_res
        t_accum += float(dt_used)

        need_mesh_snap = t_accum >= float(next_snap_t)
        need_line_snap = bool(sample_map) and t_accum >= float(next_line_snap_t)
        need_coupling_snap = (coupling_controller is not None) and (t_accum >= float(next_coupling_snap_t))

        h_s = hu_s = hv_s = None
        if need_mesh_snap or need_line_snap or need_coupling_snap:
            _t_state3 = time.perf_counter()
            h_s, hu_s, hv_s = backend.get_state()
            state_ms += (time.perf_counter() - _t_state3) * 1000.0

        if need_mesh_snap and h_s is not None and hu_s is not None and hv_s is not None:
            snapshot_timesteps.append((t_accum, h_s.copy(), hu_s.copy(), hv_s.copy()))
            if experimental_3d_runtime:
                s3 = get_3d_patch_stats_callback()
                v3 = get_3d_patch_vof_callback()
                if s3 is not None and v3 is not None:
                    u3 = v3c = w3 = None
                    if get_3d_patch_velocity_callback is not None:
                        vel = get_3d_patch_velocity_callback()
                        if isinstance(vel, tuple) and len(vel) == 3:
                            u3 = np.asarray(vel[0], dtype=np.float64).ravel()
                            v3c = np.asarray(vel[1], dtype=np.float64).ravel()
                            w3 = np.asarray(vel[2], dtype=np.float64).ravel()
                    append_3d_patch_snapshot_callback(t_accum, s3, v3, u3, v3c, w3)
            next_snap_t += float(output_interval_s)

        if need_line_snap and cell_solver_z is not None and h_s is not None and hu_s is not None and hv_s is not None:
            rows, profile_rows = sample_line_metrics_callback(
                sample_map,
                t_accum,
                h_s,
                hu_s,
                hv_s,
                cell_solver_z,
            )
            if rows:
                line_snapshot_rows.extend(rows)
            if profile_rows:
                line_snapshot_profile_rows.extend(profile_rows)
            next_line_snap_t += float(line_output_interval_s)

        if need_coupling_snap and h_s is not None:
            c_rows = sample_coupling_object_metrics_callback(coupling_controller, t_accum, h_s)
            if c_rows:
                coupling_snapshot_rows.extend(c_rows)
            next_coupling_snap_t += float(line_output_interval_s)

        _now_wall = time.perf_counter()
        if _now_wall - float(last_process_events_wall) >= float(process_events_interval_s):
            _t_ui0 = time.perf_counter()
            process_events_callback()
            ui_ms += (time.perf_counter() - _t_ui0) * 1000.0
            last_process_events_wall = _now_wall

        step_wall_ms = (time.perf_counter() - step_wall_t0) * 1000.0
        timing_totals_ms["wall"] += step_wall_ms
        timing_totals_ms["step"] += step_ms
        timing_totals_ms["coupling"] += coupling_ms
        timing_totals_ms["source"] += source_ms
        timing_totals_ms["state"] += state_ms
        timing_totals_ms["bc"] += bc_ms
        timing_totals_ms["ui"] += ui_ms
        timing_samples += 1

        pct = int(min(100.0, (t_accum / max(float(run_duration_s), 1.0e-9)) * 100.0))
        set_progress_callback(pct)
        i += 1

        should_log_default = (i == 1 or i % 10 == 0 or pct >= 100)
        should_log_perf = (i == 1 or i % 200 == 0 or pct >= 100)
        if (not perf_mode and should_log_default) or (perf_mode and should_log_perf):
            max_courant = last_valid_cmax
            max_wse_res = last_valid_wse_res
            cmax_txt = f"{max_courant:.5f}" if np.isfinite(max_courant) and max_courant >= 0.0 else "n/a"
            wse_res_txt = f"{max_wse_res:.6e}" if np.isfinite(max_wse_res) and max_wse_res >= 0.0 else "n/a"
            rain_diag_txt = ""
            rain_arr_diag = np.asarray(rain_src, dtype=np.float64)
            if (not perf_mode) and np.any(rain_arr_diag > 0.0):
                _t_state4 = time.perf_counter()
                h_d, hu_d, hv_d = backend.get_state()
                state_ms += (time.perf_counter() - _t_state4) * 1000.0
                h_d = np.asarray(h_d, dtype=np.float64)
                hu_d = np.asarray(hu_d, dtype=np.float64)
                hv_d = np.asarray(hv_d, dtype=np.float64)
                wet_mask = h_d > float(h_min)
                if np.any(wet_mask):
                    inv_h = 1.0 / np.maximum(h_d[wet_mask], 1.0e-12)
                    speed = np.sqrt((hu_d[wet_mask] * inv_h) ** 2 + (hv_d[wet_mask] * inv_h) ** 2)
                    umax = float(np.max(speed)) if speed.size else 0.0
                    hmin_wet = float(np.min(h_d[wet_mask]))
                    hmax = float(np.max(h_d)) if h_d.size else 0.0
                    rain_diag_txt = (
                        f" rain:umax={umax:.3e} {length_unit_name}/s"
                        f" hminWet={hmin_wet:.3e} {length_unit_name}"
                        f" hmax={hmax:.3e} {length_unit_name}"
                    )
                else:
                    rain_diag_txt = " rain:all-dry"
            log_callback(
                (
                    f"step={i} t={t_accum / 3600.0:.3f} hr / {run_duration_s / 3600.0:.3f} hr "
                    f"dt={float(last_diag.get('dt', 0.0)):.5f} "
                    f"gpu={bool(last_diag.get('gpu_active', False))} wet={last_diag.get('wet_cells', '?')} "
                    f"Cmax={cmax_txt} WSEres={wse_res_txt} "
                    f"graph_step={int(last_diag.get('gpu_graph_launches_step', 0))} "
                    f"graph_total={int(last_diag.get('gpu_graph_launches_total', 0))}"
                    f"{rain_diag_txt}"
                )
            )
            tiny_keys = (
                "tiny_mode_requested",
                "tiny_mode_selected",
                "tiny_mode_effective",
                "tiny_mode_fallback",
                "tiny_mode_fallback_count_total",
            )
            if all(k in last_diag for k in tiny_keys):
                log_callback(
                    "  tiny: "
                    f"req={int(last_diag.get('tiny_mode_requested', -1))} "
                    f"sel={int(last_diag.get('tiny_mode_selected', -1))} "
                    f"eff={int(last_diag.get('tiny_mode_effective', -1))} "
                    f"fallback={bool(last_diag.get('tiny_mode_fallback', False))} "
                    f"fallback_total={int(last_diag.get('tiny_mode_fallback_count_total', 0))}"
                )
            if perf_mode:
                return {
                    "t_accum": t_accum,
                    "last_valid_cmax": last_valid_cmax,
                    "last_valid_wse_res": last_valid_wse_res,
                    "next_snap_t": next_snap_t,
                    "next_line_snap_t": next_line_snap_t,
                    "next_coupling_snap_t": next_coupling_snap_t,
                    "last_process_events_wall": last_process_events_wall,
                    "timing_samples": timing_samples,
                    "i": i,
                    "state_ms": state_ms,
                    "ui_ms": ui_ms,
                }
            if experimental_3d_runtime:
                stats = get_3d_patch_stats_callback()
                if stats is not None:
                    log_callback(
                        "  3d_patch: "
                        f"vof=[{float(stats.get('vof_min', float('nan'))):.3e}, "
                        f"{float(stats.get('vof_max', float('nan'))):.3e}] "
                        f"vof_sum={float(stats.get('vof_sum', float('nan'))):.6e} "
                        f"u_rms={float(stats.get('u_rms', float('nan'))):.3e} "
                        f"v_rms={float(stats.get('v_rms', float('nan'))):.3e} "
                        f"w_rms={float(stats.get('w_rms', float('nan'))):.3e} "
                        f"p_abs_max={float(stats.get('p_max_abs', float('nan'))):.3e}"
                    )
                if physics_diag_enabled:
                    try:
                        dt_step = float(last_diag.get("dt", dt_used) or dt_used)
                    except Exception:
                        dt_step = float(dt_used)
                    predictor_damping_coeff = 0.05
                    predictor_damp = 1.0 / (1.0 + predictor_damping_coeff * max(0.0, dt_step))
                    log_callback(
                        "  3d_phys: "
                        f"dt={dt_step:.6g}s "
                        f"predictor_damp_coeff={predictor_damping_coeff:.6g} "
                        f"predictor_damp={predictor_damp:.6e} "
                        f"front_flux_damping={float(front_flux_damping_value):.6g} "
                        f"zmax_mode={int(zmax_bc_mode) if zmax_bc_mode is not None else -1}"
                    )
                    if get_3d_patch_velocity_callback is not None:
                        s3 = get_3d_patch_stats_callback()
                        vel3 = get_3d_patch_velocity_callback()
                        if (
                            isinstance(s3, dict)
                            and isinstance(vel3, tuple)
                            and len(vel3) == 3
                        ):
                            try:
                                nx3 = max(0, int(s3.get("nx", 0) or 0))
                                ny3 = max(0, int(s3.get("ny", 0) or 0))
                                nz3 = max(0, int(s3.get("nz", 0) or 0))
                                dx3 = float(s3.get("dx", 0.0) or 0.0)
                                dy3 = float(s3.get("dy", 0.0) or 0.0)
                                nxy3 = nx3 * ny3
                                if nxy3 > 0 and nz3 > 0 and dx3 > 0.0 and dy3 > 0.0:
                                    w3 = np.asarray(vel3[2], dtype=np.float64).ravel()
                                    n3 = nxy3 * nz3
                                    if w3.size == n3:
                                        top = w3[(nz3 - 1) * nxy3 : nz3 * nxy3]
                                        top_pos = np.maximum(top, 0.0)
                                        vent_q_est = float(np.sum(top_pos) * dx3 * dy3)
                                        top_w_max = float(np.max(top)) if top.size else 0.0
                                        log_callback(
                                            "  3d_phys_vent: "
                                            f"q_est={vent_q_est:.6e} {length_unit_name}^3/s "
                                            f"w_top_max={top_w_max:.6e} {length_unit_name}/s"
                                        )
                            except Exception:
                                pass
            if timing_samples > 0:
                avg_wall = timing_totals_ms["wall"] / timing_samples
                avg_step = timing_totals_ms["step"] / timing_samples
                avg_cpl = timing_totals_ms["coupling"] / timing_samples
                avg_src = timing_totals_ms["source"] / timing_samples
                avg_state = timing_totals_ms["state"] / timing_samples
                avg_bc = timing_totals_ms["bc"] / timing_samples
                avg_ui = timing_totals_ms["ui"] / timing_samples
                step_gpu_frac = 100.0 * step_ms / max(step_wall_ms, 1.0e-9)
                avg_gpu_frac = 100.0 * avg_step / max(avg_wall, 1.0e-9)
                other_ms = max(0.0, step_wall_ms - (step_ms + coupling_ms + source_ms + state_ms + bc_ms + ui_ms))
                avg_other = max(0.0, avg_wall - (avg_step + avg_cpl + avg_src + avg_state + avg_bc + avg_ui))
                log_callback(
                    "  timing(ms): "
                    f"wall={step_wall_ms:.2f} step={step_ms:.2f} coupling={coupling_ms:.2f} "
                    f"source={source_ms:.2f} state={state_ms:.2f} bc={bc_ms:.2f} ui={ui_ms:.2f} other={other_ms:.2f} "
                    f"gpu_frac={step_gpu_frac:.1f}%"
                )
                log_callback(
                    "  timing-avg(ms): "
                    f"wall={avg_wall:.2f} step={avg_step:.2f} coupling={avg_cpl:.2f} "
                    f"source={avg_src:.2f} state={avg_state:.2f} bc={avg_bc:.2f} ui={avg_ui:.2f} other={avg_other:.2f} "
                    f"gpu_frac={avg_gpu_frac:.1f}%"
                )
            if coupling_controller is not None:
                cdiag = coupling_controller.last_diag
                limiter_events = float(cdiag.component_sums.get("drainage_limiter_events", 0.0))
                limiter_vol_model3 = float(cdiag.component_sums.get("drainage_limiter_volume_m3", 0.0))
                drain_substeps = float(cdiag.component_sums.get("drainage_substeps_used", 1.0))
                native_iterative = int(cdiag.component_sums.get("drainage_native_iterative", 0))
                log_callback(
                    "  coupling: "
                    f"drain_qmax={cdiag.drainage_max_link_flow:.4f} {length_unit_name}^3/s, "
                    f"drain_hmax={cdiag.drainage_max_node_depth:.4f} {length_unit_name}, "
                    f"struct_qsum={cdiag.structure_total_flow:.4f} {length_unit_name}^3/s, "
                    f"src_range=[{cdiag.source_min_mps:.3e}, {cdiag.source_max_mps:.3e}] {length_unit_name}/s, "
                    f"drain_substeps={drain_substeps:.0f}, "
                    f"native_iter={native_iterative}, "
                    f"limiter_events={limiter_events:.0f}, "
                    f"limiter_vol={limiter_vol_model3:.6f} {length_unit_name}^3"
                )

        return {
            "t_accum": t_accum,
            "last_valid_cmax": last_valid_cmax,
            "last_valid_wse_res": last_valid_wse_res,
            "next_snap_t": next_snap_t,
            "next_line_snap_t": next_line_snap_t,
            "next_coupling_snap_t": next_coupling_snap_t,
            "last_process_events_wall": last_process_events_wall,
            "timing_samples": timing_samples,
            "i": i,
            "state_ms": state_ms,
            "ui_ms": ui_ms,
        }
