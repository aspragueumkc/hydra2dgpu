#!/usr/bin/env python3
"""Runtime step executor seam for SWE2D workbench.

Phase 8 goal: extract per-step source/coupling/solve execution branch
from `_on_run` into a focused helper module.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

import numpy as np


class SWE2DRuntimeStepExecutor:
    """Executes one runtime step with stage-coupled or standard source path."""

    def execute_step(
        self,
        *,
        backend: Any,
        t_accum: float,
        last_diag: Optional[Dict[str, Any]],
        dt_cfg: float,
        dt_request: float,
        stage_coupled_imex_enabled: bool,
        coupling_controller: Any,
        dynamic_bc: bool,
        native_bc_forcing: bool,
        bc_n0: np.ndarray,
        bc_n1: np.ndarray,
        bc_tp: np.ndarray,
        bc_vl: np.ndarray,
        side_hydrographs: Dict[str, Any],
        edge_hydrographs: Dict[Any, Any],
        apply_timeseries_bc_values_callback: Callable[..., Any],
        distribute_total_flow_to_unit_q_callback: Callable[..., np.ndarray],
        rain_source_for_window_callback: Callable[..., Any],
        cell_source_model_at_time_callback: Callable[[float], Optional[np.ndarray]],
        accumulate_source_volume_model_callback: Callable[..., None],
        apply_external_sources_callback: Callable[..., None],
        native_source_injection_mode: bool,
        apply_3d_patch_face_bc_callback: Optional[Callable[..., None]] = None,
    ) -> Dict[str, Any]:
        step_ms = 0.0
        coupling_ms = 0.0
        source_ms = 0.0
        state_ms = 0.0
        bc_ms = 0.0

        if dynamic_bc and not native_bc_forcing:
            _t_bc0 = time.perf_counter()
            bc_tp_step, bc_vl_step = apply_timeseries_bc_values_callback(
                bc_n0, bc_n1, bc_tp, bc_vl, side_hydrographs, t_accum, edge_hydrographs
            )
            bc_vl_step = distribute_total_flow_to_unit_q_callback(
                bc_n0,
                bc_n1,
                bc_tp_step,
                bc_vl_step,
                bc_tp,
                side_hydrographs,
                edge_hydrographs,
            )
            backend.set_boundary_conditions(bc_n0, bc_n1, bc_tp_step, bc_vl_step)
            bc_ms += (time.perf_counter() - _t_bc0) * 1000.0

        rain_src = 0.0
        dt_used = float(dt_cfg)

        if stage_coupled_imex_enabled:
            _t_state0 = time.perf_counter()
            h0_c, hu0_c, hv0_c = backend.get_state()
            state_ms += (time.perf_counter() - _t_state0) * 1000.0
            if dt_request <= 0.0:
                dt_stage_guess = float(last_diag.get("dt", dt_cfg)) if isinstance(last_diag, dict) else dt_cfg
            else:
                dt_stage_guess = float(dt_request)
            cell_source_model_0 = cell_source_model_at_time_callback(t_accum)
            _t_cpl0 = time.perf_counter()
            coupled_source_rate_0 = coupling_controller.compute_source_rates(
                t_accum,
                dt_stage_guess,
                h0_c,
                hu0_c,
                hv0_c,
            )
            coupling_ms += (time.perf_counter() - _t_cpl0) * 1000.0
            rain_src_pred = rain_source_for_window_callback(
                t_accum,
                t_accum + dt_stage_guess,
                accumulate=False,
                mutate_state=False,
            )
            _t_src0 = time.perf_counter()
            apply_external_sources_callback(
                backend,
                dt_stage_guess,
                rain_src_pred,
                cell_source_model_0,
                coupled_source_rate_0,
                prefer_native_injection=native_source_injection_mode,
            )
            source_ms += (time.perf_counter() - _t_src0) * 1000.0
            if apply_3d_patch_face_bc_callback is not None:
                apply_3d_patch_face_bc_callback(backend)
            _t_step0 = time.perf_counter()
            _diag_predict = backend.step(dt_request)
            step_ms += (time.perf_counter() - _t_step0) * 1000.0
            dt_used = float(_diag_predict.get("dt", dt_cfg))
            _t_state1 = time.perf_counter()
            h1_c, hu1_c, hv1_c = backend.get_state()
            state_ms += (time.perf_counter() - _t_state1) * 1000.0
            _t_cpl1 = time.perf_counter()
            coupled_source_rate_1 = coupling_controller.compute_source_rates(
                t_accum + dt_used,
                dt_used,
                h1_c,
                hu1_c,
                hv1_c,
            )
            coupling_ms += (time.perf_counter() - _t_cpl1) * 1000.0
            coupled_source_rate = 0.5 * (
                np.asarray(coupled_source_rate_0, dtype=np.float64)
                + np.asarray(coupled_source_rate_1, dtype=np.float64)
            )
            backend.set_state(h0_c, hu0_c, hv0_c)
            if dynamic_bc and not native_bc_forcing:
                _t_bc1 = time.perf_counter()
                bc_tp_step, bc_vl_step = apply_timeseries_bc_values_callback(
                    bc_n0, bc_n1, bc_tp, bc_vl, side_hydrographs, t_accum, edge_hydrographs
                )
                bc_vl_step = distribute_total_flow_to_unit_q_callback(
                    bc_n0,
                    bc_n1,
                    bc_tp_step,
                    bc_vl_step,
                    bc_tp,
                    side_hydrographs,
                    edge_hydrographs,
                )
                backend.set_boundary_conditions(bc_n0, bc_n1, bc_tp_step, bc_vl_step)
                bc_ms += (time.perf_counter() - _t_bc1) * 1000.0
            rain_src = rain_source_for_window_callback(
                t_accum,
                t_accum + dt_used,
                accumulate=True,
                mutate_state=True,
            )
            cell_source_model_1 = cell_source_model_at_time_callback(t_accum + dt_used)
            if cell_source_model_0 is None:
                cell_source_model_stage = cell_source_model_1
            elif cell_source_model_1 is None:
                cell_source_model_stage = cell_source_model_0
            else:
                cell_source_model_stage = 0.5 * (
                    np.asarray(cell_source_model_0, dtype=np.float64)
                    + np.asarray(cell_source_model_1, dtype=np.float64)
                )
            _t_src1 = time.perf_counter()
            accumulate_source_volume_model_callback(
                dt_used,
                rain_src,
                cell_source_model_stage,
                coupled_source_rate,
            )
            apply_external_sources_callback(
                backend,
                dt_used,
                rain_src,
                cell_source_model_stage,
                coupled_source_rate,
                prefer_native_injection=native_source_injection_mode,
            )
            source_ms += (time.perf_counter() - _t_src1) * 1000.0
            if apply_3d_patch_face_bc_callback is not None:
                apply_3d_patch_face_bc_callback(backend)
            _t_step1 = time.perf_counter()
            last_diag = backend.step(dt_used)
            step_ms += (time.perf_counter() - _t_step1) * 1000.0
        else:
            if dt_request <= 0.0:
                dt_source_guess = float(last_diag.get("dt", dt_cfg)) if isinstance(last_diag, dict) else dt_cfg
            else:
                dt_source_guess = float(dt_request)
            cell_source_model_step = cell_source_model_at_time_callback(t_accum)
            coupled_source_rate = None
            if coupling_controller is not None:
                _t_state2 = time.perf_counter()
                h_c, hu_c, hv_c = backend.get_state()
                state_ms += (time.perf_counter() - _t_state2) * 1000.0
                _t_cpl2 = time.perf_counter()
                coupled_source_rate = coupling_controller.compute_source_rates(
                    t_accum,
                    dt_source_guess,
                    h_c,
                    hu_c,
                    hv_c,
                )
                coupling_ms += (time.perf_counter() - _t_cpl2) * 1000.0
            rain_src = rain_source_for_window_callback(
                t_accum,
                t_accum + dt_source_guess,
                accumulate=True,
                mutate_state=True,
            )
            _t_src2 = time.perf_counter()
            accumulate_source_volume_model_callback(
                dt_source_guess,
                rain_src,
                cell_source_model_step,
                coupled_source_rate,
            )
            apply_external_sources_callback(
                backend,
                dt_source_guess,
                rain_src,
                cell_source_model_step,
                coupled_source_rate,
                prefer_native_injection=native_source_injection_mode,
            )
            source_ms += (time.perf_counter() - _t_src2) * 1000.0
            if apply_3d_patch_face_bc_callback is not None:
                apply_3d_patch_face_bc_callback(backend)
            _t_step2 = time.perf_counter()
            last_diag = backend.step(dt_request)
            step_ms += (time.perf_counter() - _t_step2) * 1000.0
            dt_used = float(last_diag.get("dt", dt_cfg))

        return {
            "last_diag": last_diag,
            "dt_used": dt_used,
            "rain_src": rain_src,
            "step_ms": step_ms,
            "coupling_ms": coupling_ms,
            "source_ms": source_ms,
            "state_ms": state_ms,
            "bc_ms": bc_ms,
        }
