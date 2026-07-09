#!/usr/bin/env python3
"""Runtime step executor seam for SWE2D workbench.

Phase 8 goal: extract per-step source/coupling/solve execution branch
from `_on_run` into a focused helper module.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

import numpy as np

import logging
logger = logging.getLogger(__name__)


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
        coupling_controller: Any,
        rain_source_for_window_callback: Callable[..., Any],
        cell_source_model_at_time_callback: Callable[[float], Optional[np.ndarray]],
        accumulate_source_volume_model_callback: Callable[..., None],
        apply_external_sources_callback: Callable[..., None],
        native_source_injection_mode: bool,
        apply_3d_patch_face_bc_callback: Optional[Callable[..., None]] = None,
    ) -> Dict[str, Any]:
        """execute step."""
        step_ms = 0.0
        coupling_ms = 0.0
        source_ms = 0.0
        state_ms = 0.0
        bc_ms = 0.0
        gpu_ms = 0.0

        rain_src = 0.0
        coupled_source_rate = None
        dt_used = float(dt_cfg)

        if dt_request <= 0.0:
            dt_source_guess = float(last_diag.get("dt", dt_cfg)) if isinstance(last_diag, dict) else dt_cfg
        else:
            dt_source_guess = float(dt_request)
        cell_source_model_step = cell_source_model_at_time_callback(t_accum)
        coupled_source_rate = None
        _native_device_applied = False
        if coupling_controller is not None:
            _t_cpl0 = time.perf_counter()
            # GPU-only path — no CPU fallback. Fail loudly if unavailable.
            _native_device_applied = bool(
                coupling_controller.apply_native_device_sources(
                    t_accum, dt_source_guess
                )
            )
            if _native_device_applied:
                coupled_source_rate = None  # sources already on GPU
            coupling_ms += (time.perf_counter() - _t_cpl0) * 1000.0
        # GPU-native coupling path — all operations use the same CUDA
        # stream (dev->d_stream), so ordering is implicit.  No sync needed.
        rain_src = rain_source_for_window_callback(
            t_accum,
            t_accum + dt_source_guess,
            accumulate=True,
            mutate_state=True,
        )
        _t_src2 = time.perf_counter()
        if not _native_device_applied:
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
            )
        else:
            # Native device coupling path: d_external_source_mps already has
            # structure/drainage sources on-device.  Accumulate rain and any
            # internal-flow source directly on the GPU without overwriting the
            # coupled contributions.
            if cell_source_model_step is not None:
                cell_arr = np.asarray(cell_source_model_step, dtype=np.float64)
                if cell_arr.size > 0 and np.any(cell_arr != 0.0):
                    try:
                        cell_area = np.asarray(backend.cell_areas(), dtype=np.float64)
                        safe_area = np.maximum(cell_area, 1.0e-8)
                        backend.accumulate_external_sources_native(cell_arr / safe_area)
                    except Exception:
                        logger.warning("Internal-flow accumulation on native coupling path failed", exc_info=True)
            if rain_src is not None and np.any(np.asarray(rain_src, dtype=np.float64) > 0.0):
                backend.accumulate_external_sources_native(rain_src)
            accumulate_source_volume_model_callback(
                dt_source_guess,
                rain_src,
                cell_source_model_step,
                None,
            )
        # When _native_device_applied is True, d_external_source_mps is
        # already populated on-device and the solver step will consume it
        # directly.  Rain is assumed to be zero or natively handled when
        # the native device coupling path is active; if non-zero rain
        # needs to be combined with on-device sources, a GPU accumulation
        # kernel or a D2H-readback + merge + H2D-upload path is required.
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
            "gpu_ms": gpu_ms,
        }
