#!/usr/bin/env python3
"""Runtime setup configurator seam for SWE2D workbench.

Phase 14+ goal: extract remaining setup blocks from `_on_run` into a
reusable helper module.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


class SWE2DRunSetupConfigurator:
    """Owns setup-stage configuration helpers used before time stepping."""

    def configure_native_rain_cn_forcing(
        self,
        *,
        backend: Any,
        thiessen_forcing: Any,
        mm_to_model_depth: float,
    ) -> Dict[str, Any]:
        """configure native rain cn forcing."""
        payload = thiessen_forcing.build_native_preprocessed_payload()
        cell_gage_idx = np.asarray(payload.get("cell_gage_idx"), dtype=np.int32).ravel()
        gage_offsets = np.asarray(payload.get("gage_offsets"), dtype=np.int32).ravel()
        hg_time_s = np.asarray(payload.get("hg_time_s"), dtype=np.float64).ravel()
        hg_cum_mm = np.asarray(payload.get("hg_cum_mm"), dtype=np.float64).ravel()
        cn_arr = np.asarray(payload.get("cn"), dtype=np.float64).ravel()
        ia_ratio = float(np.asarray(payload.get("ia_ratio", [0.0]), dtype=np.float64).ravel()[0])

        if cell_gage_idx.size > 0 and np.any(cell_gage_idx >= 0):
            backend.set_rain_cn_forcing_native(
                cell_gage_idx=cell_gage_idx,
                gage_offsets=gage_offsets,
                hg_time_s=hg_time_s,
                hg_cum_mm=hg_cum_mm,
                cn=cn_arr,
                ia_ratio=ia_ratio,
                mm_to_model_depth=float(mm_to_model_depth),
            )
            infil_method_native = str(getattr(thiessen_forcing, "infiltration_method", "scs_cn") or "scs_cn").lower().strip()
            return {
                "configured": True,
                "infiltration_method": infil_method_native,
                "groups": max(0, int(gage_offsets.size) - 1),
            }

        return {
            "configured": False,
            "infiltration_method": None,
            "groups": 0,
        }

    def configure_constant_rain_rate_native(
        self,
        *,
        backend: Any,
        rate_model_mps: float,
        mm_to_model_depth: float,
    ) -> Dict[str, Any]:
        """Upload a constant rain rate via the native rain-CN API.

        Builds a single-gauge linear hyetograph so the GPU kernel reads
        the same rate every step.  No per-step H2D transfers.
        """
        n_cells = int(backend.n_cells)
        if n_cells <= 0 or float(rate_model_mps) <= 0.0:
            return {"configured": False, "infiltration_method": None, "groups": 0}
        cell_gage_idx = np.zeros(n_cells, dtype=np.int32)
        gage_offsets = np.array([0, 2], dtype=np.int32)
        rate_mm_s = float(rate_model_mps) / float(mm_to_model_depth)
        hg_time_s = np.array([0.0, 1.0e9], dtype=np.float64)
        hg_cum_mm = np.array([0.0, rate_mm_s * 1.0e9], dtype=np.float64)
        cn = np.full(n_cells, 100.0, dtype=np.float64)
        backend.set_rain_cn_forcing_native(
            cell_gage_idx=cell_gage_idx,
            gage_offsets=gage_offsets,
            hg_time_s=hg_time_s,
            hg_cum_mm=hg_cum_mm,
            cn=cn,
            ia_ratio=0.0,
            mm_to_model_depth=float(mm_to_model_depth),
        )
        return {
            "configured": True,
            "infiltration_method": "constant_rate",
            "groups": 1,
        }

    def configure_native_source_injection(self, *, backend: Any) -> Dict[str, Any]:
        """configure native source injection."""
        native_source_injection_mode = hasattr(backend, "set_external_sources_native")
        if not native_source_injection_mode:
            return {
                "native_source_injection_mode": False,
                "configured": False,
            }

        backend.set_external_sources_native(None)
        return {
            "native_source_injection_mode": True,
            "configured": True,
        }

    def resolve_stage_coupled_imex(
        self,
        *,
        requested: bool,
        coupling_controller: Any,
        temporal_scheme: Any,
        required_temporal_scheme: Any,
        native_source_injection_mode: bool,
    ) -> Dict[str, Any]:
        """resolve stage coupled imex."""
        stage_coupled_imex_enabled = False
        stage_reasons: List[str] = []

        if requested:
            if coupling_controller is None:
                stage_reasons.append("no coupling sources configured")
            if temporal_scheme != required_temporal_scheme:
                stage_reasons.append("temporal scheme is not RK2")
            if not native_source_injection_mode:
                stage_reasons.append("native source injection unavailable")
            if not stage_reasons:
                stage_coupled_imex_enabled = True

        return {
            "enabled": stage_coupled_imex_enabled,
            "reasons": stage_reasons,
        }
