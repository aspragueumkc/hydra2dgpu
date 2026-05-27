#!/usr/bin/env python3
"""Runtime source and budget helper for SWE2D workbench.

Phase 7 goal: extract per-step source sampling and mass-budget accounting
helpers from `_on_run` into a focused module.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np


class SWE2DRuntimeSourceManager:
    """Encapsulates runtime source sampling and volume accounting."""

    def __init__(
        self,
        *,
        rain_rate_model: Any,
        thiessen_forcing: Any,
        native_rain_cn_forcing: bool,
        internal_flow_forcing: Any,
        rain_stats_acc: Dict[str, float],
        area_model: np.ndarray,
        edge_len_bc: np.ndarray,
        edge_group_labels: List[str],
        inflow_q_bc_type: int,
        rain_rate_si_to_model_callback: Callable[[Any], Any],
        internal_flow_source_cms_at_time_callback: Callable[[Any, float], Any],
        flow_si_to_model_callback: Callable[[Any], Any],
    ):
        self._rain_rate_model = rain_rate_model
        self._thiessen_forcing = thiessen_forcing
        self._native_rain_cn_forcing = bool(native_rain_cn_forcing)
        self._internal_flow_forcing = internal_flow_forcing
        self._rain_stats_acc = rain_stats_acc
        self._area_model = np.asarray(area_model, dtype=np.float64).ravel()
        self._n_area = int(self._area_model.size)
        self._edge_len_bc = np.asarray(edge_len_bc, dtype=np.float64).ravel()
        self._edge_group_labels = list(edge_group_labels)
        self._inflow_q_bc_type = int(inflow_q_bc_type)
        self._rain_rate_si_to_model = rain_rate_si_to_model_callback
        self._internal_flow_source_cms_at_time = internal_flow_source_cms_at_time_callback
        self._flow_si_to_model = flow_si_to_model_callback

        self.source_budget_model: Dict[str, float] = {
            "rain": 0.0,
            "cell": 0.0,
            "coupling": 0.0,
        }
        self._source_time_s = 0.0
        self.source_step_rows_model: List[Dict[str, float]] = []
        self.boundary_flux_budget_model: Dict[str, float] = {}
        self._boundary_flux_time_s = 0.0
        self.boundary_flux_step_rows_model: List[Dict[str, float]] = []

    def rain_source_for_window(self, t0_s: float, t1_s: float, accumulate: bool, mutate_state: bool) -> Any:
        rain_src_local = self._rain_rate_model
        if self._thiessen_forcing is not None and not self._native_rain_cn_forcing:
            rain_src_si_local, rain_diag_local = self._thiessen_forcing.step_net_rainfall_mps(
                t0_s,
                t1_s,
                mutate_state=mutate_state,
            )
            rain_src_local = self._rain_rate_si_to_model(rain_src_si_local)
            if accumulate:
                self._rain_stats_acc["rain_mm"] += float(rain_diag_local.get("rain_mm_mean", 0.0))
                self._rain_stats_acc["excess_mm"] += float(rain_diag_local.get("excess_mm_mean", 0.0))
                self._rain_stats_acc["samples"] += 1
        elif self._native_rain_cn_forcing:
            rain_src_local = 0.0
        return rain_src_local

    def cell_source_model_at_time(self, t_s: float) -> Optional[np.ndarray]:
        src_si = self._internal_flow_source_cms_at_time(self._internal_flow_forcing, t_s)
        if src_si is None:
            return None
        return self._flow_si_to_model(src_si)

    def accumulate_source_volume_model(
        self,
        dt_apply_s: float,
        rain_rate_model_local: Any,
        cell_source_model_local: Optional[np.ndarray],
        coupled_source_rate_local: Optional[np.ndarray],
    ) -> None:
        dt_apply = max(0.0, float(dt_apply_s))
        if dt_apply <= 0.0 or self._n_area <= 0:
            return

        rain_vol = 0.0
        cell_vol = 0.0
        cpl_vol = 0.0

        rain_arr = np.asarray(rain_rate_model_local, dtype=np.float64)
        if rain_arr.ndim == 0:
            rain_vol = float(rain_arr) * float(np.sum(self._area_model)) * dt_apply
        else:
            n = min(int(rain_arr.size), self._n_area)
            rain_vol = float(np.sum(rain_arr[:n] * self._area_model[:n]) * dt_apply)
        if np.isfinite(rain_vol):
            self.source_budget_model["rain"] += rain_vol

        if cell_source_model_local is not None:
            cell_arr = np.asarray(cell_source_model_local, dtype=np.float64).ravel()
            if cell_arr.size > 0:
                n = min(int(cell_arr.size), self._n_area)
                cell_vol = float(np.sum(cell_arr[:n]) * dt_apply)
                if np.isfinite(cell_vol):
                    self.source_budget_model["cell"] += cell_vol

        if coupled_source_rate_local is not None:
            cpl_arr = np.asarray(coupled_source_rate_local, dtype=np.float64).ravel()
            if cpl_arr.size > 0:
                n = min(int(cpl_arr.size), self._n_area)
                cpl_vol = float(np.sum(cpl_arr[:n] * self._area_model[:n]) * dt_apply)
                if np.isfinite(cpl_vol):
                    self.source_budget_model["coupling"] += cpl_vol

        self._source_time_s = float(self._source_time_s + dt_apply)
        self.source_step_rows_model.append(
            {
                "t_s": float(self._source_time_s),
                "rain_vol_model": float(rain_vol) if np.isfinite(rain_vol) else 0.0,
                "cell_vol_model": float(cell_vol) if np.isfinite(cell_vol) else 0.0,
                "coupling_vol_model": float(cpl_vol) if np.isfinite(cpl_vol) else 0.0,
                "source_total_vol_model": float(
                    (rain_vol if np.isfinite(rain_vol) else 0.0)
                    + (cell_vol if np.isfinite(cell_vol) else 0.0)
                    + (cpl_vol if np.isfinite(cpl_vol) else 0.0)
                ),
            }
        )

    def accumulate_boundary_flux_volume_model(
        self,
        dt_apply_s: float,
        bc_type_local: np.ndarray,
        bc_val_local: np.ndarray,
    ) -> None:
        dt_apply = max(0.0, float(dt_apply_s))
        if dt_apply <= 0.0:
            return

        bt = np.asarray(bc_type_local, dtype=np.int32).ravel()
        bv = np.asarray(bc_val_local, dtype=np.float64).ravel()
        n = min(int(bt.size), int(bv.size), int(self._edge_len_bc.size), len(self._edge_group_labels))
        if n <= 0:
            return

        flow_mask = bt[:n] == self._inflow_q_bc_type
        if not np.any(flow_mask):
            return

        q_total = np.asarray(bv[:n], dtype=np.float64) * np.asarray(self._edge_len_bc[:n], dtype=np.float64)
        vol = q_total * dt_apply
        idx = np.nonzero(flow_mask)[0]
        group_acc: Dict[str, Dict[str, float]] = {}
        for ii in idx.tolist():
            grp = str(self._edge_group_labels[ii])
            qv = float(q_total[ii])
            vv = float(vol[ii])
            if not (np.isfinite(vv) and np.isfinite(qv)):
                continue
            self.boundary_flux_budget_model[grp] = float(self.boundary_flux_budget_model.get(grp, 0.0) + vv)
            acc = group_acc.setdefault(grp, {"vol": 0.0, "q": 0.0})
            acc["vol"] = float(acc["vol"] + vv)
            acc["q"] = float(acc["q"] + qv)

        self._boundary_flux_time_s = float(self._boundary_flux_time_s + dt_apply)
        if group_acc:
            for grp in sorted(group_acc):
                vals = group_acc[grp]
                self.boundary_flux_step_rows_model.append(
                    {
                        "t_s": float(self._boundary_flux_time_s),
                        "group": str(grp),
                        "q_model": float(vals["q"]),
                        "vol_model": float(vals["vol"]),
                    }
                )
