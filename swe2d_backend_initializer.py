#!/usr/bin/env python3
"""Backend initialization seam for SWE2D workbench.

Phase 6 goal: extract native backend build/initialize setup from `_on_run`
into a focused helper module.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np


class SWE2DBackendInitializer:
    """Build and initialize native backend from prepared run inputs/options."""

    def __init__(
        self,
        ui: Any,
        apply_env_overrides_callback: Callable[[Dict[str, str]], Any],
        restore_env_overrides_callback: Callable[[Any], None],
        apply_timeseries_bc_values_callback: Callable[..., Any],
        distribute_total_flow_to_unit_q_callback: Callable[..., np.ndarray],
    ):
        self._ui = ui
        self._apply_env_overrides = apply_env_overrides_callback
        self._restore_env_overrides = restore_env_overrides_callback
        self._apply_timeseries_bc_values = apply_timeseries_bc_values_callback
        self._distribute_total_flow_to_unit_q = distribute_total_flow_to_unit_q_callback

    def build_and_initialize(
        self,
        *,
        backend_cls: Any,
        swe3d_env_overrides: Dict[str, str],
        dynamic_bc: bool,
        node_x: np.ndarray,
        node_y: np.ndarray,
        node_z: np.ndarray,
        cell_nodes: np.ndarray,
        face_offsets: Optional[np.ndarray],
        face_nodes: Optional[np.ndarray],
        bc_n0: np.ndarray,
        bc_n1: np.ndarray,
        bc_tp: np.ndarray,
        bc_vl: np.ndarray,
        side_hydrographs: Dict[str, object],
        edge_hydrographs: Dict[Any, Any],
        h0: np.ndarray,
        hu0: np.ndarray,
        hv0: np.ndarray,
        n_mann_cell: Optional[np.ndarray],
        dt_fixed: float,
        dt_max: float,
        model_options: Any,
        reconstruction_mode: int,
        temporal_scheme: Any,
        godunov_mode: Any,
    ) -> Any:
        _prev_env = self._apply_env_overrides(swe3d_env_overrides)
        try:
            b = backend_cls()

            bc_tp_init = bc_tp.copy()
            bc_vl_init = bc_vl.copy()
            if dynamic_bc:
                bc_tp_init, bc_vl_init = self._apply_timeseries_bc_values(
                    bc_n0, bc_n1, bc_tp_init, bc_vl_init, side_hydrographs, 0.0, edge_hydrographs
                )
            bc_vl_init = self._distribute_total_flow_to_unit_q(
                bc_n0,
                bc_n1,
                bc_tp_init,
                bc_vl_init,
                bc_tp,
                side_hydrographs,
                edge_hydrographs,
            )

            if face_offsets is not None and face_nodes is not None:
                b.build_mesh(
                    node_x,
                    node_y,
                    node_z,
                    face_nodes,
                    bc_n0,
                    bc_n1,
                    bc_tp_init,
                    bc_vl_init,
                    face_offsets,
                )
            else:
                b.build_mesh(node_x, node_y, node_z, cell_nodes, bc_n0, bc_n1, bc_tp_init, bc_vl_init)

            b.initialize(
                h0,
                hu0,
                hv0,
                g=float(self._ui._gravity),
                n_mann=float(self._ui.n_mann_spin.value()),
                n_mann_cell=n_mann_cell,
                cfl=float(self._ui.cfl_spin.value()),
                h_min=float(self._ui.h_min_spin.value()),
                dt_fixed=dt_fixed,
                dt_max=dt_max,
                max_inv_area=float(self._ui.max_inv_area_spin.value()),
                cfl_lambda_cap=float(self._ui.cfl_lambda_cap_spin.value()),
                momentum_cap_min_speed=float(self._ui.momentum_cap_min_speed_spin.value()),
                momentum_cap_celerity_mult=float(self._ui.momentum_cap_celerity_mult_spin.value()),
                depth_cap=float(self._ui.depth_cap_spin.value()),
                max_rel_depth_increase=float(self._ui.max_rel_depth_increase_spin.value()),
                shallow_damping_depth=float(self._ui.shallow_damping_depth_spin.value()),
                extreme_rain_mode=bool(self._ui.extreme_rain_mode_chk.isChecked()),
                source_cfl_beta=float(self._ui.source_cfl_beta_spin.value()),
                source_max_substeps=int(self._ui.source_max_substeps_spin.value()),
                source_rate_cap=float(self._ui.max_source_rate_spin.value()),
                source_depth_step_cap=float(self._ui.max_source_depth_step_spin.value()),
                source_true_subcycling=bool(self._ui.source_true_subcycling_chk.isChecked()),
                source_imex_split=bool(self._ui.source_imex_split_chk.isChecked()),
                enable_shallow_front_recon_fallback=bool(self._ui.shallow_front_recon_fallback_chk.isChecked()),
                gpu_diag_sync_interval_steps=int(self._ui.gpu_diag_sync_interval_spin.value()),
                model_options=model_options,
                spatial_discretization=reconstruction_mode,
                temporal_scheme=temporal_scheme,
                godunov_mode=godunov_mode,
                degen_mode=int(self._ui.degen_mode_combo.currentData()),
                front_flux_damping=float(self._ui.front_flux_damping_spin.value()),
                active_set_hysteresis=bool(self._ui.active_set_hysteresis_chk.isChecked()),
            )
            return b
        finally:
            self._restore_env_overrides(_prev_env)
