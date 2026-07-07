#!/usr/bin/env python3
"""Backend initialization seam for SWE2D workbench.

Phase 6 goal: extract native backend build/initialize setup from `_on_run`
into a focused helper module.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

import logging
logger = logging.getLogger(__name__)

from swe2d.runtime.backend import build_mesh as shared_build_mesh
from swe2d.services.gpkg_persistence_service import persist_baked_mesh


class SWE2DBackendInitializer:
    """Build and initialize native backend from prepared run inputs/options."""

    def __init__(
        self,
        apply_timeseries_bc_values_callback: Optional[Callable[..., Any]] = None,
        distribute_total_flow_to_unit_q_callback: Optional[Callable[..., np.ndarray]] = None,
    ):
        self._apply_timeseries_bc_values = apply_timeseries_bc_values_callback
        self._distribute_total_flow_to_unit_q = distribute_total_flow_to_unit_q_callback

    def build_and_initialize(
        self,
        *,
        backend_cls: Any,
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
        side_hydrographs: dict,
        edge_hydrographs: dict,
        h0: np.ndarray,
        hu0: np.ndarray,
        hv0: np.ndarray,
        n_mann_cell: Optional[np.ndarray],
        dt_fixed: float,
        dt_max: float,
        dt_initial: float = 0.0,
        reconstruction_mode: int,
        temporal_scheme: Any,
        gravity: float,
        k_mann: float,
        n_mann: float,
        cfl: float,
        h_min: float,
        max_inv_area: float,
        cfl_lambda_cap: float,
        momentum_cap_min_speed: float,
        momentum_cap_celerity_mult: float,
        depth_cap: float,
        max_rel_depth_increase: float,
        shallow_damping_depth: float,
        source_cfl_beta: float,
        source_max_substeps: int,
        source_rate_cap: float,
        source_depth_step_cap: float,
        source_true_subcycling: bool,
        source_imex_split: bool,
        enable_shallow_front_recon_fallback: bool,
        gpu_diag_sync_interval_steps: int,
        tiny_mode: int,
        tiny_wet_cell_threshold: int,
        degen_mode: int,
        front_flux_damping: float,
        open_bc_relaxation: float,
        bc_relax: np.ndarray,
        active_set_hysteresis: bool,
        # Baked mesh persistence
        gpkg_path: str = "",
        mesh_name: str = "",
        mesh_crs_wkt: str = "",
    ) -> Any:
        """Build and initialize."""
        b = backend_cls()

        bc_tp_init = bc_tp.copy()
        bc_vl_init = bc_vl.copy()
        if dynamic_bc and self._apply_timeseries_bc_values is not None:
            bc_tp_init, bc_vl_init = self._apply_timeseries_bc_values(
                bc_n0, bc_n1, bc_tp_init, bc_vl_init, side_hydrographs, 0.0, edge_hydrographs
            )
        if self._distribute_total_flow_to_unit_q is not None:
            bc_vl_init = self._distribute_total_flow_to_unit_q(
                bc_n0,
                bc_n1,
                bc_tp_init,
                bc_vl_init,
                bc_tp,
                side_hydrographs,
                edge_hydrographs,
            )

        # Shared helper (same logic as CLI headless runner)
        shared_build_mesh(
            b,
            node_x=node_x, node_y=node_y, node_z=node_z,
            cell_nodes=cell_nodes,
            cell_face_offsets=face_offsets,
            cell_face_nodes=face_nodes,
            bc_edge_node0=bc_n0, bc_edge_node1=bc_n1,
            bc_edge_type=bc_tp_init, bc_edge_val=bc_vl_init,
        )

        # Persist baked mesh blob if a GPKG path was provided
        if gpkg_path and b._mesh_h is not None:
            try:
                _mesh_name = str(mesh_name or "").strip()
                if not _mesh_name:
                    import datetime as _dt
                    _mesh_name = f"mesh_{_dt.datetime.now(_dt.timezone.utc).strftime('%Y%m%d_%H%M%S')}"
                baked_blob = b._mod.swe2d_serialize_mesh(b._mesh_h)
                info = b._mod.swe2d_mesh_info(b._mesh_h)
                persist_baked_mesh(
                    gpkg_path, _mesh_name, baked_blob,
                    n_nodes=info["n_nodes"],
                    n_cells=info["n_cells"],
                    n_edges=info["n_edges"],
                    crs_wkt=mesh_crs_wkt,
                )
            except Exception as exc:
                logger.warning("Failed to persist baked mesh: %s", exc)

        b.initialize(
            h0,
            hu0,
            hv0,
            g=float(gravity),
            k_mann=float(k_mann),
            n_mann=float(n_mann),
            n_mann_cell=n_mann_cell,
            cfl=float(cfl),
            h_min=float(h_min),
            dt_fixed=dt_fixed,
            dt_max=dt_max,
            dt_initial=dt_initial,
            max_inv_area=float(max_inv_area),
            cfl_lambda_cap=float(cfl_lambda_cap),
            momentum_cap_min_speed=float(momentum_cap_min_speed),
            momentum_cap_celerity_mult=float(momentum_cap_celerity_mult),
            depth_cap=float(depth_cap),
            max_rel_depth_increase=float(max_rel_depth_increase),
            shallow_damping_depth=float(shallow_damping_depth),
            source_cfl_beta=float(source_cfl_beta),
            source_max_substeps=int(source_max_substeps),
            source_rate_cap=float(source_rate_cap),
            source_depth_step_cap=float(source_depth_step_cap),
            source_true_subcycling=bool(source_true_subcycling),
            source_imex_split=bool(source_imex_split),
            enable_shallow_front_recon_fallback=bool(enable_shallow_front_recon_fallback),
            gpu_diag_sync_interval_steps=int(gpu_diag_sync_interval_steps),
            tiny_mode=int(tiny_mode),
            tiny_wet_cell_threshold=int(tiny_wet_cell_threshold),
            spatial_discretization=reconstruction_mode,
            temporal_scheme=temporal_scheme,
            degen_mode=int(degen_mode),
            front_flux_damping=float(front_flux_damping),
            open_bc_relaxation=float(open_bc_relaxation),
            active_set_hysteresis=bool(active_set_hysteresis),
        )
        if bc_relax is not None and bc_relax.size > 0:
            b.set_boundary_relaxation(bc_n0, bc_n1, bc_relax)
        return b
