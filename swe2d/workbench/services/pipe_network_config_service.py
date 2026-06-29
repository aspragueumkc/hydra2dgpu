from __future__ import annotations

"""Service for loading and validating pipe-network configuration from GeoPackage layers."""

from typing import Callable, Optional

import numpy as np

from swe2d.workbench.services.pipe_network_service import build_pipe_network_config


def build_pipe_network_config_from_widgets(
    *,
    mesh_data: Optional[dict],
    have_qgis_core: bool,
    pipe_network_config_cls,
    node_layer,
    link_layer,
    inlet_layer,
    node_inlet_layer,
    cell_min_bed: Optional[np.ndarray],
    nearest_cell_fn: Optional[Callable[[float, float], int]],
    gravity: float,
    solver_mode_name: str,
    solver_mode,
    coupling_substeps: int,
    max_coupling_substeps: int,
    gpu_method: str,
    head_deadband: float,
    dynamic_relaxation: float,
    adaptive_depth_fraction: float,
    adaptive_wave_courant: float,
    implicit_iters: int,
    implicit_relax: float,
    log_fn: Optional[Callable[[str], None]] = None,
):
    """Build a PipeNetworkConfig from resolved values.

    Contains the guard logic and config-dict assembly that was previously
    inline in ``_build_pipe_network_config``.  All widget access lives in
    the caller (the View layer); this function is pure-service with zero Qt.
    """
    if (
        mesh_data is None
        or not have_qgis_core
        or pipe_network_config_cls is None
        or node_layer is None
        or link_layer is None
    ):
        return None

    return build_pipe_network_config(
        mesh_data=mesh_data,
        node_layer=node_layer,
        link_layer=link_layer,
        inlet_layer=inlet_layer,
        node_inlet_layer=node_inlet_layer,
        cell_min_bed=cell_min_bed,
        nearest_cell_fn=nearest_cell_fn,
        gravity=gravity,
        config={
            "solver_mode": solver_mode,
            "solver_mode_name": solver_mode_name,
            "coupling_substeps": coupling_substeps,
            "max_coupling_substeps": max_coupling_substeps,
            "gpu_method": gpu_method,
            "head_deadband": head_deadband,
            "dynamic_relaxation": dynamic_relaxation,
            "adaptive_depth_fraction": adaptive_depth_fraction,
            "adaptive_wave_courant": adaptive_wave_courant,
            "implicit_iters": implicit_iters,
            "implicit_relax": implicit_relax,
        },
        log_fn=log_fn,
    )
