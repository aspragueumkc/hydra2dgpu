#!/usr/bin/env python3
"""Run input assembly builder for SWE2D workbench.

Phase 4 goal: extract solver input data assembly from the dialog `_on_run`
method into a small, testable module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class SWE2DRunInputData:
    """Structured input payload needed by the run execution body."""

    node_x: np.ndarray
    node_y: np.ndarray
    node_z: np.ndarray
    cell_nodes: np.ndarray
    face_offsets: Optional[np.ndarray]
    face_nodes: Optional[np.ndarray]
    bc_n0: np.ndarray
    bc_n1: np.ndarray
    bc_tp: np.ndarray
    bc_vl: np.ndarray
    bc_relax: np.ndarray
    side_hydrographs: Dict[str, object]
    edge_hydrographs: Dict[Tuple[int, int], object]
    edge_group_overrides: Dict[Tuple[int, int], object]
    h0: np.ndarray
    hu0: np.ndarray
    hv0: np.ndarray
    n_mann_cell: Optional[np.ndarray]


class SWE2DRunDataBuilder:
    """Builds run input data by invoking dialog-provided callbacks."""

    def __init__(
        self,
        get_mesh_data_callback: Callable[[], Optional[Dict[str, Any]]],
        collect_boundary_arrays_callback: Callable[[], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        build_side_hydrographs_callback: Callable[[], Dict[str, object]],
        collect_bc_layer_hydrographs_callback: Callable[[np.ndarray, np.ndarray], Dict[Tuple[int, int], object]],
        collect_bc_layer_edge_groups_callback: Callable[[np.ndarray, np.ndarray], Dict[Tuple[int, int], object]],
        initial_state_callback: Callable[..., Tuple[np.ndarray, np.ndarray, np.ndarray]],
        build_spatial_manning_array_callback: Callable[[], Optional[np.ndarray]],
        update_unit_system_callback: Callable[[], None],
    ):
        self._get_mesh_data_callback = get_mesh_data_callback
        self._collect_boundary_arrays_callback = collect_boundary_arrays_callback
        self._build_side_hydrographs_callback = build_side_hydrographs_callback
        self._collect_bc_layer_hydrographs_callback = collect_bc_layer_hydrographs_callback
        self._collect_bc_layer_edge_groups_callback = collect_bc_layer_edge_groups_callback
        self._initial_state_callback = initial_state_callback
        self._build_spatial_manning_array_callback = build_spatial_manning_array_callback
        self._update_unit_system_callback = update_unit_system_callback

    def build(self) -> SWE2DRunInputData:
        """build."""
        mesh_data = self._get_mesh_data_callback()
        if not isinstance(mesh_data, dict):
            raise RuntimeError("Run input assembly failed: mesh data is not available.")

        try:
            node_x = mesh_data["node_x"]
            node_y = mesh_data["node_y"]
            node_z = mesh_data["node_z"]
            cell_nodes = mesh_data["cell_nodes"]
        except KeyError as exc:
            raise RuntimeError(f"Run input assembly failed: missing mesh field {exc}.") from exc

        face_offsets = mesh_data.get("cell_face_offsets")
        face_nodes = mesh_data.get("cell_face_nodes")

        bc_n0, bc_n1, bc_tp, bc_vl, bc_relax = self._collect_boundary_arrays_callback()
        side_hydrographs = self._build_side_hydrographs_callback()
        edge_hydrographs = self._collect_bc_layer_hydrographs_callback(bc_n0, bc_n1)
        edge_group_overrides = self._collect_bc_layer_edge_groups_callback(bc_n0, bc_n1)
        h0, hu0, hv0 = self._initial_state_callback(bc_n0=bc_n0, bc_n1=bc_n1, bc_tp=bc_tp)
        n_mann_cell = self._build_spatial_manning_array_callback()
        self._update_unit_system_callback()

        return SWE2DRunInputData(
            node_x=node_x,
            node_y=node_y,
            node_z=node_z,
            cell_nodes=cell_nodes,
            face_offsets=face_offsets,
            face_nodes=face_nodes,
            bc_n0=bc_n0,
            bc_n1=bc_n1,
            bc_tp=bc_tp,
            bc_vl=bc_vl,
            bc_relax=bc_relax,
            side_hydrographs=side_hydrographs,
            edge_hydrographs=edge_hydrographs,
            edge_group_overrides=edge_group_overrides,
            h0=h0,
            hu0=hu0,
            hv0=hv0,
            n_mann_cell=n_mann_cell,
        )
