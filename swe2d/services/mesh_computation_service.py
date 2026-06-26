"""Mesh computation service — pure-Python, zero-Qt.

Extracted from ``studio_dialog.py`` to respect the MVP service-layer boundary.
All functions accept explicit ``mesh_data`` dicts; no widget access.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def mesh_boundary_edges(mesh_data: Optional[Dict[str, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (edge_n0, edge_n1) for boundary (single-adjacent) edges."""
    from swe2d.boundary_and_forcing.boundary_runtime_logic import mesh_boundary_edges as _logic
    return _logic(mesh_data)


def default_bc_for_edges(
    mesh_data: Dict[str, np.ndarray],
    edge_n0: np.ndarray,
    edge_n1: np.ndarray,
    default_bc_type: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Classify boundary edges and return (bc_type, bc_val) arrays.

    Parameters
    ----------
    mesh_data : dict
        Mesh data dict with ``node_x``, ``node_y``.
    edge_n0, edge_n1 : np.ndarray
        Boundary edge node indices.
    default_bc_type : int
        Default BC type to assign (from UI combo currentData()).
    """
    from swe2d.boundary_and_forcing.boundary_runtime_logic import classify_boundary_edges as _classify
    node_x = mesh_data["node_x"]
    node_y = mesh_data["node_y"]
    side_idx = _classify(edge_n0, edge_n1, node_x, node_y)
    bc_type = np.zeros(edge_n0.shape[0], dtype=np.int32)
    bc_val = np.zeros(edge_n0.shape[0], dtype=np.float64)
    bc_type[:] = default_bc_type
    return bc_type, bc_val


def mesh_cell_centroids(mesh_data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute cell centroids (cx, cy) from mesh topology."""
    from swe2d.mesh.mesh_runtime_logic import mesh_cell_centroids as _logic
    return _logic(mesh_data)


def mesh_cell_areas(mesh_data: Dict[str, np.ndarray]) -> np.ndarray:
    """Compute per-cell areas from mesh topology."""
    from swe2d.mesh.mesh_runtime_logic import mesh_cell_areas as _logic
    return _logic(mesh_data)


def mesh_cell_min_bed(mesh_data: Dict[str, np.ndarray]) -> np.ndarray:
    """Return per-cell minimum bed elevation."""
    from swe2d.mesh.mesh_runtime_logic import mesh_cell_min_bed as _logic
    return _logic(mesh_data)


def mesh_cell_solver_bed(mesh_data: Dict[str, np.ndarray]) -> np.ndarray:
    """Return per-cell solver bed elevation."""
    from swe2d.mesh.mesh_runtime_logic import mesh_cell_solver_bed as _logic
    return _logic(mesh_data)


def boundary_buffer_cells(mesh_data: Optional[Dict[str, np.ndarray]], n_rings: int) -> np.ndarray:
    """Return cell indices within *n_rings* of the mesh boundary."""
    from swe2d.mesh.mesh_runtime_logic import boundary_buffer_cells as _logic
    return _logic(mesh_data, n_rings)
