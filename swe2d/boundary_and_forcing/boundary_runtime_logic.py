from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np


def mesh_boundary_edges(mesh_data: Optional[Dict[str, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    mesh boundary edges.

    Parameters
    ----------
    mesh_data : Optional[Dict[str, np.ndarray]]
        Description of mesh_data.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
    """
    if mesh_data is None:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    edge_count: Dict[Tuple[int, int], int] = {}
    edge_oriented: Dict[Tuple[int, int], Tuple[int, int]] = {}

    if "cell_face_offsets" in mesh_data and "cell_face_nodes" in mesh_data:
        offsets = mesh_data["cell_face_offsets"].astype(np.int32)
        faces = mesh_data["cell_face_nodes"].astype(np.int32)
        for i in range(offsets.size - 1):
            s = int(offsets[i])
            e = int(offsets[i + 1])
            poly = faces[s:e]
            if poly.size < 3:
                continue
            for k in range(poly.size):
                a = int(poly[k])
                b = int(poly[(k + 1) % poly.size])
                key = (a, b) if a < b else (b, a)
                edge_count[key] = edge_count.get(key, 0) + 1
                if key not in edge_oriented:
                    edge_oriented[key] = (a, b)
    else:
        tris = mesh_data["cell_nodes"].reshape((-1, 3)).astype(np.int32)
        for tri in tris:
            a0, a1, a2 = int(tri[0]), int(tri[1]), int(tri[2])
            for a, b in ((a0, a1), (a1, a2), (a2, a0)):
                key = (a, b) if a < b else (b, a)
                edge_count[key] = edge_count.get(key, 0) + 1
                if key not in edge_oriented:
                    edge_oriented[key] = (a, b)

    n0 = []
    n1 = []
    for key, cnt in edge_count.items():
        if cnt == 1:
            a, b = edge_oriented[key]
            n0.append(a)
            n1.append(b)

    if not n0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    return np.asarray(n0, dtype=np.int32), np.asarray(n1, dtype=np.int32)


def collect_boundary_arrays(
    *,
    mesh_data: Optional[Dict[str, np.ndarray]],
    mesh_boundary_edges_fn: Callable[[], Tuple[np.ndarray, np.ndarray]],
    default_bc_type: int = 0,
    apply_bc_layer_overrides_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
    log_fn: Callable[[str], None],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect boundary condition arrays from mesh geometry and apply defaults.

    Parameters
    ----------
    mesh_data : Optional[Dict[str, np.ndarray]]
        Mesh data dictionary containing node coordinates and connectivity.
    mesh_boundary_edges_fn : Callable[[], Tuple[np.ndarray, np.ndarray]]
        Function returning boundary edge node indices (edge_n0, edge_n1).
    default_bc_type : int
        Default boundary condition type to apply (0=wall, 1=flow, etc.).
    apply_bc_layer_overrides_fn : Callable
        Function to apply BC layer overrides from QGIS.
    log_fn : Callable[[str], None]
        Logging callback.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        edge_n0, edge_n1, bc_type, bc_val arrays.
    """
    if mesh_data is None:
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float64),
        )

    from swe2d.services.mesh_computation_service import default_bc_for_edges as _compute_default_bc

    edge_n0, edge_n1 = mesh_boundary_edges_fn()
    if edge_n0.size == 0:
        log_fn("No boundary edges detected in mesh.")
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float64),
        )

    bc_type, bc_val = _compute_default_bc(mesh_data, edge_n0, edge_n1, default_bc_type=default_bc_type)
    bc_type, bc_val = apply_bc_layer_overrides_fn(edge_n0, edge_n1, bc_type, bc_val)
    return edge_n0, edge_n1, bc_type, bc_val


def classify_boundary_edges(
    edge_n0: np.ndarray,
    edge_n1: np.ndarray,
    node_x: np.ndarray,
    node_y: np.ndarray,
) -> np.ndarray:
    """Classify each boundary edge into a side index (0=left,1=right,2=bottom,3=top).

    Pure geometry computation — no widget access, no Qt.  Called by View
    methods that need to map boundary edges to side-specific widget values.
    """
    xmin = float(np.min(node_x))
    xmax = float(np.max(node_x))
    ymin = float(np.min(node_y))
    ymax = float(np.max(node_y))
    mx = 0.5 * (node_x[edge_n0] + node_x[edge_n1])
    my = 0.5 * (node_y[edge_n0] + node_y[edge_n1])
    d_left = np.abs(mx - xmin)
    d_right = np.abs(mx - xmax)
    d_bottom = np.abs(my - ymin)
    d_top = np.abs(my - ymax)
    d = np.vstack([d_left, d_right, d_bottom, d_top])
    side_idx = np.argmin(d, axis=0)
    return side_idx
