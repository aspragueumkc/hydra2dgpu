from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np


def mesh_boundary_edges(mesh_data: Optional[Dict[str, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
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
    default_bc_for_edges_fn: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
    apply_bc_layer_overrides_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
    log_fn: Callable[[str], None],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if mesh_data is None:
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float64),
        )

    edge_n0, edge_n1 = mesh_boundary_edges_fn()
    if edge_n0.size == 0:
        log_fn("No boundary edges detected in mesh.")
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float64),
        )

    bc_type, bc_val = default_bc_for_edges_fn(edge_n0, edge_n1)
    bc_type, bc_val = apply_bc_layer_overrides_fn(edge_n0, edge_n1, bc_type, bc_val)
    return edge_n0, edge_n1, bc_type, bc_val
