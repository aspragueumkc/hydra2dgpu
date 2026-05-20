from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


def mesh_cell_centroids(mesh_data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    node_x = mesh_data["node_x"]
    node_y = mesh_data["node_y"]

    if "cell_face_offsets" in mesh_data and "cell_face_nodes" in mesh_data:
        offs = mesh_data["cell_face_offsets"].astype(np.int32)
        faces = mesh_data["cell_face_nodes"].astype(np.int32)
        cx = np.zeros(offs.size - 1, dtype=np.float64)
        cy = np.zeros(offs.size - 1, dtype=np.float64)
        for i in range(offs.size - 1):
            s = int(offs[i])
            e = int(offs[i + 1])
            ids = faces[s:e]
            if ids.size == 0:
                continue
            cx[i] = float(np.mean(node_x[ids]))
            cy[i] = float(np.mean(node_y[ids]))
        return cx, cy

    tris = mesh_data["cell_nodes"].reshape((-1, 3))
    return node_x[tris].mean(axis=1), node_y[tris].mean(axis=1)


def mesh_cell_areas(mesh_data: Dict[str, np.ndarray]) -> np.ndarray:
    node_x = mesh_data["node_x"]
    node_y = mesh_data["node_y"]

    if "cell_face_offsets" in mesh_data and "cell_face_nodes" in mesh_data:
        offs = mesh_data["cell_face_offsets"].astype(np.int32)
        faces = mesh_data["cell_face_nodes"].astype(np.int32)
        area = np.zeros(offs.size - 1, dtype=np.float64)
        for i in range(offs.size - 1):
            s = int(offs[i])
            e = int(offs[i + 1])
            ids = faces[s:e]
            if ids.size < 3:
                continue
            x = node_x[ids]
            y = node_y[ids]
            area[i] = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
        return area

    tris = mesh_data["cell_nodes"].reshape((-1, 3)).astype(np.int32)
    x0 = node_x[tris[:, 0]]
    y0 = node_y[tris[:, 0]]
    x1 = node_x[tris[:, 1]]
    y1 = node_y[tris[:, 1]]
    x2 = node_x[tris[:, 2]]
    y2 = node_y[tris[:, 2]]
    return 0.5 * np.abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))


def mesh_cell_min_bed(mesh_data: Dict[str, np.ndarray]) -> np.ndarray:
    node_z = mesh_data["node_z"]
    if "cell_face_offsets" in mesh_data and "cell_face_nodes" in mesh_data:
        offs = mesh_data["cell_face_offsets"].astype(np.int32)
        faces = mesh_data["cell_face_nodes"].astype(np.int32)
        out = np.zeros(offs.size - 1, dtype=np.float64)
        for i in range(offs.size - 1):
            s = int(offs[i])
            e = int(offs[i + 1])
            ids = faces[s:e]
            if ids.size:
                out[i] = float(np.min(node_z[ids]))
        return out
    tri = mesh_data["cell_nodes"].reshape(-1, 3).astype(np.int32)
    return np.min(node_z[tri], axis=1).astype(np.float64)


def inflow_adjacent_cells(
    mesh_data: Dict[str, np.ndarray],
    bc_n0: np.ndarray,
    bc_n1: np.ndarray,
    bc_tp: np.ndarray,
    inflow_types: Sequence[int] = (2, 6, 102),
) -> np.ndarray:
    inflow_mask = np.isin(bc_tp.astype(np.int32), list(inflow_types))
    if not np.any(inflow_mask):
        return np.empty(0, dtype=np.int32)

    inflow_n0 = set(int(v) for v in bc_n0[inflow_mask])
    inflow_n1 = set(int(v) for v in bc_n1[inflow_mask])
    inflow_nodes = inflow_n0 | inflow_n1

    hit: List[int] = []
    if "cell_face_offsets" in mesh_data and "cell_face_nodes" in mesh_data:
        offs = mesh_data["cell_face_offsets"].astype(np.int32)
        faces = mesh_data["cell_face_nodes"].astype(np.int32)
        for ci in range(offs.size - 1):
            s = int(offs[ci])
            e = int(offs[ci + 1])
            poly = faces[s:e]
            for k in range(poly.size):
                a = int(poly[k])
                b = int(poly[(k + 1) % poly.size])
                key_a = min(a, b)
                key_b = max(a, b)
                if key_a in inflow_nodes and key_b in inflow_nodes:
                    hit.append(ci)
                    break
    else:
        tris = mesh_data["cell_nodes"].reshape((-1, 3)).astype(np.int32)
        for ci, tri in enumerate(tris):
            for k in range(3):
                a = int(tri[k])
                b = int(tri[(k + 1) % 3])
                key_a = min(a, b)
                key_b = max(a, b)
                if key_a in inflow_nodes and key_b in inflow_nodes:
                    hit.append(ci)
                    break
    return np.asarray(hit, dtype=np.int32)


def boundary_buffer_cells(mesh_data: Optional[Dict[str, np.ndarray]], n_rings: int) -> np.ndarray:
    if mesh_data is None or int(n_rings) <= 0:
        return np.empty(0, dtype=np.int32)

    edge_cells: Dict[Tuple[int, int], List[int]] = {}
    if "cell_face_offsets" in mesh_data and "cell_face_nodes" in mesh_data:
        offs = mesh_data["cell_face_offsets"].astype(np.int32)
        faces = mesh_data["cell_face_nodes"].astype(np.int32)
        n_cells = int(offs.size - 1)
        for ci in range(n_cells):
            s = int(offs[ci])
            e = int(offs[ci + 1])
            poly = faces[s:e]
            for k in range(poly.size):
                a = int(poly[k])
                b = int(poly[(k + 1) % poly.size])
                key = (min(a, b), max(a, b))
                edge_cells.setdefault(key, []).append(ci)
    else:
        tris = mesh_data["cell_nodes"].reshape((-1, 3)).astype(np.int32)
        n_cells = int(tris.shape[0])
        for ci, tri in enumerate(tris):
            for k in range(3):
                a = int(tri[k])
                b = int(tri[(k + 1) % 3])
                key = (min(a, b), max(a, b))
                edge_cells.setdefault(key, []).append(ci)

    neighbors: List[set] = [set() for _ in range(n_cells)]
    ring = set()
    for owners in edge_cells.values():
        if len(owners) == 1:
            ring.add(int(owners[0]))
        elif len(owners) == 2:
            c0 = int(owners[0])
            c1 = int(owners[1])
            neighbors[c0].add(c1)
            neighbors[c1].add(c0)

    if not ring:
        return np.empty(0, dtype=np.int32)

    selected = set(ring)
    for _ in range(1, int(n_rings)):
        nxt = set()
        for c in ring:
            nxt.update(neighbors[c])
        nxt.difference_update(selected)
        if not nxt:
            break
        selected.update(nxt)
        ring = nxt

    return np.asarray(sorted(selected), dtype=np.int32)


def initial_state(
    *,
    mesh_data: Dict[str, np.ndarray],
    mode: str,
    initial_depth: float,
    initial_wse: float,
    h_min: float,
    bc_n0: Optional[np.ndarray] = None,
    bc_n1: Optional[np.ndarray] = None,
    bc_tp: Optional[np.ndarray] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cell_x, _ = mesh_cell_centroids(mesh_data)
    h0 = np.zeros_like(cell_x, dtype=np.float64)

    if mode == "uniform_depth":
        h0[:] = max(0.0, float(initial_depth))
    elif mode == "uniform_wse":
        bed = mesh_cell_min_bed(mesh_data).astype(np.float64)
        h0 = np.maximum(0.0, float(initial_wse) - bed)
    elif mode == "dry" and bc_n0 is not None and bc_n1 is not None and bc_tp is not None:
        prime_depth = max(float(h_min) * 100.0, 1.0e-4)
        adj = inflow_adjacent_cells(mesh_data, bc_n0, bc_n1, bc_tp)
        if adj.size > 0:
            h0[adj] = prime_depth
            if log_fn is not None:
                log_fn(
                    f"Dry start: primed {adj.size} inflow-adjacent cell(s) with h={prime_depth:.2e} m "
                    "to enable boundary-driven wetting."
                )

    hu0 = np.zeros_like(h0)
    hv0 = np.zeros_like(h0)
    return h0, hu0, hv0
