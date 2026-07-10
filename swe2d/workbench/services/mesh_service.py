"""Pure-Python, Qt-free service for SWE2D mesh computation.

Provides numpy-heavy mesh computation extracted from
SWE2DWorkbenchDialog methods — zero Qt imports, fully testable
without QApplication.
"""

from __future__ import annotations

from typing import Tuple

def edge_lengths(node_x: np.ndarray, node_y: np.ndarray, n0: np.ndarray, n1: np.ndarray) -> np.ndarray:
    """Compute edge lengths between node pairs."""
    return np.hypot(node_x[n1] - node_x[n0], node_y[n1] - node_y[n0]).astype(np.float64)


def mesh_bounds(node_x: np.ndarray, node_y: np.ndarray):
    """Return (xmin, xmax, ymin, ymax) of mesh nodes."""
    return (
        float(np.min(node_x)) if node_x.size else 0.0,
        float(np.max(node_x)) if node_x.size else 0.0,
        float(np.min(node_y)) if node_y.size else 0.0,
        float(np.max(node_y)) if node_y.size else 0.0,
    )

import numpy as np


def build_node_coords(
    node_x: np.ndarray,
    node_y: np.ndarray,
) -> np.ndarray:
    """Build (N, 2) node coordinate array from separate x/y arrays.

    Parameters
    ----------
    node_x : (N,) ndarray
        X-coordinates of mesh nodes.
    node_y : (N,) ndarray
        Y-coordinates of mesh nodes.

    Returns
    -------
    node_coords : (N, 2) ndarray
        Column-stacked (x, y) coordinate array.
    """
    return np.column_stack(
        [np.asarray(node_x, dtype=np.float64), np.asarray(node_y, dtype=np.float64)]
    )


# ---------------------------------------------------------------------------
# Raster sampling
# ---------------------------------------------------------------------------


def assign_node_z_from_terrain(
    node_coords: np.ndarray,
    raster_data: np.ndarray,
    raster_transform: Tuple[float, ...],
    default_z: float = 0.0,
) -> np.ndarray:
    """Sample raster at node coordinates using nearest-neighbor.

    Parameters
    ----------
    node_coords : (N, 2) ndarray
        (x, y) coordinates of mesh nodes.
    raster_data : (H, W) ndarray
        Raster band values (rows = Y, columns = X).
    raster_transform : tuple of 6 floats
        GDAL-style geotransform:
        (x_origin, dx, x_rot, y_origin, y_rot, dy).
    default_z : float
        Value assigned to nodes whose raster coordinates fall outside
        the raster extent.

    Returns
    -------
    node_z : (N,) ndarray
        Sampled elevation per node.
    """
    n = node_coords.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float64)

    if raster_data is None or raster_data.size == 0:
        return np.full(n, default_z, dtype=np.float64)

    ox, dx, _, oy, _, dy = raster_transform[:6]
    x = np.asarray(node_coords[:, 0], dtype=np.float64)
    y = np.asarray(node_coords[:, 1], dtype=np.float64)

    col_f = (x - ox) / dx - 0.5 if abs(dx) > 1e-30 else np.full(n, -1.0)
    row_f = (y - oy) / dy - 0.5 if abs(dy) > 1e-30 else np.full(n, -1.0)

    col = np.round(col_f).astype(np.int32)
    row = np.round(row_f).astype(np.int32)

    nrows, ncols = raster_data.shape
    inside = (col >= 0) & (col < ncols) & (row >= 0) & (row < nrows)

    node_z = np.full(n, default_z, dtype=np.float64)
    if np.any(inside):
        node_z[inside] = raster_data[row[inside], col[inside]].astype(np.float64, copy=False)

    return node_z


def apply_cell_permutation(
    mesh_data: dict, cell_perm: np.ndarray
) -> dict:
    """Apply an RCMK cell permutation to mesh_data in place and return it.

    Reorders ``cell_nodes`` (1D flat fan triangulation), ``cell_face_offsets``,
    and ``cell_face_nodes`` to match *cell_perm* order.  Works for mixed
    polygon types (tri, quad, etc.) by using ``cell_face_offsets`` to
    determine per-cell vertex counts.  Mutates *mesh_data* in place and
    returns it for convenience.
    """
    n_cells = int(cell_perm.size)
    if n_cells <= 0:
        return mesh_data

    cfo = mesh_data.get("cell_face_offsets")
    cfn = mesh_data.get("cell_face_nodes")

    if cfo is not None and cfn is not None:
        # ── Face data permutation ──────────────────────────────────────
        old_offsets = np.asarray(cfo, dtype=np.int32).ravel()
        old_nodes = np.asarray(cfn, dtype=np.int32).ravel()
        new_offsets = np.empty(n_cells + 1, dtype=np.int32)
        new_offsets[0] = old_offsets[0]
        for ci in range(n_cells):
            orig = int(cell_perm[ci])
            s = int(old_offsets[orig])
            e = int(old_offsets[orig + 1])
            new_offsets[ci + 1] = new_offsets[ci] + (e - s)
        nfn = int(new_offsets[-1])
        new_nodes = np.empty(nfn, dtype=np.int32)
        for ci in range(n_cells):
            orig = int(cell_perm[ci])
            s = int(old_offsets[orig])
            e = int(old_offsets[orig + 1])
            ds = int(new_offsets[ci])
            new_nodes[ds:ds + (e - s)] = old_nodes[s:e]
        mesh_data["cell_face_offsets"] = new_offsets
        mesh_data["cell_face_nodes"] = new_nodes

        # ── Regenerate fan triangulation from permuted face data ───────
        tri_list: list = []
        for ci in range(n_cells):
            s = int(new_offsets[ci])
            e = int(new_offsets[ci + 1])
            ring = new_nodes[s:e]
            # Fan from first vertex: (v0, v1, v2), (v0, v2, v3), ...
            for k in range(1, int(ring.size) - 1):
                tri_list.append(int(ring[0]))
                tri_list.append(int(ring[k]))
                tri_list.append(int(ring[k + 1]))
        mesh_data["cell_nodes"] = np.asarray(tri_list, dtype=np.int32)

    else:
        # ── No face data — pure triangle mesh ──────────────────────────
        cn = mesh_data.get("cell_nodes")
        if cn is not None and cn.size > 0:
            if cn.ndim == 1:
                reshaped = cn.reshape(-1, 3)
                if reshaped.shape[0] == n_cells:
                    mesh_data["cell_nodes"] = reshaped[cell_perm].ravel()
            else:
                if cn.shape[0] == n_cells:
                    mesh_data["cell_nodes"] = cn[cell_perm]

    return mesh_data


def classify_boundary_edges(
    node_x: np.ndarray, node_y: np.ndarray,
    bc_n0: np.ndarray, bc_n1: np.ndarray,
):
    """Return (edge_lengths, side_index, side_names) for boundary edges.

    side_index: 0=left, 1=right, 2=bottom, 3=top (nearest mesh boundary).
    """
    edge_len = edge_lengths(node_x, node_y, bc_n0, bc_n1)
    xmin, xmax, ymin, ymax = mesh_bounds(node_x, node_y)
    mx = 0.5 * (node_x[bc_n0] + node_x[bc_n1]) if bc_n0.size else np.empty(0, dtype=np.float64)
    my = 0.5 * (node_y[bc_n0] + node_y[bc_n1]) if bc_n0.size else np.empty(0, dtype=np.float64)
    if bc_n0.size:
        d = np.vstack([
            np.abs(mx - xmin), np.abs(mx - xmax),
            np.abs(my - ymin), np.abs(my - ymax),
        ])
        side_idx = np.argmin(d, axis=0)
    else:
        side_idx = np.empty(0, dtype=np.int32)
    return edge_len, side_idx, ["left", "right", "bottom", "top"]
