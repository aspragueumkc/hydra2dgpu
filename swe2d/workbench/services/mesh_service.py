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
