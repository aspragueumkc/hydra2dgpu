"""Terrain/surface assignment - raster sampling and PointZ interpolation.

Pure Python + numpy - zero Qt.
"""
from __future__ import annotations

import numpy as np


def sample_raster_at_nodes(
    node_x: np.ndarray,
    node_y: np.ndarray,
    raster_data: np.ndarray,
    geo_transform: tuple[float, ...],
    default_z: float = 0.0,
) -> np.ndarray:
    """Sample a raster at node coordinates using GDAL-style geotransform."""
    origin_x, pixel_width, _, origin_y, _, pixel_height = geo_transform
    cols = ((node_x - origin_x) / pixel_width).astype(np.int32)
    rows = ((node_y - origin_y) / pixel_height).astype(np.int32)

    nrows, ncols = raster_data.shape
    valid = (cols >= 0) & (cols < ncols) & (rows >= 0) & (rows < nrows)
    z = np.full(len(node_x), default_z, dtype=raster_data.dtype)
    z[valid] = raster_data[rows[valid], cols[valid]]
    return z


def idw_interpolate_points(
    node_x: np.ndarray,
    node_y: np.ndarray,
    point_x: np.ndarray,
    point_y: np.ndarray,
    point_z: np.ndarray,
    k: int = 4,
    power: float = 2.0,
    default_z: float = 0.0,
) -> np.ndarray:
    """Inverse-distance-weighted interpolation of a PointZ layer at mesh nodes.

    For each mesh node, finds the ``k`` nearest PointZ points and
    interpolates z as the IDW (1/d^power) weighted average of their z
    values. Falls back to the colocated source z when the nearest point
    is at zero distance.

    Args:
        node_x, node_y: Mesh node coordinate arrays.
        point_x, point_y: PointZ source coordinates.
        point_z: PointZ source z-coordinates.
        k: Number of nearest neighbours to use (default 4).
        power: Distance exponent (default 2.0 = classic IDW).
        default_z: Fallback z for degenerate cases.

    Returns:
        Array of interpolated z-values, same length as node_x.
    """
    node_x = np.asarray(node_x, dtype=np.float64)
    node_y = np.asarray(node_y, dtype=np.float64)
    point_x = np.asarray(point_x, dtype=np.float64)
    point_y = np.asarray(point_y, dtype=np.float64)
    point_z = np.asarray(point_z, dtype=np.float64)

    n_nodes = node_x.size
    n_pts = point_x.size
    if n_pts == 0:
        return np.full(n_nodes, default_z, dtype=np.float64)

    k_eff = min(k, n_pts)
    out = np.empty(n_nodes, dtype=np.float64)

    # Process mesh nodes in chunks so the (chunk x n_pts) distance matrix
    # stays memory-bounded for large meshes.
    chunk = max(1, min(8192, n_nodes))
    for start in range(0, n_nodes, chunk):
        end = min(start + chunk, n_nodes)
        nx = node_x[start:end][:, None]
        ny = node_y[start:end][:, None]
        dx = nx - point_x[None, :]
        dy = ny - point_y[None, :]
        dist2 = dx * dx + dy * dy

        # k smallest distances per mesh node
        idx_part = np.argpartition(dist2, k_eff - 1, axis=1)[:, :k_eff]
        rows = np.arange(idx_part.shape[0])[:, None]
        nn_d2 = dist2[rows, idx_part]
        nn_z = point_z[idx_part]

        # Handle zero-distance hits (mesh node lies on a source point).
        zero = nn_d2[:, 0] == 0.0
        if np.any(zero):
            chunk_out = out[start:end]
            chunk_out[zero] = nn_z[zero, 0]
            out[start:end] = chunk_out

        # Remaining nodes: classic IDW with 1/d^power.
        rest = ~zero
        if np.any(rest):
            weights = 1.0 / np.power(np.maximum(nn_d2[rest], 1e-30), power)
            wsum = weights.sum(axis=1)
            chunk_out = out[start:end]
            chunk_out[rest] = (weights * nn_z[rest]).sum(axis=1) / wsum
            out[start:end] = chunk_out

    return out


def assign_node_z_from_layer_features(
    node_z: np.ndarray,
    features: list[dict],
) -> int:
    """Update node_z array from feature attribute dicts.

    Args:
        node_z: In-memory node elevation array (modified in place).
        features: List of dicts with ``node_id`` (int) and ``bed_z`` (float).

    Returns:
        Number of nodes updated.
    """
    updated = 0
    for feat in features:
        nid = int(feat["node_id"])
        if 0 <= nid < len(node_z):
            node_z[nid] = float(feat["bed_z"])
            updated += 1
    return updated
