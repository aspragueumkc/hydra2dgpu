"""Terrain/surface assignment — raster sampling and layer-feature node_z import.

Pure Python + numpy — zero Qt.
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
    """Sample a raster at node coordinates using GDAL-style geotransform.

    Args:
        node_x, node_y: Coordinate arrays in raster CRS units.
        raster_data: 2D numpy array (rows x cols), GDAL row-major ordering.
        geo_transform: 6-element GDAL geotransform (origin_x, pixel_width,
                       rotation_x, origin_y, rotation_y, pixel_height).
        default_z: Value to use for out-of-bounds nodes.

    Returns:
        Array of sampled z-values, same length as node_x.
    """
    origin_x, pixel_width, _, origin_y, _, pixel_height = geo_transform
    cols = ((node_x - origin_x) / pixel_width).astype(np.int32)
    rows = ((node_y - origin_y) / pixel_height).astype(np.int32)

    nrows, ncols = raster_data.shape
    valid = (cols >= 0) & (cols < ncols) & (rows >= 0) & (rows < nrows)
    z = np.full(len(node_x), default_z, dtype=raster_data.dtype)
    z[valid] = raster_data[rows[valid], cols[valid]]
    return z


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
