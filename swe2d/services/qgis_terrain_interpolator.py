"""QGIS-side terrain interpolation adapters.

Wraps ``QgsIDWInterpolator`` (for PointZ layers) and
``QgsRasterDataProvider.identify()`` (for rasters, GDAL underneath) so
the workbench can use QGIS-native interpolation when available.

CLI path uses the pure-numpy functions in
:mod:`swe2d.services.terrain_assignment_service` instead — both modules
share the same input/output contract:

    interpolate_at_node_coords(node_x, node_y, source_layer) -> z_array

where ``source_layer`` is a QGIS layer (raster or vector with Z) in the
GUI, or an ``ElevationSource``-shaped object in the CLI.
"""
from __future__ import annotations

import numpy as np

from qgis.core import (
    QgsGeometry,
    QgsPointXY,
    QgsRasterDataProvider,
    QgsVectorLayer,
)
from qgis.analysis import QgsInterpolator, QgsIDWInterpolator


# ── Vector (PointZ) interpolation via QGIS IDW ─────────────────────────

def interpolate_point_layer_idw(
    node_x: np.ndarray,
    node_y: np.ndarray,
    layer: QgsVectorLayer,
    *,
    distance_coefficient: float = 2.0,
    point_distance_threshold: float = 1.0e7,  # ignore break lines for IDW
) -> np.ndarray:
    """IDW interpolation of a PointZ vector layer at mesh node coords.

    Uses ``QgsIDWInterpolator`` (QGIS DualEdge triangulation under the
    hood, GDAL/GEOS-backed). Returns ``np.nan`` for nodes outside the
    convex hull / point support.

    Args:
        node_x, node_y: Mesh node coordinates (float64, shape ``(n,)``).
        layer: A QGIS vector layer with Z dimension or a z-attribute field.
        distance_coefficient: IDW distance exponent (default 2.0).
        point_distance_threshold: Used by QGIS interpolation engine.

    Returns:
        Float64 array of length ``n`` with interpolated z values.
    """
    layer_data = QgsInterpolator.LayerData()
    layer_data.source = layer
    layer_data.valueSource = QgsInterpolator.ValueZ   # use PointZ dim
    layer_data.sourceType = QgsInterpolator.SourcePoints
    layer_data.pointDistanceThreshold = point_distance_threshold

    interpolator = QgsIDWInterpolator([layer_data])
    interpolator.setDistanceCoefficient(distance_coefficient)

    out = np.empty(node_x.size, dtype=np.float64)
    for i, (x, y) in enumerate(zip(node_x, node_y)):
        result = interpolator.interpolatePoint(QgsPointXY(float(x), float(y)))
        out[i] = float(result)
    return out


# ── Raster sampling via QGIS provider (GDAL) ───────────────────────────

def interpolate_raster_provider(
    node_x: np.ndarray,
    node_y: np.ndarray,
    raster_layer,
    *,
    default_z: float = np.nan,
) -> np.ndarray:
    """Sample a raster layer at mesh node coords using QGIS provider.

    Uses ``QgsRasterDataProvider.identify()`` which delegates to GDAL.
    Returns ``default_z`` for nodes outside the raster extent or where
    the cell value is the no-data value.

    Args:
        node_x, node_y: Mesh node coordinates (float64, shape ``(n,)``).
        raster_layer: A QGIS raster layer.
        default_z: Fallback for outside-extent / no-data hits.

    Returns:
        Float64 array of length ``n`` with sampled z values.
    """
    provider: QgsRasterDataProvider = raster_layer.dataProvider()
    extent = provider.extent()
    crs = provider.crs()
    width = provider.xSize()
    height = provider.ySize()
    if width <= 0 or height <= 0:
        return np.full(node_x.size, default_z, dtype=np.float64)

    no_data_value = provider.sourceNoDataValue(1)

    out = np.full(node_x.size, default_z, dtype=np.float64)
    for i, (x, y) in enumerate(zip(node_x, node_y)):
        # QGIS uses extent.xMinimum/yMaximum as the top-left of the
        # upper-left pixel and (width, height) as pixel grid size.
        px = (x - extent.xMinimum()) / extent.width()
        py = (extent.yMaximum() - y) / extent.height()
        col = int(px * width)
        row = int(py * height)
        if col < 0 or col >= width or row < 0 or row >= height:
            continue
        result = provider.identify(QgsPointXY(float(x), float(y)))
        if result is None or not result.isValid():
            continue
        results = result.results()
        band_value = results.get(1)
        if band_value is None:
            continue
        try:
            v = float(band_value)
        except (TypeError, ValueError):
            continue
        if no_data_value is not None and v == float(no_data_value):
            continue
        out[i] = v
    return out