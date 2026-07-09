"""Elevation-source assignment for mesh node_z.

QGIS-aware but QtWidgets-free. Reads a raster or Z-bearing vector layer and
samples/interpolates elevations onto mesh node coordinates.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np


def assign_node_z_from_elevation_source(
    mesh_data: dict[str, np.ndarray],
    elevation_layer: Any,
    log_fn: Optional[Callable[[str], None]] = None,
) -> bool:
    """Assign node_z in mesh_data from a QGIS elevation source layer.

    Parameters
    ----------
    mesh_data:
        Mesh dictionary with ``node_x`` and ``node_y``. ``node_z`` is
        created/updated in place.
    elevation_layer:
        A ``QgsRasterLayer`` or Z-bearing ``QgsVectorLayer``.
    log_fn:
        Optional logging callback.

    Returns
    -------
    True if node_z was assigned, False otherwise.
    """
    from qgis.core import QgsRasterLayer, QgsVectorLayer

    if mesh_data is None or elevation_layer is None:
        return False

    node_x = mesh_data.get("node_x")
    node_y = mesh_data.get("node_y")
    if node_x is None or node_y is None:
        if log_fn:
            log_fn("[WARNING] Mesh missing node_x/node_y; cannot assign elevations.")
        return False

    if isinstance(elevation_layer, QgsRasterLayer):
        return _assign_from_raster(mesh_data, elevation_layer, node_x, node_y, log_fn)

    if isinstance(elevation_layer, QgsVectorLayer):
        return _assign_from_vector(mesh_data, elevation_layer, node_x, node_y, log_fn)

    if log_fn:
        log_fn(
            f"[WARNING] Elevation source layer type not supported: "
            f"{type(elevation_layer).__name__}."
        )
    return False


def auto_assign_node_z_from_view_elevation_source(view, mesh_data: dict[str, np.ndarray]) -> bool:
    """Read the elevation source from a view and assign node_z in place.

    Parameters
    ----------
    view:
        A view exposing ``get_topo_elevation_layer_id()`` and ``_log``.
    mesh_data:
        Mesh dictionary to update in place.

    Returns
    -------
    True if node_z was assigned, False otherwise.
    """
    from qgis.core import QgsProject

    layer_id = view.get_topo_elevation_layer_id()
    if not layer_id:
        return False
    layer = QgsProject.instance().mapLayer(layer_id)
    if layer is None:
        view._log(f"[WARNING] Elevation source layer {layer_id} not found.")
        return False
    ok = assign_node_z_from_elevation_source(mesh_data, layer, log_fn=view._log)
    if ok:
        view._log(f"Assigned node z from elevation source: {layer.name()}")
    return ok


def _assign_from_raster(
    mesh_data: dict[str, np.ndarray],
    raster_layer: Any,
    node_x: np.ndarray,
    node_y: np.ndarray,
    log_fn: Optional[Callable[[str], None]],
) -> bool:
    from swe2d.services.terrain_assignment_service import sample_raster_at_nodes

    provider = raster_layer.dataProvider()
    extent = raster_layer.extent()
    block = provider.block(1, extent, raster_layer.width(), raster_layer.height())
    if not block.isValid():
        if log_fn:
            log_fn("[ERROR] Could not read elevation raster block.")
        return False

    data_type_map = {
        1: np.uint8,
        2: np.uint16,
        3: np.int16,
        4: np.uint32,
        5: np.int32,
        6: np.float32,
        7: np.float64,
    }
    dtype = data_type_map.get(block.dataType(), np.float64)
    raster_data = np.frombuffer(bytes(block.data()), dtype=dtype)
    raster_data = raster_data.reshape(block.height(), block.width())
    geo_transform = (
        extent.xMinimum(),
        raster_layer.rasterUnitsPerPixelX(),
        0.0,
        extent.yMaximum(),
        0.0,
        -raster_layer.rasterUnitsPerPixelY(),
    )
    mesh_data["node_z"] = sample_raster_at_nodes(
        node_x, node_y, raster_data, geo_transform
    )
    return True


def _assign_from_vector(
    mesh_data: dict[str, np.ndarray],
    vector_layer: Any,
    node_x: np.ndarray,
    node_y: np.ndarray,
    log_fn: Optional[Callable[[str], None]],
) -> bool:
    from swe2d.services.terrain_assignment_service import idw_interpolate_points

    point_x: list[float] = []
    point_y: list[float] = []
    point_z: list[float] = []

    for feat in vector_layer.getFeatures():
        geom = feat.geometry()
        if geom is None or geom.isEmpty():
            continue
        for pt in geom.vertices():
            z = pt.z()
            if z is None or np.isnan(z):
                continue
            point_x.append(pt.x())
            point_y.append(pt.y())
            point_z.append(z)

    if not point_x:
        if log_fn:
            log_fn("[WARNING] Elevation source vector layer has no Z points.")
        return False

    mesh_data["node_z"] = idw_interpolate_points(
        node_x,
        node_y,
        np.asarray(point_x, dtype=np.float64),
        np.asarray(point_y, dtype=np.float64),
        np.asarray(point_z, dtype=np.float64),
    )
    return True
