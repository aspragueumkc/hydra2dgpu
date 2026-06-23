"""Lumped hydrology GeoPackage template creation.

Provides function to write a GeoPackage with three template layers
(subbasins, flow paths, rain events).  Zero Qt — no PyQt5 imports.
"""
from __future__ import annotations


def create_lumped_subbasins_layer(crs_auth: str = "EPSG:4326"):
    """Return in-memory QgsVectorLayer for lumped subbasins."""
    from qgis.core import QgsVectorLayer
    uri = (
        f"Polygon?crs={crs_auth}"
        "&field=sub_id:string(64)"
        "&field=name:string(128)"
        "&field=area_km2:double"
        "&field=cn:double"
        "&field=imperv_pct:double"
        "&field=tc_hr:double"
    )
    return QgsVectorLayer(uri, "lumped_subbasins", "memory")


def create_lumped_flow_paths_layer(crs_auth: str = "EPSG:4326"):
    """Return in-memory QgsVectorLayer for lumped flow paths."""
    from qgis.core import QgsVectorLayer
    uri = (
        f"LineString?crs={crs_auth}"
        "&field=sub_id:string(64)"
        "&field=segment:string(32)"
        "&field=length_m:double"
        "&field=velocity_mps:double"
        "&field=slope:double"
    )
    return QgsVectorLayer(uri, "lumped_flow_paths", "memory")


def create_lumped_rain_events_layer():
    """Return in-memory QgsVectorLayer for lumped rain events (table, no geom)."""
    from qgis.core import QgsVectorLayer
    uri = (
        "None?"
        "field=event_id:string(64)"
        "&field=Time:string(32)"
        "&field=Value:double"
        "&field=value_type:string(24)"
        "&field=units:string(24)"
        "&field=description:string(256)"
    )
    return QgsVectorLayer(uri, "lumped_rain_events", "memory")


def write_memory_layer_to_gpkg(lyr, gpkg_path: str, layer_name: str, *, create_file: bool = False):
    """Write a single in-memory QgsVectorLayer to a GeoPackage."""
    from qgis.core import QgsVectorFileWriter
    options = QgsVectorFileWriter.SaveVectorOptions()
    if not create_file:
        options.actionOnExistingFile = QgsVectorFileWriter.CreateOrOverwriteLayer
    options.layerName = layer_name
    options.fileEncoding = "UTF-8"
    QgsVectorFileWriter.writeAsVectorFormatV3(lyr, gpkg_path, lyr.transformContext(), options)


def write_lumped_hydrology_geopackage(out_path: str, crs_auth: str = "EPSG:4326") -> None:
    """Write a GeoPackage with the three lumped-hydrology template layers."""
    layers = [
        (create_lumped_subbasins_layer(crs_auth), "lumped_subbasins"),
        (create_lumped_flow_paths_layer(crs_auth), "lumped_flow_paths"),
        (create_lumped_rain_events_layer(), "lumped_rain_events"),
    ]
    for i, (lyr, name) in enumerate(layers):
        write_memory_layer_to_gpkg(lyr, out_path, name, create_file=(i == 0))
