"""2D model GeoPackage layer loading and validation.

Loads 18 named vector layers from a GeoPackage into the QGIS project.
Zero Qt — no PyQt5 imports.
"""
from __future__ import annotations

_MODEL_GPKG_LAYER_NAMES = [
    "swe2d_topo_nodes",
    "swe2d_topo_arcs",
    "swe2d_topo_regions",
    "swe2d_topo_constraints",
    "swe2d_topo_quad_edges",
    "swe2d_manning_zones",
    "swe2d_bc_lines",
    "swe2d_sample_lines",
    "swe2d_rain_gages",
    "swe2d_storm_areas",
    "swe2d_cn_zones",
    "swe2d_hyetographs",
    "swe2d_hydrographs",
    "swe2d_drainage_nodes",
    "swe2d_drainage_links",
    "swe2d_drainage_inlets",
    "swe2d_drainage_node_inlets",
    "swe2d_structures",
]


def load_layers_from_gpkg(gpkg_path: str) -> dict[str, "QgsVectorLayer"]:
    """Load all 18 model layers from a 2D model GeoPackage.

    Args:
        gpkg_path: Path to the GeoPackage file.

    Returns:
        Dict mapping layer name to valid QgsVectorLayer.
        Missing layers are silently skipped.
    """
    from qgis.core import QgsVectorLayer

    layers: dict[str, "QgsVectorLayer"] = {}
    for lname in _MODEL_GPKG_LAYER_NAMES:
        lyr = QgsVectorLayer(
            f"{gpkg_path}|layername={lname}", lname, "ogr",
        )
        if lyr is not None and lyr.isValid():
            layers[lname] = lyr
    return layers


def get_model_gpkg_layer_names() -> list[str]:
    """Return the ordered list of expected layer names in a 2D model gpkg."""
    return list(_MODEL_GPKG_LAYER_NAMES)
