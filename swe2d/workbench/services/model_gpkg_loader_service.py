"""2D model GeoPackage layer loading and validation.

Loads 18 named vector layers from a GeoPackage into the QGIS project.
Zero Qt — no PyQt5 imports.  Layer names sourced from ``schema_definitions``.
"""
from __future__ import annotations

from swe2d.workbench.services.schema_definitions import get_layer_names


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
    for lname in get_layer_names():
        lyr = QgsVectorLayer(
            f"{gpkg_path}|layername={lname}", lname, "ogr",
        )
        if lyr is not None and lyr.isValid():
            layers[lname] = lyr
    return layers


def get_model_gpkg_layer_names() -> list[str]:
    """Return the ordered list of expected layer names in a 2D model gpkg."""
    return list(get_layer_names())
