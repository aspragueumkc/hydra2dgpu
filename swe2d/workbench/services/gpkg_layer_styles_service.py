"""Apply QML layer styles from the bundled ``QML/`` directory.

The QML files on disk are the single source of truth for layer styling.
Users edit the ``*.qml`` files in QGIS and they take effect on every load.
"""
from __future__ import annotations

import os
import tempfile

from swe2d.workbench.services.schema_definitions import get_layer_names


def apply_qml_style_from_gpkg(layer, gpkg_path: str) -> bool:
    """Load a layer's QML style from the bundled ``QML/`` directory.

    Args:
        layer: QgsVectorLayer to style.
        gpkg_path: Path to the GeoPackage (used to locate QML dir).

    Returns:
        True if a QML file was found and applied.
    """
    lname = layer.name()
    qml_path = _qml_file_path(lname)
    if qml_path is None or not os.path.exists(qml_path):
        return False

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".qml", delete=False, encoding="utf-8"
    ) as tmp:
        with open(qml_path, "r", encoding="utf-8") as f:
            tmp.write(f.read())
        tmp_path = tmp.name
    try:
        ok = layer.loadNamedStyle(tmp_path)
        return bool(ok)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _qml_file_path(layer_name: str) -> str | None:
    """Return the path to a layer's QML file, or None."""
    qml_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        "QML",
    )
    path = os.path.join(qml_dir, f"{layer_name}.qml")
    return path if os.path.exists(path) else None
