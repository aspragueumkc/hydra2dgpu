"""Store QML layer styles in a GeoPackage's ``layer_styles`` table.

When a 2D model GeoPackage is created, this service reads the pre-generated
QML files from ``QML/`` and inserts them into the GPKG ``layer_styles``
table so that QGIS auto-applies the editor widget config (ValueMaps,
constraints, aliases, defaults) whenever the layers are loaded.

Usage::

    from swe2d.workbench.services.gpkg_layer_styles_service import (
        write_qml_styles_to_gpkg,
        apply_qml_style_from_gpkg,
    )

    # After creating layers in a GPKG:
    write_qml_styles_to_gpkg(gpkg_path, qml_dir)

    # When loading a single layer with auto-style:
    apply_qml_style_from_gpkg(layer, gpkg_path)
"""
from __future__ import annotations

import os
import sqlite3

from swe2d.workbench.services.schema_definitions import (
    get_layer_names,
    get_geom_column,
)


def _ensure_layer_styles_table(conn: sqlite3.Connection) -> None:
    """Create the ``layer_styles`` table if it does not exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS layer_styles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            f_table_catalog TEXT DEFAULT '',
            f_table_schema TEXT DEFAULT '',
            f_table_name TEXT NOT NULL,
            f_geometry_column TEXT DEFAULT '',
            styleName TEXT,
            styleQML TEXT,
            styleSLD TEXT,
            useAsDefault BOOLEAN DEFAULT true,
            description TEXT DEFAULT '',
            owner TEXT DEFAULT '',
            ui TEXT DEFAULT '',
            update_time DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)


def write_qml_styles_to_gpkg(gpkg_path: str, qml_dir: str) -> None:
    """Read each layer's QML file and write it to the GPKG ``layer_styles`` table.

    Args:
        gpkg_path: Path to an existing 2D model GeoPackage.
        qml_dir: Directory containing ``{layer_name}.qml`` files.
    """
    conn = sqlite3.connect(gpkg_path)
    try:
        _ensure_layer_styles_table(conn)
        for layer_name in get_layer_names():
            qml_path = os.path.join(qml_dir, f"{layer_name}.qml")
            if not os.path.exists(qml_path):
                continue
            with open(qml_path, "r", encoding="utf-8") as f:
                style_qml = f.read()

            geom_col = get_geom_column(layer_name)

            # Remove any existing style entry for this layer to avoid duplicates
            conn.execute(
                "DELETE FROM layer_styles WHERE f_table_name = ?",
                (layer_name,),
            )
            conn.execute(
                """INSERT INTO layer_styles
                   (f_table_catalog, f_table_schema, f_table_name,
                    f_geometry_column, styleName, styleQML, useAsDefault,
                    description, update_time)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
                (
                    "",
                    "",
                    layer_name,
                    geom_col,
                    "default",
                    style_qml,
                    1,
                    f"SWE2D {layer_name} style",
                ),
            )
        conn.commit()
    finally:
        conn.close()


def apply_qml_style_from_gpkg(layer, gpkg_path: str) -> bool:
    """Load a layer's QML style directly from the GPKG ``layer_styles`` table.

    This is a fallback / explicit alternative to ``loadAllStoredStyles``.
    Returns ``True`` if a style was found and applied.
    """
    lname = layer.name()
    conn = sqlite3.connect(gpkg_path)
    try:
        # Bail if the layer_styles table doesn't exist (old GPKG or write failed)
        has_table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='layer_styles'"
        ).fetchone()
        if has_table is None:
            return False

        row = conn.execute(
            "SELECT styleQML FROM layer_styles WHERE f_table_name = ?",
            (lname,),
        ).fetchone()
        if row is None:
            return False
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".qml", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(row[0])
            tmp_path = tmp.name
        try:
            ok = layer.loadNamedStyle(tmp_path)
            return bool(ok)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    finally:
        conn.close()
