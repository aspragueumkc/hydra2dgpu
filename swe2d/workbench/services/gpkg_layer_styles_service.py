"""Apply QML layer styles from the bundled ``QML/`` directory.

The QML files on disk are the single source of truth for layer styling.
Users edit the ``*.qml`` files in QGIS and they take effect on every load.

The QML explicitly defines symbology, labels, and other styling categories.
The attribute form config (form init, dropdown widgets) is overridden in
code after QML load so the inline Python init code never triggers the
QGIS "Allow macros?" prompt — the dropdowns are wired via the built-in
``ValueRelation`` widget instead.
"""
from __future__ import annotations

import os
import re
import tempfile

from qgis.core import (
    QgsEditFormConfig,
    QgsEditorWidgetSetup,
    QgsProject,
)

from swe2d.workbench.services.schema_definitions import get_layer_names


# (field_name, source_layer_name, key_field) per auto-loaded layer
_VALUE_RELATION_DROPDOWNS = {
    "swe2d_bc_lines": [
        ("hydrograph_id", "swe2d_hydrographs", "hydrograph_id"),
    ],
    "swe2d_rain_gages": [
        ("hyetograph_id", "swe2d_hyetographs", "hyetograph_id"),
    ],
    "swe2d_internal_flow_sources": [
        ("hydrograph_id", "swe2d_hydrographs", "hydrograph_id"),
    ],
}


# Strip the inline Python init code from the QML before load. QGIS evaluates
# `<editforminitcode>` in its own scope and then tries to call the function
# named in `<editforminit>` in a *different* scope, which raises
# `NameError: name '<fn>' is not defined` on every form open. 3 = ProvidedValue.
_FORM_INIT_FIELDS = [
    ("editforminit", ""),
    ("editforminitcodesource", "3"),
    ("editforminitfilepath", ""),
]


def apply_qml_style_from_gpkg(layer, gpkg_path: str) -> bool:
    """Load a layer's QML style from the bundled ``QML/`` directory.

    After loading the QML, override the form config so the inline Python
    init code is replaced with programmatic settings (no macro prompt).
    """
    lname = layer.name()
    qml_path = _qml_file_path(lname)
    if qml_path is None or not os.path.exists(qml_path):
        return False

    with open(qml_path, "r", encoding="utf-8") as f:
        qml_text = _strip_form_init_code(f.read())

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".qml", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(qml_text)
        tmp_path = tmp.name
    try:
        ok = layer.loadNamedStyle(tmp_path)
        if ok:
            _apply_clean_form_config(layer, lname)
        return bool(ok)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _strip_form_init_code(qml_text: str) -> str:
    """Remove the inline Python init code from a QML string in-place.

    Drops ``<editforminitcode>`` (the function definition) and rewrites
    ``<editforminit>/<editforminitcodesource>/<editforminitfilepath>`` so
    QGIS has nothing to evaluate when the form opens.
    """
    qml_text = re.sub(
        r"<editforminitcode>.*?</editforminitcode>",
        "<editforminitcode></editforminitcode>",
        qml_text,
        flags=re.DOTALL,
    )
    for tag, replacement in _FORM_INIT_FIELDS:
        qml_text = re.sub(
            rf"<{tag}>.*?</{tag}>",
            f"<{tag}>{replacement}</{tag}>",
            qml_text,
            flags=re.DOTALL,
        )
    return qml_text


def _apply_clean_form_config(layer, layer_name: str) -> None:
    """Strip the QML's Python init code and wire dropdowns via ValueRelation.

    Keeping the QML's inline ``<editforminitcode>`` makes QGIS fire the
    "Allow macros?" prompt every time the attribute form is opened. We
    drop the init code and rebuild the dynamic-needed widgets through the
    built-in ``ValueRelation`` widget, which reads its values from the
    source layer at runtime.
    """
    cfg = layer.editFormConfig()
    cfg.setInitCodeSource(QgsEditFormConfig.CodeSourceProvidedValue)
    cfg.setInitFunction("")
    layer.setEditFormConfig(cfg)

    fields = layer.fields()
    project_layers = {
        lyr.name(): lyr
        for lyr in QgsProject.instance().mapLayers().values()
    }
    for field_name, source_layer_name, key_field in _VALUE_RELATION_DROPDOWNS.get(
        layer_name, ()
    ):
        field_idx = fields.indexFromName(field_name)
        if field_idx < 0:
            continue
        source = project_layers.get(source_layer_name)
        if source is None:
            continue
        setup = QgsEditorWidgetSetup(
            "ValueRelation",
            {
                "Layer": source.id(),
                "Key": key_field,
                "Value": key_field,
                "AllowNull": False,
                "AllowMulti": False,
                "FilterExpression": "",
            },
        )
        layer.setEditorWidgetSetup(field_idx, setup)


def _qml_file_path(layer_name: str) -> str | None:
    """Return the path to a layer's QML file, or None."""
    qml_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        "QML",
    )
    path = os.path.join(qml_dir, f"{layer_name}.qml")
    return path if os.path.exists(path) else None
