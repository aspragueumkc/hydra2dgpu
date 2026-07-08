from qgis.PyQt.QtWidgets import QComboBox, QLineEdit
import sqlite3
import os


def _populate_id_combo(dialog, layer, feature, field_name, source_table):
    le = dialog.findChild(QLineEdit, field_name)
    if le is None:
        return
    source = layer.source()
    gpkg_path = source.split("|")[0] if "|" in source else source
    if not os.path.exists(gpkg_path):
        return

    combo = QComboBox(le.parentWidget())
    combo.addItem("", "")

    id_field = "hyetograph_id" if source_table == "swe2d_hyetographs" else "hydrograph_id"
    conn = sqlite3.connect(gpkg_path)
    try:
        for row in conn.execute(
            f"SELECT DISTINCT {id_field} FROM {source_table} ORDER BY {id_field}"
        ):
            if row[0]:
                combo.addItem(str(row[0]), str(row[0]))
    finally:
        conn.close()

    val = str(feature.attribute(field_name) or "")
    idx = combo.findData(val)
    if idx >= 0:
        combo.setCurrentIndex(idx)

    combo.currentTextChanged.connect(lambda t: le.setText(t))

    parent_layout = le.parentWidget().layout()
    if parent_layout:
        parent_layout.insertWidget(parent_layout.indexOf(le) + 1, combo)

    if val:
        le.setText(val)


def rain_gages_form_init(dialog, layer, feature):
    _populate_id_combo(dialog, layer, feature, "hyetograph_id", "swe2d_hyetographs")


def bc_lines_form_init(dialog, layer, feature):
    _populate_id_combo(dialog, layer, feature, "hydrograph_id", "swe2d_hydrographs")


def internal_flow_form_init(dialog, layer, feature):
    _populate_id_combo(dialog, layer, feature, "hydrograph_id", "swe2d_hydrographs")
