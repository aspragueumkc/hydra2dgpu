#!/usr/bin/env python3
"""Topology attribute table editor dialog."""

from __future__ import annotations

import logging
from typing import List

from qgis.PyQt import QtWidgets
from qgis.core import QgsFeature

logger_wb = logging.getLogger(__name__)


class TopologyAttributeTableDialog(QtWidgets.QDialog):
    def __init__(self, layer, title: str, field_specs, sort_fields=None, note: str = "", parent=None):
        super().__init__(parent)
        self.layer = layer
        self.field_specs = list(field_specs)
        self.sort_fields = list(sort_fields or [])
        self._row_feature_ids: List[int] = []

        self.setWindowTitle(title)
        self.resize(920, 440)

        root = QtWidgets.QVBoxLayout(self)
        hint = QtWidgets.QLabel(
            note
            or "Edit topology attributes here. Geometry remains edited in the map canvas or native QGIS layer tools."
        )
        hint.setWordWrap(True)
        root.addWidget(hint)

        self.table = QtWidgets.QTableWidget(0, len(self.field_specs))
        self.table.setHorizontalHeaderLabels([spec[1] for spec in self.field_specs])
        self.table.horizontalHeader().setStretchLastSection(True)
        root.addWidget(self.table, stretch=1)

        row_btns = QtWidgets.QHBoxLayout()
        self.refresh_btn = QtWidgets.QPushButton("Reload From Layer")
        self.refresh_btn.setToolTip("Reload all features from the topology layer into the table.")
        self.add_row_btn = QtWidgets.QPushButton("Add Row")
        self.add_row_btn.setToolTip("Insert a new blank row for a new topology feature.")
        self.remove_row_btn = QtWidgets.QPushButton("Remove Selected")
        self.remove_row_btn.setToolTip("Remove the selected rows (deletes features on save).")
        self.refresh_btn.clicked.connect(self._load_rows)
        self.add_row_btn.clicked.connect(self._add_blank_row)
        self.remove_row_btn.clicked.connect(self._remove_selected_rows)
        row_btns.addWidget(self.refresh_btn)
        row_btns.addWidget(self.add_row_btn)
        row_btns.addWidget(self.remove_row_btn)
        row_btns.addStretch(1)
        root.addLayout(row_btns)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._save_and_accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._load_rows()

    def _sorted_features(self):
        """Return layer features sorted by the configured sort fields."""
        feats = list(self.layer.getFeatures()) if self.layer is not None else []
        if not self.sort_fields:
            return feats

        def _key(ft):
            """Sort key tuple for a feature based on sort_fields."""
            vals = []
            for name in self.sort_fields:
                try:
                    value = ft[name]
                except KeyError:
                    logger_wb.warning("Exception parsing feature value", exc_info=True)
                    value = None
                vals.append((value is None, value))
            return vals

        feats.sort(key=_key)
        return feats

    def _set_editor(self, row: int, col: int, value, spec):
        """Set the cell editor widget for a given field spec (enum or plain text)."""
        kind = spec[2]
        if kind == "enum":
            combo = QtWidgets.QComboBox()
            for option in spec[3]:
                combo.addItem(str(option), str(option))
            text = str(value or "").strip().lower()
            idx = combo.findData(text)
            combo.setCurrentIndex(max(0, idx))
            self.table.setCellWidget(row, col, combo)
            return
        item = QtWidgets.QTableWidgetItem("" if value in (None, "") else str(value))
        self.table.setItem(row, col, item)

    def _editor_value(self, row: int, col: int, spec):
        """Extract the typed value from a cell editor widget."""
        kind = spec[2]
        if kind == "enum":
            combo = self.table.cellWidget(row, col)
            return None if combo is None else combo.currentData()
        item = self.table.item(row, col)
        text = "" if item is None else str(item.text()).strip()
        if text == "":
            return None
        if kind == "int":
            return int(round(float(text)))
        if kind == "float":
            return float(text)
        return text

    def _load_rows(self):
        """Populate the table widget from the current layer features."""
        self.table.setRowCount(0)
        self._row_feature_ids = []
        for ft in self._sorted_features():
            row = self.table.rowCount()
            self.table.insertRow(row)
            self._row_feature_ids.append(int(ft.id()))
            for col, spec in enumerate(self.field_specs):
                field_name = spec[0]
                try:
                    value = ft[field_name]
                except (KeyError, ValueError, TypeError):
                    self._log("[WARNING] Exception parsing feature value")
                    value = None
                self._set_editor(row, col, value, spec)

    def _add_blank_row(self):
        """Insert a blank row at the end of the table."""
        row = self.table.rowCount()
        self.table.insertRow(row)
        self._row_feature_ids.append(-1)
        for col, spec in enumerate(self.field_specs):
            self._set_editor(row, col, "", spec)

    def _remove_selected_rows(self):
        """Remove the currently selected rows from the table."""
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.table.removeRow(row)
            if 0 <= row < len(self._row_feature_ids):
                self._row_feature_ids.pop(row)

    def _save_and_accept(self):
        """Save all edits to the layer, commit changes, and close the dialog."""
        if self.layer is None:
            self.accept()
            return
        started_here = False
        try:
            if not self.layer.isEditable():
                if not self.layer.startEditing():
                    raise RuntimeError("Could not start layer editing session.")
                started_here = True

            field_idx = {spec[0]: self.layer.fields().indexOf(spec[0]) for spec in self.field_specs}
            provider = self.layer.dataProvider()

            for row, fid in enumerate(self._row_feature_ids):
                if fid < 0:
                    feat = QgsFeature(self.layer.fields())
                    if not provider.addFeatures([feat]):
                        raise RuntimeError("Failed to add new feature row to layer.")
                    fids = [f.id() for f in self.layer.getFeatures()]
                    fid = int(fids[-1]) if fids else -1
                    self._row_feature_ids[row] = fid
                for col, spec in enumerate(self.field_specs):
                    idx = field_idx.get(spec[0], -1)
                    if idx < 0:
                        continue
                    value = self._editor_value(row, col, spec)
                    self.layer.changeAttributeValue(fid, idx, value)

            if started_here and not self.layer.commitChanges():
                raise RuntimeError("Layer changes could not be committed.")
            self.layer.triggerRepaint()
            self.accept()
        except Exception as exc:
            if started_here:
                try:
                    self.layer.rollBack()
                except Exception:
                    self._log("[WARNING] Unexpected Exception silently caught — review this handler")
            QtWidgets.QMessageBox.warning(self, "Topology Editor", f"Failed to save layer edits: {exc}")
