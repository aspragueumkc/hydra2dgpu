#!/usr/bin/env python3
"""Dedicated viewer for swe2d_simulation_configs table rows.

Parses the widget_state JSON column and displays widget names/values/types
in a sortable table instead of showing raw JSON text.
"""

from __future__ import annotations

import json
import os
import sqlite3

from qgis.PyQt import QtWidgets


class SWE2DSimulationConfigViewerDialog(QtWidgets.QDialog):
    """Viewer for saved simulation configurations.

    Shows a combo selector of available configs, metadata labels,
    and a tabular breakdown of the saved widget values.
    """

    def __init__(self, gpkg_path: str, parent=None):
        super().__init__(parent)
        self._gpkg_path = str(gpkg_path or "")
        self.setWindowTitle("Simulation Configuration Viewer")
        self.resize(860, 520)

        root = QtWidgets.QVBoxLayout(self)
        root.addWidget(QtWidgets.QLabel(f"GeoPackage: {self._gpkg_path}"))

        # Config selector
        sel_row = QtWidgets.QHBoxLayout()
        sel_row.addWidget(QtWidgets.QLabel("Configuration:"))
        self.config_combo = QtWidgets.QComboBox()
        self.config_combo.setMinimumWidth(300)
        sel_row.addWidget(self.config_combo, stretch=1)
        root.addLayout(sel_row)

        # Metadata labels
        self.meta_lbl = QtWidgets.QLabel("")
        self.meta_lbl.setWordWrap(True)
        root.addWidget(self.meta_lbl)

        # Widget values table
        self.table = QtWidgets.QTableWidget()
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Widget", "Value", "Type"])
        self.table.horizontalHeader().setStretchLastSection(True)
        root.addWidget(self.table, stretch=1)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._load_configs()
        self.config_combo.currentIndexChanged.connect(self._show_config)
        if self.config_combo.count() > 0:
            self.config_combo.setCurrentIndex(0)
            self._show_config(0)

    def _load_configs(self):
        """Read all rows from swe2d_simulation_configs and populate the combo."""
        if not self._gpkg_path or not os.path.exists(self._gpkg_path):
            return
        try:
            conn = sqlite3.connect(self._gpkg_path)
            cur = conn.cursor()
            cur.execute(
                "SELECT config_id, mesh_name, created_utc, run_duration_s, widget_state "
                "FROM swe2d_simulation_configs ORDER BY created_utc DESC"
            )
            self._rows = list(cur.fetchall())
            conn.close()
        except sqlite3.Error:
            self._rows = []

        if not self._rows:
            self._rows = []
            self.config_combo.addItem("(no configs found)")
            self.config_combo.setEnabled(False)
            return

        for row in self._rows:
            config_id = row[0] or ""
            created = row[2] or ""
            label = f"{config_id}  ({created})"
            self.config_combo.addItem(label)

    def _show_config(self, idx: int):
        """Display the selected config's metadata and widget values."""
        if idx < 0 or idx >= len(self._rows):
            return
        row = self._rows[idx]
        config_id, mesh_name, created_utc, duration_s, widget_state_json = row

        meta_parts = [f"ID: {config_id}"]
        if mesh_name:
            meta_parts.append(f"Mesh: {mesh_name}")
        if created_utc:
            meta_parts.append(f"Created: {created_utc}")
        if duration_s is not None:
            meta_parts.append(f"Duration: {duration_s} s")
        self.meta_lbl.setText("  |  ".join(meta_parts))

        self.table.setRowCount(0)
        widgets = {}
        if widget_state_json:
            try:
                parsed = json.loads(str(widget_state_json))
                widgets = parsed.get("widgets", {})
            except (json.JSONDecodeError, TypeError):
                pass

        if not widgets:
            self.table.setRowCount(1)
            self.table.setItem(0, 0, QtWidgets.QTableWidgetItem("(no widget data saved)"))
            self.table.setItem(0, 1, QtWidgets.QTableWidgetItem(""))
            self.table.setItem(0, 2, QtWidgets.QTableWidgetItem(""))
            return

        self.table.setRowCount(len(widgets))
        for i, (name, info) in enumerate(sorted(widgets.items())):
            typ = info.get("type", "") if isinstance(info, dict) else ""
            val = info.get("value", info) if isinstance(info, dict) else info
            self.table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(name)))
            self.table.setItem(i, 1, QtWidgets.QTableWidgetItem(str(val)))
            self.table.setItem(i, 2, QtWidgets.QTableWidgetItem(str(typ)))
        self.table.resizeColumnsToContents()
