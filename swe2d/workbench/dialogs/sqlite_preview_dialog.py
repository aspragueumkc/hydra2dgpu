#!/usr/bin/env python3
"""SQLite/GeoPackage table preview dialog."""

from __future__ import annotations

import os

from qgis.PyQt import QtWidgets

from swe2d.workbench.services.gpkg_service import get_table_contents, get_table_info


class SWE2DSQLiteTablePreviewDialog(QtWidgets.QDialog):
    """Simple SQLite/GeoPackage table preview dialog."""

    def __init__(self, gpkg_path: str, table_name: str, title: str = "Table Preview", parent=None):
        super().__init__(parent)
        self.setWindowTitle(str(title or "Table Preview"))
        self.resize(980, 640)
        self._gpkg_path = str(gpkg_path or "")
        self._table_name = str(table_name or "")

        root = QtWidgets.QVBoxLayout(self)
        root.addWidget(QtWidgets.QLabel(f"Source: {self._gpkg_path}\nTable: {self._table_name}"))

        self.table = QtWidgets.QTableWidget()
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        root.addWidget(self.table, stretch=1)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Limit:"))
        self.limit_spin = QtWidgets.QSpinBox()
        self.limit_spin.setToolTip("Maximum number of rows to display in the preview.")
        self.limit_spin.setRange(10, 5000)
        self.limit_spin.setValue(250)
        row.addWidget(self.limit_spin)
        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        self.refresh_btn.setToolTip("Re-query the table with the current limit.")
        row.addWidget(self.refresh_btn)
        row.addStretch(1)
        root.addLayout(row)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        root.addWidget(buttons)

        self.refresh_btn.clicked.connect(self.refresh_table)
        self.limit_spin.valueChanged.connect(lambda _v: self.refresh_table())
        self.refresh_table()

    def refresh_table(self):
        """Reload the table preview from the GeoPackage with the current row limit."""
        self.table.setRowCount(0)
        self.table.setColumnCount(0)
        if not self._gpkg_path or not self._table_name or not os.path.exists(self._gpkg_path):
            return
        cols = get_table_info(self._gpkg_path, self._table_name)
        if not cols:
            return
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        lim = int(self.limit_spin.value())
        rows = get_table_contents(self._gpkg_path, self._table_name, limit=lim)
        self.table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                self.table.setItem(i, j, QtWidgets.QTableWidgetItem("" if val is None else str(val)))
