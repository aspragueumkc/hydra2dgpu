#!/usr/bin/env python3
"""Hydrograph editor dialog for editing BC time series."""

from __future__ import annotations

import csv
from typing import List

from qgis.PyQt import QtWidgets


class HydrographEditorDialog(QtWidgets.QDialog):
    def __init__(self, side: str, initial_text: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Hydrograph Editor - {side.capitalize()} Boundary")
        self.resize(560, 380)

        root = QtWidgets.QVBoxLayout(self)
        hint = QtWidgets.QLabel(
            "Enter one point per row. Time accepts decimal hours or HH:MM(:SS). "
            "For Flow BCs, Value is total discharge Q."
        )
        hint.setWordWrap(True)
        root.addWidget(hint)

        self.table = QtWidgets.QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Time", "Value (Q or stage)"])
        self.table.horizontalHeader().setStretchLastSection(True)
        root.addWidget(self.table, stretch=1)

        row_btns = QtWidgets.QHBoxLayout()
        self.add_row_btn = QtWidgets.QPushButton("Add Row")
        self.add_row_btn.setToolTip("Add a new time-value row to the hydrograph table.")
        self.remove_row_btn = QtWidgets.QPushButton("Remove Selected")
        self.remove_row_btn.setToolTip("Remove the selected rows from the table.")
        self.load_csv_btn = QtWidgets.QPushButton("Load CSV")
        self.load_csv_btn.setToolTip("Load hydrograph time-value pairs from a CSV file.")
        self.save_csv_btn = QtWidgets.QPushButton("Save CSV")
        self.save_csv_btn.setToolTip("Save the hydrograph table to a CSV file.")
        self.add_row_btn.clicked.connect(self._add_row)
        self.remove_row_btn.clicked.connect(self._remove_selected_rows)
        self.load_csv_btn.clicked.connect(self._load_csv)
        self.save_csv_btn.clicked.connect(self._save_csv)
        row_btns.addWidget(self.add_row_btn)
        row_btns.addWidget(self.remove_row_btn)
        row_btns.addStretch(1)
        row_btns.addWidget(self.load_csv_btn)
        row_btns.addWidget(self.save_csv_btn)
        root.addLayout(row_btns)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._load_text(initial_text)
        if self.table.rowCount() == 0:
            self._add_row()

    def _add_row(self, time_text: str = "", value_text: str = ""):
        """Append a row with the given time and value strings."""
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(str(time_text)))
        self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(str(value_text)))

    def _remove_selected_rows(self):
        """Remove the currently selected rows from the table."""
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)

    def _load_text(self, text: str):
        """Parse semicolon/comma-separated text into table rows."""
        raw = str(text or "").strip()
        if not raw:
            return
        for chunk in raw.replace("\n", ";").split(";"):
            c = chunk.strip()
            if not c:
                continue
            if "," in c:
                a, b = c.split(",", 1)
            elif "=" in c:
                a, b = c.split("=", 1)
            else:
                continue
            self._add_row(a.strip(), b.strip())

    def _load_csv(self):
        """Load hydrograph time-value pairs from a CSV file."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Hydrograph CSV", "", "CSV files (*.csv)")
        if not path:
            return
        try:
            rows = []
            with open(path, "r", encoding="utf-8", newline="") as f:
                for rec in csv.reader(f):
                    if len(rec) < 2:
                        continue
                    t = str(rec[0]).strip()
                    v = str(rec[1]).strip()
                    if not t or not v:
                        continue
                    if t.lower() in ("time", "hours"):
                        continue
                    rows.append((t, v))
            self.table.setRowCount(0)
            for t, v in rows:
                self._add_row(t, v)
            if self.table.rowCount() == 0:
                self._add_row()
        except (OSError, csv.Error) as exc:
            QtWidgets.QMessageBox.warning(self, "Hydrograph CSV", f"Failed to load CSV: {exc}")

    def _save_csv(self):
        """Save hydrograph table contents to a CSV file."""
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Hydrograph CSV", "hydrograph.csv", "CSV files (*.csv)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["time", "value"])
                for r in range(self.table.rowCount()):
                    t_item = self.table.item(r, 0)
                    v_item = self.table.item(r, 1)
                    t = t_item.text().strip() if t_item else ""
                    v = v_item.text().strip() if v_item else ""
                    if not t and not v:
                        continue
                    w.writerow([t, v])
        except OSError as exc:
            QtWidgets.QMessageBox.warning(self, "Hydrograph CSV", f"Failed to save CSV: {exc}")

    def hydrograph_text(self) -> str:
        """Serialize the table contents to a semicolon-separated string."""
        parts = []
        for r in range(self.table.rowCount()):
            t_item = self.table.item(r, 0)
            v_item = self.table.item(r, 1)
            t = t_item.text().strip() if t_item else ""
            v = v_item.text().strip() if v_item else ""
            if not t and not v:
                continue
            parts.append(f"{t},{v}")
        return "; ".join(parts)
