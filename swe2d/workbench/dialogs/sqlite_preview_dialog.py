#!/usr/bin/env python3
"""SQLite/GeoPackage table preview dialog."""

from __future__ import annotations

import os

from qgis.PyQt import QtWidgets

from swe2d.results.db_utils import get_table_contents, get_table_info


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

        def _fmt_blob(val, col_idx: int, row_idx: int) -> str:
            blob = memoryview(val).cast("B") if isinstance(val, memoryview) else val
            if not isinstance(blob, (bytes, memoryview)):
                return str(val) if val is not None else ""
            n_bytes = len(blob)
            tn = self._table_name
            cn = cols[col_idx]

            def _ival(col_name: str):
                try:
                    idx = cols.index(col_name)
                    v = row[idx]
                    return int(v) if v is not None else None
                except (ValueError, TypeError):
                    return None

            if tn == "swe2d_baked_mesh" and cn == "baked_blob":
                nn = _ival("n_nodes")
                nc = _ival("n_cells")
                ne = _ival("n_edges")
                bits = []
                if nn is not None: bits.append(f"{nn} nodes")
                if nc is not None: bits.append(f"{nc} cells")
                if ne is not None: bits.append(f"{ne} edges")
                return f"Binary mesh ({', '.join(bits)})" if bits else f"binary {n_bytes} bytes"
            if cn in ("h_blob", "hu_blob", "hv_blob"):
                nt = _ival("n_timesteps")
                nc = _ival("n_cells")
                if nt is not None and nc is not None:
                    return f"float64[{nt}×{nc}]"
                return f"float64 {n_bytes} bytes"
            if cn == "times_blob":
                nt = _ival("n_timesteps")
                if nt is not None:
                    return f"float64[{nt}]"
                return f"float64 {n_bytes} bytes"
            return f"binary {n_bytes} bytes"

        for i, row in enumerate(rows):
            self.table.setRowCount(i + 1)
            for j, val in enumerate(row):
                text = _fmt_blob(val, j, i)
                self.table.setItem(i, j, QtWidgets.QTableWidgetItem(text))
