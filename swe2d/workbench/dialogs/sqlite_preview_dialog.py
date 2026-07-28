#!/usr/bin/env python3
"""Enhanced GeoPackage table preview dialog with structured filtering,
blob deserialization, array viewer, and CSV export."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
from qgis.PyQt import QtCore, QtWidgets

from swe2d.results.db_utils import get_table_info, get_table_contents
from swe2d.workbench.dialogs.gpkg_array_viewer_widget import ArrayViewerWidget
from swe2d.workbench.services.numpy_blob_service import (
    combine_1d_arrays,
    deserialize_blob_to_array,
    discover_plottable_columns,
    export_table_to_csv,
    get_filtered_rows,
)

logger = logging.getLogger(__name__)

_TABLE_KIND_ACTIONS = {
    "run_log": "open+preview",
    "config": "open+preview",
    "line_results": "open+preview",
    "coupling_results": "open+preview",
    "mesh_results": "open+preview",
    "system": "preview",
    "table": "preview",
}


class SWE2DEnhancedTablePreviewDialog(QtWidgets.QDialog):
    """Enhanced table preview dialog with filter, dual-panel view, and array inspection."""

    plot_requested = QtCore.pyqtSignal(str, str)  # gpkg_path, table_name

    def __init__(
        self,
        gpkg_path: str,
        table_name: str,
        title: str = "Table Viewer",
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(str(title or "Table Viewer"))
        self.resize(1100, 750)
        self._gpkg_path = str(gpkg_path or "")
        self._table_name = str(table_name or "")
        self._current_rows: List[Dict[str, Any]] = []
        self._plottable_cols: List[Dict[str, Any]] = []
        self._blob_data_cache: Dict[str, Any] = {}

        root = QtWidgets.QVBoxLayout(self)
        root.addWidget(QtWidgets.QLabel(f"Source: {self._gpkg_path}\nTable: {self._table_name}"))

        # ── Filter bar ────────────────────────────────────────────────────
        filter_layout = QtWidgets.QHBoxLayout()
        filter_layout.addWidget(QtWidgets.QLabel("Filter:"))
        self._filter_col_combo = QtWidgets.QComboBox()
        self._filter_col_combo.setMinimumWidth(160)
        filter_layout.addWidget(self._filter_col_combo)
        self._filter_op_combo = QtWidgets.QComboBox()
        self._filter_op_combo.addItems(["=", "!=", ">", "<", ">=", "<=", "LIKE", "IN", "BETWEEN", "IS NULL", "IS NOT NULL"])
        filter_layout.addWidget(self._filter_op_combo)
        self._filter_value_edit = QtWidgets.QLineEdit()
        self._filter_value_edit.setPlaceholderText("Filter value...")
        self._filter_value_edit.setMinimumWidth(120)
        filter_layout.addWidget(self._filter_value_edit)
        self._filter_apply_btn = QtWidgets.QPushButton("Apply")
        self._filter_apply_btn.clicked.connect(self._apply_filter)
        filter_layout.addWidget(self._filter_apply_btn)
        self._filter_clear_btn = QtWidgets.QPushButton("Clear")
        self._filter_clear_btn.clicked.connect(self._clear_filter)
        filter_layout.addWidget(self._filter_clear_btn)
        filter_layout.addStretch(1)

        # Filter op toggle visibility
        self._filter_op_combo.currentTextChanged.connect(self._on_filter_op_changed)
        self._on_filter_op_changed(self._filter_op_combo.currentText())

        root.addLayout(filter_layout)

        # ── Splitter: metadata table (top) + array viewer (bottom) ────────
        self._splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)

        # Top: metadata table
        self._meta_table = QtWidgets.QTableWidget()
        self._meta_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self._meta_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self._meta_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self._meta_table.setAlternatingRowColors(True)
        self._meta_table.horizontalHeader().setStretchLastSection(True)
        self._meta_table.itemSelectionChanged.connect(self._on_meta_row_selected)
        self._splitter.addWidget(self._meta_table)

        # Bottom: array viewer
        self._array_viewer = ArrayViewerWidget()
        self._array_viewer.setMinimumHeight(200)
        self._splitter.addWidget(self._array_viewer)

        self._splitter.setStretchFactor(0, 2)
        self._splitter.setStretchFactor(1, 1)
        root.addWidget(self._splitter, stretch=1)

        # ── Bottom controls ───────────────────────────────────────────────
        ctrl_row = QtWidgets.QHBoxLayout()
        ctrl_row.addWidget(QtWidgets.QLabel("Limit:"))
        self._limit_spin = QtWidgets.QSpinBox()
        self._limit_spin.setRange(10, 10_000_000)
        self._limit_spin.setValue(250)
        self._limit_spin.valueChanged.connect(lambda _v: self.refresh_table())
        ctrl_row.addWidget(self._limit_spin)

        self._refresh_btn = QtWidgets.QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self.refresh_table)
        ctrl_row.addWidget(self._refresh_btn)

        self._export_csv_btn = QtWidgets.QPushButton("Export CSV")
        self._export_csv_btn.clicked.connect(self._export_csv)
        ctrl_row.addWidget(self._export_csv_btn)

        self._send_plot_btn = QtWidgets.QPushButton("Send to Explorer Plot")
        self._send_plot_btn.clicked.connect(self._send_to_plot)
        ctrl_row.addWidget(self._send_plot_btn)

        ctrl_row.addStretch(1)

        self._row_count_lbl = QtWidgets.QLabel("")
        ctrl_row.addWidget(self._row_count_lbl)
        root.addLayout(ctrl_row)

        # ── Buttons ───────────────────────────────────────────────────────
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self.refresh_table()

    def _on_filter_op_changed(self, op: str):
        """Hide value input for IS NULL / IS NOT NULL."""
        show = op not in ("IS NULL", "IS NOT NULL")
        self._filter_value_edit.setVisible(show)

    def _populate_filter_columns(self):
        """Fill filter column dropdown with non-blob columns."""
        self._filter_col_combo.clear()
        cols = get_table_info(self._gpkg_path, self._table_name)
        for name in cols:
            self._filter_col_combo.addItem(name)

    def refresh_table(self):
        """(Re)load metadata rows and rebuild the top table."""
        self._meta_table.setRowCount(0)
        self._meta_table.setColumnCount(0)
        self._array_viewer.show_array(None)  # clear
        self._blob_data_cache.clear()

        if not self._gpkg_path or not self._table_name or not os.path.exists(self._gpkg_path):
            return

        self._plottable_cols = discover_plottable_columns(self._gpkg_path, self._table_name)
        self._populate_filter_columns()
        self._load_rows()

    def _load_rows(self):
        """Load rows (with current filter if active) into metadata table."""
        limit = int(self._limit_spin.value())
        filter_col = self._filter_col_combo.currentText()
        filter_op = self._filter_op_combo.currentText()
        filter_val = self._filter_value_edit.text().strip()

        if filter_col and filter_op:
            if filter_op not in ("IS NULL", "IS NOT NULL") and not filter_val:
                self._current_rows = get_filtered_rows(self._gpkg_path, self._table_name, limit=limit)
            else:
                self._current_rows = get_filtered_rows(
                    self._gpkg_path, self._table_name,
                    column=filter_col, op=filter_op, value=filter_val,
                    limit=limit,
                )
        else:
            self._current_rows = get_filtered_rows(self._gpkg_path, self._table_name, limit=limit)

        if not self._current_rows:
            self._row_count_lbl.setText("0 rows")
            return

        # Build columns set from all rows
        all_cols = list(self._current_rows[0].keys())
        self._meta_table.setColumnCount(len(all_cols))
        self._meta_table.setHorizontalHeaderLabels(all_cols)

        blob_cols = set()
        for col in self._plottable_cols:
            if col["kind"].startswith("blob"):
                blob_cols.add(col["name"])

        for i, row in enumerate(self._current_rows):
            self._meta_table.setRowCount(i + 1)
            for j, col_name in enumerate(all_cols):
                val = row.get(col_name)
                if col_name in blob_cols:
                    blob_key = self._blob_display_key(row, col_name)
                    item = QtWidgets.QTableWidgetItem(blob_key)
                    item.setForeground(QtCore.Qt.GlobalColor.blue)
                    item.setToolTip("Click row to view this blob in the array viewer")
                elif isinstance(val, (bytes, memoryview)):
                    n = len(val)
                    item = QtWidgets.QTableWidgetItem(f"<blob {n} bytes>")
                elif val is None:
                    item = QtWidgets.QTableWidgetItem("")
                elif isinstance(val, float):
                    item = QtWidgets.QTableWidgetItem(f"{val:.6g}")
                else:
                    item = QtWidgets.QTableWidgetItem(str(val))
                self._meta_table.setItem(i, j, item)

        self._meta_table.resizeColumnsToContents()
        self._row_count_lbl.setText(f"{len(self._current_rows)} rows")

    def _blob_display_key(self, row: Dict[str, Any], col_name: str) -> str:
        """Build a display string for a blob column."""
        col_info = next((c for c in self._plottable_cols if c["name"] == col_name), None)
        if col_info and col_info["shape"]:
            shape_str = "×".join(str(s) for s in col_info["shape"])
            return f"float64[{shape_str}]"
        val = row.get(col_name)
        if isinstance(val, (bytes, memoryview)):
            return f"<blob {len(val)} bytes>"
        return "<blob ?>"

    def _on_meta_row_selected(self):
        """When a row is selected, deserialize every blob column for that row
        and display them combined as columns (alongside an Index column).

        All 1D blobs of equal length are combined into a single 2D table
        (e.g. swe2d_baked_pipe_cell_ts: Index | times_blob | values_blob).

        Mismatched shapes fall back to showing only the first blob.
        """
        row_idx = self._meta_table.currentRow()
        if row_idx < 0 or row_idx >= len(self._current_rows):
            return
        row = self._current_rows[row_idx]

        # Find all blob columns (preserve PRAGMA order)
        blob_col_names = [c["name"] for c in self._plottable_cols if c["kind"].startswith("blob")]
        if not blob_col_names:
            self._array_viewer.show_array(None)
            return

        # Use first metadata column as PK (table convention: run_id, link_id, etc.)
        pk_col = list(row.keys())[0]
        pk_value = row[pk_col]

        # Deserialize each blob column for this row
        arrays_by_name: Dict[str, np.ndarray] = {}
        for col_name in blob_col_names:
            cache_key = f"{row_idx}_{col_name}"
            if cache_key in self._blob_data_cache:
                arr = self._blob_data_cache[cache_key]
            else:
                arr = deserialize_blob_to_array(
                    self._gpkg_path, self._table_name, col_name,
                    **{pk_col: pk_value},
                )
                if arr is not None:
                    self._blob_data_cache[cache_key] = arr
            if arr is not None:
                arrays_by_name[col_name] = arr

        if not arrays_by_name:
            self._array_viewer.show_array(None)
            return

        # Friendly display names: strip _blob suffix
        named: Dict[str, np.ndarray] = {}
        for col_name, arr in arrays_by_name.items():
            label = col_name[:-len("_blob")] if col_name.endswith("_blob") else col_name
            named[label] = arr

        # Use service-layer to combine arrays (handles empty/1D/mismatched shape cases)
        column_labels, combined = combine_1d_arrays(named)
        if combined is not None and combined.ndim == 2 and len(column_labels) > 1:
            self._array_viewer.show_combined(column_labels, combined, row_index_label="Index")
            return

        # Fallback: show only the first available array
        first_name, first_arr = next(iter(named.items()))
        self._array_viewer.show_array(first_arr, first_name)

    def _apply_filter(self):
        self._load_rows()

    def _clear_filter(self):
        self._filter_col_combo.setCurrentIndex(0)
        self._filter_op_combo.setCurrentText("=")
        self._filter_value_edit.clear()
        self._load_rows()

    def _export_csv(self):
        """Export currently filtered rows to CSV."""
        if not self._current_rows:
            QtWidgets.QMessageBox.warning(self, "Export CSV", "No data to export")
            return
        filepath, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Table to CSV", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not filepath:
            return
        filter_col = self._filter_col_combo.currentText()
        filter_op = self._filter_op_combo.currentText()
        filter_val = self._filter_value_edit.text().strip()
        try:
            export_table_to_csv(
                self._gpkg_path, self._table_name, filepath,
                column=filter_col if filter_col and filter_op and filter_val else None,
                op=filter_op if filter_col and filter_op and filter_val else None,
                value=filter_val if filter_col and filter_op and filter_val else None,
            )
            QtWidgets.QMessageBox.information(self, "Export CSV", f"Data exported to {filepath}")
        except (ValueError, OSError) as exc:
            QtWidgets.QMessageBox.warning(self, "Export CSV", str(exc))

    def _send_to_plot(self):
        """Emit signal to switch explorer to plot tab for this table."""
        self.plot_requested.emit(self._gpkg_path, self._table_name)
        QtWidgets.QMessageBox.information(
            self, "Send to Plot",
            f"Table '{self._table_name}' data sent to the Explorer Plot tab.",
        )
