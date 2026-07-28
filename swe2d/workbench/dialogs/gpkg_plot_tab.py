"""swe2d/workbench/dialogs/gpkg_plot_tab.py

XY plot tab widget for the GeoPackage Explorer. Lets users select X and Y
columns (including deserialized blob arrays) and renders a matplotlib plot.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
from qgis.PyQt import QtCore, QtWidgets

from swe2d.workbench.dialogs._plot_utils import try_import_matplotlib_qt
from swe2d.workbench.services.numpy_blob_service import (
    deserialize_blob_to_array,
    discover_plottable_columns,
    export_table_to_csv,
)

logger = logging.getLogger(__name__)

FigureCanvasQt, Figure, _mtri = try_import_matplotlib_qt()


class GpkgPlotTab(QtWidgets.QWidget):
    """Plot tab embedded in the explorer dialog for XY plotting."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._gpkg_path: str = ""
        self._table: str = ""
        self._x_col: str = ""
        self._y_col: str = ""
        self._available_cols: List[Dict[str, Any]] = []
        self._pk_col: str = ""

        root = QtWidgets.QVBoxLayout(self)

        source_row = QtWidgets.QHBoxLayout()
        source_row.addWidget(QtWidgets.QLabel("GeoPackage:"))
        self._gpkg_lbl = QtWidgets.QLabel("(none)")
        source_row.addWidget(self._gpkg_lbl, stretch=1)
        root.addLayout(source_row)

        source_row2 = QtWidgets.QHBoxLayout()
        source_row2.addWidget(QtWidgets.QLabel("Table:"))
        self._table_lbl = QtWidgets.QLabel("(none)")
        source_row2.addWidget(self._table_lbl, stretch=1)
        root.addLayout(source_row2)

        controls_row = QtWidgets.QHBoxLayout()
        controls_row.addWidget(QtWidgets.QLabel("X-axis:"))
        self._x_combo = QtWidgets.QComboBox()
        self._x_combo.setMinimumWidth(180)
        controls_row.addWidget(self._x_combo)

        controls_row.addWidget(QtWidgets.QLabel("Y-axis:"))
        self._y_combo = QtWidgets.QComboBox()
        self._y_combo.setMinimumWidth(180)
        controls_row.addWidget(self._y_combo)

        controls_row.addWidget(QtWidgets.QLabel("Plot type:"))
        self._plot_type_combo = QtWidgets.QComboBox()
        self._plot_type_combo.addItems(["Scatter", "Line", "Line+Scatter"])
        controls_row.addWidget(self._plot_type_combo)

        self._log_x_cb = QtWidgets.QCheckBox("Log X")
        controls_row.addWidget(self._log_x_cb)
        self._log_y_cb = QtWidgets.QCheckBox("Log Y")
        controls_row.addWidget(self._log_y_cb)

        self._plot_btn = QtWidgets.QPushButton("Plot")
        self._plot_btn.clicked.connect(self._render_plot)
        controls_row.addWidget(self._plot_btn)
        controls_row.addStretch(1)
        root.addLayout(controls_row)

        # Slice controls for 2D arrays
        slice_row = QtWidgets.QHBoxLayout()
        self._slice_lbl = QtWidgets.QLabel("Slice (for 2D arrays):")
        self._slice_spin = QtWidgets.QSpinBox()
        self._slice_spin.setRange(0, 0)
        self._slice_spin.setToolTip("For 2D arrays, which index to extract as 1D series")
        slice_row.addWidget(self._slice_lbl)
        slice_row.addWidget(self._slice_spin)
        slice_row.addStretch(1)
        self._slice_widget = QtWidgets.QWidget()
        slice_inner = QtWidgets.QHBoxLayout(self._slice_widget)
        slice_inner.setContentsMargins(0, 0, 0, 0)
        slice_inner.addLayout(slice_row)
        self._slice_widget.setVisible(False)
        root.addWidget(self._slice_widget)

        # Canvas
        self._canvas_container = QtWidgets.QWidget()
        self._canvas_layout = QtWidgets.QVBoxLayout(self._canvas_container)
        self._canvas_layout.setContentsMargins(0, 0, 0, 0)
        self._canvas = None
        root.addWidget(self._canvas_container, stretch=1)

        # Export buttons
        btn_row = QtWidgets.QHBoxLayout()
        self._export_csv_btn = QtWidgets.QPushButton("Export CSV")
        self._export_csv_btn.clicked.connect(self._export_csv)
        btn_row.addWidget(self._export_csv_btn)

        if FigureCanvasQt is not None:
            self._export_png_btn = QtWidgets.QPushButton("Export PNG")
            self._export_png_btn.clicked.connect(self._export_png)
            btn_row.addWidget(self._export_png_btn)

        btn_row.addStretch(1)
        root.addLayout(btn_row)

        # Enable state
        self._x_combo.currentTextChanged.connect(self._on_column_changed)
        self._y_combo.currentTextChanged.connect(self._on_column_changed)
        self._on_column_changed()

    def set_table(self, gpkg_path: str, table: str):
        """Point the plot tab at a specific table and refresh column dropdowns."""
        self._gpkg_path = gpkg_path
        self._table = table
        self._gpkg_lbl.setText(os.path.basename(gpkg_path))
        self._table_lbl.setText(table)
        self._refresh_columns()

    def _refresh_columns(self):
        """Discover plottable columns and populate dropdowns."""
        self._x_combo.blockSignals(True)
        self._y_combo.blockSignals(True)
        self._x_combo.clear()
        self._y_combo.clear()

        if not self._gpkg_path or not self._table:
            self._available_cols = []
            self._x_combo.blockSignals(False)
            self._y_combo.blockSignals(False)
            return

        self._available_cols = discover_plottable_columns(self._gpkg_path, self._table)
        for col in self._available_cols:
            name = col["name"]
            kind = col["kind"]
            suffix = ""
            if kind == "blob_1d":
                suffix = " [1D]"
            elif kind == "blob_2d":
                suffix = " [2D]"
            elif kind == "metadata_numeric":
                suffix = " [num]"
            label = f"{name}{suffix}"
            self._x_combo.addItem(label, name)
            self._y_combo.addItem(label, name)

        if self._x_combo.count() > 0:
            self._x_combo.setCurrentIndex(0)
        if self._y_combo.count() > 1:
            self._y_combo.setCurrentIndex(1)
        elif self._y_combo.count() > 0:
            self._y_combo.setCurrentIndex(0)

        self._x_combo.blockSignals(False)
        self._y_combo.blockSignals(False)

    def _on_column_changed(self):
        x_name = self._x_combo.currentData() or ""
        y_name = self._y_combo.currentData() or ""
        has_both = bool(x_name and y_name)
        self._plot_btn.setEnabled(has_both)

        # Show slice controls if either column is 2D
        has_2d = False
        for col in self._available_cols:
            if col["name"] in (x_name, y_name) and col.get("ndim", 1) == 2:
                has_2d = True
                max_slice = (col["shape"][0] - 1) if col["shape"] else 0
                self._slice_spin.setRange(0, max(0, max_slice))
                break
        self._slice_widget.setVisible(has_2d)

    def _get_column_data(self, col_name: str) -> Optional[np.ndarray]:
        """Extract a 1D numpy array for a column (metadata or blob)."""
        col_info = next((c for c in self._available_cols if c["name"] == col_name), None)
        if col_info is None:
            return None

        if col_info["kind"].startswith("metadata_numeric"):
            rows = self._get_metadata_rows()
            if not rows:
                return None
            vals = []
            for r in rows:
                v = r.get(col_name)
                if v is not None:
                    try:
                        vals.append(float(v))
                    except (ValueError, TypeError):
                        vals.append(float("nan"))
                else:
                    vals.append(float("nan"))
            return np.array(vals)

        if col_info["kind"].startswith("blob"):
            blob_arr = deserialize_blob_to_array(self._gpkg_path, self._table, col_name, **self._get_pk())
            if blob_arr is None:
                return None
            if blob_arr.ndim == 2:
                idx = self._slice_spin.value()
                return blob_arr[idx, :]
            return blob_arr

        return None

    def _get_metadata_rows(self):
        """Fetch first N rows of metadata columns."""
        from swe2d.workbench.services.numpy_blob_service import get_filtered_rows
        return get_filtered_rows(self._gpkg_path, self._table, limit=0)

    def _get_pk(self) -> dict:
        """Get the primary key column name and first value."""
        if not self._available_cols:
            return {}
        rows = self._get_metadata_rows()
        if not rows:
            return {}
        first_col = self._available_cols[0]["name"]
        return {first_col: rows[0].get(first_col, "")}

    def _render_plot(self):
        x_name = self._x_combo.currentData() or ""
        y_name = self._y_combo.currentData() or ""
        if not x_name or not y_name:
            return

        x_data = self._get_column_data(x_name)
        y_data = self._get_column_data(y_name)
        if x_data is None or y_data is None:
            QtWidgets.QMessageBox.warning(self, "Plot Error", f"Cannot extract data for {x_name} vs {y_name}")
            return

        # Ensure same length
        min_len = min(len(x_data), len(y_data))
        if min_len == 0:
            QtWidgets.QMessageBox.warning(self, "Plot Error", "No data points to plot")
            return
        x_data = x_data[:min_len]
        y_data = y_data[:min_len]

        self._x_col = x_name
        self._y_col = y_name

        if FigureCanvasQt is None:
            QtWidgets.QMessageBox.warning(self, "Plot Error", "matplotlib not available")
            return

        if self._canvas is not None:
            self._canvas_layout.removeWidget(self._canvas)
            self._canvas.deleteLater()
            self._canvas = None

        fig = Figure(figsize=(6, 4))
        ax = fig.add_subplot(111)

        plot_type = self._plot_type_combo.currentText()

        if self._log_x_cb.isChecked():
            ax.set_xscale("log")
        if self._log_y_cb.isChecked():
            ax.set_yscale("log")

        if plot_type == "Scatter":
            ax.scatter(x_data, y_data, s=8, alpha=0.7)
        elif plot_type == "Line":
            ax.plot(x_data, y_data)
        else:
            ax.plot(x_data, y_data)
            ax.scatter(x_data, y_data, s=8, alpha=0.7)

        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.set_title(f"{y_name} vs {x_name}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        self._canvas = FigureCanvasQt(fig)
        self._canvas_layout.addWidget(self._canvas)

    def _export_csv(self):
        """Export plotted X and Y data to CSV."""
        if not self._x_col or not self._y_col:
            QtWidgets.QMessageBox.warning(self, "Export CSV", "No plot data to export")
            return

        filepath, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Data to CSV", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not filepath:
            return

        x_data = self._get_column_data(self._x_col)
        y_data = self._get_column_data(self._y_col)
        if x_data is None or y_data is None:
            return

        min_len = min(len(x_data), len(y_data))
        import csv
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([self._x_col, self._y_col])
            for i in range(min_len):
                writer.writerow([f"{x_data[i]:.10g}", f"{y_data[i]:.10g}"])

        QtWidgets.QMessageBox.information(self, "Export CSV", f"Exported {min_len} rows to {filepath}")

    def _export_png(self):
        """Export the current plot as PNG."""
        if self._canvas is None:
            QtWidgets.QMessageBox.warning(self, "Export PNG", "No plot to export")
            return
        filepath, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Plot to PNG", "", "PNG Files (*.png);;All Files (*)"
        )
        if not filepath:
            return
        self._canvas.figure.savefig(filepath, dpi=150, bbox_inches="tight")
        QtWidgets.QMessageBox.information(self, "Export PNG", f"Plot saved to {filepath}")
