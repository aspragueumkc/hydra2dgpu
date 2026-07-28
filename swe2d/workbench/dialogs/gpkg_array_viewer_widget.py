"""swe2d/workbench/dialogs/gpkg_array_viewer_widget.py

Reusable widget to display a deserialized numpy array in a QTableWidget
with a mini-plot tab.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from qgis.PyQt import QtCore, QtWidgets

from swe2d.workbench.dialogs._plot_utils import try_import_matplotlib_qt

logger = logging.getLogger(__name__)

FigureCanvasQt, Figure, _mtri = try_import_matplotlib_qt()


class ArrayViewerWidget(QtWidgets.QWidget):
    """Dual-tab widget showing array data as a table + quick plot.

    Displays a numpy array: 1D arrays show as a single column;
    2D arrays show rows=dim0, cols=dim1 with slice controls.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._array: Optional[np.ndarray] = None
        self._col_name: str = ""
        self._slice_row: int = 0
        self._slice_col: int = 0

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        self._info_lbl = QtWidgets.QLabel("No array selected")
        root.addWidget(self._info_lbl)

        self._slice_row_layout = QtWidgets.QHBoxLayout()
        self._slice_row_lbl = QtWidgets.QLabel("Row (timestep):")
        self._slice_row_spin = QtWidgets.QSpinBox()
        self._slice_row_spin.setRange(0, 0)
        self._slice_row_spin.valueChanged.connect(self._on_slice_changed)
        self._slice_row_layout.addWidget(self._slice_row_lbl)
        self._slice_row_layout.addWidget(self._slice_row_spin)

        self._slice_col_layout = QtWidgets.QHBoxLayout()
        self._slice_col_lbl = QtWidgets.QLabel("Col (cell):")
        self._slice_col_spin = QtWidgets.QSpinBox()
        self._slice_col_spin.setRange(0, 0)
        self._slice_col_spin.valueChanged.connect(self._on_slice_changed)
        self._slice_col_layout.addWidget(self._slice_col_lbl)
        self._slice_col_layout.addWidget(self._slice_col_spin)

        self._slice_widget = QtWidgets.QWidget()
        slice_inner = QtWidgets.QHBoxLayout(self._slice_widget)
        slice_inner.setContentsMargins(0, 0, 0, 0)
        slice_inner.addLayout(self._slice_row_layout)
        slice_inner.addLayout(self._slice_col_layout)
        slice_inner.addStretch(1)
        self._slice_widget.setVisible(False)
        root.addWidget(self._slice_widget)

        self._tabs = QtWidgets.QTabWidget()
        root.addWidget(self._tabs, stretch=1)

        self._data_table = QtWidgets.QTableWidget()
        self._data_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self._data_table.setAlternatingRowColors(True)
        self._tabs.addTab(self._data_table, "Array Data")

        self._plot_widget = QtWidgets.QWidget()
        self._plot_layout = QtWidgets.QVBoxLayout(self._plot_widget)
        self._plot_layout.setContentsMargins(0, 0, 0, 0)
        self._canvas = None
        self._tabs.addTab(self._plot_widget, "Quick Plot")

    def show_array(self, array: np.ndarray, col_name: str = ""):
        """Display a numpy array. Pass None to clear."""
        self._array = array
        self._col_name = col_name
        self._column_labels = [""]

        if array is None:
            self._info_lbl.setText("No array selected")
            self._data_table.setRowCount(0)
            self._data_table.setColumnCount(0)
            self._slice_widget.setVisible(False)
            self._col_name = ""
            return

        shape_str = "×".join(str(s) for s in array.shape)
        dtype_str = str(array.dtype)
        self._info_lbl.setText(f"{col_name}: {dtype_str}[{shape_str}]" if col_name else f"{dtype_str}[{shape_str}]")

        ndim = array.ndim
        self._slice_widget.setVisible(ndim == 2)

        if ndim == 2:
            self._slice_row_spin.blockSignals(True)
            self._slice_col_spin.blockSignals(True)
            self._slice_row_spin.setRange(0, array.shape[0] - 1)
            self._slice_col_spin.setRange(0, array.shape[1] - 1)
            self._slice_row_spin.setValue(0)
            self._slice_col_spin.setValue(0)
            self._slice_row_spin.blockSignals(False)
            self._slice_col_spin.blockSignals(False)
            self._populate_table_2d(array)
        else:
            self._populate_table_1d(array)

        self._update_quick_plot()

    def show_combined(self, column_labels, combined_array, row_index_label="Index"):
        """Display a 2D ``(rows × cols)`` ``np.ndarray`` with an index column.

        ``column_labels`` provides display names for each array column; the
        leading index column is prepended automatically using
        ``row_index_label``. If ``combined_array`` is None or empty, the
        viewer is cleared.
        """
        if combined_array is None or len(column_labels) == 0:
            self.show_array(None)
            return

        if combined_array.ndim != 2:
            self.show_array(combined_array, column_labels[0])
            return

        n = combined_array.shape[0]
        all_labels = [row_index_label] + list(column_labels)
        self._column_labels = all_labels
        self._array = combined_array
        self._col_name = ", ".join(column_labels)

        info_parts = ", ".join(f"{name}[{n}]" for name in column_labels)
        self._info_lbl.setText(f"Combined: {info_parts}")

        self._slice_widget.setVisible(False)

        self._data_table.setColumnCount(len(all_labels))
        self._data_table.setHorizontalHeaderLabels(all_labels)
        self._data_table.setRowCount(n)
        for i in range(n):
            self._data_table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(i)))
            for j in range(combined_array.shape[1]):
                self._data_table.setItem(i, j + 1, QtWidgets.QTableWidgetItem(f"{combined_array[i, j]:.6g}"))
        self._data_table.resizeColumnsToContents()

        self._update_quick_plot_combined(column_labels, combined_array)

    def _update_quick_plot_combined(self, column_labels, combined_array):
        """Render a quick plot using combined_array (column 0 as X, others as Y series)."""
        if FigureCanvasQt is None or combined_array is None or combined_array.ndim != 2:
            return

        if self._canvas is not None:
            self._plot_layout.removeWidget(self._canvas)
            self._canvas.deleteLater()
            self._canvas = None

        from matplotlib.figure import Figure
        fig = Figure(figsize=(5, 3))
        ax = fig.add_subplot(111)

        if combined_array.shape[1] > 1:
            x = combined_array[:, 0]
            for j in range(1, combined_array.shape[1]):
                ax.plot(x, combined_array[:, j], label=column_labels[j], linewidth=1)
            ax.set_xlabel(column_labels[0])
            ax.set_ylabel("value")
            ax.legend(fontsize=8)
        else:
            ax.plot(combined_array[:, 0])
            ax.set_xlabel("Index")
            ax.set_ylabel(column_labels[0])

        ax.set_title(f"Quick Plot - {', '.join(column_labels)}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        self._canvas = FigureCanvasQt(fig)
        self._plot_layout.addWidget(self._canvas)

    def _populate_table_1d(self, array: np.ndarray):
        """Fill table for 1D array."""
        self._data_table.setColumnCount(2)
        self._data_table.setHorizontalHeaderLabels(["Index", self._col_name or "Value"])
        self._data_table.setRowCount(len(array))
        for i in range(len(array)):
            self._data_table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(i)))
            self._data_table.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{array[i]:.6g}"))
        self._data_table.resizeColumnsToContents()

    def _populate_table_2d(self, array: np.ndarray):
        """Fill table showing entire 2D array (cells as columns, timesteps as rows)."""
        self._data_table.setColumnCount(array.shape[1])
        self._data_table.setRowCount(array.shape[0])
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                self._data_table.setItem(i, j, QtWidgets.QTableWidgetItem(f"{array[i, j]:.6g}"))
        self._data_table.resizeColumnsToContents()

    def _on_slice_changed(self):
        if self._array is None or self._array.ndim < 2:
            return
        r = self._slice_row_spin.value()
        c = self._slice_col_spin.value()
        if self._tabs.currentIndex() == 0:
            self._populate_table_slice_1d(self._array[r, :], c)
        self._update_quick_plot()

    def _populate_table_slice_1d(self, array_1d: np.ndarray, col_idx: int):
        """Show a 1D slice of a 2D array (one column of the 2D matrix)."""
        self._data_table.setColumnCount(2)
        self._data_table.setHorizontalHeaderLabels(["Index", f"{self._col_name} [{col_idx}]"])
        self._data_table.setRowCount(len(array_1d))
        for i in range(len(array_1d)):
            self._data_table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(i)))
            self._data_table.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{array_1d[i]:.6g}"))
        self._data_table.resizeColumnsToContents()

    def get_slice_for_plot(self) -> Optional[np.ndarray]:
        """Return the currently visible 1D slice for XY plotting."""
        if self._array is None:
            return None
        if self._array.ndim == 1:
            return self._array
        if self._array.ndim >= 2:
            r = self._slice_row_spin.value()
            return self._array[r, :]
        return None

    def _update_quick_plot(self):
        if FigureCanvasQt is None:
            return
        if self._array is None:
            return

        if self._canvas is not None:
            self._plot_layout.removeWidget(self._canvas)
            self._canvas.deleteLater()
            self._canvas = None

        from matplotlib.figure import Figure
        fig = Figure(figsize=(5, 3))
        ax = fig.add_subplot(111)

        if self._array.ndim == 1:
            ax.plot(self._array)
            ax.set_xlabel("Index")
            ax.set_ylabel(self._col_name or "Value")
        elif self._array.ndim >= 2:
            r = self._slice_row_spin.value()
            ax.plot(self._array[r, :])
            ax.set_xlabel("Index")
            ax.set_ylabel(f"{self._col_name} [row {r}]" if self._col_name else f"Row {r}")

        ax.set_title(f"Quick Plot - {self._col_name}" if self._col_name else "Quick Plot")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        self._canvas = FigureCanvasQt(fig)
        self._plot_layout.addWidget(self._canvas)
