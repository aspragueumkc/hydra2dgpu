from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
import pyqtgraph as pg

from swe2d.workbench.services.graph_editor_service import (
    csv_columns,
    delete_graph,
    list_graph_ids,
    load_graphs,
    parse_csv,
    save_hyetograph,
    save_hydrograph,
)

_BC_TYPE_NAMES = {
    102: "102 (Timeseries Flow Q)",
    103: "103 (Timeseries Stage)",
}


def _hours_to_hhmm(h: float) -> str:
    """Convert decimal hours to HH:MM string."""
    if not np.isfinite(h) or h < 0:
        return str(h)
    total_min = int(round(h * 60))
    hrs = total_min // 60
    mins = total_min % 60
    return f"{hrs}:{mins:02d}"


def _parse_time(s: str) -> float | None:
    """Parse time string — accepts '1:30' format or decimal."""
    s = s.strip()
    if not s:
        return None
    if ":" in s:
        parts = s.split(":")
        try:
            return float(parts[0]) + float(parts[1]) / 60
        except (ValueError, IndexError):
            return None
    try:
        return float(s)
    except ValueError:
        return None


class GraphEditorDialog(QDialog):
    """Dialog for creating/editing hyetographs and hydrographs."""

    def __init__(self, gpkg_path: str, parent=None):
        super().__init__(parent)
        self._gpkg_path = gpkg_path
        self._current_type: str | None = None  # "hyetographs" or "hydrographs"
        self._current_id: str | None = None
        self._dirty = False

        self.setWindowTitle("Graph Editor")
        self.resize(900, 700)
        self._build_ui()
        self._load_graph_list()

    # ── UI build ─────────────────────────────────────────────────────

    def _build_ui(self):
        splitter = QSplitter(Qt.Horizontal)

        # Left panel
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self._graph_list = QListWidget()
        self._graph_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self._graph_list.currentItemChanged.connect(self._on_selection_changed)
        left_layout.addWidget(self._graph_list)

        btn_row = QHBoxLayout()
        self._new_btn = QPushButton("+ New")
        self._new_btn.clicked.connect(self._on_new)
        self._delete_btn = QPushButton("Delete")
        self._delete_btn.clicked.connect(self._on_delete)
        btn_row.addWidget(self._new_btn)
        btn_row.addWidget(self._delete_btn)
        btn_row.addStretch()
        left_layout.addLayout(btn_row)

        splitter.addWidget(left)

        # Right panel
        right = QWidget()
        right_layout = QVBoxLayout(right)

        # Name row
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:"))
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("Unique identifier for this graph")
        self._name_edit.textChanged.connect(lambda _: self._mark_dirty())
        name_row.addWidget(self._name_edit)
        right_layout.addLayout(name_row)

        # Type-specific fields row
        spec_row = QHBoxLayout()
        spec_row.addWidget(QLabel("Type:"))
        self._type_label = QLabel("hyetograph")
        spec_row.addWidget(self._type_label)
        spec_row.addStretch()
        self._vt_label = QLabel("Value type:")
        self._vt_combo = QComboBox()
        self._vt_combo.addItems(["intensity", "incremental", "cumulative"])
        self._vt_combo.currentTextChanged.connect(lambda _: self._mark_dirty())
        self._units_label = QLabel("Units:")
        self._units_combo = QComboBox()
        self._units_combo.addItems(["mm/hr", "in/hr", "mm", "in"])
        self._units_combo.currentTextChanged.connect(lambda _: self._mark_dirty())
        self._bc_label = QLabel("BC type:")
        self._bc_combo = QComboBox()
        for _code in sorted(_BC_TYPE_NAMES):
            self._bc_combo.addItem(_BC_TYPE_NAMES[_code], _code)
        self._bc_combo.currentTextChanged.connect(lambda _: self._mark_dirty())
        self._desc_label = QLabel("Description:")
        self._desc_edit = QLineEdit()
        self._desc_edit.textChanged.connect(lambda _: self._mark_dirty())
        spec_row.addWidget(self._vt_label)
        spec_row.addWidget(self._vt_combo)
        spec_row.addWidget(self._units_label)
        spec_row.addWidget(self._units_combo)
        spec_row.addWidget(self._bc_label)
        spec_row.addWidget(self._bc_combo)
        spec_row.addWidget(self._desc_label)
        spec_row.addWidget(self._desc_edit)
        right_layout.addLayout(spec_row)
        self._show_hyeto_fields(True)

        # Table — start with blank rows
        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(["Time", "Value"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.cellChanged.connect(lambda _: self._mark_dirty())
        self._init_blank_rows(25)
        right_layout.addWidget(self._table)

        # Button row
        btn_row = QHBoxLayout()
        add_row_btn = QPushButton("Add Row")
        add_row_btn.clicked.connect(self._add_row)
        del_row_btn = QPushButton("Del Row")
        del_row_btn.clicked.connect(self._del_row)
        load_csv_btn = QPushButton("Load CSV…")
        load_csv_btn.clicked.connect(self._load_csv)
        btn_row.addWidget(add_row_btn)
        btn_row.addWidget(del_row_btn)
        btn_row.addWidget(load_csv_btn)
        btn_row.addStretch()
        right_layout.addLayout(btn_row)

        # Plot
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setBackground("w")
        self._plot_widget.getAxis("bottom").setPen("k")
        self._plot_widget.getAxis("left").setPen("k")
        self._plot_widget.setLabel("bottom", "Time")
        self._plot_widget.setLabel("left", "Value")
        self._plot_curve = self._plot_widget.plot(pen="b")
        right_layout.addWidget(self._plot_widget, stretch=1)

        # Bottom buttons
        bbox = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Close)
        bbox.accepted.connect(self._on_save)
        bbox.rejected.connect(self.reject)
        right_layout.addWidget(bbox)

        splitter.addWidget(right)
        splitter.setSizes([250, 650])

        layout = QVBoxLayout(self)
        layout.addWidget(splitter)

    # ── Data loading ────────────────────────────────────────────────

    def _load_graph_list(self):
        self._graph_list.blockSignals(True)
        self._graph_list.clear()
        ids = list_graph_ids(self._gpkg_path)

        if ids["hyetographs"]:
            h_item = QListWidgetItem("HYETOGRAPHS")
            h_item.setFlags(h_item.flags() & ~Qt.ItemIsSelectable)
            font = h_item.font()
            font.setBold(True)
            h_item.setFont(font)
            self._graph_list.addItem(h_item)
            for gid in ids["hyetographs"]:
                it = QListWidgetItem(f"  {gid}")
                it.setData(Qt.UserRole, ("hyetographs", gid))
                self._graph_list.addItem(it)

        if ids["hydrographs"]:
            h_item = QListWidgetItem("HYDROGRAPHS")
            h_item.setFlags(h_item.flags() & ~Qt.ItemIsSelectable)
            font = h_item.font()
            font.setBold(True)
            h_item.setFont(font)
            self._graph_list.addItem(h_item)
            for gid in ids["hydrographs"]:
                it = QListWidgetItem(f"  {gid}")
                it.setData(Qt.UserRole, ("hydrographs", gid))
                self._graph_list.addItem(it)

        self._graph_list.blockSignals(False)

    def _load_graph_data(self, graph_type: str, gid: str):
        all_data = load_graphs(self._gpkg_path)
        graphs = all_data.get(graph_type, {})
        info = graphs.get(gid)
        if info is None:
            return
        self._current_type = graph_type
        self._current_id = gid

        is_hyeto = graph_type == "hyetographs"
        self._show_hyeto_fields(is_hyeto)

        self._name_edit.blockSignals(True)
        self._name_edit.setText(gid)
        self._name_edit.blockSignals(False)

        if is_hyeto:
            self._vt_combo.blockSignals(True)
            vt = info.get("value_type", "")
            idx = self._vt_combo.findText(vt)
            if idx >= 0:
                self._vt_combo.setCurrentIndex(idx)
            self._vt_combo.blockSignals(False)

            self._units_combo.blockSignals(True)
            u = info.get("units", "")
            idx = self._units_combo.findText(u)
            if idx >= 0:
                self._units_combo.setCurrentIndex(idx)
            self._units_combo.blockSignals(False)
        else:
            self._bc_combo.blockSignals(True)
            bt = str(info.get("bc_type", "0"))
            idx = self._bc_combo.findData(int(bt))
            if idx >= 0:
                self._bc_combo.setCurrentIndex(idx)
            self._bc_combo.blockSignals(False)

            self._desc_edit.blockSignals(True)
            self._desc_edit.setText(info.get("description", ""))
            self._desc_edit.blockSignals(False)

        self._table.blockSignals(True)
        self._table.setRowCount(0)
        for t, v in info["data"]:
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._table.setItem(row, 0, QTableWidgetItem(_hours_to_hhmm(t)))
            self._table.setItem(row, 1, QTableWidgetItem(str(v)))
        # Pad to at least 25 blank rows
        if self._table.rowCount() < 25:
            for _ in range(25 - self._table.rowCount()):
                row = self._table.rowCount()
                self._table.insertRow(row)
                self._table.setItem(row, 0, QTableWidgetItem(""))
                self._table.setItem(row, 1, QTableWidgetItem(""))
        self._table.blockSignals(False)

        self._update_plot()

    def _show_hyeto_fields(self, is_hyeto: bool):
        self._vt_label.setVisible(is_hyeto)
        self._vt_combo.setVisible(is_hyeto)
        self._units_label.setVisible(is_hyeto)
        self._units_combo.setVisible(is_hyeto)
        self._bc_label.setVisible(not is_hyeto)
        self._bc_combo.setVisible(not is_hyeto)
        self._desc_label.setVisible(not is_hyeto)
        self._desc_edit.setVisible(not is_hyeto)
        self._type_label.setText("hyetograph" if is_hyeto else "hydrograph")

    # ── Plot ─────────────────────────────────────────────────────────

    def _init_blank_rows(self, count: int):
        """Fill the table with blank rows."""
        self._table.blockSignals(True)
        for _ in range(count):
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._table.setItem(row, 0, QTableWidgetItem(""))
            self._table.setItem(row, 1, QTableWidgetItem(""))
        self._table.blockSignals(False)

    def _update_plot(self):
        data = self._table_data()
        if not data:
            self._plot_curve.setData([], [])
            return
        times, values = zip(*data)
        self._plot_curve.setData(times, values)

    def _table_data(self) -> List[Tuple[float, float]]:
        """Return parsed (time, value) pairs, skipping blank/invalid rows."""
        data: List[Tuple[float, float]] = []
        for row in range(self._table.rowCount()):
            t_item = self._table.item(row, 0)
            v_item = self._table.item(row, 1)
            if t_item is None or v_item is None:
                continue
            t = _parse_time(t_item.text())
            try:
                v = float(v_item.text())
            except (ValueError, TypeError):
                continue
            if t is None:
                continue
            data.append((t, v))
        return data

    # ── Table actions ────────────────────────────────────────────────

    def _add_row(self):
        self._table.blockSignals(True)
        row = self._table.rowCount()
        self._table.insertRow(row)
        self._table.setItem(row, 0, QTableWidgetItem("0"))
        self._table.setItem(row, 1, QTableWidgetItem("0"))
        self._table.blockSignals(False)
        self._mark_dirty()

    def _del_row(self):
        rows = set(i.row() for i in self._table.selectedIndexes())
        if not rows:
            if self._table.rowCount() > 0:
                rows = {self._table.rowCount() - 1}
        self._table.blockSignals(True)
        for r in sorted(rows, reverse=True):
            self._table.removeRow(r)
        self._table.blockSignals(False)
        self._mark_dirty()

    def _load_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load CSV", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return
        cols = csv_columns(path)
        if len(cols) < 2:
            QMessageBox.warning(self, "CSV Error", "CSV must have at least 2 columns.")
            return

        # Simple column picker dialog
        picker = _ColumnPicker(cols, self)
        if picker.exec() != QDialog.Accepted:
            return
        time_col, value_col = picker.selected_columns()
        data = parse_csv(path, time_col, value_col)
        if not data:
            QMessageBox.information(self, "CSV Import", "No valid data rows found.")
            return

        self._table.blockSignals(True)
        for t, v in data:
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._table.setItem(row, 0, QTableWidgetItem(str(t)))
            self._table.setItem(row, 1, QTableWidgetItem(str(v)))
        self._table.blockSignals(False)
        self._mark_dirty()

    # ── Selection ────────────────────────────────────────────────────

    def _on_selection_changed(self, current: QListWidgetItem, _prev):
        if current is None:
            return
        data = current.data(Qt.UserRole)
        if data is None:
            return
        gtype, gid = data
        self._load_graph_data(gtype, gid)

    # ── Actions ──────────────────────────────────────────────────────

    def _on_new(self):
        """Create a new blank graph (prompts for type)."""
        from qgis.PyQt.QtWidgets import QDialog as _QD
        from qgis.PyQt.QtWidgets import QDialogButtonBox as _QDB

        dlg = _QD(self)
        dlg.setWindowTitle("New Graph")
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel("Type:"))
        type_combo = QComboBox()
        type_combo.addItems(["hyetograph", "hydrograph"])
        layout.addWidget(type_combo)
        bbox = _QDB(_QDB.Ok | _QDB.Cancel)
        bbox.accepted.connect(dlg.accept)
        bbox.rejected.connect(dlg.reject)
        layout.addWidget(bbox)
        if dlg.exec() != _QD.Accepted:
            return

        gtype = "hyetographs" if type_combo.currentText() == "hyetograph" else "hydrographs"
        self._current_type = gtype
        self._current_id = None
        self._name_edit.setText("")
        self._show_hyeto_fields(gtype == "hyetographs")
        self._table.setRowCount(0)
        self._init_blank_rows(25)
        self._plot_curve.setData([], [])
        self._mark_dirty()

    def _on_delete(self):
        if self._current_id is None or self._current_type is None:
            return
        table = "swe2d_hyetographs" if self._current_type == "hyetographs" else "swe2d_hydrographs"
        reply = QMessageBox.question(
            self, "Delete", f"Delete '{self._current_id}'?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        delete_graph(self._gpkg_path, table, self._current_id)
        self._current_id = None
        self._current_type = None
        self._name_edit.setText("")
        self._table.setRowCount(0)
        self._plot_curve.setData([], [])
        self._load_graph_list()

    def _on_save(self):
        gid = self._name_edit.text().strip()
        if not gid:
            QMessageBox.warning(self, "Save", "Name is required.")
            return
        data = self._table_data()
        if not data:
            QMessageBox.warning(self, "Save", "At least one data row is required.")
            return
        if self._current_type == "hyetographs":
            save_hyetograph(
                self._gpkg_path, gid, data,
                value_type=self._vt_combo.currentText(),
                units=self._units_combo.currentText(),
            )
        else:
            try:
                bc = int(self._bc_combo.currentData())
            except ValueError:
                bc = 0
            save_hydrograph(
                self._gpkg_path, gid, data,
                bc_type=bc, description=self._desc_edit.text(),
            )

        self._dirty = False
        self._current_id = gid
        self._load_graph_list()
        # Re-select the saved item
        for i in range(self._graph_list.count()):
            item = self._graph_list.item(i)
            if item and item.data(Qt.UserRole) == (self._current_type, gid):
                self._graph_list.setCurrentItem(item)
                break

    def _mark_dirty(self):
        self._dirty = True
        self._update_plot()

    def closeEvent(self, event):
        if self._dirty:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "Save changes before closing?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            )
            if reply == QMessageBox.Save:
                self._on_save()
            elif reply == QMessageBox.Cancel:
                event.ignore()
                return
        super().closeEvent(event)


class _ColumnPicker(QDialog):
    """Minimal dialog to pick time/value columns from CSV header."""

    def __init__(self, columns: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("CSV Column Mapping")
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Time column:"))
        self._time_combo = QComboBox()
        self._time_combo.addItems(columns)
        layout.addWidget(self._time_combo)

        layout.addWidget(QLabel("Value column:"))
        self._value_combo = QComboBox()
        self._value_combo.addItems(columns)
        if len(columns) > 1:
            self._value_combo.setCurrentIndex(1)
        layout.addWidget(self._value_combo)

        bbox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bbox.accepted.connect(self.accept)
        bbox.rejected.connect(self.reject)
        layout.addWidget(bbox)

    def selected_columns(self) -> Tuple[str, str]:
        return self._time_combo.currentText(), self._value_combo.currentText()
