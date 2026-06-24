"""Batch Simulation Dialog: define, launch, and monitor multiple simulation variants."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional

from qgis.PyQt import QtCore, QtWidgets


_COL_SIM_ID = 0
_COL_PARAMS = 1
_COL_STATUS = 2
_COL_PROGRESS = 3
_COUNT_COLS = 4


class BatchSimulationDialog(QtWidgets.QDialog):
    """Batch simulation dialog with parameter grid and execution monitoring."""

    def __init__(self, parent=None, base_params: Optional[Dict[str, Any]] = None,
                 mesh_gpkg: str = "", results_gpkg: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Batch Simulation")
        self.resize(900, 500)

        self._base_params = dict(base_params or {})
        self._mesh_gpkg = str(mesh_gpkg)
        self._results_gpkg = str(results_gpkg)
        self._param_sets: List[Dict[str, Any]] = []
        self._processes: List[Optional[subprocess.Popen]] = []
        self._next_idx = 0
        self._active = 0
        self._completed = 0
        self._failed = 0
        self._running = False

        self._build_ui()
        self._add_row()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Toolbar
        toolbar = QtWidgets.QHBoxLayout()
        self._add_row_btn = QtWidgets.QPushButton("Add Row")
        self._add_row_btn.clicked.connect(self._add_row)
        self._remove_row_btn = QtWidgets.QPushButton("Remove Selected")
        self._remove_row_btn.clicked.connect(self._remove_selected_rows)
        self._clear_btn = QtWidgets.QPushButton("Clear")
        self._clear_btn.clicked.connect(self._clear_all)
        self._export_btn = QtWidgets.QPushButton("Export JSON")
        self._export_btn.clicked.connect(self._export_json)
        self._import_btn = QtWidgets.QPushButton("Import JSON")
        self._import_btn.clicked.connect(self._import_json)
        toolbar.addWidget(self._add_row_btn)
        toolbar.addWidget(self._remove_row_btn)
        toolbar.addWidget(self._clear_btn)
        toolbar.addSpacing(20)
        toolbar.addWidget(self._export_btn)
        toolbar.addWidget(self._import_btn)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        # Table
        self._table = QtWidgets.QTableWidget(0, _COUNT_COLS)
        self._table.setHorizontalHeaderLabels(["Sim ID", "Parameters (JSON)", "Status", "Progress"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(
            _COL_PARAMS, QtWidgets.QHeaderView.Stretch)
        self._table.setSelectionBehavior(QtWidgets.QTableWidget.SelectRows)
        layout.addWidget(self._table)

        # Run controls
        controls = QtWidgets.QHBoxLayout()
        self._max_workers_spin = QtWidgets.QSpinBox()
        self._max_workers_spin.setRange(1, 16)
        self._max_workers_spin.setValue(4)
        self._max_workers_spin.setToolTip("Max concurrent simulations")
        controls.addWidget(QtWidgets.QLabel("Max workers:"))
        controls.addWidget(self._max_workers_spin)
        controls.addStretch()
        self._run_btn = QtWidgets.QPushButton("Run Batch")
        self._run_btn.clicked.connect(self._run_batch)
        self._cancel_btn = QtWidgets.QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self._cancel_batch)
        self._cancel_btn.setEnabled(False)
        controls.addWidget(self._run_btn)
        controls.addWidget(self._cancel_btn)
        layout.addLayout(controls)

    def _add_row(self):
        """Add a new parameter row."""
        row = self._table.rowCount()
        self._table.insertRow(row)
        sid = f"sim_{row + 1}"
        id_item = QtWidgets.QTableWidgetItem(sid)
        id_item.setFlags(id_item.flags() | QtCore.Qt.ItemIsEditable)
        self._table.setItem(row, _COL_SIM_ID, id_item)
        params_item = QtWidgets.QTableWidgetItem(json.dumps(self._base_params, indent=2))
        params_item.setFlags(params_item.flags() | QtCore.Qt.ItemIsEditable)
        self._table.setItem(row, _COL_PARAMS, params_item)
        self._table.setItem(row, _COL_STATUS, QtWidgets.QTableWidgetItem("pending"))
        self._table.setItem(row, _COL_PROGRESS, QtWidgets.QTableWidgetItem(""))

    def _remove_selected_rows(self):
        rows = sorted(set(i.row() for i in self._table.selectedIndexes()), reverse=True)
        for r in rows:
            self._table.removeRow(r)

    def _clear_all(self):
        self._table.setRowCount(0)

    def _collect_param_sets(self) -> List[Dict[str, Any]]:
        sets = []
        for r in range(self._table.rowCount()):
            sid_item = self._table.item(r, _COL_SIM_ID)
            sid = sid_item.text().strip() if sid_item else f"sim_{r + 1}"
            params_item = self._table.item(r, _COL_PARAMS)
            try:
                params = json.loads(params_item.text())
            except (json.JSONDecodeError, AttributeError):
                params = dict(self._base_params)
            params["id"] = sid
            sets.append(params)
        return sets

    def _add_row_from_entry(self, entry: Dict[str, Any]):
        row = self._table.rowCount()
        self._table.insertRow(row)
        sid = str(entry.get("id", f"sim_{row + 1}"))
        self._table.setItem(row, _COL_SIM_ID, QtWidgets.QTableWidgetItem(sid))
        params_item = QtWidgets.QTableWidgetItem(json.dumps(entry, indent=2))
        params_item.setFlags(params_item.flags() | QtCore.Qt.ItemIsEditable)
        self._table.setItem(row, _COL_PARAMS, params_item)
        self._table.setItem(row, _COL_STATUS, QtWidgets.QTableWidgetItem("pending"))
        self._table.setItem(row, _COL_PROGRESS, QtWidgets.QTableWidgetItem(""))

    def _export_json(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export Batch JSON", "", "JSON (*.json)")
        if not path:
            return
        sets = self._collect_param_sets()
        with open(path, "w") as f:
            json.dump(sets, f, indent=2)

    def _import_json(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import Batch JSON", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, dict):
                data = [data]
            if not isinstance(data, list):
                raise ValueError("JSON must be an array or object")
            self._clear_all()
            for entry in data:
                self._add_row_from_entry(entry)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Import Error", str(exc))

    def _run_batch(self):
        if self._running:
            return
        param_sets = self._collect_param_sets()
        if not param_sets:
            QtWidgets.QMessageBox.information(self, "Batch Run", "No parameter sets defined.")
            return
        if not self._mesh_gpkg or not os.path.isfile(self._mesh_gpkg):
            QtWidgets.QMessageBox.warning(self, "Batch Run", "Mesh GPKG not found.")
            return

        self._running = True
        self._run_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._param_sets = param_sets
        self._processes = [None] * len(param_sets)
        self._next_idx = 0
        self._active = 0
        self._completed = 0
        self._failed = 0

        self._start_next_batch()
        QtCore.QTimer.singleShot(500, self._tick_run)

    def _start_next_batch(self):
        max_workers = self._max_workers_spin.value()
        while self._active < max_workers and self._next_idx < len(self._param_sets):
            idx = self._next_idx
            self._next_idx += 1
            ps = self._param_sets[idx]
            params_json = json.dumps(ps)
            cmd = [
                sys.executable, "-m", "swe2d.cli", "run",
                self._mesh_gpkg, params_json,
                "--results",
                self._results_gpkg or os.path.splitext(self._mesh_gpkg)[0] + "_batch_results.gpkg",
            ]
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            self._processes[idx] = proc
            self._active += 1
            item = self._table.item(idx, _COL_STATUS)
            if item:
                item.setText("running")

    def _tick_run(self):
        if not self._running:
            return
        newly_done = []
        for i, proc in enumerate(self._processes):
            if proc is not None and proc.poll() is not None:
                newly_done.append(i)
        for i in newly_done:
            proc = self._processes[i]
            rc = proc.returncode
            self._active -= 1
            status_item = self._table.item(i, _COL_STATUS)
            progress_item = self._table.item(i, _COL_PROGRESS)
            if rc == 0:
                self._completed += 1
                if status_item:
                    status_item.setText("completed")
            else:
                self._failed += 1
                if status_item:
                    status_item.setText("failed")
                stderr = proc.stderr.read() if proc.stderr else ""
                if progress_item:
                    progress_item.setText(stderr.strip()[:100])
            self._processes[i] = None

        self._start_next_batch()

        done = self._completed + self._failed
        total = len(self._param_sets)
        if done >= total:
            self._running = False
            self._run_btn.setEnabled(True)
            self._cancel_btn.setEnabled(False)
            QtWidgets.QMessageBox.information(
                self, "Batch Complete",
                f"Completed: {self._completed}/{total}\nFailed: {self._failed}",
            )

    def _cancel_batch(self):
        for proc in self._processes:
            if proc is not None and proc.poll() is None:
                proc.terminate()
        self._running = False
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
