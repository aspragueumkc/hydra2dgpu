---
type: plan
status: complete
created: 2026-06-24
completed: 2026-07-25
---

# QGIS Batch UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** QGIS dialog for defining, launching, and monitoring batch simulations with a parameter grid and subprocess-based execution.

**Architecture:** New dialog class `BatchSimulationDialog` using a QTableWidget for the parameter grid. Each row becomes a `hydra run` subprocess. A monitor thread tracks completion. Results accumulate in the project GPKG.

**Tech Stack:** PyQt5, QTableWidget, QThreadPool/QProcess for subprocess management

---

### Task 1: Create BatchSimulationDialog

**Files:**
- Create: `swe2d/workbench/dialogs/batch_simulation_dialog.py`

- [ ] **Step 1: Create dialog skeleton**

```python
"""Batch Simulation Dialog: define, launch, and monitor multiple simulation variants."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

from qgis.PyQt import QtCore, QtGui, QtWidgets


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
        self._processes: List[subprocess.Popen] = []
        self._running = False

        self._build_ui()
        self._populate_from_base()

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
        self._table.horizontalHeader().setSectionResizeMode(_COL_PARAMS, QtWidgets.QHeaderView.Stretch)
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

    def _populate_from_base(self):
        """Initialize with one row from the current base params."""
        self._add_row()
```

- [ ] **Step 2: Implement row management**

Add after `_populate_from_base`:

```python
    def _add_row(self):
        """Add a new parameter row."""
        row = self._table.rowCount()
        self._table.insertRow(row)

        sim_id = f"sim_{row + 1}"
        id_item = QtWidgets.QTableWidgetItem(sim_id)
        id_item.setFlags(id_item.flags() | QtCore.Qt.ItemIsEditable)
        self._table.setItem(row, _COL_SIM_ID, id_item)

        params_item = QtWidgets.QTableWidgetItem(json.dumps(self._base_params, indent=2))
        params_item.setFlags(params_item.flags() | QtCore.Qt.ItemIsEditable)
        self._table.setItem(row, _COL_PARAMS, params_item)

        self._table.setItem(row, _COL_STATUS, QtWidgets.QTableWidgetItem("pending"))
        self._table.setItem(row, _COL_PROGRESS, QtWidgets.QTableWidgetItem(""))

    def _remove_selected_rows(self):
        """Remove selected rows."""
        rows = sorted(set(i.row() for i in self._table.selectedIndexes()), reverse=True)
        for r in rows:
            self._table.removeRow(r)

    def _clear_all(self):
        """Clear all rows."""
        self._table.setRowCount(0)

    def _collect_param_sets(self) -> List[Dict[str, Any]]:
        """Read all rows into param set list."""
        sets = []
        for r in range(self._table.rowCount()):
            sid = self._table.item(r, _COL_SIM_ID).text().strip() if self._table.item(r, _COL_SIM_ID) else f"sim_{r+1}"
            try:
                params = json.loads(self._table.item(r, _COL_PARAMS).text())
            except (json.JSONDecodeError, AttributeError):
                params = dict(self._base_params)
            params["id"] = sid
            sets.append(params)
        return sets
```

- [ ] **Step 3: Implement import/export**

```python
    def _export_json(self):
        """Export parameter sets to JSON file."""
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export Batch JSON", "", "JSON (*.json)")
        if not path:
            return
        sets = self._collect_param_sets()
        with open(path, "w") as f:
            json.dump(sets, f, indent=2)

    def _import_json(self):
        """Import parameter sets from JSON file."""
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
            self._param_sets = data
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Import Error", str(exc))

    def _add_row_from_entry(self, entry: Dict[str, Any]):
        """Add a table row from a param dict."""
        row = self._table.rowCount()
        self._table.insertRow(row)
        sid = str(entry.get("id", f"sim_{row + 1}"))
        self._table.setItem(row, _COL_SIM_ID, QtWidgets.QTableWidgetItem(sid))
        params_item = QtWidgets.QTableWidgetItem(json.dumps(entry, indent=2))
        params_item.setFlags(params_item.flags() | QtCore.Qt.ItemIsEditable)
        self._table.setItem(row, _COL_PARAMS, params_item)
        self._table.setItem(row, _COL_STATUS, QtWidgets.QTableWidgetItem("pending"))
        self._table.setItem(row, _COL_PROGRESS, QtWidgets.QTableWidgetItem(""))
```

- [ ] **Step 4: Implement run/cancel**

```python
    def _run_batch(self):
        """Launch batch execution via subprocess."""
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

        max_workers = self._max_workers_spin.value()
        # Start subprocess for each param set (up to max_workers at once)
        self._param_sets = param_sets
        self._processes = []
        self._next_idx = 0
        self._active = 0
        self._completed = 0
        self._failed = 0

        def _tick():
            self._tick_run()
            if self._running:
                QtCore.QTimer.singleShot(500, _tick)
        QtCore.QTimer.singleShot(500, _tick)
        self._start_next_batch(max_workers)

    def _start_next_batch(self, max_workers: int):
        """Launch subprocesses up to max_workers."""
        while self._active < max_workers and self._next_idx < len(self._param_sets):
            idx = self._next_idx
            self._next_idx += 1
            ps = self._param_sets[idx]
            params_json = json.dumps(ps)
            cmd = [
                sys.executable, "-m", "swe2d.cli", "run",
                self._mesh_gpkg, params_json,
                "--results", self._results_gpkg or os.path.splitext(self._mesh_gpkg)[0] + "_batch_results.gpkg",
            ]
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            self._processes.append(proc)
            self._active += 1
            self._table.item(idx, _COL_STATUS).setText("running")

    def _tick_run(self):
        """Poll running processes, update UI."""
        if not self._running:
            return
        newly_done = []
        for i, proc in enumerate(self._processes):
            if proc.poll() is not None:
                newly_done.append(i)
        for i in reversed(newly_done):
            proc = self._processes[i]
            returncode = proc.returncode
            self._active -= 1
            if returncode == 0:
                self._completed += 1
                self._table.item(i, _COL_STATUS).setText("completed")
            else:
                self._failed += 1
                stderr = proc.stderr.read() if proc.stderr else ""
                self._table.item(i, _COL_STATUS).setText("failed")
                self._table.item(i, _COL_PROGRESS).setText(stderr.strip()[:100])
            self._processes[i] = None

        # Launch more if slots open
        if self._next_idx < len(self._param_sets):
            self._start_next_batch(self._max_workers_spin.value())

        done = self._completed + self._failed
        total = len(self._param_sets)
        if done >= total:
            self._running = False
            self._run_btn.setEnabled(True)
            self._cancel_btn.setEnabled(False)
            QtWidgets.QMessageBox.information(
                self, "Batch Complete",
                f"Completed: {self._completed}/{total}\nFailed: {self._failed}"
            )

    def _cancel_batch(self):
        """Terminate all running processes."""
        for proc in self._processes:
            if proc is not None and proc.poll() is None:
                proc.terminate()
        self._running = False
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
```

- [ ] **Step 5: Syntax check**

```bash
python3 -c "import py_compile; py_compile.compile('swe2d/workbench/dialogs/batch_simulation_dialog.py', doraise=True); print('OK')"
```

Expected: OK

- [ ] **Step 6: Commit**

```bash
git add swe2d/workbench/dialogs/batch_simulation_dialog.py && git commit -m "feat: add BatchSimulationDialog with parameter grid and subprocess execution"
```

### Task 2: Wire batch dialog into workbench

**Files:**
- Modify: `swe2d/workbench/studio_dialog.py`

- [ ] **Step 1: Add "Batch Simulation" button to the run controls area**

Find the run controls section (around the "Run" button, likely in the model_tab_view or studio_dialog).  Add a button:

```python
self._batch_btn = QtWidgets.QPushButton("Batch Simulation...")
self._batch_btn.clicked.connect(self._open_batch_dialog)
```

Place it near the main "Run" button but visually separated.

- [ ] **Step 2: Add handler**

```python
def _open_batch_dialog(self):
    """Open the batch simulation dialog."""
    from swe2d.workbench.dialogs.batch_simulation_dialog import BatchSimulationDialog

    gpkg = str(self._model_gpkg_path_edit.text()) if hasattr(self, "_model_gpkg_path_edit") else ""
    if not gpkg:
        gpkg = str(getattr(self, "_model_gpkg_path", ""))
    if not gpkg:
        QtWidgets.QMessageBox.warning(self, "Batch", "No project GeoPackage path set.")
        return

    # Build base params from current workbench state
    base_params = {
        "mesh": "",  # filled from current mesh name if known
        "bc_lines": str(self._map_tab_view.bc_layers_combo.currentText()) if hasattr(self._map_tab_view, "bc_layers_combo") else "",
        "params": {
            "rain_rate_mmhr": self._model_tab_view.rain_rate_spin.value() if hasattr(self._model_tab_view, "rain_rate_spin") else 0.0,
            "n_mann": self._model_tab_view.n_mann_spin.value() if hasattr(self._model_tab_view, "n_mann_spin") else 0.035,
            "duration_s": 3600.0,
        },
    }

    dlg = BatchSimulationDialog(
        parent=self,
        base_params=base_params,
        mesh_gpkg=gpkg,
        results_gpkg=gpkg,
    )
    dlg.exec()
```

- [ ] **Step 3: Syntax check**

```bash
python3 -c "import py_compile; py_compile.compile('swe2d/workbench/studio_dialog.py', doraise=True); print('OK')"
```

Expected: OK

- [ ] **Step 4: Commit**

```bash
git add swe2d/workbench/studio_dialog.py && git commit -m "feat: wire batch simulation dialog into workbench run controls"
```

### Task 3: Test batch dialog opens

- [ ] **Step 1: Verify no import errors**

```bash
python3 -c "
from swe2d.workbench.dialogs.batch_simulation_dialog import BatchSimulationDialog
print('Import OK')
"
```

Expected: `Import OK`

- [ ] **Step 2: Commit (no code changes)**

```bash
git add -A && git commit -m "test: verify batch dialog imports cleanly" || true
```
