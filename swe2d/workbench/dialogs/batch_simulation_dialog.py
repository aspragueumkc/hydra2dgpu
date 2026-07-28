"""Batch Simulation Dialog: define, launch, and monitor multiple simulation variants.

Features:
  - Parameter grid with multi-line JSON editing (double-click a row)
  - "Snapshot Current Setup" — pulls current widget values from the main dialog
  - "From GPKG" — imports run metadata from a results GeoPackage
  - Export/Import JSON files for sharing batch configurations
  - Each subprocess gets a unique status file for progress monitoring
"""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)

import json
import os
import sqlite3
import tempfile
from typing import Any, Dict, List, Optional

from qgis.PyQt import QtCore, QtWidgets

from swe2d.workbench.services.batch_manager import BatchManager, BatchConfig, SimState


_COL_SIM_ID = 0
_COL_PARAMS = 1
_COL_STATUS = 2
_COL_PROGRESS = 3
_COUNT_COLS = 4


# ── Helper: convert widget suffix names to CLI param names ────────────

_WIDGET_TO_CLI_MAP = {
    "n_mann_spin": "n_mann",
    "cfl_spin": "cfl",
    "h_min_spin": "h_min",
    "dt_spin": "dt_max",
    "initial_dt_spin": "initial_dt",
    "shallow_damping_depth_spin": "shallow_damping_depth",
    "depth_cap_spin": "depth_cap",
    "momentum_cap_min_speed_spin": "momentum_cap_min_speed",
    "momentum_cap_celerity_mult_spin": "momentum_cap_celerity_mult",
    "max_inv_area_spin": "max_inv_area",
    "cfl_lambda_cap_spin": "cfl_lambda_cap",
    "max_rel_depth_increase_spin": "max_rel_depth_increase",
    "max_source_depth_step_spin": "source_depth_step_cap",
    "max_source_rate_spin": "source_rate_cap",
    "source_cfl_beta_spin": "source_cfl_beta",
    "source_max_substeps_spin": "source_max_substeps",
    "rain_rate_spin": "rain_rate_mmhr",
    "gpu_diag_sync_interval_spin": "gpu_diag_sync_interval_steps",
    "tiny_wet_cell_threshold_spin": "tiny_wet_cell_threshold",
    "front_flux_damping_spin": "front_flux_damping",
    "open_bc_relax_spin": "open_bc_relaxation",
    "k_mann_spin": "k_mann",
}

_BOOL_WIDGET_TO_CLI_MAP = {
    "adaptive_cfl_dt_chk": "adaptive_cfl_dt",
    "source_true_subcycling_chk": "source_true_subcycling",
    "source_imex_split_chk": "source_imex_split",
    "active_set_hysteresis_chk": "active_set_hysteresis",
    "enable_cuda_graphs_chk": "enable_cuda_graphs",
    "swe2d_perf_mode_chk": "swe2d_perf_mode",
    "culvert_face_flux_chk": "use_culvert_face_flux",
    "use_redistribution_chk": "use_redistribution",
}

_COMBO_WIDGET_TO_CLI_MAP = {
    "reconstruction_combo": "spatial_scheme",
    "temporal_order_combo": "temporal_scheme",
    "tiny_mode_combo": "tiny_mode",
    "culvert_solver_mode_combo": "culvert_solver_mode",
    "drainage_gpu_method_combo": "drainage_gpu_method",
    "bridge_stacked_coupling_mode_combo": "bridge_coupling_mode",
    "degen_mode_combo": "degen_mode",
}


def _widget_params_to_run_params(widget_params: dict) -> dict:
    """Convert the flat widget-param dict into a CLI run parameters dict.

    Strips common widget suffixes (``_spin``, ``_chk``, ``_combo``) and
    maps to the expected keys in the run JSON ``params`` block.
    """
    rp: Dict[str, Any] = {}

    # Scalar values from spin/double-spin widgets
    for wkey, ckey in _WIDGET_TO_CLI_MAP.items():
        val = widget_params.get(wkey)
        if val is not None:
            rp[ckey] = float(val)

    # Boolean values from checkboxes
    for wkey, ckey in _BOOL_WIDGET_TO_CLI_MAP.items():
        val = widget_params.get(wkey)
        if val is not None:
            rp[ckey] = bool(val)

    # Combo box currentData values
    for wkey, ckey in _COMBO_WIDGET_TO_CLI_MAP.items():
        val = widget_params.get(wkey)
        if val is not None:
            rp[ckey] = int(val)

    # Duration: parse from run_time_edit (decimal hours or HH:MM)
    raw_dur = str(widget_params.get("run_time_edit", "") or "").strip()
    if raw_dur:
        if ":" in raw_dur:
            parts = raw_dur.split(":")
            try:
                dur_hrs = float(parts[0]) + float(parts[1]) / 60.0
            except (ValueError, IndexError):
                dur_hrs = 1.0
        else:
            try:
                dur_hrs = float(raw_dur)
            except ValueError:
                dur_hrs = 1.0
        rp["duration_s"] = dur_hrs * 3600.0

    # Output interval
    raw_out = str(widget_params.get("output_interval_edit", "") or "").strip()
    if raw_out:
        if ":" in raw_out:
            parts = raw_out.split(":")
            try:
                out_hrs = float(parts[0]) + float(parts[1]) / 60.0
            except (ValueError, IndexError):
                out_hrs = 0.5
        else:
            try:
                out_hrs = float(raw_out)
            except ValueError:
                out_hrs = 0.5
        rp["output_interval_s"] = out_hrs * 3600.0

    # Save-max-only (inferred from save_max_only_chk if present)
    smc = widget_params.get("save_max_only_chk")
    if smc is not None:
        rp["save_max_only"] = bool(smc)

    return rp


def _parse_run_duration_hours(text: str) -> float:
    """Parse a run duration string to hours.  Accepts HH:MM or decimal."""
    s = str(text or "").strip()
    if not s:
        return 1.0
    if ":" in s:
        parts = s.split(":")
        try:
            return float(parts[0]) + float(parts[1]) / 60.0
        except (ValueError, IndexError):
            return 1.0
    try:
        return float(s)
    except ValueError:
        return 1.0


# ── JSON Editor Dialog ────────────────────────────────────────────────


class JsonEditorDialog(QtWidgets.QDialog):
    """Multi-line JSON editor for a single batch row's parameters."""

    def __init__(self, param_json: str, title: str = "Edit Parameters", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(700, 500)

        layout = QtWidgets.QVBoxLayout(self)
        self._editor = QtWidgets.QPlainTextEdit()
        self._editor.setPlainText(param_json)
        self._editor.setTabStopDistance(
            self._editor.fontMetrics().horizontalAdvance(" ") * 2
        )
        layout.addWidget(self._editor)

        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addStretch()
        ok_btn = QtWidgets.QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

    def get_json(self) -> str:
        return str(self._editor.toPlainText()).strip()


# ── Batch Simulation Dialog ───────────────────────────────────────────


class BatchSimulationDialog(QtWidgets.QDialog):
    """Batch simulation dialog with parameter grid and execution monitoring.

    Each subprocess gets a unique status file for progress monitoring.
    A "Check Batch Status" button reads all status files on demand.

    Features:
    - Snapshot Current Setup — pulls widget values from the parent dialog
    - From GPKG — imports run metadata from a results GeoPackage
    - Double-click a row to edit parameters in a multi-line JSON editor
    """

    def __init__(self, parent=None, base_params: Optional[Dict[str, Any]] = None,
                 mesh_gpkg: str = "", batch_manager: Optional[BatchManager] = None):
        super().__init__(parent)
        self.setWindowTitle("Batch Simulation")
        self.resize(900, 500)

        self._base_params = dict(base_params or {})
        self._mesh_gpkg = str(mesh_gpkg)
        self._mesh_name = ""
        self._param_sets: List[Dict[str, Any]] = []
        self._batch_manager = batch_manager

        self._build_ui()

        # Auto-populate mesh selector from the current model if available
        parent_view = self.parent()
        if parent_view is not None:
            model_gpkg = str(getattr(parent_view, "_model_gpkg_path", "") or "")
            mesh_name = str(
                (getattr(parent_view, "_mesh_data", None) or {}).get("mesh_name", "")
                or ""
            )
            if model_gpkg and mesh_name and os.path.isfile(model_gpkg):
                self._set_mesh(model_gpkg, mesh_name)

        self._add_row()

    # ── UI Construction ───────────────────────────────────────────────

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Mesh selection
        mesh_layout = QtWidgets.QHBoxLayout()
        mesh_layout.addWidget(QtWidgets.QLabel("Mesh:"))
        self._mesh_display_edit = QtWidgets.QLineEdit()
        self._mesh_display_edit.setReadOnly(True)
        self._mesh_display_edit.setPlaceholderText("Select a mesh...")
        mesh_layout.addWidget(self._mesh_display_edit, 1)
        self._select_mesh_btn = QtWidgets.QPushButton("Select Mesh...")
        self._select_mesh_btn.setToolTip(
            "Choose a GeoPackage and select a mesh from it."
        )
        self._select_mesh_btn.clicked.connect(self._select_mesh)
        mesh_layout.addWidget(self._select_mesh_btn)
        self._apply_mesh_all_btn = QtWidgets.QPushButton("Apply to All")
        self._apply_mesh_all_btn.setToolTip("Set the selected mesh name on ALL rows")
        self._apply_mesh_all_btn.setEnabled(False)
        self._apply_mesh_all_btn.clicked.connect(self._apply_mesh_to_all)
        mesh_layout.addWidget(self._apply_mesh_all_btn)
        layout.addLayout(mesh_layout)

        # Toolbar
        toolbar = QtWidgets.QHBoxLayout()
        self._add_row_btn = QtWidgets.QPushButton("Add Row")
        self._add_row_btn.setToolTip("Add a new blank parameter row to the table.")
        self._add_row_btn.clicked.connect(self._add_row)
        self._remove_row_btn = QtWidgets.QPushButton("Remove Selected")
        self._remove_row_btn.setToolTip("Remove the selected rows from the table.")
        self._remove_row_btn.clicked.connect(self._remove_selected_rows)
        self._clear_btn = QtWidgets.QPushButton("Clear")
        self._clear_btn.setToolTip("Clear all rows from the table.")
        self._clear_btn.clicked.connect(self._clear_all)
        self._export_btn = QtWidgets.QPushButton("Export JSON")
        self._export_btn.setToolTip("Export batch configuration to a JSON file.")
        self._export_btn.clicked.connect(self._export_json)
        self._import_btn = QtWidgets.QPushButton("Import JSON")
        self._import_btn.setToolTip("Import batch configuration from a JSON file.")
        self._import_btn.clicked.connect(self._import_json)
        self._edit_sel_btn = QtWidgets.QPushButton("Edit Selected")
        self._edit_sel_btn.setToolTip("Edit the selected row's JSON in a proper multi-line editor")
        self._edit_sel_btn.clicked.connect(self._edit_selected_row)
        self._view_log_btn = QtWidgets.QPushButton("View Log")
        self._view_log_btn.setToolTip("View log for the selected simulation")
        self._view_log_btn.clicked.connect(self._open_log_panel)
        toolbar.addWidget(self._add_row_btn)
        toolbar.addWidget(self._remove_row_btn)
        toolbar.addWidget(self._clear_btn)
        toolbar.addSpacing(20)
        toolbar.addWidget(self._edit_sel_btn)
        toolbar.addWidget(self._view_log_btn)
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
        self._table.cellDoubleClicked.connect(self._on_table_double_click)
        layout.addWidget(self._table)

        # Data source toolbar
        data_toolbar = QtWidgets.QHBoxLayout()
        self._snapshot_btn = QtWidgets.QPushButton("Snapshot Current Setup")
        self._snapshot_btn.setToolTip(
            "Pull current solver parameters from the main dialog and add as a new row"
        )
        self._snapshot_btn.clicked.connect(self._snapshot_current_setup)
        self._from_gpkg_btn = QtWidgets.QPushButton("From GPKG")
        self._from_gpkg_btn.setToolTip(
            "Import run settings from a results GeoPackage and add as rows"
        )
        self._from_gpkg_btn.clicked.connect(self._import_from_gpkg)
        self._apply_mesh_selected_btn = QtWidgets.QPushButton("Apply to Selected")
        self._apply_mesh_selected_btn.setToolTip(
            "Set the selected mesh name on all currently selected table rows"
        )
        self._apply_mesh_selected_btn.setEnabled(False)
        self._apply_mesh_selected_btn.clicked.connect(self._apply_mesh_to_selected)
        data_toolbar.addWidget(self._snapshot_btn)
        data_toolbar.addWidget(self._from_gpkg_btn)
        data_toolbar.addWidget(self._apply_mesh_selected_btn)
        data_toolbar.addStretch()
        layout.addLayout(data_toolbar)

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
        self._run_btn.setToolTip("Start executing all batch parameter sets.")
        self._run_btn.clicked.connect(self._run_batch)
        self._cancel_btn = QtWidgets.QPushButton("Cancel")
        self._cancel_btn.setToolTip("Cancel the currently running batch.")
        self._cancel_btn.clicked.connect(self._cancel_batch)
        self._cancel_btn.setEnabled(False)
        controls.addWidget(self._run_btn)
        controls.addWidget(self._cancel_btn)
        layout.addLayout(controls)

    # ── Modeless Lifecycle ───────────────────────────────────────────────

    def closeEvent(self, event):
        """Hide instead of destroy if a batch is running."""
        if self._batch_manager is not None and self._batch_manager.is_running():
            event.ignore()
            self.hide()
        else:
            event.accept()

    def showEvent(self, event):
        """Reconnect signals and refresh state when dialog is shown."""
        super().showEvent(event)
        self.connect_batch_signals()
        self._refresh_table_from_manager()

    def hideEvent(self, event):
        """Disconnect signals when dialog is hidden."""
        super().hideEvent(event)
        self.disconnect_batch_signals()

    # ── BatchManager Signal Wiring ───────────────────────────────────────

    def connect_batch_signals(self):
        """Connect to BatchManager signals for live updates."""
        bm = self._batch_manager
        if not bm:
            return
        bm.sim_started.connect(self._on_sim_started)
        bm.sim_progress.connect(self._on_sim_progress)
        bm.sim_completed.connect(self._on_sim_completed)
        bm.sim_failed.connect(self._on_sim_failed)
        bm.log_message.connect(self._on_log_message)
        bm.batch_finished.connect(self._on_batch_finished)

    def disconnect_batch_signals(self):
        """Disconnect from BatchManager signals."""
        bm = self._batch_manager
        if not bm:
            return
        try:
            bm.sim_started.disconnect(self._on_sim_started)
            bm.sim_progress.disconnect(self._on_sim_progress)
            bm.sim_completed.disconnect(self._on_sim_completed)
            bm.sim_failed.disconnect(self._on_sim_failed)
            bm.log_message.disconnect(self._on_log_message)
            bm.batch_finished.disconnect(self._on_batch_finished)
        except (TypeError, RuntimeError):
            pass

    def _find_row_by_sim_id(self, sim_id: str) -> int:
        """Find the table row for a given sim_id."""
        for i in range(self._table.rowCount()):
            item = self._table.item(i, _COL_SIM_ID)
            if item and item.text() == sim_id:
                return i
        return -1

    def _on_sim_started(self, sim_id: str):
        row = self._find_row_by_sim_id(sim_id)
        if row >= 0:
            item = self._table.item(row, _COL_STATUS)
            if item:
                item.setText("running")

    def _on_sim_progress(self, sim_id: str, percent: float, status_text: str):
        row = self._find_row_by_sim_id(sim_id)
        if row >= 0:
            progress_item = self._table.item(row, _COL_PROGRESS)
            if progress_item:
                progress_item.setText(f"{percent:.0f}% {status_text}")

    def _on_sim_completed(self, sim_id: str, results_path: str):
        row = self._find_row_by_sim_id(sim_id)
        if row >= 0:
            item = self._table.item(row, _COL_STATUS)
            if item:
                item.setText("completed")
            progress_item = self._table.item(row, _COL_PROGRESS)
            if progress_item:
                progress_item.setText("100%")

    def _on_sim_failed(self, sim_id: str, error: str):
        row = self._find_row_by_sim_id(sim_id)
        if row >= 0:
            item = self._table.item(row, _COL_STATUS)
            if item:
                item.setText("failed")
            progress_item = self._table.item(row, _COL_PROGRESS)
            if progress_item:
                progress_item.setText(error[:100])

    def _on_log_message(self, sim_id: str, line: str):
        """Append log line to the main workbench log."""
        parent_log = getattr(self.parent(), "_log", None)
        if parent_log:
            parent_log(f"batch> [{sim_id}] {line}")

    def _on_batch_finished(self, cancelled: bool, completed: int, failed: int):
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        total = completed + failed
        msg = f"Completed: {completed}/{total}\nFailed: {failed}"
        if cancelled:
            msg = "(Cancelled)\n" + msg
        QtWidgets.QMessageBox.information(self, "Batch Complete", msg)

    def _refresh_table_from_manager(self):
        """Populate table from BatchManager state (for reopening)."""
        if not self._batch_manager:
            return
        status = self._batch_manager.get_status()
        for sim_id, state in status.items():
            row = self._find_row_by_sim_id(sim_id)
            if row >= 0:
                status_item = self._table.item(row, _COL_STATUS)
                if status_item:
                    status_item.setText(state.status)
                progress_item = self._table.item(row, _COL_PROGRESS)
                if progress_item:
                    if state.status == "completed":
                        progress_item.setText("100%")
                    elif state.status == "failed":
                        progress_item.setText((state.error or "")[:100])
                    elif state.status == "running":
                        progress_item.setText(f"{state.progress:.0f}% {state.status_text}")

    # ── Mesh Selection ─────────────────────────────────────────────────

    def _select_mesh(self):
        """Open a file picker, then a mesh picker, and store the result."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select GeoPackage",
            "",
            "GeoPackage (*.gpkg *.gpkgx);;All Files (*)",
        )
        if not path:
            return
        path = str(path).strip()
        if not os.path.isfile(path):
            QtWidgets.QMessageBox.warning(
                self, "Select Mesh", f"GeoPackage not found:\n{path}"
            )
            return
        from swe2d.workbench.dialogs.mesh_picker_dialog import MeshPickerDialog
        picker = MeshPickerDialog(path, parent=self)
        if picker.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        mesh_name = picker.selected_mesh_name()
        if not mesh_name:
            return
        self._set_mesh(path, mesh_name)

    def _set_mesh(self, gpkg_path: str, mesh_name: str) -> None:
        """Store the selected mesh and update the read-only display."""
        self._mesh_gpkg = str(gpkg_path).strip()
        self._mesh_name = str(mesh_name).strip()
        display = f"{self._mesh_name}"
        if self._mesh_gpkg:
            display += f"  ({os.path.basename(self._mesh_gpkg)})"
        self._mesh_display_edit.setText(display)
        self._mesh_display_edit.setToolTip(
            f"{self._mesh_name}\n{self._mesh_gpkg}"
        )
        self._apply_mesh_all_btn.setEnabled(bool(self._mesh_name))
        self._apply_mesh_selected_btn.setEnabled(bool(self._mesh_name))

    # ── Row Management ────────────────────────────────────────────────

    def _add_row(self):
        """Add a new parameter row with default base params."""
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

    def _add_row_from_entry(self, entry: Dict[str, Any], prepend: bool = False):
        """Add a row from a pre-built param dict."""
        if prepend:
            self._table.insertRow(0)
            row = 0
        else:
            row = self._table.rowCount()
            self._table.insertRow(row)
        sid = str(entry.get("id", f"sim_{row + 1}"))
        self._table.setItem(row, _COL_SIM_ID, QtWidgets.QTableWidgetItem(sid))
        params_item = QtWidgets.QTableWidgetItem(json.dumps(entry, indent=2))
        params_item.setFlags(params_item.flags() | QtCore.Qt.ItemIsEditable)
        self._table.setItem(row, _COL_PARAMS, params_item)
        self._table.setItem(row, _COL_STATUS, QtWidgets.QTableWidgetItem("pending"))
        self._table.setItem(row, _COL_PROGRESS, QtWidgets.QTableWidgetItem(""))

    # ── JSON Editor (multi-line) ──────────────────────────────────────

    def _on_table_double_click(self, row: int, col: int):
        """Open the multi-line JSON editor on double-click if the params
        column or sim ID column is clicked."""
        if col not in (_COL_PARAMS, _COL_SIM_ID):
            return
        self._edit_row(row)

    def _edit_selected_row(self):
        """Edit the first selected row in the multi-line JSON editor."""
        sel = self._table.selectedIndexes()
        if not sel:
            QtWidgets.QMessageBox.information(self, "Edit", "Select a row to edit first.")
            return
        row = sel[0].row()
        self._edit_row(row)

    def _edit_row(self, row: int):
        """Open the multi-line JSON editor for a given row."""
        params_item = self._table.item(row, _COL_PARAMS)
        if params_item is None:
            return
        raw = params_item.text()
        sid_item = self._table.item(row, _COL_SIM_ID)
        sid = sid_item.text().strip() if sid_item else f"sim_{row + 1}"
        dlg = JsonEditorDialog(raw, title=f"Edit Parameters — {sid}", parent=self)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        new_text = dlg.get_json()
        if not new_text:
            return
        # Validate that the JSON parses before accepting
        try:
            json.loads(new_text)
        except json.JSONDecodeError as exc:
            QtWidgets.QMessageBox.warning(
                self, "Invalid JSON",
                f"Cannot save — the JSON is invalid:\n{exc}",
            )
            return
        params_item.setText(new_text)

    # ── Snapshot Current Setup ────────────────────────────────────────

    def _snapshot_current_setup(self):
        """Read the parent dialog's current widget values and add a new row.

        Reuses the same ``build_replay_payload`` builder as the main dialog's
        Export Config to JSON action, so the snapshot format is identical.
        """
        import datetime
        parent = self.parent()
        if parent is None:
            QtWidgets.QMessageBox.warning(
                self, "Snapshot", "No parent dialog to snapshot from."
            )
            return

        collect_fn = getattr(parent, "collect_widget_state_for_save", None)
        if collect_fn is None:
            QtWidgets.QMessageBox.warning(
                self, "Snapshot",
                "Parent dialog does not implement collect_widget_state_for_save.",
            )
            return
        try:
            widget_state = collect_fn()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self, "Snapshot Error", f"Failed to collect widget state:\n{exc}"
            )
            return
        if not isinstance(widget_state, dict):
            return

        # Fallback: if no mesh name is selected but a GPKG is set, use the
        # latest mesh name from the GPKG so the snapshot isn't created with
        # an empty mesh name.
        if not self._mesh_name and self._mesh_gpkg and os.path.isfile(self._mesh_gpkg):
            try:
                with sqlite3.connect(self._mesh_gpkg) as conn:
                    row = conn.execute(
                        "SELECT mesh_name FROM swe2d_baked_mesh ORDER BY created_utc DESC LIMIT 1"
                    ).fetchone()
                    if row and row[0]:
                        self._mesh_name = str(row[0])
                        self._set_mesh(self._mesh_gpkg, self._mesh_name)
            except Exception:
                pass

        run_id = datetime.datetime.now().astimezone().strftime(
            "swe2d_%Y%m%dT%H%M%S%z"
        )
        build_fn = getattr(parent, "build_replay_payload", None)
        if build_fn is None:
            QtWidgets.QMessageBox.warning(
                self, "Snapshot",
                "Parent dialog does not implement build_replay_payload.",
            )
            return
        try:
            entry = build_fn(
                widget_state=widget_state,
                mesh_name=self._mesh_name,
                run_duration_s=0.0,
                mesh_gpkg_path=self._mesh_gpkg,
                run_id=run_id,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self, "Snapshot Error", f"Failed to build replay payload:\n{exc}"
            )
            return
        if not isinstance(entry, dict):
            return

        self._add_row_from_entry(entry)
        log = getattr(parent, "_log", None)
        if log:
            log("batch> snapshot added row from current setup")

    # ── Import From GPKG ─────────────────────────────────────────────

    def _import_from_gpkg(self):
        """Open a GPKG, read run logs, and add one row per run."""
        gpkg = self._gpkg_path()
        if not gpkg:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Select Results GeoPackage", "", "GeoPackage (*.gpkg *.gpkgx);;All Files (*)"
            )
            if not path:
                return
            gpkg = path
            self._set_mesh(gpkg, self._mesh_name)

        if not os.path.isfile(gpkg):
            QtWidgets.QMessageBox.warning(self, "Import", f"GeoPackage not found:\n{gpkg}")
            return

        runs = self._query_runs_from_gpkg(gpkg)
        if not runs:
            QtWidgets.QMessageBox.information(
                self, "Import", "No run logs found in the selected GeoPackage."
            )
            return

        count = 0
        for run_id, metadata in runs:
            params_str = str(metadata.get("params", "") or "{}")
            try:
                params_dict = json.loads(params_str)
            except (json.JSONDecodeError, TypeError):
                params_dict = {"params": {"n_mann": 0.035, "duration_s": 3600.0}}

            if not isinstance(params_dict, dict):
                params_dict = {}

            # Merge with mesh name from the run log or the GPKG
            if "mesh" not in params_dict or not params_dict["mesh"]:
                params_dict["mesh"] = metadata.get("mesh_name", "")

            params_dict["id"] = str(run_id)
            self._add_row_from_entry(params_dict)
            count += 1

        QtWidgets.QMessageBox.information(
            self, "Import Complete",
            f"Imported {count} run{'s' if count != 1 else ''} from GPKG.\n"
            "Review and edit each row before running.",
        )

    def _query_runs_from_gpkg(self, gpkg_path: str) -> List:
        """Query run metadata from a results GPKG.

        Prefers ``swe2d_run_replays`` table when available (replay JSON format),
        falls back to legacy ``swe2d_run_logs`` / ``swe2d_baked_results``.
        Returns a list of ``(run_id, metadata_dict)`` tuples.
        """
        runs = []
        try:
            conn = sqlite3.connect(gpkg_path)
            cur = conn.cursor()
            for table in ("swe2d_run_replays", "swe2d_run_logs", "swe2d_baked_results"):
                cur.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (table,),
                )
                if cur.fetchone():
                    break
            else:
                return runs

            if table == "swe2d_run_replays":
                cur.execute(
                    "SELECT run_id, mesh_name, created_utc, replay_json FROM swe2d_run_replays "
                    "ORDER BY created_utc DESC"
                )
                for row in cur.fetchall():
                    run_id = str(row[0])
                    mesh_name = str(row[1] or "")
                    created = str(row[2] or "")
                    replay_json = str(row[3] or "{}")
                    try:
                        payload = json.loads(replay_json)
                    except (json.JSONDecodeError, TypeError):
                        payload = {}
                    payload["id"] = run_id
                    runs.append((run_id, {"created_utc": created, "params": json.dumps(payload), "mesh_name": mesh_name}))
            elif table == "swe2d_baked_results":
                cur.execute(
                    "SELECT run_id, created_utc, mesh_name FROM swe2d_baked_results "
                    "ORDER BY created_utc DESC"
                )
                for row in cur.fetchall():
                    run_id = str(row[0])
                    created = str(row[1] or "")
                    metadata = {
                        "created_utc": created,
                        "params": "{}",
                        "mesh_name": str(row[2] or ""),
                    }
                    runs.append((run_id, metadata))
            else:
                cur.execute(
                    f"SELECT run_id, created_utc, params FROM \"{table}\" ORDER BY created_utc DESC"
                )
                for row in cur.fetchall():
                    run_id = str(row[0])
                    created = str(row[1] or "")
                    params_raw = str(row[2] or "{}")
                    metadata = {
                        "created_utc": created,
                        "params": params_raw,
                    }
                    runs.append((run_id, metadata))
            conn.close()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self, "GPKG Error",
                f"Error reading GeoPackage:\n{exc}",
            )
        return runs

    # ── JSON Export / Import ──────────────────────────────────────────

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

    def _apply_mesh_to_selected(self):
        """Set the selected mesh name on all currently selected table rows."""
        mesh = self._mesh_name
        if not mesh:
            QtWidgets.QMessageBox.information(
                self, "Apply Mesh", "Select a mesh first."
            )
            return
        rows = set(i.row() for i in self._table.selectedIndexes())
        if not rows:
            QtWidgets.QMessageBox.information(self, "Apply Mesh", "Select rows in the table first.")
            return
        for r in rows:
            self._set_row_mesh(r, mesh)

    def _apply_mesh_to_all(self):
        """Set the selected mesh name on ALL rows."""
        mesh = self._mesh_name
        if not mesh:
            return
        for r in range(self._table.rowCount()):
            self._set_row_mesh(r, mesh)

    def _set_row_mesh(self, row: int, mesh_name: str):
        """Update the ``mesh`` key inside a row's JSON parameters."""
        item = self._table.item(row, _COL_PARAMS)
        if item is None:
            return
        try:
            params = json.loads(item.text())
        except (json.JSONDecodeError, TypeError):
            params = dict(self._base_params)
        if not isinstance(params, dict):
            params = {}
        existing_mesh = params.get("mesh")
        if isinstance(existing_mesh, dict):
            existing_mesh["mesh_name"] = str(mesh_name)
        else:
            params["mesh"] = str(mesh_name)
        item.setText(json.dumps(params, indent=2))

    def _gpkg_path(self) -> str:
        return str(self._mesh_gpkg or "").strip()

    # ── Batch Execution ───────────────────────────────────────────────

    def _run_batch(self):
        if self._batch_manager is None:
            QtWidgets.QMessageBox.warning(
                self, "Batch Run", "Batch manager not available."
            )
            return
        if self._batch_manager.is_running():
            return
        param_sets = self._collect_param_sets()
        if not param_sets:
            QtWidgets.QMessageBox.information(self, "Batch Run", "No parameter sets defined.")
            return
        gpkg = self._gpkg_path()
        if not gpkg or not os.path.isfile(gpkg):
            QtWidgets.QMessageBox.warning(self, "Batch Run", "GeoPackage not found. Select a valid file.")
            return

        config = BatchConfig(
            max_workers=self._max_workers_spin.value(),
            results_dir=tempfile.mkdtemp(prefix="hydra_batch_"),
            mesh_path=gpkg,
        )

        self._run_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)

        self._batch_manager.start_batch(param_sets, config)

    def _cancel_batch(self):
        if self._batch_manager is not None:
            self._batch_manager.cancel_batch()

    def _open_log_panel(self):
        """Open a log dock widget for the selected row."""
        rows = set(i.row() for i in self._table.selectedIndexes())
        if not rows:
            return
        row = min(rows)
        sim_id_item = self._table.item(row, _COL_SIM_ID)
        if not sim_id_item:
            return
        sim_id = sim_id_item.text()

        bm = self._batch_manager
        if not bm:
            return
        status = bm.get_status()
        state = status.get(sim_id)
        if not state or not state.log_file:
            return

        dock = LogDockWidget(sim_id, state.log_file, parent=self)
        if state.status in ("completed", "failed"):
            dock.load_full_log()
        else:
            dock.start_auto_refresh()
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, dock)
        dock.show()


class LogDockWidget(QtWidgets.QDockWidget):
    """Dock widget that displays tail of a per-sim log file."""

    def __init__(self, sim_id: str, log_path: str, parent=None):
        super().__init__(f"Log: {sim_id}", parent)
        self._sim_id = sim_id
        self._log_path = log_path
        self._log_view = QtWidgets.QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setMaximumBlockCount(500)
        font = QtWidgets.QFont("Monospace")
        font.setStyleHint(QtWidgets.QFont.TypeWriter)
        self._log_view.setFont(font)
        self.setWidget(self._log_view)

        self._refresh_timer = QtCore.QTimer(self)
        self._refresh_timer.timeout.connect(self._refresh_log)
        self._last_pos = 0

    def start_auto_refresh(self):
        """Start auto-refreshing the log tail every 1s."""
        self._last_pos = 0
        self._refresh_log()
        self._refresh_timer.start(1000)

    def stop_auto_refresh(self):
        self._refresh_timer.stop()

    def _refresh_log(self):
        """Read new lines from the log file and append."""
        try:
            if not os.path.isfile(self._log_path):
                return
            with open(self._log_path, "r") as f:
                f.seek(self._last_pos)
                new_lines = f.readlines()
                self._last_pos = f.tell()
            for line in new_lines:
                self._log_view.appendPlainText(line.rstrip("\n"))
        except Exception:
            pass

    def load_full_log(self):
        """Load the entire log file (for completed/failed sims)."""
        try:
            if not os.path.isfile(self._log_path):
                self._log_view.setPlainText("(no log file found)")
                return
            with open(self._log_path, "r") as f:
                self._log_view.setPlainText(f.read())
        except Exception as e:
            self._log_view.setPlainText(f"(error reading log: {e})")
