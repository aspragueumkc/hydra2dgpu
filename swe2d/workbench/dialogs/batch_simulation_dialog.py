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
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

from qgis.PyQt import QtCore, QtWidgets


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
    "k_mann_spin": "k_mann",
}

_BOOL_WIDGET_TO_CLI_MAP = {
    "adaptive_cfl_dt_chk": "adaptive_cfl_dt",
    "extreme_rain_mode_chk": "extreme_rain_mode",
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

    # Line output interval (same parsing as mesh output)
    raw_line = str(widget_params.get("line_output_interval_edit", "") or "").strip()
    if raw_line:
        if ":" in raw_line:
            parts = raw_line.split(":")
            try:
                line_hrs = float(parts[0]) + float(parts[1]) / 60.0
            except (ValueError, IndexError):
                line_hrs = 0.5
        else:
            try:
                line_hrs = float(raw_line)
            except ValueError:
                line_hrs = 0.5
        rp["line_output_interval_s"] = line_hrs * 3600.0

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
                 mesh_gpkg: str = "", results_gpkg: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Batch Simulation")
        self.resize(900, 500)

        self._base_params = dict(base_params or {})
        self._mesh_gpkg = str(mesh_gpkg)
        self._results_gpkg = str(results_gpkg)
        self._param_sets: List[Dict[str, Any]] = []
        self._processes: List[Optional[subprocess.Popen]] = []
        self._status_files: List[str] = []
        self._next_idx = 0
        self._active = 0
        self._completed = 0
        self._failed = 0
        self._running = False

        self._build_ui()
        # Populate the mesh combo from the auto-filled GPKG path
        self._refresh_mesh_list()
        self._add_row()

    # ── UI Construction ───────────────────────────────────────────────

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # GPKG selection
        gpkg_layout = QtWidgets.QHBoxLayout()
        gpkg_layout.addWidget(QtWidgets.QLabel("GeoPackage:"))
        self._gpkg_path_edit = QtWidgets.QLineEdit(self._mesh_gpkg)
        self._gpkg_path_edit.setPlaceholderText("Select or enter path to GeoPackage...")
        gpkg_layout.addWidget(self._gpkg_path_edit)
        self._gpkg_browse_btn = QtWidgets.QPushButton("Browse...")
        self._gpkg_browse_btn.setToolTip("Browse for a GeoPackage file.")
        self._gpkg_browse_btn.clicked.connect(self._browse_gpkg)
        gpkg_layout.addWidget(self._gpkg_browse_btn)
        self._gpkg_clear_btn = QtWidgets.QPushButton("Clear")
        self._gpkg_clear_btn.setToolTip("Clear the GPKG path (no auto-population)")
        self._gpkg_clear_btn.clicked.connect(lambda: self._gpkg_path_edit.clear())
        gpkg_layout.addWidget(self._gpkg_clear_btn)
        layout.addLayout(gpkg_layout)

        # Mesh selector (populated from GPKG when path changes)
        mesh_layout = QtWidgets.QHBoxLayout()
        mesh_layout.addWidget(QtWidgets.QLabel("Mesh:"))
        self._mesh_combo = QtWidgets.QComboBox()
        self._mesh_combo.setToolTip(
            "Select a mesh from the GPKG above and click 'Apply to Selected' "
            "to set the mesh on all selected rows, or edit per-row in the JSON editor."
        )
        self._mesh_combo.setEnabled(False)
        mesh_layout.addWidget(self._mesh_combo, 1)
        self._apply_mesh_btn = QtWidgets.QPushButton("Apply to Selected")
        self._apply_mesh_btn.setToolTip(
            "Set the selected mesh name on all currently selected table rows"
        )
        self._apply_mesh_btn.setEnabled(False)
        self._apply_mesh_btn.clicked.connect(self._apply_mesh_to_selected)
        mesh_layout.addWidget(self._apply_mesh_btn)
        self._refresh_mesh_btn = QtWidgets.QPushButton("Refresh")
        self._refresh_mesh_btn.setToolTip("Re-read mesh list from the GPKG")
        self._refresh_mesh_btn.setEnabled(False)
        self._refresh_mesh_btn.clicked.connect(lambda: self._refresh_mesh_list())
        mesh_layout.addWidget(self._refresh_mesh_btn)
        layout.addLayout(mesh_layout)

        # Wire GPKG path changes to refresh the mesh list
        self._gpkg_path_edit.textChanged.connect(self._on_gpkg_path_changed)

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
        toolbar.addWidget(self._add_row_btn)
        toolbar.addWidget(self._remove_row_btn)
        toolbar.addWidget(self._clear_btn)
        toolbar.addSpacing(20)
        toolbar.addWidget(self._edit_sel_btn)
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
        self._apply_mesh_all_btn = QtWidgets.QPushButton("Apply Mesh to All")
        self._apply_mesh_all_btn.setToolTip("Set the selected mesh name on ALL rows")
        self._apply_mesh_all_btn.setEnabled(False)
        self._apply_mesh_all_btn.clicked.connect(self._apply_mesh_to_all)
        data_toolbar.addWidget(self._snapshot_btn)
        data_toolbar.addWidget(self._from_gpkg_btn)
        data_toolbar.addWidget(self._apply_mesh_all_btn)
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
        self._status_btn = QtWidgets.QPushButton("Check Batch Status")
        self._status_btn.setToolTip("Check and log the status of all batch simulations.")
        self._status_btn.setEnabled(True)
        self._status_btn.clicked.connect(self._check_batch_status)
        controls.addWidget(self._status_btn)
        layout.addLayout(controls)

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
        """Read the parent dialog's current widget values and add a new row."""
        parent = self.parent()
        if parent is None:
            QtWidgets.QMessageBox.warning(self, "Snapshot", "No parent dialog to snapshot from.")
            return

        # Get the current widget parameters
        collect_fn = getattr(parent, "collect_run_widget_params", None)
        if collect_fn is None:
            QtWidgets.QMessageBox.warning(self, "Snapshot", "Parent dialog has no collect_run_widget_params.")
            return
        try:
            widget_params = collect_fn()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self, "Snapshot Error",
                f"Failed to collect widget params:\n{exc}",
            )
            return

        if not isinstance(widget_params, dict):
            return

        run_params = _widget_params_to_run_params(widget_params)

        # Mesh name from the mesh combo (populated from the GPKG)
        mesh_name = str(self._mesh_combo.currentData() or "")
        if not mesh_name:
            # Fall back to reading the GPKG directly
            gpkg = self._gpkg_path()
            if gpkg and os.path.isfile(gpkg):
                try:
                    conn = sqlite3.connect(gpkg)
                    cur = conn.cursor()
                    cur.execute(
                        "SELECT mesh_name FROM swe2d_baked_mesh ORDER BY created_utc DESC LIMIT 1"
                    )
                    row = cur.fetchone()
                    if row:
                        mesh_name = str(row[0])
                    conn.close()
                except Exception as _e:

                    logger.warning(f"[ERROR] Exception in batch_simulation_dialog.py: {_e}")

        # ── Helper: resolve GPKG table name + GPKG path from QGIS layer ──
        def _get_layer_info(combo):
            """Return (table_name, gpkg_path) from combo's stored layer ID.

            gpkg_path is '' if the layer doesn't come from a GeoPackage.
            """
            if combo is None:
                return ("", "")
            lid = combo.currentData()
            if not lid:
                return ("", "")
            from qgis.core import QgsProject
            layer = QgsProject.instance().mapLayer(lid)
            if layer is None:
                return ("", "")
            src = str(layer.source())
            # GPKG source: "/path/to/file.gpkg|layername=table_name"
            if "|layername=" in src:
                gpkg_path, _, table = src.partition("|layername=")
                return (table.strip(), gpkg_path.strip())
            return (str(layer.name()).strip(), "")

        mesh_gpkg = self._gpkg_path()

        def _dict_with_gpkg(table: str, gpkg_path: str, **extra) -> dict:
            """Build a data-source dict with optional gpkg key."""
            d = {"table": table, **extra}
            if gpkg_path and gpkg_path != mesh_gpkg:
                d["gpkg"] = gpkg_path
            return d

        # ── Capture top-level keys the headless runner needs ──────────
        mtab = getattr(parent, "_model_tab_view", None)

        bc_lines = None
        hyetograph_cfg = None
        rain_cn_cfg = None
        infiltration_method = ""
        drainage_cfg = None
        structures_cfg = None
        sample_lines_cfg = None

        if mtab is not None:
            bc_tbl, bc_gpkg = _get_layer_info(
                getattr(mtab, "bc_lines_layer_combo", None))
            if bc_tbl:
                bc_lines = _dict_with_gpkg(bc_tbl, bc_gpkg)

            hg_tbl, hg_gpkg = _get_layer_info(
                getattr(mtab, "hyetograph_layer_combo", None))
            rg_tbl, rg_gpkg = _get_layer_info(
                getattr(mtab, "rain_gage_layer_combo", None))
            if hg_tbl and rg_tbl:
                src_gpkg = hg_gpkg or rg_gpkg
                hyetograph_cfg = _dict_with_gpkg(
                    hg_tbl, src_gpkg,
                    gauge_layer=rg_tbl,
                )

            cn_tbl, cn_gpkg = _get_layer_info(
                getattr(mtab, "cn_layer_combo", None))
            if cn_tbl:
                rain_cn_cfg = _dict_with_gpkg(cn_tbl, cn_gpkg, cn_field="cn")

            dn_tbl, dn_gpkg = _get_layer_info(
                getattr(mtab, "drain_nodes_layer_combo", None))
            dl_tbl, dl_gpkg = _get_layer_info(
                getattr(mtab, "drain_links_layer_combo", None))
            if dn_tbl and dl_tbl:
                drainage_cfg = {"nodes_layer": dn_tbl, "links_layer": dl_tbl}
                _drain_gpkg = dn_gpkg or mesh_gpkg
                if _drain_gpkg and _drain_gpkg != mesh_gpkg:
                    drainage_cfg["gpkg"] = _drain_gpkg

            struct_cfg = getattr(parent, "_build_hydraulic_structure_config", lambda: None)()
            if struct_cfg is not None:
                structures_cfg = struct_cfg.to_dict()

            sl_tbl, sl_gpkg = _get_layer_info(
                getattr(mtab, "sample_lines_layer_combo", None))
            if sl_tbl:
                sample_lines_cfg = _dict_with_gpkg(sl_tbl, sl_gpkg)

        if mtab is not None:
            combo = getattr(mtab, "infiltration_method_combo", None)
            im = str(combo.currentData() or "none") if combo else "none"
            if im and im != "none":
                infiltration_method = str(im)

        entry = {
            "id": "current_setup",
            "mesh": mesh_name,
            "mesh_gpkg": mesh_gpkg,
            "params": run_params,
        }
        if bc_lines:
            entry["bc_lines"] = bc_lines
        if hyetograph_cfg is not None:
            entry["hyetograph"] = hyetograph_cfg
        if rain_cn_cfg is not None:
            entry["rain_cn"] = rain_cn_cfg
        if infiltration_method:
            entry["infiltration_method"] = infiltration_method
        if drainage_cfg is not None:
            entry["drainage"] = drainage_cfg
        if structures_cfg is not None:
            entry["structures"] = structures_cfg
        if sample_lines_cfg is not None:
            entry["sample_lines"] = sample_lines_cfg

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
            self._gpkg_path_edit.setText(gpkg)

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

        Returns a list of ``(run_id, metadata_dict)`` tuples.
        """
        runs = []
        try:
            conn = sqlite3.connect(gpkg_path)
            cur = conn.cursor()
            for table in ("swe2d_run_logs", "swe2d_baked_results"):
                cur.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (table,),
                )
                if cur.fetchone():
                    break
            else:
                return runs
            if table == "swe2d_baked_results":
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

    def _on_gpkg_path_changed(self):
        """Refresh the mesh combo when the GPKG path text changes."""
        self._refresh_mesh_list()

    def _refresh_mesh_list(self):
        """Query the GPKG for available mesh names and populate the combo."""
        gpkg = self._gpkg_path()
        if not gpkg or not os.path.isfile(gpkg):
            self._mesh_combo.clear()
            self._mesh_combo.setEnabled(False)
            self._apply_mesh_btn.setEnabled(False)
            self._apply_mesh_all_btn.setEnabled(False)
            self._refresh_mesh_btn.setEnabled(False)
            return
        try:
            conn = sqlite3.connect(gpkg)
            cur = conn.cursor()
            cur.execute(
                "SELECT DISTINCT mesh_name FROM swe2d_baked_mesh "
                "WHERE mesh_name IS NOT NULL AND mesh_name != '' "
                "ORDER BY created_utc DESC"
            )
            rows = cur.fetchall()
            conn.close()
            self._mesh_combo.clear()
            if not rows:
                self._mesh_combo.setEnabled(False)
                self._apply_mesh_btn.setEnabled(False)
                self._apply_mesh_all_btn.setEnabled(False)
                self._refresh_mesh_btn.setEnabled(True)
                return
            for (name,) in rows:
                self._mesh_combo.addItem(str(name), str(name))
            self._mesh_combo.setEnabled(True)
            self._apply_mesh_btn.setEnabled(True)
            self._apply_mesh_all_btn.setEnabled(True)
            self._refresh_mesh_btn.setEnabled(True)
        except Exception:
            self._mesh_combo.clear()
            self._mesh_combo.setEnabled(False)
            self._apply_mesh_btn.setEnabled(False)
            self._apply_mesh_all_btn.setEnabled(False)
            self._refresh_mesh_btn.setEnabled(True)

    def _apply_mesh_to_selected(self):
        """Set the selected mesh name on all currently selected table rows."""
        mesh = self._mesh_combo.currentData()
        if not mesh:
            return
        rows = set(i.row() for i in self._table.selectedIndexes())
        if not rows:
            QtWidgets.QMessageBox.information(self, "Apply Mesh", "Select rows in the table first.")
            return
        for r in rows:
            self._set_row_mesh(r, mesh)

    def _apply_mesh_to_all(self):
        """Set the selected mesh name on ALL rows."""
        mesh = self._mesh_combo.currentData()
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
        params["mesh"] = str(mesh_name)
        item.setText(json.dumps(params, indent=2))

    def _browse_gpkg(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select GeoPackage", "", "GeoPackage (*.gpkg *.gpkgx);;All Files (*)")
        if path:
            self._gpkg_path_edit.setText(path)

    def _gpkg_path(self) -> str:
        return str(self._gpkg_path_edit.text()).strip()

    # ── Batch Execution ───────────────────────────────────────────────

    def _run_batch(self):
        if self._running:
            return
        param_sets = self._collect_param_sets()
        if not param_sets:
            QtWidgets.QMessageBox.information(self, "Batch Run", "No parameter sets defined.")
            return
        gpkg = self._gpkg_path()
        if not gpkg or not os.path.isfile(gpkg):
            QtWidgets.QMessageBox.warning(self, "Batch Run", "GeoPackage not found. Select a valid file.")
            return

        self._running = True
        self._run_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._status_btn.setEnabled(True)
        self._param_sets = param_sets
        self._processes = [None] * len(param_sets)
        self._status_files = [""] * len(param_sets)
        self._next_idx = 0
        self._active = 0
        self._completed = 0
        self._failed = 0

        status_dir = tempfile.mkdtemp(prefix="hydra_batch_status_")
        self._start_next_batch(status_dir)
        self._poll_tick()

    def _poll_tick(self):
        """Repeating poller — calls _tick_run and reschedules itself."""
        self._tick_run()
        if self._running:
            QtCore.QTimer.singleShot(500, self._poll_tick)

    def _start_next_batch(self, status_dir: str = ""):
        max_workers = self._max_workers_spin.value()
        while self._active < max_workers and self._next_idx < len(self._param_sets):
            idx = self._next_idx
            self._next_idx += 1
            ps = self._param_sets[idx]
            params_json = json.dumps(ps)
            gpkg = self._gpkg_path()
            results = self._results_gpkg or os.path.splitext(gpkg)[0] + "_batch_results.gpkg"
            cmd = [
                sys.executable, "-m", "swe2d.cli", "run",
                gpkg, params_json,
                "--results", results,
            ]
            status_file = ""
            if status_dir:
                status_file = os.path.join(status_dir, f"sim_{idx}.json")
                cmd += ["--status-file-path", status_file, "--status-interval", "2.0"]
            self._status_files[idx] = status_file
            # Don't capture stderr — let it flow to the QGIS terminal so
            # the user sees the full traceback live.  stdout is still
            # captured (used for progress-table display on failure).
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=None, text=True,
            )
            self._processes[idx] = proc
            self._active += 1
            item = self._table.item(idx, _COL_STATUS)
            if item:
                item.setText("running")

    def _check_batch_status(self):
        parent_log = getattr(self.parent(), "_log", None)
        if not parent_log:
            return
        total = len(self._param_sets)
        running_count = 0
        for i, sf in enumerate(self._status_files):
            if not sf or not os.path.exists(sf):
                continue
            try:
                with open(sf) as f:
                    status = json.load(f)
            except Exception:
                continue
            s = status.get("status", "")
            sid = str(self._param_sets[i].get("id", f"sim_{i}"))
            if s == "running":
                running_count += 1
                t = status.get("t", 0.0)
                step = status.get("step", 0)
                wet = status.get("wet_cells", -1)
                parent_log(f"batch> {sid} step={step} t={t:.1f}s wet={wet}")
            elif s == "done":
                parent_log(f"batch> {sid} done")
            elif s == "error":
                err = status.get("error", "unknown")
                parent_log(f"batch> {sid} error: {err}")
        if running_count == 0 and not self._running:
            parent_log(f"batch> all simulations complete ({self._completed}/{total})")
        elif running_count > 0:
            parent_log(f"batch> {running_count}/{total} simulations still running")
        else:
            parent_log(f"batch> no running simulations ({self._completed + self._failed}/{total})")

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
                stdout = proc.stdout.read() if proc.stdout else ""
                if progress_item:
                    progress_item.setText(stderr.strip()[:100])
                # Log full stderr so the user can diagnose the failure
                parent_log = getattr(self.parent(), "_log", None)
                if parent_log:
                    sid = str(self._param_sets[i].get("id", f"sim_{i}"))
                    for line in stderr.strip().split("\n"):
                        parent_log(f"batch> [{sid} ERROR] {line}")
                    for line in stdout.strip().split("\n"):
                        if line.strip():
                            parent_log(f"batch> [{sid} stdout] {line}")
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
