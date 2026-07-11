#!/usr/bin/env python3
"""GeoPackage table explorer dialog for table-aware viewers and management."""

from __future__ import annotations

import logging
import os
import sqlite3
from typing import Callable

from qgis.PyQt import QtCore, QtWidgets

from swe2d.workbench.dialogs.sqlite_preview_dialog import SWE2DSQLiteTablePreviewDialog
from swe2d.workbench.dialogs.simulation_config_viewer_dialog import SWE2DSimulationConfigViewerDialog
from swe2d.workbench.services.gpkg_operations_service import (
    drop_table,
    get_table_row_count,
    list_run_ids_from_tables as _list_run_ids_from_table_names,
    list_tables,
    rename_table,
    delete_run,
    delete_run_partial,
)

logger_wb = logging.getLogger(__name__)


class SWE2DModelGeoPackageExplorerDialog(QtWidgets.QDialog):
    """GeoPackage table explorer for opening table-aware viewers and table management."""

    def __init__(
        self,
        gpkg_path: str,
        open_run_log_viewer: Callable[[], None],
        open_line_results_viewer: Callable[[], None],
        logger: Callable[[str], None],
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Model GeoPackage Explorer")
        self.resize(980, 660)
        self._gpkg_path = str(gpkg_path or "")
        self._open_run_log_viewer = open_run_log_viewer
        self._open_line_results_viewer = open_line_results_viewer
        self._log = logger if callable(logger) else (lambda _msg: None)

        root = QtWidgets.QVBoxLayout(self)
        self.source_lbl = QtWidgets.QLabel(f"GeoPackage: {self._gpkg_path}")
        self.source_lbl.setWordWrap(True)
        root.addWidget(self.source_lbl)

        self.table = QtWidgets.QTableWidget()
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Table", "Rows", "Type", "Actions"])
        self.table.horizontalHeader().setStretchLastSection(True)
        root.addWidget(self.table, stretch=1)

        row = QtWidgets.QHBoxLayout()
        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        self.refresh_btn.setToolTip("Reload the table listing from the GeoPackage.")
        self.open_btn = QtWidgets.QPushButton("Open Viewer")
        self.open_btn.setToolTip("Open the appropriate viewer for the selected table.")
        self.preview_btn = QtWidgets.QPushButton("Preview Table")
        self.preview_btn.setToolTip("Preview the selected table contents.")
        self.rename_btn = QtWidgets.QPushButton("Rename Table")
        self.rename_btn.setToolTip("Rename the selected model table (swe2d_* tables only).")
        self.delete_btn = QtWidgets.QPushButton("Delete Table")
        self.delete_btn.setToolTip("Permanently delete the selected table from the GeoPackage.")
        self.delete_run_btn = QtWidgets.QPushButton("Delete by Run ID")
        self.delete_run_btn.setToolTip("Delete all result tables associated with a specific run ID.")
        for btn in (self.refresh_btn, self.open_btn, self.preview_btn, self.rename_btn, self.delete_btn, self.delete_run_btn):
            row.addWidget(btn)
        row.addStretch(1)
        root.addLayout(row)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        root.addWidget(buttons)

        self.refresh_btn.clicked.connect(self.refresh_tables)
        self.open_btn.clicked.connect(self.open_selected)
        self.preview_btn.clicked.connect(self.preview_selected)
        self.rename_btn.clicked.connect(self.rename_selected)
        self.delete_btn.clicked.connect(self.delete_selected)
        self.delete_run_btn.clicked.connect(self._delete_by_run_id)
        self.table.itemSelectionChanged.connect(self._sync_button_state)
        self.table.itemDoubleClicked.connect(lambda _item: self.open_selected())

        self.refresh_tables()

    def _selected_table(self) -> str:
        """Return the name of the currently selected table, or empty string."""
        row = self.table.currentRow()
        if row < 0:
            return ""
        item = self.table.item(row, 0)
        return "" if item is None else str(item.text() or "").strip()

    def _table_kind(self, name: str) -> str:
        """Classify a table name into a result kind (run_log, line_results, etc.)."""
        t = str(name or "").strip().lower()
        if t == "swe2d_run_logs" or t.endswith("_swe2d_run_logs"):
            return "run_log"
        if t == "swe2d_simulation_configs":
            return "config"
        if t.startswith("swe2d_baked_line"):
            return "line_results"
        if t == "swe2d_baked_coupling":
            return "coupling_results"
        if t in ("swe2d_baked_results", "swe2d_baked_mesh"):
            return "mesh_results"
        if t.startswith("gpkg_") or t.startswith("sqlite_") or t.startswith("rtree_"):
            return "system"
        return "table"

    def _is_mutable_model_table(self, name: str) -> bool:
        """Check if the table name starts with swe2d_ (i.e. is user-mutable)."""
        t = str(name or "").strip().lower()
        return t.startswith("swe2d_")

    def _sync_button_state(self):
        """Enable/disable action buttons based on current table selection."""
        name = self._selected_table()
        has_sel = bool(name)
        self.open_btn.setEnabled(has_sel)
        self.preview_btn.setEnabled(has_sel)
        mutable = has_sel and self._is_mutable_model_table(name)
        self.rename_btn.setEnabled(mutable)
        self.delete_btn.setEnabled(mutable)

    def refresh_tables(self):
        """Reload the table listing from the GeoPackage."""
        self.table.setRowCount(0)
        if not self._gpkg_path or not os.path.exists(self._gpkg_path):
            self._sync_button_state()
            return
        names = list_tables(self._gpkg_path)
        for name in names:
            row_idx = self.table.rowCount()
            self.table.insertRow(row_idx)
            self.table.setItem(row_idx, 0, QtWidgets.QTableWidgetItem(name))
            n_rows = get_table_row_count(self._gpkg_path, name)
            self.table.setItem(row_idx, 1, QtWidgets.QTableWidgetItem("?" if n_rows < 0 else str(n_rows)))
            self.table.setItem(row_idx, 2, QtWidgets.QTableWidgetItem(self._table_kind(name)))
            actions = "open+preview"
            if self._is_mutable_model_table(name):
                actions += "+rename+delete"
            self.table.setItem(row_idx, 3, QtWidgets.QTableWidgetItem(actions))
        self.table.resizeColumnsToContents()
        self._sync_button_state()

    def _open_preview(self, name: str, title: str):
        """Open the SQLite table preview dialog for the given table."""
        dlg = SWE2DSQLiteTablePreviewDialog(self._gpkg_path, name, title=title, parent=self)
        dlg.exec()

    def open_selected(self):
        """Open the appropriate viewer for the selected table."""
        name = self._selected_table()
        if not name:
            return
        kind = self._table_kind(name)
        if kind == "run_log":
            self._open_run_log_viewer()
            return
        if kind == "line_results":
            self._open_line_results_viewer()
            return
        if kind == "config":
            dlg = SWE2DSimulationConfigViewerDialog(self._gpkg_path, parent=self)
            dlg.exec()
            return
        if kind == "mesh_results":
            self._open_preview(name, title=f"Mesh Results Viewer - {name}")
            return
        self._open_preview(name, title=f"Table Viewer - {name}")

    def preview_selected(self):
        """Open the table preview for the selected table."""
        name = self._selected_table()
        if not name:
            return
        self._open_preview(name, title=f"Table Viewer - {name}")

    def rename_selected(self):
        """Prompt for a new name and rename the selected table."""
        old_name = self._selected_table()
        if not old_name:
            return
        if not self._is_mutable_model_table(old_name):
            QtWidgets.QMessageBox.warning(self, "Rename Table", "Only model tables (swe2d_*) can be renamed from this explorer.")
            return
        new_name, ok = QtWidgets.QInputDialog.getText(self, "Rename Table", "New table name:", text=old_name)
        if not ok:
            return
        new_name = str(new_name or "").strip()
        if not new_name or new_name == old_name:
            return
        try:
            rename_table(self._gpkg_path, old_name, new_name)
            self._log(f"GeoPackage explorer renamed table: {old_name} -> {new_name}")
            self.refresh_tables()
        except RuntimeError as exc:
            QtWidgets.QMessageBox.warning(self, "Rename Table", str(exc))

    def delete_selected(self):
        """Prompt for confirmation and delete the selected table."""
        name = self._selected_table()
        if not name:
            return
        if not self._is_mutable_model_table(name):
            QtWidgets.QMessageBox.warning(self, "Delete Table", "Only model tables (swe2d_*) can be deleted from this explorer.")
            return
        ans = QtWidgets.QMessageBox.question(
            self,
            "Delete Table",
            f"Delete table '{name}' from GeoPackage?\n\nThis cannot be undone.",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if ans != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        try:
            drop_table(self._gpkg_path, name)
            self._log(f"GeoPackage explorer deleted table: {name}")
            self.refresh_tables()
        except RuntimeError as exc:
            QtWidgets.QMessageBox.warning(self, "Delete Table", f"Failed to delete table:\n{exc}")

    def _delete_by_run_id(self):
        """Delete result data for selected run IDs with table-type selection."""
        if not self._gpkg_path or not os.path.exists(self._gpkg_path):
            QtWidgets.QMessageBox.warning(self, "Delete by Run ID", "No GeoPackage path set.")
            return

        try:
            run_ids = self._collect_run_ids()
            if not run_ids:
                return
            table_kinds = self._select_tables(run_ids)
            if not table_kinds:
                return
            deleted = delete_run_partial(self._gpkg_path, run_ids, **table_kinds)
            self._log(f"GeoPackage explorer deleted {len(deleted)} table(s) for {len(run_ids)} run(s)")
            QtWidgets.QMessageBox.information(
                self, "Delete Complete",
                f"Deleted {len(deleted)} table(s) for {len(run_ids)} run(s)."
            )
            self.refresh_tables()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Delete by Run ID", f"Failed to delete by run ID:\n{exc}")

    def _collect_run_ids(self) -> list[str]:
        """Show multi-select dialog for run IDs. Returns selected run_ids."""
        conn = sqlite3.connect(self._gpkg_path)
        try:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='swe2d_run_logs'")
            has_run_logs = cur.fetchone() is not None

            run_ids: list[str] = []
            if has_run_logs:
                cur.execute("SELECT DISTINCT run_id FROM swe2d_run_logs ORDER BY run_id")
                run_ids = [str(r[0]) for r in cur.fetchall() if r and r[0] is not None]
            else:
                run_ids = _list_run_ids_from_table_names(list_tables(self._gpkg_path))
        finally:
            conn.close()

        if not run_ids:
            QtWidgets.QMessageBox.information(
                self, "Delete by Run ID", "No run IDs found in the GeoPackage."
            )
            return []

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Delete by Run ID — Select Runs")
        dlg.resize(450, 380)
        layout = QtWidgets.QVBoxLayout(dlg)

        layout.addWidget(QtWidgets.QLabel("Select run IDs to delete:"))

        toggle_row = QtWidgets.QHBoxLayout()
        select_all_btn = QtWidgets.QPushButton("Select All")
        deselect_all_btn = QtWidgets.QPushButton("Deselect All")
        toggle_row.addWidget(select_all_btn)
        toggle_row.addWidget(deselect_all_btn)
        toggle_row.addStretch(1)
        layout.addLayout(toggle_row)

        list_widget = QtWidgets.QListWidget()
        list_widget.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        for rid in sorted(run_ids):
            item = QtWidgets.QListWidgetItem(rid)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            list_widget.addItem(item)
        layout.addWidget(list_widget, stretch=1)

        def _toggle_all(checked: bool):
            for i in range(list_widget.count()):
                list_widget.item(i).setCheckState(
                    Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
                )

        select_all_btn.clicked.connect(lambda: _toggle_all(True))
        deselect_all_btn.clicked.connect(lambda: _toggle_all(False))

        next_btn = QtWidgets.QPushButton("Next")
        next_btn.setEnabled(True)
        cancel_btn = QtWidgets.QPushButton("Cancel")

        def _on_selection_changed():
            has_checked = any(
                list_widget.item(i).checkState() == Qt.CheckState.Checked
                for i in range(list_widget.count())
            )
            next_btn.setEnabled(has_checked)

        list_widget.itemChanged.connect(_on_selection_changed)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(next_btn)
        layout.addLayout(btn_row)

        next_btn.clicked.connect(dlg.accept)
        cancel_btn.clicked.connect(dlg.reject)

        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return []

        selected = []
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                selected.append(str(item.text()))
        return selected

    def _select_tables(self, run_ids: list[str]) -> dict | None:
        """Show multi-select dialog for table types. Returns kwargs dict or None."""
        conn = sqlite3.connect(self._gpkg_path)
        try:
            cur = conn.cursor()

            def _table_exists(name: str) -> bool:
                cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,))
                return cur.fetchone() is not None

            # Count legacy per-run tables
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            all_tables = [str(r[0]) for r in cur.fetchall() if r and r[0] is not None]
            legacy_count = 0
            for t in all_tables:
                if t.startswith("gpkg_") or t.startswith("sqlite_") or t.startswith("rtree_"):
                    continue
                suffix = t.rsplit("_", 1)
                if len(suffix) == 2 and suffix[1] in run_ids:
                    legacy_count += 1
        finally:
            conn.close()

        # Build table options
        table_options: list[tuple[str, str, str]] = []  # (label, warning, kind_key)
        if _table_exists("swe2d_run_logs"):
            table_options.append(("swe2d_run_logs", "", "delete_run_logs"))
        if _table_exists("swe2d_baked_results"):
            table_options.append(("swe2d_baked_results", "", "delete_baked_results"))
        if _table_exists("swe2d_baked_line_ts"):
            table_options.append(("swe2d_baked_line_ts", "", "delete_baked_line_ts"))
        if _table_exists("swe2d_baked_line_profiles"):
            table_options.append(("swe2d_baked_line_profiles", "", "delete_baked_line_profiles"))
        if _table_exists("swe2d_baked_coupling"):
            table_options.append(("swe2d_baked_coupling", "", "delete_baked_coupling"))
        if _table_exists("swe2d_baked_mesh"):
            table_options.append((
                "swe2d_baked_mesh",
                "Shared across runs — may orphan other results",
                "delete_baked_mesh",
            ))
        if _table_exists("swe2d_simulation_configs"):
            table_options.append((
                "swe2d_simulation_configs",
                "Contains ALL configs, not just selected runs",
                "delete_simulation_configs",
            ))
        if legacy_count > 0:
            table_options.append((
                f"Legacy per-run tables ({legacy_count} table(s))",
                "Per-run tables matching selected run IDs will be dropped",
                "delete_legacy_tables",
            ))

        if not table_options:
            QtWidgets.QMessageBox.information(
                self, "Delete by Run ID", "No deletable tables found."
            )
            return None

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Delete by Run ID — Select Tables")
        dlg.resize(500, 400)
        layout = QtWidgets.QVBoxLayout(dlg)

        run_summary = ", ".join(run_ids[:5])
        if len(run_ids) > 5:
            run_summary += f" (+{len(run_ids) - 5} more)"
        layout.addWidget(QtWidgets.QLabel(f"Tables to delete for run(s): {run_summary}"))

        list_widget = QtWidgets.QListWidget()
        list_widget.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        for label, warning, _key in table_options:
            item = QtWidgets.QListWidgetItem(label)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            if warning:
                item.setToolTip(warning)
                item.setForeground(Qt.GlobalColor.darkYellow)
            list_widget.addItem(item)
        layout.addWidget(list_widget, stretch=1)

        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText("Delete")
        btn_box.accepted.connect(dlg.accept)
        btn_box.rejected.connect(dlg.reject)
        layout.addWidget(btn_box)

        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None

        result: dict = {}
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                _, _, key = table_options[i]
                result[key] = True
        return result if result else None
