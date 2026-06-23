#!/usr/bin/env python3
"""GeoPackage table explorer dialog for table-aware viewers and management."""

from __future__ import annotations

import logging
import os
import sqlite3
from typing import Callable

from qgis.PyQt import QtWidgets

from swe2d.workbench.dialogs.sqlite_preview_dialog import SWE2DSQLiteTablePreviewDialog
from swe2d.workbench.services.gpkg_operations_service import (
    drop_table,
    get_table_row_count,
    list_tables,
    rename_table,
    delete_run,
)

logger_wb = logging.getLogger(__name__)


def _list_run_ids_from_table_names(
    tables: list[str],
) -> list[str]:
    """Extract unique run IDs from SWE2D result table names."""
    run_ids: list[str] = []
    seen: set[str] = set()
    for t in tables:
        parts = t.rsplit("_", 1)
        if len(parts) == 2 and parts[1] and parts[1] not in seen:
            run_ids.append(parts[1])
            seen.add(parts[1])
    return run_ids


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
        self.open_btn = QtWidgets.QPushButton("Open Viewer")
        self.preview_btn = QtWidgets.QPushButton("Preview Table")
        self.rename_btn = QtWidgets.QPushButton("Rename Table")
        self.delete_btn = QtWidgets.QPushButton("Delete Table")
        self.delete_run_btn = QtWidgets.QPushButton("Delete by Run ID")
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
        if t.startswith("swe2d_line_results") or "_swe2d_line_results" in t:
            return "line_results"
        if t.startswith("swe2d_coupling_results") or "_swe2d_coupling_results" in t:
            return "coupling_results"
        if t.startswith("swe2d_mesh_results") or t.endswith("_swe2d_mesh_results") or t in ("swe2d_face_flux_results", "swe2d_face_results", "swe2d_flux_faces"):
            return "mesh_results"
        if (
            t.startswith("swe2d_conservation")
            or t.startswith("swe2d_boundary_flux_forensics")
            or t.startswith("swe2d_source_budget_forensics")
            or "_swe2d_conservation" in t
            or "_swe2d_boundary_flux_forensics" in t
            or "_swe2d_source_budget_forensics" in t
        ):
            return "conservation"
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
        """Delete all result tables associated with a selected run ID."""
        if not self._gpkg_path or not os.path.exists(self._gpkg_path):
            QtWidgets.QMessageBox.warning(self, "Delete by Run ID", "No GeoPackage path set.")
            return

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

            if not run_ids:
                QtWidgets.QMessageBox.information(
                    self, "Delete by Run ID",
                    "No run IDs found in the GeoPackage."
                )
                return

            dlg = QtWidgets.QDialog(self)
            dlg.setWindowTitle("Delete Results by Run ID")
            dlg.resize(500, 320)
            layout = QtWidgets.QVBoxLayout(dlg)

            layout.addWidget(QtWidgets.QLabel("Select the run ID to delete:"))

            combo = QtWidgets.QComboBox()
            combo.addItems(sorted(run_ids))
            combo.setCurrentIndex(-1)
            layout.addWidget(combo)

            info_label = QtWidgets.QLabel("")
            info_label.setWordWrap(True)
            layout.addWidget(info_label)

            def _on_run_id_changed(idx):
                """Update the info label with tables matching the selected run ID."""
                if idx < 0:
                    info_label.setText("")
                    return
                rid = str(combo.itemText(idx))
                all_tables = list_tables(self._gpkg_path)
                matching = [t for t in all_tables if t.endswith("_" + rid)]
                if rid in matching:
                    matching.remove(rid)
                if matching:
                    info_label.setText("Tables to delete:\n  " + "\n  ".join(matching))
                else:
                    info_label.setText("No result tables found for this run ID (only run_log entry will be removed).")

            combo.currentIndexChanged.connect(_on_run_id_changed)

            btn_box = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
            )
            btn_box.accepted.connect(dlg.accept)
            btn_box.rejected.connect(dlg.reject)
            layout.addWidget(btn_box)

            if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                return

            rid = str(combo.currentText()).strip()
            if not rid:
                return

            ans = QtWidgets.QMessageBox.question(
                self,
                "Confirm Delete",
                f"Permanently delete all result tables for run ID '{rid}'?\n\nThis cannot be undone.",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if ans != QtWidgets.QMessageBox.StandardButton.Yes:
                return

            delete_run(self._gpkg_path, rid)
            self._log(f"GeoPackage explorer deleted run ID '{rid}'")
            QtWidgets.QMessageBox.information(
                self, "Delete Complete",
                f"Tables for run ID '{rid}' have been deleted."
            )
            self.refresh_tables()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Delete by Run ID", f"Failed to delete by run ID:\n{exc}")
        finally:
            conn.close()
