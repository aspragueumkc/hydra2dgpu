"""Lightweight mesh picker dialog for selecting a mesh from a GeoPackage."""
from __future__ import annotations

import os
import sqlite3

from qgis.PyQt import QtWidgets


class MeshPickerDialog(QtWidgets.QDialog):
    """Dialog that lists mesh names from a GeoPackage and returns the selected one."""

    def __init__(self, gpkg_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Mesh")
        self.resize(400, 300)
        self._gpkg_path = str(gpkg_path or "")
        self._selected_mesh_name = ""

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel(f"GeoPackage: {self._gpkg_path}"))

        self._list = QtWidgets.QListWidget()
        self._list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        layout.addWidget(self._list, 1)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._load_mesh_names()

    def accept(self) -> None:
        """Override QDialog.accept: validate selection before closing."""
        item = self._list.currentItem()
        if item is None:
            QtWidgets.QMessageBox.warning(
                self, "Select Mesh", "Please select a mesh."
            )
            return
        self._selected_mesh_name = str(item.text())
        super().accept()

    def _load_mesh_names(self) -> None:
        """Populate the list with mesh names from swe2d_baked_mesh."""
        if not self._gpkg_path or not os.path.isfile(self._gpkg_path):
            QtWidgets.QMessageBox.warning(
                self, "Select Mesh", "GeoPackage file not found."
            )
            return
        try:
            with sqlite3.connect(self._gpkg_path) as conn:
                cur = conn.cursor()
                cur.execute(
                    "SELECT DISTINCT mesh_name FROM swe2d_baked_mesh "
                    "WHERE mesh_name IS NOT NULL AND mesh_name != '' "
                    "ORDER BY created_utc DESC"
                )
                for row in cur.fetchall():
                    self._list.addItem(str(row[0]))
                if self._list.count() == 0:
                    QtWidgets.QMessageBox.information(
                        self, "Select Mesh",
                        "No mesh names found in the selected GeoPackage.",
                    )
        except sqlite3.Error as exc:
            QtWidgets.QMessageBox.warning(
                self, "Select Mesh", f"Failed to read mesh list:\n{exc}"
            )

    def selected_mesh_name(self) -> str:
        """Return the selected mesh name (empty if dialog was cancelled)."""
        return self._selected_mesh_name

    def gpkg_path(self) -> str:
        """Return the GeoPackage path passed to the dialog."""
        return self._gpkg_path
