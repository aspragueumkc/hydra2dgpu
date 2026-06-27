"""Dialog for selecting which runs to load from a GeoPackage."""

from __future__ import annotations

from typing import List, Set

from qgis.PyQt import QtWidgets
from qgis.PyQt.QtCore import Qt


class RunSelectionDialog(QtWidgets.QDialog):
    """Show available runs from a GPKG and let the user pick which to load."""

    def __init__(self, run_records, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Runs to Load")
        self.setMinimumWidth(400)
        self.setMinimumHeight(300)

        layout = QtWidgets.QVBoxLayout(self)

        label = QtWidgets.QLabel("Select runs to load from the GeoPackage:")
        layout.addWidget(label)

        self._list = QtWidgets.QListWidget()
        self._list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self._records = run_records

        for rec in run_records:
            item = QtWidgets.QListWidgetItem(rec.display_label())
            item.setCheckState(Qt.Unchecked)
            item.setData(Qt.UserRole, rec.key)
            item.setToolTip(f"Run: {rec.run_id}\nGPKG: {rec.gpkg_path}")
            self._list.addItem(item)

        layout.addWidget(self._list)

        btn_row = QtWidgets.QHBoxLayout()
        select_all_btn = QtWidgets.QPushButton("Select All")
        select_all_btn.setToolTip("Check all runs in the list.")
        select_all_btn.clicked.connect(self._select_all)
        clear_all_btn = QtWidgets.QPushButton("Clear All")
        clear_all_btn.setToolTip("Uncheck all runs in the list.")
        clear_all_btn.clicked.connect(self._clear_all)
        btn_row.addWidget(select_all_btn)
        btn_row.addWidget(clear_all_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _select_all(self):
        """Check every item in the list."""
        for i in range(self._list.count()):
            self._list.item(i).setCheckState(Qt.Checked)

    def _clear_all(self):
        """Uncheck every item in the list."""
        for i in range(self._list.count()):
            self._list.item(i).setCheckState(Qt.Unchecked)

    def selected_keys(self) -> Set[str]:
        """Return the set of run keys the user checked."""
        keys = set()
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.checkState() == Qt.Checked:
                key = item.data(Qt.UserRole)
                if key:
                    keys.add(key)
        return keys
