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
        invert_btn = QtWidgets.QPushButton("Invert")
        invert_btn.setToolTip("Toggle every run check state.")
        invert_btn.clicked.connect(self._invert_selection)
        newest_btn = QtWidgets.QPushButton("Only newest")
        newest_btn.setToolTip("Keep only the most recently created run checked.")
        newest_btn.clicked.connect(self._select_only_newest)
        btn_row.addWidget(select_all_btn)
        btn_row.addWidget(clear_all_btn)
        btn_row.addWidget(invert_btn)
        btn_row.addWidget(newest_btn)
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

    def _invert_selection(self):
        """Toggle every item's check state."""
        for i in range(self._list.count()):
            item = self._list.item(i)
            item.setCheckState(
                Qt.Checked if item.checkState() == Qt.Unchecked else Qt.Unchecked
            )

    def _select_only_newest(self):
        """Check only the run with the latest created_utc timestamp."""
        newest_idx = None
        newest_ts = ""
        for i in range(self._list.count()):
            item = self._list.item(i)
            item.setCheckState(Qt.Unchecked)
            rec = self._records[i]
            ts = str(rec.created_utc or "")
            if ts and (not newest_ts or ts > newest_ts):
                newest_ts = ts
                newest_idx = i
        if newest_idx is None:
            # Fallback: use the first item if no timestamps are present.
            newest_idx = 0
        self._list.item(newest_idx).setCheckState(Qt.Checked)

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
