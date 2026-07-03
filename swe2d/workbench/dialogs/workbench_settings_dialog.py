"""Small settings dialog for HYDRA workbench feature flags."""
from __future__ import annotations

from typing import Dict

from qgis.PyQt import QtWidgets


class WorkbenchSettingsDialog(QtWidgets.QDialog):
    """Let the user toggle workbench module feature flags."""

    _FLAGS = [
        ("rainfall", "Enable rainfall module"),
        ("drainage_structures", "Enable drainage networks"),
        ("hydraulic_structures", "Enable hydraulic structures (weirs, culverts, bridges)"),
        ("bridge_stacked_coupling", "Enable bridge stacked coupling (experimental)"),
    ]

    def __init__(self, feature_flags: Dict[str, bool], parent=None):
        super().__init__(parent)
        self.setWindowTitle("HYDRA Settings")
        self._checks: Dict[str, QtWidgets.QCheckBox] = {}
        layout = QtWidgets.QVBoxLayout(self)
        for key, label in self._FLAGS:
            chk = QtWidgets.QCheckBox(label)
            chk.setChecked(feature_flags.get(key, True))
            layout.addWidget(chk)
            self._checks[key] = chk
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def flags(self) -> Dict[str, bool]:
        """Return the updated feature flag map."""
        return {key: chk.isChecked() for key, chk in self._checks.items()}
