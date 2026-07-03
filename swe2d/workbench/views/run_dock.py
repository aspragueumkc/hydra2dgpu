"""Dedicated Run dock for the HYDRA2D workbench."""
from __future__ import annotations

from qgis.PyQt import QtWidgets


class RunDockWidget(QtWidgets.QWidget):
    """Bottom dock with Run/Cancel/Snapshot/Batch and a progress bar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(4)

        row = QtWidgets.QHBoxLayout()
        self.run_btn = QtWidgets.QPushButton("▶ Run 2D Model")
        self.run_btn.setObjectName("run_btn")
        self.cancel_btn = QtWidgets.QPushButton("⏹ Cancel")
        self.cancel_btn.setObjectName("cancel_btn")
        self.cancel_btn.setEnabled(False)
        self.snapshot_btn = QtWidgets.QPushButton("📸 Snapshot")
        self.snapshot_btn.setObjectName("snapshot_btn")
        self.batch_btn = QtWidgets.QPushButton("Batch…")
        self.batch_btn.setObjectName("batch_btn")

        row.addWidget(self.run_btn)
        row.addWidget(self.cancel_btn)
        row.addWidget(self.snapshot_btn)
        row.addStretch(1)
        row.addWidget(self.batch_btn)
        layout.addLayout(row)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setObjectName("progress_bar")
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
