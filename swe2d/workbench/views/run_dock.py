"""Dedicated Run dock for the HYDRA2D workbench.

Owns the *execution* surface: Run / Cancel / Snapshot / Batch buttons
and the progress bar. The output-configuration widgets
(output_interval, results_table_name,
results_gpkg_path + Browse, and the Preview / Load / Save config
buttons) live on the Simulation tab's Output page now
(``ModelTabView._build_run_output_section``). Read them from there.
"""
from __future__ import annotations

from qgis.PyQt import QtWidgets


class RunDockWidget(QtWidgets.QWidget):
    """Bottom dock with Run controls and progress bar only."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(4)

        # -- Execution controls --
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

    # ------------------------------------------------------------------
    # Direct accessors for execution-surface widgets
    # ------------------------------------------------------------------

    def set_run_button_enabled(self, enabled: bool) -> None:
        self.run_btn.setEnabled(enabled)

    def set_cancel_button_enabled(self, enabled: bool) -> None:
        self.cancel_btn.setEnabled(enabled)

    def set_progress_bar_value(self, value: int) -> None:
        self.progress_bar.setValue(value)

    def get_run_btn(self) -> QtWidgets.QPushButton:
        return self.run_btn

    def get_cancel_btn(self) -> QtWidgets.QPushButton:
        return self.cancel_btn

    def get_progress_bar(self) -> QtWidgets.QProgressBar:
        return self.progress_bar
