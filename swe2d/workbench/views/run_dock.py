"""Dedicated Run dock for the HYDRA2D workbench."""
from __future__ import annotations

from qgis.PyQt import QtWidgets


class RunDockWidget(QtWidgets.QWidget):
    """Bottom dock with Run controls, progress, and output configuration."""

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

        # -- Output configuration --
        output_form = QtWidgets.QFormLayout()
        output_form.setObjectName("run_output_form")
        output_form.setContentsMargins(0, 0, 0, 0)

        self.output_interval_edit = QtWidgets.QLineEdit("00:30")
        self.output_interval_edit.setObjectName("output_interval_edit")
        self.output_interval_edit.setToolTip(
            "Time interval between 2D mesh result output writes. "
            "Format: decimal hours (e.g. 0.5) or HH:MM (e.g. 00:30). "
            "Smaller intervals produce larger result files."
        )
        output_form.addRow("Output interval (hr or HH:MM):", self.output_interval_edit)

        self.line_output_interval_edit = QtWidgets.QLineEdit("00:05")
        self.line_output_interval_edit.setObjectName("line_output_interval_edit")
        self.line_output_interval_edit.setToolTip(
            "Time interval between sample-line (cross-section) result outputs. "
            "Format: decimal hours or HH:MM. Default: 00:05 (5 min)."
        )
        output_form.addRow("Line output interval:", self.line_output_interval_edit)

        self.results_table_name_edit = QtWidgets.QLineEdit()
        self.results_table_name_edit.setObjectName("results_table_name_edit")
        self.results_table_name_edit.setToolTip(
            "Optional prefix for GeoPackage result table names. "
            "Useful when storing multiple model runs in the same GeoPackage."
        )
        self.results_table_name_edit.setPlaceholderText("optional table prefix")
        output_form.addRow("Table prefix:", self.results_table_name_edit)

        gpkg_row = QtWidgets.QHBoxLayout()
        self.results_gpkg_path_edit = QtWidgets.QLineEdit()
        self.results_gpkg_path_edit.setObjectName("results_gpkg_path_edit")
        self.results_gpkg_path_edit.setToolTip(
            "Path to the output GeoPackage for storing simulation results. "
            "Leave empty to use the model GeoPackage."
        )
        self.results_gpkg_path_edit.setPlaceholderText("GeoPackage path (optional)")
        self.select_results_gpkg_btn = QtWidgets.QPushButton("Browse…")
        self.select_results_gpkg_btn.setObjectName("select_results_gpkg_btn")
        self.select_results_gpkg_btn.setToolTip(
            "Browse for an existing GeoPackage to store/load simulation results."
        )
        gpkg_row.addWidget(self.results_gpkg_path_edit, 1)
        gpkg_row.addWidget(self.select_results_gpkg_btn)
        output_form.addRow("Results GPKG:", gpkg_row)

        layout.addLayout(output_form)

        # -- Preview / config --
        preview_row = QtWidgets.QHBoxLayout()
        self.preview_overrides_btn = QtWidgets.QPushButton("Preview Overrides")
        self.preview_overrides_btn.setObjectName("preview_overrides_btn")
        self.preview_overrides_btn.setToolTip(
            "Display a summary of all current parameter overrides "
            "before running the simulation."
        )
        self.preview_coupling_btn = QtWidgets.QPushButton("Preview Coupling")
        self.preview_coupling_btn.setObjectName("preview_coupling_btn")
        self.preview_coupling_btn.setToolTip(
            "Preview the 1D-2D coupling configuration for drainage "
            "and hydraulic structures before running."
        )
        self.load_run_settings_btn = QtWidgets.QPushButton("Load Config from GPKG…")
        self.load_run_settings_btn.setObjectName("load_run_settings_btn")
        self.load_run_settings_btn.setToolTip(
            "Open a GeoPackage and restore a saved simulation configuration "
            "(all widget values, solver params, and layer references)."
        )
        self.save_settings_btn = QtWidgets.QPushButton("Save Config to GPKG…")
        self.save_settings_btn.setObjectName("save_settings_btn")
        self.save_settings_btn.setToolTip(
            "Save the current widget configuration to the active GeoPackage "
            "so it can be restored later via Load Config."
        )
        preview_row.addWidget(self.preview_overrides_btn)
        preview_row.addWidget(self.preview_coupling_btn)
        preview_row.addStretch(1)
        preview_row.addWidget(self.load_run_settings_btn)
        preview_row.addWidget(self.save_settings_btn)
        layout.addLayout(preview_row)

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

    def get_results_gpkg_path(self) -> str:
        return str(self.results_gpkg_path_edit.text())

    def set_results_gpkg_path(self, path: str) -> None:
        self.results_gpkg_path_edit.setText(path)

    def get_output_interval(self) -> str:
        return str(self.output_interval_edit.text())

    def get_line_output_interval(self) -> str:
        return str(self.line_output_interval_edit.text())

    def get_results_table_prefix(self) -> str:
        return str(self.results_table_name_edit.text())

    def collect_params(self) -> dict:
        """Return output-config parameter values as a flat dict."""
        return {
            "output_interval_edit": str(self.output_interval_edit.text()),
            "line_output_interval_edit": str(self.line_output_interval_edit.text()),
            "results_table_name_edit": str(self.results_table_name_edit.text()),
            "results_gpkg_path_edit": str(self.results_gpkg_path_edit.text()),
        }
