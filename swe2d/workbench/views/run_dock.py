"""Dedicated Run dock for the HYDRA2D workbench.

The dock now only owns the *execution* surface: Run / Cancel / Snapshot /
Batch buttons and the progress bar. The output-configuration widgets
(output_interval, line_output_interval, results_table_name,
results_gpkg_path + Browse, and the Preview / Load / Save config
buttons) moved to the Simulation tab's Output page.

The class still exposes legacy getters/setters that delegate to the
new host (model_tab_view) when the moved widgets are no longer
present, so external callers that read these values keep working.
"""
from __future__ import annotations

from qgis.PyQt import QtWidgets


class RunDockWidget(QtWidgets.QWidget):
    """Bottom dock with Run controls and progress bar only."""

    def __init__(self, parent=None):
        super().__init__(parent)
        # Optional reference to the model tab view where the moved
        # output-config widgets live now. Set by the dialog builder.
        self._output_widgets_host = None
        self._build_ui()

    def attach_output_widgets_host(self, host) -> None:
        """Bind the host that owns the moved output-config widgets.

        ``host`` is typically the ModelTabView instance. After this call,
        legacy accessors like ``get_output_interval()`` delegate to the
        host so external code that reads the Run dock keeps working.
        """
        self._output_widgets_host = host

    def _host_attr(self, attr: str):
        """Return the host attribute if available, else None."""
        host = self._output_widgets_host
        if host is None:
            return None
        return getattr(host, attr, None)

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

        # Output-configuration widgets (output_interval_edit,
        # line_output_interval_edit, results_table_name_edit,
        # results_gpkg_path_edit, select_results_gpkg_btn,
        # preview_overrides_btn, preview_coupling_btn,
        # load_run_settings_btn, save_settings_btn) were moved to the
        # Simulation tab's Output page — see ModelTabView.

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

    # ------------------------------------------------------------------
    # Legacy accessors for moved output-config widgets
    # ------------------------------------------------------------------
    # These delegate to the host (ModelTabView) where the widgets
    # physically live now. Returning "" / a no-op when the host is not
    # attached keeps external callers from breaking before the host
    # binding is established.

    def get_results_gpkg_path(self) -> str:
        edit = self._host_attr("results_gpkg_path_edit")
        if edit is None:
            return ""
        try:
            return str(edit.text())
        except RuntimeError:
            return ""

    def set_results_gpkg_path(self, path: str) -> None:
        edit = self._host_attr("results_gpkg_path_edit")
        if edit is None:
            return
        try:
            edit.setText(str(path))
        except RuntimeError:
            pass

    def get_output_interval(self) -> str:
        edit = self._host_attr("output_interval_edit")
        if edit is None:
            return ""
        try:
            return str(edit.text())
        except RuntimeError:
            return ""

    def get_line_output_interval(self) -> str:
        edit = self._host_attr("line_output_interval_edit")
        if edit is None:
            return ""
        try:
            return str(edit.text())
        except RuntimeError:
            return ""

    def get_results_table_prefix(self) -> str:
        edit = self._host_attr("results_table_name_edit")
        if edit is None:
            return ""
        try:
            return str(edit.text())
        except RuntimeError:
            return ""

    def collect_params(self) -> dict:
        """Return output-config parameter values as a flat dict.

        Reads from the host (ModelTabView) where these widgets now
        physically live. Returns empty strings for any field not yet
        available, so callers don't have to special-case it.
        """
        def _text(attr: str) -> str:
            w = self._host_attr(attr)
            if w is None:
                return ""
            try:
                return str(w.text())
            except RuntimeError:
                return ""

        return {
            "output_interval_edit": _text("output_interval_edit"),
            "line_output_interval_edit": _text("line_output_interval_edit"),
            "results_table_name_edit": _text("results_table_name_edit"),
            "results_gpkg_path_edit": _text("results_gpkg_path_edit"),
        }
