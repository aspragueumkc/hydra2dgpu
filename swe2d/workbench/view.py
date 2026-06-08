#!/usr/bin/env python3
"""Thin view adapter for SWE2D workbench UI widgets.

Phase 1 extraction goal: define a stable UI facade for future controller
modularization without changing existing runtime behavior.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from qgis.PyQt import QtWidgets

logger = logging.getLogger(__name__)


class SWE2DWorkbenchViewAdapter:
    """Facade over dialog widgets used by run/orchestration paths.

    This adapter intentionally exposes only lightweight getters/setters and
    signal binding helpers so orchestration can move out of the dialog class
    in subsequent refactor phases.
    """

    def __init__(self, dialog: QtWidgets.QDialog):
        self._dialog = dialog

    def on_run_requested(self, callback: Callable[[], None]) -> None:
        btn = getattr(self._dialog, "run_btn", None)
        if btn is not None:
            btn.clicked.connect(callback)

    def on_cancel_requested(self, callback: Callable[[], None]) -> None:
        btn = getattr(self._dialog, "cancel_btn", None)
        if btn is not None:
            btn.clicked.connect(callback)

    def on_snapshot_requested(self, callback: Callable[[], None]) -> None:
        btn = getattr(self._dialog, "snapshot_btn", None)
        if btn is not None:
            btn.clicked.connect(callback)

    def set_run_state(self, running: bool) -> None:
        run_btn = getattr(self._dialog, "run_btn", None)
        cancel_btn = getattr(self._dialog, "cancel_btn", None)
        if run_btn is not None:
            run_btn.setEnabled(not running)
        if cancel_btn is not None:
            cancel_btn.setEnabled(running)

    def set_progress_value(self, value: int) -> None:
        bar = getattr(self._dialog, "progress_bar", None)
        if bar is not None:
            bar.setValue(int(value))

    def append_runtime_log(self, message: str) -> None:
        log_view = getattr(self._dialog, "log_view", None)
        if log_view is not None:
            log_view.appendPlainText(str(message))

    def runtime_log_text(self) -> str:
        log_view = getattr(self._dialog, "log_view", None)
        if log_view is None:
            return ""
        try:
            return str(log_view.toPlainText())
        except Exception as exc:
            logger.debug("[UI] runtime_log_text fallback: %s", exc)
            return ""

    def run_duration_text(self) -> str:
        widget = getattr(self._dialog, "run_time_edit", None)
        return "" if widget is None else str(widget.text())

    def output_interval_text(self) -> str:
        widget = getattr(self._dialog, "output_interval_edit", None)
        return "" if widget is None else str(widget.text())

    def line_output_interval_text(self) -> str:
        widget = getattr(self._dialog, "line_output_interval_edit", None)
        return "" if widget is None else str(widget.text())

    def adaptive_dt_enabled(self) -> bool:
        widget = getattr(self._dialog, "adaptive_cfl_dt_chk", None)
        if widget is None:
            return False
        try:
            return bool(widget.isChecked())
        except Exception as exc:
            logger.debug("[UI] adaptive_dt_enabled fallback: %s", exc)
            return False

    def requested_dt(self) -> float:
        widget = getattr(self._dialog, "dt_spin", None)
        if widget is None:
            return 0.0
        try:
            return float(widget.value())
        except Exception as exc:
            logger.debug("[UI] requested_dt fallback: %s", exc)
            return 0.0

    def selected_view_mode(self) -> Optional[str]:
        combo = getattr(self._dialog, "view_mode_combo", None)
        if combo is None:
            return None
        try:
            return str(combo.currentText())
        except Exception as exc:
            logger.debug("[UI] selected_view_mode fallback: %s", exc)
            return None
