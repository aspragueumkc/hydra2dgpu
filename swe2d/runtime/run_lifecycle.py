#!/usr/bin/env python3
"""Run lifecycle seam for SWE2D workbench.

Phase 11 goal: extract run failure handling and final cleanup from `_on_run`
into a focused helper module.
"""

from __future__ import annotations

from typing import Any, Callable


class SWE2DRunLifecycle:
    """Owns run failure reporting and final UI/backend cleanup."""

    def __init__(self, ui: Any):
        self._ui = ui

    def handle_run_failure(self, exc: Exception, show_error_callback: Callable[[str], None]) -> None:
        """handle run failure."""
        self._ui._log_exception("Run failed", exc)
        show_error_callback(
            "Run failed. Full traceback has been written to the runtime log pane.\n"
            f"Error: {exc}"
        )

    def finalize_cleanup(self, backend: Any) -> None:
        """finalize cleanup."""
        try:
            if backend is not None:
                backend.destroy()
        except Exception as exc:
            self._ui._log(f"[BACKEND] Backend destroy() failed: {exc}")
        self._ui.set_run_button_enabled(True)
        self._ui.set_cancel_button_enabled(False)
