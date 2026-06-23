#!/usr/bin/env python3
"""Run orchestration seam for SWE2D workbench.

Phase 2 goal: centralize run entry and request snapshot handling while
preserving legacy execution behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional


@dataclass(frozen=True)
class SWE2DRunRequest:
    """Immutable run request snapshot captured from UI state."""

    run_duration_text: str
    output_interval_text: str
    line_output_interval_text: str
    adaptive_dt_enabled: bool
    requested_dt: float
    created_utc: str

    @staticmethod
    def from_ui_values(
        run_duration_text: str,
        output_interval_text: str,
        line_output_interval_text: str,
        adaptive_dt_enabled: bool,
        requested_dt: float,
    ) -> "SWE2DRunRequest":
        """from ui values."""
        return SWE2DRunRequest(
            run_duration_text=str(run_duration_text or "").strip(),
            output_interval_text=str(output_interval_text or "").strip(),
            line_output_interval_text=str(line_output_interval_text or "").strip(),
            adaptive_dt_enabled=bool(adaptive_dt_enabled),
            requested_dt=float(requested_dt),
            created_utc=datetime.now(timezone.utc).isoformat(),
        )


class SWE2DRunOrchestrator:
    """Thin orchestration wrapper around existing run execution."""

    def __init__(
        self,
        execute_callback: Callable[[SWE2DRunRequest], None],
        log_callback: Optional[Callable[[str], None]] = None,
    ):
        self._execute_callback = execute_callback
        self._log_callback = log_callback
        self._run_active = False

    def run(self, request: SWE2DRunRequest) -> bool:
        """run."""
        if self._run_active:
            if callable(self._log_callback):
                self._log_callback("Run request ignored: another run is already active.")
            return False

        self._run_active = True
        try:
            self._execute_callback(request)
            return True
        finally:
            self._run_active = False
