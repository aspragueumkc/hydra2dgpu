"""Diagnostics infrastructure for SWE2D runtime."""

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class SWE2DRunReport:
    """Captures fallback events and execution metadata for a SWE2D run."""
    fallback_events: List[Dict[str, object]] = field(default_factory=list)
    gpu_available: bool = True
    openmp_enabled: bool = True
    backend_version: int = 0

    def record_fallback(self, component: str, reason: str, details: str = "") -> None:
        self.fallback_events.append({
            "component": component,
            "reason": reason,
            "details": details,
        })

    def summary(self) -> str:
        n = len(self.fallback_events)
        return f"[RUNTIME] Run report: {n} fallback(s), GPU={self.gpu_available}, OpenMP={self.openmp_enabled}"


def _configure_diagnostics_mode() -> None:
    """Enable DEBUG-level logging for swe2d when BACKWATER_SWE2D_DIAG_MODE=1."""
    if os.environ.get("BACKWATER_SWE2D_DIAG_MODE", "0") == "1":
        logging.getLogger("swe2d").setLevel(logging.DEBUG)
        logging.getLogger("swe2d").info("[RUNTIME] Diagnostics mode enabled via BACKWATER_SWE2D_DIAG_MODE=1")
