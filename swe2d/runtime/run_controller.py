#!/usr/bin/env python3
"""Run preflight controller for SWE2D workbench.

Phase 3 goal: isolate run-readiness orchestration (mesh/backend checks)
from dialog UI and solver execution body.
"""

from __future__ import annotations

from typing import Callable, Optional


class SWE2DRunController:
    """Coordinates preflight checks before dispatching a run request."""

    def __init__(
        self,
        ensure_mesh_callback: Callable[[], None],
        has_mesh_callback: Callable[[], bool],
        backend_ready_callback: Callable[[], bool],
        backend_unavailable_callback: Optional[Callable[[str], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
    ):
        self._ensure_mesh_callback = ensure_mesh_callback
        self._has_mesh_callback = has_mesh_callback
        self._backend_ready_callback = backend_ready_callback
        self._backend_unavailable_callback = backend_unavailable_callback
        self._log_callback = log_callback

    def run_preflight(self, request=None) -> bool:
        """Run preflight."""
        self._ensure_mesh_callback()

        if not bool(self._has_mesh_callback()):
            if callable(self._log_callback):
                self._log_callback("Run preflight aborted: mesh is not available.")
            return False

        if not bool(self._backend_ready_callback()):
            msg = (
                "GPU backend (CUDA) is not available. "
                "Ensure CUDA toolkit and the hydra_swe2d module are built correctly."
            )
            if callable(self._backend_unavailable_callback):
                self._backend_unavailable_callback(msg)
            if callable(self._log_callback):
                self._log_callback("Run preflight aborted: native backend unavailable.")
            return False

        return True
