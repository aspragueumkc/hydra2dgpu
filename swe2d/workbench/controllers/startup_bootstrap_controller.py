#!/usr/bin/env python3
"""Startup bootstrap helper for SWE2D workbench seam wiring."""

from __future__ import annotations

from typing import Any, Callable, Dict


def bootstrap_startup_run_components(
    dialog: Any,
    wire_fn: Callable[[Any, Dict[str, Any]], None],
    *,
    run_orchestrator: Any,
    run_request: Any,
    run_controller: Any,
    backend_initializer: Any,
    run_finalizer: Any,
    run_lifecycle: Any,
) -> None:
    """Wire startup run seam components."""
    startup_ns = {
        "SWE2DRunOrchestrator": run_orchestrator,
        "SWE2DRunRequest": run_request,
        "SWE2DRunController": run_controller,
        "SWE2DBackendInitializer": backend_initializer,
        "SWE2DRunFinalizer": run_finalizer,
        "SWE2DRunLifecycle": run_lifecycle,
    }
    wire_fn(dialog, startup_ns)
