#!/usr/bin/env python3
"""Startup wiring helper for SWE2D workbench run/runtime seam components."""

from __future__ import annotations

from typing import Any, Dict


def wire_startup_run_components(dialog: Any, ns: Dict[str, Any]) -> None:
    """Instantiate run/runtime seam components directly."""
    SWE2DRunOrchestrator = ns["SWE2DRunOrchestrator"]
    SWE2DRunRequest = ns["SWE2DRunRequest"]
    SWE2DRunController = ns["SWE2DRunController"]
    SWE2DBackendInitializer = ns["SWE2DBackendInitializer"]
    SWE2DRunFinalizer = ns["SWE2DRunFinalizer"]
    SWE2DRunLifecycle = ns["SWE2DRunLifecycle"]

    dialog._run_orchestrator = SWE2DRunOrchestrator(
        dialog._dispatch_run_request, dialog._log,
    )

    dialog._run_controller = SWE2DRunController(
        ensure_mesh_callback=dialog._ensure_mesh_for_run_preflight,
        has_mesh_callback=dialog._has_mesh_for_run_preflight,
        backend_ready_callback=dialog._backend_ready_for_run_preflight,
        backend_unavailable_callback=dialog._show_backend_unavailable_for_run_preflight,
        log_callback=dialog._log,
    )

    dialog._backend_initializer = SWE2DBackendInitializer(
        apply_timeseries_bc_values_callback=dialog._apply_timeseries_bc_values,
        distribute_total_flow_to_unit_q_callback=dialog._distribute_total_flow_to_unit_q,
    )

    from swe2d.workbench.controllers.finalization_adapter import FinalizationAdapter

    dialog._run_finalizer = SWE2DRunFinalizer(FinalizationAdapter(dialog))

    dialog._run_lifecycle = SWE2DRunLifecycle(dialog)
