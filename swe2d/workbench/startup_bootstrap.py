#!/usr/bin/env python3
"""Startup bootstrap helper for SWE2D workbench seam wiring."""

from __future__ import annotations

from typing import Any, Callable, Dict


def bootstrap_startup_run_components(
    dialog: Any,
    wire_fn: Callable[[Any, Dict[str, Any]], None] | None,
    *,
    view_adapter: Any,
    run_orchestrator: Any,
    run_request: Any,
    run_controller: Any,
    run_data_builder: Any,
    run_options_builder: Any,
    backend_initializer: Any,
    run_finalizer: Any,
    run_lifecycle: Any,
    swe2d_gpu_available: Any,
    temporal_scheme: Any,
    spatial_discretization: Any,
    solver_model_options: Any,
) -> None:
    """Wire startup run seams or emit the existing unavailable warning path."""
    if wire_fn is not None:
        startup_ns = {
            "SWE2DWorkbenchViewAdapter": view_adapter,
            "SWE2DRunOrchestrator": run_orchestrator,
            "SWE2DRunRequest": run_request,
            "SWE2DRunController": run_controller,
            "SWE2DRunDataBuilder": run_data_builder,
            "SWE2DRunOptionsBuilder": run_options_builder,
            "SWE2DBackendInitializer": backend_initializer,
            "SWE2DRunFinalizer": run_finalizer,
            "SWE2DRunLifecycle": run_lifecycle,
            "swe2d_gpu_available": swe2d_gpu_available,
            "TemporalScheme": temporal_scheme,
            "SpatialDiscretization": spatial_discretization,
            "SolverModelOptions": solver_model_options,
        }
        wire_fn(dialog, startup_ns)
        return

    dialog._note_startup_component_missing("run seam wiring helper", required_for_run=True)
    if dialog._startup_run_component_errors:
        dialog._log(
            "Startup run seam readiness warning: "
            + ", ".join(sorted(set(dialog._startup_run_component_errors)))
        )
