#!/usr/bin/env python3
"""Import-resolution shim for SWE2D workbench seam components."""

from __future__ import annotations


try:
    from swe2d.workbench.run_component_wiring import wire_startup_run_components
except Exception:
    try:
        from .run_component_wiring import wire_startup_run_components
    except Exception:
        wire_startup_run_components = None

try:
    from swe2d.workbench.view import SWE2DWorkbenchViewAdapter
except Exception:
    try:
        from .view import SWE2DWorkbenchViewAdapter
    except Exception:
        SWE2DWorkbenchViewAdapter = None

try:
    from swe2d.runtime import (
        SWE2DRunController,
        SWE2DRunDataBuilder,
        SWE2DRunFinalizer,
        SWE2DRunLifecycle,
        SWE2DRunOptionsBuilder,
        SWE2DRunOrchestrator,
        SWE2DRunRequest,
    )
except Exception:
    SWE2DRunController = None
    SWE2DRunDataBuilder = None
    SWE2DRunFinalizer = None
    SWE2DRunLifecycle = None
    SWE2DRunOptionsBuilder = None
    SWE2DRunOrchestrator = None
    SWE2DRunRequest = None

try:
    from swe2d.runtime import SWE2DBackendInitializer
except Exception:
    SWE2DBackendInitializer = None

try:
    from swe2d.runtime import (
        SWE2DNativeBoundaryHydrographConfigurator,
        SWE2DRunSetupConfigurator,
        SWE2DRuntimeReporter,
        SWE2DRuntimeSourceManager,
        SWE2DRuntimeStepExecutor,
    )
except Exception:
    SWE2DNativeBoundaryHydrographConfigurator = None
    SWE2DRunSetupConfigurator = None
    SWE2DRuntimeReporter = None
    SWE2DRuntimeSourceManager = None
    SWE2DRuntimeStepExecutor = None
