#!/usr/bin/env python3
"""Import-resolution shim for SWE2D workbench seam components."""

from __future__ import annotations

from swe2d.workbench.run_component_wiring import wire_startup_run_components
from swe2d.workbench.view import SWE2DWorkbenchViewAdapter

from swe2d.runtime import (
    SWE2DBackendInitializer,
    SWE2DNativeBoundaryHydrographConfigurator,
    SWE2DRunController,
    SWE2DRunDataBuilder,
    SWE2DRunFinalizer,
    SWE2DRunLifecycle,
    SWE2DRunOptionsBuilder,
    SWE2DRunOrchestrator,
    SWE2DRunRequest,
    SWE2DRunSetupConfigurator,
    SWE2DRuntimeReporter,
    SWE2DRuntimeSourceManager,
    SWE2DRuntimeStepExecutor,
)
