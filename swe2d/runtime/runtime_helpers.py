#!/usr/bin/env python3
"""Temporary aggregation module for SWE2D runtime helper seams.

This keeps existing implementations in their current files while allowing
the workbench to import a compact runtime-helper surface.
"""

from __future__ import annotations

from swe2d.runtime.native_bc_forcing import SWE2DNativeBoundaryHydrographConfigurator
from swe2d.runtime.runtime_reporting import SWE2DRuntimeReporter
from swe2d.runtime.runtime_setup_configurator import SWE2DRunSetupConfigurator
from swe2d.runtime.runtime_sources import SWE2DRuntimeSourceManager
from swe2d.runtime.runtime_step_executor import SWE2DRuntimeStepExecutor

__all__ = [
    "SWE2DNativeBoundaryHydrographConfigurator",
    "SWE2DRunSetupConfigurator",
    "SWE2DRuntimeReporter",
    "SWE2DRuntimeSourceManager",
    "SWE2DRuntimeStepExecutor",
]
