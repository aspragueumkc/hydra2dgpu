"""Run/runtime orchestration exports for incremental package migration."""

from swe2d.runtime.diagnostics import SWE2DRunReport, _configure_diagnostics_mode

_configure_diagnostics_mode()

from swe2d.runtime.bridge_stacked_runtime import build_bridge_stacked_plans_for_runtime
from swe2d.runtime.backend_initializer import SWE2DBackendInitializer
from swe2d.runtime.native_bc_forcing import SWE2DNativeBoundaryHydrographConfigurator
from swe2d.runtime.run_controller import SWE2DRunController
from swe2d.runtime.run_data_builder import SWE2DRunDataBuilder
from swe2d.runtime.run_finalizer import SWE2DRunFinalizer
from swe2d.runtime.run_lifecycle import SWE2DRunLifecycle
from swe2d.runtime.run_options_builder import SWE2DRunOptionsBuilder
from swe2d.runtime.run_orchestrator import SWE2DRunOrchestrator, SWE2DRunRequest
from swe2d.runtime.runtime_reporting import SWE2DRuntimeReporter
from swe2d.runtime.runtime_setup_configurator import SWE2DRunSetupConfigurator
from swe2d.runtime.runtime_sources import SWE2DRuntimeSourceManager
from swe2d.runtime.runtime_step_executor import SWE2DRuntimeStepExecutor

__all__ = [
    "SWE2DBackendInitializer",
    "SWE2DNativeBoundaryHydrographConfigurator",
    "SWE2DRunController",
    "SWE2DRunDataBuilder",
    "SWE2DRunFinalizer",
    "SWE2DRunLifecycle",
    "SWE2DRunOptionsBuilder",
    "SWE2DRunOrchestrator",
    "SWE2DRunRequest",
    "SWE2DRunReport",
    "SWE2DRuntimeReporter",
    "SWE2DRunSetupConfigurator",
    "SWE2DRuntimeSourceManager",
    "SWE2DRuntimeStepExecutor",
    "build_bridge_stacked_plans_for_runtime",
]
