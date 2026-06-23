"""Builder for SWE2DWorkbenchStudioDialog.

Extracts the post-init orchestration from the dialog's __init__
into a dedicated class. The dialog's __init__ becomes a thin
bootstrapper that calls this builder.

Created as part of Phase 1 Task 2 (extract __init__ logic into builder).
"""
from __future__ import annotations

import logging

from swe2d.workbench.post_init import run_workbench_post_bootstrap_setup
from swe2d.workbench.controllers.run_component_wiring_controller import wire_startup_run_components
from swe2d.runtime import (
    SWE2DBackendInitializer,
    SWE2DRunController,
    SWE2DRunDataBuilder,
    SWE2DRunFinalizer,
    SWE2DRunLifecycle,
    SWE2DRunOptionsBuilder,
    SWE2DRunOrchestrator,
    SWE2DRunRequest,
)
from swe2d.workbench.controllers.startup_bootstrap_controller import bootstrap_startup_run_components
from swe2d.workbench.startup_state import initialize_workbench_startup_state
from swe2d.workbench.workbench_view_state import WorkbenchViewState
from swe2d.workbench.controllers.run_controller import RunController
from swe2d.workbench.controllers.layer_controller import LayerController
from swe2d.workbench.controllers.mesh_controller import MeshController
from swe2d.workbench.controllers.overlay_controller import OverlayController
from swe2d.workbench.controllers.topology_controller import TopologyController
from swe2d.workbench.controllers.workbench_controller import WorkbenchController
from swe2d.mesh.gmsh_backend import _gmsh_available
from swe2d.runtime.backend import (
    SpatialDiscretization,
    SolverModelOptions,
    TemporalScheme,
    swe2d_gpu_available,
)

logger = logging.getLogger(__name__)


class WorkbenchDialogBuilder:
    """Builds and configures a SWE2DWorkbenchStudioDialog.

    The dialog's __init__ should be a thin bootstrapper that:
    1. Calls super().__init__(parent)
    2. Stores minimal state
    3. Calls this builder's configure() to do all the post-init orchestration
    """

    def __init__(self, dialog):
        self._dialog = dialog

    def configure(self) -> None:
        """Run all post-init configuration on the dialog.

        This is the single entry point for dialog setup after
        super().__init__() and minimal attribute initialization.
        """
        from qgis.PyQt import QtCore as _QtCore
        import concurrent.futures as _concurrent_futures

        dlg = self._dialog
        dlg._state = WorkbenchViewState(iface=dlg.iface)
        dlg._controller = RunController(view=dlg)
        dlg._layer_controller = LayerController(view=dlg)
        dlg._mesh_controller = MeshController(view=dlg)
        dlg._overlay_controller = OverlayController(view=dlg)
        dlg._topology_controller = TopologyController(view=dlg)
        dlg._workbench_controller = WorkbenchController(
            view=dlg,
            run_controller=dlg._controller,
            mesh_controller=dlg._mesh_controller,
            overlay_controller=dlg._overlay_controller,
            topology_controller=dlg._topology_controller,
            layer_controller=dlg._layer_controller,
        )
        initialize_workbench_startup_state(
            dlg,
            qtcore_module=_QtCore,
            concurrent_futures_module=_concurrent_futures,
            try_import_matplotlib_qt=dlg._try_import_matplotlib_qt,
        )
        dlg._wire_runtime_log_handler()
        dlg._build_ui()
        bootstrap_startup_run_components(
            dlg,
            wire_startup_run_components,
            run_orchestrator=SWE2DRunOrchestrator,
            run_request=SWE2DRunRequest,
            run_controller=SWE2DRunController,
            run_data_builder=SWE2DRunDataBuilder,
            run_options_builder=SWE2DRunOptionsBuilder,
            backend_initializer=SWE2DBackendInitializer,
            run_finalizer=SWE2DRunFinalizer,
            run_lifecycle=SWE2DRunLifecycle,
            swe2d_gpu_available=swe2d_gpu_available,
            temporal_scheme=TemporalScheme,
            spatial_discretization=SpatialDiscretization,
            solver_model_options=SolverModelOptions,
        )
        run_workbench_post_bootstrap_setup(
            dlg,
            swe2d_gpu_available_fn=swe2d_gpu_available,
            gmsh_available_fn=_gmsh_available,
        )
