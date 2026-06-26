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
        initialize_workbench_startup_state(
            dlg,
            qtcore_module=_QtCore,
            concurrent_futures_module=_concurrent_futures,
            try_import_matplotlib_qt=dlg._try_import_matplotlib_qt,
        )
        dlg._wire_runtime_log_handler()
        self._build_dialog_ui()
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

    # ── UI construction (extracted from studio_dialog._build_ui) ────────

    def _build_dialog_ui(self):
        """Build the Studio dialog UI with all dock components, toolbar, and tabs."""
        from qgis.PyQt import QtCore, QtWidgets

        dlg = self._dialog
        try:
            dlg._log("[Studio] _build_ui entered")
        except Exception as e:
            dlg._log(f"[ERROR] build ui failed: {e}")
        root = dlg.layout()
        if not isinstance(root, QtWidgets.QVBoxLayout):
            root = QtWidgets.QVBoxLayout(dlg)

        # View/theme combos
        view_bar = QtWidgets.QWidget()
        view_bar_layout = QtWidgets.QHBoxLayout(view_bar)
        view_bar_layout.setContentsMargins(6, 2, 6, 2)
        view_bar_layout.addWidget(QtWidgets.QLabel(" View: "))
        dlg._state.studio_view_mode_combo = QtWidgets.QComboBox()
        dlg._state.studio_view_mode_combo.addItems([
            "Mesh", "Depth", "Velocity magnitude",
            "Time-Series", "Profile", "Structures", "Network",
        ])
        view_bar_layout.addWidget(dlg._state.studio_view_mode_combo)
        view_bar_layout.addWidget(QtWidgets.QLabel(" Theme: "))
        dlg._state.studio_theme_combo = QtWidgets.QComboBox()
        dlg._state.studio_theme_combo.addItems(["Default", "Diagnostics", "Presentation"])
        view_bar_layout.addWidget(dlg._state.studio_theme_combo)
        view_bar_layout.addStretch(1)
        root.addWidget(view_bar)

        from swe2d.workbench.views.studio_viewer import SWE2DStudioViewer
        dlg._studio_viewer = SWE2DStudioViewer()

        if dlg.iface is None:
            root.addWidget(dlg._studio_viewer, stretch=1)

        self._build_component(
            name="viewer",
            title="HYDRA2D View",
            area=QtCore.Qt.RightDockWidgetArea,
            tab_with="inspector",
            populate=lambda dock: dock.setWidget(dlg._studio_viewer),
            iface=dlg.iface,
        )

        self._build_component(
            name="setup",
            title="HYDRA2D Model Setup",
            area=QtCore.Qt.LeftDockWidgetArea,
            populate=self._populate_setup_dock,
            iface=dlg.iface,
        )
        dlg._state.studio_left_dock = dlg._state.studio_components["setup"].dock

        dlg._ts_dock = None
        dlg._prof_dock = None
        dlg._struct_dock = None
        dlg._network_dock = None

        self._build_component(
            name="inspector",
            title="HYDRA2D CFD Inspector",
            area=QtCore.Qt.RightDockWidgetArea,
            populate=self._populate_inspector_dock,
            iface=dlg.iface,
        )
        dlg._state.studio_inspector_dock = dlg._state.studio_components["inspector"].dock

        from swe2d.workbench.views.results_controls import ResultsToolbox
        dlg._results_toolbox = ResultsToolbox()
        dlg._results_toolbox.overlay_toggled.connect(
            dlg._overlay_controller.on_high_perf_canvas_overlay_toggled)
        dlg._results_toolbox.overlay_style_changed.connect(
            dlg._overlay_controller.on_high_perf_canvas_overlay_style_changed)
        dlg._results_toolbox.overlay_export_geotiff.connect(
            dlg._overlay_controller.export_high_perf_overlay_to_geotiff)
        dlg._results_toolbox.run_selection_changed.connect(
            dlg._on_run_selection_changed)
        dlg._results_toolbox.run_refresh_requested.connect(
            dlg._on_results_refresh)
        dlg._results_toolbox.run_add_requested.connect(
            dlg._on_results_add)
        dlg._results_toolbox.run_remove_requested.connect(
            dlg._on_results_remove)
        dlg._results_toolbox.run_show_all.connect(
            dlg._on_results_show_all)
        dlg._results_toolbox.run_hide_all.connect(
            dlg._on_results_hide_all)
        self._build_component(
            name="results",
            title="HYDRA2D Results",
            area=QtCore.Qt.RightDockWidgetArea,
            tab_with="inspector",
            populate=lambda dock: dock.setWidget(dlg._results_toolbox),
            iface=dlg.iface,
        )
        dlg._state.studio_results_dock = dlg._state.studio_components["results"].dock

        from swe2d.workbench.views.temporal_dock import TemporalDockWidget
        dlg._temporal_dock = TemporalDockWidget()
        self._build_component(
            name="temporal",
            title="HYDRA2D Temporal",
            area=QtCore.Qt.BottomDockWidgetArea,
            populate=lambda dock: dock.setWidget(dlg._temporal_dock),
            iface=dlg.iface,
        )

        self._build_component(
            name="log",
            title="HYDRA2D Log",
            area=QtCore.Qt.BottomDockWidgetArea,
            populate=self._populate_log_dock,
            iface=dlg.iface,
        )

        footer = QtWidgets.QStatusBar(dlg)
        dlg._state.studio_status_label = QtWidgets.QLabel("")
        footer.addPermanentWidget(dlg._state.studio_status_label, 1)
        root.addWidget(footer)
        dlg._studio_update_status()

        if hasattr(dlg, "view_mode_combo") and dlg.view_mode_combo is not None:
            try:
                dlg._state.studio_view_mode_combo.setCurrentIndex(
                    max(0, int(dlg.view_mode_combo.currentIndex())))
            except Exception as e:
                dlg._log(f"[ERROR] build ui failed: {e}")

        dlg._state.studio_view_mode_combo.currentIndexChanged.connect(
            self._studio_sync_view_mode)
        dlg._state.studio_theme_combo.currentTextChanged.connect(
            self._studio_apply_visual_profile)

        self._studio_apply_visual_profile("Default")
        dlg._studio_apply_feature_filters()
        dlg._layer_controller.refresh_layer_combos()
        self._validate_widget_bindings()

    def _build_component(
        self,
        name: str,
        title: str,
        area: "QtCore.Qt.DockWidgetArea" = None,
        tab_with: str | None = None,
        populate: "Callable[[QtWidgets.QDockWidget], None] | None" = None,
        iface: "Any | None" = None,
    ):
        """Build, register, and optionally attach a dock component."""
        from qgis.PyQt import QtCore, QtWidgets

        dlg = self._dialog
        from swe2d.workbench.views.studio_component_view import StudioComponent
        dock_parent = (
            iface.mainWindow()
            if iface is not None and hasattr(iface, "mainWindow")
            else dlg
        )
        if area is None:
            area = QtCore.Qt.RightDockWidgetArea
        dock = QtWidgets.QDockWidget(title, dock_parent)
        dock.setObjectName(f"HYDRA2D{name.title()}Dock")
        dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFloatable
            | QtWidgets.QDockWidget.DockWidgetClosable
        )
        if populate is not None:
            populate(dock)
        comp = StudioComponent(
            name=name,
            dock=dock,
            area=area,
            title=title,
            object_name=dock.objectName(),
            tab_with=tab_with,
        )
        self._register_component(comp)
        if iface is not None and hasattr(iface, "addDockWidget"):
            try:
                iface.addDockWidget(area, dock)
            except Exception as e:
                logger.warning("[ERROR] iface.addDockWidget failed for '%s': %s", name, e)
            try:
                dock.setFloating(False)
                dock.show()
            except Exception as e:
                logger.warning("[ERROR] dock show failed for '%s': %s", name, e)

    def _register_component(self, component):
        """Register a dock component for automated host-window extraction."""
        dlg = self._dialog
        existing = dlg._state.studio_components.get(component.name)
        if existing is not None:
            dlg._log(f"[Studio] Overwriting registered component '{component.name}'")
        dlg._state.studio_components[component.name] = component

    def _studio_sync_view_mode(self, idx: int) -> None:
        """Sync view mode from combo to viewer (stub, no-op)."""
        pass

    def _sync_view_mode_to_studio(self, idx: int) -> None:
        """Sync view mode to studio from combo (stub, no-op)."""
        pass

    def _studio_apply_visual_profile(self, profile: str) -> None:
        """Apply a visual stylesheet profile to the dialog."""
        dlg = self._dialog
        profile_key = str(profile or "").strip().lower()
        target = dlg
        if profile_key == "diagnostics":
            target.setStyleSheet(
                "QMainWindow { background: #1f232a; }"
                "QDockWidget::title { background: #2d3640; color: #f2f4f8; padding: 4px; }"
                "QToolBar { background: #2b3139; border-bottom: 1px solid #3a424c; }"
                "QStatusBar { background: #2b3139; color: #e7edf5; }"
            )
        elif profile_key == "presentation":
            target.setStyleSheet(
                "QMainWindow { background: #f2f5f8; }"
                "QDockWidget::title { background: #d9e2ec; color: #243b53; padding: 4px; }"
                "QToolBar { background: #e4ebf2; border-bottom: 1px solid #c9d4df; }"
                "QStatusBar { background: #e4ebf2; color: #243b53; }"
            )
        else:
            target.setStyleSheet("")

    def _populate_setup_dock(self, dock):
        """Populate the setup dock widget with the left pane."""
        from qgis.PyQt import QtWidgets
        dlg = self._dialog
        left_host = QtWidgets.QWidget()
        dlg._compose_left_pane(left_host)
        dock.setWidget(left_host)

    def _resolve_widget_attr(self, attr: str):
        """Resolve a widget by attribute name across all tab views + results toolbox."""
        dlg = self._dialog
        for tab_attr in ("_model_tab_view", "_map_tab_view", "_topology_tab_view",
                         "_mesh_tab_view", "_boundary_tab_view", "_results_toolbox"):
            tab = getattr(dlg, tab_attr, None)
            if tab is not None:
                w = getattr(tab, attr, None)
                if w is not None:
                    return w
        return None

    def _populate_inspector_dock(self, dock):
        """Populate the inspector dock with settings tree and help tabs."""
        from qgis.PyQt import QtWidgets
        from typing import Dict, List

        dlg = self._dialog
        inspector_tabs = QtWidgets.QTabWidget()
        inspector_tabs.setDocumentMode(True)

        model_page = QtWidgets.QWidget()
        model_layout = QtWidgets.QVBoxLayout(model_page)
        model_layout.setContentsMargins(6, 6, 6, 6)

        dlg._settings_tree = QtWidgets.QTreeWidget()
        dlg._settings_tree.setHeaderLabels(["Parameter", "Value"])
        dlg._settings_tree.setAlternatingRowColors(True)
        dlg._settings_tree.setAnimated(True)

        _groups = {
            "Solver": ["temporal_order_combo", "spatial_scheme_combo", "equation_set_combo",
                       "bed_friction_model_combo", "turbulence_model_combo"],
            "Time Stepping": ["cfl_spin", "dt_max_spin", "dt_fixed_spin", "dt_initial_spin"],
            "Physics": ["h_min_spin", "n_mann_spin", "k_mann_spin",
                        "shallow_damping_depth_spin", "max_rel_depth_increase_spin"],
            "Stability": ["momentum_cap_min_speed_spin", "momentum_cap_celerity_mult_spin",
                          "depth_cap_spin", "source_cfl_beta_spin"],
            "Mesh": ["max_inv_area_spin"],
        }
        for group, keys in _groups.items():
            grp_item = QtWidgets.QTreeWidgetItem([group, ""])
            grp_item.setExpanded(True)
            for key in keys:
                w = self._resolve_widget_attr(key)
                if w is not None:
                    try:
                        _ = w.objectName()
                    except RuntimeError:
                        w = None
                if w is not None:
                    try:
                        val = w.value() if hasattr(w, "value") else w.currentText()
                        lbl = w.toolTip() or key
                    except Exception:
                        val = "\u2014"
                        lbl = key
                    grp_item.addChild(QtWidgets.QTreeWidgetItem(
                        [str(lbl).replace("&", ""), str(val)]))
                else:
                    grp_item.addChild(QtWidgets.QTreeWidgetItem([key, "\u2014"]))
            dlg._settings_tree.addTopLevelItem(grp_item)

        run_item = QtWidgets.QTreeWidgetItem(["Runtime", ""])
        run_item.setExpanded(True)
        for key in ("output_interval_edit", "run_duration_edit",
                     "line_output_interval_edit", "n_thread_spin",
                     "source_max_substeps_spin", "gpu_diag_sync_interval_spin"):
            w = self._resolve_widget_attr(key)
            if w is not None:
                try:
                    _ = w.objectName()
                except RuntimeError:
                    w = None
            if w is not None:
                try:
                    val = w.value() if hasattr(w, "value") else str(w.text() or "")
                    lbl = w.toolTip() or key
                except Exception:
                    val = "\u2014"
                    lbl = key
                run_item.addChild(QtWidgets.QTreeWidgetItem(
                    [str(lbl).replace("&", ""), str(val)]))
            else:
                run_item.addChild(QtWidgets.QTreeWidgetItem([key, "\u2014"]))
        dlg._settings_tree.addTopLevelItem(run_item)

        model_layout.addWidget(dlg._settings_tree)
        inspector_tabs.addTab(model_page, "Model Settings")

        mesh_page = QtWidgets.QWidget()
        mesh_layout = QtWidgets.QVBoxLayout(mesh_page)
        mesh_layout.setContentsMargins(6, 6, 6, 6)

        dlg._mesh_settings_tree = QtWidgets.QTreeWidget()
        dlg._mesh_settings_tree.setHeaderLabels(["Parameter", "Value"])
        dlg._mesh_settings_tree.setAlternatingRowColors(True)
        dlg._mesh_settings_tree.setAnimated(True)

        _mesh_groups = {
            "Topology Layers": [
                "topo_nodes_combo", "topo_arcs_combo", "topo_regions_combo",
                "topo_constraints_combo", "topo_quad_edges_combo",
            ],
            "Mesh Generation": [
                "nx_spin", "ny_spin", "lx_spin", "ly_spin",
                "bed_amp_spin", "mesh_layout_combo", "mesh_info_lbl",
            ],
            "Topo Controls": [
                "topo_backend_combo", "topo_default_size_spin",
                "topo_default_cell_type_combo",
            ],
            "Gmsh Advanced": [
                "topo_gmsh_tri_algo_combo", "topo_gmsh_quad_algo_combo",
                "topo_gmsh_recombine_algo_combo",
                "topo_gmsh_global_recombine_chk",
                "topo_gmsh_quad_full_region_flow_align_chk",
                "topo_gmsh_smoothing_spin", "topo_gmsh_optimize_iters_spin",
                "topo_gmsh_verbosity_spin", "topo_gmsh_optimize_netgen_chk",
                "topo_gmsh_arc_mode_combo",
            ],
            "Quality Controls": [
                "topo_gmsh_quality_enable_chk", "topo_quality_min_angle_spin",
                "topo_quality_max_aspect_spin", "topo_quality_max_non_orth_spin",
                "topo_quality_size_scales_edit", "topo_quality_smooth_increments_edit",
                "topo_quality_strict_chk", "topo_gmsh_quality_max_iters_spin",
                "topo_gmsh_quality_time_limit_spin",
                "topo_gmsh_algo_switch_on_failure_chk",
                "topo_gmsh_recombine_node_repositioning_chk",
            ],
        }
        for group, keys in _mesh_groups.items():
            grp_item = QtWidgets.QTreeWidgetItem([group, ""])
            grp_item.setExpanded(True)
            for key in keys:
                w = self._resolve_widget_attr(key)
                if w is not None:
                    try:
                        _ = w.objectName()
                    except RuntimeError:
                        w = None
                if w is not None:
                    try:
                        val = w.value() if hasattr(w, "value") else w.currentText() if hasattr(w, "currentText") else str(w.text() or "")
                        val = str(val)
                        lbl = w.toolTip() or key
                    except Exception:
                        val = "\u2014"
                        lbl = key
                    grp_item.addChild(QtWidgets.QTreeWidgetItem(
                        [str(lbl).replace("&", ""), val]))
                else:
                    grp_item.addChild(QtWidgets.QTreeWidgetItem([key, "\u2014"]))
            dlg._mesh_settings_tree.addTopLevelItem(grp_item)

        mesh_layout.addWidget(dlg._mesh_settings_tree)
        inspector_tabs.addTab(mesh_page, "Mesh Settings")

        from swe2d.workbench.views.doc_viewer import DocHubWidget
        inspector_tabs.addTab(DocHubWidget(parent=dlg), "Help")

        dock.setWidget(inspector_tabs)

    def _populate_log_dock(self, dock):
        """Populate the log dock widget with a read-only text view."""
        from qgis.PyQt import QtWidgets
        dlg = self._dialog
        log_view = QtWidgets.QPlainTextEdit()
        log_view.setObjectName("log_view")
        log_view.setReadOnly(True)
        dlg.log_view = log_view
        dock.setWidget(log_view)

    def _find_widget(self, wtype, name):
        """Search for a widget in the dialog and QGIS host window."""
        from qgis.PyQt import QtWidgets
        dlg = self._dialog
        w = dlg.findChild(wtype, name)
        if w is None and dlg.iface is not None:
            try:
                mw = dlg.iface.mainWindow()
                if mw is not None:
                    w = mw.findChild(wtype, name)
            except Exception as _e:

                logger.warning(f"[ERROR] Exception in workbench_dialog_builder.py: {_e}")
        return w

    def _validate_widget_bindings(self) -> None:
        """Check that critical widgets have Python bindings."""
        from qgis.PyQt import QtWidgets
        from typing import List

        dlg = self._dialog
        critical = {"run_btn": QtWidgets.QPushButton}
        optional = {
            "cfl_spin": QtWidgets.QDoubleSpinBox,
            "dt_spin": QtWidgets.QDoubleSpinBox,
            "n_mann_spin": QtWidgets.QDoubleSpinBox,
            "snapshot_btn": QtWidgets.QPushButton,
        }
        missing_optional: List[str] = []

        for name, wtype in critical.items():
            w = self._find_widget(wtype, name)
            if w is None:
                raise RuntimeError(
                    f"Critical widget '{name}' ({wtype.__name__}) has no Python "
                    f"binding in {type(dlg).__name__}. Check that the widget "
                    f"objectName is correct and the .ui file is loaded."
                )

        for name, wtype in optional.items():
            w = self._find_widget(wtype, name)
            if w is None:
                missing_optional.append(name)

        if missing_optional:
            dlg._log(
                f"[Studio] Optional widgets missing bindings: {', '.join(missing_optional)}"
            )
