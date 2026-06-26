"""Tab page builders and signal wiring for the Studio dialog.

Extracted from SWE2DWorkbenchStudioDialog. Each function takes the dialog
as the first parameter — this module is stateless.
"""

from typing import Any

from qgis.PyQt import QtCore, QtGui, QtWidgets


def build_map_tab(dialog) -> QtWidgets.QWidget:
    """Build the Map tab page and wrap it in a scroll area."""
    page, data_layout, actions_layout, tools_layout = dialog._build_map_tab_page()
    return wrap_left_tab_page(dialog, page)


def build_topology_tab(dialog) -> QtWidgets.QWidget:
    """Build the Topology tab page and wrap it in a scroll area."""
    page = dialog._build_topology_tab_page()
    return wrap_left_tab_page(dialog, page)


def build_model_tab(dialog) -> QtWidgets.QWidget:
    """Build the Model tab page and wrap it in a scroll area."""
    page, solver_form, rain_form, drain_form, run_page = dialog._build_model_tab_page()
    return wrap_left_tab_page(dialog, page)


def compose_left_pane(dialog, left_host: QtWidgets.QWidget) -> QtWidgets.QWidget:
    """Build the left pane tab widget (Layers/Mesh/Parameters) and embed in the host."""
    left = left_host
    left_layout = left.layout()
    if not isinstance(left_layout, QtWidgets.QVBoxLayout):
        left_layout = QtWidgets.QVBoxLayout(left)
    left_layout.setContentsMargins(0, 0, 0, 0)
    left_layout.setSpacing(4)
    while left_layout.count():
        item = left_layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()
    dialog._left_tabs = QtWidgets.QTabWidget()
    dialog._left_tabs.setDocumentMode(True)
    left_layout.addWidget(dialog._left_tabs, stretch=1)
    dialog._left_tabs.addTab(build_map_tab(dialog), "Layers")
    dialog._left_tabs.addTab(build_topology_tab(dialog), "Mesh")
    dialog._left_tabs.addTab(build_model_tab(dialog), "Parameters")
    left.setMinimumWidth(0)
    for _cb in left.findChildren(QtWidgets.QComboBox):
        _cb.setMinimumContentsLength(0)
        _cb.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
    for _btn in left.findChildren(QtWidgets.QPushButton):
        _btn.setMinimumWidth(0)
    for _sp in left.findChildren(
        (QtWidgets.QDoubleSpinBox, QtWidgets.QSpinBox)
    ):
        _sp.setMinimumWidth(0)
    make_left_controls_compact(dialog, left)
    register_detachable_tab_widget(dialog, dialog._left_tabs)
    return left


# ── Map tab ──────────────────────────────────────────────────────────────────


def build_map_tab_page(dialog):
    """Build the Map tab page view and wire signal handlers."""
    from swe2d.workbench.views.map_tab_view import MapTabView
    dialog._map_tab_view = MapTabView()
    map_tab_page = dialog._map_tab_view
    map_data_layout = map_tab_page.findChild(QtWidgets.QGridLayout, "map_data_layout")
    map_actions_layout = map_tab_page.findChild(QtWidgets.QGridLayout, "map_actions_layout")
    map_tools_layout = map_tab_page.findChild(QtWidgets.QGridLayout, "map_tools_layout")
    if map_data_layout is None or map_actions_layout is None or map_tools_layout is None:
        raise RuntimeError("Map tab UI missing one or more expected group layouts")
    wire_map_tab_data_signals(dialog)
    wire_map_tab_action_signals(dialog)
    wire_map_tab_tools_signals(dialog)
    return map_tab_page, map_data_layout, map_actions_layout, map_tools_layout


def wire_map_tab_data_signals(dialog) -> None:
    """Wire the Map tab Data page button signals to the controller."""
    from swe2d.workbench.signal_helpers import safe_disconnect
    v = dialog._map_tab_view
    safe_disconnect(v.autopop_group_btn.clicked, dialog._workbench_controller.autopopulate_layer_combos_from_group)
    v.autopop_group_btn.clicked.connect(dialog._workbench_controller.autopopulate_layer_combos_from_group)
    safe_disconnect(v.refresh_layers_btn.clicked, dialog._workbench_controller.refresh_layer_combos)
    v.refresh_layers_btn.clicked.connect(dialog._workbench_controller.refresh_layer_combos)
    safe_disconnect(v.create_model_gpkg_btn.clicked, dialog._workbench_controller.create_2d_model_geopackage)
    v.create_model_gpkg_btn.clicked.connect(dialog._workbench_controller.create_2d_model_geopackage)


def wire_map_tab_action_signals(dialog) -> None:
    """Wire the Map tab Actions page button signals to the dialog handlers."""
    from swe2d.workbench.signal_helpers import safe_disconnect
    v = dialog._map_tab_view
    handlers = {
        "load_model_gpkg_btn": (v.load_model_gpkg_btn, dialog._load_2d_model_geopackage),
        "export_mesh_layers_btn": (v.export_mesh_layers_btn, dialog._export_mesh_to_layers),
        "export_mesh_ugrid_btn": (v.export_mesh_ugrid_btn, dialog._export_mesh_to_ugrid),
        "save_mesh_gpkg_btn": (v.save_mesh_gpkg_btn, dialog._save_mesh_to_gpkg),
        "import_mesh_layers_btn": (v.import_mesh_layers_btn, dialog._workbench_controller.import_mesh_from_layers),
        "load_mesh_gpkg_btn": (v.load_mesh_gpkg_btn, dialog._load_mesh_from_gpkg),
        "terrain_to_nodes_btn": (v.terrain_to_nodes_btn, dialog._assign_node_z_from_terrain),
        "pull_node_z_btn": (v.pull_node_z_btn, dialog._pull_node_z_from_layer),
        "export_results_ugrid_btn": (v.export_results_ugrid_btn, dialog._export_results_to_ugrid),
    }
    for attr, (btn, cb) in handlers.items():
        safe_disconnect(btn.clicked, cb)
        btn.clicked.connect(cb)


def wire_map_tab_tools_signals(dialog) -> None:
    """Wire the Map tab Utilities page button signals to the dialog handlers."""
    from swe2d.workbench.signal_helpers import safe_disconnect
    v = dialog._map_tab_view
    handlers = {
        "open_model_gpkg_explorer_btn": (v.open_model_gpkg_explorer_btn, dialog._open_model_gpkg_explorer),
        "open_run_log_viewer_btn": (v.open_run_log_viewer_btn, dialog._open_run_log_viewer),
    }
    for attr, (btn, cb) in handlers.items():
        safe_disconnect(btn.clicked, cb)
        btn.clicked.connect(cb)


# ── Topology tab ─────────────────────────────────────────────────────────────


def build_topology_tab_page(dialog) -> QtWidgets.QWidget:
    """Build the Topology tab page view and wire static signal handlers."""
    from swe2d.workbench.views.topology_tab_view import TopologyTabView
    dialog._topology_tab_view = TopologyTabView()
    dialog._topology_tab_view.view = dialog
    dialog._topology_tab_view.set_callbacks(log_fn=dialog._log, combo_layer_fn=dialog._combo_layer)
    topology_tab_page = dialog._topology_tab_view
    wire_topology_tab_static_signals(dialog)
    return topology_tab_page


def wire_topology_tab_static_signals(dialog) -> None:
    """Wire the Topology tab static button signals to the dialog handlers."""
    from swe2d.workbench.signal_helpers import safe_disconnect
    v = dialog._topology_tab_view
    handlers = {
        "topo_export_template_btn": (v.topo_export_template_btn, dialog._create_topology_template_layers),
        "topo_generate_btn": (v.topo_generate_btn, dialog._generate_mesh_from_topology_layers),
        "topo_terminate_btn": (v.topo_terminate_btn, dialog._on_terminate_topology_mesh),
    }
    for attr, (btn, cb) in handlers.items():
        safe_disconnect(btn.clicked, cb)
        btn.clicked.connect(cb)


# ── Model tab ────────────────────────────────────────────────────────────────


def build_model_tab_page(dialog):
    """Build the Model tab page view and wire run signal handlers."""
    from swe2d.workbench.views.model_tab_view import ModelTabView
    dialog._model_tab_view = ModelTabView()
    model_tab_page = dialog._model_tab_view
    solver_form = model_tab_page.findChild(QtWidgets.QFormLayout, "model_solver_form")
    rain_form = model_tab_page.findChild(QtWidgets.QFormLayout, "model_rain_form")
    drain_form = model_tab_page.findChild(QtWidgets.QFormLayout, "model_drain_form")
    run_page = model_tab_page.findChild(QtWidgets.QWidget, "model_run_page")
    if solver_form is None or rain_form is None or drain_form is None:
        raise RuntimeError("Model tab UI missing one or more form layouts")
    wire_run_tab_signals(dialog)
    return model_tab_page, solver_form, rain_form, drain_form, run_page


def wire_run_tab_signals(dialog) -> None:
    """Wire the Model tab Run page button signals to the controller."""
    from swe2d.workbench.signal_helpers import safe_disconnect
    v = dialog._model_tab_view
    handlers = [
        (v.run_btn, dialog._controller.on_run),
        (v.batch_sim_btn, dialog._open_batch_simulation_dialog),
        (v.cancel_btn, dialog._controller.on_cancel),
        (v.preview_overrides_btn, dialog._controller.on_preview_overrides),
        (v.preview_coupling_btn, dialog._controller.on_preview_coupling),
        (v.snapshot_btn, dialog._controller.on_snapshot),
    ]
    for btn, cb in handlers:
        safe_disconnect(btn.clicked, cb)
        btn.clicked.connect(cb)
    load_btn = getattr(v, "load_run_settings_btn", None)
    if load_btn is not None:
        safe_disconnect(load_btn.clicked, dialog._controller.on_load_run_settings_from_results)
        load_btn.clicked.connect(dialog._controller.on_load_run_settings_from_results)


# ── Tab lifecycle ────────────────────────────────────────────────────────────


def wrap_left_tab_page(dialog, widget):
    """Wrap a tab page widget in a scroll area with no frame."""
    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    scroll.setWidget(widget)
    return scroll


def expand_toolbox_pages(dialog, toolbox):
    """Set all toolbox pages to the Expanding size policy."""
    for i in range(toolbox.count()):
        page = toolbox.widget(i)
        if page is None:
            continue
        page.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding
        )


def make_left_controls_compact(dialog, parent_widget):
    """Reduce margins and spacing for a compact left pane layout."""
    for layout in parent_widget.findChildren(QtWidgets.QLayout):
        try:
            layout.setContentsMargins(4, 4, 4, 4)
        except Exception:
            pass
        try:
            if hasattr(layout, "setSpacing"):
                layout.setSpacing(4)
        except Exception:
            pass
        if isinstance(layout, QtWidgets.QFormLayout):
            try:
                layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
                layout.setHorizontalSpacing(6)
                layout.setVerticalSpacing(4)
            except Exception:
                pass


def register_detachable_tab_widget(dialog, tab_widget):
    """Enable tab detaching via custom context menu on the tab bar."""
    if tab_widget is None:
        return
    tab_widget.tabBar().setMovable(True)
    tab_widget.tabBar().setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
    tab_widget.tabBar().customContextMenuRequested.connect(
        lambda pos, tw=tab_widget: dialog._show_tab_detach_menu(tw, pos)
    )


# ── TopologyMeshView protocol helpers ────────────────────────────────────────
