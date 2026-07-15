"""Tab page builders and signal wiring for the Studio dialog.

Extracted from SWE2DWorkbenchStudioDialog. Each function takes the dialog
as the first parameter — this module is stateless.
"""

from typing import Any

import logging

from qgis.PyQt import QtCore, QtGui, QtWidgets

logger = logging.getLogger(__name__)


def build_topology_tab(dialog) -> QtWidgets.QWidget:
    """Build the Topology tab page and wrap it in a scroll area."""
    page = build_topology_tab_page(dialog)
    return wrap_left_tab_page(dialog, page)


def build_model_tab(dialog) -> QtWidgets.QWidget:
    """Build the Model tab page and wrap it in a scroll area."""
    page, _solver_form, _rain_form, _drain_form, _run_page = build_model_tab_page(dialog)
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
    dialog._left_tabs.addTab(build_topology_tab(dialog), "Mesh Generation")
    dialog._left_tabs.addTab(build_model_tab(dialog), "Simulation")
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
    """Wire the Topology tab static button signals to the dialog handlers.

    Includes the mesh I/O buttons that live on the Topology tab's
    Import/Export page (moved from the Map tab's Mesh Setup page).
    """
    from swe2d.workbench.signal_helpers import safe_disconnect
    v = dialog._topology_tab_view
    handlers = {
        "topo_generate_btn": (v.topo_generate_btn, dialog._topology_controller.generate_mesh_from_topology_layers),
        "topo_terminate_btn": (v.topo_terminate_btn, dialog._topology_controller.on_terminate_topology_mesh),
        # Mesh I/O buttons — moved from Map tab to the Import/Export page
        "export_mesh_layers_btn": (v.export_mesh_layers_btn, dialog._mesh_controller.export_mesh_to_layers),
        "export_mesh_ugrid_btn": (v.export_mesh_ugrid_btn, dialog._mesh_controller.export_mesh_to_ugrid),
        "save_mesh_gpkg_btn": (v.save_mesh_gpkg_btn, dialog._save_mesh_to_gpkg),
        "import_mesh_layers_btn": (v.import_mesh_layers_btn, dialog._mesh_controller.import_mesh_from_layers),
        "load_mesh_gpkg_btn": (v.load_mesh_gpkg_btn, dialog._load_mesh_from_gpkg),
        "export_results_ugrid_btn": (v.export_results_ugrid_btn, dialog._mesh_controller.export_results_to_ugrid),
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
    if solver_form is None or rain_form is None or drain_form is None:
        raise RuntimeError("Model tab UI missing one or more form layouts")
    wire_model_tab_layers_signals(dialog)
    wire_run_tab_signals(dialog)
    return model_tab_page, solver_form, rain_form, drain_form, None


def wire_run_dock_signals(dialog) -> None:
    """Wire the Run dock buttons to controller handlers.

    The Run dock only owns execution-surface buttons now (Run / Cancel /
    Snapshot / Batch). Output-config widgets (Preview / Load / Save /
    Browse GPKG) moved to the Simulation tab's Output page — see
    :func:`wire_run_tab_signals`.
    """
    from swe2d.workbench.signal_helpers import safe_disconnect
    d = dialog._run_dock
    safe_disconnect(d.run_btn.clicked, dialog._controller.on_run)
    d.run_btn.clicked.connect(dialog._controller.on_run)
    safe_disconnect(d.cancel_btn.clicked, dialog._controller.on_cancel)
    d.cancel_btn.clicked.connect(dialog._controller.on_cancel)
    safe_disconnect(d.snapshot_btn.clicked, dialog._controller.on_snapshot)
    d.snapshot_btn.clicked.connect(dialog._controller.on_snapshot)
    safe_disconnect(d.batch_btn.clicked, dialog._controller.open_batch_simulation_dialog)
    d.batch_btn.clicked.connect(dialog._controller.open_batch_simulation_dialog)


def wire_model_tab_layers_signals(dialog) -> None:
    """Wire the Simulation tab Layers page combos to auto-refresh.

    The 14 layer combos (nodes, cells, terrain, manning, CN, rain gages,
    hyetographs, sample lines, drainage nodes/links/inlets, structures,
    BC lines) now live on the Simulation tab's "Layers" page. They
    auto-refresh when the user changes the active layer so the project
    layer list stays current.
    """
    from swe2d.workbench.signal_helpers import safe_disconnect
    v = dialog._model_tab_view
    lc = dialog._layer_controller

    def _on_combo_changed() -> None:
        lc.refresh_layer_combos()

    for attr in (
        "manning_layer_combo",
        "cn_layer_combo",
        "rain_gage_layer_combo",
        "hyetograph_layer_combo",
        "sample_lines_layer_combo",
        "drain_nodes_layer_combo",
        "drain_links_layer_combo",
        "drain_inlets_layer_combo",
        "drain_node_inlets_layer_combo",
        "structures_layer_combo",
        "bc_lines_layer_combo",
        "storm_area_layer_combo",
    ):
        combo = getattr(v, attr, None)
        if combo is not None:
            safe_disconnect(combo.currentIndexChanged, _on_combo_changed)
            combo.currentIndexChanged.connect(_on_combo_changed)


def wire_run_tab_signals(dialog) -> None:
    """Wire the moved output-config buttons on the Simulation tab.

    The output interval / line interval / results GPKG / preview / load /
    save widgets live on the Model tab view's Output page now (moved
    from below the Run dock progress bar).
    """
    from swe2d.workbench.signal_helpers import safe_disconnect
    v = dialog._model_tab_view
    handlers = {
        "load_run_settings_btn": (v.load_run_settings_btn, dialog._controller.on_load_simulation_config),
        "save_settings_btn": (v.save_settings_btn, dialog._controller.on_save_simulation_config),
        "select_results_gpkg_btn": (v.select_results_gpkg_btn, dialog._mesh_controller.on_select_results_gpkg),
    }
    for attr, (btn, cb) in handlers.items():
        safe_disconnect(btn.clicked, cb)
        btn.clicked.connect(cb)


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


def _size_button(btn: QtWidgets.QPushButton, role: str = "action") -> None:
    """Apply a canonical size to a button based on its role."""
    if role == "icon":
        btn.setFixedSize(24, 24)
    elif role == "primary":
        btn.setMinimumSize(100, 32)
        font = btn.font()
        font.setBold(True)
        btn.setFont(font)
    else:
        btn.setMinimumSize(80, 28)


def make_left_controls_compact(dialog, parent_widget):
    """Reduce margins and spacing for a compact left pane layout."""
    for layout in parent_widget.findChildren(QtWidgets.QLayout):
        try:
            layout.setContentsMargins(4, 4, 4, 4)
        except Exception as _e:

            logger.warning(f"[ERROR] Exception in studio_tab_builder.py: {_e}")
        try:
            if hasattr(layout, "setSpacing"):
                layout.setSpacing(4)
        except Exception as _e:

            logger.warning(f"[ERROR] Exception in studio_tab_builder.py: {_e}")
        if isinstance(layout, QtWidgets.QFormLayout):
            try:
                layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
                layout.setHorizontalSpacing(6)
                layout.setVerticalSpacing(4)
            except Exception as _e:

                logger.warning(f"[ERROR] Exception in studio_tab_builder.py: {_e}")

    for btn in parent_widget.findChildren(QtWidgets.QPushButton):
        try:
            _size_button(btn, "action")
        except Exception as _e:
            logger.warning(f"[ERROR] Exception in studio_tab_builder.py: {_e}")


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
