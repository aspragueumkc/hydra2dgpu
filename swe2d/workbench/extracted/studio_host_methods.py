from __future__ import annotations

# Extracted helpers depend on symbols defined in swe2d_workbench_qt.
from swe2d_workbench_qt import *  # type: ignore F401,F403
from swe2d_workbench_qt import (
    _SWE2D_STUDIO_COMPONENT_DOCKS,
    _SWE2D_STUDIO_HOST_DIALOG,
    _SWE2D_STUDIO_HOST_MENU,
    _SWE2D_STUDIO_HOST_TOOLBAR,
    _SWE2D_WORKBENCH_STUDIO_DOCK,
    _remove_workbench_dock_instance as _base_remove_workbench_dock_instance,
)

def _remove_workbench_studio_dock(iface_obj) -> None:
    global _SWE2D_WORKBENCH_STUDIO_DOCK, _SWE2D_STUDIO_COMPONENT_DOCKS, _SWE2D_STUDIO_HOST_DIALOG
    seen = set()

    for dock in [_SWE2D_WORKBENCH_STUDIO_DOCK] + list(_SWE2D_STUDIO_COMPONENT_DOCKS.values()):
        if dock is None:
            continue
        key = id(dock)
        if key in seen:
            continue
        seen.add(key)
        _base_remove_workbench_dock_instance(dock, iface_obj)

    _SWE2D_WORKBENCH_STUDIO_DOCK = None
    _SWE2D_STUDIO_COMPONENT_DOCKS = {}

    if _SWE2D_STUDIO_HOST_DIALOG is not None:
        try:
            _SWE2D_STUDIO_HOST_DIALOG.close()
        except Exception:
            pass
        try:
            _SWE2D_STUDIO_HOST_DIALOG.deleteLater()
        except Exception:
            pass
        _SWE2D_STUDIO_HOST_DIALOG = None

    _clear_studio_host_controls(iface_obj)




def _attach_host_dock_widget(iface_obj, host_window, dock: QtWidgets.QDockWidget, area) -> bool:
    attached = False
    try:
        if iface_obj is not None and hasattr(iface_obj, "addDockWidget"):
            iface_obj.addDockWidget(area, dock)
            attached = True
    except Exception:
        attached = False
    if not attached:
        try:
            if host_window is not None and hasattr(host_window, "addDockWidget"):
                host_window.addDockWidget(area, dock)
                attached = True
        except Exception:
            attached = False
    if not attached:
        try:
            dock.show()
        except Exception:
            pass
        return False
    try:
        dock.setFloating(False)
    except Exception:
        pass
    try:
        dock.show()
        dock.raise_()
    except Exception:
        pass
    return True




def _studio_take_dock_widget(studio_dock, fallback_text: str) -> QtWidgets.QWidget:
    widget = None
    try:
        widget = studio_dock.widget() if studio_dock is not None else None
    except Exception:
        widget = None

    if widget is None:
        fallback = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(fallback)
        lay.setContentsMargins(8, 8, 8, 8)
        lbl = QtWidgets.QLabel(fallback_text)
        lbl.setWordWrap(True)
        lay.addWidget(lbl)
        lay.addStretch(1)
        return fallback

    try:
        if studio_dock is not None:
            studio_dock.setWidget(QtWidgets.QWidget())
    except Exception:
        pass
    try:
        widget.setParent(None)
    except Exception:
        pass
    return widget




def _build_studio_component_docks(iface_obj, host_window, dlg) -> Dict[str, QtWidgets.QDockWidget]:
    component_docks: Dict[str, QtWidgets.QDockWidget] = {}

    setup_widget = _studio_take_dock_widget(
        getattr(dlg, "_studio_left_dock", None),
        "Model Setup panel is unavailable.",
    )
    inspector_widget = _studio_take_dock_widget(
        getattr(dlg, "_studio_inspector_dock", None),
        "CFD Inspector panel is unavailable.",
    )

    view_widget = None
    log_widget = getattr(dlg, "log_view", None)
    split = getattr(dlg, "_right_vertical_split", None)
    if split is not None and hasattr(split, "widget"):
        try:
            if split.count() > 0:
                view_widget = split.widget(0)
        except Exception:
            view_widget = None

    if view_widget is None:
        try:
            mw = getattr(dlg, "_studio_main_window", None)
            if mw is not None:
                view_widget = mw.centralWidget()
        except Exception:
            view_widget = None

    if view_widget is None:
        fallback = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(fallback)
        lay.setContentsMargins(8, 8, 8, 8)
        lbl = QtWidgets.QLabel("View panel is unavailable.")
        lbl.setWordWrap(True)
        lay.addWidget(lbl)
        lay.addStretch(1)
        view_widget = fallback

    if log_widget is None:
        log_widget = QtWidgets.QPlainTextEdit()
        log_widget.setReadOnly(True)
        log_widget.setPlainText("Runtime log panel initialized.")

    for w in (view_widget, log_widget):
        try:
            w.setParent(None)
        except Exception:
            pass

    def _mkdock(title: str, obj_name: str, widget: QtWidgets.QWidget) -> QtWidgets.QDockWidget:
        dock = QtWidgets.QDockWidget(title, host_window)
        dock.setObjectName(obj_name)
        dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFloatable
            | QtWidgets.QDockWidget.DockWidgetClosable,
        )
        dock.setWidget(widget)
        return dock

    component_docks["setup"] = _mkdock(
        "SWE2D Studio - Model Setup",
        "SWE2DStudioSetupHostDock",
        setup_widget,
    )
    component_docks["view"] = _mkdock(
        "SWE2D Studio - View",
        "SWE2DStudioViewHostDock",
        view_widget,
    )
    component_docks["log"] = _mkdock(
        "SWE2D Studio - Runtime Log",
        "SWE2DStudioLogHostDock",
        log_widget,
    )
    component_docks["inspector"] = _mkdock(
        "SWE2D Studio - CFD Inspector",
        "SWE2DStudioInspectorHostDock",
        inspector_widget,
    )

    _attach_host_dock_widget(iface_obj, host_window, component_docks["setup"], QtCore.Qt.LeftDockWidgetArea)
    _attach_host_dock_widget(iface_obj, host_window, component_docks["view"], QtCore.Qt.RightDockWidgetArea)
    _attach_host_dock_widget(iface_obj, host_window, component_docks["inspector"], QtCore.Qt.RightDockWidgetArea)
    _attach_host_dock_widget(iface_obj, host_window, component_docks["log"], QtCore.Qt.BottomDockWidgetArea)

    try:
        if host_window is not None and hasattr(host_window, "tabifyDockWidget"):
            host_window.tabifyDockWidget(component_docks["view"], component_docks["inspector"])
    except Exception:
        pass

    return component_docks




def _studio_host_main_window(iface_obj, fallback_parent=None):
    host_window = None
    if iface_obj is not None and hasattr(iface_obj, "mainWindow"):
        try:
            host_window = iface_obj.mainWindow()
        except Exception:
            host_window = None
    if host_window is None:
        host_window = fallback_parent
    return host_window




def _clear_studio_host_controls(iface_obj, fallback_parent=None) -> None:
    global _SWE2D_STUDIO_HOST_TOOLBAR, _SWE2D_STUDIO_HOST_MENU

    host_window = _studio_host_main_window(iface_obj, fallback_parent)

    if _SWE2D_STUDIO_HOST_TOOLBAR is not None:
        try:
            if iface_obj is not None and hasattr(iface_obj, "mainWindow") and host_window is not None:
                host_window.removeToolBar(_SWE2D_STUDIO_HOST_TOOLBAR)
        except Exception:
            pass
        try:
            _SWE2D_STUDIO_HOST_TOOLBAR.deleteLater()
        except Exception:
            pass
        _SWE2D_STUDIO_HOST_TOOLBAR = None

    if _SWE2D_STUDIO_HOST_MENU is not None:
        try:
            act = _SWE2D_STUDIO_HOST_MENU.menuAction()
            parent = act.parentWidget()
            if parent is not None:
                parent.removeAction(act)
        except Exception:
            pass
        try:
            _SWE2D_STUDIO_HOST_MENU.deleteLater()
        except Exception:
            pass
        _SWE2D_STUDIO_HOST_MENU = None




def _install_studio_host_controls(
    iface_obj,
    dlg,
    fallback_parent=None,
    component_docks: Optional[Dict[str, QtWidgets.QDockWidget]] = None,
) -> None:
    global _SWE2D_STUDIO_HOST_TOOLBAR, _SWE2D_STUDIO_HOST_MENU

    host_window = _studio_host_main_window(iface_obj, fallback_parent)
    if host_window is None:
        return

    _clear_studio_host_controls(iface_obj, fallback_parent)
    component_docks = dict(component_docks or {})

    def _focus_panel(name: str) -> None:
        dock = component_docks.get(str(name or "").strip().lower())
        if dock is None:
            return
        try:
            dock.show()
            dock.raise_()
        except Exception:
            pass

    def _close_studio_panels() -> None:
        try:
            _remove_workbench_studio_dock(iface_obj)
        except Exception:
            pass

    menu_bar = None
    try:
        menu_bar = host_window.menuBar()
    except Exception:
        menu_bar = None

    if menu_bar is not None:
        menu = QtWidgets.QMenu("SWE2D Studio", menu_bar)
        menu.setObjectName("SWE2DStudioHostMenu")

        rainfall_act = menu.addAction("Enable Rainfall")
        rainfall_act.setCheckable(True)
        rainfall_act.setChecked(True)

        drainage_act = menu.addAction("Enable Drainage")
        drainage_act.setCheckable(True)
        drainage_act.setChecked(True)

        structures_act = menu.addAction("Enable Structures")
        structures_act.setCheckable(True)
        structures_act.setChecked(True)

        menu.addSeparator()
        menu.addAction("Focus Mesh", lambda: dlg._studio_select_tab("mesh"))
        menu.addAction("Focus Model", lambda: dlg._studio_select_tab("model"))
        menu.addAction("Focus Run", lambda: dlg._studio_select_tab("run"))
        menu.addAction("Focus Map", lambda: dlg._studio_select_tab("map"))

        if component_docks:
            menu.addSeparator()
            menu.addAction("Show Model Setup Panel", lambda: _focus_panel("setup"))
            menu.addAction("Show View Panel", lambda: _focus_panel("view"))
            menu.addAction("Show Runtime Log Panel", lambda: _focus_panel("log"))
            menu.addAction("Show CFD Inspector Panel", lambda: _focus_panel("inspector"))

        menu.addSeparator()
        menu.addAction("Close Studio Panels", _close_studio_panels)

        rainfall_act.toggled.connect(lambda checked: dlg._studio_set_feature_enabled("rainfall", checked))
        drainage_act.toggled.connect(lambda checked: dlg._studio_set_feature_enabled("drainage", checked))
        structures_act.toggled.connect(lambda checked: dlg._studio_set_feature_enabled("structures", checked))

        menu_bar.addMenu(menu)
        _SWE2D_STUDIO_HOST_MENU = menu

    toolbar = QtWidgets.QToolBar("SWE2D Studio", host_window)
    toolbar.setObjectName("SWE2DStudioHostToolbar")
    toolbar.setMovable(True)
    toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)

    if component_docks:
        show_setup = toolbar.addAction("Setup Panel")
        show_view = toolbar.addAction("View Panel")
        show_log = toolbar.addAction("Log Panel")
        show_inspector = toolbar.addAction("Inspector Panel")
        show_setup.triggered.connect(lambda: _focus_panel("setup"))
        show_view.triggered.connect(lambda: _focus_panel("view"))
        show_log.triggered.connect(lambda: _focus_panel("log"))
        show_inspector.triggered.connect(lambda: _focus_panel("inspector"))
        toolbar.addSeparator()

    act_mesh = toolbar.addAction("Mesh")
    act_model = toolbar.addAction("Model")
    act_run = toolbar.addAction("Run")
    act_map = toolbar.addAction("Map")
    toolbar.addSeparator()

    act_refresh = toolbar.addAction("Refresh")
    act_snapshot = toolbar.addAction("Snapshot")
    toolbar.addSeparator()

    rainfall_tb = toolbar.addAction("Rainfall")
    rainfall_tb.setCheckable(True)
    rainfall_tb.setChecked(True)

    drainage_tb = toolbar.addAction("Drainage")
    drainage_tb.setCheckable(True)
    drainage_tb.setChecked(True)

    structures_tb = toolbar.addAction("Structures")
    structures_tb.setCheckable(True)
    structures_tb.setChecked(True)

    act_mesh.triggered.connect(lambda: dlg._studio_select_tab("mesh"))
    act_model.triggered.connect(lambda: dlg._studio_select_tab("model"))
    act_run.triggered.connect(lambda: dlg._studio_select_tab("run"))
    act_map.triggered.connect(lambda: dlg._studio_select_tab("map"))
    act_refresh.triggered.connect(dlg._refresh_layer_combos)
    act_snapshot.triggered.connect(
        lambda: dlg.snapshot_btn.click()
        if hasattr(dlg, "snapshot_btn") and dlg.snapshot_btn is not None
        else None
    )

    rainfall_tb.toggled.connect(lambda checked: dlg._studio_set_feature_enabled("rainfall", checked))
    drainage_tb.toggled.connect(lambda checked: dlg._studio_set_feature_enabled("drainage", checked))
    structures_tb.toggled.connect(lambda checked: dlg._studio_set_feature_enabled("structures", checked))

    toolbar.addSeparator()
    toolbar.addWidget(QtWidgets.QLabel(" View: "))
    host_view_combo = QtWidgets.QComboBox(toolbar)
    host_view_combo.addItems(["Mesh", "Depth", "Velocity magnitude", "Runtime Log"])
    try:
        source_idx = int(getattr(dlg, "view_mode_combo", host_view_combo).currentIndex())
        host_view_combo.setCurrentIndex(max(0, min(source_idx, host_view_combo.count() - 1)))
    except Exception:
        pass
    host_view_combo.currentIndexChanged.connect(
        lambda idx: dlg.view_mode_combo.setCurrentIndex(idx)
        if hasattr(dlg, "view_mode_combo") and dlg.view_mode_combo is not None
        else None
    )
    toolbar.addWidget(host_view_combo)

    toolbar.addWidget(QtWidgets.QLabel(" Theme: "))
    host_theme_combo = QtWidgets.QComboBox(toolbar)
    host_theme_combo.addItems(["Default", "Diagnostics", "Presentation"])
    host_theme_combo.currentTextChanged.connect(dlg._studio_apply_visual_profile)
    toolbar.addWidget(host_theme_combo)

    toolbar.addSeparator()
    act_close = toolbar.addAction("Close Studio")
    act_close.triggered.connect(_close_studio_panels)

    try:
        host_window.addToolBar(QtCore.Qt.TopToolBarArea, toolbar)
        _SWE2D_STUDIO_HOST_TOOLBAR = toolbar
    except Exception:
        try:
            toolbar.deleteLater()
        except Exception:
            pass





def enforce_studio_shell_visible(dlg: "SWE2DWorkbenchStudioDialog") -> None:
    try:
        mw = getattr(dlg, "_studio_main_window", None)
        if mw is None:
            return
        try:
            if mw.isWindow():
                mw.setWindowFlags(QtCore.Qt.Widget)
                mw.setParent(dlg)
        except Exception:
            pass
        center = mw.centralWidget()
        if center is None:
            fallback = QtWidgets.QWidget()
            lay = QtWidgets.QVBoxLayout(fallback)
            lay.setContentsMargins(12, 12, 12, 12)
            msg = QtWidgets.QLabel(
                "Studio workspace recovered from an invalid layout state.\n"
                "Use the left Model Setup dock to continue."
            )
            msg.setWordWrap(True)
            lay.addWidget(msg)
            lay.addStretch(1)
            mw.setCentralWidget(fallback)
            center = fallback
        try:
            center.show()
        except Exception:
            pass
        left_dock = getattr(dlg, "_studio_left_dock", None)
        if left_dock is not None and not left_dock.isVisible():
            left_dock.show()
        inspector_dock = getattr(dlg, "_studio_inspector_dock", None)
        if inspector_dock is not None and not inspector_dock.isVisible():
            inspector_dock.show()
    except Exception:
        pass
