from __future__ import annotations

# Extracted methods depend on symbols defined in swe2d_workbench_qt.
from swe2d_workbench_qt import *  # type: ignore F401,F403
from swe2d_workbench_qt import _HAVE_QGIS_CORE

def designer_populate_left_tabs(self, shell: QtWidgets.QWidget) -> None:
    self._left_tabs = shell.findChild(QtWidgets.QTabWidget, "left_tabs")
    if self._left_tabs is None:
        raise RuntimeError("Designer workbench shell is missing left_tabs")

    for tab_name in ("mesh_tab", "map_tab", "topology_tab", "boundary_tab", "model_tab", "run_tab"):
        self._ensure_designer_tab_scroll(
            self._designer_host_widget(shell, tab_name),
            f"{tab_name}_scroll",
        )

    mesh_tab_page = self._designer_host_widget(shell, "mesh_tab")
    self._bind_mesh_tab_controls(mesh_tab_page)

    map_tab_page = self._designer_host_widget(shell, "map_tab")
    map_data_layout = map_tab_page.findChild(QtWidgets.QGridLayout, "map_data_layout")
    map_actions_layout = map_tab_page.findChild(QtWidgets.QGridLayout, "map_actions_layout")
    map_results_layout = map_tab_page.findChild(QtWidgets.QGridLayout, "map_results_layout")
    map_tools_layout = map_tab_page.findChild(QtWidgets.QGridLayout, "map_tools_layout")
    if (
        map_data_layout is None
        or map_actions_layout is None
        or map_results_layout is None
        or map_tools_layout is None
    ):
        raise RuntimeError("Designer workbench shell map tab is missing one or more expected layouts")
    self._bind_map_tab_data_controls(map_tab_page, map_data_layout)
    self._bind_map_tab_action_controls(map_tab_page, map_actions_layout)
    self._bind_map_tab_results_controls(map_tab_page, map_results_layout)
    self._bind_map_tab_tools_controls(map_tab_page, map_tools_layout)

    topology_tab_page = self._designer_host_widget(shell, "topology_tab")
    topo_layout = topology_tab_page.findChild(QtWidgets.QGridLayout, "topo_layout")
    if topo_layout is None:
        raise RuntimeError("Designer workbench shell topology tab is missing topo_layout")
    self._bind_topology_tab_static_controls(topology_tab_page, topo_layout)
    self._bind_topology_tab_dynamic_controls(topology_tab_page, topo_layout)

    boundary_tab_page = self._designer_host_widget(shell, "boundary_tab")
    bc_grid = boundary_tab_page.findChild(QtWidgets.QGridLayout, "bc_grid")
    if bc_grid is None:
        raise RuntimeError("Designer workbench shell boundary tab is missing bc_grid")
    self._populate_boundary_tab_controls(bc_grid)

    model_tab_page = self._designer_host_widget(shell, "model_tab")
    param_form = model_tab_page.findChild(QtWidgets.QFormLayout, "model_param_form")
    if param_form is None:
        raise RuntimeError("Designer workbench shell model tab is missing model_param_form")
    self._bind_model_tab_core_controls(model_tab_page, param_form)
    self._bind_model_tab_hydrology_controls(model_tab_page, param_form)
    self._bind_model_tab_solver_controls(model_tab_page, param_form)
    self._bind_model_tab_3d_patch_controls(model_tab_page, param_form)
    self._bind_model_tab_3d_subgrid_drainage_controls(model_tab_page, param_form)  # legacy: single param_form, no separate solver_form

    run_tab_page = self._designer_host_widget(shell, "run_tab")
    self._bind_run_tab_controls(run_tab_page)

    self._left_tabs.setMinimumWidth(0)
    for _cb in self._left_tabs.findChildren(QtWidgets.QComboBox):
        _cb.setMinimumContentsLength(0)
        _cb.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
    for _btn in self._left_tabs.findChildren(QtWidgets.QPushButton):
        _btn.setMinimumWidth(0)
    for _sp in self._left_tabs.findChildren(
        (QtWidgets.QDoubleSpinBox, QtWidgets.QSpinBox)  # type: ignore[arg-type]
    ):
        _sp.setMinimumWidth(0)
    self._make_left_controls_compact(self._left_tabs)
    self._register_detachable_tab_widget(self._left_tabs)



def designer_build_ui(self):
    root = self.layout()
    if not isinstance(root, QtWidgets.QVBoxLayout):
        root = QtWidgets.QVBoxLayout(self)
    while root.count():
        item = root.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()

    shell = self._build_designer_workbench_shell()
    root.addWidget(shell, stretch=1)

    header = shell.findChild(QtWidgets.QLabel, "header_lbl")
    if header is not None:
        header.setText(
            "Interactive 2D SWE workflow: generate mesh, assign side BCs, set model parameters, "
            "run, and visualize results."
        )
        header.setWordWrap(True)

    self._populate_designer_left_tabs(shell)
    self._bind_right_pane_controls(self._designer_host_widget(shell, "right_pane_host"))

    split = shell.findChild(QtWidgets.QSplitter, "main_splitter")
    if split is not None:
        split.setSizes([420, 740])

    buttons = shell.findChild(QtWidgets.QDialogButtonBox, "bottom_buttons")
    if buttons is not None:
        try:
            buttons.rejected.disconnect(self.reject)
        except Exception:
            pass
        try:
            buttons.accepted.disconnect(self.accept)
        except Exception:
            pass
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)

    self._refresh_layer_combos()



def studio_project_scope_key(self) -> str:
    project_key = "default"
    if _HAVE_QGIS_CORE and QgsProject is not None:
        try:
            proj = QgsProject.instance()
            file_name = str(proj.fileName() or "").strip()
            if file_name:
                project_key = file_name
            else:
                project_key = str(proj.homePath() or "").strip() or project_key
        except Exception:
            pass
    safe = "".join(ch if (ch.isalnum() or ch in ("_", "-", ".")) else "_" for ch in project_key)
    if not safe:
        safe = "default"
    return safe



def studio_layout_settings_keys(self) -> Tuple[str, str]:
    scope = self._studio_project_scope_key()
    base = f"Backwater2DWorkbenchStudio/v2/{scope}"
    return f"{base}/layout_state", f"{base}/layout_geometry"



def restore_studio_layout_state(self) -> None:
    if self._studio_main_window is None:
        return
    state_key, _geom_key = self._studio_layout_settings_keys()
    settings = QtCore.QSettings()
    try:
        state_raw = settings.value(state_key, "")
    except Exception:
        state_raw = ""

    restored = False
    if state_raw:
        try:
            state_bytes = QtCore.QByteArray.fromBase64(str(state_raw).encode("ascii"))
            restored = bool(self._studio_main_window.restoreState(state_bytes))
        except Exception:
            pass

    # Safety: always keep the core panes visible so Studio cannot reopen blank.
    try:
        center = self._studio_main_window.centralWidget()
        if center is not None:
            center.show()
    except Exception:
        pass
    try:
        if self._studio_left_dock is not None and not self._studio_left_dock.isVisible():
            self._studio_left_dock.show()
    except Exception:
        pass
    try:
        if self._studio_inspector_dock is not None and not self._studio_inspector_dock.isVisible():
            self._studio_inspector_dock.show()
    except Exception:
        pass

    if state_raw and not restored:
        self._studio_main_window.resize(1200, 760)



def save_studio_layout_state(self) -> None:
    if self._studio_main_window is None:
        return
    state_key, _geom_key = self._studio_layout_settings_keys()
    settings = QtCore.QSettings()
    try:
        state_b64 = bytes(self._studio_main_window.saveState().toBase64()).decode("ascii")
        settings.setValue(state_key, state_b64)
        settings.sync()
    except Exception:
        pass



def studio_mount_widget(self, host: QtWidgets.QWidget, widget: QtWidgets.QWidget) -> None:
    layout = host.layout()
    if not isinstance(layout, QtWidgets.QVBoxLayout):
        layout = QtWidgets.QVBoxLayout(host)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    while layout.count():
        item = layout.takeAt(0)
        child = item.widget()
        if child is not None:
            child.deleteLater()
    layout.addWidget(widget, stretch=1)



def studio_select_tab(self, name: str) -> None:
    if not hasattr(self, "_left_tabs") or self._left_tabs is None:
        return
    target = str(name or "").strip().lower()
    for idx in range(self._left_tabs.count()):
        if str(self._left_tabs.tabText(idx) or "").strip().lower() == target:
            self._left_tabs.setCurrentIndex(idx)
            return



def studio_set_feature_enabled(self, feature: str, enabled: bool) -> None:
    """Set a Studio feature flag and re-apply visibility filters.

    Valid feature keys are defined in self._studio_feature_flags.
    After updating the flag, calls _studio_apply_feature_filters() to
    immediately show/hide matching widgets and tabs.

    To add a new feature:
      1. Add the key to self._studio_feature_flags in the dialog __init__
      2. Add keyword entries in studio_feature_keywords() below
      3. Add menu + toolbar toggles in _install_studio_host_controls()
    """
    key = str(feature or "").strip().lower()
    if key not in self._studio_feature_flags:
        return
    self._studio_feature_flags[key] = bool(enabled)
    self._studio_apply_feature_filters()



def studio_feature_keywords(self) -> Dict[str, Tuple[str, ...]]:
    """Return {feature_key: (keyword_strings)} for widget-to-feature matching.

    Each widget's objectName and text is lowercased and checked against
    these keywords.  If ANY keyword from a feature set appears in the
    widget's blob, the widget is controlled by that feature's visibility.

    When adding a new feature, add an entry here with keywords that
    appear in the objectNames of the widgets that feature controls.
    Use precise keywords (e.g. "3d_patch" not just "patch") to avoid
    unintended matches with widgets in other tabs.

    See docs/STUDIO_UI_ARCHITECTURE.md section C.
    """
    return {
        "rainfall": ("rain", "gauge", "hyet", "storm", "runoff", "precip"),
        "drainage_structures": (
            "drain", "node", "link", "inlet", "outfall", "pipe", "network",
            "structure", "culvert", "weir", "orifice", "gate", "spillway",
            "coupling",
        ),
        "3d_patch": ("3d_patch", "patch_3d", "swe3d"),
    }



def studio_widget_text_blob(self, widget: QtWidgets.QWidget) -> str:
    parts = [str(widget.objectName() or "")]
    try:
        if hasattr(widget, "text") and callable(widget.text):
            parts.append(str(widget.text() or ""))
    except Exception:
        pass
    try:
        if hasattr(widget, "title") and callable(widget.title):
            parts.append(str(widget.title() or ""))
    except Exception:
        pass
    try:
        parts.append(str(widget.toolTip() or ""))
    except Exception:
        pass
    return " ".join(parts).lower()



def studio_apply_feature_filters(self) -> None:
    if not hasattr(self, "_left_tabs") or self._left_tabs is None:
        return
    keywords = self._studio_feature_keywords()
    for widget in self._left_tabs.findChildren(QtWidgets.QWidget):
        if widget is self._left_tabs:
            continue
        blob = self._studio_widget_text_blob(widget)
        matched = []
        for feature, words in keywords.items():
            if any(word in blob for word in words):
                matched.append(feature)
        if not matched:
            continue
        visible = all(self._studio_feature_flags.get(feature, True) for feature in matched)
        try:
            widget.setVisible(visible)
        except Exception:
            pass
    # Sync tab page visibility: hide/show tabs whose page or content matches
    # a feature flag, so the tab bar entry disappears when the feature is off.
    tabs = self._left_tabs
    for i in range(tabs.count()):
        page = tabs.widget(i)
        if page is None:
            continue
        blob = self._studio_widget_text_blob(page)
        matched = []
        for feature, words in keywords.items():
            if any(word in blob for word in words):
                matched.append(feature)
        if not matched:
            continue
        visible = all(self._studio_feature_flags.get(feature, True) for feature in matched)
        try:
            tabs.setTabVisible(i, visible)
        except Exception:
            pass



def studio_sync_view_mode(self, idx: int) -> None:
    if not hasattr(self, "view_mode_combo") or self.view_mode_combo is None:
        return
    if idx < 0:
        return
    try:
        self.view_mode_combo.setCurrentIndex(idx)
    except Exception:
        pass



def studio_apply_visual_profile(self, profile: str) -> None:
    profile_key = str(profile or "").strip().lower()
    if self._studio_main_window is None:
        return
    if profile_key == "diagnostics":
        self._studio_main_window.setStyleSheet(
            "QMainWindow { background: #1f232a; }"
            "QDockWidget::title { background: #2d3640; color: #f2f4f8; padding: 4px; }"
            "QToolBar { background: #2b3139; border-bottom: 1px solid #3a424c; }"
            "QStatusBar { background: #2b3139; color: #e7edf5; }"
        )
    elif profile_key == "presentation":
        self._studio_main_window.setStyleSheet(
            "QMainWindow { background: #f2f5f8; }"
            "QDockWidget::title { background: #d9e2ec; color: #243b53; padding: 4px; }"
            "QToolBar { background: #e4ebf2; border-bottom: 1px solid #c9d4df; }"
            "QStatusBar { background: #e4ebf2; color: #243b53; }"
        )
    else:
        self._studio_main_window.setStyleSheet("")



def studio_update_status(self) -> None:
    if self._studio_status_label is None:
        return
    project_name = "(no project)"
    project_home = ""
    if _HAVE_QGIS_CORE and QgsProject is not None:
        try:
            proj = QgsProject.instance()
            project_name = str(proj.baseName() or "").strip() or "(unnamed project)"
            project_home = str(proj.homePath() or "").strip()
        except Exception:
            pass
    mode_txt = str(getattr(self, "_swe2d_workbench_host_mode", "window") or "window")
    detail = f"Project: {project_name}"
    if project_home:
        detail += f" | Home: {project_home}"
    detail += f" | Host mode: {mode_txt}"
    self._studio_status_label.setText(detail)



def studio_build_ui(self):
    # Diagnostic: log that studio_build_ui was entered
    try:
        self._log("[Studio] studio_build_ui entered")
    except Exception:
        pass
    root = self.layout()
    if not isinstance(root, QtWidgets.QVBoxLayout):
        root = QtWidgets.QVBoxLayout(self)
    root.setContentsMargins(0, 0, 0, 0)
    root.setSpacing(0)
    while root.count():
        item = root.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()

    self._studio_main_window = QtWidgets.QMainWindow(self)
    self._studio_main_window.setWindowFlags(QtCore.Qt.Widget)
    self._studio_main_window.setObjectName("SWE2DStudioMainWindow")
    self._studio_main_window.setDockOptions(
        QtWidgets.QMainWindow.AllowNestedDocks
        | QtWidgets.QMainWindow.AllowTabbedDocks
        | QtWidgets.QMainWindow.AnimatedDocks
    )
    root.addWidget(self._studio_main_window, stretch=1)

    toolbar = QtWidgets.QToolBar("CFD Workspace", self._studio_main_window)
    toolbar.setObjectName("SWE2DStudioToolbar")
    toolbar.setMovable(False)
    toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
    self._studio_main_window.addToolBar(QtCore.Qt.TopToolBarArea, toolbar)

    act_mesh = toolbar.addAction("Mesh")
    act_model = toolbar.addAction("Model")
    act_run = toolbar.addAction("Run")
    act_map = toolbar.addAction("Map")
    toolbar.addSeparator()
    act_3d_patch = toolbar.addAction("3D Patch")
    act_3d_patch.setCheckable(True)
    act_3d_patch.setChecked(False)
    try:
        self._log("[Studio] 3D Patch toolbar action created")
    except Exception:
        pass
    toolbar.addSeparator()
    act_refresh = toolbar.addAction("Refresh Layers")
    act_snapshot = toolbar.addAction("Take Snapshot")
    toolbar.addSeparator()
    act_close = toolbar.addAction("Close")

    toolbar.addSeparator()
    toolbar.addWidget(QtWidgets.QLabel(" View: "))
    self._studio_view_mode_combo = QtWidgets.QComboBox()
    self._studio_view_mode_combo.addItems(["Mesh", "Depth", "Velocity magnitude", "Runtime Log"])
    toolbar.addWidget(self._studio_view_mode_combo)

    toolbar.addWidget(QtWidgets.QLabel(" Theme: "))
    self._studio_theme_combo = QtWidgets.QComboBox()
    self._studio_theme_combo.addItems(["Default", "Diagnostics", "Presentation"])
    toolbar.addWidget(self._studio_theme_combo)

    center_host = QtWidgets.QWidget()
    self._studio_mount_widget(center_host, self._build_right_pane())
    self._studio_main_window.setCentralWidget(center_host)

    self._studio_left_dock = QtWidgets.QDockWidget("Model Setup", self._studio_main_window)
    self._studio_left_dock.setObjectName("SWE2DStudioSetupDock")
    self._studio_left_dock.setFeatures(
        QtWidgets.QDockWidget.DockWidgetMovable
        | QtWidgets.QDockWidget.DockWidgetFloatable
        | QtWidgets.QDockWidget.DockWidgetClosable
    )
    left_host = QtWidgets.QWidget()
    self._compose_left_pane(left_host)
    self._studio_left_dock.setWidget(left_host)
    self._studio_main_window.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self._studio_left_dock)

    self._studio_inspector_dock = QtWidgets.QDockWidget("CFD Inspector", self._studio_main_window)
    self._studio_inspector_dock.setObjectName("SWE2DStudioInspectorDock")
    self._studio_inspector_dock.setFeatures(
        QtWidgets.QDockWidget.DockWidgetMovable
        | QtWidgets.QDockWidget.DockWidgetFloatable
        | QtWidgets.QDockWidget.DockWidgetClosable
    )

    inspector_tabs = QtWidgets.QTabWidget()
    inspector_tabs.setDocumentMode(True)

    tree_page = QtWidgets.QWidget()
    tree_layout = QtWidgets.QVBoxLayout(tree_page)
    tree_layout.setContentsMargins(6, 6, 6, 6)
    workspace_tree = QtWidgets.QTreeWidget()
    workspace_tree.setHeaderLabels(["Workspace Area", "Purpose"])
    root_item = QtWidgets.QTreeWidgetItem(["SWE2D CFD Studio", "QGIS-integrated workflow shell"])
    root_item.addChild(QtWidgets.QTreeWidgetItem(["Setup Dock", "Mesh/Boundary/Model tabs"]))
    root_item.addChild(QtWidgets.QTreeWidgetItem(["Central Workspace", "Runtime view and logs"]))
    root_item.addChild(QtWidgets.QTreeWidgetItem(["Inspector Dock", "QA checks and quick actions"]))
    workspace_tree.addTopLevelItem(root_item)
    workspace_tree.expandAll()
    tree_layout.addWidget(workspace_tree)
    inspector_tabs.addTab(tree_page, "Workspace")

    quick_page = QtWidgets.QWidget()
    quick_layout = QtWidgets.QVBoxLayout(quick_page)
    quick_layout.setContentsMargins(6, 6, 6, 6)
    tools_box = QtWidgets.QToolBox()

    nav_page = QtWidgets.QWidget()
    nav_layout = QtWidgets.QVBoxLayout(nav_page)
    cmd_mesh = QtWidgets.QCommandLinkButton("Open Mesh Setup", "Jump to grid generation and controls")
    cmd_model = QtWidgets.QCommandLinkButton("Open Model Setup", "Jump to solver and roughness settings")
    cmd_run = QtWidgets.QCommandLinkButton("Open Run Tab", "Jump to runtime and output controls")
    nav_layout.addWidget(cmd_mesh)
    nav_layout.addWidget(cmd_model)
    nav_layout.addWidget(cmd_run)
    nav_layout.addStretch(1)
    tools_box.addItem(nav_page, "Navigation")

    qa_page = QtWidgets.QWidget()
    qa_layout = QtWidgets.QVBoxLayout(qa_page)
    qa_hint = QtWidgets.QPlainTextEdit()
    qa_hint.setReadOnly(True)
    qa_hint.setPlainText(
        "CFD pre-run checks:\n"
        "1. Confirm mesh exists and BC sides are configured.\n"
        "2. Verify timestep mode and CFL consistency.\n"
        "3. Confirm output intervals and runtime duration.\n"
        "4. Enable 3D export toggles only when needed."
    )
    qa_layout.addWidget(qa_hint)
    qa_layout.addStretch(1)
    tools_box.addItem(qa_page, "Pre-run QA")

    quick_layout.addWidget(tools_box)
    inspector_tabs.addTab(quick_page, "Operations")

    self._studio_inspector_dock.setWidget(inspector_tabs)
    self._studio_main_window.addDockWidget(QtCore.Qt.RightDockWidgetArea, self._studio_inspector_dock)

    footer = QtWidgets.QStatusBar(self._studio_main_window)
    self._studio_main_window.setStatusBar(footer)
    self._studio_status_label = QtWidgets.QLabel("")
    footer.addPermanentWidget(self._studio_status_label, 1)
    self._studio_update_status()

    if hasattr(self, "view_mode_combo") and self.view_mode_combo is not None:
        try:
            self._studio_view_mode_combo.setCurrentIndex(max(0, int(self.view_mode_combo.currentIndex())))
        except Exception:
            pass

    act_mesh.triggered.connect(lambda: self._studio_select_tab("mesh"))
    act_model.triggered.connect(lambda: self._studio_select_tab("model"))
    act_run.triggered.connect(lambda: self._studio_select_tab("run"))
    act_map.triggered.connect(lambda: self._studio_select_tab("map"))
    act_refresh.triggered.connect(self._refresh_layer_combos)
    act_snapshot.triggered.connect(lambda: self.snapshot_btn.click() if hasattr(self, "snapshot_btn") and self.snapshot_btn is not None else None)

    # 3D Patch toggle — creates/hides a dock widget
    _SWE2D_3D_PATCH_DOCK_ATTR = "_swe2d_3d_patch_dock"

    def _toggle_3d_patch(checked: bool) -> None:
        try:
            self._log(f"[Studio] 3D Patch toggled: {checked}")
        except Exception:
            pass
        dock = getattr(self, _SWE2D_3D_PATCH_DOCK_ATTR, None)
        if checked and dock is None:
            patch_page = self._build_3d_patch_tab_page()
            dock = QtWidgets.QDockWidget("3D Patch Settings", self._studio_main_window)
            dock.setObjectName("SWE2D3DPatchDock")
            dock.setWidget(patch_page)
            dock.setFeatures(
                QtWidgets.QDockWidget.DockWidgetMovable
                | QtWidgets.QDockWidget.DockWidgetFloatable
                | QtWidgets.QDockWidget.DockWidgetClosable
            )
            dock.visibilityChanged.connect(
                lambda visible: act_3d_patch.setChecked(visible)
            )
            self._studio_main_window.addDockWidget(
                QtCore.Qt.RightDockWidgetArea, dock
            )
            setattr(self, _SWE2D_3D_PATCH_DOCK_ATTR, dock)
        elif not checked and dock is not None:
            self._studio_main_window.removeDockWidget(dock)
            dock.deleteLater()
            setattr(self, _SWE2D_3D_PATCH_DOCK_ATTR, None)

    act_3d_patch.toggled.connect(_toggle_3d_patch)
    act_open_coupling_results = toolbar.addAction("Open Drainage/Structure Results")
    act_open_coupling_results.triggered.connect(
        lambda: self._open_coupling_results_viewer()
        if hasattr(self, "_open_coupling_results_viewer")
        else None
    )
    act_close.triggered.connect(self.reject)

    cmd_mesh.clicked.connect(lambda: self._studio_select_tab("mesh"))
    cmd_model.clicked.connect(lambda: self._studio_select_tab("model"))
    cmd_run.clicked.connect(lambda: self._studio_select_tab("run"))

    self._studio_view_mode_combo.currentIndexChanged.connect(self._studio_sync_view_mode)
    self._studio_theme_combo.currentTextChanged.connect(self._studio_apply_visual_profile)

    self._restore_studio_layout_state()
    self._studio_apply_visual_profile("Default")
    self._studio_apply_feature_filters()
    self._refresh_layer_combos()


def scenario_apply_preset(self, preset_name: str) -> None:
    preset = self._SCENARIO_PRESETS.get(str(preset_name), None)
    if not isinstance(preset, dict):
        return

    if hasattr(self, "cfl_spin") and self.cfl_spin is not None:
        self.cfl_spin.setValue(float(preset["cfl"]))
    if hasattr(self, "dt_spin") and self.dt_spin is not None:
        self.dt_spin.setValue(float(preset["dt"]))
    if hasattr(self, "n_mann_spin") and self.n_mann_spin is not None:
        self.n_mann_spin.setValue(float(preset["n_mann"]))
    if hasattr(self, "adaptive_cfl_dt_chk") and self.adaptive_cfl_dt_chk is not None:
        self.adaptive_cfl_dt_chk.setChecked(bool(preset["adaptive"]))
    if hasattr(self, "rain_rate_spin") and self.rain_rate_spin is not None:
        self.rain_rate_spin.setValue(float(preset["rain"]))
    if hasattr(self, "output_interval_edit") and self.output_interval_edit is not None:
        self.output_interval_edit.setText(str(preset["output_interval"]))
    if hasattr(self, "line_output_interval_edit") and self.line_output_interval_edit is not None:
        self.line_output_interval_edit.setText(str(preset["line_output_interval"]))
    self._log(f"Scenario preset applied: {preset_name}")



def scenario_build_ui(self):
    root = self.layout()
    if not isinstance(root, QtWidgets.QVBoxLayout):
        root = QtWidgets.QVBoxLayout(self)
    while root.count():
        item = root.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()

    header = QtWidgets.QLabel(
        "Scenario-first mode: choose a run profile first, then tune details in tabs."
    )
    header.setWordWrap(True)
    root.addWidget(header)

    scenario_group = QtWidgets.QGroupBox("Scenario Profiles")
    scenario_layout = QtWidgets.QHBoxLayout(scenario_group)
    scenario_layout.addWidget(QtWidgets.QLabel("Preset:"))
    self._scenario_profile_combo = QtWidgets.QComboBox()
    self._scenario_profile_combo.addItems(["Balanced", "Stable", "Fast"])
    scenario_layout.addWidget(self._scenario_profile_combo)
    apply_btn = QtWidgets.QPushButton("Apply Preset")
    scenario_layout.addWidget(apply_btn)
    quick_balanced_btn = QtWidgets.QPushButton("Balanced")
    quick_stable_btn = QtWidgets.QPushButton("Stable")
    quick_fast_btn = QtWidgets.QPushButton("Fast")
    scenario_layout.addWidget(quick_balanced_btn)
    scenario_layout.addWidget(quick_stable_btn)
    scenario_layout.addWidget(quick_fast_btn)
    scenario_layout.addStretch(1)
    root.addWidget(scenario_group)

    split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
    left_host = QtWidgets.QWidget()
    right_host = QtWidgets.QWidget()
    split.addWidget(left_host)
    split.addWidget(right_host)
    split.setSizes([430, 740])
    root.addWidget(split, stretch=1)

    buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
    root.addWidget(buttons)

    self._compose_left_pane(left_host)
    right_layout = QtWidgets.QVBoxLayout(right_host)
    right_layout.setContentsMargins(0, 0, 0, 0)
    right_layout.setSpacing(0)
    right_layout.addWidget(self._build_right_pane(), stretch=1)

    apply_btn.clicked.connect(lambda: self._apply_scenario_preset(str(self._scenario_profile_combo.currentText())))
    quick_balanced_btn.clicked.connect(lambda: self._apply_scenario_preset("Balanced"))
    quick_stable_btn.clicked.connect(lambda: self._apply_scenario_preset("Stable"))
    quick_fast_btn.clicked.connect(lambda: self._apply_scenario_preset("Fast"))

    try:
        buttons.rejected.disconnect(self.reject)
    except Exception:
        pass
    try:
        buttons.accepted.disconnect(self.accept)
    except Exception:
        pass
    buttons.rejected.connect(self.reject)
    buttons.accepted.connect(self.accept)
    self._refresh_layer_combos()




