"""QGIS plugin glue for HYDRA solver UI.

This plugin creates a dockable UI by embedding the existing HYDRA
widget.
Note: This code is intended to be loaded inside QGIS where `iface` is
available. When running outside QGIS it will not function.
"""
import os
import sys
from qgis.PyQt import QtCore
from qgis.PyQt.QtWidgets import QAction, QApplication, QMainWindow, QMenu
from qgis.PyQt.QtCore import Qt
from qgis.core import Qgis


class _RogueWindowCloseGuard(QtCore.QObject):
    """Intercept close events for rogue duplicate top-level windows."""

    def __init__(self, plugin):
        super().__init__()
        self._plugin = plugin

    def eventFilter(self, obj, event):
        try:
            if event is None or event.type() != QtCore.QEvent.Close:
                return False
            if self._plugin is None:
                return False
            if self._plugin._is_rogue_duplicate_main_window(obj):
                try:
                    event.ignore()
                except Exception:
                    pass
                try:
                    obj.hide()
                except Exception:
                    pass
                try:
                    obj.deleteLater()
                except Exception:
                    pass
                self._plugin._emit_rogue_window_warning(
                    "Blocked close on rogue blank top-level window and removed it."
                )
                return True
        except Exception:
            return False
        return False


class HydraQgisPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        # try to ensure workspace root is on path so we can import HYDRA modules
        root = os.path.abspath(os.path.join(self.plugin_dir, '..'))
        if root not in sys.path:
            sys.path.insert(0, root)

        # import the UI factory and utils. Prefer plugin-local modules, but
        # fall back to top-level `hydra_qt` when the package-local file
        # is not present (useful when plugin references shared workspace files).
        try:
            try:
                from .hydra_qt import create_hydra_dockwidget
            except Exception:
                # fallback to workspace-level hydra_qt
                try:
                    import hydra_qt as _bwqt
                    create_hydra_dockwidget = _bwqt.create_hydra_dockwidget
                except Exception:
                    create_hydra_dockwidget = None

            # ensure the plugin ui_adapter uses the QGIS iface
            try:
                from . import ui_adapter
                ui_adapter.iface = iface
            except Exception:
                try:
                    # fallback import
                    #from qgis_HYDRA_plugin import ui_adapter
                    import ui_adapter
                    ui_adapter.iface = iface
                except Exception:
                    pass

            self._create_dock = create_hydra_dockwidget
        except Exception:
            # If imports fail in unexpected ways, allow plugin to load but
            # actions will report errors at runtime.
            self._create_dock = None

        self.dock = None
        self.action = None
        self.main_menu = None
        self.options_menu = None
        self.main_menu_actions = []
        self.action_solver_py = None
        self.action_solver_scipy = None
        self.action_alpha_conveyance = None
        self.action_alpha_area = None
        self.action_swe2d_workbench_docked = None
        self._owns_main_menu = False
        self._plugin_menu_path = '&HYDRA'
        self._orig_quit_on_last_window_closed = None
        self._qt_quit_hardened = False
        self._window_guard_log_emitted = False
        self._close_guard_filter = None
        self._enable_app_event_filter = str(os.environ.get('HYDRA_ENABLE_APP_EVENT_FILTER', '')).strip().lower() in (
            '1', 'true', 'yes', 'on'
        )

    def initGui(self):
        #self.action = QAction('Open HYDRA Panel', self.iface.mainWindow())
        #self.action.triggered.connect(self.run)
        #self.iface.addToolBarIcon(self.action)
        #self.iface.addPluginToMenu(self._plugin_menu_path, self.action)
        self._harden_qt_quit_behavior()
        # App-wide Qt event filters can trigger SIP conversion crashes in some
        # QGIS/PyQt builds; keep this guard opt-in for debug use only.
        if self._enable_app_event_filter:
            self._install_close_guard_filter()
        self._install_main_menu_bar_menu()

    def unload(self):
        #if self.action:
            #self.iface.removePluginMenu(self._plugin_menu_path, self.action)
            #self.iface.removeToolBarIcon(self.action)
        self._remove_main_menu_bar_menu()
        if self.dock:
            try:
                self.iface.removeDockWidget(self.dock)
            except Exception:
                pass
        self._remove_close_guard_filter()
        self._restore_qt_quit_behavior()

    def _install_close_guard_filter(self):
        if self._close_guard_filter is not None:
            return
        try:
            app = QApplication.instance()
            if app is None:
                return
            filt = _RogueWindowCloseGuard(self)
            app.installEventFilter(filt)
            self._close_guard_filter = filt
        except Exception:
            self._close_guard_filter = None

    def _remove_close_guard_filter(self):
        filt = self._close_guard_filter
        self._close_guard_filter = None
        if filt is None:
            return
        try:
            app = QApplication.instance()
            if app is not None:
                app.removeEventFilter(filt)
        except Exception:
            pass

    def _emit_rogue_window_warning(self, message: str):
        if self._window_guard_log_emitted:
            return
        try:
            self.iface.messageBar().pushWarning('HYDRA', str(message or 'Rogue top-level window removed.'))
        except Exception:
            pass
        self._window_guard_log_emitted = True

    def _is_rogue_duplicate_main_window(self, win) -> bool:
        main = self.iface.mainWindow() if self.iface is not None else None
        if win is None or main is None or win is main:
            return False
        if not isinstance(win, QMainWindow):
            return False
        try:
            if not win.isVisible():
                return False
        except Exception:
            return False

        try:
            main_title = str(main.windowTitle() or '').strip()
            win_title = str(win.windowTitle() or '').strip()
        except Exception:
            return False
        if not main_title or win_title != main_title:
            return False

        # Any additional visible QMainWindow with the same dynamic project title
        # as iface.mainWindow() is considered rogue in this plugin context.
        return True

    def _harden_qt_quit_behavior(self):
        try:
            from qgis.PyQt.QtWidgets import QApplication

            app = QApplication.instance()
            if app is None:
                return
            if self._orig_quit_on_last_window_closed is None:
                self._orig_quit_on_last_window_closed = bool(app.quitOnLastWindowClosed())
            app.setQuitOnLastWindowClosed(False)
            self._qt_quit_hardened = True
        except Exception:
            pass

    def _restore_qt_quit_behavior(self):
        if not bool(self._qt_quit_hardened):
            return
        try:
            from qgis.PyQt.QtWidgets import QApplication

            app = QApplication.instance()
            if app is None:
                return
            if self._orig_quit_on_last_window_closed is not None:
                app.setQuitOnLastWindowClosed(bool(self._orig_quit_on_last_window_closed))
        except Exception:
            pass
        finally:
            self._qt_quit_hardened = False

    def run(self):
        self._harden_qt_quit_behavior()
        if self._enable_app_event_filter:
            self._install_close_guard_filter()
        if not self._create_dock:
            self.iface.messageBar().pushMessage('HYDRA', 'UI components not found', level=Qgis.Critical)
            return
        if not self.dock:
            self.dock = self._create_dock(parent=self.iface.mainWindow(), title='HYDRA')
            try:
                self.dock.setObjectName('HYDRAMainDock')
            except Exception:
                pass
            self.iface.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock)
            try:
                widget = self.dock.widget()
                if widget is not None and hasattr(widget, 'set_dock_host_window'):
                    widget.set_dock_host_window(self.iface.mainWindow())
            except Exception:
                pass
        self.dock.show()
        self.dock.raise_()
        self._sync_main_menu_state()

    def _menu_bar(self):
        try:
            mw = self.iface.mainWindow()
            return mw.menuBar() if mw is not None else None
        except Exception:
            return None

    def _find_hydra_main_menu(self):
        menu_bar = self._menu_bar()
        if menu_bar is None:
            return None
        for action in menu_bar.actions():
            try:
                menu = action.menu()
            except Exception:
                menu = None
            if menu is None:
                continue
            try:
                if menu.objectName() == 'HYDRAMainMenu':
                    return menu
            except Exception:
                pass
            try:
                if str(menu.title()).replace('&', '').strip().lower() == 'hydra':
                    return menu
            except Exception:
                pass
        return None

    def _invoke_widget_method(self, method_name: str, ensure_dock: bool = True):
        if ensure_dock and self.dock is None:
            self.run()
        widget = self.dock.widget() if self.dock is not None else None
        if widget is None:
            self.iface.messageBar().pushMessage(
                'HYDRA',
                'HYDRA panel is not available.',
                level=Qgis.Warning,
                duration=6,
            )
            return
        fn = getattr(widget, method_name, None)
        if not callable(fn):
            self.iface.messageBar().pushMessage(
                'HYDRA',
                f'Action not available: {method_name}',
                level=Qgis.Warning,
                duration=6,
            )
            return
        try:
            fn()
        except Exception as exc:
            self.iface.messageBar().pushMessage(
                'HYDRA',
                f'{method_name} failed: {exc}',
                level=Qgis.Critical,
                duration=8,
            )

    def _with_widget(self, callback, ensure_dock: bool = True):
        if ensure_dock and self.dock is None:
            self.run()
        widget = self.dock.widget() if self.dock is not None else None
        if widget is None:
            self.iface.messageBar().pushMessage(
                'HYDRA',
                'HYDRA panel is not available.',
                level=Qgis.Warning,
                duration=6,
            )
            return False
        try:
            callback(widget)
            return True
        except Exception as exc:
            self.iface.messageBar().pushMessage(
                'HYDRA',
                f'Options update failed: {exc}',
                level=Qgis.Warning,
                duration=6,
            )
            return False

    def _set_solver_option(self, solver_name: str):
        normalized = str(solver_name).strip().lower()

        def _set(widget):
            combo = getattr(widget, 'solver_combo', None)
            if combo is None:
                raise RuntimeError('Solver selector not available')
            idx = combo.findText(normalized)
            if idx < 0:
                raise RuntimeError(f'Solver option not found: {normalized}')
            combo.setCurrentIndex(idx)

        if self._with_widget(_set, ensure_dock=True):
            self._set_option_checks(solver_name=normalized)

    def _set_alpha_option(self, alpha_name: str):
        normalized = str(alpha_name).strip().lower()

        def _set(widget):
            combo = getattr(widget, 'alpha_combo', None)
            if combo is None:
                raise RuntimeError('Alpha selector not available')
            idx = combo.findText(normalized)
            if idx < 0:
                raise RuntimeError(f'Alpha method not found: {normalized}')
            combo.setCurrentIndex(idx)

        if self._with_widget(_set, ensure_dock=True):
            self._set_option_checks(alpha_name=normalized)

    def _set_option_checks(self, solver_name: str = None, alpha_name: str = None):
        if self.action_solver_py is not None and solver_name is not None:
            self.action_solver_py.setChecked(str(solver_name).lower() == 'py')
        if self.action_solver_scipy is not None and solver_name is not None:
            self.action_solver_scipy.setChecked(str(solver_name).lower() == 'scipy')
        if self.action_alpha_conveyance is not None and alpha_name is not None:
            self.action_alpha_conveyance.setChecked(str(alpha_name).lower() == 'conveyance')
        if self.action_alpha_area is not None and alpha_name is not None:
            self.action_alpha_area.setChecked(str(alpha_name).lower() == 'area')

    def _refresh_option_checks_from_widget(self):
        def _refresh(widget):
            solver_name = 'py'
            alpha_name = 'conveyance'
            swe2d_host_mode = 'window'
            solver_combo = getattr(widget, 'solver_combo', None)
            if solver_combo is not None:
                solver_name = str(solver_combo.currentText()).strip().lower() or solver_name
            alpha_combo = getattr(widget, 'alpha_combo', None)
            if alpha_combo is not None:
                alpha_name = str(alpha_combo.currentText()).strip().lower() or alpha_name
            swe2d_host_mode = str(getattr(widget, '_swe2d_workbench_host_mode', 'window') or 'window').strip().lower()
            self._set_option_checks(solver_name=solver_name, alpha_name=alpha_name)
            if self.action_swe2d_workbench_docked is not None:
                self.action_swe2d_workbench_docked.blockSignals(True)
                self.action_swe2d_workbench_docked.setChecked(swe2d_host_mode == 'dock')
                self.action_swe2d_workbench_docked.blockSignals(False)

        if self.dock is None:
            self._set_option_checks(solver_name='py', alpha_name='conveyance')
            if self.action_swe2d_workbench_docked is not None:
                self.action_swe2d_workbench_docked.blockSignals(True)
                self.action_swe2d_workbench_docked.setChecked(False)
                self.action_swe2d_workbench_docked.blockSignals(False)
            return
        self._with_widget(_refresh, ensure_dock=False)

    def _set_swe2d_workbench_host_mode(self, checked: bool):
        mode = 'dock' if checked else 'window'

        def _set(widget):
            fn = getattr(widget, '_on_toggle_swe2d_workbench_host_mode', None)
            if callable(fn):
                fn(bool(checked))
                return
            setattr(widget, '_swe2d_workbench_host_mode', mode)
            save_fn = getattr(widget, '_save_swe2d_workbench_host_mode', None)
            if callable(save_fn):
                save_fn(mode)

        self._with_widget(_set, ensure_dock=True)

    def _sync_main_menu_state(self):
        can_open = bool(self._create_dock)
        for action in self.main_menu_actions:
            try:
                action.setEnabled(can_open)
            except Exception:
                pass
        try:
            self._refresh_option_checks_from_widget()
        except Exception:
            pass

    def _install_main_menu_bar_menu(self):
        menu_bar = self._menu_bar()
        if menu_bar is None:
            return

        menu = self._find_hydra_main_menu()
        if menu is None:
            menu = QMenu('HYDRA', menu_bar)
            menu.setObjectName('HYDRAMainMenu')
            menu_bar.addMenu(menu)
            self._owns_main_menu = True
        else:
            self._owns_main_menu = False

        self.main_menu = menu

        action_specs = [
            ('HYDRAMenuOpenPanelAction', 'Open HYDRA Panel', lambda: self.run()),
            ('HYDRAMenuCreateModelAction', 'Create New Model GeoPackage...', lambda: self._invoke_widget_method('on_new_model')),
            ('HYDRAMenuLoadModelAction', 'Load Model GeoPackage...', lambda: self._invoke_widget_method('on_menu_open_model')),
            ('HYDRAMenuSaveModelAction', 'Save Model GeoPackage As...', lambda: self._invoke_widget_method('on_save_geopackage')),
            ('HYDRAMenuRunAction', 'Run Model', lambda: self._invoke_widget_method('on_run')),
            ('HYDRAMenuResultsPlotAction', 'Open Results Plot', lambda: self._invoke_widget_method('open_results_plot')),
            ('HYDRAMenuResultsTableAction', 'Open Results Table', lambda: self._invoke_widget_method('open_results_table')),
            ('HYDRAMenuToggleEditingAction', 'Enable/Disable Layer Editing', lambda: self._invoke_widget_method('on_toggle_geopackage_editing')),
            ('HYDRAMenuSaveEditsAction', 'Save Layer Edits', lambda: self._invoke_widget_method('on_save_layer_edits')),
            ('HYDRAMenuUnsteadyInputDialogAction', 'Unsteady Input...', lambda: self._invoke_widget_method('open_unsteady_input_dialog')),
            ('HYDRAMenuSWE2DDemoAction', '2D SWE Workbench...', lambda: self._invoke_widget_method('open_swe2d_demo_dialog')),
            ('HYDRAMenuSWE2DDesignerAction', '2D SWE Workbench (Designer UI)...', lambda: self._invoke_widget_method('open_swe2d_designer_dialog')),
            ('HYDRAMenuSWE2DStudioAction', '2D SWE Workbench (Studio)...', lambda: self._invoke_widget_method('open_swe2d_studio_dialog')),
            ('HYDRAMenuSWE2DScenarioAction', '2D SWE Workbench (Scenario-first)...', lambda: self._invoke_widget_method('open_swe2d_scenario_dialog')),
            ('HYDRAMenuRunUnsteadyAction', 'Run Unsteady Model', lambda: self._invoke_widget_method('on_run_unsteady')),
            ('HYDRAMenuLoadUnsteadyRunAction', 'Load Saved Unsteady Run...', lambda: self._invoke_widget_method('on_load_unsteady_results')),
            ('HYDRAMenuUnsteadyDebugOptionsAction', 'Unsteady Debug Options...', lambda: self._invoke_widget_method('open_unsteady_debug_dialog')),
            ('HYDRAMenuUnsteadyDebugLogViewerAction', 'View Unsteady Debug Log...', lambda: self._invoke_widget_method('open_unsteady_debug_log_viewer')),
            ('HYDRAMenuUnsteadyProfileAction', 'Open Unsteady Profile Plot', lambda: self._invoke_widget_method('open_unsteady_results_plot')),
            ('HYDRAMenuUnsteadyHydroAction', 'Open Stage Hydrograph Plot', lambda: self._invoke_widget_method('open_unsteady_hydro_plot')),
            ('HYDRAMenuUnsteadySectionAction', 'Open Unsteady Section Results', lambda: self._invoke_widget_method('open_unsteady_section_results_plot')),
            ('HYDRAMenuMaxWSETableAction', 'Open Max WSE Table', lambda: self._invoke_widget_method('open_max_wse_table')),
        ]

        existing = {a.objectName(): a for a in menu.actions() if a is not None}
        for object_name, _text, _cb in action_specs:
            stale = existing.get(object_name)
            if stale is not None:
                menu.removeAction(stale)

        self.main_menu_actions = []
        for idx, (object_name, text, callback) in enumerate(action_specs):
            if idx in (1, 4, 7, 10):
                menu.addSeparator()
            action = QAction(text, self.iface.mainWindow())
            action.setObjectName(object_name)
            action.triggered.connect(callback)
            menu.addAction(action)
            self.main_menu_actions.append(action)

        menu.addSeparator()
        stale_options = menu.findChild(QMenu, 'HYDRAMenuOptionsSubmenu')
        if stale_options is not None:
            try:
                menu.removeAction(stale_options.menuAction())
            except Exception:
                pass

        options_menu = menu.addMenu('Options')
        options_menu.setObjectName('HYDRAMenuOptionsSubmenu')

        solver_menu = options_menu.addMenu('Solver')
        solver_menu.setObjectName('HYDRAMenuSolverSubmenu')

        self.action_solver_py = QAction('Python (py)', self.iface.mainWindow())
        self.action_solver_py.setObjectName('HYDRAMenuSolverPyAction')
        self.action_solver_py.setCheckable(True)
        self.action_solver_py.triggered.connect(lambda checked: self._set_solver_option('py') if checked else None)
        solver_menu.addAction(self.action_solver_py)
        self.main_menu_actions.append(self.action_solver_py)

        self.action_solver_scipy = QAction('SciPy (scipy)', self.iface.mainWindow())
        self.action_solver_scipy.setObjectName('HYDRAMenuSolverScipyAction')
        self.action_solver_scipy.setCheckable(True)
        self.action_solver_scipy.triggered.connect(lambda checked: self._set_solver_option('scipy') if checked else None)
        solver_menu.addAction(self.action_solver_scipy)
        self.main_menu_actions.append(self.action_solver_scipy)

        alpha_menu = options_menu.addMenu('Alpha Method')
        alpha_menu.setObjectName('HYDRAMenuAlphaSubmenu')

        self.action_alpha_conveyance = QAction('Conveyance', self.iface.mainWindow())
        self.action_alpha_conveyance.setObjectName('HYDRAMenuAlphaConveyanceAction')
        self.action_alpha_conveyance.setCheckable(True)
        self.action_alpha_conveyance.triggered.connect(lambda checked: self._set_alpha_option('conveyance') if checked else None)
        alpha_menu.addAction(self.action_alpha_conveyance)
        self.main_menu_actions.append(self.action_alpha_conveyance)

        self.action_alpha_area = QAction('Area', self.iface.mainWindow())
        self.action_alpha_area.setObjectName('HYDRAMenuAlphaAreaAction')
        self.action_alpha_area.setCheckable(True)
        self.action_alpha_area.triggered.connect(lambda checked: self._set_alpha_option('area') if checked else None)
        alpha_menu.addAction(self.action_alpha_area)
        self.main_menu_actions.append(self.action_alpha_area)

        options_menu.addSeparator()
        self.action_swe2d_workbench_docked = QAction('Dock 2D SWE Workbench Panel', self.iface.mainWindow())
        self.action_swe2d_workbench_docked.setObjectName('HYDRAMenuSWE2DDockedAction')
        self.action_swe2d_workbench_docked.setCheckable(True)
        self.action_swe2d_workbench_docked.toggled.connect(self._set_swe2d_workbench_host_mode)
        options_menu.addAction(self.action_swe2d_workbench_docked)
        self.main_menu_actions.append(self.action_swe2d_workbench_docked)

        options_menu.aboutToShow.connect(self._refresh_option_checks_from_widget)
        self.options_menu = options_menu
        self._refresh_option_checks_from_widget()

        self._sync_main_menu_state()

    def _remove_main_menu_bar_menu(self):
        menu = self.main_menu
        if menu is None:
            return

        for action in list(self.main_menu_actions):
            try:
                menu.removeAction(action)
            except Exception:
                pass
        self.main_menu_actions = []

        if self._owns_main_menu:
            try:
                menu_bar = self._menu_bar()
                if menu_bar is not None:
                    menu_bar.removeAction(menu.menuAction())
            except Exception:
                pass

        self.main_menu = None
        self.options_menu = None
        self.action_solver_py = None
        self.action_solver_scipy = None
        self.action_alpha_conveyance = None
        self.action_alpha_area = None
        self.action_swe2d_workbench_docked = None
        self._owns_main_menu = False
