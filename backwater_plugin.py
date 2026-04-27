"""QGIS plugin glue for Backwater solver UI.

This plugin creates a dockable UI by embedding the existing Backwater
widget.
Note: This code is intended to be loaded inside QGIS where `iface` is
available. When running outside QGIS it will not function.
"""
import os
import sys
from qgis.PyQt.QtWidgets import QAction, QMenu
from qgis.PyQt.QtCore import Qt
from qgis.core import Qgis


class BackwaterQgisPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        # try to ensure workspace root is on path so we can import backwater modules
        root = os.path.abspath(os.path.join(self.plugin_dir, '..'))
        if root not in sys.path:
            sys.path.insert(0, root)

        # import the UI factory and utils. Prefer plugin-local modules, but
        # fall back to top-level `backwater_qt` when the package-local file
        # is not present (useful when plugin references shared workspace files).
        try:
            try:
                from .backwater_qt import create_backwater_dockwidget
            except Exception:
                # fallback to workspace-level backwater_qt
                try:
                    import backwater_qt as _bwqt
                    create_backwater_dockwidget = _bwqt.create_backwater_dockwidget
                except Exception:
                    create_backwater_dockwidget = None

            # ensure the plugin ui_adapter uses the QGIS iface
            try:
                from . import ui_adapter
                ui_adapter.iface = iface
            except Exception:
                try:
                    # fallback import
                    #from qgis_backwater_plugin import ui_adapter
                    import ui_adapter
                    ui_adapter.iface = iface
                except Exception:
                    pass

            self._create_dock = create_backwater_dockwidget
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
        self._owns_main_menu = False
        self._plugin_menu_path = '&Backwater'

    def initGui(self):
        #self.action = QAction('Open Backwater Panel', self.iface.mainWindow())
        #self.action.triggered.connect(self.run)
        #self.iface.addToolBarIcon(self.action)
        #self.iface.addPluginToMenu(self._plugin_menu_path, self.action)
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

    def run(self):
        if not self._create_dock:
            self.iface.messageBar().pushMessage('Backwater', 'UI components not found', level=Qgis.Critical)
            return
        if not self.dock:
            self.dock = self._create_dock(parent=self.iface.mainWindow(), title='Backwater')
            try:
                self.dock.setObjectName('BackwaterMainDock')
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

    def _find_backwater_main_menu(self):
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
                if menu.objectName() == 'BackwaterMainMenu':
                    return menu
            except Exception:
                pass
            try:
                if str(menu.title()).replace('&', '').strip().lower() == 'backwater':
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
                'Backwater',
                'Backwater panel is not available.',
                level=Qgis.Warning,
                duration=6,
            )
            return
        fn = getattr(widget, method_name, None)
        if not callable(fn):
            self.iface.messageBar().pushMessage(
                'Backwater',
                f'Action not available: {method_name}',
                level=Qgis.Warning,
                duration=6,
            )
            return
        try:
            fn()
        except Exception as exc:
            self.iface.messageBar().pushMessage(
                'Backwater',
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
                'Backwater',
                'Backwater panel is not available.',
                level=Qgis.Warning,
                duration=6,
            )
            return False
        try:
            callback(widget)
            return True
        except Exception as exc:
            self.iface.messageBar().pushMessage(
                'Backwater',
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
            solver_combo = getattr(widget, 'solver_combo', None)
            if solver_combo is not None:
                solver_name = str(solver_combo.currentText()).strip().lower() or solver_name
            alpha_combo = getattr(widget, 'alpha_combo', None)
            if alpha_combo is not None:
                alpha_name = str(alpha_combo.currentText()).strip().lower() or alpha_name
            self._set_option_checks(solver_name=solver_name, alpha_name=alpha_name)

        if self.dock is None:
            self._set_option_checks(solver_name='py', alpha_name='conveyance')
            return
        self._with_widget(_refresh, ensure_dock=False)

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

        menu = self._find_backwater_main_menu()
        if menu is None:
            menu = QMenu('Backwater', menu_bar)
            menu.setObjectName('BackwaterMainMenu')
            menu_bar.addMenu(menu)
            self._owns_main_menu = True
        else:
            self._owns_main_menu = False

        self.main_menu = menu

        action_specs = [
            ('BackwaterMenuOpenPanelAction', 'Open Backwater Panel', lambda: self.run()),
            ('BackwaterMenuCreateModelAction', 'Create New Model GeoPackage...', lambda: self._invoke_widget_method('on_new_model')),
            ('BackwaterMenuLoadModelAction', 'Load Model GeoPackage...', lambda: self._invoke_widget_method('on_menu_open_model')),
            ('BackwaterMenuSaveModelAction', 'Save Model GeoPackage As...', lambda: self._invoke_widget_method('on_save_geopackage')),
            ('BackwaterMenuRunAction', 'Run Model', lambda: self._invoke_widget_method('on_run')),
            ('BackwaterMenuResultsPlotAction', 'Open Results Plot', lambda: self._invoke_widget_method('open_results_plot')),
            ('BackwaterMenuResultsTableAction', 'Open Results Table', lambda: self._invoke_widget_method('open_results_table')),
            ('BackwaterMenuToggleEditingAction', 'Enable/Disable Layer Editing', lambda: self._invoke_widget_method('on_toggle_geopackage_editing')),
            ('BackwaterMenuSaveEditsAction', 'Save Layer Edits', lambda: self._invoke_widget_method('on_save_layer_edits')),
        ]

        existing = {a.objectName(): a for a in menu.actions() if a is not None}
        for object_name, _text, _cb in action_specs:
            stale = existing.get(object_name)
            if stale is not None:
                menu.removeAction(stale)

        self.main_menu_actions = []
        for idx, (object_name, text, callback) in enumerate(action_specs):
            if idx in (1, 4, 7):
                menu.addSeparator()
            action = QAction(text, self.iface.mainWindow())
            action.setObjectName(object_name)
            action.triggered.connect(callback)
            menu.addAction(action)
            self.main_menu_actions.append(action)

        menu.addSeparator()
        stale_options = menu.findChild(QMenu, 'BackwaterMenuOptionsSubmenu')
        if stale_options is not None:
            try:
                menu.removeAction(stale_options.menuAction())
            except Exception:
                pass

        options_menu = menu.addMenu('Options')
        options_menu.setObjectName('BackwaterMenuOptionsSubmenu')

        solver_menu = options_menu.addMenu('Solver')
        solver_menu.setObjectName('BackwaterMenuSolverSubmenu')

        self.action_solver_py = QAction('Python (py)', self.iface.mainWindow())
        self.action_solver_py.setObjectName('BackwaterMenuSolverPyAction')
        self.action_solver_py.setCheckable(True)
        self.action_solver_py.triggered.connect(lambda checked: self._set_solver_option('py') if checked else None)
        solver_menu.addAction(self.action_solver_py)
        self.main_menu_actions.append(self.action_solver_py)

        self.action_solver_scipy = QAction('SciPy (scipy)', self.iface.mainWindow())
        self.action_solver_scipy.setObjectName('BackwaterMenuSolverScipyAction')
        self.action_solver_scipy.setCheckable(True)
        self.action_solver_scipy.triggered.connect(lambda checked: self._set_solver_option('scipy') if checked else None)
        solver_menu.addAction(self.action_solver_scipy)
        self.main_menu_actions.append(self.action_solver_scipy)

        alpha_menu = options_menu.addMenu('Alpha Method')
        alpha_menu.setObjectName('BackwaterMenuAlphaSubmenu')

        self.action_alpha_conveyance = QAction('Conveyance', self.iface.mainWindow())
        self.action_alpha_conveyance.setObjectName('BackwaterMenuAlphaConveyanceAction')
        self.action_alpha_conveyance.setCheckable(True)
        self.action_alpha_conveyance.triggered.connect(lambda checked: self._set_alpha_option('conveyance') if checked else None)
        alpha_menu.addAction(self.action_alpha_conveyance)
        self.main_menu_actions.append(self.action_alpha_conveyance)

        self.action_alpha_area = QAction('Area', self.iface.mainWindow())
        self.action_alpha_area.setObjectName('BackwaterMenuAlphaAreaAction')
        self.action_alpha_area.setCheckable(True)
        self.action_alpha_area.triggered.connect(lambda checked: self._set_alpha_option('area') if checked else None)
        alpha_menu.addAction(self.action_alpha_area)
        self.main_menu_actions.append(self.action_alpha_area)

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
        self._owns_main_menu = False
