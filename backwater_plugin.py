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
        self.main_menu_actions = []
        self._owns_main_menu = False
        self._plugin_menu_path = '&Backwater'

    def initGui(self):
        self.action = QAction('Open Backwater Panel', self.iface.mainWindow())
        self.action.triggered.connect(self.run)
        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToMenu(self._plugin_menu_path, self.action)
        self._install_main_menu_bar_menu()

    def unload(self):
        if self.action:
            self.iface.removePluginMenu(self._plugin_menu_path, self.action)
            self.iface.removeToolBarIcon(self.action)
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

    def _sync_main_menu_state(self):
        can_open = bool(self._create_dock)
        for action in self.main_menu_actions:
            try:
                action.setEnabled(can_open)
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
        self._owns_main_menu = False
