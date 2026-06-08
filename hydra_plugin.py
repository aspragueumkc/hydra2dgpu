"""QGIS plugin glue for HYDRA2DGPU solver UI.

Opens the 2D SWE GPU workbench dialog directly (no 1D/lumped hydrology dock).
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
        root = os.path.abspath(os.path.join(self.plugin_dir, '..'))
        if root not in sys.path:
            sys.path.insert(0, root)

        self.main_menu = None
        self.main_menu_actions = []
        self._owns_main_menu = False
        self._swe2d_dialog = None
        self._plugin_menu_path = '&HYDRA2DGPU'
        self._orig_quit_on_last_window_closed = None
        self._qt_quit_hardened = False
        self._window_guard_log_emitted = False
        self._close_guard_filter = None
        self._enable_app_event_filter = str(os.environ.get('HYDRA_ENABLE_APP_EVENT_FILTER', '')).strip().lower() in (
            '1', 'true', 'yes', 'on'
        )

    def initGui(self):
        self._harden_qt_quit_behavior()
        if self._enable_app_event_filter:
            self._install_close_guard_filter()
        self._install_main_menu_bar_menu()

    def unload(self):
        self._remove_main_menu_bar_menu()
        if self._swe2d_dialog is not None:
            try:
                self._swe2d_dialog.close()
            except Exception:
                pass
            self._swe2d_dialog = None
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
            self.iface.messageBar().pushWarning('HYDRA2DGPU', str(message or 'Rogue top-level window removed.'))
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
        """Open the HYDRA2DGPU workbench docked into the QGIS main window."""
        self._harden_qt_quit_behavior()
        if self._enable_app_event_filter:
            self._install_close_guard_filter()
        try:
            from swe2d_workbench_qt import launch_swe2d_workbench_studio
            launch_swe2d_workbench_studio(parent=self.iface.mainWindow(), iface=self.iface, host_mode="dock")
        except Exception as exc:
            self.iface.messageBar().pushMessage(
                'HYDRA2DGPU', f'Failed to open workbench: {exc}', level=Qgis.Critical
            )

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
                if menu.objectName() == 'HYDRA2DGMainMenu':
                    return menu
            except Exception:
                pass
            try:
                if str(menu.title()).replace('&', '').strip().lower() == 'hydra2dgpu':
                    return menu
            except Exception:
                pass
        return None

    def _install_main_menu_bar_menu(self):
        menu_bar = self._menu_bar()
        if menu_bar is None:
            return

        menu = self._find_hydra_main_menu()
        if menu is None:
            menu = QMenu('HYDRA2DGPU', menu_bar)
            menu.setObjectName('HYDRA2DGMainMenu')
            menu_bar.addMenu(menu)
            self._owns_main_menu = True
        else:
            self._owns_main_menu = False

        self.main_menu = menu

        action_specs = [
            ('HYDRA2DMenuOpenPanelAction', 'Open HYDRA2DGPU Panel', lambda: self.run()),
        ]

        existing = {a.objectName(): a for a in menu.actions() if a is not None}
        for object_name, _text, _cb in action_specs:
            stale = existing.get(object_name)
            if stale is not None:
                menu.removeAction(stale)

        self.main_menu_actions = []
        for object_name, text, callback in action_specs:
            action = QAction(text, self.iface.mainWindow())
            action.setObjectName(object_name)
            action.triggered.connect(callback)
            menu.addAction(action)
            self.main_menu_actions.append(action)

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
