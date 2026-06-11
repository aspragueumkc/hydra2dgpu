"""QGIS plugin glue for HYDRA2DGPU solver UI.

Opens the 2D SWE GPU workbench dialog directly (no 1D/lumped hydrology dock).
"""
import os
import subprocess
import sys
from qgis.PyQt import QtCore
from qgis.PyQt.QtWidgets import (
    QAction,
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)
from qgis.PyQt.QtCore import Qt, QSettings
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
            ('HYDRA2DMenuSettingsAction', 'Settings...', lambda: self.open_settings()),
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

    def open_settings(self):
        """Open the HYDRA2DGPU Settings dialog."""
        dlg = HYDRASettingsDialog(self.iface.mainWindow())
        dlg.exec_()

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


class HYDRASettingsDialog(QDialog):
    """Settings dialog for HYDRA2DGPU — CUDA DLL path and dependency management."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("HYDRA2DGPU Settings")
        self.setMinimumWidth(520)
        self._settings = QSettings("HYDRA2DGPU", "HYDRA2DGPU")

        layout = QVBoxLayout(self)

        # ── CUDA DLL Path section ────────────────────────────────────────
        layout.addWidget(QLabel("<b>CUDA Runtime DLL</b>"))
        layout.addWidget(QLabel(
            "Path to the folder containing cudart64_*.dll. "
            "Leave empty to use the bundled DLL in the plugin directory."
        ))

        path_layout = QHBoxLayout()
        self._cuda_path_edit = QLineEdit()
        self._cuda_path_edit.setPlaceholderText("(use bundled DLL)")
        self._cuda_path_edit.setText(self._settings.value("cuda_dll_path", ""))
        path_layout.addWidget(self._cuda_path_edit)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_cuda_dll)
        path_layout.addWidget(browse_btn)

        reset_btn = QPushButton("Reset to Default")
        reset_btn.clicked.connect(self._reset_cuda_path)
        path_layout.addWidget(reset_btn)

        layout.addLayout(path_layout)

        # ── Dependencies section ─────────────────────────────────────────
        layout.addSpacing(16)
        layout.addWidget(QLabel("<b>Python Dependencies</b>"))
        layout.addWidget(QLabel(
            "Check for missing required packages (numpy, gmsh) and install them "
            "into the QGIS Python environment."
        ))

        deps_btn = QPushButton("Check & Install Dependencies")
        deps_btn.clicked.connect(self._check_and_install_deps)
        layout.addWidget(deps_btn)

        self._deps_output = QTextEdit()
        self._deps_output.setReadOnly(True)
        self._deps_output.setMaximumHeight(150)
        self._deps_output.setPlaceholderText("Dependency check results will appear here...")
        layout.addWidget(self._deps_output)

        # ── Bottom buttons ───────────────────────────────────────────────
        layout.addSpacing(16)
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _browse_cuda_dll(self):
        """Open a file dialog to pick a CUDA DLL file or directory."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select CUDA Runtime DLL", "",
            "CUDA DLL (cudart64_*.dll);;All files (*)"
        )
        if path:
            self._cuda_path_edit.setText(os.path.dirname(path))

    def _reset_cuda_path(self):
        """Clear the custom CUDA DLL path (revert to bundled default)."""
        self._cuda_path_edit.setText("")

    def _on_accept(self):
        """Save settings and close."""
        self._settings.setValue("cuda_dll_path", self._cuda_path_edit.text().strip())
        self._settings.sync()
        self.accept()

    def _check_and_install_deps(self):
        """Run the dependency checker/installer inside the QGIS Python interpreter."""
        self._deps_output.clear()
        self._deps_output.append("Checking dependencies...\n")

        # Locate check_deps.py relative to this plugin
        plugin_dir = os.path.dirname(os.path.abspath(__file__))
        check_deps_path = os.path.join(plugin_dir, "tools", "check_deps.py")

        if not os.path.isfile(check_deps_path):
            self._deps_output.append("ERROR: tools/check_deps.py not found in plugin directory.")
            return

        # Run inside QGIS's Python: sys.executable is guaranteed to be QGIS's Python
        self._deps_output.append(f"Python: {sys.executable}")
        self._deps_output.append(f"Script: {check_deps_path}\n")

        try:
            result = subprocess.run(
                [sys.executable, check_deps_path, "--install", "--all"],
                capture_output=True, text=True, timeout=180,
            )
            self._deps_output.append(result.stdout)
            if result.stderr:
                self._deps_output.append(f"\n[stderr]\n{result.stderr}")
            if result.returncode == 0:
                self._deps_output.append("\n✅ All dependencies installed successfully.")
            else:
                self._deps_output.append(f"\n❌ Some dependencies failed (exit code {result.returncode}).")
        except subprocess.TimeoutExpired:
            self._deps_output.append("\n❌ Timed out waiting for pip install.")
        except Exception as exc:
            self._deps_output.append(f"\n❌ Error: {exc}")

        # Scroll to top
        self._deps_output.moveCursor(QtCore.QTextCursor.Start)
        self._deps_output.ensureCursorVisible()
