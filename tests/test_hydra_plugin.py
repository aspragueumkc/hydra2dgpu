"""Tests for the HYDRA2DGPU QGIS plugin entry point.

The plugin module lives at ``qgis_plugin/HYDRA2DGPU/hydra_plugin.py`` and is
importable under a real (headless, offscreen) QGIS application — no mocks.

Run inside the qgis_stable mamba env:

    python3 -m unittest tests.test_hydra_plugin -v
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

from tests.qgis_real_env import ensure_qgis_app, requires_qgis, stub_iface

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PLUGIN_DIR = os.path.join(_REPO_ROOT, "qgis_plugin", "HYDRA2DGPU")


def _import_hydra_plugin():
    """Import the plugin entry module from its shipped location."""
    if _PLUGIN_DIR not in sys.path:
        sys.path.insert(0, _PLUGIN_DIR)
    import hydra_plugin

    return hydra_plugin


@requires_qgis
class TestHydraPluginOpenPanelAction(unittest.TestCase):
    """Verify the plugin entry point drives the workbench launch path."""

    @classmethod
    def setUpClass(cls):
        ensure_qgis_app()
        cls._hp = _import_hydra_plugin()

    def setUp(self):
        self._iface = stub_iface()

    def tearDown(self):
        # launch_swe2d_workbench_studio records the active dialog in a module
        # global — never let a mock leak into other tests.
        import swe2d.workbench.views.studio_host_methods as _shm

        _shm._studio_active_dialog = None

    def _make_plugin(self):
        return self._hp.HydraQgisPlugin(self._iface)

    def test_open_panel_action_triggers_launch(self):
        """run() calls launch_swe2d_workbench_studio with our iface."""
        with patch.object(
            self._hp.HydraQgisPlugin, "_backend_available", return_value=True
        ), patch(
            "swe2d.workbench.studio_dialog.launch_swe2d_workbench_studio"
        ) as mock_launch:
            plugin = self._make_plugin()
            plugin.run()
            mock_launch.assert_called_once()
            _args, kwargs = mock_launch.call_args
            self.assertIs(kwargs.get("iface"), self._iface)

    def test_launch_swe2d_workbench_studio_no_crash(self):
        """launch_swe2d_workbench_studio handles bad iface without NameError.

        Regression: the function used ``self._log()`` in exception handlers
        but is a module-level function — ``self`` was undefined.
        """
        from swe2d.workbench.views.studio_host_methods import (
            launch_swe2d_workbench_studio,
        )

        iface = stub_iface()
        iface.mainWindow.side_effect = RuntimeError("no main window")

        with patch(
            "swe2d.workbench.studio_dialog.SWE2DWorkbenchStudioDialog"
        ) as mock_dlg_cls, patch(
            "swe2d.workbench.views.studio_host_methods._persist_workbench_was_open"
        ):
            mock_dlg = MagicMock()
            mock_dlg_cls.return_value = mock_dlg
            dlg = launch_swe2d_workbench_studio(parent=None, iface=iface)
            self.assertIs(dlg, mock_dlg)

    def test_launch_propagates_dialog_init_error(self):
        """If SWE2DWorkbenchStudioDialog.__init__ fails, error propagates."""
        from swe2d.workbench.views.studio_host_methods import (
            launch_swe2d_workbench_studio,
        )

        with patch(
            "swe2d.workbench.studio_dialog.SWE2DWorkbenchStudioDialog",
            side_effect=ImportError("simulated init failure"),
        ):
            with self.assertRaises(ImportError):
                launch_swe2d_workbench_studio(parent=None, iface=self._iface)

    def test_init_gui_creates_menu(self):
        """initGui() creates main_menu with the expected objectName (real Qt)."""
        from qgis.PyQt.QtWidgets import QMainWindow, QMenu

        plugins_menu = QMenu("Plugins")
        main_window = QMainWindow()
        self._iface.pluginMenu.return_value = plugins_menu
        self._iface.mainWindow.return_value = main_window

        plugin = self._make_plugin()
        plugin.initGui()
        self.addCleanup(plugin.unload)

        self.assertIsNotNone(plugin.main_menu)
        self.assertEqual(plugin.main_menu.objectName(), "HYDRA2DGMainMenu")
        action_names = {a.objectName() for a in plugin.main_menu_actions}
        self.assertIn("HYDRA2DMenuOpenWorkbenchAction", action_names)
        self.assertIn("HYDRA2DMenuCloseWorkbenchAction", action_names)
        self.assertIn("HYDRA2DMenuSettingsAction", action_names)

    def test_unload_clears_menu(self):
        """unload() removes the menu and clears action list."""
        from qgis.PyQt.QtWidgets import QMainWindow, QMenu

        plugins_menu = QMenu("Plugins")
        main_window = QMainWindow()
        self._iface.pluginMenu.return_value = plugins_menu
        self._iface.mainWindow.return_value = main_window

        plugin = self._make_plugin()
        plugin.initGui()
        plugin.unload()

        self.assertIsNone(plugin.main_menu)
        self.assertEqual(len(plugin.main_menu_actions), 0)

    def test_harden_qt_quit_behavior(self):
        """_harden_qt_quit_behavior sets quitOnLastWindowClosed False (real app)."""
        from qgis.PyQt.QtWidgets import QApplication

        app = QApplication.instance()
        self.assertIsNotNone(app)
        original = bool(app.quitOnLastWindowClosed())
        plugin = self._make_plugin()
        try:
            plugin._harden_qt_quit_behavior()
            self.assertTrue(plugin._qt_quit_hardened)
            self.assertFalse(app.quitOnLastWindowClosed())
        finally:
            plugin._restore_qt_quit_behavior()
        self.assertEqual(bool(app.quitOnLastWindowClosed()), original)

    def test_restore_qt_quit_behavior(self):
        """_restore_qt_quit_behavior clears the flag and restores the property."""
        from qgis.PyQt.QtWidgets import QApplication

        app = QApplication.instance()
        self.assertIsNotNone(app)
        original = bool(app.quitOnLastWindowClosed())
        plugin = self._make_plugin()
        plugin._harden_qt_quit_behavior()
        plugin._restore_qt_quit_behavior()
        self.assertFalse(plugin._qt_quit_hardened)
        self.assertEqual(bool(app.quitOnLastWindowClosed()), original)

    def test_on_project_read_restarts_active_workbench(self):
        """When a QGIS project is read and the workbench is active, the
        workbench must be torn down and re-launched so the new project's
        persisted widget values are discovered from a clean state."""
        plugin = self._make_plugin()

        with patch(
            "swe2d.workbench.views.studio_host_methods.launch_swe2d_workbench_studio"
        ) as mock_launch, patch(
            "swe2d.workbench.views.studio_host_methods._remove_workbench_studio_dock"
        ) as mock_remove, patch(
            "swe2d.workbench.views.studio_host_methods._capture_and_persist_window_state"
        ) as mock_capture, patch(
            "swe2d.workbench.views.studio_host_methods._studio_active_dialog",
            new=MagicMock(),
        ):
            plugin._restart_workbench_for_project()
            mock_capture.assert_called_once()
            mock_remove.assert_called_once()
            mock_launch.assert_called_once()

    def test_on_project_read_noop_when_workbench_inactive(self):
        """If the workbench is not active, _on_project_read does NOT
        auto-open it — only an open workbench gets restarted."""
        plugin = self._make_plugin()

        with patch(
            "swe2d.workbench.views.studio_host_methods.launch_swe2d_workbench_studio"
        ) as mock_launch, patch(
            "swe2d.workbench.views.studio_host_methods._remove_workbench_studio_dock"
        ) as mock_remove, patch(
            "swe2d.workbench.views.studio_host_methods._studio_active_dialog",
            new=None,
        ):
            plugin._restart_workbench_for_project()
            mock_remove.assert_not_called()
            mock_launch.assert_not_called()


@requires_qgis
class TestHydraPluginImports(unittest.TestCase):
    """Verify module-level symbols import correctly."""

    @classmethod
    def setUpClass(cls):
        ensure_qgis_app()

    def test_settings_dialog_is_qdialog_subclass(self):
        from qgis.PyQt.QtWidgets import QDialog

        hp = _import_hydra_plugin()
        self.assertTrue(issubclass(hp.HYDRASettingsDialog, QDialog))

    def test_rogue_window_guard_is_qobject_subclass(self):
        from qgis.PyQt.QtCore import QObject

        hp = _import_hydra_plugin()
        self.assertTrue(issubclass(hp._RogueWindowCloseGuard, QObject))


if __name__ == "__main__":
    unittest.main(verbosity=2)
