"""Tests for the hydra_plugin.py QGIS plugin entry point.

Run with real PyQt5 from the qgis_stable environment:

    mamba run -n qgis_stable python3 -m unittest tests.test_hydra_plugin -v
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from qgis.PyQt.QtWidgets import QApplication as _QApp

_test_app = _QApp.instance()
if _test_app is None:
    _test_app = _QApp([])

from tests.mocks.qgis_env import install_qgis_mocks

install_qgis_mocks()


class TestHydraPluginOpenPanelAction(unittest.TestCase):
    """Verify the 'Open HYDRA2DGPU Panel' action triggers run()."""

    def setUp(self):
        self._iface = MagicMock()
        self._iface.mainWindow.return_value = MagicMock()

    def _make_plugin(self):
        from hydra_plugin import HydraQgisPlugin
        return HydraQgisPlugin(self._iface)

    def test_open_panel_action_triggers_launch(self):
        """run() calls launch_swe2d_workbench_studio."""
        with patch("swe2d_workbench_qt.launch_swe2d_workbench_studio") as mock_launch:
            plugin = self._make_plugin()
            plugin.run()
            mock_launch.assert_called_once()
            _args, kwargs = mock_launch.call_args
            self.assertEqual(kwargs.get("host_mode"), "dock")
            self.assertIs(kwargs.get("iface"), self._iface)

    def test_launch_swe2d_workbench_studio_no_crash(self):
        """launch_swe2d_workbench_studio handles bad iface without NameError.

        Regression: the function used ``self._log()`` in exception handlers
        but is a module-level function — ``self`` was undefined.
        """
        from swe2d_workbench_qt import launch_swe2d_workbench_studio

        iface = MagicMock()
        iface.mainWindow.side_effect = RuntimeError("no main window")

        with patch("swe2d_workbench_qt.SWE2DWorkbenchStudioDialog") as mock_dlg_cls, \
             patch("swe2d_workbench_qt._build_studio_component_docks",
                   return_value={}) as mock_build, \
             patch("swe2d_workbench_qt._install_studio_host_controls") as mock_install:
            mock_dlg = MagicMock()
            mock_dlg_cls.return_value = mock_dlg
            mock_dlg._studio_update_status = MagicMock()
            launch_swe2d_workbench_studio(
                parent=None, iface=iface, host_mode="dock"
            )
            mock_build.assert_called_once()
            mock_install.assert_called_once()

    def test_launch_propagates_dialog_init_error(self):
        """If SWE2DWorkbenchStudioDialog.__init__ fails, error propagates."""
        from swe2d_workbench_qt import launch_swe2d_workbench_studio

        iface = MagicMock()
        iface.mainWindow.return_value = MagicMock()

        with patch("swe2d_workbench_qt.SWE2DWorkbenchStudioDialog",
                   side_effect=ImportError("simulated init failure")):
            with self.assertRaises(ImportError):
                launch_swe2d_workbench_studio(
                    parent=None, iface=iface, host_mode="dock"
                )

    def test_init_gui_creates_menu(self):
        """initGui() creates main_menu with the expected objectName."""
        import hydra_plugin as _hp

        def _named():
            m = MagicMock()
            m._on = ""
            m.setObjectName = lambda n: setattr(m, "_on", str(n))
            m.objectName = lambda: m._on
            return m

        fake_menu = _named()
        fake_menu.title.return_value = ""
        fake_menu._acts = []
        fake_menu.addAction = lambda *a: (
            (lambda act: (fake_menu._acts.append(act), act)[1])(_named()) if not a or not isinstance(a[0], MagicMock)
            else (fake_menu._acts.append(a[0]), a[0])[1]
        )
        fake_menu.actions = lambda: list(fake_menu._acts)
        fake_menu.removeAction = lambda act: fake_menu._acts.remove(act) if act in fake_menu._acts else None
        fake_menu.menuAction = MagicMock

        fake_action = _named()
        fake_action.text = lambda: ""
        fake_action.triggered = MagicMock()
        fake_action.trigger = MagicMock()

        with patch.multiple(_hp, QMenu=MagicMock(return_value=fake_menu),
                            QAction=MagicMock(return_value=fake_action),
                            QMainWindow=MagicMock):
            plugin = self._make_plugin()
            plugin.initGui()

        self.assertIsNotNone(plugin.main_menu)
        self.assertEqual(plugin.main_menu.objectName(), "HYDRA2DGMainMenu")

    def test_unload_clears_menu(self):
        """unload() removes the menu and clears action list."""
        import hydra_plugin as _hp

        def _named():
            m = MagicMock()
            m._on = ""
            m.setObjectName = lambda n: setattr(m, "_on", str(n))
            m.objectName = lambda: m._on
            return m

        fake_menu = _named()
        fake_menu.title.return_value = ""
        fake_menu._acts = []
        fake_menu.addAction = lambda *a: (
            (lambda act: (fake_menu._acts.append(act), act)[1])(_named()) if not a or not isinstance(a[0], MagicMock)
            else (fake_menu._acts.append(a[0]), a[0])[1]
        )
        fake_menu.actions = lambda: list(fake_menu._acts)
        fake_menu.removeAction = lambda act: fake_menu._acts.remove(act) if act in fake_menu._acts else None
        fake_menu.menuAction = MagicMock

        fake_action = _named()
        fake_action.text = lambda: ""
        fake_action.triggered = MagicMock()
        fake_action.trigger = MagicMock()

        with patch.multiple(_hp, QMenu=MagicMock(return_value=fake_menu),
                            QAction=MagicMock(return_value=fake_action),
                            QMainWindow=MagicMock):
            plugin = self._make_plugin()
            plugin.initGui()
            plugin.unload()

        self.assertIsNone(plugin.main_menu)
        self.assertEqual(len(plugin.main_menu_actions), 0)

    def test_harden_qt_quit_behavior(self):
        """_harden_qt_quit_behavior sets the hardened flag."""
        from hydra_plugin import HydraQgisPlugin
        plugin = HydraQgisPlugin(self._iface)
        plugin._harden_qt_quit_behavior()
        self.assertTrue(plugin._qt_quit_hardened)

    def test_restore_qt_quit_behavior(self):
        """_restore_qt_quit_behavior clears the hardened flag."""
        from hydra_plugin import HydraQgisPlugin
        plugin = HydraQgisPlugin(self._iface)
        plugin._harden_qt_quit_behavior()
        plugin._restore_qt_quit_behavior()
        self.assertFalse(plugin._qt_quit_hardened)


class TestHydraPluginImports(unittest.TestCase):
    """Verify module-level symbols import correctly."""

    def test_settings_dialog_is_qdialog_subclass(self):
        from hydra_plugin import HYDRASettingsDialog
        from qgis.PyQt.QtWidgets import QDialog
        self.assertTrue(issubclass(HYDRASettingsDialog, QDialog))

    def test_rogue_window_guard_is_qobject_subclass(self):
        from hydra_plugin import _RogueWindowCloseGuard
        from qgis.PyQt.QtCore import QObject
        self.assertTrue(issubclass(_RogueWindowCloseGuard, QObject))


if __name__ == "__main__":
    unittest.main(verbosity=2)
