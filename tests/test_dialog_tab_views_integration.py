"""Integration tests: dialog uses tab view instances."""
import unittest
from unittest.mock import MagicMock

from qgis.PyQt import QtWidgets
from qgis.PyQt.QtWidgets import QApplication

_app = None


def _ensure_app():
    global _app
    if _app is None:
        _app = QApplication.instance() or QApplication([])


def _make_iface():
    """Return a MagicMock iface with a real QMainWindow as mainWindow."""
    main_win = QtWidgets.QMainWindow()
    iface = MagicMock()
    iface.mainWindow.return_value = main_win
    iface.addDockWidget = lambda area, dock: main_win.addDockWidget(area, dock)
    return iface


class TestDialogHasTabViews(unittest.TestCase):
    """Verify the dialog instantiates the existing tab view classes."""

    def setUp(self):
        _ensure_app()

    def _make_dialog(self):
        from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog
        return SWE2DWorkbenchStudioDialog(iface=_make_iface())

    def test_dialog_has_map_tab_view(self):
        from swe2d.workbench.views.map_tab_view import MapTabView
        dlg = self._make_dialog()
        try:
            self.assertTrue(hasattr(dlg, "_map_tab_view"))
            self.assertIsInstance(dlg._map_tab_view, MapTabView)
        finally:
            dlg.close()

    def test_dialog_has_topology_tab_view(self):
        from swe2d.workbench.views.topology_tab_view import TopologyTabView
        dlg = self._make_dialog()
        try:
            self.assertTrue(hasattr(dlg, "_topology_tab_view"))
            self.assertIsInstance(dlg._topology_tab_view, TopologyTabView)
        finally:
            dlg.close()

    def test_dialog_has_model_tab_view(self):
        from swe2d.workbench.views.model_tab_view import ModelTabView
        dlg = self._make_dialog()
        try:
            self.assertTrue(hasattr(dlg, "_model_tab_view"))
            self.assertIsInstance(dlg._model_tab_view, ModelTabView)
        finally:
            dlg.close()


class TestTabViewsAreInLeftTabs(unittest.TestCase):
    """Verify all left tab views are added to the left tab widget."""

    def setUp(self):
        _ensure_app()

    def test_all_tabs_present(self):
        from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog
        dlg = SWE2DWorkbenchStudioDialog(iface=_make_iface())
        try:
            tabs = dlg._left_tabs
            tab_texts = [tabs.tabText(i) for i in range(tabs.count())]
            for expected in ["Setup", "Mesh Generation", "Simulation"]:
                self.assertIn(expected, tab_texts, f"Missing tab: {expected}")
        finally:
            dlg.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
