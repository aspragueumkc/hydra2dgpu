"""Tests for WorkbenchDialogBuilder and thin __init__ pattern.

Validates two things:

1. The ``WorkbenchDialogBuilder`` class is a well-formed, importable,
   instantiable object that stores a dialog reference and exposes a
   callable ``configure`` entry point.

2. The ``SWE2DWorkbenchStudioDialog.__init__`` is a *thin* bootstrapper
   that:
   * delegates all post-init orchestration to ``WorkbenchDialogBuilder``
   * sets the expected window title before the builder runs
   * produces a non-null ``_studio_main_window`` after the builder runs
   * contains no business logic (no direct db / numerical / json work)

Created as part of Phase 1 Task 2 (extract __init__ logic into builder)
and Phase 1 Task 4 (comprehensive tests for thin plugin entry + builder).
"""
import inspect
import unittest
from unittest.mock import MagicMock, patch

from qgis.PyQt.QtWidgets import QApplication


_app = None


def _ensure_app():
    global _app
    if _app is None:
        _app = QApplication.instance() or QApplication([])


class TestWorkbenchDialogBuilder(unittest.TestCase):
    """Verify the builder class is a usable, well-formed Python object."""

    def test_builder_imports(self):
        """The builder class can be imported from its expected location."""
        from swe2d.workbench.workbench_dialog_builder import WorkbenchDialogBuilder
        self.assertIsNotNone(WorkbenchDialogBuilder)

    def test_builder_can_be_instantiated(self):
        """The builder can be constructed with a dialog reference."""
        from swe2d.workbench.workbench_dialog_builder import WorkbenchDialogBuilder
        builder = WorkbenchDialogBuilder(dialog=None)
        self.assertIsNotNone(builder)
        self.assertTrue(hasattr(builder, "configure"))

    def test_builder_stores_dialog_reference(self):
        """The builder stores the dialog reference passed in __init__."""
        from swe2d.workbench.workbench_dialog_builder import WorkbenchDialogBuilder
        sentinel = object()
        builder = WorkbenchDialogBuilder(dialog=sentinel)
        self.assertIs(builder._dialog, sentinel)

    def test_builder_configure_is_callable(self):
        """The builder exposes a callable `configure` entry point."""
        from swe2d.workbench.workbench_dialog_builder import WorkbenchDialogBuilder
        builder = WorkbenchDialogBuilder(dialog=None)
        self.assertTrue(callable(builder.configure))


def _make_iface():
    """Return a MagicMock iface with a real QMainWindow as mainWindow."""
    from qgis.PyQt import QtWidgets
    main_win = QtWidgets.QMainWindow()
    iface = MagicMock()
    iface.mainWindow.return_value = main_win
    iface.addDockWidget = lambda area, dock: main_win.addDockWidget(area, dock)
    return iface


class TestThinInitPattern(unittest.TestCase):
    """Verify ``__init__`` is a thin bootstrapper that delegates to builder.

    The dialog's ``__init__`` is expected to:
    1. Call ``super().__init__(parent)``
    2. Set a few primitive attributes (window title, modal flags)
    3. Delegate all post-init orchestration to ``WorkbenchDialogBuilder``

    No direct database access, numerical work, or JSON I/O should happen
    inside ``__init__`` — that work belongs in the builder and the
    post-init helpers it calls.
    """

    def setUp(self):
        _ensure_app()

    def test_init_calls_builder(self):
        """The dialog's __init__ should call WorkbenchDialogBuilder.configure()."""
        with patch(
            "swe2d.workbench.studio_dialog.WorkbenchDialogBuilder"
        ) as MockBuilder:
            mock_instance = MagicMock()
            MockBuilder.return_value = mock_instance

            from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog
            dlg = SWE2DWorkbenchStudioDialog(iface=_make_iface())
            try:
                MockBuilder.assert_called_once_with(dlg)
                mock_instance.configure.assert_called_once()
            finally:
                dlg.close()

    def test_init_sets_window_title(self):
        """The dialog sets its window title before the builder runs."""
        from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog
        dlg = SWE2DWorkbenchStudioDialog(iface=_make_iface())
        try:
            self.assertEqual(dlg.windowTitle(), "2D SWE Workbench (Studio)")
        finally:
            dlg.close()

    def test_init_creates_main_window(self):
        """The builder should create the main window during configure()."""
        from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog
        dlg = SWE2DWorkbenchStudioDialog(iface=_make_iface())
        try:
            self.assertIsNotNone(dlg._studio_main_window)
        finally:
            dlg.close()

    def test_init_does_not_hold_business_logic(self):
        """The __init__ should be thin — no db / numerical / json work."""
        from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog
        source = inspect.getsource(SWE2DWorkbenchStudioDialog.__init__)
        line_count = len(source.split("\n"))
        self.assertLess(
            line_count,
            50,
            f"__init__ has {line_count} lines, should be < 50",
        )
        self.assertNotIn("sqlite3", source)
        self.assertNotIn("numpy", source)
        self.assertNotIn("json.", source)

    def test_keyboard_shortcuts_constant_exists(self):
        from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog, KEYBOARD_SHORTCUTS
        self.assertIsInstance(KEYBOARD_SHORTCUTS, list)
        self.assertGreaterEqual(len(KEYBOARD_SHORTCUTS), 3)
        names = [s[0] for s in KEYBOARD_SHORTCUTS]
        self.assertIn("run", names)
        self.assertIn("cancel", names)

    def test_settings_dialog_imports(self):
        from swe2d.workbench.dialogs.workbench_settings_dialog import WorkbenchSettingsDialog
        self.assertIsNotNone(WorkbenchSettingsDialog)

    def test_settings_dialog_returns_flags(self):
        from swe2d.workbench.dialogs.workbench_settings_dialog import WorkbenchSettingsDialog
        _ensure_app()
        dlg = WorkbenchSettingsDialog({"rainfall": True, "drainage_structures": False})
        flags = dlg.flags()
        self.assertTrue(flags["rainfall"])
        self.assertFalse(flags["drainage_structures"])

    def test_open_workbench_settings_applies_flags(self):
        from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog
        from qgis.PyQt import QtWidgets
        dlg = SWE2DWorkbenchStudioDialog(iface=_make_iface())
        try:
            with patch("swe2d.workbench.dialogs.workbench_settings_dialog.WorkbenchSettingsDialog") as MockDlg:
                instance = MockDlg.return_value
                instance.exec.return_value = QtWidgets.QDialog.DialogCode.Accepted
                instance.flags.return_value = {"rainfall": False, "drainage_structures": True}
                dlg._open_workbench_settings()
                self.assertFalse(dlg._state.studio_feature_flags["rainfall"])
                self.assertTrue(dlg._state.studio_feature_flags["drainage_structures"])
        finally:
            dlg.close()

    def test_open_documentation_hub_focuses_help_tab(self):
        from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog
        from qgis.PyQt import QtWidgets
        dlg = SWE2DWorkbenchStudioDialog(iface=_make_iface())
        try:
            dock = QtWidgets.QDockWidget()
            tabs = QtWidgets.QTabWidget()
            tabs.addTab(QtWidgets.QWidget(), "Map")
            tabs.addTab(QtWidgets.QWidget(), "Help")
            dock.setWidget(tabs)
            dlg._state.studio_inspector_dock = dock
            dlg._open_documentation_hub()
            self.assertEqual(tabs.tabText(tabs.currentIndex()), "Help")
            self.assertTrue(dock.isVisible())
        finally:
            dlg.close()

    def test_run_dock_is_created(self):
        from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog
        dlg = SWE2DWorkbenchStudioDialog(iface=_make_iface())
        try:
            self.assertTrue(hasattr(dlg, "_run_dock"))
            self.assertIsNotNone(dlg._run_dock)
            self.assertTrue(hasattr(dlg._run_dock, "run_btn"))
            self.assertTrue(hasattr(dlg._run_dock, "cancel_btn"))
            self.assertTrue(hasattr(dlg._run_dock, "snapshot_btn"))
            self.assertTrue(hasattr(dlg._run_dock, "batch_btn"))
            self.assertTrue(hasattr(dlg._run_dock, "progress_bar"))
        finally:
            dlg.close()

    def test_remember_model_gpkg_tracks_recent_paths(self):
        from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog
        dlg = SWE2DWorkbenchStudioDialog(iface=_make_iface())
        try:
            dlg._remember_model_gpkg("/tmp/first.gpkg")
            dlg._remember_model_gpkg("/tmp/second.gpkg")
            dlg._remember_model_gpkg("/tmp/first.gpkg")
            self.assertEqual(dlg._recent_model_gpkgs, ["/tmp/first.gpkg", "/tmp/second.gpkg"])
        finally:
            dlg.close()

    def test_load_2d_model_geopackage_records_recent_path(self):
        from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog
        dlg = SWE2DWorkbenchStudioDialog(iface=_make_iface())
        try:
            dlg._mesh_controller = MagicMock()
            dlg._model_gpkg_path = "/tmp/model.gpkg"
            dlg._load_2d_model_geopackage()
            self.assertIn("/tmp/model.gpkg", dlg._recent_model_gpkgs)
            dlg._mesh_controller.load_2d_model_geopackage.assert_called_once()
        finally:
            dlg.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
