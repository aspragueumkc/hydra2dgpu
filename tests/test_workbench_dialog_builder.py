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
            dlg = SWE2DWorkbenchStudioDialog(iface=MagicMock())
            try:
                MockBuilder.assert_called_once_with(dlg)
                mock_instance.configure.assert_called_once()
            finally:
                dlg.close()

    def test_init_sets_window_title(self):
        """The dialog sets its window title before the builder runs."""
        from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog
        dlg = SWE2DWorkbenchStudioDialog(iface=MagicMock())
        try:
            self.assertEqual(dlg.windowTitle(), "2D SWE Workbench (Studio)")
        finally:
            dlg.close()

    def test_init_creates_main_window(self):
        """The builder should create the main window during configure()."""
        from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog
        dlg = SWE2DWorkbenchStudioDialog(iface=MagicMock())
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
