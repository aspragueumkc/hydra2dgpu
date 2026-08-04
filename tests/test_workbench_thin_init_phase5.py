"""Tests for Phase 5 Task 24: thin SWE2DWorkbenchStudioDialog.__init__.

The dialog's ``__init__`` must be a thin bootstrapper that:
1. Calls ``super().__init__(parent)``
2. Sets the window title
3. Stores the ``iface`` reference as the public ``self.iface``
4. Creates the central ``QMainWindow`` (``self._studio_main_window``)
5. Delegates all remaining state initialization to ``WorkbenchDialogBuilder``

The view-state attributes (components, docks, feature flags, overlay item,
persist suppression, runtime-log detached dialogs) live on a
``WorkbenchViewState`` dataclass accessed via ``self._state``.

Note: this file originally asserted an earlier Phase-5 spec (``self._iface``,
``studio_docks``, ``mesh_view_detached_dialogs`` state fields). Those were
deliberately removed from the codebase by later refactors; the assertions
here have been re-derived from the current architecture during the
real-QGIS test migration (2026-08-02).

Created as part of Phase 5 Task 24 (empty __init__).
"""
import inspect
import textwrap
import unittest
from typing import get_type_hints
from unittest.mock import MagicMock, patch

from tests.qgis_real_env import ensure_qgis_app, requires_qgis


@requires_qgis
class TestWorkbenchViewStateExists(unittest.TestCase):
    """The WorkbenchViewState dataclass must exist and hold the required fields."""

    @classmethod
    def setUpClass(cls):
        ensure_qgis_app()

    def test_view_state_class_importable(self):
        from swe2d.workbench.workbench_view_state import WorkbenchViewState
        self.assertIsNotNone(WorkbenchViewState)

    def test_view_state_has_all_required_fields(self):
        from swe2d.workbench.workbench_view_state import WorkbenchViewState
        hints = get_type_hints(WorkbenchViewState)
        required = {
            "iface",
            "studio_status_label",
            "studio_view_mode_combo",
            "studio_theme_combo",
            "studio_left_dock",
            "studio_inspector_dock",
            "studio_results_dock",
            "studio_components",
            "studio_feature_flags",
            "high_perf_canvas_overlay_item",
            "persist_suppressed",
            "runtime_log_detached_dialogs",
        }
        missing = required - set(hints.keys())
        self.assertEqual(
            missing,
            set(),
            f"WorkbenchViewState is missing required fields: {missing}",
        )

    def test_view_state_default_feature_flags(self):
        from swe2d.workbench.workbench_view_state import WorkbenchViewState
        state = WorkbenchViewState()
        self.assertEqual(
            state.studio_feature_flags,
            {
                "rainfall": True,
                "drainage_structures": True,
                "hydraulic_structures": True,
                "bridge_stacked_coupling": False,
                "gpu_viewer": False,
            },
        )

    def test_view_state_default_collections_are_independent(self):
        from swe2d.workbench.workbench_view_state import WorkbenchViewState
        s1 = WorkbenchViewState()
        s2 = WorkbenchViewState()
        s1.studio_components["foo"] = "bar"
        s1.runtime_log_detached_dialogs.append("x")
        self.assertNotIn("foo", s2.studio_components)
        self.assertNotIn("x", s2.runtime_log_detached_dialogs)

    def test_view_state_iface_defaults_to_none(self):
        from swe2d.workbench.workbench_view_state import WorkbenchViewState
        state = WorkbenchViewState()
        self.assertIsNone(state.iface)

    def test_view_state_iface_settable(self):
        from swe2d.workbench.workbench_view_state import WorkbenchViewState
        sentinel = object()
        state = WorkbenchViewState(iface=sentinel)
        self.assertIs(state.iface, sentinel)


@requires_qgis
class TestDialogInitIsThin(unittest.TestCase):
    """The dialog's __init__ must be a thin 4-line bootstrapper."""

    @classmethod
    def setUpClass(cls):
        ensure_qgis_app()

    def _init_body_lines(self):
        from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog
        source = inspect.getsource(SWE2DWorkbenchStudioDialog.__init__)
        return textwrap.dedent(source).strip().split("\n")

    def test_init_body_is_at_most_six_lines(self):
        """The body of __init__ must be <= 6 lines (allowing for def + docstring)."""
        body = self._init_body_lines()
        body_count = len(body) - 1
        self.assertLessEqual(
            body_count,
            6,
            f"__init__ body has {body_count} lines:\n{chr(10).join(body)}",
        )

    def test_init_calls_super(self):
        """__init__ must call super().__init__(parent)."""
        body = self._init_body_lines()
        joined = "\n".join(body)
        self.assertIn("super().__init__(parent)", joined)

    def test_init_stores_iface(self):
        """__init__ must store self.iface = iface (public, read by the builder)."""
        body = self._init_body_lines()
        joined = "\n".join(body)
        self.assertIn("self.iface = iface", joined)

    def test_init_calls_builder(self):
        """__init__ must delegate to WorkbenchDialogBuilder.configure()."""
        body = self._init_body_lines()
        joined = "\n".join(body)
        self.assertIn("WorkbenchDialogBuilder(self).configure()", joined)

    def test_init_has_no_studio_main_window_init(self):
        """__init__ must NOT bulk-init self._studio_main_window = None.

        The main window IS created directly in __init__ (current design:
        ``self._studio_main_window = QtWidgets.QMainWindow(self)``) — only
        the old ``= None`` placeholder form is forbidden.
        """
        body = self._init_body_lines()
        joined = "\n".join(body)
        self.assertNotIn("self._studio_main_window = None", joined)

    def test_init_has_no_studio_attribute_inits(self):
        """__init__ must NOT contain the 15 removed attribute inits."""
        body = self._init_body_lines()
        joined = "\n".join(body)
        forbidden = [
            "self._studio_main_window = None",
            "self._studio_status_label = None",
            "self._studio_view_mode_combo = None",
            "self._studio_theme_combo = None",
            "self._studio_left_dock = None",
            "self._studio_inspector_dock = None",
            "self._studio_results_dock = None",
            "self._studio_docks:",
            "self._studio_components:",
            "self._studio_feature_flags = {",
            "self._high_perf_canvas_overlay_item = None",
            "self._persist_suppressed = False",
            "self._runtime_log_detached_dialogs = []",
        ]
        for line in forbidden:
            self.assertNotIn(
                line,
                joined,
                f"__init__ still contains forbidden line: {line!r}",
            )

    def test_init_has_no_window_setup_calls(self):
        """__init__ must NOT contain resize/setModal/setWindowModality.

        ``setWindowTitle`` is allowed — the title is set directly in
        ``__init__`` under the current thin-bootstrapper design.
        """
        body = self._init_body_lines()
        joined = "\n".join(body)
        self.assertNotIn("self.resize(", joined)
        self.assertNotIn("self.setModal(", joined)
        self.assertNotIn("self.setWindowModality(", joined)


@requires_qgis
class TestDialogHoldsState(unittest.TestCase):
    """The dialog must hold a WorkbenchViewState instance."""

    @classmethod
    def setUpClass(cls):
        ensure_qgis_app()

    def _make_dialog_with_fake_builder(self, extra_state_setup=None, iface=None):
        """Construct a dialog with a stubbed builder that only sets state.

        Avoids the pre-existing dialog construction errors (legacy method
        migration gaps) by short-circuiting the builder's configure().
        """
        from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog
        from swe2d.workbench.workbench_view_state import WorkbenchViewState
        from swe2d.workbench import workbench_dialog_builder as wdb_module

        def fake_configure(builder_instance):
            dlg = builder_instance._dialog
            dlg._state = WorkbenchViewState(iface=dlg.iface)
            if extra_state_setup is not None:
                extra_state_setup(dlg)

        original_configure = wdb_module.WorkbenchDialogBuilder.configure
        wdb_module.WorkbenchDialogBuilder.configure = fake_configure
        try:
            dlg = SWE2DWorkbenchStudioDialog(
                iface=iface if iface is not None else MagicMock()
            )
        finally:
            wdb_module.WorkbenchDialogBuilder.configure = original_configure
        # Prevent close() from triggering UI cleanup that accesses state
        dlg.close = MagicMock()
        return dlg

    def test_dialog_constructor_calls_builder(self):
        """Calling __init__ must call WorkbenchDialogBuilder.configure()."""
        with patch(
            "swe2d.workbench.studio_dialog.WorkbenchDialogBuilder"
        ) as MockBuilder:
            mock_instance = MagicMock()
            mock_instance.configure.return_value = None
            MockBuilder.return_value = mock_instance
            from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog
            dlg = SWE2DWorkbenchStudioDialog(iface=MagicMock())
            dlg.close = MagicMock()
            try:
                MockBuilder.assert_called_once_with(dlg)
                mock_instance.configure.assert_called_once()
            finally:
                dlg.deleteLater()

    def test_state_is_workbench_view_state(self):
        from swe2d.workbench.workbench_view_state import WorkbenchViewState
        dlg = self._make_dialog_with_fake_builder()
        try:
            self.assertIsInstance(dlg._state, WorkbenchViewState)
        finally:
            dlg.deleteLater()

    def test_state_studio_components_empty_at_init(self):
        """After builder sets state, containers are empty until _build_ui runs."""
        dlg = self._make_dialog_with_fake_builder()
        try:
            self.assertEqual(dlg._state.studio_components, {})
            self.assertEqual(dlg._state.runtime_log_detached_dialogs, [])
        finally:
            dlg.deleteLater()

    def test_state_iface_propagates_from_dialog(self):
        dlg = self._make_dialog_with_fake_builder()
        try:
            self.assertIs(dlg._state.iface, dlg.iface)
        finally:
            dlg.deleteLater()

    def test_dialog_exposes_studio_main_window_directly(self):
        """The central QMainWindow is a direct dialog attribute by design."""
        from qgis.PyQt import QtWidgets

        dlg = self._make_dialog_with_fake_builder()
        try:
            self.assertIsInstance(dlg._studio_main_window, QtWidgets.QMainWindow)
        finally:
            dlg.deleteLater()

    def test_dialog_does_not_expose_studio_components_directly(self):
        dlg = self._make_dialog_with_fake_builder()
        try:
            with self.assertRaises(AttributeError):
                _ = dlg._studio_components
        finally:
            dlg.deleteLater()

    def test_dialog_does_not_expose_persist_suppressed_directly(self):
        dlg = self._make_dialog_with_fake_builder()
        try:
            with self.assertRaises(AttributeError):
                _ = dlg._persist_suppressed
        finally:
            dlg.deleteLater()

    def test_dialog_does_not_expose_studio_feature_flags_directly(self):
        dlg = self._make_dialog_with_fake_builder()
        try:
            with self.assertRaises(AttributeError):
                _ = dlg._studio_feature_flags
        finally:
            dlg.deleteLater()

    def test_dialog_does_not_expose_high_perf_canvas_overlay_item_directly(self):
        dlg = self._make_dialog_with_fake_builder()
        try:
            with self.assertRaises(AttributeError):
                _ = dlg._high_perf_canvas_overlay_item
        finally:
            dlg.deleteLater()

    def test_dialog_does_not_expose_detached_dialogs_directly(self):
        dlg = self._make_dialog_with_fake_builder()
        try:
            with self.assertRaises(AttributeError, msg="dlg._runtime_log_detached_dialogs"):
                _ = dlg._runtime_log_detached_dialogs
        finally:
            dlg.deleteLater()

    def test_dialog_iface_still_set_directly(self):
        """The dialog must expose the public ``iface`` attribute (read by the builder)."""
        sentinel = MagicMock(name="iface_sentinel")
        dlg = self._make_dialog_with_fake_builder(iface=sentinel)
        try:
            self.assertIs(dlg.iface, sentinel)
        finally:
            dlg.deleteLater()

    def test_dialog_does_not_expose_studio_view_mode_combo_directly(self):
        dlg = self._make_dialog_with_fake_builder()
        try:
            with self.assertRaises(AttributeError):
                _ = dlg._studio_view_mode_combo
        finally:
            dlg.deleteLater()

    def test_dialog_does_not_expose_studio_theme_combo_directly(self):
        dlg = self._make_dialog_with_fake_builder()
        try:
            with self.assertRaises(AttributeError):
                _ = dlg._studio_theme_combo
        finally:
            dlg.deleteLater()

    def test_dialog_does_not_expose_studio_left_dock_directly(self):
        dlg = self._make_dialog_with_fake_builder()
        try:
            with self.assertRaises(AttributeError):
                _ = dlg._studio_left_dock
        finally:
            dlg.deleteLater()

    def test_dialog_does_not_expose_studio_inspector_dock_directly(self):
        dlg = self._make_dialog_with_fake_builder()
        try:
            with self.assertRaises(AttributeError):
                _ = dlg._studio_inspector_dock
        finally:
            dlg.deleteLater()

    def test_dialog_does_not_expose_studio_results_dock_directly(self):
        dlg = self._make_dialog_with_fake_builder()
        try:
            with self.assertRaises(AttributeError):
                _ = dlg._studio_results_dock
        finally:
            dlg.deleteLater()

    def test_dialog_does_not_expose_studio_status_label_directly(self):
        dlg = self._make_dialog_with_fake_builder()
        try:
            with self.assertRaises(AttributeError):
                _ = dlg._studio_status_label
        finally:
            dlg.deleteLater()


if __name__ == "__main__":
    unittest.main(verbosity=2)
