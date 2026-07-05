"""Tests for Workbench tab view QWidget subclasses.

Verifies that each tab view:
1. Can be imported and instantiated without QGIS
2. Owns its expected widget references
3. Uses proper objectName attributes for findChild compatibility
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from qgis.PyQt.QtWidgets import QApplication

_app = None


def _ensure_app():
    global _app
    if _app is None:
        _app = QApplication.instance() or QApplication([])


# ═══════════════════════════════════════════════════════════════════════════
# MapTabView tests
# ═══════════════════════════════════════════════════════════════════════════

class TestMapTabView(unittest.TestCase):
    """MapTabView — Map tab view fixture."""

    @classmethod
    def setUpClass(cls):
        _ensure_app()

    def test_import_and_instantiate(self):
        from swe2d.workbench.views.map_tab_view import MapTabView
        view = MapTabView()
        self.assertIsNotNone(view)

    def test_has_expected_data_widgets(self):
        from swe2d.workbench.views.map_tab_view import MapTabView
        view = MapTabView()
        for attr in (
            "nodes_layer_combo", "cells_layer_combo",
            "manning_layer_combo", "cn_layer_combo", "rain_gage_layer_combo",
            "hyetograph_layer_combo", "sample_lines_layer_combo",
            "drain_nodes_layer_combo", "drain_links_layer_combo",
            "drain_inlets_layer_combo", "drain_node_inlets_layer_combo",
            "structures_layer_combo",
        ):
            with self.subTest(attr=attr):
                self.assertTrue(
                    hasattr(view, attr),
                    f"MapTabView missing widget: {attr}",
                )

    def test_has_no_action_widgets(self):
        """The Mesh Setup page moved to TopologyTabView as Import/Export."""
        from swe2d.workbench.views.map_tab_view import MapTabView
        view = MapTabView()
        for attr in (
            "export_mesh_layers_btn",
            "import_mesh_layers_btn",
            "export_mesh_ugrid_btn",
            "save_mesh_gpkg_btn",
            "load_mesh_gpkg_btn",
            "export_results_ugrid_btn",
        ):
            with self.subTest(attr=attr):
                self.assertFalse(
                    hasattr(view, attr),
                    f"MapTabView should not own {attr} anymore",
                )

    def test_has_expected_tools_widgets(self):
        from swe2d.workbench.views.map_tab_view import MapTabView
        view = MapTabView()
        for attr in (
            "open_model_gpkg_explorer_btn",
            "open_run_log_viewer_btn",
        ):
            with self.subTest(attr=attr):
                self.assertTrue(
                    hasattr(view, attr),
                    f"MapTabView missing tool widget: {attr}",
                )

    def test_is_qwidget_subclass(self):
        from qgis.PyQt import QtWidgets
        from swe2d.workbench.views.map_tab_view import MapTabView
        self.assertTrue(issubclass(MapTabView, QtWidgets.QWidget))


# ═══════════════════════════════════════════════════════════════════════════
# ModelTabView tests
# ═══════════════════════════════════════════════════════════════════════════

class TestModelTabView(unittest.TestCase):
    """ModelTabView — Model tab view fixture."""

    @classmethod
    def setUpClass(cls):
        _ensure_app()

    def test_import_and_instantiate(self):
        from swe2d.workbench.views.model_tab_view import ModelTabView
        view = ModelTabView()
        self.assertIsNotNone(view)

    def test_has_expected_widgets(self):
        from swe2d.workbench.views.model_tab_view import ModelTabView
        view = ModelTabView()
        for attr in (
            "model_toolbox",
            "model_solver_page", "model_rain_page",
            "model_drain_page", "model_output_page",
            "n_mann_spin", "cfl_spin", "h_min_spin",
            "run_time_edit",
            "save_mesh_chk", "save_line_chk",
            "save_coupling_chk", "save_max_only_chk", "save_log_chk",
        ):
            with self.subTest(attr=attr):
                self.assertTrue(
                    hasattr(view, attr),
                    f"ModelTabView missing widget: {attr}",
                )

    def test_is_qwidget_subclass(self):
        from qgis.PyQt import QtWidgets
        from swe2d.workbench.views.model_tab_view import ModelTabView
        self.assertTrue(issubclass(ModelTabView, QtWidgets.QWidget))

    def test_has_toolbox_with_pages(self):
        from swe2d.workbench.views.model_tab_view import ModelTabView
        view = ModelTabView()
        self.assertIsNotNone(view.model_toolbox)
        # 4 original pages + 1 Output page (moved from ResultsToolbox)
        self.assertGreaterEqual(view.model_toolbox.count(), 5)

    def test_output_page_is_bottom(self):
        """The Output page must be the last (bottom) page in the model toolbox."""
        from swe2d.workbench.views.model_tab_view import ModelTabView
        view = ModelTabView()
        last_idx = view.model_toolbox.count() - 1
        self.assertEqual(
            view.model_toolbox.itemText(last_idx), "Output"
        )
        self.assertIs(
            view.model_toolbox.widget(last_idx), view.model_output_page
        )

    def test_output_page_has_run_output_widgets(self):
        """The Run Output widgets (moved from below the Run dock progress
        bar) live on the Model tab's Output page."""
        from swe2d.workbench.views.model_tab_view import ModelTabView
        view = ModelTabView()
        for attr in (
            "output_interval_edit", "line_output_interval_edit",
            "results_table_name_edit", "results_gpkg_path_edit",
            "select_results_gpkg_btn", "load_run_settings_btn", "save_settings_btn",
        ):
            with self.subTest(attr=attr):
                self.assertTrue(
                    hasattr(view, attr),
                    f"ModelTabView missing moved run-output widget: {attr}",
                )

    def test_collect_storage_params_matches_legacy_schema(self):
        """collect_storage_params must produce the legacy key set expected
        by run_controller and batch_simulation_dialog."""
        from swe2d.workbench.views.model_tab_view import ModelTabView
        view = ModelTabView()
        # Defaults: save_mesh=True, save_max_only=False
        params = view.collect_storage_params()
        for key in (
            "save_mesh_results_to_gpkg_chk",
            "save_line_results_to_gpkg_chk",
            "save_coupling_results_to_gpkg_chk",
            "save_max_only_chk",
            "save_run_log_to_gpkg_chk",
        ):
            self.assertIn(key, params)

    def test_run_output_buttons_have_parent(self):
        """The Load / Save Config buttons must be parented (added to the
        form layout) so filter's setVisible(True) doesn't float them as
        a top-level orphan window."""
        from swe2d.workbench.views.model_tab_view import ModelTabView
        view = ModelTabView()
        for attr in (
            "load_run_settings_btn", "save_settings_btn",
        ):
            with self.subTest(attr=attr):
                btn = getattr(view, attr)
                self.assertIsNotNone(
                    btn.parent(),
                    f"{attr} has no parent — would float as orphan window",
                )
                self.assertFalse(
                    btn.isWindow(),
                    f"{attr} is a top-level window — orphan regression",
                )

    def test_save_settings_keyboard_shortcut_targets_controller(self):
        """Regression: Ctrl+S used to call dlg._run_dock.save_settings_btn
        .click() but save_settings_btn moved off the run dock. The
        shortcut must now call the controller method directly."""
        import inspect
        from swe2d.workbench import studio_dialog
        src = inspect.getsource(studio_dialog)
        self.assertNotIn(
            "_run_dock.save_settings_btn",
            src,
            "Stale keyboard shortcut still references _run_dock.save_settings_btn",
        )


# ═══════════════════════════════════════════════════════════════════════════
# TopologyTabView tests
# ═══════════════════════════════════════════════════════════════════════════

class TestTopologyTabView(unittest.TestCase):
    """TopologyTabView — Topology tab view fixture."""

    @classmethod
    def setUpClass(cls):
        _ensure_app()

    def test_import_and_instantiate(self):
        from swe2d.workbench.views.topology_tab_view import TopologyTabView
        view = TopologyTabView()
        self.assertIsNotNone(view)

    def test_has_expected_widgets(self):
        from swe2d.workbench.views.topology_tab_view import TopologyTabView
        view = TopologyTabView()
        for attr in (
            "topo_nodes_combo", "topo_arcs_combo", "topo_regions_combo",
            "topo_constraints_combo", "topo_quad_edges_combo",
            "topo_backend_combo", "topo_default_size_spin",
            "topo_default_cell_type_combo",
            "topo_generate_btn",
            "topo_terminate_btn",
            # Import/Export page — moved from Map tab
            "export_mesh_layers_btn", "import_mesh_layers_btn",
            "export_mesh_ugrid_btn", "save_mesh_gpkg_btn",
            "load_mesh_gpkg_btn", "export_results_ugrid_btn",
        ):
            with self.subTest(attr=attr):
                self.assertTrue(
                    hasattr(view, attr),
                    f"TopologyTabView missing widget: {attr}",
                )

    def test_import_export_page_is_top_of_toolbox(self):
        """The Import/Export page must be the first (top) page in the
        topology toolbox, before Layer Setup."""
        from swe2d.workbench.views.topology_tab_view import TopologyTabView
        view = TopologyTabView()
        self.assertEqual(view._toolbox.itemText(0), "Import/Export")
        # The page that follows must be Layer Setup
        self.assertEqual(view._toolbox.itemText(1), "Layer Setup")

    def test_is_qwidget_subclass(self):
        from qgis.PyQt import QtWidgets
        from swe2d.workbench.views.topology_tab_view import TopologyTabView
        self.assertTrue(issubclass(TopologyTabView, QtWidgets.QWidget))

    def test_set_callbacks_does_not_crash(self):
        from swe2d.workbench.views.topology_tab_view import TopologyTabView
        view = TopologyTabView()
        view.set_callbacks(log_fn=MagicMock(), combo_layer_fn=MagicMock())
        self.assertIsNotNone(view._log_fn)
        self.assertIsNotNone(view._combo_layer_fn)

    def test_update_control_summary_does_not_crash(self):
        from swe2d.workbench.views.topology_tab_view import TopologyTabView
        view = TopologyTabView()
        view.set_callbacks(log_fn=MagicMock(), combo_layer_fn=MagicMock())
        view.update_control_summary()

    def test_get_topo_widget_value_returns_none_for_unknown(self):
        from swe2d.workbench.views.topology_tab_view import TopologyTabView
        view = TopologyTabView()
        result = view.get_topo_widget_value("nonexistent_attr")
        self.assertIsNone(result)


# ═══════════════════════════════════════════════════════════════════════════
# Import/Export combo dispatch tests
# ═══════════════════════════════════════════════════════════════════════════


class TestImportExportComboDispatch(unittest.TestCase):
    """The Import/Export page replaces a 6-button stack with two combos
    + Run buttons. Selecting an option + clicking Run must fire the
    matching underlying QPushButton's clicked signal so the existing
    ``wire_topology_tab_static_signals`` keeps working unchanged."""

    @classmethod
    def setUpClass(cls):
        _ensure_app()

    def _make_view(self):
        from swe2d.workbench.views.topology_tab_view import TopologyTabView
        return TopologyTabView()

    def test_import_combo_has_both_load_options(self):
        view = self._make_view()
        self.assertEqual(view.import_combo.count(), 2)
        # Each item's userData is the underlying QPushButton.
        for i in range(view.import_combo.count()):
            with self.subTest(i=i):
                data = view.import_combo.itemData(i)
                self.assertIsNotNone(data, f"import_combo item {i} has no userData")
                self.assertTrue(
                    hasattr(data, "click"),
                    f"import_combo item {i} userData is not a button",
                )

    def test_export_combo_has_four_export_options(self):
        view = self._make_view()
        self.assertEqual(view.export_combo.count(), 4)
        for i in range(view.export_combo.count()):
            with self.subTest(i=i):
                data = view.export_combo.itemData(i)
                self.assertIsNotNone(data)
                self.assertTrue(hasattr(data, "click"))

    def test_import_combo_userdata_maps_to_known_buttons(self):
        view = self._make_view()
        # Order matches _build_import_export_page: import_mesh_layers_btn, load_mesh_gpkg_btn.
        self.assertIs(
            view.import_combo.itemData(0), view.import_mesh_layers_btn,
        )
        self.assertIs(
            view.import_combo.itemData(1), view.load_mesh_gpkg_btn,
        )

    def test_export_combo_userdata_maps_to_known_buttons(self):
        view = self._make_view()
        # Order matches _build_import_export_page.
        self.assertIs(
            view.export_combo.itemData(0), view.export_mesh_layers_btn,
        )
        self.assertIs(
            view.export_combo.itemData(1), view.export_mesh_ugrid_btn,
        )
        self.assertIs(
            view.export_combo.itemData(2), view.save_mesh_gpkg_btn,
        )
        self.assertIs(
            view.export_combo.itemData(3), view.export_results_ugrid_btn,
        )

    def test_run_import_dispatches_to_selected_button(self):
        view = self._make_view()
        # Spy on the underlying button click.
        fired = []
        view.import_mesh_layers_btn.clicked.connect(
            lambda *_: fired.append("import_mesh_layers_btn")
        )
        # Default combo selection is 0 → import_mesh_layers_btn.
        view._run_selected_import()
        self.assertEqual(fired, ["import_mesh_layers_btn"])

    def test_run_export_dispatches_to_selected_button(self):
        view = self._make_view()
        fired = []
        view.export_mesh_ugrid_btn.clicked.connect(
            lambda *_: fired.append("export_mesh_ugrid_btn")
        )
        view.export_combo.setCurrentIndex(1)
        view._run_selected_export()
        self.assertEqual(fired, ["export_mesh_ugrid_btn"])

    def test_run_import_button_dispatches_through_combo(self):
        """Clicking 'Run Import' must fire the QPushButton backing the
        currently-selected Import combo item. We verify the end-to-end
        dispatch by attaching a spy to the underlying button's clicked
        signal and asserting it fires after Run Import is clicked.
        """
        view = self._make_view()
        fired = []
        view.import_mesh_layers_btn.clicked.connect(
            lambda *_: fired.append("import_mesh_layers_btn")
        )
        view.load_mesh_gpkg_btn.clicked.connect(
            lambda *_: fired.append("load_mesh_gpkg_btn")
        )
        # Default selection is index 0 → import_mesh_layers_btn.
        view.run_import_btn.click()
        self.assertEqual(fired, ["import_mesh_layers_btn"])
        # Switch to index 1 → load_mesh_gpkg_btn.
        view.import_combo.setCurrentIndex(1)
        view.run_import_btn.click()
        self.assertEqual(fired, [
            "import_mesh_layers_btn", "load_mesh_gpkg_btn",
        ])

    def test_run_export_button_dispatches_through_combo(self):
        view = self._make_view()
        fired = []
        for attr in (
            "export_mesh_layers_btn", "export_mesh_ugrid_btn",
            "save_mesh_gpkg_btn", "export_results_ugrid_btn",
        ):
            getattr(view, attr).clicked.connect(
                lambda *_, a=attr: fired.append(a)
            )
        for i, expected in enumerate([
            "export_mesh_layers_btn", "export_mesh_ugrid_btn",
            "save_mesh_gpkg_btn", "export_results_ugrid_btn",
        ]):
            view.export_combo.setCurrentIndex(i)
            view.run_export_btn.click()
            self.assertEqual(fired[-1], expected)

    def test_underlying_mesh_io_buttons_preserved_for_wiring(self):
        """The 6 original mesh-I/O buttons must still exist as instance
        attributes (with their objectNames) so the existing wiring in
        ``wire_topology_tab_static_signals`` (``v.export_mesh_layers_btn``,
        etc.) keeps working unchanged. The page layout changes; the
        controller-facing surface does not.
        """
        view = self._make_view()
        for attr, expected_objname in [
            ("import_mesh_layers_btn", "import_mesh_layers_btn"),
            ("load_mesh_gpkg_btn", "load_mesh_gpkg_btn"),
            ("export_mesh_layers_btn", "export_mesh_layers_btn"),
            ("export_mesh_ugrid_btn", "export_mesh_ugrid_btn"),
            ("save_mesh_gpkg_btn", "save_mesh_gpkg_btn"),
            ("export_results_ugrid_btn", "export_results_ugrid_btn"),
        ]:
            with self.subTest(attr=attr):
                btn = getattr(view, attr)
                self.assertEqual(btn.objectName(), expected_objname)


# ═══════════════════════════════════════════════════════════════════════════
# ResultsToolbox tests
# ═══════════════════════════════════════════════════════════════════════════

class TestResultsToolbox(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_app()

    def test_results_toolbox_has_overlay_and_runs_pages(self):
        """Overlay page holds overlay controls; Runs page holds the run list.
        Storage page was moved to ModelTabView (Output)."""
        from swe2d.workbench.views.results_controls import ResultsToolbox
        toolbox = ResultsToolbox()
        self.assertEqual(toolbox.toolbox.count(), 2)
        texts = [toolbox.toolbox.itemText(i) for i in range(toolbox.toolbox.count())]
        self.assertIn("Overlay", texts)
        self.assertIn("Runs", texts)
        self.assertNotIn("Storage", texts)

    def test_arrow_children_disable_with_checkbox(self):
        from swe2d.workbench.views.results_controls import ResultsToolbox
        toolbox = ResultsToolbox()
        toolbox.arrows_chk.setChecked(True)
        self.assertTrue(toolbox.arrow_density_spin.isEnabled())
        toolbox.arrows_chk.setChecked(False)
        self.assertFalse(toolbox.arrow_density_spin.isEnabled())

    def test_no_overarching_gpkg_label_in_runs_section(self):
        """The Runs section must not expose a 'Currently loaded GeoPackage path'
        label — results are loaded from one or more GeoPackages via 'Add
        Results', not from a single overarching model GPKG path."""
        from swe2d.workbench.views.results_controls import ResultsToolbox
        toolbox = ResultsToolbox()
        self.assertFalse(
            hasattr(toolbox, "gpkg_lbl"),
            "ResultsToolbox.gpkg_lbl exposes a single 'current GPKG' concept "
            "that does not match the multi-GPKG Add Results workflow.",
        )


# ═══════════════════════════════════════════════════════════════════════════
# StudioTabBuilder helper tests
# ═══════════════════════════════════════════════════════════════════════════

class TestStudioTabBuilderHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_app()

    def test_size_button_sets_minimum_action_size(self):
        from qgis.PyQt.QtWidgets import QPushButton
        from swe2d.workbench.views.studio_tab_builder import _size_button
        btn = QPushButton("Test")
        _size_button(btn, "action")
        self.assertGreaterEqual(btn.minimumSize().width(), 80)
        self.assertGreaterEqual(btn.minimumSize().height(), 28)


# ═══════════════════════════════════════════════════════════════════════════
# Studio host toolbar/menu tests
# ═══════════════════════════════════════════════════════════════════════════

class TestStudioHostToolbarMenu(unittest.TestCase):
    """Verify global toolbar and Plugins menu installation/cleanup."""

    @classmethod
    def setUpClass(cls):
        _ensure_app()

    def _make_mock_dialog(self):
        """Return a mock dialog with the callbacks toolbar/menu need."""
        dlg = MagicMock()
        dlg._controller = MagicMock()
        dlg._overlay_controller = MagicMock()
        dlg._mesh_controller = MagicMock()
        dlg._recent_model_gpkgs = []
        return dlg

    def test_install_studio_host_controls_sets_view_combo_corner_widget(self):
        """The view-mode combo is placed in the menu bar's top-right corner."""
        from qgis.PyQt import QtCore, QtWidgets
        from swe2d.workbench.views.studio_host_methods import (
            _install_studio_host_controls,
        )
        host = QtWidgets.QMainWindow()
        host.menuBar()  # ensure menuBar is created
        dlg = self._make_mock_dialog()
        dlg.view_mode_combo = QtWidgets.QComboBox()
        _install_studio_host_controls(None, dlg, host)
        corner = host.menuBar().cornerWidget()
        self.assertIsNotNone(corner)
        self.assertIsInstance(corner, QtWidgets.QComboBox)

    def test_clear_studio_host_controls_is_idempotent_noop(self):
        """_clear_studio_host_controls is a no-op — toolbar and HYDRA menu
        are owned by hydra_plugin, not the workbench."""
        from qgis.PyQt import QtWidgets
        from swe2d.workbench.views.studio_host_methods import (
            _clear_studio_host_controls,
        )
        host = QtWidgets.QMainWindow()
        host.menuBar()
        # Must not raise.
        _clear_studio_host_controls(None, host)


# ═══════════════════════════════════════════════════════════════════════════
# Workbench main menu / QGIS shortcut manager tests
# ═══════════════════════════════════════════════════════════════════════════


class TestWorkbenchMainMenuShortcuts(unittest.TestCase):
    """Verify the workbench main menu actions are registered with QGIS's
    shortcut manager under the "HYDRA2DGPU" section, with no default
    shortcut assigned (user binds their own in Settings → Keyboard
    Shortcuts)."""

    @classmethod
    def setUpClass(cls):
        _ensure_app()

    def _make_mock_dialog(self):
        dlg = MagicMock()
        dlg._controller = MagicMock()
        dlg._overlay_controller = MagicMock()
        dlg._mesh_controller = MagicMock()
        dlg._topology_controller = MagicMock()
        dlg._recent_model_gpkgs = []
        return dlg

    def _make_iface_with_menu_bar(self):
        """Build a real QMainWindow + a MagicMock iface whose
        mainWindow() returns it, so install_workbench_main_menu can
        resolve a real menu bar."""
        from qgis.PyQt import QtWidgets
        host = QtWidgets.QMainWindow()
        host.menuBar()
        iface = MagicMock()
        iface.mainWindow = lambda: host
        return iface, host

    def test_menu_actions_have_no_default_shortcut(self):
        """No workbench menu action is hard-coded with a key sequence —
        all shortcuts come from the user's Settings → Keyboard Shortcuts
        binding via the QGIS shortcut manager."""
        from swe2d.workbench.views.workbench_main_menu import (
            install_workbench_main_menu,
            remove_workbench_main_menu,
        )
        iface, _host = self._make_iface_with_menu_bar()
        dlg = self._make_mock_dialog()
        try:
            menu = install_workbench_main_menu(dlg, iface)
            self.assertIsNotNone(menu)
            # Iterate every action on the menu and confirm none has a
            # hard-coded shortcut (defaultShortcut would have been set
            # via setShortcut during construction).
            for action in menu.actions():
                if action.isSeparator():
                    continue
                seq = action.shortcut()
                self.assertTrue(
                    seq.isEmpty(),
                    f"Action {action.objectName()} has a default shortcut "
                    f"({seq.toString()}) — should be empty so the user "
                    f"binds it via Settings → Keyboard Shortcuts.",
                )
        finally:
            remove_workbench_main_menu(iface)

    def test_menu_actions_are_registered_with_qgis_shortcut_manager(self):
        """Every non-separator menu action is registered with
        QgisGui.shortcutsManager() under the 'HYDRA2DGPU' section.

        Skips gracefully when running under the test-only mock QGIS
        environment (which doesn't provide qgis.gui.QgsGui). The
        install path still registers in real QGIS — see the smoke test
        under mamba run -n qgis_stable.
        """
        from swe2d.workbench.views.workbench_main_menu import (
            install_workbench_main_menu,
            remove_workbench_main_menu,
        )
        iface, _host = self._make_iface_with_menu_bar()
        dlg = self._make_mock_dialog()
        try:
            menu = install_workbench_main_menu(dlg, iface)
            self.assertIsNotNone(menu)
            try:
                from qgis.gui import QgsGui
            except ImportError:
                self.skipTest(
                    "qgis.gui.QgsGui not available in this test environment"
                )
            manager = QgsGui.shortcutsManager()
            registered = set()
            for action in menu.actions():
                if action.isSeparator():
                    continue
                # actionByName uses the action's objectName — verify it
                # resolves back to the same action object.
                found = manager.actionByName(action.objectName())
                if found is action:
                    registered.add(action.objectName())
            # We expect every HYDRA2DMenu* action to be registered.
            expected = {
                a.objectName() for a in menu.actions()
                if not a.isSeparator() and a.objectName().startswith("HYDRA2DMenu")
            }
            missing = expected - registered
            self.assertEqual(
                missing, set(),
                f"Actions not registered with QGIS shortcut manager: {missing}",
            )
        finally:
            remove_workbench_main_menu(iface)

    def test_stale_keyboard_shortcuts_module_removed(self):
        """Regression: the module-level KEYBOARD_SHORTCUTS list and the
        QShortcut-based _install_keyboard_shortcuts() method were
        removed in favor of QGIS's shortcut manager. Make sure they
        don't sneak back in."""
        import inspect
        from swe2d.workbench import studio_dialog
        from swe2d.workbench.views import workbench_main_menu

        # Module-level KEYBOARD_SHORTCUTS attribute must not exist.
        self.assertFalse(
            hasattr(studio_dialog, "KEYBOARD_SHORTCUTS"),
            "studio_dialog.KEYBOARD_SHORTCUTS should be removed — use "
            "QGIS shortcut manager via workbench_main_menu actions.",
        )
        # _install_keyboard_shortcuts method must not exist on the dialog.
        self.assertFalse(
            hasattr(studio_dialog.SWE2DWorkbenchStudioDialog, "_install_keyboard_shortcuts"),
            "_install_keyboard_shortcuts() bypasses QGIS's shortcut "
            "manager — use workbench_main_menu actions instead.",
        )
        # No setShortcut(...) calls remain in workbench_main_menu —
        # the only shortcut registration path is via
        # _register_with_qgis_shortcut_manager → registerAction('').
        src = inspect.getsource(workbench_main_menu)
        self.assertNotIn(
            "act.setShortcut(",
            src,
            "workbench_main_menu should not call act.setShortcut(...) — "
            "use the QGIS shortcut manager with defaultShortcut=''.",
        )


# ═══════════════════════════════════════════════════════════════════════════
# Doc search tests
# ═══════════════════════════════════════════════════════════════════════════

class TestDocViewer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_app()

    def test_search_returns_snippets(self):
        from swe2d.workbench.views.doc_viewer import _search_all_docs
        results = _search_all_docs("solver")
        self.assertIsInstance(results, dict)
        for hits in results.values():
            for hit in hits:
                self.assertTrue(len(hit.snippet) > 0)


# ═══════════════════════════════════════════════════════════════════════════
# Run selection dialog tests
# ═══════════════════════════════════════════════════════════════════════════

class TestRunSelectionDialog(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_app()

    def _fake_records(self):
        class FakeRecord:
            __slots__ = (
                "run_id",
                "gpkg_path",
                "color",
                "enabled",
                "label",
                "key",
                "created_utc",
            )

            def __init__(self, run_id, gpkg_path, color, enabled, label, key, created_utc):
                self.run_id = run_id
                self.gpkg_path = gpkg_path
                self.color = color
                self.enabled = enabled
                self.label = label
                self.key = key
                self.created_utc = created_utc

            def display_label(self):
                return self.label

        return [
            FakeRecord(
                "run_a",
                "/tmp/a.gpkg",
                (0, 0, 0),
                True,
                "run_a",
                "/tmp/a.gpkg::run_a",
                "2026-07-01T00:00:00",
            ),
            FakeRecord(
                "run_b",
                "/tmp/a.gpkg",
                (0, 0, 0),
                True,
                "run_b",
                "/tmp/a.gpkg::run_b",
                "2026-07-02T00:00:00",
            ),
        ]

    def test_invert_selection_toggles_all(self):
        from swe2d.workbench.dialogs.run_selection_dialog import RunSelectionDialog
        dlg = RunSelectionDialog(self._fake_records())
        dlg._select_all()
        dlg._invert_selection()
        self.assertEqual(dlg.selected_keys(), set())

    def test_only_newest_selects_latest(self):
        from swe2d.workbench.dialogs.run_selection_dialog import RunSelectionDialog
        dlg = RunSelectionDialog(self._fake_records())
        dlg._select_only_newest()
        self.assertEqual(dlg.selected_keys(), {"/tmp/a.gpkg::run_b"})


# ═══════════════════════════════════════════════════════════════════════════
# Coupling results dialog tests
# ═══════════════════════════════════════════════════════════════════════════

class TestCouplingResultsDialog(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_app()

    def test_splitter_defaults_favor_plot(self):
        from qgis.PyQt import QtWidgets
        from swe2d.workbench.dialogs.coupling_results_dialog import (
            SWE2DCouplingResultsViewerDialog,
        )

        dlg = SWE2DCouplingResultsViewerDialog([], "run", "/tmp/x.gpkg")
        try:
            sizes = dlg.findChild(QtWidgets.QSplitter).sizes()
            self.assertGreater(sizes[1], sizes[0])
        finally:
            dlg.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
