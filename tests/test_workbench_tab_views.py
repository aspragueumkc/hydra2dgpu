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
            "nodes_layer_combo", "cells_layer_combo", "terrain_layer_combo",
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
            "extended_outputs_chk", "save_mesh_chk", "save_line_chk",
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
            "preview_overrides_btn", "preview_coupling_btn",
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
        # Defaults: extended=True, save_mesh=True, save_max_only=False
        params = view.collect_storage_params()
        for key in (
            "extended_outputs_chk",
            "save_mesh_results_to_gpkg_chk",
            "save_line_results_to_gpkg_chk",
            "save_coupling_results_to_gpkg_chk",
            "save_max_only_chk",
            "save_run_log_to_gpkg_chk",
        ):
            self.assertIn(key, params)

    def test_run_output_buttons_have_parent(self):
        """The Preview / Load / Save buttons are wrapped in a QWidget row
        container that must be parented (added to the form layout).
        Otherwise the filter's setVisible(True) floats them as a
        top-level window — the orphan-window regression."""
        from swe2d.workbench.views.model_tab_view import ModelTabView
        view = ModelTabView()
        for attr in (
            "preview_overrides_btn", "preview_coupling_btn",
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
            "topo_export_template_btn",
            "topo_backend_combo", "topo_default_size_spin",
            "topo_default_cell_type_combo",
            "topo_controls_summary_lbl",
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
# ResultsToolbox tests
# ═══════════════════════════════════════════════════════════════════════════

class TestResultsToolbox(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_app()

    def test_extended_outputs_tooltip_is_not_truncated(self):
        # extended_outputs_chk moved to ModelTabView Output page.
        from swe2d.workbench.views.model_tab_view import ModelTabView
        view = ModelTabView()
        tip = view.extended_outputs_chk.toolTip()
        self.assertNotIn("...", tip)
        self.assertIn("Froude", tip)

    def test_results_toolbox_has_one_page(self):
        """Storage page was moved to ModelTabView (Output)."""
        from swe2d.workbench.views.results_controls import ResultsToolbox
        toolbox = ResultsToolbox()
        self.assertEqual(toolbox.toolbox.count(), 1)
        texts = [toolbox.toolbox.itemText(i) for i in range(toolbox.toolbox.count())]
        self.assertIn("Display", texts)
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
