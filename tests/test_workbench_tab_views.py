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
            "structures_layer_combo", "layer_group_combo",
            "autopop_group_btn", "refresh_layers_btn", "create_model_gpkg_btn",
        ):
            with self.subTest(attr=attr):
                self.assertTrue(
                    hasattr(view, attr),
                    f"MapTabView missing widget: {attr}",
                )

    def test_has_expected_action_widgets(self):
        from swe2d.workbench.views.map_tab_view import MapTabView
        view = MapTabView()
        for attr in (
            "load_model_gpkg_btn",
            "export_mesh_layers_btn",
            "import_mesh_layers_btn", "terrain_to_nodes_btn",
            "pull_node_z_btn",
        ):
            with self.subTest(attr=attr):
                self.assertTrue(
                    hasattr(view, attr),
                    f"MapTabView missing action widget: {attr}",
                )

    def test_has_expected_tools_widgets(self):
        from swe2d.workbench.views.map_tab_view import MapTabView
        view = MapTabView()
        for attr in (
            "open_model_gpkg_explorer_btn",
            "open_run_log_viewer_btn",
            "layer_status_lbl",
        ):
            with self.subTest(attr=attr):
                self.assertTrue(
                    hasattr(view, attr),
                    f"MapTabView missing tool widget: {attr}",
                )

    def test_map_actions_layout_is_form(self):
        from qgis.PyQt.QtWidgets import QFormLayout
        from swe2d.workbench.views.map_tab_view import MapTabView
        view = MapTabView()
        self.assertIsInstance(
            view.findChild(QFormLayout, "map_actions_layout"), QFormLayout
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
            "model_drain_page", "model_run_page",
            "n_mann_spin", "cfl_spin", "h_min_spin",
            "run_time_edit", "output_interval_edit",
            "run_btn", "cancel_btn", "snapshot_btn",
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
        self.assertGreaterEqual(view.model_toolbox.count(), 4)


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
        ):
            with self.subTest(attr=attr):
                self.assertTrue(
                    hasattr(view, attr),
                    f"TopologyTabView missing widget: {attr}",
                )

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
        from swe2d.workbench.views.results_controls import ResultsToolbox
        toolbox = ResultsToolbox()
        tip = toolbox.extended_outputs_chk.toolTip()
        self.assertNotIn("...", tip)
        self.assertIn("Froude", tip)

    def test_results_toolbox_has_two_pages(self):
        from swe2d.workbench.views.results_controls import ResultsToolbox
        toolbox = ResultsToolbox()
        self.assertEqual(toolbox.toolbox.count(), 2)
        texts = [toolbox.toolbox.itemText(i) for i in range(toolbox.toolbox.count())]
        self.assertIn("Display", texts)
        self.assertIn("Storage", texts)

    def test_arrow_children_disable_with_checkbox(self):
        from swe2d.workbench.views.results_controls import ResultsToolbox
        toolbox = ResultsToolbox()
        toolbox.arrows_chk.setChecked(True)
        self.assertTrue(toolbox.arrow_density_spin.isEnabled())
        toolbox.arrows_chk.setChecked(False)
        self.assertFalse(toolbox.arrow_density_spin.isEnabled())


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

    def test_build_studio_toolbar_creates_hydra_toolbar(self):
        from qgis.PyQt import QtWidgets
        from swe2d.workbench.views.studio_host_methods import _build_studio_toolbar
        host = QtWidgets.QMainWindow()
        dlg = self._make_mock_dialog()
        _build_studio_toolbar(None, dlg, host)
        toolbar = host.findChild(QtWidgets.QToolBar, "HydraRunToolbar")
        self.assertIsNotNone(toolbar)
        self.assertGreaterEqual(len(toolbar.actions()), 3)

    def test_build_studio_menu_creates_hydra_menu(self):
        from qgis.PyQt import QtWidgets
        from swe2d.workbench.views.studio_host_methods import _build_studio_menu
        host = QtWidgets.QMainWindow()
        dlg = self._make_mock_dialog()
        _build_studio_menu(None, dlg, host)
        menu = host.findChild(QtWidgets.QMenu, "HydraPluginMenu")
        self.assertIsNotNone(menu)
        self.assertGreaterEqual(len(menu.actions()), 3)

    def test_clear_studio_host_controls_removes_toolbar_and_menu(self):
        from qgis.PyQt import QtWidgets
        from swe2d.workbench.views import studio_host_methods
        from swe2d.workbench.views.studio_host_methods import (
            _build_studio_toolbar,
            _build_studio_menu,
            _clear_studio_host_controls,
        )
        host = QtWidgets.QMainWindow()
        dlg = self._make_mock_dialog()
        _build_studio_toolbar(None, dlg, host)
        _build_studio_menu(None, dlg, host)
        self.assertIsNotNone(studio_host_methods._SWE2D_STUDIO_HOST_TOOLBAR)
        self.assertIsNotNone(studio_host_methods._SWE2D_STUDIO_HOST_MENU)
        _clear_studio_host_controls(None, host)
        QtWidgets.QApplication.processEvents()
        self.assertIsNone(studio_host_methods._SWE2D_STUDIO_HOST_TOOLBAR)
        self.assertIsNone(studio_host_methods._SWE2D_STUDIO_HOST_MENU)
        self.assertIsNone(host.findChild(QtWidgets.QToolBar, "HydraRunToolbar"))
        self.assertIsNone(host.findChild(QtWidgets.QMenu, "HydraPluginMenu"))


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
