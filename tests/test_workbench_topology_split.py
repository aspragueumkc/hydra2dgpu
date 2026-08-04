"""TDD tests for Phase 4 Task 20: Separate UI from logic in topology_and_io_methods.py.

Tests verify that:
1. _build_topology_tab_controls creates widgets and returns a dict
2. _wire_topology_tab_controls connects signals without errors
"""

from __future__ import annotations

import unittest

from tests.qgis_real_env import ensure_qgis_app, requires_qgis

from qgis.PyQt.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
    QCheckBox,
    QLineEdit,
)


class _TopologyTestBase(unittest.TestCase):
    """Base: real headless QgsApplication + a fresh parent widget per test."""

    @classmethod
    def setUpClass(cls):
        cls._app = ensure_qgis_app()

    def setUp(self):
        self.parent = QWidget()
        self.parent.setLayout(QVBoxLayout())

    def tearDown(self):
        self.parent.deleteLater()

    def _build_controls(self):
        from swe2d.workbench.views.topology_tab_view import (
            _build_topology_tab_controls,
        )
        return _build_topology_tab_controls(
            parent=self.parent,
            topology_tab_page=self.parent,
        )


@requires_qgis
class TestBuildTopologyTabControls(_TopologyTestBase):
    """Tests for _build_topology_tab_controls — pure view widget creation."""

    def _widget_names_in_dict(self, d):
        """Return set of object names from widgets in dict."""
        names = set()
        for key, val in d.items():
            if hasattr(val, 'objectName'):
                try:
                    names.add(val.objectName())
                except RuntimeError:
                    pass
        return names

    def test_returns_dict(self):
        """_build_topology_tab_controls returns a dict."""
        result = self._build_controls()
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)

    def test_contains_expected_gmsh_widgets(self):
        """Dict contains all critical gmsh control widget names."""
        result = self._build_controls()
        widget_names = self._widget_names_in_dict(result)
        expected = {
            "topo_backend_combo",
            "topo_default_cell_type_combo",
            "topo_default_size_spin",
            "topo_gmsh_tri_algo_combo",
            "topo_gmsh_quad_algo_combo",
            "topo_gmsh_recombine_algo_combo",
            "topo_gmsh_global_recombine_chk",
            "topo_gmsh_quality_enable_chk",
            "topo_gmsh_quality_max_iters_spin",
            "topo_gmsh_quality_time_limit_spin",
            "topo_quality_min_angle_spin",
            "topo_quality_max_aspect_spin",
            "topo_quality_max_non_orth_spin",
            "topo_quality_min_area_edit",
            "topo_quality_size_scales_edit",
            "topo_quality_smooth_increments_edit",
            "topo_gmsh_quality_recombine_topology_passes_edit",
            "topo_gmsh_quality_recombine_min_quality_edit",
            "topo_gmsh_quality_random_factors_edit",
            "topo_gmsh_quality_optimize_methods_edit",
            "topo_gmsh_algo_switch_on_failure_chk",
            "topo_gmsh_recombine_node_repositioning_chk",
        }
        missing = expected - widget_names
        self.assertFalse(
            missing,
            f"Missing widget objectNames: {missing}"
        )

    def test_contains_quality_form_widgets(self):
        """Quality form widgets are in the controls widget."""
        result = self._build_controls()
        widget_names = self._widget_names_in_dict(result)
        expected = {
            "topo_quality_min_angle_spin",
            "topo_quality_max_aspect_spin",
            "topo_quality_max_non_orth_spin",
            "topo_quality_min_area_edit",
            "topo_quality_size_scales_edit",
            "topo_quality_smooth_increments_edit",
            "topo_quality_strict_chk",
        }
        missing = expected - widget_names
        self.assertFalse(missing, f"Missing quality widgets: {missing}")

    def test_contains_interface_widgets(self):
        """Interface transition control widgets are present."""
        result = self._build_controls()
        widget_names = self._widget_names_in_dict(result)
        expected = {
            "topo_gmsh_interface_transition_enable_chk",
            "topo_gmsh_interface_transition_dist_factor_spin",
            "topo_gmsh_interface_transition_min_ratio_spin",
        }
        missing = expected - widget_names
        self.assertFalse(missing, f"Missing interface widgets: {missing}")

    def test_widgets_are_proper_types(self):
        """Each widget in the dict is the expected Qt type."""
        result = self._build_controls()
        type_map = {
            "topo_backend_combo": QComboBox,
            "topo_default_cell_type_combo": QComboBox,
            "topo_default_size_spin": QDoubleSpinBox,
            "topo_gmsh_tri_algo_combo": QComboBox,
            "topo_gmsh_smoothing_spin": QSpinBox,
            "topo_gmsh_verbosity_spin": QSpinBox,
            "topo_gmsh_global_recombine_chk": QCheckBox,
            "topo_quality_min_area_edit": QLineEdit,
            "topo_quality_size_scales_edit": QLineEdit,
            "topo_gmsh_quality_enable_chk": QCheckBox,
            "topo_gmsh_quality_max_iters_spin": QSpinBox,
            "topo_gmsh_quality_time_limit_spin": QDoubleSpinBox,
            "topo_gmsh_mesh_size_min_spin": QDoubleSpinBox,
        }
        for name, expected_type in type_map.items():
            w = result.get(name)
            self.assertIsNotNone(
                w, f"Widget '{name}' missing from result dict"
            )
            self.assertIsInstance(
                w, expected_type,
                f"Widget '{name}' expected {expected_type.__name__}, got {type(w).__name__}"
            )

    def test_backend_combo_shows_gmsh_recommended(self):
        """Backend combo always shows 'Gmsh (recommended)' — gmsh availability is no longer gating."""
        result = self._build_controls()
        backend_combo = result.get("topo_backend_combo")
        self.assertIsNotNone(backend_combo)
        first_text = backend_combo.itemText(0).lower()
        self.assertIn("recommended", first_text)


@requires_qgis
class TestWireTopologyTabControls(_TopologyTestBase):
    """Tests for _wire_topology_tab_controls — controller signal wiring."""

    def _make_toy_widgets(self):
        result = self._build_controls()
        # Keep reference to parent to prevent GC
        self._parent_ref = self.parent
        return result

    def _make_controller(self):
        from unittest.mock import MagicMock

        from swe2d.workbench.controllers.topology_controller import (
            TopologyController,
        )
        return TopologyController(MagicMock(name="TopologyMeshView"))

    def test_wire_does_not_crash(self):
        widgets = self._make_toy_widgets()
        update_called = [False]

        def update_summary():
            update_called[0] = True

        self._make_controller()._wire_topology_tab_controls(
            widgets=widgets,
            update_summary_fn=update_summary,
        )
        self.assertTrue(True, "Signal wiring completed without error")

    def test_update_summary_called_after_wire(self):
        """The update summary function is called at the end of wiring."""
        widgets = self._make_toy_widgets()
        update_called = [False]

        def update_summary():
            update_called[0] = True

        self._make_controller()._wire_topology_tab_controls(
            widgets=widgets,
            update_summary_fn=update_summary,
        )
        self.assertTrue(
            update_called[0],
            "update_summary_fn should be called after wiring",
        )

    def test_wire_with_empty_widgets_does_not_raise(self):
        """Wiring with empty widget dict should not crash (graceful skip)."""
        self._make_controller()._wire_topology_tab_controls(
            widgets={},
            update_summary_fn=lambda: None,
        )
        self.assertTrue(True, "Empty wiring completed without error")


if __name__ == "__main__":
    unittest.main(verbosity=2)
