"""Tests for overlay_parameters_service.

The service is the SOLE source of overlay-parameter collection logic for the
workbench (Phase 1 Task 1). The dialog calls ``collect_overlay_parameters(view,
t_use)`` and the service returns the complete dict consumed by
``swe2d.results.high_perf_viewer.render_unstructured_snapshot_image``.
"""
import unittest
from typing import Any, Dict
from unittest.mock import MagicMock


def _build_mock_view() -> MagicMock:
    """Build a MagicMock with the widget/runtime attrs the service reads."""
    import numpy as np
    view = MagicMock()

    # Overlay geometry arrays now live on _results_data
    data = MagicMock()
    data.overlay_cell_x = np.array([0.0, 1.0])
    data.overlay_cell_y = np.array([0.0, 1.0])
    data.overlay_cell_bed = np.array([0.0, 0.0])
    data.overlay_node_x = np.array([0.0, 1.0])
    data.overlay_node_y = np.array([0.0, 1.0])
    data.overlay_cell_nodes = np.array([[0, 1]])
    data.overlay_tri_to_cell = np.array([0])
    view._results_data = data

    view._snapshot_timesteps = [
        (0.0, np.array([1.0, 1.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0]))
    ]
    view._gravity = 9.81
    view._mannings_n = 0.035
    view._length_unit_name = "m"

    # Overlay widgets live on _results_toolbox
    tb = MagicMock()
    tb.field_combo.currentData.return_value = "depth"
    tb.wse_render_combo.currentData.return_value = "cell"
    tb.cmap_combo.currentData.return_value = "turbo"
    tb.visible_only_chk.isChecked.return_value = False
    tb.lock_canvas_chk.isChecked.return_value = False
    tb.auto_contrast_chk.isChecked.return_value = True
    tb.res_combo.currentData.return_value = (1280, 720)
    tb.opacity_spin.value.return_value = 1.0
    tb.arrows_chk.isChecked.return_value = False
    tb.arrow_density_spin.value.return_value = 28.0
    tb.arrow_length_spin.value.return_value = 1.0
    tb.arrow_head_length_spin.value.return_value = 1.0
    tb.arrow_head_width_spin.value.return_value = 1.0
    tb.streamlines_chk.isChecked.return_value = False
    tb.streamline_backend_combo.currentData.return_value = "auto"
    tb.streamline_seed_spin.value.return_value = 48.0
    tb.streamline_steps_spin.value.return_value = 24.0
    view._results_toolbox = tb

    view._resolve_map_canvas.return_value = None
    return view


class TestOverlayParametersService(unittest.TestCase):
    def test_service_imports(self):
        from swe2d.workbench.services.overlay_parameters_service import collect_overlay_parameters
        self.assertIsNotNone(collect_overlay_parameters)

    def test_collect_returns_dict(self):
        from swe2d.workbench.services.overlay_parameters_service import collect_overlay_parameters
        result = collect_overlay_parameters(_build_mock_view(), t_use=1.0)
        self.assertIsInstance(result, dict)

    def test_collect_has_all_render_keys(self):
        """Service returns the full keyword set consumed by the render function."""
        from swe2d.workbench.services.overlay_parameters_service import collect_overlay_parameters
        result = collect_overlay_parameters(_build_mock_view(), t_use=1.0)
        expected_keys = {
            "cell_x", "cell_y", "cell_bed", "node_x", "node_y", "cell_nodes",
            "tri_to_cell", "timesteps", "current_time_s", "field_key",
            "wse_render_mode", "cmap_key", "resolution", "auto_contrast",
            "show_velocity_arrows", "arrow_stride_px", "arrow_length_scale",
            "arrow_head_length_scale", "arrow_head_width_scale",
            "show_streamlines", "streamline_backend", "streamline_seed_count",
            "streamline_steps", "visible_extent_world", "render_extent_world",
            "gravity", "courant_cell_size", "courant_dt", "manning_n",
            "show_legend", "legend_label",
        }
        self.assertEqual(set(result.keys()), expected_keys)

    def test_collect_reads_widget_state(self):
        from swe2d.workbench.services.overlay_parameters_service import collect_overlay_parameters
        result = collect_overlay_parameters(_build_mock_view(), t_use=1.0)
        self.assertEqual(result["field_key"], "depth")
        self.assertEqual(result["cmap_key"], "turbo")
        self.assertEqual(result["current_time_s"], 1.0)

    def test_collect_returns_runtime_state(self):
        from swe2d.workbench.services.overlay_parameters_service import collect_overlay_parameters
        result = collect_overlay_parameters(_build_mock_view(), t_use=2.5)
        self.assertEqual(result["gravity"], 9.81)
        self.assertEqual(result["manning_n"], 0.035)
        self.assertEqual(len(result["timesteps"]), 1)

    def test_collect_handles_missing_widgets(self):
        """A view with no widgets returns defaults (no exception)."""
        from swe2d.workbench.services.overlay_parameters_service import collect_overlay_parameters
        view = MagicMock()
        view._results_data = None
        view._resolve_map_canvas.return_value = None
        view._snapshot_timesteps = []
        view._length_unit_name = "m"
        view._gravity = 9.81
        view._mannings_n = 0.035
        # toolbox with missing widgets — _safe helpers return defaults
        tb = MagicMock(spec=[])
        view._results_toolbox = tb
        result = collect_overlay_parameters(view, t_use=0.0)
        self.assertEqual(result["field_key"], "depth")
        self.assertEqual(result["current_time_s"], 0.0)

    def test_collect_sets_overlay_opacity_side_effect(self):
        """The service writes ``_overlay_opacity`` on the view (legacy side effect)."""
        from swe2d.workbench.services.overlay_parameters_service import collect_overlay_parameters
        view = _build_mock_view()
        collect_overlay_parameters(view, t_use=1.0)
        self.assertEqual(view._overlay_opacity, 1.0)

    def test_collect_legend_label_for_depth(self):
        from swe2d.workbench.services.overlay_parameters_service import collect_overlay_parameters
        view = _build_mock_view()
        view._results_toolbox.field_combo.currentData.return_value = "depth"
        view._length_unit_name = "ft"
        result = collect_overlay_parameters(view, t_use=1.0)
        self.assertEqual(result["legend_label"], "Depth (ft)")


if __name__ == "__main__":
    unittest.main()
