"""Tests for swe2d.workbench.services.results_render_service — matplotlib rendering."""

import unittest
from unittest.mock import MagicMock
import numpy as np

from swe2d.workbench.services.results_render_service import render_structures_on_figure


class TestRenderStructuresOnFigure(unittest.TestCase):
    """Test render_structures_on_figure with and without selection markers."""

    def _make_result_data(self, records=None, current_time_sec=0.0):
        """Create a mock result_data object with coupling records."""
        data = MagicMock()
        data._coupling_records = records or []
        data.current_time_sec = current_time_sec
        return data

    def _make_figure(self):
        """Create a mock figure object."""
        fig = MagicMock()
        fig.clear = MagicMock()
        fig.add_subplot = MagicMock(return_value=MagicMock())
        return fig

    def test_renders_selection_markers_for_selected_structure(self):
        """Render structure plot with selection markers when selected_elements is provided."""
        # Setup
        records = [
            {"object_id": "s1", "t_s": 0.0, "value": 5.0},
            {"object_id": "s1", "t_s": 1.0, "value": 7.0},
            {"object_id": "s2", "t_s": 0.0, "value": 3.0},
        ]
        result_data = self._make_result_data(records=records, current_time_sec=2.0)
        fig = self._make_figure()
        ax = MagicMock()
        fig.add_subplot.return_value = ax

        selected_elements = {
            "s1": {
                "type": "structure",
                "flow_cms": 6.0,  # Current flow at current time
            }
        }

        # Execute
        render_structures_on_figure(
            fig=fig,
            mesh_data=None,
            result_data=result_data,
            mode="flow_cms",
            h_min=0.0,
            selected_elements=selected_elements,
        )

        # Verify
        fig.clear.assert_called_once()
        ax.plot.assert_called()
        # Should have 3 plot calls: s1 timeseries (2 points), s2 timeseries (1 point), selection marker
        plot_calls = ax.plot.call_args_list
        self.assertEqual(len(plot_calls), 3)

        # First call should be for s1 timeseries
        x1, y1 = plot_calls[0][0]
        self.assertTrue(np.array_equal(x1, np.array([0.0, 1.0 / 3600.0])))
        self.assertTrue(np.array_equal(y1, np.array([5.0, 7.0])))

        # Second call should be for selection marker at current time
        x2, y2 = plot_calls[1][0]
        self.assertEqual(len(x2), 1)
        self.assertEqual(len(y2), 1)
        self.assertAlmostEqual(x2[0], 2.0 / 3600.0, places=6)
        self.assertAlmostEqual(y2[0], 6.0, places=6)

    def test_renders_no_selection_markers_when_no_selected_elements(self):
        """Render structure plot without selection markers when selected_elements is None."""
        # Setup
        records = [
            {"object_id": "s1", "t_s": 0.0, "value": 5.0},
        ]
        result_data = self._make_result_data(records=records, current_time_sec=1.0)
        fig = self._make_figure()
        ax = MagicMock()
        fig.add_subplot.return_value = ax

        selected_elements = None

        # Execute
        render_structures_on_figure(
            fig=fig,
            mesh_data=None,
            result_data=result_data,
            mode="flow_cms",
            h_min=0.0,
            selected_elements=selected_elements,
        )

        # Verify
        fig.clear.assert_called_once()
        ax.plot.assert_called()
        plot_calls = ax.plot.call_args_list
        self.assertEqual(len(plot_calls), 1)

        # Only timeseries call, no selection marker
        x, y = plot_calls[0][0]
        self.assertTrue(np.array_equal(x, np.array([0.0 / 3600.0])))
        self.assertTrue(np.array_equal(y, np.array([5.0])))

    def test_skips_non_structure_selections(self):
        """Skip selection markers for non-structure elements."""
        # Setup
        records = [
            {"object_id": "s1", "t_s": 0.0, "value": 5.0},
        ]
        result_data = self._make_result_data(records=records, current_time_sec=1.0)
        fig = self._make_figure()
        ax = MagicMock()
        fig.add_subplot.return_value = ax

        selected_elements = {
            "s1": {
                "type": "drainage_node",  # Not a structure
                "flow_cms": 6.0,
            }
        }

        # Execute
        render_structures_on_figure(
            fig=fig,
            mesh_data=None,
            result_data=result_data,
            mode="flow_cms",
            h_min=0.0,
            selected_elements=selected_elements,
        )

        # Verify
        fig.clear.assert_called_once()
        ax.plot.assert_called()
        plot_calls = ax.plot.call_args_list
        self.assertEqual(len(plot_calls), 1)  # Only timeseries, no selection marker

    def test_handles_empty_selected_elements(self):
        """Handle empty selected_elements dictionary."""
        # Setup
        records = [
            {"object_id": "s1", "t_s": 0.0, "value": 5.0},
        ]
        result_data = self._make_result_data(records=records, current_time_sec=1.0)
        fig = self._make_figure()
        ax = MagicMock()
        fig.add_subplot.return_value = ax

        selected_elements = {}

        # Execute
        render_structures_on_figure(
            fig=fig,
            mesh_data=None,
            result_data=result_data,
            mode="flow_cms",
            h_min=0.0,
            selected_elements=selected_elements,
        )

        # Verify
        fig.clear.assert_called_once()
        ax.plot.assert_called()
        plot_calls = ax.plot.call_args_list
        self.assertEqual(len(plot_calls), 1)  # Only timeseries, no selection markers

    def test_handles_missing_flow_cms_in_selected_element(self):
        """Handle selected element without flow_cms field."""
        # Setup
        records = [
            {"object_id": "s1", "t_s": 0.0, "value": 5.0},
        ]
        result_data = self._make_result_data(records=records, current_time_sec=1.0)
        fig = self._make_figure()
        ax = MagicMock()
        fig.add_subplot.return_value = ax

        selected_elements = {
            "s1": {
                "type": "structure",
                # Missing flow_cms
            }
        }

        # Execute
        render_structures_on_figure(
            fig=fig,
            mesh_data=None,
            result_data=result_data,
            mode="flow_cms",
            h_min=0.0,
            selected_elements=selected_elements,
        )

        # Verify - should still render timeseries but no selection marker
        fig.clear.assert_called_once()
        ax.plot.assert_called()
        plot_calls = ax.plot.call_args_list
        self.assertEqual(len(plot_calls), 1)  # Only timeseries (no selection marker since flow_cms missing)


class TestRenderProfileOnFigure(unittest.TestCase):
    """Test render_profile_on_figure with selection markers."""

    def _make_result_data(self, records=None, current_time_sec=0.0, line_id="line1"):
        """Create a mock result_data object for profile rendering."""
        data = MagicMock()
        data._coupling_records = records or []
        data.current_time_sec = current_time_sec
        data.line_id = line_id
        data.prof_var_key = "wse_bed"
        data.prof_fill_key = "none"
        data.prof_cmap = "viridis"
        data.prof_show_structures = True
        return data

    def _make_figure(self):
        """Create a mock figure object."""
        fig = MagicMock()
        fig.clear = MagicMock()
        fig.add_subplot = MagicMock(return_value=MagicMock())
        fig.text = MagicMock()
        return fig

    def test_renders_selection_markers_for_selected_structures(self):
        """Render profile with selection markers when selected_elements is provided."""
        # Setup
        result_data = self._make_result_data(
            records=[],
            current_time_sec=100.0,
            line_id="line1",
        )
        fig = self._make_figure()
        ax = MagicMock()
        fig.add_subplot.return_value = ax
        ax.get_ylim.return_value = (0.0, 10.0)

        selected_elements = {
            "s1": {
                "type": "structure",
                "station": 100.0,
                "elev": 5.0,
                "flow": 6.5,
                "object_id": "s1",
            }
        }

        # Mock load_structure_flows_fn to return structure data
        def mock_load_structure_flows(gpkg, rid, t, t_tol=1.0):
            return [
                {
                    "object_id": "s1",
                    "value": 6.5,
                }
            ]

        # Execute
        from swe2d.workbench.services.results_render_service import render_profile_on_figure

        render_profile_on_figure(
            fig=fig,
            mesh_data=None,
            result_data=result_data,
            mode="wse_bed",
            h_min=0.0,
            selected_elements=selected_elements,
            load_structure_flows_fn=mock_load_structure_flows,
        )

        # Verify
        fig.clear.assert_called_once()
        # Should call ax.axvline for selection marker
        ax.axvline.assert_called()
        ax.plot.assert_called()
        ax.text.assert_called()

        # Get the axvline call
        axvline_call = ax.axvline.call_args
        self.assertIsNotNone(axvline_call)
        kwargs = axvline_call[1] if len(axvline_call) > 1 else {}
        self.assertEqual(kwargs.get("color"), "red")
        self.assertAlmostEqual(kwargs.get("linewidth", 0), 2.0, places=1)

        # Get the plot call for elevation marker
        plot_call = ax.plot.call_args
        self.assertIsNotNone(plot_call)
        kwargs = plot_call[1] if len(plot_call) > 1 else {}
        self.assertEqual(kwargs.get("marker"), "v")
        self.assertEqual(kwargs.get("color"), "red")

        # Get the text call
        text_call = ax.text.call_args
        self.assertIsNotNone(text_call)
        args = text_call[0] if len(text_call) > 0 else ()
        kwargs = text_call[1] if len(text_call) > 1 else {}
        self.assertEqual(len(args), 3)  # x, y, text
        self.assertIn("s1", args[2])
        self.assertIn("6.5", args[2])

    def test_renders_no_selection_markers_when_none(self):
        """Render profile without selection markers when selected_elements is None."""
        # Setup
        result_data = self._make_result_data(
            records=[],
            current_time_sec=100.0,
            line_id="line1",
        )
        fig = self._make_figure()
        ax = MagicMock()
        fig.add_subplot.return_value = ax
        ax.get_ylim.return_value = (0.0, 10.0)

        selected_elements = None

        # Execute
        from swe2d.workbench.services.results_render_service import render_profile_on_figure

        render_profile_on_figure(
            fig=fig,
            mesh_data=None,
            result_data=result_data,
            mode="wse_bed",
            h_min=0.0,
            selected_elements=selected_elements,
        )

        # Verify
        fig.clear.assert_called_once()
        # Should NOT call ax.axvline when selected_elements is None
        ax.axvline.assert_not_called()
        ax.plot.assert_called()
        ax.text.assert_not_called()

    def test_handles_empty_selected_elements(self):
        """Handle empty selected_elements dictionary."""
        # Setup
        result_data = self._make_result_data(
            records=[],
            current_time_sec=100.0,
            line_id="line1",
        )
        fig = self._make_figure()
        ax = MagicMock()
        fig.add_subplot.return_value = ax
        ax.get_ylim.return_value = (0.0, 10.0)

        selected_elements = {}

        # Execute
        from swe2d.workbench.services.results_render_service import render_profile_on_figure

        render_profile_on_figure(
            fig=fig,
            mesh_data=None,
            result_data=result_data,
            mode="wse_bed",
            h_min=0.0,
            selected_elements=selected_elements,
        )

        # Verify
        fig.clear.assert_called_once()
        ax.axvline.assert_not_called()

    def test_handles_nan_elevation_in_selected_structure(self):
        """Handle selected structure with NaN elevation."""
        # Setup
        result_data = self._make_result_data(
            records=[],
            current_time_sec=100.0,
            line_id="line1",
        )
        fig = self._make_figure()
        ax = MagicMock()
        fig.add_subplot.return_value = ax
        ax.get_ylim.return_value = (0.0, 10.0)

        selected_elements = {
            "s1": {
                "type": "structure",
                "station": 100.0,
                "elev": float("nan"),  # NaN elevation
                "flow": 6.5,
                "object_id": "s1",
            }
        }

        # Execute
        from swe2d.workbench.services.results_render_service import render_profile_on_figure

        render_profile_on_figure(
            fig=fig,
            mesh_data=None,
            result_data=result_data,
            mode="wse_bed",
            h_min=0.0,
            selected_elements=selected_elements,
        )

        # Verify
        fig.clear.assert_called_once()
        ax.axvline.assert_called()
        ax.plot.assert_called()
        ax.text.assert_called()

    def test_skips_non_structure_selections(self):
        """Skip selection markers for non-structure elements."""
        # Setup
        result_data = self._make_result_data(
            records=[],
            current_time_sec=100.0,
            line_id="line1",
        )
        fig = self._make_figure()
        ax = MagicMock()
        fig.add_subplot.return_value = ax
        ax.get_ylim.return_value = (0.0, 10.0)

        selected_elements = {
            "n1": {
                "type": "drainage_node",  # Not a structure
                "station": 100.0,
                "elev": 5.0,
                "flow": 6.5,
                "object_id": "n1",
            }
        }

        # Execute
        from swe2d.workbench.services.results_render_service import render_profile_on_figure

        render_profile_on_figure(
            fig=fig,
            mesh_data=None,
            result_data=result_data,
            mode="wse_bed",
            h_min=0.0,
            selected_elements=selected_elements,
        )

        # Verify
        fig.clear.assert_called_once()
        ax.axvline.assert_not_called()


if __name__ == "__main__":
    unittest.main()