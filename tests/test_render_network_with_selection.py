"""Tests for render_network_on_figure — EPASWMM-style link profile."""

import unittest
from unittest.mock import MagicMock

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.collections import PolyCollection


class TestRenderNetworkWithSelection(unittest.TestCase):
    """Test render_network_on_figure EPASWMM-style link profile rendering."""

    def setUp(self):
        """Set up test fixtures with consistent node naming."""
        # Create a mock result_data with coupling records.
        # Node names match across object_id and object_name so the
        # profile parser correctly links node depths to link endpoints.
        self.result_data = MagicMock()
        self.result_data._coupling_records = [
            {
                "object_id": "node_1",
                "object_name": "node_1",
                "metric": "depth",
                "value": 1.5,
                "component": "drainage_node",
            },
            {
                "object_id": "node_2",
                "object_name": "node_2",
                "metric": "depth",
                "value": 0.8,
                "component": "drainage_node",
            },
            {
                "object_id": "node_3",
                "object_name": "node_3",
                "metric": "depth",
                "value": 0.5,
                "component": "drainage_node",
            },
            {
                "object_id": "link_1",
                "object_name": "node_1 -> node_2",
                "metric": "flow",
                "value": 2.3,
                "component": "drainage_link",
            },
        ]
        self.result_data.current_time_sec = 3600.0

        # Create mock mesh_data (not used by the profile renderer but
        # required by the function signature).
        self.mesh_data = MagicMock()

    # ------------------------------------------------------------------
    # Test 1: Selection marker profile rendering
    # ------------------------------------------------------------------

    def test_render_network_with_selection_markers(self):
        """Verify EPASWMM-style profile renders with a selected link."""
        fig = Figure()

        from swe2d.services.results_render_service import (
            render_network_on_figure,
        )

        render_network_on_figure(
            fig=fig,
            mesh_data=self.mesh_data,
            result_data=self.result_data,
            mode="network",
            h_min=0.0,
            selected_element_id="link_1",
        )

        # ---- Single subplot (no plan view) ----
        self.assertEqual(len(fig.axes), 1)
        ax = fig.axes[0]

        # ---- No Circle patches exist (old plan-view markers) ----
        circles = [c for c in ax.get_children()
                   if isinstance(c, plt.Circle)]
        self.assertEqual(len(circles), 0)

        # ---- Depth fill / bed fill (PolyCollection from fill_between) ----
        self.assertGreaterEqual(
            len(ax.collections), 1,
            "Expected at least one fill_between PolyCollection",
        )

        # ---- Flow annotation text ----
        flow_texts = [t for t in ax.texts
                      if t.get_text().startswith("Q =")]
        self.assertEqual(len(flow_texts), 1)
        self.assertIn("2.300", flow_texts[0].get_text())

        # ---- Triangle markers at node inverts ----
        marker_lines = [l for l in ax.lines
                        if l.get_marker() in ("v", "^")]
        self.assertEqual(
            len(marker_lines), 2,
            "Expected upstream (v) and downstream (^) triangle markers",
        )

        # ---- Node ID labels ----
        node_labels = [
            t for t in ax.texts
            if t.get_text().strip() in ("node_1", "node_2")
        ]
        self.assertEqual(len(node_labels), 2)

    # ------------------------------------------------------------------
    # Test 2: No explicit selection — default first-link profile
    # ------------------------------------------------------------------

    def test_render_network_without_selection(self):
        """Verify profile renders for the first link when no selection given."""
        fig = Figure()

        from swe2d.services.results_render_service import (
            render_network_on_figure,
        )

        render_network_on_figure(
            fig=fig,
            mesh_data=self.mesh_data,
            result_data=self.result_data,
            mode="network",
            h_min=0.0,
        )

        # ---- Single subplot ----
        self.assertEqual(len(fig.axes), 1)
        ax = fig.axes[0]

        # ---- Link profile elements exist ----
        self.assertGreaterEqual(
            len(ax.collections), 1,
            "Expected fill_between collections for bed/depth fill",
        )
        self.assertGreaterEqual(
            len(ax.lines), 1,
            "Expected at least the invert line",
        )

        # ---- No error text ----
        error_texts = [t for t in ax.texts
                       if "No link" in t.get_text()]
        self.assertEqual(len(error_texts), 0)

    # ------------------------------------------------------------------
    # Test 3: Empty / non-matching selection — error text
    # ------------------------------------------------------------------

    def test_render_network_with_empty_selection(self):
        """Verify empty selection shows 'No link data' error text."""
        fig = Figure()

        from swe2d.services.results_render_service import (
            render_network_on_figure,
        )

        render_network_on_figure(
            fig=fig,
            mesh_data=self.mesh_data,
            result_data=self.result_data,
            mode="network",
            h_min=0.0,
            selected_element_id="nonexistent",
        )

        # ---- Single subplot ----
        self.assertEqual(len(fig.axes), 1)
        ax = fig.axes[0]

        # ---- Error text displayed ----
        error_texts = [t for t in ax.texts
                       if "No link data" in t.get_text()]
        self.assertEqual(len(error_texts), 1)


if __name__ == "__main__":
    unittest.main()
