"""Tests for PlotViewWidget selection toolbar controls."""

import unittest
from unittest.mock import MagicMock

from qgis.PyQt import QtWidgets, QtCore

try:
    from swe2d.workbench.views.studio_viewer_plot import PlotViewWidget
    _HAVE_MPL = True
except ImportError:
    _HAVE_MPL = False


# Setup QApplication before any Qt widgets
app = QtWidgets.QApplication.instance()
if app is None:
    app = QtWidgets.QApplication([])

try:
    from swe2d.workbench.views.studio_viewer_plot import PlotViewWidget
    _HAVE_MPL = True
except ImportError:
    _HAVE_MPL = False


class TestPlotViewWidgetSelectionToolbar(unittest.TestCase):
    """Test PlotViewWidget selection toolbar controls."""

    @unittest.skipIf(not _HAVE_MPL, "Matplotlib not available")
    def test_selection_toolbar_widgets_created(self):
        """Verify selection toolbar widgets are created in _build_ui()."""
        widget = PlotViewWidget("Structure")
        self.assertIsNotNone(widget._filter_combo)
        self.assertIsNotNone(widget._click_check)
        self.assertIsNotNone(widget._selected_elements)
        self.assertIsNotNone(widget._element_filter)
        self.assertIsNotNone(widget._click_to_select)

    @unittest.skipIf(not _HAVE_MPL, "Matplotlib not available")
    def test_filter_combo_items(self):
        """Verify filter combo box has correct items."""
        widget = PlotViewWidget("Structure")
        self.assertEqual(widget._filter_combo.count(), 4)
        self.assertEqual(widget._filter_combo.itemText(0), "All")
        self.assertEqual(widget._filter_combo.itemText(1), "Structures")
        self.assertEqual(widget._filter_combo.itemText(2), "Nodes")
        self.assertEqual(widget._filter_combo.itemText(3), "Links")

    @unittest.skipIf(not _HAVE_MPL, "Matplotlib not available")
    def test_filter_change_updates_state(self):
        """Verify filter change updates _element_filter and triggers _populate_table."""
        widget = PlotViewWidget("Structure")

        # Mock _populate_table to track calls
        original_populate = widget._populate_table
        widget._populate_table = MagicMock()

        # Change filter
        widget._on_filter_change("Nodes")

        # Verify state updated
        self.assertEqual(widget._element_filter, "Nodes")

        # Verify _populate_table was called
        widget._populate_table.assert_called_once()

    @unittest.skipIf(not _HAVE_MPL, "Matplotlib not available")
    def test_click_toggle_updates_state(self):
        """Verify click-to-select toggle updates _click_to_select."""
        widget = PlotViewWidget("Structure")

        # Toggle off
        widget._on_click_toggle(False)
        self.assertFalse(widget._click_to_select)

        # Toggle on
        widget._on_click_toggle(True)
        self.assertTrue(widget._click_to_select)

    @unittest.skipIf(not _HAVE_MPL, "Matplotlib not available")
    def test_clear_selection_empty_dict(self):
        """Verify clear selection clears _selected_elements and triggers _populate_table."""
        widget = PlotViewWidget("Structure")

        # Set some initial selection
        widget._selected_elements = {"node1": {"id": 1, "type": "Node"}}
        widget._populate_table = MagicMock()

        # Clear selection
        widget._on_clear_selection()

        # Verify dict is empty
        self.assertEqual(len(widget._selected_elements), 0)

        # Verify _populate_table was called
        widget._populate_table.assert_called_once()

    @unittest.skipIf(not _HAVE_MPL, "Matplotlib not available")
    def test_clear_selection_nonempty_dict(self):
        """Verify clear selection clears a non-empty selection."""
        widget = PlotViewWidget("Structure")

        # Set some initial selection
        widget._selected_elements = {
            "node1": {"id": 1, "type": "Node"},
            "link1": {"id": 2, "type": "Link"},
            "structure1": {"id": 3, "type": "Structure"},
        }
        widget._populate_table = MagicMock()

        # Clear selection
        widget._on_clear_selection()

        # Verify dict is empty
        self.assertEqual(len(widget._selected_elements), 0)
        widget._populate_table.assert_called_once()

    @unittest.skipIf(not _HAVE_MPL, "Matplotlib not available")
    def test_plot_click_handler_not_in_axes_returns_early(self):
        """Verify _on_plot_click returns early when click is outside axes."""
        widget = PlotViewWidget("Structure")

        # Mock event with no inaxes
        mock_event = MagicMock()
        mock_event.inaxes = None

        # Should not raise
        widget._on_plot_click(mock_event)

    @unittest.skipIf(not _HAVE_MPL, "Matplotlib not available")
    def test_plot_click_handler_disabled_returns_early(self):
        """Verify _on_plot_click returns early when click-to-select is disabled."""
        widget = PlotViewWidget("Structure")
        widget._click_to_select = False

        # Mock event with inaxes
        mock_event = MagicMock()
        mock_event.inaxes = MagicMock()
        mock_event.xdata = 100.0
        mock_event.ydata = 200.0

        # Should not raise
        widget._on_plot_click(mock_event)

    @unittest.skipIf(not _HAVE_MPL, "Matplotlib not available")
    def test_plot_click_handler_with_node_selection(self):
        """Verify _on_plot_click selects a node when clicked near its marker."""
        widget = PlotViewWidget("Structure")

        # Set up mock selected elements with node
        node_id = "node_1"
        node_data = {
            "id": 1,
            "type": "node",
            "x": 100.0,
            "y": 200.0,
        }
        widget._selected_elements = {node_id: node_data}

        # Mock event near the node
        mock_event = MagicMock()
        mock_event.inaxes = MagicMock()
        mock_event.xdata = 105.0  # Within 15px threshold
        mock_event.ydata = 198.0

        # Track calls to set_selected_element and refresh
        widget.set_selected_element = MagicMock()
        widget.refresh = MagicMock()

        # Call handler
        widget._on_plot_click(mock_event)

        # Verify selection was updated
        widget.set_selected_element.assert_called_once_with(node_id, node_data)
        widget.refresh.assert_called_once()

    @unittest.skipIf(not _HAVE_MPL, "Matplotlib not available")
    def test_plot_click_handler_with_link_selection(self):
        """Verify _on_plot_click selects a link when clicked near its marker."""
        widget = PlotViewWidget("Structure")

        # Set up mock selected elements with link
        link_id = "link_1"
        link_data = {
            "id": 2,
            "type": "link",
            "x0": 100.0,
            "y0": 200.0,
            "x1": 150.0,
            "y1": 250.0,
        }
        widget._selected_elements = {link_id: link_data}

        # Mock event near the link midpoint
        mock_event = MagicMock()
        mock_event.inaxes = MagicMock()
        mock_event.xdata = 125.0  # Within 15px threshold
        mock_event.ydata = 225.0

        # Track calls
        widget.set_selected_element = MagicMock()
        widget.refresh = MagicMock()

        # Call handler
        widget._on_plot_click(mock_event)

        # Verify selection was updated
        widget.set_selected_element.assert_called_once_with(link_id, link_data)
        widget.refresh.assert_called_once()

    @unittest.skipIf(not _HAVE_MPL, "Matplotlib not available")
    def test_plot_click_handler_with_structure_selection(self):
        """Verify _on_plot_click selects a structure when clicked near its marker."""
        widget = PlotViewWidget("Structure")

        # Set up mock selected elements with structure
        struct_id = "struct_1"
        struct_data = {
            "id": 3,
            "type": "structure",
            "station_m": 125.0,
            "elev_m": 50.0,
        }
        widget._selected_elements = {struct_id: struct_data}

        # Mock event near the structure
        mock_event = MagicMock()
        mock_event.inaxes = MagicMock()
        mock_event.xdata = 125.0  # Within 15px threshold
        mock_event.ydata = 51.0

        # Track calls
        widget.set_selected_element = MagicMock()
        widget.refresh = MagicMock()

        # Call handler
        widget._on_plot_click(mock_event)

        # Verify selection was updated
        widget.set_selected_element.assert_called_once_with(struct_id, struct_data)
        widget.refresh.assert_called_once()

    @unittest.skipIf(not _HAVE_MPL, "Matplotlib not available")
    def test_plot_click_handler_far_from_elements(self):
        """Verify _on_plot_click does nothing when click is far from all markers."""
        widget = PlotViewWidget("Structure")

        # Set up mock selected elements
        widget._selected_elements = {
            "node_1": {"id": 1, "type": "node", "x": 100.0, "y": 200.0},
            "link_1": {"id": 2, "type": "link", "x0": 100.0, "y0": 200.0, "x1": 150.0, "y1": 250.0},
        }

        # Mock event far from all elements
        mock_event = MagicMock()
        mock_event.inaxes = MagicMock()
        mock_event.xdata = 1000.0
        mock_event.ydata = 2000.0

        # Track calls
        widget.set_selected_element = MagicMock()
        widget.refresh = MagicMock()

        # Call handler
        widget._on_plot_click(mock_event)

        # Verify selection was NOT updated
        widget.set_selected_element.assert_not_called()
        widget.refresh.assert_called_once()

    @unittest.skipIf(not _HAVE_MPL, "Matplotlib not available")
    def test_plot_click_handler_no_markers(self):
        """Verify _on_plot_click does nothing when no elements selected."""
        widget = PlotViewWidget("Structure")
        widget._selected_elements = {}

        # Mock event
        mock_event = MagicMock()
        mock_event.inaxes = MagicMock()
        mock_event.xdata = 100.0
        mock_event.ydata = 200.0

        # Track calls
        widget.set_selected_element = MagicMock()
        widget.refresh = MagicMock()

        # Call handler
        widget._on_plot_click(mock_event)

        # Verify selection was NOT updated
        widget.set_selected_element.assert_not_called()
        widget.refresh.assert_called_once()


if __name__ == "__main__":
    unittest.main()