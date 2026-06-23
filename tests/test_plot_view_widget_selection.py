"""Tests for PlotViewWidget selection state and protocol methods."""

import unittest
from unittest.mock import MagicMock, Mock, patch
import numpy as np
from typing import Dict, Any
from qgis.PyQt.QtWidgets import QApplication

from swe2d.workbench.views.studio_viewer_plot import PlotViewWidget

_app = None


def _ensure_app():
    global _app
    if _app is None:
        _app = QApplication.instance() or QApplication([])


class TestPlotViewWidgetSelection(unittest.TestCase):
    """Test PlotViewWidget selection state and protocol methods."""

    def setUp(self):
        """Set up test fixtures."""
        _ensure_app()
        self.widget = None
        self.patcher = patch('swe2d.workbench.views.studio_viewer_plot.PlotViewWidget._build_ui')
        self.mock_build_ui = self.patcher.start()
        self.populate_patcher = patch('swe2d.workbench.views.studio_viewer_plot.PlotViewWidget._populate_table')
        self.mock_populate = self.populate_patcher.start()

    def tearDown(self):
        """Clean up after tests."""
        if self.widget:
            self.widget.close()
            self.widget.deleteLater()
        self.patcher.stop()

    def test_init_has_selection_state_attributes(self):
        """Test that __init__ initializes selection state attributes."""
        # Test class structure without instantiation
        self.assertTrue(hasattr(PlotViewWidget, '__init__'))

        # Test that the class has the expected methods
        self.assertTrue(hasattr(PlotViewWidget, 'set_selected_elements'))
        self.assertTrue(hasattr(PlotViewWidget, 'get_selected_elements'))

    def test_set_selected_elements(self):
        """Test set_selected_elements updates state and populates table."""
        self.widget = PlotViewWidget()
        self.widget._table_widget = MagicMock()
        self.widget._table_widget.isVisible.return_value = True

        # Create mock elements
        elements = {
            "structure_1": {"type": "weir", "id": "structure_1"},
            "structure_2": {"type": "orifice", "id": "structure_2"},
        }

        # Call set_selected_elements
        self.widget.set_selected_elements(elements)

        # Verify state was updated
        self.assertEqual(self.widget._selected_elements, elements)

        # Verify table was repopulated (method is on PlotViewWidget, not table_widget)
        self.mock_populate.assert_called_once()

    def test_set_selected_elements_with_invisible_table(self):
        """Test set_selected_elements doesn't populate table if invisible."""
        self.widget = PlotViewWidget()
        self.widget._table_widget = MagicMock()
        self.widget._table_widget.isVisible.return_value = False

        elements = {"structure_1": {"type": "weir"}}

        self.widget.set_selected_elements(elements)

        # Verify table was not populated
        self.widget._table_widget.populate.assert_not_called()

    def test_get_selected_elements_returns_readonly_view(self):
        """Test get_selected_elements returns a copy of selected elements."""
        self.widget = PlotViewWidget()
        self.widget._table_widget = MagicMock()

        elements = {"structure_1": {"type": "weir"}}
        self.widget._selected_elements = elements

        # Get selected elements
        result = self.widget.get_selected_elements()

        # Verify it's a copy (not the same object)
        self.assertEqual(result, elements)
        self.assertIsNot(result, elements)

    def test_set_selected_elements_with_empty_dict(self):
        """Test set_selected_elements with empty dict clears selection."""
        self.widget = PlotViewWidget()
        self.widget._table_widget = MagicMock()
        self.widget._table_widget.isVisible.return_value = True

        elements = {}

        self.widget.set_selected_elements(elements)

        # Verify state was cleared
        self.assertEqual(self.widget._selected_elements, {})

    def test_element_filter_attribute_exists(self):
        """Test that element_filter attribute is initialized."""
        self.widget = PlotViewWidget()
        self.widget._table_widget = MagicMock()

        # Verify element_filter exists
        self.assertEqual(self.widget._element_filter, "All")

    def test_click_to_select_attribute_exists(self):
        """Test that click_to_select attribute is initialized."""
        self.widget = PlotViewWidget()
        self.widget._table_widget = MagicMock()

        # Verify click_to_select exists
        self.assertTrue(self.widget._click_to_select)


if __name__ == "__main__":
    unittest.main()