"""Tests for PlotViewWidget table checkbox selection functionality."""

import unittest
from unittest.mock import MagicMock, Mock, patch
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


class TestPlotViewWidgetTableCheckbox(unittest.TestCase):
    """Test PlotViewWidget table checkbox selection functionality."""

    @unittest.skipIf(not _HAVE_MPL, "Matplotlib not available")
    def setUp(self):
        """Set up test fixtures."""
        self.widget = PlotViewWidget("Structure")

    @unittest.skipIf(not _HAVE_MPL, "Matplotlib not available")
    def test_checkbox_column_added_for_structure_mode(self):
        """Verify checkbox column is added to table in Structure mode."""
        # Set mock data with coupling records
        mock_data = MagicMock()
        mock_data._coupling_records = [
            {
                "object_id": "struct_1",
                "type": "weir",
                "station_m": 100.0,
                "elev_m": 50.0,
            },
            {
                "object_id": "struct_2",
                "type": "orifice",
                "station_m": 200.0,
                "elev_m": 60.0,
            },
        ]
        self.widget._result_data = mock_data

        # Call _populate_table
        self.widget._populate_table()

        # Verify checkbox column exists at index 0
        self.assertEqual(self.widget._table_widget.columnCount(), 5)  # 1 checkbox + 4 data columns

        # Verify header has "Select" as first label
        header = self.widget._table_widget.horizontalHeader()
        self.assertEqual(header.sectionText(0), "Select")

    @unittest.skipIf(not _HAVE_MPL, "Matplotlib not available")
    def test_checkbox_column_added_for_network_mode(self):
        """Verify checkbox column is added to table in Network mode."""
        mock_data = MagicMock()
        mock_data._coupling_records = [
            {
                "object_id": "node_1",
                "type": "node",
                "x": 100.0,
                "y": 200.0,
            },
        ]
        self.widget._result_data = mock_data
        self.widget._mode = "Network"

        self.widget._populate_table()

        # Verify checkbox column exists
        self.assertEqual(self.widget._table_widget.columnCount(), 5)

    @unittest.skipIf(not _HAVE_MPL, "Matplotlib not available")
    def test_checkbox_item_created_with_correct_check_state(self):
        """Verify checkbox items are created with correct check states."""
        # Set up selected elements
        self.widget._selected_elements = {
            "struct_1": {"object_id": "struct_1", "type": "weir"},
            "struct_2": {"object_id": "struct_2", "type": "orifice"},
        }

        mock_data = MagicMock()
        mock_data._coupling_records = [
            {"object_id": "struct_1", "type": "weir", "station_m": 100.0},
            {"object_id": "struct_2", "type": "orifice", "station_m": 200.0},
            {"object_id": "struct_3", "type": "weir", "station_m": 300.0},
        ]
        self.widget._result_data = mock_data

        self.widget._populate_table()

        # Check first row (struct_1 - should be checked)
        checkbox_item = self.widget._table_widget.item(0, 0)
        self.assertIsNotNone(checkbox_item)
        self.assertEqual(checkbox_item.checkState(), QtCore.Qt.Checked)

        # Check second row (struct_2 - should be checked)
        checkbox_item = self.widget._table_widget.item(1, 0)
        self.assertIsNotNone(checkbox_item)
        self.assertEqual(checkbox_item.checkState(), QtCore.Qt.Checked)

        # Check third row (struct_3 - should be unchecked)
        checkbox_item = self.widget._table_widget.item(2, 0)
        self.assertIsNotNone(checkbox_item)
        self.assertEqual(checkbox_item.checkState(), QtCore.Qt.Unchecked)

    @unittest.skipIf(not _HAVE_MPL, "Matplotlib not available")
    def test_checkbox_item_stores_record_data(self):
        """Verify checkbox items store the associated record data."""
        mock_data = MagicMock()
        mock_data._coupling_records = [
            {"object_id": "struct_1", "type": "weir", "station_m": 100.0},
        ]
        self.widget._result_data = mock_data

        self.widget._populate_table()

        # Get the checkbox item
        checkbox_item = self.widget._table_widget.item(0, 0)

        # Verify record data is stored
        record = checkbox_item.data(0, "row_record")
        self.assertIsNotNone(record)
        self.assertEqual(record["object_id"], "struct_1")
        self.assertEqual(record["type"], "weir")
        self.assertEqual(record["station_m"], 100.0)

    @unittest.skipIf(not _HAVE_MPL, "Matplotlib not available")
    def test_checkbox_toggle_adds_to_selection(self):
        """Verify checkbox toggle adds element to selection."""
        # Initialize with empty selection
        self.widget._selected_elements = {}
        self.widget.refresh = MagicMock()

        # Set up mock data to populate table
        mock_data = MagicMock()
        mock_data._coupling_records = [
            {"object_id": "struct_1", "type": "weir", "station_m": 100.0},
        ]
        self.widget._result_data = mock_data
        self.widget._populate_table()

        # Find the checkbox item
        checkbox_item = self.widget._table_widget.item(0, 0)

        # Simulate checkbox toggle to checked
        checkbox_item.setCheckState(QtCore.Qt.Checked)

        # Call handler
        self.widget._on_table_checkbox(0, 0)

        # Verify element was added to selection
        self.assertEqual(len(self.widget._selected_elements), 1)
        self.assertIn("struct_1", self.widget._selected_elements)

    @unittest.skipIf(not _HAVE_MPL, "Matplotlib not available")
    def test_checkbox_toggle_removes_from_selection(self):
        """Verify checkbox toggle removes element from selection."""
        # Initialize with selection
        self.widget._selected_elements = {
            "struct_1": {"object_id": "struct_1", "type": "weir"},
        }
        self.widget.refresh = MagicMock()

        # Set up mock data to populate table
        mock_data = MagicMock()
        mock_data._coupling_records = [
            {"object_id": "struct_1", "type": "weir", "station_m": 100.0},
        ]
        self.widget._result_data = mock_data
        self.widget._populate_table()

        # Find the checkbox item
        checkbox_item = self.widget._table_widget.item(0, 0)

        # Simulate checkbox toggle to unchecked
        checkbox_item.setCheckState(QtCore.Qt.Unchecked)

        # Call handler
        self.widget._on_table_checkbox(0, 0)

        # Verify element was removed from selection
        self.assertEqual(len(self.widget._selected_elements), 0)
        self.assertNotIn("struct_1", self.widget._selected_elements)

    @unittest.skipIf(not _HAVE_MPL, "Matplotlib not available")
    def test_checkbox_toggle_updates_multiple_rows(self):
        """Verify checkbox toggle updates multiple elements correctly."""
        # Initialize with mixed selection
        self.widget._selected_elements = {
            "struct_1": {"object_id": "struct_1", "type": "weir"},
            "struct_2": {"object_id": "struct_2", "type": "orifice"},
        }
        self.widget.refresh = MagicMock()

        # Set up mock data to populate table
        mock_data = MagicMock()
        mock_data._coupling_records = [
            {"object_id": "struct_1", "type": "weir", "station_m": 100.0},
            {"object_id": "struct_2", "type": "orifice", "station_m": 200.0},
        ]
        self.widget._result_data = mock_data
        self.widget._populate_table()

        # Toggle struct_1 to unchecked
        checkbox_item = self.widget._table_widget.item(0, 0)
        checkbox_item.setCheckState(QtCore.Qt.Unchecked)
        self.widget._on_table_checkbox(0, 0)

        # Toggle struct_2 to unchecked
        checkbox_item = self.widget._table_widget.item(1, 0)
        checkbox_item.setCheckState(QtCore.Qt.Unchecked)
        self.widget._on_table_checkbox(1, 0)

        # Verify both were removed
        self.assertEqual(len(self.widget._selected_elements), 0)

    @unittest.skipIf(not _HAVE_MPL, "Matplotlib not available")
    def test_checkbox_handler_ignores_non_checkbox_column(self):
        """Verify _on_table_checkbox ignores changes to non-checkbox columns."""
        self.widget._selected_elements = {}
        self.widget.refresh = MagicMock()

        # Simulate cell change in non-checkbox column (column 1)
        self.widget._on_table_checkbox(0, 1)

        # Verify selection was NOT updated
        self.assertEqual(len(self.widget._selected_elements), 0)

    @unittest.skipIf(not _HAVE_MPL, "Matplotlib not available")
    def test_checkbox_handler_ignores_none_item(self):
        """Verify _on_table_checkbox handles None item gracefully."""
        self.widget._selected_elements = {}
        self.widget.refresh = MagicMock()

        # Simulate cell change where item is None
        self.widget._on_table_checkbox(0, 0)

        # Verify no crash and selection unchanged
        self.assertEqual(len(self.widget._selected_elements), 0)

    @unittest.skipIf(not _HAVE_MPL, "Matplotlib not available")
    def test_checkbox_handler_ignores_none_record(self):
        """Verify _on_table_checkbox handles None record gracefully."""
        self.widget._selected_elements = {}
        self.widget.refresh = MagicMock()

        # Mock item with None record
        mock_item = MagicMock()
        mock_item.data.return_value = None
        self.widget._table_widget.item.return_value = mock_item

        # Simulate cell change
        self.widget._on_table_checkbox(0, 0)

        # Verify no crash and selection unchanged
        self.assertEqual(len(self.widget._selected_elements), 0)

    @unittest.skipIf(not _HAVE_MPL, "Matplotlib not available")
    def test_checkbox_handler_ignores_missing_object_id(self):
        """Verify _on_table_checkbox handles missing object_id gracefully."""
        self.widget._selected_elements = {}
        self.widget.refresh = MagicMock()

        # Mock item with record missing object_id
        mock_item = MagicMock()
        mock_item.data.return_value = {"type": "weir", "station_m": 100.0}
        self.widget._table_widget.item.return_value = mock_item

        # Simulate cell change
        self.widget._on_table_checkbox(0, 0)

        # Verify no crash and selection unchanged
        self.assertEqual(len(self.widget._selected_elements), 0)

    @unittest.skipIf(not _HAVE_MPL, "Matplotlib not available")
    def test_checkbox_column_header_labels_updated(self):
        """Verify header labels include 'Select' as first column."""
        mock_data = MagicMock()
        mock_data._coupling_records = [
            {"object_id": "struct_1", "type": "weir", "station_m": 100.0},
        ]
        self.widget._result_data = mock_data

        self.widget._populate_table()

        # Verify header labels
        header = self.widget._table_widget.horizontalHeader()
        self.assertEqual(header.sectionText(0), "Select")
        self.assertEqual(header.sectionText(1), "Type")
        self.assertEqual(header.sectionText(2), "Station (m)")
        self.assertEqual(header.sectionText(3), "Elevation (m)")
        self.assertEqual(header.sectionText(4), "Discharge (cms)")

    @unittest.skipIf(not _HAVE_MPL, "Matplotlib not available")
    def test_checkbox_column_inserted_before_data_columns(self):
        """Verify checkbox column is inserted before existing data columns."""
        mock_data = MagicMock()
        mock_data._coupling_records = [
            {"object_id": "struct_1", "type": "weir", "station_m": 100.0},
        ]
        self.widget._result_data = mock_data

        self.widget._populate_table()

        # Verify column count (1 checkbox + 4 data columns)
        self.assertEqual(self.widget._table_widget.columnCount(), 5)

        # Verify checkbox is at index 0
        checkbox_item = self.widget._table_widget.item(0, 0)
        self.assertIsNotNone(checkbox_item)

        # Verify data columns start at index 1
        data_item = self.widget._table_widget.item(0, 1)
        self.assertIsNotNone(data_item)

    @unittest.skipIf(not _HAVE_MPL, "Matplotlib not available")
    def test_checkbox_handler_calls_refresh(self):
        """Verify _on_table_checkbox calls refresh after updating selection."""
        self.widget._selected_elements = {}
        self.widget.refresh = MagicMock()

        mock_data = MagicMock()
        mock_data._coupling_records = [
            {"object_id": "struct_1", "type": "weir", "station_m": 100.0},
        ]
        self.widget._result_data = mock_data
        self.widget._populate_table()

        # Toggle checkbox
        checkbox_item = self.widget._table_widget.item(0, 0)
        checkbox_item.setCheckState(QtCore.Qt.Checked)
        self.widget._on_table_checkbox(0, 0)

        # Verify refresh was called
        self.widget.refresh.assert_called_once()


if __name__ == "__main__":
    unittest.main()