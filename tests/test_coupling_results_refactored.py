"""Tests for coupling results refactoring — service layer extraction."""
from __future__ import annotations

import unittest

import numpy as np

from tests.mocks.qgis_env import install_qgis_mocks
install_qgis_mocks()

from swe2d.results.overlay_service import prepare_coupling_timeseries


class TestPrepareCouplingTimeseries(unittest.TestCase):
    """Test the extracted time-series data prep function."""

    def test_empty_records_returns_empty_dict(self):
        result = prepare_coupling_timeseries([])
        self.assertEqual(result, {})

    def test_single_record_grouped_by_object_id(self):
        records = [
            {
                "t_s": 0.0, "value": 1.5,
                "object_id": "weir1", "object_name": "Weir 1",
            },
        ]
        result = prepare_coupling_timeseries(records)
        self.assertIn("weir1", result)
        entry = result["weir1"]
        np.testing.assert_array_almost_equal(entry["times"], [0.0])
        np.testing.assert_array_almost_equal(entry["values"], [1.5])
        self.assertEqual(entry["name"], "Weir 1")

    def test_times_converted_to_hours(self):
        records = [
            {"t_s": 0.0, "value": 1.0, "object_id": "o1", "object_name": ""},
            {"t_s": 3600.0, "value": 2.0, "object_id": "o1", "object_name": ""},
        ]
        result = prepare_coupling_timeseries(records)
        entry = result["o1"]
        np.testing.assert_array_almost_equal(entry["times"], [0.0, 1.0])
        np.testing.assert_array_almost_equal(entry["values"], [1.0, 2.0])

    def test_non_finite_values_filtered_out(self):
        records = [
            {"t_s": 0.0, "value": 1.0, "object_id": "o1", "object_name": ""},
            {"t_s": 3600.0, "value": float("nan"), "object_id": "o1", "object_name": ""},
        ]
        result = prepare_coupling_timeseries(records)
        entry = result["o1"]
        self.assertEqual(len(entry["times"]), 1)
        self.assertEqual(len(entry["values"]), 1)

    def test_non_finite_t_s_filtered_out(self):
        records = [
            {"t_s": float("inf"), "value": 1.0, "object_id": "o1", "object_name": ""},
        ]
        result = prepare_coupling_timeseries(records)
        self.assertEqual(result, {})

    def test_multiple_objects_separate_groups(self):
        records = [
            {"t_s": 0.0, "value": 1.0, "object_id": "a", "object_name": "Obj A"},
            {"t_s": 0.0, "value": 2.0, "object_id": "b", "object_name": "Obj B"},
        ]
        result = prepare_coupling_timeseries(records)
        self.assertIn("a", result)
        self.assertIn("b", result)
        self.assertEqual(result["a"]["name"], "Obj A")
        self.assertEqual(result["b"]["name"], "Obj B")

    def test_sorted_by_time_per_object(self):
        records = [
            {"t_s": 7200.0, "value": 3.0, "object_id": "o1", "object_name": ""},
            {"t_s": 0.0, "value": 1.0, "object_id": "o1", "object_name": ""},
            {"t_s": 3600.0, "value": 2.0, "object_id": "o1", "object_name": ""},
        ]
        result = prepare_coupling_timeseries(records)
        entry = result["o1"]
        np.testing.assert_array_almost_equal(entry["times"], [0.0, 1.0, 2.0])
        np.testing.assert_array_almost_equal(entry["values"], [1.0, 2.0, 3.0])

    def test_value_error_skipped(self):
        records = [
            {"t_s": 0.0, "value": "not_a_number", "object_id": "o1", "object_name": ""},
        ]
        result = prepare_coupling_timeseries(records)
        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
