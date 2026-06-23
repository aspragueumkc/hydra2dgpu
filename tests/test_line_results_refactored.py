"""Tests for line_results dialog refactoring — service layer extraction."""

import os
import tempfile
import unittest

import numpy as np

from swe2d.results.profile_service import extract_profile_arrays
from swe2d.results.export_service import export_table_to_csv


class TestExtractProfileArrays(unittest.TestCase):
    """profile_service.extract_profile_arrays converts record dicts to arrays."""

    EMPTY_KEYS = {"station_m", "wse_m", "bed_m", "depth_m", "wet"}

    def test_empty_records_returns_all_keys_as_empty_arrays(self):
        result = extract_profile_arrays([])
        for k in self.EMPTY_KEYS:
            self.assertIn(k, result)
            self.assertEqual(0, len(result[k]), f"key={k} should be empty")

    def test_single_record_returns_correct_values(self):
        records = [
            {"station_m": 0.0, "wse_m": 5.0, "bed_m": 4.0,
             "depth_m": 1.0, "wet": 1},
        ]
        result = extract_profile_arrays(records)
        self.assertEqual(1, len(result["station_m"]))
        self.assertAlmostEqual(0.0, result["station_m"][0])
        self.assertAlmostEqual(5.0, result["wse_m"][0])
        self.assertAlmostEqual(4.0, result["bed_m"][0])
        self.assertAlmostEqual(1.0, result["depth_m"][0])
        self.assertEqual(1.0, result["wet"][0])

    def test_sorts_by_station_m(self):
        records = [
            {"station_m": 3.0, "wse_m": 7.0, "bed_m": 4.0, "depth_m": 3.0,
             "wet": 1},
            {"station_m": 1.0, "wse_m": 5.0, "bed_m": 4.0, "depth_m": 1.0,
             "wet": 1},
            {"station_m": 2.0, "wse_m": 6.0, "bed_m": 4.0, "depth_m": 2.0,
             "wet": 1},
        ]
        result = extract_profile_arrays(records)
        np.testing.assert_array_almost_equal(
            result["station_m"], [1.0, 2.0, 3.0],
        )
        np.testing.assert_array_almost_equal(
            result["wse_m"], [5.0, 6.0, 7.0],
        )

    def test_missing_metric_keys_become_nan(self):
        records = [{"station_m": 0.0}]
        result = extract_profile_arrays(records)
        self.assertTrue(np.isnan(result["wse_m"][0]))
        self.assertTrue(np.isnan(result["bed_m"][0]))
        self.assertTrue(np.isnan(result["depth_m"][0]))
        self.assertTrue(np.isnan(result["wet"][0]))

    def test_additional_keys_preserved(self):
        records = [
            {"station_m": 0.0, "wse_m": 5.0, "bed_m": 4.0, "depth_m": 1.0,
             "wet": 1, "velocity_ms": 0.5, "fr": 0.1,
             "flow_qn": 2.0},
        ]
        result = extract_profile_arrays(records)
        self.assertIn("velocity_ms", result)
        self.assertAlmostEqual(0.5, result["velocity_ms"][0])
        self.assertIn("fr", result)
        self.assertAlmostEqual(0.1, result["fr"][0])
        self.assertIn("flow_qn", result)
        self.assertAlmostEqual(2.0, result["flow_qn"][0])

    def test_all_values_are_ndarray(self):
        records = [
            {"station_m": 0.0, "wse_m": 5.0, "bed_m": 4.0, "depth_m": 1.0,
             "wet": 1},
        ]
        result = extract_profile_arrays(records)
        for v in result.values():
            self.assertIsInstance(v, np.ndarray)

    def test_wet_field_promoted_to_float(self):
        records = [
            {"station_m": 0.0, "wse_m": 5.0, "bed_m": 4.0, "depth_m": 1.0,
             "wet": True},
        ]
        result = extract_profile_arrays(records)
        self.assertEqual(np.float64, result["wet"].dtype)
        self.assertEqual(1.0, result["wet"][0])

    def test_unsorted_flag_skips_sort(self):
        records = [
            {"station_m": 2.0, "wse_m": 6.0, "bed_m": 4.0, "depth_m": 2.0,
             "wet": 1},
            {"station_m": 1.0, "wse_m": 5.0, "bed_m": 4.0, "depth_m": 1.0,
             "wet": 1},
        ]
        result = extract_profile_arrays(records, sort_by_station=False)
        np.testing.assert_array_almost_equal(
            result["station_m"], [2.0, 1.0],
        )


class TestExportCsvDelegation(unittest.TestCase):
    """Verify export_table_to_csv works for the pattern used by line_results."""

    def test_writes_headers_and_rows(self):
        headers = ["Station (m)", "Depth (m)"]
        rows = [["0.0", "1.0"], ["1.0", "1.5"]]
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w+",
                                         delete=False) as f:
            path = f.name
        try:
            export_table_to_csv(path, headers, rows)
            with open(path, "r", newline="") as f:
                content = f.read()
            lines = [l for l in content.replace("\r\n", "\n").split("\n") if l]
            self.assertEqual(3, len(lines))
            self.assertEqual("Station (m),Depth (m)", lines[0])
            self.assertEqual("0.0,1.0", lines[1])
            self.assertEqual("1.0,1.5", lines[2])
        finally:
            os.unlink(path)

    def test_empty_rows_produces_header_only(self):
        headers = ["A", "B"]
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w+",
                                         delete=False) as f:
            path = f.name
        try:
            export_table_to_csv(path, headers, [])
            with open(path, "r", newline="") as f:
                content = f.read()
            # csv.writer uses \r\n; normalize for cross-platform
            content = content.replace("\r\n", "\n")
            self.assertEqual(content, "A,B\n")
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
