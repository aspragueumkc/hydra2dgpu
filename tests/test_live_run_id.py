"""Tests for _live_run_id tracking on SWE2DResultsData."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestLiveRunId(unittest.TestCase):
    def test_live_run_id_starts_empty(self):
        from swe2d.results.data import SWE2DResultsData
        data = SWE2DResultsData()
        self.assertEqual(data._live_run_id, "")

    def test_clear_live_snapshots_clears_live_run_id(self):
        from swe2d.results.data import SWE2DResultsData
        data = SWE2DResultsData()
        data._live_run_id = "run_123"
        data._live_times = np.array([0.0, 10.0])
        data.clear_live_snapshots()
        self.assertEqual(data._live_run_id, "")
        self.assertEqual(data._live_times.size, 0)

    def test_load_profile_live_filters_by_run_id(self):
        """load_baked_line_profile only returns live data when _live_run_id matches."""
        from swe2d.results.data import SWE2DResultsData
        from swe2d.services.gpkg_persistence_service import load_baked_line_profile

        data = SWE2DResultsData()
        data._live_run_id = "run_A"
        data._live_times = np.array([0.0, 10.0])
        data._live_line_profile[1] = {
            "station_m": np.array([0.0, 50.0, 100.0]),
            "wse_m": np.full((2, 3), 105.0),
            "bed_m": np.full((2, 3), 95.0),
            "depth_m": np.full((2, 3), 10.0),
            "velocity_ms": np.full((2, 3), 1.0),
            "flow_qn": np.full((2, 3), 50.0),
            "fr": np.full((2, 3), 0.3),
            "wet": np.ones((2, 3), dtype=np.int32),
        }

        # Matching run_id -> returns live data
        result = load_baked_line_profile(data, "run_A", 1, 5.0)
        self.assertIn("station_m", result)

        # Non-matching run_id -> returns empty (triggers GPKG fallback)
        result = load_baked_line_profile(data, "run_B", 1, 5.0)
        self.assertEqual(result, {})

    def test_load_timeseries_live_filters_by_run_id(self):
        """load_baked_line_timeseries only returns live data when _live_run_id matches."""
        from swe2d.results.data import SWE2DResultsData
        from swe2d.services.gpkg_persistence_service import load_baked_line_timeseries

        data = SWE2DResultsData()
        data._live_run_id = "run_A"
        data._live_times = np.array([0.0, 10.0])
        data._live_line_ts[1] = {
            "line_name": "test",
            "depth_m": np.array([1.0, 2.0]),
            "velocity_ms": np.array([0.5, 1.0]),
            "wse_m": np.array([100.0, 101.0]),
            "bed_m": np.array([95.0, 95.0]),
            "flow_cms": np.array([10.0, 20.0]),
            "wet_frac": np.array([1.0, 1.0]),
            "fr": np.array([0.3, 0.4]),
        }

        # Matching run_id -> returns live data
        result = load_baked_line_timeseries(data, "run_A", 1)
        self.assertIn("depth_m", result)

        # Non-matching run_id -> returns empty
        result = load_baked_line_timeseries(data, "run_B", 1)
        self.assertEqual(result, {})
