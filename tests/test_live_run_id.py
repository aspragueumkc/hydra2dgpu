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
