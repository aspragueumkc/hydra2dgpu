"""Smoke test: load line results from GPKG and create matplotlib plot.

Uses the services layer (SWE2DResultsData, queries) to verify the full
data pipeline works end-to-end without any GUI.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

TEST_GPKG = os.path.join(
    os.path.dirname(__file__), "..", "example_project", "culvert_test_results.gpkg"
)


class TestLineResultsPlotSmoke(unittest.TestCase):
    """Load line results from a real GPKG and verify the data pipeline."""

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(TEST_GPKG):
            raise unittest.SkipTest(f"Test GPKG not found: {TEST_GPKG}")
        from swe2d.results.data import SWE2DResultsData

        cls.data = SWE2DResultsData()
        cls.records = cls.data.discover_runs()
        # Filter out snapshot runs
        cls.runs = [r for r in cls.records if r.enabled and not r.run_id.startswith("swe2d_snapshot_")]
        if not cls.runs:
            cls.runs = [r for r in cls.records if r.enabled]

    def test_discover_runs(self):
        self.assertGreater(len(self.records), 0, "No runs found in GPKG")
        self.assertGreater(len(self.runs), 0, "No non-snapshot runs found")

    def test_get_line_ids(self):
        line_ids = self.data.get_line_ids()
        self.assertIsInstance(line_ids, list)
        self.assertGreater(len(line_ids), 0, "No line IDs found")
        # Should be (int, str) tuples
        lid = line_ids[0]
        if isinstance(lid, tuple):
            self.assertIsInstance(lid[0], int)
        else:
            self.assertIsInstance(lid, int)

    def test_load_timeseries(self):
        line_ids = self.data.get_line_ids()
        if not line_ids:
            self.skipTest("No line IDs")
        lid = line_ids[0][0] if isinstance(line_ids[0], tuple) else line_ids[0]
        ts = self.data.load_timeseries(self.runs[0], lid, "flow_cms")
        self.assertIsInstance(ts, dict)
        for key in ("t_s", "depth_m", "velocity_ms", "wse_m", "bed_m", "flow_cms"):
            self.assertIn(key, ts, f"Missing key: {key}")
            self.assertGreater(len(ts[key]), 0, f"Empty array for {key}")

    def test_plot_saved(self):
        line_ids = self.data.get_line_ids()
        if not line_ids:
            self.skipTest("No line IDs")
        lid = line_ids[0][0] if isinstance(line_ids[0], tuple) else line_ids[0]
        ts = self.data.load_timeseries(self.runs[0], lid, "flow_cms")
        if not ts:
            self.skipTest("No timeseries data")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        t_hr = ts["t_s"] / 3600.0
        ax.plot(t_hr, ts["flow_cms"], "b-", linewidth=1)
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Flow (m³/s)")
        ax.set_title(f"Line {lid} — {self.runs[0].run_id}")
        ax.grid(True, alpha=0.3)

        out_path = os.path.join(os.path.dirname(__file__), "..", "example_project", "test_line_plot.png")
        plt.savefig(out_path, dpi=100)
        plt.close(fig)
        self.assertTrue(os.path.exists(out_path), f"Plot not saved to {out_path}")


if __name__ == "__main__":
    unittest.main()
