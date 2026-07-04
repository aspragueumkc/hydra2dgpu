"""Diagnostic test: profile viewer must fall back to GPKG when live profile is empty.

Bug: studio_viewer_profile_pg.py uses a ternary that picks EITHER live data
OR GPKG path based on whether _live_times is non-empty.  When live times
exist (from overlay snapshots) but _live_line_profile doesn't have the
requested line_id, the profile viewer gets empty data and shows nothing —
even though the GPKG has the baked profile.

The OLD matplotlib code always used rec.gpkg_path and never had this issue.
"""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestProfileFallbackToGpkg(unittest.TestCase):
    """Profile loading must fall back to GPKG when live profile is empty."""

    def test_live_path_returns_empty_when_profile_not_populated(self):
        """When _live_times is non-empty but _live_line_profile is empty for
        the requested line_id, load_baked_line_profile(data, ...) returns {}."""
        from swe2d.results.data import SWE2DResultsData
        from swe2d.services.gpkg_persistence_service import load_baked_line_profile

        data = SWE2DResultsData()
        # Simulate: live snapshots exist (from overlay), but no profile data
        data._live_times = np.array([0.0, 10.0, 20.0])
        data._live_h = np.zeros((3, 5))
        data._live_hu = np.zeros((3, 5))
        data._live_hv = np.zeros((3, 5))
        # _live_line_profile is empty — populate_live_line_metrics wasn't called
        # or didn't produce data for this line_id
        assert data._live_line_profile == {}

        # Live path returns empty
        result = load_baked_line_profile(data, "run1", 1, 10.0)
        self.assertEqual(result, {})

    def test_gpkg_path_works_when_live_profile_empty(self):
        """When GPKG has baked profile data, load_baked_line_profile from the
        GPKG path returns the data even when the live path would fail."""
        from swe2d.services.gpkg_persistence_service import (
            load_baked_line_profile,
            persist_baked_line_profile,
        )

        tmp = tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False)
        tmp.close()
        try:
            gpkg = tmp.name
            run_id = "test_run"
            line_id = 1
            n_sta = 5
            n_ts = 3
            station = np.linspace(0, 100, n_sta, dtype=np.float64)
            times = np.array([0.0, 10.0, 20.0], dtype=np.float64)
            depth = np.random.rand(n_ts, n_sta)
            vel = np.random.rand(n_ts, n_sta)
            wse = np.random.rand(n_ts, n_sta) + 100
            bed = np.full((n_ts, n_sta), 95.0)
            qn = np.random.rand(n_ts, n_sta)
            fr = np.random.rand(n_ts, n_sta)
            wet = np.ones((n_ts, n_sta), dtype=np.int32)

            persist_baked_line_profile(
                gpkg, run_id, line_id, "test_line",
                station, times, depth, vel, wse, bed, qn, fr, wet,
            )

            # GPKG path works
            result = load_baked_line_profile(gpkg, run_id, line_id, 10.0)
            self.assertIn("station_m", result)
            self.assertEqual(result["station_m"].size, n_sta)
            self.assertIn("wse_m", result)
            self.assertIn("bed_m", result)

        finally:
            os.unlink(gpkg)

    def test_profile_viewer_ternary_does_not_fall_back(self):
        """Inspect the source of studio_viewer_profile_pg.py refresh().

        The current code uses a ternary:
            data if _live_times.size > 0 else rec.gpkg_path

        This means when _live_times is non-empty, it ONLY tries the live path.
        If the live path returns empty, there's no fallback to GPKG.

        After the fix, the code should try live first, then GPKG.
        """
        import inspect
        import re
        from swe2d.workbench.views import studio_viewer_profile_pg

        src = inspect.getsource(studio_viewer_profile_pg.PGProfileWidget.refresh)

        # Count actual prof_data assignments from load_baked_line_profile
        # (excluding imports).  Currently 1 (the ternary).  After the fix
        # it should be 2 (live attempt + GPKG fallback).
        calls = re.findall(
            r'prof_data\s*=\s*load_baked_line_profile\s*\(', src
        )
        self.assertGreaterEqual(
            len(calls), 2,
            f"Profile refresh() has {len(calls)} load_baked_line_profile "
            f"assignment(s) — expected >= 2 (live + GPKG fallback). "
            f"The ternary picks ONE source with no fallback.",
        )


if __name__ == "__main__":
    unittest.main()