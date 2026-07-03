"""Regression tests for coupling snapshot write/read bugs.

Bug #1: append_coupling_snapshot used a global _coupling_snap_idx counter
that advanced per ROW, but arrays are sized per TIME STEP.  With multiple
rows per snap (drainage depth+invert, link flow+length, structure flow,
culvert 7-metric block), the counter overflows the pre-allocated arrays
within the first ~10 snaps, silently dropping ~90% of coupling data.

The fix: pass snap_idx explicitly and only advance it per snap (not per row).
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestCouplingSnapshotIndexing(unittest.TestCase):
    """Bug #1: append_coupling_snapshot must use per-snap indexing."""

    def test_multi_row_per_snap_does_not_overflow(self):
        from swe2d.results.data import SWE2DResultsData

        data = SWE2DResultsData()
        # 50 time steps, 2 coupling keys
        keys = [
            ("drainage_node", "n1", "depth"),
            ("drainage_node", "n1", "invert"),
        ]
        data.preallocate_output_schedule(
            n_line_snaps=50,
            coupling_keys=keys,
            coupling_object_names={k: k[1] for k in keys},
        )

        # Simulate the reporter: 50 snaps, each producing 2 rows (depth + invert)
        for snap_idx in range(50):
            t_s = float(snap_idx) * 0.5
            data.append_coupling_snapshot(
                {"t_s": t_s, "component": "drainage_node",
                 "object_id": "n1", "metric": "depth", "value": float(snap_idx)},
                snap_idx=snap_idx,
            )
            data.append_coupling_snapshot(
                {"t_s": t_s, "component": "drainage_node",
                 "object_id": "n1", "metric": "invert", "value": 10.0 + float(snap_idx)},
                snap_idx=snap_idx,
            )

        # Each key's t_s array should contain exactly 50 entries with
        # monotonically increasing timestamps and NO zero padding in between.
        depth_key = ("drainage_node", "n1", "depth")
        depth_times = data._live_coupling[depth_key]["t_s"]
        depth_values = data._live_coupling[depth_key]["values"]
        self.assertEqual(depth_times.size, 50)
        # All entries must be finite (no zero padding)
        self.assertTrue(np.all(np.isfinite(depth_times)))
        # Timestamps must be monotonically increasing
        diffs = np.diff(depth_times)
        self.assertTrue(np.all(diffs > 0), f"non-monotonic times: {depth_times[:10]}")
        # First entry at t=0, last at t=24.5
        self.assertAlmostEqual(depth_times[0], 0.0)
        self.assertAlmostEqual(depth_times[-1], 24.5)
        # Values: snap_idx 0..49
        np.testing.assert_array_equal(depth_values, np.arange(50, dtype=np.float64))

    def test_single_row_per_snap_works(self):
        """Backwards-compatible behavior for single-row-per-snap callers."""
        from swe2d.results.data import SWE2DResultsData

        data = SWE2DResultsData()
        keys = [("structure", "s1", "flow")]
        data.preallocate_output_schedule(
            n_line_snaps=20,
            coupling_keys=keys,
            coupling_object_names={k: "S1" for k in keys},
        )

        for snap_idx in range(20):
            data.append_coupling_snapshot(
                {"t_s": float(snap_idx), "component": "structure",
                 "object_id": "s1", "metric": "flow", "value": float(snap_idx) * 2},
                snap_idx=snap_idx,
            )

        times = data._live_coupling[keys[0]]["t_s"]
        values = data._live_coupling[keys[0]]["values"]
        self.assertEqual(times.size, 20)
        np.testing.assert_array_equal(values, np.arange(20, dtype=np.float64) * 2)


class TestGetLiveCouplingSnapshotRows(unittest.TestCase):
    """The reader must produce clean per-snap rows after the fix."""

    def test_no_zero_padding_between_valid_entries(self):
        from swe2d.results.data import SWE2DResultsData

        data = SWE2DResultsData()
        keys = [
            ("drainage_node", "n1", "depth"),
            ("drainage_node", "n2", "depth"),
        ]
        data.preallocate_output_schedule(
            n_line_snaps=10,
            coupling_keys=keys,
            coupling_object_names={k: k[1] for k in keys},
        )

        for snap_idx in range(10):
            t_s = float(snap_idx)
            data.append_coupling_snapshot(
                {"t_s": t_s, "component": "drainage_node",
                 "object_id": "n1", "metric": "depth", "value": float(snap_idx)},
                snap_idx=snap_idx,
            )
            data.append_coupling_snapshot(
                {"t_s": t_s, "component": "drainage_node",
                 "object_id": "n2", "metric": "depth", "value": float(snap_idx) + 100},
                snap_idx=snap_idx,
            )
        # In production the reporter maintains _coupling_snap_idx; in unit
        # tests we set it explicitly to expose the written window.
        data._coupling_snap_idx = 10

        rows = data.get_live_coupling_snapshot_rows()
        # 10 snaps × 2 keys = 20 rows
        self.assertEqual(len(rows), 20)
        # All rows must have finite t_s and values
        for r in rows:
            self.assertTrue(np.isfinite(r["t_s"]), f"non-finite t_s: {r}")
            self.assertTrue(np.isfinite(r["value"]), f"non-finite value: {r}")


if __name__ == "__main__":
    unittest.main()