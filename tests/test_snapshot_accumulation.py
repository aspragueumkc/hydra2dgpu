"""Test that snapshots accumulate across multiple read_snapshots() calls.

Reproduces the bug where a live readback mid-run clears the device buffer,
and the end-of-run readback only returns post-live-readback snapshots.
The finalizer must see ALL snapshots, not just the last batch.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from swe2d.results.data import SWE2DResultsData


class TestSnapshotAccumulation(unittest.TestCase):

    def _make_snaps(self, start, count, n_cells=1):
        """Build list of (t_s, h, hu, hv) tuples."""
        return [
            (float(start + i),
             np.full(n_cells, float(start + i), dtype=np.float64),
             np.zeros(n_cells, dtype=np.float64),
             np.zeros(n_cells, dtype=np.float64))
            for i in range(count)
        ]

    def test_merge_preserves_earlier_readback(self):
        """Simulates: live readback [0..3], then end-of-run readback [4..7].

        Without merging, set_live_snapshot_timesteps(batch2) replaces
        batch1, and the finalizer only sees [4..7].  With merging,
        the finalizer sees all [0..7].
        """
        data = SWE2DResultsData()

        # Simulate first readback (live "Fetch Device Results")
        batch1 = self._make_snaps(0, 4)
        existing = data.get_live_snapshot_timesteps()
        data.set_live_snapshot_timesteps(existing + batch1)
        self.assertEqual(len(data.get_live_snapshot_timesteps()), 4)

        # Simulate second readback (end-of-run)
        batch2 = self._make_snaps(4, 4)
        existing = data.get_live_snapshot_timesteps()
        data.set_live_snapshot_timesteps(existing + batch2)

        all_snaps = data.get_live_snapshot_timesteps()
        self.assertEqual(len(all_snaps), 8,
                         f"Expected 8 accumulated snapshots, got {len(all_snaps)}")
        times = [s[0] for s in all_snaps]
        self.assertEqual(times, list(range(8)))

    def test_replace_loses_earlier_readback(self):
        """Documents the bug: replace semantics lose earlier data.

        This test verifies that WITHOUT merging, data is lost —
        proving the fix is necessary.
        """
        data = SWE2DResultsData()

        batch1 = self._make_snaps(0, 4)
        data.set_live_snapshot_timesteps(batch1)
        self.assertEqual(len(data.get_live_snapshot_timesteps()), 4)

        # Replace (bug behavior)
        batch2 = self._make_snaps(4, 4)
        data.set_live_snapshot_timesteps(batch2)

        all_snaps = data.get_live_snapshot_timesteps()
        self.assertEqual(len(all_snaps), 4,
                         "Replace semantics lost the first 4 snapshots")
        times = [s[0] for s in all_snaps]
        self.assertEqual(times, [4.0, 5.0, 6.0, 7.0])


if __name__ == "__main__":
    unittest.main()
