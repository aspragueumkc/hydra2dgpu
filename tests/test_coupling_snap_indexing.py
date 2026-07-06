"""Regression tests for coupling snapshot write/read with dynamic lists."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestCouplingSnapshotIndexing(unittest.TestCase):

    def test_multi_row_per_snap(self):
        from swe2d.results.data import SWE2DResultsData

        data = SWE2DResultsData()
        keys = [
            ("drainage_node", "n1", "depth"),
            ("drainage_node", "n1", "invert"),
        ]
        data.init_coupling_storage(
            coupling_keys=keys,
            coupling_object_names={k: k[1] for k in keys},
        )

        for snap_idx in range(50):
            t_s = float(snap_idx) * 0.5
            data.append_coupling_snapshot(
                {"t_s": t_s, "component": "drainage_node",
                 "object_id": "n1", "metric": "depth", "value": float(snap_idx)})
            data.append_coupling_snapshot(
                {"t_s": t_s, "component": "drainage_node",
                 "object_id": "n1", "metric": "invert", "value": 10.0 + float(snap_idx)})

        depth_key = ("drainage_node", "n1", "depth")
        depth_times = data._live_coupling[depth_key]["t_s"]
        depth_values = data._live_coupling[depth_key]["values"]
        self.assertEqual(len(depth_times), 50)
        self.assertTrue(all(0 <= v < 50 for v in depth_values),
                        f"values should be 0..49, got first 5: {depth_values[:5]}")
        self.assertAlmostEqual(depth_times[0], 0.0)
        self.assertAlmostEqual(depth_times[-1], 24.5)

    def test_single_row_per_snap(self):
        from swe2d.results.data import SWE2DResultsData

        data = SWE2DResultsData()
        keys = [("structure", "s1", "flow")]
        data.init_coupling_storage(
            coupling_keys=keys,
            coupling_object_names={k: "S1" for k in keys},
        )

        for snap_idx in range(20):
            data.append_coupling_snapshot(
                {"t_s": float(snap_idx), "component": "structure",
                 "object_id": "s1", "metric": "flow", "value": float(snap_idx) * 2})

        times = data._live_coupling[keys[0]]["t_s"]
        values = data._live_coupling[keys[0]]["values"]
        self.assertEqual(len(times), 20)
        self.assertEqual(values, [float(i) * 2 for i in range(20)])


class TestGetLiveCouplingSnapshotRows(unittest.TestCase):

    def test_rows_from_lists(self):
        from swe2d.results.data import SWE2DResultsData

        data = SWE2DResultsData()
        keys = [
            ("drainage_node", "n1", "depth"),
            ("drainage_node", "n2", "depth"),
        ]
        data.init_coupling_storage(
            coupling_keys=keys,
            coupling_object_names={k: k[1] for k in keys},
        )

        for snap_idx in range(10):
            t_s = float(snap_idx)
            data.append_coupling_snapshot(
                {"t_s": t_s, "component": "drainage_node",
                 "object_id": "n1", "metric": "depth", "value": float(snap_idx)})
            data.append_coupling_snapshot(
                {"t_s": t_s, "component": "drainage_node",
                 "object_id": "n2", "metric": "depth", "value": float(snap_idx) + 100})

        rows = data.get_live_coupling_snapshot_rows()
        self.assertEqual(len(rows), 20)
        for r in rows:
            self.assertTrue(np.isfinite(r["t_s"]), f"non-finite t_s: {r}")
            self.assertTrue(np.isfinite(r["value"]), f"non-finite value: {r}")


if __name__ == "__main__":
    unittest.main()
