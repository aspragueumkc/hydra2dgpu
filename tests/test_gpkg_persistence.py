"""Test that baked GPKG persistence actually writes and reads data.

Exercises the baked BLOB persistence functions directly.
No QGIS environment needed — pure SQLite verify.
"""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from swe2d.services.gpkg_persistence_service import (
    persist_baked_mesh,
    load_baked_mesh,
    persist_baked_results,
    load_baked_snapshot,
    persist_baked_coupling,
    load_baked_coupling_timeseries,
    persist_baked_line_ts,
    persist_baked_line_profile,
    load_baked_line_timeseries,
    load_baked_line_profile,
    load_baked_timesteps,
    collect_baked_runs_from_gpkg,
    persist_baked_pipe_cell_ts,
    load_baked_pipe_cell_ts,
)


class TestBakedGpkgPersistence(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False)
        self.gpkg_path = self.tmp.name
        self.tmp.close()
        self.mesh_name = "test_mesh"
        self.run_id = "test_run_001"

    def tearDown(self):
        if os.path.exists(self.gpkg_path):
            os.unlink(self.gpkg_path)

    def _make_baked_blob(self) -> bytes:
        """Produce a minimal but valid serialized mesh BLOB."""
        import struct
        data = struct.pack("<Q", 19)
        arrays = [
            np.array([0.0, 10.0, 10.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float64),
            np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
            np.array([0, 4], dtype=np.int64),
            np.array([0, 1, 2, 0, 2, 3], dtype=np.int64),
            np.array([100.0], dtype=np.float64),
            np.array([0.0], dtype=np.float64),
            np.array([5.0], dtype=np.float64),
            np.array([5.0], dtype=np.float64),
            np.array([0.01], dtype=np.float64),
            np.array([0], dtype=np.int64),
            np.array([0, 1, 2], dtype=np.int64),
            np.array([-1, 1, 2], dtype=np.int64),
            np.array([0, 1, 2], dtype=np.int64),
            np.array([1, 2, 3], dtype=np.int64),
            np.array([0.0, 10.0, 0.0], dtype=np.float64),
            np.array([-1.0, 0.0, 1.0], dtype=np.float64),
            np.array([10.0, 10.0, 10.0], dtype=np.float64),
            np.array([0, 1, 1], dtype=np.int32),
        ]
        for arr in arrays:
            raw = np.ascontiguousarray(arr).tobytes()
            data += struct.pack("<Q", len(raw)) + raw
        return data

    def _log(self, msg):
        pass

    # ── Baked mesh ──────────────────────────────────────────────────────

    def test_baked_mesh_persist_and_load(self):
        blob = self._make_baked_blob()
        persist_baked_mesh(self.gpkg_path, self.mesh_name, blob,
                           n_nodes=4, n_cells=1, log_fn=self._log)
        loaded = load_baked_mesh(self.gpkg_path, self.mesh_name)
        self.assertIsNotNone(loaded)
        self.assertIsInstance(loaded, bytes)
        self.assertEqual(loaded, blob)

    def test_baked_mesh_not_found(self):
        loaded = load_baked_mesh(self.gpkg_path, "nonexistent")
        self.assertIsNone(loaded)

    # ── Baked results ───────────────────────────────────────────────────

    def test_baked_results_persist_and_load(self):
        t_s = np.array([0.0, 10.0], dtype=np.float64)
        h = np.array([[1.0], [2.0]], dtype=np.float64)
        hu = np.array([[0.1], [0.2]], dtype=np.float64)
        hv = np.array([[0.0], [0.0]], dtype=np.float64)
        snapshots = [(t_s[0], h[0], hu[0], hv[0]),
                     (t_s[1], h[1], hu[1], hv[1])]
        persist_baked_results(self.gpkg_path, self.run_id, self.mesh_name,
                              snapshots, log_fn=self._log)
        loaded = load_baked_snapshot(self.gpkg_path, self.run_id, 0.0)
        self.assertIsNotNone(loaded)
        np.testing.assert_array_almost_equal(loaded["h"], h[0])

    # ── Baked coupling ──────────────────────────────────────────────────

    def test_baked_coupling_persist_and_load(self):
        times = np.array([0.0, 10.0], dtype=np.float64)
        values = np.array([0.0, 5.0], dtype=np.float64)
        persist_baked_coupling(self.gpkg_path, self.run_id,
                               "structure", "s1", "culvert", "flow",
                               times, values, log_fn=self._log)
        loaded = load_baked_coupling_timeseries(
            self.gpkg_path, self.run_id, "structure", "s1", "flow")
        self.assertIsNotNone(loaded)
        self.assertEqual(len(loaded), 2)
        np.testing.assert_array_almost_equal(loaded[0], times)
        np.testing.assert_array_almost_equal(loaded[1], values)

    # ── Baked line timeseries ───────────────────────────────────────────

    def test_baked_line_ts_persist_and_load(self):
        t_s = np.array([0.0, 10.0], dtype=np.float64)
        depth = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float64)
        velocity = np.zeros_like(depth)
        wse = np.ones_like(depth)
        bed = np.zeros_like(depth)
        flow = np.zeros_like(depth)
        wet_frac = np.ones_like(depth)
        fr = np.ones_like(depth) * 0.03
        persist_baked_line_ts(self.gpkg_path, self.run_id, 1, "line_1",
                              t_s, depth, velocity, wse, bed,
                              flow, wet_frac, fr, log_fn=self._log)
        loaded = load_baked_line_timeseries(self.gpkg_path, self.run_id, 1)
        self.assertIsNotNone(loaded)
        self.assertIn("t_s", loaded)
        np.testing.assert_array_almost_equal(loaded["t_s"], t_s)

    # ── Baked line profile ──────────────────────────────────────────────

    def test_baked_line_profile_persist_and_load(self):
        station_m = np.array([0.0, 5.0, 10.0], dtype=np.float64)
        times = np.array([5.0], dtype=np.float64)
        depth_m = np.array([[0.5, 0.6, 0.7]], dtype=np.float64)
        velocity_ms = np.zeros_like(depth_m)
        wse_m = np.ones_like(depth_m)
        bed_m = np.zeros_like(depth_m)
        flow_qn = np.zeros_like(depth_m)
        fr = np.ones_like(depth_m) * 0.03
        wet = np.ones(depth_m.shape, dtype=np.int32)
        persist_baked_line_profile(self.gpkg_path, self.run_id, 1, "profile_1",
                                   station_m, times, depth_m, velocity_ms,
                                   wse_m, bed_m, flow_qn, fr, wet,
                                   log_fn=self._log)
        loaded = load_baked_line_profile(self.gpkg_path, self.run_id, 1, 5.0)
        self.assertIsNotNone(loaded)
        self.assertIn("depth", loaded)
        np.testing.assert_array_almost_equal(loaded["depth"], depth_m[0])

    # ── Utility functions ───────────────────────────────────────────────

    def test_load_baked_timesteps(self):
        t_s = np.array([0.0, 10.0], dtype=np.float64)
        h = np.array([[1.0], [2.0]], dtype=np.float64)
        hu = np.zeros_like(h)
        hv = np.zeros_like(h)
        snapshots = [(t_s[0], h[0], hu[0], hv[0]),
                     (t_s[1], h[1], hu[1], hv[1])]
        persist_baked_results(self.gpkg_path, self.run_id, self.mesh_name,
                              snapshots, log_fn=self._log)
        timesteps = load_baked_timesteps(self.gpkg_path, self.run_id)
        self.assertIsNotNone(timesteps)
        np.testing.assert_array_almost_equal(timesteps, t_s)

    def test_collect_baked_runs_from_gpkg(self):
        t_s = np.array([0.0, 10.0], dtype=np.float64)
        h = np.array([[1.0], [2.0]], dtype=np.float64)
        hu = np.zeros_like(h)
        hv = np.zeros_like(h)
        snapshots = [(t_s[0], h[0], hu[0], hv[0]),
                     (t_s[1], h[1], hu[1], hv[1])]
        persist_baked_results(self.gpkg_path, self.run_id, self.mesh_name,
                              snapshots, log_fn=self._log)
        runs = collect_baked_runs_from_gpkg(self.gpkg_path)
        self.assertGreater(len(runs), 0)
        self.assertTrue(any(r["run_id"] == self.run_id for r in runs))


    def test_pipe_cell_ts_legacy_schema_migration(self):
        """Writing pipe-cell data to a legacy 7-column table auto-migrates it."""
        import sqlite3

        # Create the legacy 7-column table (pre-geometry schema)
        conn = sqlite3.connect(self.gpkg_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS swe2d_baked_pipe_cell_ts (
                run_id TEXT,
                link_id TEXT,
                cell_sub_idx INTEGER,
                metric TEXT,
                n_timesteps INTEGER,
                times_blob BLOB,
                values_blob BLOB,
                PRIMARY KEY (run_id, link_id, cell_sub_idx, metric))
        """)
        conn.commit()
        conn.close()

        items = [{
            "link_id": "L1",
            "cell_sub_idx": 0,
            "metric": "depth",
            "times": np.array([0.0, 1.0], dtype=np.float64),
            "values": np.array([0.5, 0.6], dtype=np.float64),
            "cell_invert": 10.0,
            "cell_width": 1.0,
            "cell_height": 1.0,
            "cell_shape_type": 0,
        }]
        # Must not raise "table has 7 columns but 11 values were supplied"
        persist_baked_pipe_cell_ts(self.gpkg_path, self.run_id, items, log_fn=self._log)

        conn = sqlite3.connect(self.gpkg_path)
        loaded = load_baked_pipe_cell_ts(conn, self.run_id)
        conn.close()

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["link_id"], "L1")
        self.assertEqual(loaded[0]["cell_sub_idx"], 0)
        self.assertEqual(loaded[0]["metric"], "depth")
        np.testing.assert_array_almost_equal(loaded[0]["times"], items[0]["times"])
        np.testing.assert_array_almost_equal(loaded[0]["values"], items[0]["values"])
        self.assertAlmostEqual(loaded[0]["cell_invert"], 10.0)
        self.assertAlmostEqual(loaded[0]["cell_width"], 1.0)
        self.assertAlmostEqual(loaded[0]["cell_height"], 1.0)
        self.assertEqual(loaded[0]["cell_shape_type"], 0)


if __name__ == "__main__":
    unittest.main()
