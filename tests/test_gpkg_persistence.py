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
    compute_max_tracking,
    persist_baked_coupling,
    load_baked_coupling_timeseries,
    persist_baked_line_ts,
    persist_baked_line_profile,
    load_baked_line_timeseries,
    load_baked_line_profile,
    load_baked_timesteps,
    collect_baked_runs_from_gpkg,
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
        """Produce a minimal but valid serialized mesh BLOB.

        Format: uint64 n_vectors, then for each vector: uint64 len + raw bytes.
        Vectors in order: node_x, node_y, node_z, cell_face_offsets,
        cell_face_nodes, cell_area, cell_zb, cell_cx, cell_cy, cell_inv_area,
        cell_perm, edge_c0, edge_c1, edge_n0, edge_n1, edge_nx, edge_ny,
        edge_len, edge_bc.
        """
        import struct
        n_nodes, n_cells, n_edges = 4, 1, 4
        data = struct.pack("<Q", 19)  # 19 vectors
        arrays = [
            np.array([0.0, 10.0, 10.0, 0.0], dtype=np.float64),  # node_x
            np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float64),  # node_y
            np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),    # node_z
            np.array([0, 4], dtype=np.int64),                      # cell_face_offsets
            np.array([0, 1, 2, 0, 2, 3], dtype=np.int64),         # cell_face_nodes
            np.array([100.0], dtype=np.float64),                   # cell_area
            np.array([0.0], dtype=np.float64),                     # cell_zb
            np.array([5.0], dtype=np.float64),                     # cell_cx
            np.array([5.0], dtype=np.float64),                     # cell_cy
            np.array([0.01], dtype=np.float64),                    # cell_inv_area
            np.array([0], dtype=np.int64),                         # cell_perm
            np.array([0, 1, 2], dtype=np.int64),                   # edge_c0
            np.array([-1, 1, 2], dtype=np.int64),                  # edge_c1
            np.array([0, 1, 2], dtype=np.int64),                   # edge_n0
            np.array([1, 2, 3], dtype=np.int64),                   # edge_n1
            np.array([0.0, 10.0, 0.0], dtype=np.float64),         # edge_nx
            np.array([-1.0, 0.0, 1.0], dtype=np.float64),         # edge_ny
            np.array([10.0, 10.0, 10.0], dtype=np.float64),       # edge_len
            np.array([0, 1, 1], dtype=np.int32),                   # edge_bc (BCType enum)
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
        self.assertEqual(loaded["blob"], blob)
        self.assertEqual(loaded["n_nodes"], 4)
        self.assertEqual(loaded["n_cells"], 1)

    def test_baked_mesh_not_found(self):
        loaded = load_baked_mesh(self.gpkg_path, "nonexistent")
        self.assertIsNone(loaded)

    # ── Baked results ───────────────────────────────────────────────────

    def _make_snapshot(self, n_cells=3):
        return (
            np.array([0.0, 10.0], dtype=np.float64),         # t_s
            np.array([[1.0], [2.0]], dtype=np.float64),       # h  (n_ts x n_cells)
            np.array([[0.1], [0.2]], dtype=np.float64),       # hu
            np.array([[0.0], [0.0]], dtype=np.float64),       # hv
        )

    def test_baked_results_persist_and_load(self):
        t_s, h, hu, hv = self._make_snapshot()
        persist_baked_results(self.gpkg_path, self.run_id, t_s, h, hu, hv,
                              interval_s=10.0, log_fn=self._log)
        loaded = load_baked_snapshot(self.gpkg_path, self.run_id, 10.0)
        self.assertIsNotNone(loaded)
        np.testing.assert_array_almost_equal(loaded["h"][0], h[0])

    # ── Baked coupling ──────────────────────────────────────────────────

    def test_baked_coupling_persist_and_load(self):
        t_s = np.array([0.0, 10.0], dtype=np.float64)
        comp = np.array(["structure", "structure"], dtype=object)
        oid = np.array(["s1", "s2"], dtype=object)
        oname = np.array(["culvert", "weir"], dtype=object)
        metric = np.array(["flow", "flow"], dtype=object)
        vals = np.array([0.0, 5.0], dtype=np.float64)
        persist_baked_coupling(self.gpkg_path, self.run_id,
                               t_s, comp, oid, oname, metric, vals,
                               interval_s=10.0, log_fn=self._log)
        loaded = load_baked_coupling_timeseries(self.gpkg_path, self.run_id)
        self.assertIsNotNone(loaded)
        self.assertIn("times", loaded)
        self.assertEqual(len(loaded["times"]), 2)

    # ── Baked line timeseries ───────────────────────────────────────────

    def test_baked_line_ts_persist_and_load(self):
        line_id = "line_1"
        t_s = np.array([0.0, 10.0], dtype=np.float64)
        # n_timesteps x n_vertices
        h = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float64)
        hu = np.zeros_like(h)
        hv = np.zeros_like(h)
        chk = np.array([0, 10], dtype=np.float64)
        persist_baked_line_ts(self.gpkg_path, self.run_id, line_id,
                              t_s, h, hu, hv, chk, log_fn=self._log)
        loaded = load_baked_line_timeseries(self.gpkg_path, self.run_id, line_id)
        self.assertIsNotNone(loaded)
        self.assertIn("times", loaded)
        np.testing.assert_array_almost_equal(loaded["times"], t_s)

    # ── Baked line profile ──────────────────────────────────────────────

    def test_baked_line_profile_persist_and_load(self):
        line_id = "profile_1"
        t_s = np.array([5.0], dtype=np.float64)
        h = np.array([[0.5, 0.6, 0.7]], dtype=np.float64)
        hu = np.zeros_like(h)
        hv = np.zeros_like(h)
        chk = np.array([5.0], dtype=np.float64)
        vertices = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]], dtype=np.float64)
        persist_baked_line_profile(self.gpkg_path, self.run_id, line_id,
                                   t_s, h, hu, hv, chk, vertices, log_fn=self._log)
        loaded = load_baked_line_profile(self.gpkg_path, self.run_id, line_id)
        self.assertIsNotNone(loaded)
        self.assertIn("h", loaded)
        self.assertEqual(loaded["h"].shape, (1, 3))

    # ── Utility functions ───────────────────────────────────────────────

    def test_load_baked_timesteps(self):
        t_s, h, hu, hv = self._make_snapshot()
        persist_baked_results(self.gpkg_path, self.run_id, t_s, h, hu, hv,
                              interval_s=10.0, log_fn=self._log)
        timesteps = load_baked_timesteps(self.gpkg_path, self.run_id)
        self.assertIsNotNone(timesteps)
        np.testing.assert_array_almost_equal(timesteps, t_s)

    def test_collect_baked_runs_from_gpkg(self):
        t_s, h, hu, hv = self._make_snapshot()
        persist_baked_results(self.gpkg_path, self.run_id, t_s, h, hu, hv,
                              interval_s=10.0, log_fn=self._log)
        runs = collect_baked_runs_from_gpkg(self.gpkg_path)
        self.assertGreater(len(runs), 0)
        self.assertTrue(any(r["run_id"] == self.run_id for r in runs))
        self.assertEqual(row[0], 1, f"Expected snapshot=1, got {row[0]}")

        # Now clear it
        update_run_snapshot_tag(self.gpkg_path, "run_020", is_snapshot=False,
                                table_name_fn=self._table_name)
        conn = sqlite3.connect(self.gpkg_path)
        cur = conn.execute(
            'SELECT snapshot FROM "swe2d_mesh_results_runs" WHERE run_id = ?',
            ("run_020",))
        row = cur.fetchone()
        conn.close()
        self.assertEqual(row[0], 0, f"Expected snapshot=0 after clear, got {row[0]}")


if __name__ == "__main__":
    unittest.main()
