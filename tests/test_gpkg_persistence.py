"""Test that GPKG persistence actually writes data when called.

Exercises persist_mesh_results_to_geopackage and
persist_coupling_results_to_geopackage directly with and without the
accumulate flag.  No QGIS environment needed — pure SQLite verify.
"""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from swe2d.services.gpkg_persistence_service import (
    persist_mesh_results_to_geopackage,
    persist_coupling_results_to_geopackage,
    update_run_snapshot_tag,
)


class TestGpkgPersistence(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False)
        self.gpkg_path = self.tmp.name
        self.tmp.close()

    def tearDown(self):
        if os.path.exists(self.gpkg_path):
            os.unlink(self.gpkg_path)

    def _make_mesh_rows(self, n_ts=2, n_cells=3):
        rows = []
        for t in range(n_ts):
            for c in range(n_cells):
                rows.append({
                    "t_s": float(t),
                    "cell_id": c,
                    "h": 1.0 + float(t) * 0.1,
                    "hu": 0.0,
                    "hv": 0.0,
                })
        return rows

    def _make_coupling_rows(self, n_ts=2):
        rows = []
        for t in range(n_ts):
            rows.append({"t_s": float(t), "component": "structure",
                         "metric": "flow", "object_id": "s1",
                         "object_name": "culvert_1", "value": float(t) * 10.0})
            rows.append({"t_s": float(t), "component": "structure",
                         "metric": "flow", "object_id": "s2",
                         "object_name": "weir_1", "value": float(t) * 5.0})
        return rows

    def _log(self, msg):
        pass

    def _table_name(self, base):
        return base

    def test_mesh_results_persisted(self):
        """Verify mesh rows end up in GPKG after persist."""
        rows = self._make_mesh_rows(n_ts=2, n_cells=3)
        persist_mesh_results_to_geopackage(
            self.gpkg_path, "run_001", rows, interval_s=60.0,
            log_fn=self._log, results_table_name_fn=self._table_name,
        )
        # Read back
        import sqlite3
        conn = sqlite3.connect(self.gpkg_path)
        cur = conn.execute('SELECT COUNT(*) FROM "swe2d_mesh_results" WHERE run_id = ?', ("run_001",))
        count = cur.fetchone()[0]
        conn.close()
        self.assertEqual(count, 6, f"Expected 6 mesh rows (2 timesteps × 3 cells), got {count}")

    def test_mesh_accumulate_appends_not_replaces(self):
        """With accumulate=True, second call adds rows without deleting first batch."""
        rows_1 = self._make_mesh_rows(n_ts=1, n_cells=3)  # t=0 only
        # Make a second batch at t=1 (all rows)
        rows_2 = []
        for c in range(3):
            rows_2.append({"t_s": 1.0, "cell_id": c, "h": 1.1, "hu": 0.0, "hv": 0.0})

        persist_mesh_results_to_geopackage(
            self.gpkg_path, "run_002", rows_1, interval_s=60.0,
            log_fn=self._log, results_table_name_fn=self._table_name,
            accumulate=False,
        )
        persist_mesh_results_to_geopackage(
            self.gpkg_path, "run_002", rows_2, interval_s=60.0,
            log_fn=self._log, results_table_name_fn=self._table_name,
            accumulate=True,
        )
        import sqlite3
        conn = sqlite3.connect(self.gpkg_path)
        cur = conn.execute('SELECT COUNT(*) FROM "swe2d_mesh_results" WHERE run_id = ?', ("run_002",))
        count = cur.fetchone()[0]
        conn.close()
        self.assertEqual(count, 6, f"Expected 6 rows (t=0 + t=1), got {count}")

    def test_mesh_no_accumulate_replaces(self):
        """With accumulate=False, second call replaces all rows."""
        rows_1 = self._make_mesh_rows(n_ts=1, n_cells=3)
        rows_2 = self._make_mesh_rows(n_ts=2, n_cells=3)

        persist_mesh_results_to_geopackage(
            self.gpkg_path, "run_003", rows_1, interval_s=60.0,
            log_fn=self._log, results_table_name_fn=self._table_name,
            accumulate=False,
        )
        persist_mesh_results_to_geopackage(
            self.gpkg_path, "run_003", rows_2, interval_s=60.0,
            log_fn=self._log, results_table_name_fn=self._table_name,
            accumulate=False,
        )
        import sqlite3
        conn = sqlite3.connect(self.gpkg_path)
        cur = conn.execute('SELECT COUNT(*) FROM "swe2d_mesh_results" WHERE run_id = ?', ("run_003",))
        count = cur.fetchone()[0]
        conn.close()
        self.assertEqual(count, 6, f"Expected 6 rows (replaced), got {count}")

    def test_coupling_results_persisted(self):
        """Verify coupling rows end up in GPKG after persist."""
        rows = self._make_coupling_rows(n_ts=2)
        persist_coupling_results_to_geopackage(
            self.gpkg_path, "run_010", rows, interval_s=60.0,
            results_table_name_fn=self._table_name, log_fn=self._log,
        )
        import sqlite3
        conn = sqlite3.connect(self.gpkg_path)
        cur = conn.execute('SELECT COUNT(*) FROM "swe2d_coupling_results" WHERE run_id = ?', ("run_010",))
        count = cur.fetchone()[0]
        conn.close()
        self.assertEqual(count, 4, f"Expected 4 coupling rows (2 ts × 2 structures), got {count}")

    def test_coupling_accumulate_appends(self):
        """With accumulate=True, coupling rows accumulate."""
        rows_1 = self._make_coupling_rows(n_ts=1)  # t=0
        rows_2 = self._make_coupling_rows(n_ts=1)  # t=0
        rows_2[0]["t_s"] = 1.0  # shift to t=1
        rows_2[1]["t_s"] = 1.0

        persist_coupling_results_to_geopackage(
            self.gpkg_path, "run_011", rows_1, interval_s=60.0,
            results_table_name_fn=self._table_name, log_fn=self._log,
            accumulate=False,
        )
        persist_coupling_results_to_geopackage(
            self.gpkg_path, "run_011", rows_2, interval_s=60.0,
            results_table_name_fn=self._table_name, log_fn=self._log,
            accumulate=True,
        )
        import sqlite3
        conn = sqlite3.connect(self.gpkg_path)
        cur = conn.execute('SELECT COUNT(*) FROM "swe2d_coupling_results" WHERE run_id = ?', ("run_011",))
        count = cur.fetchone()[0]
        conn.close()
        self.assertEqual(count, 4, f"Expected 4 coupling rows (accumulated), got {count}")

    def test_update_snapshot_tag(self):
        """Verify update_run_snapshot_tag sets snapshot column."""
        rows = self._make_mesh_rows(n_ts=1, n_cells=1)
        persist_mesh_results_to_geopackage(
            self.gpkg_path, "run_020", rows, interval_s=60.0,
            log_fn=self._log, results_table_name_fn=self._table_name,
        )
        update_run_snapshot_tag(self.gpkg_path, "run_020", is_snapshot=True,
                                table_name_fn=self._table_name)
        import sqlite3
        conn = sqlite3.connect(self.gpkg_path)
        cur = conn.execute(
            'SELECT snapshot FROM "swe2d_mesh_results_runs" WHERE run_id = ?',
            ("run_020",))
        row = cur.fetchone()
        conn.close()
        self.assertIsNotNone(row, "run_020 should exist in runs table")
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
