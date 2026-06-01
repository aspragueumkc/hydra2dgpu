import sqlite3
import tempfile
import unittest
from pathlib import Path

from swe2d.results.queries import discover_line_result_runs


class TestSWE2DResultsQueries(unittest.TestCase):
    def test_discover_runs_includes_mesh_only_snapshot_run(self):
        with tempfile.TemporaryDirectory() as td:
            gpkg = Path(td) / "results.gpkg"
            conn = sqlite3.connect(str(gpkg))
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    CREATE TABLE scenario_a_swe2d_mesh_results_runs (
                        run_id TEXT PRIMARY KEY,
                        created_utc TEXT,
                        interval_s REAL,
                        row_count INTEGER
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE scenario_a_swe2d_mesh_results (
                        run_id TEXT,
                        t_s REAL,
                        cell_id INTEGER,
                        h REAL,
                        hu REAL,
                        hv REAL,
                        PRIMARY KEY (run_id, t_s, cell_id)
                    )
                    """
                )
                cur.execute(
                    """
                    INSERT INTO scenario_a_swe2d_mesh_results_runs(run_id, created_utc, interval_s, row_count)
                    VALUES (?, ?, ?, ?)
                    """,
                    ("swe2d_snapshot_20260530T120000+0000", "2026-05-30T12:00:00+00:00", 60.0, 2),
                )
                cur.executemany(
                    """
                    INSERT INTO scenario_a_swe2d_mesh_results(run_id, t_s, cell_id, h, hu, hv)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        ("swe2d_snapshot_20260530T120000+0000", 0.0, 1, 0.1, 0.0, 0.0),
                        ("swe2d_snapshot_20260530T120000+0000", 60.0, 1, 0.2, 0.01, 0.0),
                    ],
                )
                conn.commit()
            finally:
                conn.close()

            runs = discover_line_result_runs(str(gpkg))
            run_ids = {str(r.get("run_id", "")) for r in runs}
            self.assertIn("swe2d_snapshot_20260530T120000+0000", run_ids)

    def test_discover_runs_merges_prefixed_line_and_mesh_without_duplicates(self):
        with tempfile.TemporaryDirectory() as td:
            gpkg = Path(td) / "results.gpkg"
            conn = sqlite3.connect(str(gpkg))
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    CREATE TABLE scenario_a_swe2d_line_results_ts (
                        run_id TEXT,
                        t_s REAL,
                        line_id INTEGER,
                        line_name TEXT,
                        depth_m REAL,
                        velocity_ms REAL,
                        wse_m REAL,
                        bed_m REAL,
                        flow_cms REAL
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE scenario_a_swe2d_line_results_runs (
                        run_id TEXT PRIMARY KEY,
                        created_utc TEXT,
                        mesh_interval_s REAL,
                        line_interval_s REAL,
                        ts_row_count INTEGER,
                        profile_row_count INTEGER
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE scenario_a_swe2d_mesh_results_runs (
                        run_id TEXT PRIMARY KEY,
                        created_utc TEXT,
                        interval_s REAL,
                        row_count INTEGER
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE scenario_a_swe2d_mesh_results (
                        run_id TEXT,
                        t_s REAL,
                        cell_id INTEGER,
                        h REAL,
                        hu REAL,
                        hv REAL,
                        PRIMARY KEY (run_id, t_s, cell_id)
                    )
                    """
                )

                run_id = "swe2d_20260530T130000+0000"
                cur.execute(
                    "INSERT INTO scenario_a_swe2d_line_results_runs(run_id, created_utc, mesh_interval_s, line_interval_s, ts_row_count, profile_row_count) VALUES (?, ?, ?, ?, ?, ?)",
                    (run_id, "2026-05-30T13:00:00+00:00", 60.0, 60.0, 1, 0),
                )
                cur.execute(
                    "INSERT INTO scenario_a_swe2d_line_results_ts(run_id, t_s, line_id, line_name, depth_m, velocity_ms, wse_m, bed_m, flow_cms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (run_id, 60.0, 1, "L1", 0.2, 0.1, 1.2, 1.0, 0.5),
                )
                cur.execute(
                    "INSERT INTO scenario_a_swe2d_mesh_results_runs(run_id, created_utc, interval_s, row_count) VALUES (?, ?, ?, ?)",
                    (run_id, "2026-05-30T13:00:00+00:00", 60.0, 1),
                )
                cur.execute(
                    "INSERT INTO scenario_a_swe2d_mesh_results(run_id, t_s, cell_id, h, hu, hv) VALUES (?, ?, ?, ?, ?, ?)",
                    (run_id, 60.0, 1, 0.2, 0.0, 0.0),
                )
                conn.commit()
            finally:
                conn.close()

            runs = discover_line_result_runs(str(gpkg))
            ids = [str(r.get("run_id", "")) for r in runs]
            self.assertEqual(ids.count("swe2d_20260530T130000+0000"), 1)


if __name__ == "__main__":
    unittest.main()
