import sqlite3
import tempfile
import unittest
from pathlib import Path

from swe2d.workbench.services.gpkg_service import (
    create_results_gpkg,
    persist_run_results,
    persist_line_results,
    get_run_metadata,
    list_runs_in_gpkg,
    delete_run_from_gpkg,
)


class TestCreateResultsGpkg(unittest.TestCase):
    def test_creates_gpkg_with_metadata_tables(self):
        with tempfile.TemporaryDirectory() as td:
            gpkg = Path(td) / "results.gpkg"
            create_results_gpkg(str(gpkg), crs_wkt='GEOGCS["WGS 84",DATUM[...]]')
            self.assertTrue(gpkg.exists())
            conn = sqlite3.connect(str(gpkg))
            try:
                cur = conn.cursor()
                cur.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name IN ('gpkg_contents', 'spatial_ref_sys', 'gpkg_geometry_columns')"
                )
                tables = {str(r[0]) for r in cur.fetchall()}
                for required in ("gpkg_contents", "spatial_ref_sys", "gpkg_geometry_columns"):
                    self.assertIn(required, tables, f"Missing OGC table: {required}")
            finally:
                conn.close()

    def test_creates_swe2d_result_tables(self):
        with tempfile.TemporaryDirectory() as td:
            gpkg = Path(td) / "results.gpkg"
            create_results_gpkg(str(gpkg), crs_wkt='GEOGCS["WGS 84",DATUM[...]]')
            conn = sqlite3.connect(str(gpkg))
            try:
                cur = conn.cursor()
                cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
                all_tables = {str(r[0]) for r in cur.fetchall()}
                for expected in (
                    "swe2d_mesh_results",
                    "swe2d_mesh_results_runs",
                    "swe2d_line_results_ts",
                    "swe2d_line_results_runs",
                    "swe2d_line_results_profile",
                ):
                    self.assertIn(expected, all_tables, f"Missing SWE2D table: {expected}")
            finally:
                conn.close()

    def test_idempotent_repeat_call(self):
        with tempfile.TemporaryDirectory() as td:
            gpkg = Path(td) / "results.gpkg"
            create_results_gpkg(str(gpkg), 'GEOGCS["WGS 84",DATUM[...]]')
            create_results_gpkg(str(gpkg), 'GEOGCS["WGS 84",DATUM[...]]')
            create_results_gpkg(str(gpkg), 'GEOGCS["WGS 84",DATUM[...]]')
            conn = sqlite3.connect(str(gpkg))
            try:
                cur = conn.cursor()
                cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
                self.assertGreater(len(cur.fetchall()), 3)
            finally:
                conn.close()

    def test_rejects_empty_path(self):
        with self.assertRaises(ValueError):
            create_results_gpkg("", 'GEOGCS["WGS 84",DATUM[...]]')


class TestPersistRunResults(unittest.TestCase):
    def test_persists_mesh_data_and_returns_run_id(self):
        with tempfile.TemporaryDirectory() as td:
            gpkg = Path(td) / "results.gpkg"
            create_results_gpkg(str(gpkg), 'GEOGCS["WGS 84",DATUM[...]]')

            import numpy as np
            n_cells = 10
            h = np.ones(n_cells, dtype=np.float64) * 0.5
            hu = np.zeros(n_cells, dtype=np.float64)
            hv = np.zeros(n_cells, dtype=np.float64)
            run_id = persist_run_results(str(gpkg), "test_run_001", h, hu, hv, interval_s=60.0)
            self.assertEqual(run_id, "test_run_001")

            conn = sqlite3.connect(str(gpkg))
            try:
                cur = conn.cursor()
                cur.execute(
                    "SELECT COUNT(*) FROM swe2d_mesh_results WHERE run_id=?",
                    (run_id,),
                )
                self.assertEqual(cur.fetchone()[0], n_cells)
                cur.execute(
                    "SELECT COUNT(*) FROM swe2d_mesh_results_runs WHERE run_id=?",
                    (run_id,),
                )
                self.assertEqual(cur.fetchone()[0], 1)
            finally:
                conn.close()

    def test_auto_generates_run_id_when_empty(self):
        with tempfile.TemporaryDirectory() as td:
            gpkg = Path(td) / "results.gpkg"
            create_results_gpkg(str(gpkg), 'GEOGCS["WGS 84",DATUM[...]]')

            import numpy as np
            h = np.ones(3, dtype=np.float64)
            hu = np.zeros(3, dtype=np.float64)
            hv = np.zeros(3, dtype=np.float64)
            run_id = persist_run_results(str(gpkg), "", h, hu, hv, interval_s=30.0)
            self.assertTrue(run_id.startswith("swe2d_"))

    def test_replaces_existing_run_data(self):
        with tempfile.TemporaryDirectory() as td:
            gpkg = Path(td) / "results.gpkg"
            create_results_gpkg(str(gpkg), 'GEOGCS["WGS 84",DATUM[...]]')

            import numpy as np
            h = np.ones(5, dtype=np.float64) * 1.0
            hu = np.zeros(5, dtype=np.float64)
            hv = np.zeros(5, dtype=np.float64)
            persist_run_results(str(gpkg), "run_replace", h, hu, hv, interval_s=10.0)

            h2 = np.ones(5, dtype=np.float64) * 2.0
            persist_run_results(str(gpkg), "run_replace", h2, hu, hv, interval_s=20.0)

            conn = sqlite3.connect(str(gpkg))
            try:
                cur = conn.cursor()
                cur.execute(
                    "SELECT COUNT(*) FROM swe2d_mesh_results WHERE run_id=?",
                    ("run_replace",),
                )
                self.assertEqual(cur.fetchone()[0], 5)
                cur.execute(
                    "SELECT h FROM swe2d_mesh_results WHERE run_id=? AND t_s=0.0 AND cell_id=0",
                    ("run_replace",),
                )
                self.assertAlmostEqual(cur.fetchone()[0], 2.0)
            finally:
                conn.close()


class TestPersistLineResults(unittest.TestCase):
    def test_persists_line_ts_and_runs_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            gpkg = Path(td) / "results.gpkg"
            create_results_gpkg(str(gpkg), 'GEOGCS["WGS 84",DATUM[...]]')

            line_data = {
                "ts_rows": [
                    {"t_s": 0.0, "line_id": 1, "line_name": "L1",
                     "depth_m": 0.5, "velocity_ms": 0.1, "wse_m": 1.5,
                     "bed_m": 1.0, "flow_cms": 0.2, "wet_frac": 1.0, "fr": 0.05},
                    {"t_s": 60.0, "line_id": 1, "line_name": "L1",
                     "depth_m": 0.6, "velocity_ms": 0.2, "wse_m": 1.6,
                     "bed_m": 1.0, "flow_cms": 0.3, "wet_frac": 1.0, "fr": 0.08},
                ],
                "profile_rows": [
                    {"t_s": 60.0, "line_id": 1, "line_name": "L1",
                     "station_m": 0.0, "depth_m": 0.6, "velocity_ms": 0.2,
                     "wse_m": 1.6, "bed_m": 1.0, "flow_qn": 0.3, "wet": 1, "fr": 0.08},
                    {"t_s": 60.0, "line_id": 1, "line_name": "L1",
                     "station_m": 10.0, "depth_m": 0.5, "velocity_ms": 0.15,
                     "wse_m": 1.5, "bed_m": 1.0, "flow_qn": 0.2, "wet": 1, "fr": 0.06},
                ],
                "mesh_interval_s": 60.0,
                "line_interval_s": 60.0,
            }

            persist_line_results(str(gpkg), "line_run_001", line_data)

            conn = sqlite3.connect(str(gpkg))
            try:
                cur = conn.cursor()
                cur.execute(
                    "SELECT COUNT(*) FROM swe2d_line_results_ts WHERE run_id=?",
                    ("line_run_001",),
                )
                self.assertEqual(cur.fetchone()[0], 2)
                cur.execute(
                    "SELECT COUNT(*) FROM swe2d_line_results_profile WHERE run_id=?",
                    ("line_run_001",),
                )
                self.assertEqual(cur.fetchone()[0], 2)
                cur.execute(
                    "SELECT COUNT(*) FROM swe2d_line_results_runs WHERE run_id=?",
                    ("line_run_001",),
                )
                self.assertEqual(cur.fetchone()[0], 1)
            finally:
                conn.close()

    def test_persist_line_without_profile(self):
        with tempfile.TemporaryDirectory() as td:
            gpkg = Path(td) / "results.gpkg"
            create_results_gpkg(str(gpkg), 'GEOGCS["WGS 84",DATUM[...]]')

            line_data = {
                "ts_rows": [
                    {"t_s": 0.0, "line_id": 1, "line_name": "L1",
                     "depth_m": 0.5, "velocity_ms": 0.1, "wse_m": 1.5,
                     "bed_m": 1.0, "flow_cms": 0.2, "wet_frac": 1.0, "fr": 0.05},
                ],
                "mesh_interval_s": 60.0,
                "line_interval_s": 60.0,
            }

            persist_line_results(str(gpkg), "line_run_no_profile", line_data)

            conn = sqlite3.connect(str(gpkg))
            try:
                cur = conn.cursor()
                cur.execute(
                    "SELECT COUNT(*) FROM swe2d_line_results_ts WHERE run_id=?",
                    ("line_run_no_profile",),
                )
                self.assertEqual(cur.fetchone()[0], 1)
                cur.execute(
                    "SELECT COUNT(*) FROM swe2d_line_results_profile WHERE run_id=?",
                    ("line_run_no_profile",),
                )
                self.assertEqual(cur.fetchone()[0], 0)
            finally:
                conn.close()


class TestGetRunMetadata(unittest.TestCase):
    def test_returns_metadata_for_existing_run(self):
        with tempfile.TemporaryDirectory() as td:
            gpkg = Path(td) / "results.gpkg"
            create_results_gpkg(str(gpkg), 'GEOGCS["WGS 84",DATUM[...]]')

            import numpy as np
            persist_run_results(str(gpkg), "meta_run", np.ones(3), np.zeros(3), np.zeros(3), interval_s=30.0)

            meta = get_run_metadata(str(gpkg), "meta_run")
            self.assertIsInstance(meta, dict)
            self.assertEqual(meta.get("run_id"), "meta_run")
            self.assertIn("created_utc", meta)
            self.assertIn("interval_s", meta)
            self.assertIn("row_count", meta)

    def test_returns_empty_dict_for_missing_run(self):
        with tempfile.TemporaryDirectory() as td:
            gpkg = Path(td) / "results.gpkg"
            create_results_gpkg(str(gpkg), 'GEOGCS["WGS 84",DATUM[...]]')
            meta = get_run_metadata(str(gpkg), "nonexistent_run")
            self.assertEqual(meta, {})


class TestListRunsInGpkg(unittest.TestCase):
    def test_lists_all_runs(self):
        with tempfile.TemporaryDirectory() as td:
            gpkg = Path(td) / "results.gpkg"
            create_results_gpkg(str(gpkg), 'GEOGCS["WGS 84",DATUM[...]]')

            import numpy as np
            persist_run_results(str(gpkg), "run_a", np.ones(2), np.zeros(2), np.zeros(2), interval_s=10.0)
            persist_run_results(str(gpkg), "run_b", np.ones(2), np.zeros(2), np.zeros(2), interval_s=20.0)

            runs = list_runs_in_gpkg(str(gpkg))
            self.assertEqual(len(runs), 2)
            run_ids = {r["run_id"] for r in runs}
            self.assertIn("run_a", run_ids)
            self.assertIn("run_b", run_ids)

    def test_returns_empty_when_no_runs(self):
        with tempfile.TemporaryDirectory() as td:
            gpkg = Path(td) / "results.gpkg"
            create_results_gpkg(str(gpkg), 'GEOGCS["WGS 84",DATUM[...]]')
            runs = list_runs_in_gpkg(str(gpkg))
            self.assertEqual(runs, [])


class TestDeleteRunFromGpkg(unittest.TestCase):
    def test_deletes_run_and_all_data(self):
        with tempfile.TemporaryDirectory() as td:
            gpkg = Path(td) / "results.gpkg"
            create_results_gpkg(str(gpkg), 'GEOGCS["WGS 84",DATUM[...]]')

            import numpy as np
            persist_run_results(str(gpkg), "del_me", np.ones(3), np.zeros(3), np.zeros(3), interval_s=10.0)
            persist_run_results(str(gpkg), "keep_me", np.ones(3), np.zeros(3), np.zeros(3), interval_s=20.0)

            delete_run_from_gpkg(str(gpkg), "del_me")

            runs = list_runs_in_gpkg(str(gpkg))
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0]["run_id"], "keep_me")

            conn = sqlite3.connect(str(gpkg))
            try:
                cur = conn.cursor()
                cur.execute(
                    "SELECT COUNT(*) FROM swe2d_mesh_results WHERE run_id=?",
                    ("del_me",),
                )
                self.assertEqual(cur.fetchone()[0], 0)
                cur.execute(
                    "SELECT COUNT(*) FROM swe2d_mesh_results WHERE run_id=?",
                    ("keep_me",),
                )
                self.assertEqual(cur.fetchone()[0], 3)
            finally:
                conn.close()

    def test_delete_nonexistent_run_does_not_error(self):
        with tempfile.TemporaryDirectory() as td:
            gpkg = Path(td) / "results.gpkg"
            create_results_gpkg(str(gpkg), 'GEOGCS["WGS 84",DATUM[...]]')
            delete_run_from_gpkg(str(gpkg), "does_not_exist")


if __name__ == "__main__":
    unittest.main()
