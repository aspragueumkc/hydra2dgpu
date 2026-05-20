import os
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from swe2d_run_log_storage import load_run_logs_from_geopackage, persist_run_log_to_geopackage  # noqa: E402


class TestSWE2DRunLogStorageMetadata(unittest.TestCase):
    def test_persist_and_load_metadata_json(self):
        fd, gpkg_path = tempfile.mkstemp(suffix=".gpkg")
        os.close(fd)
        try:
            ok = persist_run_log_to_geopackage(
                gpkg_path=gpkg_path,
                run_id="run-001",
                start_wallclock="2026-05-19 10:00:00",
                end_wallclock="2026-05-19 10:00:12",
                duration_s=12.0,
                log_text="test log",
                metadata={
                    "swe3d_geometry_gate": {
                        "strict": True,
                        "max_solid_fraction": 0.9,
                        "max_seed_leak_fallbacks": 1,
                        "violation_count": 2,
                        "violations": ["v1", "v2"],
                    }
                },
            )
            self.assertTrue(ok)

            rows = load_run_logs_from_geopackage(gpkg_path=gpkg_path)
            self.assertEqual(len(rows), 1)
            self.assertIn("metadata_json", rows[0])
            self.assertIn("metadata", rows[0])
            gate_meta = rows[0]["metadata"].get("swe3d_geometry_gate", {})
            self.assertTrue(bool(gate_meta.get("strict", False)))
            self.assertEqual(int(gate_meta.get("violation_count", -1)), 2)
        finally:
            try:
                os.remove(gpkg_path)
            except Exception:
                pass

    def test_schema_migration_adds_metadata_column(self):
        fd, gpkg_path = tempfile.mkstemp(suffix=".gpkg")
        os.close(fd)
        try:
            conn = sqlite3.connect(gpkg_path)
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    CREATE TABLE swe2d_run_logs (
                        run_id TEXT PRIMARY KEY,
                        created_utc TEXT,
                        start_wallclock TEXT,
                        end_wallclock TEXT,
                        duration_s REAL,
                        log_text TEXT
                    )
                    """
                )
                conn.commit()
            finally:
                conn.close()

            ok = persist_run_log_to_geopackage(
                gpkg_path=gpkg_path,
                run_id="run-legacy",
                start_wallclock="",
                end_wallclock="",
                duration_s=0.0,
                log_text="legacy",
                metadata={"probe": 1},
            )
            self.assertTrue(ok)

            conn = sqlite3.connect(gpkg_path)
            try:
                cur = conn.cursor()
                cur.execute("PRAGMA table_info(swe2d_run_logs)")
                cols = {str(row[1]) for row in cur.fetchall()}
                self.assertIn("metadata_json", cols)
            finally:
                conn.close()

            rows = load_run_logs_from_geopackage(gpkg_path=gpkg_path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["metadata"].get("probe"), 1)
        finally:
            try:
                os.remove(gpkg_path)
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()
