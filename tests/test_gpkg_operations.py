"""Tests for swe2d/workbench/gpkg_operations.py."""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from swe2d.workbench.services.gpkg_operations_service import (
    list_tables,
    get_table_row_count,
    rename_table,
    drop_table,
    get_table_info,
    get_table_contents,
    delete_run,
)


def _make_gpkg(path: str) -> sqlite3.Connection:
    """Create a minimal GPKG with a few tables for testing."""
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE IF NOT EXISTS gpkg_contents (table_name TEXT PRIMARY KEY, data_type TEXT)"
    )
    cur.execute("CREATE TABLE swe2d_run_logs (run_id TEXT, t REAL)")
    cur.execute("CREATE TABLE swe2d_mesh_results_run_001 (cell_id INT, h REAL)")
    cur.execute("CREATE TABLE swe2d_line_results_run_001 (line_id INT, val REAL)")
    cur.execute("CREATE TABLE spatial_ref_sys (srs_id INT)")
    cur.execute("CREATE TABLE rtree_mytable_geom (id INT)")
    cur.execute("CREATE TABLE swe2d_conservation_run_001 (mass REAL)")

    cur.execute("INSERT INTO swe2d_run_logs (run_id, t) VALUES ('run_001', 0.0)")
    cur.execute("INSERT INTO swe2d_run_logs (run_id, t) VALUES ('run_001', 1.0)")
    cur.execute("INSERT INTO swe2d_run_logs (run_id, t) VALUES ('run_002', 0.0)")
    cur.execute("INSERT INTO swe2d_mesh_results_run_001 (cell_id, h) VALUES (0, 0.5)")
    cur.execute("INSERT INTO swe2d_line_results_run_001 (line_id, val) VALUES (1, 1.0)")
    conn.commit()
    return conn


class TestListTables(unittest.TestCase):
    def test_returns_user_tables_excluding_system_tables(self):
        with tempfile.TemporaryDirectory() as td:
            gpkg = Path(td) / "test.gpkg"
            conn = _make_gpkg(str(gpkg))
            conn.close()

            tables = list_tables(str(gpkg))
            self.assertIn("swe2d_run_logs", tables)
            self.assertIn("swe2d_mesh_results_run_001", tables)
            self.assertIn("swe2d_line_results_run_001", tables)
            self.assertIn("swe2d_conservation_run_001", tables)
            self.assertNotIn("gpkg_contents", tables)
            self.assertNotIn("rtree_mytable_geom", tables)

    def test_returns_empty_for_empty_gpkg(self):
        with tempfile.TemporaryDirectory() as td:
            gpkg = Path(td) / "empty.gpkg"
            conn = sqlite3.connect(str(gpkg))
            conn.close()
            tables = list_tables(str(gpkg))
            self.assertEqual(tables, [])

    def test_returns_empty_for_nonexistent_file(self):
        tables = list_tables("/nonexistent/path.gpkg")
        self.assertEqual(tables, [])


class TestGetTableRowCount(unittest.TestCase):
    def test_returns_row_count(self):
        with tempfile.TemporaryDirectory() as td:
            gpkg = Path(td) / "test.gpkg"
            conn = _make_gpkg(str(gpkg))
            conn.close()

            count = get_table_row_count(str(gpkg), "swe2d_run_logs")
            self.assertEqual(count, 3)

    def test_returns_zero_for_empty_table(self):
        with tempfile.TemporaryDirectory() as td:
            gpkg = Path(td) / "test.gpkg"
            conn = _make_gpkg(str(gpkg))
            conn.close()

            count = get_table_row_count(str(gpkg), "spatial_ref_sys")
            self.assertEqual(count, 0)

    def test_returns_zero_for_nonexistent_table(self):
        with tempfile.TemporaryDirectory() as td:
            gpkg = Path(td) / "test.gpkg"
            conn = _make_gpkg(str(gpkg))
            conn.close()

            count = get_table_row_count(str(gpkg), "nonexistent")
            self.assertEqual(count, 0)


class TestRenameTable(unittest.TestCase):
    def test_renames_table(self):
        with tempfile.TemporaryDirectory() as td:
            gpkg = Path(td) / "test.gpkg"
            conn = _make_gpkg(str(gpkg))
            conn.close()

            rename_table(str(gpkg), "swe2d_run_logs", "swe2d_run_logs_renamed")

            conn2 = sqlite3.connect(str(gpkg))
            cur = conn2.cursor()
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                ("swe2d_run_logs_renamed",),
            )
            self.assertIsNotNone(cur.fetchone())
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                ("swe2d_run_logs",),
            )
            self.assertIsNone(cur.fetchone())
            conn2.close()

    def test_raises_on_nonexistent_table(self):
        with tempfile.TemporaryDirectory() as td:
            gpkg = Path(td) / "test.gpkg"
            conn = _make_gpkg(str(gpkg))
            conn.close()

            with self.assertRaises(RuntimeError):
                rename_table(str(gpkg), "nonexistent", "new_name")

    def test_raises_on_duplicate_new_name(self):
        with tempfile.TemporaryDirectory() as td:
            gpkg = Path(td) / "test.gpkg"
            conn = _make_gpkg(str(gpkg))
            conn.close()

            with self.assertRaises(RuntimeError):
                rename_table(str(gpkg), "swe2d_run_logs", "swe2d_mesh_results_run_001")


class TestDropTable(unittest.TestCase):
    def test_drops_table(self):
        with tempfile.TemporaryDirectory() as td:
            gpkg = Path(td) / "test.gpkg"
            conn = _make_gpkg(str(gpkg))
            conn.close()

            drop_table(str(gpkg), "swe2d_run_logs")

            conn2 = sqlite3.connect(str(gpkg))
            cur = conn2.cursor()
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                ("swe2d_run_logs",),
            )
            self.assertIsNone(cur.fetchone())
            conn2.close()

    def test_drop_nonexistent_table_is_noop(self):
        with tempfile.TemporaryDirectory() as td:
            gpkg = Path(td) / "test.gpkg"
            conn = _make_gpkg(str(gpkg))
            conn.close()

            drop_table(str(gpkg), "nonexistent")


class TestGetTableInfo(unittest.TestCase):
    def test_returns_column_info(self):
        with tempfile.TemporaryDirectory() as td:
            gpkg = Path(td) / "test.gpkg"
            conn = _make_gpkg(str(gpkg))
            conn.close()

            info = get_table_info(str(gpkg), "swe2d_run_logs")
            self.assertIsInstance(info, list)
            self.assertGreater(len(info), 0)
            col_names = [c["name"] for c in info]
            self.assertIn("run_id", col_names)
            self.assertIn("t", col_names)

    def test_returns_empty_for_nonexistent_table(self):
        with tempfile.TemporaryDirectory() as td:
            gpkg = Path(td) / "test.gpkg"
            conn = _make_gpkg(str(gpkg))
            conn.close()

            info = get_table_info(str(gpkg), "nonexistent")
            self.assertEqual(info, [])


class TestGetTableContents(unittest.TestCase):
    def test_returns_rows_as_dicts(self):
        with tempfile.TemporaryDirectory() as td:
            gpkg = Path(td) / "test.gpkg"
            conn = _make_gpkg(str(gpkg))
            conn.close()

            rows = get_table_contents(str(gpkg), "swe2d_run_logs", limit=10)
            self.assertIsInstance(rows, list)
            self.assertEqual(len(rows), 3)
            self.assertIn("run_id", rows[0])
            self.assertIn("t", rows[0])

    def test_respects_limit(self):
        with tempfile.TemporaryDirectory() as td:
            gpkg = Path(td) / "test.gpkg"
            conn = _make_gpkg(str(gpkg))
            conn.close()

            rows = get_table_contents(str(gpkg), "swe2d_run_logs", limit=1)
            self.assertEqual(len(rows), 1)

    def test_returns_empty_for_nonexistent_table(self):
        with tempfile.TemporaryDirectory() as td:
            gpkg = Path(td) / "test.gpkg"
            conn = _make_gpkg(str(gpkg))
            conn.close()

            rows = get_table_contents(str(gpkg), "nonexistent")
            self.assertEqual(rows, [])


class TestDeleteRun(unittest.TestCase):
    def test_deletes_tables_matching_run_id(self):
        with tempfile.TemporaryDirectory() as td:
            gpkg = Path(td) / "test.gpkg"
            conn = _make_gpkg(str(gpkg))
            conn.close()

            delete_run(str(gpkg), "run_001")

            conn2 = sqlite3.connect(str(gpkg))
            cur = conn2.cursor()

            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%run_001%'"
            )
            remaining = [str(r[0]) for r in cur.fetchall()]
            # swe2d_run_logs, swe2d_mesh_results_run_001,
            # swe2d_line_results_run_001, swe2d_conservation_run_001
            # should all be gone
            for tbl in remaining:
                self.assertFalse(
                    tbl.endswith("_run_001") or tbl == "swe2d_run_logs",
                    f"Table {tbl} should have been deleted",
                )

            # run_log entry for run_001 should be gone too
            cur.execute(
                "SELECT COUNT(*) FROM swe2d_run_logs WHERE run_id=?",
                ("run_001",),
            )
            self.assertEqual(cur.fetchone()[0], 0)

            # run_002 should be unaffected
            cur.execute(
                "SELECT COUNT(*) FROM swe2d_run_logs WHERE run_id=?",
                ("run_002",),
            )
            self.assertEqual(cur.fetchone()[0], 1)

            conn2.close()

    def test_delete_nonexistent_run_is_noop(self):
        with tempfile.TemporaryDirectory() as td:
            gpkg = Path(td) / "test.gpkg"
            conn = _make_gpkg(str(gpkg))
            conn.close()

            delete_run(str(gpkg), "nonexistent_run")


if __name__ == "__main__":
    unittest.main()
