"""Tests for the refactored SQLite preview dialog and service layer."""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from swe2d.workbench.services.gpkg_service import get_table_info, get_table_contents
from swe2d.workbench.dialogs.sqlite_preview_dialog import SWE2DSQLiteTablePreviewDialog


def _make_test_gpkg(path: str) -> None:
    conn = sqlite3.connect(path)
    try:
        cur = conn.cursor()
        cur.execute(
            "CREATE TABLE test_table (id INTEGER PRIMARY KEY, name TEXT, value REAL)"
        )
        cur.execute(
            "INSERT INTO test_table (name, value) VALUES ('alpha', 1.0), "
            "('beta', 2.0), ('gamma', 3.0)"
        )
        conn.commit()
    finally:
        conn.close()


class TestGetTableInfo(unittest.TestCase):
    def test_returns_column_names(self):
        with tempfile.TemporaryDirectory() as td:
            gpkg = Path(td) / "test.gpkg"
            _make_test_gpkg(str(gpkg))
            cols = get_table_info(str(gpkg), "test_table")
            self.assertEqual(cols, ["id", "name", "value"])

    def test_returns_empty_for_missing_table(self):
        with tempfile.TemporaryDirectory() as td:
            gpkg = Path(td) / "test.gpkg"
            _make_test_gpkg(str(gpkg))
            cols = get_table_info(str(gpkg), "no_such_table")
            self.assertEqual(cols, [])


class TestGetTableContents(unittest.TestCase):
    def test_returns_all_rows(self):
        with tempfile.TemporaryDirectory() as td:
            gpkg = Path(td) / "test.gpkg"
            _make_test_gpkg(str(gpkg))
            rows = get_table_contents(str(gpkg), "test_table", limit=100)
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0], (1, "alpha", 1.0))

    def test_respects_limit(self):
        with tempfile.TemporaryDirectory() as td:
            gpkg = Path(td) / "test.gpkg"
            _make_test_gpkg(str(gpkg))
            rows = get_table_contents(str(gpkg), "test_table", limit=2)
            self.assertEqual(len(rows), 2)

    def test_returns_empty_for_missing_table(self):
        with tempfile.TemporaryDirectory() as td:
            gpkg = Path(td) / "test.gpkg"
            _make_test_gpkg(str(gpkg))
            rows = get_table_contents(str(gpkg), "no_such_table", limit=100)
            self.assertEqual(rows, [])


class TestDialogDelegatesToService(unittest.TestCase):
    def test_dialog_imports_service_functions(self):
        # Verify the dialog no longer imports sqlite3 directly
        import inspect
        src = inspect.getsource(SWE2DSQLiteTablePreviewDialog.refresh_table)
        self.assertNotIn("sqlite3.connect", src,
                         "refresh_table should not directly connect to sqlite3")

    def test_dialog_uses_get_table_info_and_contents(self):
        import inspect
        src = inspect.getsource(SWE2DSQLiteTablePreviewDialog.refresh_table)
        self.assertIn("get_table_info", src,
                      "refresh_table should call get_table_info")
        self.assertIn("get_table_contents", src,
                      "refresh_table should call get_table_contents")


if __name__ == "__main__":
    unittest.main()
