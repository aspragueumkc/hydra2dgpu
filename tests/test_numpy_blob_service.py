"""tests/test_numpy_blob_service.py"""
from __future__ import annotations

import io
import os
import sqlite3
import struct
import tempfile
import unittest

import numpy as np

from swe2d.workbench.services.numpy_blob_service import (
    build_where_clause,
    combine_1d_arrays,
    deserialize_blob_to_array,
    discover_plottable_columns,
    export_table_to_csv,
    get_filtered_rows,
)


def _make_gpkg_with_baked_results(path: str):
    """Create a test GPKG with a swe2d_baked_results row."""
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE swe2d_baked_results (
            run_id TEXT PRIMARY KEY,
            mesh_name TEXT NOT NULL,
            n_cells INTEGER NOT NULL,
            n_timesteps INTEGER NOT NULL,
            created_utc TEXT NOT NULL,
            times_blob BLOB NOT NULL,
            h_blob BLOB NOT NULL,
            hu_blob BLOB NOT NULL,
            hv_blob BLOB NOT NULL)
    """)
    n_cells = 100
    n_timesteps = 50
    times = np.linspace(0, 10, n_timesteps, dtype=np.float64)
    h = np.random.rand(n_timesteps, n_cells).astype(np.float64)
    hu = np.random.rand(n_timesteps, n_cells).astype(np.float64)
    hv = np.random.rand(n_timesteps, n_cells).astype(np.float64)
    conn.execute(
        "INSERT INTO swe2d_baked_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "run_001", "mesh_01", n_cells, n_timesteps, "2024-01-01T00:00:00",
            times.tobytes(), h.tobytes(), hu.tobytes(), hv.tobytes(),
        ),
    )
    conn.commit()
    conn.close()


class TestDeserializeBlobToArray(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False)
        self.path = self._tmp.name
        self._tmp.close()
        _make_gpkg_with_baked_results(self.path)

    def tearDown(self):
        os.unlink(self.path)

    def test_deserialize_times_blob_1d(self):
        arr = deserialize_blob_to_array(self.path, "swe2d_baked_results", "times_blob", run_id="run_001")
        self.assertIsNotNone(arr)
        self.assertEqual(arr.dtype, np.float64)
        self.assertEqual(arr.ndim, 1)
        self.assertEqual(arr.shape[0], 50)

    def test_deserialize_h_blob_2d(self):
        arr = deserialize_blob_to_array(self.path, "swe2d_baked_results", "h_blob", run_id="run_001")
        self.assertIsNotNone(arr)
        self.assertEqual(arr.dtype, np.float64)
        self.assertEqual(arr.ndim, 2)
        self.assertEqual(arr.shape, (50, 100))

    def test_deserialize_missing_column_returns_none(self):
        arr = deserialize_blob_to_array(self.path, "swe2d_baked_results", "nonexistent", run_id="run_001")
        self.assertIsNone(arr)

    def test_deserialize_nonexistent_table_returns_none(self):
        arr = deserialize_blob_to_array(self.path, "nonexistent", "times_blob", run_id="run_001")
        self.assertIsNone(arr)


class TestDiscoverPlottableColumns(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False)
        self.path = self._tmp.name
        self._tmp.close()
        _make_gpkg_with_baked_results(self.path)

    def tearDown(self):
        os.unlink(self.path)

    def test_discover_returns_metadata_and_blob_cols(self):
        cols = discover_plottable_columns(self.path, "swe2d_baked_results")
        col_names = [c["name"] for c in cols]
        self.assertIn("n_cells", col_names)
        self.assertIn("n_timesteps", col_names)
        self.assertIn("times_blob", col_names)
        self.assertIn("h_blob", col_names)

    def test_discover_blob_has_shape_metadata(self):
        cols = discover_plottable_columns(self.path, "swe2d_baked_results")
        h_info = next(c for c in cols if c["name"] == "h_blob")
        self.assertEqual(h_info["ndim"], 2)
        self.assertEqual(h_info["shape"], (50, 100))

    def test_discover_unknown_table_returns_empty(self):
        cols = discover_plottable_columns(self.path, "nonexistent")
        self.assertEqual(cols, [])


class TestBuildWhereClause(unittest.TestCase):
    def test_numeric_equals(self):
        sql, params = build_where_clause("n_cells", "=", "5000")
        self.assertIn("?", sql)
        self.assertEqual(params, [5000])

    def test_string_like(self):
        sql, params = build_where_clause("run_id", "LIKE", "run%")
        self.assertIn("?", sql)
        self.assertEqual(params, ["run%"])

    def test_is_null(self):
        sql, params = build_where_clause("mesh_name", "IS NULL", "")
        self.assertIn("IS NULL", sql)
        self.assertEqual(params, [])

    def test_is_not_null(self):
        sql, params = build_where_clause("mesh_name", "IS NOT NULL", "")
        self.assertIn("IS NOT NULL", sql)
        self.assertEqual(params, [])

    def test_in_clause(self):
        sql, params = build_where_clause("run_id", "IN", "a,b,c")
        self.assertIn("IN (?,?,?)", sql)
        self.assertEqual(params, ["a", "b", "c"])

    def test_injection_attempt(self):
        sql, params = build_where_clause("run_id", "=", "' OR 1=1 --")
        self.assertIn("?", sql)
        self.assertEqual(params, ["' OR 1=1 --"])

    def test_greater_than(self):
        sql, params = build_where_clause("n_cells", ">", "100")
        self.assertIn("?", sql)
        self.assertEqual(params, [100])


class TestExportTableToCsv(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False)
        self.path = self._tmp.name
        self._tmp.close()
        _make_gpkg_with_baked_results(self.path)

    def tearDown(self):
        os.unlink(self.path)

    def test_export_csv_creates_file(self):
        csv_path = self.path + ".csv"
        export_table_to_csv(self.path, "swe2d_baked_results", csv_path, columns=["run_id", "n_cells", "n_timesteps"])
        self.assertTrue(os.path.exists(csv_path))
        with open(csv_path) as f:
            lines = f.readlines()
        self.assertGreaterEqual(len(lines), 2)  # header + 1 data row

    def test_export_csv_with_where_clause(self):
        csv_path = self.path + "_filtered.csv"
        export_table_to_csv(
            self.path, "swe2d_baked_results", csv_path,
            columns=["run_id", "n_cells"],
            column="n_cells", op=">", value="0",
        )
        self.assertTrue(os.path.exists(csv_path))

    def test_export_csv_unknown_table_raises(self):
        with self.assertRaises(ValueError):
            export_table_to_csv(self.path, "nonexistent", "/tmp/out.csv")

    def test_export_csv_writes_correct_header(self):
        csv_path = self.path + "_header.csv"
        export_table_to_csv(self.path, "swe2d_baked_results", csv_path, columns=["run_id", "n_cells"])
        with open(csv_path) as f:
            header = f.readline().strip()
        self.assertEqual(header, "run_id,n_cells")


class TestGetFilteredRows(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False)
        self.path = self._tmp.name
        self._tmp.close()
        _make_gpkg_with_baked_results(self.path)

    def tearDown(self):
        os.unlink(self.path)

    def test_get_filtered_rows_returns_dicts(self):
        rows = get_filtered_rows(self.path, "swe2d_baked_results", limit=10)
        self.assertGreaterEqual(len(rows), 1)
        self.assertIn("run_id", rows[0])

    def test_get_filtered_rows_with_where(self):
        rows = get_filtered_rows(
            self.path, "swe2d_baked_results",
            column="n_cells", op="=", value="100",
            limit=10,
        )
        self.assertEqual(len(rows), 1)

    def test_get_filtered_rows_empty_result(self):
        rows = get_filtered_rows(
            self.path, "swe2d_baked_results",
            column="n_cells", op=">", value="999999",
            limit=10,
        )
        self.assertEqual(len(rows), 0)


class TestCombine1DArrays(unittest.TestCase):
    def test_combines_equal_length_arrays(self):
        times = np.array([0.0, 1.0, 2.0])
        values = np.array([10.0, 20.0, 30.0])
        labels, combined = combine_1d_arrays({"times": times, "values": values})
        self.assertEqual(labels, ["times", "values"])
        self.assertEqual(combined.shape, (3, 2))
        self.assertEqual(combined[0, 0], 0.0)
        self.assertEqual(combined[2, 1], 30.0)

    def test_returns_first_for_mismatched_lengths(self):
        a = np.array([0.0, 1.0, 2.0])
        b = np.array([10.0, 20.0])
        labels, combined = combine_1d_arrays({"a": a, "b": b})
        self.assertEqual(labels, ["a"])
        self.assertEqual(combined.shape, (3,))

    def test_returns_first_for_mixed_dim(self):
        a = np.array([0.0, 1.0])
        b = np.zeros((2, 3))
        labels, combined = combine_1d_arrays({"a": a, "b": b})
        self.assertEqual(labels, ["a"])

    def test_empty_input_returns_empty(self):
        labels, combined = combine_1d_arrays({})
        self.assertEqual(labels, [])
        self.assertIsNone(combined)

    def test_skips_none_values(self):
        times = np.array([0.0, 1.0])
        labels, combined = combine_1d_arrays({"times": times, "missing": None})
        self.assertEqual(labels, ["times"])
        self.assertEqual(combined.shape, (2, 1))
