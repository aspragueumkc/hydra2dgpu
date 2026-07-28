---
type: plan
status: complete
created: 2026-07-18
completed: 2026-07-25
---

# GeoPackage Explorer Enhanced Viewer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the GeoPackage Explorer from a table-listing tool into a full results viewer with blob deserialization, structured filtering, custom XY plots, and CSV export.

**Architecture:** Service layer (pure Python numpy blob ops) → View widgets (array viewer, plot canvas, enhanced preview dialog) → Explorer dialog (QTabWidget wrapping existing + new Plot tab).

**Tech Stack:** PyQt5, numpy, matplotlib, sqlite3, Python csv module

**Parallel strategy:**
- Batch 1 (parallel): Task 1 (service) + Task 2 (array viewer widget) + Task 4 (tests for service)
- Batch 2 (parallel): Task 3 (enhanced preview dialog, depends on 1+2) + Task 5 (plot tab, depends on 1)
- Batch 3: Task 6 (explorer dialog mods, depends on 5) + remaining tests
- Task 7: Final integration verification

---

### Task 1: numpy_blob_service.py — Blob deserialization, column discovery, WHERE builder, CSV export

**Files:**
- Create: `swe2d/workbench/services/numpy_blob_service.py`
- Test: `tests/test_numpy_blob_service.py`

- [ ] **Step 1: Write failing tests for blob deserialization**

```python
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
```

Run: `mamba run -n qgis_stable python3 -m pytest tests/test_numpy_blob_service.py -v`
Expected: All tests FAIL (module not found / import errors)

- [ ] **Step 2: Run tests to verify they fail**

Execute the command above. Confirm FAIL.

- [ ] **Step 3: Write the service module**

```python
"""swe2d/workbench/services/numpy_blob_service.py

Pure Python, zero-Qt service for blob deserialization, column discovery,
filtered queries, and CSV export for SWE2D GeoPackage tables.
"""

from __future__ import annotations

import csv
import logging
import os
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _open_conn(gpkg_path: str) -> sqlite3.Connection:
    return sqlite3.connect(gpkg_path)


def _get_blob_shape_from_metadata(
    conn: sqlite3.Connection,
    table: str,
    blob_col: str,
    row_pk: dict[str, Any],
) -> Optional[Tuple[int, ...]]:
    """Determine blob shape using sibling metadata columns (n_cells, n_timesteps, etc.)."""
    cur = conn.cursor()
    meta_cols_map = {
        "swe2d_baked_results": {"n_cells", "n_timesteps"},
        "swe2d_baked_line_ts": {"n_timesteps"},
        "swe2d_baked_line_profiles": {"n_stations", "n_timesteps"},
        "swe2d_baked_coupling": {"n_timesteps"},
        "swe2d_baked_pipe_cell_ts": {"n_timesteps"},
        "swe2d_baked_overlay_fields": {"n_timesteps"},
        "swe2d_baked_mesh": {"n_nodes", "n_cells", "n_edges"},
    }
    meta_cols = meta_cols_map.get(table, set())
    if not meta_cols:
        return None
    col_list = ", ".join(_quote_ident(c) for c in meta_cols)
    if not col_list:
        return None
    pk_col, pk_val = next(iter(row_pk.items()))
    try:
        cur.execute(
            f"SELECT {col_list} FROM {_quote_ident(table)} WHERE {_quote_ident(pk_col)} = ?",
            (pk_val,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        meta_vals = dict(zip(meta_cols, row))
    except sqlite3.Error:
        return None

    blob_col_lower = blob_col.lower()

    if table == "swe2d_baked_results":
        n_cells = meta_vals.get("n_cells")
        n_timesteps = meta_vals.get("n_timesteps")
        if n_cells is None or n_timesteps is None:
            return None
        if blob_col_lower == "times_blob":
            return (int(n_timesteps),)
        if blob_col_lower in ("h_blob", "hu_blob", "hv_blob"):
            return (int(n_timesteps), int(n_cells))
        if blob_col_lower in ("max_h_blob", "max_hu_blob", "max_hv_blob"):
            return (int(n_cells),)
        return None

    if table in ("swe2d_baked_line_ts", "swe2d_baked_coupling", "swe2d_baked_pipe_cell_ts", "swe2d_baked_overlay_fields"):
        n_timesteps = meta_vals.get("n_timesteps")
        if n_timesteps is not None:
            if blob_col_lower in ("times_blob", "depth_blob", "vel_blob", "wse_blob",
                                   "bed_blob", "flow_blob", "wet_frac_blob", "fr_blob",
                                   "values_blob"):
                return (int(n_timesteps),)
        return None

    if table == "swe2d_baked_line_profiles":
        n_stations = meta_vals.get("n_stations")
        n_timesteps = meta_vals.get("n_timesteps")
        if n_stations is None or n_timesteps is None:
            return None
        if blob_col_lower == "station_blob":
            return (int(n_stations),)
        if blob_col_lower == "times_blob":
            return (int(n_timesteps),)
        if blob_col_lower in ("depth_blob", "vel_blob", "wse_blob", "bed_blob",
                               "flow_qn_blob", "fr_blob", "wet_blob"):
            return (int(n_stations), int(n_timesteps))
        return None

    if table == "swe2d_baked_mesh":
        n_nodes = meta_vals.get("n_nodes")
        n_cells = meta_vals.get("n_cells")
        n_edges = meta_vals.get("n_edges")
        if blob_col_lower == "baked_blob":
            # Return None — this is a serialized mesh, not a simple array
            return None
        return None

    return None


def deserialize_blob_to_array(
    gpkg_path: str,
    table: str,
    column: str,
    **pk_values: Any,
) -> Optional[np.ndarray]:
    """Deserialize a BLOB column into a numpy array.

    Reads the blob from the row identified by pk_values, determines
    shape from sibling metadata columns, and reshapes.

    Returns None if the column is not a blob, the table doesn't exist,
    or the shape cannot be determined.
    """
    if not gpkg_path or not os.path.exists(gpkg_path):
        return None
    if not pk_values:
        logger.warning("deserialize_blob_to_array requires pk_values for row identification")
        return None
    try:
        conn = _open_conn(gpkg_path)
        try:
            cur = conn.cursor()
            pk_col, pk_val = next(iter(pk_values.items()))
            cur.execute(
                f"SELECT {_quote_ident(column)} FROM {_quote_ident(table)} "
                f"WHERE {_quote_ident(pk_col)} = ?",
                (pk_val,),
            )
            row = cur.fetchone()
            if row is None or row[0] is None:
                return None
            blob = row[0]
            try:
                arr_1d = np.frombuffer(blob, dtype=np.float64)
            except Exception:
                return None
            shape = _get_blob_shape_from_metadata(conn, table, column, pk_values)
            if shape is not None and len(shape) > 0:
                expected_len = 1
                for s in shape:
                    expected_len *= s
                if len(arr_1d) == expected_len:
                    return arr_1d.reshape(shape)
            return arr_1d
        finally:
            conn.close()
    except sqlite3.Error:
        logger.exception("Error deserializing blob %s.%s", table, column)
        return None


def discover_plottable_columns(gpkg_path: str, table: str) -> List[Dict[str, Any]]:
    """Discover numeric and deserializable blob columns in a table.

    Returns a list of dicts:
        { "name": str, "kind": "metadata_int"|"metadata_float"|"blob_1d"|"blob_2d",
          "ndim": int, "shape": tuple|None }
    """
    if not gpkg_path or not os.path.exists(gpkg_path):
        return []
    try:
        conn = _open_conn(gpkg_path)
        try:
            cur = conn.cursor()
            cur.execute(f"PRAGMA table_info({_quote_ident(table)})")
            columns_info = cur.fetchall()
            if not columns_info:
                return []

            result: List[Dict[str, Any]] = []
            col_names = [str(r[1]) for r in columns_info]
            col_types = [str(r[2] or "").upper() for r in columns_info]

            for name, ctype in zip(col_names, col_types):
                if ctype in ("INTEGER", "REAL", "FLOAT", "DOUBLE", "NUMERIC"):
                    result.append({
                        "name": name,
                        "kind": "metadata_numeric",
                        "ndim": 1,
                        "shape": None,
                    })
                elif ctype in ("TEXT",):
                    result.append({
                        "name": name,
                        "kind": "metadata_text",
                        "ndim": 0,
                        "shape": None,
                    })
                elif ctype == "BLOB":
                    # Sample first row to determine shape
                    cur.execute(
                        f"SELECT * FROM {_quote_ident(table)} LIMIT 1"
                    )
                    sample = cur.fetchone()
                    if sample is None:
                        continue
                    blob_idx = col_names.index(name)
                    blob = sample[blob_idx]
                    if blob is None:
                        continue
                    try:
                        arr_1d = np.frombuffer(blob, dtype=np.float64)
                    except Exception:
                        continue

                    shape = _get_blob_shape_from_metadata(
                        conn, table, name,
                        {col_names[0]: sample[0]},  # use first column as PK
                    )
                    if shape is not None:
                        result.append({
                            "name": name,
                            "kind": "blob_1d" if len(shape) == 1 else "blob_2d",
                            "ndim": len(shape),
                            "shape": shape,
                        })
                    else:
                        result.append({
                            "name": name,
                            "kind": "blob_unknown",
                            "ndim": 1,
                            "shape": (len(arr_1d),),
                        })
                else:
                    result.append({"name": name, "kind": "other", "ndim": 0, "shape": None})

            return result
        finally:
            conn.close()
    except sqlite3.Error:
        logger.exception("Error discovering columns in %s.%s", gpkg_path, table)
        return []


def build_where_clause(column: str, operator: str, value: str) -> Tuple[str, List[Any]]:
    """Build a parameterized SQL WHERE clause fragment.

    Returns (sql_fragment, params_list).

    Supported operators: =, !=, >, <, >=, <=, LIKE, IN, BETWEEN, IS NULL, IS NOT NULL.
    """
    col_quoted = _quote_ident(column)
    op = operator.strip().upper()

    if op in ("IS NULL", "IS NOT NULL"):
        return f"{col_quoted} {op}", []

    if op == "IN":
        parts = [p.strip() for p in value.split(",") if p.strip()]
        placeholders = ",".join("?" for _ in parts)
        return f"{col_quoted} IN ({placeholders})", parts

    if op == "BETWEEN":
        parts = [p.strip() for p in value.split(",") if p.strip()]
        if len(parts) != 2:
            parts = [value, value]
        return f"{col_quoted} BETWEEN ? AND ?", parts

    # Numeric comparison or LIKE
    return f"{col_quoted} {op} ?", [value]


def get_filtered_rows(
    gpkg_path: str,
    table: str,
    *,
    column: Optional[str] = None,
    op: Optional[str] = None,
    value: Optional[str] = None,
    limit: int = 250,
) -> List[Dict[str, Any]]:
    """Query table rows with optional structured filter.

    Returns list of dicts keyed by column name.
    """
    if not gpkg_path or not os.path.exists(gpkg_path):
        return []
    try:
        conn = _open_conn(gpkg_path)
        try:
            cur = conn.cursor()
            base_sql = f"SELECT * FROM {_quote_ident(table)}"

            params: List[Any] = []
            if column and op:
                where_sql, params = build_where_clause(column, op, value or "")
                base_sql += f" WHERE {where_sql}"

            base_sql += f" LIMIT {int(limit)}"
            cur.execute(base_sql, params)
            col_names = [str(d[0]) for d in cur.description]
            return [dict(zip(col_names, row)) for row in cur.fetchall()]
        finally:
            conn.close()
    except sqlite3.Error:
        logger.exception("Error querying %s.%s", gpkg_path, table)
        return []


def export_table_to_csv(
    gpkg_path: str,
    table: str,
    filepath: str,
    *,
    columns: Optional[List[str]] = None,
    column: Optional[str] = None,
    op: Optional[str] = None,
    value: Optional[str] = None,
) -> None:
    """Export filtered table rows to CSV.

    Args:
        gpkg_path: Path to GeoPackage.
        table: Table name.
        filepath: Output .csv path.
        columns: Subset of columns to export; None = all.
        column, op, value: Optional structured filter.
    """
    if not gpkg_path or not os.path.exists(gpkg_path):
        raise ValueError(f"GeoPackage not found: {gpkg_path}")

    rows = get_filtered_rows(gpkg_path, table, column=column, op=op, value=value, limit=0)
    if not rows:
        raise ValueError(f"No data found in {table}")

    if columns is not None:
        fieldnames = [c for c in columns if c in rows[0]]
    else:
        fieldnames = list(rows[0].keys())

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {}
            for fn in fieldnames:
                val = row.get(fn)
                if isinstance(val, (bytes, memoryview)):
                    out[fn] = f"<blob {len(val)} bytes>"
                elif isinstance(val, float):
                    out[fn] = f"{val:.6g}"
                else:
                    out[fn] = str(val) if val is not None else ""
            writer.writerow(out)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `mamba run -n qgis_stable python3 -m pytest tests/test_numpy_blob_service.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add swe2d/workbench/services/numpy_blob_service.py tests/test_numpy_blob_service.py
git commit -m "feat: add numpy blob service for GPKG explorer enhanced viewer"
```


### Task 2: gpkg_array_viewer_widget.py — Reusable array inspection widget

**Files:**
- Create: `swe2d/workbench/dialogs/gpkg_array_viewer_widget.py`

**Dependencies:** Task 1 (numpy_blob_service.py)

- [ ] **Step 1: Write the array viewer widget**

```python
"""swe2d/workbench/dialogs/gpkg_array_viewer_widget.py

Reusable widget to display a deserialized numpy array in a QTableWidget
with a mini-plot tab.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from qgis.PyQt import QtCore, QtWidgets

from swe2d.workbench.dialogs._plot_utils import try_import_matplotlib_qt

logger = logging.getLogger(__name__)

FigureCanvasQt, Figure, _mtri = try_import_matplotlib_qt()


class ArrayViewerWidget(QtWidgets.QWidget):
    """Dual-tab widget showing array data as a table + quick plot.

    Displays a numpy array: 1D arrays show as a single column;
    2D arrays show rows=dim0, cols=dim1 with slice controls.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._array: Optional[np.ndarray] = None
        self._col_name: str = ""
        self._slice_row: int = 0
        self._slice_col: int = 0

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        self._info_lbl = QtWidgets.QLabel("No array selected")
        root.addWidget(self._info_lbl)

        self._slice_row_layout = QtWidgets.QHBoxLayout()
        self._slice_row_lbl = QtWidgets.QLabel("Row (timestep):")
        self._slice_row_spin = QtWidgets.QSpinBox()
        self._slice_row_spin.setRange(0, 0)
        self._slice_row_spin.valueChanged.connect(self._on_slice_changed)
        self._slice_row_layout.addWidget(self._slice_row_lbl)
        self._slice_row_layout.addWidget(self._slice_row_spin)

        self._slice_col_layout = QtWidgets.QHBoxLayout()
        self._slice_col_lbl = QtWidgets.QLabel("Col (cell):")
        self._slice_col_spin = QtWidgets.QSpinBox()
        self._slice_col_spin.setRange(0, 0)
        self._slice_col_spin.valueChanged.connect(self._on_slice_changed)
        self._slice_col_layout.addWidget(self._slice_col_lbl)
        self._slice_col_layout.addWidget(self._slice_col_spin)

        self._slice_widget = QtWidgets.QWidget()
        slice_inner = QtWidgets.QHBoxLayout(self._slice_widget)
        slice_inner.setContentsMargins(0, 0, 0, 0)
        slice_inner.addLayout(self._slice_row_layout)
        slice_inner.addLayout(self._slice_col_layout)
        slice_inner.addStretch(1)
        self._slice_widget.setVisible(False)
        root.addWidget(self._slice_widget)

        self._tabs = QtWidgets.QTabWidget()
        root.addWidget(self._tabs, stretch=1)

        self._data_table = QtWidgets.QTableWidget()
        self._data_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self._data_table.setAlternatingRowColors(True)
        self._tabs.addTab(self._data_table, "Array Data")

        self._plot_widget = QtWidgets.QWidget()
        self._plot_layout = QtWidgets.QVBoxLayout(self._plot_widget)
        self._plot_layout.setContentsMargins(0, 0, 0, 0)
        self._canvas = None
        self._tabs.addTab(self._plot_widget, "Quick Plot")

    def show_array(self, array: np.ndarray, col_name: str = ""):
        """Display a numpy array."""
        self._array = array
        self._col_name = col_name

        shape_str = "×".join(str(s) for s in array.shape)
        dtype_str = str(array.dtype)
        self._info_lbl.setText(f"{col_name}: {dtype_str}[{shape_str}]" if col_name else f"{dtype_str}[{shape_str}]")

        ndim = array.ndim
        self._slice_widget.setVisible(ndim == 2)

        if ndim == 2:
            self._slice_row_spin.blockSignals(True)
            self._slice_col_spin.blockSignals(True)
            self._slice_row_spin.setRange(0, array.shape[0] - 1)
            self._slice_col_spin.setRange(0, array.shape[1] - 1)
            self._slice_row_spin.setValue(0)
            self._slice_col_spin.setValue(0)
            self._slice_row_spin.blockSignals(False)
            self._slice_col_spin.blockSignals(False)
            self._populate_table_2d(array)
        else:
            self._populate_table_1d(array)

        self._update_quick_plot()

    def _populate_table_1d(self, array: np.ndarray):
        """Fill table for 1D array."""
        self._data_table.setColumnCount(2)
        self._data_table.setHorizontalHeaderLabels(["Index", self._col_name or "Value"])
        self._data_table.setRowCount(len(array))
        for i in range(len(array)):
            self._data_table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(i)))
            self._data_table.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{array[i]:.6g}"))
        self._data_table.resizeColumnsToContents()

    def _populate_table_2d(self, array: np.ndarray):
        """Fill table showing entire 2D array (cells as columns, timesteps as rows)."""
        self._data_table.setColumnCount(array.shape[1])
        self._data_table.setRowCount(array.shape[0])
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                self._data_table.setItem(i, j, QtWidgets.QTableWidgetItem(f"{array[i, j]:.6g}"))
        self._data_table.resizeColumnsToContents()

    def _on_slice_changed(self):
        if self._array is None or self._array.ndim < 2:
            return
        r = self._slice_row_spin.value()
        c = self._slice_col_spin.value()
        if self._tabs.currentIndex() == 0:
            self._populate_table_slice_1d(self._array[r, :], c)
        self._update_quick_plot()

    def _populate_table_slice_1d(self, array_1d: np.ndarray, col_idx: int):
        """Show a 1D slice of a 2D array (one column of the 2D matrix)."""
        self._data_table.setColumnCount(2)
        self._data_table.setHorizontalHeaderLabels(["Index", f"{self._col_name} [{col_idx}]"])
        self._data_table.setRowCount(len(array_1d))
        for i in range(len(array_1d)):
            self._data_table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(i)))
            self._data_table.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{array_1d[i]:.6g}"))
        self._data_table.resizeColumnsToContents()

    def get_slice_for_plot(self) -> Optional[np.ndarray]:
        """Return the currently visible 1D slice for XY plotting."""
        if self._array is None:
            return None
        if self._array.ndim == 1:
            return self._array
        if self._array.ndim >= 2:
            r = self._slice_row_spin.value()
            return self._array[r, :]
        return None

    def _update_quick_plot(self):
        if FigureCanvasQt is None:
            return
        if self._array is None:
            return

        if self._canvas is not None:
            self._plot_layout.removeWidget(self._canvas)
            self._canvas.deleteLater()
            self._canvas = None

        from matplotlib.figure import Figure
        fig = Figure(figsize=(5, 3))
        ax = fig.add_subplot(111)

        if self._array.ndim == 1:
            ax.plot(self._array)
            ax.set_xlabel("Index")
            ax.set_ylabel(self._col_name or "Value")
        elif self._array.ndim >= 2:
            r = self._slice_row_spin.value()
            ax.plot(self._array[r, :])
            ax.set_xlabel("Index")
            ax.set_ylabel(f"{self._col_name} [row {r}]" if self._col_name else f"Row {r}")

        ax.set_title(f"Quick Plot - {self._col_name}" if self._col_name else "Quick Plot")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        self._canvas = FigureCanvasQt(fig)
        self._plot_layout.addWidget(self._canvas)
```

- [ ] **Step 2: Commit**

```bash
git add swe2d/workbench/dialogs/gpkg_array_viewer_widget.py
git commit -m "feat: add reusable array viewer widget for GPKG explorer"
```


### Task 3: gpkg_plot_tab.py — XY plot tab widget

**Files:**
- Create: `swe2d/workbench/dialogs/gpkg_plot_tab.py`

**Dependencies:** Task 1 (numpy_blob_service.py)

- [ ] **Step 1: Write the plot tab widget**

```python
"""swe2d/workbench/dialogs/gpkg_plot_tab.py

XY plot tab widget for the GeoPackage Explorer. Lets users select X and Y
columns (including deserialized blob arrays) and renders a matplotlib plot.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
from qgis.PyQt import QtCore, QtWidgets

from swe2d.workbench.dialogs._plot_utils import try_import_matplotlib_qt
from swe2d.workbench.services.numpy_blob_service import (
    deserialize_blob_to_array,
    discover_plottable_columns,
    export_table_to_csv,
)

logger = logging.getLogger(__name__)

FigureCanvasQt, Figure, _mtri = try_import_matplotlib_qt()


class GpkgPlotTab(QtWidgets.QWidget):
    """Plot tab embedded in the explorer dialog for XY plotting."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._gpkg_path: str = ""
        self._table: str = ""
        self._x_col: str = ""
        self._y_col: str = ""
        self._available_cols: List[Dict[str, Any]] = []
        self._pk_col: str = ""

        root = QtWidgets.QVBoxLayout(self)

        source_row = QtWidgets.QHBoxLayout()
        source_row.addWidget(QtWidgets.QLabel("GeoPackage:"))
        self._gpkg_lbl = QtWidgets.QLabel("(none)")
        source_row.addWidget(self._gpkg_lbl, stretch=1)
        root.addLayout(source_row)

        source_row2 = QtWidgets.QHBoxLayout()
        source_row2.addWidget(QtWidgets.QLabel("Table:"))
        self._table_lbl = QtWidgets.QLabel("(none)")
        source_row2.addWidget(self._table_lbl, stretch=1)
        root.addLayout(source_row2)

        controls_row = QtWidgets.QHBoxLayout()
        controls_row.addWidget(QtWidgets.QLabel("X-axis:"))
        self._x_combo = QtWidgets.QComboBox()
        self._x_combo.setMinimumWidth(180)
        controls_row.addWidget(self._x_combo)

        controls_row.addWidget(QtWidgets.QLabel("Y-axis:"))
        self._y_combo = QtWidgets.QComboBox()
        self._y_combo.setMinimumWidth(180)
        controls_row.addWidget(self._y_combo)

        controls_row.addWidget(QtWidgets.QLabel("Plot type:"))
        self._plot_type_combo = QtWidgets.QComboBox()
        self._plot_type_combo.addItems(["Scatter", "Line", "Line+Scatter"])
        controls_row.addWidget(self._plot_type_combo)

        self._log_x_cb = QtWidgets.QCheckBox("Log X")
        controls_row.addWidget(self._log_x_cb)
        self._log_y_cb = QtWidgets.QCheckBox("Log Y")
        controls_row.addWidget(self._log_y_cb)

        self._plot_btn = QtWidgets.QPushButton("Plot")
        self._plot_btn.clicked.connect(self._render_plot)
        controls_row.addWidget(self._plot_btn)
        controls_row.addStretch(1)
        root.addLayout(controls_row)

        # Slice controls for 2D arrays
        slice_row = QtWidgets.QHBoxLayout()
        self._slice_lbl = QtWidgets.QLabel("Slice (for 2D arrays):")
        self._slice_spin = QtWidgets.QSpinBox()
        self._slice_spin.setRange(0, 0)
        self._slice_spin.setToolTip("For 2D arrays, which index to extract as 1D series")
        slice_row.addWidget(self._slice_lbl)
        slice_row.addWidget(self._slice_spin)
        slice_row.addStretch(1)
        self._slice_widget = QtWidgets.QWidget()
        slice_inner = QtWidgets.QHBoxLayout(self._slice_widget)
        slice_inner.setContentsMargins(0, 0, 0, 0)
        slice_inner.addLayout(slice_row)
        self._slice_widget.setVisible(False)
        root.addWidget(self._slice_widget)

        # Canvas
        self._canvas_container = QtWidgets.QWidget()
        self._canvas_layout = QtWidgets.QVBoxLayout(self._canvas_container)
        self._canvas_layout.setContentsMargins(0, 0, 0, 0)
        self._canvas = None
        root.addWidget(self._canvas_container, stretch=1)

        # Export buttons
        btn_row = QtWidgets.QHBoxLayout()
        self._export_csv_btn = QtWidgets.QPushButton("Export CSV")
        self._export_csv_btn.clicked.connect(self._export_csv)
        btn_row.addWidget(self._export_csv_btn)

        if FigureCanvasQt is not None:
            self._export_png_btn = QtWidgets.QPushButton("Export PNG")
            self._export_png_btn.clicked.connect(self._export_png)
            btn_row.addWidget(self._export_png_btn)

        btn_row.addStretch(1)
        root.addLayout(btn_row)

        # Enable state
        self._x_combo.currentTextChanged.connect(self._on_column_changed)
        self._y_combo.currentTextChanged.connect(self._on_column_changed)
        self._on_column_changed()

    def set_table(self, gpkg_path: str, table: str):
        """Point the plot tab at a specific table and refresh column dropdowns."""
        self._gpkg_path = gpkg_path
        self._table = table
        self._gpkg_lbl.setText(os.path.basename(gpkg_path))
        self._table_lbl.setText(table)
        self._refresh_columns()

    def _refresh_columns(self):
        """Discover plottable columns and populate dropdowns."""
        self._x_combo.blockSignals(True)
        self._y_combo.blockSignals(True)
        self._x_combo.clear()
        self._y_combo.clear()

        if not self._gpkg_path or not self._table:
            self._available_cols = []
            self._x_combo.blockSignals(False)
            self._y_combo.blockSignals(False)
            return

        self._available_cols = discover_plottable_columns(self._gpkg_path, self._table)
        for col in self._available_cols:
            name = col["name"]
            kind = col["kind"]
            suffix = ""
            if kind == "blob_1d":
                suffix = " [1D]"
            elif kind == "blob_2d":
                suffix = " [2D]"
            elif kind == "metadata_numeric":
                suffix = " [num]"
            label = f"{name}{suffix}"
            self._x_combo.addItem(label, name)
            self._y_combo.addItem(label, name)

        if self._x_combo.count() > 0:
            self._x_combo.setCurrentIndex(0)
        if self._y_combo.count() > 1:
            self._y_combo.setCurrentIndex(1)
        elif self._y_combo.count() > 0:
            self._y_combo.setCurrentIndex(0)

        self._x_combo.blockSignals(False)
        self._y_combo.blockSignals(False)

    def _on_column_changed(self):
        x_name = self._x_combo.currentData() or ""
        y_name = self._y_combo.currentData() or ""
        has_both = bool(x_name and y_name)
        self._plot_btn.setEnabled(has_both)

        # Show slice controls if either column is 2D
        has_2d = False
        for col in self._available_cols:
            if col["name"] in (x_name, y_name) and col.get("ndim", 1) == 2:
                has_2d = True
                max_slice = (col["shape"][0] - 1) if col["shape"] else 0
                self._slice_spin.setRange(0, max(0, max_slice))
                break
        self._slice_widget.setVisible(has_2d)

    def _get_column_data(self, col_name: str) -> Optional[np.ndarray]:
        """Extract a 1D numpy array for a column (metadata or blob)."""
        col_info = next((c for c in self._available_cols if c["name"] == col_name), None)
        if col_info is None:
            return None

        if col_info["kind"].startswith("metadata_numeric"):
            rows = self._get_metadata_rows()
            if not rows:
                return None
            vals = []
            for r in rows:
                v = r.get(col_name)
                if v is not None:
                    try:
                        vals.append(float(v))
                    except (ValueError, TypeError):
                        vals.append(float("nan"))
                else:
                    vals.append(float("nan"))
            return np.array(vals)

        if col_info["kind"].startswith("blob"):
            blob_arr = deserialize_blob_to_array(self._gpkg_path, self._table, col_name, **self._get_pk())
            if blob_arr is None:
                return None
            if blob_arr.ndim == 2:
                idx = self._slice_spin.value()
                return blob_arr[idx, :]
            return blob_arr

        return None

    def _get_metadata_rows(self):
        """Fetch first N rows of metadata columns."""
        from swe2d.workbench.services.numpy_blob_service import get_filtered_rows
        return get_filtered_rows(self._gpkg_path, self._table, limit=5000)

    def _get_pk(self) -> dict:
        """Get the primary key column name and first value."""
        if not self._available_cols:
            return {}
        rows = self._get_metadata_rows()
        if not rows:
            return {}
        first_col = self._available_cols[0]["name"]
        return {first_col: rows[0].get(first_col, "")}

    def _render_plot(self):
        x_name = self._x_combo.currentData() or ""
        y_name = self._y_combo.currentData() or ""
        if not x_name or not y_name:
            return

        x_data = self._get_column_data(x_name)
        y_data = self._get_column_data(y_name)
        if x_data is None or y_data is None:
            QtWidgets.QMessageBox.warning(self, "Plot Error", f"Cannot extract data for {x_name} vs {y_name}")
            return

        # Ensure same length
        min_len = min(len(x_data), len(y_data))
        if min_len == 0:
            QtWidgets.QMessageBox.warning(self, "Plot Error", "No data points to plot")
            return
        x_data = x_data[:min_len]
        y_data = y_data[:min_len]

        self._x_col = x_name
        self._y_col = y_name

        if FigureCanvasQt is None:
            QtWidgets.QMessageBox.warning(self, "Plot Error", "matplotlib not available")
            return

        if self._canvas is not None:
            self._canvas_layout.removeWidget(self._canvas)
            self._canvas.deleteLater()
            self._canvas = None

        fig = Figure(figsize=(6, 4))
        ax = fig.add_subplot(111)

        plot_type = self._plot_type_combo.currentText()

        if self._log_x_cb.isChecked():
            ax.set_xscale("log")
        if self._log_y_cb.isChecked():
            ax.set_yscale("log")

        if plot_type == "Scatter":
            ax.scatter(x_data, y_data, s=8, alpha=0.7)
        elif plot_type == "Line":
            ax.plot(x_data, y_data)
        else:
            ax.plot(x_data, y_data)
            ax.scatter(x_data, y_data, s=8, alpha=0.7)

        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.set_title(f"{y_name} vs {x_name}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        self._canvas = FigureCanvasQt(fig)
        self._canvas_layout.addWidget(self._canvas)

    def _export_csv(self):
        """Export plotted X and Y data to CSV."""
        if not self._x_col or not self._y_col:
            QtWidgets.QMessageBox.warning(self, "Export CSV", "No plot data to export")
            return

        filepath, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Data to CSV", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not filepath:
            return

        x_data = self._get_column_data(self._x_col)
        y_data = self._get_column_data(self._y_col)
        if x_data is None or y_data is None:
            return

        min_len = min(len(x_data), len(y_data))
        import csv
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([self._x_col, self._y_col])
            for i in range(min_len):
                writer.writerow([f"{x_data[i]:.10g}", f"{y_data[i]:.10g}"])

        QtWidgets.QMessageBox.information(self, "Export CSV", f"Exported {min_len} rows to {filepath}")

    def _export_png(self):
        """Export the current plot as PNG."""
        if self._canvas is None:
            QtWidgets.QMessageBox.warning(self, "Export PNG", "No plot to export")
            return
        filepath, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Plot to PNG", "", "PNG Files (*.png);;All Files (*)"
        )
        if not filepath:
            return
        self._canvas.figure.savefig(filepath, dpi=150, bbox_inches="tight")
        QtWidgets.QMessageBox.information(self, "Export PNG", f"Plot saved to {filepath}")
```

- [ ] **Step 2: Commit**

```bash
git add swe2d/workbench/dialogs/gpkg_plot_tab.py
git commit -m "feat: add XY plot tab widget for GPKG explorer"
```


### Task 4: Enhanced preview dialog — rewrite sqlite_preview_dialog.py

**Files:**
- Modify: `swe2d/workbench/dialogs/sqlite_preview_dialog.py` (full rewrite)

**Dependencies:** Task 1, Task 2

- [ ] **Step 1: Write the enhanced preview dialog**

Replace the entire `sqlite_preview_dialog.py` with:

```python
#!/usr/bin/env python3
"""Enhanced GeoPackage table preview dialog with structured filtering,
blob deserialization, array viewer, and CSV export."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from qgis.PyQt import QtCore, QtWidgets

from swe2d.results.db_utils import get_table_info, get_table_contents
from swe2d.workbench.dialogs.gpkg_array_viewer_widget import ArrayViewerWidget
from swe2d.workbench.services.numpy_blob_service import (
    deserialize_blob_to_array,
    discover_plottable_columns,
    export_table_to_csv,
    get_filtered_rows,
)

logger = logging.getLogger(__name__)

_TABLE_KIND_ACTIONS = {
    "run_log": "open+preview",
    "config": "open+preview",
    "line_results": "open+preview",
    "coupling_results": "open+preview",
    "mesh_results": "open+preview",
    "system": "preview",
    "table": "preview",
}


class SWE2DEnhancedTablePreviewDialog(QtWidgets.QDialog):
    """Enhanced table preview dialog with filter, dual-panel view, and array inspection."""

    plot_requested = QtCore.pyqtSignal(str, str)  # gpkg_path, table_name

    def __init__(
        self,
        gpkg_path: str,
        table_name: str,
        title: str = "Table Viewer",
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(str(title or "Table Viewer"))
        self.resize(1100, 750)
        self._gpkg_path = str(gpkg_path or "")
        self._table_name = str(table_name or "")
        self._current_rows: List[Dict[str, Any]] = []
        self._plottable_cols: List[Dict[str, Any]] = []
        self._blob_data_cache: Dict[str, Any] = {}

        root = QtWidgets.QVBoxLayout(self)
        root.addWidget(QtWidgets.QLabel(f"Source: {self._gpkg_path}\nTable: {self._table_name}"))

        # ── Filter bar ────────────────────────────────────────────────────
        filter_layout = QtWidgets.QHBoxLayout()
        filter_layout.addWidget(QtWidgets.QLabel("Filter:"))
        self._filter_col_combo = QtWidgets.QComboBox()
        self._filter_col_combo.setMinimumWidth(160)
        filter_layout.addWidget(self._filter_col_combo)
        self._filter_op_combo = QtWidgets.QComboBox()
        self._filter_op_combo.addItems(["=", "!=", ">", "<", ">=", "<=", "LIKE", "IN", "BETWEEN", "IS NULL", "IS NOT NULL"])
        filter_layout.addWidget(self._filter_op_combo)
        self._filter_value_edit = QtWidgets.QLineEdit()
        self._filter_value_edit.setPlaceholderText("Filter value...")
        self._filter_value_edit.setMinimumWidth(120)
        filter_layout.addWidget(self._filter_value_edit)
        self._filter_apply_btn = QtWidgets.QPushButton("Apply")
        self._filter_apply_btn.clicked.connect(self._apply_filter)
        filter_layout.addWidget(self._filter_apply_btn)
        self._filter_clear_btn = QtWidgets.QPushButton("Clear")
        self._filter_clear_btn.clicked.connect(self._clear_filter)
        filter_layout.addWidget(self._filter_clear_btn)
        filter_layout.addStretch(1)

        # Filter op toggle visibility
        self._filter_op_combo.currentTextChanged.connect(self._on_filter_op_changed)
        self._on_filter_op_changed(self._filter_op_combo.currentText())

        root.addLayout(filter_layout)

        # ── Splitter: metadata table (top) + array viewer (bottom) ────────
        self._splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)

        # Top: metadata table
        self._meta_table = QtWidgets.QTableWidget()
        self._meta_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self._meta_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self._meta_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self._meta_table.setAlternatingRowColors(True)
        self._meta_table.horizontalHeader().setStretchLastSection(True)
        self._meta_table.itemSelectionChanged.connect(self._on_meta_row_selected)
        self._splitter.addWidget(self._meta_table)

        # Bottom: array viewer
        self._array_viewer = ArrayViewerWidget()
        self._array_viewer.setMinimumHeight(200)
        self._splitter.addWidget(self._array_viewer)

        self._splitter.setStretchFactor(0, 2)
        self._splitter.setStretchFactor(1, 1)
        root.addWidget(self._splitter, stretch=1)

        # ── Bottom controls ───────────────────────────────────────────────
        ctrl_row = QtWidgets.QHBoxLayout()
        ctrl_row.addWidget(QtWidgets.QLabel("Limit:"))
        self._limit_spin = QtWidgets.QSpinBox()
        self._limit_spin.setRange(10, 10000)
        self._limit_spin.setValue(250)
        self._limit_spin.valueChanged.connect(lambda _v: self.refresh_table())
        ctrl_row.addWidget(self._limit_spin)

        self._refresh_btn = QtWidgets.QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self.refresh_table)
        ctrl_row.addWidget(self._refresh_btn)

        self._export_csv_btn = QtWidgets.QPushButton("Export CSV")
        self._export_csv_btn.clicked.connect(self._export_csv)
        ctrl_row.addWidget(self._export_csv_btn)

        self._send_plot_btn = QtWidgets.QPushButton("Send to Explorer Plot")
        self._send_plot_btn.clicked.connect(self._send_to_plot)
        ctrl_row.addWidget(self._send_plot_btn)

        ctrl_row.addStretch(1)

        self._row_count_lbl = QtWidgets.QLabel("")
        ctrl_row.addWidget(self._row_count_lbl)
        root.addLayout(ctrl_row)

        # ── Buttons ───────────────────────────────────────────────────────
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self.refresh_table()

    def _on_filter_op_changed(self, op: str):
        """Hide value input for IS NULL / IS NOT NULL."""
        show = op not in ("IS NULL", "IS NOT NULL")
        self._filter_value_edit.setVisible(show)

    def _populate_filter_columns(self):
        """Fill filter column dropdown with non-blob columns."""
        self._filter_col_combo.clear()
        cols = get_table_info(self._gpkg_path, self._table_name)
        for name in cols:
            self._filter_col_combo.addItem(name)

    def refresh_table(self):
        """(Re)load metadata rows and rebuild the top table."""
        self._meta_table.setRowCount(0)
        self._meta_table.setColumnCount(0)
        self._array_viewer.show_array(None)  # clear
        self._blob_data_cache.clear()

        if not self._gpkg_path or not self._table_name or not os.path.exists(self._gpkg_path):
            return

        self._plottable_cols = discover_plottable_columns(self._gpkg_path, self._table_name)
        self._populate_filter_columns()
        self._load_rows()

    def _load_rows(self):
        """Load rows (with current filter if active) into metadata table."""
        limit = int(self._limit_spin.value())
        filter_col = self._filter_col_combo.currentText()
        filter_op = self._filter_op_combo.currentText()
        filter_val = self._filter_value_edit.text().strip()

        if filter_col and filter_op:
            if filter_op not in ("IS NULL", "IS NOT NULL") and not filter_val:
                self._current_rows = get_filtered_rows(self._gpkg_path, self._table_name, limit=limit)
            else:
                self._current_rows = get_filtered_rows(
                    self._gpkg_path, self._table_name,
                    column=filter_col, op=filter_op, value=filter_val,
                    limit=limit,
                )
        else:
            self._current_rows = get_filtered_rows(self._gpkg_path, self._table_name, limit=limit)

        if not self._current_rows:
            self._row_count_lbl.setText("0 rows")
            return

        # Build columns set from all rows
        all_cols = list(self._current_rows[0].keys())
        self._meta_table.setColumnCount(len(all_cols))
        self._meta_table.setHorizontalHeaderLabels(all_cols)

        blob_cols = set()
        for col in self._plottable_cols:
            if col["kind"].startswith("blob"):
                blob_cols.add(col["name"])

        for i, row in enumerate(self._current_rows):
            self._meta_table.setRowCount(i + 1)
            for j, col_name in enumerate(all_cols):
                val = row.get(col_name)
                if col_name in blob_cols:
                    blob_key = self._blob_display_key(row, col_name)
                    item = QtWidgets.QTableWidgetItem(blob_key)
                    item.setForeground(QtCore.Qt.GlobalColor.blue)
                    item.setToolTip("Click row to view this blob in the array viewer")
                elif isinstance(val, (bytes, memoryview)):
                    n = len(val)
                    item = QtWidgets.QTableWidgetItem(f"<blob {n} bytes>")
                elif val is None:
                    item = QtWidgets.QTableWidgetItem("")
                elif isinstance(val, float):
                    item = QtWidgets.QTableWidgetItem(f"{val:.6g}")
                else:
                    item = QtWidgets.QTableWidgetItem(str(val))
                self._meta_table.setItem(i, j, item)

        self._meta_table.resizeColumnsToContents()
        self._row_count_lbl.setText(f"{len(self._current_rows)} rows")

    def _blob_display_key(self, row: Dict[str, Any], col_name: str) -> str:
        """Build a display string for a blob column."""
        col_info = next((c for c in self._plottable_cols if c["name"] == col_name), None)
        if col_info and col_info["shape"]:
            shape_str = "×".join(str(s) for s in col_info["shape"])
            return f"float64[{shape_str}]"
        val = row.get(col_name)
        if isinstance(val, (bytes, memoryview)):
            return f"<blob {len(val)} bytes>"
        return "<blob ?>"

    def _on_meta_row_selected(self):
        """When a row is selected in the metadata table, show the first blob column in the array viewer."""
        row_idx = self._meta_table.currentRow()
        if row_idx < 0 or row_idx >= len(self._current_rows):
            return
        row = self._current_rows[row_idx]

        # Find first blob column
        blob_col = None
        for col in self._plottable_cols:
            if col["kind"].startswith("blob"):
                blob_col = col
                break

        if blob_col is None:
            self._array_viewer.show_array(None)
            return

        cache_key = f"{row_idx}_{blob_col['name']}"
        if cache_key in self._blob_data_cache:
            arr = self._blob_data_cache[cache_key]
        else:
            pk_col = list(row.keys())[0]
            arr = deserialize_blob_to_array(
                self._gpkg_path, self._table_name, blob_col["name"],
                **{pk_col: row[pk_col]},
            )
            if arr is not None:
                self._blob_data_cache[cache_key] = arr

        if arr is not None:
            self._array_viewer.show_array(arr, blob_col["name"])
        else:
            self._array_viewer.show_array(None)

    def _apply_filter(self):
        self._load_rows()

    def _clear_filter(self):
        self._filter_col_combo.setCurrentIndex(0)
        self._filter_op_combo.setCurrentText("=")
        self._filter_value_edit.clear()
        self._load_rows()

    def _export_csv(self):
        """Export currently filtered rows to CSV."""
        if not self._current_rows:
            QtWidgets.QMessageBox.warning(self, "Export CSV", "No data to export")
            return
        filepath, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Table to CSV", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not filepath:
            return
        filter_col = self._filter_col_combo.currentText()
        filter_op = self._filter_op_combo.currentText()
        filter_val = self._filter_value_edit.text().strip()
        try:
            export_table_to_csv(
                self._gpkg_path, self._table_name, filepath,
                column=filter_col if filter_col and filter_op and filter_val else None,
                op=filter_op if filter_col and filter_op and filter_val else None,
                value=filter_val if filter_col and filter_op and filter_val else None,
            )
            QtWidgets.QMessageBox.information(self, "Export CSV", f"Data exported to {filepath}")
        except (ValueError, OSError) as exc:
            QtWidgets.QMessageBox.warning(self, "Export CSV", str(exc))

    def _send_to_plot(self):
        """Emit signal to switch explorer to plot tab for this table."""
        self.plot_requested.emit(self._gpkg_path, self._table_name)
        QtWidgets.QMessageBox.information(
            self, "Send to Plot",
            f"Table '{self._table_name}' data sent to the Explorer Plot tab.",
        )
```

- [ ] **Step 2: Commit**

```bash
git add swe2d/workbench/dialogs/sqlite_preview_dialog.py
git commit -m "feat: rewrite preview dialog with enhanced viewer, filter, array inspection"
```


### Task 5: Explorer dialog modifications — add QTabWidget, plot tab, wiring

**Files:**
- Modify: `swe2d/workbench/dialogs/gpkg_explorer_dialog.py`

**Dependencies:** Task 1, Task 3

- [ ] **Step 1: Modify explorer dialog to wrap in QTabWidget and add plot tab**

Replace the layout of `SWE2DModelGeoPackageExplorerDialog.__init__` to:

1. Wrap the table list in a QTabWidget as "Tables" tab
2. Add a "Plot" tab using `GpkgPlotTab`
3. Add wiring for enhanced preview dialog (instead of old one)
4. Add CSV export button to the tables tab

```python
# In __init__, after self.resize(980, 660):
# Replace the root VBoxLayout content

# At top of file, add imports:
from swe2d.workbench.dialogs.sqlite_preview_dialog import SWE2DEnhancedTablePreviewDialog
from swe2d.workbench.dialogs.gpkg_plot_tab import GpkgPlotTab
from swe2d.workbench.services.numpy_blob_service import export_table_to_csv

# In __init__, replace from line ~47 to ~93 with:

self._tabs = QtWidgets.QTabWidget()

# ── Tables tab (existing content) ───────────────────────────
tables_tab = QtWidgets.QWidget()
tables_layout = QtWidgets.QVBoxLayout(tables_tab)
tables_layout.setContentsMargins(0, 0, 0, 0)

self.source_lbl = QtWidgets.QLabel(f"GeoPackage: {self._gpkg_path}")
self.source_lbl.setWordWrap(True)
tables_layout.addWidget(self.source_lbl)

self.table = QtWidgets.QTableWidget()
self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
self.table.setAlternatingRowColors(True)
self.table.setColumnCount(4)
self.table.setHorizontalHeaderLabels(["Table", "Rows", "Type", "Actions"])
self.table.horizontalHeader().setStretchLastSection(True)
tables_layout.addWidget(self.table, stretch=1)

row = QtWidgets.QHBoxLayout()
self.refresh_btn = QtWidgets.QPushButton("Refresh")
self.refresh_btn.setToolTip("Reload the table listing from the GeoPackage.")
self.open_btn = QtWidgets.QPushButton("Open Viewer")
self.open_btn.setToolTip("Open the enhanced viewer for the selected table.")
self.rename_btn = QtWidgets.QPushButton("Rename Table")
self.rename_btn.setToolTip("Rename the selected model table (swe2d_* tables only).")
self.delete_btn = QtWidgets.QPushButton("Delete Table")
self.delete_btn.setToolTip("Permanently delete the selected table from the GeoPackage.")
self.delete_run_btn = QtWidgets.QPushButton("Delete by Run ID")
self.delete_run_btn.setToolTip("Delete all result tables associated with a specific run ID.")
self.export_csv_btn = QtWidgets.QPushButton("Export CSV")
self.export_csv_btn.setToolTip("Export selected table to CSV.")
for btn in (self.refresh_btn, self.open_btn, self.rename_btn, self.delete_btn, self.delete_run_btn, self.export_csv_btn):
    row.addWidget(btn)
row.addStretch(1)
tables_layout.addLayout(row)

self._tabs.addTab(tables_tab, "Tables")

# ── Plot tab ────────────────────────────────────────────────
self._plot_tab = GpkgPlotTab()
self._tabs.addTab(self._plot_tab, "Plot")

root = QtWidgets.QVBoxLayout(self)
root.addWidget(self._tabs)

# Wire buttons (same as before plus new ones):
self.refresh_btn.clicked.connect(self.refresh_tables)
self.open_btn.clicked.connect(self.open_selected)
self.rename_btn.clicked.connect(self.rename_selected)
self.delete_btn.clicked.connect(self.delete_selected)
self.delete_run_btn.clicked.connect(self._delete_by_run_id)
self.export_csv_btn.clicked.connect(self._export_selected_csv)
self.table.itemSelectionChanged.connect(self._sync_button_state)
self.table.itemDoubleClicked.connect(lambda _item: self.open_selected())

# Remove the old QDialogButtonBox and add it at the end of root
buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
buttons.rejected.connect(self.reject)
buttons.accepted.connect(self.accept)
root.addWidget(buttons)

self.refresh_tables()
```

Also add the `_export_selected_csv` method and update `open_selected` and `preview_selected` to use the enhanced dialog:

```python
def _export_selected_csv(self):
    name = self._selected_table()
    if not name:
        return
    filepath, _ = QtWidgets.QFileDialog.getSaveFileName(
        self, f"Export {name} to CSV", "", "CSV Files (*.csv);;All Files (*)"
    )
    if not filepath:
        return
    try:
        from swe2d.workbench.services.numpy_blob_service import export_table_to_csv
        export_table_to_csv(self._gpkg_path, name, filepath)
        self._log(f"Exported {name} to {filepath}")
    except (ValueError, OSError) as exc:
        QtWidgets.QMessageBox.warning(self, "Export CSV", str(exc))

def open_selected(self):
    name = self._selected_table()
    if not name:
        return
    kind = self._table_kind(name)
    if kind == "run_log":
        self._open_run_log_viewer()
        return
    if kind == "line_results":
        self._open_line_results_viewer()
        return
    if kind == "config":
        from swe2d.workbench.dialogs.simulation_config_viewer_dialog import SWE2DSimulationConfigViewerDialog
        dlg = SWE2DSimulationConfigViewerDialog(self._gpkg_path, parent=self)
        dlg.exec()
        return
    # Use enhanced viewer for everything else
    self._open_enhanced_viewer(name)

def _open_enhanced_viewer(self, name: str, title: str = ""):
    dlg = SWE2DEnhancedTablePreviewDialog(
        self._gpkg_path, name,
        title=title or f"Table Viewer - {name}",
        parent=self,
    )
    dlg.plot_requested.connect(self._switch_to_plot_tab)
    dlg.exec()

def _switch_to_plot_tab(self, gpkg_path: str, table: str):
    self._plot_tab.set_table(gpkg_path, table)
    self._tabs.setCurrentWidget(self._plot_tab)

def preview_selected(self):
    name = self._selected_table()
    if not name:
        return
    self._open_enhanced_viewer(name)
```

Remove the old `_open_preview` method — it's replaced by `_open_enhanced_viewer`.

Also keep the `refresh_tables` method the same — it still fills the table list.

- [ ] **Step 2: Commit**

```bash
git add swe2d/workbench/dialogs/gpkg_explorer_dialog.py
git commit -m "refactor: add QTabWidget and plot tab to GPKG explorer"
```


### Task 6: Wire controller entry points — update menu/button to support plot handoff

**Files:**
- Modify: `swe2d/workbench/controllers/topology_controller.py` (minor, lines 826–855)
- Modify: `swe2d/workbench/views/map_tab_view.py` (no change needed probably)

- [ ] **Step 1: Verify controller wiring — ensure no breaking changes**

Check that `TopologyController.open_model_gpkg_explorer()` still works — it creates the dialog and calls `dlg.exec()`. The changes to the dialog are internal, so this should work unchanged.

- [ ] **Step 2: Update imports in test files that reference the old dialog**

```bash
# Verify existing tests still import
mamba run -n qgis_stable python3 -c "
from swe2d.workbench.dialogs.gpkg_explorer_dialog import SWE2DModelGeoPackageExplorerDialog
from swe2d.workbench.dialogs.sqlite_preview_dialog import SWE2DEnhancedTablePreviewDialog
print('Imports OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add .
git commit -m "fix: update imports for enhanced GPKG preview dialog"
```


### Task 7: Final integration verification

**Dependencies:** All previous tasks

- [ ] **Step 1: Verify all existing tests pass**

```bash
mamba run -n qgis_stable python3 -m pytest tests/test_gpkg_operations.py tests/test_numpy_blob_service.py tests/test_results_path_audit_fixes.py -v
```

- [ ] **Step 2: Verify the dialog creates without QGIS (import test)**

```bash
mamba run -n qgis_stable python3 -c "
from qgis.PyQt.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])
from swe2d.workbench.dialogs.gpkg_explorer_dialog import SWE2DModelGeoPackageExplorerDialog
from swe2d.workbench.dialogs.sqlite_preview_dialog import SWE2DEnhancedTablePreviewDialog
from swe2d.workbench.dialogs.gpkg_plot_tab import GpkgPlotTab
from swe2d.workbench.dialogs.gpkg_array_viewer_widget import ArrayViewerWidget
print('All dialogs import OK')
"
```

- [ ] **Step 3: Purge pycache and commit final**

```bash
find . -type d -name __pycache__ -exec rm -rf {} +
git add -A
git commit -m "feat: complete GPKG explorer enhanced viewer with blob display, filtering, XY plots, CSV export"
```
