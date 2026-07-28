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


def _coerce_numeric(value: Any) -> Any:
    """Coerce a numeric string to int/float; return the original value otherwise."""
    if not isinstance(value, str):
        return value
    try:
        return int(value)
    except (ValueError, TypeError):
        pass
    try:
        return float(value)
    except (ValueError, TypeError):
        pass
    return value


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
    if op in ("=", "!=", ">", "<", ">=", "<="):
        return f"{col_quoted} {op} ?", [_coerce_numeric(value)]
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

            if limit and int(limit) > 0:
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


def combine_1d_arrays(arrays_by_name: Dict[str, np.ndarray]) -> Tuple[List[str], Optional[np.ndarray]]:
    """Combine a mapping of column-name -> 1D ``np.ndarray`` into a single 2D
    array suitable for table display.

    Args:
        arrays_by_name: Map of display labels (e.g. ``"times"``, ``"depth"``)
            to 1D numpy arrays. Values are cast to ``float64``.

    Returns:
        ``(column_labels, combined_2d_array)`` where ``column_labels`` is the
        input key order and ``combined_2d_array`` has shape
        ``(n_rows, n_cols)``, or ``([name], array)`` of the first input if
        shapes are inconsistent, or ``([], None)`` if the input is empty.
    """
    arrs = [(name, arr) for name, arr in arrays_by_name.items() if arr is not None]
    if not arrs:
        return [], None

    column_labels = [name for name, _ in arrs]
    arrays = [arr for _, arr in arrs]

    if not all(a.ndim == 1 for a in arrays):
        return [column_labels[0]], np.asarray(arrays[0], dtype=np.float64)

    lengths = {len(a) for a in arrays}
    if len(lengths) != 1:
        return [column_labels[0]], np.asarray(arrays[0], dtype=np.float64)

    n = arrays[0].shape[0]
    combined = np.column_stack([np.asarray(a, dtype=np.float64).reshape(n) for a in arrays])
    return column_labels, combined
