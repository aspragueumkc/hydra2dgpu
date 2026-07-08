from __future__ import annotations

import csv
import sqlite3
from typing import Dict, List, Tuple


def load_graphs(gpkg_path: str) -> Dict[str, Dict]:
    """Load all hyetographs and hydrographs from GPKG.

    Returns:
        {"hyetographs": {id: {"data": [(t,v),...], "value_type": ..., "units": ...}},
         "hydrographs": {id: {"data": [(t,v),...], "bc_type": ..., "description": ...}}}
    """
    hyetographs: Dict[str, Dict] = {}
    hydrographs: Dict[str, Dict] = {}

    conn = sqlite3.connect(gpkg_path)
    try:
        for row in conn.execute(
            "SELECT hyetograph_id, Time, Value, value_type, units "
            "FROM swe2d_hyetographs ORDER BY hyetograph_id, Time"
        ):
            hid, t, v, vt, u = row
            if hid not in hyetographs:
                hyetographs[hid] = {"data": [], "value_type": vt or "", "units": u or ""}
            hyetographs[hid]["data"].append((float(t), float(v)))

        for row in conn.execute(
            "SELECT hydrograph_id, Time, Value, bc_type, description "
            "FROM swe2d_hydrographs ORDER BY hydrograph_id, Time"
        ):
            hid, t, v, bt, desc = row
            if hid not in hydrographs:
                hydrographs[hid] = {"data": [], "bc_type": bt, "description": desc or ""}
            hydrographs[hid]["data"].append((float(t), float(v)))
    finally:
        conn.close()

    return {"hyetographs": hyetographs, "hydrographs": hydrographs}


def save_hyetograph(
    gpkg_path: str, hid: str, data: List[Tuple[float, float]],
    value_type: str = "", units: str = "",
) -> None:
    """Save a hyetograph, replacing any existing data for this ID."""
    conn = sqlite3.connect(gpkg_path)
    try:
        conn.execute("DELETE FROM swe2d_hyetographs WHERE hyetograph_id = ?", (hid,))
        conn.executemany(
            "INSERT INTO swe2d_hyetographs (hyetograph_id, Time, Value, value_type, units) "
            "VALUES (?, ?, ?, ?, ?)",
            [(hid, t, v, value_type, units) for t, v in data],
        )
        conn.commit()
    finally:
        conn.close()


def save_hydrograph(
    gpkg_path: str, hid: str, data: List[Tuple[float, float]],
    bc_type: int = 0, description: str = "",
) -> None:
    """Save a hydrograph, replacing any existing data for this ID."""
    conn = sqlite3.connect(gpkg_path)
    try:
        conn.execute("DELETE FROM swe2d_hydrographs WHERE hydrograph_id = ?", (hid,))
        conn.executemany(
            "INSERT INTO swe2d_hydrographs (hydrograph_id, bc_type, Time, Value, description) "
            "VALUES (?, ?, ?, ?, ?)",
            [(hid, bc_type, t, v, description) for t, v in data],
        )
        conn.commit()
    finally:
        conn.close()


def delete_graph(gpkg_path: str, table: str, hid: str) -> None:
    """Delete all rows for a given ID from the specified table."""
    id_field = "hyetograph_id" if table == "swe2d_hyetographs" else "hydrograph_id"
    conn = sqlite3.connect(gpkg_path)
    try:
        conn.execute(f"DELETE FROM {table} WHERE {id_field} = ?", (hid,))
        conn.commit()
    finally:
        conn.close()


def list_graph_ids(gpkg_path: str) -> Dict[str, List[str]]:
    """List all distinct graph IDs grouped by type."""
    hyeto: List[str] = []
    hydro: List[str] = []
    conn = sqlite3.connect(gpkg_path)
    try:
        hyeto = [
            r[0] for r in conn.execute(
                "SELECT DISTINCT hyetograph_id FROM swe2d_hyetographs ORDER BY hyetograph_id"
            ) if r[0]
        ]
        hydro = [
            r[0] for r in conn.execute(
                "SELECT DISTINCT hydrograph_id FROM swe2d_hydrographs ORDER BY hydrograph_id"
            ) if r[0]
        ]
    finally:
        conn.close()
    return {"hyetographs": hyeto, "hydrographs": hydro}


def csv_columns(csv_path: str) -> list[str]:
    """Read CSV header and return column names."""
    import csv as _csv
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = _csv.DictReader(f)
        return reader.fieldnames or []


def parse_csv(csv_path: str, time_col: str, value_col: str) -> List[Tuple[float, float]]:
    """Parse a CSV file, extracting time/value from specified columns."""
    data: List[Tuple[float, float]] = []
    import csv as _csv
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = _csv.DictReader(f)
        if not reader.fieldnames:
            return data
        if time_col not in reader.fieldnames or value_col not in reader.fieldnames:
            return data
        for row in reader:
            try:
                data.append((float(row[time_col]), float(row[value_col])))
            except (ValueError, TypeError):
                continue
    return data
