"""Pure Python, zero-Qt GPKG export service for SWE2D results.

Provides creation, persistence, query, and deletion of SWE2D simulation
results stored in OGC GeoPackage format.  No Qt dependency — testable
without QApplication.
"""

from __future__ import annotations

import datetime
import os
import sqlite3
from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# GPKG creation
# ---------------------------------------------------------------------------

def create_results_gpkg(gpkg_path: str, crs_wkt: str) -> None:
    """Create a results GeoPackage with all required OGC and SWE2D tables.

    Creates the OGC standard metadata tables (``gpkg_contents``,
    ``spatial_ref_sys``, ``gpkg_geometry_columns``) and all SWE2D result
    tables (mesh results, line results, runs metadata).  Idempotent — safe
    to call on an existing file.

    Raises ``ValueError`` if *gpkg_path* is empty.
    """
    if not gpkg_path:
        raise ValueError("gpkg_path must not be empty")

    conn = sqlite3.connect(gpkg_path)
    try:
        cur = conn.cursor()

        _ensure_ogc_tables(cur, crs_wkt)
        _ensure_swe2d_mesh_tables(cur)
        _ensure_swe2d_line_tables(cur)

        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def persist_run_results(
    gpkg_path: str,
    run_id: str,
    h: np.ndarray,
    hu: np.ndarray,
    hv: np.ndarray,
    interval_s: float = 60.0,
    t_s_values: Optional[np.ndarray] = None,
) -> str:
    """Persist mesh simulation results to a GPKG.

    Parameters
    ----------
    gpkg_path : str
        Path to an existing results GPKG (created via :func:`create_results_gpkg`).
    run_id : str
        Run identifier.  If empty or ``None``, one is auto-generated.
    h, hu, hv : np.ndarray
        1-D float64 arrays of cell-centred depth and momentum components.
        All three must have the same length (number of cells × number of
        timesteps if *t_s_values* is provided).
    interval_s : float
        Output interval in model seconds (default 60).
    t_s_values : np.ndarray, optional
        Per-timestep time values.  If ``None``, a single timestep ``[0.0]``
        is used and the arrays are treated as one snapshot.

    Returns
    -------
    str
        The *run_id* (auto-generated or user-supplied).
    """
    if not run_id:
        run_id = _auto_run_id()

    if t_s_values is None:
        t_s_values = np.array([0.0], dtype=np.float64)

    n_steps = len(t_s_values)
    n_cells = len(h) // n_steps

    rows: List[Dict[str, object]] = []
    for i_step in range(n_steps):
        ts = float(t_s_values[i_step])
        start = i_step * n_cells
        end = start + n_cells
        for cell_id in range(n_cells):
            idx = start + cell_id
            rows.append({
                "t_s": ts,
                "cell_id": cell_id,
                "h": float(h[idx]) if idx < len(h) else 0.0,
                "hu": float(hu[idx]) if idx < len(hu) else 0.0,
                "hv": float(hv[idx]) if idx < len(hv) else 0.0,
            })

    conn = sqlite3.connect(gpkg_path)
    try:
        cur = conn.cursor()
        _write_mesh_run(cur, run_id, rows, interval_s)
        conn.commit()
    finally:
        conn.close()

    return run_id


def persist_line_results(
    gpkg_path: str,
    run_id: str,
    line_data: Dict[str, Any],
) -> None:
    """Persist line-sampling results to a GPKG.

    *line_data* must contain at least ``ts_rows`` (list of dicts) and may
    optionally contain ``profile_rows``, ``mesh_interval_s``, and
    ``line_interval_s``.

    Expected keys in each ``ts_rows`` dict:
        ``t_s``, ``line_id``, ``line_name``, ``depth_m``, ``velocity_ms``,
        ``wse_m``, ``bed_m``, ``flow_cms``, ``wet_frac``, ``fr``

    Expected keys in each ``profile_rows`` dict:
        ``t_s``, ``line_id``, ``line_name``, ``station_m``, ``depth_m``,
        ``velocity_ms``, ``wse_m``, ``bed_m``, ``flow_qn``, ``wet``, ``fr``
    """
    if not gpkg_path or not run_id:
        return

    ts_rows = list(line_data.get("ts_rows", []))
    profile_rows = list(line_data.get("profile_rows", []))
    mesh_interval_s = float(line_data.get("mesh_interval_s", 60.0))
    line_interval_s = float(line_data.get("line_interval_s", 60.0))

    conn = sqlite3.connect(gpkg_path)
    try:
        cur = conn.cursor()
        _write_line_run(
            cur, run_id, ts_rows, profile_rows,
            mesh_interval_s, line_interval_s,
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


def _quote_ident(name: str) -> str:
    """Quote a SQLite identifier (table/column name) safely."""
    return '"' + str(name).replace('"', '""') + '"'


def get_table_info(gpkg_path: str, table_name: str) -> list[str]:
    """Return column names for *table_name* in the given GPKG.

    Returns an empty list if the table does not exist.
    """
    conn = sqlite3.connect(gpkg_path)
    try:
        cur = conn.cursor()
        cur.execute(f"PRAGMA table_info({_quote_ident(table_name)})")
        return [str(r[1]) for r in cur.fetchall()]
    except Exception:
        return []
    finally:
        conn.close()


def get_table_contents(gpkg_path: str, table_name: str, limit: int = 250) -> list[tuple]:
    """Return up to *limit* rows from *table_name* in the given GPKG.

    Returns an empty list if the table does not exist.
    """
    conn = sqlite3.connect(gpkg_path)
    try:
        cur = conn.cursor()
        cur.execute(
            f"SELECT * FROM {_quote_ident(table_name)} LIMIT ?",
            (int(limit),),
        )
        return list(cur.fetchall())
    except Exception:
        return []
    finally:
        conn.close()


def get_run_metadata(gpkg_path: str, run_id: str) -> Dict[str, Any]:
    """Query run metadata from a GPKG.

    Returns a dict with keys from the runs table (``run_id``, ``created_utc``,
    ``interval_s``, ``row_count``) or an empty dict if the run is not found.
    """
    conn = sqlite3.connect(gpkg_path)
    try:
        cur = conn.cursor()
        for runs_table in ("swe2d_mesh_results_runs", "swe2d_line_results_runs"):
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (runs_table,),
            )
            if cur.fetchone() is None:
                continue
            cur.execute(
                f"SELECT * FROM \"{runs_table}\" WHERE run_id=?",
                (run_id,),
            )
            row = cur.fetchone()
            if row is None:
                continue
            col_names = [str(d[0]) for d in cur.description]
            return dict(zip(col_names, row))
    except Exception:
        return {}
    finally:
        conn.close()

    return {}


def list_runs_in_gpkg(gpkg_path: str) -> List[Dict[str, Any]]:
    """List all runs stored in a GPKG.

    Returns a list of dicts with keys ``run_id``, ``created_utc``,
    ``interval_s``, ``row_count``.  Returns an empty list if the file
    contains no runs.
    """
    out: List[Dict[str, Any]] = []
    conn = sqlite3.connect(gpkg_path)
    try:
        cur = conn.cursor()
        for runs_table in ("swe2d_mesh_results_runs", "swe2d_line_results_runs"):
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (runs_table,),
            )
            if cur.fetchone() is None:
                continue
            cur.execute(
                f"SELECT * FROM \"{runs_table}\" ORDER BY rowid"
            )
            col_names = [str(d[0]) for d in cur.description]
            for row in cur.fetchall():
                rec = dict(zip(col_names, row))
                if str(rec.get("run_id", "")).strip():
                    out.append(rec)
    except Exception:
        return []
    finally:
        conn.close()

    return out


def load_mesh_snapshot(
    gpkg_path: str,
    run_id: str,
    t_s: float,
) -> Optional[Dict[str, Any]]:
    """Load mesh snapshot data (h, hu, hv arrays) for a given run at the nearest timestep.

    Returns a dict with keys:
        ``h``, ``hu``, ``hv`` (np.ndarray of cell-centred values),
        ``t_s`` (float, the actual timestep loaded — nearest match),
        ``cell_count`` (int, number of cells).

    Returns ``None`` if the GPKG is missing, the mesh results table does not
    exist, or no rows match the requested run/timestep.
    """
    if not gpkg_path or not os.path.exists(gpkg_path):
        return None
    try:
        conn = sqlite3.connect(gpkg_path)
        cur = conn.cursor()

        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND (name LIKE 'swe2d_mesh_results' "
            "OR name LIKE '%_swe2d_mesh_results') ORDER BY name LIMIT 1"
        )
        row = cur.fetchone()
        if not row:
            conn.close()
            return None
        table_name = str(row[0])

        cur.execute(
            f'SELECT t_s FROM "{table_name}" '
            "WHERE run_id = ? ORDER BY ABS(t_s - ?) LIMIT 1",
            (run_id, float(t_s)),
        )
        row = cur.fetchone()
        if not row:
            conn.close()
            return None
        nearest_ts = float(row[0])

        cur.execute(
            f'SELECT h, hu, hv FROM "{table_name}" '
            "WHERE run_id = ? AND t_s = ? ORDER BY cell_id",
            (run_id, nearest_ts),
        )
        rows = cur.fetchall()
        conn.close()
        if not rows:
            return None

        h = np.asarray([float(r[0]) for r in rows], dtype=np.float64)
        hu = np.asarray([float(r[1]) for r in rows], dtype=np.float64)
        hv = np.asarray([float(r[2]) for r in rows], dtype=np.float64)
        if h.size == 0:
            return None

        return {
            "h": h,
            "hu": hu,
            "hv": hv,
            "t_s": nearest_ts,
            "cell_count": int(h.size),
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------

def delete_run_from_gpkg(gpkg_path: str, run_id: str) -> None:
    """Delete a run and all its associated data from a GPKG.

    Removes entries from ``swe2d_mesh_results``, ``swe2d_mesh_results_runs``,
    ``swe2d_line_results_ts``, ``swe2d_line_results_runs``,
    ``swe2d_line_results_profile`` for the given *run_id*.

    Safe to call for run IDs that do not exist (no-op).
    """
    conn = sqlite3.connect(gpkg_path)
    try:
        cur = conn.cursor()
        _delete_run(cur, run_id, "swe2d_mesh_results", "swe2d_mesh_results_runs")
        _delete_run(cur, run_id, "swe2d_line_results_ts", "swe2d_line_results_runs", extra_tables=("swe2d_line_results_profile",))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ===================================================================
# Internal helpers
# ===================================================================

def _auto_run_id() -> str:
    """auto run id."""
    now = datetime.datetime.now().astimezone()
    ts = now.strftime("%Y%m%dT%H%M%S") + now.strftime("%z")
    return f"swe2d_{ts}"


def _q(name: str) -> str:
    """q."""
    return '"' + str(name).replace('"', '""') + '"'


def _ensure_ogc_tables(cur: sqlite3.Cursor, crs_wkt: str) -> None:
    """ensure ogc tables."""
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='spatial_ref_sys'"
    )
    if cur.fetchone() is not None:
        return

    cur.execute(
        """CREATE TABLE spatial_ref_sys (
            srs_id INTEGER PRIMARY KEY,
            srs_name TEXT NOT NULL,
            srs_type TEXT NOT NULL,
            organization TEXT NOT NULL,
            organization_coordsys_id INTEGER NOT NULL,
            definition TEXT NOT NULL,
            description TEXT
        )"""
    )
    cur.execute(
        "INSERT INTO spatial_ref_sys(srs_id, srs_name, srs_type, organization, "
        "organization_coordsys_id, definition, description) "
        "VALUES(?, ?, ?, ?, ?, ?, ?)",
        (4326, "WGS 84 geodetic", "geodetic", "EPSG", 4326, str(crs_wkt), "WGS 84"),
    )

    cur.execute(
        """CREATE TABLE gpkg_contents (
            table_name TEXT NOT NULL PRIMARY KEY,
            data_type TEXT NOT NULL,
            identifier TEXT,
            description TEXT DEFAULT '',
            last_change DATETIME NOT NULL,
            min_x DOUBLE,
            min_y DOUBLE,
            max_x DOUBLE,
            max_y DOUBLE,
            srs_id INTEGER,
            CONSTRAINT fk_gpkg_contents_srs FOREIGN KEY (srs_id) REFERENCES spatial_ref_sys(srs_id)
        )"""
    )

    cur.execute(
        """CREATE TABLE gpkg_geometry_columns (
            table_name TEXT NOT NULL,
            column_name TEXT NOT NULL,
            geometry_type_name TEXT NOT NULL,
            srs_id INTEGER NOT NULL,
            z TINYINT NOT NULL,
            m TINYINT NOT NULL,
            CONSTRAINT pk_geom_cols PRIMARY KEY (table_name, column_name),
            CONSTRAINT fk_gpkg_geometry_columns_srs FOREIGN KEY (srs_id) REFERENCES spatial_ref_sys(srs_id)
        )"""
    )

    cur.execute(
        "INSERT INTO gpkg_contents(table_name, data_type, identifier, last_change, srs_id) "
        "VALUES('spatial_ref_sys', 'attributes', 'spatial_ref_sys', ?, ?)",
        (datetime.datetime.now().astimezone().isoformat(), 4326),
    )
    cur.execute(
        "INSERT INTO gpkg_contents(table_name, data_type, identifier, last_change, srs_id) "
        "VALUES('gpkg_contents', 'attributes', 'gpkg_contents', ?, ?)",
        (datetime.datetime.now().astimezone().isoformat(), 4326),
    )
    cur.execute(
        "INSERT INTO gpkg_contents(table_name, data_type, identifier, last_change, srs_id) "
        "VALUES('gpkg_geometry_columns', 'attributes', 'gpkg_geometry_columns', ?, ?)",
        (datetime.datetime.now().astimezone().isoformat(), 4326),
    )


def _ensure_swe2d_mesh_tables(cur: sqlite3.Cursor) -> None:
    """ensure swe2d mesh tables."""
    cur.execute(
        f"""CREATE TABLE IF NOT EXISTS {_q("swe2d_mesh_results_runs")} (
            run_id TEXT PRIMARY KEY,
            created_utc TEXT,
            interval_s REAL,
            row_count INTEGER
        )"""
    )
    cur.execute(
        f"""CREATE TABLE IF NOT EXISTS {_q("swe2d_mesh_results")} (
            run_id TEXT,
            t_s REAL,
            cell_id INTEGER,
            h REAL,
            hu REAL,
            hv REAL,
            PRIMARY KEY (run_id, t_s, cell_id)
        )"""
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_swe2d_mesh_results_run_t_cell "
        f"ON {_q('swe2d_mesh_results')}(run_id, t_s, cell_id)"
    )


def _ensure_swe2d_line_tables(cur: sqlite3.Cursor) -> None:
    """ensure swe2d line tables."""
    cur.execute(
        f"""CREATE TABLE IF NOT EXISTS {_q("swe2d_line_results_runs")} (
            run_id TEXT PRIMARY KEY,
            created_utc TEXT,
            mesh_interval_s REAL,
            line_interval_s REAL,
            row_count INTEGER
        )"""
    )
    cur.execute(
        f"""CREATE TABLE IF NOT EXISTS {_q("swe2d_line_results_ts")} (
            run_id TEXT,
            t_s REAL,
            line_id INTEGER,
            line_name TEXT,
            depth_m REAL,
            velocity_ms REAL,
            wse_m REAL,
            bed_m REAL,
            flow_cms REAL,
            wet_frac REAL,
            fr REAL,
            PRIMARY KEY (run_id, t_s, line_id)
        )"""
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_swe2d_line_results_ts_run_line_t "
        f"ON {_q('swe2d_line_results_ts')}(run_id, line_id, t_s)"
    )
    cur.execute(
        f"""CREATE TABLE IF NOT EXISTS {_q("swe2d_line_results_profile")} (
            run_id TEXT,
            t_s REAL,
            line_id INTEGER,
            line_name TEXT,
            station_m REAL,
            depth_m REAL,
            velocity_ms REAL,
            wse_m REAL,
            bed_m REAL,
            flow_qn REAL,
            wet INTEGER,
            fr REAL,
            PRIMARY KEY (run_id, t_s, line_id, station_m)
        )"""
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_swe2d_line_results_profile_run_line_t_s "
        f"ON {_q('swe2d_line_results_profile')}(run_id, line_id, t_s, station_m)"
    )


def _write_mesh_run(
    cur: sqlite3.Cursor,
    run_id: str,
    rows: List[Dict[str, object]],
    interval_s: float,
) -> None:
    """write mesh run."""
    q_table = _q("swe2d_mesh_results")
    q_runs = _q("swe2d_mesh_results_runs")

    cur.execute(f"DELETE FROM {q_table} WHERE run_id = ?", (run_id,))
    cur.execute(
        f"""INSERT OR REPLACE INTO {q_runs}
            (run_id, created_utc, interval_s, row_count)
            VALUES (?, ?, ?, ?)""",
        (
            run_id,
            datetime.datetime.now().astimezone().replace(microsecond=0).isoformat(),
            float(interval_s),
            int(len(rows)),
        ),
    )
    batch = [
        (run_id, float(r["t_s"]), int(r["cell_id"]),
         float(r["h"]), float(r["hu"]), float(r["hv"]))
        for r in rows
    ]
    cur.executemany(
        f"""INSERT OR REPLACE INTO {q_table}
            (run_id, t_s, cell_id, h, hu, hv)
            VALUES (?, ?, ?, ?, ?, ?)""",
        batch,
    )


def _write_line_run(
    cur: sqlite3.Cursor,
    run_id: str,
    ts_rows: List[Dict[str, object]],
    profile_rows: List[Dict[str, object]],
    mesh_interval_s: float,
    line_interval_s: float,
) -> None:
    """write line run."""
    q_runs = _q("swe2d_line_results_runs")
    q_ts = _q("swe2d_line_results_ts")
    q_profile = _q("swe2d_line_results_profile")

    cur.execute(f"DELETE FROM {q_ts} WHERE run_id = ?", (run_id,))
    cur.execute(f"DELETE FROM {q_profile} WHERE run_id = ?", (run_id,))
    cur.execute(
        f"""INSERT OR REPLACE INTO {q_runs}
            (run_id, created_utc, mesh_interval_s, line_interval_s, row_count)
            VALUES (?, ?, ?, ?, ?)""",
        (
            run_id,
            datetime.datetime.now().astimezone().replace(microsecond=0).isoformat(),
            float(mesh_interval_s),
            float(line_interval_s),
            int(len(ts_rows)),
        ),
    )

    if ts_rows:
        ts_batch = [
            (
                run_id,
                float(r.get("t_s", 0.0)),
                int(r.get("line_id", -1)),
                str(r.get("line_name", "") or ""),
                float(r.get("depth_m", float("nan"))),
                float(r.get("velocity_ms", float("nan"))),
                float(r.get("wse_m", float("nan"))),
                float(r.get("bed_m", float("nan"))),
                float(r.get("flow_cms", float("nan"))),
                float(r.get("wet_frac", float("nan"))),
                float(r.get("fr", float("nan"))),
            )
            for r in ts_rows
        ]
        cur.executemany(
            f"""INSERT OR REPLACE INTO {q_ts}
                (run_id, t_s, line_id, line_name, depth_m, velocity_ms,
                 wse_m, bed_m, flow_cms, wet_frac, fr)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ts_batch,
        )

    if profile_rows:
        prof_batch = [
            (
                run_id,
                float(r.get("t_s", 0.0)),
                int(r.get("line_id", -1)),
                str(r.get("line_name", "") or ""),
                float(r.get("station_m", 0.0)),
                float(r.get("depth_m", float("nan"))),
                float(r.get("velocity_ms", float("nan"))),
                float(r.get("wse_m", float("nan"))),
                float(r.get("bed_m", float("nan"))),
                float(r.get("flow_qn", float("nan"))),
                int(r.get("wet", 0)),
                float(r.get("fr", float("nan"))),
            )
            for r in profile_rows
        ]
        cur.executemany(
            f"""INSERT OR REPLACE INTO {q_profile}
                (run_id, t_s, line_id, line_name, station_m, depth_m,
                 velocity_ms, wse_m, bed_m, flow_qn, wet, fr)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            prof_batch,
        )


def _delete_run(
    cur: sqlite3.Cursor,
    run_id: str,
    data_table: str,
    runs_table: str,
    extra_tables: tuple = (),
) -> None:
    """delete run."""
    for tbl in (data_table,) + extra_tables:
        cur.execute(f"DELETE FROM {_q(tbl)} WHERE run_id = ?", (run_id,))
    cur.execute(f"DELETE FROM {_q(runs_table)} WHERE run_id = ?", (run_id,))
