"""swe2d_results_queries.py

Pure-Python, Qt-free data layer for reading SWE2D line results stored in a
GeoPackage (SQLite).  All functions return empty containers on error rather
than raising, so callers never need try/except just to check availability.

Schema assumed in the GeoPackage
---------------------------------
swe2d_line_results_ts_{run_id}
    t_s REAL, line_id INTEGER, line_name TEXT,
    depth_m REAL, velocity_ms REAL, wse_m REAL, bed_m REAL, flow_cms REAL

swe2d_line_results_profile_{run_id}
    t_s REAL, line_id INTEGER, line_name TEXT, station_m REAL,
    depth_m REAL, velocity_ms REAL, wse_m REAL, bed_m REAL,
    flow_qn REAL, fr REAL, wet INTEGER
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from swe2d.results.db_utils import open_ro as _open_ro_shared, table_exists as _table_exists_shared
except Exception:
    from .db_utils import open_ro as _open_ro_shared, table_exists as _table_exists_shared

# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

@dataclass
class ResultsDataset:
    """Lightweight descriptor for a single run + line combination."""
    run_id: str
    gpkg_path: str
    line_id: int
    line_name: str = ""
    timesteps: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    color: Tuple[int, int, int] = (31, 119, 180)   # matplotlib blue
    label: str = ""                                  # display label; falls back to run_id

    def display_label(self) -> str:
        return self.label or self.run_id


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _open_ro(gpkg_path: str) -> Optional[sqlite3.Connection]:
    """Open the GPKG read-only; return None on failure."""
    return _open_ro_shared(gpkg_path, row_factory=sqlite3.Row)


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    return _table_exists_shared(conn, table_name)


def _resolve_ts_table(conn: sqlite3.Connection, run_id: str) -> Tuple[str, bool]:
    """Return (table_name, uses_run_id_column) for timeseries storage."""
    if _table_exists(conn, "swe2d_line_results_ts"):
        return "swe2d_line_results_ts", True
    legacy = f"swe2d_line_results_ts_{run_id}"
    if _table_exists(conn, legacy):
        return legacy, False
    return "", False


def _resolve_profile_table(conn: sqlite3.Connection, run_id: str) -> Tuple[str, bool]:
    """Return (table_name, uses_run_id_column) for profile storage."""
    if _table_exists(conn, "swe2d_line_results_profile"):
        return "swe2d_line_results_profile", True
    legacy = f"swe2d_line_results_profile_{run_id}"
    if _table_exists(conn, legacy):
        return legacy, False
    return "", False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def discover_line_result_runs(gpkg_path: str) -> List[Dict]:
    """Return metadata for every run stored in *gpkg_path*.

    Returns a list of dicts:
        {
            "run_id":      str,
            "table_ts":    str,   # e.g. "swe2d_line_results_ts_20260512_abc"
            "table_profile": str, # e.g. "swe2d_line_results_profile_20260512_abc"
            "has_profile": bool,
        }

    Returns an empty list if the file does not exist, is not a valid SQLite
    database, or contains no matching tables.
    """
    conn = _open_ro(gpkg_path)
    if conn is None:
        return []
    try:
        results: List[Dict] = []

        # New shared-schema support: swe2d_line_results_ts/profile with run_id columns.
        if _table_exists(conn, "swe2d_line_results_ts"):
            run_ids: List[str] = []
            if _table_exists(conn, "swe2d_line_results_runs"):
                cur = conn.execute(
                    "SELECT run_id FROM swe2d_line_results_runs "
                    "ORDER BY datetime(created_utc) DESC, rowid DESC"
                )
                run_ids = [str(r[0]) for r in cur.fetchall() if str(r[0] or "").strip()]
            if not run_ids:
                cur = conn.execute(
                    "SELECT DISTINCT run_id FROM swe2d_line_results_ts ORDER BY run_id"
                )
                run_ids = [str(r[0]) for r in cur.fetchall() if str(r[0] or "").strip()]

            has_profile_table = _table_exists(conn, "swe2d_line_results_profile")
            for run_id in run_ids:
                has_profile = False
                if has_profile_table:
                    cur = conn.execute(
                        "SELECT 1 FROM swe2d_line_results_profile WHERE run_id=? LIMIT 1",
                        (run_id,),
                    )
                    has_profile = cur.fetchone() is not None
                results.append(
                    {
                        "run_id": run_id,
                        "table_ts": "swe2d_line_results_ts",
                        "table_profile": "swe2d_line_results_profile",
                        "has_profile": has_profile,
                    }
                )

        # Legacy per-run table support: swe2d_line_results_ts_<run_id>.
        cur = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name LIKE 'swe2d_line_results_ts_%' "
            "ORDER BY name"
        )
        rows = cur.fetchall()
        existing = {str(r.get("run_id", "")) for r in results}
        for row in rows:
            table_ts = str(row[0])
            prefix = "swe2d_line_results_ts_"
            if not table_ts.startswith(prefix):
                continue
            run_id = table_ts[len(prefix):]
            if run_id in existing:
                continue
            table_profile = f"swe2d_line_results_profile_{run_id}"
            has_profile = _table_exists(conn, table_profile)
            results.append(
                {
                    "run_id": run_id,
                    "table_ts": table_ts,
                    "table_profile": table_profile,
                    "has_profile": has_profile,
                }
            )
        return results
    except Exception:
        return []
    finally:
        conn.close()


def load_line_ids(gpkg_path: str, run_id: str) -> List[Tuple[int, str]]:
    """Return ``[(line_id, line_name), ...]`` sorted by *line_id*.

    Returns an empty list on any error.
    """
    conn = _open_ro(gpkg_path)
    if conn is None:
        return []
    try:
        table, shared = _resolve_ts_table(conn, str(run_id))
        if not table:
            return []
        if shared:
            cur = conn.execute(
                f"SELECT DISTINCT line_id, line_name FROM \"{table}\" WHERE run_id=? ORDER BY line_id",
                (str(run_id),),
            )
        else:
            cur = conn.execute(
                f"SELECT DISTINCT line_id, line_name FROM \"{table}\" ORDER BY line_id"
            )
        return [(int(r[0]), str(r[1] or "")) for r in cur.fetchall()]
    except Exception:
        return []
    finally:
        conn.close()


def load_timesteps(gpkg_path: str, run_id: str, line_id: int) -> np.ndarray:
    """Return sorted array of ``t_s`` values for *(run_id, line_id)*.

    Returns an empty float64 array on any error.
    """
    conn = _open_ro(gpkg_path)
    if conn is None:
        return np.empty(0, dtype=np.float64)
    try:
        table, shared = _resolve_ts_table(conn, str(run_id))
        if not table:
            return np.empty(0, dtype=np.float64)
        if shared:
            cur = conn.execute(
                f"SELECT DISTINCT t_s FROM \"{table}\" WHERE run_id=? AND line_id=? ORDER BY t_s",
                (str(run_id), int(line_id)),
            )
        else:
            cur = conn.execute(
                f"SELECT DISTINCT t_s FROM \"{table}\" WHERE line_id=? ORDER BY t_s",
                (int(line_id),),
            )
        vals = [float(r[0]) for r in cur.fetchall()]
        return np.asarray(vals, dtype=np.float64)
    except Exception:
        return np.empty(0, dtype=np.float64)
    finally:
        conn.close()


def load_timeseries(gpkg_path: str, run_id: str, line_id: int) -> Dict[str, np.ndarray]:
    """Load the full time-series record for *(run_id, line_id)*.

    Returns a dict with keys:
        ``t_s``, ``depth_m``, ``velocity_ms``, ``wse_m``, ``bed_m``, ``flow_cms``
    Each value is a 1-D float64 numpy array of equal length, sorted by *t_s*.

    Returns an empty dict on any error or if no rows match.
    """
    conn = _open_ro(gpkg_path)
    if conn is None:
        return {}
    try:
        table, shared = _resolve_ts_table(conn, str(run_id))
        if not table:
            return {}
        if shared:
            cur = conn.execute(
                f"SELECT t_s, depth_m, velocity_ms, wse_m, bed_m, flow_cms "
                f"FROM \"{table}\" WHERE run_id=? AND line_id=? ORDER BY t_s",
                (str(run_id), int(line_id)),
            )
        else:
            cur = conn.execute(
                f"SELECT t_s, depth_m, velocity_ms, wse_m, bed_m, flow_cms "
                f"FROM \"{table}\" WHERE line_id=? ORDER BY t_s",
                (int(line_id),),
            )
        rows = cur.fetchall()
        if not rows:
            return {}
        t_s, depth, vel, wse, bed, flow = zip(*[(
            float(r[0]), float(r[1]), float(r[2]),
            float(r[3]), float(r[4]), float(r[5]),
        ) for r in rows])
        return {
            "t_s":         np.asarray(t_s,    dtype=np.float64),
            "depth_m":     np.asarray(depth,  dtype=np.float64),
            "velocity_ms": np.asarray(vel,    dtype=np.float64),
            "wse_m":       np.asarray(wse,    dtype=np.float64),
            "bed_m":       np.asarray(bed,    dtype=np.float64),
            "flow_cms":    np.asarray(flow,   dtype=np.float64),
        }
    except Exception:
        return {}
    finally:
        conn.close()


def load_profile(
    gpkg_path: str,
    run_id: str,
    line_id: int,
    t_sec: float,
    t_tol: float = 0.5,
) -> Dict[str, np.ndarray]:
    """Load the cross-section profile for *(run_id, line_id)* at time *t_sec*.

    Uses ``ABS(t_s - t_sec) < t_tol`` tolerance for the timestamp comparison.

    Returns a dict with keys:
        ``station_m``, ``depth_m``, ``velocity_ms``, ``wse_m``, ``bed_m``,
        ``flow_qn``, ``fr``, ``wet``
    Returns an empty dict on any error or if the profile table does not exist.
    """
    conn = _open_ro(gpkg_path)
    if conn is None:
        return {}
    try:
        table, shared = _resolve_profile_table(conn, str(run_id))
        if not table:
            return {}
        if shared:
            cur = conn.execute(
                f"SELECT station_m, depth_m, velocity_ms, wse_m, bed_m, flow_qn, fr, wet "
                f"FROM \"{table}\" "
                f"WHERE run_id=? AND line_id=? AND ABS(t_s - ?) < ? "
                f"ORDER BY station_m",
                (str(run_id), int(line_id), float(t_sec), float(t_tol)),
            )
        else:
            cur = conn.execute(
                f"SELECT station_m, depth_m, velocity_ms, wse_m, bed_m, flow_qn, fr, wet "
                f"FROM \"{table}\" "
                f"WHERE line_id=? AND ABS(t_s - ?) < ? "
                f"ORDER BY station_m",
                (int(line_id), float(t_sec), float(t_tol)),
            )
        rows = cur.fetchall()
        if not rows:
            return {}
        station, depth, vel, wse, bed, qn, fr, wet = zip(*[(
            float(r[0]), float(r[1]), float(r[2]), float(r[3]),
            float(r[4]), float(r[5]), float(r[6]), float(r[7]),
        ) for r in rows])
        return {
            "station_m":   np.asarray(station, dtype=np.float64),
            "depth_m":     np.asarray(depth,   dtype=np.float64),
            "velocity_ms": np.asarray(vel,     dtype=np.float64),
            "wse_m":       np.asarray(wse,     dtype=np.float64),
            "bed_m":       np.asarray(bed,     dtype=np.float64),
            "flow_qn":     np.asarray(qn,      dtype=np.float64),
            "fr":          np.asarray(fr,       dtype=np.float64),
            "wet":         np.asarray(wet,      dtype=np.float64),
        }
    except Exception:
        return {}
    finally:
        conn.close()


def find_nearest_timestep(
    gpkg_path: str,
    run_id: str,
    line_id: int,
    t_sec: float,
) -> float:
    """Return the stored timestep value nearest to *t_sec*.

    Returns *t_sec* unchanged if no timesteps are available.
    """
    ts = load_timesteps(gpkg_path, run_id, line_id)
    if ts.size == 0:
        return float(t_sec)
    idx = int(np.argmin(np.abs(ts - float(t_sec))))
    return float(ts[idx])


def load_structure_flows_at_time(
    gpkg_path: str,
    run_id: str,
    t_sec: float,
    t_tol: float = 0.5,
) -> List[Dict[str, object]]:
    """Return structure-flow rows near *t_sec* from coupling results.

    Each row has keys: ``object_id``, ``object_name``, ``value``, ``t_s``.
    Returns an empty list when coupling tables are missing or no rows match.
    """
    conn = _open_ro(gpkg_path)
    if conn is None:
        return []
    try:
        if not _table_exists(conn, "swe2d_coupling_results"):
            return []
        cur = conn.execute(
            """
            SELECT object_id, object_name, value, t_s
            FROM swe2d_coupling_results
            WHERE run_id = ?
              AND component = 'structure'
              AND metric = 'flow'
              AND ABS(t_s - ?) < ?
            ORDER BY object_id
            """,
            (str(run_id), float(t_sec), float(t_tol)),
        )
        out: List[Dict[str, object]] = []
        for object_id, object_name, value, ts in cur.fetchall():
            out.append(
                {
                    "object_id": str(object_id or ""),
                    "object_name": str(object_name or ""),
                    "value": float(value),
                    "t_s": float(ts),
                }
            )
        return out
    except Exception:
        return []
    finally:
        conn.close()
