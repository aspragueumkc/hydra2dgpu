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

import logging
import sqlite3
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from swe2d.results.data import SWE2DResultsData

try:
    from qgis.core import QgsVectorLayer
except ImportError:
    QgsVectorLayer = None  # ponytail: QGIS layer queries degrade gracefully

from swe2d.results.db_utils import open_ro as _open_ro_shared, table_exists as _table_exists_shared

logger = logging.getLogger(__name__)

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
        """display label."""
        return self.label or self.run_id


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _open_ro(gpkg_path: str) -> Optional[sqlite3.Connection]:
    """Open the GPKG read-only; return None on failure."""
    return _open_ro_shared(gpkg_path, row_factory=sqlite3.Row)


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """table exists."""
    return _table_exists_shared(conn, table_name)


def _find_prefixed_or_default_table(conn: sqlite3.Connection, base_name: str) -> str:
    """find prefixed or default table."""
    base = str(base_name or "").strip()
    if not base:
        return ""
    if _table_exists(conn, base):
        return base
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ? ORDER BY name",
            (f"%_{base}",),
        )
        for row in cur.fetchall():
            nm = str(row[0] or "").strip()
            if nm.endswith("_" + base):
                return nm
    except Exception:
        return ""
    return ""


def _find_all_prefixed_or_default_tables(conn: sqlite3.Connection, base_name: str) -> List[str]:
    """Return all matching tables for a shared base name.

    Includes the exact base table (if present) plus any prefixed variants like
    ``<prefix>_<base_name>``.
    """
    base = str(base_name or "").strip()
    if not base:
        return []
    out: List[str] = []
    seen = set()
    if _table_exists(conn, base):
        out.append(base)
        seen.add(base)
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ? ORDER BY name",
            (f"%_{base}",),
        )
        for row in cur.fetchall():
            nm = str(row[0] or "").strip()
            if not nm or nm in seen:
                continue
            if nm.endswith("_" + base):
                out.append(nm)
                seen.add(nm)
    except Exception:
        return out
    return out


def _resolve_ts_table(conn: sqlite3.Connection, run_id: str) -> Tuple[str, bool]:
    """Return (table_name, uses_run_id_column) for timeseries storage."""
    shared = _find_prefixed_or_default_table(conn, "swe2d_line_results_ts")
    if shared:
        return shared, True
    legacy = f"swe2d_line_results_ts_{run_id}"
    if _table_exists(conn, legacy):
        return legacy, False
    return "", False


def _resolve_profile_table(conn: sqlite3.Connection, run_id: str) -> Tuple[str, bool]:
    """Return (table_name, uses_run_id_column) for profile storage."""
    shared = _find_prefixed_or_default_table(conn, "swe2d_line_results_profile")
    if shared:
        return shared, True
    legacy = f"swe2d_line_results_profile_{run_id}"
    if _table_exists(conn, legacy):
        return legacy, False
    return "", False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def discover_line_result_runs(gpkg_path: str) -> List[Dict]:
    """Return metadata for every discoverable SWE2D run stored in *gpkg_path*.

    Returns a list of dicts:
        {
            "run_id":      str,
            "table_ts":    str,   # e.g. "swe2d_line_results_ts_20260512_abc"
            "table_profile": str, # e.g. "swe2d_line_results_profile_20260512_abc"
            "has_profile": bool,
        }

    Discovery is line-results-first (shared or legacy schemas), with a mesh
    results fallback so mesh-only snapshot runs still appear in the multi-run
    results panel.

    Returns an empty list if the file does not exist, is not a valid SQLite
    database, or contains no matching SWE2D result tables.
    """
    conn = _open_ro(gpkg_path)
    if conn is None:
        return []
    try:
        results: List[Dict] = []

        def _upsert_run(run_id: str, table_ts: str, table_profile: str, has_profile: bool) -> None:
            """upsert run."""
            rid = str(run_id or "").strip()
            if not rid:
                return
            for rec in results:
                if str(rec.get("run_id", "")) != rid:
                    continue
                # Prefer entries that have line/profile context; merge flags.
                if table_ts and str(rec.get("table_ts", "")) != table_ts:
                    rec["table_ts"] = str(table_ts)
                if table_profile and not str(rec.get("table_profile", "")):
                    rec["table_profile"] = str(table_profile)
                rec["has_profile"] = bool(rec.get("has_profile", False) or bool(has_profile))
                return
            results.append(
                {
                    "run_id": rid,
                    "table_ts": str(table_ts or ""),
                    "table_profile": str(table_profile or ""),
                    "has_profile": bool(has_profile),
                }
            )

        # New shared-schema support: swe2d_line_results_ts/profile with run_id columns.
        # Scan all prefixed variants to avoid silently missing runs when multiple
        # results-table prefixes exist in one GeoPackage.
        shared_ts_tables = _find_all_prefixed_or_default_tables(conn, "swe2d_line_results_ts")
        for shared_ts in shared_ts_tables:
            run_ids: List[str] = []
            shared_runs = f"{shared_ts}_runs"
            if not _table_exists(conn, shared_runs):
                # Legacy naming fallback when table names are not strict companions.
                shared_runs = _find_prefixed_or_default_table(conn, "swe2d_line_results_runs")
            if shared_runs:
                cur = conn.execute(
                    f"SELECT run_id FROM \"{shared_runs}\" "
                    "ORDER BY datetime(created_utc) DESC, rowid DESC"
                )
                run_ids = [str(r[0]) for r in cur.fetchall() if str(r[0] or "").strip()]
            if not run_ids:
                cur = conn.execute(
                    f"SELECT DISTINCT run_id FROM \"{shared_ts}\" ORDER BY run_id"
                )
                run_ids = [str(r[0]) for r in cur.fetchall() if str(r[0] or "").strip()]

            shared_profile = shared_ts.replace("_line_results_ts", "_line_results_profile")
            if not _table_exists(conn, shared_profile):
                shared_profile = _find_prefixed_or_default_table(conn, "swe2d_line_results_profile")
            has_profile_table = bool(shared_profile)
            for run_id in run_ids:
                has_profile = False
                if has_profile_table:
                    cur = conn.execute(
                        f"SELECT 1 FROM \"{shared_profile}\" WHERE run_id=? LIMIT 1",
                        (run_id,),
                    )
                    has_profile = cur.fetchone() is not None
                _upsert_run(run_id, shared_ts, shared_profile or "", has_profile)

        # Legacy per-run table support: swe2d_line_results_ts_<run_id>.
        cur = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name LIKE 'swe2d_line_results_ts_%' "
            "ORDER BY name"
        )
        rows = cur.fetchall()
        for row in rows:
            table_ts = str(row[0])
            prefix = "swe2d_line_results_ts_"
            if not table_ts.startswith(prefix):
                continue
            run_id = table_ts[len(prefix):]
            table_profile = f"swe2d_line_results_profile_{run_id}"
            has_profile = _table_exists(conn, table_profile)
            _upsert_run(run_id, table_ts, table_profile, has_profile)

        # Mesh-results fallback for snapshot runs where line sampling is absent.
        mesh_runs_tables = _find_all_prefixed_or_default_tables(conn, "swe2d_mesh_results_runs")
        for runs_table in mesh_runs_tables:
            mesh_table = runs_table[:-5] if runs_table.endswith("_runs") else ""
            if not mesh_table or not _table_exists(conn, mesh_table):
                continue
            cur = conn.execute(
                f"SELECT run_id FROM \"{runs_table}\" ORDER BY datetime(created_utc) DESC, rowid DESC"
            )
            for row in cur.fetchall():
                rid = str(row[0] or "").strip()
                if rid:
                    _upsert_run(rid, mesh_table, "", False)

        mesh_tables = _find_all_prefixed_or_default_tables(conn, "swe2d_mesh_results")
        for mesh_table in mesh_tables:
            cur = conn.execute(
                f"SELECT DISTINCT run_id FROM \"{mesh_table}\" WHERE run_id IS NOT NULL ORDER BY run_id"
            )
            for row in cur.fetchall():
                rid = str(row[0] or "").strip()
                if rid:
                    _upsert_run(rid, mesh_table, "", False)

        # Max-results discovery: register runs stored in swe2d_mesh_max_results.
        max_runs_tables = _find_all_prefixed_or_default_tables(conn, "swe2d_mesh_max_results_runs")
        for runs_table in max_runs_tables:
            cur = conn.execute(
                f"SELECT run_id FROM \"{runs_table}\" ORDER BY rowid DESC"
            )
            for row in cur.fetchall():
                rid = str(row[0] or "").strip()
                if rid:
                    _upsert_run(rid, "swe2d_mesh_max_results", "", False)

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


def load_timeseries_from_live(
    data: "SWE2DResultsData", run_id: str, line_id: int
) -> Dict[str, np.ndarray]:
    """Load time-series from in-memory snapshots during a live run.

    Returns the same dict shape as load_timeseries():
        ``t_s``, ``depth_m``, ``velocity_ms``, ``wse_m``, ``bed_m``, ``flow_cms``
    Each value is a 1-D float64 numpy array, sorted by *t_s*.

    Returns an empty dict if no matching data is found.
    """
    rows = data.get_live_line_snapshot_rows()
    if not rows or line_id < 0:
        return {}
    matched = [r for r in rows if int(r.get("line_id", -1)) == line_id]
    if not matched:
        return {}
    matched.sort(key=lambda r: float(r.get("t_s", 0.0)))
    out: Dict[str, list] = {}
    keys = ["t_s", "depth_m", "velocity_ms", "wse_m", "bed_m", "flow_cms"]
    for k in keys:
        out[k] = []
    for r in matched:
        for k in keys:
            out[k].append(float(r.get(k, 0.0)))
    return {k: np.array(v, dtype=np.float64) for k, v in out.items()}


def load_profile_from_live(
    data: "SWE2DResultsData", run_id: str, line_id: int, t_sec: float
) -> Dict[str, np.ndarray]:
    """Load profile from in-memory snapshots during a live run.

    Handles station-per-row format where each row stores a single
    station along the profile line with ``station_m``, ``wse_m``,
    ``bed_m``, ``depth_m`` keys (the format produced by
    :meth:`_sample_line_metrics`).

    Returns the same dict shape as load_profile():
        ``dist_m``, ``wse_m``, ``bed_m``, ``depth_m``

    Returns an empty dict if no matching data is found.
    """
    rows = data.get_live_line_profile_rows()
    if not rows or line_id < 0:
        return {}
    # Filter by line_id (live profile rows don't store run_id)
    matched = [r for r in rows if int(r.get("line_id", -1)) == line_id]
    if not matched:
        return {}
    # Group by t_s and pick the timestep closest to t_sec
    from collections import defaultdict
    by_ts: dict = defaultdict(list)
    for r in matched:
        ts = float(r.get("t_s", 0.0))
        by_ts[ts].append(r)
    if not by_ts:
        return {}
    best_ts = min(by_ts.keys(), key=lambda ts: abs(ts - t_sec))
    stations = by_ts[best_ts]
    # Sort by station_m and extract profile arrays
    stations.sort(key=lambda r: float(r.get("station_m", 0.0)))
    dist = np.array([float(r.get("station_m", 0.0)) for r in stations], dtype=np.float64)
    wse  = np.array([float(r.get("wse_m", float("nan"))) for r in stations], dtype=np.float64)
    bed  = np.array([float(r.get("bed_m", float("nan"))) for r in stations], dtype=np.float64)
    depth = np.array([float(r.get("depth_m", float("nan"))) for r in stations], dtype=np.float64)
    return {"dist_m": dist, "wse_m": wse, "bed_m": bed, "depth_m": depth}


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
        coupling_table = _find_prefixed_or_default_table(conn, "swe2d_coupling_results")
        if not coupling_table:
            return []
        cur = conn.execute(
            f"""
            SELECT object_id, object_name, value, t_s
            FROM "{coupling_table}"
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

