"""Pure-Python, Qt-free timestep service for SWE2D results.

Provides timestep loading, union computation, and time/frame conversion
without any Qt dependency.  Testable without QApplication.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from swe2d.results.db_utils import open_ro as _open_ro
from swe2d.results.queries import (
    _find_prefixed_or_default_table,
    _resolve_ts_table,
)


# ---------------------------------------------------------------------------
# Timestep loading
# ---------------------------------------------------------------------------

def load_timesteps(gpkg_path: str, run_id: str) -> np.ndarray:
    """Load all distinct timesteps for *run_id* across all lines.

    Returns a sorted float64 array, or an empty array on any error.
    """
    conn = _open_ro(gpkg_path, row_factory=None)
    if conn is None:
        return np.empty(0, dtype=np.float64)
    try:
        table, shared = _resolve_ts_table(conn, str(run_id))
        if not table:
            return np.empty(0, dtype=np.float64)
        if shared:
            cur = conn.execute(
                f"SELECT DISTINCT t_s FROM \"{table}\" WHERE run_id=? ORDER BY t_s",
                (str(run_id),),
            )
        else:
            cur = conn.execute(
                f"SELECT DISTINCT t_s FROM \"{table}\" ORDER BY t_s"
            )
        vals = [float(r[0]) for r in cur.fetchall()]
        return np.asarray(vals, dtype=np.float64)
    except Exception:
        return np.empty(0, dtype=np.float64)
    finally:
        conn.close()


def load_line_timesteps(gpkg_path: str, line_id: int) -> np.ndarray:
    """Load all distinct timesteps for *line_id* across all runs in the GPKG.

    For the shared schema this queries the shared table filtering by
    *line_id*.  For legacy per-run tables it iterates over all
    discoverable run tables and unions their timesteps for the given
    line.

    Returns a sorted float64 array, or an empty array on any error.
    """
    conn = _open_ro(gpkg_path, row_factory=None)
    if conn is None:
        return np.empty(0, dtype=np.float64)
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND (name LIKE 'swe2d_line_results_ts' "
            "OR name LIKE 'swe2d_line_results_ts_%') "
            "ORDER BY name"
        )
        tables = [str(r[0]) for r in cur.fetchall()]
        if not tables:
            return np.empty(0, dtype=np.float64)
        all_ts: List[float] = []
        for tbl in tables:
            cur2 = conn.execute(
                f"SELECT DISTINCT t_s FROM \"{tbl}\" WHERE line_id=? ORDER BY t_s",
                (int(line_id),),
            )
            for row in cur2.fetchall():
                all_ts.append(float(row[0]))
        if not all_ts:
            return np.empty(0, dtype=np.float64)
        return np.asarray(sorted(set(all_ts)), dtype=np.float64)
    except Exception:
        return np.empty(0, dtype=np.float64)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Union
# ---------------------------------------------------------------------------

def compute_timestep_union(timestep_sets: List[np.ndarray]) -> np.ndarray:
    """Union of multiple timestep arrays (unique sorted).

    Returns an empty float64 array when *timestep_sets* is empty.
    """
    if not timestep_sets:
        return np.empty(0, dtype=np.float64)
    return np.unique(np.concatenate(timestep_sets))


# ---------------------------------------------------------------------------
# Time <-> frame conversion
# ---------------------------------------------------------------------------

def time_sec_to_frame_idx(t_sec: float, timesteps: np.ndarray) -> int:
    """Convert time in seconds to nearest frame index.

    Returns 0 when *timesteps* is empty.
    """
    if timesteps.size == 0:
        return 0
    return int(np.argmin(np.abs(timesteps - float(t_sec))))


def frame_idx_to_time_sec(idx: int, timesteps: np.ndarray) -> float:
    """Convert frame index to time in seconds.

    Returns 0.0 when *timesteps* is empty.
    Clamps *idx* to valid range [0, n-1].
    """
    n = timesteps.size
    if n == 0:
        return 0.0
    return float(timesteps[max(0, min(idx, n - 1))])


# ---------------------------------------------------------------------------
# Coupling loading
# ---------------------------------------------------------------------------

def load_coupling_for_run(gpkg_path: str, run_id: str) -> List[Dict]:
    """Load coupling data for *run_id* from *gpkg_path*.

    Returns a list of dicts with keys ``t_s``, ``component``, ``metric``,
    ``object_id``, ``object_name``, ``value``.  Returns an empty list on
    any error or when no coupling table exists.
    """
    conn = _open_ro(gpkg_path, row_factory=None)
    if conn is None:
        return []
    try:
        ct = _find_prefixed_or_default_table(conn, "swe2d_coupling_results")
        if not ct:
            return []
        cur = conn.execute(
            f'SELECT t_s, component, metric, object_id, object_name, value '
            f'FROM "{ct}" WHERE run_id = ? ORDER BY t_s',
            (str(run_id),),
        )
        cols = ["t_s", "component", "metric", "object_id", "object_name", "value"]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    except Exception:
        return []
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Baked-aware wrappers
# ---------------------------------------------------------------------------

def load_baked_timesteps(gpkg_path: str, run_id: str) -> np.ndarray:
    """Load timesteps from baked_results BLOB (delegates to persistence service)."""
    from swe2d.services.gpkg_persistence_service import load_baked_timesteps as _baked
    return _baked(gpkg_path, run_id)


def load_baked_coupling_for_run(gpkg_path: str, run_id: str) -> List[Dict]:
    """Load coupling records from baked_coupling table."""
    import sqlite3
    try:
        conn = sqlite3.connect(f"file:{gpkg_path}?mode=ro", uri=True)
        rows = conn.execute(
            "SELECT component, object_id, object_name, metric, n_timesteps "
            "FROM swe2d_baked_coupling WHERE run_id=?",
            (run_id,),
        ).fetchall()
        return [
            {"component": r[0], "object_id": r[1],
             "object_name": r[2], "metric": r[3],
             "n_timesteps": r[4]}
            for r in rows
        ]
    except Exception:
        return []
    finally:
        try:
            conn.close()
        except Exception:
            pass
