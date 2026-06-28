"""swe2d_results_queries.py

Pure-Python, Qt-free data layer for reading SWE2D results from baked GPKG tables.
All functions delegate to swe2d.services.gpkg_persistence_service.load_baked_*
and return empty containers on error.

Baked tables: swe2d_baked_results, swe2d_baked_line_ts, swe2d_baked_coupling
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from swe2d.results.data import SWE2DResultsData

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
    color: Tuple[int, int, int] = (31, 119, 180)
    label: str = ""

    def display_label(self) -> str:
        """display label."""
        return self.label or self.run_id


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _conn(gpkg_path: str):
    """Open read-only connection, or None."""
    if not gpkg_path:
        return None
    try:
        c = sqlite3.connect(f"file:{gpkg_path}?mode=ro", uri=True)
        c.row_factory = sqlite3.Row
        return c
    except Exception:
        return None


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (name,),
        )
        return cur.fetchone() is not None
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Public API — delegates to baked GPKG persistence service
# ---------------------------------------------------------------------------

def discover_line_result_runs(gpkg_path: str) -> List[Dict]:
    """Return metadata for every discoverable run in a GPKG from baked tables.

    Delegates to collect_baked_runs_from_gpkg. Returns [] if the GPKG
    does not contain baked result tables (clean-sheet — no legacy fallback).
    """
    from swe2d.services.gpkg_persistence_service import collect_baked_runs_from_gpkg
    raw = collect_baked_runs_from_gpkg(gpkg_path)
    out = []
    for r in raw:
        out.append({
            "run_id": r["run_id"],
            "table_ts": "swe2d_baked_line_ts",
            "table_profile": "swe2d_baked_line_profiles",
            "has_profile": r.get("has_lines", False),
        })
    return out


def load_timeseries(gpkg_path: str, run_id: str, line_id: int) -> dict:
    """Load line timeseries via baked delegate."""
    from swe2d.services.gpkg_persistence_service import load_baked_line_timeseries
    return load_baked_line_timeseries(gpkg_path, run_id, line_id)


def load_timeseries_from_live(data: SWE2DResultsData, run_id: str, line_id: int) -> dict:
    """Load line timeseries from live data via baked delegate (duck-typed)."""
    from swe2d.services.gpkg_persistence_service import load_baked_line_timeseries
    return load_baked_line_timeseries(data, run_id, line_id)


def load_profile(gpkg_path: str, run_id: str, line_id: int, t_s: float) -> dict:
    """Load line profile at time via baked delegate."""
    from swe2d.services.gpkg_persistence_service import load_baked_line_profile
    return load_baked_line_profile(gpkg_path, run_id, line_id, t_s)


def load_profile_from_live(data: SWE2DResultsData, run_id: str, line_id: int, t_s: float) -> dict:
    """Load line profile from live data via baked delegate."""
    from swe2d.services.gpkg_persistence_service import load_baked_line_profile
    return load_baked_line_profile(data, run_id, line_id, t_s)


def find_nearest_timestep(gpkg_path: str, run_id: str, line_id: int, t_sec: float) -> float:
    """Find nearest timestep for a run via baked times blob."""
    from swe2d.services.gpkg_persistence_service import load_baked_timesteps
    times = load_baked_timesteps(gpkg_path, run_id)
    if times.size == 0:
        return t_sec
    return float(times[int(np.argmin(np.abs(times - t_sec)))])


def load_structure_flows_at_time(gpkg_path: str, run_id: str, t_sec: float) -> List[Dict]:
    """Load structure flow values from baked coupling at nearest time."""
    from swe2d.services.gpkg_persistence_service import load_baked_coupling_timeseries
    # Probe for structure/metrics from the baked coupling table
    conn = _conn(gpkg_path)
    if conn is None:
        return []
    try:
        rows = conn.execute(
            "SELECT DISTINCT component, object_id, object_name, metric "
            "FROM swe2d_baked_coupling WHERE run_id=? AND component='structure'",
            (run_id,),
        ).fetchall()
        results = []
        for r in rows:
            times, values = load_baked_coupling_timeseries(
                gpkg_path, run_id, str(r[0]), str(r[1]), str(r[3]),
            )
            if times is not None and times.size > 0:
                i = int(np.argmin(np.abs(times - t_sec)))
                results.append({
                    "component": str(r[0]),
                    "object_id": str(r[1]),
                    "object_name": str(r[2]),
                    "metric": str(r[3]),
                    "t_s": float(times[i]),
                    "value": float(values[i]),
                })
        return results
    except Exception:
        return []
    finally:
        conn.close()


def load_line_ids(gpkg_path: str, run_id: str) -> List[Tuple[int, str]]:
    """Load line IDs and names from baked_line_ts table."""
    conn = _conn(gpkg_path)
    if conn is None:
        return []
    try:
        if not _table_exists(conn, "swe2d_baked_line_ts"):
            return []
        cur = conn.execute(
            "SELECT DISTINCT line_id, line_name FROM swe2d_baked_line_ts "
            "WHERE run_id=? ORDER BY line_id",
            (run_id,),
        )
        return [(int(r[0]), str(r[1] or "")) for r in cur.fetchall()]
    except Exception:
        return []
    finally:
        conn.close()
