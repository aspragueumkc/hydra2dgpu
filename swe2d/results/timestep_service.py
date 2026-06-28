"""Pure-Python, Qt-free timestep service for SWE2D results.

Provides timestep loading, union computation, and time/frame conversion
without any Qt dependency.  Testable without QApplication.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from swe2d.results.db_utils import open_ro as _open_ro


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
