"""Pure-Python, Qt-free service for SWE2D profile data computation.

Provides profile record extraction and conversion to structured numpy
arrays without any Qt or matplotlib dependency.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def extract_profile_arrays(
    records: List[Dict],
    sort_by_station: bool = True,
) -> Dict[str, np.ndarray]:
    """Convert list of profile record dicts to structured numpy arrays.

    Parameters
    ----------
    records : list of dict
        Profile records with at minimum a ``station_m`` key.
    sort_by_station : bool
        If True (default), sort records by station_m ascending.

    Returns
    -------
    dict of ndarray
        Always contains keys ``station_m``, ``wse_m``, ``bed_m``,
        ``depth_m``, ``wet``, plus any extra numeric keys found in
        the first record.
    """
    if not records:
        base = {
            "station_m": np.empty(0, dtype=np.float64),
            "wse_m": np.empty(0, dtype=np.float64),
            "bed_m": np.empty(0, dtype=np.float64),
            "depth_m": np.empty(0, dtype=np.float64),
            "wet": np.empty(0, dtype=np.float64),
        }
        return base

    src = list(records)
    if sort_by_station:
        src.sort(key=lambda r: float(r.get("station_m", 0.0)))

    n = len(src)
    station = np.empty(n, dtype=np.float64)
    wse = np.empty(n, dtype=np.float64)
    bed = np.empty(n, dtype=np.float64)
    depth = np.empty(n, dtype=np.float64)
    wet = np.empty(n, dtype=np.float64)

    extra_keys: set = set()
    for i, rec in enumerate(src):
        station[i] = float(rec.get("station_m", 0.0))
        wse[i] = _safe_float(rec.get("wse_m"))
        bed[i] = _safe_float(rec.get("bed_m"))
        depth[i] = _safe_float(rec.get("depth_m"))
        wet[i] = _safe_float(rec.get("wet"))
        for k, v in rec.items():
            if k in ("station_m", "wse_m", "bed_m", "depth_m", "wet"):
                continue
            try:
                float(v)
                extra_keys.add(k)
            except (TypeError, ValueError):
                pass

    result: Dict[str, np.ndarray] = {
        "station_m": station,
        "wse_m": wse,
        "bed_m": bed,
        "depth_m": depth,
        "wet": wet,
    }
    for k in sorted(extra_keys):
        arr = np.empty(n, dtype=np.float64)
        for i, rec in enumerate(src):
            arr[i] = _safe_float(rec.get(k))
        result[k] = arr

    return result


def _safe_float(val) -> float:
    """safe float."""
    if val is None:
        return float("nan")
    try:
        return float(val)
    except (TypeError, ValueError):
        return float("nan")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
