"""Pure-Python, Qt-free overlay data service for SWE2D results.

Provides coupling time-series data preparation extracted from
swe2d/results/panel.py.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Field computations
# ---------------------------------------------------------------------------

def compute_velocity_magnitude(
    hu: np.ndarray, hv: np.ndarray, h: np.ndarray
) -> np.ndarray:
    """Compute velocity magnitude from discharge and depth arrays."""
    h_safe = np.maximum(np.abs(h), 1e-12)
    u = np.divide(hu, h_safe, out=np.zeros_like(hu), where=h_safe > 0)
    v = np.divide(hv, h_safe, out=np.zeros_like(hv), where=h_safe > 0)
    return np.sqrt(u * u + v * v)


def compute_wse(h: np.ndarray, bed: np.ndarray) -> np.ndarray:
    """Compute water surface elevation: bed + depth."""
    return bed + h


def compute_froude(
    hu: np.ndarray, hv: np.ndarray, h: np.ndarray, g: float = 9.81, h_min: float = 0.001
) -> np.ndarray:
    """Compute Froude number from discharge and depth arrays."""
    h_safe = np.maximum(np.abs(h), h_min)
    u = np.divide(hu, h_safe, out=np.zeros_like(hu), where=h_safe > 0)
    v = np.divide(hv, h_safe, out=np.zeros_like(hv), where=h_safe > 0)
    speed = np.sqrt(u * u + v * v)
    return speed / np.sqrt(g * h_safe)


# ---------------------------------------------------------------------------
# Coupling time-series data prep
# ---------------------------------------------------------------------------

def prepare_coupling_timeseries(
    records: List[Dict],
) -> Dict[str, Dict[str, object]]:
    """Group coupling records by *object_id* and prepare numpy arrays.

    Filters out records with non-finite ``value`` or ``t_s``, and
    converts time from seconds to hours.  Returns a dict mapping
    ``object_id`` to ``{"times": np.ndarray, "values": np.ndarray, "name": str}``.
    """
    by_object: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    names: Dict[str, str] = {}

    for rec in records:
        try:
            t_s = float(rec.get("t_s", 0.0))
            value = float(rec.get("value", float("nan")))
        except (ValueError, TypeError):
            continue
        if not np.isfinite(t_s) or not np.isfinite(value):
            continue
        oid = str(rec.get("object_id", "") or "")
        by_object[oid].append((t_s, value))
        names[oid] = str(rec.get("object_name", "") or "")

    result: Dict[str, Dict[str, object]] = {}
    for oid in by_object:
        pairs = sorted(by_object[oid], key=lambda x: x[0])
        times = np.asarray([p[0] / 3600.0 for p in pairs], dtype=np.float64)
        values = np.asarray([p[1] for p in pairs], dtype=np.float64)
        result[oid] = {"times": times, "values": values, "name": names.get(oid, "")}

    return result
