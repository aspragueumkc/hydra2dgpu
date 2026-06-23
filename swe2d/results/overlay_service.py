"""Pure-Python, Qt-free overlay data service for SWE2D results.

Provides coupling time-series data preparation extracted from
swe2d/results/panel.py.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np


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
