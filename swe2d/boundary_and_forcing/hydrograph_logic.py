from __future__ import annotations
import logging

logger = logging.getLogger(__name__)

from typing import Callable, List, Optional, Tuple

import numpy as np


def parse_time_hours(token: str) -> float:
    """
    parse time hours.

    Parameters
    ----------
    token : str
        Description of token.

    Returns
    -------
    float
    """
    t = str(token).strip()
    if not t:
        raise ValueError("empty time token")
    if ":" in t:
        parts = t.split(":")
        try:
            if len(parts) == 2:
                return float(parts[0]) + (float(parts[1]) / 60.0)
            if len(parts) == 3:
                return float(parts[0]) + (float(parts[1]) / 60.0) + (float(parts[2]) / 3600.0)
        except ValueError:
            pass
        raise ValueError(f"invalid HH:MM(:SS) token '{t}'")
    try:
        return float(t)
    except ValueError:
        raise ValueError(f"invalid time token '{t}' — use HH:MM(:SS) or decimal hours")


def parse_hydrograph_text(
    text: str,
    parse_time_hours_fn: Callable[[str], float] = parse_time_hours,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    parse hydrograph text.

    Parameters
    ----------
    text : str
        Description of text.
    parse_time_hours_fn : Callable[[str], float]
        Description of parse_time_hours_fn.

    Returns
    -------
    Optional[Tuple[np.ndarray, np.ndarray]]
    """
    raw = str(text or "").strip()
    if not raw:
        return None

    pairs: List[Tuple[float, float]] = []
    chunks = raw.replace("\n", ";").split(";")
    for chunk in chunks:
        c = chunk.strip()
        if not c:
            continue
        if "," in c:
            a, b = c.split(",", 1)
        elif "=" in c:
            a, b = c.split("=", 1)
        else:
            raise ValueError(f"hydrograph entry '{c}' must use ',' or '=' between time and value")
        th = parse_time_hours_fn(a.strip())
        vv = float(b.strip())
        pairs.append((th * 3600.0, vv))

    if not pairs:
        return None

    pairs.sort(key=lambda x: x[0])
    tsec = np.array([p[0] for p in pairs], dtype=np.float64)
    vals = np.array([p[1] for p in pairs], dtype=np.float64)

    uniq_t = []
    uniq_v = []
    for ti, vi in zip(tsec.tolist(), vals.tolist()):
        if uniq_t and abs(ti - uniq_t[-1]) < 1.0e-9:
            uniq_v[-1] = vi
        else:
            uniq_t.append(ti)
            uniq_v.append(vi)

    return np.asarray(uniq_t, dtype=np.float64), np.asarray(uniq_v, dtype=np.float64)


def hydrograph_from_layer(
    layer,
    hydrograph_id: str = "",
    bc_type: Optional[int] = None,
    parse_time_hours_fn: Callable[[str], float] = parse_time_hours,
    vector_layer_type: Optional[type] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    hydrograph from layer.

    Parameters
    ----------
    layer
        Description of layer.
    hydrograph_id : str
        Description of hydrograph_id.
    bc_type : Optional[int]
        Description of bc_type.
    parse_time_hours_fn : Callable[[str], float]
        Description of parse_time_hours_fn.
    vector_layer_type : Optional[type]
        Description of vector_layer_type.

    Returns
    -------
    Optional[Tuple[np.ndarray, np.ndarray]]
    """
    if layer is None:
        return None
    if vector_layer_type is not None and not isinstance(layer, vector_layer_type):
        return None

    fields = set(layer.fields().names())
    t_field = None
    for cand in ("Time", "time", "t", "hours"):
        if cand in fields:
            t_field = cand
            break
    v_field = None
    for cand in ("Value", "value", "val", "q", "stage"):
        if cand in fields:
            v_field = cand
            break
    if t_field is None or v_field is None:
        return None

    hid_field = "hydrograph_id" if "hydrograph_id" in fields else None
    bct_field = "bc_type" if "bc_type" in fields else None

    pairs: List[Tuple[float, float]] = []
    for ft in layer.getFeatures():
        if hid_field is not None and hydrograph_id:
            hid = str(ft[hid_field] or "").strip()
            if hid != hydrograph_id:
                continue
        if bct_field is not None and bc_type is not None:
            try:
                if int(ft[bct_field]) != int(bc_type):
                    continue
            except Exception as _e:

                logger.warning(f"[ERROR] Exception in hydrograph_logic.py: {_e}")
        try:
            th = parse_time_hours_fn(str(ft[t_field]).strip())
            vv = float(ft[v_field])
        except Exception:
            continue
        pairs.append((th * 3600.0, vv))

    if not pairs:
        return None

    pairs.sort(key=lambda x: x[0])
    tsec = np.asarray([p[0] for p in pairs], dtype=np.float64)
    vals = np.asarray([p[1] for p in pairs], dtype=np.float64)
    return tsec, vals
