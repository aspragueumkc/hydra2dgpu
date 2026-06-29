from __future__ import annotations

"""QGIS geometry-to-cell-index mapping for internal flow forcing polygons."""

from typing import Optional, Tuple

import numpy as np


def internal_flow_geom_to_indices_weights_qgis(
    geom,
    cx: np.ndarray,
    cy: np.ndarray,
    *,
    qgs_wkb_types,
    qgs_geometry_cls,
    qgs_pointxy_cls,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Map a QGIS geometry to cell-centroid indices and area weights for internal flow."""
    try:
        wkb_type = int(geom.wkbType())
    except Exception:
        wkb_type = -1

    if qgs_wkb_types.geometryType(wkb_type) == qgs_wkb_types.GeometryType.PolygonGeometry:
        hit_ids = []
        for i in range(cx.shape[0]):
            p = qgs_geometry_cls.fromPointXY(qgs_pointxy_cls(float(cx[i]), float(cy[i])))
            if geom.contains(p) or geom.intersects(p):
                hit_ids.append(i)
        if not hit_ids:
            return None
        idx_arr = np.asarray(hit_ids, dtype=np.int32)
        wt_arr = np.full(idx_arr.shape[0], 1.0 / float(idx_arr.shape[0]), dtype=np.float64)
        return idx_arr, wt_arr

    rp = geom.centroid().asPoint() if not geom.centroid().isEmpty() else None
    if rp is None:
        return None
    dx = cx - float(rp.x())
    dy = cy - float(rp.y())
    idx = int(np.argmin(dx * dx + dy * dy))
    idx_arr = np.asarray([idx], dtype=np.int32)
    wt_arr = np.asarray([1.0], dtype=np.float64)
    return idx_arr, wt_arr
