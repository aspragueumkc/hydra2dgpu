from __future__ import annotations

from typing import Callable, Optional

import logging

import numpy as np

logger = logging.getLogger(__name__)


def sample_terrain_min_z_for_roi_qgis(
    *,
    have_qgis_core: bool,
    qgs_pointxy_cls,
    terrain_layer_combo,
    combo_layer_fn: Callable[[object, str], Optional[object]],
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    nx_hint: int = 64,
    ny_hint: int = 64,
) -> Optional[float]:
    if not have_qgis_core or qgs_pointxy_cls is None:
        return None
    if terrain_layer_combo is None:
        return None
    if not (np.isfinite(xmin) and np.isfinite(xmax) and np.isfinite(ymin) and np.isfinite(ymax)):
        return None
    if not (xmax > xmin and ymax > ymin):
        return None

    raster_layer = combo_layer_fn(terrain_layer_combo, "raster")
    if raster_layer is None:
        return None

    try:
        provider = raster_layer.dataProvider()
    except Exception as exc:
        logger.debug("[DRAINAGE] Failed to get raster provider: %s", exc)
        return None

    sx = max(8, min(256, int(nx_hint) if int(nx_hint) > 0 else 64))
    sy = max(8, min(256, int(ny_hint) if int(ny_hint) > 0 else 64))
    xs = np.linspace(float(xmin), float(xmax), sx, dtype=np.float64)
    ys = np.linspace(float(ymin), float(ymax), sy, dtype=np.float64)

    min_val: Optional[float] = None
    for yv in ys:
        for xv in xs:
            try:
                val, ok = provider.sample(qgs_pointxy_cls(float(xv), float(yv)), 1)
            except Exception as exc:
                logger.debug("[DRAINAGE] Failed to sample raster point: %s", exc)
                ok = False
                val = np.nan
            if ok and np.isfinite(val):
                zv = float(val)
                if min_val is None or zv < min_val:
                    min_val = zv

    return min_val
