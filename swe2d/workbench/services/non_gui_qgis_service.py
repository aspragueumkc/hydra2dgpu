from __future__ import annotations

"""QGIS service helpers accessible without a visible dialog (headless/batch)."""

from typing import Optional

import numpy as np

import logging
logger = logging.getLogger(__name__)


def resolve_layer_field_name(layer: object, requested_name: str) -> str:
    """resolve layer field name."""
    target = str(requested_name or "").strip().lower()
    if not target or layer is None:
        return ""
    try:
        names = [str(v) for v in layer.fields().names()]
    except Exception:
        return ""
    by_lower = {str(name).strip().lower(): str(name) for name in names}
    return str(by_lower.get(target, ""))


def parse_feature_float(feature: object, field_name: str, default: float) -> float:
    """parse feature float."""
    fname = str(field_name or "").strip()
    if not fname:
        return float(default)
    try:
        value = feature[fname]
    except Exception:
        return float(default)
    if value in (None, ""):
        return float(default)
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def infer_obj_path_from_layer_3d_renderer(layer: object) -> str:
    """infer obj path from layer 3d renderer."""
    if layer is None:
        return ""
    try:
        renderer_3d = layer.renderer3D() if hasattr(layer, "renderer3D") else None
    except Exception:
        logger.warning("Silent fallback in Exception handler", exc_info=True)
        renderer_3d = None
    if renderer_3d is None:
        return ""

    probe_objects = [renderer_3d]
    try:
        symbol = renderer_3d.symbol() if hasattr(renderer_3d, "symbol") else None
    except Exception:
        logger.warning("Silent fallback in Exception handler", exc_info=True)
        symbol = None
    if symbol is not None:
        probe_objects.append(symbol)

    probe_attrs = (
        "modelPath",
        "modelFile",
        "filePath",
        "path",
        "model",
        "shape",
        "uri",
        "url",
    )

    for obj in probe_objects:
        for attr in probe_attrs:
            try:
                value = getattr(obj, attr, None)
                if callable(value):
                    value = value()
            except Exception:
                logger.warning("Silent fallback in Exception handler", exc_info=True)
                continue
            txt = str(value or "").strip()
            if txt.lower().endswith(".obj"):
                return txt
    return ""


def build_patch_terrain_surface(
    *,
    spec: object,
    raster_layer: object,
    qgs_point_xy_cls: object,
) -> Optional[np.ndarray]:
    """Build patch terrain surface."""
    if raster_layer is None or qgs_point_xy_cls is None:
        return None

    try:
        provider = raster_layer.dataProvider()
    except Exception:
        return None

    nx = int(getattr(spec, "nx", 0))
    ny = int(getattr(spec, "ny", 0))
    dx = float(getattr(spec, "dx", 0.0))
    dy = float(getattr(spec, "dy", 0.0))
    ox = float(getattr(spec, "origin_x", 0.0))
    oy = float(getattr(spec, "origin_y", 0.0))
    if nx <= 0 or ny <= 0 or dx <= 0.0 or dy <= 0.0:
        return None

    x_centers = ox + (np.arange(nx, dtype=np.float64) + 0.5) * dx
    y_centers = oy + (np.arange(ny, dtype=np.float64) + 0.5) * dy
    terrain = np.full((ny, nx), np.nan, dtype=np.float64)

    for j, yv in enumerate(y_centers):
        for i, xv in enumerate(x_centers):
            try:
                val, ok = provider.sample(qgs_point_xy_cls(float(xv), float(yv)), 1)
            except Exception:
                logger.warning("Silent fallback in Exception handler", exc_info=True)
                ok = False
                val = np.nan
            if ok and np.isfinite(val):
                terrain[j, i] = float(val)

    return terrain
