from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np


def parse_optional_float_text(text: str) -> Optional[float]:
    txt = str(text or "").strip()
    if not txt:
        return None
    return float(txt)


def collect_3d_patch_env_overrides(
    *,
    mesh_data: Dict[str, np.ndarray],
    target_len_x: float,
    target_len_y: float,
    target_len_z: float,
    xmin_override: Optional[float],
    xmax_override: Optional[float],
    ymin_override: Optional[float],
    ymax_override: Optional[float],
    zmin_override: Optional[float],
    zmax_override: Optional[float],
    terrain_zmin: Optional[float],
    bed_manning_n: float,
) -> Tuple[Dict[str, str], Dict[str, object]]:
    node_x = np.asarray(mesh_data.get("node_x", np.empty(0)), dtype=np.float64).ravel()
    node_y = np.asarray(mesh_data.get("node_y", np.empty(0)), dtype=np.float64).ravel()
    node_z = np.asarray(mesh_data.get("node_z", np.empty(0)), dtype=np.float64).ravel()
    if node_x.size <= 0 or node_y.size <= 0:
        raise RuntimeError("Mesh node coordinates are missing for 3D patch setup.")

    target_len_x = max(float(target_len_x), 1.0e-6)
    target_len_y = max(float(target_len_y), 1.0e-6)
    target_len_z = max(float(target_len_z), 1.0e-6)

    xmin_d = float(np.min(node_x))
    xmax_d = float(np.max(node_x))
    ymin_d = float(np.min(node_y))
    ymax_d = float(np.max(node_y))
    zmin_d = float(np.min(node_z)) if node_z.size > 0 else 0.0
    zmax_d = float(np.max(node_z)) if node_z.size > 0 else 1.0

    xmin = xmin_d if xmin_override is None else float(xmin_override)
    xmax = xmax_d if xmax_override is None else float(xmax_override)
    ymin = ymin_d if ymin_override is None else float(ymin_override)
    ymax = ymax_d if ymax_override is None else float(ymax_override)
    zmin_ui = zmin_d if zmin_override is None else float(zmin_override)
    zmax = zmax_d if zmax_override is None else float(zmax_override)
    zmin = zmin_ui

    if not (xmax > xmin and ymax > ymin):
        raise ValueError("3D patch ROI must satisfy xmax>xmin and ymax>ymin.")

    span_x = max(xmax - xmin, 1.0e-9)
    span_y = max(ymax - ymin, 1.0e-9)
    nx = max(2, int(math.ceil(span_x / target_len_x)))
    ny = max(2, int(math.ceil(span_y / target_len_y)))

    terrain_zmin_used = False
    if terrain_zmin is not None and np.isfinite(float(terrain_zmin)):
        zmin = float(terrain_zmin)
        terrain_zmin_used = True

    if zmax <= zmin:
        zmax = zmin + 1.0

    span_z = max(zmax - zmin, 1.0e-9)
    nz = max(2, int(math.ceil(span_z / target_len_z)))

    dx = max(span_x / float(nx), 1.0e-9)
    dy = max(span_y / float(ny), 1.0e-9)
    dz = max(span_z / float(nz), 1.0e-9)

    overrides = {
        "BACKWATER_SWE3D_PATCH_FACE_LEN_X": f"{target_len_x:.17g}",
        "BACKWATER_SWE3D_PATCH_FACE_LEN_Y": f"{target_len_y:.17g}",
        "BACKWATER_SWE3D_PATCH_FACE_LEN_Z": f"{target_len_z:.17g}",
        "BACKWATER_SWE3D_PATCH_NX": str(nx),
        "BACKWATER_SWE3D_PATCH_NY": str(ny),
        "BACKWATER_SWE3D_PATCH_NZ": str(nz),
        "BACKWATER_SWE3D_PATCH_DX": f"{dx:.17g}",
        "BACKWATER_SWE3D_PATCH_DY": f"{dy:.17g}",
        "BACKWATER_SWE3D_PATCH_DZ": f"{dz:.17g}",
        "BACKWATER_SWE3D_PATCH_ORIGIN_X": f"{xmin:.17g}",
        "BACKWATER_SWE3D_PATCH_ORIGIN_Y": f"{ymin:.17g}",
        "BACKWATER_SWE3D_PATCH_ORIGIN_Z": f"{zmin:.17g}",
        "BACKWATER_SWE3D_GRAVITY_Z_SIGN": "-1",
        "BACKWATER_SWE3D_ENABLE_BED_DRAG": "1",
        "BACKWATER_SWE3D_BED_MANNING_N": f"{float(bed_manning_n):.17g}",
        "BACKWATER_SWE3D_BED_DRAG_HREF": f"{max(dz, 1.0e-3):.17g}",
        "BACKWATER_SWE3D_BED_DRAG_LAYERS": "1",
    }

    metadata: Dict[str, object] = {
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "zmin": zmin,
        "zmax": zmax,
        "zmin_ui": zmin_ui,
        "terrain_zmin_used": terrain_zmin_used,
    }
    return overrides, metadata
