from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from swe2d.mesh.mesh_models import (
    MeshResult,
    _GmshQualityConfig,
    _TQMeshQualityConfig,
)


def _polygon_area_xy(xs: np.ndarray, ys: np.ndarray) -> float:
    """polygon area xy"""
    if xs.size < 3:
        return 0.0
    x2 = np.roll(xs, -1)
    y2 = np.roll(ys, -1)
    return 0.5 * float(np.sum(xs * y2 - x2 * ys))


def _mesh_quality_stats(
    vx: np.ndarray,
    vy: np.ndarray,
    tris: np.ndarray,
    quads: np.ndarray,
    bbox_area: float = 0.0,
) -> dict:
    """mesh quality stats"""
    n_tri = tris.shape[0] if tris.size > 0 else 0
    n_quad = quads.shape[0] if quads.size > 0 else 0
    n_total = n_tri + n_quad
    if n_total == 0 or bbox_area <= 0.0:
        return {
            "n_tri": int(n_tri),
            "n_quad": int(n_quad),
            "min_angle_deg": 90.0,
            "max_aspect_ratio": 0.0,
            "min_area": 0.0,
            "min_area_rel_bbox": 0.0,
            "coverage": 0.0,
            "bbox_area": float(bbox_area),
        }
    min_angles: List[float] = []
    aspect_ratios: List[float] = []
    min_area = float("inf")
    if n_tri > 0:
        for tri in tris:
            ax, ay = float(vx[tri[0]]), float(vy[tri[0]])
            bx, by = float(vx[tri[1]]), float(vy[tri[1]])
            cx, cy = float(vx[tri[2]]), float(vy[tri[2]])
            a = _tri_area(ax, ay, bx, by, cx, cy)
            if a > 0.0:
                min_area = min(min_area, a)
            angles = _tri_angles(ax, ay, bx, by, cx, cy)
            if angles:
                min_angles.append(min(angles))
            ar = _tri_aspect_ratio(ax, ay, bx, by, cx, cy)
            if ar > 0.0:
                aspect_ratios.append(ar)
    if n_quad > 0:
        for quad in quads:
            ax, ay = float(vx[quad[0]]), float(vy[quad[0]])
            bx, by = float(vx[quad[1]]), float(vy[quad[1]])
            cx, cy = float(vx[quad[2]]), float(vy[quad[2]])
            dx, dy = float(vx[quad[3]]), float(vy[quad[3]])
            a1 = _tri_area(ax, ay, bx, by, cx, cy)
            a2 = _tri_area(ax, ay, cx, cy, dx, dy)
            area = a1 + a2
            if area > 0.0:
                min_area = min(min_area, area)
            angles = _quad_angles(ax, ay, bx, by, cx, cy, dx, dy)
            if angles:
                min_angles.append(min(angles))
            ar = max(
                _tri_aspect_ratio(ax, ay, bx, by, cx, cy),
                _tri_aspect_ratio(ax, ay, cx, cy, dx, dy),
            )
            if ar > 0.0:
                aspect_ratios.append(ar)
    total_mesh_area = sum(
        abs(_polygon_area_xy(vx[face], vy[face]))
        for face in _face_rings(tris, quads)
    )
    return {
        "n_tri": int(n_tri),
        "n_quad": int(n_quad),
        "min_angle_deg": float(min(min_angles)) if min_angles else 90.0,
        "max_aspect_ratio": float(max(aspect_ratios)) if aspect_ratios else 0.0,
        "min_area": float(min_area) if min_area != float("inf") else 0.0,
        "min_area_rel_bbox": float(min_area / max(bbox_area, 1.0e-30)) if min_area != float("inf") else 0.0,
        "coverage": float(min(total_mesh_area / max(bbox_area, 1.0e-30), 1.0)),
        "bbox_area": float(bbox_area),
    }


def _tri_area(ax, ay, bx, by, cx, cy) -> float:
    """tri area"""
    return 0.5 * abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay))


def _tri_angles(ax, ay, bx, by, cx, cy) -> List[float]:
    """tri angles"""
    import math
    a = math.hypot(bx - cx, by - cy)
    b = math.hypot(cx - ax, cy - ay)
    c = math.hypot(ax - bx, ay - by)
    if a <= 0.0 or b <= 0.0 or c <= 0.0:
        return []
    angles = [
        math.degrees(math.acos(max(-1.0, min(1.0, (b*b + c*c - a*a) / (2.0*b*c))))),
        math.degrees(math.acos(max(-1.0, min(1.0, (a*a + c*c - b*b) / (2.0*a*c))))),
        math.degrees(math.acos(max(-1.0, min(1.0, (a*a + b*b - c*c) / (2.0*a*b))))),
    ]
    return angles


def _tri_aspect_ratio(ax, ay, bx, by, cx, cy) -> float:
    """tri aspect ratio"""
    import math
    a = math.hypot(bx - cx, by - cy)
    b = math.hypot(cx - ax, cy - ay)
    c = math.hypot(ax - bx, ay - by)
    s = (a + b + c) / 2.0
    if s <= 0.0:
        return 0.0
    area = math.sqrt(max(0.0, s * (s - a) * (s - b) * (s - c)))
    if area <= 0.0:
        return 0.0
    r_circ = (a * b * c) / (4.0 * area)
    r_in = area / s
    return float(r_circ / r_in) if r_in > 0.0 else 0.0


def _quad_angles(ax, ay, bx, by, cx, cy, dx, dy) -> List[float]:
    """quad angles"""
    return (
        _tri_angles(ax, ay, bx, by, cx, cy) +
        _tri_angles(ax, ay, cx, cy, dx, dy)
    )


def _face_rings(tris: np.ndarray, quads: np.ndarray):
    """face rings"""
    for tri in tris:
        yield tri[:3]
    for quad in quads:
        yield quad[:4]


def _quality_passes(stats: dict, cfg: _TQMeshQualityConfig) -> bool:
    """quality passes"""
    if stats["n_tri"] + stats["n_quad"] == 0:
        return False
    if stats.get("min_angle_deg", 90.0) < cfg.min_angle_deg:
        return False
    if stats.get("max_aspect_ratio", 0.0) > cfg.max_aspect_ratio:
        return False
    if stats.get("min_area_rel_bbox", 1.0) < cfg.min_area_rel_bbox:
        return False
    return True


def _quality_score(stats: dict, cfg: _TQMeshQualityConfig) -> float:
    """quality score"""
    score = 0.0
    if stats.get("min_angle_deg", 90.0) >= cfg.min_angle_deg:
        score += 1.0
    if stats.get("max_aspect_ratio", 0.0) <= cfg.max_aspect_ratio:
        score += 1.0
    if stats.get("min_area_rel_bbox", 1.0) >= cfg.min_area_rel_bbox:
        score += 1.0
    return score


def _face_mesh_quality_stats(
    mesh: MeshResult,
    cfg: _GmshQualityConfig,
) -> Dict[str, float]:
    """face mesh quality stats"""
    import math
    offs = mesh.cell_face_offsets.astype(np.int32)
    faces = mesh.cell_face_nodes.astype(np.int32)
    n_cells = max(0, int(offs.size) - 1)
    if n_cells == 0:
        return {
            "min_angle_deg": 90.0,
            "max_aspect_ratio": 0.0,
            "min_area": 0.0,
            "max_non_orth_deg": 0.0,
            "failed_any_cells": 0,
        }
    x = mesh.node_x
    y = mesh.node_y
    min_angle_deg = 90.0
    max_aspect_ratio = 0.0
    min_area = float("inf")
    max_non_orth_deg = 0.0
    failed_min_angle = 0
    failed_max_aspect = 0
    failed_min_area = 0
    failed_max_non_orth = 0
    for i in range(n_cells):
        s = int(offs[i])
        e = int(offs[i + 1])
        ids = faces[s:e]
        if ids.size < 3:
            continue
        px = x[ids]
        py = y[ids]
        area = abs(_polygon_area_xy(px, py))
        if area <= 0.0:
            continue
        min_area = min(min_area, area)
        nv = ids.size
        angles_deg: List[float] = []
        edge_lens: List[float] = []
        for k in range(nv):
            x0, y0 = float(px[k]), float(py[k])
            x1, y1 = float(px[(k + 1) % nv]), float(py[(k + 1) % nv])
            x2, y2 = float(px[(k + 2) % nv]), float(py[(k + 2) % nv])
            a = math.hypot(x2 - x1, y2 - y1)
            b = math.hypot(x0 - x2, y0 - y2)
            c = math.hypot(x1 - x0, y1 - y0)
            edge_lens.append(c)
            if a <= 0.0 or b <= 0.0 or c <= 0.0:
                continue
            ang = math.degrees(math.acos(
                max(-1.0, min(1.0, (b*b + c*c - a*a) / (2.0*b*c)))
            ))
            angles_deg.append(ang)
        if angles_deg:
            ca = min(angles_deg)
            min_angle_deg = min(min_angle_deg, ca)
            if ca < cfg.min_angle_deg:
                failed_min_angle += 1
        if edge_lens:
            s_el = sum(edge_lens) / 2.0
            if s_el > 0.0 and area > 0.0:
                r_in = area / s_el
                r_circ = (edge_lens[0] * edge_lens[1] * edge_lens[2]) / (4.0 * area) \
                    if nv >= 4 and len(edge_lens) >= 3 else max(edge_lens) / 2.0
                ar = (r_circ / r_in) if r_in > 0.0 else 0.0
                max_aspect_ratio = max(max_aspect_ratio, ar)
                if ar > cfg.max_aspect_ratio:
                    failed_max_aspect += 1
        if area < cfg.min_area_rel_bbox * 1.0:
            failed_min_area += 1
    bbox_area_val = (float(np.max(x)) - float(np.min(x))) * (float(np.max(y)) - float(np.min(y)))
    rel_area = min_area / max(bbox_area_val, 1.0e-30) if min_area != float("inf") else 0.0
    failed_any = 0
    if failed_min_angle > 0 or failed_max_aspect > 0 or failed_min_area > 0 or failed_max_non_orth > 0:
        failed_any = failed_min_angle + failed_max_aspect + failed_min_area + failed_max_non_orth
    return {
        "min_angle_deg": float(min_angle_deg),
        "max_aspect_ratio": float(max_aspect_ratio),
        "min_area": float(min_area) if min_area != float("inf") else 0.0,
        "min_area_rel_bbox": float(rel_area),
        "max_non_orth_deg": float(max_non_orth_deg),
        "failed_any_cells": int(failed_any),
        "failed_min_angle_cells": int(failed_min_angle),
        "failed_max_aspect_cells": int(failed_max_aspect),
        "failed_min_area_cells": int(failed_min_area),
        "failed_max_non_orth_cells": int(failed_max_non_orth),
    }


def _gmsh_quality_passes(stats: dict, cfg: _GmshQualityConfig) -> bool:
    """gmsh quality passes"""
    if int(stats.get("failed_any_cells", 0)) > 0:
        return False
    if float(stats.get("min_angle_deg", 90.0)) < cfg.min_angle_deg - 1.0e-12:
        return False
    if float(stats.get("max_aspect_ratio", 0.0)) > cfg.max_aspect_ratio + 1.0e-12:
        return False
    if float(stats.get("min_area", 0.0)) < (cfg.min_area_rel_bbox * 1.0 - 1.0e-18):
        return False
    return True


def _gmsh_quality_score(stats: dict, _cfg: _GmshQualityConfig) -> float:
    """gmsh quality score"""
    score = 0.0
    score += max(0.0, min(1.0, float(stats.get("min_angle_deg", 90.0)) / 30.0))
    score += max(0.0, min(1.0, 1.0 / max(1.0, float(stats.get("max_aspect_ratio", 1.0)))))
    score += max(0.0, min(1.0, 100.0 * float(stats.get("min_area", 0.0))))
    return score


__all__ = [
    "_face_mesh_quality_stats",
    "_gmsh_quality_passes",
    "_gmsh_quality_score",
    "_mesh_quality_stats",
    "_polygon_area_xy",
    "_quality_passes",
    "_quality_score",
]
