#!/usr/bin/env python3
"""Face-centric meshing utilities for SWE2D.

Topology-first meshing pipeline inspired by HEC-RAS 2025 concepts
(conceptual topology -> computational mesh).  Backend-agnostic: swap the
generator without changing GUI or solver code.

Available backends
------------------
"gmsh"       (default) : Gmsh 4.x constrained meshing.  True Blossom quad
                         recombination, Transfinite structured zones, breakline
                         embedding, per-zone size fields.  Requires: pip install gmsh.
"structured" (fallback) : Deterministic structured grid.  No dependencies.
                         Use when gmsh is not available or for quick tests.

Output contract
---------------
- cell_face_offsets / cell_face_nodes : polygon CSR topology for the solver.
- cell_nodes                          : triangle-fan decomposition for plotting.
- cell_type per face reflects the source conceptual type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import os
import time
import warnings

import numpy as np


@dataclass
class ConceptualNode:
    node_id: int
    x: float
    y: float


@dataclass
class ConceptualArc:
    arc_id: int
    node0: int = -1
    node1: int = -1
    points_xy: Optional[List[Tuple[float, float]]] = None


@dataclass
class ConceptualRegion:
    region_id: int
    ring_xy: List[Tuple[float, float]]
    default_size: float
    default_cell_type: str
    edge_lengths: Optional[List[float]] = None
    hole_rings: Optional[List[List[Tuple[float, float]]]] = None


@dataclass
class CellConstraint:
    constraint_id: int
    ring_xy: List[Tuple[float, float]]
    target_size: float
    cell_type: str


@dataclass
class QuadEdgeControl:
    region_id: int
    edge_id: int
    points_xy: List[Tuple[float, float]]
    target_size: Optional[float] = None
    n_layers: int = 0
    first_height: Optional[float] = None
    growth_rate: float = 1.0


@dataclass
class ConceptualModel:
    nodes: List[ConceptualNode]
    arcs: List[ConceptualArc]
    regions: List[ConceptualRegion]
    constraints: List[CellConstraint]
    quad_edges: List[QuadEdgeControl]


@dataclass
class MeshResult:
    node_x: np.ndarray
    node_y: np.ndarray
    node_z: np.ndarray
    cell_nodes: np.ndarray
    cell_face_offsets: np.ndarray
    cell_face_nodes: np.ndarray
    cell_type: np.ndarray
    region_id: np.ndarray
    target_size: np.ndarray
    quality_summary: Optional[Dict[str, object]] = None


_CELL_TYPES = {"triangular", "quadrilateral", "cartesian", "empty"}


@dataclass
class _TQMeshQualityConfig:
    min_angle_deg: float
    max_aspect_ratio: float
    min_area_rel_bbox: float
    strict: bool
    size_scales: Tuple[float, ...]
    smooth_increments: Tuple[int, ...]


@dataclass
class _GmshQualityConfig:
    enabled: bool
    strict: bool
    min_angle_deg: float
    max_aspect_ratio: float
    min_area_rel_bbox: float
    max_non_orth_deg: float
    max_iterations: int
    time_limit_s: float
    size_scales: Tuple[float, ...]
    smooth_increments: Tuple[int, ...]


def _polygon_area_xy(xs: np.ndarray, ys: np.ndarray) -> float:
    if xs.size < 3:
        return 0.0
    x2 = np.roll(xs, -1)
    y2 = np.roll(ys, -1)
    return 0.5 * float(np.sum(xs * y2 - x2 * ys))


def _repair_mesh_result(mesh: MeshResult, area_tol: float = 1.0e-10) -> MeshResult:
    """Remove degenerate faces and normalize face rings for solver robustness."""
    if mesh.cell_face_offsets.size < 2:
        return mesh

    offs = mesh.cell_face_offsets.astype(np.int32)
    nodes = mesh.cell_face_nodes.astype(np.int32)
    keep_face_nodes: List[int] = []
    keep_offsets: List[int] = [0]
    keep_idx: List[int] = []

    for i in range(offs.size - 1):
        s = int(offs[i])
        e = int(offs[i + 1])
        poly = nodes[s:e].tolist()
        if len(poly) < 3:
            continue

        # Remove immediate duplicate vertices.
        compact: List[int] = []
        for v in poly:
            if not compact or compact[-1] != int(v):
                compact.append(int(v))
        if len(compact) >= 2 and compact[0] == compact[-1]:
            compact = compact[:-1]
        if len(compact) < 3:
            continue

        x = mesh.node_x[np.asarray(compact, dtype=np.int32)]
        y = mesh.node_y[np.asarray(compact, dtype=np.int32)]
        area = _polygon_area_xy(x, y)
        if abs(area) <= area_tol:
            continue
        if area < 0.0:
            compact = list(reversed(compact))

        keep_face_nodes.extend(compact)
        keep_offsets.append(len(keep_face_nodes))
        keep_idx.append(i)

    if not keep_idx:
        raise ValueError("Mesh repair removed all faces (all faces degenerate).")

    keep_idx_arr = np.asarray(keep_idx, dtype=np.int32)

    # Rebuild plotting triangles from repaired polygon faces.
    tri_plot: List[int] = []
    ko = np.asarray(keep_offsets, dtype=np.int32)
    kn = np.asarray(keep_face_nodes, dtype=np.int32)
    for i in range(ko.size - 1):
        s = int(ko[i])
        e = int(ko[i + 1])
        poly = kn[s:e]
        if poly.size == 3:
            tri_plot.extend([int(poly[0]), int(poly[1]), int(poly[2])])
        elif poly.size > 3:
            for k in range(1, poly.size - 1):
                tri_plot.extend([int(poly[0]), int(poly[k]), int(poly[k + 1])])

    # Drop unreferenced/orphan nodes so exported node layers align exactly with
    # computational faces (important for Gmsh workflows with embedded geometry points).
    used_nodes = np.unique(kn)
    if used_nodes.size <= 0:
        raise ValueError("Mesh repair produced faces but no referenced nodes.")

    remap = np.full(mesh.node_x.shape[0], -1, dtype=np.int32)
    remap[used_nodes] = np.arange(used_nodes.size, dtype=np.int32)

    kn = remap[kn]
    tri_arr = np.asarray(tri_plot, dtype=np.int32)
    if tri_arr.size > 0:
        tri_arr = remap[tri_arr]

    if mesh.node_z.shape[0] == mesh.node_x.shape[0]:
        compact_z = mesh.node_z[used_nodes]
    else:
        compact_z = np.zeros(used_nodes.size, dtype=np.float64)

    return MeshResult(
        node_x=mesh.node_x[used_nodes],
        node_y=mesh.node_y[used_nodes],
        node_z=compact_z,
        cell_nodes=tri_arr,
        cell_face_offsets=np.asarray(keep_offsets, dtype=np.int32),
        cell_face_nodes=kn,
        cell_type=mesh.cell_type[keep_idx_arr],
        region_id=mesh.region_id[keep_idx_arr],
        target_size=mesh.target_size[keep_idx_arr],
    )


def _require_nonempty_mesh(mesh: MeshResult, backend_name: str) -> MeshResult:
    n_nodes = int(np.asarray(mesh.node_x).size)
    n_faces = max(0, int(np.asarray(mesh.cell_face_offsets).size) - 1)
    if n_nodes <= 0 or n_faces <= 0:
        raise RuntimeError(
            f"{backend_name} produced an empty mesh (nodes={n_nodes}, faces={n_faces}). "
            "Check conceptual polygons/constraints and retry."
        )
    return mesh


def _weld_mesh_nodes(
    node_x: np.ndarray,
    node_y: np.ndarray,
    *connectivity_arrays: np.ndarray,
    tol: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, ...]]:
    """Merge coincident vertices and remap connectivity.

    TQMesh multi-region support currently meshes regions independently.  For
    adjacent, non-overlapping regions with shared boundaries, the merged solver
    mesh must collapse identical interface coordinates back onto the same global
    node ids so shared edges can be reconstructed from polygon connectivity.
    """
    if node_x.size != node_y.size:
        raise ValueError("node_x and node_y must have the same length")
    if node_x.size == 0:
        return node_x, node_y, tuple(arr.copy() for arr in connectivity_arrays)

    if tol is None:
        scale = max(
            1.0,
            float(np.max(np.abs(node_x))),
            float(np.max(np.abs(node_y))),
            float(np.ptp(node_x)),
            float(np.ptp(node_y)),
        )
        tol = max(1.0e-9, scale * 1.0e-9)

    buckets: Dict[Tuple[int, int], List[int]] = {}
    unique_x: List[float] = []
    unique_y: List[float] = []
    remap = np.empty(node_x.size, dtype=np.int32)

    for old_idx, (x, y) in enumerate(zip(node_x.tolist(), node_y.tolist())):
        key = (int(np.rint(x / tol)), int(np.rint(y / tol)))
        new_idx = None
        for cand in buckets.get(key, []):
            if abs(unique_x[cand] - x) <= tol and abs(unique_y[cand] - y) <= tol:
                new_idx = cand
                break
        if new_idx is None:
            new_idx = len(unique_x)
            unique_x.append(float(x))
            unique_y.append(float(y))
            buckets.setdefault(key, []).append(new_idx)
        remap[old_idx] = int(new_idx)

    remapped_arrays: List[np.ndarray] = []
    for arr in connectivity_arrays:
        arr32 = np.asarray(arr, dtype=np.int32)
        if arr32.size == 0:
            remapped_arrays.append(arr32.copy())
        else:
            remapped_arrays.append(remap[arr32])

    return (
        np.asarray(unique_x, dtype=np.float64),
        np.asarray(unique_y, dtype=np.float64),
        tuple(remapped_arrays),
    )


def _as_float(v, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _as_int(v, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _normalize_cell_type(v: str, default: str = "triangular") -> str:
    s = str(v or "").strip().lower()
    if s not in _CELL_TYPES:
        return default
    return s


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, "1" if default else "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _env_csv_floats(name: str, default: Sequence[float]) -> Tuple[float, ...]:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return tuple(float(v) for v in default)
    vals: List[float] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            vals.append(float(tok))
        except Exception:
            continue
    if not vals:
        return tuple(float(v) for v in default)
    return tuple(vals)


def _mesh_quality_stats(
    vx: np.ndarray,
    vy: np.ndarray,
    tris: np.ndarray,
    quads: np.ndarray,
) -> Dict[str, float]:
    """Compute compact quality metrics used by adaptive retries."""
    n_tri = int(tris.shape[0]) if tris.ndim == 2 else 0
    n_quad = int(quads.shape[0]) if quads.ndim == 2 else 0
    n_cells = n_tri + n_quad
    if n_cells <= 0:
        return {
            "n_cells": 0.0,
            "min_angle_deg": 0.0,
            "max_aspect_ratio": float("inf"),
            "min_area": 0.0,
            "bbox_area": 0.0,
        }

    min_angle = float("inf")
    max_aspect = 0.0
    min_area = float("inf")

    def _cell_metrics(conn: np.ndarray) -> Tuple[float, float, float]:
        pts_x = vx[conn]
        pts_y = vy[conn]
        px2 = np.roll(pts_x, -1)
        py2 = np.roll(pts_y, -1)
        area = abs(_polygon_area_xy(pts_x, pts_y))

        ex = px2 - pts_x
        ey = py2 - pts_y
        el = np.hypot(ex, ey)
        min_len = float(np.min(el)) if el.size else 0.0
        max_len = float(np.max(el)) if el.size else float("inf")
        aspect = (max_len / max(min_len, 1e-14)) if np.isfinite(max_len) else float("inf")

        best_min_angle = float("inf")
        n = conn.size
        for i in range(n):
            ip = (i - 1) % n
            inx = (i + 1) % n
            v1x = pts_x[ip] - pts_x[i]
            v1y = pts_y[ip] - pts_y[i]
            v2x = pts_x[inx] - pts_x[i]
            v2y = pts_y[inx] - pts_y[i]
            n1 = max(float(np.hypot(v1x, v1y)), 1e-14)
            n2 = max(float(np.hypot(v2x, v2y)), 1e-14)
            cosang = (v1x * v2x + v1y * v2y) / (n1 * n2)
            cosang = max(-1.0, min(1.0, float(cosang)))
            ang = float(np.degrees(np.arccos(cosang)))
            best_min_angle = min(best_min_angle, ang)
        return best_min_angle, aspect, area

    if n_tri:
        for tri in tris:
            a, ar, area = _cell_metrics(np.asarray(tri, dtype=np.int32))
            min_angle = min(min_angle, a)
            max_aspect = max(max_aspect, ar)
            min_area = min(min_area, area)

    if n_quad:
        for quad in quads:
            a, ar, area = _cell_metrics(np.asarray(quad, dtype=np.int32))
            min_angle = min(min_angle, a)
            max_aspect = max(max_aspect, ar)
            min_area = min(min_area, area)

    bbox_area = max(float(np.ptp(vx)) * float(np.ptp(vy)), 0.0)
    return {
        "n_cells": float(n_cells),
        "min_angle_deg": float(min_angle if np.isfinite(min_angle) else 0.0),
        "max_aspect_ratio": float(max_aspect if np.isfinite(max_aspect) else float("inf")),
        "min_area": float(min_area if np.isfinite(min_area) else 0.0),
        "bbox_area": float(bbox_area),
    }


def _quality_passes(stats: Dict[str, float], cfg: _TQMeshQualityConfig) -> bool:
    if stats.get("n_cells", 0.0) <= 0.0:
        return False
    area_floor = max(cfg.min_area_rel_bbox * max(stats.get("bbox_area", 0.0), 1.0), 1e-18)
    return (
        stats.get("min_angle_deg", 0.0) >= cfg.min_angle_deg
        and stats.get("max_aspect_ratio", float("inf")) <= cfg.max_aspect_ratio
        and stats.get("min_area", 0.0) >= area_floor
    )


def _quality_score(stats: Dict[str, float], cfg: _TQMeshQualityConfig) -> float:
    """Higher score is better; used to pick best non-passing candidate in non-strict mode."""
    area_floor = max(cfg.min_area_rel_bbox * max(stats.get("bbox_area", 0.0), 1.0), 1e-18)
    angle_term = stats.get("min_angle_deg", 0.0) / max(cfg.min_angle_deg, 1e-6)
    aspect_term = max(cfg.max_aspect_ratio, 1e-6) / max(stats.get("max_aspect_ratio", float("inf")), 1e-6)
    area_term = stats.get("min_area", 0.0) / max(area_floor, 1e-18)
    return float(min(angle_term, 2.0) + min(aspect_term, 2.0) + min(area_term, 2.0))


def _face_mesh_quality_stats(mesh: MeshResult, cfg: Optional[_GmshQualityConfig] = None) -> Dict[str, float]:
    """Compute quality metrics on polygon face topology.

    Metrics are used by the iterative Gmsh quality loop and include an
    approximate non-orthogonality estimate on interior edges.
    """
    vx = np.asarray(mesh.node_x, dtype=np.float64)
    vy = np.asarray(mesh.node_y, dtype=np.float64)
    offs = np.asarray(mesh.cell_face_offsets, dtype=np.int32)
    nodes = np.asarray(mesh.cell_face_nodes, dtype=np.int32)

    n_cells = max(0, int(offs.size) - 1)
    if n_cells <= 0:
        return {
            "n_cells": 0.0,
            "min_angle_deg": 0.0,
            "max_aspect_ratio": float("inf"),
            "min_area": 0.0,
            "bbox_area": 0.0,
            "max_non_orth_deg": 90.0,
        }

    min_angle = float("inf")
    max_aspect = 0.0
    min_area = float("inf")
    centroids = np.zeros((n_cells, 2), dtype=np.float64)
    cell_min_angle = np.full(n_cells, np.nan, dtype=np.float64)
    cell_aspect = np.full(n_cells, np.nan, dtype=np.float64)
    cell_area = np.full(n_cells, np.nan, dtype=np.float64)
    cell_max_non_orth = np.zeros(n_cells, dtype=np.float64)
    edge_map: Dict[Tuple[int, int], List[Tuple[int, float, float, float, float, float, float]]] = {}

    for c in range(n_cells):
        s = int(offs[c])
        e = int(offs[c + 1])
        conn = np.asarray(nodes[s:e], dtype=np.int32)
        if conn.size >= 2 and conn[0] == conn[-1]:
            conn = conn[:-1]
        if conn.size < 3:
            continue

        px = vx[conn]
        py = vy[conn]
        centroids[c, 0] = float(np.mean(px))
        centroids[c, 1] = float(np.mean(py))

        area = abs(_polygon_area_xy(px, py))
        min_area = min(min_area, area)
        cell_area[c] = float(area)

        px2 = np.roll(px, -1)
        py2 = np.roll(py, -1)
        ex = px2 - px
        ey = py2 - py
        el = np.hypot(ex, ey)
        min_len = float(np.min(el)) if el.size else 0.0
        max_len = float(np.max(el)) if el.size else float("inf")
        aspect = (max_len / max(min_len, 1.0e-14)) if np.isfinite(max_len) else float("inf")
        max_aspect = max(max_aspect, aspect)
        cell_aspect[c] = float(aspect)

        best_min_angle = float("inf")
        n = conn.size
        for i in range(n):
            ip = (i - 1) % n
            inx = (i + 1) % n
            v1x = px[ip] - px[i]
            v1y = py[ip] - py[i]
            v2x = px[inx] - px[i]
            v2y = py[inx] - py[i]
            n1 = max(float(np.hypot(v1x, v1y)), 1.0e-14)
            n2 = max(float(np.hypot(v2x, v2y)), 1.0e-14)
            cosang = (v1x * v2x + v1y * v2y) / (n1 * n2)
            cosang = max(-1.0, min(1.0, float(cosang)))
            ang = float(np.degrees(np.arccos(cosang)))
            best_min_angle = min(best_min_angle, ang)
        min_angle = min(min_angle, best_min_angle)
        cell_min_angle[c] = float(best_min_angle)

        for i in range(n):
            a = int(conn[i])
            b = int(conn[(i + 1) % n])
            x0 = float(vx[a])
            y0 = float(vy[a])
            x1 = float(vx[b])
            y1 = float(vy[b])
            midx = 0.5 * (x0 + x1)
            midy = 0.5 * (y0 + y1)
            key = (a, b) if a < b else (b, a)
            edge_map.setdefault(key, []).append((c, midx, midy, x0, y0, x1, y1))

    max_non_orth = 0.0
    for vals in edge_map.values():
        if len(vals) != 2:
            continue
        c0, _, _, x0, y0, x1, y1 = vals[0]
        c1, _, _, _, _, _, _ = vals[1]
        dcx = centroids[c1, 0] - centroids[c0, 0]
        dcy = centroids[c1, 1] - centroids[c0, 1]
        dn = float(np.hypot(dcx, dcy))
        if dn <= 1.0e-14:
            continue
        ex = x1 - x0
        ey = y1 - y0
        # Face-normal direction from edge tangent; sign is irrelevant due to abs().
        nx = ey
        ny = -ex
        nn = float(np.hypot(nx, ny))
        if nn <= 1.0e-14:
            continue
        cosang = abs((nx * dcx + ny * dcy) / (nn * dn))
        cosang = max(-1.0, min(1.0, cosang))
        non_orth = float(np.degrees(np.arccos(cosang)))
        max_non_orth = max(max_non_orth, non_orth)
        cell_max_non_orth[c0] = max(float(cell_max_non_orth[c0]), non_orth)
        cell_max_non_orth[c1] = max(float(cell_max_non_orth[c1]), non_orth)

    bbox_area = max(float(np.ptp(vx)) * float(np.ptp(vy)), 0.0)
    out = {
        "n_cells": float(n_cells),
        "min_angle_deg": float(min_angle if np.isfinite(min_angle) else 0.0),
        "max_aspect_ratio": float(max_aspect if np.isfinite(max_aspect) else float("inf")),
        "min_area": float(min_area if np.isfinite(min_area) else 0.0),
        "bbox_area": float(bbox_area),
        "max_non_orth_deg": float(max_non_orth),
    }

    if cfg is not None:
        area_floor = max(float(cfg.min_area_rel_bbox) * max(float(bbox_area), 1.0), 1.0e-18)
        valid = np.isfinite(cell_min_angle) & np.isfinite(cell_aspect) & np.isfinite(cell_area)
        if np.any(valid):
            fail_angle = int(np.count_nonzero(cell_min_angle[valid] < float(cfg.min_angle_deg)))
            fail_aspect = int(np.count_nonzero(cell_aspect[valid] > float(cfg.max_aspect_ratio)))
            fail_area = int(np.count_nonzero(cell_area[valid] < area_floor))
            fail_non_orth = int(np.count_nonzero(cell_max_non_orth[valid] > float(cfg.max_non_orth_deg)))
            fail_any = int(
                np.count_nonzero(
                    (cell_min_angle[valid] < float(cfg.min_angle_deg))
                    | (cell_aspect[valid] > float(cfg.max_aspect_ratio))
                    | (cell_area[valid] < area_floor)
                    | (cell_max_non_orth[valid] > float(cfg.max_non_orth_deg))
                )
            )
            out.update(
                {
                    "area_floor": float(area_floor),
                    "failed_min_angle_cells": float(fail_angle),
                    "failed_max_aspect_cells": float(fail_aspect),
                    "failed_min_area_cells": float(fail_area),
                    "failed_max_non_orth_cells": float(fail_non_orth),
                    "failed_any_cells": float(fail_any),
                }
            )
        else:
            out.update(
                {
                    "area_floor": float(area_floor),
                    "failed_min_angle_cells": 0.0,
                    "failed_max_aspect_cells": 0.0,
                    "failed_min_area_cells": 0.0,
                    "failed_max_non_orth_cells": 0.0,
                    "failed_any_cells": 0.0,
                }
            )

    return out


def _gmsh_quality_passes(stats: Dict[str, float], cfg: _GmshQualityConfig) -> bool:
    if stats.get("n_cells", 0.0) <= 0.0:
        return False
    area_floor = max(cfg.min_area_rel_bbox * max(stats.get("bbox_area", 0.0), 1.0), 1.0e-18)
    return (
        stats.get("min_angle_deg", 0.0) >= cfg.min_angle_deg
        and stats.get("max_aspect_ratio", float("inf")) <= cfg.max_aspect_ratio
        and stats.get("min_area", 0.0) >= area_floor
        and stats.get("max_non_orth_deg", 90.0) <= cfg.max_non_orth_deg
    )


def _gmsh_quality_score(stats: Dict[str, float], cfg: _GmshQualityConfig) -> float:
    area_floor = max(cfg.min_area_rel_bbox * max(stats.get("bbox_area", 0.0), 1.0), 1.0e-18)
    angle_term = stats.get("min_angle_deg", 0.0) / max(cfg.min_angle_deg, 1.0e-6)
    aspect_term = max(cfg.max_aspect_ratio, 1.0e-6) / max(stats.get("max_aspect_ratio", float("inf")), 1.0e-6)
    area_term = stats.get("min_area", 0.0) / max(area_floor, 1.0e-18)
    non_orth = max(stats.get("max_non_orth_deg", 90.0), 1.0e-6)
    non_orth_term = max(cfg.max_non_orth_deg, 1.0e-6) / non_orth
    return float(min(angle_term, 2.0) + min(aspect_term, 2.0) + min(area_term, 2.0) + min(non_orth_term, 2.0))


def _point_in_polygon(x: float, y: float, ring: Sequence[Tuple[float, float]]) -> bool:
    if len(ring) < 3:
        return False
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i]
        xj, yj = ring[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) if abs(yj - yi) > 1e-15 else 1e-15) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def _bbox_from_ring(ring: Sequence[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return min(xs), min(ys), max(xs), max(ys)


def _ring_centroid_xy(ring: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    n = len(ring)
    if n <= 0:
        return 0.0, 0.0
    return (
        float(sum(p[0] for p in ring) / float(n)),
        float(sum(p[1] for p in ring) / float(n)),
    )


def _cell_overlaps_ring(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    ring: Sequence[Tuple[float, float]],
) -> bool:
    """Conservative overlap probe between an axis-aligned cell and a polygon ring."""
    probes = (
        (x0, y0),
        (x1, y0),
        (x1, y1),
        (x0, y1),
        (0.5 * (x0 + x1), 0.5 * (y0 + y1)),
    )
    for px, py in probes:
        if _point_in_polygon(float(px), float(py), ring):
            return True

    cx, cy = _ring_centroid_xy(ring)
    if x0 < cx < x1 and y0 < cy < y1:
        return True
    return False


def _iter_qgis_polygon_parts(
    geom,
) -> List[Tuple[List[Tuple[float, float]], List[List[Tuple[float, float]]]]]:
    """Extract polygon parts as (outer_ring, [hole_rings]) from QGIS geometry."""
    parts: List[Tuple[List[Tuple[float, float]], List[List[Tuple[float, float]]]]] = []
    if geom is None or geom.isEmpty():
        return parts

    def _ring_xy(points) -> List[Tuple[float, float]]:
        ring = [(float(p.x()), float(p.y())) for p in points[:-1]]
        if len(ring) >= 3 and ring[0] == ring[-1]:
            ring = ring[:-1]
        return ring

    try:
        multi = geom.asMultiPolygon()
        if multi:
            for poly in multi:
                if not poly or not poly[0]:
                    continue
                outer = _ring_xy(poly[0])
                if len(outer) < 3:
                    continue
                holes: List[List[Tuple[float, float]]] = []
                for inner in poly[1:]:
                    hring = _ring_xy(inner)
                    if len(hring) >= 3:
                        holes.append(hring)
                parts.append((outer, holes))
            if parts:
                return parts
    except Exception:
        pass

    try:
        poly = geom.asPolygon()
        if poly and poly[0]:
            outer = _ring_xy(poly[0])
            if len(outer) >= 3:
                holes: List[List[Tuple[float, float]]] = []
                for inner in poly[1:]:
                    hring = _ring_xy(inner)
                    if len(hring) >= 3:
                        holes.append(hring)
                parts.append((outer, holes))
    except Exception:
        pass

    return parts


def _iter_qgis_polygon_outer_rings(geom) -> List[List[Tuple[float, float]]]:
    """Extract outer rings from QGIS Polygon or MultiPolygon geometries."""
    return [outer for outer, _holes in _iter_qgis_polygon_parts(geom)]


def _constraints_for_region(
    model: ConceptualModel,
    region_outer_ring: Sequence[Tuple[float, float]],
) -> List[CellConstraint]:
    out: List[CellConstraint] = []
    for cst in model.constraints:
        if len(cst.ring_xy) < 3:
            continue
        cx, cy = _ring_centroid_xy(cst.ring_xy)
        if _point_in_polygon(cx, cy, region_outer_ring):
            out.append(cst)
    return out


def _region_exclusion_zones(
    model: ConceptualModel,
    region: ConceptualRegion,
    region_outer_ring: Optional[Sequence[Tuple[float, float]]] = None,
) -> List[Tuple[List[Tuple[float, float]], float]]:
    """Return exclusion polygons inside a region as ``(ring, target_size)``.

    Exclusions come from:
    - interior rings of the region polygon,
    - other regions marked ``cell_type=empty``,
    - constraints marked ``cell_type=empty``.
    """
    outer = list(region_outer_ring) if region_outer_ring is not None else list(region.ring_xy)
    if outer and outer[0] == outer[-1]:
        outer = outer[:-1]
    if len(outer) < 3:
        return []

    zones: List[Tuple[List[Tuple[float, float]], float]] = []
    seen = set()

    def _add_zone(ring: Sequence[Tuple[float, float]], size: float) -> None:
        rr = list(ring)
        if rr and rr[0] == rr[-1]:
            rr = rr[:-1]
        if len(rr) < 3:
            return
        cx, cy = _ring_centroid_xy(rr)
        if not _point_in_polygon(cx, cy, outer):
            return
        key = tuple((round(float(x), 7), round(float(y), 7)) for x, y in rr)
        if key in seen:
            return
        seen.add(key)
        zones.append((rr, max(float(size), 1.0e-9)))

    for hring in (region.hole_rings or []):
        _add_zone(hring, region.default_size)

    for candidate in model.regions:
        if candidate is region:
            continue
        if str(candidate.default_cell_type).strip().lower() != "empty":
            continue
        _add_zone(candidate.ring_xy, candidate.default_size)

    for cst in model.constraints:
        if str(cst.cell_type).strip().lower() != "empty":
            continue
        _add_zone(cst.ring_xy, cst.target_size)

    return zones


def _polyline_length(points: Sequence[Tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    length = 0.0
    for i in range(1, len(points)):
        x0, y0 = points[i - 1]
        x1, y1 = points[i]
        length += float(np.hypot(x1 - x0, y1 - y0))
    return length


def _sample_polyline(points: Sequence[Tuple[float, float]], target_size: float) -> List[Tuple[float, float]]:
    if len(points) < 2:
        return list(points)
    step = max(float(target_size), 1.0e-10)
    sampled: List[Tuple[float, float]] = [tuple(points[0])]
    for i in range(1, len(points)):
        x0, y0 = points[i - 1]
        x1, y1 = points[i]
        seg_len = float(np.hypot(x1 - x0, y1 - y0))
        if seg_len <= 1.0e-12:
            continue
        n_div = max(1, int(np.ceil(seg_len / step)))
        for j in range(1, n_div + 1):
            frac = float(j) / float(n_div)
            pt = (x0 + frac * (x1 - x0), y0 + frac * (y1 - y0))
            if np.hypot(pt[0] - sampled[-1][0], pt[1] - sampled[-1][1]) > 1.0e-12:
                sampled.append(pt)
    return sampled


def _rdp_open_polyline(points: Sequence[Tuple[float, float]], tol: float) -> List[Tuple[float, float]]:
    """Ramer-Douglas-Peucker simplification for an open polyline."""
    if len(points) <= 2:
        return list(points)

    p0 = np.asarray(points[0], dtype=np.float64)
    p1 = np.asarray(points[-1], dtype=np.float64)
    seg = p1 - p0
    seg_norm = float(np.hypot(seg[0], seg[1]))

    max_dist = -1.0
    max_idx = -1
    for i in range(1, len(points) - 1):
        p = np.asarray(points[i], dtype=np.float64)
        if seg_norm <= 1.0e-14:
            d = float(np.hypot(*(p - p0)))
        else:
            d = float(abs(seg[0] * (p0[1] - p[1]) - (p0[0] - p[0]) * seg[1]) / seg_norm)
        if d > max_dist:
            max_dist = d
            max_idx = i

    if max_dist <= tol or max_idx < 0:
        return [tuple(points[0]), tuple(points[-1])]

    left = _rdp_open_polyline(points[: max_idx + 1], tol)
    right = _rdp_open_polyline(points[max_idx:], tol)
    return left[:-1] + right


def _simplify_closed_ring(
    ring: Sequence[Tuple[float, float]],
    tol: float,
    max_vertices: Optional[int] = None,
) -> List[Tuple[float, float]]:
    """Simplify a closed polygon ring while preserving closure topology."""
    pts = list(ring)
    if len(pts) < 4:
        return pts
    if pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(pts) < 4:
        return pts

    tol = max(float(tol), 0.0)
    if tol <= 0.0 and (max_vertices is None or len(pts) <= int(max_vertices)):
        return pts

    arr = np.asarray(pts, dtype=np.float64)
    ctr = np.mean(arr, axis=0)
    d2 = np.sum((arr - ctr) ** 2, axis=1)
    i0 = int(np.argmax(d2))

    rotated = pts[i0:] + pts[:i0]
    open_poly = rotated + [rotated[0]]
    simplified = _rdp_open_polyline(open_poly, tol)
    if len(simplified) >= 2 and simplified[0] == simplified[-1]:
        simplified = simplified[:-1]

    if len(simplified) < 3:
        simplified = rotated

    if max_vertices is not None and len(simplified) > int(max_vertices):
        n_keep = max(3, int(max_vertices))
        idx = np.linspace(0, len(simplified) - 1, n_keep, dtype=int)
        simplified = [simplified[int(i)] for i in idx]

    return simplified


def _orient_quad_edge_chains(edges: Sequence[QuadEdgeControl]) -> List[QuadEdgeControl]:
    if len(edges) != 4:
        return list(edges)

    def _score(option: List[QuadEdgeControl]) -> float:
        total = 0.0
        for i in range(len(option)):
            a = option[i].points_xy[-1]
            b = option[(i + 1) % len(option)].points_xy[0]
            total += float(np.hypot(a[0] - b[0], a[1] - b[1]))
        return total

    ordered = sorted(edges, key=lambda edge: int(edge.edge_id))
    candidates: List[List[QuadEdgeControl]] = []
    for reverse_first in (False, True):
        current: List[QuadEdgeControl] = []
        first_points = list(reversed(ordered[0].points_xy)) if reverse_first else list(ordered[0].points_xy)
        current.append(
            QuadEdgeControl(
                region_id=ordered[0].region_id,
                edge_id=ordered[0].edge_id,
                points_xy=first_points,
                target_size=ordered[0].target_size,
                n_layers=ordered[0].n_layers,
                first_height=ordered[0].first_height,
                growth_rate=ordered[0].growth_rate,
            )
        )
        for edge in ordered[1:]:
            forward = list(edge.points_xy)
            reverse = list(reversed(edge.points_xy))
            prev_end = current[-1].points_xy[-1]
            d_fwd = float(np.hypot(prev_end[0] - forward[0][0], prev_end[1] - forward[0][1]))
            d_rev = float(np.hypot(prev_end[0] - reverse[0][0], prev_end[1] - reverse[0][1]))
            points_xy = forward if d_fwd <= d_rev else reverse
            current.append(
                QuadEdgeControl(
                    region_id=edge.region_id,
                    edge_id=edge.edge_id,
                    points_xy=points_xy,
                    target_size=edge.target_size,
                    n_layers=edge.n_layers,
                    first_height=edge.first_height,
                    growth_rate=edge.growth_rate,
                )
            )
        candidates.append(current)
    candidates.sort(key=_score)
    return candidates[0]


def _quad_controls_for_region(
    model: ConceptualModel,
    region: ConceptualRegion,
) -> Optional[Tuple[List[Tuple[float, float]], List[QuadEdgeControl]]]:
    """Build ordered/sampled quad-edge controls for a single region.

    Returns
    -------
    (ring, edges) if the region has a complete 4-edge quad definition.
    None otherwise.
    """
    quad_edges = [edge for edge in model.quad_edges if int(edge.region_id) == int(region.region_id)]
    if len(quad_edges) != 4:
        return None
    if {int(edge.edge_id) for edge in quad_edges} != {1, 2, 3, 4}:
        return None

    oriented = _orient_quad_edge_chains(quad_edges)
    ring: List[Tuple[float, float]] = []
    normalized_edges: List[QuadEdgeControl] = []
    default_edge_sizes = list(region.edge_lengths) if region.edge_lengths and len(region.edge_lengths) == 4 else [region.default_size] * 4
    for edge in oriented:
        edge_size = edge.target_size
        if edge_size is None and 1 <= int(edge.edge_id) <= 4:
            edge_size = float(default_edge_sizes[int(edge.edge_id) - 1])
        if edge_size is None or edge_size <= 0.0:
            edge_size = float(region.default_size)
        sampled = _sample_polyline(edge.points_xy, edge_size)
        if len(sampled) < 2:
            return None
        if ring:
            join = sampled[0]
            prev = ring[-1]
            if np.hypot(join[0] - prev[0], join[1] - prev[1]) <= 1.0e-6:
                sampled = [prev] + sampled[1:]
            ring.extend(sampled[1:])
        else:
            ring.extend(sampled)
        normalized_edges.append(
            QuadEdgeControl(
                region_id=edge.region_id,
                edge_id=edge.edge_id,
                points_xy=sampled,
                target_size=edge_size,
                n_layers=edge.n_layers,
                first_height=edge.first_height,
                growth_rate=edge.growth_rate,
            )
        )

    if len(ring) >= 2 and np.hypot(ring[0][0] - ring[-1][0], ring[0][1] - ring[-1][1]) <= 1.0e-6:
        ring = ring[:-1]
    if len(ring) < 4:
        return None
    area = _polygon_area_xy(
        np.asarray([p[0] for p in ring], dtype=np.float64),
        np.asarray([p[1] for p in ring], dtype=np.float64),
    )
    if area < 0.0:
        ring = list(reversed(ring))
        normalized_edges = [
            QuadEdgeControl(
                region_id=edge.region_id,
                edge_id=edge.edge_id,
                points_xy=list(reversed(edge.points_xy)),
                target_size=edge.target_size,
                n_layers=edge.n_layers,
                first_height=edge.first_height,
                growth_rate=edge.growth_rate,
            )
            for edge in reversed(normalized_edges)
        ]
    return ring, normalized_edges


class MeshingBackend:
    """Backend interface for computational mesh generation.

    A future CGAL-backed implementation should subclass this and implement
    `generate` while preserving the MeshResult output contract.
    """

    name = "base"

    def generate(self, model: ConceptualModel) -> MeshResult:
        raise NotImplementedError()


def _interp_polyline_fraction(points: Sequence[Tuple[float, float]], frac: float) -> Tuple[float, float]:
    if not points:
        return (0.0, 0.0)
    if len(points) == 1:
        return (float(points[0][0]), float(points[0][1]))
    target = max(0.0, min(1.0, float(frac))) * _polyline_length(points)
    if target <= 0.0:
        return (float(points[0][0]), float(points[0][1]))
    total = 0.0
    for i in range(1, len(points)):
        x0, y0 = points[i - 1]
        x1, y1 = points[i]
        seg = float(np.hypot(x1 - x0, y1 - y0))
        if total + seg >= target and seg > 1.0e-15:
            local = (target - total) / seg
            return (x0 + local * (x1 - x0), y0 + local * (y1 - y0))
        total += seg
    return (float(points[-1][0]), float(points[-1][1]))


def _transition_widths(edge: QuadEdgeControl) -> List[float]:
    if edge.n_layers <= 0 or edge.first_height is None or edge.first_height <= 0.0:
        return []
    growth = max(float(edge.growth_rate), 1.0e-6)
    first = max(float(edge.first_height), 1.0e-10)
    return [first * (growth ** i) for i in range(max(0, int(edge.n_layers)))]


def _build_axis_params(
    total_len: float,
    base_target: float,
    start_edge: Optional[QuadEdgeControl],
    end_edge: Optional[QuadEdgeControl],
) -> np.ndarray:
    total_len = max(float(total_len), 1.0e-9)
    base_target = max(float(base_target), 1.0e-9)

    start_widths = _transition_widths(start_edge) if start_edge is not None else []
    end_widths = _transition_widths(end_edge) if end_edge is not None else []
    reserved = float(sum(start_widths) + sum(end_widths))
    if reserved >= total_len * 0.98 and reserved > 0.0:
        scale = (total_len * 0.98) / reserved
        start_widths = [w * scale for w in start_widths]
        end_widths = [w * scale for w in end_widths]
        reserved = float(sum(start_widths) + sum(end_widths))

    middle_len = max(total_len - reserved, 0.0)
    middle_target_candidates = [base_target]
    if start_edge is not None and start_edge.target_size is not None and start_edge.target_size > 0.0:
        middle_target_candidates.append(float(start_edge.target_size))
    if end_edge is not None and end_edge.target_size is not None and end_edge.target_size > 0.0:
        middle_target_candidates.append(float(end_edge.target_size))
    middle_target = max(float(sum(middle_target_candidates) / len(middle_target_candidates)), 1.0e-9)
    middle_count = int(max(0, round(middle_len / middle_target))) if middle_len > 1.0e-12 else 0
    middle_widths = [middle_len / middle_count] * middle_count if middle_count > 0 else []

    widths = list(start_widths) + list(middle_widths) + list(reversed(end_widths))
    if not widths:
        widths = [total_len]
    width_sum = float(sum(widths))
    if width_sum <= 0.0:
        widths = [total_len]
        width_sum = total_len
    scale = total_len / width_sum
    widths = [w * scale for w in widths]

    coords = [0.0]
    acc = 0.0
    for width in widths:
        acc += width
        coords.append(min(acc / total_len, 1.0))
    coords[-1] = 1.0
    return np.asarray(coords, dtype=np.float64)


def _transfinite_quad_point(
    bottom: Sequence[Tuple[float, float]],
    right: Sequence[Tuple[float, float]],
    top: Sequence[Tuple[float, float]],
    left: Sequence[Tuple[float, float]],
    xi: float,
    eta: float,
) -> Tuple[float, float]:
    bx, by = _interp_polyline_fraction(bottom, xi)
    tx, ty = _interp_polyline_fraction(top, xi)
    lx, ly = _interp_polyline_fraction(left, eta)
    rx, ry = _interp_polyline_fraction(right, eta)

    x00, y00 = bottom[0]
    x10, y10 = bottom[-1]
    x01, y01 = top[0]
    x11, y11 = top[-1]

    px = ((1.0 - eta) * bx + eta * tx + (1.0 - xi) * lx + xi * rx)
    py = ((1.0 - eta) * by + eta * ty + (1.0 - xi) * ly + xi * ry)
    px -= (
        (1.0 - xi) * (1.0 - eta) * x00
        + xi * (1.0 - eta) * x10
        + (1.0 - xi) * eta * x01
        + xi * eta * x11
    )
    py -= (
        (1.0 - xi) * (1.0 - eta) * y00
        + xi * (1.0 - eta) * y10
        + (1.0 - xi) * eta * y01
        + xi * eta * y11
    )
    return (float(px), float(py))


def _structured_quad_region_mesh(
    region: ConceptualRegion,
    quad_controls: List[QuadEdgeControl],
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    if len(quad_controls) != 4:
        return None

    bottom = list(quad_controls[0].points_xy)
    right = list(quad_controls[1].points_xy)
    top = list(reversed(quad_controls[2].points_xy))
    left = list(reversed(quad_controls[3].points_xy))
    if len(bottom) < 2 or len(right) < 2 or len(top) < 2 or len(left) < 2:
        return None

    default_edge_sizes = list(region.edge_lengths) if region.edge_lengths and len(region.edge_lengths) == 4 else [region.default_size] * 4
    x_target = max(float(0.5 * (default_edge_sizes[0] + default_edge_sizes[2])), 1.0e-9)
    y_target = max(float(0.5 * (default_edge_sizes[1] + default_edge_sizes[3])), 1.0e-9)
    x_len = 0.5 * (_polyline_length(bottom) + _polyline_length(top))
    y_len = 0.5 * (_polyline_length(left) + _polyline_length(right))

    xi_vals = _build_axis_params(x_len, x_target, quad_controls[3], quad_controls[1])
    eta_vals = _build_axis_params(y_len, y_target, quad_controls[0], quad_controls[2])
    nx = max(1, int(xi_vals.size - 1))
    ny = max(1, int(eta_vals.size - 1))

    node_x: List[float] = []
    node_y: List[float] = []
    face_nodes: List[int] = []
    face_offsets: List[int] = [0]
    tri_nodes: List[int] = []

    def idx(i: int, j: int) -> int:
        return j * (nx + 1) + i

    for j, eta in enumerate(eta_vals):
        for i, xi in enumerate(xi_vals):
            px, py = _transfinite_quad_point(bottom, right, top, left, float(xi), float(eta))
            node_x.append(px)
            node_y.append(py)

    for j in range(ny):
        for i in range(nx):
            n00 = idx(i, j)
            n10 = idx(i + 1, j)
            n01 = idx(i, j + 1)
            n11 = idx(i + 1, j + 1)
            face_nodes.extend([n00, n10, n11, n01])
            face_offsets.append(len(face_nodes))
            tri_nodes.extend([n00, n10, n11, n00, n11, n01])

    cell_count = nx * ny
    target_sizes = np.full(cell_count, float(region.default_size), dtype=np.float64)
    return (
        np.asarray(node_x, dtype=np.float64),
        np.asarray(node_y, dtype=np.float64),
        np.asarray(tri_nodes, dtype=np.int32),
        np.asarray(face_offsets, dtype=np.int32),
        np.asarray(face_nodes, dtype=np.int32),
        target_sizes,
    )


class StructuredFaceCentricBackend(MeshingBackend):
    """Face-centric generator using a structured seed with topology constraints.

    This backend is deterministic, fast, and suitable as a baseline before
    introducing a CGAL constrained Delaunay backend.
    """

    name = "structured-face-centric"

    def generate(self, model: ConceptualModel) -> MeshResult:
        if not model.regions:
            raise ValueError("No conceptual regions provided.")

        all_nodes_x: List[float] = []
        all_nodes_y: List[float] = []
        all_nodes_z: List[float] = []
        all_tris: List[int] = []
        all_face_offsets: List[int] = [0]
        all_face_nodes: List[int] = []
        all_cell_type: List[str] = []
        all_region_id: List[int] = []
        all_size: List[float] = []

        for region in model.regions:
            ring = region.ring_xy
            if len(ring) < 3:
                continue

            if str(region.default_cell_type).strip().lower() == "empty":
                continue

            region_constraints = _constraints_for_region(model, ring)
            region_exclusions = _region_exclusion_zones(model, region, ring)

            if region.default_cell_type in ("cartesian", "quadrilateral") and not region_constraints and not region_exclusions:
                quad_setup = _quad_controls_for_region(model, region)
                if quad_setup is not None:
                    _, quad_controls = quad_setup
                    block = _structured_quad_region_mesh(region, quad_controls)
                    if block is not None:
                        block_x, block_y, block_tris, block_face_offsets, block_face_nodes, block_sizes = block
                        node_offset = len(all_nodes_x)
                        all_nodes_x.extend(block_x.tolist())
                        all_nodes_y.extend(block_y.tolist())
                        all_nodes_z.extend([0.0] * int(block_x.size))
                        all_tris.extend((block_tris + node_offset).tolist())

                        shifted_faces = block_face_nodes + node_offset
                        for cell_idx in range(int(block_face_offsets.size - 1)):
                            s = int(block_face_offsets[cell_idx])
                            e = int(block_face_offsets[cell_idx + 1])
                            all_face_nodes.extend(shifted_faces[s:e].tolist())
                            all_face_offsets.append(len(all_face_nodes))

                        n_cells = int(block_face_offsets.size - 1)
                        all_cell_type.extend([region.default_cell_type] * n_cells)
                        all_region_id.extend([region.region_id] * n_cells)
                        all_size.extend(block_sizes.tolist())
                        continue

            xmin, ymin, xmax, ymax = _bbox_from_ring(ring)
            base_size = max(region.default_size, 1e-6)
            nx = max(1, int(np.ceil((xmax - xmin) / base_size)))
            ny = max(1, int(np.ceil((ymax - ymin) / base_size)))

            dx = (xmax - xmin) / nx if nx > 0 else base_size
            dy = (ymax - ymin) / ny if ny > 0 else base_size

            node_index: Dict[Tuple[int, int], int] = {}

            def idx(i: int, j: int) -> int:
                key = (i, j)
                if key in node_index:
                    return node_index[key]
                x = xmin + i * dx
                y = ymin + j * dy
                node_index[key] = len(all_nodes_x)
                all_nodes_x.append(float(x))
                all_nodes_y.append(float(y))
                all_nodes_z.append(0.0)
                return node_index[key]

            for j in range(ny):
                for i in range(nx):
                    x0 = xmin + i * dx
                    y0 = ymin + j * dy
                    x1 = x0 + dx
                    y1 = y0 + dy
                    cx = xmin + (i + 0.5) * dx
                    cy = ymin + (j + 0.5) * dy
                    if not _point_in_polygon(cx, cy, ring):
                        continue
                    if any(_cell_overlaps_ring(x0, y0, x1, y1, ering) for ering, _esize in region_exclusions):
                        continue

                    local_size = base_size
                    local_type = region.default_cell_type
                    for cst in region_constraints:
                        if str(cst.cell_type).strip().lower() == "empty":
                            continue
                        if _point_in_polygon(cx, cy, cst.ring_xy):
                            local_size = max(cst.target_size, 1e-6)
                            local_type = cst.cell_type

                    if local_type == "empty":
                        continue

                    n00 = idx(i, j)
                    n10 = idx(i + 1, j)
                    n01 = idx(i, j + 1)
                    n11 = idx(i + 1, j + 1)

                    # Solver faces are native polygons; `cell_nodes` remains triangulated
                    # for plotting and layer export compatibility.
                    if local_type in ("cartesian", "quadrilateral"):
                        all_face_nodes.extend([n00, n10, n11, n01])
                        all_face_offsets.append(len(all_face_nodes))
                        all_tris.extend([n00, n10, n11, n00, n11, n01])
                        all_cell_type.append(local_type)
                        all_region_id.append(region.region_id)
                        all_size.append(local_size)
                    elif local_type == "triangular":
                        # Alternating diagonal to reduce directional bias in structured grids.
                        if (i + j) % 2 == 0:
                            all_face_nodes.extend([n00, n10, n11])
                            all_face_offsets.append(len(all_face_nodes))
                            all_face_nodes.extend([n00, n11, n01])
                            all_face_offsets.append(len(all_face_nodes))
                            all_tris.extend([n00, n10, n11, n00, n11, n01])
                        else:
                            all_face_nodes.extend([n00, n10, n01])
                            all_face_offsets.append(len(all_face_nodes))
                            all_face_nodes.extend([n10, n11, n01])
                            all_face_offsets.append(len(all_face_nodes))
                            all_tris.extend([n00, n10, n01, n10, n11, n01])
                        all_cell_type.extend([local_type, local_type])
                        all_region_id.extend([region.region_id, region.region_id])
                        all_size.extend([local_size, local_size])

        if not all_tris:
            raise ValueError("Topology meshing produced no computational cells.")

        node_x = np.asarray(all_nodes_x, dtype=np.float64)
        node_y = np.asarray(all_nodes_y, dtype=np.float64)
        cell_nodes = np.asarray(all_tris, dtype=np.int32)
        face_nodes = np.asarray(all_face_nodes, dtype=np.int32)
        if node_x.size:
            node_x, node_y, (cell_nodes, face_nodes) = _weld_mesh_nodes(node_x, node_y, cell_nodes, face_nodes)

        out = MeshResult(
            node_x=node_x,
            node_y=node_y,
            node_z=np.zeros_like(node_x),
            cell_nodes=cell_nodes,
            cell_face_offsets=np.asarray(all_face_offsets, dtype=np.int32),
            cell_face_nodes=face_nodes,
            cell_type=np.asarray(all_cell_type, dtype=object),
            region_id=np.asarray(all_region_id, dtype=np.int32),
            target_size=np.asarray(all_size, dtype=np.float64),
        )
        return _repair_mesh_result(out)


def _gmsh_available() -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec("gmsh") is not None
    except Exception:
        return False


class GmshBackend(MeshingBackend):
    """Production meshing backend using Gmsh 4.x.

    Geometry mapping:
    - Each ConceptualRegion  -> Gmsh Surface with per-region cell-type flags.
    - Each ConceptualArc     -> Gmsh embedded Curve (breakline/constraint).
    - Each CellConstraint    -> Gmsh Size-field override zone (Threshold field).

    Cell-type controls:
    - "triangular"   : Frontal-Delaunay algorithm (Gmsh algorithm 6).
    - "quadrilateral": Blossom quad recombination on top of Delaunay triangles.
    - "cartesian"    : Transfinite Surface + Recombine (structured grid, fast).
    - "empty"        : Surface excluded from mesh entirely.

    Output:
    - Polygon CSR topology (cell_face_offsets / cell_face_nodes) for the solver.
    - Triangulated cell_nodes (triangles-only fan decomposition) for plotting.
    - cell_type per face reflects the source conceptual type.
    """

    name = "gmsh"

    # Gmsh meshing algorithm codes
    _ALGO_FRONTAL = 6           # Frontal-Delaunay (quality triangles)
    _ALGO_DELAUNAY = 5          # Delaunay (fast fallback)
    _ALGO_PACKING_OF_PARALLELOGRAMS = 9  # good for recombination

    def __init__(self, options: Optional[Dict[str, object]] = None):
        self._options = dict(options or {})

    def _opt_int(self, name: str, default: int) -> int:
        value = self._options.get(name)
        if value is None:
            return int(default)
        try:
            return int(round(float(value)))
        except Exception:
            return int(default)

    def _opt_bool(self, name: str, default: bool) -> bool:
        value = self._options.get(name)
        if value is None:
            return bool(default)
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
        return bool(default)

    def _opt_float(self, name: str, default: float) -> float:
        value = self._options.get(name)
        if value is None:
            return float(default)
        try:
            return float(value)
        except Exception:
            return float(default)

    def _opt_float_tuple(self, name: str, default: Tuple[float, ...]) -> Tuple[float, ...]:
        value = self._options.get(name)
        if value is None:
            return tuple(float(v) for v in default)
        if isinstance(value, str):
            raw_items = value.replace(";", ",").split(",")
        elif isinstance(value, (list, tuple)):
            raw_items = list(value)
        else:
            return tuple(float(v) for v in default)
        parsed: List[float] = []
        for item in raw_items:
            text = str(item).strip()
            if not text:
                continue
            try:
                parsed.append(float(text))
            except Exception:
                continue
        if not parsed:
            return tuple(float(v) for v in default)
        return tuple(parsed)

    def _gmsh_quality_config(self) -> _GmshQualityConfig:
        enabled = self._opt_bool(
            "gmsh_quality_enable",
            _env_bool("BACKWATER_GMSH_QUALITY_ENABLE", False),
        )
        strict = self._opt_bool(
            "gmsh_quality_strict",
            self._opt_bool(
                "tqmesh_quality_strict",
                _env_bool("BACKWATER_GMSH_QUALITY_STRICT", False),
            ),
        )
        min_angle_deg = self._opt_float(
            "gmsh_min_angle_deg",
            self._opt_float(
                "tqmesh_min_angle_deg",
                _env_float("BACKWATER_GMSH_MIN_ANGLE_DEG", 18.0),
            ),
        )
        max_aspect_ratio = self._opt_float(
            "gmsh_max_aspect_ratio",
            self._opt_float(
                "tqmesh_max_aspect_ratio",
                _env_float("BACKWATER_GMSH_MAX_ASPECT_RATIO", 12.0),
            ),
        )
        min_area_rel_bbox = self._opt_float(
            "gmsh_min_area_rel_bbox",
            self._opt_float(
                "tqmesh_min_area_rel_bbox",
                _env_float("BACKWATER_GMSH_MIN_AREA_REL_BBOX", 1.0e-11),
            ),
        )
        max_non_orth_deg = self._opt_float(
            "gmsh_max_non_orth_deg",
            _env_float("BACKWATER_GMSH_MAX_NON_ORTH_DEG", 82.0),
        )
        max_iterations = max(
            1,
            self._opt_int(
                "gmsh_quality_max_iterations",
                int(round(_env_float("BACKWATER_GMSH_QUALITY_MAX_ITERATIONS", 6.0))),
            ),
        )
        time_limit_s = max(
            1.0,
            self._opt_float(
                "gmsh_quality_time_limit_s",
                _env_float("BACKWATER_GMSH_QUALITY_TIME_LIMIT_S", 60.0),
            ),
        )
        size_scales = tuple(
            max(1.0e-3, float(v))
            for v in self._opt_float_tuple(
                "gmsh_quality_size_scales",
                self._opt_float_tuple("tqmesh_size_scales", (1.0, 0.9, 0.8, 0.7)),
            )
        )
        smooth_increments = tuple(
            max(0, int(round(v)))
            for v in self._opt_float_tuple(
                "gmsh_quality_smooth_increments",
                self._opt_float_tuple("tqmesh_smooth_increments", (0.0, 3.0, 6.0)),
            )
        )
        if not size_scales:
            size_scales = (1.0,)
        if not smooth_increments:
            smooth_increments = (0,)
        # Guard against no-op retry ladders (e.g. "1.0" and "0").
        # When iterative quality is enabled, ensure attempts explore distinct
        # candidates even if the UI left legacy single-value defaults.
        if enabled and max_iterations > 1:
            if all(abs(float(v) - 1.0) <= 1.0e-12 for v in size_scales):
                size_scales = (1.0, 0.9, 0.8, 0.7)
            if all(int(v) == 0 for v in smooth_increments):
                smooth_increments = (0, 2, 4, 6)
        return _GmshQualityConfig(
            enabled=enabled,
            strict=strict,
            min_angle_deg=max(1.0, float(min_angle_deg)),
            max_aspect_ratio=max(1.1, float(max_aspect_ratio)),
            min_area_rel_bbox=max(0.0, float(min_area_rel_bbox)),
            max_non_orth_deg=min(89.9, max(1.0, float(max_non_orth_deg))),
            max_iterations=int(max_iterations),
            time_limit_s=float(time_limit_s),
            size_scales=size_scales,
            smooth_increments=smooth_increments,
        )

    def generate(self, model: ConceptualModel) -> MeshResult:
        import gmsh

        if not model.regions:
            raise ValueError("No conceptual regions provided.")

        tri_algo = self._opt_int("gmsh_tri_algorithm", self._ALGO_FRONTAL)
        quad_algo = self._opt_int("gmsh_quad_algorithm", self._ALGO_FRONTAL)
        smoothing_passes = max(0, self._opt_int("gmsh_smoothing", 5))
        optimize_iters = max(0, self._opt_int("gmsh_optimize_iters", 3))
        recomb_algo = self._opt_int("gmsh_recombination_algorithm", 1)
        optimize_netgen = self._opt_bool("gmsh_optimize_netgen", False)
        verbosity = max(0, self._opt_int("gmsh_verbosity", 1))
        quality_cfg = self._gmsh_quality_config()

        # `interruptible=False` avoids installing a SIGINT handler, which lets
        # the Python API run from the QGIS bridge worker thread.
        gmsh.initialize(interruptible=False)
        gmsh.option.setNumber("General.Verbosity", float(verbosity))

        try:
            if not quality_cfg.enabled:
                gmsh.model.add("swe2d")
                return _require_nonempty_mesh(
                    self._build(
                        gmsh,
                        model,
                        tri_algo=tri_algo,
                        quad_algo=quad_algo,
                        smoothing_passes=smoothing_passes,
                        optimize_iters=optimize_iters,
                        recomb_algo=recomb_algo,
                        optimize_netgen=optimize_netgen,
                        size_scale=1.0,
                    ),
                    "Gmsh",
                )

            start_t = time.perf_counter()
            best_mesh: Optional[MeshResult] = None
            best_stats: Optional[Dict[str, float]] = None
            best_score = -1.0e30
            attempts = 0
            scale_i = 0
            smooth_i = 0
            had_passing_candidate = False

            # Alternate between configured and fallback algorithms so retries
            # can escape deterministic local minima with identical topology.
            tri_algo_ladder = [int(tri_algo)]
            tri_alt = self._ALGO_DELAUNAY if int(tri_algo) != self._ALGO_DELAUNAY else self._ALGO_FRONTAL
            if tri_alt not in tri_algo_ladder:
                tri_algo_ladder.append(int(tri_alt))

            quad_algo_ladder = [int(quad_algo)]
            quad_alt = self._ALGO_DELAUNAY if int(quad_algo) != self._ALGO_DELAUNAY else self._ALGO_FRONTAL
            if quad_alt not in quad_algo_ladder:
                quad_algo_ladder.append(int(quad_alt))

            recomb_ladder = [int(recomb_algo)]
            recomb_alt = 0 if int(recomb_algo) != 0 else 1
            if recomb_alt not in recomb_ladder:
                recomb_ladder.append(int(recomb_alt))

            while attempts < quality_cfg.max_iterations:
                elapsed = time.perf_counter() - start_t
                if elapsed >= quality_cfg.time_limit_s:
                    break

                gmsh.clear()
                gmsh.model.add(f"swe2d_try_{attempts + 1}")

                size_scale = quality_cfg.size_scales[scale_i % len(quality_cfg.size_scales)]
                smooth_inc = quality_cfg.smooth_increments[smooth_i % len(quality_cfg.smooth_increments)]
                scale_i += 1
                if scale_i % len(quality_cfg.size_scales) == 0:
                    smooth_i += 1
                tri_try = tri_algo_ladder[attempts % len(tri_algo_ladder)]
                quad_try = quad_algo_ladder[(attempts // len(tri_algo_ladder)) % len(quad_algo_ladder)]
                recomb_try = recomb_ladder[(attempts // max(1, len(tri_algo_ladder) * len(quad_algo_ladder))) % len(recomb_ladder)]

                try:
                    mesh = _require_nonempty_mesh(
                        self._build(
                            gmsh,
                            model,
                            tri_algo=tri_try,
                            quad_algo=quad_try,
                            smoothing_passes=max(0, smoothing_passes + int(smooth_inc)),
                            optimize_iters=optimize_iters,
                            recomb_algo=recomb_try,
                            optimize_netgen=optimize_netgen,
                            size_scale=float(size_scale),
                        ),
                        "Gmsh",
                    )
                    stats = _face_mesh_quality_stats(mesh, quality_cfg)
                    score = _gmsh_quality_score(stats, quality_cfg)

                    if score > best_score:
                        best_score = score
                        best_mesh = mesh
                        best_stats = stats

                    if _gmsh_quality_passes(stats, quality_cfg):
                        had_passing_candidate = True
                        if quality_cfg.strict:
                            # Strict mode only needs the first passing candidate.
                            mesh.quality_summary = {
                                "attempts": int(attempts + 1),
                                "strict_requested": bool(quality_cfg.strict),
                                "had_passing_candidate": True,
                                "best_stats": dict(stats),
                            }
                            return mesh
                except Exception as exc:
                    warnings.warn(
                        f"Gmsh quality attempt {attempts + 1} failed for tri={tri_try}, quad={quad_try}, "
                        f"recomb={recomb_try}, size_scale={size_scale:.3f}, smooth={smoothing_passes + int(smooth_inc)}: {exc}",
                        RuntimeWarning,
                    )
                else:
                    warnings.warn(
                        "Gmsh quality attempt "
                        f"{attempts + 1}: "
                        f"fail_cells(any/angle/aspect/area/non_orth)="
                        f"{int(stats.get('failed_any_cells', 0.0))}/"
                        f"{int(stats.get('failed_min_angle_cells', 0.0))}/"
                        f"{int(stats.get('failed_max_aspect_cells', 0.0))}/"
                        f"{int(stats.get('failed_min_area_cells', 0.0))}/"
                        f"{int(stats.get('failed_max_non_orth_cells', 0.0))}",
                        RuntimeWarning,
                    )

                attempts += 1

            if best_mesh is None or best_stats is None:
                raise RuntimeError("Gmsh quality loop produced no valid non-empty mesh candidate.")

            if had_passing_candidate:
                best_mesh.quality_summary = {
                    "attempts": int(attempts),
                    "strict_requested": bool(quality_cfg.strict),
                    "had_passing_candidate": True,
                    "best_stats": dict(best_stats),
                }
                return best_mesh

            diag = (
                "min_angle={:.2f} deg, max_aspect={:.2f}, min_area={:.3e}, max_non_orth={:.2f} deg"
                .format(
                    float(best_stats.get("min_angle_deg", 0.0)),
                    float(best_stats.get("max_aspect_ratio", float("inf"))),
                    float(best_stats.get("min_area", 0.0)),
                    float(best_stats.get("max_non_orth_deg", 90.0)),
                )
            )
            summary = {
                "attempts": int(attempts),
                "strict_requested": bool(quality_cfg.strict),
                "had_passing_candidate": False,
                "best_stats": dict(best_stats),
            }
            best_mesh.quality_summary = summary
            warnings.warn(
                "Gmsh quality constraints were not met; using best available candidate "
                f"(attempts={attempts}, time_limit_s={quality_cfg.time_limit_s:.1f}). {diag}",
                RuntimeWarning,
            )
            return best_mesh
        finally:
            gmsh.finalize()

    # ------------------------------------------------------------------
    # Internal construction helpers
    # ------------------------------------------------------------------

    def _build(
        self,
        gmsh,
        model: ConceptualModel,
        tri_algo: int,
        quad_algo: int,
        smoothing_passes: int,
        optimize_iters: int,
        recomb_algo: int,
        optimize_netgen: bool,
        size_scale: float,
    ) -> MeshResult:
        # Tolerance for point deduplication (scaled to typical hydraulic coords).
        tol = 1e-6
        surface_tags: List[int] = []
        surface_meta: List[Tuple[int, str, float]] = []  # (region_id, cell_type, target_size)
        surface_curve_tags: Dict[int, List[int]] = {}
        surface_quad_controls: Dict[int, Optional[List[QuadEdgeControl]]] = {}

        # Shared geometry registries for conforming inter-region interfaces.
        # Points and single-segment lines on shared boundaries are reused so
        # Gmsh meshes that interface curve exactly once.  Without this, each
        # region independently creates duplicate points/curves at the same
        # physical location; Gmsh then discretises the shared edge twice with
        # potentially different node counts, producing hanging nodes that
        # immediately destabilise the FVM solver.
        _pt_prec = 6  # rounding digits ≈ 1 µm — sufficient for hydraulic coords
        pt_reg: Dict[Tuple[float, float], int] = {}   # (rx,ry) -> gmsh point tag
        seg_reg: Dict[Tuple[int, int], int] = {}       # (p0,p1) -> signed curve tag

        def _geo_pt(x: float, y: float, lc: float) -> int:
            """Return existing gmsh point tag at (x,y) or create a new one."""
            key = (round(float(x), _pt_prec), round(float(y), _pt_prec))
            if key in pt_reg:
                return pt_reg[key]
            tag = gmsh.model.geo.addPoint(float(x), float(y), 0.0, lc)
            pt_reg[key] = tag
            return tag

        def _geo_seg(p0: int, p1: int) -> int:
            """Return signed line tag for directed segment p0->p1, sharing if it
            already exists in either direction."""
            if (p0, p1) in seg_reg:
                return seg_reg[(p0, p1)]
            if (p1, p0) in seg_reg:
                return -seg_reg[(p1, p0)]
            tag = gmsh.model.geo.addLine(p0, p1)
            seg_reg[(p0, p1)] = tag
            return tag

        # ---- 1. Build one Gmsh surface per region ----------------------
        for region in model.regions:
            ring = list(region.ring_xy)
            if ring and ring[0] == ring[-1]:
                ring = ring[:-1]
            if len(ring) < 3:
                continue

            ctype = str(region.default_cell_type).strip().lower()
            if ctype == "empty":
                continue
            region_size = max(float(region.default_size) * float(size_scale), 1.0e-9)

            quad_controls = None
            if ctype in ("quadrilateral", "cartesian"):
                quad_setup = _quad_controls_for_region(model, region)
                if quad_setup is not None:
                    ring, quad_controls = quad_setup

            lines: List[int] = []
            if quad_controls is not None:
                first_pt_tag: Optional[int] = None
                first_xy: Optional[Tuple[float, float]] = None
                prev_end_tag: Optional[int] = None
                for ei, edge in enumerate(quad_controls):
                    edge_pts = list(edge.points_xy)
                    if len(edge_pts) < 2:
                        continue
                    edge_lc = float(edge.target_size) * float(size_scale) if (edge.target_size is not None and edge.target_size > 0.0) else float(region_size)
                    edge_tags: List[int] = []
                    for pj, (x, y) in enumerate(edge_pts):
                        if ei > 0 and pj == 0 and prev_end_tag is not None:
                            edge_tags.append(prev_end_tag)
                            continue
                        if ei == len(quad_controls) - 1 and pj == len(edge_pts) - 1 and first_pt_tag is not None and first_xy is not None:
                            if np.hypot(x - first_xy[0], y - first_xy[1]) <= tol:
                                edge_tags.append(first_pt_tag)
                                continue
                        ptag = _geo_pt(x, y, edge_lc)
                        edge_tags.append(ptag)
                        if first_pt_tag is None:
                            first_pt_tag = ptag
                            first_xy = (float(x), float(y))
                    if len(edge_tags) < 2:
                        continue
                    try:
                        # Share single-segment edges via _geo_seg so adjacent
                        # regions referencing the same boundary line reuse the
                        # same Gmsh curve tag (possibly negated for direction).
                        curve = gmsh.model.geo.addSpline(edge_tags) if len(edge_tags) > 2 else _geo_seg(edge_tags[0], edge_tags[1])
                        lines.append(curve)
                    except Exception:
                        for k in range(len(edge_tags) - 1):
                            lines.append(_geo_seg(edge_tags[k], edge_tags[k + 1]))
                    prev_end_tag = edge_tags[-1]
                if first_pt_tag is not None and prev_end_tag is not None and prev_end_tag != first_pt_tag:
                    lines.append(_geo_seg(prev_end_tag, first_pt_tag))
            else:
                pts = [_geo_pt(x, y, region_size) for x, y in ring]
                for i in range(len(pts)):
                    lines.append(_geo_seg(pts[i], pts[(i + 1) % len(pts)]))

            if len(lines) < 3:
                continue

            loop = gmsh.model.geo.addCurveLoop(lines)
            hole_loops: List[int] = []
            exclusion_zones = _region_exclusion_zones(model, region, ring)
            if exclusion_zones:
                outer_area = _polygon_area_xy(
                    np.asarray([p[0] for p in ring], dtype=np.float64),
                    np.asarray([p[1] for p in ring], dtype=np.float64),
                )
                outer_ccw = bool(outer_area > 0.0)
                for ering, esize in exclusion_zones:
                    hring = list(ering)
                    if hring and hring[0] == hring[-1]:
                        hring = hring[:-1]
                    if len(hring) < 3:
                        continue

                    h_area = _polygon_area_xy(
                        np.asarray([p[0] for p in hring], dtype=np.float64),
                        np.asarray([p[1] for p in hring], dtype=np.float64),
                    )
                    if bool(h_area > 0.0) == outer_ccw:
                        hring = list(reversed(hring))

                    hole_size = max(float(esize) * float(size_scale), 1.0e-9)
                    hole_pts = [_geo_pt(x, y, hole_size) for x, y in hring]
                    if len(hole_pts) < 3:
                        continue
                    hlines: List[int] = []
                    for i in range(len(hole_pts)):
                        hlines.append(_geo_seg(hole_pts[i], hole_pts[(i + 1) % len(hole_pts)]))
                    if len(hlines) < 3:
                        continue
                    try:
                        hole_loops.append(gmsh.model.geo.addCurveLoop(hlines))
                    except Exception:
                        pass

            surf = gmsh.model.geo.addPlaneSurface([loop] + hole_loops)
            surface_tags.append(surf)
            surface_meta.append((region.region_id, ctype, region_size))
            surface_curve_tags[surf] = lines
            surface_quad_controls[surf] = quad_controls

        if not surface_tags:
            raise ValueError("GmshBackend: no non-empty regions to mesh.")

        # ---- 2. Embed arc breaklines into surfaces ----------------------
        if model.arcs:
            arc_curve_tags: List[int] = []
            # Build a quick node-id -> (x,y) lookup
            node_xy = {n.node_id: (n.x, n.y) for n in model.nodes}
            arc_lc = min(
                (
                    max(float(r.default_size) * float(size_scale), 1.0e-9)
                    for r in model.regions
                    if str(r.default_cell_type).strip().lower() != "empty"
                ),
                default=1.0,
            )
            for arc in model.arcs:
                pts_xy = list(arc.points_xy or [])
                if len(pts_xy) >= 2:
                    gp_tags: List[int] = []
                    for x, y in pts_xy:
                        ptag = _geo_pt(float(x), float(y), arc_lc)
                        if not gp_tags or gp_tags[-1] != ptag:
                            gp_tags.append(ptag)
                    for i in range(len(gp_tags) - 1):
                        seg = _geo_seg(gp_tags[i], gp_tags[i + 1])
                        arc_curve_tags.append(abs(int(seg)))
                    continue

                # Backward-compatible fallback: endpoint IDs in topo_nodes.
                p0_xy = node_xy.get(arc.node0)
                p1_xy = node_xy.get(arc.node1)
                if p0_xy is None or p1_xy is None:
                    continue
                gp0 = _geo_pt(p0_xy[0], p0_xy[1], arc_lc)
                gp1 = _geo_pt(p1_xy[0], p1_xy[1], arc_lc)
                arc_curve_tags.append(abs(int(_geo_seg(gp0, gp1))))

            if arc_curve_tags:
                arc_curve_tags = sorted({int(tag) for tag in arc_curve_tags if int(tag) > 0})
                gmsh.model.geo.synchronize()
                for surf in surface_tags:
                    try:
                        gmsh.model.mesh.embed(1, arc_curve_tags, 2, surf)
                    except Exception:
                        pass  # arc may not intersect this surface; skip

        gmsh.model.geo.synchronize()

        # ---- 3. Constraint refinement zones (background field) ----------
        # Build a region baseline size field and overlay per-constraint
        # threshold fields derived from polygon-clipped point sampling.
        # This is stronger than pure point embedding and enforces local sizing.
        base_surface_fields: List[int] = []
        for surf, (_, _, sz) in zip(surface_tags, surface_meta):
            f_const = gmsh.model.mesh.field.add("MathEval")
            gmsh.model.mesh.field.setString(f_const, "F", f"{max(float(sz), 1.0e-9):.16g}")
            f_restrict = gmsh.model.mesh.field.add("Restrict")
            gmsh.model.mesh.field.setNumber(f_restrict, "InField", float(f_const))
            gmsh.model.mesh.field.setNumbers(f_restrict, "SurfacesList", [int(surf)])
            base_surface_fields.append(f_restrict)

        constraint_point_lists: List[List[int]] = []
        constraint_target_sizes: List[float] = []
        for cst in model.constraints:
            if len(cst.ring_xy) < 3 or str(cst.cell_type).strip().lower() == "empty":
                continue
            ring = list(cst.ring_xy)
            if ring[0] == ring[-1]:
                ring = ring[:-1]
            if len(ring) < 3:
                continue

            pt_tags: List[int] = []
            cst_size = max(float(cst.target_size) * float(size_scale), 1.0e-9)

            # Boundary samples.
            for x, y in ring:
                try:
                    pt_tags.append(gmsh.model.geo.addPoint(float(x), float(y), 0.0, cst_size))
                except Exception:
                    pass

            # Interior samples clipped to the polygon footprint.
            #
            # Important: avoid one-sided sampling truncation. The previous
            # implementation stopped after a fixed point cap while scanning
            # ymin->ymax, which could leave only part of a large constraint
            # polygon refined. Here we choose an area-adaptive step so sampling
            # remains approximately bounded while covering the full polygon.
            xs = [p[0] for p in ring]
            ys = [p[1] for p in ring]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)

            base_step = max(cst_size, tol * 10.0)
            target_pts = 6000.0
            poly_area = abs(_polygon_area_xy(
                np.asarray(xs, dtype=np.float64),
                np.asarray(ys, dtype=np.float64),
            ))
            if poly_area > 0.0:
                step = max(base_step, float(np.sqrt(poly_area / target_pts)))
            else:
                step = base_step

            y = ymin + 0.5 * step
            while y < ymax - 0.5 * step:
                x = xmin + 0.5 * step
                while x < xmax - 0.5 * step:
                    if _point_in_polygon(x, y, ring):
                        try:
                            pt_tags.append(gmsh.model.geo.addPoint(float(x), float(y), 0.0, cst_size))
                        except Exception:
                            pass
                    x += step
                y += step

            dedup_tags = list(dict.fromkeys(pt_tags))
            if dedup_tags:
                constraint_point_lists.append(dedup_tags)
                constraint_target_sizes.append(cst_size)

        gmsh.model.geo.synchronize()

        if constraint_point_lists:
            all_fields: List[int] = list(base_surface_fields)
            max_region_size = max(max(float(sz), 1.0e-9) for (_, _, sz) in surface_meta)
            for pt_list, cst_size in zip(constraint_point_lists, constraint_target_sizes):
                f_dist = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(f_dist, "PointsList", [int(t) for t in pt_list])

                f_thresh = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(f_thresh, "InField", float(f_dist))
                gmsh.model.mesh.field.setNumber(f_thresh, "SizeMin", float(cst_size))
                gmsh.model.mesh.field.setNumber(f_thresh, "SizeMax", float(max_region_size))
                gmsh.model.mesh.field.setNumber(f_thresh, "DistMin", 0.0)
                gmsh.model.mesh.field.setNumber(f_thresh, "DistMax", float(1.5 * cst_size))
                gmsh.model.mesh.field.setNumber(f_thresh, "StopAtDistMax", 1.0)

                f_restrict = gmsh.model.mesh.field.add("Restrict")
                gmsh.model.mesh.field.setNumber(f_restrict, "InField", float(f_thresh))
                gmsh.model.mesh.field.setNumbers(f_restrict, "SurfacesList", [int(s) for s in surface_tags])
                all_fields.append(f_restrict)

            if len(all_fields) == 1:
                bg_field = all_fields[0]
            else:
                bg_field = gmsh.model.mesh.field.add("Min")
                gmsh.model.mesh.field.setNumbers(bg_field, "FieldsList", [int(fid) for fid in all_fields])

            gmsh.model.mesh.field.setAsBackgroundMesh(int(bg_field))
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1.0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0.0)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0.0)

        # ---- 4. Per-surface algorithm and recombination flags ----------
        want_recombine = False
        for surf, (rid, ctype, sz) in zip(surface_tags, surface_meta):
            region = next((r for r in model.regions if int(r.region_id) == int(rid)), None)
            lines = surface_curve_tags.get(surf, [])
            quad_controls = surface_quad_controls.get(surf)
            if ctype == "cartesian":
                # Transfinite + Recombine: structured, fast, pure quads.
                if region is not None and region.edge_lengths and len(lines) == 4 and len(region.edge_lengths) == 4:
                    try:
                        edge_geom_len = []
                        if quad_controls is not None and len(quad_controls) == 4:
                            edge_geom_len = [_polyline_length(edge.points_xy) for edge in quad_controls]
                        else:
                            p_ring = list(region.ring_xy)
                            if p_ring and p_ring[0] == p_ring[-1]:
                                p_ring = p_ring[:-1]
                            for i in range(4):
                                x0, y0 = p_ring[i]
                                x1, y1 = p_ring[(i + 1) % 4]
                                edge_geom_len.append(float(np.hypot(x1 - x0, y1 - y0)))
                        counts = []
                        for i in range(4):
                            tlen = max(float(region.edge_lengths[i]), tol)
                            ndiv = max(1, int(round(edge_geom_len[i] / tlen)))
                            counts.append(max(2, ndiv + 1))

                        # Opposite edges must match for transfinite surface.
                        n0 = max(counts[0], counts[2])
                        n1 = max(counts[1], counts[3])
                        counts[0] = counts[2] = n0
                        counts[1] = counts[3] = n1

                        for ltag, npt in zip(lines, counts):
                            # abs(): shared reversed curves carry negative tags
                            gmsh.model.mesh.setTransfiniteCurve(abs(ltag), int(npt))
                        gmsh.model.mesh.setTransfiniteSurface(surf)
                    except Exception:
                        try:
                            gmsh.model.mesh.setTransfiniteSurface(surf)
                        except Exception:
                            pass
                else:
                    try:
                        gmsh.model.mesh.setTransfiniteSurface(surf)
                    except Exception:
                        pass  # Works best for 4-sided surfaces.
                gmsh.model.mesh.setRecombine(2, surf)
                want_recombine = True
                # Packing of Parallelograms requires a scaled cross field and
                # is brittle on real project geometries.  For structured quad
                # surfaces, transfinite constraints plus recombination are the
                # controlling inputs; keep the base 2D algorithm on the safer
                # frontal path.
                try:
                    gmsh.model.mesh.setAlgorithm(2, surf, quad_algo)
                except Exception:
                    gmsh.option.setNumber("Mesh.Algorithm", float(quad_algo))
            elif ctype == "quadrilateral":
                # Unstructured quads via Blossom recombination.
                if region is not None and region.edge_lengths and len(lines) == 4 and len(region.edge_lengths) == 4:
                    try:
                        edge_geom_len = []
                        if quad_controls is not None and len(quad_controls) == 4:
                            edge_geom_len = [_polyline_length(edge.points_xy) for edge in quad_controls]
                        else:
                            p_ring = list(region.ring_xy)
                            if p_ring and p_ring[0] == p_ring[-1]:
                                p_ring = p_ring[:-1]
                            for i in range(4):
                                x0, y0 = p_ring[i]
                                x1, y1 = p_ring[(i + 1) % 4]
                                edge_geom_len.append(float(np.hypot(x1 - x0, y1 - y0)))
                        counts = []
                        for i in range(4):
                            tlen = max(float(region.edge_lengths[i]), tol)
                            ndiv = max(1, int(round(edge_geom_len[i] / tlen)))
                            counts.append(max(2, ndiv + 1))
                        n0 = max(counts[0], counts[2])
                        n1 = max(counts[1], counts[3])
                        counts[0] = counts[2] = n0
                        counts[1] = counts[3] = n1
                        for ltag, npt in zip(lines, counts):
                            gmsh.model.mesh.setTransfiniteCurve(abs(ltag), int(npt))
                        gmsh.model.mesh.setTransfiniteSurface(surf)
                    except Exception:
                        pass
                gmsh.model.mesh.setRecombine(2, surf)
                want_recombine = True
                # For general quad regions, generate triangles with the frontal
                # algorithm and let Blossom handle recombination.  This avoids
                # the scaled-cross-field requirement that triggers terminal
                # errors like: "Packing of Parallelograms require a scaled
                # cross field".
                try:
                    gmsh.model.mesh.setAlgorithm(2, surf, quad_algo)
                except Exception:
                    gmsh.option.setNumber("Mesh.Algorithm", float(quad_algo))
            else:
                # triangular: frontal Delaunay for quality.
                try:
                    gmsh.model.mesh.setAlgorithm(2, surf, tri_algo)
                except Exception:
                    gmsh.option.setNumber("Mesh.Algorithm", float(tri_algo))

        # ---- 5. Global mesh options ------------------------------------
        gmsh.option.setNumber("Mesh.RecombineAll", 0)          # per-surface only
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", float(recomb_algo))
        gmsh.option.setNumber("Mesh.Smoothing", float(smoothing_passes))
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1.0 if optimize_netgen else 0.0)

        # ---- 6. Generate -----------------------------------------------
        gmsh.model.mesh.generate(2)
        if want_recombine:
            try:
                gmsh.model.mesh.recombine()
            except Exception:
                pass
        if optimize_iters > 0:
            gmsh.model.mesh.optimize("Laplace2D", niter=optimize_iters)

        # ---- 7. Extract nodes ------------------------------------------
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        # node_coords: flat [x0,y0,z0, x1,y1,z1, ...]
        node_coords = np.array(node_coords, dtype=np.float64).reshape(-1, 3)
        tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}
        node_x = node_coords[:, 0].copy()
        node_y = node_coords[:, 1].copy()
        node_z = np.zeros(node_x.shape[0], dtype=np.float64)

        # ---- 8. Extract elements per surface with metadata -------------
        all_face_offsets: List[int] = [0]
        all_face_nodes: List[int] = []
        all_tris: List[int] = []
        all_cell_type: List[str] = []
        all_region_id: List[int] = []
        all_size: List[float] = []

        # Gmsh element type codes: 2 = 3-node triangle, 3 = 4-node quad
        for surf, (rid, ctype, sz) in zip(surface_tags, surface_meta):
            elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2, surf)
            for etype, _, enodes in zip(elem_types, elem_tags, elem_node_tags):
                enodes = np.array(enodes, dtype=np.int64)
                if etype == 2:  # triangle
                    n_elems = len(enodes) // 3
                    enodes = enodes.reshape(n_elems, 3)
                    for tri in enodes:
                        v = [tag_to_idx[int(t)] for t in tri]
                        all_face_nodes.extend(v)
                        all_face_offsets.append(len(all_face_nodes))
                        all_tris.extend(v)
                        all_cell_type.append(ctype)
                        all_region_id.append(rid)
                        all_size.append(sz)
                elif etype == 3:  # quad
                    n_elems = len(enodes) // 4
                    enodes = enodes.reshape(n_elems, 4)
                    for quad in enodes:
                        v = [tag_to_idx[int(t)] for t in quad]
                        all_face_nodes.extend(v)
                        all_face_offsets.append(len(all_face_nodes))
                        # Fan-triangulate for plotting: 0-1-2, 0-2-3
                        all_tris.extend([v[0], v[1], v[2], v[0], v[2], v[3]])
                        all_cell_type.append(ctype)
                        all_region_id.append(rid)
                        all_size.append(sz)

        if not all_face_offsets or len(all_face_offsets) == 1:
            raise ValueError("GmshBackend: no elements extracted from mesh.")

        out = MeshResult(
            node_x=node_x,
            node_y=node_y,
            node_z=node_z,
            cell_nodes=np.asarray(all_tris, dtype=np.int32),
            cell_face_offsets=np.asarray(all_face_offsets, dtype=np.int32),
            cell_face_nodes=np.asarray(all_face_nodes, dtype=np.int32),
            cell_type=np.asarray(all_cell_type, dtype=object),
            region_id=np.asarray(all_region_id, dtype=np.int32),
            target_size=np.asarray(all_size, dtype=np.float64),
        )
        return _repair_mesh_result(out)


def conceptual_from_qgis_layers(
    nodes_layer,
    arcs_layer,
    regions_layer,
    constraints_layer=None,
    quad_edges_layer=None,
    default_size: float = 20.0,
    default_cell_type: str = "triangular",
) -> ConceptualModel:
    """Build conceptual topology model from QGIS layers.

        Expected fields (optional unless noted):
        - nodes: node_id
        - arcs: arc_id
            - breakline is read from arc geometry vertices (preferred)
            - node0/node1 are optional fallback endpoints
    - regions (required geometry): region_id, target_size, cell_type
    - constraints: constraint_id, target_size, cell_type
    - quad_edges: region_id, edge_id, target_size, n_layers, first_height, growth_rate
    """
    if regions_layer is None:
        raise ValueError("regions layer is required for topology meshing")

    nodes: List[ConceptualNode] = []
    arcs: List[ConceptualArc] = []
    regions: List[ConceptualRegion] = []
    constraints: List[CellConstraint] = []
    quad_edges: List[QuadEdgeControl] = []

    if nodes_layer is not None:
        node_fields = set(nodes_layer.fields().names())
        auto_id = 0
        for ft in nodes_layer.getFeatures():
            geom = ft.geometry()
            if geom is None or geom.isEmpty():
                continue
            pt = geom.asPoint()
            nid = _as_int(ft["node_id"], auto_id) if "node_id" in node_fields else auto_id
            nodes.append(ConceptualNode(node_id=nid, x=float(pt.x()), y=float(pt.y())))
            auto_id += 1

    if arcs_layer is not None:
        arc_fields = set(arcs_layer.fields().names())
        auto_id = 0
        for ft in arcs_layer.getFeatures():
            geom = ft.geometry()
            pts: List[Tuple[float, float]] = []
            if geom is not None and not geom.isEmpty():
                try:
                    line = geom.asPolyline()
                    if line:
                        pts = [(float(p.x()), float(p.y())) for p in line]
                except Exception:
                    pts = []
                if not pts:
                    try:
                        multi = geom.asMultiPolyline()
                        if multi and multi[0]:
                            pts = [(float(p.x()), float(p.y())) for p in multi[0]]
                    except Exception:
                        pts = []
            a_id = _as_int(ft["arc_id"], auto_id) if "arc_id" in arc_fields else auto_id
            n0 = _as_int(ft["node0"], -1) if "node0" in arc_fields else -1
            n1 = _as_int(ft["node1"], -1) if "node1" in arc_fields else -1
            arcs.append(
                ConceptualArc(
                    arc_id=a_id,
                    node0=n0,
                    node1=n1,
                    points_xy=pts if len(pts) >= 2 else None,
                )
            )
            auto_id += 1

    region_fields = set(regions_layer.fields().names())
    auto_rid = 0
    for ft in regions_layer.getFeatures():
        geom = ft.geometry()
        if geom is None or geom.isEmpty():
            continue
        parts = _iter_qgis_polygon_parts(geom)
        if not parts:
            continue
        rid = _as_int(ft["region_id"], auto_rid) if "region_id" in region_fields else auto_rid
        size = _as_float(ft["target_size"], default_size) if "target_size" in region_fields else default_size
        ctype = _normalize_cell_type(ft["cell_type"], default_cell_type) if "cell_type" in region_fields else default_cell_type
        edge_lengths = None
        edge_fields = []
        for prefix in ("edge_len_", "cell_len_"):
            cand = [f"{prefix}{i}" for i in (1, 2, 3, 4)]
            if all(c in region_fields for c in cand):
                edge_fields = cand
                break
        if edge_fields:
            vals = [_as_float(ft[nm], size) for nm in edge_fields]
            if all(v > 0 for v in vals):
                edge_lengths = vals
        for part_idx, (ring, holes) in enumerate(parts):
            part_rid = rid if part_idx == 0 else int(f"{rid}{part_idx}")
            regions.append(
                ConceptualRegion(
                    region_id=part_rid,
                    ring_xy=ring,
                    default_size=size,
                    default_cell_type=ctype,
                    edge_lengths=edge_lengths,
                    hole_rings=holes,
                )
            )
        auto_rid += 1

    if constraints_layer is not None:
        c_fields = set(constraints_layer.fields().names())
        auto_cid = 0
        for ft in constraints_layer.getFeatures():
            geom = ft.geometry()
            if geom is None or geom.isEmpty():
                continue
            parts = _iter_qgis_polygon_parts(geom)
            if not parts:
                continue
            cid = _as_int(ft["constraint_id"], auto_cid) if "constraint_id" in c_fields else auto_cid
            size = _as_float(ft["target_size"], default_size) if "target_size" in c_fields else default_size
            ctype = _normalize_cell_type(ft["cell_type"], default_cell_type) if "cell_type" in c_fields else default_cell_type
            for part_idx, (ring, _holes) in enumerate(parts):
                part_cid = cid if part_idx == 0 else int(f"{cid}{part_idx}")
                constraints.append(CellConstraint(constraint_id=part_cid, ring_xy=ring, target_size=size, cell_type=ctype))
            auto_cid += 1

    if quad_edges_layer is not None:
        q_fields = set(quad_edges_layer.fields().names())
        auto_edge_id = 1
        for ft in quad_edges_layer.getFeatures():
            geom = ft.geometry()
            if geom is None or geom.isEmpty():
                continue

            pts: List[Tuple[float, float]] = []
            try:
                line = geom.asPolyline()
                if line:
                    pts = [(float(p.x()), float(p.y())) for p in line]
            except Exception:
                pts = []
            if not pts:
                try:
                    multi = geom.asMultiPolyline()
                    if multi and multi[0]:
                        pts = [(float(p.x()), float(p.y())) for p in multi[0]]
                except Exception:
                    pts = []
            if len(pts) < 2:
                continue

            region_id = _as_int(ft["region_id"], -1) if "region_id" in q_fields else -1
            edge_id = _as_int(ft["edge_id"], auto_edge_id) if "edge_id" in q_fields else auto_edge_id
            target_size = _as_float(ft["target_size"], np.nan) if "target_size" in q_fields else np.nan
            n_layers = _as_int(ft["n_layers"], 0) if "n_layers" in q_fields else 0
            first_height = _as_float(ft["first_height"], np.nan) if "first_height" in q_fields else np.nan
            growth_rate = _as_float(ft["growth_rate"], 1.0) if "growth_rate" in q_fields else 1.0

            quad_edges.append(
                QuadEdgeControl(
                    region_id=region_id,
                    edge_id=edge_id,
                    points_xy=pts,
                    target_size=(None if not np.isfinite(target_size) or target_size <= 0.0 else float(target_size)),
                    n_layers=max(0, int(n_layers)),
                    first_height=(None if not np.isfinite(first_height) or first_height <= 0.0 else float(first_height)),
                    growth_rate=max(float(growth_rate), 1.0e-6),
                )
            )
            auto_edge_id += 1

    if not regions:
        raise ValueError("No valid regions in topology layer.")

    return ConceptualModel(
        nodes=nodes,
        arcs=arcs,
        regions=regions,
        constraints=constraints,
        quad_edges=quad_edges,
    )


def _tqmesh_available() -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec("hydra_tqmesh") is not None
    except Exception:
        return False


class TQMeshBackend(MeshingBackend):
    """Mesh generator backed by TQMesh's advancing-front algorithm.

    Advantages over Gmsh for 2D SWE meshes:
    - Advancing-front naturally produces well-shaped triangles, avoiding
      near-zero-area (degenerate) cells that cause FVM overflow.
    - Single-include C++ header library — no external process or dependency.
    - Per-vertex local size hints support smooth size grading without
      post-process size fields.
    - Fixed interior vertices / constraint zones natively supported.

    Requires the ``hydra_tqmesh`` C++ extension module built from
    ``cpp/src/tqmesh_bindings.cpp``.
    """

    name = "tqmesh"

    def __init__(self, options: Optional[Dict[str, object]] = None):
        self._options = dict(options or {})

    def _quality_config(self) -> _TQMeshQualityConfig:
        options = self._options

        def _opt_float(name: str, default: float, min_value: float) -> float:
            value = options.get(name)
            if value is None:
                return max(default, min_value)
            try:
                return max(float(value), min_value)
            except Exception:
                return max(default, min_value)

        def _opt_bool(name: str, default: bool) -> bool:
            value = options.get(name)
            if value is None:
                return bool(default)
            if isinstance(value, bool):
                return value
            text = str(value).strip().lower()
            if text in {"1", "true", "yes", "on"}:
                return True
            if text in {"0", "false", "no", "off"}:
                return False
            return bool(default)

        def _opt_float_tuple(name: str, default: Tuple[float, ...]) -> Tuple[float, ...]:
            value = options.get(name)
            if value is None:
                return tuple(float(v) for v in default)
            if isinstance(value, str):
                parts = [p.strip() for p in value.split(",") if p.strip()]
                vals = []
                for part in parts:
                    try:
                        vals.append(float(part))
                    except Exception:
                        continue
                return tuple(vals) or tuple(float(v) for v in default)
            try:
                vals = [float(v) for v in value]  # type: ignore[arg-type]
            except Exception:
                return tuple(float(v) for v in default)
            return tuple(vals) or tuple(float(v) for v in default)

        def _opt_int_tuple(name: str, default: Tuple[int, ...]) -> Tuple[int, ...]:
            value = options.get(name)
            if value is None:
                return tuple(int(v) for v in default)
            if isinstance(value, str):
                parts = [p.strip() for p in value.split(",") if p.strip()]
                vals = []
                for part in parts:
                    try:
                        vals.append(int(round(float(part))))
                    except Exception:
                        continue
                return tuple(vals) or tuple(int(v) for v in default)
            try:
                vals = [int(round(float(v))) for v in value]  # type: ignore[arg-type]
            except Exception:
                return tuple(int(v) for v in default)
            return tuple(vals) or tuple(int(v) for v in default)

        return _TQMeshQualityConfig(
            # Quality targets are INFORMATIONAL in non-strict mode — they drive
            # warnings and best-candidate selection but will not exhaust the
            # retry ladder.  The real retry trigger is TQMesh completeness
            # failure (RuntimeError from the C++ binding).
            #
            # Defaults are intentionally relaxed so that most real-world
            # watershed polygons pass on the first attempt.  Tighten via env
            # vars only if you need a high-quality mesh and are willing to wait.
            min_angle_deg=_opt_float(
                "tqmesh_min_angle_deg",
                _env_float("BACKWATER_TQMESH_MIN_ANGLE_DEG", 5.0),
                0.0,
            ),
            max_aspect_ratio=_opt_float(
                "tqmesh_max_aspect_ratio",
                _env_float("BACKWATER_TQMESH_MAX_ASPECT", 20.0),
                1.0,
            ),
            min_area_rel_bbox=_opt_float(
                "tqmesh_min_area_rel_bbox",
                _env_float("BACKWATER_TQMESH_MIN_AREA_REL_BBOX", 1.0e-14),
                0.0,
            ),
            strict=_opt_bool("tqmesh_quality_strict", _env_bool("BACKWATER_TQMESH_QUALITY_STRICT", False)),
            # Fast default: single attempt at requested size.
            size_scales=_opt_float_tuple(
                "tqmesh_size_scales",
                _env_csv_floats("BACKWATER_TQMESH_SIZE_SCALES", (1.0,)),
            ),
            # Fast default: no smoothing.
            smooth_increments=_opt_int_tuple(
                "tqmesh_smooth_increments",
                tuple(
                    int(round(v)) for v in _env_csv_floats("BACKWATER_TQMESH_SMOOTH_INCREMENTS", (0.0,))
                ),
            ),
        )

    @staticmethod
    def _quad_controls_for_region(model: ConceptualModel, region: ConceptualRegion) -> Optional[Tuple[List[Tuple[float, float]], List[QuadEdgeControl]]]:
        return _quad_controls_for_region(model, region)

    def generate(self, model: ConceptualModel) -> MeshResult:
        try:
            import hydra_tqmesh as _tq
        except ImportError as exc:
            raise RuntimeError(
                "hydra_tqmesh C++ module not found.  "
                "Rebuild the plugin (cmake + make) to compile TQMesh bindings."
            ) from exc

        if not model.regions:
            raise ValueError("TQMeshBackend: no conceptual regions provided.")

        quality_cfg = self._quality_config()

        # ---- Process each region independently then merge results ----------
        # For the common single-region case this is straightforward.
        # Multi-region models are meshed separately and node indices merged.

        all_vx:   List[float] = []
        all_vy:   List[float] = []
        all_tris: List[int]   = []  # flat (n*3)
        all_quads: List[int]  = []  # flat (n*4)
        all_bv0:  List[int]   = []
        all_bv1:  List[int]   = []
        all_bc:   List[int]   = []
        all_ctype: List[str]  = []
        all_rid:   List[int]  = []
        all_size:  List[float]= []

        for region in model.regions:
            ring = list(region.ring_xy)
            if len(ring) < 3:
                continue

            # Close ring for deduplication check, then strip closing point
            if ring[0] == ring[-1]:
                ring = ring[:-1]
            if len(ring) < 3:
                continue

            ctype = str(region.default_cell_type).strip().lower()
            if ctype == "empty":
                continue

            target_size = max(float(region.default_size), 1e-10)
            region_constraints = _constraints_for_region(model, ring)
            region_exclusions = _region_exclusion_zones(model, region, ring)

            quad_controls = None
            quad_boundary = None
            if ctype in ("quadrilateral", "cartesian"):
                quad_setup = self._quad_controls_for_region(model, region)
                if quad_setup is not None:
                    quad_boundary, quad_controls = quad_setup

            # Exterior boundary — TQMesh expects CCW; ensure correct winding
            ext_verts = list(quad_boundary) if quad_boundary is not None else ring
            if quad_controls is None:
                simp_factor = max(_env_float("BACKWATER_TQMESH_BOUNDARY_SIMPLIFY_FACTOR", 0.35), 0.0)
                simp_tol = float(target_size) * simp_factor
                simp_max = max(8, int(round(_env_float("BACKWATER_TQMESH_BOUNDARY_MAX_VERTS", 64.0))))
                ext_verts = _simplify_closed_ring(ext_verts, tol=simp_tol, max_vertices=simp_max)
            if quad_controls is None and not self._is_ccw(ext_verts):
                ext_verts = list(reversed(ext_verts))
            ext_is_ccw = self._is_ccw(ext_verts)

            # All exterior edges get color 1 by default; real BC colors are
            # applied post-mesh in the workbench from swe2d_bc_lines (same
            # as the gmsh backend does).
            ext_colors = [1] * len(ext_verts)

            int_boundaries: List[List[List[float]]] = []
            int_colors: List[List[int]] = []
            for ering, _esize in region_exclusions:
                hring = list(ering)
                if hring and hring[0] == hring[-1]:
                    hring = hring[:-1]
                if len(hring) < 3:
                    continue
                if self._is_ccw(hring) == ext_is_ccw:
                    hring = list(reversed(hring))
                int_boundaries.append([[float(v[0]), float(v[1])] for v in hring])
                int_colors.append([1] * len(hring))

            # Constraint zones that overlap this region
            constraint_verts_list: List[List[tuple]] = []
            constraint_sizes_list: List[float] = []
            for cst in region_constraints:
                if len(cst.ring_xy) < 3 or str(cst.cell_type).strip().lower() == "empty":
                    continue
                constraint_verts_list.append(list(cst.ring_xy))
                constraint_sizes_list.append(float(cst.target_size))

            # Call the C++ binding
            active_quad_layers = []
            if quad_controls is not None:
                active_quad_layers = [
                    [
                        edge.points_xy[0][0],
                        edge.points_xy[0][1],
                        edge.points_xy[-1][0],
                        edge.points_xy[-1][1],
                        float(edge.n_layers),
                        float(edge.first_height if edge.first_height is not None else target_size),
                        float(edge.growth_rate),
                    ]
                    for edge in quad_controls
                    if edge.n_layers > 0 and edge.first_height is not None and edge.first_height > 0.0
                ]
                # Applying boundary layers on all four sides of a closed quad region
                # is not robust in TQMesh. For the HEC-RAS-style four-edge case,
                # use the explicit boundary sampling plus tri-to-quad conversion.
                if len(active_quad_layers) >= 4:
                    active_quad_layers = []

            # TQMesh can be sensitive to some combinations (quad-layers + tri2quad +
            # smoothing) for specific regions.  Try a stable cascade before failing.
            base_args = dict(
                ext_verts=[[v[0], v[1]] for v in ext_verts],
                ext_colors=ext_colors,
                int_boundaries=int_boundaries,
                int_colors=int_colors,
                constraint_verts=[[list(v) for v in cverts] for cverts in constraint_verts_list],
                constraint_sizes=constraint_sizes_list,
                target_size=target_size,
            )
            want_quads = ctype in ("quadrilateral", "cartesian")

            attempts = [
                ("requested", active_quad_layers, want_quads, 3),
            ]
            if active_quad_layers:
                attempts.append(("no-quad-layers", [], want_quads, 3))
            if want_quads:
                attempts.append(("triangles-only", [], False, 1))
            attempts.append(("minimal", [], False, 0))

            result = None
            errors: List[str] = []
            seen_cfg = set()
            used_label = "requested"
            used_quality: Optional[Dict[str, float]] = None
            best_nonpassing = None
            best_nonpassing_score = -float("inf")
            for label, quad_layers_try, tri_to_quad_try, n_smooth_try in attempts:
                for size_scale in quality_cfg.size_scales:
                    target_try = max(target_size * max(float(size_scale), 1e-6), 1e-10)
                    csz_try = [max(float(cs) * max(float(size_scale), 1e-6), 1e-10) for cs in constraint_sizes_list]
                    for ds in quality_cfg.smooth_increments:
                        smooth_try = max(0, int(n_smooth_try) + int(ds))
                        cfg_key = (
                            label,
                            tuple(tuple(q) for q in quad_layers_try),
                            bool(tri_to_quad_try),
                            int(smooth_try),
                            float(round(target_try, 12)),
                        )
                        if cfg_key in seen_cfg:
                            continue
                        seen_cfg.add(cfg_key)

                        try:
                            candidate = _tq.generate_triangular_mesh(
                                ext_verts=base_args["ext_verts"],
                                ext_colors=base_args["ext_colors"],
                                int_boundaries=base_args["int_boundaries"],
                                int_colors=base_args["int_colors"],
                                constraint_verts=base_args["constraint_verts"],
                                constraint_sizes=csz_try,
                                target_size=target_try,
                                quad_layers=quad_layers_try,
                                tri_to_quad=tri_to_quad_try,
                                n_smooth=smooth_try,
                            )
                        except Exception as exc:
                            errors.append(
                                f"{label} (size={target_try:.4g}, smooth={smooth_try}): {exc}"
                            )
                            continue

                        cand_vx = np.asarray(candidate["verts_x"], dtype=np.float64)
                        cand_vy = np.asarray(candidate["verts_y"], dtype=np.float64)
                        cand_tris = np.asarray(candidate["triangles"], dtype=np.int32)
                        cand_quads = np.asarray(candidate["quads"], dtype=np.int32)
                        stats = _mesh_quality_stats(cand_vx, cand_vy, cand_tris, cand_quads)

                        if _quality_passes(stats, quality_cfg):
                            result = candidate
                            used_label = f"{label} (size={target_try:.4g}, smooth={smooth_try})"
                            used_quality = stats
                            break

                        score = _quality_score(stats, quality_cfg)
                        if score > best_nonpassing_score:
                            best_nonpassing_score = score
                            best_nonpassing = (candidate, label, target_try, smooth_try, stats)
                        errors.append(
                            "quality-fail "
                            f"{label} (size={target_try:.4g}, smooth={smooth_try}): "
                            f"min_angle={stats['min_angle_deg']:.2f}, "
                            f"max_aspect={stats['max_aspect_ratio']:.2f}, "
                            f"min_area={stats['min_area']:.3e}"
                        )
                    if result is not None:
                        break
                if result is not None:
                    break

            if result is None and (not quality_cfg.strict) and best_nonpassing is not None:
                result, base_label, used_target_size, used_smooth, used_quality = best_nonpassing
                used_label = (
                    f"{base_label} (best-nonpassing; size={used_target_size:.4g}, "
                    f"smooth={used_smooth})"
                )
                warnings.warn(
                    "TQMesh quality thresholds not fully met for region "
                    f"{region.region_id}; using best available candidate in non-strict mode. "
                    f"Metrics: min_angle={used_quality['min_angle_deg']:.2f}, "
                    f"max_aspect={used_quality['max_aspect_ratio']:.2f}, "
                    f"min_area={used_quality['min_area']:.3e}",
                    RuntimeWarning,
                )

            if result is None:
                raise RuntimeError(
                    "TQMesh failed for region "
                    f"{region.region_id} after fallback attempts. "
                    + " | ".join(errors)
                )

            if errors:
                qmsg = ""
                if used_quality is not None:
                    qmsg = (
                        f" quality(min_angle={used_quality['min_angle_deg']:.2f},"
                        f" max_aspect={used_quality['max_aspect_ratio']:.2f},"
                        f" min_area={used_quality['min_area']:.3e})"
                    )
                warnings.warn(
                    "TQMesh fallback used for region "
                    f"{region.region_id} ({used_label}) due to prior failure(s): "
                    + " | ".join(errors)
                    + qmsg,
                    RuntimeWarning,
                )

            vx: np.ndarray = np.asarray(result["verts_x"], dtype=np.float64)
            vy: np.ndarray = np.asarray(result["verts_y"], dtype=np.float64)
            tris: np.ndarray  = np.asarray(result["triangles"], dtype=np.int32)
            quads: np.ndarray = np.asarray(result["quads"],     dtype=np.int32)
            bv0: np.ndarray   = np.asarray(result["bdry_v0"],   dtype=np.int32)
            bv1: np.ndarray   = np.asarray(result["bdry_v1"],   dtype=np.int32)
            bc:  np.ndarray   = np.asarray(result["bdry_color"],dtype=np.int32)

            offset = len(all_vx)
            all_vx.extend(vx.tolist())
            all_vy.extend(vy.tolist())

            if tris.size > 0:
                all_tris.extend((tris.ravel() + offset).tolist())
                all_ctype.extend(["triangular"] * tris.shape[0])
                all_rid.extend([region.region_id] * tris.shape[0])
                all_size.extend([target_size] * tris.shape[0])

            if quads.size > 0:
                all_quads.extend((quads.ravel() + offset).tolist())
                all_ctype.extend(["quadrilateral"] * quads.shape[0])
                all_rid.extend([region.region_id] * quads.shape[0])
                all_size.extend([target_size] * quads.shape[0])

            all_bv0.extend((bv0 + offset).tolist())
            all_bv1.extend((bv1 + offset).tolist())
            all_bc.extend(bc.tolist())

        if not all_vx:
            raise ValueError("TQMesh generated no vertices.")
        if not all_tris and not all_quads:
            raise ValueError("TQMesh generated no cells.")

        tri_conn = np.asarray(all_tris, dtype=np.int32)
        quad_conn = np.asarray(all_quads, dtype=np.int32)
        node_x, node_y, (tri_conn, quad_conn) = _weld_mesh_nodes(
            np.asarray(all_vx, dtype=np.float64),
            np.asarray(all_vy, dtype=np.float64),
            tri_conn,
            quad_conn,
        )
        node_z = np.zeros(node_x.size, dtype=np.float64)

        # Build CSR face topology from triangles + quads
        face_nodes_list: List[int] = []
        face_offsets: List[int] = [0]
        plot_tris: List[int] = []

        tris_arr = tri_conn.reshape(-1, 3) if tri_conn.size else np.empty((0,3), np.int32)
        quads_arr = quad_conn.reshape(-1, 4) if quad_conn.size else np.empty((0,4), np.int32)

        for tri in tris_arr:
            face_nodes_list.extend(tri.tolist())
            face_offsets.append(len(face_nodes_list))
            plot_tris.extend(tri.tolist())

        for quad in quads_arr:
            face_nodes_list.extend(quad.tolist())
            face_offsets.append(len(face_nodes_list))
            # Fan-decompose quad for plotting triangles
            plot_tris.extend([quad[0], quad[1], quad[2],
                               quad[0], quad[2], quad[3]])

        out = MeshResult(
            node_x=node_x,
            node_y=node_y,
            node_z=node_z,
            cell_nodes=np.asarray(plot_tris, dtype=np.int32),
            cell_face_offsets=np.asarray(face_offsets, dtype=np.int32),
            cell_face_nodes=np.asarray(face_nodes_list, dtype=np.int32),
            cell_type=np.asarray(all_ctype, dtype=object),
            region_id=np.asarray(all_rid, dtype=np.int32),
            target_size=np.asarray(all_size, dtype=np.float64),
        )
        return _repair_mesh_result(out)

    @staticmethod
    def _is_ccw(ring: List[Tuple[float, float]]) -> bool:
        """Return True if the ring has counter-clockwise winding (positive area)."""
        area = _polygon_area_xy(
            np.asarray([p[0] for p in ring]),
            np.asarray([p[1] for p in ring]),
        )
        return area > 0.0


def generate_face_centric_mesh(
        model: ConceptualModel,
        backend: str = "gmsh",
    options: Optional[Dict[str, object]] = None,
) -> MeshResult:
    """Generate a computational mesh from a ConceptualModel.

    Parameters
    ----------
    model   : ConceptualModel built from QGIS topology layers.
    backend : ``"gmsh"`` (default), ``"structured"``, ``"tqmesh"``.
              ``"gmsh"`` requires the ``gmsh`` Python package (pip install gmsh).
              ``"tqmesh"`` uses the built-in TQMesh advancing-front generator.
    options : Optional backend-specific options dictionary used for TQMesh and
              Gmsh advanced controls from the GUI.
    """
    if backend == "gmsh":
        if not _gmsh_available():
            raise RuntimeError(
                "gmsh Python package is not installed.  "
                "Run: pip install gmsh   (or select the 'Structured' backend)."
            )
        return GmshBackend(options=options).generate(model)
    if backend == "structured":
        return StructuredFaceCentricBackend().generate(model)
    if backend == "tqmesh":
        return TQMeshBackend(options=options).generate(model)
    raise ValueError(f"Unknown meshing backend: {backend!r}. Choose 'gmsh', 'structured', or 'tqmesh'.")
