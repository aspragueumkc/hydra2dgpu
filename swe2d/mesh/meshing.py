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

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import json
import os
import re
import time
import warnings

import numpy as np

from .mfem_opt import available_mfem_presets, optimize_with_mfem


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
    region_id: int = -1
    arc_role: Optional[str] = None
    points_xy: Optional[List[Tuple[float, float]]] = None
    use_global_arc_ctrl: bool = True
    arc_mode_override: Optional[str] = None
    arc_soft_size_override: Optional[float] = None
    arc_soft_dist_override: Optional[float] = None


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


_CELL_TYPES = {"triangular", "quadrilateral", "cartesian", "channel_generator", "empty"}


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
    recombine_topology_passes: Tuple[int, ...]
    recombine_min_quality: Tuple[float, ...]
    random_factors: Tuple[float, ...]
    optimize_methods: Tuple[str, ...]
    algorithm_switch_on_failure: bool
    recombine_node_repositioning: bool


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
    candidate_faces: List[List[int]] = []
    candidate_idx: List[int] = []

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

        # Guard against malformed polygons that repeat an undirected edge.
        # Such faces can trigger non-manifold construction errors downstream.
        seen_face_edges = set()
        repeated_edge = False
        for k in range(len(compact)):
            a = int(compact[k])
            b = int(compact[(k + 1) % len(compact)])
            key = (a, b) if a < b else (b, a)
            if key in seen_face_edges:
                repeated_edge = True
                break
            seen_face_edges.add(key)
        if repeated_edge:
            continue

        candidate_faces.append(compact)
        candidate_idx.append(i)

    # Keep a manifold-compatible subset: an undirected edge can belong to at
    # most two faces in the native SWE2D mesh builder.
    keep_face_nodes: List[int] = []
    keep_offsets: List[int] = [0]
    keep_idx: List[int] = []
    edge_owner_counts: Dict[Tuple[int, int], int] = {}

    for local_i, poly in enumerate(candidate_faces):
        face_edges = []
        for k in range(len(poly)):
            a = int(poly[k])
            b = int(poly[(k + 1) % len(poly)])
            key = (a, b) if a < b else (b, a)
            face_edges.append(key)

        if any(int(edge_owner_counts.get(key, 0)) >= 2 for key in face_edges):
            continue

        keep_face_nodes.extend(poly)
        keep_offsets.append(len(keep_face_nodes))
        keep_idx.append(int(candidate_idx[local_i]))
        for key in face_edges:
            edge_owner_counts[key] = int(edge_owner_counts.get(key, 0)) + 1

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
        quality_summary=dict(mesh.quality_summary or {}),
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


def _env_csv_strings(name: str, default: Sequence[str]) -> Tuple[str, ...]:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return tuple(str(v) for v in default)
    vals: List[str] = []
    for tok in raw.replace(";", ",").split(","):
        tok = str(tok).strip()
        if tok:
            vals.append(tok)
    if not vals:
        return tuple(str(v) for v in default)
    return tuple(vals)


def _write_mesh_checkpoint_npz(
    path: str,
    mesh: MeshResult,
    quality_summary: Optional[Dict[str, object]] = None,
) -> None:
    """Persist a best-so-far mesh checkpoint atomically for timeout recovery."""
    cp = str(path or "").strip()
    if not cp:
        return
    cp_dir = os.path.dirname(cp)
    if cp_dir:
        os.makedirs(cp_dir, exist_ok=True)

    payload = {
        "node_x": np.asarray(mesh.node_x, dtype=np.float64),
        "node_y": np.asarray(mesh.node_y, dtype=np.float64),
        "node_z": np.asarray(mesh.node_z, dtype=np.float64),
        "cell_nodes": np.asarray(mesh.cell_nodes, dtype=np.int32),
        "cell_face_offsets": np.asarray(mesh.cell_face_offsets, dtype=np.int32),
        "cell_face_nodes": np.asarray(mesh.cell_face_nodes, dtype=np.int32),
        "cell_type": np.asarray(mesh.cell_type).astype(np.str_),
        "region_id": np.asarray(mesh.region_id, dtype=np.int32),
        "target_size": np.asarray(mesh.target_size, dtype=np.float64),
        "quality_summary_json": np.asarray(
            json.dumps(dict(quality_summary or {}), default=float),
            dtype=np.str_,
        ),
    }
    tmp_path = f"{cp}.tmp"
    with open(tmp_path, "wb") as fh:
        np.savez_compressed(fh, **payload)
    os.replace(tmp_path, cp)


def _write_json_atomic(path: str, payload: Dict[str, object]) -> None:
    """Write JSON payload atomically for debug/recovery artifacts."""
    out = str(path or "").strip()
    if not out:
        return
    out_dir = os.path.dirname(out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    tmp_path = f"{out}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    os.replace(tmp_path, out)


def _serialize_xy_points(points: Sequence[Tuple[float, float]]) -> List[List[float]]:
    return [[float(x), float(y)] for x, y in points]


def _serialize_xy_lines(lines: Sequence[Sequence[Tuple[float, float]]]) -> List[List[List[float]]]:
    return [_serialize_xy_points(line) for line in lines]


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


def _normalize_conceptual_model_to_local_origin(model: ConceptualModel) -> Tuple[float, float]:
    """Translate conceptual geometry to a local origin for numeric robustness."""
    xs: List[float] = []
    ys: List[float] = []

    for node in model.nodes:
        xs.append(float(node.x))
        ys.append(float(node.y))

    for arc in model.arcs:
        if arc.points_xy:
            for x, y in arc.points_xy:
                xs.append(float(x))
                ys.append(float(y))

    for region in model.regions:
        for x, y in region.ring_xy:
            xs.append(float(x))
            ys.append(float(y))
        if region.hole_rings:
            for hole in region.hole_rings:
                for x, y in hole:
                    xs.append(float(x))
                    ys.append(float(y))

    for c in model.constraints:
        for x, y in c.ring_xy:
            xs.append(float(x))
            ys.append(float(y))

    for q in model.quad_edges:
        for x, y in q.points_xy:
            xs.append(float(x))
            ys.append(float(y))

    if not xs:
        return 0.0, 0.0

    x0 = float(min(xs))
    y0 = float(min(ys))
    if abs(x0) <= 0.0 and abs(y0) <= 0.0:
        return 0.0, 0.0

    for node in model.nodes:
        node.x = float(node.x) - x0
        node.y = float(node.y) - y0

    for arc in model.arcs:
        if arc.points_xy:
            arc.points_xy = [(float(x) - x0, float(y) - y0) for x, y in arc.points_xy]

    for region in model.regions:
        region.ring_xy = [(float(x) - x0, float(y) - y0) for x, y in region.ring_xy]
        if region.hole_rings:
            region.hole_rings = [
                [(float(x) - x0, float(y) - y0) for x, y in hole]
                for hole in region.hole_rings
            ]

    for c in model.constraints:
        c.ring_xy = [(float(x) - x0, float(y) - y0) for x, y in c.ring_xy]

    for q in model.quad_edges:
        q.points_xy = [(float(x) - x0, float(y) - y0) for x, y in q.points_xy]

    return x0, y0


def _restore_mesh_coordinates(mesh: MeshResult, x_shift: float, y_shift: float) -> MeshResult:
    if x_shift == 0.0 and y_shift == 0.0:
        return mesh
    mesh.node_x = np.asarray(mesh.node_x, dtype=np.float64) + float(x_shift)
    mesh.node_y = np.asarray(mesh.node_y, dtype=np.float64) + float(y_shift)
    return mesh


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


def _breakline_fixed_edges_for_region(
    model: ConceptualModel,
    region: ConceptualRegion,
    region_outer_ring: Optional[Sequence[Tuple[float, float]]] = None,
) -> List[List[Tuple[float, float]]]:
    """Collect arc breaklines for a region as interior fixed-edge polylines."""
    outer = list(region_outer_ring) if region_outer_ring is not None else list(region.ring_xy)
    if outer and outer[0] == outer[-1]:
        outer = outer[:-1]
    if len(outer) < 3:
        return []

    out: List[List[Tuple[float, float]]] = []
    seen = set()

    def _inside_region(x: float, y: float) -> bool:
        return _point_in_polygon(float(x), float(y), outer)

    def _line_hits_region(points: Sequence[Tuple[float, float]]) -> bool:
        for x, y in points:
            if _inside_region(x, y):
                return True
        for i in range(len(points) - 1):
            x0, y0 = points[i]
            x1, y1 = points[i + 1]
            mx = 0.5 * (float(x0) + float(x1))
            my = 0.5 * (float(y0) + float(y1))
            if _inside_region(mx, my):
                return True
        return False

    def _clip_to_region_segments(points: Sequence[Tuple[float, float]]) -> List[List[Tuple[float, float]]]:
        chunks: List[List[Tuple[float, float]]] = []
        cur: List[Tuple[float, float]] = []
        for i in range(len(points) - 1):
            p0 = points[i]
            p1 = points[i + 1]
            mx = 0.5 * (p0[0] + p1[0])
            my = 0.5 * (p0[1] + p1[1])
            keep = _inside_region(mx, my) or _inside_region(p0[0], p0[1]) or _inside_region(p1[0], p1[1])
            if keep:
                if not cur:
                    cur = [p0, p1]
                else:
                    if cur[-1] != p0:
                        cur.append(p0)
                    cur.append(p1)
            else:
                if len(cur) >= 2:
                    chunks.append(cur)
                cur = []
        if len(cur) >= 2:
            chunks.append(cur)
        return chunks

    for arc in model.arcs:
        pts = list(arc.points_xy or [])
        if len(pts) < 2:
            continue

        role = str(arc.arc_role or "").strip().lower()
        if role and role != "breakline":
            continue

        rid = int(getattr(arc, "region_id", -1))
        region_id_int = int(region.region_id)
        if rid not in {-1, region_id_int}:
            continue

        if not _line_hits_region(pts):
            continue

        clean: List[Tuple[float, float]] = []
        for x, y in pts:
            xx = float(x)
            yy = float(y)
            if clean:
                px, py = clean[-1]
                if float(np.hypot(xx - px, yy - py)) <= 1.0e-12:
                    continue
            clean.append((xx, yy))
        if len(clean) < 2:
            continue

        for chunk in _clip_to_region_segments(clean):
            if len(chunk) < 2:
                continue
            key = tuple((round(x, 7), round(y, 7)) for x, y in chunk)
            if key in seen:
                continue
            seen.add(key)
            out.append(chunk)

    return out


def _split_polyline_max_segment_length(
    points: Sequence[Tuple[float, float]],
    max_seg_len: float,
) -> List[Tuple[float, float]]:
    """Densify an open polyline so each segment length is <= max_seg_len."""
    pts = list(points)
    if len(pts) < 2:
        return pts
    if (not np.isfinite(float(max_seg_len))) or float(max_seg_len) <= 0.0:
        return [(float(x), float(y)) for x, y in pts]

    sampled = _sample_polyline([(float(x), float(y)) for x, y in pts], float(max_seg_len))
    if len(sampled) < 2:
        return [(float(pts[0][0]), float(pts[0][1])), (float(pts[-1][0]), float(pts[-1][1]))]
    return sampled


def _point_to_segment_projection(
    p: Tuple[float, float],
    a: Tuple[float, float],
    b: Tuple[float, float],
) -> Tuple[Tuple[float, float], float, float]:
    """Return closest point on segment AB to P as (point, t, distance)."""
    px, py = float(p[0]), float(p[1])
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    vx = bx - ax
    vy = by - ay
    den = vx * vx + vy * vy
    if den <= 1.0e-30:
        d = float(np.hypot(px - ax, py - ay))
        return (ax, ay), 0.0, d
    t = ((px - ax) * vx + (py - ay) * vy) / den
    t = max(0.0, min(1.0, float(t)))
    qx = ax + t * vx
    qy = ay + t * vy
    d = float(np.hypot(px - qx, py - qy))
    return (qx, qy), t, d


def _segment_intersection_point(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    q0: Tuple[float, float],
    q1: Tuple[float, float],
    eps: float = 1.0e-9,
) -> Optional[Tuple[Tuple[float, float], float, float]]:
    """Return segment intersection as (point, t_on_p, u_on_q), if any.

    Collinear overlap is handled conservatively by returning endpoint touches only.
    """
    p0x, p0y = float(p0[0]), float(p0[1])
    p1x, p1y = float(p1[0]), float(p1[1])
    q0x, q0y = float(q0[0]), float(q0[1])
    q1x, q1y = float(q1[0]), float(q1[1])

    rx = p1x - p0x
    ry = p1y - p0y
    sx = q1x - q0x
    sy = q1y - q0y

    def _cross(ax: float, ay: float, bx: float, by: float) -> float:
        return ax * by - ay * bx

    rxs = _cross(rx, ry, sx, sy)
    qpx = q0x - p0x
    qpy = q0y - p0y
    qpxr = _cross(qpx, qpy, rx, ry)

    if abs(rxs) <= eps:
        if abs(qpxr) > eps:
            return None
        # Collinear: only return when endpoints nearly touch.
        for pt in ((p0x, p0y), (p1x, p1y)):
            _, u, d = _point_to_segment_projection((pt[0], pt[1]), (q0x, q0y), (q1x, q1y))
            if d <= eps:
                seg_len_p = max(float(np.hypot(rx, ry)), 1.0e-30)
                t = float(np.hypot(pt[0] - p0x, pt[1] - p0y)) / seg_len_p
                return ((pt[0], pt[1]), t, float(u))
        for pt in ((q0x, q0y), (q1x, q1y)):
            _, t, d = _point_to_segment_projection((pt[0], pt[1]), (p0x, p0y), (p1x, p1y))
            if d <= eps:
                seg_len_q = max(float(np.hypot(sx, sy)), 1.0e-30)
                u = float(np.hypot(pt[0] - q0x, pt[1] - q0y)) / seg_len_q
                return ((pt[0], pt[1]), float(t), u)
        return None

    t = _cross(qpx, qpy, sx, sy) / rxs
    u = _cross(qpx, qpy, rx, ry) / rxs
    if t < -eps or t > 1.0 + eps or u < -eps or u > 1.0 + eps:
        return None

    t = max(0.0, min(1.0, float(t)))
    u = max(0.0, min(1.0, float(u)))
    ix = p0x + t * rx
    iy = p0y + t * ry
    return ((ix, iy), t, u)


def _snap_and_split_boundary_for_breaklines(
    ring: Sequence[Tuple[float, float]],
    fixed_edge_lines: Sequence[Sequence[Tuple[float, float]]],
    vertex_snap_tol: float = 0.1,
) -> Tuple[List[Tuple[float, float]], List[List[Tuple[float, float]]]]:
    """Snap breakline vertices to nearby boundary vertices and split boundary edges.

    Rules:
    - If a breakline vertex is within ``vertex_snap_tol`` of a boundary vertex,
      snap breakline vertex to that boundary vertex.
    - Otherwise, split boundary edges at breakline intersections/touches.
    """
    base_ring = list(ring)
    if len(base_ring) < 3:
        return list(base_ring), [list(line) for line in fixed_edge_lines]
    if base_ring[0] == base_ring[-1]:
        base_ring = base_ring[:-1]
    if len(base_ring) < 3:
        return list(base_ring), [list(line) for line in fixed_edge_lines]

    vtol = max(float(vertex_snap_tol), 0.0)
    line_hit_tol = max(1.0e-9, 1.0e-7 * max(1.0, max(abs(p[0]) + abs(p[1]) for p in base_ring)))

    snapped_lines: List[List[Tuple[float, float]]] = []
    for line in fixed_edge_lines:
        clean: List[Tuple[float, float]] = []
        for x, y in line:
            pt = (float(x), float(y))
            best = None
            best_d = float("inf")
            if vtol > 0.0:
                for rv in base_ring:
                    d = float(np.hypot(pt[0] - rv[0], pt[1] - rv[1]))
                    if d < best_d:
                        best_d = d
                        best = rv
                if best is not None and best_d < vtol:
                    pt = (float(best[0]), float(best[1]))
            if clean:
                if float(np.hypot(pt[0] - clean[-1][0], pt[1] - clean[-1][1])) <= 1.0e-12:
                    continue
            clean.append(pt)
        if len(clean) >= 2:
            snapped_lines.append(clean)

    # Collect split parameters per boundary edge index.
    split_params: Dict[int, List[Tuple[float, Tuple[float, float]]]] = {}
    n_ring = len(base_ring)
    for i in range(n_ring):
        split_params[i] = []

    for line in snapped_lines:
        # Boundary touches from explicit breakline vertices.
        for p in line:
            for i in range(n_ring):
                a = base_ring[i]
                b = base_ring[(i + 1) % n_ring]
                q, u, d = _point_to_segment_projection(p, a, b)
                if d <= line_hit_tol:
                    split_params[i].append((float(u), (float(q[0]), float(q[1]))))

        # True segment intersections.
        for li in range(len(line) - 1):
            p0 = line[li]
            p1 = line[li + 1]
            if float(np.hypot(p1[0] - p0[0], p1[1] - p0[1])) <= 1.0e-15:
                continue
            for i in range(n_ring):
                q0 = base_ring[i]
                q1 = base_ring[(i + 1) % n_ring]
                inter = _segment_intersection_point(p0, p1, q0, q1, eps=line_hit_tol)
                if inter is None:
                    continue
                ipt, _t, u = inter
                split_params[i].append((float(u), (float(ipt[0]), float(ipt[1]))))

    # Rebuild boundary ring with inserted split vertices.
    new_ring: List[Tuple[float, float]] = []
    for i in range(n_ring):
        a = (float(base_ring[i][0]), float(base_ring[i][1]))
        if not new_ring:
            new_ring.append(a)
        elif float(np.hypot(a[0] - new_ring[-1][0], a[1] - new_ring[-1][1])) > 1.0e-12:
            new_ring.append(a)

        entries = split_params.get(i, [])
        if not entries:
            continue

        # Stable/unique interior split points on current edge.
        entries.sort(key=lambda item: item[0])
        filtered: List[Tuple[float, Tuple[float, float]]] = []
        for u, pt in entries:
            uu = max(0.0, min(1.0, float(u)))
            if uu <= 1.0e-10 or uu >= 1.0 - 1.0e-10:
                continue
            if filtered and abs(uu - filtered[-1][0]) <= 1.0e-9:
                continue
            filtered.append((uu, pt))

        for _u, pt in filtered:
            q = (float(pt[0]), float(pt[1]))
            if float(np.hypot(q[0] - new_ring[-1][0], q[1] - new_ring[-1][1])) <= 1.0e-12:
                continue
            new_ring.append(q)

    # Remove accidental trailing duplicate closure.
    while len(new_ring) >= 2 and float(np.hypot(new_ring[0][0] - new_ring[-1][0], new_ring[0][1] - new_ring[-1][1])) <= 1.0e-12:
        new_ring.pop()

    if len(new_ring) < 3:
        new_ring = list(base_ring)

    # Final snap pass against updated boundary vertices.
    if vtol > 0.0:
        snapped2: List[List[Tuple[float, float]]] = []
        for line in snapped_lines:
            out: List[Tuple[float, float]] = []
            for x, y in line:
                p = (float(x), float(y))
                best = None
                best_d = float("inf")
                for rv in new_ring:
                    d = float(np.hypot(p[0] - rv[0], p[1] - rv[1]))
                    if d < best_d:
                        best_d = d
                        best = rv
                if best is not None and best_d < vtol:
                    p = (float(best[0]), float(best[1]))
                if out and float(np.hypot(p[0] - out[-1][0], p[1] - out[-1][1])) <= 1.0e-12:
                    continue
                out.append(p)
            if len(out) >= 2:
                snapped2.append(out)
        snapped_lines = snapped2

    return new_ring, snapped_lines


def _boundary_contact_vertices(
    ring: Sequence[Tuple[float, float]],
    fixed_edge_lines: Sequence[Sequence[Tuple[float, float]]],
    tol: float = 1.0e-6,
) -> List[Tuple[float, float]]:
    """Return ring vertices that coincide with any fixed-edge polyline vertex."""
    rr = list(ring)
    if rr and rr[0] == rr[-1]:
        rr = rr[:-1]
    if not rr:
        return []

    ttol = max(float(tol), 1.0e-12)
    out: List[Tuple[float, float]] = []
    for rv in rr:
        keep = False
        for line in fixed_edge_lines:
            for p in line:
                if float(np.hypot(float(rv[0]) - float(p[0]), float(rv[1]) - float(p[1]))) <= ttol:
                    keep = True
                    break
            if keep:
                break
        if keep:
            out.append((float(rv[0]), float(rv[1])))
    return out


def _parse_invalid_boundary_edge_sample_points(error_text: str) -> List[Tuple[float, float]]:
    """Extract boundary edge sample endpoints from TQMesh error text."""
    txt = str(error_text or "")
    tag = "invalid_boundary_edge_samples=["
    i0 = txt.find(tag)
    if i0 < 0:
        return []
    i1 = txt.find(";", i0)
    if i1 < 0:
        i1 = len(txt)
    sample_txt = txt[i0:i1]
    pat = re.compile(
        r"\(([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\)\s*->\s*\(([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\)"
    )
    pts: List[Tuple[float, float]] = []
    seen = set()
    for m in pat.finditer(sample_txt):
        p0 = (float(m.group(1)), float(m.group(2)))
        p1 = (float(m.group(3)), float(m.group(4)))
        for p in (p0, p1):
            key = (round(float(p[0]), 6), round(float(p[1]), 6))
            if key in seen:
                continue
            seen.add(key)
            pts.append((float(p[0]), float(p[1])))
    return pts


def _collapse_boundary_microchains_near_points(
    ring: Sequence[Tuple[float, float]],
    focus_points: Sequence[Tuple[float, float]],
    target_size: float,
    protect_points: Optional[Sequence[Tuple[float, float]]] = None,
    protect_tol: Optional[float] = None,
) -> List[Tuple[float, float]]:
    """Collapse short near-collinear boundary chains near given focus points."""
    pts = list(ring)
    if len(pts) < 4 or not focus_points:
        return pts

    tgt = max(float(target_size), 1.0e-9)
    focus_radius = max(1.75 * tgt, 8.0)
    short_len = max(0.8 * tgt, 1.0)
    cross_tol = max(0.12 * tgt, 0.5)

    prot = list(protect_points or [])
    p_tol = max(float(protect_tol) if protect_tol is not None else max(1.0e-6, 1.0e-3 * tgt), 1.0e-12)

    def _is_protected(p: Tuple[float, float]) -> bool:
        for q in prot:
            if float(np.hypot(float(p[0]) - float(q[0]), float(p[1]) - float(q[1]))) <= p_tol:
                return True
        return False

    def _near_focus(p: Tuple[float, float]) -> bool:
        for q in focus_points:
            if float(np.hypot(float(p[0]) - float(q[0]), float(p[1]) - float(q[1]))) <= focus_radius:
                return True
        return False

    work = [(float(x), float(y)) for x, y in pts]
    for _ in range(6):
        if len(work) <= 3:
            break
        changed = False
        out: List[Tuple[float, float]] = []
        n = len(work)
        for i in range(n):
            p_prev = work[(i - 1) % n]
            p_cur = work[i]
            p_next = work[(i + 1) % n]

            if _is_protected(p_cur) or (not _near_focus(p_cur)):
                out.append(p_cur)
                continue

            a = np.asarray([p_cur[0] - p_prev[0], p_cur[1] - p_prev[1]], dtype=np.float64)
            b = np.asarray([p_next[0] - p_cur[0], p_next[1] - p_cur[1]], dtype=np.float64)
            c = np.asarray([p_next[0] - p_prev[0], p_next[1] - p_prev[1]], dtype=np.float64)
            la = float(np.hypot(a[0], a[1]))
            lb = float(np.hypot(b[0], b[1]))
            lc = float(np.hypot(c[0], c[1]))

            drop = False
            if min(la, lb) <= short_len:
                if lc <= 1.0e-14:
                    drop = True
                else:
                    perp = float(abs(c[0] * (p_prev[1] - p_cur[1]) - (p_prev[0] - p_cur[0]) * c[1]) / lc)
                    dot = float((a[0] * b[0] + a[1] * b[1]) / max(la * lb, 1.0e-30))
                    if perp <= cross_tol or dot > 0.20:
                        drop = True

            if drop and len(work) - 1 >= 3:
                changed = True
                continue
            out.append(p_cur)

        if not changed or len(out) < 3:
            break
        work = out

    return work if len(work) >= 3 else pts


def _ring_key(ring: Sequence[Tuple[float, float]], ndigits: int = 6) -> Tuple[Tuple[float, float], ...]:
    return tuple((round(float(x), ndigits), round(float(y), ndigits)) for x, y in ring)


def _jitter_boundary_vertices_near_points(
    ring: Sequence[Tuple[float, float]],
    focus_points: Sequence[Tuple[float, float]],
    jitter_scale: float,
    variant_index: int = 0,
    protect_points: Optional[Sequence[Tuple[float, float]]] = None,
    protect_tol: Optional[float] = None,
) -> List[Tuple[float, float]]:
    """Deterministically jitter boundary vertices near focus points."""
    pts = [(float(x), float(y)) for x, y in ring]
    if len(pts) < 3 or not focus_points:
        return pts

    scale = max(float(jitter_scale), 0.0)
    if scale <= 0.0:
        return pts

    prot = list(protect_points or [])
    p_tol = max(float(protect_tol) if protect_tol is not None else 1.0e-6, 1.0e-12)
    focus_radius = max(4.0 * scale, 8.0)

    def _is_protected(p: Tuple[float, float]) -> bool:
        for q in prot:
            if float(np.hypot(float(p[0]) - float(q[0]), float(p[1]) - float(q[1]))) <= p_tol:
                return True
        return False

    out: List[Tuple[float, float]] = []
    for idx, p in enumerate(pts):
        if _is_protected(p):
            out.append(p)
            continue

        nearest = None
        nearest_d = float("inf")
        for q in focus_points:
            d = float(np.hypot(float(p[0]) - float(q[0]), float(p[1]) - float(q[1])))
            if d < nearest_d:
                nearest_d = d
                nearest = (float(q[0]), float(q[1]))
        if nearest is None or nearest_d > focus_radius:
            out.append(p)
            continue

        vx = float(p[0]) - float(nearest[0])
        vy = float(p[1]) - float(nearest[1])
        vn = float(np.hypot(vx, vy))
        if vn <= 1.0e-12:
            ang = 0.5 * (1.0 + float(np.sin((idx + 1) * 0.913 + (variant_index + 1) * 1.371))) * (2.0 * np.pi)
            vx = float(np.cos(ang))
            vy = float(np.sin(ang))
            vn = 1.0
        ux = vx / vn
        uy = vy / vn
        tx = -uy
        ty = ux

        radial = 1.0 + 0.25 * float(np.sin((idx + 1) * (variant_index + 1) * 0.73))
        tangential = 0.35 * float(np.cos((idx + 1) * (variant_index + 2) * 0.61))
        sign = 1.0 if ((idx + variant_index) % 2 == 0) else -1.0
        dx = scale * sign * (radial * ux + tangential * tx)
        dy = scale * sign * (radial * uy + tangential * ty)
        out.append((float(p[0] + dx), float(p[1] + dy)))

    return out


def _insert_focus_points_on_ring_segments(
    ring: Sequence[Tuple[float, float]],
    focus_points: Sequence[Tuple[float, float]],
    max_dist: float,
) -> List[Tuple[float, float]]:
    """Insert projected focus points on nearest ring segments when close enough."""
    pts = [(float(x), float(y)) for x, y in ring]
    if len(pts) < 3 or not focus_points:
        return pts

    dmax = max(float(max_dist), 1.0e-12)
    n = len(pts)
    inserts: Dict[int, List[Tuple[float, Tuple[float, float]]]] = {i: [] for i in range(n)}

    for fp in focus_points:
        p = (float(fp[0]), float(fp[1]))
        best_i = -1
        best_u = 0.0
        best_q = (0.0, 0.0)
        best_d = float("inf")
        for i in range(n):
            a = pts[i]
            b = pts[(i + 1) % n]
            q, u, d = _point_to_segment_projection(p, a, b)
            if d < best_d:
                best_d = float(d)
                best_i = i
                best_u = float(u)
                best_q = (float(q[0]), float(q[1]))
        if best_i < 0 or best_d > dmax:
            continue
        if best_u <= 1.0e-10 or best_u >= 1.0 - 1.0e-10:
            continue
        inserts[best_i].append((best_u, best_q))

    out: List[Tuple[float, float]] = []
    for i in range(n):
        a = pts[i]
        if not out or float(np.hypot(a[0] - out[-1][0], a[1] - out[-1][1])) > 1.0e-12:
            out.append(a)
        items = inserts.get(i, [])
        if not items:
            continue
        items.sort(key=lambda it: it[0])
        filtered: List[Tuple[float, Tuple[float, float]]] = []
        for u, q in items:
            if filtered and abs(float(u) - filtered[-1][0]) <= 1.0e-8:
                continue
            filtered.append((float(u), (float(q[0]), float(q[1]))))
        for _u, q in filtered:
            if float(np.hypot(q[0] - out[-1][0], q[1] - out[-1][1])) <= 1.0e-12:
                continue
            out.append(q)

    while len(out) >= 2 and float(np.hypot(out[0][0] - out[-1][0], out[0][1] - out[-1][1])) <= 1.0e-12:
        out.pop()
    return out if len(out) >= 3 else pts


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


def _split_closed_ring_max_segment_length(
    ring: Sequence[Tuple[float, float]],
    max_seg_len: float,
) -> List[Tuple[float, float]]:
    """Densify a closed ring so each segment length is <= max_seg_len."""
    pts = list(ring)
    if len(pts) < 3:
        return pts
    if pts and pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(pts) < 3:
        return pts
    if (not np.isfinite(float(max_seg_len))) or float(max_seg_len) <= 0.0:
        return [(float(x), float(y)) for x, y in pts]

    out: List[Tuple[float, float]] = []
    n = len(pts)
    for i in range(n):
        a = (float(pts[i][0]), float(pts[i][1]))
        b = (float(pts[(i + 1) % n][0]), float(pts[(i + 1) % n][1]))
        seg = _split_polyline_max_segment_length([a, b], float(max_seg_len))
        if i == 0:
            out.extend(seg)
        else:
            out.extend(seg[1:])

    # remove accidental closure duplicate
    while len(out) >= 2 and float(np.hypot(out[0][0] - out[-1][0], out[0][1] - out[-1][1])) <= 1.0e-12:
        out.pop()
    return out if len(out) >= 3 else [(float(x), float(y)) for x, y in pts]


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


def _sanitize_closed_ring(
    ring: Sequence[Tuple[float, float]],
    length_tol: float,
    collinear_tol: float,
    protect_points: Optional[Sequence[Tuple[float, float]]] = None,
    protect_tol: Optional[float] = None,
) -> List[Tuple[float, float]]:
    """Clean a closed ring for meshing robustness.

    Removes near-duplicate points, tiny backtracking spikes, and nearly
    collinear vertices while preserving ring order.
    """
    pts = list(ring)
    if len(pts) < 3:
        return pts
    if pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(pts) < 3:
        return pts

    length_tol = max(float(length_tol), 1.0e-12)
    collinear_tol = max(float(collinear_tol), 1.0e-12)
    prot = list(protect_points or [])
    prot_tol = max(float(protect_tol) if protect_tol is not None else length_tol, 1.0e-12)

    def _is_protected(pt: Tuple[float, float]) -> bool:
        if not prot:
            return False
        for qq in prot:
            if float(np.hypot(float(pt[0]) - float(qq[0]), float(pt[1]) - float(qq[1]))) <= prot_tol:
                return True
        return False

    dedup: List[Tuple[float, float]] = []
    for p in pts:
        pp = (float(p[0]), float(p[1]))
        if not dedup:
            dedup.append(pp)
            continue
        if float(np.hypot(pp[0] - dedup[-1][0], pp[1] - dedup[-1][1])) > length_tol:
            dedup.append(pp)

    while len(dedup) >= 2 and float(np.hypot(dedup[0][0] - dedup[-1][0], dedup[0][1] - dedup[-1][1])) <= length_tol:
        dedup.pop()
    if len(dedup) < 3:
        return dedup

    work = list(dedup)
    changed = True
    for _ in range(6):
        if not changed or len(work) <= 3:
            break
        changed = False
        out: List[Tuple[float, float]] = []
        n = len(work)
        for i in range(n):
            p_prev = work[(i - 1) % n]
            p_cur = work[i]
            p_next = work[(i + 1) % n]

            a = np.asarray([p_cur[0] - p_prev[0], p_cur[1] - p_prev[1]], dtype=np.float64)
            b = np.asarray([p_next[0] - p_cur[0], p_next[1] - p_cur[1]], dtype=np.float64)
            c = np.asarray([p_next[0] - p_prev[0], p_next[1] - p_prev[1]], dtype=np.float64)
            la = float(np.hypot(a[0], a[1]))
            lb = float(np.hypot(b[0], b[1]))
            lc = float(np.hypot(c[0], c[1]))

            drop = False
            if _is_protected(p_cur):
                out.append(p_cur)
                continue
            if la <= length_tol or lb <= length_tol:
                drop = True
            elif lc > 1.0e-14:
                perp = float(abs(c[0] * (p_prev[1] - p_cur[1]) - (p_prev[0] - p_cur[0]) * c[1]) / lc)
                dot = float((a[0] * b[0] + a[1] * b[1]) / max(la * lb, 1.0e-30))
                if perp <= collinear_tol:
                    drop = True
                elif dot < -0.985 and min(la, lb) <= max(4.0 * length_tol, 5.0 * collinear_tol):
                    drop = True

            if drop and len(work) - 1 >= 3:
                changed = True
                continue
            out.append(p_cur)

        if len(out) < 3:
            break
        work = out

    return work if len(work) >= 3 else dedup


def _resample_closed_ring(points: Sequence[Tuple[float, float]], target_step: float) -> List[Tuple[float, float]]:
    """Uniformly resample a closed ring with approximate step size."""
    pts = list(points)
    if len(pts) < 3:
        return pts
    if pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(pts) < 3:
        return pts
    closed = list(pts) + [pts[0]]
    sampled = _sample_polyline(closed, max(float(target_step), 1.0e-9))
    if len(sampled) >= 2 and float(np.hypot(sampled[0][0] - sampled[-1][0], sampled[0][1] - sampled[-1][1])) <= 1.0e-9:
        sampled = sampled[:-1]
    return sampled if len(sampled) >= 3 else pts


def _stitch_boundary_microchains(
    ring: Sequence[Tuple[float, float]],
    target_size: float,
    protect_points: Optional[Sequence[Tuple[float, float]]] = None,
    protect_tol: Optional[float] = None,
) -> List[Tuple[float, float]]:
    """Collapse short near-collinear boundary chains that can stall front closure."""
    pts = list(ring)
    if len(pts) < 4:
        return pts

    tgt = max(float(target_size), 1.0e-9)
    short_len = 1.35 * tgt
    tiny_len = 0.35 * tgt
    cross_tol = 0.30 * tgt
    prot = list(protect_points or [])
    prot_tol = max(float(protect_tol) if protect_tol is not None else max(1.0e-6, 1.0e-3 * tgt), 1.0e-12)

    def _is_protected(pt: Tuple[float, float]) -> bool:
        if not prot:
            return False
        for qq in prot:
            if float(np.hypot(float(pt[0]) - float(qq[0]), float(pt[1]) - float(qq[1]))) <= prot_tol:
                return True
        return False

    work = list(pts)
    for _ in range(8):
        if len(work) <= 3:
            break
        changed = False
        out: List[Tuple[float, float]] = []
        n = len(work)
        for i in range(n):
            p_prev = work[(i - 1) % n]
            p_cur = work[i]
            p_next = work[(i + 1) % n]

            a = np.asarray([p_cur[0] - p_prev[0], p_cur[1] - p_prev[1]], dtype=np.float64)
            b = np.asarray([p_next[0] - p_cur[0], p_next[1] - p_cur[1]], dtype=np.float64)
            la = float(np.hypot(a[0], a[1]))
            lb = float(np.hypot(b[0], b[1]))

            drop = False
            if _is_protected(p_cur):
                out.append(p_cur)
                continue
            if la <= tiny_len or lb <= tiny_len:
                drop = True
            elif la <= short_len and lb <= short_len:
                denom = max(la * lb, 1.0e-30)
                dot = float((a[0] * b[0] + a[1] * b[1]) / denom)
                c = np.asarray([p_next[0] - p_prev[0], p_next[1] - p_prev[1]], dtype=np.float64)
                lc = float(np.hypot(c[0], c[1]))
                if lc > 1.0e-14:
                    perp = float(abs(c[0] * (p_prev[1] - p_cur[1]) - (p_prev[0] - p_cur[0]) * c[1]) / lc)
                else:
                    perp = 0.0
                # Accept both straight-ish continuation and tiny jogs.
                if dot > 0.35 or perp <= cross_tol:
                    drop = True

            if drop and len(work) - 1 >= 3:
                changed = True
                continue
            out.append(p_cur)

        if not changed:
            break
        if len(out) < 3:
            break
        work = out

    return work if len(work) >= 3 else pts


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
    max_cells: Optional[int] = None,
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

    def _edge_target(edge_idx: int, fallback: float) -> float:
        if 0 <= edge_idx < len(quad_controls):
            ts = quad_controls[edge_idx].target_size
            if ts is not None and float(ts) > 0.0:
                return float(ts)
        return float(fallback)

    # Prefer explicit quad-edge target sizes when available.
    x_target = max(float(0.5 * (_edge_target(0, default_edge_sizes[0]) + _edge_target(2, default_edge_sizes[2]))), 1.0e-9)
    y_target = max(float(0.5 * (_edge_target(1, default_edge_sizes[1]) + _edge_target(3, default_edge_sizes[3]))), 1.0e-9)
    x_len = 0.5 * (_polyline_length(bottom) + _polyline_length(top))
    y_len = 0.5 * (_polyline_length(left) + _polyline_length(right))

    xi_vals = _build_axis_params(x_len, x_target, quad_controls[3], quad_controls[1])
    eta_vals = _build_axis_params(y_len, y_target, quad_controls[0], quad_controls[2])
    nx = max(1, int(xi_vals.size - 1))
    ny = max(1, int(eta_vals.size - 1))

    max_cells_int = int(max_cells) if max_cells is not None else 0
    if max_cells_int > 0 and nx * ny > max_cells_int:
        # Coarsen uniformly to keep full-region aligned generation bounded.
        scale = float(np.sqrt(float(nx * ny) / float(max_cells_int)))
        scale = max(scale, 1.0)
        xi_vals = _build_axis_params(x_len, x_target * scale, quad_controls[3], quad_controls[1])
        eta_vals = _build_axis_params(y_len, y_target * scale, quad_controls[0], quad_controls[2])
        nx = max(1, int(xi_vals.size - 1))
        ny = max(1, int(eta_vals.size - 1))

    if max_cells_int > 0 and nx * ny > max_cells_int:
        aspect = max(float(x_len), 1.0e-9) / max(float(y_len), 1.0e-9)
        nx_cap = max(1, int(round(np.sqrt(float(max_cells_int) * aspect))))
        ny_cap = max(1, int(max_cells_int // max(nx_cap, 1)))
        xi_vals = np.linspace(0.0, 1.0, nx_cap + 1, dtype=np.float64)
        eta_vals = np.linspace(0.0, 1.0, ny_cap + 1, dtype=np.float64)
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


def _point_to_segment_distance_s(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> Tuple[float, float]:
    vx = float(bx - ax)
    vy = float(by - ay)
    wx = float(px - ax)
    wy = float(py - ay)
    vv = float(vx * vx + vy * vy)
    if vv <= 1.0e-20:
        return float(np.hypot(px - ax, py - ay)), 0.0
    t = max(0.0, min(1.0, float((wx * vx + wy * vy) / vv)))
    qx = ax + t * vx
    qy = ay + t * vy
    return float(np.hypot(px - qx, py - qy)), t


def _polyline_distance_and_s(points: Sequence[Tuple[float, float]], x: float, y: float) -> Tuple[float, float]:
    if len(points) < 2:
        return float("inf"), 0.0
    best_d = float("inf")
    best_s = 0.0
    acc = 0.0
    for i in range(len(points) - 1):
        ax, ay = points[i]
        bx, by = points[i + 1]
        seg = float(np.hypot(bx - ax, by - ay))
        d, t = _point_to_segment_distance_s(x, y, ax, ay, bx, by)
        s = acc + t * seg
        if d < best_d:
            best_d = d
            best_s = s
        acc += seg
    return best_d, best_s


def _region_node_sets_from_mesh(mesh: MeshResult) -> Dict[int, set]:
    rid = np.asarray(mesh.region_id, dtype=np.int32)
    offs = np.asarray(mesh.cell_face_offsets, dtype=np.int32)
    nodes = np.asarray(mesh.cell_face_nodes, dtype=np.int32)
    out: Dict[int, set] = {}
    for r in np.unique(rid):
        nset: set = set()
        for ci in np.where(rid == int(r))[0]:
            s = int(offs[ci])
            e = int(offs[ci + 1])
            nset.update(nodes[s:e].tolist())
        out[int(r)] = nset
    return out


def _region_boundary_node_sets_from_mesh(mesh: MeshResult) -> Dict[int, set]:
    """Return region->node ids that lie on region exterior or inter-region interfaces."""
    rid = np.asarray(mesh.region_id, dtype=np.int32)
    offs = np.asarray(mesh.cell_face_offsets, dtype=np.int32)
    nodes = np.asarray(mesh.cell_face_nodes, dtype=np.int32)

    edge_regions: Dict[Tuple[int, int], set] = {}
    edge_nodes: Dict[Tuple[int, int], Tuple[int, int]] = {}

    for ci in range(offs.size - 1):
        s = int(offs[ci])
        e = int(offs[ci + 1])
        poly = nodes[s:e]
        if poly.size < 2:
            continue
        rr = int(rid[ci])
        for k in range(poly.size):
            a = int(poly[k])
            b = int(poly[(k + 1) % poly.size])
            key = (a, b) if a < b else (b, a)
            if key not in edge_nodes:
                edge_nodes[key] = (a, b)
            edge_regions.setdefault(key, set()).add(rr)

    out: Dict[int, set] = {}
    for key, owners in edge_regions.items():
        a, b = edge_nodes[key]
        for rr in owners:
            out.setdefault(int(rr), set())

        # Region exterior edge (single owner) or interface (multi-owner):
        # keep both endpoints as boundary candidates for participating regions.
        for rr in owners:
            bucket = out.setdefault(int(rr), set())
            bucket.add(int(a))
            bucket.add(int(b))

    return out


def _enforce_quad_interface_conformance(
    mesh: MeshResult,
    model: ConceptualModel,
    snap_tol: float = 1.0,
) -> MeshResult:
    """Snap adjacent-region interface nodes onto quad-region edge node lines.

    This enforces shared node placement along interfaces for independently meshed
    regions (e.g. triangular region next to a flow-aligned quad block).
    """
    tol = max(float(snap_tol), 1.0e-9)
    node_x = np.asarray(mesh.node_x, dtype=np.float64).copy()
    node_y = np.asarray(mesh.node_y, dtype=np.float64).copy()

    region_nodes = _region_node_sets_from_mesh(mesh)
    region_boundary_nodes = _region_boundary_node_sets_from_mesh(mesh)
    if not region_nodes:
        return mesh

    rid_to_region: Dict[int, ConceptualRegion] = {int(r.region_id): r for r in model.regions}
    quad_region_ids = [
        rid
        for rid, region in rid_to_region.items()
        if str(region.default_cell_type).strip().lower() in {"quadrilateral", "cartesian", "channel_generator"}
    ]

    for qrid in quad_region_ids:
        region = rid_to_region.get(int(qrid))
        if region is None:
            continue
        quad_setup = _quad_controls_for_region(model, region)
        if quad_setup is None:
            continue
        _ring, quad_edges = quad_setup
        q_nodes_all = region_boundary_nodes.get(int(qrid), set())
        if not q_nodes_all:
            continue

        for edge in quad_edges:
            edge_pts = [(float(x), float(y)) for (x, y) in edge.points_xy]
            if len(edge_pts) < 2:
                continue

            ex = np.asarray([p[0] for p in edge_pts], dtype=np.float64)
            ey = np.asarray([p[1] for p in edge_pts], dtype=np.float64)
            xmin = float(np.min(ex) - tol)
            xmax = float(np.max(ex) + tol)
            ymin = float(np.min(ey) - tol)
            ymax = float(np.max(ey) + tol)

            q_edge_pairs: List[Tuple[float, int]] = []
            for n in q_nodes_all:
                x = float(node_x[n])
                y = float(node_y[n])
                if x < xmin or x > xmax or y < ymin or y > ymax:
                    continue
                d, s = _polyline_distance_and_s(edge_pts, float(node_x[n]), float(node_y[n]))
                if d <= tol:
                    q_edge_pairs.append((s, int(n)))
            if len(q_edge_pairs) < 2:
                continue

            q_edge_pairs.sort(key=lambda p: p[0])
            q_s = np.asarray([p[0] for p in q_edge_pairs], dtype=np.float64)
            q_n = np.asarray([p[1] for p in q_edge_pairs], dtype=np.int32)

            # Pick adjacent region with strongest geometric support on this edge.
            best_adj = None
            best_count = 0
            for other_rid, other_nodes in region_boundary_nodes.items():
                if int(other_rid) == int(qrid):
                    continue
                count = 0
                for n in other_nodes:
                    x = float(node_x[n])
                    y = float(node_y[n])
                    if x < xmin or x > xmax or y < ymin or y > ymax:
                        continue
                    d, _s = _polyline_distance_and_s(edge_pts, x, y)
                    if d <= tol:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_adj = int(other_rid)
            if best_adj is None or best_count <= 0:
                continue

            for n in region_boundary_nodes.get(best_adj, set()):
                x = float(node_x[n])
                y = float(node_y[n])
                if x < xmin or x > xmax or y < ymin or y > ymax:
                    continue
                d, s = _polyline_distance_and_s(edge_pts, x, y)
                if d > tol:
                    continue
                j = int(np.argmin(np.abs(q_s - s)))
                ref = int(q_n[j])
                node_x[n] = node_x[ref]
                node_y[n] = node_y[ref]

    # Weld after snapping so interfaces become topologically shared.
    remap_conn = (
        np.asarray(mesh.cell_nodes, dtype=np.int32),
        np.asarray(mesh.cell_face_nodes, dtype=np.int32),
    )
    node_x2, node_y2, (cell_nodes2, cell_face_nodes2) = _weld_mesh_nodes(
        node_x,
        node_y,
        remap_conn[0],
        remap_conn[1],
        tol=1.0e-6,
    )

    return MeshResult(
        node_x=node_x2,
        node_y=node_y2,
        node_z=np.zeros_like(node_x2),
        cell_nodes=cell_nodes2,
        cell_face_offsets=np.asarray(mesh.cell_face_offsets, dtype=np.int32),
        cell_face_nodes=cell_face_nodes2,
        cell_type=np.asarray(mesh.cell_type, dtype=object),
        region_id=np.asarray(mesh.region_id, dtype=np.int32),
        target_size=np.asarray(mesh.target_size, dtype=np.float64),
        quality_summary=dict(mesh.quality_summary or {}),
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

            if region.default_cell_type in ("cartesian", "quadrilateral", "channel_generator") and not region_constraints and not region_exclusions:
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
                    if local_type in ("cartesian", "quadrilateral", "channel_generator"):
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


def _mfem_meshopt_available() -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec("hydra_mfem_meshopt") is not None
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

    def _opt_str_tuple(self, name: str, default: Tuple[str, ...]) -> Tuple[str, ...]:
        value = self._options.get(name)
        if value is None:
            return tuple(str(v) for v in default)
        if isinstance(value, str):
            raw_items = value.replace(";", ",").split(",")
        elif isinstance(value, (list, tuple)):
            raw_items = list(value)
        else:
            return tuple(str(v) for v in default)
        parsed: List[str] = []
        for item in raw_items:
            text = str(item).strip()
            if text:
                parsed.append(text)
        if not parsed:
            return tuple(str(v) for v in default)
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
        recombine_topology_passes = tuple(
            max(0, int(round(v)))
            for v in self._opt_float_tuple(
                "gmsh_quality_recombine_topology_passes",
                _env_csv_floats("BACKWATER_GMSH_QUALITY_RECOMBINE_TOPOLOGY_PASSES", (5.0, 12.0, 20.0)),
            )
        )
        recombine_min_quality = tuple(
            max(0.0, float(v))
            for v in self._opt_float_tuple(
                "gmsh_quality_recombine_minimum_quality",
                _env_csv_floats("BACKWATER_GMSH_QUALITY_RECOMBINE_MIN_QUALITY", (0.01, 0.03, 0.06)),
            )
        )
        random_factors = tuple(
            max(0.0, float(v))
            for v in self._opt_float_tuple(
                "gmsh_quality_random_factors",
                _env_csv_floats("BACKWATER_GMSH_QUALITY_RANDOM_FACTORS", (1.0e-9, 1.0e-7, 1.0e-6)),
            )
        )
        optimize_methods = tuple(
            str(v)
            for v in self._opt_str_tuple(
                "gmsh_quality_optimize_methods",
                _env_csv_strings("BACKWATER_GMSH_QUALITY_OPTIMIZE_METHODS", ("Laplace2D", "Relocate2D")),
            )
            if str(v).strip()
        )
        algorithm_switch_on_failure = self._opt_bool(
            "gmsh_algorithm_switch_on_failure",
            _env_bool("BACKWATER_GMSH_ALGO_SWITCH_ON_FAILURE", True),
        )
        recombine_node_repositioning = self._opt_bool(
            "gmsh_quality_recombine_node_repositioning",
            _env_bool("BACKWATER_GMSH_RECOMBINE_NODE_REPOSITIONING", True),
        )
        if not size_scales:
            size_scales = (1.0,)
        if not smooth_increments:
            smooth_increments = (0,)
        if not recombine_topology_passes:
            recombine_topology_passes = (5,)
        if not recombine_min_quality:
            recombine_min_quality = (0.01,)
        if not random_factors:
            random_factors = (1.0e-9,)
        # Guard against no-op retry ladders (e.g. "1.0" and "0").
        # When iterative quality is enabled, ensure attempts explore distinct
        # candidates even if the UI left legacy single-value defaults.
        if enabled and max_iterations > 1:
            if all(abs(float(v) - 1.0) <= 1.0e-12 for v in size_scales):
                size_scales = (1.0, 0.9, 0.8, 0.7)
            if all(int(v) == 0 for v in smooth_increments):
                smooth_increments = (0, 2, 4, 6)
            if len(recombine_topology_passes) == 1:
                recombine_topology_passes = (recombine_topology_passes[0], max(8, recombine_topology_passes[0] * 2))
            if len(recombine_min_quality) == 1:
                recombine_min_quality = (recombine_min_quality[0], max(0.02, recombine_min_quality[0] * 1.5))
            if len(random_factors) == 1:
                random_factors = (random_factors[0], max(1.0e-8, random_factors[0] * 100.0))
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
            recombine_topology_passes=recombine_topology_passes,
            recombine_min_quality=recombine_min_quality,
            random_factors=random_factors,
            optimize_methods=optimize_methods,
            algorithm_switch_on_failure=bool(algorithm_switch_on_failure),
            recombine_node_repositioning=bool(recombine_node_repositioning),
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
        checkpoint_path = str(self._options.get("gmsh_quality_checkpoint_path", "") or "").strip()

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
            attempt_errors: List[str] = []
            scale_i = 0
            smooth_i = 0
            had_passing_candidate = False
            last_attempt_duration_s: Optional[float] = None
            hit_time_budget = False

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

            recomb_topology_ladder = [int(v) for v in quality_cfg.recombine_topology_passes if int(v) >= 0]
            if not recomb_topology_ladder:
                recomb_topology_ladder = [5]
            recomb_min_quality_ladder = [max(0.0, float(v)) for v in quality_cfg.recombine_min_quality]
            if not recomb_min_quality_ladder:
                recomb_min_quality_ladder = [0.01]
            random_factor_ladder = [max(0.0, float(v)) for v in quality_cfg.random_factors]
            if not random_factor_ladder:
                random_factor_ladder = [1.0e-9]

            while attempts < quality_cfg.max_iterations:
                elapsed = time.perf_counter() - start_t
                if elapsed >= quality_cfg.time_limit_s:
                    hit_time_budget = True
                    break

                # Avoid starting a fresh attempt when little time remains. A
                # single Gmsh attempt is non-interruptible, so launching a retry
                # too close to the deadline can overrun and get killed by the
                # outer watchdog before best-candidate export runs.
                remaining_s = max(0.0, quality_cfg.time_limit_s - elapsed)
                if last_attempt_duration_s is not None and attempts > 0:
                    min_retry_window_s = max(2.0, 0.75 * float(last_attempt_duration_s))
                    if remaining_s < min_retry_window_s:
                        hit_time_budget = True
                        warnings.warn(
                            "Gmsh quality loop stopping retries early due to low remaining budget "
                            f"(remaining={remaining_s:.2f}s, needed~{min_retry_window_s:.2f}s); "
                            "returning best available candidate.",
                            RuntimeWarning,
                        )
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
                recomb_topology_try = recomb_topology_ladder[attempts % len(recomb_topology_ladder)]
                recomb_min_quality_try = recomb_min_quality_ladder[attempts % len(recomb_min_quality_ladder)]
                random_factor_try = random_factor_ladder[attempts % len(random_factor_ladder)]

                attempt_start_t = time.perf_counter()
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
                            recombine_optimize_topology=int(recomb_topology_try),
                            recombine_node_repositioning=bool(quality_cfg.recombine_node_repositioning),
                            recombine_minimum_quality=float(recomb_min_quality_try),
                            optimize_methods=tuple(quality_cfg.optimize_methods),
                            random_factor=float(random_factor_try),
                            algorithm_switch_on_failure=bool(quality_cfg.algorithm_switch_on_failure),
                        ),
                        "Gmsh",
                    )
                    stats = _face_mesh_quality_stats(mesh, quality_cfg)
                    score = _gmsh_quality_score(stats, quality_cfg)

                    attempt_summary = {
                        "attempts": int(attempts + 1),
                        "strict_requested": bool(quality_cfg.strict),
                        "had_passing_candidate": bool(_gmsh_quality_passes(stats, quality_cfg)),
                        "best_stats": dict(stats),
                        "recombine_topology_passes": int(recomb_topology_try),
                        "recombine_minimum_quality": float(recomb_min_quality_try),
                        "random_factor": float(random_factor_try),
                        "optimize_methods": list(quality_cfg.optimize_methods),
                        "checkpoint": True,
                    }
                    if checkpoint_path:
                        try:
                            _write_mesh_checkpoint_npz(checkpoint_path, mesh, attempt_summary)
                        except Exception as cp_exc:
                            warnings.warn(
                                f"Gmsh quality checkpoint write failed (attempt {attempts + 1}): {cp_exc}",
                                RuntimeWarning,
                            )

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
                    err_msg = (
                        f"Gmsh quality attempt {attempts + 1} failed for tri={tri_try}, quad={quad_try}, "
                        f"recomb={recomb_try}, topo={int(recomb_topology_try)}, minq={float(recomb_min_quality_try):.3f}, "
                        f"rand={float(random_factor_try):.2e}, size_scale={size_scale:.3f}, "
                        f"smooth={smoothing_passes + int(smooth_inc)}: {exc}"
                    )
                    attempt_errors.append(err_msg)
                    warnings.warn(
                        err_msg,
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

                last_attempt_duration_s = max(0.0, time.perf_counter() - attempt_start_t)
                attempts += 1

            if best_mesh is None or best_stats is None:
                # Best-effort fallback: regardless of iterative quality failures,
                # run one plain baseline build so downstream export still has a mesh
                # whenever geometry is meshable at all.
                try:
                    gmsh.clear()
                    gmsh.model.add("swe2d_best_effort_fallback")
                    fallback_mesh = _require_nonempty_mesh(
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
                            recombine_optimize_topology=int(recomb_topology_ladder[0]),
                            recombine_node_repositioning=bool(quality_cfg.recombine_node_repositioning),
                            recombine_minimum_quality=float(recomb_min_quality_ladder[0]),
                            optimize_methods=tuple(quality_cfg.optimize_methods),
                            random_factor=float(random_factor_ladder[0]),
                            algorithm_switch_on_failure=bool(quality_cfg.algorithm_switch_on_failure),
                        ),
                        "Gmsh",
                    )
                    fallback_stats = _face_mesh_quality_stats(fallback_mesh, quality_cfg)
                    fallback_mesh.quality_summary = {
                        "attempts": int(attempts + 1),
                        "strict_requested": bool(quality_cfg.strict),
                        "had_passing_candidate": bool(_gmsh_quality_passes(fallback_stats, quality_cfg)),
                        "best_stats": dict(fallback_stats),
                        "best_effort_fallback": True,
                        "time_budget_exhausted": bool(hit_time_budget),
                    }
                    if checkpoint_path:
                        try:
                            _write_mesh_checkpoint_npz(
                                checkpoint_path,
                                fallback_mesh,
                                fallback_mesh.quality_summary,
                            )
                        except Exception as cp_exc:
                            warnings.warn(
                                f"Gmsh fallback checkpoint write failed: {cp_exc}",
                                RuntimeWarning,
                            )
                    warnings.warn(
                        "Gmsh quality loop produced no valid candidate during iterative retries; "
                        "using best-effort fallback mesh for export.",
                        RuntimeWarning,
                    )
                    return fallback_mesh
                except Exception as fallback_exc:
                    tail = "; ".join(attempt_errors[-3:]) if attempt_errors else "no attempt diagnostics"
                    raise RuntimeError(
                        "Gmsh quality loop produced no valid non-empty mesh candidate, and "
                        f"best-effort fallback also failed: {fallback_exc}. "
                        f"Recent attempt errors: {tail}"
                    )

            if had_passing_candidate:
                best_mesh.quality_summary = {
                    "attempts": int(attempts),
                    "strict_requested": bool(quality_cfg.strict),
                    "had_passing_candidate": True,
                    "best_stats": dict(best_stats),
                    "time_budget_exhausted": bool(hit_time_budget),
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
                "time_budget_exhausted": bool(hit_time_budget),
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
        recombine_optimize_topology: int = 5,
        recombine_node_repositioning: bool = True,
        recombine_minimum_quality: float = 0.01,
        optimize_methods: Tuple[str, ...] = (),
        random_factor: float = 1.0e-9,
        algorithm_switch_on_failure: bool = True,
    ) -> MeshResult:
        arc_mode = str(self._options.get("gmsh_arc_mode", "hard_embed") or "hard_embed").strip().lower()
        if arc_mode not in {"hard_embed", "soft_size_hint", "disabled"}:
            arc_mode = "hard_embed"
        mesh_size_min = max(0.0, self._opt_float("gmsh_mesh_size_min", 0.0))
        tolerance_edge_length = max(0.0, self._opt_float("gmsh_tolerance_edge_length", 0.0))
        mesh_size_from_points = self._opt_bool("gmsh_mesh_size_from_points", True)
        arc_soft_size_factor = min(1.0, max(0.05, self._opt_float("gmsh_arc_soft_size_factor", 0.5)))
        arc_soft_dist_factor = max(0.1, self._opt_float("gmsh_arc_soft_dist_factor", 2.0))

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
        arc_soft_groups: Dict[Tuple[float, float], Dict[str, List[int]]] = {}
        if model.arcs and arc_mode != "disabled":
            arc_hard_curve_tags: List[int] = []
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

            channel_region_ids = {
                int(r.region_id)
                for r in model.regions
                if str(r.default_cell_type).strip().lower() == "channel_generator"
            }

            def _arc_mode_for(arc: ConceptualArc) -> str:
                mode_local = str(getattr(arc, "arc_mode_override", "") or "").strip().lower()
                if mode_local in {"hard_embed", "soft_size_hint", "disabled"}:
                    return mode_local

                role_local = str(getattr(arc, "arc_role", "") or "").strip().lower()
                in_channel_region = int(getattr(arc, "region_id", -1)) in channel_region_ids
                if in_channel_region and role_local in {"left_bank", "right_bank"}:
                    return "hard_embed"
                if in_channel_region and role_local == "centerline":
                    return "soft_size_hint"

                if bool(getattr(arc, "use_global_arc_ctrl", True)):
                    return arc_mode
                return arc_mode

            def _arc_soft_size_factor_for(arc: ConceptualArc) -> float:
                if bool(getattr(arc, "use_global_arc_ctrl", True)):
                    return float(arc_soft_size_factor)
                cand = getattr(arc, "arc_soft_size_override", None)
                if cand is None:
                    return float(arc_soft_size_factor)
                return min(1.0, max(0.05, float(cand)))

            def _arc_soft_dist_factor_for(arc: ConceptualArc) -> float:
                if bool(getattr(arc, "use_global_arc_ctrl", True)):
                    return float(arc_soft_dist_factor)
                cand = getattr(arc, "arc_soft_dist_override", None)
                if cand is None:
                    return float(arc_soft_dist_factor)
                return max(0.1, float(cand))

            for arc in model.arcs:
                pts_xy = list(arc.points_xy or [])
                arc_point_tags_local: List[int] = []
                arc_curve_tags_local: List[int] = []
                if len(pts_xy) >= 2:
                    gp_tags: List[int] = []
                    for x, y in pts_xy:
                        ptag = _geo_pt(float(x), float(y), arc_lc)
                        if not gp_tags or gp_tags[-1] != ptag:
                            gp_tags.append(ptag)
                    arc_point_tags_local.extend(gp_tags)
                    for i in range(len(gp_tags) - 1):
                        seg = _geo_seg(gp_tags[i], gp_tags[i + 1])
                        seg_abs = abs(int(seg))
                        arc_curve_tags_local.append(seg_abs)
                else:
                    # Backward-compatible fallback: endpoint IDs in topo_nodes.
                    p0_xy = node_xy.get(arc.node0)
                    p1_xy = node_xy.get(arc.node1)
                    if p0_xy is None or p1_xy is None:
                        continue
                    gp0 = _geo_pt(p0_xy[0], p0_xy[1], arc_lc)
                    gp1 = _geo_pt(p1_xy[0], p1_xy[1], arc_lc)
                    arc_point_tags_local.extend([gp0, gp1])
                    arc_curve_tags_local.append(abs(int(_geo_seg(gp0, gp1))))

                mode_local = _arc_mode_for(arc)
                if mode_local == "hard_embed":
                    arc_hard_curve_tags.extend(arc_curve_tags_local)
                elif mode_local == "soft_size_hint":
                    size_factor_local = _arc_soft_size_factor_for(arc)
                    dist_factor_local = _arc_soft_dist_factor_for(arc)
                    key = (round(float(size_factor_local), 6), round(float(dist_factor_local), 6))
                    group = arc_soft_groups.setdefault(key, {"curves": [], "points": []})
                    group["curves"].extend(arc_curve_tags_local)
                    group["points"].extend(arc_point_tags_local)

            if arc_hard_curve_tags:
                arc_curve_tags = sorted({int(tag) for tag in arc_hard_curve_tags if int(tag) > 0})
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

        if constraint_point_lists or arc_soft_groups:
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

            if arc_soft_groups:
                min_region_size = max(min(float(sz) for (_, _, sz) in surface_meta), 1.0e-9)
                for (size_factor_local, dist_factor_local), group in arc_soft_groups.items():
                    arc_curves = sorted({int(t) for t in group.get("curves", []) if int(t) > 0})
                    arc_pts = sorted({int(t) for t in group.get("points", []) if int(t) > 0})
                    if not arc_curves and not arc_pts:
                        continue

                    arc_size = max(mesh_size_min, min_region_size * float(size_factor_local))
                    arc_dist = max(arc_size, float(dist_factor_local) * arc_size)

                    f_dist = gmsh.model.mesh.field.add("Distance")
                    if arc_curves:
                        gmsh.model.mesh.field.setNumbers(f_dist, "CurvesList", arc_curves)
                    if arc_pts:
                        gmsh.model.mesh.field.setNumbers(f_dist, "PointsList", arc_pts)

                    f_thresh = gmsh.model.mesh.field.add("Threshold")
                    gmsh.model.mesh.field.setNumber(f_thresh, "InField", float(f_dist))
                    gmsh.model.mesh.field.setNumber(f_thresh, "SizeMin", float(arc_size))
                    gmsh.model.mesh.field.setNumber(f_thresh, "SizeMax", float(max_region_size))
                    gmsh.model.mesh.field.setNumber(f_thresh, "DistMin", 0.0)
                    gmsh.model.mesh.field.setNumber(f_thresh, "DistMax", float(arc_dist))
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
            elif ctype in {"quadrilateral", "channel_generator"}:
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
        gmsh.option.setNumber("Mesh.RecombineOptimizeTopology", float(max(0, int(recombine_optimize_topology))))
        gmsh.option.setNumber("Mesh.RecombineNodeRepositioning", 1.0 if recombine_node_repositioning else 0.0)
        gmsh.option.setNumber("Mesh.RecombineMinimumQuality", max(0.0, float(recombine_minimum_quality)))
        gmsh.option.setNumber("Mesh.Smoothing", float(smoothing_passes))
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1.0 if optimize_netgen else 0.0)
        gmsh.option.setNumber("Mesh.AlgorithmSwitchOnFailure", 1.0 if algorithm_switch_on_failure else 0.0)
        gmsh.option.setNumber("Mesh.RandomFactor", max(0.0, float(random_factor)))
        gmsh.option.setNumber("Mesh.MeshSizeMin", float(mesh_size_min))
        gmsh.option.setNumber("Mesh.ToleranceEdgeLength", float(tolerance_edge_length))
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1.0 if mesh_size_from_points else 0.0)

        # ---- 6. Generate -----------------------------------------------
        gmsh.model.mesh.generate(2)
        if want_recombine:
            try:
                gmsh.model.mesh.recombine()
            except Exception:
                pass
        if optimize_iters > 0:
            methods = tuple(str(m).strip() for m in (optimize_methods or ()) if str(m).strip())
            if not methods:
                methods = ("Laplace2D",)
            for method in methods:
                try:
                    gmsh.model.mesh.optimize(method, niter=int(optimize_iters))
                except TypeError:
                    gmsh.model.mesh.optimize(method)

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
            - optional channel metadata: region_id, arc_role
                        - optional per-arc controls:
                            use_global_arc_ctrl (0/1), arc_mode_override
                            (hard_embed|soft_size_hint|disabled),
                            arc_soft_size_override, arc_soft_dist_override
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
        def _as_bool(v, default: bool) -> bool:
            if v in (None, ""):
                return bool(default)
            if isinstance(v, bool):
                return bool(v)
            txt = str(v).strip().lower()
            if txt in {"1", "true", "yes", "on", "y"}:
                return True
            if txt in {"0", "false", "no", "off", "n"}:
                return False
            try:
                return float(v) != 0.0
            except Exception:
                return bool(default)

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
            region_id = _as_int(ft["region_id"], -1) if "region_id" in arc_fields else -1
            arc_role = None
            if "arc_role" in arc_fields:
                role_txt = str(ft["arc_role"] or "").strip().lower()
                if role_txt in {"centerline", "left_bank", "right_bank", "breakline"}:
                    arc_role = role_txt
            use_global_arc_ctrl = _as_bool(ft["use_global_arc_ctrl"], True) if "use_global_arc_ctrl" in arc_fields else True

            arc_mode_override = None
            if "arc_mode_override" in arc_fields:
                mode_txt = str(ft["arc_mode_override"] or "").strip().lower()
                if mode_txt in {"hard_embed", "soft_size_hint", "disabled"}:
                    arc_mode_override = mode_txt

            arc_soft_size_override = None
            if "arc_soft_size_override" in arc_fields:
                cand = _as_float(ft["arc_soft_size_override"], -1.0)
                if cand > 0.0:
                    arc_soft_size_override = float(cand)

            arc_soft_dist_override = None
            if "arc_soft_dist_override" in arc_fields:
                cand = _as_float(ft["arc_soft_dist_override"], -1.0)
                if cand > 0.0:
                    arc_soft_dist_override = float(cand)

            arcs.append(
                ConceptualArc(
                    arc_id=a_id,
                    node0=n0,
                    node1=n1,
                    region_id=region_id,
                    arc_role=arc_role,
                    points_xy=pts if len(pts) >= 2 else None,
                    use_global_arc_ctrl=use_global_arc_ctrl,
                    arc_mode_override=arc_mode_override,
                    arc_soft_size_override=arc_soft_size_override,
                    arc_soft_dist_override=arc_soft_dist_override,
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

    @staticmethod
    def _is_ccw(ring: List[Tuple[float, float]]) -> bool:
        """Return True if the ring has counter-clockwise winding (positive area)."""
        area = _polygon_area_xy(
            np.asarray([p[0] for p in ring]),
            np.asarray([p[1] for p in ring]),
        )
        return area > 0.0

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
        debug_dump_dir = str(
            self._options.get(
                "tqmesh_debug_dump_dir",
                os.environ.get("BACKWATER_TQMESH_DEBUG_DUMP_DIR", ""),
            )
            or ""
        ).strip()

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
            ring_initial = list(ring)
            boundary_split_max_length = _as_float(
                self._options.get("tqmesh_boundary_split_max_length"),
                _env_float("BACKWATER_TQMESH_BOUNDARY_SPLIT_MAX_LENGTH", 0.0),
            )
            if (not np.isfinite(boundary_split_max_length)) or boundary_split_max_length <= 0.0:
                boundary_split_max_length = 0.0
            region_constraints = _constraints_for_region(model, ring)
            region_exclusions = _region_exclusion_zones(model, region, ring)
            fixed_edge_lines = _breakline_fixed_edges_for_region(model, region, ring)
            fixed_edge_lines_raw = [list(line) for line in fixed_edge_lines]
            if fixed_edge_lines:
                ring, fixed_edge_lines = _snap_and_split_boundary_for_breaklines(
                    ring,
                    fixed_edge_lines,
                    vertex_snap_tol=0.1,
                )
            ring_after_breakline_preprocess = list(ring)
            fixed_edge_lines_after_breakline_preprocess = [list(line) for line in fixed_edge_lines]
            if boundary_split_max_length > 0.0 and fixed_edge_lines:
                split_lines: List[List[Tuple[float, float]]] = []
                for line in fixed_edge_lines:
                    densified = _split_polyline_max_segment_length(line, boundary_split_max_length)
                    if len(densified) >= 2:
                        split_lines.append(densified)
                fixed_edge_lines = split_lines
            fixed_edge_lines_after_densify = [list(line) for line in fixed_edge_lines]

            quad_controls = None
            quad_boundary = None
            if ctype in ("quadrilateral", "cartesian"):
                quad_setup = self._quad_controls_for_region(model, region)
                if quad_setup is not None:
                    quad_boundary, quad_controls = quad_setup

            full_quad_align = self._opt_bool(
                "tqmesh_quad_full_region_flow_align",
                _env_bool("BACKWATER_TQMESH_QUAD_FULL_REGION_FLOW_ALIGN", True),
            )
            quad_full_region_max_cells = _as_int(
                self._options.get("tqmesh_quad_full_region_max_cells"),
                _as_int(os.environ.get("BACKWATER_TQMESH_QUAD_FULL_REGION_MAX_CELLS", 250000), 250000),
            )
            if quad_full_region_max_cells <= 0:
                quad_full_region_max_cells = 0
            if quad_controls is not None and full_quad_align:
                # Build a full-region flow-aligned quad block using transfinite
                # interpolation and quad-edge spacing/layer controls.
                block = _structured_quad_region_mesh(
                    region,
                    quad_controls,
                    max_cells=quad_full_region_max_cells,
                )
                if block is not None:
                    vx, vy, tris, face_offsets, face_nodes, _target_sizes = block
                    offset = len(all_vx)
                    all_vx.extend(vx.tolist())
                    all_vy.extend(vy.tolist())

                    shifted_faces = np.asarray(face_nodes, dtype=np.int32) + int(offset)
                    for ci in range(int(face_offsets.size - 1)):
                        s = int(face_offsets[ci])
                        e = int(face_offsets[ci + 1])
                        poly = shifted_faces[s:e].tolist()
                        if len(poly) == 4:
                            all_quads.extend(poly)
                            all_ctype.append("quadrilateral")
                            all_rid.append(region.region_id)
                            all_size.append(target_size)
                        elif len(poly) == 3:
                            all_tris.extend(poly)
                            all_ctype.append("triangular")
                            all_rid.append(region.region_id)
                            all_size.append(target_size)
                    continue

            # Exterior boundary — TQMesh expects CCW; ensure correct winding
            ext_verts = list(quad_boundary) if quad_boundary is not None else ring
            ext_verts_raw_count = len(ext_verts)
            protected_boundary_points = _boundary_contact_vertices(
                ext_verts,
                fixed_edge_lines_after_breakline_preprocess,
                tol=max(1.0e-6, 1.0e-3 * float(target_size)),
            )
            if quad_controls is None:
                # Preserve boundary coincidence points used by fixed edges.
                if not protected_boundary_points:
                    simp_factor = max(_env_float("BACKWATER_TQMESH_BOUNDARY_SIMPLIFY_FACTOR", 0.35), 0.0)
                    simp_tol = float(target_size) * simp_factor
                    simp_max = max(8, int(round(_env_float("BACKWATER_TQMESH_BOUNDARY_MAX_VERTS", 64.0))))
                    ext_verts = _simplify_closed_ring(ext_verts, tol=simp_tol, max_vertices=simp_max)
            ext_verts = _sanitize_closed_ring(
                ext_verts,
                length_tol=max(1.0e-6, 1.0e-3 * float(target_size)),
                collinear_tol=max(1.0e-6, 1.5e-3 * float(target_size)),
                protect_points=protected_boundary_points,
                protect_tol=max(1.0e-6, 1.0e-3 * float(target_size)),
            )
            ext_verts_post_sanitize_count = len(ext_verts)
            ext_verts = _stitch_boundary_microchains(
                ext_verts,
                target_size=float(target_size),
                protect_points=protected_boundary_points,
                protect_tol=max(1.0e-6, 1.0e-3 * float(target_size)),
            )
            ext_verts_post_stitch_count = len(ext_verts)
            resample_applied = False
            resample_max_seg = 0.0
            if quad_controls is None and _env_bool("BACKWATER_TQMESH_BOUNDARY_RESAMPLE", False):
                seg_lens = []
                for i in range(len(ext_verts)):
                    a = ext_verts[i]
                    b = ext_verts[(i + 1) % len(ext_verts)]
                    seg_lens.append(float(np.hypot(b[0] - a[0], b[1] - a[1])))
                max_seg = max(seg_lens) if seg_lens else 0.0
                resample_max_seg = float(max_seg)
                # Resample when long segments could destabilize front closure.
                if max_seg > 4.0 * float(target_size):
                    ext_verts = _resample_closed_ring(ext_verts, target_step=max(0.75 * float(target_size), 1.0e-6))
                    ext_verts = _sanitize_closed_ring(
                        ext_verts,
                        length_tol=max(1.0e-6, 1.0e-3 * float(target_size)),
                        collinear_tol=1.0e-12,
                    )
                    resample_applied = True
            ext_verts_post_resample_count = len(ext_verts)
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
                hring = _sanitize_closed_ring(
                    hring,
                    length_tol=max(1.0e-6, 1.0e-3 * float(target_size)),
                    collinear_tol=max(1.0e-6, 1.5e-3 * float(target_size)),
                )
                hring = _stitch_boundary_microchains(hring, target_size=float(target_size))
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
                fixed_edges=[[[float(x), float(y)] for (x, y) in line] for line in fixed_edge_lines],
                target_size=target_size,
            )
            breakline_fixed_edges_enabled = self._opt_bool(
                "tqmesh_breakline_fixed_edges",
                _env_bool("BACKWATER_TQMESH_BREAKLINE_FIXED_EDGES", True),
            )
            breakline_fixed_edges_strict = self._opt_bool(
                "tqmesh_breakline_fixed_edges_strict",
                _env_bool("BACKWATER_TQMESH_BREAKLINE_FIXED_EDGES_STRICT", False),
            )
            strict_fixed_edge_region = bool(
                breakline_fixed_edges_enabled
                and breakline_fixed_edges_strict
                and len(base_args["fixed_edges"]) > 0
            )
            if not breakline_fixed_edges_enabled:
                base_args["fixed_edges"] = []

            boundary_split_for_call = float(boundary_split_max_length)

            fixed_edge_variants: List[Tuple[str, List[List[List[float]]]]] = []
            if len(base_args["fixed_edges"]) > 0:
                fixed_edge_variants.append(("with-fixed-edges", base_args["fixed_edges"]))
                if not breakline_fixed_edges_strict:
                    fixed_edge_variants.append(("no-fixed-edges", []))
            else:
                fixed_edge_variants.append(("no-fixed-edges", []))

            want_quads = ctype in ("quadrilateral", "cartesian")
            has_fixed_edges = len(base_args["fixed_edges"]) > 0
            requested_smooth = 0 if has_fixed_edges else 3
            tri_only_smooth = 0 if has_fixed_edges else 1

            attempts = [
                ("requested", active_quad_layers, want_quads, requested_smooth),
            ]
            if active_quad_layers:
                attempts.append(("no-quad-layers", [], want_quads, requested_smooth))
            if want_quads:
                attempts.append(("triangles-only", [], False, tri_only_smooth))
            attempts.append(("minimal", [], False, 0))

            result = None
            errors: List[str] = []
            debug_attempts: List[Dict[str, object]] = []
            seen_cfg = set()
            used_label = "requested"
            used_quality: Optional[Dict[str, float]] = None
            best_nonpassing = None
            best_nonpassing_score = -float("inf")
            microchain_retry_done = set()
            for fixed_label, fixed_edges_try in fixed_edge_variants:
                for label, quad_layers_try, tri_to_quad_try, n_smooth_try in attempts:
                    for size_scale in quality_cfg.size_scales:
                        target_try = max(target_size * max(float(size_scale), 1e-6), 1e-10)
                        csz_try = [max(float(cs) * max(float(size_scale), 1e-6), 1e-10) for cs in constraint_sizes_list]
                        for ds in quality_cfg.smooth_increments:
                            smooth_try = max(0, int(n_smooth_try) + int(ds))
                            cfg_key = (
                                fixed_label,
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
                                    fixed_edges=fixed_edges_try,
                                    target_size=target_try,
                                    quad_layers=quad_layers_try,
                                    tri_to_quad=tri_to_quad_try,
                                    n_smooth=smooth_try,
                                    boundary_split_max_length=float(boundary_split_for_call),
                                )
                            except Exception as exc:
                                exc_txt = str(exc)
                                debug_attempts.append(
                                    {
                                        "fixed_variant": str(fixed_label),
                                        "attempt_label": str(label),
                                        "target_size": float(target_try),
                                        "smooth": int(smooth_try),
                                        "tri_to_quad": bool(tri_to_quad_try),
                                        "quad_layers_count": int(len(quad_layers_try)),
                                        "fixed_edges_count": int(len(fixed_edges_try)),
                                        "fixed_edge_vertices": int(sum(len(line) for line in fixed_edges_try)),
                                        "status": "exception",
                                        "error": exc_txt,
                                    }
                                )

                                # Targeted strict-mode rescue path for known boundary
                                # microchain completeness failures in region 2.
                                can_microchain_retry = (
                                    int(region.region_id) == 2
                                    and str(fixed_label) == "with-fixed-edges"
                                    and len(fixed_edges_try) > 0
                                    and "invalid_boundary_edge_samples" in exc_txt
                                    and cfg_key not in microchain_retry_done
                                )
                                if can_microchain_retry:
                                    microchain_retry_done.add(cfg_key)
                                    focus_points = _parse_invalid_boundary_edge_sample_points(exc_txt)
                                    ext_before = [(float(v[0]), float(v[1])) for v in base_args["ext_verts"]]
                                    ext_retry = _collapse_boundary_microchains_near_points(
                                        ext_before,
                                        focus_points,
                                        target_size=float(target_try),
                                        protect_points=protected_boundary_points,
                                        protect_tol=max(1.0e-6, 1.0e-3 * float(target_try)),
                                    )

                                    candidate_rings: List[Tuple[str, List[Tuple[float, float]]]] = []
                                    ring_seen = set()

                                    def _push_candidate(tag: str, rr: List[Tuple[float, float]]) -> None:
                                        if len(rr) < 3:
                                            return
                                        key = _ring_key(rr, ndigits=6)
                                        if key in ring_seen:
                                            return
                                        ring_seen.add(key)
                                        candidate_rings.append((tag, rr))

                                    if len(ext_retry) >= 3 and ext_retry != ext_before:
                                        _push_candidate("microchain-merge", ext_retry)
                                    else:
                                        debug_attempts.append(
                                            {
                                                "fixed_variant": str(fixed_label),
                                                "attempt_label": f"{label}/microchain-merge",
                                                "target_size": float(target_try),
                                                "smooth": int(smooth_try),
                                                "tri_to_quad": bool(tri_to_quad_try),
                                                "quad_layers_count": int(len(quad_layers_try)),
                                                "fixed_edges_count": int(len(fixed_edges_try)),
                                                "fixed_edge_vertices": int(sum(len(line) for line in fixed_edges_try)),
                                                "status": "skipped",
                                                "reason": "microchain-no-geometry-change",
                                                "focus_points": [[float(x), float(y)] for x, y in focus_points],
                                                "ext_vertices_before": int(len(base_args["ext_verts"])),
                                                "ext_vertices_after": int(len(ext_retry)),
                                            }
                                        )

                                    focus_insert = _insert_focus_points_on_ring_segments(
                                        ext_before,
                                        focus_points,
                                        max_dist=max(1.5 * float(target_try), 12.0),
                                    )
                                    if len(focus_insert) >= 3 and focus_insert != ext_before:
                                        _push_candidate("focus-split", focus_insert)
                                    else:
                                        debug_attempts.append(
                                            {
                                                "fixed_variant": str(fixed_label),
                                                "attempt_label": f"{label}/focus-split",
                                                "target_size": float(target_try),
                                                "smooth": int(smooth_try),
                                                "tri_to_quad": bool(tri_to_quad_try),
                                                "quad_layers_count": int(len(quad_layers_try)),
                                                "fixed_edges_count": int(len(fixed_edges_try)),
                                                "fixed_edge_vertices": int(sum(len(line) for line in fixed_edges_try)),
                                                "status": "skipped",
                                                "reason": "focus-split-no-geometry-change",
                                                "focus_points": [[float(x), float(y)] for x, y in focus_points],
                                                "ext_vertices_before": int(len(base_args["ext_verts"])),
                                                "ext_vertices_after": int(len(focus_insert)),
                                            }
                                        )

                                    jitter_tries = _as_int(
                                        self._options.get("tqmesh_strict_local_jitter_tries"),
                                        _as_int(os.environ.get("BACKWATER_TQMESH_STRICT_LOCAL_JITTER_TRIES", 3), 3),
                                    )
                                    jitter_tries = max(0, int(jitter_tries))
                                    jitter_frac = _as_float(
                                        self._options.get("tqmesh_strict_local_jitter_frac"),
                                        _env_float("BACKWATER_TQMESH_STRICT_LOCAL_JITTER_FRAC", 0.02),
                                    )
                                    jitter_frac = min(max(float(jitter_frac), 1.0e-5), 0.2)
                                    for ji in range(jitter_tries):
                                        js = max(1.0e-4, float(target_try) * jitter_frac * float(ji + 1))
                                        jitter_base = focus_insert if len(focus_insert) >= 3 else ext_before
                                        ext_j = _jitter_boundary_vertices_near_points(
                                            jitter_base,
                                            focus_points,
                                            jitter_scale=js,
                                            variant_index=ji,
                                            protect_points=protected_boundary_points,
                                            protect_tol=max(1.0e-6, 1.0e-3 * float(target_try)),
                                        )
                                        ext_j = _collapse_boundary_microchains_near_points(
                                            ext_j,
                                            focus_points,
                                            target_size=float(target_try),
                                            protect_points=protected_boundary_points,
                                            protect_tol=max(1.0e-6, 1.0e-3 * float(target_try)),
                                        )
                                        if len(ext_j) >= 3 and ext_j != ext_before:
                                            _push_candidate(f"local-jitter-{ji + 1}", ext_j)
                                        else:
                                            debug_attempts.append(
                                                {
                                                    "fixed_variant": str(fixed_label),
                                                    "attempt_label": f"{label}/local-jitter-{ji + 1}",
                                                    "target_size": float(target_try),
                                                    "smooth": int(smooth_try),
                                                    "tri_to_quad": bool(tri_to_quad_try),
                                                    "quad_layers_count": int(len(quad_layers_try)),
                                                    "fixed_edges_count": int(len(fixed_edges_try)),
                                                    "fixed_edge_vertices": int(sum(len(line) for line in fixed_edges_try)),
                                                    "status": "skipped",
                                                    "reason": "jitter-no-geometry-change",
                                                    "jitter_scale": float(js),
                                                    "focus_points": [[float(x), float(y)] for x, y in focus_points],
                                                    "ext_vertices_before": int(len(base_args["ext_verts"])),
                                                    "ext_vertices_after": int(len(ext_j)),
                                                }
                                            )

                                    for variant_tag, ext_variant in candidate_rings:
                                        try:
                                            candidate = _tq.generate_triangular_mesh(
                                                ext_verts=[[float(v[0]), float(v[1])] for v in ext_variant],
                                                ext_colors=[1] * int(len(ext_variant)),
                                                int_boundaries=base_args["int_boundaries"],
                                                int_colors=base_args["int_colors"],
                                                constraint_verts=base_args["constraint_verts"],
                                                constraint_sizes=csz_try,
                                                fixed_edges=fixed_edges_try,
                                                target_size=target_try,
                                                quad_layers=quad_layers_try,
                                                tri_to_quad=tri_to_quad_try,
                                                n_smooth=smooth_try,
                                                boundary_split_max_length=float(boundary_split_for_call),
                                            )
                                        except Exception as exc2:
                                            debug_attempts.append(
                                                {
                                                    "fixed_variant": str(fixed_label),
                                                    "attempt_label": f"{label}/{variant_tag}",
                                                    "target_size": float(target_try),
                                                    "smooth": int(smooth_try),
                                                    "tri_to_quad": bool(tri_to_quad_try),
                                                    "quad_layers_count": int(len(quad_layers_try)),
                                                    "fixed_edges_count": int(len(fixed_edges_try)),
                                                    "fixed_edge_vertices": int(sum(len(line) for line in fixed_edges_try)),
                                                    "status": "exception",
                                                    "error": str(exc2),
                                                    "focus_points": [[float(x), float(y)] for x, y in focus_points],
                                                    "ext_vertices_before": int(len(base_args["ext_verts"])),
                                                    "ext_vertices_after": int(len(ext_variant)),
                                                }
                                            )
                                            errors.append(
                                                f"{fixed_label}/{label}/{variant_tag} (size={target_try:.4g}, smooth={smooth_try}): {exc2}"
                                            )
                                            continue

                                        cand_vx = np.asarray(candidate["verts_x"], dtype=np.float64)
                                        cand_vy = np.asarray(candidate["verts_y"], dtype=np.float64)
                                        cand_tris = np.asarray(candidate["triangles"], dtype=np.int32)
                                        cand_quads = np.asarray(candidate["quads"], dtype=np.int32)
                                        stats = _mesh_quality_stats(cand_vx, cand_vy, cand_tris, cand_quads)

                                        if _quality_passes(stats, quality_cfg):
                                            debug_attempts.append(
                                                {
                                                    "fixed_variant": str(fixed_label),
                                                    "attempt_label": f"{label}/{variant_tag}",
                                                    "target_size": float(target_try),
                                                    "smooth": int(smooth_try),
                                                    "tri_to_quad": bool(tri_to_quad_try),
                                                    "quad_layers_count": int(len(quad_layers_try)),
                                                    "fixed_edges_count": int(len(fixed_edges_try)),
                                                    "fixed_edge_vertices": int(sum(len(line) for line in fixed_edges_try)),
                                                    "status": "quality-pass",
                                                    "n_vertices": int(cand_vx.size),
                                                    "n_triangles": int(cand_tris.shape[0]) if cand_tris.ndim == 2 else 0,
                                                    "n_quads": int(cand_quads.shape[0]) if cand_quads.ndim == 2 else 0,
                                                    "quality": {
                                                        "min_angle_deg": float(stats["min_angle_deg"]),
                                                        "max_aspect_ratio": float(stats["max_aspect_ratio"]),
                                                        "min_area": float(stats["min_area"]),
                                                        "bbox_area": float(stats["bbox_area"]),
                                                    },
                                                    "focus_points": [[float(x), float(y)] for x, y in focus_points],
                                                    "ext_vertices_before": int(len(base_args["ext_verts"])),
                                                    "ext_vertices_after": int(len(ext_variant)),
                                                }
                                            )
                                            result = candidate
                                            used_label = (
                                                f"{fixed_label}/{label}/{variant_tag} "
                                                f"(size={target_try:.4g}, smooth={smooth_try})"
                                            )
                                            used_quality = stats
                                            break

                                        score = _quality_score(stats, quality_cfg)
                                        if score > best_nonpassing_score:
                                            best_nonpassing_score = score
                                            best_nonpassing = (
                                                candidate,
                                                f"{fixed_label}/{label}/{variant_tag}",
                                                target_try,
                                                smooth_try,
                                                stats,
                                            )
                                        debug_attempts.append(
                                            {
                                                "fixed_variant": str(fixed_label),
                                                "attempt_label": f"{label}/{variant_tag}",
                                                "target_size": float(target_try),
                                                "smooth": int(smooth_try),
                                                "tri_to_quad": bool(tri_to_quad_try),
                                                "quad_layers_count": int(len(quad_layers_try)),
                                                "fixed_edges_count": int(len(fixed_edges_try)),
                                                "fixed_edge_vertices": int(sum(len(line) for line in fixed_edges_try)),
                                                "status": "quality-fail",
                                                "n_vertices": int(cand_vx.size),
                                                "n_triangles": int(cand_tris.shape[0]) if cand_tris.ndim == 2 else 0,
                                                "n_quads": int(cand_quads.shape[0]) if cand_quads.ndim == 2 else 0,
                                                "quality": {
                                                    "min_angle_deg": float(stats["min_angle_deg"]),
                                                    "max_aspect_ratio": float(stats["max_aspect_ratio"]),
                                                    "min_area": float(stats["min_area"]),
                                                    "bbox_area": float(stats["bbox_area"]),
                                                },
                                                "focus_points": [[float(x), float(y)] for x, y in focus_points],
                                                "ext_vertices_before": int(len(base_args["ext_verts"])),
                                                "ext_vertices_after": int(len(ext_variant)),
                                            }
                                        )
                                        errors.append(
                                            "quality-fail "
                                            f"{fixed_label}/{label}/{variant_tag} (size={target_try:.4g}, smooth={smooth_try}): "
                                            f"min_angle={stats['min_angle_deg']:.2f}, "
                                            f"max_aspect={stats['max_aspect_ratio']:.2f}, "
                                            f"min_area={stats['min_area']:.3e}"
                                        )

                                    if result is not None:
                                        break

                                errors.append(
                                    f"{fixed_label}/{label} (size={target_try:.4g}, smooth={smooth_try}): {exc_txt}"
                                )
                                if result is not None:
                                    break
                                continue

                            cand_vx = np.asarray(candidate["verts_x"], dtype=np.float64)
                            cand_vy = np.asarray(candidate["verts_y"], dtype=np.float64)
                            cand_tris = np.asarray(candidate["triangles"], dtype=np.int32)
                            cand_quads = np.asarray(candidate["quads"], dtype=np.int32)
                            stats = _mesh_quality_stats(cand_vx, cand_vy, cand_tris, cand_quads)

                            if _quality_passes(stats, quality_cfg):
                                debug_attempts.append(
                                    {
                                        "fixed_variant": str(fixed_label),
                                        "attempt_label": str(label),
                                        "target_size": float(target_try),
                                        "smooth": int(smooth_try),
                                        "tri_to_quad": bool(tri_to_quad_try),
                                        "quad_layers_count": int(len(quad_layers_try)),
                                        "fixed_edges_count": int(len(fixed_edges_try)),
                                        "fixed_edge_vertices": int(sum(len(line) for line in fixed_edges_try)),
                                        "status": "quality-pass",
                                        "n_vertices": int(cand_vx.size),
                                        "n_triangles": int(cand_tris.shape[0]) if cand_tris.ndim == 2 else 0,
                                        "n_quads": int(cand_quads.shape[0]) if cand_quads.ndim == 2 else 0,
                                        "quality": {
                                            "min_angle_deg": float(stats["min_angle_deg"]),
                                            "max_aspect_ratio": float(stats["max_aspect_ratio"]),
                                            "min_area": float(stats["min_area"]),
                                            "bbox_area": float(stats["bbox_area"]),
                                        },
                                    }
                                )
                                result = candidate
                                used_label = f"{fixed_label}/{label} (size={target_try:.4g}, smooth={smooth_try})"
                                used_quality = stats
                                break

                            score = _quality_score(stats, quality_cfg)
                            if score > best_nonpassing_score:
                                best_nonpassing_score = score
                                best_nonpassing = (candidate, f"{fixed_label}/{label}", target_try, smooth_try, stats)
                            debug_attempts.append(
                                {
                                    "fixed_variant": str(fixed_label),
                                    "attempt_label": str(label),
                                    "target_size": float(target_try),
                                    "smooth": int(smooth_try),
                                    "tri_to_quad": bool(tri_to_quad_try),
                                    "quad_layers_count": int(len(quad_layers_try)),
                                    "fixed_edges_count": int(len(fixed_edges_try)),
                                    "fixed_edge_vertices": int(sum(len(line) for line in fixed_edges_try)),
                                    "status": "quality-fail",
                                    "n_vertices": int(cand_vx.size),
                                    "n_triangles": int(cand_tris.shape[0]) if cand_tris.ndim == 2 else 0,
                                    "n_quads": int(cand_quads.shape[0]) if cand_quads.ndim == 2 else 0,
                                    "quality": {
                                        "min_angle_deg": float(stats["min_angle_deg"]),
                                        "max_aspect_ratio": float(stats["max_aspect_ratio"]),
                                        "min_area": float(stats["min_area"]),
                                        "bbox_area": float(stats["bbox_area"]),
                                    },
                                }
                            )
                            errors.append(
                                "quality-fail "
                                f"{fixed_label}/{label} (size={target_try:.4g}, smooth={smooth_try}): "
                                f"min_angle={stats['min_angle_deg']:.2f}, "
                                f"max_aspect={stats['max_aspect_ratio']:.2f}, "
                                f"min_area={stats['min_area']:.3e}"
                            )
                        if result is not None:
                            break
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
                ext_vertex_count = len(base_args["ext_verts"])
                ext_vertex_raw_count = int(ext_verts_raw_count)
                ext_vertex_post_sanitize_count = int(ext_verts_post_sanitize_count)
                ext_vertex_post_stitch_count = int(ext_verts_post_stitch_count)
                ext_vertex_post_resample_count = int(ext_verts_post_resample_count)
                ext_resample_applied = int(resample_applied)
                ext_resample_max_seg = float(resample_max_seg)
                hole_count = len(base_args["int_boundaries"])
                hole_vertices = sum(len(hole) for hole in base_args["int_boundaries"])
                constraint_count = len(base_args["constraint_verts"])
                constraint_vertices = sum(len(cverts) for cverts in base_args["constraint_verts"])
                fixed_edge_count = len(base_args["fixed_edges"])
                fixed_edge_vertices = sum(len(line) for line in base_args["fixed_edges"])
                quad_layer_count = len(active_quad_layers)
                if debug_dump_dir:
                    dump_path = os.path.join(
                        debug_dump_dir,
                        f"tqmesh_region_{int(region.region_id)}_failure.json",
                    )
                    _write_json_atomic(
                        dump_path,
                        {
                            "region_id": int(region.region_id),
                            "cell_type": str(ctype),
                            "target_size": float(target_size),
                            "boundary_split_max_length": float(boundary_split_max_length),
                            "boundary_split_for_call": float(boundary_split_for_call),
                            "strict_fixed_edge_region": bool(strict_fixed_edge_region),
                            "breakline_vertex_snap_tol": 0.1,
                            "ring_initial": _serialize_xy_points(ring_initial),
                            "ring_after_breakline_preprocess": _serialize_xy_points(ring_after_breakline_preprocess),
                            "fixed_edges_raw": _serialize_xy_lines(fixed_edge_lines_raw),
                            "fixed_edges_after_breakline_preprocess": _serialize_xy_lines(fixed_edge_lines_after_breakline_preprocess),
                            "fixed_edges_after_densify": _serialize_xy_lines(fixed_edge_lines_after_densify),
                            "ext_verts_for_tqmesh": [[float(v[0]), float(v[1])] for v in base_args["ext_verts"]],
                            "int_boundaries_for_tqmesh": base_args["int_boundaries"],
                            "constraint_counts": {
                                "constraints": int(constraint_count),
                                "constraint_vertices": int(constraint_vertices),
                                "holes": int(hole_count),
                                "hole_vertices": int(hole_vertices),
                            },
                            "ext_debug_counts": {
                                "ext_vertices_raw": int(ext_vertex_raw_count),
                                "ext_vertices_post_sanitize": int(ext_vertex_post_sanitize_count),
                                "ext_vertices_post_stitch": int(ext_vertex_post_stitch_count),
                                "ext_vertices_post_resample": int(ext_vertex_post_resample_count),
                                "ext_resample_applied": int(ext_resample_applied),
                                "ext_resample_max_seg": float(ext_resample_max_seg),
                                "ext_vertices_final": int(ext_vertex_count),
                            },
                            "fixed_edge_debug_counts": {
                                "fixed_edges": int(fixed_edge_count),
                                "fixed_edge_vertices": int(fixed_edge_vertices),
                                "active_quad_layers": int(quad_layer_count),
                            },
                            "attempts": debug_attempts,
                            "errors": [str(e) for e in errors],
                            "used_label": str(used_label),
                            "used_quality": None if used_quality is None else {
                                "min_angle_deg": float(used_quality["min_angle_deg"]),
                                "max_aspect_ratio": float(used_quality["max_aspect_ratio"]),
                                "min_area": float(used_quality["min_area"]),
                                "bbox_area": float(used_quality["bbox_area"]),
                            },
                        },
                    )
                raise RuntimeError(
                    "TQMesh failed for region "
                    f"{region.region_id} after fallback attempts. "
                    f"region_debug(cell_type={ctype}, target_size={target_size:.6g}, "
                    f"ext_vertices_raw={ext_vertex_raw_count}, "
                    f"ext_vertices_post_sanitize={ext_vertex_post_sanitize_count}, "
                    f"ext_vertices_post_stitch={ext_vertex_post_stitch_count}, "
                    f"ext_vertices_post_resample={ext_vertex_post_resample_count}, "
                    f"ext_resample_applied={ext_resample_applied}, "
                    f"ext_resample_max_seg={ext_resample_max_seg:.6g}, "
                    f"boundary_split_max_length={boundary_split_max_length:.6g}, "
                    f"ext_vertices={ext_vertex_count}, holes={hole_count}, "
                    f"hole_vertices={hole_vertices}, constraints={constraint_count}, "
                    f"constraint_vertices={constraint_vertices}, "
                    f"fixed_edges={fixed_edge_count}, "
                    f"fixed_edge_vertices={fixed_edge_vertices}, "
                    f"active_quad_layers={quad_layer_count}). "
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
            n_fixed_input = int(result.get("n_fixed_edges_input", len(base_args.get("fixed_edges", []))))
            n_fixed_added = int(result.get("n_fixed_edges_added", 0))
            if debug_dump_dir:
                dump_path = os.path.join(
                    debug_dump_dir,
                    f"tqmesh_region_{int(region.region_id)}_success.json",
                )
                _write_json_atomic(
                    dump_path,
                    {
                        "region_id": int(region.region_id),
                        "cell_type": str(ctype),
                        "target_size": float(target_size),
                        "boundary_split_max_length": float(boundary_split_max_length),
                        "boundary_split_for_call": float(boundary_split_for_call),
                        "strict_fixed_edge_region": bool(strict_fixed_edge_region),
                        "breakline_vertex_snap_tol": 0.1,
                        "ring_initial": _serialize_xy_points(ring_initial),
                        "ring_after_breakline_preprocess": _serialize_xy_points(ring_after_breakline_preprocess),
                        "fixed_edges_raw": _serialize_xy_lines(fixed_edge_lines_raw),
                        "fixed_edges_after_breakline_preprocess": _serialize_xy_lines(fixed_edge_lines_after_breakline_preprocess),
                        "fixed_edges_after_densify": _serialize_xy_lines(fixed_edge_lines_after_densify),
                        "ext_verts_for_tqmesh": [[float(v[0]), float(v[1])] for v in base_args["ext_verts"]],
                        "int_boundaries_for_tqmesh": base_args["int_boundaries"],
                        "attempts": debug_attempts,
                        "used_label": str(used_label),
                        "used_quality": None if used_quality is None else {
                            "min_angle_deg": float(used_quality["min_angle_deg"]),
                            "max_aspect_ratio": float(used_quality["max_aspect_ratio"]),
                            "min_area": float(used_quality["min_area"]),
                            "bbox_area": float(used_quality["bbox_area"]),
                        },
                        "result_counts": {
                            "n_vertices": int(vx.size),
                            "n_triangles": int(tris.shape[0]) if tris.ndim == 2 else 0,
                            "n_quads": int(quads.shape[0]) if quads.ndim == 2 else 0,
                            "n_fixed_edges_input": int(n_fixed_input),
                            "n_fixed_edges_added": int(n_fixed_added),
                        },
                    },
                )
            if n_fixed_input > 0 and n_fixed_added <= 0:
                warnings.warn(
                    f"TQMesh region {region.region_id}: fixed-edge breaklines were provided "
                    f"(count={n_fixed_input}) but none were accepted by the core mesher.",
                    RuntimeWarning,
                )

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
        iface_conformance = self._opt_bool(
            "tqmesh_interface_conformance",
            _env_bool("BACKWATER_TQMESH_INTERFACE_CONFORMANCE", True),
        )
        if iface_conformance:
            snap_tol = _as_float(
                self._options.get("tqmesh_interface_snap_tol"),
                _env_float("BACKWATER_TQMESH_INTERFACE_SNAP_TOL", 1.0),
            )
            out = _enforce_quad_interface_conformance(out, model, snap_tol=snap_tol)
        return _repair_mesh_result(out)


def _hybrid_cpp_available() -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec("hydra_hybridmesh") is not None
    except Exception:
        return False


class HybridCppBackend(MeshingBackend):
    """Custom C++ hybrid backend tailored for topology-layer workflows.

    This backend uses a deterministic region-wise Cartesian sweep in C++ and
    emits quad-like faces for ``cartesian``, ``quadrilateral``, and
    ``channel_generator`` regions while preserving triangular regions.
    """

    name = "hybrid-cpp"

    def __init__(self, options: Optional[Dict[str, object]] = None):
        self._options = dict(options or {})

    def generate(self, model: ConceptualModel) -> MeshResult:
        try:
            import hydra_hybridmesh as _hm
        except Exception as exc:
            # Fallback: load the freshly-built extension from local build/.
            try:
                import importlib.util
                from pathlib import Path

                root = Path(__file__).resolve().parents[2]
                build_dir = root / "build"
                cand = sorted(build_dir.glob("hydra_hybridmesh*.so"))
                if not cand:
                    raise FileNotFoundError("hydra_hybridmesh*.so not found under build/")
                spec = importlib.util.spec_from_file_location("hydra_hybridmesh", str(cand[0]))
                if spec is None or spec.loader is None:
                    raise RuntimeError("could not create module spec for hydra_hybridmesh")
                _hm = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(_hm)
            except Exception as load_exc:
                raise RuntimeError(
                    "hydra_hybridmesh C++ module not found. Rebuild native extensions "
                    "(cmake --build build -j) before using backend='hybrid_cpp'."
                ) from load_exc

        region_rings = [
            [(float(x), float(y)) for (x, y) in region.ring_xy]
            for region in model.regions
        ]
        region_holes = [
            [
                [(float(x), float(y)) for (x, y) in hring]
                for hring in (region.hole_rings or [])
            ]
            for region in model.regions
        ]
        region_target_sizes = [float(max(region.default_size, 1.0e-9)) for region in model.regions]
        region_cell_types = [str(region.default_cell_type) for region in model.regions]
        region_ids = [int(region.region_id) for region in model.regions]

        constraint_rings = [
            [(float(x), float(y)) for (x, y) in cst.ring_xy]
            for cst in model.constraints
            if len(cst.ring_xy) >= 3
        ]
        constraint_target_sizes = [
            float(max(cst.target_size, 1.0e-9))
            for cst in model.constraints
            if len(cst.ring_xy) >= 3
        ]
        constraint_cell_types = [
            str(cst.cell_type)
            for cst in model.constraints
            if len(cst.ring_xy) >= 3
        ]

        arc_region_ids = []
        arc_roles = []
        arc_lines = []
        for arc in model.arcs:
            if not arc.points_xy or len(arc.points_xy) < 2:
                continue
            role = str(arc.arc_role or "").strip().lower()
            if role not in {"centerline", "left_bank", "right_bank", "breakline"}:
                continue
            arc_region_ids.append(int(arc.region_id))
            arc_roles.append(role)
            arc_lines.append([(float(x), float(y)) for (x, y) in arc.points_xy])

        tri_meshing_method = str(
            self._options.get("tri_meshing_method", "frontal_delaunay")
        ).strip().lower()
        transition_width_factor = float(self._options.get("transition_width_factor", 1.25))
        transition_outer_factor = float(self._options.get("transition_outer_factor", 2.5))
        overbank_grading_factor = float(self._options.get("overbank_grading_factor", 4.0))
        constrained_edge_snap_tol = float(
            self._options.get("hybridcpp_constrained_edge_snap_tol", 12.0)
        )
        constrained_edge_max_flips = int(
            self._options.get("hybridcpp_constrained_edge_max_flips", 128)
        )
        region_conformance_band_factor = float(
            self._options.get("hybridcpp_region_conformance_band_factor", 0.55)
        )
        arc_conformance_band_factor = float(
            self._options.get("hybridcpp_arc_conformance_band_factor", 0.45)
        )
        strict_conformance_mode = bool(
            self._options.get("hybridcpp_strict_conformance_mode", False)
        )

        if strict_conformance_mode:
            constrained_edge_snap_tol = max(constrained_edge_snap_tol, 16.0)
            constrained_edge_max_flips = max(constrained_edge_max_flips, 1024)
            region_conformance_band_factor = max(region_conformance_band_factor, 0.90)
            arc_conformance_band_factor = max(arc_conformance_band_factor, 0.90)

        raw = _hm.generate_hybrid_mesh(
            region_rings=region_rings,
            region_holes=region_holes,
            region_target_sizes=region_target_sizes,
            region_cell_types=region_cell_types,
            region_ids=region_ids,
            constraint_rings=constraint_rings,
            constraint_target_sizes=constraint_target_sizes,
            constraint_cell_types=constraint_cell_types,
            arc_region_ids=arc_region_ids,
            arc_roles=arc_roles,
            arc_lines=arc_lines,
            tri_meshing_method=tri_meshing_method,
            transition_width_factor=transition_width_factor,
            transition_outer_factor=transition_outer_factor,
            overbank_grading_factor=overbank_grading_factor,
            constrained_edge_snap_tol=constrained_edge_snap_tol,
            constrained_edge_max_flips=constrained_edge_max_flips,
            region_conformance_band_factor=region_conformance_band_factor,
            arc_conformance_band_factor=arc_conformance_band_factor,
            strict_conformance_mode=strict_conformance_mode,
        )

        node_x = np.asarray(raw["node_x"], dtype=np.float64)
        node_y = np.asarray(raw["node_y"], dtype=np.float64)
        cell_nodes = np.asarray(raw["cell_nodes"], dtype=np.int32)
        cell_face_offsets = np.asarray(raw["cell_face_offsets"], dtype=np.int32)
        cell_face_nodes = np.asarray(raw["cell_face_nodes"], dtype=np.int32)
        cell_type = np.asarray(raw["cell_type"], dtype=object)
        region_id = np.asarray(raw["region_id"], dtype=np.int32)
        target_size = np.asarray(raw["target_size"], dtype=np.float64)

        out = MeshResult(
            node_x=node_x,
            node_y=node_y,
            node_z=np.zeros_like(node_x),
            cell_nodes=cell_nodes,
            cell_face_offsets=cell_face_offsets,
            cell_face_nodes=cell_face_nodes,
            cell_type=cell_type,
            region_id=region_id,
            target_size=target_size,
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
    opts = dict(options or {})
    # Always mesh in a local coordinate frame, then restore original CRS-space
    # coordinates on the result for downstream IO and visualization.
    work_model = copy.deepcopy(model)
    x_shift, y_shift = _normalize_conceptual_model_to_local_origin(work_model)

    if backend == "gmsh":
        if not _gmsh_available():
            raise RuntimeError(
                "gmsh Python package is not installed.  "
                "Run: pip install gmsh   (or select the 'Structured' backend)."
            )
        mesh = GmshBackend(options=opts).generate(work_model)
        mesh = _apply_optional_post_optimization(mesh, work_model, opts, backend_name="gmsh")
        return _restore_mesh_coordinates(mesh, x_shift, y_shift)
    if backend == "structured":
        mesh = StructuredFaceCentricBackend().generate(work_model)
        mesh = _apply_optional_post_optimization(mesh, work_model, opts, backend_name="structured")
        return _restore_mesh_coordinates(mesh, x_shift, y_shift)
    if backend == "tqmesh":
        mesh = TQMeshBackend(options=opts).generate(work_model)
        mesh = _apply_optional_post_optimization(mesh, work_model, opts, backend_name="tqmesh")
        return _restore_mesh_coordinates(mesh, x_shift, y_shift)
    if backend in {"hybrid_cpp", "mfem_opt"}:
        # TEMPORARY: disabled per user request; keep implementation in-tree for later re-enable.
        raise ValueError(
            f"Meshing backend {backend!r} is temporarily disabled. "
            "Choose 'gmsh', 'structured', or 'tqmesh'."
        )
    raise ValueError(f"Unknown meshing backend: {backend!r}. Choose 'gmsh', 'structured', or 'tqmesh'.")


def _as_bool_opt(value: object, default: bool = False) -> bool:
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


def _normalize_post_opt_backend(options: Dict[str, object]) -> str:
    # TEMPORARY: MFEM post-optimization is disabled in backend execution paths.
    # backend = str(options.get("post_opt_backend") or "").strip().lower()
    # enabled = _as_bool_opt(options.get("mfem_post_opt_enable"), False)
    # if not backend:
    #     backend = "mfem_tmop" if enabled else "none"
    # if backend in {"", "none", "off", "disabled"}:
    #     return "none"
    # if backend in {"mfem", "mfem_tmop", "tmop"}:
    #     return "mfem_tmop"
    # return backend
    return "none"


def _collect_arc_constraints(model: ConceptualModel) -> Tuple[List[int], List[str], List[List[Tuple[float, float]]]]:
    arc_region_ids: List[int] = []
    arc_roles: List[str] = []
    arc_lines: List[List[Tuple[float, float]]] = []
    for arc in model.arcs:
        if not arc.points_xy or len(arc.points_xy) < 2:
            continue
        role = str(arc.arc_role or "").strip().lower()
        if role not in {"centerline", "left_bank", "right_bank", "breakline"}:
            continue
        arc_region_ids.append(int(arc.region_id))
        arc_roles.append(role)
        arc_lines.append([(float(x), float(y)) for (x, y) in arc.points_xy])
    return arc_region_ids, arc_roles, arc_lines


def _result_from_mapping(raw: Dict[str, Any], fallback: MeshResult) -> MeshResult:
    node_x = np.asarray(raw.get("node_x", fallback.node_x), dtype=np.float64)
    node_y = np.asarray(raw.get("node_y", fallback.node_y), dtype=np.float64)
    node_z = np.asarray(raw.get("node_z", fallback.node_z), dtype=np.float64)
    if node_z.shape[0] != node_x.shape[0]:
        node_z = np.zeros_like(node_x)

    cell_nodes = np.asarray(raw.get("cell_nodes", fallback.cell_nodes), dtype=np.int32)
    cell_face_offsets = np.asarray(raw.get("cell_face_offsets", fallback.cell_face_offsets), dtype=np.int32)
    cell_face_nodes = np.asarray(raw.get("cell_face_nodes", fallback.cell_face_nodes), dtype=np.int32)

    cell_type = raw.get("cell_type", fallback.cell_type)
    cell_type = np.asarray(cell_type, dtype=object)
    region_id = np.asarray(raw.get("region_id", fallback.region_id), dtype=np.int32)
    target_size = np.asarray(raw.get("target_size", fallback.target_size), dtype=np.float64)

    quality_summary = fallback.quality_summary
    if isinstance(raw.get("quality_summary"), dict):
        quality_summary = dict(raw["quality_summary"])

    return MeshResult(
        node_x=node_x,
        node_y=node_y,
        node_z=node_z,
        cell_nodes=cell_nodes,
        cell_face_offsets=cell_face_offsets,
        cell_face_nodes=cell_face_nodes,
        cell_type=cell_type,
        region_id=region_id,
        target_size=target_size,
        quality_summary=quality_summary,
    )


def _apply_mfem_tmop_post_optimization(
    mesh: MeshResult,
    model: ConceptualModel,
    options: Dict[str, object],
    backend_name: str,
) -> MeshResult:
    strict = _as_bool_opt(options.get("mfem_post_opt_strict"), False)
    preset_name = str(options.get("mfem_post_opt_preset") or "balanced_shape_size").strip().lower()
    if preset_name not in available_mfem_presets():
        msg = f"Unknown MFEM preset: {preset_name!r}"
        if strict:
            raise RuntimeError(msg)
        warnings.warn(f"{msg}; using balanced_shape_size.", RuntimeWarning)
        preset_name = "balanced_shape_size"

    real_opt_exc: Optional[Exception] = None
    try:
        optimized = optimize_with_mfem(
            mesh,
            preset_name=preset_name,
            max_iterations=int(options.get("mfem_post_opt_max_iterations", 120) or 120),
            quality_weight=float(options.get("mfem_post_opt_quality_weight", 1.0) or 1.0),
            boundary_fit_weight=float(options.get("mfem_post_opt_boundary_fit_weight", 0.35) or 0.35),
            interface_fit_weight=float(options.get("mfem_post_opt_interface_fit_weight", 0.25) or 0.25),
            min_det_j=float(options.get("mfem_post_opt_min_det_j", 1.0e-9) or 1.0e-9),
            preserve_boundary=_as_bool_opt(options.get("mfem_post_opt_preserve_boundary"), True),
            lock_boundary_nodes=_as_bool_opt(options.get("mfem_post_opt_lock_boundary_nodes"), True),
        )
        summary = dict(optimized.quality_summary or {})
        summary.update(
            {
                "engine": "mfem_mesh_optimizer",
                "preset": preset_name,
                "backend_name": backend_name,
                "quality_weight": float(options.get("mfem_post_opt_quality_weight", 1.0) or 1.0),
                "boundary_fit_weight": float(options.get("mfem_post_opt_boundary_fit_weight", 0.35) or 0.35),
                "interface_fit_weight": float(options.get("mfem_post_opt_interface_fit_weight", 0.25) or 0.25),
                "max_iterations": int(options.get("mfem_post_opt_max_iterations", 120) or 120),
                "min_det_j": float(options.get("mfem_post_opt_min_det_j", 1.0e-9) or 1.0e-9),
                "lock_boundary_nodes": _as_bool_opt(options.get("mfem_post_opt_lock_boundary_nodes"), True),
            }
        )
        optimized.quality_summary = summary
        optimized = _repair_mesh_result(optimized)
        return _require_nonempty_mesh(optimized, f"{backend_name}+mfem_tmop")
    except Exception as exc:
        real_opt_exc = exc
        warnings.warn(
            f"MFEM mesh-optimizer execution failed: {real_opt_exc}. Falling back to module path.",
            RuntimeWarning,
        )

    try:
        import hydra_mfem_meshopt as _mfem_opt
    except Exception as exc:
        # Fallback: try loading a freshly built extension from local build/.
        try:
            import importlib.util
            from pathlib import Path

            root = Path(__file__).resolve().parents[2]
            build_dir = root / "build"
            cand = sorted(build_dir.glob("hydra_mfem_meshopt*.so"))
            if not cand:
                raise FileNotFoundError("hydra_mfem_meshopt*.so not found under build/")
            spec = importlib.util.spec_from_file_location("hydra_mfem_meshopt", str(cand[0]))
            if spec is None or spec.loader is None:
                raise RuntimeError("could not create module spec for hydra_mfem_meshopt")
            _mfem_opt = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_mfem_opt)
        except Exception as load_exc:
            msg = (
                "MFEM TMOP post-optimization requested but hydra_mfem_meshopt is unavailable. "
                "Build with MFEM support or disable MFEM post-opt."
            )
            if strict:
                if real_opt_exc is not None:
                    raise RuntimeError(
                        f"MFEM mesh-optimizer execution failed: {real_opt_exc}; and {msg}"
                    ) from load_exc
                raise RuntimeError(msg) from load_exc
            warnings.warn(f"{msg} Continuing with base {backend_name} mesh.", RuntimeWarning)
            return mesh

    optimize_fn = getattr(_mfem_opt, "optimize_mesh_tmop", None)
    if optimize_fn is None:
        optimize_fn = getattr(_mfem_opt, "optimize_mesh", None)
    if optimize_fn is None:
        msg = "hydra_mfem_meshopt module does not export optimize_mesh_tmop/optimize_mesh."
        if strict:
            raise RuntimeError(msg)
        warnings.warn(f"{msg} Continuing with base {backend_name} mesh.", RuntimeWarning)
        return mesh

    arc_region_ids, arc_roles, arc_lines = _collect_arc_constraints(model)
    call_payload: Dict[str, Any] = {
        "node_x": mesh.node_x,
        "node_y": mesh.node_y,
        "cell_face_offsets": mesh.cell_face_offsets,
        "cell_face_nodes": mesh.cell_face_nodes,
        "cell_nodes": mesh.cell_nodes,
        "cell_type": mesh.cell_type,
        "region_id": mesh.region_id,
        "target_size": mesh.target_size,
        "arc_region_ids": arc_region_ids,
        "arc_roles": arc_roles,
        "arc_lines": arc_lines,
        "quality_weight": float(options.get("mfem_post_opt_quality_weight", 1.0) or 1.0),
        "boundary_fit_weight": float(options.get("mfem_post_opt_boundary_fit_weight", 0.35) or 0.35),
        "interface_fit_weight": float(options.get("mfem_post_opt_interface_fit_weight", 0.25) or 0.25),
        "max_iterations": int(options.get("mfem_post_opt_max_iterations", 120) or 120),
        "min_det_j": float(options.get("mfem_post_opt_min_det_j", 1.0e-9) or 1.0e-9),
        "preserve_boundary": _as_bool_opt(options.get("mfem_post_opt_preserve_boundary"), True),
        "lock_boundary_nodes": _as_bool_opt(options.get("mfem_post_opt_lock_boundary_nodes"), True),
    }

    try:
        raw = optimize_fn(**call_payload)
    except TypeError:
        # Backward-compatible fallback for simpler bindings.
        raw = optimize_fn(
            node_x=mesh.node_x,
            node_y=mesh.node_y,
            cell_face_offsets=mesh.cell_face_offsets,
            cell_face_nodes=mesh.cell_face_nodes,
        )
    except Exception as exc:
        msg = f"MFEM TMOP post-optimization failed: {exc}"
        if strict:
            raise RuntimeError(msg) from exc
        warnings.warn(f"{msg}. Continuing with base {backend_name} mesh.", RuntimeWarning)
        return mesh

    if not isinstance(raw, dict):
        msg = "MFEM TMOP post-optimization returned unexpected payload type."
        if strict:
            raise RuntimeError(msg)
        warnings.warn(f"{msg} Continuing with base {backend_name} mesh.", RuntimeWarning)
        return mesh

    out = _result_from_mapping(raw, mesh)
    summary = dict(out.quality_summary or {})
    summary.setdefault("engine", "hydra_mfem_meshopt_beta_stub")
    summary.setdefault("preset", preset_name)
    out.quality_summary = summary
    out = _repair_mesh_result(out)
    return _require_nonempty_mesh(out, f"{backend_name}+mfem_tmop")


def _apply_optional_post_optimization(
    mesh: MeshResult,
    model: ConceptualModel,
    options: Dict[str, object],
    backend_name: str,
) -> MeshResult:
    base = _require_nonempty_mesh(_repair_mesh_result(mesh), backend_name)
    post_opt_backend = _normalize_post_opt_backend(options)
    if post_opt_backend == "none":
        return base
    if post_opt_backend == "mfem_tmop":
        return _apply_mfem_tmop_post_optimization(base, model, options, backend_name)
    raise ValueError(
        f"Unknown post optimization backend: {post_opt_backend!r}. "
        "Choose 'none' or 'mfem_tmop'."
    )
