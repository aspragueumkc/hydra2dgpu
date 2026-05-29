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
import hashlib
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


_HYDRA_MESHING_NATIVE_MODULE = None
_HYDRA_MESHING_NATIVE_LOAD_ATTEMPTED = False


def _gmsh_cpp_prebuild_enabled() -> bool:
    return _env_bool("BACKWATER_GMSH_CPP_PREBUILD", True)


def _load_hydra_meshing_native():
    global _HYDRA_MESHING_NATIVE_MODULE, _HYDRA_MESHING_NATIVE_LOAD_ATTEMPTED
    if not _gmsh_cpp_prebuild_enabled():
        return None
    if _HYDRA_MESHING_NATIVE_LOAD_ATTEMPTED:
        return _HYDRA_MESHING_NATIVE_MODULE

    _HYDRA_MESHING_NATIVE_LOAD_ATTEMPTED = True
    try:
        import hydra_meshing_native as _mn
        _HYDRA_MESHING_NATIVE_MODULE = _mn
        return _HYDRA_MESHING_NATIVE_MODULE
    except Exception:
        pass

    try:
        import importlib.util
        from pathlib import Path

        root = Path(__file__).resolve().parents[2]
        build_dir = root / "build"
        cand = sorted(build_dir.glob("hydra_meshing_native*.so"))
        if not cand:
            return None
        spec = importlib.util.spec_from_file_location("hydra_meshing_native", str(cand[0]))
        if spec is None or spec.loader is None:
            return None
        _mn = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_mn)
        _HYDRA_MESHING_NATIVE_MODULE = _mn
        return _HYDRA_MESHING_NATIVE_MODULE
    except Exception:
        _HYDRA_MESHING_NATIVE_MODULE = None
        return None


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

    def _is_hard_breakline_arc(arc: ConceptualArc) -> bool:
        mode = str(getattr(arc, "arc_mode_override", "") or "").strip().lower()
        if mode == "hard_embed":
            return True
        if mode in {"soft_size_hint", "disabled"}:
            return False

        role = str(getattr(arc, "arc_role", "") or "").strip().lower()
        if role:
            return role == "breakline"

        # Legacy compatibility for older projects that used untyped topo arcs
        # as breaklines; disabled by default to avoid accidental over-constraint.
        return _env_bool("BACKWATER_TQMESH_UNTYPED_ARCS_ARE_BREAKLINES", False)

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

        if not _is_hard_breakline_arc(arc):
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
    rx = [float(p[0]) for p in pts]
    ry = [float(p[1]) for p in pts]
    ring_bbox = (float(min(rx)), float(min(ry)), float(max(rx)), float(max(ry)))
    n = len(pts)
    inserts: Dict[int, List[Tuple[float, Tuple[float, float]]]] = {i: [] for i in range(n)}

    for fp in focus_points:
        p = (float(fp[0]), float(fp[1]))
        if (
            p[0] < ring_bbox[0] - dmax
            or p[0] > ring_bbox[2] + dmax
            or p[1] < ring_bbox[1] - dmax
            or p[1] > ring_bbox[3] + dmax
        ):
            continue
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


def _ring_from_quad_controls(controls: Sequence[QuadEdgeControl]) -> List[Tuple[float, float]]:
    """Assemble a closed ring (without duplicate closure point) from ordered edges."""
    ring: List[Tuple[float, float]] = []
    for edge in controls:
        pts = [(float(x), float(y)) for (x, y) in list(edge.points_xy or [])]
        if len(pts) < 2:
            continue
        if ring:
            prev = ring[-1]
            cur0 = pts[0]
            if np.hypot(prev[0] - cur0[0], prev[1] - cur0[1]) <= 1.0e-6:
                ring.extend(pts[1:])
            else:
                ring.extend(pts)
        else:
            ring.extend(pts)
    if len(ring) >= 2 and np.hypot(ring[0][0] - ring[-1][0], ring[0][1] - ring[-1][1]) <= 1.0e-6:
        ring = ring[:-1]
    return ring


def _densify_polyline_subset(
    points: Sequence[Tuple[float, float]],
    subset_start_frac: float,
    subset_end_frac: float,
    target_spacing: float,
) -> List[Tuple[float, float]]:
    """Insert points only within an arc-length subset of a polyline."""
    pts = [(float(x), float(y)) for (x, y) in list(points or [])]
    if len(pts) < 2:
        return pts

    total_len = _polyline_length(pts)
    if total_len <= 1.0e-12:
        return pts

    s0_frac = max(0.0, min(1.0, float(subset_start_frac)))
    s1_frac = max(0.0, min(1.0, float(subset_end_frac)))
    if s1_frac <= s0_frac + 1.0e-12:
        return pts

    spacing = max(float(target_spacing), 1.0e-9)
    s0 = s0_frac * total_len
    s1 = s1_frac * total_len

    out: List[Tuple[float, float]] = [pts[0]]
    acc = 0.0
    for i in range(1, len(pts)):
        ax, ay = pts[i - 1]
        bx, by = pts[i]
        seg = float(np.hypot(bx - ax, by - ay))
        if seg <= 1.0e-15:
            continue

        seg_s0 = acc
        seg_s1 = acc + seg
        t_vals = [0.0, 1.0]

        over0 = max(seg_s0, s0)
        over1 = min(seg_s1, s1)
        if over1 - over0 > 1.0e-12:
            over_len = over1 - over0
            n_div = max(1, int(np.ceil(over_len / spacing)))
            for j in range(1, n_div):
                sj = over0 + (float(j) / float(n_div)) * over_len
                tj = (sj - seg_s0) / seg
                if 1.0e-12 < tj < 1.0 - 1.0e-12:
                    t_vals.append(float(tj))

        t_vals = sorted(set(float(round(t, 12)) for t in t_vals))
        for t in t_vals[1:]:
            x = ax + t * (bx - ax)
            y = ay + t * (by - ay)
            if np.hypot(x - out[-1][0], y - out[-1][1]) <= 1.0e-12:
                continue
            out.append((float(x), float(y)))
        acc += seg

    if np.hypot(out[-1][0] - pts[-1][0], out[-1][1] - pts[-1][1]) > 1.0e-12:
        out.append(pts[-1])
    return out


def _sample_closed_polyline(points: Sequence[Tuple[float, float]], step: float) -> List[Tuple[float, float]]:
    pts = [(float(x), float(y)) for (x, y) in list(points or [])]
    if len(pts) < 3:
        return pts
    if np.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) <= 1.0e-12:
        pts = pts[:-1]
    if len(pts) < 3:
        return pts

    h = max(float(step), 1.0e-9)
    out: List[Tuple[float, float]] = [pts[0]]
    n = len(pts)
    for i in range(n):
        ax, ay = pts[i]
        bx, by = pts[(i + 1) % n]
        seg = float(np.hypot(bx - ax, by - ay))
        ndiv = max(1, int(np.ceil(seg / h)))
        for j in range(1, ndiv + 1):
            t = float(j) / float(ndiv)
            x = ax + t * (bx - ax)
            y = ay + t * (by - ay)
            if np.hypot(x - out[-1][0], y - out[-1][1]) <= 1.0e-12:
                continue
            out.append((float(x), float(y)))

    if len(out) >= 2 and np.hypot(out[0][0] - out[-1][0], out[0][1] - out[-1][1]) <= 1.0e-12:
        out.pop()
    return out


def _longest_cyclic_true_run(mask: Sequence[bool]) -> Optional[Tuple[int, int, int]]:
    flags = [bool(v) for v in list(mask or [])]
    n = len(flags)
    if n == 0 or not any(flags):
        return None

    doubled = flags + flags
    best_len = 0
    best_end = -1
    cur = 0
    for i, v in enumerate(doubled):
        if v:
            cur = min(n, cur + 1)
            if cur > best_len:
                best_len = cur
                best_end = i
        else:
            cur = 0

    if best_len <= 0 or best_end < 0:
        return None
    start = (best_end - best_len + 1) % n
    end = best_end % n
    return int(start), int(end), int(best_len)


def _downsample_polyline_samples(
    points: Sequence[Tuple[float, float]],
    max_points: int,
) -> List[Tuple[float, float]]:
    pts = [(float(x), float(y)) for (x, y) in list(points or [])]
    max_pts = max(4, int(max_points))
    if len(pts) <= max_pts:
        return pts
    stride = max(1, int(np.ceil(float(len(pts)) / float(max_pts))))
    out = [pts[i] for i in range(0, len(pts), stride)]
    if out[-1] != pts[-1]:
        out.append(pts[-1])
    return out


def _interface_overlap_metrics(
    ring_a: Sequence[Tuple[float, float]],
    ring_b: Sequence[Tuple[float, float]],
    sample_step: float,
    near_tol: float,
) -> Dict[str, float]:
    native = _load_hydra_meshing_native()
    if native is not None and hasattr(native, "interface_overlap_metrics_closed"):
        try:
            out_native = native.interface_overlap_metrics_closed(
                ring_a,
                ring_b,
                float(sample_step),
                float(near_tol),
                1800,
            )
            return {
                "overlap_ab": float(out_native.get("overlap_ab", 0.0)),
                "overlap_ba": float(out_native.get("overlap_ba", 0.0)),
                "endpoint_delta_ab_max": float(out_native.get("endpoint_delta_ab_max", float("inf"))),
                "endpoint_delta_ba_max": float(out_native.get("endpoint_delta_ba_max", float("inf"))),
                "endpoint_delta_ab_mean": float(out_native.get("endpoint_delta_ab_mean", float("inf"))),
                "endpoint_delta_ba_mean": float(out_native.get("endpoint_delta_ba_mean", float("inf"))),
            }
        except Exception:
            pass

    pa = _sample_closed_polyline(ring_a, step=sample_step)
    pb = _sample_closed_polyline(ring_b, step=sample_step)
    # This preflight is diagnostic-only. Cap sample cardinality to avoid
    # quadratic distance sweeps on very long/split boundaries.
    pa = _downsample_polyline_samples(pa, max_points=1800)
    pb = _downsample_polyline_samples(pb, max_points=1800)
    if len(pa) < 2 or len(pb) < 2:
        return {
            "overlap_ab": 0.0,
            "overlap_ba": 0.0,
            "endpoint_delta_ab_max": float("inf"),
            "endpoint_delta_ba_max": float("inf"),
            "endpoint_delta_ab_mean": float("inf"),
            "endpoint_delta_ba_mean": float("inf"),
        }

    pb_open = list(pb) + [pb[0]]
    pa_open = list(pa) + [pa[0]]

    d_ab = [float(_polyline_distance_and_s(pb_open, float(x), float(y))[0]) for (x, y) in pa]
    d_ba = [float(_polyline_distance_and_s(pa_open, float(x), float(y))[0]) for (x, y) in pb]
    near_ab = [bool(d <= near_tol) for d in d_ab]
    near_ba = [bool(d <= near_tol) for d in d_ba]

    overlap_ab = float(sum(1 for v in near_ab if v)) / float(max(1, len(near_ab)))
    overlap_ba = float(sum(1 for v in near_ba if v)) / float(max(1, len(near_ba)))

    def _endpoint_deltas(samples: Sequence[Tuple[float, float]], near_mask: Sequence[bool], ref_open: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
        run = _longest_cyclic_true_run(near_mask)
        if run is None:
            return float("inf"), float("inf")
        i0, i1, _ = run
        p0 = samples[int(i0)]
        p1 = samples[int(i1)]
        d0 = float(_polyline_distance_and_s(ref_open, float(p0[0]), float(p0[1]))[0])
        d1 = float(_polyline_distance_and_s(ref_open, float(p1[0]), float(p1[1]))[0])
        return float(max(d0, d1)), float(0.5 * (d0 + d1))

    ep_ab_max, ep_ab_mean = _endpoint_deltas(pa, near_ab, pb_open)
    ep_ba_max, ep_ba_mean = _endpoint_deltas(pb, near_ba, pa_open)

    return {
        "overlap_ab": float(overlap_ab),
        "overlap_ba": float(overlap_ba),
        "endpoint_delta_ab_max": float(ep_ab_max),
        "endpoint_delta_ba_max": float(ep_ba_max),
        "endpoint_delta_ab_mean": float(ep_ab_mean),
        "endpoint_delta_ba_mean": float(ep_ba_mean),
    }


def _gmsh_interface_coincidence_report(
    model: ConceptualModel,
    region_quad_setups: Optional[Dict[int, Tuple[List[Tuple[float, float]], List[QuadEdgeControl]]]] = None,
) -> List[Dict[str, object]]:
    region_quad_setups = region_quad_setups or {}

    boundaries: List[
        Tuple[
            int,
            str,
            float,
            List[Tuple[float, float]],
            Tuple[float, float, float, float],
        ]
    ] = []
    for region in model.regions:
        ctype = str(region.default_cell_type).strip().lower()
        if ctype == "empty":
            continue
        rid = int(region.region_id)
        setup = region_quad_setups.get(rid)
        ring_src = list(setup[0]) if setup is not None and len(setup[0]) >= 3 else list(region.ring_xy)
        if ring_src and np.hypot(float(ring_src[0][0]) - float(ring_src[-1][0]), float(ring_src[0][1]) - float(ring_src[-1][1])) <= 1.0e-12:
            ring_src = ring_src[:-1]
        ring = [(float(x), float(y)) for (x, y) in ring_src]
        if len(ring) < 3:
            continue
        size_ref = max(float(region.default_size), 1.0e-9)
        xmin, ymin, xmax, ymax = _bbox_from_ring(ring)
        boundaries.append((rid, ctype, size_ref, ring, (float(xmin), float(ymin), float(xmax), float(ymax))))

    report: List[Dict[str, object]] = []
    for i in range(len(boundaries)):
        rid_a, ctype_a, size_a, ring_a, bbox_a = boundaries[i]
        for j in range(i + 1, len(boundaries)):
            rid_b, ctype_b, size_b, ring_b, bbox_b = boundaries[j]
            size_ref = max(min(float(size_a), float(size_b)), 1.0e-9)
            near_tol = max(1.0e-6, min(0.5, 0.05 * size_ref))
            sample_step = max(near_tol, 0.25 * size_ref)

            # Cheap spatial reject before expensive overlap metrics.
            if (
                float(bbox_a[2]) < float(bbox_b[0]) - float(near_tol)
                or float(bbox_b[2]) < float(bbox_a[0]) - float(near_tol)
                or float(bbox_a[3]) < float(bbox_b[1]) - float(near_tol)
                or float(bbox_b[3]) < float(bbox_a[1]) - float(near_tol)
            ):
                continue

            m = _interface_overlap_metrics(
                ring_a=ring_a,
                ring_b=ring_b,
                sample_step=sample_step,
                near_tol=near_tol,
            )
            overlap_ab = float(m["overlap_ab"])
            overlap_ba = float(m["overlap_ba"])
            if max(overlap_ab, overlap_ba) < 0.05:
                continue

            endpoint_delta_max = max(float(m["endpoint_delta_ab_max"]), float(m["endpoint_delta_ba_max"]))
            endpoint_delta_mean = max(float(m["endpoint_delta_ab_mean"]), float(m["endpoint_delta_ba_mean"]))
            overlap_delta = abs(float(overlap_ab) - float(overlap_ba))

            report.append(
                {
                    "region_a": int(rid_a),
                    "region_b": int(rid_b),
                    "cell_type_a": str(ctype_a),
                    "cell_type_b": str(ctype_b),
                    "near_tol": float(near_tol),
                    "sample_step": float(sample_step),
                    "overlap_ab": float(overlap_ab),
                    "overlap_ba": float(overlap_ba),
                    "overlap_delta": float(overlap_delta),
                    "endpoint_delta_max": float(endpoint_delta_max),
                    "endpoint_delta_mean": float(endpoint_delta_mean),
                }
            )

    report.sort(
        key=lambda r: (
            -max(float(r.get("overlap_ab", 0.0)), float(r.get("overlap_ba", 0.0))),
            float(r.get("endpoint_delta_max", float("inf"))),
            int(r.get("region_a", 0)),
            int(r.get("region_b", 0)),
        )
    )
    return report


def _sample_open_polyline(points: Sequence[Tuple[float, float]], step: float) -> List[Tuple[float, float]]:
    pts = [(float(x), float(y)) for (x, y) in list(points or [])]
    if len(pts) < 2:
        return pts
    h = max(float(step), 1.0e-9)
    out: List[Tuple[float, float]] = [pts[0]]
    for i in range(1, len(pts)):
        ax, ay = pts[i - 1]
        bx, by = pts[i]
        seg = float(np.hypot(bx - ax, by - ay))
        ndiv = max(1, int(np.ceil(seg / h)))
        for j in range(1, ndiv + 1):
            t = float(j) / float(ndiv)
            x = ax + t * (bx - ax)
            y = ay + t * (by - ay)
            if np.hypot(x - out[-1][0], y - out[-1][1]) <= 1.0e-12:
                continue
            out.append((float(x), float(y)))
    return out


def _polyline_overlap_fractions_open(
    poly_a: Sequence[Tuple[float, float]],
    poly_b: Sequence[Tuple[float, float]],
    sample_step: float,
    near_tol: float,
) -> Tuple[float, float]:
    native = _load_hydra_meshing_native()
    if native is not None and hasattr(native, "polyline_overlap_fractions_open"):
        try:
            ov_ab, ov_ba = native.polyline_overlap_fractions_open(
                poly_a,
                poly_b,
                float(sample_step),
                float(near_tol),
                1200,
            )
            return float(ov_ab), float(ov_ba)
        except Exception:
            pass

    a_pts = [(float(x), float(y)) for (x, y) in list(poly_a or [])]
    b_pts = [(float(x), float(y)) for (x, y) in list(poly_b or [])]
    if len(a_pts) < 2 or len(b_pts) < 2:
        return 0.0, 0.0

    ax = [p[0] for p in a_pts]
    ay = [p[1] for p in a_pts]
    bx = [p[0] for p in b_pts]
    by = [p[1] for p in b_pts]
    tol = max(float(near_tol), 0.0)
    if (
        max(ax) < min(bx) - tol
        or max(bx) < min(ax) - tol
        or max(ay) < min(by) - tol
        or max(by) < min(ay) - tol
    ):
        return 0.0, 0.0

    pa = _sample_open_polyline(a_pts, step=sample_step)
    pb = _sample_open_polyline(b_pts, step=sample_step)
    if len(pa) < 2 or len(pb) < 2:
        return 0.0, 0.0

    d_ab = [float(_polyline_distance_and_s(pb, float(x), float(y))[0]) for (x, y) in pa]
    d_ba = [float(_polyline_distance_and_s(pa, float(x), float(y))[0]) for (x, y) in pb]
    overlap_ab = float(sum(1 for d in d_ab if d <= near_tol)) / float(max(1, len(d_ab)))
    overlap_ba = float(sum(1 for d in d_ba if d <= near_tol)) / float(max(1, len(d_ba)))
    return float(overlap_ab), float(overlap_ba)


def _split_polyline_at_focus_points(
    points: Sequence[Tuple[float, float]],
    focus_points: Sequence[Tuple[float, float]],
    dmax: float,
) -> List[Tuple[float, float]]:
    pts = [(float(x), float(y)) for (x, y) in list(points or [])]
    if len(pts) < 2 or not focus_points:
        return pts

    n_seg = len(pts) - 1
    inserts: Dict[int, List[Tuple[float, Tuple[float, float]]]] = {i: [] for i in range(n_seg)}
    dlim = max(float(dmax), 1.0e-12)

    for fp in focus_points:
        px, py = float(fp[0]), float(fp[1])
        best_i = -1
        best_u = 0.0
        best_q = (0.0, 0.0)
        best_d = float("inf")
        for i in range(n_seg):
            q, u, d = _point_to_segment_projection((px, py), pts[i], pts[i + 1])
            if d < best_d:
                best_d = float(d)
                best_i = i
                best_u = float(u)
                best_q = (float(q[0]), float(q[1]))
        if best_i < 0 or best_d > dlim:
            continue
        if best_u <= 1.0e-10 or best_u >= 1.0 - 1.0e-10:
            continue
        inserts[best_i].append((best_u, best_q))

    out: List[Tuple[float, float]] = [pts[0]]
    for i in range(n_seg):
        items = sorted(inserts.get(i, []), key=lambda it: it[0])
        prev_u = -1.0
        for u, q in items:
            if abs(float(u) - prev_u) <= 1.0e-8:
                continue
            prev_u = float(u)
            if np.hypot(float(q[0]) - out[-1][0], float(q[1]) - out[-1][1]) <= 1.0e-12:
                continue
            out.append((float(q[0]), float(q[1])))
        p1 = pts[i + 1]
        if np.hypot(float(p1[0]) - out[-1][0], float(p1[1]) - out[-1][1]) <= 1.0e-12:
            continue
        out.append((float(p1[0]), float(p1[1])))

    return out


def _junction_points_on_interface(
    interface_chain: Sequence[Tuple[float, float]],
    owner_region_ids: Sequence[int],
    all_region_rings: Dict[int, Sequence[Tuple[float, float]]],
    snap_tol: float,
    endpoint_margin_frac: float = 0.02,
) -> List[Tuple[float, float]]:
    chain = [(float(x), float(y)) for (x, y) in list(interface_chain or [])]
    if len(chain) < 2:
        return []

    owner_set = {int(v) for v in owner_region_ids}
    ttol = max(float(snap_tol), 1.0e-9)

    c_len = _polyline_length(chain)
    if c_len <= 1.0e-12:
        return []

    margin_frac = max(0.0, min(0.49, float(endpoint_margin_frac)))
    s_start = margin_frac * c_len
    s_end = (1.0 - margin_frac) * c_len
    found: List[Tuple[float, float]] = []

    def _consider_point(px: float, py: float) -> None:
        d, s = _polyline_distance_and_s(chain, float(px), float(py))
        if (not np.isfinite(float(d))) or float(d) > ttol:
            return
        if float(s) <= s_start or float(s) >= s_end:
            return
        q = _interp_polyline_fraction(chain, float(s) / c_len)
        found.append((float(q[0]), float(q[1])))

    for rid, ring_in in all_region_rings.items():
        if int(rid) in owner_set:
            continue
        ring = [(float(x), float(y)) for (x, y) in list(ring_in or [])]
        if ring and np.hypot(ring[0][0] - ring[-1][0], ring[0][1] - ring[-1][1]) <= 1.0e-12:
            ring = ring[:-1]
        if len(ring) < 2:
            continue

        for px, py in ring:
            _consider_point(float(px), float(py))

        for i in range(1, len(ring)):
            p0 = (float(ring[i - 1][0]), float(ring[i - 1][1]))
            p1 = (float(ring[i][0]), float(ring[i][1]))
            for j in range(1, len(chain)):
                q0 = (float(chain[j - 1][0]), float(chain[j - 1][1]))
                q1 = (float(chain[j][0]), float(chain[j][1]))
                inter = _segment_intersection_point(p0, p1, q0, q1, eps=ttol)
                if inter is None:
                    continue
                ip, _t, _u = inter
                _consider_point(float(ip[0]), float(ip[1]))

    out: List[Tuple[float, float]] = []
    seen = set()
    for p in found:
        key = (round(float(p[0]), 6), round(float(p[1]), 6))
        if key in seen:
            continue
        seen.add(key)
        out.append((float(p[0]), float(p[1])))
    return out


def _harmonize_transfinite_shared_quad_interfaces(
    region_quad_setups: Dict[int, Tuple[List[Tuple[float, float]], List[QuadEdgeControl]]],
    region_cell_types: Dict[int, str],
    gmsh_quad_full_region_flow_align: bool,
    all_region_rings: Optional[Dict[int, Sequence[Tuple[float, float]]]] = None,
    opposite_subset_start_frac: float = 0.30,
    opposite_subset_end_frac: float = 0.70,
    opposite_subset_density_scale: float = 0.50,
    subset_containment_enable: bool = True,
    subset_containment_high_overlap: float = 0.95,
    subset_containment_min_overlap: float = 0.02,
    subset_containment_max_length_ratio: float = 0.35,
    debug_capture: Optional[Dict[str, object]] = None,
) -> Tuple[Dict[Tuple[int, int], int], Dict[str, int]]:
    """Share densest interface chains across transfinite neighbors.

    Returns
    -------
    edge_min_nodes: map of ``(region_id, edge_id) -> min transfinite nodes``.
    stats: integer counters used for runtime diagnostics.
    """

    def _is_transfinite_region(region_id: int) -> bool:
        ctype = str(region_cell_types.get(int(region_id), "")).strip().lower()
        if ctype == "cartesian":
            return True
        if gmsh_quad_full_region_flow_align and ctype in {"quadrilateral", "channel_generator"}:
            return True
        return False

    subset_enable = bool(subset_containment_enable)
    subset_high_overlap = max(0.0, min(1.0, float(subset_containment_high_overlap)))
    subset_min_overlap = max(0.0, min(1.0, float(subset_containment_min_overlap)))
    subset_max_ratio = max(1.0e-6, float(subset_containment_max_length_ratio))
    collect_debug = isinstance(debug_capture, dict)

    def _edge_key(edge: QuadEdgeControl) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        pts = list(edge.points_xy or [])
        if len(pts) < 2:
            return None
        a = (round(float(pts[0][0]), 6), round(float(pts[0][1]), 6))
        b = (round(float(pts[-1][0]), 6), round(float(pts[-1][1]), 6))
        return (a, b) if a <= b else (b, a)

    def _density_nodes(edge: QuadEdgeControl) -> int:
        pts = [(float(x), float(y)) for (x, y) in list(edge.points_xy or [])]
        if len(pts) < 2:
            return 2
        length = max(_polyline_length(pts), 1.0e-9)
        spacing = edge.target_size if (edge.target_size is not None and float(edge.target_size) > 0.0) else (length / max(1.0, float(len(pts) - 1)))
        spacing = max(float(spacing), 1.0e-9)
        est = max(2, int(round(length / spacing)) + 1)
        return max(int(est), int(len(pts)))

    def _oriented_chain_like(edge: QuadEdgeControl, chain: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
        pts = list(edge.points_xy or [])
        cand = [(float(x), float(y)) for (x, y) in chain]
        if len(cand) < 2 or len(pts) < 2:
            return cand
        fwd = float(np.hypot(float(pts[0][0]) - cand[0][0], float(pts[0][1]) - cand[0][1]))
        fwd += float(np.hypot(float(pts[-1][0]) - cand[-1][0], float(pts[-1][1]) - cand[-1][1]))
        rev = float(np.hypot(float(pts[0][0]) - cand[-1][0], float(pts[0][1]) - cand[-1][1]))
        rev += float(np.hypot(float(pts[-1][0]) - cand[0][0], float(pts[-1][1]) - cand[0][1]))
        if rev + 1.0e-12 < fwd:
            cand = list(reversed(cand))
        return cand

    edge_records: List[Tuple[int, int, QuadEdgeControl]] = []
    edge_keys: List[Optional[Tuple[Tuple[float, float], Tuple[float, float]]]] = []
    edge_points_xy: List[List[Tuple[float, float]]] = []
    edge_lengths: List[float] = []
    edge_bboxes: List[Tuple[float, float, float, float]] = []
    edge_size_hints: List[float] = []
    edge_bucket_scales: List[float] = []
    for rid, (_ring, controls) in region_quad_setups.items():
        if not _is_transfinite_region(int(rid)):
            continue
        for edge in list(controls or []):
            edge_records.append((int(rid), int(edge.edge_id), edge))
            edge_keys.append(_edge_key(edge))

            pts = [(float(x), float(y)) for (x, y) in list(edge.points_xy or [])]
            edge_points_xy.append(pts)
            if len(pts) >= 2:
                edge_len = max(_polyline_length(pts), 1.0e-12)
                xs = [float(p[0]) for p in pts]
                ys = [float(p[1]) for p in pts]
                edge_bboxes.append((float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))))
            else:
                edge_len = 0.0
                edge_bboxes.append((0.0, 0.0, 0.0, 0.0))
            edge_lengths.append(float(edge_len))

            ts_hint = float(edge.target_size) if (edge.target_size is not None and float(edge.target_size) > 0.0) else max(float(edge_len), 1.0e-9)
            edge_size_hints.append(float(ts_hint))
            edge_bucket_scales.append(max(float(ts_hint), max(float(edge_len), 1.0e-9)))

    n_edges = len(edge_records)
    if n_edges == 0:
        if isinstance(debug_capture, dict):
            debug_capture.clear()
            debug_capture.update(
                {
                    "n_edges": 0,
                    "candidate_pair_count": 0,
                    "overlap_rule": {
                        "min_overlap_strict": 0.55,
                        "min_overlap_relaxed": 0.35,
                        "max_overlap_relaxed": 0.75,
                        "subset_containment_enable": bool(subset_enable),
                        "subset_containment_high_overlap": float(subset_high_overlap),
                        "subset_containment_min_overlap": float(subset_min_overlap),
                        "subset_containment_max_length_ratio": float(subset_max_ratio),
                    },
                    "pair_debug": [],
                    "region_pair_summary": [],
                    "groups": [],
                }
            )
        return {}, {
            "shared_groups": 0,
            "canonicalized_edges": 0,
            "opposite_subset_requests": 0,
            "junction_points_inserted": 0,
            "subset_containment_requests": 0,
            "singleton_external_junction_edges": 0,
            "candidate_pair_count_prefilter": 0,
            "candidate_pair_count": 0,
            "pair_bbox_reject_count": 0,
        }

    parent = list(range(n_edges))

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def _union(i: int, j: int) -> None:
        ri = _find(i)
        rj = _find(j)
        if ri != rj:
            parent[rj] = ri

    key_first: Dict[Tuple[Tuple[float, float], Tuple[float, float]], int] = {}
    for i, key in enumerate(edge_keys):
        if key is None:
            continue
        j = key_first.get(key)
        if j is None:
            key_first[key] = i
        else:
            _union(i, j)

    pair_debug_records: List[Dict[str, object]] = []
    subset_pair_requests: List[Dict[str, object]] = []
    pair_bbox_reject_count = 0
    candidate_pair_count_eval = 0

    def _edge_endpoints(points: Sequence[Tuple[float, float]]) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
        pts = [(float(x), float(y)) for (x, y) in list(points or [])]
        if len(pts) < 2:
            return None, None
        a = (float(pts[0][0]), float(pts[0][1]))
        b = (float(pts[-1][0]), float(pts[-1][1]))
        return a, b

    candidate_neighbors: List[set] = [set() for _ in range(n_edges)]
    max_pair_near_tol = 0.5
    if n_edges <= 32:
        for i in range(n_edges):
            for j in range(i + 1, n_edges):
                candidate_neighbors[i].add(int(j))
    else:
        valid_scales = [
            float(edge_bucket_scales[i])
            for i in range(n_edges)
            if len(edge_points_xy[i]) >= 2 and np.isfinite(float(edge_bucket_scales[i])) and float(edge_bucket_scales[i]) > 0.0
        ]
        if valid_scales:
            bucket_size = max(2.0 * float(max_pair_near_tol), float(np.median(np.asarray(valid_scales, dtype=np.float64))))
        else:
            bucket_size = 1.0
        bucket_size = max(float(bucket_size), 1.0e-6)

        edge_buckets: Dict[Tuple[int, int], List[int]] = {}
        for idx in range(n_edges):
            if len(edge_points_xy[idx]) < 2:
                continue
            xmin, ymin, xmax, ymax = edge_bboxes[idx]
            ix0 = int(np.floor((float(xmin) - float(max_pair_near_tol)) / float(bucket_size)))
            ix1 = int(np.floor((float(xmax) + float(max_pair_near_tol)) / float(bucket_size)))
            iy0 = int(np.floor((float(ymin) - float(max_pair_near_tol)) / float(bucket_size)))
            iy1 = int(np.floor((float(ymax) + float(max_pair_near_tol)) / float(bucket_size)))

            # Guard against pathological huge boxes exploding bucket inserts.
            if (ix1 - ix0 + 1) * (iy1 - iy0 + 1) > 2048:
                cx = 0.5 * (float(xmin) + float(xmax))
                cy = 0.5 * (float(ymin) + float(ymax))
                ix0 = int(np.floor((cx - float(max_pair_near_tol)) / float(bucket_size)))
                ix1 = int(np.floor((cx + float(max_pair_near_tol)) / float(bucket_size)))
                iy0 = int(np.floor((cy - float(max_pair_near_tol)) / float(bucket_size)))
                iy1 = int(np.floor((cy + float(max_pair_near_tol)) / float(bucket_size)))

            for ix in range(ix0, ix1 + 1):
                for iy in range(iy0, iy1 + 1):
                    edge_buckets.setdefault((int(ix), int(iy)), []).append(int(idx))

        for members in edge_buckets.values():
            uniq = sorted(set(int(v) for v in members))
            for ai in range(len(uniq)):
                i = int(uniq[ai])
                for bj in range(ai + 1, len(uniq)):
                    j = int(uniq[bj])
                    lo = int(min(i, j))
                    hi = int(max(i, j))
                    candidate_neighbors[lo].add(hi)

        # Safety fallback to preserve legacy behavior if the spatial prefilter
        # produced no candidates due to degenerate indexing.
        if not any(len(v) > 0 for v in candidate_neighbors):
            for i in range(n_edges):
                for j in range(i + 1, n_edges):
                    candidate_neighbors[i].add(int(j))

    bucket_prefilter_candidate_pairs = int(sum(len(v) for v in candidate_neighbors))

    for i in range(n_edges):
        rid_i, eid_i, edge_i = edge_records[i]
        edge_i_pts = edge_points_xy[i]
        if len(edge_i_pts) < 2:
            continue
        for j in sorted(candidate_neighbors[i]):
            rid_j, eid_j, edge_j = edge_records[j]
            edge_j_pts = edge_points_xy[j]
            if rid_i == rid_j:
                continue
            if len(edge_j_pts) < 2:
                continue

            ts_i = float(edge_size_hints[i])
            ts_j = float(edge_size_hints[j])
            size_ref = max(min(float(ts_i), float(ts_j)), 1.0e-9)
            near_tol = max(1.0e-6, min(0.5, 0.05 * size_ref))

            bbi = edge_bboxes[i]
            bbj = edge_bboxes[j]
            if (
                float(bbi[2]) < float(bbj[0]) - float(near_tol)
                or float(bbj[2]) < float(bbi[0]) - float(near_tol)
                or float(bbi[3]) < float(bbj[1]) - float(near_tol)
                or float(bbj[3]) < float(bbi[1]) - float(near_tol)
            ):
                pair_bbox_reject_count += 1
                continue

            candidate_pair_count_eval += 1
            sample_step = max(near_tol, 0.25 * size_ref)

            o_ij, o_ji = _polyline_overlap_fractions_open(
                edge_i_pts,
                edge_j_pts,
                sample_step=sample_step,
                near_tol=near_tol,
            )
            edge_len_i = max(float(edge_lengths[i]), 1.0e-12)
            edge_len_j = max(float(edge_lengths[j]), 1.0e-12)
            overlap_min = min(float(o_ij), float(o_ji))
            overlap_max = max(float(o_ij), float(o_ji))
            pass_strict = overlap_min >= 0.55
            pass_relaxed = overlap_max >= 0.75 and overlap_min >= 0.35
            length_ratio = min(float(edge_len_i), float(edge_len_j)) / max(float(edge_len_i), float(edge_len_j))
            pass_subset_containment = bool(
                subset_enable
                and (not pass_strict)
                and (not pass_relaxed)
                and overlap_max >= float(subset_high_overlap)
                and overlap_min >= float(subset_min_overlap)
                and length_ratio <= float(subset_max_ratio)
            )

            pre_grouped = bool(_find(i) == _find(j))
            grouped_by = "key" if pre_grouped and edge_keys[i] is not None and edge_keys[i] == edge_keys[j] else ("prior" if pre_grouped else "none")
            if (not pre_grouped) and (pass_strict or pass_relaxed or pass_subset_containment):
                _union(i, j)
                if pass_strict or pass_relaxed:
                    grouped_by = "overlap"
                else:
                    grouped_by = "overlap_subset_containment"
                    if float(o_ij) >= float(o_ji):
                        subset_pair_requests.append(
                            {
                                "contained_index": int(i),
                                "container_index": int(j),
                                "near_tol": float(near_tol),
                            }
                        )
                    else:
                        subset_pair_requests.append(
                            {
                                "contained_index": int(j),
                                "container_index": int(i),
                                "near_tol": float(near_tol),
                            }
                        )

            post_grouped = bool(_find(i) == _find(j))
            if collect_debug:
                e0a, e0b = _edge_endpoints(edge_i_pts)
                e1a, e1b = _edge_endpoints(edge_j_pts)
                pair_debug_records.append(
                    {
                        "region_i": int(rid_i),
                        "edge_i": int(eid_i),
                        "region_j": int(rid_j),
                        "edge_j": int(eid_j),
                        "edge_i_n_points": int(len(edge_i_pts)),
                        "edge_j_n_points": int(len(edge_j_pts)),
                        "edge_i_len": float(edge_len_i),
                        "edge_j_len": float(edge_len_j),
                        "length_ratio": float(length_ratio),
                        "edge_i_start": e0a,
                        "edge_i_end": e0b,
                        "edge_j_start": e1a,
                        "edge_j_end": e1b,
                        "key_match": bool(edge_keys[i] is not None and edge_keys[i] == edge_keys[j]),
                        "near_tol": float(near_tol),
                        "sample_step": float(sample_step),
                        "overlap_ij": float(o_ij),
                        "overlap_ji": float(o_ji),
                        "overlap_min": float(overlap_min),
                        "overlap_max": float(overlap_max),
                        "pass_strict": bool(pass_strict),
                        "pass_relaxed": bool(pass_relaxed),
                        "pass_subset_containment": bool(pass_subset_containment),
                        "pre_grouped": bool(pre_grouped),
                        "grouped": bool(post_grouped),
                        "grouped_by": str(grouped_by),
                    }
                )

    edge_groups: Dict[int, List[Tuple[int, int, QuadEdgeControl]]] = {}
    edge_groups_indices: Dict[int, List[int]] = {}
    for i, rec in enumerate(edge_records):
        root = _find(i)
        edge_groups.setdefault(int(root), []).append(rec)
        edge_groups_indices.setdefault(int(root), []).append(int(i))

    edge_min_nodes: Dict[Tuple[int, int], int] = {}
    opposite_subset_requests: Dict[Tuple[int, int], float] = {}
    stats = {
        "shared_groups": 0,
        "canonicalized_edges": 0,
        "opposite_subset_requests": 0,
        "junction_points_inserted": 0,
        "subset_containment_requests": 0,
        "singleton_external_junction_edges": 0,
        "candidate_pair_count_prefilter": int(bucket_prefilter_candidate_pairs),
        "candidate_pair_count": int(candidate_pair_count_eval),
        "pair_bbox_reject_count": int(pair_bbox_reject_count),
    }

    subset_container_indices = {
        int(req.get("container_index", -1))
        for req in subset_pair_requests
        if int(req.get("container_index", -1)) >= 0
    }
    subset_contained_indices = {
        int(req.get("contained_index", -1))
        for req in subset_pair_requests
        if int(req.get("contained_index", -1)) >= 0
    }

    region_rings = dict(all_region_rings or {})

    def _split_edge_with_external_junctions(
        rid: int,
        edge: QuadEdgeControl,
        owner_region_ids: Sequence[int],
    ) -> int:
        if not region_rings:
            return 0
        pts = [(float(x), float(y)) for (x, y) in list(edge.points_xy or [])]
        if len(pts) < 2:
            return 0

        if edge.target_size is not None and float(edge.target_size) > 0.0:
            size_ref = max(float(edge.target_size), 1.0e-6)
        else:
            size_ref = max(_polyline_length(pts) / max(1, len(pts) - 1), 1.0e-6)
        # Singleton interfaces can be slightly offset from neighboring rings.
        # Use an adaptive tolerance and filter by overlap to avoid unrelated rings.
        snap_tol = max(1.0e-6, min(8.0, max(0.5, 0.25 * float(size_ref))))

        owner_set = {int(v) for v in owner_region_ids} if owner_region_ids else {int(rid)}

        chain_x = [float(p[0]) for p in pts]
        chain_y = [float(p[1]) for p in pts]
        xmin = min(chain_x) - float(snap_tol)
        xmax = max(chain_x) + float(snap_tol)
        ymin = min(chain_y) - float(snap_tol)
        ymax = max(chain_y) + float(snap_tol)

        sample_step = max(float(snap_tol), 0.25 * float(size_ref))
        candidate_rings: Dict[int, Sequence[Tuple[float, float]]] = {}
        for rrid, ring_src in region_rings.items():
            if int(rrid) in owner_set:
                continue
            ring_pts = [(float(x), float(y)) for (x, y) in list(ring_src or [])]
            if ring_pts and np.hypot(float(ring_pts[0][0]) - float(ring_pts[-1][0]), float(ring_pts[0][1]) - float(ring_pts[-1][1])) <= 1.0e-12:
                ring_pts = ring_pts[:-1]
            if len(ring_pts) < 2:
                continue

            ring_x = [float(p[0]) for p in ring_pts]
            ring_y = [float(p[1]) for p in ring_pts]
            if max(ring_x) < xmin or min(ring_x) > xmax or max(ring_y) < ymin or min(ring_y) > ymax:
                continue

            ring_open = list(ring_pts)
            if np.hypot(
                float(ring_open[0][0]) - float(ring_open[-1][0]),
                float(ring_open[0][1]) - float(ring_open[-1][1]),
            ) > 1.0e-12:
                ring_open.append((float(ring_open[0][0]), float(ring_open[0][1])))

            overlap_edge, overlap_ring = _polyline_overlap_fractions_open(
                pts,
                ring_open,
                sample_step=float(sample_step),
                near_tol=float(snap_tol),
            )
            if max(float(overlap_edge), float(overlap_ring)) < 0.01:
                continue
            candidate_rings[int(rrid)] = ring_pts

        if not candidate_rings:
            return 0

        jpts = _junction_points_on_interface(
            interface_chain=pts,
            owner_region_ids=sorted(owner_set),
            all_region_rings=candidate_rings,
            snap_tol=float(snap_tol),
            endpoint_margin_frac=0.002,
        )
        if not jpts:
            return 0

        split_pts = _split_polyline_at_focus_points(
            points=pts,
            focus_points=jpts,
            dmax=float(snap_tol),
        )
        if len(split_pts) <= len(pts):
            return 0

        edge.points_xy = split_pts
        return int(len(split_pts) - len(pts))

    for _root, members in edge_groups.items():
        member_indices = list(edge_groups_indices.get(int(_root), []))
        if len(member_indices) != len(members):
            member_indices = []
        owner_ids = sorted(set(int(rid) for rid, _eid, _edge in members))
        if len(owner_ids) < 2:
            # Even without a transfinite peer edge, split this transfinite edge
            # at neighboring region-ring junctions so interface vertices are
            # projected consistently across mixed transfinite/non-transfinite
            # adjacencies.
            for rid, _eid, edge in members:
                inserted = _split_edge_with_external_junctions(
                    rid=int(rid),
                    edge=edge,
                    owner_region_ids=[int(rid)],
                )
                if inserted > 0:
                    stats["junction_points_inserted"] += int(inserted)
                    stats["singleton_external_junction_edges"] += 1
            continue

        stats["shared_groups"] += 1
        densest = max(members, key=lambda item: _density_nodes(item[2]))
        dense_chain = [(float(x), float(y)) for (x, y) in list(densest[2].points_xy or [])]
        if len(dense_chain) < 2:
            continue
        old_dense_len = len(dense_chain)
        positive_targets = [float(edge.target_size) for (_rid, _eid, edge) in members if edge.target_size is not None and float(edge.target_size) > 0.0]
        dense_target = min(positive_targets) if positive_targets else None

        if region_rings:
            if dense_target is not None and dense_target > 0.0:
                snap_tol = max(1.0e-6, min(0.5, 0.05 * float(dense_target)))
            else:
                avg_step = max(_polyline_length(dense_chain) / max(1, len(dense_chain) - 1), 1.0e-6)
                snap_tol = max(1.0e-6, min(0.5, 0.10 * float(avg_step)))

            jpts = _junction_points_on_interface(
                interface_chain=dense_chain,
                owner_region_ids=owner_ids,
                all_region_rings=region_rings,
                snap_tol=float(snap_tol),
            )
            if jpts:
                dense_chain = _split_polyline_at_focus_points(
                    points=dense_chain,
                    focus_points=jpts,
                    dmax=float(snap_tol),
                )
                if len(dense_chain) > old_dense_len:
                    stats["junction_points_inserted"] += int(len(dense_chain) - old_dense_len)

        dense_len = max(_polyline_length(dense_chain), 1.0e-9)
        if dense_target is not None and dense_target > 0.0:
            dense_nodes = max(int(len(dense_chain)), int(round(dense_len / float(dense_target))) + 1)
        else:
            dense_nodes = max(int(len(dense_chain)), int(_density_nodes(densest[2])))

        for member_k, (rid, eid, edge) in enumerate(members):
            idx_member = int(member_indices[member_k]) if member_k < len(member_indices) else -1
            subset_related = bool(
                idx_member in subset_container_indices or idx_member in subset_contained_indices
            )
            member_pts = list(edge.points_xy or [])
            if len(member_pts) < 2:
                continue
            ts_member = float(edge.target_size) if (edge.target_size is not None and float(edge.target_size) > 0.0) else max(_polyline_length(member_pts), 1.0e-9)
            ts_dense = float(dense_target) if (dense_target is not None and float(dense_target) > 0.0) else ts_member
            size_ref_member = max(min(float(ts_member), float(ts_dense)), 1.0e-9)
            near_tol_member = max(1.0e-6, min(0.5, 0.05 * float(size_ref_member)))
            sample_step_member = max(float(near_tol_member), 0.25 * float(size_ref_member))
            o_member_dense, o_dense_member = _polyline_overlap_fractions_open(
                member_pts,
                dense_chain,
                sample_step=sample_step_member,
                near_tol=near_tol_member,
            )
            overlap_min_member = min(float(o_member_dense), float(o_dense_member))
            overlap_max_member = max(float(o_member_dense), float(o_dense_member))
            pass_member_strict = overlap_min_member >= 0.55
            pass_member_relaxed = overlap_max_member >= 0.75 and overlap_min_member >= 0.35
            if not (pass_member_strict or pass_member_relaxed):
                # Subset-contained interfaces are handled by targeted container-edge
                # splitting/densification later; avoid replacing full edge geometry.
                continue

            oriented_chain = _oriented_chain_like(edge, dense_chain)
            if len(oriented_chain) >= 2:
                if len(oriented_chain) != len(list(edge.points_xy or [])):
                    stats["canonicalized_edges"] += 1
                edge.points_xy = oriented_chain
            if dense_target is not None and (not subset_related):
                edge.target_size = min(float(dense_target), float(edge.target_size)) if (edge.target_size is not None and float(edge.target_size) > 0.0) else float(dense_target)

            if subset_related:
                # Subset containment should not enforce whole-edge transfinite
                # floors, otherwise a localized junction refinement propagates
                # density across the entire opposite edge pair.
                continue

            edge_min_nodes[(int(rid), int(eid))] = max(int(edge_min_nodes.get((int(rid), int(eid)), 0)), int(dense_nodes))

            opp_id = 1 + ((int(eid) + 1) % 4)
            controls = region_quad_setups.get(int(rid), ([], []))[1]
            opp_edge = next((e for e in controls if int(e.edge_id) == int(opp_id)), None)
            if opp_edge is None:
                continue
            opp_len = _polyline_length(list(opp_edge.points_xy or []))
            if opp_len <= 1.0e-9 or dense_nodes <= 2:
                continue
            desired_spacing = opp_len / max(float(dense_nodes - 1), 1.0)
            subset_spacing = max(1.0e-9, float(opposite_subset_density_scale) * float(desired_spacing))
            prev = opposite_subset_requests.get((int(rid), int(opp_id)))
            opposite_subset_requests[(int(rid), int(opp_id))] = subset_spacing if prev is None else min(float(prev), float(subset_spacing))

    for (rid, opp_id), spacing in opposite_subset_requests.items():
        controls = region_quad_setups.get(int(rid), ([], []))[1]
        opp_edge = next((e for e in controls if int(e.edge_id) == int(opp_id)), None)
        if opp_edge is None:
            continue
        densified = _densify_polyline_subset(
            points=list(opp_edge.points_xy or []),
            subset_start_frac=float(opposite_subset_start_frac),
            subset_end_frac=float(opposite_subset_end_frac),
            target_spacing=float(spacing),
        )
        if len(densified) > len(list(opp_edge.points_xy or [])):
            stats["opposite_subset_requests"] += 1
            opp_edge.points_xy = densified

    for req in subset_pair_requests:
        i_contained = int(req.get("contained_index", -1))
        i_container = int(req.get("container_index", -1))
        if i_contained < 0 or i_container < 0:
            continue
        if i_contained >= len(edge_records) or i_container >= len(edge_records):
            continue
        if _find(i_contained) != _find(i_container):
            continue

        rid_contained, _eid_contained, edge_contained = edge_records[i_contained]
        rid_container, eid_container, edge_container = edge_records[i_container]
        contained_pts = list(edge_contained.points_xy or [])
        container_pts = list(edge_container.points_xy or [])
        if len(contained_pts) < 2 or len(container_pts) < 2:
            continue

        near_tol_req = max(1.0e-6, float(req.get("near_tol", 1.0e-6)))
        focus_pts = [
            (float(p[0]), float(p[1]))
            for p in contained_pts
        ]
        if len(focus_pts) < 2:
            continue
        split_container = _split_polyline_at_focus_points(
            points=container_pts,
            focus_points=focus_pts,
            dmax=max(near_tol_req, 2.0e-6),
        )
        if len(split_container) < 2:
            continue

        d0, s0 = _polyline_distance_and_s(split_container, focus_pts[0][0], focus_pts[0][1])
        d1, s1 = _polyline_distance_and_s(split_container, focus_pts[-1][0], focus_pts[-1][1])
        if (not np.isfinite(float(d0))) or (not np.isfinite(float(d1))):
            continue
        if float(max(d0, d1)) > 3.0 * float(near_tol_req):
            continue

        if len(split_container) > len(container_pts):
            edge_container.points_xy = split_container
            stats["subset_containment_requests"] += 1

    for rid, (_ring, controls) in list(region_quad_setups.items()):
        ring_new = _ring_from_quad_controls(controls)
        if len(ring_new) >= 4:
            region_quad_setups[int(rid)] = (ring_new, controls)

    if collect_debug:
        region_pair_summary_map: Dict[Tuple[int, int], Dict[str, object]] = {}
        for rec in pair_debug_records:
            a = int(rec["region_i"])
            b = int(rec["region_j"])
            key = (min(a, b), max(a, b))
            ent = region_pair_summary_map.get(key)
            if ent is None:
                ent = {
                    "region_a": int(key[0]),
                    "region_b": int(key[1]),
                    "pair_count": 0,
                    "grouped_pair_count": 0,
                    "grouped_any": False,
                    "best_overlap_max": 0.0,
                    "best_overlap_min": 0.0,
                    "best_pair": None,
                }
                region_pair_summary_map[key] = ent

            ent["pair_count"] = int(ent["pair_count"]) + 1
            if bool(rec.get("grouped", False)):
                ent["grouped_pair_count"] = int(ent["grouped_pair_count"]) + 1
                ent["grouped_any"] = True

            best_now = (float(rec.get("overlap_max", 0.0)), float(rec.get("overlap_min", 0.0)))
            best_prev = (float(ent.get("best_overlap_max", 0.0)), float(ent.get("best_overlap_min", 0.0)))
            if best_now > best_prev:
                ent["best_overlap_max"] = float(best_now[0])
                ent["best_overlap_min"] = float(best_now[1])
                ent["best_pair"] = {
                    "edge_a": int(rec["edge_i"]) if int(rec["region_i"]) == int(key[0]) else int(rec["edge_j"]),
                    "edge_b": int(rec["edge_j"]) if int(rec["region_j"]) == int(key[1]) else int(rec["edge_i"]),
                    "grouped": bool(rec.get("grouped", False)),
                    "grouped_by": str(rec.get("grouped_by", "none")),
                    "overlap_ij": float(rec.get("overlap_ij", 0.0)),
                    "overlap_ji": float(rec.get("overlap_ji", 0.0)),
                    "near_tol": float(rec.get("near_tol", 0.0)),
                }

        groups_debug: List[Dict[str, object]] = []
        for root, members in edge_groups.items():
            owners = sorted({int(rid) for rid, _eid, _edge in members})
            groups_debug.append(
                {
                    "group_id": int(root),
                    "owner_region_ids": [int(v) for v in owners],
                    "member_count": int(len(members)),
                    "members": [
                        {
                            "region_id": int(rid),
                            "edge_id": int(eid),
                            "n_points": int(len(list(edge.points_xy or []))),
                            "target_size": None if edge.target_size is None else float(edge.target_size),
                        }
                        for rid, eid, edge in members
                    ],
                }
            )

        region_pair_summary = sorted(
            list(region_pair_summary_map.values()),
            key=lambda x: (
                -float(x.get("best_overlap_max", 0.0)),
                -float(x.get("best_overlap_min", 0.0)),
                int(x.get("region_a", 0)),
                int(x.get("region_b", 0)),
            ),
        )

        debug_capture.clear()
        debug_capture.update(
            {
                "n_edges": int(n_edges),
                "candidate_pair_count": int(candidate_pair_count_eval),
                "candidate_pair_count_prefilter": int(bucket_prefilter_candidate_pairs),
                "bbox_reject_count": int(pair_bbox_reject_count),
                "overlap_rule": {
                    "min_overlap_strict": 0.55,
                    "min_overlap_relaxed": 0.35,
                    "max_overlap_relaxed": 0.75,
                    "subset_containment_enable": bool(subset_enable),
                    "subset_containment_high_overlap": float(subset_high_overlap),
                    "subset_containment_min_overlap": float(subset_min_overlap),
                    "subset_containment_max_length_ratio": float(subset_max_ratio),
                },
                "pair_debug": list(pair_debug_records),
                "region_pair_summary": region_pair_summary,
                "groups": groups_debug,
            }
        )

    return edge_min_nodes, stats


def _gmsh_flow_aligned_curve_counts(
    quad_controls: Sequence[QuadEdgeControl],
    fallback_size: float,
    min_nodes: Optional[Sequence[int]] = None,
) -> Optional[List[int]]:
    """Compute transfinite node counts for a 4-edge flow-aligned block.

    Opposite edges are matched to satisfy Gmsh transfinite surface rules.
    """
    if len(quad_controls) != 4:
        return None

    base_size = max(float(fallback_size), 1.0e-9)
    counts: List[int] = []
    for edge in quad_controls:
        pts = list(edge.points_xy)
        if len(pts) < 2:
            return None
        edge_len = max(_polyline_length(pts), 1.0e-9)
        spacing = base_size
        if edge.target_size is not None:
            try:
                edge_sz = float(edge.target_size)
                if np.isfinite(edge_sz) and edge_sz > 0.0:
                    spacing = max(edge_sz, 1.0e-9)
            except Exception:
                pass
        ndiv = max(1, int(round(edge_len / spacing)))
        counts.append(max(2, ndiv + 1))

    if min_nodes is not None:
        mins = [max(0, int(v)) for v in list(min_nodes)]
        if len(mins) == 4:
            for i in range(4):
                if mins[i] > 0:
                    counts[i] = max(int(counts[i]), int(mins[i]))

    # Transfinite requires equal node counts on opposite edges. Prefer
    # canonical edge-id pairing when available so ordering/reversal of ring
    # traversal does not change which edges are equalized.
    edge_ids = [int(getattr(edge, "edge_id", -1)) for edge in quad_controls]
    if len(edge_ids) == 4 and set(edge_ids) == {1, 2, 3, 4}:
        idx_by_edge = {int(eid): int(i) for i, eid in enumerate(edge_ids)}
        i1 = idx_by_edge.get(1)
        i3 = idx_by_edge.get(3)
        if i1 is not None and i3 is not None:
            paired = max(int(counts[i1]), int(counts[i3]))
            counts[i1] = int(paired)
            counts[i3] = int(paired)
        i2 = idx_by_edge.get(2)
        i4 = idx_by_edge.get(4)
        if i2 is not None and i4 is not None:
            paired = max(int(counts[i2]), int(counts[i4]))
            counts[i2] = int(paired)
            counts[i4] = int(paired)
    else:
        counts[0] = counts[2] = max(counts[0], counts[2])
        counts[1] = counts[3] = max(counts[1], counts[3])
    return counts


def _gmsh_flow_align_region_preflight(
    region_id: int,
    cell_type: str,
    curve_tags: Sequence[int],
    edge_controls: Optional[Sequence[QuadEdgeControl]],
    fallback_size: float,
    min_nodes: Optional[Sequence[int]] = None,
) -> Dict[str, object]:
    """Validate whether full-region flow-aligned transfinite is safe to apply."""
    diag: Dict[str, object] = {
        "region_id": int(region_id),
        "cell_type": str(cell_type),
        "curve_count": int(len(curve_tags)),
        "eligible": False,
        "fallback": True,
        "reasons": [],
    }
    reasons: List[str] = []
    notes: List[str] = []

    if edge_controls is None:
        reasons.append("missing-quad-edge-controls")
    else:
        controls = list(edge_controls)
        diag["edge_count"] = int(len(controls))
        edge_ids = sorted(int(getattr(e, "edge_id", -1)) for e in controls)
        diag["edge_ids"] = edge_ids
        edge_vertex_counts_by_id: Dict[int, int] = {}
        for e in controls:
            eid = int(getattr(e, "edge_id", -1))
            edge_vertex_counts_by_id[eid] = int(len(list(e.points_xy or [])))
        if edge_vertex_counts_by_id:
            diag["edge_vertex_counts"] = [
                {
                    "edge_id": int(eid),
                    "n_vertices": int(edge_vertex_counts_by_id[eid]),
                }
                for eid in sorted(edge_vertex_counts_by_id.keys())
            ]
            if 1 in edge_vertex_counts_by_id and 3 in edge_vertex_counts_by_id:
                diag["edge_vertex_count_delta_1_3"] = int(
                    abs(int(edge_vertex_counts_by_id[1]) - int(edge_vertex_counts_by_id[3]))
                )
            if 2 in edge_vertex_counts_by_id and 4 in edge_vertex_counts_by_id:
                diag["edge_vertex_count_delta_2_4"] = int(
                    abs(int(edge_vertex_counts_by_id[2]) - int(edge_vertex_counts_by_id[4]))
                )
        if len(controls) != 4:
            reasons.append(f"expected-4-quad-edges-got-{len(controls)}")
        if set(edge_ids) != {1, 2, 3, 4}:
            reasons.append(f"edge-ids-must-be-1-2-3-4-got-{edge_ids}")

        join_gap_tol = max(1.0e-6, 1.0e-3 * max(float(fallback_size), 1.0e-9))
        join_gaps: List[float] = []
        for i in range(len(controls)):
            a_pts = list(controls[i].points_xy or [])
            b_pts = list(controls[(i + 1) % len(controls)].points_xy or [])
            if (not a_pts) or (not b_pts):
                join_gaps.append(float("inf"))
                continue
            ax, ay = float(a_pts[-1][0]), float(a_pts[-1][1])
            bx, by = float(b_pts[0][0]), float(b_pts[0][1])
            join_gaps.append(float(np.hypot(ax - bx, ay - by)))
        if join_gaps:
            diag["join_gaps"] = [float(v) for v in join_gaps]
            bad_join_idx = [i + 1 for i, g in enumerate(join_gaps) if np.isfinite(g) and g > join_gap_tol]
            if bad_join_idx:
                reasons.append(
                    "quad-edge-chain-disconnected "
                    f"(joins={bad_join_idx}, tol={join_gap_tol:.3e})"
                )

        ring: List[Tuple[float, float]] = []
        edge_lengths: List[float] = []
        tiny_seg_count = 0
        tiny_seg_tol = max(1.0e-9, 1.0e-3 * max(float(fallback_size), 1.0e-9))

        for ei, edge in enumerate(controls):
            pts = [(float(p[0]), float(p[1])) for p in list(edge.points_xy or [])]
            if len(pts) < 2:
                reasons.append(f"edge-{ei + 1}-has-fewer-than-2-points")
                continue
            edge_len = _polyline_length(pts)
            edge_lengths.append(float(edge_len))
            if edge_len <= 1.0e-9:
                reasons.append(f"edge-{ei + 1}-has-near-zero-length")

            for pi in range(1, len(pts)):
                seg_len = float(np.hypot(pts[pi][0] - pts[pi - 1][0], pts[pi][1] - pts[pi - 1][1]))
                if seg_len <= tiny_seg_tol:
                    tiny_seg_count += 1

            if not ring:
                ring.extend(pts)
            else:
                ring.extend(pts[1:])

        if edge_lengths:
            diag["edge_lengths"] = [float(v) for v in edge_lengths]
        if tiny_seg_count > 0:
            notes.append(f"contains-{tiny_seg_count}-very-short-segments<= {tiny_seg_tol:.3e}")

        if len(ring) >= 2 and np.hypot(ring[0][0] - ring[-1][0], ring[0][1] - ring[-1][1]) <= 1.0e-6:
            ring = ring[:-1]
        if len(ring) >= 4:
            inter_tol = _ring_intersection_tolerance(ring)
            inter_hits = _ring_self_intersections(ring, tol=inter_tol)
            if inter_hits:
                reasons.append(f"ring-self-intersections={len(inter_hits)}")

        counts = _gmsh_flow_aligned_curve_counts(
            controls,
            fallback_size=fallback_size,
            min_nodes=min_nodes,
        )
        if counts is None:
            reasons.append("could-not-compute-transfinite-counts")
        else:
            diag["transfinite_counts"] = [int(v) for v in counts]
            max_count = int(max(counts)) if counts else 0
            if max_count > 4096:
                reasons.append(f"transfinite-count-too-large={max_count}")

    if len(curve_tags) < 4:
        reasons.append(f"surface-must-have-4-curves-got-{len(curve_tags)}")
    elif len(curve_tags) != 4:
        notes.append(f"surface-curves-are-split({len(curve_tags)})")

    if notes:
        diag["notes"] = notes
    diag["reasons"] = reasons
    diag["eligible"] = len(reasons) == 0
    diag["fallback"] = not bool(diag["eligible"])
    return diag


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


def _sample_polyline_to_count(points: Sequence[Tuple[float, float]], n_vertices: int) -> List[Tuple[float, float]]:
    """Resample a polyline to an exact vertex count (including endpoints)."""
    pts = [(float(p[0]), float(p[1])) for p in list(points or [])]
    if len(pts) < 2:
        return pts

    n = max(2, int(n_vertices))
    if len(pts) == n:
        return pts

    out: List[Tuple[float, float]] = []
    for i in range(n):
        frac = float(i) / float(max(1, n - 1))
        out.append(_interp_polyline_fraction(pts, frac))

    if out:
        out[0] = (float(pts[0][0]), float(pts[0][1]))
        out[-1] = (float(pts[-1][0]), float(pts[-1][1]))
    return out


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


def _relax_fixed_edges_and_hints(
    fixed_edge_lines: Sequence[Sequence[Tuple[float, float]]],
    target_size: float,
    simplify_tol_factor: float = 0.35,
    hint_size_factor: float = 0.65,
    hint_spacing_factor: float = 1.25,
    hint_box_half_factor: float = 0.45,
    max_hint_boxes: int = 256,
) -> Tuple[List[List[Tuple[float, float]]], List[List[Tuple[float, float]]], List[float]]:
    """Build a relaxed breakline representation for fallback meshing.

    Returns simplified fixed-edge polylines plus soft local-size hint polygons.
    The hints preserve breakline influence without forcing strict edge conformance.
    """
    tsize = max(float(target_size), 1.0e-9)
    tol = max(1.0e-6, float(simplify_tol_factor) * tsize)
    hint_size = max(1.0e-9, float(hint_size_factor) * tsize)
    hint_step = max(1.0e-6, float(hint_spacing_factor) * tsize)
    hint_half = max(1.0e-6, float(hint_box_half_factor) * tsize)
    max_boxes = max(0, int(max_hint_boxes))

    relaxed_lines: List[List[Tuple[float, float]]] = []
    hint_polygons: List[List[Tuple[float, float]]] = []
    hint_sizes: List[float] = []
    seen_hint_keys = set()

    for line in fixed_edge_lines:
        clean: List[Tuple[float, float]] = []
        for p in line:
            pt = (float(p[0]), float(p[1]))
            if clean:
                if float(np.hypot(pt[0] - clean[-1][0], pt[1] - clean[-1][1])) <= 1.0e-12:
                    continue
            clean.append(pt)
        if len(clean) < 2:
            continue

        simplified = list(clean)
        if len(clean) > 2:
            simplified = _rdp_open_polyline(clean, tol=tol)
            if len(simplified) < 2:
                simplified = [clean[0], clean[-1]]
            else:
                simplified[0] = clean[0]
                simplified[-1] = clean[-1]

        relaxed_lines.append([(float(x), float(y)) for (x, y) in simplified])

        if max_boxes <= 0:
            continue
        sampled = _sample_polyline(simplified, hint_step)
        if not sampled:
            sampled = [simplified[0], simplified[-1]]

        for x, y in sampled:
            key = (int(np.rint(float(x) / hint_half)), int(np.rint(float(y) / hint_half)))
            if key in seen_hint_keys:
                continue
            seen_hint_keys.add(key)

            hint_polygons.append(
                [
                    (float(x - hint_half), float(y - hint_half)),
                    (float(x + hint_half), float(y - hint_half)),
                    (float(x + hint_half), float(y + hint_half)),
                    (float(x - hint_half), float(y + hint_half)),
                ]
            )
            hint_sizes.append(float(hint_size))
            if len(hint_polygons) >= max_boxes:
                break

        if len(hint_polygons) >= max_boxes:
            break

    return relaxed_lines, hint_polygons, hint_sizes


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


def _ring_self_intersections(
    ring: Sequence[Tuple[float, float]],
    tol: float = 1.0e-6,
) -> List[Tuple[int, int, Tuple[float, float]]]:
    """Return non-adjacent segment intersections in a closed ring.

    The returned tuples are ``(edge_i, edge_j, (x, y))`` where ``edge_i`` is the
    segment ``ring[i] -> ring[(i+1)%n]``.
    """
    pts = [(float(x), float(y)) for (x, y) in ring]
    if len(pts) >= 2 and np.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) <= tol:
        pts = pts[:-1]
    n = len(pts)
    if n < 4:
        return []

    hits: List[Tuple[int, int, Tuple[float, float]]] = []
    for i in range(n):
        p0 = pts[i]
        p1 = pts[(i + 1) % n]
        if np.hypot(p1[0] - p0[0], p1[1] - p0[1]) <= tol:
            continue
        for j in range(i + 1, n):
            # Adjacent edges share a vertex by construction and are excluded.
            if j == i or j == (i + 1) % n or i == (j + 1) % n:
                continue
            if i == 0 and j == n - 1:
                continue
            q0 = pts[j]
            q1 = pts[(j + 1) % n]
            if np.hypot(q1[0] - q0[0], q1[1] - q0[1]) <= tol:
                continue
            inter = _segment_intersection_point(p0, p1, q0, q1, eps=tol)
            if inter is None:
                continue
            hits.append((i, j, (float(inter[0][0]), float(inter[0][1]))))
    return hits


def _ring_intersection_tolerance(ring: Sequence[Tuple[float, float]]) -> float:
    if not ring:
        return 1.0e-6
    rx = np.asarray([float(p[0]) for p in ring], dtype=np.float64)
    ry = np.asarray([float(p[1]) for p in ring], dtype=np.float64)
    return max(1.0e-6, 1.0e-9 * max(float(np.ptp(rx)), float(np.ptp(ry)), 1.0))


def _recover_tqmesh_exterior_boundary(
    ext_ring: Sequence[Tuple[float, float]],
    fallback_ring: Optional[Sequence[Tuple[float, float]]],
    target_size: float,
    protect_points: Optional[Sequence[Tuple[float, float]]] = None,
    protect_tol: float = 1.0e-6,
    region_id: Optional[int] = None,
) -> Tuple[List[Tuple[float, float]], bool]:
    """Recover a safer exterior boundary if sanitize/stitch produced crossings.

    Returns ``(ring, used_fallback_source)``.  If no improvement is found,
    the original ``ext_ring`` is returned unchanged.
    """
    base = [(float(x), float(y)) for (x, y) in ext_ring]
    if len(base) >= 2 and np.hypot(base[0][0] - base[-1][0], base[0][1] - base[-1][1]) <= 1.0e-12:
        base = base[:-1]
    if len(base) < 4:
        return base, False

    tsize = max(float(target_size), 1.0e-9)
    tol = _ring_intersection_tolerance(base)
    base_hits = _ring_self_intersections(base, tol=tol)
    if not base_hits:
        return base, False

    best_ring = list(base)
    best_hits = len(base_hits)
    best_tag = "base"
    best_used_fallback = False

    def _register_candidate(tag: str, ring_xy: Sequence[Tuple[float, float]], used_fallback: bool) -> None:
        nonlocal best_ring, best_hits, best_tag, best_used_fallback
        cand = [(float(x), float(y)) for (x, y) in ring_xy]
        if len(cand) >= 2 and np.hypot(cand[0][0] - cand[-1][0], cand[0][1] - cand[-1][1]) <= 1.0e-12:
            cand = cand[:-1]
        if len(cand) < 3:
            return
        ct = _ring_intersection_tolerance(cand)
        hits = len(_ring_self_intersections(cand, tol=ct))
        if hits < best_hits:
            best_ring = cand
            best_hits = hits
            best_tag = tag
            best_used_fallback = bool(used_fallback)

    def _prepare_from_source(
        source_ring: Sequence[Tuple[float, float]],
        do_stitch: bool,
        do_simplify: bool,
    ) -> List[Tuple[float, float]]:
        rr = [(float(x), float(y)) for (x, y) in source_ring]
        if len(rr) >= 2 and np.hypot(rr[0][0] - rr[-1][0], rr[0][1] - rr[-1][1]) <= 1.0e-12:
            rr = rr[:-1]
        if len(rr) < 3:
            return []
        if do_simplify:
            simp_max = max(24, min(192, int(len(rr))))
            rr = _simplify_closed_ring(rr, tol=max(1.0e-6, 0.15 * tsize), max_vertices=simp_max)
        rr = _sanitize_closed_ring(
            rr,
            length_tol=max(1.0e-6, 1.0e-3 * tsize),
            collinear_tol=max(1.0e-8, 7.5e-4 * tsize),
            protect_points=protect_points,
            protect_tol=max(1.0e-6, float(protect_tol)),
        )
        if do_stitch:
            rr = _stitch_boundary_microchains(
                rr,
                target_size=float(tsize),
                protect_points=protect_points,
                protect_tol=max(1.0e-6, float(protect_tol)),
            )
        return [(float(x), float(y)) for (x, y) in rr]

    _register_candidate("base-no-stitch", _prepare_from_source(base, do_stitch=False, do_simplify=False), used_fallback=False)
    _register_candidate("base-simplified", _prepare_from_source(base, do_stitch=True, do_simplify=True), used_fallback=False)

    if fallback_ring is not None:
        fb = [(float(x), float(y)) for (x, y) in fallback_ring]
        _register_candidate("fallback-no-stitch", _prepare_from_source(fb, do_stitch=False, do_simplify=False), used_fallback=True)
        _register_candidate("fallback-stitched", _prepare_from_source(fb, do_stitch=True, do_simplify=False), used_fallback=True)
        _register_candidate("fallback-simplified", _prepare_from_source(fb, do_stitch=True, do_simplify=True), used_fallback=True)

    if best_hits < len(base_hits):
        rid_txt = "?" if region_id is None else str(int(region_id))
        warnings.warn(
            "TQMesh exterior boundary recovery adjusted region "
            f"{rid_txt} after sanitize/stitch produced self-intersections "
            f"(before={len(base_hits)}, after={best_hits}, strategy={best_tag}).",
            RuntimeWarning,
        )
        return best_ring, bool(best_used_fallback)

    return base, False


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

    # Ensure opposite edges carry identical sampled vertex counts, so shared
    # transfinite interfaces are not sensitive to length-only discretization.
    idx_by_edge_id = {
        int(edge.edge_id): int(i)
        for i, edge in enumerate(normalized_edges)
        if int(edge.edge_id) in {1, 2, 3, 4}
    }
    for edge_a, edge_b in ((1, 3), (2, 4)):
        ia = idx_by_edge_id.get(int(edge_a))
        ib = idx_by_edge_id.get(int(edge_b))
        if ia is None or ib is None:
            continue

        pts_a = list(normalized_edges[int(ia)].points_xy or [])
        pts_b = list(normalized_edges[int(ib)].points_xy or [])
        if len(pts_a) < 2 or len(pts_b) < 2:
            continue

        n_target = max(2, int(len(pts_a)), int(len(pts_b)))
        if len(pts_a) != n_target:
            normalized_edges[int(ia)] = QuadEdgeControl(
                region_id=normalized_edges[int(ia)].region_id,
                edge_id=normalized_edges[int(ia)].edge_id,
                points_xy=_sample_polyline_to_count(pts_a, int(n_target)),
                target_size=normalized_edges[int(ia)].target_size,
                n_layers=normalized_edges[int(ia)].n_layers,
                first_height=normalized_edges[int(ia)].first_height,
                growth_rate=normalized_edges[int(ia)].growth_rate,
            )
        if len(pts_b) != n_target:
            normalized_edges[int(ib)] = QuadEdgeControl(
                region_id=normalized_edges[int(ib)].region_id,
                edge_id=normalized_edges[int(ib)].edge_id,
                points_xy=_sample_polyline_to_count(pts_b, int(n_target)),
                target_size=normalized_edges[int(ib)].target_size,
                n_layers=normalized_edges[int(ib)].n_layers,
                first_height=normalized_edges[int(ib)].first_height,
                growth_rate=normalized_edges[int(ib)].growth_rate,
            )

    # Rebuild closed ring from parity-adjusted edge samples.
    ring = []
    for edge in normalized_edges:
        sampled = [(float(p[0]), float(p[1])) for p in list(edge.points_xy or [])]
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

    # Validity gate: reject self-intersecting assembled quad rings and fall
    # back to non-structured region meshing.
    quad_ring: List[Tuple[float, float]] = []
    for chain in (bottom, right, top, left):
        if not quad_ring:
            quad_ring.extend((float(x), float(y)) for (x, y) in chain)
            continue
        prev = quad_ring[-1]
        cur = chain[0]
        if np.hypot(prev[0] - cur[0], prev[1] - cur[1]) <= 1.0e-6:
            quad_ring.extend((float(x), float(y)) for (x, y) in chain[1:])
        else:
            quad_ring.extend((float(x), float(y)) for (x, y) in chain)
    if len(quad_ring) >= 2 and np.hypot(quad_ring[0][0] - quad_ring[-1][0], quad_ring[0][1] - quad_ring[-1][1]) <= 1.0e-6:
        quad_ring = quad_ring[:-1]
    if len(quad_ring) < 4:
        return None

    rx = np.asarray([p[0] for p in quad_ring], dtype=np.float64)
    ry = np.asarray([p[1] for p in quad_ring], dtype=np.float64)
    gate_tol = max(1.0e-6, 1.0e-9 * max(float(np.ptp(rx)), float(np.ptp(ry)), 1.0))
    intersections = _ring_self_intersections(quad_ring, tol=gate_tol)
    if intersections:
        preview = ", ".join(
            f"e{i}-e{j}@({pt[0]:.3f},{pt[1]:.3f})"
            for i, j, pt in intersections[:3]
        )
        warnings.warn(
            "Structured quad ring validity gate rejected region "
            f"{int(region.region_id)} due to self-intersections "
            f"(count={len(intersections)}, sample={preview}). "
            "Falling back to non-structured meshing for this region.",
            RuntimeWarning,
        )
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


def _is_transfinite_cell_type_label(cell_type_label: object) -> bool:
    label = str(cell_type_label).strip().lower()
    return label in {"cartesian", "quadrilateral", "channel_generator"}


def _mixed_transfinite_tri_near_unshared_report(
    mesh: MeshResult,
    tol: float = 1.0e-3,
    max_pairs: int = 32,
) -> Dict[str, object]:
    """Detect near-coincident but unshared nodes on mixed transfinite/tri interfaces.

    This catches the classic hanging-node pattern where one side has a split
    edge but the neighboring side does not share the intermediate node.
    """
    tol_use = max(float(tol), 1.0e-9)
    tol2 = float(tol_use * tol_use)

    node_x = np.asarray(mesh.node_x, dtype=np.float64)
    node_y = np.asarray(mesh.node_y, dtype=np.float64)
    offs = np.asarray(mesh.cell_face_offsets, dtype=np.int32)
    conn = np.asarray(mesh.cell_face_nodes, dtype=np.int32)
    rid = np.asarray(mesh.region_id, dtype=np.int32)
    ctype = np.asarray(mesh.cell_type)

    if offs.size < 2:
        return {
            "tol": float(tol_use),
            "pair_count_checked": 0,
            "flagged_pair_count": 0,
            "flagged_pairs": [],
        }

    edge_owner_cells: Dict[Tuple[int, int], List[int]] = {}
    boundary_nodes_by_region: Dict[int, set] = {}

    for ci in range(int(offs.size) - 1):
        s = int(offs[ci])
        e = int(offs[ci + 1])
        poly = conn[s:e]
        if poly.size < 2:
            continue
        rr = int(rid[ci])
        rnodes = boundary_nodes_by_region.setdefault(rr, set())
        for k in range(int(poly.size)):
            a = int(poly[k])
            b = int(poly[(k + 1) % poly.size])
            key = (a, b) if a < b else (b, a)
            edge_owner_cells.setdefault(key, []).append(int(ci))
            rnodes.add(int(a))
            rnodes.add(int(b))

    pair_shared_edges: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    pair_shared_nodes: Dict[Tuple[int, int], set] = {}

    for key, owners in edge_owner_cells.items():
        if len(owners) != 2:
            continue
        c0 = int(owners[0])
        c1 = int(owners[1])
        r0 = int(rid[c0])
        r1 = int(rid[c1])
        if r0 == r1:
            continue

        c0_tf = _is_transfinite_cell_type_label(ctype[c0])
        c1_tf = _is_transfinite_cell_type_label(ctype[c1])
        c0_tri = str(ctype[c0]).strip().lower() == "triangular"
        c1_tri = str(ctype[c1]).strip().lower() == "triangular"
        is_mixed = (c0_tf and c1_tri) or (c1_tf and c0_tri)
        if not is_mixed:
            continue

        pair = (r0, r1) if r0 < r1 else (r1, r0)
        pair_shared_edges.setdefault(pair, []).append((int(key[0]), int(key[1])))
        pnodes = pair_shared_nodes.setdefault(pair, set())
        pnodes.add(int(key[0]))
        pnodes.add(int(key[1]))

    flagged_pairs: List[Dict[str, object]] = []
    pair_count_checked = 0

    for pair, edges in sorted(pair_shared_edges.items(), key=lambda kv: len(kv[1]), reverse=True):
        if not edges:
            continue
        pair_count_checked += 1
        ra, rb = int(pair[0]), int(pair[1])
        segs = [
            (
                float(node_x[int(a)]),
                float(node_y[int(a)]),
                float(node_x[int(b)]),
                float(node_y[int(b)]),
            )
            for a, b in edges
        ]
        sx = [float(v) for seg in segs for v in (seg[0], seg[2])]
        sy = [float(v) for seg in segs for v in (seg[1], seg[3])]
        seg_bbox = (
            float(min(sx) - tol_use),
            float(min(sy) - tol_use),
            float(max(sx) + tol_use),
            float(max(sy) + tol_use),
        )

        def _near_shared_geometry(node_id: int) -> bool:
            px = float(node_x[int(node_id)])
            py = float(node_y[int(node_id)])
            if px < seg_bbox[0] or px > seg_bbox[2] or py < seg_bbox[1] or py > seg_bbox[3]:
                return False
            for ax, ay, bx, by in segs:
                d, _ = _point_to_segment_distance_s(px, py, ax, ay, bx, by)
                if float(d) <= float(tol_use):
                    return True
            return False

        cand_a = [int(n) for n in boundary_nodes_by_region.get(int(ra), set()) if _near_shared_geometry(int(n))]
        cand_b = [int(n) for n in boundary_nodes_by_region.get(int(rb), set()) if _near_shared_geometry(int(n))]

        if not cand_a or not cand_b:
            continue

        set_a = set(int(v) for v in cand_a)
        set_b = set(int(v) for v in cand_b)
        shared_exact = int(len(set_a & set_b))

        xb = node_x[np.asarray(cand_b, dtype=np.int32)]
        yb = node_y[np.asarray(cand_b, dtype=np.int32)]
        xa = node_x[np.asarray(cand_a, dtype=np.int32)]
        ya = node_y[np.asarray(cand_a, dtype=np.int32)]

        near_only_a = 0
        for n in set_a:
            if int(n) in set_b:
                continue
            dx = xb - float(node_x[int(n)])
            dy = yb - float(node_y[int(n)])
            if np.any((dx * dx + dy * dy) <= tol2):
                near_only_a += 1

        near_only_b = 0
        for n in set_b:
            if int(n) in set_a:
                continue
            dx = xa - float(node_x[int(n)])
            dy = ya - float(node_y[int(n)])
            if np.any((dx * dx + dy * dy) <= tol2):
                near_only_b += 1

        if near_only_a > 0 or near_only_b > 0:
            flagged_pairs.append(
                {
                    "region_pair": [int(ra), int(rb)],
                    "shared_edge_count": int(len(edges)),
                    "shared_nodes_exact": int(shared_exact),
                    "candidate_nodes_a": int(len(cand_a)),
                    "candidate_nodes_b": int(len(cand_b)),
                    "near_only_a": int(near_only_a),
                    "near_only_b": int(near_only_b),
                }
            )

    return {
        "tol": float(tol_use),
        "pair_count_checked": int(pair_count_checked),
        "flagged_pair_count": int(len(flagged_pairs)),
        "flagged_pairs": list(flagged_pairs[: max(1, int(max_pairs))]),
    }


def _enforce_quad_interface_conformance(
    mesh: MeshResult,
    model: ConceptualModel,
    snap_tol: float = 1.0,
    centroid_merge: bool = False,
) -> MeshResult:
    """Snap adjacent-region interface nodes onto quad-region edge node lines.

    This enforces shared node placement along interfaces for independently meshed
    regions (e.g. triangular region next to a flow-aligned quad block).

    When ``centroid_merge`` is enabled, matched interface node groups are moved
    to their centroid prior to welding instead of snapping one side directly
    onto the other side's node coordinates.
    """
    tol = max(float(snap_tol), 1.0e-9)
    node_x = np.asarray(mesh.node_x, dtype=np.float64).copy()
    node_y = np.asarray(mesh.node_y, dtype=np.float64).copy()
    orig_x = node_x.copy()
    orig_y = node_y.copy()

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

    matched_pairs: List[Tuple[int, int]] = []
    matched_pair_seen: set = set()

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

                if int(n) == int(ref):
                    continue

                pair = (int(min(n, ref)), int(max(n, ref)))
                if pair in matched_pair_seen:
                    continue
                matched_pair_seen.add(pair)
                matched_pairs.append((int(n), int(ref)))

    centroid_group_count = 0
    if matched_pairs:
        if bool(centroid_merge):
            # Merge matched interface node groups at their group centroid.
            parent = np.arange(node_x.size, dtype=np.int32)
            rank = np.zeros(node_x.size, dtype=np.int8)

            def _find(i: int) -> int:
                j = int(i)
                while int(parent[j]) != j:
                    parent[j] = parent[int(parent[j])]
                    j = int(parent[j])
                return int(j)

            def _union(a: int, b: int) -> None:
                ra = _find(int(a))
                rb = _find(int(b))
                if ra == rb:
                    return
                if int(rank[ra]) < int(rank[rb]):
                    parent[ra] = int(rb)
                elif int(rank[ra]) > int(rank[rb]):
                    parent[rb] = int(ra)
                else:
                    parent[rb] = int(ra)
                    rank[ra] = np.int8(int(rank[ra]) + 1)

            touched: set = set()
            for n, ref in matched_pairs:
                touched.add(int(n))
                touched.add(int(ref))
                _union(int(n), int(ref))

            groups: Dict[int, List[int]] = {}
            for idx in touched:
                root = _find(int(idx))
                groups.setdefault(int(root), []).append(int(idx))

            for group_nodes in groups.values():
                if not group_nodes:
                    continue
                gx = float(np.mean(orig_x[np.asarray(group_nodes, dtype=np.int32)]))
                gy = float(np.mean(orig_y[np.asarray(group_nodes, dtype=np.int32)]))
                for nid in group_nodes:
                    node_x[int(nid)] = float(gx)
                    node_y[int(nid)] = float(gy)
            centroid_group_count = int(len(groups))
        else:
            for n, ref in matched_pairs:
                node_x[int(n)] = node_x[int(ref)]
                node_y[int(n)] = node_y[int(ref)]

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

    quality_summary = dict(mesh.quality_summary or {})
    quality_summary["interface_conformance_postprocess"] = {
        "snap_tol": float(tol),
        "centroid_merge": bool(centroid_merge),
        "matched_pair_count": int(len(matched_pairs)),
        "centroid_group_count": int(centroid_group_count),
    }

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
        quality_summary=quality_summary,
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
        self._last_flow_align_diagnostics: List[Dict[str, object]] = []
        self._last_build_order_fingerprint: Dict[str, object] = {}
        self._last_build_order_stage_ladder: Dict[str, object] = {}
        self._last_pre_generate_entity_signature: Dict[str, object] = {}

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
        progress_path = str(self._options.get("gmsh_progress_path", "") or "").strip()
        progress_emit_interval_s = _as_float(
            self._options.get("gmsh_progress_emit_interval_s"),
            0.75,
        )
        if (not np.isfinite(progress_emit_interval_s)) or progress_emit_interval_s <= 0.0:
            progress_emit_interval_s = 0.75
        progress_emit_interval_s = max(float(progress_emit_interval_s), 0.2)
        progress_seq = 0
        progress_last_emit = -1.0
        t_start = time.perf_counter()

        def _clip_progress_detail(detail: object, max_len: int = 240) -> str:
            txt = str(detail or "").strip()
            if len(txt) <= max_len:
                return txt
            return txt[: max_len - 3] + "..."

        def _emit_progress(
            stage: str,
            attempt: Optional[int] = None,
            detail: str = "",
            force: bool = False,
        ) -> None:
            nonlocal progress_seq, progress_last_emit
            if not progress_path:
                return
            now = time.perf_counter()
            if (not force) and progress_last_emit >= 0.0:
                if (now - progress_last_emit) < progress_emit_interval_s:
                    return
            progress_seq += 1
            payload: Dict[str, object] = {
                "seq": int(progress_seq),
                "stage": str(stage),
                "timestamp": float(time.time()),
                "elapsed_s": float(max(0.0, now - t_start)),
                "backend": "gmsh",
                "quality_loop_enabled": bool(quality_cfg.enabled),
            }
            if attempt is not None:
                payload["attempt"] = int(attempt)
            clipped = _clip_progress_detail(detail)
            if clipped:
                payload["detail"] = clipped
            try:
                _write_json_atomic(progress_path, payload)
                progress_last_emit = now
            except Exception:
                pass

        _emit_progress(
            "start",
            detail=(
                f"quality_loop={bool(quality_cfg.enabled)} max_iters={int(quality_cfg.max_iterations)} "
                f"budget_s={float(quality_cfg.time_limit_s):.2f}"
            ),
            force=True,
        )

        gmsh_logger_started = False
        gmsh_logger_emitted: set = set()

        def _emit_gmsh_logger_warnings() -> None:
            if not gmsh_logger_started:
                return
            try:
                msgs = list(gmsh.logger.get())
            except Exception:
                return
            for raw in msgs:
                msg = str(raw).strip()
                if not msg:
                    continue
                low = msg.lower()
                if ("warning" not in low) and ("error" not in low):
                    continue
                if msg in gmsh_logger_emitted:
                    continue
                gmsh_logger_emitted.add(msg)
                warnings.warn(
                    f"Gmsh logger: {msg}",
                    RuntimeWarning,
                )

        # `interruptible=False` avoids installing a SIGINT handler, which lets
        # the Python API run from the QGIS bridge worker thread.
        gmsh.initialize(interruptible=False)
        gmsh.option.setNumber("General.Verbosity", float(verbosity))
        try:
            gmsh.logger.start()
            gmsh_logger_started = True
        except Exception:
            gmsh_logger_started = False

        try:
            if not quality_cfg.enabled:
                flow_align_requested = self._opt_bool(
                    "gmsh_quad_full_region_flow_align",
                    _env_bool("BACKWATER_GMSH_QUAD_FULL_REGION_FLOW_ALIGN", False),
                )
                gmsh.model.add("swe2d")
                _emit_progress("build-start", detail="single-pass mode", force=True)
                try:
                    mesh = _require_nonempty_mesh(
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
                    _emit_gmsh_logger_warnings()
                    n_nodes = int(np.asarray(mesh.node_x).size)
                    n_faces = max(0, int(np.asarray(mesh.cell_face_offsets).size) - 1)
                    _emit_progress("done", detail=f"nodes={n_nodes} faces={n_faces}", force=True)
                    return mesh
                except Exception as exc:
                    _emit_gmsh_logger_warnings()
                    if not flow_align_requested:
                        _emit_progress("fail", detail=f"single-pass build failed: {exc}", force=True)
                        raise

                    _emit_progress(
                        "flow-align-fallback-start",
                        detail=f"initial flow-align build failed: {exc}",
                        force=True,
                    )
                    diagnostics = copy.deepcopy(self._last_flow_align_diagnostics)
                    warnings.warn(
                        "Gmsh full-region flow-aligned quads failed on initial pass; "
                        "retrying with per-region flow-align disabled. "
                        f"Initial error: {exc}",
                        RuntimeWarning,
                    )

                    prev_flow_align = self._options.get("gmsh_quad_full_region_flow_align", None)
                    self._options["gmsh_quad_full_region_flow_align"] = False
                    try:
                        gmsh.clear()
                        gmsh.model.add("swe2d_fallback_no_flow_align")
                        _emit_progress(
                            "build-retry",
                            detail="retry with per-region flow-align disabled",
                            force=True,
                        )
                        try:
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
                                ),
                                "Gmsh",
                            )
                            _emit_gmsh_logger_warnings()
                            n_nodes = int(np.asarray(fallback_mesh.node_x).size)
                            n_faces = max(0, int(np.asarray(fallback_mesh.cell_face_offsets).size) - 1)
                            _emit_progress(
                                "done",
                                detail=(
                                    f"flow-align fallback success nodes={n_nodes} faces={n_faces} "
                                    "mode=no-flow-align"
                                ),
                                force=True,
                            )
                        except Exception as fallback_exc:
                            _emit_gmsh_logger_warnings()
                            diag_txt = "none"
                            if diagnostics:
                                parts = []
                                for d in diagnostics:
                                    rid_txt = str(d.get("region_id", "?"))
                                    status_txt = str(d.get("status", "unknown"))
                                    reasons_txt = ",".join(str(x) for x in d.get("reasons", []) if str(x))
                                    if not reasons_txt:
                                        reasons_txt = "none"
                                    parts.append(
                                        f"region={rid_txt};status={status_txt};reasons={reasons_txt}"
                                    )
                                diag_txt = " | ".join(parts)
                            _emit_progress(
                                "fail",
                                detail=(
                                    "flow-align fallback failed "
                                    f"initial={_clip_progress_detail(exc)} "
                                    f"fallback={_clip_progress_detail(fallback_exc)}"
                                ),
                                force=True,
                            )
                            raise RuntimeError(
                                "Gmsh flow-align fallback retry failed. "
                                f"initial_error={exc}; fallback_error={fallback_exc}; "
                                f"per_region_diagnostics={diag_txt}"
                            )
                    finally:
                        if prev_flow_align is None:
                            self._options.pop("gmsh_quad_full_region_flow_align", None)
                        else:
                            self._options["gmsh_quad_full_region_flow_align"] = prev_flow_align

                    merged_summary = dict(fallback_mesh.quality_summary or {})
                    merged_summary["gmsh_flow_align_runtime_fallback"] = {
                        "triggered": True,
                        "initial_error": str(exc),
                    }
                    if diagnostics:
                        merged_summary["gmsh_flow_align_diagnostics"] = diagnostics
                    fallback_mesh.quality_summary = merged_summary
                    return fallback_mesh

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
                    _emit_progress(
                        "budget-stop",
                        attempt=int(attempts),
                        detail=(
                            f"elapsed={elapsed:.2f}s reached budget={float(quality_cfg.time_limit_s):.2f}s"
                        ),
                        force=True,
                    )
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
                        _emit_progress(
                            "budget-stop",
                            attempt=int(attempts),
                            detail=(
                                f"remaining={remaining_s:.2f}s too low for new attempt "
                                f"need~{min_retry_window_s:.2f}s"
                            ),
                            force=True,
                        )
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

                _emit_progress(
                    "attempt-start",
                    attempt=int(attempts + 1),
                    detail=(
                        f"tri={tri_try} quad={quad_try} recomb={recomb_try} "
                        f"topo={int(recomb_topology_try)} minq={float(recomb_min_quality_try):.3f} "
                        f"rand={float(random_factor_try):.2e} size_scale={float(size_scale):.3f} "
                        f"smooth={int(max(0, smoothing_passes + int(smooth_inc)))}"
                    ),
                    force=True,
                )

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
                    _emit_gmsh_logger_warnings()
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
                            _emit_progress(
                                "done",
                                attempt=int(attempts + 1),
                                detail="strict mode accepted first passing candidate",
                                force=True,
                            )
                            # Strict mode only needs the first passing candidate.
                            summary = dict(mesh.quality_summary or {})
                            summary.update({
                                "attempts": int(attempts + 1),
                                "strict_requested": bool(quality_cfg.strict),
                                "had_passing_candidate": True,
                                "best_stats": dict(stats),
                            })
                            mesh.quality_summary = summary
                            return mesh
                except Exception as exc:
                    _emit_gmsh_logger_warnings()
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
                    _emit_progress(
                        "attempt-fail",
                        attempt=int(attempts + 1),
                        detail=err_msg,
                        force=True,
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
                    _emit_progress(
                        "attempt-done",
                        attempt=int(attempts + 1),
                        detail=(
                            f"passed={bool(_gmsh_quality_passes(stats, quality_cfg))} "
                            f"fail_any={int(stats.get('failed_any_cells', 0.0))}"
                        ),
                        force=True,
                    )

                last_attempt_duration_s = max(0.0, time.perf_counter() - attempt_start_t)
                attempts += 1

            if best_mesh is None or best_stats is None:
                # Best-effort fallback: regardless of iterative quality failures,
                # run one plain baseline build so downstream export still has a mesh
                # whenever geometry is meshable at all.
                try:
                    _emit_progress("fallback-start", detail="building best-effort baseline candidate", force=True)
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
                    _emit_gmsh_logger_warnings()
                    fallback_stats = _face_mesh_quality_stats(fallback_mesh, quality_cfg)
                    fallback_summary = dict(fallback_mesh.quality_summary or {})
                    fallback_summary.update({
                        "attempts": int(attempts + 1),
                        "strict_requested": bool(quality_cfg.strict),
                        "had_passing_candidate": bool(_gmsh_quality_passes(fallback_stats, quality_cfg)),
                        "best_stats": dict(fallback_stats),
                        "best_effort_fallback": True,
                        "time_budget_exhausted": bool(hit_time_budget),
                    })
                    fallback_mesh.quality_summary = fallback_summary
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
                    n_nodes = int(np.asarray(fallback_mesh.node_x).size)
                    n_faces = max(0, int(np.asarray(fallback_mesh.cell_face_offsets).size) - 1)
                    _emit_progress(
                        "done",
                        detail=f"best-effort fallback nodes={n_nodes} faces={n_faces}",
                        force=True,
                    )
                    return fallback_mesh
                except Exception as fallback_exc:
                    tail = "; ".join(attempt_errors[-3:]) if attempt_errors else "no attempt diagnostics"
                    _emit_progress(
                        "fail",
                        detail=(
                            "quality loop had no viable candidate and fallback failed: "
                            f"{_clip_progress_detail(fallback_exc)}"
                        ),
                        force=True,
                    )
                    raise RuntimeError(
                        "Gmsh quality loop produced no valid non-empty mesh candidate, and "
                        f"best-effort fallback also failed: {fallback_exc}. "
                        f"Recent attempt errors: {tail}"
                    )

            if had_passing_candidate:
                summary = dict(best_mesh.quality_summary or {})
                summary.update({
                    "attempts": int(attempts),
                    "strict_requested": bool(quality_cfg.strict),
                    "had_passing_candidate": True,
                    "best_stats": dict(best_stats),
                    "time_budget_exhausted": bool(hit_time_budget),
                })
                best_mesh.quality_summary = summary
                n_nodes = int(np.asarray(best_mesh.node_x).size)
                n_faces = max(0, int(np.asarray(best_mesh.cell_face_offsets).size) - 1)
                _emit_progress(
                    "done",
                    detail=(
                        f"best passing candidate nodes={n_nodes} faces={n_faces} "
                        f"attempts={int(attempts)}"
                    ),
                    force=True,
                )
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
            summary = dict(best_mesh.quality_summary or {})
            summary.update({
                "attempts": int(attempts),
                "strict_requested": bool(quality_cfg.strict),
                "had_passing_candidate": False,
                "best_stats": dict(best_stats),
                "time_budget_exhausted": bool(hit_time_budget),
            })
            best_mesh.quality_summary = summary
            warnings.warn(
                "Gmsh quality constraints were not met; using best available candidate "
                f"(attempts={attempts}, time_limit_s={quality_cfg.time_limit_s:.1f}). {diag}",
                RuntimeWarning,
            )
            n_nodes = int(np.asarray(best_mesh.node_x).size)
            n_faces = max(0, int(np.asarray(best_mesh.cell_face_offsets).size) - 1)
            _emit_progress(
                "done",
                detail=(
                    f"best nonpassing candidate nodes={n_nodes} faces={n_faces} "
                    f"attempts={int(attempts)}"
                ),
                force=True,
            )
            return best_mesh
        finally:
            _emit_gmsh_logger_warnings()
            if gmsh_logger_started:
                try:
                    gmsh.logger.stop()
                except Exception:
                    pass
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
        build_started_at = time.perf_counter()
        gmsh_phase_timings_s: Dict[str, float] = {}

        def _record_phase(phase_name: str, started_at: float) -> None:
            gmsh_phase_timings_s[str(phase_name)] = float(max(0.0, time.perf_counter() - started_at))

        arc_mode = str(self._options.get("gmsh_arc_mode", "hard_embed") or "hard_embed").strip().lower()
        if arc_mode not in {"hard_embed", "soft_size_hint", "disabled"}:
            arc_mode = "hard_embed"
        mesh_size_min = max(0.0, self._opt_float("gmsh_mesh_size_min", 0.0))
        tolerance_edge_length = max(0.0, self._opt_float("gmsh_tolerance_edge_length", 0.0))
        mesh_size_from_points = self._opt_bool("gmsh_mesh_size_from_points", True)
        gmsh_num_threads = max(
            0,
            int(
                round(
                    self._opt_float(
                        "gmsh_num_threads",
                        _env_float("BACKWATER_GMSH_NUM_THREADS", 1.0),
                    )
                )
            ),
        )
        gmsh_max_num_threads_2d = max(
            0,
            int(
                round(
                    self._opt_float(
                        "gmsh_max_num_threads_2d",
                        _env_float("BACKWATER_GMSH_MAX_NUM_THREADS_2D", 0.0),
                    )
                )
            ),
        )
        gmsh_global_recombine = self._opt_bool(
            "gmsh_global_recombine",
            _env_bool("BACKWATER_GMSH_GLOBAL_RECOMBINE", False),
        )
        gmsh_quad_full_region_flow_align = self._opt_bool(
            "gmsh_quad_full_region_flow_align",
            _env_bool("BACKWATER_GMSH_QUAD_FULL_REGION_FLOW_ALIGN", False),
        )
        gmsh_interface_transition_enable = self._opt_bool(
            "gmsh_interface_transition_enable",
            _env_bool("BACKWATER_GMSH_INTERFACE_TRANSITION_ENABLE", True),
        )
        gmsh_interface_transition_dist_factor = max(
            0.25,
            self._opt_float(
                "gmsh_interface_transition_dist_factor",
                _env_float("BACKWATER_GMSH_INTERFACE_TRANSITION_DIST_FACTOR", 2.5),
            ),
        )
        gmsh_interface_transition_min_ratio = max(
            1.0,
            self._opt_float(
                "gmsh_interface_transition_min_ratio",
                _env_float("BACKWATER_GMSH_INTERFACE_TRANSITION_MIN_RATIO", 1.25),
            ),
        )
        gmsh_transfinite_shared_interface_harmonize = self._opt_bool(
            "gmsh_transfinite_shared_interface_harmonize",
            _env_bool("BACKWATER_GMSH_TRANSFINITE_SHARED_INTERFACE_HARMONIZE", False),
        )
        gmsh_interface_conformance = self._opt_bool(
            "gmsh_interface_conformance",
            _env_bool("BACKWATER_GMSH_INTERFACE_CONFORMANCE", False),
        )
        gmsh_transverse_interface_centroid_merge = self._opt_bool(
            "gmsh_transverse_interface_centroid_merge",
            _env_bool("BACKWATER_GMSH_TRANSVERSE_INTERFACE_CENTROID_MERGE", False),
        )
        gmsh_interface_snap_tol = max(
            1.0e-9,
            self._opt_float(
                "gmsh_interface_snap_tol",
                _env_float("BACKWATER_GMSH_INTERFACE_SNAP_TOL", 1.0),
            ),
        )
        gmsh_interface_reject_near_unshared = self._opt_bool(
            "gmsh_interface_reject_near_unshared",
            _env_bool("BACKWATER_GMSH_INTERFACE_REJECT_NEAR_UNSHARED", True),
        )
        gmsh_interface_reject_tol = max(
            1.0e-9,
            self._opt_float(
                "gmsh_interface_reject_tol",
                _env_float("BACKWATER_GMSH_INTERFACE_REJECT_TOL", 1.0e-3),
            ),
        )
        if gmsh_transverse_interface_centroid_merge:
            gmsh_interface_conformance = True
        gmsh_shared_transverse_edge_count_normalize = True
        gmsh_transfinite_opposite_subset_start = max(
            0.0,
            min(
                1.0,
                self._opt_float(
                    "gmsh_transfinite_opposite_subset_start",
                    _env_float("BACKWATER_GMSH_TRANSFINITE_OPPOSITE_SUBSET_START", 0.30),
                ),
            ),
        )
        gmsh_transfinite_opposite_subset_end = max(
            0.0,
            min(
                1.0,
                self._opt_float(
                    "gmsh_transfinite_opposite_subset_end",
                    _env_float("BACKWATER_GMSH_TRANSFINITE_OPPOSITE_SUBSET_END", 0.70),
                ),
            ),
        )
        gmsh_transfinite_opposite_subset_density_scale = max(
            0.05,
            self._opt_float(
                "gmsh_transfinite_opposite_subset_density_scale",
                _env_float("BACKWATER_GMSH_TRANSFINITE_OPPOSITE_SUBSET_DENSITY_SCALE", 0.50),
            ),
        )
        gmsh_transfinite_interface_debug = self._opt_bool(
            "gmsh_transfinite_interface_debug",
            _env_bool("BACKWATER_GMSH_TRANSFINITE_INTERFACE_DEBUG", False),
        )
        gmsh_transfinite_subset_containment_enable = self._opt_bool(
            "gmsh_transfinite_subset_containment_enable",
            _env_bool("BACKWATER_GMSH_TRANSFINITE_SUBSET_CONTAINMENT_ENABLE", True),
        )
        gmsh_transfinite_subset_containment_high_overlap = max(
            0.50,
            min(
                1.0,
                self._opt_float(
                    "gmsh_transfinite_subset_containment_high_overlap",
                    _env_float("BACKWATER_GMSH_TRANSFINITE_SUBSET_CONTAINMENT_HIGH_OVERLAP", 0.95),
                ),
            ),
        )
        gmsh_transfinite_subset_containment_min_overlap = max(
            0.0,
            min(
                gmsh_transfinite_subset_containment_high_overlap,
                self._opt_float(
                    "gmsh_transfinite_subset_containment_min_overlap",
                    _env_float("BACKWATER_GMSH_TRANSFINITE_SUBSET_CONTAINMENT_MIN_OVERLAP", 0.02),
                ),
            ),
        )
        gmsh_transfinite_subset_containment_max_length_ratio = max(
            1.0e-6,
            self._opt_float(
                "gmsh_transfinite_subset_containment_max_length_ratio",
                _env_float("BACKWATER_GMSH_TRANSFINITE_SUBSET_CONTAINMENT_MAX_LENGTH_RATIO", 0.35),
            ),
        )
        arc_soft_size_factor = min(1.0, max(0.05, self._opt_float("gmsh_arc_soft_size_factor", 0.5)))
        arc_soft_dist_factor = max(0.1, self._opt_float("gmsh_arc_soft_dist_factor", 2.0))

        # Tolerance for point deduplication (scaled to typical hydraulic coords).
        tol = 1e-6
        surface_tags: List[int] = []
        surface_meta: List[Tuple[int, str, float]] = []  # (region_id, cell_type, target_size)
        surface_curve_tags: Dict[int, List[int]] = {}
        surface_quad_controls: Dict[int, Optional[List[QuadEdgeControl]]] = {}
        surface_quad_edge_curve_groups: Dict[int, Optional[List[List[int]]]] = {}
        flow_align_diagnostics: List[Dict[str, object]] = []
        self._last_flow_align_diagnostics = []

        # Shared geometry registries for conforming inter-region interfaces.
        # Points and single-segment lines on shared boundaries are reused so
        # Gmsh meshes that interface curve exactly once.  Without this, each
        # region independently creates duplicate points/curves at the same
        # physical location; Gmsh then discretises the shared edge twice with
        # potentially different node counts, producing hanging nodes that
        # immediately destabilise the FVM solver.
        _pt_prec = 6  # rounding digits ≈ 1 µm — sufficient for hydraulic coords
        pt_reg: Dict[Tuple[float, float], int] = {}   # (rx,ry) -> gmsh point tag
        pt_xy_by_tag: Dict[int, Tuple[float, float]] = {}
        seg_reg: Dict[Tuple[int, int], int] = {}       # (p0,p1) -> signed curve tag
        polycurve_reg: Dict[Tuple[int, ...], int] = {}  # polyline point tag chain -> curve tag
        polycurve_chain_by_tag: Dict[int, Tuple[int, ...]] = {}
        quad_curve_chain_by_abs: Dict[int, Tuple[int, ...]] = {}
        quad_curve_candidates_by_endpoint: Dict[int, List[int]] = {}
        build_order_events: List[str] = []
        build_order_stage_marks: List[Tuple[str, int]] = []
        build_order_event_cap = min(
            200000,
            max(2000, self._opt_int("gmsh_build_order_event_cap", 50000)),
        )
        build_order_overflow = False
        global_option_tokens: List[str] = []

        def _fmt_event_part(value: object) -> str:
            if isinstance(value, float):
                if np.isfinite(float(value)):
                    return f"{float(value):.12g}"
                return "nan"
            if isinstance(value, (list, tuple)):
                return "[" + ",".join(_fmt_event_part(v) for v in value) + "]"
            return str(value)

        def _record_build_event(event: str, *parts: object) -> None:
            nonlocal build_order_overflow
            if len(build_order_events) >= int(build_order_event_cap):
                build_order_overflow = True
                return
            if parts:
                build_order_events.append(
                    str(event) + "|" + "|".join(_fmt_event_part(p) for p in parts)
                )
            else:
                build_order_events.append(str(event))

        def _sha256_lines(lines: Sequence[str]) -> str:
            digest = hashlib.sha256()
            for line in lines:
                digest.update(str(line).encode("utf-8", "replace"))
                digest.update(b"\n")
            return digest.hexdigest()

        def _preview_tokens(lines: Sequence[str], n: int = 12) -> Dict[str, List[str]]:
            items = [str(v) for v in list(lines or [])]
            n_use = max(0, int(n))
            if len(items) <= 2 * n_use:
                return {"head": items, "tail": []}
            return {"head": items[:n_use], "tail": items[-n_use:]}

        def _hash_int_sequence(values: Sequence[int], limit: int = 1024) -> str:
            vals = [int(v) for v in list(values or [])[: max(1, int(limit))]]
            return _sha256_lines([",".join(str(v) for v in vals)])

        def _build_order_fingerprint_payload() -> Dict[str, object]:
            return {
                "sha256": _sha256_lines(build_order_events),
                "event_count": int(len(build_order_events)),
                "event_cap": int(build_order_event_cap),
                "overflow": bool(build_order_overflow),
                "preview": _preview_tokens(build_order_events, n=16),
            }

        def _mark_build_stage(label: str) -> None:
            build_order_stage_marks.append((str(label), int(len(build_order_events))))

        def _build_order_stage_ladder_payload() -> Dict[str, object]:
            stages: List[Dict[str, object]] = []
            prev_idx = 0
            marks = list(build_order_stage_marks)
            if not marks:
                marks = [("full", int(len(build_order_events)))]

            for label, end_idx_raw in marks:
                end_idx = max(prev_idx, min(int(end_idx_raw), int(len(build_order_events))))
                stage_lines = list(build_order_events[prev_idx:end_idx])
                stage_sha = _sha256_lines(stage_lines)
                cumulative_sha = _sha256_lines(build_order_events[:end_idx])
                stages.append({
                    "label": str(label),
                    "start_index": int(prev_idx),
                    "end_index": int(end_idx),
                    "event_count": int(max(0, end_idx - prev_idx)),
                    "stage_sha256": str(stage_sha),
                    "cumulative_sha256": str(cumulative_sha),
                })
                prev_idx = int(end_idx)

            if prev_idx < len(build_order_events):
                end_idx = int(len(build_order_events))
                stage_lines = list(build_order_events[prev_idx:end_idx])
                stage_sha = _sha256_lines(stage_lines)
                cumulative_sha = _sha256_lines(build_order_events[:end_idx])
                stages.append({
                    "label": "tail",
                    "start_index": int(prev_idx),
                    "end_index": int(end_idx),
                    "event_count": int(max(0, end_idx - prev_idx)),
                    "stage_sha256": str(stage_sha),
                    "cumulative_sha256": str(cumulative_sha),
                })

            compact_lines = [
                f"{str(s.get('label', ''))}|{int(s.get('event_count', 0))}|{str(s.get('stage_sha256', ''))}"
                for s in stages
            ]
            return {
                "sha256": _sha256_lines(compact_lines),
                "stage_count": int(len(stages)),
                "stages": stages,
            }

        def _build_order_stage_ladder_compact_text(payload: Dict[str, object]) -> str:
            stages = list(payload.get("stages") or [])
            parts: List[str] = []
            for stage in stages:
                label = str(stage.get("label", ""))
                count = int(stage.get("event_count", 0) or 0)
                sha = str(stage.get("stage_sha256", ""))
                parts.append(f"{label}:{count}:{sha[:12]}")
            txt = ",".join(parts)
            if len(txt) > 420:
                return txt[:417] + "..."
            return txt

        def _global_option_value_text(value: object) -> str:
            return _fmt_event_part(value).replace("\n", "\\n")

        def _record_global_option(name: str, value: object) -> None:
            opt_name = str(name)
            opt_value = _global_option_value_text(value)
            global_option_tokens.append(f"{opt_name}={opt_value}")
            _record_build_event("mesh-option", opt_name, opt_value)

        def _global_options_payload() -> Dict[str, object]:
            entries = [str(v) for v in global_option_tokens]
            return {
                "sha256": _sha256_lines(entries),
                "count": int(len(entries)),
                "entries": entries,
                "preview": _preview_tokens(entries, n=20),
            }

        def _global_options_compact_text(payload: Dict[str, object]) -> str:
            entries = [str(v) for v in list(payload.get("entries") or [])]
            txt = ";".join(entries)
            if len(txt) > 420:
                return txt[:417] + "..."
            return txt

        def _fmt_float_token(value: object, digits: int = 9) -> str:
            try:
                fv = float(value)
            except Exception:
                return "nan"
            if not np.isfinite(fv):
                return "nan"
            rounded = round(float(fv), int(digits))
            return f"{rounded:.{int(digits)}f}"

        def _safe_bbox(dim: int, tag: int) -> Tuple[float, float, float, float, float, float]:
            try:
                bbox = gmsh.model.getBoundingBox(int(dim), int(tag))
                if bbox is None or len(bbox) != 6:
                    raise ValueError("invalid bbox")
                return tuple(float(v) for v in bbox)
            except Exception:
                return (float("nan"),) * 6

        def _entity_tokens_pre_generate() -> Tuple[List[str], List[str], List[str]]:
            point_tokens: List[str] = []
            curve_tokens: List[str] = []
            surface_tokens: List[str] = []

            try:
                point_entities = gmsh.model.getEntities(0)
            except Exception:
                point_entities = []
            point_tags = sorted(
                int(tag)
                for dim, tag in list(point_entities or [])
                if int(dim) == 0
            )
            for ptag in point_tags:
                pxy = pt_xy_by_tag.get(int(ptag))
                if pxy is not None:
                    px = float(pxy[0])
                    py = float(pxy[1])
                else:
                    bb = _safe_bbox(0, int(ptag))
                    px = float(bb[0])
                    py = float(bb[1])
                point_tokens.append(
                    f"{int(ptag)}:{_fmt_float_token(px)}:{_fmt_float_token(py)}"
                )

            try:
                curve_entities = gmsh.model.getEntities(1)
            except Exception:
                curve_entities = []
            curve_tags = sorted(
                int(tag)
                for dim, tag in list(curve_entities or [])
                if int(dim) == 1
            )
            for ctag in curve_tags:
                try:
                    boundary = gmsh.model.getBoundary(
                        [(1, int(ctag))],
                        combined=False,
                        oriented=True,
                        recursive=False,
                    )
                except Exception:
                    boundary = []
                btags = [
                    int(t)
                    for d, t in list(boundary or [])
                    if int(d) == 0
                ]
                bb = _safe_bbox(1, int(ctag))
                bb_tok = ",".join(_fmt_float_token(v, digits=6) for v in bb)
                bnd_tok = ",".join(str(int(v)) for v in btags)
                curve_tokens.append(f"{int(ctag)}:{bnd_tok}:{bb_tok}")

            try:
                surface_entities = gmsh.model.getEntities(2)
            except Exception:
                surface_entities = []
            surface_tags_local = sorted(
                int(tag)
                for dim, tag in list(surface_entities or [])
                if int(dim) == 2
            )
            for stag in surface_tags_local:
                try:
                    boundary = gmsh.model.getBoundary(
                        [(2, int(stag))],
                        combined=False,
                        oriented=True,
                        recursive=False,
                    )
                except Exception:
                    boundary = []
                btags = [
                    int(t)
                    for d, t in list(boundary or [])
                    if int(d) == 1
                ]
                bb = _safe_bbox(2, int(stag))
                bb_tok = ",".join(_fmt_float_token(v, digits=6) for v in bb)
                bnd_tok = ",".join(str(int(v)) for v in btags)
                surface_tokens.append(f"{int(stag)}:{bnd_tok}:{bb_tok}")

            return point_tokens, curve_tokens, surface_tokens

        def _pre_generate_entity_signature_payload() -> Dict[str, object]:
            point_tokens, curve_tokens, surface_tokens = _entity_tokens_pre_generate()
            point_sha = _sha256_lines(point_tokens)
            curve_sha = _sha256_lines(curve_tokens)
            surface_sha = _sha256_lines(surface_tokens)
            all_sha = _sha256_lines([point_sha, curve_sha, surface_sha])
            return {
                "sha256": all_sha,
                "counts": {
                    "points": int(len(point_tokens)),
                    "curves": int(len(curve_tokens)),
                    "surfaces": int(len(surface_tokens)),
                },
                "point_sha256": point_sha,
                "curve_sha256": curve_sha,
                "surface_sha256": surface_sha,
                "point_preview": _preview_tokens(point_tokens, n=10),
                "curve_preview": _preview_tokens(curve_tokens, n=10),
                "surface_preview": _preview_tokens(surface_tokens, n=10),
            }

        def _compact_ptag_chain(ptags: Sequence[int]) -> List[int]:
            tags = [int(t) for t in ptags]
            if not tags:
                return []
            out = [tags[0]]
            for t in tags[1:]:
                if t != out[-1]:
                    out.append(t)
            return out

        def _register_quad_curve_candidate(curve_tag: int, edge_ptags: Sequence[int]) -> None:
            cabs = abs(int(curve_tag))
            if cabs <= 0:
                return
            chain = polycurve_chain_by_tag.get(cabs)
            if not chain:
                compact = _compact_ptag_chain(edge_ptags)
                if len(compact) < 2:
                    return
                chain = tuple(compact if int(curve_tag) > 0 else list(reversed(compact)))
            if len(chain) < 2:
                return
            if cabs in quad_curve_chain_by_abs:
                return
            quad_curve_chain_by_abs[cabs] = tuple(int(t) for t in chain)
            a = int(chain[0])
            b = int(chain[-1])
            quad_curve_candidates_by_endpoint.setdefault(a, []).append(cabs)
            if b != a:
                quad_curve_candidates_by_endpoint.setdefault(b, []).append(cabs)

        def _match_quad_curve_along_ring(
            ptags: Sequence[int],
            start_idx: int,
        ) -> Optional[Tuple[int, int, int, int]]:
            n = len(ptags)
            if n < 2:
                return None
            a = int(ptags[start_idx])
            match_tol = max(1.0e-6, 10.0 * float(tol))
            candidate_abs = list(quad_curve_candidates_by_endpoint.get(a, []))
            if not candidate_abs:
                a_xy = pt_xy_by_tag.get(a)
                if a_xy is not None:
                    for cabs, chain in quad_curve_chain_by_abs.items():
                        if len(chain) < 2:
                            continue
                        p0 = pt_xy_by_tag.get(int(chain[0]))
                        p1 = pt_xy_by_tag.get(int(chain[-1]))
                        if p0 is None or p1 is None:
                            continue
                        d0 = float(np.hypot(float(a_xy[0]) - float(p0[0]), float(a_xy[1]) - float(p0[1])))
                        d1 = float(np.hypot(float(a_xy[0]) - float(p1[0]), float(a_xy[1]) - float(p1[1])))
                        if d0 <= match_tol or d1 <= match_tol:
                            candidate_abs.append(int(cabs))
            if not candidate_abs:
                return None

            best: Optional[Tuple[int, int, float, int, int]] = None
            # (span_edges, signed_curve_tag, proj_err_sum, start_tag, end_tag)
            for cabs in candidate_abs:
                chain = quad_curve_chain_by_abs.get(int(cabs))
                if not chain or len(chain) < 2:
                    continue

                if a == int(chain[0]):
                    target = int(chain[-1])
                    oriented_chain = tuple(int(t) for t in chain)
                    signed = int(cabs)
                elif a == int(chain[-1]):
                    rev = tuple(reversed(chain))
                    oriented_chain = tuple(int(t) for t in rev)
                    target = int(oriented_chain[-1])
                    signed = -int(cabs)
                else:
                    a_xy = pt_xy_by_tag.get(a)
                    p0 = pt_xy_by_tag.get(int(chain[0]))
                    p1 = pt_xy_by_tag.get(int(chain[-1]))
                    if a_xy is None or p0 is None or p1 is None:
                        continue
                    d0 = float(np.hypot(float(a_xy[0]) - float(p0[0]), float(a_xy[1]) - float(p0[1])))
                    d1 = float(np.hypot(float(a_xy[0]) - float(p1[0]), float(a_xy[1]) - float(p1[1])))
                    if d0 <= d1 and d0 <= match_tol:
                        oriented_chain = tuple(int(t) for t in chain)
                        target = int(oriented_chain[-1])
                        signed = int(cabs)
                    elif d1 < d0 and d1 <= match_tol:
                        oriented_chain = tuple(int(t) for t in reversed(chain))
                        target = int(oriented_chain[-1])
                        signed = -int(cabs)
                    else:
                        continue

                idx_map = {int(t): i for i, t in enumerate(oriented_chain)}
                chain_xy: List[Tuple[float, float]] = []
                for t in oriented_chain:
                    xy = pt_xy_by_tag.get(int(t))
                    if xy is None:
                        chain_xy = []
                        break
                    chain_xy.append((float(xy[0]), float(xy[1])))
                if len(chain_xy) < 2:
                    continue
                chain_len = max(_polyline_length(chain_xy), 1.0e-12)
                target_xy = pt_xy_by_tag.get(int(target))

                prev_pos = 0.0
                span_edges = 0
                ok = True
                proj_err_sum = 0.0
                while span_edges < n:
                    p = int(ptags[(start_idx + span_edges + 1) % n])
                    span_edges += 1
                    ci = idx_map.get(p)
                    p_xy = pt_xy_by_tag.get(int(p))
                    if p_xy is None:
                        ok = False
                        break
                    if ci is not None:
                        pos = float(ci)
                        proj_err = 0.0
                    else:
                        proj_err, s_pos = _polyline_distance_and_s(chain_xy, float(p_xy[0]), float(p_xy[1]))
                        if (not np.isfinite(float(proj_err))) or float(proj_err) > match_tol:
                            ok = False
                            break
                        pos = (float(s_pos) / float(chain_len)) * float(max(1, len(oriented_chain) - 1))

                    if pos + 1.0e-8 < prev_pos:
                        ok = False
                        break
                    prev_pos = float(pos)
                    proj_err_sum += float(proj_err)

                    if p == target:
                        break
                    if target_xy is not None and np.hypot(float(p_xy[0]) - float(target_xy[0]), float(p_xy[1]) - float(target_xy[1])) <= match_tol:
                        break

                if not ok:
                    continue
                end_tag = int(ptags[(start_idx + span_edges) % n])
                end_ok = (end_tag == target)
                if not end_ok:
                    end_xy = pt_xy_by_tag.get(int(end_tag))
                    if end_xy is not None and target_xy is not None:
                        end_ok = np.hypot(float(end_xy[0]) - float(target_xy[0]), float(end_xy[1]) - float(target_xy[1])) <= match_tol
                if not end_ok:
                    continue
                if span_edges <= 0:
                    continue

                if best is None:
                    best = (
                        int(span_edges),
                        int(signed),
                        float(proj_err_sum),
                        int(oriented_chain[0]),
                        int(oriented_chain[-1]),
                    )
                else:
                    if int(span_edges) > int(best[0]) or (
                        int(span_edges) == int(best[0]) and float(proj_err_sum) < float(best[2])
                    ):
                        best = (
                            int(span_edges),
                            int(signed),
                            float(proj_err_sum),
                            int(oriented_chain[0]),
                            int(oriented_chain[-1]),
                        )

            if best is None:
                return None
            return int(best[0]), int(best[1]), int(best[3]), int(best[4])

        def _nearest_quad_endpoint_tag(x: float, y: float, snap_tol: float) -> Optional[int]:
            if float(snap_tol) <= 0.0:
                return None
            endpoint_tags = list(quad_curve_candidates_by_endpoint.keys())
            if not endpoint_tags:
                return None
            x0 = float(x)
            y0 = float(y)
            best_tag: Optional[int] = None
            best_d = float(snap_tol)
            for ptag in endpoint_tags:
                pxy = pt_xy_by_tag.get(int(ptag))
                if pxy is None:
                    continue
                d = float(np.hypot(x0 - float(pxy[0]), y0 - float(pxy[1])))
                if d <= best_d:
                    best_d = d
                    best_tag = int(ptag)
            return best_tag

        def _geo_pt(x: float, y: float, lc: float, *, endpoint_snap_tol: Optional[float] = None) -> int:
            """Return existing gmsh point tag at (x,y) or create a new one."""
            key = (round(float(x), _pt_prec), round(float(y), _pt_prec))
            if key in pt_reg:
                tag = int(pt_reg[key])
                pt_xy_by_tag.setdefault(tag, (float(x), float(y)))
                _record_build_event("geo-pt-reuse", key[0], key[1], int(tag))
                return tag

            snap_tol = float(endpoint_snap_tol) if endpoint_snap_tol is not None else 0.0
            if snap_tol > 0.0:
                snap_tag = _nearest_quad_endpoint_tag(float(x), float(y), float(snap_tol))
                if snap_tag is not None:
                    pt_reg[key] = int(snap_tag)
                    _record_build_event(
                        "geo-pt-snap",
                        key[0],
                        key[1],
                        int(snap_tag),
                        float(snap_tol),
                    )
                    return int(snap_tag)

            tag = gmsh.model.geo.addPoint(float(x), float(y), 0.0, lc)
            pt_reg[key] = tag
            pt_xy_by_tag[int(tag)] = (float(x), float(y))
            _record_build_event("geo-pt-new", int(tag), key[0], key[1], float(lc))
            return tag

        def _geo_seg(p0: int, p1: int) -> int:
            """Return signed line tag for directed segment p0->p1, sharing if it
            already exists in either direction."""
            if (p0, p1) in seg_reg:
                tag = int(seg_reg[(p0, p1)])
                polycurve_chain_by_tag.setdefault(abs(tag), (int(p0), int(p1)))
                _record_build_event("geo-seg-reuse-fwd", int(tag), int(p0), int(p1))
                return tag
            if (p1, p0) in seg_reg:
                tag = int(seg_reg[(p1, p0)])
                polycurve_chain_by_tag.setdefault(abs(tag), (int(p1), int(p0)))
                _record_build_event("geo-seg-reuse-rev", int(tag), int(p0), int(p1))
                return -tag
            tag = gmsh.model.geo.addLine(p0, p1)
            seg_reg[(p0, p1)] = tag
            polycurve_chain_by_tag[int(tag)] = (int(p0), int(p1))
            _record_build_event("geo-seg-new", int(tag), int(p0), int(p1))
            return int(tag)

        def _geo_polycurve(ptags: Sequence[int]) -> int:
            """Return a shared directed curve for a polyline point-tag sequence.

            Reuses existing spline/line entities in forward or reversed direction
            so neighboring regions can share exact same interface entities.
            """
            compact = _compact_ptag_chain(ptags)
            if len(compact) < 2:
                raise ValueError("polycurve requires at least two points")

            if len(compact) == 2:
                return _geo_seg(compact[0], compact[1])

            fwd = tuple(compact)
            rev = tuple(reversed(compact))
            if fwd in polycurve_reg:
                tag_reuse = int(polycurve_reg[fwd])
                _record_build_event(
                    "geo-polycurve-reuse-fwd",
                    int(tag_reuse),
                    int(len(fwd)),
                    int(fwd[0]),
                    int(fwd[-1]),
                )
                return int(tag_reuse)
            if rev in polycurve_reg:
                tag_reuse = int(polycurve_reg[rev])
                _record_build_event(
                    "geo-polycurve-reuse-rev",
                    int(tag_reuse),
                    int(len(rev)),
                    int(rev[0]),
                    int(rev[-1]),
                )
                return -int(tag_reuse)

            tag = gmsh.model.geo.addSpline(list(compact))
            polycurve_reg[fwd] = int(tag)
            polycurve_chain_by_tag[int(tag)] = tuple(compact)
            _record_build_event(
                "geo-polycurve-new",
                int(tag),
                int(len(compact)),
                int(compact[0]),
                int(compact[-1]),
                _hash_int_sequence(compact),
            )
            return int(tag)

        prebuild_subphase_started_at = time.perf_counter()

        region_cell_types: Dict[int, str] = {
            int(r.region_id): str(r.default_cell_type).strip().lower()
            for r in model.regions
        }
        region_quad_setups: Dict[int, Tuple[List[Tuple[float, float]], List[QuadEdgeControl]]] = {}
        for region in model.regions:
            ctype_local = str(region.default_cell_type).strip().lower()
            if ctype_local not in {"quadrilateral", "cartesian", "channel_generator"}:
                continue
            quad_setup_local = _quad_controls_for_region(model, region)
            if quad_setup_local is None:
                continue
            region_quad_setups[int(region.region_id)] = quad_setup_local
        _record_phase("prebuild_region_quad_setup", prebuild_subphase_started_at)

        prebuild_subphase_started_at = time.perf_counter()
        region_rings_for_junctions: Dict[int, List[Tuple[float, float]]] = {}
        for region in model.regions:
            rid_local = int(region.region_id)
            setup_local = region_quad_setups.get(rid_local)
            ring_local = list(setup_local[0]) if setup_local is not None else list(region.ring_xy)
            if ring_local and np.hypot(float(ring_local[0][0]) - float(ring_local[-1][0]), float(ring_local[0][1]) - float(ring_local[-1][1])) <= 1.0e-12:
                ring_local = ring_local[:-1]
            if len(ring_local) >= 3:
                region_rings_for_junctions[rid_local] = [(float(x), float(y)) for (x, y) in ring_local]

        transfinite_edge_min_nodes: Dict[Tuple[int, int], int] = {}
        transfinite_harmonize_stats: Dict[str, int] = {
            "shared_groups": 0,
            "canonicalized_edges": 0,
            "opposite_subset_requests": 0,
            "junction_points_inserted": 0,
            "subset_containment_requests": 0,
            "singleton_external_junction_edges": 0,
            "candidate_pair_count_prefilter": 0,
            "candidate_pair_count": 0,
            "pair_bbox_reject_count": 0,
            "nontrans_chain_bbox_reject_count": 0,
            "nontrans_overlap_pair_count": 0,
            "nontrans_point_bbox_reject_count": 0,
        }
        transfinite_harmonize_debug: Dict[str, object] = {}
        if gmsh_transfinite_shared_interface_harmonize and region_quad_setups:
            transfinite_edge_min_nodes, transfinite_harmonize_stats = _harmonize_transfinite_shared_quad_interfaces(
                region_quad_setups=region_quad_setups,
                region_cell_types=region_cell_types,
                gmsh_quad_full_region_flow_align=bool(gmsh_quad_full_region_flow_align),
                all_region_rings=region_rings_for_junctions,
                opposite_subset_start_frac=float(gmsh_transfinite_opposite_subset_start),
                opposite_subset_end_frac=float(gmsh_transfinite_opposite_subset_end),
                opposite_subset_density_scale=float(gmsh_transfinite_opposite_subset_density_scale),
                subset_containment_enable=bool(gmsh_transfinite_subset_containment_enable),
                subset_containment_high_overlap=float(gmsh_transfinite_subset_containment_high_overlap),
                subset_containment_min_overlap=float(gmsh_transfinite_subset_containment_min_overlap),
                subset_containment_max_length_ratio=float(gmsh_transfinite_subset_containment_max_length_ratio),
                debug_capture=transfinite_harmonize_debug if gmsh_transfinite_interface_debug else None,
            )
        _record_phase("prebuild_transfinite_harmonize", prebuild_subphase_started_at)

        def _is_transfinite_region_local(region_id: int) -> bool:
            ctype_local = str(region_cell_types.get(int(region_id), "")).strip().lower()
            if ctype_local == "cartesian":
                return True
            if gmsh_quad_full_region_flow_align and ctype_local in {"quadrilateral", "channel_generator"}:
                return True
            return False

        # Project/split non-transfinite neighboring rings against transfinite
        # interface chains so mixed interfaces can reuse shared geometry.
        prebuild_subphase_started_at = time.perf_counter()
        nontrans_neighbor_projection_rings = 0
        nontrans_chain_bbox_reject_count = 0
        nontrans_overlap_pair_count = 0
        nontrans_point_bbox_reject_count = 0
        transfinite_interface_chains: List[
            Tuple[
                int,
                int,
                List[Tuple[float, float]],
                float,
                Tuple[float, float, float, float],
            ]
        ] = []
        for rid_tf, (_ring_tf, controls_tf) in region_quad_setups.items():
            if not _is_transfinite_region_local(int(rid_tf)):
                continue
            for edge_tf in list(controls_tf or []):
                pts_tf = [(float(x), float(y)) for (x, y) in list(edge_tf.points_xy or [])]
                if len(pts_tf) < 2:
                    continue
                if edge_tf.target_size is not None and float(edge_tf.target_size) > 0.0:
                    size_ref_tf = float(edge_tf.target_size)
                else:
                    size_ref_tf = max(_polyline_length(pts_tf) / max(1, len(pts_tf) - 1), 1.0e-6)
                tx = [float(p[0]) for p in pts_tf]
                ty = [float(p[1]) for p in pts_tf]
                chain_bbox = (float(min(tx)), float(min(ty)), float(max(tx)), float(max(ty)))
                transfinite_interface_chains.append(
                    (
                        int(rid_tf),
                        int(edge_tf.edge_id),
                        pts_tf,
                        float(max(size_ref_tf, 1.0e-6)),
                        chain_bbox,
                    )
                )

        if transfinite_interface_chains:
            for region in model.regions:
                rid_nt = int(region.region_id)
                if _is_transfinite_region_local(rid_nt):
                    continue

                ring_nt = [(float(x), float(y)) for (x, y) in list(region.ring_xy or [])]
                if ring_nt and np.hypot(
                    float(ring_nt[0][0]) - float(ring_nt[-1][0]),
                    float(ring_nt[0][1]) - float(ring_nt[-1][1]),
                ) <= 1.0e-12:
                    ring_nt = ring_nt[:-1]
                if len(ring_nt) < 3:
                    continue

                ring_changed = False
                rx = [float(p[0]) for p in ring_nt]
                ry = [float(p[1]) for p in ring_nt]
                ring_bbox = (float(min(rx)), float(min(ry)), float(max(rx)), float(max(ry)))

                for _owner_tf, _eid_tf, chain_tf, size_ref_tf, chain_bbox in transfinite_interface_chains:
                    size_ref = max(min(float(max(region.default_size, 1.0e-6)), float(size_ref_tf)), 1.0e-6)
                    near_tol = max(1.0e-6, min(8.0, max(0.5, 0.25 * float(size_ref))))
                    sample_step = max(float(near_tol), 0.25 * float(size_ref))

                    if (
                        float(chain_bbox[2]) < float(ring_bbox[0]) - float(near_tol)
                        or float(ring_bbox[2]) < float(chain_bbox[0]) - float(near_tol)
                        or float(chain_bbox[3]) < float(ring_bbox[1]) - float(near_tol)
                        or float(ring_bbox[3]) < float(chain_bbox[1]) - float(near_tol)
                    ):
                        nontrans_chain_bbox_reject_count += 1
                        continue

                    ring_open = list(ring_nt)
                    if ring_open:
                        ring_open.append((float(ring_open[0][0]), float(ring_open[0][1])))

                    nontrans_overlap_pair_count += 1
                    overlap_tf_nt, overlap_nt_tf = _polyline_overlap_fractions_open(
                        chain_tf,
                        ring_open,
                        sample_step=float(sample_step),
                        near_tol=float(near_tol),
                    )
                    if max(float(overlap_tf_nt), float(overlap_nt_tf)) < 0.01:
                        continue

                    ring_split = _insert_focus_points_on_ring_segments(
                        ring=ring_nt,
                        focus_points=chain_tf,
                        max_dist=float(near_tol),
                    )
                    if len(ring_split) < 3:
                        continue

                    chain_len = _polyline_length(chain_tf)
                    if chain_len <= 1.0e-12:
                        continue

                    ring_proj: List[Tuple[float, float]] = []
                    moved_any = False
                    native = _load_hydra_meshing_native()
                    if native is not None and hasattr(native, "project_ring_to_chain"):
                        try:
                            proj_out = native.project_ring_to_chain(
                                ring_split,
                                chain_tf,
                                float(near_tol),
                            )
                            ring_proj = [
                                (float(p[0]), float(p[1]))
                                for p in list(proj_out.get("ring_proj", []))
                                if isinstance(p, (list, tuple)) and len(p) >= 2
                            ]
                            moved_any = bool(proj_out.get("moved_any", False))
                            nontrans_point_bbox_reject_count += int(proj_out.get("point_bbox_reject_count", 0))
                        except Exception:
                            ring_proj = []

                    if not ring_proj:
                        chain_bbox_exp = (
                            float(chain_bbox[0]) - float(near_tol),
                            float(chain_bbox[1]) - float(near_tol),
                            float(chain_bbox[2]) + float(near_tol),
                            float(chain_bbox[3]) + float(near_tol),
                        )
                        for px, py in ring_split:
                            if (
                                float(px) < float(chain_bbox_exp[0])
                                or float(px) > float(chain_bbox_exp[2])
                                or float(py) < float(chain_bbox_exp[1])
                                or float(py) > float(chain_bbox_exp[3])
                            ):
                                nontrans_point_bbox_reject_count += 1
                                ring_proj.append((float(px), float(py)))
                                continue
                            d_loc, s_loc = _polyline_distance_and_s(chain_tf, float(px), float(py))
                            if np.isfinite(float(d_loc)) and float(d_loc) <= float(near_tol):
                                frac_loc = max(0.0, min(1.0, float(s_loc) / float(chain_len)))
                                qx, qy = _interp_polyline_fraction(chain_tf, frac_loc)
                                ring_proj.append((float(qx), float(qy)))
                                if float(np.hypot(float(qx) - float(px), float(qy) - float(py))) > 1.0e-10:
                                    moved_any = True
                            else:
                                ring_proj.append((float(px), float(py)))

                    if len(ring_proj) < 3:
                        continue
                    ring_clean: List[Tuple[float, float]] = []
                    for px, py in ring_proj:
                        if ring_clean and np.hypot(float(px) - float(ring_clean[-1][0]), float(py) - float(ring_clean[-1][1])) <= 1.0e-12:
                            continue
                        ring_clean.append((float(px), float(py)))
                    while len(ring_clean) >= 2 and np.hypot(
                        float(ring_clean[0][0]) - float(ring_clean[-1][0]),
                        float(ring_clean[0][1]) - float(ring_clean[-1][1]),
                    ) <= 1.0e-12:
                        ring_clean.pop()
                    if len(ring_clean) < 3:
                        continue

                    if moved_any or len(ring_clean) > len(ring_nt):
                        ring_nt = ring_clean
                        rx = [float(p[0]) for p in ring_nt]
                        ry = [float(p[1]) for p in ring_nt]
                        ring_bbox = (float(min(rx)), float(min(ry)), float(max(rx)), float(max(ry)))
                        ring_changed = True

                if ring_changed:
                    region.ring_xy = [(float(x), float(y)) for (x, y) in ring_nt]
                    region_rings_for_junctions[rid_nt] = [(float(x), float(y)) for (x, y) in ring_nt]
                    nontrans_neighbor_projection_rings += 1
                    transfinite_harmonize_stats["nontrans_neighbor_projection_rings"] = int(
                        transfinite_harmonize_stats.get("nontrans_neighbor_projection_rings", 0)
                    ) + 1
        transfinite_harmonize_stats["nontrans_chain_bbox_reject_count"] = int(
            transfinite_harmonize_stats.get("nontrans_chain_bbox_reject_count", 0)
        ) + int(nontrans_chain_bbox_reject_count)
        transfinite_harmonize_stats["nontrans_overlap_pair_count"] = int(
            transfinite_harmonize_stats.get("nontrans_overlap_pair_count", 0)
        ) + int(nontrans_overlap_pair_count)
        transfinite_harmonize_stats["nontrans_point_bbox_reject_count"] = int(
            transfinite_harmonize_stats.get("nontrans_point_bbox_reject_count", 0)
        ) + int(nontrans_point_bbox_reject_count)
        _record_phase("prebuild_nontrans_projection", prebuild_subphase_started_at)

        prebuild_subphase_started_at = time.perf_counter()
        interface_coincidence_report: List[Dict[str, object]] = []
        interface_coincidence_suspects: List[Dict[str, object]] = []
        try:
            interface_coincidence_report = _gmsh_interface_coincidence_report(
                model,
                region_quad_setups=region_quad_setups,
            )
            for entry in interface_coincidence_report:
                overlap_delta = float(entry.get("overlap_delta", 0.0))
                endpoint_delta_max = float(entry.get("endpoint_delta_max", float("inf")))
                near_tol = max(float(entry.get("near_tol", 1.0e-6)), 1.0e-9)
                if overlap_delta > 0.20 or endpoint_delta_max > 2.0 * near_tol:
                    interface_coincidence_suspects.append(dict(entry))
            if interface_coincidence_suspects:
                preview = "; ".join(
                    (
                        f"{int(e.get('region_a', -1))}-{int(e.get('region_b', -1))} "
                        f"(overlap_delta={float(e.get('overlap_delta', 0.0)):.3f}, "
                        f"endpoint_delta_max={float(e.get('endpoint_delta_max', float('inf'))):.4g})"
                    )
                    for e in interface_coincidence_suspects[:6]
                )
                warnings.warn(
                    "Gmsh interface coincidence preflight flagged potential geometry mismatches: "
                    + preview,
                    RuntimeWarning,
                )
        except Exception:
            interface_coincidence_report = []
            interface_coincidence_suspects = []
        _record_phase("prebuild_interface_coincidence", prebuild_subphase_started_at)

        _record_phase("prebuild_setup", build_started_at)
        _mark_build_stage("after-prebuild-setup")

        # ---- 1. Build one Gmsh surface per region ----------------------
        phase_started_at = time.perf_counter()
        def _region_priority(r: ConceptualRegion) -> int:
            c = str(r.default_cell_type).strip().lower()
            return 0 if c in {"quadrilateral", "cartesian", "channel_generator"} else 1

        for region in sorted(model.regions, key=_region_priority):
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
            if ctype in ("quadrilateral", "cartesian", "channel_generator"):
                quad_setup = region_quad_setups.get(int(region.region_id))
                if quad_setup is not None:
                    ring, quad_controls = quad_setup

            lines: List[int] = []
            edge_curve_groups: List[List[int]] = []
            if quad_controls is not None:
                first_pt_tag: Optional[int] = None
                first_xy: Optional[Tuple[float, float]] = None
                prev_end_tag: Optional[int] = None
                # Use a slightly looser closure tolerance than point-dedup tol
                # to prevent tiny residual seam segments on assembled quad
                # rings, which can break transfinite opposite-side matching.
                closure_snap_tol = max(float(tol), min(1.0e-3, 0.01 * float(region_size)))
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
                            if np.hypot(x - first_xy[0], y - first_xy[1]) <= float(closure_snap_tol):
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
                        # Build quad interfaces as shared segment chains so
                        # neighboring non-transfinite regions can reuse
                        # interior subsets of the same interface geometry.
                        edge_curves: List[int] = []
                        for k in range(len(edge_tags) - 1):
                            seg = int(_geo_seg(edge_tags[k], edge_tags[k + 1]))
                            edge_curves.append(int(seg))
                            lines.append(int(seg))
                            _register_quad_curve_candidate(int(seg), [edge_tags[k], edge_tags[k + 1]])
                        if edge_curves:
                            edge_curve_groups.append(edge_curves)
                    except Exception:
                        edge_curve_groups.append([])
                        for k in range(len(edge_tags) - 1):
                            lines.append(_geo_seg(edge_tags[k], edge_tags[k + 1]))
                    prev_end_tag = edge_tags[-1]
                if first_pt_tag is not None and prev_end_tag is not None and prev_end_tag != first_pt_tag:
                    p_prev = pt_xy_by_tag.get(int(prev_end_tag))
                    p_first = pt_xy_by_tag.get(int(first_pt_tag))
                    can_snap_close = False
                    if p_prev is not None and p_first is not None:
                        can_snap_close = bool(
                            np.hypot(
                                float(p_prev[0]) - float(p_first[0]),
                                float(p_prev[1]) - float(p_first[1]),
                            ) <= float(closure_snap_tol)
                        )
                    if can_snap_close:
                        prev_end_tag = first_pt_tag
                    else:
                        closing_seg = _geo_seg(prev_end_tag, first_pt_tag)
                        lines.append(closing_seg)
                        if edge_curve_groups:
                            edge_curve_groups[-1].append(int(closing_seg))
            else:
                # Canonicalize near-coincident interface junction points onto
                # previously built quad endpoints to avoid duplicate corner
                # entities across mixed transfinite/non-transfinite neighbors.
                junction_snap_tol = max(20.0 * float(tol), min(1.0e-3, 0.01 * float(region_size)))
                pts = [
                    _geo_pt(x, y, region_size, endpoint_snap_tol=float(junction_snap_tol))
                    for x, y in ring
                ]

                pts_compact: List[int] = []
                for ptag in pts:
                    if not pts_compact or int(ptag) != int(pts_compact[-1]):
                        pts_compact.append(int(ptag))
                while len(pts_compact) >= 2 and int(pts_compact[0]) == int(pts_compact[-1]):
                    pts_compact.pop()
                if len(pts_compact) >= 3:
                    pts = [int(t) for t in pts_compact]

                n_pts = len(pts)
                i = 0
                consumed_edges = 0
                while consumed_edges < n_pts:
                    match = _match_quad_curve_along_ring(pts, i)
                    if match is not None:
                        span, signed_curve, start_tag, end_tag = match
                        if span > 0:
                            # Force exact endpoint tag reuse at matched curve
                            # boundaries so subsequent ring segments stay
                            # topologically connected to the shared chain.
                            if int(pts[i]) != int(start_tag):
                                pts[i] = int(start_tag)
                            end_idx = (i + int(span)) % n_pts
                            if int(pts[end_idx]) != int(end_tag):
                                pts[end_idx] = int(end_tag)
                            lines.append(int(signed_curve))
                            i = int(end_idx)
                            consumed_edges += int(span)
                            continue
                    lines.append(_geo_seg(pts[i], pts[(i + 1) % n_pts]))
                    i = (i + 1) % n_pts
                    consumed_edges += 1

            if len(lines) < 3:
                continue

            loop = gmsh.model.geo.addCurveLoop(lines)
            _record_build_event(
                "geo-loop-new",
                int(loop),
                int(len(lines)),
                _hash_int_sequence(lines),
            )
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
                        hole_loop = gmsh.model.geo.addCurveLoop(hlines)
                        hole_loops.append(hole_loop)
                        _record_build_event(
                            "geo-hole-loop-new",
                            int(hole_loop),
                            int(len(hlines)),
                            _hash_int_sequence(hlines),
                        )
                    except Exception:
                        pass

            surf = gmsh.model.geo.addPlaneSurface([loop] + hole_loops)
            _record_build_event(
                "geo-surface-new",
                int(surf),
                int(region.region_id),
                str(ctype),
                int(loop),
                int(len(hole_loops)),
                int(len(lines)),
            )
            surface_tags.append(surf)
            surface_meta.append((region.region_id, ctype, region_size))
            surface_curve_tags[surf] = lines
            surface_quad_controls[surf] = quad_controls
            surface_quad_edge_curve_groups[surf] = edge_curve_groups if quad_controls is not None else None

        if not surface_tags:
            raise ValueError("GmshBackend: no non-empty regions to mesh.")
        _record_phase("build_surfaces", phase_started_at)
        _mark_build_stage("after-build-surfaces")

        # ---- 2. Embed arc breaklines into surfaces ----------------------
        phase_started_at = time.perf_counter()
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
                _record_build_event(
                    "geo-sync",
                    "arc-hard-embed-start",
                    int(len(arc_curve_tags)),
                    _hash_int_sequence(arc_curve_tags),
                )
                gmsh.model.geo.synchronize()
                for surf in surface_tags:
                    try:
                        gmsh.model.mesh.embed(1, arc_curve_tags, 2, surf)
                        _record_build_event(
                            "mesh-embed-curves",
                            int(surf),
                            int(len(arc_curve_tags)),
                            _hash_int_sequence(arc_curve_tags),
                        )
                    except Exception:
                        pass  # arc may not intersect this surface; skip

        _record_build_event("geo-sync", "post-arc-and-surfaces")
        gmsh.model.geo.synchronize()

        surface_size_map: Dict[int, float] = {int(s): float(sz) for s, (_, _, sz) in zip(surface_tags, surface_meta)}
        surface_ctype_map: Dict[int, str] = {int(s): str(ct) for s, (_, ct, _) in zip(surface_tags, surface_meta)}
        protected_transfinite_surfaces: set = set()
        for surf in surface_tags:
            s = int(surf)
            ctype = str(surface_ctype_map.get(s, "")).strip().lower()
            if ctype == "cartesian":
                protected_transfinite_surfaces.add(s)
            elif gmsh_quad_full_region_flow_align and ctype in {"quadrilateral", "channel_generator"}:
                protected_transfinite_surfaces.add(s)

        interface_transition_specs: List[Dict[str, object]] = []
        if gmsh_interface_transition_enable:
            curve_to_surfaces: Dict[int, List[int]] = {}
            for surf, lines in surface_curve_tags.items():
                s = int(surf)
                for ltag in lines:
                    cabs = abs(int(ltag))
                    if cabs <= 0:
                        continue
                    curve_to_surfaces.setdefault(cabs, []).append(s)

            for cabs, owners in curve_to_surfaces.items():
                uniq = sorted(set(int(v) for v in owners))
                if len(uniq) < 2:
                    continue

                sizes = [float(surface_size_map.get(s, 0.0)) for s in uniq if float(surface_size_map.get(s, 0.0)) > 0.0]
                if len(sizes) < 2:
                    continue
                smin = max(min(sizes), 1.0e-9)
                smax = max(sizes)
                if smax < float(gmsh_interface_transition_min_ratio) * smin:
                    continue

                target_surfaces = [int(s) for s in uniq if int(s) not in protected_transfinite_surfaces]
                if not target_surfaces:
                    continue

                interface_transition_specs.append(
                    {
                        "curve_tag": int(cabs),
                        "owner_surfaces": [int(s) for s in uniq],
                        "target_surfaces": [int(s) for s in target_surfaces],
                        "size_min": float(smin),
                        "size_max": float(smax),
                    }
                )
        _record_phase("embed_arcs_and_interfaces", phase_started_at)
        _mark_build_stage("after-embed-arcs-and-interfaces")

        # ---- 3. Constraint refinement zones (background field) ----------
        phase_started_at = time.perf_counter()
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

        interface_transition_field_count = 0
        if constraint_point_lists or arc_soft_groups or interface_transition_specs:
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

            if interface_transition_specs:
                for spec in interface_transition_specs:
                    curve_tag = int(spec["curve_tag"])
                    target_surfaces = [int(s) for s in spec["target_surfaces"]]
                    size_min_local = max(mesh_size_min, float(spec["size_min"]))
                    size_max_local = max(size_min_local, float(spec["size_max"]))
                    dist_max_local = max(
                        size_min_local,
                        float(gmsh_interface_transition_dist_factor) * size_max_local,
                    )

                    f_dist = gmsh.model.mesh.field.add("Distance")
                    gmsh.model.mesh.field.setNumbers(f_dist, "CurvesList", [int(curve_tag)])

                    f_thresh = gmsh.model.mesh.field.add("Threshold")
                    gmsh.model.mesh.field.setNumber(f_thresh, "InField", float(f_dist))
                    gmsh.model.mesh.field.setNumber(f_thresh, "SizeMin", float(size_min_local))
                    gmsh.model.mesh.field.setNumber(f_thresh, "SizeMax", float(size_max_local))
                    gmsh.model.mesh.field.setNumber(f_thresh, "DistMin", 0.0)
                    gmsh.model.mesh.field.setNumber(f_thresh, "DistMax", float(dist_max_local))
                    gmsh.model.mesh.field.setNumber(f_thresh, "StopAtDistMax", 1.0)

                    f_restrict = gmsh.model.mesh.field.add("Restrict")
                    gmsh.model.mesh.field.setNumber(f_restrict, "InField", float(f_thresh))
                    gmsh.model.mesh.field.setNumbers(f_restrict, "SurfacesList", [int(s) for s in target_surfaces])
                    all_fields.append(f_restrict)
                    interface_transition_field_count += 1

            if len(all_fields) == 1:
                bg_field = all_fields[0]
            else:
                bg_field = gmsh.model.mesh.field.add("Min")
                gmsh.model.mesh.field.setNumbers(bg_field, "FieldsList", [int(fid) for fid in all_fields])

            gmsh.model.mesh.field.setAsBackgroundMesh(int(bg_field))
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0.0)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0.0)
        _record_phase("build_size_fields", phase_started_at)
        _mark_build_stage("after-build-size-fields")

        # ---- 4. Per-surface algorithm and recombination flags ----------
        phase_started_at = time.perf_counter()
        want_recombine = False

        def _oriented_curve_chain(curve_tag_signed: int) -> Optional[Tuple[int, ...]]:
            chain = polycurve_chain_by_tag.get(abs(int(curve_tag_signed)))
            if not chain:
                return None
            if int(curve_tag_signed) < 0:
                return tuple(int(t) for t in reversed(chain))
            return tuple(int(t) for t in chain)

        def _curve_length_from_chain(chain: Sequence[int]) -> float:
            if len(chain) < 2:
                return 0.0
            total = 0.0
            for i in range(len(chain) - 1):
                p0 = pt_xy_by_tag.get(int(chain[i]))
                p1 = pt_xy_by_tag.get(int(chain[i + 1]))
                if p0 is None or p1 is None:
                    continue
                total += float(np.hypot(float(p1[0]) - float(p0[0]), float(p1[1]) - float(p0[1])))
            return float(total)

        def _distribute_divisions(total_div: int, seg_lengths: Sequence[float]) -> List[int]:
            n = int(len(seg_lengths))
            if n <= 0:
                return []
            total_div_local = max(int(total_div), int(n))
            base = [1] * n
            remaining = int(total_div_local - n)
            if remaining <= 0:
                return base

            lengths = [max(float(v), 0.0) for v in seg_lengths]
            lsum = float(sum(lengths))
            if lsum <= 1.0e-12:
                for i in range(remaining):
                    base[i % n] += 1
                return base

            raw = [float(remaining) * (float(v) / lsum) for v in lengths]
            adds = [int(np.floor(v)) for v in raw]
            used = int(sum(adds))
            rem = int(remaining - used)
            if rem > 0:
                frac_rank = sorted(
                    [(i, float(raw[i]) - float(adds[i])) for i in range(n)],
                    key=lambda it: it[1],
                    reverse=True,
                )
                for i in range(rem):
                    adds[int(frac_rank[i % n][0])] += 1
            return [int(base[i] + adds[i]) for i in range(n)]

        def _transfinite_corners_from_edge_groups(edge_curve_groups: Sequence[Sequence[int]]) -> Optional[List[int]]:
            if len(edge_curve_groups) != 4:
                return None
            corners: List[int] = []
            for group in edge_curve_groups:
                if not group:
                    return None
                chain = _oriented_curve_chain(int(group[0]))
                if chain is None or len(chain) < 2:
                    return None
                corners.append(int(chain[0]))
            if len(corners) != 4:
                return None
            return corners

        def _transfinite_corners_from_edge_controls(
            edge_controls: Optional[Sequence[QuadEdgeControl]],
        ) -> Optional[List[int]]:
            if edge_controls is None or len(edge_controls) != 4:
                return None
            corners: List[int] = []
            corner_lc = max(float(tol), 1.0e-9)
            for edge in edge_controls:
                pts = list(edge.points_xy or [])
                if not pts:
                    return None
                x0 = float(pts[0][0])
                y0 = float(pts[0][1])
                tag = _geo_pt(x0, y0, float(corner_lc))
                corners.append(int(tag))
            if len(corners) != 4:
                return None
            if len(set(int(v) for v in corners)) != 4:
                return None
            return corners

        def _apply_flow_aligned_transfinite(
            surf_tag: int,
            curve_tags: Sequence[int],
            edge_controls: Optional[List[QuadEdgeControl]],
            fallback_size: float,
            counts_override: Optional[Sequence[int]] = None,
            min_nodes: Optional[Sequence[int]] = None,
            edge_curve_groups: Optional[Sequence[Sequence[int]]] = None,
        ) -> Tuple[bool, Optional[str]]:
            if edge_controls is None:
                return False, "missing edge controls"
            edge_ids_local = [int(getattr(edge, "edge_id", -1)) for edge in list(edge_controls)]
            counts = list(counts_override) if counts_override is not None else _gmsh_flow_aligned_curve_counts(
                edge_controls,
                fallback_size=fallback_size,
                min_nodes=min_nodes,
            )
            if counts is None:
                return False, "could not compute transfinite counts"
            counts = [int(max(2, int(v))) for v in list(counts)]
            if len(edge_ids_local) == 4 and set(edge_ids_local) == {1, 2, 3, 4}:
                idx_by_edge_local = {int(eid): int(i) for i, eid in enumerate(edge_ids_local)}
                i1 = idx_by_edge_local.get(1)
                i3 = idx_by_edge_local.get(3)
                if i1 is not None and i3 is not None:
                    paired = max(int(counts[i1]), int(counts[i3]))
                    counts[i1] = int(paired)
                    counts[i3] = int(paired)
                i2 = idx_by_edge_local.get(2)
                i4 = idx_by_edge_local.get(4)
                if i2 is not None and i4 is not None:
                    paired = max(int(counts[i2]), int(counts[i4]))
                    counts[i2] = int(paired)
                    counts[i4] = int(paired)
            else:
                counts[0] = counts[2] = max(int(counts[0]), int(counts[2]))
                counts[1] = counts[3] = max(int(counts[1]), int(counts[3]))
            try:
                _record_build_event(
                    "transfinite-apply-start",
                    int(surf_tag),
                    int(len(curve_tags)),
                    int(len(counts)),
                    _hash_int_sequence([int(v) for v in counts]),
                )
                groups = list(edge_curve_groups) if edge_curve_groups is not None else []
                has_group_data = len(groups) == 4 and all(len(g) > 0 for g in groups)
                if has_group_data:
                    # Segmented interfaces require at least one division per
                    # segment. If a side has many segments, raise counts to a
                    # feasible minimum and then re-equalize opposite edges.
                    effective_counts = [int(max(2, int(v))) for v in counts]
                    for ei in range(4):
                        nseg = int(len(groups[ei]))
                        effective_counts[ei] = max(int(effective_counts[ei]), int(nseg + 1))
                    if len(edge_ids_local) == 4 and set(edge_ids_local) == {1, 2, 3, 4}:
                        idx_by_edge_local = {int(eid): int(i) for i, eid in enumerate(edge_ids_local)}
                        i1 = idx_by_edge_local.get(1)
                        i3 = idx_by_edge_local.get(3)
                        if i1 is not None and i3 is not None:
                            paired = max(int(effective_counts[i1]), int(effective_counts[i3]))
                            effective_counts[i1] = int(paired)
                            effective_counts[i3] = int(paired)
                        i2 = idx_by_edge_local.get(2)
                        i4 = idx_by_edge_local.get(4)
                        if i2 is not None and i4 is not None:
                            paired = max(int(effective_counts[i2]), int(effective_counts[i4]))
                            effective_counts[i2] = int(paired)
                            effective_counts[i4] = int(paired)
                    else:
                        pair0 = max(int(effective_counts[0]), int(effective_counts[2]))
                        pair1 = max(int(effective_counts[1]), int(effective_counts[3]))
                        effective_counts[0] = effective_counts[2] = int(pair0)
                        effective_counts[1] = effective_counts[3] = int(pair1)

                    for ei in range(4):
                        group = [int(v) for v in groups[ei]]
                        n_total = max(2, int(effective_counts[ei]))
                        seg_lengths: List[float] = []
                        for ltag in group:
                            chain = _oriented_curve_chain(int(ltag))
                            seg_lengths.append(_curve_length_from_chain(chain or ()))
                        divs = _distribute_divisions(max(1, int(n_total) - 1), seg_lengths)
                        for ltag, div in zip(group, divs):
                            gmsh.model.mesh.setTransfiniteCurve(abs(int(ltag)), int(max(2, int(div) + 1)))
                            _record_build_event(
                                "transfinite-curve",
                                int(abs(int(ltag))),
                                int(max(2, int(div) + 1)),
                            )

                    corners = _transfinite_corners_from_edge_controls(edge_controls)
                    if corners is None:
                        corners = _transfinite_corners_from_edge_groups(groups)
                    if corners is not None and len(corners) == 4:
                        gmsh.model.mesh.setTransfiniteSurface(int(surf_tag), "Left", [int(v) for v in corners])
                        _record_build_event(
                            "transfinite-surface",
                            int(surf_tag),
                            "Left",
                            _hash_int_sequence([int(v) for v in corners]),
                        )
                    else:
                        gmsh.model.mesh.setTransfiniteSurface(int(surf_tag))
                        _record_build_event("transfinite-surface", int(surf_tag), "auto")
                else:
                    if len(curve_tags) != 4:
                        return False, "missing edge group data for non-4-curve surface"
                    for ltag, npt in zip(curve_tags, counts):
                        gmsh.model.mesh.setTransfiniteCurve(abs(int(ltag)), int(npt))
                        _record_build_event(
                            "transfinite-curve",
                            int(abs(int(ltag))),
                            int(npt),
                        )
                    gmsh.model.mesh.setTransfiniteSurface(int(surf_tag))
                    _record_build_event("transfinite-surface", int(surf_tag), "auto")
                return True, None
            except Exception as exc:
                return False, str(exc)

        shared_transverse_count_normalize_diag: Dict[str, object] = {
            "enabled": bool(gmsh_shared_transverse_edge_count_normalize),
            "shared_group_count": 0,
            "affected_surface_count": 0,
        }
        shared_transverse_count_overrides: Dict[int, List[int]] = {}
        if gmsh_shared_transverse_edge_count_normalize and gmsh_quad_full_region_flow_align:
            def _edge_group_key(curve_group: Sequence[int]) -> Tuple[int, ...]:
                return tuple(sorted({abs(int(t)) for t in list(curve_group or []) if int(t) != 0}))

            base_counts_by_surface: Dict[int, List[int]] = {}
            edge_ids_by_surface: Dict[int, List[int]] = {}
            region_id_by_surface: Dict[int, int] = {}
            entries: List[Tuple[int, int, int, Tuple[int, ...]]] = []

            for surf, (rid, ctype, sz) in zip(surface_tags, surface_meta):
                ctype_local = str(ctype).strip().lower()
                if ctype_local not in {"cartesian", "quadrilateral", "channel_generator"}:
                    continue
                edge_controls_local = surface_quad_controls.get(surf)
                groups_local = surface_quad_edge_curve_groups.get(surf)
                if edge_controls_local is None or len(edge_controls_local) != 4:
                    continue
                if groups_local is None or len(groups_local) != 4:
                    continue

                min_nodes_local = [
                    int(transfinite_edge_min_nodes.get((int(rid), int(edge.edge_id)), 0))
                    for edge in edge_controls_local
                ]
                min_nodes_local_use: Optional[List[int]] = None
                if any(int(v) > 0 for v in min_nodes_local):
                    min_nodes_local_use = [int(v) for v in min_nodes_local]

                counts_local = _gmsh_flow_aligned_curve_counts(
                    edge_controls_local,
                    fallback_size=float(sz),
                    min_nodes=min_nodes_local_use,
                )
                if counts_local is None:
                    continue

                s = int(surf)
                base_counts_by_surface[s] = [int(v) for v in counts_local]
                edge_ids_local = [int(getattr(edge, "edge_id", -1)) for edge in edge_controls_local]
                edge_ids_by_surface[s] = edge_ids_local
                region_id_by_surface[s] = int(rid)

                for idx, curve_group in enumerate(groups_local):
                    if idx >= len(edge_ids_local):
                        continue
                    edge_id_local = int(edge_ids_local[idx])
                    if edge_id_local not in {1, 2, 3, 4}:
                        continue
                    key = _edge_group_key(curve_group)
                    if not key:
                        continue
                    entries.append((s, int(idx), int(edge_id_local), key))

            owners_by_key: Dict[Tuple[int, ...], set] = {}
            for s, _idx, _eid, key in entries:
                owners_by_key.setdefault(key, set()).add(int(s))

            target_by_key: Dict[Tuple[int, ...], int] = {}
            for key, owners in owners_by_key.items():
                if len(owners) < 2:
                    continue
                vals: List[int] = []
                for s, idx, _eid, key_local in entries:
                    if key_local != key:
                        continue
                    if int(s) not in owners:
                        continue
                    vals.append(int(base_counts_by_surface.get(int(s), [0, 0, 0, 0])[int(idx)]))
                if vals:
                    target_by_key[key] = int(max(vals))

            preview: List[Dict[str, object]] = []
            entries_by_surface: Dict[int, List[Tuple[int, int, int, Tuple[int, ...]]]] = {}
            for rec in entries:
                s, _idx, _eid, key = rec
                if key not in target_by_key:
                    continue
                entries_by_surface.setdefault(int(s), []).append(rec)

            for s, recs in entries_by_surface.items():
                base_counts = list(base_counts_by_surface.get(int(s), []))
                edge_ids_local = list(edge_ids_by_surface.get(int(s), []))
                if len(base_counts) != 4 or len(edge_ids_local) != 4:
                    continue

                counts_new = [int(v) for v in base_counts]
                changed = False
                for _s, idx, edge_id_local, key in recs:
                    target_nodes = int(target_by_key.get(key, 0))
                    if target_nodes <= 0:
                        continue
                    if int(target_nodes) > int(counts_new[idx]):
                        counts_new[idx] = int(target_nodes)
                        changed = True
                    if len(preview) < 12:
                        preview.append(
                            {
                                "surface_tag": int(s),
                                "region_id": int(region_id_by_surface.get(int(s), -1)),
                                "edge_id": int(edge_id_local),
                                "base_nodes": int(base_counts[idx]),
                                "target_nodes": int(target_nodes),
                                "shared_curve_count": int(len(key)),
                            }
                        )

                if not changed:
                    continue

                if len(edge_ids_local) == 4 and set(edge_ids_local) == {1, 2, 3, 4}:
                    idx_by_edge_local = {int(eid): int(i) for i, eid in enumerate(edge_ids_local)}
                    i1 = idx_by_edge_local.get(1)
                    i3 = idx_by_edge_local.get(3)
                    if i1 is not None and i3 is not None:
                        paired = max(int(counts_new[i1]), int(counts_new[i3]))
                        counts_new[i1] = int(paired)
                        counts_new[i3] = int(paired)
                    i2 = idx_by_edge_local.get(2)
                    i4 = idx_by_edge_local.get(4)
                    if i2 is not None and i4 is not None:
                        paired = max(int(counts_new[i2]), int(counts_new[i4]))
                        counts_new[i2] = int(paired)
                        counts_new[i4] = int(paired)
                else:
                    counts_new[0] = counts_new[2] = max(int(counts_new[0]), int(counts_new[2]))
                    counts_new[1] = counts_new[3] = max(int(counts_new[1]), int(counts_new[3]))

                shared_transverse_count_overrides[int(s)] = [int(v) for v in counts_new]

            shared_transverse_count_normalize_diag["shared_group_count"] = int(len(target_by_key))
            shared_transverse_count_normalize_diag["affected_surface_count"] = int(len(shared_transverse_count_overrides))
            if preview:
                shared_transverse_count_normalize_diag["preview"] = list(preview)

        for surf, (rid, ctype, sz) in zip(surface_tags, surface_meta):
            region = next((r for r in model.regions if int(r.region_id) == int(rid)), None)
            lines = surface_curve_tags.get(surf, [])
            quad_controls = surface_quad_controls.get(surf)
            edge_curve_groups = surface_quad_edge_curve_groups.get(surf)
            edge_min_nodes: Optional[List[int]] = None
            if quad_controls is not None and len(quad_controls) == 4:
                min_nodes_local = [
                    int(transfinite_edge_min_nodes.get((int(rid), int(edge.edge_id)), 0))
                    for edge in quad_controls
                ]
                if any(int(v) > 0 for v in min_nodes_local):
                    edge_min_nodes = min_nodes_local
            flow_aligned_applied = False
            flow_align_preflight_fallback = False
            if gmsh_quad_full_region_flow_align and ctype in {"cartesian", "quadrilateral", "channel_generator"}:
                diag = _gmsh_flow_align_region_preflight(
                    region_id=int(rid),
                    cell_type=str(ctype),
                    curve_tags=lines,
                    edge_controls=quad_controls,
                    fallback_size=float(sz),
                    min_nodes=edge_min_nodes,
                )
                diag["surface_tag"] = int(surf)
                diag["requested"] = True
                normalized_counts_override = shared_transverse_count_overrides.get(int(surf))
                if normalized_counts_override is not None:
                    diag["shared_transverse_count_normalized"] = True
                    diag["transfinite_counts_normalized"] = [int(v) for v in list(normalized_counts_override)]
                else:
                    diag["shared_transverse_count_normalized"] = False
                counts_for_apply = normalized_counts_override
                if counts_for_apply is None:
                    counts_for_apply = diag.get("transfinite_counts")
                if bool(diag.get("eligible", False)):
                    ok, err = _apply_flow_aligned_transfinite(
                        surf_tag=int(surf),
                        curve_tags=lines,
                        edge_controls=quad_controls,
                        fallback_size=float(sz),
                        counts_override=counts_for_apply,
                        min_nodes=edge_min_nodes,
                        edge_curve_groups=edge_curve_groups,
                    )
                    flow_aligned_applied = bool(ok)
                    if ok:
                        diag["status"] = "applied"
                        diag["fallback"] = False
                    else:
                        diag["status"] = "fallback"
                        diag["fallback"] = True
                        flow_align_preflight_fallback = True
                        diag["reasons"] = list(diag.get("reasons", [])) + [
                            "gmsh-transfinite-apply-failed"
                        ]
                        if err:
                            diag["apply_error"] = str(err)
                        warnings.warn(
                            "Gmsh flow-align fallback for region "
                            f"{int(rid)}: transfinite apply failed ({err}).",
                            RuntimeWarning,
                        )
                else:
                    diag["status"] = "fallback"
                    flow_align_preflight_fallback = True
                    reason_txt = ", ".join(str(x) for x in diag.get("reasons", []) if str(x)) or "unknown"
                    warnings.warn(
                        "Gmsh flow-align fallback for region "
                        f"{int(rid)}: {reason_txt}",
                        RuntimeWarning,
                    )
                if flow_align_preflight_fallback:
                    diag["transfinite_skipped_after_fallback"] = True
                flow_align_diagnostics.append(diag)
                self._last_flow_align_diagnostics = list(flow_align_diagnostics)
            if ctype == "cartesian":
                # Transfinite + Recombine: structured, fast, pure quads.
                if (
                    (not flow_align_preflight_fallback)
                    and
                    (not flow_aligned_applied)
                    and region is not None
                    and region.edge_lengths
                    and len(region.edge_lengths) == 4
                    and quad_controls is not None
                    and len(quad_controls) == 4
                ):
                    try:
                        edge_geom_len = []
                        edge_geom_len = [_polyline_length(edge.points_xy) for edge in quad_controls]
                        counts = []
                        for i in range(4):
                            tlen = max(float(region.edge_lengths[i]), tol)
                            ndiv = max(1, int(round(edge_geom_len[i] / tlen)))
                            counts.append(max(2, ndiv + 1))

                        if edge_min_nodes is not None and len(edge_min_nodes) == 4:
                            counts = [max(int(c), int(mn)) if int(mn) > 0 else int(c) for c, mn in zip(counts, edge_min_nodes)]

                        # Opposite edges must match for transfinite surface.
                        n0 = max(counts[0], counts[2])
                        n1 = max(counts[1], counts[3])
                        counts[0] = counts[2] = n0
                        counts[1] = counts[3] = n1
                        ok_tf, _err_tf = _apply_flow_aligned_transfinite(
                            surf_tag=int(surf),
                            curve_tags=lines,
                            edge_controls=quad_controls,
                            fallback_size=float(sz),
                            counts_override=counts,
                            min_nodes=edge_min_nodes,
                            edge_curve_groups=edge_curve_groups,
                        )
                        if not ok_tf:
                            gmsh.model.mesh.setTransfiniteSurface(surf)
                    except Exception:
                        try:
                            gmsh.model.mesh.setTransfiniteSurface(surf)
                        except Exception:
                            pass
                elif (not flow_align_preflight_fallback) and (not flow_aligned_applied):
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
                if (
                    (not flow_align_preflight_fallback)
                    and
                    (not flow_aligned_applied)
                    and region is not None
                    and region.edge_lengths
                    and len(region.edge_lengths) == 4
                    and quad_controls is not None
                    and len(quad_controls) == 4
                ):
                    try:
                        edge_geom_len = []
                        edge_geom_len = [_polyline_length(edge.points_xy) for edge in quad_controls]
                        counts = []
                        for i in range(4):
                            tlen = max(float(region.edge_lengths[i]), tol)
                            ndiv = max(1, int(round(edge_geom_len[i] / tlen)))
                            counts.append(max(2, ndiv + 1))
                        if edge_min_nodes is not None and len(edge_min_nodes) == 4:
                            counts = [max(int(c), int(mn)) if int(mn) > 0 else int(c) for c, mn in zip(counts, edge_min_nodes)]
                        n0 = max(counts[0], counts[2])
                        n1 = max(counts[1], counts[3])
                        counts[0] = counts[2] = n0
                        counts[1] = counts[3] = n1
                        _apply_flow_aligned_transfinite(
                            surf_tag=int(surf),
                            curve_tags=lines,
                            edge_controls=quad_controls,
                            fallback_size=float(sz),
                            counts_override=counts,
                            min_nodes=edge_min_nodes,
                            edge_curve_groups=edge_curve_groups,
                        )
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
        _record_phase("configure_per_surface", phase_started_at)
        _mark_build_stage("after-configure-per-surface")

        # ---- 5. Global mesh options ------------------------------------
        phase_started_at = time.perf_counter()
        def _set_global_mesh_option(name: str, value: object) -> None:
            numeric_value = float(value)
            gmsh.option.setNumber(str(name), float(numeric_value))
            _record_global_option(str(name), float(numeric_value))

        _set_global_mesh_option("Mesh.RecombineAll", 0.0)  # per-surface only
        _set_global_mesh_option("Mesh.RecombinationAlgorithm", float(recomb_algo))
        _set_global_mesh_option("Mesh.RecombineOptimizeTopology", float(max(0, int(recombine_optimize_topology))))
        _set_global_mesh_option("Mesh.RecombineNodeRepositioning", 1.0 if recombine_node_repositioning else 0.0)
        _set_global_mesh_option("Mesh.RecombineMinimumQuality", max(0.0, float(recombine_minimum_quality)))
        _set_global_mesh_option("Mesh.Smoothing", float(smoothing_passes))
        _set_global_mesh_option("Mesh.OptimizeNetgen", 1.0 if optimize_netgen else 0.0)
        _set_global_mesh_option("Mesh.AlgorithmSwitchOnFailure", 1.0 if algorithm_switch_on_failure else 0.0)
        _set_global_mesh_option("Mesh.RandomFactor", max(0.0, float(random_factor)))
        _set_global_mesh_option("Mesh.MeshSizeMin", float(mesh_size_min))
        _set_global_mesh_option("Mesh.ToleranceEdgeLength", float(tolerance_edge_length))
        _set_global_mesh_option("Mesh.MeshSizeFromPoints", 1.0 if mesh_size_from_points else 0.0)
        _set_global_mesh_option("General.NumThreads", float(gmsh_num_threads))
        _set_global_mesh_option("Mesh.MaxNumThreads2D", float(gmsh_max_num_threads_2d))
        _record_build_event(
            "mesh-options-summary",
            int(tri_algo),
            int(quad_algo),
            int(recomb_algo),
            int(smoothing_passes),
            int(optimize_iters),
            float(mesh_size_min),
            float(tolerance_edge_length),
            bool(mesh_size_from_points),
            int(gmsh_num_threads),
            int(gmsh_max_num_threads_2d),
        )
        _record_phase("configure_global_options", phase_started_at)
        _mark_build_stage("after-configure-global-options")

        # ---- 6. Generate -----------------------------------------------
        phase_started_at = time.perf_counter()
        gmsh_build_order_fingerprint = _build_order_fingerprint_payload()
        gmsh_build_order_stage_ladder = _build_order_stage_ladder_payload()
        gmsh_global_options = _global_options_payload()
        gmsh_pre_generate_entity_signature = _pre_generate_entity_signature_payload()
        self._last_build_order_fingerprint = dict(gmsh_build_order_fingerprint)
        self._last_build_order_stage_ladder = dict(gmsh_build_order_stage_ladder)
        self._last_pre_generate_entity_signature = dict(gmsh_pre_generate_entity_signature)
        try:
            gmsh.model.mesh.generate(2)
        except Exception as exc:
            raise RuntimeError(
                "Gmsh mesh.generate(2) failed "
                f"(build_order_sha256={gmsh_build_order_fingerprint.get('sha256', '')}, "
                f"build_stage_ladder_sha256={gmsh_build_order_stage_ladder.get('sha256', '')}, "
                f"build_stage_ladder={_build_order_stage_ladder_compact_text(gmsh_build_order_stage_ladder)}, "
                f"global_options_sha256={gmsh_global_options.get('sha256', '')}, "
                f"global_options={_global_options_compact_text(gmsh_global_options)}, "
                f"entity_sha256={gmsh_pre_generate_entity_signature.get('sha256', '')}, "
                f"entity_counts={gmsh_pre_generate_entity_signature.get('counts', {})}): {exc}"
            ) from exc
        if want_recombine and bool(gmsh_global_recombine):
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
        _record_phase("generate_and_optimize", phase_started_at)

        phase_started_at = time.perf_counter()
        duplicate_cleanup_summary: Optional[Dict[str, object]] = None
        duplicate_before_count = 0
        duplicate_after_count = 0
        duplicate_cleanup_ran = False
        try:
            dup_before = gmsh.model.mesh.getDuplicateNodes([])
        except TypeError:
            dup_before = gmsh.model.mesh.getDuplicateNodes()
        except Exception:
            dup_before = []
        if dup_before is None:
            dup_before = []
        duplicate_before_count = int(len(dup_before))
        if duplicate_before_count > 0:
            duplicate_cleanup_ran = True
            warnings.warn(
                "Gmsh mesh duplicate-node cleanup triggered "
                f"(duplicates={duplicate_before_count}).",
                RuntimeWarning,
            )
            try:
                gmsh.model.mesh.removeDuplicateNodes([])
            except TypeError:
                gmsh.model.mesh.removeDuplicateNodes()
            except Exception:
                pass
            try:
                gmsh.model.mesh.removeDuplicateElements([])
            except TypeError:
                gmsh.model.mesh.removeDuplicateElements()
            except Exception:
                pass
            try:
                dup_after = gmsh.model.mesh.getDuplicateNodes([])
            except TypeError:
                dup_after = gmsh.model.mesh.getDuplicateNodes()
            except Exception:
                dup_after = []
            if dup_after is None:
                dup_after = []
            duplicate_after_count = int(len(dup_after))
            if duplicate_after_count > 0:
                warnings.warn(
                    "Gmsh duplicate-node cleanup completed with remaining duplicates "
                    f"(remaining={duplicate_after_count}).",
                    RuntimeWarning,
                )
            duplicate_cleanup_summary = {
                "duplicate_nodes_before": int(duplicate_before_count),
                "duplicate_nodes_after": int(duplicate_after_count),
                "cleanup_ran": bool(duplicate_cleanup_ran),
            }
        _record_phase("duplicate_cleanup", phase_started_at)

        # ---- 7. Extract nodes ------------------------------------------
        phase_started_at = time.perf_counter()
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        # node_coords: flat [x0,y0,z0, x1,y1,z1, ...]
        node_coords = np.array(node_coords, dtype=np.float64).reshape(-1, 3)
        tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}
        node_x = node_coords[:, 0].copy()
        node_y = node_coords[:, 1].copy()
        node_z = np.zeros(node_x.shape[0], dtype=np.float64)
        _record_phase("extract_nodes", phase_started_at)

        # ---- 8. Extract elements per surface with metadata -------------
        phase_started_at = time.perf_counter()
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
            self._last_flow_align_diagnostics = list(flow_align_diagnostics)
            if flow_align_diagnostics:
                diag_parts: List[str] = []
                for d in flow_align_diagnostics:
                    rid_txt = str(d.get("region_id", "?"))
                    status_txt = str(d.get("status", "unknown"))
                    reasons_txt = ",".join(str(x) for x in d.get("reasons", []) if str(x))
                    if not reasons_txt:
                        reasons_txt = "none"
                    diag_parts.append(
                        f"region={rid_txt};status={status_txt};reasons={reasons_txt}"
                    )
                raise ValueError(
                    "GmshBackend: no elements extracted from mesh. "
                    "Flow-align per-region diagnostics: " + " | ".join(diag_parts)
                )
            raise ValueError("GmshBackend: no elements extracted from mesh.")
        _record_phase("extract_elements", phase_started_at)
        gmsh_phase_timings_s["total_build"] = float(max(0.0, time.perf_counter() - build_started_at))

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
        if bool(gmsh_interface_conformance):
            out = _enforce_quad_interface_conformance(
                out,
                model,
                snap_tol=float(gmsh_interface_snap_tol),
                centroid_merge=bool(gmsh_transverse_interface_centroid_merge),
            )
        out.quality_summary = dict(out.quality_summary or {})
        if bool(gmsh_interface_reject_near_unshared):
            near_unshared_report = _mixed_transfinite_tri_near_unshared_report(
                out,
                tol=float(gmsh_interface_reject_tol),
            )
            out.quality_summary["gmsh_interface_near_unshared_check"] = dict(near_unshared_report)
            flagged_pair_count = int(near_unshared_report.get("flagged_pair_count", 0) or 0)
            if flagged_pair_count > 0:
                flagged_pairs = list(near_unshared_report.get("flagged_pairs") or [])
                preview = "; ".join(
                    (
                        f"{int(p.get('region_pair', [-1, -1])[0])}-{int(p.get('region_pair', [-1, -1])[1])} "
                        f"(near_only={int(p.get('near_only_a', 0))}/{int(p.get('near_only_b', 0))}, "
                        f"shared_edges={int(p.get('shared_edge_count', 0))})"
                    )
                    for p in flagged_pairs[:6]
                )
                raise ValueError(
                    "Gmsh mixed transfinite/tri interface check failed: detected "
                    f"{flagged_pair_count} region pair(s) with near-coincident unshared nodes "
                    f"(tol={float(gmsh_interface_reject_tol):.6g}). "
                    + (f"Examples: {preview}" if preview else "")
                )
        out.quality_summary["gmsh_phase_timings_s"] = dict(gmsh_phase_timings_s)
        out.quality_summary["gmsh_build_order_fingerprint"] = dict(gmsh_build_order_fingerprint)
        out.quality_summary["gmsh_build_order_stage_ladder"] = dict(gmsh_build_order_stage_ladder)
        out.quality_summary["gmsh_global_options"] = dict(gmsh_global_options)
        out.quality_summary["gmsh_pre_generate_entity_signature"] = dict(gmsh_pre_generate_entity_signature)
        out.quality_summary["gmsh_phase_counts"] = {
            "surface_count": int(len(surface_tags)),
            "constraint_count": int(len(constraint_point_lists)),
            "arc_count": int(len(model.arcs)),
            "face_count": int(max(0, len(all_face_offsets) - 1)),
        }
        meshing_native = _load_hydra_meshing_native()
        out.quality_summary["gmsh_cpp_prebuild_native"] = {
            "enabled": bool(_gmsh_cpp_prebuild_enabled()),
            "module_loaded": bool(meshing_native is not None),
            "has_interface_overlap_metrics_closed": bool(
                meshing_native is not None and hasattr(meshing_native, "interface_overlap_metrics_closed")
            ),
            "has_polyline_overlap_fractions_open": bool(
                meshing_native is not None and hasattr(meshing_native, "polyline_overlap_fractions_open")
            ),
            "has_project_ring_to_chain": bool(
                meshing_native is not None and hasattr(meshing_native, "project_ring_to_chain")
            ),
        }
        has_transfinite_harmonize_diag = any(int(v) > 0 for v in transfinite_harmonize_stats.values())
        has_transfinite_harmonize_debug = bool(transfinite_harmonize_debug)
        has_interface_coincidence_diag = bool(interface_coincidence_report)
        has_shared_transverse_count_normalize_diag = bool(gmsh_shared_transverse_edge_count_normalize)
        if flow_align_diagnostics or duplicate_cleanup_summary is not None or interface_transition_specs or has_transfinite_harmonize_diag or has_transfinite_harmonize_debug or has_interface_coincidence_diag or has_shared_transverse_count_normalize_diag:
            out.quality_summary["gmsh_flow_align_diagnostics"] = list(flow_align_diagnostics)
            if duplicate_cleanup_summary is not None:
                out.quality_summary["gmsh_duplicate_cleanup"] = dict(duplicate_cleanup_summary)
            if interface_transition_specs:
                out.quality_summary["gmsh_interface_transition"] = {
                    "enabled": bool(gmsh_interface_transition_enable),
                    "protected_transfinite_surfaces": sorted(int(s) for s in protected_transfinite_surfaces),
                    "spec_count": int(len(interface_transition_specs)),
                    "field_count": int(interface_transition_field_count),
                }
            if has_shared_transverse_count_normalize_diag:
                out.quality_summary["gmsh_shared_transverse_edge_count_normalize"] = dict(shared_transverse_count_normalize_diag)
            if has_transfinite_harmonize_diag:
                out.quality_summary["gmsh_transfinite_interface_harmonize"] = {
                    "enabled": bool(gmsh_transfinite_shared_interface_harmonize),
                    "subset_start": float(gmsh_transfinite_opposite_subset_start),
                    "subset_end": float(gmsh_transfinite_opposite_subset_end),
                    "subset_density_scale": float(gmsh_transfinite_opposite_subset_density_scale),
                    "subset_containment_enable": bool(gmsh_transfinite_subset_containment_enable),
                    "subset_containment_high_overlap": float(gmsh_transfinite_subset_containment_high_overlap),
                    "subset_containment_min_overlap": float(gmsh_transfinite_subset_containment_min_overlap),
                    "subset_containment_max_length_ratio": float(gmsh_transfinite_subset_containment_max_length_ratio),
                    "shared_groups": int(transfinite_harmonize_stats.get("shared_groups", 0)),
                    "canonicalized_edges": int(transfinite_harmonize_stats.get("canonicalized_edges", 0)),
                    "opposite_subset_densified": int(transfinite_harmonize_stats.get("opposite_subset_requests", 0)),
                    "junction_points_inserted": int(transfinite_harmonize_stats.get("junction_points_inserted", 0)),
                    "subset_containment_densified": int(transfinite_harmonize_stats.get("subset_containment_requests", 0)),
                    "singleton_external_junction_edges": int(transfinite_harmonize_stats.get("singleton_external_junction_edges", 0)),
                    "nontrans_neighbor_projection_rings": int(transfinite_harmonize_stats.get("nontrans_neighbor_projection_rings", 0)),
                    "candidate_pair_count_prefilter": int(transfinite_harmonize_stats.get("candidate_pair_count_prefilter", 0)),
                    "candidate_pair_count": int(transfinite_harmonize_stats.get("candidate_pair_count", 0)),
                    "pair_bbox_reject_count": int(transfinite_harmonize_stats.get("pair_bbox_reject_count", 0)),
                    "nontrans_chain_bbox_reject_count": int(transfinite_harmonize_stats.get("nontrans_chain_bbox_reject_count", 0)),
                    "nontrans_overlap_pair_count": int(transfinite_harmonize_stats.get("nontrans_overlap_pair_count", 0)),
                    "nontrans_point_bbox_reject_count": int(transfinite_harmonize_stats.get("nontrans_point_bbox_reject_count", 0)),
                }
            if has_transfinite_harmonize_debug:
                out.quality_summary["gmsh_transfinite_interface_debug"] = dict(transfinite_harmonize_debug)
            if has_interface_coincidence_diag:
                out.quality_summary["gmsh_interface_coincidence_report"] = {
                    "pair_count": int(len(interface_coincidence_report)),
                    "suspect_pair_count": int(len(interface_coincidence_suspects)),
                    "pairs": list(interface_coincidence_report),
                }
            self._last_flow_align_diagnostics = list(flow_align_diagnostics)
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

    def _quad_region_method(self) -> str:
        raw = self._options.get(
            "tqmesh_quad_region_method",
            os.environ.get("BACKWATER_TQMESH_QUAD_REGION_METHOD", "auto"),
        )
        text = str(raw).strip().lower()
        if text in {"qgis_structured", "structured", "structured_full_region", "full_region"}:
            return "qgis_structured"
        if text in {"tqmesh_native_recipe", "native", "recipe", "quad_recipe"}:
            return "tqmesh_native_recipe"
        return "auto"

    def _use_structured_quad_region_method(self) -> bool:
        method = self._quad_region_method()
        if method == "qgis_structured":
            return True
        if method == "tqmesh_native_recipe":
            return False
        return self._opt_bool(
            "tqmesh_quad_full_region_flow_align",
            _env_bool("BACKWATER_TQMESH_QUAD_FULL_REGION_FLOW_ALIGN", True),
        )

    @staticmethod
    def _is_ccw(ring: List[Tuple[float, float]]) -> bool:
        """Return True if the ring has counter-clockwise winding (positive area)."""
        area = _polygon_area_xy(
            np.asarray([p[0] for p in ring]),
            np.asarray([p[1] for p in ring]),
        )
        return area > 0.0

    def _try_generate_native_merged_mesh(self, _tq, model: ConceptualModel, progress_emit=None) -> Optional[MeshResult]:
        if len(model.regions) <= 1:
            return None

        def _emit(stage: str, region_id: Optional[int] = None, detail: str = "", force: bool = False) -> None:
            if progress_emit is None:
                return
            try:
                progress_emit(stage=stage, region_id=region_id, detail=detail, force=force)
            except Exception:
                pass

        use_structured_quad_region = self._use_structured_quad_region_method()
        breakline_fixed_edges_enabled = self._opt_bool(
            "tqmesh_breakline_fixed_edges",
            _env_bool("BACKWATER_TQMESH_BREAKLINE_FIXED_EDGES", True),
        )
        boundary_split_max_length = _as_float(
            self._options.get("tqmesh_boundary_split_max_length"),
            _env_float("BACKWATER_TQMESH_BOUNDARY_SPLIT_MAX_LENGTH", 0.0),
        )
        if (not np.isfinite(boundary_split_max_length)) or boundary_split_max_length <= 0.0:
            boundary_split_max_length = 0.0

        mesh_specs: List[Dict[str, object]] = []
        region_size_by_id: Dict[int, float] = {}

        _emit("native-merge-prepare", detail=f"regions={len(model.regions)}", force=True)
        for region_index, region in enumerate(model.regions, start=1):
            ring = list(region.ring_xy)
            if ring and ring[0] == ring[-1]:
                ring = ring[:-1]
            if len(ring) < 3:
                continue

            ctype = str(region.default_cell_type).strip().lower()
            if ctype == "empty":
                continue

            _emit(
                "native-merge-region",
                region_id=int(region.region_id),
                detail=f"{region_index}/{len(model.regions)} cell_type={ctype}",
            )

            target_size = max(float(region.default_size), 1.0e-10)
            region_constraints = _constraints_for_region(model, ring)
            region_exclusions = _region_exclusion_zones(model, region, ring)
            fixed_edge_lines = _breakline_fixed_edges_for_region(model, region, ring)
            if fixed_edge_lines:
                ring, fixed_edge_lines = _snap_and_split_boundary_for_breaklines(
                    ring,
                    fixed_edge_lines,
                    vertex_snap_tol=0.1,
                )
            if boundary_split_max_length > 0.0 and fixed_edge_lines:
                split_lines: List[List[Tuple[float, float]]] = []
                for line in fixed_edge_lines:
                    densified = _split_polyline_max_segment_length(line, boundary_split_max_length)
                    if len(densified) >= 2:
                        split_lines.append(densified)
                fixed_edge_lines = split_lines

            quad_controls = None
            quad_boundary = None
            if ctype in ("quadrilateral", "cartesian"):
                quad_setup = self._quad_controls_for_region(model, region)
                if quad_setup is not None:
                    quad_boundary, quad_controls = quad_setup

            # The structured full-region flow-aligned branch is still handled by
            # the legacy Python path.
            if quad_controls is not None and use_structured_quad_region:
                return None

            ext_verts = list(quad_boundary) if quad_boundary is not None else ring
            protected_boundary_points = _boundary_contact_vertices(
                ext_verts,
                fixed_edge_lines,
                tol=max(1.0e-6, 1.0e-3 * float(target_size)),
            )
            ext_verts = _sanitize_closed_ring(
                ext_verts,
                length_tol=max(1.0e-6, 1.0e-3 * float(target_size)),
                collinear_tol=max(1.0e-6, 1.5e-3 * float(target_size)),
                protect_points=protected_boundary_points,
                protect_tol=max(1.0e-6, 1.0e-3 * float(target_size)),
            )
            ext_verts = _stitch_boundary_microchains(
                ext_verts,
                target_size=float(target_size),
                protect_points=protected_boundary_points,
                protect_tol=max(1.0e-6, 1.0e-3 * float(target_size)),
            )
            ext_verts, used_fallback_boundary = _recover_tqmesh_exterior_boundary(
                ext_verts,
                fallback_ring=ring,
                target_size=float(target_size),
                protect_points=protected_boundary_points,
                protect_tol=max(1.0e-6, 1.0e-3 * float(target_size)),
                region_id=int(region.region_id),
            )
            if used_fallback_boundary:
                quad_controls = None
                quad_boundary = None
            if len(ext_verts) < 3:
                continue
            if not self._is_ccw(ext_verts):
                ext_verts = list(reversed(ext_verts))
            ext_is_ccw = self._is_ccw(ext_verts)

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

            constraint_verts_list: List[List[Tuple[float, float]]] = []
            constraint_sizes_list: List[float] = []
            for cst in region_constraints:
                if len(cst.ring_xy) < 3 or str(cst.cell_type).strip().lower() == "empty":
                    continue
                constraint_verts_list.append([(float(x), float(y)) for (x, y) in cst.ring_xy])
                constraint_sizes_list.append(float(cst.target_size))

            active_quad_layers: List[List[float]] = []
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
                if len(active_quad_layers) >= 4:
                    active_quad_layers = []

            want_quads = ctype in ("quadrilateral", "cartesian")
            quad_refinements = _as_int(
                self._options.get("tqmesh_quad_refinements"),
                _as_int(os.environ.get("BACKWATER_TQMESH_QUAD_REFINEMENTS", 0), 0),
            )
            if quad_refinements < 0:
                quad_refinements = 0
            fixed_for_spec = fixed_edge_lines if breakline_fixed_edges_enabled else []
            smooth_for_spec = 0 if len(fixed_for_spec) > 0 else 3
            region_id = int(region.region_id)

            mesh_specs.append(
                {
                    "ext_verts": [[float(v[0]), float(v[1])] for v in ext_verts],
                    "ext_colors": [1] * len(ext_verts),
                    "int_boundaries": int_boundaries,
                    "int_colors": int_colors,
                    "constraint_verts": [
                        [[float(v[0]), float(v[1])] for v in cverts]
                        for cverts in constraint_verts_list
                    ],
                    "constraint_sizes": [float(cs) for cs in constraint_sizes_list],
                    "fixed_edges": [
                        [[float(x), float(y)] for (x, y) in line]
                        for line in fixed_for_spec
                    ],
                    "target_size": float(target_size),
                    "quad_layers": active_quad_layers,
                    "tri_to_quad": bool(want_quads),
                    "quad_refinements": int(quad_refinements if want_quads else 0),
                    "n_smooth": int(smooth_for_spec),
                    "boundary_split_max_length": float(boundary_split_max_length),
                    "mesh_id": int(region_id),
                    "element_color": int(region_id),
                }
            )
            region_size_by_id[region_id] = float(target_size)

        if len(mesh_specs) <= 1:
            return None

        _emit("native-merge-run", detail=f"mesh_specs={len(mesh_specs)}", force=True)

        merged = None
        merge_errors: List[str] = []
        for receiver_index in range(len(mesh_specs)):
            try:
                merged = _tq.generate_merged_triangular_meshes(
                    mesh_specs=mesh_specs,
                    receiver_index=int(receiver_index),
                    tri_to_quad=False,
                    n_smooth=0,
                    boundary_split_max_length=float(boundary_split_max_length),
                    post_merge_smooth=0,
                )
                break
            except Exception as exc:
                merge_errors.append(f"receiver_index={receiver_index}: {exc}")

        if merged is None:
            _emit("native-merge-fail", detail="all receiver indices failed", force=True)
            raise RuntimeError(
                "Native TQMesh merge failed for all receiver indices: "
                + " | ".join(merge_errors)
            )

        node_x = np.asarray(merged["verts_x"], dtype=np.float64)
        node_y = np.asarray(merged["verts_y"], dtype=np.float64)
        tris_arr = np.asarray(merged["triangles"], dtype=np.int32)
        quads_arr = np.asarray(merged["quads"], dtype=np.int32)
        tri_colors = np.asarray(merged.get("tri_colors", []), dtype=np.int32)
        quad_colors = np.asarray(merged.get("quad_colors", []), dtype=np.int32)

        if tris_arr.size:
            tris_arr = tris_arr.reshape((-1, 3))
        else:
            tris_arr = np.empty((0, 3), dtype=np.int32)
        if quads_arr.size:
            quads_arr = quads_arr.reshape((-1, 4))
        else:
            quads_arr = np.empty((0, 4), dtype=np.int32)

        if tri_colors.size != tris_arr.shape[0]:
            tri_colors = np.full((tris_arr.shape[0],), 0, dtype=np.int32)
        if quad_colors.size != quads_arr.shape[0]:
            quad_colors = np.full((quads_arr.shape[0],), 0, dtype=np.int32)

        face_nodes_list: List[int] = []
        face_offsets: List[int] = [0]
        plot_tris: List[int] = []

        for tri in tris_arr:
            face_nodes_list.extend(tri.tolist())
            face_offsets.append(len(face_nodes_list))
            plot_tris.extend(tri.tolist())

        for quad in quads_arr:
            face_nodes_list.extend(quad.tolist())
            face_offsets.append(len(face_nodes_list))
            plot_tris.extend([quad[0], quad[1], quad[2], quad[0], quad[2], quad[3]])

        all_cell_types = ["triangular"] * int(tris_arr.shape[0]) + ["quadrilateral"] * int(quads_arr.shape[0])
        all_region_ids = tri_colors.tolist() + quad_colors.tolist()
        all_target_sizes = [
            float(region_size_by_id.get(int(rid), 0.0))
            for rid in all_region_ids
        ]

        out = MeshResult(
            node_x=node_x,
            node_y=node_y,
            node_z=np.zeros(node_x.size, dtype=np.float64),
            cell_nodes=np.asarray(plot_tris, dtype=np.int32),
            cell_face_offsets=np.asarray(face_offsets, dtype=np.int32),
            cell_face_nodes=np.asarray(face_nodes_list, dtype=np.int32),
            cell_type=np.asarray(all_cell_types, dtype=object),
            region_id=np.asarray(all_region_ids, dtype=np.int32),
            target_size=np.asarray(all_target_sizes, dtype=np.float64),
            quality_summary={
                "backend": "tqmesh_native_merge",
                "merged_mesh_count": int(len(mesh_specs)),
            },
        )
        _emit(
            "native-merge-done",
            detail=(
                f"nodes={int(node_x.size)} triangles={int(tris_arr.shape[0])} "
                f"quads={int(quads_arr.shape[0])}"
            ),
            force=True,
        )
        return _repair_mesh_result(out)

    def generate(self, model: ConceptualModel) -> MeshResult:
        try:
            import hydra_tqmesh as _tq
        except ImportError:
            _tq = None

        if _tq is None or not hasattr(_tq, "generate_triangular_mesh"):
            try:
                import importlib.util
                from pathlib import Path

                root = Path(__file__).resolve().parents[2]
                build_dir = root / "build"
                cand = sorted(build_dir.glob("hydra_tqmesh*.so"))
                if not cand:
                    raise FileNotFoundError("hydra_tqmesh*.so not found under build/")
                spec = importlib.util.spec_from_file_location("hydra_tqmesh", str(cand[0]))
                if spec is None or spec.loader is None:
                    raise RuntimeError("could not create module spec for hydra_tqmesh")
                _tq = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(_tq)
            except Exception as load_exc:
                raise RuntimeError(
                    "hydra_tqmesh C++ module not found.  "
                    "Rebuild the plugin (cmake + make) to compile TQMesh bindings."
                ) from load_exc
        elif not hasattr(_tq, "generate_merged_triangular_meshes"):
            # Prefer a freshly built module if the imported one is stale.
            try:
                import importlib.util
                from pathlib import Path

                root = Path(__file__).resolve().parents[2]
                build_dir = root / "build"
                cand = sorted(build_dir.glob("hydra_tqmesh*.so"))
                if cand:
                    spec = importlib.util.spec_from_file_location("hydra_tqmesh", str(cand[0]))
                    if spec is not None and spec.loader is not None:
                        fresh = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(fresh)
                        if hasattr(fresh, "generate_merged_triangular_meshes"):
                            _tq = fresh
            except Exception:
                pass

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

        progress_path = str(self._options.get("tqmesh_progress_path", "") or "").strip()
        progress_emit_interval_s = _as_float(
            self._options.get("tqmesh_progress_emit_interval_s"),
            0.75,
        )
        if (not np.isfinite(progress_emit_interval_s)) or progress_emit_interval_s <= 0.0:
            progress_emit_interval_s = 0.75
        progress_emit_interval_s = max(float(progress_emit_interval_s), 0.2)
        progress_seq = 0
        progress_last_emit = -1.0

        def _clip_detail(detail: object, max_len: int = 220) -> str:
            txt = str(detail or "").strip()
            if len(txt) <= max_len:
                return txt
            return txt[: max_len - 3] + "..."

        def _emit_progress(
            stage: str,
            region_id: Optional[int] = None,
            attempt: Optional[int] = None,
            detail: str = "",
            force: bool = False,
        ) -> None:
            nonlocal progress_seq, progress_last_emit
            if not progress_path:
                return
            now = time.perf_counter()
            if (not force) and progress_last_emit >= 0.0:
                if (now - progress_last_emit) < progress_emit_interval_s:
                    return
            progress_seq += 1
            payload: Dict[str, object] = {
                "seq": int(progress_seq),
                "stage": str(stage),
                "timestamp": float(time.time()),
                "elapsed_s": float(max(0.0, now - t_start)),
                "regions_total": int(len(model.regions)),
            }
            if region_id is not None:
                payload["region_id"] = int(region_id)
            if attempt is not None:
                payload["attempt"] = int(attempt)
            clipped = _clip_detail(detail)
            if clipped:
                payload["detail"] = clipped
            try:
                _write_json_atomic(progress_path, payload)
                progress_last_emit = now
            except Exception:
                pass

        t_start = time.perf_counter()
        _emit_progress("start", detail=f"regions={len(model.regions)}", force=True)

        native_merge_enabled = self._opt_bool(
            "tqmesh_native_merge",
            _env_bool("BACKWATER_TQMESH_NATIVE_MERGE", True),
        )
        native_merge_strict = self._opt_bool(
            "tqmesh_native_merge_strict",
            _env_bool("BACKWATER_TQMESH_NATIVE_MERGE_STRICT", False),
        )
        native_merge_available = hasattr(_tq, "generate_merged_triangular_meshes")
        if native_merge_enabled and native_merge_available:
            try:
                _emit_progress("native-merge-start", force=True)
                native_merged = self._try_generate_native_merged_mesh(
                    _tq,
                    model,
                    progress_emit=_emit_progress,
                )
                if native_merged is not None:
                    _emit_progress(
                        "done",
                        detail=(
                            f"native-merge nodes={int(native_merged.node_x.size)} "
                            f"faces={max(0, int(native_merged.cell_face_offsets.size) - 1)}"
                        ),
                        force=True,
                    )
                    return native_merged
            except Exception as exc:
                if native_merge_strict:
                    _emit_progress("native-merge-fail", detail=_clip_detail(exc), force=True)
                    raise RuntimeError(
                        f"TQMesh native merge path failed in strict mode: {exc}"
                    ) from exc
                _emit_progress("native-merge-fallback", detail=_clip_detail(exc), force=True)
                warnings.warn(
                    f"TQMesh native merge path failed; falling back to legacy region weld path: {exc}",
                    RuntimeWarning,
                )
        elif native_merge_enabled and (not native_merge_available):
            _emit_progress("native-merge-unavailable", force=True)
            warnings.warn(
                "TQMesh native merge requested but this hydra_tqmesh module does not expose "
                "generate_merged_triangular_meshes; using legacy region weld path.",
                RuntimeWarning,
            )

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
        all_tri_rid: List[int] = []
        all_quad_rid: List[int] = []
        all_tri_size: List[float] = []
        all_quad_size: List[float] = []

        for region_index, region in enumerate(model.regions, start=1):
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

            _emit_progress(
                "region-start",
                region_id=int(region.region_id),
                detail=f"{region_index}/{len(model.regions)} cell_type={ctype}",
                force=True,
            )

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

            use_structured_quad_region = self._use_structured_quad_region_method()
            quad_full_region_max_cells = _as_int(
                self._options.get("tqmesh_quad_full_region_max_cells"),
                _as_int(os.environ.get("BACKWATER_TQMESH_QUAD_FULL_REGION_MAX_CELLS", 250000), 250000),
            )
            if quad_full_region_max_cells <= 0:
                quad_full_region_max_cells = 0
            if quad_controls is not None and use_structured_quad_region:
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
                            all_quad_rid.append(int(region.region_id))
                            all_quad_size.append(float(target_size))
                        elif len(poly) == 3:
                            all_tris.extend(poly)
                            all_tri_rid.append(int(region.region_id))
                            all_tri_size.append(float(target_size))
                    _emit_progress(
                        "region-done",
                        region_id=int(region.region_id),
                        detail="structured-quad-region",
                        force=True,
                    )
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

            ext_verts, used_fallback_boundary = _recover_tqmesh_exterior_boundary(
                ext_verts,
                fallback_ring=ring,
                target_size=float(target_size),
                protect_points=protected_boundary_points,
                protect_tol=max(1.0e-6, 1.0e-3 * float(target_size)),
                region_id=int(region.region_id),
            )
            if used_fallback_boundary:
                quad_controls = None
                quad_boundary = None
            ext_verts_post_resample_count = len(ext_verts)

            if len(ext_verts) < 3:
                raise RuntimeError(
                    f"TQMesh region {region.region_id}: boundary recovery produced fewer than 3 vertices"
                )
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

            fixed_edge_variants: List[
                Tuple[str, List[List[List[float]]], List[List[List[float]]], List[float]]
            ] = []
            if len(base_args["fixed_edges"]) > 0:
                fixed_edge_variants.append(("with-fixed-edges", base_args["fixed_edges"], [], []))
                if not breakline_fixed_edges_strict:
                    relaxed_breakline_fallback = self._opt_bool(
                        "tqmesh_breakline_relaxed_fallback",
                        _env_bool("BACKWATER_TQMESH_BREAKLINE_RELAXED_FALLBACK", True),
                    )
                    if relaxed_breakline_fallback:
                        relax_simplify_tol_factor = _as_float(
                            self._options.get("tqmesh_breakline_relax_simplify_tol_factor"),
                            _env_float("BACKWATER_TQMESH_BREAKLINE_RELAX_SIMPLIFY_TOL_FACTOR", 0.35),
                        )
                        relax_hint_size_factor = _as_float(
                            self._options.get("tqmesh_breakline_relax_hint_size_factor"),
                            _env_float("BACKWATER_TQMESH_BREAKLINE_RELAX_HINT_SIZE_FACTOR", 0.65),
                        )
                        relax_hint_spacing_factor = _as_float(
                            self._options.get("tqmesh_breakline_relax_hint_spacing_factor"),
                            _env_float("BACKWATER_TQMESH_BREAKLINE_RELAX_HINT_SPACING_FACTOR", 1.25),
                        )
                        relax_hint_box_half_factor = _as_float(
                            self._options.get("tqmesh_breakline_relax_hint_box_half_factor"),
                            _env_float("BACKWATER_TQMESH_BREAKLINE_RELAX_HINT_BOX_HALF_FACTOR", 0.45),
                        )
                        relax_max_hint_boxes = _as_int(
                            self._options.get("tqmesh_breakline_relax_max_hint_boxes"),
                            _as_int(os.environ.get("BACKWATER_TQMESH_BREAKLINE_RELAX_MAX_HINT_BOXES", 256), 256),
                        )

                        relaxed_lines_xy, relaxed_hint_xy, relaxed_hint_sizes = _relax_fixed_edges_and_hints(
                            [
                                [(float(x), float(y)) for x, y in line]
                                for line in base_args["fixed_edges"]
                            ],
                            target_size=float(target_size),
                            simplify_tol_factor=float(relax_simplify_tol_factor),
                            hint_size_factor=float(relax_hint_size_factor),
                            hint_spacing_factor=float(relax_hint_spacing_factor),
                            hint_box_half_factor=float(relax_hint_box_half_factor),
                            max_hint_boxes=int(relax_max_hint_boxes),
                        )

                        relaxed_edges = [
                            [[float(x), float(y)] for (x, y) in line]
                            for line in relaxed_lines_xy
                            if len(line) >= 2
                        ]
                        relaxed_hint_verts = [
                            [[float(x), float(y)] for (x, y) in ring]
                            for ring in relaxed_hint_xy
                            if len(ring) >= 3
                        ]

                        if relaxed_edges and relaxed_edges != base_args["fixed_edges"]:
                            fixed_edge_variants.append(
                                (
                                    "with-relaxed-fixed-edges",
                                    relaxed_edges,
                                    relaxed_hint_verts,
                                    [float(v) for v in relaxed_hint_sizes],
                                )
                            )
                        if relaxed_hint_verts:
                            fixed_edge_variants.append(
                                (
                                    "soft-breakline-hints",
                                    [],
                                    relaxed_hint_verts,
                                    [float(v) for v in relaxed_hint_sizes],
                                )
                            )

                    fixed_edge_variants.append(("no-fixed-edges", [], [], []))
            else:
                fixed_edge_variants.append(("no-fixed-edges", [], [], []))

            want_quads = ctype in ("quadrilateral", "cartesian")
            quad_refinements_requested = _as_int(
                self._options.get("tqmesh_quad_refinements"),
                _as_int(os.environ.get("BACKWATER_TQMESH_QUAD_REFINEMENTS", 0), 0),
            )
            if quad_refinements_requested < 0:
                quad_refinements_requested = 0
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
            attempt_counter = 0
            for fixed_label, fixed_edges_try, extra_constraint_verts, extra_constraint_sizes in fixed_edge_variants:
                constraint_verts_try = list(base_args["constraint_verts"]) + list(extra_constraint_verts)
                constraint_sizes_try_base = list(constraint_sizes_list) + [float(v) for v in extra_constraint_sizes]
                for label, quad_layers_try, tri_to_quad_try, n_smooth_try in attempts:
                    for size_scale in quality_cfg.size_scales:
                        target_try = max(target_size * max(float(size_scale), 1e-6), 1e-10)
                        csz_try = [
                            max(float(cs) * max(float(size_scale), 1e-6), 1e-10)
                            for cs in constraint_sizes_try_base
                        ]
                        for ds in quality_cfg.smooth_increments:
                            smooth_try = max(0, int(n_smooth_try) + int(ds))
                            refine_try = int(quad_refinements_requested if tri_to_quad_try else 0)
                            cfg_key = (
                                fixed_label,
                                label,
                                tuple(tuple(q) for q in quad_layers_try),
                                bool(tri_to_quad_try),
                                int(refine_try),
                                int(smooth_try),
                                float(round(target_try, 12)),
                            )
                            if cfg_key in seen_cfg:
                                continue
                            seen_cfg.add(cfg_key)
                            attempt_counter += 1
                            _emit_progress(
                                "attempt",
                                region_id=int(region.region_id),
                                attempt=int(attempt_counter),
                                detail=(
                                    f"{fixed_label}/{label} size={target_try:.4g} "
                                    f"smooth={smooth_try} tri_to_quad={int(bool(tri_to_quad_try))}"
                                ),
                            )

                            try:
                                candidate = _tq.generate_triangular_mesh(
                                    ext_verts=base_args["ext_verts"],
                                    ext_colors=base_args["ext_colors"],
                                    int_boundaries=base_args["int_boundaries"],
                                    int_colors=base_args["int_colors"],
                                    constraint_verts=constraint_verts_try,
                                    constraint_sizes=csz_try,
                                    fixed_edges=fixed_edges_try,
                                    target_size=target_try,
                                    quad_layers=quad_layers_try,
                                    tri_to_quad=tri_to_quad_try,
                                    quad_refinements=refine_try,
                                    n_smooth=smooth_try,
                                    boundary_split_max_length=float(boundary_split_for_call),
                                )
                            except Exception as exc:
                                exc_txt = str(exc)
                                _emit_progress(
                                    "attempt-exception",
                                    region_id=int(region.region_id),
                                    attempt=int(attempt_counter),
                                    detail=_clip_detail(exc_txt),
                                    force=True,
                                )
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
                                                constraint_verts=constraint_verts_try,
                                                constraint_sizes=csz_try,
                                                fixed_edges=fixed_edges_try,
                                                target_size=target_try,
                                                quad_layers=quad_layers_try,
                                                tri_to_quad=tri_to_quad_try,
                                                quad_refinements=refine_try,
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
                _emit_progress(
                    "region-fail",
                    region_id=int(region.region_id),
                    detail=(
                        f"attempts={attempt_counter} errors={len(errors)} "
                        f"cell_type={ctype}"
                    ),
                    force=True,
                )
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

            _emit_progress(
                "region-done",
                region_id=int(region.region_id),
                detail=(
                    f"attempts={attempt_counter} used={used_label} "
                    f"errors={len(errors)}"
                ),
                force=True,
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
                all_tri_rid.extend([int(region.region_id)] * int(tris.shape[0]))
                all_tri_size.extend([float(target_size)] * int(tris.shape[0]))

            if quads.size > 0:
                all_quads.extend((quads.ravel() + offset).tolist())
                all_quad_rid.extend([int(region.region_id)] * int(quads.shape[0]))
                all_quad_size.extend([float(target_size)] * int(quads.shape[0]))

            all_bv0.extend((bv0 + offset).tolist())
            all_bv1.extend((bv1 + offset).tolist())
            all_bc.extend(bc.tolist())

        if not all_vx:
            _emit_progress("fail", detail="generated no vertices", force=True)
            raise ValueError("TQMesh generated no vertices.")
        if not all_tris and not all_quads:
            _emit_progress("fail", detail="generated no cells", force=True)
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

        face_cell_type = np.asarray(
            ["triangular"] * len(all_tri_rid) + ["quadrilateral"] * len(all_quad_rid),
            dtype=object,
        )
        face_region_id = np.asarray(all_tri_rid + all_quad_rid, dtype=np.int32)
        face_target_size = np.asarray(all_tri_size + all_quad_size, dtype=np.float64)
        n_faces = int(len(face_offsets) - 1)
        if face_cell_type.size != n_faces or face_region_id.size != n_faces or face_target_size.size != n_faces:
            raise RuntimeError(
                "TQMesh assembly metadata mismatch: "
                f"faces={n_faces}, cell_type={face_cell_type.size}, "
                f"region_id={face_region_id.size}, target_size={face_target_size.size}"
            )

        out = MeshResult(
            node_x=node_x,
            node_y=node_y,
            node_z=node_z,
            cell_nodes=np.asarray(plot_tris, dtype=np.int32),
            cell_face_offsets=np.asarray(face_offsets, dtype=np.int32),
            cell_face_nodes=np.asarray(face_nodes_list, dtype=np.int32),
            cell_type=face_cell_type,
            region_id=face_region_id,
            target_size=face_target_size,
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
        _emit_progress(
            "done",
            detail=(
                f"nodes={int(out.node_x.size)} faces={max(0, int(out.cell_face_offsets.size) - 1)} "
                f"triangles={int(len(all_tri_rid))} quads={int(len(all_quad_rid))}"
            ),
            force=True,
        )
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
