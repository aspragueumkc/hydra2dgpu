#!/usr/bin/env python3
"""Face-centric meshing utilities for SWE2D.

Topology-first meshing pipeline
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
from typing import Any, Dict, List, Optional, Sequence, Tuple
import json
import logging
import os
import re
import sys
import time
import warnings

import numpy as np

from swe2d.mesh.mesh_models import (
    MeshingBackend,
    CellConstraint,
    ConceptualArc,
    ConceptualModel,
    ConceptualNode,
    ConceptualRegion,
    MeshResult,
    QuadEdgeControl,
    _CELL_TYPES,
    _GmshQualityConfig,
)
from swe2d.mesh.mesh_quality import (
    _face_mesh_quality_stats,
    _gmsh_quality_passes,
    _gmsh_quality_score,
    _mesh_quality_stats,
    _polygon_area_xy,
)

logger = logging.getLogger(__name__)


def __getattr__(name: str):
    """Lazy re-export of GmshBackend to avoid circular import with gmsh_backend."""
    if name == "GmshBackend":
        from swe2d.mesh.gmsh_backend import GmshBackend as _gb
        return _gb
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    """require nonempty mesh"""
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

    For adjacent, non-overlapping regions with shared boundaries, the merged solver
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
    """as float"""
    try:
        if v is None:
            return float(default)
        if isinstance(v, float):
            return v
        if isinstance(v, int):
            return float(v)
        s = str(v).strip()
        if s.upper() in ("NULL", "NONE", "N/A", ""):
            return float(default)
        return float(s)
    except Exception as e:
        logger.warning("_as_float conversion failed: %s", e, exc_info=True)
        return float(default)


def _as_int(v, default: int) -> int:
    """as int"""
    try:
        if v is None:
            return int(default)
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            return int(v)
        s = str(v).strip()
        if s.upper() in ("NULL", "NONE", "N/A", ""):
            return int(default)
        return int(s)
    except Exception as e:
        logger.warning("_as_int conversion failed: %s", e, exc_info=True)
        return int(default)


def _normalize_cell_type(v: str, default: str = "triangular") -> str:
    """normalize cell type"""
    s = str(v or "").strip().lower()
    if s not in _CELL_TYPES:
        return default
    return s


def _env_float(name: str, default: float) -> float:
    """env float"""
    try:
        return float(os.environ.get(name, default))
    except Exception as e:
        logger.warning("_env_float failed for %s: %s", name, e, exc_info=True)
        return float(default)


def _env_bool(name: str, default: bool) -> bool:
    """env bool"""
    raw = str(os.environ.get(name, "1" if default else "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


_HYDRA_MESHING_NATIVE_MODULE = None
_HYDRA_MESHING_NATIVE_LOAD_ATTEMPTED = False


def _gmsh_cpp_prebuild_enabled() -> bool:
    """gmsh cpp prebuild enabled"""
    return _env_bool("BACKWATER_GMSH_CPP_PREBUILD", True)


def _load_hydra_meshing_native():
    """load hydra meshing native"""
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
    except Exception as e:
        logger.warning("hydra_meshing_native direct import failed: %s", e, exc_info=True)
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
    except Exception as e:
        logger.warning("hydra_meshing_native fallback import failed: %s", e, exc_info=True)
        _HYDRA_MESHING_NATIVE_MODULE = None
        return None


def _env_csv_floats(name: str, default: Sequence[float]) -> Tuple[float, ...]:
    """env csv floats"""
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
        except Exception as e:
            logger.warning("_env_csv_floats token parse failed: %s", e, exc_info=True)
            continue
    if not vals:
        return tuple(float(v) for v in default)
    return tuple(vals)


def _env_csv_strings(name: str, default: Sequence[str]) -> Tuple[str, ...]:
    """env csv strings"""
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
    """serialize xy points"""
    return [[float(x), float(y)] for x, y in points]


def _serialize_xy_lines(lines: Sequence[Sequence[Tuple[float, float]]]) -> List[List[List[float]]]:
    """serialize xy lines"""
    return [_serialize_xy_points(line) for line in lines]


def _point_in_polygon(x: float, y: float, ring: Sequence[Tuple[float, float]]) -> bool:
    """point in polygon"""
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
    """bbox from ring"""
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return min(xs), min(ys), max(xs), max(ys)


def _ring_centroid_xy(ring: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    """ring centroid xy"""
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
        """ring xy"""
        ring = [(float(p.x()), float(p.y())) for p in points[:-1]]
        if len(ring) >= 3 and ring[0] == ring[-1]:
            ring = ring[:-1]
        return ring

    try:
        is_multi = bool(getattr(geom, "isMultipart", lambda: True)())
        if is_multi:
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
    except Exception as e:
        logger.debug("_iter_qgis_polygon_parts geom.asMultiPolygon failed: %s", e, exc_info=True)
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
    except Exception as e:
        logger.warning("_iter_qgis_polygon_parts geom.asPolygon failed: %s", e, exc_info=True)
        pass

    return parts


def _iter_qgis_polygon_outer_rings(geom) -> List[List[Tuple[float, float]]]:
    """Extract outer rings from QGIS Polygon or MultiPolygon geometries."""
    return [outer for outer, _holes in _iter_qgis_polygon_parts(geom)]


def _constraints_for_region(
    model: ConceptualModel,
    region_outer_ring: Sequence[Tuple[float, float]],
) -> List[CellConstraint]:
    """constraints for region"""
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
        """add zone"""
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
        """is hard breakline arc"""
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
        """inside region"""
        return _point_in_polygon(float(x), float(y), outer)

    def _line_hits_region(points: Sequence[Tuple[float, float]]) -> bool:
        """line hits region"""
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
        """clip to region segments"""
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
        """cross"""
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
    """Extract boundary edge sample endpoints from error text."""
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
        """is protected"""
        for q in prot:
            if float(np.hypot(float(p[0]) - float(q[0]), float(p[1]) - float(q[1]))) <= p_tol:
                return True
        return False

    def _near_focus(p: Tuple[float, float]) -> bool:
        """near focus"""
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
    """ring key"""
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
        """is protected"""
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
    """polyline length"""
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
    """sample closed polyline"""
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
    """longest cyclic true run"""
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
    """downsample polyline samples"""
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
    """interface overlap metrics"""
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
        except Exception as e:
            logger.warning("gmsh native overlap diagnostics failed: %s", e, exc_info=True)
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
        """endpoint deltas"""
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
    """gmsh interface coincidence report"""
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
    """sample open polyline"""
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
    """polyline overlap fractions open"""
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
        except Exception as e:
            logger.warning("gmsh native polyline_overlap_fractions failed: %s", e, exc_info=True)
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
    """split polyline at focus points"""
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
    """junction points on interface"""
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
        """consider point"""
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
        """is transfinite region"""
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
        """edge key"""
        pts = list(edge.points_xy or [])
        if len(pts) < 2:
            return None
        a = (round(float(pts[0][0]), 6), round(float(pts[0][1]), 6))
        b = (round(float(pts[-1][0]), 6), round(float(pts[-1][1]), 6))
        return (a, b) if a <= b else (b, a)

    def _density_nodes(edge: QuadEdgeControl) -> int:
        """density nodes"""
        pts = [(float(x), float(y)) for (x, y) in list(edge.points_xy or [])]
        if len(pts) < 2:
            return 2
        length = max(_polyline_length(pts), 1.0e-9)
        spacing = edge.target_size if (edge.target_size is not None and float(edge.target_size) > 0.0) else (length / max(1.0, float(len(pts) - 1)))
        spacing = max(float(spacing), 1.0e-9)
        est = max(2, int(round(length / spacing)) + 1)
        return max(int(est), int(len(pts)))

    def _oriented_chain_like(edge: QuadEdgeControl, chain: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """oriented chain like"""
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
        """find"""
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def _union(i: int, j: int) -> None:
        """union"""
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
        """edge endpoints"""
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
        """split edge with external junctions"""
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
            except Exception as e:
                logger.warning("edge target_size parse failed: %s", e, exc_info=True)
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
    """sample polyline"""
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
        """is protected"""
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
        """is protected"""
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
    """orient quad edge chains"""
    if len(edges) != 4:
        return list(edges)

    def _score(option: List[QuadEdgeControl]) -> float:
        """score"""
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
    """ring intersection tolerance"""
    if not ring:
        return 1.0e-6
    rx = np.asarray([float(p[0]) for p in ring], dtype=np.float64)
    ry = np.asarray([float(p[1]) for p in ring], dtype=np.float64)
    return max(1.0e-6, 1.0e-9 * max(float(np.ptp(rx)), float(np.ptp(ry)), 1.0))



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


def _interp_polyline_fraction(points: Sequence[Tuple[float, float]], frac: float) -> Tuple[float, float]:
    """interp polyline fraction"""
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
    """transition widths"""
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
    """build axis params"""
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
    """transfinite quad point"""
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
    """structured quad region mesh"""
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
        """edge target"""
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
        """
        idx.

        Parameters
        ----------
        i : int
            Description of i.
        j : int
            Description of j.

        Returns
        -------
        int
        """
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
    """point to segment distance s"""
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
    """polyline distance and s"""
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
    """region node sets from mesh"""
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
    """is transfinite cell type label"""
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
            """near shared geometry"""
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
                """find"""
                j = int(i)
                while int(parent[j]) != j:
                    parent[j] = parent[int(parent[j])]
                    j = int(parent[j])
                return int(j)

            def _union(a: int, b: int) -> None:
                """union"""
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
        """
        generate.

        Parameters
        ----------
        model : ConceptualModel
            Description of model.

        Returns
        -------
        MeshResult
        """
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
                """
                idx.

                Parameters
                ----------
                i : int
                    Description of i.
                j : int
                    Description of j.

                Returns
                -------
                int
                """
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
            """as bool"""
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
            except Exception as e:
                logger.warning("_as_bool conversion failed: %s", e, exc_info=True)
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
                except Exception as e:
                    logger.warning("arc asPolyline extraction failed: %s", e, exc_info=True)
                    pts = []
                if not pts:
                    try:
                        multi = geom.asMultiPolyline()
                        if multi and multi[0]:
                            pts = [(float(p.x()), float(p.y())) for p in multi[0]]
                    except Exception as e:
                        logger.warning("arc asMultiPolyline extraction failed: %s", e, exc_info=True)
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
            except Exception as e:
                logger.warning("quad_edge asPolyline extraction failed: %s", e, exc_info=True)
                pts = []
            if not pts:
                try:
                    multi = geom.asMultiPolyline()
                    if multi and multi[0]:
                        pts = [(float(p.x()), float(p.y())) for p in multi[0]]
                except Exception as e:
                    logger.warning("quad_edge asMultiPolyline extraction failed: %s", e, exc_info=True)
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



def _as_bool_opt(value: object, default: bool = False) -> bool:
    """as bool opt"""
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
    """restore mesh coordinates"""
    if x_shift == 0.0 and y_shift == 0.0:
        return mesh
    mesh.node_x = np.asarray(mesh.node_x, dtype=np.float64) + float(x_shift)
    mesh.node_y = np.asarray(mesh.node_y, dtype=np.float64) + float(y_shift)
    return mesh


def _apply_optional_post_optimization(
    mesh: MeshResult,
    model: ConceptualModel,
    options: Dict[str, object],
    backend_name: str,
) -> MeshResult:
    """apply optional post optimization"""
    return _require_nonempty_mesh(_repair_mesh_result(mesh), backend_name)


# ── Topology mesh job helpers (moved from legacy swe2d_workbench_qt) ──────

logger_wb = logging.getLogger(__name__)


def _run_topology_mesh_job(
    conceptual,
    backend_name: str,
    options: Optional[Dict[str, object]] = None,
):
    """Run heavy topology meshing work off the GUI thread/process."""
    options_local = dict(options or {})

    opts = dict(options_local)
    work_model = copy.deepcopy(conceptual)
    x_shift, y_shift = _normalize_conceptual_model_to_local_origin(work_model)

    if backend_name == "gmsh":
        from swe2d.mesh.gmsh_backend import GmshBackend, _gmsh_available
        if not _gmsh_available():
            raise RuntimeError(
                "gmsh Python package is not installed.  "
                "Run: pip install gmsh   (or select the 'Structured' backend)."
            )
        mesh = GmshBackend(options=opts).generate(work_model)
        mesh = _apply_optional_post_optimization(mesh, work_model, opts, backend_name="gmsh")
        mesh = _restore_mesh_coordinates(mesh, x_shift, y_shift)
    elif backend_name == "structured":
        mesh = StructuredFaceCentricBackend().generate(work_model)
        mesh = _apply_optional_post_optimization(mesh, work_model, opts, backend_name="structured")
        mesh = _restore_mesh_coordinates(mesh, x_shift, y_shift)
    else:
        raise ValueError(f"Unknown meshing backend: {backend_name!r}. Choose 'gmsh' or 'structured'.")
    try:
        summary = dict(getattr(mesh, "quality_summary", {}) or {})
        mesh.quality_summary = summary
    except Exception:
        logger_wb.warning("Unexpected Exception silently caught — review this handler", exc_info=True)
    return mesh


def _clone_conceptual_without_constraints(conceptual):
    """Deep-copy a conceptual model and strip all constraints."""
    clone = copy.deepcopy(conceptual)
    clone.constraints = []
    return clone
