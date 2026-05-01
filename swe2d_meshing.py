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

import numpy as np


@dataclass
class ConceptualNode:
    node_id: int
    x: float
    y: float


@dataclass
class ConceptualArc:
    arc_id: int
    node0: int
    node1: int


@dataclass
class ConceptualRegion:
    region_id: int
    ring_xy: List[Tuple[float, float]]
    default_size: float
    default_cell_type: str
    edge_lengths: Optional[List[float]] = None


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


_CELL_TYPES = {"triangular", "quadrilateral", "cartesian", "empty"}


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

    return MeshResult(
        node_x=mesh.node_x,
        node_y=mesh.node_y,
        node_z=mesh.node_z,
        cell_nodes=np.asarray(tri_plot, dtype=np.int32),
        cell_face_offsets=np.asarray(keep_offsets, dtype=np.int32),
        cell_face_nodes=np.asarray(keep_face_nodes, dtype=np.int32),
        cell_type=mesh.cell_type[keep_idx_arr],
        region_id=mesh.region_id[keep_idx_arr],
        target_size=mesh.target_size[keep_idx_arr],
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


class MeshingBackend:
    """Backend interface for computational mesh generation.

    A future CGAL-backed implementation should subclass this and implement
    `generate` while preserving the MeshResult output contract.
    """

    name = "base"

    def generate(self, model: ConceptualModel) -> MeshResult:
        raise NotImplementedError()


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
                    cx = xmin + (i + 0.5) * dx
                    cy = ymin + (j + 0.5) * dy
                    if not _point_in_polygon(cx, cy, ring):
                        continue

                    local_size = base_size
                    local_type = region.default_cell_type
                    for cst in model.constraints:
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

        out = MeshResult(
            node_x=np.asarray(all_nodes_x, dtype=np.float64),
            node_y=np.asarray(all_nodes_y, dtype=np.float64),
            node_z=np.asarray(all_nodes_z, dtype=np.float64),
            cell_nodes=np.asarray(all_tris, dtype=np.int32),
            cell_face_offsets=np.asarray(all_face_offsets, dtype=np.int32),
            cell_face_nodes=np.asarray(all_face_nodes, dtype=np.int32),
            cell_type=np.asarray(all_cell_type, dtype=object),
            region_id=np.asarray(all_region_id, dtype=np.int32),
            target_size=np.asarray(all_size, dtype=np.float64),
        )
        return _repair_mesh_result(out)


def _gmsh_available() -> bool:
    try:
        import gmsh  # noqa: F401
        return True
    except ImportError:
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

    def generate(self, model: ConceptualModel) -> MeshResult:
        import gmsh

        if not model.regions:
            raise ValueError("No conceptual regions provided.")

        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 1)
        gmsh.model.add("swe2d")

        try:
            return self._build(gmsh, model)
        finally:
            gmsh.finalize()

    # ------------------------------------------------------------------
    # Internal construction helpers
    # ------------------------------------------------------------------

    def _build(self, gmsh, model: ConceptualModel) -> MeshResult:
        # Tolerance for point deduplication (scaled to typical hydraulic coords).
        tol = 1e-6
        surface_tags: List[int] = []
        surface_meta: List[Tuple[int, str, float]] = []  # (region_id, cell_type, target_size)
        surface_curve_tags: Dict[int, List[int]] = {}

        # ---- 1. Build one Gmsh surface per region ----------------------
        for region in model.regions:
            ring = list(region.ring_xy)
            if ring and ring[0] == ring[-1]:
                ring = ring[:-1]
            if len(ring) < 3:
                continue

            ctype = region.default_cell_type
            if ctype == "empty":
                continue

            pts = [gmsh.model.geo.addPoint(x, y, 0.0, region.default_size) for x, y in ring]
            lines = []
            for i in range(len(pts)):
                lines.append(gmsh.model.geo.addLine(pts[i], pts[(i + 1) % len(pts)]))
            loop = gmsh.model.geo.addCurveLoop(lines)
            surf = gmsh.model.geo.addPlaneSurface([loop])
            surface_tags.append(surf)
            surface_meta.append((region.region_id, ctype, region.default_size))
            surface_curve_tags[surf] = lines

        if not surface_tags:
            raise ValueError("GmshBackend: no non-empty regions to mesh.")

        # ---- 2. Embed arc breaklines into surfaces ----------------------
        if model.arcs:
            arc_curve_tags: List[int] = []
            # Build a quick node-id -> (x,y) lookup
            node_xy = {n.node_id: (n.x, n.y) for n in model.nodes}
            for arc in model.arcs:
                p0_xy = node_xy.get(arc.node0)
                p1_xy = node_xy.get(arc.node1)
                if p0_xy is None or p1_xy is None:
                    continue
                gp0 = gmsh.model.geo.addPoint(p0_xy[0], p0_xy[1], 0.0)
                gp1 = gmsh.model.geo.addPoint(p1_xy[0], p1_xy[1], 0.0)
                arc_curve_tags.append(gmsh.model.geo.addLine(gp0, gp1))

            if arc_curve_tags:
                for surf in surface_tags:
                    gmsh.model.geo.synchronize()
                    try:
                        gmsh.model.geo.embed(1, arc_curve_tags, 2, surf)
                    except Exception:
                        pass  # arc may not intersect this surface; skip

        gmsh.model.geo.synchronize()

        # ---- 3. Constraint refinement zones (polygon-conforming) -------
        # Avoid bbox-based refinement (which over-refines for irregular polygons)
        # by seeding target-size points only on/inside each constraint polygon.
        constraint_seed_points: List[int] = []
        for cst in model.constraints:
            if len(cst.ring_xy) < 3 or cst.cell_type == "empty":
                continue
            ring = list(cst.ring_xy)
            if ring[0] == ring[-1]:
                ring = ring[:-1]
            if len(ring) < 3:
                continue

            # Boundary seed points at target size.
            for x, y in ring:
                try:
                    constraint_seed_points.append(gmsh.model.geo.addPoint(x, y, 0.0, cst.target_size))
                except Exception:
                    pass

            # Interior seed points on a clipped grid so refinement follows
            # polygon footprint instead of axis-aligned bbox.
            xs = [p[0] for p in ring]
            ys = [p[1] for p in ring]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            step = max(float(cst.target_size), tol * 10.0)
            max_pts = 6000
            n_added = 0
            y = ymin + 0.5 * step
            while y < ymax - 0.5 * step and n_added < max_pts:
                x = xmin + 0.5 * step
                while x < xmax - 0.5 * step and n_added < max_pts:
                    if _point_in_polygon(x, y, ring):
                        try:
                            constraint_seed_points.append(gmsh.model.geo.addPoint(x, y, 0.0, cst.target_size))
                            n_added += 1
                        except Exception:
                            pass
                    x += step
                y += step

        gmsh.model.geo.synchronize()
        if constraint_seed_points:
            # Deduplicate while preserving order.
            dedup_pts = list(dict.fromkeys(constraint_seed_points))
            for surf in surface_tags:
                try:
                    gmsh.model.mesh.embed(0, dedup_pts, 2, surf)
                except Exception:
                    pass

        # ---- 4. Per-surface algorithm and recombination flags ----------
        want_recombine = False
        for surf, (rid, ctype, sz) in zip(surface_tags, surface_meta):
            region = next((r for r in model.regions if int(r.region_id) == int(rid)), None)
            lines = surface_curve_tags.get(surf, [])
            if ctype == "cartesian":
                # Transfinite + Recombine: structured, fast, pure quads.
                if region is not None and region.edge_lengths and len(lines) == 4 and len(region.edge_lengths) == 4:
                    try:
                        p_ring = list(region.ring_xy)
                        if p_ring and p_ring[0] == p_ring[-1]:
                            p_ring = p_ring[:-1]
                        edge_geom_len = []
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
                            gmsh.model.mesh.setTransfiniteCurve(ltag, int(npt))
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
                try:
                    gmsh.model.mesh.setAlgorithm(2, surf, self._ALGO_PACKING_OF_PARALLELOGRAMS)
                except Exception:
                    gmsh.option.setNumber("Mesh.Algorithm", self._ALGO_PACKING_OF_PARALLELOGRAMS)
            elif ctype == "quadrilateral":
                # Unstructured quads via Blossom recombination.
                if region is not None and region.edge_lengths and len(lines) == 4 and len(region.edge_lengths) == 4:
                    try:
                        p_ring = list(region.ring_xy)
                        if p_ring and p_ring[0] == p_ring[-1]:
                            p_ring = p_ring[:-1]
                        edge_geom_len = []
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
                            gmsh.model.mesh.setTransfiniteCurve(ltag, int(npt))
                        gmsh.model.mesh.setTransfiniteSurface(surf)
                    except Exception:
                        pass
                gmsh.model.mesh.setRecombine(2, surf)
                want_recombine = True
                try:
                    gmsh.model.mesh.setAlgorithm(2, surf, self._ALGO_PACKING_OF_PARALLELOGRAMS)
                except Exception:
                    gmsh.option.setNumber("Mesh.Algorithm", self._ALGO_PACKING_OF_PARALLELOGRAMS)
            else:
                # triangular: frontal Delaunay for quality.
                try:
                    gmsh.model.mesh.setAlgorithm(2, surf, self._ALGO_FRONTAL)
                except Exception:
                    gmsh.option.setNumber("Mesh.Algorithm", self._ALGO_FRONTAL)

        # ---- 5. Global mesh options ------------------------------------
        gmsh.option.setNumber("Mesh.RecombineAll", 0)          # per-surface only
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)  # Blossom
        gmsh.option.setNumber("Mesh.Smoothing", 5)              # Laplacian passes
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 0)

        # ---- 6. Generate -----------------------------------------------
        gmsh.model.mesh.generate(2)
        if want_recombine:
            try:
                gmsh.model.mesh.recombine()
            except Exception:
                pass
        gmsh.model.mesh.optimize("Laplace2D", niter=3)

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
    - arcs: arc_id, node0, node1
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
            a_id = _as_int(ft["arc_id"], auto_id) if "arc_id" in arc_fields else auto_id
            n0 = _as_int(ft["node0"], -1) if "node0" in arc_fields else -1
            n1 = _as_int(ft["node1"], -1) if "node1" in arc_fields else -1
            arcs.append(ConceptualArc(arc_id=a_id, node0=n0, node1=n1))
            auto_id += 1

    region_fields = set(regions_layer.fields().names())
    auto_rid = 0
    for ft in regions_layer.getFeatures():
        geom = ft.geometry()
        if geom is None or geom.isEmpty():
            continue
        poly = geom.asPolygon()
        if not poly or not poly[0]:
            continue
        ring = [(float(p.x()), float(p.y())) for p in poly[0][:-1]]
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
        regions.append(
            ConceptualRegion(
                region_id=rid,
                ring_xy=ring,
                default_size=size,
                default_cell_type=ctype,
                edge_lengths=edge_lengths,
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
            poly = geom.asPolygon()
            if not poly or not poly[0]:
                continue
            ring = [(float(p.x()), float(p.y())) for p in poly[0][:-1]]
            cid = _as_int(ft["constraint_id"], auto_cid) if "constraint_id" in c_fields else auto_cid
            size = _as_float(ft["target_size"], default_size) if "target_size" in c_fields else default_size
            ctype = _normalize_cell_type(ft["cell_type"], default_cell_type) if "cell_type" in c_fields else default_cell_type
            constraints.append(CellConstraint(constraint_id=cid, ring_xy=ring, target_size=size, cell_type=ctype))
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
        import backwater_tqmesh  # noqa: F401
        return True
    except ImportError:
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

    Requires the ``backwater_tqmesh`` C++ extension module built from
    ``cpp/src/tqmesh_bindings.cpp``.
    """

    name = "tqmesh"

    @staticmethod
    def _quad_controls_for_region(model: ConceptualModel, region: ConceptualRegion) -> Optional[Tuple[List[Tuple[float, float]], List[QuadEdgeControl]]]:
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

    def generate(self, model: ConceptualModel) -> MeshResult:
        try:
            import backwater_tqmesh as _tq
        except ImportError as exc:
            raise RuntimeError(
                "backwater_tqmesh C++ module not found.  "
                "Rebuild the plugin (cmake + make) to compile TQMesh bindings."
            ) from exc

        if not model.regions:
            raise ValueError("TQMeshBackend: no conceptual regions provided.")

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

            ctype = region.default_cell_type
            if ctype == "empty":
                continue

            target_size = max(float(region.default_size), 1e-10)

            quad_controls = None
            quad_boundary = None
            if ctype in ("quadrilateral", "cartesian"):
                quad_setup = self._quad_controls_for_region(model, region)
                if quad_setup is not None:
                    quad_boundary, quad_controls = quad_setup

            # Exterior boundary — TQMesh expects CCW; ensure correct winding
            ext_verts = list(quad_boundary) if quad_boundary is not None else ring
            if quad_controls is None and not self._is_ccw(ext_verts):
                ext_verts = list(reversed(ext_verts))

            # All exterior edges get color 1 by default; real BC colors are
            # applied post-mesh in the workbench from swe2d_bc_lines (same
            # as the gmsh backend does).
            ext_colors = [1] * len(ext_verts)

            # Constraint zones that overlap this region
            constraint_verts_list: List[List[tuple]] = []
            constraint_sizes_list: List[float] = []
            for cst in model.constraints:
                if len(cst.ring_xy) < 3 or cst.cell_type == "empty":
                    continue
                # Only include constraints whose centroid is inside this region
                cx = sum(p[0] for p in cst.ring_xy) / len(cst.ring_xy)
                cy = sum(p[1] for p in cst.ring_xy) / len(cst.ring_xy)
                if _point_in_polygon(cx, cy, ring):
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

            result = _tq.generate_triangular_mesh(
                ext_verts=[[v[0], v[1]] for v in ext_verts],
                ext_colors=ext_colors,
                int_boundaries=[],
                int_colors=[],
                constraint_verts=[[list(v) for v in cverts] for cverts in constraint_verts_list],
                constraint_sizes=constraint_sizes_list,
                target_size=target_size,
                quad_layers=active_quad_layers,
                tri_to_quad=(ctype in ("quadrilateral", "cartesian")),
                n_smooth=3,
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

        node_x = np.asarray(all_vx, dtype=np.float64)
        node_y = np.asarray(all_vy, dtype=np.float64)
        node_z = np.zeros(len(all_vx), dtype=np.float64)

        # Build CSR face topology from triangles + quads
        face_nodes_list: List[int] = []
        face_offsets: List[int] = [0]
        plot_tris: List[int] = []

        tris_arr = np.asarray(all_tris, dtype=np.int32).reshape(-1, 3) if all_tris else np.empty((0,3), np.int32)
        quads_arr = np.asarray(all_quads, dtype=np.int32).reshape(-1, 4) if all_quads else np.empty((0,4), np.int32)

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
) -> MeshResult:
    """Generate a computational mesh from a ConceptualModel.

    Parameters
    ----------
    model   : ConceptualModel built from QGIS topology layers.
    backend : ``"gmsh"`` (default), ``"structured"``, ``"tqmesh"``.
              ``"gmsh"`` requires the ``gmsh`` Python package (pip install gmsh).
              ``"tqmesh"`` uses the built-in TQMesh advancing-front generator.
    """
    if backend == "gmsh":
        if not _gmsh_available():
            raise RuntimeError(
                "gmsh Python package is not installed.  "
                "Run: pip install gmsh   (or select the 'Structured' backend)."
            )
        return GmshBackend().generate(model)
    if backend == "structured":
        return StructuredFaceCentricBackend().generate(model)
    if backend == "tqmesh":
        return TQMeshBackend().generate(model)
    raise ValueError(f"Unknown meshing backend: {backend!r}. Choose 'gmsh', 'structured', or 'tqmesh'.")
