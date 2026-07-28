"""Canonical plain-data sample-line geometry service for HYDRA SWE2D.

Spec: docs/specs/2026-07-27-canonical-sample-line-sampling.md.

This module owns the canonical sample-line geometry contract
(:func:`build_canonical_line_sampling_map`) and the executor-facing
flatten/configure helpers (:func:`flatten_canonical_sample_line_map`,
:func:`configure_canonical_sample_line_map`) plus the independent
flow-rate reference oracle (:func:`reference_line_flow`).

NO SILENT FALLBACKS:
    * Missing canonical fields raise ``KeyError``; inconsistent per-line
      array lengths raise ``ValueError``; out-of-range cell indices raise
      ``ValueError``.  No silent clipping or substitution.
    * Invalid sample-line geometry, missing required fields, or computation
      failures raise ``TypeError`` / ``ValueError`` so the GPKG and GUI
      adapters fail fast instead of returning partial maps.

The historical ``build_line_sampling_map`` (QGIS/OGR ``QgsVectorLayer``)
and ``build_line_sampling_map_numpy`` (numpy/OGR triangle-only) builders
were removed when their only callers migrated to
:func:`build_canonical_line_sampling_map`.  See Task 8 of
``docs/plans/2026-07-27-canonical-sample-line-sampling.md``.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

# osgeo.ogr is required by the canonical geometry backend; it ships with
# the QGIS Python env and is the same dependency the removed legacy
# ``build_line_sampling_map_numpy`` used.
from osgeo import ogr


__all__ = [
    "build_canonical_line_sampling_map",
    "reference_line_flow",
    "flatten_canonical_sample_line_map",
    "configure_canonical_sample_line_map",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def reference_line_flow(
    *,
    h: np.ndarray,
    hu: np.ndarray,
    hv: np.ndarray,
    cell_idx: np.ndarray,
    weights: np.ndarray,
    normal_x: float,
    normal_y: float,
    h_min: float,
) -> float:
    """Independent reference for the kernel's wet-cell line-flow formula.

    Implements the spec §11.1 oracle without touching any line-metrics
    implementation:

        wet = h > h_min
        qn  = hu * normal_x + hv * normal_y
        Q   = sum(weights * where(wet, qn, 0))

    Inputs are independent state-array slices; ``hu`` and ``hv`` are
    treated as already-h-multiplied momentum fluxes (h*u, h*v), matching
    the CUDA line-metric kernel.
    """
    h_arr = np.asarray(h, dtype=np.float64).ravel()
    hu_arr = np.asarray(hu, dtype=np.float64).ravel()
    hv_arr = np.asarray(hv, dtype=np.float64).ravel()
    cell_arr = np.asarray(cell_idx, dtype=np.int64).ravel()
    weight_arr = np.asarray(weights, dtype=np.float64).ravel()
    if cell_arr.shape != weight_arr.shape:
        raise ValueError(
            "reference_line_flow: cell_idx and weights must have equal length"
        )
    if cell_arr.size and (
        np.any(cell_arr < 0) or np.any(cell_arr >= h_arr.size)
    ):
        raise ValueError(
            "reference_line_flow: cell_idx contains an out-of-range index"
        )
    if h_arr.size != hu_arr.size or h_arr.size != hv_arr.size:
        raise ValueError(
            "reference_line_flow: h, hu, hv must have the same length"
        )
    wet = h_arr[cell_arr] > float(h_min)
    qn = hu_arr[cell_arr] * float(normal_x) + hv_arr[cell_arr] * float(normal_y)
    return float(np.sum(weight_arr * np.where(wet, qn, 0.0)))


# ---------------------------------------------------------------------------
# Canonical plain-data sample-line sampling (spec §5/§6/§7)
# ---------------------------------------------------------------------------
def _canonical_validate_sample_line(line: dict, source_idx: int) -> Optional[np.ndarray]:
    """Validate one sample-line record. Returns the validated ``(N, 2)``
    float64 points array, or ``None`` when the line should be silently
    skipped (disabled, zero length).
    """
    if not isinstance(line, dict):
        raise TypeError(
            f"sample_lines[{source_idx}] is not a dict (got {type(line).__name__})"
        )
    enabled = bool(line.get("enabled", True))
    if not enabled:
        return None
    pts = line.get("points")
    if pts is None:
        raise ValueError(
            f"sample_lines[{source_idx}] missing required 'points' array"
        )
    arr = np.asarray(pts, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(
            f"sample_lines[{source_idx}] 'points' must have shape (N, 2); "
            f"got {arr.shape}"
        )
    if arr.shape[0] < 2:
        return None
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            f"sample_lines[{source_idx}] 'points' contains non-finite values"
        )
    return arr


def _canonical_validate_mesh_cells(
    mesh_cells: List[dict],
) -> List[Tuple[int, "ogr.Geometry", "ogr.Geometry", Tuple[float, float, float, float]]]:
    """Validate plain mesh-cell records and build OGR polygon geometries.

    Returns a list of ``(cell_idx, polygon, ring_geom, bbox)`` tuples
    ready for intersection. ``ring_geom`` is the closed-ring linestring
    (used for corner-touch detection); ``bbox`` is
    ``(xmin, ymin, xmax, ymax)``. Cells with fewer than 3 points or with
    non-finite coordinates are skipped.
    """
    from osgeo import ogr as _ogr

    out: List[Tuple[int, "_ogr.Geometry", "_ogr.Geometry", Tuple[float, float, float, float]]] = []
    for ci, cell in enumerate(mesh_cells):
        if not isinstance(cell, dict):
            raise TypeError(
                f"mesh_cells[{ci}] is not a dict (got {type(cell).__name__})"
            )
        cell_idx = int(cell.get("cell_idx", ci))
        pts = cell.get("points")
        if pts is None:
            raise ValueError(
                f"mesh_cells[{cell_idx}] missing required 'points' array"
            )
        # Accept both ``(xs, ys)`` tuples and ``(N, 2)`` arrays.
        if isinstance(pts, tuple) and len(pts) == 2:
            xs = np.asarray(pts[0], dtype=np.float64).ravel()
            ys = np.asarray(pts[1], dtype=np.float64).ravel()
            arr = np.column_stack([xs, ys])
        else:
            arr = np.asarray(pts, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(
                f"mesh_cells[{cell_idx}] 'points' must have shape (N, 2); "
                f"got {arr.shape}"
            )
        if arr.shape[0] < 3:
            continue
        if not np.all(np.isfinite(arr)):
            continue
        ring_pts = ", ".join(f"{float(x)} {float(y)}" for x, y in arr)
        ring_wkt = f"LINESTRING ({ring_pts})"
        ring_geom = _ogr.CreateGeometryFromWkt(ring_wkt)
        if ring_geom is None:
            continue
        closed_pts = np.vstack([arr, arr[:1]])
        closed_wkt = "POLYGON ((" + ", ".join(
            f"{float(x)} {float(y)}" for x, y in closed_pts
        ) + "))"
        poly = _ogr.CreateGeometryFromWkt(closed_wkt)
        if poly is None:
            continue
        try:
            env = poly.GetEnvelope()
        except Exception:
            continue
        bbox = (float(env[0]), float(env[2]), float(env[1]), float(env[3]))
        out.append((cell_idx, poly, ring_geom, bbox))
    return out


def _canonical_intersect_one(
    line_geom,
    cells,
    *,
    nx: float,
    ny: float,
) -> Tuple[List[int], List[float], List[float]]:
    """Intersect one oriented line with each prepared cell polygon.

    Returns ``(cell_idxs, weights, stations)`` for every positive-length
    intersection, in the order they were encountered (deterministic by
    ``cell_idx`` because cells are sorted before iteration).

    Shared-edge deduplication: when a line sits exactly on an edge shared
    by two cells, OGR assigns the intersection to both. We deduplicate by
    keeping only the cell with the smallest index per identical
    intersection WKT.
    """
    cell_idxs: List[int] = []
    weights: List[float] = []
    stations: List[float] = []
    seen_wkt: dict = {}
    line_bbox = line_geom.GetEnvelope()
    line_bbox_t = (float(line_bbox[0]), float(line_bbox[2]),
                   float(line_bbox[1]), float(line_bbox[3]))
    for cell_idx, poly, _ring, bbox in cells:
        if not _bbox_overlap(line_bbox_t, bbox):
            continue
        try:
            inter = line_geom.Intersection(poly)
        except Exception:
            inter = None
        if inter is None or inter.IsEmpty():
            continue
        length = float(inter.Length())
        if length <= 0.0:
            continue
        try:
            wkt_key = inter.ExportToWkt()
        except Exception:
            wkt_key = None
        if wkt_key is not None:
            existing = seen_wkt.get(wkt_key)
            if existing is not None:
                if int(cell_idx) >= int(existing):
                    continue
                idx = cell_idxs.index(int(existing))
                cell_idxs[idx] = int(cell_idx)
                seen_wkt[wkt_key] = int(cell_idx)
                weights[idx] = length
                continue
            seen_wkt[wkt_key] = int(cell_idx)
        try:
            centroid = inter.Centroid()
            if centroid is None or centroid.IsEmpty():
                continue
            sx = float(centroid.GetX(0))
            sy = float(centroid.GetY(0))
        except Exception:
            continue
        # Project centroid onto oriented line to recover station.
        try:
            line_pts = line_geom.GetPoints()
            if not line_pts:
                raise ValueError("empty line")
            px = np.asarray([p[0] for p in line_pts], dtype=np.float64)
            py = np.asarray([p[1] for p in line_pts], dtype=np.float64)
        except Exception:
            line_pts = [(float(line_geom.GetX(0)), float(line_geom.GetY(0)))]
            px = np.asarray([line_pts[0][0]], dtype=np.float64)
            py = np.asarray([line_pts[0][1]], dtype=np.float64)
        seg_dx = np.diff(px)
        seg_dy = np.diff(py)
        seg_len = np.sqrt(seg_dx * seg_dx + seg_dy * seg_dy)
        cumulative = np.concatenate([np.zeros(1, dtype=np.float64), np.cumsum(seg_len)])
        nearest_i = 0
        best_d2 = float("inf")
        for i in range(px.size):
            ddx = px[i] - sx
            ddy = py[i] - sy
            d2 = ddx * ddx + ddy * ddy
            if d2 < best_d2:
                best_d2 = d2
                nearest_i = i
        base = float(cumulative[nearest_i])
        tx = float(line_geom.GetX(0))
        ty = float(line_geom.GetY(0))
        end_x = float(line_geom.GetX(line_geom.GetPointCount() - 1))
        end_y = float(line_geom.GetY(line_geom.GetPointCount() - 1))
        line_len = math.hypot(end_x - tx, end_y - ty)
        if line_len <= 0.0:
            station = base
        else:
            tx_u = (end_x - tx) / line_len
            ty_u = (end_y - ty) / line_len
            station = base + ((sx - px[nearest_i]) * tx_u + (sy - py[nearest_i]) * ty_u)
        if station < 0.0:
            station = 0.0
        cell_idxs.append(int(cell_idx))
        weights.append(length)
        stations.append(station)
    return cell_idxs, weights, stations


def _bbox_overlap(a, b) -> bool:
    """Axis-aligned bounding-box overlap test (xmin, ymin, xmax, ymax)."""
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def build_canonical_line_sampling_map(
    sample_lines,
    mesh_cells,
) -> List[Dict[str, object]]:
    """Canonical plain-data sample-line sampling map (spec §5/§6/§7).

    Inputs are plain Python records, not Qt/QGIS objects:

    ``sample_lines`` — iterable of dicts with keys ``line_id``,
    ``line_name``, ``enabled``, ``points`` (shape ``(N, 2)``, ``N >= 2``).

    ``mesh_cells`` — iterable of dicts with keys ``cell_idx`` (defaults
    to enumeration order) and ``points`` (shape ``(N, 2)``, ``N >= 3``,
    polygon ring; need not be closed).

    Returns one dict per enabled, valid sample line with
    ``line_id``, ``line_name``, ``normal_x``, ``normal_y``, ``cell_idx``
    (int32), ``weights`` (float64), ``station_m`` (float64). Equal
    array lengths, sorted by ``station_m`` along the oriented line.
    Degenerate or invalid inputs raise ``TypeError``/``ValueError``.
    """
    from osgeo import ogr as _ogr

    if sample_lines is None:
        raise TypeError("sample_lines is None")
    if mesh_cells is None:
        raise TypeError("mesh_cells is None")
    if not isinstance(mesh_cells, list):
        mesh_cells = list(mesh_cells)
    if not isinstance(sample_lines, list):
        sample_lines = list(sample_lines)

    prepared = _canonical_validate_mesh_cells(mesh_cells)
    prepared.sort(key=lambda r: r[0])

    out: List[Dict[str, object]] = []
    for src_idx, line in enumerate(sample_lines):
        pts = _canonical_validate_sample_line(line, src_idx)
        if pts is None:
            continue
        # Build OGR linestring.
        wkt_pts = ", ".join(f"{float(x)} {float(y)}" for x, y in pts)
        line_geom = _ogr.CreateGeometryFromWkt(f"LINESTRING ({wkt_pts})")
        if line_geom is None or line_geom.IsEmpty():
            raise ValueError(
                f"sample_lines[{src_idx}] failed to construct an OGR geometry"
            )
        # Tangent and unit normal.
        p0 = pts[0]
        pN = pts[-1]
        dx = float(pN[0]) - float(p0[0])
        dy = float(pN[1]) - float(p0[1])
        length = math.hypot(dx, dy)
        if length <= 0.0:
            continue
        tx = dx / length
        ty = dy / length
        # Right-handed normal (rotates tangent 90° CCW): nx = -ty, ny = tx.
        nx = -ty
        ny = tx

        cell_idxs, weights, stations = _canonical_intersect_one(
            line_geom, prepared, nx=nx, ny=ny,
        )

        if not cell_idxs:
            # Spec §6: every enabled, valid sample line produces an entry.
            out.append({
                "line_id": int(line.get("line_id", src_idx)),
                "line_name": str(line.get("line_name", "")),
                "normal_x": float(nx),
                "normal_y": float(ny),
                "cell_idx": np.empty(0, dtype=np.int32),
                "weights": np.empty(0, dtype=np.float64),
                "station_m": np.empty(0, dtype=np.float64),
            })
            continue

        order = np.argsort(np.asarray(stations, dtype=np.float64))
        cell_idxs_sorted = [int(cell_idxs[i]) for i in order]
        weights_sorted = [float(weights[i]) for i in order]
        stations_sorted = [float(stations[i]) for i in order]
        out.append({
            "line_id": int(line.get("line_id", src_idx)),
            "line_name": str(line.get("line_name", "")),
            "normal_x": float(nx),
            "normal_y": float(ny),
            "cell_idx": np.asarray(cell_idxs_sorted, dtype=np.int32),
            "weights": np.asarray(weights_sorted, dtype=np.float64),
            "station_m": np.asarray(stations_sorted, dtype=np.float64),
        })
    return out


def flatten_canonical_sample_line_map(
    sample_map,
    *,
    n_cells: int,
):
    """Flatten canonical plain-data sample-line records into GPU-friendly arrays.

    Reads only the canonical fields produced by
    :func:`build_canonical_line_sampling_map` (``line_id``, ``line_name``,
    ``normal_x``, ``normal_y``, ``cell_idx``, ``weights``, ``station_m``).
    No silent fallback: missing keys raise ``KeyError`` and inconsistent
    per-line array lengths raise ``ValueError``.

    Parameters
    ----------
    sample_map : iterable of dict
        Per-line records as produced by
        :func:`build_canonical_line_sampling_map`. Iteration order is
        preserved (caller controls line ordering).
    n_cells : int
        Total mesh-cell count. Used to validate ``cell_idx`` is in
        ``[0, n_cells)``.

    Returns
    -------
    tuple
        ``(station_offsets, cell_idx, weights, normal_x, normal_y,
        station_m, line_ids_ordered, line_names_by_id)``:

        - ``station_offsets`` — ``(L+1,)`` int32 cumulative station counts.
        - ``cell_idx`` — ``(T,)`` int32 flat cell-index array.
        - ``weights`` — ``(T,)`` float64 flat intersection lengths.
        - ``normal_x`` / ``normal_y`` — ``(T,)`` float64 per-station normals
          (broadcast of the per-line scalar).
        - ``station_m`` — ``(T,)`` float64 flat station positions.
        - ``line_ids_ordered`` — ``list[int]`` in iteration order; GPU
          index ``i`` maps back to ``line_ids_ordered[i]`` for
          :meth:`SWE2DResultsData.populate_live_line_metrics_from_gpu`.
        - ``line_names_by_id`` — ``dict[int, str]`` keyed by ``line_id``.

    Raises
    ------
    KeyError
        if a record is missing any canonical field.
    ValueError
        if per-line ``cell_idx`` / ``weights`` / ``station_m`` lengths
        disagree, or any ``cell_idx`` value is outside ``[0, n_cells)``.
    """
    station_offsets_list = [0]
    cell_idx_parts = []
    weights_parts = []
    normal_x_parts = []
    normal_y_parts = []
    station_m_parts = []
    line_ids_ordered = []
    line_names_by_id = {}

    for src_idx, sm in enumerate(sample_map):
        if not isinstance(sm, dict):
            raise TypeError(
                f"sample_map[{src_idx}] is not a dict (got {type(sm).__name__})"
            )
        # Validate the source values before narrowing to int32.  Casting first
        # can wrap large int64/uint64 values into an apparently valid index and
        # truncate fractional floats.
        raw_cell_idx = np.asarray(sm["cell_idx"]).ravel()
        if raw_cell_idx.dtype.kind not in "iuf":
            raise ValueError(
                f"sample_map[{src_idx}] (line_id={sm['line_id']}): cell_idx "
                "must contain numeric values"
            )
        try:
            cell_idx_values = np.asarray(raw_cell_idx, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"sample_map[{src_idx}] (line_id={sm['line_id']}): cell_idx "
                "must be exactly representable as int32"
            ) from exc
        if cell_idx_values.size and (
            not np.all(np.isfinite(cell_idx_values))
            or not np.all(cell_idx_values == np.trunc(cell_idx_values))
            or float(np.min(cell_idx_values)) < np.iinfo(np.int32).min
            or float(np.max(cell_idx_values)) > np.iinfo(np.int32).max
        ):
            raise ValueError(
                f"sample_map[{src_idx}] (line_id={sm['line_id']}): cell_idx "
                "must be integral and exactly representable as int32"
            )
        ci = np.asarray(cell_idx_values, dtype=np.int32)
        wt = np.asarray(sm["weights"], dtype=np.float64).ravel()
        nx = float(sm["normal_x"])
        ny = float(sm["normal_y"])
        st = np.asarray(sm["station_m"], dtype=np.float64).ravel()
        lid = int(sm["line_id"])
        lname = str(sm["line_name"])
        if ci.size != wt.size or ci.size != st.size:
            raise ValueError(
                f"sample_map[{src_idx}] (line_id={lid}): inconsistent array "
                f"lengths cell_idx={ci.size}, weights={wt.size}, "
                f"station_m={st.size} (must be equal)."
            )
        if ci.size > 0:
            if int(np.min(ci)) < 0 or int(np.max(ci)) >= int(n_cells):
                raise ValueError(
                    f"sample_map[{src_idx}] (line_id={lid}): cell_idx "
                    f"contains out-of-range indices (n_cells={int(n_cells)})."
                )
        line_names_by_id[lid] = lname
        line_ids_ordered.append(lid)
        n = int(ci.size)
        station_offsets_list.append(station_offsets_list[-1] + n)
        cell_idx_parts.append(ci)
        weights_parts.append(wt)
        normal_x_parts.append(np.full(n, nx, dtype=np.float64))
        normal_y_parts.append(np.full(n, ny, dtype=np.float64))
        station_m_parts.append(st)

    station_offsets = np.array(station_offsets_list, dtype=np.int32)
    cell_idx_arr = (
        np.concatenate(cell_idx_parts).astype(np.int32)
        if cell_idx_parts else np.empty(0, dtype=np.int32)
    )
    weights_arr = (
        np.concatenate(weights_parts).astype(np.float64)
        if weights_parts else np.empty(0, dtype=np.float64)
    )
    normal_x_arr = (
        np.concatenate(normal_x_parts).astype(np.float64)
        if normal_x_parts else np.empty(0, dtype=np.float64)
    )
    normal_y_arr = (
        np.concatenate(normal_y_parts).astype(np.float64)
        if normal_y_parts else np.empty(0, dtype=np.float64)
    )
    station_m_arr = (
        np.concatenate(station_m_parts).astype(np.float64)
        if station_m_parts else np.empty(0, dtype=np.float64)
    )
    return (
        station_offsets,
        cell_idx_arr,
        weights_arr,
        normal_x_arr,
        normal_y_arr,
        station_m_arr,
        line_ids_ordered,
        line_names_by_id,
    )


def configure_canonical_sample_line_map(
    *,
    backend,
    sample_map,
    n_cells: int,
    gravity: float,
    h_min: float,
):
    """Flatten canonical lines and configure the backend with the result."""
    flattened = flatten_canonical_sample_line_map(
        sample_map,
        n_cells=n_cells,
    )
    (
        station_offsets,
        cell_idx_arr,
        weights_arr,
        normal_x_arr,
        normal_y_arr,
        station_m_arr,
        _line_ids_ordered,
        _line_names_by_id,
    ) = flattened
    backend.configure_line_sampling(
        station_offsets=station_offsets,
        cell_idx=cell_idx_arr,
        weights=weights_arr,
        normal_x=normal_x_arr,
        normal_y=normal_y_arr,
        station_m=station_m_arr,
        gravity=float(gravity),
        h_min=float(h_min),
    )
    return flattened
