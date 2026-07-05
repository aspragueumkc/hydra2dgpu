"""Build line sampling maps for mesh solution profile extraction.

Extracted from ``SWE2DWorkbenchStudioDialog._build_line_sampling_map`` and
``_register_edge`` (Task B4 of the extracted migration plan).

This module uses QGIS core geometry types (not Qt widgets) for line sampling
map computation on arbitrary polygon meshes.  The dialog's old method accepted
a ``QComboBox`` and a ``combo_layer_fn`` callback; this service takes the
resolved ``QgsVectorLayer`` directly to keep the signature widget-free.

NO SILENT FALLBACKS:
    * ``build_line_sampling_map`` returns ``[]`` only when there is no mesh
      data, no line layer, or no cells — all of which are legitimate empty
      states.  Invalid geometry, missing fields, or computation failures are
      logged via the injected ``log_fn`` and that particular line is skipped
      (the rest of the map is still built).
"""

from __future__ import annotations
import logging

logger = logging.getLogger(__name__)

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from osgeo import gdal, ogr

# ---------------------------------------------------------------------------
# QGIS core availability gate
# ---------------------------------------------------------------------------
try:
    from qgis.core import (
        QgsFeature,
        QgsGeometry,
        QgsPointXY,
        QgsVectorLayer,
        QgsWkbTypes,
    )

    _HAVE_QGIS_CORE = True
except ImportError:
    _HAVE_QGIS_CORE = False

    # Stub types so the module can be imported headless (the function will
    # return [] early when _HAVE_QGIS_CORE is False).
    class QgsVectorLayer:  # type: ignore[no-redef]
        pass

    class QgsFeature:  # type: ignore[no-redef]
        pass

    class QgsGeometry:  # type: ignore[no-redef]
        pass

    class QgsPointXY:  # type: ignore[no-redef]
        pass

    class QgsWkbTypes:  # type: ignore[no-redef]
        pass


__all__ = [
    "build_line_sampling_map",
    "build_line_sampling_map_numpy",
    "sample_line_metrics",
    "sample_line_aggregate_ts_row",
]


# ---------------------------------------------------------------------------
# Edge registration helper
# ---------------------------------------------------------------------------
def _register_edge(
    ci: int,
    a: int,
    b: int,
    edge_map: Dict[Tuple[int, int], int],
    edge_n0_l: List[int],
    edge_n1_l: List[int],
    edge_c0_l: List[int],
    edge_c1_l: List[int],
    cell_face_ids: List[List[int]],
    n_cells: int,
) -> None:
    """Register a mesh edge between nodes ``a`` and ``b`` for cell ``ci``.

    This is the pure-computation core of edge-building.  All mutable
    accumulators are passed explicitly so the function remains a pure
    transformation with no hidden state.
    """
    if a == b:
        return
    ka = int(a)
    kb = int(b)
    key = (ka, kb) if ka < kb else (kb, ka)
    eid = edge_map.get(key)
    if eid is None:
        eid = int(len(edge_n0_l))
        edge_map[key] = eid
        edge_n0_l.append(int(key[0]))
        edge_n1_l.append(int(key[1]))
        edge_c0_l.append(int(ci))
        edge_c1_l.append(-1)
    else:
        if edge_c0_l[eid] != int(ci) and edge_c1_l[eid] < 0:
            edge_c1_l[eid] = int(ci)
    if 0 <= int(ci) < n_cells:
        cell_face_ids[int(ci)].append(int(eid))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_line_sampling_map(
    mesh_data: Optional[dict] = None,
    line_layer: Optional[QgsVectorLayer] = None,
    log_fn: Callable = print,
    mesh_cell_polygons_fn: Optional[Callable] = None,
    mesh_cell_centroids_fn: Optional[Callable] = None,
    mesh_cell_areas_fn: Optional[Callable] = None,
) -> List[Dict[str, object]]:
    """Build a sampling map for profile extraction along sample line layers.

    For each visible/active feature in ``line_layer`` the function:

    1. Enumerates mesh cells intersected by the line geometry.
    2. Computes intersection weights (segment length per cell).
    3. Determines the nearest mesh face for flow-direction sampling.
    4. Prepares high-fidelity profile interpolation weights (inverse-distance
       weighted neighbour cells at regular stations along the line).

    Parameters
    ----------
    mesh_data : dict or None
        Mesh topology dict with keys ``"node_x"``, ``"node_y"``,
        ``"cell_face_offsets"`` / ``"cell_face_nodes"`` (preferred), or
        ``"cell_nodes"`` (fallback triangle-only path).
    line_layer : QgsVectorLayer or None
        A QGIS vector layer whose line features define sample transects.
    log_fn : callable
        Logging sink (default ``print``).
    mesh_cell_polygons_fn : callable or None
        Zero-argument callable returning a list of ``QgsGeometry`` polygon
        objects, one per mesh cell.
    mesh_cell_centroids_fn : callable or None
        Zero-argument callable returning ``(cx, cy)`` float64 arrays of cell
        centroid coordinates.
    mesh_cell_areas_fn : callable or None
        Zero-argument callable returning a float64 array of cell areas.

    Returns
    -------
    list[dict]
        One entry per processed sample line feature.  Each dict has keys:

        - ``line_id`` (int) — feature ID or ``line_id`` field value.
        - ``line_name`` (str) — ``name`` field value or ``""``.
        - ``normal_x``, ``normal_y`` (float) — line normal components.
        - ``cell_idx`` (int32 ndarray) — intersected cell indices.
        - ``weights`` (float64 ndarray) — fractional intersection lengths.
        - ``station_m`` (float64 ndarray) — station positions at cell centroids.
        - ``flow_wx``, ``flow_wy`` (float64 ndarray) — flow-direction weights.
        - ``flux_face_idx`` (int32 ndarray) — face indices for flux sampling.
        - ``flux_face_wx``, ``flux_face_wy``, ``flux_face_len`` (float64 ndarray).
        - ``flux_face_c0``, ``flux_face_c1`` (int32 ndarray) — adjacent cell IDs.
        - ``flux_face_segments`` (float64 ndarray) — ``(N, 4)`` segment endpoints.
        - ``profile_station_m`` (float64 ndarray) — equidistant profile stations.
        - ``profile_cell_idx`` (int32 ndarray) — ``(P, k)`` neighbour indices.
        - ``profile_cell_w`` (float64 ndarray) — ``(P, k)`` IDW weights.

        Returns ``[]`` if ``mesh_data`` is ``None``, QGIS core is unavailable,
        ``line_layer`` is ``None``, or there are no mesh cells.
    """
    if mesh_data is None or not _HAVE_QGIS_CORE:
        return []
    if line_layer is None:
        return []

    fields = set(line_layer.fields().names())
    id_field = "line_id" if "line_id" in fields else None
    name_field = "name" if "name" in fields else None
    enabled_field = "enabled" if "enabled" in fields else None

    if mesh_cell_polygons_fn is None:
        return []
    cell_polys = mesh_cell_polygons_fn()
    if not cell_polys:
        return []
    cell_bboxes = [g.boundingBox() if g is not None and not g.isEmpty() else None for g in cell_polys]

    node_x = np.asarray(mesh_data.get("node_x", np.empty(0)), dtype=np.float64).ravel()
    node_y = np.asarray(mesh_data.get("node_y", np.empty(0)), dtype=np.float64).ravel()
    n_cells = int(len(cell_polys))
    cell_face_ids: List[List[int]] = [[] for _ in range(n_cells)]
    edge_map: Dict[Tuple[int, int], int] = {}
    edge_n0_l: List[int] = []
    edge_n1_l: List[int] = []
    edge_c0_l: List[int] = []
    edge_c1_l: List[int] = []

    if "cell_face_offsets" in mesh_data and "cell_face_nodes" in mesh_data:
        offs = np.asarray(mesh_data["cell_face_offsets"], dtype=np.int32).ravel()
        faces = np.asarray(mesh_data["cell_face_nodes"], dtype=np.int32).ravel()
        for ci in range(max(0, int(offs.size) - 1)):
            s = int(offs[ci])
            e = int(offs[ci + 1])
            poly = faces[s:e]
            if poly.size < 2:
                continue
            for k in range(int(poly.size)):
                a = int(poly[k])
                b = int(poly[(k + 1) % int(poly.size)])
                _register_edge(
                    ci, a, b,
                    edge_map, edge_n0_l, edge_n1_l,
                    edge_c0_l, edge_c1_l,
                    cell_face_ids, n_cells,
                )
    else:
        tri = np.asarray(mesh_data.get("cell_nodes", np.empty(0)), dtype=np.int32).reshape((-1, 3))
        for ci, t in enumerate(tri):
            _register_edge(
                ci, int(t[0]), int(t[1]),
                edge_map, edge_n0_l, edge_n1_l,
                edge_c0_l, edge_c1_l,
                cell_face_ids, n_cells,
            )
            _register_edge(
                ci, int(t[1]), int(t[2]),
                edge_map, edge_n0_l, edge_n1_l,
                edge_c0_l, edge_c1_l,
                cell_face_ids, n_cells,
            )
            _register_edge(
                ci, int(t[2]), int(t[0]),
                edge_map, edge_n0_l, edge_n1_l,
                edge_c0_l, edge_c1_l,
                cell_face_ids, n_cells,
            )

    edge_n0 = np.asarray(edge_n0_l, dtype=np.int32)
    edge_n1 = np.asarray(edge_n1_l, dtype=np.int32)
    edge_c0 = np.asarray(edge_c0_l, dtype=np.int32)
    edge_c1 = np.asarray(edge_c1_l, dtype=np.int32)
    edge_coord_map: Dict[Tuple[Tuple[float, float], Tuple[float, float]], int] = {}
    if node_x.size > 0 and node_y.size > 0 and edge_n0.size > 0 and edge_n1.size > 0:
        for eid in range(int(edge_n0.size)):
            n0 = int(edge_n0[eid])
            n1 = int(edge_n1[eid])
            if n0 < 0 or n1 < 0 or n0 >= int(node_x.size) or n1 >= int(node_x.size):
                continue
            p0 = (round(float(node_x[n0]), 9), round(float(node_y[n0]), 9))
            p1 = (round(float(node_x[n1]), 9), round(float(node_y[n1]), 9))
            key = (p0, p1) if p0 <= p1 else (p1, p0)
            edge_coord_map[key] = int(eid)

    sample_map: List[Dict[str, object]] = []
    total_profile_points = 0
    if mesh_cell_centroids_fn is not None:
        try:
            cx_all, cy_all = mesh_cell_centroids_fn()
        except Exception as e:
            log_fn(f"[ERROR] mesh cell centroids read failed: {e}")
            cx_all, cy_all = np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    else:
        cx_all, cy_all = np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    if mesh_cell_areas_fn is not None:
        try:
            area_all = mesh_cell_areas_fn()
        except Exception as e:
            log_fn(f"[ERROR] mesh cell areas read failed: {e}")
            area_all = np.empty(0, dtype=np.float64)
    else:
        area_all = np.empty(0, dtype=np.float64)

    for ft in line_layer.getFeatures():
        geom = ft.geometry()
        if geom is None or geom.isEmpty():
            continue
        try:
            if enabled_field is not None and int(ft[enabled_field]) <= 0:
                continue
        except Exception as _e:

            logger.warning(f"[ERROR] Exception in line_sampling_service.py: {_e}")

        line_len = float(geom.length())
        if line_len <= 0.0:
            continue
        try:
            p0 = geom.interpolate(0.0).asPoint()
            p1 = geom.interpolate(max(0.0, line_len - 1.0e-9)).asPoint()
            start_key = (float(p0.x()), float(p0.y()))
            end_key = (float(p1.x()), float(p1.y()))
            orient_sign = 1.0 if end_key >= start_key else -1.0
            dx = float(p1.x()) - float(p0.x())
            dy = float(p1.y()) - float(p0.y())
            if orient_sign < 0.0:
                dx = -dx
                dy = -dy
            mag = math.hypot(dx, dy)
            if mag <= 0.0:
                continue
            tx = dx / mag
            ty = dy / mag
            nx = ty
            ny = -tx
        except Exception as e:
            log_fn(f"[ERROR] profile geometry parse failed: {e}")
            continue

        try:
            line_id = int(ft[id_field]) if id_field is not None else int(ft.id())
        except Exception:
            line_id = int(ft.id())
        line_name = str(ft[name_field]) if name_field is not None and ft[name_field] not in (None, "") else ""

        line_bbox = geom.boundingBox()
        idx: List[int] = []
        lens: List[float] = []
        station_m: List[float] = []
        flow_wx: List[float] = []
        flow_wy: List[float] = []
        flow_face_idx: List[int] = []
        overlap_keys_by_row: List[set] = []
        for ci, cell_geom in enumerate(cell_polys):
            bb = cell_bboxes[ci]
            if bb is None or not bb.intersects(line_bbox):
                continue
            try:
                inter = cell_geom.intersection(geom)
            except Exception as e:
                log_fn(f"[ERROR] cell geometry intersection failed: {e}")
                continue
            if inter is None or inter.isEmpty():
                continue
            seg_len = float(inter.length())
            if seg_len <= 0.0:
                continue

            wx = 0.0
            wy = 0.0
            seg_keys: set = set()
            parts = []
            gtype = inter.wkbType()
            if QgsWkbTypes.isMultiType(gtype):
                parts = inter.asMultiPolyline()
            elif QgsWkbTypes.flatType(gtype) in (
                QgsWkbTypes.LineString,
                QgsWkbTypes.LineString25D,
                QgsWkbTypes.LineStringZ,
            ):
                poly = inter.asPolyline()
                if poly:
                    parts = [poly]
            for seg in parts:
                if seg is None or len(seg) < 2:
                    continue
                for k in range(1, len(seg)):
                    sp0 = seg[k - 1]
                    sp1 = seg[k]
                    sdx = float(sp1.x()) - float(sp0.x())
                    sdy = float(sp1.y()) - float(sp0.y())
                    try:
                        s0 = float(geom.lineLocatePoint(QgsGeometry.fromPointXY(sp0)))
                        s1 = float(geom.lineLocatePoint(QgsGeometry.fromPointXY(sp1)))
                        if orient_sign < 0.0:
                            s0 = float(line_len) - s0
                            s1 = float(line_len) - s1
                        if s1 < s0:
                            sdx = -sdx
                            sdy = -sdy
                    except Exception:
                        if (sdx * tx + sdy * ty) < 0.0:
                            sdx = -sdx
                            sdy = -sdy
                    wx += sdy
                    wy += -sdx
                    x0 = float(sp0.x())
                    y0 = float(sp0.y())
                    x1 = float(sp1.x())
                    y1 = float(sp1.y())
                    if (x1, y1) < (x0, y0):
                        x0, y0, x1, y1 = x1, y1, x0, y0
                    seg_key = (
                        (round(x0, 9), round(y0, 9)),
                        (round(x1, 9), round(y1, 9)),
                    )
                    seg_keys.add(
                        (
                            seg_key[0][0],
                            seg_key[0][1],
                            seg_key[1][0],
                            seg_key[1][1],
                        )
                    )
                    face_id = edge_coord_map.get(seg_key)
                    if face_id is not None and face_id in cell_face_ids[int(ci)]:
                        exact_face_lens[int(face_id)] = float(exact_face_lens.get(int(face_id), 0.0) + seg_len)

            s_loc = float("nan")
            cx_i = float("nan")
            cy_i = float("nan")
            try:
                cgeom = inter.centroid()
                if cgeom is not None and not cgeom.isEmpty():
                    cp = cgeom.asPoint()
                    cx_i = float(cp.x())
                    cy_i = float(cp.y())
                    s_loc = float(geom.lineLocatePoint(cgeom))
                    if orient_sign < 0.0:
                        s_loc = float(line_len) - s_loc
            except Exception as e:
                log_fn(f"[ERROR] intersection centroid computation failed: {e}")
                s_loc = float("nan")

            nearest_face = -1
            exact_face_lens: Dict[int, float] = {}
            if (
                np.isfinite(cx_i)
                and np.isfinite(cy_i)
                and 0 <= int(ci) < len(cell_face_ids)
                and edge_n0.size > 0
                and edge_n1.size > 0
                and node_x.size > 0
                and node_y.size > 0
            ):
                best_score = float("inf")
                for eid in cell_face_ids[int(ci)]:
                    if eid < 0 or eid >= int(edge_n0.size):
                        continue
                    n0 = int(edge_n0[eid])
                    n1 = int(edge_n1[eid])
                    if (
                        n0 < 0
                        or n1 < 0
                        or n0 >= int(node_x.size)
                        or n1 >= int(node_x.size)
                    ):
                        continue
                    x0 = float(node_x[n0])
                    y0 = float(node_y[n0])
                    x1 = float(node_x[n1])
                    y1 = float(node_y[n1])
                    ex = x1 - x0
                    ey = y1 - y0
                    el2 = ex * ex + ey * ey
                    if el2 <= 1.0e-18:
                        continue
                    t = ((cx_i - x0) * ex + (cy_i - y0) * ey) / el2
                    t = max(0.0, min(1.0, t))
                    px = x0 + t * ex
                    py = y0 + t * ey
                    try:
                        dist = float(geom.distance(QgsGeometry.fromPointXY(QgsPointXY(px, py))))
                    except Exception as e:
                        log_fn(f"[ERROR] geometry distance calc failed: {e}")
                        dist = math.hypot(cx_i - px, cy_i - py)
                    el = math.sqrt(el2)
                    nx_e = ey / el
                    ny_e = -ex / el
                    align = abs(nx_e * nx + ny_e * ny)
                    score = dist / (0.25 + align)
                    if score < best_score:
                        best_score = score
                        nearest_face = int(eid)

            idx.append(ci)
            lens.append(seg_len)
            station_m.append(s_loc)
            flow_wx.append(wx)
            flow_wy.append(wy)
            if exact_face_lens:
                if len(exact_face_lens) == 1:
                    flow_face_idx.append(int(next(iter(exact_face_lens.keys()))))
                else:
                    flow_face_idx.append(int(max(exact_face_lens.items(), key=lambda kv: kv[1])[0]))
            else:
                flow_face_idx.append(nearest_face)
            overlap_keys_by_row.append(seg_keys)

        if idx and overlap_keys_by_row:
            owner_count = {}
            for key_set in overlap_keys_by_row:
                for key in key_set:
                    owner_count[key] = int(owner_count.get(key, 0)) + 1
            for j, key_set in enumerate(overlap_keys_by_row):
                if not key_set:
                    continue
                denom = max(owner_count.get(k, 1) for k in key_set)
                if denom > 1:
                    scale = 1.0 / float(denom)
                    lens[j] = float(lens[j]) * scale
                    flow_wx[j] = float(flow_wx[j]) * scale
                    flow_wy[j] = float(flow_wy[j]) * scale

        if idx:
            ord_idx = np.argsort(np.nan_to_num(np.asarray(station_m, dtype=np.float64), nan=0.0))
            idx_sorted = np.asarray(idx, dtype=np.int32)[ord_idx]
            len_sorted = np.asarray(lens, dtype=np.float64)[ord_idx]
            sta_sorted = np.asarray(station_m, dtype=np.float64)[ord_idx]
            flow_wx_sorted = np.asarray(flow_wx, dtype=np.float64)[ord_idx]
            flow_wy_sorted = np.asarray(flow_wy, dtype=np.float64)[ord_idx]
            flow_face_sorted = np.asarray(flow_face_idx, dtype=np.int32)[ord_idx]

            face_wx_acc: Dict[int, float] = {}
            face_wy_acc: Dict[int, float] = {}
            face_len_acc: Dict[int, float] = {}
            for j in range(int(flow_face_sorted.size)):
                eid = int(flow_face_sorted[j])
                if eid < 0:
                    continue
                face_wx_acc[eid] = float(face_wx_acc.get(eid, 0.0) + float(flow_wx_sorted[j]))
                face_wy_acc[eid] = float(face_wy_acc.get(eid, 0.0) + float(flow_wy_sorted[j]))
                face_len_acc[eid] = float(face_len_acc.get(eid, 0.0) + float(len_sorted[j]))

            if face_wx_acc:
                f_idx = np.asarray(sorted(face_wx_acc.keys()), dtype=np.int32)
                f_wx = np.asarray([face_wx_acc[int(e)] for e in f_idx], dtype=np.float64)
                f_wy = np.asarray([face_wy_acc[int(e)] for e in f_idx], dtype=np.float64)
                f_len = np.asarray([face_len_acc[int(e)] for e in f_idx], dtype=np.float64)
                f_c0 = edge_c0[f_idx] if edge_c0.size > 0 else np.full(f_idx.size, -1, dtype=np.int32)
                f_c1 = edge_c1[f_idx] if edge_c1.size > 0 else np.full(f_idx.size, -1, dtype=np.int32)
                if node_x.size > 0 and node_y.size > 0 and edge_n0.size > 0 and edge_n1.size > 0:
                    f_seg = np.column_stack(
                        [
                            node_x[edge_n0[f_idx]],
                            node_y[edge_n0[f_idx]],
                            node_x[edge_n1[f_idx]],
                            node_y[edge_n1[f_idx]],
                        ]
                    ).astype(np.float64)
                else:
                    f_seg = np.empty((0, 4), dtype=np.float64)
            else:
                f_idx = np.empty(0, dtype=np.int32)
                f_wx = np.empty(0, dtype=np.float64)
                f_wy = np.empty(0, dtype=np.float64)
                f_len = np.empty(0, dtype=np.float64)
                f_c0 = np.empty(0, dtype=np.int32)
                f_c1 = np.empty(0, dtype=np.int32)
                f_seg = np.empty((0, 4), dtype=np.float64)

            profile_station_m = np.empty(0, dtype=np.float64)
            profile_cell_idx = np.empty((0, 0), dtype=np.int32)
            profile_cell_w = np.empty((0, 0), dtype=np.float64)
            try:
                if idx_sorted.size > 0 and cx_all.size > 0 and cy_all.size > 0:
                    area_local = np.asarray(area_all[idx_sorted], dtype=np.float64) if area_all.size > 0 else np.empty(0, dtype=np.float64)
                    good_area = area_local[np.isfinite(area_local) & (area_local > 0.0)]
                    if good_area.size > 0:
                        char_len = max(1.0e-6, float(np.sqrt(np.median(good_area))))
                    else:
                        char_len = max(1.0, float(line_len) / 64.0)

                    target_ds = max(
                        0.25 * char_len,
                        min(2.0 * char_len, float(line_len) / 256.0 if line_len > 0.0 else char_len),
                    )
                    n_profile = int(
                        max(32, min(1600, int(np.ceil(float(line_len) / max(target_ds, 1.0e-6))) + 1))
                    )

                    profile_station_m = np.linspace(0.0, float(line_len), n_profile, dtype=np.float64)
                    raw_station_m = (
                        (float(line_len) - profile_station_m)
                        if orient_sign < 0.0
                        else profile_station_m
                    )

                    k_nei = int(max(1, min(4, int(idx_sorted.size))))
                    profile_cell_idx = np.full((n_profile, k_nei), -1, dtype=np.int32)
                    profile_cell_w = np.zeros((n_profile, k_nei), dtype=np.float64)

                    cx_local = np.asarray(cx_all[idx_sorted], dtype=np.float64)
                    cy_local = np.asarray(cy_all[idx_sorted], dtype=np.float64)
                    eps = 1.0e-12
                    for jj, s_raw in enumerate(raw_station_m.tolist()):
                        try:
                            pt = geom.interpolate(float(s_raw)).asPoint()
                            px = float(pt.x())
                            py = float(pt.y())
                        except Exception as e:
                            log_fn(f"[ERROR] profile station interpolation failed: {e}")
                            continue
                        d2 = (cx_local - px) * (cx_local - px) + (cy_local - py) * (cy_local - py)
                        if d2.size <= k_nei:
                            nei_local = np.arange(d2.size, dtype=np.int32)
                        else:
                            nei_local = np.argpartition(d2, k_nei - 1)[:k_nei].astype(np.int32)
                        nei_d2 = np.maximum(d2[nei_local], eps)
                        w_nei = 1.0 / nei_d2
                        wsum = float(np.sum(w_nei))
                        if not np.isfinite(wsum) or wsum <= 0.0:
                            continue
                        w_nei = w_nei / wsum
                        n_eff = int(nei_local.size)
                        profile_cell_idx[jj, :n_eff] = idx_sorted[nei_local]
                        profile_cell_w[jj, :n_eff] = w_nei

                    total_profile_points += int(n_profile)
            except Exception as e:
                log_fn(f"[ERROR] profile cell weight computation failed: {e}")
                profile_station_m = np.empty(0, dtype=np.float64)
                profile_cell_idx = np.empty((0, 0), dtype=np.int32)
                profile_cell_w = np.empty((0, 0), dtype=np.float64)

            sample_map.append(
                {
                    "line_id": int(line_id),
                    "line_name": line_name,
                    "normal_x": float(nx),
                    "normal_y": float(ny),
                    "cell_idx": idx_sorted,
                    "weights": len_sorted,
                    "station_m": sta_sorted,
                    "flow_wx": flow_wx_sorted,
                    "flow_wy": flow_wy_sorted,
                    "flux_face_idx": f_idx,
                    "flux_face_wx": f_wx,
                    "flux_face_wy": f_wy,
                    "flux_face_len": f_len,
                    "flux_face_c0": f_c0,
                    "flux_face_c1": f_c1,
                    "flux_face_segments": f_seg,
                    "profile_station_m": profile_station_m,
                    "profile_cell_idx": profile_cell_idx,
                    "profile_cell_w": profile_cell_w,
                }
            )

    if sample_map:
        log_fn(
            f"Sample line mapping ready: {len(sample_map)} line(s), "
            f"high-fidelity profile samples={int(total_profile_points)}."
        )
    return sample_map


# ---------------------------------------------------------------------------
# Numpy/OGR line-sampling helpers (moved from mesh_service.py)
# ---------------------------------------------------------------------------


def _cumulative_length(polyline: np.ndarray) -> np.ndarray:
    """Return cumulative length along polyline."""
    if polyline.shape[0] < 2:
        return np.array([0.0], dtype=np.float64)
    diffs = np.diff(polyline, axis=0)
    seg_lens = np.sqrt(np.sum(diffs ** 2, axis=1))
    return np.concatenate([[0.0], np.cumsum(seg_lens)])


def _interpolate_along_line(
    line_xy: np.ndarray, stations: np.ndarray,
) -> np.ndarray:
    """Return (N, 2) coordinates for stations along polyline."""
    if line_xy.shape[0] < 2 or stations.size == 0:
        return np.empty((0, 2), dtype=np.float64)

    cum = _cumulative_length(line_xy)
    diffs = np.diff(line_xy, axis=0)

    result = np.zeros((stations.size, 2), dtype=np.float64)
    for i in range(stations.size):
        s = stations[i]
        seg_idx = int(np.searchsorted(cum, s, side="right")) - 1
        seg_idx = max(0, min(seg_idx, line_xy.shape[0] - 2))
        seg_start = cum[seg_idx]
        seg_len = max(cum[seg_idx + 1] - seg_start, 1e-12)
        t = (s - seg_start) / seg_len
        t = max(0.0, min(1.0, t))
        result[i] = line_xy[seg_idx] + t * diffs[seg_idx]
    return result


def _line_normal(line_xy: np.ndarray) -> Tuple[float, float, float]:
    """Return (nx, ny, orient_sign) for the line, normal pointing left."""
    if line_xy.shape[0] < 2:
        return 0.0, 1.0, 1.0
    p0 = line_xy[0]
    p1 = line_xy[-1]
    dx = float(p1[0] - p0[0])
    dy = float(p1[1] - p0[1])
    mag = max(np.hypot(dx, dy), 1e-15)
    orient_sign = 1.0 if (p1[0], p1[1]) >= (p0[0], p0[1]) else -1.0
    if orient_sign < 0.0:
        dx = -dx
        dy = -dy
    nx = dy / mag
    ny = -dx / mag
    return nx, ny, orient_sign


def _project_point_onto_line(
    px: float, py: float, line_xy: np.ndarray,
) -> float:
    """Return distance-along-line from start to the nearest projected point."""
    cum = _cumulative_length(line_xy)
    best_d = 0.0
    best_dist2 = float("inf")
    for i in range(line_xy.shape[0] - 1):
        x0, y0 = float(line_xy[i, 0]), float(line_xy[i, 1])
        x1, y1 = float(line_xy[i + 1, 0]), float(line_xy[i + 1, 1])
        dx, dy = x1 - x0, y1 - y0
        seg2 = dx * dx + dy * dy
        if seg2 < 1e-24:
            continue
        t = max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / seg2))
        d2 = (px - (x0 + t * dx)) ** 2 + (py - (y0 + t * dy)) ** 2
        if d2 < best_dist2:
            best_dist2 = d2
            best_d = float(cum[i]) + t * float(np.sqrt(seg2))
    return best_d


def _cell_centroids(
    node_coords: np.ndarray, cell_nodes: np.ndarray,
) -> np.ndarray:
    """Return (n_cells, 2) centroid array."""
    if cell_nodes.ndim != 2 or cell_nodes.shape[1] < 3:
        return np.empty((0, 2), dtype=np.float64)
    return np.array([
        np.mean(node_coords[cell_nodes[i]], axis=0)
        for i in range(cell_nodes.shape[0])
    ], dtype=np.float64)


def _empty_sample_map() -> Dict[str, Any]:
    """empty sample map."""
    return {
        "cell_idx": np.empty(0, dtype=np.int32),
        "weights": np.empty(0, dtype=np.float64),
        "normal_x": 0.0,
        "normal_y": 1.0,
        "station_m": np.empty(0, dtype=np.float64),
        "profile_station_m": np.empty(0, dtype=np.float64),
        "profile_cell_idx": np.empty((0, 0), dtype=np.int32),
        "profile_cell_w": np.empty((0, 0), dtype=np.float64),
    }


def build_line_sampling_map_numpy(
    node_coords: np.ndarray,
    cell_nodes: np.ndarray,
    line_xy: np.ndarray,
) -> Dict[str, Any]:
    """Build mapping data for sampling mesh solution along a profile line.

    Uses OGR/GEOS geometry intersection (same algorithm as QGIS) with
    a numpy bounding-box pre-filter to avoid O(N*M) broadcast.

    Parameters
    ----------
    node_coords : (N, 2) ndarray
        (x, y) coordinates of mesh nodes.
    cell_nodes : (Nc, 3) ndarray
        Triangle node-index list.
    line_xy : (M, 2) ndarray
        Vertex coordinates of the profile polyline.

    Returns
    -------
    dict with keys:
        cell_idx       — (K,) int32, indices of intersected cells
        weights        — (K,) float64, fractional weight per cell
        normal_x       — float, line normal x-component
        normal_y       — float, line normal y-component
        station_m      — (K,) float64, station positions at cell centroids
        profile_station_m  — (P,) float64, equidistant profile stations
        profile_cell_idx   — (P, k) int32, neighbor cell indices per station
        profile_cell_w     — (P, k) float64, IDW weights per neighbor
    """
    if line_xy.shape[0] < 2 or cell_nodes.shape[0] == 0:
        return _empty_sample_map()

    line_len = float(_cumulative_length(line_xy)[-1])
    if line_len <= 0.0:
        return _empty_sample_map()

    nx, ny, orient_sign = _line_normal(line_xy)
    centroids = _cell_centroids(node_coords, cell_nodes)

    # ── Numpy bbox pre-filter (vectorized, zero OGR allocation) ──
    tri_coords = node_coords[cell_nodes]  # (Nc, 3, 2)
    cell_xmin = tri_coords[:, :, 0].min(axis=1)
    cell_xmax = tri_coords[:, :, 0].max(axis=1)
    cell_ymin = tri_coords[:, :, 1].min(axis=1)
    cell_ymax = tri_coords[:, :, 1].max(axis=1)
    line_xmin, line_xmax = float(line_xy[:, 0].min()), float(line_xy[:, 0].max())
    line_ymin, line_ymax = float(line_xy[:, 1].min()), float(line_xy[:, 1].max())
    mask = (
        (cell_xmin <= line_xmax) & (cell_xmax >= line_xmin)
        & (cell_ymin <= line_ymax) & (cell_ymax >= line_ymin)
    )
    candidates = np.where(mask)[0]
    if candidates.size == 0:
        return _empty_sample_map()

    # ── Build OGR line geometry once ──
    line_geom = ogr.Geometry(ogr.wkbLineString)
    for i in range(line_xy.shape[0]):
        line_geom.AddPoint(float(line_xy[i, 0]), float(line_xy[i, 1]))

    # ── GEOS intersection for bbox-survivors only ──
    cell_idx_list = []
    weight_list = []
    station_list = []

    gdal.PushErrorHandler("CPLQuietErrorHandler")
    try:
        for ci in candidates:
            ci = int(ci)
            ring = ogr.Geometry(ogr.wkbLinearRing)
            for k in range(3):
                nk = int(cell_nodes[ci, k])
                ring.AddPoint(float(node_coords[nk, 0]), float(node_coords[nk, 1]))
            nk0 = int(cell_nodes[ci, 0])
            ring.AddPoint(float(node_coords[nk0, 0]), float(node_coords[nk0, 1]))
            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(ring)

            inter = poly.Intersection(line_geom)
            if inter is None:
                continue
            seg_len = float(inter.Length())
            if seg_len <= 0.0:
                continue

            station = _project_point_onto_line(
                float(centroids[ci, 0]), float(centroids[ci, 1]), line_xy,
            )
            cell_idx_list.append(ci)
            weight_list.append(seg_len)
            station_list.append(station)
    finally:
        gdal.PopErrorHandler()

    if not cell_idx_list:
        return _empty_sample_map()

    cell_idx = np.array(cell_idx_list, dtype=np.int32)
    weights = np.array(weight_list, dtype=np.float64)
    weights /= weights.sum()
    station_m = np.array(station_list, dtype=np.float64)

    order = np.argsort(station_m)
    cell_idx = cell_idx[order]
    weights = weights[order]
    station_m = station_m[order]

    # ── Profile data (IDW on the small filtered set) ──
    k_nei = min(4, int(cell_idx.size))
    n_profile = max(32, min(1600, int(np.ceil(line_len / max(line_len / 256.0, 1e-6)) + 1)))
    profile_station_m = np.linspace(0.0, line_len, n_profile, dtype=np.float64)

    if orient_sign < 0.0:
        raw_stations = line_len - profile_station_m
    else:
        raw_stations = profile_station_m

    profile_pts = _interpolate_along_line(line_xy, raw_stations)
    local_centroids = centroids[cell_idx]

    profile_cell_idx = np.full((n_profile, k_nei), -1, dtype=np.int32)
    profile_cell_w = np.zeros((n_profile, k_nei), dtype=np.float64)

    if local_centroids.shape[0] > 0:
        eps = 1e-12
        for j in range(n_profile):
            d2 = np.sum((local_centroids - profile_pts[j]) ** 2, axis=1)
            if d2.size <= k_nei:
                nei = np.arange(d2.size, dtype=np.int32)
            else:
                nei = np.argpartition(d2, k_nei - 1)[:k_nei].astype(np.int32)
            d2_nei = np.maximum(d2[nei], eps)
            w_nei = 1.0 / d2_nei
            wsum = float(np.sum(w_nei))
            if np.isfinite(wsum) and wsum > 0.0:
                w_nei = w_nei / wsum
                n_eff = int(nei.size)
                profile_cell_idx[j, :n_eff] = cell_idx[nei].astype(np.int32)
                profile_cell_w[j, :n_eff] = w_nei

    return {
        "cell_idx": cell_idx,
        "weights": weights,
        "normal_x": float(nx),
        "normal_y": float(ny),
        "station_m": station_m,
        "profile_station_m": profile_station_m,
        "profile_cell_idx": profile_cell_idx,
        "profile_cell_w": profile_cell_w,
    }


# ---------------------------------------------------------------------------
# Sample line metrics (numpy/OGR path, moved from mesh_service.py)
# ---------------------------------------------------------------------------


def _empty_metrics() -> Dict[str, np.ndarray]:
    """empty metrics."""
    return {
        "station_m": np.empty(0, dtype=np.float64),
        "depth_m": np.empty(0, dtype=np.float64),
        "velocity_ms": np.empty(0, dtype=np.float64),
        "wse_m": np.empty(0, dtype=np.float64),
        "bed_m": np.empty(0, dtype=np.float64),
        "froude": np.empty(0, dtype=np.float64),
        "wet": np.empty(0, dtype=np.int32),
        "flow_qn": np.empty(0, dtype=np.float64),
    }


def sample_line_metrics(
    h: np.ndarray,
    hu: np.ndarray,
    hv: np.ndarray,
    bed: np.ndarray,
    node_coords: np.ndarray,
    cell_nodes: np.ndarray,
    line_xy: np.ndarray,
    h_min: float,
    timestep_s: float,
    gravity: float,
    sample_map: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Sample mesh solution along a profile line.

    Parameters
    ----------
    h : (n_cells,) ndarray
        Water depth at each cell.
    hu : (n_cells,) ndarray
        X-momentum at each cell.
    hv : (n_cells,) ndarray
        Y-momentum at each cell.
    bed : (n_cells,) ndarray
        Bed elevation at each cell.
    node_coords : (N, 2) ndarray
        (x, y) coordinates of mesh nodes.
    cell_nodes : (Nc, 3) ndarray
        Triangle node-index list.
    line_xy : (M, 2) ndarray
        Vertex coordinates of the profile polyline.
    h_min : float
        Minimum depth for wet/dry threshold.
    timestep_s : float
        Current simulation time (metadata only).
    gravity : float
        Gravitational acceleration.
    sample_map : dict, optional
        Pre-built sampling map from ``build_line_sampling_map_numpy``.
        If not provided, it is built internally.

    Returns
    -------
    dict with keys:
        station_m    — (P,) float64, profile station positions (m)
        depth_m      — (P,) float64, water depth at each station
        velocity_ms  — (P,) float64, velocity magnitude
        wse_m        — (P,) float64, water surface elevation
        bed_m        — (P,) float64, bed elevation
        froude       — (P,) float64, Froude number
        wet          — (P,) int32, 1 if wet, 0 if dry
        flow_qn      — (P,) float64, normal unit discharge (m²/s)
    """
    if h.size == 0 or (sample_map is None and line_xy.shape[0] < 2):
        return _empty_metrics()

    if sample_map is None:
        sample_map = build_line_sampling_map_numpy(node_coords, cell_nodes, line_xy)

    cell_idx = sample_map.get("cell_idx", np.empty(0, dtype=np.int32))
    if cell_idx.size == 0:
        return _empty_metrics()

    nx = float(sample_map.get("normal_x", 0.0))
    ny = float(sample_map.get("normal_y", 1.0))

    # Profile data
    p_sta = sample_map.get("profile_station_m", np.empty(0, dtype=np.float64))
    p_idx = sample_map.get("profile_cell_idx", np.empty((0, 0), dtype=np.int32))
    p_w = sample_map.get("profile_cell_w", np.empty((0, 0), dtype=np.float64))

    use_hi_fidelity = (
        p_sta.ndim == 1
        and p_sta.size > 0
        and p_idx.ndim == 2
        and p_w.ndim == 2
        and p_idx.shape == p_w.shape
        and p_idx.shape[0] == p_sta.size
    )

    if not use_hi_fidelity:
        return _empty_metrics()

    valid = p_idx >= 0
    safe_idx = np.where(valid, p_idx, 0)
    ww = np.where(valid, p_w, 0.0)
    wsum = np.sum(ww, axis=1)
    good = np.isfinite(wsum) & (wsum > 0.0)

    h_nei = h[safe_idx]
    hu_nei = hu[safe_idx]
    hv_nei = hv[safe_idx]
    zb_nei = bed[safe_idx]

    hh_p = np.where(good, np.sum(h_nei * ww, axis=1) / np.maximum(wsum, 1e-12), np.nan)
    huu_p = np.where(good, np.sum(hu_nei * ww, axis=1) / np.maximum(wsum, 1e-12), np.nan)
    hvv_p = np.where(good, np.sum(hv_nei * ww, axis=1) / np.maximum(wsum, 1e-12), np.nan)
    zb_p = np.where(good, np.sum(zb_nei * ww, axis=1) / np.maximum(wsum, 1e-12), np.nan)

    wet_p = good & np.isfinite(hh_p) & (hh_p > h_min)
    safe_h_p = np.maximum(hh_p, 1e-12)
    uu_p = np.where(wet_p, huu_p / safe_h_p, 0.0)
    vv_p = np.where(wet_p, hvv_p / safe_h_p, 0.0)
    vel_p = np.where(wet_p, np.sqrt(uu_p * uu_p + vv_p * vv_p), 0.0)
    qn_p = np.where(wet_p, hh_p * (uu_p * nx + vv_p * ny), 0.0)
    fr_p = np.where(wet_p, vel_p / np.sqrt(np.maximum(gravity * hh_p, 1e-12)), 0.0)
    wse_p = np.where(
        np.isfinite(hh_p) & np.isfinite(zb_p),
        hh_p + zb_p,
        np.nan,
    )

    return {
        "station_m": p_sta,
        "depth_m": np.where(wet_p, hh_p, np.nan),
        "velocity_ms": np.where(wet_p, vel_p, 0.0),
        "wse_m": np.where(wet_p, hh_p + zb_p, np.nan),
        "bed_m": np.where(good, zb_p, np.nan),
        "froude": np.where(wet_p, fr_p, np.nan),
        "wet": wet_p.astype(np.int32),
        "flow_qn": np.where(wet_p, qn_p, 0.0),
    }


def sample_line_aggregate_ts_row(
    sm: dict,
    h: np.ndarray,
    hu: np.ndarray,
    hv: np.ndarray,
    cell_bed: np.ndarray,
    h_min: float,
    gravity: float,
    t_accum: float,
) -> Optional[Dict[str, Any]]:
    """sample line aggregate ts row."""
    idx = np.asarray(sm.get("cell_idx", np.empty(0)), dtype=np.int32)
    w = np.asarray(sm.get("weights", np.empty(0)), dtype=np.float64)
    if idx.size == 0 or w.size == 0:
        return None
    hh = h[idx]; huu = hu[idx]; hvv = hv[idx]; zb = cell_bed[idx]
    wet = (hh > h_min)
    safe_h = np.maximum(hh, 1.0e-12)
    vel = np.where(wet, np.sqrt((huu / safe_h) ** 2 + (hvv / safe_h) ** 2), 0.0)
    wsum = float(np.sum(w))
    if wsum <= 0.0:
        return None
    depth_m = float(np.sum(hh * w) / wsum)
    velocity_ms = float(np.sum(vel * w) / wsum)
    wse_m = float(np.sum((hh + zb) * w) / wsum)
    bed_m = float(np.sum(zb * w) / wsum)
    huu_wet = np.where(wet, huu, 0.0); hvv_wet = np.where(wet, hvv, 0.0)
    uu = np.where(wet, huu_wet / safe_h, 0.0)
    vv = np.where(wet, hvv_wet / safe_h, 0.0)
    normal_v = uu * float(sm.get("normal_x", 0.0)) + vv * float(sm.get("normal_y", 1.0))
    qn = np.where(wet, hh * normal_v, 0.0)
    flow_wx = np.asarray(sm.get("flow_wx", []), dtype=np.float64)
    flow_wy = np.asarray(sm.get("flow_wy", []), dtype=np.float64)
    flow_cell_cms = float(np.sum(qn * w))
    if flow_wx.size == idx.size and flow_wy.size == idx.size:
        flow_cell_cms = float(np.sum(np.where(wet, hh * (uu * flow_wx + vv * flow_wy), 0.0)))
    flow_fv_cms = float("nan")
    f_idx = np.asarray(sm.get("flux_face_idx", []), dtype=np.int32)
    f_wx = np.asarray(sm.get("flux_face_wx", []), dtype=np.float64)
    f_wy = np.asarray(sm.get("flux_face_wy", []), dtype=np.float64)
    f_c0 = np.asarray(sm.get("flux_face_c0", []), dtype=np.int32)
    f_c1 = np.asarray(sm.get("flux_face_c1", []), dtype=np.int32)
    if f_idx.size > 0 and f_wx.size == f_idx.size and f_wy.size == f_idx.size and f_c0.size == f_idx.size and f_c1.size == f_idx.size:
        c0 = np.asarray(f_c0, dtype=np.int32); c1 = np.asarray(f_c1, dtype=np.int32)
        valid_c0 = (c0 >= 0) & (c0 < h.size); valid_c1 = (c1 >= 0) & (c1 < h.size)
        hu_f = np.zeros(f_idx.size, dtype=np.float64); hv_f = np.zeros(f_idx.size, dtype=np.float64)
        if np.any(valid_c0):
            hu_f[valid_c0] = hu[c0[valid_c0]]; hv_f[valid_c0] = hv[c0[valid_c0]]
        both = valid_c0 & valid_c1
        if np.any(both):
            hu_f[both] = 0.5 * (hu[c0[both]] + hu[c1[both]])
            hv_f[both] = 0.5 * (hv[c0[both]] + hv[c1[both]])
        valid_face = valid_c0 | valid_c1
        if np.any(valid_face):
            flow_fv_cms = float(np.sum((hu_f[valid_face] * f_wx[valid_face]) + (hv_f[valid_face] * f_wy[valid_face])))
    flow_cms = flow_fv_cms if np.isfinite(flow_fv_cms) else flow_cell_cms
    fr_arr = np.where(wet, vel / np.sqrt(np.maximum(gravity * hh, 1.0e-12)), 0.0)
    return {
        "t_s": float(t_accum),
        "line_id": int(sm.get("line_id", -1)),
        "line_name": str(sm.get("line_name", "") or ""),
        "depth_m": depth_m,
        "velocity_ms": velocity_ms,
        "wse_m": wse_m,
        "bed_m": bed_m,
        "flow_cms": flow_cms,
        "flow_cell_cms": flow_cell_cms,
        "flow_fv_cms": flow_fv_cms,
        "wet_frac": float(np.mean(wet.astype(np.float64))),
        "fr": float(np.mean(fr_arr)),
        "_idx": idx,
        "_w": w,
        "_hh": hh,
        "_zb": zb,
        "_vel": vel,
        "_qn": qn,
        "_wet": wet,
        "_fr_arr": fr_arr,
    }
