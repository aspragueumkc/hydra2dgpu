"""Pure-Python, Qt-free service for SWE2D mesh computation.

Provides numpy-heavy mesh computation extracted from
SWE2DWorkbenchDialog methods — zero Qt imports, fully testable
without QApplication.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from osgeo import gdal, ogr


def edge_lengths(node_x: np.ndarray, node_y: np.ndarray, n0: np.ndarray, n1: np.ndarray) -> np.ndarray:
    """Compute edge lengths between node pairs."""
    return np.hypot(node_x[n1] - node_x[n0], node_y[n1] - node_y[n0]).astype(np.float64)


def mesh_bounds(node_x: np.ndarray, node_y: np.ndarray):
    """Return (xmin, xmax, ymin, ymax) of mesh nodes."""
    return (
        float(np.min(node_x)) if node_x.size else 0.0,
        float(np.max(node_x)) if node_x.size else 0.0,
        float(np.min(node_y)) if node_y.size else 0.0,
        float(np.max(node_y)) if node_y.size else 0.0,
    )

import numpy as np


def build_node_coords(
    node_x: np.ndarray,
    node_y: np.ndarray,
) -> np.ndarray:
    """Build (N, 2) node coordinate array from separate x/y arrays.

    Parameters
    ----------
    node_x : (N,) ndarray
        X-coordinates of mesh nodes.
    node_y : (N,) ndarray
        Y-coordinates of mesh nodes.

    Returns
    -------
    node_coords : (N, 2) ndarray
        Column-stacked (x, y) coordinate array.
    """
    return np.column_stack(
        [np.asarray(node_x, dtype=np.float64), np.asarray(node_y, dtype=np.float64)]
    )


# ---------------------------------------------------------------------------
# Raster sampling
# ---------------------------------------------------------------------------


def assign_node_z_from_terrain(
    node_coords: np.ndarray,
    raster_data: np.ndarray,
    raster_transform: Tuple[float, ...],
    default_z: float = 0.0,
) -> np.ndarray:
    """Sample raster at node coordinates using nearest-neighbor.

    Parameters
    ----------
    node_coords : (N, 2) ndarray
        (x, y) coordinates of mesh nodes.
    raster_data : (H, W) ndarray
        Raster band values (rows = Y, columns = X).
    raster_transform : tuple of 6 floats
        GDAL-style geotransform:
        (x_origin, dx, x_rot, y_origin, y_rot, dy).
    default_z : float
        Value assigned to nodes whose raster coordinates fall outside
        the raster extent.

    Returns
    -------
    node_z : (N,) ndarray
        Sampled elevation per node.
    """
    n = node_coords.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float64)

    if raster_data is None or raster_data.size == 0:
        return np.full(n, default_z, dtype=np.float64)

    ox, dx, _, oy, _, dy = raster_transform[:6]
    x = np.asarray(node_coords[:, 0], dtype=np.float64)
    y = np.asarray(node_coords[:, 1], dtype=np.float64)

    col_f = (x - ox) / dx - 0.5 if abs(dx) > 1e-30 else np.full(n, -1.0)
    row_f = (y - oy) / dy - 0.5 if abs(dy) > 1e-30 else np.full(n, -1.0)

    col = np.round(col_f).astype(np.int32)
    row = np.round(row_f).astype(np.int32)

    nrows, ncols = raster_data.shape
    inside = (col >= 0) & (col < ncols) & (row >= 0) & (row < nrows)

    node_z = np.full(n, default_z, dtype=np.float64)
    if np.any(inside):
        node_z[inside] = raster_data[row[inside], col[inside]].astype(np.float64, copy=False)

    return node_z


# ---------------------------------------------------------------------------
# Line sampling map  (OGR/GEOS line-to-mesh intersection)
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
    total = float(cum[-1])
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


def build_line_sampling_map(
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


# ---------------------------------------------------------------------------
# Sample line metrics
# ---------------------------------------------------------------------------


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
        Pre-built sampling map from ``build_line_sampling_map``.
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
        sample_map = build_line_sampling_map(node_coords, cell_nodes, line_xy)

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
