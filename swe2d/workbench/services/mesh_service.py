"""Pure-Python, Qt-free service for SWE2D mesh computation.

Provides numpy-heavy mesh computation extracted from
SWE2DWorkbenchDialog methods — zero Qt imports, fully testable
without QApplication.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


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
# Line sampling map  (pure-numpy line-to-mesh intersection)
# ---------------------------------------------------------------------------


def _barycentric_coords(
    pts: np.ndarray,
    tri_a: np.ndarray,
    tri_b: np.ndarray,
    tri_c: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return barycentric (u, v, w) for each point w.r.t. each triangle.

    Parameters
    ----------
    pts : (N, 2)
        Query points.
    tri_a, tri_b, tri_c : (M, 2)
        Triangle vertices.

    Returns
    -------
    u, v, w : (N, M) ndarray
        Barycentric coordinates.
    """
    v0 = tri_c - tri_a
    v1 = tri_b - tri_a
    v2 = pts[:, np.newaxis, :] - tri_a[np.newaxis, :, :]

    dot00 = np.sum(v0 * v0, axis=1)
    dot01 = np.sum(v0 * v1, axis=1)
    dot11 = np.sum(v1 * v1, axis=1)
    denom = dot00 * dot11 - dot01 * dot01
    denom = np.where(np.abs(denom) < 1e-30, 1e-30, denom)
    inv_denom = 1.0 / denom

    dot02 = np.sum(v2 * v0[np.newaxis, :, :], axis=2)
    dot12 = np.sum(v2 * v1[np.newaxis, :, :], axis=2)

    u = (dot11[np.newaxis, :] * dot02 - dot01[np.newaxis, :] * dot12) * inv_denom[np.newaxis, :]
    v = (dot00[np.newaxis, :] * dot12 - dot01[np.newaxis, :] * dot02) * inv_denom[np.newaxis, :]
    w = 1.0 - u - v
    return u, v, w


def _points_in_triangles(
    pts: np.ndarray, tri_verts: np.ndarray,
) -> np.ndarray:
    """Return (N,) array of containing triangle index (-1 if none).

    Parameters
    ----------
    pts : (N, 2)
        Query points.
    tri_verts : (M, 3, 2)
        Triangle vertex coordinates.

    Returns
    -------
    containing : (N,) int32
        Index of containing triangle, or -1.
    """
    if pts.shape[0] == 0 or tri_verts.shape[0] == 0:
        return np.full(pts.shape[0], -1, dtype=np.int32)

    u, v, w = _barycentric_coords(pts, tri_verts[:, 0], tri_verts[:, 1], tri_verts[:, 2])
    inside = (u >= 0) & (v >= 0) & (w >= 0)
    containing = np.where(np.any(inside, axis=1), np.argmax(inside, axis=1), -1).astype(np.int32)
    return containing


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

    Uses pure-numpy barycentric point-in-triangle tests to determine
    which cells the line passes through and their relative weights.

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

    # Line geometry
    line_len = float(_cumulative_length(line_xy)[-1])
    if line_len <= 0.0:
        return _empty_sample_map()

    nx, ny, orient_sign = _line_normal(line_xy)

    # Build triangle vertex array
    tri_verts = node_coords[cell_nodes]  # (Nc, 3, 2)

    # Cell centroids
    centroids = _cell_centroids(node_coords, cell_nodes)

    # Compute number of sample points along the line
    diag = max(np.sqrt(np.sum((node_coords.max(axis=0) - node_coords.min(axis=0)) ** 2)), 1.0)
    n_samples = max(64, min(4096, int(line_len / max(diag / cell_nodes.shape[0], 1e-12) * 8)))

    # Generate sample points along the line
    sample_stations = np.linspace(0.0, line_len, n_samples, dtype=np.float64)
    sample_pts = _interpolate_along_line(line_xy, sample_stations)

    if sample_pts.shape[0] == 0:
        return _empty_sample_map()

    # Find containing cell for each sample point
    containing = _points_in_triangles(sample_pts, tri_verts)

    # Accumulate weights per cell
    valid_mask = containing >= 0
    containing_valid = containing[valid_mask]
    stations_valid = sample_stations[valid_mask]

    if containing_valid.size == 0:
        return _empty_sample_map()

    unique_cells, counts = np.unique(containing_valid, return_counts=True)
    weights = counts.astype(np.float64) / float(counts.sum())
    # Sort by station
    cell_stations = np.array([
        float(np.mean(stations_valid[containing_valid == c]))
        for c in unique_cells
    ], dtype=np.float64)
    order = np.argsort(cell_stations)
    cell_idx = unique_cells[order].astype(np.int32)
    weights = weights[order]
    station_m = cell_stations[order]

    # Build profile data (high-fidelity stations)
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
    if h.size == 0 or line_xy.shape[0] < 2:
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
