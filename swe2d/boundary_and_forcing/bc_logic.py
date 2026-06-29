from __future__ import annotations

"""Boundary-condition interpolation and hydrograph-to-edge mapping logic."""

from typing import Dict, Optional, Tuple

import numpy as np

Hydrograph = Tuple[np.ndarray, np.ndarray]
EdgeHydrographMap = Optional[Dict[int, Tuple[int, Hydrograph]]]


def interp_hydrograph(hg: Hydrograph, t_sec: float) -> float:
    """
    Interpolate a hydrograph at a given time.

    Parameters
    ----------
    hg : Hydrograph
        Tuple of (times, values) arrays.
    t_sec : float
        Query time in seconds.

    Returns
    -------
    float
        Interpolated value at *t_sec*, clamped to the
        hydrograph time range.
    """
    t, v = hg
    if t.size == 1:
        return float(v[0])
    if t_sec <= float(t[0]):
        return float(v[0])
    if t_sec >= float(t[-1]):
        return float(v[-1])
    return float(np.interp(t_sec, t, v))


def _bc_side_classification(
    edge_n0: np.ndarray,
    edge_n1: np.ndarray,
    node_x: np.ndarray,
    node_y: np.ndarray,
    node_z: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Precompute side classification and edge geometry invariants.

    All returned arrays depend only on (edge_n0, edge_n1, node_x, node_y)
    and are mesh-constant — safe to cache for the lifetime of a run.

    Parameters
    ----------
    edge_n0, edge_n1 : np.ndarray
        Boundary edge node indices.
    node_x, node_y : np.ndarray
        Node coordinates.

    Returns
    -------
    side_idx : np.ndarray
        Side index per edge (0=left, 1=right, 2=bottom, 3=top).
    edge_len : np.ndarray
        Edge length (hypotenuse of node deltas).
    edge_z : Optional[np.ndarray]
        Mean bed elevation per edge, or None if node_z not provided.
    side_names : List[str]
        Side name per edge.
    """
    xmin = float(np.min(node_x))
    xmax = float(np.max(node_x))
    ymin = float(np.min(node_y))
    ymax = float(np.max(node_y))
    mx = 0.5 * (node_x[edge_n0] + node_x[edge_n1])
    my = 0.5 * (node_y[edge_n0] + node_y[edge_n1])
    d = np.vstack([np.abs(mx - xmin), np.abs(mx - xmax), np.abs(my - ymin), np.abs(my - ymax)])
    side_idx = np.argmin(d, axis=0)
    edge_len = np.hypot(node_x[edge_n1] - node_x[edge_n0], node_y[edge_n1] - node_y[edge_n0])
    edge_z = None
    if node_z is not None:
        edge_z = 0.5 * (node_z[edge_n0] + node_z[edge_n1])
    return side_idx, edge_len, edge_z, mx, my, xmin, xmax, ymin, ymax


def distribute_total_flow_to_unit_q(
    edge_n0: np.ndarray,
    edge_n1: np.ndarray,
    bc_type_step: np.ndarray,
    bc_val_step: np.ndarray,
    bc_type_template: np.ndarray,
    side_hydrographs: Dict[str, Hydrograph],
    node_x: np.ndarray,
    node_y: np.ndarray,
    node_z: np.ndarray,
    progressive: bool,
    ts_flow_code: int,
    edge_hydrographs: EdgeHydrographMap = None,
    edge_groups: Optional[Dict[int, str]] = None,
    *,
    _side_idx: Optional[np.ndarray] = None,
    _edge_len: Optional[np.ndarray] = None,
    _edge_z: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Convert total discharge Q inputs into unit discharge q [L^2/T].

    Distributes total-flow boundary values across active edges within
    each side/group, weighted by edge length.  Supports progressive
    (time-ramped) distribution and optional pre-computed geometry invariants.

    Parameters
    ----------
    edge_n0, edge_n1 : np.ndarray
        Boundary edge node indices.
    bc_type_step : np.ndarray
        Per-edge BC type at current step.
    bc_val_step : np.ndarray
        Per-edge BC value at current step (total Q).
    bc_type_template : np.ndarray
        Static BC type template.
    side_hydrographs : dict
        Hydrograph per side name: ``{"left": (t, v), ...}``.
    node_x, node_y, node_z : np.ndarray
        Node coordinates and bed elevation.
    progressive : bool
        If True, distribute only a fraction of total Q proportional
        to elapsed fraction of peak hydrograph value.
    ts_flow_code : int
        BC code for timeseries flow (typically 102).
    edge_hydrographs : EdgeHydrographMap, optional
        Per-edge hydrograph overrides.
    edge_groups : dict, optional
        Per-edge group labels for grouped distribution.
    _side_idx, _edge_len, _edge_z : np.ndarray, optional
        Pre-computed geometry invariants for performance.

    Returns
    -------
    np.ndarray
        Per-edge unit discharge q [L^2/T], shape (E,).
    """
    if edge_n0.size == 0:
        return bc_val_step

    out_val = bc_val_step.astype(np.float64, copy=True)
    flow_idx = np.where(bc_type_step.astype(np.int32) == 2)[0]
    if flow_idx.size == 0:
        return out_val

    if _side_idx is not None:
        side_idx = _side_idx
    else:
        xmin = float(np.min(node_x))
        xmax = float(np.max(node_x))
        ymin = float(np.min(node_y))
        ymax = float(np.max(node_y))
        mx = 0.5 * (node_x[edge_n0] + node_x[edge_n1])
        my = 0.5 * (node_y[edge_n0] + node_y[edge_n1])
        d = np.vstack([np.abs(mx - xmin), np.abs(mx - xmax), np.abs(my - ymin), np.abs(my - ymax)])
        side_idx = np.argmin(d, axis=0)
    side_names = ["left", "right", "bottom", "top"]

    if _edge_len is not None:
        edge_len = _edge_len
    else:
        edge_len = np.hypot(node_x[edge_n1] - node_x[edge_n0], node_y[edge_n1] - node_y[edge_n0])
    if _edge_z is not None:
        edge_z = _edge_z
    else:
        edge_z = 0.5 * (node_z[edge_n0] + node_z[edge_n1])

    groups: Dict[Tuple, Dict[str, object]] = {}
    for i in flow_idx.tolist():
        side = side_names[int(side_idx[i])]

        peak_q = abs(float(out_val[i]))
        key: Tuple

        group_label = ""
        if edge_groups is not None:
            try:
                group_label = str(edge_groups.get(int(i), "") or "")
            except Exception:
                group_label = ""

        if edge_hydrographs is not None and i in edge_hydrographs and int(edge_hydrographs[i][0]) == ts_flow_code:
            hg = edge_hydrographs[i][1]
            try:
                peak_q = float(np.max(np.abs(hg[1]))) if hg[1].size else abs(float(out_val[i]))
            except Exception:
                peak_q = abs(float(out_val[i]))
            key = ("edge_hg", id(hg))
        elif int(bc_type_template[i]) == ts_flow_code:
            hg = side_hydrographs.get(side)
            if hg is not None:
                try:
                    peak_q = float(np.max(np.abs(hg[1]))) if hg[1].size else abs(float(out_val[i]))
                except Exception:
                    peak_q = abs(float(out_val[i]))
            if group_label:
                key = ("side_hg_group", group_label)
            else:
                key = ("side_hg", side)
        else:
            if group_label:
                key = ("static_group", group_label, round(float(out_val[i]), 12))
            else:
                key = ("static", side, round(float(out_val[i]), 12))

        if key not in groups:
            groups[key] = {
                "idx": [],
                "peak_q": max(peak_q, 0.0),
            }
        groups[key]["idx"].append(i)
        groups[key]["peak_q"] = max(float(groups[key]["peak_q"]), max(peak_q, 0.0))

    eps = 1.0e-12
    for grp in groups.values():
        idx = np.asarray(grp["idx"], dtype=np.int32)
        if idx.size == 0:
            continue

        q_total = float(out_val[idx[0]])
        if abs(q_total) <= eps:
            out_val[idx] = 0.0
            continue

        g_len = edge_len[idx]
        g_z = edge_z[idx]
        total_len = float(np.sum(g_len))
        if total_len <= eps:
            out_val[idx] = 0.0
            continue

        if progressive:
            peak_q = max(float(grp["peak_q"]), abs(q_total))
            frac = min(1.0, abs(q_total) / max(peak_q, eps))
            target_len = frac * total_len
        else:
            target_len = total_len

        if target_len <= eps:
            out_val[idx] = 0.0
            continue

        order = np.argsort(g_z, kind="stable")
        idx_sorted = idx[order]
        len_sorted = g_len[order]
        csum = np.cumsum(len_sorted)
        n_active = int(np.searchsorted(csum, target_len, side="left") + 1)
        n_active = max(1, min(n_active, idx_sorted.size))
        active_idx = idx_sorted[:n_active]
        active_len = float(np.sum(edge_len[active_idx]))
        if active_len <= eps:
            out_val[idx] = 0.0
            continue

        q_unit = q_total / active_len
        out_val[idx] = 0.0
        out_val[active_idx] = q_unit

    return out_val


def normalize_inflow_to_uniform_velocity(
    bc_val_step: np.ndarray,
    bc_type_step: np.ndarray,
    edge_h: np.ndarray,
    edge_len: np.ndarray,
    eps: float = 1.0e-12,
) -> np.ndarray:
    """
    Reweight unit discharge *q* so that *u = q/h* is uniform across inflow edges.

    Parameters
    ----------
    bc_val_step : np.ndarray
        Current per-edge unit discharge q [L2/T] after standard distribution.
    bc_type_step : np.ndarray
        Per-edge BC type code.
    edge_h : np.ndarray
        SWE depth *h* at the interior cell adjacent to each boundary edge.
    edge_len : np.ndarray
        Length of each boundary edge.
    eps : float
        Small value to avoid division by zero.

    Returns
    -------
    np.ndarray
        Updated *bc_val_step* with uniform-velocity-normalized q values.
    """
    out = bc_val_step.astype(np.float64, copy=True)
    flow_idx = np.where(bc_type_step.astype(np.int32) == 2)[0]
    if flow_idx.size == 0:
        return out

    qi = np.abs(out[flow_idx])
    hi = np.asarray(edge_h, dtype=np.float64)[flow_idx]
    li = np.asarray(edge_len, dtype=np.float64)[flow_idx]

    valid = (hi > eps) & (li > eps) & (np.isfinite(qi))
    if not np.any(valid):
        return out

    qi = qi[valid]
    hi = hi[valid]
    li = li[valid]

    total_q = float(np.sum(qi * li))
    total_area = float(np.sum(hi * li))
    if total_area <= eps or total_q <= eps:
        return out
    u_target = total_q / total_area

    flow_idx_valid = flow_idx[valid]
    for ii, idx in enumerate(flow_idx_valid):
        out[idx] = u_target * hi[ii]

    return out


def apply_timeseries_bc_values(
    edge_n0: np.ndarray,
    edge_n1: np.ndarray,
    bc_type: np.ndarray,
    bc_val: np.ndarray,
    side_hydrographs: Dict[str, Hydrograph],
    node_x: np.ndarray,
    node_y: np.ndarray,
    t_sec: float,
    ts_flow_code: int,
    ts_stage_code: int,
    edge_hydrographs: EdgeHydrographMap = None,
    *,
    _side_idx: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply timeseries BC values at time *t_sec* to boundary edges.

    Parameters prefixed with ``_`` are optional pre-computed geometry
    invariants (side classification).  When provided, the per-call
    min/max/argmin computation is skipped.

    Parameters
    ----------
    edge_n0, edge_n1 : np.ndarray
        Boundary edge node indices.
    bc_type : np.ndarray
        Per-edge BC type codes.
    bc_val : np.ndarray
        Per-edge BC values.
    side_hydrographs : dict
        Hydrograph per side name.
    node_x, node_y : np.ndarray
        Node coordinates.
    t_sec : float
        Current simulation time [s].
    ts_flow_code : int
        BC code for timeseries flow (typically 102).
    ts_stage_code : int
        BC code for timeseries stage (typically 103).
    edge_hydrographs : EdgeHydrographMap, optional
        Per-edge hydrograph overrides.
    _side_idx : np.ndarray, optional
        Pre-computed side index per edge.

    Returns
    -------
    out_type : np.ndarray
        Updated per-edge BC type codes.
    out_val : np.ndarray
        Updated per-edge BC values.
    """
    if edge_n0.size == 0:
        return bc_type, bc_val

    if _side_idx is not None:
        side_idx = _side_idx
    else:
        xmin = float(np.min(node_x))
        xmax = float(np.max(node_x))
        ymin = float(np.min(node_y))
        ymax = float(np.max(node_y))
        mx = 0.5 * (node_x[edge_n0] + node_x[edge_n1])
        my = 0.5 * (node_y[edge_n0] + node_y[edge_n1])
        d = np.vstack([np.abs(mx - xmin), np.abs(mx - xmax), np.abs(my - ymin), np.abs(my - ymax)])
        side_idx = np.argmin(d, axis=0)
    side_names = ["left", "right", "bottom", "top"]

    out_type = bc_type.astype(np.int32, copy=True)
    out_val = bc_val.astype(np.float64, copy=True)
    for i in range(edge_n0.size):
        if edge_hydrographs is not None and i in edge_hydrographs:
            tcode, hg = edge_hydrographs[i]
            out_val[i] = interp_hydrograph(hg, t_sec)
            out_type[i] = 2 if int(tcode) == ts_flow_code else 3
            continue

        tcode = int(out_type[i])
        if tcode not in (ts_flow_code, ts_stage_code):
            continue
        side = side_names[int(side_idx[i])]
        if side not in side_hydrographs:
            continue
        out_val[i] = interp_hydrograph(side_hydrographs[side], t_sec)
        out_type[i] = 2 if tcode == ts_flow_code else 3
    return out_type, out_val
