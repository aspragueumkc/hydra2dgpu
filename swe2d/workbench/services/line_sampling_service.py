"""Qt-free line-sampling orchestration for runtime profile extraction."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _triangulate_polygon_mesh(mesh_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Triangulate a polygon mesh via centroid-vertex fan.

    Parameters
    ----------
    mesh_data : dict
        Mesh topology with ``node_x``, ``node_y``, ``cell_face_offsets`` and
        ``cell_face_nodes``.

    Returns
    -------
    node_coords : (N + n_cells, 2) ndarray
        Original nodes plus cell-centroid nodes.
    cell_nodes : (n_triangles, 3) ndarray
        Triangle node indices.
    """
    _nx = np.asarray(mesh_data.get("node_x", np.empty(0)), dtype=np.float64).ravel()
    _ny = np.asarray(mesh_data.get("node_y", np.empty(0)), dtype=np.float64).ravel()
    _fo = np.asarray(mesh_data.get("cell_face_offsets"), dtype=np.int32).ravel()
    _fn = np.asarray(mesh_data.get("cell_face_nodes"), dtype=np.int32).ravel()
    n_cells = _fo.size - 1
    total_tris = sum(int(_fo[i + 1] - _fo[i]) for i in range(n_cells))
    cell_nodes = np.empty((total_tris, 3), dtype=np.int32)
    nx_ext = np.concatenate([_nx, np.zeros(n_cells)])
    ny_ext = np.concatenate([_ny, np.zeros(n_cells)])
    idx = 0
    for c in range(n_cells):
        s, e = int(_fo[c]), int(_fo[c + 1])
        cx = float(np.mean(_nx[_fn[s:e]]))
        cy = float(np.mean(_ny[_fn[s:e]]))
        nx_ext[_nx.size + c] = cx
        ny_ext[_ny.size + c] = cy
        for i in range(s, e):
            na = int(_fn[i])
            nb = int(_fn[(i + 1 - s) % (e - s) + s])
            cell_nodes[idx] = [na, nb, _nx.size + c]
            idx += 1
    return np.column_stack([nx_ext, ny_ext]), cell_nodes


def _sample_line_metrics_logic(
    sample_map: Sequence[Dict[str, object]],
    t_accum: float,
    h_s: np.ndarray,
    hu_s: np.ndarray,
    hv_s: np.ndarray,
    cell_solver_z: np.ndarray,
    gravity: float,
    h_min: float,
    mesh_data: Optional[Dict[str, Any]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Sample line metrics (time-series and profile) from solver state.

    Qt-free helper that mirrors the historical implementation in
    ``SWE2DWorkbenchStudioDialog._sample_line_metrics`` so the same logic can
    run in a worker thread.
    """
    if not sample_map:
        return [], []
    from swe2d.services.line_sampling_service import sample_line_metrics as _svc
    from swe2d.services.line_sampling_service import sample_line_aggregate_ts_row as _agg_svc

    g = float(gravity)
    h = np.asarray(h_s, dtype=np.float64)
    hu = np.asarray(hu_s, dtype=np.float64)
    hv = np.asarray(hv_s, dtype=np.float64)
    cell_bed = np.asarray(cell_solver_z, dtype=np.float64)
    if mesh_data is not None:
        _nx = np.asarray(mesh_data.get("node_x", np.empty(0)), dtype=np.float64).ravel()
        _ny = np.asarray(mesh_data.get("node_y", np.empty(0)), dtype=np.float64).ravel()
        _nc = np.asarray(mesh_data.get("cell_nodes", np.empty((0, 3), dtype=np.int32)))
        node_coords = np.column_stack([_nx, _ny]) if _nx.size > 0 and _ny.size > 0 else np.empty((0, 2), dtype=np.float64)
        _fo = mesh_data.get("cell_face_offsets")
        _fn = mesh_data.get("cell_face_nodes")
        if _fo is not None and _fn is not None:
            node_coords, cell_nodes = _triangulate_polygon_mesh(mesh_data)
        elif _nc.size > 0:
            cell_nodes = _nc.reshape((-1, 3)).astype(np.int32)
        else:
            cell_nodes = np.empty((0, 3), dtype=np.int32)
    else:
        node_coords = np.empty((0, 2), dtype=np.float64)
        cell_nodes = np.empty((0, 3), dtype=np.int32)

    out_ts: List[Dict[str, object]] = []
    out_prof: List[Dict[str, object]] = []
    for sm in sample_map:
        agg = _agg_svc(sm, h, hu, hv, cell_bed, h_min, g, t_accum)
        if agg is None:
            continue
        out_ts.append({
            "t_s": agg["t_s"], "line_id": agg["line_id"],
            "line_name": agg["line_name"],
            "depth_m": agg["depth_m"], "velocity_ms": agg["velocity_ms"],
            "wse_m": agg["wse_m"], "bed_m": agg["bed_m"],
            "flow_cms": agg["flow_cms"],
            "flow_cell_cms": agg["flow_cell_cms"],
            "flow_fv_cms": agg["flow_fv_cms"],
            "wet_frac": agg["wet_frac"], "fr": agg["fr"],
        })
        result = _svc(
            h=h, hu=hu, hv=hv, bed=cell_bed,
            node_coords=node_coords, cell_nodes=cell_nodes,
            line_xy=np.empty((0, 2), dtype=np.float64),
            h_min=h_min, timestep_s=float(t_accum), gravity=g,
            sample_map=sm,
        )
        station_arr = np.asarray(result.get("station_m", np.empty(0)), dtype=np.float64)
        if station_arr.size > 0:
            out_prof.append({
                "t_s": float(t_accum),
                "line_id": int(sm.get("line_id", -1)),
                "line_name": str(sm.get("line_name", "") or ""),
                "station_m": station_arr,
                "depth_m": np.asarray(result.get("depth_m", np.empty(0)), dtype=np.float64),
                "velocity_ms": np.asarray(result.get("velocity_ms", np.empty(0)), dtype=np.float64),
                "wse_m": np.asarray(result.get("wse_m", np.empty(0)), dtype=np.float64),
                "bed_m": np.asarray(result.get("bed_m", np.empty(0)), dtype=np.float64),
                "flow_qn": np.asarray(result.get("flow_qn", np.empty(0)), dtype=np.float64),
                "fr": np.asarray(result.get("froude", np.empty(0)), dtype=np.float64),
                "wet": np.asarray(result.get("wet", np.empty(0)), dtype=np.int32),
            })
        else:
            idx = agg["_idx"]
            hh = agg["_hh"]
            zb = agg["_zb"]
            vel = agg["_vel"]
            qn = agg["_qn"]
            wet = agg["_wet"]
            fr_arr = agg["_fr_arr"]
            sta = np.asarray(sm.get("station_m", np.arange(idx.size, dtype=np.float64)), dtype=np.float64)
            if sta.size != idx.size:
                sta = np.linspace(0.0, float(idx.size - 1), idx.size, dtype=np.float64)
            out_prof.append({
                "t_s": float(t_accum),
                "line_id": int(sm.get("line_id", -1)),
                "line_name": str(sm.get("line_name", "") or ""),
                "station_m": sta,
                "depth_m": np.asarray(hh, dtype=np.float64),
                "velocity_ms": np.asarray(vel, dtype=np.float64),
                "wse_m": np.asarray(hh + zb, dtype=np.float64),
                "bed_m": np.asarray(zb, dtype=np.float64),
                "flow_qn": np.asarray(qn, dtype=np.float64),
                "fr": np.asarray(fr_arr, dtype=np.float64),
                "wet": np.asarray(wet, dtype=np.int32),
            })
    return out_ts, out_prof
