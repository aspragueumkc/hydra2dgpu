"""Qt-free application of runtime external source terms."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from swe2d.boundary_and_forcing.bc_logic import (
    EdgeHydrographMap,
    Hydrograph,
    _bc_side_classification,
    distribute_total_flow_to_unit_q,
)


def _apply_external_sources_logic(
    backend,
    dt_step: float,
    rain_rate_model,
    cell_source_model: Optional[np.ndarray],
    coupled_source_rate: Optional[np.ndarray],
    mesh_cell_areas: Optional[np.ndarray],
    max_source_rate: float,
    h_min: float,
    max_rel_depth_increase: float,
    max_source_depth_step: float,
    shallow_damping_depth: float,
    momentum_cap_min_speed: float,
    momentum_cap_celerity_mult: float,
) -> None:
    """Apply external source terms without touching Qt widgets.

    All parameters are forwarded to
    :func:`swe2d.boundary_and_forcing.runtime_source_logic.apply_external_sources`.
    Some parameters (``h_min``, ``max_rel_depth_increase``, ``max_source_depth_step``,
    ``shallow_damping_depth``, ``momentum_cap_min_speed``,
    ``momentum_cap_celerity_mult``) are currently passed through for callers that
    already construct the argument list; the underlying function may use them in
    future source-term limiting.
    """
    from swe2d.boundary_and_forcing.runtime_source_logic import (
        apply_external_sources as _logic,
    )

    _logic(
        backend=backend,
        dt_step=dt_step,
        rain_rate_model=rain_rate_model,
        cell_source_model=cell_source_model,
        coupled_source_rate=coupled_source_rate,
        mesh_cell_areas=mesh_cell_areas,
        max_source_rate=max_source_rate,
        h_min=h_min,
        max_rel_depth_increase=max_rel_depth_increase,
        max_source_depth_step=max_source_depth_step,
        shallow_damping_depth=shallow_damping_depth,
        momentum_cap_min_speed=momentum_cap_min_speed,
        momentum_cap_celerity_mult=momentum_cap_celerity_mult,
    )


def _distribute_total_flow_to_unit_q_logic(
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
    edge_hydrographs: EdgeHydrographMap = None,
    edge_groups: Optional[Dict[int, str]] = None,
    *,
    _side_idx: Optional[np.ndarray] = None,
    _edge_len: Optional[np.ndarray] = None,
    _edge_z: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Distribute total flow BC values to unit discharge per edge without Qt.

    All parameters are forwarded to
    :func:`swe2d.boundary_and_forcing.bc_logic.distribute_total_flow_to_unit_q`.
    Optional pre-computed geometry invariants (``_side_idx``, ``_edge_len``,
    ``_edge_z``) are used when all three are supplied; otherwise they are
    computed from the mesh.
    """
    if _side_idx is None or _edge_len is None or _edge_z is None:
        side_idx, edge_len, edge_z, *_ = _bc_side_classification(
            edge_n0, edge_n1, node_x, node_y, node_z,
        )
    else:
        side_idx, edge_len, edge_z = _side_idx, _edge_len, _edge_z
    return distribute_total_flow_to_unit_q(
        edge_n0=edge_n0, edge_n1=edge_n1,
        bc_type_step=bc_type_step, bc_val_step=bc_val_step,
        bc_type_template=bc_type_template,
        side_hydrographs=side_hydrographs,
        node_x=node_x, node_y=node_y, node_z=node_z,
        progressive=progressive,
        ts_flow_code=102,
        edge_hydrographs=edge_hydrographs,
        edge_groups=edge_groups,
        _side_idx=side_idx,
        _edge_len=edge_len,
        _edge_z=edge_z,
    )


def _sample_line_metrics_logic(
    sample_map,
    t_accum,
    h_s,
    hu_s,
    hv_s,
    cell_solver_z,
    gravity,
    h_min,
    mesh_data,
):
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
        # Handle polygon meshes — triangulate if cell_face_offsets is available
        _fo = mesh_data.get("cell_face_offsets")
        _fn = mesh_data.get("cell_face_nodes")
        if _fo is not None and _fn is not None:
            # Polygon mesh: triangulate via centroid-vertex fan
            _fo_arr = np.asarray(_fo, dtype=np.int32).ravel()
            _fn_arr = np.asarray(_fn, dtype=np.int32).ravel()
            n_cells = _fo_arr.size - 1
            total_tris = sum(int(_fo_arr[i+1] - _fo_arr[i]) for i in range(n_cells))
            cell_nodes = np.empty((total_tris, 3), dtype=np.int32)
            idx = 0
            nx_ext = np.concatenate([_nx, np.zeros(n_cells)])
            ny_ext = np.concatenate([_ny, np.zeros(n_cells)])
            for c in range(n_cells):
                s, e = int(_fo_arr[c]), int(_fo_arr[c+1])
                cx = float(np.mean(_nx[_fn_arr[s:e]]))
                cy = float(np.mean(_ny[_fn_arr[s:e]]))
                nx_ext[len(_nx) + c] = cx
                ny_ext[len(_ny) + c] = cy
                for i in range(s, e):
                    na, nb = int(_fn_arr[i]), int(_fn_arr[(i+1-s)%(e-s)+s])
                    cell_nodes[idx] = [na, nb, len(_nx) + c]
                    idx += 1
            node_coords = np.column_stack([nx_ext, ny_ext])
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
