from __future__ import annotations

"""Runtime service for headless/batch simulation execution without GUI."""

import logging
import os
import time
import urllib.parse
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    """env flag."""
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return bool(default)
    return raw not in {"0", "false", "no", "off"}


def _env_float(name: str, default: float) -> float:
    """env float."""
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return float(default)
    try:
        val = float(raw)
    except Exception:
        return float(default)
    return float(val) if np.isfinite(val) else float(default)


def _env_int(name: str, default: int) -> int:
    """env int."""
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def build_mesh_snapshot_rows(snapshot_timesteps: Sequence[Tuple[object, object, object, object]]) -> List[Dict[str, object]]:
    """Build mesh snapshot rows."""
    rows: List[Dict[str, object]] = []
    if not snapshot_timesteps:
        return rows

    # Pre-validate and extract flat arrays per snapshot
    snap_arrays: List[Tuple[float, np.ndarray, np.ndarray, np.ndarray, int]] = []
    total_rows = 0
    for snap in snapshot_timesteps:
        try:
            t_s, h, hu, hv = snap
            hh = np.asarray(h, dtype=np.float64).ravel()
            huu = np.asarray(hu, dtype=np.float64).ravel()
            hvv = np.asarray(hv, dtype=np.float64).ravel()
            n = min(hh.size, huu.size, hvv.size)
            if n == 0:
                continue
            snap_arrays.append((float(t_s), hh, huu, hvv, n))
            total_rows += n
        except Exception:
            logger.warning("Snapshot row extraction failed", exc_info=True)
            continue

    if not snap_arrays:
        return rows

    # Pre-allocate then fill via vectorized slicing
    rows = [{} for _ in range(total_rows)]
    offset = 0
    for t_s, hh, huu, hvv, n in snap_arrays:
        ts_val = float(t_s)
        for ci in range(n):
            rows[offset + ci] = {
                "t_s": ts_val,
                "cell_id": int(ci),
                "h": float(hh[ci]),
                "hu": float(huu[ci]),
                "hv": float(hvv[ci]),
            }
        offset += n

    return rows


def parse_obj_scale_value(raw_value: object) -> Tuple[float, float, float]:
    """parse obj scale value."""
    if raw_value is None:
        return (1.0, 1.0, 1.0)

    if isinstance(raw_value, (int, float, np.integer, np.floating)):
        s = float(raw_value)
        if not np.isfinite(s):
            raise ValueError("scale value must be finite")
        return (s, s, s)

    txt = str(raw_value).strip()
    if not txt:
        return (1.0, 1.0, 1.0)

    tokens = [p for p in txt.replace(",", " ").replace(";", " ").split() if p]
    if len(tokens) == 1:
        s = float(tokens[0])
        if not np.isfinite(s):
            raise ValueError("scale value must be finite")
        return (s, s, s)
    if len(tokens) >= 3:
        sx = float(tokens[0])
        sy = float(tokens[1])
        sz = float(tokens[2])
        if not (np.isfinite(sx) and np.isfinite(sy) and np.isfinite(sz)):
            raise ValueError("scale tuple must contain finite values")
        return (sx, sy, sz)

    raise ValueError("scale value must be a scalar or sx,sy,sz tuple")


def resolve_obj_model_path(
    *,
    raw_path: str,
    model_gpkg_path: str,
    project_file_path: str,
    module_dir: str,
    cwd: str,
) -> str:
    """resolve obj model path."""
    path_txt = str(raw_path or "").strip().strip('"').strip("'")
    if not path_txt:
        return ""

    if path_txt.lower().startswith("file://"):
        try:
            parsed = urllib.parse.urlparse(path_txt)
            uri_path = urllib.parse.unquote(str(parsed.path or ""))
            if uri_path:
                path_txt = uri_path
        except Exception as exc:
            logger.debug("[UI] failed to parse file:// URI %s: %s", path_txt, exc)

    candidates: List[str] = []
    if os.path.isabs(path_txt):
        candidates.append(path_txt)
    else:
        if model_gpkg_path:
            candidates.append(os.path.join(os.path.dirname(model_gpkg_path), path_txt))
        if project_file_path:
            candidates.append(os.path.join(os.path.dirname(project_file_path), path_txt))
        if module_dir:
            candidates.append(os.path.join(module_dir, path_txt))
        if cwd:
            candidates.append(os.path.join(cwd, path_txt))

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return os.path.abspath(candidate)

    if candidates:
        return os.path.abspath(candidates[0])
    return path_txt


def boundary_edge_owner_cells(
    *,
    mesh_data: Optional[Dict[str, object]],
    edge_n0: np.ndarray,
    edge_n1: np.ndarray,
) -> np.ndarray:
    """Return owner cells boundary edge info."""
    owners = np.full(int(edge_n0.size), -1, dtype=np.int32)
    if mesh_data is None:
        return owners

    edge_owner: Dict[Tuple[int, int], int] = {}
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
                key = (a, b) if a < b else (b, a)
                edge_owner[key] = ci if key not in edge_owner else -1
    else:
        tris = np.asarray(mesh_data["cell_nodes"], dtype=np.int32).reshape((-1, 3))
        for ci, tri in enumerate(tris):
            for k in range(3):
                a = int(tri[k])
                b = int(tri[(k + 1) % 3])
                key = (a, b) if a < b else (b, a)
                edge_owner[key] = int(ci) if key not in edge_owner else -1

    n = min(int(edge_n0.size), int(edge_n1.size))
    for i in range(n):
        a = int(edge_n0[i])
        b = int(edge_n1[i])
        key = (a, b) if a < b else (b, a)
        owner = int(edge_owner.get(key, -1))
        if owner >= 0:
            owners[i] = owner

    return owners


def _sample_coupling_object_metrics(cc, t_s: float, _h_s) -> list:
    """Return per-element coupling rows from the coupling controller."""
    rows = []
    if cc is None:
        return rows
    # Drainage nodes
    if hasattr(cc, "_gpu_node_depth") and cc._gpu_node_depth is not None:
        cfg = getattr(getattr(cc, "drainage", None), "cfg", None)
        if cfg is not None:
            for i, node in enumerate(getattr(cfg, "nodes", [])):
                if i < len(cc._gpu_node_depth):
                    depth = float(cc._gpu_node_depth[i])
                    rows.append({
                        "t_s": t_s,
                        "component": "drainage_node",
                        "metric": "depth",
                        "object_id": str(getattr(node, "node_id", str(i))),
                        "object_name": str(getattr(node, "node_id", str(i))),
                        "value": depth,
                    })
                    invert = float(getattr(node, "invert_elev", 0.0))
                    rows.append({
                        "t_s": t_s,
                        "component": "drainage_node",
                        "metric": "invert",
                        "object_id": str(getattr(node, "node_id", str(i))),
                        "object_name": str(getattr(node, "node_id", str(i))),
                        "value": invert,
                    })
    # Drainage links
    if hasattr(cc, "_gpu_link_flow") and cc._gpu_link_flow is not None:
        cfg = getattr(getattr(cc, "drainage", None), "cfg", None)
        if cfg is not None:
            for i, link in enumerate(getattr(cfg, "links", [])):
                if i < len(cc._gpu_link_flow):
                    flow = float(cc._gpu_link_flow[i])
                    from_id = getattr(link, "from_node_id", "")
                    to_id = getattr(link, "to_node_id", "")
                    rows.append({
                        "t_s": t_s,
                        "component": "drainage_link",
                        "metric": "flow",
                        "object_id": str(getattr(link, "link_id", str(i))),
                        "object_name": f"{from_id} -> {to_id}",
                        "value": flow,
                    })
                    link_len = float(getattr(link, "length", 0.0))
                    rows.append({
                        "t_s": t_s,
                        "component": "drainage_link",
                        "metric": "length",
                        "object_id": str(getattr(link, "link_id", str(i))),
                        "object_name": f"{from_id} -> {to_id}",
                        "value": link_len,
                    })
    # Structures (with culvert-specific diagnostics)
    if hasattr(cc, "_structures_cfg") and cc._structures_cfg:
        nb_flows = getattr(cc, "_last_structure_flows", None)
        from swe2d.extensions.extension_models import StructureType as _StructType

        # Culvert diagnostics: read from GPU kernel buffer.
        # ponytail: kernel computes values, Python reads them back — no re-computation.
        _culvert_diag_arr = None
        _n_cfg = len(cc._structures_cfg)
        struct_mod = getattr(cc, "structures", None)
        if struct_mod is None:
            try:
                from swe2d.extensions.structures import SWE2DStructureModule as _SM
                cfg = getattr(cc, "_structures_cfg", None)
                if cfg:
                    struct_mod = _SM(cfg)
            except Exception:
                struct_mod = None
        try:
            from swe2d.runtime.backend import load_swe2d_native_module
            _nm = load_swe2d_native_module()
            if hasattr(_nm, "swe2d_gpu_readback_culvert_diagnostics"):
                _nm.swe2d_gpu_device_sync()
                _nm.swe2d_gpu_ensure_culvert_diagnostics()
                _culvert_diag_arr = _nm.swe2d_gpu_readback_culvert_diagnostics()
                if _culvert_diag_arr is not None and _culvert_diag_arr.size > 0 and np.any(_culvert_diag_arr[:, 1:]):
                    logger.debug(
                        "culvert_diag: shape=%s nonzero_diag=%d n_cfg=%d",
                        _culvert_diag_arr.shape, int(np.sum(_culvert_diag_arr[:, 1:] != 0)), _n_cfg,
                    )
                elif _culvert_diag_arr is not None:
                    logger.debug(
                        "culvert_diag: shape=%s n_cfg=%d all zeros (expected before first solver step)",
                        _culvert_diag_arr.shape, _n_cfg,
                    )
        except Exception as _exc:
            logger.warning("culvert_diag readback failed: %s", _exc)
            _culvert_diag_arr = None

        for i, st in enumerate(cc._structures_cfg):
            sid = str(getattr(st, "structure_id", str(i)))
            sname = str(getattr(st, "name", sid))
            stype = getattr(st, "structure_type", None)
            is_culvert = (stype == _StructType.CULVERT)
            meta = getattr(st, "metadata", {})

            if nb_flows is not None and i < len(nb_flows):
                val = float(nb_flows[i])
            else:
                val = 0.0
            rows.append({
                "t_s": t_s,
                "component": "structure",
                "metric": "flow",
                "object_id": sid,
                "object_name": sname,
                "value": val,
            })

            if not is_culvert:
                continue

            if _culvert_diag_arr is not None and _culvert_diag_arr.ndim == 2 and i < _culvert_diag_arr.shape[0]:
                _row = _culvert_diag_arr[i]
                _METRICS = [
                    ("inlet_control_flow", 1),
                    ("outlet_control_flow", 2),
                    ("orifice_cap", 3),
                    ("manning_cap", 4),
                    ("embankment_flow", 5),
                    ("available_head_up", 6),
                    ("tailwater_depth", 7),
                ]
                for mname, col_idx in _METRICS:
                    mval = float(_row[col_idx]) if col_idx < len(_row) else 0.0
                    rows.append({
                        "t_s": t_s,
                        "component": "structure",
                        "metric": mname,
                        "object_id": sid,
                        "object_name": sname,
                        "value": mval,
                    })
    return rows


def execute_run_timestep_loop(
    *,
    wb: object,
    backend: object,
    runtime_step_executor: object,
    runtime_reporter: object,
    run_duration_s: float,
    t_accum: float,
    i: int,
    last_diag: Optional[Dict[str, object]],
    last_valid_cmax: float,
    last_valid_wse_res: float,
    dt_cfg: float,
    dt_request: float,
    stage_coupled_imex_enabled: bool,
    coupling_controller: object,
    dynamic_bc: bool,
    native_bc_forcing: bool,
    bc_n0: np.ndarray,
    bc_n1: np.ndarray,
    bc_tp: np.ndarray,
    bc_vl: np.ndarray,
    side_hydrographs: object,
    edge_hydrographs: object,
    rain_source_for_window_callback: object,
    cell_source_model_at_time_callback: object,
    accumulate_source_volume_model_callback: object,
    native_source_injection_mode: bool,
    accumulate_boundary_flux_volume_model_callback: object,
    sample_map: object,
    cell_solver_z: object,
    timing_totals_ms: Dict[str, float],
    timing_samples: int,
    next_snap_t: float,
    next_line_snap_t: float,
    next_coupling_snap_t: float,
    output_interval_s: float,
    line_output_interval_s: float,
    process_events_interval_s: float,
    last_process_events_wall: float,
    process_events_callback: object,
    h_min: float,
    uniform_inflow_velocity: bool = False,
    progress_callback: Optional[Callable] = None,
    perf_mode: bool = False,
) -> Dict[str, object]:

    # Pre-compute boundary edge lengths for uniform-velocity normalization.
    """execute run timestep loop."""
    _bc_edge_len = None
    if bc_n0.size > 0:
        try:
            _node_x = np.asarray(wb._mesh_data["node_x"], dtype=np.float64)
            _node_y = np.asarray(wb._mesh_data["node_y"], dtype=np.float64)
            _bc_edge_len = np.hypot(
                _node_x[bc_n1] - _node_x[bc_n0],
                _node_y[bc_n1] - _node_y[bc_n0],
            ).astype(np.float64)
        except Exception:
            logger.warning("Silent fallback in Exception handler", exc_info=True)
            _bc_edge_len = None

    def _make_uniform_velocity_cb() -> Optional[Callable[..., np.ndarray]]:
        """make uniform velocity cb."""
        if not hasattr(wb, "uniform_inflow_velocity_chk"):
            return None
        try:
            if not uniform_inflow_velocity:
                return None
        except Exception:
            return None
        if _bc_edge_len is None or backend is None:
            return None
        edge_cells = backend.boundary_edge_cells()
        if edge_cells is None:
            return None

        def _normalize(bc_vl_step, bc_tp_step, backend_obj):
            """normalize."""
            try:
                h_arr, _, _ = backend_obj.get_state()
                edge_h = np.asarray(h_arr, dtype=np.float64)[edge_cells]
                from swe2d.boundary_and_forcing.bc_logic import normalize_inflow_to_uniform_velocity as _norm
                return _norm(bc_vl_step, bc_tp_step, edge_h, _bc_edge_len)
            except Exception:
                return bc_vl_step

        return _normalize

    _uniform_velocity_cb = _make_uniform_velocity_cb()

    while float(t_accum) < float(run_duration_s):
        if bool(getattr(wb, "_cancel_requested", False)):
            break

        step_wall_t0 = time.perf_counter()
        step_ms = 0.0
        coupling_ms = 0.0
        source_ms = 0.0
        state_ms = 0.0
        bc_ms = 0.0
        ui_ms = 0.0

        step_result = runtime_step_executor.execute_step(
            backend=backend,
            t_accum=t_accum,
            last_diag=last_diag,
            dt_cfg=dt_cfg,
            dt_request=dt_request,
            stage_coupled_imex_enabled=stage_coupled_imex_enabled,
            coupling_controller=coupling_controller,
            dynamic_bc=dynamic_bc,
            native_bc_forcing=native_bc_forcing,
            bc_n0=bc_n0,
            bc_n1=bc_n1,
            bc_tp=bc_tp,
            bc_vl=bc_vl,
            side_hydrographs=side_hydrographs,
            edge_hydrographs=edge_hydrographs,
            apply_timeseries_bc_values_callback=wb._apply_timeseries_bc_values,
            distribute_total_flow_to_unit_q_callback=wb._distribute_total_flow_to_unit_q,
            rain_source_for_window_callback=rain_source_for_window_callback,
            cell_source_model_at_time_callback=cell_source_model_at_time_callback,
            accumulate_source_volume_model_callback=accumulate_source_volume_model_callback,
            apply_external_sources_callback=wb._apply_external_sources,
            native_source_injection_mode=native_source_injection_mode,
            uniform_inflow_velocity_normalize_callback=_uniform_velocity_cb,
        )
        last_diag = step_result["last_diag"]
        dt_used = float(step_result["dt_used"])
        rain_src = step_result["rain_src"]
        step_ms += float(step_result["step_ms"])
        coupling_ms += float(step_result["coupling_ms"])
        source_ms += float(step_result["source_ms"])
        state_ms += float(step_result["state_ms"])
        bc_ms += float(step_result["bc_ms"])

        if not perf_mode and bc_n0.size > 0:
            _t_bc_acc = time.perf_counter()
            _t_bc_diag = 0.0
            _t_bc_accu = 0.0
            if dynamic_bc:
                _t0 = time.perf_counter()
                bc_tp_flux, bc_vl_flux = wb._apply_timeseries_bc_values(
                    bc_n0,
                    bc_n1,
                    bc_tp,
                    bc_vl,
                    side_hydrographs,
                    t_accum,
                    edge_hydrographs,
                )
                _t_bc_diag += (time.perf_counter() - _t0) * 1000.0
                _t0 = time.perf_counter()
                bc_vl_flux = wb._distribute_total_flow_to_unit_q(
                    bc_n0,
                    bc_n1,
                    bc_tp_flux,
                    bc_vl_flux,
                    bc_tp,
                    side_hydrographs,
                    edge_hydrographs,
                )
                _t_bc_diag += (time.perf_counter() - _t0) * 1000.0
            else:
                bc_tp_flux = bc_tp
                _t0 = time.perf_counter()
                bc_vl_flux = wb._distribute_total_flow_to_unit_q(
                    bc_n0,
                    bc_n1,
                    bc_tp_flux,
                    bc_vl,
                    bc_tp,
                    side_hydrographs,
                    edge_hydrographs,
                )
                _t_bc_diag += (time.perf_counter() - _t0) * 1000.0
            _t0 = time.perf_counter()
            accumulate_boundary_flux_volume_model_callback(dt_used, bc_tp_flux, bc_vl_flux)
            _t_bc_accu += (time.perf_counter() - _t0) * 1000.0
            bc_ms += (time.perf_counter() - _t_bc_acc) * 1000.0
            # [BC_DIAG] removed for performance

        report_result = runtime_reporter.process_step(
            backend=backend,
            t_accum=t_accum,
            dt_used=dt_used,
            last_diag=last_diag,
            last_valid_cmax=last_valid_cmax,
            last_valid_wse_res=last_valid_wse_res,
            sample_map=sample_map,
            cell_solver_z=cell_solver_z,
            coupling_controller=coupling_controller,
            rain_src=rain_src,
            state_ms=state_ms,
            ui_ms=ui_ms,
            step_wall_t0=step_wall_t0,
            step_ms=step_ms,
            coupling_ms=coupling_ms,
            source_ms=source_ms,
            bc_ms=bc_ms,
            timing_totals_ms=timing_totals_ms,
            timing_samples=timing_samples,
            i=i,
            run_duration_s=run_duration_s,
            next_snap_t=next_snap_t,
            next_line_snap_t=next_line_snap_t,
            next_coupling_snap_t=next_coupling_snap_t,
            output_interval_s=output_interval_s,
            line_output_interval_s=line_output_interval_s,
            process_events_interval_s=process_events_interval_s,
            last_process_events_wall=last_process_events_wall,
            h_min=h_min,
            length_unit_name=wb._length_unit_name,
            results_data=wb._results_data,
            sample_line_metrics_callback=wb._sample_line_metrics,
            sample_coupling_object_metrics_callback=_sample_coupling_object_metrics,
            process_events_callback=process_events_callback,
            set_progress_callback=progress_callback,
            log_callback=wb._log,
            perf_mode=perf_mode,
        )
        t_accum = float(report_result["t_accum"])
        last_valid_cmax = float(report_result["last_valid_cmax"])
        last_valid_wse_res = float(report_result["last_valid_wse_res"])
        next_snap_t = float(report_result["next_snap_t"])
        next_line_snap_t = float(report_result["next_line_snap_t"])
        next_coupling_snap_t = float(report_result["next_coupling_snap_t"])
        last_process_events_wall = float(report_result["last_process_events_wall"])
        timing_samples = int(report_result["timing_samples"])
        i = int(report_result["i"])

    return {
        "t_accum": float(t_accum),
        "i": int(i),
        "last_diag": last_diag,
        "last_valid_cmax": float(last_valid_cmax),
        "last_valid_wse_res": float(last_valid_wse_res),
        "next_snap_t": float(next_snap_t),
        "next_line_snap_t": float(next_line_snap_t),
        "next_coupling_snap_t": float(next_coupling_snap_t),
        "last_process_events_wall": float(last_process_events_wall),
        "timing_samples": int(timing_samples),
    }
