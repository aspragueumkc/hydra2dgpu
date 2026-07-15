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


def _copy_overlay_cell_data_from_coupling(cc, results_data, t_s, dt):
    """Copy rain overlay cell arrays into results_data at output intervals.

    Called from _sample_coupling_object_metrics on every output-interval step.
    Rain arrays come from C++ via readback_coupling_state(); rain rate is
    recomputed from the Python forcing. Manning's n and curve number are
    populated separately from the mesh by Path 5.
    """
    if results_data is None:
        return

    # Rain CN state — from C++ kernel via readback ─────────────────────────
    try:
        state = cc.readback_coupling_state()
    except Exception:
        state = {}

    rain_cum = state.get("rain_cum_mm")
    rain_excess = state.get("rain_excess_cum_mm")
    if rain_cum is not None and rain_cum.size > 0:
        results_data.overlay_cell_cumulative_rain = np.ascontiguousarray(rain_cum, dtype=np.float64)
    if rain_excess is not None and rain_excess.size > 0:
        results_data.overlay_cell_cumulative_excess = np.ascontiguousarray(rain_excess, dtype=np.float64)

    # Rain rate (m/s) still comes from Python forcing — it is computed here
    # and uploaded to the C++ kernel each step; read back via Python for overlay.
    forcing = getattr(cc, "_rain_forcing", None)
    if forcing is None and hasattr(cc, "drainage"):
        forcing = getattr(cc.drainage, "_rain_forcing", None)
    if forcing is not None:
        rate_mps, _stats = forcing.step_net_rainfall_mps(t_s, t_s + dt, mutate_state=False)
        if rate_mps is not None:
            arr = np.asarray(rate_mps, dtype=np.float64)
            if arr.size > 0:
                results_data.overlay_cell_rainfall_rate = np.ascontiguousarray(arr)

    results_data.append_overlay_field_snapshot(t_s)


def _sample_coupling_object_metrics(cc, t_s: float, dt: float, _h_s, _results_data=None) -> list:
    """Return per-element coupling rows from the coupling controller.

    Uses ``cc.readback_coupling_state()`` for drainage node depths, link flows,
    and structure flows — a small D2H readback at output intervals (not per-step).

    The optional ``_results_data`` parameter is used internally to copy live
    rain overlay arrays on output-interval steps. Manning/CN arrays are
    populated separately from the mesh.
    """
    rows = []
    if cc is None:
        return rows
    # Copy live rain overlay cell arrays on output-interval steps.
    if _results_data is not None:
        _copy_overlay_cell_data_from_coupling(cc, _results_data, t_s, dt)
    state = cc.readback_coupling_state()
    cfg = getattr(getattr(cc, "drainage", None), "cfg", None)
    if cfg is not None:
        for i, node in enumerate(getattr(cfg, "nodes", [])):
            node_id = str(getattr(node, "node_id", str(i)))
            if i < len(state["node_depth"]):
                depth = float(state["node_depth"][i])
                rows.append({
                    "t_s": t_s,
                    "component": "drainage_node",
                    "metric": "depth",
                    "object_id": node_id,
                    "object_name": str(getattr(node, "node_id", str(i))),
                    "value": depth,
                })
                invert = float(getattr(node, "invert_elev", 0.0))
                rows.append({
                    "t_s": t_s,
                    "component": "drainage_node",
                    "metric": "invert",
                    "object_id": node_id,
                    "object_name": str(getattr(node, "node_id", str(i))),
                    "value": invert,
                })
        for li, link in enumerate(getattr(cfg, "links", [])):
            link_id = str(getattr(link, "link_id", str(li)))
            if li < len(state["link_flow"]):
                flow = float(state["link_flow"][li])
                from_id = getattr(link, "from_node_id", "")
                to_id = getattr(link, "to_node_id", "")
                rows.append({
                    "t_s": t_s,
                    "component": "drainage_link",
                    "metric": "flow",
                    "object_id": link_id,
                    "object_name": f"{from_id} -> {to_id}",
                    "value": flow,
                })
                link_len = float(getattr(link, "length", 0.0))
                rows.append({
                    "t_s": t_s,
                    "component": "drainage_link",
                    "metric": "length",
                    "object_id": link_id,
                    "object_name": f"{from_id} -> {to_id}",
                    "value": link_len,
                })
        # Drainage cell: per-pipe-cell velocity, depth, flow, head.
        # cell_sub_idx and cell_owner_link now come directly from C++ readback.
        cell_velocity = state.get("cell_velocity")
        cell_depth = state.get("cell_depth")
        cell_flow = state.get("cell_flow")
        cell_head = state.get("cell_head")
        cell_owner_link = state.get("cell_owner_link")
        cell_sub_idx = state.get("cell_sub_idx")  # directly from C++
        # Per-cell geometry for profile rendering (crown = invert + cell_width for circular)
        cell_invert = state.get("cell_invert")
        cell_width = state.get("cell_width")

        if cell_velocity is not None and cell_owner_link is not None and cfg is not None:
            links = getattr(cfg, "links", [])
            n_cells = len(cell_velocity)
            if cell_sub_idx is not None and len(cell_sub_idx) == n_cells:
                # Fast path: use C++-sourced cell_sub_idx and cell_owner_link directly.
                for c in range(n_cells):
                    li = int(cell_owner_link[c])
                    if li < 0 or li >= len(links):
                        continue
                    link = links[li]
                    link_id = str(getattr(link, "link_id", str(li)))
                    sub_idx = int(cell_sub_idx[c])
                    for metric, arr in [("velocity", cell_velocity), ("depth", cell_depth),
                                       ("flow", cell_flow), ("head", cell_head)]:
                        rows.append({
                            "t_s": t_s,
                            "component": "drainage_cell",
                            "metric": metric,
                            "object_id": f"{link_id}#{sub_idx}",
                            "object_name": f"{link_id} cell {sub_idx}",
                            "value": float(arr[c]),
                            # Per-cell geometry — constant per sub-cell, same for all 4 metrics
                            "cell_invert": float(cell_invert[c]) if cell_invert is not None and c < len(cell_invert) else 0.0,
                            "cell_width": float(cell_width[c]) if cell_width is not None and c < len(cell_width) else 1.0,
                        })
            else:
                # Fallback: derive sub_idx from enumeration — indicates a bug if
                # pipe-link coupling is active and sub_idx is missing from C++.
                log_fn = getattr(cc, "_log", None) or (lambda _m: None)
                log_fn(f"[drainage_cell] WARNING: cell_sub_idx not returned by C++ readback "
                       f"(cell_sub_idx={cell_sub_idx is not None}) — using enumeration fallback. "
                       f"This may indicate a bug if pipe-link coupling is active.")
                for c in range(n_cells):
                    li = int(cell_owner_link[c])
                    if li < 0 or li >= len(links):
                        continue
                    link = links[li]
                    link_id = str(getattr(link, "link_id", str(li)))
                    for metric, arr in [("velocity", cell_velocity), ("depth", cell_depth), ("flow", cell_flow), ("head", cell_head)]:
                        rows.append({
                            "t_s": t_s,
                            "component": "drainage_cell",
                            "metric": metric,
                            "object_id": f"{link_id}#{c}",
                            "object_name": f"{link_id} cell {c}",
                            "value": float(arr[c]),
                            "cell_invert": float(cell_invert[c]) if cell_invert is not None and c < len(cell_invert) else 0.0,
                            "cell_width": float(cell_width[c]) if cell_width is not None and c < len(cell_width) else 1.0,
                        })

        # Cell index metrics — one row per element (value does not vary with time).
        # Drainage nodes: cell index from associated inlet/pipe_end/outfall exchanges
        node_cell: Dict[str, int] = {}
        for ex in getattr(cfg, "inlets", []) or []:
            nid = str(getattr(ex, "node_id", ""))
            cid = int(getattr(ex, "cell_id", -1))
            if nid and cid >= 0:
                node_cell[nid] = cid
        for ex in getattr(cfg, "pipe_ends", []) or []:
            nid = str(getattr(ex, "node_id", ""))
            cid = int(getattr(ex, "cell_id", -1))
            if nid and cid >= 0:
                node_cell[nid] = cid
        for ex in getattr(cfg, "outfalls", []) or []:
            nid = str(getattr(ex, "node_id", ""))
            cid = int(getattr(ex, "cell_id", -1))
            if nid and cid >= 0:
                node_cell[nid] = cid
        for node in getattr(cfg, "nodes", []) or []:
            nid = str(getattr(node, "node_id", ""))
            if nid in node_cell:
                rows.append({
                    "t_s": t_s,
                    "component": "drainage_node",
                    "metric": "cell",
                    "object_id": nid,
                    "object_name": nid,
                    "value": float(node_cell[nid]),
                })
        # Inlets: cell index
        for i, inlet in enumerate(getattr(cfg, "inlets", []) or []):
            rows.append({
                "t_s": t_s,
                "component": "drainage_inlet",
                "metric": "cell",
                "object_id": str(getattr(inlet, "inlet_id", str(i))),
                "object_name": str(getattr(inlet, "inlet_id", str(i))),
                "value": float(getattr(inlet, "cell_id", -1)),
            })
        # Outfalls: cell index
        for i, outfall in enumerate(getattr(cfg, "outfalls", []) or []):
            rows.append({
                "t_s": t_s,
                "component": "drainage_outfall",
                "metric": "cell",
                "object_id": str(getattr(outfall, "outfall_id", str(i))),
                "object_name": str(getattr(outfall, "outfall_id", str(i))),
                "value": float(getattr(outfall, "cell_id", -1)),
            })
        # Pipe ends: cell index
        for i, pe in enumerate(getattr(cfg, "pipe_ends", []) or []):
            rows.append({
                "t_s": t_s,
                "component": "drainage_pipe_end",
                "metric": "cell",
                "object_id": str(getattr(pe, "pipe_end_id", str(i))),
                "object_name": str(getattr(pe, "pipe_end_id", str(i))),
                "value": float(getattr(pe, "cell_id", -1)),
            })
    # Structures (with culvert-specific diagnostics)
    if hasattr(cc, "_structures_cfg") and cc._structures_cfg:
        nb_flows = state["struct_flow"] if state["struct_flow"].size > 0 else getattr(cc, "_last_structure_flows", None)
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


def build_coupling_keys(cc) -> Tuple[List[Tuple[str, str, str]], Dict[Tuple[str, str, str], str]]:
    """Return the fixed (component, object_id, metric) keys and object names for the coupling controller.

    Returns
    -------
    keys : list of (component, object_id, metric) tuples
    object_names : dict mapping key → object_name string
    """
    keys: List[Tuple[str, str, str]] = []
    object_names: Dict[Tuple[str, str, str], str] = {}
    if cc is None:
        return keys, object_names
    cfg = getattr(getattr(cc, "drainage", None), "cfg", None)
    if cfg is not None:
        for node in getattr(cfg, "nodes", []):
            node_id = str(getattr(node, "node_id", ""))
            keys.append(("drainage_node", node_id, "depth"))
            keys.append(("drainage_node", node_id, "invert"))
            object_names[("drainage_node", node_id, "depth")] = node_id
            object_names[("drainage_node", node_id, "invert")] = node_id
        for link in getattr(cfg, "links", []):
            link_id = str(getattr(link, "link_id", ""))
            from_id = getattr(link, "from_node_id", "")
            to_id = getattr(link, "to_node_id", "")
            name = f"{from_id} -> {to_id}"
            keys.append(("drainage_link", link_id, "flow"))
            keys.append(("drainage_link", link_id, "length"))
            object_names[("drainage_link", link_id, "flow")] = name
            object_names[("drainage_link", link_id, "length")] = name
    if hasattr(cc, "_structures_cfg") and cc._structures_cfg:
        from swe2d.extensions.extension_models import StructureType as _StructType
        for st in cc._structures_cfg:
            sid = str(getattr(st, "structure_id", ""))
            sname = str(getattr(st, "name", sid))
            keys.append(("structure", sid, "flow"))
            object_names[("structure", sid, "flow")] = sname
            if getattr(st, "structure_type", None) == _StructType.CULVERT:
                for mname, _ in [
                    ("inlet_control_flow", 1), ("outlet_control_flow", 2),
                    ("orifice_cap", 3), ("manning_cap", 4),
                    ("embankment_flow", 5), ("available_head_up", 6),
                    ("tailwater_depth", 7),
                ]:
                    keys.append(("structure", sid, mname))
                    object_names[("structure", sid, mname)] = sname
    return keys, object_names


def build_pipe_cell_keys(cc) -> List[Tuple[str, int, str]]:
    """Return the (link_id, cell_sub_idx, metric) keys for per-pipe-cell storage.

    Sub-cell counts are derived from link_length / max_cell_length, matching the
    C++ subdivision formula used in pipe1d_init.
    """
    import math as _math
    keys: List[Tuple[str, int, str]] = []
    if cc is None:
        return keys
    dsoa = getattr(cc, "_drainage_soa", None)
    cfg = getattr(getattr(cc, "drainage", None), "cfg", None)
    if dsoa is None or cfg is None:
        return keys
    link_lengths = getattr(dsoa, "link_length", [])
    mcl = float(getattr(dsoa, "max_cell_length", 0.0))
    links = getattr(cfg, "links", [])
    if len(link_lengths) == 0 or not links:
        return keys
    for li, link in enumerate(links):
        link_id = str(getattr(link, "link_id", str(li)))
        L = float(link_lengths[li]) if li < len(link_lengths) else 0.0
        n_sub = 1
        if mcl > 0.0 and L > 0.0:
            n_sub = max(1, int(_math.ceil(L / mcl)))
        for sub_idx in range(n_sub):
            for metric in ("velocity", "depth", "flow", "head"):
                keys.append((link_id, sub_idx, metric))
    return keys


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
    timing_totals_ms: Dict[str, float],
    timing_samples: int,
    next_snap_t: float,
    output_interval_s: float,
    process_events_interval_s: float,
    last_process_events_wall: float,
    process_events_callback: Optional[Callable[[], None]] = None,
    h_min: float,
    uniform_enabled: bool = False,
    progress_callback: Optional[Callable] = None,
    perf_mode: bool = False,
    line_names_by_id: Optional[Dict[int, str]] = None,
    line_ids_ordered: Optional[List[int]] = None,
) -> Dict[str, object]:

    """execute run timestep loop."""
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
            coupling_controller=coupling_controller,
            rain_source_for_window_callback=rain_source_for_window_callback,
            cell_source_model_at_time_callback=cell_source_model_at_time_callback,
            accumulate_source_volume_model_callback=accumulate_source_volume_model_callback,
            apply_external_sources_callback=wb._apply_external_sources,
            native_source_injection_mode=native_source_injection_mode,
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

        # Wrap _sample_coupling_object_metrics to pass results_data for
        # live rain/Manning/CN overlay array population on output-interval steps.
        _results_data = wb._results_data

        def _metrics_cb(cc, t_s, _h_s):
            return _sample_coupling_object_metrics(cc, t_s, dt_used, _h_s, _results_data)

        report_result = runtime_reporter.process_step(
            backend=backend,
            t_accum=t_accum,
            dt_used=dt_used,
            last_diag=last_diag,
            last_valid_cmax=last_valid_cmax,
            last_valid_wse_res=last_valid_wse_res,
            sample_map=sample_map,
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
            output_interval_s=output_interval_s,
            process_events_interval_s=process_events_interval_s,
            last_process_events_wall=last_process_events_wall,
            h_min=h_min,
            length_unit_name=wb._length_unit_name,
            results_data=_results_data,
            sample_coupling_object_metrics_callback=_metrics_cb,
            process_events_callback=process_events_callback,
            set_progress_callback=progress_callback,
            log_callback=wb._log,
            perf_mode=perf_mode,
            line_names_by_id=line_names_by_id,
            line_ids_ordered=line_ids_ordered,
        )
        t_accum = float(report_result["t_accum"])
        last_valid_cmax = float(report_result["last_valid_cmax"])
        last_valid_wse_res = float(report_result["last_valid_wse_res"])
        next_snap_t = float(report_result["next_snap_t"])
        last_process_events_wall = float(report_result["last_process_events_wall"])
        timing_totals_ms = report_result["timing_totals_ms"]
        timing_samples = int(report_result["timing_samples"])
        i = int(report_result["i"])

    return {
        "t_accum": float(t_accum),
        "i": int(i),
        "last_diag": last_diag,
        "last_valid_cmax": float(last_valid_cmax),
        "last_valid_wse_res": float(last_valid_wse_res),
        "next_snap_t": float(next_snap_t),
        "last_process_events_wall": float(last_process_events_wall),
        "timing_totals_ms": timing_totals_ms,
        "timing_samples": timing_samples,
    }
