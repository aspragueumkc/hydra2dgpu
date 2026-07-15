"""Shared RunContext builder — bridges CLI JSON (and future GUI dict) to RunContext.

The CLI path follows the same architecture as the GUI: load layers from
GPKG via QgsVectorLayer (or raw sqlite3 for non-spatial data), then call
the identical service functions the GUI uses.  The only difference is that
the CLI supplies data from file paths / JSON values instead of QComboBoxes.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import datetime
import threading
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from swe2d.workbench.workers.run_context import RunContext
from swe2d.cli.gpkg_adapter import query_mesh_from_gpkg

logger = logging.getLogger(__name__)


def _norm_key(a, b):
    return (a, b) if a < b else (b, a)


# ── GUI-widget → RunContext parameter name mapping ─────────────────────────────
# Module-level so it can be shared by build_run_context_from_dict and
# widget_state_to_flat_params (for extracting flat params from versioned
# widget_state format saved by collect_workbench_widget_state).
WIDGET_TO_RC: Dict[str, str] = {
    # Spin boxes
    "n_mann_spin": "n_mann",
    "cfl_spin": "cfl",
    "h_min_spin": "h_min",
    "dt_spin": "dt_cfg",
    "initial_dt_spin": "initial_dt",
    "cfl_lambda_cap_spin": "cfl_lambda_cap",
    "gpu_diag_sync_interval_spin": "gpu_diag_sync_interval_steps",
    "max_rel_depth_increase_spin": "max_rel_depth_increase",
    "max_source_depth_step_spin": "source_depth_step_cap",
    "max_source_rate_spin": "source_rate_cap",
    "source_cfl_beta_spin": "source_cfl_beta",
    "source_max_substeps_spin": "source_max_substeps",
    "shallow_damping_depth_spin": "shallow_damping_depth",
    "depth_cap_spin": "depth_cap",
    "momentum_cap_min_speed_spin": "momentum_cap_min_speed",
    "momentum_cap_celerity_mult_spin": "momentum_cap_celerity_mult",
    "max_inv_area_spin": "max_inv_area",
    "tiny_wet_cell_threshold_spin": "tiny_wet_cell_threshold",
    "front_flux_damping_spin": "front_flux_damping",
    "open_bc_relax_spin": "open_bc_relaxation",
    "gpu_diag_sync_interval_raw": "gpu_diag_sync_interval_steps",
    # Checkboxes  (gui name → rc name)
    "adaptive_cfl_dt_chk": "adaptive_cfl_dt",
    "source_true_subcycling_chk": "source_true_subcycling",
    "source_imex_split_chk": "source_imex_split",
    "use_redistribution_chk": "use_redistribution",
    "swe2d_perf_mode_chk": "swe2d_perf_mode",
    "culvert_face_flux_chk": "culvert_face_flux_enabled",
    "enable_cuda_graphs_chk": "cuda_graphs_enabled",
    "active_set_hysteresis_chk": "active_set_hysteresis",
    "inflow_progressive_chk": "inflow_progressive",
    # Storage / results checkboxes
    "save_max_only_chk": "save_max_only",
    "save_mesh_results_to_gpkg_chk": "save_mesh_results",
    "save_line_results_to_gpkg_chk": "save_line_results",
    "save_coupling_results_to_gpkg_chk": "save_coupling_results",
    "save_run_log_to_gpkg_chk": "save_run_log",
    # Raw storage checkbox widget names (on _model_tab_view)
    "save_mesh_chk": "save_mesh_results",
    "save_line_chk": "save_line_results",
    "save_coupling_chk": "save_coupling_results",
    "save_log_chk": "save_run_log",
    # Combos
    "reconstruction_combo": "reconstruction_mode",
    "temporal_order_combo": "temporal_scheme",
    "tiny_mode_combo": "tiny_mode",
    "degen_mode_combo": "degen_mode",
    "drainage_gpu_method_combo": "drainage_gpu_method_mode",
    "culvert_solver_mode_combo": "culvert_solver_mode",
    "bridge_stacked_coupling_mode_combo": "bridge_stacked_coupling_mode",
    "default_bc_type_combo": "default_bc_type",
    # LineEdits (stored as strings, parsed by _v or callers)
    "run_time_edit": "run_duration_s",
    "output_interval_edit": "output_interval_s",
    "results_gpkg_path_edit": "results_gpkg_path",
    "results_table_name_edit": "results_table_name",
    # Alternate keys (CLI JSON may use these)
    "duration_s": "run_duration_s",
    "dt_max": "dt_cfg",
}


# ── Combo display-text → param name mapping ──────────────────────────────────────
# collect_workbench_widget_state captures currentText() for combos as
# "{combo_name}_text" entries.  This map translates them to the
# RunContext param names that to_replay_json / from_replay_json expect.
_COMBO_TEXT_TO_PARAM: Dict[str, str] = {
    "reconstruction_combo_text": "reconstruction_name",
    "temporal_order_combo_text": "temporal_scheme_name",
}


def widget_state_to_flat_params(
    widget_state: dict,
    *,
    mesh_gpkg: str = "",
    mesh_name: str = "",
) -> dict:
    """Extract flat RunContext-params from versioned widget_state.

    ``collect_workbench_widget_state`` returns
    ``{"version": 1, "widgets": {"n_mann_spin": {"type": "...", "value": 0.035}, ...}}``.
    This function converts it to flat ``{rc_param_name: value}`` dict using
    ``WIDGET_TO_RC`` (for widget values) and ``_COMBO_TEXT_TO_PARAM`` (for
    combo display-text entries), so it can be stored in the ``params`` block
    of the ``swe2d-replay/1`` schema that ``build_run_context_from_dict`` reads.

    If ``mesh_gpkg`` and ``mesh_name`` are provided, the ``units`` block is
    also computed from the mesh CRS and returned as a side-effect in
    ``flat["_units_block"]``.
    """
    widgets = widget_state.get("widgets", {}) if isinstance(widget_state, dict) else {}
    flat: Dict[str, Any] = {}

    def _parse_duration(val: Any) -> Any:
        """Parse a time string (HH:MM or fraction-of-an-hour) to float seconds.

        If the value is not a recognisable time string it is returned unchanged,
        allowing callers to apply their own interpretation.
        """
        if not isinstance(val, str):
            return val
        s = val.strip()
        if ":" in s:
            try:
                parts = s.split(":")
                return (float(parts[0]) + float(parts[1]) / 60.0) * 3600.0
            except (ValueError, IndexError):
                return val
        return val

    for wname, winfo in widgets.items():
        if not isinstance(winfo, dict):
            continue
        value = winfo.get("value")
        if value is None:
            continue
        # Map GUI widget name → RunContext param name
        rc_name = WIDGET_TO_RC.get(wname, wname)
        # Time-edit widgets store HH:MM strings — convert to seconds
        flat[rc_name] = _parse_duration(value)

    # Handle combo display-text entries (e.g. reconstruction_combo_text → reconstruction_name)
    for text_key, param_name in _COMBO_TEXT_TO_PARAM.items():
        winfo = widgets.get(text_key)
        if isinstance(winfo, dict):
            text_val = winfo.get("value")
            if text_val is not None:
                flat[param_name] = text_val

    # Compute units from mesh CRS if mesh_gpkg is available
    if mesh_gpkg and mesh_name:
        from swe2d.cli.gpkg_adapter import query_mesh_from_gpkg
        from swe2d import units as _u2
        try:
            md = query_mesh_from_gpkg(mesh_gpkg, mesh_name)
            if md is not None:
                crs_wkt = str(md.get("crs_wkt", "") or "")
                si_m_per_model = _u2.si_m_per_model_from_wkt(crs_wkt) if crs_wkt else 1.0
                flat["_units_block"] = {
                    "length_unit_name": "ft" if si_m_per_model < 0.5 else "m",
                    "length_scale_si_to_model": si_m_per_model,
                    "rain_mm_to_model_depth": si_m_per_model,
                    "rain_rate_si_to_model": si_m_per_model,
                    "flow_si_to_model": si_m_per_model,
                }
        except Exception:
            pass

    return flat


def build_run_context_from_dict(
    p: Dict[str, Any],
    *,
    mesh_gpkg: str = "",
    results_gpkg: str = "",
    cancel_event: Optional[threading.Event] = None,
) -> RunContext:
    """Build a fully-populated RunContext from a flat parameter dict.

    This is the single bridge between a parameter dict (from CLI JSON, replay
    JSON, or a future GUI dict layer) and the RunContext that
    ``SimulationWorker._execute()`` expects.

    Parameters
    ----------
    p : dict
        Flat parameter dict with optionally nested ``params``, ``mesh``,
        ``results``, ``units``, ``drainage``, etc. sub-dicts.  Top-level
        keys override nested ones.
    mesh_gpkg : str
        Path to the model GPKG.  May also be in p["mesh_gpkg"] or
        p["mesh"]["gpkg_path"].
    results_gpkg : str
        Path to the results GPKG.  May also be in p["results_gpkg_path"] or
        p["results"]["results_gpkg_path"].
    cancel_event : threading.Event or None

    Returns
    -------
    RunContext
    """
    # ── Resolve mesh GPKG and name ─────────────────────────────────────
    mesh = p.get("mesh", {}) or {}
    _mesh_gpkg = mesh_gpkg or str(p.get("mesh_gpkg", "") or mesh.get("gpkg_path", ""))
    mesh_name = str(p.get("mesh_name", "") or mesh.get("mesh_name", "") or p.get("mesh", ""))
    if isinstance(mesh_name, dict):
        mesh_name = str(mesh_name.get("mesh_name", ""))
    if not _mesh_gpkg or not os.path.isfile(_mesh_gpkg):
        raise FileNotFoundError(f"Mesh GPKG not found: {_mesh_gpkg}")
    if not mesh_name:
        raise ValueError("mesh_name required in params")

    md = query_mesh_from_gpkg(_mesh_gpkg, mesh_name)
    if md is None:
        raise ValueError(f"Mesh '{mesh_name}' not found in {_mesh_gpkg}")

    # ── CRS / unit system ──────────────────────────────────────────────
    crs_wkt = str(md.get("crs_wkt", "") or mesh.get("crs_wkt", "") or "")
    from swe2d import units as _u
    si_m_per_model = _u.si_m_per_model_from_wkt(crs_wkt) if crs_wkt else 1.0
    _u.configure(si_m_per_model)

    units_cfg = p.get("units", {}) or {}
    params = p.get("params", {}) or {}
    results_cfg = p.get("results", {}) or {}

    # ── GUI-widget → RunContext parameter name mapping ──────────────────
    # collect_params() saves GUI widget names (n_mann_spin, cfl_spin,
    # reconstruction_combo, temporal_order_combo, etc.) but _v() and the
    # RunContext constructor expect RunContext names (n_mann, cfl,
    # reconstruction_mode, temporal_scheme, etc.).  Apply the mapping so
    # both old GUI-saved configs and CLI JSON formats work.
    # Scan both top-level p and nested params for GUI widget names;
    # copy their values to RunContext-named top-level keys.
    for _gui_key, _rc_key in WIDGET_TO_RC.items():
        if _gui_key in p and _rc_key not in p:
            p[_rc_key] = p[_gui_key]
        if isinstance(params, dict) and _gui_key in params and _rc_key not in p:
            p[_rc_key] = params[_gui_key]

    # If this is a saved config (widget_state from GPKG), its _data_sources
    # key (nested inside widget_state) OR top-level data_sources key contains
    # the layer references that the CLI JSON format expects at top-level.
    # Merge them in so builder code finds bc_lines/drainage/etc.
    data_sources = p.get("_data_sources") or p.get("data_sources") or {}
    for _ds_key in ("bc_lines", "drainage", "hyetograph", "rain_cn", "sample_lines",
                    "structures", "infiltration_method", "storm_areas", "internal_flow_sources"):
        if _ds_key in data_sources and _ds_key not in p:
            p[_ds_key] = data_sources[_ds_key]

    # Helper: resolve param from top-level, then nested params, then default
    def _v(key: str, default: Any = None) -> Any:
        if key in p:
            return p[key]
        return params.get(key, default)

    # ── Resolve results GPKG ───────────────────────────────────────────
    _results_gpkg = results_gpkg or str(
        p.get("results_gpkg_path", "") or results_cfg.get("results_gpkg_path", "")
    )

    # ── Run identity ───────────────────────────────────────────────────
    run_id = str(p.get("run_id", "") or p.get("id", "") or
                 datetime.datetime.now().astimezone().strftime("swe2d_%Y%m%dT%H%M%S%z"))
    run_wallclock_start = str(p.get("run_wallclock_start", ""))
    run_log_start_idx = int(p.get("run_log_start_idx", 0))

    # ── Pre-compute cell geometry callbacks from mesh ──────────────────
    # (needed before backend build for coupling controller setup)
    node_x = md["node_x"]
    node_y = md["node_y"]
    node_z = md["node_z"]
    cell_nodes = md["cell_nodes"]
    face_offsets = md.get("cell_face_offsets")
    face_nodes = md.get("cell_face_nodes")
    n_cells = int(cell_nodes.shape[0]) if face_offsets is None else int(face_offsets.size - 1)

    from swe2d.services.mesh_computation_service import (
        mesh_cell_areas as _svc_cell_areas,
        mesh_cell_min_bed as _svc_cell_min_bed,
        mesh_cell_centroids as _svc_cell_centroids,
    )
    _cell_areas = _svc_cell_areas(md)
    _cell_bed = _svc_cell_min_bed(md)

    def _mesh_cell_areas():
        return _cell_areas

    def _mesh_cell_min_bed():
        return _cell_bed

    def _mesh_cell_centroids():
        return _svc_cell_centroids(md)

    # ── BC arrays ──────────────────────────────────────────────────────
    # The mesh BLOB only provides boundary edge topology (n0, n1).
    # BC type/values are computed from config: default_bc_type applied to
    # all boundary edges, then bc_lines layer overrides specific edges.
    bc_n0 = md.get("bc_edge_node0", np.empty(0, dtype=np.int32))
    bc_n1 = md.get("bc_edge_node1", np.empty(0, dtype=np.int32))
    bc_relax = np.zeros(bc_n0.size, dtype=np.float64)

    # Check p first (mapped from GUI name default_bc_type_combo), then params
    default_bc_type = int(p.get("default_bc_type") or params.get("default_bc_type", 1))
    if bc_n0.size > 0:
        md_for_bc = {"node_x": node_x, "node_y": node_y}
        from swe2d.services.mesh_computation_service import default_bc_for_edges
        bc_tp, bc_vl = default_bc_for_edges(md_for_bc, bc_n0, bc_n1, default_bc_type=default_bc_type)
    else:
        bc_tp = np.empty(0, dtype=np.int32)
        bc_vl = np.empty(0, dtype=np.float64)

    # Override from bc_lines config using the same apply_bc_layer_overrides_qgis
    # code path as the GUI — not a separate node-snapping implementation.
    bc_cfg = p.get("bc_lines") or {}
    if isinstance(bc_cfg, dict) and bc_cfg.get("table") and bc_n0.size > 0:
        try:
            from qgis.core import QgsApplication as _QgsApp
            if _QgsApp.instance() is not None:
                from swe2d.boundary_and_forcing.boundary_qgis_adapter import apply_bc_layer_overrides_from_gpkg
                bc_tp, bc_vl, bc_relax = apply_bc_layer_overrides_from_gpkg(
                    gpkg_path=bc_cfg.get("gpkg", _mesh_gpkg),
                    table_name=bc_cfg["table"],
                    mesh_data={"node_x": node_x, "node_y": node_y},
                    edge_n0=bc_n0,
                    edge_n1=bc_n1,
                    bc_type=bc_tp,
                    bc_val=bc_vl,
                    default_relax=float(_v("open_bc_relaxation", 0.0)),
                    log_fn=logger.info,
                )
        except Exception as exc:
            logger.warning("bc_lines override from GPKG failed: %s", exc)

    # ── Hydrograph BCs from bc_lines ──────────────────────────────────
    side_hydrographs: Dict[str, Any] = {}
    edge_hydrographs: Dict[int, Any] = {}
    if isinstance(bc_cfg, dict) and bc_cfg.get("table") and bc_cfg.get("hydrograph_table"):
        from swe2d.cli.gpkg_adapter import load_hydrograph_edge_data
        try:
            bc_conn = sqlite3.connect(bc_cfg.get("gpkg", _mesh_gpkg))
            hyd_conn = sqlite3.connect(bc_cfg.get("gpkg", _mesh_gpkg))
            try:
                edge_hg_data = load_hydrograph_edge_data(
                    bc_conn, bc_cfg["table"],
                    hyd_conn, bc_cfg.get("hydrograph_table", "SWE2D_Hydrographs"),
                    node_x, node_y, bc_n0, bc_n1, logger,
                )
                if edge_hg_data:
                    edge_hydrographs = edge_hg_data
                    logger.info("Loaded %d edge hydrographs from GPKG", len(edge_hydrographs))
            finally:
                hyd_conn.close()
                bc_conn.close()
        except Exception as exc:
            logger.warning("Failed to load hydrographs: %s", exc)

    # ── Thiessen forcing (rain + CN) ───────────────────────────────────
    thiessen_forcing = None
    hyeto = p.get("hyetograph") or {}
    if isinstance(hyeto, dict) and hyeto.get("table") and hyeto.get("gauge_layer"):
        from swe2d.cli.gpkg_adapter import build_forced_thiessen_from_gpkg
        h_gpkg = hyeto.get("gpkg", _mesh_gpkg)
        htable = hyeto["table"]
        gtable = hyeto["gauge_layer"]
        cntable = p.get("rain_cn") or {}
        cn_table = cntable.get("table") if isinstance(cntable, dict) else None
        cn_field = cntable.get("cn_field", "cn") if isinstance(cntable, dict) else "cn"
        infil = str(p.get("infiltration_method", "") or params.get("infiltration_method", "scs_cn"))
        h_conn = sqlite3.connect(h_gpkg)
        try:
            thiessen_forcing = build_forced_thiessen_from_gpkg(
                h_conn, n_cells, node_x, node_y, cell_nodes,
                cell_face_offsets=face_offsets,
                hyetograph_table=htable,
                gauge_table=gtable,
                cn_table=cn_table,
                cn_field=cn_field,
                infiltration_method=infil,
            )
        finally:
            h_conn.close()

    # ── Internal flow sources ──────────────────────────────────────────
    internal_flow_forcing = None
    ifs_cfg = p.get("internal_flow_sources") or {}
    if isinstance(ifs_cfg, dict) and ifs_cfg.get("table"):
        # TODO: wire up when needed — for now pass None
        logger.info("Internal flow sources requested but not yet wired in builder")

    # ── Drainage network config ────────────────────────────────────────
    pipe_network_cfg = None
    drainage_cfg = p.get("drainage") or {}
    if isinstance(drainage_cfg, dict) and "nodes_layer" in drainage_cfg:
        _dgpkg = drainage_cfg.get("gpkg") or _mesh_gpkg
        try:
            from qgis.core import QgsVectorLayer
            nl_uri = f"{_dgpkg}|layername={drainage_cfg['nodes_layer']}"
            ll_uri = f"{_dgpkg}|layername={drainage_cfg['links_layer']}"
            node_layer = QgsVectorLayer(nl_uri, "drain_nodes", "ogr")
            link_layer = QgsVectorLayer(ll_uri, "drain_links", "ogr")
            inlet_layer = None
            if drainage_cfg.get("inlets_layer"):
                il_uri = f"{_dgpkg}|layername={drainage_cfg['inlets_layer']}"
                inlet_layer = QgsVectorLayer(il_uri, "drain_inlets", "ogr")
            node_inlet_layer = None
            if drainage_cfg.get("node_inlets_layer"):
                ni_uri = f"{_dgpkg}|layername={drainage_cfg['node_inlets_layer']}"
                node_inlet_layer = QgsVectorLayer(ni_uri, "drain_node_inlets", "ogr")

            if node_layer.isValid() and link_layer.isValid():
                from swe2d.mesh.mesh_runtime_logic import nearest_cell_index, mesh_cell_centroids
                cell_cx, cell_cy = mesh_cell_centroids(md)

                def _nearest_cell(x, y):
                    return nearest_cell_index(x, y, cell_cx, cell_cy)

                from swe2d.workbench.services.pipe_network_service import build_pipe_network_config
                pipe_network_cfg = build_pipe_network_config(
                    mesh_data=md,
                    node_layer=node_layer,
                    link_layer=link_layer,
                    inlet_layer=inlet_layer,
                    node_inlet_layer=node_inlet_layer,
                    cell_min_bed=_cell_bed,
                    nearest_cell_fn=_nearest_cell,
                    gravity=_u.gravity(),
                    config={
                        "solver_mode": int(_v("culvert_solver_mode", 0)),
                        "coupling_substeps": int(_v("coupling_substeps", 1)),
                        "gpu_method": str(_v("drainage_gpu_method_mode", "step")),
                        "head_deadband": float(_v("head_deadband", 0.001)),
                        "dynamic_relaxation": float(_v("dynamic_relaxation", 0.7)),
                        "implicit_iters": int(_v("implicit_iters", 3)),
                        "implicit_relax": float(_v("implicit_relax", 0.8)),
                    },
                    log_fn=logger.info,
                )
        except Exception as exc:
            logger.warning("drainage config from GPKG failed: %s", exc)

    # ── Hydraulic structures config ────────────────────────────────────
    hydraulic_structures_cfg = None
    structures_data = p.get("structures") or {}
    if isinstance(structures_data, dict):
        from swe2d.extensions.structures import build_structures_config_from_json
        hydraulic_structures_cfg = build_structures_config_from_json(structures_data, n_cells)

    # ── Bridge stacked plans ───────────────────────────────────────────
    bridge_stacked_plans: List[Any] = []
    try:
        from swe2d.runtime.bridge_stacked_runtime import build_bridge_stacked_plans_for_runtime
        bridge_stacked_plans = build_bridge_stacked_plans_for_runtime(
            md, hydraulic_structures_cfg, log_fn=logger.info,
        )
    except Exception:
        pass

    # ── Coupling SOA (packed mesh→structure mapping) ───────────────────
    coupling_soa = None
    try:
        from swe2d.runtime.coupling import pack_coupling_soa
        if pack_coupling_soa is not None and (pipe_network_cfg is not None or hydraulic_structures_cfg is not None):
            coupling_soa = pack_coupling_soa(
                n_cells=n_cells,
                pipe_network=pipe_network_cfg,
                hydraulic_structures=hydraulic_structures_cfg,
            )
    except Exception:
        pass

    # ── n_mann_cell from mesh ──────────────────────────────────────────
    n_mann_cell = md.get("n_mann_cell")

    # ── Initial state ──────────────────────────────────────────────────
    _h0_user = p.get("h0") or params.get("h0")
    if _h0_user is not None:
        h0 = np.asarray(_h0_user, dtype=np.float64)
        if h0.size != n_cells:
            raise ValueError(f"h0 has {h0.size} elements but mesh has {n_cells} cells")
    else:
        h0 = np.zeros(n_cells, dtype=np.float64)

    # ── Assemble RunContext ────────────────────────────────────────────
    return RunContext(
        run_id=run_id,
        run_wallclock_start=run_wallclock_start,
        run_log_start_idx=run_log_start_idx,
        results_gpkg_path=_results_gpkg,
        model_gpkg_path=_mesh_gpkg,
        mesh_name=mesh_name,
        mesh_crs_wkt=crs_wkt,

        # Time
        run_duration_s=float(_v("run_duration_s", _v("duration_s", 3600.0))),
        output_interval_s=float(_v("output_interval_s", float(_v("run_duration_s", 3600.0)))),
        dt_cfg=float(_v("dt_cfg", _v("dt_max", 0.2))),
        dt_request=float(_v("dt_request", 0.05)),
        dt_fixed=float(_v("dt_fixed", -1.0)),
        initial_dt=float(_v("initial_dt", 0.05)),
        adaptive_cfl_dt=bool(_v("adaptive_cfl_dt", False)),

        # Solver modes
        reconstruction_mode=int(_v("reconstruction_mode", 0)),
        reconstruction_name=str(_v("reconstruction_name", "")),
        temporal_scheme=_v("temporal_scheme", 2),
        temporal_scheme_name=str(_v("temporal_scheme_name", "")),
        solver_backend_mode=str(_v("solver_backend_mode", "gpu")).strip().lower(),
        coupling_loop_mode=str(_v("coupling_loop_mode", "cuda")).strip().lower(),
        drainage_solver_backend_mode=str(_v("drainage_solver_backend_mode", "gpu")).strip().lower(),
        drainage_gpu_method_mode=str(_v("drainage_gpu_method_mode", "step")).strip().lower(),
        culvert_solver_mode=int(_v("culvert_solver_mode", 0)),
        cuda_graphs_enabled=bool(_v("cuda_graphs_enabled", False)),
        bridge_cuda_coupling=bool(_v("bridge_cuda_coupling", False)),
        bridge_stacked_coupling_mode=str(_v("bridge_stacked_coupling_mode", "phase3_spatial")),
        culvert_face_flux_mode=str(_v("culvert_face_flux_mode", "off")),

        # Numerics
        gravity=float(_v("gravity", _u.gravity())),
        k_mann=float(_v("k_mann", _u.manning_factor())),
        n_mann=float(_v("n_mann", 0.035)),
        cfl=float(_v("cfl", 0.45)),
        h_min=float(_v("h_min", 1.0e-4)),
        max_inv_area=float(_v("max_inv_area", 0.0)),
        cfl_lambda_cap=float(_v("cfl_lambda_cap", 0.0)),
        momentum_cap_min_speed=float(_v("momentum_cap_min_speed", 0.0)),
        momentum_cap_celerity_mult=float(_v("momentum_cap_celerity_mult", 0.0)),
        depth_cap=float(_v("depth_cap", 0.0)),
        max_rel_depth_increase=float(_v("max_rel_depth_increase", 0.0)),
        shallow_damping_depth=float(_v("shallow_damping_depth", 0.0)),
        source_cfl_beta=float(_v("source_cfl_beta", 0.0)),
        source_max_substeps=int(_v("source_max_substeps", 1)),
        source_rate_cap=float(_v("source_rate_cap", 0.0)),
        source_depth_step_cap=float(_v("source_depth_step_cap", 0.0)),
        source_true_subcycling=bool(_v("source_true_subcycling", False)),
        source_imex_split=bool(_v("source_imex_split", False)),
        gpu_diag_sync_interval_steps=int(_v("gpu_diag_sync_interval_steps", 0)),
        tiny_mode=int(_v("tiny_mode", 0)),
        tiny_wet_cell_threshold=int(_v("tiny_wet_cell_threshold", 0)),
        degen_mode=int(_v("degen_mode", 0)),
        front_flux_damping=float(_v("front_flux_damping", 0.0)),
        open_bc_relaxation=float(_v("open_bc_relaxation", 0.0)),
        active_set_hysteresis=bool(_v("active_set_hysteresis", False)),
        use_redistribution=bool(_v("use_redistribution", False)),
        inflow_progressive=bool(_v("inflow_progressive", False)),
        uniform_inflow_enabled=bool(_v("uniform_inflow_enabled", False)),
        rain_update_interval_s=float(_v("rain_update_interval_s", 60.0)),

        # Mesh arrays
        node_x=node_x,
        node_y=node_y,
        node_z=node_z,
        cell_nodes=cell_nodes,
        face_offsets=face_offsets,
        face_nodes=face_nodes,
        bc_n0=bc_n0,
        bc_n1=bc_n1,
        bc_tp=bc_tp,
        bc_vl=bc_vl,
        bc_relax=bc_relax,
        side_hydrographs=side_hydrographs,
        edge_hydrographs=edge_hydrographs,
        edge_group_overrides={},
        h0=h0,
        hu0=np.zeros(n_cells, dtype=np.float64),
        hv0=np.zeros(n_cells, dtype=np.float64),
        n_mann_cell=n_mann_cell,
        cell_areas=_cell_areas,
        cell_centroids=np.empty((0, 2), dtype=np.float64),

        # Forcing / coupling
        rain_rate_model=0.0,
        internal_flow_forcing=internal_flow_forcing,
        cell_source_model=None,
        thiessen_forcing=thiessen_forcing,
        pipe_network_cfg=pipe_network_cfg,
        hydraulic_structures_cfg=hydraulic_structures_cfg,
        bridge_stacked_plans=bridge_stacked_plans,
        coupling_soa=coupling_soa,

        # Storage flags
        save_mesh_results=bool(_v("save_mesh_results", results_cfg.get("save_mesh_results", True))),
        save_line_results=bool(_v("save_line_results", results_cfg.get("save_line_results", False))),
        save_coupling_results=bool(_v("save_coupling_results", results_cfg.get("save_coupling_results", False))),
        save_run_log=bool(_v("save_run_log", results_cfg.get("save_run_log", False))),
        save_max_only=bool(_v("save_max_only", results_cfg.get("save_max_only", False))),

        # Units
        length_unit_name=str(units_cfg.get("length_unit_name", _u.length_unit_name())),
        length_scale_si_to_model=float(units_cfg.get("length_scale_si_to_model", _u.si_m_per_model())),
        rain_mm_to_model_depth=float(units_cfg.get("rain_mm_to_model_depth", _u.si_m_per_model())),
        rain_rate_si_to_model=float(units_cfg.get("rain_rate_si_to_model", _u.si_m_per_model())),
        flow_si_to_model=float(units_cfg.get("flow_si_to_model", _u.model_per_si_m() ** 3)),

        # Callbacks — pre-computed from mesh data (no Qt dependency)
        mesh_cell_areas=_mesh_cell_areas,
        mesh_cell_min_bed=_mesh_cell_min_bed,
        mesh_cell_centroids=_mesh_cell_centroids,
        apply_timeseries_bc_values=lambda *a, **k: None,
        distribute_total_flow_to_unit_q=lambda *a, **k: None,
        apply_external_sources=lambda *a, **k: None,
        build_line_sampling_map=lambda: None,
        internal_flow_source_cms_at_time=lambda *a, **k: None,

        # Misc
        sample_map_data=[],
        inflow_progressive_enabled=bool(_v("inflow_progressive", False)),
        edge_groups={},
        cancel_event=cancel_event or threading.Event(),
    )
