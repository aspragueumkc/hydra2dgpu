"""Headless runner: execute a simulation from JSON params + GPKG without QGIS.

Usage:
    from swe2d.cli.headless_runner import execute_run
    results = execute_run(mesh_gpkg, params)
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from typing import Any, Callable, Dict, List, Optional


logger = logging.getLogger(__name__)

import numpy as np

from swe2d.cli.gpkg_adapter import (
    build_forced_thiessen_from_gpkg,
    query_bc_arrays,
    query_mesh_from_gpkg,
    query_sample_lines_from_qgis,
)
from swe2d.mesh.mesh_runtime_logic import mesh_cell_centroids


def _si_m_per_model_from_wkt(wkt: str) -> float:
    """Extract SI meters per model unit from CRS WKT's CS LENGTHUNIT section."""
    try:
        idx = wkt.find("CS[")
        if idx < 0:
            return 1.0
        cs_section = wkt[idx:]
        lu_idx = cs_section.find("LENGTHUNIT[")
        if lu_idx < 0:
            return 1.0
        unit_part = cs_section[lu_idx + len("LENGTHUNIT["):]
        comma_idx = unit_part.find(",")
        if comma_idx < 0:
            return 1.0
        return float(unit_part[comma_idx + 1:].split("]")[0].strip())
    except Exception:
        return 1.0
from swe2d.runtime.backend import SWE2DBackend, build_mesh as shared_build_mesh


def _parse_params(param_source: str) -> Dict[str, Any]:
    """Load params from a JSON string or file path."""
    s = str(param_source).strip()
    if os.path.isfile(s):
        with open(s) as f:
            return json.load(f)
    return json.loads(s)


def _atomic_write_json(path: str, payload: dict) -> None:
    """Atomically write a JSON dict to a file (write-then-rename)."""
    import tempfile
    fd, tmp = tempfile.mkstemp(suffix=".json", dir=os.path.dirname(path) or None)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except Exception as _e:

            logger.warning("[ERROR] Exception in headless_runner.py: %s", _e)


def execute_run(
    mesh_gpkg: str,
    params: Dict[str, Any],
    results_gpkg: Optional[str] = None,
    progress_callback: Optional[Callable[[float, Dict[str, Any]], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    status_file_path: Optional[str] = None,
    status_interval_s: float = 5.0,
) -> Dict[str, Any]:
    """Run a simulation from GPKG-stored mesh + JSON params.

    If ``status_file_path`` is set, a JSON status file is written every
    ``status_interval_s`` seconds during the simulation loop.  This allows
    a separate process (e.g. the QGIS UI) to check progress without polling
    subprocess stdout/stderr or parsing the results GPKG.

    The status file contains:
        {"step": int, "t": float, "dt": float, "wet_cells": int,
         "elapsed_s": float, "status": "running"|"done"|"error",
         "error": str|null}

    Returns dict with keys: h, hu, hv, max_results (optional), diags.
    """
    if not os.path.isfile(mesh_gpkg):
        raise FileNotFoundError(f"Mesh GPKG not found: {mesh_gpkg}")

    # Load mesh
    p = params
    mesh_name = p.get("mesh", "")
    if not mesh_name:
        raise ValueError("'mesh' key required in params JSON")
    mesh_data = query_mesh_from_gpkg(mesh_gpkg, mesh_name)
    if mesh_data is None:
        raise ValueError(f"Mesh '{mesh_name}' not found in {mesh_gpkg}")

    # Build backend — shared helper handles polygon mesh logic
    # Configure unit system from mesh CRS so gravity + manning are correct
    from swe2d import units as _u
    crs_wkt = mesh_data.get("crs_wkt", "")
    si_m_per_model = _si_m_per_model_from_wkt(crs_wkt) if crs_wkt else 1.0
    _u.configure(si_m_per_model)
    logger.info("Units configured: si_m_per_model=%.6f, gravity=%.4f",
                 si_m_per_model, _u.gravity())
    backend = SWE2DBackend()
    shared_build_mesh(
        backend,
        node_x=mesh_data["node_x"],
        node_y=mesh_data["node_y"],
        node_z=mesh_data["node_z"],
        cell_nodes=mesh_data["cell_nodes"],
        cell_face_offsets=mesh_data.get("cell_face_offsets"),
        cell_face_nodes=mesh_data.get("cell_face_nodes"),
        bc_edge_node0=mesh_data.get("bc_edge_node0"),
        bc_edge_node1=mesh_data.get("bc_edge_node1"),
        bc_edge_type=mesh_data.get("bc_edge_type"),
        bc_edge_val=mesh_data.get("bc_edge_val"),
    )
    nnodes = int(mesh_data["node_x"].size)
    ncells = int(backend.n_cells)

    # ── Resolve GPKG connection for each data source ─────────────────
    # Each data-source key is a dict with "table" + optional "gpkg".
    # If "gpkg" is omitted the table lives in mesh_gpkg.
    def _open_cfg(cfg, default_gpkg=mesh_gpkg):
        """Open a connection for *cfg* (dict with optional "gpkg" key).
        Returns (table_name, conn_or_None)."""
        if not cfg or not isinstance(cfg, dict):
            return ("", None)
        tbl = cfg.get("table", "")
        gpkg = cfg.get("gpkg", default_gpkg)
        return tbl, sqlite3.connect(gpkg)

    # Read BC arrays — handles both pre-split edge tables and geometry-based tables
    from swe2d.cli.gpkg_adapter import query_bc_arrays as _query_bc
    bc: Dict[str, np.ndarray] = {}
    bc_table, bc_conn = _open_cfg(p.get("bc_lines"))
    if bc_conn is not None:
        try:
            if bc_table:
                bc = _query_bc(
                    bc_conn, bc_table,
                    node_x=mesh_data.get("node_x"),
                    node_y=mesh_data.get("node_y"),
                )
        finally:
            bc_conn.close()
    bc_n0 = bc.get("bc_edge_node0", np.empty(0, dtype=np.int32))
    bc_n1 = bc.get("bc_edge_node1", np.empty(0, dtype=np.int32))
    bc_tp = bc.get("bc_edge_type", np.empty(0, dtype=np.int32))
    bc_vl = bc.get("bc_edge_val", np.empty(0, dtype=np.float64))

    # Build Thiessen forcing from GPKG (if configured)
    thiessen_forcing = None
    hyetograph_cfg = p.get("hyetograph")
    if hyetograph_cfg is not None and isinstance(hyetograph_cfg, dict):
        htable = hyetograph_cfg.get("table", "")
        gtable = hyetograph_cfg.get("gauge_layer", "")
        cntable = p.get("rain_cn")
        cn_table = None
        if isinstance(cntable, dict):
            cn_table = cntable.get("table")
        if htable and gtable:
            th_conn = _open_cfg(hyetograph_cfg)[1]
            try:
                thiessen_forcing = build_forced_thiessen_from_gpkg(
                    th_conn, ncells,
                    mesh_data["node_x"], mesh_data["node_y"],
                    mesh_data["cell_nodes"],
                    cell_face_offsets=mesh_data.get("cell_face_offsets"),
                    hyetograph_table=htable,
                    gauge_table=gtable,
                    cn_table=cn_table,
                    cn_field=cntable.get("cn_field", "cn") if isinstance(cntable, dict) else "cn",
                    infiltration_method=p.get("infiltration_method", "scs_cn"),
                )
            finally:
                th_conn.close()

    # Build run options
    rp = p.get("params", {})
    thiessen_kwargs = {"thiessen_forcing": thiessen_forcing} if thiessen_forcing else {}

    # Initialize solver — read ALL params the UI provides, not just a subset
    from swe2d.runtime.backend import BCType

    h0 = np.zeros(ncells, dtype=np.float64)
    backend.initialize(
        h0=h0,
        k_mann=float(rp.get("k_mann", 1.0)),
        n_mann=float(rp.get("n_mann", 0.035)),
        h_min=float(rp.get("h_min", 1e-4)),
        cfl=float(rp.get("cfl", 0.45)),
        dt_max=float(rp.get("dt_max", 0.2)),
        dt_initial=float(rp.get("initial_dt", 0.05)),
        max_inv_area=float(rp.get("max_inv_area", 1e6)),
        cfl_lambda_cap=float(rp.get("cfl_lambda_cap", 1e6)),
        momentum_cap_min_speed=float(rp.get("momentum_cap_min_speed", 50.0)),
        momentum_cap_celerity_mult=float(rp.get("momentum_cap_celerity_mult", 20.0)),
        depth_cap=float(rp.get("depth_cap", 1e6)),
        max_rel_depth_increase=float(rp.get("max_rel_depth_increase", 2.0)),
        shallow_damping_depth=float(rp.get("shallow_damping_depth", 1e-4)),
        extreme_rain_mode=bool(rp.get("extreme_rain_mode", False)),
        source_cfl_beta=float(rp.get("source_cfl_beta", 0.25)),
        source_max_substeps=int(rp.get("source_max_substeps", 16)),
        source_rate_cap=float(rp.get("source_rate_cap", 0.0)),
        source_depth_step_cap=float(rp.get("source_depth_step_cap", 0.0)),
        source_true_subcycling=bool(rp.get("source_true_subcycling", False)),
        source_imex_split=bool(rp.get("source_imex_split", False)),
        gpu_diag_sync_interval_steps=int(rp.get("gpu_diag_sync_interval_steps", 100)),
        tiny_mode=int(rp.get("tiny_mode", 0)),
        tiny_wet_cell_threshold=int(rp.get("tiny_wet_cell_threshold", 2000)),
        front_flux_damping=float(rp.get("front_flux_damping", 0.5)),
        active_set_hysteresis=bool(rp.get("active_set_hysteresis", True)),
        degen_mode=int(rp.get("degen_mode", 0)),
        spatial_discretization=int(rp.get("spatial_scheme", 0)),
        temporal_scheme=int(rp.get("temporal_scheme", 2)),
        enable_shallow_front_recon_fallback=bool(rp.get("enable_shallow_front_recon_fallback", False)),
    )

    # Configure boundary conditions if BC arrays were found
    if bc_n0.size > 0:
        try:
            backend.set_boundary_conditions(bc_n0, bc_n1, bc_tp, bc_vl)
        except Exception as _e:
            logger.warning("Failed to set boundary conditions: %s", _e)

    # Configure native rain if Thiessen forcing is present
    if thiessen_forcing is not None:
        from swe2d.runtime.runtime_setup_configurator import SWE2DRunSetupConfigurator
        cfg = SWE2DRunSetupConfigurator()
        mm_to_model = 1.0e-3
        try:
            cfg_res = cfg.configure_native_rain_cn_forcing(
                backend=backend,
                thiessen_forcing=thiessen_forcing,
                mm_to_model_depth=mm_to_model,
            )
        except Exception as _e:
            logger.warning("Failed to configure native rain-CN forcing: %s", _e)

    # ── Coupling controller (drainage + structures) ──────────────────
    from swe2d.cli.gpkg_adapter import (
        build_drainage_config_from_json,
        build_structures_config_from_json,
        read_drainage_config_from_gpkg,
    )
    drainage_data = p.get("drainage")
    if isinstance(drainage_data, dict) and "nodes_layer" in drainage_data:
        _dgpkg = drainage_data.get("gpkg") or mesh_gpkg
        _dconn = sqlite3.connect(_dgpkg)
        try:
            drainage_inline = read_drainage_config_from_gpkg(
                _dconn,
                drainage_data["nodes_layer"],
                drainage_data["links_layer"],
                mesh_data["node_x"],
                mesh_data["node_y"],
                mesh_data["cell_nodes"],
                cell_face_offsets=mesh_data.get("cell_face_offsets"),
                inlets_table=drainage_data.get("inlets_layer"),
                node_inlets_table=drainage_data.get("node_inlets_layer"),
            )
            drainage_data = drainage_inline
        finally:
            _dconn.close()
    drainage_cfg = build_drainage_config_from_json(drainage_data, ncells)
    if drainage_cfg is not None:
        drainage_cfg.gravity = _u.gravity()
    structures_cfg = build_structures_config_from_json(p.get("structures"), ncells)
    coupling_controller = None
    if drainage_cfg is not None or structures_cfg is not None:
        from swe2d.runtime.coupling import SWE2DCouplingController
        from swe2d.extensions.drainage_network import SWE2DUrbanDrainageModule
        from swe2d.extensions.structures import SWE2DStructureModule
        cell_area = backend.cell_areas()
        cell_zb = getattr(backend, "_cell_zb", np.zeros(ncells, dtype=np.float64))
        drainage_mod = SWE2DUrbanDrainageModule(drainage_cfg) if drainage_cfg is not None else None
        structures_mod = SWE2DStructureModule(structures_cfg) if structures_cfg is not None else None
        coupling_controller = SWE2DCouplingController(
            cell_area=cell_area,
            cell_bed=cell_zb,
            drainage=drainage_mod,
            structures=structures_mod,
            length_scale_si_to_model=si_m_per_model,
            log_callback=lambda msg: logger.info("[COUPLING] %s", msg),
        )
        _inv_perm = getattr(backend, "_inv_cell_perm", None)
        if _inv_perm is not None and _inv_perm.size > 0:
            try:
                coupling_controller._inv_cell_perm = _inv_perm.copy()
            except Exception as _e:
                logger.warning("[COUPLING] failed to sync RCMK inverse perm: %s", _e)
        try:
            cx, cy = mesh_cell_centroids(mesh_data)
            if hasattr(coupling_controller, "set_cell_centroids"):
                coupling_controller.set_cell_centroids(cx, cy)
            if hasattr(coupling_controller, "_build_redistribution_data"):
                coupling_controller._build_redistribution_data()
        except Exception as _e:
            logger.warning("[COUPLING] failed to set cell centroids: %s", _e)

    # ── Sample line setup ────────────────────────────────────────────
    sample_map_list: List[Dict[str, Any]] = []
    # Resolve intervals early so sample-line setup can use line_output_interval
    t_end = float(rp.get("duration_s", 3600.0))
    output_interval = float(rp.get("output_interval_s", t_end))
    line_output_interval = float(rp.get("line_output_interval_s", t_end))
    if line_output_interval > 0:
        sl_cfg = p.get("sample_lines")
        if isinstance(sl_cfg, dict) and sl_cfg.get("table"):
            sl_gpkg = sl_cfg.get("gpkg") or mesh_gpkg
            sl_table = sl_cfg["table"]
            raw_lines = query_sample_lines_from_qgis(sl_gpkg, sl_table)
            if raw_lines:
                from swe2d.workbench.services.mesh_service import (
                    build_line_sampling_map,
                    sample_line_metrics,
                )
                node_coords = np.stack([mesh_data["node_x"], mesh_data["node_y"]], axis=1)
                cell_nodes = mesh_data["cell_nodes"]
                cell_bed = getattr(backend, "_cell_zb", np.zeros(ncells, dtype=np.float64))
                for raw in raw_lines:
                    sm = build_line_sampling_map(
                        node_coords, cell_nodes, raw["line_xy"]
                    )
                    if sm.get("cell_idx", np.array([])).size > 0:
                        sm["line_id"] = raw["line_id"]
                        sm["line_name"] = raw.get("line_name", "")
                        sm["line_xy"] = raw["line_xy"]
                        sample_map_list.append(sm)
                logger.info("Built sample maps for %d line(s)", len(sample_map_list))

    # ── Simulation run ───────────────────────────────────────────────
    coupling_interval = float(structures_cfg.control_interval_s) if structures_cfg else 1.0

    _t0 = time.time()
    _status_last_write = [0.0]
    _status_step = [0]

    def _write_status(stage: str, t: float = 0.0, dt: float = 0.0,
                      wet: int = -1, err: Optional[str] = None):
        if not status_file_path:
            return
        now = time.time()
        if stage == "running" and (now - _status_last_write[0]) < status_interval_s:
            return
        _status_last_write[0] = now
        payload = {
            "step": _status_step[0],
            "t": float(t),
            "dt": float(dt),
            "wet_cells": int(wet) if wet >= 0 else -1,
            "elapsed_s": float(time.time() - _t0),
            "status": str(stage),
        }
        if err:
            payload["error"] = str(err)
        try:
            _atomic_write_json(status_file_path, payload)
        except Exception as _e:
            logger.warning("[ERROR] Status write failed: %s", _e)

    _write_status("running", t=0.0)

    diags: list = []
    t = 0.0
    step = 0
    coupling_snapshots = {}
    next_snap_t = output_interval
    next_line_snap_t = line_output_interval
    next_coupling_snap_t = coupling_interval

    def _collect_coupling_snapshot(cc, t_s: float) -> None:
        if cc is None:
            return
        if hasattr(cc, "_gpu_node_depth") and cc._gpu_node_depth is not None:
            cfg = getattr(getattr(cc, "drainage", None), "cfg", None)
            if cfg is not None:
                for i, node in enumerate(getattr(cfg, "nodes", [])):
                    if i >= len(cc._gpu_node_depth):
                        continue
                    key = ("drainage_node", str(getattr(node, "node_id", str(i))), "depth")
                    coupling_snapshots.setdefault(key, {"times": [], "values": []})["times"].append(t_s)
                    coupling_snapshots[key]["values"].append(float(cc._gpu_node_depth[i]))
        if hasattr(cc, "_gpu_link_flow") and cc._gpu_link_flow is not None:
            cfg = getattr(getattr(cc, "drainage", None), "cfg", None)
            if cfg is not None:
                for i, link in enumerate(getattr(cfg, "links", [])):
                    if i >= len(cc._gpu_link_flow):
                        continue
                    key = ("drainage_link", str(getattr(link, "link_id", str(i))), "flow")
                    coupling_snapshots.setdefault(key, {"times": [], "values": []})["times"].append(t_s)
                    coupling_snapshots[key]["values"].append(float(cc._gpu_link_flow[i]))
        nb_flows = getattr(cc, "_last_structure_flows", None)
        structures_cfg_local = getattr(cc, "_structures_cfg", ())
        if nb_flows is not None and structures_cfg_local:
            for i, st in enumerate(structures_cfg_local):
                sid = str(getattr(st, "structure_id", str(i)))
                if i < len(nb_flows):
                    key = ("structure", sid, "flow")
                    coupling_snapshots.setdefault(key, {"times": [], "values": []})["times"].append(t_s)
                    coupling_snapshots[key]["values"].append(float(nb_flows[i]))

    _status_step[0] = 0
    dt_request = float(rp.get("dt_request", rp.get("dt_max", 0.2)))

    while t < t_end:
        if cancel_check and cancel_check():
            _write_status("cancelled", t=t)
            break
        if coupling_controller is not None:
            try:
                coupling_controller.apply_native_device_sources(t, dt_request)
            except Exception as _e:
                logger.warning("[COUPLING] apply step failed: %s", _e)
        diag = backend.step(dt_request)
        dt = float(diag.get("dt", 0.0))
        t += dt
        step += 1
        _status_step[0] = step
        diags.append(diag)

        # GPU ring buffer: device-only copy, auto-dumps to host on memory pressure
        if t >= next_snap_t - 1e-9:
            backend.store_snapshot(t)
            next_snap_t += output_interval

        # Coupling snapshots at coupling interval
        if coupling_controller is not None and t >= next_coupling_snap_t - 1e-9:
            _collect_coupling_snapshot(coupling_controller, t)
            next_coupling_snap_t += coupling_interval

        if progress_callback:
            progress_callback(t, diag)
        _write_status("running", t=t, dt=dt, wet=diag.get("wet_cells", -1))

    _write_status("done")

    # Read all snapshots from GPU ring buffer + host auto-dump buffer
    snap_data = backend.read_snapshots()
    snapshot_timesteps: list = []
    if snap_data is not None and "t_s" in snap_data:
        ts_arr = snap_data["t_s"]
        h_arr = snap_data["h"]
        hu_arr = snap_data["hu"]
        hv_arr = snap_data["hv"]
        for i in range(int(ts_arr.shape[0])):
            snapshot_timesteps.append((
                float(ts_arr[i]),
                np.ascontiguousarray(h_arr[i, :]),
                np.ascontiguousarray(hu_arr[i, :]),
                np.ascontiguousarray(hv_arr[i, :]),
            ))
    if not snapshot_timesteps:
        h_term, hu_term, hv_term = backend.get_state()
        snapshot_timesteps = [(float(t), h_term, hu_term, hv_term)]

    max_results = backend.get_max_tracking()
    h, hu, hv = backend.get_state()

    out: Dict[str, Any] = {
        "h": h,
        "hu": hu,
        "hv": hv,
        "diags": diags,
    }
    if max_results is not None:
        out["max_results"] = max_results

    # ── Persist to results GPKG ────────────────────────────────────
    if results_gpkg:
        from swe2d.services.gpkg_persistence_service import (
            load_baked_mesh,
            persist_baked_coupling,
            persist_baked_mesh,
            persist_baked_results,
        )

        run_id = str(p.get("id", "run"))

        try:
            baked = load_baked_mesh(mesh_gpkg, mesh_name)
            if baked:
                persist_baked_mesh(results_gpkg, mesh_name, baked["blob"])
        except Exception as exc:
            logger.warning("Failed to persist baked mesh: %s", exc)

        persist_baked_results(
            results_gpkg, run_id, mesh_name,
            snapshot_timesteps,
            max_tracking=max_results,
            crs_wkt=mesh_data.get("crs_wkt", ""),
        )

        # ── Sample line results ─────────────────────────────────────────
        if sample_map_list and snapshot_timesteps:
            from swe2d.workbench.services.mesh_service import (
                sample_line_aggregate_ts_row,
                sample_line_metrics,
            )
            gravity_g = _u.gravity()
            h_min = float(rp.get("h_min", 1e-4))
            node_coords = np.stack([mesh_data["node_x"], mesh_data["node_y"]], axis=1)
            cell_nodes = mesh_data["cell_nodes"]
            cell_bed = getattr(backend, "_cell_zb", np.zeros(ncells, dtype=np.float64))

            for sm in sample_map_list:
                line_id = int(sm.get("line_id", 0))
                line_name = str(sm.get("line_name", ""))
                n_snaps = len(snapshot_timesteps)

                ts_times = np.empty(n_snaps, dtype=np.float64)
                depth_ts = np.empty(n_snaps, dtype=np.float64)
                vel_ts = np.empty(n_snaps, dtype=np.float64)
                wse_ts = np.empty(n_snaps, dtype=np.float64)
                bed_ts = np.empty(n_snaps, dtype=np.float64)
                flow_ts = np.empty(n_snaps, dtype=np.float64)
                wet_ts = np.empty(n_snaps, dtype=np.float64)
                froude_ts = np.empty(n_snaps, dtype=np.float64)

                station_arr = np.asarray(sm.get("profile_station_m", np.array([0.0])), dtype=np.float64)
                n_stations = max(1, station_arr.size)

                depth_prof = np.empty((n_snaps, n_stations), dtype=np.float64)
                vel_prof = np.empty((n_snaps, n_stations), dtype=np.float64)
                wse_prof = np.empty((n_snaps, n_stations), dtype=np.float64)
                bed_prof = np.empty((n_snaps, n_stations), dtype=np.float64)
                flow_prof = np.empty((n_snaps, n_stations), dtype=np.float64)
                fr_prof = np.empty((n_snaps, n_stations), dtype=np.float64)
                wet_prof = np.zeros((n_snaps, n_stations), dtype=np.int32)

                for snap_i, (snap_t, h_s, hu_s, hv_s) in enumerate(snapshot_timesteps):
                    row = sample_line_aggregate_ts_row(
                        sm, h_s, hu_s, hv_s, cell_bed, h_min, gravity_g, snap_t,
                    )
                    ts_times[snap_i] = snap_t
                    if row:
                        depth_ts[snap_i] = float(row.get("depth_m", 0.0))
                        vel_ts[snap_i] = float(row.get("velocity_ms", 0.0))
                        wse_ts[snap_i] = float(row.get("wse_m", 0.0))
                        bed_ts[snap_i] = float(row.get("bed_m", 0.0))
                        flow_ts[snap_i] = float(row.get("flow_cms", 0.0))
                        wet_ts[snap_i] = float(row.get("wet_frac", 0.0))
                        froude_ts[snap_i] = float(row.get("fr", 0.0))

                    m = sample_line_metrics(
                        h_s, hu_s, hv_s, cell_bed,
                        node_coords, cell_nodes,
                        sm.get("line_xy", np.zeros((2, 2), dtype=np.float64)),
                        h_min, snap_t, gravity_g, sample_map=sm,
                    )
                    d = np.asarray(m.get("depth_m", np.array([])), dtype=np.float64)
                    v = np.asarray(m.get("velocity_ms", np.array([])), dtype=np.float64)
                    w = np.asarray(m.get("wse_m", np.array([])), dtype=np.float64)
                    b = np.asarray(m.get("bed_m", np.array([])), dtype=np.float64)
                    q = np.asarray(m.get("flow_qn", np.array([])), dtype=np.float64)
                    f = np.asarray(m.get("froude", np.array([])), dtype=np.float64)
                    wt = np.asarray(m.get("wet", np.array([], dtype=np.int32)), dtype=np.int32)
                    max_s = min(n_stations, d.size)
                    if max_s > 0:
                        depth_prof[snap_i, :max_s] = d[:max_s]
                        vel_prof[snap_i, :max_s] = v[:max_s]
                        wse_prof[snap_i, :max_s] = w[:max_s]
                        bed_prof[snap_i, :max_s] = b[:max_s]
                        flow_prof[snap_i, :max_s] = q[:max_s]
                        fr_prof[snap_i, :max_s] = f[:max_s]
                        wet_prof[snap_i, :max_s] = wt[:max_s]

                try:
                    from swe2d.services.gpkg_persistence_service import (
                        persist_baked_line_profile,
                        persist_baked_line_ts,
                    )
                    persist_baked_line_ts(
                        results_gpkg, run_id, line_id, line_name,
                        ts_times, depth_ts, vel_ts, wse_ts, bed_ts,
                        flow_ts, wet_ts, froude_ts,
                        log_fn=lambda m: logger.info("[LINE] %s", m),
                    )
                    persist_baked_line_profile(
                        results_gpkg, run_id, line_id, line_name,
                        station_arr, ts_times,
                        depth_prof, vel_prof, wse_prof, bed_prof,
                        flow_prof, fr_prof, wet_prof,
                        log_fn=lambda m: logger.info("[LINE] %s", m),
                    )
                except Exception as exc:
                    logger.warning("Failed to persist line results: %s", exc)

        if coupling_snapshots:
            for (component, object_id, metric), d in coupling_snapshots.items():
                if d["times"]:
                    persist_baked_coupling(
                        results_gpkg, run_id,
                        component, object_id,
                        str(object_id),  # object_name
                        metric,
                        np.array(d["times"], dtype=np.float64),
                        np.array(d["values"], dtype=np.float64),
                        log_fn=lambda m: logger.info("[COUPLING] %s", m),
                    )

    backend.destroy()
    return out


