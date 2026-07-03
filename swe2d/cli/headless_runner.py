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
from swe2d.runtime.run_finalizer import SWE2DRunFinalizer


class _HeadlessFinalizationView:
    """Headless view implementing the RunFinalizationView protocol.

    All GUI-only operations (overlay sync, plot refresh) are no-ops.
    """

    def __init__(self, results_gpkg: str, mesh_name: str, mesh_data: Dict[str, Any],
                 length_unit: str, length_scale_si: float):
        self._results_gpkg = results_gpkg
        self._mesh_name = mesh_name
        self._mesh_data = mesh_data
        self._length_unit = length_unit
        self._length_scale_si = length_scale_si
        self._log_lines: List[str] = []

    def log_message(self, msg: str) -> None:
        logger.info("%s", msg)
        self._log_lines.append(msg)

    def get_line_results_storage_path(self) -> str:
        return self._results_gpkg

    def sync_overlay_data(self) -> None:
        pass

    def refresh_plot(self) -> None:
        pass

    def results_table_name(self, base: str) -> str:
        return base

    def length_unit_name(self) -> str:
        return self._length_unit

    def length_scale_si_to_model(self) -> float:
        return self._length_scale_si

    def update_overlay_time(self, t: float) -> None:
        pass

    def runtime_log_lines(self) -> List[str]:
        return self._log_lines

    def collect_run_log_metadata(self) -> Dict[str, object]:
        return {}

    def persist_run_log(self, gpkg_path: str, run_id: str,
                        run_wallclock_start: str, run_wallclock_end: str,
                        run_duration_wallclock_s: float, run_log_text: str,
                        *, metadata: Dict[str, object]) -> None:
        pass

    def is_cancel_requested(self) -> bool:
        return False

    def results_data(self):
        return None


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
        state = cc.readback_coupling_state()
        cfg = getattr(getattr(cc, "drainage", None), "cfg", None)
        if cfg is not None:
            for i, node in enumerate(getattr(cfg, "nodes", [])):
                if i < len(state["node_depth"]):
                    key = ("drainage_node", str(getattr(node, "node_id", str(i))), "depth")
                    coupling_snapshots.setdefault(key, {"times": [], "values": []})["times"].append(t_s)
                    coupling_snapshots[key]["values"].append(float(state["node_depth"][i]))
            for i, link in enumerate(getattr(cfg, "links", [])):
                if i < len(state["link_flow"]):
                    key = ("drainage_link", str(getattr(link, "link_id", str(i))), "flow")
                    coupling_snapshots.setdefault(key, {"times": [], "values": []})["times"].append(t_s)
                    coupling_snapshots[key]["values"].append(float(state["link_flow"][i]))
        structures_cfg_local = getattr(cc, "_structures_cfg", ())
        nb_mask = getattr(cc, "_structure_non_bridge_mask", None)
        for i, st in enumerate(structures_cfg_local):
            if nb_mask is not None and not nb_mask[i]:
                continue
            idx = i if nb_mask is None else int(np.flatnonzero(nb_mask[:i+1])[-1]) if np.any(nb_mask[:i+1]) else -1
            if idx < 0 or idx >= len(state["struct_flow"]):
                continue
            sid = str(getattr(st, "structure_id", str(i)))
            key = ("structure", sid, "flow")
            coupling_snapshots.setdefault(key, {"times": [], "values": []})["times"].append(t_s)
            coupling_snapshots[key]["values"].append(float(state["struct_flow"][idx]))

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

    # ── Persist to results GPKG via shared SWE2DRunFinalizer ───────────────
    if results_gpkg:
        run_id = str(p.get("id", "run"))

        from swe2d.services.gpkg_persistence_service import load_baked_mesh, persist_baked_mesh
        try:
            baked_blob = load_baked_mesh(mesh_gpkg, mesh_name)
            if baked_blob:
                persist_baked_mesh(results_gpkg, mesh_name, baked_blob,
                                   crs_wkt=mesh_data.get("crs_wkt", ""))
        except Exception as exc:
            print(f"Failed to persist baked mesh: {exc}", flush=True)

        gravity_g = _u.gravity()
        h_min = float(rp.get("h_min", 1e-4))
        node_coords = np.stack([mesh_data["node_x"], mesh_data["node_y"]], axis=1)
        cell_nodes = mesh_data["cell_nodes"]
        cell_bed = getattr(backend, "_cell_zb", np.zeros(ncells, dtype=np.float64))

        precomputed_line_results: Dict[int, Dict[str, Any]] = {}
        if sample_map_list and snapshot_timesteps:
            from swe2d.workbench.services.mesh_service import (
                sample_line_aggregate_ts_row,
                sample_line_metrics,
            )
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
                precomputed_line_results[line_id] = {
                    "line_name": line_name,
                    "t_s": list(ts_times),
                    "ts_depth_m": list(depth_ts),
                    "ts_velocity_ms": list(vel_ts),
                    "ts_wse_m": list(wse_ts),
                    "ts_bed_m": list(bed_ts),
                    "ts_flow_cms": list(flow_ts),
                    "ts_wet_frac": list(wet_ts),
                    "ts_fr": list(froude_ts),
                    "station_m": station_arr,
                    "prof_depth_m": [np.ascontiguousarray(depth_prof[i, :]) for i in range(n_snaps)],
                    "prof_velocity_ms": [np.ascontiguousarray(vel_prof[i, :]) for i in range(n_snaps)],
                    "prof_wse_m": [np.ascontiguousarray(wse_prof[i, :]) for i in range(n_snaps)],
                    "prof_bed_m": [np.ascontiguousarray(bed_prof[i, :]) for i in range(n_snaps)],
                    "prof_flow_qn": [np.ascontiguousarray(flow_prof[i, :]) for i in range(n_snaps)],
                    "prof_fr": [np.ascontiguousarray(fr_prof[i, :]) for i in range(n_snaps)],
                    "prof_wet": [np.ascontiguousarray(wet_prof[i, :]) for i in range(n_snaps)],
                }

        length_unit = mesh_data.get("length_unit", "m")
        length_scale_si = _si_m_per_model_from_wkt(mesh_data.get("crs_wkt", ""))
        fv = _HeadlessFinalizationView(results_gpkg, mesh_name, mesh_data, length_unit, length_scale_si)
        finalizer = SWE2DRunFinalizer(fv)
        finalizer.finalize_and_persist(
            h=h, hu=hu, hv=hv,
            final_sim_time_s=float(diags[-1].get("t", 0.0)) if diags else 0.0,
            n_area=ncells,
            area_model=getattr(backend, "_cell_area", np.ones(ncells, dtype=np.float64)),
            storage_start_model=0.0,
            source_budget_model={"rain": 0.0, "cell": 0.0, "coupling": 0.0},
            source_step_rows_model=[],
            run_duration_s=t,
            boundary_flux_budget_model={},
            boundary_flux_step_rows_model=[],
            run_id=run_id,
            output_interval_s=float(rp.get("output_interval_s", t)),
            line_output_interval_s=float(rp.get("line_output_interval_s", t)),
            run_perf_start=0.0,
            run_wallclock_start="",
            run_log_start_idx=0,
            thiessen_forcing=None,
            rain_stats_acc={"samples": 0, "rain_mm": 0.0, "excess_mm": 0.0},
            save_line_results=bool(sample_map_list),
            save_coupling_results=bool(coupling_snapshots),
            save_mesh_results=True,
            save_run_log=False,
            h_min=h_min,
            mesh_name=mesh_name,
            max_tracking=max_results,
            coupling_controller=coupling_controller,
            sample_map=None,
            cell_solver_z=cell_bed,
            sample_line_metrics_callback=None,
            snapshot_timesteps=snapshot_timesteps,
            coupling_snapshots=coupling_snapshots,
            precomputed_line_results=precomputed_line_results if precomputed_line_results else None,
        )

    backend.destroy()
    return out


