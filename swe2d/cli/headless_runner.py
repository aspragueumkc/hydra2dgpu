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
from typing import Any, Callable, Dict, Optional


logger = logging.getLogger(__name__)

import numpy as np

from swe2d.runtime.backend import SWE2DBackend, build_mesh as shared_build_mesh
from swe2d.cli.gpkg_adapter import (
    build_forced_thiessen_from_gpkg,
    query_bc_arrays,
    query_mesh_from_gpkg,
)


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

            logger.warning(f"[ERROR] Exception in headless_runner.py: {_e}")


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
        gravity=float(rp.get("gravity", 9.81)),
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
    )
    drainage_cfg = build_drainage_config_from_json(p.get("drainage"), ncells)
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
            log_callback=lambda msg: logger.info("[COUPLING] %s", msg),
        )
        # Sync RCMK inverse permutation from backend to coupling controller.
        _inv_perm = getattr(backend, "_inv_cell_perm", None)
        if _inv_perm is not None and _inv_perm.size > 0 and coupling_controller is not None:
            try:
                coupling_controller._inv_cell_perm = _inv_perm.copy()
            except Exception as _e:

                logger.warning(f"[ERROR] Exception in headless_runner.py: {_e}")

    # Run simulation
    t_end = float(rp.get("duration_s", 3600.0))
    output_interval = float(rp.get("output_interval_s", t_end))
    save_max_only = bool(rp.get("save_max_only", True))

    # Status file writer (on-demand, no continuous polling)
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

            logger.warning(f"[ERROR] Exception in headless_runner.py: {_e}")

    if save_max_only:
        diags: list = []
        t = 0.0
        step = 0
        _status_step[0] = 0
        _write_status("running", t=t)
        snapshot_timesteps: list = []
        next_snap_t = output_interval
        while t < t_end:
            if cancel_check and cancel_check():
                _write_status("cancelled", t=t)
                break
            if coupling_controller is not None:
                try:
                    coupling_controller.apply_native_device_sources(t, float(rp.get("dt_request", 0.2)))
                except Exception as _e:
                    logger.warning("Coupling step failed: %s", _e)
            diag = backend.step(rp.get("dt_request", -1.0))
            dt = float(diag.get("dt", 0.0))
            t += dt
            step += 1
            _status_step[0] = step
            # Accumulate snapshots at output interval
            if t >= next_snap_t - 1.0e-9:
                h_snap, hu_snap, hv_snap = backend.get_state()
                snapshot_timesteps.append((t, h_snap, hu_snap, hv_snap))
                next_snap_t += output_interval
            diags.append(diag)
            if progress_callback:
                progress_callback(t, diag)
            _write_status("running", t=t, dt=dt, wet=diag.get("wet_cells", -1))
        max_results = backend.get_max_tracking()
        h, hu, hv = backend.get_state()
    else:
        diags = backend.run(
            t_end,
            dt_request=rp.get("dt_request", -1.0),
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )
        max_results = None
        h, hu, hv = backend.get_state()

    _write_status("done")

    out: Dict[str, Any] = {
        "h": h,
        "hu": hu,
        "hv": hv,
        "diags": diags,
    }
    if max_results is not None:
        out["max_results"] = max_results

    # Persist to results GPKG if provided — use same functions as workbench
    if results_gpkg:
        from swe2d.services.gpkg_persistence_service import (
            persist_baked_results, persist_baked_mesh,
        )
        snapshot_timesteps = locals().get("snapshot_timesteps", [])
        if not snapshot_timesteps:
            h, hu, hv = backend.get_state()
            snapshot_timesteps = [(0.0, h, hu, hv)]

        # Copy baked mesh BLOB so results GPKG is self-contained
        try:
            from swe2d.services.gpkg_persistence_service import load_baked_mesh
            baked = load_baked_mesh(mesh_gpkg, mesh_name)
            if baked:
                persist_baked_mesh(results_gpkg, mesh_name, baked["blob"])
        except Exception as exc:
            logger.warning("Failed to persist baked mesh: %s", exc)

        # Persist all accumulated snapshots
        persist_baked_results(
            results_gpkg, p.get("id", "run"), mesh_name,
            snapshot_timesteps,
            max_tracking=max_results,
            crs_wkt=mesh_data.get("crs_wkt", ""),
        )

    backend.destroy()
    return out


