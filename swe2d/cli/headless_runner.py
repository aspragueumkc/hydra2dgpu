"""Headless runner: execute a simulation from JSON params + GPKG without QGIS.

Usage:
    from swe2d.cli.headless_runner import execute_run
    results = execute_run(mesh_gpkg, params)
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any, Callable, Dict, Optional

import numpy as np

from swe2d.runtime.backend import SWE2DBackend
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


def execute_run(
    mesh_gpkg: str,
    params: Dict[str, Any],
    results_gpkg: Optional[str] = None,
    progress_callback: Optional[Callable[[float, Dict[str, Any]], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """Run a simulation from GPKG-stored mesh + JSON params.

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

    # Build backend
    backend = SWE2DBackend()
    backend.build_mesh(**mesh_data)
    nnodes = int(mesh_data["node_x"].size)
    ncells = int(backend.n_cells)

    # Read BC arrays from GPKG tables
    conn = sqlite3.connect(mesh_gpkg)
    try:
        bc_table = p.get("bc_lines", "")
        bc = {}
        if bc_table:
            bc = query_bc_arrays(conn, bc_table)
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
                thiessen_forcing = build_forced_thiessen_from_gpkg(
                    conn, ncells,
                    mesh_data["node_x"], mesh_data["node_y"],
                    mesh_data["cell_nodes"],
                    hyetograph_table=htable,
                    gauge_table=gtable,
                    cn_table=cn_table,
                    cn_field=cntable.get("cn_field", "cn") if isinstance(cntable, dict) else "cn",
                    infiltration_method=p.get("infiltration_method", "scs_cn"),
                )
    finally:
        conn.close()

    # Build run options
    rp = p.get("params", {})
    from swe2d.runtime.run_options_builder import RunOptionsBuilder

    builder = RunOptionsBuilder(
        length_unit_si_to_model_fn=lambda v: v,
        flow_si_to_model_fn=lambda v: v,
        rain_rate_si_to_model_fn=lambda v: v,
        internal_flow_source_cms_at_time_fn=lambda f, t: None,
        build_thiessen_rain_cn_forcing_callback=lambda: thiessen_forcing,
    )
    run_options = builder.build(
        dt=float(rp.get("dt_cfg", 0.2)),
        rain_rate_mmhr=float(rp.get("rain_rate_mmhr", 0.0)),
        n_mann=float(rp.get("n_mann", 0.035)),
        h_min=float(rp.get("h_min", 1e-4)),
        dt_max=float(rp.get("dt_max", 0.2)),
        cfl=float(rp.get("cfl", 0.45)),
    )

    # Initialize solver
    from swe2d.runtime.backend import BCType

    h0 = np.zeros(ncells, dtype=np.float64)
    backend.initialize(
        h0=h0,
        n_mann=float(rp.get("n_mann", 0.035)),
        h_min=float(rp.get("h_min", 1e-4)),
        cfl=float(rp.get("cfl", 0.45)),
        dt_max=float(rp.get("dt_max", 0.2)),
        gpu_diag_sync_interval_steps=int(rp.get("gpu_diag_sync_interval_steps", 100)),
    )

    # Configure boundary conditions if BC arrays were found
    if bc_n0.size > 0:
        try:
            backend.set_boundary_conditions(bc_n0, bc_n1, bc_tp, bc_vl)
        except Exception:
            pass

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
        except Exception:
            pass

    # Run simulation
    t_end = float(rp.get("duration_s", 3600.0))
    output_interval = float(rp.get("output_interval_s", t_end))
    save_max_only = bool(rp.get("save_max_only", True))

    if save_max_only:
        diags: list = []
        t = 0.0
        step = 0
        while t < t_end:
            if cancel_check and cancel_check():
                break
            diag = backend.step(rp.get("dt_request", -1.0))
            dt = float(diag.get("dt", 0.0))
            t += dt
            step += 1
            diags.append(diag)
            if progress_callback:
                progress_callback(t, diag)
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

    out: Dict[str, Any] = {
        "h": h,
        "hu": hu,
        "hv": hv,
        "diags": diags,
    }
    if max_results is not None:
        out["max_results"] = max_results

    # Persist to results GPKG if provided
    if results_gpkg:
        _persist_results(results_gpkg, p.get("id", "run"), ncells, h, hu, hv, max_results)

    backend.destroy()
    return out


def _persist_results(
    gpkg_path: str,
    run_id: str,
    n_cells: int,
    h: np.ndarray,
    hu: np.ndarray,
    hv: np.ndarray,
    max_results: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    """Write final results to a results GPKG."""
    import sqlite3

    conn = sqlite3.connect(gpkg_path)
    try:
        cur = conn.cursor()
        if max_results is not None:
            from swe2d.workbench.services.gpkg_persistence_service import (
                persist_mesh_max_results_to_geopackage,
            )
            persist_mesh_max_results_to_geopackage(gpkg_path, run_id, max_results)
    finally:
        conn.close()
