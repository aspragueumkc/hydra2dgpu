"""Headless runner: execute a simulation from JSON params + GPKG without QGIS.

Usage:
    from swe2d.cli.headless_runner import execute_run
    results = execute_run(mesh_gpkg, params)
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)

import numpy as np

from swe2d.cli.gpkg_adapter import query_mesh_from_gpkg


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
    mesh_gpkg: Optional[str],
    params: Dict[str, Any],
    results_gpkg: Optional[str] = None,
    progress_callback: Optional[Callable[[float, Dict[str, Any]], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    status_file_path: Optional[str] = None,
    status_interval_s: float = 5.0,
) -> Dict[str, Any]:
    """Run a simulation from GPKG-stored mesh + JSON params using the GUI path.

    Uses ``SimulationWorker._execute()`` via ``execute_swe2d_headless()``,
    sharing the same RunContext → backend → timestep loop pipeline as the
    QGIS workbench.  The old raw ``while`` loop is retired — this produces
    byte-identical results.

    If ``status_file_path`` is set, a JSON status file is written every
    ``status_interval_s`` seconds during the simulation.

    The status file contains:
        {"step": int, "t": float, "dt": float, "wet_cells": int,
         "elapsed_s": float, "status": "running"|"done"|"error",
         "error": str|null}

    Returns dict with keys: h, hu, hv, max_results (optional), diags.
    """
    # Allow mesh_gpkg to come from params so JSON snapshots are self-contained
    p = params

    # Support both string mesh name and dict mesh spec (serialized RunContext):
    #   "mesh": "mesh_name"            ← string (normal CLI/old format)
    #   "mesh": {"mesh_name": ..., "gpkg_path": ..., "crs_wkt": ...}  ← dict
    mesh_val = p.get("mesh", "")
    if isinstance(mesh_val, dict):
        mesh_name = str(mesh_val.get("mesh_name", ""))
        _mesh_gpkg_from_params = mesh_val.get("gpkg_path", "")
        if _mesh_gpkg_from_params:
            mesh_gpkg = str(_mesh_gpkg_from_params)
    else:
        mesh_name = str(mesh_val) if mesh_val else ""
        if not mesh_gpkg:
            mesh_gpkg = str(params.get("mesh_gpkg", ""))

    if not mesh_gpkg:
        raise ValueError("mesh_gpkg must be provided as argument or in params['mesh']['gpkg_path']")
    if not os.path.isfile(mesh_gpkg):
        raise FileNotFoundError(f"Mesh GPKG not found: {mesh_gpkg}")
    if not mesh_name:
        raise ValueError("'mesh' key (string or dict with 'mesh_name') required in params JSON")

    # ── Configure unit system from mesh CRS ───────────────────────────
    md = query_mesh_from_gpkg(mesh_gpkg, mesh_name)
    if md is None:
        raise ValueError(f"Mesh '{mesh_name}' not found in {mesh_gpkg}")

    from swe2d import units as _u
    crs_wkt = md.get("crs_wkt", "")
    si_m_per_model = _u.si_m_per_model_from_wkt(crs_wkt) if crs_wkt else 1.0
    _u.configure(si_m_per_model)

    # ── Build a flat params dict for the RunContext builder ───────────
    # Top-level keys (mesh_gpkg, mesh_name, results_gpkg, etc.) override
    # nested sub-dicts.  The builder reads p["mesh_gpkg"], p["mesh_name"],
    # p["params"]["output_interval_s"], etc.
    builder_params: Dict[str, Any] = dict(p)
    builder_params["mesh_gpkg"] = mesh_gpkg
    builder_params["mesh_name"] = mesh_name
    # Replace mesh dict with string so builder doesn't receive a dict
    if isinstance(builder_params.get("mesh"), dict):
        builder_params["mesh"] = mesh_name
    if results_gpkg:
        builder_params["results_gpkg_path"] = results_gpkg

    # ── Build RunContext and execute via shared headless pipeline ──────
    from swe2d.runtime.run_context_builder import build_run_context_from_dict
    from swe2d.cli.headless_executor import execute_swe2d_headless

    if cancel_check is not None:
        builder_params["cancel_event"] = _CancelEventWrapper(cancel_check)

    ctx = build_run_context_from_dict(builder_params)

    # ── Progress / status-file wrapping ───────────────────────────────
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

    def _progress_wrapper(pct: int) -> None:
        _status_step[0] = pct  # pct is 0..100
        if progress_callback:
            progress_callback(pct, {})
        _write_status("running")

    compute_result = execute_swe2d_headless(
        ctx,
        log_cb=logger.info,
        progress_cb=_progress_wrapper if (progress_callback or status_file_path) else None,
    )

    _write_status("done")
    if compute_result.error_message:
        _write_status("error", err=compute_result.error_message)

    # ── Build result dict (compatible with old callers) ────────────────
    out: Dict[str, Any] = {
        "h": compute_result.h,
        "hu": compute_result.hu,
        "hv": compute_result.hv,
        "diags": [],
    }
    if compute_result.max_tracking is not None:
        out["max_results"] = compute_result.max_tracking

    return out


class _CancelEventWrapper:
    """Wrap a cancel_check callable as a threading.Event-like gate."""

    def __init__(self, check_fn: Callable[[], bool]):
        self._check = check_fn

    def is_set(self) -> bool:
        return self._check()

    def set(self) -> None:
        pass  # no-op: cancel only flows one direction


def execute_replay(
    replay_file: str,
    log_cb: Any = None,
    progress_cb: Any = None,
) -> Dict[str, Any]:
    """Replay a run from a canonical replay JSON file using the GUI execution path.

    Uses ``SimulationWorker._execute()`` via the headless executor so that the
    CLI produces byte-identical results to the QGIS workbench GUI.
    """
    with open(replay_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    from swe2d.runtime.run_context_builder import build_run_context_from_dict
    from swe2d.cli.headless_executor import execute_swe2d_headless

    ctx = build_run_context_from_dict(payload)

    compute_result = execute_swe2d_headless(
        ctx,
        log_cb=log_cb,
        progress_cb=progress_cb,
    )

    out: Dict[str, Any] = {
        "run_id": compute_result.run_id,
        "mesh_name": compute_result.mesh_name,
        "duration_s": compute_result.run_duration_s,
        "final_t": compute_result.final_sim_time_s,
        "n_steps": 0,
        "status": "completed" if compute_result.ok else "failed",
        "h": compute_result.h,
        "hu": compute_result.hu,
        "hv": compute_result.hv,
        "max_h": None,
        "snapshots": compute_result.snapshot_timesteps,
        "line_results": None,
        "coupling_results": compute_result.coupling_snapshots,
    }
    if compute_result.error_message:
        out["error"] = compute_result.error_message
        out["status"] = "failed"
    return out



