"""Headless runner: execute a simulation from JSON params + GPKG without QGIS.

Usage:
    from swe2d.cli.headless_runner import execute_run
    results = execute_run(mesh_gpkg, params)
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from swe2d.core.gpkg_io import query_mesh_from_gpkg

logger = logging.getLogger(__name__)


class _HeadlessSink:
    """Sink implementation for headless CLI execution."""

    def __init__(
        self,
        log_cb: Any = None,
        progress_cb: Any = None,
        snapshot_cb: Any = None,
        status_file_path: Optional[str] = None,
        status_interval_s: float = 5.0,
    ) -> None:
        self._log_cb = log_cb
        self._progress_cb = progress_cb
        self._snapshot_cb = snapshot_cb
        self._status_file_path = status_file_path
        self._status_interval_s = status_interval_s
        self._error_message: Optional[str] = None
        self._status_last_write = 0.0
        self._step = 0
        self._last_pct = 0.0
        self._last_t = 0.0
        self._last_dt = 0.0
        self._last_wet_cells = -1
        self._last_elapsed_s = 0.0
        self._t0 = time.time()
        self.snapshot_request_event = threading.Event()

    def _write_status(self, stage: str, err: Optional[str] = None) -> None:
        if not self._status_file_path:
            return
        now = time.time()
        if (
            stage == "running"
            and now - self._status_last_write < self._status_interval_s
        ):
            return
        self._status_last_write = now
        payload = {
            "step": self._step,
            "step_idx": self._step,
            "pct": self._last_pct,
            "t": self._last_t,
            "dt": self._last_dt,
            "wet_cells": self._last_wet_cells,
            "elapsed_s": max(self._last_elapsed_s, now - self._t0),
            "status": str(stage),
        }
        if err:
            payload["error"] = str(err)
        try:
            _atomic_write_json(self._status_file_path, payload)
        except Exception as exc:
            logger.warning("[ERROR] Status write failed: %s", exc)

    def log(self, message: str) -> None:
        if self._log_cb:
            self._log_cb(message)

    def progress(self, percent: float, diagnostics: Dict[str, Any]) -> None:
        self._step = int(diagnostics["step"])
        self._last_pct = float(percent)
        self._last_t = float(diagnostics["t_s"])
        self._last_dt = float(diagnostics["dt"])
        self._last_wet_cells = int(diagnostics["wet_cells"])
        self._last_elapsed_s = float(diagnostics["elapsed_s"])
        callback_diagnostics = {
            "dt": self._last_dt,
            "wet_cells": self._last_wet_cells,
            "elapsed_s": self._last_elapsed_s,
        }
        if self._progress_cb:
            self._progress_cb(self._last_t, callback_diagnostics)
        self._write_status("running")

    def snapshot(self, fields: List[Any]) -> None:
        if self._snapshot_cb:
            self._snapshot_cb(*fields)

    def finished(self, result: Dict[str, Any]) -> None:
        # No-op for CLI - result is returned directly
        pass

    def failed(self, error: str) -> None:
        self._error_message = error
        if self._log_cb:
            self._log_cb(f"[ERROR] {error}")

    def permutation(self, cell_perm: Any, result: Any) -> None:
        # Headless runs do not need a main-thread permutation callback.
        result.event.set()

    def request_snapshot(self) -> None:
        self.snapshot_request_event.set()


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
    """Run a simulation from GPKG-stored mesh + JSON params using the shared executor.

    Uses ``swe2d.core.executor.execute_run()`` with a plain ``Sink`` so the CLI
    produces byte-identical results to the QGIS workbench GUI without importing
    Qt. The old raw ``while`` loop is retired.

    If ``status_file_path`` is set, a JSON status file is written every
    ``status_interval_s`` seconds during the simulation. ``progress_callback``
    receives ``(sim_time_s, {"dt": ..., "wet_cells": ..., "elapsed_s": ...})``
    after each solver step.

    The status file contains:
        {"step": int, "step_idx": int, "pct": float,
         "t": float, "dt": float, "wet_cells": int,
         "elapsed_s": float, "status": "running"|"done"|"error",
         "error": str|null}

    ``step`` and ``step_idx`` are the actual 0-indexed solver step number
    reported by the runtime context. ``pct`` is the independent 0-100
    simulation progress percentage.

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
    # Keep mesh dict intact so builder can read crs_wkt from it.
    if results_gpkg:
        builder_params["results_gpkg_path"] = results_gpkg

    # ── Build RunContext and execute via shared headless pipeline ──────
    from swe2d.core.builder import build_run_context_from_dict
    from swe2d.core.executor import execute_run

    cancel_event = (
        _CancelEventWrapper(cancel_check) if cancel_check is not None else None
    )
    ctx = build_run_context_from_dict(builder_params, cancel_event=cancel_event)

    sink = _HeadlessSink(
        log_cb=logger.info,
        progress_cb=progress_callback,
        status_file_path=status_file_path,
        status_interval_s=status_interval_s,
    )
    sink._write_status("running")

    compute_result = execute_run(ctx, sink)

    if compute_result.error_message:
        sink._write_status("error", err=compute_result.error_message)
    else:
        sink._write_status("done")

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
    status_file_path: Optional[str] = None,
    status_interval_s: float = 5.0,
) -> Dict[str, Any]:
    """Replay a run from a canonical replay JSON file using the shared executor.

    Uses ``swe2d.core.executor.execute_run()`` so the CLI produces the same
    results as the QGIS workbench GUI.
    """
    with open(replay_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    from swe2d.core.builder import build_run_context_from_dict
    from swe2d.core.executor import execute_run

    ctx = build_run_context_from_dict(payload)

    sink = _HeadlessSink(
        log_cb=log_cb,
        progress_cb=progress_cb,
        status_file_path=status_file_path,
        status_interval_s=status_interval_s,
    )
    sink._write_status("running")

    compute_result = execute_run(ctx, sink)

    if compute_result.error_message:
        sink._write_status("error", err=compute_result.error_message)
    else:
        sink._write_status("done")

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



