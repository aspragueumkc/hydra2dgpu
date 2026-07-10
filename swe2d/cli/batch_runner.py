"""Batch runner: execute multiple simulations via subprocess isolation.

Manages MPS daemon for single-GPU concurrent kernel scheduling.
Supports status-file output for optional external monitoring.
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import time
import concurrent.futures
from typing import Any, Dict, List, Optional, Callable


_VALID_SCHEMES: frozenset = frozenset(range(9))


def validate_scheme(scheme: int) -> int:
    """Validate and warn about scheme number. Returns valid scheme or raises."""
    if scheme not in _VALID_SCHEMES:
        raise ValueError(
            f"Invalid spatial_scheme={scheme}. Must be 0-8."
        )
    if scheme == 6:
        import logging
        logging.getLogger(__name__).warning(
            "spatial_scheme=6 was FV_WENO5; now it is FV_WENO3 (true 3-sub-stencil). "
            "To keep WENO5, use spatial_scheme=7."
        )
    return scheme


def _ensure_mps() -> bool:
    """Start the NVIDIA MPS daemon if it's not already running.

    Returns True if MPS is active (either started now or already running).
    Logs a message to stderr explaining what happened.
    """
    import shutil
    if not shutil.which("nvidia-cuda-mps-control"):
        print("[MPS] nvidia-cuda-mps-control not found. "
              "Install NVIDIA CUDA Tools and restart. "
              "Simulations will run sequentially.", file=sys.stderr, flush=True)
        return False
    try:
        result = subprocess.run(
            ["nvidia-cuda-mps-control", "-d"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            print("[MPS] Daemon started for concurrent GPU scheduling.", file=sys.stderr, flush=True)
            return True
        if "already running" in (result.stderr or "").lower():
            print("[MPS] Daemon already running.", file=sys.stderr, flush=True)
            return True
        print(f"[MPS] Failed to start daemon: {result.stderr.strip()}. "
              "Simulations will run sequentially.", file=sys.stderr, flush=True)
        return False
    except FileNotFoundError:
        print("[MPS] nvidia-cuda-mps-control not available. "
              "Simulations will run sequentially.", file=sys.stderr, flush=True)
        return False
    except subprocess.TimeoutExpired:
        print("[MPS] Daemon start timed out. "
              "Simulations will run sequentially.", file=sys.stderr, flush=True)
        return False


def _stop_mps_if_we_started(started: bool) -> None:
    """Shut down the MPS daemon only if we started it ourselves."""
    if not started:
        return
    try:
        p = subprocess.Popen(
            ["nvidia-cuda-mps-control"],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            text=True,
        )
        p.communicate("quit\n", timeout=5)
        print("[MPS] Daemon stopped.", file=sys.stderr, flush=True)
    except Exception:
        print("[MPS] Daemon stop failed (daemon may remain active).", file=sys.stderr, flush=True)


def _expand_sweep(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand sweep keys into individual param sets (Cartesian product)."""
    sweep = params.get("sweep")
    if not sweep:
        return [params]

    import itertools

    keys = list(sweep.keys())
    values = list(sweep.values())
    expanded: List[Dict[str, Any]] = []
    id_template = str(params.get("id_template", "").strip() or "")

    for combo in itertools.product(*values):
        p = copy.deepcopy(params)
        combo_dict = dict(zip(keys, combo))
        for k, v in combo_dict.items():
            parts = k.split(".")
            target = p
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            target[parts[-1]] = v
        if id_template:
            p["id"] = id_template.format(**{k.replace("params.", ""): v for k, v in combo_dict.items()})
        expanded.append(p)

    return expanded


def run_batch(
    batch_json_path: str,
    mesh_gpkg: str,
    results_gpkg: str = "",
    max_workers: int = 0,
    status_callback: Optional[Callable[[int, int, int, float], None]] = None,
) -> None:
    """Read batch JSON, expand sweeps, run sims in subprocess pool.

    Args:
        batch_json_path: Path to batch JSON file.
        mesh_gpkg: Path to mesh GeoPackage.
        results_gpkg: Path to results GeoPackage.
        max_workers: Max concurrent simulations (0 = auto).
        status_callback: Optional callback(done, total, failed, elapsed_s)
            called as each sim completes. Can be used by QGIS to update UI
            without polling subprocess output.
    """
    with open(batch_json_path) as f:
        batch_config = json.load(f)

    if isinstance(batch_config, list):
        param_sets = batch_config
    elif isinstance(batch_config, dict):
        param_sets = [batch_config]
    else:
        raise ValueError("batch JSON must be an array or object")

    all_params: List[Dict[str, Any]] = []
    for ps in param_sets:
        all_params.extend(_expand_sweep(dict(ps)))

    if not all_params:
        print("No param sets to run.")
        if status_callback:
            status_callback(0, 0, 0, 0.0)
        return

    if max_workers <= 0:
        max_workers = min(len(all_params), 4)

    # Start MPS for concurrent GPU scheduling
    mps_started = _ensure_mps() if max_workers > 1 else False

    stime = time.perf_counter()
    results_gpkg = results_gpkg or os.path.splitext(mesh_gpkg)[0] + "_batch_results.gpkg"

    def _run_one(param_set: Dict[str, Any]) -> str:
        sim_id = str(param_set.get("id", "unknown"))
        params_json = json.dumps(param_set)
        cmd = [
            sys.executable, "-m", "swe2d.cli", "run",
            mesh_gpkg, params_json,
            "--results", results_gpkg,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=7200
        )
        if result.returncode != 0:
            return f"{sim_id}: FAILED ({result.stderr.strip()[:200]})"
        return f"{sim_id}: OK"

    try:
        print(f"Running {len(all_params)} simulations ({max_workers} workers)...")
        done = 0
        failed = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_run_one, ps): ps.get("id", f"sim_{i}")
                       for i, ps in enumerate(all_params)}
            for future in concurrent.futures.as_completed(futures):
                sid = futures[future]
                try:
                    msg = future.result()
                except Exception as e:
                    msg = f"{sid}: EXCEPTION ({e})"
                if "FAILED" in msg or "EXCEPTION" in msg:
                    failed += 1
                done += 1
                elapsed = time.perf_counter() - stime
                print(f"  [{done}/{len(all_params)}] {msg}")
                if status_callback:
                    status_callback(done, len(all_params), failed, elapsed)

        elapsed = time.perf_counter() - stime
        print(f"Batch complete: {done}/{len(all_params)} in {elapsed:.1f}s")
        if status_callback:
            status_callback(done, len(all_params), failed, elapsed)
    finally:
        _stop_mps_if_we_started(mps_started)


class BatchOrchestrator:
    """Owns subprocess pool lifecycle for batch simulation runs.

    Accepts pre-built param_sets, constructs ``swe2d run`` commands, and
    launches them via :func:`subprocess.Popen`.  Callbacks fire on
    completion so the caller (dialog or CLI) can update its own state.
    """

    def __init__(
        self,
        param_sets: List[Dict[str, Any]],
        workdir: str,
        mesh_gpkg: str = "",
        max_workers: int = 0,
        on_progress: Optional[Callable[[int, int], None]] = None,
        on_completed: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_failed: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self._param_sets = param_sets
        self._workdir = workdir
        self._mesh_gpkg = mesh_gpkg
        self._max_workers = max_workers if max_workers > 0 else min(len(param_sets), 4)
        self._on_progress = on_progress
        self._on_completed = on_completed
        self._on_failed = on_failed
        self._cancelled = False
        self._active_procs: list = []

    def run(self) -> List[Dict[str, Any]]:
        """Execute all param sets and return a list of result dicts."""
        results: List[Dict[str, Any]] = []
        for idx, ps in enumerate(self._param_sets):
            if self._cancelled:
                break
            sim_id = str(ps.get("id", f"sim_{idx}"))
            params_json = json.dumps(ps)
            cmd = [
                sys.executable, "-m", "swe2d.cli", "run",
                self._mesh_gpkg, params_json,
            ]
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            self._active_procs.append(proc)
            # Handle both real Popen (has poll()) and test fakes (has returncode).
            if hasattr(proc, "poll"):
                while proc.poll() is None:
                    time.sleep(0.05)
                rc = proc.returncode
            else:
                rc = proc.returncode
            stdout_text = (proc.stdout.read() if hasattr(proc, "stdout") and proc.stdout else "")
            stderr_text = (proc.stderr.read() if hasattr(proc, "stderr") and proc.stderr else "")
            status = "completed" if rc == 0 else "failed"
            result = {
                "id": sim_id,
                "status": status,
                "returncode": rc,
                "stdout": stdout_text,
                "stderr": stderr_text,
            }
            results.append(result)
            if self._on_progress:
                self._on_progress(len(results), len(self._param_sets))
            if rc == 0 and self._on_completed:
                self._on_completed(result)
            elif rc != 0 and self._on_failed:
                self._on_failed(result)
        return results

    def cancel(self) -> None:
        """Terminate any running subprocesses and stop the batch."""
        self._cancelled = True
        for proc in self._active_procs:
            if proc.poll() is None:
                proc.terminate()
