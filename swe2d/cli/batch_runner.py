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
    sweep = params.pop("sweep", None)
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
