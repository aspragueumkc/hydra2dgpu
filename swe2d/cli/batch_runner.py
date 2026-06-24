"""Batch runner: execute multiple simulations via subprocess isolation."""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import time
import concurrent.futures
from typing import Any, Dict, List, Optional


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
) -> None:
    """Read batch JSON, expand sweeps, run sims in subprocess pool."""
    with open(batch_json_path) as f:
        batch_config = json.load(f)

    if isinstance(batch_config, list):
        param_sets = batch_config
    elif isinstance(batch_config, dict):
        param_sets = [batch_config]
    else:
        raise ValueError("batch JSON must be an array or object")

    # Expand sweeps
    all_params: List[Dict[str, Any]] = []
    for ps in param_sets:
        all_params.extend(_expand_sweep(dict(ps)))

    if not all_params:
        print("No param sets to run.")
        return

    if max_workers <= 0:
        max_workers = min(len(all_params), 4)

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

    print(f"Running {len(all_params)} simulations ({max_workers} workers)...")
    done = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_one, ps): ps.get("id", f"sim_{i}")
                   for i, ps in enumerate(all_params)}
        for future in concurrent.futures.as_completed(futures):
            sid = futures[future]
            try:
                msg = future.result()
            except Exception as e:
                msg = f"{sid}: EXCEPTION ({e})"
            done += 1
            print(f"  [{done}/{len(all_params)}] {msg}")

    elapsed = time.perf_counter() - stime
    print(f"Batch complete: {done}/{len(all_params)} in {elapsed:.1f}s")
