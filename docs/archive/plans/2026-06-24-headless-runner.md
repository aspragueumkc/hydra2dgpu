---
type: plan
status: complete
created: 2026-06-24
completed: 2026-07-25
---

# Headless Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run a simulation from command line without QGIS, using GPKG-stored mesh and direct SQL for boundary/hyetograph data.

**Architecture:** New `swe2d/cli/` package with a GPKG adapter (reads spatial data via sqlite3 instead of QGIS layer API) and a headless runner (wraps the existing `swe2d/runtime/` pipeline). The CLI entry point (`hydra run`) is a thin script that parses JSON params, calls the adapter + runner, and writes results.

**Tech Stack:** numpy, sqlite3, shapely (optional for geometry), existing `swe2d/runtime/` modules

---

### Task 1: Create `swe2d/cli/` package structure

**Files:**
- Create: `swe2d/cli/__init__.py`
- Create: `swe2d/cli/gpkg_adapter.py`

- [ ] **Step 1: Create `swe2d/cli/__init__.py`**

```python
"""Headless CLI tools for HYDRA2DGPU."""
```

- [ ] **Step 2: Create `swe2d/cli/gpkg_adapter.py`**

This file provides QGIS-free direct-sql read functions for data that the workbench currently reads via QGIS layers.

```python
"""GPKG adapter: read forcing data directly from GeoPackage without QGIS.

Each function mirrors a QGIS-layer-reader in the workbench but uses sqlite3
directly.  Returns the same Python objects (numpy arrays, ThiessenRainCNForcing,
etc.) so the existing runtime pipeline works unchanged.
"""
from __future__ import annotations

import sqlite3
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from swe2d.boundary_and_forcing.rainfall_hydrology import (
    Hyetograph,
    ThiessenRainCNForcing,
    build_hyetograph,
)


def query_mesh_from_gpkg(gpkg_path: str, mesh_name: str) -> Optional[Dict[str, np.ndarray]]:
    """Load mesh arrays from swe2d_mesh table (delegates to persistence service)."""
    from swe2d.workbench.services.gpkg_persistence_service import load_mesh_from_geopackage
    return load_mesh_from_geopackage(gpkg_path, mesh_name)


def query_bc_arrays(conn: sqlite3.Connection, bc_table: str) -> Dict[str, np.ndarray]:
    """Read boundary condition edge arrays from a BC lines layer table.

    Expects table with columns: node0 INTEGER, node1 INTEGER, bc_type INTEGER, bc_val REAL.
    Returns dict with keys: bc_edge_node0, bc_edge_node1, bc_edge_type, bc_edge_val.
    """
    cur = conn.cursor()
    cur.execute(f"SELECT node0, node1, bc_type, bc_val FROM \"{bc_table}\" ORDER BY rowid")
    rows = cur.fetchall()
    if not rows:
        return {}
    out = {
        "bc_edge_node0": np.array([r[0] for r in rows], dtype=np.int32),
        "bc_edge_node1": np.array([r[1] for r in rows], dtype=np.int32),
        "bc_edge_type": np.array([r[2] for r in rows], dtype=np.int32),
        "bc_edge_val": np.array([r[3] for r in rows], dtype=np.float64),
    }
    return out


def query_hyetograph_rows(
    conn: sqlite3.Connection,
    hyetograph_table: str,
    hyetograph_id_field: str = "hyetograph_id",
    time_field: str = "Time",
    value_field: str = "Value",
    value_type_field: str = "value_type",
    units_field: str = "units",
) -> Dict[str, List[Dict[str, Any]]]:
    """Read hyetograph rows grouped by hyetograph_id.

    Returns dict mapping hyetograph_id -> list of row dicts for build_hyetograph().
    """
    cur = conn.cursor()
    cur.execute(f"SELECT DISTINCT \"{hyetograph_id_field}\" FROM \"{hyetograph_table}\"")
    ids = [r[0] for r in cur.fetchall()]
    result: Dict[str, List[Dict[str, Any]]] = {}
    for hid in ids:
        cur.execute(
            f"SELECT \"{time_field}\", \"{value_field}\", "
            f"\"{value_type_field}\", \"{units_field}\" "
            f"FROM \"{hyetograph_table}\" "
            f"WHERE \"{hyetograph_id_field}\" = ? ORDER BY rowid",
            (hid,),
        )
        rows = []
        for time_val, value_val, vt, u in cur.fetchall():
            rows.append({
                "Time": str(time_val),
                "Value": float(value_val),
                "value_type": str(vt),
                "units": str(u),
            })
        result[str(hid)] = rows
    return result


def query_gauge_layer(
    conn: sqlite3.Connection,
    gauge_table: str,
    gauge_id_field: str = "gage_id",
    hyetograph_id_field: str = "hyetograph_id",
    x_field: str = "x",
    y_field: str = "y",
) -> List[Dict[str, Any]]:
    """Read gauge positions from a rain gage layer table."""
    cur = conn.cursor()
    cur.execute(
        f"SELECT \"{gauge_id_field}\", \"{hyetograph_id_field}\", "
        f"\"{x_field}\", \"{y_field}\" FROM \"{gauge_table}\" ORDER BY rowid"
    )
    return [
        {"gauge_id": str(r[0]), "hyetograph_id": str(r[1]), "x": float(r[2]), "y": float(r[3])}
        for r in cur.fetchall()
    ]


def query_cn_grid(
    conn: sqlite3.Connection,
    cn_table: str,
    cn_field: str = "cn",
    ia_ratio_field: str = "ia_ratio",
    cell_id_field: str = "cell_id",
) -> Tuple[np.ndarray, float]:
    """Read per-cell curve number array and Ia ratio from a CN raster table."""
    cur = conn.cursor()
    try:
        cur.execute(f"SELECT \"{cn_field}\" FROM \"{cn_table}\" ORDER BY rowid")
        cn = np.array([float(r[0]) for r in cur.fetchall()], dtype=np.float64)
    except Exception:
        cn = np.empty(0, dtype=np.float64)
    try:
        cur.execute(f"SELECT \"{ia_ratio_field}\" FROM \"{cn_table}\" LIMIT 1")
        row = cur.fetchone()
        ia_ratio = float(row[0]) if row else 0.2
    except Exception:
        ia_ratio = 0.2
    return cn, ia_ratio


def build_forced_thiessen_from_gpkg(
    conn: sqlite3.Connection,
    n_cells: int,
    mesh_node_x: np.ndarray,
    mesh_node_y: np.ndarray,
    cell_nodes: np.ndarray,
    *,
    hyetograph_table: str,
    gauge_table: str,
    cn_table: Optional[str] = None,
    cn_field: str = "cn",
    ia_ratio_field: str = "ia_ratio",
    hyetograph_id_field: str = "hyetograph_id",
    gauge_id_field: str = "gage_id",
    x_field: str = "x",
    y_field: str = "y",
    time_field: str = "Time",
    value_field: str = "Value",
    value_type_field: str = "value_type",
    units_field: str = "units",
    infiltration_method: str = "scs_cn",
) -> Optional[ThiessenRainCNForcing]:
    """Build ThiessenRainCNForcing directly from GPKG tables.

    Mirrors swe2d/boundary_and_forcing/spatial_forcing_qgis_adapter.py
    but reads from raw GPKG tables instead of QGIS vector layers.
    """
    from swe2d.boundary_and_forcing.rainfall_hydrology import (
        ThiessenRainCNForcing,
    )

    gauge_rows = query_gauge_layer(
        conn, gauge_table, gauge_id_field=gauge_id_field,
        hyetograph_id_field=hyetograph_id_field,
        x_field=x_field, y_field=y_field,
    )
    if not gauge_rows:
        return None

    hy_rows_by_id = query_hyetograph_rows(
        conn, hyetograph_table,
        hyetograph_id_field=hyetograph_id_field,
        time_field=time_field, value_field=value_field,
        value_type_field=value_type_field, units_field=units_field,
    )

    gauges = []
    hy_by_gauge_index: Dict[int, Hyetograph] = {}
    for gi, gr in enumerate(gauge_rows):
        hid = gr["hyetograph_id"]
        if hid not in hy_rows_by_id:
            continue
        hy = build_hyetograph(hy_rows_by_id[hid])
        gauges.append({
            "gauge_id": gr["gauge_id"],
            "x": gr["x"],
            "y": gr["y"],
            "hyetograph_id": hid,
        })
        hy_by_gauge_index[gi] = hy

    if not gauges:
        return None

    gx = np.array([g["x"] for g in gauges], dtype=np.float64)
    gy = np.array([g["y"] for g in gauges], dtype=np.float64)

    cell_centroids = _compute_cell_centroids(mesh_node_x, mesh_node_y, cell_nodes)
    n_cells_actual = min(cell_centroids.shape[0], n_cells)
    cx = cell_centroids[:n_cells_actual, 0]
    cy = cell_centroids[:n_cells_actual, 1]

    # Nearest-gauge assignment
    cell_to_gauge = np.full(n_cells_actual, -1, dtype=np.int32)
    for ci in range(n_cells_actual):
        dist = np.hypot(gx - cx[ci], gy - cy[ci])
        cell_to_gauge[ci] = int(np.argmin(dist))

    cn_arr, ia_ratio = query_cn_grid(
        conn, cn_table or "swe2d_rain_cn",
        cn_field=cn_field, ia_ratio_field=ia_ratio_field,
    ) if cn_table else (np.full(n_cells_actual, 75.0, dtype=np.float64), 0.2)

    if cn_arr.size != n_cells_actual:
        cn_arr = np.full(n_cells_actual, float(cn_arr.mean()) if cn_arr.size > 0 else 75.0, dtype=np.float64)

    return ThiessenRainCNForcing(
        cell_to_gauge=cell_to_gauge,
        gauge_hyetographs=hy_by_gauge_index,
        curve_number=cn_arr,
        ia_ratio=float(ia_ratio),
        infiltration_method=str(infiltration_method),
    )


def _compute_cell_centroids(
    node_x: np.ndarray, node_y: np.ndarray, cell_nodes: np.ndarray
) -> np.ndarray:
    """Compute cell centroids from mesh topology.
    Simple average of cell vertex coordinates.
    """
    tris = cell_nodes.reshape((-1, 3))
    cx = np.mean(node_x[tris], axis=1)
    cy = np.mean(node_y[tris], axis=1)
    return np.column_stack((cx, cy))
```

- [ ] **Step 3: Syntax check**

```bash
python3 -c "import py_compile; py_compile.compile('swe2d/cli/gpkg_adapter.py', doraise=True); print('OK')"
```

Expected: OK

- [ ] **Step 4: Commit**

```bash
git add swe2d/cli/ && git commit -m "feat: add swe2d/cli/ package with GPKG adapter for QGIS-free reads"
```

### Task 2: Headless runner (`hydra run` command)

**Files:**
- Create: `swe2d/cli/headless_runner.py`
- Create: `swe2d/cli/__main__.py`

- [ ] **Step 1: Create `swe2d/cli/headless_runner.py`**

```python
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
        # Run with no interval snapshots, just max tracking
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
        # Run with interval snapshots
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
```

- [ ] **Step 2: Create `swe2d/cli/__main__.py`**

```python
"""CLI entry point: python -m swe2d.cli run mesh.gpkg params.json [--results out.gpkg]"""
import argparse
import json
import sys
import os

from swe2d.cli.headless_runner import execute_run


def main():
    parser = argparse.ArgumentParser(description="HYDRA2DGPU headless runner")
    sub = parser.add_subparsers(dest="command")

    run_parser = sub.add_parser("run", help="Run a single simulation")
    run_parser.add_argument("mesh_gpkg", help="Path to mesh GeoPackage")
    run_parser.add_argument("params", help="Path to JSON params file, or JSON string")
    run_parser.add_argument("--results", "-r", default="", help="Results GeoPackage path")
    run_parser.add_argument("--progress", action="store_true", help="Print progress per step")

    batch_parser = sub.add_parser("batch", help="Run batch of simulations")
    batch_parser.add_argument("batch_json", help="Path to batch JSON file")
    batch_parser.add_argument("mesh_gpkg", help="Path to mesh GeoPackage")
    batch_parser.add_argument("--results", "-r", default="", help="Results GeoPackage path")
    batch_parser.add_argument("--max-workers", "-w", type=int, default=0, help="Max concurrent workers")

    args = parser.parse_args()

    if args.command == "run":
        params = _load_params(args.params)
        results = execute_run(
            args.mesh_gpkg,
            params,
            results_gpkg=args.results or "",
            progress_callback=_make_progress(args.progress),
        )
        print(f"Run complete: {results['h'].size} cells, {len(results['diags'])} steps")
        if "max_results" in results:
            print(f"Max tracking: h_max range [{results['max_results']['max_h'].min():.6f}, {results['max_results']['max_h'].max():.6f}]")

    elif args.command == "batch":
        from swe2d.cli.batch_runner import run_batch
        run_batch(args.batch_json, args.mesh_gpkg, args.results, args.max_workers)

    else:
        parser.print_help()


def _load_params(param_source: str) -> dict:
    """Load params from JSON file or string."""
    s = str(param_source).strip()
    if os.path.isfile(s):
        with open(s) as f:
            return json.load(f)
    return json.loads(s)


def _make_progress(enabled: bool):
    if not enabled:
        return None
    def cb(t, diag):
        print(f"  t={t:.3f}s  dt={diag.get('dt', 0):.5f}  wet={diag.get('wet_cells', -1)}", file=sys.stderr)
    return cb


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Syntax check**

```bash
python3 -c "
import py_compile
for f in ['swe2d/cli/__init__.py', 'swe2d/cli/gpkg_adapter.py', 'swe2d/cli/headless_runner.py', 'swe2d/cli/__main__.py']:
    py_compile.compile(f, doraise=True)
    print(f'{f}: OK')
"
```

Expected: OK

- [ ] **Step 4: Commit**

```bash
git add swe2d/cli/ && git commit -m "feat: add headless runner (hydra run) with GPKG adapter"
```

### Task 3: Batch runner

**Files:**
- Create: `swe2d/cli/batch_runner.py`

- [ ] **Step 1: Create batch runner**

```python
"""Batch runner: execute multiple simulations via subprocess isolation."""
from __future__ import annotations

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
        p = dict(params)
        combo_dict = dict(zip(keys, combo))
        # Apply swept values into nested params
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

    # Auto-detect max_workers
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
```

- [ ] **Step 2: Syntax check**

```bash
python3 -c "import py_compile; py_compile.compile('swe2d/cli/batch_runner.py', doraise=True); print('OK')"
```

Expected: OK

- [ ] **Step 3: Commit**

```bash
git add swe2d/cli/batch_runner.py && git commit -m "feat: add batch runner with subprocess pool and sweep expansion"
```

### Task 4: Integration smoke test

**Files:**
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write smoke test**

```python
"""Smoke tests for CLI headless runner."""
import json
import os
import tempfile
import numpy as np

from swe2d.workbench.services.gpkg_persistence_service import (
    persist_mesh_to_geopackage,
)


def test_sweep_expansion_simple():
    """_expand_sweep produces correct Cartesian product."""
    from swe2d.cli.batch_runner import _expand_sweep

    params = {
        "sweep": {
            "params.n_mann": [0.020, 0.030, 0.040],
        },
        "id_template": "n_{n_mann:.3f}",
        "params": {"duration_s": 3600},
    }
    expanded = _expand_sweep(params)
    assert len(expanded) == 3
    assert expanded[0]["params"]["n_mann"] == 0.020
    assert expanded[1]["params"]["n_mann"] == 0.030
    assert expanded[2]["params"]["n_mann"] == 0.040
    assert expanded[0]["id"] == "n_0.020"


def test_sweep_expansion_layer():
    """Sweep over a layer reference (string values)."""
    from swe2d.cli.batch_runner import _expand_sweep

    params = {
        "sweep": {
            "mannings_layer": ["landuse_a", "landuse_b"],
        },
        "id_template": "{mannings_layer}",
        "params": {"duration_s": 3600},
    }
    expanded = _expand_sweep(params)
    assert len(expanded) == 2
    assert expanded[0]["mannings_layer"] == "landuse_a"
    assert expanded[1]["mannings_layer"] == "landuse_b"


def test_mesh_persist_and_load_round_trip():
    """Full round trip: build small mesh, save to GPKG, load back."""
    mesh_data = {
        "node_x": np.array([0.0, 10.0, 5.0], dtype=np.float64),
        "node_y": np.array([0.0, 0.0, 10.0], dtype=np.float64),
        "node_z": np.array([5.0, 5.0, 4.0], dtype=np.float64),
        "cell_nodes": np.array([0, 1, 2], dtype=np.int32),
    }
    with tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False) as f:
        gpkg = f.name
    try:
        persist_mesh_to_geopackage(gpkg, "test", mesh_data)
        from swe2d.cli.gpkg_adapter import query_mesh_from_gpkg
        loaded = query_mesh_from_gpkg(gpkg, "test")
        assert loaded is not None
        for k in ("node_x", "node_y", "node_z", "cell_nodes"):
            np.testing.assert_array_almost_equal(loaded[k], mesh_data[k])
    finally:
        if os.path.exists(gpkg):
            os.unlink(gpkg)
```

- [ ] **Step 2: Run tests**

```bash
python3 -m pytest tests/test_cli.py -v
```

Expected: 3 PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_cli.py && git commit -m "test: add CLI unit tests (sweep expansion, mesh round-trip)"
```
