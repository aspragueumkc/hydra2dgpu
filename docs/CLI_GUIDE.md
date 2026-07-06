# Headless CLI Guide

> **Audience**: users running simulations outside QGIS, batch sweeps, CI/CD
> pipelines, and anyone who wants to script the solver.

The CLI runs the GPU solver directly from a Python process with no QGIS or
Qt dependency. It reads a mesh from a GeoPackage, applies a JSON params
file, writes results to a separate GeoPackage, and reports progress via
stdin or a JSON status file.

The CLI module lives at `swe2d/cli/`:

| File | Purpose |
|------|---------|
| `__main__.py` | Argparse entry point — `python -m swe2d.cli run\|batch` |
| `headless_runner.py` | `execute_run()` — full simulation pipeline |
| `batch_runner.py` | Concurrent batch runs via MPS daemon |
| `gpkg_adapter.py` | Direct sqlite3 reads of mesh/BC/forcing data |

---

## Quick Start

### 1. Run a single simulation

```bash
python -m swe2d.cli run \
    mesh.gpkg \
    params.json \
    --results out.gpkg \
    --progress
```

| Argument | Required | Purpose |
|----------|:--------:|---------|
| `mesh_gpkg` | ✅ | GeoPackage containing the baked mesh (`swe2d_baked_mesh` table) |
| `params` | ✅ | Path to JSON file, or an inline JSON string |
| `--results`, `-r` | ❌ | Output GeoPackage for snapshots, line results, coupling timeseries |
| `--progress` | ❌ | Print per-step progress (t, dt, wet_cells) to stderr |
| `--status-file-path` | ❌ | Write periodic JSON status file (see [Status Monitoring](#status-monitoring)) |
| `--status-interval` | ❌ | Status write interval in seconds (default 5.0) |

### 2. Run a batch of simulations

```bash
python -m swe2d.cli batch \
    batch.json \
    mesh.gpkg \
    --results out.gpkg \
    --max-workers 4
```

| Argument | Required | Purpose |
|----------|:--------:|---------|
| `batch_json` | ✅ | JSON file listing parameter sets to run |
| `mesh_gpkg` | ✅ | Shared mesh GeoPackage |
| `--results`, `-r` | ❌ | Output GeoPackage (all runs appended) |
| `--max-workers`, `-w` | ❌ | Concurrent workers (0 = auto; requires NVIDIA MPS) |

The batch runner uses **NVIDIA MPS** (Multi-Process Service) to allow
multiple GPU processes to share the device concurrently. If MPS is not
available, runs execute sequentially.

---

## Parameters JSON

The params file is the same shape as the Studio UI's persisted settings
(JSON snapshot under the `workbench_state_json` project key, minus the
widget types). Required keys:

```json
{
    "id": "run_001",
    "mesh": "mesh_main",

    "params": {
        "duration_s": 3600.0,
        "output_interval_s": 60.0,
        "dt_request": 0.1,
        "n_mann": 0.035,
        "cfl": 0.45,
        "spatial_scheme": 0,
        "temporal_scheme": 2,
        "extreme_rain_mode": false,
        "source_cfl_beta": 0.25,
        "source_max_substeps": 16
    },

    "bc_lines": { "table": "swe2d_bc_lines" },

    "hyetograph": {
        "table": "swe2d_hyetographs",
        "gauge_layer": "swe2d_rain_gages"
    },
    "rain_cn": { "table": "swe2d_cn_zones", "cn_field": "cn" },

    "drainage": {
        "nodes_layer": "swe2d_drainage_nodes",
        "links_layer": "swe2d_drainage_links",
        "inlets_layer": "swe2d_drainage_inlets",
        "node_inlets_layer": "swe2d_drainage_node_inlets"
    },

    "structures": { "table": "swe2d_structures" },

    "sample_lines": { "table": "swe2d_sample_lines" }
}
```

Each top-level source key accepts a `{ "table": "...", "gpkg": "..." }`
dict. The optional `gpkg` field overrides the default `mesh_gpkg` — useful
when forcing data lives in a separate file.

### Capturing a params file from the Studio UI

The Studio UI persists its current settings to the project's
`workbench_state_json` key on every widget change and on save. To get a
matching CLI params file:

1. Open a project in QGIS and configure the simulation in the UI.
2. Save the project (Ctrl+S).
3. Open the model GPKG in the GPKG Explorer — the `workbench_state_json`
   is stored in the QGIS project file, not the GPKG, so you need to
   save the QGIS project separately.
4. Extract the `workbench_state_json` from the `.qgz` (it's a zip) and
   strip the `widgets.type` fields, leaving only `widgets.value` and the
   non-widget top-level keys shown above.

---

## Status Monitoring

When `--status-file-path /tmp/run_status.json` is set, the CLI writes a
JSON status file every `--status-interval` seconds:

```json
{
    "step": 1234,
    "t": 12.34,
    "dt": 0.05,
    "wet_cells": 1456,
    "elapsed_s": 8.7,
    "status": "running"
}
```

`status` transitions through `"running"` → `"done"` (or `"error"` on
failure). This lets a separate process — typically the QGIS workbench —
monitor headless progress without parsing stdout or reading the results
GPKG. The batch dialog uses this mechanism to show progress bars for
subprocess-launched runs.

Writes are atomic (temp file + rename), so external readers never see a
half-written file.

---

## Output GeoPackage

When `--results` is given, results land in a separate GeoPackage with the
same tables the Studio UI produces. See [Results GeoPackage Schema](RESULTS_GEOPACKAGE_SCHEMA.md)
for column definitions. Key tables:

| Table | Contents |
|-------|----------|
| `swe2d_baked_mesh` | Serialized mesh BLOB (copied from input) |
| `swe2d_baked_results` | `(h, hu, hv)` snapshots per output interval |
| `swe2d_baked_line_ts` | Sample-line timeseries |
| `swe2d_baked_line_profiles` | Cross-section profiles |
| `swe2d_baked_coupling` | Structure + drainage timeseries |

If `--results` is **not** given, the run still completes but results live
only in process memory — useful for short test runs where you only care
about the final state returned via Python.

---

## Programmatic API

Skip the CLI layer entirely and call the runner from Python:

```python
from swe2d.cli.headless_runner import execute_run
import json

with open("params.json") as f:
    params = json.load(f)

results = execute_run(
    mesh_gpkg="mesh.gpkg",
    params=params,
    results_gpkg="out.gpkg",
    progress_callback=lambda t, d: print(f"t={t:.2f}  dt={d['dt']:.4f}"),
)

# results["h"], results["hu"], results["hv"] — final state arrays
# results["max_results"] — max-tracking arrays (if enabled)
# results["diags"] — list of per-step diagnostic dicts
```

For batch runs:

```python
from swe2d.cli.batch_runner import run_batch

run_batch(
    batch_json="batch.json",
    mesh_gpkg="mesh.gpkg",
    results_gpkg="out.gpkg",
    max_workers=4,
)
```

---

## Requirements

- **Python 3.12+** with `numpy`, `gmsh`, and the `osgeo` (GDAL) bindings
- **`hydra_swe2d` native module built** (`cmake .. && make` in `build/`)
- **NVIDIA GPU** with CUDA toolkit (the CLI is GPU-first)
- For batch runs: **NVIDIA MPS** (`nvidia-cuda-mps-control`)

The CLI uses the same `mamba run -n qgis_stable` environment as the
plugin. Activate it first:

```bash
mamba activate qgis_stable
python -m swe2d.cli run mesh.gpkg params.json
```

---

## Troubleshooting

### `Mesh GPKG not found: ...`

The first positional arg is the path to the GeoPackage, not the mesh
name. The mesh name is a key inside the params JSON.

### `Mesh 'mesh_main' not found in mesh.gpkg`

Run `python tools/gmsh_topology_mesher.py ...` first, or generate the
mesh via the Studio UI. The mesh must be baked into the `swe2d_baked_mesh`
table.

### `osgeo` import error

`osgeo` (GDAL Python bindings) is required by `gpkg_adapter.py`. Install
via the QGIS environment:

```bash
mamba run -n qgis_stable pip install gdal==<version-matching-qgis>
```

Or use the system GDAL Python bindings (`apt install python3-gdal` on
Debian/Ubuntu).

### MPS daemon won't start

Batch runs without MPS fall back to sequential execution. To enable
concurrent runs, install the NVIDIA CUDA Toolkit (provides
`nvidia-cuda-mps-control`) and ensure no stale MPS control process is
running (`echo quit | nvidia-cuda-mps-control`).

---

## Related Documentation

- **[Documentation Index](INDEX.md)** — All guides by audience
- **[User Guide](USER_GUIDE.md)** — Capturing params from the Studio UI
- **[Model GeoPackage Schema](MODEL_GEOPACKAGE_SCHEMA.md)** — Input GPKG tables
- **[Results GeoPackage Schema](RESULTS_GEOPACKAGE_SCHEMA.md)** — Output GPKG tables
- **[Gmsh Meshing Guide](GMSH_MESHING_GUIDE.md)** — Generating mesh GPKGs headlessly
- **[Developer Guide](DEVELOPER_GUIDE.md)** — `swe2d.cli` module reference