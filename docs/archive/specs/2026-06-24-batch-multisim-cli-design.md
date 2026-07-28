---
type: spec
status: complete
created: 2026-06-24
completed: 2026-07-25
---

# Batch Multi-Sim, CLI, and Mesh Persistence Design

## Overview

Three tightly-coupled capabilities:

1. **Mesh persistence** — save/load solver mesh arrays from GeoPackage BLOBs
2. **Headless runner** — run a simulation from command line without QGIS
3. **Batch multi-sim** — concurrent/sequential execution of multiple simulations
   with different parameters, sharing mesh + data from a single GPKG

## Architecture

```
hydra/                          # CLI package (new)
  __init__.py
  __main__.py                   # python -m hydra
  run.py                        # single simulation
  batch.py                      # multi-sim orchestration
  mesh.py                       # mesh export/import

swe2d/cli/                      # shared runtime (new, QGIS-free path)
  headless_runner.py            # execute_run() without QGIS widgets
  gpkg_adapter.py               # direct GPKG reads for BC/hyetograph/layers

swe2d/runtime/                  # existing — no QGIS dependency
  backend.py
  runtime_step_executor.py
  runtime_sources.py
  run_options_builder.py
  ...

swe2d/workbench/                # existing — QGIS UI
  controllers/run_controller.py  # calls execute_run() shared path
  ...
```

## Sub-Project 1: Mesh Persistence

### GPKG Schema

```sql
-- One row per saved mesh
CREATE TABLE swe2d_mesh (
    mesh_name TEXT PRIMARY KEY,
    created_utc TEXT,
    nnodes INTEGER,
    ncells INTEGER,
    crs_wkt TEXT,
    hash TEXT,               -- sha256 of all BLOB content for dedup
    node_x BLOB,             -- float64, zlib-compressed
    node_y BLOB,
    node_z BLOB,
    cell_nodes BLOB,         -- int32, flat (triplets or polygon CSR)
    face_offsets BLOB,       -- int32, nullable (null = triangles)
    face_nodes BLOB,         -- int32, nullable
    bc_n0 BLOB,              -- int32, nullable (boundary edges captured at mesh time)
    bc_n1 BLOB,
    bc_type BLOB,
    bc_val BLOB,
    terrain_source TEXT,     -- "geotiff" or "gpkg"
    terrain_path TEXT,       -- path or table name
    description TEXT
);
```

Arrays serialized as raw numpy bytes (`ndarray.tobytes()`), compressed with
zlib before storage.  Decompressed on read and passed directly to
`SWE2DBackend.build_mesh()`.

### Backend Methods

```python
class SWE2DBackend:
    def export_mesh_data(self) -> dict:
        """Return dict of mesh arrays: node_x/y/z, cell_nodes, bc arrays."""
        ...

# Standalone functions in swe2d/cli/gpkg_adapter.py (or gpkg_persistence_service)
def persist_mesh_to_geopackage(gpkg_path, mesh_name, mesh_data, crs_wkt="", description=""):
    """Compress and write mesh BLOBs to swe2d_mesh table."""

def load_mesh_from_geopackage(gpkg_path, mesh_name) -> dict:
    """Read and decompress mesh BLOBs from swe2d_mesh table."""
```

### QGIS UI

- After mesh generation, a "Save mesh to GPKG..." button prompts for name.
- Mesh loading combo box shows available meshes from the project GPKG.
- Selecting a saved mesh populates `_mesh_data` for the workbench.

## Sub-Project 2: Headless Runner

### CLI Entry Point

```
hydra run <mesh.gpkg> <params.json> [--results results.gpkg]
```

`params.json` schema:

```json
{
  "mesh": "kansas_river_mesh",
  "terrain": {
    "source": "geotiff",
    "path": "/data/dem_1m.tif"
  },
  "bc_lines": "swe2d_bc_lines",
  "hyetograph": {
    "table": "hyetographs",
    "gauge_layer": "rain_gages"
  },
  "rain_cn": {
    "table": "rain_cn_zones",
    "cn_field": "cn",
    "ia_ratio": 0.2
  },
  "drain_nodes": null,
  "structures": null,
  "params": {
    "rain_rate_mmhr": 0,
    "n_mann": 0.035,
    "h_min": 1e-4,
    "duration_s": 3600,
    "dt_cfg": 0.2,
    "cfl": 0.45,
    "reconstruction_mode": 1,
    "temporal_scheme": 1,
    "gpu_diag_sync_interval_steps": 100,
    "output_interval_s": 300,
    "save_mesh_results": true,
    "save_max_only": false
  }
}
```

### Execution Flow

1. Open GPKG, query `swe2d_mesh` table → decompress BLOBs → `build_mesh()`
2. Query BC layer, hyetograph layer, rain CN layer from GPKG directly via SQL
3. Configure Thiessen forcing, native rain, native BC hydrographs
4. `initialize()` → step loop → finalize
5. Write results to `--results` GPKG (or back to source GPKG)
6. If `save_max_only`, persist max tracking results instead of interval snapshots

### QGIS-Free Dependency Chain

```
swe2d/runtime/backend.py              ← hydra_swe2d (.so)
swe2d/runtime/runtime_step_executor.py
swe2d/runtime/runtime_sources.py
swe2d/runtime/runtime_setup_configurator.py
swe2d/runtime/run_options_builder.py
swe2d/runtime/run_finalizer.py
swe2d/boundary_and_forcing/            ← rainfall_hydrology, bc_logic
swe2d/extensions/                      ← structure/coupling models
swe2d/mesh/                            ← mesh models, runtime logic
numpy, sqlite3                          ← stdlib
```

No PyQt5, no qgis.core, no QgsApplication.

## Sub-Project 3: Batch Multi-Sim

### CLI Entry Point

```
hydra batch <batch.json> <mesh.gpkg> [--results results.gpkg] [--max-workers N]
```

`batch.json` is a JSON array of per-sim parameter sets:

```json
[
  {
    "id": "baseline",
    "mesh": "kansas_river_mesh",
    "params": { "n_mann": 0.035, "duration_s": 3600 }
  },
  {
    "id": "mann_002",
    "mesh": "kansas_river_mesh",
    "params": { "n_mann": 0.020, "duration_s": 3600 }
  },
  {
    "id": "100yr_storm",
    "mesh": "kansas_river_mesh",
    "hyetograph": { "table": "hyetographs", "gauge_layer": "rain_gages_100yr" },
    "params": { "duration_s": 7200 }
  }
]
```

Each entry merges with a baseline config (same layers, mesh, terrain).  Only
overridden fields need to appear.

### Sweep Expansion

A `sweep` key in any param set expands into multiple sims automatically.
Sweep keys refer to **any JSON path** in the config — not just scalar
`params.*`.  A dotted key like `mannings_layer` or `hyetograph.table` replaces
the corresponding field in the config before passing it to `hydra run`.

**Scalar sweep** (vary a single numeric parameter):

```json
{
  "sweep": {
    "params.n_mann": [0.020, 0.030, 0.040, 0.050, 0.060, 0.080]
  },
  "id_template": "n_{n_mann:.3f}"
}
```

**Layer/reference sweep** (vary a spatial input layer):

```json
{
  "sweep": {
    "mannings_layer": ["landuse_current", "landuse_forested", "landuse_developed"],
    "hyetograph.table": ["design_storm_10yr", "design_storm_100yr"]
  },
  "id_template": "{mannings_layer}_{hyetograph.table}"
}
```

The batch runner takes the Cartesian product of all sweep arrays, merges
each combination into the baseline config, and assigns an ID from
`id_template` (interpolated from the swept values).  The underlying
`hydra run` call is identical — the sweep layer is purely a pre-processing
step in the batch orchestrator.

### Execution Model: Subprocess Isolation

```python
import subprocess
import concurrent.futures

def run_sim(param_set, mesh_gpkg, results_gpkg, max_workers):
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = []
        for ps in param_set:
            params_json = json.dumps(ps)
            futures.append(pool.submit(
                run_one, mesh_gpkg, params_json, results_gpkg
            ))
        for future in concurrent.futures.as_completed(futures):
            # collect results, update progress
            pass

def run_one(mesh_gpkg, params_json, results_gpkg):
    """Run in a subprocess — clean hydra_swe2d load, clean CUDA context."""
    subprocess.run([
        sys.executable, "-m", "hydra", "run",
        mesh_gpkg, params_json,
        "--results", results_gpkg,
    ], check=True)
```

Each subprocess (`hydra run`) is a fresh Python process:
- Loads its own `hydra_swe2d` module
- Creates its own CUDA context on the same device
- GPU time-slices via CUDA streams automatically
- No `s_coupling_dev` conflict — each process has its own global

`--max-workers` defaults to the GPU's VRAM-based heuristic (e.g., 4 for 471K
cells, 10+ for 20K cells) with an override.

### Results Storage

All batch results go into the same results GPKG.  Each sim writes to
`swe2d_run_results` with an `sim_id` column (from the batch JSON `id` field).
The batch orchestrator creates a summary table:

```sql
CREATE TABLE swe2d_batch_runs (
    batch_id TEXT,
    sim_id TEXT,
    status TEXT,         -- queued / running / completed / failed
    started_utc TEXT,
    completed_utc TEXT,
    error TEXT,
    wallclock_s REAL,
    PRIMARY KEY (batch_id, sim_id)
);
```

## Sub-Project 4: QGIS Batch UI

- New dialog: "Batch Simulation" with a parameter table grid
- Rows = simulation variants, columns = overridable parameters
- Populate from selected mesh + default layers
- Export/import batch config as JSON
- Launch via subprocess, monitor progress bar, cancel individual sims
- Browse results across sims in the results panel with a "variant" selector

## Sub-Project 5: Alternative / Frequency Analysis

- Post-processing across batch results
- For each cell, compute statistics across sims: min, max, mean, std, percentiles
- Store in `swe2d_batch_stats` table
- UI toggle between individual variant view and "max envelope" view

## Implementation Order

| Step | What | Depends On |
|------|------|------------|
| 1 | Mesh persistence (GPKG schema + export/import + backend bindings) | Nothing |
| 2 | Headless runner QGIS-free path (gpkg_adapter, headless_runner) | Step 1 |
| 3 | `hydra run` CLI entry point | Step 2 |
| 4 | `hydra batch` with subprocess pool | Step 3 |
| 5 | QGIS batch UI + progress monitoring | Steps 1, 4 |
| 6 | Alternative analysis (post-processing, stats) | Step 4 |

Each step is independently testable and mergable.

## SPOTPY Integration (Documentation Only)

[SPOTPY](https://github.com/thouska/spotpy) is an optional external dependency
for formal sensitivity analysis and parameter optimization.  No integration
code lives in `hydra`.  A user writes their own driver script:

```python
import spotpy
import subprocess, json, sqlite3
import numpy as np

class HydraCalibration(spotpy.objectivefunctions._objectivefunction):
    def parameters(self):
        return [spotpy.parameter.Uniform("n_mann", 0.015, 0.100)]

    def evaluation(self):
        return observed_wse  # or None for pure SA sweep

    def simulation(self, params):
        cfg = load_baseline()          # base params.json
        cfg["params"]["n_mann"] = params["n_mann"]
        cfg_json = json.dumps(cfg)
        subprocess.run(["hydra", "run", "mesh.gpkg", cfg_json, "results.gpkg"], check=True)
        conn = sqlite3.connect("results.gpkg")
        wse = np.array(conn.execute("SELECT max_wse FROM swe2d_mesh_max_results").fetchall())
        conn.close()
        return wse

sampler = spotpy.algorithms.latinhypercube(HydraCalibration(), dbname="sa_results")
sampler.sample(100)
```

The only requirement from `hydra` is a stable param JSON schema and a stable
results GPKG schema — both documented in this spec.  No API hooks, no
callbacks, no plugin system needed.

## Deferred (Not in Scope)

- Real-time inter-sim communication / coupling
- Distributed GPU (multi-node MPI)
- Adaptive mesh refinement
