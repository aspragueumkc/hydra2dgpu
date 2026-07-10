# Implementation Plan Revision 1: GUI-as-Source-of-Truth + Replay JSON

| | |
|---|---|
| **Plan ID** | `SWE2D-RC-2026-07-10-r1` |
| **Status** | Revision (supersedes §1, §2, §3, §9, §11 of base plan) |
| **Owner** | SWE2D runtime + workbench teams |
| **Created** | 2026-07-10 |
| **Supersedes** | [base plan §1.3, §2, §3, §9, §11](2026-07-10-unified-run-controller.md) |
| **Target branch** | `feature/unified-run-controller` |

---

## 0. What changed and why

The base plan assumed CLI and GUI are symmetric and would be **merged toward a middle ground.** That assumption is wrong. The corrected requirement:

1. **GUI behavior is the source of truth.** Kernel inputs (what `SWE2DBackend.initialize` sees) must remain bit-identical to today's GUI run.
2. **CLI behavior changes.** CLI is rewritten to call the same code path the GUI uses, producing identical kernel inputs for the same widget state.
3. **GUI persists a "replay JSON"** into the results GeoPackage on every run. That JSON captures every widget value verbatim.
4. **Round-trip property**: load that JSON into the CLI → the kernel sees the same inputs as the original GUI run → outputs match.

This revision rewrites the affected sections. Phases, file list, and test plan unchanged unless called out below.

---

## 1. Revised problem statement

### 1.1 Today's two paths (unchanged from base plan)

| Path | Endpoints touched | Capture source |
|------|-------------------|----------------|
| **GUI** | `SWE2DBackendInitializer` → `SWE2DBackend.initialize(...)` with 40+ kwargs captured by `collect_run_widget_params()` + `RunContext` | Widgets + callbacks |
| **CLI** | Inline `SWE2DBackend.initialize(...)` with 40+ kwargs read from JSON `params.params` dict | JSON |

Both paths eventually land at the same C++ `swe2d_gpu.cu` kernel, **but they construct `SWE2DBackend` with subtly different defaults, ordering, and pre-processing.** Examples from the trace:

| Detail | GUI behavior | CLI behavior | Risk |
|--------|--------------|--------------|------|
| BC override validation | Skipped — uses baked-mesh BCs | Validates against mesh boundary, falls back | Different kernel inputs |
| `_inv_cell_perm` applied | Before coupling construction | After coupling construction | Coupling sees different cell order |
| Drainage config defaults | Read from layer; `enabled` set in config builder | Set to `True` after construction | Coupling may be silently disabled |
| `_build_redistribution_data()` | Called from worker, post-centroids | Called from CLI, post-centroids | Order-equivalent today but fragile |
| Status file writes | GUI: writes via `_write_status`-equivalent | CLI: writes via `_atomic_write_json` | Different semantics |
| Scheme migration warning | **Absent in GUI path** | Present in CLI | GUI silently accepts old scheme numbers |

### 1.2 What "two files don't get created" means now

The base plan's "single unified controller at `runtime/run_controller.py`" is preserved. **But the controller's job changes from "merge CLI and GUI" to "execute what the GUI captured."**

Concretely:

- The controller accepts **exactly the data structure the GUI builds today** (the `RunContext`).
- The controller calls `SWE2DBackend.initialize(...)` with **exactly the kwargs the GUI passes today** (via `SWE2DBackendInitializer`).
- The CLI's job shrinks to: **read the replay JSON → build a `RunContext` → call the controller.** Everything else is gone.
- The GUI's `SimulationWorker` becomes a thin Qt-thread wrapper around the controller. No behavior change in user-visible terms.

### 1.3 Replay JSON is the contract

The **replay JSON** is the lingua franca between GUI and CLI. Schema:

```jsonc
{
  "schema_version": "swe2d-replay/1",
  "run_id": "swe2d_20260710T123456-0500",
  "mesh": {
    "gpkg_path": "/path/to/model.gpkg",
    "mesh_name": "domain_v3",
    "crs_wkt": "PROJCS[\"...\",GEOGCS[\"...\"],...]"
  },
  "params": {
    // All 40+ solver/BC/coupling/sample-line knobs — flat dict, JSON-serializable.
    // Every key here maps 1:1 to a kwarg the GUI's RunContext passes to backend.initialize
    // or to a setter (set_boundary_conditions, configure_line_sampling, etc.)
    "spatial_scheme": 6,
    "temporal_scheme": 2,
    "reconstruction_mode": 4,
    "n_mann": 0.035,
    "cfl": 0.45,
    "dt_max": 0.2,
    "dt_request": 0.0,
    "initial_dt": 0.05,
    "h_min": 0.0001,
    "max_inv_area": 1000000.0,
    // ... full set, see §3 for canonical list
    "duration_s": 3600.0,
    "output_interval_s": 300.0
  },
  "data_sources": {
    // Tables + (optional) GPKG paths for each optional layer the GUI's view reads from.
    "bc_lines":   {"table": "swe2d_bc_lines",   "gpkg": null},
    "hyetograph": {"table": "swe2d_hyetographs", "gpkg": null, "gauge_layer": "swe2d_rain_gages"},
    "rain_cn":    {"table": "swe2d_cn_zones",    "gpkg": null, "cn_field": "cn"},
    "drainage":   {"nodes_layer": "swe2d_drainage_nodes", "links_layer": "swe2d_drainage_links", "inlets_layer": null, "node_inlets_layer": null, "gpkg": null},
    "sample_lines": {"table": "swe2d_sample_lines", "gpkg": null},
    "structures": null
  },
  // Mesh-relative pointers (resolved at run time, not stored in JSON):
  // - mesh BLOB lives in mesh GPKG (load_baked_mesh)
  // - coupling/BC layers live in model GPKG (config above)
  "results": {
    "results_gpkg_path": "/path/to/results.gpkg",
    "save_line_results": true,
    "save_coupling_results": true,
    "save_mesh_results": true,
    "save_run_log": true,
    "save_max_only": false
  },
  "coupling_soa_blob_b64": null,
  "bridge_stacked_plans_b64": null,
  "h0": null,
  "side_hydrographs": null,
  "edge_hydrographs": null,
  "edge_group_overrides": null
}
```

The schema is **the exact set of fields the GUI's `RunContext` carries today**, flattened to JSON. Two helpers convert:

- `RunContext.from_view(view, request) -> RunContext` (existing; produces DTO from widgets)
- `RunContext.from_replay_json(path_or_dict) -> RunContext` (new; produces DTO from JSON)
- `RunContext.to_replay_json() -> dict` (new; serializes DTO to JSON for storage)

GUI calls `from_view` to build the `RunContext`, then **always** calls `to_replay_json` and writes the result to the results GPKG as part of run completion (see §4). The same JSON can then be replayed via CLI:

```bash
swe2d-cli run --replay-json /path/to/results.gpkg:swe2d_20260710T123456-0500
# or
swe2d-cli run --replay-json-file /path/to/saved_replay.json
```

---

## 2. Revised goals

| # | Goal | Acceptance |
|--:|------|------------|
| **G1** | GUI run output (h, hu, hv, max_results, diags, GPKG contents) is **bit-identical** before and after refactor | Regression test: snapshot of kernel inputs (40+ kwargs) matches pre-refactor capture |
| **G2** | CLI replay of a GUI run produces bit-identical kernel outputs | Round-trip test: GUI run → JSON → CLI run → same h/hu/hv to ULP |
| **G3** | Every GUI run writes a replay JSON row to `swe2d_run_replays` in the results GPKG | Schema migration; every new GUI run has a replay row; old runs have NULL replay |
| **G4** | One execution path — `SWE2DRunController.execute(ctx)` — called by both GUI and CLI | Both paths' body shrinks; only one place calls `backend.initialize(...)` |
| **G5** | CLI's `execute_run()` body becomes `<100` lines | Code size assertion in CI |
| **G6** | `RunContext` becomes the canonical DTO; built by either factory | Type-checked; one definition |
| **G7** | Scheme migration warning + CFL clamping live in one place (the controller) | Both CLI and GUI get them automatically |
| **G8** | GUI's `SimulationWorker` body becomes `<50` lines | Code size assertion in CI |
| **G9** | All GUI dialog buttons, view updates, plot refreshes, snapshot merging behave identically | GUI tests pass unchanged |

---

## 3. The canonical RunContext surface (replaces base plan §3.2)

The existing `RunContext` is GUI-flavored (carries callables for mesh-derived arrays, view callbacks). Two factories today:

```python
# Existing (GUI): captures widget state
def from_view(view, request) -> RunContext: ...
```

One factory is added:

```python
# New: replays a JSON-serialized GUI run
def from_replay_json(payload: dict | str | Path) -> RunContext: ...
```

`from_replay_json` does **exactly what `from_view` does**, but reads from JSON fields instead of widgets. Both produce a `RunContext` with the same field set. The controller is agnostic to which factory built it.

### 3.1 RunContext field set — the canonical list

These are the fields the GUI's `RunContext` carries today (from [`workbench/workers/run_context.py`](../../swe2d/workbench/workers/run_context.py)), grouped by destination in the controller. Every field gets a JSON key.

#### Group A — mesh identity & output paths (5 fields)
| Field | JSON key | Type |
|-------|----------|------|
| `run_id` | `run_id` | str |
| `mesh_name` | `mesh.mesh_name` | str |
| `mesh_crs_wkt` | `mesh.crs_wkt` | str |
| `model_gpkg_path` | `mesh.gpkg_path` | str |
| `results_gpkg_path` | `results.results_gpkg_path` | str |

#### Group B — solver numerical params (40+ fields, all flat in `params`)
Every key here is a kwarg to `SWE2DBackend.initialize(...)`. **No nested structures** — all flat scalar/bool values:

```
spatial_scheme, temporal_scheme, reconstruction_mode,
k_mann, n_mann, h_min, cfl, dt_max, dt_initial, dt_request, dt_fixed,
max_inv_area, cfl_lambda_cap, momentum_cap_min_speed, momentum_cap_celerity_mult,
depth_cap, max_rel_depth_increase, shallow_damping_depth,
source_cfl_beta, source_max_substeps, source_rate_cap, source_depth_step_cap,
source_true_subcycling, source_imex_split,
gpu_diag_sync_interval_steps, tiny_mode, tiny_wet_cell_threshold,
degen_mode, front_flux_damping, open_bc_relaxation,
active_set_hysteresis, use_redistribution, inflow_progressive,
adaptive_cfl_dt, enable_cuda_graphs, swe2d_perf_mode,
gravity, culvert_face_flux_mode, bridge_cuda_coupling,
bridge_stacked_coupling_mode, culvert_solver_mode,
drainage_gpu_method, drainage_solver_backend_mode,
solver_backend_mode, coupling_loop_mode
```

Plus run-control fields:

```
duration_s, output_interval_s, rain_rate_mmhr
```

#### Group C — geometry arrays (passed to backend.build_mesh)
```
node_x, node_y, node_z,
cell_nodes, face_offsets, face_nodes,
bc_n0, bc_n1, bc_tp, bc_vl, bc_relax,
h0, hu0, hv0, n_mann_cell,
cell_areas (flat array, not callable)
```

**Not** in the JSON — these are derived from `mesh_name + mesh.gpkg_path` at replay time:
- `cell_centroids` (computed by `mesh_cell_centroids(...)`)
- `cell_min_bed` (computed by `backend._cell_zb` after `build_mesh`)

#### Group D — coupling config (typed dataclasses; not widget values)
These are **constructed** by factories, not widget-captured:

| Field | Source |
|-------|--------|
| `pipe_network_cfg` | `build_drainage_config_from_json(drainage_data, ncells)` (existing) |
| `hydraulic_structures_cfg` | `build_structures_config_from_json(structures_data, ncells)` (existing) |
| `bridge_stacked_plans` | `build_bridge_stacked_plans_for_runtime(...)` (existing) |
| `coupling_soa` | `pack_coupling_soa(...)` (existing) |

The **inputs to these factories** go in `data_sources`:
```
data_sources.drainage.nodes_layer
data_sources.drainage.links_layer
data_sources.drainage.inlets_layer
data_sources.drainage.node_inlets_layer
data_sources.drainage.gpkg
data_sources.structures  (dict, can be null)
```

#### Group E — sample-line + side hydrographs
```
sample_map_data (List[dict]),
side_hydrographs (dict),
edge_hydrographs (dict),
edge_group_overrides (dict),
edge_groups (dict),
inflow_progressive_enabled (bool),
rain_update_interval_s (float),
uniform_inflow_enabled (bool),
thiessen_forcing (typed dataclass or None)
```

For replay, `thiessen_forcing` is **rebuilt** from `data_sources.hyetograph` + `data_sources.rain_cn` rather than serialized (it can be large and is fully derivable).

#### Group F — output flags (from widget checkboxes)
```
save_mesh_results, save_line_results, save_coupling_results,
save_run_log, save_max_only
```

#### Group G — units (2 fields)
```
length_unit_name, length_scale_si_to_model, rain_mm_to_model_depth
```

`length_scale_si_to_model` and `rain_mm_to_model_depth` are **derived** at replay from `mesh.crs_wkt` (CLI has `_si_m_per_model_from_wkt` already; GUI has the same helper via `view._length_scale_si_to_model`).

---

## 4. The GPKG replay-row schema (new)

### 4.1 New table

```sql
CREATE TABLE IF NOT EXISTS swe2d_run_replays (
    run_id           TEXT PRIMARY KEY,
    mesh_name        TEXT,
    created_utc      TEXT NOT NULL,
    replay_json      TEXT NOT NULL,         -- full RunContext as JSON
    json_schema_ver  TEXT NOT NULL,         -- 'swe2d-replay/1'
    cli_replay_cmd   TEXT NOT NULL          -- exact command-line to replay this run
);

CREATE INDEX IF NOT EXISTS idx_replay_mesh ON swe2d_run_replays(mesh_name, created_utc DESC);
```

### 4.2 CLI replay surface

```bash
# Replay the most recent run for a mesh in a results GPKG
swe2d-cli replay --results-gpkg /path/to/results.gpkg --mesh domain_v3 [--run-id <id>]

# Replay from a standalone JSON file (exported by GUI or hand-edited)
swe2d-cli replay --replay-file /path/to/replay.json
```

Both commands resolve the same way: load replay JSON → build `RunContext` → call `SWE2DRunController.execute(ctx)`.

### 4.3 CLI command template

The replay table stores `cli_replay_cmd` so users can copy-paste:

```bash
swe2d-cli replay \
    --results-gpkg /path/to/results.gpkg \
    --run-id swe2d_20260710T123456-0500 \
    --mesh-gpkg /path/to/model.gpkg
```

(All paths absolute; mesh GPKG resolved from `replay_json.mesh.gpkg_path`.)

### 4.4 Migration

Old runs (pre-revision) have no replay row. CLI replay only works for runs started after this plan merges. Add `python -m swe2d.cli.replay_migrate <results_gpkg>` to backfill from `swe2d_simulation_configs` table (best-effort — only widget-state params are recoverable).

---

## 5. Revised architecture (replaces base plan §3)

### 5.1 Module layout

```
swe2d/runtime/
├── run_controller.py        (REWRITTEN — unified controller; CLI/GUI single entry)
├── run_context.py           (existing — add from_replay_json, to_replay_json)
├── backend.py               (unchanged)
├── backend_initializer.py   (unchanged — wrapped by RunController)
├── run_finalizer.py         (unchanged)
└── coupling.py              (unchanged)

swe2d/cli/
├── headless_runner.py       (SHRUNK — JSON parse + RunContext factory + return dict)
├── replay_cli.py            (NEW — `swe2d-cli replay` command; uses headless_runner internals)
└── replay_persistence.py    (NEW — load/save swe2d_run_replays table)

swe2d/services/
└── gpkg_persistence_service.py  (EXTENDED — add persist_run_replay, load_run_replay)

swe2d/workbench/
├── controllers/run_controller.py  (REWORKED — RunController class replaced by adapter)
└── workers/simulation_worker.py   (REWORKED — QThread wrapper around unified controller)
```

### 5.2 Revised data flow

```
                       ┌──────────────────────────┐
                       │  RunContext (immutable)  │
                       │  - widget-derived fields │
                       │  - coupling configs      │
                       │  - sample-line data      │
                       └────────────┬─────────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │     RunController    │
                         │                      │
                         │ .execute(ctx, sink): │
                         │   ├─ preflight       │
                         │   ├─ build_mesh      │  ← uses mesh_name + gpkg_path
                         │   ├─ init_backend    │  ← calls SWE2DBackendInitializer (GUI's path)
                         │   ├─ build_coupling  │
                         │   ├─ build_lines     │
                         │   ├─ step_loop       │
                         │   ├─ finalize        │
                         │   └─ persist_replay  │  ← only when ctx.persist_replay=True
                         └────────┬─────────────┘
                                  │
                ┌─────────────────┴─────────────────┐
                ▼                                   ▼
      ┌─────────────────────┐            ┌─────────────────────┐
      │ FileProgressSink    │            │ QtProgressSink      │
      │  (CLI)              │            │  (GUI)              │
      └─────────────────────┘            └─────────────────────┘
```

### 5.3 RunController is GUI-faithful

The controller is implemented to **do exactly what `SimulationWorker._build_and_initialize_backend` does today.** It calls `SWE2DBackendInitializer.build_and_initialize(...)` with the same kwargs in the same order. **No reordering, no "rationalization."** The CLI replays GUI behavior because the controller *is* GUI behavior.

### 5.4 SimulationWorker shrinks to a Qt-thread wrapper

```python
class SimulationWorker(QThread):
    def __init__(self, ctx: RunContext, parent=None):
        super().__init__(parent)
        self._ctx = ctx
        self._qt_sink = QtProgressSink(self)  # owns Qt signals

    def run(self):
        controller = SWE2DRunController(self._ctx, self._qt_sink)
        try:
            result = controller.execute()
            self.compute_finished.emit(result)
        except PreflightError as e:
            self.compute_failed.emit(str(e))
        except CancelledError:
            self.compute_cancelled.emit()
        except Exception as e:
            self.compute_failed.emit(f"{type(e).__name__}: {e}")
        finally:
            self._qt_sink.flush()
```

### 5.5 Headless runner shrinks to JSON parsing + RunContext factory

```python
def execute_run(mesh_gpkg, params, results_gpkg=None, ...):
    ctx = RunContext.from_replay_json(params)  # JSON factory (or legacy compat path)
    sink = FileProgressSink(status_file_path, status_interval_s)
    controller = SWE2DRunController(ctx, sink)
    result = controller.execute()
    return result.to_dict()
```

The `params` dict format used by today's CLI is **deprecated** but supported for one release via a compatibility shim that converts old CLI param keys to new RunContext fields.

---

## 6. The run-completion "persist replay" step (new)

Where in the controller's `_finalize()` phase:

```python
def _finalize(self, result: RunResult):
    self._backend.destroy()
    self._sink.on_finished(result)

    # ── Persist replay JSON to results GPKG ──────────────────────────
    if self._ctx.persist_replay and self._ctx.results_gpkg_path:
        try:
            replay_payload = self._ctx.to_replay_json()
            cli_cmd = build_cli_replay_command(self._ctx, replay_payload)
            persist_run_replay(
                gpkg_path=self._ctx.results_gpkg_path,
                run_id=self._ctx.run_id,
                mesh_name=self._ctx.mesh_name,
                replay_payload=replay_payload,
                cli_replay_cmd=cli_cmd,
            )
            self._sink.on_log(f"[REPLAY] Saved replay JSON for run {self._ctx.run_id}")
        except Exception as e:
            self._sink.on_log(f"[REPLAY] Failed to persist replay JSON: {e}")
```

**When `persist_replay` is True**: controller writes the replay row. Both GUI (always) and CLI (when results_gpkg is given) get this for free.

**GUI behavior change**: zero user-visible change. The replay row is written silently on every GUI run. (Existing users won't see anything new in the UI; the row is visible in GPKG Explorer.)

---

## 7. Test plan additions (replaces base plan §6 tests)

| Test | What | Acceptance |
|------|------|------------|
| `test_gui_kernel_inputs_unchanged.py` | Snapshot of all kwargs passed to `SWE2DBackend.initialize()` for a known GUI run; compared to pre-refactor capture | Bit-identical |
| `test_cli_replay_round_trip.py` | GUI run → capture JSON → CLI replay with same JSON → compare `h, hu, hv` to ULP | Bit-identical |
| `test_replay_json_schema.py` | Every GUI widget param has a JSON key; every JSON key has a corresponding `RunContext` field | Schema-consistent |
| `test_replay_persistence.py` | After a GUI run, `swe2d_run_replays` table has a row with the correct run_id | Row exists, JSON parses |
| `test_cli_replay_command.py` | `build_cli_replay_command()` produces a copy-paste-able bash command | Shell-parses |
| `test_legacy_params_compat.py` | Old CLI `params.json` shape still runs (one release) | Shims work, warns on deprecated keys |

### 7.1 The round-trip test (most important)

```python
def test_gui_to_cli_round_trip():
    """Run a known scenario via the GUI path, capture RunContext JSON,
    then run via CLI with the same JSON. Outputs must match to ULP.
    """
    # 1. Run via GUI factory
    gui_ctx = RunContext.from_view(synthetic_view, request=None)
    gui_ctx.persist_replay = False  # don't pollute GPKG
    sink1 = NullProgressSink()
    controller1 = SWE2DRunController(gui_ctx, sink1)
    gui_result = controller1.execute()

    # 2. Serialize + replay
    replay_json = gui_ctx.to_replay_json()
    cli_ctx = RunContext.from_replay_json(replay_json)
    cli_ctx.persist_replay = False
    sink2 = NullProgressSink()
    controller2 = SWE2DRunController(cli_ctx, sink2)
    cli_result = controller2.execute()

    # 3. Compare
    assert np.array_equal(gui_result.h, cli_result.h)
    assert np.array_equal(gui_result.hu, cli_result.hu)
    assert np.array_equal(gui_result.hv, cli_result.hv)
```

This test is the contract. **If it fails, the refactor is wrong.**

---

## 8. Revised acceptance criteria (replaces base plan §11)

- [ ] `RunContext.from_view` and `RunContext.from_replay_json` produce the same DTO for the same inputs.
- [ ] `RunContext.to_replay_json` round-trips: `from_replay_json(ctx.to_replay_json()) == ctx`.
- [ ] GUI regression test `test_gui_kernel_inputs_unchanged.py` passes (bit-identical kwargs).
- [ ] Round-trip test `test_cli_replay_round_trip.py` passes (bit-identical h, hu, hv).
- [ ] Every GUI run writes a `swe2d_run_replays` row.
- [ ] `swe2d-cli replay --results-gpkg ... --run-id ...` produces identical outputs to the original GUI run.
- [ ] `grep -r 'backend.initialize' swe2d/cli swe2d/workbench` returns only the unified controller (and `backend_initializer.py`).
- [ ] `headless_runner.execute_run()` body is `<100` lines.
- [ ] `SimulationWorker.run()` body is `<50` lines.
- [ ] Scheme migration warning + CFL clamping both live in the unified controller.
- [ ] CHANGELOG entry describes GUI behavior unchanged + replay JSON feature.

---

## 9. Open questions (revised)

1. **Should the replay JSON include BLOB pointers for `coupling_soa_blob` and `bridge_stacked_plans`?** — Recommendation: yes, as base64-encoded fields. Avoids re-running the planning logic at replay time.
2. **What happens to old CLI `params.json` files after this merges?** — Recommendation: one-release deprecation; CLI accepts both formats, logs a warning on legacy keys, errors out only on truly unknown keys.
3. **Should the GPKG Explorer dialog show replay rows?** — Recommendation: yes, as a sub-table view. Low priority; can ship in v2.
4. **Replay in CLI should respect the same `BACKWATER_ENABLE_CUDA_GRAPHS` env var the GUI sets?** — Recommendation: yes — controller reads env var at execute-time, same as GUI's worker.
5. **`swe2d_simulation_configs` (existing widget-state-save table) becomes redundant with `swe2d_run_replays`?** — Recommendation: keep both for one release; `swe2d_simulation_configs` is for manual user-saved configs (no run_id), `swe2d_run_replays` is for auto-saved per-run JSON. Plan to deprecate the former in a follow-up.

---

## 10. Summary of deltas vs base plan

| Section | Base plan | Revision |
|---------|-----------|----------|
| §1.3 (cost of duplication) | Lists divergence as a bug to fix | Lists divergence as **GUI behavior to preserve**; CLI adopts GUI |
| §2 (goals) | Symmetric merge | **Asymmetric**: GUI behavior preserved, CLI replays GUI |
| §3 (architecture) | Unified controller, both paths converge | Unified controller, but controller **is** GUI behavior |
| §9 (migration path) | Behavioral compat for both | **CLI behavior changes**, GUI behavior bit-identical |
| §11 (acceptance) | Symmetric regression tests | **Asymmetric**: GUI regression + round-trip test |
| **§4 (new)** | n/a | GPKG replay row schema |
| **§6 (new)** | n/a | Persist-replay step in `_finalize()` |
| **§7 (new)** | n/a | Round-trip test definition |
| File additions | 1 (run_controller.py) | 3 (+ `replay_cli.py`, `replay_persistence.py`) |
| File deletions | 0 | 0 |
| File reworks | 4 | 5 (+ `gpkg_persistence_service.py` extended) |

The CLI work expands, the GUI work contracts (no behavioral risk). Net effect: GUI runs become **reproducible** (a property they don't have today). CLI becomes a **verifier** for GUI runs — useful for CI, regression, and audit.