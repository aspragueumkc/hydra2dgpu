---
type: reference
status: complete
created: 2026-07-21
completed: 2026-07-25
---

# CLI/GUI Parity — Drift Prevention

## Phase 4 Status (ENFORCED)

The CLI-first refactor now enforces:
- **Canonical builder** (`swe2d.core.builder.build_run_context`): one builder and defaults table for CLI and GUI runs.
- **GUI adapter** (`swe2d.workbench.adapters.run_context_adapter.build_run_context_from_gui`): thin widget-to-spec translation only.
- **Shared executor** (`swe2d.core.executor.execute_run`): both entry paths run the same timestep loop.
- **Strict schema checks**: unknown top-level and nested keys fail with suggestions.
- **Typed load errors**: configured data sources and baked meshes cannot silently degrade to `None`.

```
CLI:  JSON/widget → canonical spec → build_run_context(spec) → RunContext
GUI:  widgets    → canonical spec → build_run_context_from_gui() → RunContext
                                    ↓
                              same RunContext
```

## Resolved drift

| Former issue | Enforced resolution |
|---|---|
| Divergent Thiessen rainfall builders | CLI and GUI route through the same QGIS-backed builder. |
| Drainage config drift and dropped inline JSON | One drainage builder accepts the schema-defined GPKG and inline forms. |
| Sample lines and edge groups discarded | Both values are wired into `RunContext` and covered by parity tests. |
| Core/CLI importing GUI workers | `swe2d.core` owns the executor; GUI visualization exports load lazily. |
| Silent data-source and mesh failures | `BuildRunContextError` and `MeshLoadError` preserve the failing spec key and cause. |

A configured value now has two outcomes: it reaches the shared `RunContext`, or
construction fails before solver startup. Silent `None`/empty fallback is not an
accepted state.

## Enforced mechanisms

### 1. RunContext diff test — enforced

`tests.test_run_context_parity` builds one `RunContext` from each entry path and recursively compares dict/array fields. Any unallowlisted difference fails the gate before kernel work proceeds.

### 2. Replay integration test — enforced

The GPU replay-equivalence gate runs the same GPKG and JSON through both paths and compares final `h` arrays within floating-point tolerance.

### 3. RunContext validation — enforced

`swe2d.core.executor.execute_run()` validates required arrays and callbacks before importing or starting the solver. Incomplete contexts raise immediately.

### 4. Builder schema and load validation — enforced

`swe2d.core.builder.build_run_context()` rejects unknown top-level and nested keys with suggestions. Configured GPKG sources raise typed errors for missing layers, loader failures, and corrupt mesh BLOBs.

## When adding a major feature (e.g. bridge solver kernel)

The data plumbing steps that need parallel implementation:

| Step | GUI path | CLI path | Validation |
|---|---|---|---|
| Widget/spec input | Dialog protocol method | JSON `params` / `data_sources` key | Nested-schema test |
| GPKG layer query | QGIS layer selection | `QgsVectorLayer(gpkg\|layername=..., "ogr")` | Typed missing-layer test |
| Build config from layer | Shared service function | Same shared service through `swe2d.core.gpkg_io` | RunContext diff test |
| Wire into RunContext | `build_run_context_from_gui()` | `build_run_context()` | RunContext diff test |
| Pass to C++ | `swe2d.core.executor.execute_run()` | Same shared executor | Replay test |

The **RunContext diff test** catches all data plumbing errors. The **replay test** catches physics drift. Both together are the complete gate.
