---
type: plan
status: complete
created: 2026-07-20
completed: 2026-07-25
---

# Pipe1D CLI/GUI Parity Fix — 2026-07-20

## Goal
Make the CLI's `build_run_context_from_dict` call the **same functions in the
same order** as the GUI's `RunController._build_run_context`. The user has
reminded 6+ times that the "CLI proves it runs" argument was bogus because
internal flow sources, drainage, and structure init were silently skipped.

The ONLY blocker for parity is the PyQGIS iface API (map canvas layer
lookup via `combo.currentLayer()`). Every wrapper layer can be made CLI-callable
by opening GeoPackage layers through `QgsVectorLayer(gpkg|layername=…, "ogr")`,
which already works in the CLI.

## GUI order of operations (the spec)

The GUI `_build_run_context` (`swe2d/workbench/controllers/run_controller.py:147-357`)
calls these in this exact order. The CLI must call the same APIs in the same
order with GPKG-loaded layer shims where the GUI uses map-canvas layers:

```
1. view._mesh_data (already loaded in GUI; CLI: query_mesh_from_gpkg)
2. view.collect_run_widget_params()             (CLI: read from JSON widget_state)
3. run_data_builder.build()  → RunInput        (CLI: build fake view, same builder)
   ├─ get_mesh_data_callback                    → mesh_data
   ├─ collect_boundary_arrays_callback          → default_bc_for_edges + apply_bc_layer_overrides_qgis
   ├─ build_side_hydrographs_callback           → {}
   ├─ collect_bc_layer_hydrographs_callback     → collect_bc_layer_hydrographs_qgis
   ├─ collect_bc_layer_edge_groups_callback     → collect_bc_layer_edge_groups_qgis
   ├─ initial_state_callback                    → mesh_runtime_logic.initial_state
   ├─ build_spatial_manning_array_callback      → build_spatial_manning_array_qgis
   └─ update_unit_system_callback               → update_unit_system_from_crs
4. run_options_builder.build(...) → RunOptions
   ├─ parse_run_duration_seconds                (already needed)
   ├─ TemporalScheme(reconstruction_mode)
   ├─ gpu_available check
   ├─ SolverModelOptions(temporal, spatial)     → model_options
   ├─ rain_rate_si_to_model(rain_rate_mmhr)     → rain_rate_model
   ├─ build_internal_flow_forcing               → internal_flow_forcing = build_internal_flow_forcing_qgis
   ├─ internal_flow_source_cms_at_time(forcing, 0.0)
   ├─ flow_si_to_model(cell_source_si)         → cell_source_model
   ├─ build_thiessen_rain_cn_forcing            → thiessen_forcing
   ├─ build_pipe_network_config                 → pipe_network_cfg
   ├─ build_hydraulic_structure_config          → hydraulic_structures_cfg
   └─ has_bridge_structures and gpu_available   → bridge_cuda_coupling
5. pack_coupling_soa(n_cells, pipe_network_cfg, hydraulic_structures_cfg)
6. build_bridge_stacked_plans_for_runtime(mesh_data, hydraulic_structures_cfg)
7. RunContext(...) construction with all assembled pieces
```

## Specific CLI gaps (current state)

1. **`internal_flow_forcing`** — `run_context_builder.py:407-422` tries to
   `from swe2d.cli.gpkg_adapter import build_internal_flow_forcing_from_gpkg`
   — that function does NOT exist.  ImportError silently caught → always None.
   Inside the try block, an undefined `mesh_gpkg_path` would also NameError
   if the import succeeded.  Logic is duplicated (lines 408-422 should be
   collapsed into one block).

2. **`cell_source_model`** — CLI passes `None`.  GUI calls
   `internal_flow_source_cms_at_time(forcing, 0.0)` then `flow_si_to_model`.

3. **`rain_rate_model`** — CLI hardcodes `0.0`.  GUI computes from widget.

4. **`bridge_cuda_coupling`** — CLI passes `False`.  GUI sets based on
   `_has_bridge_structures(hydraulic_structures_cfg) and gpu_available`.

5. **`bridge_stacked_coupling_mode`** — CLI reads from dict (works).

6. **`thiessen_forcing`** — currently populated only when `hyeto["table"]
   and hyeto["gauge_layer"]` are both set (line 388), and uses
   `build_forced_thiessen_from_gpkg` which does NOT match the GUI's
   `build_thiessen_rain_cn_forcing_qgis` (different signatures, different
   cell-to-gauge mapping).  Need to add a GPKG shim that calls into the
   QGIS-equivalent.

7. **`pipe_network_cfg`** — loaded only if `drainage["nodes_layer"]` set;
   falls back to exception swallowing with `exc: warning`.  Function
   works correctly when it runs, but conditional triggering is wrong.

8. **`hydraulic_structures_cfg`** — CLI calls `build_structures_config_from_json`
   on the `structures` dict (JSON-keyed), not the GUI's
   `build_hydraulic_structure_config_from_layer` (GPKG-loaded vector layer).

9. **`mesh_cell_centroids` callback** — CLI passes arrays, but
   `mesh_cell_centroids()` returns `cell_centroids` (GUI uses
   `view._mesh_cell_centroids()` for live updates — basically the same).

10. **`cell_centroids` field** — `np.empty((0, 2))` in CLI; GUI passes real array.

11. **`edge_groups` / `edge_group_overrides`** — `{}` in CLI; GUI populates
    from `_collect_bc_layer_edge_groups`.

12. **`sample_map_data`** — `[]` in CLI; GUI computes from `_build_line_sampling_map`.

13. **`internal_flow_source_cms_at_time`** callback — no-op lambda in CLI;
    GUI calls `internal_flow_source_cms_at_time` from `runtime_source_logic`.

14. **`apply_external_sources` / `distribute_total_flow_to_unit_q` /
    `apply_timeseries_bc_values` callbacks** — no-op lambdas in CLI.
    These are NOT actually invoked by the worker (it constructs its own
    `_WorkbenchShim` from pure logic).  Cosmetic only but should still
    be set correctly for protocol consistency.

## Plan

### Phase 1: GPKG shim wrappers (mirror `apply_bc_layer_overrides_from_gpkg`)

In `swe2d/cli/gpkg_adapter.py`, add:

```python
def build_internal_flow_forcing_from_gpkg(
    *, gpkg_path, table_name, mesh_data, length_unit_name="m",
    flow_si_to_model_factor=1.0, log_fn=None,
) -> Optional[Dict]:
    """Wrapper that loads SWE2D_Internal_Flow_Sources via QgsVectorLayer
    and delegates to build_internal_flow_forcing_qgis."""
    from qgis.core import QgsVectorLayer, QgsWkbTypes, QgsGeometry, QgsPointXY
    from swe2d.boundary_and_forcing.internal_flow_qgis_adapter import (
        build_internal_flow_forcing_qgis as _logic,
    )
    if not gpkg_path or not table_name:
        return None
    uri = f"{gpkg_path}|layername={table_name}"
    layer = QgsVectorLayer(uri, "internal_flow", "ogr")
    if not layer.isValid():
        return None
    fields = set(layer.fields().names())
    field_name = None
    for cand in ("q_cms", "flow_cms", "q", "flow"):
        if cand in fields:
            field_name = cand; break
    if field_name is None:
        return None
    # Load hydrograph_source_layer if present in GPKG
    hydro_layer_uri = f"{gpkg_path}|layername=SWE2D_Hydrographs"
    hydro_layer = QgsVectorLayer(hydro_layer_uri, "swe2d_hydrographs", "ogr")
    def _iter_project_layers_fn():
        return [layer] + ([hydro_layer] if hydro_layer.isValid() else [])
    from swe2d.mesh.mesh_runtime_logic import mesh_cell_centroids as _cctds
    cx, cy = _cctds(mesh_data)
    return _logic(
        mesh_data=mesh_data, have_qgis_core=True,
        internal_flow_layer_combo="__gpkg_layer__",
        combo_layer_fn=lambda combo, layer_type: layer,
        requested_field_name=field_name,
        iter_project_layers_fn=_iter_project_layers_fn,
        mesh_cell_centroids_fn=lambda: (cx, cy),
        parse_hydrograph_text_fn=_parse_hydrograph_text_factory(),
        hydrograph_from_layer_fn=lambda lyr, hydrograph_id="", bc_type=None:
            hydrograph_from_layer(lyr, hydrograph_id=hydrograph_id, bc_type=bc_type,
                                  parse_time_hours_fn=parse_time_hours,
                                  vector_layer_type=QgsVectorLayer),
        qgs_vector_layer_cls=QgsVectorLayer,
        qgs_wkb_types=QgsWkbTypes,
        qgs_geometry_cls=QgsGeometry,
        qgs_pointxy_cls=QgsPointXY,
        log_fn=log_fn,
    )
```

Similar for:
- `build_thiessen_rain_cn_forcing_from_gpkg` (delegate to
  `build_thiessen_rain_cn_forcing_qgis`)
- `build_pipe_network_config_from_gpkg` (delegate to
  `build_pipe_network_config` after opening nodes/links/inlets layers)
- `build_hydraulic_structure_config_from_gpkg` (delegate to
  `build_hydraulic_structure_config_from_layer`)
- `build_initial_state_from_widgets` (delegate to
  `mesh_runtime_logic.initial_state`)

### Phase 2: Refactor `build_run_context_from_dict`

Reorder to call the SAME functions in the SAME order as the GUI.

### Phase 3: Wire callbacks

Replace no-op lambdas with real callbacks:
- `internal_flow_source_cms_at_time = lambda forcing, t:
    internal_flow_source_cms_at_time(forcing, t, _interp_hydrograph)`
- `mesh_cell_centroids = _mesh_cell_centroids` (already a callback)
- `apply_external_sources = ...` (real closure matching GUI's _WorkbenchShim)

### Phase 4: Update `pipe_1d_test.json`

Add `internal_flow_sources` config entry pointing to
`SWE2D_Internal_Flow_Sources` + `SWE2D_Hydrographs` in
`example_test_project.gpkg`.

### Phase 5: CLI test

Run the headless executor with updated JSON.  Two outcomes:

- **(A)** CLI now reproduces the GUI crash on the first FVM step →
  root cause is shared (kernel state, face classification, cuda graph
  capture).  Resume existing pipe1d investigation (deferred C++ changes:
  private stream-ordered memory pool, host face cache, sync debug).

- **(B)** CLI still runs cleanly → there's still another GUI-only init
  step missing (e.g., qgis.core initialisation side effects, geometry
  topology permutation, native BC forcing setup, CUDA graphs capture
  ordering).  Iterate.

### Phase 6: Cleanup deferred pipe1d C++ work

If (A), pick up the deferred changes:
- `Pipe1DDeviceState.d_stream` (cudaStream_t) for private memory pool
- `cudaMallocAsync`/`cudaMemsetAsync` on that stream
- Host face-class cache vectors with corruption detection
- Stream sync at end of mesh build
- `cudaGetLastError()` clearing after every kernel

Also remove temporary debug `fprintf` calls.
