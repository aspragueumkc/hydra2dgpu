# SWE2D Structural Placement Audit

**Date:** 2026-07-04
**Scope:** All 156 Python files / ~55k LOC under `swe2d/`
**Method:** 8 parallel agents, one per subdirectory cluster, each applying the
two rules below.

## Rules applied

### Rule 1 — MVP Architecture Violation
From `AGENTS.md`:

- **Service layer** (`swe2d/services/`, `swe2d/runtime/`, `swe2d/boundary_and_forcing/`,
  `swe2d/mesh/`, `swe2d/results/`, `swe2d/extensions/`) **MUST NOT import PyQt5,
  touch widgets, or reference any UI element.**
- **GUI-only services** (`swe2d/workbench/services/`) **MAY import `qgis.core`
  but MUST NOT import `PyQt5.QtWidgets`.**
- **CLI** (`swe2d/cli/`) **MUST NOT import from `swe2d/workbench/`.**
- **Controllers** (`swe2d/workbench/controllers/`) **MUST NOT reach through the
  View to access Qt widgets directly.** They use View protocol methods.

### Rule 2 — Logical Scope Violation
A function's job must match its file's name. The filename declares the scope;
functions whose job a reasonable engineer would not expect from the filename
are misplaced.

User's canonical example: a `results_export_service.py` containing a function
that pulls a Manning's n value from a widget and feeds it to the
runtime_orchestrator would be illogical — outside the file's intended scope.

---

## TIER 1 — Textbook "illogical placement"

### 1. `workbench/studio_dialog.py` — mesh math + solver serialization in a dialog

- **`_save_mesh_to_gpkg`** `studio_dialog.py:669` — full mesh
  build→serialize→persist pipeline (~55 LOC) inlined on the dialog.
  → `services/gpkg_persistence_service.py::save_mesh_to_gpkg`.
- **`_load_mesh_from_gpkg`** `studio_dialog.py:726` — opens `sqlite3.connect()`,
  deserializes C-extension mesh, builds the `mesh_data` dict. 80 LOC.
  → `services/gpkg_persistence_service.py::load_mesh_from_gpkg`.
- **`_sample_line_metrics`** `studio_dialog.py:2011` — polygon mesh
  triangulation via centroid-vertex fan (~110 LOC).
  → `workbench/services/mesh_service.py::triangulate_cell_nodes`.
- **`_mesh_cell_polygons`** `studio_dialog.py:2210` — builds
  `QgsGeometry.fromPolygonXY(...)` rings inline; sibling methods correctly
  delegate to `_mesh_svc.*`. Should delegate.
- **~15 run-prep methods** `studio_dialog.py:1651-2288` —
  `_collect_boundary_arrays`, `_build_spatial_manning_array`,
  `_build_internal_flow_forcing`, `_build_pipe_network_config`,
  `_build_hydraulic_structure_config`, `_apply_timeseries_bc_values`,
  `_distribute_total_flow_to_unit_q`, `_apply_external_sources`,
  `_apply_bc_layer_overrides`, `_build_spatial_cn_array`, `_initial_state`,
  `_collect_bc_layer_hydrographs`, `_collect_bc_layer_edge_groups`,
  `_build_thiessen_rain_cn_forcing`, `_internal_flow_source_cms_at_time`.
  Each pulls widget values + mesh data and calls into
  `boundary_and_forcing`/`extensions`/services to build solver inputs.
  **Inverted MVP** — controllers call `view._build_side_hydrographs()` etc.
  instead of reading typed getters and calling services themselves.
  → `workbench/controllers/run_controller.py` (or a new `run_data_collector.py`).
- **15 pass-through wrappers** `studio_dialog.py:1623-2304` —
  `_mesh_boundary_edges`, `_mesh_cell_centroids`, `_mesh_cell_areas`,
  `_mesh_cell_min_bed`, `_mesh_cell_solver_bed`, `_boundary_buffer_cells`,
  `_length_scale_si_to_model`, `_rain_mm_to_model_depth`, `_rain_rate_si_to_model`,
  `_flow_si_to_model`, `_interp_hydrograph`, `_parse_hydrograph_text`,
  `_hydrograph_from_layer`, `_detect_map_unit`, `_is_us_customary_units`.
  One-line forwarders to `_mesh_svc` / `_unit_svc`. Bloat dialog surface area
  to ~120 methods. → Delete; have callers hold service refs directly.
- **4 controller-private pass-throughs** `studio_dialog.py:2128-2142` —
  `_preflight_validate_mesh`, `_collect_bc_for_edges`, `_prepare_run_inputs`,
  `_collect_simulation_settings`. Each is `return self._controller._<x>()` —
  reaches into controller privates from the view. (Also dead, see below.)
- **`_build_topology_meshing_options`** `studio_dialog.py:461` — 90-line
  widget→dict translator. → `controllers/topology_controller.py`.
- **`_current_line_results_storage_path`** `studio_dialog.py:1308` — 50-line
  3-tier path resolution reaching into a layer combo's data provider.
  → new `workbench/services/results_path_service.py`.

### 2. `workbench/dialogs/batch_simulation_dialog.py` — runtime in a dialog (1022 LOC)

- **`_run_batch` / `_poll_tick` / `_start_next_batch` / `_check_batch_status` /
  `_tick_run` / `_cancel_batch`** `batch_simulation_dialog.py:869-1016` —
  subprocess pool, status-file polling, ~155 LOC.
  → `cli/batch_runner.py` (or new `runtime/batch_runner_service.py`).
- **`_widget_params_to_run_params`** `batch_simulation_dialog.py:83` + 3 mapping
  tables at `:36-80` + `_parse_run_duration_hours` `:165` — widget→CLI-key
  translation. Must stay in sync with CLI parser.
  → `cli/run_params.py` or `runtime/run_data_builder.py`.
- **`_query_runs_from_gpkg` / `_refresh_mesh_list`**
  `batch_simulation_dialog.py:700,784` (plus inline sqlite in
  `_snapshot_current_setup` `:519-528`) — re-implement raw sqlite queries that
  `results/run_service.py` and `results/db_utils.py` already provide.
  → `results/run_service.py` (add `collect_mesh_names_from_gpkg`).
- **`_snapshot_current_setup`** `batch_simulation_dialog.py:487` — 165 LOC
  walking `parent._model_tab_view` / `parent._map_tab_view` by hardcoded
  attribute names. Bypasses the parent's `collect_run_widget_params()` API and
  the `WorkbenchView`/`ModelTabViewProtocol` protocols.
  → Parent controller method `snapshot_run_setup() -> dict`.

### 3. `workbench/services/mesh_service.py` — line sampling squatting (390 LOC)

- `build_line_sampling_map` `:203` — cell-intersection weights, normals,
  stations, IDW profile weights.
- `sample_line_metrics` `:376` — per-step profile metrics sampler.
- `sample_line_aggregate_ts_row` `:514` — aggregate timeseries row builder.
- 5 line-geometry helpers `_cumulative_length`, `_interpolate_along_line`,
  `_line_normal`, `_project_point_onto_line`, `_cell_centroids` `:118-200`.

A **second** `build_line_sampling_map` already exists at
`line_sampling_service.py:110`. Confusing duplication.
→ Consolidate in `workbench/services/line_sampling_service.py`.

### 4. `runtime/native_bc_forcing.py` — pure BC preprocessing in runtime

- `SWE2DNativeBoundaryHydrographConfigurator.configure`
  `native_bc_forcing.py:15` — ~200 LOC of side-detection, hydrograph
  grouping, progressive-BC elevation sorting, BC-code conversion (102→2,
  103→3). Repo already ships `boundary_and_forcing/boundary_runtime_logic.py`
  and `hydrograph_logic.py` for exactly this.
  → `boundary_and_forcing/native_bc_forcing.py` (or merge into existing
  `boundary_runtime_logic.py`/`hydrograph_logic.py`, leaving only the thin
  `backend.set_boundary_hydrographs_native(...)` upload call in runtime).

### 5. `cli/gpkg_adapter.py` — kitchen sink (1067 LOC; ~400 LOC misplaced/dead)

- **`build_drainage_config_from_json`** `:560` — JSON→`PipeNetworkConfig`
  (no sqlite3, no GPKG). → `extensions/drainage_network.py`.
- **`build_structures_config_from_json`** `:697` — JSON→`HydraulicStructureConfig`
  (no GPKG). → `extensions/structures.py`.
- **`_compute_cell_centroids`** `:882` — duplicates
  `mesh/mesh_runtime_logic.py:10` and `mesh_computation_service.py:46`.
  → Delete; import canonical.
- **`apply_bc_overrides_from_gpkg`** `:308` — 145 LOC, 0 callers (see DEAD CODE).
- **`_parse_linestring_coords`** `:455` — only reachable from the dead function
  above; near-duplicate of `_parse_wkt_linestring_coords` `:293`.

### 6. `controllers/run_controller.py` — controller implementing solver math

The `_execute_run` method spans `:62-1101` (~1040 LOC) and inlines:

- **RCMK mesh permutation** `:577-608` — reorders `cell_nodes`,
  `cell_face_offsets`, `cell_face_nodes` via `backend._cell_perm`.
  → `services/mesh_service.py::apply_cell_permutation`.
- **Coupling/drainage/structure assembly + unit conversion** `:268-306` —
  builds `SWE2DUrbanDrainageModule`, `SWE2DStructureModule`,
  `SWE2DCouplingController`, `_build_redistribution_data`.
  → `runtime/coupling.py` (or new `runtime/coupling_setup_service.py`).
- **Native BC + rain forcing configuration** `:660-748` — drives
  `SWE2DNativeBoundaryHydrographConfigurator.configure(...)` and
  `SWE2DRunSetupConfigurator.configure_native_rain_cn_forcing(...)`.
  → `runtime/runtime_sources.py`.
- **Edge-length / mesh-bounds / side classification** `:763-786`.
  → `services/mesh_service.py`.
- **`on_preview_overrides`** `:1299-1349` — BC override differential analysis
  inline. → `services/mesh_computation_service.py`.
- **`on_preview_coupling`** `:1571-1698` — drainage sanity validation (~130 LOC).
  → `runtime/coupling.py::validate_coupling_configs` /
  `coupling_sanity_report`.
- **`on_load_simulation_config`** `:1413-1449` — deserializes baked mesh inline
  via `swe2d_deserialize_mesh`, builds full `mesh_data` dict.
  → `controllers/mesh_controller.py`.

### 7. `controllers/mesh_controller.py:659` — run-log viewer in mesh controller

- `open_run_log_viewer` picks a GPKG, loads `swe2d.results.run_log_storage`,
  prompts for a run, opens `SWE2DRunLogViewerDialog`. Nothing mesh about it.
  → `controllers/run_controller.py` or new `controllers/results_controller.py`.

### 8. `views/topology_tab_view.py:1595` — module-level "Controller:" function

- `_wire_topology_tab_controls` (free function, ~120 LOC). Docstring literally
  reads `"""Controller: connect topology tab widget signals."""`. Inner `_w`
  helper references `self._log(...)` from module scope → NameError on error
  path, swallowed silently. **Silent fallback (AGENTS.md worst-case).**
  → `controllers/topology_controller.py`; use `signal_helpers.safe_connect`.
- `_build_topology_tab_controls` `:800-1594` — 800-line module-level widget
  factory (LOW confidence, borderline). ~920 of 1718 file LOC are outside the
  `TopologyTabView` class. Consider moving into class methods or splitting
  `topology_tab_widgets.py`.

### 9. `extensions/extension_models.py:525` — engine classes in "models" file

- `DrainageCouplingEngine`, `RainfallSourceEngine`, `HydraulicStructureEngine`
  hold state + lifecycle methods (`initialize`, `exchange_step`,
  `sample_cell_rain`). File docstring says "data-model skeletons only".
  Concrete subclasses already exist:
  - `SWE2DUrbanDrainageModule(DrainageCouplingEngine)` in `drainage_network.py`
  - `SWE2DStructureModule(HydraulicStructureEngine)` in `structures.py`
  → Move each base next to its concrete subclass; `RainfallSourceEngine` →
  new `extensions/rainfall.py`.

### 10. Smaller logical-scope outliers

- **`dialogs/coupling_results_dialog.py:18`** `prepare_coupling_timeseries` —
  pure data-shaping helper (groups by `object_id`, filters non-finite, sorts,
  converts to hours). No widget access. → `runtime/coupling.py` or
  `results/coupling_results_service.py`.
- **`dialogs/gpkg_explorer_dialog.py:240`** `_delete_by_run_id` — opens
  `sqlite3.connect()` directly while the rest of the file uses
  `gpkg_operations_service`. Also `_list_run_ids_from_table_names` `:25` parses
  run IDs by `rsplit("_", 1)`. → `gpkg_operations_service.py::list_run_ids`.
- **`controllers/overlay_controller.py:420-597`**
  `export_high_perf_overlay_to_geotiff` — 180 LOC of GDAL GeoTIFF writing
  inlined (`GetDriverByName("GTiff")`, `Create`, `SetGeoTransform`,
  `SetProjection`, `WriteArray`, `FlushCache`). Rendering is correctly
  delegated; the write path is not. → new `services/geotiff_export_service.py`
  or extend `results/high_perf_viewer.py`.
- **`services/gpkg_persistence_service.py:92`**
  `current_line_results_storage_path(dialog)` — resolves a results GPKG path
  by reading `dialog._model_tab_view.results_gpkg_path_edit.text()` and
  walking layer combos. The dialog already has its own
  `_current_line_results_storage_path` at `studio_dialog.py:1308`; this
  service-layer copy has exactly 1 caller (`studio_results_panel.py:336`).
  → Delete; route the 1 caller through the dialog method.
- **`services/gpkg_persistence_service.py:54`** `collect_run_log_metadata` —
  free function, 0 callers (see DEAD CODE).
- **`meshing.py:1559, 2588, 2650`** `_gmsh_interface_coincidence_report`,
  `_gmsh_flow_aligned_curve_counts`, `_gmsh_flow_align_region_preflight` —
  gmsh-specific helpers consumed solely by `gmsh_backend.py` (LOW confidence:
  deliberate circular-import workaround).

---

## TIER 2 — MVP widget-access violations (controllers reaching through views)

`OverlayView` already demonstrates the correct protocol-method pattern. The
others bypass it.

- **`controllers/overlay_controller.py:323-335`**
  `refresh_high_perf_canvas_overlay` — `blockSignals(True); cs_min.setValue(...);
  blockSignals(False)` on `QDoubleSpinBox` objects reached via
  `view._results_toolbox`. Most blatant widget-touch.
  → `OverlayView.set_overlay_color_range(vmin, vmax)`.
- **`controllers/run_controller.py`** direct `QtWidgets.QMessageBox.critical/
  information/warning`, `QFileDialog.getOpenFileName/getSaveFileName`,
  `QInputDialog.getText` at lines `1098, 1282, 1294, 1353, 1374, 1478, 1492,
  1521, 1541, 1553, 1700`.
  → Add protocol methods accessors to `RunView`.
- **`controllers/run_controller.py:1216-1220`** — reaches
  `view._model_tab_view.results_gpkg_path_edit.text()`.
  → `RunView.get_results_gpkg_path()`.
- **`controllers/run_controller.py:1441-1442`**,
  **`controllers/topology_controller.py:273-274, 619-621`**,
  **`controllers/mesh_controller.py:122-123`** —
  `_studio_viewer.tab_widget.setCurrentWidget(_viewer.plot_widgets.get("Mesh"))`
  (×5). → `RunView.show_mesh_tab()` / `MeshView.show_mesh_tab()` /
  `TopologyMeshView.show_mesh_tab()`.
- **`controllers/topology_controller.py:499`** — reads
  `topo_status_lbl.text()` off a `QLabel`. → `TopologyMeshView.get_topo_status()`.
- **`controllers/topology_controller.py:736, 794`**,
  **`controllers/mesh_controller.py`** at 15 sites (`214, 278, 314, 331, 337,
  394, 399, 430, 533, 585, 622, 631, 665, 686, 695`) — same direct
  `QFileDialog`/`QMessageBox`/`QInputDialog` pattern. → `MeshView` /
  `TopologyMeshView` protocol accessors.
- **`controllers/topology_controller.py:985-1002`** `_ensure_timer` —
  controller constructs `QtCore.QTimer(view)` on the view. Timer should be
  view-owned; controller subscribes via `on_tick`.
- **`workbench/services/non_gui_runtime_service.py:421`**
  `execute_run_timestep_loop` (via `_make_uniform_velocity_cb`) —
  `if not hasattr(wb._model_tab_view, "uniform_inflow_velocity_chk"): return None`
  directly probes a `QCheckBox`. Both MVP violation AND self-contradicting
  filename. → Caller resolves the flag and passes it as `bool`.

---

## TIER 3 — Bright-line CLI MVP violation

- **`cli/headless_runner.py:441, 622`** — late imports of
  `swe2d.workbench.services.mesh_service` (`build_line_sampling_map` /
  `sample_line_metrics` / `sample_line_aggregate_ts_row`) inside `execute_run`.
  AGENTS.md: "CLI MUST NOT import from `swe2d/workbench/`."
  **Root cause is structural**: those three functions are pure numpy (the file's
  own docstring says "Pure-Python, Qt-free service"). They live under
  `workbench/` for historical reasons only. Moving them to `swe2d/services/`
  (per Tier 1 #3) fixes this violation automatically — no CLI logic change
  needed.

---

## DEAD CODE (per AGENTS.md — deletion pending user confirmation)

| Symbol | Location | ~LOC | Notes |
|---|---|---|---|
| `collect_run_log_metadata` | `services/gpkg_persistence_service.py:54` | 30 | free function, 0 callers; same-named methods elsewhere are unrelated |
| `RainfallSourceEngine` | `extensions/extension_models.py:571` | 15 | stub returning `[default_mm_per_hr/1000/3600] * n_cells`; session log confirms "never implemented" |
| `DrainageCouplingEngine.exchange_step` | `extensions/extension_models.py:565` | 15 | skeleton returning `([], [])`; session log confirms "disabled before GPU coupling was complete" |
| 5 hydraulic formula fns: `compute_orifice_flow`, `compute_weir_flow`, `compute_pipe_manning_capacity_full`, `circular_section_from_depth`, `convert_cell_flows_to_depth_rates` | `extensions/extension_models.py:390-484` | 100 | exported in `__all__` but 0 callers in `swe2d/`; C++ kernels handle on-device |
| `source_rate_callback` | `runtime/coupling.py:1523` | small | deprecated `raise RuntimeError` stub |
| `_gmsh_available` (duplicate) | `mesh/meshing.py:334` | 15 | shadowed by canonical `gmsh_backend.py:38`; local def never used |
| `runoff_depth_mm_from_event_rain_mm`, `composite_curve_number`, `time_of_concentration_hours_velocity_method` | `boundary_and_forcing/rainfall_hydrology.py:605, 631, 658` | 50 | 0 callers |
| `_noop` | `controllers/run_controller.py:60` | small | 0 refs |
| `_preflight_validate_mesh`, `_collect_bc_for_edges`, `_prepare_run_inputs`, `_collect_simulation_settings` | `controllers/run_controller.py:1105-1180` + 4 wrappers in `studio_dialog.py:2128-2142` | 80 | entire cluster unreferenced |
| `_opt_float`, `_opt_bool` | `controllers/topology_controller.py:20-39` | small | 0 callers |
| 14 underscore-aliases (`_BC_INFLOW_Q`, `_BC_TS_FLOW`, `_BC_VALUE_MAP`, etc.) | `workbench/services/constants_service.py:123-137` | 15 | comment says "remove once all call-sites updated" — they are |
| `apply_bc_overrides_from_gpkg` + `_parse_linestring_coords` | `cli/gpkg_adapter.py:308, 455` | 150 | 0 callers; headless_runner uses inline path |
| `iter_with_parents` | `workbench/devtools/widget_walker.py:165` | small | 0 callers; body just yields nodes, no parent logic despite name |
| Unreachable log line | `studio_dialog.py:1881` | 1 | after `return` on line 1857 |

---

## BUGS surfaced by the audit (silent-fallback violations per AGENTS.md)

The "no silent fallback" rule in `AGENTS.md` is the worst failure mode short of
data loss. These all silently swallow real errors:

- **`workbench/services/widget_persistence_service.py:128`** —
  `self._log(f"[ERROR] Exception in widget_persistence_service.py: {_e}")`
  inside a module-level function. `self` is undefined → `NameError` swallowed
  by inner `try/except: pass`. `log_fn` is the in-scope parameter.
- **`workbench/services/unit_conversion_service.py:128`** — identical
  `self._log(...)` NameError-swallowed pattern in
  `update_unit_system_from_crs`. CRS read failures currently vanish.
- **`workbench/views/topology_tab_view.py:1595`** — inner `_w` helper in
  module-level `_wire_topology_tab_controls` references `self._log(...)` →
  NameError on the error path.
- **`workbench/dialogs/widget_inspector.py:70, 87`** — `logger.warning(...)` 
  where `logger` is never imported.
- **`workbench/signal_helpers.py:88`** — `connect_lambda._handler` references
  undefined `obj`.
- **`controllers/mesh_controller.py:507` + `controllers/topology_controller.py:106`**
  — call `self.refresh_layer_combos()` but the method lives on `LayerController`.
  Currently swallowed by `try/except` (combo lists silently don't refresh).

---

## Other non-placement notes

- **`bc_logic.py:40`** `_bc_side_classification` declares a return type of
  `Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]` (4 elements) but
  actually returns 9 values. Sole caller `studio_dialog.py:1966` unpacks with
  `side_idx, edge_len, edge_z, *_ = ...` so it runs, but the annotation is
  wrong.
- The side-classification algorithm is duplicated three times across
  `bc_logic.py` and once in `boundary_runtime_logic.py` — DRY concern, outside
  this structural audit.

---

## CLEAN (no findings)

These files were reviewed and have no placement, MVP, or scope violations:

**`swe2d/services/`**: `lumped_hydrology_service.py`, `mesh_computation_service.py`,
`mesh_export_service.py`, `mesh_extraction_service.py`, `mesh_render_service.py`,
`qgis_terrain_interpolator.py`, `results_render_service.py`,
`terrain_assignment_service.py`, `ugrid_export_service.py`.

**`swe2d/plotting/`**: `__init__.py`, `viewer_plots.py` (borderline pointless
18-line facade, but not a placement violation).

**`swe2d/runtime/`**: `__init__.py`, `backend.py`, `backend_initializer.py`,
`bridge_stacked_runtime.py`, `coupling.py` (other than dead
`source_rate_callback`), `native_binding_compat.py`, `run_controller.py`,
`run_data_builder.py`, `run_finalizer.py`, `run_lifecycle.py`,
`run_options_builder.py`, `run_orchestrator.py`, `runtime_reporting.py`,
`runtime_setup_configurator.py`, `runtime_sources.py`,
`runtime_step_executor.py`.

**`swe2d/extensions/`**: `__init__.py`, `drainage_network.py`, `structures.py`.

**`swe2d/results/`**: `__init__.py`, `animation.py`, `data.py`, `db_utils.py`,
`export_service.py`, `high_perf_viewer.py`, `profile_service.py`, `queries.py`,
`run_log_storage.py`, `run_service.py`, `structure_service.py`,
`timestep_service.py`. (`manning_n`/`gravity` in `high_perf_viewer.py:445,599`
are legitimate post-processing shear-stress viz inputs — not pulled from
widgets, not sent to a runtime.)

**`swe2d/mesh/`**: `__init__.py`, `bridge_stacked_mesh.py`, `gmsh_backend.py`,
`mesh_models.py`, `mesh_quality.py`, `mesh_runtime_logic.py`, `meshing.py`
(large but coherent general-purpose mesh-geometry toolkit).

**`swe2d/boundary_and_forcing/`**: `bc_logic.py`, `boundary_runtime_logic.py`,
`internal_flow_logic.py`, `internal_flow_qgis_adapter.py`,
`internal_flow_qgis_geometry.py`, `boundary_qgis_adapter.py`,
`spatial_forcing_qgis_adapter.py`, `runtime_source_logic.py`,
`rainfall_hydrology.py` (no placement issues; 3 dead functions listed
separately). One mild outlier: `hydrograph_logic.py:91`
`hydrograph_from_layer` performs direct `QgsVectorLayer` schema introspection
in a `_logic.py` file (MEDIUM confidence; sibling `internal_flow_logic.py`
injects layer ops as callables instead — pattern inconsistency rather than
outright wrong file).

**`swe2d/workbench/controllers/`**: `layer_controller.py`,
`protocols_controller.py`, `startup_bootstrap_controller.py`,
`run_component_wiring_controller.py`, `finalization_adapter.py`.

**`swe2d/workbench/bridges/`**: `project_settings_bridge.py` (bridges touch
widgets by design; this one stays in lane with `qtwidgets_module` injected).

**`swe2d/workbench/services/`**: `pipe_network_service.py`,
`pipe_network_config_service.py`, `line_sampling_service.py` (the correct home
for the misplaced mesh_service trio), `mesh_data_prep_service.py`,
`hecras_export_service.py`, `text_parser_service.py`,
`gpkg_operations_service.py`, `gpkg_layer_styles_service.py`,
`non_gui_qgis_service.py`, `overlay_parameters_service.py`,
`schema_definitions.py`, `topology_template_service.py`,
`structure_config_service.py`, `model_gpkg_loader_service.py`, `run_service.py`,
`widget_persistence_service.py` (scope correct; bug listed separately),
`unit_conversion_service.py` (scope correct; bug listed separately).

**`swe2d/workbench/forms/`**: `swe2d_structures_form.py`.

**`swe2d/workbench/views/`**: `view_protocols.py`, `model_tab_view.py`,
`map_tab_view.py`, `run_dock.py`, `temporal_dock.py`, `studio_viewer.py`,
`studio_component_view.py`, `studio_tab_builder.py`, `studio_host_methods.py`,
`workbench_main_menu.py`, `studio_viewer_plot.py`, `results_controls.py`,
`widget_filter_helper.py`, `studio_results_panel.py`, `doc_viewer.py`,
`studio_viewer_pg.py`, `studio_viewer_profile_pg.py`.

**`swe2d/workbench/dialogs/`**: `run_log_viewer_dialog.py`,
`simulation_config_dialog.py`, `sqlite_preview_dialog.py`,
`topo_attr_table_dialog.py`, `hydrograph_editor.py`, `run_selection_dialog.py`,
`detached_mesh_dialog.py`, `detached_panel_dialog.py`, `detached_log_dialog.py`,
`workbench_settings_dialog.py`, `_plot_utils.py`, `widget_inspector.py`
(scope correct; bug listed separately).

**`swe2d/workbench/`** top-level: `__init__.py`, `map_tools.py`, `post_init.py`,
`startup_state.py`, `signal_helpers.py` (scope correct; bug listed separately),
`workbench_view_state.py`, `workbench_api.py`, `workbench_dialog_builder.py`.

**`swe2d/cli/`**: `__init__.py`, `__main__.py`, `batch_runner.py`.

**`swe2d/workbench/devtools/`**: `__init__.py`, `inspector_dock.py`,
`ast_patterns.py`, `validation.py`, `property_editor.py`, `patch_builder.py`,
`menu.py`.

**Top-level**: `swe2d/__init__.py`, `swe2d/units.py`.

---

## Recommended execution order (highest leverage / lowest risk first)

1. **Move line-sampling trio** `mesh_service.py:118-514` →
   `line_sampling_service.py`. Fixes Tier 1 #3 AND Tier 3 (CLI import
   violation) in one move.
2. **Move mesh serialize/load** `studio_dialog.py:669, 726` →
   `gpkg_persistence_service.py`. The user's textbook example.
3. **Move batch subprocess pool** `batch_simulation_dialog.py:869-1016` →
   `cli/batch_runner.py`.
4. **Move BC preprocessing** `runtime/native_bc_forcing.py` →
   `boundary_and_forcing/`.
5. **Delete confirmed dead code** — clears ~500 LOC across the repo.
6. **Fix the `self._log` silent-fallback bugs** — quick, prevents real failures
   from vanishing.

The widget-access MVP violations (Tier 2) are systematic — fixing them means
adding protocol methods to `RunView`/`MeshView`/`TopologyMeshView` and
rewiring ~30 call sites. Worth doing but bigger blast radius.
