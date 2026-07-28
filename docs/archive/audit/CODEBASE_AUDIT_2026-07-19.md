---
type: audit
status: complete
created: 2026-07-19
completed: 2026-07-25
---

# Codebase Audit — 2026-07-19 (Pre-Release)

**Scope:** `swe2d/` (189 py, ~69k LOC), `tests/`, `tools/`, `hydra_plugin.py`, `CMakeLists.txt`, `Makefile`, `pyproject.toml`, `docs/`, `README.md`, `CHANGELOG.md`, `metadata.txt`.
**Excluded** (per `.gitignore` / `.publicsync-ignore`): `build*/`, `__pycache__/`, `graphify-out/`, `.opencode/`, `.agents/`, `.vscode/`, `docs/superpowers/`, `reference/` (except md files used as context), `marketing/`, `tests/results/*.log`, `*.db`.
**Method:** 5 parallel audit agents (duplicates/dead code, MVP architecture, potential bugs, documentation, quick improvements). Every finding verified by reading code; all file:line references spot-checked. Research only — no code modified.
**Release context:** A new release is being prepared; `reference/qgis-repo-and-qgis4-migration.md` will likely be implemented on a new branch of the public repo. Public-synced docs must not reference private-only paths.

---

## 0. Executive Summary — Pre-Release Priorities

### P0 — Likely real bugs (fix before release)

| # | Finding | Evidence |
|---|---|---|
| 1 | **CLI replay rainfall 1000× too deep (SI) and inverted length scale (USC)** — defaults use `_u.si_m_per_model()` where the RunContext convention is `model_per_si_m()` / `1e-3·model_per_si_m`; replay payload always ships an empty units block so these defaults always fire | `swe2d/runtime/run_context_builder.py:622-625`; convention at `run_controller.py:343-344`; `"units": {}` at `run_controller.py:888` |
| 2 | **Internal flow sources never converted SI→model** — identity `flow_si_to_model_callback` leaves m³/s interpreted as ft³/s in USC models (35.3× undercount); `ctx.flow_si_to_model` exists but unwired | `swe2d/workbench/workers/simulation_worker.py:731`; also line 729 reads global `_u.rain_si_to_model(1.0)` instead of `ctx.rain_rate_si_to_model` |
| 3 | **`NameError: mesh_gpkg_path`** in the internal-flow-sources CLI branch — crashes any CLI run with `internal_flow_sources.table` set; exception type not in the catch list | `swe2d/runtime/run_context_builder.py:416` (local is `_mesh_gpkg`, line 224); dead duplicated block at 409-412 |
| 4 | **PIPE deadlock in batch runner** — polls `proc.poll()` without draining stdout/stderr; any child emitting >64 KB blocks forever; `_active_procs` grows unboundedly | `swe2d/cli/batch_runner.py:275-285` |
| 5 | **CLI headless path re-applies RCMK to an already-RCMK-ordered BLOB mesh** and `INSERT OR REPLACE`s the re-permuted BLOB back, silently invalidating previously baked results for that mesh_name; the designed escape hatch `build_mesh_from_baked` has **zero call sites** | `backend_initializer.py:111-138`, `gpkg_persistence_service.py:94-97`, `backend.py:434-490`; spec: `mesh_persistence_service.load_baked_mesh` docstring, `backend.py:831` |
| 6 | **Manning's n default differs across layers** — 0.020 (UI) / 0.035 (backend API) / 0.030 (backend docstring): GUI and headless runs get silently different friction | `model_tab_view.py:559`, `runtime/backend.py:937`, `runtime/backend.py:11` |
| 7 | **`batch_worker.py` parallel mode races** — unlocked `_replay_files_to_cleanup` list; `_cleanup_replay_files` deletes other in-flight sims' replay JSONs; `_failed` incremented without lock on some paths; `proc.wait(timeout)` unreachable while child holds stdout open | `workers/batch_worker.py:116-121, 136-143, 157-196` |
| 8 | **Bridge plans / coupling SOA build failures swallowed with no log** — runs silently execute without configured structures; neighboring blocks log warnings, these don't | `run_context_builder.py:489-490, 502-503`; also `runtime/coupling.py:2516-2520` (redistribution silently disabled) |
| 9 | **`signal_helpers.connect_lambda` broken** — `obj = ref()` never assigned → `NameError` on first signal; currently dead code, a trap | `swe2d/workbench/signal_helpers.py:84-93` |
| 10 | **Duplicate `PGTimeSeriesWidget` properties** — 5 properties defined twice in one class; the second `selected_element_id` setter silently overrides the first and drops the `_element_id_combo` sync | `views/studio_viewer_pg.py:281-318` vs `:650-692` |

### P0 — Release hygiene (pure junk / broken first-run)

- Committed junk files that **sync to the public repo**: root files `1000`, `10000` (accidental shell captures), `session-ses_093a.md`, `sqlite3/`, `parameters.tex`, `symbology-style.db`, `user-history.db` (the DBs are gitignored but present on disk — check tracking), `tests/results/_qgis_launcher.sh` + `_qgis_exec_script.py` (generated but tracked), `cpp/api/` (540K doxygen output, redundant with `docs/cpp/api`).
- **Stale root `Makefile`** — CMake-generated in-source artifact pointing at a nonexistent path (`/home/aaron/QGIS_Plugins_dev/hydra2dgpu`) and deleted targets (`hydra_hybridmesh`). Delete it.
- **README points at nonexistent `requirements.txt`** (`README.md:62,84`; only `docs/requirements.txt` exists). `__init__.py`'s own hint builds the same broken path.
- **`tools/package_release.py` would ship junk** — exclusions miss `.opencode/`, `.agents/`, session files, DBs; dead no-op at line 82-83; missing-native-lib is a WARNING not a failure.
- **Missing test fixture** `tests/mocks/empty_project.qgs` referenced by `tests/run_headless_qgis_tests.sh:193,270`.
- **PLANNING.md verification gate references nonexistent `tests.test_workbench_imports`** (real file: `test_workbench_gui.py`).

### P1 — Architecture & docs (fix before or during migration branch)

- `view_protocols.py` legitimizes 6 widget-returning protocol methods (Rule 2); `overlay_parameters_service.py` reads widgets with silent fallbacks (Rules 3+7+8); `run_controller.py` `_capture_inflow_progressive` + `wp.get(...)` fallbacks (Rules 1+7+8). See §2.
- `docs/STUDIO_GUI_API.md` and `docs/DEVELOPER_GUIDE.md` §8 describe removed APIs — **fix/delete before dispatching the QGIS4 migration plan** (subagents will write against phantom APIs).
- `metadata.txt` far below migration-plan requirements (`qgisMinimumVersion=3.0` but README says 3.28+; missing `about`, `qgisMaximumVersion`, `icon`, etc.; leading space in author).
- `docs/cpp/api/` (436 Doxygen HTML files, 3.8 MB) is git-tracked and will bloat the public repo — INDEX.md's claim that generated docs are untracked is false.
- CHANGELOG `[Unreleased]` is ~40 commits behind (entire pipe1d rewrite, Network Profile Viewer, GPKG Explorer rewrite missing).

---

## 1. Duplicate Code & Dead Code

### 1.1 Duplicate code

**High**

- **D1. Duplicated property block in `PGTimeSeriesWidget`** — see P0 #10. Bug-class duplicate.
- **D2. Divergent duplicate hydrograph parsers** — `boundary_and_forcing/hydrograph_logic.py:13,34` (`parse_time_hours`/`parse_hydrograph_text`, returns `None` on empty) vs `workbench/services/text_parser_service.py:28,77` (raises `ValueError` on empty). Both live: hydrograph_logic via `studio_dialog._parse_hydrograph_text` (studio_dialog.py:2314); text_parser_service via `model_tab_view.py:1522`. Same algorithm, different error contracts.
- **D3. `_quote_ident` copy-pasted into 5 modules** (+ a 6th renamed twin `_quote_sqlite_ident`): `results/db_utils.py:47`, `services/drainage_graph_service.py:28`, `numpy_blob_service.py:20`, `profile_persistence_service.py:50`, `profile_pipeline_service.py:72`, `gpkg_operations_service.py`. Consolidate into `db_utils`.
- **D4. `_unit_labels`/`_label_for_var` triplicated** — `services/results_render_service.py:24,32`; `views/studio_viewer_pg.py:37,45`; `views/studio_viewer_profile_pg.py:44,52` (view copies byte-identical).
- **D5. Plot export methods near-verbatim across the two pyqtgraph viewers** — `_save_plot_png/svg/pdf` + `_save_data_csv`: `studio_viewer_pg.py:545-620` vs `studio_viewer_profile_pg.py:1352-1410`; `_c2q` identical at `studio_viewer_pg.py:72` / `studio_viewer_profile_pg.py:68`. Candidate: shared mixin.

**Medium**

- **D6.** `classify_boundary_edges` — same min-distance algorithm in `boundary_and_forcing/boundary_runtime_logic.py:116` and `workbench/services/mesh_service.py:178`.
- **D7.** `get_table_info`/`get_table_contents` duplicated: `results/db_utils.py:52,69` vs `gpkg_operations_service.py:193,214`.
- **D8.** `_BC_OPTIONS` triplicated: `constants_service.py:27`, `studio_dialog.py:110` (dead), `map_tab_view.py:16`.
- **D9.** `_apply_timeseries_bc_values`/`_apply_external_sources`/`_distribute_total_flow_to_unit_q` adapter wrappers duplicated: `studio_dialog.py:2087-2165` vs `workers/simulation_worker.py:180-263` (GUI thread vs worker thread copies).
- **D10.** `_as_float`/`_as_int`/`_normalize_cell_type` near-copies: `mesh/meshing.py:271,289,307` vs `tools/gmsh_topology_mesher.py:63,70,84`.

**Low:** D11 `_require_h5py` verbatim (`hecras_model_export_service.py:401` / `hecras_export_service.py:21`); D12 `_env_float` near-copies (`meshing.py:315` / `non_gui_runtime_service.py:24`, both dead); D13 `_open_conn` twin (`numpy_blob_service.py:24` / `profile_pipeline_service.py:76`); D14 `_nearest_cell` closure duplicated (`run_context_builder.py:448` / `studio_dialog.py:2015`); D15 near-identical run-record dataclasses (`results/queries.py:30` / `results/run_service.py:31`); D16 `unit_conversion_service.py:58-80` pure 3-line delegates to `swe2d.units` (redundant facade).

### 1.2 Dead code

**Whole modules never imported (production or anywhere):**

| Finding | Severity |
|---|---|
| `swe2d/workbench/map_tools.py` (`SWE2DLineDrawTool`) — never imported | high |
| `swe2d/workbench/dialogs/hydrograph_editor.py` (`HydrographEditorDialog`) — never imported | high |
| `swe2d/workbench/dialogs/detached_mesh_dialog.py` (`SWE2DDetachedMeshViewDialog`) — never imported | high |
| `swe2d/workbench/dialogs/detached_panel_dialog.py` — never imported; `studio_dialog._detach_tab` (studio_dialog.py:1157) builds a plain QDialog inline; class references nonexistent `self._log` (line 46) | high |
| `swe2d/services/qgis_terrain_interpolator.py` — superseded by `terrain_assignment_service.idw_interpolate_points` (used at `elevation_source_service.py:139`); only a test docstring mentions it | high |
| `swe2d/workbench/workbench_api.py` — Protocol module never imported by any file | medium |
| `tools/inline_topology_helpers.py` — broken one-off script (missing `import os`, targets nonexistent path) | high |

**Production-dead modules exercised only by tests:** `results/export_service.py` (medium); `results/structure_service.py` `load_line_geometry:38`, `filter_structure_records:128`, `resolve_structure_profile_overlays:150`, `load_structure_overlay_data:348` (medium); `workbench/services/run_service.py` `collect_run_parameters:10`, `compute_progress:89`, `validate_run_configuration:142` (medium).

**Dead chains / classes / methods:**

- **F1. Entire run-orchestrator chain dead (high):** `SWE2DRunOrchestrator.run` (`runtime/run_orchestrator.py:55`), `SWE2DRunRequest.from_ui_values` (`:26`), `SWE2DRunLifecycle.handle_run_failure`/`finalize_cleanup` (`run_lifecycle.py:19,27`), `studio_dialog._dispatch_run_request` (`studio_dialog.py:2184`).
- **F2. `results/data.py` — 17 public methods with zero call sites (high, ~200 lines):** `append_line_snapshot:260`, `append_line_profile_snapshot:276`, `get_live_line_snapshot_rows:447`, `get_live_line_profile_rows:464`, `add_manual_selected_keys:750`, `add_results_files:754`, `run_ids_for_gpkg:775`, `all_timesteps:842`, `current_frame_idx:857`, `set_current_time:862`, `t_sec_to_frame_idx:866`, `frame_idx_to_t_sec:870`, `get_coupling_run_id:968`, `active_overlay_run_id:986`, `get_snapshot_at_time:1007`, `save_to_project:1097`, `restore_from_project:1111`.
- **F3. 11 dead protocol getters (high)** superseded by `collect_params`/`collect_storage_params`: `model_tab_view.py:383,386,389,392,395,398,1516,1545,1593,1602,1606` + matching declarations in `view_protocols.py:23,44,68,74,77,80-95`.
- **F4. 4 dead functions in `cli/gpkg_adapter.py` (high):** `query_sample_lines_from_qgis:77`, `query_bc_arrays:131`, `read_drainage_config_from_gpkg:520`, `load_and_configure_hydrographs:774`.
- **F5. `RunContext` replay API dead (high):** `to_replay_json`/`from_replay_json`/`from_widget_params` (`workers/run_context.py:145,246,323`) — payload built manually in `run_controller.build_replay_payload:893`.
- **F6. `studio_dialog.py` dead methods (high):** `_export_mesh_to_layers:650`, `_export_mesh_to_ugrid:654`, `_assign_node_z_from_terrain:745`, `_export_results_to_ugrid:749`, `_pull_node_z_from_layer:753`, `_resolve_overlay_time:1377`, `get_uniform_inflow_velocity:2370`.
- **F7. Dead module-level constants (medium):** `_BC_OPTIONS:110`, `_BC_TS_FLOW:122`, `_BC_TS_STAGE:123`, `_BC_INFLOW_Q:124` in `studio_dialog.py` (shadowed by local imports at 1833-34).
- **F8. 9 unused constant tables in `constants_service.py` (medium):** `BC_OPTIONS:27`, `RECONSTRUCTION_OPTIONS:41`, `TEMPORAL_ORDER_OPTIONS:53`, `STRUCTURE_TYPE_VALUE_MAP:64`, `DRAIN_NODE_TYPE_VALUE_MAP:74`, `DRAIN_LINK_SHAPE_VALUE_MAP:82`, `RAIN_GAGE_UNITS_VALUE_MAP:91`, `HYETOGRAPH_VALUE_TYPE_MAP:100`, `HYETOGRAPH_UNITS_VALUE_MAP:107`.
- **F9. Scattered dead helpers (high unless noted):** `results_render_service.py:154,177` (`update_vline`, `render_timeseries_on_figure`); `mesh_service.py:29` `build_node_coords`; `schema_definitions.py:391` `get_geom_column`; `model_gpkg_loader_service.py:33` `get_model_gpkg_layer_names`; `project_settings_bridge.py:12,87,101` layer-selector trio; `non_gui_runtime_service.py:16,24,36` `_env_*`; `meshing.py:315` `_env_float`; `batch_simulation_dialog.py:148,977` (medium); `detached_log_dialog.py:28` (low); `studio_viewer_profile_pg.py:73` `_c2q_alpha` (low).
- **F10. Test-only public functions (medium/low):** `results/queries.py:102,114`; `results/timestep_service.py:18,64`; `results/profile_service.py:14`; `profile_persistence_service.py:191`; `units.py:89` (`si_m3_per_model_volume`).

**Micro-findings:** ~20 unused imports/locals in `hecras_model_export_service.py` (`import textwrap:21`, `start_date:1101`, 8 unused locals), `units.py:20-27` unused constants, `gpkg_adapter.py:538,821,972` unused locals, `high_perf_viewer.py:707`, `line_sampling_service.py:1030`, `studio_viewer_profile_pg.py:586-587,1102,1116` unused locals. `tests/vulture_whitelist.py:53` `_load_tests` is itself dead (hook must be named `load_tests`).

**Positive:** AST scan found zero unreachable statements and zero commented-out code blocks ≥6 lines. `tests/test_no_dead_imports.py` actively passes. `tests/vulture_whitelist.py` module refs are all still valid.

**Top 10 cleanup candidates (ranked by removal confidence):** F2 (17 methods in `results/data.py`) → M1-M4 (dead dialogs/map tools) → F1 (run-orchestrator chain) → M5 (`qgis_terrain_interpolator.py`) → F5 (RunContext replay API) → F3 (11 dead getters) → F4 (gpkg_adapter) → F6/F7 (studio_dialog) → F9 cluster → M7 (`tools/inline_topology_helpers.py`).

---

## 2. MVP Architecture Violations

Spec: `.opencode/rules/MVP_ARCHITECTURE.md`. Summary table:

| Rule | Violations | Highest severity |
|---|---|---|
| 1 — Controller widget reach-through | 7 sites | High |
| 2 — Protocol methods returning widgets | 10 methods | High |
| 3 — Qt in service layer | 5 real + 3 low | High |
| 4 — numpy in View (incl. controller numpy) | 6 sites | High |
| 5 — `__getattr__` proxy | 0 | Pass |
| 6 — Widget reparenting | 0 | Pass |
| 7 — `.get(key, fallback)` on widget params | 3 | High |
| 8 — Silent backwards-compat fallbacks | 5 sites | High |
| 9 — Callback anti-pattern | 3 | High |
| 10 — Freestanding layouts | 1 (test-only) | Low |

### Rule 1 — Controller → widget reach-through

- **HIGH** `run_controller.py:39-46` — `_capture_inflow_progressive` reaches View → sub-view → QCheckBox with getattr-None fallback (also Rule 8). Fix: `view.model_tab.is_inflow_progressive()` exists at `model_tab_view.py:1597`; expose via host protocol, fail fast.
- **HIGH** `topology_controller.py:34-50, 84-94, 212-213` — `topo = view._topology_tab_view`; controller receives a raw widget bag from `_populate_gmsh_quality_controls()` and connects `currentIndexChanged`/`valueChanged` signals itself. Fix: move signal wiring into `TopologyTabView`.
- **MEDIUM** `mesh_controller.py:78-104, 511-525` — controller constructs and execs its own QDialogs ("Load Mesh From Layers", "Assign Node Z From Terrain"). Fix: View-layer dialog returning plain data.
- **MEDIUM** `overlay_controller.py:301-302, 316-318, 488` — controller drives `QgsMapCanvasItem` lifecycle (`removeItem`, `setVisible`, `setZValue`, `clear`).
- **MEDIUM** `run_controller.py:307-308, 1044, 1126` — `view.model_tab.<method>()` reaches through the property (Rule 2 enabler). Methods return plain data, so structural only.
- **MEDIUM** `run_controller.py:405-407` — `view._temporal_dock.set_data(rd)` via getattr.
- **LOW** — widespread `view._log(...)`, `view._mesh_cell_centroids()` (`run_controller.py:328`), `view._refresh_plot()` (`topology_controller.py:376`): private-method calls; promote to public protocol methods.

### Rule 2 — Protocol methods returning widget references

- **HIGH** `views/view_protocols.py:74, 95, 115, 145-151` — the Protocol layer itself legitimizes widget returns: `get_inflow_progressive_chk -> QCheckBox`, `get_storage_checkboxes -> Dict[str, QCheckBox]`, `get_run_list_widget -> QListWidget`, `get_run_btn/get_cancel_btn/get_progress_bar`. Redundant with existing plain-data methods (`is_inflow_progressive`, `collect_storage_params`). Fix: delete.
- **HIGH** `studio_dialog.py:645-648` `get_run_list_widget` — live caller `views/studio_results_panel.py:157`. Fix: `get_selected_run_keys() -> list`.
- **MEDIUM** `studio_dialog.py:619-632` — `model_tab`/`results_toolbox`/`run_dock` properties return concrete Qt sub-views (exploited at `run_controller.py:307`). Fix: narrow protocol adapter or host-level pass-throughs.
- **LOW** — latent widget getters: `run_dock.py:64`, `model_tab_view.py:398,1602`; `canvas` properties in viewers appear view-internal (rename with underscore or document).

### Rule 3 — Qt in the service layer

- **HIGH** `workbench/services/overlay_parameters_service.py:25-52, 111-166` — reads results-toolbox widgets (`currentData`/`isChecked`/`value`) via `_safe_*` swallow-everything helpers and writes `view._overlay_opacity` (line 150). Docstring claims Rule-8 compliance; the `_safe_*` helpers *are* silent fallbacks. Fix: View collects params into a plain dict.
- **HIGH** `workbench/services/widget_persistence_service.py:136, 234-262, 282` — service whose entire job is widget I/O (`isChecked`/`setValue`/`setText`, lazy `from PyQt5 import QtWidgets`). Fix: split View-side reader/writer + service-side dict (de)serialization.
- **MEDIUM** `workbench/services/batch_manager.py:12` — top-level `from PyQt5.QtCore import QObject, pyqtSignal`. Move to `workers/`.
- **MEDIUM** `results/animation.py:15-20` — Qt playback controller in a Qt-free dir. Relocate to `workbench/`.
- **MEDIUM** `results/high_perf_viewer.py:241-243, 330-332, 389-391, 506-508, 1013-1018` — defines `SWE2DHighPerfCanvasOverlayItem(QgsMapCanvasItem)` inside the service layer; the pure rasterizer is clean. Split: renderer stays, canvas item moves to `views/`.
- **LOW** — `runtime/backend.py:154` (win32-guarded QSettings), `runtime/run_context_builder.py:340,430` (lazy guarded `qgis.core`), `services/qgis_terrain_interpolator.py:20-26` (top-level `qgis.core`; module is dead anyway — see M5). Lazy `qgis.core` imports elsewhere in services are data-model adapters, not violations.
- **PASS:** `swe2d/mesh/`, `swe2d/extensions/`, `swe2d/plotting/`, widget-call checks in `runtime/` and `boundary_and_forcing/`.

### Rule 4 — numpy computation in View/Controller

- **HIGH** `views/studio_viewer_profile_pg.py:586-643, 772-773, 811-860` — WSE derivation, wet masks, color-range normalization, `np.argmin` timestep snapping on results arrays. Fix: move series prep into `results/profile_service.py`.
- **MEDIUM** `studio_viewer_pg.py:397` + `studio_viewer_profile_pg.py:1108` — duplicated hover-snap `np.argmin`. Shared service helper.
- **MEDIUM** `studio_dialog.py:1887-1896` (`_preview_spatial_manning`) — array stats in View.
- **LOW** `studio_dialog.py:2256-2286` (`_mesh_cell_polygons`) — per-cell geometry loop; belongs in `mesh_computation_service`.
- **Controller numpy:** HIGH `overlay_controller.py:608-611, 700-701` (`np.nanmin/nanmax` extents); MEDIUM `topology_controller.py:324-327, 659-662` (grid derivation); LOW `run_controller.py:778` (Manning stats for a log line).
- **PASS:** `model_tab_view.py`, `map_tab_view.py`, `topology_tab_view.py`, `results_controls.py`, `run_dock.py`.

### Rules 5 & 6 — PASS

No `__getattr__` proxy in View; `studio_dialog.py:1154` `page.setParent(None)` is the deliberate detach-tab feature, not cross-view theft.

### Rule 7 — `.get(key, fallback)` on widget params

- **HIGH** `run_controller.py:303, 341` — `wp.get("open_bc_relax_spin", 0.0)` and `wp.get("save_max_only_chk", False)`; all sibling keys use fail-fast `wp["..."]`. Fix: bracket access.
- **MEDIUM** `run_controller.py:890` — `flat_params.get("run_duration_s", 0.0)`.

### Rule 8 — Silent backwards-compat fallbacks

- **HIGH** `run_controller.py:37-48` — three-level getattr chain ending `return False` (inflow-progressive silently off if renamed).
- **HIGH** `overlay_parameters_service.py:113-115` — `_w(name)` getattr-None for every widget lookup.
- **MEDIUM** `topology_controller.py:343, 483-485`; `studio_dialog.py:2249-2254` (`get_n_mann_value` silent 0.03 default).
- **LOW** `studio_dialog.py:636-638` (`getattr(self, "_results_toolbox", None)`).

### Rule 9 — Callback anti-pattern

- **HIGH** `run_controller.py:345-349` → `studio_dialog.py:2107-2145` — the documented anti-pattern: runtime invokes View methods with raw mesh/BC arrays. Mitigating: View delegates immediately to `_logic` services. Fix: capture widget values at run setup; pass service functions, not View methods.
- **MEDIUM** `studio_dialog.py:2147-2168` (`_apply_external_sources`) — reads **7 spin widgets per invocation** during time stepping. Snapshot at run start.
- **MEDIUM** `studio_dialog.py:1817-1854` — raw combo widgets forwarded through service signatures (adapter never calls widget methods — no Rule-3 break). Resolve layers in the View first.
- **GOOD EXAMPLE** `studio_dialog.py:2288-2307` (`_build_line_sampling_map`) — matches the spec's CORRECT pattern.

### Rule 10 — Freestanding layout

- **LOW** `views/topology_tab_view.py:908-910` — deliberate never-parented `QFormLayout` kept for legacy standalone tests (comment admits it). Production path returns real layouts. All other no-parent layouts are canonical nested-row patterns — PASS.

### Architecture test gaps

- `tests/test_mvp_imports.py` loopholes: matches only `m == "PyQt5.QtWidgets"` — `from PyQt5 import QtWidgets` (module `"PyQt5"`) and `from qgis.PyQt.QtWidgets import X` evade it (this is how `high_perf_viewer.py:1015` and `animation.py:19` pass); `PyQt5.QtCore/QtGui` unrestricted (how `batch_manager.py:12` passes); `swe2d/plotting/` not in `_SHARED_SERVICES_DIRS`; no checks for Rules 1, 2, 4, 7, 8.
- `tests/test_view_protocols_complete.py` — presence-only `hasattr` checks; does not check return types (implicitly blesses the 6 widget-returning methods); misses `ModelTabViewProtocol`/`RunDockProtocol`/`ResultsToolboxProtocol`.

**Top three fixes by impact:** (1) `overlay_parameters_service.py` → View protocol + fail fast; (2) `run_controller.py` `_capture_inflow_progressive` + `wp.get` fallbacks; (3) delete the 6 widget-returning methods from `view_protocols.py`.

---

## 3. Potential Bugs

### 3.1 Exception handling (62 except-pass blocks, 40 `except Exception`; high-risk subset)

- **`run_context_builder.py:489-490, 502-503`** — bridge plans / coupling SOA failures swallowed silently (P0 #8). Neighboring blocks log; these don't.
- **`runtime/coupling.py:2516-2520`** — `use_redistribution=True` silently degrades to no redistribution.
- **`runtime/coupling.py:1538-1539, 1550-1551`** — structure-flow/diagnostic readback swallowed; stale `_last_structure_flows` reported as current (adjacent block at 1525 logs).
- **`run_context_builder.py:185-186`** — units block computation swallowed; combined with wrong defaults (P0 #1) silently yields bad unit factors.
- **`cli/gpkg_adapter.py:73-74`** — `query_mesh_from_gpkg` swallows BLOB-deserialization/DB-corruption errors; every caller reports "Mesh not found", misdiagnosing corruption. Inner block at 58-67 leaks `_c` on exception.
- **`results/timestep_service.py:80-86`** — `conn` unbound in `finally` if `connect` raises (works by accident via NameError catch); corrupt GPKG silently looks like "no coupling data".
- **`units.py:172-174`** — `si_m_per_model_from_wkt` swallows parse errors → returns 1.0 (SI): malformed WKT silently flips USC model to SI conversions (wrong gravity, Manning factor, rain depth).
- **`cli/headless_runner.py:31-37`** — `_atomic_write_json` swallows write failures; status file silently missing.
- **Leftover debug noise:** `simulation_worker.py:533-535, 768-773, 793`, `run_controller.py:439-447` (`[LINE_DIAG]`/`[PipeCell]` at WARNING in production); `simulation_worker.py:806` logs "The numbers go UP! They go UP UP UP!!!" every run.
- **Benign-by-design (verified, no action):** ~30 blocks incl. all 12 Qt `RuntimeError` deleted-wrapper blocks, `backend.py:1406`, `backend.py:416-419/488-490`.

### 3.2 Mutable state

- No mutable default args or shared mutable class attributes found. 
- **Low-medium:** `units.py:34-41` module-level mutable unit globals mutated by `configure()` from both GUI and CLI; `simulation_worker.py:729` reads the global instead of `ctx` (latent cross-contamination within a session).

### 3.3 Indexing / permutation

- **CLI re-RCMK hazard** — see P0 #5.
- **Low:** `coupling.py:2011-2013` `_remap_cells_for_gpu` passes positive out-of-range indices through unclamped (defense-in-depth only; `pack_coupling_soa` pre-clamps). `terrain_assignment_service.py:83` and `mesh_service.py:163-173` verified correct. `run_finalizer.py:519-523` ragged snapshots fail loud (acceptable).

### 3.4 Threading / Qt

- **Medium:** `simulation_worker.py:770-771` shallow `dict()` copies share numpy line-data arrays the worker keeps mutating while GUI reads (`data.py:274` writes after `snapshot_ready.emit`; GUI reads at `run_controller.py:442-444`). Coupling dict is properly copied at 774-779 — line data was missed.
- **Medium (latent, dead code):** `PersistenceWorker` never instantiated; if used it passes raw `view` to `SWE2DRunFinalizer` and calls widget methods from `QThread.run()`.
- **Medium-high:** `batch_worker.py` races — see P0 #7.
- **Verified OK:** `SimulationWorker` permutation handshake (`threading.Event`, 60 s timeout); `_WorkbenchShim._cancel_requested`; `BatchWorker._batch_done` locking.

### 3.5 Resource leaks

- `dialogs/batch_simulation_dialog.py:756-817` — `sqlite3.connect` without try/finally; `else: return runs` (~767) and exception paths skip `close()`.
- `studio_viewer_profile_pg.py:920-928` — `close()` skipped if `execute` raises.
- `topology_controller.py:1038-1043` — `_mesh_in.pkl` temp file never unlinked (leak per mesh run).
- **Low-medium:** `SWE2DBackend.destroy()` (`backend.py:1397-1409`) does not call `free_snapshot_buf`; if native destroy doesn't free the device ring buffer, device memory leaks per run (needs native-side confirmation).
- All other ~50 `sqlite3.connect` sites verified try/finally.

### 3.6 Type / unit hazards (per `docs/UNIT_ASSUMPTIONS_AND_USC_DEFAULT.md`)

- **P0 #1** (`run_context_builder.py:622-625`) and **P0 #2** (`simulation_worker.py:731`) — the two big ones.
- Same wrong unit defaults duplicated in `workers/run_context.py:122, 314, 388` (`rain_mm_to_model_depth: float = 1.0`); dead `_units_block` at `run_context_builder.py:178-184` repeats inverted values (`flow_si_to_model` should be `model_per_si_m()**3`) — a trap since `run_controller.py:866` currently pops it.
- **Medium:** `cli/headless_executor.py:181-185` — hardcoded `"m"`/`1.0` in headless finalization → wrong SI-reference mass balance for all USC headless runs.
- **Medium:** `extensions/structures.py:96, 113-115` — unknown structure type silently → CULVERT; missing `upstream_cell`/`downstream_cell` silently → cell 0 (flow injected at arbitrary cell instead of config error).
- **Low-medium:** `run_context_builder.py:133-148` `_parse_duration` drops seconds in `HH:MM:SS` (rainfall_hydrology.py:55-61 handles 3 parts — inconsistent); `:528-529` `float()` on possibly `"HH:MM"` string → unhelpful ValueError; `pipe_network_service.py:204` outfall `max_depth` silent default 10.0.
- **Low:** `coupling.py:2495` unknown `culvert_solver_mode` silently → 0.

### 3.7 Logic smells / stale references

- `terrain_assignment_service.py:89` float `== 0.0` (exact-computed, works; tolerance more robust). `qgis_terrain_interpolator.py:124` exact nodata compare.
- `run_controller.py:486-492` dead branch (`not ok and not cancelled` — worker sets `ok = not cancelled`).
- `topology_controller.py:1206` `getattr(view, "_topology_mesh_backend", "gmsh")` — attribute initialized to `None`, so `"gmsh"` default is dead (cosmetic log line).
- **Medium-low:** `hydra_plugin.py:644-649` — unload path tries importing nonexistent `swe2d.processing` and calls `addProvider` (inverted logic + guaranteed ImportError → warning on every plugin unload).
- **Low:** `hydra_plugin.py:309` — `app.notify = _traced_notify` can't override the C++ virtual; HYDRA_TRACE_SIGNALS tracer silently does nothing.
- No dict/set mutation-during-iteration; no shadowed builtins.

### 3.8 Tests

- **Coverage gaps tied to findings:** no test pins unit defaults in `build_run_context_from_dict` (the CLI 1000×-rainfall bug is untested); `tests/test_cli.py:51` uses a 1-cell mesh (RCMK identity — cannot catch the re-permutation hazard).
- `tests/test_line_results_plot.py:16-18` references nonexistent `example_project/` → **always skips** (false coverage).
- `tests/test_swmm_validation_v2/v3/v4` are scaffolding meta-tests, not duplicates (437 lines of hasattr-asserts could be consolidated; optional).
- No unconditional `@unittest.skip`, no early-return-disabled tests; conditional GPU skips are legitimate. Assert-free tests verified as smoke tests or helper-delegated.

**Top 10 most likely real bugs:** see P0 table (items 1-10 map to bugs-audit rankings 1, 2, 3, 4, 5, 7(batch_worker), 9(signal_helpers), plus cross-thread line data, bridge-plan swallow, headless units). Honorable mentions: `PersistenceWorker`, `hydra_plugin.py:644-649`, `timestep_service.py:82-86`, `gpkg_adapter.py:73`.

---

## 4. Documentation Audit

### 4.1 Stale / inaccurate docs (per-file)

**`docs/STUDIO_GUI_API.md` — heavily stale (describes removed public API):**
- `workbench_controller.py`/`WorkbenchController` referenced (lines 136, 162-165) — file doesn't exist (removed in 1.2.0); controllers now in `workbench/controllers/`.
- Wrong module path for `collect_overlay_parameters` (lines 128, 142 → actual `workbench/services/overlay_parameters_service.py:101`); missing required `t_use` arg (line 124).
- `load_gpkg_data` claimed at lines 117/143 — function doesn't exist in `gpkg_operations_service.py`.
- Lists `mesh_tab_view.py`/`boundary_tab_view.py` (lines 137, 140, 20) — only `map/model/topology` tab views exist.
- Dead link `../graphify-out/wiki/index.md` (line 204).

**`docs/DEVELOPER_GUIDE.md` — "v2.0, 2026-06-14", predates the 1.2.0 refactor:**
- `swe2d/core/` shim (lines 34, 428-438) — package removed; `results/panel.py`, `velocity_layer.py` (65, 94-96, 387-391) — removed; wrong paths for non_gui modules (408-425); `forms/*.ui "~12 designers"` (30) — only one form module, no `.ui`; `meshing.py (10,337 LOC)` + `TQMeshBackend` (140, 148) — actual 4,724 LOC, backends are GmshBackend/StructuredFaceCentricBackend.
- Phantom build target `hydra_hybridmesh`/`hybrid_mesh_bindings.cpp` (49, 501-518); wrong option `USE_CUDA` (actual `BACKWATER_USE_CUDA`, CMakeLists.txt:51).
- §5.4 test catalog (742-774): ~10 rows reference deleted test files.
- §6.4 (871-875): "no CI", "conftest.py empty" — both false (`.github/workflows/` has 3 workflows; `tests/conftest.py` + `tests/mocks/qgis_env.py` are substantial).
- §7.7 references nonexistent `tools/ui_bind_sync.py` (982).
- **§8 (987-1143) describes a superseded Studio component API** (`studio_component.py`, `_destroy_component`, `_register_left_tab`, `SWE2DStudio{Name}Dock` naming) — none exist; actual: `views/studio_component_view.py`, `WorkbenchDialogBuilder._build_component` (`workbench_dialog_builder.py:261,311`), `HYDRA2D{name.title()}Dock` (:284). **Fix/delete before the QGIS4 migration branch.**
- Dangling private refs: `.opencode/rules/MVP_ARCHITECTURE.md` (114), root `AGENTS.md` (713 — doesn't exist).
- Enum docs (332) missing SSP_RK3/GRAPH_SAFE_RK5 (`extension_models.py:35-42`); no mention of `swe2d/cli/`, `swe2d/services/`, `swe2d/plotting/`.

**`docs/USER_GUIDE.md` — pre-restructure UI map:**
- §3-§5 tab/toolbox map stale — actual tabs "Mesh Generation"/"Simulation" (`studio_tab_builder.py:44-45`), 5 toolbox pages each; model tab's last page is "Output" not "Run / Output".
- §4.3 + `GPKG_EXPLORER_GUIDE.md:8` — Explorer launched from menu action (`workbench_main_menu.py:213`), not the Utilities page.
- §6.1: MP5 listed as valid (457) but disabled in GUI (`model_tab_view.py:605-614`); RK3 missing (458; present at 629-637).
- Line 110: removed `SWE2D_STATE_FP32` flag (CMakeLists.txt:56-60; CHANGELOG:147); lines 112-116: phantom `hydra_native.so` target.
- §9 references private `reference/anuga_validation_tests/` (817-824); §10 QML table missing `swe2d_internal_flow_sources.qml` (881-900); footer wrong repo URL `qgis-hydra-plugin` (918); header "Version 2.0" vs product 1.2.

**`docs/MODEL_GEOPACKAGE_SCHEMA.md`** vs `schema_definitions.py`:
- Claims 18 tables (line 7) — actually 19 (`swe2d_internal_flow_sources` missing; even the code docstring at :382 says "18").
- `swe2d_bc_lines`: phantom `hydrograph_layer` field (151-153); `bc_relax` per-edge override undocumented.
- `swe2d_drainage_nodes`: 12 undocumented fields (outfall 5-mode BC, junction overflow) — doc has 10, schema has 22 (:194-232).
- `swe2d_drainage_links`: 14 undocumented fields (culvert_* block, inverts, losses, `max_cell_length`) — doc 17 vs schema 31 (:233-266).
- `swe2d_drainage_inlets`: 10 undocumented fields (HEC-22 inlet types) — doc 8 vs schema 18 (:267-290).
- `swe2d_structures`: accurate ✓.

**`docs/RESULTS_GEOPACKAGE_SCHEMA.md`** vs `gpkg_persistence_service.py`:
- `swe2d_baked_pipe_cell_ts`: doc 8 cols vs actual 11 (`cell_invert/width/height/shape_type`, :684-697).
- False claim line 45: "dry cells omitted" — writer flattens full n_steps×n_cells (:516-549).
- `swe2d_run_replays` read by `batch_simulation_dialog.py:770` but **no writer exists anywhere** and it's undocumented; the INDEX "revision-2" consolidation hasn't landed.

**`docs/CLI_GUIDE.md`:**
- Missing `headless_executor.py` in module table (13-18); **undocumented `replay` subcommand** (`__main__.py:46-48` → `headless_runner.py:191`).
- `--progress` doc overstates output (39): prints only `progress=<pct>%`, callback dict is empty — the doc's example would `KeyError` (`__main__.py:125-130`, `headless_runner.py:149-153`).
- Requirements missing `scipy` (used at `gpkg_adapter.py:193,629`).

**`docs/DRAINAGE_SOLVER_MODE_GUIDE.md`:**
- Lines 27-29: `cell_k_loss` "in the flux accumulation kernel" — superseded (losses now in implicit momentum denominator, per session log 2026-07).
- Lines 50-58: "DYNAMIC — Full Saint-Venant" — contradicted by session log finding (local-inertia/diffusive approximation).
- No mention of the July 2026 pipe1d rewrite (Godunov FVM, HLLE/HLLC face-flux coupling, Preissmann slot, MUSCL-minmod, semi-implicit friction, `friction_method`/`surcharge_method`).

**`docs/UNIT_ASSUMPTIONS_AND_USC_DEFAULT.md`** — self-dated 2026-07-15 but describes pre-fix state: its "What Is NOT Unit-Aware" list (lines 12, 19-23) was fixed the same day in commit `e779774`. The doc now instructs fixing already-fixed code.

**`docs/RAINFALL_CN_GUIDE.md`:** §5 subcycle default 100 → actual GUI default 16 (`model_tab_view.py:955-960`); `green_ampt` infiltration default has no GUI exposure (doc's table misleading; GUI offers only `scs_cn`/`none`, :864-868).

**`docs/GPKG_EXPLORER_GUIDE.md`** — predates the 2026-07-18 enhanced-viewer rewrite (Plot tab, `SWE2DEnhancedTablePreviewDialog`, blob inspection, CSV export all undocumented).

**`docs/GMSH_MESHING_GUIDE.md`** — §4 references "Map tab"/"Topology tab" (lines 140, 152); actual tabs renamed.

**`docs/cpp/ARCHITECTURE.md`** — omits `pipe1d.cu/cuh`, `swe2d_reconstruct.cu`, `swe2d_xsect_constants.h`; lists phantom `hybrid_mesh_bindings.cpp`/`hydra_hybridmesh`.

**`docs/SWE2D_GPU_ARCHITECTURE_REPORT.md`** — line 10 scheme enum stale (0-5 vs actual 0-8); needs a "snapshot in time" caveat.

**Pipe1D plan docs — statuses never updated:** all 5 carry `Status: Proposed` while the work is committed (`pipe_end_face_flux_plan.md` ← `27d016f`; `pipe1d_phase_a_implicit_plan.md` ← `2d919e1`; slot fix ← `f0b4ed7`; Casulli-Hu phases B/C/D explicitly deferred). `PIPE1D_AUDIT_2026-07-17.md` lacks "resolved by" annotations for `230ed91`/`27d016f`.

**`README.md`:** line 49 removed FP32 flag; lines 11-12 incomplete scheme lists; lines 62/84 nonexistent `requirements.txt`; line 96 dead `graphify-out/` link; lines 100-113 layout omits `swe2d/cli/`, `swe2d/services/`.

**`docs/AGENT_SESSION_RECOVERY_LOG.md`** — current (latest entry 2026-07-19 = HEAD `1f9ef0a`), but: links into excluded `docs/superpowers/` (lines 174, 252, 293, 364-367) and `reference/` (46, 79, 94); several "Files changed (uncommitted)" entries are stale (48, 81, 111, 139, 170, 206, 239, 299).

### 4.2 Index / version sync

- **`docs/INDEX.md` dead links (broken in both repos):** `IMPLEMENTATION_PLANS/*.md` ×4 (lines 42, 45-47 — directory doesn't exist; also linked from `SOLVER_ORDER_AND_STENCIL.md:16`, `ADVANCED_SPATIAL_SCHEMES.md:3`).
- INDEX missing: `AGENT_SESSION_RECOVERY_LOG.md`, `UNIT_ASSUMPTIONS_AND_USC_DEFAULT.md`, `PIPE1D_AUDIT_2026-07-17.md`, all 4 pipe1d plan docs.
- `graphify-out/` section (79-98) unusable from fresh clones.
- **CHANGELOG `[Unreleased]` ~40 commits behind** — missing: entire pipe1d rewrite (8+ commits), GPKG Explorer rewrite, Network Profile Viewer (12 commits), advanced-widget filtering. `metadata.txt` 1.2 ↔ CHANGELOG 1.2.0 in sync; but doc versions diverge (USER_GUIDE "2.0", DEVELOPER_GUIDE "2.0", pyproject 1.2.0).
- CHANGELOG false claims: line 24 "GUI shows all 9 schemes" (MP5 disabled); lines 154-165 list 10 docs as `docs/…` that actually live in `reference/docs/` (private); line 166 references nonexistent root `AGENTS.md`; line 51 "18 structure layer schemas" — actually 19.

### 4.3 Documentation gaps (ranked)

1. **Network Profile Viewer** — new flagship feature (12 commits, 8 new files), zero user docs.
2. **pipe1d solver rewrite** — current state documented only in 5 overlapping "Proposed" plan docs + private specs; `DRAINAGE_SOLVER_MODE_GUIDE.md` actively mis-describes it. Needs one consolidated guide + plan-status reconciliation.
3. **GPKG Explorer enhanced viewer** — user doc describes the pre-rewrite tool.
4. **CLI `replay` subcommand + GUI→CLI replay JSON workflow** (+ `swe2d_run_replays` table).
5. **`swe2d/services/` package (15 modules)** — absent from DEVELOPER_GUIDE; HEC-RAS export has no doc at all.
6. **Batch simulation GUI dialog** (only CLI batch documented).
7. **`swe2d/workbench/devtools/`** — no developer doc.
8. **Bridge coupling / stacked-bridge architecture** — design docs private only.
9. **`tools/` scripts** — only 3 of 19 documented; `package_release.py`, `export_to_hecras.py`, sanitizers, QML generators, `add_pipe1d_fields_to_existing_gpkg.py` undocumented; DEVELOPER_GUIDE references nonexistent `ui_bind_sync.py`.
10. **SWMM validation workflow** — no doc on running/interpreting.
11. **`swe2d/plotting/`, `swe2d/extensions/`, `units.py` public contract** — thin.

### 4.4 Public-release blockers (docs → private-only/untracked paths)

| File:line | Reference | Problem |
|---|---|---|
| `docs/INDEX.md:43-44` | `superpowers/specs|plans/…` | Dead in public (excluded) |
| `docs/INDEX.md:42,45-47`; `SOLVER_ORDER_AND_STENCIL.md:16`; `ADVANCED_SPATIAL_SCHEMES.md:3` | `IMPLEMENTATION_PLANS/…` | Dead in **both** repos |
| 8 docs (INDEX 79-98; DEVELOPER_GUIDE 1147-1215; README 96; USER_GUIDE 932; STUDIO_GUI_API 204; UI_COMPONENT_GUIDE 207; DRAINAGE_SOLVER_MODE_GUIDE 109; RESULTS_PATH_GUIDE 189) | `graphify-out/…` | Gitignored → dead everywhere |
| `docs/AGENT_SESSION_RECOVERY_LOG.md` (multiple) | `docs/superpowers/…`, `reference/…` | Log syncs; targets don't |
| `USER_GUIDE.md:817-824`; pipe1d plan docs (5 files) | `reference/…` | Dead in public |
| `DEVELOPER_GUIDE.md:114` | `.opencode/rules/MVP_ARCHITECTURE.md` | Excluded |
| `DEVELOPER_GUIDE.md:713`; `CHANGELOG.md:166` | `AGENTS.md` | Doesn't exist at root |
| `CHANGELOG.md:154-165` | 10 spec docs as `docs/…` | Actually `reference/docs/…` (wrong path + private) |
| `USER_GUIDE.md:918` | wrong GitHub repo URL | — |
| `ADVANCED_SPATIAL_SCHEMES.md:5` | relative link to `swe2d/…` resolves under `docs/` | Broken |
| `docs/cpp/api/` (436 files, 3.8 MB) | tracked generated Doxygen | Bloats public repo; INDEX claim "not tracked" is false — add to `.publicsync-ignore` or untrack |
| Root: `session-ses_093a.md`, `sqlite3/`, `parameters.tex`, `1000`, `10000` | tracked stray files | Will sync to public |

### 4.5 QGIS4 migration doc implications (per `reference/qgis-repo-and-qgis4-migration.md`)

1. **`metadata.txt` far below plan requirements:** `qgisMinimumVersion=3.0` (README/USER_GUIDE say 3.28+ — wrong even for current release); missing `about`, `qgisMaximumVersion`, `category`, `icon`, `tags`, `changelog`, `tracker`, `repository`, `experimental`, `deprecated`; leading space in `author=`; plan expects `qgis_plugin/HYDRA2DGPU/metadata.txt` (only root file exists).
2. **PyQt5→qgis.PyQt scope larger than plan Task 4.1:** direct `from PyQt5` imports in 5 production files — `results/animation.py`, `results/high_perf_viewer.py`, `workbench/services/batch_manager.py`, `workbench/services/widget_persistence_service.py`, `workbench/workers/batch_worker.py`. Plan **misses `batch_manager.py` and `batch_worker.py`**.
3. **Packaging conflict:** plan Task 1.2 creates `tools/package_plugin.py`; repo already has `tools/package_release.py` which **bundles native .so binaries** — contrary to the binary-free goal. Undocumented; resolve supersession before the branch.
4. **`__init__.py` eager imports + `sys.path` manipulation** (`__init__.py:14-19, 130-147`) — exactly what plan Task 1.1 Step 3 removes; also points users at the nonexistent root `requirements.txt`.
5. **`pyproject.toml` divergence:** setuptools/`hydra2dgpu`/`>=3.12` vs plan's scikit-build-core/`hydra-swe2d`/`>=3.10,<3.13` — decide canonical before release.
6. **CUDA arch floor consistent** (CMakeLists 75-90, CUDA 13, CC≥7.5) but no public build doc captures the system-g++-13 rule; `docs/DISTRIBUTION.md` doesn't exist.
7. **Plan-promised artifacts missing:** `qgis_plugin/` tree, `pixi.toml`, `build-wheels.yml`/`release.yml` workflows.
8. **Doc-debt hazard for migration implementers:** DEVELOPER_GUIDE §8 + UI_COMPONENT_GUIDE describe phantom APIs — fix/delete before dispatching the migration plan.

---

## 5. Quick Improvements

**Positive non-findings (verified clean):** zero TODO/FIXME/HACK markers; zero wildcard imports; zero bare `except:`; no path string-concatenation; GPU tests properly auto-skipped; `%`-formatting only in logging calls; no unreachable code.

### 5.1 Ranked top 15 quick wins

| # | Finding | Effort | Severity |
|---|---|---|---|
| 1 | `git rm 1000 10000 session-ses_093a.md` (committed junk, syncs to public) | S | high |
| 2 | Delete stale root `Makefile` (wrong absolute path, dead targets) | S | high |
| 3 | Fix README `requirements.txt` references (file doesn't exist) | S | high |
| 4 | Fix `tools/package_release.py` exclusions + fail hard on missing native libs + remove dead no-op (82-83) and `"build/"` entry that can't match | S | high |
| 5 | Unify Manning's n default (0.020/0.035/0.030) — P0 #6 | S | high |
| 6 | Cache `safe_area` in `runtime_step_executor.py:100` — per-timestep `cell_areas().copy()` + `np.asarray` + `np.maximum` triple allocation on hot path | S | high |
| 7 | Untrack `cpp/api/` doxygen output (redundant with `docs/cpp/api`) | S | medium |
| 8 | Create `tests/mocks/empty_project.qgs` or drop the `--project` flag | S | high |
| 9 | Untrack generated `tests/results/_qgis_launcher.sh`/`_qgis_exec_script.py`; gitignore `tests/results/_*` | S | medium |
| 10 | Remove ~40 verified unused imports (list below); add ruff F401 gate to CI | S | medium |
| 11 | Move LTO block below `option()` in CMakeLists.txt:13-16 (currently dead unless flag passed explicitly) | S | medium |
| 12 | Add `logger.debug(exc_info=True)` to silent excepts in `gpkg_persistence_service.py` (8 sites: 152, 357, 361, 376, 407, 413, 458) | S | medium |
| 13 | Reconcile `pyproject.toml` deps (numpy, gmsh) with `docs/requirements.txt` (h5py, netCDF4, matplotlib, pdoc) via optional-dependencies | S | medium |
| 14 | Fix or delete always-skipping `tests/test_line_results_plot.py` | S | medium |
| 15 | Add `encoding="utf-8"` to ~15 text-mode `open()` calls (cli/__main__.py:16,120; batch_runner.py:141; batch_simulation_dialog.py:826,834,985,1000; simulation_config_dialog.py:212; export_service.py:14,25; studio_viewer_pg.py:602; studio_viewer_profile_pg.py:1404; gpkg_plot_tab.py:311) | S | low |

**Honorable mentions:** vectorize boundary-flux accumulation loop (`runtime_sources.py:166`, M); guard `-march=native -ffast-math` for redistributed binaries (CMakeLists.txt:160, SIGILL/FP-semantics risk, S); document/align `run_finalizer.py:133` h_min=1e-4 (S); quote SQLite URI paths (`db_utils.py:15`, `timestep_service.py:68`, `queries.py:54` — breaks on `?`/`#` in paths, S); shared constants module for the `1e-6` h_min literal (10+ sites, S); consolidate `test_swmm_validation_v2/v3/v4` (S); rename `BACKWATER_USE_CUDA` → `HYDRA_USE_CUDA` with compat alias (S).

### 5.2 Verified unused imports (F401, ~40)

`cli/gpkg_adapter.py:9,844`; `cli/headless_executor.py:9,11`; `cli/headless_runner.py:13,18`; `studio_dialog.py:8-13,17,21` (keep intentional re-exports used by `hydra_plugin.py:317`); `workbench_view_state.py:27`; `workbench_dialog_builder.py:372`; `studio_viewer.py:13`; `studio_viewer_profile_pg.py:494,528`; `temporal_dock.py:8`; `graph_editor_dialog.py:3,14`; `workbench_main_menu.py:28,29`; `studio_tab_builder.py:7,11`; `view_protocols.py:9`; `results_controls.py:14`; `studio_component_view.py:18`; `network_profile_map_tool.py:16,20`; `batch_simulation_dialog.py:23`; `coupling_results_dialog.py:9`; `sqlite_preview_dialog.py:9,14`; `run_selection_dialog.py:5`; `hydrograph_editor.py:7`; `save_config_dialog.py:5,7,10`.

### 5.3 Consistency issues

- **Manning's n default** — P0 #6 (high).
- **h_min divergence:** 1e-6 everywhere except `run_finalizer.py:133` (1e-4) — undocumented; verify intent.
- os.path (50 files) vs pathlib (14) — 2 files mix both (`meshing.py`, `run_compute_sanitizer.py`); don't migrate, just stop mixing.

### 5.4 Performance quick wins (Python)

- **High:** `runtime_step_executor.py:100` — see #6 above.
- **Medium:** `runtime_sources.py:166` per-timestep Python loop over boundary-flux edges (`idx.tolist()` + per-element dict.get) — vectorize with precomputed group indices + `np.bincount`.
- **Low (verified setup-only):** `backend.py:411,483,521,564` dict construction; `backend.py:786-831` snapshot merge reallocation (on-demand only).

### 5.5 Test infrastructure

- Missing fixture `tests/mocks/empty_project.qgs` (#8); generated files tracked (#9); benchmark artifacts `tests/artifacts/*` + `tests/merewether_mesh_comparison.png` in git (low); always-skipping `test_line_results_plot.py` (#14).

### 5.6 Build / release

- Stale root `Makefile` (#2); `cpp/api/` tracked (#7); README requirements (#3); pyproject/requirements mismatch (#13); `package_release.py` issues (#4); CMake LTO block ordering (#11); `-march=native` guard (honorable mention).

---

## 6. Suggested Pre-Release Sequence

1. **P0 bugs** (§0 table): items 1-4, 6 are small targeted fixes with clear evidence; item 5 (re-RCMK) needs C++-side confirmation of RCMK idempotency — if non-idempotent, wire `build_mesh_from_baked` into the headless path; items 7-10 likewise bounded.
2. **Release hygiene**: §0 P0 hygiene list + quick wins 1-4, 8-9 (all <1 hr each).
3. **Docs triage before migration branch**: fix/delete DEVELOPER_GUIDE §8 + STUDIO_GUI_API.md; update CHANGELOG Unreleased; scrub private-path links from synced docs (§4.4); bump `metadata.txt` per migration plan Task 1.3.
4. **Architecture quick fixes** (§2 top three) — small, high-signal; leave larger Rule 1/3/9 refactors for post-release.
5. **Dead-code removal** (§1.2 top-10 list) — safe mechanical deletions; run `tests/test_no_dead_imports.py` + full unittest gate after.
6. Post-release: duplicate consolidation (D1-D5), service-layer Qt relocation, test gate strengthening (`test_mvp_imports.py` loopholes, return-type checks in `test_view_protocols_complete.py`).

*Verification gate after any fix phase:* `find . -type d -name __pycache__ -exec rm -rf {} +` then `mamba run -n qgis_stable python3 -m unittest -v tests.test_workbench_gui tests.test_workbench_persistence` (note: `tests.test_workbench_imports` in PLANNING.md doesn't exist — use `test_workbench_gui`).

---

## 7. Appendix C — C++ / CUDA Audit (added 2026-07-19)

**Scope:** `cpp/src/` (10k+ LoC across .cu/.cuh/.cpp/.hpp/.h), `CMakeLists.txt`, `docs/cpp/`, `pybind11` binding surface vs Python callers.
**Verifies:** the cross-language impact of Python P0 #5 (re-RCMK on baked BLOB), the QGIS4 migration plan, and `docs/cpp/` accuracy.

### 7.1 Top C++ findings (ranked)

| # | File:line | Issue | Severity |
|---|---|---|---|
| 1 | `cpp/src/swe2d_mesh.cpp:487-490` | RCMK level sort uses unstable `std::sort` with `degree[a] < degree[b]` only → not strictly idempotent (see §7.2 verdict) | **High** |
| 2 | `CMakeLists.txt:13-16` | `if(BACKWATER_USE_CUDA)` block executes before `option()` at line 52 → dead on first configure | High |
| 3 | `docs/cpp/ARCHITECTURE.md:23,35` + `docs/DEVELOPER_GUIDE.md:501-518` | References `hybrid_mesh_bindings.cpp` / target `hydra_hybridmesh` which don't exist; `hydra_overlay_backend` should be `hydra_overlay` | High |
| 4 | `cpp/src/swe2d_gpu.cu:1204-1222` | `cell_ring2_ids[k]` dereferenced (`cell_h[j]`, `cell_zb[j]`) without `j < n_cells` bounds check | Medium |
| 5 | `cpp/src/swe2d_bindings.cpp:911-935` | `read_snapshots` returns pinned-host numpy views; Python never frees the device ring buffer between reads, output array grows monotonically | Medium |
| 6 | `docs/cpp/GPU_KERNEL_STRATEGY.md` | Lists kernel names that don't exist (`swe2d_gpu_apply_bc_kernel`, `_bed_slope_kernel`, `_diag_kernel`, `_friction_kernel`, `_gradient_2ring_kernel`); actual names: `swe2d_implicit_friction_kernel`, `swe2d_lsq_gradient_kernel`, `swe2d_cfl_kernel` | Medium |
| 7 | `cpp/src/pipe1d.cu:4611-4613` | `swe2d_pipe1d_junction_overflow_kernel` launch missing the `CUDA_CHECK(cudaGetLastError());` that every other launch in the file has | Low |
| 8 | `cpp/src/swe2d_gpu.cu:9440-9457` | `swe2d_gpu_readback_pipe_face_diag` is dead (no binding, no callers); also non-obvious operator-precedence | Low |
| 9 | `cpp/src/swe2d_gpu.cu` | All launches + H2D/D2H copies run on `dev->d_stream` (one stream, `swe2d_gpu.cu:5793`). Pipe1D shares the SWE2D stream — they could overlap | Low (perf) |
| 10 | `cpp/src/swe2d_bindings.cpp` | 28 pybind11 exports have no Python caller (full list in agent log: `swe2d_contract_*`, `swe2d_gpu_compute_bridge_coupling_sources`, `swe2d_gpu_readback_*` family, `swe2d_pipe1d_*` batch, etc.) | Low |

### 7.2 RCMK idempotency verdict — confirms Python P0 #5

Function: `swe2d_renumber_cells_for_gpu` (`cpp/src/swe2d_mesh.cpp:408-574`). Called from `swe2d_build_mesh_poly` at line 287.

- **Deterministic for same input on same platform** (no persistent state; only `std::deque<int32_t>` BFS, `std::sort`).
- **NOT strictly idempotent** because:
  - Unstable `std::sort` at line 487 on `degree[a] < degree[b]` only — equal-degree cells break by input order, and the input order differs between RCMK runs after cells are renumbered.
  - Root selection at lines 446-453 picks the **lowest-index** min-degree cell — after the first RCMK renumbers cells, the second RCMK starts from a different root by old index.

**Severity of Python P0 #5: HIGH, REAL.** `build_mesh_from_baked` (`backend.py:434-490`) exists precisely to skip the second RCMK but has zero call sites; the CLI path goes `query_mesh_from_gpkg` → `shared_build_mesh` → `backend.build_mesh` → `swe2d_build_mesh_poly` → RCMK applied a second time on already-RCMK-ordered mesh. On a single dev machine `cp2` usually equals `cp1` (`apply_cell_permutation` masks it), but:
- It will bite on different `libstdc++`/libc++/MSVC STL, on meshes with many degree-tied interior cells, and any change to the RCMK algorithm.
- After masking, the Python backend `INSERT-OR-REPLACE`s a different baked BLOB back to GPKG, silently changing physical results on the **next** run-from-baked (max-flood extents, hydrograph timing, structure flows).

Recommended fix: **Python side — wire `build_mesh_from_baked`** into `backend_initializer.build_and_initialize` (`backend_initializer.py:111`) whenever the mesh came from a deserialized BLOB. **C++ defense-in-depth — replace the level sort with** `std::stable_sort` keyed on `(degree[a], a)`.

### 7.3 Build / artifact consistency

- All targets (`hydra_swe2d`, `hydra_meshing_native`, `hydra_overlay`) have their declared sources on disk.
- Compile define is **`HYDRA_HAS_CUDA`** (`CMakeLists.txt:123,199`), NOT `HYDRA_USE_CUDA` (which doesn't exist). The Python audit's "BACKWATER_USE_CUDA vs HYDRA_USE_CUDA" wording was imprecise — the canonical CMake option is `BACKWATER_USE_CUDA`; the canonical compile define is `HYDRA_HAS_CUDA`. README/USER_GUIDE/DEVELOPER_GUIDE variously call it `USE_CUDA` (wrong).
- Missing from `docs/cpp/ARCHITECTURE.md` module table: `pipe1d.cu`, `pipe1d.cuh`, `swe2d_reconstruct.cu`, `swe2d_xsect_constants.h`.
- Existing but undocumented in architecture guide: `CULVERT_DIAG` option (`CMakeLists.txt:53`).
- No current SWiG bindings — only 3 pybind11 modules: `swe2d_bindings.cpp:709`, `meshing_native_bindings.cpp:468`, `overlay_backend.cpp:600`.

### 7.4 Memory management (mostly clean)

- `SWE2DBackend.destroy()` → `swe2d_destroy` → `swe2d_gpu_destroy` → `swe2d_gpu_free_snapshot_buf` (`swe2d_gpu.cu:11149`). The Python audit's worry about `free_snapshot_buf` not being called is **incorrect** — it's invoked transitively.
- No per-step `cudaMalloc`/`cudaMallocManaged` on hot paths. Ring buffers grow geometrically.
- All ~50 `cudaMallocHost`/`cudaFreeHost` paths verified lifecycle-correct.

### 7.5 Hot-path numerical / algorithmic risks

- **Shared-memory races**: none. All `__syncthreads()` reductions verified with `awk` sweep across `*.cu` — no divergent `return` between sync and use.
- **Missing `cudaGetLastError()`**: `pipe1d.cu:4613` (`swe2d_pipe1d_junction_overflow_kernel`); other launches in same file follow the pattern. Cosmetic, non-silent in production.
- **Host/device pointers**: clean. `swe2d_gpu.cu:10677-10680` synchronous + capsule-deleter wrapping confirmed.
- **Boundary handling**: `swe2d_gpu.cu:1204-1222` unguarded `cell_ring2_ids[k]` read is the only one of concern (Medium, depends on builder correctness).
- **Per-step allocations**: none on production path; debug path (`BACKWATER_SWE2D_DEBUG_GPU_EDGE_FLUX`) is conditional and cleanly freed.
- **Live threading**: `swe2d_gpu_accumulate_external_source` (`swe2d_gpu.cu:3046`) + `pipe1d.cu` step could overlap on a worker stream — pure perf opportunity.

### 7.6 docs/cpp/ accuracy (full audit)

Major staleness vs source, all verified:

- **Spatial schemes** (`ARCHITECTURE.md:108-110`, `GPU_KERNEL_STRATEGY.md:27-68`, `SWE2D_GPU_ARCHITECTURE_REPORT.md:11,72-85,503`) — describe pre-migration 0-5 with WENO5 at 6; actual enum is **0-8** (`swe2d_solver.hpp:12-22`: 5=BJ, 6=WENO3, 7=WENO5, 8=MP5). `SWE2D_GPU_ARCHITECTURE_REPORT.md` and `docs/cpp` describe a pre-migration enum; `SOLVER_ORDER_AND_STENCIL.md` describes the post-migration one.
- **Temporal schemes** (`SWE2D_GPU_ARCHITECTURE_REPORT.md:11,87-96`) — lists "RK2/RK4/RK5/RK6", no RK6 exists, **omits Euler and SSPRK3** (`swe2d_solver.cpp:46-48,297-421`: 1=Euler, 2=SSPRK2, 3=SSPRK3, 4=RK4, 5=graph-safe RK4, 6=RK5).
- **WENO3 math** (`ADVANCED_SPATIAL_SCHEMES.md:142-176`) — describes 3-sub-stencil 2D LSQ WENO3 with ideal weights `(0.1, 0.6, 0.3)`. Actual implementation `swe2d_reconstruct.cu:61-101,209-343` is a projected 1D two-stencil reconstruction with weights `(1/3, 2/3)`.
- **WENO5** (`SOLVER_ORDER_AND_STENCIL.md:41`, `ADVANCED_SPATIAL_SCHEMES.md:291-300,374-383`) — describes canonical 5-sub-stencil WENO5. Actual `swe2d_gpu.cu:2136-2204` is a 3-candidate (midpoint + 2-ring LSQ ±TV) blend with weights `(0.10, 0.30, 0.60)`. The architecture report's "≈3rd-order" characterization is closer to reality than the solver-order guide's "true 5th-order" wording.
- **MP5** (`ADVANCED_SPATIAL_SCHEMES.md:244-277`) — describes Suresh-Huynh 4-case mapped limiter; actual `swe2d_reconstruct.cu:550-633` is MP5-inspired but adds a gradient-difference L/R split not in the paper.
- **Build options**: `SWE2D_STATE_FP32` (README:49, USER_GUIDE:110) is removed (CMakeLists.txt:55-60, `swe2d_gpu.cuh:10-13` pins `State=double`); `BACKWATER_SWE2D_MIXED_PRECISION` (MIXED_PRECISION_GPU_PLAN.md) was never implemented — selective FP32 inside the flux kernel is documented as live but the kernel uses doubles (`swe2d_gpu.cu:1931-2330`).
- **CPU fallback**: false — claimed by `GPU_KERNEL_STRATEGY.md:5`, `SWE2D_GPU_ARCHITECTURE_REPORT.md:3-5`, `ARCHITECTURE.md:34`. Source: `swe2d_solver.hpp:3-5` "Always uses CUDA GPU path — CPU/OpenMP fallback has been removed." `swe2d_solver.cpp:442-443` throws without a GPU.
- **Coupling SoA / structures SoA**: `COUPLING_KERNELS.md:65-99` field tables don't match the actual structs (`coupling.py:213-344`); culvert face-flux docs describe a direct edge-flux path that doesn't exist (the binding uses donor/receiver cell + face normal + width + invert).
- **`swe2d_gpu_drainage_step`** (`COUPLING_KERNELS.md:56-63`, `DEVELOPER_GUIDE.md:480-489`, `SWE2D_GPU_ARCHITECTURE_REPORT.md:348-355`) — **phantom**. Current API splits: `swe2d_gpu_apply_coupling_drainage` (swe2d_bindings.cpp:1157-1167) + `swe2d_pipe1d_step` (swe2d_bindings.cpp:1934-1988).
- **`swe2d_gpu_compute_coupling_sources`** — pybind signature (`swe2d_bindings.cpp:1657-1697`) accepts inlet arrays but never uses them; documented behavior (rainfall, infiltration, drainage combined in one kernel) is wrong.
- **Source sign convention** — `SWE2D_GPU_ARCHITECTURE_REPORT.md:223-225` says +Q upstream / -Q downstream; actual `swe2d_coupling_structure_source_kernel` (`swe2d_gpu.cu:3096-3100`) does the opposite.
- **HDS-5 helper API** (`CULVERT_HDS5.md:58-71`) — names listed don't exist; current names: `bw2d_circular_area`, `bw2d_pipe_manning_capacity_full`, `bw2d_orifice_q`, `swe2d_culvert_inlet_controlled_flow_cfs_cuda`, `swe2d_culvert_outlet_control_flow_cms_cuda`.
- **Units claim** — `SWE2D_GPU_ARCHITECTURE_REPORT.md:234-246,481-489` says all structure dims are packed in feet and returns SI; `ARCHITECTURE.md:59-67` correctly says model units. Source confirms model-unit packing (`coupling.py:762-770,818-823,1003-1005,958-964`); `ARCHITECTURE.md` is right.
- **`compute_structure_flows` test reference** (`CULVERT_HDS5.md:73-77`) — cites `test_swe2d_drainage_structures.py` which doesn't exist; actual is `tests/test_culvert_hds5_validation.py:286-295`.
- **Turbulence/friction enums** (`ARCHITECTURE.md:110-111`) — `SMAGORINSKY`/`K_EPSILON`/`K_OMEGA_SST` not in C++ (`swe2d_solver.hpp:24-30` has only `NONE` and `MANNING`); broader values live in Python enum only (`extension_models.py:45-58`) and `backend.py:1035-1038` itself labels them "scaffolded".
- **Same-document contradiction** (`SOLVER_ORDER_AND_STENCIL.md:28` says no scheme expands beyond 1-ring; line 41 says WENO5 uses 2-ring) — `swe2d_mesh.hpp:78-82` + `swe2d_gpu.cuh:89-95` confirm 2-ring CSR data exists.

### 7.7 Tracked Doxygen output (public-release hazard)

- `docs/cpp/api/` (436 files, 3.8 MB) **AND** `cpp/api/` (55 files) are both git-tracked. `docs/INDEX.md:68-75` says generated docs "land in `_build/` and are not tracked" — **false**. `.gitignore` and `.publicsync-ignore` ignore neither.
- The tracked `docs/cpp/api/` still lists `hybrid_mesh_bindings.cpp`, `generate_hybrid_mesh()`, `PYBIND11_MODULE(hydra_hybridmesh, m)`, the pre-migration scheme enum (6=WENO5), and a removed `swe2d_step_cpu`. It cites dates of June 22, 2026 vs current source has later pipe1d refactors.
- Doxygen `api/index.html:96` contains a local absolute path anchored from `/home/aaron/QGIS_Plugins_dev/public-repo-hydra2dgpu/README` — machine-specific metadata.
- Resolution: add both `docs/cpp/api/` and `cpp/api/` to `.publicsync-ignore` (or `.gitignore`), regenerate fresh on docs builds, and stop tracking the output.

### 7.8 QGIS4 / wheel-build readiness (cross-checked with `reference/qgis-repo-and-qgis4-migration.md`)

- **Distribution vs import name**: plan proposes distribution `hydra-swe2d`, import `hydra_swe2d`. Current source: distribution `hydra2dgpu` (`pyproject.toml:5-9`), target `hydra_swe2d` (`CMakeLists.txt:2,178`), pybind module default `hydra_swe2d` (`swe2d_bindings.cpp:705-709`), Python imports as `hydra_swe2d` (`backend.py:189`). Plan does not change the import name; clarify both names in public docs.
- **Python floor**: `pyproject.toml:9` `>=3.12`; README says 3.12+; plan says `>=3.10,<3.13`; CI workflow `test.yml:14-17` already runs 3.10 + 3.12 → conflict.
- **CUDA toolkit**: README says 11.x/12.x; `build-release.yml:28-33,107-112` ships CUDA 12.4; plan calls for CUDA 13.0; CMake supports both families (`CMakeLists.txt:92-100`) — sm_75 floor under CUDA 13.
- **scikit-build-core vs direct CMake**: `docs/cpp/ARCHITECTURE.md:28-30` describes direct CMake + `pybind11_add_module`; `pyproject.toml:1-3` uses setuptools (does NOT invoke CMake); no `install(TARGETS ...)` rules exist for any pybind module. Plan is incompatible with current packaging until CMake `install` rules + a real package layout are added.
- **`BACKWATER_USE_CUDA=OFF` wheel**: plan suggests this at line 414 — but `swe2d_solver.cpp:442-443` refuses to step without a GPU device, so a CUDA-less wheel is not a usable 2D solver.
- **Unimplemented "ready" features**: `MIXED_PRECISION_GPU_PLAN.md` (selective FP32) and `LTS_implementation_plan.md` (5 LTS kernels + reductions + step loop) — neither is implemented; treat as plans only, don't describe as current capability.

### 7.9 C++ audit top-priority fixes

1. Replace `std::sort` in `swe2d_mesh.cpp:487` with `std::stable_sort` keyed on `(degree, idx)` (defense-in-depth for P0 #5).
2. Move the LTO block at `CMakeLists.txt:13-16` below the `option()` declaration at line 52 (Python audit #11 already flagged; confirmed C++ side).
3. Remove `hybrid_mesh_bindings.cpp` / `hydra_hybridmesh` references from `ARCHITECTURE.md`, `DEVELOPER_GUIDE.md:501-518`, generated `docs/cpp/api/`. Correct `hydra_overlay_backend` → `hydra_overlay`.
4. Rewrite `ARCHITECTURE.md`, `GPU_KERNEL_STRATEGY.md`, `SWE2D_GPU_ARCHITECTURE_REPORT.md`, `ADVANCED_SPATIAL_SCHEMES.md`, `SOLVER_ORDER_AND_STENCIL.md`, `COUPLING_KERNELS.md`, `CULVERT_HDS5.md` from the actual source (scheme numbering, kernel names, WENO3/WENO5/MP5 actual math, SoA fields, source signs).
5. Add bounds check `j < n_cells` at `swe2d_gpu.cu:1204-1222`.
6. Add `CUDA_CHECK(cudaGetLastError())` after `swe2d_pipe1d_junction_overflow_kernel` launch at `pipe1d.cu:4613`.
7. Untrack `docs/cpp/api/` and `cpp/api/` (or at minimum add to `.publicsync-ignore`); regenerate fresh from current source if kept tracked.
8. Remove retired flags: `SWE2D_STATE_FP32` (README:49, USER_GUIDE:110) and `USE_CUDA` (DEVELOPER_GUIDE:514 → `BACKWATER_USE_CUDA`).
9. Delete dead C++ functions: `swe2d_gpu_readback_pipe_face_diag` (`swe2d_gpu.cu:9440`).
10. Decision package: name (hydra2dgpu vs hydra-swe2d / hydra_swe2d), Python floor (>=3.12 vs >=3.10,<3.13), CUDA toolkit (12.4 vs 13.0), CMake packaging. These must be settled before the migration branch.

### 7.10 Pipe1D architectural refactor — 2026-07-19 decision

The pipe1D solver's hybrid "cells + network-nodes + virtual-nodes" discretization is being refactored to a pure face-indexed FVM mesh. **Decision recorded 2026-07-19** after this audit revealed:

- The `d_vnode_*` array family is a face-state workaround indexed as if it were a node, creating dual bookkeeping between `d_node_net_q` and the cell-side flux sum (root cause of F1–F15 mass-conservation findings in `docs/PIPE1D_AUDIT_2026-07-17.md`).
- The relaxation-law boundary flux at the cell↔network-node interface is not a finite-volume flux, leaving residual truncation error that the F1 fix alone cannot eliminate (the −25 % drift in `test_closed_system_conserves_mass_diffusion_wave` over 200 steps).
- Three separate face-flux kernels (`swe2d_pipe1d_flux_kernel`, `swe2d_pipe_face_flux_kernel`, `swe2d_culvert_face_flux_kernel`) solve essentially the same Riemann problem with different conventions — exactly the CFD wheel being reinvented.

**Locked-in design** (see `docs/pipe1d_face_indexed_refactor_plan.md` for the full plan):

1. **Mesh representation**: manholes, junction boxes, and inlets become full FV cells with `(A, Q, invert, depth, top_width, surface_area, crown, rim)`. Pipe-ends are direct face couplings to 2D SWE2D cells (no node-side storage at the pipe-end).
2. **Coupling**: ONE unified face-flux kernel handles every face class (INTERIOR / OUTFALL_BC / INLET_BC / SURFACE_2D / CULVERT). The three current face kernels collapse into one. `d_pipe_end_q_2d` buffer and `swe2d_fold_pipe_end_q_to_source_kernel` retire; the unified kernel writes `d_ext_struct_flux_h[u,v]` directly.
3. **F1–F15 scope**: drop all of them. Mass conservation is automatic by FV construction. F6 (wave speed `sqrt(g·A/T)`) lands in the new HLLC eigenvalue computation. F8 (pipe-end datum) lands in how the new mesh sets the SURFACE_2D face's neighbor invert. The `docs/PIPE1D_AUDIT_2026-07-17.md` and `docs/archive/plans/2026-07-17-pipe1d-mass-conservation-fixes.md` stay as historical record only.

**Implications for the rest of this audit:**

- The §3.4 "Mass conservation regression" and §3.4 `batch_worker.py` parallel-mode hazard are unrelated to the pipe1D refactor and remain valid bug findings.
- The per-step `cudaMalloc`/`cudaFree` in `swe2d_pipe1d_step:3176-3188` (Phase 2.3 of the refactor) and the `cudaDeviceSynchronize` at `swe2d_gpu.cu:9414` (Phase 2.2 of the refactor) will be fixed as part of the refactor. They do not need to be addressed separately.
- The MVP architecture violations (§2), duplicate code (§1), documentation gaps (§4), and quick improvements (§5) of this audit are unaffected by the pipe1D refactor and remain valid.
- Pre-release is no longer a schedule constraint (the project owner has confirmed this is single-owner work with no external deadline). The pipe1D refactor is being done before the release; everything else in this audit can wait until after.

**Status as of 2026-07-19:** Phase-0 dependency map complete (see `docs/pipe1d_face_indexed_refactor_plan.md` §A). Phase-1 test scaffolding next.
