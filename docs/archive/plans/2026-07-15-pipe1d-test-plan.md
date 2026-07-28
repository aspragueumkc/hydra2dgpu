---
type: plan
status: complete
created: 2026-07-15
completed: 2026-07-25
---

# Pipe1D Solver Rewrite — Parallel-Track Test Plan

> **Parent plan:** `docs/archive/plans/2026-07-15-pipe1d-solver-rewrite.md`, Step 21  
> **Solver spec:** `docs/archive/specs/2026-07-15-pipe1d-solver-rewrite-spec.md`  
> **Purpose:** identify which existing tests remain valid, which require revision after the rewrite, and which missing tests must be added in a later testing phase. This document does not execute or modify tests.

## 1. Test inventory

The inventory uses the function names returned by `rg -n "def test_"`. The current checkout contains 14 methods in `tests/test_swe2d_pipe1d.py` and 18 in `tests/test_swe2d_gpu_drainage_network.py`; those current names, rather than earlier approximate counts, are authoritative for this plan.

Cross-reference rows are method-filtered: they include tests whose fixtures or assertions exercise Pipe1D geometry, flow, surcharge, drainage exchange, readback, persistence, or the unchanged Python coupling/configuration contract. Pure 2D, structure-only, generic export, and unrelated UI methods from the broad `test_coupling_*.py` and `test_results_*.py` globs are outside this inventory.

### 1.1 Primary Pipe1D tests

| Test file | Test method | What it asserts (1-2 lines) |
|---|---|---|
| `tests/test_swe2d_pipe1d.py` | `test_build_mesh_single_link` | A one-link mesh builds without crashing and readback returns two node depths plus one cell area and discharge. |
| `tests/test_swe2d_pipe1d.py` | `test_subcell_index_increases_downstream` | A 100 m link split into ten cells has ordered `cell_sub_idx` values and monotonically decreasing cell inverts from upstream to downstream. |
| `tests/test_swe2d_pipe1d.py` | `test_diffusion_wave_updates_area` | One diffusion-wave step changes cell area under a head difference and leaves finite node depths. |
| `tests/test_swe2d_pipe1d.py` | `test_fully_dynamic_updates_area_and_q` | Fully dynamic mode changes cell area and produces a finite discharge under a pressure gradient. |
| `tests/test_swe2d_pipe1d.py` | `test_dry_pipe_no_change` | Zero-depth nodes retain the SWMM-style minimum area floor and produce negligible flow. |
| `tests/test_swe2d_pipe1d.py` | `test_dry_pipe_wets_from_upstream_node` | A floor-area dry pipe gains area and positive downstream discharge after its upstream node is wetted. |
| `tests/test_swe2d_pipe1d.py` | `test_substeps_produce_smaller_area_than_single` | Both one and four diffusion-wave substeps cause outflow, represented by an area below full area. |
| `tests/test_swe2d_pipe1d.py` | `test_upload_node_depth_changes_area` | Changing uploaded node heads changes the area reached after a diffusion-wave step. |
| `tests/test_swe2d_pipe1d.py` | `test_rectangular_link_diffusion` | Rectangular full area equals width times height and area/discharge remain finite after stepping. |
| `tests/test_swe2d_pipe1d.py` | `test_elliptical_link_diffusion` | Elliptical full area equals `pi*(w/2)*(h/2)` and area/discharge remain finite after stepping. |
| `tests/test_swe2d_pipe1d.py` | `test_box_shape_without_explicit_shape_arrays` | A zero-diameter fallback and a later explicit elliptical shape both avoid crashes and non-finite state. |
| `tests/test_swe2d_pipe1d.py` | `test_init_area_from_depth` | Zero depth initializes to the area floor, while circular depth equal to diameter initializes approximately to full area. |
| `tests/test_swe2d_pipe1d.py` | `test_fully_dynamic_convective_term_affects_flow` | Two cases with equal head difference and midpoint area but different end-area gradients produce different finite discharges. |
| `tests/test_swe2d_pipe1d.py` | `test_fully_dynamic_mass_conservation_with_and_without_sub_cells` | Total node-plus-pipe volume is conserved over ten fully dynamic steps for one cell and a two-cell split. |
| `tests/test_swe2d_pipe1d_surcharge.py` | `test_surcharge_node_depth_uncapped` | Node depth can exceed `node_max_depth`; the current test also expects cell area not to exceed circular full area. |
| `tests/test_swe2d_pipe1d_surcharge.py` | `test_full_cell_flux_stability` | A surcharged fully dynamic step leaves cell area, discharge, and node depth finite. |
| `tests/test_swe2d_pipe1d_surcharge.py` | `test_mass_conservation_surcharge` | A closed, initially surcharged one-pipe system conserves node-plus-pipe volume over five steps. |
| `tests/test_swe2d_pipe1d_surcharge.py` | `test_two_pipe_surcharge_propagation` | A two-link, three-node surcharged network remains finite; current assertions cap both cell areas at full area. |

### 1.2 GPU drainage-network tests

| Test file | Test method | What it asserts (1-2 lines) |
|---|---|---|
| `tests/test_swe2d_gpu_drainage_network.py` | `test_inlet_grate_weir` | Low-head grate capture matches the HEC-22 weir equation. |
| `tests/test_swe2d_gpu_drainage_network.py` | `test_inlet_grate_orifice` | High-head grate capture matches the orifice equation. |
| `tests/test_swe2d_gpu_drainage_network.py` | `test_inlet_grate_transition` | At the grate transition head, computed capture agrees with the weir result within 1%. |
| `tests/test_swe2d_gpu_drainage_network.py` | `test_inlet_grate_opening_ratio` | Different grate opening fractions produce the expected individual flows and flow ratio. |
| `tests/test_swe2d_gpu_drainage_network.py` | `test_inlet_curb_weir` | Low-head curb capture matches its HEC-22 weir equation. |
| `tests/test_swe2d_gpu_drainage_network.py` | `test_inlet_curb_orifice` | High-head curb capture matches its orifice equation. |
| `tests/test_swe2d_gpu_drainage_network.py` | `test_inlet_slotted_weir` | Low-head slotted-inlet capture matches its weir equation. |
| `tests/test_swe2d_gpu_drainage_network.py` | `test_inlet_slotted_orifice` | High-head slotted-inlet capture matches its orifice equation. |
| `tests/test_swe2d_gpu_drainage_network.py` | `test_inlet_combo` | A combination inlet equals the sum of grate and remaining curb-sweep capture. |
| `tests/test_swe2d_gpu_drainage_network.py` | `test_inlet_relief` | A higher pipe node head drives relief flow from the node to the dry surface cell at the expected rate. |
| `tests/test_swe2d_gpu_drainage_network.py` | `test_inlet_availability_limiter` | Remaining node storage limits inlet capture to available volume divided by the coupling timestep. |
| `tests/test_swe2d_gpu_drainage_network.py` | `test_drainage_only_no_structures` | Drainage-only coupling with no hydraulic structures computes positive capture without crashing. |
| `tests/test_swe2d_gpu_drainage_network.py` | `test_pipe_end_upload_allocates_all_arrays` | Non-empty pipe-end parameter arrays upload successfully and can be used in a coupling step. |
| `tests/test_swe2d_gpu_drainage_network.py` | `test_pipe_end_moves_water_downhill` | Higher WSE in one surface cell moves water through a pipe to the lower cell and approximately conserves exchanged mass. |
| `tests/test_swe2d_gpu_drainage_network.py` | `test_dry_surface_cells_make_pipe_nodes_dry` | Applying pipe-end BCs to two dry surface cells resets both connected pipe-node depths to zero. |
| `tests/test_swe2d_gpu_drainage_network.py` | `test_readback_coupling_state_returns_cell_arrays` | Coupling readback exposes typed velocity, depth, flow, head, and owner-link arrays for pipe cells. |
| `tests/test_swe2d_gpu_drainage_network.py` | `test_append_pipe_cell_snapshot_stores_geometry` | Pipe-cell snapshots retain first-write geometry, defaults, sibling propagation, times, and values. |
| `tests/test_swe2d_gpu_drainage_network.py` | `test_append_coupling_snapshot_routes_geometry_to_pipe_cell` | A `drainage_cell` coupling row forwards geometry into pipe-cell live storage. |

### 1.3 Direct solver and local-loss cross-references

| Test file | Test method | What it asserts (1-2 lines) |
|---|---|---|
| `tests/test_pipe1d_accumulation.py` | `test_normal_forward_reduces` | A positive-flow HEC-22 loss reduces discharge without reaching or crossing zero. |
| `tests/test_pipe1d_accumulation.py` | `test_normal_forward_old_vs_new` | The old and clamped formulas agree for a non-loss-dominated positive flow. |
| `tests/test_pipe1d_accumulation.py` | `test_loss_dominated_forward_does_not_reverse` | A very large positive-flow loss clamps discharge to a non-negative value. |
| `tests/test_pipe1d_accumulation.py` | `test_loss_dominated_old_reverses` | The historical unclamped formula reverses a loss-dominated positive flow, documenting the old defect. |
| `tests/test_pipe1d_accumulation.py` | `test_dry_pipe_skips_loss` | Zero area bypasses the local-loss correction. |
| `tests/test_pipe1d_accumulation.py` | `test_zero_k_skips_loss` | A zero loss coefficient leaves discharge unchanged. |
| `tests/test_pipe1d_accumulation.py` | `test_normal_reverse_reduces` | A negative-flow HEC-22 loss reduces magnitude without changing sign. |
| `tests/test_pipe1d_accumulation.py` | `test_normal_reverse_old_vs_new` | Old and clamped formulas agree for a non-loss-dominated reverse flow. |
| `tests/test_pipe1d_accumulation.py` | `test_loss_dominated_reverse_does_not_reverse` | A very large reverse-flow loss clamps discharge to a non-positive value. |
| `tests/test_pipe1d_accumulation.py` | `test_loss_dominated_old_reverses` | The historical formula reverses a loss-dominated negative flow, documenting the old defect. |
| `tests/test_pipe1d_accumulation.py` | `test_zero_q` | HEC-22 loss leaves zero discharge at zero. |
| `tests/test_pipe1d_accumulation.py` | `test_zero_q_dry` | Zero discharge and zero area remain zero. |
| `tests/test_pipe1d_accumulation.py` | `test_zero_q_old` | The historical formula also leaves zero discharge unchanged. |
| `tests/test_pipe1d_accumulation.py` | `test_forward_after_reverse` | A sign sweep never increases magnitude or flips the sign selected for that step. |
| `tests/test_pipe1d_accumulation.py` | `test_tiny_a_range` | A broad area/discharge sweep remains sign-preserving and non-amplifying. |
| `tests/test_pipe1d_vs_swmm.py` | `test_half_pipe_reasonable` | Half-full open-channel flow is positive and within a factor of two of a Manning estimate. |
| `tests/test_pipe1d_vs_swmm.py` | `test_slope_scaling` | Increasing slope by a factor of four increases discharge substantially, approximately following square-root scaling. |
| `tests/test_pipe1d_vs_swmm.py` | `test_nonzero_head_gives_flow` | Zero head difference gives negligible flow, while a nonzero head difference gives positive flow. |
| `tests/test_pipe1d_vs_swmm.py` | `test_pipe1d_finite_above_crown` | Above-crown head with local losses yields finite, positive Pipe1D discharge. |
| `tests/test_pipe1d_vs_swmm.py` | `test_pipe1d_stable_at_large_dt` | Diffusion-wave flow remains finite at `dt=0.25`. |
| `tests/test_pipe1d_vs_swmm.py` | `test_pipe1d_vs_swmm` | A pressurized Pipe1D discharge agrees with a SWMM reference within 5% for the configured head. |
| `tests/test_pipe1d_vs_swmm.py` | `test_swmm_free_outfall_surcharges` | The SWMM reference alone surcharges above crown when inflow exceeds pipe capacity. |
| `tests/test_pipe1d_vs_swmm.py` | `test_entrance_loss_reduces_flow` | Nonzero entrance/exit loss coefficients reduce Pipe1D flow by at least 5%. |
| `tests/test_pipe1d_vs_swmm.py` | `test_reversal_stable` | After changing from forward to strongly reversed heads, all sub-cell discharges remain finite and negative. |

### 1.4 Coupling, readback, persistence, and presentation cross-references

| Test file | Test method | What it asserts (1-2 lines) |
|---|---|---|
| `tests/test_pipe_cell_coupling_output.py` | `test_gpkg_pipe_cell_roundtrip` | Four pipe-cell metrics persist with the expected row and timestep counts. |
| `tests/test_pipe_cell_coupling_output.py` | `test_gpkg_pipe_cell_roundtrip_multiple_links_and_cells` | Multiple links, sub-cells, metrics, and times persist without losing indices or arrays. |
| `tests/test_pipe_cell_coupling_output.py` | `test_gpkg_pipe_cell_geometry_roundtrip` | Cell invert, width, height, and shape type round-trip through the GeoPackage. |
| `tests/test_pipe_cell_coupling_output.py` | `test_gpkg_pipe_cell_legacy_fallback` | Legacy pipe-cell rows load with default geometry. |
| `tests/test_pipe_cell_coupling_output.py` | `test_build_pipe_cell_items_includes_geometry` | Building persistence items carries live pipe-cell geometry fields through. |
| `tests/test_gpkg_persistence.py` | `test_pipe_cell_ts_legacy_schema_migration` | Writing pipe-cell geometry to a legacy seven-column table migrates and round-trips the schema. |
| `tests/test_swe2d_gpu_coupling_integration.py` | `test_drainage_and_structures_coupling` | The full drainage-plus-structures coupling loop remains GPU-active, finite, and non-negative. |
| `tests/test_swe2d_gpu_coupling_integration.py` | `test_drainage_only_coupling` | Drainage-only coupling repeatedly applies without the former missing-WSE-buffer crash. |
| `tests/test_swe2d_gpu_coupling_integration.py` | `test_structures_only_coupling` | The structures-only branch remains finite when no drainage network exists. |
| `tests/test_swe2d_gpu_coupling_integration.py` | `test_new_schemes_through_coupling_path` | Spatial schemes 5, 6, and 8 remain finite and non-negative through the full coupling path. |
| `tests/test_swe2d_gpu_coupling_integration.py` | `test_rebuild_coupling_mid_simulation` | Coupling can be destroyed and rebuilt mid-run without invalid state or non-finite depth. |
| `tests/test_coupling_diagnostics_readback.py` | `test_cli_run_persists_drainage_link_flow_rows` | A real CLI run writes non-empty `drainage_link/flow` time/value rows. |
| `tests/test_coupling_diagnostics_readback.py` | `test_cli_run_writes_expected_coupling_schema` | CLI output contains drainage-link flow and drainage-node depth components. |
| `tests/test_coupling_integration.py` | `test_apply_external_sources_drainage_only_uses_coupled_source` | A drainage-only coupled source array is passed unchanged to native source injection. |
| `tests/test_coupling_integration.py` | `test_pipe_cell_snapshot_at_t0_is_zero` | Mocked per-cell depth rows sampled at time zero contain no spurious priming. |
| `tests/test_coupling_integration.py` | `test_real_pipe1d_readback_at_t0_is_zero` | Real GPU node and pipe-cell readback starts at zero rather than exposing uninitialized memory. |
| `tests/test_coupling_integration.py` | `test_drainage_exchange_upload_runs_when_mesh_built_eagerly` | Exchange parameters still upload on the first step when the Pipe1D mesh was built eagerly. |
| `tests/test_coupling_integration.py` | `test_readback_returns_zeros_on_size_mismatch` | C++ readback returns zero-filled node/cell arrays when requested counts do not match device state. |
| `tests/test_coupling_setup.py` | `test_build_coupling_controller_with_pipe_network` | A minimal pipe-network configuration creates a coupling controller. |
| `tests/test_coupling_rain_matrix.py` | `test_full_matrix` | The source/coupling matrix covers drainage on/off across native and callback paths without invalid state. |
| `tests/test_coupling_snap_indexing.py` | `test_multi_row_per_snap` | Multiple drainage-node rows per snapshot retain aligned times and values. |
| `tests/test_coupling_snap_indexing.py` | `test_single_row_per_snap` | One row per snapshot remains correctly indexed. |
| `tests/test_coupling_snap_indexing.py` | `test_rows_from_lists` | List-based drainage-node snapshots are converted into expected live rows. |
| `tests/test_drainage_inlet_outfall_vs_swmm.py` | `test_inlet_outfall_1_link_depth_matches_swmm` | A one-link junction-to-outfall Pipe1D run produces nontrivial flow and upstream depth within a factor of two of SWMM. |
| `tests/test_engine_class_placement.py` | `test_drainage_engine_in_drainage_network_module` | The drainage coupling engine remains importable from its canonical module. |
| `tests/test_engine_class_placement.py` | `test_structure_engine_in_structures_module` | The structure engine remains importable from its canonical module. |
| `tests/test_engine_class_placement.py` | `test_backward_compat_reexport` | Legacy extension-model imports continue to re-export both engine classes. |
| `tests/test_extensions_config_builders.py` | `test_build_drainage_config_importable_from_extensions` | JSON drainage nodes and links build through the canonical extension API. |
| `tests/test_extensions_config_builders.py` | `test_build_structures_config_importable_from_extensions` | Structure JSON still builds through the canonical extension API. |
| `tests/test_extensions_config_builders.py` | `test_build_structures_bare_list_form` | Bare-list structure input auto-enables the configuration. |
| `tests/test_extensions_config_builders.py` | `test_build_drainage_empty_returns_none` | Empty drainage JSON produces no configuration. |
| `tests/test_extensions_config_builders.py` | `test_centroid_helper_is_canonical` | The CLI adapter delegates centroid calculation instead of redefining it. |
| `tests/test_render_network_with_selection.py` | `test_render_network_with_selection_markers` | Selected drainage-link results render a profile, flow annotation, and endpoint markers. |
| `tests/test_render_network_with_selection.py` | `test_render_network_without_selection` | The first drainage link renders by default without an error. |
| `tests/test_render_network_with_selection.py` | `test_render_network_with_empty_selection` | A missing selected link renders the expected no-data message. |
| `tests/test_results_panel.py` | `test_drainage_viewer_import` | The standalone drainage-network viewer remains importable. |
| `tests/test_results_path_wiring.py` | `test_drainage_gpkg_path_is_included` | Drainage replay configuration includes the GeoPackage path and node/link table names. |
| `tests/test_results_timestep_service.py` | `test_returns_records` | Coupling records for a run load from the resolved results table. |
| `tests/test_results_timestep_service.py` | `test_no_table_returns_empty` | Missing coupling-results tables return an empty result. |
| `tests/test_results_timestep_service.py` | `test_open_failure_returns_empty` | A coupling-results database open failure returns an empty result. |

## 2. Bucket assignment

`KEEP` means the existing setup and assertion remain a useful regression contract. `UPDATE` means the test should remain but its fixture, count assumptions, or numerical assertion must be revised for the corrected solver. `NEW-NEEDED` identifies a missing test to add only in a later testing phase.

### 2.1 Existing-test buckets

| Test | File | Bucket | Reason |
|---|---|---|---|
| `test_build_mesh_single_link` | `tests/test_swe2d_pipe1d.py` | UPDATE | Preserve the smoke check but assert the new cell/virtual-node layout and state from §2.1-§2.2. |
| `test_subcell_index_increases_downstream` | `tests/test_swe2d_pipe1d.py` | UPDATE | Ordering remains valid, but it must also verify real virtual-node connectivity under §2.1. |
| `test_diffusion_wave_updates_area` | `tests/test_swe2d_pipe1d.py` | UPDATE | Re-derive the expected direction/bound after the corrected network-boundary flux in §2.13.2. |
| `test_fully_dynamic_updates_area_and_q` | `tests/test_swe2d_pipe1d.py` | UPDATE | Dynamic-wave state now follows the Picard and end-face formulation in §2.3 and §2.14. |
| `test_dry_pipe_no_change` | `tests/test_swe2d_pipe1d.py` | UPDATE | Keep the dry regression, but assert the explicit dry-cell boundary rule and CFL floor in §2.13.2. |
| `test_dry_pipe_wets_from_upstream_node` | `tests/test_swe2d_pipe1d.py` | UPDATE | Wetting must be checked against corrected boundary flux and actual `cell_A` under §2.13.2. |
| `test_substeps_produce_smaller_area_than_single` | `tests/test_swe2d_pipe1d.py` | UPDATE | Replace the weak below-full check with virtual-node conservation and CFL bounds from §2.13.1-§2.13.2. |
| `test_upload_node_depth_changes_area` | `tests/test_swe2d_pipe1d.py` | UPDATE | Use geometric `A(y)` and corrected boundary transfer from §2.4 and §2.13.2. |
| `test_rectangular_link_diffusion` | `tests/test_swe2d_pipe1d.py` | KEEP | The rectangular geometry identity remains required by §2.4. |
| `test_elliptical_link_diffusion` | `tests/test_swe2d_pipe1d.py` | KEEP | The elliptical full-area identity remains required by §2.4. |
| `test_box_shape_without_explicit_shape_arrays` | `tests/test_swe2d_pipe1d.py` | KEEP | Shape fallback stability remains an unchanged geometry robustness contract under §2.4. |
| `test_init_area_from_depth` | `tests/test_swe2d_pipe1d.py` | UPDATE | Add partial-depth geometric area and per-sub-cell initialization assertions for §2.1 and §2.4. |
| `test_fully_dynamic_convective_term_affects_flow` | `tests/test_swe2d_pipe1d.py` | UPDATE | Re-derive the comparison for the rewritten dynamic-wave Picard solve in §2.14. |
| `test_fully_dynamic_mass_conservation_with_and_without_sub_cells` | `tests/test_swe2d_pipe1d.py` | UPDATE | Volume must include every sub-cell and exclude virtual nodes from network mass balance per §2.1 and §2.7. |
| `test_surcharge_node_depth_uncapped` | `tests/test_swe2d_pipe1d_surcharge.py` | UPDATE | The `cell_A <= A_full` assertion conflicts with slot area above crown in §2.5. |
| `test_full_cell_flux_stability` | `tests/test_swe2d_pipe1d_surcharge.py` | UPDATE | Retain finiteness but add slot-area and slot-wave-speed expectations from §2.5. |
| `test_mass_conservation_surcharge` | `tests/test_swe2d_pipe1d_surcharge.py` | UPDATE | Volume accounting must include Preissmann-slot area above crown under §2.5. |
| `test_two_pipe_surcharge_propagation` | `tests/test_swe2d_pipe1d_surcharge.py` | UPDATE | Remove full-area caps and verify virtual-node plus junction behavior under §2.1, §2.5, and §2.10. |
| `test_inlet_grate_weir` | `tests/test_swe2d_gpu_drainage_network.py` | KEEP | Inlet capture is outside the solver rewrite and the coupling surface remains unchanged under §2.15. |
| `test_inlet_grate_orifice` | `tests/test_swe2d_gpu_drainage_network.py` | KEEP | Inlet capture equations are not changed by §2.1-§2.14; retain the §2.15 integration contract. |
| `test_inlet_grate_transition` | `tests/test_swe2d_gpu_drainage_network.py` | KEEP | Grate regime transition is independent of the Pipe1D rewrite; §2.15 preserves coupling. |
| `test_inlet_grate_opening_ratio` | `tests/test_swe2d_gpu_drainage_network.py` | KEEP | Grate opening behavior is unchanged and remains a §2.15 coupling regression. |
| `test_inlet_curb_weir` | `tests/test_swe2d_gpu_drainage_network.py` | KEEP | Curb-inlet capture is outside the corrected solver and preserved by §2.15. |
| `test_inlet_curb_orifice` | `tests/test_swe2d_gpu_drainage_network.py` | KEEP | Curb-inlet capture is outside the corrected solver and preserved by §2.15. |
| `test_inlet_slotted_weir` | `tests/test_swe2d_gpu_drainage_network.py` | KEEP | Surface-inlet slot geometry is distinct from the Pipe1D Preissmann slot in §2.5; keep unchanged. |
| `test_inlet_slotted_orifice` | `tests/test_swe2d_gpu_drainage_network.py` | KEEP | Surface-inlet orifice behavior is unchanged by §2.5 and preserved by §2.15. |
| `test_inlet_combo` | `tests/test_swe2d_gpu_drainage_network.py` | KEEP | Combination-inlet capture remains outside scope under §2.15 and §2.17. |
| `test_inlet_relief` | `tests/test_swe2d_gpu_drainage_network.py` | KEEP | Relief exchange remains a coupling contract; the network graph/API is unchanged under §2.15. |
| `test_inlet_availability_limiter` | `tests/test_swe2d_gpu_drainage_network.py` | KEEP | Node-storage availability remains required; §2.10 changes junction hydraulics, not inlet capacity bookkeeping. |
| `test_drainage_only_no_structures` | `tests/test_swe2d_gpu_drainage_network.py` | KEEP | The no-structures crash regression remains valid through the unchanged §2.15 coupling path. |
| `test_pipe_end_upload_allocates_all_arrays` | `tests/test_swe2d_gpu_drainage_network.py` | KEEP | Allocation completeness is independent of the hydraulic changes in §2.9. |
| `test_pipe_end_moves_water_downhill` | `tests/test_swe2d_gpu_drainage_network.py` | UPDATE | Expected exchange should use geometric `A(y)` and the new pipe-end BC in §2.4 and §2.9. |
| `test_dry_surface_cells_make_pipe_nodes_dry` | `tests/test_swe2d_gpu_drainage_network.py` | UPDATE | Confirm the dry boundary rule after the §2.9 and §2.13.2 boundary rewrite. |
| `test_readback_coupling_state_returns_cell_arrays` | `tests/test_swe2d_gpu_drainage_network.py` | UPDATE | Mock counts and expected arrays must represent all §2.1 sub-cells and new §2.2 state. |
| `test_append_pipe_cell_snapshot_stores_geometry` | `tests/test_swe2d_gpu_drainage_network.py` | KEEP | Persistence keys remain per link/sub-cell because §2.15 leaves the Python graph contract unchanged. |
| `test_append_coupling_snapshot_routes_geometry_to_pipe_cell` | `tests/test_swe2d_gpu_drainage_network.py` | KEEP | Routing of plain geometry fields is unchanged by §2.1-§2.14. |
| `test_normal_forward_reduces` | `tests/test_pipe1d_accumulation.py` | KEEP | End-face local loss must still reduce magnitude without reversal under §2.12. |
| `test_normal_forward_old_vs_new` | `tests/test_pipe1d_accumulation.py` | UPDATE | Replace historical-formula equivalence with an assertion against the §2.12 end-face equation. |
| `test_loss_dominated_forward_does_not_reverse` | `tests/test_pipe1d_accumulation.py` | KEEP | Non-reversal remains required when §2.12 local loss exceeds momentum. |
| `test_loss_dominated_old_reverses` | `tests/test_pipe1d_accumulation.py` | UPDATE | The old-formula demonstration should become a production-facing §2.12 regression. |
| `test_dry_pipe_skips_loss` | `tests/test_pipe1d_accumulation.py` | KEEP | Dry-face loss bypass remains required by §2.12 and §2.13.2. |
| `test_zero_k_skips_loss` | `tests/test_pipe1d_accumulation.py` | KEEP | Zero coefficient must remain a no-op under §2.12. |
| `test_normal_reverse_reduces` | `tests/test_pipe1d_accumulation.py` | KEEP | Reverse-flow loss must reduce magnitude without sign reversal under §2.12. |
| `test_normal_reverse_old_vs_new` | `tests/test_pipe1d_accumulation.py` | UPDATE | Replace old/new equivalence with the corrected reverse end-face formula from §2.12. |
| `test_loss_dominated_reverse_does_not_reverse` | `tests/test_pipe1d_accumulation.py` | KEEP | Reverse-flow non-reversal remains a §2.12 requirement. |
| `test_loss_dominated_old_reverses` | `tests/test_pipe1d_accumulation.py` | UPDATE | Convert the historical helper check to a direct regression of the §2.12 implementation. |
| `test_zero_q` | `tests/test_pipe1d_accumulation.py` | KEEP | Zero-flow invariance remains valid under §2.12. |
| `test_zero_q_dry` | `tests/test_pipe1d_accumulation.py` | KEEP | Zero flow at a dry face remains valid under §2.12-§2.13.2. |
| `test_zero_q_old` | `tests/test_pipe1d_accumulation.py` | UPDATE | Remove dependence on the obsolete helper and test §2.12 production behavior. |
| `test_forward_after_reverse` | `tests/test_pipe1d_accumulation.py` | KEEP | Stepwise sign preservation remains required by §2.12. |
| `test_tiny_a_range` | `tests/test_pipe1d_accumulation.py` | KEEP | Near-dry loss robustness remains required by §2.12 and §2.13.2. |
| `test_half_pipe_reasonable` | `tests/test_pipe1d_vs_swmm.py` | UPDATE | Tighten or re-derive Manning comparison after geometric area and regime override changes in §2.4 and §2.6. |
| `test_slope_scaling` | `tests/test_pipe1d_vs_swmm.py` | UPDATE | Re-derive scaling where §2.6 normal-flow capping may become active. |
| `test_nonzero_head_gives_flow` | `tests/test_pipe1d_vs_swmm.py` | KEEP | Basic head-gradient direction remains required by §2.3. |
| `test_pipe1d_finite_above_crown` | `tests/test_pipe1d_vs_swmm.py` | UPDATE | Add explicit depth/area-above-crown checks for the Preissmann slot in §2.5. |
| `test_pipe1d_stable_at_large_dt` | `tests/test_pipe1d_vs_swmm.py` | UPDATE | Stability must be tied to the explicit CFL limits in §2.13, not only finiteness. |
| `test_pipe1d_vs_swmm` | `tests/test_pipe1d_vs_swmm.py` | UPDATE | Re-baseline after §2.5 slot, §2.6 regime override, and §2.8 outfall semantics. |
| `test_swmm_free_outfall_surcharges` | `tests/test_pipe1d_vs_swmm.py` | KEEP | This remains a useful SWMM reference for §2.5 and §2.8 even though it does not exercise Pipe1D. |
| `test_entrance_loss_reduces_flow` | `tests/test_pipe1d_vs_swmm.py` | UPDATE | Assert loss only at network end faces and no sub-cell dilution per §2.12. |
| `test_reversal_stable` | `tests/test_pipe1d_vs_swmm.py` | UPDATE | Recheck all virtual-node faces and local-loss direction after §2.1, §2.12, and §2.13.1. |
| `test_gpkg_pipe_cell_roundtrip` | `tests/test_pipe_cell_coupling_output.py` | KEEP | Per-cell metric schema is unchanged by the internal virtual-node rewrite under §2.15. |
| `test_gpkg_pipe_cell_roundtrip_multiple_links_and_cells` | `tests/test_pipe_cell_coupling_output.py` | KEEP | Variable sub-cell counts remain represented by the existing link/index schema under §2.1 and §2.15. |
| `test_gpkg_pipe_cell_geometry_roundtrip` | `tests/test_pipe_cell_coupling_output.py` | KEEP | Persisted geometry remains plain per-cell data under §2.2 and §2.15. |
| `test_gpkg_pipe_cell_legacy_fallback` | `tests/test_pipe_cell_coupling_output.py` | KEEP | Legacy persistence behavior is outside the hydraulic rewrite under §2.17. |
| `test_build_pipe_cell_items_includes_geometry` | `tests/test_pipe_cell_coupling_output.py` | KEEP | Geometry item construction remains compatible with §2.2 cell state. |
| `test_pipe_cell_ts_legacy_schema_migration` | `tests/test_gpkg_persistence.py` | KEEP | Database migration is unaffected by §2.1-§2.14 and remains protected by §2.17. |
| `test_drainage_and_structures_coupling` | `tests/test_swe2d_gpu_coupling_integration.py` | KEEP | The full Python/CUDA coupling entry point remains unchanged under §2.15. |
| `test_drainage_only_coupling` | `tests/test_swe2d_gpu_coupling_integration.py` | KEEP | The drainage-only crash guard remains required through §2.15. |
| `test_structures_only_coupling` | `tests/test_swe2d_gpu_coupling_integration.py` | KEEP | Structure-only behavior is outside Pipe1D scope under §2.17. |
| `test_new_schemes_through_coupling_path` | `tests/test_swe2d_gpu_coupling_integration.py` | KEEP | 2D scheme compatibility remains an unchanged §2.15 integration contract. |
| `test_rebuild_coupling_mid_simulation` | `tests/test_swe2d_gpu_coupling_integration.py` | KEEP | Reallocation lifecycle remains required after new §2.2 device arrays are introduced. |
| `test_cli_run_persists_drainage_link_flow_rows` | `tests/test_coupling_diagnostics_readback.py` | KEEP | Network-link aggregation and output keys remain unchanged under §2.15. |
| `test_cli_run_writes_expected_coupling_schema` | `tests/test_coupling_diagnostics_readback.py` | KEEP | The external coupling schema is explicitly preserved by §2.15. |
| `test_apply_external_sources_drainage_only_uses_coupled_source` | `tests/test_coupling_integration.py` | KEEP | Source injection is outside the internal solver rewrite and preserved by §2.15. |
| `test_pipe_cell_snapshot_at_t0_is_zero` | `tests/test_coupling_integration.py` | UPDATE | The mocked cell arrays/count metadata must reflect §2.1 sub-cell construction. |
| `test_real_pipe1d_readback_at_t0_is_zero` | `tests/test_coupling_integration.py` | UPDATE | Verify all new §2.2 cell and virtual-node state is initialized, not just legacy arrays. |
| `test_drainage_exchange_upload_runs_when_mesh_built_eagerly` | `tests/test_coupling_integration.py` | KEEP | Upload ordering remains an unchanged §2.15 lifecycle contract. |
| `test_readback_returns_zeros_on_size_mismatch` | `tests/test_coupling_integration.py` | UPDATE | Extend zero-fill checks to any newly exposed §2.2 readback arrays and revised counts. |
| `test_build_coupling_controller_with_pipe_network` | `tests/test_coupling_setup.py` | KEEP | The network graph and controller API remain unchanged under §2.15. |
| `test_full_matrix` | `tests/test_coupling_rain_matrix.py` | KEEP | Source-path combinations remain outside the internal hydraulics and are preserved by §2.15. |
| `test_multi_row_per_snap` | `tests/test_coupling_snap_indexing.py` | KEEP | Dynamic result-list indexing is unaffected by §2.1-§2.14. |
| `test_single_row_per_snap` | `tests/test_coupling_snap_indexing.py` | KEEP | Single-row snapshot indexing is unaffected by §2.1-§2.14. |
| `test_rows_from_lists` | `tests/test_coupling_snap_indexing.py` | KEEP | Plain-list result conversion remains part of the unchanged §2.15 interface. |
| `test_inlet_outfall_1_link_depth_matches_swmm` | `tests/test_drainage_inlet_outfall_vs_swmm.py` | UPDATE | Select an explicit §2.8 outfall mode and re-baseline depth after §2.4, §2.6, and §2.13 fixes. |
| `test_drainage_engine_in_drainage_network_module` | `tests/test_engine_class_placement.py` | KEEP | Python class placement is unchanged under §2.15 and outside hydraulic scope under §2.17. |
| `test_structure_engine_in_structures_module` | `tests/test_engine_class_placement.py` | KEEP | Structure class placement is outside Pipe1D scope under §2.17. |
| `test_backward_compat_reexport` | `tests/test_engine_class_placement.py` | KEEP | Import compatibility is unaffected by §2.1-§2.14. |
| `test_build_drainage_config_importable_from_extensions` | `tests/test_extensions_config_builders.py` | KEEP | The Python drainage configuration surface remains unchanged under §2.15. |
| `test_build_structures_config_importable_from_extensions` | `tests/test_extensions_config_builders.py` | KEEP | Structure configuration is outside the solver rewrite under §2.17. |
| `test_build_structures_bare_list_form` | `tests/test_extensions_config_builders.py` | KEEP | Bare-list structure handling is outside Pipe1D scope under §2.17. |
| `test_build_drainage_empty_returns_none` | `tests/test_extensions_config_builders.py` | KEEP | Empty-network handling remains an unchanged §2.15 contract. |
| `test_centroid_helper_is_canonical` | `tests/test_extensions_config_builders.py` | KEEP | Centroid helper placement is unrelated to §2.1-§2.14. |
| `test_render_network_with_selection_markers` | `tests/test_render_network_with_selection.py` | KEEP | Link/node result identifiers remain unchanged because §2.15 preserves the network graph. |
| `test_render_network_without_selection` | `tests/test_render_network_with_selection.py` | KEEP | Default link rendering consumes the unchanged §2.15 result contract. |
| `test_render_network_with_empty_selection` | `tests/test_render_network_with_selection.py` | KEEP | Missing-selection handling is independent of the solver rewrite. |
| `test_drainage_viewer_import` | `tests/test_results_panel.py` | KEEP | Viewer importability is outside the hydraulic changes under §2.17. |
| `test_drainage_gpkg_path_is_included` | `tests/test_results_path_wiring.py` | KEEP | Replay path wiring is unchanged by §2.1-§2.14. |
| `test_returns_records` | `tests/test_results_timestep_service.py` | KEEP | Coupling-result loading consumes the unchanged §2.15 persistence schema. |
| `test_no_table_returns_empty` | `tests/test_results_timestep_service.py` | KEEP | Missing-table behavior is outside the solver rewrite under §2.17. |
| `test_open_failure_returns_empty` | `tests/test_results_timestep_service.py` | KEEP | Database failure behavior is outside the solver rewrite under §2.17. |

### 2.2 New-needed buckets

| Test | File | Bucket | Reason |
|---|---|---|---|
| `test_subcell_count_scales_with_link_length_and_virtual_nodes_pass_through` | `tests/test_swe2d_pipe1d.py` | NEW-NEEDED | No existing test proves that longer links create more sub-cells connected by pass-through virtual nodes as required by §2.1. |
| `test_preissmann_slot_area_and_depth_above_crown` | `tests/test_swe2d_pipe1d_surcharge.py` | NEW-NEEDED | Existing surcharge tests cap area at full area and therefore do not validate §2.5. |
| `test_regime_override_caps_supercritical_flow_at_normal_flow` | `tests/test_pipe1d_vs_swmm.py` | NEW-NEEDED | No test isolates the §2.6 `checkNormalFlow` override. |
| `test_outfall_normal_depth_circular_bisection` | `tests/test_swe2d_pipe1d.py` | NEW-NEEDED | No existing test validates the circular normal-depth bisection in §2.8. |
| `test_outfall_rating_curve_interpolates_discharge` | `tests/test_swe2d_pipe1d.py` | NEW-NEEDED | No existing test validates §2.8 rating-curve interpolation. |
| `test_outfall_free_fixed_wse_and_tabular_modes` | `tests/test_swe2d_pipe1d.py` | NEW-NEEDED | The remaining three §2.8 modes need a parameterized boundary-condition contract. |
| `test_junction_surcharges_at_node_crown` | `tests/test_swe2d_pipe1d_surcharge.py` | NEW-NEEDED | Existing tests use `node_max_depth`, not the §2.10 `node_crown` threshold. |
| `test_virtual_node_hlle_upwind_flux_is_conservative_and_cfl_limited` | `tests/test_swe2d_pipe1d.py` | NEW-NEEDED | No test isolates the interior-face HLLE, upwind, and CFL requirements in §2.13.1. |
| `test_boundary_flux_uses_cell_area_not_full_area` | `tests/test_swe2d_pipe1d.py` | NEW-NEEDED | Existing boundary tests do not distinguish `cell_A` from `A_full` as required by §2.13.2. |
| `test_boundary_flux_respects_cfl_volume_limit` | `tests/test_swe2d_pipe1d.py` | NEW-NEEDED | No existing assertion checks the §2.13.2 boundary-volume CFL cap. |
| `test_boundary_flux_is_zero_for_dry_cell` | `tests/test_swe2d_pipe1d.py` | NEW-NEEDED | Existing dry tests are end-to-end and do not isolate the §2.13.2 dry-cell branch. |

### 2.3 Bucket totals

| Bucket | Count | Interpretation |
|---|---:|---|
| KEEP | 66 | Existing assertion remains useful without a solver-specific re-baseline. |
| UPDATE | 34 | Existing test should remain, but setup/count/numerical expectations must change. |
| NEW-NEEDED | 11 | Missing tests proposed in §4 for a later implementation/testing phase. |
| **Total planned rows** | **111** | 100 existing tests plus 11 proposed tests. |

## 3. Bug→test traceability

This table distinguishes a direct catch from a nearby test that only exercises the same code path. A weak or partial test is not credited as closing the bug.

| Spec bug | Existing test that catches it | Traceability assessment |
|---|---|---|
| §1.1 catastrophic drain | **NONE — gap.** Closest: `test_diffusion_wave_updates_area` and `test_inlet_outfall_1_link_depth_matches_swmm`. | The former only expects area to decrease and could pass during severe over-drainage; the latter is a long-run, factor-of-two comparison and does not isolate one-step boundary flux magnitude. |
| §1.2 sub-cell wiring | **Partial only:** `test_subcell_index_increases_downstream`; `test_fully_dynamic_mass_conservation_with_and_without_sub_cells`. | They check ordering and aggregate volume, but neither proves that each interior face connects through a virtual node or that flow passes from one sub-cell into the next. The bug remains a gap until §2.1 gets a dedicated test. |
| §1.3 incorrect initialization area | `test_init_area_from_depth`; secondarily `test_dry_pipe_no_change`. | These catch floor/full-area endpoint behavior. They do not cover partial circular depth or every sub-cell, so update the existing initialization test for §2.4 geometry. |
| §1.4 wrong initialization used in diffusion wave | **NONE — gap.** Closest: `test_diffusion_wave_updates_area` and `test_upload_node_depth_changes_area`. | Both inspect post-step changes but do not prove which initialized area the diffusion-wave kernel consumed. A pre-step/readback plus one-step analytic case is needed. |
| §1.5 no regime detection | **NONE — gap.** Closest: `test_half_pipe_reasonable` and `test_slope_scaling`. | Neither constructs a known supercritical or adverse-gradient case nor checks the Manning normal-flow cap required by §2.6. |
| §1.6 no Preissmann slot | **NONE — gap.** Closest: `test_surcharge_node_depth_uncapped`, `test_full_cell_flux_stability`, and `test_pipe1d_finite_above_crown`. | Existing tests only require uncapped/finite state; two explicitly expect `cell_A <= A_full`, which is incompatible with a slot-area assertion under §2.5. |
| §1.7 local-loss dilution | **Partial only:** `test_entrance_loss_reduces_flow` and the sign/magnitude tests in `tests/test_pipe1d_accumulation.py`. | They prove that loss reduces flow and does not reverse it, but not that `k_in/k_out` is applied once at network end faces rather than diluted across all sub-cells (§2.12). |
| §1.8 mass-balance boundary cells | `test_fully_dynamic_mass_conservation_with_and_without_sub_cells`; `test_mass_conservation_surcharge`; `test_pipe_end_moves_water_downhill`. | These can catch aggregate leakage, especially with sub-cells and pipe ends. They should be strengthened to isolate boundary-cell volume and the §2.7 virtual-node exclusion. |
| §1.9 no `crown_elev` | **NONE — gap.** Closest: `test_surcharge_node_depth_uncapped` and `test_swmm_free_outfall_surcharges`. | Existing tests compare depth with `node_max_depth` or pipe diameter, not a separately computed junction `node_crown`; they cannot validate §2.2/§2.10 crown-based surcharge. |

## 4. Proposed new test cases

These are designs only. Names assume placement in an existing test file; this plan does not create those tests.

| Proposed test | Setup | Expected assertion | Spec section |
|---|---|---|---|
| `test_subcell_count_scales_with_link_length_and_virtual_nodes_pass_through` | Build otherwise identical 10 m and 100 m one-link networks with `max_cell_length=10`; initialize an upstream pulse and dry/low downstream state. Read every sub-cell after several small steps. | Cell count is 1 versus 10; indices remain ordered; interior faces conserve transferred volume; downstream sub-cells become wet in order, proving pass-through rather than all cells being wired directly to network endpoints. | §2.1 |
| `test_preissmann_slot_area_and_depth_above_crown` | Build a circular pipe with known diameter, equal above-crown heads, SLOT surcharge mode, and no head gradient. Initialize/read back area and step at a small CFL-safe `dt`. | Depth remains above crown, `cell_A > A_full`, excess area equals `(y-yFull)*slot_width` within tolerance, and all state remains finite. | §2.5 |
| `test_regime_override_caps_supercritical_flow_at_normal_flow` | Construct a steep, partially full circular pipe whose uncapped diffusion/dynamic estimate is supercritical; compute independent Manning `Q_n` from geometric `A`, `R`, slope, and roughness. Run both solver modes if the API permits. | Returned `Q` has the expected sign and `abs(Q) <= abs(Q_n)+tol`; a nearby subcritical control case is not spuriously capped. | §2.6 |
| `test_outfall_normal_depth_circular_bisection` | Use a circular link ending at a `normal_depth` outfall with known slope, roughness, and imposed inflow. Independently solve Manning's equation for depth by a CPU bisection oracle over `[0, diameter]`. | Outfall depth converges to the oracle within geometric/numerical tolerance; discharge is finite and mass-consistent. Include near-zero and near-full bracket cases. | §2.8 |
| `test_outfall_rating_curve_interpolates_discharge` | Configure a monotone head-discharge table with three points; run heads exactly at knots, between knots, and outside the table range according to the specified endpoint policy. | Knot values are exact, interior discharge is linearly interpolated, endpoint behavior is deterministic, and discharge is non-negative/monotone. | §2.8 |
| `test_outfall_free_fixed_wse_and_tabular_modes` | Parameterize three one-link outfalls: `free`, `fixed_wse`, and `tabular`; supply a fixed elevation and a two-time-level stage series for the latter modes. | Free mode follows its zero-gradient/free-discharge rule, fixed WSE clamps the boundary head, and tabular mode interpolates stage in time. Each mode keeps finite state and correct flow direction. | §2.8 |
| `test_junction_surcharges_at_node_crown` | Join at least two links with different invert/crown elevations at one storage junction. Add enough volume to raise node head just below and then just above the maximum incident crown while keeping `node_max_depth` distinct. | Surcharge behavior switches at computed `node_crown`, not `node_max_depth`; storage mass remains conservative and end-face local losses are applied only on incident network faces. | §2.2, §2.10, §2.12 |
| `test_virtual_node_hlle_upwind_flux_is_conservative_and_cfl_limited` | Build a two-sub-cell horizontal link with unequal `A` and `Q`, no network-boundary forcing, and a deliberately large requested `dt`. Compute one interior-face HLLE flux and upwinded transported state independently. | The two cells receive equal-and-opposite volume change, flux direction matches upwinding, no negative area occurs, and transferred volume is bounded by the §2.13.1 CFL limit. | §2.13.1 |
| `test_boundary_flux_uses_cell_area_not_full_area` | Use a partially full boundary cell where `cell_A` differs greatly from `A_full`; hold node/cell heads and all other parameters fixed. Compare one step against an analytic boundary-flux calculation based on actual area. | Observed volume/discharge matches the `cell_A` calculation and demonstrably differs from the old `A_full` result. | §2.13.2 |
| `test_boundary_flux_respects_cfl_volume_limit` | Give a wet boundary cell a very large node-head difference and request a `dt` above its face CFL limit. Record initial area and one-step transferred volume. | Removed volume never exceeds available cell volume/CFL allowance, area remains at or above the floor, and decreasing requested `dt` removes the expected smaller amount. | §2.13.2 |
| `test_boundary_flux_is_zero_for_dry_cell` | Initialize a boundary cell and its adjacent node below the dry threshold, then repeat with the node wet enough to trigger intentional wetting. | Dry/dry boundary flux is zero (or numerical floor only); wet-node/dry-cell control permits bounded inflow without NaN, negative area, or catastrophic drainage. | §2.13.2 |

## 5. Out-of-scope / non-goals

| Non-goal | Boundary for this parallel-track plan |
|---|---|
| Actual test execution | This phase inventories and designs tests only; it does not run CUDA, unit, integration, or coverage suites. |
| Validation against SWMM gold-standard runs | Existing SWMM comparisons are catalogued, but no new SWMM input, baseline, tolerance, or gold result is generated or validated here. |
| Python test harness refactoring | Existing runners, fixtures, helper functions, parameter builders, and persistence utilities are not reorganized in this phase. |

## 6. Open questions / risks

| Open question or risk | Why it affects the later test strategy |
|---|---|
| If Step 4's flux kernel signs the network-boundary flux differently from the convention assumed by existing tests, which side owns positive inflow? | Assertions in `test_diffusion_wave_updates_area`, `test_dry_pipe_wets_from_upstream_node`, and the three proposed §2.13.2 tests must be re-derived from the implemented sign convention rather than inverted until they pass. |
| Will virtual-node state be exposed through readback, or only represented indirectly by per-cell arrays? | §2.1 pass-through can be tested directly only if vnode head/flow is observable; otherwise the test needs a conservative pulse-propagation oracle and may be less diagnostic. |
| Spec §2.13.2 subtracts local loss before computing effective node head. Which API/config field is the canonical source for nonzero `k_in/k_out` in direct-kernel tests? | Tests asserting node depth or boundary flux under local losses must opt into nonzero coefficients explicitly; using zero defaults would miss §1.7 and §2.12 entirely. |
| What exact endpoint policy applies to §2.8 rating curves and tabular stage series: clamp, extrapolate, or reject? | The proposed out-of-range assertions cannot be finalized until the implementation and review agree on deterministic endpoint semantics. |
| Is the Preissmann slot enabled for all shapes or only circular shapes, and is slot width observable in readback? | The §2.5 oracle needs the exact shape policy and either exposed slot width or a stable independent formula; otherwise it can assert only `A>A_full` and finiteness. |
| Does `node_crown` mean the maximum crown of all incident links, or a direction-sensitive upstream/downstream crown? | The §2.10 junction test must use asymmetric incident crowns to distinguish these interpretations and prevent an accidentally passing symmetric setup. |
| Are CFL limits applied per internal face, per boundary face, or globally through the caller's substep loop? | Test setup and tolerances differ substantially: an isolated one-step kernel assertion is appropriate for per-face limiting, while a host-level test must account for substep partitioning. |
| Existing test counts differ from prior research (14 rather than 15 in `test_swe2d_pipe1d.py`; 18 rather than approximately 25 in `test_swe2d_gpu_drainage_network.py`). | Code review should confirm whether tests were renamed/removed on another branch before using this inventory as a merge gate. |
