# GUI Tooltip Audit & Implementation Plan

## Summary

The codebase has **partial** tooltip coverage. 3 files (`map_tab_view.py`, `model_tab_view.py`, `temporal_dock.py`) have complete coverage. The remaining 14 GUI files are missing tooltips on 80+ widgets.

**Phase priority**: tabs the user sees first → dialogs → viewers.

---

## File-by-file inventory

### File 1: `swe2d/workbench/views/topology_tab_view.py`

**Direct widgets in `_build_ui()` — all missing tooltips:**

| Widget | Purpose |
|--------|---------|
| `topo_nodes_combo` | Select topology nodes layer |
| `topo_arcs_combo` | Select topology arcs layer |
| `topo_regions_combo` | Select topology regions layer |
| `topo_constraints_combo` | Select constraints layer |
| `topo_quad_edges_combo` | Select quad edges / transition layers |
| `topo_export_template_btn` | Create topology template layers |
| `topo_backend_combo` | Choose meshing backend (Gmsh / Structured) |
| `topo_default_size_spin` | Default target mesh element size |
| `topo_default_cell_type_combo` | Default cell type (tri/quad/cartesian/empty) |
| `topo_generate_btn` | Generate the mesh |
| `topo_terminate_btn` | Terminate mesh generation |
| `topo_status_lbl` | Status information label |
| `progress_bar` | Mesh generation progress |

**Widgets from `_build_topology_tab_controls()` — all missing tooltips:**

| Widget | Page |
|--------|------|
| `topo_gmsh_tri_algo_combo` | Algorithm |
| `topo_gmsh_quad_algo_combo` | Algorithm |
| `topo_gmsh_recombine_algo_combo` | Algorithm |
| `topo_gmsh_smoothing_spin` | Algorithm |
| `topo_gmsh_optimize_iters_spin` | Algorithm |
| `topo_gmsh_verbosity_spin` | Algorithm |
| `topo_gmsh_optimize_netgen_chk` | Algorithm |
| `topo_gmsh_arc_mode_combo` | Arcs & Interfaces |
| `topo_gmsh_mesh_size_min_spin` | Sizing |
| `topo_gmsh_tolerance_edge_length_spin` | Sizing |
| `topo_gmsh_mesh_size_from_points_chk` | Sizing |
| `topo_gmsh_quality_enable_chk` | Quality |
| `topo_gmsh_quality_max_iters_spin` | Quality |
| `topo_gmsh_quality_time_limit_spin` | Quality |
| `topo_quality_min_angle_spin` | Quality |
| `topo_quality_max_aspect_spin` | Quality |
| `topo_quality_max_non_orth_spin` | Quality |
| `topo_quality_min_area_edit` | Quality |
| `topo_quality_strict_chk` | Quality |

**Total: 33 missing tooltips**

---

### File 2: `swe2d/workbench/views/results_controls.py`

All widgets in overlay page, output page, and runs page lack tooltips:

| Widget | Purpose |
|--------|---------|
| `field_combo` | Field to render (Depth, Velocity, WSE, etc.) |
| `wse_render_combo` | WSE rendering mode (cell/nodal) |
| `cmap_combo` | Color map |
| `res_combo` | Render resolution |
| `opacity_spin` | Overlay opacity |
| `auto_contrast_chk` | Auto-contrast toggle |
| `min_depth_spin` | Min depth threshold for overlay |
| `color_min_spin` | Manual color range min |
| `color_max_spin` | Manual color range max |
| `lock_canvas_chk` | Lock canvas extent to overlay |
| `visible_only_chk` | Render only visible cells |
| `arrows_chk` | Show velocity arrows |
| `arrow_density_spin` | Arrow spacing in pixels |
| `arrow_length_spin` | Arrow length scale |
| `arrow_head_length_spin` | Arrow head length |
| `arrow_head_width_spin` | Arrow head width |
| `streamlines_chk` | Show streamlines |
| `streamline_backend_combo` | Streamline backend |
| `streamline_seed_spin` | Streamline seed count |
| `streamline_steps_spin` | Streamline integration steps |
| `overlay_enabled_chk` | Enable high-performance overlay |
| `export_btn` | Export overlay to GeoTIFF |
| `export_res_spin` | GeoTIFF pixel size |
| `extended_outputs_chk` | Include extended outputs |
| `save_mesh_chk` | Save mesh results |
| `save_line_chk` | Save line results |
| `save_coupling_chk` | Save coupling results |
| `save_max_only_chk` | Save max results only |
| `save_log_chk` | Save run log |
| `gpkg_lbl` | Current GPKG path |
| `refresh_btn` | Re-scan GPKG for new runs |
| `add_btn` | Add results from GeoPackages |
| `remove_btn` | Remove selected runs |
| `show_all_btn` | Show all runs |
| `hide_all_btn` | Hide all runs |

**Total: 34 missing tooltips**

---

### File 3: `swe2d/workbench/views/studio_viewer_plot.py`

| Widget | Purpose |
|--------|---------|
| `show_table_toggle` | Toggle data table visibility |

**Total: 1 missing tooltip**

---

### File 4: `swe2d/workbench/views/studio_viewer_pg.py`

| Widget | Purpose |
|--------|---------|
| `_element_type_combo` | Element type selector |
| `_element_id_combo` | Element ID selector |
| `_metric_combo` | Variable/metric selector |
| `show_table_toggle` | Show data table toggle |

**Total: 4 missing tooltips**

---

### File 5: `swe2d/workbench/views/studio_viewer_profile_pg.py`

| Widget | Purpose |
|--------|---------|
| `_etype_combo` | Element type selector |
| `_element_id_combo` | Element ID selector |
| `_var_combo` | Variable/metric selector |
| `_fill_combo` | Fill variable selector |
| `_cmap_combo` | Colormap selector |
| `_show_struct_chk` | Toggle structure annotations |
| `show_table_toggle` | Show data table toggle |

**Total: 7 missing tooltips**

---

### File 6: `swe2d/workbench/views/temporal_dock.py`

| Widget | Purpose |
|--------|---------|
| `_time_slider` | Timeline position slider |

**Total: 1 missing tooltip** (others already have tooltips)

---

### File 7: `swe2d/workbench/dialogs/batch_simulation_dialog.py`

| Widget | Purpose |
|--------|---------|
| `_gpkg_browse_btn` | Browse for GeoPackage |
| `_add_row_btn` | Add a new parameter row |
| `_remove_row_btn` | Remove selected rows |
| `_clear_btn` | Clear all rows |
| `_export_btn` | Export batch config to JSON |
| `_import_btn` | Import batch config from JSON |
| `_run_btn` | Run all batch simulations |
| `_cancel_btn` | Cancel running batch |
| `_status_btn` | Check batch status |

**Total: 9 missing tooltips**

---

### File 8: `swe2d/workbench/dialogs/coupling_results_dialog.py`

| Widget | Purpose |
|--------|---------|
| `component_combo` | Filter by coupling component |
| `metric_combo` | Filter by coupling metric |
| `object_combo` | Filter by coupling object ID |

**Total: 3 missing tooltips**

---

### File 9: `swe2d/workbench/dialogs/hydrograph_editor.py`

| Widget | Purpose |
|--------|---------|
| `add_row_btn` | Add a hydrograph data row |
| `remove_row_btn` | Remove selected rows |
| `load_csv_btn` | Load hydrograph from CSV |
| `save_csv_btn` | Save hydrograph to CSV |

**Total: 4 missing tooltips**

---

### File 10: `swe2d/workbench/dialogs/gpkg_explorer_dialog.py`

| Widget | Purpose |
|--------|---------|
| `refresh_btn` | Refresh table listing |
| `open_btn` | Open table viewer |
| `preview_btn` | Preview table contents |
| `rename_btn` | Rename selected table |
| `delete_btn` | Delete selected table |
| `delete_run_btn` | Delete all tables for a run ID |

**Total: 6 missing tooltips**

---

### File 11: `swe2d/workbench/dialogs/detached_mesh_dialog.py`

| Widget | Purpose |
|--------|---------|
| `view_mode_combo` | Select mesh view mode |
| `refresh_btn` | Refresh mesh view |

**Total: 2 missing tooltips**

---

### File 12: `swe2d/workbench/dialogs/run_log_viewer_dialog.py`

| Widget | Purpose |
|--------|---------|
| `run_combo` | Select run to view logs |
| `_apply_btn` | Apply selected run's inputs to UI |

**Total: 2 missing tooltips**

---

### File 13: `swe2d/workbench/dialogs/run_selection_dialog.py`

| Widget | Purpose |
|--------|---------|
| `select_all_btn` | Select all runs |
| `clear_all_btn` | Clear all run selections |

**Total: 2 missing tooltips**

---

### File 14: `swe2d/workbench/dialogs/sqlite_preview_dialog.py`

| Widget | Purpose |
|--------|---------|
| `limit_spin` | Row limit for preview |
| `refresh_btn` | Refresh table preview |

**Total: 2 missing tooltips**

---

### File 15: `swe2d/workbench/dialogs/topo_attr_table_dialog.py`

| Widget | Purpose |
|--------|---------|
| `refresh_btn` | Reload attributes from layer |
| `add_row_btn` | Add blank attribute row |
| `remove_row_btn` | Remove selected rows |

**Total: 3 missing tooltips**

---

### File 16: `hydra_plugin.py` — HYDRASettingsDialog

| Widget | Purpose |
|--------|---------|
| `_cuda_path_edit` | CUDA DLL path |
| `browse_btn` | Browse for CUDA DLL |
| `reset_btn` | Reset CUDA path to default |
| `deps_btn` | Check & install dependencies |

**Total: 4 missing tooltips**

---

## Grand Totals

| File | Missing | Priority |
|------|---------|----------|
| `topology_tab_view.py` | 33 | **P0 — main tab** |
| `results_controls.py` | 34 | **P0 — main results panel** |
| `studio_viewer_pg.py` | 4 | P1 — viewer |
| `studio_viewer_profile_pg.py` | 7 | P1 — viewer |
| `batch_simulation_dialog.py` | 9 | P1 — dialog |
| `studio_viewer_plot.py` | 1 | P2 — fallback viewer |
| `temporal_dock.py` | 1 | P2 — animation bar |
| `coupling_results_dialog.py` | 3 | P2 — dialog |
| `hydrograph_editor.py` | 4 | P2 — dialog |
| `gpkg_explorer_dialog.py` | 6 | P2 — dialog |
| `detached_mesh_dialog.py` | 2 | P2 — dialog |
| `run_log_viewer_dialog.py` | 2 | P2 — dialog |
| `run_selection_dialog.py` | 2 | P2 — dialog |
| `sqlite_preview_dialog.py` | 2 | P2 — dialog |
| `topo_attr_table_dialog.py` | 3 | P2 — dialog |
| `hydra_plugin.py` | 4 | P2 — settings |
| **Total** | **117** | |

---

## Implementation Approach

For each widget, add a single `setToolTip(...)` call immediately after widget construction. Follow existing patterns in `model_tab_view.py` and `map_tab_view.py` — concise, descriptive, single-sentence descriptions with optional detail sentences.

**Pattern:**
```python
widget.setToolTip("Short purpose. "
    "Optional detail about valid ranges or behavior.")
```

### Execution

1. **Phase 1 (P0)** — `topology_tab_view.py`, `results_controls.py` (67 tooltips)
2. **Phase 2 (P1)** — `studio_viewer_pg.py`, `studio_viewer_profile_pg.py`, `batch_simulation_dialog.py` (20 tooltips)
3. **Phase 3 (P2)** — Remaining 8 dialog files + `temporal_dock.py` + `studio_viewer_plot.py` + `hydra_plugin.py` (30 tooltips)
