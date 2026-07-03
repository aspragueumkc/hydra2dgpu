# SWE2D GUI — In-Depth Analysis & Recommendations

**Date**: 2026-07-03
**Scope**: Workbench view tabs, docks, menus, toolbars, widget grouping, and naming
**Status**: Recommendations — not yet implemented

---

## 1. Summary of Current Structure

The workbench uses **three left tabs** (`Layers`, `Mesh`, `Parameters`), each backed by a `QToolBox`, plus three right docks (`View`, `Inspector`, `Results`) and two bottom docks (`Temporal`, `Log`). There are **no custom toolbars, menus, or `.ui` files** — everything is Python-constructed. There is **zero use of `QGroupBox`** for visual grouping; sections are separated only by bold labels.

Below are the concrete issues and recommendations, grouped by severity.

---

## 2. Issues Found

### 2.1 Naming & label inconsistencies (low risk, high polish)

| Where | Issue | Why it matters |
|---|---|---|
| Topology tab | Page titled `"Arcs && Interfaces"` (`topology_tab_view.py`) | Double-ampersand is a Qt mnemonic shortcut (will render as `Arcs & Interfaces` with the `&` swallowed). Inconsistent with `Boundary Conditions` and `Debugging`. |
| Map tab | Buttons say `Export Mesh To Map Layers` and `Load Mesh From Selected Layers` while the same dialog page also has `Import Mesh Layers` and `Load Mesh From GPKG` — verb mix | Mixing imperative and gerund creates friction. |
| Mesh tab | `topo_export_template_btn` → label likely `"Create Topology Template Layers"` but no internal "Topology" prefix used elsewhere | Naming drift between `topo_*` internal IDs and the user-facing labels. |
| Model tab | `"Run time"` (single word) vs `"Run duration"` referenced in [USER_GUIDE.md](docs/guides/USER_GUIDE.md) | Guide and widget disagree. |
| Results | Tooltip `"Include extended outputs (...)"` is truncated with `...` instead of full text | Users hover and see an ellipsis. |
| Structures page | `"Bridge stacked coupling mode"` is a 4-word label for a combo, and carries a separate "Not Production Ready" warning box | Either hide by default or move to a `QGroupBox` titled "Experimental" with collapsible arrow. |
| Topology | Quality page lives in its own toolbox page but is only meaningful when the Gmsh backend is selected | Page should be disabled/hidden when backend ≠ Gmsh. |

### 2.2 Grouping & information density (medium risk)

The Model tab's `Solver Parameters` page lists **19 form rows in a single QFormLayout** with no visual breakpoints. Same for `Stability Controls` (9 rows). Users can't tell at a glance which spinner belongs to which subsystem.

The current "section headers" are just bold labels in `QVBoxLayout`. **No `QGroupBox`, no `QFrame`, no collapsible sections** anywhere in the workbench. This is unusual for a CFD-style tool with this many knobs.

### 2.3 Menu / toolbar absence (high-impact gap)

The plugin currently exposes:
- A view-mode `QComboBox` injected into QGIS' menu bar corner
- A few right-click context menus on tabs/log
- The save/settings menus inside the plot widget

**No plugin-level menu or toolbar exists.** Common SWE2D operations are scattered:

| Operation | Current location | Should be a toolbar button |
|---|---|---|
| Run 2D Model | Parameters → Run/Output page (scroll down) | ✅ Toolbar play button |
| Cancel | Same page, red button | ✅ Toolbar stop button |
| Take Snapshot | Same page | ✅ Toolbar camera button |
| Open Run Log Viewer | Layers → Utilities page | ✅ Toolbar log icon |
| Open Model GeoPackage Explorer | Layers → Utilities page | ✅ Toolbar folder icon |
| Discover Runs / Refresh runs | Results dock → Runs page | ✅ Toolbar refresh |
| Export Overlay to GeoTIFF | Results dock → Overlay page | ✅ Toolbar export |
| Batch Simulation | Parameters → Run/Output (cluttered) | ✅ Toolbar |

These are core workflow actions; burying them in nested toolbox pages hurts discoverability.

### 2.4 The Inspector dock duplication (medium risk)

The right-dock `Inspector` has two tabs (`Model Settings`, `Mesh Settings`) that **re-display the exact same widgets that already exist in the left dock's Parameters/Mesh tabs** — but as a read-only summary tree. Users see the same value in two places and may not understand which is the "source of truth."

The current convention seems to be: left dock = input, right dock = read-only snapshot. This is fine in principle but the **two-tree architecture is undocumented and confusing** without clear labels (e.g. "Live Parameters" vs "Live Mesh Settings" with a 🔒 icon).

### 2.5 Feature-flag UX (medium risk)

[studio_dialog.py](swe2d/workbench/studio_dialog.py) defines `_studio_feature_keywords` that hides widgets whose `objectName`/`text`/`title`/`toolTip` contains keywords like `"rain"`, `"drain"`, `"structure"`. This means:

- Renaming any widget text accidentally hides or reveals it.
- Users see widgets appear/disappear with no explanation.
- **No "feature toggles" UI surface for the user** — these flags exist in code but the user can't disable "rainfall" or "drainage" from the UI; they only get toggled programmatically.

### 2.6 Results dock layout (medium risk)

The Results dock has 3 pages but the most-used action (`Run List`) is on page 3 (`Runs`) while display controls are on page 1 (`Results & Overlay`) and storage toggles on page 2 (`Output`). Most users want to:
1. Pick a field to color
2. Toggle a run
3. Scrub timeline

These three actions should be **co-located at the top of one page**, with storage toggles on a separate (less-used) page.

---

## 3. Specific Recommendations

### 3.1 Introduce `QGroupBox` containers in the Model tab

The Model tab currently uses bare `QFormLayout`. Replace with **`QGroupBox` containers with checkable optional sections**:

```python
# Example structure
self.timestep_group = QGroupBox("Time Stepping")
self.timestep_group.setCheckable(False)
form = QFormLayout(self.timestep_group)
form.addRow("CFL:", cfl_spin)
form.addRow("dt (max):", dt_spin)
form.addRow("Initial dt:", initial_dt_spin)
form.addRow("Variable timestep:", adaptive_cfl_dt_chk)

self.physics_group = QGroupBox("Physics & Friction")
# ...

self.initial_conditions_group = QGroupBox("Initial Conditions")
self.initial_conditions_group.setCheckable(True)
self.initial_conditions_group.setChecked(False)  # collapsed by default
```

Proposed grouping for `Solver Parameters` (19 widgets → 5 groups):

| Group | Widgets |
|---|---|
| **Time Stepping** | CFL, dt, initial_dt, variable_timestep |
| **Physics & Friction** | Manning n, h_min, internal_flow_layer, internal_flow_field |
| **Spatial Reconstruction** | reconstruction_combo |
| **Temporal Integration** | temporal_order_combo |
| **Initial Conditions** | initial_condition_combo, initial_depth_spin, initial_wse_spin |
| **Numerical Options** | gpu_diag_sync_interval_spin, tiny_mode_combo, tiny_wet_cell_threshold_spin |
| **Performance** | enable_cuda_graphs_chk, swe2d_perf_mode_chk, degen_mode_combo |
| **Run Duration** | run_time_edit |

For `Stability Controls` (9 widgets → 3 groups):

| Group | Widgets |
|---|---|
| **Wet/Dry Front** | shallow_damping_depth_spin, shallow_front_recon_fallback_chk, front_flux_damping_spin, active_set_hysteresis_chk |
| **Capping** | depth_cap_spin, momentum_cap_min_speed_spin, momentum_cap_celerity_mult_spin |
| **Solver Safety** | max_inv_area_spin, cfl_lambda_cap_spin |

For `Structures & Drainage` (15 widgets → 4 groups):

| Group | Widgets |
|---|---|
| **Culvert / Bridge** | culvert_solver_mode_combo, culvert_face_flux_chk, use_redistribution_chk, bridge_stacked_coupling_mode_combo |
| **Drainage Network — Equation Set** | drainage_solver_mode_combo, drainage_gpu_method_combo |
| **Drainage — Substepping** | drainage_coupling_substeps_spin, drainage_max_coupling_substeps_spin |
| **Drainage — Stability** | drainage_head_deadband_spin, drainage_dynamic_relaxation_spin, drainage_adaptive_depth_fraction_spin, drainage_adaptive_wave_courant_spin, drainage_implicit_iters_spin, drainage_implicit_relax_spin |

For `Rain / Hydrology` (16 widgets → 3 groups):

| Group | Widgets |
|---|---|
| **Rainfall Input** | rain_rate_spin, use_spatial_rain_cn_chk, rain_update_interval_spin, storm_area_layer_combo, rain_boundary_buffer_rings_spin |
| **Infiltration** | infiltration_method_combo, cn_default_spin, ia_ratio_spin |
| **Source Stability** | max_rel_depth_increase_spin, max_source_depth_step_spin, max_source_rate_spin, extreme_rain_mode_chk, source_cfl_beta_spin, source_max_substeps_spin, source_true_subcycling_chk, source_imex_split_chk |

### 3.2 Add a top-level `QToolBar` with named actions

Build a `QToolBar` in [studio_host_methods.py](swe2d/workbench/views/studio_host_methods.py) and register it with QGIS:

```python
self.run_toolbar = QToolBar("HYDRA Run", self.iface.mainWindow())
self.run_toolbar.setObjectName("HydraRunToolbar")
self.iface.addToolBar(Qt.TopToolBarArea, self.run_toolbar)

actions = [
    ("play.png",  "Run 2D Model",         self._on_run_clicked,     False),
    ("stop.png",  "Cancel Run",           self._on_cancel_clicked,  True),
    ("camera.png","Take Snapshot",        self._on_snapshot,        True),
    ("sep",       None,                   None,                     False),
    ("list.png",  "Batch Simulation…",    self._on_batch,           False),
    ("sep",       None,                   None,                     False),
    ("folder.png","Open GeoPackage…",     self._on_open_gpkg,       False),
    ("log.png",   "Open Run Log",         self._on_open_log,        True),
    ("tools.png", "GPKG Explorer…",       self._on_open_explorer,   False),
    ("sep",       None,                   None,                     False),
    ("refresh.png","Discover Runs",       self._on_refresh_runs,    True),
    ("export.png", "Export Overlay TIFF…",self._on_export_overlay,  True),
]
```

### 3.3 Add a plugin menu under QGIS' Plugins menu

Currently the plugin only registers as `Plugins → HYDRA → Open Workbench`. Augment:

```
Plugins
└── HYDRA
    ├── Open Workbench
    ├── ─────────────
    ├── Recent Model GeoPackages ▸
    │   ├── model_v3.gpkg
    │   └── downtown_v2.gpkg
    ├── ─────────────
    ├── Run Last Simulation (Ctrl+R)
    ├── Batch Simulation… (Ctrl+B)
    ├── Open Run Log
    ├── Open GeoPackage Explorer
    ├── ─────────────
    ├── Export Current Results as GeoTIFF…
    └── Help → Documentation Hub
```

### 3.4 Relabel and consolidate the left dock

Current: `Layers | Mesh | Parameters` — these nouns are ambiguous. `Layers` could mean any QGIS layer; `Mesh` and `Parameters` are bare domain terms.

Proposed:

| Current | Proposed |
|---|---|
| **Layers** | **Setup** (or **Inputs**) |
| **Mesh** | **Mesh Generation** |
| **Parameters** | **Simulation** |

Or with emoji/icon to disambiguate without text: 📥 Setup / 🔷 Mesh / ⚙ Simulation.

### 3.5 Add a search/filter bar to the Parameters tab

With 70+ parameters across 5 toolbox pages, a single `QLineEdit` at the top of the Model tab that filters which widget rows are visible (matching against `toolTip`, label, or `objectName`) would be a major usability win. Implementation:

```python
self.param_search = QLineEdit()
self.param_search.setPlaceholderText("🔍 Filter parameters…")
self.param_search.textChanged.connect(self._filter_model_tab)
```

```python
def _filter_model_tab(self, text: str):
    text = text.lower().strip()
    for group in self._all_param_groups:  # list of QGroupBox
        any_visible = False
        for row_widget in group.findChildren(QWidget):
            label = self._label_for(row_widget).lower()
            tooltip = (row_widget.toolTip() or "").lower()
            objectname = row_widget.objectName().lower()
            if not text or text in label or text in tooltip or text in objectname:
                row_widget.show()
                any_visible = True
            else:
                row_widget.hide()
        group.setVisible(any_visible)
```

### 3.6 Collapse advanced parameters behind a "Show Advanced" toggle

Most users want 5–10 parameters. The remaining 60+ are tuning knobs. Add a single `QCheckBox` "Show advanced parameters" at the top of the Model tab that hides:

- All `max_inv_area_spin`, `cfl_lambda_cap_spin`, `gpu_diag_sync_interval_spin`
- All `momentum_cap_*` spinners
- All `culvert_*`, `bridge_*` combos
- All `drainage_*` (when no drainage network is loaded)
- All `source_*` (when rain is disabled)

Tag each widget with a `setProperty("advanced", True)` and filter them in one place. This is the same mechanism as `_studio_feature_keywords` but explicit and user-visible.

### 3.7 Make the Inspector read-only state clearer

Rename:
- `Model Settings` → `Parameters (read-only)`
- `Mesh Settings` → `Mesh Settings (read-only)`
- Add a 🔒 icon prefix to each tab label

Or alternatively, **replace the duplicate trees with a single "Inspector" tab that has a checkbox row at the top** (`[x] Show Model  [x] Show Mesh`) and renders both trees as collapsible sections in one tab.

### 3.8 Reorganize the Results dock

Collapse the 3 toolbox pages into 2:

**Page 1: "Display"** (the only page most users ever see)

```
┌─ Field & Colormap ─────────────────────┐
│ Field:       [depth       ▾]           │
│ Colormap:    [viridis     ▾]           │
│ WSE render:  [filled      ▾]           │
└────────────────────────────────────────┘
┌─ Color Range ──────────────────────────┐
│ [x] Auto contrast                      │
│ [ ] Lock canvas extent                 │
│ Min depth: [0.0]   Min: [auto] Max: [auto]│
└────────────────────────────────────────┘
┌─ Overlay Style ────────────────────────┐
│ Opacity:     [████████░░] 80%           │
│ Resolution:  [auto ▾]                   │
│ [ ] Show velocity arrows               │
│     Arrow spacing: [20] px             │
│ [ ] Show streamlines                   │
└────────────────────────────────────────┘
┌─ Runs ─────────────────────────────────┐
│ ☑ run_20260703_a  🟦                  │
│ ☐ run_20260702_b  🟧                  │
│ [+ Add…] [⟳] [−] [✓ All] [□ None]      │
└────────────────────────────────────────┘
```

**Page 2: "Storage"** (admin-level, less visited)

```
┌─ What to Save ─────────────────────────┐
│ [x] Mesh snapshots                     │
│ [x] Sample line timeseries             │
│ [x] Coupling timeseries                │
│ [x] Run log                            │
│ [ ] Extended outputs                   │
│ [ ] Max results only                   │
└────────────────────────────────────────┘
GPKG: [/path/to/results.gpkg] [Browse…]
```

### 3.9 Fix the double-ampersand and naming glitches

In [topology_tab_view.py](swe2d/workbench/views/topology_tab_view.py):

```python
# Old
self._toolbox.addItem(self._arcs_page, "Arcs && Interfaces")
# New — Qt's addItem interprets && as mnemonic; either escape or rename
self._toolbox.addItem(self._arcs_page, "Arcs & Interfaces")  # still bad — shows "Arcs"
# Cleanest:
self._toolbox.addItem(self._arcs_page, "Arcs and Interfaces")
```

Standardize capitalization: page titles use **Title Case** (`"Layer Setup"`, `"General"`), but a few widgets use **Sentence case** (`"Max rel depth increase:"`). Pick one — Qt convention is Title Case for group/tab labels, Sentence case for field labels ending in `:`.

### 3.10 Add validation badges and inline help

Currently each widget has a tooltip but no inline state. Add:

- **Yellow dot** on the tab title when any required field is empty (e.g. no GPKG path set in Run/Output).
- **Inline hint labels** under each group: e.g. under "Initial Conditions" show `ℹ Dry start uses bed elevation only`.
- **Inline error labels** when invalid: e.g. `⚠ dt > 1.0 may be unstable at CFL 0.45`.

Implementation pattern:

```python
class HintLabel(QLabel):
    def __init__(self, text=""):
        super().__init__(text)
        self.setProperty("role", "hint")
        self.setStyleSheet("color: #888; font-style: italic; padding-left: 12px;")
```

### 3.11 Replace ad-hoc Run/Output page with a dedicated Run dock

The `Run / Output` toolbox page currently mixes:

- The actual "Run" button (top)
- A progress bar
- Output interval fields
- A "Debugging" subsection with snapshot/preview buttons
- Storage path fields
- Load/save settings buttons

Move "Run" controls to a **dedicated `Run` dock at the bottom**, near the temporal dock and log. This puts all execution-related controls physically together (timeline ↔ run controls ↔ log). The Run/Output page in Parameters can keep only the interval and storage fields.

Proposed bottom dock layout:

```
┌─ HYDRA Run ──────────────┬─ HYDRA Temporal ─────────────┐
│ [▶ Run] [⏹ Cancel]       │ ◄ ▶ ▶  T = 1.234 hr  [1× ▾]  │
│ ▓▓▓▓▓░░░░░░  45%         │                              │
│ [📸 Snapshot] [⏏ Batch…]  │                              │
└──────────────────────────┴──────────────────────────────┘
┌─ HYDRA Log ──────────────────────────────────────────┐
│ [INFO] Step 1024, t=2.0s, dt=0.5...                   │
│ ...                                                    │
└──────────────────────────────────────────────────────┘
```

### 3.12 Surface the feature flags in a Settings dialog

Right now `rainfall` and `drainage_structures` flags are set programmatically and hide widgets via keyword matching. Add a `HYDRA → Settings…` dialog with checkboxes:

```
[ ] Enable rainfall module
[ ] Enable drainage networks
[ ] Enable hydraulic structures (weirs, culverts, bridges)
[ ] Enable bridge stacked coupling (experimental)
```

This both fixes the discoverability problem and replaces the brittle keyword-based widget hiding.

### 3.13 Standardize combo box labels with units

Several combos have values that imply units but don't display them:

| Combo | Current | Proposed |
|---|---|---|
| `tiny_mode_combo` | "Off (0)", "Auto (1)", ... | "Disabled", "Auto-detect", "Fused", "Persistent" |
| `temporal_order_combo` | "Euler RK1 (1)", ... | "RK1 (Euler)", "RK2 (Heun)", "RK4", "RK4 (graph-safe)", "RK5 (graph-safe)" |
| `reconstruction_combo` | "MUSCL Fast (1)", ... | "1st-order", "MUSCL + Superbee", "MUSCL + MinMod", "MUSCL + MC", "MUSCL + Van Leer", "WENO3-like", "WENO5" |
| `bridge_stacked_coupling_mode_combo` | "phase3_spatial", "legacy_scalar_weighting" | "Phase 3 — Spatial", "Legacy — Scalar" |

### 3.14 Make the inspector trees actually useful

The Inspector currently shows a flat list of widget→value pairs. Make it hierarchical and contextual:

- Add a "Current focus" indicator (bold) on the row corresponding to the currently-visible toolbox page
- Group rows under collapsible sections (Solver / Time / Physics / etc.)
- Highlight rows whose value has changed since the last run (orange dot)
- Add a "Reset to default" right-click menu on each row

---

## 4. Suggested Refactor Sequence

| Phase | Effort | Impact | Work |
|---|---|---|---|
| **A. Polish** | 1 day | Medium | Rename `&&` glitch, standardize Title Case, fix tooltip truncation, fix unit-suffix consistency in combo labels. |
| **B. Group boxes** | 3 days | High | Introduce `QGroupBox` containers in Model tab per §3.1. Verify `workbench_dialog_builder.py:372-425` (the Inspector tree) still references widgets via `findChild`. |
| **C. Toolbar + menu** | 2 days | High | Add QToolBar + Plugins menu per §3.2-3.3. Move Run/Cancel/Snapshot/Refresh/Export buttons out of toolbox pages. |
| **D. Filter & advanced toggle** | 2 days | High | Implement search filter and "Show advanced" toggle per §3.5-3.6. |
| **E. Results reorganization** | 1 day | Medium | Collapse to Display + Storage pages per §3.8. |
| **F. Run dock** | 2 days | High | Extract Run dock per §3.11. Wire signals through `run_controller.py`. |
| **G. Settings dialog** | 2 days | Medium | Replace keyword flag system with explicit Settings dialog per §3.12. |

---

## 5. What I Did NOT Recommend (and why)

- **Switching to `.ui` files (Qt Designer):** all current views are programmatic; refactoring 70+ widgets into XML adds risk for marginal benefit. Keep Python.
- **Docking out every page into its own dock:** would fragment the workflow. Keep the QToolBox pages inside Parameters tab but group them with QGroupBox.
- **Moving from `QToolBox` to `QTabWidget`:** `QToolBox` is the right choice for many pages of related content; the problem is the *content within* each page, not the container.

---

## 6. Validation Targets

After each phase, run:
- `tests/test_workbench_tab_views.py`
- `tests/test_workbench_dialog_builder.py`
- `tests/test_model_tab_view.py`
- `tests/test_topology_tab_view.py`
- `tests/test_dialog_tab_views_integration.py`

These exercise the widget tree and will catch any references that break when widgets are regrouped.