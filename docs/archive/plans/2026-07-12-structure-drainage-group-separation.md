---
type: plan
status: complete
created: 2026-07-12
completed: 2026-07-25
---

# Structure / Drainage Group Separation — Implementation Plan

> **For agentic workers:** Use subagent-driven-development skill or executing-plans to implement this plan task-by-task.

**Date:** 2026-07-12
**Status:** Draft
**Goal:** Restructure the Model tab → Structures & Drainage page so that structure-coupling widgets and drainage-network widgets live in visually distinct groups, making it clear which controls affect which feature.

**Architecture overview:** Single file modification: `swe2d/workbench/views/model_tab_view.py` — specifically the `_build_drain_form_widgets` method (lines 1129–~1350). The current "Layer Setup" group holds 5 layer combos (4 drainage + 1 structure) interleaved. The current "Culvert / Bridge" group holds structure-specific controls mixed with the shared `coupling_loop_combo`. The new layout splits this into two parallel columns of groups: "Structure Coupling" and "Drainage Network" sections, each with their own layer setup and tuning groups. The shared `coupling_loop_combo` appears once in the Structure section (since structure coupling drives it).

---

## 1. Problem

The Structures & Drainage page currently mixes structure-coupling controls (weirs, orifices, bridges, culverts, pumps) with drainage-network controls (1D pipe network, inlets, substepping) in adjacent group boxes without clear visual separation. Users cannot tell at a glance:
- "Are these controls for the bridge coupling or the drainage network?"
- "Will changing the culvert face-flux toggle affect drainage?"

The 5 layer combos in "Layer Setup" are similarly interleaved (drain_nodes, drain_links, drain_inlets, drain_node_inlets, structures) without any visual grouping.

---

## 2. Design

### New page structure

```
QFormLayout (page)
  ├── QGroupBox("Structure Coupling")
  │    ├── QGroupBox("Structures — Layers")     ← inner, non-checkable
  │    │    └── structures_layer_combo
  │    └── QGroupBox("Structures — Coupling Settings", advanced=True)
  │         ├── coupling_loop_combo                ← shared with drainage
  │         ├── culvert_solver_mode_combo
  │         ├── culvert_face_flux_chk
  │         ├── use_redistribution_chk
  │         └── bridge_stacked_coupling_mode_combo
  └── QGroupBox("Drainage Network")
       ├── QGroupBox("Drainage — Layers")
       │    ├── drain_nodes_layer_combo
       │    ├── drain_links_layer_combo
       │    ├── drain_inlets_layer_combo
       │    └── drain_node_inlets_layer_combo
       ├── QGroupBox("Drainage — Equation Set", advanced=True)
       │    ├── drainage_solver_mode_combo
       │    └── drainage_gpu_method_combo
       ├── QGroupBox("Drainage — Substepping", advanced=True)
       │    ├── drainage_coupling_substeps_spin
       │    └── drainage_max_coupling_substeps_spin
       └── QGroupBox("Drainage — Stability", advanced=True)
            ├── drainage_head_deadband_spin
            ├── drainage_dynamic_relaxation_spin
            ├── drainage_adaptive_depth_fraction_spin
            ├── drainage_adaptive_wave_courant_spin
            ├── drainage_implicit_iters_spin
            └── drainage_implicit_relax_spin
```

### Widget placement rationale

**Structure Coupling section:**
- `structures_layer_combo` — its own layer group
- `coupling_loop_combo` — listed in structure coupling settings (it's primarily about structure coupling backend, even though drainage uses the same loop). A tooltip note explains it's shared.
- All culvert/bridge/redistribution controls — clearly structure-related

**Drainage Network section:**
- 4 drainage layer combos (nodes, links, inlets, node-inlets) — its own layer group
- All drainage equation, substepping, and stability controls — clearly drainage-related

### Naming notes

The outer containers "Structure Coupling" and "Drainage Network" are **plain titled QGroupBoxes** (not checkable). They serve as visual section dividers — the toggle from the prior plan (`structures_enable_group`, `drainage_enable_group`) wraps the entire outer container if/when that plan is also implemented. For now, the outer groups are just headers.

The inner groups ("Structures — Layers", "Structures — Coupling Settings", "Drainage — Layers", "Drainage — Equation Set", etc.) are also plain titled QGroupBoxes, matching the existing pattern of sectioned groups within the page.

### Existing object names preserved

All widget `objectName` attributes remain unchanged so existing code (`studio_dialog.py` reads them via `getattr(self._model_tab_view, "structures_layer_combo", None)`, etc.) keeps working. Only the **parent group layout** changes.

---

## 3. Files to Modify

| File | Change |
|-------|--------|
| `swe2d/workbench/views/model_tab_view.py` | Restructure `_build_drain_form_widgets` (lines 1129–~1350): split into "Structure Coupling" and "Drainage Network" outer groups, each containing their own inner layer + tuning groups |
| `tests/test_model_tab_view.py` | Verify no widget objectName changes break test assertions |

---

## 4. Step-by-Step Implementation

- [ ] **Step 1: Read `_build_drain_form_widgets` in full** (lines 1129 to ~1350) to understand exact current widget placement and objectName registrations.

- [ ] **Step 2: Verify the existing widget order** — Confirm each widget's exact line and existing objectName so the new structure preserves them all.

- [ ] **Step 3: Restructure `_build_drain_form_widgets`.** Replace the body of the function with the new layout:
  - Outer group: "Structure Coupling"
    - Inner group: "Structures — Layers" containing `structures_layer_combo`
    - Inner group: "Structures — Coupling Settings" (advanced=True) containing `coupling_loop_combo`, `culvert_solver_mode_combo`, `culvert_face_flux_chk`, `use_redistribution_chk`, `bridge_stacked_coupling_mode_combo`
  - Outer group: "Drainage Network"
    - Inner group: "Drainage — Layers" containing `drain_nodes_layer_combo`, `drain_links_layer_combo`, `drain_inlets_layer_combo`, `drain_node_inlets_layer_combo`
    - Inner group: "Drainage — Equation Set" (advanced=True) containing `drainage_solver_mode_combo`, `drainage_gpu_method_combo`
    - Inner group: "Drainage — Substepping" (advanced=True) containing `drainage_coupling_substeps_spin`, `drainage_max_coupling_substeps_spin`
    - Inner group: "Drainage — Stability" (advanced=True) containing `drainage_head_deadband_spin`, `drainage_dynamic_relaxation_spin`, `drainage_adaptive_depth_fraction_spin`, `drainage_adaptive_wave_courant_spin`, `drainage_implicit_iters_spin`, `drainage_implicit_relax_spin`

- [ ] **Step 4: Preserve all tooltips.** Move each `setToolTip(...)` call alongside the corresponding widget creation in its new group.

- [ ] **Step 5: Verify object names.** Spot-check that no widget loses its `setObjectName(...)` call.

- [ ] **Step 6: Run existing tests.** Confirm `tests/test_model_tab_view.py` still passes since only the group layout changed, not widget object names.
  ```bash
  find . -type d -name __pycache__ -exec rm -rf {} +
  mamba run -n qgis_stable python3 -m unittest -v tests.test_model_tab_view
  ```

- [ ] **Step 7: Smoke test in QGIS.** Open the Studio workbench, Model tab → Structures & Drainage page:
  - Verify two clear outer sections visible: "Structure Coupling" and "Drainage Network".
  - Verify each outer section has its own Layers group containing the right combos.
  - Verify all structure coupling settings are visible under "Structures — Coupling Settings".
  - Verify all drainage equation/substepping/stability controls are visible under their respective inner groups.
  - Confirm the param_search filter still works on the new layout.

---

## 5. Backward Compatibility

- **No widget object names change** → `studio_dialog.py`, `tests/`, and any other consumers that read widgets via `getattr(self._model_tab_view, ...)` continue to work.
- **Group widget object names DO change** (the new outer section headers and the inner layer/settings groups). Any code that referenced the old "Layer Setup", "Culvert / Bridge", "Drainage Network — Equation Set", etc. group objects by name needs updating. Quick scan:
  - `grep -rn "layer_setup_group\|culvert_/bridge_group\|drainage_network" /home/aaron/.../swe2d/` — only the view file references these, no external consumers.
- **Param search / advanced filter** — `FilterableRowRegistry` is keyed on widgets, not groups. The inner groups all use `_add_param_row` so all widgets remain filterable.

---

## 6. Future Considerations (out of scope)

- This restructure is compatible with the earlier "Module Enable/Disable Toggles" plan — when implemented, the toggles will wrap the new outer "Structure Coupling" and "Drainage Network" groups.
- If structure-specific widgets ever need to be visually separated from culvert/bridge settings (e.g., a separate "Weirs & Orifices" group), the inner group structure allows for further splitting without re-arranging outer containers.