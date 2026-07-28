---
type: plan
status: complete
created: 2026-07-12
completed: 2026-07-25
---

# Module Enable/Disable Toggles — Implementation Plan

> **For agentic workers:** Use subagent-driven-development skill or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Date:** 2026-07-12
**Status:** Draft
**Goal:** Add three toggle buttons to the Model tab that independently enable/disable hydraulic structures, drainage networks, and rainfall forcing — regardless of whether the corresponding layer combos are populated. Also remove the non-functional "Initial Conditions" group box that was mislabeled as a toggle.

**Architecture overview:** Three files are modified — `swe2d/workbench/views/model_tab_view.py` (UI, creates the toggles), `swe2d/workbench/studio_dialog.py` (logic, reads the toggles to gate config construction), and `tests/test_model_tab_view.py` (tests). The existing checkable group box pattern in `_start_param_group` is reused. The toggle guards in `studio_dialog.py` propagate all the way to the GPU solver: `None` configs → `SWE2DRunOptionsData` → `RunContext` → `SimulationWorker` → `build_coupling_controller()` never called (for structures/drainage) and `thiessen_forcing` never applied (for rainfall). No new files are needed.

---

## 1. Problem Statement

1. **Structures, drainage, and rainfall** can only be "disabled" if the corresponding layer combos are empty or lack valid layers. There is no explicit user-facing toggle to turn these features off regardless of layer state.
2. **The "Initial Conditions" group box** in the Solver Parameters page has `checkable=True` with `setChecked(False)` at startup — but this checkbox is never read anywhere in the codebase. Initial conditions are always applied. The group visually suggests a feature can be disabled, which is misleading.

---

## 2. Design

### Toggle widgets

| Toggle | Widget type | Object name | Default | Location |
|--------|-------------|-------------|---------|----------|
| Enable hydraulic structures | `QGroupBox` (checkable) | `structures_enable_group` | **Checked** (on) | Model tab → Structures & Drainage page, top of the page |
| Enable drainage network | `QGroupBox` (checkable) | `drainage_enable_group` | **Checked** (on) | Model tab → Structures & Drainage page, below structures toggle |
| Enable rainfall / hydrology | `QGroupBox` (checkable) | `rainfall_enable_group` | **Checked** (on) | Model tab → Rain / Hydrology page, top of the page |

Each toggle is a checkable `QGroupBox` with a label (e.g., "Enable hydraulic structures") that owns the relevant child widgets. When unchecked, `setEnabled(False)` is called on all child widgets in that group.

**Structures and drainage are separate toggles.** They currently share the same layer combos (drain_nodes, drain_links, structures), but they have independent solver impact — structures affect 2D cell face fluxes, drainage affects 1D pipe network coupling. Two independent toggles give the user fine-grained control. The layer combos remain enabled at all times (they're just data selectors); the toggles gate whether those data selections produce active forcing in the solver.

**Rationale for checkable group boxes (not bare checkboxes):**
- Matches the existing pattern for "Initial Conditions" already in the codebase.
- Qt handles `setEnabled(False)` propagation to all children automatically.
- Visually groups the feature's controls clearly.

### Removal: Initial Conditions checkable group

- Remove `checkable=True` from the "Initial Conditions" `QGroupBox` in `_build_solver_form_widgets` (line 649).
- The group becomes a plain titled section; the combo and spin boxes are always visible and enabled.
- No functional change in behavior (initial conditions were always applied anyway).

### Logic integration (studio_dialog.py)

The three builder callbacks (`_build_thiessen_rain_cn_forcing`, `_build_pipe_network_config`, `_build_hydraulic_structure_config`) are called unconditionally in `SWE2DRunOptionsBuilder.build`. Each already returns `None` when layers are not populated. The toggles add a second guard that propagates all the way to the GPU solver:

```
studio_dialog._build_*() → returns None (toggle off)
    → SWE2DRunOptionsData (cfg = None)
    → RunContext (ctx.cfg = None)
    → SimulationWorker
        → build_coupling_controller() not called → no drainage/structure modules (simulation_worker.py:580)
        → ctx.thiessen_forcing is None → no rainfall applied (simulation_worker.py:517)
```

```python
# In studio_dialog.py — each builder gains a toggle guard
def _build_hydraulic_structure_config(self):
    if getattr(self._model_tab_view, "structures_enable_group", None) is not None:
        if not self._model_tab_view.structures_enable_group.isChecked():
            self._log("[Structures] disabled by user toggle")
            return None
    # ... existing null checks ...

def _build_pipe_network_config(self):
    if getattr(self._model_tab_view, "drainage_enable_group", None) is not None:
        if not self._model_tab_view.drainage_enable_group.isChecked():
            self._log("[Drainage] disabled by user toggle")
            return None
    # ... existing null checks ...

def _build_thiessen_rain_cn_forcing(self):
    if getattr(self._model_tab_view, "rainfall_enable_group", None) is not None:
        if not self._model_tab_view.rainfall_enable_group.isChecked():
            self._log("[Rainfall] disabled by user toggle")
            return None
    # ... existing null checks ...
```

Using `getattr(..., None)` ensures backward compatibility: if the toggle widgets don't exist (e.g. in headless CLI context where `run_options_builder` is used directly), the builders fall through to their existing layer-population checks.

---

## 3. Files to Modify

| File | Change |
|-------|--------|
| `swe2d/workbench/views/model_tab_view.py` | Add 4 checkable group toggles (structures, drainage, rainfall — plus the inner non-checkable groups); remove `checkable=True` from Initial Conditions |
| `swe2d/workbench/studio_dialog.py` | Add toggle guards to 3 builder methods (structures, drainage, rainfall) |
| `tests/test_model_tab_view.py` | Add test cases for 3 new toggle widgets |

---

## 4. Step-by-Step Implementation

- [ ] **Step 1: Read `model_tab_view.py` around lines 1140–1175** (Layer Setup group, structures+drainage) and lines 780–820 (Rainfall Input group). Identify the exact code where the "Layer Setup" group starts so the toggle can be inserted at the top.

- [ ] **Step 2: Read `model_tab_view.py` around line 649** (Initial Conditions group). Identify the `_start_param_group` call with `checkable=True`.

- [ ] **Step 3: In `model_tab_view.py` — Remove `checkable=True` from Initial Conditions.** Change `form = self._start_param_group(param_form, "Initial Conditions", checkable=True)` → `form = self._start_param_group(param_form, "Initial Conditions")`.

- [ ] **Step 4: In `model_tab_view.py` — Add structures enable toggle.** Before the existing "Layer Setup" group in `_build_drain_page_widgets`, insert:
  ```python
  form = self._start_param_group(param_form, "Enable hydraulic structures", checkable=True)
  form.setChecked(True)
  # (children: drain_nodes_layer_combo, drain_links_layer_combo, drain_inlets_layer_combo,
  #  drain_node_inlets_layer_combo, structures_layer_combo — unchanged)
  ```
  **Important:** Wrap the existing "Layer Setup" group content inside this new checkable group. The existing `_start_param_group(param_form, "Layer Setup")` call becomes the second child group inside the checkable `structures_enable_group`.

- [ ] **Step 5: In `model_tab_view.py` — Add drainage enable toggle.** Insert a second checkable group `drainage_enable_group` below the structures toggle. Both toggles own the same layer combos (drain_nodes, drain_links, structures) — this is by design since structures and drainage share infrastructure. Each toggle independently controls its own builder in `studio_dialog.py`. The layer combos themselves stay enabled at all times (just data selectors); their effect on the solver is gated by the separate toggles.

  The page structure becomes:
  ```
  QFormLayout (page)
    ├── QGroupBox("Enable hydraulic structures")   ← checkable, owns:
    │    └── QFormLayout
    │         ├── QGroupBox("Layer Setup")          ← inner, non-checkable
    │         │    └── (drain_nodes_layer_combo, drain_links_layer_combo,
    │         │         drain_inlets_layer_combo, drain_node_inlets_layer_combo,
    │         │         structures_layer_combo)
    │         ├── QGroupBox("Culvert / Bridge")      ← advanced, child of toggle
    │         ├── QGroupBox("Drainage Network — Equation Set") ← advanced
    │         ├── QGroupBox("Drainage — Substepping") ← advanced
    │         └── QGroupBox("Drainage — Stability")   ← advanced
    ├── QGroupBox("Enable drainage network")      ← checkable (owns same layer combos)
    │    └── [same child structure as above — shared widgets]
    └── (other pages unchanged)
  ```

  **Note:** Since both checkable groups own the same child widgets, Qt's `setEnabled(False)` on either group will grey out those shared widgets. This is the correct UX — toggling either feature off greys out the layer combos, and both builder guards ensure that even if layers are selected, only the enabled feature produces solver output.

- [ ] **Step 6: In `model_tab_view.py` — Add rainfall enable toggle.** In `_build_rain_form_widgets`, before the existing `_start_param_group(param_form, "Rainfall Input")`, insert a checkable group. Rename the inner group to avoid naming conflict:
  ```python
  self.rainfall_enable_group = self._start_param_group(param_form, "Enable rainfall / hydrology", checkable=True)
  self.rainfall_enable_group.setChecked(True)
  form = self._start_param_group(param_form, "Rainfall Input")   # inner group, inside the toggle
  ```
  Note: the existing "Rainfall Input" group will become a child of the checkable group. The structure changes from:
  ```
  QFormLayout
    └── QGroupBox("Rainfall Input")   ← becomes inner group
  ```
  To:
  ```
  QFormLayout
    └── QGroupBox("Enable rainfall / hydrology")   ← checkable, owns layout
         └── QFormLayout
              └── QGroupBox("Rainfall Input")        ← inner, always-enabled container
  ```
  The inner "Rainfall Input" group should NOT be checkable. Move all existing rainfall widgets (`rain_gage_layer_combo`, `hyetograph_layer_combo`, `rain_rate_spin`, `use_spatial_rain_cn_chk`, `rain_update_interval_spin`, `storm_area_layer_combo`, `rain_boundary_buffer_rings_spin`) into the inner group.
  **Same pattern applies to the Structures & Drainage page:** outer checkable group → inner "Layer Setup" group → existing widgets.

- [ ] **Step 7: In `model_tab_view.py` — Add `_on_toggle_changed` handler for each toggle.** Connect each checkable group's `toggled` signal to a slot that logs the state:
  ```python
  def _on_structures_toggle(self, checked: bool) -> None:
      self._log(f"[UI] Structures & drainage {'enabled' if checked else 'disabled'} by user toggle")
  self.structures_enable_group.toggled.connect(self._on_structures_toggle)
  ```
  (Same pattern for rainfall.)

- [ ] **Step 8: In `studio_dialog.py` — Add toggle guards to `_build_hydraulic_structure_config`** (around line 1967). Add the guard at the very top of the method, before any existing checks.

- [ ] **Step 9: In `studio_dialog.py` — Add toggle guards to `_build_pipe_network_config`** (around line 1903). Add the guard at the very top.

- [ ] **Step 10: In `studio_dialog.py` — Add toggle guards to `_build_thiessen_rain_cn_forcing`** (around line 1868). Add the guard at the very top.

- [ ] **Step 11: In `studio_dialog.py` — Remove unused `checkable=True` imports/patterns.** Verify that no other code reads `initial_conditions_group.isChecked()` — it should be clean since the feature was never wired.

- [ ] **Step 12: Update `tests/test_model_tab_view.py`.** Add test cases:
  - `test_view_has_structures_enable_group` — asserts the widget is a `QGroupBox`, is checkable, and default is checked.
  - `test_view_has_drainage_enable_group` — asserts the widget is a `QGroupBox`, is checkable, and default is checked.
  - `test_view_has_rainfall_enable_group` — asserts the widget is a `QGroupBox`, is checkable, and default is checked.
  - `test_view_initial_conditions_group_not_checkable` — asserts the Initial Conditions group is NOT checkable.

- [ ] **Step 13: Verification.** Run existing tests:
  ```bash
  find . -type d -name __pycache__ -exec rm -rf {} +
  mamba run -n qgis_stable python3 -m unittest -v tests.test_model_tab_view
  ```

- [ ] **Step 14: Manual smoke test in QGIS.** Open the Studio workbench, Model tab:
  - Verify the three new toggles are visible and checked by default.
  - Uncheck each toggle and confirm child widgets are greyed out.
  - Uncheck rainfall toggle, leave a valid rain gage layer — confirm the forcing is not applied at run time (check log output).
  - Uncheck structures toggle, leave a valid structures layer — confirm the forcing is not applied.
  - Verify Initial Conditions group is no longer checkable.

---

## 5. Rollout & Backward Compatibility

- **Existing saved state / geopackage projects:** The toggle state is not persisted (no new state fields). On load, toggles default to checked (on). This is the safe default — existing projects that had layers configured will continue to work.
- **Headless / CLI:** The `getattr(..., None)` guard ensures builders work without the toggle widgets present.
- **Layer combo behavior unchanged:** Layer combos still accept any vector/table layer. The toggle provides an additional override layer on top of layer-population checks.

---

## 6. Future Extensibility (out of scope)

- Persisting toggle state per-project in the geopackage settings table.
- Toggle state propagated to the CLI `--no-structures`, `--no-drainage`, `--no-rainfall` flags.
- Decoupling the layer combos so structures toggle only greys out `structures_layer_combo` and drainage toggle only greys out `drain_nodes_layer_combo`, `drain_links_layer_combo`, etc. (Currently both toggles share the same layer combos, so either toggle greys out all of them.)
