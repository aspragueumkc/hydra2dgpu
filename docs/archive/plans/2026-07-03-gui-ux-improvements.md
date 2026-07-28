---
type: plan
status: complete
created: 2026-07-03
completed: 2026-07-25
---

# SWE2D GUI/UX Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the GUI/UX improvements described in `docs/GUI_UX_RECOMMENDATIONS.md` and `docs/QT_GUI_UX_IMPROVEMENTS.md` across the HYDRA2D workbench.

**Architecture:** Keep the current programmatic Qt widget tree and MVP separation. Changes are localized to view files (`swe2d/workbench/views/`), host helpers (`swe2d/workbench/views/studio_host_methods.py`), the dialog builder (`swe2d/workbench/workbench_dialog_builder.py`), and supporting dialogs. Existing `objectName` attributes are preserved so `findChild`/attribute-based tests keep passing.

**Tech Stack:** Python 3.12, `qgis.PyQt` (QtWidgets/Gui/Core), QGIS plugin APIs.

---

## Preflight

- [ ] **Step 1: Verify git state**

```bash
git status --short
```

Expected: empty output (or only expected local changes).

- [ ] **Step 2: Baseline the relevant tests**

> These tests need a display. If you are headless, prepend `xvfb-run -a`.

```bash
mamba run -n qgis_stable pytest tests/test_model_tab_view.py tests/test_workbench_tab_views.py tests/test_topology_tab_view.py tests/test_workbench_dialog_builder.py tests/test_dialog_tab_views_integration.py -q
```

Expected: all pass (some may be slow because they construct `QApplication`).

---

## Phase A: Naming, labels, and quick polish

### Task A.1: Fix topology page title and Gmsh-only visibility

**Files:**
- Modify: `swe2d/workbench/views/topology_tab_view.py:186-219` and `update_control_summary`
- Modify: `tests/test_topology_tab_view.py`

**Why:** `Arcs && Interfaces` is a Qt mnemonic bug; disabled Gmsh-only pages give no feedback.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_topology_tab_view.py`:

```python
    def test_arcs_page_title_is_plain_text(self):
        from swe2d.workbench.views.topology_tab_view import TopologyTabView
        view = TopologyTabView()
        self.assertEqual(view._toolbox.itemText(view._arcs_idx), "Arcs and Interfaces")

    def test_non_gmsh_pages_are_disabled_and_suffixed(self):
        from swe2d.workbench.views.topology_tab_view import TopologyTabView
        view = TopologyTabView()
        view.topo_backend_combo.setCurrentIndex(view.topo_backend_combo.findData("structured"))
        view.update_control_summary()
        for idx in (view._algo_idx, view._arcs_idx, view._sizing_idx,
                    view._threading_idx, view._transfinite_idx, view._quality_idx):
            self.assertFalse(view._toolbox.isItemEnabled(idx))
            self.assertIn("(Gmsh only)", view._toolbox.itemText(idx))
```

- [ ] **Step 2: Run the test and confirm it fails**

```bash
mamba run -n qgis_stable pytest tests/test_topology_tab_view.py::TestTopologyTabView::test_arcs_page_title_is_plain_text tests/test_topology_tab_view.py::TestTopologyTabView::test_non_gmsh_pages_are_disabled_and_suffixed -v
```

Expected: FAIL — title is `Arcs && Interfaces`, no suffix.

- [ ] **Step 3: Update the view**

In `swe2d/workbench/views/topology_tab_view.py`, replace the arcs page insertion:

```python
        self._arcs_idx = self._toolbox.addItem(arcs_page, "Arcs && Interfaces")
```

with:

```python
        self._arcs_idx = self._toolbox.addItem(arcs_page, "Arcs and Interfaces")
```

After the Quality page block, add:

```python
        self._gmsh_only_indices = (
            self._algo_idx, self._arcs_idx, self._sizing_idx,
            self._threading_idx, self._transfinite_idx, self._quality_idx,
        )
        self._gmsh_only_base_titles = {
            self._algo_idx: "Algorithm",
            self._arcs_idx: "Arcs and Interfaces",
            self._sizing_idx: "Sizing",
            self._threading_idx: "Threading",
            self._transfinite_idx: "Transfinite",
            self._quality_idx: "Quality",
        }
```

Replace the existing Gmsh toggle loop in `update_control_summary` (around line 441):

```python
        for idx in (getattr(self, "_algo_idx", None), ...):
            if idx is not None:
                self._toolbox.setItemEnabled(idx, is_gmsh)
```

with:

```python
        for idx in self._gmsh_only_indices:
            base = self._gmsh_only_base_titles.get(idx, "")
            if is_gmsh:
                self._toolbox.setItemText(idx, base)
                self._toolbox.setItemEnabled(idx, True)
            else:
                self._toolbox.setItemText(idx, f"{base} (Gmsh only)")
                self._toolbox.setItemEnabled(idx, False)
```

- [ ] **Step 4: Run the tests again**

```bash
mamba run -n qgis_stable pytest tests/test_topology_tab_view.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add swe2d/workbench/views/topology_tab_view.py tests/test_topology_tab_view.py
git commit -m "fix(topology): clean Arcs title and dim Gmsh-only pages"
```

---

### Task A.2: Standardize combo labels and model tab casing

**Files:**
- Modify: `swe2d/workbench/views/model_tab_view.py:388-534`
- Modify: `tests/test_model_tab_view.py` (add label assertions)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_model_tab_view.py`:

```python
    def test_combo_labels_are_human_readable(self):
        view = self._make_view()
        tiny_labels = [view.tiny_mode_combo.itemText(i) for i in range(view.tiny_mode_combo.count())]
        self.assertIn("Persistent", tiny_labels)
        recon_labels = [view.reconstruction_combo.itemText(i) for i in range(view.reconstruction_combo.count())]
        self.assertIn("MUSCL + Superbee", recon_labels)
        bridge_labels = [view.bridge_stacked_coupling_mode_combo.itemText(i)
                         for i in range(view.bridge_stacked_coupling_mode_combo.count())]
        self.assertIn("Phase 3 — Spatial", bridge_labels)
```

- [ ] **Step 2: Run the test and confirm it fails**

```bash
mamba run -n qgis_stable pytest tests/test_model_tab_view.py::TestModelTabView::test_combo_labels_are_human_readable -v
```

Expected: FAIL — old labels present.

- [ ] **Step 3: Update the combo item texts**

In `swe2d/workbench/views/model_tab_view.py`, replace the `tiny_mode_combo` block:

```python
        self.tiny_mode_combo.addItem("Off (0)", 0)
        self.tiny_mode_combo.addItem("Auto (1)", 1)
        self.tiny_mode_combo.addItem("Fused (2)", 2)
        self.tiny_mode_combo.addItem("Persistent (3)", 3)
```

with:

```python
        self.tiny_mode_combo.addItem("Disabled", 0)
        self.tiny_mode_combo.addItem("Auto-detect", 1)
        self.tiny_mode_combo.addItem("Fused", 2)
        self.tiny_mode_combo.addItem("Persistent", 3)
```

Replace the `reconstruction_combo` block:

```python
        for text, data in [
            ("First-order (baseline)", 0),
            ("MUSCL Fast (high-throughput)", 1),
            ("MUSCL MinMod (robust)", 2),
            ("MUSCL MC (less-diffusive TVD)", 3),
            ("MUSCL Van Leer (smooth TVD)", 4),
            ("WENO3-like (GPU experimental)", 5),
            ("WENO5 (GPU, 3rd-order LSQ)", 6),
        ]:
```

with:

```python
        for text, data in [
            ("1st-order", 0),
            ("MUSCL + Superbee", 1),
            ("MUSCL + MinMod", 2),
            ("MUSCL + MC", 3),
            ("MUSCL + Van Leer", 4),
            ("WENO3-like", 5),
            ("WENO5", 6),
        ]:
```

Replace the `temporal_order_combo` block:

```python
        for text, data in [
            ("Euler (RK1, 1st-order)", 1),
            ("RK2 (Heun, 2nd-order, default)", 2),
            ("RK3 (SSP Shu-Osher, 3rd-order)", 3),
            ("RK4 (classic, 4th-order)", 4),
            ("Graph-safe RK4 (true staged)", 5),
            ("Graph-safe RK5 (Cash-Karp)", 6),
        ]:
```

with:

```python
        for text, data in [
            ("RK1 (Euler)", 1),
            ("RK2 (Heun)", 2),
            ("RK3 (SSP Shu-Osher)", 3),
            ("RK4 (classic)", 4),
            ("RK4 (graph-safe)", 5),
            ("RK5 (graph-safe)", 6),
        ]:
```

Replace the `bridge_stacked_coupling_mode_combo` block:

```python
        self.bridge_stacked_coupling_mode_combo.addItem(
            "Phase 3 spatial redistribution (recommended)", "phase3_spatial"
        )
        self.bridge_stacked_coupling_mode_combo.addItem(
            "Legacy scalar weighting (backward-compatible)", "legacy_scalar"
        )
```

with:

```python
        self.bridge_stacked_coupling_mode_combo.addItem(
            "Phase 3 — Spatial", "phase3_spatial"
        )
        self.bridge_stacked_coupling_mode_combo.addItem(
            "Legacy — Scalar", "legacy_scalar"
        )
```

- [ ] **Step 4: Run the tests**

```bash
mamba run -n qgis_stable pytest tests/test_model_tab_view.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add swe2d/workbench/views/model_tab_view.py tests/test_model_tab_view.py
git commit -m "refactor(model): human-readable combo labels"
```

---

### Task A.3: Fix truncated Results tooltip

**Files:**
- Modify: `swe2d/workbench/views/results_controls.py:291-297`

- [ ] **Step 1: Update the tooltip text**

Replace:

```python
        self.extended_outputs_chk = QtWidgets.QCheckBox(
            "Include extended outputs (momentum, qmag, wet mask, Fr, Manning)")
        self.extended_outputs_chk.setToolTip(
            "Include additional output fields beyond depth and velocity. "
            "Increases result file size."
        )
```

with:

```python
        self.extended_outputs_chk = QtWidgets.QCheckBox(
            "Include extended outputs (momentum, qmag, wet mask, Fr, Manning)")
        self.extended_outputs_chk.setToolTip(
            "Include extended output fields: momentum components, discharge magnitude, "
            "wet mask, Froude number, and Manning n. Increases result file size."
        )
```

- [ ] **Step 2: Add a regression test**

Add to `tests/test_workbench_tab_views.py` under a new `TestResultsToolbox` class:

```python
class TestResultsToolbox(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_app()

    def test_extended_outputs_tooltip_is_not_truncated(self):
        from swe2d.workbench.views.results_controls import ResultsToolbox
        toolbox = ResultsToolbox()
        tip = toolbox.extended_outputs_chk.toolTip()
        self.assertNotIn("...", tip)
        self.assertIn("Froude", tip)
```

- [ ] **Step 3: Run and commit**

```bash
mamba run -n qgis_stable pytest tests/test_workbench_tab_views.py::TestResultsToolbox -v
git add swe2d/workbench/views/results_controls.py tests/test_workbench_tab_views.py
git commit -m "fix(results): full tooltip for extended outputs"
```

---

### Task A.4: Relabel left dock tabs

**Files:**
- Modify: `swe2d/workbench/views/studio_tab_builder.py:46-48`
- Modify: `tests/test_dialog_tab_views_integration.py:83`

- [ ] **Step 1: Update tab labels**

In `compose_left_pane`:

```python
    dialog._left_tabs.addTab(build_map_tab(dialog), "Setup")
    dialog._left_tabs.addTab(build_topology_tab(dialog), "Mesh Generation")
    dialog._left_tabs.addTab(build_model_tab(dialog), "Simulation")
```

- [ ] **Step 2: Update the integration test expected set**

Replace:

```python
            for expected in ["Mesh", "Layers", "Topo Mesh", "Boundary", "Parameters"]:
```

with:

```python
            for expected in ["Setup", "Mesh Generation", "Simulation", "Boundary", "Mesh"]:
```

- [ ] **Step 3: Run and commit**

```bash
mamba run -n qgis_stable pytest tests/test_dialog_tab_views_integration.py::TestTabViewsAreInLeftTabs -v
git add swe2d/workbench/views/studio_tab_builder.py tests/test_dialog_tab_views_integration.py
git commit -m "refactor(workbench): clearer left dock tab labels"
```

---

### Task A.5: Add keyboard shortcuts

**Files:**
- Modify: `swe2d/workbench/studio_dialog.py`
- Modify: `tests/test_workbench_dialog_builder.py`

- [ ] **Step 1: Add the shortcuts constant and installer**

Add near the top of `SWE2DWorkbenchStudioDialog` (after `_CTRL_TIME_UNIT`):

```python
KEYBOARD_SHORTCUTS = [
    ("run", "Ctrl+R", lambda dlg: dlg._controller.on_run()),
    ("cancel", "Ctrl+.", lambda dlg: dlg._controller.on_cancel()),
    ("save_config", "Ctrl+S", lambda dlg: dlg._model_tab_view.save_settings_btn.click()
     if hasattr(dlg, "_model_tab_view") and dlg._model_tab_view is not None else None),
    ("open_gpkg", "Ctrl+O", lambda dlg: dlg._mesh_controller.load_2d_model_geopackage()),
    ("refresh_results", "F5", lambda dlg: dlg._on_results_refresh()),
]
```

Add a method in `SWE2DWorkbenchStudioDialog`:

```python
    def _install_keyboard_shortcuts(self) -> None:
        """Install global application shortcuts for common workbench actions."""
        from qgis.PyQt.QtGui import QKeySequence
        from qgis.PyQt.QtWidgets import QShortcut
        for _name, seq, cb in KEYBOARD_SHORTCUTS:
            shortcut = QShortcut(QKeySequence(seq), self)
            shortcut.setContext(QtCore.Qt.ApplicationShortcut)
            shortcut.activated.connect(lambda _cb=cb: _cb(self))
```

Call it at the end of `WorkbenchDialogBuilder._build_dialog_ui`:

```python
        self._studio_apply_visual_profile("Default")
        dlg._studio_apply_feature_filters()
        dlg._install_keyboard_shortcuts()  # add this line
        dlg._layer_controller.refresh_layer_combos()
```

- [ ] **Step 2: Add a test**

Add to `tests/test_workbench_dialog_builder.py`:

```python
    def test_keyboard_shortcuts_constant_exists(self):
        from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog, KEYBOARD_SHORTCUTS
        self.assertIsInstance(KEYBOARD_SHORTCUTS, list)
        self.assertGreaterEqual(len(KEYBOARD_SHORTCUTS), 3)
        names = [s[0] for s in KEYBOARD_SHORTCUTS]
        self.assertIn("run", names)
        self.assertIn("cancel", names)
```

- [ ] **Step 3: Run and commit**

```bash
mamba run -n qgis_stable pytest tests/test_workbench_dialog_builder.py -v
git add swe2d/workbench/studio_dialog.py tests/test_workbench_dialog_builder.py
git commit -m "feat(workbench): keyboard shortcuts for run/cancel/save/open/refresh"
```

---

### Task A.6: Standardize button sizing

**Files:**
- Modify: `swe2d/workbench/views/studio_tab_builder.py`

- [ ] **Step 1: Add the sizing helper and apply it**

Add at module level in `swe2d/workbench/views/studio_tab_builder.py`:

```python
def _size_button(btn: QtWidgets.QPushButton, role: str = "action") -> None:
    """Apply a canonical size to a button based on its role."""
    if role == "icon":
        btn.setFixedSize(24, 24)
    elif role == "primary":
        btn.setMinimumSize(100, 32)
        font = btn.font()
        font.setBold(True)
        btn.setFont(font)
    else:
        btn.setMinimumSize(80, 28)
```

In `make_left_controls_compact`, after the layout loops, add:

```python
    for btn in parent_widget.findChildren(QtWidgets.QPushButton):
        try:
            _size_button(btn, "action")
        except Exception as _e:
            logger.warning(f"[ERROR] Exception in studio_tab_builder.py: {_e}")
```

- [ ] **Step 2: Add a unit-style test**

Add to `tests/test_workbench_tab_views.py`:

```python
class TestStudioTabBuilderHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_app()

    def test_size_button_sets_minimum_action_size(self):
        from qgis.PyQt.QtWidgets import QPushButton
        from swe2d.workbench.views.studio_tab_builder import _size_button
        btn = QPushButton("Test")
        _size_button(btn, "action")
        self.assertGreaterEqual(btn.minimumSize().width(), 80)
        self.assertGreaterEqual(btn.minimumSize().height(), 28)
```

- [ ] **Step 3: Run and commit**

```bash
mamba run -n qgis_stable pytest tests/test_workbench_tab_views.py::TestStudioTabBuilderHelpers -v
git add swe2d/workbench/views/studio_tab_builder.py tests/test_workbench_tab_views.py
git commit -m "style(tab-builder): canonical button sizing helper"
```

---

## Phase B: Group Model tab parameters with `QGroupBox`

### Task B.1: Add QGroupBox infrastructure to `ModelTabView`

**Files:**
- Modify: `swe2d/workbench/views/model_tab_view.py:75-135`

- [ ] **Step 1: Update `_build_ui` to host filter/group state**

Replace:

```python
        self.model_toolbox = QtWidgets.QToolBox()
        self.model_toolbox.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding
        )
        root_layout.addWidget(self.model_toolbox)
```

with:

```python
        self._param_groups: List[QtWidgets.QGroupBox] = []
        self._param_rows: List[Tuple[QtWidgets.QGroupBox, QtWidgets.QLabel, QtWidgets.QWidget, bool]] = []
        self.model_toolbox = QtWidgets.QToolBox()
        self.model_toolbox.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding
        )
        root_layout.addWidget(self.model_toolbox)
```

Add the grouping helper inside `ModelTabView`:

```python
    def _start_param_group(
        self,
        page_layout: QtWidgets.QFormLayout,
        title: str,
        checkable: bool = False,
        advanced: bool = False,
    ) -> QtWidgets.QFormLayout:
        """Create a titled, collapsible group box and add it to a page."""
        group = QtWidgets.QGroupBox(title)
        group.setObjectName(title.lower().replace(" ", "_").replace("&", "and") + "_group")
        group.setCheckable(checkable)
        if checkable:
            group.setChecked(False)
        if advanced:
            group.setProperty("advanced", True)
        group_layout = QtWidgets.QFormLayout(group)
        group_layout.setObjectName(group.objectName() + "_layout")
        page_layout.addRow(group)
        self._param_groups.append(group)
        return group_layout

    def _add_param_row(
        self,
        group_layout: QtWidgets.QFormLayout,
        label_text: str,
        widget: QtWidgets.QWidget,
        advanced: bool = False,
    ) -> None:
        """Add a labeled widget to a group and register it for filtering."""
        label = QtWidgets.QLabel(label_text)
        group_layout.addRow(label, widget)
        group = group_layout.parentWidget()
        self._param_rows.append((group, label, widget, advanced))
        if advanced:
            widget.setProperty("advanced", True)
            label.setProperty("advanced", True)
```

Add the `HintLabel` class at module level:

```python
class HintLabel(QtWidgets.QLabel):
    """Small italic hint text under a parameter group."""
    def __init__(self, text: str = ""):
        super().__init__(text)
        self.setStyleSheet("color: #888; font-style: italic; padding-left: 12px;")
        self.setWordWrap(True)
```

- [ ] **Step 2: Commit the infrastructure before refactoring the forms**

```bash
mamba run -n qgis_stable pytest tests/test_model_tab_view.py -v
git add swe2d/workbench/views/model_tab_view.py
git commit -m "feat(model): QGroupBox infrastructure for parameter pages"
```

---

### Task B.2: Group Solver Parameters page

**Files:**
- Modify: `swe2d/workbench/views/model_tab_view.py:_build_solver_form_widgets`

- [ ] **Step 1: Replace `_build_solver_form_widgets` with grouped version**

Replace the entire `_build_solver_form_widgets` method with:

```python
    def _build_solver_form_widgets(self, param_form: QtWidgets.QFormLayout) -> None:
        """Populate the Solver Parameters page with grouped controls."""
        # -- Time Stepping --
        form = self._start_param_group(param_form, "Time Stepping")
        self._add_param_row(form, "CFL:", self.cfl_spin)
        self._add_param_row(form, "dt (max):", self.dt_spin)
        self._add_param_row(form, "Initial dt:", self.initial_dt_spin)
        self._add_param_row(form, "Variable timestep:", self.adaptive_cfl_dt_chk)
        form.addRow(HintLabel("Adaptive dt uses the CFL condition each step."))

        # -- Physics & Friction --
        form = self._start_param_group(param_form, "Physics & Friction")
        self._add_param_row(form, "Manning n:", self.n_mann_spin)
        self._add_param_row(form, "h_min:", self.h_min_spin)
        self._add_param_row(form, "Internal flow layer:", self.internal_flow_layer_combo)
        self._add_param_row(form, "Internal flow field:", self.internal_flow_field_edit)

        # -- Spatial Reconstruction --
        form = self._start_param_group(param_form, "Spatial Reconstruction")
        self._add_param_row(form, "Reconstruction:", self.reconstruction_combo)

        # -- Temporal Integration --
        form = self._start_param_group(param_form, "Temporal Integration")
        self._add_param_row(form, "Temporal discretization:", self.temporal_order_combo)

        # -- Initial Conditions --
        form = self._start_param_group(param_form, "Initial Conditions", checkable=True)
        self._add_param_row(form, "Initial condition:", self.initial_condition_combo)
        self._add_param_row(form, "Initial depth:", self.initial_depth_spin)
        self._add_param_row(form, "Initial WSE:", self.initial_wse_spin)
        form.addRow(HintLabel("Dry start uses bed elevation only."))

        # -- Numerical Options --
        form = self._start_param_group(param_form, "Numerical Options", advanced=True)
        self._add_param_row(form, "GPU diag sync (steps):", self.gpu_diag_sync_interval_spin, advanced=True)
        self._add_param_row(form, "Tiny mode:", self.tiny_mode_combo)
        self._add_param_row(form, "Tiny active/wet cell threshold:", self.tiny_wet_cell_threshold_spin)
        self._add_param_row(form, "Degenerate cell mode:", self.degen_mode_combo)

        # -- Performance --
        form = self._start_param_group(param_form, "Performance", advanced=True)
        self._add_param_row(form, "CUDA graph replay:", self.enable_cuda_graphs_chk, advanced=True)
        self._add_param_row(form, "SWE2D perf mode:", self.swe2d_perf_mode_chk, advanced=True)

        # -- Run Duration --
        form = self._start_param_group(param_form, "Run Duration")
        self._add_param_row(form, "Run duration:", self.run_time_edit)
```

Note: the widgets (`self.cfl_spin`, etc.) are still created in the method below. Keep the existing creation code, just change the `param_form.addRow(...)` calls to the grouped pattern above.

- [ ] **Step 2: Add tests for the groups**

Add to `tests/test_model_tab_view.py`:

```python
    def test_solver_page_has_group_boxes(self):
        view = self._make_view()
        page = view.findChild(QWidget, "model_solver_page")
        groups = page.findChildren(QtWidgets.QGroupBox)
        titles = {g.title() for g in groups}
        for expected in ("Time Stepping", "Physics & Friction", "Initial Conditions", "Run Duration"):
            self.assertIn(expected, titles)
```

- [ ] **Step 3: Run and commit**

```bash
mamba run -n qgis_stable pytest tests/test_model_tab_view.py -v
git add swe2d/workbench/views/model_tab_view.py tests/test_model_tab_view.py
git commit -m "feat(model): group Solver Parameters with QGroupBox"
```

---

### Task B.3: Group Stability Controls page

**Files:**
- Modify: `swe2d/workbench/views/model_tab_view.py:_build_stability_form_widgets`

- [ ] **Step 1: Refactor into three groups**

Replace `_build_stability_form_widgets` with:

```python
    def _build_stability_form_widgets(self, param_form: QtWidgets.QFormLayout) -> None:
        """Populate the Stability Controls page with grouped damping/cap controls."""
        form = self._start_param_group(param_form, "Wet/Dry Front")
        self._add_param_row(form, "Shallow damping depth:", self.shallow_damping_depth_spin)
        self._add_param_row(form, "Shallow-front recon fallback:", self.shallow_front_recon_fallback_chk)
        self._add_param_row(form, "Front flux damping:", self.front_flux_damping_spin)
        self._add_param_row(form, "Active-set hysteresis:", self.active_set_hysteresis_chk)

        form = self._start_param_group(param_form, "Capping", advanced=True)
        self._add_param_row(form, "Depth cap:", self.depth_cap_spin, advanced=True)
        self._add_param_row(form, "Momentum cap min speed:", self.momentum_cap_min_speed_spin, advanced=True)
        self._add_param_row(form, "Momentum cap celerity mult:", self.momentum_cap_celerity_mult_spin, advanced=True)

        form = self._start_param_group(param_form, "Solver Safety", advanced=True)
        self._add_param_row(form, "Max inv area:", self.max_inv_area_spin, advanced=True)
        self._add_param_row(form, "CFL lambda cap:", self.cfl_lambda_cap_spin, advanced=True)
```

- [ ] **Step 2: Test and commit**

```bash
mamba run -n qgis_stable pytest tests/test_model_tab_view.py tests/test_workbench_tab_views.py -v
git add swe2d/workbench/views/model_tab_view.py
git commit -m "feat(model): group Stability Controls with QGroupBox"
```

---

### Task B.4: Group Rain / Hydrology page

**Files:**
- Modify: `swe2d/workbench/views/model_tab_view.py:_build_rain_form_widgets`

- [ ] **Step 1: Refactor into three groups**

Replace `_build_rain_form_widgets` with:

```python
    def _build_rain_form_widgets(self, param_form: QtWidgets.QFormLayout) -> None:
        """Populate the Rain / Hydrology page with grouped rainfall controls."""
        form = self._start_param_group(param_form, "Rainfall Input")
        self._add_param_row(form, "Rain rate:", self.rain_rate_spin)
        self._add_param_row(form, "Spatial rainfall:", self.use_spatial_rain_cn_chk)
        self._add_param_row(form, "Rain rate update interval (s):", self.rain_update_interval_spin)
        self._add_param_row(form, "Storm area layer (optional):", self.storm_area_layer_combo)
        self._add_param_row(form, "Rain boundary buffer rings:", self.rain_boundary_buffer_rings_spin)

        form = self._start_param_group(param_form, "Infiltration")
        self._add_param_row(form, "Infiltration method:", self.infiltration_method_combo)
        self._add_param_row(form, "Default CN:", self.cn_default_spin)
        self._add_param_row(form, "SCS Ia/S ratio:", self.ia_ratio_spin)

        form = self._start_param_group(param_form, "Source Stability", advanced=True)
        self._add_param_row(form, "Max rel depth increase:", self.max_rel_depth_increase_spin, advanced=True)
        self._add_param_row(form, "Max source dh/step:", self.max_source_depth_step_spin, advanced=True)
        self._add_param_row(form, "Max source rate:", self.max_source_rate_spin, advanced=True)
        self._add_param_row(form, "Extreme rain mode:", self.extreme_rain_mode_chk, advanced=True)
        self._add_param_row(form, "Source CFL beta:", self.source_cfl_beta_spin, advanced=True)
        self._add_param_row(form, "Source max substeps:", self.source_max_substeps_spin, advanced=True)
        self._add_param_row(form, "True source subcycling:", self.source_true_subcycling_chk, advanced=True)
        self._add_param_row(form, "IMEX source split:", self.source_imex_split_chk, advanced=True)
```

- [ ] **Step 2: Test and commit**

```bash
mamba run -n qgis_stable pytest tests/test_model_tab_view.py -v
git add swe2d/workbench/views/model_tab_view.py
git commit -m "feat(model): group Rain/Hydrology with QGroupBox"
```

---

### Task B.5: Group Structures & Drainage page

**Files:**
- Modify: `swe2d/workbench/views/model_tab_view.py:_build_drain_form_widgets`

- [ ] **Step 1: Refactor into four groups**

Replace `_build_drain_form_widgets` with:

```python
    def _build_drain_form_widgets(self, param_form: QtWidgets.QFormLayout) -> None:
        """Populate the Structures & Drainage page with grouped coupling controls."""
        form = self._start_param_group(param_form, "Culvert / Bridge", advanced=True)
        self._add_param_row(form, "Coupling loop:", self.coupling_loop_combo)
        self._add_param_row(form, "Culvert solver mode:", self.culvert_solver_mode_combo, advanced=True)
        self._add_param_row(form, "Culvert coupling mode:", self.culvert_face_flux_chk, advanced=True)
        self._add_param_row(form, self.use_redistribution_chk.text(), self.use_redistribution_chk, advanced=True)
        self._add_param_row(form, "Bridge stacked coupling mode:", self.bridge_stacked_coupling_mode_combo, advanced=True)

        form = self._start_param_group(param_form, "Drainage Network — Equation Set", advanced=True)
        self._add_param_row(form, "Drainage equation set:", self.drainage_solver_mode_combo, advanced=True)
        self._add_param_row(form, "Drainage GPU method:", self.drainage_gpu_method_combo, advanced=True)

        form = self._start_param_group(param_form, "Drainage — Substepping", advanced=True)
        self._add_param_row(form, "Drainage substeps:", self.drainage_coupling_substeps_spin, advanced=True)
        self._add_param_row(form, "Drainage max adaptive substeps:", self.drainage_max_coupling_substeps_spin, advanced=True)

        form = self._start_param_group(param_form, "Drainage — Stability", advanced=True)
        self._add_param_row(form, "Drainage head deadband:", self.drainage_head_deadband_spin, advanced=True)
        self._add_param_row(form, "Drainage dynamic relaxation:", self.drainage_dynamic_relaxation_spin, advanced=True)
        self._add_param_row(form, "Drainage adaptive depth fraction:", self.drainage_adaptive_depth_fraction_spin, advanced=True)
        self._add_param_row(form, "Drainage adaptive wave Courant:", self.drainage_adaptive_wave_courant_spin, advanced=True)
        self._add_param_row(form, "Drainage implicit iterations (GPU):", self.drainage_implicit_iters_spin, advanced=True)
        self._add_param_row(form, "Drainage implicit relaxation (GPU):", self.drainage_implicit_relax_spin, advanced=True)

        param_form.addRow(self.gpu_default_lbl)
        param_form.addRow(self.unit_system_lbl)
```

- [ ] **Step 2: Test and commit**

```bash
mamba run -n qgis_stable pytest tests/test_model_tab_view.py -v
git add swe2d/workbench/views/model_tab_view.py
git commit -m "feat(model): group Structures & Drainage with QGroupBox"
```

---

## Phase C: Global toolbar and plugin menu

### Task C.1: Build toolbar and menu in `studio_host_methods.py`

**Files:**
- Modify: `swe2d/workbench/views/studio_host_methods.py`
- Create: `swe2d/workbench/dialogs/workbench_settings_dialog.py` (used by the menu)
- Modify: `swe2d/workbench/studio_dialog.py` (handlers for Settings and Help)

- [ ] **Step 1: Create the Settings dialog**

Create `swe2d/workbench/dialogs/workbench_settings_dialog.py`:

```python
"""Small settings dialog for HYDRA workbench feature flags."""
from __future__ import annotations

from typing import Dict

from qgis.PyQt import QtWidgets


class WorkbenchSettingsDialog(QtWidgets.QDialog):
    """Let the user toggle workbench module feature flags."""

    _FLAGS = [
        ("rainfall", "Enable rainfall module"),
        ("drainage_structures", "Enable drainage networks"),
        ("hydraulic_structures", "Enable hydraulic structures (weirs, culverts, bridges)"),
        ("bridge_stacked_coupling", "Enable bridge stacked coupling (experimental)"),
    ]

    def __init__(self, feature_flags: Dict[str, bool], parent=None):
        super().__init__(parent)
        self.setWindowTitle("HYDRA Settings")
        self._checks: Dict[str, QtWidgets.QCheckBox] = {}
        layout = QtWidgets.QVBoxLayout(self)
        for key, label in self._FLAGS:
            chk = QtWidgets.QCheckBox(label)
            chk.setChecked(feature_flags.get(key, True))
            layout.addWidget(chk)
            self._checks[key] = chk
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def flags(self) -> Dict[str, bool]:
        """Return the updated feature flag map."""
        return {key: chk.isChecked() for key, chk in self._checks.items()}
```

- [ ] **Step 2: Add menu/toolbar builders and helpers**

Add to `swe2d/workbench/views/studio_host_methods.py` after `_clear_studio_host_controls`:

```python
def _build_studio_toolbar(iface_obj, dlg, host_window):
    """Create the HYDRA run toolbar in the QGIS main window."""
    global _SWE2D_STUDIO_HOST_TOOLBAR
    toolbar = QtWidgets.QToolBar("HYDRA Run", host_window)
    toolbar.setObjectName("HydraRunToolbar")
    toolbar.setIconSize(QtCore.QSize(24, 24))
    actions = [
        ("media-playback-start", "Run 2D Model", lambda: dlg._controller.on_run(), True),
        ("media-playback-stop", "Cancel Run", lambda: dlg._controller.on_cancel(), False),
        ("camera-photo", "Take Snapshot", lambda: dlg._controller.on_snapshot(), True),
        None,
        ("applications-utilities", "Batch Simulation…", lambda: dlg._open_batch_simulation_dialog(), True),
        None,
        ("folder-open", "Open GeoPackage…", lambda: dlg._mesh_controller.load_2d_model_geopackage(), True),
        ("document-open", "Open Run Log", lambda: dlg._open_run_log_viewer(), True),
        ("folder", "GPKG Explorer…", lambda: dlg._open_model_gpkg_explorer(), True),
        None,
        ("view-refresh", "Discover Runs", lambda: dlg._on_results_refresh(), False),
        ("document-save-as", "Export Overlay TIFF…", lambda: dlg._overlay_controller.export_high_perf_overlay_to_geotiff(), False),
    ]
    for spec in actions:
        if spec is None:
            toolbar.addSeparator()
            continue
        icon_name, text, cb, enabled = spec
        action = QtWidgets.QAction(text, host_window)
        icon = QtGui.QIcon.fromTheme(icon_name)
        if not icon.isNull():
            action.setIcon(icon)
        action.setEnabled(enabled)
        action.triggered.connect(cb)
        toolbar.addAction(action)
    host_window.addToolBar(QtCore.Qt.TopToolBarArea, toolbar)
    _SWE2D_STUDIO_HOST_TOOLBAR = toolbar


def _build_studio_menu(iface_obj, dlg, host_window):
    """Add the HYDRA submenu under QGIS Plugins menu."""
    global _SWE2D_STUDIO_HOST_MENU
    menu_bar = host_window.menuBar()
    if menu_bar is None:
        return
    plugins_menu = None
    for action in menu_bar.actions():
        menu = action.menu()
        if menu is not None and str(menu.title()).replace("&", "").strip().lower() == "plugins":
            plugins_menu = menu
            break
    if plugins_menu is None:
        plugins_menu = menu_bar.addMenu("Plugins")
    hydra_menu = plugins_menu.addMenu("HYDRA")
    hydra_menu.setObjectName("HydraPluginMenu")
    _SWE2D_STUDIO_HOST_MENU = hydra_menu

    def add_action(text, cb, shortcut=None):
        act = QtWidgets.QAction(text, host_window)
        act.triggered.connect(cb)
        if shortcut:
            act.setShortcut(QtGui.QKeySequence(shortcut))
        hydra_menu.addAction(act)
        return act

    add_action("Open Workbench", lambda: dlg.show())
    hydra_menu.addSeparator()
    recent_menu = hydra_menu.addMenu("Recent Model GeoPackages")

    def _refresh_recent():
        recent_menu.clear()
        paths = list(getattr(dlg, "_recent_model_gpkgs", []))[:5]
        for p in paths:
            recent_menu.addAction(p, lambda path=p: dlg._mesh_controller.load_2d_model_geopackage(path_override=path))
        if not paths:
            no_item = recent_menu.addAction("(no recent files)")
            no_item.setEnabled(False)

    recent_menu.aboutToShow.connect(_refresh_recent)
    hydra_menu.addSeparator()
    add_action("Run Last Simulation", lambda: dlg._controller.on_run(), "Ctrl+R")
    add_action("Batch Simulation…", lambda: dlg._open_batch_simulation_dialog(), "Ctrl+B")
    add_action("Open Run Log", lambda: dlg._open_run_log_viewer())
    add_action("Open GeoPackage Explorer", lambda: dlg._open_model_gpkg_explorer())
    hydra_menu.addSeparator()
    add_action("Export Current Results as GeoTIFF…", lambda: dlg._overlay_controller.export_high_perf_overlay_to_geotiff())
    hydra_menu.addSeparator()
    add_action("Settings…", lambda: dlg._open_workbench_settings())
    add_action("Help → Documentation Hub", lambda: dlg._open_documentation_hub())
```

Add the needed imports at the top of `studio_host_methods.py`:

```python
from qgis.PyQt import QtCore, QtGui, QtWidgets
```

- [ ] **Step 3: Add dialog handlers and recent-file tracking**

In `swe2d/workbench/studio_dialog.py`, add:

```python
    def _open_workbench_settings(self) -> None:
        """Open the workbench feature-flag settings dialog."""
        from swe2d.workbench.dialogs.workbench_settings_dialog import WorkbenchSettingsDialog
        dlg = WorkbenchSettingsDialog(self._state.studio_feature_flags, parent=self)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            for key, value in dlg.flags().items():
                self._studio_set_feature_enabled(key, value)

    def _open_documentation_hub(self) -> None:
        """Focus the inspector dock on the Help tab."""
        dock = getattr(self._state, "studio_inspector_dock", None)
        if dock is None:
            return
        tabs = dock.findChild(QtWidgets.QTabWidget)
        if tabs is not None:
            for i in range(tabs.count()):
                if "Help" in tabs.tabText(i):
                    tabs.setCurrentIndex(i)
                    break
        dock.setVisible(True)
        dock.raise_()

    def _remember_model_gpkg(self, path: str) -> None:
        """Track recently opened model GeoPackages for the Plugins menu."""
        if not path:
            return
        if not hasattr(self, "_recent_model_gpkgs"):
            self._recent_model_gpkgs = []
        p = str(path)
        if p in self._recent_model_gpkgs:
            self._recent_model_gpkgs.remove(p)
        self._recent_model_gpkgs.insert(0, p)
        self._recent_model_gpkgs = self._recent_model_gpkgs[:5]
```

Update `_load_2d_model_geopackage` to record the path:

```python
    def _load_2d_model_geopackage(self, path_override=None) -> None:
        """Load a model GeoPackage and update recent files."""
        self._mesh_controller.load_2d_model_geopackage(path_override=path_override)
        path = path_override or getattr(self, "_model_gpkg_path", "")
        if path:
            self._remember_model_gpkg(path)
```

- [ ] **Step 4: Wire builders into host control installation**

In `_install_studio_host_controls`, after the view-mode combo block, add:

```python
    try:
        _build_studio_toolbar(iface_obj, dlg, host_window)
    except Exception as e:
        logger_wb.warning("[ERROR] toolbar install failed: %s", e)
    try:
        _build_studio_menu(iface_obj, dlg, host_window)
    except Exception as e:
        logger_wb.warning("[ERROR] menu install failed: %s", e)
```

- [ ] **Step 5: Add tests**

Add to `tests/test_workbench_dialog_builder.py`:

```python
    def test_settings_dialog_imports(self):
        from swe2d.workbench.dialogs.workbench_settings_dialog import WorkbenchSettingsDialog
        self.assertIsNotNone(WorkbenchSettingsDialog)

    def test_settings_dialog_returns_flags(self):
        from swe2d.workbench.dialogs.workbench_settings_dialog import WorkbenchSettingsDialog
        _ensure_app()
        dlg = WorkbenchSettingsDialog({"rainfall": True, "drainage_structures": False})
        flags = dlg.flags()
        self.assertTrue(flags["rainfall"])
        self.assertFalse(flags["drainage_structures"])
```

- [ ] **Step 6: Run and commit**

```bash
mamba run -n qgis_stable pytest tests/test_workbench_dialog_builder.py tests/test_workbench_tab_views.py -v
git add swe2d/workbench/views/studio_host_methods.py swe2d/workbench/dialogs/workbench_settings_dialog.py swe2d/workbench/studio_dialog.py tests/test_workbench_dialog_builder.py
git commit -m "feat(workbench): HYDRA toolbar and Plugins menu"
```

---

## Phase D: Parameter search filter and advanced toggle

### Task D.1: Add filter bar and advanced toggle to Model tab

**Files:**
- Modify: `swe2d/workbench/views/model_tab_view.py:75-135` and filtering helpers
- Modify: `tests/test_model_tab_view.py`

- [ ] **Step 1: Add the filter UI**

In `_build_ui`, before adding `model_toolbox` to the root layout, add:

```python
        filter_bar = QtWidgets.QHBoxLayout()
        self.param_search = QtWidgets.QLineEdit()
        self.param_search.setObjectName("param_search")
        self.param_search.setPlaceholderText("Filter parameters…")
        self.param_search.textChanged.connect(self._filter_model_tab)
        self.show_advanced_chk = QtWidgets.QCheckBox("Show advanced parameters")
        self.show_advanced_chk.setObjectName("show_advanced_chk")
        self.show_advanced_chk.setChecked(False)
        self.show_advanced_chk.toggled.connect(self._filter_model_tab)
        filter_bar.addWidget(self.param_search, 1)
        filter_bar.addWidget(self.show_advanced_chk)
        root_layout.addLayout(filter_bar)
```

- [ ] **Step 2: Add the filter implementation**

Add methods to `ModelTabView`:

```python
    def _filter_model_tab(self, _value=None) -> None:
        """Show/hide parameter rows based on search text and advanced toggle."""
        text = self.param_search.text().lower().strip()
        show_advanced = self.show_advanced_chk.isChecked()
        group_visibility: Dict[QtWidgets.QGroupBox, bool] = {}
        for group, label, widget, advanced in self._param_rows:
            label_text = label.text().lower()
            tooltip = (widget.toolTip() or "").lower()
            obj_name = widget.objectName().lower()
            matches = (not text) or (text in label_text) or (text in tooltip) or (text in obj_name)
            visible = matches and (show_advanced or not advanced)
            label.setVisible(visible)
            widget.setVisible(visible)
            group_visibility[group] = group_visibility.get(group, False) or visible
        for group, visible in group_visibility.items():
            group.setVisible(visible)
```

- [ ] **Step 3: Add tests**

Add to `tests/test_model_tab_view.py`:

```python
    def test_model_tab_has_search_filter(self):
        view = self._make_view()
        self.assertIsInstance(view.param_search, QLineEdit)

    def test_filter_hides_non_matching_rows(self):
        from qgis.PyQt.QtWidgets import QGroupBox
        view = self._make_view()
        view.show_advanced_chk.setChecked(True)
        view.param_search.setText("cfl")
        view._filter_model_tab()
        visible_groups = [g for g in view._param_groups if g.isVisible()]
        self.assertGreater(len(visible_groups), 0)
        self.assertTrue(all("cfl" in g.title().lower() or any(
            "cfl" in (w.toolTip() or "").lower() or "cfl" in w.objectName().lower()
            for _, _, w, _ in view._param_rows if _g is g)
            for _g in visible_groups))
```

- [ ] **Step 4: Run and commit**

```bash
mamba run -n qgis_stable pytest tests/test_model_tab_view.py -v
git add swe2d/workbench/views/model_tab_view.py tests/test_model_tab_view.py
git commit -m "feat(model): parameter search filter and advanced toggle"
```

---

## Phase E: Reorganize Results dock

### Task E.1: Collapse Results toolbox to Display + Storage pages

**Files:**
- Modify: `swe2d/workbench/views/results_controls.py`
- Modify: `tests/test_workbench_tab_views.py`

- [ ] **Step 1: Replace page builders**

In `_build_ui`, replace:

```python
        self._build_overlay_page(self._toolbox)
        self._build_output_page(self._toolbox)
        self._build_runs_page(self._toolbox)
```

with:

```python
        self._build_display_page(self._toolbox)
        self._build_storage_page(self._toolbox)
```

Rename `_build_overlay_page` to `_build_display_page`. At the end, move the Runs section into this page.

Add helper for child enabling:

```python
    def _wire_checkbox_children(self, chk, children):
        """Enable/disable *children* when *chk* is toggled."""
        def _update(enabled):
            for child in children:
                child.setEnabled(enabled)
        chk.toggled.connect(_update)
        _update(chk.isChecked())
```

Use it in the display page after creating arrow/streamline widgets:

```python
        self._wire_checkbox_children(
            self.arrows_chk,
            [self.arrow_density_spin, self.arrow_length_spin,
             self.arrow_head_length_spin, self.arrow_head_width_spin]
        )
        self._wire_checkbox_children(
            self.streamlines_chk,
            [self.streamline_backend_combo, self.streamline_seed_spin, self.streamline_steps_spin]
        )
```

Wrap the display page content in `QGroupBox` sections named `Field & Colormap`, `Color Range`, `Overlay Style`, and `Runs`.

Keep `_build_output_page` renamed to `_build_storage_page` and its page title `Storage`.

- [ ] **Step 2: Add tests**

Add to `tests/test_workbench_tab_views.py` in `TestResultsToolbox`:

```python
    def test_results_toolbox_has_two_pages(self):
        from swe2d.workbench.views.results_controls import ResultsToolbox
        toolbox = ResultsToolbox()
        self.assertEqual(toolbox.toolbox.count(), 2)
        texts = [toolbox.toolbox.itemText(i) for i in range(toolbox.toolbox.count())]
        self.assertIn("Display", texts)
        self.assertIn("Storage", texts)

    def test_arrow_children_disable_with_checkbox(self):
        from swe2d.workbench.views.results_controls import ResultsToolbox
        toolbox = ResultsToolbox()
        toolbox.arrows_chk.setChecked(True)
        self.assertTrue(toolbox.arrow_density_spin.isEnabled())
        toolbox.arrows_chk.setChecked(False)
        self.assertFalse(toolbox.arrow_density_spin.isEnabled())
```

- [ ] **Step 3: Run and commit**

```bash
mamba run -n qgis_stable pytest tests/test_workbench_tab_views.py::TestResultsToolbox -v
git add swe2d/workbench/views/results_controls.py tests/test_workbench_tab_views.py
git commit -m "feat(results): Display + Storage pages and checkbox child enabling"
```

---

## Phase F: Dedicated Run dock

### Task F.1: Create `RunDockWidget`

**Files:**
- Create: `swe2d/workbench/views/run_dock.py`
- Modify: `swe2d/workbench/workbench_dialog_builder.py`
- Modify: `swe2d/workbench/views/studio_tab_builder.py` (wiring)
- Modify: `swe2d/workbench/views/model_tab_view.py` (remove visible run controls)
- Modify: `swe2d/workbench/studio_dialog.py` (view protocol updates)
- Modify: `tests/test_model_tab_view.py` and `tests/test_workbench_dialog_builder.py`

- [ ] **Step 1: Create the new view**

Create `swe2d/workbench/views/run_dock.py`:

```python
"""Dedicated Run dock for the HYDRA2D workbench."""
from __future__ import annotations

from qgis.PyQt import QtWidgets


class RunDockWidget(QtWidgets.QWidget):
    """Bottom dock with Run/Cancel/Snapshot/Batch and a progress bar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(4)

        row = QtWidgets.QHBoxLayout()
        self.run_btn = QtWidgets.QPushButton("▶ Run 2D Model")
        self.run_btn.setObjectName("run_btn")
        self.cancel_btn = QtWidgets.QPushButton("⏹ Cancel")
        self.cancel_btn.setObjectName("cancel_btn")
        self.cancel_btn.setEnabled(False)
        self.snapshot_btn = QtWidgets.QPushButton("📸 Snapshot")
        self.snapshot_btn.setObjectName("snapshot_btn")
        self.batch_btn = QtWidgets.QPushButton("Batch…")
        self.batch_btn.setObjectName("batch_btn")

        row.addWidget(self.run_btn)
        row.addWidget(self.cancel_btn)
        row.addWidget(self.snapshot_btn)
        row.addStretch(1)
        row.addWidget(self.batch_btn)
        layout.addLayout(row)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setObjectName("progress_bar")
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
```

- [ ] **Step 2: Register the dock**

In `WorkbenchDialogBuilder._build_dialog_ui`, after the log component:

```python
        from swe2d.workbench.views.run_dock import RunDockWidget
        dlg._run_dock = RunDockWidget()
        self._build_component(
            name="run",
            title="HYDRA Run",
            area=QtCore.Qt.BottomDockWidgetArea,
            populate=lambda dock: dock.setWidget(dlg._run_dock),
            iface=dlg.iface,
        )
```

- [ ] **Step 3: Wire Run dock buttons**

In `swe2d/workbench/views/studio_tab_builder.py`, add after `wire_run_tab_signals`:

```python
def wire_run_dock_signals(dialog) -> None:
    """Wire the Run dock buttons to the same handlers as the Model tab."""
    from swe2d.workbench.signal_helpers import safe_disconnect
    d = dialog._run_dock
    safe_disconnect(d.run_btn.clicked, dialog._controller.on_run)
    d.run_btn.clicked.connect(dialog._controller.on_run)
    safe_disconnect(d.cancel_btn.clicked, dialog._controller.on_cancel)
    d.cancel_btn.clicked.connect(dialog._controller.on_cancel)
    safe_disconnect(d.snapshot_btn.clicked, dialog._controller.on_snapshot)
    d.snapshot_btn.clicked.connect(dialog._controller.on_snapshot)
    safe_disconnect(d.batch_btn.clicked, dialog._open_batch_simulation_dialog)
    d.batch_btn.clicked.connect(dialog._open_batch_simulation_dialog)
```

Call it in `build_model_tab_page` after `wire_run_tab_signals`:

```python
    wire_run_tab_signals(dialog)
    wire_run_dock_signals(dialog)
```

- [ ] **Step 4: Keep Model tab attributes but remove visible run controls**

In `swe2d/workbench/views/model_tab_view.py`, modify `_build_run_page_widgets` so the run row, progress bar, and debug rows are still created as attributes but not added to `run_page_layout`. Only the output interval row and the storage/load row should remain visible.

Change the beginning of the method so:

```python
        for attr, text, tip in [
            ("run_btn", "Run 2D Model", ...),
            ("batch_sim_btn", "Batch Simulation...", ...),
            ("cancel_btn", "Cancel", ...),
        ]:
            btn = QtWidgets.QPushButton(text)
            ...
            setattr(self, attr, btn)
            # Intentionally not added to run_page_layout — these live in the Run dock now.

        self.progress_bar = QtWidgets.QProgressBar()
        ...
        # Intentionally not added to run_page_layout.
```

Keep the snapshot/preview buttons as attributes but do not add `debug_row` to `run_page_layout`. The output interval row and the GPKG/storage row remain.

- [ ] **Step 5: Update view protocol methods**

In `swe2d/workbench/studio_dialog.py`, update:

```python
    def set_run_button_enabled(self, enabled: bool) -> None:
        """Enable or disable the Run button."""
        for widget in (getattr(self._run_dock, "run_btn", None),
                       getattr(self._model_tab_view, "run_btn", None)):
            if widget is not None:
                widget.setEnabled(enabled)

    def set_cancel_button_enabled(self, enabled: bool) -> None:
        """Enable or disable the Cancel button."""
        for widget in (getattr(self._run_dock, "cancel_btn", None),
                       getattr(self._model_tab_view, "cancel_btn", None)):
            if widget is not None:
                widget.setEnabled(enabled)

    def set_run_progress(self, value: int) -> None:
        """Set the run progress bar value."""
        for widget in (getattr(self._run_dock, "progress_bar", None),
                       getattr(self._model_tab_view, "progress_bar", None)):
            if widget is not None:
                widget.setValue(value)
```

- [ ] **Step 6: Add tests**

Add to `tests/test_workbench_dialog_builder.py`:

```python
    def test_run_dock_is_created(self):
        from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog
        dlg = SWE2DWorkbenchStudioDialog(iface=MagicMock())
        try:
            self.assertTrue(hasattr(dlg, "_run_dock"))
        finally:
            dlg.close()
```

Add to `tests/test_model_tab_view.py`:

```python
    def test_run_controls_still_exist_as_attributes(self):
        view = self._make_view()
        self.assertIsInstance(view.run_btn, QPushButton)
        self.assertIsInstance(view.cancel_btn, QPushButton)
```

- [ ] **Step 7: Run and commit**

```bash
mamba run -n qgis_stable pytest tests/test_model_tab_view.py tests/test_workbench_dialog_builder.py tests/test_workbench_tab_views.py -v
git add swe2d/workbench/views/run_dock.py swe2d/workbench/workbench_dialog_builder.py swe2d/workbench/views/studio_tab_builder.py swe2d/workbench/views/model_tab_view.py swe2d/workbench/studio_dialog.py tests/test_model_tab_view.py tests/test_workbench_dialog_builder.py
git commit -m "feat(workbench): dedicated Run dock"
```

---

## Phase G: Feature flags settings dialog

### Task G.1: Extend feature flags and keyword map

**Files:**
- Modify: `swe2d/workbench/workbench_view_state.py`
- Modify: `swe2d/workbench/studio_dialog.py:_studio_feature_keywords`
- Modify: `tests/test_workbench_dialog_builder.py`

- [ ] **Step 1: Add new default flags**

In `swe2d/workbench/workbench_view_state.py`:

```python
_STUDIO_DEFAULT_FEATURE_FLAGS: Dict[str, bool] = {
    "rainfall": True,
    "drainage_structures": True,
    "hydraulic_structures": True,
    "bridge_stacked_coupling": False,
}
```

- [ ] **Step 2: Update the keyword map**

In `swe2d/workbench/studio_dialog.py`, replace `_studio_feature_keywords` with:

```python
    def _studio_feature_keywords(self) -> Dict[str, Tuple[str, ...]]:
        """Return keyword mappings from feature flags to widget text patterns."""
        return {
            "rainfall": ("rain", "gauge", "hyet", "storm", "runoff", "precip"),
            "drainage_structures": (
                "drain", "node", "link", "inlet", "outfall", "pipe", "network",
            ),
            "hydraulic_structures": (
                "structure", "culvert", "weir", "orifice", "gate", "spillway",
            ),
            "bridge_stacked_coupling": ("bridge_stacked",),
        }
```

- [ ] **Step 3: Add a test**

Add to `tests/test_workbench_dialog_builder.py`:

```python
    def test_feature_flags_include_new_keys(self):
        from swe2d.workbench.workbench_view_state import _STUDIO_DEFAULT_FEATURE_FLAGS
        self.assertIn("hydraulic_structures", _STUDIO_DEFAULT_FEATURE_FLAGS)
        self.assertIn("bridge_stacked_coupling", _STUDIO_DEFAULT_FEATURE_FLAGS)
```

- [ ] **Step 4: Run and commit**

```bash
mamba run -n qgis_stable pytest tests/test_workbench_dialog_builder.py -v
git add swe2d/workbench/workbench_view_state.py swe2d/workbench/studio_dialog.py tests/test_workbench_dialog_builder.py
git commit -m "feat(settings): split hydraulic/bridge feature flags"
```

---

## Phase H: Inspector clarity

### Task H.1: Rename inspector tabs and add grouping

**Files:**
- Modify: `swe2d/workbench/workbench_dialog_builder.py:_populate_inspector_dock`

- [ ] **Step 1: Rename tabs and add a lock indicator**

Replace:

```python
        inspector_tabs.addTab(model_page, "Model Settings")
```

with:

```python
        inspector_tabs.addTab(model_page, "\U0001F512 Parameters (read-only)")
```

and:

```python
        inspector_tabs.addTab(mesh_page, "Mesh Settings")
```

with:

```python
        inspector_tabs.addTab(mesh_page, "\U0001F512 Mesh Settings (read-only)")
```

- [ ] **Step 2: Add a test**

Add to `tests/test_workbench_dialog_builder.py`:

```python
    def test_inspector_tabs_are_read_only_labelled(self):
        from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog
        dlg = SWE2DWorkbenchStudioDialog(iface=MagicMock())
        try:
            tabs = dlg._state.studio_inspector_dock.findChild(QtWidgets.QTabWidget)
            texts = [tabs.tabText(i) for i in range(tabs.count())]
            self.assertTrue(any("read-only" in t for t in texts))
        finally:
            dlg.close()
```

- [ ] **Step 3: Run and commit**

```bash
mamba run -n qgis_stable pytest tests/test_workbench_dialog_builder.py -v
git add swe2d/workbench/workbench_dialog_builder.py tests/test_workbench_dialog_builder.py
git commit -m "feat(inspector): read-only tab labels"
```

---

## Phase I: Doc search snippets

### Task I.1: Verify search snippets are surfaced

**Files:**
- Modify: `tests/test_workbench_tab_views.py`

- [ ] **Step 1: Add a test**

Add to `tests/test_workbench_tab_views.py`:

```python
class TestDocViewer(unittest.TestCase):
    def test_search_returns_snippets(self):
        from swe2d.workbench.views.doc_viewer import _search_all_docs
        results = _search_all_docs("solver")
        self.assertIsInstance(results, dict)
        for hits in results.values():
            for hit in hits:
                self.assertTrue(len(hit.snippet) > 0)
```

- [ ] **Step 2: Run and commit**

```bash
mamba run -n qgis_stable pytest tests/test_workbench_tab_views.py::TestDocViewer -v
git add tests/test_workbench_tab_views.py
git commit -m "test(doc-viewer): verify search snippets exist"
```

---

## Phase J: Run selection helpers

### Task J.1: Add Invert and Only-newest buttons

**Files:**
- Modify: `swe2d/results/queries.py`
- Modify: `swe2d/results/run_service.py`
- Modify: `swe2d/workbench/dialogs/run_selection_dialog.py`
- Modify: `tests/test_workbench_tab_views.py`

- [ ] **Step 1: Surface `created_utc` in run discovery**

In `swe2d/results/queries.py`, update `discover_line_result_runs`:

```python
        out.append({
            "run_id": r["run_id"],
            "table_ts": "swe2d_baked_line_ts",
            "table_profile": "swe2d_baked_line_profiles",
            "has_profile": r.get("has_lines", False),
            "created_utc": r.get("created_utc", ""),
        })
```

- [ ] **Step 2: Add `created_utc` to `RunRecord`**

In `swe2d/results/run_service.py`:

```python
@dataclasses.dataclass
class RunRecord:
    run_id: str
    gpkg_path: str
    color: Tuple[int, int, int]
    enabled: bool = True
    label: str = ""
    has_profile: bool = False
    created_utc: str = ""
```

and pass it in `collect_runs_from_gpkg`:

```python
                has_profile=bool(meta.get("has_profile", False)),
                created_utc=str(meta.get("created_utc", "")),
                label=f"{gpkg_short}:{rid}{suffix}",
```

- [ ] **Step 3: Add the buttons and logic**

In `swe2d/workbench/dialogs/run_selection_dialog.py`, add after the existing `Clear All` button:

```python
        invert_btn = QtWidgets.QPushButton("Invert")
        invert_btn.setToolTip("Toggle every run check state.")
        invert_btn.clicked.connect(self._invert_selection)
        newest_btn = QtWidgets.QPushButton("Only newest")
        newest_btn.setToolTip("Keep only the most recently created run checked.")
        newest_btn.clicked.connect(self._select_only_newest)
        btn_row.addWidget(invert_btn)
        btn_row.addWidget(newest_btn)
```

Add methods:

```python
    def _invert_selection(self):
        """Toggle every item's check state."""
        for i in range(self._list.count()):
            item = self._list.item(i)
            item.setCheckState(
                Qt.Checked if item.checkState() == Qt.Unchecked else Qt.Unchecked
            )

    def _select_only_newest(self):
        """Check only the run with the latest created_utc timestamp."""
        newest_idx = None
        newest_ts = ""
        for i in range(self._list.count()):
            item = self._list.item(i)
            item.setCheckState(Qt.Unchecked)
            rec = self._records[i]
            ts = str(rec.created_utc or "")
            if ts and (not newest_ts or ts > newest_ts):
                newest_ts = ts
                newest_idx = i
        if newest_idx is None:
            # Fallback: use the first item if no timestamps are present.
            newest_idx = 0
        self._list.item(newest_idx).setCheckState(Qt.Checked)
```

- [ ] **Step 4: Add tests**

Add to `tests/test_workbench_tab_views.py`:

```python
class TestRunSelectionDialog(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_app()

    def _fake_records(self):
        from collections import namedtuple
        Rec = namedtuple("Rec", ["run_id", "gpkg_path", "color", "enabled", "label", "key", "created_utc"])
        return [
            Rec("run_a", "/tmp/a.gpkg", (0, 0, 0), True, "run_a", "/tmp/a.gpkg::run_a", "2026-07-01T00:00:00"),
            Rec("run_b", "/tmp/a.gpkg", (0, 0, 0), True, "run_b", "/tmp/a.gpkg::run_b", "2026-07-02T00:00:00"),
        ]

    def test_invert_selection_toggles_all(self):
        from swe2d.workbench.dialogs.run_selection_dialog import RunSelectionDialog
        dlg = RunSelectionDialog(self._fake_records())
        dlg._select_all()
        dlg._invert_selection()
        self.assertEqual(dlg.selected_keys(), set())

    def test_only_newest_selects_latest(self):
        from swe2d.workbench.dialogs.run_selection_dialog import RunSelectionDialog
        dlg = RunSelectionDialog(self._fake_records())
        dlg._select_only_newest()
        self.assertEqual(dlg.selected_keys(), {"/tmp/a.gpkg::run_b"})
```

- [ ] **Step 5: Run and commit**

```bash
mamba run -n qgis_stable pytest tests/test_workbench_tab_views.py::TestRunSelectionDialog -v
git add swe2d/results/queries.py swe2d/results/run_service.py swe2d/workbench/dialogs/run_selection_dialog.py tests/test_workbench_tab_views.py
git commit -m "feat(run-selection): invert and only-newest helpers"
```

---

## Phase K: Map Mesh Setup grid fix

### Task K.1: Replace column-cycling grid with a form layout

**Files:**
- Modify: `swe2d/workbench/views/map_tab_view.py:_build_actions_page`
- Modify: `tests/test_workbench_tab_views.py`

- [ ] **Step 1: Refactor the actions page**

Replace the grid-based actions layout with a `QFormLayout`:

```python
    def _build_actions_page(self, toolbox: QtWidgets.QToolBox) -> None:
        """Build the Mesh Setup page with mesh I/O and BC controls."""
        page = QtWidgets.QWidget()
        page.setObjectName("map_actions_page")
        actions_layout = QtWidgets.QFormLayout(page)
        actions_layout.setObjectName("map_actions_layout")
        actions_layout.setContentsMargins(4, 4, 4, 4)

        btn_specs = [
            ("load_model_gpkg_btn", "Load 2D Model GeoPackage"),
            ("export_mesh_layers_btn", "Export Mesh To Map Layers"),
            ("export_mesh_ugrid_btn", "Export Mesh To UGRID"),
            ("save_mesh_gpkg_btn", "Save Mesh to GPKG"),
            ("import_mesh_layers_btn", "Load Mesh From Selected Layers"),
            ("terrain_to_nodes_btn", "Assign Node Z From Terrain"),
            ("pull_node_z_btn", "Pull Node Z From Nodes Layer"),
            ("export_results_ugrid_btn", "Export Results to UGRID"),
            ("load_mesh_gpkg_btn", "Load Mesh from GPKG..."),
        ]
        for attr, text in btn_specs:
            btn = QtWidgets.QPushButton(text)
            btn.setObjectName(attr)
            setattr(self, attr, btn)
            actions_layout.addRow(btn)

        # ... keep existing tooltip assignments and BC controls below ...
```

Then add the BC controls with `actions_layout.addRow(...)` instead of grid positions.

- [ ] **Step 2: Add a test**

Add to `tests/test_workbench_tab_views.py`:

```python
    def test_map_actions_layout_is_form(self):
        from swe2d.workbench.views.map_tab_view import MapTabView
        view = MapTabView()
        from qgis.PyQt.QtWidgets import QFormLayout
        self.assertIsInstance(view.findChild(QFormLayout, "map_actions_layout"), QFormLayout)
```

- [ ] **Step 3: Run and commit**

```bash
mamba run -n qgis_stable pytest tests/test_workbench_tab_views.py::TestMapTabView -v
git add swe2d/workbench/views/map_tab_view.py tests/test_workbench_tab_views.py
git commit -m "fix(map): use QFormLayout for Mesh Setup buttons"
```

---

## Phase L: Coupling dialog splitter defaults

### Task L.1: Swap default splitter sizes

**Files:**
- Modify: `swe2d/workbench/dialogs/coupling_results_dialog.py:130`
- Modify: `tests/test_workbench_tab_views.py`

- [ ] **Step 1: Change the sizes**

Replace:

```python
        split.setSizes([380, 220])
```

with:

```python
        split.setSizes([220, 380])
```

- [ ] **Step 2: Add a regression test**

Add to `tests/test_workbench_tab_views.py`:

```python
class TestCouplingResultsDialog(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_app()

    def test_splitter_defaults_favor_plot(self):
        from swe2d.workbench.dialogs.coupling_results_dialog import SWE2DCouplingResultsViewerDialog
        dlg = SWE2DCouplingResultsViewerDialog([], "run", "/tmp/x.gpkg")
        try:
            sizes = dlg.findChild(QtWidgets.QSplitter).sizes()
            self.assertGreater(sizes[1], sizes[0])
        finally:
            dlg.close()
```

- [ ] **Step 3: Run and commit**

```bash
mamba run -n qgis_stable pytest tests/test_workbench_tab_views.py::TestCouplingResultsDialog -v
git add swe2d/workbench/dialogs/coupling_results_dialog.py tests/test_workbench_tab_views.py
git commit -m "fix(coupling-dialog): default splitter favors plot"
```

---

## Final validation

Run the full validation suite listed in the spec:

```bash
mamba run -n qgis_stable pytest \
  tests/test_workbench_tab_views.py \
  tests/test_workbench_dialog_builder.py \
  tests/test_model_tab_view.py \
  tests/test_topology_tab_view.py \
  tests/test_dialog_tab_views_integration.py \
  -q
```

Expected: all pass.

Purge Python caches before any QGIS restart:

```bash
find . -type d -name __pycache__ -exec rm -rf {} +
```

---

## Spec coverage self-review

| Spec section | Implementing task |
|---|---|
| 3.1 QGroupBox in Model tab | B.1–B.5 |
| 3.2 Toolbar | C.1 |
| 3.3 Plugins menu | C.1 |
| 3.4 Left dock relabel | A.4 |
| 3.5 Parameter search | D.1 |
| 3.6 Advanced toggle | D.1 |
| 3.7 Inspector read-only | H.1 |
| 3.8 Results reorg | E.1 |
| 3.9 Naming glitches (`&&`, combo labels) | A.1, A.2 |
| 3.10 Validation badges / inline help | B.1 (`HintLabel`) |
| 3.11 Run dock | F.1 |
| 3.12 Feature flags settings | C.1, G.1 |
| 3.13 Combo unit labels | A.2 |
| QT #2 Gmsh-only pages | A.1 |
| QT #3 Disable child widgets | E.1 |
| QT #4 Filter bar | D.1 |
| QT #5 Doc snippets | I.1 |
| QT #6 Run selection helpers | J.1 |
| QT #7 Map grid | K.1 |
| QT #8 Button sizing | A.6 |
| QT #9 Coupling splitter | L.1 |
| QT #10 Keyboard shortcuts | A.5 |

---

## Superpowers workflow

- **writing-plans** — used to produce this document.
- **subagent-driven-development** — recommended for task-by-task execution.
- **verification-before-completion** — run the full validation suite before claiming done.

---

## Auto-agent selector payload

```json
{
  "plan_id": "2026-07-03-gui-ux-improvements",
  "phases": [
    {"phase": "A", "theme": "polish", "agent": "python-pro", "model": "default"},
    {"phase": "B", "theme": "model-tab-grouping", "agent": "python-pro", "model": "default"},
    {"phase": "C", "theme": "toolbar-menu", "agent": "python-pro", "model": "default"},
    {"phase": "D", "theme": "search-filter", "agent": "python-pro", "model": "default"},
    {"phase": "E", "theme": "results-reorg", "agent": "python-pro", "model": "default"},
    {"phase": "F", "theme": "run-dock", "agent": "python-pro", "model": "default"},
    {"phase": "G", "theme": "feature-flags", "agent": "python-pro", "model": "default"},
    {"phase": "H", "theme": "inspector", "agent": "python-pro", "model": "default"},
    {"phase": "I", "theme": "doc-snippets", "agent": "python-pro", "model": "default"},
    {"phase": "J", "theme": "run-selection", "agent": "python-pro", "model": "default"},
    {"phase": "K", "theme": "map-grid", "agent": "python-pro", "model": "default"},
    {"phase": "L", "theme": "coupling-splitter", "agent": "python-pro", "model": "default"}
  ]
}
```
