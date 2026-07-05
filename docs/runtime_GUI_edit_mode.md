Excellent. This gives me everything I need. The view files are **structured Python with clear patterns** — pages, form layouts, group boxes, parameter rows. The model_tab_view.py has actually already been refactored with `_start_param_group` and `_add_param_row` helpers which is exactly the kind of structure an editor tool needs.

Let me write a realistic, honest design document.

---

# Design: Runtime GUI Editor for SWE2D

You're describing what Qt Designer does, but for Python source. Let me be straight about what's achievable and what isn't.

## What's actually possible

The repo's view files have a **very regular structure** that makes AST-based source rewriting tractable:

```python
# pattern repeated 80+ times in model_tab_view.py:
group_layout = self._start_param_group(self.model_solver_form, "Time Stepping")
self.cfl_spin = QtWidgets.QDoubleSpinBox()
self.cfl_spin.setObjectName("cfl_spin")
self._add_param_row(group_layout, "CFL:", self.cfl_spin)
```

And in `map_tab_view.py`, topology_tab_view.py, `results_controls.py` the patterns are equally repetitive: `QtWidgets.QGroupBox(title)`, `QFormLayout`, `setObjectName`, `addWidget`, `addRow`. The subagent confirmed **all 5 view files use a small set of Qt containers** (`QGroupBox`, `QFormLayout`, `QVBoxLayout`, `QHBoxLayout`, `QToolBox`, `QTabWidget`) and **no `.ui` files** — which means every widget's creation is one of ~10 recognizable AST patterns.

This makes a **round-tripping visual editor** feasible. The architecture:

```
┌─────────────────────────────────────────────────────────────┐
│  Edit Mode (toggle in HYDRA toolbar, dev-only)              │
│  ┌────────────────────┐ ┌────────────────────────────────┐  │
│  │  Outline tree      │ │  Live preview                  │  │
│  │  ▸ Setup           │ │  [real widgets, dimmed]        │  │
│  │    ▸ Load Layers   │ │                                │  │
│  │      ▾ grid        │ │                                │  │
│  │        • terrain   │ │                                │  │
│  │        • manning   │ │                                │  │
│  │    ▸ Mesh Setup    │ │                                │  │
│  │  ▸ Mesh Generation │ │                                │  │
│  │  ▸ Simulation      │ │                                │  │
│  └────────────────────┘ └────────────────────────────────┘  │
│  Property editor (right panel):                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ objectName:  cfl_spin            [rename]            │  │
│  │ label:       CFL:                [edit]              │  │
│  │ tooltip:     Courant number...   [edit]              │  │
│  │ group:       Time Stepping       [move ▾]            │  │
│  │ page:        Solver Parameters   [move ▾]            │  │
│  │ advanced:    ☐                   [toggle]            │  │
│  └──────────────────────────────────────────────────────┘  │
│  [Apply & Write Source]  [Reset]  [Diff vs HEAD]            │
└─────────────────────────────────────────────────────────────┘
```

When you click **Apply & Write Source**, it parses each affected view file with `ast.parse`, locates the widget by `objectName`, and rewrites the source tree in place. Then it offers to run `git diff` and `pytest` before you commit.

---

## The 4 capabilities the editor will support

### 1. Rename & retitle (lowest risk, highest value)

- Change a widget's `objectName` (must stay unique; editor validates)
- Change its label text (`"CFL:"` → `"CFL (0.01–0.99):"`)
- Change its `setToolTip(...)` content
- Change the title of any `QGroupBox`, `QToolBox` page, or `QTabWidget` tab

This alone fixes the `"Arcs && Interfaces"` glitch and the `Run time` vs `Run duration` mismatch, and is the most-edited thing in real design work.

### 2. Move between groups (medium risk)

- Drag a widget from "Time Stepping" group → "Numerical Options" group
- Drag a row in a `QToolBox` to reorder pages
- Add a new empty `QGroupBox` to a page (writes the `_start_param_group(...)` call in the right place)
- Delete an empty group (writes the corresponding `_start_param_group` removal)

### 3. Reorder & restructure within a page (medium risk)

- Reorder widgets in a `QFormLayout`
- Split a single column into two side-by-side groups
- Add new widgets: `QSpinBox`, `QDoubleSpinBox`, `QComboBox`, `QCheckBox`, `QLineEdit` — with auto-generated objectName from label

### 4. Page-level moves (medium-high risk)

- Move a widget from one `QToolBox` page to another (e.g. `max_rel_depth_increase_spin` from Rain → Stability)
- Move a widget from one view file to another (e.g. from model_tab_view.py → `results_controls.py`)

This is the most disruptive — it also requires updating the **`findChild`/`getattr` references** in `studio_tab_builder.py`, workbench_dialog_builder.py, widget_persistence_service.py, and the test files. The editor must emit a **changelog of all renamed/moved widgets** so the developer can update those references.

---

## What the editor will NOT do (out of scope)

- **Edit signal-slot connections.** Drag-to-wire signals is theoretically possible but adds a *lot* of complexity (you'd need to detect existing `connect(...)` calls, rewrite them, and verify nothing breaks). For now, signal wiring stays in the existing `studio_tab_builder.py` and the developer adds new connections manually after re-running the editor.

- **Edit non-form widgets.** The rich plot widgets in `studio_viewer_pg.py` / `studio_viewer_profile_pg.py` / `temporal_dock.py` are not form controls — they're custom `QWidget` subclasses with embedded matplotlib/pyqtgraph canvases. Don't try to edit them.

- **Edit the dock layout.** QGIS owns that, as we discussed.

- **Edit `signal_helpers.py`, controllers, services, or backend code.** The editor is GUI-only. Cross-cutting concerns (signal wiring, persistence, validation) are still maintained by hand.

- **Live in-process rewrite.** The editor **saves the source to disk** and requires a QGIS restart to see the new GUI. Auto-reloading Python modules inside a running Qt app is fragile.

---

## Architecture

### Files to add

```
swe2d/workbench/devtools/
    __init__.py
    editor_mode.py            # QAction, toggle button, mode controller
    outline_tree.py           # QTreeWidget showing widget hierarchy
    property_editor.py        # Property editor dock
    widget_inspector.py       # Click-to-select, hover highlight
    live_preview.py           # Dimmed widget tree during edit
    source_rewriter.py        # AST-based source writer
    ast_patterns.py           # Recognise: QGroupBox title=, setObjectName, _add_param_row
    validation.py             # objectName uniqueness, missing references
    git_diff_view.py          # Show proposed diff before write
    tests/
        test_source_rewriter.py
        test_ast_patterns.py
        test_validation.py
```

That's ~1500 lines. ~600 of those are in `source_rewriter.py` alone.

### How `source_rewriter.py` works

Each view file is parsed with `ast.parse`. The rewriter walks the AST looking for these patterns:

```python
# Pattern 1: QGroupBox creation with title
group = QtWidgets.QGroupBox(title)
# → ast.Call(func=QGroupBox, args=[title])

# Pattern 2: objectName assignment
self.cfl_spin.setObjectName("cfl_spin")
# → ast.Call(func=Attribute(setObjectName), args=[Constant("cfl_spin")])

# Pattern 3: helper-based row add (model_tab_view.py)
self._add_param_row(group_layout, "CFL:", self.cfl_spin)
# → ast.Call(func=Attribute(_add_param_row), args=[..., Constant("CFL:"), Attribute(cfl_spin)])

# Pattern 4: toolbox addItem
self.model_toolbox.addItem(page, "Solver Parameters")
# → ast.Call(func=Attribute(addItem), args=[..., Constant("Solver Parameters")])

# Pattern 5: direct addRow
form.addRow("CFL:", self.cfl_spin)
# → ast.Call(func=Attribute(addRow), args=[Constant("CFL:"), Attribute(cfl_spin)])
```

For each recognized pattern, the rewriter records `(objectName → AST node, parent node, source line)`. When the developer clicks "Apply & Write Source", the rewriter:

1. Walks the AST and patches the matching nodes (e.g. swap `Constant("CFL:")` for `Constant("CFL (0.01–0.99):")`)
2. Uses `ast.unparse()` to render the patched tree
3. Runs `black` (or just `autopep8`) on the result to keep formatting
4. Writes the file with a backup `.bak` next to it
5. Spawns `git diff` in a side panel so the dev can review

### Risk mitigation

Because we're rewriting source code, we MUST guard against silent corruption. The `validation.py` module runs these checks before any write:

- [ ] objectName uniqueness across the **whole workbench** (not just one file)
- [ ] every `findChild(QWidget, "name")` in studio_dialog.py / workbench_dialog_builder.py / tests still resolves to a widget that exists in the rewritten file
- [ ] no AST node has been orphaned (every widget assigned to `self.X` is still referenced)
- [ ] Python syntax check: `compile(new_src, path, 'exec')` succeeds

If any check fails, the editor refuses to write and shows the diff with the offending section highlighted.

### Mode toggle

The editor is gated behind a developer flag. Two options:

1. **Environment variable**: `SWE2D_DEVTOOLS=1` enables the menu item. Otherwise invisible.
2. **QSettings flag**: stored in `QSettings("HYDRA2DGPU", "HYDRA2DGPU")` as a checkbox the dev enables once.

I'd recommend **(1)** because it can't be accidentally enabled in a deployed plugin. A bonus: `import` of the devtools package only happens when the env var is set, so zero runtime cost in production.

---

## Workflow

```
1. Set SWE2D_DEVTOOLS=1
2. Launch QGIS, open HYDRA workbench
3. Click "Edit GUI" in the (dev-only) toolbar menu
4. Editor docks appear: Outline (left), Property Editor (right)
5. Live widgets dim; click a widget → selects it in Outline
6. Drag a widget in the Outline to a different group
7. Edit labels/tooltips in Property Editor
8. Click "Apply & Write Source"
9. Review diff in side panel
10. Confirm → writes .bak + new file
11. Restart QGIS
12. Run pytest, commit, push
```

The Python-import-and-restart step is unavoidable. We can make it slightly less painful with a "QGIS Restart" button that spawns a new `qgis` process and closes the current one, but that's a 20-line nicety.

---

## What I'd build first (incremental delivery)

| Sprint | Deliverable | Lines | Risk |
|---|---|---|---|
| 1 | **Read-only inspector mode**: outline tree + property panel showing current widget tree, no edits | ~400 | none |
| 2 | **Rename-only writes**: change `objectName`, label, tooltip, group title. Writes back via AST. Tests with round-trip diff. | ~600 | low |
| 3 | **Move within group**: reorder rows in a `QFormLayout`, drag between groups in same page | ~400 | low |
| 4 | **Add/delete widgets** + group boxes | ~400 | medium |
| 5 | **Cross-page moves** + changelog emitter for downstream references | ~300 | medium-high |

I'd ship sprints 1-2 in the first PR and gate sprint 3+ behind opt-in. The first two already let you fix `&&` glitches, rename `Run time` to `Run duration`, update tooltips, and reorder groups — which is the bulk of what a "visual editor" is actually used for day-to-day.

---

## Honest caveats

1. **AST round-tripping is lossy.** Comments inside modified functions may move. We can preserve comments with `ast.parse(..., type_comments=True, feature_version=(3,12))` and careful round-tripping, but it's never perfect. The `.bak` file is your safety net.

2. **The editor cannot fix pre-existing bugs in the source.** If model_tab_view.py has a typo, the editor won't catch it. It just rewrites the patterns it recognizes.

3. **Auto-generated objectNames will collide.** When you add a new `QCheckBox` for "Lock canvas", the editor will auto-suggest `lock_canvas_chk` — but if one already exists in another file, it'll fail the uniqueness check. The dev has to rename.

4. **QGIS restart cycle.** Every "Apply & Write" needs a QGIS restart to take effect. That's annoying. We can mitigate by spawning a `qgis` subprocess and watching for the old process to exit, but I haven't tested how reliable that is on different platforms.

5. **This is ~1500 lines of new code in a plugin.** That's substantial. But it's all in one new `devtools/` subpackage, easily removed if it doesn't earn its keep. Zero impact on production users.

---

## Decision points before I start coding

1. **Tool name**: `swe2d_devtools`, `hydra_designer`, `gui_editor`, `qt_designer_lite`?
2. **Toggle mechanism**: env var `SWE2D_DEVTOOLS=1`, or hidden menu triggered by Ctrl+Shift+D, or QSettings flag?
3. **Output format**: writes directly to source, OR writes to a `.patch` file you apply manually? The `.patch` approach is safer but adds friction.
4. **Scope of first sprint**: just the read-only inspector + rename, or also move-between-groups in sprint 1?

I'd default to: name = `swe2d_devtools`, env var = `SWE2D_DEVTOOLS=1`, writes to source with `.bak`, sprint 1 = read-only + rename. But I want your call before I write 1500 lines.