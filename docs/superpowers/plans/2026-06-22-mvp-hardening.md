# MVP Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 3 MVP architecture violations: missing protocol method on View, callback anti-pattern passing mesh arrays through View, and plotting code mixed into Qt-dependent View layer.

**Architecture:** MVP with 3 layers: View (Qt widgets + protocol methods), Controller (orchestration only, no numpy/Qt widget access), Service (pure Python, zero Qt).

**Tech Stack:** Python, PyQt5, numpy, matplotlib

---

### Task 1: Add missing protocol method (controller widget boundary)

The controller at `swe2d/workbench/controllers/run_controller.py:1098-1099` reaches through the View to access a Qt spinbox directly — violates Rule 1.

**Files:**
- Modify: `swe2d/workbench/studio_dialog.py`
- Modify: `swe2d/workbench/controllers/run_controller.py`

- [ ] **Step 1: Add `get_n_mann_value()` protocol method to View**

In `swe2d/workbench/studio_dialog.py`, add to the `SWE2DWorkbenchStudioDialog` class in the `# ── View helper methods (read widget values, delegate to services) ──` section (after line 2278):

```python
def get_n_mann_value(self) -> float:
    mtv = getattr(self, "_model_tab_view", None)
    if mtv is not None:
        spin = getattr(mtv, "n_mann_spin", None)
        if spin is not None:
            return float(spin.value())
    return 0.03
```

- [ ] **Step 2: Update controller to use the protocol method**

In `swe2d/workbench/controllers/run_controller.py:1098-1099`, replace:

```python
mann_range = f"{float(view._model_tab_view.n_mann_spin.value()):.5f}"
```

with:

```python
mann_range = f"{view.get_n_mann_value():.5f}"
```

- [ ] **Step 3: Verify compliance**

```bash
! grep -n 'view\._model_tab_view\.\|view\._map_tab_view\.\|view\._topo_widgets\.' \
  swe2d/workbench/controllers/run_controller.py | grep -v "getattr\|is None" \
  && echo "PASS" || echo "VIOLATIONS REMAIN"
```

Expected: PASS

---

### Task 2: Fix callback anti-pattern (mesh arrays through View)

Service `collect_boundary_arrays()` in `boundary_runtime_logic.py` takes a `default_bc_for_edges_fn: Callable[[ndarray, ndarray], ...]` callback. The View's `_default_bc_for_edges` receives mesh arrays (`edge_n0`, `edge_n1`).

**Fix:** Change the service to accept `default_bc_type: int` instead of the callback. The service calls `_mesh_svc.default_bc_for_edges` internally. The View reads the widget and passes the int.

**Files:**
- Modify: `swe2d/boundary_and_forcing/boundary_runtime_logic.py`
- Modify: `swe2d/workbench/studio_dialog.py`
- Modify: `swe2d/workbench/controllers/run_controller.py`

- [ ] **Step 1: Update `collect_boundary_arrays` service signature**

In `swe2d/boundary_and_forcing/boundary_runtime_logic.py:54-82`, replace the function:

```python
def collect_boundary_arrays(
    *,
    mesh_data: Optional[Dict[str, np.ndarray]],
    mesh_boundary_edges_fn: Callable[[], Tuple[np.ndarray, np.ndarray]],
    default_bc_type: int = 0,
    apply_bc_layer_overrides_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
    log_fn: Callable[[str], None],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if mesh_data is None:
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float64),
        )

    from swe2d.mesh.boundary_edges_service import default_bc_for_edges as _compute_default_bc

    edge_n0, edge_n1 = mesh_boundary_edges_fn()
    if edge_n0.size == 0:
        log_fn("No boundary edges detected in mesh.")
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float64),
        )

    bc_type, bc_val = _compute_default_bc(mesh_data, edge_n0, edge_n1, default_bc_type=default_bc_type)
    bc_type, bc_val = apply_bc_layer_overrides_fn(edge_n0, edge_n1, bc_type, bc_val)
    return edge_n0, edge_n1, bc_type, bc_val
```

The old `default_bc_for_edges_fn` parameter is removed. The new `default_bc_type: int = 0` replaces it. The import of `_compute_default_bc` is added inside the function body (lazy import avoids circular dependencies).

- [ ] **Step 2: Update View's `_collect_boundary_arrays`**

In `swe2d/workbench/studio_dialog.py:1806-1814`, replace:

```python
def _collect_boundary_arrays(self):
    from swe2d.boundary_and_forcing.boundary_runtime_logic import collect_boundary_arrays as _logic
    default_bc_type = 0
    default_bc_combo = getattr(self._map_tab_view, "default_bc_type_combo", None)
    if default_bc_combo is not None:
        default_bc_type = int(default_bc_combo.currentData())
    return _logic(
        mesh_data=self._mesh_data,
        mesh_boundary_edges_fn=self._mesh_boundary_edges,
        default_bc_type=default_bc_type,
        apply_bc_layer_overrides_fn=self._apply_bc_layer_overrides,
        log_fn=self._log,
    )
```

- [ ] **Step 3: Remove `_default_bc_for_edges` from View**

Delete lines 2271-2278 in `swe2d/workbench/studio_dialog.py` (the entire `_default_bc_for_edges` method).

- [ ] **Step 4: Update controller callers**

In `swe2d/workbench/controllers/run_controller.py`:

**4a — `_collect_bc_for_edges` (line 912):** Replace:

```python
bc_type, bc_val = view._default_bc_for_edges(edge_n0, edge_n1)
bc_type, bc_val = view._apply_bc_layer_overrides(edge_n0, edge_n1, bc_type, bc_val)
```

with:

```python
_, _, bc_type, bc_val = view._collect_boundary_arrays()
```

This calls the View's boundary arrays method which reads the widget and delegates to the updated service — no mesh arrays pass through the View.

**4b — `_preview_override_summary` (lines 1073-1078):** Replace:

```python
bc_type_default, bc_val_default = view._default_bc_for_edges(edge_n0, edge_n1)
bc_type_preview = bc_type_default.copy()
bc_val_preview = bc_val_default.copy()
bc_type_preview, bc_val_preview = view._apply_bc_layer_overrides(
    edge_n0, edge_n1, bc_type_preview, bc_val_preview
)
```

with:

```python
bc_type_preview, bc_val_preview, _, _ = view._collect_boundary_arrays()
bc_type_preview = bc_type_preview.copy()
bc_val_preview = bc_val_preview.copy()
```

- [ ] **Step 5: Verify compliance**

```bash
# No remaining calls to _default_bc_for_edges from controller
! grep -n 'view\._default_bc_for_edges' swe2d/workbench/controllers/run_controller.py \
  && echo "PASS: no _default_bc_for_edges calls from controller"

# Confirm _default_bc_for_edges method is deleted from View
! grep -n '_default_bc_for_edges' swe2d/workbench/studio_dialog.py \
  && echo "PASS: _default_bc_for_edges removed from View"
```

Expected: PASS on both.

---

### Task 3: Extract plot math to Qt-free service

`studio_viewer_plot.py` mixes matplotlib rendering code into a QWidget View. For CLI use, the plotting logic needs to be callable without Qt (render to PNG/buffer).

**Approach:** Extract the render function wrappers and figure-level calls into a new `swe2d/plotting/viewer_plots.py` service module. `studio_viewer_plot.py` becomes a thin View wrapper that calls the service.

**Files:**
- Create: `swe2d/plotting/__init__.py`
- Create: `swe2d/plotting/viewer_plots.py`
- Modify: `swe2d/workbench/views/studio_viewer_plot.py`
- Modify: `swe2d/workbench/views/studio_viewer.py`

- [ ] **Step 1: Create `swe2d/plotting/` package**

```bash
mkdir -p swe2d/plotting
```

Create `swe2d/plotting/__init__.py` — empty file.

- [ ] **Step 2: Create `swe2d/plotting/viewer_plots.py`**

This module contains the figure-level render functions that are currently registered as `_render_fn` callbacks on each `PlotViewWidget`. Each function accepts a `fig` object, `mesh_data`, `result_data`, `mode`, `h_min`, and kwargs — but NOT a QWidget or Qt canvas. They return the figure (or modify it in place).

```python
"""Viewer plotting service — matplotlib rendering for all 5 plot modes.

Each function accepts a matplotlib Figure and data, modifies the Figure
in place, and returns it. No Qt imports. Callable from CLI to render
directly to file::

    fig = Figure(figsize=(6.4, 4.2))
    render_mesh_view(fig, mesh_data, None, "mesh", 1e-6)
    fig.savefig("mesh.png")
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from swe2d import units as _u
from swe2d.workbench.services.mesh_render_service import plot_mesh_view_on_figure
from swe2d.workbench.services.results_render_service import (
    render_timeseries_on_figure,
    render_profile_on_figure,
    render_structures_on_figure,
    render_network_on_figure,
)


def render_viewer_figure(
    fig: Any,
    mesh_data: Optional[Dict[str, np.ndarray]],
    result_data: Any,
    mode: str,
    h_min: float,
    selected_element_id: str = "",
    selected_metric: str = "flow",
    length_unit: str = "",
) -> Any:
    """Render a figure for the given viewer tab mode.

    Dispatches to the appropriate internal renderer based on *mode*.
    Returns the (modified) figure.
    """
    dispatch = {
        "Mesh": plot_mesh_view_on_figure,
        "Time Series": render_timeseries_on_figure,
        "Profile": render_profile_on_figure,
        "Structure": render_structures_on_figure,
        "Network": render_network_on_figure,
    }
    renderer = dispatch.get(mode)
    if renderer is None:
        fig.clear()
        fig.text(0.5, 0.5, f"Unknown mode: {mode}", ha="center", va="center", color="gray")
        return fig

    renderer(
        fig=fig,
        mesh_data=mesh_data,
        result_data=result_data,
        mode=mode,
        h_min=h_min,
        selected_element_id=selected_element_id,
        selected_metric=selected_metric,
        length_unit=length_unit,
    )
    return fig
```

- [ ] **Step 3: Update `studio_viewer_plot.py` to use the service**

Replace the `refresh()` method and `set_render_fn` usage:

**3a — Remove old render fn registration in `__init__`:**
Remove `self._render_fn: Optional[Callable] = None` and instead store the mode-to-dispatcher reference. Replace with `self._dispatched = True` (sentinel).

In `__init__`, replace:
```python
self._render_fn: Optional[Callable] = None
```
with:
```python
self._render_fn: Optional[Callable] = None  # ponytail: kept for backward compat during transition
```

**3b — Update `refresh()` in `studio_viewer_plot.py`:**

Replace:
```python
def refresh(self) -> None:
    if not _HAVE_MPL or self._fig is None:
        return
    if self._render_fn is None:
        return
    from swe2d import units as _u
    self._render_fn(self._fig, self._mesh_data, self._result_data,
                    self._mode, self._h_min,
                    selected_metric=self.selected_metric,
                    selected_element_id=self.selected_element_id,
                    length_unit=_u.length_unit_name())
    self._canvas.draw_idle()
    if self._table_widget is not None and self._table_widget.isVisible():
        self._populate_table()
```

with:

```python
def refresh(self) -> None:
    if not _HAVE_MPL or self._fig is None:
        return
    from swe2d.plotting.viewer_plots import render_viewer_figure
    render_viewer_figure(
        fig=self._fig,
        mesh_data=self._mesh_data,
        result_data=self._result_data,
        mode=self._mode,
        h_min=self._h_min,
        selected_element_id=self.selected_element_id,
        selected_metric=self.selected_metric,
        length_unit=_u.length_unit_name(),
    )
    self._canvas.draw_idle()
    if self._table_widget is not None and self._table_widget.isVisible():
        self._populate_table()
```

Also remove `self._render_fn` usage in `set_render_fn` and `set_data` — since we no longer need external render fn registration. Or keep `set_render_fn` as a no-op for backward compat.

**3c — Remove `_register_default_renderers` dependency:**

The `set_render_fn` calls in `studio_viewer.py:_register_default_renderers()` are now unnecessary since `refresh()` goes through the service dispatcher. Update `studio_viewer.py:_register_default_renderers`:

```python
def _register_default_renderers(self) -> None:
    """Renderers are dispatched by swe2d.plotting.viewer_plots — no per-widget registration needed."""
    pass
```

- [ ] **Step 4: Remove unused imports**

After step 3c, `studio_viewer.py` no longer imports from `results_render_service` or `mesh_render_service`. Remove these imports:

```python
# DELETE these imports from studio_viewer.py:
# from swe2d.workbench.services.mesh_render_service import plot_mesh_view_on_figure
# from swe2d.workbench.services.results_render_service import (
#     render_timeseries_on_figure,
#     render_profile_on_figure,
#     render_structures_on_figure,
#     render_network_on_figure,
# )
```

- [ ] **Step 5: Verify it renders**

```bash
python3 -c "
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from swe2d.plotting.viewer_plots import render_viewer_figure
fig = Figure()
render_viewer_figure(fig, None, None, 'Mesh', 1e-6)
fig.savefig('/tmp/test_mesh_plot.png')
print('PASS: plot rendered to /tmp/test_mesh_plot.png')
"
```

Expected: PASS, file created at `/tmp/test_mesh_plot.png`.
