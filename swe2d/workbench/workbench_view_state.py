"""View state dataclass for SWE2DWorkbenchStudioDialog.

This module owns the view-level state attributes that were previously
inlined in ``SWE2DWorkbenchStudioDialog.__init__``. The dialog now delegates
state initialization to ``WorkbenchViewState``.

The state is composed of:

* **Dock widget references** (status label, view/theme combos, three named
  dock widgets) — set during ``_build_ui``.
* **Component registry** (``studio_components`` dict) — populated as
  components are registered.
* **Feature flags** (``studio_feature_flags``) — toggle groups of widgets
  based on keywords.
* **High-perf canvas overlay** — the QGraphicsItem attached to the map
  canvas.
* **Persistence suppression** — a flag set during ``_restore_*`` so that
  signal-driven persists don't re-save the state we just loaded.
* **Detached dialog tracking** — three lists/slots for mesh-view and
  runtime-log windows that have been detached from the main dock layout.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QComboBox,
    QDialog,
    QDockWidget,
    QLabel,
)

from swe2d.workbench.views.studio_component_view import StudioComponent


_STUDIO_DEFAULT_FEATURE_FLAGS: Dict[str, bool] = {
    "rainfall": True,
    "drainage_structures": True,
}


@dataclass
class WorkbenchViewState:
    """All view-level state owned by the Workbench Studio dialog.

    The dialog's ``__init__`` is now a thin 4-line bootstrapper that stores
    ``_iface`` on itself and delegates everything else to the
    ``WorkbenchDialogBuilder``. The builder constructs one of these state
    objects, attaches it to the dialog as ``self._state``, and populates
    fields as it builds the UI.

    Attributes
    ----------
    iface:
        Optional QGIS interface reference, propagated from ``__init__`` so
        the state object is the single source of truth.
    studio_status_label:
        The footer status label widget.
    studio_view_mode_combo:
        Combo for switching between Mesh / Depth / Velocity / etc. views.
    studio_theme_combo:
        Combo for switching between Default / Diagnostics / Presentation
        visual profiles.
    studio_left_dock:
        The Model Setup dock, registered under ``"setup"``.
    studio_inspector_dock:
        The CFD Inspector dock, registered under ``"inspector"``.
    studio_results_dock:
        The SWE2D Results dock, registered under ``"results"``.
    studio_components:
        Name -> StudioComponent registry populated by ``_register_component``.
    studio_feature_flags:
        Map of feature key -> enabled. Mutated by
        ``_studio_set_feature_enabled`` and read by ``_studio_apply_feature_filters``.
    high_perf_canvas_overlay_item:
        The QGraphicsItem attached to the QGIS map canvas for the
        high-perf overlay path.
    persist_suppressed:
        Set to True during ``_restore_project_workbench_state`` so signal
        handlers do not re-persist the state we just loaded.
    mesh_view_detached_dialogs:
        List of detached mesh-view dialogs.
    mesh_view_detached_dialog:
        The most-recently-opened detached mesh-view dialog.
    runtime_log_detached_dialogs:
        List of detached runtime-log dialogs.
    """

    iface: Optional[Any] = None
    studio_status_label: Optional[QLabel] = None
    studio_view_mode_combo: Optional[QComboBox] = None
    studio_theme_combo: Optional[QComboBox] = None
    studio_left_dock: Optional[QDockWidget] = None
    studio_inspector_dock: Optional[QDockWidget] = None
    studio_results_dock: Optional[QDockWidget] = None
    studio_components: Dict[str, StudioComponent] = field(default_factory=dict)
    studio_feature_flags: Dict[str, bool] = field(
        default_factory=lambda: dict(_STUDIO_DEFAULT_FEATURE_FLAGS)
    )
    high_perf_canvas_overlay_item: Optional[Any] = None
    persist_suppressed: bool = False

    runtime_log_detached_dialogs: List[QDialog] = field(default_factory=list)


__all__ = ["WorkbenchViewState"]
