"""Workbench GUI API — public interfaces for the Studio workbench.

This module consolidates the public protocols, interfaces, and types
that external code (services, controllers, tests) can rely on for
interacting with the Studio workbench.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │  Plugin Entry (thin)                                        │
    │  - __init__ just creates dialog and shows it                │
    └─────────────────────────────────────────────────────────────┘
         │ creates & shows
         ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  View (Qt UI)                                               │
    │  - Tab QWidget subclasses                                   │
    │  - Implements WorkbenchView protocol                        │
    │  - Owns widget references                                   │
    └─────────────────────────────────────────────────────────────┘
         ▲                          ▲
         │ update()                 │ signal
         │                          │
    ┌────┴────────────────────────────────────────────────────────┐
    │  Controller / Presenter                                     │
    │  - WorkbenchController (the brain)                          │
    │  - Receives View signals, calls services, updates View     │
    └─────────────────────────────────────────────────────────────┘
         │                          ▲
         │ call                     │ return
         ▼                          │
    ┌─────────────────────────────────────────────────────────────┐
    │  Service Layer (zero Qt)                                    │
    │  - run_service, gpkg_service, mesh_service, etc.            │
    │  - Pure Python business logic                               │
    └─────────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable


# ═══════════════════════════════════════════════════════════════════════
# View interfaces (Protocols)
# ═══════════════════════════════════════════════════════════════════════


@runtime_checkable
class WorkbenchView(Protocol):
    """The View in MVP. Any object the Controller interacts with.
    
    The Studio dialog implements this implicitly by exposing the
    expected attributes (_results_panel, _mesh_data, _log, etc.).
    """
    
    # State attributes the controller reads
    _results_panel: Any
    _high_perf_overlay_cell_x: Any
    _mesh_data: Any
    
    # Method the controller calls for logging
    def _log(self, msg: str) -> None: """log."""

@runtime_checkable
class OverlayView(Protocol):
    """A View that provides overlay parameters.
    
    Implemented by tab widgets that contain overlay controls
    (e.g., Map tab). The overlay_parameters_service uses this
    protocol to read parameters without coupling to the dialog.
    """
    
    def get_field_key(self) -> str: """Return field key."""

    def get_colormap(self) -> str: """Return colormap."""

    def get_opacity(self) -> float: """Return opacity."""

    def get_resolution(self) -> str: """Return resolution."""

    def get_wse_render(self) -> str: """Return wse render."""

    def get_arrow_density(self) -> float: """Return arrow density."""

    def get_arrow_length(self) -> float: """Return arrow length."""

    def get_arrow_head_length(self) -> float: """Return arrow head length."""

    def get_arrow_head_width(self) -> float: """Return arrow head width."""

    def get_streamline_backend(self) -> str: """Return streamline backend."""

    def get_streamline_seeds(self) -> int: """Return streamline seeds."""

    def get_streamline_steps(self) -> int: """Return streamline steps."""

    def get_lock_canvas(self) -> bool: """Return lock canvas."""

    def get_visible_only(self) -> bool: """Return visible only."""

    def get_auto_contrast(self) -> bool: """Return auto contrast."""

    def get_arrow_enabled(self) -> bool: """Return arrow enabled."""

    def get_streamline_enabled(self) -> bool: """Return streamline enabled."""

    def get_h_min(self) -> float: """Return h min."""

    def get_gravity(self) -> float: """Return gravity."""

@runtime_checkable
class WorkbenchControllerProtocol(Protocol):
    """The Controller interface. Anything the View can call on the
    controller is defined here.
    
    The Studio dialog's _controller attribute conforms to this
    protocol implicitly.
    """
    
    _view: "WorkbenchView"
    
    def load_mesh_snapshot_for_overlay(self, t_s: float) -> bool: """Load mesh snapshot for overlay."""

# ═══════════════════════════════════════════════════════════════════════
# View Plotter API — register / dispatch arbitrary matplotlib views
# ═══════════════════════════════════════════════════════════════════════

ViewPlotter = Callable[[Any, Dict[str, Any], Optional[Dict[str, Any]], str, float], None]
"""Signature: ``plotter(fig, mesh_data, result_data, mode, h_min) -> None``.

A "view plotter" is a standalone function that takes an existing
matplotlib ``Figure`` (already attached to a canvas in a Qt layout)
and renders the current mesh view onto it.  The plotter is
responsible for ``fig.clear()``, creating subplots, and calling
``fig.canvas.draw_idle()`` if needed.

This decouples the plotting logic from the dialog class so any
external code can register a new view mode.
"""

# Module-level registry: view_mode_name -> plotter
_VIEW_PLOTTERS: Dict[str, ViewPlotter] = {}


def register_view_plotter(name: str, plotter: ViewPlotter) -> None:
    """Register a view plotter under a view-mode name.

    The name should match the view combo's ``currentText()``
    (e.g. ``"Mesh"``, ``"Depth"``, ``"Velocity magnitude"``).
    """
    _VIEW_PLOTTERS[name] = plotter


def get_view_plotter(name: str) -> Optional[ViewPlotter]:
    """Retrieve a registered view plotter by name, or ``None``."""
    return _VIEW_PLOTTERS.get(name)


def list_view_plotters() -> List[str]:
    """Return all registered view plotter names."""
    return list(_VIEW_PLOTTERS.keys())


def clear_view_plotters() -> None:
    """Remove all registered view plotters (useful in tests)."""
    _VIEW_PLOTTERS.clear()


# ═══════════════════════════════════════════════════════════════════════
# Service interfaces (Protocols)
# ═══════════════════════════════════════════════════════════════════════


@runtime_checkable
class MeshSnapshotLoader(Protocol):
    """A service that loads mesh snapshot data from a GPKG."""
    
    def __call__(
        self,
        gpkg_path: str,
        run_id: str,
        t_s: float,
    ) -> Optional[Dict[str, Any]]: """Load mesh snapshot data from a GPKG."""


@runtime_checkable
class OverlayParametersCollector(Protocol):
    """A service that collects overlay parameters from a view."""

    def __call__(self, view: OverlayView) -> Dict[str, Any]: """call."""

# ═══════════════════════════════════════════════════════════════════════
# Public API exports
# ═══════════════════════════════════════════════════════════════════════

__all__ = [
    # View interfaces
    "WorkbenchView",
    "OverlayView",
    "WorkbenchControllerProtocol",
    # View Plotter API
    "ViewPlotter",
    "register_view_plotter",
    "get_view_plotter",
    "list_view_plotters",
    "clear_view_plotters",
    # Service interfaces
    "MeshSnapshotLoader",
    "OverlayParametersCollector",
]
