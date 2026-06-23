"""View Protocol interfaces for MVP architecture.

Each domain controller receives a typed View protocol — not the full dialog.
This prevents controllers from accessing widgets directly and makes them
testable with simple mocks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np
from qgis.PyQt.QtCore import QObject


class RunView(Protocol):
    """View protocol for the run pipeline controller."""

    @property
    def _mesh_data(self) -> Optional[Dict[str, np.ndarray]]:
        """Mesh data from the currently loaded mesh."""

    @property
    def _model_tab_view(self) -> Any:
        """The model tab view instance."""

    def _log(self, msg: str) -> None:
        """Log a message through the view's logging mechanism."""

    def set_run_button_enabled(self, enabled: bool) -> None:
        """Enable or disable the run button."""

    def set_cancel_button_enabled(self, enabled: bool) -> None:
        """Enable or disable the cancel button."""


class LayerView(Protocol):
    """View protocol for layer combo management."""

    def _log(self, msg: str) -> None:
        """Log a message through the view's logging mechanism."""
        ...
    def populate_layer_combo(self, combo_attr: str, layers: List, layer_type_hint: str = "") -> None:
        """Fill a combo by attribute name with layers, preserving current selection."""
        ...
    def get_combo_current_text(self, combo_attr: str) -> str:
        """Get the current display text of a combo by attribute name."""
        ...
    def select_layer_in_combo(self, combo_attr: str, layer_id: str) -> None:
        """Select an item by layer ID in a combo by attribute name."""
        ...
    def get_topo_combo(self, attr: str) -> Any:
        """Get a topology tab combo by attribute name, or None."""
        ...


class MeshView(Protocol):
    """View protocol for mesh import, terrain, and GeoPackage operations."""

    _mesh_data: Optional[Dict[str, np.ndarray]]

    def _log(self, msg: str) -> None:
        """Log a message through the view's logging mechanism."""

    def _refresh_plot(self) -> None:
        """Refresh the mesh plot display."""

    def _reset_runtime_snapshot_overlay_cache(self, reason: str) -> None:
        """Invalidate cached overlay snapshot data for a given reason."""

    def _combo_layer(self, combo: Any, expected_kind: str) -> Any:
        """Resolve a combo widget's selected layer by expected type."""

    def set_layer_status_text(self, text: str) -> None:
        """Update the status label for the active layer."""

    def set_results_gpkg_path(self, path: str) -> None:
        """Set the GeoPackage results file path."""

    def get_combo_selected_layer(self, combo_attr: str, kind: str = "vector") -> Any:
        """Get the selected QGIS layer from a combo by attribute name."""

    def get_combo_widget(self, combo_attr: str) -> Any:
        """Get a combo widget by attribute name."""


class OverlayView(Protocol):
    """View protocol for high-perf canvas overlay."""

    _high_perf_canvas_overlay_enabled: bool
    _high_perf_overlay_cell_x: Any
    _high_perf_overlay_cell_y: Any
    _snapshot_timesteps: Any

    def _log(self, msg: str) -> None:
        """Log a message through the view's logging mechanism."""

    def _sync_high_perf_overlay_data(self) -> None:
        """Synchronise overlay data from the current results snapshot."""

    def _refresh_high_perf_canvas_overlay(self, t_s: Any) -> None:
        """Redraw the high-performance canvas overlay at a given time."""

    def _resolve_qgis_iface(self) -> Any:
        """Resolve the current QGIS interface instance."""

    def sync_overlay_widget_states(self) -> None:
        """Enable/disable overlay controls based on current selections."""
        ...
    def get_overlay_export_field(self) -> str:
        """Get the currently selected overlay export field name."""

    def get_overlay_export_cmap(self) -> str:
        """Get the currently selected overlay export colour map."""

    def get_overlay_export_wse_render_mode(self) -> str:
        """Get the WSE render mode for overlay export."""

    def get_overlay_auto_contrast(self) -> bool:
        """Get whether auto-contrast is enabled for the overlay."""

    def show_warning_message(self, title: str, message: str) -> None:
        """Show a warning dialog with the given title and message."""

    def show_get_save_file(self, title: str, start_dir: str, filter_str: str) -> str:
        """Show a save-file dialog and return the selected path."""

    def show_get_double(self, title: str, label: str, value: float, min_v: float, max_v: float) -> Tuple[float, bool]:
        """Show a double-input dialog and return the value and acceptance flag."""

    def refresh_map_canvas(self) -> None:
        """Force the map canvas to redraw."""


class TopologyMeshView(Protocol):
    """View protocol for topology-based meshing."""

    _topology_tab_view: Any
    topo_status_lbl: Any

    def _log(self, msg: str) -> None:
        """Log a message through the view's logging mechanism."""

    def update_topo_status(self, text: str) -> None:
        """Update the topology status label text."""

    def update_topo_controls_summary(self, text: str) -> None:
        """Update the topology controls summary label text."""

    def get_topo_widget_value(self, attr: str) -> Any:
        """Read a widget value (spin, checkbox, or combo data) by attribute name."""
        ...
    def set_topo_widget_visible(self, attr: str, visible: bool) -> None:
        """Show or hide a topology tab widget by attribute name."""

    def get_topo_combo_data(self, attr: str) -> Any:
        """Return currentData() of a topology combo by attribute name."""
        ...
