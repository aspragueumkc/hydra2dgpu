#!/usr/bin/env python3
"""Results-panel and velocity-overlay bridge helpers for workbench dialog."""

from __future__ import annotations

from typing import Any


def maybe_create_results_panel(dialog: Any) -> None:
    """Create and register the dockable results panel if available."""
    try:
        from swe2d.results.panel import SWE2DResultsPanel
    except ImportError:
        dialog._log("[Results Panel] swe2d.results.panel not found - panel unavailable.")
        return

    gpkg = dialog._model_gpkg_path or ""
    iface = getattr(dialog, "_iface", None)
    try:
        dialog._results_panel = SWE2DResultsPanel(gpkg_path=gpkg, iface=iface, parent=None)
        dialog._results_panel.setWindowTitle("SWE2D Results")
        try:
            dialog._results_panel.timestep_changed.disconnect(dialog._on_results_panel_timestep_changed)
        except Exception:
            pass
        dialog._results_panel.timestep_changed.connect(dialog._on_results_panel_timestep_changed)
        try:
            dialog._results_panel.velocity_overlay_changed.disconnect(dialog._on_velocity_overlay_changed)
        except Exception:
            pass
        dialog._results_panel.velocity_overlay_changed.connect(dialog._on_velocity_overlay_changed)
        try:
            dialog._results_panel.velocity_overlay_add_requested.disconnect(dialog._on_velocity_overlay_add_requested)
        except Exception:
            pass
        dialog._results_panel.velocity_overlay_add_requested.connect(dialog._on_velocity_overlay_add_requested)
        if iface is not None and hasattr(iface, "addDockWidget"):
            try:
                from qgis.PyQt import QtCore

                iface.addDockWidget(QtCore.Qt.BottomDockWidgetArea, dialog._results_panel)
                dialog._results_panel.hide()
            except Exception as exc:
                dialog._log(f"[Results Panel] addDockWidget failed: {exc}")
    except Exception as exc:
        dialog._results_panel = None
        dialog._log(f"[Results Panel] Initialization failed: {exc}")


def get_velocity_vector_builder(dialog: Any) -> Any:
    """Return cached velocity builder or create one lazily."""
    if dialog._velocity_vector_builder is not None:
        return dialog._velocity_vector_builder
    try:
        from swe2d.results.velocity_layer import VelocityVectorBuilder

        dialog._velocity_vector_builder = VelocityVectorBuilder(max_cache_entries=24)
    except Exception as exc:
        dialog._log(f"Velocity overlay unavailable: could not import builder ({exc})")
        dialog._velocity_vector_builder = None
    return dialog._velocity_vector_builder
