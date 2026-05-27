#!/usr/bin/env python3
"""Results-panel and velocity-overlay bridge helpers for workbench dialog."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import traceback
from typing import Any


def _load_results_panel_class(dialog: Any):
    """Resolve SWE2DResultsPanel across canonical and fallback module paths."""
    errors = []

    # Preferred canonical package import.
    try:
        from swe2d.results.panel import SWE2DResultsPanel

        return SWE2DResultsPanel
    except Exception as exc:
        errors.append(f"swe2d.results.panel import failed: {exc}")

    # Legacy flat module fallback, if present in a dev tree.
    try:
        mod = importlib.import_module("swe2d_results_panel")
        cls = getattr(mod, "SWE2DResultsPanel", None)
        if cls is not None:
            return cls
        errors.append("swe2d_results_panel imported but SWE2DResultsPanel not found")
    except Exception as exc:
        errors.append(f"swe2d_results_panel import failed: {exc}")

    # Direct source-file fallback to avoid stale module-resolution issues.
    try:
        panel_path = Path(__file__).resolve().parents[2] / "swe2d" / "results" / "panel.py"
        if panel_path.exists():
            spec = importlib.util.spec_from_file_location("swe2d.results.panel_fallback", str(panel_path))
            if spec is not None and spec.loader is not None:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                cls = getattr(mod, "SWE2DResultsPanel", None)
                if cls is not None:
                    return cls
                errors.append(f"{panel_path} loaded but SWE2DResultsPanel not found")
            else:
                errors.append(f"Could not build import spec for {panel_path}")
        else:
            errors.append(f"Panel source file not found: {panel_path}")
    except Exception as exc:
        errors.append(f"panel.py direct load failed: {exc}")

    for msg in errors:
        dialog._log(f"[Results Panel] {msg}")
    return None


def maybe_create_results_panel(dialog: Any) -> None:
    """Create and register the dockable results panel if available."""

    def _connect_if_available(signal_name: str, handler_name: str) -> None:
        sig = getattr(dialog._results_panel, signal_name, None)
        handler = getattr(dialog, handler_name, None)
        if sig is None:
            dialog._log(f"[Results Panel] Signal '{signal_name}' not found on panel.")
            return
        if handler is None:
            dialog._log(f"[Results Panel] Handler '{handler_name}' not found on dialog; skipping.")
            return
        try:
            sig.disconnect(handler)
        except Exception:
            pass
        sig.connect(handler)

    SWE2DResultsPanel = _load_results_panel_class(dialog)
    if SWE2DResultsPanel is None:
        dialog._log("[Results Panel] Panel unavailable after all import attempts.")
        return

    gpkg = dialog._model_gpkg_path or ""
    iface = getattr(dialog, "_iface", None)
    try:
        dialog._results_panel = SWE2DResultsPanel(gpkg_path=gpkg, iface=iface, parent=None)
        dialog._results_panel.setWindowTitle("SWE2D Results")
        _connect_if_available("timestep_changed", "_on_results_panel_timestep_changed")
        _connect_if_available("velocity_overlay_changed", "_on_velocity_overlay_changed")
        _connect_if_available("velocity_overlay_add_requested", "_on_velocity_overlay_add_requested")
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
        dialog._log(traceback.format_exc())


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
