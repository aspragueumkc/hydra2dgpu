"""Overlay parameters service.

Collects overlay rendering parameters from a view and returns them as a
plain dict consumed by ``swe2d.results.high_perf_viewer.render_unstructured_snapshot_image``.

The View is expected to expose the dialog's widget attributes
(combos, checkboxes, spinboxes) and the runtime state used by the
high-perf overlay (``_high_perf_overlay_cell_x``, ``_snapshot_timesteps``,
``_gravity``, ``_mannings_n``, ``_length_unit_name``, ``_resolve_map_canvas``).
The service reads via ``getattr`` with sensible rendering defaults so
that a partial view (e.g. a test double) still produces a usable dict.

Zero Qt imports in this module — Qt is accessed only through the view.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


def _safe_current_data(combo: Any, default: Any) -> Any:
    """safe current data."""
    if combo is None:
        return default
    try:
        return combo.currentData()
    except Exception:
        return default


def _safe_is_checked(chk: Any, default: bool) -> bool:
    """safe is checked."""
    if chk is None:
        return default
    try:
        return bool(chk.isChecked())
    except Exception:
        return default


def _safe_value(spin: Any, default: float) -> float:
    """safe value."""
    if spin is None:
        return default
    try:
        return float(spin.value())
    except Exception:
        return default


def _safe_extent(canvas: Any) -> Optional[tuple]:
    """safe extent."""
    if canvas is None or not hasattr(canvas, "extent"):
        return None
    try:
        ex = canvas.extent()
        return (
            float(ex.xMinimum()),
            float(ex.xMaximum()),
            float(ex.yMinimum()),
            float(ex.yMaximum()),
        )
    except Exception:
        return None


def collect_overlay_parameters(view: Any, t_use: float) -> Dict[str, Any]:
    """Collect all overlay rendering parameters from ``view``.

    ``view`` is ``SWE2DWorkbenchStudioDialog``.  Overlay widgets live on
    ``view._results_toolbox`` (a ``ResultsToolbox``).  There is no fallback
    to direct dialog attributes — that path violates MVP Rule 8.

    Returns a dict matching the keyword arguments of
    ``swe2d.results.high_perf_viewer.render_unstructured_snapshot_image``.
    """
    tb = view._results_toolbox

    def _w(name: str) -> Any:
        """w."""
        return getattr(tb, name)
    field_key = str(_safe_current_data(_w("field_combo"), "depth") or "depth")
    wse_render_mode = str(_safe_current_data(_w("wse_render_combo"), "cell") or "cell")
    cmap_key = str(_safe_current_data(_w("cmap_combo"), "turbo") or "turbo")

    visible_only = _safe_is_checked(_w("visible_only_chk"), False)
    lock_canvas = _safe_is_checked(_w("lock_canvas_chk"), False)
    auto_contrast = _safe_is_checked(_w("auto_contrast_chk"), True)
    if visible_only:
        auto_contrast = False

    canvas_resolver = getattr(view, "_resolve_map_canvas", None)
    canvas = canvas_resolver() if callable(canvas_resolver) else None

    if lock_canvas and canvas is not None:
        try:
            res = (max(64, int(canvas.width())), max(64, int(canvas.height())))
        except Exception:
            res = (1280, 720)
    else:
        raw_res = _safe_current_data(_w("res_combo"), None)
        if isinstance(raw_res, tuple) and len(raw_res) == 2:
            try:
                res = (max(64, int(raw_res[0])), max(64, int(raw_res[1])))
            except Exception:
                res = (1280, 720)
        else:
            res = (1280, 720)

    opacity = _safe_value(_w("opacity_spin"), 1.0)
    try:
        view._overlay_opacity = opacity
    except Exception:
        pass

    show_velocity_arrows = _safe_is_checked(_w("arrows_chk"), False)
    arrow_stride_px = int(round(_safe_value(_w("arrow_density_spin"), 28.0)))
    arrow_length_scale = _safe_value(_w("arrow_length_spin"), 1.0)
    arrow_head_length_scale = _safe_value(_w("arrow_head_length_spin"), 1.0)
    arrow_head_width_scale = _safe_value(_w("arrow_head_width_spin"), 1.0)

    show_streamlines = _safe_is_checked(_w("streamlines_chk"), False)
    streamline_backend = str(
        _safe_current_data(_w("streamline_backend_combo"), "auto") or "auto"
    )
    streamline_seed_count = int(
        round(_safe_value(_w("streamline_seed_spin"), 48.0))
    )
    streamline_steps = int(
        round(_safe_value(_w("streamline_steps_spin"), 24.0))
    )

    visible_extent_world = _safe_extent(canvas)
    render_extent_world = (
        visible_extent_world
        if (visible_only and visible_extent_world is not None)
        else None
    )

    courant_cell_size = 0.0
    cx = getattr(view, "_high_perf_overlay_cell_x", None)
    cy = getattr(view, "_high_perf_overlay_cell_y", None)
    if cx is not None and cy is not None and cx.size > 0 and cy.size > 0:
        try:
            bbox_area = (float(cx.max()) - float(cx.min())) * (
                float(cy.max()) - float(cy.min())
            )
            if bbox_area > 0:
                courant_cell_size = float((bbox_area / max(cx.size, 1)) ** 0.5)
        except Exception:
            courant_cell_size = 0.0

    courant_dt = 0.0
    ts_array = getattr(view, "_snapshot_timesteps", []) or []
    try:
        if len(ts_array) >= 2:
            courant_dt = abs(float(ts_array[-1][0]) - float(ts_array[-2][0]))
        elif len(ts_array) == 1:
            courant_dt = max(1.0, float(ts_array[0][0]))
        else:
            courant_dt = 1.0
    except Exception:
        courant_dt = 1.0

    length_unit_name = getattr(view, "_length_unit_name", "m")
    legend_label = (
        f"Depth ({length_unit_name})" if field_key == "depth"
        else f"Velocity ({length_unit_name}/s)" if field_key == "speed"
        else f"Water Surface ({length_unit_name})" if field_key == "wse"
        else "Froude Number" if field_key == "froude"
        else "Courant Number" if field_key == "courant"
        else "Shear Stress (Pa)"
    )

    return {
        "cell_x": getattr(view, "_high_perf_overlay_cell_x", None),
        "cell_y": getattr(view, "_high_perf_overlay_cell_y", None),
        "cell_bed": getattr(view, "_high_perf_overlay_cell_bed", None),
        "node_x": getattr(view, "_high_perf_overlay_node_x", None),
        "node_y": getattr(view, "_high_perf_overlay_node_y", None),
        "cell_nodes": getattr(view, "_high_perf_overlay_cell_nodes", None),
        "tri_to_cell": getattr(view, "_high_perf_overlay_tri_to_cell", None),
        "timesteps": getattr(view, "_snapshot_timesteps", []),
        "current_time_s": float(t_use),
        "field_key": field_key,
        "wse_render_mode": wse_render_mode,
        "cmap_key": cmap_key,
        "resolution": res,
        "auto_contrast": auto_contrast,
        "show_velocity_arrows": show_velocity_arrows,
        "arrow_stride_px": arrow_stride_px,
        "arrow_length_scale": arrow_length_scale,
        "arrow_head_length_scale": arrow_head_length_scale,
        "arrow_head_width_scale": arrow_head_width_scale,
        "show_streamlines": show_streamlines,
        "streamline_backend": streamline_backend,
        "streamline_seed_count": streamline_seed_count,
        "streamline_steps": streamline_steps,
        "visible_extent_world": visible_extent_world,
        "render_extent_world": render_extent_world,
        "gravity": getattr(view, "_gravity", 9.81),
        "courant_cell_size": courant_cell_size,
        "courant_dt": courant_dt,
        "manning_n": getattr(view, "_mannings_n", 0.035),
        "show_legend": True,
        "legend_label": legend_label,
    }
