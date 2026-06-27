"""Results plotting service — matplotlib rendering for timeseries and profile.

Extracted from ``StudioResultsView`` to enforce MVP: the View owns the
Figure/Canvas Qt widgets, this service owns all numpy computation and
matplotlib drawing.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from swe2d import units as _u

logger = logging.getLogger(__name__)

# ponytail: dynamic unit labels driven by swe2d.units (CRS-derived)
_TIME_UNIT = "hr"  # time is always displayed in hours


def _unit_labels(length_unit: str = "") -> dict:
    """Return model-unit-aware label strings.  Empty string → fall back to swe2d.units."""
    lu = str(length_unit or _u.length_unit_name() or "m").strip().lower()
    if lu == "ft":
        return {"len": "ft", "flow": "ft³/s", "vel": "ft/s"}
    return {"len": "m", "flow": "m³/s", "vel": "m/s"}


def _label_for_var(var_key: str, length_unit: str = "") -> str:
    """Build a (unit-aware) display label for a known TS / profile variable key."""
    u = _unit_labels(length_unit)
    table = {
        "flow_cms":      f"Flow ({u['flow']})",
        "depth_m":       f"Depth ({u['len']})",
        "wse_m":         f"WSE ({u['len']})",
        "velocity_ms":   f"Velocity ({u['vel']})",
        "station_m":     f"Station ({u['len']})",
        "bed_m":         f"Bed ({u['len']})",
        "egl_m":         f"EGL Error ({u['len']})",
        "fr":            "Froude number",
        "flow_qn":       f"Normal flow ({u['flow']})",
    }
    return table.get(str(var_key), str(var_key))


# Backwards-compat aliases for callers that import these names directly.
def _ts_var_labels(length_unit: str = "") -> List[Tuple[str, str]]:
    """ts var labels."""
    u = _unit_labels(length_unit)
    return [
        (f"Flow ({u['flow']})",          "flow_cms"),
        (f"Depth ({u['len']})",          "depth_m"),
        (f"WSE ({u['len']})",            "wse_m"),
        (f"Velocity ({u['vel']})",       "velocity_ms"),
    ]


def _profile_var_labels(length_unit: str = "") -> List[Tuple[str, str]]:
    """profile var labels."""
    u = _unit_labels(length_unit)
    return [
        ("WSE + Bed",                    "wse_bed"),
        (f"Depth ({u['len']})",          "depth_m"),
        (f"Velocity ({u['vel']})",       "velocity_ms"),
        (f"EGLError ({u['len']})",       "egl_m"),
    ]


def _profile_fill_labels(length_unit: str = "") -> List[Tuple[str, str]]:
    """profile fill labels."""
    u = _unit_labels(length_unit)
    return [
        ("None",                         "none"),
        (f"Depth ({u['len']})",          "depth_m"),
        (f"Velocity ({u['vel']})",       "velocity_ms"),
        (f"Flow ({u['flow']})",          "flow_cms"),
    ]


# Keep the static _TS_VARIABLES / _PROFILE_VARIABLES / _PROFILE_FILL_OPTIONS symbols
# for any existing imports, but prefer the unit-aware builders above.  These assume
# SI ("m") as a static fallback for imports that happen before the unit system
# is configured.
_TS_VARIABLES: List[Tuple[str, str]] = _ts_var_labels()
_PROFILE_VARIABLES: List[Tuple[str, str]] = _profile_var_labels()
_PROFILE_FILL_OPTIONS: List[Tuple[str, str]] = _profile_fill_labels()

_PROFILE_CMAP_OPTIONS: List[Tuple[str, str]] = [
    ("Viridis", "viridis"),
    ("Turbo", "turbo"),
    ("Plasma", "plasma"),
    ("Inferno", "inferno"),
    ("Coolwarm", "coolwarm"),
]

_SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def c2f(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """Convert (R, G, B) 0-255 tuple to matplotlib 0-1 float tuple."""
    return (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)


def render_timeseries(
    ax: Any,
    run_records: list,
    line_id: int,
    var_key: str,
    var_label: str,
    current_time_sec: float,
    load_timeseries_fn: Any,
    length_unit: str = "",
) -> int:
    """Render timeseries plot on the given axes.

    Returns the number of runs plotted.
    """
    ax.cla()
    plotted = 0

    for rec in run_records:
        if line_id < 0:
            continue
        raw = load_timeseries_fn(rec, line_id, var_key)
        if not raw or var_key not in raw:
            continue
        t_hr = raw["t_s"] / 3600.0
        vals = raw[var_key]
        ax.plot(t_hr, vals, color=c2f(rec.color), linewidth=1.6,
                label=rec.display_label())
        plotted += 1

    t_hr_now = current_time_sec / 3600.0
    vline = ax.axvline(x=t_hr_now, color="0.5", linewidth=0.9,
                       linestyle="--", zorder=5)

    # ponytail: prefer the caller's var_label (already unit-aware) over the bare key
    label = str(var_label) if str(var_label).strip() else _label_for_var(var_key, length_unit)
    ax.set_xlabel(f"Time ({_TIME_UNIT})")
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend(fontsize=8, loc="best")
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, color="gray")

    return plotted


def update_vline(ax: Any, canvas: Any, vline: Any, current_time_sec: float):
    """Update or create the vertical time marker. Returns the vline artist."""
    t_hr = current_time_sec / 3600.0
    if vline is not None:
        try:
            vline.set_xdata([t_hr, t_hr])
            canvas.draw_idle()
            return vline
        except Exception as _e:

            logger.warning(f"[ERROR] Exception in results_render_service.py: {_e}")
    vline = ax.axvline(x=t_hr, color="0.5", linewidth=0.9,
                       linestyle="--", zorder=5)
    canvas.draw_idle()
    return vline


# ---------------------------------------------------------------------------
# Figure-level wrappers — match PlotViewWidget render_fn signature:
#   fn(fig, mesh_data, result_data, mode, h_min, **kwargs)
# result_data is expected to be a SWE2DResultsData instance.
# ---------------------------------------------------------------------------

def render_timeseries_on_figure(
    fig: Any,
    mesh_data: Any,
    result_data: Any,
    mode: str,
    h_min: float,
    length_unit: str = "",
    **kwargs,
) -> None:
    """Render a time-series plot on *fig* using *result_data*."""
    if result_data is None:
        fig.clear()
        fig.text(0.5, 0.5, "No result data", ha="center", va="center", color="gray")
        return
    from swe2d.results.queries import load_timeseries as _load_ts
    from swe2d.results.queries import load_timeseries_from_live as _load_ts_live
    is_live = result_data.data_source == "live"

    fig.clear()
    ax = fig.add_subplot(111)
    var_key = getattr(result_data, "ts_var_key", "flow_cms")
    var_label = _label_for_var(var_key, length_unit)
    render_timeseries(
        ax=ax,
        run_records=result_data.get_enabled_run_records(),
        line_id=result_data.line_id,
        var_key=var_key,
        var_label=var_label,
        current_time_sec=result_data.current_time_sec,
        load_timeseries_fn=lambda rec, lid, vk: (
            _load_ts_live(result_data, str(rec.run_id), int(lid))
            if is_live else
            _load_ts(str(rec.gpkg_path), str(rec.run_id), int(lid))
        ),
        length_unit=length_unit,
    )
# ponytail: render_structures_on_figure, render_network_on_figure
# deleted — Network/Structure tabs removed
