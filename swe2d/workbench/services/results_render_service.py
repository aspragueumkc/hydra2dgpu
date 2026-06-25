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
        except Exception:
            pass
    vline = ax.axvline(x=t_hr, color="0.5", linewidth=0.9,
                       linestyle="--", zorder=5)
    canvas.draw_idle()
    return vline


def render_profile(
    ax: Any,
    fig: Any,
    run_records: list,
    line_id: int,
    t_sec: float,
    mode: str,
    fill_key: str,
    render_mode: str,
    cmap_name: str,
    use_fill_cmap: bool,
    show_structures: bool,
    find_nearest_timestep_fn: Any,
    load_profile_fn: Any,
    load_structure_flows_fn: Any,
    load_bound_layer_name_fn: Any,
    load_line_geometry_fn: Any,
    resolve_structure_profile_overlays_fn: Any,
    prof_fill_cbar: Any = None,
    length_unit: str = "",
) -> Tuple[int, Any]:
    """Render profile plot on the given axes.

    Returns (plotted_count, colorbar_or_None).
    """
    if prof_fill_cbar is not None:
        try:
            prof_fill_cbar.remove()
        except Exception:
            pass
        prof_fill_cbar = None

    ax.cla()
    plotted = 0
    bed_drawn = False
    structure_rows: List[Dict[str, object]] = []
    fill_segments: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
    fill_values: List[float] = []

    for rec in run_records:
        if line_id < 0:
            continue
        t = find_nearest_timestep_fn(rec.gpkg_path, rec.run_id, line_id, t_sec)
        data = load_profile_fn(rec.gpkg_path, rec.run_id, line_id, t)
        if not data:
            continue

        color = c2f(rec.color)
        station = data.get("station_m", np.empty(0))

        if mode == "wse_bed":
            wse = data.get("wse_m", np.full_like(station, np.nan))
            bed = data.get("bed_m", np.full_like(station, np.nan))
            depth = data.get("depth_m", np.full_like(station, np.nan))
            wet = data.get("wet", np.ones_like(station))

            ok = np.isfinite(wse) & np.isfinite(bed)
            if not np.any(ok):
                continue

            x_ok = station[ok]
            wse_ok = wse[ok]
            bed_ok = bed[ok]
            depth_ok = depth[ok]
            wet_ok = wet[ok]
            wet_mask = np.where(
                np.isfinite(wet_ok), wet_ok > 0.5, depth_ok > 1e-9
            )
            wse_phys = np.maximum(wse_ok, bed_ok)

            if render_mode == "raw":
                fill_mask = np.isfinite(wse_ok) & np.isfinite(bed_ok)
                wse_fill = wse_ok
                wse_plot = wse_ok
            else:
                fill_mask = wet_mask
                wse_fill = wse_phys
                wse_plot = np.where(wet_mask, wse_phys, np.nan)

            if not bed_drawn and x_ok.size:
                bed_min = float(np.min(bed_ok)) - 0.05 * max(float(np.ptp(bed_ok)), 0.1)
                ax.fill_between(x_ok, bed_min, bed_ok,
                                color="#8B7355", alpha=0.5, zorder=1)
                ax.plot(x_ok, bed_ok, color="#5C4033", linewidth=0.9, zorder=2)
                bed_drawn = True

            if use_fill_cmap:
                fill_metric = np.asarray(
                    data.get(fill_key, np.full_like(station, np.nan)),
                    dtype=np.float64,
                )
                fill_ok = fill_metric[ok]
                for i in range(len(x_ok) - 1):
                    if not (fill_mask[i] and fill_mask[i + 1]):
                        continue
                    if not (np.isfinite(fill_ok[i]) and np.isfinite(fill_ok[i + 1])):
                        continue
                    vmid = 0.5 * (float(fill_ok[i]) + float(fill_ok[i + 1]))
                    fill_values.append(vmid)
                    fill_segments.append((
                        x_ok[i : i + 2], bed_ok[i : i + 2],
                        wse_fill[i : i + 2], vmid,
                    ))
            else:
                ax.fill_between(x_ok, bed_ok, wse_fill,
                                where=fill_mask, interpolate=True,
                                color=color, alpha=0.18, zorder=3)
            ax.plot(x_ok, wse_plot, color=color, linewidth=1.5, zorder=4,
                    label=f"{rec.display_label()} WSE")
            plotted += 1
        else:
            if mode == "egl_m":
                wse = data.get("wse_m")
                vel = data.get("velocity_ms")
                if wse is None or vel is None:
                    continue
                y = np.asarray(wse, dtype=np.float64) + (
                    np.asarray(vel, dtype=np.float64) ** 2.0
                ) / (2.0 * _u.gravity())
            else:
                if mode not in data:
                    continue
                y = data[mode]
            ok = np.isfinite(station) & np.isfinite(y)
            if not np.any(ok):
                continue
            ax.plot(station[ok], y[ok], color=color, linewidth=1.5,
                    label=rec.display_label())
            plotted += 1

        if show_structures:
            try:
                rows = load_structure_flows_fn(rec.gpkg_path, rec.run_id, t, t_tol=1.0)
                if rows:
                    placed_ids = {str(r.get("object_id", "")) for r in structure_rows}
                    for rr in rows:
                        sid = str(rr.get("object_id", ""))
                        if sid in placed_ids:
                            continue
                        structure_rows.append({
                            "run_label": rec.display_label(),
                            "object_id": sid,
                            "flow": float(rr.get("value", 0.0)),
                            "station": float("nan"),
                            "elev": float("nan"),
                            "placement": "unplaced",
                        })
            except Exception as exc:
                logger.warning("[RESULTS] Structure overlay load failed: %s", exc)

    if use_fill_cmap and fill_segments and fill_values:
        try:
            from matplotlib import cm as mpl_cm, colors as mpl_colors
            vals = np.asarray(fill_values, dtype=np.float64)
            finite = np.isfinite(vals)
            if np.any(finite):
                vmin = float(np.nanmin(vals[finite]))
                vmax = float(np.nanmax(vals[finite]))
                if vmax <= vmin:
                    vmax = vmin + 1.0
                norm = mpl_colors.Normalize(vmin=vmin, vmax=vmax)
                cmap = mpl_cm.get_cmap(cmap_name)
                for x_seg, bed_seg, wse_seg, vmid in fill_segments:
                    ax.fill_between(x_seg, bed_seg, wse_seg,
                                    color=cmap(norm(vmid)), alpha=0.85,
                                    linewidth=0.0, zorder=3)
                sm = mpl_cm.ScalarMappable(norm=norm, cmap=cmap)
                sm.set_array([])
                prof_fill_cbar = fig.colorbar(sm, ax=ax, label=_label_for_var(fill_key, length_unit))
        except Exception as exc:
            logger.debug("[RESULTS] Fill segment render failed: %s", exc)

    if plotted and show_structures and structure_rows:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        placed = [r for r in structure_rows if np.isfinite(float(r.get("station", float("nan"))))]
        if np.isfinite(x0) and np.isfinite(x1) and x1 > x0 and np.isfinite(y0) and np.isfinite(y1) and placed:
            placed = sorted(placed, key=lambda r: float(r.get("station", 0.0)))[:12]
            y_span = max(y1 - y0, 1.0e-6)
            for i, row in enumerate(placed):
                xs = float(row.get("station", 0.0))
                elev = float(row.get("elev", float("nan")))
                q_val = float(row.get("flow", 0.0))
                sid = str(row.get("object_id", ""))
                y_anchor = elev if np.isfinite(elev) else y1
                y_anchor = min(max(y_anchor, y0 + 0.08 * y_span), y1 - 0.02 * y_span)
                y_text = min(y1 - 0.02 * y_span, y_anchor + (0.04 + 0.035 * (i % 3)) * y_span)
                ax.axvline(xs, color="0.35", linewidth=0.9, linestyle=":", alpha=0.5, zorder=2)
                if np.isfinite(elev):
                    ax.plot([xs], [elev], marker="v", markersize=4.0, color="0.25", zorder=6)
                ax.text(xs, y_text, f"{sid} {q_val:.2f}",
                        fontsize=7, rotation=90, va="top", ha="center",
                        color="0.35", zorder=6)

    _u_lab = _unit_labels(length_unit)
    _len_label = _u_lab["len"]
    ax.set_xlabel(f"Station ({_len_label})")
    ax.set_ylabel(f"Elevation ({_len_label})" if mode == "wse_bed" else _label_for_var(mode, length_unit))
    t_hr = t_sec / 3600.0
    ax.set_title(f"t = {t_hr:.3f} {_TIME_UNIT}", fontsize=9)
    ax.grid(True, alpha=0.3)

    if plotted:
        ax.legend(fontsize=8, loc="best")
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, color="gray")

    return plotted, prof_fill_cbar


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


def render_profile_on_figure(
    fig: Any,
    mesh_data: Any,
    result_data: Any,
    mode: str,
    h_min: float,
    length_unit: str = "",
    **kwargs,
) -> None:
    """Render a profile plot on *fig* using *result_data*."""
    if result_data is None:
        fig.clear()
        fig.text(0.5, 0.5, "No result data", ha="center", va="center", color="gray")
        return
    from swe2d.results.queries import (
        find_nearest_timestep,
        load_profile,
        load_profile_from_live,
        load_structure_flows_at_time,
    )
    is_live = result_data.data_source == "live"

    fig.clear()
    ax = fig.add_subplot(111)
    var_key = getattr(result_data, "prof_var_key", "wse_bed")
    fill_key = getattr(result_data, "prof_fill_key", "none")
    cmap = getattr(result_data, "prof_cmap", "viridis")
    show_struct = getattr(result_data, "prof_show_structures", True)
    use_fill_cmap = fill_key != "none"

    render_profile(
        ax=ax,
        fig=fig,
        run_records=result_data.get_enabled_run_records(),
        line_id=result_data.line_id,
        t_sec=result_data.current_time_sec,
        mode=var_key,
        fill_key=fill_key,
        render_mode="raw",
        cmap_name=cmap,
        use_fill_cmap=use_fill_cmap,
        show_structures=show_struct,
        find_nearest_timestep_fn=lambda gpkg, rid, lid, t: find_nearest_timestep(
            gpkg, rid, lid, t,
        ),
        load_profile_fn=lambda gpkg, rid, lid, t: (
            load_profile_from_live(result_data, str(rid), int(lid), float(t))
            if is_live else
            load_profile(gpkg, rid, lid, t)
        ),
        load_structure_flows_fn=lambda gpkg, rid, t, t_tol: load_structure_flows_at_time(
            gpkg, rid, t, t_tol,
        ),
        load_bound_layer_name_fn=lambda gpkg: "",
        load_line_geometry_fn=lambda gpkg, rid, lid: load_line_geometry(gpkg, rid, lid),
        resolve_structure_profile_overlays_fn=lambda run_records, line_id, t_sec: [],
        length_unit=length_unit,
    )


def render_structures_on_figure(
    fig: Any,
    mesh_data: Any,
    result_data: Any,
    mode: str,
    h_min: float,
    selected_element_id: str = "",
    selected_metric: str = "flow",
    length_unit: str = "",
    component: str = "structure",
) -> None:
    """render structures on figure."""
    if result_data is None:
        fig.clear()
        fig.text(0.5, 0.5, "No result data", ha="center", va="center", color="gray")
        return

    fig.clear()
    ax = fig.add_subplot(111)
    records = result_data.get_coupling_records()
    if not records:
        ax.text(0.5, 0.5, "No structure records", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        return

    by_obj: Dict[str, List[Tuple[float, float]]] = {}
    for rec in records:
        if str(rec.get("metric", "") or "") != selected_metric:
            continue
        if str(rec.get("component", "") or "") != component:
            continue
        try:
            t_s = float(rec.get("t_s", 0.0))
            val = float(rec.get("value", float("nan")))
        except (ValueError, TypeError):
            continue
        if not np.isfinite(val):
            continue
        oid = str(rec.get("object_id", "") or "")
        if selected_element_id and oid != selected_element_id:
            continue
        by_obj.setdefault(oid, []).append((t_s, val))

    if not by_obj:
        ax.text(0.5, 0.5, "No records", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        return

    for oid in sorted(by_obj.keys()):
        pairs = sorted(by_obj[oid], key=lambda x: x[0])
        x = np.asarray([p[0] / 3600.0 for p in pairs], dtype=np.float64)
        y = np.asarray([p[1] for p in pairs], dtype=np.float64)
        ax.plot(x, y, linewidth=1.8, label=oid)

    ax.set_xlabel(f"Time ({_TIME_UNIT})")
    ax.set_ylabel(_label_for_var(selected_metric, length_unit))
    ax.grid(True, alpha=0.3)
    if len(by_obj) > 1:
        ax.legend(fontsize=7, loc="best")


def render_network_on_figure(
    fig: Any,
    mesh_data: Any,
    result_data: Any,
    mode: str,
    h_min: float,
    selected_element_id: str = "",
    selected_metric: str = "flow",
    length_unit: str = "",
) -> None:
    """Render EPASWMM-style link profile on *fig* from coupling records.

    Draws a single subplot showing a longitudinal profile from upstream
    to downstream node with invert markers, bed fill, water depth fill,
    and flow annotation.
    """
    if result_data is None:
        fig.clear()
        fig.text(0.5, 0.5, "No result data", ha="center", va="center", color="gray")
        return

    fig.clear()
    ax = fig.add_subplot(111)
    records = result_data.get_coupling_records()

    # ── Filter records to closest coupling timestamp ──
    t_target = float(getattr(result_data, "current_time_sec", 0.0))
    all_ts: set = set()
    for rec in records:
        try:
            all_ts.add(float(rec.get("t_s", 0.0)))
        except (ValueError, TypeError):
            pass
    if all_ts:
        if t_target <= 0.0:
            t_target = min(all_ts)
        else:
            t_target = min(all_ts, key=lambda t: abs(t - t_target))
        records = [rec for rec in records if abs(float(rec.get("t_s", 0.0)) - t_target) < 1e-6]

    # Collect node inverts/depths, link flow records, and link lengths
    node_inverts: Dict[str, float] = {}
    node_depths: Dict[str, float] = {}
    links: Dict[str, Dict[str, Any]] = {}
    link_lengths: Dict[str, float] = {}

    for rec in records:
        oid = str(rec.get("object_id", "") or "")
        oname = str(rec.get("object_name", "") or "")
        metric = str(rec.get("metric", "") or "")
        try:
            val = float(rec.get("value", float("nan")))
        except (ValueError, TypeError):
            continue
        if not np.isfinite(val):
            continue
        comp = str(rec.get("component", "") or "")

        if comp == "drainage_node":
            if metric == "invert":
                node_inverts[oid] = val
            elif metric == "depth":
                node_depths[oid] = val
        elif comp == "drainage_link":
            if metric == "flow":
                m = re.match(r"(\S+)\s*->\s*(\S+)", oname)
                if m:
                    links[oid] = {
                        "from": m.group(1),
                        "to": m.group(2),
                        "flow": abs(val),
                    }
            elif metric == "length":
                link_lengths[oid] = val

    if not links:
        ax.text(
            0.5, 0.5, "No link data",
            ha="center", va="center",
            transform=ax.transAxes, color="gray",
        )
        return

    # Determine which link to profile
    candidate_link_ids: List[str] = []
    if selected_element_id:
        if selected_element_id in links:
            candidate_link_ids = [selected_element_id]
        else:
            # selected_element_id is a node — find links containing it
            candidate_link_ids = [
                lid
                for lid, ld in links.items()
                if ld["from"] == selected_element_id or ld["to"] == selected_element_id
            ]
        if not candidate_link_ids:
            ax.text(
                0.5, 0.5, "No link data for selected element",
                ha="center", va="center",
                transform=ax.transAxes, color="gray",
            )
            return
    else:
        candidate_link_ids = list(links.keys())

    # Use the first candidate link
    lid = candidate_link_ids[0]
    ld = links[lid]
    from_node = ld["from"]
    to_node = ld["to"]
    flow_val = ld["flow"]

    # Retrieve node invert elevations and depths
    inv_f = node_inverts.get(from_node, 0.0)
    inv_t = node_inverts.get(to_node, 0.0)
    depth_f = node_depths.get(from_node, 0.0)
    depth_t = node_depths.get(to_node, 0.0)

    # Guard against NaN/inf values that would break matplotlib rendering
    if not np.isfinite(inv_f):
        inv_f = 0.0
    if not np.isfinite(inv_t):
        inv_t = 0.0
    if not np.isfinite(depth_f):
        depth_f = 0.0
    if not np.isfinite(depth_t):
        depth_t = 0.0

    # If no invert data exists, derive a plausible bed from depth alone
    if inv_f == 0.0 and inv_t == 0.0 and (depth_f > 0.0 or depth_t > 0.0):
        inv_f = -depth_f * 5.0
        inv_t = -depth_t * 5.0

    # Use actual link length if available, else fall back to normalized 1.0
    link_len = link_lengths.get(lid, 0.0)
    if not np.isfinite(link_len) or link_len <= 0.0:
        link_len = 1.0

    x_dist = [0.0, link_len]
    y_inv = [inv_f, inv_t]
    y_ws = [inv_f + depth_f, inv_t + depth_t]

    # ---- Bed fill (below invert line) ----
    y_min = min(inv_f, inv_t) - 1.0
    ax.fill_between(x_dist, y_inv, y_min, color="saddlebrown", alpha=0.3)

    # ---- Invert line (bed profile) ----
    ax.plot(x_dist, y_inv, color="saddlebrown", linewidth=1.5)

    # ---- Water depth fill (above invert) ----
    ax.fill_between(x_dist, y_inv, y_ws, color="steelblue", alpha=0.5)

    # ---- Triangle markers at invert elevations ----
    ax.plot(0.0, inv_f, "v", color="black", ms=8, zorder=5,
            label=f"Upstream: {from_node}")
    ax.plot(link_len, inv_t, "^", color="black", ms=8, zorder=5,
            label=f"Downstream: {to_node}")

    # ---- Node ID labels ----
    ax.text(0.0, inv_f, f"  {from_node}", ha="left", va="bottom",
            fontsize=8, zorder=6)
    ax.text(link_len, inv_t, f"  {to_node}", ha="left", va="bottom",
            fontsize=8, zorder=6)

    # ---- Flow annotation ----
    ul = _unit_labels(length_unit)
    flow_units = ul["flow"]
    ax.text(
        0.5, 0.95, f"Q = {flow_val:.3f} {flow_units}",
        transform=ax.transAxes,
        ha="center", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
    )

    # ---- Labels, grid, title ----
    ax.set_title("Link Profile", fontsize=10)
    ax.set_xlabel(f"Distance ({ul['len']})")
    ax.set_ylabel(f"Elevation ({ul['len']})")
    ax.set_xlim(-0.05 * max(link_len, 1.0), link_len * 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")


# ---------------------------------------------------------------------------
# Table renderer — 6th tab: tabular coupling record display
# ---------------------------------------------------------------------------

# ponytail: render_table_on_figure deleted — Table tab uses QTableWidget
