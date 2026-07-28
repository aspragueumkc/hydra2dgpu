"""swe2d/workbench/dialogs/network_profile_plot_widget.py

Reusable matplotlib widget that renders a ProfileArrays as a longitudinal
profile (SWMM-style). Matches the existing single-link PG viewer
(studio_viewer_profile_pg.py) conceptually:

  * Per-cell invert (read from the GPKG geometry columns, NOT hard-coded)
  * Per-cell crown (invert + cell_width for circular, invert + cell_height
    for rectangular — per cell, not link-uniform)
  * HGL = invert + depth per cell
  * Segment-by-segment water fill colored by the active metric colormap
  * Ground/rim line interpolated between node endpoints
  * Node "cylinder" markers at each chain endpoint

Axis labels are unit-aware: the value in ProfileOptions is treated as a
*template* — the widget appends the active length unit (e.g. "Distance"
becomes "Distance (m)" or "Distance (ft)" based on
swe2d.units.length_unit_name()).  Pass a fully-spelled label
(e.g. "Distance (m)") to override the unit substitution.
"""

from __future__ import annotations

import csv
import logging
from typing import Optional

import numpy as np
from qgis.PyQt import QtWidgets

from swe2d.workbench.dialogs._plot_utils import try_import_matplotlib_qt
from swe2d.workbench.dialogs.profile_options_dialog import ProfileOptions
from swe2d.workbench.services.profile_pipeline_service import (
    ProfileArrays,
    profile_at_variable,
)

logger = logging.getLogger(__name__)

FigureCanvasQt, Figure, _mtri = try_import_matplotlib_qt()


def _length_unit() -> str:
    """Return the active length unit string ('m' or 'ft') from swe2d.units.

    Returns the empty string when the units module has not been
    initialised (no project loaded) — NEVER a hard-coded 'm' fallback.
    The plot widget must reflect what the project's CRS is actually
    using, not assume SI.
    """
    try:
        from swe2d import units as _u
        return str(_u.length_unit_name() or "").strip()
    except Exception:
        return ""


def _resolve_axis_label(template: str, unit: str) -> str:
    """Return a unit-aware axis label.

    The user types a short label (e.g. ``"Distance"``) and the active
    unit is appended at draw time.  If no unit is available (empty
    ``unit``), the label is returned unchanged rather than being
    decorated with a hard-coded ``(m)`` fallback.

    The label may already contain a unit like ``"Distance (m)"`` from
    a previous version — that stale string is stripped before the
    active unit is appended, so the result always reflects the live
    project unit.  (The previous implementation preserved the literal
    text, which silently lied about the project's units.)
    """
    import re as _re
    s = str(template or "").strip()
    if not s:
        return s
    # Strip any stale unit suffix in parentheses at the end of the label.
    s = _re.sub(r"\s*\([A-Za-z0-9/]+\)\s*$", "", s).rstrip()
    if not unit:
        return s
    return f"{s} ({unit})"


def _cmap_lookup(cmap_name: str):
    """Return a matplotlib colormap by name, or fall back to 'viridis'."""
    try:
        import matplotlib.cm as _cm
        return getattr(_cm, cmap_name, _cm.viridis)
    except Exception:
        return None


class NetworkProfilePlotWidget(QtWidgets.QWidget):
    """Reusable matplotlib widget that renders a ProfileArrays."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._options = ProfileOptions()
        self._profile: Optional[ProfileArrays] = None
        self._colorbar = None  # Live matplotlib colorbar (or None)
        # Tracks whether the current ``self._colorbar`` was created for a
        # fill-metric render (True) or a no-overlay render (False).
        # When the fill state changes, the colorbar is removed and
        # recreated; when it stays the same, the existing colorbar is
        # updated in place.  This keeps the plot area at a fixed size
        # across timestep changes.
        self._colorbar_fill_state: Optional[bool] = None

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        if FigureCanvasQt is None:
            self._error_lbl = QtWidgets.QLabel("matplotlib not available")
            root.addWidget(self._error_lbl)
            self._figure = None
            self._ax = None
            self._canvas = None
            self._colorbar = None
            return

        self._figure = Figure(figsize=(8, 4.5))
        # Reserve space on the right for the colorbar so the plot area
        # stays at a fixed width — tight_layout() would re-flow the
        # axes every time the colorbar is recreated, shrinking the
        # plot.  Adjust these margins once at construction and never
        # again.
        self._figure.subplots_adjust(left=0.10, right=0.88, top=0.95, bottom=0.12)
        self._ax = self._figure.add_subplot(111)
        self._canvas = FigureCanvasQt(self._figure)
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        root.addWidget(self._toolbar)
        root.addWidget(self._canvas, stretch=1)

    def set_options(self, options: ProfileOptions):
        self._options = options
        if self._profile is not None:
            self.draw_profile(self._profile)

    # Metrics that drive the water-fill colormap (per-cell color shading),
    # matching the existing single-link PG viewer's _prof_fill_key options.
    _FILL_METRICS = ("depth", "velocity", "flow")
    # Metrics that are drawn as a line overlay (head only — matches the
    # single-link PG viewer's _prof_var_key behaviour for derived metrics).
    _LINE_METRICS = ("head",)

    def draw_profile(
        self,
        profile: ProfileArrays,
        variable: str = "—none—",
    ):
        """Render the profile.

        Args:
            profile: The chain profile arrays.
            variable: One of:
                * ``"—none—"`` — no overlay
                * one of ``_FILL_METRICS`` (``depth``/``velocity``/``flow``)
                  — colors the water fill segment-by-segment
                * one of ``_LINE_METRICS`` (``head``) — overlays an
                  explicit head line on top of the standard HGL line

        The fill/colormap update is independent of the line overlay:
        depth/velocity/flow use the colormap, head uses a line.
        """
        if self._ax is None:
            return
        self._profile = profile
        opts = self._options
        unit = _length_unit() or ""

        has_fill_now = variable in self._FILL_METRICS
        # Toggle colorbar visibility based on fill state.  We keep the
        # colorbar axes around (rather than removing/recreating it)
        # so its position never changes and the plot area never
        # shifts around.  This is what was causing the plot to shrink
        # on every timestep change.
        if self._colorbar is not None and self._colorbar_fill_state != has_fill_now:
            # Fill state changed: toggle the colorbar's visibility
            # instead of recreating it.  When switching from no-fill
            # to fill, the colorbar was hidden — show it.  When
            # switching from fill to no-fill, the colorbar was shown
            # — hide it.
            self._colorbar.ax.set_visible(has_fill_now)
            self._colorbar_fill_state = has_fill_now
            if self._canvas is not None:
                self._canvas.draw_idle()
        self._ax.clear()

        if profile.station_m.size == 0:
            self._ax.text(
                0.5, 0.5, "No chain selected", ha="center", va="center",
                transform=self._ax.transAxes,
            )
            self._canvas.draw()
            return

        # ── Compute the per-cell edge stations ────────────────────────
        # Cell centers are at profile.station_m; the per-cell Polygon
        # patches extend half a sub-step to each side.  To avoid the
        # line plots visibly ending before the polygon does (the
        # previous bug), we extend the line x-coordinates to match the
        # polygon's full extent.  The edge stations are also used as
        # the x for the solid water fill so it lines up with the
        # polygons.
        x_edges = _cell_edge_stations(profile.station_m)

        has_fill = variable in self._FILL_METRICS

        # ── 1. Water polygon: invert → HGL ────────────────────────────
        # If a fill metric is active, suppress the solid blue base layer
        # entirely: the per-cell colormap polygons at full opacity will
        # fill the same region with the colormap.  Otherwise keep the
        # solid blue base layer (semi-transparent) so the water is
        # visible with no overlay.
        if not has_fill:
            self._ax.fill_between(
                profile.station_m, profile.invert_m, profile.hgl_m,
                color=opts.water_color, alpha=0.45, linewidth=0,
            )

        # ── 1b. Per-cell colormap fill (depth/velocity/flow only) ──
        # Drawn at full opacity (alpha=1.0) so the colormap shows the
        # true value of each cell — no blending between adjacent
        # segments, so the colorbar legend matches what's on screen.
        if has_fill:
            try:
                fill_values, _ = profile_at_variable(profile, variable)
                self._draw_segment_fill(profile, fill_values, opts)
            except ValueError:
                pass

        # ── 2. Invert line (single polyline, extended to polygon range) ─
        # The line covers the same x-extent as the polygons so its
        # endpoints aren't visible as a gap on the first/last cell.
        self._ax.plot(
            x_edges, np.concatenate([[profile.invert_m[0]], profile.invert_m, [profile.invert_m[-1]]]),
            color=opts.invert_color,
            linewidth=2 if opts.thick_lines else 1,
        )

        # ── 3. Crown line (dashed, extended to polygon range) ─────
        self._ax.plot(
            x_edges, np.concatenate([[profile.crown_m[0]], profile.crown_m, [profile.crown_m[-1]]]),
            color=opts.crown_color,
            linewidth=1.5 if opts.thick_lines else 1,
            linestyle="--",
        )

        # ── 5. HGL line (water surface, extended to polygon range) ─
        self._ax.plot(
            x_edges, np.concatenate([[profile.hgl_m[0]], profile.hgl_m, [profile.hgl_m[-1]]]),
            color=opts.water_color,
            linewidth=2,
        )

        # ── 6. Node markers (small rectangles at each node endpoint)
        for s, nid in zip(profile.node_stations, profile.node_ids):
            inv = float(_invert_at(s, profile))
            self._ax.add_patch(
                _make_rect_xy(s - 0.25, inv, 0.5, 0.5,
                              opts.conduit_color),
            )
            if opts.node_labels_on_plot:
                self._ax.text(
                    s, inv + 0.6, nid,
                    ha="center", va="bottom", fontsize=opts.font_size_pt,
                )

        # ── 7. Head line overlay (only when explicitly selected) ────
        # Drawn last so it sits on top of the standard HGL line.  Uses
        # a different colour and dash style so the user can tell the
        # overlay is active.  This is the only "line" metric — every
        # other option in the combo drives a colormap fill instead.
        if variable in self._LINE_METRICS:
            try:
                vals, stations = profile_at_variable(profile, variable)
                if vals.size:
                    # Extend the line endpoints by half a sub-step on
                    # each side so it covers the same x-range as the
                    # invert/crown/HGL lines (the cell-edge polygon
                    # range), avoiding a visible gap at the first/last
                    # point.
                    head_x = x_edges
                    head_y = np.concatenate([
                        [vals[0]], vals, [vals[-1]],
                    ])
                    self._ax.plot(
                        head_x, head_y,
                        color="#CC3366", linewidth=2.0, linestyle="--",
                        label=variable,
                    )
            except ValueError:
                pass

        # ── Axis labels (unit-aware) ──────────────────────────────
        self._ax.set_xlabel(
            _resolve_axis_label(opts.x_label, unit),
            fontsize=opts.font_size_pt,
        )
        self._ax.set_ylabel(
            _resolve_axis_label(opts.y_label, unit),
            fontsize=opts.font_size_pt,
        )

        # ── Auto / manual Y range ──────────────────────────────────
        if not opts.auto_scale:
            self._ax.set_ylim(opts.y_min, opts.y_max)
        else:
            all_y = np.concatenate([
                profile.invert_m, profile.crown_m, profile.hgl_m,
            ])
            all_y = all_y[~np.isnan(all_y)]
            if all_y.size:
                self._ax.set_ylim(
                    float(np.floor(all_y.min())) - 1,
                    float(np.ceil(all_y.max())) + 1,
                )

        self._ax.grid(True, alpha=0.3)
        # Skip tight_layout() — it would re-arrange the axes every time
        # the colorbar is recreated, shrinking the plot.  The subplot
        # margins are fixed at construction (see __init__).
        self._canvas.draw()

    def _draw_segment_fill(self, profile, fill_values, opts):
        """Optional per-cell colored fill overlay (when ``fill_metric`` is set).

        The water polygon is rendered as a single connected shape
        covering the entire invert→HGL region between the first cell's
        left edge and the last cell's right edge.  The polygon's upper
        edge (HGL) and lower edge (invert) are interpolated linearly
        between cell centers, so adjacent cells with different values
        share a sloped edge — no visible "stairstep" rectangles.

        The fill colour is the per-cell value's normalized position on
        the colormap.  Each cell's color is blended across its
        footprint (using a Polygon that includes 4 vertices per cell,
        with the two edge vertices interpolated to the neighbour's
        values), so the colour transitions smoothly along the chain
        when neighbouring cells have different fill values.

        The colorbar is created once and updated on every redraw via
        ``_refresh_colorbar``.  Stale colorbars from a previous render
        are removed first so the legend never accumulates.
        """
        n = profile.station_m.size
        if n < 1 or fill_values is None or fill_values.size != n:
            # The fill metric isn't usable; toggle the colorbar off
            # if it's currently visible.  Keep the colorbar axes
            # around so its position doesn't move.
            if self._colorbar is not None:
                self._colorbar.ax.set_visible(False)
                self._colorbar_fill_state = False
                if self._canvas is not None:
                    self._canvas.draw_idle()
            return
        # Suppress all-NaN slice warning: if the metric is entirely
        # missing for this run (e.g. no flow rows in the GPKG), the
        # colormap can't be built — fall back to no overlay.
        with np.errstate(invalid="ignore"):
            vmin = float(np.nanmin(fill_values))
            vmax = float(np.nanmax(fill_values))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = 0.0, 1.0
        cmap = _cmap_lookup("viridis")
        if cmap is None:
            if self._colorbar is not None:
                self._colorbar.ax.set_visible(False)
                self._colorbar_fill_state = False
            return
        from matplotlib.patches import Polygon
        norm_range = vmax - vmin

        # ── Compute cell-edge x positions ─────────────────────────────
        # Each cell's footprint spans from x_left to x_right, where
        # x_left and x_right are the midpoints between the cell's
        # center and its neighbours' centers (or the extrapolated
        # half-step past the first/last cell).
        if n == 1:
            x_left = np.array([float(profile.station_m[0]) - 0.5])
            x_right = np.array([float(profile.station_m[0]) + 0.5])
        else:
            x_left = np.empty(n)
            x_right = np.empty(n)
            for i in range(n):
                if i == 0:
                    x_left[i] = (float(profile.station_m[0])
                                 - 0.5 * (float(profile.station_m[1])
                                          - float(profile.station_m[0])))
                else:
                    x_left[i] = 0.5 * (float(profile.station_m[i - 1])
                                      + float(profile.station_m[i]))
                if i == n - 1:
                    x_right[i] = (float(profile.station_m[-1])
                                  + 0.5 * (float(profile.station_m[-1])
                                           - float(profile.station_m[-2])))
                else:
                    x_right[i] = 0.5 * (float(profile.station_m[i])
                                       + float(profile.station_m[i + 1]))

        # ── Interpolate invert / HGL at each cell edge ──────────────
        # Each cell's left edge picks up the average invert / HGL of
        # itself and its left neighbour (or extrapolates for cell 0).
        # Same on the right.  Adjacent cells now share the same edge
        # invert/HGL, so the polygon's upper and lower edges flow
        # continuously across cell boundaries.
        inv_left = np.empty(n)
        hgl_left = np.empty(n)
        inv_right = np.empty(n)
        hgl_right = np.empty(n)
        for i in range(n):
            inv_i = float(profile.invert_m[i])
            hgl_i = float(profile.hgl_m[i])
            if i == 0:
                inv_left[i] = inv_i
                hgl_left[i] = hgl_i
            else:
                inv_left[i] = 0.5 * (float(profile.invert_m[i - 1]) + inv_i)
                hgl_left[i] = 0.5 * (float(profile.hgl_m[i - 1]) + hgl_i)
            if i == n - 1:
                inv_right[i] = inv_i
                hgl_right[i] = hgl_i
            else:
                inv_right[i] = 0.5 * (inv_i + float(profile.invert_m[i + 1]))
                hgl_right[i] = 0.5 * (hgl_i + float(profile.hgl_m[i + 1]))

        # ── Draw one trapezoid per cell ──────────────────────────────
        drawn_count = 0
        for i in range(n):
            if not (np.isfinite(profile.invert_m[i])
                    and np.isfinite(profile.hgl_m[i])
                    and np.isfinite(fill_values[i])):
                continue
            t = float(np.clip(
                (float(fill_values[i]) - vmin) / norm_range, 0.0, 1.0,
            ))
            rgb = cmap(t)
            # Trapezoid: 4 vertices per cell, with the upper and lower
            # edges interpolated from the neighbour at the shared
            # boundary.  This produces a connected polygon that fills
            # the invert→HGL region continuously along the chain,
            # not stairstep rectangles.
            poly = Polygon(
                [
                    (float(x_left[i]),  inv_left[i]),   # bottom-left
                    (float(x_right[i]), inv_right[i]),  # bottom-right
                    (float(x_right[i]), hgl_right[i]),  # top-right
                    (float(x_left[i]),  hgl_left[i]),   # top-left
                ],
                closed=True,
                facecolor=rgb, edgecolor="none", alpha=1.0,
            )
            self._ax.add_patch(poly)
            drawn_count += 1
        if drawn_count == 0:
            # No polygons were drawn (all cells had NaN fill values).
            # Toggle the colorbar off if it's visible, but keep the
            # colorbar axes around so its position doesn't shift.
            if self._colorbar is not None:
                self._colorbar.ax.set_visible(False)
                self._colorbar_fill_state = False
                if self._canvas is not None:
                    self._canvas.draw_idle()
            return
        self._refresh_colorbar(cmap, vmin, vmax, opts)

    def _refresh_colorbar(self, cmap, vmin, vmax, opts):
        """Create or update a single colorbar for the active colormap.

        On the first call, ``figure.colorbar()`` is invoked with a
        fixed ``fraction`` / ``pad`` so the colorbar axes land at a
        stable position (right of the main axes).  On subsequent
        redraws the existing colorbar is reused — only its mappable's
        norm and cmap are updated, then ``update_normal()`` refreshes
        the gradient and tick labels.  This avoids both the "colorbar
        accumulates one per timestep" bug and the "plot shrinks because
        a new colorbar keeps stealing margin" bug.
        """
        import matplotlib.cm as _cm
        import matplotlib.colors as _colors

        norm = _colors.Normalize(vmin=vmin, vmax=vmax)
        if self._colorbar is None:
            sm = _cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            # Anchor the colorbar to the figure's right margin (which
            # is fixed at construction via subplots_adjust).  Without
            # these explicit args matplotlib's auto-placement can shift
            # the colorbar between redraws, which is what was shrinking
            # the plot on every timestep change.
            self._colorbar = self._figure.colorbar(
                sm, ax=self._ax, fraction=0.04, pad=0.02,
            )
            # Pin the colorbar axes to its initial position so it
            # never moves on subsequent redraws.
            self._colorbar.ax.set_position([
                0.90, 0.12, 0.015, 0.83,  # [left, bottom, width, height]
            ])
            self._colorbar_fill_state = True
        else:
            sm = self._colorbar.mappable
            sm.set_norm(norm)
            sm.set_cmap(cmap)
            sm.set_array([])
            self._colorbar.update_normal()
        unit = _length_unit()
        label = f"value ({unit})" if unit else "value"
        self._colorbar.set_label(label, fontsize=opts.font_size_pt)
        if self._canvas is not None:
            self._canvas.draw_idle()

    def _remove_colorbar(self):
        """Remove the existing colorbar (if any) from the figure."""
        if self._colorbar is None:
            return
        try:
            self._colorbar.remove()
        except Exception:
            # Older matplotlib API — just clear axes
            try:
                self._colorbar.ax.clear()
                self._colorbar.ax.set_visible(False)
            except Exception:
                pass
        self._colorbar = None
        self._colorbar_fill_state = None
        # Schedule a redraw so the empty colorbar slot disappears from
        # the screen promptly.
        if self._canvas is not None:
            self._canvas.draw_idle()

    def export_png(self, filepath: str):
        if self._figure is not None:
            self._figure.savefig(filepath, dpi=150, bbox_inches="tight")

    def export_csv(self, filepath: str, profile: ProfileArrays):
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "station", "invert", "crown_offset", "crown",
                "ground", "hgl", "depth", "velocity", "flow",
            ])
            n = profile.station_m.size
            arrs = [
                profile.station_m, profile.invert_m, profile.crown_offset_m,
                profile.crown_m, profile.hgl_m,
                profile.depth_m, profile.velocity_ms, profile.flow_cms,
            ]
            for i in range(n):
                writer.writerow([f"{a[i]:.6g}" for a in arrs])


def _invert_at(station: float, profile: ProfileArrays) -> float:
    """Linearly interpolate the cell-center invert at an arbitrary station.

    Used to draw the bottom of node cylinders at the correct elevation
    when the invert varies along the chain.
    """
    x = profile.station_m
    inv = profile.invert_m
    if x.size == 0:
        return 0.0
    if station <= x[0]:
        return float(inv[0])
    if station >= x[-1]:
        return float(inv[-1])
    for j in range(x.size - 1):
        if x[j] <= station <= x[j + 1]:
            l, r = float(x[j]), float(x[j + 1])
            if r == l:
                return float(inv[j])
            t = (station - l) / (r - l)
            return float(inv[j] * (1 - t) + inv[j + 1] * t)
    return float(inv[-1])


def _cell_edge_stations(station_m: np.ndarray) -> np.ndarray:
    """Return the full x-extent of the cell-centre profile for line plots.

    The per-cell colormap Polygon patches are drawn from
    ``(x[i] - half_w_left, ...)`` to ``(x[i] + half_w_right, ...)``
    where half_w is half the gap to the neighbouring cell.  Without
    extension, a line plot through ``station_m`` ends at the first
    and last cell centers, leaving the polygon at each end visibly
    extending past the line.

    The returned array has length ``n + 2`` and starts at the
    first cell's left edge, steps through every cell center, and
    ends at the last cell's right edge.  Pair this with a y-array
    of the same length (e.g. ``np.concatenate([[y[0]], y, [y[-1]]])``)
    to draw a line that fully covers the polygon's x-extent.

    For n=5 (stations at 0, 25, 50, 75, 100) this returns
    ``[-12.5, 0, 25, 50, 75, 100, 112.5]`` so the line endpoints
    align with the polygon's left and right edges.
    """
    n = int(station_m.size)
    if n == 0:
        return np.zeros(0)
    if n == 1:
        x0 = float(station_m[0])
        return np.array([x0 - 0.5, x0, x0 + 0.5])
    # First and last "edges" extend a full sub-step past the first
    # and last cell centers so the polygon is symmetric about each
    # cell center.  Interior points are the cell centers themselves
    # (not the inter-cell edges), so the line passes through every
    # cell-center value.
    half_step_left = float(station_m[1] - station_m[0]) * 0.5
    half_step_right = float(station_m[-1] - station_m[-2]) * 0.5
    first = float(station_m[0]) - half_step_left
    last = float(station_m[-1]) + half_step_right
    return np.concatenate([[first], station_m, [last]])


def _make_rect_xy(x, y, w, h, color):
    from matplotlib.patches import Rectangle
    return Rectangle((x, y), w, h, facecolor=color, edgecolor="black", linewidth=0.5)
