"""tests/test_network_profile_plot_widget.py"""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from qgis.PyQt.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

from swe2d.workbench.services.profile_pipeline_service import ProfileArrays
from swe2d.workbench.dialogs.network_profile_plot_widget import (
    NetworkProfilePlotWidget,
    _length_unit,
    _resolve_axis_label,
)
from swe2d.workbench.dialogs.profile_options_dialog import ProfileOptions


def _make_profile(n=20):
    station_m = np.linspace(0, 100, n)
    invert_m = np.full(n, 0.0)
    crown_m = np.full(n, 2.0)
    crown_offset_m = np.full(n, 2.0)
    depth_m = np.linspace(0.5, 1.5, n)
    hgl_m = invert_m + depth_m
    velocity_ms = np.linspace(0.5, 1.0, n)
    flow_cms = np.linspace(1.0, 3.0, n)
    node_stations = [0.0, 50.0, 100.0]
    node_ids = ["N1", "N2", "N3"]
    link_boundaries = [(0, "L1"), (10, "L2")]
    return ProfileArrays(
        station_m=station_m, invert_m=invert_m,
        crown_offset_m=crown_offset_m, crown_m=crown_m,
        hgl_m=hgl_m, depth_m=depth_m,
        velocity_ms=velocity_ms, flow_cms=flow_cms,
        node_stations=node_stations, node_ids=node_ids,
        link_boundaries=link_boundaries, crown_style="circular",
    )


class TestNetworkProfilePlotWidget(unittest.TestCase):
    def test_draw_profile_renders_axes(self):
        w = NetworkProfilePlotWidget()
        p = _make_profile()
        # Should not raise
        w.draw_profile(p)
        self.assertIsNotNone(w._ax)

    def test_draw_with_depth_variable(self):
        w = NetworkProfilePlotWidget()
        p = _make_profile()
        w.draw_profile(p, variable="depth")
        # depth overlay line drawn
        self.assertGreater(len(w._ax.lines), 0)

    def test_export_png_creates_file(self):
        w = NetworkProfilePlotWidget()
        w.draw_profile(_make_profile())
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        try:
            w.export_png(tmp.name)
            self.assertGreater(os.path.getsize(tmp.name), 0)
        finally:
            os.unlink(tmp.name)

    def test_export_csv_has_correct_header(self):
        w = NetworkProfilePlotWidget()
        p = _make_profile()
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        tmp.close()
        try:
            w.export_csv(tmp.name, p)
            with open(tmp.name) as f:
                header = f.readline().strip()
            # Headers are unit-agnostic ("station", "invert", "hgl", ...)
            # because axis labels carry the unit string (e.g. "Distance (m)").
            self.assertIn("station", header)
            self.assertIn("invert", header)
            self.assertIn("hgl", header)
            self.assertIn("crown", header)
        finally:
            os.unlink(tmp.name)

    def test_draw_empty_profile_no_raise(self):
        w = NetworkProfilePlotWidget()
        empty = ProfileArrays(
            station_m=np.zeros(0), invert_m=np.zeros(0),
            crown_offset_m=np.zeros(0), crown_m=np.zeros(0),
            velocity_ms=np.zeros(0), flow_cms=np.zeros(0),
            node_stations=[], node_ids=[], link_boundaries=[], crown_style="circular",
        )
        w.draw_profile(empty)
        # no exception


class TestUnitAwareAxisLabels(unittest.TestCase):
    """Regression: hard-coded 'Distance (m)' violates the project unit rule.
    The widget must append the active length unit, not assume SI.

    The previous implementation preserved explicit labels like
    'Distance (m)' verbatim and fell back to 'm' when the units module
    was unreachable — both behaviours silently lied about the project's
    units.  The fix: always reflect the active unit; never inject a
    hard-coded unit; never preserve a stale hard-coded one.
    """

    def test_resolve_appends_unit_to_template(self):
        self.assertEqual(_resolve_axis_label("Distance", "m"), "Distance (m)")
        self.assertEqual(_resolve_axis_label("Distance", "ft"), "Distance (ft)")

    def test_resolve_substitutes_active_unit_even_when_template_has_paren(self):
        # The old behaviour preserved "Distance (m)" even when the active
        # unit was "ft".  That was a bug: the user expects the label to
        # always reflect the live unit, not a hard-coded string.
        # Since the project hard-codes (m), the result should still
        # be "Distance (m)" for an (m) project — but it must come from
        # the active unit, not from the literal text.
        # If active unit is "ft", the result must be "Distance (ft)".
        self.assertEqual(
            _resolve_axis_label("Distance (m)", "ft"),
            "Distance (ft)",
        )
        self.assertEqual(
            _resolve_axis_label("Distance (ft)", "m"),
            "Distance (m)",
        )

    def test_resolve_omits_unit_when_unavailable(self):
        # If no unit is available, the label is returned as-is — the
        # widget must NEVER inject a hard-coded '(m)'.
        self.assertEqual(_resolve_axis_label("Distance", ""), "Distance")
        # Stale "(m)" suffix is stripped — the label is rebuilt from the
        # template's text, not preserved as a literal.
        self.assertEqual(_resolve_axis_label("Distance (m)", ""), "Distance")

    def test_resolve_empty_template_passes_through(self):
        self.assertEqual(_resolve_axis_label("", "m"), "")
        self.assertEqual(_resolve_axis_label("", ""), "")

    def test_length_unit_returns_empty_string_when_units_unavailable(self):
        # When the units module is unreachable (e.g. no project loaded,
        # or the units module is mid-init), the function returns the
        # empty string — NEVER a hard-coded 'm' fallback.
        with patch.dict("sys.modules", {"swe2d": None, "swe2d.units": None}):
            u = _length_unit()
        self.assertEqual(u, "")


class TestColorbarLifecycle(unittest.TestCase):
    """Regression: every redraw used to call figure.colorbar() without
    removing the previous instance, so a new colorbar was added on
    top of the old one for every timestep change.  The widget must
    keep at most one colorbar at a time."""

    def _make_profile_with_n(self, n):
        station_m = np.linspace(0, 100, n)
        invert_m = np.full(n, 4.0)
        crown_offset_m = np.full(n, 2.0)
        crown_m = invert_m + crown_offset_m
        depth_m = np.linspace(0.5, 1.5, n)
        hgl_m = invert_m + depth_m
        velocity_ms = np.linspace(0.5, 1.0, n)
        flow_cms = np.linspace(1.0, 3.0, n)
        node_stations = [0.0, 100.0]
        node_ids = ["N1", "N2"]
        link_boundaries = [(0, "L1")]
        return ProfileArrays(
            station_m=station_m, invert_m=invert_m,
            crown_offset_m=crown_offset_m, crown_m=crown_m,
            velocity_ms=velocity_ms, flow_cms=flow_cms,
            node_stations=node_stations, node_ids=node_ids,
            link_boundaries=link_boundaries, crown_style="circular",
        )

    def test_redraws_with_fill_metric_do_not_accumulate_colorbars(self):
        w = NetworkProfilePlotWidget()
        p = self._make_profile_with_n(5)
        for _ in range(5):
            w.draw_profile(p, variable="depth")
        # Only one colorbar is ever attached to the figure, regardless
        # of how many times draw_profile is called.
        colorbars = [ax for ax in w._figure.axes if ax is not w._ax]
        # The colorbar lives on its own axes; with the fix there is
        # exactly one.  Without the fix there would be five.
        self.assertEqual(
            len(colorbars), 1,
            f"Expected exactly 1 colorbar after 5 redraws, got "
            f"{len(colorbars)} (colorbar axes {colorbars})",
        )

    def test_colorbar_refreshes_on_each_timestep_change(self):
        """Regression: the colorbar's vmin/vmax must reflect the latest
        fill_values, not stay stuck on the first draw.  This catches
        the bug where ``update_normal(self._canvas.draw_idle())`` was
        passing draw_idle's return value (None) as an argument, which
        silently did nothing and left the colorbar showing the first
        timestep's range forever."""
        w = NetworkProfilePlotWidget()

        # First redraw with depth range [1.0, 2.0] (n=5 linspace from 1 to 2)
        p1 = self._make_profile_with_n(5)
        p1 = ProfileArrays(
            **{**{f: getattr(p1, f) for f in (
                "station_m", "invert_m", "crown_offset_m", "crown_m",
                "node_stations", "node_ids", "link_boundaries", "crown_style",
            )},
               "depth_m": np.linspace(1.0, 2.0, 5)}
        )
        w.draw_profile(p1, variable="depth")
        sm1 = w._colorbar.mappable
        vmin1, vmax1 = float(sm1.norm.vmin), float(sm1.norm.vmax)
        self.assertAlmostEqual(vmin1, 1.0)
        self.assertAlmostEqual(vmax1, 2.0)

        # Second redraw with depth range [10.0, 20.0]
        p2 = ProfileArrays(
            **{**{f: getattr(p1, f) for f in (
                "station_m", "invert_m", "crown_offset_m", "crown_m",
                "node_stations", "node_ids", "link_boundaries", "crown_style",
            )},
               "depth_m": np.linspace(10.0, 20.0, 5)}
        )
        w.draw_profile(p2, variable="depth")
        sm2 = w._colorbar.mappable
        vmin2, vmax2 = float(sm2.norm.vmin), float(sm2.norm.vmax)
        self.assertAlmostEqual(
            vmin2, 10.0,
            msg=f"colorbar vmin did not refresh: expected 10.0, got {vmin2}",
        )
        self.assertAlmostEqual(
            vmax2, 20.0,
            msg=f"colorbar vmax did not refresh: expected 20.0, got {vmax2}",
        )

        # And a third redraw with a completely different range to
        # be sure the colorbar is not just "close enough".
        p3 = ProfileArrays(
            **{**{f: getattr(p1, f) for f in (
                "station_m", "invert_m", "crown_offset_m", "crown_m",
                "node_stations", "node_ids", "link_boundaries", "crown_style",
            )},
               "depth_m": np.linspace(-5.0, 5.0, 5)}
        )
        w.draw_profile(p3, variable="depth")
        sm3 = w._colorbar.mappable
        vmin3, vmax3 = float(sm3.norm.vmin), float(sm3.norm.vmax)
        self.assertAlmostEqual(vmin3, -5.0)
        self.assertAlmostEqual(vmax3, 5.0)

    def test_colorbar_hidden_when_fill_metric_turned_off(self):
        """When a fill metric is turned off, the colorbar stays in
        place (its axes are reused) but is hidden so the plot area
        doesn't shift.  This is what keeps the plot from shrinking on
        every redraw — recreating the colorbar was the culprit."""
        w = NetworkProfilePlotWidget()
        p = self._make_profile_with_n(5)
        w.draw_profile(p, variable="depth")
        self.assertTrue(w._colorbar.ax.get_visible())
        self.assertEqual(w._colorbar_fill_state, True)
        # Now redraw without a fill metric
        w.draw_profile(p, variable="—none—")
        # The colorbar is hidden but still in the figure so its
        # position doesn't move.
        self.assertIsNotNone(w._colorbar)
        self.assertFalse(w._colorbar.ax.get_visible())
        self.assertEqual(w._colorbar_fill_state, False)

    def test_colorbar_starts_unset(self):
        w = NetworkProfilePlotWidget()
        self.assertIsNone(w._colorbar)


def _hex_to_rgb(hex_color: str):
    """Convert '#A0763D' (or 'A0763D') into a normalised (r, g, b) tuple."""
    s = hex_color.lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    r = int(s[0:2], 16) / 255.0
    g = int(s[2:4], 16) / 255.0
    b = int(s[4:6], 16) / 255.0
    return (r, g, b)


class TestGroundLineOffByDefault(unittest.TestCase):
    """Regression: the ground/rim line was showing values that
    were clearly wrong (the rim_elev interpolation across nodes
    does not match the mesh bed for many GPKG layouts).  Default
    to OFF so the line doesn't ship with bad data; users can
    re-enable it via the Options dialog if they want to debug."""

    def test_default_options_have_ground_line_off(self):
        from swe2d.workbench.dialogs.profile_options_dialog import ProfileOptions
        opts = ProfileOptions()
        self.assertFalse(opts.ground_line_visible)

    def test_draw_profile_does_not_render_ground_line_by_default(self):
        from swe2d.workbench.dialogs.profile_options_dialog import ProfileOptions
        w = NetworkProfilePlotWidget()
        # Default options must keep the ground line off.
        self.assertFalse(w._options.ground_line_visible)

        n = 5
        station_m = np.linspace(0, 100, n)
        invert_m = np.full(n, 4.0)
        crown_offset_m = np.full(n, 2.0)
        crown_m = invert_m + crown_offset_m
        depth_m = np.full(n, 1.0)
        hgl_m = invert_m + depth_m
        p = ProfileArrays(
            station_m=station_m, invert_m=invert_m,
            crown_offset_m=crown_offset_m, crown_m=crown_m,
            velocity_ms=np.zeros(n), flow_cms=np.zeros(n),
            node_stations=[0.0, 100.0], node_ids=["N1", "N2"],
            link_boundaries=[(0, "L1")], crown_style="circular",
        )
        w.draw_profile(p)

        # With ground_line_visible=False, no line should reference
        # the ground colour.  Match against the ground colour hex.
        ground_hex = "#A0763D"
        ground_lines = [
            ln for ln in w._ax.get_lines()
            if ln.get_color() in (ground_hex, _hex_to_rgb(ground_hex))
        ]
        self.assertEqual(
            ground_lines, [],
            "Ground line was drawn even though ground_line_visible=False",
        )


# Accent colour used for the head line overlay.  Defined here so
# the regression test can match against it.
_HEAD_OVERLAY_COLOR = "#CC3366"


def _lines_with_color(ax, hex_or_rgb):
    """Return ax.lines whose colour matches ``hex_or_rgb`` (loose)."""
    target = hex_or_rgb
    out = []
    for ln in ax.get_lines():
        c = ln.get_color()
        # get_color() returns '#rrggbb' or rgba tuple depending on version.
        if isinstance(target, str) and isinstance(c, str) and c.lower() == target.lower():
            out.append(ln)
        elif isinstance(target, tuple) and isinstance(c, tuple):
            # match rgb with tolerance
            if all(abs(a - b) < 1e-3 for a, b in zip(c, target[:3])):
                out.append(ln)
    return out


class TestFillMetricRouting(unittest.TestCase):
    """Regression: the variable combo was previously treated as a
    'line overlay' for *every* metric, so selecting depth/velocity/flow
    drew a single line through the chain instead of the per-cell
    colormap shading used by the single-link PG viewer.

    The new routing is:
        * depth / velocity / flow → per-cell colormap shading of the
          water polygon (segment-by-segment Polygon patches)
        * head → explicit line overlay (the only 'line' metric)
        * —none— → no overlay
    """

    def _make_profile(self):
        n = 5
        station_m = np.linspace(0, 100, n)
        invert_m = np.full(n, 4.0)
        crown_offset_m = np.full(n, 2.0)
        crown_m = invert_m + crown_offset_m
        depth_m = np.linspace(0.5, 1.5, n)
        hgl_m = invert_m + depth_m
        return ProfileArrays(
            station_m=station_m, invert_m=invert_m,
            crown_offset_m=crown_offset_m, crown_m=crown_m,
            velocity_ms=np.zeros(n), flow_cms=np.zeros(n),
            node_stations=[0.0, 100.0], node_ids=["N1", "N2"],
            link_boundaries=[(0, "L1")], crown_style="circular",
        )

    def _count_polygon_patches(self, ax):
        """Count matplotlib.patches.Polygon instances (colormap fill), excluding Rectangle (node cylinders)."""
        from matplotlib.patches import Polygon
        return sum(1 for p in ax.patches if isinstance(p, Polygon))

    def test_depth_produces_colormap_polygons_not_a_line(self):
        w = NetworkProfilePlotWidget()
        p = self._make_profile()
        w.draw_profile(p, variable="depth")
        # Colormap fill: Polygon patches (one per cell).
        self.assertGreaterEqual(
            self._count_polygon_patches(w._ax), 1,
            "depth should produce per-cell colormap Polygon patches",
        )
        # No line at the head-overlay colour.
        head_lines = _lines_with_color(w._ax, _HEAD_OVERLAY_COLOR)
        self.assertEqual(
            head_lines, [],
            "depth should NOT produce a head-overlay line; it should "
            "drive the per-cell colormap fill instead",
        )

    def test_velocity_produces_colormap_polygons_not_a_line(self):
        w = NetworkProfilePlotWidget()
        p = self._make_profile()
        w.draw_profile(p, variable="velocity")
        self.assertGreaterEqual(self._count_polygon_patches(w._ax), 1)
        self.assertEqual(_lines_with_color(w._ax, _HEAD_OVERLAY_COLOR), [])

    def test_flow_produces_colormap_polygons_not_a_line(self):
        w = NetworkProfilePlotWidget()
        p = self._make_profile()
        w.draw_profile(p, variable="flow")
        self.assertGreaterEqual(self._count_polygon_patches(w._ax), 1)
        self.assertEqual(_lines_with_color(w._ax, _HEAD_OVERLAY_COLOR), [])

    def test_head_produces_line_overlay_not_polygons(self):
        w = NetworkProfilePlotWidget()
        p = self._make_profile()
        w.draw_profile(p, variable="head")
        # No colormap fill polygons when only head is selected.
        self.assertEqual(
            self._count_polygon_patches(w._ax), 0,
            "head should NOT produce colormap fill polygons; it is a "
            "line-only overlay metric",
        )
        # The head line overlay is present.
        head_lines = _lines_with_color(w._ax, _HEAD_OVERLAY_COLOR)
        self.assertEqual(
            len(head_lines), 1,
            "head should produce exactly one head-overlay line",
        )

    def test_none_produces_no_overlay(self):
        w = NetworkProfilePlotWidget()
        p = self._make_profile()
        w.draw_profile(p, variable="—none—")
        # No colormap fill polygons when —none— is selected.
        self.assertEqual(self._count_polygon_patches(w._ax), 0)
        # No head-overlay line either.
        self.assertEqual(_lines_with_color(w._ax, _HEAD_OVERLAY_COLOR), [])
        # No colorbar.
        self.assertIsNone(w._colorbar)

    def test_filling_metrics_trigger_colorbar(self):
        """depth/velocity/flow each create + maintain a colorbar."""
        for fm in ("depth", "velocity", "flow"):
            w = NetworkProfilePlotWidget()
            p = self._make_profile()
            w.draw_profile(p, variable=fm)
            self.assertIsNotNone(
                w._colorbar,
                f"{fm} should produce a colorbar",
            )

    def test_head_does_not_create_colorbar(self):
        """head is a line overlay, not a colormap — no colorbar."""
        w = NetworkProfilePlotWidget()
        p = self._make_profile()
        w.draw_profile(p, variable="head")
        self.assertIsNone(
            w._colorbar,
            "head should not produce a colorbar; only fill metrics do",
        )


class TestLineEndpointsCoverPolygon(unittest.TestCase):
    """Regression: the invert / crown / HGL lines used to end at
    the first and last cell centers, leaving a small gap where the
    per-cell polygon extended past the line.  The lines now cover
    the polygon's full x-extent (one half-sub-step past each end)."""

    def _make_profile(self):
        n = 5
        station_m = np.linspace(0, 100, n)
        invert_m = np.full(n, 4.0)
        crown_offset_m = np.full(n, 2.0)
        crown_m = invert_m + crown_offset_m
        depth_m = np.linspace(0.5, 1.5, n)
        hgl_m = invert_m + depth_m
        return ProfileArrays(
            station_m=station_m, invert_m=invert_m,
            crown_offset_m=crown_offset_m, crown_m=crown_m,
            velocity_ms=np.zeros(n), flow_cms=np.zeros(n),
            node_stations=[0.0, 100.0], node_ids=["N1", "N2"],
            link_boundaries=[(0, "L1")], crown_style="circular",
        )

    def _polygon_x_extent(self, ax):
        from matplotlib.patches import Polygon
        xs = []
        for p in ax.patches:
            if isinstance(p, Polygon):
                verts = p.get_xy()
                xs.append((verts[0][0], verts[2][0]))
        if not xs:
            return None
        return min(l for l, _ in xs), max(r for _, r in xs)

    def test_invert_line_covers_polygon_extent(self):
        w = NetworkProfilePlotWidget()
        p = self._make_profile()
        w.draw_profile(p, variable="depth")
        ext = self._polygon_x_extent(w._ax)
        # The invert line (invert_color) should cover the polygon extent.
        for ln in w._ax.get_lines():
            if ln.get_color().lower() == "#2a2a2a":
                xd = ln.get_xdata()
                self.assertLessEqual(
                    ext[0], xd[0],
                    f"invert line left endpoint {xd[0]} is past polygon left edge {ext[0]}",
                )
                self.assertGreaterEqual(
                    ext[1], xd[-1],
                    f"invert line right endpoint {xd[-1]} is before polygon right edge {ext[1]}",
                )
                return
        self.fail("invert line not found")

    def test_crown_line_covers_polygon_extent(self):
        w = NetworkProfilePlotWidget()
        p = self._make_profile()
        w.draw_profile(p, variable="depth")
        ext = self._polygon_x_extent(w._ax)
        for ln in w._ax.get_lines():
            if ln.get_color().lower() == "#888888":
                xd = ln.get_xdata()
                self.assertLessEqual(ext[0], xd[0])
                self.assertGreaterEqual(ext[1], xd[-1])
                return
        self.fail("crown line not found")

    def test_hgl_line_covers_polygon_extent(self):
        w = NetworkProfilePlotWidget()
        p = self._make_profile()
        w.draw_profile(p, variable="depth")
        ext = self._polygon_x_extent(w._ax)
        for ln in w._ax.get_lines():
            if ln.get_color().lower() == "#3366cc":
                xd = ln.get_xdata()
                self.assertLessEqual(ext[0], xd[0])
                self.assertGreaterEqual(ext[1], xd[-1])
                return
        self.fail("HGL line not found")


class TestFillMetricOpacity(unittest.TestCase):
    """Regression: when a fill metric is active, the per-cell
    polygons must be at full opacity (alpha=1.0) so the colormap
    legend matches the on-screen colors.  When —none— is active, the
    default solid blue water fill (alpha=0.45) is shown."""

    def _make_profile(self):
        n = 5
        station_m = np.linspace(0, 100, n)
        invert_m = np.full(n, 4.0)
        crown_offset_m = np.full(n, 2.0)
        crown_m = invert_m + crown_offset_m
        depth_m = np.linspace(0.5, 1.5, n)
        hgl_m = invert_m + depth_m
        return ProfileArrays(
            station_m=station_m, invert_m=invert_m,
            crown_offset_m=crown_offset_m, crown_m=crown_m,
            velocity_ms=np.zeros(n), flow_cms=np.zeros(n),
            node_stations=[0.0, 100.0], node_ids=["N1", "N2"],
            link_boundaries=[(0, "L1")], crown_style="circular",
        )

    def test_polygons_are_full_alpha_when_fill_metric_active(self):
        w = NetworkProfilePlotWidget()
        p = self._make_profile()
        w.draw_profile(p, variable="depth")
        from matplotlib.patches import Polygon
        polys = [p for p in w._ax.patches if isinstance(p, Polygon)]
        self.assertGreaterEqual(len(polys), 1)
        for poly in polys:
            self.assertEqual(
                poly.get_alpha(), 1.0,
                "colormap fill polygons must be at alpha=1.0 so the "
                "colorbar matches the on-screen color",
            )

    def test_no_blue_base_water_when_fill_metric_active(self):
        """When depth/velocity/flow is selected, the solid blue base
        water layer is suppressed — only the colormap polygons fill the
        invert→HGL region."""
        w = NetworkProfilePlotWidget()
        p = self._make_profile()
        w.draw_profile(p, variable="depth")
        # The solid blue water color is #3366CC.  Look for any
        # PolyCollection (which fill_between produces) whose facecolor
        # matches that hex.
        from matplotlib.collections import PolyCollection
        for coll in w._ax.collections:
            if isinstance(coll, PolyCollection):
                facecolors = coll.get_facecolor()
                # facecolors is shape (N, 4); check first row
                for fc in facecolors:
                    if (abs(fc[0] - 0.2) < 0.05
                            and abs(fc[1] - 0.4) < 0.05
                            and abs(fc[2] - 0.8) < 0.05):
                        self.fail(
                            "Solid blue water fill is present alongside "
                            "the colormap polygons — should be suppressed "
                            "when a fill metric is active"
                        )

    def test_blue_base_water_present_when_no_overlay(self):
        """When —none— is selected, the default solid blue water fill
        (alpha=0.45) is shown so the water is visible."""
        w = NetworkProfilePlotWidget()
        p = self._make_profile()
        w.draw_profile(p, variable="—none—")
        from matplotlib.collections import PolyCollection
        # find the water-color PolyCollection
        found = False
        for coll in w._ax.collections:
            if isinstance(coll, PolyCollection):
                facecolors = coll.get_facecolor()
                for fc in facecolors:
                    if (abs(fc[0] - 0.2) < 0.05
                            and abs(fc[1] - 0.4) < 0.05
                            and abs(fc[2] - 0.8) < 0.05):
                        # alpha should be around 0.45
                        self.assertLessEqual(fc[3], 0.5)
                        found = True
        self.assertTrue(found, "blue base water fill should be present when no overlay")


class TestPolygonShapeAndPlotStability(unittest.TestCase):
    """Regression: the per-cell Polygon was previously a flat-topped
    rectangle (stairstep between cells with different invert/HGL).
    It is now a trapezoid whose top and bottom edges interpolate to
    the neighbour's values at the shared boundary, so the fill
    region flows continuously along the chain.

    Also: redrawing with a different timestep previously shrank the
    plot because the colorbar was recreated on every redraw and
    matplotlib's auto-placement kept stealing margin.  The colorbar
    is now reused (with visible toggled) so the plot area stays
    at a fixed width across timestep changes."""

    def _make_profile(self, invert_m=None, depth_m=None):
        n = 5
        station_m = np.linspace(0, 100, n)
        if invert_m is None:
            invert_m = np.array([4.0, 4.5, 5.0, 5.5, 6.0])
        if depth_m is None:
            depth_m = np.linspace(0.5, 1.5, n)
        crown_offset_m = np.full(n, 2.0)
        crown_m = invert_m + crown_offset_m
        hgl_m = invert_m + depth_m
        return ProfileArrays(
            station_m=station_m, invert_m=invert_m,
            crown_offset_m=crown_offset_m, crown_m=crown_m,
            velocity_ms=np.zeros(n), flow_cms=np.zeros(n),
            node_stations=[0.0, 100.0], node_ids=["N1", "N2"],
            link_boundaries=[(0, "L1")], crown_style="circular",
        )

    def test_polygons_are_trapezoids_with_interpolated_edges(self):
        """When adjacent cells have different invert values, the
        polygon at their shared boundary should pick up the
        interpolated (averaged) invert value, not the cell's own
        value — otherwise the fill has a visible step."""
        w = NetworkProfilePlotWidget()
        p = self._make_profile()
        w.draw_profile(p, variable="depth")
        from matplotlib.patches import Polygon
        polys = [p for p in w._ax.patches if isinstance(p, Polygon)]
        # Polygon 0's right edge should be at the average of cell 0
        # and cell 1's invert: (4.0 + 4.5) / 2 = 4.25.
        # Polygon 1's left edge should be the same value 4.25
        # (so the two polygons share the edge cleanly).
        poly0_right = polys[0].get_xy()[1]   # bottom-right of cell 0
        poly1_left = polys[1].get_xy()[0]    # bottom-left of cell 1
        self.assertAlmostEqual(
            float(poly0_right[1]), 4.25,
            msg=f"cell 0 right invert should be 4.25 (avg of 4.0+4.5), "
                f"got {poly0_right[1]}",
        )
        self.assertAlmostEqual(
            float(poly1_left[1]), 4.25,
            msg=f"cell 1 left invert should be 4.25 (avg of 4.0+4.5), "
                f"got {poly1_left[1]}",
        )
        # And the HGL edge is also interpolated.
        # cell 0 HGL = 4.0 + 0.5 = 4.5; cell 1 HGL = 4.5 + 0.75 = 5.25
        # shared edge HGL = (4.5 + 5.25) / 2 = 4.875
        poly0_hgl_right = polys[0].get_xy()[2]  # top-right of cell 0
        self.assertAlmostEqual(
            float(poly0_hgl_right[1]), 4.875, places=3,
        )

    def test_plot_width_stable_across_timestep_changes(self):
        """The plot area should not shrink when the timestep changes
        while a fill metric is active.  The colorbar is reused
        (not recreated) so matplotlib's auto-placement can't steal
        margin from the main axes."""
        w = NetworkProfilePlotWidget()
        widths = []
        for d_max in (1.0, 5.0, 10.0, 2.5, 8.0):
            depth_m = np.linspace(0.5, d_max, 5)
            p = self._make_profile(depth_m=depth_m)
            w.draw_profile(p, variable="depth")
            widths.append(float(w._ax.get_position().width))
        # All widths should be equal (no shrinkage across redraws).
        for w_ in widths[1:]:
            self.assertAlmostEqual(
                widths[0], w_, places=6,
                msg=f"plot width changed from {widths[0]} to {w_} "
                    f"after a timestep change",
            )

    def test_colorbar_axes_not_recreated_on_redraw(self):
        """The colorbar's underlying axes object should be the same
        instance across redraws — only its mappable is updated.
        If a new axes is created each time, it shifts the plot."""
        w = NetworkProfilePlotWidget()
        p = self._make_profile()
        w.draw_profile(p, variable="depth")
        first_axes = w._colorbar.ax
        for d_max in (1.0, 5.0, 10.0):
            depth_m = np.linspace(0.5, d_max, 5)
            p = self._make_profile(depth_m=depth_m)
            w.draw_profile(p, variable="depth")
            self.assertIs(
                w._colorbar.ax, first_axes,
                "colorbar axes were recreated on a timestep change",
            )
