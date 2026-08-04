"""Phase 3.7: config round-trip symmetry test.

The canonical serializer is :func:`swe2d.core.builder.widget_state_to_widget_state_dict`
which converts a RunContext into the GUI's versioned widget_state dict
(``{"version": 1, "widgets": {wname: {"type": ..., "value": ...}}}``).

The inverse is :func:`swe2d.core.builder.widget_state_to_flat_params` which
takes the same widget_state dict and produces a flat ``{rc_param_name: value}``
dict that the canonical builder (``build_run_context``) accepts as input.

The round-trip property is:

  ctx  →  widget_state_dict  →  flat_params  →  ctx'

where ``ctx'`` is equivalent to ``ctx`` (same scalar fields).

This test is the acceptance gate for Phase 3.7 — the GUI auto-save path
is wired through this same serializer so the saved widget_state matches
the post-run RunContext.
"""
from __future__ import annotations

import tempfile
import os
import unittest

import numpy as np

from tests.qgis_real_env import ensure_qgis_app, requires_qgis

from tests._swe2d_test_helpers import (
    _make_cartesian_quad_mesh,
    _serialize_and_persist_mesh,
)


# ── Shared mesh fixture ────────────────────────────────────────────────────────

class MeshFixture:
    """Tiny 4×2 quad mesh in a temporary GPKG, used by round-trip tests."""

    NX, NY = 4, 2
    LX, LY = 40.0, 10.0

    def __init__(self):
        self._tmpdir = tempfile.mkdtemp(prefix="rt_test_")
        self.gpkg = os.path.join(self._tmpdir, "rt_mesh.gpkg")
        self.mesh_name = "test_mesh"

    def build(self):
        node_x, node_y, _, cell_nodes, _, _ = _make_cartesian_quad_mesh(
            self.NX, self.NY, self.LX, self.LY,
        )
        node_z = np.zeros_like(node_x)
        bc_n0 = np.empty(0, dtype=np.int32)
        bc_n1 = np.empty(0, dtype=np.int32)
        bc_tp = np.empty(0, dtype=np.int32)
        bc_vl = np.empty(0, dtype=np.float64)
        _serialize_and_persist_mesh(
            self.gpkg, self.mesh_name,
            node_x, node_y, node_z, cell_nodes,
            bc_n0, bc_n1, bc_tp, bc_vl,
        )
        return self

    def close(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)


# ── Scalar fields compared in the round-trip ─────────────────────────────────
# These are the RunContext scalar fields that the canonical serializer maps
# to widget names (via the inverse of WIDGET_TO_RC plus the storage
# checkbox mapping).  Array fields and callbacks are excluded because they're
# loaded from the mesh, not the GUI.  Fields not widget-captured
# (``gravity``, ``k_mann``, ``uniform_inflow_enabled``, …) are excluded
# because the GUI's ``collect_widget_state_for_save`` doesn't emit widgets
# for them and the builder falls back to defaults on the inverse path.
SCALAR_FIELDS = (
    "run_duration_s", "output_interval_s", "dt_cfg",
    "n_mann", "cfl", "h_min",
    "adaptive_cfl_dt", "cuda_graphs_enabled", "swe2d_perf_mode",
    "save_mesh_results", "save_line_results",
    "save_coupling_results", "save_run_log",
    "save_max_only", "inflow_progressive",
    "active_set_hysteresis",
)


@requires_qgis
class TestWidgetStateRoundTrip(unittest.TestCase):
    """ctx → widget_state_dict → flat_params → ctx' preserves scalar fields."""

    @classmethod
    def setUpClass(cls):
        ensure_qgis_app()
        cls.fixture = MeshFixture().build()
        cls.addClassCleanup(cls.fixture.close)

    def _build_ctx_with_nondefaults(self):
        """Build a RunContext with deliberately non-default values.

        ``run_duration_s`` and ``output_interval_s`` are multiples of 60
        because the GUI's QLineEdit widgets store HH:MM (minute granularity);
        sub-minute values would round-trip to zero.
        """
        from swe2d.core.builder import build_run_context
        spec = {
            "mesh": {"gpkg_path": self.fixture.gpkg, "mesh_name": self.fixture.mesh_name},
            "run_duration_s": 300.0,        # 5 min
            "output_interval_s": 60.0,      # 1 min
            "dt_cfg": 0.1,
            "n_mann": 0.030,
            "cfl": 0.50,
            "h_min": 0.005,
            "adaptive_cfl_dt": True,
            "cuda_graphs_enabled": True,
            "swe2d_perf_mode": False,
            "save_mesh_results": True,
            "save_line_results": True,
            "save_coupling_results": False,
            "save_run_log": True,
            "save_max_only": False,
            "inflow_progressive": True,
            "active_set_hysteresis": True,
        }
        return build_run_context(spec, mesh_gpkg=self.fixture.gpkg)

    def test_round_trip_scalars_match(self):
        """ctx → dict → flat → ctx' preserves every scalar field."""
        from swe2d.core.builder import (
            widget_state_to_widget_state_dict,
            widget_state_to_flat_params,
            build_run_context,
        )
        original = self._build_ctx_with_nondefaults()

        widget_state = widget_state_to_widget_state_dict(original)
        flat = widget_state_to_flat_params(widget_state)
        rebuilt = build_run_context(
            {"mesh": {"mesh_name": self.fixture.mesh_name}, "params": flat},
            mesh_gpkg=self.fixture.gpkg,
        )

        for field in SCALAR_FIELDS:
            self.assertEqual(
                getattr(original, field), getattr(rebuilt, field),
                f"round-trip changed {field}: "
                f"{getattr(original, field)!r} → {getattr(rebuilt, field)!r}",
            )

    def test_widget_state_dict_has_version_1(self):
        """The canonical serializer emits version 1 (matches GUI format)."""
        from swe2d.core.builder import widget_state_to_widget_state_dict
        ctx = self._build_ctx_with_nondefaults()
        ws = widget_state_to_widget_state_dict(ctx)
        self.assertEqual(ws.get("version"), 1)
        self.assertIn("widgets", ws)
        self.assertIsInstance(ws["widgets"], dict)

    def test_widget_state_dict_includes_run_duration(self):
        """run_duration_s is captured as the run_time_edit line-edit widget."""
        from swe2d.core.builder import widget_state_to_widget_state_dict
        ctx = self._build_ctx_with_nondefaults()
        ws = widget_state_to_widget_state_dict(ctx)
        # run_time_edit stores HH:MM (line-edit text); flat inverse parses it
        self.assertIn("run_time_edit", ws["widgets"])
        self.assertEqual(ws["widgets"]["run_time_edit"]["type"], "QLineEdit")

    def test_widget_state_dict_includes_storage_checkboxes(self):
        """save_mesh_results etc. are emitted as QCheckBox widgets."""
        from swe2d.core.builder import widget_state_to_widget_state_dict
        ctx = self._build_ctx_with_nondefaults()
        ws = widget_state_to_widget_state_dict(ctx)
        for wname in ("save_mesh_chk", "save_line_chk", "save_coupling_chk",
                      "save_log_chk", "save_max_only_chk"):
            self.assertIn(wname, ws["widgets"], f"missing {wname}")
            self.assertEqual(ws["widgets"][wname]["type"], "QCheckBox")

    def test_widget_state_dict_includes_spin_boxes(self):
        """Scalar float fields are emitted as QDoubleSpinBox widgets."""
        from swe2d.core.builder import widget_state_to_widget_state_dict
        ctx = self._build_ctx_with_nondefaults()
        ws = widget_state_to_widget_state_dict(ctx)
        for wname in ("n_mann_spin", "cfl_spin", "h_min_spin", "dt_spin"):
            self.assertIn(wname, ws["widgets"], f"missing {wname}")
            self.assertEqual(ws["widgets"][wname]["type"], "QDoubleSpinBox")

    def test_widget_state_dict_includes_checkboxes(self):
        """Boolean fields are emitted as QCheckBox widgets.

        Note: ``cuda_graphs_enabled`` maps to ``enable_cuda_graphs_chk`` in
        WIDGET_TO_RC, so the widget name differs from the RunContext field.
        """
        from swe2d.core.builder import widget_state_to_widget_state_dict
        ctx = self._build_ctx_with_nondefaults()
        ws = widget_state_to_widget_state_dict(ctx)
        for wname in ("adaptive_cfl_dt_chk", "enable_cuda_graphs_chk",
                      "swe2d_perf_mode_chk", "inflow_progressive_chk"):
            self.assertIn(wname, ws["widgets"], f"missing {wname}")
            self.assertEqual(ws["widgets"][wname]["type"], "QCheckBox")

    def test_round_trip_via_replay_payload_matches(self):
        """GUI save path: ctx → widget_state_dict → flat → build → ctx'
        yields the same scalar fields.  Mirrors the auto-save in
        run_controller.on_simulation_worker_finished."""
        from swe2d.core.builder import (
            widget_state_to_widget_state_dict,
            widget_state_to_flat_params,
            build_run_context,
        )
        original = self._build_ctx_with_nondefaults()

        # Simulate the auto-save: collect widget_state from ctx, build flat,
        # then build a fresh ctx (as replay would).
        widget_state = widget_state_to_widget_state_dict(original)
        flat = widget_state_to_flat_params(widget_state)

        # The auto-save adds run_duration_s from result (here we already have it).
        if original.run_duration_s:
            flat["run_duration_s"] = original.run_duration_s

        replay_ctx = build_run_context(
            {"mesh": {"mesh_name": self.fixture.mesh_name}, "params": flat},
            mesh_gpkg=self.fixture.gpkg,
        )

        for field in SCALAR_FIELDS:
            self.assertEqual(getattr(original, field), getattr(replay_ctx, field))


if __name__ == "__main__":
    unittest.main()
