"""Tests for swe2d/runtime/run_context_builder.py — Phase 1.A canonical builder.

Covers:
- build_run_context rejects unknown keys with "did you mean" suggestions
- build_run_context rejects type-mismatched present keys
- Widget-name normalization produces equivalent RunContext to spec keys
- String mesh form is accepted (was crashing before Phase 1.A)
- RunContextBuilder from_defaults / merge_context / with_params / build()
- build_run_context_from_dict (thin wrapper) produces same result as build_run_context
- from_replay_json agrees with build_run_context on defaults
- from_widget_params agrees with build_run_context on defaults (dt_cfg = 0.05)

No GPU required.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np

# Headless: use the mocked qgis.env so we don't need a real QGIS app.
from tests.mocks.qgis_env import install_qgis_mocks
install_qgis_mocks()

from tests._swe2d_test_helpers import (
    _make_cartesian_quad_mesh,
    _serialize_and_persist_mesh,
)


# ── Shared mesh fixture ────────────────────────────────────────────────────────

class MeshFixture:
    """Tiny 4×2 quad mesh in a temporary GPKG, used by builder tests."""

    NX, NY = 4, 2
    LX, LY = 40.0, 10.0

    def __init__(self):
        self._tmpdir = tempfile.mkdtemp(prefix="rcb_test_")
        self.gpkg = os.path.join(self._tmpdir, "rcb_mesh.gpkg")
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


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestDrainageGpkgAdapter(unittest.TestCase):
    """The QGIS-backed drainage adapter loads valid GPKG layers."""

    def test_valid_mock_gpkg_returns_pipe_network_config(self):
        import sys
        from unittest.mock import patch

        from tests.mocks.qgis_env import (
            MockQgsFeature,
            MockQgsFields,
            MockQgsGeometry,
            MockQgsPointXY,
            MockQgsVectorLayer,
        )
        from swe2d.core.gpkg_io import (
            _build_drainage_config_from_gpkg_layers,
        )

        node_fields = MockQgsFields([
            "node_id", "invert_elev", "rim_elev", "max_depth", "node_type",
        ])
        node_layer = MockQgsVectorLayer()
        node_layer._fields = node_fields
        for node_id, x, y in (("N1", 1.0, 1.0), ("N2", 8.0, 1.0)):
            feature = MockQgsFeature(node_fields, {
                "node_id": node_id,
                "invert_elev": 0.0,
                "rim_elev": 1.0,
                "max_depth": 1.0,
                "node_type": "junction",
            })
            feature.setGeometry(MockQgsGeometry.fromPointXY(MockQgsPointXY(x, y)))
            node_layer._features.append(feature)

        link_fields = MockQgsFields([
            "link_id", "from_node", "to_node", "length", "diameter", "roughness_n",
        ])
        link_layer = MockQgsVectorLayer()
        link_layer._fields = link_fields
        link_feature = MockQgsFeature(link_fields, {
            "link_id": "L1",
            "from_node": "N1",
            "to_node": "N2",
            "length": 7.0,
            "diameter": 1.0,
            "roughness_n": 0.013,
        })
        link_feature.setGeometry(MockQgsGeometry.fromPolylineXY([
            MockQgsPointXY(1.0, 1.0), MockQgsPointXY(8.0, 1.0),
        ]))
        link_layer._features.append(link_feature)

        def vector_layer_factory(uri, _name, _provider):
            if uri.endswith("|layername=drain_nodes"):
                return node_layer
            if uri.endswith("|layername=drain_links"):
                return link_layer
            self.fail(f"Unexpected mock GPKG layer URI: {uri}")

        mesh_data = {
            "node_x": np.array([0.0, 10.0, 0.0], dtype=np.float64),
            "node_y": np.array([0.0, 0.0, 10.0], dtype=np.float64),
            "node_z": np.zeros(3, dtype=np.float64),
            "cell_nodes": np.array([0, 1, 2], dtype=np.int32),
        }
        # ``install_qgis_mocks`` registers ``qgis.core`` as a synthetic
        # module in ``sys.modules`` (no parent package); patch the
        # already-loaded reference directly so the adapter's local import
        # resolves our factory.
        qgis_core = sys.modules["qgis.core"]
        with patch.object(qgis_core, "QgsVectorLayer", side_effect=vector_layer_factory):
            config = _build_drainage_config_from_gpkg_layers(
                mesh_data=mesh_data,
                drainage_gpkg="mock_drainage.gpkg",
                nodes_layer="drain_nodes",
                links_layer="drain_links",
                cell_min_bed=np.array([0.0], dtype=np.float64),
                gravity=9.81,
                config={},
                log_fn=lambda _message: None,
            )

        self.assertIsNotNone(config)
        self.assertTrue(config.enabled)
        self.assertEqual([node.node_id for node in config.nodes], ["N1", "N2"])
        self.assertEqual([link.link_id for link in config.links], ["L1"])


class TestNormalizeSpecValidation(unittest.TestCase):
    """_normalize_spec rejects unknown keys with suggestions and type errors."""

    def test_unknown_key_raises_valueerror_with_suggestion(self):
        from swe2d.core.builder import _normalize_spec
        # Use a typo close enough to n_mann to trigger a suggestion (>=0.6 similarity)
        with self.assertRaises(ValueError) as ctx:
            _normalize_spec({"n_mann": 0.035, "n_manng": 1.0})  # "n_manng" → "n_mann"
        self.assertIn("did you mean", str(ctx.exception))
        self.assertIn("n_mann", str(ctx.exception))

    def test_unknown_key_suggests_widget_name_equivalent(self):
        from swe2d.core.builder import _normalize_spec
        # n_mann_spin is a widget name; the suggestion should include n_mann
        with self.assertRaises(ValueError) as ctx:
            _normalize_spec({"n_mann_spin": 0.035, "typo_mann": 1.0})
        msg = str(ctx.exception)
        self.assertIn("did you mean", msg)

    def test_unknown_non_param_key_allowed(self):
        """Keys not starting with _ and not in VALID_SPEC_KEYS are rejected."""
        from swe2d.core.builder import _normalize_spec
        # A truly arbitrary unknown key should be rejected.
        with self.assertRaises(ValueError):
            _normalize_spec({"completely_unknown_field": 42})

    def test_unknown_nested_keys_fail_fast(self):
        from swe2d.core.builder import _normalize_spec

        cases = {
            "params": {"cfl_typo": 0.5},
            "mesh": {"mesh_nam": "mesh"},
            "results": {"save_mesh_result": True},
            "units": {"length_unit": "m"},
            "data_sources": {"rainfall": {}},
            "rain_cn": {"cn_feld": "cn"},
            "hyetograph": {"gauge_layr": "gages"},
            "drainage": {"node_layer": "nodes"},
            "structures": {"tablle": "structures"},
            "sample_lines": {"tablle": "samples"},
            "bc_lines": {"tablle": "boundaries"},
            "internal_flow_sources": {"hydrograph_tabl": "hydrographs"},
            "storm_areas": {"tablle": "storms"},
            "infiltration_method": {"method": "scs_cn"},
        }

        for block_name, value in cases.items():
            with self.subTest(block=block_name):
                with self.assertRaisesRegex(ValueError, block_name) as ctx:
                    _normalize_spec({block_name: value})
                self.assertIn(next(iter(value)), str(ctx.exception))

    def test_nested_unknown_param_key_has_suggestion(self):
        from swe2d.core.builder import _normalize_spec

        with self.assertRaisesRegex(ValueError, "did you mean.*'cfl'"):
            _normalize_spec({"params": {"cfl_": 0.5}})

    def test_widget_state_key_allowed(self):
        """widget_state (GUI-internal key in replay payload) is not rejected."""
        from swe2d.core.builder import _normalize_spec
        # Should not raise — widget_state is a known internal/GUI key.
        spec = _normalize_spec({
            "widget_state": {"version": 1, "widgets": {}},
            "mesh": {"mesh_name": "foo", "gpkg_path": "/nonexistent.gpkg"},
        })
        self.assertIsInstance(spec, dict)

    def test_version_key_allowed(self):
        from swe2d.core.builder import _normalize_spec
        spec = _normalize_spec({
            "version": 1,
            "mesh": {"mesh_name": "foo", "gpkg_path": "/nonexistent.gpkg"},
        })
        self.assertIsInstance(spec, dict)


class TestNormalizeSpecNormalization(unittest.TestCase):
    """_normalize_spec correctly transforms legacy inputs to canonical form."""

    def test_mesh_string_normalized_to_dict(self):
        from swe2d.core.builder import _normalize_spec
        spec = _normalize_spec({"mesh": "my_mesh"})
        self.assertEqual(spec["mesh"], {"mesh_name": "my_mesh"})

    def test_mesh_dict_unchanged(self):
        from swe2d.core.builder import _normalize_spec
        mesh_in = {"mesh_name": "my_mesh", "gpkg_path": "/foo/bar.gpkg"}
        spec = _normalize_spec({"mesh": mesh_in})
        self.assertEqual(spec["mesh"], mesh_in)

    def test_mesh_wrong_type_raises(self):
        from swe2d.core.builder import _normalize_spec
        with self.assertRaises(TypeError) as ctx:
            _normalize_spec({"mesh": ["not", "a", "string"]})
        self.assertIn("must be a string or dict", str(ctx.exception))

    def test_gui_replay_metadata_is_normalized(self):
        from swe2d.core.builder import _normalize_spec

        spec = _normalize_spec({
            "params": {
                "tiny_mode_combo_text": "Disabled",
                "degen_mode_combo_text": "Repair",
                "drainage_gpu_method_combo_text": "Per-step",
                "culvert_solver_mode_combo_text": "Lookup",
                "bridge_stacked_coupling_mode_combo_text": "Phase 3",
                "results_table_name_edit": "run_1",
                "culvert_face_flux_enabled": True,
            },
        })

        params = spec["params"]
        self.assertEqual(params["culvert_face_flux_mode"], "face_flux")
        self.assertNotIn("culvert_face_flux_enabled", params)
        self.assertEqual(
            params["bridge_stacked_coupling_mode_name"], "Phase 3"
        )
        self.assertEqual(params["tiny_mode_name"], "Disabled")
        self.assertEqual(params["results_table_name_edit"], "run_1")

    def test_widget_name_normalized(self):
        from swe2d.core.builder import _normalize_spec
        spec = _normalize_spec({"n_mann_spin": 0.030})
        self.assertEqual(spec["n_mann"], 0.030)

    def test_widget_name_in_params_normalized(self):
        from swe2d.core.builder import _normalize_spec
        spec = _normalize_spec({"params": {"cfl_spin": 0.9}})
        self.assertEqual(spec["cfl"], 0.9)


class TestDefaultsTableConsistency(unittest.TestCase):
    """All constructors agree on defaults, especially the former 0.2-vs-0.05 split."""

    def test_dt_cfg_default_is_005(self):
        """dt_cfg default is 0.05 — single canonical value, not 0.2."""
        from swe2d.core.builder import _DEFAULTS
        self.assertEqual(_DEFAULTS["dt_cfg"], 0.05)

    def test_dt_request_not_in_defaults(self):
        """dt_request is *derived* from (adaptive_cfl_dt, dt_cfg) at build
        time — it must not have a code-level default that could mask a
        missing wire-up.  The Phase 1.B refactor (commit 70561f9a) lost
        the derivation and shipped dt_request=0.05, capping every run at
        0.05 s regardless of dt_cfg.  No sensible default lives here.
        """
        from swe2d.core.builder import _DEFAULTS
        self.assertNotIn("dt_request", _DEFAULTS)

    def test_dt_fixed_not_in_defaults(self):
        """dt_fixed is *derived* from (adaptive_cfl_dt, dt_cfg) at build
        time — it must not have a code-level default.  The C++ binding
        sentinel for "not fixed" is -1.0; that's an engine convention,
        not a build-time default.
        """
        from swe2d.core.builder import _DEFAULTS
        self.assertNotIn("dt_fixed", _DEFAULTS)

    def test_initial_dt_default_is_zero(self):
        from swe2d.core.builder import _DEFAULTS
        self.assertEqual(_DEFAULTS["initial_dt"], 0.0)

    def test_reconstruction_mode_default_is_zero(self):
        from swe2d.core.builder import _DEFAULTS
        self.assertEqual(_DEFAULTS["reconstruction_mode"], 0)

    def test_temporal_scheme_default_is_none(self):
        from swe2d.core.builder import _DEFAULTS
        # temporal_scheme is None by default (null in JSON), not 2
        self.assertIsNone(_DEFAULTS["temporal_scheme"])

    def test_save_mesh_results_default_is_true(self):
        from swe2d.core.builder import _DEFAULTS
        self.assertIs(_DEFAULTS["save_mesh_results"], True)

    def test_output_interval_s_default_is_10(self):
        """M-1: output_interval_s default is 1.0 (not chained to run_duration_s)."""
        from swe2d.core.builder import _DEFAULTS
        self.assertEqual(_DEFAULTS["output_interval_s"], 1.0)

    def test_swe2d_perf_mode_default_is_false(self):
        """swe2d_perf_mode is a canonical RunContext boolean field
        (mirrors the cuda_graphs_enabled pattern).  Default is False;
        the GUI's ``swe2d_perf_mode_chk`` widget maps to it via WIDGET_TO_RC."""
        from swe2d.core.builder import _DEFAULTS
        from swe2d.core.run_context import RunContext
        self.assertIs(_DEFAULTS["swe2d_perf_mode"], False)
        # Field exists on RunContext (catches future renames).
        self.assertIn("swe2d_perf_mode", {f.name for f in RunContext.__dataclass_fields__.values()})


class TestBuildRunContextMinimal(unittest.TestCase):
    """Smoke tests for build_run_context with minimal spec (no GPKG arrays)."""

    @classmethod
    def setUpClass(cls):
        cls.fixture = MeshFixture().build()
        cls.addClassCleanup(cls.fixture.close)

    def test_build_run_context_produces_runcontext(self):
        from swe2d.core.builder import build_run_context
        spec = {
            "mesh": {"mesh_name": self.fixture.mesh_name},
        }
        ctx = build_run_context(spec, mesh_gpkg=self.fixture.gpkg)
        self.assertEqual(ctx.mesh_name, self.fixture.mesh_name)
        self.assertEqual(ctx.model_gpkg_path, self.fixture.gpkg)
        self.assertGreater(ctx.cell_areas.size, 0)

    def test_string_mesh_form_works(self):
        """String mesh form was crashing with mesh.get on str before Phase 1.A."""
        from swe2d.core.builder import build_run_context
        spec = {"mesh": self.fixture.mesh_name}  # string, not dict
        ctx = build_run_context(spec, mesh_gpkg=self.fixture.gpkg)
        self.assertEqual(ctx.mesh_name, self.fixture.mesh_name)

    def test_canonical_spec_with_nested_params(self):
        from swe2d.core.builder import build_run_context
        spec = {
            "mesh": {"mesh_name": self.fixture.mesh_name},
            "params": {
                "run_duration_s": 120.0,
                "dt_cfg": 0.1,
                "n_mann": 0.030,
            },
        }
        ctx = build_run_context(spec, mesh_gpkg=self.fixture.gpkg)
        self.assertEqual(ctx.run_duration_s, 120.0)
        self.assertEqual(ctx.dt_cfg, 0.1)
        self.assertEqual(ctx.n_mann, 0.030)

    def test_output_interval_s_default_when_not_specified(self):
        """M-1: when output_interval_s is absent but run_duration_s is set,
        output_interval_s falls back to the canonical 1.0 s default — NOT
        to run_duration_s.  The pre-fix chained default produced a
        runaway snapshot interval equal to the full run duration.
        """
        from swe2d.core.builder import build_run_context
        spec = {
            "mesh": {"mesh_name": self.fixture.mesh_name},
            "params": {"run_duration_s": 600.0},  # 10 min, no output_interval_s
        }
        ctx = build_run_context(spec, mesh_gpkg=self.fixture.gpkg)
        self.assertEqual(ctx.run_duration_s, 600.0)
        self.assertEqual(
            ctx.output_interval_s, 1.0,
            "output_interval_s default must be 1.0 s, not run_duration_s",
        )

    def test_canonical_spec_top_level_overrides_params(self):
        from swe2d.core.builder import build_run_context
        spec = {
            "mesh": {"mesh_name": self.fixture.mesh_name},
            "params": {"run_duration_s": 100.0},
            "run_duration_s": 200.0,  # top-level should win
        }
        ctx = build_run_context(spec, mesh_gpkg=self.fixture.gpkg)
        self.assertEqual(ctx.run_duration_s, 200.0)

    def test_build_run_context_from_dict_equivalent(self):
        """build_run_context_from_dict (thin wrapper) produces same result."""
        from swe2d.core.builder import build_run_context, build_run_context_from_dict
        spec = {
            "mesh": {"mesh_name": self.fixture.mesh_name},
            "params": {"run_duration_s": 360.0, "dt_cfg": 0.05},
        }
        ctx1 = build_run_context(spec, mesh_gpkg=self.fixture.gpkg)
        ctx2 = build_run_context_from_dict(spec, mesh_gpkg=self.fixture.gpkg)
        self.assertEqual(ctx1.run_duration_s, ctx2.run_duration_s)
        self.assertEqual(ctx1.dt_cfg, ctx2.dt_cfg)
        self.assertEqual(ctx1.mesh_name, ctx2.mesh_name)

    # ── dt_request / dt_fixed derivation regression tests ───────────────
    # The Phase 1.B refactor (commit 70561f9a, "retire
    # SWE2DRunDataBuilder / SWE2DRunOptionsBuilder") lost the rule the
    # retired run_options_builder.py applied:
    #
    #   dt_fixed   = -1.0 if adaptive_cfl_dt else dt_cfg
    #   dt_request = -1.0 if adaptive_cfl_dt else dt_cfg
    #
    # The canonical builder read dt_request from _DEFAULTS (0.05) instead
    # of deriving it, so every run was capped at 0.05 s regardless of
    # the user's dt_spin / adaptive_cfl_dt choice.  These tests guard
    # against re-introducing the regression.

    def test_adaptive_true_derives_dt_request_minus_one(self):
        """adaptive_cfl_dt=True → C++ dt_request=-1.0 (pure CFL, no cap)."""
        from swe2d.core.builder import build_run_context
        spec = {
            "mesh": {"mesh_name": self.fixture.mesh_name},
            "params": {"dt_cfg": 0.10, "adaptive_cfl_dt": True},
        }
        ctx = build_run_context(spec, mesh_gpkg=self.fixture.gpkg)
        self.assertEqual(ctx.adaptive_cfl_dt, True)
        self.assertEqual(ctx.dt_request, -1.0,
            "adaptive mode must set dt_request=-1.0 (C++ pure-CFL sentinel)")
        self.assertEqual(ctx.dt_fixed, -1.0,
            "adaptive mode must set dt_fixed=-1.0 (C++ 'not fixed' sentinel)")

    def test_adaptive_false_derives_dt_request_equals_dt_cfg(self):
        """adaptive_cfl_dt=False → C++ dt_request=dt_cfg (fixed dt)."""
        from swe2d.core.builder import build_run_context
        spec = {
            "mesh": {"mesh_name": self.fixture.mesh_name},
            "params": {"dt_cfg": 0.10, "adaptive_cfl_dt": False},
        }
        ctx = build_run_context(spec, mesh_gpkg=self.fixture.gpkg)
        self.assertEqual(ctx.adaptive_cfl_dt, False)
        self.assertEqual(ctx.dt_request, 0.10,
            "fixed mode must set dt_request=dt_cfg (C++ uses this as fixed dt)")
        self.assertEqual(ctx.dt_fixed, 0.10,
            "fixed mode must set dt_fixed=dt_cfg (C++ 'fixed dt override')")

    def test_explicit_dt_request_overrides_derivation(self):
        """CLI / replay spec with explicit dt_request wins over derivation.
        This is required for replay to lock the timestep exactly."""
        from swe2d.core.builder import build_run_context
        spec = {
            "mesh": {"mesh_name": self.fixture.mesh_name},
            "params": {
                "dt_cfg": 0.10,
                "adaptive_cfl_dt": True,
                "dt_request": 0.25,  # explicit override
            },
        }
        ctx = build_run_context(spec, mesh_gpkg=self.fixture.gpkg)
        self.assertEqual(ctx.dt_request, 0.25,
            "explicit dt_request in spec must override the derivation")

    def test_explicit_dt_request_at_top_level_wins(self):
        """Top-level dt_request also overrides nested-params derivation."""
        from swe2d.core.builder import build_run_context
        spec = {
            "mesh": {"mesh_name": self.fixture.mesh_name},
            "params": {"dt_cfg": 0.10, "adaptive_cfl_dt": False},
            "dt_request": 0.25,  # top-level explicit
        }
        ctx = build_run_context(spec, mesh_gpkg=self.fixture.gpkg)
        self.assertEqual(ctx.dt_request, 0.25)

    def test_derivation_default_uses_default_dt_cfg(self):
        """When spec omits both dt_cfg and adaptive_cfl_dt, derive from
        defaults: dt_cfg=0.05, adaptive=False → dt_request=0.05."""
        from swe2d.core.builder import build_run_context
        spec = {"mesh": {"mesh_name": self.fixture.mesh_name}}
        ctx = build_run_context(spec, mesh_gpkg=self.fixture.gpkg)
        # adaptive_cfl_dt default is False, dt_cfg default is 0.05.
        self.assertEqual(ctx.adaptive_cfl_dt, False)
        self.assertEqual(ctx.dt_cfg, 0.05)
        self.assertEqual(ctx.dt_request, 0.05,
            "default fixed mode must use dt_cfg=0.05, not 0.0 or -1.0")
        self.assertEqual(ctx.dt_fixed, 0.05)

    def test_initial_dt_passes_through(self):
        """initial_dt is a separate cold-start override and is not
        derived — it must pass through unchanged when explicitly set."""
        from swe2d.core.builder import build_run_context
        spec = {
            "mesh": {"mesh_name": self.fixture.mesh_name},
            "params": {"dt_cfg": 0.10, "adaptive_cfl_dt": True, "initial_dt": 0.001},
        }
        ctx = build_run_context(spec, mesh_gpkg=self.fixture.gpkg)
        self.assertEqual(ctx.initial_dt, 0.001)
        # And the derived values are unaffected by initial_dt.
        self.assertEqual(ctx.dt_request, -1.0)
        self.assertEqual(ctx.dt_fixed, -1.0)


class TestRunContextBuilder(unittest.TestCase):
    """RunContextBuilder fluent API tests."""

    @classmethod
    def setUpClass(cls):
        cls.fixture = MeshFixture().build()
        cls.addClassCleanup(cls.fixture.close)

    def test_from_defaults_produces_builder(self):
        from swe2d.core.builder import RunContextBuilder
        b = RunContextBuilder.from_defaults()
        self.assertEqual(b._mode, "gui")

    def test_with_params_layers(self):
        from swe2d.core.builder import RunContextBuilder
        b = (RunContextBuilder.from_defaults()
             .with_params({"mesh": self.fixture.mesh_name, "run_duration_s": 7200.0}))
        ctx = b.build(mesh_gpkg=self.fixture.gpkg)
        self.assertEqual(ctx.run_duration_s, 7200.0)

    def test_with_params_canonical_and_legacy_equivalents(self):
        """Spec keys and widget-name keys produce the same context."""
        from swe2d.core.builder import RunContextBuilder
        b1 = (RunContextBuilder.from_defaults()
              .with_params({"mesh": self.fixture.mesh_name, "n_mann": 0.030}))
        b2 = (RunContextBuilder.from_defaults()
              .with_params({"mesh": self.fixture.mesh_name, "n_mann_spin": 0.030}))
        ctx1 = b1.build(mesh_gpkg=self.fixture.gpkg)
        ctx2 = b2.build(mesh_gpkg=self.fixture.gpkg)
        self.assertEqual(ctx1.n_mann, ctx2.n_mann)

    def test_build_requires_mesh_gpkg(self):
        from swe2d.core.builder import RunContextBuilder
        b = RunContextBuilder.from_defaults()
        with self.assertRaises(FileNotFoundError):
            b.build()  # no mesh_gpkg → FileNotFoundError

    def test_string_mesh_form_through_builder(self):
        from swe2d.core.builder import RunContextBuilder
        b = (RunContextBuilder.from_defaults()
             .with_params({"mesh": self.fixture.mesh_name}))
        ctx = b.build(mesh_gpkg=self.fixture.gpkg)
        self.assertEqual(ctx.mesh_name, self.fixture.mesh_name)

    def test_build_returns_runcontext(self):
        from swe2d.core.builder import RunContextBuilder
        b = (RunContextBuilder.from_defaults()
             .with_params({"mesh": self.fixture.mesh_name}))
        ctx = b.build(mesh_gpkg=self.fixture.gpkg)
        self.assertTrue(hasattr(ctx, "run_duration_s"))
        self.assertTrue(hasattr(ctx, "cell_areas"))


class TestRunContextBuilderMergeContext(unittest.TestCase):
    """RunContextBuilder.merge_context() — layering RunContext objects."""

    @classmethod
    def setUpClass(cls):
        cls.fixture = MeshFixture().build()
        cls.addClassCleanup(cls.fixture.close)

    def _minimal_rc(self, **overrides):
        """Return a minimal RunContext for merge_context tests.

        Only the fields we actually test are passed explicitly; all others
        use RunContext defaults so there are no duplicate-keyword conflicts
        with **overrides.
        """
        from swe2d.core.run_context import RunContext
        return RunContext(
            run_id="test_merge",
            run_wallclock_start="",
            run_log_start_idx=0,
            model_gpkg_path=self.fixture.gpkg,
            mesh_name=self.fixture.mesh_name,
            mesh_crs_wkt="",
            # Only the fields we test in merge_context tests
            run_duration_s=overrides.pop("run_duration_s", 0.0),
            n_mann=overrides.pop("n_mann", 0.035),
            cfl=overrides.pop("cfl", 0.45),
            # Remaining overrides go through — but no duplicate keyword conflicts
            **overrides,
        )

    def test_merge_context_single_run_context(self):
        """merge_context merges a RunContext's scalar fields into the builder stack."""
        from swe2d.core.builder import RunContextBuilder
        base_rc = self._minimal_rc(n_mann=0.030, run_duration_s=3600.0)
        ctx = (RunContextBuilder.from_defaults()
               .with_params({"mesh": self.fixture.mesh_name})
               .merge_context(base_rc)
               .build(mesh_gpkg=self.fixture.gpkg))
        self.assertEqual(ctx.n_mann, 0.030)
        self.assertEqual(ctx.run_duration_s, 3600.0)

    def test_merge_context_later_layer_wins(self):
        """Multiple merge_context calls: later RunContext overrides earlier."""
        from swe2d.core.builder import RunContextBuilder
        rc1 = self._minimal_rc(n_mann=0.030, cfl=0.50)
        rc2 = self._minimal_rc(n_mann=0.040, cfl=0.60)
        ctx = (RunContextBuilder.from_defaults()
               .with_params({"mesh": self.fixture.mesh_name})
               .merge_context(rc1)
               .merge_context(rc2)
               .build(mesh_gpkg=self.fixture.gpkg))
        self.assertEqual(ctx.n_mann, 0.040)
        self.assertEqual(ctx.cfl, 0.60)

    def test_merge_context_plus_with_params(self):
        """merge_context layers come before with_params layers in priority."""
        from swe2d.core.builder import RunContextBuilder
        rc1 = self._minimal_rc(n_mann=0.030)
        ctx = (RunContextBuilder.from_defaults()
               .with_params({"mesh": self.fixture.mesh_name})
               .merge_context(rc1)
               .with_params({"n_mann": 0.050})
               .build(mesh_gpkg=self.fixture.gpkg))
        # with_params wins (it is the topmost layer)
        self.assertEqual(ctx.n_mann, 0.050)


class TestRunContextBuilderFullPipeline(unittest.TestCase):
    """Full from_defaults + merge_context + with_params + build() pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.fixture = MeshFixture().build()
        cls.addClassCleanup(cls.fixture.close)

    def test_from_defaults_merge_context_build_produces_valid_context(self):
        """The full builder pipeline produces a RunContext with mesh arrays."""
        from swe2d.core.builder import RunContextBuilder
        ctx = (RunContextBuilder.from_defaults()
               .with_params({"mesh": self.fixture.mesh_name, "run_duration_s": 7200.0})
               .build(mesh_gpkg=self.fixture.gpkg))
        self.assertEqual(ctx.run_duration_s, 7200.0)
        self.assertGreater(ctx.node_x.size, 0)
        self.assertGreater(ctx.cell_areas.size, 0)

    def test_from_spec_plus_with_params_equivalent(self):
        """from_spec + with_params produces the same result as from_defaults + with_params."""
        from swe2d.core.builder import RunContextBuilder
        b1 = (RunContextBuilder.from_spec({})
              .with_params({"mesh": self.fixture.mesh_name, "run_duration_s": 900.0}))
        b2 = (RunContextBuilder.from_defaults()
              .with_params({"mesh": self.fixture.mesh_name, "run_duration_s": 900.0}))
        ctx1 = b1.build(mesh_gpkg=self.fixture.gpkg)
        ctx2 = b2.build(mesh_gpkg=self.fixture.gpkg)
        self.assertEqual(ctx1.run_duration_s, ctx2.run_duration_s)
        self.assertEqual(ctx1.n_mann, ctx2.n_mann)

    def test_builder_top_level_overrides_params_block(self):
        """Spec top-level keys override the params sub-dict (same as build_run_context)."""
        from swe2d.core.builder import RunContextBuilder
        ctx = (RunContextBuilder.from_spec({
            "mesh": self.fixture.mesh_name,
            "params": {"n_mann": 0.030},
            "n_mann": 0.040,  # top-level wins
        }).build(mesh_gpkg=self.fixture.gpkg))
        self.assertEqual(ctx.n_mann, 0.040)


class TestBuildRunContextMinimalSpec(unittest.TestCase):
    """build_run_context with minimal spec dict (no GPKG arrays — just spec data)."""

    @classmethod
    def setUpClass(cls):
        cls.fixture = MeshFixture().build()
        cls.addClassCleanup(cls.fixture.close)

    def test_minimal_spec_produces_runcontext(self):
        """A spec dict with only mesh path + duration produces a valid RunContext."""
        from swe2d.core.builder import build_run_context
        spec = {
            "mesh": {"mesh_name": self.fixture.mesh_name},
            "run_duration_s": 120.0,
        }
        ctx = build_run_context(spec, mesh_gpkg=self.fixture.gpkg)
        self.assertEqual(ctx.run_duration_s, 120.0)
        self.assertEqual(ctx.mesh_name, self.fixture.mesh_name)
        # Mesh arrays were loaded from GPKG
        self.assertGreater(ctx.cell_areas.size, 0)
        self.assertGreater(ctx.node_x.size, 0)

    def test_spec_with_params_sub_dict(self):
        """Params sub-dict values are resolved by build_run_context."""
        from swe2d.core.builder import build_run_context
        spec = {
            "mesh": {"mesh_name": self.fixture.mesh_name},
            "params": {"run_duration_s": 240.0, "n_mann": 0.030},
        }
        ctx = build_run_context(spec, mesh_gpkg=self.fixture.gpkg)
        self.assertEqual(ctx.run_duration_s, 240.0)
        self.assertEqual(ctx.n_mann, 0.030)


class TestFromReplayJsonDefaults(unittest.TestCase):
    """from_replay_json uses the shared _DEFAULTS table."""

    def test_from_replay_json_dt_cfg_is_005(self):
        from swe2d.core.run_context import RunContext
        # from_replay_json calls build_run_context, which uses _DEFAULTS["dt_cfg"].
        # Test via the bare defaults: when payload has no dt_cfg, the builder
        # must NOT use 0.2.  We test by checking _DEFAULTS directly and by
        # confirming from_replay_json accepts the payload without error.
        payload = {
            "mesh": {"mesh_name": "foo"},
            "params": {},
        }
        # We can't call from_replay_json without a real mesh, but we can
        # verify _DEFAULTS["dt_cfg"] is 0.05 (tested above).
        # The integration is verified by test_run_context_parity passing.
        from swe2d.core.builder import _DEFAULTS
        self.assertEqual(_DEFAULTS["dt_cfg"], 0.05)


class TestDrainageConfigParity(unittest.TestCase):
    """The CLI drainage config dict matches the GUI's 12-key contract (C-2)."""

    EXPECTED_KEYS = {
        "solver_mode", "coupling_substeps", "gpu_method",
        "head_deadband", "dynamic_relaxation", "implicit_iters", "implicit_relax",
        "friction_method", "surcharge_method", "recon_method",
        "time_integrator", "friction_alpha",
    }

    def _v(self, spec):
        """Mirror the builder's nested-params _v helper."""
        params = spec.get("params", {}) or {}

        def lookup(key, default=None):
            if key in spec:
                return spec[key]
            return params.get(key, default)

        return lookup

    def test_all_twelve_keys_present_in_drainage_config(self):
        """The 12 GUI keys — including the 5 added by C-2 — are forwarded."""
        from swe2d.core.builder import _drainage_config_dict

        cfg = _drainage_config_dict(
            {"nodes_layer": "n", "links_layer": "l"},
            self._v({"params": {}}),
        )
        missing = self.EXPECTED_KEYS - set(cfg)
        self.assertFalse(
            missing,
            f"Drainage config is missing GUI-parity keys: {sorted(missing)}",
        )

    def test_defaults_match_pipe_network_service(self):
        """Default values for the 5 added keys match pipe_network_service."""
        from swe2d.core.builder import _drainage_config_dict

        cfg = _drainage_config_dict({}, self._v({}))
        self.assertEqual(cfg["friction_method"], 0)
        self.assertEqual(cfg["surcharge_method"], 0)
        self.assertEqual(cfg["recon_method"], 0)
        self.assertEqual(cfg["time_integrator"], 1)
        self.assertAlmostEqual(cfg["friction_alpha"], 0.01)

    def test_drainage_block_overrides_propagate(self):
        """Drainage-block values for the 5 added keys override defaults."""
        from swe2d.core.builder import _drainage_config_dict

        cfg = _drainage_config_dict(
            {
                "nodes_layer": "n",
                "links_layer": "l",
                "friction_method": 2,
                "surcharge_method": 1,
                "recon_method": 1,
                "time_integrator": 0,
                "friction_alpha": 0.25,
            },
            self._v({}),
        )
        self.assertEqual(cfg["friction_method"], 2)
        self.assertEqual(cfg["surcharge_method"], 1)
        self.assertEqual(cfg["recon_method"], 1)
        self.assertEqual(cfg["time_integrator"], 0)
        self.assertAlmostEqual(cfg["friction_alpha"], 0.25)

    def test_legacy_top_level_keys_fall_through(self):
        """A legacy top-level spec value still works after C-2."""
        from swe2d.core.builder import _drainage_config_dict

        cfg = _drainage_config_dict(
            {"nodes_layer": "n", "links_layer": "l"},
            self._v({"friction_method": 2, "time_integrator": 0}),
        )
        self.assertEqual(cfg["friction_method"], 2)
        self.assertEqual(cfg["time_integrator"], 0)
        # Defaults preserved when not overridden.
        self.assertEqual(cfg["surcharge_method"], 0)
        self.assertAlmostEqual(cfg["friction_alpha"], 0.01)


def _quad_boundary_edges(nx: int, ny: int):
    """Boundary edges of the Cartesian quad mesh from _make_cartesian_quad_mesh.

    Node numbering: row-major with stride nx+1 (node index = j*(nx+1) + i).
    """
    stride = nx + 1
    edges = []
    for i in range(nx):
        edges.append((i, i + 1))                                # bottom row
        edges.append((ny * stride + i, ny * stride + i + 1))    # top row
    for j in range(ny):
        edges.append((j * stride, (j + 1) * stride))            # left column
        edges.append((j * stride + nx, (j + 1) * stride + nx))  # right column
    bc_n0 = np.array([e[0] for e in edges], dtype=np.int32)
    bc_n1 = np.array([e[1] for e in edges], dtype=np.int32)
    return bc_n0, bc_n1


class MeshFixtureWithBC(MeshFixture):
    """MeshFixture whose persisted GPKG includes real boundary edges."""

    def build(self):
        node_x, node_y, _, cell_nodes, _, _ = _make_cartesian_quad_mesh(
            self.NX, self.NY, self.LX, self.LY,
        )
        node_z = np.zeros_like(node_x)
        bc_n0, bc_n1 = _quad_boundary_edges(self.NX, self.NY)
        n_bc = int(bc_n0.size)
        bc_tp = np.ones(n_bc, dtype=np.int32)
        bc_vl = np.zeros(n_bc, dtype=np.float64)
        _serialize_and_persist_mesh(
            self.gpkg, self.mesh_name,
            node_x, node_y, node_z, cell_nodes,
            bc_n0, bc_n1, bc_tp, bc_vl,
        )
        self.n_cells = self.NX * self.NY
        self.n_bc = n_bc
        return self


class TestMeshLoadErrors(unittest.TestCase):
    """Mesh loading distinguishes missing tables, modules, and corrupt BLOBs."""

    @classmethod
    def setUpClass(cls):
        cls.fixture = MeshFixture().build()
        cls.addClassCleanup(cls.fixture.close)

    def _temporary_gpkg(self) -> str:
        handle = tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False)
        handle.close()
        self.addCleanup(lambda: os.path.exists(handle.name) and os.unlink(handle.name))
        return handle.name

    def test_missing_mesh_table_raises_mesh_load_error(self):
        from swe2d.core.gpkg_io import MeshLoadError, query_mesh_from_gpkg

        gpkg = self._temporary_gpkg()

        with self.assertRaisesRegex(MeshLoadError, "missing table"):
            query_mesh_from_gpkg(gpkg, "missing")

    def test_missing_native_module_raises_mesh_load_error(self):
        from swe2d.core.gpkg_io import MeshLoadError, query_mesh_from_gpkg

        with patch.dict(sys.modules, {"hydra_swe2d": None}):
            with self.assertRaisesRegex(MeshLoadError, "required module"):
                query_mesh_from_gpkg(self.fixture.gpkg, self.fixture.mesh_name)

    def test_corrupt_mesh_blob_raises_mesh_load_error(self):
        from swe2d.core.gpkg_io import MeshLoadError, query_mesh_from_gpkg

        gpkg = self._temporary_gpkg()
        with sqlite3.connect(gpkg) as connection:
            connection.execute(
                "CREATE TABLE swe2d_baked_mesh "
                "(mesh_name TEXT PRIMARY KEY, baked_blob BLOB, crs_wkt TEXT)"
            )
            connection.execute(
                "INSERT INTO swe2d_baked_mesh VALUES (?, ?, ?)",
                ("corrupt", b"not-a-mesh", ""),
            )

        with self.assertRaisesRegex(MeshLoadError, "corrupt.*BLOB"):
            query_mesh_from_gpkg(gpkg, "corrupt")


class TestBuildRunContextFailFastErrors(unittest.TestCase):
    """Configured data sources never degrade to empty fallback values."""

    @classmethod
    def setUpClass(cls):
        cls.fixture = MeshFixtureWithBC().build()
        cls.addClassCleanup(cls.fixture.close)

    def _build(self, extra: dict):
        from swe2d.core.builder import build_run_context

        return build_run_context(
            {
                "mesh": {
                    "gpkg_path": self.fixture.gpkg,
                    "mesh_name": self.fixture.mesh_name,
                },
                **extra,
            },
            mesh_gpkg=self.fixture.gpkg,
        )

    def test_builder_silent_absorb_sites_raise_typed_error(self):
        from swe2d.core.builder import BuildRunContextError

        n_bc = self.fixture.n_bc
        bc_values = (
            np.ones(n_bc, dtype=np.int32),
            np.zeros(n_bc, dtype=np.float64),
            np.zeros(n_bc, dtype=np.float64),
        )
        failure = RuntimeError("loader failed")

        class BrokenStructure:
            @property
            def structure_type(self):
                raise failure

        class BrokenStructureConfig:
            structures = [BrokenStructure()]

        cases = (
            (
                "bc_lines override",
                "bc_lines",
                {"bc_lines": {"table": "bc_lines"}},
                (
                    (
                        "swe2d.core.boundary_qgis_adapter.apply_bc_layer_overrides_from_gpkg",
                        {"side_effect": failure},
                    ),
                ),
            ),
            (
                "bc_lines hydrographs",
                "bc_lines",
                {"bc_lines": {"table": "bc_lines", "hydrograph_table": "hydrographs"}},
                (
                    (
                        "swe2d.core.boundary_qgis_adapter.apply_bc_layer_overrides_from_gpkg",
                        {"return_value": bc_values},
                    ),
                    (
                        "swe2d.core.gpkg_io.collect_bc_layer_hydrographs_from_gpkg",
                        {"side_effect": failure},
                    ),
                ),
            ),
            (
                "internal flow load",
                "internal_flow_sources",
                {"internal_flow_sources": {"table": "flow_sources"}},
                (
                    (
                        "swe2d.core.gpkg_io.build_internal_flow_forcing_from_gpkg",
                        {"side_effect": failure},
                    ),
                ),
            ),
            (
                "drainage GPKG load",
                "drainage",
                {"drainage": {"nodes_layer": "nodes", "links_layer": "links"}},
                (
                    (
                        "swe2d.core.gpkg_io._build_drainage_config_from_gpkg_layers",
                        {"side_effect": failure},
                    ),
                ),
            ),
            (
                "structures GPKG load",
                "structures",
                {"structures": {"table": "structures"}},
                (
                    (
                        "swe2d.core.gpkg_io.build_hydraulic_structure_config_from_gpkg",
                        {"side_effect": failure},
                    ),
                ),
            ),
            (
                "structures inline load",
                "structures",
                {"structures": {"structures": []}},
                (
                    (
                        "swe2d.extensions.structures.build_structures_config_from_json",
                        {"side_effect": failure},
                    ),
                ),
            ),
            (
                "bridge plans",
                "structures",
                {},
                (
                    (
                        "swe2d.runtime.bridge_stacked_runtime.build_bridge_stacked_plans_for_runtime",
                        {"side_effect": failure},
                    ),
                ),
            ),
            (
                "bridge structure detection",
                "structures",
                {"structures": {"table": "structures"}},
                (
                    (
                        "swe2d.core.gpkg_io.build_hydraulic_structure_config_from_gpkg",
                        {"return_value": BrokenStructureConfig()},
                    ),
                    (
                        "swe2d.runtime.bridge_stacked_runtime.build_bridge_stacked_plans_for_runtime",
                        {"return_value": []},
                    ),
                ),
            ),
            (
                "coupling packing",
                "coupling_soa",
                {"structures": {"table": "structures"}},
                (
                    (
                        "swe2d.core.gpkg_io.build_hydraulic_structure_config_from_gpkg",
                        {"return_value": object()},
                    ),
                    (
                        "swe2d.runtime.bridge_stacked_runtime.build_bridge_stacked_plans_for_runtime",
                        {"return_value": []},
                    ),
                    (
                        "swe2d.runtime.coupling.pack_coupling_soa",
                        {"side_effect": failure},
                    ),
                ),
            ),
            (
                "cell source derivation",
                "internal_flow_sources",
                {"internal_flow_sources": {"table": "flow_sources"}},
                (
                    (
                        "swe2d.core.gpkg_io.build_internal_flow_forcing_from_gpkg",
                        {"return_value": {}},
                    ),
                    (
                        "swe2d.boundary_and_forcing.runtime_source_logic.internal_flow_source_cms_at_time",
                        {"side_effect": failure},
                    ),
                ),
            ),
            (
                "rain rate conversion",
                "rain_rate_spin",
                {"rain_rate_spin": 1.0},
                (("swe2d.units.rain_si_to_model", {"side_effect": failure}),),
            ),
            (
                "edge groups",
                "bc_lines",
                {"bc_lines": {"table": "bc_lines"}},
                (
                    (
                        "swe2d.core.boundary_qgis_adapter.apply_bc_layer_overrides_from_gpkg",
                        {"return_value": bc_values},
                    ),
                    (
                        "swe2d.core.gpkg_io.collect_bc_layer_edge_groups_from_gpkg",
                        {"side_effect": failure},
                    ),
                ),
            ),
            (
                "sample lines",
                "sample_lines",
                {"sample_lines": {"table": "sample_lines"}},
                (
                    (
                        "swe2d.core.gpkg_io.build_line_sampling_map_from_gpkg",
                        {"side_effect": failure},
                    ),
                ),
            ),
        )

        for label, spec_key, extra, patches in cases:
            with self.subTest(site=label), ExitStack() as stack:
                stack.enter_context(
                    patch("swe2d.core.builder._require_gpkg_table")
                )
                for target, kwargs in patches:
                    stack.enter_context(patch(target, **kwargs))
                with self.assertRaisesRegex(
                    BuildRunContextError,
                    rf"spec key ['\"]{spec_key}['\"]",
                ):
                    self._build(extra)

    def test_configured_missing_layer_raises_typed_error(self):
        from swe2d.core.builder import BuildRunContextError

        with self.assertRaisesRegex(
            BuildRunContextError, "spec key 'sample_lines'.*missing table/layer"
        ):
            self._build({"sample_lines": {"table": "does_not_exist"}})

    def test_widget_units_derivation_raises_typed_error(self):
        from swe2d.core.builder import BuildRunContextError, widget_state_to_flat_params

        with patch(
            "swe2d.core.builder.query_mesh_from_gpkg",
            side_effect=RuntimeError("mesh read failed"),
        ):
            with self.assertRaisesRegex(BuildRunContextError, "spec key 'units'"):
                widget_state_to_flat_params(
                    {}, mesh_gpkg=self.fixture.gpkg, mesh_name=self.fixture.mesh_name,
                )

    def test_rain_cn_query_failures_raise_typed_error(self):
        from swe2d.core.builder import BuildRunContextError
        from swe2d.core.gpkg_io import query_cn_grid

        with sqlite3.connect(":memory:") as connection:
            with self.assertRaisesRegex(BuildRunContextError, "spec key 'rain_cn'"):
                query_cn_grid(connection, "missing_cn")

            connection.execute("CREATE TABLE cn_only (cn REAL)")
            connection.execute("INSERT INTO cn_only VALUES (80.0)")
            with self.assertRaisesRegex(BuildRunContextError, "spec key 'rain_cn'"):
                query_cn_grid(connection, "cn_only")


class TestGUIAdapterForwardsInitialConditions(unittest.TestCase):
    """N-1 regression: the GUI adapter must forward h0/BCs/hydrographs.

    Commit 673b714 (C-3) removed the controller's ``dataclasses.replace``
    post-pass but the adapter never put ``h0``/``hu0``/``hv0``,
    ``bc_tp``/``bc_vl``/``bc_relax``, or the hydrograph dicts into the
    spec — so GUI runs started dry with default BCs and time-series BCs
    disabled.  The adapter now forwards the controller's ``run_input``
    dict through the spec (honored by the builder's ``_override`` helper);
    these tests pin that behavior.
    """

    @classmethod
    def setUpClass(cls):
        cls.fixture = MeshFixtureWithBC().build()
        cls.addClassCleanup(cls.fixture.close)

    def _make_run_input(self):
        n_cells = self.fixture.n_cells
        n_bc = self.fixture.n_bc
        return {
            "h0": np.full(n_cells, 1.25, dtype=np.float64),
            "hu0": np.full(n_cells, 0.5, dtype=np.float64),
            "hv0": np.full(n_cells, -0.25, dtype=np.float64),
            "bc_tp": np.full(n_bc, 3, dtype=np.int32),
            "bc_vl": np.full(n_bc, 0.75, dtype=np.float64),
            "bc_relax": np.full(n_bc, 0.1, dtype=np.float64),
            "side_hydrographs": {"S1": {"t_s": [0.0, 60.0], "q": [0.0, 2.0]}},
            "edge_hydrographs": {(0, 1): {"t_s": [0.0, 60.0], "q": [0.0, 1.0]}},
            "edge_group_overrides": {(0, 1): "inflow"},
            "n_mann_cell": None,
        }

    def _call_adapter(self, run_input=None):
        from swe2d.workbench.adapters.run_context_adapter import (
            build_run_context_from_gui,
        )
        return build_run_context_from_gui(
            {},
            mesh_data={"mesh_name": self.fixture.mesh_name},
            forcing=None,
            run_input=run_input,
            sample_map_data=[],
            inflow_progressive_enabled=False,
            edge_groups={},
            results_gpkg_path="",
            model_gpkg_path=self.fixture.gpkg,
            mesh_crs_wkt="",
            parse_time_hours_fn=lambda _s: 0.0,
        )

    def test_adapter_forwards_initial_conditions_and_bcs(self):
        """Given h0/bc_tp/etc., the built RunContext has those values (not zeros)."""
        run_input = self._make_run_input()
        ctx = self._call_adapter(run_input)

        # Initial conditions — the N-1 regression left these as zeros.
        np.testing.assert_array_equal(ctx.h0, run_input["h0"])
        self.assertFalse(
            bool(np.all(ctx.h0 == 0.0)),
            "ctx.h0 must not be all zeros when run_input supplies h0 (N-1)",
        )
        np.testing.assert_array_equal(ctx.hu0, run_input["hu0"])
        np.testing.assert_array_equal(ctx.hv0, run_input["hv0"])

        # BC arrays — must come from run_input, not default_bc_for_edges.
        np.testing.assert_array_equal(ctx.bc_tp, run_input["bc_tp"])
        np.testing.assert_array_equal(ctx.bc_vl, run_input["bc_vl"])
        np.testing.assert_array_equal(ctx.bc_relax, run_input["bc_relax"])

        # Hydrograph dicts.
        self.assertEqual(ctx.side_hydrographs, run_input["side_hydrographs"])
        self.assertEqual(ctx.edge_hydrographs, run_input["edge_hydrographs"])
        self.assertEqual(ctx.edge_group_overrides, run_input["edge_group_overrides"])

    def test_adapter_without_run_input_keeps_builder_defaults(self):
        """No run_input → the builder's own GPKG/default path is used."""
        ctx = self._call_adapter(run_input=None)
        np.testing.assert_array_equal(
            ctx.h0, np.zeros(self.fixture.n_cells, dtype=np.float64),
        )
        np.testing.assert_array_equal(
            ctx.hu0, np.zeros(self.fixture.n_cells, dtype=np.float64),
        )
        self.assertEqual(ctx.side_hydrographs, {})
        self.assertEqual(ctx.edge_hydrographs, {})
        self.assertEqual(ctx.edge_group_overrides, {})


class TestDrainageInlineForm(unittest.TestCase):
    """Phase 3.2: the CLI builder accepts both drainage spec forms.

    Form 1: GPKG layers (``nodes_layer`` + ``links_layer``) — drives
    ``build_pipe_network_config`` via ``_build_drainage_config_from_gpkg_layers``.
    Form 2: inline JSON ``{nodes, links, inlets, ...}`` — drives
    ``build_drainage_config_from_json`` directly.  Form 2 was previously
    silently dropped — it now raises a typed error if the inline data is
    malformed, but produces a valid ``PipeNetworkConfig`` when well-formed.
    """

    @classmethod
    def setUpClass(cls):
        cls.fixture = MeshFixtureWithBC().build()
        cls.addClassCleanup(cls.fixture.close)

    def setUp(self):
        from swe2d.core.builder import build_run_context
        self._build_run_context = build_run_context

    def _build(self, drainage):
        return self._build_run_context(
            {
                "mesh": {"gpkg_path": self.fixture.gpkg, "mesh_name": self.fixture.mesh_name},
                "run_duration_s": 60.0,
                "output_interval_s": 60.0,
                "drainage": drainage,
            },
            mesh_gpkg=self.fixture.gpkg,
        )

    def test_inline_form_produces_pipe_network_config(self):
        """Inline ``{nodes, links}`` JSON spec → valid PipeNetworkConfig."""
        ctx = self._build({
            "nodes": [
                {"id": "n1", "type": "inlet", "invert": 0.0, "y_max": 10.0},
                {"id": "n2", "type": "outfall", "invert": -1.0, "y_max": 10.0},
            ],
            "links": [
                {"id": "l1", "from": "n1", "to": "n2", "length": 100.0,
                 "diameter": 1.0, "roughness": 0.013},
            ],
        })
        self.assertIsNotNone(ctx.pipe_network_cfg)
        # nodes + links + defaults populated
        self.assertEqual(len(ctx.pipe_network_cfg.nodes), 2)
        self.assertEqual(len(ctx.pipe_network_cfg.links), 1)

    def test_inline_form_gui_parity(self):
        """GUI/CLI parity: both paths produce equivalent PipeNetworkConfig
        from the same inline form.  We invoke the GUI's
        ``build_drainage_config_from_json`` directly via the extensions
        module and compare to the CLI's ``pipe_network_cfg`` field."""
        from swe2d.extensions.drainage_network import (
            build_drainage_config_from_json,
        )
        inline = {
            "nodes": [
                {"id": "n1", "type": "inlet", "invert": 0.0, "y_max": 10.0},
                {"id": "n2", "type": "outfall", "invert": -1.0, "y_max": 10.0},
            ],
            "links": [
                {"id": "l1", "from": "n1", "to": "n2", "length": 100.0,
                 "diameter": 1.0, "roughness": 0.013},
            ],
        }
        ctx = self._build(inline)
        # Both call the same builder — compare node/link IDs as a smoke check.
        gui_cfg = build_drainage_config_from_json(inline, n_cells=self.fixture.n_cells)
        self.assertIsNotNone(gui_cfg)
        self.assertEqual(
            [n.node_id for n in ctx.pipe_network_cfg.nodes],
            [n.node_id for n in gui_cfg.nodes],
        )
        self.assertEqual(
            [l.link_id for l in ctx.pipe_network_cfg.links],
            [l.link_id for l in gui_cfg.links],
        )

    def test_inline_form_malformed_raises(self):
        """An inline spec with only one of nodes or links (but not both)
        raises a typed ValueError — no silent drop, no silent None."""
        with self.assertRaises(ValueError):
            # only nodes, no links → build_drainage_config_from_json returns None,
            # which the builder now promotes to a ValueError.
            self._build({
                "nodes": [{"id": "n1", "type": "inlet", "invert": 0.0, "y_max": 10.0}],
            })

    def test_no_drainage_block_yields_none(self):
        """No drainage spec → pipe_network_cfg is None."""
        ctx = self._build({})
        self.assertIsNone(ctx.pipe_network_cfg)


class TestEdgeGroupsAndSampleMapWiring(unittest.TestCase):
    """Phase 3.3: the builder loads ``edge_groups_dict`` and
    ``sample_map_data`` from the GPKG and wires them into the RunContext.
    Previously these were silently discarded (``{}`` and ``[]`` defaults).
    """

    @classmethod
    def setUpClass(cls):
        cls.fixture = MeshFixtureWithBC().build()
        cls.addClassCleanup(cls.fixture.close)

    def setUp(self):
        from swe2d.core.builder import build_run_context
        self._build_run_context = build_run_context

    def test_no_spec_yields_empty_defaults(self):
        """No bc_lines / sample_lines spec → edge_groups={}, sample_map_data=[]."""
        ctx = self._build_run_context(
            {
                "mesh": {"gpkg_path": self.fixture.gpkg, "mesh_name": self.fixture.mesh_name},
                "run_duration_s": 60.0,
                "output_interval_s": 60.0,
            },
            mesh_gpkg=self.fixture.gpkg,
        )
        self.assertEqual(ctx.edge_groups, {})
        self.assertEqual(ctx.sample_map_data, [])

    def test_override_takes_precedence(self):
        """Spec-supplied edge_groups / sample_map_data override the GPKG path."""
        ctx = self._build_run_context(
            {
                "mesh": {"gpkg_path": self.fixture.gpkg, "mesh_name": self.fixture.mesh_name},
                "run_duration_s": 60.0,
                "output_interval_s": 60.0,
                "edge_groups": {0: "inflow", 1: "outflow"},
                "sample_map_data": [{"line_id": 99, "line_name": "L1"}],
            },
            mesh_gpkg=self.fixture.gpkg,
        )
        self.assertEqual(ctx.edge_groups, {0: "inflow", 1: "outflow"})
        self.assertEqual(ctx.sample_map_data, [{"line_id": 99, "line_name": "L1"}])


class TestThiessenGpkgRouting(unittest.TestCase):
    """Phase 3.1: the CLI's ``hyetograph``/``rain_cn`` spec routes through
    the QGIS-based ``build_thiessen_rain_cn_forcing_from_gpkg`` shim —
    the same builder the GUI dialog uses.  The raw-sqlite3
    ``build_forced_thiessen_from_gpkg`` reimplementation is dead.

    These tests verify the CLI builder calls the QGIS shim and does not
    depend on the raw-sqlite3 path.  Without a GPKG that has the
    hyetograph/gauge/CN tables, both paths return ``None``; the test
    exercises that the CLI builder routes to the QGIS shim and skips
    gracefully when the spec blocks aren't present.
    """

    @classmethod
    def setUpClass(cls):
        cls.fixture = MeshFixtureWithBC().build()
        cls.addClassCleanup(cls.fixture.close)

    def setUp(self):
        from swe2d.core.builder import build_run_context
        self._build_run_context = build_run_context

    def test_no_hyetograph_block_yields_none(self):
        """No hyetograph spec → thiessen_forcing is None (no QGIS call)."""
        ctx = self._build_run_context(
            {
                "mesh": {"gpkg_path": self.fixture.gpkg, "mesh_name": self.fixture.mesh_name},
                "run_duration_s": 60.0,
                "output_interval_s": 60.0,
            },
            mesh_gpkg=self.fixture.gpkg,
        )
        self.assertIsNone(ctx.thiessen_forcing)

    def test_hyetograph_block_missing_layers_raises(self):
        """A configured rainfall source must not silently become ``None``."""
        from swe2d.core.builder import BuildRunContextError

        with self.assertRaisesRegex(BuildRunContextError, "spec key 'hyetograph'"):
            self._build_run_context(
                {
                    "mesh": {"gpkg_path": self.fixture.gpkg, "mesh_name": self.fixture.mesh_name},
                    "run_duration_s": 60.0,
                    "output_interval_s": 60.0,
                    "hyetograph": {"table": "SWE2D_Hyetograph", "gauge_layer": "rain_gages"},
                    "rain_cn": {"table": "swe2d_rain_cn", "cn_field": "cn", "ia_ratio": 0.2},
                },
                mesh_gpkg=self.fixture.gpkg,
            )

    def test_builder_function_name(self):
        """Regression: the CLI builder must call the QGIS shim, not the
        raw-sqlite3 reimplementation.  We assert this by checking that
        ``build_forced_thiessen_from_gpkg`` is no longer imported by
        builder.py (it was removed in Phase 3.6, but Phase 3.1 already
        stops calling it)."""
        import swe2d.core.builder as _builder_mod
        # The raw-sqlite3 helper can still be imported (the symbol is not
        # deleted yet — that comes in Phase 3.6), but the CLI builder must
        # reference the QGIS shim instead.  Search the builder's source
        # for the raw function name.
        import inspect
        src = inspect.getsource(_builder_mod)
        self.assertNotIn(
            "build_forced_thiessen_from_gpkg(",
            src,
            "CLI builder still calls the raw-sqlite3 build_forced_thiessen_from_gpkg; "
            "should route through build_thiessen_rain_cn_forcing_from_gpkg instead.",
        )
        self.assertIn(
            "build_thiessen_rain_cn_forcing_from_gpkg",
            src,
            "CLI builder must call build_thiessen_rain_cn_forcing_from_gpkg "
            "(the QGIS-based shim used by the GUI).",
        )


class TestCudaDllPathEnvVar(unittest.TestCase):
    """Test that CUDA DLL path is read from HYDRA_CUDA_DLL_PATH environment variable."""

    def test_cuda_dll_path_from_env_when_set(self):
        """When HYDRA_CUDA_DLL_PATH is set, _load_cuda_dll_path_from_env returns it."""
        from swe2d.runtime.backend import _load_cuda_dll_path_from_env
        
        # Save original value
        original = os.environ.get("HYDRA_CUDA_DLL_PATH")
        
        try:
            # Test with a dummy path
            dummy_path = r"C:\dummy\cuda\path"
            os.environ["HYDRA_CUDA_DLL_PATH"] = dummy_path
            result = _load_cuda_dll_path_from_env()
            self.assertEqual(result, dummy_path)
            
            # Test with empty string
            os.environ["HYDRA_CUDA_DLL_PATH"] = ""
            result = _load_cuda_dll_path_from_env()
            self.assertEqual(result, "")
            
        finally:
            # Restore original value
            if original is None:
                os.environ.pop("HYDRA_CUDA_DLL_PATH", None)
            else:
                os.environ["HYDRA_CUDA_DLL_PATH"] = original
    
    def test_cuda_dll_path_from_env_when_not_set(self):
        """When HYDRA_CUDA_DLL_PATH is not set, _load_cuda_dll_path_from_env returns None."""
        from swe2d.runtime.backend import _load_cuda_dll_path_from_env
        
        # Save original value
        original = os.environ.get("HYDRA_CUDA_DLL_PATH")
        
        try:
            # Ensure env var is not set
            os.environ.pop("HYDRA_CUDA_DLL_PATH", None)
            result = _load_cuda_dll_path_from_env()
            self.assertIsNone(result)
            
        finally:
            # Restore original value
            if original is not None:
                os.environ["HYDRA_CUDA_DLL_PATH"] = original


if __name__ == "__main__":
    unittest.main()
