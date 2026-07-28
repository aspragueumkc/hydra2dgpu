"""Validate V2 deliverables: parsed JSON → config → synthetic variants → SoA packing.

Tests the full chain built by the V2 subagent fleet:
- 4 parsed JSON files (Phase 1)
- 4 config modules (Phase 2)
- 24 synthetic variants across 4 scenarios (Phase 3)
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import unittest
from pathlib import Path

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))


V2_ROOT = REPO_ROOT / "tests" / "swmm_validation"


class TestV2ParsedJSONs(unittest.TestCase):
    """Phase 1: 4 canonical .inp files parsed to JSON."""

    EXPECTED_SCENARIOS = [
        "site-drainage",
        "detention-pond",
        "inlet-drains",
        "culvert",
    ]

    def test_all_parsed_json_exist(self):
        for s in self.EXPECTED_SCENARIOS:
            path = V2_ROOT / "parsed" / f"{s}.json"
            self.assertTrue(path.exists(), f"missing {path}")
            self.assertGreater(path.stat().st_size, 500, f"{path} too small")

    def test_parsed_json_valid(self):
        for s in self.EXPECTED_SCENARIOS:
            path = V2_ROOT / "parsed" / f"{s}.json"
            with open(path) as f:
                data = json.load(f)
            self.assertIn("scenario_name", data)
            self.assertIn("options", data)
            self.assertIn("junctions", data)
            self.assertIn("conduits", data)
            # scenario_name like "Site_Drainage_Model" should normalize to "site-drainage"
            normalized = data["scenario_name"].lower().replace("_", "-").replace("-model", "")
            self.assertEqual(normalized, s)


class TestV2Configs(unittest.TestCase):
    """Phase 2: 4 config modules that translate parsed JSON to PipeNetworkConfig."""

    EXPECTED_SCENARIOS = [
        # (scenario, n_junctions, n_links, n_outfalls, n_storage, n_pipe_ends)
        # Note: links includes both conduits and weir/orifice translations where applicable.
        # The culvert subagent chose to store the Roadway weir in metadata (not as a DrainageLink)
        # since the project's pipe1D has no explicit weir element.
        ("site_drainage", 11, 11, 1, 0, 0),
        ("detention_pond", 12, 14, 1, 1, 0),   # 12 conduits + 1 weir + 1 orifice
        ("inlet_drains", 19, 23, 1, 0, 0),
        ("culvert", 1, 2, 1, 1, 0),            # 2 conduits; weir stored in metadata
    ]

    def test_all_config_modules_importable(self):
        for scen, *_ in self.EXPECTED_SCENARIOS:
            m = importlib.import_module(
                f"tests.swmm_validation.configs.{scen}")
            self.assertTrue(hasattr(m, f"make_{scen}_config"))

    def test_config_topology(self):
        for scen, n_junc, n_cond, n_out, n_stor, n_pe in self.EXPECTED_SCENARIOS:
            m = importlib.import_module(
                f"tests.swmm_validation.configs.{scen}")
            cfg = m.make_X_config = getattr(m, f"make_{scen}_config")()
            n_junc_actual = sum(1 for n in cfg.nodes if n.node_type == "junction")
            n_out_actual = sum(1 for n in cfg.nodes if n.node_type == "outfall")
            n_stor_actual = sum(1 for n in cfg.nodes if n.node_type == "storage")
            n_total = n_junc_actual + n_out_actual + n_stor_actual
            self.assertEqual(
                n_total, n_junc + n_out + n_stor,
                f"{scen}: expected {n_junc + n_out + n_stor} nodes "
                f"({n_junc} junctions + {n_out} outfalls + {n_stor} storage), "
                f"got {n_total} ({n_junc_actual} junctions + {n_out_actual} outfalls + {n_stor_actual} storage)",
            )
            self.assertEqual(len(cfg.links), n_cond,
                             f"{scen}: expected {n_cond} links, got {len(cfg.links)}")
            self.assertEqual(len(cfg.outfalls), n_out,
                             f"{scen}: expected {n_out} outfalls, got {len(cfg.outfalls)}")
            self.assertEqual(len(cfg.pipe_ends), n_pe,
                             f"{scen}: expected {n_pe} pipe_ends, got {len(cfg.pipe_ends)}")
            self.assertEqual(cfg.gravity, 9.81)
            self.assertIn(cfg.pipe_solver_mode, ("diffusion_wave", "fully_dynamic"))


class TestV2Synthetics(unittest.TestCase):
    """Phase 3: 24 synthetic variants (4 scenarios × 6 variants) covering all outfall modes."""

    EXPECTED_SCENARIOS = ["site_drainage", "detention_pond", "inlet_drains", "culvert"]
    EXPECTED_VARIANTS = [
        "normal_depth",
        "fixed_wse",
        "rating_curve",
        "tabular",
        "supercritical",
        "pipe_end",
    ]

    def test_all_synthetic_variants_importable(self):
        for scen in self.EXPECTED_SCENARIOS:
            for variant in self.EXPECTED_VARIANTS:
                name = f"synth_{scen}_{variant}"
                m = importlib.import_module(
                    f"tests.swmm_validation.synthetic.{scen}")
                self.assertTrue(
                    hasattr(m, name),
                    f"missing {name} in tests.swmm_validation.synthetic.{scen}",
                )

    def test_all_variants_produce_valid_configs(self):
        for scen in self.EXPECTED_SCENARIOS:
            for variant in self.EXPECTED_VARIANTS:
                m = importlib.import_module(
                    f"tests.swmm_validation.synthetic.{scen}")
                fn = getattr(m, f"synth_{scen}_{variant}")
                cfg = fn()
                self.assertIsNotNone(cfg)
                self.assertGreater(len(cfg.nodes), 0, f"{scen}.{variant} has no nodes")
                self.assertGreater(len(cfg.links), 0, f"{scen}.{variant} has no links")

    def test_outfall_modes_covered(self):
        """At least one of each mode: free, normal_depth, fixed_wse, rating_curve, tabular, pipe_end."""
        seen_modes = set()
        for scen in self.EXPECTED_SCENARIOS:
            for variant in self.EXPECTED_VARIANTS:
                m = importlib.import_module(
                    f"tests.swmm_validation.synthetic.{scen}")
                cfg = getattr(m, f"synth_{scen}_{variant}")()
                for n in cfg.nodes:
                    if n.node_type == "outfall":
                        seen_modes.add(getattr(n, "outfall_mode", "free"))
                    elif n.node_type == "pipe_end":
                        seen_modes.add("pipe_end")
        expected = {"free", "normal_depth", "fixed_wse", "rating_curve", "tabular", "pipe_end"}
        missing = expected - seen_modes
        self.assertFalse(missing, f"missing outfall modes: {missing}")
        self.assertTrue(seen_modes.issuperset(expected),
                        f"seen={seen_modes}, expected={expected}")


class TestV2SoAPacking(unittest.TestCase):
    """All configs (4 base + 24 synthetics) must pack into a SoA without errors."""

    def _get_all_configs(self):
        """Return list of (name, config) tuples for all 4 base + 24 synthetic configs."""
        configs = []
        for scen in TestV2Synthetics.EXPECTED_SCENARIOS:
            m = importlib.import_module(
                f"tests.swmm_validation.configs.{scen}")
            cfg = getattr(m, f"make_{scen}_config")()
            configs.append((f"base/{scen}", cfg))
            for variant in TestV2Synthetics.EXPECTED_VARIANTS:
                sm = importlib.import_module(
                    f"tests.swmm_validation.synthetic.{scen}")
                cfg2 = getattr(sm, f"synth_{scen}_{variant}")()
                configs.append((f"synth/{scen}/{variant}", cfg2))
        return configs

    def test_all_configs_pack_to_soa(self):
        from swe2d.runtime.coupling import pack_pipe_network_soa
        for name, cfg in self._get_all_configs():
            try:
                soa = pack_pipe_network_soa(cfg, n_cells=100)
                self.assertIsNotNone(soa,
                                     f"pack_pipe_network_soa returned None for {name}")
            except Exception as e:
                self.fail(f"pack failed for {name}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    unittest.main()
