"""Validate V4 deliverable: tolerance framework + regime detector."""

from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))


class TestV4Tolerances(unittest.TestCase):
    """V4: per-regime tolerance framework + regime detector."""

    def test_module_imports(self):
        m = importlib.import_module("tests.swmm_validation.tolerances")
        self.assertTrue(hasattr(m, "TOLERANCES"))
        self.assertTrue(hasattr(m, "RegimeTolerances"))
        self.assertTrue(hasattr(m, "detect_regime"))
        self.assertTrue(hasattr(m, "get_tolerance"))
        self.assertTrue(hasattr(m, "is_within_tolerance"))
        self.assertTrue(hasattr(m, "classify_node_regime"))

    def test_tolerances_dict_covers_all_regimes(self):
        m = importlib.import_module("tests.swmm_validation.tolerances")
        required = {"open_channel", "transition", "surcharged",
                    "fixed_wse_outfall", "junction_overflow",
                    "supercritical", "pump"}
        actual = set(m.TOLERANCES.keys())
        self.assertTrue(actual.issuperset(required),
                        f"missing: {required - actual}")

    def test_regime_detection_open_channel(self):
        m = importlib.import_module("tests.swmm_validation.tolerances")
        # depth well below full → open_channel
        self.assertEqual(m.detect_regime(0.3, 1.0, 1.0), "open_channel")
        self.assertEqual(m.detect_regime(0.5, 1.0, 1.0), "open_channel")
        self.assertEqual(m.detect_regime(0.89, 1.0, 1.0), "open_channel")

    def test_regime_detection_transition(self):
        m = importlib.import_module("tests.swmm_validation.tolerances")
        # depth between 0.9*d_full and d_crown → transition
        self.assertEqual(m.detect_regime(0.9, 1.0, 1.0), "transition")
        self.assertEqual(m.detect_regime(0.95, 1.0, 1.0), "transition")
        self.assertEqual(m.detect_regime(0.99, 1.0, 1.0), "transition")

    def test_regime_detection_surcharged(self):
        m = importlib.import_module("tests.swmm_validation.tolerances")
        # depth above d_crown → surcharged
        self.assertEqual(m.detect_regime(1.01, 1.0, 1.0), "surcharged")
        self.assertEqual(m.detect_regime(1.5, 1.0, 1.0), "surcharged")
        self.assertEqual(m.detect_regime(2.0, 1.0, 1.0), "surcharged")

    def test_regime_detection_froude_above_1(self):
        m = importlib.import_module("tests.swmm_validation.tolerances")
        # Froude only matters when depth >= d_crown; below crown the depth
        # branch (open_channel or transition) wins regardless of froude.
        # Subcritical pressurised: depth above crown, froude < 1
        self.assertEqual(m.detect_regime(1.5, 1.0, 1.0, froude=0.5),
                         "surcharged")
        # Supercritical pressurised: depth above crown, froude >= 1
        self.assertEqual(m.detect_regime(1.5, 1.0, 1.0, froude=1.5),
                         "surcharged")
        # Below crown, high froude is supercritical open-channel (NOT surcharged)
        self.assertEqual(m.detect_regime(0.5, 1.0, 1.0, froude=1.5),
                         "open_channel")

    def test_get_tolerance_known_regimes(self):
        m = importlib.import_module("tests.swmm_validation.tolerances")
        # Open-channel: 5% depth, 10% flow
        self.assertEqual(m.get_tolerance("open_channel", "depth_max_rel"), 0.05)
        self.assertEqual(m.get_tolerance("open_channel", "flow_max_rel"), 0.10)
        # Surcharged: tighter, 2% depth, 5% flow
        self.assertEqual(m.get_tolerance("surcharged", "depth_max_rel"), 0.02)
        self.assertEqual(m.get_tolerance("surcharged", "flow_max_rel"), 0.05)
        # Pump: very tight, 2%
        self.assertEqual(m.get_tolerance("pump", "flow_max_rel"), 0.02)

    def test_get_tolerance_unknown_raises(self):
        m = importlib.import_module("tests.swmm_validation.tolerances")
        with self.assertRaises(KeyError):
            m.get_tolerance("nonexistent_regime", "depth_max_rel")
        with self.assertRaises(KeyError):
            m.get_tolerance("open_channel", "nonexistent_metric")

    def test_is_within_tolerance(self):
        m = importlib.import_module("tests.swmm_validation.tolerances")
        from tests.swmm_validation.compare import ErrorStats

        # Open-channel: 5% tolerance
        stats_pass = ErrorStats(rmse=0.04, max_abs=0.06, max_rel=0.05, n_points=100)
        stats_fail = ErrorStats(rmse=0.04, max_abs=0.06, max_rel=0.08, n_points=100)
        self.assertTrue(m.is_within_tolerance(stats_pass, "open_channel", "depth_max_rel"))
        self.assertFalse(m.is_within_tolerance(stats_fail, "open_channel", "depth_max_rel"))

        # Surcharged: 2% tolerance — same value fails
        self.assertFalse(m.is_within_tolerance(stats_pass, "surcharged", "depth_max_rel"))

    def test_classify_node_regime_worst_case(self):
        m = importlib.import_module("tests.swmm_validation.tolerances")
        # All open_channel
        self.assertEqual(
            m.classify_node_regime([0.3, 0.4, 0.5], 1.0, 1.0),
            "open_channel",
        )
        # Mostly open but has one transition
        self.assertEqual(
            m.classify_node_regime([0.3, 0.5, 0.95], 1.0, 1.0),
            "transition",
        )
        # Has at least one surcharged
        self.assertEqual(
            m.classify_node_regime([0.3, 0.5, 1.2], 1.0, 1.0),
            "surcharged",
        )
        # All surcharged
        self.assertEqual(
            m.classify_node_regime([1.1, 1.2, 1.3], 1.0, 1.0),
            "surcharged",
        )


if __name__ == "__main__":
    unittest.main()
