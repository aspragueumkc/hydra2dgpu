"""Validate V3 deliverable: SWMM-vs-pipe1D comparison harness."""

from __future__ import annotations

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))


class TestV3CompareHarness(unittest.TestCase):
    """V3: SWMM-vs-pipe1D comparison harness."""

    def test_harness_module_imports(self):
        m = importlib.import_module("tests.swmm_validation.compare")
        self.assertTrue(hasattr(m, "run_comparison"))
        self.assertTrue(hasattr(m, "ScenarioBundle"))
        self.assertTrue(hasattr(m, "ToleranceSpec"))
        self.assertTrue(hasattr(m, "TimeSeries"))
        self.assertTrue(hasattr(m, "ErrorStats"))
        self.assertTrue(hasattr(m, "ComparisonResult"))

    def test_dataclass_field_signatures(self):
        m = importlib.import_module("tests.swmm_validation.compare")
        # ScenarioBundle fields
        sb = m.ScenarioBundle.__dataclass_fields__
        for fname in ("name", "swmm_inp_path", "pipe1d_config", "duration_s",
                      "hydrology", "expected_regimes"):
            self.assertIn(fname, sb)
        # ToleranceSpec has regimes dict
        ts = m.ToleranceSpec.__dataclass_fields__
        self.assertIn("regimes", ts)
        # ComparisonResult has pass_fail + node/link series + errors
        cr = m.ComparisonResult.__dataclass_fields__
        for fname in ("scenario", "node_series", "link_series",
                      "node_errors", "link_errors", "pass_fail", "metadata"):
            self.assertIn(fname, cr)

    def test_run_comparison_runs_and_writes_json(self):
        """End-to-end test: run a comparison and verify JSON written."""
        from tests.swmm_validation.compare import (
            run_comparison, ScenarioBundle, ToleranceSpec
        )
        from tests.swmm_validation.configs.site_drainage import (
            make_site_drainage_config, get_rainfall_timeseries,
        )

        cfg = make_site_drainage_config()
        rain = get_rainfall_timeseries()

        bundle = ScenarioBundle(
            name="site_drainage_v3_test",
            swmm_inp_path=REPO_ROOT / "reference/swmm_canonical/Site_Drainage_Model.inp",
            pipe1d_config=cfg,
            duration_s=24 * 3600.0,
            hydrology={"rainfall": rain},
            expected_regimes=["open_channel"],
        )

        tols = ToleranceSpec(
            regimes={"open_channel": {"depth_max_rel": 1.0, "flow_max_rel": 1.0}},
        )

        with tempfile.TemporaryDirectory() as td:
            workdir = Path(td)
            result = run_comparison(bundle, tols, workdir)
            self.assertIsNotNone(result)
            self.assertEqual(result.scenario, "site_drainage_v3_test")
            # pass_fail should have entries (either True/False for matched regimes,
            # or empty dict if both solvers failed)
            self.assertIsInstance(result.pass_fail, dict)
            # JSON file should be written even on partial failure
            json_path = workdir / "result.json"
            self.assertTrue(json_path.exists(), "result.json not written")
            # JSON should be valid + contain expected top-level fields
            with open(json_path) as f:
                data = json.load(f)
            self.assertIn("scenario", data)
            self.assertEqual(data["scenario"], "site_drainage_v3_test")
            self.assertIn("pass_fail", data)
            self.assertIn("metadata", data)

    def test_run_comparison_graceful_when_swmm_unavailable(self):
        """Harness should not crash if SWMM cannot run — return result with metadata.error."""
        from tests.swmm_validation.compare import (
            run_comparison, ScenarioBundle, ToleranceSpec
        )
        from tests.swmm_validation.configs.culvert import make_culvert_config

        cfg = make_culvert_config()
        # Point to a non-existent file to force SWMM failure
        bundle = ScenarioBundle(
            name="graceful_test",
            swmm_inp_path=Path("/nonexistent/path/SWMM_FAIL.inp"),
            pipe1d_config=cfg,
            duration_s=24 * 3600.0,
            hydrology={},
            expected_regimes=["open_channel"],
        )

        tols = ToleranceSpec(regimes={"open_channel": {}})

        with tempfile.TemporaryDirectory() as td:
            workdir = Path(td)
            # Should not raise — should return a ComparisonResult with error info
            try:
                result = run_comparison(bundle, tols, workdir)
                self.assertIsNotNone(result)
                # Pass_fail may be empty or partial
                self.assertIsInstance(result.pass_fail, dict)
            except Exception as e:
                self.fail(f"run_comparison raised on missing file: {type(e).__name__}: {e}")


if __name__ == "__main__":
    unittest.main()
