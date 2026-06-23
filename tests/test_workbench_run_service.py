"""Tests for swe2d/workbench/run_service.py."""

import unittest
import math

from swe2d.workbench.services.run_service import (
    collect_run_parameters,
    compute_progress,
    validate_run_configuration,
)


class TestCollectRunParameters(unittest.TestCase):
    """collect_run_parameters returns a typed dict of solver parameters."""

    def test_returns_dict_with_required_keys(self):
        result = collect_run_parameters(
            cfl=0.8,
            dt=0.1,
            run_duration=3600.0,
            output_interval=300.0,
            n_mann=0.035,
            spatial_scheme=2,
            temporal_scheme=1,
            gravity=9.81,
            h_min=1e-6,
            use_gpu=True,
            checkpoint_interval=600.0,
        )
        self.assertIsInstance(result, dict)
        # Basic solver parameters
        self.assertIn("cfl", result)
        self.assertIn("dt", result)
        self.assertIn("run_duration_s", result)
        self.assertIn("output_interval_s", result)
        self.assertIn("n_mann", result)
        self.assertIn("spatial_scheme", result)
        self.assertIn("temporal_scheme", result)
        self.assertIn("gravity", result)
        self.assertIn("h_min", result)
        self.assertIn("use_gpu", result)
        self.assertIn("checkpoint_interval_s", result)

    def test_preserves_numeric_values(self):
        result = collect_run_parameters(
            cfl=0.9,
            dt=0.05,
            run_duration=7200.0,
            output_interval=600.0,
            n_mann=0.040,
            spatial_scheme=3,
            temporal_scheme=2,
            gravity=32.17,
            h_min=1e-5,
            use_gpu=False,
            checkpoint_interval=1200.0,
        )
        self.assertAlmostEqual(result["cfl"], 0.9)
        self.assertAlmostEqual(result["dt"], 0.05)
        self.assertAlmostEqual(result["run_duration_s"], 7200.0)
        self.assertAlmostEqual(result["output_interval_s"], 600.0)
        self.assertAlmostEqual(result["n_mann"], 0.040)
        self.assertEqual(result["spatial_scheme"], 3)
        self.assertEqual(result["temporal_scheme"], 2)
        self.assertAlmostEqual(result["gravity"], 32.17)
        self.assertAlmostEqual(result["h_min"], 1e-5)
        self.assertFalse(result["use_gpu"])
        self.assertAlmostEqual(result["checkpoint_interval_s"], 1200.0)

    def test_accepts_optional_advanced_parameters(self):
        result = collect_run_parameters(
            cfl=0.8,
            dt=0.1,
            run_duration=3600.0,
            output_interval=300.0,
            n_mann=0.035,
            spatial_scheme=2,
            temporal_scheme=1,
            gravity=9.81,
            h_min=1e-6,
            use_gpu=True,
            checkpoint_interval=600.0,
            max_rel_depth_increase=2.0,
            gpu_diag_sync_interval=10,
            shallow_damping_depth=1e-4,
            depth_cap=1e6,
            momentum_cap_min_speed=50.0,
            momentum_cap_celerity_mult=20.0,
            max_inv_area=1e8,
            cfl_lambda_cap=1e6,
            extreme_rain_mode=False,
            source_cfl_beta=0.25,
            source_max_substeps=16,
            source_true_subcycling=False,
            source_imex_split=False,
            source_stage_coupled_imex_rk2=False,
            tiny_mode=0,
            tiny_wet_cell_threshold=200,
            inflow_progressive=True,
        )
        self.assertEqual(result.get("max_rel_depth_increase"), 2.0)
        self.assertEqual(result.get("gpu_diag_sync_interval"), 10)
        self.assertEqual(result.get("shallow_damping_depth"), 1e-4)
        self.assertEqual(result.get("depth_cap"), 1e6)
        self.assertEqual(result.get("tiny_mode"), 0)
        self.assertEqual(result.get("inflow_progressive"), True)

    def test_requires_mandatory_parameters(self):
        with self.assertRaises(TypeError):
            collect_run_parameters()  # type: ignore[call-arg]


class TestComputeProgress(unittest.TestCase):
    """compute_progress returns progress metrics given simulation state."""

    def test_returns_expected_keys(self):
        result = compute_progress(
            run_time=1800.0,
            total_duration=3600.0,
            wall_elapsed=60.0,
        )
        self.assertIsInstance(result, dict)
        self.assertIn("percent", result)
        self.assertIn("eta_s", result)
        self.assertIn("wall_elapsed_s", result)
        self.assertIn("speedup", result)

    def test_halfway_progress(self):
        result = compute_progress(
            run_time=1800.0,
            total_duration=3600.0,
            wall_elapsed=90.0,
        )
        self.assertAlmostEqual(result["percent"], 50.0)
        self.assertAlmostEqual(result["eta_s"], 90.0)

    def test_zero_progress(self):
        result = compute_progress(
            run_time=0.0,
            total_duration=3600.0,
            wall_elapsed=0.0,
        )
        self.assertEqual(result["percent"], 0.0)

    def test_complete(self):
        result = compute_progress(
            run_time=3600.0,
            total_duration=3600.0,
            wall_elapsed=600.0,
        )
        self.assertAlmostEqual(result["percent"], 100.0)
        self.assertAlmostEqual(result["eta_s"], 0.0)

    def test_speedup_computation(self):
        result = compute_progress(
            run_time=100.0,
            total_duration=200.0,
            wall_elapsed=50.0,
        )
        # speedup = run_time / wall_elapsed = 100/50 = 2
        self.assertAlmostEqual(result["speedup"], 2.0)

    def test_zero_wall_elapsed_speedup_is_zero(self):
        result = compute_progress(
            run_time=0.0,
            total_duration=3600.0,
            wall_elapsed=0.0,
        )
        self.assertEqual(result["speedup"], 0.0)

    def test_small_positive_progress(self):
        result = compute_progress(
            run_time=1.0,
            total_duration=3600.0,
            wall_elapsed=0.5,
        )
        self.assertGreater(result["percent"], 0.0)
        self.assertLess(result["percent"], 1.0)
        # ETA: remaining (3599) * (wall_elapsed / run_time) = 3599 * 0.5 / 1.0 = 1799.5
        expected_eta = (3599.0 * 0.5) / 1.0
        self.assertAlmostEqual(result["eta_s"], expected_eta, places=5)

    def test_no_negative_percent(self):
        result = compute_progress(
            run_time=-10.0,
            total_duration=3600.0,
            wall_elapsed=5.0,
        )
        self.assertGreaterEqual(result["percent"], 0.0)

    def test_no_negative_eta(self):
        result = compute_progress(
            run_time=4000.0,
            total_duration=3600.0,
            wall_elapsed=100.0,
        )
        self.assertGreaterEqual(result["eta_s"], 0.0)


class TestValidateRunConfiguration(unittest.TestCase):
    """validate_run_configuration returns list of error messages."""

    def test_valid_config_returns_empty_list(self):
        params = {
            "cfl": 0.8,
            "dt": 0.1,
            "run_duration_s": 3600.0,
            "output_interval_s": 300.0,
            "n_mann": 0.035,
            "spatial_scheme": 2,
            "temporal_scheme": 1,
            "gravity": 9.81,
            "h_min": 1e-6,
            "use_gpu": True,
            "checkpoint_interval_s": 600.0,
        }
        errors = validate_run_configuration(params)
        self.assertEqual(errors, [])

    def test_detects_negative_cfl(self):
        params = {
            "cfl": -0.1,
            "dt": 0.1,
            "run_duration_s": 3600.0,
            "output_interval_s": 300.0,
        }
        errors = validate_run_configuration(params)
        self.assertTrue(any("cfl" in e.lower() for e in errors))

    def test_detects_zero_dt(self):
        params = {
            "cfl": 0.8,
            "dt": 0.0,
            "run_duration_s": 3600.0,
            "output_interval_s": 300.0,
        }
        errors = validate_run_configuration(params)
        self.assertTrue(any("dt" in e.lower() or "timestep" in e.lower() for e in errors))

    def test_detects_negative_run_duration(self):
        params = {
            "cfl": 0.8,
            "dt": 0.1,
            "run_duration_s": -100.0,
            "output_interval_s": 300.0,
        }
        errors = validate_run_configuration(params)
        self.assertTrue(any("duration" in e.lower() for e in errors))

    def test_detects_zero_output_interval(self):
        params = {
            "cfl": 0.8,
            "dt": 0.1,
            "run_duration_s": 3600.0,
            "output_interval_s": 0.0,
        }
        errors = validate_run_configuration(params)
        self.assertTrue(any("output" in e.lower() for e in errors))

    def test_detects_negative_n_mann(self):
        params = {
            "cfl": 0.8,
            "dt": 0.1,
            "run_duration_s": 3600.0,
            "output_interval_s": 300.0,
            "n_mann": -0.01,
        }
        errors = validate_run_configuration(params)
        self.assertTrue(any("mann" in e.lower() or "mannings" in e.lower() for e in errors))

    def test_detects_out_of_range_spatial_scheme(self):
        params = {
            "cfl": 0.8,
            "dt": 0.1,
            "run_duration_s": 3600.0,
            "output_interval_s": 300.0,
            "spatial_scheme": 99,
        }
        errors = validate_run_configuration(params)
        self.assertTrue(any("spatial" in e.lower() or "scheme" in e.lower() for e in errors))

    def test_missing_required_key_reports_error(self):
        params = {
            "cfl": 0.8,
            "dt": 0.1,
        }
        errors = validate_run_configuration(params)
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("missing" in e.lower() or "required" in e.lower() for e in errors))

    def test_accumulates_multiple_errors(self):
        params = {
            "cfl": -0.5,
            "dt": 0.0,
            "run_duration_s": -100.0,
            "output_interval_s": 0.0,
        }
        errors = validate_run_configuration(params)
        self.assertGreaterEqual(len(errors), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
