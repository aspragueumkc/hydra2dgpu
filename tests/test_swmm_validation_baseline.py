"""V8: SWMM-vs-pipe1D regression baseline guard.

Verifies that no scenario regresses from the locked baseline pass/fail state.
Run with:
    python3 -m unittest tests.test_swmm_validation_baseline -v

File ownership: ONLY this file. Do NOT modify any other file.
"""

import importlib
import json
import sys
import traceback
import unittest
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))


# --------------------------------------------------------------------------- #
# Scenario definitions (mirrors generate_report.py SCENARIOS)
# --------------------------------------------------------------------------- #
# Each tuple: (scenario_id, swmm_inp_filename, config_module, regime,
#              forcing_type, is_steady, extra_config_fn)

SCENARIOS: list[tuple[str, str, str, str, str, bool, Optional[str]]] = [
    # ── Steady-state (V5) ───────────────────────────────────────────────────
    ("open_channel_steady",                "Site_Drainage_Model.inp",  "site_drainage",
     "open_channel",     "inflow",        True,  None),
    ("surcharge_steady",                  "Detention_Pond_Model.inp", "detention_pond",
     "surcharged",      "inflow",        True,  None),
    ("outfall_backwater_fixed_wse",       "Site_Drainage_Model.inp",  "site_drainage",
     "fixed_wse_outfall","inflow",        True,  "fixed_wse"),
    ("culvert_pressurized",               "Culvert_Model.inp",        "culvert",
     "surcharged",      "inflow",        True,  None),
    ("inlet_drains_steady",               "Inlet_Drains_Model.inp",   "inlet_drains",
     "open_channel",    "inflow",        True,  None),
    # ── Dynamic (V6) ─────────────────────────────────────────────────────────
    ("uniform_hyetograph_open_channel",   "Site_Drainage_Model.inp",  "site_drainage",
     "open_channel",    "uniform_rain",  False, None),
    ("scs_type_ii_to_surcharge",         "Site_Drainage_Model.inp",  "site_drainage",
     "transition",      "scs_rain",      False, None),
    ("uniform_hyetograph_surcharge",     "Detention_Pond_Model.inp", "detention_pond",
     "surcharged",      "uniform_rain",  False, None),
    ("inlet_drains_dynamic",             "Inlet_Drains_Model.inp",   "inlet_drains",
     "transition",      "scs_rain",      False, None),
    ("culvert_surcharge_dynamic",        "Culvert_Model.inp",        "culvert",
     "surcharged",      "uniform_rain",  False, None),
]


# --------------------------------------------------------------------------- #
# Forcing generators (mirrors generate_report.py)
# --------------------------------------------------------------------------- #

def _const_inflow(node_id: str, flow_cms: float, duration_h: float = 24.0) -> dict:
    """Constant inflow time series for steady-state scenarios."""
    return {node_id: [(0.0, float(flow_cms)), (float(duration_h), float(flow_cms))]}


def _uniform_hyetograph(
    duration_s: float,
    peak_intensity_mm_hr: float,
    dt_s: float = 300.0,
) -> list[tuple[float, float]]:
    """5-minute uniform rainfall."""
    times_s: list[float] = []
    values: list[float] = []
    t = 0.0
    while t <= duration_s:
        times_s.append(t)
        values.append(float(peak_intensity_mm_hr))
        t += dt_s
    if times_s[-1] < duration_s:
        times_s.append(float(duration_s))
        values.append(float(peak_intensity_mm_hr))
    return list(zip(times_s, values))


def _scs_type_ii_hyetograph(
    duration_s: float,
    peak_intensity_mm_hr: float,
    n_points: int = 48,
) -> list[tuple[float, float]]:
    """SCS Type II 24-hour storm."""
    times_dimless = [i / float(n_points) for i in range(n_points + 1)]
    cum_frac: list[float] = []
    for lam in times_dimless:
        lam2 = lam * lam
        denom = 1.0 - lam + lam2
        cum_frac.append(lam2 / denom if denom > 0.0 else 1.0)

    incr_frac = [cum_frac[0]] + [cum_frac[i] - cum_frac[i - 1] for i in range(1, len(cum_frac))]
    total = sum(incr_frac)
    if total > 0.0:
        incr_frac = [v / total for v in incr_frac]

    result: list[tuple[float, float]] = []
    for i, frac in enumerate(incr_frac):
        t_s = float(i) / float(n_points) * float(duration_s)
        intensity = frac * float(peak_intensity_mm_hr) * float(n_points)
        result.append((t_s, intensity))
    return result


# --------------------------------------------------------------------------- #
# Scenario runner (simplified from generate_report.py)
# --------------------------------------------------------------------------- #

def _run_single_scenario(
    scenario_id: str,
    swmm_inp_filename: str,
    config_module: str,
    regime: str,
    forcing_type: str,
    is_steady: bool,
    extra_config_fn: Optional[str],
    workdir: Path,
) -> Tuple[Optional[dict], Optional[str]]:
    """Run a single scenario and return (pass_fail_dict, regime).

    Returns (None, error_str) if SWMM is unavailable or run fails.
    """
    from tests.swmm_validation.compare import (
        ComparisonResult,
        run_comparison,
        ScenarioBundle,
        ToleranceSpec,
    )
    from tests.swmm_validation.tolerances import TOLERANCES

    # Build PipeNetworkConfig
    config_mod = __import__(
        f"tests.swmm_validation.configs.{config_module}",
        fromlist=["make_*"],
    )
    cfg_factory = getattr(config_mod, f"make_{config_module}_config")
    pipe_config = cfg_factory()

    # Apply extra config variant if needed
    if extra_config_fn == "fixed_wse":
        from tests.swmm_validation.synthetic.site_drainage import (
            synth_site_drainage_fixed_wse,
        )
        pipe_config = synth_site_drainage_fixed_wse()

    # Build forcing
    swmm_inp_path = REPO_ROOT / "reference" / "swmm_canonical" / swmm_inp_filename

    if is_steady:
        if config_module == "site_drainage":
            inflow_node = "J1"
            inflow_cms = 0.5
        elif config_module == "detention_pond":
            inflow_node = "J1"
            inflow_cms = 3.0
        elif config_module == "culvert":
            inflow_node = "Inlet"
            inflow_cms = 2.0
        elif config_module == "inlet_drains":
            inflow_node = "Aux1"
            inflow_cms = 0.3
        else:
            inflow_node = "J1"
            inflow_cms = 0.5

        hydrology = _const_inflow(inflow_node, inflow_cms, duration_h=24.0)
        duration_s = 24.0 * 3600.0
    else:
        gauge_name = "Gage1"
        if config_module == "site_drainage":
            if forcing_type == "uniform_rain":
                duration_s = 3.0 * 3600.0
                hyeto = _uniform_hyetograph(duration_s, 20.0, dt_s=300.0)
            else:  # scs_rain
                duration_s = 24.0 * 3600.0
                hyeto = _scs_type_ii_hyetograph(duration_s, 60.0, n_points=48)
        elif config_module == "detention_pond":
            duration_s = 2.0 * 3600.0
            hyeto = _uniform_hyetograph(duration_s, 80.0, dt_s=300.0)
        elif config_module == "inlet_drains":
            duration_s = 24.0 * 3600.0
            hyeto = _scs_type_ii_hyetograph(duration_s, 50.0, n_points=48)
        elif config_module == "culvert":
            duration_s = 2.0 * 3600.0
            hyeto = _uniform_hyetograph(duration_s, 60.0, dt_s=300.0)
        else:
            duration_s = 3.0 * 3600.0
            hyeto = _uniform_hyetograph(duration_s, 20.0, dt_s=300.0)

        rainfall_ts = [(t_s / 3600.0, intensity_mm_hr) for t_s, intensity_mm_hr in hyeto]
        hydrology = {"rainfall": {gauge_name: rainfall_ts}}

    # Build bundle
    bundle = ScenarioBundle(
        name=scenario_id,
        swmm_inp_path=swmm_inp_path,
        pipe1d_config=pipe_config,
        duration_s=duration_s,
        hydrology=hydrology,
        expected_regimes=[regime],
    )

    # Build tolerance spec
    tol = ToleranceSpec(
        regimes={
            regime: {
                "depth_rmse_rel": TOLERANCES[regime].depth_rmse_rel_max,
                "flow_rmse_rel": TOLERANCES[regime].flow_rmse_rel_max,
                "depth_max_rel": TOLERANCES[regime].depth_max_rel_max,
                "flow_max_rel": TOLERANCES[regime].flow_max_rel_max,
            }
        }
    )

    # Run comparison
    result: Optional[ComparisonResult] = None
    error_msg: Optional[str] = None
    try:
        result = run_comparison(bundle, tol, workdir)
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

    if result is None:
        return None, error_msg or "run_comparison returned None"

    if result.metadata.get("error") and "SWMM run failed" in str(result.metadata["error"]):
        return None, str(result.metadata["error"])

    return result.pass_fail, regime


# --------------------------------------------------------------------------- #
# Test cases
# --------------------------------------------------------------------------- #

class TestBaselineRegression(unittest.TestCase):
    """V8: CI guard — fail if any scenario regresses from baseline."""

    def setUp(self):
        """Load baseline."""
        baseline_path = REPO_ROOT / "tests/swmm_validation/baseline.json"
        if not baseline_path.exists():
            self.skipTest(f"baseline.json not found at {baseline_path}")
        self.baseline = json.loads(baseline_path.read_text())

    def _run_scenario(self, scenario_name: str) -> Tuple[Optional[dict], Optional[str]]:
        """Re-run a scenario and return (pass_fail_dict, regime)."""
        # Find scenario definition
        scenario_def = None
        for s in SCENARIOS:
            if s[0] == scenario_name:
                scenario_def = s
                break

        if scenario_def is None:
            return None, f"Unknown scenario: {scenario_name}"

        (
            scenario_id,
            swmm_inp_filename,
            config_module,
            regime,
            forcing_type,
            is_steady,
            extra_config_fn,
        ) = scenario_def

        workdir = REPO_ROOT / "tests/swmm_validation/runs" / scenario_id
        workdir.mkdir(parents=True, exist_ok=True)

        return _run_single_scenario(
            scenario_id=scenario_id,
            swmm_inp_filename=swmm_inp_filename,
            config_module=config_module,
            regime=regime,
            forcing_type=forcing_type,
            is_steady=is_steady,
            extra_config_fn=extra_config_fn,
            workdir=workdir,
        )

    def test_no_scenario_regresses(self):
        """For each scenario in baseline, re-run and verify pass/fail matches expected."""
        for name, expected in self.baseline["scenarios"].items():
            pass_fail, regime = self._run_scenario(name)

            # Check for SWMM unavailability
            if pass_fail is None:
                if regime is not None and "SWMM run failed" in regime:
                    self.skipTest(f"SWMM unavailable: {regime}")
                self.fail(f"Scenario '{name}' failed to run: {regime}")

            expected_regime = expected["expected_regime"]
            expected_pass = expected["expected_pass"]

            # Verify regime is present
            self.assertIn(
                expected_regime,
                pass_fail,
                f"{name}: expected regime '{expected_regime}' not in pass_fail {pass_fail}",
            )

            # Verify pass/fail
            actual_pass = pass_fail[expected_regime]
            if expected_pass and not actual_pass:
                self.fail(
                    f"REGRESSION: {name} regime '{expected_regime}' was passing, now failing"
                )
            elif not expected_pass and actual_pass:
                # This is an IMPROVEMENT, not a regression — log but don't fail
                print(f"  IMPROVEMENT: {name} regime '{expected_regime}' now passing")

    def test_all_scenarios_present(self):
        """Verify baseline contains all 10 expected scenarios."""
        expected = {
            "open_channel_steady",
            "surcharge_steady",
            "outfall_backwater_fixed_wse",
            "culvert_pressurized",
            "inlet_drains_steady",
            "uniform_hyetograph_open_channel",
            "scs_type_ii_to_surcharge",
            "uniform_hyetograph_surcharge",
            "inlet_drains_dynamic",
            "culvert_surcharge_dynamic",
        }
        actual = set(self.baseline["scenarios"].keys())
        self.assertEqual(
            expected,
            actual,
            f"baseline missing: {expected - actual}",
        )


if __name__ == "__main__":
    unittest.main()
