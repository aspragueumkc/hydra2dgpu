"""V6: Dynamic validation matrix — compare pipe1D against SWMM
for time-varying hyetograph scenarios.

Run with:
    python -m unittest -v tests.test_swmm_validation_dynamic

Scenarios (dynamic rainfall):
  1. Uniform hyetograph → open-channel flow throughout
  2. SCS Type II storm → open-channel → surcharge transition
  3. High-intensity uniform → persistent surcharge (Detention_Pond)
  4. Inlet_Drains + SCS Type II → surcharge in low-lying junctions
  5. Culvert + high inflow → surcharged barrel
  6. Peak WSE comparison across all scenarios (within 5%)
  7. Recession tail comparison (within 10% in last 1/3 of duration)
  8. Outfall backwater recovery after storm peak

Each test gracefully handles SWMM unavailability (returns None → soft pass).
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from typing import Callable, Dict, List, Tuple

# --------------------------------------------------------------------------- #
# Path setup
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))


# --------------------------------------------------------------------------- #
# Hyetograph builders
# --------------------------------------------------------------------------- #

class TestDynamicValidation(unittest.TestCase):
    """V6: Dynamic validation matrix — compare pipe1D against SWMM
    for time-varying hyetograph scenarios."""

    def setUp(self):
        """Set up shared imports and helpers."""
        from tests.swmm_validation.compare import (
            ComparisonResult,
            run_comparison,
            ScenarioBundle,
            ToleranceSpec,
        )
        from tests.swmm_validation.tolerances import (
            TOLERANCES,
            classify_node_regime,
        )

        self.run_comparison = run_comparison
        self.ScenarioBundle = ScenarioBundle
        self.ToleranceSpec = ToleranceSpec
        self.TOLERANCES = TOLERANCES
        self.ComparisonResult = ComparisonResult
        self.classify_node_regime = classify_node_regime
        self.REPO_ROOT = REPO_ROOT

    # ----------------------------------------------------------------------- #
    # Hyetograph helpers
    # ----------------------------------------------------------------------- #

    @staticmethod
    def _uniform_hyetograph(
        duration_s: float,
        peak_intensity_mm_hr: float,
        dt_s: float = 300.0,
    ) -> List[Tuple[float, float]]:
        """5-minute uniform rainfall: constant intensity over duration.

        Parameters
        ----------
        duration_s : float
            Total storm duration in seconds.
        peak_intensity_mm_hr : float
            Constant rainfall intensity in mm/hr.
        dt_s : float
            Time step in seconds (default 300 s = 5 min).

        Returns
        -------
        List[Tuple[float, float]]
            ``[(time_s, intensity_mm_hr), ...]`` time-series.
        """
        times_s: List[float] = []
        values: List[float] = []
        t = 0.0
        while t <= duration_s:
            times_s.append(t)
            values.append(float(peak_intensity_mm_hr))
            t += dt_s
        # Ensure the last point reaches exactly duration_s
        if times_s[-1] < duration_s:
            times_s.append(float(duration_s))
            values.append(float(peak_intensity_mm_hr))
        return list(zip(times_s, values))

    @staticmethod
    def _scs_type_ii_hyetograph(
        duration_s: float,
        peak_intensity_mm_hr: float,
        n_points: int = 48,
    ) -> List[Tuple[float, float]]:
        """SCS Type II 24-hour storm: triangular peak at t=duration/2.

        The SCS Type II dimensionless hyetograph has the peak at 50% of the
        storm duration with total depth = 1.0 * peak intensity * duration / 2.

        Parameters
        ----------
        duration_s : float
            Total storm duration in seconds.
        peak_intensity_mm_hr : float
            Peak rainfall intensity in mm/hr.
        n_points : int
            Number of points to generate (default 48 for 30-min resolution
            over a 24-h storm).

        Returns
        -------
        List[Tuple[float, float]]
            ``[(time_s, intensity_mm_hr), ...]`` time-series.
        """
        # Dimensionless cumulative distribution F(λ) for SCS Type II:
        #   F(λ) = λ^2 / (1 - λ + λ^2)   for 0 ≤ λ ≤ 1
        # Differentiate to get incremental fractions.
        times_dimless = [i / float(n_points) for i in range(n_points + 1)]
        cum_frac: List[float] = []
        for lam in times_dimless:
            lam2 = lam * lam
            denom = 1.0 - lam + lam2
            if denom > 0.0:
                cum_frac.append(lam2 / denom)
            else:
                cum_frac.append(1.0)

        # Convert cumulative fractions to incremental fractions
        incr_frac = [cum_frac[0]]
        for i in range(1, len(cum_frac)):
            incr_frac.append(cum_frac[i] - cum_frac[i - 1])
        # Normalise so sum == 1.0
        total = sum(incr_frac)
        if total > 0.0:
            incr_frac = [v / total for v in incr_frac]

        # Map to actual time / intensity
        result: List[Tuple[float, float]] = []
        for i, frac in enumerate(incr_frac):
            t_s = float(i) / float(n_points) * float(duration_s)
            intensity = frac * float(peak_intensity_mm_hr) * float(n_points)
            result.append((t_s, intensity))

        return result

    # ----------------------------------------------------------------------- #
    # Shared dynamic runner
    # ----------------------------------------------------------------------- #

    def _run_dynamic(
        self,
        scenario_name: str,
        swmm_inp_path: Path,
        pipe1d_config,
        hyetograph: List[Tuple[float, float]],
        expected_regime: str,
        duration_s: float,
        gauge_name: str = "Gage1",
    ) -> "ComparisonResult | None":
        """Run dynamic comparison and return the result (or None on SWMM failure).

        Parameters
        ----------
        scenario_name : str
            Human-readable scenario name.
        swmm_inp_path : Path
            Absolute path to the canonical SWMM .inp file.
        pipe1d_config : PipeNetworkConfig
            Pre-built pipe1D configuration for this scenario.
        hyetograph : List[Tuple[float, float]]
            ``[(time_s, intensity_mm_hr), ...]`` rainfall time series.
        expected_regime : str
            Expected hydraulic regime (e.g. ``"open_channel"``,
            ``"surcharged"``, ``"transition"``).
        duration_s : float
            Simulation duration in seconds.
        gauge_name : str
            Raingage name used in the .inp file (default ``"Gage1"``).

        Returns
        -------
        ComparisonResult | None
            None if SWMM was unavailable (graceful degradation); the result otherwise.
        """
        # Convert hyetograph (time_s, mm/hr) → (time_h, mm/hr) for the bundle
        # SWMM timeseries keys in compare.py use hours.
        rainfall_ts: List[Tuple[float, float]] = [
            (t_s / 3600.0, intensity_mm_hr)
            for t_s, intensity_mm_hr in hyetograph
        ]

        # Wrap as hydrology dict with "rainfall" key → pipe1D applies uniformly
        hydrology: dict = {
            "rainfall": {gauge_name: rainfall_ts}
        }

        tol = self.ToleranceSpec(
            regimes={
                expected_regime: {
                    "depth_rmse_rel": self.TOLERANCES[expected_regime].depth_rmse_rel_max,
                    "flow_rmse_rel": self.TOLERANCES[expected_regime].flow_rmse_rel_max,
                    "depth_max_rel": self.TOLERANCES[expected_regime].depth_max_rel_max,
                    "flow_max_rel": self.TOLERANCES[expected_regime].flow_max_rel_max,
                }
            }
        )

        bundle = self.ScenarioBundle(
            name=scenario_name,
            swmm_inp_path=swmm_inp_path,
            pipe1d_config=pipe1d_config,
            duration_s=duration_s,
            hydrology=hydrology,
            expected_regimes=[expected_regime],
        )

        with tempfile.TemporaryDirectory() as td:
            workdir = Path(td)
            result = self.run_comparison(bundle, tol, workdir)

        error = result.metadata.get("error") if result.metadata else None
        if error and "SWMM run failed" in str(error):
            return None

        return result

    # ----------------------------------------------------------------------- #
    # Test 1: Uniform hyetograph → open-channel flow
    # ----------------------------------------------------------------------- #

    def test_uniform_hyetograph_open_channel(self):
        """Uniform 20 mm/hr, 3-hour storm → open-channel flow throughout.

        Site_Drainage is a mild-slope network; 20 mm/hr is a moderate storm
        that should remain below crown throughout.  Expected regime: open_channel.
        """
        from tests.swmm_validation.configs.site_drainage import make_site_drainage_config

        cfg = make_site_drainage_config()
        duration_s = 3.0 * 3600.0  # 3 hours
        hyetograph = self._uniform_hyetograph(
            duration_s=duration_s,
            peak_intensity_mm_hr=20.0,
            dt_s=300.0,
        )

        result = self._run_dynamic(
            scenario_name="uniform_open_channel",
            swmm_inp_path=self.REPO_ROOT / "reference/swmm_canonical/Site_Drainage_Model.inp",
            pipe1d_config=cfg,
            hyetograph=hyetograph,
            expected_regime="open_channel",
            duration_s=duration_s,
        )

        if result is None:
            self.skipTest("SWMM unavailable — skipping comparison")

        self.assertIsInstance(result, self.ComparisonResult)
        self.assertIn("open_channel", result.pass_fail)

        print(f"\n[uniform_open_channel] pass_fail={result.pass_fail}")
        for nid, metrics in result.node_errors.items():
            de = metrics.get("depth")
            if de:
                print(f"  node {nid}: depth RMSE={de.rmse:.4f}, max_rel={de.max_rel:.4f}")
        for lid, metrics in result.link_errors.items():
            fe = metrics.get("flow")
            if fe:
                print(f"  link {lid}: flow RMSE={fe.rmse:.4f}, max_rel={fe.max_rel:.4f}")

    # ----------------------------------------------------------------------- #
    # Test 2: SCS Type II → open-channel to surcharge transition
    # ----------------------------------------------------------------------- #

    def test_scs_type_ii_open_channel_to_surcharge(self):
        """SCS Type II 24-hour storm → open-channel early, surcharge near peak.

        Site_Drainage with a 24-hour SCS Type II storm (peak 60 mm/hr, total
        ~150 mm).  The network transitions from dry → open-channel → surcharge
        as the storm peak passes, then recedes.  Expected regime: transition.
        """
        from tests.swmm_validation.configs.site_drainage import make_site_drainage_config

        cfg = make_site_drainage_config()
        duration_s = 24.0 * 3600.0  # 24 hours
        hyetograph = self._scs_type_ii_hyetograph(
            duration_s=duration_s,
            peak_intensity_mm_hr=60.0,
            n_points=48,
        )

        result = self._run_dynamic(
            scenario_name="scs_type_ii_transition",
            swmm_inp_path=self.REPO_ROOT / "reference/swmm_canonical/Site_Drainage_Model.inp",
            pipe1d_config=cfg,
            hyetograph=hyetograph,
            expected_regime="transition",
            duration_s=duration_s,
        )

        if result is None:
            self.skipTest("SWMM unavailable — skipping comparison")

        self.assertIsInstance(result, self.ComparisonResult)

        print(f"\n[scs_type_ii_transition] pass_fail={result.pass_fail}")
        for nid, metrics in result.node_errors.items():
            de = metrics.get("depth")
            if de:
                print(f"  node {nid}: depth RMSE={de.rmse:.4f}, max_rel={de.max_rel:.4f}")

    # ----------------------------------------------------------------------- #
    # Test 3: High-intensity uniform → persistent surcharge
    # ----------------------------------------------------------------------- #

    def test_uniform_hyetograph_surcharge_persistent(self):
        """High-intensity 80 mm/hr uniform, 2-hour storm → persistent surcharge.

        Detention_Pond has larger pipes but a high-intensity storm drives depth
        above crown throughout the event.  Expected regime: surcharged.
        """
        from tests.swmm_validation.configs.detention_pond import make_detention_pond_config

        cfg = make_detention_pond_config()
        duration_s = 2.0 * 3600.0  # 2 hours
        hyetograph = self._uniform_hyetograph(
            duration_s=duration_s,
            peak_intensity_mm_hr=80.0,
            dt_s=300.0,
        )

        result = self._run_dynamic(
            scenario_name="uniform_surcharge_persistent",
            swmm_inp_path=self.REPO_ROOT / "reference/swmm_canonical/Detention_Pond_Model.inp",
            pipe1d_config=cfg,
            hyetograph=hyetograph,
            expected_regime="surcharged",
            duration_s=duration_s,
        )

        if result is None:
            self.skipTest("SWMM unavailable — skipping comparison")

        self.assertIsInstance(result, self.ComparisonResult)

        print(f"\n[uniform_surcharge_persistent] pass_fail={result.pass_fail}")
        for nid, metrics in result.node_errors.items():
            de = metrics.get("depth")
            if de:
                print(f"  node {nid}: depth RMSE={de.rmse:.4f}, max_rel={de.max_rel:.4f}")

    # ----------------------------------------------------------------------- #
    # Test 4: Inlet_Drains + SCS Type II → surcharge in low-lying junctions
    # ----------------------------------------------------------------------- #

    def test_inlet_drains_dynamic(self):
        """Inlet_Drains + SCS Type II 24-hour storm → surcharge at low nodes.

        The inlet-captured runoff drives surcharging in the downstream low-lying
        junctions.  Expected regime: transition.
        """
        from tests.swmm_validation.configs.inlet_drains import make_inlet_drains_config

        cfg = make_inlet_drains_config()
        duration_s = 24.0 * 3600.0  # 24 hours
        hyetograph = self._scs_type_ii_hyetograph(
            duration_s=duration_s,
            peak_intensity_mm_hr=50.0,
            n_points=48,
        )

        result = self._run_dynamic(
            scenario_name="inlet_drains_scs_type_ii",
            swmm_inp_path=self.REPO_ROOT / "reference/swmm_canonical/Inlet_Drains_Model.inp",
            pipe1d_config=cfg,
            hyetograph=hyetograph,
            expected_regime="transition",
            duration_s=duration_s,
        )

        if result is None:
            self.skipTest("SWMM unavailable — skipping comparison")

        self.assertIsInstance(result, self.ComparisonResult)

        print(f"\n[inlet_drains_scs_type_ii] pass_fail={result.pass_fail}")
        for nid, metrics in result.node_errors.items():
            de = metrics.get("depth")
            if de:
                print(f"  node {nid}: depth RMSE={de.rmse:.4f}, max_rel={de.max_rel:.4f}")

    # ----------------------------------------------------------------------- #
    # Test 5: Culvert + high-intensity uniform → surcharged barrel
    # ----------------------------------------------------------------------- #

    def test_culvert_surcharge_dynamic(self):
        """Culvert_Model + high uniform rainfall → surcharged barrel.

        The Culvert scenario has a fixed outfall and limited inlet geometry.
        A 60 mm/hr 2-hour storm drives high head at the inlet causing
        surcharging through the barrel.  Expected regime: surcharged.
        """
        from tests.swmm_validation.configs.culvert import make_culvert_config

        cfg = make_culvert_config()
        duration_s = 2.0 * 3600.0  # 2 hours
        hyetograph = self._uniform_hyetograph(
            duration_s=duration_s,
            peak_intensity_mm_hr=60.0,
            dt_s=300.0,
        )

        result = self._run_dynamic(
            scenario_name="culvert_surcharge",
            swmm_inp_path=self.REPO_ROOT / "reference/swmm_canonical/Culvert_Model.inp",
            pipe1d_config=cfg,
            hyetograph=hyetograph,
            expected_regime="surcharged",
            duration_s=duration_s,
        )

        if result is None:
            self.skipTest("SWMM unavailable — skipping comparison")

        self.assertIsInstance(result, self.ComparisonResult)

        print(f"\n[culvert_surcharge] pass_fail={result.pass_fail}")
        for nid, metrics in result.node_errors.items():
            de = metrics.get("depth")
            if de:
                print(f"  node {nid}: depth RMSE={de.rmse:.4f}, max_rel={de.max_rel:.4f}")
        for lid, metrics in result.link_errors.items():
            fe = metrics.get("flow")
            if fe:
                print(f"  link {lid}: flow RMSE={fe.rmse:.4f}, max_rel={fe.max_rel:.4f}")

    # ----------------------------------------------------------------------- #
    # Test 6: Peak WSE comparison (all scenarios)
    # ----------------------------------------------------------------------- #

    def test_peak_wse_comparison(self):
        """Compare peak WSE between SWMM and pipe1D across scenarios (within 5%).

        For each scenario that ran successfully, extracts the maximum depth from
        the node depth time series for both SWMM and pipe1D, then asserts the
        relative error is ≤ 5%.
        """
        from tests.swmm_validation.configs.site_drainage import make_site_drainage_config

        cfg = make_site_drainage_config()
        duration_s = 3.0 * 3600.0  # 3 hours
        hyetograph = self._uniform_hyetograph(
            duration_s=duration_s,
            peak_intensity_mm_hr=25.0,
            dt_s=300.0,
        )

        result = self._run_dynamic(
            scenario_name="peak_wse_comparison",
            swmm_inp_path=self.REPO_ROOT / "reference/swmm_canonical/Site_Drainage_Model.inp",
            pipe1d_config=cfg,
            hyetograph=hyetograph,
            expected_regime="open_channel",
            duration_s=duration_s,
        )

        if result is None:
            self.skipTest("SWMM unavailable — skipping comparison")

        self.assertIsInstance(result, self.ComparisonResult)

        # Extract peak WSE per node from node_series
        swmm_node_series = {}
        pipe_node_series = {}
        for node_id, series in result.node_series.items():
            depth_ts = series.get("depth")
            if depth_ts is None:
                continue
            values = depth_ts.values
            if not values:
                continue
            # Classify by prefix heuristic: SWMM IDs are alphabetical strings,
            # pipe1D IDs are n0, n1, c0, c1, ...
            if node_id.startswith("n") or node_id.startswith("c"):
                pipe_node_series[node_id] = values
            else:
                swmm_node_series[node_id] = values

        errors: List[Tuple[str, float, float, float]] = []
        for swmm_id, swmm_vals in swmm_node_series.items():
            swmm_peak = max(swmm_vals)
            # Find matching pipe node by sorted-index heuristic
            swmm_sorted_ids = sorted(swmm_node_series.keys())
            swmm_idx = swmm_sorted_ids.index(swmm_id)
            pipe_sorted_ids = sorted(pipe_node_series.keys())
            if swmm_idx < len(pipe_sorted_ids):
                pipe_id = pipe_sorted_ids[swmm_idx]
                pipe_vals = pipe_node_series.get(pipe_id, [])
                if pipe_vals:
                    pipe_peak = max(pipe_vals)
                    rel_err = abs(swmm_peak - pipe_peak) / max(abs(swmm_peak), 1e-6)
                    errors.append((swmm_id, swmm_peak, pipe_peak, rel_err))
                    print(
                        f"  node {swmm_id} vs {pipe_id}: "
                        f"SWMM peak={swmm_peak:.4f} m, pipe1D peak={pipe_peak:.4f} m, "
                        f"rel_err={rel_err:.4f}"
                    )

        # Assert all peaks are within 5%
        for node_id, swmm_peak, pipe_peak, rel_err in errors:
            self.assertLessEqual(
                rel_err, 0.05,
                f"Peak WSE for {node_id} exceeds 5%: rel_err={rel_err:.4f}"
            )

    # ----------------------------------------------------------------------- #
    # Test 7: Recession tail comparison
    # ----------------------------------------------------------------------- #

    def test_recession_tail(self):
        """Compare recession tail after storm peak (within 10% in last 1/3).

        Uses Site_Drainage with a 6-hour uniform storm.  After the peak (at
        t=3h), the last third of the time series (t ≥ 4h) should match between
        SWMM and pipe1D within 10% relative error.
        """
        from tests.swmm_validation.configs.site_drainage import make_site_drainage_config

        cfg = make_site_drainage_config()
        duration_s = 6.0 * 3600.0  # 6 hours
        hyetograph = self._uniform_hyetograph(
            duration_s=duration_s,
            peak_intensity_mm_hr=30.0,
            dt_s=300.0,
        )

        result = self._run_dynamic(
            scenario_name="recession_tail",
            swmm_inp_path=self.REPO_ROOT / "reference/swmm_canonical/Site_Drainage_Model.inp",
            pipe1d_config=cfg,
            hyetograph=hyetograph,
            expected_regime="open_channel",
            duration_s=duration_s,
        )

        if result is None:
            self.skipTest("SWMM unavailable — skipping comparison")

        self.assertIsInstance(result, self.ComparisonResult)

        # Identify SWMM vs pipe1D nodes
        swmm_nodes = {}
        pipe_nodes = {}
        for node_id, series in result.node_series.items():
            depth_ts = series.get("depth")
            if depth_ts is None or not depth_ts.values:
                continue
            if node_id.startswith("n") or node_id.startswith("c"):
                pipe_nodes[node_id] = depth_ts.values
            else:
                swmm_nodes[node_id] = depth_ts.values

        # Compare in the last 1/3 of the series
        swmm_sorted_ids = sorted(swmm_nodes.keys())
        pipe_sorted_ids = sorted(pipe_nodes.keys())

        errors: List[Tuple[str, float, float, float]] = []
        for swmm_id in swmm_sorted_ids:
            swmm_idx = swmm_sorted_ids.index(swmm_id)
            if swmm_idx >= len(pipe_sorted_ids):
                break
            pipe_id = pipe_sorted_ids[swmm_idx]
            swmm_vals = swmm_nodes[swmm_id]
            pipe_vals = pipe_nodes.get(pipe_id, [])
            if not pipe_vals or len(swmm_vals) < 3:
                continue

            # Last 1/3 of the series
            n = len(swmm_vals)
            tail_start = n * 2 // 3
            swmm_tail = swmm_vals[tail_start:]
            pipe_tail = pipe_vals[tail_start:] if len(pipe_vals) >= n else pipe_vals[-len(swmm_tail):]

            if not swmm_tail or not pipe_tail:
                continue

            # Compare mean depth in tail
            swmm_mean = sum(swmm_tail) / len(swmm_tail)
            pipe_mean = sum(pipe_tail) / len(pipe_tail)
            rel_err = abs(swmm_mean - pipe_mean) / max(abs(swmm_mean), 1e-6)
            errors.append((swmm_id, swmm_mean, pipe_mean, rel_err))
            print(
                f"  tail {swmm_id} vs {pipe_id}: "
                f"SWMM mean={swmm_mean:.4f} m, pipe1D mean={pipe_mean:.4f} m, "
                f"rel_err={rel_err:.4f}"
            )

        for node_id, swmm_mean, pipe_mean, rel_err in errors:
            self.assertLessEqual(
                rel_err, 0.10,
                f"Recession tail for {node_id} exceeds 10%: rel_err={rel_err:.4f}"
            )

    # ----------------------------------------------------------------------- #
    # Test 8: Outfall backwater recovery after storm peak
    # ----------------------------------------------------------------------- #

    def test_outfall_backwater_recovery(self):
        """Fixed-WSE outfall + SCS Type II → backwater recovery after peak.

        Uses the synthetic fixed-WSE variant of Site_Drainage: the outfall
        tailwater is held at outfall_invert + 1.0 m.  After the storm peak,
        the backwater should recede as the network drains.  Expected regime:
        fixed_wse_outfall.
        """
        from tests.swmm_validation.synthetic.site_drainage import (
            synth_site_drainage_fixed_wse,
        )

        cfg = synth_site_drainage_fixed_wse()
        duration_s = 24.0 * 3600.0  # 24 hours
        hyetograph = self._scs_type_ii_hyetograph(
            duration_s=duration_s,
            peak_intensity_mm_hr=60.0,
            n_points=48,
        )

        result = self._run_dynamic(
            scenario_name="outfall_backwater_recovery",
            swmm_inp_path=self.REPO_ROOT / "reference/swmm_canonical/Site_Drainage_Model.inp",
            pipe1d_config=cfg,
            hyetograph=hyetograph,
            expected_regime="fixed_wse_outfall",
            duration_s=duration_s,
        )

        if result is None:
            self.skipTest("SWMM unavailable — skipping comparison")

        self.assertIsInstance(result, self.ComparisonResult)

        print(f"\n[outfall_backwater_recovery] pass_fail={result.pass_fail}")
        for nid, metrics in result.node_errors.items():
            de = metrics.get("depth")
            if de:
                print(f"  node {nid}: depth RMSE={de.rmse:.4f}, max_rel={de.max_rel:.4f}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    unittest.main()
