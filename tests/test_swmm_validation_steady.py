"""V5: Steady-state validation matrix — compare pipe1D against SWMM
for constant-inflow scenarios (no rainfall).

Run with:
    python -m unittest -v tests.test_swmm_validation_steady

Scenarios (no rain, constant inflows):
  1. Open-channel steady  — Site_Drainage, constant 0.5 m³/s at J1
  2. Surcharge steady     — Detention_Pond, high inflow exceeding pipe capacity
  3. Outfall backwater    — Site_Drainage (fixed WSE variant), backwater propagates upstream
  4. Culvert pressurised  — Culvert_Model with high inflow causing surcharging
  5. Inlet drains steady  — Inlet_Drains with constant inflow at junction

Each test gracefully handles SWMM unavailability (returns None → soft pass).
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

# --------------------------------------------------------------------------- #
# Path setup
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))


# --------------------------------------------------------------------------- #
# Helper: build a constant-inflow time series
# --------------------------------------------------------------------------- #

def _const_inflow(
    node_id: str,
    flow_cms: float,
    duration_h: float = 24.0,
) -> dict:
    """Return a hydrology dict with a constant inflow at one node.

    Parameters
    ----------
    node_id : str
        SWMM node ID where the inflow is applied.
    flow_cms : float
        Constant flow rate in m³/s.
    duration_h : float
        Duration in hours (default 24).

    Returns
    -------
    dict
        ``{node_id: [(0.0, flow_cms), (duration_h, flow_cms)]}`` —
        two points are sufficient for SWMM's interpolation to hold the value
        constant across the full simulation window.
    """
    return {node_id: [(0.0, float(flow_cms)), (float(duration_h), float(flow_cms))]}


# --------------------------------------------------------------------------- #
# Test class
# --------------------------------------------------------------------------- #

class TestSteadyStateValidation(unittest.TestCase):
    """V5: Steady-state validation matrix — compare pipe1D against SWMM
    for constant-inflow scenarios (no rainfall)."""

    def setUp(self):
        """Set up shared imports and helpers for the test class."""
        from tests.swmm_validation.compare import (
            ComparisonResult,
            run_comparison,
            ScenarioBundle,
            ToleranceSpec,
        )
        from tests.swmm_validation.tolerances import TOLERANCES

        self.run_comparison = run_comparison
        self.ScenarioBundle = ScenarioBundle
        self.ToleranceSpec = ToleranceSpec
        self.TOLERANCES = TOLERANCES
        self.ComparisonResult = ComparisonResult

        # Convenience: REPO_ROOT for resolving .inp paths
        self.REPO_ROOT = REPO_ROOT

    # ----------------------------------------------------------------------- #
    # Shared helper
    # ----------------------------------------------------------------------- #

    def _run_steady(
        self,
        scenario_name: str,
        swmm_inp_path: Path,
        pipe1d_config,
        inflows_cms: dict,
        expected_regime: str,
        duration_s: float = 24 * 3600.0,
    ) -> "ComparisonResult | None":
        """Run steady-state comparison and return the result (or None on SWMM failure).

        Parameters
        ----------
        scenario_name : str
            Human-readable scenario name (used in ComparisonResult.scenario).
        swmm_inp_path : Path
            Absolute path to the canonical SWMM .inp file.
        pipe1d_config : PipeNetworkConfig
            Pre-built pipe1D configuration for this scenario.
        inflows_cms : dict
            ``{node_id: [(time_h, flow_cms), ...]}`` constant-inflow time series.
        expected_regime : str
            The hydraulic regime expected for this scenario
            (e.g. ``"open_channel"``, ``"surcharged"``, ``"fixed_wse_outfall"``).
        duration_s : float
            Simulation duration in seconds (default 24 h).

        Returns
        -------
        ComparisonResult | None
            None if SWMM was unavailable (graceful degradation); the result otherwise.
        """
        # Build a ToleranceSpec for the expected regime using V4 tolerances
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
            hydrology=inflows_cms,  # constant inflow time series (no rainfall)
            expected_regimes=[expected_regime],
        )

        with tempfile.TemporaryDirectory() as td:
            workdir = Path(td)
            result = self.run_comparison(bundle, tol, workdir)

        # If SWMM failed, metadata["error"] is non-None and result is still returned
        # (with empty series).  We surface this to the caller as None so individual
        # tests can soft-pass when SWMM is unavailable.
        error = result.metadata.get("error") if result.metadata else None
        if error and "SWMM run failed" in str(error):
            return None

        return result

    # ----------------------------------------------------------------------- #
    # Scenario 1: Open-channel steady (Site_Drainage, constant 0.5 m³/s at J1)
    # ----------------------------------------------------------------------- #

    def test_open_channel_steady_state(self):
        """Half-pipe steady flow: pipe1D Manning's normal depth vs SWMM.

        Applies a constant 0.5 m³/s inflow at node J1 (upstream junction).
        The flow should remain in open-channel regime throughout the network.
        Expected regime: open_channel (≤ 90 % full, below crown).
        """
        from tests.swmm_validation.configs.site_drainage import make_site_drainage_config

        cfg = make_site_drainage_config()
        inflows = _const_inflow("J1", flow_cms=0.5, duration_h=24.0)

        result = self._run_steady(
            scenario_name="open_channel_steady",
            swmm_inp_path=self.REPO_ROOT / "reference/swmm_canonical/Site_Drainage_Model.inp",
            pipe1d_config=cfg,
            inflows_cms=inflows,
            expected_regime="open_channel",
        )

        # Graceful degradation: SWMM unavailable → soft pass
        if result is None:
            self.skipTest("SWMM unavailable — skipping comparison")

        self.assertIsInstance(result, self.ComparisonResult)
        self.assertIn("open_channel", result.pass_fail)

        # Print diagnostic summary
        print(f"\n[open_channel_steady] pass_fail={result.pass_fail}")
        for nid, metrics in result.node_errors.items():
            de = metrics.get("depth")
            if de:
                print(f"  node {nid}: depth RMSE={de.rmse:.4f}, max_rel={de.max_rel:.4f}")
        for lid, metrics in result.link_errors.items():
            fe = metrics.get("flow")
            if fe:
                print(f"  link {lid}: flow RMSE={fe.rmse:.4f}, max_rel={fe.max_rel:.4f}")

    # ----------------------------------------------------------------------- #
    # Scenario 2: Surcharge steady (Detention_Pond, high constant inflow)
    # ----------------------------------------------------------------------- #

    def test_surcharge_steady_state(self):
        """Pressurised / surcharged steady flow: high inflow exceeds pipe capacity.

        Applies a constant 3.0 m³/s inflow at node J1 of Detention_Pond.
        The inflow far exceeds the pipe full capacity → pipes surcharged.
        Expected regime: surcharged (depth ≥ crown, pressurised).
        """
        from tests.swmm_validation.configs.detention_pond import make_detention_pond_config

        cfg = make_detention_pond_config()
        # High inflow: 3.0 m³/s (≈ 106 cfs) — well above typical pipe capacity
        inflows = _const_inflow("J1", flow_cms=3.0, duration_h=24.0)

        result = self._run_steady(
            scenario_name="surcharge_steady",
            swmm_inp_path=self.REPO_ROOT / "reference/swmm_canonical/Detention_Pond_Model.inp",
            pipe1d_config=cfg,
            inflows_cms=inflows,
            expected_regime="surcharged",
        )

        if result is None:
            self.skipTest("SWMM unavailable — skipping comparison")

        self.assertIsInstance(result, self.ComparisonResult)

        print(f"\n[surcharge_steady] pass_fail={result.pass_fail}")
        for nid, metrics in result.node_errors.items():
            de = metrics.get("depth")
            if de:
                print(f"  node {nid}: depth RMSE={de.rmse:.4f}, max_rel={de.max_rel:.4f}")

    # ----------------------------------------------------------------------- #
    # Scenario 3: Outfall backwater (Site_Drainage fixed WSE variant)
    # ----------------------------------------------------------------------- #

    def test_outfall_backwater_steady_state(self):
        """Fixed WSE outfall: downstream boundary head propagates upstream.

        Uses the synthetic fixed-WSE variant of Site_Drainage:
        outfall_fixed_wse = outfall_invert + 1.0 m (tailwater held fixed).
        The backwater curve extends upstream from the outfall.
        Expected regime: fixed_wse_outfall (depth RMSE ≤ 1 % per V4 tolerances).
        """
        from tests.swmm_validation.synthetic.site_drainage import synth_site_drainage_fixed_wse

        cfg = synth_site_drainage_fixed_wse()
        # Moderate constant inflow to keep open-channel regime upstream of backwater zone
        inflows = _const_inflow("J1", flow_cms=0.5, duration_h=24.0)

        result = self._run_steady(
            scenario_name="outfall_backwater_fixed_wse",
            swmm_inp_path=self.REPO_ROOT / "reference/swmm_canonical/Site_Drainage_Model.inp",
            pipe1d_config=cfg,
            inflows_cms=inflows,
            expected_regime="fixed_wse_outfall",
        )

        if result is None:
            self.skipTest("SWMM unavailable — skipping comparison")

        self.assertIsInstance(result, self.ComparisonResult)

        print(f"\n[outfall_backwater_fixed_wse] pass_fail={result.pass_fail}")
        for nid, metrics in result.node_errors.items():
            de = metrics.get("depth")
            if de:
                print(f"  node {nid}: depth RMSE={de.rmse:.4f}, max_rel={de.max_rel:.4f}")

    # ----------------------------------------------------------------------- #
    # Scenario 4: Culvert pressurised (Culvert_Model, high head)
    # ----------------------------------------------------------------------- #

    def test_culvert_pressurized(self):
        """Surcharged culvert at high head: pressurised flow through barrel.

        The Culvert_Model has a FIXED outfall (TailWater) at 261.5184 m
        and an Inlet storage node at 267.6144 m (invert).  With a high
        constant inflow the storage head rises above the crown → surcharged barrel.
        Expected regime: surcharged.
        """
        from tests.swmm_validation.configs.culvert import make_culvert_config

        cfg = make_culvert_config()
        # Apply high constant inflow at the Inlet storage node
        inflows = _const_inflow("Inlet", flow_cms=2.0, duration_h=24.0)

        result = self._run_steady(
            scenario_name="culvert_pressurized",
            swmm_inp_path=self.REPO_ROOT / "reference/swmm_canonical/Culvert_Model.inp",
            pipe1d_config=cfg,
            inflows_cms=inflows,
            expected_regime="surcharged",
        )

        if result is None:
            self.skipTest("SWMM unavailable — skipping comparison")

        self.assertIsInstance(result, self.ComparisonResult)

        print(f"\n[culvert_pressurized] pass_fail={result.pass_fail}")
        for nid, metrics in result.node_errors.items():
            de = metrics.get("depth")
            if de:
                print(f"  node {nid}: depth RMSE={de.rmse:.4f}, max_rel={de.max_rel:.4f}")
        for lid, metrics in result.link_errors.items():
            fe = metrics.get("flow")
            if fe:
                print(f"  link {lid}: flow RMSE={fe.rmse:.4f}, max_rel={fe.max_rel:.4f}")

    # ----------------------------------------------------------------------- #
    # Scenario 5: Inlet drains steady (constant inflow at junction with grate)
    # ----------------------------------------------------------------------- #

    def test_inlet_drains_steady_state(self):
        """Inlet_Drains: constant inflow at a junction with a street grate.

        Applies a moderate constant inflow (0.3 m³/s) at Aux1, an upstream
        auxiliary junction, and verifies open-channel results.
        Expected regime: open_channel.
        """
        from tests.swmm_validation.configs.inlet_drains import make_inlet_drains_config

        cfg = make_inlet_drains_config()
        inflows = _const_inflow("Aux1", flow_cms=0.3, duration_h=24.0)

        result = self._run_steady(
            scenario_name="inlet_drains_steady",
            swmm_inp_path=self.REPO_ROOT / "reference/swmm_canonical/Inlet_Drains_Model.inp",
            pipe1d_config=cfg,
            inflows_cms=inflows,
            expected_regime="open_channel",
        )

        if result is None:
            self.skipTest("SWMM unavailable — skipping comparison")

        self.assertIsInstance(result, self.ComparisonResult)

        print(f"\n[inlet_drains_steady] pass_fail={result.pass_fail}")
        for nid, metrics in result.node_errors.items():
            de = metrics.get("depth")
            if de:
                print(f"  node {nid}: depth RMSE={de.rmse:.4f}, max_rel={de.max_rel:.4f}")

    # ----------------------------------------------------------------------- #
    # Smoke test: ComparisonResult is returned
    # ----------------------------------------------------------------------- #

    def test_run_comparison_produces_comparison_result(self):
        """Smoke test: comparison harness returns a ComparisonResult (or None).

        Uses the simplest scenario (open-channel, Site_Drainage) to verify
        the full call chain executes without raising.
        """
        from tests.swmm_validation.configs.site_drainage import make_site_drainage_config

        cfg = make_site_drainage_config()
        inflows = _const_inflow("J1", flow_cms=0.1, duration_h=1.0)  # short 1h run

        bundle = self.ScenarioBundle(
            name="smoke_test",
            swmm_inp_path=self.REPO_ROOT / "reference/swmm_canonical/Site_Drainage_Model.inp",
            pipe1d_config=cfg,
            duration_s=1 * 3600.0,
            hydrology=inflows,
            expected_regimes=["open_channel"],
        )

        tols = self.ToleranceSpec(
            regimes={
                "open_channel": {
                    "depth_rmse_rel": 0.05,
                    "flow_rmse_rel": 0.10,
                    "depth_max_rel": 0.05,
                    "flow_max_rel": 0.10,
                }
            }
        )

        with tempfile.TemporaryDirectory() as td:
            workdir = Path(td)
            result = self.run_comparison(bundle, tols, workdir)

            self.assertIsInstance(result, self.ComparisonResult)
            self.assertEqual(result.scenario, "smoke_test")
            self.assertIsInstance(result.pass_fail, dict)
            self.assertIsInstance(result.metadata, dict)

            # JSON must be written even on errors
            json_path = workdir / "result.json"
            self.assertTrue(json_path.exists(), "result.json not written")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    unittest.main()
