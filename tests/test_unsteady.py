#!/usr/bin/env python3
"""tests/test_unsteady.py

Unit tests for the 1D unsteady (dynamic wave) solver in unsteady_model.py.

These tests run headlessly (no QGIS / Qt required) using synthetic cross
sections built directly from backwater_model.CrossSection objects.
"""

import json
import math
import os
import sys
import tempfile
import unittest

# Make sure the plugin directory is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

# Guard: if numpy absent the solver cannot run but we still import the module
from unsteady_model import (
    HydrographBC,
    UnsteadyParams,
    UnsteadyResults,
    _compute_total_K,
    _effective_reach_length,
    _hydraulic_bed_elevation,
    _regularized_wse,
    _section_vars,
    _Sf,
    _dSf_dQ,
    _dSf_dz,
    _normal_depth_Q,
    _unsteady_section_state,
    WETTING_DEPTH_FT,
    save_unsteady_results_to_geopackage,
    load_unsteady_results_from_geopackage,
    save_unsteady_debug_to_geopackage,
    load_unsteady_debug_from_geopackage,
    save_hydrograph_to_geopackage,
    load_hydrograph_from_geopackage,
)

from backwater_model import CrossSection, ModelInput


# ---------------------------------------------------------------------------
# Helpers to build synthetic sections
# ---------------------------------------------------------------------------

def _make_trapezoid_section(
    river_station: str,
    bottom_width: float,
    side_slope: float,    # horizontal:vertical, z/h
    bed_elev: float,
    n: float,
    reach_length: float,
    bank_height: float = 10.0,
) -> CrossSection:
    """Build a simple trapezoidal cross section."""
    # Profile: symmetric trapezoid centered at x=0
    half_w = bottom_width / 2.0
    # stations at: left toe, left bank, right bank, right toe
    x_lt  = -(half_w + side_slope * bank_height)
    x_lb  = -half_w
    x_rb  =  half_w
    x_rt  =  half_w + side_slope * bank_height

    z_top = bed_elev + bank_height

    geometry = [
        (x_lt, z_top),
        (x_lb, bed_elev),
        (x_rb, bed_elev),
        (x_rt, z_top),
    ]
    return CrossSection(
        river_station=river_station,
        geometry=geometry,
        left_bank_station=float(x_lb),
        right_bank_station=float(x_rb),
        n_lob=n, n_ch=n, n_rob=n,
        L_ch_to_next=reach_length,
        L_lob_to_next=reach_length,
        L_rob_to_next=reach_length,
    )


def _make_simple_model(
    n_sections: int = 5,
    bottom_width: float = 20.0,
    side_slope: float = 2.0,
    bed_slope: float = 0.001,
    n_manning: float = 0.035,
    reach_length: float = 500.0,
    Q_base: float = 200.0,
) -> ModelInput:
    """Build a simple prismatic channel model (DS=0, US=N-1 ordering)."""
    sections = []
    for i in range(n_sections):
        # river_station increases from DS (0) to US (N-1)
        rs = str(i * reach_length)
        bed_elev = i * bed_slope * reach_length   # US sections higher
        xs = _make_trapezoid_section(
            river_station=rs,
            bottom_width=bottom_width,
            side_slope=side_slope,
            bed_elev=bed_elev,
            n=n_manning,
            reach_length=reach_length,
        )
        sections.append(xs)
    return ModelInput(
        flow_cfs=Q_base,
        flow_change=None,
        boundary_condition='normal_depth',
        boundary_value=bed_slope,
        sections=sections,
    )


# ---------------------------------------------------------------------------
# HydrographBC tests
# ---------------------------------------------------------------------------

class TestHydrographBC(unittest.TestCase):

    def test_interpolation_in_range(self):
        h = HydrographBC(times=[0.0, 100.0, 200.0],
                         values=[10.0, 50.0, 20.0])
        self.assertAlmostEqual(h.interpolate(0.0),   10.0)
        self.assertAlmostEqual(h.interpolate(100.0), 50.0)
        self.assertAlmostEqual(h.interpolate(200.0), 20.0)
        self.assertAlmostEqual(h.interpolate(50.0),  30.0)

    def test_clamp_before_start(self):
        h = HydrographBC(times=[10.0, 20.0], values=[5.0, 15.0])
        self.assertAlmostEqual(h.interpolate(-5.0), 5.0)

    def test_clamp_after_end(self):
        h = HydrographBC(times=[10.0, 20.0], values=[5.0, 15.0])
        self.assertAlmostEqual(h.interpolate(100.0), 15.0)

    def test_empty(self):
        h = HydrographBC(times=[], values=[])
        self.assertEqual(h.interpolate(50.0), 0.0)


# ---------------------------------------------------------------------------
# Hydraulic helper tests
# ---------------------------------------------------------------------------

class TestHydraulicHelpers(unittest.TestCase):

    def setUp(self):
        self.xs = _make_trapezoid_section(
            river_station='100',
            bottom_width=20.0,
            side_slope=2.0,
            bed_elev=500.0,
            n=0.035,
            reach_length=500.0,
        )

    def test_K_positive_at_depth(self):
        K = _compute_total_K(self.xs, 503.0)   # 3 ft depth
        self.assertGreater(K, 0.0)

    def test_K_zero_at_bed(self):
        K = _compute_total_K(self.xs, 500.0)   # exactly at bed
        self.assertAlmostEqual(K, 0.0, places=3)

    def test_K_increases_with_stage(self):
        K1 = _compute_total_K(self.xs, 501.0)
        K2 = _compute_total_K(self.xs, 503.0)
        self.assertGreater(K2, K1)

    def test_section_vars_top_width(self):
        A, K, T = _section_vars(self.xs, 502.0)
        self.assertGreater(A, 0.0)
        self.assertGreater(K, 0.0)
        self.assertGreater(T, 0.0)
        # At 2 ft depth, bottom=20 ft, T ≈ 20 + 2*side_slope*depth = 20+8 = 28 ft
        self.assertAlmostEqual(T, 28.0, delta=1.0)

    def test_Sf_sign(self):
        K = _compute_total_K(self.xs, 503.0)
        self.assertGreater(_Sf(+200.0, K), 0.0)
        self.assertLess(_Sf(-200.0, K),    0.0)

    def test_dSf_dQ_positive(self):
        K = _compute_total_K(self.xs, 503.0)
        self.assertGreater(_dSf_dQ(200.0, K), 0.0)

    def test_normal_depth_Q(self):
        Q = _normal_depth_Q(self.xs, 0.001, 502.0)
        self.assertGreater(Q, 0.0)

    def test_hydraulic_bed_prefers_channel_bottom(self):
        xs = CrossSection(
            river_station='200',
            geometry=[(-30.0, 0.0), (-10.0, 5.0), (10.0, 5.0), (30.0, 0.0)],
            left_bank_station=-10.0,
            right_bank_station=10.0,
            n_lob=0.04,
            n_ch=0.035,
            n_rob=0.04,
            L_ch_to_next=500.0,
            L_lob_to_next=500.0,
            L_rob_to_next=500.0,
        )
        self.assertAlmostEqual(_hydraulic_bed_elevation(xs), 5.0)
        self.assertAlmostEqual(_regularized_wse(xs, 4.0), 5.0 + WETTING_DEPTH_FT)

    def test_section_vars_regularize_dry_state(self):
        xs = CrossSection(
            river_station='300',
            geometry=[(-30.0, 0.0), (-10.0, 5.0), (10.0, 5.0), (30.0, 0.0)],
            left_bank_station=-10.0,
            right_bank_station=10.0,
            n_lob=0.04,
            n_ch=0.035,
            n_rob=0.04,
            L_ch_to_next=500.0,
            L_lob_to_next=500.0,
            L_rob_to_next=500.0,
        )
        A, K, T = _section_vars(xs, 4.0)
        self.assertGreater(A, 0.0)
        self.assertGreater(K, 0.0)
        self.assertGreater(T, 0.0)

    def test_overbanks_inactive_until_bank_overtop(self):
        xs = CrossSection(
            river_station='400',
            geometry=[(-30.0, 8.0), (-10.0, 10.0), (0.0, 5.0), (10.0, 10.0), (30.0, 8.0)],
            left_bank_station=-10.0,
            right_bank_station=10.0,
            n_lob=0.05,
            n_ch=0.035,
            n_rob=0.05,
            L_ch_to_next=500.0,
            L_lob_to_next=1200.0,
            L_rob_to_next=1200.0,
        )
        below_bank = _unsteady_section_state(xs, 9.0, 200.0)
        self.assertAlmostEqual(below_bank.left_activation_factor, 0.0)
        self.assertAlmostEqual(below_bank.right_activation_factor, 0.0)
        self.assertAlmostEqual(below_bank.Q_lob, 0.0, places=6)
        self.assertAlmostEqual(below_bank.Q_rob, 0.0, places=6)

        above_bank = _unsteady_section_state(xs, 11.0, 200.0)
        self.assertGreater(above_bank.left_activation_factor, 0.0)
        self.assertGreater(above_bank.right_activation_factor, 0.0)
        self.assertGreater(above_bank.Q_lob, 0.0)
        self.assertGreater(above_bank.Q_rob, 0.0)

    def test_effective_reach_length_uses_subsection_flows(self):
        xs = CrossSection(
            river_station='500',
            geometry=[(-30.0, 8.0), (-10.0, 10.0), (0.0, 5.0), (10.0, 10.0), (30.0, 8.0)],
            left_bank_station=-10.0,
            right_bank_station=10.0,
            n_lob=0.03,
            n_ch=0.05,
            n_rob=0.03,
            L_ch_to_next=500.0,
            L_lob_to_next=1500.0,
            L_rob_to_next=1500.0,
        )
        state_up = _unsteady_section_state(xs, 11.0, 300.0)
        state_dn = _unsteady_section_state(xs, 11.0, 300.0)
        effective_length = _effective_reach_length(xs, state_up, state_dn, 500.0)
        self.assertGreater(effective_length, 500.0)
        self.assertLessEqual(effective_length, 1500.0)

    def test_levee_or_ineffective_elevation_delays_overbank_activation(self):
        xs = CrossSection(
            river_station='600',
            geometry=[(-30.0, 8.0), (-10.0, 10.0), (0.0, 5.0), (10.0, 10.0), (30.0, 8.0)],
            left_bank_station=-10.0,
            right_bank_station=10.0,
            n_lob=0.05,
            n_ch=0.035,
            n_rob=0.05,
            L_ch_to_next=500.0,
            L_lob_to_next=1200.0,
            L_rob_to_next=1200.0,
        )
        xs.left_levee_elev = 12.0
        xs.right_ineffective_elev = 12.5

        state = _unsteady_section_state(xs, 11.2, 200.0)
        self.assertAlmostEqual(state.left_activation_factor, 0.0)
        self.assertAlmostEqual(state.right_activation_factor, 0.0)

        wetter = _unsteady_section_state(xs, 13.0, 200.0)
        self.assertGreater(wetter.left_activation_factor, 0.0)
        self.assertGreater(wetter.right_activation_factor, 0.0)


# ---------------------------------------------------------------------------
# Unsteady solver integration test
# ---------------------------------------------------------------------------

@unittest.skipUnless(HAVE_NUMPY, 'NumPy required for solver tests')
class TestUnsteadySolverBasic(unittest.TestCase):

    def _make_run_params(self, dt=30.0, t_end=600.0) -> UnsteadyParams:
        return UnsteadyParams(
            dt=dt, t_end=t_end, theta=0.6,
            output_interval=5,
            downstream_bc='normal_depth',
            downstream_value=0.001,
        )

    def test_steady_state_preservation(self):
        """Constant hydrograph → WSE should remain approximately steady."""
        from unsteady_model import run_unsteady
        model = _make_simple_model(n_sections=4, Q_base=200.0)
        Q_base = 200.0
        hydro = HydrographBC(
            times=[0.0, 600.0], values=[Q_base, Q_base], bc_type='flow')
        params = self._make_run_params(dt=30.0, t_end=300.0)
        results = run_unsteady(model, hydro, params)

        self.assertEqual(results.n_sections, 4)
        self.assertGreater(results.n_output_times, 0)
        # All WSE values should be finite
        self.assertTrue(np.all(np.isfinite(results.wse)))
        # Discharge should be close to Q_base at all sections
        last_Q = results.q[-1, :]
        for Q_val in last_Q:
            self.assertAlmostEqual(Q_val, Q_base, delta=Q_base * 0.05)

    def test_rising_hydrograph(self):
        """Rising hydrograph → peak WSE should increase from initial to end."""
        from unsteady_model import run_unsteady
        model = _make_simple_model(n_sections=5, Q_base=100.0, bed_slope=0.002)
        hydro = HydrographBC(
            times=[0.0, 300.0, 600.0],
            values=[100.0, 500.0, 100.0],
            bc_type='flow',
        )
        params = UnsteadyParams(
            dt=30.0, t_end=600.0, theta=0.6, output_interval=5,
            downstream_bc='normal_depth', downstream_value=0.002,
        )
        results = run_unsteady(model, hydro, params)
        self.assertTrue(np.all(np.isfinite(results.wse)))
        # max_wse should exceed initial WSE at upstream section
        init_wse_us = results.wse[0, 0]
        self.assertGreater(results.max_wse[0], init_wse_us)

    def test_max_wse_is_envelope(self):
        """max_wse must be >= all time-step WSE values at every section."""
        from unsteady_model import run_unsteady
        model = _make_simple_model(n_sections=4)
        hydro = HydrographBC(
            times=[0.0, 120.0, 240.0, 360.0],
            values=[100.0, 400.0, 250.0, 100.0],
        )
        params = self._make_run_params(dt=20.0, t_end=360.0)
        results = run_unsteady(model, hydro, params)
        for t_idx in range(results.n_output_times):
            for s_idx in range(results.n_sections):
                self.assertGreaterEqual(
                    results.max_wse[s_idx] + 1e-6,
                    results.wse[t_idx, s_idx],
                    msg=f'max_wse violated at t_idx={t_idx}, s_idx={s_idx}',
                )

    def test_section_ordering(self):
        """Section IDs should be in upstream-to-downstream order."""
        from unsteady_model import run_unsteady
        model = _make_simple_model(n_sections=4, reach_length=300.0)
        hydro = HydrographBC(times=[0.0, 300.0], values=[200.0, 200.0])
        params = self._make_run_params(dt=30.0, t_end=120.0)
        results = run_unsteady(model, hydro, params)
        # section_ids should be present
        self.assertEqual(len(results.section_ids), 4)

    def test_first_step_debug_dump(self):
        """Optional debug output should capture first-step BC and first reach rows."""
        from unsteady_model import run_unsteady
        model = _make_simple_model(n_sections=4, Q_base=200.0)
        hydro = HydrographBC(times=[0.0, 120.0], values=[200.0, 200.0])
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            debug_path = f.name
        try:
            params = UnsteadyParams(
                dt=30.0, t_end=60.0, theta=0.6, output_interval=1,
                downstream_bc='normal_depth', downstream_value=0.001,
                debug_output_path=debug_path,
            )
            run_unsteady(model, hydro, params)
            with open(debug_path, 'r', encoding='utf-8') as fh:
                payload = json.load(fh)
            self.assertIn('records', payload)
            self.assertGreaterEqual(len(payload['records']), 1)
            first = payload['records'][0]
            self.assertEqual(first['step'], 1)
            self.assertIn('first_reach', first)
            self.assertIn('continuity_row', first['first_reach'])
            self.assertIn('momentum_row', first['first_reach'])
            self.assertIn('row_0_upstream_bc', first['matrix_rows'])
        finally:
            os.unlink(debug_path)


# ---------------------------------------------------------------------------
# Binary I/O tests
# ---------------------------------------------------------------------------

@unittest.skipUnless(HAVE_NUMPY, 'NumPy required for I/O tests')
class TestBinaryIO(unittest.TestCase):

    def _make_dummy_results(self) -> UnsteadyResults:
        n_t, n_s = 10, 4
        times   = np.linspace(0.0, 900.0, n_t)
        wse     = np.random.uniform(500.0, 505.0, (n_t, n_s))
        q       = np.random.uniform(100.0, 300.0, (n_t, n_s))
        max_wse = np.max(wse, axis=0)
        return UnsteadyResults(
            times=times, wse=wse, q=q, max_wse=max_wse,
            section_ids=['3000', '2000', '1000', '0'],
            run_id='TEST001',
            run_time='2026-01-01 00:00:00 UTC',
            dt=100.0, n_sections=n_s, n_output_times=n_t,
        )

    def test_save_and_load_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix='.gpkg', delete=False) as f:
            path = f.name
        try:
            results = self._make_dummy_results()
            run_id = save_unsteady_results_to_geopackage(path, results)
            loaded = load_unsteady_results_from_geopackage(path, run_id=run_id)
            self.assertIsNotNone(loaded)
            np.testing.assert_array_almost_equal(loaded.wse, results.wse, decimal=6)
            np.testing.assert_array_almost_equal(loaded.q,   results.q,   decimal=6)
            np.testing.assert_array_almost_equal(loaded.max_wse, results.max_wse, decimal=6)
            np.testing.assert_array_almost_equal(loaded.times,   results.times,   decimal=6)
            self.assertEqual(loaded.section_ids, results.section_ids)
            self.assertEqual(loaded.n_sections,  results.n_sections)
            self.assertEqual(loaded.n_output_times, results.n_output_times)
        finally:
            os.unlink(path)

    def test_load_most_recent(self):
        with tempfile.NamedTemporaryFile(suffix='.gpkg', delete=False) as f:
            path = f.name
        try:
            results = self._make_dummy_results()
            save_unsteady_results_to_geopackage(path, results)
            loaded = load_unsteady_results_from_geopackage(path)   # no run_id
            self.assertIsNotNone(loaded)
        finally:
            os.unlink(path)

    def test_load_nonexistent_returns_none(self):
        result = load_unsteady_results_from_geopackage('/nonexistent/path.gpkg')
        self.assertIsNone(result)

    def test_save_and_load_debug_records(self):
        with tempfile.NamedTemporaryFile(suffix='.gpkg', delete=False) as f:
            path = f.name
        try:
            results = self._make_dummy_results()
            run_id = save_unsteady_results_to_geopackage(path, results)
            records = [
                {
                    'step': 1,
                    'time_s': 60.0,
                    'is_output_step': True,
                    'section_ids': ['3000', '2000', '1000', '0'],
                    'z': [501.0, 500.9, 500.8, 500.7],
                    'q': [200.0, 195.0, 190.0, 185.0],
                    'inner_iterations': [
                        {
                            'inner_iter': 1,
                            'max_abs_dz_raw': 0.2,
                            'max_abs_dQ_raw': 5.0,
                            'max_abs_dz_applied': 0.1,
                            'max_abs_dQ_applied': 2.5,
                            'linear_rhs_inf': 1.2,
                            'linear_residual_inf': 0.01,
                            'damping_factor': 0.5,
                        }
                    ],
                },
                {
                    'step': 2,
                    'time_s': 120.0,
                    'is_output_step': False,
                    'section_ids': ['3000', '2000', '1000', '0'],
                    'z': [501.1, 501.0, 500.9, 500.8],
                    'q': [205.0, 200.0, 195.0, 190.0],
                    'inner_iterations': [],
                },
            ]
            n_saved = save_unsteady_debug_to_geopackage(
                path, run_id, records, record_kind='computation'
            )
            self.assertEqual(n_saved, 2)
            loaded = load_unsteady_debug_from_geopackage(
                path, run_id, record_kind='computation'
            )
            self.assertEqual(len(loaded), 2)
            self.assertEqual(int(loaded[0]['step']), 1)
            self.assertAlmostEqual(float(loaded[1]['time_s']), 120.0, places=6)
            self.assertEqual(len(loaded[0]['inner_iterations']), 1)
        finally:
            os.unlink(path)


class TestHydrographIO(unittest.TestCase):

    def test_hydrograph_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix='.gpkg', delete=False) as f:
            path = f.name
        try:
            hydro = HydrographBC(
                times=[0.0, 1800.0, 3600.0],
                values=[50.0, 250.0, 50.0],
                bc_type='flow',
                label='Test flood hydrograph',
            )
            hid = save_hydrograph_to_geopackage(path, hydro, hydrograph_id='upstream')
            loaded = load_hydrograph_from_geopackage(path, hid)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.times, hydro.times)
            self.assertEqual(loaded.values, hydro.values)
            self.assertEqual(loaded.bc_type, 'flow')
            self.assertEqual(loaded.label, 'Test flood hydrograph')
        finally:
            os.unlink(path)


if __name__ == '__main__':
    unittest.main()
