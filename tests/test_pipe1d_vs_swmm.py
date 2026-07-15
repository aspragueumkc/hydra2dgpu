"""Validate pipe1d solver basic physics.

Open-channel tests: sanity check against Manning's (empirical, ±50 %).
Pressurized test: cross-compare with SWMM (same Saint-Venant formulation).
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from tests.swmm_runner import SWMMRunner, make_drainage_inp
from tests.pipe1d_runner import Pipe1DRunner, Pipe1DConfig


def mannings_q(diameter: float, slope: float, n: float, depth_frac: float = 1.0,
               ) -> float:
    if depth_frac <= 0.0 or diameter <= 0.0:
        return 0.0
    depth_frac = min(1.0, depth_frac)
    A = 0.25 * math.pi * diameter * diameter * depth_frac
    P = math.pi * diameter * depth_frac
    R = A / max(1e-10, P)
    return (1.0 / n) * A * (R ** (2.0 / 3.0)) * math.sqrt(abs(slope))


def _gpu_available():
    try:
        import hydra_swe2d as m
        return bool(m.swe2d_gpu_available())
    except Exception:
        return False


PIPE_D = 3.0
PIPE_L = 100.0
PIPE_N = 0.01
NODE_AREA = 50.0


def _pipe1d_q(depth_n0: float, depth_n1: float, slope: float,
               k_in: float = 0.0, k_out: float = 0.0,
               n_steps: int = 500, dt: float = 0.25,
               solver_mode: str = "diffusion_wave",
               implicit_iters: int = 2) -> float:
    depth_arr = np.array([depth_n0, depth_n1], dtype=np.float64)
    runner = Pipe1DRunner()
    try:
        cfg = Pipe1DConfig(
            link_from=np.array([0], dtype=np.int32),
            link_to=np.array([1], dtype=np.int32),
            link_length=np.array([PIPE_L], dtype=np.float64),
            link_diameter=np.array([PIPE_D], dtype=np.float64),
            link_roughness_n=np.array([PIPE_N], dtype=np.float64),
            link_inlet_loss_k=np.array([k_in], dtype=np.float64),
            link_outlet_loss_k=np.array([k_out], dtype=np.float64),
            link_invert_in=np.array([0.0], dtype=np.float64),
            link_invert_out=np.array([-slope * PIPE_L], dtype=np.float64),
            node_invert=np.array([0.0, -slope * PIPE_L], dtype=np.float64),
            node_surface_area=np.array([NODE_AREA, NODE_AREA],
                                       dtype=np.float64),
            node_max_depth=np.array([50.0, 50.0], dtype=np.float64),
            max_cell_length=25,
        )
        runner.build_mesh(cfg)
        runner.set_node_depth(depth_arr)
        runner.init_area_from_depth()
        for _ in range(n_steps):
            runner.set_node_depth(depth_arr)
            runner.step(dt=dt, solver_mode=solver_mode,
                        implicit_iters=implicit_iters)
        result = runner.readback()
        return float(result.cell_Q["c0"][0])
    finally:
        runner.destroy()


@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
class TestOpenChannel(unittest.TestCase):
    """Open-channel flow at uniform depth — Manning's serves as rough check."""

    def test_half_pipe_reasonable(self):
        """50 % full: Q within factor 2 of Manning's."""
        slope = 0.01
        d_frac = 0.5
        depth = PIPE_D * d_frac
        q = _pipe1d_q(depth_n0=depth, depth_n1=depth, slope=slope, n_steps=200)
        expected = mannings_q(PIPE_D, slope, PIPE_N, d_frac)
        self.assertGreater(q, 0.01)
        self.assertLess(q / expected, 2.0)
        self.assertGreater(q / expected, 0.5)

    def test_slope_scaling(self):
        """Q scales with sqrt(slope) — sanity check."""
        q_low = _pipe1d_q(depth_n0=1.5, depth_n1=1.5, slope=0.005, n_steps=200)
        q_high = _pipe1d_q(depth_n0=1.5, depth_n1=1.5, slope=0.02, n_steps=200)
        ratio = q_high / max(1e-10, q_low)
        self.assertGreater(ratio, 1.4,
                           f"Q ratio = {ratio:.3f}, expected sqrt(4)=2")

    def test_nonzero_head_gives_flow(self):
        """No head difference → no flow. Head difference → flow."""
        q_no = _pipe1d_q(depth_n0=1.5, depth_n1=1.5, slope=0.0, n_steps=100)
        q_yes = _pipe1d_q(depth_n0=2.0, depth_n1=1.5, slope=0.01, n_steps=200)
        self.assertLess(q_no, 0.01, "zero head → zero flow")
        self.assertGreater(q_yes, 0.01, "head difference → flow")


@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
class TestPressurizedFlow(unittest.TestCase):
    """Pressurized (full) pipe flow compared against SWMM."""

    def test_pipe1d_finite_above_crown(self):
        """Above-crown head produces finite Q (no blow-up)."""
        q = _pipe1d_q(depth_n0=6.0, depth_n1=6.0, slope=0.01,
                       k_in=0.5, k_out=1.0, n_steps=500)
        self.assertTrue(math.isfinite(q))
        self.assertGreater(q, 1.0)

    def test_pipe1d_stable_at_large_dt(self):
        """Implicit friction keeps the solver stable at dt=0.25 (no blow-up)."""
        q = _pipe1d_q(depth_n0=6.0, depth_n1=6.0, slope=0.01,
                       k_in=0.5, k_out=1.0, n_steps=500, dt=0.25,
                       solver_mode="diffusion_wave")
        self.assertTrue(math.isfinite(q),
                        "Q must be finite at dt=0.25 with implicit friction")

    def test_pipe1d_vs_swmm(self):
        """Pressurized flow: pipe1d and SWMM Q agree within 5 % for the same head.

        pipe1d uses implicit Manning friction (same pattern as the 2D solver:
        ``hu /= 1 + dt*cf*spd``), making the pipe solver unconditionally stable
        for any dt.  The comparison here uses dt=0.001 to minimise the influence
        of the fixed-head BC reset on the convergence path.
        """
        import importlib.util
        if importlib.util.find_spec("swmm") is None:
            self.skipTest("swmm-toolkit not installed")

        slope = 0.01
        inflow = 1.5 * mannings_q(PIPE_D, slope, PIPE_N)

        inp = make_drainage_inp(
            junctions=[("n1", 0.0, 50.0)],
            outfalls=[("n2", -slope * PIPE_L)],
            conduits=[("c1", "n1", "n2", PIPE_L, PIPE_N, PIPE_D)],
            xsections=[("c1", "CIRCULAR", PIPE_D)],
            inflows=[("n1", "TS1")],
            timeseries=[("TS1", 0, inflow), ("TS1", 2, inflow)],
            end_time="03:00:00",
            routing_step_s=5.0,
        )
        runner = SWMMRunner()
        try:
            _, nodes, links = runner.run(inp, max_steps=100)
        finally:
            pass

        depths = [n1.depth for n1 in nodes["n1"]]
        flows = [r.flow for r in links["c1"]]
        if not depths or not flows:
            self.skipTest("SWMM produced no output")

        swmm_depth = float(np.mean(depths[-20:]))
        swmm_q = float(np.mean(flows[-20:]))

        # SWMM must surcharge and conserve mass
        self.assertGreater(swmm_depth, PIPE_D,
                           f"SWMM depth={swmm_depth:.3f} > crown={PIPE_D}m")
        self.assertAlmostEqual(swmm_q / inflow, 1.0, delta=0.05,
                               msg=f"SWMM Q/inflow={swmm_q/inflow:.3f}")

        # pipe1d with the same upstream head, small dt for friction stability
        q_pipe1d = _pipe1d_q(depth_n0=swmm_depth, depth_n1=0.0, slope=slope,
                              k_in=0.5, k_out=1.0,
                              dt=0.001, n_steps=25000,
                              solver_mode="diffusion_wave")
        self.assertTrue(math.isfinite(q_pipe1d))
        self.assertGreater(q_pipe1d, 1.0)
        ratio = q_pipe1d / max(1e-10, swmm_q)
        self.assertAlmostEqual(ratio, 1.0, delta=0.05,
                               msg=f"pipe1d/SWMM={ratio:.3f}")

    def test_swmm_free_outfall_surcharges(self):
        """SWMM surcharges above crown when inflow exceeds pipe capacity."""
        import importlib.util
        if importlib.util.find_spec("swmm") is None:
            self.skipTest("swmm-toolkit not installed")

        slope = 0.01
        # Push 2x the full-pipe Manning Q — junction should surcharge
        inflow = 2.0 * mannings_q(PIPE_D, slope, PIPE_N)
        inp = make_drainage_inp(
            junctions=[("n1", 0.0, 50.0)],
            outfalls=[("n2", -slope * PIPE_L)],
            conduits=[("c1", "n1", "n2", PIPE_L, PIPE_N, PIPE_D)],
            xsections=[("c1", "CIRCULAR", PIPE_D)],
            inflows=[("n1", "TS1")],
            timeseries=[("TS1", 0, inflow), ("TS1", 2, inflow)],
            end_time="03:00:00",
            routing_step_s=2.0,
        )
        runner = SWMMRunner()
        try:
            _, nodes, _ = runner.run(inp, max_steps=100)
        finally:
            pass
        depths = [n1.depth for n1 in nodes["n1"]]
        if not depths:
            self.skipTest("SWMM produced no output")
        max_d = float(np.max(depths))
        self.assertGreater(max_d, PIPE_D,
                           f"surcharged depth={max_d:.2f} > crown={PIPE_D}m")


@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
class TestPipeEntranceLoss(unittest.TestCase):
    """HEC-22 entrance/exit loss coefficients reduce Q."""

    def test_entrance_loss_reduces_flow(self):
        slope = 0.01
        depth = PIPE_D * 0.5
        q_no_loss = _pipe1d_q(depth_n0=depth, depth_n1=depth, slope=slope,
                               k_in=0.0, k_out=0.0, n_steps=200)
        q_with_loss = _pipe1d_q(depth_n0=depth, depth_n1=depth, slope=slope,
                                k_in=0.5, k_out=1.0, n_steps=500)
        self.assertGreater(q_no_loss, 0.01)
        self.assertGreater(q_with_loss, 0.01)
        self.assertLess(q_with_loss / max(1e-10, q_no_loss), 0.95,
                        "loss coefficients should reduce Q")


@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
class TestPipeFlowReversal(unittest.TestCase):
    """Flow reversal must not blow up the solver."""

    def test_reversal_stable(self):
        runner = Pipe1DRunner()
        try:
            cfg = Pipe1DConfig(
                link_from=np.array([0], dtype=np.int32),
                link_to=np.array([1], dtype=np.int32),
                link_length=np.array([PIPE_L], dtype=np.float64),
                link_diameter=np.array([PIPE_D], dtype=np.float64),
                link_roughness_n=np.array([PIPE_N], dtype=np.float64),
                link_inlet_loss_k=np.array([0.5], dtype=np.float64),
                link_outlet_loss_k=np.array([1.0], dtype=np.float64),
                link_invert_in=np.array([0.0], dtype=np.float64),
                link_invert_out=np.array([-10.0], dtype=np.float64),
                node_invert=np.array([0.0, -10.0], dtype=np.float64),
                node_surface_area=np.array([NODE_AREA, NODE_AREA],
                                           dtype=np.float64),
                node_max_depth=np.array([50.0, 50.0], dtype=np.float64),
                max_cell_length=25,
            )
            runner.build_mesh(cfg)

            fwd_arr = np.array([PIPE_D, 0.0], dtype=np.float64)
            for _ in range(100):
                runner.set_node_depth(fwd_arr)
                runner.init_area_from_depth()
                runner.step(dt=0.25, solver_mode="diffusion_wave")
            fwd = runner.readback().cell_Q["c0"][0]
            self.assertTrue(math.isfinite(fwd))
            self.assertGreater(fwd, 0)

            rev_arr = np.array([0.0, 20.0], dtype=np.float64)
            for _ in range(500):
                runner.set_node_depth(rev_arr)
                runner.init_area_from_depth()
                runner.step(dt=0.25, solver_mode="diffusion_wave")

            import hydra_swe2d as _m
            state = _m.swe2d_pipe1d_readback_node_state(
                runner._dev_ptr, runner._n_nodes, runner._n_cells)
            cell_q = state["cell_Q"]
            for ci, q in enumerate(cell_q):
                self.assertTrue(math.isfinite(q),
                                f"cell[{ci}] Q={q} should be finite")
                self.assertLess(q, 0,
                                f"cell[{ci}] Q={q:.3f} should be negative")
        finally:
            runner.destroy()
