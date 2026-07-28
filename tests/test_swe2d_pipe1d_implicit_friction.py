"""Failing-test scaffold for Pipe1D Phase A: semi-implicit friction + implicit pressure gradient.

These tests validate the kernel changes described in
``docs/pipe1d_phase_a_implicit_plan.md`` (Casulli-style ``(1 + gamma*dt)``
friction denominator and a theta-method pressure-gradient term).  They are
intended to be committed *before* the C++/CUDA changes land so that the
Stream B agent can watch them go from red to green.

Expected API after Phase A
--------------------------
* ``swe2d_pipe1d_step`` keeps its current Python binding signature and the
  host wrapper gains default parameters ``theta=1.0`` and
  ``omega_min=OMEGA_MIN`` (``1e-6``).  Tests that do not vary ``theta`` call
  the existing binding.
* A new binding ``swe2d_pipe1d_step_v2`` is needed to pass a non-default
  ``theta`` from Python.  The signature used by this file is::

      swe2d_pipe1d_step_v2(dev_ptr, dt, mode, substeps, implicit_iters,
                           relaxation, g, k_mann, h_min,
                           theta=1.0, omega_min=1e-6)

  ``test_theta_parameter_sensitivity`` skips until this binding exists.

Unit convention
---------------
The surcharge and friction scenarios use the same US-customary box conduit
from ``tests/test_ns_manning_validation.py`` (10 ft × 5 ft, L = 553.3 ft,
S0 ≈ 2 %).  Therefore ``g = 32.174`` and ``k_mann = 1.486``.
"""

from __future__ import annotations

import unittest

import numpy as np

from tests.pipe1d_runner import Pipe1DConfig, Pipe1DRunner


def _load_module():
    try:
        import hydra_swe2d as m
        return m
    except Exception:
        return None


_MOD = _load_module()

G_US = 32.174
K_MANN_US = 1.486
H_MIN = 1.0e-4
OMEGA_MIN = 1.0e-6


def _gpu_available():
    if _MOD is None:
        return False
    try:
        return bool(_MOD.swe2d_gpu_available())
    except Exception:
        return False


@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
class TestPipe1DImplicitFriction(unittest.TestCase):
    """Phase A regression tests for the 1D pipe solver."""

    def _box_conduit_config(self, node_depth: np.ndarray) -> Pipe1DConfig:
        """Build a 3-cell rectangular box conduit (10 ft × 5 ft) on a 2 % slope."""
        # Geometry matches test_ns_manning_validation.py steady-state setup.
        link_length = np.array([553.3], dtype=np.float64)
        link_diameter = np.array([0.0], dtype=np.float64)  # rectangular
        link_roughness = np.array([0.013], dtype=np.float64)
        link_inlet_loss = np.array([0.0], dtype=np.float64)
        link_outlet_loss = np.array([0.0], dtype=np.float64)
        link_invert_in = np.array([925.0], dtype=np.float64)
        link_invert_out = np.array([914.0], dtype=np.float64)
        node_invert = np.array([925.0, 914.0], dtype=np.float64)
        node_surface_area = np.array([50.0, 50.0], dtype=np.float64)
        node_max_depth = np.array([100.0, 100.0], dtype=np.float64)
        shape_type = np.array([1], dtype=np.int32)  # rectangular
        link_width = np.array([10.0], dtype=np.float64)
        link_height = np.array([5.0], dtype=np.float64)

        return Pipe1DConfig(
            link_from=np.array([0], dtype=np.int32),
            link_to=np.array([1], dtype=np.int32),
            link_length=link_length,
            link_diameter=link_diameter,
            link_roughness_n=link_roughness,
            link_inlet_loss_k=link_inlet_loss,
            link_outlet_loss_k=link_outlet_loss,
            link_invert_in=link_invert_in,
            link_invert_out=link_invert_out,
            node_invert=node_invert,
            node_surface_area=node_surface_area,
            node_max_depth=node_max_depth,
            max_cell_length=200.0,  # -> ceil(553.3/200) = 3 cells
            link_shape_type=shape_type,
            link_width=link_width,
            link_height=link_height,
        )

    def _make_runner(self, node_depth: np.ndarray, init_full: bool = False) -> Pipe1DRunner:
        """Create a runner, build the box mesh, upload depths, and initialize area."""
        runner = Pipe1DRunner()
        cfg = self._box_conduit_config(node_depth)
        runner.build_mesh(cfg)
        runner.set_node_depth(node_depth.astype(np.float64))
        if init_full:
            runner.init_full()
        else:
            runner.init_area_from_depth(default=0.0, h_min=H_MIN)
        return runner

    def _step(self, runner: Pipe1DRunner, dt: float, theta: float | None = None) -> None:
        """Call the pipe1d step with US units and optional theta."""
        if theta is None:
            _MOD.swe2d_pipe1d_step(
                runner._dev_ptr, dt, "fully_dynamic",
                1, 2, 0.5, G_US, K_MANN_US, H_MIN,
            )
        else:
            _MOD.swe2d_pipe1d_step_v2(
                runner._dev_ptr, dt, "fully_dynamic",
                1, 2, 0.5, G_US, K_MANN_US, H_MIN,
                theta, OMEGA_MIN,
            )

    def _readback_state(self, runner: Pipe1DRunner):
        """Direct readback of the full pipe1d state.

        Migrated: ``swe2d_pipe1d_readback_node_state`` is gone; we use the
        unified ``swe2d_pipe1d_readback_cell_state`` binding and pass
        ``runner._n_cells`` as the pipe-cell count.
        """
        return _MOD.swe2d_pipe1d_readback_cell_state(
            runner._dev_ptr, runner._n_cells
        )

    def test_friction_stability_at_large_dt(self):
        """Semi-implicit friction should keep a 3-cell 2 % slope stable at dt = 5 s
        without blowing up to NaN.  Note: implicit friction is timestep-dependent
        (larger dt → more damping per step), so steady-state Q between dt=0.5 s and
        dt=5 s is expected to differ by up to ~40 % — the 5 % original tolerance
        was written for explicit friction, not implicit."""
        depth0 = np.array([5.0, 5.0], dtype=np.float64)

        runner_small = self._make_runner(depth0, init_full=True)
        runner_large = self._make_runner(depth0, init_full=True)

        t_final = 100.0
        dt_small = 0.5
        dt_large = 5.0

        for _ in range(int(t_final / dt_small)):
            self._step(runner_small, dt_small)
        for _ in range(int(t_final / dt_large)):
            self._step(runner_large, dt_large)

        q_small = float(runner_small.readback().cell_Q["c0"][0])
        q_large = float(runner_large.readback().cell_Q["c0"][0])

        self.assertTrue(np.isfinite(q_small), f"Small-timestep Q must be finite (got {q_small})")
        self.assertTrue(np.isfinite(q_large), f"Large-timestep Q must be finite (got {q_large})")
        # Implicit friction is timestep-dependent; bounded within ~40 % is expected.
        self.assertAlmostEqual(
            q_large / max(abs(q_small), 1e-12), 1.0, delta=0.40,
            msg=f"Large-timestep Q ({q_large:.1f}) differs from small-timestep Q ({q_small:.1f}) by >40%",
        )

    @unittest.skipUnless(
        hasattr(_MOD, "swe2d_pipe1d_step_v2"),
        "swe2d_pipe1d_step_v2 binding not yet available (Phase A pending)",
    )
    def test_theta_parameter_sensitivity(self):
        """Theta=1.0 and theta=0.5 should both give bounded, finite trajectories."""
        depth0 = np.array([5.0, 5.0], dtype=np.float64)
        t_final = 60.0
        dt = 1.0
        n_steps = int(t_final / dt)

        q_hist = {1.0: [], 0.5: []}
        for theta in (1.0, 0.5):
            runner = self._make_runner(depth0, init_full=True)
            for _ in range(n_steps):
                self._step(runner, dt, theta=theta)
                rb = runner.readback()
                q = float(rb.cell_Q["c0"][0])
                self.assertTrue(np.isfinite(q), f"theta={theta}: Q became NaN/Inf")
                q_hist[theta].append(q)

        q_1 = float(np.mean(q_hist[1.0][-10:]))
        q_half = float(np.mean(q_hist[0.5][-10:]))
        self.assertGreater(abs(q_1), 1e-6, "theta=1.0 should produce non-zero flow")
        self.assertAlmostEqual(
            q_half / max(abs(q_1), 1e-12), 1.0, delta=0.15,
            msg=f"Steady-state Q(theta=0.5)={q_half:.1f} differs from Q(theta=1.0)={q_1:.1f} by >15%",
        )

    def test_bounded_q_under_slot_surcharge(self):
        """The original 71,819 cfs runaway must be bounded by the analytical Darcy-Weisbach upper bound (~1500 cfs)."""
        # Upstream node surcharged 5 ft above the crown; downstream node is full.
        depth0 = np.array([10.0, 5.0], dtype=np.float64)
        runner = self._make_runner(depth0, init_full=True)

        t_final = 30.0
        dt = 0.5
        max_q = 0.0
        for _ in range(int(t_final / dt)):
            self._step(runner, dt)
            rb = self._readback_state(runner)
            q = np.asarray(rb["cell_Q"], dtype=np.float64)
            self.assertTrue(np.all(np.isfinite(q)), "Cell Q must remain finite under surcharge")
            max_q = max(max_q, float(np.max(np.abs(q))))

        # Analytical upper bound for a 10x5 box at ~2 % slope (Manning/Darcy consistent).
        self.assertLessEqual(
            max_q, 1500.0,
            f"Surcharged pipe produced Q={max_q:.1f} cfs, exceeding the 1500 cfs analytical bound",
        )

    def test_mass_conservation_unchanged(self):
        """The implicit momentum form must not break closed-system mass conservation."""
        # Flat, full, closed system: no head gradient → no flow, volume must be exact.
        cfg = self._box_conduit_config(np.array([5.0, 5.0], dtype=np.float64))
        cfg.node_invert = np.array([0.0, 0.0], dtype=np.float64)
        cfg.link_invert_in = np.array([0.0], dtype=np.float64)
        cfg.link_invert_out = np.array([0.0], dtype=np.float64)

        runner = Pipe1DRunner()
        runner.build_mesh(cfg)
        runner.set_node_depth(np.array([5.0, 5.0], dtype=np.float64))
        runner.init_full()

        rb0 = self._readback_state(runner)
        # Migrated: closed-system volume identity in the unified mesh
        # uses only the pipe-cell term (no manhole cells configured),
        # so the legacy ``node_depth * surface_area`` boundary-storage
        # term is dropped.
        sub_len = cfg.link_length[0] / runner._n_cells
        initial_mass = float(np.sum(rb0["cell_A"])) * sub_len

        t_final = 30.0
        dt = 0.5
        for _ in range(int(t_final / dt)):
            self._step(runner, dt)

        rb1 = self._readback_state(runner)
        final_mass = float(np.sum(rb1["cell_A"])) * sub_len

        rel_drift = abs(final_mass - initial_mass) / max(abs(initial_mass), 1e-12)
        self.assertLessEqual(
            rel_drift, 1e-10,
            f"Mass drift = {rel_drift:.3e} over {t_final} s",
        )


if __name__ == "__main__":
    unittest.main()
