"""Closed-pipe mass conservation test — machine-epsilon verification.

Creates a 2-node, 1-link pipe network with WALL_BC at both ends (reflective
ghost, zero net flux).  Initialized with a cosine perturbation, runs
RK2+Muscl+HLLC, and asserts total mass (sum A·dx) is conserved to
machine epsilon.  Mirrors the Python prototype at tools/test_1d_fvm_prototype.py
but exercises the actual GPU hot path through swe2d_pipe1d_step.

All pipe-end faces use class 7 (WALL_BC) — the mesh builder assigns this
when no node_is_outfall / node_is_inlet arrays are provided.
"""

import numpy as np
import unittest
import math
import importlib

_G = 9.80665
_K_MANN = 1.0
_H_MIN = 1.0e-10

_MOD = None
try:
    _MOD = importlib.import_module("hydra_swe2d")
except ImportError:
    pass


def _gpu_available():
    if _MOD is None:
        return False
    try:
        return bool(_MOD.swe2d_gpu_available())
    except Exception:
        return False


@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
class TestPipe1DClosedMassConservation(unittest.TestCase):
    """Mass conservation to machine epsilon for closed pipe networks."""

    _n_cells = 10
    _dx = 10.0

    @classmethod
    def setUpClass(cls):
        from swe2d.runtime.backend import SWE2DBackend
        cls._backend = SWE2DBackend()
        cls._backend.build_mesh(
            np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
            np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
            np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
            np.asarray([0, 1, 2], dtype=np.int32),
        )
        cls._backend.initialize(
            h0=np.asarray([0.1], dtype=np.float64),
            hu0=np.zeros(1, dtype=np.float64),
            hv0=np.zeros(1, dtype=np.float64),
            dt_fixed=0.05, dt_max=0.05,
        )
        cls._dev = int(_MOD.swe2d_get_coupling_dev_ptr())
        cls._build_base_mesh()

    @classmethod
    def _build_base_mesh(cls):
        _MOD.swe2d_build_unified_mesh(
            dev_ptr=cls._dev,
            n_links=1,
            link_from=np.array([0], dtype=np.int32),
            link_to=np.array([1], dtype=np.int32),
            L=np.array([100.0], dtype=np.float64),
            D=np.array([1.0], dtype=np.float64),
            n_mann=np.array([0.013], dtype=np.float64),
            S0=np.zeros(1, dtype=np.float64),
            node_invert=np.array([10.0, 10.0], dtype=np.float64),
            mcl=10.0,
            link_shape_type=np.zeros(1, dtype=np.int32),
            link_width=np.array([1.0], dtype=np.float64),
            link_height=np.array([1.0], dtype=np.float64),
        )

    def setUp(self):
        self._build_base_mesh()
        # Cosine perturbation in circular pipe
        L, D = 100.0, 1.0
        r = D / 2.0
        xc = np.linspace(0.5 * L / self._n_cells, L - 0.5 * L / self._n_cells,
                         self._n_cells)
        h_field = 0.6 + 0.2 * np.cos(2.0 * math.pi * (xc - 0.5 * L) / L)
        _MOD.swe2d_pipe1d_upload_cell_h(self._dev, np.asarray(h_field, dtype=np.float64))
        _MOD.swe2d_pipe1d_init_cell_area(self._dev, _H_MIN)

    def _mass(self):
        st = _MOD.swe2d_pipe1d_readback_cell_state(self._dev, self._n_cells)
        A = np.asarray(st.get("cell_A", np.zeros(self._n_cells)), dtype=np.float64)
        return float(np.sum(A) * self._dx)

    def _step_kw(self, dt=0.5):
        _MOD.swe2d_pipe1d_step(self._dev, dt, "rk2", 1, 2, 0.5,
                               _G, _K_MANN, _H_MIN,
                               surcharge_method=1, recon_method=1, theta=1.0)

    def test_mass_conserved_circular_open_100s(self):
        """Circular pipe, open-channel, 100 s — mass drift ≈ machine epsilon."""
        m0 = self._mass()
        for _ in range(200):
            self._step_kw()
        m1 = self._mass()
        self.assertAlmostEqual(m1 / m0, 1.0, places=10,
                               msg=f"Circ open: drift {(m1 - m0) / m0:+.2e}")

    def test_mass_conserved_circular_pressurised_100s(self):
        """Circular pipe, pressurised (slot surcharge), 100 s."""
        h_field = np.full(self._n_cells, 1.3, dtype=np.float64)
        _MOD.swe2d_pipe1d_upload_cell_h(self._dev, h_field)
        _MOD.swe2d_pipe1d_init_cell_area(self._dev, _H_MIN)
        m0 = self._mass()
        for _ in range(2000):
            self._step_kw(dt=0.05)
        m1 = self._mass()
        self.assertAlmostEqual(m1 / m0, 1.0, places=8,
                               msg=f"Circ press: drift {(m1 - m0) / m0:+.2e}")

    def test_mass_conserved_elliptical_open_100s(self):
        """Elliptical pipe, open-channel, 100 s."""
        _MOD.swe2d_build_unified_mesh(
            dev_ptr=self._dev,
            n_links=1,
            link_from=np.array([0], dtype=np.int32),
            link_to=np.array([1], dtype=np.int32),
            L=np.array([100.0], dtype=np.float64),
            D=np.array([1.0], dtype=np.float64),
            n_mann=np.array([0.013], dtype=np.float64),
            S0=np.zeros(1, dtype=np.float64),
            node_invert=np.array([10.0, 10.0], dtype=np.float64),
            mcl=10.0,
            link_shape_type=np.full(1, 2, dtype=np.int32),  # ELLIPTICAL
            link_width=np.array([1.5], dtype=np.float64),
            link_height=np.array([1.0], dtype=np.float64),
        )
        L, H = 100.0, 1.0
        xc = np.linspace(0.5 * L / 10, L - 0.5 * L / 10, 10)
        h_field = 0.6 + 0.2 * np.sin(math.pi * xc / L)
        _MOD.swe2d_pipe1d_upload_cell_h(self._dev, np.asarray(h_field, dtype=np.float64))
        _MOD.swe2d_pipe1d_init_cell_area(self._dev, _H_MIN)
        self._dx = 10.0
        m0 = self._mass()
        for _ in range(200):
            self._step_kw()
        m1 = self._mass()
        self.assertAlmostEqual(m1 / m0, 1.0, places=10,
                               msg=f"Ellip open: drift {(m1 - m0) / m0:+.2e}")


if __name__ == "__main__":
    unittest.main()
