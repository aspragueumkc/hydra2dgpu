"""Diagnostic test: simplest pipe flow via SURFACE_2D_PIPE_END coupling.

Config (model units — set by the units API):
  - 2D mesh: 2 triangular cells (upstream wet, downstream dry)
  - Pipe: 1 link, 10 m long, 2 m diameter, 1% slope, 4 sub-cells
  - Both pipe nodes are pipe_ends connected to the 2D cells
  - Initial: h[0] = 4.0 (head on upstream), h[1] = 0.0 (dry downstream)

If the SURFACE_2D_PIPE_END coupling works, water should flow from cell 0
through the pipe and out to cell 1.  If not, both cells stay at their
initial values — confirming the core mechanism is broken.

Unit-agnostic: uses swe2d.units to configure the solver's g and k_mann
for whichever model units the mesh is in.

Run:
    mamba run -n qgis_stable python3 -m unittest -v tests.test_pipe_end_surface_coupling
"""
from __future__ import annotations

import unittest

import numpy as np


def _load_module():
    try:
        import hydra_swe2d as m
        return m
    except Exception:
        return None


_MOD = _load_module()


def _gpu_available():
    if _MOD is None:
        return False
    try:
        return bool(_MOD.swe2d_gpu_available())
    except Exception:
        return False


def _configure_units_si():
    """Configure solver units to SI (metres, seconds, kg)."""
    from swe2d import units as _u
    _u.configure(1.0)  # 1.0 SI m per model m
    return _u.gravity(), _u.manning_factor()


H_MIN = 1.0e-4


@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
class TestPipeEndSurfaceCoupling(unittest.TestCase):
    g: float = 9.81
    k_mann: float = 1.0

    @classmethod
    def setUpClass(cls):
        from swe2d.runtime.backend import SWE2DBackend

        # Configure unit system BEFORE creating the solver so g / k_mann
        # are consistent with whatever units the mesh is in.
        g, k_mann = _configure_units_si()
        cls.g = g
        cls.k_mann = k_mann

        cls._backend = SWE2DBackend()

        # 2-cell triangular mesh: two disjoint triangles
        # Cell 0 (upstream): nodes 0,1,2 at z=9.0
        # Cell 1 (downstream): nodes 3,4,5 at z=8.9
        cls._backend.build_mesh(
            node_x=np.asarray([0.0, 1.0, 0.0, 2.0, 3.0, 2.0], dtype=np.float64),
            node_y=np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float64),
            node_z=np.asarray([9.0, 9.0, 9.0, 8.9, 8.9, 8.9], dtype=np.float64),
            cell_nodes=np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int32),
        )
        # h[0]=4.0 (head on upstream), h[1]=0.0 (dry downstream)
        cls._backend.initialize(
            h0=np.asarray([4.0, 0.0], dtype=np.float64),
            hu0=np.zeros(2, dtype=np.float64),
            hv0=np.zeros(2, dtype=np.float64),
            dt_fixed=0.01,
            dt_max=0.01,
            h_min=H_MIN,
            g=g,
            k_mann=k_mann,
            n_mann=0.013,
        )
        cls._dev = int(_MOD.swe2d_get_coupling_dev_ptr())

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls._backend, "destroy"):
            cls._backend.destroy()

    def _readback_2d(self):
        """Read back 2D solver state (h, hu, hv)."""
        return _MOD.swe2d_readback_state(self._backend._solver_h, 2)

    def _readback_pipe1d(self, n_pipe_cells):
        """Read back pipe1D cell state."""
        return _MOD.swe2d_pipe1d_readback_cell_state(self._dev, n_pipe_cells, 0, 0)

    def test_pipe_flow_from_wet_to_dry(self):
        """Water should flow from wet upstream 2D cell through pipe to dry downstream."""
        # Pipe config: 1 link, 10 m, 2 m diameter, 1% slope, 4 sub-cells
        L = np.asarray([10.0], dtype=np.float64)
        D = np.asarray([2.0], dtype=np.float64)
        n_mann_arr = np.asarray([0.013], dtype=np.float64)
        node_invert = np.asarray([10.0, 9.9], dtype=np.float64)  # 1% slope over 10 m
        mcl = 3.0  # ceil(10/3) = 4 sub-cells

        _MOD.swe2d_build_unified_mesh(
            dev_ptr=self._dev,
            n_links=1,
            link_from=np.asarray([0], dtype=np.int32),
            link_to=np.asarray([1], dtype=np.int32),
            L=L,
            D=D,
            n_mann=n_mann_arr,
            S0=np.asarray([0.01], dtype=np.float64),  # 1% slope
            node_invert=node_invert,
            mcl=mcl,
            n_pipe_ends=2,
            pipe_end_node_ids=np.asarray([0, 1], dtype=np.int32),
            node_is_outfall=np.asarray([0, 0], dtype=np.int32),
        )

        # Wire pipe_end faces to 2D cells:
        # Node 0 (upstream pipe end) → 2D cell 0
        # Node 1 (downstream pipe end) → 2D cell 1
        _MOD.swe2d_pipe1d_upload_pipe_end_surface_faces(
            self._dev,
            np.asarray([0, 1], dtype=np.int32),
        )

        # Init pipe cell areas from zero depth (dry pipe)
        _MOD.swe2d_pipe1d_init_cell_area(self._dev, H_MIN)

        # Read initial state
        n_pipe_cells = 4
        st0 = self._readback_pipe1d(n_pipe_cells)
        cell_A0 = np.asarray(st0["cell_A"], dtype=np.float64).copy()

        state0 = self._readback_2d()
        h0_2d = np.asarray(state0["h"], dtype=np.float64).copy()
        self.assertAlmostEqual(h0_2d[0], 4.0, places=3)
        self.assertAlmostEqual(h0_2d[1], 0.0, places=4)

        # Run pipe1D + 2D steps
        dt = 0.01
        for _ in range(50):
            _MOD.swe2d_pipe1d_step(
                self._dev, dt, "fully_dynamic", 1, 2, 0.5,
                self.g, self.k_mann, H_MIN,
                surcharge_method=1,
                friction_method=1,
                time_integrator=2,
            )
            self._backend.step(dt)

        # Read final state
        st1 = self._readback_pipe1d(n_pipe_cells)
        cell_A1 = np.asarray(st1["cell_A"], dtype=np.float64)

        state1 = self._readback_2d()
        h1_2d = np.asarray(state1["h"], dtype=np.float64)

        print(f"\n2D h before: [{h0_2d[0]:.4f}, {h0_2d[1]:.4f}]")
        print(f"2D h after:  [{h1_2d[0]:.4f}, {h1_2d[1]:.4f}]")
        print(f"Pipe A before: {cell_A0}")
        print(f"Pipe A after:  {cell_A1}")

        # Assertions:
        self.assertLess(
            h1_2d[0], h0_2d[0] * 0.99,
            f"Upstream 2D cell h did not decrease: {h0_2d[0]:.4f} → {h1_2d[0]:.4f}",
        )
        pipe_volume_before = float(np.sum(cell_A0) * (10.0 / n_pipe_cells))
        pipe_volume_after = float(np.sum(cell_A1) * (10.0 / n_pipe_cells))
        self.assertGreater(
            pipe_volume_after, pipe_volume_before + 1.0e-8,
            f"Pipe volume did not increase: {pipe_volume_before:.6f} → {pipe_volume_after:.6f}",
        )


if __name__ == "__main__":
    unittest.main()
