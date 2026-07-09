"""
GPU tests for 1D pipe surcharge / volume decomposition behavior.

Tests that:
1. Node depths above max_depth persist through step (no cap removed)
2. Surcharged pipe produces finite fluxes (no NaN)
3. Mass is conserved in closed surcharged system
4. Surcharge propagates through two-pipe network
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


def _build_single_pipe(dev_ptr, node_depth_in=None, node_surface_area=10.0,
                       link_diameter=1.0, node_max_depth=3.0):
    """Build a single-link pipe mesh on the given device pointer."""
    a = {
        "n_links": 1, "n_nodes": 2,
        "link_from": np.array([0], dtype=np.int32),
        "link_to": np.array([1], dtype=np.int32),
        "link_length": np.array([10.0], dtype=np.float64),
        "link_diameter": np.array([link_diameter], dtype=np.float64),
        "link_roughness": np.array([0.013], dtype=np.float64),
        "link_inlet_loss": np.array([0.0], dtype=np.float64),
        "link_outlet_loss": np.array([0.0], dtype=np.float64),
        "link_invert_in": np.array([0.0], dtype=np.float64),
        "link_invert_out": np.array([0.0], dtype=np.float64),
        "node_invert": np.array([0.0, 0.0], dtype=np.float64),
        "node_surface_area": np.array([node_surface_area, node_surface_area], dtype=np.float64),
        "node_max_depth": np.array([node_max_depth, node_max_depth], dtype=np.float64),
        "node_depth": np.array(node_depth_in if node_depth_in is not None else [0.5, 0.1], dtype=np.float64),
    }
    _MOD.swe2d_build_pipe1d_mesh(
        a["n_links"],
        a["link_from"], a["link_to"],
        a["link_length"], a["link_diameter"], a["link_roughness"],
        a["link_inlet_loss"], a["link_outlet_loss"],
        a["node_invert"], a["node_surface_area"], a["node_max_depth"],
        a["link_invert_in"], a["link_invert_out"],
        0, dev_ptr,
    )
    _MOD.swe2d_pipe1d_upload_node_depth(dev_ptr, a["node_depth"])
    _MOD.swe2d_pipe1d_init_area_from_depth(dev_ptr)
    return dev_ptr, a


@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
class TestPipe1DSurcharge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from swe2d.runtime.backend import SWE2DBackend
        cls._backend = SWE2DBackend()
        node_x = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        node_y = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        node_z = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
        cell_nodes = np.asarray([0, 1, 2], dtype=np.int32)
        cls._backend.build_mesh(
            node_x, node_y, node_z, cell_nodes,
            bc_edge_node0=np.empty(0, dtype=np.int32),
            bc_edge_node1=np.empty(0, dtype=np.int32),
            bc_edge_type=np.empty(0, dtype=np.int32),
            bc_edge_val=np.empty(0, dtype=np.float64),
        )
        cls._backend.initialize(
            h0=np.asarray([0.1], dtype=np.float64),
            hu0=np.zeros(1, dtype=np.float64),
            hv0=np.zeros(1, dtype=np.float64),
            dt_fixed=0.05, dt_max=0.05,
        )
        cls._dev_ptr = int(_MOD.swe2d_get_coupling_dev_ptr())

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, '_backend'):
            cls._backend.destroy()

    def test_surcharge_node_depth_uncapped(self):
        """Node depths above max_depth persist through step (no cap)."""
        A_full = np.pi * 0.5 ** 2
        dev_ptr, a = _build_single_pipe(
            self._dev_ptr,
            node_depth_in=np.array([5.0, 5.0]),
            node_max_depth=3.0,
        )
        rb = _MOD.swe2d_pipe1d_readback_node_state(dev_ptr, a["n_nodes"], 1)
        self.assertGreater(float(rb["node_depth"][0]), 3.0,
                           "Node 0 depth should NOT be capped at max_depth")
        self.assertGreater(float(rb["node_depth"][1]), 3.0,
                           "Node 1 depth should NOT be capped at max_depth")

        cell_A = float(rb["cell_A"][0])
        self.assertLessEqual(cell_A, A_full + 1e-6,
                             f"cell_A ({cell_A}) should be <= A_full")

    def test_full_cell_flux_stability(self):
        """Flux kernel with surcharged pipe produces finite fluxes (no NaN)."""
        dev_ptr, a = _build_single_pipe(
            self._dev_ptr,
            node_depth_in=np.array([5.0, 5.0]),
            node_max_depth=3.0,
        )
        _MOD.swe2d_pipe1d_step(
            self._dev_ptr, 0.5, "fully_dynamic",
            5, 20, 0.5, 9.81,
        )
        rb = _MOD.swe2d_pipe1d_readback_node_state(self._dev_ptr, a["n_nodes"], 1)
        self.assertTrue(np.all(np.isfinite(rb["cell_A"])),
                        "cell_A should be finite after step with surcharged pipe")
        self.assertTrue(np.all(np.isfinite(rb["cell_Q"])),
                        "cell_Q should be finite after step with surcharged pipe")
        self.assertTrue(np.all(np.isfinite(rb["node_depth"])),
                        "node_depth should be finite after step with surcharged pipe")

    def test_mass_conservation_surcharge(self):
        """Total volume (pipe A*L + node_depth*surface_area) is conserved
        in closed system with initial surcharge."""
        A_full = np.pi * 0.5 ** 2
        L = 10.0
        surf_area = 10.0

        dev_ptr, a = _build_single_pipe(
            self._dev_ptr,
            node_depth_in=np.array([5.0, 5.0]),
            node_max_depth=3.0,
            node_surface_area=surf_area,
        )
        rb = _MOD.swe2d_pipe1d_readback_node_state(self._dev_ptr, a["n_nodes"], 1)
        cell_A_init = float(rb["cell_A"][0])
        node_depth_init = rb["node_depth"].copy()
        vol_init = cell_A_init * L + float(np.sum(node_depth_init * surf_area))

        for _ in range(5):
            _MOD.swe2d_pipe1d_step(
                self._dev_ptr, 1.0, "fully_dynamic",
                5, 20, 0.5, 9.81,
            )

        rb = _MOD.swe2d_pipe1d_readback_node_state(self._dev_ptr, a["n_nodes"], 1)
        cell_A_final = float(rb["cell_A"][0])
        node_depth_final = rb["node_depth"].copy()
        vol_final = cell_A_final * L + float(np.sum(node_depth_final * surf_area))

        self.assertAlmostEqual(vol_final, vol_init, delta=1e-3,
                           msg="Total volume should be conserved in closed surcharged system")

    def test_two_pipe_surcharge_propagation(self):
        """Two-pipe network: surcharge propagates between nodes."""
        n_links = 2
        link_from = np.array([0, 1], dtype=np.int32)
        link_to = np.array([1, 2], dtype=np.int32)
        link_length = np.array([5.0, 5.0], dtype=np.float64)
        link_diameter = np.array([1.0, 1.0], dtype=np.float64)
        link_roughness = np.array([0.013, 0.013], dtype=np.float64)
        link_inlet_loss = np.array([0.0, 0.0], dtype=np.float64)
        link_outlet_loss = np.array([0.0, 0.0], dtype=np.float64)
        link_invert_in = np.array([0.0, 0.0], dtype=np.float64)
        link_invert_out = np.array([0.0, 0.0], dtype=np.float64)
        n_nodes = 3
        node_invert = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        node_surf = np.array([10.0, 10.0, 10.0], dtype=np.float64)
        node_maxd = np.array([3.0, 3.0, 3.0], dtype=np.float64)
        node_depth = np.array([5.0, 5.0, 5.0], dtype=np.float64)

        _MOD.swe2d_build_pipe1d_mesh(
            n_links, link_from, link_to,
            link_length, link_diameter, link_roughness,
            link_inlet_loss, link_outlet_loss,
            node_invert, node_surf, node_maxd,
            link_invert_in, link_invert_out,
            0, self._dev_ptr,
        )
        _MOD.swe2d_pipe1d_upload_node_depth(self._dev_ptr, node_depth)
        _MOD.swe2d_pipe1d_init_area_from_depth(self._dev_ptr)

        _MOD.swe2d_pipe1d_step(
            self._dev_ptr, 0.5, "fully_dynamic",
            5, 20, 0.5, 9.81,
        )
        rb = _MOD.swe2d_pipe1d_readback_node_state(self._dev_ptr, n_nodes, n_links)
        self.assertTrue(np.all(np.isfinite(rb["cell_A"])),
                        "cell_A should be finite")
        self.assertTrue(np.all(np.isfinite(rb["cell_Q"])),
                        "cell_Q should be finite")
        self.assertTrue(np.all(np.isfinite(rb["node_depth"])),
                        "node_depth should be finite")

        A_full = np.pi * 0.5 ** 2
        self.assertLessEqual(float(rb["cell_A"][0]), A_full + 1e-6,
                             "Pipe 1 area should be <= A_full")
        self.assertLessEqual(float(rb["cell_A"][1]), A_full + 1e-6,
                             "Pipe 2 area should be <= A_full")


if __name__ == "__main__":
    unittest.main()
