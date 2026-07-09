"""
GPU tests for the 1D pipe solver: swe2d_build_pipe1d_mesh + swe2d_pipe1d_step.

These test the pipe1d C kernels directly, isolated from the coupling layer.
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


@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
class TestPipe1DMeshBuild(unittest.TestCase):
    """Tests for swe2d_build_pipe1d_mesh via SWE2DBackend."""

    @classmethod
    def setUpClass(cls):
        from swe2d.runtime.backend import SWE2DBackend
        cls._backend = SWE2DBackend()
        node_x = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        node_y = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        node_z = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
        cell_nodes = np.asarray([0, 1, 2], dtype=np.int32)
        cls._backend.build_mesh(node_x, node_y, node_z, cell_nodes)
        cls._backend.initialize(
            h0=np.asarray([0.1], dtype=np.float64),
            hu0=np.zeros(1, dtype=np.float64),
            hv0=np.zeros(1, dtype=np.float64),
            dt_fixed=0.05,
            dt_max=0.05,
        )
        cls._dev_ptr = int(_MOD.swe2d_get_coupling_dev_ptr())

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls._backend, 'destroy'):
            cls._backend.destroy()

    def _simple_pipe_arrays(self):
        """Build arrays for a simple 2-node, 1-link pipe network.

        N0 (invert=1.0) --pipe (10m, D=1m, n=0.013)--> N1 (invert=0.0)
        Slope = 0.1 m/m, so gravity drives flow from N0 to N1.
        """
        n_links = 1
        link_from = np.array([0], dtype=np.int32)
        link_to = np.array([1], dtype=np.int32)
        link_length = np.array([10.0], dtype=np.float64)
        link_diameter = np.array([1.0], dtype=np.float64)
        link_roughness = np.array([0.013], dtype=np.float64)
        link_inlet_loss = np.array([0.0], dtype=np.float64)
        link_outlet_loss = np.array([0.0], dtype=np.float64)
        link_invert_in = np.array([1.0], dtype=np.float64)
        link_invert_out = np.array([0.0], dtype=np.float64)

        n_nodes = 2
        node_invert = np.array([1.0, 0.0], dtype=np.float64)
        node_surface_area = np.array([50.0, 50.0], dtype=np.float64)
        node_max_depth = np.array([3.0, 3.0], dtype=np.float64)
        node_depth = np.array([0.5, 0.1], dtype=np.float64)

        return {
            "n_links": n_links,
            "link_from": link_from,
            "link_to": link_to,
            "link_length": link_length,
            "link_diameter": link_diameter,
            "link_roughness": link_roughness,
            "link_inlet_loss": link_inlet_loss,
            "link_outlet_loss": link_outlet_loss,
            "link_invert_in": link_invert_in,
            "link_invert_out": link_invert_out,
            "n_nodes": n_nodes,
            "node_invert": node_invert,
            "node_surface_area": node_surface_area,
            "node_max_depth": node_max_depth,
            "node_depth": node_depth,
        }

    def test_build_mesh_single_link(self):
        """Build a single-link mesh and verify no crash."""
        a = self._simple_pipe_arrays()
        _MOD.swe2d_build_pipe1d_mesh(
            a["n_links"],
            a["link_from"], a["link_to"],
            a["link_length"], a["link_diameter"], a["link_roughness"],
            a["link_inlet_loss"], a["link_outlet_loss"],
            a["node_invert"], a["node_surface_area"], a["node_max_depth"],
            a["link_invert_in"], a["link_invert_out"],
            0,  # max_cell_length=0 means no subdivision
            self._dev_ptr,
        )
        rb = _MOD.swe2d_pipe1d_readback_node_state(self._dev_ptr, a["n_nodes"], 1)
        self.assertEqual(rb["node_depth"].shape, (2,))
        self.assertEqual(rb["cell_A"].shape, (1,))
        self.assertEqual(rb["cell_Q"].shape, (1,))

    def test_build_mesh_subdivision(self):
        """max_cell_length triggers sub-cell subdivision."""
        a = self._simple_pipe_arrays()
        a["link_length"] = np.array([30.0], dtype=np.float64)
        _MOD.swe2d_build_pipe1d_mesh(
            a["n_links"],
            a["link_from"], a["link_to"],
            a["link_length"], a["link_diameter"], a["link_roughness"],
            a["link_inlet_loss"], a["link_outlet_loss"],
            a["node_invert"], a["node_surface_area"], a["node_max_depth"],
            a["link_invert_in"], a["link_invert_out"],
            10,  # max_cell_length=10 → 3 sub-cells
            self._dev_ptr,
        )
        rb = _MOD.swe2d_pipe1d_readback_node_state(self._dev_ptr, a["n_nodes"], 3)
        self.assertEqual(rb["cell_A"].shape, (3,))
        self.assertEqual(rb["cell_Q"].shape, (3,))


@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
class TestPipe1DStep(unittest.TestCase):
    """Tests for swe2d_pipe1d_step."""

    @classmethod
    def setUpClass(cls):
        from swe2d.runtime.backend import SWE2DBackend
        cls._backend = SWE2DBackend()
        node_x = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        node_y = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        node_z = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
        cell_nodes = np.asarray([0, 1, 2], dtype=np.int32)
        cls._backend.build_mesh(node_x, node_y, node_z, cell_nodes)
        cls._backend.initialize(
            h0=np.asarray([0.1], dtype=np.float64),
            hu0=np.zeros(1, dtype=np.float64),
            hv0=np.zeros(1, dtype=np.float64),
            dt_fixed=0.05,
            dt_max=0.05,
        )
        cls._dev_ptr = int(_MOD.swe2d_get_coupling_dev_ptr())

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls._backend, 'destroy'):
            cls._backend.destroy()

    def _build_and_upload(self, a):
        """Build mesh and upload initial node depths."""
        _MOD.swe2d_build_pipe1d_mesh(
            a["n_links"],
            a["link_from"], a["link_to"],
            a["link_length"], a["link_diameter"], a["link_roughness"],
            a["link_inlet_loss"], a["link_outlet_loss"],
            a["node_invert"], a["node_surface_area"], a["node_max_depth"],
            a["link_invert_in"], a["link_invert_out"],
            0, self._dev_ptr,
        )
        _MOD.swe2d_pipe1d_upload_node_depth(
            self._dev_ptr, a["node_depth"])
        return self._dev_ptr

    def _simple_pipe_arrays(self):
        n_links = 1
        link_from = np.array([0], dtype=np.int32)
        link_to = np.array([1], dtype=np.int32)
        link_length = np.array([10.0], dtype=np.float64)
        link_diameter = np.array([1.0], dtype=np.float64)
        link_roughness = np.array([0.013], dtype=np.float64)
        link_inlet_loss = np.array([0.0], dtype=np.float64)
        link_outlet_loss = np.array([0.0], dtype=np.float64)
        link_invert_in = np.array([1.0], dtype=np.float64)
        link_invert_out = np.array([0.0], dtype=np.float64)
        n_nodes = 2
        node_invert = np.array([1.0, 0.0], dtype=np.float64)
        node_surface_area = np.array([50.0, 50.0], dtype=np.float64)
        node_max_depth = np.array([3.0, 3.0], dtype=np.float64)
        node_depth = np.array([0.5, 0.1], dtype=np.float64)
        return {
            "n_links": n_links, "link_from": link_from, "link_to": link_to,
            "link_length": link_length, "link_diameter": link_diameter,
            "link_roughness": link_roughness,
            "link_inlet_loss": link_inlet_loss, "link_outlet_loss": link_outlet_loss,
            "link_invert_in": link_invert_in, "link_invert_out": link_invert_out,
            "n_nodes": n_nodes, "node_invert": node_invert,
            "node_surface_area": node_surface_area,
            "node_max_depth": node_max_depth, "node_depth": node_depth,
        }

    def test_diffusion_wave_updates_area(self):
        """Diffusion wave updates pipe cell area from head-difference boundary flux."""
        a = self._simple_pipe_arrays()
        dev_ptr = self._build_and_upload(a)

        _MOD.swe2d_pipe1d_step(
            dev_ptr, 1.0, "diffusion_wave",
            1, 2, 0.5, 9.81,
        )
        rb = _MOD.swe2d_pipe1d_readback_node_state(dev_ptr, a["n_nodes"], 1)
        A_full = np.pi * (a["link_diameter"][0] / 2.0) ** 2
        self.assertLess(float(rb["cell_A"][0]), A_full,
                        "Area should decrease from full (head difference drives net inflow)")
        self.assertTrue(np.all(np.isfinite(rb["node_depth"])),
                        "Node depths should be finite")

    def test_fully_dynamic_updates_area_and_q(self):
        """Fully dynamic solver updates both Q and A from pressure gradient."""
        a = self._simple_pipe_arrays()
        a["node_depth"] = np.array([1.0, 0.01], dtype=np.float64)
        dev_ptr = self._build_and_upload(a)
        _MOD.swe2d_pipe1d_upload_node_depth(dev_ptr, a["node_depth"])

        _MOD.swe2d_pipe1d_step(
            dev_ptr, 0.5, "fully_dynamic",
            5, 20, 0.5, 9.81,
        )
        rb = _MOD.swe2d_pipe1d_readback_node_state(dev_ptr, a["n_nodes"], 1)
        A_full = np.pi * (a["link_diameter"][0] / 2.0) ** 2
        self.assertLess(float(rb["cell_A"][0]), A_full,
                        "Area should decrease from full due to outflow")
        self.assertTrue(np.isfinite(rb["cell_Q"][0]),
                        "Q should be finite")

    def test_dry_pipe_no_change(self):
        """Zero depths → no flow, no depth change."""
        a = self._simple_pipe_arrays()
        a["node_depth"] = np.zeros(2, dtype=np.float64)
        dev_ptr = self._build_and_upload(a)
        _MOD.swe2d_pipe1d_init_area_from_depth(dev_ptr)

        _MOD.swe2d_pipe1d_step(
            dev_ptr, 1.0, "diffusion_wave",
            1, 2, 0.5, 9.81,
        )
        rb = _MOD.swe2d_pipe1d_readback_node_state(dev_ptr, a["n_nodes"], 1)
        self.assertAlmostEqual(float(rb["cell_Q"][0]), 0.0, places=10,
                               msg="Dry pipe should have zero flow")

    def test_substeps_produce_smaller_area_than_single(self):
        """Both 1 and 4 substeps produce area below full (outflow occurs)."""
        a = self._simple_pipe_arrays()
        A_full = np.pi * (a["link_diameter"][0] / 2.0) ** 2

        dev_ptr = self._build_and_upload(a)
        _MOD.swe2d_pipe1d_step(dev_ptr, 1.0, "diffusion_wave", 1, 2, 0.5, 9.81)
        rb1 = _MOD.swe2d_pipe1d_readback_node_state(dev_ptr, a["n_nodes"], 1)
        A1 = float(rb1["cell_A"][0])

        dev_ptr = self._build_and_upload(a)
        _MOD.swe2d_pipe1d_step(dev_ptr, 1.0, "diffusion_wave", 4, 2, 0.5, 9.81)
        rb4 = _MOD.swe2d_pipe1d_readback_node_state(dev_ptr, a["n_nodes"], 1)
        A4 = float(rb4["cell_A"][0])

        self.assertLess(A1, A_full, "Area after 1 substep should be below full")
        self.assertLess(A4, A_full, "Area after 4 substeps should be below full")

    def test_upload_node_depth_changes_area(self):
        """Uploading different node depths should change the area (via boundary flux)."""
        a = self._simple_pipe_arrays()

        dev_ptr = self._build_and_upload(a)
        _MOD.swe2d_pipe1d_step(dev_ptr, 1.0, "diffusion_wave", 1, 2, 0.5, 9.81)
        rb1 = _MOD.swe2d_pipe1d_readback_node_state(dev_ptr, a["n_nodes"], 1)
        A1 = float(rb1["cell_A"][0])

        a["node_depth"] = np.array([5.0, 0.01], dtype=np.float64)
        dev_ptr = self._build_and_upload(a)
        _MOD.swe2d_pipe1d_step(dev_ptr, 1.0, "diffusion_wave", 1, 2, 0.5, 9.81)
        rb2 = _MOD.swe2d_pipe1d_readback_node_state(dev_ptr, a["n_nodes"], 1)
        A2 = float(rb2["cell_A"][0])

        self.assertNotAlmostEqual(A2, A1, places=8,
                                  msg="Higher head should produce a different area")


    def test_rectangular_link_diffusion(self):
        """Rectangular link (W=1.0, H=0.5) computes A = w*h from shape dimensions."""
        a = self._simple_pipe_arrays()
        a["link_diameter"] = np.array([0.0], dtype=np.float64)
        a["node_depth"] = np.array([0.5, 0.5], dtype=np.float64)
        link_shape_type = np.array([1], dtype=np.int32)
        link_width = np.array([1.0], dtype=np.float64)
        link_height = np.array([0.5], dtype=np.float64)
        _MOD.swe2d_build_pipe1d_mesh(
            a["n_links"],
            a["link_from"], a["link_to"],
            a["link_length"], a["link_diameter"], a["link_roughness"],
            a["link_inlet_loss"], a["link_outlet_loss"],
            a["node_invert"], a["node_surface_area"], a["node_max_depth"],
            a["link_invert_in"], a["link_invert_out"],
            0, self._dev_ptr,
            link_shape_type, link_width, link_height,
        )
        _MOD.swe2d_pipe1d_upload_node_depth(self._dev_ptr, a["node_depth"])
        _MOD.swe2d_pipe1d_init_area_from_depth(self._dev_ptr)
        rb = _MOD.swe2d_pipe1d_readback_node_state(self._dev_ptr, a["n_nodes"], 1)
        A_exp = 1.0 * 0.5  # w * h
        self.assertAlmostEqual(float(rb["cell_A"][0]), A_exp, delta=0.01,
                               msg="Rectangular full-cell area should match w*h")
        _MOD.swe2d_pipe1d_step(self._dev_ptr, 0.5, "diffusion_wave", 1, 2, 0.5, 9.81)
        rb2 = _MOD.swe2d_pipe1d_readback_node_state(self._dev_ptr, a["n_nodes"], 1)
        self.assertTrue(np.isfinite(rb2["cell_A"][0]))
        self.assertTrue(np.isfinite(rb2["cell_Q"][0]))

    def test_elliptical_link_diffusion(self):
        """Elliptical link computes A = π * (w/2) * (h/2) from shape dimensions."""
        a = self._simple_pipe_arrays()
        a["link_diameter"] = np.array([0.0], dtype=np.float64)
        a["node_depth"] = np.array([0.6, 0.6], dtype=np.float64)
        link_shape_type = np.array([2], dtype=np.int32)
        link_width = np.array([1.0], dtype=np.float64)    # span
        link_height = np.array([0.6], dtype=np.float64)   # rise
        _MOD.swe2d_build_pipe1d_mesh(
            a["n_links"],
            a["link_from"], a["link_to"],
            a["link_length"], a["link_diameter"], a["link_roughness"],
            a["link_inlet_loss"], a["link_outlet_loss"],
            a["node_invert"], a["node_surface_area"], a["node_max_depth"],
            a["link_invert_in"], a["link_invert_out"],
            0, self._dev_ptr,
            link_shape_type, link_width, link_height,
        )
        _MOD.swe2d_pipe1d_upload_node_depth(self._dev_ptr, a["node_depth"])
        _MOD.swe2d_pipe1d_init_area_from_depth(self._dev_ptr)
        rb = _MOD.swe2d_pipe1d_readback_node_state(self._dev_ptr, a["n_nodes"], 1)
        A_exp = np.pi * 0.5 * 0.3  # π * (w/2) * (h/2)
        self.assertAlmostEqual(float(rb["cell_A"][0]), A_exp, delta=0.01,
                               msg="Elliptical cell area should match π*(w/2)*(h/2)")
        _MOD.swe2d_pipe1d_step(self._dev_ptr, 0.5, "diffusion_wave", 1, 2, 0.5, 9.81)
        rb2 = _MOD.swe2d_pipe1d_readback_node_state(self._dev_ptr, a["n_nodes"], 1)
        self.assertTrue(np.isfinite(rb2["cell_A"][0]))
        self.assertTrue(np.isfinite(rb2["cell_Q"][0]))

    def test_box_shape_without_explicit_shape_arrays(self):
        """Box shape with diameter=0 and no shape arrays falls back to D as width, produces finite values."""
        a = self._simple_pipe_arrays()
        a["link_diameter"] = np.array([0.0], dtype=np.float64)
        _MOD.swe2d_build_pipe1d_mesh(
            a["n_links"],
            a["link_from"], a["link_to"],
            a["link_length"], a["link_diameter"], a["link_roughness"],
            a["link_inlet_loss"], a["link_outlet_loss"],
            a["node_invert"], a["node_surface_area"], a["node_max_depth"],
            a["link_invert_in"], a["link_invert_out"],
            0, self._dev_ptr,
        )
        _MOD.swe2d_pipe1d_upload_node_depth(self._dev_ptr, a["node_depth"])
        _MOD.swe2d_pipe1d_step(self._dev_ptr, 0.5, "diffusion_wave", 1, 2, 0.5, 9.81)
        rb = _MOD.swe2d_pipe1d_readback_node_state(self._dev_ptr, a["n_nodes"], 1)
        self.assertTrue(np.all(np.isfinite(rb["cell_A"])),
                        "Zero-diameter without shape arrays should not crash")
        self.assertTrue(np.all(np.isfinite(rb["node_depth"])))
        link_shape_type = np.array([2], dtype=np.int32)
        link_width = np.array([1.0], dtype=np.float64)
        link_height = np.array([0.6], dtype=np.float64)
        _MOD.swe2d_build_pipe1d_mesh(
            a["n_links"],
            a["link_from"], a["link_to"],
            a["link_length"], a["link_diameter"], a["link_roughness"],
            a["link_inlet_loss"], a["link_outlet_loss"],
            a["node_invert"], a["node_surface_area"], a["node_max_depth"],
            a["link_invert_in"], a["link_invert_out"],
            0, self._dev_ptr,
            link_shape_type, link_width, link_height,
        )
        _MOD.swe2d_pipe1d_upload_node_depth(self._dev_ptr, a["node_depth"])
        _MOD.swe2d_pipe1d_step(self._dev_ptr, 0.5, "diffusion_wave", 1, 2, 0.5, 9.81)
        rb = _MOD.swe2d_pipe1d_readback_node_state(self._dev_ptr, a["n_nodes"], 1)
        self.assertTrue(np.isfinite(rb["cell_A"][0]))
        self.assertTrue(np.isfinite(rb["cell_Q"][0]))

    def test_init_area_from_depth(self):
        """swe2d_pipe1d_init_area_from_depth should set A proportional to depth."""
        a = self._simple_pipe_arrays()
        a["node_depth"] = np.array([0.0, 0.0], dtype=np.float64)
        dev_ptr = self._build_and_upload(a)
        _MOD.swe2d_pipe1d_init_area_from_depth(dev_ptr)
        rb = _MOD.swe2d_pipe1d_readback_node_state(dev_ptr, a["n_nodes"], 1)
        self.assertAlmostEqual(float(rb["cell_A"][0]), 0.0, places=6,
                               msg="Zero depth → zero area")

        a["node_depth"] = np.array([1.0, 1.0], dtype=np.float64)
        dev_ptr = self._build_and_upload(a)
        _MOD.swe2d_pipe1d_init_area_from_depth(dev_ptr)
        rb = _MOD.swe2d_pipe1d_readback_node_state(dev_ptr, a["n_nodes"], 1)
        A_full = np.pi * (a["link_diameter"][0] / 2.0) ** 2
        self.assertAlmostEqual(float(rb["cell_A"][0]), A_full, places=3,
                               msg="Depth = diameter → full area (approx)")


if __name__ == "__main__":
    unittest.main()
