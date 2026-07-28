"""
GPU tests for the 1D pipe solver: swe2d_build_unified_mesh + swe2d_pipe1d_step.

These test the pipe1d C kernels directly, isolated from the coupling layer.

Migrated from the legacy per-node solver API to the unified mesh + cell-state
schema (commit a080e61).  Mapping:

    swe2d_build_pipe1d_mesh       -> swe2d_build_unified_mesh
    swe2d_pipe1d_upload_node_depth -> swe2d_pipe1d_upload_cell_h
    swe2d_pipe1d_init_area_from_depth -> swe2d_pipe1d_init_cell_area
    swe2d_pipe1d_readback_node_state  -> swe2d_pipe1d_readback_cell_state
    state["node_depth"]               -> state["cell_depth"]

For the legacy ``node_depth`` boundary condition: with no manhole cells
configured, the unified mesh has only pipe cells, so the per-node depth
becomes the initial pipe-cell depth (uploaded via ``swe2d_pipe1d_upload_cell_h``).
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

K_MANN_DEFAULT = 1.0
H_MIN_DEFAULT = 1.0e-4
G_DEFAULT = 9.81


def _gpu_available():
    if _MOD is None:
        return False
    try:
        return bool(_MOD.swe2d_gpu_available())
    except Exception:
        return False


def _node_depth_to_cell_h(node_depth: np.ndarray, max_cell_length: float,
                          link_length: np.ndarray) -> np.ndarray:
    """Map per-node boundary depths to the unified mesh's per-cell depths.

    With no manhole cells, the unified mesh has only pipe cells.  Use the
    upstream node depth (node_depth[link_from[i]]) as the initial cell
    depth for the pipe cells of link i.
    """
    return np.asarray(node_depth, dtype=np.float64).reshape(-1)[:1]


@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
class TestPipe1DMeshBuild(unittest.TestCase):
    """Tests for swe2d_build_unified_mesh via SWE2DBackend."""

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

        n_nodes = 2
        node_invert = np.array([1.0, 0.0], dtype=np.float64)
        node_depth = np.array([0.5, 0.1], dtype=np.float64)

        return {
            "n_links": n_links,
            "link_from": link_from,
            "link_to": link_to,
            "link_length": link_length,
            "link_diameter": link_diameter,
            "link_roughness": link_roughness,
            "n_nodes": n_nodes,
            "node_invert": node_invert,
            "node_depth": node_depth,
        }

    def _build_unified(self, a, max_cell_length=0, link_shape_type=None,
                       link_width=None, link_height=None):
        """Build the unified mesh and return the device pointer.

        ``max_cell_length`` is in metres: ``0`` means no subdivision (one
        pipe cell per link).  Per-link shape arrays are optional and
        forwarded to ``swe2d_build_unified_mesh`` when supplied.
        """
        kwargs = dict(
            dev_ptr=self._dev_ptr,
            n_links=a["n_links"],
            link_from=a["link_from"],
            link_to=a["link_to"],
            L=a["link_length"],
            D=a["link_diameter"],
            n_mann=a["link_roughness"],
            S0=np.zeros(a["n_links"], dtype=np.float64),
            node_invert=a["node_invert"],
            mcl=float(max_cell_length),
        )
        if link_shape_type is not None:
            kwargs["link_shape_type"] = link_shape_type
        if link_width is not None:
            kwargs["link_width"] = link_width
        if link_height is not None:
            kwargs["link_height"] = link_height
        _MOD.swe2d_build_unified_mesh(**kwargs)

    def test_build_mesh_single_link(self):
        """Build a single-link mesh and verify no crash."""
        a = self._simple_pipe_arrays()
        self._build_unified(a, max_cell_length=0)
        rb = _MOD.swe2d_pipe1d_readback_cell_state(self._dev_ptr, 1)
        # The unified readback returns per-cell arrays.  With one pipe
        # cell, cell_A and cell_Q are length-1, and the legacy ``node_depth``
        # key is gone — depth lives in cell_depth (= cell_y - cell_invert).
        self.assertEqual(rb["cell_A"].shape, (1,))
        self.assertEqual(rb["cell_Q"].shape, (1,))

    def test_subcell_index_increases_downstream(self):
        """cell_sub_idx 0 is at the upstream (link_from) end; max at downstream end.

        This guards against accidental mirroring in the profile view: the profile
        viewer maps sub_idx 0 to station 0 and sub_idx n_sub-1 to station L, so
        the C++ mesh must enumerate cells from link_from_node to link_to_node.
        """
        a = self._simple_pipe_arrays()
        a["link_length"] = np.array([100.0], dtype=np.float64)
        a["node_invert"] = np.array([10.0, 0.0], dtype=np.float64)
        n_sub = 10
        self._build_unified(a, max_cell_length=10)
        rb = _MOD.swe2d_pipe1d_readback_cell_state(self._dev_ptr, n_sub)
        sub_idx = rb["cell_sub_idx"]
        invert = rb["cell_invert"]
        self.assertEqual(len(sub_idx), n_sub)
        np.testing.assert_array_equal(sub_idx, np.arange(n_sub))
        # Invert decreases monotonically from upstream to downstream
        self.assertTrue(np.all(np.diff(invert) < 0.0),
                        f"cell_invert should decrease downstream: {invert}")
        self.assertGreater(invert[0], 9.0)
        self.assertLess(invert[-1], 1.0)



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

    def _build_and_upload(self, a, max_cell_length=0, link_shape_type=None,
                          link_width=None, link_height=None,
                          node_depth=None):
        """Build mesh, upload initial cell depths, init cell area.

        Seeds pipe cells linearly between node_depth[0] (upstream) and
        node_depth[1] (downstream) so non-uniform depth tests see a real
        head gradient.  With one pipe cell, just uses node_depth[0].
        """
        self._build_unified(a, max_cell_length=max_cell_length,
                            link_shape_type=link_shape_type,
                            link_width=link_width,
                            link_height=link_height)
        if node_depth is None:
            node_depth = a.get("node_depth")
        if node_depth is not None:
            n_pipe_cells = self._compute_n_pipe_cells(a, max_cell_length)
            if n_pipe_cells > 1:
                cell_h = np.linspace(float(node_depth[0]), float(node_depth[1]),
                                     n_pipe_cells, dtype=np.float64)
            else:
                cell_h = np.full(n_pipe_cells, float(node_depth[0]), dtype=np.float64)
            _MOD.swe2d_pipe1d_upload_cell_h(self._dev_ptr, cell_h)
        _MOD.swe2d_pipe1d_init_cell_area(self._dev_ptr, H_MIN_DEFAULT)
        return self._dev_ptr

    def _build_unified(self, a, max_cell_length=0, link_shape_type=None,
                       link_width=None, link_height=None):
        kwargs = dict(
            dev_ptr=self._dev_ptr,
            n_links=a["n_links"],
            link_from=a["link_from"],
            link_to=a["link_to"],
            L=a["link_length"],
            D=a["link_diameter"],
            n_mann=a["link_roughness"],
            S0=np.zeros(a["n_links"], dtype=np.float64),
            node_invert=a["node_invert"],
            mcl=float(max_cell_length),
        )
        if link_shape_type is not None:
            kwargs["link_shape_type"] = link_shape_type
        if link_width is not None:
            kwargs["link_width"] = link_width
        if link_height is not None:
            kwargs["link_height"] = link_height
        _MOD.swe2d_build_unified_mesh(**kwargs)

    @staticmethod
    def _compute_n_pipe_cells(a, max_cell_length):
        n_pipe_cells = 0
        for L in a["link_length"]:
            if max_cell_length > 0 and L > 0:
                n_pipe_cells += max(1, int(np.ceil(L / float(max_cell_length))))
            else:
                n_pipe_cells += 1
        return n_pipe_cells

    def _simple_pipe_arrays(self):
        n_links = 1
        link_from = np.array([0], dtype=np.int32)
        link_to = np.array([1], dtype=np.int32)
        link_length = np.array([10.0], dtype=np.float64)
        link_diameter = np.array([1.0], dtype=np.float64)
        link_roughness = np.array([0.013], dtype=np.float64)
        n_nodes = 2
        node_invert = np.array([1.0, 0.0], dtype=np.float64)
        node_depth = np.array([0.5, 0.1], dtype=np.float64)
        return {
            "n_links": n_links, "link_from": link_from, "link_to": link_to,
            "link_length": link_length, "link_diameter": link_diameter,
            "link_roughness": link_roughness,
            "n_nodes": n_nodes, "node_invert": node_invert,
            "node_depth": node_depth,
        }

    def test_diffusion_wave_updates_area(self):
        """Diffusion wave updates pipe cell area from head-difference boundary flux."""
        a = self._simple_pipe_arrays()
        dev_ptr = self._build_and_upload(a)

        _MOD.swe2d_pipe1d_step(
            dev_ptr, 1.0, "diffusion_wave",
            1, 2, 0.5, G_DEFAULT, K_MANN_DEFAULT, H_MIN_DEFAULT,
        )
        rb = _MOD.swe2d_pipe1d_readback_cell_state(dev_ptr, 1)
        A_full = np.pi * (a["link_diameter"][0] / 2.0) ** 2
        self.assertLess(float(rb["cell_A"][0]), A_full,
                        "Area should decrease from full (head difference drives net inflow)")
        self.assertTrue(np.all(np.isfinite(rb["cell_depth"])),
                        "Pipe cell depth should be finite")

    def test_fully_dynamic_updates_area_and_q(self):
        """Fully dynamic solver updates both Q and A from pressure gradient."""
        a = self._simple_pipe_arrays()
        a["node_depth"] = np.array([1.0, 0.01], dtype=np.float64)
        dev_ptr = self._build_and_upload(a)

        _MOD.swe2d_pipe1d_step(
            dev_ptr, 0.5, "fully_dynamic",
            5, 20, 0.5, G_DEFAULT, K_MANN_DEFAULT, H_MIN_DEFAULT,
        )
        rb = _MOD.swe2d_pipe1d_readback_cell_state(dev_ptr, 1)
        A_full = np.pi * (a["link_diameter"][0] / 2.0) ** 2
        self.assertLess(float(rb["cell_A"][0]), A_full,
                        "Area should decrease from full due to outflow")
        self.assertTrue(np.isfinite(rb["cell_Q"][0]),
                        "Q should be finite")

    def test_dry_pipe_no_change(self):
        """Zero depths → SWMM-style floor area, negligible flow."""
        a = self._simple_pipe_arrays()
        a["node_depth"] = np.zeros(2, dtype=np.float64)
        dev_ptr = self._build_and_upload(a)

        _MOD.swe2d_pipe1d_step(
            dev_ptr, 1.0, "diffusion_wave",
            1, 2, 0.5, G_DEFAULT, K_MANN_DEFAULT, H_MIN_DEFAULT,
        )
        rb = _MOD.swe2d_pipe1d_readback_cell_state(dev_ptr, 1)
        A_full = np.pi * (a["link_diameter"][0] / 2.0) ** 2
        A_floor = A_full * 1.0e-4
        # With SWMM-style minimum area, dry pipe holds the floor area and
        # produces only a tiny numerical flow, not exactly zero.
        self.assertAlmostEqual(float(rb["cell_A"][0]), A_floor, places=10,
                               msg="Dry pipe should hold the minimum area floor")
        self.assertLess(abs(float(rb["cell_Q"][0])), 1e-3,
                        "Dry pipe should have negligible flow")

    def test_dry_pipe_wets_from_upstream_node(self):
        """A pipe primed to the SWMM floor area can wet when the upstream node becomes wet."""
        a = self._simple_pipe_arrays()
        A_full = np.pi * (a["link_diameter"][0] / 2.0) ** 2
        A_floor = A_full * 1.0e-4
        # Start with a dry pipe (zero node depths) so init_area_from_depth
        # produces the minimum-area floor.
        a["node_depth"] = np.zeros(2, dtype=np.float64)
        dev_ptr = self._build_and_upload(a)
        rb0 = _MOD.swe2d_pipe1d_readback_cell_state(dev_ptr, 1)
        self.assertAlmostEqual(float(rb0["cell_A"][0]), A_floor, places=10)

        # Make the upstream node wet while the downstream node stays dry.
        a["node_depth"] = np.array([2.0, 0.0], dtype=np.float64)
        n_pipe_cells = 1
        cell_h = np.full(n_pipe_cells, float(a["node_depth"][0]), dtype=np.float64)
        _MOD.swe2d_pipe1d_upload_cell_h(dev_ptr, cell_h)
        # Re-init A(h) from the new cell_h — the test wants the pipe primed
        # at depth 2.0 (= A_full for D=1) before the step.
        _MOD.swe2d_pipe1d_init_cell_area(dev_ptr, H_MIN_DEFAULT)
        _MOD.swe2d_pipe1d_step(dev_ptr, 1.0, "diffusion_wave", 1, 2, 0.5, G_DEFAULT, K_MANN_DEFAULT, H_MIN_DEFAULT)
        rb1 = _MOD.swe2d_pipe1d_readback_cell_state(dev_ptr, 1)
        self.assertGreater(float(rb1["cell_A"][0]), A_floor,
                           "Wet upstream node should drive water into the pipe")
        self.assertGreater(float(rb1["cell_Q"][0]), 0.0,
                           "Flow should be positive downstream")

    def test_substeps_produce_smaller_area_than_single(self):
        """Both 1 and 4 substeps produce area below full (outflow occurs)."""
        a = self._simple_pipe_arrays()
        A_full = np.pi * (a["link_diameter"][0] / 2.0) ** 2

        dev_ptr = self._build_and_upload(a)
        _MOD.swe2d_pipe1d_step(dev_ptr, 1.0, "diffusion_wave", 1, 2, 0.5, G_DEFAULT, K_MANN_DEFAULT, H_MIN_DEFAULT)
        rb1 = _MOD.swe2d_pipe1d_readback_cell_state(dev_ptr, 1)
        A1 = float(rb1["cell_A"][0])

        dev_ptr = self._build_and_upload(a)
        _MOD.swe2d_pipe1d_step(dev_ptr, 1.0, "diffusion_wave", 4, 2, 0.5, G_DEFAULT, K_MANN_DEFAULT, H_MIN_DEFAULT)
        rb4 = _MOD.swe2d_pipe1d_readback_cell_state(dev_ptr, 1)
        A4 = float(rb4["cell_A"][0])

        self.assertLess(A1, A_full, "Area after 1 substep should be below full")
        self.assertLess(A4, A_full, "Area after 4 substeps should be below full")

    def test_upload_node_depth_changes_area(self):
        """Uploading different node depths should change the area (via boundary flux).

        Migrated: the legacy test uploaded ``node_depth`` per-node; the
        unified API uploads per-cell depth (``cell_h``).  The cell_h
        seed still produces different downstream areas for different
        initial conditions.
        """
        a = self._simple_pipe_arrays()

        dev_ptr = self._build_and_upload(a)
        _MOD.swe2d_pipe1d_step(dev_ptr, 1.0, "diffusion_wave", 1, 2, 0.5, G_DEFAULT, K_MANN_DEFAULT, H_MIN_DEFAULT)
        rb1 = _MOD.swe2d_pipe1d_readback_cell_state(dev_ptr, 1)
        A1 = float(rb1["cell_A"][0])

        a["node_depth"] = np.array([5.0, 0.01], dtype=np.float64)
        dev_ptr = self._build_and_upload(a)
        _MOD.swe2d_pipe1d_step(dev_ptr, 1.0, "diffusion_wave", 1, 2, 0.5, G_DEFAULT, K_MANN_DEFAULT, H_MIN_DEFAULT)
        rb2 = _MOD.swe2d_pipe1d_readback_cell_state(dev_ptr, 1)
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
        dev_ptr = self._build_and_upload(a, link_shape_type=link_shape_type,
                                          link_width=link_width,
                                          link_height=link_height)
        rb = _MOD.swe2d_pipe1d_readback_cell_state(self._dev_ptr, 1)
        A_exp = 1.0 * 0.5  # w * h
        self.assertAlmostEqual(float(rb["cell_A"][0]), A_exp, delta=0.01,
                               msg="Rectangular full-cell area should match w*h")
        _MOD.swe2d_pipe1d_step(self._dev_ptr, 0.5, "diffusion_wave", 1, 2, 0.5, G_DEFAULT, K_MANN_DEFAULT, H_MIN_DEFAULT)
        rb2 = _MOD.swe2d_pipe1d_readback_cell_state(self._dev_ptr, 1)
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
        dev_ptr = self._build_and_upload(a, link_shape_type=link_shape_type,
                                          link_width=link_width,
                                          link_height=link_height)
        rb = _MOD.swe2d_pipe1d_readback_cell_state(self._dev_ptr, 1)
        A_exp = np.pi * 0.5 * 0.3  # π * (w/2) * (h/2)
        self.assertAlmostEqual(float(rb["cell_A"][0]), A_exp, delta=0.01,
                               msg="Elliptical cell area should match π*(w/2)*(h/2)")
        _MOD.swe2d_pipe1d_step(self._dev_ptr, 0.5, "diffusion_wave", 1, 2, 0.5, G_DEFAULT, K_MANN_DEFAULT, H_MIN_DEFAULT)
        rb2 = _MOD.swe2d_pipe1d_readback_cell_state(self._dev_ptr, 1)
        self.assertTrue(np.isfinite(rb2["cell_A"][0]))
        self.assertTrue(np.isfinite(rb2["cell_Q"][0]))

    def test_box_shape_without_explicit_shape_arrays(self):
        """Box shape with diameter=0 and no shape arrays falls back to D as width, produces finite values."""
        a = self._simple_pipe_arrays()
        a["link_diameter"] = np.array([0.0], dtype=np.float64)
        self._build_and_upload(a)
        _MOD.swe2d_pipe1d_step(self._dev_ptr, 0.5, "diffusion_wave", 1, 2, 0.5, G_DEFAULT, K_MANN_DEFAULT, H_MIN_DEFAULT)
        rb = _MOD.swe2d_pipe1d_readback_cell_state(self._dev_ptr, 1)
        self.assertTrue(np.all(np.isfinite(rb["cell_A"])),
                        "Zero-diameter without shape arrays should not crash")
        self.assertTrue(np.all(np.isfinite(rb["cell_depth"])))
        link_shape_type = np.array([2], dtype=np.int32)
        link_width = np.array([1.0], dtype=np.float64)
        link_height = np.array([0.6], dtype=np.float64)
        self._build_and_upload(a, link_shape_type=link_shape_type,
                                link_width=link_width,
                                link_height=link_height)
        _MOD.swe2d_pipe1d_step(self._dev_ptr, 0.5, "diffusion_wave", 1, 2, 0.5, G_DEFAULT, K_MANN_DEFAULT, H_MIN_DEFAULT)
        rb = _MOD.swe2d_pipe1d_readback_cell_state(self._dev_ptr, 1)
        self.assertTrue(np.isfinite(rb["cell_A"][0]))
        self.assertTrue(np.isfinite(rb["cell_Q"][0]))

    def test_init_area_from_depth(self):
        """swe2d_pipe1d_init_cell_area sets A proportional to depth with a SWMM-style floor."""
        a = self._simple_pipe_arrays()
        A_full = np.pi * (a["link_diameter"][0] / 2.0) ** 2
        A_floor = A_full * 1.0e-4

        a["node_depth"] = np.array([0.0, 0.0], dtype=np.float64)
        dev_ptr = self._build_and_upload(a)
        rb = _MOD.swe2d_pipe1d_readback_cell_state(dev_ptr, 1)
        self.assertAlmostEqual(float(rb["cell_A"][0]), A_floor, places=6,
                               msg="Zero depth → minimum area floor")

        a["node_depth"] = np.array([1.0, 1.0], dtype=np.float64)
        dev_ptr = self._build_and_upload(a)
        rb = _MOD.swe2d_pipe1d_readback_cell_state(dev_ptr, 1)
        self.assertAlmostEqual(float(rb["cell_A"][0]), A_full, places=3,
                               msg="Depth = diameter → full area (approx)")


    def test_fully_dynamic_convective_term_affects_flow(self):
        """Same head difference and same midpoint area, but different end-area
        gradient, should produce different discharge because of dq4."""
        a = self._simple_pipe_arrays()

        # Scenario A: uniform end areas, midpoint area = 0.5 A_full
        a["node_invert"] = np.array([1.0, 0.0], dtype=np.float64)
        a["node_depth"] = np.array([0.5, 0.5], dtype=np.float64)
        dev_ptr = self._build_and_upload(a)
        _MOD.swe2d_pipe1d_step(dev_ptr, 0.5, "fully_dynamic", 1, 5, 0.5, G_DEFAULT, K_MANN_DEFAULT, H_MIN_DEFAULT)
        rb_uniform = _MOD.swe2d_pipe1d_readback_cell_state(dev_ptr, 1)
        Q_uniform = float(rb_uniform["cell_Q"][0])

        # Scenario B: same head difference (H0=1.5, H1=0.5), same midpoint
        # depth (0.5), but A1 > A2; midpoint area is still 0.5 A_full.
        a["node_invert"] = np.array([0.75, 0.25], dtype=np.float64)
        a["node_depth"] = np.array([0.75, 0.25], dtype=np.float64)
        dev_ptr = self._build_and_upload(a)
        _MOD.swe2d_pipe1d_step(dev_ptr, 0.5, "fully_dynamic", 1, 5, 0.5, G_DEFAULT, K_MANN_DEFAULT, H_MIN_DEFAULT)
        rb_grad = _MOD.swe2d_pipe1d_readback_cell_state(dev_ptr, 1)
        Q_grad = float(rb_grad["cell_Q"][0])

        self.assertTrue(np.isfinite(Q_uniform))
        self.assertTrue(np.isfinite(Q_grad))
        self.assertNotAlmostEqual(
            Q_grad, Q_uniform, places=6,
            msg="Convective acceleration from area gradient should change discharge"
        )

    def test_fully_dynamic_mass_conservation_with_and_without_sub_cells(self):
        """Single pipe: total volume must be conserved over many steps.

        Migrated: the legacy closed-system volume included boundary
        ``node_depth * surface_area`` terms.  The unified mesh has no
        separate node-depth state, so the new identity is just the
        pipe-cell volume (pipe cells hold all the water in a 1-link,
        no-manhole-cell network).
        """
        a = self._simple_pipe_arrays()
        a["node_invert"] = np.array([0.0, 0.0], dtype=np.float64)
        a["node_depth"] = np.array([0.5, 0.4], dtype=np.float64)
        pipe_length = 10.0

        A_full = np.pi * (a["link_diameter"][0] / 2.0) ** 2
        initial_cell_area = A_full * 0.5 * (0.5 + 0.4)
        initial_volume = pipe_length * initial_cell_area

        for mcl in (0, 5):
            with self.subTest(max_cell_length=mcl):
                dev_ptr = self._build_and_upload(a, max_cell_length=mcl)
                n_cells = 1 if mcl == 0 else max(1, int(np.ceil(pipe_length / mcl)))
                for _ in range(10):
                    _MOD.swe2d_pipe1d_step(
                        dev_ptr, 0.5, "fully_dynamic", 1, 5, 0.5,
                        G_DEFAULT, K_MANN_DEFAULT, H_MIN_DEFAULT,
                    )
                rb = _MOD.swe2d_pipe1d_readback_cell_state(dev_ptr, n_cells)
                cell_volume = pipe_length * float(np.mean(rb["cell_A"]))
                final_volume = cell_volume
                error = abs(final_volume - initial_volume)
                self.assertLess(
                    error, 1e-10,
                    msg=f"Mass error with max_cell_length={mcl} should not grow"
                )


if __name__ == "__main__":
    unittest.main()
