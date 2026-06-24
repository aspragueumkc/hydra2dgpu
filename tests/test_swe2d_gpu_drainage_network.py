"""
GPU compute-sanitizer test: 1D drainage network CUDA paths.

Exercises these module-level functions that are NOT covered by the
core solver tests:
  - swe2d_gpu_drainage_step
  - swe2d_gpu_drainage_step_iterative
  - swe2d_gpu_upload_culvert_face_flux_params
  - swe2d_gpu_apply_culvert_face_flux
  - swe2d_gpu_fold_culvert_mass_to_source
  - swe2d_gpu_alloc_ext_struct_flux

Run with compute-sanitizer:
  compute-sanitizer --tool=memcheck python -m pytest tests/test_swe2d_gpu_drainage_network.py -v
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


class TestGPUDrainageStepComputeSanitizer(unittest.TestCase):
    """GPU drainage function coverage for compute-sanitizer."""

    @classmethod
    def setUpClass(cls):
        if _MOD is None:
            raise unittest.SkipTest("hydra_swe2d not available")
        if not hasattr(_MOD, "swe2d_gpu_drainage_step"):
            raise unittest.SkipTest("GPU drainage functions not compiled")
        if not bool(_MOD.swe2d_gpu_available()):
            raise unittest.SkipTest("CUDA GPU not available")

    # ── Simple 2-node, 1-link pipe network ──────────────────────────────
    def _simple_network_arrays(self):
        """Build arrays for a simple 2-node, 1-link drainage network.

        N0 (invert=0.0) --pipe--> N1 (invert=0.0)
        Inlet I0 from cell 0 to N0.
        """
        n_cells = 2
        cell_area = np.array([5.0, 5.0], dtype=np.float64)
        cell_wse = np.array([1.5, 1.0], dtype=np.float64)  # h + zb
        cell_bed = np.array([0.0, 0.0], dtype=np.float64)
        cell_depth = np.array([1.5, 1.0], dtype=np.float64)

        n_nodes = 2
        node_invert = np.array([0.0, 0.0], dtype=np.float64)
        node_max_depth = np.array([3.0, 3.0], dtype=np.float64)
        node_surface_area = np.array([10.0, 10.0], dtype=np.float64)
        node_depth = np.array([0.5, 0.1], dtype=np.float64)  # initial node ponding

        n_links = 1
        link_from = np.array([0], dtype=np.int32)
        link_to = np.array([1], dtype=np.int32)
        link_length = np.array([10.0], dtype=np.float64)
        link_roughness = np.array([0.013], dtype=np.float64)
        link_diameter = np.array([1.0], dtype=np.float64)
        link_max_flow = np.array([-1.0], dtype=np.float64)
        link_flow = np.array([0.0], dtype=np.float64)

        n_inlets = 1
        inlet_cell = np.array([0], dtype=np.int32)
        inlet_node = np.array([0], dtype=np.int32)
        inlet_crest = np.array([0.5], dtype=np.float64)
        inlet_width = np.array([1.0], dtype=np.float64)
        inlet_coeff = np.array([0.62], dtype=np.float64)
        inlet_max_capture = np.array([1.0], dtype=np.float64)

        n_outfalls = 0
        outfall_cell = np.empty(0, dtype=np.int32)
        outfall_node = np.empty(0, dtype=np.int32)
        outfall_invert = np.empty(0, dtype=np.float64)
        outfall_diameter = np.empty(0, dtype=np.float64)
        outfall_coeff = np.empty(0, dtype=np.float64)
        outfall_max_flow = np.empty(0, dtype=np.float64)
        outfall_zero_storage = np.empty(0, dtype=np.int32)

        n_pipe_ends = 0
        pipe_end_cell = np.empty(0, dtype=np.int32)
        pipe_end_node = np.empty(0, dtype=np.int32)
        pipe_end_invert = np.empty(0, dtype=np.float64)
        pipe_end_diameter = np.empty(0, dtype=np.float64)
        pipe_end_area = np.empty(0, dtype=np.float64)
        pipe_end_inlet_loss_k = np.empty(0, dtype=np.float64)
        pipe_end_outlet_loss_k = np.empty(0, dtype=np.float64)

        return {
            "cell_wse": cell_wse,
            "cell_area": cell_area,
            "cell_bed": cell_bed,
            "cell_depth": cell_depth,
            "node_invert": node_invert,
            "node_max_depth": node_max_depth,
            "node_surface_area": node_surface_area,
            "node_depth": node_depth,
            "link_from": link_from,
            "link_to": link_to,
            "link_length": link_length,
            "link_roughness": link_roughness,
            "link_diameter": link_diameter,
            "link_max_flow": link_max_flow,
            "link_flow": link_flow,
            "inlet_cell": inlet_cell,
            "inlet_node": inlet_node,
            "inlet_crest": inlet_crest,
            "inlet_width": inlet_width,
            "inlet_coeff": inlet_coeff,
            "inlet_max_capture": inlet_max_capture,
            "outfall_cell": outfall_cell,
            "outfall_node": outfall_node,
            "outfall_invert": outfall_invert,
            "outfall_diameter": outfall_diameter,
            "outfall_coeff": outfall_coeff,
            "outfall_max_flow": outfall_max_flow,
            "outfall_zero_storage": outfall_zero_storage,
            "pipe_end_cell": pipe_end_cell,
            "pipe_end_node": pipe_end_node,
            "pipe_end_invert": pipe_end_invert,
            "pipe_end_diameter": pipe_end_diameter,
            "pipe_end_area": pipe_end_area,
            "pipe_end_inlet_loss_k": pipe_end_inlet_loss_k,
            "pipe_end_outlet_loss_k": pipe_end_outlet_loss_k,
        }

    def test_drainage_step_simple_network(self):
        """swe2d_gpu_drainage_step with a simple pipe network (EGL mode)."""
        a = self._simple_network_arrays()

        nd_out, lf_out, diag = _MOD.swe2d_gpu_drainage_step(
            a["cell_wse"],
            a["cell_area"],
            a["node_invert"],
            a["node_max_depth"],
            a["node_surface_area"],
            a["link_from"],
            a["link_to"],
            a["link_length"],
            a["link_roughness"],
            a["link_diameter"],
            a["link_max_flow"],
            a["inlet_cell"],
            a["inlet_node"],
            a["inlet_crest"],
            a["inlet_width"],
            a["inlet_coeff"],
            a["inlet_max_capture"],
            a["outfall_cell"],
            a["outfall_node"],
            a["outfall_invert"],
            a["outfall_diameter"],
            a["outfall_coeff"],
            a["outfall_max_flow"],
            a["outfall_zero_storage"],
            a["pipe_end_cell"],
            a["pipe_end_node"],
            a["pipe_end_invert"],
            a["pipe_end_diameter"],
            a["pipe_end_area"],
            a["pipe_end_inlet_loss_k"],
            a["pipe_end_outlet_loss_k"],
            a["cell_depth"],
            a["node_depth"],
            a["link_flow"],
            1.0,   # dt_s
            9.81,  # gravity
            0,     # solver_mode (EGL)
            1.0e-3,  # head_deadband
            1.0,   # dynamic_flow_relaxation
        )

        self.assertEqual(nd_out.shape, (2,))
        self.assertEqual(lf_out.shape, (1,))
        # Node depths updated (may increase or decrease depending on
        # solver mode, inlet exchange direction, and pipe routing).
        self.assertTrue(np.all(np.isfinite(nd_out)),
                        "Node depths should be finite")
        self.assertIn("max_node_depth", diag)
        self.assertIn("max_link_flow", diag)

    def test_drainage_step_dynamic_mode(self):
        """swe2d_gpu_drainage_step with DYNAMIC solver mode."""
        a = self._simple_network_arrays()
        # Higher head difference to get measurable flow
        a["node_depth"] = np.array([2.0, 0.2], dtype=np.float64)

        nd_out, lf_out, diag = _MOD.swe2d_gpu_drainage_step(
            a["cell_wse"],
            a["cell_area"],
            a["node_invert"],
            a["node_max_depth"],
            a["node_surface_area"],
            a["link_from"],
            a["link_to"],
            a["link_length"],
            a["link_roughness"],
            a["link_diameter"],
            a["link_max_flow"],
            a["inlet_cell"],
            a["inlet_node"],
            a["inlet_crest"],
            a["inlet_width"],
            a["inlet_coeff"],
            a["inlet_max_capture"],
            a["outfall_cell"],
            a["outfall_node"],
            a["outfall_invert"],
            a["outfall_diameter"],
            a["outfall_coeff"],
            a["outfall_max_flow"],
            a["outfall_zero_storage"],
            a["pipe_end_cell"],
            a["pipe_end_node"],
            a["pipe_end_invert"],
            a["pipe_end_diameter"],
            a["pipe_end_area"],
            a["pipe_end_inlet_loss_k"],
            a["pipe_end_outlet_loss_k"],
            a["cell_depth"],
            a["node_depth"],
            a["link_flow"],
            1.0, 9.81,
            1,    # solver_mode (DYNAMIC)
            1.0e-3, 1.0,
        )

        self.assertEqual(nd_out.shape, (2,))
        self.assertEqual(lf_out.shape, (1,))
        # Flow should be positive (N0 → N1)
        self.assertGreaterEqual(float(lf_out[0]), 0.0)

    def test_drainage_step_iterative(self):
        """swe2d_gpu_drainage_step_iterative with substeps."""
        a = self._simple_network_arrays()
        a["node_depth"] = np.array([2.5, 0.1], dtype=np.float64)

        result = _MOD.swe2d_gpu_drainage_step_iterative(
            a["cell_bed"],
            a["cell_area"],
            a["node_invert"],
            a["node_max_depth"],
            a["node_surface_area"],
            a["link_from"],
            a["link_to"],
            a["link_length"],
            a["link_roughness"],
            a["link_diameter"],
            a["link_max_flow"],
            a["inlet_cell"],
            a["inlet_node"],
            a["inlet_crest"],
            a["inlet_width"],
            a["inlet_coeff"],
            a["inlet_max_capture"],
            a["outfall_cell"],
            a["outfall_node"],
            a["outfall_invert"],
            a["outfall_diameter"],
            a["outfall_coeff"],
            a["outfall_max_flow"],
            a["outfall_zero_storage"],
            a["pipe_end_cell"],
            a["pipe_end_node"],
            a["pipe_end_invert"],
            a["pipe_end_diameter"],
            a["pipe_end_area"],
            a["pipe_end_inlet_loss_k"],
            a["pipe_end_outlet_loss_k"],
            a["cell_depth"],
            a["node_depth"],
            a["link_flow"],
            1.0, 9.81,
            0,    # solver_mode (EGL)
            1.0e-3, 1.0,
            2,    # n_substeps
            2,    # implicit_iters
            0.5,  # coupling_relaxation
        )

        nd_out, lf_out, q_cell, diag = result
        self.assertEqual(nd_out.shape, (2,))
        self.assertEqual(lf_out.shape, (1,))
        self.assertEqual(q_cell.shape, (2,))
        self.assertIn("substeps_used", diag)
        self.assertIn("implicit_iters_used", diag)

    def test_drainage_step_inactive_fastpath(self):
        """Drainage step with no water — should hit inactive fastpath."""
        a = self._simple_network_arrays()
        # Set all depths to zero
        a["cell_depth"] = np.zeros(2, dtype=np.float64)
        a["node_depth"] = np.zeros(2, dtype=np.float64)
        a["link_flow"] = np.zeros(1, dtype=np.float64)
        a["cell_wse"] = a["cell_bed"].copy()

        nd_out, lf_out, diag = _MOD.swe2d_gpu_drainage_step(
            a["cell_wse"],
            a["cell_area"],
            a["node_invert"],
            a["node_max_depth"],
            a["node_surface_area"],
            a["link_from"],
            a["link_to"],
            a["link_length"],
            a["link_roughness"],
            a["link_diameter"],
            a["link_max_flow"],
            a["inlet_cell"],
            a["inlet_node"],
            a["inlet_crest"],
            a["inlet_width"],
            a["inlet_coeff"],
            a["inlet_max_capture"],
            a["outfall_cell"],
            a["outfall_node"],
            a["outfall_invert"],
            a["outfall_diameter"],
            a["outfall_coeff"],
            a["outfall_max_flow"],
            a["outfall_zero_storage"],
            a["pipe_end_cell"],
            a["pipe_end_node"],
            a["pipe_end_invert"],
            a["pipe_end_diameter"],
            a["pipe_end_area"],
            a["pipe_end_inlet_loss_k"],
            a["pipe_end_outlet_loss_k"],
            a["cell_depth"],
            a["node_depth"],
            a["link_flow"],
            1.0, 9.81, 0, 1.0e-3, 1.0,
        )

        # Verify the kernel ran without error (compute-sanitizer coverage).
        self.assertTrue(np.all(np.isfinite(nd_out)),
                        "Node depths should be finite even in dry state")
        self.assertTrue(np.all(np.isfinite(lf_out)),
                        "Link flows should be finite in dry state")

    # ── Outfall exchange ────────────────────────────────────────────────
    def test_drainage_outfall_step(self):
        """Drainage step with an outfall."""
        n_cells = 1
        cell_wse = np.array([1.5], dtype=np.float64)
        cell_area = np.array([5.0], dtype=np.float64)
        cell_bed = np.array([0.0], dtype=np.float64)
        cell_depth = np.array([1.5], dtype=np.float64)

        n_nodes = 1
        node_invert = np.array([0.0], dtype=np.float64)
        node_max_depth = np.array([3.0], dtype=np.float64)
        node_surface_area = np.array([8.0], dtype=np.float64)
        node_depth = np.array([1.2], dtype=np.float64)

        n_links = 0
        lf = np.empty(0, dtype=np.int32)
        lt = np.empty(0, dtype=np.int32)
        ll = np.empty(0, dtype=np.float64)
        lr = np.empty(0, dtype=np.float64)
        ld = np.empty(0, dtype=np.float64)
        lmf = np.empty(0, dtype=np.float64)
        lfl = np.empty(0, dtype=np.float64)

        n_inlets = 0
        ic = np.empty(0, dtype=np.int32)
        ind = np.empty(0, dtype=np.int32)
        icr = np.empty(0, dtype=np.float64)
        iw = np.empty(0, dtype=np.float64)
        ico = np.empty(0, dtype=np.float64)
        imc = np.empty(0, dtype=np.float64)

        n_outfalls = 1
        oc = np.array([0], dtype=np.int32)
        ond = np.array([0], dtype=np.int32)
        oi = np.array([0.0], dtype=np.float64)
        od = np.array([1.0], dtype=np.float64)
        oc_ = np.array([0.82], dtype=np.float64)
        omf = np.array([-1.0], dtype=np.float64)
        ozs = np.array([0], dtype=np.int32)

        n_pe = 0
        pec = np.empty(0, dtype=np.int32)
        pen = np.empty(0, dtype=np.int32)
        pei = np.empty(0, dtype=np.float64)
        ped = np.empty(0, dtype=np.float64)
        pea = np.empty(0, dtype=np.float64)
        peik = np.empty(0, dtype=np.float64)
        peok = np.empty(0, dtype=np.float64)

        nd_out, lf_out, diag = _MOD.swe2d_gpu_drainage_step(
            cell_wse, cell_area, node_invert, node_max_depth,
            node_surface_area, lf, lt, ll, lr, ld, lmf,
            ic, ind, icr, iw, ico, imc,
            oc, ond, oi, od, oc_, omf, ozs,
            pec, pen, pei, ped, pea, peik, peok,
            cell_depth, node_depth, lfl,
            1.0, 9.81, 0, 1.0e-3, 1.0,
        )

        self.assertEqual(nd_out.shape, (1,))
        self.assertEqual(lf_out.shape, (0,))


class TestGPUCulvertFaceFluxComputeSanitizer(unittest.TestCase):
    """Culvert face-flux CUDA path coverage."""

    @classmethod
    def setUpClass(cls):
        if _MOD is None:
            raise unittest.SkipTest("hydra_swe2d not available")
        if not hasattr(_MOD, "swe2d_gpu_upload_culvert_face_flux_params"):
            raise unittest.SkipTest("GPU culvert face-flux functions not compiled")
        if not bool(_MOD.swe2d_gpu_available()):
            raise unittest.SkipTest("CUDA GPU not available")

    def test_culvert_face_flux_alloc_and_apply(self):
        """Culvert face flux: allocate, upload params, apply.

        NOTE: upload_culvert_face_flux_params requires a solver with CUDA
        device state to be active.  Without one, it raises RuntimeError.
        This test verifies memory correctness when a device IS available,
        so we just check shapes/reads to exercise the kernel paths.
        """
        n_cells = 2

        # Allocate (does not require solver device)
        _MOD.swe2d_gpu_alloc_ext_struct_flux(n_cells)

        # Upload — requires solver device.  Skip gracefully if unavailable.
        struct_idx = np.array([0], dtype=np.int32)
        face_nx = np.array([1.0], dtype=np.float64)
        face_ny = np.array([0.0], dtype=np.float64)
        face_width = np.array([1.0], dtype=np.float64)
        donor_cell = np.array([0], dtype=np.int32)
        receiver_cell = np.array([1], dtype=np.int32)
        invert_elev = np.array([0.0], dtype=np.float64)
        depth_safety = np.array([0.1], dtype=np.float64)
        donor_area = np.array([5.0], dtype=np.float64)
        enq_up = np.array([0], dtype=np.int32)
        enq_dn = np.array([1], dtype=np.int32)

        try:
            _MOD.swe2d_gpu_upload_culvert_face_flux_params(
                struct_idx, face_nx, face_ny, face_width,
                donor_cell, receiver_cell, invert_elev, depth_safety,
                donor_area, True,
                enq_up, enq_dn,
            )
        except RuntimeError as ex:
            self.skipTest(f"No GPU device state: {ex}")

        # Apply with small timestep
        _MOD.swe2d_gpu_apply_culvert_face_flux(0.1, 1.0e-6)
        _MOD.swe2d_gpu_fold_culvert_mass_to_source(n_cells)

        # Readback
        h_out, hu_out, hv_out = _MOD.swe2d_gpu_readback_ext_struct_flux(n_cells)
        self.assertEqual(h_out.shape, (2,))
        self.assertEqual(hu_out.shape, (2,))
        self.assertEqual(hv_out.shape, (2,))


if __name__ == "__main__":
    unittest.main()
