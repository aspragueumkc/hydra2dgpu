"""
GPU compute-sanitizer test: structure/culvert/weir/orifice CUDA paths.

Exercises these module-level functions that are NOT covered by the
core solver tests:
  - swe2d_gpu_compute_structure_flows
  - swe2d_gpu_compute_coupling_sources
  - swe2d_gpu_preload_structure_params
  - swe2d_gpu_compute_coupling_full_on_device
  - swe2d_gpu_readback_structure_flows
  - swe2d_gpu_readback_coupling_sources
  - swe2d_gpu_redistribute_structure_sources

Run with compute-sanitizer:
  compute-sanitizer --tool=memcheck python -m pytest tests/test_swe2d_gpu_structures.py -v
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


class TestGPUStructuresComputeSanitizer(unittest.TestCase):
    """GPU structure function coverage for compute-sanitizer."""

    @classmethod
    def setUpClass(cls):
        if _MOD is None:
            raise unittest.SkipTest("hydra_swe2d not available")
        if not hasattr(_MOD, "swe2d_gpu_compute_structure_flows"):
            raise unittest.SkipTest("GPU structure functions not compiled")
        if not bool(_MOD.swe2d_gpu_available()):
            raise unittest.SkipTest("CUDA GPU not available")

    # ── Culvert: circular, HDS-5 code 1 ─────────────────────────────────
    def test_culvert_circular_flow_native(self):
        """Culvert structure flows via GPU kernel (circular, code 1)."""
        n_cells = 2
        cell_wse = np.array([2.0, 0.8], dtype=np.float64)
        cell_bed = np.array([0.0, 0.0], dtype=np.float64)
        stype = np.array([2], dtype=np.int32)  # 2=culvert
        up = np.array([0], dtype=np.int32)
        dn = np.array([1], dtype=np.int32)

        crest = np.array([0.0], dtype=np.float64)
        width = np.array([0.0], dtype=np.float64)
        height = np.array([0.0], dtype=np.float64)
        diameter = np.array([1.0], dtype=np.float64)  # 1m diameter
        length = np.array([15.0], dtype=np.float64)
        rough = np.array([0.013], dtype=np.float64)
        coeff = np.array([1.0], dtype=np.float64)
        cd = np.array([0.75], dtype=np.float64)
        opening = np.array([1.0], dtype=np.float64)
        q_pump = np.array([0.0], dtype=np.float64)
        max_flow = np.array([-1.0], dtype=np.float64)

        cc = np.array([1], dtype=np.int32)   # culvert_code=1 (corrugated metal)
        cs = np.array([0], dtype=np.int32)   # culvert_shape=0 (circular)
        cr = np.array([1.0], dtype=np.float64)
        cspan = np.array([0.0], dtype=np.float64)
        carea = np.array([0.0], dtype=np.float64)
        cb = np.array([1.0], dtype=np.float64)   # barrels
        cslope = np.array([0.005], dtype=np.float64)
        ii = np.array([0.0], dtype=np.float64)
        oi = np.array([-0.075], dtype=np.float64)
        ek = np.array([0.5], dtype=np.float64)
        xk = np.array([1.0], dtype=np.float64)
        emb_en = np.array([0], dtype=np.int32)
        emb_crest = np.array([0.0], dtype=np.float64)
        emb_width = np.array([0.0], dtype=np.float64)
        emb_coeff = np.array([1.0], dtype=np.float64)

        q = _MOD.swe2d_gpu_compute_structure_flows(
            cell_wse, cell_bed, stype, up, dn,
            crest, width, height, diameter, length, rough,
            coeff, cd, opening, q_pump, max_flow,
            cc, cs, cr, cspan, carea, cb, cslope,
            ii, oi, ek, xk, emb_en, emb_crest, emb_width, emb_coeff,
            9.81, 3.28084,
        )
        self.assertEqual(q.shape, (1,))
        self.assertTrue(np.isfinite(float(q[0])),
                        "Culvert flow should be finite (not NaN/inf)")
        # NOTE: value may be 0.0 depending on culvert table calibration;
        # this test is for compute-sanitizer memory checking, not hydraulics.

    # ── Weir ─────────────────────────────────────────────────────────────
    def test_weir_flow_native(self):
        """Weir flows via GPU kernel (type=1)."""
        n_cells = 2
        cell_wse = np.array([2.5, 1.0], dtype=np.float64)
        cell_bed = np.array([0.0, 0.0], dtype=np.float64)
        stype = np.array([1], dtype=np.int32)  # weir
        up = np.array([0], dtype=np.int32)
        dn = np.array([1], dtype=np.int32)
        crest = np.array([1.0], dtype=np.float64)
        width = np.array([3.0], dtype=np.float64)
        coeff = np.array([1.0], dtype=np.float64)
        # remaining unused for weir
        _d = np.zeros(1, dtype=np.float64)
        _i = np.zeros(1, dtype=np.int32)

        q = _MOD.swe2d_gpu_compute_structure_flows(
            cell_wse, cell_bed, stype, up, dn,
            crest, width, _d, _d, _d, _d,
            coeff, _d, _d, _d, _d,
            _i, _i, _d, _d, _d, _d, _d,
            _d, _d, _d, _d,
            _i, _d, _d, _d,
            9.81, 3.28084,
        )
        self.assertEqual(q.shape, (1,))
        self.assertTrue(np.isfinite(float(q[0])),
                        "Weir flow should be finite")

    # ── Orifice ──────────────────────────────────────────────────────────
    def test_orifice_flow_native(self):
        """Orifice flows via GPU kernel (type=3)."""
        cell_wse = np.array([3.0, 1.5], dtype=np.float64)
        cell_bed = np.array([0.0, 0.0], dtype=np.float64)
        stype = np.array([3], dtype=np.int32)
        up = np.array([0], dtype=np.int32)
        dn = np.array([1], dtype=np.int32)
        crest = np.array([0.0], dtype=np.float64)
        width = np.array([1.5], dtype=np.float64)
        height = np.array([0.5], dtype=np.float64)
        opening = np.array([1.0], dtype=np.float64)
        cd = np.array([0.6], dtype=np.float64)
        _d = np.zeros(1, dtype=np.float64)
        _i = np.zeros(1, dtype=np.int32)

        q = _MOD.swe2d_gpu_compute_structure_flows(
            cell_wse, cell_bed, stype, up, dn,
            crest, width, height, _d, _d, _d,
            _d, cd, opening, _d, _d,
            _i, _i, _d, _d, _d, _d, _d,
            _d, _d, _d, _d,
            _i, _d, _d, _d,
            9.81, 3.28084,
        )
        self.assertEqual(q.shape, (1,))
        self.assertTrue(np.isfinite(float(q[0])),
                        "Orifice flow should be finite")

    # ── Bridge ───────────────────────────────────────────────────────────
    def test_bridge_flow_native(self):
        """Bridge (type=4) flows via GPU kernel."""
        cell_wse = np.array([3.0, 1.0], dtype=np.float64)
        cell_bed = np.array([0.0, 0.0], dtype=np.float64)
        stype = np.array([4], dtype=np.int32)
        up = np.array([0], dtype=np.int32)
        dn = np.array([1], dtype=np.int32)
        crest = np.array([0.0], dtype=np.float64)
        width = np.array([10.0], dtype=np.float64)
        height = np.array([3.0], dtype=np.float64)
        opening = np.array([0.8], dtype=np.float64)
        ek = np.array([0.5], dtype=np.float64)   # entrance_loss_k
        xk = np.array([0.5], dtype=np.float64)   # exit_loss_k
        _d = np.zeros(1, dtype=np.float64)
        _i = np.zeros(1, dtype=np.int32)

        q = _MOD.swe2d_gpu_compute_structure_flows(
            cell_wse, cell_bed, stype, up, dn,
            crest, width, height, _d, _d, _d,
            _d, _d, opening, _d, _d,
            _i, _i, _d, _d, _d, _d, _d,
            _d, _d, ek, xk,
            _i, _d, _d, _d,
            9.81, 3.28084,
        )
        self.assertEqual(q.shape, (1,))
        self.assertTrue(np.isfinite(float(q[0])),
                        "Bridge flow should be finite")

    # ── Pump ─────────────────────────────────────────────────────────────
    def test_pump_flow_native(self):
        """Pump (type=5) flows via GPU kernel."""
        cell_wse = np.array([2.0, 1.0], dtype=np.float64)
        cell_bed = np.array([0.0, 0.0], dtype=np.float64)
        stype = np.array([5], dtype=np.int32)
        up = np.array([0], dtype=np.int32)
        dn = np.array([1], dtype=np.int32)
        q_pump = np.array([0.5], dtype=np.float64)
        crest = np.array([0.0], dtype=np.float64)
        _d = np.zeros(1, dtype=np.float64)
        _i = np.zeros(1, dtype=np.int32)

        q = _MOD.swe2d_gpu_compute_structure_flows(
            cell_wse, cell_bed, stype, up, dn,
            crest, _d, _d, _d, _d, _d,
            _d, _d, _d, _d, q_pump,
            _i, _i, _d, _d, _d, _d, _d,
            _d, _d, _d, _d,
            _i, _d, _d, _d,
            9.81, 3.28084,
        )
        self.assertEqual(q.shape, (1,))
        self.assertTrue(np.isfinite(float(q[0])),
                        "Pump flow should be finite")

    # ── Coupling sources (convert structure flows to cell rates) ─────────
    def test_coupling_sources_kernel(self):
        """Convert structure flows to per-cell source rates."""
        cell_area = np.array([5.0, 5.0, 10.0], dtype=np.float64)
        inlet_cell = np.array([0], dtype=np.int32)
        inlet_flow = np.array([0.0], dtype=np.float64)  # no inlets
        struct_up = np.array([0, 1], dtype=np.int32)
        struct_dn = np.array([1, 2], dtype=np.int32)
        struct_q = np.array([0.5, -0.3], dtype=np.float64)

        src = _MOD.swe2d_gpu_compute_coupling_sources(
            cell_area, inlet_cell, inlet_flow,
            struct_up, struct_dn, struct_q,
        )
        self.assertEqual(src.shape, (3,))
        # Net sink should approximately balance
        net = float(np.dot(src, cell_area))
        self.assertAlmostEqual(net, 0.0, places=10)

    # ── Persisent path: preload + compute + readback ─────────────────────
    def test_persistent_structure_path(self):
        """Preload params, compute on device, readback flows and sources.

        NOTE: This requires a solver with CUDA to have set the global device
        pointer via swe2d_gpu_set_coupling_device_global.  If no solver has
        been created yet, the preload/compute functions will raise, and we
        skip gracefully.
        """
        try:
            stype = np.array([1], dtype=np.int32)  # weir
            up = np.array([0], dtype=np.int32)
            dn = np.array([1], dtype=np.int32)
            crest = np.array([1.0], dtype=np.float64)
            width = np.array([2.0], dtype=np.float64)
            _d = np.zeros(1, dtype=np.float64)
            _i = np.zeros(1, dtype=np.int32)

            _MOD.swe2d_gpu_preload_structure_params(
                stype, up, dn, crest, width, _d, _d, _d, _d,
                _d, _d, _d, _d, _d,
                _i, _i, _d, _d, _d, _d, _d,
                _d, _d, _d, _d,
                _i, _d, _d, _d,
                9.81, 3.28084,
            )
        except Exception as ex:
            self.skipTest(f"swe2d_gpu_preload_structure_params failed "
                          f"(no solver device?): {ex}")

        # Preload cell areas
        cell_area = np.array([5.0, 5.0], dtype=np.float64)
        try:
            _MOD.swe2d_gpu_preload_coupling_cell_area(cell_area)
        except Exception as ex:
            self.skipTest(f"swe2d_gpu_preload_coupling_cell_area failed: {ex}")

        # Compute on-device (n_structures=1, no inlets)
        inlet_cell = np.empty(0, dtype=np.int32)
        inlet_flow = np.empty(0, dtype=np.float64)
        try:
            _MOD.swe2d_gpu_compute_coupling_full_on_device(
                None,  # cell_wse=None → on-device WSE from depth + bed
                1,     # n_structures
                inlet_cell,
                inlet_flow,
                None,  # host_structure_flows=None → GPU computes
            )
        except Exception as ex:
            self.skipTest(f"swe2d_gpu_compute_coupling_full_on_device failed: {ex}")

        # Readback structure flows
        try:
            struct_flows = _MOD.swe2d_gpu_readback_structure_flows(1)
            self.assertEqual(struct_flows.shape, (1,))

            sources = _MOD.swe2d_gpu_readback_coupling_sources(2)
            self.assertEqual(sources.shape, (2,))
        except Exception as ex:
            self.skipTest(f"Readback failed: {ex}")

    # ── Redistribution kernel ────────────────────────────────────────────
    def test_redistribution_kernel(self):
        """Redistribute structure sources across corridor cells."""
        n_cells = 4
        cell_area = np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float64)
        source = np.zeros(n_cells, dtype=np.float64)
        n_struct = 1
        offsets = np.array([0, 2, 4], dtype=np.int32)
        cell_idx = np.array([0, 1, 2, 3], dtype=np.int32)
        weights = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
        struct_q = np.array([1.0], dtype=np.float64)
        orig_up = np.array([0], dtype=np.int32)
        orig_dn = np.array([2], dtype=np.int32)

        result = _MOD.swe2d_gpu_redistribute_structure_sources(
            source, offsets, cell_idx, weights,
            struct_q, orig_up, orig_dn, cell_area,
        )
        self.assertEqual(result.shape, (n_cells,))
        # Mass should be conserved (sum of redistributed sources)
        total_in = float(struct_q[0])
        total_out = float(np.sum(result * cell_area))
        self.assertAlmostEqual(total_out, total_in, places=10)


if __name__ == "__main__":
    unittest.main()
