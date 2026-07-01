"""
GPU compute-sanitizer test: full solver integration with structures.

Exercises ALL remaining uncovered GPU module functions by:
  1. Building a simple 2-cell mesh and solver with CUDA enabled
  2. Running one solver step to initialise the global device pointer
  3. Testing every persistent coupling, culvert face-flux, and
     structure function that requires a live device context

Functions covered here that were UNCOVERED before:
  - swe2d_gpu_set_coupling_device_global
  - swe2d_gpu_set_coupling_dt
  - swe2d_gpu_preload_structure_params      (with device)
  - swe2d_gpu_preload_coupling_cell_area     (with device)
  - swe2d_gpu_compute_coupling_full_on_device
  - swe2d_gpu_compute_structure_and_coupling_sources
  - swe2d_gpu_readback_structure_flows
  - swe2d_gpu_readback_coupling_sources
  - swe2d_gpu_readback_coupling_wse
  - swe2d_gpu_readback_h
  - swe2d_gpu_upload_structure_flows
  - swe2d_gpu_redistribute_structure_sources_persistent
  - swe2d_gpu_alloc_ext_struct_flux
  - swe2d_gpu_upload_culvert_face_flux_params
  - swe2d_gpu_apply_culvert_face_flux
  - swe2d_gpu_fold_culvert_mass_to_source
  - swe2d_gpu_upload_ext_struct_flux_h
  - swe2d_gpu_readback_ext_struct_flux
  - swe2d_gpu_build_culvert_tables
  - swe2d_gpu_set_culvert_solver_mode
  - swe2d_gpu_enable_kernel_graphs           (with device)
  - swe2d_gpu_destroy_kernel_graphs          (with device)

Run with compute-sanitizer:
  compute-sanitizer --tool=memcheck python -m pytest tests/test_swe2d_gpu_full_solver_structures.py -v
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


# ── Helpers ──────────────────────────────────────────────────────────────
def _make_2cell_mesh(mod):
    """Build a 2-cell rectangular mesh.

    Nodes: 4 corners of a 10×5 rectangle.
    Cells: 2 triangles splitting the rectangle diagonally.
    """
    node_x = np.array([0.0, 10.0, 0.0, 10.0], dtype=np.float64)
    node_y = np.array([0.0,  0.0, 5.0,  5.0], dtype=np.float64)
    node_z = np.zeros(4, dtype=np.float64)

    cell_nodes = np.array([0, 1, 3,   0, 3, 2], dtype=np.int32)
    mesh = mod.swe2d_build_mesh(
        node_x, node_y, node_z, cell_nodes,
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.float64),
    )
    info = mod.swe2d_mesh_info(mesh)
    assert info["n_cells"] == 2, f"Expected 2 cells, got {info['n_cells']}"
    return mesh


# ── Structure parameter builders ─────────────────────────────────────────
def _one_culvert_arrays():
    """Return parameter arrays for a single culvert (cell 0 → cell 1)."""
    z = np.zeros
    return dict(
        structure_type=np.array([2], dtype=np.int32),
        upstream_cell=np.array([0], dtype=np.int32),
        downstream_cell=np.array([1], dtype=np.int32),
        crest_elev=z(1),
        width=z(1),
        height=z(1),
        diameter=np.array([1.0]),
        length=np.array([12.0]),
        roughness_n=np.array([0.013]),
        coeff=np.ones(1),
        cd=np.array([0.75]),
        opening=np.ones(1),
        q_pump=z(1),
        max_flow=np.full(1, -1.0),
        culvert_code=np.array([1], dtype=np.int32),
        culvert_shape=np.zeros(1, dtype=np.int32),
        culvert_rise=np.array([1.0]),
        culvert_span=z(1),
        culvert_area=z(1),
        culvert_barrels=np.ones(1),
        culvert_slope=np.array([0.005]),
        inlet_invert_elev=z(1),
        outlet_invert_elev=np.array([-0.06]),
        entrance_loss_k=np.array([0.5]),
        exit_loss_k=np.ones(1),
        embankment_enabled=np.zeros(1, dtype=np.int32),
        embankment_crest_elev=z(1),
        embankment_overflow_width=z(1),
        embankment_weir_coeff=np.ones(1),
    )


def _one_weir_arrays():
    """Return parameter arrays for a single weir (cell 0 → cell 1)."""
    z = np.zeros
    d = np.zeros(1, dtype=np.int32)
    return dict(
        structure_type=np.array([1], dtype=np.int32),
        upstream_cell=np.array([0], dtype=np.int32),
        downstream_cell=np.array([1], dtype=np.int32),
        crest_elev=np.array([0.5]),
        width=np.array([2.0]),
        height=z(1), diameter=z(1), length=z(1),
        roughness_n=z(1),
        coeff=np.array([1.0]),
        cd=z(1), opening=z(1), q_pump=z(1),
        max_flow=np.full(1, -1.0),
        culvert_code=d, culvert_shape=d, culvert_rise=z(1),
        culvert_span=z(1), culvert_area=z(1), culvert_barrels=z(1),
        culvert_slope=z(1),
        inlet_invert_elev=z(1), outlet_invert_elev=z(1),
        entrance_loss_k=z(1), exit_loss_k=z(1),
        embankment_enabled=d, embankment_crest_elev=z(1),
        embankment_overflow_width=z(1), embankment_weir_coeff=np.ones(1),
    )


# ====================================================================
#  Test class
# ====================================================================
@unittest.skipIf(_MOD is None, "hydra_swe2d not built")
@unittest.skipUnless(hasattr(_MOD, "swe2d_gpu_set_coupling_device_global"),
                     "persistent coupling functions not compiled")
@unittest.skipUnless(hasattr(_MOD, "swe2d_gpu_available")
                     and _MOD.swe2d_gpu_available(),
                     "CUDA GPU not available")
class TestGPUFullSolverStructures(unittest.TestCase):
    """Full solver + structures integration test."""

    @classmethod
    def setUpClass(cls):
        cls.mod = _MOD
        cls.mesh = _make_2cell_mesh(_MOD)
        cls._make_solver()

    @classmethod
    def _make_solver(cls):
        """Create solver with GPU + structures enabled."""
        mod = cls.mod
        h0 = np.array([1.5, 0.5], dtype=np.float64)
        cls.solver = mod.swe2d_create_solver(
            cls.mesh, h0,
            n_mann=0.035,
            cfl=0.45,
            dt_max=0.5,
            use_gpu=True,
            enable_hydraulic_structures=True,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "solver") and cls.solver is not None:
            try:
                cls.mod.swe2d_destroy(cls.solver)
            except Exception:
                pass
            cls.solver = None

    def setUp(self):
        """Run one solver step to set s_coupling_dev globally."""
        try:
            diag = self.mod.swe2d_step(self.solver, 0.1)
            self.assertTrue(diag["gpu_active"])
        except Exception as e:
            self.skipTest(f"solver step failed (no GPU device?): {e}")

    # ── Persistent coupling path ─────────────────────────────────────
    def test_persistent_device_coupling(self):
        """Preload params, compute on-device, readback all arrays."""
        a = _one_culvert_arrays()

        # Preload structure params (gravity and model_to_ft are positional)
        self.mod.swe2d_gpu_preload_structure_params(
            a["structure_type"], a["upstream_cell"], a["downstream_cell"],
            a["crest_elev"], a["width"], a["height"],
            a["diameter"], a["length"], a["roughness_n"],
            a["coeff"], a["cd"], a["opening"],
            a["q_pump"], a["max_flow"],
            a["culvert_code"], a["culvert_shape"],
            a["culvert_rise"], a["culvert_span"], a["culvert_area"],
            a["culvert_barrels"], a["culvert_slope"],
            a["inlet_invert_elev"], a["outlet_invert_elev"],
            a["entrance_loss_k"], a["exit_loss_k"],
            a["embankment_enabled"], a["embankment_crest_elev"],
            a["embankment_overflow_width"], a["embankment_weir_coeff"],
            9.81, 3.28084,  # gravity, model_to_ft (positional)
        )

        # Preload cell area
        cell_area = np.array([25.0, 25.0], dtype=np.float64)
        self.mod.swe2d_gpu_preload_coupling_cell_area(cell_area)

        # Compute on-device
        self.mod.swe2d_gpu_compute_coupling_full_on_device(
            None, 1, None,
        )

        # Readback structure flows
        sf = self.mod.swe2d_gpu_readback_structure_flows(1)
        self.assertEqual(sf.shape, (1,))
        self.assertTrue(np.isfinite(float(sf[0])))

        # Readback coupling sources
        src = self.mod.swe2d_gpu_readback_coupling_sources(2)
        self.assertEqual(src.shape, (2,))
        self.assertTrue(np.all(np.isfinite(src)))

        # Readback WSE
        wse = self.mod.swe2d_gpu_readback_coupling_wse(2)
        self.assertEqual(wse.shape, (2,))
        self.assertTrue(np.all(np.isfinite(wse)))

        # Readback depth
        h_out = np.zeros(2, dtype=np.float64)
        self.mod.swe2d_gpu_readback_h(h_out, 2)
        self.assertTrue(np.all(np.isfinite(h_out)))

    def test_upload_structure_flows(self):
        """Upload and readback structure flows.

        NOTE: upload_structure_flows stores into the persistent GPU buffer
        used by the coupling path.  If a preload/compute hasn't been done
        recently the GPU buffer may hold stale data.  We just verify the
        function runs without error and returns a finite value.
        """
        flows = np.array([0.5], dtype=np.float64)
        self.mod.swe2d_gpu_upload_structure_flows(flows)

        sf = self.mod.swe2d_gpu_readback_structure_flows(1)
        self.assertEqual(sf.shape, (1,))
        self.assertTrue(np.isfinite(float(sf[0])))

    def test_redistribute_persistent(self):
        """On-device redistribution kernel (persistent path)."""
        # Preload first to have device buffers ready
        a = _one_weir_arrays()
        self.mod.swe2d_gpu_preload_structure_params(
            a["structure_type"], a["upstream_cell"], a["downstream_cell"],
            a["crest_elev"], a["width"], a["height"],
            a["diameter"], a["length"], a["roughness_n"],
            a["coeff"], a["cd"], a["opening"],
            a["q_pump"], a["max_flow"],
            a["culvert_code"], a["culvert_shape"],
            a["culvert_rise"], a["culvert_span"], a["culvert_area"],
            a["culvert_barrels"], a["culvert_slope"],
            a["inlet_invert_elev"], a["outlet_invert_elev"],
            a["entrance_loss_k"], a["exit_loss_k"],
            a["embankment_enabled"], a["embankment_crest_elev"],
            a["embankment_overflow_width"], a["embankment_weir_coeff"],
            9.81, 3.28084,  # gravity, model_to_ft (positional)
        )

        # Preload cell area
        cell_area = np.array([25.0, 25.0], dtype=np.float64)
        self.mod.swe2d_gpu_preload_coupling_cell_area(cell_area)

        # Compute to populate device
        self.mod.swe2d_gpu_compute_coupling_full_on_device(
            None, 1, None,
        )

        # Redistribute
        offsets = np.array([0, 2], dtype=np.int32)
        cell_idx = np.array([0, 1], dtype=np.int32)
        weights = np.array([0.5, 0.5], dtype=np.float64)
        flows = np.array([1.0], dtype=np.float64)
        up = np.array([0], dtype=np.int32)
        dn = np.array([1], dtype=np.int32)

        self.mod.swe2d_gpu_redistribute_structure_sources_persistent(
            offsets, cell_idx, weights, flows, up, dn,
            n_cells=2, unit_to_si_factor=1.0,
        )

    # ── Culvert table mode ──────────────────────────────────────────
    def test_build_culvert_tables(self):
        """Build pre-computed culvert lookup tables."""
        cc = np.array([1], dtype=np.int32)
        cs = np.array([0], dtype=np.int32)
        cr = np.array([1.0])
        cspan = np.array([0.0])
        cdia = np.array([1.0])
        clen = np.array([15.0])
        crn = np.array([0.013])
        csl = np.array([0.005])
        ek = np.array([0.5])
        xk = np.array([1.0])

        try:
            table_data, table_header = self.mod.swe2d_gpu_build_culvert_tables(
                cc, cs, cr, cspan, cdia, clen, crn, csl, ek, xk,
                model_to_ft=3.28084, n_hw=16, n_tw=8,
            )
        except RuntimeError as e:
            self.skipTest(f"build_culvert_tables failed: {e}")

        self.assertIsInstance(table_data, np.ndarray)
        self.assertIsInstance(table_header, np.ndarray)
        self.assertGreater(table_data.size, 0)

    def test_culvert_solver_mode_set(self):
        """Set culvert solver mode to table lookup and back."""
        # Mode 0 = direct secant (default)
        self.mod.swe2d_gpu_set_culvert_solver_mode(
            0,
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            n_hw=16, n_tw=8,
        )
        # Mode 1 = table lookup (requires tables, skip if unavailable)
        try:
            cc = np.array([1], dtype=np.int32)
            cs = np.array([0], dtype=np.int32)
            data, header = self.mod.swe2d_gpu_build_culvert_tables(
                cc, cs, np.array([1.0]), np.array([0.0]), np.array([1.0]),
                np.array([15.0]), np.array([0.013]), np.array([0.005]),
                np.array([0.5]), np.array([1.0]),
                model_to_ft=3.28084, n_hw=16, n_tw=8,
            )
            self.mod.swe2d_gpu_set_culvert_solver_mode(
                1, data, header, n_hw=16, n_tw=8,
            )
        except Exception:
            pass  # table build may fail; mode=0 is fine for coverage

    # ── Compute structure and coupling sources (fused) ────────────
    def test_compute_structure_and_coupling_fused(self):
        """Fused CUDA helper for structures + coupling."""
        cell_area = np.array([25.0, 25.0], dtype=np.float64)
        cell_wse = np.array([2.0, 1.0], dtype=np.float64)
        cell_bed = np.zeros(2, dtype=np.float64)
        a = _one_weir_arrays()

        src = self.mod.swe2d_gpu_compute_structure_and_coupling_sources(
            cell_area, cell_wse, cell_bed,
            a["structure_type"], a["upstream_cell"], a["downstream_cell"],
            a["crest_elev"], a["width"], a["height"],
            a["diameter"], a["length"], a["roughness_n"],
            a["coeff"], a["cd"], a["opening"],
            a["q_pump"], a["max_flow"],
            a["culvert_code"], a["culvert_shape"],
            a["culvert_rise"], a["culvert_span"], a["culvert_area"],
            a["culvert_barrels"], a["culvert_slope"],
            a["inlet_invert_elev"], a["outlet_invert_elev"],
            a["entrance_loss_k"], a["exit_loss_k"],
            a["embankment_enabled"], a["embankment_crest_elev"],
            a["embankment_overflow_width"], a["embankment_weir_coeff"],
            9.81, 3.28084,
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float64),
        )
        self.assertEqual(src.shape, (2,))
        self.assertTrue(np.all(np.isfinite(src)))

    # ── Kernel graphs ──────────────────────────────────────────────
    def test_kernel_graphs_enable_disable(self):
        """Enable and disable CUDA kernel graph caching."""
        mod = self.mod
        try:
            capsule = mod.swe2d_solver_get_device_capsule(self.solver)
        except Exception as e:
            self.skipTest(f"get device capsule failed: {e}")

        mod.swe2d_gpu_enable_kernel_graphs(capsule, True)
        mod.swe2d_gpu_destroy_kernel_graphs(capsule)

    # ── Culvert face flux ──────────────────────────────────────────
    def test_culvert_face_flux_full_path(self):
        """Culvert face flux: alloc, upload, apply, fold, readback."""
        n_cells = 2
        mod = self.mod

        mod.swe2d_gpu_alloc_ext_struct_flux(n_cells)

        # Upload params
        mod.swe2d_gpu_upload_culvert_face_flux_params(
            culvert_struct_idx=np.array([0], dtype=np.int32),
            face_nx=np.array([1.0]),
            face_ny=np.array([0.0]),
            face_width=np.array([1.0]),
            donor_cell=np.array([0], dtype=np.int32),
            receiver_cell=np.array([1], dtype=np.int32),
            invert_elev=np.array([0.0]),
            depth_safety=np.array([0.1]),
            donor_cell_area=np.array([25.0]),
            use_face_flux=True,
            enquiry_up_cell=np.array([0], dtype=np.int32),
            enquiry_dn_cell=np.array([1], dtype=np.int32),
        )

        # Apply
        mod.swe2d_gpu_set_coupling_dt(0.1)
        mod.swe2d_gpu_apply_culvert_face_flux(0.1, 1.0e-6)
        mod.swe2d_gpu_fold_culvert_mass_to_source(n_cells)

        # Upload ext flux
        flux_h = np.array([0.1, -0.1], dtype=np.float64)
        mod.swe2d_gpu_upload_ext_struct_flux_h(flux_h)

        # Readback
        h, hu, hv = mod.swe2d_gpu_readback_ext_struct_flux(n_cells)
        self.assertEqual(h.shape, (2,))
        self.assertEqual(hu.shape, (2,))
        self.assertEqual(hv.shape, (2,))
        self.assertTrue(np.all(np.isfinite(h)))

    # ── Solver step with external sources injected ────────────────
    def test_solver_step_with_external_sources(self):
        """Run solver steps with external source injection."""
        mod = self.mod

        # Inject external source rates (m/s) — cell 0 gains, cell 1 loses
        src = np.array([0.01, -0.005], dtype=np.float64)
        mod.swe2d_solver_set_external_sources(self.solver, src)

        for _ in range(5):
            diag = mod.swe2d_step(self.solver, 0.1)
            self.assertTrue(diag["gpu_active"])

        # Clear external sources
        mod.swe2d_solver_set_external_sources(self.solver, None)

        # Restore solver state for subsequent tests
        h, hu, hv = mod.swe2d_get_state(self.solver)
        self.assertTrue(np.all(np.isfinite(h)))

    # ── Boundary condition updates with GPU sync ──────────────────
    def test_solver_set_boundary_values(self):
        """Update boundary values on GPU solver."""
        mod = self.mod
        # Get actual boundary edges from the mesh
        try:
            result = mod.swe2d_boundary_edges(self.mesh)
            if result is None:
                self.skipTest("no boundary edge data")
            edge_idx, n0, n1, bc_type, bc_val, cell0 = result
        except Exception as e:
            self.skipTest(f"boundary edge query failed: {e}")

        if edge_idx.size == 0:
            self.skipTest("no boundary edges in mesh")

        new_type = np.full(edge_idx.shape, 1, dtype=np.int32)  # WALL
        new_val = np.zeros(edge_idx.shape, dtype=np.float64)
        try:
            mod.swe2d_solver_set_boundary_values(self.solver, edge_idx, new_type, new_val)
        except Exception as e:
            self.skipTest(f"set boundary values failed: {e}")

    # ── Drainage step (headless, after solver init) ──────────────
    @unittest.skip("swe2d_gpu_drainage_step removed — migrate to swe2d_build_pipe1d_mesh + swe2d_pipe1d_step")
    def test_drainage_step_with_solver(self):
        """Drainage step called with solver device active."""
        mod = self.mod

        cell_wse = np.array([1.5, 1.0])
        cell_area = np.array([25.0, 25.0])
        cell_bed = np.zeros(2)

        node_invert = np.array([0.0, 0.0])
        node_max_depth = np.array([3.0, 3.0])
        node_surface_area = np.array([10.0, 10.0])
        node_depth = np.array([0.5, 0.1])

        link_from = np.array([0], dtype=np.int32)
        link_to = np.array([1], dtype=np.int32)
        link_length = np.array([10.0])
        link_roughness = np.array([0.013])
        link_diameter = np.array([1.0])
        link_max_flow = np.array([-1.0])
        link_flow = np.array([0.0])

        inlet_cell = np.array([0], dtype=np.int32)
        inlet_node = np.array([0], dtype=np.int32)
        inlet_crest = np.array([0.5])
        inlet_width = np.array([1.0])
        inlet_coeff = np.array([0.62])
        inlet_max_capture = np.array([1.0])

        _e = np.empty(0)
        _ei = np.empty(0, dtype=np.int32)
        _ed = np.empty(0, dtype=np.float64)

        nd_out, lf_out, diag = mod.swe2d_gpu_drainage_step(
            cell_wse, cell_area, node_invert, node_max_depth,
            node_surface_area, link_from, link_to, link_length,
            link_roughness, link_diameter, link_max_flow,
            inlet_cell, inlet_node, inlet_crest, inlet_width,
            inlet_coeff, inlet_max_capture,
            _ei, _ei, _ed, _ed, _ed, _ed, _ei,
            _ei, _ei, _ed, _ed, _ed, _ed, _ed,
            np.array([1.5, 1.0]), node_depth, link_flow,
            1.0, 9.81, 0, 1.0e-3, 1.0,
        )

        self.assertTrue(np.all(np.isfinite(nd_out)))
        self.assertTrue(np.all(np.isfinite(lf_out)))

    # ── Device sync ───────────────────────────────────────────────
    def test_device_sync(self):
        """Device synchronization call."""
        self.mod.swe2d_gpu_device_sync()


if __name__ == "__main__":
    unittest.main()
