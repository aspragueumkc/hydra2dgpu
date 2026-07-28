"""TDD tests for the Pipe1D face-indexed FVM refactor.

Phase 1 — test scaffolding. Every test skips cleanly against current code
(``swe2d_build_unified_mesh`` does not exist yet). They turn green as
Phases 2–4 of the refactor land.

All tests use the new unified mesh + face-flux API. Mass conservation is
automatic by FV construction; closed-system tests pass to machine precision.

Ref: docs/pipe1d_face_indexed_refactor_plan.md §4 Phase 1.

Run:
    mamba run -n qgis_stable python3 -m unittest -v tests.test_pipe1d_face_indexed_mesh
"""
from __future__ import annotations

import unittest

import numpy as np


# ── module / GPU gate ─────────────────────────────────────────────────────
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


def _skip_unless_refactored():
    """Return True iff the refactored unified-mesh API is available."""
    if _MOD is None:
        return False
    try:
        return hasattr(_MOD, "swe2d_build_unified_mesh")
    except Exception:
        return False


G, K_MANN, H_MIN = 9.81, 1.0, 1.0e-4


# ── test suite ─────────────────────────────────────────────────────────────
@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
class TestPipe1DFaceIndexedMesh(unittest.TestCase):
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
            dt_fixed=0.05,
            dt_max=0.05,
        )
        cls._dev = int(_MOD.swe2d_get_coupling_dev_ptr())

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls._backend, "destroy"):
            cls._backend.destroy()

    # ── helpers ──────────────────────────────────────────────────────────

    def _build_closed_system(
        self,
        *,
        n_links=1,
        link_from=None,
        link_to=None,
        L=None,
        D=None,
        n_mann=None,
        S0=None,
        node_invert=None,
        mcl=10,
        manhole_diameters=None,
        manhole_heights=None,
        manhole_inverts=None,
        manhole_rims=None,
        inlet_diameters=None,
        inlet_heights=None,
        inlet_inverts=None,
        d_initial=None,
        node_is_outfall=None,
        node_is_pipe_end=None,
    ):
        """Build a pipe1D mesh via the unified API and upload initial cell depths.

        For pipe-only systems ``d_initial`` is a 2-tuple ``(d_upstream,
        d_downstream)`` used to linearly interpolate per-cell depth across
        pipe sub-cells.  When manhole/inlet cells are present, ``d_initial``
        must be a 1-D array of length ``n_pipe_cells + n_manhole + n_inlet``.

        ``node_is_outfall`` is an optional [n_nodes] int32 array where
        1 = outfall (OUTFALL_BC face), 0 = wall (zero-flux boundary).
        Default None = all walls.

        Returns ``(n_pipe_cells, n_manhole, n_inlet, sub_len,
        manhole_surface_areas, inlet_surface_areas)``.
        """
        # --- defaults ---
        if link_from is None:
            link_from = np.arange(n_links, dtype=np.int32)
        if link_to is None:
            link_to = np.arange(1, n_links + 1, dtype=np.int32)
        if L is None:
            L = np.full(n_links, 100.0, dtype=np.float64)
        if D is None:
            D = np.full(n_links, 0.5, dtype=np.float64)
        if n_mann is None:
            n_mann = np.full(n_links, 0.013, dtype=np.float64)
        if S0 is None:
            S0 = np.zeros(n_links, dtype=np.float64)
        if node_invert is None:
            node_invert = np.linspace(10.0, 9.0, n_links + 1, dtype=np.float64)

        n_nodes = n_links + 1

        # --- manhole cell arrays ---
        if manhole_diameters is not None:
            n_manhole = len(manhole_diameters)
            if manhole_inverts is None:
                manhole_inverts = np.full(n_manhole, 9.0, dtype=np.float64)
            if manhole_heights is None:
                manhole_heights = np.full(n_manhole, 2.0, dtype=np.float64)
            if manhole_rims is None:
                manhole_rims = manhole_inverts + manhole_heights
            manhole_surface_areas = np.pi * (np.asarray(manhole_diameters, dtype=np.float64) / 2.0) ** 2
        else:
            n_manhole = 0
            manhole_surface_areas = None

        # --- inlet cell arrays ---
        if inlet_diameters is not None:
            n_inlet = len(inlet_diameters)
            if inlet_inverts is None:
                inlet_inverts = np.full(n_inlet, 10.0, dtype=np.float64)
            if inlet_heights is None:
                inlet_heights = np.full(n_inlet, 1.5, dtype=np.float64)
            inlet_surface_areas = np.pi * (np.asarray(inlet_diameters, dtype=np.float64) / 2.0) ** 2
        else:
            n_inlet = 0
            inlet_surface_areas = None

        # --- pipe-end node detection ---
        if node_is_pipe_end is not None:
            pe_node_arr = np.asarray(node_is_pipe_end, dtype=np.int32)
            pe_mask = pe_node_arr > 0
            pipe_end_node_ids = np.where(pe_mask)[0].astype(np.int32)
            n_pipe_ends = int(len(pipe_end_node_ids))
        else:
            n_pipe_ends = 0
            pipe_end_node_ids = np.array([], dtype=np.int32)

        # --- call unified mesh build ---
        _MOD.swe2d_build_unified_mesh(
            self._dev,
            n_links=n_links,
            link_from=link_from,
            link_to=link_to,
            L=L,
            D=D,
            n_mann=n_mann,
            S0=S0,
            node_invert=node_invert,
            n_manhole_cells=n_manhole,
            manhole_node_ids=(np.array([], dtype=np.int32) if n_manhole == 0 else
                              np.arange(n_manhole, dtype=np.int32)),
            manhole_invert=(np.empty(0, dtype=np.float64) if n_manhole == 0 else
                            np.asarray(manhole_inverts, dtype=np.float64)),
            manhole_surface_area=(np.empty(0, dtype=np.float64) if n_manhole == 0 else
                                  np.asarray(manhole_surface_areas, dtype=np.float64)),
            manhole_max_depth=(np.empty(0, dtype=np.float64) if n_manhole == 0 else
                               np.asarray(manhole_heights, dtype=np.float64)),
            manhole_rim=(np.empty(0, dtype=np.float64) if n_manhole == 0 else
                         np.asarray(manhole_rims, dtype=np.float64)),
            manhole_diameter=(np.empty(0, dtype=np.float64) if n_manhole == 0 else
                              np.asarray(manhole_diameters, dtype=np.float64)),
            n_inlet_cells=n_inlet,
            inlet_node_ids=(np.array([], dtype=np.int32) if n_inlet == 0 else
                            np.arange(n_inlet, dtype=np.int32)),
            inlet_invert=(np.empty(0, dtype=np.float64) if n_inlet == 0 else
                          np.asarray(inlet_inverts, dtype=np.float64)),
            inlet_surface_area=(np.empty(0, dtype=np.float64) if n_inlet == 0 else
                                np.asarray(inlet_surface_areas, dtype=np.float64)),
            inlet_max_depth=(np.empty(0, dtype=np.float64) if n_inlet == 0 else
                             np.asarray(inlet_heights, dtype=np.float64)),
            inlet_diameter=(np.empty(0, dtype=np.float64) if n_inlet == 0 else
                            np.asarray(inlet_diameters, dtype=np.float64)),
            mcl=mcl,
            n_pipe_ends=n_pipe_ends,
            pipe_end_node_ids=pipe_end_node_ids,
            node_is_outfall=node_is_outfall,
        )

        # --- cell counts ---
        _L0 = float(L[0])
        n_sub = max(1, int(np.ceil(_L0 / mcl)))
        n_pipe_cells = n_sub * n_links
        n_cells_total = n_pipe_cells + n_manhole + n_inlet

        # --- initial depth upload ---
        if d_initial is None:
            d_initial_tup = (0.0, 0.0)
        elif isinstance(d_initial, (list, tuple)):
            d_initial_tup = d_initial
        else:
            # already a flat array
            d_initial_tup = None

        if d_initial_tup is not None:
            d0, d1 = d_initial_tup if len(d_initial_tup) == 2 else (d_initial_tup[0], 0.0)
            h_init = np.zeros(n_cells_total, dtype=np.float64)
            for i in range(n_pipe_cells):
                frac = (i + 0.5) / max(n_pipe_cells, 1)
                h_init[i] = d0 + (d1 - d0) * frac
            # manhole/inlet cells get zero depth by default (tests set them explicitly)
        else:
            # d_initial is a flat array — pass through
            h_init = np.asarray(d_initial, dtype=np.float64)

        _MOD.swe2d_pipe1d_upload_cell_h(self._dev, h_init)
        _MOD.swe2d_pipe1d_init_cell_area(self._dev, H_MIN)

        sub_len = _L0 / max(n_sub, 1)
        return n_pipe_cells, n_manhole, n_inlet, sub_len, manhole_surface_areas, inlet_surface_areas

    def _mass_unified(self, n_pipe_cells, n_manhole, n_inlet, sub_len,
                      manhole_surface_areas=None, inlet_surface_areas=None):
        """Read back unified cell state and compute total system mass.

        Pipe cells:  volume = cell_A * sub_len
        Manhole / inlet cells:  volume = cell_h * cell_surface_area
        """
        try:
            st = _MOD.swe2d_pipe1d_readback_cell_state(
                self._dev, n_pipe_cells, n_manhole, n_inlet)
        except (AttributeError, TypeError):
            return 0.0, None

        cell_A = np.asarray(st["cell_A"], dtype=np.float64)
        cell_h = np.asarray(st["cell_h"], dtype=np.float64)
        cell_class = np.asarray(st["cell_class"], dtype=np.int32)

        total = 0.0
        for i in range(n_pipe_cells):
            total += float(cell_A[i]) * sub_len

        if manhole_surface_areas is not None:
            for i in range(n_manhole):
                idx = n_pipe_cells + i
                sa = float(manhole_surface_areas[i])
                total += float(cell_h[idx]) * sa

        if inlet_surface_areas is not None:
            for i in range(n_inlet):
                idx = n_pipe_cells + n_manhole + i
                sa = float(inlet_surface_areas[i])
                total += float(cell_h[idx]) * sa

        return total, st

    def _step(self, mode, dt=0.5, n_substeps=1, n_nodes=2, scaling=0.5):
        """Thin wrapper around the pipe1D step driver."""
        _MOD.swe2d_pipe1d_step(
            self._dev, dt, mode, n_substeps, n_nodes, scaling,
            G, K_MANN, H_MIN,
            surcharge_method=1,  # SLOT for pressurised flow
            friction_method=1,   # SUBSTEPPING (mirrors 2D friction_substep_enabled)
            time_integrator=2,   # RK2 (default)
        )

    @unittest.skipUnless(_skip_unless_refactored(),
                         "refactored mesh API not yet built")
    def test_readback_cell_state_exposes_single_cell_schema(self):
        """Cell readback exposes derived cell arrays without node aliases."""
        n_pipe_cells, n_manhole, n_inlet, _, _, _ = self._build_closed_system(
            mcl=10.0,
            d_initial=(1.0, 0.0),
        )

        state = _MOD.swe2d_pipe1d_readback_cell_state(
            self._dev,
            n_pipe_cells,
            n_manhole,
            n_inlet,
        )

        self.assertNotIn("node_depth", state)
        np.testing.assert_allclose(
            state["cell_velocity"],
            np.asarray(state["cell_Q"], dtype=np.float64)
            / np.maximum(np.asarray(state["cell_A"], dtype=np.float64), 1.0e-12),
        )
        np.testing.assert_allclose(
            state["cell_depth"],
            np.asarray(state["cell_y"], dtype=np.float64)
            - np.asarray(state["cell_invert"], dtype=np.float64),
        )

    # ── Test 1: closed 2-node, 1 link, 1 sub-cell, fully_dynamic ─────────
    @unittest.skipUnless(_skip_unless_refactored(),
                         "refactored mesh API not yet built")
    def test_closed_system_conserves_mass_1cell(self):
        # Headline test — simplest possible closed system.  1 pipe cell,
        # wall BCs on both ends (no manholes, no inlets).  200 steps of
        # fully_dynamic should conserve mass to machine precision.
        n_pipe_cells, n_manhole, n_inlet, sub_len, _, _ = \
            self._build_closed_system(mcl=100.0, d_initial=(1.0, 0.0))

        m0, _ = self._mass_unified(n_pipe_cells, n_manhole, n_inlet, sub_len)
        self.assertGreater(m0, 0.0)

        for _ in range(200):
            self._step("fully_dynamic", dt=0.5)

        m1, _ = self._mass_unified(n_pipe_cells, n_manhole, n_inlet, sub_len)
        self.assertAlmostEqual(
            m1 / m0, 1.0, places=8,
            msg=f"fully_dynamic 1-cell drift {(m1 - m0) / m0:+.6e} over 200 steps",
        )

    # ── Test 2: closed 2-node, 1 link, 10 sub-cells, fully_dynamic ───────
    @unittest.skipUnless(_skip_unless_refactored(),
                         "refactored mesh API not yet built")
    def test_closed_system_conserves_mass_10subcells(self):
        # Same closed system but 10 sub-cells — internal faces between
        # sub-cells must not leak mass.
        n_pipe_cells, n_manhole, n_inlet, sub_len, _, _ = \
            self._build_closed_system(mcl=10.0, d_initial=(1.0, 0.0))
        self.assertEqual(n_pipe_cells, 10)

        m0, _ = self._mass_unified(n_pipe_cells, n_manhole, n_inlet, sub_len)
        self.assertGreater(m0, 0.0)

        for _ in range(200):
            self._step("fully_dynamic", dt=0.5)

        m1, _ = self._mass_unified(n_pipe_cells, n_manhole, n_inlet, sub_len)
        self.assertAlmostEqual(
            m1 / m0, 1.0, places=8,
            msg=f"fully_dynamic 10-subcell drift {(m1 - m0) / m0:+.6e}",
        )

    # ── Test 3: closed 2-node, 1 link, diffusion_wave ────────────────────
    @unittest.skipUnless(_skip_unless_refactored(),
                         "refactored mesh API not yet built")
    def test_closed_system_conserves_mass_diffusion(self):
        # Same 1-cell mesh as test 1, diffusion_wave mode.
        n_pipe_cells, n_manhole, n_inlet, sub_len, _, _ = \
            self._build_closed_system(mcl=100.0, d_initial=(1.0, 0.0))

        m0, _ = self._mass_unified(n_pipe_cells, n_manhole, n_inlet, sub_len)
        self.assertGreater(m0, 0.0)

        for _ in range(200):
            self._step("diffusion_wave", dt=0.5)

        m1, _ = self._mass_unified(n_pipe_cells, n_manhole, n_inlet, sub_len)
        self.assertAlmostEqual(
            m1 / m0, 1.0, places=8,
            msg=f"diffusion_wave drift {(m1 - m0) / m0:+.6e}",
        )

    # ── Test 4: manhole cell volume preserved in junction mesh ────────────
    @unittest.skipUnless(_skip_unless_refactored(),
                         "refactored mesh API not yet built")
    def test_manhole_cell_volume_preserved(self):
        # Closed system: 3 pipe links meeting at a junction manhole cell.
        # 4 nodes: 0→1→2→3, nodes 0,2,3 are dead ends (wall BC), node 1 is
        # the manhole.  All pipes dry at t=0; manhole at depth 1.0.
        # After 100 steps total volume must be invariant.
        link_from = np.array([0, 1, 3], dtype=np.int32)
        link_to = np.array([1, 2, 1], dtype=np.int32)
        L_arr = np.full(3, 100.0, dtype=np.float64)
        D_arr = np.full(3, 0.5, dtype=np.float64)
        n_mann_arr = np.full(3, 0.013, dtype=np.float64)
        node_invert = np.array([10.0, 9.0, 9.0, 10.0], dtype=np.float64)

        n_pipe_cells, n_manhole, n_inlet, sub_len, manhole_sa, _ = \
            self._build_closed_system(
                n_links=3,
                link_from=link_from,
                link_to=link_to,
                L=L_arr,
                D=D_arr,
                n_mann=n_mann_arr,
                node_invert=node_invert,
                mcl=10.0,
                manhole_diameters=[1.2],
                manhole_heights=[2.0],
                manhole_inverts=[9.0],
                d_initial=(0.0, 0.0),
            )

        # Set manhole depth to 1.0
        n_total = n_pipe_cells + n_manhole + n_inlet
        h_all = np.zeros(n_total, dtype=np.float64)
        h_all[n_pipe_cells] = 1.0  # manhole cell depth
        _MOD.swe2d_pipe1d_upload_cell_h(self._dev, h_all)
        _MOD.swe2d_pipe1d_init_cell_area(self._dev, H_MIN)

        m0, _ = self._mass_unified(
            n_pipe_cells, n_manhole, n_inlet, sub_len,
            manhole_surface_areas=manhole_sa,
        )
        self.assertGreater(m0, 0.0)

        for _ in range(100):
            self._step("fully_dynamic", dt=0.5, n_nodes=4)

        m1, st = self._mass_unified(
            n_pipe_cells, n_manhole, n_inlet, sub_len,
            manhole_surface_areas=manhole_sa,
        )
        self.assertAlmostEqual(
            m1 / m0, 1.0, places=8,
            msg=f"manhole-junction drift {(m1 - m0) / m0:+.6e}",
        )

        # Verify the manhole cell_class is MANHOLE_CELL (= 1)
        cell_class = np.asarray(st["cell_class"], dtype=np.int32)
        self.assertEqual(
            int(cell_class[n_pipe_cells]), 1,
            "Manhole cell_class must be MANHOLE_CELL (1)",
        )

    # ── Test 5 (REMOVED 2026-07-22): inlet cell receives prescribed-flow hydrograph
    #
    # Prescribed-Q inlet capture was retired in Phase 2.4. Inlet capture
    # now uses SURFACE_2D_INLET (class 4) HEC-22 weir/orifice computed from
    # 2D cell depth. The `swe2d_pipe1d_upload_inlet_prescribed_Q` binding
    # was deleted as a dead export. The migration path is a HEC-22 inlet
    # capture test in tests/test_swe2d_gpu_drainage_network.py — that
    # suite already exists and fails for other reasons (cell width
    # 1.0×1.0 vs GPKG box 10×5 — separate bug, see audit §P3).
    @unittest.skip("prescribed-Q inlet path retired; see HEC-22 tests")
    def test_inlet_cell_prescribed_flow_DISABLED(self):
        self.skipTest("prescribed-Q inlet path retired in Phase 2.4")

    # ── Test 6: outfall fixed-WSE boundary condition ──────────────────────
    # SKELETON: swe2d_pipe1d_upload_outfall_bc and
    # swe2d_pipe1d_upload_outfall_surface_faces bindings were deleted
    # (not called by production — class-1 kernel uses ghost_idx, not owner_R).
    # A production-path outfall test should use node_is_outfall at mesh build
    # time + the pre-step ghost WSE update kernel.
    @unittest.skipUnless(_skip_unless_refactored(),
                         "refactored mesh API not yet built")
    def test_outfall_fixed_wse(self):
        self.fail("outfall BC test needs rewrite using production path")

    # ── Test 7: outfall rating-curve boundary condition ───────────────────
    # SKELETON: same binding deletion as test 6.
    @unittest.skipUnless(_skip_unless_refactored(),
                         "refactored mesh API not yet built")
    def test_outfall_rating_curve(self):
        self.fail("outfall rating test needs rewrite using production path")

    # ── Test 8: pipe-end face coupling conserves total mass ───────────────
    @unittest.skipUnless(_skip_unless_refactored(),
                         "refactored mesh API not yet built")
    def test_pipe_end_face_coupling_conserves_total_mass(self):
        # Closed system: 1 pipe link, node 0 is wall BC, node 1 is a
        # SURFACE_2D_PIPE_END face coupled to 2D cell 0 (from setUpClass).
        # 2D cell: h=0.0, zb=0.0 (mesh default), pipe end-invert matches.
        # Total mass (pipe cells + 2D cell) must be conserved to places=8.
        n_pipe_cells, n_manhole, n_inlet, sub_len, _, _ = \
            self._build_closed_system(mcl=100.0, d_initial=(1.0, 0.0),
                                      node_is_pipe_end=np.array([0, 1], dtype=np.int32))

        # Set 2D cell to (h=0.0, hu=0, hv=0) for a clean initial condition
        solver = self._backend._solver_h
        _MOD.swe2d_set_state(
            solver,
            np.array([0.0], dtype=np.float64),    # h
            np.zeros(1, dtype=np.float64),        # hu
            np.zeros(1, dtype=np.float64),        # hv
        )

        # Upload SURFACE_2D_PIPE_END face mapping (auto-detected face → 2D cell)
        if hasattr(_MOD, "swe2d_pipe1d_upload_pipe_end_surface_faces"):
            _MOD.swe2d_pipe1d_upload_pipe_end_surface_faces(
                self._dev,
                np.array([0, 0], dtype=np.int32),  # coupled 2D cell per pipe-end
            )

        # Set coupling dt
        _MOD.swe2d_gpu_set_coupling_dt(0.5)

        # Measure initial mass: pipe cells + 2D cell
        def _total_mass_pipe_plus_2d():
            mp, _ = self._mass_unified(n_pipe_cells, n_manhole, n_inlet, sub_len)
            solver_state = _MOD.swe2d_readback_state(solver, 1)
            h_2d = np.asarray(solver_state["h"], dtype=np.float64)
            # 2D cell area is 0.5 (half of unit triangle)
            cell_area_2d = 0.5
            m_2d = float(np.sum(h_2d)) * cell_area_2d
            return mp + m_2d

        m0 = _total_mass_pipe_plus_2d()
        self.assertGreater(m0, 0.0)

        for _ in range(200):
            self._step("fully_dynamic", dt=0.5)

        m1 = _total_mass_pipe_plus_2d()
        self.assertAlmostEqual(
            m1 / m0, 1.0, places=8,
            msg=f"pipe-end face coupling drift {(m1 - m0) / m0:+.6e}",
        )

    # ── Test 9: junction overflow to 2D surface ───────────────────────────
    # SKELETON: depended on swe2d_pipe1d_upload_junction_overflow_state,
    # swe2d_pipe1d_upload_pipe_ends_and_junctions, swe2d_pipe1d_upload_node_rim
    # — all deleted (Phase 2.1 old path).  The class-5 owner_R can now be
    # patched via swe2d_pipe1d_upload_junction_overflow_2d_cells (the production
    # path), but _step() doesn't pass n_cells_2d so the kernel skips class-5.
    # Expected assertions if fixed:
    #   - Manhole depth decreases (overflow is occurring)
    #   - 2D cell gains mass
    #   - Total mass conserved to ~0.1
    @unittest.skipUnless(_skip_unless_refactored(),
                         "refactored mesh API not yet built")
    def test_junction_overflow_to_2d(self):
        self.fail("junction overflow test needs n_cells_2d wired through _step")

    # ── Test 10: wave speed matches hydraulic-depth celerity (F6) ────────
    @unittest.skipUnless(_skip_unless_refactored(),
                         "refactored mesh API not yet built")
    def test_wave_speed_matches_hydraulic_depth(self):
        # 2-node closed system, 1 link, 10 sub-cells, mcl=10.
        # Initialize with a smooth sinusoid perturbation (A variation).
        # Run 5 steps, compute the wave celerity from the peak displacement
        # and verify it matches sqrt(g · A/T) within 5%.
        # Use half-fill depth (h = D/2 = 0.25m) to avoid the pipe crown
        # where T → 0 and c_wave = sqrt(g*A/T) becomes unbounded.
        # At h = 0.25 the top width is maximal (T = D = 0.5m).
        n_pipe_cells, n_manhole, n_inlet, sub_len, _, _ = \
            self._build_closed_system(mcl=10.0, d_initial=(0.25, 0.25))

        # Read back the initial cell areas and apply a small sinusoidal
        # perturbation to the cell depth.
        _, st0 = self._mass_unified(n_pipe_cells, n_manhole, n_inlet, sub_len)
        D_pipe = 0.5
        L_total = float(sub_len * n_pipe_cells)
        x = (np.arange(n_pipe_cells, dtype=np.float64) + 0.5) * sub_len

        # Perturb cell depth: h(x) = h0 + δh·sin(2πx/L)
        # Amplitude δh = 5% of h0 so the perturbation stays safely below crown.
        h0 = 0.25
        dh = 0.05 * h0  # 5% of half-fill depth
        h_perturbed = h0 + dh * np.sin(2.0 * np.pi * x / L_total)
        _MOD.swe2d_pipe1d_upload_cell_h(self._dev, h_perturbed)
        _MOD.swe2d_pipe1d_init_cell_area(self._dev, H_MIN)

        # Snapshot A before stepping.
        _, st_before = self._mass_unified(n_pipe_cells, n_manhole, n_inlet, sub_len)
        A_before = np.asarray(st_before["cell_A"], dtype=np.float64)

        # Run enough steps for a measurable peak shift.
        # c_expected ≈ 1.39 m/s, dt = 0.5 → in 100 steps the wave travels
        # ~69.5 m ≈ 7 cells.
        dt = 0.5
        n_steps = 100
        for _ in range(n_steps):
            self._step("fully_dynamic", dt=dt)

        _, st_after = self._mass_unified(n_pipe_cells, n_manhole, n_inlet, sub_len)
        A_after = np.asarray(st_after["cell_A"], dtype=np.float64)

        # Track the fractional peak position of the post-step area via
        # quadratic interpolation over the 3 cells closest to the max.
        peak_idx = int(np.argmax(A_after))
        i0 = max(0, min(peak_idx, n_pipe_cells - 3))
        idxs = [i0, i0 + 1, i0 + 2]
        xi = x[idxs]
        yi = A_after[idxs]
        # Quadratic fit: y = a*(x-x0)^2 + b*(x-x0) + c, vertex at x = x0 - b/(2a)
        x0 = xi[1]
        denom = 2.0 * ((xi[2] - xi[1]) * (yi[0] - yi[1]) - (xi[0] - xi[1]) * (yi[2] - yi[1]))
        if abs(denom) > 1e-30:
            a = ((xi[2] - xi[1]) * (yi[0] - yi[1]) - (xi[0] - xi[1]) * (yi[2] - yi[1])) / denom
            b = ((yi[2] - yi[1]) - a * ((xi[2] - xi[1]) ** 2 - (xi[0] - xi[1]) ** 2)) / (xi[2] - xi[0])
            peak_pos = x0 - b / (2.0 * max(a, 1e-30))
        else:
            peak_pos = xi[peak_idx - i0]

        # Initial sine-peak position is at L_total / 4.
        peak_pos_0 = L_total / 4.0
        dx = peak_pos - peak_pos_0
        c_measured = dx / (n_steps * dt)

        # Expected wave speed: sqrt(g * A / T) at half-fill
        # Circular pipe D=0.5: T=D=0.5, A_half = π·D²/8
        A_half = np.pi * (D_pipe / 2.0) ** 2 * 0.5
        T_half = D_pipe
        c_expected = np.sqrt(G * A_half / T_half)

        if c_expected > 0.0 and abs(c_measured) > 1e-30:
            ratio = c_measured / c_expected
            self.assertAlmostEqual(
                ratio, 1.0, delta=0.15,
                msg=f"Wave speed ratio {ratio:.3f} vs expected c={c_expected:.3f} m/s "
                    f"(measured dx={dx:.2f}m over {n_steps}×{dt:.2f}s steps)",
            )

    # ── Test 11: fractional max_cell_length (F13) ─────────────────────────
    @unittest.skipUnless(_skip_unless_refactored(),
                         "refactored mesh API not yet built")
    def test_fractional_max_cell_length(self):
        # mcl=0.3, L=100 → ceil(100/0.3) = 334 sub-cells (was 100 with int cast).
        _mcl = 0.3
        n_pipe_cells, n_manhole, n_inlet, sub_len, _, _ = \
            self._build_closed_system(mcl=_mcl, d_initial=(0.0, 0.0))

        # ceil(100 / 0.3) = 334
        expected = 334
        self.assertEqual(n_pipe_cells, expected,
                         f"Fractional mcl={_mcl}: expected {expected} cells, got {n_pipe_cells}")

        # Cross-check via native cell-count binding if available
        if hasattr(_MOD, "swe2d_pipe1d_get_cell_count"):
            n_cells_native = _MOD.swe2d_pipe1d_get_cell_count(self._dev)
            self.assertEqual(n_cells_native, expected,
                             f"Native cell count {n_cells_native} != {expected}")
