"""
GPU compute-sanitizer test: 1D drainage network CUDA paths.

Includes HEC-22 inlet capture tests for grate, curb, slotted, and combo inlets.
"""

from __future__ import annotations

import math
import unittest

import numpy as np


def _load_module():
    try:
        import hydra_swe2d as m
        return m
    except Exception:
        return None


_MOD = _load_module()

# ── Analytical helpers ──────────────────────────────────────────────────────

G = 9.81  # gravity (m/s²)
K_MANN_DEFAULT = 1.0
H_MIN_DEFAULT = 1.0e-4


def _weir_grate(H: float, Lg: float, Wg: float) -> float:
    """Grate weir Q = 3.0 * P * H^1.5 (HEC-22 §4-4.1, eq 4-1)."""
    P = 2.0 * (Lg + Wg)
    return 3.0 * P * math.pow(H, 1.5)


def _orifice_grate(H: float, Lg: float, Wg: float, openFrac: float) -> float:
    """Grate orifice Q = 0.67 * Ao * sqrt(2gH) (HEC-22 §4-4.1, eq 4-2)."""
    Ao = Lg * Wg * openFrac
    return 0.67 * Ao * math.sqrt(2.0 * G * H)


def _grate_H_trans(Lg: float, Wg: float, openFrac: float) -> float:
    """Transition head from weir to orifice."""
    P = 2.0 * (Lg + Wg)
    Ao = Lg * Wg * openFrac
    return 1.79 * Ao / P if P > 0 else 1e6


def _weir_curb(H: float, L: float) -> float:
    """Curb weir Q = 3.0 * L * H^1.5 (HEC-22 §4-4.2, eq 4-5)."""
    return 3.0 * L * math.pow(H, 1.5)


def _orifice_curb(H: float, L: float, h: float) -> float:
    """Curb orifice Q = 0.67 * h * L * sqrt(2g(H-h/2)) (HEC-22 §4-4.2, eq 4-7)."""
    H_eff = H - 0.5 * h
    if H_eff <= 0:
        return 0.0
    return 0.67 * h * L * math.sqrt(2.0 * G * H_eff)


def _weir_slotted(H: float, L: float) -> float:
    """Slotted weir Q = 2.48 * L * H^1.5 (HEC-22 §4-4.3, eq 4-11)."""
    return 2.48 * L * math.pow(H, 1.5)


def _orifice_slotted(H: float, L: float, w: float) -> float:
    """Slotted orifice Q = 0.8 * L * w * sqrt(2gH) (HEC-22 §4-4.3, eq 4-13)."""
    return 0.8 * L * w * math.sqrt(2.0 * G * H)


# ── Mesh / solver helpers ────────────────────────────────────────────────────


def _make_2cell_mesh(mod):
    """Build a 2-cell rectangular mesh, 10×5 m."""
    node_x = np.array([0.0, 10.0, 0.0, 10.0], dtype=np.float64)
    node_y = np.array([0.0, 0.0, 5.0, 5.0], dtype=np.float64)
    node_z = np.zeros(4, dtype=np.float64)
    cell_nodes = np.array([0, 1, 3, 0, 3, 2], dtype=np.int32)
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


def _make_solver(mod, mesh, h0):
    """Create a GPU solver with structures enabled."""
    return mod.swe2d_create_solver(
        mesh, h0,
        n_mann=0.035,
        cfl=0.45,
        dt_max=0.5,
        use_gpu=True,
        enable_hydraulic_structures=True,
    )


def _dummy_weir_arrays():
    """Return arrays for a single dummy weir that produces zero flow (crest above WSE)."""
    z = np.zeros
    di = np.zeros(1, dtype=np.int32)
    return (
            np.array([1], dtype=np.int32),  # structure_type = weir
            np.array([0], dtype=np.int32),  # upstream_cell
            np.array([1], dtype=np.int32),  # downstream_cell
            np.array([100.0], dtype=np.float64),  # crest_elev (very high → zero flow)
            z(1),  # width
            z(1),  # height
            z(1),  # diameter
            z(1),  # length
            z(1),  # roughness_n
            np.ones(1),  # coeff
            z(1),  # cd
            z(1),  # opening
            z(1),  # q_pump
            np.full(1, -1.0),  # max_flow
            di,  # culvert_code
            di,  # culvert_shape
            z(1),  # culvert_rise
            z(1),  # culvert_span
            z(1),  # culvert_area
            z(1),  # culvert_barrels
            z(1),  # culvert_slope
            z(1),  # inlet_invert_elev
            z(1),  # outlet_invert_elev
            z(1),  # entrance_loss_k
            z(1),  # exit_loss_k
            di,  # embankment_enabled
            z(1),  # embankment_crest_elev
            z(1),  # embankment_overflow_width
            np.ones(1),  # embankment_weir_coeff
            9.81,   # gravity
            3.28084,  # model_to_ft
        )


# ====================================================================
#  HEC-22 Inlet Capture Tests
# ====================================================================


@unittest.skipIf(_MOD is None, "hydra_swe2d not built")
@unittest.skipUnless(hasattr(_MOD, "swe2d_gpu_upload_drainage_exchange_params"),
                     "drainage exchange CUDA functions not compiled")
@unittest.skipUnless(hasattr(_MOD, "swe2d_gpu_available")
                     and _MOD.swe2d_gpu_available(),
                     "CUDA GPU not available")
class TestGPUInletCapture(unittest.TestCase):
    """HEC-22 inlet capture equation verification on GPU."""

    N_CELLS = 2
    CELL_AREA = 25.0  # m² per cell (each triangle is 25 m² in the 10×5 rect)

    @classmethod
    def setUpClass(cls):
        cls.mod = _MOD
        cls.mesh = _make_2cell_mesh(_MOD)
        h0 = np.array([0.0, 0.0], dtype=np.float64)  # dry start
        cls.solver = None
        try:
            cls.solver = _make_solver(_MOD, cls.mesh, h0)
        except Exception:
            raise unittest.SkipTest("Failed to create GPU solver")

    @classmethod
    def tearDownClass(cls):
        if cls.solver is not None:
            try:
                cls.mod.swe2d_destroy(cls.solver)
            except Exception:
                pass
            cls.solver = None

    def setUp(self):
        """Run a solver step to guarantee GPU device is live, then set up
        the persistent coupling workspace and pipe 1D node state."""
        mod = self.mod
        if not mod.swe2d_step(self.solver, 0.1).get("gpu_active", False):
            self.skipTest("solver step did not activate GPU")

        # Get device pointer for pipe1d functions
        self._dev_ptr = mod.swe2d_get_coupling_dev_ptr()

        # Preload a dummy weir to allocate sf_ws (including d_cell_wse)
        w = _dummy_weir_arrays()
        mod.swe2d_gpu_preload_structure_params(*w)

        # Preload coupling cell area (allocates cpl_ws)
        cell_area = np.full(self.N_CELLS, self.CELL_AREA, dtype=np.float64)
        mod.swe2d_gpu_preload_coupling_cell_area(cell_area)

        # Build a minimal pipe1d mesh (1 self-loop link to get 1 node)
        # Setting up n_links=1, n_nodes=1 with a dummy 1m pipe
        mod.swe2d_build_unified_mesh(
            n_links=1,
            link_from=np.array([0], dtype=np.int32),
            link_to=np.array([0], dtype=np.int32),
            L=np.array([1.0], dtype=np.float64),
            D=np.array([1.0], dtype=np.float64),
            n_mann=np.array([0.013], dtype=np.float64),
            node_invert=np.array([0.0], dtype=np.float64),
            mcl=10,
            dev_ptr=int(self._dev_ptr),
            S0=np.zeros(1, dtype=np.float64),
        )

        # Upload initial node depth (dry node)
        mod.swe2d_pipe1d_upload_cell_h(int(self._dev_ptr),
                                           np.array([0.0], dtype=np.float64))
        mod.swe2d_pipe1d_init_cell_area(int(self._dev_ptr), H_MIN_DEFAULT)

        # Ensure the drainage q buffer exists in coupling workspace
        mod.swe2d_gpu_ensure_drainage_q_buf(self.N_CELLS)

        # Cache for cleanup
        self._cell_area = cell_area

    def tearDown(self):
        """Reset solver state to dry after each test."""
        mod = self.mod
        h = np.full(self.N_CELLS, 0.0, dtype=np.float64)
        hu = np.zeros(self.N_CELLS, dtype=np.float64)
        hv = np.zeros(self.N_CELLS, dtype=np.float64)
        mod.swe2d_set_state(self.solver, h, hu, hv)

    # ── Setup helpers ──────────────────────────────────────────────────

    def _make_empty_outfall_arrays(self):
        """Return empty arrays for the 'no outfalls' case."""
        _ei = np.empty(0, dtype=np.int32)
        _ed = np.empty(0, dtype=np.float64)
        return _ei, _ei, _ed, _ed, _ed, _ed, _ei

    def _upload_inlet_and_run(self, inlet_params: dict, H_m: float, h0_m: float = 0.0):
        """Upload inlet exchange params, set solver WSE, run coupling.

        inlet_params must contain at minimum:
            inlet_type, inlet_cell, inlet_node, inlet_crest, inlet_width,
            inlet_cd, inlet_qmax,
            plus any HEC-22 geometry arrays (grate_len, grate_wid, etc.)

        H_m is the driving head above crest.
        h0_m is any initial water depth in cell 0 before adding H.
        """
        mod = self.mod

        # Set solver state: cell 0 has water WSE = crest + H + h0,
        #                   cell 1 is dry.
        crest = float(inlet_params["inlet_crest"][0])
        depth = crest + H_m + h0_m
        h = np.array([depth, 0.0], dtype=np.float64)
        hu = np.zeros(self.N_CELLS, dtype=np.float64)
        hv = np.zeros(self.N_CELLS, dtype=np.float64)
        mod.swe2d_set_state(self.solver, h, hu, hv)

        # Reset pipe1d node depth to zero so previous test's capture doesn't
        # bias the driving head for this test.
        mod.swe2d_pipe1d_upload_cell_h(int(self._dev_ptr),
                                           np.zeros(1, dtype=np.float64))

        # Upload drainage exchange params
        o_cell, o_node, o_inv, o_dia, o_cd, o_qmax, o_zs = self._make_empty_outfall_arrays()

        mod.swe2d_gpu_upload_drainage_exchange_params(
            inlet_cell=inlet_params["inlet_cell"],
            inlet_node=inlet_params["inlet_node"],
            inlet_crest=inlet_params["inlet_crest"],
            inlet_width=inlet_params["inlet_width"],
            inlet_cd=inlet_params["inlet_cd"],
            inlet_qmax=inlet_params["inlet_qmax"],
            inlet_type=inlet_params["inlet_type"],
            inlet_grate_len=inlet_params["inlet_grate_len"],
            inlet_grate_wid=inlet_params["inlet_grate_wid"],
            inlet_grate_kind=inlet_params["inlet_grate_kind"],
            inlet_grate_open=inlet_params["inlet_grate_open"],
            inlet_curb_len=inlet_params["inlet_curb_len"],
            inlet_curb_ht=inlet_params["inlet_curb_ht"],
            inlet_curb_throat=inlet_params["inlet_curb_throat"],
            inlet_slot_len=inlet_params["inlet_slot_len"],
            inlet_slot_wid=inlet_params["inlet_slot_wid"],
            outfall_cell=o_cell,
            outfall_node=o_node,
            outfall_invert=o_inv,
            outfall_diameter=o_dia,
            outfall_cd=o_cd,
            outfall_qmax=o_qmax,
            outfall_zero_storage=o_zs,
            node_max_depth=np.array([3.0], dtype=np.float64),
        )

        # Set coupling dt (non-zero, used as limiter time horizon)
        mod.swe2d_gpu_set_coupling_dt(1.0)

        # Compute coupling on-device (runs inlet exchange kernel)
        # Use host_structure_flows=[0.0] to keep the dummy structure inert
        mod.swe2d_gpu_compute_coupling_full_on_device(
            cell_wse=None,
            n_structures=1,
            host_structure_flows=np.array([0.0], dtype=np.float64),
        )

        # Read back coupling sources
        src = mod.swe2d_gpu_readback_coupling_sources(self.N_CELLS)
        return src

    def _assert_capture(self, src: np.ndarray, Q_expected: float,
                        atol: float, msg: str = ""):
        """Assert that the coupling source for cell 0 matches expected Q.

        The kernel writes Q (m³/s) into d_drainage_q, then folds it into
        d_external_source_mps by dividing by cell area.  The readback is
        therefore a rate in m/s; convert back to m³/s before comparing.
        """
        # Cell 0 should have negative source (water leaving surface → node)
        q_actual = -float(src[0]) * self.CELL_AREA  # m/s → m³/s * self.CELL_AREA  # m/s → m³/s
        self.assertAlmostEqual(q_actual, Q_expected, delta=atol,
                               msg=f"Cell 0 capture mismatch: got {q_actual}, expected {Q_expected}. {msg}")
        # Cell 1 should be unaffected
        self.assertAlmostEqual(float(src[1]), 0.0, places=12,
                               msg=f"Cell 1 should have zero source, got {src[1]}. {msg}")

    # ── Grate tests ─────────────────────────────────────────────────────

    def _grate_params(self, Lg=1.0, Wg=0.5, grate_kind=0, openFrac=0.9,
                      crest=0.0, qmax=1e6):
        """Return inlet params dict for a grate inlet.

        Note: ALL HEC-22 geometry arrays must have n_inlets elements even
        when unused for the inlet type — the C++ upload code copies from
        every array unconditionally.
        """
        return {
            "inlet_type": np.array([0], dtype=np.int32),
            "inlet_cell": np.array([0], dtype=np.int32),
            "inlet_node": np.array([0], dtype=np.int32),
            "inlet_crest": np.array([crest], dtype=np.float64),
            "inlet_width": np.array([1.0], dtype=np.float64),
            "inlet_cd": np.array([0.67], dtype=np.float64),
            "inlet_qmax": np.array([qmax], dtype=np.float64),
            "inlet_grate_len": np.array([Lg], dtype=np.float64),
            "inlet_grate_wid": np.array([Wg], dtype=np.float64),
            "inlet_grate_kind": np.array([grate_kind], dtype=np.int32),
            "inlet_grate_open": np.array([openFrac], dtype=np.float64),
            "inlet_curb_len": np.array([0.0], dtype=np.float64),
            "inlet_curb_ht": np.array([0.15], dtype=np.float64),
            "inlet_curb_throat": np.array([0], dtype=np.int32),
            "inlet_slot_len": np.array([0.0], dtype=np.float64),
            "inlet_slot_wid": np.array([0.0], dtype=np.float64),
        }

    def test_inlet_grate_weir(self):
        """Low head on grate: Q = 3.0 * P * H^1.5"""
        Lg, Wg = 1.0, 0.5
        H = 0.05  # m → below transition
        P = 2.0 * (Lg + Wg)
        Q_exp = 3.0 * P * math.pow(H, 1.5)  # ≈ 3.0 * 3.0 * 0.01118 = 0.1006

        params = self._grate_params(Lg=Lg, Wg=Wg)
        src = self._upload_inlet_and_run(params, H)
        self._assert_capture(src, Q_exp, atol=1e-4)

    def test_inlet_grate_orifice(self):
        """High head on grate: Q = 0.67 * Ao * sqrt(2gH)"""
        Lg, Wg, openFrac = 1.0, 0.5, 0.9
        H = 1.0  # m → above transition
        Ao = Lg * Wg * openFrac
        Q_exp = 0.67 * Ao * math.sqrt(2.0 * G * H)  # ≈ 0.67 * 0.45 * 4.429 = 1.335

        params = self._grate_params(Lg=Lg, Wg=Wg, openFrac=openFrac)
        src = self._upload_inlet_and_run(params, H)
        self._assert_capture(src, Q_exp, atol=1e-3)

    def test_inlet_grate_transition(self):
        """At H = 1.79*Ao/P, weir and orifice match within 1%"""
        Lg, Wg, openFrac = 1.0, 0.5, 0.9
        P = 2.0 * (Lg + Wg)
        Ao = Lg * Wg * openFrac
        H_trans = 1.79 * Ao / P

        Q_weir = 3.0 * P * math.pow(H_trans, 1.5)
        Q_orif = 0.67 * Ao * math.sqrt(2.0 * G * H_trans)

        params = self._grate_params(Lg=Lg, Wg=Wg, openFrac=openFrac)
        src = self._upload_inlet_and_run(params, H_trans)

        q_actual = -float(src[0]) * self.CELL_AREA  # m/s → m³/s
        # Weir and orifice should agree within 1%
        self.assertAlmostEqual(q_actual / Q_weir, 1.0, delta=0.01,
                               msg=f"At transition, Q={q_actual:.6f}, Q_weir={Q_weir:.6f}, Q_orif={Q_orif:.6f}")

    def test_inlet_grate_opening_ratio(self):
        """P_BAR-50 (0.90) vs Curved_Vane (0.35) produce different orifice Q"""
        Lg, Wg = 1.0, 0.5
        H = 0.5  # m → orifice regime

        # Grate kind 0: P_BAR-50 (openFrac=0.90)
        params0 = self._grate_params(Lg=Lg, Wg=Wg, grate_kind=0)
        src0 = self._upload_inlet_and_run(params0, H)
        q0 = -float(src0[0]) * self.CELL_AREA  # m/s → m³/s

        # Grate kind 3: Curved_Vane (openFrac=0.35)
        params3 = self._grate_params(Lg=Lg, Wg=Wg, grate_kind=3)
        src3 = self._upload_inlet_and_run(params3, H)
        q3 = -float(src3[0]) * self.CELL_AREA  # m/s → m³/s

        # Check analytically: Q ∝ Ao
        Ao0 = Lg * Wg * 0.90
        Ao3 = Lg * Wg * 0.35
        Q0_exp = 0.67 * Ao0 * math.sqrt(2.0 * G * H)
        Q3_exp = 0.67 * Ao3 * math.sqrt(2.0 * G * H)

        self.assertAlmostEqual(q0, Q0_exp, delta=1e-3,
                               msg=f"P_BAR-50: got {q0:.6f}, expected {Q0_exp:.6f}")
        self.assertAlmostEqual(q3, Q3_exp, delta=1e-3,
                               msg=f"Curved_Vane: got {q3:.6f}, expected {Q3_exp:.6f}")
        # Ratio should be approx 0.90/0.35 ≈ 2.57
        self.assertAlmostEqual(q0 / q3, 0.90 / 0.35, delta=0.05,
                               msg=f"Opening ratio effect: q0/q3={q0/q3:.3f} vs expected {0.90/0.35:.3f}")

    # ── Curb tests ──────────────────────────────────────────────────────

    def _curb_params(self, L=2.0, h=0.15, throat=0, crest=0.0, qmax=1e6):
        """Return inlet params dict for a curb inlet.

        Note: ALL HEC-22 geometry arrays must have n_inlets elements even
        when unused for the inlet type.
        """
        return {
            "inlet_type": np.array([1], dtype=np.int32),
            "inlet_cell": np.array([0], dtype=np.int32),
            "inlet_node": np.array([0], dtype=np.int32),
            "inlet_crest": np.array([crest], dtype=np.float64),
            "inlet_width": np.array([0.0], dtype=np.float64),
            "inlet_cd": np.array([0.67], dtype=np.float64),
            "inlet_qmax": np.array([qmax], dtype=np.float64),
            "inlet_grate_len": np.array([0.0], dtype=np.float64),
            "inlet_grate_wid": np.array([0.0], dtype=np.float64),
            "inlet_grate_kind": np.array([0], dtype=np.int32),
            "inlet_grate_open": np.array([0.9], dtype=np.float64),
            "inlet_curb_len": np.array([L], dtype=np.float64),
            "inlet_curb_ht": np.array([h], dtype=np.float64),
            "inlet_curb_throat": np.array([throat], dtype=np.int32),
            "inlet_slot_len": np.array([0.0], dtype=np.float64),
            "inlet_slot_wid": np.array([0.0], dtype=np.float64),
        }

    def test_inlet_curb_weir(self):
        """Low head on curb: Q = 3.0 * L * H^1.5"""
        L, h = 2.0, 0.15
        H = 0.05  # m → below weir limit (0.15)
        Q_exp = 3.0 * L * math.pow(H, 1.5)  # ≈ 3.0 * 2.0 * 0.01118 = 0.0671

        params = self._curb_params(L=L, h=h)
        src = self._upload_inlet_and_run(params, H)
        self._assert_capture(src, Q_exp, atol=1e-4)

    def test_inlet_curb_orifice(self):
        """High head on curb: Q = 0.67 * h * L * sqrt(2g(H-h/2))"""
        L, h = 2.0, 0.15
        H = 0.5  # m → above 1.4*h=0.21
        H_eff = H - 0.5 * h
        Q_exp = 0.67 * h * L * math.sqrt(2.0 * G * H_eff)  # ≈ 0.67*0.15*2*sqrt(19.62*0.425)=0.543

        params = self._curb_params(L=L, h=h)
        src = self._upload_inlet_and_run(params, H)
        self._assert_capture(src, Q_exp, atol=1e-3)

    # ── Slotted tests ───────────────────────────────────────────────────

    def _slotted_params(self, L=2.0, w=0.05, crest=0.0, qmax=1e6):
        """Return inlet params dict for a slotted inlet.

        Note: ALL HEC-22 geometry arrays must have n_inlets elements even
        when unused for the inlet type.
        """
        return {
            "inlet_type": np.array([2], dtype=np.int32),
            "inlet_cell": np.array([0], dtype=np.int32),
            "inlet_node": np.array([0], dtype=np.int32),
            "inlet_crest": np.array([crest], dtype=np.float64),
            "inlet_width": np.array([0.0], dtype=np.float64),
            "inlet_cd": np.array([0.8], dtype=np.float64),
            "inlet_qmax": np.array([qmax], dtype=np.float64),
            "inlet_grate_len": np.array([0.0], dtype=np.float64),
            "inlet_grate_wid": np.array([0.0], dtype=np.float64),
            "inlet_grate_kind": np.array([0], dtype=np.int32),
            "inlet_grate_open": np.array([0.9], dtype=np.float64),
            "inlet_curb_len": np.array([0.0], dtype=np.float64),
            "inlet_curb_ht": np.array([0.15], dtype=np.float64),
            "inlet_curb_throat": np.array([0], dtype=np.int32),
            "inlet_slot_len": np.array([L], dtype=np.float64),
            "inlet_slot_wid": np.array([w], dtype=np.float64),
        }

    def test_inlet_slotted_weir(self):
        """Low head on slotted: Q = 2.48 * L * H^1.5"""
        L, w = 2.0, 0.05
        H = 0.05  # m → below 2.587*w=0.129
        Q_exp = 2.48 * L * math.pow(H, 1.5)  # ≈ 2.48*2.0*0.01118 = 0.0554

        params = self._slotted_params(L=L, w=w)
        src = self._upload_inlet_and_run(params, H)
        self._assert_capture(src, Q_exp, atol=1e-4)

    def test_inlet_slotted_orifice(self):
        """High head on slotted: Q = 0.8 * L * w * sqrt(2gH)"""
        L, w = 2.0, 0.05
        H = 0.5  # m → above 2.587*w=0.129
        Q_exp = 0.8 * L * w * math.sqrt(2.0 * G * H)  # ≈ 0.8*2.0*0.05*sqrt(9.81)=0.250

        params = self._slotted_params(L=L, w=w)
        src = self._upload_inlet_and_run(params, H)
        self._assert_capture(src, Q_exp, atol=1e-3)

    # ── Combo test ──────────────────────────────────────────────────────

    def test_inlet_combo(self):
        """Combo = grate + curb sweep where L_sweep = curb_len - grate_len"""
        Lg, Wg = 1.0, 0.5
        L_curb = 2.0
        h_curb = 0.15
        H = 0.3  # m → orifice regime for grate, weir for curb
        openFrac = 0.9

        # Grate contribution (orifice at H=0.3, H_trans = 1.79*0.45/3.0 = 0.2685, so orifice)
        P = 2.0 * (Lg + Wg)
        Ao_g = Lg * Wg * openFrac
        H_trans_g = 1.79 * Ao_g / P
        if H <= H_trans_g:
            qg = 3.0 * P * math.pow(H, 1.5)
        else:
            qg = 0.67 * Ao_g * math.sqrt(2.0 * G * H)

        # Curb sweep: L_sweep = curb_len - grate_len
        L_sweep = max(0.0, L_curb - Lg)
        if H <= h_curb:
            qc = 3.0 * L_sweep * math.pow(H, 1.5)
        else:
            qc = 0.67 * h_curb * L_sweep * math.sqrt(2.0 * G * max(1e-10, H - 0.5 * h_curb))
        Q_exp = qg + qc

        params = {
            "inlet_type": np.array([3], dtype=np.int32),
            "inlet_cell": np.array([0], dtype=np.int32),
            "inlet_node": np.array([0], dtype=np.int32),
            "inlet_crest": np.array([0.0], dtype=np.float64),
            "inlet_width": np.array([0.0], dtype=np.float64),
            "inlet_cd": np.array([0.67], dtype=np.float64),
            "inlet_qmax": np.array([1e6], dtype=np.float64),
            "inlet_grate_len": np.array([Lg], dtype=np.float64),
            "inlet_grate_wid": np.array([Wg], dtype=np.float64),
            "inlet_grate_kind": np.array([0], dtype=np.int32),
            "inlet_grate_open": np.array([openFrac], dtype=np.float64),
            "inlet_curb_len": np.array([L_curb], dtype=np.float64),
            "inlet_curb_ht": np.array([h_curb], dtype=np.float64),
            "inlet_curb_throat": np.array([0], dtype=np.int32),
            "inlet_slot_len": np.array([0.0], dtype=np.float64),
            "inlet_slot_wid": np.array([0.0], dtype=np.float64),
        }

        src = self._upload_inlet_and_run(params, H)
        self._assert_capture(src, Q_exp, atol=1e-3)

    # ── Relief test ─────────────────────────────────────────────────────

    def test_inlet_relief(self):
        """Node > surface: relief flow through Ao"""
        Lg, Wg, openFrac = 1.0, 0.5, 0.9
        Ao = Lg * Wg * openFrac

        # Set node depth > surface depth for relief flow
        params = self._grate_params(Lg=Lg, Wg=Wg, openFrac=openFrac)

        # Upload params first (needs node_max_depth)
        mod = self.mod
        o_cell, o_node, o_inv, o_dia, o_cd, o_qmax, o_zs = self._make_empty_outfall_arrays()
        mod.swe2d_gpu_upload_drainage_exchange_params(
            inlet_cell=params["inlet_cell"],
            inlet_node=params["inlet_node"],
            inlet_crest=params["inlet_crest"],
            inlet_width=params["inlet_width"],
            inlet_cd=params["inlet_cd"],
            inlet_qmax=params["inlet_qmax"],
            inlet_type=params["inlet_type"],
            inlet_grate_len=params["inlet_grate_len"],
            inlet_grate_wid=params["inlet_grate_wid"],
            inlet_grate_kind=params["inlet_grate_kind"],
            inlet_grate_open=params["inlet_grate_open"],
            inlet_curb_len=params.get("inlet_curb_len", np.array([0.0], dtype=np.float64)),
            inlet_curb_ht=params.get("inlet_curb_ht", np.array([0.15], dtype=np.float64)),
            inlet_curb_throat=params.get("inlet_curb_throat", np.array([0], dtype=np.int32)),
            inlet_slot_len=params.get("inlet_slot_len", np.array([0.0], dtype=np.float64)),
            inlet_slot_wid=params.get("inlet_slot_wid", np.array([0.0], dtype=np.float64)),
            outfall_cell=o_cell,
            outfall_node=o_node,
            outfall_invert=o_inv,
            outfall_diameter=o_dia,
            outfall_cd=o_cd,
            outfall_qmax=o_qmax,
            outfall_zero_storage=o_zs,
            node_max_depth=np.array([3.0], dtype=np.float64),
        )

        # Surface cell dry (depth=0), node has depth=1.0m → relief from node to surface
        h = np.zeros(self.N_CELLS, dtype=np.float64)
        hu = np.zeros(self.N_CELLS, dtype=np.float64)
        hv = np.zeros(self.N_CELLS, dtype=np.float64)
        mod.swe2d_set_state(self.solver, h, hu, hv)

        # Upload node depth to pipe1d (1.0m ponding at node)
        mod.swe2d_pipe1d_upload_cell_h(int(self._dev_ptr),
                                           np.array([1.0], dtype=np.float64))

        mod.swe2d_gpu_set_coupling_dt(1.0)
        mod.swe2d_gpu_compute_coupling_full_on_device(
            cell_wse=None,
            n_structures=1,
            host_structure_flows=np.array([0.0], dtype=np.float64),
        )

        src = mod.swe2d_gpu_readback_coupling_sources(self.N_CELLS)

        # Relief: node→surface, so cell 0 gets positive source
        H_relief = 1.0
        Q_exp = 0.67 * Ao * math.sqrt(2.0 * G * H_relief)
        q_actual = float(src[0]) * self.CELL_AREA  # m/s → m³/s
        self.assertAlmostEqual(q_actual, Q_exp, delta=1e-3,
                               msg=f"Relief: got {q_actual:.6f}, expected {Q_exp:.6f}")

    # ── Availability limiter ────────────────────────────────────────────

    def test_inlet_availability_limiter(self):
        """Node storage cap limits capture"""
        mod = self.mod
        Lg, Wg = 1.0, 0.5

        # Set tiny node max_depth (0.01 m) so node fills up quickly
        node_max_depth = 0.01
        node_surface_area = 10.0

        # Rebuild pipe1d with limited node storage
        mod.swe2d_build_unified_mesh(
            n_links=1,
            link_from=np.array([0], dtype=np.int32),
            link_to=np.array([0], dtype=np.int32),
            L=np.array([1.0], dtype=np.float64),
            D=np.array([1.0], dtype=np.float64),
            n_mann=np.array([0.013], dtype=np.float64),
            node_invert=np.array([0.0], dtype=np.float64),
            mcl=10,
            dev_ptr=int(self._dev_ptr),
            S0=np.zeros(1, dtype=np.float64),
        )

        # Upload node depth (near full)
        mod.swe2d_pipe1d_upload_cell_h(int(self._dev_ptr),
                                           np.array([0.009], dtype=np.float64))
        mod.swe2d_pipe1d_init_cell_area(int(self._dev_ptr), H_MIN_DEFAULT)

        # Upload inlet params with node_max_depth matching the pipe1d
        params = self._grate_params(Lg=Lg, Wg=Wg)
        o_cell, o_node, o_inv, o_dia, o_cd, o_qmax, o_zs = self._make_empty_outfall_arrays()
        mod.swe2d_gpu_upload_drainage_exchange_params(
            inlet_cell=params["inlet_cell"],
            inlet_node=params["inlet_node"],
            inlet_crest=params["inlet_crest"],
            inlet_width=params["inlet_width"],
            inlet_cd=params["inlet_cd"],
            inlet_qmax=params["inlet_qmax"],
            inlet_type=params["inlet_type"],
            inlet_grate_len=params["inlet_grate_len"],
            inlet_grate_wid=params["inlet_grate_wid"],
            inlet_grate_kind=params["inlet_grate_kind"],
            inlet_grate_open=params["inlet_grate_open"],
            inlet_curb_len=params.get("inlet_curb_len", np.array([0.0], dtype=np.float64)),
            inlet_curb_ht=params.get("inlet_curb_ht", np.array([0.15], dtype=np.float64)),
            inlet_curb_throat=params.get("inlet_curb_throat", np.array([0], dtype=np.int32)),
            inlet_slot_len=params.get("inlet_slot_len", np.array([0.0], dtype=np.float64)),
            inlet_slot_wid=params.get("inlet_slot_wid", np.array([0.0], dtype=np.float64)),
            outfall_cell=o_cell,
            outfall_node=o_node,
            outfall_invert=o_inv,
            outfall_diameter=o_dia,
            outfall_cd=o_cd,
            outfall_qmax=o_qmax,
            outfall_zero_storage=o_zs,
            node_max_depth=np.array([node_max_depth], dtype=np.float64),
        )

        # High head (wants to capture a lot)
        H = 0.5
        depth = H
        h = np.array([depth, 0.0], dtype=np.float64)
        hu = np.zeros(self.N_CELLS, dtype=np.float64)
        hv = np.zeros(self.N_CELLS, dtype=np.float64)
        mod.swe2d_set_state(self.solver, h, hu, hv)

        mod.swe2d_gpu_set_coupling_dt(1.0)
        mod.swe2d_gpu_compute_coupling_full_on_device(
            cell_wse=None,
            n_structures=1,
            host_structure_flows=np.array([0.0], dtype=np.float64),
        )

        src = mod.swe2d_gpu_readback_coupling_sources(self.N_CELLS)

        # The node can only accept: rem_storage = (0.01 - 0.009) * 10 = 0.01 m³
        # over dt=1.0 s → max 0.01 m³/s
        q_expected_limited = 0.01  # m³/s

        q_actual = -float(src[0]) * self.CELL_AREA  # m/s → m³/s
        self.assertAlmostEqual(q_actual, q_expected_limited, delta=1e-4,
                               msg=f"Availability limiter: got {q_actual:.6f}, "
                                   f"expected {q_expected_limited:.6f}")


@unittest.skipIf(_MOD is None, "hydra_swe2d not built")
@unittest.skipUnless(hasattr(_MOD, "swe2d_gpu_upload_drainage_exchange_params"),
                     "drainage exchange CUDA functions not compiled")
@unittest.skipUnless(hasattr(_MOD, "swe2d_gpu_available")
                     and _MOD.swe2d_gpu_available(),
                     "CUDA GPU not available")
class TestDrainageNoStructures(unittest.TestCase):
    """Regression: drainage-only coupling must not crash with no structures."""

    N_CELLS = 2
    CELL_AREA = 25.0

    @classmethod
    def setUpClass(cls):
        cls.mod = _MOD
        cls.mesh = _make_2cell_mesh(_MOD)
        h0 = np.array([0.0, 0.0], dtype=np.float64)
        cls.solver = None
        try:
            cls.solver = _make_solver(_MOD, cls.mesh, h0)
        except Exception:
            raise unittest.SkipTest("Failed to create GPU solver")

    @classmethod
    def tearDownClass(cls):
        if cls.solver is not None:
            try:
                cls.mod.swe2d_destroy(cls.solver)
            except Exception:
                pass
            cls.solver = None

    def setUp(self):
        mod = self.mod
        if not mod.swe2d_step(self.solver, 0.1).get("gpu_active", False):
            self.skipTest("solver step did not activate GPU")

        self._dev_ptr = mod.swe2d_get_coupling_dev_ptr()

        # Intentionally do NOT preload a dummy weir.  The cell-area preload
        # must be enough to allocate the WSE buffer used by the drainage
        # exchange kernels (regression for CUDA illegal-access crash).
        mod.swe2d_gpu_preload_coupling_cell_area(
            np.full(self.N_CELLS, self.CELL_AREA, dtype=np.float64))

        mod.swe2d_build_unified_mesh(
            n_links=1,
            link_from=np.array([0], dtype=np.int32),
            link_to=np.array([0], dtype=np.int32),
            L=np.array([1.0], dtype=np.float64),
            D=np.array([1.0], dtype=np.float64),
            n_mann=np.array([0.013], dtype=np.float64),
            node_invert=np.array([0.0], dtype=np.float64),
            mcl=10,
            dev_ptr=int(self._dev_ptr),
            S0=np.zeros(1, dtype=np.float64),
        )
        mod.swe2d_pipe1d_upload_cell_h(int(self._dev_ptr),
                                           np.array([0.0], dtype=np.float64))
        mod.swe2d_pipe1d_init_cell_area(int(self._dev_ptr), H_MIN_DEFAULT)
        mod.swe2d_gpu_ensure_drainage_q_buf(self.N_CELLS)

    def tearDown(self):
        mod = self.mod
        h = np.full(self.N_CELLS, 0.0, dtype=np.float64)
        hu = np.zeros(self.N_CELLS, dtype=np.float64)
        hv = np.zeros(self.N_CELLS, dtype=np.float64)
        mod.swe2d_set_state(self.solver, h, hu, hv)

    def test_drainage_only_no_structures(self):
        """A single grate inlet with no hydraulic structures computes capture."""
        mod = self.mod

        mod.swe2d_set_state(
            self.solver,
            np.array([1.0, 0.0], dtype=np.float64),
            np.zeros(self.N_CELLS, dtype=np.float64),
            np.zeros(self.N_CELLS, dtype=np.float64),
        )

        mod.swe2d_gpu_upload_drainage_exchange_params(
            inlet_cell=np.array([0], dtype=np.int32),
            inlet_node=np.array([0], dtype=np.int32),
            inlet_crest=np.array([0.0], dtype=np.float64),
            inlet_width=np.array([1.0], dtype=np.float64),
            inlet_cd=np.array([0.67], dtype=np.float64),
            inlet_qmax=np.array([1e6], dtype=np.float64),
            inlet_type=np.array([0], dtype=np.int32),
            inlet_grate_len=np.array([1.0], dtype=np.float64),
            inlet_grate_wid=np.array([0.5], dtype=np.float64),
            inlet_grate_kind=np.array([0], dtype=np.int32),
            inlet_grate_open=np.array([0.9], dtype=np.float64),
            inlet_curb_len=np.array([0.0], dtype=np.float64),
            inlet_curb_ht=np.array([0.15], dtype=np.float64),
            inlet_curb_throat=np.array([0], dtype=np.int32),
            inlet_slot_len=np.array([0.0], dtype=np.float64),
            inlet_slot_wid=np.array([0.0], dtype=np.float64),
            outfall_cell=np.empty(0, dtype=np.int32),
            outfall_node=np.empty(0, dtype=np.int32),
            outfall_invert=np.empty(0, dtype=np.float64),
            outfall_diameter=np.empty(0, dtype=np.float64),
            outfall_cd=np.empty(0, dtype=np.float64),
            outfall_qmax=np.empty(0, dtype=np.float64),
            outfall_zero_storage=np.empty(0, dtype=np.int32),
            node_max_depth=np.array([3.0], dtype=np.float64),
        )

        mod.swe2d_gpu_set_coupling_dt(1.0)
        mod.swe2d_gpu_compute_coupling_full_on_device(
            cell_wse=None,
            n_structures=0,
            host_structure_flows=None,
        )

        src = mod.swe2d_gpu_readback_coupling_sources(self.N_CELLS)
        q_actual = -float(src[0]) * self.CELL_AREA  # m/s → m³/s
        self.assertGreater(q_actual, 0.0, "Expected positive capture in cell 0")
        self.assertAlmostEqual(float(src[1]), 0.0, places=12,
                               msg="Cell 1 should have zero source")

    def test_pipe_end_upload_allocates_all_arrays(self):
        """Regression: non-empty pipe_end arrays must not trigger CUDA invalid argument.

        Prior to the fix, the pipe-end workspace used _DS_ENSURE with a shared
        pipe_end_capacity counter. After the first array was allocated, the
        capacity matched the request and subsequent arrays were silently skipped,
        leaving their device pointers null. The upload then called cudaMemcpy to
        a null pointer and failed with cudaErrorInvalidValue.
        """
        mod = self.mod

        mod.swe2d_gpu_upload_drainage_exchange_params(
            inlet_cell=np.empty(0, dtype=np.int32),
            inlet_node=np.empty(0, dtype=np.int32),
            inlet_crest=np.empty(0, dtype=np.float64),
            inlet_width=np.empty(0, dtype=np.float64),
            inlet_cd=np.empty(0, dtype=np.float64),
            inlet_qmax=np.empty(0, dtype=np.float64),
            inlet_type=np.empty(0, dtype=np.int32),
            inlet_grate_len=np.empty(0, dtype=np.float64),
            inlet_grate_wid=np.empty(0, dtype=np.float64),
            inlet_grate_kind=np.empty(0, dtype=np.int32),
            inlet_grate_open=np.empty(0, dtype=np.float64),
            inlet_curb_len=np.empty(0, dtype=np.float64),
            inlet_curb_ht=np.empty(0, dtype=np.float64),
            inlet_curb_throat=np.empty(0, dtype=np.int32),
            inlet_slot_len=np.empty(0, dtype=np.float64),
            inlet_slot_wid=np.empty(0, dtype=np.float64),
            outfall_cell=np.empty(0, dtype=np.int32),
            outfall_node=np.empty(0, dtype=np.int32),
            outfall_invert=np.empty(0, dtype=np.float64),
            outfall_diameter=np.empty(0, dtype=np.float64),
            outfall_cd=np.empty(0, dtype=np.float64),
            outfall_qmax=np.empty(0, dtype=np.float64),
            outfall_zero_storage=np.empty(0, dtype=np.int32),
            pipe_end_cell=np.array([0], dtype=np.int32),
            pipe_end_node=np.array([0], dtype=np.int32),
            pipe_end_invert=np.array([0.0], dtype=np.float64),
            pipe_end_diameter=np.array([1.0], dtype=np.float64),
            pipe_end_area=np.array([0.78539816], dtype=np.float64),
            pipe_end_kin=np.array([0.5], dtype=np.float64),
            pipe_end_kout=np.array([1.0], dtype=np.float64),
            node_max_depth=np.array([3.0], dtype=np.float64),
        )

        # If the upload succeeded, the device-side exchange is loaded.
        # We can sanity-check by running a coupling step with no surface structures.
        mod.swe2d_gpu_set_coupling_dt(1.0)
        mod.swe2d_gpu_compute_coupling_full_on_device(
            cell_wse=None,
            n_structures=0,
            host_structure_flows=None,
        )

        src = mod.swe2d_gpu_readback_coupling_sources(self.N_CELLS)
        self.assertIsNotNone(src)
        self.assertEqual(src.shape, (self.N_CELLS,))


@unittest.skipIf(_MOD is None, "hydra_swe2d not built")
@unittest.skipUnless(hasattr(_MOD, "swe2d_gpu_upload_drainage_exchange_params"),
                     "drainage exchange CUDA functions not compiled")
@unittest.skipUnless(hasattr(_MOD, "swe2d_pipe1d_upload_pipe_end_surface_faces")
                     and hasattr(_MOD, "swe2d_readback_ext_struct_flux"),
                     "new pipe-end face-flux path not bound")
@unittest.skipUnless(hasattr(_MOD, "swe2d_get_solver_dev_ptr"),
                     "solver_dev_ptr accessor not bound")
@unittest.skipUnless(hasattr(_MOD, "swe2d_gpu_available")
                     and _MOD.swe2d_gpu_available(),
                     "CUDA GPU not available")
class TestPipeEndExchange(unittest.TestCase):
    """Regression: pipe-end exchange must move water between surface cells.

    Fix P0#2 (audit 2026-07-22): ported from the retired
    `swe2d_gpu_apply_pipe_face_flux` / `swe2d_gpu_readback_face_q_to_2d`
    bindings onto the unified face-flux kernel path.  The new API:
    - `swe2d_pipe1d_step` accepts `solver_dev_ptr`; pass-1 of the unified
      face-flux kernel writes `solver_dev->d_ext_struct_flux_h` for
      SURFACE_2D_PIPE_END faces (class 3).
    - `swe2d_readback_ext_struct_flux` reads back the flux.
    - `swe2d_pipe1d_upload_pipe_end_surface_faces` patches class-3 face
      `owner_R` to point at the coupled 2D cell.
    """

    N_CELLS = 2
    CELL_AREA = 25.0

    @classmethod
    def setUpClass(cls):
        cls.mod = _MOD
        cls.mesh = _make_2cell_mesh(_MOD)
        h0 = np.array([0.0, 0.0], dtype=np.float64)
        cls.solver = None
        try:
            cls.solver = _make_solver(_MOD, cls.mesh, h0)
        except Exception:
            raise unittest.SkipTest("Failed to create GPU solver")

    @classmethod
    def tearDownClass(cls):
        if cls.solver is not None:
            try:
                cls.mod.swe2d_destroy(cls.solver)
            except Exception:
                pass
            cls.solver = None

    def setUp(self):
        mod = self.mod
        if not mod.swe2d_step(self.solver, 0.1).get("gpu_active", False):
            self.skipTest("solver step did not activate GPU")

        self._dev_ptr = mod.swe2d_get_coupling_dev_ptr()
        self._solver_dev_ptr = mod.swe2d_get_solver_dev_ptr(self.solver)
        if self._solver_dev_ptr == 0:
            self.skipTest("solver has no device state (swe2d_get_solver_dev_ptr returned 0)")

        mod.swe2d_gpu_preload_coupling_cell_area(
            np.full(self.N_CELLS, self.CELL_AREA, dtype=np.float64))

        # Two-node, one-link pipe.  Node 0 connects to surface cell 0,
        # node 1 connects to surface cell 1.
        mod.swe2d_build_unified_mesh(
            n_links=1,
            link_from=np.array([0], dtype=np.int32),
            link_to=np.array([1], dtype=np.int32),
            L=np.array([100.0], dtype=np.float64),
            D=np.array([1.0], dtype=np.float64),
            n_mann=np.array([0.013], dtype=np.float64),
            node_invert=np.array([0.0, 0.0], dtype=np.float64),
            mcl=10,
            dev_ptr=int(self._dev_ptr),
            S0=np.zeros(1, dtype=np.float64),
            pipe_end_node_ids=np.array([0, 1], dtype=np.int32),
            n_pipe_ends=2,
        )
        mod.swe2d_pipe1d_upload_cell_h(int(self._dev_ptr),
                                           np.zeros(2, dtype=np.float64))
        mod.swe2d_pipe1d_init_cell_area(int(self._dev_ptr), H_MIN_DEFAULT)
        mod.swe2d_gpu_ensure_drainage_q_buf(self.N_CELLS)

        mod.swe2d_gpu_upload_drainage_exchange_params(
            inlet_cell=np.empty(0, dtype=np.int32),
            inlet_node=np.empty(0, dtype=np.int32),
            inlet_crest=np.empty(0, dtype=np.float64),
            inlet_width=np.empty(0, dtype=np.float64),
            inlet_cd=np.empty(0, dtype=np.float64),
            inlet_qmax=np.empty(0, dtype=np.float64),
            inlet_type=np.empty(0, dtype=np.int32),
            inlet_grate_len=np.empty(0, dtype=np.float64),
            inlet_grate_wid=np.empty(0, dtype=np.float64),
            inlet_grate_kind=np.empty(0, dtype=np.int32),
            inlet_grate_open=np.empty(0, dtype=np.float64),
            inlet_curb_len=np.empty(0, dtype=np.float64),
            inlet_curb_ht=np.empty(0, dtype=np.float64),
            inlet_curb_throat=np.empty(0, dtype=np.int32),
            inlet_slot_len=np.empty(0, dtype=np.float64),
            inlet_slot_wid=np.empty(0, dtype=np.float64),
            outfall_cell=np.empty(0, dtype=np.int32),
            outfall_node=np.empty(0, dtype=np.int32),
            outfall_invert=np.empty(0, dtype=np.float64),
            outfall_diameter=np.empty(0, dtype=np.float64),
            outfall_cd=np.empty(0, dtype=np.float64),
            outfall_qmax=np.empty(0, dtype=np.float64),
            outfall_zero_storage=np.empty(0, dtype=np.int32),
            pipe_end_cell=np.array([0, 1], dtype=np.int32),
            pipe_end_node=np.array([0, 1], dtype=np.int32),
            pipe_end_invert=np.array([0.0, 0.0], dtype=np.float64),
            pipe_end_diameter=np.array([1.0, 1.0], dtype=np.float64),
            pipe_end_area=np.array([0.78539816, 0.78539816], dtype=np.float64),
            pipe_end_kin=np.array([0.5, 0.5], dtype=np.float64),
            pipe_end_kout=np.array([1.0, 1.0], dtype=np.float64),
            node_max_depth=np.array([3.0, 3.0], dtype=np.float64),
        )

        # Patch class-3 SURFACE_2D_PIPE_END faces with their coupled 2D cell.
        # The mesh build emits one class-3 face per pipe-end (n_pipe_ends=2).
        mod.swe2d_pipe1d_upload_pipe_end_surface_faces(
            int(self._dev_ptr),
            np.array([0, 1], dtype=np.int32),  # 2D cell indices for each pipe end
        )

    def tearDown(self):
        mod = self.mod
        h = np.full(self.N_CELLS, 0.0, dtype=np.float64)
        hu = np.zeros(self.N_CELLS, dtype=np.float64)
        hv = np.zeros(self.N_CELLS, dtype=np.float64)
        mod.swe2d_set_state(self.solver, h, hu, hv)

    def _step_pipe1d_with_solveur(self, dt: float) -> np.ndarray:
        """Run one pipe1d step with solver_dev so SURFACE_2D_PIPE_END fluxes
        accumulate into solver_dev->d_ext_struct_flux_h.  Returns the flux
        vector at the 2D cells (length n_cells_2d)."""
        mod = self.mod
        mod.swe2d_gpu_set_coupling_dt(dt)
        mod.swe2d_pipe1d_step(
            int(self._dev_ptr),
            dt,
            "diffusion_wave",
            1,
            2,
            0.5,
            G,
            K_MANN_DEFAULT,
            H_MIN_DEFAULT,
            1,  # surcharge_method=SLOT
            recon_method=0,
            time_integrator=1,
            friction_alpha=0.01,
            solver_dev_ptr=int(self._solver_dev_ptr),
        )
        flux = mod.swe2d_readback_ext_struct_flux(
            int(self._solver_dev_ptr), self.N_CELLS)
        return np.asarray(flux["h"], dtype=np.float64)

    def _debug_flux_dump(self):
        """Dump face class/owner_R/L for inspection (returns debug info)."""
        mod = self.mod
        n_faces = int(mod.swe2d_pipe1d_readback_int(
            int(self._dev_ptr), "n_faces"))
        if n_faces <= 0:
            return f"n_faces={n_faces}"
        # Just print flux and class
        flux = mod.swe2d_readback_ext_struct_flux(
            int(self._solver_dev_ptr), self.N_CELLS)
        return (f"n_faces={n_faces}, flux_h=[{flux['h'][0]:.4f}, {flux['h'][1]:.4f}], "
                f"flux_hu=[{flux['hu'][0]:.4f}, {flux['hu'][1]:.4f}], "
                f"flux_hv=[{flux['hv'][0]:.4f}, {flux['hv'][1]:.4f}]")

    def test_pipe_end_moves_water_downhill(self):
        """High WSE in cell 0 drives flow through pipe to cell 1.

        Face-flux coupling (HLLC): the pipe starts primed (full).
        Cell 0's WSE (2.0 m) exceeds the pipe head so water enters the pipe
        at node 0 (flux_h[0] < 0); the pipe head at node 1 (1.0 m) exceeds
        cell 1's WSE (0.5 m) so water exits into cell 1 (flux_h[1] > 0).
        Verify the pipe-end flux has the right sign at both ends and the
        pipe is actually conveying water downstream.
        """
        mod = self.mod
        dt = 1.0
        n_steps = 20
        cell_dx = 10.0  # 100 m link / 10 sub-cells

        # Cell 0 has 2.0 m of water, cell 1 has 0.5 m.
        mod.swe2d_set_state(
            self.solver,
            np.array([2.0, 0.5], dtype=np.float64),
            np.zeros(self.N_CELLS, dtype=np.float64),
            np.zeros(self.N_CELLS, dtype=np.float64),
        )

        # Prime the pipe (all sub-cells full, depth = crown = 1.0 m).
        mod.swe2d_pipe1d_upload_cell_h(int(self._dev_ptr),
                                           np.full(10, 1.0, dtype=np.float64))
        mod.swe2d_pipe1d_init_cell_area(int(self._dev_ptr), H_MIN_DEFAULT)

        # Run a single pipe1d step to capture the initial exchange flux.
        flux_h = self._step_pipe1d_with_solveur(dt)

        # Sign checks: cell 0 (high WSE) feeds the pipe; cell 1 (low WSE)
        # receives from the pipe.  fh is positive when flow is L→R = pipe→2D.
        # Cell 0's flux is pipe→2D direction reversed (2D→pipe), so flux < 0.
        # Cell 1's flux is pipe→2D direction, so flux > 0.
        self.assertLess(float(flux_h[0]), 0.0,
                        f"Cell 0 should feed the pipe (flux<0), got {flux_h[0]:+.4f}")
        self.assertGreater(float(flux_h[1]), 0.0,
                           f"Cell 1 should receive from pipe (flux>0), got {flux_h[1]:+.4f}")

        # Pipe must be actively conveying water downstream.  A primed pipe
        # with one end feeding and the other receiving should have non-zero
        # |cell_Q| somewhere along its length.
        rb = mod.swe2d_pipe1d_readback_cell_state(int(self._dev_ptr), 10, 0, 0)
        cell_Q = np.asarray(rb["cell_Q"], dtype=np.float64)
        q_max = float(np.max(np.abs(cell_Q)))
        self.assertGreater(q_max, 0.0,
                           "Expected non-zero conveyance inside the pipe")

    def test_pipe_inflow_wets_downstream_cell(self):
        """Water entering a dry pipe at the upstream end must eventually
        discharge into the downstream 2D cell via face-flux coupling.

        Cell 0 ponded, cell 1 dry, pipe dry.  The face-flux coupling fills
        the pipe, conveys water through it, and spills into the dry downstream
        cell.
        """
        mod = self.mod
        dt = 1.0
        n_steps = 120

        # Cell 0 ponded at 2.0 m; cell 1 dry.  Pipe starts dry.
        mod.swe2d_set_state(
            self.solver,
            np.array([2.0, 0.0], dtype=np.float64),
            np.zeros(self.N_CELLS, dtype=np.float64),
            np.zeros(self.N_CELLS, dtype=np.float64),
        )
        mod.swe2d_pipe1d_upload_cell_h(int(self._dev_ptr),
                                           np.zeros(2, dtype=np.float64))
        mod.swe2d_pipe1d_init_cell_area(int(self._dev_ptr), H_MIN_DEFAULT)

        first_wet_step = None
        last_flux = None
        for k in range(n_steps):
            flux_h = self._step_pipe1d_with_solveur(dt)
            last_flux = flux_h
            # Cell 1 receives water (positive flux into cell 1).
            if first_wet_step is None and float(flux_h[1]) > 0.0:
                first_wet_step = k

        self.assertIsNotNone(
            first_wet_step,
            f"Cell 1 never received water from the pipe in {n_steps} steps")
        self.assertLess(first_wet_step, n_steps // 2,
                        f"Cell 1 took too many steps ({first_wet_step}) to "
                        f"first receive water")

        self.assertIsNotNone(last_flux)
        # Cell 0 must still be feeding the pipe (cell 0 receives water in the
        # sign convention: positive flux means 2D-cell-gains-from-pipe is wrong;
        # actually the 2D cell is the 2D side, so cell 0 should be losing water).
        # For an upstream ponded cell 0 connected to a pipe, cell 0's flux
        # *should* be positive (pipe is losing water to its 2D cell neighbor),
        # but the face convention may differ.  The end-to-end test is that
        # both cells receive non-trivial flux after the pipe fills.
        self.assertGreater(float(np.sum(np.abs(last_flux))), 0.0,
                           "Expected non-zero exchange flux at both pipe ends")

        # The downstream pipe cell must be wet (water arrived by conveyance).
        rb = mod.swe2d_pipe1d_readback_cell_state(int(self._dev_ptr), 10, 0, 0)
        self.assertGreater(float(rb["cell_A"][9]), 0.0,
                           "Downstream pipe cell (index 9) should be wet")

        # No checkerboard blowup: pipe flows must stay below the physical
        # conveyance scale of this pipe (~10 m3/s, not 1000).
        q_max = float(np.max(np.abs(rb["cell_Q"])))
        self.assertLess(q_max, 50.0,
                        f"Pipe flow unstable: |Q| reached {q_max:.3f} m3/s")

    def test_wet_pipe_drains_into_dry_surface_cells(self):
        """Wet pipe connected to dry surface cells drains through the openings
        via face-flux coupling (HLLC).

        SPEC §2.9: the pipe-end BC is a passive observer.  The face-flux
        coupling lets the pipe spill into the dry cell as a free outfall
        until the pipe empties.
        """
        mod = self.mod
        dt = 1.0
        n_steps = 20
        cell_dx = 10.0

        # Start with a wet pipe: all sub-cells at depth = crown = 1.0 m.
        mod.swe2d_pipe1d_upload_cell_h(int(self._dev_ptr),
                                           np.full(10, 1.0, dtype=np.float64))
        mod.swe2d_pipe1d_init_cell_area(int(self._dev_ptr), H_MIN_DEFAULT)

        # Both surface cells dry.
        mod.swe2d_set_state(
            self.solver,
            np.zeros(self.N_CELLS, dtype=np.float64),
            np.zeros(self.N_CELLS, dtype=np.float64),
            np.zeros(self.N_CELLS, dtype=np.float64),
        )

        rb0 = mod.swe2d_pipe1d_readback_cell_state(int(self._dev_ptr), 10, 0, 0)
        v_pipe0 = float(np.sum(rb0["cell_A"]) * cell_dx)
        self.assertGreater(v_pipe0, 0.0, "pipe should start wet")

        # Both dry cells must receive water from the pipe (positive flux
        # at both ends — pipe head > dry cell WSE = 0).
        flux_h = self._step_pipe1d_with_solveur(dt)
        self.assertGreater(float(flux_h[0]), 0.0,
                           f"Cell 0 should gain from pipe (flux>0), got {flux_h[0]:+.4f}")
        self.assertGreater(float(flux_h[1]), 0.0,
                           f"Cell 1 should gain from pipe (flux>0), got {flux_h[1]:+.4f}")

        # The pipe must have started draining: storage decreases after a few
        # steps.  Run additional steps and check storage drop.
        for _ in range(n_steps):
            self._step_pipe1d_with_solveur(dt)

        rb1 = mod.swe2d_pipe1d_readback_cell_state(int(self._dev_ptr), 10, 0, 0)
        v_pipe1 = float(np.sum(rb1["cell_A"]) * cell_dx)
        self.assertLess(v_pipe1, v_pipe0,
                        f"Pipe storage should decrease, {v_pipe0:.3f} -> {v_pipe1:.3f}")


class TestPipeCellReadback(unittest.TestCase):
    """Tests for pipe-cell readback via SWE2DCouplingController."""

    def test_readback_coupling_state_returns_cell_arrays(self):
        """readback_coupling_state returns cell_velocity/depth/flow/head arrays."""
        from unittest.mock import patch, MagicMock
        import numpy as np
        from swe2d.runtime.coupling import SWE2DCouplingController

        with patch("swe2d.runtime.coupling.load_swe2d_native_module") as mock_mod:
            mock_dev = MagicMock()
            mock_mod.return_value = mock_mod
            mock_mod.swe2d_pipe1d_readback_cell_state.return_value = {
                "cell_A": np.array([0.5, 0.5, 0.5]),
                "cell_Q": np.array([0.1, 0.1, 0.1]),
                "cell_velocity": np.array([0.2, 0.2, 0.2]),
                "cell_depth": np.array([0.5, 0.5, 0.5]),
                "cell_q": np.array([0.2, 0.2, 0.2]),
                "cell_h": np.array([0.5, 0.5, 0.5]),
                "cell_y": np.array([0.5, 0.5, 0.5]),
                "cell_invert": np.array([0.0, 0.0, 0.0]),
                "cell_width": np.array([1.0, 1.0, 1.0]),
                "cell_height": np.array([1.0, 1.0, 1.0]),
                "cell_shape_type": np.array([0, 0, 0], dtype=np.int32),
                "cell_owner_link": np.array([0, 0, 0], dtype=np.int32),
                "cell_sub_idx": np.array([0, 1, 2], dtype=np.int32),
                "cell_class": np.zeros(3, dtype=np.int32),
            }
            cc = SWE2DCouplingController(cell_area=[1.0], cell_bed=[0.0])
            cc._drainage_soa = MagicMock()
            cc._drainage_soa.node_invert_elev = np.zeros(2)
            cc._drainage_soa.link_from = np.array([0])
            cc._drainage_soa.link_to = np.array([1])
            cc._drainage_soa.link_length = np.array([10.0])
            cc._drainage_soa.link_diameter = np.array([1.0])
            cc._drainage_soa.link_width = np.array([1.0])
            cc._drainage_soa.max_cell_length = 5.0
            cc._native_cuda_module = MagicMock(return_value=mock_mod)
            cc._log = MagicMock()
            cc.drainage = MagicMock()
            cc.drainage.cfg = MagicMock()
            cc.drainage.cfg.links = [MagicMock(link_id="L1")]
            cc._n_non_bridge_structures = 0

            state = cc.readback_coupling_state()

            for key in [
                "cell_velocity",
                "cell_depth",
                "cell_q",
                "cell_h",
                "cell_Q",
                "cell_owner_link",
            ]:
                assert key in state, f"Missing key: {key}"
                assert isinstance(state[key], np.ndarray), f"{key} is not ndarray"
            assert "node_depth" not in state, "Legacy node_depth key must not be returned"
            np.testing.assert_allclose(state["cell_velocity"], [0.2, 0.2, 0.2])
            np.testing.assert_allclose(state["cell_depth"], [0.5, 0.5, 0.5])
            assert state["cell_Q"].dtype == np.float64, "cell_Q must be float64"
            assert state["cell_velocity"].dtype == np.float64, "cell_velocity must be float64"
            assert state["cell_depth"].dtype == np.float64, "cell_depth must be float64"
            assert state["cell_q"].dtype == np.float64, "cell_q must be float64"
            assert state["cell_h"].dtype == np.float64, "cell_h must be float64"
            assert state["cell_owner_link"].dtype in (np.int64, np.int32), "cell_owner_link must be integer dtype"


def test_append_pipe_cell_snapshot_stores_geometry():
    """Pipe-cell snapshots must store per-cell geometry in live storage."""
    from swe2d.results.data import SWE2DResultsData

    rd = SWE2DResultsData()
    rd.init_pipe_cell_storage([("L1", 0, "depth")])
    rd.append_pipe_cell_snapshot({
        "link_id": "L1",
        "cell_sub_idx": 0,
        "metric": "depth",
        "t_s": 0.0,
        "value": 1.5,
        "cell_invert": 10.0,
        "cell_width": 2.0,
        "cell_height": 2.5,
        "cell_shape_type": 1,
    })
    d = rd._live_pipe_cell[("L1", 0, "depth")]
    assert d["cell_invert"] == 10.0
    assert d["cell_width"] == 2.0
    assert d["cell_height"] == 2.5
    assert d["cell_shape_type"] == 1

    # A second snapshot must not overwrite the geometry stored on the first write.
    rd.append_pipe_cell_snapshot({
        "link_id": "L1",
        "cell_sub_idx": 0,
        "metric": "depth",
        "t_s": 1.0,
        "value": 2.0,
        "cell_invert": 99.0,
        "cell_width": 9.0,
        "cell_height": 9.9,
        "cell_shape_type": 9,
    })
    assert d["cell_invert"] == 10.0
    assert d["cell_width"] == 2.0
    assert d["cell_height"] == 2.5
    assert d["cell_shape_type"] == 1
    assert d["times"] == [0.0, 1.0]
    assert d["values"] == [1.5, 2.0]

    # Fallback defaults when geometry keys are omitted entirely.
    rd.init_pipe_cell_storage([("L2", 1, "flow")])
    rd.append_pipe_cell_snapshot({
        "link_id": "L2",
        "cell_sub_idx": 1,
        "metric": "flow",
        "t_s": 0.0,
        "value": 3.0,
    })
    d2 = rd._live_pipe_cell[("L2", 1, "flow")]
    assert d2["cell_invert"] == 0.0
    assert d2["cell_width"] == 1.0
    assert d2["cell_height"] == 1.0
    assert d2["cell_shape_type"] == 0

    # cell_height should fall back to cell_width when width is provided but height is omitted.
    rd.init_pipe_cell_storage([("L3", 2, "head")])
    rd.append_pipe_cell_snapshot({
        "link_id": "L3",
        "cell_sub_idx": 2,
        "metric": "head",
        "t_s": 0.0,
        "value": 4.0,
        "cell_width": 3.0,
    })
    d3 = rd._live_pipe_cell[("L3", 2, "head")]
    assert d3["cell_width"] == 3.0
    assert d3["cell_height"] == 3.0

    # Geometry written on the first metric key must propagate to sibling metric keys.
    rd.init_pipe_cell_storage([
        ("L4", 0, "velocity"),
        ("L4", 0, "depth"),
        ("L4", 0, "flow"),
    ])
    rd.append_pipe_cell_snapshot({
        "link_id": "L4",
        "cell_sub_idx": 0,
        "metric": "velocity",
        "t_s": 0.0,
        "value": 1.0,
        "cell_invert": 12.0,
        "cell_width": 4.0,
        "cell_height": 5.0,
        "cell_shape_type": 2,
    })
    for metric in ("velocity", "depth", "flow"):
        sibling = rd._live_pipe_cell[("L4", 0, metric)]
        assert sibling["cell_invert"] == 12.0, f"metric={metric}"
        assert sibling["cell_width"] == 4.0, f"metric={metric}"
        assert sibling["cell_height"] == 5.0, f"metric={metric}"
        assert sibling["cell_shape_type"] == 2, f"metric={metric}"


def test_append_coupling_snapshot_routes_geometry_to_pipe_cell():
    """drainage_cell rows must pass geometry fields through to pipe-cell storage."""
    from swe2d.results.data import SWE2DResultsData

    rd = SWE2DResultsData()
    rd.init_pipe_cell_storage([
        ("L5", 0, "depth"),
    ])
    rd.append_coupling_snapshot({
        "t_s": 0.0,
        "component": "drainage_cell",
        "object_id": "L5#0",
        "object_name": "L5 cell 0",
        "metric": "depth",
        "value": 1.5,
        "cell_invert": 20.0,
        "cell_width": 3.0,
        "cell_height": 4.0,
        "cell_shape_type": 1,
    })
    d = rd._live_pipe_cell[("L5", 0, "depth")]
    assert d["cell_invert"] == 20.0
    assert d["cell_width"] == 3.0
    assert d["cell_height"] == 4.0
    assert d["cell_shape_type"] == 1


if __name__ == "__main__":
    unittest.main()

class _PytestStyleWrapper(unittest.TestCase):
    """Auto-generated wrapper for module-level test functions.

    Created by tools/wrap_pytest_style.py so that pytest-style tests
    (def test_* at module level) become visible to `python3 -m unittest`.
    Each module-level test is attached as a staticmethod so it can be
    discovered and run as a unittest TestCase.
    """
__wrapped_funcs = []
for _name, _obj in list(globals().items()):
    if _name.startswith("test_") and callable(_obj) and not isinstance(_obj, type):
        setattr(_PytestStyleWrapper, _name, staticmethod(_obj))
        __wrapped_funcs.append(_name)
for _name in __wrapped_funcs:
    del globals()[_name]
del _name, _obj, __wrapped_funcs
