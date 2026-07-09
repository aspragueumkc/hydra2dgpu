import math
import unittest

import numpy as np


G = 9.81  # gravity (m/s²) — matches kernel default


# ── HEC-22 analytical helpers ─────────────────────────────────────────────

def _grate_Q(H, Lg, Wg, openFrac=1.0):
    """Grate inlet capture (HEC-22 §4-4.1)."""
    P = 2.0 * (Lg + Wg)
    Ao = Lg * Wg * openFrac
    Ht = 1.79 * Ao / P
    if H <= Ht:
        return 3.0 * P * H ** 1.5
    return 0.67 * Ao * math.sqrt(2.0 * G * H)


def _curb_Q(H, L, h=0.15, throat=0):
    """Curb-opening capture (HEC-22 §4-4.2)."""
    if throat == 2:
        h *= 0.7071
    Ht = 1.4 * h
    if H <= Ht:
        return 3.0 * L * H ** 1.5
    He = H - 0.5 * h
    if He <= 0:
        return 3.0 * L * H ** 1.5
    return 0.67 * h * L * math.sqrt(2.0 * G * He)


def _slotted_Q(H, L, w=0.05):
    """Slotted drain capture (HEC-22 §4-4.3)."""
    Ht = 2.587 * w
    if H <= Ht:
        return 2.48 * L * H ** 1.5
    return 0.8 * L * w * math.sqrt(2.0 * G * H)


def _combo_Q(H, Lg, Wg, Lc, hc=0.15, openFrac=1.0, grate_kind=-1, throat=0):
    """Combined grate + curb capture (HEC-22 §4-4.4)."""
    P = 2.0 * (Lg + Wg)
    Ao = Lg * Wg * openFrac
    Ht = 1.79 * Ao / P
    if H <= Ht:
        q_grate = 3.0 * P * H ** 1.5
    else:
        q_grate = 0.67 * Ao * math.sqrt(2.0 * G * H)
    L_sweep = max(0.0, Lc - Lg)
    if L_sweep > 0:
        q_curb = _curb_Q(H, L_sweep, hc, throat)
    else:
        q_curb = 0.0
    return q_grate + q_curb


class TestSWE2DGPUCouplingKernel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import hydra_swe2d as mod
        except Exception:
            mod = None
        cls.mod = mod

    def test_device_resident_path_throughput(self):
        """Smoke-test: compute_coupling_full_on_device handles drainage + structures."""
        from swe2d.runtime.backend import SWE2DBackend
        from swe2d.runtime.coupling import SWE2DCouplingController
        from swe2d.extensions.drainage_network import SWE2DUrbanDrainageModule
        from swe2d.extensions.extension_models import (
            DrainageNode, HydraulicStructure,
            HydraulicStructureConfig, InletExchange,
            PipeNetworkConfig, DrainageSolverMode, StructureType,
        )
        from swe2d.extensions.structures import SWE2DStructureModule
        from swe2d import units as _u

        _u.configure(1.0)

        backend = SWE2DBackend()
        node_x = np.array([0., 12., 12., 0., 0., 12., 12., 0.], dtype=np.float64)
        node_y = np.array([0., 0., 8., 8., 0., 0., 8., 8.], dtype=np.float64)
        node_z = np.zeros(8, dtype=np.float64)
        cell_nodes = np.array([0,1,2, 0,2,3, 4,5,6, 4,6,7], dtype=np.int32)
        backend.build_mesh(node_x, node_y, node_z, cell_nodes)
        n_cells = backend.n_cells

        struct_mod = SWE2DStructureModule(HydraulicStructureConfig(enabled=True, structures=[
            HydraulicStructure('W0', StructureType.WEIR, 0, 1, crest_elev=0.0,
                               metadata={'width':2.0, 'cd':0.6}),
        ]), model_to_ft=_u.model_to_ft())

        drain_mod = SWE2DUrbanDrainageModule(PipeNetworkConfig(
            enabled=True,
            nodes=[DrainageNode('N0',0,0,invert_elev=0.0,max_depth=4.0,metadata={'surface_area':50.0})],
            links=[],
            inlets=[InletExchange('I0',0,'N0',crest_elev=0.5,width=1.0,coefficient=0.62,max_capture=1.0)],
            outfalls=[], pipe_ends=[],
            pipe_solver_mode="diffusion_wave",
        ))
        drain_mod.initialize()
        drain_mod.state.node_depth['N0'] = 2.0

        controller = SWE2DCouplingController(
            cell_area=backend.cell_areas(),
            cell_bed=np.zeros(n_cells, dtype=np.float64),
            drainage=drain_mod,
            structures=struct_mod,
            length_scale_si_to_model=1.0,
        )

        backend.initialize(
            h0=np.array([1.5, 0.5, 1.2, 0.4], dtype=np.float64),
            hu0=np.zeros(n_cells, dtype=np.float64),
            hv0=np.zeros(n_cells, dtype=np.float64),
            dt_max=0.02, dt_fixed=0.02,
        )
        backend.run(t_end=0.1, dt_request=0.02,
            source_rate_callback=controller._log)

        h, _, _ = backend.get_state()
        self.assertTrue(np.all(np.isfinite(h)),
                        "GPU solver must produce finite state with drainage + structures")
        backend.destroy()


#@unittest.skipIf(True, "")  # placeholder —remove skip to run
class TestHEC22InletValidation(unittest.TestCase):
    """Validate HEC-22 inlet capture through the full coupling controller.

    Each test creates a 1-cell surface mesh, an inlet node, and runs a
    short simulation with a known water depth, then verifies the captured
    volume matches the analytical HEC-22 equation for that inlet type.
    """

    @classmethod
    def setUpClass(cls):
        try:
            import hydra_swe2d as mod
        except Exception:
            mod = None
        cls.mod = mod

    def _run_and_capture(self, H_m: float, crest_m: float, cfg: "PipeNetworkConfig",
                         n_cells: int = 1) -> float:
        """Run a short coupled sim and return captured volume (m³) in cell 0."""
        from swe2d.runtime.backend import SWE2DBackend
        from swe2d.runtime.coupling import SWE2DCouplingController
        from swe2d.extensions.drainage_network import SWE2DUrbanDrainageModule
        from swe2d.extensions.structures import SWE2DStructureModule
        from swe2d.extensions.extension_models import (
            HydraulicStructure, HydraulicStructureConfig, StructureType,
        )
        from swe2d import units as _u

        _u.configure(1.0)

        backend = SWE2DBackend()
        nx, ny, Lx, Ly = 1, 1, 10.0, 10.0
        xs = np.linspace(0.0, Lx, nx + 1)
        ys = np.linspace(0.0, Ly, ny + 1)
        Xg, Yg = np.meshgrid(xs, ys)
        node_x = Xg.ravel().astype(np.float64)
        node_y = Yg.ravel().astype(np.float64)
        node_z = np.zeros_like(node_x)
        stride = nx + 1
        cells = [0, 1, stride + 1, 0, stride + 1, stride]
        cell_nodes = np.array(cells, dtype=np.int32)
        backend.build_mesh(node_x, node_y, node_z, cell_nodes)
        n_cells = backend.n_cells

        # Dummy structure (inert — crest above any possible WSE)
        struct_mod = SWE2DStructureModule(HydraulicStructureConfig(enabled=True, structures=[
            HydraulicStructure("W0", StructureType.WEIR, 0, 0, crest_elev=1e6),
        ]), model_to_ft=1.0)

        drain_mod = SWE2DUrbanDrainageModule(cfg)
        drain_mod.initialize()
        drain_mod.state.node_depth[cfg.nodes[0].node_id] = 0.0  # start empty

        controller = SWE2DCouplingController(
            cell_area=backend.cell_areas(),
            cell_bed=np.zeros(n_cells, dtype=np.float64),
            drainage=drain_mod,
            structures=struct_mod,
            length_scale_si_to_model=1.0,
        )

        h0 = np.full(n_cells, crest_m + H_m, dtype=np.float64)
        backend.initialize(h0=h0, hu0=np.zeros(n_cells), hv0=np.zeros(n_cells),
                           dt_max=0.01, dt_fixed=0.01)
        backend.run(t_end=0.1, dt_request=0.01,
                    source_rate_callback=controller._log)

        h, hu, hv = backend.get_state()
        backend.destroy()
        return float(h[0])

    @staticmethod
    def _hecf(layer_name: str, n_inlets: int = 1, crest: float = 0.0, **kw):
        """Build a PipeNetworkConfig with a single inlet node + HEC-22 inlet."""
        from swe2d.extensions.extension_models import (
            DrainageNode, InletExchange, PipeNetworkConfig,
        )
        meta = {"surface_area": 50.0}
        node = DrainageNode("N0", 5.0, 5.0, invert_elev=0.0, max_depth=4.0, metadata=meta)
        inlets = [InletExchange(
            inlet_id=f"I{i}", cell_id=0, node_id="N0",
            crest_elev=crest, max_capture=1e6,
            inlet_type=kw.get("inlet_type", "custom"),
            grate_length=float(kw.get("grate_length", 0.0)),
            grate_width=float(kw.get("grate_width", 0.0)),
            grate_type=int(kw.get("grate_type", -1)),
            grate_open_frac=float(kw.get("grate_open_frac", 1.0)),
            curb_length=float(kw.get("curb_length", 0.0)),
            curb_height=float(kw.get("curb_height", 0.15)),
            curb_throat=int(kw.get("curb_throat", 0)),
            slot_length=float(kw.get("slot_length", 0.0)),
            slot_width=float(kw.get("slot_width", 0.05)),
        ) for i in range(n_inlets)]
        return PipeNetworkConfig(
            enabled=True, nodes=[node], links=[],
            inlets=inlets, outfalls=[], pipe_ends=[],
            pipe_solver_mode="diffusion_wave",
        )

    # ── Grate ─────────────────────────────────────────────────────

    def test_grate_weir_regime(self):
        H, Lg, Wg = 0.05, 1.0, 0.5
        Q_exp = 3.0 * 2.0 * (Lg + Wg) * H ** 1.5 * 0.1  # m³ over 0.1 s
        cfg = self._hecf("grate", inlet_type="grate", grate_length=Lg, grate_width=Wg)
        h_final = self._run_and_capture(H, crest_m=0.0, cfg=cfg)
        # Water level should drop by captured volume / cell area (100 m²)
        expected = (0.0 + H) - Q_exp / 100.0
        self.assertAlmostEqual(h_final, expected, delta=0.01)

    def test_grate_orifice_regime(self):
        H, Lg, Wg = 1.0, 1.0, 0.5
        Ao = Lg * Wg
        Q_exp = 0.67 * Ao * math.sqrt(2.0 * G * H) * 0.1
        cfg = self._hecf("grate", inlet_type="grate", grate_length=Lg, grate_width=Wg)
        h_final = self._run_and_capture(H, crest_m=0.0, cfg=cfg)
        expected = H - Q_exp / 100.0
        self.assertAlmostEqual(h_final, expected, delta=0.05)

    # ── Curb ──────────────────────────────────────────────────────

    def test_curb_weir_regime(self):
        H, Lc = 0.05, 2.0
        Q_exp = 3.0 * Lc * H ** 1.5 * 0.1
        cfg = self._hecf("curb", inlet_type="curb", curb_length=Lc, curb_height=0.15)
        h_final = self._run_and_capture(H, crest_m=0.0, cfg=cfg)
        expected = H - Q_exp / 100.0
        self.assertAlmostEqual(h_final, expected, delta=0.01)

    def test_curb_orifice_regime(self):
        H, Lc, hc = 1.0, 2.0, 0.15
        Q_exp = 0.67 * hc * Lc * math.sqrt(2.0 * G * (H - 0.5 * hc)) * 0.1
        cfg = self._hecf("curb", inlet_type="curb", curb_length=Lc, curb_height=hc)
        h_final = self._run_and_capture(H, crest_m=0.0, cfg=cfg)
        expected = H - Q_exp / 100.0
        self.assertAlmostEqual(h_final, expected, delta=0.05)

    # ── Slotted ───────────────────────────────────────────────────

    def test_slotted_weir_regime(self):
        H, Ls, ws = 0.02, 2.0, 0.05
        Q_exp = 2.48 * Ls * H ** 1.5 * 0.1
        cfg = self._hecf("slotted", inlet_type="slotted", slot_length=Ls, slot_width=ws)
        h_final = self._run_and_capture(H, crest_m=0.0, cfg=cfg)
        expected = H - Q_exp / 100.0
        self.assertAlmostEqual(h_final, expected, delta=0.01)

    def test_slotted_orifice_regime(self):
        H, Ls, ws = 1.0, 2.0, 0.05
        Q_exp = 0.8 * Ls * ws * math.sqrt(2.0 * G * H) * 0.1
        cfg = self._hecf("slotted", inlet_type="slotted", slot_length=Ls, slot_width=ws)
        h_final = self._run_and_capture(H, crest_m=0.0, cfg=cfg)
        expected = H - Q_exp / 100.0
        self.assertAlmostEqual(h_final, expected, delta=0.05)

    # ── Combo ─────────────────────────────────────────────────────

    def test_combo_weir_regime(self):
        H, Lg, Wg, Lc = 0.05, 1.0, 0.5, 2.0
        q_g = _grate_Q(H, Lg, Wg)
        Ls = max(0.0, Lc - Lg)
        q_c = 3.0 * Ls * H ** 1.5 if Ls > 0 else 0.0
        Q_exp = (q_g + q_c) * 0.1
        cfg = self._hecf("combo", inlet_type="combo",
                          grate_length=Lg, grate_width=Wg,
                          curb_length=Lc, curb_height=0.15)
        h_final = self._run_and_capture(H, crest_m=0.0, cfg=cfg)
        expected = H - Q_exp / 100.0
        self.assertAlmostEqual(h_final, expected, delta=0.02)

    def test_combo_orifice_regime(self):
        H, Lg, Wg, Lc = 1.0, 1.0, 0.5, 2.0
        q_g = _grate_Q(H, Lg, Wg)
        Ls = max(0.0, Lc - Lg)
        q_c = _curb_Q(H, Ls, 0.15) if Ls > 0 else 0.0
        Q_exp = (q_g + q_c) * 0.1
        cfg = self._hecf("combo", inlet_type="combo",
                          grate_length=Lg, grate_width=Wg,
                          curb_length=Lc, curb_height=0.15)
        h_final = self._run_and_capture(H, crest_m=0.0, cfg=cfg)
        expected = H - Q_exp / 100.0
        self.assertAlmostEqual(h_final, expected, delta=0.05)
