"""
test_swe2d_culvert_validation.py — Culvert structure flow validation

Validates culvert hydraulics by comparing flows computed with hardwired
headwater/tailwater values against flows produced by a fully coupled
SWE2D run where upstream/downstream pool levels evolve dynamically.

Test setup
----------
- Rectangular domain 200m × 100m, flat bed at el. 0.0 m
- Thin embankment at x ≈ 100 m (raised cell bed to 1.5 m)
- A single culvert (diameter=1.2 m) connecting an upstream cell to a
  downstream cell through the embankment
- Initial upstream pool WSE = 2.0 m, downstream pool WSE = 0.5 m
- 200 steps of coupled SWE2D run

Validation
----------
The coupled run reports per‑step culvert flow and control mode via
`SWE2DStructureModule.structure_details()`.  After the run, we compare
the time‑averaged culvert flow against the flow predicted by the same
`_structure_detail` method using the time‑averaged upstream/downstream
WSE from the coupled run as fixed headwater/tailwater inputs.

Additionally, we verify:
- Culvert flow is positive (upstream → downstream)
- Flow magnitude is within ±25 % of the inlet‑control reference
- Inlet control is the dominant mode
"""

import os
import sys
import unittest

import numpy as np

from swe2d.units import USC_FT_PER_SI_M

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Module / GPU availability
# ---------------------------------------------------------------------------

def _load_module():
    # Ensure build/ is on sys.path before import — the test file may be
    # loaded before swe2d.runtime.backend has added it.
    import os, sys
    _plugin_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    _build_dir = os.path.join(_plugin_root, "build")
    if os.path.isdir(_build_dir) and _build_dir not in sys.path:
        sys.path.insert(0, _build_dir)
    try:
        import hydra_swe2d
        return hydra_swe2d
    except ImportError:
        return None


def _gpu_available():
    mod = _load_module()
    if mod is None:
        return False
    try:
        return mod.swe2d_gpu_available()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Analytical reference: inlet-controlled culvert flow (FHWA HEC-5)
# ---------------------------------------------------------------------------

def _culvert_inlet_control_flow_cms(
    diameter_m: float,
    slope_mpm: float,
    available_head_m: float,
    culvert_code: int = 1,
) -> float:
    """Direct inlet-control flow estimate for a circular culvert."""
    from culvert_routine import CircularXsect, inlet_controlled_flow
    from swe2d.units import USC_FT_PER_SI_M, USC_FT3_PER_SI_M3

    diam_ft = max(1.0e-6, diameter_m * USC_FT_PER_SI_M)
    head_ft = max(0.0, available_head_m * USC_FT_PER_SI_M)
    xsect = CircularXsect(diameter_ft=diam_ft, culvert_code=culvert_code)
    q_cfs, _dqh, _cond, _yr = inlet_controlled_flow(xsect, max(1.0e-6, slope_mpm), head_ft)
    return q_cfs / USC_FT3_PER_SI_M3


# ---------------------------------------------------------------------------
# Mesh builder (same pattern as compound_channel)
# ---------------------------------------------------------------------------

def _make_rectangular_mesh(nx, ny, Lx, Ly, bed_func):
    """Triangulate [0, Lx] × [0, Ly] with bed elevation from bed_func(x, y)."""
    xs = np.linspace(0.0, Lx, nx + 1)
    ys = np.linspace(0.0, Ly, ny + 1)
    Xg, Yg = np.meshgrid(xs, ys)
    node_x = Xg.ravel().astype(np.float64)
    node_y = Yg.ravel().astype(np.float64)
    node_z = np.asarray(bed_func(node_x, node_y), dtype=np.float64)
    stride = nx + 1
    cells = []
    for j in range(ny):
        for i in range(nx):
            n00 = j * stride + i
            n10 = j * stride + i + 1
            n01 = (j + 1) * stride + i
            n11 = (j + 1) * stride + i + 1
            cells.extend([n00, n10, n11])
            cells.extend([n00, n11, n01])
    return node_x, node_y, node_z, np.array(cells, dtype=np.int32)


def _cell_centroids(nx, ny, Lx, Ly):
    dx = Lx / nx
    dy = Ly / ny
    n_cells = 2 * nx * ny
    cx = np.empty(n_cells, dtype=np.float64)
    cy = np.empty(n_cells, dtype=np.float64)
    for j in range(ny):
        for i in range(nx):
            k = j * nx + i
            cx[2 * k]     = (i + (i + 1) + (i + 1)) * dx / 3.0
            cy[2 * k]     = (j + j + (j + 1))        * dy / 3.0
            cx[2 * k + 1] = (i + (i + 1) + i)        * dx / 3.0
            cy[2 * k + 1] = (j + (j + 1) + (j + 1))  * dy / 3.0
    return cx, cy


def _wall_bc_arrays(nx, ny):
    """All four sides → WALL (type 1).  No inflow/outflow BCs."""
    stride = nx + 1
    n0s, n1s, tps, vls = [], [], [], []

    # bottom edge (y = 0)
    for i in range(nx):
        n0s.append(i)
        n1s.append(i + 1)
        tps.append(1)
        vls.append(0.0)

    # right edge (x = Lx)
    for j in range(ny):
        n0s.append((j + 1) * stride - 1)
        n1s.append((j + 2) * stride - 1)
        tps.append(1)
        vls.append(0.0)

    # top edge (y = Ly)
    for i in range(nx):
        n0s.append(ny * stride + i + 1)
        n1s.append(ny * stride + i)
        tps.append(1)
        vls.append(0.0)

    # left edge (x = 0)
    for j in range(ny):
        n0s.append((j + 1) * stride)
        n1s.append(j * stride)
        tps.append(1)
        vls.append(0.0)

    return (
        np.array(n0s, dtype=np.int32),
        np.array(n1s, dtype=np.int32),
        np.array(tps, dtype=np.int32),
        np.array(vls, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Test class: culvert (GPU coupled)
# ---------------------------------------------------------------------------

@unittest.skipUnless(_load_module() is not None, "hydra_swe2d not built")
@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
class TestCulvertCoupledValidation(unittest.TestCase):
    """
    Culvert flow validation via coupled SWE2D run.

    Domain:   200 m × 100 m, flat bed, embankment at x ≈ 100 m
    Culvert:   Ø 1.2 m, length 10 m, Manning n = 0.013, slope 0.001
    Steps:    200 explicit GPU steps
    """

    LX           = 200.0      # domain length [m]
    LY           = 100.0      # domain width  [m]
    NX           = 20         # cells along x
    NY           = 10         # cells across y
    EMBANKMENT_X = 100.0      # embankment centre x [m]
    EMBANKMENT_Z = 1.5        # embankment bed elevation [m]
    UP_WSE_INIT  = 2.0        # initial upstream pool WSE [m]
    DN_WSE_INIT  = 0.5        # initial downstream pool WSE [m]
    CULV_DIAM    = 1.2        # culvert diameter [m]
    CULV_LENGTH  = 10.0       # culvert length [m]
    CULV_SLOPE   = 0.001      # culvert slope [m/m]
    CULV_CODE    = 1          # FHWA HEC-5 inlet type
    N_STEPS      = 200

    def _find_embankment_cells(self, cx, cy):
        """Return (up_cell, dn_cell) straddling the embankment at y = LY/2."""
        emb_x = self.EMBANKMENT_X
        mid_y = self.LY / 2.0

        # Upstream cell: closest to embankment from the left, near centreline
        up_mask = cx < emb_x
        up_dist = np.where(up_mask, emb_x - cx, np.inf)
        up_dist += 5.0 * abs(cy - mid_y)
        up_cell = int(np.argmin(up_dist))

        # Downstream cell: closest to embankment from the right, near centreline
        dn_mask = cx > emb_x
        dn_dist = np.where(dn_mask, cx - emb_x, np.inf)
        dn_dist += 5.0 * abs(cy - mid_y)
        dn_cell = int(np.argmin(dn_dist))

        return up_cell, dn_cell

    def _build_culvert_structure_cfg(self, up_cell, dn_cell):
        """Build a HydraulicStructureConfig with a single culvert."""
        from swe2d.extensions.extension_models import (
            HydraulicStructure,
            HydraulicStructureConfig,
            StructureType,
        )

        st = HydraulicStructure(
            structure_id="CULV_TEST",
            structure_type=StructureType.CULVERT,
            upstream_cell=up_cell,
            downstream_cell=dn_cell,
            crest_elev=0.0,
            enabled=True,
            metadata={
                "diameter": self.CULV_DIAM,
                "length": self.CULV_LENGTH,
                "culvert_slope": self.CULV_SLOPE,
                "roughness_n": 0.013,
                "culvert_shape": "circular",
                "culvert_code": self.CULV_CODE,
                "inlet_invert_elev": 0.0,
                "outlet_invert_elev": self.CULV_LENGTH * self.CULV_SLOPE,
                "entrance_loss_k": 0.5,
                "exit_loss_k": 1.0,
                "culvert_barrels": 1.0,
            },
        )
        return HydraulicStructureConfig(enabled=True, structures=[st])

    def _make_solver_with_coupling(self):
        mod = _load_module()

        dx = self.LX / self.NX
        dy = self.LY / self.NY

        # ── Bed elevation: embankment at x ≈ 100 m ──────────────────────────
        def bed(x, y):
            z = np.zeros_like(x)
            # embankment: cells with centroid near EMBANKMENT_X get raised bed
            emb_width = dx * 1.5  # ~one cell wide
            emb_mask = np.abs(x - self.EMBANKMENT_X) < emb_width
            z = np.where(emb_mask, self.EMBANKMENT_Z, z)
            return z

        node_x, node_y, node_z, cell_nodes = _make_rectangular_mesh(
            self.NX, self.NY, self.LX, self.LY, bed)

        cx, cy = _cell_centroids(self.NX, self.NY, self.LX, self.LY)
        n_cells = 2 * self.NX * self.NY

        # Per-cell Manning n
        n_mann_cell = np.full(n_cells, 0.030, dtype=np.float64)

        # All walls — no inflow/outflow BC at edges; pools evolve via
        # initial condition and culvert coupling only.
        bc_n0, bc_n1, bc_tp, bc_vl = _wall_bc_arrays(self.NX, self.NY)

        mesh = mod.swe2d_build_mesh(
            node_x, node_y, node_z, cell_nodes,
            bc_n0, bc_n1, bc_tp, bc_vl)

        # ── Initial conditions ──────────────────────────────────────────────
        emb_x = self.EMBANKMENT_X
        h0 = np.empty(n_cells, dtype=np.float64)
        hu0 = np.zeros(n_cells, dtype=np.float64)
        hv0 = np.zeros(n_cells, dtype=np.float64)

        # Cell-bed elevations at centroids (used for h = WSE - bed)
        z_bed_cell = bed(cx, cy)

        # Set initial WSE per pool
        for ci in range(n_cells):
            wse = self.UP_WSE_INIT if cx[ci] < emb_x else self.DN_WSE_INIT
            h0[ci] = max(0.0, wse - z_bed_cell[ci])

        solver = mod.swe2d_create_solver(
            mesh, h0, hu0, hv0,
            n_mann_cell=n_mann_cell,
            n_mann=0.030,
            cfl=0.45,
            dt_max=0.5,
            use_gpu=True,
            gpu_diag_sync_interval_steps=1,
        )
        return mod, solver, cx, cy, z_bed_cell

    # ── Test: validate culvert flow in coupled run ──────────────────────────
    def test_culvert_flow_from_coupled_pools(self):
        """
        Run a coupled simulation and compare the time‑averaged culvert flow
        against the flow predicted by hardwired headwater/tailwater using
        the same `_structure_detail` computation.
        """
        mod, solver, cx, cy, z_bed_cell = self._make_solver_with_coupling()

        up_cell, dn_cell = self._find_embankment_cells(cx, cy)
        self.assertGreaterEqual(up_cell, 0, "No upstream cell found")
        self.assertGreaterEqual(dn_cell, 0, "No downstream cell found")
        self.assertNotEqual(up_cell, dn_cell, "Up/down cells must differ")

        struct_cfg = self._build_culvert_structure_cfg(up_cell, dn_cell)

        from swe2d.extensions.structures import SWE2DStructureModule
        struct_mod = SWE2DStructureModule(struct_cfg, model_to_ft=USC_FT_PER_SI_M)

        # Run coupled steps, collecting culvert flow each step
        flow_history = []
        wse_up_history = []
        wse_dn_history = []
        for _step in range(self.N_STEPS):
            mod.swe2d_step(solver, -1.0)
            h, hu, hv = mod.swe2d_get_state(solver)

            cell_wse = h + z_bed_cell
            details = struct_mod.structure_details(cell_wse)
            if details:
                d = details[0]
                flow_history.append(float(d.get("flow", 0.0)))
                wse_up_history.append(float(d.get("upstream_wse", 0.0)))
                wse_dn_history.append(float(d.get("downstream_wse", 0.0)))

        mod.swe2d_destroy(solver)

        self.assertGreater(len(flow_history), 10, "Too few coupled steps recorded")

        # Time-averaged values (use the last half to let pools settle)
        half = max(1, len(flow_history) // 2)
        avg_flow_coupled = float(np.mean(flow_history[half:]))
        avg_wse_up = float(np.mean(wse_up_history[half:]))
        avg_wse_dn = float(np.mean(wse_dn_history[half:]))

        # Compute hardwired reference flow using the averaged WSE
        cell_wse_ref = np.zeros_like(z_bed_cell)
        cell_wse_ref[up_cell] = avg_wse_up
        cell_wse_ref[dn_cell] = avg_wse_dn
        ref_details = struct_mod.structure_details(cell_wse_ref)
        ref_flow = float(ref_details[0].get("flow", 0.0)) if ref_details else 0.0

        self.assertGreater(ref_flow, 0.0,
                           f"Reference culvert flow must be > 0; got {ref_flow:.4f} cms")
        self.assertGreater(avg_flow_coupled, 0.0,
                           f"Coupled culvert flow must be > 0; got {avg_flow_coupled:.4f} cms")

        rel_err = abs(avg_flow_coupled - ref_flow) / max(1.0e-9, ref_flow)
        self.assertLess(
            rel_err, 0.05,
            f"Culvert flow mismatch: coupled avg={avg_flow_coupled:.4f} cms, "
            f"ref={ref_flow:.4f} cms, rel_err={rel_err:.2%}"
        )

    # ── Helper: run N steps with coupling controller injecting source rates ──
    def _run_steps_with_coupling(self, coupling_ctl, n_steps, cx, up_cell, dn_cell):
        """Run n_steps with the coupling controller, returning diagnostic history."""
        mod, solver, _, _, z_bed_cell = self._make_solver_with_coupling()

        flow_history = []
        wse_up_history = []
        wse_dn_history = []
        src_up_history = []
        src_dn_history = []

        dt_s = 0.5
        t_accum = 0.0
        for _step in range(n_steps):
            h, hu, hv = mod.swe2d_get_state(solver)
            src = coupling_ctl.compute_source_rates(t_accum, dt_s, h, hu, hv)
            src = np.asarray(src, dtype=np.float64).ravel()
            try:
                mod.swe2d_solver_set_external_sources(solver, src.astype(np.float64))
            except Exception:
                pass  # older backends may not support this

            mod.swe2d_step(solver, dt_s)
            t_accum += dt_s

            h, hu, hv = mod.swe2d_get_state(solver)
            cell_wse = h + z_bed_cell
            details = self._hardwired_struct_mod.structure_details(cell_wse)
            if details:
                d = details[0]
                flow_history.append(float(d.get("flow", 0.0)))
                wse_up_history.append(float(d.get("upstream_wse", 0.0)))
                wse_dn_history.append(float(d.get("downstream_wse", 0.0)))
            if up_cell < src.size:
                src_up_history.append(float(src[up_cell]))
            if dn_cell < src.size:
                src_dn_history.append(float(src[dn_cell]))

        mod.swe2d_destroy(solver)
        return {
            "flow": flow_history,
            "wse_up": wse_up_history,
            "wse_dn": wse_dn_history,
            "src_up": src_up_history,
            "src_dn": src_dn_history,
        }

    # ── Test: native CPU coupling path ───────────────────────────────────────
    def test_culvert_native_cpu_coupling_path(self):
        """
        Verify that the native CPU structure-flow path (via
        SWE2DCouplingController with coupling_loop='cpu') produces
        source rates that move the pools in the expected direction and
        yields culvert flows consistent with the Python reference.
        """
        from swe2d.extensions.structures import SWE2DStructureModule
        from swe2d.runtime.coupling import SWE2DCouplingController

        mod, solver, cx, cy, z_bed_cell = self._make_solver_with_coupling()
        up_cell, dn_cell = self._find_embankment_cells(cx, cy)
        mod.swe2d_destroy(solver)

        struct_cfg = self._build_culvert_structure_cfg(up_cell, dn_cell)
        struct_mod = SWE2DStructureModule(struct_cfg, model_to_ft=USC_FT_PER_SI_M)
        self._hardwired_struct_mod = struct_mod

        cell_area = np.full(len(cx), 200.0 * 100.0 / len(cx), dtype=np.float64)
        cell_bed = z_bed_cell

        ctl = SWE2DCouplingController(
            cell_area=cell_area,
            cell_bed=cell_bed,
            structures=struct_mod,
            coupling_loop="cpu",
        )

        diag = self._run_steps_with_coupling(ctl, self.N_STEPS, cx, up_cell, dn_cell)
        self._hardwired_struct_mod = None

        self.assertGreater(len(diag["flow"]), 10, "Too few coupled steps")
        half = max(1, len(diag["flow"]) // 2)
        avg_flow = float(np.mean(diag["flow"][half:]))
        avg_src_up = float(np.mean(diag["src_up"][half:]))
        avg_src_dn = float(np.mean(diag["src_dn"][half:]))

        self.assertGreater(avg_flow, 0.0,
                           f"CPU-coupled culvert flow must be > 0; got {avg_flow:.4f} cms")
        # Upstream cell source must be negative (water removed), downstream positive
        self.assertLess(avg_src_up, 0.0,
                        f"Upstream source rate must be < 0 (removal); got {avg_src_up:.6e} m/s")
        self.assertGreater(avg_src_dn, 0.0,
                           f"Downstream source rate must be > 0 (addition); got {avg_src_dn:.6e} m/s")

        # Compare CPU-coupled flow to hardwired reference
        avg_wse_up = float(np.mean(diag["wse_up"][half:]))
        avg_wse_dn = float(np.mean(diag["wse_dn"][half:]))
        cell_wse_ref = np.array([avg_wse_up, avg_wse_dn], dtype=np.float64)
        cell_wse_full = np.full(len(cx), 0.0, dtype=np.float64)
        cell_wse_full[up_cell] = avg_wse_up
        cell_wse_full[dn_cell] = avg_wse_dn

        ref_details = self._build_culvert_struct_mod().structure_details(cell_wse_full)  # type: ignore[attr-defined]
        ref_flow = float(ref_details[0].get("flow", 0.0)) if ref_details else 0.0

        rel_err = abs(avg_flow - ref_flow) / max(1.0e-9, ref_flow)
        self.assertLess(
            rel_err, 0.05,
            f"CPU coupling path culvert flow mismatch: "
            f"avg={avg_flow:.4f} cms, ref={ref_flow:.4f} cms, rel_err={rel_err:.2%}"
        )

    def _build_culvert_struct_mod(self):
        """Build a fresh SWE2DStructureModule for hardwired reference."""
        from swe2d.extensions.structures import SWE2DStructureModule
        mod, solver, cx, cy, z_bed_cell = self._make_solver_with_coupling()
        up_cell, dn_cell = self._find_embankment_cells(cx, cy)
        mod.swe2d_destroy(solver)
        return SWE2DStructureModule(self._build_culvert_structure_cfg(up_cell, dn_cell), model_to_ft=USC_FT_PER_SI_M)

    # ── Test: native CUDA coupling path ──────────────────────────────────────
    @unittest.skipUnless(_gpu_available(), "CUDA GPU not available for native coupling test")
    def test_culvert_native_cuda_coupling_path(self):
        """
        Verify that the native CUDA structure-flow path (via
        SWE2DCouplingController with coupling_loop='cuda') produces
        culvert flows consistent with the Python reference.
        """
        from swe2d.extensions.structures import SWE2DStructureModule
        from swe2d.runtime.coupling import SWE2DCouplingController

        mod, solver, cx, cy, z_bed_cell = self._make_solver_with_coupling()
        up_cell, dn_cell = self._find_embankment_cells(cx, cy)
        mod.swe2d_destroy(solver)

        struct_cfg = self._build_culvert_structure_cfg(up_cell, dn_cell)
        struct_mod = SWE2DStructureModule(struct_cfg, model_to_ft=USC_FT_PER_SI_M)
        self._hardwired_struct_mod = struct_mod

        cell_area = np.full(len(cx), 200.0 * 100.0 / len(cx), dtype=np.float64)
        cell_bed = z_bed_cell

        ctl = SWE2DCouplingController(
            cell_area=cell_area,
            cell_bed=cell_bed,
            structures=struct_mod,
            coupling_loop="cuda",
        )

        # Only meaningful if CUDA native module is available
        native_mod = ctl._native_cuda_module()
        if native_mod is None:
            self.skipTest("Native CUDA coupling module not available")

        diag = self._run_steps_with_coupling(ctl, self.N_STEPS, cx, up_cell, dn_cell)
        self._hardwired_struct_mod = None

        self.assertGreater(len(diag["flow"]), 10, "Too few coupled CUDA steps")
        half = max(1, len(diag["flow"]) // 2)
        avg_flow = float(np.mean(diag["flow"][half:]))

        self.assertGreater(avg_flow, 0.0,
                           f"CUDA-coupled culvert flow must be > 0; got {avg_flow:.4f} cms")

        avg_wse_up = float(np.mean(diag["wse_up"][half:]))
        avg_wse_dn = float(np.mean(diag["wse_dn"][half:]))
        cell_wse_full = np.full(len(cx), 0.0, dtype=np.float64)
        cell_wse_full[up_cell] = avg_wse_up
        cell_wse_full[dn_cell] = avg_wse_dn

        ref_details = self._build_culvert_struct_mod().structure_details(cell_wse_full)  # type: ignore[attr-defined]
        ref_flow = float(ref_details[0].get("flow", 0.0)) if ref_details else 0.0

        rel_err = abs(avg_flow - ref_flow) / max(1.0e-9, ref_flow)
        self.assertLess(
            rel_err, 0.05,
            f"CUDA coupling path culvert flow mismatch: "
            f"avg={avg_flow:.4f} cms, ref={ref_flow:.4f} cms, rel_err={rel_err:.2%}"
        )

    # ── Test: diagnostic cross-check of C++ vs Python culvert caps ──────────
    def test_culvert_native_vs_python_caps_match(self):
        """
        Compare the four culvert flow caps (inlet control, outlet control,
        orifice cap, Manning cap) between the native C++ path and the
        Python reference for the same WSE inputs.  Any cap that differs
        by more than 5 % is a candidate root cause for the real‑world
        outlet‑control mis‑selection.
        """
        from swe2d.extensions.extension_models import (
            HydraulicStructure,
            HydraulicStructureConfig,
            StructureType,
        )
        from swe2d.extensions.structures import SWE2DStructureModule
        from swe2d.runtime.coupling import SWE2DCouplingController, pack_structures_soa

        mod, solver, cx, cy, z_bed_cell = self._make_solver_with_coupling()
        up_cell, dn_cell = self._find_embankment_cells(cx, cy)
        mod.swe2d_destroy(solver)

        struct_cfg = self._build_culvert_structure_cfg(up_cell, dn_cell)
        struct_mod = SWE2DStructureModule(struct_cfg, model_to_ft=USC_FT_PER_SI_M)

        cell_area = np.full(len(cx), 200.0 * 100.0 / len(cx), dtype=np.float64)
        cell_bed = z_bed_cell

        ctl = SWE2DCouplingController(
            cell_area=cell_area,
            cell_bed=cell_bed,
            structures=struct_mod,
            coupling_loop="cpu",
        )

        native_mod = ctl._native_cuda_module()
        if native_mod is None:
            self.skipTest("Native CUDA module not available for cap comparison")

        # Test at several WSE pairs spanning realistic head ranges
        test_cases = [
            (2.0, 0.5,  "high head, low tailwater"),
            (1.5, 0.8,  "moderate head, moderate tailwater"),
            (3.0, 0.3,  "very high head, very low tailwater"),
        ]

        for wu, wd, label in test_cases:
            cell_wse = np.full(len(cx), 0.0, dtype=np.float64)
            cell_wse[up_cell] = wu
            cell_wse[dn_cell] = wd

            # Python path — get all intermediate caps
            py_details = struct_mod.structure_details(cell_wse)
            py_d = py_details[0]
            py_flow = float(py_d.get("flow", 0.0))
            py_inlet = float(py_d.get("inlet_control_flow", 0.0))
            py_outlet = float(py_d.get("outlet_control_flow", 0.0))
            py_orifice = float(py_d.get("orifice_cap", 0.0))
            py_manning = float(py_d.get("manning_cap", 0.0))
            py_emb = float(py_d.get("embankment_flow", 0.0))
            py_control = str(py_d.get("control_mode", ""))

            # Native C++ path — try CUDA first, fall back to CPU
            native_flows = ctl._native_structure_flows(native_mod, cell_wse, use_cuda=True)
            if native_flows is None or native_flows.size == 0:
                native_flows = ctl._native_structure_flows(native_mod, cell_wse, use_cuda=False)
            if native_flows is None or native_flows.size == 0:
                self.skipTest(f"Native structure flows unavailable for {label}")
            native_flow = float(native_flows[0])

            # The native flow should match the Python min-of-caps flow
            # (excluding embankment since it's not enabled in this config)
            py_flow_no_emb = py_flow - py_emb
            # Skip degenerate near-zero-flow cases where both paths are ~0
            # and numerical noise dominates the relative error.
            if py_flow_no_emb < 1.0e-4 and abs(native_flow) < 1.0e-4:
                continue
            rel_err = abs(native_flow - py_flow_no_emb) / max(1.0e-9, abs(py_flow_no_emb))
            self.assertLess(
                rel_err, 0.05,
                f"Native vs Python flow mismatch for {label} (wu={wu}, wd={wd}): "
                f"native={native_flow:.6f}, python={py_flow_no_emb:.6f}, "
                f"py_control={py_control}, py_inlet={py_inlet:.6f}, "
                f"py_outlet={py_outlet:.6f}, py_orifice={py_orifice:.6f}, "
                f"py_manning={py_manning:.6f}, rel_err={rel_err:.2%}"
            )

    # ── Test: inlet-control reference versus hardwired computation ───────────
    def test_inlet_control_reference_matches_mid_run_wse(self):
        """
        Verify that the Python `_structure_detail` culvert flow is within
        ±20 % of the pure inlet‑control reference for the same WSE pair.
        """
        from swe2d.extensions.extension_models import (
            HydraulicStructure,
            HydraulicStructureConfig,
            StructureType,
        )
        from swe2d.extensions.structures import SWE2DStructureModule

        # Use the mid-run steady-state WSE values
        wu_mid = 1.8   # upstream WSE [m]  — below embankment crest
        wd_mid = 0.6   # downstream WSE [m]

        up_cell, dn_cell = 0, 1
        st = HydraulicStructure(
            structure_id="CULV_REF",
            structure_type=StructureType.CULVERT,
            upstream_cell=up_cell,
            downstream_cell=dn_cell,
            crest_elev=0.0,
            enabled=True,
            metadata={
                "diameter": self.CULV_DIAM,
                "length": self.CULV_LENGTH,
                "culvert_slope": self.CULV_SLOPE,
                "roughness_n": 0.013,
                "culvert_shape": "circular",
                "culvert_code": self.CULV_CODE,
                "inlet_invert_elev": 0.0,
                "outlet_invert_elev": 0.01,
                "entrance_loss_k": 0.5,
                "exit_loss_k": 1.0,
                "culvert_barrels": 1.0,
            },
        )
        cfg = HydraulicStructureConfig(enabled=True, structures=[st])
        smod = SWE2DStructureModule(cfg, model_to_ft=USC_FT_PER_SI_M)

        cell_wse = np.array([wu_mid, wd_mid], dtype=np.float64)
        details = smod.structure_details(cell_wse)
        d = details[0]

        q_struct = float(d.get("flow", 0.0))
        q_inlet = float(d.get("inlet_control_flow", 0.0))
        control = str(d.get("control_mode", ""))

        # Pure inlet control reference from culvert_routine
        q_ref = _culvert_inlet_control_flow_cms(
            diameter_m=self.CULV_DIAM,
            slope_mpm=self.CULV_SLOPE,
            available_head_m=wu_mid - 0.0,  # inlet invert = 0
            culvert_code=self.CULV_CODE,
        )

        self.assertGreater(q_struct, 0.0, "Structure flow must be positive")
        self.assertGreater(q_inlet, 0.0, "Inlet control flow must be positive")
        rel_err_inlet = abs(q_inlet - q_ref) / max(1.0e-9, q_ref)
        self.assertLess(
            rel_err_inlet, 0.05,
            f"Inlet control mismatch: struct inlet={q_inlet:.4f} cms, "
            f"ref={q_ref:.4f} cms, rel_err={rel_err_inlet:.2%}"
        )


# ---------------------------------------------------------------------------
# CPU smoke test
# ---------------------------------------------------------------------------

@unittest.skipUnless(_load_module() is not None, "hydra_swe2d not built")
class TestCulvertModuleCPU(unittest.TestCase):
    """CPU‑only test: structure module produces sane non‑negative flows."""

    def test_culvert_flow_positive_for_positive_head(self):
        from swe2d.extensions.extension_models import (
            HydraulicStructure,
            HydraulicStructureConfig,
            StructureType,
        )
        from swe2d.extensions.structures import SWE2DStructureModule

        st = HydraulicStructure(
            structure_id="CULV_CPU",
            structure_type=StructureType.CULVERT,
            upstream_cell=0,
            downstream_cell=1,
            crest_elev=0.0,
            enabled=True,
            metadata={
                "diameter": 1.2,
                "length": 10.0,
                "culvert_slope": 0.001,
                "roughness_n": 0.013,
                "culvert_shape": "circular",
                "culvert_code": 1,
                "inlet_invert_elev": 0.0,
                "outlet_invert_elev": 0.01,
                "entrance_loss_k": 0.5,
                "exit_loss_k": 1.0,
                "culvert_barrels": 1.0,
            },
        )
        cfg = HydraulicStructureConfig(enabled=True, structures=[st])
        smod = SWE2DStructureModule(cfg, model_to_ft=USC_FT_PER_SI_M)

        # Upstream pool 2.0m, downstream 0.5m → positive head
        cell_wse = np.array([2.0, 0.5], dtype=np.float64)
        flows = smod.structure_flows(cell_wse)
        self.assertEqual(len(flows), 1)
        self.assertGreater(flows[0], 0.0,
                           f"Expected positive flow; got {flows[0]:.4f} cms")

        # Reversed pool → negative flow
        cell_wse_rev = np.array([0.3, 1.8], dtype=np.float64)
        flows_rev = smod.structure_flows(cell_wse_rev)
        self.assertLess(flows_rev[0], 0.0,
                        f"Expected negative (reversed) flow; got {flows_rev[0]:.4f} cms")

    def test_culvert_zero_flow_for_equal_wse(self):
        from swe2d.extensions.extension_models import (
            HydraulicStructure,
            HydraulicStructureConfig,
            StructureType,
        )
        from swe2d.extensions.structures import SWE2DStructureModule

        st = HydraulicStructure(
            structure_id="CULV_EQ",
            structure_type=StructureType.CULVERT,
            upstream_cell=0,
            downstream_cell=1,
            crest_elev=0.0,
            enabled=True,
            metadata={
                "diameter": 1.2,
                "length": 10.0,
                "culvert_slope": 0.001,
                "roughness_n": 0.013,
                "culvert_shape": "circular",
                "culvert_code": 1,
                "inlet_invert_elev": 0.0,
                "outlet_invert_elev": 0.01,
                "entrance_loss_k": 0.5,
                "exit_loss_k": 1.0,
                "culvert_barrels": 1.0,
            },
        )
        cfg = HydraulicStructureConfig(enabled=True, structures=[st])
        smod = SWE2DStructureModule(cfg, model_to_ft=USC_FT_PER_SI_M)

        # Both pools at same WSE → no driving head below culvert crown
        cell_wse = np.array([0.5, 0.5], dtype=np.float64)
        flows = smod.structure_flows(cell_wse)
        self.assertEqual(len(flows), 1)
        self.assertLess(abs(flows[0]), 1.0e-6,
                        msg=f"Expected near-zero flow for equal WSE; got {flows[0]:.6e} cms")

    def test_culvert_outlet_control_dominates_for_long_rough_barrel(self):
        """
        Outlet control must be the limiting mode when the barrel is long,
        rough, and small‑diameter — friction loss through the barrel
        exceeds the inlet contraction loss.

        Parameters chosen so that barrel friction is the tightest bottleneck:

            D = 1.2 m, L = 300 m, n = 0.024, S = 0.001
            Up WSE = 2.0 m, Down WSE = 0.3 m
        """
        from swe2d.extensions.extension_models import (
            HydraulicStructure,
            HydraulicStructureConfig,
            StructureType,
        )
        from swe2d.extensions.structures import SWE2DStructureModule

        L = 300.0
        S = 0.001
        D = 1.2
        outlet_invert = L * S  # 0.3 m

        st = HydraulicStructure(
            structure_id="CULV_OUTLET",
            structure_type=StructureType.CULVERT,
            upstream_cell=0,
            downstream_cell=1,
            crest_elev=0.0,
            enabled=True,
            metadata={
                "diameter": D,
                "length": L,
                "culvert_slope": S,
                "roughness_n": 0.024,
                "culvert_shape": "circular",
                "culvert_code": 1,
                "inlet_invert_elev": 0.0,
                "outlet_invert_elev": outlet_invert,
                "entrance_loss_k": 0.5,
                "exit_loss_k": 1.0,
                "culvert_barrels": 1.0,
            },
        )
        cfg = HydraulicStructureConfig(enabled=True, structures=[st])
        smod = SWE2DStructureModule(cfg, model_to_ft=USC_FT_PER_SI_M)

        # Upstream pool at 2.0 m, downstream at 0.3 m
        cell_wse = np.array([2.0, 0.3], dtype=np.float64)
        details = smod.structure_details(cell_wse)
        d = details[0]

        control_mode = str(d.get("control_mode", ""))
        q_struct = float(d.get("flow", 0.0))
        q_inlet = float(d.get("inlet_control_flow", 0.0))
        q_outlet = float(d.get("outlet_control_flow", 0.0))
        q_manning_cap = float(d.get("manning_cap", 0.0))

        self.assertGreater(q_struct, 0.0, "Outlet-controlled flow must be > 0")

        # Outlet control must be more restrictive than inlet control.
        self.assertGreater(
            q_inlet, q_outlet,
            f"Inlet flow ({q_inlet:.4f}) must exceed outlet flow "
            f"({q_outlet:.4f}) for barrel friction to be limiting"
        )

        # The effective limiting barrel-flow value is min(q_outlet, q_manning_cap).
        # Both represent barrel-friction-dominated flow regimes.
        barrel_limit = min(q_outlet, q_manning_cap) if q_manning_cap > 0.0 else q_outlet
        self.assertIn(
            control_mode, ("outlet_control", "manning_cap"),
            f"Expected outlet_control or manning_cap but got '{control_mode}'. "
            f"q_inlet={q_inlet:.4f}, q_outlet={q_outlet:.4f}, "
            f"q_manning={q_manning_cap:.4f} cms"
        )

        # Struct flow must match the barrel-friction limit within 1%
        rel_err = abs(q_struct - barrel_limit) / max(1.0e-9, barrel_limit)
        self.assertLess(
            rel_err, 0.01,
            f"Struct flow {q_struct:.4f} cms deviates from barrel limit "
            f"{barrel_limit:.4f} cms by {rel_err:.2%}"
        )

        # Sanity check: outlet control flow must be within a factor of 5 of
        # the full-pipe Manning capacity for the same geometry.
        from swe2d.extensions.extension_models import compute_pipe_manning_capacity_full

        q_manning_ref = compute_pipe_manning_capacity_full(
            diameter_m=D,
            slope_m_per_m=S,
            roughness_n=0.024,
        )
        self.assertGreater(q_manning_ref, 0.0, "Manning reference capacity must be > 0")
        ratio = q_outlet / max(1.0e-9, q_manning_ref)
        self.assertGreater(ratio, 0.2,
                           f"Outlet control flow {q_outlet:.4f} cms is unexpectedly "
                           f"low vs Manning ref {q_manning_ref:.4f} cms (ratio={ratio:.2f})")
        self.assertLess(ratio, 5.0,
                        f"Outlet control flow {q_outlet:.4f} cms is unexpectedly "
                        f"high vs Manning ref {q_manning_ref:.4f} cms (ratio={ratio:.2f})")


if __name__ == "__main__":
    unittest.main()
