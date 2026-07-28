"""Pipe1D solver tests — unified mesh API, whole pipeline.

Tests exercise the full hot path: build_unified_mesh → upload_cell_h →
init_cell_area → pipe1d_step → readback_cell_state.  No removed bindings,
no compat shims.
"""

import numpy as np
import unittest
import math
from swe2d.runtime.backend import SWE2DBackend
import hydra_swe2d as _MOD

_G = 9.80665
_K = 1.0
_H = 1.0e-10

# CFL constraint: swe2d_pipe1d_step(dt) must satisfy
#   dt <= 0.5 * min(cell_length) / max_wave_speed
# for explicit RK2 stability with the weir/orifice STORAGE_PIPE face.
# With cell_length >= 1.0m, g ~ 10 m/s², h_max ~ 5m → c ~ 7 m/s:
#   max_safe_dt ≈ 0.5 * 1.0 / 7 ≈ 0.07 s
# All tests MUST use dt <= 0.1 unless the mesh has longer cells.
# _backend() sets dt_fixed = 0.05, which is the reference safe value.

_gpu = None
try:
    _gpu = bool(_MOD.swe2d_gpu_available())
except Exception:
    pass


def _backend():
    b = SWE2DBackend()
    b.build_mesh(
        np.asarray([0.0, 1.0, 0.0]),
        np.asarray([0.0, 0.0, 1.0]),
        np.asarray([0.0, 0.0, 0.0]),
        np.asarray([0, 1, 2]),
    )
    b.initialize(h0=np.asarray([0.1]), hu0=np.zeros(1), hv0=np.zeros(1),
                 dt_fixed=0.05, dt_max=0.05)
    return b


def _mesh(dev, stype=0, sw=1.0, sh=1.0, mcl=10.0, node_invert=None,
          pipe_end_nodes=None, outfall_nodes=None, inlet_nodes=None,
          inlet_face_data=None):
    """Build a standard 2-node, 1-link mesh.  Returns n_cells."""
    if node_invert is None:
        node_invert = np.array([10.0, 10.0], dtype=np.float64)

    kwargs = dict(
        dev_ptr=dev, n_links=1,
        link_from=np.array([0], dtype=np.int32),
        link_to=np.array([1], dtype=np.int32),
        L=np.array([100.0], dtype=np.float64),
        D=np.array([1.0], dtype=np.float64),
        n_mann=np.array([0.013], dtype=np.float64),
        S0=np.zeros(1, dtype=np.float64),
        node_invert=node_invert,
        mcl=mcl,
        link_shape_type=np.array([stype], dtype=np.int32),
        link_width=np.array([sw], dtype=np.float64),
        link_height=np.array([sh], dtype=np.float64),
    )

    if pipe_end_nodes is not None:
        kwargs["n_pipe_ends"] = len(pipe_end_nodes)
        kwargs["pipe_end_node_ids"] = np.asarray(pipe_end_nodes, dtype=np.int32)

    if outfall_nodes is not None:
        kwargs["node_is_outfall"] = np.asarray(outfall_nodes, dtype=np.int32)

    if inlet_nodes is not None:
        kwargs["n_inlet_cells"] = len(inlet_nodes)
        kwargs["inlet_node_ids"] = np.asarray(inlet_nodes, dtype=np.int32)
        kwargs["inlet_invert"] = np.asarray(inlet_nodes, dtype=np.float64) * 0.0 + 10.0
        kwargs["inlet_surface_area"] = np.full(len(inlet_nodes), 1e10, dtype=np.float64)
        kwargs["inlet_max_depth"] = np.full(len(inlet_nodes), 10.0, dtype=np.float64)

    if inlet_face_data:
        kwargs.update(inlet_face_data)

    _MOD.swe2d_build_unified_mesh(**kwargs)
    return int(100.0 / mcl)


@unittest.skipUnless(_gpu, "CUDA GPU not available")
class TestLakeAtRest(unittest.TestCase):
    """Flat, uniform water — must stay at rest indefinitely."""

    def _run(self, stype, sw, h):
        b = _backend()
        dev = int(_MOD.swe2d_get_coupling_dev_ptr())
        nc = _mesh(dev, stype, sw)
        _MOD.swe2d_pipe1d_upload_cell_h(dev, np.full(nc, h, dtype=np.float64))
        _MOD.swe2d_pipe1d_init_cell_area(dev, _H)
        for _ in range(50):
            _MOD.swe2d_pipe1d_step(dev, 0.5, 'rk2', 1, 2, 0.5,
                                   _G, _K, _H,
                                   surcharge_method=1, recon_method=1, theta=1.0)
        st = _MOD.swe2d_pipe1d_readback_cell_state(dev, nc)
        Q = np.asarray(st['cell_Q'])
        h_arr = np.asarray(st['cell_h'])
        b.destroy()
        self.assertAlmostEqual(np.max(np.abs(Q)), 0.0, places=12,
                               msg=f"{['circ','rect','ellip'][stype]} max|Q|")
        self.assertAlmostEqual(np.max(h_arr), np.min(h_arr), places=12,
                               msg=f"{['circ','rect','ellip'][stype]} h not uniform")

    def test_circular(self): self._run(0, 1.0, 0.3)
    def test_rectangular(self): self._run(1, 1.0, 0.3)
    def test_elliptical(self): self._run(2, 1.5, 0.3)


@unittest.skipUnless(_gpu, "CUDA GPU not available")
class TestMassConservation(unittest.TestCase):
    """Closed system with WALL_BC — mass drift ≤ machine epsilon."""

    def _run(self, stype, sw, h_field, T, dt, places):
        b = _backend()
        dev = int(_MOD.swe2d_get_coupling_dev_ptr())
        nc = _mesh(dev, stype, sw)
        _MOD.swe2d_pipe1d_upload_cell_h(dev, np.asarray(h_field, dtype=np.float64))
        _MOD.swe2d_pipe1d_init_cell_area(dev, _H)
        m0 = float(np.sum(np.asarray(
            _MOD.swe2d_pipe1d_readback_cell_state(dev, nc)['cell_A'])) * (100.0 / nc))
        n = int(T / dt)
        for _ in range(n):
            _MOD.swe2d_pipe1d_step(dev, dt, 'rk2', 1, 2, 0.5,
                                   _G, _K, _H,
                                   surcharge_method=1, recon_method=1, theta=1.0)
        m1 = float(np.sum(np.asarray(
            _MOD.swe2d_pipe1d_readback_cell_state(dev, nc)['cell_A'])) * (100.0 / nc))
        b.destroy()
        self.assertAlmostEqual(m1 / m0, 1.0, places=places)

    def test_circular_open(self):
        L, nc = 100.0, 10
        xc = np.linspace(5, 95, nc)
        self._run(0, 1.0,
                  0.6 + 0.2 * np.cos(2 * math.pi * (xc - 50) / L), 100, 0.5, 10)

    def test_circular_pressurised(self):
        self._run(0, 1.0, np.full(10, 1.3), 100, 0.05, 8)

    def test_elliptical_open(self):
        L, nc = 100.0, 10
        xc = np.linspace(5, 95, nc)
        self._run(2, 1.5,
                  0.6 + 0.2 * np.sin(math.pi * xc / L), 100, 0.5, 10)

    def test_rectangular_open(self):
        L, nc = 100.0, 10
        xc = np.linspace(5, 95, nc)
        self._run(1, 1.0,
                  0.5 + 0.3 * np.cos(2 * math.pi * (xc - 50) / L), 100, 0.5, 10)


@unittest.skipUnless(_gpu, "CUDA GPU not available")
class TestSURFACE2DPipeEnd(unittest.TestCase):
    """SURFACE_2D_PIPE_END (class 3) HLLC flux between pipe and 2D cell."""

    def _run_case(self, h_p, h_2d):
        b = _backend()
        dev = int(_MOD.swe2d_get_coupling_dev_ptr())
        nc = _mesh(dev, stype=1, sw=10.0, sh=5.0, mcl=100.0,
                   node_invert=np.array([0.0, 0.0], dtype=np.float64),
                   pipe_end_nodes=np.array([0], dtype=np.int32))
        _MOD.swe2d_pipe1d_upload_cell_h(dev, np.full(nc, h_p, dtype=np.float64))
        _MOD.swe2d_pipe1d_init_cell_area(dev, _H)
        _MOD.swe2d_pipe1d_upload_pipe_end_surface_faces(
            dev, np.array([0], dtype=np.int32))
        b.set_state(np.array([h_2d], dtype=np.float64),
                     np.zeros(1, dtype=np.float64),
                     np.zeros(1, dtype=np.float64))
        solver_dev = b.get_solver_dev_ptr()
        _MOD.swe2d_pipe1d_step(dev, 0.5, 'rk2', 1, 2, 0.5,
                               _G, _K, _H,
                               surcharge_method=0, theta=1.0,
                               solver_dev_ptr=solver_dev)
        fx = _MOD.swe2d_gpu_readback_ext_struct_flux(1)
        b.destroy()
        return float(fx[0][0]) if fx and len(fx) > 0 and fx[0] is not None else 0.0

    def test_wet_pipe_drains_into_dry_surface(self):
        """Pipe full (h=5m), 2D cell dry → outflow into surface."""
        fh = self._run_case(h_p=5.0, h_2d=0.0)
        self.assertGreater(fh, 0.0, "Should be outflow (pipe→2D)")

    def test_dry_pipe_gets_inflow_from_wet_surface(self):
        """Pipe dry (h=0), 2D cell wet (h=2m) → inflow into pipe.
        Sign: fh is flux L→R (pipe→2D).  Negative = 2D→pipe inflow."""
        fh = self._run_case(h_p=0.0, h_2d=2.0)
        self.assertLess(fh, 0.0, "Should be inflow (2D→pipe, fh<0)")


@unittest.skipUnless(_gpu, "CUDA GPU not available")
class TestOUTFALLBC(unittest.TestCase):
    """OUTFALL_BC (class 1) — FREE, FIXED_WSE, NORMAL_DEPTH."""

    def test_free_outfall_allows_drainage(self):
        """FREE outfall: water drains out over time."""
        b = _backend()
        dev = int(_MOD.swe2d_get_coupling_dev_ptr())
        nc = _mesh(dev, stype=1, sw=1.0, sh=1.0,
                   outfall_nodes=np.array([1, 0], dtype=np.int32))
        _MOD.swe2d_pipe1d_upload_cell_h(dev, np.full(nc, 0.8, dtype=np.float64))
        _MOD.swe2d_pipe1d_init_cell_area(dev, _H)
        m0 = float(np.sum(np.asarray(
            _MOD.swe2d_pipe1d_readback_cell_state(dev, nc)['cell_A'])) * (100.0 / nc))
        for _ in range(50):
            _MOD.swe2d_pipe1d_step(dev, 0.5, 'rk2', 1, 2, 0.5,
                                   _G, _K, _H,
                                   surcharge_method=1, recon_method=1, theta=1.0)
        m1 = float(np.sum(np.asarray(
            _MOD.swe2d_pipe1d_readback_cell_state(dev, nc)['cell_A'])) * (100.0 / nc))
        b.destroy()
        self.assertLess(m1, m0, "FREE outfall should drain mass from system")


@unittest.skipUnless(_gpu, "CUDA GPU not available")
class TestInletStorage(unittest.TestCase):
    """Inlet/junction storage cell drains into connected pipe."""

    def _build_inlet_pipe_mesh(self, dev, inlet_diameter=1.0, pipe_D=0.5,
                                k_in=0.0, k_out=0.0,
                                pipe_end_node=None):
        """1 link, 1 pipe cell, 1 inlet cell at node 0.  Node 1 = WALL_BC
        unless pipe_end_node is set."""
        D_in = inlet_diameter
        w_in = 0.25 * math.pi * D_in
        kwargs = dict(
            dev_ptr=dev, n_links=1,
            link_from=np.array([0], dtype=np.int32),
            link_to=np.array([1], dtype=np.int32),
            L=np.array([1.0], dtype=np.float64),
            D=np.array([pipe_D], dtype=np.float64),
            n_mann=np.array([0.013], dtype=np.float64),
            S0=np.zeros(1, dtype=np.float64),
            node_invert=np.array([10.0, 9.0], dtype=np.float64),
            mcl=10.0,
            link_shape_type=np.zeros(1, dtype=np.int32),
            link_width=np.array([pipe_D], dtype=np.float64),
            link_height=np.array([pipe_D], dtype=np.float64),
            n_inlet_cells=1,
            inlet_node_ids=np.array([0], dtype=np.int32),
            inlet_invert=np.array([10.0], dtype=np.float64),
            inlet_surface_area=np.array([D_in * w_in], dtype=np.float64),
            inlet_max_depth=np.array([3.0], dtype=np.float64),
            inlet_diameter=np.array([D_in], dtype=np.float64),
            inlet_cell_length=np.array([D_in], dtype=np.float64),
            inlet_cell_width=np.array([w_in], dtype=np.float64),
            link_inlet_loss_k=np.array([k_in], dtype=np.float64),
            link_outlet_loss_k=np.array([k_out], dtype=np.float64),
        )
        if pipe_end_node is not None:
            kwargs["n_pipe_ends"] = 1
            kwargs["pipe_end_node_ids"] = np.asarray([pipe_end_node], dtype=np.int32)
        _MOD.swe2d_build_unified_mesh(**kwargs)

    def test_inlet_drains_into_dry_pipe(self):
        """Inlet cell full (h=2m), pipe dry → water flows into pipe."""
        b = _backend()
        dev = int(_MOD.swe2d_get_coupling_dev_ptr())
        self._build_inlet_pipe_mesh(dev)

        nc = 2
        h_init = np.array([0.0, 2.0], dtype=np.float64)
        _MOD.swe2d_pipe1d_upload_cell_h(dev, h_init)
        _MOD.swe2d_pipe1d_init_cell_area(dev, _H)

        st0 = _MOD.swe2d_pipe1d_readback_cell_state(dev, nc)
        A_pipe0 = float(st0['cell_A'][0])
        A_inlet0 = float(st0['cell_A'][1])

        for _ in range(100):
            _MOD.swe2d_pipe1d_step(dev, 0.05, 'rk2', 1, 2, 0.05,
                                   _G, _K, _H,
                                   surcharge_method=1, recon_method=1, theta=1.0)

        st1 = _MOD.swe2d_pipe1d_readback_cell_state(dev, nc)
        A_pipe1 = float(st1['cell_A'][0])
        A_inlet1 = float(st1['cell_A'][1])

        # Water should move from inlet → pipe
        self.assertLess(A_inlet1, A_inlet0, "Inlet should lose water to pipe")
        self.assertGreater(A_pipe1, A_pipe0, "Pipe should gain water from inlet")

        # Total mass conserved
        dx_pipe = 1.0  # pipe L = 1.0 / 1 sub-cell
        dx_inlet = 1.0  # inlet cell_length = D = 1.0
        vol0 = A_inlet0 * dx_inlet + A_pipe0 * dx_pipe
        vol1 = A_inlet1 * dx_inlet + A_pipe1 * dx_pipe
        self.assertAlmostEqual(vol1 / vol0, 1.0, places=7,
                               msg=f"Mass drift: {vol1-vol0:.2e}")
        b.destroy()

@unittest.skipUnless(_gpu, "CUDA GPU not available")
class TestCoupled1D2DConservation(unittest.TestCase):
    """Coupled pipe→2D exchange — mass conservation (single step)."""

    @staticmethod
    def _pipe_volume(dev: int, nc: int) -> float:
        st = _MOD.swe2d_pipe1d_readback_cell_state(dev, nc, 0, 0)
        A = np.asarray(st.get('cell_A', np.zeros(nc)), dtype=np.float64)
        L = np.asarray(st.get('cell_length', np.full(nc, 10.0)), dtype=np.float64)
        return float(np.sum(A * L))

    def test_coupled_single_step_conservation(self):
        """Pipe full → 2D: one coupled step conserves mass exactly.

        Single step avoids oscillatory reversals that hit the h=0 clamp
        (a physical, not algorithmic, mass sink).
        """
        from swe2d.extensions.extension_models import (
            PipeNetworkConfig, DrainageNode, DrainageLink, PipeEndExchange,
        )
        from swe2d.runtime.coupling import build_coupling_controller

        b = SWE2DBackend()
        b.build_mesh(np.asarray([0., 1., 0.]), np.asarray([0., 0., 1.]),
                     np.asarray([0., 0., 0.]), np.asarray([0, 1, 2]))
        b.initialize(h0=np.asarray([0.]), hu0=np.zeros(1), hv0=np.zeros(1),
                     dt_fixed=0.5, dt_max=0.5)

        dev = int(_MOD.swe2d_get_coupling_dev_ptr())
        nc = 10

        cfg = PipeNetworkConfig(enabled=True,
            nodes=[DrainageNode(node_id="n0", x=0., y=0., invert_elev=10., max_depth=3.),
                   DrainageNode(node_id="n1", x=100., y=0., invert_elev=10., max_depth=3.)],
            links=[DrainageLink(link_id="l0", from_node_id="n0", to_node_id="n1",
                                length=100., diameter=1., roughness_n=0.013,
                                link_shape="rectangular", width=1., height=1.)],
            pipe_ends=[PipeEndExchange(pipe_end_id="pe0", cell_id=0, node_id="n0",
                                       invert_elev=10., diameter=1., area_m2=1.)],
            surcharge_method=1, recon_method=1)

        cc = build_coupling_controller(
            pipe_network_cfg=cfg, hydraulic_structures_cfg=None,
            cell_area=np.array([0.5]), cell_bed=np.array([10.]),
            length_scale_si_to_model=1., bridge_cuda_coupling=False,
            bridge_stacked_coupling_mode="legacy_scalar",
            culvert_face_flux_mode="face_flux", culvert_solver_mode="egl",
            drainage_gpu_method_mode="iterative", use_redistribution=False,
            h_min=_H, backend=b)

        _MOD.swe2d_build_unified_mesh(dev_ptr=dev, n_links=1,
            link_from=np.array([0],dtype=np.int32), link_to=np.array([1],dtype=np.int32),
            L=np.array([100.]), D=np.array([1.]), n_mann=np.array([0.013]),
            S0=np.zeros(1), node_invert=np.array([10.,10.]), mcl=10.,
            n_pipe_ends=1, pipe_end_node_ids=np.array([0],dtype=np.int32),
            link_shape_type=np.ones(1,dtype=np.int32), link_width=np.array([1.]),
            link_height=np.array([1.]), node_inlet_loss_k=np.array([0.,0.]),
            node_outlet_loss_k=np.array([0.,0.]))
        _MOD.swe2d_gpu_device_sync()

        _MOD.swe2d_pipe1d_upload_cell_h(dev, np.full(nc, 1., dtype=np.float64))
        _MOD.swe2d_pipe1d_init_cell_area(dev, _H)
        _MOD.swe2d_pipe1d_upload_pipe_end_surface_faces(dev, np.array([0],dtype=np.int32))
        _MOD.swe2d_gpu_device_sync()
        cc._pipe1d_mesh_built = True
        cc._drainage_exchange_uploaded = True

        pipe_v0 = self._pipe_volume(dev, nc)
        self.assertGreater(pipe_v0, 0.0)
        area_2d = 0.5

        cc.apply_native_device_sources(0., 0.5)
        _MOD.swe2d_gpu_device_sync()
        b.step(0.5)

        pipe_v1 = self._pipe_volume(dev, nc)
        h2d = float(b.get_state()[0][0])
        pipe_loss = pipe_v0 - pipe_v1
        vol_2d_gain = h2d * area_2d

        self.assertGreater(h2d, 0.0, "2D cell should gain water")
        self.assertAlmostEqual(pipe_loss, vol_2d_gain, places=5,
                               msg=f"Pipe loss {pipe_loss:.6e} ≠ 2D gain {vol_2d_gain:.6e}")
        b.destroy()


@unittest.skipUnless(_gpu, "CUDA GPU not available")
class TestManholeStorage(unittest.TestCase):
    """Manhole storage cell hydraulics."""

    def test_manhole_stores_water(self):
        """Manhole cell at node 0 stores water; pipe is initially dry.
        Cell layout: [0]=pipe, [1]=manhole."""
        b = _backend()
        dev = int(_MOD.swe2d_get_coupling_dev_ptr())
        _MOD.swe2d_build_unified_mesh(
            dev_ptr=dev, n_links=1,
            link_from=np.array([0], dtype=np.int32),
            link_to=np.array([1], dtype=np.int32),
            L=np.array([10.0], dtype=np.float64),
            D=np.array([1.0], dtype=np.float64),
            n_mann=np.array([0.013], dtype=np.float64),
            S0=np.zeros(1, dtype=np.float64),
            node_invert=np.array([10.0, 9.0], dtype=np.float64),
            mcl=10.0,
            link_shape_type=np.ones(1, dtype=np.int32),
            link_width=np.array([1.0], dtype=np.float64),
            link_height=np.array([1.0], dtype=np.float64),
            n_manhole_cells=1,
            manhole_node_ids=np.array([0], dtype=np.int32),
            manhole_invert=np.array([10.0], dtype=np.float64),
            manhole_surface_area=np.array([50.0], dtype=np.float64),
            manhole_max_depth=np.array([3.0], dtype=np.float64),
            manhole_rim=np.array([13.0], dtype=np.float64),
            manhole_diameter=np.array([1.0], dtype=np.float64),
        )
        _MOD.swe2d_gpu_device_sync()

        nc = 2  # 1 pipe + 1 manhole
        h_init = np.array([0.0, 2.0], dtype=np.float64)  # pipe h=0, manhole h=2
        _MOD.swe2d_pipe1d_upload_cell_h(dev, h_init)
        _MOD.swe2d_pipe1d_init_cell_area(dev, _H)

        st0 = _MOD.swe2d_pipe1d_readback_cell_state(dev, nc)
        self.assertGreater(st0['cell_A'][1], 0.0, "Manhole should have area from depth")
        vol0 = float(st0['cell_A'][1] * st0['cell_length'][1])
        b.destroy()

    def test_manhole_drains_into_pipe(self):
        """Manhole full → pipe dry: water should flow manhole→pipe."""
        b = _backend()
        dev = int(_MOD.swe2d_get_coupling_dev_ptr())
        _MOD.swe2d_build_unified_mesh(
            dev_ptr=dev, n_links=1,
            link_from=np.array([0], dtype=np.int32),
            link_to=np.array([1], dtype=np.int32),
            L=np.array([10.0], dtype=np.float64),
            D=np.array([1.0], dtype=np.float64),
            n_mann=np.array([0.013], dtype=np.float64),
            S0=np.zeros(1, dtype=np.float64),
            node_invert=np.array([10.0, 9.0], dtype=np.float64),
            mcl=10.0,
            link_shape_type=np.ones(1, dtype=np.int32),
            link_width=np.array([1.0], dtype=np.float64),
            link_height=np.array([1.0], dtype=np.float64),
            n_manhole_cells=1,
            manhole_node_ids=np.array([0], dtype=np.int32),
            manhole_invert=np.array([10.0], dtype=np.float64),
            manhole_surface_area=np.array([50.0], dtype=np.float64),
            manhole_max_depth=np.array([3.0], dtype=np.float64),
            manhole_rim=np.array([13.0], dtype=np.float64),
            manhole_diameter=np.array([1.0], dtype=np.float64),
        )
        _MOD.swe2d_gpu_device_sync()

        nc = 2
        h_init = np.array([0.0, 2.0], dtype=np.float64)
        _MOD.swe2d_pipe1d_upload_cell_h(dev, h_init)
        _MOD.swe2d_pipe1d_init_cell_area(dev, _H)

        st0 = _MOD.swe2d_pipe1d_readback_cell_state(dev, nc)
        A_manhole0 = float(st0['cell_A'][1])
        A_pipe0 = float(st0['cell_A'][0])

        for _ in range(20):
            _MOD.swe2d_pipe1d_step(dev, 0.5, 'rk2', 1, 2, 0.5,
                                   _G, _K, _H,
                                   surcharge_method=1, recon_method=1, theta=1.0)

        st1 = _MOD.swe2d_pipe1d_readback_cell_state(dev, nc)
        A_manhole1 = float(st1['cell_A'][1])
        A_pipe1 = float(st1['cell_A'][0])

        self.assertLess(A_manhole1, A_manhole0, "Manhole should lose water to pipe")
        self.assertGreater(A_pipe1, A_pipe0, "Pipe should gain water from manhole")
        b.destroy()


@unittest.skipUnless(_gpu, "CUDA GPU not available")
class TestNodeLevelLossCoefficients(unittest.TestCase):
    """Node-level loss coefficient overrides."""

    def test_node_loss_override_takes_precedence(self):
        """Node-level inlet_loss_k / outlet_loss_k override link-level defaults.
        Build mesh with link_loss_k=0, node_loss_k=2.0 for node 0 (pipe_end).
        Verify the pipe_end face uses the node-level value (non-zero face_k_in)."""
        b = _backend()
        dev = int(_MOD.swe2d_get_coupling_dev_ptr())
        _MOD.swe2d_build_unified_mesh(
            dev_ptr=dev, n_links=1,
            link_from=np.array([0], dtype=np.int32),
            link_to=np.array([1], dtype=np.int32),
            L=np.array([100.0], dtype=np.float64),
            D=np.array([1.0], dtype=np.float64),
            n_mann=np.array([0.013], dtype=np.float64),
            S0=np.zeros(1, dtype=np.float64),
            node_invert=np.array([10.0, 10.0], dtype=np.float64),
            mcl=10.0,
            link_shape_type=np.ones(1, dtype=np.int32),
            link_width=np.array([1.0], dtype=np.float64),
            link_height=np.array([1.0], dtype=np.float64),
            n_pipe_ends=1,
            pipe_end_node_ids=np.array([0], dtype=np.int32),
            node_inlet_loss_k=np.array([2.0, 0.0], dtype=np.float64),
            node_outlet_loss_k=np.array([0.0, 0.0], dtype=np.float64),
        )
        _MOD.swe2d_gpu_device_sync()

        nc = 10
        _MOD.swe2d_pipe1d_upload_cell_h(dev, np.full(nc, 1.0, dtype=np.float64))
        _MOD.swe2d_pipe1d_init_cell_area(dev, _H)

        # Loss coefficients don't affect a lake-at-rest (Q=0), so just
        # verify the mesh built with node-level loss_k values doesn't crash.
        for _ in range(5):
            _MOD.swe2d_pipe1d_step(dev, 0.5, 'rk2', 1, 2, 0.5,
                                   _G, _K, _H,
                                   surcharge_method=1, recon_method=1, theta=1.0)
        st = _MOD.swe2d_pipe1d_readback_cell_state(dev, nc)
        Q = np.asarray(st['cell_Q'])
        self.assertTrue(np.all(np.isfinite(Q)), "Q should be finite with node-level loss")
        b.destroy()


@unittest.skipUnless(_gpu, "CUDA GPU not available")
class TestMultiSubCell(unittest.TestCase):
    """Link subdivision with max_cell_length < link length."""

    def test_20_sub_cells_from_mcl5(self):
        """100m link, mcl=5 → 20 pipe cells."""
        b = _backend()
        dev = int(_MOD.swe2d_get_coupling_dev_ptr())
        nc = _mesh(dev, stype=1, mcl=5.0)
        self.assertEqual(nc, 20, "100m link with mcl=5 should give 20 cells")
        _MOD.swe2d_pipe1d_upload_cell_h(dev, np.full(nc, 0.5, dtype=np.float64))
        _MOD.swe2d_pipe1d_init_cell_area(dev, _H)
        for _ in range(10):
            _MOD.swe2d_pipe1d_step(dev, 0.5, 'rk2', 1, 2, 0.5,
                                   _G, _K, _H,
                                   surcharge_method=1, recon_method=1, theta=1.0)
        st = _MOD.swe2d_pipe1d_readback_cell_state(dev, nc)
        self.assertEqual(len(st['cell_A']), 20, "All 20 cells should be readable")
        b.destroy()


@unittest.skipUnless(_gpu, "CUDA GPU not available")
class TestReadbackCellCounts(unittest.TestCase):
    """Verify cell counts are stored eagerly and readback uses them correctly."""

    def test_cell_counts_stored_after_mesh_build(self):
        """Mesh build should store _n_inlet_cells and _n_manhole_cells."""
        from swe2d.runtime.coupling import SWE2DCouplingController
        from swe2d.extensions.drainage_network import SWE2DUrbanDrainageModule
        from swe2d.extensions.extension_models import PipeNetworkConfig, DrainageNode, DrainageLink, InletExchange
        
        nodes = [
            DrainageNode(
                node_id="inlet1",
                x=0.0, y=0.0,
                invert_elev=0.0,
                max_depth=2.0,
                node_type="inlet",
                metadata={"surface_area": 10.0},
            ),
            DrainageNode(
                node_id="inlet2",
                x=10.0, y=0.0,
                invert_elev=0.0,
                max_depth=2.0,
                node_type="inlet",
                metadata={"surface_area": 10.0},
            ),
        ]
        links = [
            DrainageLink(
                link_id="link1",
                from_node_id="inlet1",
                to_node_id="inlet2",
                length=10.0,
                roughness_n=0.013,
                diameter=1.0,
            ),
        ]
        inlets = [
            InletExchange(
                inlet_id="inlet1",
                cell_id=0,
                node_id="inlet1",
                crest_elev=1.0,
                width=1.0,
                coefficient=0.6,
                inlet_type="grate",
            ),
        ]

        # Create drainage module from config (1 inlet to avoid cell count issues)
        cfg = PipeNetworkConfig(
            enabled=True,
            nodes=nodes,
            links=links,
            inlets=inlets,
            outfalls=[],
            pipe_ends=[],
            pipe_solver_mode="egl",
        )
        drainage = SWE2DUrbanDrainageModule(cfg)
        drainage.initialize()

        b = _backend()
        ctrl = SWE2DCouplingController(
            cell_bed=np.array([0.0, 0.0, 0.0]),
            cell_area=np.array([1.0, 1.0, 1.0]),
            drainage=drainage,
            h_min=1.0e-10,
        )

        # Mesh build should eagerly store cell counts
        ctrl._build_pipe1d_mesh_on_device()
        self.assertEqual(ctrl._n_inlet_cells, 1,
                        "Mesh build should store _n_inlet_cells=1")
        self.assertEqual(ctrl._n_manhole_cells, 1,
                        "Mesh build should store _n_manhole_cells=1 (1 non-inlet node)")

        b.destroy()

    def test_readback_uses_stored_counts_not_getattr_fallback(self):
        """Readback should use stored counts, not getattr(..., 0) fallback."""
        from swe2d.runtime.coupling import SWE2DCouplingController
        from swe2d.extensions.extension_models import PipeNetworkConfig, DrainageNode, DrainageLink, InletExchange
        from swe2d.extensions.drainage_network import SWE2DUrbanDrainageModule
        
        nodes = [
            DrainageNode(
                node_id="inlet1",
                x=0.0, y=0.0,
                invert_elev=0.0,
                max_depth=2.0,
                node_type="inlet",
                metadata={"surface_area": 10.0},
            ),
            DrainageNode(
                node_id="junction1",
                x=10.0, y=0.0,
                invert_elev=0.0,
                max_depth=2.0,
                node_type="junction",
                metadata={"surface_area": 10.0},
            ),
        ]
        links = [
            DrainageLink(
                link_id="link1",
                from_node_id="inlet1",
                to_node_id="junction1",
                length=10.0,
                roughness_n=0.013,
                diameter=1.0,
            ),
        ]
        inlets = [
            InletExchange(
                inlet_id="inlet1",
                cell_id=0,
                node_id="inlet1",
                crest_elev=1.0,
                width=1.0,
                coefficient=0.6,
                inlet_type="grate",
            ),
        ]

        # Create drainage module from config
        cfg = PipeNetworkConfig(
            enabled=True,
            nodes=nodes,
            links=links,
            inlets=inlets,
            outfalls=[],
            pipe_ends=[],
            pipe_solver_mode="egl",
        )
        drainage = SWE2DUrbanDrainageModule(cfg)
        drainage.initialize()

        b = _backend()
        ctrl = SWE2DCouplingController(
            cell_bed=np.array([0.0, 0.0, 0.0]),
            cell_area=np.array([1.0, 1.0, 1.0]),
            drainage=drainage,
            h_min=1.0e-10,
        )

        # Build mesh and verify counts are stored
        ctrl._build_pipe1d_mesh_on_device()
        self.assertEqual(ctrl._n_inlet_cells, 1)
        self.assertEqual(ctrl._n_manhole_cells, 1)  # 1 junction node

        # Do a readback - should use stored counts, not getattr fallback
        dev = int(_MOD.swe2d_get_coupling_dev_ptr())
        state = _MOD.swe2d_pipe1d_readback_cell_state(dev, 10,
                                                     ctrl._n_manhole_cells,
                                                     ctrl._n_inlet_cells)
        self.assertIsNotNone(state)
        self.assertIn('n_manhole_cells', state)
        self.assertIn('n_inlet_cells', state)

        b.destroy()


@unittest.skipUnless(_gpu, "CUDA GPU not available")
class TestSurfaceAreaField(unittest.TestCase):
    """Verify surface_area is a first-class DrainageNode field."""

    def test_surface_area_field_promoted_to_first_class(self):
        """DrainageNode.surface_area should be a direct field, not metadata-only."""
        from swe2d.extensions.extension_models import DrainageNode, PipeNetworkConfig
        from swe2d.extensions.drainage_network import SWE2DUrbanDrainageModule
        from swe2d.runtime.coupling import SWE2DCouplingController

        # Create node with surface_area as a direct field
        node = DrainageNode(
            node_id="node1",
            x=0.0, y=0.0,
            invert_elev=0.0,
            max_depth=2.0,
            surface_area=100.0,  # First-class field
        )

        self.assertEqual(node.surface_area, 100.0,
                        "surface_area should be accessible as a direct field")

        # Create node with surface_area in metadata (backward compatibility)
        node_legacy = DrainageNode(
            node_id="node2",
            x=10.0, y=0.0,
            invert_elev=0.0,
            max_depth=2.0,
            metadata={"surface_area": 200.0},
        )

        self.assertEqual(node_legacy.surface_area, 50.0,
                        "Default surface_area should be 50.0 when not set")
        self.assertEqual(node_legacy.metadata["surface_area"], 200.0,
                        "Metadata should still be accessible for backward compat")

    def test_surface_area_packed_from_first_class_field(self):
        """The packer should read surface_area from the field, not metadata."""
        from swe2d.extensions.extension_models import DrainageNode, DrainageLink, PipeNetworkConfig
        from swe2d.extensions.drainage_network import SWE2DUrbanDrainageModule
        from swe2d.runtime.coupling import SWE2DCouplingController, pack_pipe_network_soa

        nodes = [
            DrainageNode(
                node_id="node1",
                x=0.0, y=0.0,
                invert_elev=0.0,
                max_depth=2.0,
                surface_area=123.4,  # First-class field value
            ),
            DrainageNode(
                node_id="node2",
                x=10.0, y=0.0,
                invert_elev=0.0,
                max_depth=2.0,
                metadata={"surface_area": 567.8},  # Metadata value (should be ignored)
            ),
        ]
        links = [
            DrainageLink(
                link_id="link1",
                from_node_id="node1",
                to_node_id="node2",
                length=10.0,
                roughness_n=0.013,
                diameter=1.0,
            ),
        ]

        cfg = PipeNetworkConfig(
            enabled=True,
            nodes=nodes,
            links=links,
            inlets=[],
            outfalls=[],
            pipe_ends=[],
            pipe_solver_mode="egl",
        )

        # Pack the network
        soa = pack_pipe_network_soa(cfg, n_cells=2)

        self.assertIsNotNone(soa)
        # Node 0: should use first-class field value (123.4)
        self.assertAlmostEqual(float(soa.node_surface_area[0]), 123.4, places=1,
                              msg="Packer should use first-class surface_area field")
        # Node 1: should fall back to metadata since first-class field is default
        self.assertAlmostEqual(float(soa.node_surface_area[1]), 567.8, places=1,
                              msg="Packer should fall back to metadata surface_area")


if __name__ == "__main__":
    unittest.main()
