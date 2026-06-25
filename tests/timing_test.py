"""Timing test: ~100K cell mesh with rain.

Measures time spent on the GPU step with rain active.
Run with:
    python -m pytest tests/timing_test.py -v -s
"""
from __future__ import annotations

import time
import unittest

import numpy as np

from swe2d.runtime.backend import SWE2DBackend, swe2d_available
from tests._swe2d_test_helpers import _make_gmsh_triangle_mesh, _make_rect_mesh


def _gpu_available():
    try:
        import hydra_swe2d as m
        return bool(m.swe2d_gpu_available())
    except Exception:
        return False


def _sloping_zb(node_x, node_y):
    return 10.0 - 0.005 * node_x - 0.003 * node_y


@unittest.skipUnless(swe2d_available(), "hydra_swe2d not built")
@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
class TimingTest(unittest.TestCase):
    """Report per-step timing for rain-on-grid on a ~100K cell mesh."""

    NX = 224
    NY = 224
    LX = 2240.0
    LY = 2240.0

    def _gmsh_zb(self, x, y):
        return 10.0 - 0.005 * x - 0.003 * y

    def test_timing_gmsh(self):
        try:
            import gmsh as _g
            _g.initialize()
            _g.finalize()
        except Exception:
            self.skipTest("gmsh not available")
        mod = None
        try:
            import hydra_swe2d as m
            mod = m
        except Exception:
            self.skipTest("hydra_swe2d import failed")

        node_x, node_y, node_z, cell_nodes, _, _ = _make_gmsh_triangle_mesh(
            2000.0, 2000.0, 8.0, zb_func=self._gmsh_zb,
        )

        ncells = int(cell_nodes.size // 3)
        print(f"\n=== GMSH Timing: ~{ncells} cells ===")

        backend = SWE2DBackend()
        backend.build_mesh(
            node_x, node_y, node_z, cell_nodes,
            bc_edge_node0=np.empty(0, dtype=np.int32),
            bc_edge_node1=np.empty(0, dtype=np.int32),
            bc_edge_type=np.empty(0, dtype=np.int32),
            bc_edge_val=np.empty(0, dtype=np.float64),
        )
        ncells = backend.n_cells

        backend.initialize(
            h0=np.full(ncells, 0.05, dtype=np.float64),
            n_mann=0.035, h_min=1e-4, cfl=0.45, dt_max=0.5,
            gpu_diag_sync_interval_steps=1,
        )

        from swe2d.runtime.runtime_setup_configurator import SWE2DRunSetupConfigurator
        cfg = SWE2DRunSetupConfigurator()
        cfg.configure_constant_rain_rate_native(
            backend=backend,
            rate_model_mps=2.0 / 1000.0 / 3600.0 * 0.0254,
            mm_to_model_depth=1e-3,
        )

        for _ in range(11):
            backend.step(-1.0)
        for _ in range(3):
            backend.step(-1.0)
        t0 = time.perf_counter()
        for _ in range(30):
            backend.step(-1.0)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        per_step = elapsed_ms / 30.0

        print(f"  30 steps: {elapsed_ms:.0f} ms total")
        print(f"  1 step:   {per_step:.1f} ms avg")
        print(f"  steps/s:  {1000.0 / per_step:.0f}")
        print(f"  cells/s:  {ncells * 1000.0 / per_step:.0f}")
        print(f"  dt={backend._last_diag.get('dt',0):.5f}s  wet={backend._last_diag.get('wet_cells',0)}")
        backend.destroy()

    def test_timing_gmsh_large(self):
        try:
            import gmsh as _g
            _g.initialize()
            _g.finalize()
        except Exception:
            self.skipTest("gmsh not available")
        mod = None
        try:
            import hydra_swe2d as m
            mod = m
        except Exception:
            self.skipTest("hydra_swe2d import failed")

        node_x, node_y, node_z, cell_nodes, _, _ = _make_gmsh_triangle_mesh(
            4000.0, 4000.0, 4.0, zb_func=self._gmsh_zb,
        )

        ncells = int(cell_nodes.size // 3)
        print(f"\n=== GMSH Large Timing: ~{ncells} cells ===")

        backend = SWE2DBackend()
        backend.build_mesh(
            node_x, node_y, node_z, cell_nodes,
            bc_edge_node0=np.empty(0, dtype=np.int32),
            bc_edge_node1=np.empty(0, dtype=np.int32),
            bc_edge_type=np.empty(0, dtype=np.int32),
            bc_edge_val=np.empty(0, dtype=np.float64),
        )
        ncells = backend.n_cells

        backend.initialize(
            h0=np.full(ncells, 0.05, dtype=np.float64),
            n_mann=0.035, h_min=1e-4, cfl=0.45, dt_max=0.5,
            gpu_diag_sync_interval_steps=1,
        )

        from swe2d.runtime.runtime_setup_configurator import SWE2DRunSetupConfigurator
        cfg = SWE2DRunSetupConfigurator()
        cfg.configure_constant_rain_rate_native(
            backend=backend,
            rate_model_mps=2.0 / 1000.0 / 3600.0 * 0.0254,
            mm_to_model_depth=1e-3,
        )

        for _ in range(11):
            backend.step(-1.0)
        for _ in range(3):
            backend.step(-1.0)
        t0 = time.perf_counter()
        for _ in range(30):
            backend.step(-1.0)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        per_step = elapsed_ms / 30.0

        print(f"  30 steps: {elapsed_ms:.0f} ms total")
        print(f"  1 step:   {per_step:.1f} ms avg")
        print(f"  steps/s:  {1000.0 / per_step:.0f}")
        print(f"  cells/s:  {ncells * 1000.0 / per_step:.0f}")
        print(f"  dt={backend._last_diag.get('dt',0):.5f}s  wet={backend._last_diag.get('wet_cells',0)}")
        backend.destroy()

    def test_timing(self):
        mod = None
        try:
            import hydra_swe2d as m
            mod = m
        except Exception:
            self.skipTest("hydra_swe2d import failed")

        node_x, node_y, node_z, cell_nodes = _make_rect_mesh(
            self.NX, self.NY, self.LX, self.LY, zb_func=_sloping_zb,
        )

        # Build BCs: walls everywhere
        stride = self.NX + 1
        bc_n0, bc_n1, bc_tp, bc_vl = [], [], [], []
        for i in range(self.NX):
            bc_n0.append(i); bc_n1.append(i+1); bc_tp.append(1); bc_vl.append(0.0)
        for i in range(self.NX):
            n0 = stride * self.NY + i
            n1 = stride * self.NY + i + 1
            bc_n0.append(n0); bc_n1.append(n1); bc_tp.append(1); bc_vl.append(0.0)
        for j in range(self.NY):
            n0 = j * stride; n1 = (j + 1) * stride
            bc_n0.append(n0); bc_n1.append(n1); bc_tp.append(1); bc_vl.append(0.0)
        for j in range(self.NY):
            n0 = j * stride + self.NX
            n1 = (j + 1) * stride + self.NX
            bc_n0.append(n0); bc_n1.append(n1); bc_tp.append(1); bc_vl.append(0.0)

        backend = SWE2DBackend()
        backend.build_mesh(
            node_x, node_y, node_z, cell_nodes,
            bc_edge_node0=np.array(bc_n0, dtype=np.int32),
            bc_edge_node1=np.array(bc_n1, dtype=np.int32),
            bc_edge_type=np.array(bc_tp, dtype=np.int32),
            bc_edge_val=np.array(bc_vl, dtype=np.float64),
        )
        ncells = backend.n_cells
        print(f"\n=== Timing: {ncells} cells ({self.NX}x{self.NY}) ===")

        backend.initialize(
            h0=np.full(ncells, 0.05, dtype=np.float64),
            n_mann=0.035, h_min=1e-4, cfl=0.45, dt_max=0.5,
            gpu_diag_sync_interval_steps=1,
        )

        # Constant-rate native rain (~2 in/hr)
        from swe2d.runtime.runtime_setup_configurator import SWE2DRunSetupConfigurator
        cfg = SWE2DRunSetupConfigurator()
        cfg.configure_constant_rain_rate_native(
            backend=backend,
            rate_model_mps=2.0 / 1000.0 / 3600.0 * 0.0254,
            mm_to_model_depth=1e-3,
        )

        # Warmup
        for _ in range(11):
            backend.step(-1.0)

        # Timed: 30 steps
        for _ in range(3):
            backend.step(-1.0)
        t0 = time.perf_counter()
        for i in range(30):
            backend.step(-1.0)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        per_step = elapsed_ms / 30.0

        print(f"  30 steps: {elapsed_ms:.0f} ms total")
        print(f"  1 step:   {per_step:.1f} ms avg")
        print(f"  steps/s:  {1000.0 / per_step:.0f}")
        print(f"  cells/s:  {ncells * 1000.0 / per_step:.0f}")
        print(f"  dt={backend._last_diag.get('dt',0):.5f}s  wet={backend._last_diag.get('wet_cells',0)}")

        backend.destroy()
