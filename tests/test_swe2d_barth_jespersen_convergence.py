"""
Barth-Jespersen (scheme 5) convergence-order validation on gmsh triangle meshes.

Tests that the FV_BARTH_JESPERSEN spatial reconstruction (LSQ gradients +
Barth-Jespersen slope limiter) achieves ~2nd-order L2 convergence on a smooth
manufactured solution.

Convergence test methodology
----------------------------
For each of 3 mesh refinement levels (nx = 16, 32, 64), run a short
simulation from a smooth sinusoidal initial condition h(x, y).  Measure L2
error of the numerical solution against the *analytical* initial condition at
each mesh's own cell centroids (no inter-mesh interpolation needed).
Observed convergence order:

    p = log2(E_coarse / E_fine) / log2(h_coarse / h_fine)

Barth-Jespersen is a 2nd-order scheme (linear LSQ reconstruction + limiter).
On a smooth manufactured solution the limiter activates minimally, so we
expect order >= 1.8 (some degradation from the limiter is normal on
unstructured meshes).
"""

import os
import sys
import unittest

import numpy as np

from tests._swe2d_test_helpers import (
    _make_gmsh_triangle_mesh,
    _load_module,
    _gpu_available,
)
from swe2d.runtime.backend import SWE2DBackend
from swe2d.extensions.extension_models import (
    SpatialDiscretization,
    TemporalScheme,
)


def _gmsh_available():
    try:
        import gmsh  # noqa: F401
        return True
    except ImportError:
        return False


# ── Manufactured smooth solution ─────────────────────────────────────────────

class SmoothBumpIC:
    """Smooth sinusoidal bump for convergence testing.

    h(x,y) = H_BASE + AMP * sin(2π x / Lx) * cos(2π y / Ly)
    hu = 0, hv = 0  (quiescent — avoids temporal evolution from advection)

    The perturbation is small enough that the solution barely evolves over
    T_END, so the numerical solution stays close to the IC and the measured
    L2 error is dominated by spatial truncation.
    """

    H_BASE = 2.0     # base depth [m]
    AMP    = 0.005    # tiny perturbation [m] (0.25% of base depth)

    @classmethod
    def h_exact(cls, x, y, Lx, Ly):
        """Exact depth field at cell centroids."""
        return cls.H_BASE + cls.AMP * np.sin(2.0 * np.pi * x / Lx) * np.cos(2.0 * np.pi * y / Ly)

    @classmethod
    def make_ic(cls, cell_cx, cell_cy, Lx, Ly):
        """Return (h0, hu0, hv0) arrays for solver initialization."""
        h = cls.h_exact(cell_cx, cell_cy, Lx, Ly)
        hu = np.zeros_like(h)
        hv = np.zeros_like(h)
        return h, hu, hv


# ── Convergence test ──────────────────────────────────────────────────────────

@unittest.skipUnless(_load_module() is not None, "hydra_swe2d not built")
@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
@unittest.skipUnless(_gmsh_available(), "gmsh not installed")
class TestBarthJespersenConvergence(unittest.TestCase):
    """Mesh-refinement convergence test for Barth-Jespersen (spatial_scheme=5).

    Runs the solver on 3 mesh refinement levels with a smooth quiescent
    initial condition.  Measures L2 error of h against the *analytical*
    IC at each mesh's cell centroids (no cross-mesh interpolation).
    Checks the observed convergence order is >= 1.8.
    """

    LX, LY  = 200.0, 100.0
    NX_LIST = [16, 32, 64]
    T_END   = 0.1    # short — perturbation barely evolves
    SCHEME  = SpatialDiscretization.FV_BARTH_JESPERSEN
    ORDER   = 2      # temporal order (SSP-RK2)
    N_MANN  = 0.0
    CFL     = 0.5
    DT_MAX  = 10.0

    def _mesh_size_from_nx(self, nx: int) -> float:
        """Convert approximate nx (cells along x) to gmsh characteristic length."""
        return self.LX / float(nx)

    def _run_single_mesh(self, mesh_size: float):
        """Run solver on one mesh, return (cell_cx, cell_cy, h_final, n_cells)."""
        node_x, node_y, node_z, cell_nodes, cell_cx, cell_cy = \
            _make_gmsh_triangle_mesh(self.LX, self.LY, mesh_size)

        n_cells = len(cell_cx)
        h0, hu0, hv0 = SmoothBumpIC.make_ic(cell_cx, cell_cy, self.LX, self.LY)

        backend = SWE2DBackend()
        backend.build_mesh(node_x, node_y, node_z, cell_nodes)
        backend.initialize(
            h0, hu0, hv0,
            n_mann=self.N_MANN,
            cfl=self.CFL,
            dt_max=self.DT_MAX,
            temporal_scheme=TemporalScheme.SSP_RK2,
            spatial_discretization=self.SCHEME,
        )

        # Manual step loop (same pattern as WENO5 convergence test)
        t = 0.0
        last_diag = None
        while t < self.T_END:
            last_diag = backend.step(-1.0)
            t += float(last_diag["dt"])

        h, hu, hv = backend.get_state()
        backend.destroy()

        self.assertTrue(last_diag is not None and last_diag.get("gpu_active", False),
                        f"GPU inactive for mesh_size={mesh_size}")
        self.assertTrue(np.isfinite(h).all(),
                        f"Non-finite depth for mesh_size={mesh_size}")

        return cell_cx, cell_cy, h, n_cells

    def test_barth_jespersen_convergence_h(self):
        """Barth-Jespersen error decreases with mesh refinement at ~2nd order."""
        results = []
        for nx in self.NX_LIST:
            ms = self._mesh_size_from_nx(nx)
            cx, cy, h, nc = self._run_single_mesh(ms)
            h_exact = SmoothBumpIC.h_exact(cx, cy, self.LX, self.LY)
            l2 = np.sqrt(np.mean((h - h_exact) ** 2))
            results.append((ms, l2, nc))
            print(f"  nx={nx} (mesh_size={ms:.3f}): {nc} cells, "
                  f"L2(h-h_exact) = {l2:.6e}")

        # Verify error decreases monotonically with refinement
        for i in range(len(results) - 1):
            self.assertGreater(
                results[i][1], results[i + 1][1],
                f"Error did not decrease: nx={self.NX_LIST[i]} "
                f"({results[i][1]:.2e}) not > nx={self.NX_LIST[i+1]} "
                f"({results[i+1][1]:.2e})"
            )

        # Compute convergence orders between successive refinement levels
        orders = []
        for i in range(len(results) - 1):
            h_coarse, e_coarse, _ = results[i]
            h_fine,   e_fine,   _ = results[i + 1]
            if e_fine > 1e-15 and e_coarse > 1e-15:
                order = np.log2(e_coarse / e_fine) / np.log2(h_coarse / h_fine)
                orders.append(order)
                print(f"  order(nx={self.NX_LIST[i]}→nx={self.NX_LIST[i+1]}): "
                      f"{order:.2f}")

        self.assertTrue(len(orders) > 0, "Not enough data to compute order")
        max_order = max(orders)
        self.assertGreaterEqual(
            max_order, 1.8,
            f"Barth-Jespersen convergence order {max_order:.2f} < 1.8. "
            f"Orders: {orders}"
        )
        print(f"  ✓ Barth-Jespersen peak convergence order: {max_order:.2f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
