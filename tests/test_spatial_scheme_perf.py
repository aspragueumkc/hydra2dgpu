"""
Performance benchmarks for spatial schemes.

Measures wall-clock time for each spatial reconstruction scheme on a
~500-cell gmsh triangle mesh.  Uses the same low-level hydra_swe2d API
as ``test_swe2d_weno5_convergence.py`` (swe2d_create_solver / swe2d_step
/ swe2d_get_state / swe2d_destroy).

Schemes benchmarked
-------------------
  * 0 — FV_FIRST_ORDER          (baseline)
  * 5 — FV_BARTH_JESPERSEN      (LSQ gradient + Barth-Jespersen limiter)
  * 6 — FV_WENO3                (true 3-sub-stencil WENO, 1-ring)
  * 8 — FV_MP5                  (Suresh-Huynh Monotonicity-Preserving)
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from swe2d.extensions.extension_models import SpatialDiscretization

BENCHMARK_SCHEMES = [0, 5, 6, 8]
SCHEME_NAMES: dict[int, str] = {
    SpatialDiscretization.FV_FIRST_ORDER: "First-order",
    SpatialDiscretization.FV_BARTH_JESPERSEN: "Barth-Jespersen",
    SpatialDiscretization.FV_WENO3: "WENO3",
    SpatialDiscretization.FV_MP5: "MP5",
}


def _load_module():
    """Import and return the hydra_swe2d native module, or None."""
    try:
        import hydra_swe2d  # noqa: F401
        return hydra_swe2d
    except ImportError:
        return None


def _gpu_available() -> bool:
    mod = _load_module()
    if mod is None:
        return False
    try:
        return bool(mod.swe2d_gpu_available())
    except Exception:
        return False


def _gmsh_available() -> bool:
    try:
        import gmsh  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(_load_module() is None, reason="hydra_swe2d not built")
@pytest.mark.skipif(not _gpu_available(), reason="CUDA GPU not available")
@pytest.mark.skipif(not _gmsh_available(), reason="gmsh not installed")
class TestSpatialSchemePerformance:
    """Run each spatial scheme on a ~500-cell mesh and verify timing."""

    # Domain / solver parameters (matching convergence test conventions)
    LX, LY = 50.0, 50.0
    N_STEPS = 500
    CFL = 0.40
    DT_MAX = 0.05

    # ── class-scoped mesh (built once, shared by all scheme parametrisations) ─

    @pytest.fixture(scope="class")
    def mesh_data(self):
        """Build a ~500-cell gmsh triangle mesh.

        Returns (node_x, node_y, node_z, cell_nodes, cell_cx, cell_cy)
        exactly as ``_make_gmsh_triangle_mesh`` does.
        """
        import gmsh

        gmsh.initialize()
        gmsh.model.add("scheme_perf")
        gmsh.model.occ.add_rectangle(0, 0, 0, self.LX, self.LY)
        gmsh.model.occ.synchronize()

        # Target ~500 triangles: mesh_size = LX / 16 ≈ 3.125 → ~500 cells
        mesh_size = self.LX / 16.0
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size * 0.5)
        gmsh.model.mesh.generate(2)

        # Extract nodes
        _, node_coords, _ = gmsh.model.mesh.getNodes()
        node_xyz = node_coords.reshape(-1, 3)
        node_x = node_xyz[:, 0].copy()
        node_y = node_xyz[:, 1].copy()
        node_z = np.zeros_like(node_x)

        # Extract triangle elements (type 2 = 3-node triangle)
        elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
        tri_type = 2
        cell_nodes_list = []
        for etype, nodes in zip(elem_types, elem_node_tags):
            if etype == tri_type:
                cell_nodes_list.append(nodes.reshape(-1, 3))
        if not cell_nodes_list:
            gmsh.finalize()
            pytest.fail("No triangle elements found in gmsh mesh")
        cell_nodes = np.vstack(cell_nodes_list).astype(np.int32) - 1  # 1→0-based

        gmsh.finalize()

        # Cell centroids
        cell_cx = np.mean(node_x[cell_nodes], axis=1)
        cell_cy = np.mean(node_y[cell_nodes], axis=1)

        return node_x, node_y, node_z, cell_nodes, cell_cx, cell_cy

    # ── parametrised benchmark ───────────────────────────────────────────────

    @pytest.mark.parametrize("scheme", BENCHMARK_SCHEMES)
    def test_scheme_performance(self, scheme: int, mesh_data):
        """Benchmark one spatial scheme: time N_STEPS, assert < 120 s."""
        mod = _load_module()
        assert mod is not None  # decorator already skipped if missing

        node_x, node_y, node_z, cell_nodes, cell_cx, cell_cy = mesh_data

        # --- Build mesh (no boundary conditions -- all interior) ---
        mesh = mod.swe2d_build_mesh(
            node_x, node_y, node_z, cell_nodes,
            np.empty(0, dtype=np.int32),   # bc_n0
            np.empty(0, dtype=np.int32),   # bc_n1
            np.empty(0, dtype=np.int32),   # bc_tp
            np.empty(0, dtype=np.float64), # bc_vl
        )
        info = mod.swe2d_mesh_info(mesh)
        n_cells = info["n_cells"]

        # --- Initial condition: flat quiescent water ---
        h0 = np.full(n_cells, 0.5, dtype=np.float64)
        hu0 = np.zeros(n_cells, dtype=np.float64)
        hv0 = np.zeros(n_cells, dtype=np.float64)

        solver = mod.swe2d_create_solver(
            mesh, h0, hu0, hv0,
            n_mann=0.0,
            cfl=self.CFL,
            dt_max=self.DT_MAX,
            temporal_order=2,      # SSP-RK2
            spatial_scheme=scheme,
            use_gpu=True,
        )

        # --- Warm-up: run solver for a short time ---
        # This ensures GPU kernels are JIT-compiled and caches are filled
        # before we start the stopwatch.
        t = 0.0
        while t < self.DT_MAX * 10.0:
            diag = mod.swe2d_step(solver, -1.0)
            t += diag["dt"]

        # --- Timed segment: N_STEPS ---
        t0 = time.perf_counter()
        for _ in range(self.N_STEPS):
            mod.swe2d_step(solver, -1.0)
        elapsed = time.perf_counter() - t0  # seconds

        # --- Read back final state (sanity: must be finite) ---
        h, hu, hv = mod.swe2d_get_state(solver)

        # --- Cleanup ---
        mod.swe2d_destroy(solver)

        # --- Per-step timing ---
        per_step_ms = elapsed / self.N_STEPS * 1000.0
        cells_per_s = n_cells * (1.0 / per_step_ms) * 1000.0

        name = SCHEME_NAMES.get(scheme, f"Scheme-{scheme}")

        print(f"\n  {name} (scheme={scheme})")
        print(f"    cells:        {n_cells}")
        print(f"    total time:   {elapsed:.3f} s")
        print(f"    per step:     {per_step_ms:.2f} ms")
        print(f"    throughput:   {cells_per_s:.0f} cells/s")

        # Sanity: solution must be finite
        assert np.isfinite(h).all(), f"{name}: non-finite depth produced"
        assert np.isfinite(hu).all(), f"{name}: non-finite x-momentum"
        assert np.isfinite(hv).all(), f"{name}: non-finite y-momentum"

        # Hard upper bound: any scheme must finish in reasonable time
        assert elapsed < 120.0, (
            f"{name} took {elapsed:.1f}s (limit 120 s)"
        )
