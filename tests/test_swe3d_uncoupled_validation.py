"""
Uncoupled 3D (single-phase free-surface) validation suite.

Execution intent (chronological):
1) Validate uncoupled 3D physics invariants first  (THIS FILE – Stage 1 gate)
2) Optimize performance and robustness after physics gates are green
3) Enable and validate 2D-3D coupling last

Stage-1 physics gates (always run, no env var required):
  * VoF boundedness  — vof ∈ [0,1] at all times
  * VoF conservation — total VoF sum constant when no sources/sinks
  * Rest stability   — zero-IC state stays exactly zero
  * Velocity damping — non-zero IC velocities decrease monotonically (scaffold damps)

Reference-case gates (gated behind BACKWATER_RUN_SWE3D_PHYSICS_CASES=1 for now):
  * Broad-crested weir free-surface profile
  * Culvert pressurisation sequence
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_PHYSICS_CASES = os.environ.get("BACKWATER_RUN_SWE3D_PHYSICS_CASES", "0") == "1"


def _load_module():
    try:
        import backwater_swe2d
        return backwater_swe2d
    except ImportError:
        return None


def _gpu_available():
    mod = _load_module()
    if mod is None:
        return False
    try:
        return bool(mod.swe2d_gpu_available())
    except Exception:
        return False


def _make_rect_mesh(mod, nx, ny, lx, ly):
    xs = np.linspace(0.0, lx, nx + 1)
    ys = np.linspace(0.0, ly, ny + 1)
    xg, yg = np.meshgrid(xs, ys)
    node_x = xg.ravel().copy()
    node_y = yg.ravel().copy()
    node_z = np.zeros_like(node_x)
    cells = []
    stride = nx + 1
    for j in range(ny):
        for i in range(nx):
            n00 = j * stride + i
            n10 = j * stride + i + 1
            n01 = (j + 1) * stride + i
            n11 = (j + 1) * stride + i + 1
            cells.extend([n00, n10, n11])
            cells.extend([n00, n11, n01])
    mesh = mod.swe2d_build_mesh(
        node_x,
        node_y,
        node_z,
        np.array(cells, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.float64),
    )
    return mesh


def _make_3d_solver(mod, mesh, h0):
    """Create an uncoupled single-phase 3D solver (no coupling contract needed)."""
    return mod.swe2d_create_solver(
        mesh,
        h0,
        use_gpu=True,
        temporal_order=2,
        coupling_mode=0,
        three_d_solver_model=1,
    )


def _flat_surface_vof(stats):
    """
    Build a VoF field with the bottom half of cells fully filled (vof=1),
    the top half empty (vof=0), mimicking a flat horizontal free surface.
    Returns a 1-D numpy array of length stats['n_cells'].
    The patch is nx x ny x nz (z is vertical).
    """
    nx, ny, nz = int(stats["nx"]), int(stats["ny"]), int(stats["nz"])
    n_cells = nx * ny * nz
    vof = np.zeros(n_cells, dtype=np.float64)
    z_half = nz // 2
    for iz in range(nz):
        if iz < z_half:
            lo = iz * nx * ny
            hi = lo + nx * ny
            vof[lo:hi] = 1.0
    return vof


@unittest.skipUnless(_load_module() is not None, "backwater_swe2d not built")
@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
class TestSWE3DUncoupledValidation(unittest.TestCase):
    """Stage-1 validation gates for uncoupled 3D mode."""

    def setUp(self):
        self.mod = _load_module()
        self.mesh = _make_rect_mesh(self.mod, 20, 10, 200.0, 100.0)
        n_cells = self.mod.swe2d_mesh_info(self.mesh)["n_cells"]
        self.h0 = np.full(n_cells, 1.0, dtype=np.float64)

    # ── Smoke ──────────────────────────────────────────────────────────────────

    def test_uncoupled_3d_mode_smoke_gpu(self):
        """8 steps finish without error; diagnostics are sane."""
        solver = _make_3d_solver(self.mod, self.mesh, self.h0)
        try:
            for _ in range(8):
                diag = self.mod.swe2d_step(solver, -1.0)
                self.assertTrue(diag["gpu_active"])
                self.assertGreater(diag["dt"], 0.0)

            h, hu, hv = self.mod.swe2d_get_state(solver)
            self.assertTrue(np.all(np.isfinite(h)))
            self.assertTrue(np.all(np.isfinite(hu)))
            self.assertTrue(np.all(np.isfinite(hv)))
            self.assertGreaterEqual(float(np.min(h)), -1.0e-10)
        finally:
            self.mod.swe2d_destroy(solver)

    def test_uncoupled_3d_does_not_require_interface_contract(self):
        """coupling_mode=0 must not raise for a missing contract."""
        solver = _make_3d_solver(self.mod, self.mesh, self.h0)
        try:
            _ = self.mod.swe2d_step(solver, -1.0)
        finally:
            self.mod.swe2d_destroy(solver)

    # ── Patch descriptor ───────────────────────────────────────────────────────

    def test_3d_patch_stats_descriptor_plausible(self):
        """Patch stats should expose valid positive dimensions."""
        solver = _make_3d_solver(self.mod, self.mesh, self.h0)
        try:
            stats = self.mod.swe2d_get_3d_patch_stats(solver)
            self.assertGreater(stats["nx"], 0)
            self.assertGreater(stats["ny"], 0)
            self.assertGreater(stats["nz"], 0)
            self.assertGreater(stats["dx"], 0.0)
            self.assertGreater(stats["dy"], 0.0)
            self.assertGreater(stats["dz"], 0.0)
            expected_n = stats["nx"] * stats["ny"] * stats["nz"]
            self.assertEqual(stats["n_cells"], expected_n)
        finally:
            self.mod.swe2d_destroy(solver)

    # ── VoF boundedness ────────────────────────────────────────────────────────

    def test_vof_bounds_preserved_flat_surface(self):
        """
        Physics invariant: VoF must remain in [0,1] for all time.
        IC: flat free surface (lower half of cells = 1, upper half = 0).
        Run 20 steps; check vof_min >= 0 and vof_max <= 1.
        """
        solver = _make_3d_solver(self.mod, self.mesh, self.h0)
        try:
            stats0 = self.mod.swe2d_get_3d_patch_stats(solver)
            vof_ic = _flat_surface_vof(stats0)
            self.mod.swe2d_set_3d_patch_vof(solver, vof_ic)

            for _ in range(20):
                self.mod.swe2d_step(solver, -1.0)

            stats = self.mod.swe2d_get_3d_patch_stats(solver)
            self.assertGreaterEqual(
                stats["vof_min"], -1.0e-10,
                f"VoF fell below 0: min={stats['vof_min']:.4e}")
            self.assertLessEqual(
                stats["vof_max"], 1.0 + 1.0e-10,
                f"VoF exceeded 1: max={stats['vof_max']:.4e}")
        finally:
            self.mod.swe2d_destroy(solver)

    # ── VoF conservation ───────────────────────────────────────────────────────

    def test_vof_sum_conserved_no_source(self):
        """
        Physics invariant: in the absence of in/outflow, total VoF must be
        conserved (sum changes by < 0.1 %).
        IC: flat free surface (half-filled).
        Run 50 steps.
        """
        solver = _make_3d_solver(self.mod, self.mesh, self.h0)
        try:
            stats0 = self.mod.swe2d_get_3d_patch_stats(solver)
            vof_ic = _flat_surface_vof(stats0)
            self.mod.swe2d_set_3d_patch_vof(solver, vof_ic)

            sum_0 = float(np.sum(vof_ic))
            for _ in range(50):
                self.mod.swe2d_step(solver, -1.0)

            stats = self.mod.swe2d_get_3d_patch_stats(solver)
            rel_err = abs(stats["vof_sum"] - sum_0) / max(sum_0, 1.0)
            self.assertLess(
                rel_err, 1.0e-3,
                f"VoF sum drifted: initial={sum_0:.4f} final={stats['vof_sum']:.4f} "
                f"rel_err={rel_err:.2e}")
        finally:
            self.mod.swe2d_destroy(solver)

    # ── Rest stability ─────────────────────────────────────────────────────────

    def test_zero_velocity_state_stays_zero(self):
        """
        Physics invariant (Gauss-Seidel rest): if u=v=w=p=0 initially, they
        remain exactly zero after N steps (damping from zero gives zero).
        """
        solver = _make_3d_solver(self.mod, self.mesh, self.h0)
        try:
            stats0 = self.mod.swe2d_get_3d_patch_stats(solver)
            n = int(stats0["n_cells"])
            zeros = np.zeros(n, dtype=np.float64)
            self.mod.swe2d_set_3d_patch_state(
                solver, u=zeros, v=zeros, w=zeros, p=zeros)

            for _ in range(20):
                self.mod.swe2d_step(solver, -1.0)

            stats = self.mod.swe2d_get_3d_patch_stats(solver)
            self.assertAlmostEqual(stats["u_rms"], 0.0, places=12,
                msg=f"u_rms should be zero; got {stats['u_rms']:.3e}")
            self.assertAlmostEqual(stats["v_rms"], 0.0, places=12,
                msg=f"v_rms should be zero; got {stats['v_rms']:.3e}")
            self.assertAlmostEqual(stats["w_rms"], 0.0, places=12,
                msg=f"w_rms should be zero; got {stats['w_rms']:.3e}")
            self.assertAlmostEqual(stats["p_max_abs"], 0.0, places=12,
                msg=f"p_max_abs should be zero; got {stats['p_max_abs']:.3e}")
        finally:
            self.mod.swe2d_destroy(solver)

    # ── Velocity damping ───────────────────────────────────────────────────────

    def test_velocity_damping_monotone_from_nonzero_ic(self):
        """
        Scaffold physics gate: the current 3D kernel applies damping each step.
        Starting from a non-zero uniform velocity field, u_rms must be strictly
        decreasing after each step (up to floating-point noise).
        This test will remain valid once real projection replaces the scaffold
        because real projection also enforces divergence-free velocity; an
        initial uniform velocity in a closed box will damp via viscosity/BC.
        NOTE: this test validates scaffold behaviour; it will be reviewed when
        real VoF advection + projection replace the stub.
        """
        solver = _make_3d_solver(self.mod, self.mesh, self.h0)
        try:
            stats0 = self.mod.swe2d_get_3d_patch_stats(solver)
            n = int(stats0["n_cells"])
            u_ic = np.full(n, 1.0, dtype=np.float64)   # uniform 1 m/s
            zeros = np.zeros(n, dtype=np.float64)
            self.mod.swe2d_set_3d_patch_state(
                solver, u=u_ic, v=zeros, w=zeros, p=zeros)

            prev_rms = None
            for step in range(10):
                self.mod.swe2d_step(solver, 0.1)  # fixed dt so damping is predictable
                stats = self.mod.swe2d_get_3d_patch_stats(solver)
                cur_rms = stats["u_rms"]
                self.assertTrue(
                    np.isfinite(cur_rms),
                    f"u_rms is not finite at step {step}")
                if prev_rms is not None:
                    self.assertLess(
                        cur_rms, prev_rms,
                        f"u_rms did not decrease at step {step}: "
                        f"{prev_rms:.6e} → {cur_rms:.6e}")
                prev_rms = cur_rms
        finally:
            self.mod.swe2d_destroy(solver)

    def test_projection_reduces_divergence_from_divergent_ic(self):
        """
        Numerics gate: projection stage should reduce divergence RMS from a
        deliberately divergent initial condition.
        """
        solver = _make_3d_solver(self.mod, self.mesh, self.h0)
        try:
            stats0 = self.mod.swe2d_get_3d_patch_stats(solver)
            nx, ny, nz = int(stats0["nx"]), int(stats0["ny"]), int(stats0["nz"])
            n = int(stats0["n_cells"])

            # u=x, v=y, w=0 induces positive divergence in the interior.
            u_ic = np.zeros(n, dtype=np.float64)
            v_ic = np.zeros(n, dtype=np.float64)
            w_ic = np.zeros(n, dtype=np.float64)
            p_ic = np.zeros(n, dtype=np.float64)
            for iz in range(nz):
                for iy in range(ny):
                    for ix in range(nx):
                        idx = iz * nx * ny + iy * nx + ix
                        u_ic[idx] = float(ix)
                        v_ic[idx] = float(iy)

            self.mod.swe2d_set_3d_patch_state(
                solver, u=u_ic, v=v_ic, w=w_ic, p=p_ic)

            before = self.mod.swe2d_get_3d_patch_stats(solver)
            self.assertGreater(
                before["divergence_rms"], 1.0e-12,
                f"Expected non-trivial initial divergence; got {before['divergence_rms']:.3e}")

            self.mod.swe2d_step(solver, 0.1)
            after = self.mod.swe2d_get_3d_patch_stats(solver)

            self.assertLess(
                after["divergence_rms"], before["divergence_rms"],
                f"Projection did not reduce divergence_rms: "
                f"{before['divergence_rms']:.6e} -> {after['divergence_rms']:.6e}")
            self.assertGreater(after["projection_iters"], 0)
            self.assertTrue(np.isfinite(after["projection_residual"]))
            self.assertGreaterEqual(after["projection_residual"], 0.0)
        finally:
            self.mod.swe2d_destroy(solver)

    # ── Reference cases (gated) ────────────────────────────────────────────────

    @unittest.skipUnless(_PHYSICS_CASES,
        "Set BACKWATER_RUN_SWE3D_PHYSICS_CASES=1 when reference datasets are staged.")
    def test_reference_case_broad_crested_weir(self):
        """
        Stage-1 reference: broad-crested weir free-surface nappe profile.
        Requires: tests/data/swe3d/broad_crested_weir.json
        """
        from tests.swe3d_reference_harness import load_case, run_and_compare
        case = load_case("broad_crested_weir")
        result = run_and_compare(self.mod, case)
        for metric_name, passed, value, ref, tol in result.iter_metrics():
            with self.subTest(metric=metric_name):
                self.assertTrue(passed,
                    f"{metric_name}: value={value:.4e}  ref={ref:.4e}  "
                    f"delta={abs(value-ref):.4e}  tol={tol:.4e}")

    @unittest.skipUnless(_PHYSICS_CASES,
        "Set BACKWATER_RUN_SWE3D_PHYSICS_CASES=1 when reference datasets are staged.")
    def test_reference_case_culvert_pressurization(self):
        """
        Stage-1 reference: culvert pressurisation transition (inlet vs outlet head).
        Requires: tests/data/swe3d/culvert_pressurization.json
        """
        from tests.swe3d_reference_harness import load_case, run_and_compare
        case = load_case("culvert_pressurization")
        result = run_and_compare(self.mod, case)
        for metric_name, passed, value, ref, tol in result.iter_metrics():
            with self.subTest(metric=metric_name):
                self.assertTrue(passed,
                    f"{metric_name}: value={value:.4e}  ref={ref:.4e}  "
                    f"delta={abs(value-ref):.4e}  tol={tol:.4e}")


if __name__ == "__main__":
    unittest.main()

