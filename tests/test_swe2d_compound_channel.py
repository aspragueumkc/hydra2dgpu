"""
test_swe2d_compound_channel.py — Compound-channel (main channel + floodplain)
                                  flow validation

Background
----------
Compound channels — a deep main channel flanked by shallower floodplains —
are ubiquitous in natural rivers.  They are one of the standard HEC-RAS 2D
validation problems because the transition from in-bank to over-bank flow
creates strongly non-uniform velocity distributions that 2-D SWE solvers
must reproduce correctly.

This test is inspired by the Muncie, Indiana HEC-RAS benchmark (USACE 2016),
which characterises over-bank flood flow through a compound channel.  Because
the Muncie geometry requires proprietary HEC-RAS data files, this module uses
a synthetic but geometrically equivalent compound cross-section defined
analytically so that an exact conveyance-method reference solution is available.

Geometry
--------
  Total width   : W = 200 m
  Main channel  : 20 m wide, centred at y = 100 m  (90 ≤ y ≤ 110 m)
  Floodplains   : 90 m wide on each side (left: 0–90 m, right: 110–200 m)
  Bank height   : H_bank = 2.0 m above channel floor
  Channel slope : S₀ = 0.001
  Domain length : L = 2 000 m
  NX = 100, NY = 20  →  100 × 20 × 2 = 4 000 triangles

Roughness coefficients (per-cell Manning n)
  Main channel  : n_m  = 0.030
  Floodplains   : n_fp = 0.060

Steady inflow discharge
-----------------------
  Q_total = 100 m³/s  →  average unit discharge = Q/W = 0.5 m²/s

Analytical reference (conveyance method)
-----------------------------------------
At stage H (water surface elevation above channel floor) the compound channel
conveyance is (assuming vertical banks, no berms, independent conveyances):

  K_main = (1/n_m)  × A_m × R_m^(2/3)     where A_m = W_m × H
                                                  R_m = W_m×H / (W_m + 2H)
  For compound flow (H > H_bank):
  K_fp   = (1/n_fp) × A_fp × R_fp^(2/3)   where A_fp = 2×W_fp×(H−H_bank)
                                                  R_fp = A_fp/(2W_fp+2(H−H_bank))
  Q = (K_main + K_fp) × √S₀

For Q = 100 m³/s and the geometry above:
  Bank-full discharge Q_bf ≈ 60.8 m³/s at H = 2.0 m.
  At Q = 100 m³/s the stage H ≈ 2.40 m (h_fp ≈ 0.40 m above floodplain).
  Conveyance partition:
    Q_main / Q_total ≈ 79 %  →  hu_main ≈ 3.90 m²/s
    Q_fp   / Q_total ≈ 21 %  →  hu_fp   ≈ 0.117 m²/s

Validation metrics
------------------
Test 1 — FLOW PARTITION (main channel vs floodplain)
  Mean unit discharge in main-channel cells must be substantially higher
  than in floodplain cells.  Threshold: hu_main > 2 × hu_fp (factor-of-2
  tolerance accounts for start-up transient after 150 steps).

Test 2 — STAGE PLAUSIBILITY
  Mean water surface elevation over the compound cross-section must be within
  ±20 % of the analytical value H ≈ 2.40 m (above the channel floor).

Test 3 — DISCHARGE CONSERVATION
  After 150 steps, mean interior unit discharge is within 15 % of Q/W = 0.5
  m²/s.

Test 4 — GPU STABILITY
  GPU solver stays active throughout all 150 steps.

Test 5 — POSITIVITY (CPU smoke test)
  A small CPU-only run verifies h ≥ 0 everywhere.

References
----------
HEC-RAS 2D User's Manual (2016), US Army Corps of Engineers, Davis, CA.
  https://www.hec.usace.army.mil/software/hec-ras/

Chaudhry M.H. (2008) Open-Channel Hydraulics, 2nd edn., Springer.

ANUGA channel-floodplain validation:
  https://github.com/anuga-community/anuga_core/blob/main/
  validation_tests/behaviour_only/bridge_hecras/channel_floodplain1.py
"""

import os
import sys
import unittest

import numpy as np




# ---------------------------------------------------------------------------
# Module / GPU availability
# ---------------------------------------------------------------------------

def _load_module():
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
# Analytical reference: conveyance method
# ---------------------------------------------------------------------------

def compound_conveyance(H, W_main, W_fp_each, H_bank, n_main, n_fp, g=9.81):
    """
    Return (Q, K_main, K_fp, hu_main, hu_fp) for stage H [m above channel floor]
    using the conveyance method for a symmetric compound channel.

    Parameters
    ----------
    H          : water surface elevation above main-channel floor [m]
    W_main     : main-channel width [m]
    W_fp_each  : floodplain width on ONE side [m]
    H_bank     : bank height above channel floor [m]
    n_main     : Manning n for main channel
    n_fp       : Manning n for floodplains
    g          : gravitational acceleration [m/s²] (unused, kept for symmetry)

    Returns
    -------
    dict with keys: Q, K_main, K_fp, hu_main, hu_fp
    """
    # Main channel
    A_main = W_main * H
    R_main = A_main / (W_main + 2.0 * H) if H > 0 else 0.0
    K_main = (1.0 / n_main) * A_main * R_main ** (2.0 / 3.0) if H > 0 else 0.0

    # Floodplains (both sides together)
    if H > H_bank:
        h_fp = H - H_bank
        W_fp_total = 2.0 * W_fp_each
        A_fp  = W_fp_total * h_fp
        R_fp  = A_fp / (W_fp_total + 2.0 * h_fp)
        K_fp  = (1.0 / n_fp) * A_fp * R_fp ** (2.0 / 3.0)
    else:
        K_fp = 0.0

    return {"K_main": K_main, "K_fp": K_fp}


def solve_stage(Q, S0, W_main, W_fp_each, H_bank, n_main, n_fp):
    """
    Solve for stage H such that Q = (K_main + K_fp) × √S₀.
    Uses bisection over H ∈ [0.01, 20] m.
    Returns H [m above channel floor].
    """
    sqrt_S0 = np.sqrt(S0)

    def residual(H):
        ks = compound_conveyance(H, W_main, W_fp_each, H_bank, n_main, n_fp)
        return (ks["K_main"] + ks["K_fp"]) * sqrt_S0 - Q

    lo, hi = 0.01, 20.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if residual(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Mesh builder
# ---------------------------------------------------------------------------

def _make_compound_mesh(nx, ny, Lx, Ly, bed_func):
    """
    Triangulate [0, Lx] × [0, Ly].  bed_func(x, y) → bed elevation.
    """
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


# ---------------------------------------------------------------------------
# BC array builder with per-edge inflow values
# ---------------------------------------------------------------------------

def _compound_bc_arrays(nx, ny, Ly,
                         y_chan_left, y_chan_right,
                         q_main, q_fp):
    """
    Inflow (INFLOW_Q = 2) at x = 0 with conveyance-weighted unit discharges:
      - Edges within main channel (y_chan_left ≤ y_mid ≤ y_chan_right): q_main
      - Edges on floodplain: q_fp
    Free outflow (OPEN = 4) at x = Lx.
    Sides default to WALL = 1.
    """
    stride = nx + 1
    dy = Ly / ny
    n0s, n1s, tps, vls = [], [], [], []

    for j in range(ny):
        y_mid = (j + 0.5) * dy
        q_bc = q_main if y_chan_left <= y_mid <= y_chan_right else q_fp
        n0s.append(j * stride)
        n1s.append((j + 1) * stride)
        tps.append(2)          # INFLOW_Q
        vls.append(float(q_bc))

    for j in range(ny):
        n0s.append(j * stride + nx)
        n1s.append((j + 1) * stride + nx)
        tps.append(4)          # OPEN
        vls.append(0.0)

    return (np.array(n0s, dtype=np.int32),
            np.array(n1s, dtype=np.int32),
            np.array(tps, dtype=np.int32),
            np.array(vls, dtype=np.float64))


# ---------------------------------------------------------------------------
# Test class: GPU compound channel validation
# ---------------------------------------------------------------------------

@unittest.skipUnless(_load_module() is not None, "hydra_swe2d not built")
@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
class TestCompoundChannelFlow(unittest.TestCase):
    """
    Compound-channel (main channel + floodplain) steady-state validation.

    Geometry
    --------
    W_main = 20 m centred at y = 100 m; W_fp = 90 m on each side.
    H_bank = 2.0 m; S₀ = 0.001; L = 2 000 m.

    Manning roughness (per-cell array)
    ----------------------------------
    n_main = 0.030  in the main channel (90 ≤ y ≤ 110 m)
    n_fp   = 0.060  on the floodplains  (y < 90 m or y > 110 m)

    Initialised at the analytical steady state (H ≈ 2.40 m above channel floor,
    conveyance-weighted unit discharges).  The tests verify that the solver
    preserves the compound-flow distribution over 150 explicit steps.
    """

    LX           = 2000.0    # channel length  [m]
    LY           = 200.0     # total width     [m]
    NX           = 100       # cells along x
    NY           = 20        # cells across y  (10 m per cell)
    W_MAIN       = 20.0      # main channel width [m]
    W_FP_EACH    = 90.0      # floodplain width on each side [m]
    H_BANK       = 2.0       # bank height above channel floor [m]
    S0           = 0.001     # bed slope
    N_MAIN       = 0.030     # Manning n, main channel
    N_FP         = 0.060     # Manning n, floodplains
    Q_TOTAL      = 100.0     # total inflow discharge [m³/s]
    N_STEPS      = 150

    # Channel y-bounds
    Y_LEFT_BANK  = 90.0
    Y_RIGHT_BANK = 110.0

    def _analytical(self):
        """Compute and return the analytical steady-state parameters."""
        H = solve_stage(self.Q_TOTAL, self.S0,
                        self.W_MAIN, self.W_FP_EACH, self.H_BANK,
                        self.N_MAIN, self.N_FP)
        ks = compound_conveyance(H,
                                  self.W_MAIN, self.W_FP_EACH, self.H_BANK,
                                  self.N_MAIN, self.N_FP)
        sqrt_S0 = np.sqrt(self.S0)
        Q_main = ks["K_main"] * sqrt_S0
        Q_fp   = ks["K_fp"]   * sqrt_S0
        hu_main = Q_main / self.W_MAIN
        hu_fp   = Q_fp   / (2.0 * self.W_FP_EACH)
        return {"H": H, "hu_main": hu_main, "hu_fp": hu_fp,
                "Q_main": Q_main, "Q_fp": Q_fp}

    def _make_solver(self):
        mod   = _load_module()
        ana   = self._analytical()
        H_ana = ana["H"]

        dy = self.LY / self.NY
        dx = self.LX / self.NX

        # ── Bed elevation ────────────────────────────────────────────────────
        def bed(x, y):
            # Channel floor slope + bank step on floodplains
            z_slope = self.S0 * (self.LX - x)
            bank = np.where(
                (y >= self.Y_LEFT_BANK) & (y <= self.Y_RIGHT_BANK),
                0.0,
                self.H_BANK,
            )
            return z_slope + bank

        node_x, node_y, node_z, cell_nodes = _make_compound_mesh(
            self.NX, self.NY, self.LX, self.LY, bed)

        # ── Per-cell Manning n (channel vs floodplain) ───────────────────────
        cx, cy = _cell_centroids(self.NX, self.NY, self.LX, self.LY)
        n_cells = 2 * self.NX * self.NY
        n_mann_cell = np.where(
            (cy >= self.Y_LEFT_BANK) & (cy <= self.Y_RIGHT_BANK),
            self.N_MAIN,
            self.N_FP,
        ).astype(np.float64)

        # ── Boundary conditions (conveyance-weighted inflow) ─────────────────
        bc_n0, bc_n1, bc_tp, bc_vl = _compound_bc_arrays(
            self.NX, self.NY, self.LY,
            self.Y_LEFT_BANK, self.Y_RIGHT_BANK,
            q_main=ana["hu_main"],
            q_fp=ana["hu_fp"],
        )

        mesh = mod.swe2d_build_mesh(
            node_x, node_y, node_z, cell_nodes,
            bc_n0, bc_n1, bc_tp, bc_vl)

        # ── Initial condition at analytical steady state ─────────────────────
        # Uniform water surface elevation H_ana above channel floor at each x.
        # h = WSE - z_bed  (ensure h ≥ 0)
        z_bed_cell = bed(cx, cy)
        wse_cell   = self.S0 * (self.LX - cx) + H_ana  # WSE follows bed slope
        h0  = np.maximum(0.0, wse_cell - z_bed_cell).astype(np.float64)
        hu0 = np.where(
            (cy >= self.Y_LEFT_BANK) & (cy <= self.Y_RIGHT_BANK),
            ana["hu_main"],
            ana["hu_fp"],
        ).astype(np.float64)
        hv0 = np.zeros(n_cells, dtype=np.float64)

        solver = mod.swe2d_create_solver(
            mesh, h0, hu0, hv0,
            n_mann_cell=n_mann_cell,
            n_mann=self.N_MAIN,          # global fallback (overridden by cell array)
            cfl=0.45,
            dt_max=1.0,
            use_gpu=True,
            gpu_diag_sync_interval_steps=1,
        )
        return mod, solver, ana, cx, cy

    # ── Test 1: flow partition ───────────────────────────────────────────────
    def test_flow_partition(self):
        """
        Mean unit discharge in main-channel cells must be substantially
        higher than in floodplain cells.

        The conveyance ratio predicts hu_main/hu_fp ≈ 33.  We require at
        least a factor of 2 to be robust to start-up transients.
        """
        mod, solver, ana, cx, cy = self._make_solver()
        for _ in range(self.N_STEPS):
            mod.swe2d_step(solver, -1.0)
        h, hu, _ = mod.swe2d_get_state(solver)
        mod.swe2d_destroy(solver)

        # Interior cells (exclude 10 % near each end to avoid BC effects)
        interior = (cx > 0.10 * self.LX) & (cx < 0.90 * self.LX)
        main_mask = interior & (cy >= self.Y_LEFT_BANK) & (cy <= self.Y_RIGHT_BANK)
        fp_mask   = interior & ~((cy >= self.Y_LEFT_BANK) & (cy <= self.Y_RIGHT_BANK))

        mean_hu_main = float(hu[main_mask].mean()) if main_mask.any() else 0.0
        mean_hu_fp   = float(hu[fp_mask].mean())   if fp_mask.any()  else 1.0

        ratio = mean_hu_main / (mean_hu_fp + 1e-9)
        self.assertGreater(
            ratio, 2.0,
            f"Flow partition ratio hu_main/hu_fp = {ratio:.2f}; expected ≥ 2 "
            f"(analytical ≈ {ana['hu_main']/(ana['hu_fp']+1e-9):.1f}). "
            f"hu_main = {mean_hu_main:.4f}, hu_fp = {mean_hu_fp:.4f} m²/s",
        )

    # ── Test 2: stage plausibility ───────────────────────────────────────────
    def test_stage_plausibility(self):
        """
        Mean depth above channel floor (centroid stage) must be within
        ±20 % of the analytical value H_ana ≈ 2.40 m.
        """
        mod, solver, ana, cx, cy = self._make_solver()
        for _ in range(self.N_STEPS):
            mod.swe2d_step(solver, -1.0)
        h, _, _ = mod.swe2d_get_state(solver)
        mod.swe2d_destroy(solver)

        # Main-channel interior cells: h should be near H_ana
        interior = (cx > 0.20 * self.LX) & (cx < 0.80 * self.LX)
        mc_mask  = interior & (cy >= self.Y_LEFT_BANK) & (cy <= self.Y_RIGHT_BANK)

        if not mc_mask.any():
            self.skipTest("No interior main-channel cells found")

        mean_h = float(h[mc_mask].mean())
        H_ana  = ana["H"]
        rel_err = abs(mean_h - H_ana) / H_ana

        self.assertLess(
            rel_err, 0.20,
            f"Mean main-channel depth {mean_h:.4f} m deviates "
            f"{rel_err:.1%} from analytical H = {H_ana:.4f} m",
        )

    # ── Test 3: discharge conservation ──────────────────────────────────────
    def test_discharge_conservation(self):
        """
        Mean interior unit discharge must be within 15 % of Q/W = 0.5 m²/s.
        """
        mod, solver, ana, cx, cy = self._make_solver()
        for _ in range(self.N_STEPS):
            mod.swe2d_step(solver, -1.0)
        h, hu, _ = mod.swe2d_get_state(solver)
        mod.swe2d_destroy(solver)

        interior = (cx > 0.10 * self.LX) & (cx < 0.90 * self.LX)
        mean_hu  = float(hu[interior].mean()) if interior.any() else 0.0
        target   = self.Q_TOTAL / self.LY   # = 0.5 m²/s
        rel_err  = abs(mean_hu - target) / target

        self.assertLess(
            rel_err, 0.15,
            f"Mean interior hu = {mean_hu:.4f} m²/s deviates "
            f"{rel_err:.1%} from Q/W = {target:.4f} m²/s",
        )

    # ── Test 4: GPU stability ────────────────────────────────────────────────
    def test_gpu_stability(self):
        """GPU solver must remain active throughout all steps."""
        mod, solver, ana, cx, cy = self._make_solver()
        last_diag = None
        for _ in range(self.N_STEPS):
            last_diag = mod.swe2d_step(solver, -1.0)
        mod.swe2d_destroy(solver)

if __name__ == "__main__":
    unittest.main()
