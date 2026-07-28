"""V8: Physics-based pipe-end validation + SWMM comparison sanity checks.

Run with:
    python -m unittest -v tests.test_swmm_validation_pipe_end

Two test classes:

  TestPipeEndPhysics (primary) — validates pipe-end exchange from first
    principles: mass conservation, flow direction, steady state, dry zero-flow.
    Uses a minimal 2-cell mesh + 2-node/1-link pipe1D network with node 1
    configured as a pipe-end coupled to cell 0.  DOES NOT compare with SWMM.

  TestPipeEndVsSWMM (secondary) — verifies the V3 ``run_comparison`` harness
    executes on a pipe-end config without crashing and writes result JSON.
    These are informational only — no assertions about SWMM correctness because
    SWMM has no direct pipe-end equivalent (spec §2.9).

File ownership: ONLY this file.  Do NOT modify any other file.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Path setup
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

GRAVITY: float = 9.81
K_MANN: float = 1.0
H_MIN: float = 1.0e-4

# --------------------------------------------------------------------------- #
# Module loader
# --------------------------------------------------------------------------- #


def _load_native():
    """Load hydra_swe2d native module, returning None if unavailable."""
    try:
        import hydra_swe2d as m
        return m
    except Exception:
        return None


_NATIVE = _load_native()


def _gpu_ok() -> bool:
    """Check the native module + GPU are usable."""
    if _NATIVE is None:
        return False
    if not hasattr(_NATIVE, "swe2d_gpu_available"):
        return False
    try:
        return bool(_NATIVE.swe2d_gpu_available())
    except Exception:
        return False


def _has_pipe_end_functions() -> bool:
    """Check that all required pipe-end GPU functions are compiled in.

    Migrated: the legacy per-node bindings ``swe2d_build_pipe1d_mesh`` /
    ``swe2d_pipe1d_upload_node_depth`` / ``swe2d_pipe1d_init_area_from_depth``
    / ``swe2d_pipe1d_readback_node_state`` have been removed; we now
    require the unified mesh + cell-state schema (commit a080e61 / ce74f7d).
    """
    if _NATIVE is None:
        return False
    required = (
        "swe2d_gpu_upload_drainage_exchange_params",
        "swe2d_gpu_apply_pipe_end_bc",
        "swe2d_gpu_readback_coupling_sources",
        "swe2d_gpu_compute_coupling_full_on_device",
        "swe2d_gpu_set_coupling_dt",
        "swe2d_gpu_ensure_drainage_q_buf",
        "swe2d_gpu_preload_coupling_cell_area",
        "swe2d_build_unified_mesh",
        "swe2d_pipe1d_step",
        "swe2d_pipe1d_upload_cell_h",
        "swe2d_pipe1d_init_cell_area",
        "swe2d_pipe1d_readback_cell_state",
    )
    return all(hasattr(_NATIVE, fn) for fn in required)


# --------------------------------------------------------------------------- #
# 2-cell mesh helpers
# --------------------------------------------------------------------------- #


def _make_2cell_mesh(native_mod):
    """Build a 2-cell rectangular mesh, 10×5 m.

    Returns
    -------
    mesh
        Opaque mesh handle.
    """
    node_x = np.array([0.0, 10.0, 0.0, 10.0], dtype=np.float64)
    node_y = np.array([0.0, 0.0, 5.0, 5.0], dtype=np.float64)
    node_z = np.zeros(4, dtype=np.float64)
    cell_nodes = np.array([0, 1, 3, 0, 3, 2], dtype=np.int32)
    mesh = native_mod.swe2d_build_mesh(
        node_x, node_y, node_z, cell_nodes,
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.float64),
    )
    info = native_mod.swe2d_mesh_info(mesh)
    assert info["n_cells"] == 2, f"Expected 2 cells, got {info['n_cells']}"
    return mesh


def _make_solver(native_mod, mesh, h0: np.ndarray):
    """Create a GPU solver with hydraulic structures enabled.

    Parameters
    ----------
    native_mod : module
        The hydra_swe2d native module.
    mesh : opaque handle
        Mesh from ``_make_2cell_mesh``.
    h0 : np.ndarray
        Initial water depth per cell (n_cells,).

    Returns
    -------
    solver : opaque handle
    """
    return native_mod.swe2d_create_solver(
        mesh, h0,
        n_mann=0.035,
        cfl=0.45,
        dt_max=0.5,
        use_gpu=True,
        enable_hydraulic_structures=True,
    )


# --------------------------------------------------------------------------- #
# Empty array helpers (for unused inlet/outfall slots)
# --------------------------------------------------------------------------- #

def _ei() -> np.ndarray:
    """Empty int32 array."""
    return np.empty(0, dtype=np.int32)


def _ed() -> np.ndarray:
    """Empty float64 array."""
    return np.empty(0, dtype=np.float64)


# ==================================================================== #
#  Class 1: Physics-Based Pipe-End Validation (primary)
# ==================================================================== #
# These tests validate pipe-end exchange (spec §2.9) via physics-based
# assertions: mass conservation, flow direction, steady state, and dry
# zero-flow.  A pipe-end is a HUDSON-specific daytime pipe terminus
# exchanging directly with a co-located 2D surface cell — no SWMM
# comparison is meaningful for this concept.
#
# Test layout:
#   2D mesh: 2 triangular cells (10×5 m rectangle), cell_area=25 m² each
#   Pipe1D:  2 nodes, 1 link (100 m, 1.0 m diameter)
#     Node 0 = junction  (supply node, may receive external inflow)
#     Node 1 = pipe-end  (coupled to 2D cell 0)
#   Cell 1 = uncoupled 2D cell


@unittest.skipUnless(_NATIVE is not None, "hydra_swe2d not built")
@unittest.skipUnless(_gpu_ok(), "CUDA GPU not available")
@unittest.skipUnless(_has_pipe_end_functions(), "pipe-end GPU functions not compiled")
class TestPipeEndPhysics(unittest.TestCase):
    """Validate pipe-end exchange (spec §2.9) via physics-based assertions.

    Pipe-end is a HUDSON-specific concept: a daylighted pipe terminus where
    the 1D pipe node exchanges flow directly with a co-located 2D surface
    cell.  SWMM does not model this, so physics (mass conservation,
    directionality, steady-state) is the correct validation strategy.
    """

    N_CELLS: int = 2
    CELL_AREA: float = 25.0  # m² per triangular cell (10×5 rectangle ÷ 2)

    @classmethod
    def setUpClass(cls):
        """Build 2-cell mesh + GPU solver; step once to activate GPU."""
        cls._native = _NATIVE
        cls._mesh = _make_2cell_mesh(cls._native)
        h0 = np.full(cls.N_CELLS, 0.0, dtype=np.float64)
        cls._solver = None
        try:
            cls._solver = _make_solver(cls._native, cls._mesh, h0)
        except Exception:
            raise unittest.SkipTest("Failed to create GPU solver")

    @classmethod
    def tearDownClass(cls):
        if cls._solver is not None:
            try:
                cls._native.swe2d_destroy(cls._solver)
            except Exception:
                pass
            cls._solver = None

    # ── Per-test setup / teardown ───────────────────────────────────────

    def setUp(self):
        """Activate GPU, preload coupling workspace, build pipe1D mesh."""
        mod = self._native

        # One warm-up step ensures GPU device is live
        if not mod.swe2d_step(self._solver, 0.1).get("gpu_active", False):
            self.skipTest("solver step did not activate GPU")

        self._dev_ptr = int(mod.swe2d_get_coupling_dev_ptr())

        # Preload cell area (allocates coupling workspace)
        mod.swe2d_gpu_preload_coupling_cell_area(
            np.full(self.N_CELLS, self.CELL_AREA, dtype=np.float64))

        # Build 2-node, 1-link pipe1D mesh.
        # L0: N0 (invert=0.0) → N1 (invert=0.0), length=100 m, D=1.0 m
        mod.swe2d_build_unified_mesh(
            n_links=1,
            link_from=np.array([0], dtype=np.int32),
            link_to=np.array([1], dtype=np.int32),
            L=np.array([100.0], dtype=np.float64),
            D=np.array([1.0], dtype=np.float64),
            n_mann=np.array([0.013], dtype=np.float64),
            node_invert=np.array([0.0, 0.0], dtype=np.float64),
            mcl=10,
            dev_ptr=self._dev_ptr,
            S0=np.zeros(1, dtype=np.float64),
        )

        # Initialise pipe node depths to zero
        mod.swe2d_pipe1d_upload_cell_h(
            self._dev_ptr, np.zeros(2, dtype=np.float64))
        mod.swe2d_pipe1d_init_cell_area(self._dev_ptr, H_MIN)

        # Ensure drainage q buffer exists
        mod.swe2d_gpu_ensure_drainage_q_buf(self.N_CELLS)

        # Upload pipe-end exchange params: node 1 → cell 0
        self._upload_pipe_end_params()

    def _upload_pipe_end_params(self):
        """Upload pipe-end exchange params with node 1 coupled to cell 0."""
        self._native.swe2d_gpu_upload_drainage_exchange_params(
            # -- inlets (none) --
            inlet_cell=_ei(),
            inlet_node=_ei(),
            inlet_crest=_ed(),
            inlet_width=_ed(),
            inlet_cd=_ed(),
            inlet_qmax=_ed(),
            inlet_type=_ei(),
            inlet_grate_len=_ed(),
            inlet_grate_wid=_ed(),
            inlet_grate_kind=_ei(),
            inlet_grate_open=_ed(),
            inlet_curb_len=_ed(),
            inlet_curb_ht=_ed(),
            inlet_curb_throat=_ei(),
            inlet_slot_len=_ed(),
            inlet_slot_wid=_ed(),
            # -- outfalls (none) --
            outfall_cell=_ei(),
            outfall_node=_ei(),
            outfall_invert=_ed(),
            outfall_diameter=_ed(),
            outfall_cd=_ed(),
            outfall_qmax=_ed(),
            outfall_zero_storage=_ei(),
            # -- pipe-end: node 1 → cell 0 --
            pipe_end_cell=np.array([0], dtype=np.int32),
            pipe_end_node=np.array([1], dtype=np.int32),
            pipe_end_invert=np.array([0.0], dtype=np.float64),
            pipe_end_diameter=np.array([1.0], dtype=np.float64),
            pipe_end_area=np.array([0.7853981633974483], dtype=np.float64),
            pipe_end_kin=np.array([0.5], dtype=np.float64),
            pipe_end_kout=np.array([1.0], dtype=np.float64),
            node_max_depth=np.array([3.0], dtype=np.float64),
        )

    def tearDown(self):
        """Reset solver to dry state after each test."""
        mod = self._native
        h = np.full(self.N_CELLS, 0.0, dtype=np.float64)
        hu = np.zeros(self.N_CELLS, dtype=np.float64)
        hv = np.zeros(self.N_CELLS, dtype=np.float64)
        mod.swe2d_set_state(self._solver, h, hu, hv)

    # ── Step helpers ─────────────────────────────────────────────────────

    def _run_one_exchange_step(self) -> np.ndarray:
        """Run one pipe-end exchange step and return coupling sources.

        Sequence: apply pipe-end BC → step pipe1D → compute coupling →
        readback sources.

        Returns
        -------
        np.ndarray
            Coupling source per cell (m/s), shape (N_CELLS,).
        """
        mod = self._native
        mod.swe2d_gpu_apply_pipe_end_bc(self.N_CELLS, H_MIN)
        mod.swe2d_gpu_set_coupling_dt(1.0)
        mod.swe2d_pipe1d_step(
            self._dev_ptr, 1.0, "diffusion_wave",
            1, 2, 0.5, GRAVITY, K_MANN, H_MIN,
        )
        mod.swe2d_gpu_compute_coupling_full_on_device(
            cell_wse=None,
            n_structures=0,
            host_structure_flows=None,
        )
        return mod.swe2d_gpu_readback_coupling_sources(self.N_CELLS)

    def _read_pipe_node_depths(self) -> np.ndarray:
        """Return pipe node depth array (2,)."""
        rb = self._native.swe2d_pipe1d_readback_cell_state(
            self._dev_ptr, 1)
        return np.asarray(rb["cell_depth"], dtype=np.float64)

    def _set_cell_wse(self, cell0_depth: float, cell1_depth: float) -> None:
        """Set 2D solver WSE for both cells."""
        self._native.swe2d_set_state(
            self._solver,
            np.array([cell0_depth, cell1_depth], dtype=np.float64),
            np.zeros(self.N_CELLS, dtype=np.float64),
            np.zeros(self.N_CELLS, dtype=np.float64),
        )

    def _set_pipe_depths(self, d0: float, d1: float) -> None:
        """Upload pipe node depths and re-init areas."""
        mod = self._native
        mod.swe2d_pipe1d_upload_cell_h(
            self._dev_ptr, np.array([d0, d1], dtype=np.float64))
        mod.swe2d_pipe1d_init_cell_area(self._dev_ptr, H_MIN)

    # ═══════════════════════════════════════════════════════════════════ #
    #  test_mass_conservation
    # ═══════════════════════════════════════════════════════════════════ #

    def test_mass_conservation(self):
        """Mass conservation under constant junction inflow.

        Node 0 (junction) receives constant inflow Q_in.  Water travels
        through the pipe link to node 1 (pipe-end), then exchanges with
        cell 0.  Over many steps, mass is conserved if:

            V_in ≈ ΔV_pipe + ΔV_cell0 + ΔV_cell1

        within 1 % of total inflow volume.
        """
        mod = self._native
        Q_IN = 0.01        # m³/s — small enough to stay open-channel
        DT = 1.0           # s per step
        N_STEPS = 1000     # ≈ 16.7 min

        # Storage tracking (volumes in m³)
        v_in = 0.0                    # cumulative inflow volume
        v_cell_change = 0.0           # net volume change from coupling sources
        pipe_v0 = 0.0                 # initial pipe storage

        # Initial pipe storage (dry)
        d_init = self._read_pipe_node_depths()
        pipe_v0 = (float(d_init[0]) * 10.0 + float(d_init[1]) * 10.0)  # area=10 m² per node

        for _step in range(N_STEPS):
            # 1. Add inflow to node 0 (junction)
            d0_cur = self._read_pipe_node_depths()
            d0_new = float(d0_cur[0]) + Q_IN * DT / 10.0  # area=10 m²
            d0_new = max(d0_new, 0.0)
            d1_cur = float(d0_cur[1])
            mod.swe2d_pipe1d_upload_cell_h(
                self._dev_ptr, np.array([d0_new, d1_cur], dtype=np.float64))

            v_in += Q_IN * DT

            # 2. Run exchange step
            src = self._run_one_exchange_step()

            # 3. Track cell volume changes
            # src[i] is in m/s; volume entering cell i = src[i] * dt * cell_area
            v_cell_change += float(src[0]) * DT * self.CELL_AREA
            v_cell_change += float(src[1]) * DT * self.CELL_AREA

        # Final pipe storage
        d_final = self._read_pipe_node_depths()
        pipe_v_final = (float(d_final[0]) * 10.0 + float(d_final[1]) * 10.0)
        delta_pipe = pipe_v_final - pipe_v0

        # Mass balance
        total_stored = delta_pipe + v_cell_change
        imbalance = abs(v_in - total_stored)
        rel_imbalance = imbalance / max(v_in, 1e-12)

        print(f"\n[mass_cons] V_in={v_in:.6f} m³, "
              f"Δpipe={delta_pipe:+.6f}, Δcell={v_cell_change:+.6f}, "
              f"imbalance={imbalance:.6f} ({rel_imbalance:.4%})")

        self.assertLess(
            rel_imbalance, 0.01,
            f"Mass imbalance {rel_imbalance:.4%} exceeds 1 % — "
            f"pipe-end exchange does not conserve mass",
        )

        # Pipe should have some water after sustained inflow
        self.assertGreater(
            pipe_v_final, 0.0,
            "After 1000 s of 0.01 m³/s inflow, pipe should not be dry "
            "(inflow should wet the network)",
        )

    # ═══════════════════════════════════════════════════════════════════ #
    #  test_outflow_direction
    # ═══════════════════════════════════════════════════════════════════ #

    def test_outflow_direction(self):
        """Outflow: pipe WSE > 2D WSE → water flows pipe→2D.

        Set node 1 (pipe-end) with 2.0 m depth, cell 0 dry.
        After one exchange step, cell 0 should gain water (src > 0)
        and pipe node 1 depth should decrease.
        """
        # Pipe node 1 deep (2.0 m), node 0 with some water to drive flow
        self._set_pipe_depths(d0=2.5, d1=2.0)
        self._set_cell_wse(cell0_depth=0.0, cell1_depth=0.0)

        d_before = self._read_pipe_node_depths()
        src = self._run_one_exchange_step()
        d_after = self._read_pipe_node_depths()

        # Cell 0 should gain water (positive source into cell)
        self.assertGreater(
            float(src[0]), 0.0,
            f"Expected cell 0 to gain water (pipe→2D), got src[0]={src[0]:.6f}",
        )

        # Pipe node 1 (pipe-end) should lose water
        self.assertLess(
            float(d_after[1]), float(d_before[1]),
            f"Pipe-end node depth should decrease (water leaves pipe), "
            f"before={d_before[1]:.4f}, after={d_after[1]:.4f}",
        )

        # Cell 1 (uncoupled) should have near-zero exchange
        self.assertAlmostEqual(
            float(src[1]), 0.0, delta=1e-6,
            msg=f"Uncoupled cell 1 should have zero exchange, got {src[1]:.10f}",
        )

        print(f"\n[outflow] src[0]={src[0]:.6f} m/s, "
              f"pipe d1: {d_before[1]:.4f}→{d_after[1]:.4f} m")

    # ═══════════════════════════════════════════════════════════════════ #
    #  test_inflow_direction
    # ═══════════════════════════════════════════════════════════════════ #

    def test_inflow_direction(self):
        """Inflow: 2D WSE > pipe invert → water flows 2D→pipe.

        Set cell 0 with 1.0 m of surface water, pipe node 1 dry (h=0).
        After pipe-end BC + exchange step, cell 0 should lose water
        (src < 0) and pipe node 1 should gain water.
        """
        self._set_pipe_depths(d0=0.0, d1=0.0)
        self._set_cell_wse(cell0_depth=1.0, cell1_depth=0.0)

        d_before = self._read_pipe_node_depths()
        src = self._run_one_exchange_step()
        d_after = self._read_pipe_node_depths()

        # Cell 0 should lose water (negative source = water leaving cell)
        self.assertLess(
            float(src[0]), 0.0,
            f"Expected cell 0 to lose water (2D→pipe), got src[0]={src[0]:.6f}",
        )

        # Pipe node 1 (pipe-end) should gain water
        self.assertGreater(
            float(d_after[1]), float(d_before[1]),
            f"Pipe-end node depth should increase (water enters pipe), "
            f"before={d_before[1]:.4f}, after={d_after[1]:.4f}",
        )

        # Cell 1 (uncoupled) should have near-zero exchange
        self.assertAlmostEqual(
            float(src[1]), 0.0, delta=1e-6,
            msg=f"Uncoupled cell 1 should have zero exchange, got {src[1]:.10f}",
        )

        print(f"\n[inflow] src[0]={src[0]:.6f} m/s, "
              f"pipe d1: {d_before[1]:.4f}→{d_after[1]:.4f} m")

    # ═══════════════════════════════════════════════════════════════════ #
    #  test_steady_state_no_exchange
    # ═══════════════════════════════════════════════════════════════════ #

    def test_steady_state_no_exchange(self):
        """Equilibrium: pipe WSE ≈ 2D WSE → no net exchange.

        Set node 1 depth = cell 0 depth = 0.5 m (equal WSE).
        After exchange step, coupling source should be ≈ 0 and node
        depths should remain unchanged (within small tolerance).
        """
        self._set_pipe_depths(d0=0.5, d1=0.5)
        self._set_cell_wse(cell0_depth=0.5, cell1_depth=0.0)

        d_before = self._read_pipe_node_depths()
        src = self._run_one_exchange_step()
        d_after = self._read_pipe_node_depths()

        # Coupling source should be negligible (no driving head)
        self.assertAlmostEqual(
            float(src[0]), 0.0, delta=1e-6,
            msg=f"At equilibrium (h_pipe = h_cell), exchange should be zero; "
            f"got src[0]={src[0]:.10f}",
        )

        # Pipe node depths should remain stable
        self.assertAlmostEqual(
            float(d_after[1]), float(d_before[1]), delta=1e-6,
            msg=f"Pipe-end depth should not change at equilibrium; "
            f"before={d_before[1]:.6f}, after={d_after[1]:.6f}",
        )

        print(f"\n[steady] src[0]={src[0]:.10f} m/s, "
              f"pipe d1: {d_before[1]:.4f}→{d_after[1]:.4f} m")

    # ═══════════════════════════════════════════════════════════════════ #
    #  test_dry_conditions_zero_flow
    # ═══════════════════════════════════════════════════════════════════ #

    def test_dry_conditions_zero_flow(self):
        """Dry: both pipe and 2D cells empty → zero exchange.

        When both the pipe-end node and the coupled cell are completely
        dry (h = 0), the exchange rate must be zero — no phantom flow.
        """
        self._set_pipe_depths(d0=0.0, d1=0.0)
        self._set_cell_wse(cell0_depth=0.0, cell1_depth=0.0)

        d_before = self._read_pipe_node_depths()
        src = self._run_one_exchange_step()
        d_after = self._read_pipe_node_depths()

        # No exchange should occur
        self.assertAlmostEqual(
            float(src[0]), 0.0, delta=1e-10,
            msg=f"Dry conditions: expected zero source in cell 0, got {src[0]:.12f}",
        )
        self.assertAlmostEqual(
            float(src[1]), 0.0, delta=1e-10,
            msg=f"Dry conditions: expected zero source in cell 1, got {src[1]:.12f}",
        )

        # Pipe depths should remain zero
        self.assertAlmostEqual(
            float(d_after[0]), 0.0, delta=1e-10,
            msg=f"Dry node 0 depth should be zero, got {d_after[0]:.12f}",
        )
        self.assertAlmostEqual(
            float(d_after[1]), 0.0, delta=1e-10,
            msg=f"Dry pipe-end depth should be zero, got {d_after[1]:.12f}",
        )

        print(f"\n[dry] src[0]={src[0]:.12f}, src[1]={src[1]:.12f}, "
              f"pipe depths=({d_after[0]:.6f},{d_after[1]:.6f})")

    # ═══════════════════════════════════════════════════════════════════ #
    #  test_dry_surface_zeroes_pipe_depths
    # ═══════════════════════════════════════════════════════════════════ #

    def test_dry_surface_zeroes_pipe_depths(self):
        """Dry surface cells force pipe-end nodes to zero.

        When both surface cells are dry, ``apply_pipe_end_bc`` must
        zero the coupled pipe node depth, even if the pipe was
        previously wet.  This is a sanity check mirroring the
        ``test_dry_surface_cells_make_pipe_nodes_dry`` behaviour.
        """
        mod = self._native

        # Pre-wet the pipe
        self._set_pipe_depths(d0=2.0, d1=1.0)
        d_wet = self._read_pipe_node_depths()
        self.assertGreater(float(d_wet[1]), 0.0,
                           "Sanity: pipe-end should be wet before drying surface")

        # Dry surface cells
        self._set_cell_wse(cell0_depth=0.0, cell1_depth=0.0)

        # Apply pipe-end BC only (no step needed)
        mod.swe2d_gpu_apply_pipe_end_bc(self.N_CELLS, H_MIN)

        d_dry = self._read_pipe_node_depths()

        # Pipe-end node must be zero
        self.assertAlmostEqual(
            float(d_dry[1]), 0.0, delta=1e-10,
            msg=f"Dry surface should zero pipe-end depth, got {d_dry[1]:.12f}",
        )

        print(f"\n[dry_surface] pipe-end depth: {d_wet[1]:.4f} → {d_dry[1]:.4f} m")


# ==================================================================== #
#  Class 2: SWMM Comparison Sanity Checks (secondary, informational)
# ==================================================================== #
# These tests verify that the V3 `run_comparison` harness executes on
# pipe-end configs without crashing.  They do NOT assert correctness
# against SWMM because SWMM has no direct pipe-end equivalent — the
# pipe-end concept (spec §2.9) is a HUDSON-specific coupling where 1D
# pipe terminates at a co-located 2D surface cell with direct exchange.


class TestPipeEndVsSWMM(unittest.TestCase):
    """Informational: verify the V3 harness works with pipe-end configs.

    These tests do NOT assert correctness against SWMM — they only verify
    the comparison infrastructure works.  SWMM-vs-pipe1D comparison for
    pipe-end scenarios is not meaningful per the spec (SWMM has no direct
    pipe-end equivalent).
    """

    def setUp(self):
        """Load harness, tolerances, and synthetic pipe-end config."""
        from tests.swmm_validation.compare import (
            ComparisonResult,
            run_comparison,
            ScenarioBundle,
            ToleranceSpec,
        )
        from tests.swmm_validation.tolerances import TOLERANCES
        from tests.swmm_validation.synthetic.site_drainage import (
            synth_site_drainage_pipe_end,
        )

        self.run_comparison = run_comparison
        self.ScenarioBundle = ScenarioBundle
        self.ToleranceSpec = ToleranceSpec
        self.TOLERANCES = TOLERANCES
        self.ComparisonResult = ComparisonResult
        self.synth_pipe_end = synth_site_drainage_pipe_end
        self.REPO_ROOT = REPO_ROOT

    # ── helper ─────────────────────────────────────────────────────────

    def _const_inflow(self, node_id: str, flow_cms: float,
                      duration_h: float = 24.0) -> dict:
        """Return a hydrology dict with constant inflow at one node."""
        return {
            node_id: [
                (0.0, float(flow_cms)),
                (float(duration_h), float(flow_cms)),
            ]
        }

    # ── tests ──────────────────────────────────────────────────────────

    def test_run_comparison_produces_result(self):
        """V3 harness runs on pipe-end config without crashing.

        Uses the simplest scenario (short dry run, 1 s) to verify the full
        call chain executes — SWMM, pipe1D, alignment, error computation,
        JSON write — without raising.  Does NOT assert pass/fail correctness
        against SWMM.
        """
        cfg = self.synth_pipe_end()
        inflows = self._const_inflow("J1", flow_cms=0.0, duration_h=1.0)

        tol = self.ToleranceSpec(
            regimes={
                "open_channel": {
                    "depth_rmse_rel": self.TOLERANCES["open_channel"].depth_rmse_rel_max,
                    "flow_rmse_rel": self.TOLERANCES["open_channel"].flow_rmse_rel_max,
                    "depth_max_rel": self.TOLERANCES["open_channel"].depth_max_rel_max,
                    "flow_max_rel": self.TOLERANCES["open_channel"].flow_max_rel_max,
                }
            }
        )

        bundle = self.ScenarioBundle(
            name="pipe_end_smoke",
            swmm_inp_path=self.REPO_ROOT
            / "reference/swmm_canonical/Site_Drainage_Model.inp",
            pipe1d_config=cfg,
            duration_s=1.0,  # minimal runtime
            hydrology=inflows,
            expected_regimes=["open_channel"],
        )

        with tempfile.TemporaryDirectory() as td:
            workdir = Path(td)
            result = self.run_comparison(bundle, tol, workdir)

        # The harness returns a ComparisonResult even on SWMM failure
        self.assertIsInstance(result, self.ComparisonResult)
        self.assertEqual(result.scenario, "pipe_end_smoke")
        self.assertIsInstance(result.pass_fail, dict)
        self.assertIsInstance(result.metadata, dict)

        # Print diagnostics
        error = result.metadata.get("error") if result.metadata else None
        if error:
            print(f"\n[smoke] SWMM/Pipe1D error (expected for pipe-end): {error[:120]}")
        else:
            print(f"\n[smoke] pass_fail={result.pass_fail}")

    def test_run_comparison_writes_json(self):
        """Comparison JSON artefact written even when comparison is not meaningful.

        The V3 harness writes ``result.json`` regardless of whether SWMM
        crashes or passes.  This test verifies the artifact is present.
        """
        cfg = self.synth_pipe_end()
        inflows = self._const_inflow("J1", flow_cms=0.0, duration_h=1.0)

        tol = self.ToleranceSpec(
            regimes={
                "open_channel": {
                    "depth_rmse_rel": self.TOLERANCES["open_channel"].depth_rmse_rel_max,
                    "flow_rmse_rel": self.TOLERANCES["open_channel"].flow_rmse_rel_max,
                    "depth_max_rel": self.TOLERANCES["open_channel"].depth_max_rel_max,
                    "flow_max_rel": self.TOLERANCES["open_channel"].flow_max_rel_max,
                }
            }
        )

        bundle = self.ScenarioBundle(
            name="pipe_end_json_written",
            swmm_inp_path=self.REPO_ROOT
            / "reference/swmm_canonical/Site_Drainage_Model.inp",
            pipe1d_config=cfg,
            duration_s=1.0,
            hydrology=inflows,
            expected_regimes=["open_channel"],
        )

        with tempfile.TemporaryDirectory() as td:
            workdir = Path(td)
            self.run_comparison(bundle, tol, workdir)

            json_path = workdir / "result.json"
            self.assertTrue(
                json_path.exists(),
                f"result.json not written by V3 harness at {json_path}",
            )

            # Verify the JSON is actually valid
            with open(json_path) as fh:
                import json
                data = json.load(fh)
            self.assertIn("scenario", data)

        print(f"\n[json] result.json written to {json_path} — {len(str(data))} chars")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    unittest.main()
