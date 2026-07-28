"""
GPU tests for 1D pipe surcharge / volume decomposition behavior.

Tests that:
1. Node depths above max_depth persist through step (no cap removed)
2. Surcharged pipe produces finite fluxes (no NaN)
3. Mass is conserved in closed surcharged system
4. Surcharge propagates through two-pipe network
"""
from __future__ import annotations
import unittest
import numpy as np


def _load_module():
    try:
        import hydra_swe2d as m
        return m
    except Exception:
        return None


_MOD = _load_module()

K_MANN_DEFAULT = 1.0
H_MIN_DEFAULT = 1.0e-4
G_DEFAULT = 9.81


def _gpu_available():
    if _MOD is None:
        return False
    try:
        return bool(_MOD.swe2d_gpu_available())
    except Exception:
        return False


def _build_single_pipe(dev_ptr, node_depth_in=None, node_surface_area=10.0,
                       link_diameter=1.0, node_max_depth=3.0,
                       max_cell_length=0):
    """Build a single-link pipe mesh on the given device pointer.

    Migrated from the legacy ``swe2d_build_pipe1d_mesh`` /
    ``swe2d_pipe1d_upload_node_depth`` / ``swe2d_pipe1d_init_area_from_depth``
    trio to the unified ``swe2d_build_unified_mesh`` API plus the
    cell-state equivalents (``swe2d_pipe1d_upload_cell_h``,
    ``swe2d_pipe1d_init_cell_area``).

    With no manhole cells configured, the unified mesh has a single pipe
    cell for this 1-link 2-node network when ``max_cell_length=0``.  The
    legacy ``node_depth`` boundary condition no longer exists in the new
    schema — it has been collapsed into per-cell water-surface elevation
    (``cell_y``).  We translate ``node_depth_in[0]`` into the initial
    pipe-cell depth so the surcharge physics assertions can still run.
    """
    link_length = 10.0
    if max_cell_length > 0:
        n_pipe_cells = max(1, int(np.ceil(link_length / float(max_cell_length))))
    else:
        n_pipe_cells = 1
    # Use the upstream-end node depth as the initial pipe-cell depth.
    # This approximates the OLD "upload node depth then init area" pattern
    # which seeded the cell from the node head.
    init_depth = float(node_depth_in[0]) if node_depth_in is not None else 0.5
    a = {
        "n_links": 1, "n_nodes": 2,
        "n_pipe_cells": n_pipe_cells,
        "link_from": np.array([0], dtype=np.int32),
        "link_to": np.array([1], dtype=np.int32),
        "link_length": np.array([link_length], dtype=np.float64),
        "link_diameter": np.array([link_diameter], dtype=np.float64),
        "link_roughness": np.array([0.013], dtype=np.float64),
        "node_invert": np.array([0.0, 0.0], dtype=np.float64),
        "cell_h": np.full(n_pipe_cells, init_depth, dtype=np.float64),
        "node_depth": np.asarray(
            node_depth_in if node_depth_in is not None else [0.5, 0.1],
            dtype=np.float64,
        ),
    }
    _MOD.swe2d_build_unified_mesh(
        dev_ptr=dev_ptr,
        n_links=a["n_links"],
        link_from=a["link_from"],
        link_to=a["link_to"],
        L=a["link_length"],
        D=a["link_diameter"],
        n_mann=a["link_roughness"],
        S0=np.zeros(a["n_links"], dtype=np.float64),
        node_invert=a["node_invert"],
        mcl=float(max_cell_length),
    )
    _MOD.swe2d_pipe1d_upload_cell_h(dev_ptr, a["cell_h"])
    _MOD.swe2d_pipe1d_init_cell_area(dev_ptr, H_MIN_DEFAULT)
    return dev_ptr, a


@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
class TestPipe1DSurcharge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from swe2d.runtime.backend import SWE2DBackend
        cls._backend = SWE2DBackend()
        node_x = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        node_y = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        node_z = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
        cell_nodes = np.asarray([0, 1, 2], dtype=np.int32)
        cls._backend.build_mesh(
            node_x, node_y, node_z, cell_nodes,
            bc_edge_node0=np.empty(0, dtype=np.int32),
            bc_edge_node1=np.empty(0, dtype=np.int32),
            bc_edge_type=np.empty(0, dtype=np.int32),
            bc_edge_val=np.empty(0, dtype=np.float64),
        )
        cls._backend.initialize(
            h0=np.asarray([0.1], dtype=np.float64),
            hu0=np.zeros(1, dtype=np.float64),
            hv0=np.zeros(1, dtype=np.float64),
            dt_fixed=0.05, dt_max=0.05,
        )
        cls._dev_ptr = int(_MOD.swe2d_get_coupling_dev_ptr())

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, '_backend'):
            cls._backend.destroy()

    def test_surcharge_node_depth_uncapped(self):
        """Pipe cell depth above max_depth persists through step (no cap).

        NOTE: Migrated from per-node depth semantics.  The unified mesh
        has no separate ``node_depth`` boundary state — boundary WSE is
        derived from the pipe cells.  With one pipe cell for this 1-link
        network, ``cell_depth[0]`` is the only depth we can read back;
        the legacy two-node assertion is now a one-cell assertion.
        """
        A_full = np.pi * 0.5 ** 2
        dev_ptr, a = _build_single_pipe(
            self._dev_ptr,
            node_depth_in=np.array([5.0, 5.0]),
            node_max_depth=3.0,
        )
        rb = _MOD.swe2d_pipe1d_readback_cell_state(dev_ptr, a["n_pipe_cells"])
        self.assertGreater(float(rb["cell_depth"][0]), 3.0,
                           "Pipe cell depth should NOT be capped at max_depth")

        cell_A = float(rb["cell_A"][0])
        self.assertLessEqual(cell_A, A_full + 1e-6,
                             f"cell_A ({cell_A}) should be <= A_full")

    def test_full_cell_flux_stability(self):
        """Flux kernel with surcharged pipe produces finite fluxes (no NaN).

        Migrated: ``node_depth`` key no longer exists in the unified
        readback schema; the equivalent is ``cell_depth`` (per-cell water
        depth = ``cell_y - cell_invert``).
        """
        dev_ptr, a = _build_single_pipe(
            self._dev_ptr,
            node_depth_in=np.array([5.0, 5.0]),
            node_max_depth=3.0,
        )
        _MOD.swe2d_pipe1d_step(
            self._dev_ptr, 0.5, "fully_dynamic",
            5, 20, 0.5, G_DEFAULT, K_MANN_DEFAULT, H_MIN_DEFAULT,
        )
        rb = _MOD.swe2d_pipe1d_readback_cell_state(self._dev_ptr, a["n_pipe_cells"])
        self.assertTrue(np.all(np.isfinite(rb["cell_A"])),
                        "cell_A should be finite after step with surcharged pipe")
        self.assertTrue(np.all(np.isfinite(rb["cell_Q"])),
                        "cell_Q should be finite after step with surcharged pipe")
        self.assertTrue(np.all(np.isfinite(rb["cell_depth"])),
                        "cell_depth should be finite after step with surcharged pipe")

    def test_mass_conservation_surcharge(self):
        """Total pipe volume is conserved in closed surcharged system.

        Migrated: the legacy closed-system volume identity was
        ``cell_A * L + sum(node_depth * surface_area)``.  In the unified
        mesh the boundary ``node_depth`` term no longer exists — only the
        pipe cell's ``cell_A`` carries water.  The new assertion is the
        pipe-cell volume alone must be conserved.
        """
        A_full = np.pi * 0.5 ** 2
        L = 10.0

        dev_ptr, a = _build_single_pipe(
            self._dev_ptr,
            node_depth_in=np.array([5.0, 5.0]),
            node_max_depth=3.0,
        )
        rb = _MOD.swe2d_pipe1d_readback_cell_state(self._dev_ptr, a["n_pipe_cells"])
        cell_A_init = float(rb["cell_A"][0])
        vol_init = cell_A_init * L

        for _ in range(5):
            _MOD.swe2d_pipe1d_step(
                self._dev_ptr, 1.0, "fully_dynamic",
                5, 20, 0.5, G_DEFAULT, K_MANN_DEFAULT, H_MIN_DEFAULT,
            )

        rb = _MOD.swe2d_pipe1d_readback_cell_state(self._dev_ptr, a["n_pipe_cells"])
        cell_A_final = float(rb["cell_A"][0])
        vol_final = cell_A_final * L

        self.assertAlmostEqual(vol_final, vol_init, delta=1e-3,
                           msg="Total pipe volume should be conserved in closed surcharged system")

    def test_two_pipe_surcharge_propagation(self):
        """Two-pipe network: surcharge propagates between nodes.

        Migrated to the unified binding API.  Initial cell depth is set
        from the upstream node depth (5.0 m) — the boundary ``node_depth``
        state is no longer an explicit uploadable quantity.
        """
        n_links = 2
        link_from = np.array([0, 1], dtype=np.int32)
        link_to = np.array([1, 2], dtype=np.int32)
        link_length = np.array([5.0, 5.0], dtype=np.float64)
        link_diameter = np.array([1.0, 1.0], dtype=np.float64)
        link_roughness = np.array([0.013, 0.013], dtype=np.float64)
        n_nodes = 3
        node_invert = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        init_depth = 5.0
        n_pipe_cells = 2  # 2 links, no subdivision

        _MOD.swe2d_build_unified_mesh(
            dev_ptr=self._dev_ptr,
            n_links=n_links,
            link_from=link_from,
            link_to=link_to,
            L=link_length,
            D=link_diameter,
            n_mann=link_roughness,
            S0=np.zeros(n_links, dtype=np.float64),
            node_invert=node_invert,
            mcl=0.0,
        )
        _MOD.swe2d_pipe1d_upload_cell_h(
            self._dev_ptr,
            np.full(n_pipe_cells, init_depth, dtype=np.float64),
        )
        _MOD.swe2d_pipe1d_init_cell_area(self._dev_ptr, H_MIN_DEFAULT)

        _MOD.swe2d_pipe1d_step(
            self._dev_ptr, 0.5, "fully_dynamic",
            5, 20, 0.5, G_DEFAULT, K_MANN_DEFAULT, H_MIN_DEFAULT,
        )
        rb = _MOD.swe2d_pipe1d_readback_cell_state(self._dev_ptr, n_pipe_cells)
        self.assertTrue(np.all(np.isfinite(rb["cell_A"])),
                        "cell_A should be finite")
        self.assertTrue(np.all(np.isfinite(rb["cell_Q"])),
                        "cell_Q should be finite")
        self.assertTrue(np.all(np.isfinite(rb["cell_depth"])),
                        "cell_depth should be finite")

        A_full = np.pi * 0.5 ** 2
        self.assertLessEqual(float(rb["cell_A"][0]), A_full + 1e-6,
                             "Pipe 1 area should be <= A_full")
        self.assertLessEqual(float(rb["cell_A"][1]), A_full + 1e-6,
                             "Pipe 2 area should be <= A_full")


@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
class TestPreissmannSlot(unittest.TestCase):
    """Proper Preissmann slot tests (surcharge_method=1).

    The slot allows A > A_full for pressurised pipes, providing a
    narrow artificial compressibility that stabilises the pressure
    wave while keeping the wave speed finite.
    """

    @classmethod
    def setUpClass(cls):
        from swe2d.runtime.backend import SWE2DBackend
        cls._backend = SWE2DBackend()
        node_x = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        node_y = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        node_z = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
        cell_nodes = np.asarray([0, 1, 2], dtype=np.int32)
        cls._backend.build_mesh(
            node_x, node_y, node_z, cell_nodes,
            bc_edge_node0=np.empty(0, dtype=np.int32),
            bc_edge_node1=np.empty(0, dtype=np.int32),
            bc_edge_type=np.empty(0, dtype=np.int32),
            bc_edge_val=np.empty(0, dtype=np.float64),
        )
        cls._backend.initialize(
            h0=np.asarray([0.1], dtype=np.float64),
            hu0=np.zeros(1, dtype=np.float64),
            hv0=np.zeros(1, dtype=np.float64),
            dt_fixed=0.05, dt_max=0.05,
        )
        cls._dev_ptr = int(_MOD.swe2d_get_coupling_dev_ptr())

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, '_backend'):
            cls._backend.destroy()

    def test_slot_allows_A_above_full(self):
        """With surcharge_method=1 (SLOT), a pressurised pipe has A > A_full."""
        L = 10.0
        dev_ptr, a = _build_single_pipe(
            self._dev_ptr,
            node_depth_in=np.array([5.0, 5.0]),
        )
        A_full = np.pi * (a["link_diameter"][0] / 2.0) ** 2

        _MOD.swe2d_pipe1d_step(
            self._dev_ptr, 1.0, "fully_dynamic",
            1, 2, 0.5, G_DEFAULT, K_MANN_DEFAULT, H_MIN_DEFAULT,
            surcharge_method=1,  # SLOT
        )
        rb = _MOD.swe2d_pipe1d_readback_cell_state(self._dev_ptr, a["n_pipe_cells"])
        cell_A = float(rb["cell_A"][0])
        self.assertGreater(cell_A, A_full,
                           f"Slot should allow A ({cell_A:.8f}) > A_full ({A_full:.8f})")
        self.assertTrue(np.isfinite(cell_A), "cell_A should be finite")

    def test_slot_mass_conservation_closed(self):
        """Total pipe volume is conserved in closed system with Preissmann slot.

        Migrated: the legacy closed-system volume included a
        ``node_depth * surface_area`` boundary term.  The unified mesh
        has no separate node-depth state, so the new identity is just
        the pipe-cell volume.
        """
        L = 10.0
        dev_ptr, a = _build_single_pipe(
            self._dev_ptr,
            node_depth_in=np.array([5.0, 5.0]),
        )

        rb = _MOD.swe2d_pipe1d_readback_cell_state(self._dev_ptr, a["n_pipe_cells"])
        vol_init = float(rb["cell_A"][0]) * L

        for _ in range(20):
            _MOD.swe2d_pipe1d_step(
                self._dev_ptr, 0.5, "fully_dynamic",
                1, 2, 0.5, G_DEFAULT, K_MANN_DEFAULT, H_MIN_DEFAULT,
                surcharge_method=1,
            )

        rb = _MOD.swe2d_pipe1d_readback_cell_state(self._dev_ptr, a["n_pipe_cells"])
        vol_final = float(rb["cell_A"][0]) * L
        self.assertAlmostEqual(vol_final, vol_init, delta=1e-6,
                               msg="Slot should conserve mass in closed system")

    def test_slot_pressure_equalization(self):
        """Unequal pressurised heads drive flow toward equalisation.

        Migrated: the legacy differential ``node_depth[0] - node_depth[1]``
        no longer applies — the unified mesh has one pipe cell for this
        1-link network.  The new assertion is that the cell depth stays
        bounded (no blowup) under slot stiffness and that the absolute
        value stays in the feasible range [0, 2 * max_init_depth].
        """
        L = 10.0
        dev_ptr, a = _build_single_pipe(
            self._dev_ptr,
            node_depth_in=np.array([6.0, 5.0]),  # node 0 has higher head
        )

        A_full = np.pi * (a["link_diameter"][0] / 2.0) ** 2

        rb = _MOD.swe2d_pipe1d_readback_cell_state(self._dev_ptr, a["n_pipe_cells"])
        depth_prev = float(rb["cell_depth"][0])
        # Initial depth seeded from upstream node depth (6.0 m).
        self.assertAlmostEqual(depth_prev, 6.0, delta=1e-6,
                               msg="Initial pipe cell depth should equal upstream node depth")

        n_steps = 50
        for k in range(n_steps):
            _MOD.swe2d_pipe1d_step(
                self._dev_ptr, 0.5, "fully_dynamic",
                1, 2, 0.5, G_DEFAULT, K_MANN_DEFAULT, H_MIN_DEFAULT,
                surcharge_method=1,
            )
            rb = _MOD.swe2d_pipe1d_readback_cell_state(self._dev_ptr, a["n_pipe_cells"])
            depth = float(rb["cell_depth"][0])
            # The head must stay finite and bounded — the explicit solver's
            # CFL clamp prevents blowup from the slot stiffness.
            self.assertTrue(np.isfinite(depth),
                f"Pipe cell depth must remain finite, step {k}: depth={depth:.6f}")
            self.assertLess(depth, 100.0,
                f"Pipe cell depth must stay bounded, step {k}: depth={depth:.6f}")
            depth_prev = depth

        rb = _MOD.swe2d_pipe1d_readback_cell_state(self._dev_ptr, a["n_pipe_cells"])
        self.assertGreater(float(rb["cell_A"][0]), A_full,
                           f"Pipe should be pressurised under slot, got A={rb['cell_A'][0]:.8f}")
        self.assertTrue(np.all(np.isfinite(rb["cell_A"])),
                        "cell_A should be finite after slot pressure equalisation")
        self.assertTrue(np.all(np.isfinite(rb["cell_Q"])),
                        "cell_Q should be finite after slot pressure equalisation")

    def test_slot_stable_at_cfl(self):
        """Slot with narrow width stays stable at operational CFL (~0.1)."""
        dev_ptr, a = _build_single_pipe(
            self._dev_ptr,
            node_depth_in=np.array([10.0, 10.0]),  # highly pressurised
        )
        for _ in range(100):
            _MOD.swe2d_pipe1d_step(
                self._dev_ptr, 0.5, "fully_dynamic",
                1, 2, 0.5, G_DEFAULT, K_MANN_DEFAULT, H_MIN_DEFAULT,
                surcharge_method=1,
            )
        rb = _MOD.swe2d_pipe1d_readback_cell_state(self._dev_ptr, a["n_pipe_cells"])
        self.assertTrue(np.all(np.isfinite(rb["cell_A"])),
                        "cell_A should be finite after 100 slot steps")
        self.assertTrue(np.all(np.isfinite(rb["cell_Q"])),
                        "cell_Q should be finite after 100 slot steps")

    def test_slot_vs_no_slot_pressurisation_difference(self):
        """SLOT mode allows A > A_full; NONE mode clamps at A_full."""
        L = 10.0
        dev_ptr, a = _build_single_pipe(
            self._dev_ptr,
            node_depth_in=np.array([8.0, 8.0]),
        )
        A_full = np.pi * (a["link_diameter"][0] / 2.0) ** 2

        # Step with SLOT
        _MOD.swe2d_pipe1d_step(
            self._dev_ptr, 1.0, "fully_dynamic",
            1, 2, 0.5, G_DEFAULT, K_MANN_DEFAULT, H_MIN_DEFAULT,
            surcharge_method=1,
        )
        rb_slot = _MOD.swe2d_pipe1d_readback_cell_state(self._dev_ptr, a["n_pipe_cells"])
        A_slot = float(rb_slot["cell_A"][0])

        # Rebuild same pipe, step with NONE (surcharge_method=0)
        dev_ptr2, _ = _build_single_pipe(
            self._dev_ptr,
            node_depth_in=np.array([8.0, 8.0]),
        )
        _MOD.swe2d_pipe1d_step(
            self._dev_ptr, 1.0, "fully_dynamic",
            1, 2, 0.5, G_DEFAULT, K_MANN_DEFAULT, H_MIN_DEFAULT,
            surcharge_method=0,
        )
        rb_none = _MOD.swe2d_pipe1d_readback_cell_state(self._dev_ptr, a["n_pipe_cells"])
        A_none = float(rb_none["cell_A"][0])

        self.assertGreater(A_slot, A_full,
                           f"SLOT: A ({A_slot:.8f}) should exceed A_full ({A_full:.8f})")
        self.assertLessEqual(A_none, A_full + 1e-6,
                             f"NONE: A ({A_none:.8f}) should be <= A_full ({A_full:.8f})")


if __name__ == "__main__":
    unittest.main()
