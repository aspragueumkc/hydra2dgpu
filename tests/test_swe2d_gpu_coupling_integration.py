"""
GPU coupling integration test — exercises the EXACT same code path
as the QGIS GUI "Run Simulation" button, without requiring QGIS.

Call chain (mirrors GUI):
   SimulationWorker._execute()
   → backend_initializer.build_and_initialize()
   → SWE2DBackend (build_mesh + initialize solver)
   → build_coupling_controller() → SWE2DCouplingController
   → execute_run_timestep_loop()
     → runtime_step_executor.execute_step()
       → coupling_controller.apply_native_device_sources()  ← THE KEY CALL
       → backend.step()

Any ordering regression in the coupling preload/B C path will crash here.
"""

from __future__ import annotations

import unittest
import numpy as np

from swe2d.runtime.coupling import build_coupling_controller
from swe2d.extensions.extension_models import (
    PipeNetworkConfig,
    DrainageNode,
    DrainageLink,
    InletExchange,
    OutfallExchange,
    PipeEndExchange,
    HydraulicStructureConfig,
    HydraulicStructure,
    StructureType,
)

# Import the backend module (not SWE2DBackend — we use direct bindings for setup)
from swe2d.runtime.backend import SWE2DBackend


def _load_module():
    try:
        import hydra_swe2d as m
        return m
    except Exception:
        return None


_MOD = _load_module()


def _gpu_available():
    if _MOD is None:
        return False
    try:
        return bool(_MOD.swe2d_gpu_available())
    except Exception:
        return False


def _create_backend():
    """Build a 4-cell SWE2DBackend with solver (mirrors GUI init)."""
    node_x = np.array([0.0, 5.0, 10.0, 0.0, 5.0, 10.0], dtype=np.float64)
    node_y = np.array([0.0, 0.0, 0.0, 5.0, 5.0, 10.0], dtype=np.float64)
    node_z = np.zeros(6, dtype=np.float64)
    cell_nodes = np.array([
        0, 1, 4,   0, 4, 3,
        1, 2, 5,   1, 5, 4,
    ], dtype=np.int32)

    backend = SWE2DBackend()
    backend.build_mesh(node_x, node_y, node_z, cell_nodes,
        bc_edge_node0=np.empty(0, dtype=np.int32),
        bc_edge_node1=np.empty(0, dtype=np.int32),
        bc_edge_type=np.empty(0, dtype=np.int32),
        bc_edge_val=np.empty(0, dtype=np.float64),
    )
    n_cells = 4

    # The GUI uses SolverModelOptions with enable_pipe_network_module to
    # enable GPU structures mode.  Pass model_options with structures
    # enabled so the solver is compatible with structures coupling.
    from swe2d.extensions.extension_models import SolverModelOptions
    opts = SolverModelOptions()
    opts.hydraulic_structures.enabled = True

    backend.initialize(
        h0=np.array([0.2, 0.2, 0.1, 0.1], dtype=np.float64),
        hu0=np.zeros(n_cells, dtype=np.float64),
        hv0=np.zeros(n_cells, dtype=np.float64),
        g=9.81,
        dt_fixed=0.25,
        dt_max=0.25,
        cfl=0.45,
        model_options=opts,
    )
    return backend


def _pipe_network_config() -> PipeNetworkConfig:
    """A minimal pipe network: 2 nodes, 1 link, inlets + outfalls + pipe-ends."""
    return PipeNetworkConfig(
        enabled=True,
        nodes=[
            DrainageNode(node_id="N0", x=0.0, y=0.0, invert_elev=0.0, max_depth=3.0,
                         metadata={"surface_area": 10.0}),
            DrainageNode(node_id="N1", x=5.0, y=0.0, invert_elev=0.0, max_depth=3.0,
                         metadata={"surface_area": 10.0}),
        ],
        links=[
            DrainageLink(
                link_id="L0", from_node_id="N0", to_node_id="N1",
                length=10.0, diameter=1.0, roughness_n=0.013,
                max_cell_length=0.0,
            ),
        ],
        inlets=[
            InletExchange(
                inlet_id="I0", cell_id=0, node_id="N0",
                crest_elev=0.2, width=1.0, max_capture=0.1,
            ),
        ],
        outfalls=[
            OutfallExchange(
                outfall_id="O0", cell_id=2, node_id="N1",
                invert_elev=0.0, diameter=1.0,
            ),
        ],
        pipe_ends=[
            PipeEndExchange(
                pipe_end_id="P0", cell_id=3, node_id="N1",
                invert_elev=0.0, diameter=0.5,
            ),
        ],
        pipe_solver_mode="diffusion_wave",
        coupling_substeps=2,
        implicit_coupling_iterations=20,
        implicit_coupling_relaxation=0.5,
        gravity=9.81,
    )


def _structures_config() -> HydraulicStructureConfig:
    """One weir from cell 1 to cell 2."""
    return HydraulicStructureConfig(
        enabled=True,
        structures=[
            HydraulicStructure(
                structure_id="W0",
                structure_type=StructureType.WEIR,
                upstream_cell=1,
                downstream_cell=2,
                crest_elev=0.3,
                enabled=True,
                metadata={"width": 2.0, "coeff": 1.0},
            ),
        ],
        gravity=9.81,
    )


# ====================================================================
#  Test class
# ====================================================================

@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
class TestGUICouplingPath(unittest.TestCase):
    """Exercises the EXACT same coupling code path as the QGIS GUI.

    Uses SWE2DBackend + build_coupling_controller + apply_native_device_sources
    — the same chain the GUI calls when the user clicks "Run Simulation".
    """

    _backend: SWE2DBackend = None

    @classmethod
    def setUpClass(cls):
        cls._backend = _create_backend()
        # One warmup step to ensure GPU device is active
        diag = cls._backend.step(0.1)
        if not diag.get("gpu_active"):
            raise unittest.SkipTest("GPU solver not active")

    @classmethod
    def tearDownClass(cls):
        if cls._backend is not None:
            try:
                cls._backend.destroy()
            except Exception:
                pass
            cls._backend = None

    def _build_controller(self, pipe_cfg, struct_cfg):
        """Build a SWE2DCouplingController via the same path as the GUI."""
        n_cells = 4
        cell_area = np.full(n_cells, 12.5, dtype=np.float64)
        cell_bed = np.zeros(n_cells, dtype=np.float64)
        controller = build_coupling_controller(
            pipe_network_cfg=pipe_cfg,
            hydraulic_structures_cfg=struct_cfg,
            cell_area=cell_area,
            cell_bed=cell_bed,
            length_scale_si_to_model=1.0,
            bridge_cuda_coupling=False,
            bridge_stacked_coupling_mode="phase3_spatial",
            culvert_face_flux_mode="face_flux",
            culvert_solver_mode=0,
            drainage_gpu_method_mode="step",
            use_redistribution=False,
            inv_cell_perm=getattr(self._backend, '_inv_cell_perm', None),
            log_fn=None,
        )
        self.assertIsNotNone(controller,
                             "build_coupling_controller returned None")
        return controller

    # ── Test 1: Drainage + Structures (full coupling path) ─────────

    def test_drainage_and_structures_coupling(self):
        """Full coupling pipeline: drainage + structures, multiple timesteps.

        This is the SAME code path the GUI uses when a user configures
        a pipe network + hydraulic structures and clicks "Run Simulation".
        """
        controller = self._build_controller(
            _pipe_network_config(),
            _structures_config(),
        )
        backend = self._backend

        for step_i in range(8):
            t = float(step_i) * 0.25
            dt = 0.25

            # Step A: Coupling sources — MUST execute before solver step
            # This is the function that was crashing (NULL d_cell_wse).
            applied = controller.apply_native_device_sources(t, dt)
            self.assertTrue(applied,
                f"apply_native_device_sources returned False at step {step_i}")

            # Step B: Solver step (same as backend.step() in GUI path)
            diag = backend.step(dt)
            self.assertTrue(diag.get("gpu_active", False),
                f"GPU not active at step {step_i}")

            # Step C: Verify solver state stays finite
            h, hu, hv = backend.get_state()
            self.assertTrue(np.all(np.isfinite(h)),
                f"h not finite at step {step_i}: min={np.min(h):.6f} max={np.max(h):.6f}")
            self.assertTrue(np.all(np.isfinite(hu)),
                f"hu not finite at step {step_i}")
            self.assertTrue(np.all(np.isfinite(hv)),
                f"hv not finite at step {step_i}")
            self.assertTrue(np.all(h >= 0.0),
                f"Negative depth at step {step_i}")

    # ── Test 2: Drainage only (no structures) — was crashing here ─

    def test_drainage_only_coupling(self):
        """Drainage-only coupling (no structures).

        This was the EXACT crash scenario: a drainage-only simulation
        with pipe-ends would crash on the first timestep because
        d_cell_wse was not allocated (the structure preload was the
        only thing that allocated it).
        """
        controller = self._build_controller(
            _pipe_network_config(),
            None,  # no structures
        )
        backend = self._backend

        for step_i in range(5):
            t = float(step_i) * 0.25
            dt = 0.25

            # This was crashing here on step_i == 0 before the fix
            applied = controller.apply_native_device_sources(t, dt)
            self.assertTrue(applied,
                f"apply_native_device_sources returned False at step {step_i}")

            diag = backend.step(dt)
            self.assertTrue(diag.get("gpu_active", False))

            h, hu, hv = backend.get_state()
            self.assertTrue(np.all(np.isfinite(h)),
                f"h not finite at step {step_i}")
            self.assertTrue(np.all(h >= 0.0),
                f"Negative depth at step {step_i}")

    # ── Test 3: Structures only (no drainage) ─────────────────────

    def test_structures_only_coupling(self):
        """Structures-only coupling path."""
        controller = self._build_controller(
            None,  # no drainage
            _structures_config(),
        )
        backend = self._backend

        for step_i in range(5):
            t = float(step_i) * 0.25
            dt = 0.25

            applied = controller.apply_native_device_sources(t, dt)
            self.assertTrue(applied)

            diag = backend.step(dt)
            self.assertTrue(diag.get("gpu_active", False))

            h, hu, hv = backend.get_state()
            self.assertTrue(np.all(np.isfinite(h)),
                f"h not finite at step {step_i}")

    # ── Test 4: Re-build coupling mid-simulation ──────────────────

    def test_rebuild_coupling_mid_simulation(self):
        """Destroy and rebuild the coupling controller mid-simulation.

        This tests the C++ lazy allocation path: the second coupling
        controller starts fresh without preloaded buffers.
        """
        backend = self._backend

        for trial in range(2):
            controller = self._build_controller(
                _pipe_network_config(),
                _structures_config(),
            )

            for step_i in range(3):
                t = float(step_i) * 0.25
                dt = 0.25

                applied = controller.apply_native_device_sources(t, dt)
                self.assertTrue(applied,
                    f"apply failed trial {trial} step {step_i}")

                diag = backend.step(dt)
                self.assertTrue(diag.get("gpu_active", False))

                h, hu, hv = backend.get_state()
                self.assertTrue(np.all(np.isfinite(h)),
                    f"h not finite trial {trial} step {step_i}")


if __name__ == "__main__":
    unittest.main()
