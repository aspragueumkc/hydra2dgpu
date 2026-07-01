#!/usr/bin/env python3
"""Matrix test: source × coupling path combinations.

Tests every combination of:
  - Rain: OFF / Native constant rain / Python rain callback
  - Drainage: OFF / Active
  - Structures: OFF / Active
  - Coupling path: N/A / GPU native (apply_native_device_sources) / Python callback

Each test runs N steps and reports:
  - max|h|  → blowup detection
  - any NaN/Inf in state
  - final wet cells
  - dt used

Usage:
    python -m pytest tests/test_coupling_rain_matrix.py -v -s 2>&1
"""
from __future__ import annotations

import sys
import unittest

import numpy as np

from swe2d.runtime.backend import SWE2DBackend, swe2d_available
from swe2d.runtime.coupling import SWE2DCouplingController
from swe2d.runtime.runtime_setup_configurator import SWE2DRunSetupConfigurator
from swe2d.extensions.drainage_network import SWE2DUrbanDrainageModule
from swe2d.extensions.structures import SWE2DStructureModule
from swe2d.extensions.extension_models import (
    DrainageNode, HydraulicStructure, HydraulicStructureConfig,
    InletExchange, PipeNetworkConfig, DrainageSolverMode, StructureType,
)
from swe2d import units as _u

from tests._swe2d_test_helpers import _make_rect_mesh


def _gpu_available():
    try:
        import hydra_swe2d as m
        return bool(m.swe2d_gpu_available())
    except Exception:
        return False


# ── Shared mesh: 4×2 rectangular, 16 cells, mild slope ───────────────────
NX, NY = 4, 2
LX, LY = 400.0, 200.0

def _slope_zb(x, y):
    return 0.5 - 0.0005 * x  # mild eastward slope, 0.5 at left, ~0.3 at right

def _build_mesh(backend):
    node_x, node_y, node_z, cell_nodes = _make_rect_mesh(NX, NY, LX, LY, zb_func=_slope_zb)
    backend.build_mesh(
        node_x, node_y, node_z, cell_nodes,
        bc_edge_node0=np.empty(0, dtype=np.int32),
        bc_edge_node1=np.empty(0, dtype=np.int32),
        bc_edge_type=np.empty(0, dtype=np.int32),
        bc_edge_val=np.empty(0, dtype=np.float64),
    )

def _get_rain_rate_model(rain_rate_mmhr):
    """Convert mm/hr → model units/s for the Python callback path."""
    return rain_rate_mmhr / 1000.0 / 3600.0 * _u.model_per_si_m()


def _drainage_module():
    """Simple 1-inlet drainage module, node depth = 2.0m so it actively drains."""
    mod = SWE2DUrbanDrainageModule(PipeNetworkConfig(
        enabled=True,
        nodes=[DrainageNode('N0', 0, 0, invert_elev=0.0, max_depth=4.0,
                            metadata={'surface_area': 50.0})],
        links=[],
        inlets=[InletExchange('I0', 0, 'N0',
                              crest_elev=0.5, width=1.0,
                              coefficient=0.62, max_capture=1.0)],
        outfalls=[], pipe_ends=[],
        pipe_solver_mode="diffusion_wave",
    ))
    mod.initialize()
    # Drainage node starts with water so it will actively drain cells
    mod.state.node_depth['N0'] = 2.0
    return mod


def _structures_module():
    """Single weir from cell 0→1."""
    return SWE2DStructureModule(HydraulicStructureConfig(enabled=True, structures=[
        HydraulicStructure('W0', StructureType.WEIR, 0, 1,
                           crest_elev=0.0,
                           metadata={'width': 2.0, 'cd': 0.6}),
    ]), model_to_ft=_u.model_to_ft())


# ── Per-case runner ──────────────────────────────────────────────────────
def _run_case(
    label,
    n_steps=30,
    rain_rate_mmhr=0.0,
    enable_drainage=False,
    enable_structures=False,
    use_native_coupling=False,   # True → apply_native_device_sources
    use_native_rain=False,       # True → set_rain_cn_forcing_native
    unit_config=1.0,
):
    """Run one matrix cell and return (max_h, has_nan, wet_cells, dt_used)."""
    _u.configure(unit_config)

    backend = SWE2DBackend()
    _build_mesh(backend)
    n_cells = backend.n_cells

    # Optional modules
    drain_mod = _drainage_module() if enable_drainage else None
    struct_mod = _structures_module() if enable_structures else None
    controller = None
    if drain_mod is not None or struct_mod is not None:
        controller = SWE2DCouplingController(
            cell_area=backend.cell_areas(),
            cell_bed=np.zeros(n_cells, dtype=np.float64),
            drainage=drain_mod,
            structures=struct_mod,
            length_scale_si_to_model=unit_config,
        )
        # Sync RCMK cell permutation from backend (the constructor accepts
        # inv_cell_perm but doesn't store it — bug/oversight in coupling.py)
        _inv_perm = getattr(backend, "_inv_cell_perm", None)
        if _inv_perm is not None and _inv_perm.size > 0:
            controller._inv_cell_perm = np.asarray(_inv_perm, dtype=np.int32).copy()

    # Initial condition: thin water layer
    h0 = np.full(n_cells, 0.05, dtype=np.float64)
    backend.initialize(
        h0=h0,
        hu0=np.zeros(n_cells, dtype=np.float64),
        hv0=np.zeros(n_cells, dtype=np.float64),
        n_mann=0.035,
        h_min=1.0e-4,
        cfl=0.45,
        dt_max=0.5,
        dt_fixed=0.5,
        gpu_diag_sync_interval_steps=1,
        spatial_discretization=1,
    )

    # Configure native rain (constant rate via GPU-native API)
    rain_rate_model = _get_rain_rate_model(rain_rate_mmhr)
    if rain_rate_mmhr > 0.0 and use_native_rain:
        mm_to_model = 1.0e-3 * _u.model_per_si_m()
        cfg = SWE2DRunSetupConfigurator()
        cfg.configure_constant_rain_rate_native(
            backend=backend,
            rate_model_mps=rain_rate_model,
            mm_to_model_depth=mm_to_model,
        )

    # Step loop — replicates the non-IMEX path from runtime_step_executor.py
    max_h = 0.0
    has_nan = False
    wet_cells = 0
    dt_used = 0.0

    for i in range(n_steps):
        # ── Replicate the step executor's source logic ──────────────
        # Phase 1: GPU native coupling (writes to d_external_source_mps)
        native_device_applied = False
        if controller is not None and use_native_coupling:
            native_device_applied = bool(
                controller.apply_native_device_sources(0.0, 0.5)
            )

        # Phase 2: Rain source
        # When native rain is configured, the Python callback returns 0.0
        # (rain is handled by swe2d_build_rain_cn_source_kernel inside step())
        rain_src_scalar = 0.0
        if rain_rate_mmhr > 0.0 and not use_native_rain:
            rain_src_scalar = rain_rate_model

        # Phase 3: Apply sources
        if not native_device_applied:
            # Python callback path: combine everything and upload
            src = np.full(n_cells, rain_src_scalar, dtype=np.float64)
            backend.set_external_sources_native(src)
        elif rain_src_scalar > 0.0:
            # GPU native coupling path with Python rain:
            # d_external_source_mps already has coupling sources,
            # accumulate rain on top via GPU kernel (no D2H readback)
            backend.accumulate_external_sources_native(
                np.full(n_cells, rain_src_scalar, dtype=np.float64)
            )
        # else: native_device_applied + native_rain → both on GPU already

        # Phase 4: Solver step
        # When native_rain is active, swe2d_build_rain_cn_source_kernel runs
        # inside step() and writes to d_cell_source_mps.  The step kernel
        # sums d_cell_source_mps + d_external_source_mps.
        diag = backend.step(-1.0)
        dt_used = float(diag.get("dt", 0.0))
        wet_cells = int(diag.get("wet_cells", 0))

        # Check state
        h, hu, hv = backend.get_state()
        max_h = max(max_h, float(np.max(h)))
        if not np.all(np.isfinite(h)) or not np.all(np.isfinite(hu)) or not np.all(np.isfinite(hv)):
            has_nan = True
            break

        # Early exit on blowup
        if max_h > 1.0e6:
            break

    backend.destroy()
    return max_h, has_nan, wet_cells, dt_used


# ── Matrix definition ────────────────────────────────────────────────────
# (label, rain_mmhr, drain, struct, native_coupling, native_rain)
CASES = [
    # ── Baselines (no coupling) ──
    ("01: No rain, no coupling (baseline)",                    0,   False,False, False,False),
    ("02: Rain native only, no coupling",                     25.4, False,False, False,True),
    ("03: Rain callback only, no coupling",                   25.4, False,False, False,False),

    # ── Drainage only ──
    ("04: Drainage only, no rain",                             0,   True, False, False,False),
    ("05: Drainage + rain callback (Python coupling)",        25.4, True, False, False,False),
    ("06: Drainage + rain native (GPU native coupling)",      25.4, True, False, True, True),
    ("07: Drainage + rain callback (GPU native coupling)",    25.4, True, False, True, False),

    # ── Structures only ──
    ("08: Structures only, no rain",                           0,   False,True,  False,False),
    ("09: Structures + rain callback (Python coupling)",      25.4, False,True,  False,False),
    ("10: Structures + rain native (GPU native coupling)",    25.4, False,True,  True, True),
    ("11: Structures + rain callback (GPU native coupling)",  25.4, False,True,  True, False),

    # ── Drainage + Structures ──
    ("12: Drain+Struct, no rain",                              0,   True, True,  False,False),
    ("13: Drain+Struct + rain callback (Python coupling)",    25.4, True, True,  False,False),
    ("14: Drain+Struct + rain native (GPU native coupling)",  25.4, True, True,  True, True),
    ("15: Drain+Struct + rain callback (GPU native coupling)",25.4, True, True,  True, False),
]


@unittest.skipUnless(swe2d_available(), "hydra_swe2d not built")
@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
class TestCouplingRainMatrix(unittest.TestCase):
    """Matrix test for source × coupling path."""

    N_STEPS = 40

    def test_full_matrix(self):
        """Run all combinations."""
        results = []
        print(f"\n{'='*90}")
        print(f"Source × Coupling Matrix Test ({self.N_STEPS} steps, SI)")
        print(f"{'='*90}")
        hdr = f"{'Case':<57} {'Result':>8} {'max|h|':>12} {'NaN':>6} {'Wet':>6} {'dt':>10}"
        print(hdr)
        print(f"{'-'*57} {'-'*8} {'-'*12} {'-'*6} {'-'*6} {'-'*10}")

        for label, rain_mmhr, drain, struct, native_cpl, native_rain in CASES:
            try:
                max_h, has_nan, wet_cells, dt_used = _run_case(
                    label,
                    n_steps=self.N_STEPS,
                    rain_rate_mmhr=rain_mmhr,
                    enable_drainage=drain,
                    enable_structures=struct,
                    use_native_coupling=native_cpl,
                    use_native_rain=native_rain,
                    unit_config=1.0,
                )
            except Exception as e:
                max_h, has_nan, wet_cells, dt_used = -1.0, True, 0, 0.0
                print(f"{label:<57} {'ERROR':>8} {'N/A':>12} {'ERR':>6} {'N/A':>6} {'N/A':>10}")
                print(f"  Exception: {e}")
                results.append((label, False, max_h, has_nan))
                continue

            ok = not has_nan and max_h < 100.0  # 100m is a reasonable sanity cap
            status = "PASS" if ok else "FAIL"
            results.append((label, ok, max_h, has_nan))
            print(f"{label:<57} {status:>8} {max_h:>12.4e} {str(has_nan):>6} {wet_cells:>6} {dt_used:>10.6f}")

        # Summary
        passed = sum(1 for _, ok, _, _ in results if ok)
        failed = sum(1 for _, ok, _, _ in results if not ok)
        print(f"{'='*90}")
        print(f"Passed: {passed}/{len(results)}, Failed: {failed}/{len(results)}")
        print(f"{'='*90}")

        if failed > 0:
            print("\n── FAILURES ──")
            for label, ok, max_h, has_nan in results:
                if not ok:
                    print(f"  {label}: max_h={max_h:.4e}, NaN={has_nan}")

        self.assertEqual(failed, 0, f"{failed} test(s) failed")


if __name__ == "__main__":
    unittest.main()
