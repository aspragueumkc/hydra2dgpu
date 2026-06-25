"""Benchmark all spatial schemes at structured 100K and GMSH ~150K.

Usage:
    python tests/benchmark_all_schemes.py            # current build
    # To compare with pre-optimization:
    # git stash && git checkout e3d6b90
    # cmake --build build -j
    # python tests/benchmark_all_schemes.py
    # git checkout main && git stash pop
"""
from __future__ import annotations

import sys
import time
import numpy as np

sys.path.insert(0, ".")

from swe2d.runtime.backend import SWE2DBackend
from tests._swe2d_test_helpers import _make_rect_mesh

SCHEMES = [
    (0, "FV_FIRST_ORDER"),
    (1, "FV_MUSCL_FAST"),
    (2, "FV_MUSCL_MINMOD"),
    (3, "FV_MUSCL_MC"),
    (4, "FV_MUSCL_VAN_LEER"),
]

N_STEPS = 30


def benchmark(name: str, node_x, node_y, node_z, cell_nodes, scheme_id: int):
    b = SWE2DBackend()
    b.build_mesh(node_x, node_y, node_z, cell_nodes)
    nc = b.n_cells
    b.initialize(
        h0=np.full(nc, 0.05, dtype=np.float64),
        n_mann=0.035, h_min=1e-4, cfl=0.45, dt_max=0.5,
        gpu_diag_sync_interval_steps=1,
        spatial_discretization=scheme_id,
    )
    for _ in range(N_STEPS):
        b.step(-1.0)
    t0 = time.perf_counter()
    for _ in range(N_STEPS):
        b.step(-1.0)
    elapsed = (time.perf_counter() - t0) * 1000.0
    per_step = elapsed / N_STEPS
    cells_s = nc * 1000.0 / per_step
    dt = b._last_diag.get("dt", 0)
    b.destroy()
    return per_step, cells_s, dt


def main():
    # Structured mesh ~100K
    print("Building structured mesh...")
    nx = ny = 224
    sx = sy = 2240.0
    node_x, node_y, node_z, cell_nodes = _make_rect_mesh(nx, ny, sx, sy)
    nc = int(cell_nodes.size // 3)

    print(f"\n{'Scheme':<25} {'ms/step':>10} {'cells/s':>12} {'dt':>8}")
    print("-" * 60)
    for sid, sname in SCHEMES:
        ms, cps, dt = benchmark("structured", node_x, node_y, node_z, cell_nodes, sid)
        print(f"{sname:<25} {ms:>8.2f}ms  {cps:>10.0f}  {dt:>8.5f}")


if __name__ == "__main__":
    main()
