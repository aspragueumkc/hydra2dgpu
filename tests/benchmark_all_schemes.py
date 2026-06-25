"""Benchmark all spatial schemes at multiple mesh sizes.

Usage:
    python tests/benchmark_all_schemes.py
"""
from __future__ import annotations

import sys
import time
import numpy as np

sys.path.insert(0, ".")

from swe2d.runtime.backend import SWE2DBackend
from tests._swe2d_test_helpers import _make_rect_mesh, _make_gmsh_triangle_mesh

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
    return per_step, cells_s, dt, nc


def zb_flat(x, y):
    return 10.0 - 0.005 * x - 0.003 * y


def main():
    meshes = []

    # Structured ~100K
    print("Building structured 100K...")
    nx = ny = 224
    node_x, node_y, node_z, cell_nodes = _make_rect_mesh(nx, ny, 2240.0, 2240.0)
    meshes.append(("Structured 100K", node_x, node_y, node_z, cell_nodes))

    # GMSH ~150K
    print("Building GMSH 150K (~mesh-size 8)...")
    d = _make_gmsh_triangle_mesh(2000.0, 2000.0, 8.0, zb_func=zb_flat)
    meshes.append(("GMSH 150K", d[0], d[1], d[2], d[3]))

    # GMSH ~500K
    print("Building GMSH 500K (~mesh-size 4.5)...")
    d = _make_gmsh_triangle_mesh(3000.0, 3000.0, 6.0, zb_func=zb_flat)
    meshes.append(("GMSH 500K", d[0], d[1], d[2], d[3]))

    # GMSH ~1M
    print("Building GMSH 1M (~mesh-size 4)...")
    d = _make_gmsh_triangle_mesh(4000.0, 4000.0, 5.0, zb_func=zb_flat)
    meshes.append(("GMSH 1M", d[0], d[1], d[2], d[3]))

    # GMSH ~2M
    print("Building GMSH 2M (~mesh-size 3.5)...")
    d = _make_gmsh_triangle_mesh(5000.0, 5000.0, 5.0, zb_func=zb_flat)
    meshes.append(("GMSH 2M", d[0], d[1], d[2], d[3]))

    for mname, nx, ny, nz, cn in meshes:
        nc = int(cn.size // 3)
        print(f"\n=== {mname} ({nc} cells) ===")
        print(f"{'Scheme':<25} {'ncells':>8} {'ms/step':>10} {'cells/s':>12} {'dt':>8}")
        print("-" * 68)
        for sid, sname in SCHEMES:
            ms, cps, dt, actual_nc = benchmark(mname, nx, ny, nz, cn, sid)
            print(f"{sname:<25} {actual_nc:>8} {ms:>8.2f}ms  {cps:>10.0f}  {dt:>8.5f}")


if __name__ == "__main__":
    main()
