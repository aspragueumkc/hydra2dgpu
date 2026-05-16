#!/usr/bin/env python3
"""A/B benchmark for SWE2D GPU spatial schemes under RK5.

Compares schemes 3/4/5 on an unstructured gmsh dam-break case and reports:
- wall runtime
- step count
- L_inf depth error against Stoker 1D reference on center strip

Usage:
  PYTHONPATH="$PWD:$PWD/build" python3 tools/swe2d_spatial_ab_benchmark.py
  PYTHONPATH="$PWD:$PWD/build" python3 tools/swe2d_spatial_ab_benchmark.py --size 35 --t-end 10
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import List

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.test_swe2d_dambreak import _make_rect_mesh, stoker_dam_break
from tests.test_swe2d_unstructured import _build_mesh, _make_gmsh_triangle_mesh


def _load_module():
    try:
        import backwater_swe2d

        return backwater_swe2d
    except Exception as exc:
        raise RuntimeError(f"Could not import backwater_swe2d: {exc}")


@dataclass
class BenchResult:
    spatial_scheme: int
    temporal_order: int
    steps: int
    wall_s: float
    dt_mean: float
    h_linf: float
    gpu_active: bool


def _run_structured_case(mod, spatial_scheme: int, temporal_order: int, t_end: float) -> BenchResult:
    nx, ny = 100, 5
    lx, ly = 1000.0, 50.0
    h_l, h_r = 2.0, 0.5

    node_x, node_y, node_z, cell_nodes = _make_rect_mesh(nx, ny, lx, ly)
    mesh = mod.swe2d_build_mesh(
        node_x,
        node_y,
        node_z,
        cell_nodes,
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.float64),
    )

    n_cells = int(mod.swe2d_mesh_info(mesh)["n_cells"])
    nx_p1 = nx + 1
    cell_cx = np.zeros(n_cells)
    for ci in range(n_cells):
        row, col = divmod(ci // 2, nx)
        stride = nx_p1
        if ci % 2 == 0:
            nodes = [row * stride + col, row * stride + col + 1, (row + 1) * stride + col + 1]
        else:
            nodes = [row * stride + col, (row + 1) * stride + col + 1, (row + 1) * stride + col]
        cell_cx[ci] = np.mean(node_x[nodes])

    h0 = np.where(cell_cx <= lx / 2.0, h_l, h_r)

    solver = mod.swe2d_create_solver(
        mesh,
        h0.copy(),
        n_mann=0.0,
        cfl=0.45,
        dt_max=0.25,
        use_gpu=True,
        temporal_order=temporal_order,
        spatial_scheme=spatial_scheme,
    )

    t = 0.0
    steps = 0
    dt_accum = 0.0
    last_diag = {"gpu_active": False, "dt": 0.0}
    t0 = time.perf_counter()
    while t < t_end:
        last_diag = mod.swe2d_step(solver, -1.0)
        dt = float(last_diag.get("dt", 0.0))
        if dt <= 0.0:
            break
        t += dt
        dt_accum += dt
        steps += 1
    wall_s = time.perf_counter() - t0

    h, _, _ = mod.swe2d_get_state(solver)
    mod.swe2d_destroy(solver)

    mid_row = ny // 2
    start = mid_row * nx * 2
    end = start + nx * 2
    cx_strip = cell_cx[start:end]
    h_strip = h[start:end]

    x_shifted = cx_strip - lx / 2.0
    h_exact = stoker_dam_break(x_shifted, t_end, h_l, h_r)
    finite = np.isfinite(h_strip) & np.isfinite(h_exact)
    h_linf = float(np.max(np.abs(h_strip[finite] - h_exact[finite]))) if np.any(finite) else float("nan")

    return BenchResult(
        spatial_scheme=spatial_scheme,
        temporal_order=temporal_order,
        steps=steps,
        wall_s=wall_s,
        dt_mean=(dt_accum / max(steps, 1)),
        h_linf=h_linf,
        gpu_active=bool(last_diag.get("gpu_active", False)),
    )


def _run_unstructured_case(mod, spatial_scheme: int, temporal_order: int, size: float, t_end: float) -> BenchResult:
    lx, ly = 1000.0, 50.0
    h_l, h_r = 2.0, 0.5

    node_x, node_y, node_z, cell_nodes, cell_cx, cell_cy = _make_gmsh_triangle_mesh(lx, ly, size)
    mesh = _build_mesh(mod, node_x, node_y, node_z, cell_nodes)
    h0 = np.where(cell_cx <= lx / 2.0, h_l, h_r)

    solver = mod.swe2d_create_solver(
        mesh,
        h0.copy(),
        n_mann=0.0,
        cfl=0.45,
        dt_max=0.5,
        temporal_order=temporal_order,
        spatial_scheme=spatial_scheme,
        use_gpu=True,
    )

    t = 0.0
    steps = 0
    dt_accum = 0.0
    last_diag = {"gpu_active": False, "dt": 0.0}
    t0 = time.perf_counter()
    while t < t_end:
        last_diag = mod.swe2d_step(solver, -1.0)
        dt = float(last_diag.get("dt", 0.0))
        if dt <= 0.0:
            break
        t += dt
        dt_accum += dt
        steps += 1
    wall_s = time.perf_counter() - t0

    h, _, _ = mod.swe2d_get_state(solver)
    mod.swe2d_destroy(solver)

    mid_y = ly / 2.0
    strip_tol = ly * 0.15
    mask = np.abs(cell_cy - mid_y) < strip_tol
    cx_strip = cell_cx[mask]
    h_strip = h[mask]

    order = np.argsort(cx_strip)
    cx_strip = cx_strip[order]
    h_strip = h_strip[order]

    x_shifted = cx_strip - lx / 2.0
    h_exact = stoker_dam_break(x_shifted, t_end, h_l, h_r) if h_strip.size else np.empty(0)
    if h_strip.size:
        finite = np.isfinite(h_strip) & np.isfinite(h_exact)
        if np.any(finite):
            h_linf = float(np.max(np.abs(h_strip[finite] - h_exact[finite])))
        else:
            h_linf = float("nan")
    else:
        h_linf = float("nan")

    return BenchResult(
        spatial_scheme=spatial_scheme,
        temporal_order=temporal_order,
        steps=steps,
        wall_s=wall_s,
        dt_mean=(dt_accum / max(steps, 1)),
        h_linf=h_linf,
        gpu_active=bool(last_diag.get("gpu_active", False)),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mesh-mode",
        type=str,
        default="structured",
        choices=["structured", "unstructured"],
        help="benchmark mesh mode",
    )
    ap.add_argument("--size", type=float, default=40.0, help="gmsh target size (m)")
    ap.add_argument("--t-end", type=float, default=10.0, help="simulation end time (s)")
    ap.add_argument("--temporal-order", type=int, default=6, help="temporal order (6=GRAPH_SAFE_RK5)")
    ap.add_argument(
        "--schemes",
        type=str,
        default="3,4,5",
        help="comma-separated spatial schemes to benchmark",
    )
    args = ap.parse_args()

    mod = _load_module()
    if not bool(mod.swe2d_gpu_available()):
        raise RuntimeError("CUDA GPU is not available in backwater_swe2d")

    schemes: List[int] = [int(s.strip()) for s in str(args.schemes).split(",") if s.strip()]
    print("SWE2D Spatial A/B Benchmark (GPU)")
    print(
        f"  mesh_mode={args.mesh_mode} temporal_order={args.temporal_order} "
        f"size={args.size} t_end={args.t_end}"
    )
    print("  schemes=", schemes)
    print()

    results: List[BenchResult] = []
    for scheme in schemes:
        if args.mesh_mode == "structured":
            res = _run_structured_case(
                mod,
                spatial_scheme=scheme,
                temporal_order=int(args.temporal_order),
                t_end=float(args.t_end),
            )
        else:
            res = _run_unstructured_case(
                mod,
                spatial_scheme=scheme,
                temporal_order=int(args.temporal_order),
                size=float(args.size),
                t_end=float(args.t_end),
            )
        results.append(res)
        print(
            f"scheme={res.spatial_scheme} steps={res.steps:4d} wall={res.wall_s:7.3f}s "
            f"dt_mean={res.dt_mean:7.4f}s h_linf={res.h_linf:9.5f} gpu={res.gpu_active}"
        )

    if results:
        base = min(results, key=lambda r: r.wall_s)
        best_err = min(results, key=lambda r: r.h_linf)
        print()
        print(
            f"fastest: scheme={base.spatial_scheme} wall={base.wall_s:.3f}s | "
            f"best_error: scheme={best_err.spatial_scheme} h_linf={best_err.h_linf:.5f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
