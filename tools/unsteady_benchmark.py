#!/usr/bin/env python3
"""Benchmark unsteady preprocessing and solve runtime from a GeoPackage.

This utility avoids PyQGIS/geopandas dependencies by reading the GeoPackage
with sqlite3 and parsing the geometry blobs directly.

Examples
--------
python3 tools/unsteady_benchmark.py --gpkg unsteady_example/unsteady_example.gpkg
python3 tools/unsteady_benchmark.py --gpkg /path/model.gpkg --dt 30 --t-end 7200 --runs 3
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import os
import sqlite3
import struct
import sys
import time
from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backwater_model import CrossSection, ModelInput
from unsteady_model import HydrographBC, UnsteadyParams, get_native_solver_runtime, run_unsteady, _build_hydraulic_tables


@dataclass
class BoundaryInfo:
    flow_cfs: Optional[float]
    boundary_type: str
    boundary_value: float


@dataclass
class DecompositionEstimate:
    short_steps: int
    long_steps: int
    short_avg_s: float
    long_avg_s: float
    estimated_startup_s: float
    estimated_per_step_s: float


@dataclass
class BackendRunSummary:
    mode: str
    run_times: List[float]
    last_results: object
    runtime_stats: Dict[str, object]


def _parse_gpkg_linestring_xyz(blob: bytes) -> List[Tuple[float, float, float]]:
    """Parse a GeoPackage geometry blob into LineString XYZ points."""
    if not blob or len(blob) < 8 or blob[0:2] != b"GP":
        return []

    flags = blob[3]
    envelope_indicator = (flags >> 1) & 0x07
    envelope_sizes = [0, 32, 48, 48, 64]
    envelope_bytes = envelope_sizes[envelope_indicator] if envelope_indicator < len(envelope_sizes) else 0

    wkb = blob[8 + envelope_bytes :]
    if len(wkb) < 9:
        return []

    bo = "<" if wkb[0] == 1 else ">"
    wkb_type = struct.unpack_from(bo + "I", wkb, 1)[0]
    geom_base = wkb_type % 1000 if wkb_type >= 1000 else wkb_type
    if geom_base != 2:  # not a LineString
        return []

    has_z = wkb_type in (1001, 1002, 3001, 3002) or wkb_type > 1000
    has_m = wkb_type in (2001, 2002, 3001, 3002)
    dims = 2 + int(has_z) + int(has_m)

    n_pts = struct.unpack_from(bo + "I", wkb, 5)[0]
    out = []
    off = 9
    for _ in range(n_pts):
        vals = struct.unpack_from(bo + ("d" * dims), wkb, off)
        off += 8 * dims
        x, y = vals[0], vals[1]
        z = vals[2] if has_z else 0.0
        out.append((float(x), float(y), float(z)))
    return out


def _xyz_to_station_elevation(points_xyz: List[Tuple[float, float, float]]) -> List[Tuple[float, float]]:
    if not points_xyz:
        return []

    out = [(0.0, points_xyz[0][2])]
    for idx in range(1, len(points_xyz)):
        x0, y0, _ = points_xyz[idx - 1]
        x1, y1, z1 = points_xyz[idx]
        ds = math.hypot(x1 - x0, y1 - y0)
        out.append((out[-1][0] + ds, z1))
    return out


def _safe_float(value, default=0.0) -> float:
    try:
        val = float(value)
        if math.isnan(val):
            return float(default)
        return val
    except Exception:
        return float(default)


def _station_sort_key(section: CrossSection) -> float:
    text = str(section.river_station or "").strip()
    if not text:
        return float("-inf")
    try:
        return float(text)
    except Exception:
        return float("-inf")


def _load_model_from_gpkg_sqlite(path: str) -> Tuple[ModelInput, BoundaryInfo]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        geom_row = conn.execute(
            "SELECT column_name FROM gpkg_geometry_columns WHERE table_name='cross_sections'"
        ).fetchone()
        if geom_row is None:
            raise ValueError("GeoPackage missing gpkg_geometry_columns entry for cross_sections")
        geom_col = str(geom_row[0])

        rows = conn.execute(f"SELECT *, {geom_col} as _geom_blob FROM cross_sections").fetchall()
        sections = []
        for row in rows:
            geom = _xyz_to_station_elevation(_parse_gpkg_linestring_xyz(bytes(row["_geom_blob"])))
            if not geom:
                continue
            sections.append(
                CrossSection(
                    river_station=str(row["river_station"] or ""),
                    geometry=geom,
                    left_bank_station=_safe_float(row["left_bank_station"], 0.0),
                    right_bank_station=_safe_float(row["right_bank_station"], 0.0),
                    n_lob=_safe_float(row["n_lob"], 0.035),
                    n_ch=_safe_float(row["n_ch"], 0.035),
                    n_rob=_safe_float(row["n_rob"], 0.035),
                    contraction_coeff=_safe_float(row["contraction_coeff"], 0.1),
                    expansion_coeff=_safe_float(row["expansion_coeff"], 0.3),
                    L_lob_to_next=_safe_float(row["L_lob_to_next"], 0.0),
                    L_ch_to_next=_safe_float(row["L_ch_to_next"], 0.0),
                    L_rob_to_next=_safe_float(row["L_rob_to_next"], 0.0),
                )
            )

        if len(sections) < 2:
            raise ValueError("Need at least two valid cross sections")

        boundary_type = "normal_depth"
        boundary_value = 0.001
        flow_cfs = None
        try:
            b_row = conn.execute(
                "SELECT boundary_type, boundary_value, flow_cfs FROM boundary_conditions LIMIT 1"
            ).fetchone()
            if b_row is not None:
                boundary_type = str(b_row["boundary_type"] or boundary_type)
                boundary_value = _safe_float(b_row["boundary_value"], boundary_value)
                if b_row["flow_cfs"] is not None:
                    flow_cfs = _safe_float(b_row["flow_cfs"], 0.0)
        except Exception:
            pass
    finally:
        conn.close()

    model = ModelInput(
        flow_cfs=float(flow_cfs if flow_cfs is not None else 100.0),
        flow_change=None,
        boundary_condition="normal_depth" if boundary_type == "normal_depth" else "known_wse",
        boundary_value=float(boundary_value),
        sections=sections,
    )
    return model, BoundaryInfo(flow_cfs=flow_cfs, boundary_type=boundary_type, boundary_value=boundary_value)


def _run_solver_silently(model: ModelInput, hydro: HydrographBC, params: UnsteadyParams):
    # Suppress legacy debug prints from steady helpers so benchmark output is clean.
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        return run_unsteady(model, hydro, params)


@contextlib.contextmanager
def _backend_mode(mode: str):
    old = os.environ.get("BACKWATER_USE_CPP_SOLVER")
    if mode == "native":
        os.environ["BACKWATER_USE_CPP_SOLVER"] = "1"
    elif mode == "python":
        os.environ["BACKWATER_USE_CPP_SOLVER"] = "0"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("BACKWATER_USE_CPP_SOLVER", None)
        else:
            os.environ["BACKWATER_USE_CPP_SOLVER"] = old


def _run_solve_benchmark(
    model: ModelInput,
    args: argparse.Namespace,
    t_end_s: float,
    runs: int,
    backend_mode: str,
) -> BackendRunSummary:
    hydro = HydrographBC(
        times=[0.0, float(t_end_s)],
        values=[float(args.q), float(args.q)],
        bc_type="flow",
        label="benchmark_const",
    )

    params = UnsteadyParams(
        dt=float(args.dt),
        t_end=float(t_end_s),
        theta=float(args.theta),
        max_iter=int(args.max_iter),
        tol=float(args.tol),
        output_interval=1,
        downstream_bc=str(args.ds_bc),
        downstream_value=float(args.ds_value),
        precompute_hydraulic_tables=not bool(args.no_precompute),
        hydraulic_table_dz=float(args.table_dz),
        hydraulic_table_padding=float(args.table_pad),
        debug_capture=False,
    )

    run_times: List[float] = []
    last_results = None
    runtime_stats: Dict[str, object] = {}
    with _backend_mode(backend_mode):
        for _ in range(max(1, int(runs))):
            tr0 = time.perf_counter()
            last_results = _run_solver_silently(model, hydro, params)
            run_times.append(time.perf_counter() - tr0)
        runtime_stats = get_native_solver_runtime()
    return BackendRunSummary(
        mode=backend_mode,
        run_times=run_times,
        last_results=last_results,
        runtime_stats=runtime_stats,
    )


def _estimate_decomposition(
    model: ModelInput,
    args: argparse.Namespace,
    base_runs: int,
    backend_mode: str,
) -> DecompositionEstimate:
    long_steps = max(2, int(round(float(args.t_end) / float(args.dt))))
    short_steps = max(1, int(args.decompose_short_steps))
    if short_steps >= long_steps:
        short_steps = max(1, long_steps // 2)

    t_short = short_steps * float(args.dt)
    t_long = long_steps * float(args.dt)

    short_summary = _run_solve_benchmark(model, args, t_end_s=t_short, runs=base_runs, backend_mode=backend_mode)
    long_summary = _run_solve_benchmark(model, args, t_end_s=t_long, runs=base_runs, backend_mode=backend_mode)

    short_times = short_summary.run_times
    long_times = long_summary.run_times

    short_avg = sum(short_times) / len(short_times)
    long_avg = sum(long_times) / len(long_times)
    step_delta = max(1, long_steps - short_steps)

    per_step = max(0.0, (long_avg - short_avg) / float(step_delta))
    startup = max(0.0, short_avg - per_step * float(short_steps))

    return DecompositionEstimate(
        short_steps=short_steps,
        long_steps=long_steps,
        short_avg_s=short_avg,
        long_avg_s=long_avg,
        estimated_startup_s=startup,
        estimated_per_step_s=per_step,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark unsteady preprocessing and solve runtime")
    parser.add_argument("--gpkg", required=True, help="Path to model GeoPackage")
    parser.add_argument("--dt", type=float, default=60.0, help="Time step (s)")
    parser.add_argument("--t-end", type=float, default=3600.0, help="Simulation duration (s)")
    parser.add_argument("--theta", type=float, default=0.6, help="Preissmann theta")
    parser.add_argument("--q", type=float, default=100.0, help="Constant upstream flow (cfs)")
    parser.add_argument("--ds-bc", default="normal_depth", choices=("normal_depth", "stage"), help="Downstream BC type")
    parser.add_argument("--ds-value", type=float, default=0.001, help="Downstream BC value (S0 or stage)")
    parser.add_argument("--table-dz", type=float, default=0.01, help="Hydraulic table dz (ft)")
    parser.add_argument("--table-pad", type=float, default=5.0, help="Hydraulic table padding (ft)")
    parser.add_argument("--max-iter", type=int, default=4, help="Max nonlinear inner iterations")
    parser.add_argument("--tol", type=float, default=1e-4, help="Inner-iteration convergence tolerance")
    parser.add_argument("--runs", type=int, default=3, help="Repeated solve runs for average timing")
    parser.add_argument("--mode", choices=("full", "decompose", "both"), default="both", help="Benchmark mode")
    parser.add_argument("--decompose-short-steps", type=int, default=10, help="Short run length in steps for decomposition mode")
    parser.add_argument("--no-precompute", action="store_true", help="Disable hydraulic table precompute during solve")
    parser.add_argument("--backend", choices=("python", "native", "compare"), default="python", help="Solver backend mode for benchmark runs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dt <= 0.0 or args.t_end <= 0.0 or args.tol <= 0.0:
        print("dt, t-end, and tol must be positive", file=sys.stderr)
        return 2
    if args.max_iter < 1:
        print("max-iter must be at least 1", file=sys.stderr)
        return 2

    t0 = time.perf_counter()
    model, boundary = _load_model_from_gpkg_sqlite(args.gpkg)
    t_load = time.perf_counter() - t0

    sections_us_to_ds = sorted(model.sections, key=_station_sort_key, reverse=True)

    t1 = time.perf_counter()
    _ = _build_hydraulic_tables(sections_us_to_ds, dz=float(args.table_dz), padding=float(args.table_pad))
    t_pre = time.perf_counter() - t1

    backend_summaries: List[BackendRunSummary] = []

    if args.mode in ("full", "both"):
        backend_modes = ["python", "native"] if args.backend == "compare" else [args.backend]
        for backend_mode in backend_modes:
            backend_summaries.append(
                _run_solve_benchmark(
                    model,
                    args,
                    t_end_s=float(args.t_end),
                    runs=max(1, int(args.runs)),
                    backend_mode=backend_mode,
                )
            )

    print("Unsteady Benchmark")
    print(f"  GPKG: {args.gpkg}")
    print(f"  Sections: {len(model.sections)}")
    print(f"  Boundary (from file): type={boundary.boundary_type}, value={boundary.boundary_value}, flow_cfs={boundary.flow_cfs}")
    print(f"  Runtime config: dt={args.dt}, t_end={args.t_end}, theta={args.theta}, q={args.q}")
    print(f"  Nonlinear config: max_iter={args.max_iter}, tol={args.tol}")
    print(f"  Table config: precompute={not args.no_precompute}, dz={args.table_dz}, pad={args.table_pad}")
    print(f"  Backend mode: {args.backend}")
    print(f"  Load time: {t_load:.3f} s")
    print(f"  Preprocess-only time: {t_pre:.3f} s")
    if backend_summaries:
        steps = max(1, int(round(args.t_end / args.dt)))
        for summary in backend_summaries:
            avg_run = sum(summary.run_times) / len(summary.run_times)
            print(f"\n  Solve backend: {summary.mode}")
            print(f"    Solve runs: {len(summary.run_times)}")
            print(f"    Solve avg: {avg_run:.3f} s")
            print(f"    Solve min/max: {min(summary.run_times):.3f} / {max(summary.run_times):.3f} s")
            print(f"    Timesteps: {steps}")
            print(f"    Timesteps/sec (avg): {steps / avg_run:.2f}")
            if summary.last_results is not None:
                print(f"    Output steps: {summary.last_results.n_output_times}")
            print(
                "    Native runtime: "
                f"timestep_success={summary.runtime_stats.get('native_timestep_success_count', 0)}, "
                f"timestep_fallback={summary.runtime_stats.get('native_timestep_fallback_count', 0)}, "
                f"assembly_success={summary.runtime_stats.get('native_assembly_success_count', 0)}, "
                f"assembly_fallback={summary.runtime_stats.get('native_assembly_fallback_count', 0)}, "
                f"damping_success={summary.runtime_stats.get('native_damping_success_count', 0)}, "
                f"damping_fallback={summary.runtime_stats.get('native_damping_fallback_count', 0)}, "
                f"solve_success={summary.runtime_stats.get('native_success_count', 0)}, "
                f"solve_fallback={summary.runtime_stats.get('native_fallback_count', 0)}"
            )
            timestep_error = str(summary.runtime_stats.get('last_timestep_fallback_error', '') or '')
            assembly_error = str(summary.runtime_stats.get('last_assembly_fallback_error', '') or '')
            damping_error = str(summary.runtime_stats.get('last_damping_fallback_error', '') or '')
            solve_error = str(summary.runtime_stats.get('last_fallback_error', '') or '')
            if timestep_error:
                print(f"    Native timestep fallback error: {timestep_error}")
            if assembly_error:
                print(f"    Native assembly fallback error: {assembly_error}")
            if damping_error:
                print(f"    Native damping fallback error: {damping_error}")
            if solve_error:
                print(f"    Native solve fallback error: {solve_error}")

        if len(backend_summaries) == 2:
            py_avg = sum(backend_summaries[0].run_times) / len(backend_summaries[0].run_times)
            native_avg = sum(backend_summaries[1].run_times) / len(backend_summaries[1].run_times)
            if native_avg > 0.0:
                print(f"\n  Compare speedup (python/native): {py_avg / native_avg:.2f}x")

    if args.mode in ("decompose", "both"):
        backend_modes = ["python", "native"] if args.backend == "compare" else [args.backend]
        print("\nDecomposition Mode")
        for backend_mode in backend_modes:
            est = _estimate_decomposition(model, args, base_runs=max(1, int(args.runs)), backend_mode=backend_mode)
            print(f"  Backend: {backend_mode}")
            print(f"    Short run: {est.short_steps} steps, avg {est.short_avg_s:.3f} s")
            print(f"    Long run: {est.long_steps} steps, avg {est.long_avg_s:.3f} s")
            print(f"    Estimated startup overhead: {est.estimated_startup_s:.3f} s")
            print(f"    Estimated per-step time: {est.estimated_per_step_s:.5f} s")
            if est.long_avg_s > 0.0:
                startup_pct = 100.0 * est.estimated_startup_s / est.long_avg_s
                print(f"    Startup share of long run: {startup_pct:.1f}%")
            if est.estimated_per_step_s > 0.0:
                print(f"    Estimated timestep throughput: {1.0 / est.estimated_per_step_s:.2f} steps/s")

    print("\nBenchmark command (copy/paste)")
    print(
        "python3 tools/unsteady_benchmark.py "
        f"--gpkg {args.gpkg} --dt {args.dt} --t-end {args.t_end} --theta {args.theta} "
        f"--max-iter {args.max_iter} --tol {args.tol} "
        f"--q {args.q} --ds-bc {args.ds_bc} --ds-value {args.ds_value} "
        f"--table-dz {args.table_dz} --table-pad {args.table_pad} --runs {max(1, int(args.runs))} "
        f"--mode {args.mode} --decompose-short-steps {max(1, int(args.decompose_short_steps))} --backend {args.backend}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
