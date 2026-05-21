#!/usr/bin/env python3
"""Run repeatable diagnostics on the bundled unsteady example model.

This script focuses on the earliest timesteps at the most upstream section,
where startup transients and coarse timesteps are most likely to look suspect.

Usage:
    python3 tools/unsteady_example_diagnostics.py
    python3 tools/unsteady_example_diagnostics.py --gpkg unsteady_example/unsteady_example.gpkg
    python3 tools/unsteady_example_diagnostics.py --trace-case ramp0_300 --trace-dt 60 --trace-theta 0.6
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import hydra_1d as bm
from unsteady_model import HydrographBC, UnsteadyParams, run_unsteady


@dataclass
class CaseSummary:
    hydro: str
    dt: float
    theta: float
    n_output_times: int
    upstream_id: str
    early_wse_span: float
    early_q_span: float
    max_abs_dq_early: float
    sign_flips_dq_early: int
    second_section_negative_q: bool


def _build_hydrographs() -> Dict[str, HydrographBC]:
    return {
        "const100": HydrographBC(
            times=[0.0, 3600.0],
            values=[100.0, 100.0],
            bc_type="flow",
            label="Constant 100 cfs",
        ),
        "ramp0_300": HydrographBC(
            times=[0.0, 300.0, 900.0, 3600.0],
            values=[0.0, 300.0, 300.0, 300.0],
            bc_type="flow",
            label="Ramp 0 to 300 cfs",
        ),
        "pulse100_500_100": HydrographBC(
            times=[0.0, 300.0, 900.0, 1800.0, 3600.0],
            values=[100.0, 500.0, 500.0, 100.0, 100.0],
            bc_type="flow",
            label="Pulse 100 to 500 to 100 cfs",
        ),
    }


def _run_solver_silently(model, hydro: HydrographBC, params: UnsteadyParams):
    # Suppress legacy DEBUG prints emitted by steady helper code.
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        return run_unsteady(model, hydro, params)


def _count_sign_flips(values: Iterable[float]) -> int:
    signs: List[int] = []
    for value in values:
        if value > 0.0:
            signs.append(1)
        elif value < 0.0:
            signs.append(-1)
    flips = 0
    for idx in range(1, len(signs)):
        if signs[idx] != signs[idx - 1]:
            flips += 1
    return flips


def _summarize_case(model, hydro_name: str, hydro: HydrographBC, dt: float, theta: float) -> CaseSummary:
    params = UnsteadyParams(
        dt=dt,
        t_end=1800.0,
        theta=theta,
        output_interval=1,
        downstream_bc="normal_depth",
        downstream_value=0.003,
        debug_capture=False,
    )
    results = _run_solver_silently(model, hydro, params)
    upstream_idx = 0
    second_idx = 1 if results.n_sections > 1 else 0
    n_early = min(10, results.n_output_times)

    wse_early = [float(v) for v in results.wse[:n_early, upstream_idx]]
    q_early = [float(v) for v in results.q[:n_early, upstream_idx]]
    second_q_early = [float(v) for v in results.q[:n_early, second_idx]]

    dq_early = [q_early[idx] - q_early[idx - 1] for idx in range(1, len(q_early))]
    max_abs_dq = max((abs(v) for v in dq_early), default=0.0)

    return CaseSummary(
        hydro=hydro_name,
        dt=dt,
        theta=theta,
        n_output_times=int(results.n_output_times),
        upstream_id=str(results.section_ids[upstream_idx]),
        early_wse_span=(max(wse_early) - min(wse_early)) if wse_early else 0.0,
        early_q_span=(max(q_early) - min(q_early)) if q_early else 0.0,
        max_abs_dq_early=max_abs_dq,
        sign_flips_dq_early=_count_sign_flips(dq_early),
        second_section_negative_q=any(v < 0.0 for v in second_q_early),
    )


def _trace_case(model, hydro_name: str, hydro: HydrographBC, dt: float, theta: float, t_end: float) -> Dict[str, object]:
    params = UnsteadyParams(
        dt=dt,
        t_end=t_end,
        theta=theta,
        output_interval=1,
        downstream_bc="normal_depth",
        downstream_value=0.003,
        debug_capture=True,
        debug_frequency="computation",
    )
    results = _run_solver_silently(model, hydro, params)
    upstream_idx = 0
    second_idx = 1 if results.n_sections > 1 else 0

    rows = []
    n_rows = min(12, results.n_output_times)
    for idx in range(n_rows):
        rows.append(
            {
                "idx": idx,
                "time_s": float(results.times[idx]),
                "q_up": float(results.q[idx, upstream_idx]),
                "wse_up": float(results.wse[idx, upstream_idx]),
                "q_2": float(results.q[idx, second_idx]),
                "wse_2": float(results.wse[idx, second_idx]),
            }
        )

    debug_rows = []
    for record in (results.debug_records or [])[:10]:
        inner = record.get("inner_iterations") or []
        last = inner[-1] if inner else {}
        debug_rows.append(
            {
                "step": record.get("step"),
                "time_s": record.get("time_s"),
                "q_up": float(record.get("q", [0.0])[upstream_idx]),
                "wse_up": float(record.get("z", [0.0])[upstream_idx]),
                "max_abs_dQ_applied": last.get("max_abs_dQ_applied"),
                "max_abs_dz_applied": last.get("max_abs_dz_applied"),
                "linear_rhs_inf": last.get("linear_rhs_inf"),
            }
        )

    return {
        "hydro": hydro_name,
        "dt": dt,
        "theta": theta,
        "upstream_id": str(results.section_ids[upstream_idx]),
        "second_id": str(results.section_ids[second_idx]),
        "rows": rows,
        "debug_rows": debug_rows,
    }


def _print_summary_table(rows: List[CaseSummary]) -> None:
    print("hydro,dt,theta,n_out,upstream_id,early_wse_span,early_q_span,max_abs_dq_early,sign_flips_dq_early,second_section_negative_q")
    for row in rows:
        print(
            f"{row.hydro},{row.dt:.1f},{row.theta:.1f},{row.n_output_times},"
            f"{row.upstream_id},{row.early_wse_span:.6f},{row.early_q_span:.6f},"
            f"{row.max_abs_dq_early:.6f},{row.sign_flips_dq_early},"
            f"{int(row.second_section_negative_q)}"
        )


def _print_highlights(rows: List[CaseSummary]) -> None:
    if not rows:
        return
    worst_wse = max(rows, key=lambda row: row.early_wse_span)
    worst_dq = max(rows, key=lambda row: row.max_abs_dq_early)
    negative_second = [row for row in rows if row.second_section_negative_q]

    print("\nHighlights")
    print(
        "- Worst early upstream WSE span: "
        f"{worst_wse.hydro} dt={worst_wse.dt:.1f} theta={worst_wse.theta:.1f} "
        f"span={worst_wse.early_wse_span:.3f} ft"
    )
    print(
        "- Largest early upstream dQ jump: "
        f"{worst_dq.hydro} dt={worst_dq.dt:.1f} theta={worst_dq.theta:.1f} "
        f"max_abs_dq={worst_dq.max_abs_dq_early:.3f} cfs"
    )
    if negative_second:
        labels = ", ".join(
            f"{row.hydro}/dt={row.dt:.0f}/theta={row.theta:.1f}" for row in negative_second
        )
        print(f"- Cases with early negative flow at second section: {labels}")
    else:
        print("- No cases showed early negative flow at the second section.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unsteady example diagnostics")
    parser.add_argument(
        "--gpkg",
        default=os.path.join("unsteady_example", "unsteady_example.gpkg"),
        help="Path to the example GeoPackage",
    )
    parser.add_argument(
        "--trace-case",
        default="ramp0_300",
        choices=sorted(_build_hydrographs().keys()),
        help="Case to print a detailed early-time trace for",
    )
    parser.add_argument("--trace-dt", type=float, default=60.0, help="Detailed trace dt")
    parser.add_argument("--trace-theta", type=float, default=0.6, help="Detailed trace theta")
    parser.add_argument("--trace-t-end", type=float, default=600.0, help="Detailed trace duration")
    parser.add_argument(
        "--json",
        default="",
        help="Optional path to write summary and trace as JSON",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not os.path.isfile(args.gpkg):
        print(f"GeoPackage not found: {args.gpkg}", file=sys.stderr)
        return 2

    model = bm.load_from_geopackage(args.gpkg)
    hydros = _build_hydrographs()

    rows: List[CaseSummary] = []
    for hydro_name, hydro in hydros.items():
        for dt in (10.0, 30.0, 60.0):
            for theta in (0.6, 0.8, 1.0):
                rows.append(_summarize_case(model, hydro_name, hydro, dt, theta))

    _print_summary_table(rows)
    _print_highlights(rows)

    trace = _trace_case(
        model,
        args.trace_case,
        hydros[args.trace_case],
        args.trace_dt,
        args.trace_theta,
        args.trace_t_end,
    )

    print("\nDetailed trace")
    print(
        f"- hydro={trace['hydro']} dt={trace['dt']:.1f} theta={trace['theta']:.1f} "
        f"upstream_id={trace['upstream_id']} second_id={trace['second_id']}"
    )
    print("idx,time_s,q_up,wse_up,q_2,wse_2")
    for row in trace["rows"]:
        print(
            f"{row['idx']},{row['time_s']:.1f},{row['q_up']:.3f},{row['wse_up']:.3f},"
            f"{row['q_2']:.3f},{row['wse_2']:.3f}"
        )

    print("\nDetailed debug rows")
    print("step,time_s,q_up,wse_up,max_abs_dQ_applied,max_abs_dz_applied,linear_rhs_inf")
    for row in trace["debug_rows"]:
        print(
            f"{row['step']},{float(row['time_s']):.1f},{row['q_up']:.3f},{row['wse_up']:.3f},"
            f"{row['max_abs_dQ_applied']},{row['max_abs_dz_applied']},{row['linear_rhs_inf']}"
        )

    if args.json:
        payload = {
            "summary": [row.__dict__ for row in rows],
            "trace": trace,
        }
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nWrote JSON report: {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
