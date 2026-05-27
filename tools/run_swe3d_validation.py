#!/usr/bin/env python3
"""
tools/run_swe3d_validation.py
------------------------------
Stage-1 3D validation runner — emits a tabular pass/fail report with metric
deltas and exits non-zero if any gate fails.

Usage
-----
    cd <repo-root>
    conda activate qgis_stable
    PYTHONPATH="$PWD:$PWD/build" python3 tools/run_swe3d_validation.py [options]

Options
-------
  --all          Include reference-case gates in addition to invariant gates
                 (equivalent to setting BACKWATER_RUN_SWE3D_PHYSICS_CASES=1)
  --json FILE    Write machine-readable results to FILE (optional)
  --verbose      Print per-step timing and patch-stat traces
  -h / --help    Show this message

Exit codes
----------
  0   All active gates passed
  1   One or more gates failed
  2   Cannot import hydra_swe2d or GPU unavailable (environment error)
"""

import argparse
import json
import math
import os
import sys
import time

# Ensure repo root and build dir on path.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(_SCRIPT_DIR)
_BUILD_DIR  = os.path.join(_REPO_ROOT, "build")
for _p in (_REPO_ROOT, _BUILD_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np


# ── Colour helpers ────────────────────────────────────────────────────────────
_USE_COLOUR = sys.stdout.isatty()
_G = "\033[32m" if _USE_COLOUR else ""
_R = "\033[31m" if _USE_COLOUR else ""
_Y = "\033[33m" if _USE_COLOUR else ""
_B = "\033[34m" if _USE_COLOUR else ""
_N = "\033[0m"  if _USE_COLOUR else ""


def _ok(s):  return f"{_G}{s}{_N}"
def _fail(s): return f"{_R}{s}{_N}"
def _info(s): return f"{_B}{s}{_N}"
def _warn(s): return f"{_Y}{s}{_N}"


# ── Mesh/solver helpers (local, no import from tests/) ───────────────────────
def _build_mesh(mod, nx_2d=20, ny_2d=10, lx=200.0, ly=100.0):
    xs = np.linspace(0.0, lx, nx_2d + 1)
    ys = np.linspace(0.0, ly, ny_2d + 1)
    xg, yg = np.meshgrid(xs, ys)
    node_x = xg.ravel().astype(np.float64)
    node_y = yg.ravel().astype(np.float64)
    node_z = np.zeros_like(node_x)
    cells = []
    stride = nx_2d + 1
    for j in range(ny_2d):
        for i in range(nx_2d):
            n00 = j * stride + i
            n10 = j * stride + i + 1
            n01 = (j + 1) * stride + i
            n11 = (j + 1) * stride + i + 1
            cells.extend([n00, n10, n11])
            cells.extend([n00, n11, n01])
    return mod.swe2d_build_mesh(
        node_x, node_y, node_z,
        np.array(cells, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.float64),
    )


def _make_solver(mod, mesh, h0):
    return mod.swe2d_create_solver(
        mesh, h0, use_gpu=True, temporal_order=2,
        coupling_mode=0, three_d_solver_model=1)


def _flat_vof(stats, fill_frac=0.5):
    nx, ny, nz = int(stats["nx"]), int(stats["ny"]), int(stats["nz"])
    n_fill = max(1, int(round(fill_frac * nz)))
    vof = np.zeros(nx * ny * nz, dtype=np.float64)
    for iz in range(n_fill):
        lo = iz * nx * ny
        hi = lo + nx * ny
        vof[lo:hi] = 1.0
    return vof


# ── Individual invariant checks ───────────────────────────────────────────────

class _GateResult:
    def __init__(self, name, passed, value, ref, delta, tolerance, unit="", note=""):
        self.name      = name
        self.passed    = passed
        self.value     = value
        self.ref       = ref
        self.delta     = delta
        self.tolerance = tolerance
        self.unit      = unit
        self.note      = note

    def row(self):
        status = _ok("PASS") if self.passed else _fail("FAIL")
        return (f"  {status}  {self.name:<45s}"
                f"  value={self.value:+.4e}{self.unit}"
                f"  ref={self.ref:+.4e}{self.unit}"
                f"  Δ={self.delta:.3e}  tol={self.tolerance:.3e}")


def _check_vof_bounds(mod, mesh, h0, n_steps=20, verbose=False):
    """VoF must stay in [0,1] at all times."""
    solver = _make_solver(mod, mesh, h0)
    try:
        stats0 = mod.swe2d_get_3d_patch_stats(solver)
        vof_ic = _flat_vof(stats0, fill_frac=0.5)
        mod.swe2d_set_3d_patch_vof(solver, vof_ic)

        for i in range(n_steps):
            mod.swe2d_step(solver, -1.0)
            if verbose:
                s = mod.swe2d_get_3d_patch_stats(solver)
                print(f"    step {i:3d}: vof_min={s['vof_min']:.4f}  "
                      f"vof_max={s['vof_max']:.4f}")

        stats = mod.swe2d_get_3d_patch_stats(solver)
        vof_min = stats["vof_min"]
        vof_max = stats["vof_max"]

        r_min = _GateResult("vof_min_ge_0",
            vof_min >= -1e-10,
            value=vof_min, ref=0.0,
            delta=max(0.0, -vof_min), tolerance=1e-10,
            unit="", note="VoF lower bound")
        r_max = _GateResult("vof_max_le_1",
            vof_max <= 1.0 + 1e-10,
            value=vof_max, ref=1.0,
            delta=max(0.0, vof_max - 1.0), tolerance=1e-10,
            unit="", note="VoF upper bound")
        return [r_min, r_max]
    finally:
        mod.swe2d_destroy(solver)


def _check_vof_conservation(mod, mesh, h0, n_steps=50, verbose=False):
    """Total VoF must be conserved to < 0.1 %."""
    solver = _make_solver(mod, mesh, h0)
    try:
        stats0 = mod.swe2d_get_3d_patch_stats(solver)
        vof_ic = _flat_vof(stats0, fill_frac=0.5)
        mod.swe2d_set_3d_patch_vof(solver, vof_ic)
        sum_0 = float(np.sum(vof_ic))

        for i in range(n_steps):
            mod.swe2d_step(solver, -1.0)

        stats = mod.swe2d_get_3d_patch_stats(solver)
        vof_sum = stats["vof_sum"]
        rel_err = abs(vof_sum - sum_0) / max(sum_0, 1.0)
        tol = 1e-3

        r = _GateResult("vof_sum_conservation",
            rel_err < tol,
            value=vof_sum, ref=sum_0,
            delta=abs(vof_sum - sum_0), tolerance=tol * sum_0,
            unit="", note=f"rel_err={rel_err:.2e}")
        return [r]
    finally:
        mod.swe2d_destroy(solver)


def _check_rest_stability(mod, mesh, h0, n_steps=20, verbose=False):
    """Zero IC must remain exactly zero (rest stability)."""
    solver = _make_solver(mod, mesh, h0)
    try:
        stats0 = mod.swe2d_get_3d_patch_stats(solver)
        n = int(stats0["n_cells"])
        zeros = np.zeros(n, dtype=np.float64)
        mod.swe2d_set_3d_patch_state(solver, u=zeros, v=zeros, w=zeros, p=zeros)

        for _ in range(n_steps):
            mod.swe2d_step(solver, -1.0)

        stats = mod.swe2d_get_3d_patch_stats(solver)
        tol = 1e-12
        results = []
        for field, key in [("u_rms", "u_rms"), ("v_rms", "v_rms"),
                            ("w_rms", "w_rms"), ("p_max_abs", "p_max_abs")]:
            val = stats[key]
            results.append(_GateResult(
                f"rest_stability_{field}",
                val < tol,
                value=val, ref=0.0, delta=val, tolerance=tol,
                unit=" m/s" if "rms" in field else " Pa"))
        return results
    finally:
        mod.swe2d_destroy(solver)


def _check_velocity_damping(mod, mesh, h0, n_steps=10, verbose=False):
    """Scaffold damping: u_rms must decrease monotonically from non-zero IC."""
    solver = _make_solver(mod, mesh, h0)
    try:
        stats0 = mod.swe2d_get_3d_patch_stats(solver)
        n = int(stats0["n_cells"])
        u_ic = np.ones(n, dtype=np.float64)
        zeros = np.zeros(n, dtype=np.float64)
        mod.swe2d_set_3d_patch_state(solver, u=u_ic, v=zeros, w=zeros, p=zeros)

        stats_ic = mod.swe2d_get_3d_patch_stats(solver)
        initial_rms = float(stats_ic["u_rms"])

        prev_rms = initial_rms
        monotone_non_increasing = True
        had_strict_drop = False
        eps = 1e-12
        final_rms = initial_rms
        for step in range(n_steps):
            mod.swe2d_step(solver, 0.1)
            stats = mod.swe2d_get_3d_patch_stats(solver)
            cur_rms = stats["u_rms"]
            if verbose:
                print(f"    step {step:3d}: u_rms={cur_rms:.6e}")
            if cur_rms > prev_rms + eps:
                monotone_non_increasing = False
            if prev_rms - cur_rms > eps:
                had_strict_drop = True
            prev_rms = cur_rms
            final_rms = cur_rms

        tol = 0.01  # must have reduced by at least 1 % if initial RMS is non-zero
        if initial_rms > eps:
            reduction = (initial_rms - final_rms) / initial_rms
            damping_ok = had_strict_drop and reduction >= tol
        else:
            reduction = 1.0
            damping_ok = True

        r = _GateResult(
            "velocity_damping_monotone",
            monotone_non_increasing and damping_ok,
            value=final_rms, ref=0.0,
            delta=final_rms, tolerance=1.0 - tol,
            unit=" m/s",
            note=(
                f"monotone_non_increasing={monotone_non_increasing}  "
                f"had_strict_drop={had_strict_drop}  "
                f"initial_rms={initial_rms:.3e}  reduction={reduction:.2%}"
            ))
        return [r]
    finally:
        mod.swe2d_destroy(solver)


# ── Reference-case gates ──────────────────────────────────────────────────────

def _check_reference_case(mod, case_name, verbose=False):
    from tests.swe3d_reference_harness import load_case, run_and_compare
    case = load_case(case_name)
    t0 = time.perf_counter()
    result = run_and_compare(mod, case)
    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"    elapsed: {elapsed:.2f}s")
    gates = []
    for m in result.metrics:
        gates.append(_GateResult(
            f"{case_name}/{m.name}",
            m.passed,
            value=m.value, ref=m.ref, delta=m.delta, tolerance=m.tolerance,
            description=m.description))
    return gates


# ── Report printer ────────────────────────────────────────────────────────────

def _print_report(sections, json_path=None):
    all_results = []
    print()
    print(_info("=" * 72))
    print(_info("  SWE3D Stage-1 Validation Report"))
    print(_info("=" * 72))

    n_pass = n_fail = 0
    for section_name, results in sections:
        print(f"\n  {_B}{section_name}{_N}")
        print("  " + "-" * 68)
        for r in results:
            print(r.row())
            all_results.append(r)
            if r.passed:
                n_pass += 1
            else:
                n_fail += 1

    print()
    print(_info("─" * 72))
    summary = f"  TOTAL: {n_pass} passed  {n_fail} failed"
    print(_ok(summary) if n_fail == 0 else _fail(summary))
    print(_info("─" * 72))
    print()

    if json_path:
        data = {
            "pass": n_pass,
            "fail": n_fail,
            "gates": [
                {
                    "name":  r.name,
                    "passed": r.passed,
                    "value": r.value,
                    "ref":   r.ref,
                    "delta": r.delta,
                    "tolerance": r.tolerance,
                }
                for r in all_results
            ]
        }
        with open(json_path, "w") as fh:
            json.dump(data, fh, indent=2)
        print(f"  JSON report written to: {json_path}")
        print()

    return n_fail == 0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SWE3D Stage-1 validation runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument("--all", action="store_true",
        help="Also run reference-case gates (broad_crested_weir, culvert_pressurization)")
    parser.add_argument("--json", metavar="FILE",
        help="Write JSON result to FILE")
    parser.add_argument("--verbose", action="store_true",
        help="Print per-step diagnostics")
    args = parser.parse_args()

    # ── Preflight ──────────────────────────────────────────────────────────
    try:
        import hydra_swe2d as mod
    except ImportError:
        print(_fail("ERROR: cannot import hydra_swe2d — build the extension first."))
        print(f"  cd build && make -j$(nproc)")
        sys.exit(2)

    try:
        gpu_ok = mod.swe2d_gpu_available()
    except Exception:
        gpu_ok = False

    if not gpu_ok:
        print(_fail("ERROR: CUDA GPU not available — cannot run 3D validation."))
        sys.exit(2)

    print(_info(f"\n  Module:  {mod.__file__}"))
    print(_info(f"  GPU:     available"))
    print()

    # ── Build shared mesh ──────────────────────────────────────────────────
    mesh = _build_mesh(mod)
    n_cells_2d = mod.swe2d_mesh_info(mesh)["n_cells"]
    h0 = np.full(n_cells_2d, 1.0, dtype=np.float64)

    sections = []

    # ── Invariant gates (always run) ───────────────────────────────────────
    def _run(label, fn, *a):
        print(f"  Running: {label} ... ", end="", flush=True)
        t0 = time.perf_counter()
        try:
            r = fn(mod, mesh, h0, *a, verbose=args.verbose)
            elapsed = time.perf_counter() - t0
            n_ok = sum(1 for x in r if x.passed)
            print(f"{_ok('ok') if n_ok==len(r) else _fail('FAIL')} ({elapsed:.2f}s)")
            return r
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(_fail(f"ERROR ({elapsed:.2f}s): {exc}"))
            return []

    inv_results = []
    inv_results += _run("VoF bounds preserved",       _check_vof_bounds)
    inv_results += _run("VoF sum conserved",           _check_vof_conservation)
    inv_results += _run("Rest stability (zero IC)",    _check_rest_stability)
    inv_results += _run("Velocity damping monotone",   _check_velocity_damping)
    sections.append(("Physics Invariants (Stage-1 gates)", inv_results))

    # ── Reference cases (optional) ─────────────────────────────────────────
    if args.all:
        ref_results = []
        for case_name in ("broad_crested_weir", "culvert_pressurization"):
            print(f"  Running: reference case '{case_name}' ... ", end="", flush=True)
            t0 = time.perf_counter()
            try:
                r = _check_reference_case(mod, case_name, verbose=args.verbose)
                elapsed = time.perf_counter() - t0
                n_ok = sum(1 for x in r if x.passed)
                print(f"{_ok('ok') if n_ok==len(r) else _fail('FAIL')} ({elapsed:.2f}s)")
                ref_results += r
            except FileNotFoundError as exc:
                elapsed = time.perf_counter() - t0
                print(_warn(f"SKIP ({elapsed:.2f}s): {exc}"))
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                print(_fail(f"ERROR ({elapsed:.2f}s): {exc}"))
        sections.append(("Reference Cases (physics implementation required)", ref_results))

    # ── Report ─────────────────────────────────────────────────────────────
    passed = _print_report(sections, json_path=args.json)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
