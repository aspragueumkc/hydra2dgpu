#!/usr/bin/env python3
"""
NVIDIA Compute Sanitizer test runner.

Wraps GPU test cases with compute-sanitizer to detect:
  --tool memcheck    : out-of-bounds, misaligned access, bank conflicts
  --tool racecheck   : global/shared memory race conditions
  --tool initcheck   : uninitialized device memory reads
  --tool synccheck   : barrier synchronization violations

Usage:
    python tools/run_compute_sanitizer.py                    # run all tools on all GPU tests
    python tools/run_compute_sanitizer.py --tool memcheck    # single tool
    python tools/run_compute_sanitizer.py --test dambreak    # single test
    python tools/run_compute_sanitizer.py --list             # list available tests

Requires:
    - NVIDIA GPU + CUDA toolkit (compute-sanitizer ships with CUDA 11.0+)
    - HYDRA2DGPU built with CUDA enabled
    - Test dependencies (numpy, gmsh, etc.)

Output:
    Results are saved to report_output/compute_sanitizer/<tool>/<test_name>.txt
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Test registry ─────────────────────────────────────────────────────────
GPU_TESTS = [
    # (display_name, module_path)
    ("dambreak", "tests.test_swe2d_gpu_dambreak"),
    ("lakerest", "tests.test_swe2d_gpu_lakerest"),
    ("nonorth_channel", "tests.test_swe2d_gpu_nonorth_channel"),
    ("hydraulics_suite", "tests.test_swe2d_gpu_hydraulics_suite"),
    ("unstructured", "tests.test_swe2d_gpu_unstructured"),
    ("unstructured_rain", "tests.test_swe2d_gpu_unstructured_rain"),
    ("weno5_convergence", "tests.test_swe2d_weno5_convergence"),
    ("structures", "tests.test_swe2d_gpu_structures"),
    ("drainage_network", "tests.test_swe2d_gpu_drainage_network"),
    ("coupling_kernel", "tests.test_swe2d_gpu_coupling_kernel"),
]
]

SANITIZER_TOOLS = ["memcheck", "racecheck", "initcheck", "synccheck"]

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = REPO_ROOT / "report_output" / "compute_sanitizer"


def _find_sanitizer() -> str | None:
    """Locate compute-sanitizer (prefer conda env, then system PATH)."""
    # Common install locations
    candidates = [
        # Conda CUDA toolkit
        os.path.expanduser("~/miniforge3/envs/qgis_stable/bin/compute-sanitizer"),
        os.path.expanduser("~/miniforge3/bin/compute-sanitizer"),
        # System CUDA on Linux
        "/usr/local/cuda/bin/compute-sanitizer",
        # Windows (CUDA toolkit default path)
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4\\bin\\compute-sanitizer.exe",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    # Fallback: PATH lookup
    import shutil
    return shutil.which("compute-sanitizer")


def _run_sanitizer(
    tool: str,
    test_name: str,
    test_module: str,
    *,
    extra_args: list[str] | None = None,
    timeout: int = 600,
) -> tuple[bool, str]:
    """Run a single compute-sanitizer session.

    Returns (passed, output_text).
    """
    sanitizer = _find_sanitizer()
    if not sanitizer:
        return False, "[ERROR] compute-sanitizer not found on PATH or common locations."

    out_dir = REPORT_DIR / tool
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{test_name}.txt"

    cmd = [
        sanitizer,
        f"--tool={tool}",
        *(["--log-file", str(out_path)] if tool != "racecheck" else []),
        sys.executable,
        "-m",
        "pytest",
        test_module,
        "-v",
        "--tb=short",
        "-x",  # stop on first failure
        *(extra_args or []),
    ]

    print(f"\n{'='*60}")
    print(f"  Tool: {tool}")
    print(f"  Test: {test_name}  ({test_module})")
    print(f"  Log:  {out_path}")
    print(f"{'='*60}\n")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout + "\n" + result.stderr

        # Write output regardless
        if tool == "racecheck":
            with open(out_path, "w") as f:
                f.write(output)

        passed = result.returncode == 0
        if passed:
            print(f"  ✅ PASSED ({tool} / {test_name})")
        else:
            print(f"  ❌ FAILED (exit code {result.returncode})")
            # Print last 30 lines of output for quick triage
            tail = "\n".join(output.strip().split("\n")[-30:])
            print(f"  ── Tail output ──\n{tail}\n  ────────────────")

        return passed, output

    except subprocess.TimeoutExpired:
        msg = f"[TIMEOUT] {test_name} exceeded {timeout}s"
        print(f"  ⏰ {msg}")
        with open(out_path, "w") as f:
            f.write(msg + "\n")
        return False, msg

    except FileNotFoundError as e:
        msg = f"[ERROR] Failed to run compute-sanitizer: {e}"
        print(f"  ❌ {msg}")
        return False, msg


def list_tests() -> None:
    """Print available GPU test modules."""
    print("Available GPU tests:\n")
    for name, module in GPU_TESTS:
        print(f"  {name:<20} {module}")
    print()
    print(f"Sanitizer tools: {', '.join(SANITIZER_TOOLS)}")


def run_all(tools: list[str] | None = None, test_filter: str | None = None) -> int:
    """Run the full matrix of tools × tests.

    Returns exit code (0 = all passed).
    """
    tools = tools or SANITIZER_TOOLS
    selected = GPU_TESTS
    if test_filter:
        selected = [(n, m) for n, m in GPU_TESTS if test_filter.lower() in n.lower()]
        if not selected:
            print(f"[ERROR] No tests match filter '{test_filter}'")
            return 1

    summary = []
    header = f" compute-sanitizer run — {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S UTC} "
    print(f"\n{'='*len(header)}\n{header}\n{'='*len(header)}")
    print(f"Host: {sys.executable}")
    print(f"Tests: {len(selected)} modules × {len(tools)} tools")
    print(f"Output: {REPORT_DIR}\n")

    overall_start = time.perf_counter()
    exit_code = 0

    for tool in tools:
        for test_name, test_module in selected:
            start = time.perf_counter()
            passed, _output = _run_sanitizer(tool, test_name, test_module)
            elapsed = time.perf_counter() - start
            summary.append((tool, test_name, passed, elapsed))
            if not passed:
                exit_code = 1
            # Brief pause between runs to let GPU cool
            time.sleep(1.0)

    # ── Summary ──────────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - overall_start
    print(f"\n{'='*60}")
    print(f"  SUMMARY  ({total_elapsed:.0f}s total)")
    print(f"{'='*60}")
    print(f"  {'Tool':<15} {'Test':<20} {'Result':<10} {'Time':<8}")
    print(f"  {'─'*53}")
    passed_count = 0
    for tool, test_name, passed, elapsed in summary:
        label = "✅ PASS" if passed else "❌ FAIL"
        if passed:
            passed_count += 1
        print(f"  {tool:<15} {test_name:<20} {label:<10} {elapsed:.0f}s")
    print(f"  {'─'*53}")
    print(f"  {passed_count}/{len(summary)} passed")
    print()

    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(
        description="NVIDIA Compute Sanitizer runner for HYDRA2DGPU GPU tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--tool", "-t",
        choices=SANITIZER_TOOLS + ["all"],
        default="all",
        help="Sanitizer tool to run (default: all)",
    )
    parser.add_argument(
        "--test", "-T",
        type=str,
        default=None,
        help="Run only tests whose name contains this substring",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available GPU test modules and exit",
    )
    args = parser.parse_args()

    if args.list:
        list_tests()
        return 0

    tools = SANITIZER_TOOLS if args.tool == "all" else [args.tool]
    return run_all(tools=tools, test_filter=args.test)


if __name__ == "__main__":
    sys.exit(main())
