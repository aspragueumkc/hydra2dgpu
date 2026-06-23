#!/usr/bin/env python3
"""
Valgrind memcheck runner for pure C++ native modules.

Runs minimal function invocations on each C++ module under Valgrind to
detect heap overruns, use-after-free, and uninitialized reads.

Usage:
    python tools/run_valgrind_cpp.py                       # all modules
    python tools/run_valgrind_cpp.py --module overlay      # single module
    python tools/run_valgrind_cpp.py --list                # list modules

Requires:
    valgrind installed (apt-get install valgrind)
    hydra_swe2d built in ./build/
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = REPO_ROOT / "build"

# Valgrind suppressions for known Qt/Conda false positives
SUPPRESSIONS = """
# Conda Python allocator interop
{
   conda_python_alloc
   Memcheck:Leak
   ...
   fun:PyMem_RawMalloc
}
{
   conda_python_free
   Memcheck:Free
   ...
   fun:PyMem_Free
}
"""


CPP_TESTS = [
    # (display_name, module_name, test_code)
    (
        "overlay_basic",
        "hydra_overlay",
        """
import numpy as np
import hydra_overlay as m

# rasterize_unstructured_accum
n = 100
np.random.seed(0)
x = np.random.uniform(0, 10, n).astype(np.float64)
y = np.random.uniform(0, 10, n).astype(np.float64)
scalar = np.random.uniform(0, 1, n).astype(np.float64)
result = m.rasterize_unstructured_accum(x, y, scalar, None, None, 32, 32, 0, 10, 0, 10)
assert isinstance(result, dict), "rasterize should return dict"

# rasterize_tri_mesh_accum (2 triangles)
nx = np.array([0., 10., 0., 10.], dtype=np.float64)
ny = np.array([0., 0., 10., 10.], dtype=np.float64)
tri = np.array([0, 1, 3, 0, 3, 2], dtype=np.int32)
cell_s = np.array([0.5, 0.3], dtype=np.float64)
tri_result = m.rasterize_tri_mesh_accum(nx, ny, tri, cell_s, None, None, 32, 32, 0, 10, 0, 10)
assert isinstance(tri_result, dict)

# finalize_scalar_field
sum_s = np.zeros((32, 32), dtype=np.float64)
cnt = np.zeros((32, 32), dtype=np.float64)
sum_s[8:24, 8:24] = 1.0
cnt[8:24, 8:24] = 1.0
field = m.finalize_scalar_field(sum_s, cnt, dilate_radius=1)
assert isinstance(field, dict)

# nearest_fill
vals = np.full((32, 32), np.nan, dtype=np.float64)
known = np.zeros((32, 32), dtype=bool)
vals[16, 16] = 1.0
known[16, 16] = True
filled = m.nearest_fill(vals, known)
assert filled.shape == (32, 32)

print("overlay_basic OK")
""",
    ),
    (
        "overlay_advect",
        "hydra_overlay",
        """
import numpy as np
import hydra_overlay as m

size = 32
u = np.zeros((size, size), dtype=np.float64)
v = np.zeros((size, size), dtype=np.float64)
speed = np.ones((size, size), dtype=np.float64)
mask = np.ones((size, size), dtype=bool)
seed = np.zeros((size, size), dtype=bool)
seed[16, 16] = True

u[16, :] = 0.5
v[:, 16] = 0.3

streams = m.advect_streamlines(u, v, speed, mask, seed,
    seed_count=10, max_steps=50, step_px=0.5, min_speed=0.01)
assert isinstance(streams, dict), f"expected dict, got {type(streams)}"
print("overlay_advect OK")
""",
    ),
    (
        "meshing_native_basic",
        "hydra_meshing_native",
        """
import numpy as np
import hydra_meshing_native as m

# polyline_overlap_fractions_open — two simple line segments
a = [(0.0, 0.0), (10.0, 0.0)]
b = [(5.0, -1.0), (5.0, 1.0)]
result = m.polyline_overlap_fractions_open(a, b, 0.5, 0.1, max_points=100)
assert isinstance(result, tuple), f"expected tuple, got {type(result)}"
print("meshing_native_basic OK")
""",
    ),
    (
        "meshing_native_ring",
        "hydra_meshing_native",
        """
import numpy as np
import hydra_meshing_native as m

# interface_overlap_metrics_closed — two concentric rings
ring_a = [(float(i), float(i)*0.1) for i in range(10)]
ring_b = [(float(i)+0.5, float(i)*0.1+0.05) for i in range(10)]
result = m.interface_overlap_metrics_closed(ring_a, ring_b, 0.5, 0.1, max_points=100)
assert isinstance(result, dict), f"expected dict, got {type(result)}"

# project_ring_to_chain
chain = [(float(i)*0.5, float(i)*0.05) for i in range(20)]
proj = m.project_ring_to_chain(ring_a, chain, 0.1)
assert isinstance(proj, dict), f"expected dict, got {type(proj)}"
print("meshing_native_ring OK")
""",
    ),
]


def list_tests() -> None:
    print("Available C++ Valgrind tests:\n")
    for name, mod, _ in CPP_TESTS:
        print(f"  {name:<25} ({mod})")
    print()


def _valgrind_cmd(python_code: str, supp_file: Path | None = None) -> list[str]:
    cmd = [
        "valgrind",
        "--tool=memcheck",
        "--leak-check=full",
        "--show-leak-kinds=definite",
        "--track-origins=yes",
        "--errors-for-leak-kinds=none",
        "--error-exitcode=1",
        "--quiet",
    ]
    if supp_file and supp_file.exists():
        cmd.append(f"--suppressions={supp_file}")
    cmd += [sys.executable, "-c", python_code]
    return cmd


def run_test(
    name: str, module_name: str, code: str, supp_file: Path | None = None
) -> tuple[bool, str, float]:
    import_line = f"import sys; sys.path.insert(0, r'{BUILD_DIR}'); "
    full_code = import_line + code

    start = time.perf_counter()
    try:
        result = subprocess.run(
            _valgrind_cmd(full_code, supp_file),
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(REPO_ROOT),
        )
        elapsed = time.perf_counter() - start
        output = result.stdout + "\n" + result.stderr
        passed = result.returncode == 0
        return passed, output, elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - start
        return False, "[TIMEOUT]", elapsed
    except FileNotFoundError:
        return False, "[ERROR] valgrind not found. Install: sudo apt-get install valgrind", 0.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Valgrind memcheck runner for C++ native modules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--module", "-m", type=str, default=None,
                        help="Run only tests whose name contains this substring")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List available tests and exit")
    args = parser.parse_args()

    if args.list:
        list_tests()
        return 0

    selected = CPP_TESTS
    if args.module:
        selected = [(n, m, c) for n, m, c in CPP_TESTS if args.module.lower() in n.lower() or args.module.lower() in m.lower()]
        if not selected:
            print(f"[ERROR] No tests match filter '{args.module}'")
            return 1

    # Write suppression file
    supp_file = REPO_ROOT / "report_output" / "valgrind" / "suppressions.txt"
    supp_file.parent.mkdir(parents=True, exist_ok=True)
    supp_file.write_text(SUPPRESSIONS)

    header = f" Valgrind memcheck — C++ native modules "
    print(f"\n{'='*len(header)}\n{header}\n{'='*len(header)}")
    print(f"Python: {sys.executable}")
    print(f"Build:  {BUILD_DIR}")
    print(f"Tests:  {len(selected)}\n")

    summary = []
    exit_code = 0

    for name, mod, code in selected:
        print(f"  [{name}] ", end="", flush=True)
        passed, output, elapsed = run_test(name, mod, code, supp_file)
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}  ({elapsed:.1f}s)")
        if not passed:
            print(f"  ── Output ──")
            for line in output.strip().split("\n")[-5:]:
                print(f"  {line}")
            print(f"  ────────────")
            exit_code = 1
        summary.append((name, passed, elapsed))

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    ok = sum(1 for _, p, _ in summary if p)
    print(f"  {ok}/{len(summary)} passed")
    print()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
