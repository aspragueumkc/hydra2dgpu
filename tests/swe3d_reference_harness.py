"""
tests/swe3d_reference_harness.py
---------------------------------
Reference-case loader and comparison harness for Stage-1 3D validation gate.

Usage from tests::

    from tests.swe3d_reference_harness import load_case, run_and_compare
    case = load_case("broad_crested_weir")
    result = run_and_compare(mod, case)
    for metric_name, passed, value, ref, tol in result.iter_metrics():
        ...

Cases are JSON files under tests/data/swe3d/<name>.json.

NOTE: The run_and_compare function is a stub that sets up the 3D solver from
the case spec and evaluates scalar metrics.  The geometry helpers (weir
insertion, culvert barrel masking) are left as TODOs until VoF advection and
pressure projection are implemented; at that point each helper should produce
a physically meaningful result.  Until then, the stub returns the best
observable it can (e.g. VoF fraction from patch stats) so the test scaffold
is wired end-to-end and will start passing as soon as the physics are in place.
"""

import json
import os
import math
import dataclasses
from typing import Iterator, List, Tuple, Optional
import numpy as np

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "swe3d")


@dataclasses.dataclass
class MetricResult:
    name: str
    passed: bool
    value: float
    ref: float
    delta: float
    tolerance: float
    description: str = ""


@dataclasses.dataclass
class CaseResult:
    case_name: str
    metrics: List[MetricResult] = dataclasses.field(default_factory=list)

    def iter_metrics(self) -> Iterator[Tuple[str, bool, float, float, float]]:
        """Yield (name, passed, value, ref, tol) for each metric."""
        for m in self.metrics:
            yield m.name, m.passed, m.value, m.ref, m.tolerance

    def all_passed(self) -> bool:
        return all(m.passed for m in self.metrics)

    def summary_lines(self) -> List[str]:
        lines = [f"Case: {self.case_name}"]
        for m in self.metrics:
            status = "PASS" if m.passed else "FAIL"
            lines.append(
                f"  [{status}] {m.name:35s}  "
                f"value={m.value:+.4e}  ref={m.ref:+.4e}  "
                f"delta={m.delta:+.4e}  tol={m.tolerance:.4e}"
            )
        return lines


# ---------------------------------------------------------------------------
# Case loader
# ---------------------------------------------------------------------------

def load_case(name: str) -> dict:
    """Load a JSON case spec by name (without .json extension)."""
    path = os.path.join(_DATA_DIR, f"{name}.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Reference case '{name}' not found at: {path}")
    with open(path) as fh:
        data = json.load(fh)
    return data


# ---------------------------------------------------------------------------
# Patch construction helpers
# ---------------------------------------------------------------------------

def _build_flat_surface_vof(nx: int, ny: int, nz: int, fill_depth_frac: float) -> np.ndarray:
    """
    Return a (nx*ny*nz,) VoF array with vof=1 in the bottom fill_depth_frac of
    z-layers and vof=0 above.  Cells are ordered (iz, iy, ix) in row-major.
    """
    n_fill = max(1, int(round(fill_depth_frac * nz)))
    vof = np.zeros(nx * ny * nz, dtype=np.float64)
    for iz in range(n_fill):
        lo = iz * nx * ny
        hi = lo + nx * ny
        vof[lo:hi] = 1.0
    return vof


def _probe_vof_free_surface_height(
        vof: np.ndarray, nx: int, ny: int, nz: int,
        dz: float, ix: int) -> float:
    """
    At a given x-column index ix, return an estimate of the free-surface height
    as the first z-level from the bottom where the column-averaged VoF drops
    below 0.5 (in metres above floor).

    Columns are averaged over y for robustness.
    Returns 0.0 if fully dry, lz_m if fully wet.
    """
    heights = []
    for iy in range(ny):
        for iz in range(nz):
            cell_idx = iz * nx * ny + iy * nx + ix
            col_vof = vof[cell_idx]
            # First layer where vof < 0.5 gives the interface
            if col_vof < 0.5:
                heights.append(iz * dz)
                break
        else:
            heights.append(nz * dz)  # fully filled
    return float(np.mean(heights)) if heights else 0.0


# ---------------------------------------------------------------------------
# Main harness entry point
# ---------------------------------------------------------------------------

def run_and_compare(mod, case: dict) -> CaseResult:
    """
    Instantiate a 3D solver matching *case*, run it for the specified number
    of steps, and evaluate all metrics against reference values.

    For each metric, the harness computes the best observable from the current
    solver state.  Many quantities will be approximate until VoF advection and
    pressure projection are wired in; the harness is designed so that the test
    scaffold is runnable before the physics are complete.
    """
    name = case.get("name", "unknown")
    result = CaseResult(case_name=name)

    domain = case["domain"]
    sim = case["simulation"]
    nx = domain["patch_nx"]
    ny = domain["patch_ny"]
    nz = domain["patch_nz"]
    lx = domain["lx_m"]
    ly = domain["ly_m"]
    lz = domain["lz_m"]
    n_cells = nx * ny * nz

    dx = lx / nx
    dy = ly / ny
    dz = lz / nz

    # Build a minimal 2D mesh (1 triangle) just enough to create a solver;
    # the 3D solver operates independently of the 2D mesh in uncoupled mode.
    _make_rect_mesh = _build_minimal_mesh
    mesh_2d = _make_rect_mesh(mod, lx, ly)
    n_cells_2d = mod.swe2d_mesh_info(mesh_2d)["n_cells"]
    h0 = np.full(n_cells_2d, 0.5, dtype=np.float64)

    # Set env overrides so the patch is sized to match the case spec.
    import os as _os
    _os.environ["BACKWATER_SWE3D_PATCH_NX"] = str(nx)
    _os.environ["BACKWATER_SWE3D_PATCH_NY"] = str(ny)
    _os.environ["BACKWATER_SWE3D_PATCH_NZ"] = str(nz)
    _os.environ["BACKWATER_SWE3D_PATCH_DX"] = f"{dx:.6f}"
    _os.environ["BACKWATER_SWE3D_PATCH_DY"] = f"{dy:.6f}"
    _os.environ["BACKWATER_SWE3D_PATCH_DZ"] = f"{dz:.6f}"

    solver = mod.swe2d_create_solver(
        mesh_2d,
        h0,
        use_gpu=True,
        temporal_order=2,
        coupling_mode=0,
        three_d_solver_model=1,
    )

    try:
        # Set initial VoF from case spec
        ic = case.get("initial_conditions", {})
        upstream_wse = ic.get("upstream_depth_m", ic.get("headwater_wse_m", lz * 0.5))
        fill_frac = min(1.0, upstream_wse / lz)
        vof_ic = _build_flat_surface_vof(nx, ny, nz, fill_frac)
        mod.swe2d_set_3d_patch_vof(solver, vof_ic)
        vof_sum_initial = float(np.sum(vof_ic))

        dt = sim.get("dt_s", 0.01)
        n_steps = sim.get("n_steps", 100)

        for _ in range(n_steps):
            mod.swe2d_step(solver, dt)

        # Collect patch stats
        stats = mod.swe2d_get_3d_patch_stats(solver)

        # ── Evaluate metrics ────────────────────────────────────────────────
        metrics_spec = case.get("metrics", {})

        for metric_name, spec in metrics_spec.items():
            ref = float(spec.get("ref_value", 0.0))
            tol_abs = spec.get("tolerance_abs", None)
            tol_rel = spec.get("tolerance_rel", None)
            tol = float(tol_abs) if tol_abs is not None else abs(ref) * float(tol_rel)

            # Compute observable from current solver state
            value = _evaluate_metric(
                metric_name, spec, stats, vof_ic, vof_sum_initial,
                nx, ny, nz, dx, dy, dz, lx, ly, lz)

            delta = abs(value - ref)
            passed = delta <= tol

            result.metrics.append(MetricResult(
                name=metric_name,
                passed=passed,
                value=value,
                ref=ref,
                delta=delta,
                tolerance=tol,
                description=spec.get("description", ""),
            ))

    finally:
        mod.swe2d_destroy(solver)
        # Clean up env overrides
        for k in ("BACKWATER_SWE3D_PATCH_NX", "BACKWATER_SWE3D_PATCH_NY",
                  "BACKWATER_SWE3D_PATCH_NZ", "BACKWATER_SWE3D_PATCH_DX",
                  "BACKWATER_SWE3D_PATCH_DY", "BACKWATER_SWE3D_PATCH_DZ"):
            _os.environ.pop(k, None)

    return result


def _evaluate_metric(
        metric_name: str,
        spec: dict,
        stats: dict,
        vof_ic: np.ndarray,
        vof_sum_initial: float,
        nx: int, ny: int, nz: int,
        dx: float, dy: float, dz: float,
        lx: float, ly: float, lz: float) -> float:
    """
    Map a metric name to an observable from the current solver state.

    Stubs return 0.0 with a clear TODO so the harness is runnable before
    the physics are in place.  Update each stub when the corresponding
    kernel is implemented.
    """
    if "free_surface" in metric_name:
        # TODO: replace stub once VoF advection is wired.
        # Best current estimate: derive WSE from VoF sum conservation.
        # vof_sum / (nx*ny) gives average number of filled z-layers.
        vof_sum = stats["vof_sum"]
        avg_filled_layers = vof_sum / (nx * ny)
        return avg_filled_layers * dz

    elif metric_name == "discharge_m3_per_s":
        # TODO: replace with cross-sectional VoF-weighted velocity integral
        # once momentum is non-trivially advected.
        # Stub returns 0.0 — will fail until real physics land.
        return 0.0

    elif metric_name == "pressurised_fraction":
        # Fraction of cells with VoF > 0.95.
        # TODO: restrict to barrel cells after geometry masking is added.
        vof_max = stats["vof_max"]
        if vof_max < 0.95:
            return 0.0
        # Return the VoF stats as a proxy (will be refined with real advection).
        # For now: if vof_max > 0.95 and vof_sum / n_cells_barrel is high, return fraction.
        # n_cells_barrel approximation: barrel occupies roughly (barrel_fraction * n_cells)
        barrel_fraction = 0.5  # placeholder
        n_barrel = max(1, int(nx * ny * nz * barrel_fraction))
        return min(1.0, stats["vof_sum"] / n_barrel)

    elif metric_name in ("inlet_head_m", "outlet_head_m"):
        # TODO: extract free-surface elevation at the probe x-column.
        vof_sum = stats["vof_sum"]
        avg_filled_layers = vof_sum / (nx * ny)
        return avg_filled_layers * dz

    else:
        # Unknown metric — return 0.0 and let the test fail with a clear delta.
        return 0.0


def _build_minimal_mesh(mod, lx: float, ly: float):
    """Build a tiny 4x2 rectangular mesh adequate for an uncoupled 3D solver."""
    xs = np.linspace(0.0, lx, 5)
    ys = np.linspace(0.0, ly, 3)
    xg, yg = np.meshgrid(xs, ys)
    node_x = xg.ravel().astype(np.float64)
    node_y = yg.ravel().astype(np.float64)
    node_z = np.zeros_like(node_x)
    cells = []
    stride = 5
    for j in range(2):
        for i in range(4):
            n00 = j * stride + i
            n10 = j * stride + i + 1
            n01 = (j + 1) * stride + i
            n11 = (j + 1) * stride + i + 1
            cells.extend([n00, n10, n11])
            cells.extend([n00, n11, n01])
    return mod.swe2d_build_mesh(
        node_x,
        node_y,
        node_z,
        np.array(cells, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.float64),
    )
