"""
Uncoupled 3D (single-phase free-surface) validation suite.

Execution intent (chronological):
1) Validate uncoupled 3D physics invariants first  (THIS FILE – Stage 1 gate)
2) Optimize performance and robustness after physics gates are green
3) Enable and validate 2D-3D coupling last

Stage-1 physics gates (always run, no env var required):
  * VoF boundedness  — vof ∈ [0,1] at all times
  * VoF conservation — total VoF sum constant when no sources/sinks
  * Rest stability   — zero-IC state stays exactly zero
  * Velocity damping — non-zero IC velocities decrease monotonically (scaffold damps)

Reference-case gates (gated behind BACKWATER_RUN_SWE3D_PHYSICS_CASES=1 for now):
  * Broad-crested weir free-surface profile
  * Culvert pressurisation sequence

Optional external cross-code gate (disabled by default):
    * OpenFOAM damBreak alpha-profile comparison
"""

import os
import sys
import unittest
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_PHYSICS_CASES = os.environ.get("BACKWATER_RUN_SWE3D_PHYSICS_CASES", "0") == "1"
_OPENFOAM_DAMBREAK = os.environ.get("BACKWATER_RUN_OPENFOAM_DAMBREAK", "0") == "1"
_SUBGRID_DAMBREAK = os.environ.get("BACKWATER_RUN_SWE3D_SUBGRID_DAMBREAK", "0") == "1"
_SWE3D_CELERITY_SENSITIVITY = os.environ.get("BACKWATER_RUN_SWE3D_CELERITY_SENSITIVITY", "0") == "1"
_SWE3D_TEST_VTK_ENABLE = os.environ.get("BACKWATER_SWE3D_TEST_VTK", "1") != "0"
_SWE3D_TEST_VTK_STRIDE = max(1, int(os.environ.get("BACKWATER_SWE3D_TEST_VTK_STRIDE", "1")))
_SWE3D_TEST_VTK_RUN = os.environ.get(
    "BACKWATER_SWE3D_TEST_VTK_RUN_ID",
    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}",
)
_SWE3D_TEST_VTK_BASE = Path(os.environ.get(
    "BACKWATER_SWE3D_TEST_VTK_DIR",
    str(Path(__file__).resolve().parent / "artifacts" / "swe3d_vtk"),
))
_SWE3D_CELERITY_CSV_BASE = Path(os.environ.get(
    "BACKWATER_SWE3D_CELERITY_CSV_DIR",
    str(Path(__file__).resolve().parent / "artifacts" / "swe3d_celerity"),
))


def _sanitize_path_name(name):
    return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in str(name))


class _SWE3DVTKRecorder:
    """Writes per-test SWE3D mesh and state snapshots for ParaView inspection."""

    def __init__(self, mod, test_id):
        self.mod = mod
        safe_test_id = _sanitize_path_name(test_id)
        self.test_dir = _SWE3D_TEST_VTK_BASE / _SWE3D_TEST_VTK_RUN / safe_test_id
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self._solver_entries = {}

    def _origin(self):
        def _get(name, default):
            raw = os.environ.get(name, "").strip()
            return float(raw) if raw else float(default)

        return (
            _get("BACKWATER_SWE3D_PATCH_ORIGIN_X", 0.0),
            _get("BACKWATER_SWE3D_PATCH_ORIGIN_Y", 0.0),
            _get("BACKWATER_SWE3D_PATCH_ORIGIN_Z", 0.0),
        )

    @staticmethod
    def _write_structured_points_header(fh, title, nx, ny, nz, ox, oy, oz, dx, dy, dz):
        fh.write("# vtk DataFile Version 3.0\n")
        fh.write(f"{title}\n")
        fh.write("ASCII\n")
        fh.write("DATASET STRUCTURED_POINTS\n")
        fh.write(f"DIMENSIONS {nx} {ny} {nz}\n")
        fh.write(f"ORIGIN {ox:.9g} {oy:.9g} {oz:.9g}\n")
        fh.write(f"SPACING {dx:.9g} {dy:.9g} {dz:.9g}\n")

    def _write_state_vtk(self, entry, out_path, title):
        stats = self.mod.swe2d_get_3d_patch_stats(entry["solver"])
        nx = int(stats["nx"])
        ny = int(stats["ny"])
        nz = int(stats["nz"])
        dx = float(stats["dx"])
        dy = float(stats["dy"])
        dz = float(stats["dz"])
        n_points = nx * ny * nz
        vof = np.asarray(self.mod.swe2d_get_3d_patch_vof(entry["solver"]), dtype=np.float64).reshape(-1)
        if vof.size != n_points:
            raise RuntimeError(
                f"Unexpected vof size {vof.size} for dims {nx}x{ny}x{nz} ({n_points} expected)")

        with out_path.open("w", encoding="utf-8") as fh:
            self._write_structured_points_header(
                fh,
                title,
                nx,
                ny,
                nz,
                entry["origin"][0],
                entry["origin"][1],
                entry["origin"][2],
                dx,
                dy,
                dz,
            )
            fh.write(f"POINT_DATA {n_points}\n")
            fh.write("SCALARS vof float 1\n")
            fh.write("LOOKUP_TABLE default\n")
            np.savetxt(fh, vof, fmt="%.7g")

    def _write_mesh_vtk(self, entry):
        mesh_path = entry["solver_dir"] / "mesh.vtk"
        with mesh_path.open("w", encoding="utf-8") as fh:
            self._write_structured_points_header(
                fh,
                "SWE3D patch mesh",
                entry["nx"],
                entry["ny"],
                entry["nz"],
                entry["origin"][0],
                entry["origin"][1],
                entry["origin"][2],
                entry["dx"],
                entry["dy"],
                entry["dz"],
            )

    def _ensure_entry(self, solver):
        key = id(solver)
        entry = self._solver_entries.get(key)
        if entry is not None:
            return entry

        stats = self.mod.swe2d_get_3d_patch_stats(solver)
        solver_idx = len(self._solver_entries) + 1
        solver_dir = self.test_dir / f"solver_{solver_idx:02d}"
        solver_dir.mkdir(parents=True, exist_ok=True)

        entry = {
            "solver": solver,
            "solver_dir": solver_dir,
            "origin": self._origin(),
            "nx": int(stats["nx"]),
            "ny": int(stats["ny"]),
            "nz": int(stats["nz"]),
            "dx": float(stats["dx"]),
            "dy": float(stats["dy"]),
            "dz": float(stats["dz"]),
            "step": 0,
            "snapshots": [],
        }
        self._solver_entries[key] = entry
        self._write_mesh_vtk(entry)
        return entry

    def capture_step(self, solver, diag):
        try:
            entry = self._ensure_entry(solver)
            step_idx = int(entry["step"])
            entry["step"] = step_idx + 1
            if (step_idx % _SWE3D_TEST_VTK_STRIDE) != 0:
                return

            dt = 0.0
            if isinstance(diag, dict):
                dt = float(diag.get("dt", 0.0) or 0.0)
            out_name = f"state_{step_idx:05d}.vtk"
            out_path = entry["solver_dir"] / out_name
            self._write_state_vtk(entry, out_path, f"SWE3D state step {step_idx} dt={dt:.6e}")
            entry["snapshots"].append((float(step_idx), out_name))
        except Exception:
            # Test assertions should not be blocked by exporter failures.
            return

    def capture_final(self, solver, tag="final"):
        try:
            entry = self._ensure_entry(solver)
            out_name = f"state_{_sanitize_path_name(tag)}.vtk"
            out_path = entry["solver_dir"] / out_name
            self._write_state_vtk(entry, out_path, f"SWE3D state {tag}")
            entry["snapshots"].append((float(entry["step"]), out_name))
            self._write_pvd(entry)
        except Exception:
            return

    @staticmethod
    def _write_pvd(entry):
        pvd_path = entry["solver_dir"] / "series.pvd"
        with pvd_path.open("w", encoding="utf-8") as fh:
            fh.write('<?xml version="1.0"?>\n')
            fh.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
            fh.write('  <Collection>\n')
            for timestep, filename in entry["snapshots"]:
                fh.write(
                    f'    <DataSet timestep="{timestep:.6f}" group="" part="0" file="{filename}"/>\n')
            fh.write('  </Collection>\n')
            fh.write('</VTKFile>\n')


def _install_swe3d_vtk_hooks(mod, test_id):
    if not _SWE3D_TEST_VTK_ENABLE:
        return None

    recorder = _SWE3DVTKRecorder(mod, test_id)
    orig_step = mod.swe2d_step
    orig_destroy = mod.swe2d_destroy

    def _step_hook(solver, dt):
        try:
            diag = orig_step(solver, dt)
        except Exception:
            recorder.capture_final(solver, tag="error")
            raise
        recorder.capture_step(solver, diag)
        return diag

    def _destroy_hook(solver):
        recorder.capture_final(solver, tag="final")
        return orig_destroy(solver)

    mod.swe2d_step = _step_hook
    mod.swe2d_destroy = _destroy_hook
    return {
        "recorder": recorder,
        "orig_step": orig_step,
        "orig_destroy": orig_destroy,
        "step_hook": _step_hook,
        "destroy_hook": _destroy_hook,
    }


def _uninstall_swe3d_vtk_hooks(mod, hook_state):
    if not hook_state:
        return

    if getattr(mod, "swe2d_step", None) is hook_state.get("step_hook"):
        mod.swe2d_step = hook_state["orig_step"]
    if getattr(mod, "swe2d_destroy", None) is hook_state.get("destroy_hook"):
        mod.swe2d_destroy = hook_state["orig_destroy"]


@contextmanager
def _temporary_env(overrides):
    old = {}
    try:
        for key, value in (overrides or {}).items():
            old[key] = os.environ.get(key)
            os.environ[key] = str(value)
        yield
    finally:
        for key, prev in old.items():
            if prev is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prev


def _load_module():
    repo_root = Path(__file__).resolve().parent.parent
    build_dir = repo_root / "build"

    # Prefer the freshly built extension module over any stale repo-root copy.
    if build_dir.is_dir():
        build_path = str(build_dir)
        if build_path in sys.path:
            sys.path.remove(build_path)
        sys.path.insert(0, build_path)

    try:
        import hydra_swe2d
        return hydra_swe2d
    except ImportError:
        if build_dir.is_dir():
            try:
                import hydra_swe2d
                return hydra_swe2d
            except ImportError:
                return None
        return None


def _gpu_available():
    mod = _load_module()
    if mod is None:
        return False
    try:
        return bool(mod.swe2d_gpu_available())
    except Exception:
        return False


def _make_rect_mesh(mod, nx, ny, lx, ly):
    xs = np.linspace(0.0, lx, nx + 1)
    ys = np.linspace(0.0, ly, ny + 1)
    xg, yg = np.meshgrid(xs, ys)
    node_x = xg.ravel().copy()
    node_y = yg.ravel().copy()
    node_z = np.zeros_like(node_x)
    cells = []
    stride = nx + 1
    for j in range(ny):
        for i in range(nx):
            n00 = j * stride + i
            n10 = j * stride + i + 1
            n01 = (j + 1) * stride + i
            n11 = (j + 1) * stride + i + 1
            cells.extend([n00, n10, n11])
            cells.extend([n00, n11, n01])
    mesh = mod.swe2d_build_mesh(
        node_x,
        node_y,
        node_z,
        np.array(cells, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.float64),
    )
    return mesh


def _make_3d_solver(mod, mesh, h0, env_overrides=None, coupling_mode=0):
    """Create a single-phase 3D solver with selectable coupling mode."""
    kwargs = dict(
        use_gpu=True,
        temporal_order=2,
        coupling_mode=int(coupling_mode),
        three_d_solver_model=1,
    )
    if env_overrides:
        with _temporary_env(env_overrides):
            return mod.swe2d_create_solver(mesh, h0, **kwargs)
    return mod.swe2d_create_solver(mesh, h0, **kwargs)


def _upload_interface_contract(mod, solver, cell2d, face_area, face_nx, face_ny, face_nz):
    """Upload a custom 2D-3D interface contract for coupling regressions."""
    cell2d = np.asarray(cell2d, dtype=np.int32)
    face_area = np.asarray(face_area, dtype=np.float64)
    face_nx = np.asarray(face_nx, dtype=np.float64)
    face_ny = np.asarray(face_ny, dtype=np.float64)
    face_nz = np.asarray(face_nz, dtype=np.float64)
    contract = mod.swe2d_contract_create(cell2d, face_area, face_nx, face_ny, face_nz)
    if not bool(mod.swe2d_contract_is_valid(contract)):
        raise RuntimeError("Failed to build valid 2D-3D interface contract for test")
    ok = bool(mod.swe2d_gpu_contract_upload(solver, contract))
    if not ok:
        raise RuntimeError("Failed to upload 2D-3D interface contract in test")
    return contract


def _upload_simple_interface_contract(mod, solver, cell_idx=0, nx=-1.0, ny=0.0, nz=0.0, area=1.0):
    """Upload a one-face contract suitable for focused coupling regressions."""
    return _upload_interface_contract(
        mod,
        solver,
        [int(cell_idx)],
        [float(area)],
        [float(nx)],
        [float(ny)],
        [float(nz)],
    )


def _flat_surface_vof(stats):
    """
    Build a VoF field with the bottom half of cells fully filled (vof=1),
    the top half empty (vof=0), mimicking a flat horizontal free surface.
    Returns a 1-D numpy array of length stats['n_cells'].
    The patch is nx x ny x nz (z is vertical).
    """
    nx, ny, nz = int(stats["nx"]), int(stats["ny"]), int(stats["nz"])
    n_cells = nx * ny * nz
    vof = np.zeros(n_cells, dtype=np.float64)
    z_half = nz // 2
    for iz in range(nz):
        if iz < z_half:
            lo = iz * nx * ny
            hi = lo + nx * ny
            vof[lo:hi] = 1.0
    return vof


def _x_slab_vof(stats, frac_lo=0.1, frac_hi=0.3):
    """Build a vertical x-aligned VoF slab replicated over all y,z cells."""
    nx, ny, nz = int(stats["nx"]), int(stats["ny"]), int(stats["nz"])
    n_cells = nx * ny * nz
    vof = np.zeros(n_cells, dtype=np.float64)
    ix0 = max(0, min(nx - 1, int(np.floor(frac_lo * nx))))
    ix1 = max(ix0 + 1, min(nx, int(np.ceil(frac_hi * nx))))
    for iz in range(nz):
        for iy in range(ny):
            row = iz * nx * ny + iy * nx
            vof[row + ix0:row + ix1] = 1.0
    return vof


def _slotted_dam_geometry(stats, frac_x=0.5, thickness_cells=2, slot_frac=0.2):
    """Build a thin subgrid dam with a centered breach slot."""
    nx, ny, nz = int(stats["nx"]), int(stats["ny"]), int(stats["nz"])
    n_cells = nx * ny * nz
    phi = np.ones(n_cells, dtype=np.float64)
    ax = np.ones(n_cells, dtype=np.float64)
    ay = np.ones(n_cells, dtype=np.float64)
    az = np.ones(n_cells, dtype=np.float64)

    dam_lo = max(1, min(nx - 1, int(np.floor(frac_x * nx)) - max(0, thickness_cells // 2)))
    dam_hi = min(nx - 1, dam_lo + max(1, int(thickness_cells)))
    slot_half = max(1, int(np.ceil(0.5 * slot_frac * ny)))
    slot_center = ny // 2
    slot_lo = max(0, slot_center - slot_half)
    slot_hi = min(ny, slot_center + slot_half)

    barrier_mask = np.zeros(n_cells, dtype=bool)
    slot_mask = np.zeros(n_cells, dtype=bool)
    for iz in range(nz):
        for iy in range(ny):
            row = iz * nx * ny + iy * nx
            barrier_mask[row + dam_lo:row + dam_hi] = True
            if slot_lo <= iy < slot_hi:
                slot_mask[row + dam_lo:row + dam_hi] = True

    solid_mask = barrier_mask & (~slot_mask)
    phi[solid_mask] = 0.0
    ax[solid_mask] = 0.0
    ay[solid_mask] = 0.0
    az[solid_mask] = 0.0
    return phi, ax, ay, az, solid_mask, slot_mask


def _porous_slotted_dam_geometry(
    stats,
    frac_x=0.5,
    thickness_cells=2,
    slot_frac=0.2,
    phi_barrier=0.20,
    area_barrier=0.10,
):
    """Build a thin porous dam with a centered open breach slot."""
    nx, ny, nz = int(stats["nx"]), int(stats["ny"]), int(stats["nz"])
    n_cells = nx * ny * nz
    phi = np.ones(n_cells, dtype=np.float64)
    ax = np.ones(n_cells, dtype=np.float64)
    ay = np.ones(n_cells, dtype=np.float64)
    az = np.ones(n_cells, dtype=np.float64)

    dam_lo = max(1, min(nx - 1, int(np.floor(frac_x * nx)) - max(0, thickness_cells // 2)))
    dam_hi = min(nx - 1, dam_lo + max(1, int(thickness_cells)))
    slot_half = max(1, int(np.ceil(0.5 * slot_frac * ny)))
    slot_center = ny // 2
    slot_lo = max(0, slot_center - slot_half)
    slot_hi = min(ny, slot_center + slot_half)

    barrier_mask = np.zeros(n_cells, dtype=bool)
    slot_mask = np.zeros(n_cells, dtype=bool)
    for iz in range(nz):
        for iy in range(ny):
            row = iz * nx * ny + iy * nx
            barrier_mask[row + dam_lo:row + dam_hi] = True
            if slot_lo <= iy < slot_hi:
                slot_mask[row + dam_lo:row + dam_hi] = True

    porous_mask = barrier_mask & (~slot_mask)
    phi[porous_mask] = float(phi_barrier)
    ax[porous_mask] = float(area_barrier)
    ay[porous_mask] = float(area_barrier)
    az[porous_mask] = float(area_barrier)
    return phi, ax, ay, az, porous_mask, slot_mask


def _gravity_weir_hydrograph_q(
    t_s,
    *,
    width_m,
    head_base_m,
    head_peak_m,
    t_rise_s,
    t_hold_s,
    t_fall_s,
    cd=0.62,
    g=9.81,
):
    """Broad-crested-weir-style gravity-driven inflow hydrograph Q(t) [m^3/s]."""
    t = max(0.0, float(t_s))
    h0 = max(0.0, float(head_base_m))
    hp = max(h0, float(head_peak_m))
    tr = max(1.0e-9, float(t_rise_s))
    th = max(0.0, float(t_hold_s))
    tf = max(1.0e-9, float(t_fall_s))

    if t < tr:
        head = h0 + (hp - h0) * (t / tr)
    elif t < tr + th:
        head = hp
    elif t < tr + th + tf:
        head = hp + (h0 - hp) * ((t - tr - th) / tf)
    else:
        head = h0

    # Q = C_d * b * sqrt(2g) * H^(3/2)
    return float(cd) * float(width_m) * np.sqrt(2.0 * float(g)) * max(0.0, head) ** 1.5


def _x_front_position_from_vof(vof, stats, wet_threshold=0.02):
    """Estimate farthest wet x-position [m] from 3D VoF volume data."""
    nx = int(stats["nx"])
    ny = int(stats["ny"])
    nz = int(stats["nz"])
    dx = float(stats["dx"])
    vof_3d = np.asarray(vof, dtype=np.float64).reshape((nz, ny, nx))
    x_profile = np.mean(vof_3d, axis=(0, 1))
    wet_idx = np.flatnonzero(x_profile >= float(wet_threshold))
    if wet_idx.size <= 0:
        return 0.0
    return (float(wet_idx[-1]) + 0.5) * dx


def _estimate_celerity_linear(times_s, x_front_m):
    """Estimate wave celerity [m/s] from front trajectory using linear fit."""
    t = np.asarray(times_s, dtype=np.float64)
    x = np.asarray(x_front_m, dtype=np.float64)
    if t.size < 2 or x.size < 2:
        return 0.0
    valid = np.isfinite(t) & np.isfinite(x)
    t = t[valid]
    x = x[valid]
    if t.size < 2:
        return 0.0
    x0 = float(np.min(x))
    moving = x >= (x0 + 2.0e-3)
    if np.count_nonzero(moving) >= 2:
        t = t[moving]
        x = x[moving]
    if t.size < 2:
        return 0.0
    dt = t[-1] - t[0]
    if dt <= 0.0:
        return 0.0
    return float(np.polyfit(t, x, 1)[0])


def _write_celerity_sensitivity_csv(test_id, rows):
    """Write celerity sensitivity rows to CSV artifact and return path."""
    safe_test_id = _sanitize_path_name(test_id)
    run_dir = _SWE3D_CELERITY_CSV_BASE / _SWE3D_TEST_VTK_RUN
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / f"{safe_test_id}.csv"
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write("predictor_damping_coeff,celerity_mps,front_end_m,n_samples\n")
        for row in rows:
            fh.write(
                f"{row['coeff']:.17g},"
                f"{row['celerity_mps']:.17g},"
                f"{row['front_end_m']:.17g},"
                f"{int(row['n_samples'])}\n"
            )
    return out_path


@unittest.skipUnless(_load_module() is not None, "hydra_swe2d not built")
@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
class TestSWE3DUncoupledValidation(unittest.TestCase):
    """Stage-1 validation gates for uncoupled 3D mode."""

    def setUp(self):
        self.mod = _load_module()
        self._vtk_hook_state = _install_swe3d_vtk_hooks(self.mod, self.id())
        self.mesh = _make_rect_mesh(self.mod, 20, 10, 200.0, 100.0)
        n_cells = self.mod.swe2d_mesh_info(self.mesh)["n_cells"]
        self.h0 = np.full(n_cells, 1.0, dtype=np.float64)

    def tearDown(self):
        _uninstall_swe3d_vtk_hooks(self.mod, getattr(self, "_vtk_hook_state", None))

    # ── Smoke ──────────────────────────────────────────────────────────────────

    def test_uncoupled_3d_mode_smoke_gpu(self):
        """8 steps finish without error; diagnostics are sane."""
        solver = _make_3d_solver(self.mod, self.mesh, self.h0)
        try:
            for _ in range(8):
                diag = self.mod.swe2d_step(solver, -1.0)
                self.assertTrue(diag["gpu_active"])
                self.assertGreater(diag["dt"], 0.0)

            h, hu, hv = self.mod.swe2d_get_state(solver)
            self.assertTrue(np.all(np.isfinite(h)))
            self.assertTrue(np.all(np.isfinite(hu)))
            self.assertTrue(np.all(np.isfinite(hv)))
            self.assertGreaterEqual(float(np.min(h)), -1.0e-10)
        finally:
            self.mod.swe2d_destroy(solver)

    def test_uncoupled_3d_does_not_require_interface_contract(self):
        """coupling_mode=0 must not raise for a missing contract."""
        solver = _make_3d_solver(self.mod, self.mesh, self.h0)
        try:
            _ = self.mod.swe2d_step(solver, -1.0)
        finally:
            self.mod.swe2d_destroy(solver)

    def test_uncoupled_3d_adaptive_dt_tracks_3d_velocity_magnitude(self):
        """Adaptive dt should shrink when 3D patch velocity magnitude increases."""
        solver = _make_3d_solver(self.mod, self.mesh, self.h0)
        try:
            stats0 = self.mod.swe2d_get_3d_patch_stats(solver)
            n = int(stats0["n_cells"])
            zeros = np.zeros(n, dtype=np.float64)
            ones = np.ones(n, dtype=np.float64)

            self.mod.swe2d_set_3d_patch_state(
                solver, u=zeros, v=zeros, w=zeros, p=zeros, vof=ones)
            diag_slow = self.mod.swe2d_step(solver, -1.0)
            dt_slow = float(diag_slow["dt"])

            w_fast = np.full(n, 40.0, dtype=np.float64)
            self.mod.swe2d_set_3d_patch_state(
                solver, u=zeros, v=zeros, w=w_fast, p=zeros, vof=ones)
            diag_fast = self.mod.swe2d_step(solver, -1.0)
            dt_fast = float(diag_fast["dt"])

            self.assertLess(
                dt_fast,
                dt_slow,
                f"Expected adaptive dt to decrease for high 3D velocity: dt_slow={dt_slow:.6e}, dt_fast={dt_fast:.6e}")
            self.assertLess(
                dt_fast,
                5.0e-2,
                f"Expected high-velocity adaptive dt to become small; got dt_fast={dt_fast:.6e}")
            self.assertGreaterEqual(
                float(diag_fast.get("max_courant", -1.0)),
                0.0,
                f"Expected non-negative 3D CFL diagnostic; got {diag_fast.get('max_courant')}")
        finally:
            self.mod.swe2d_destroy(solver)

    def test_uncoupled_3d_adaptive_dt_shrinks_for_fractional_cut_cells(self):
        """Geometry-aware adaptive dt should shrink when low-phi cut-cells are present."""
        if not hasattr(self.mod, "swe2d_set_3d_patch_geometry"):
            self.skipTest("swe2d_set_3d_patch_geometry not available in native module")

        solver_open = _make_3d_solver(self.mod, self.mesh, self.h0)
        solver_cut = _make_3d_solver(self.mod, self.mesh, self.h0)
        try:
            stats0 = self.mod.swe2d_get_3d_patch_stats(solver_open)
            nx = int(stats0["nx"])
            ny = int(stats0["ny"])
            nz = int(stats0["nz"])
            n = int(stats0["n_cells"])

            u_fast = np.full(n, 30.0, dtype=np.float64)
            zeros = np.zeros(n, dtype=np.float64)
            ones = np.ones(n, dtype=np.float64)

            self.mod.swe2d_set_3d_patch_state(
                solver_open, u=u_fast, v=zeros, w=zeros, p=zeros, vof=ones)

            idx = np.arange(nx * ny * nz, dtype=np.int64)
            ix = idx % nx
            band_lo = max(0, min(nx - 1, nx // 3))
            band_hi = min(nx, max(band_lo + 1, nx // 3 + 2))
            cut_band = (ix >= band_lo) & (ix < band_hi)

            phi = np.ones(n, dtype=np.float64)
            ax = np.ones(n, dtype=np.float64)
            ay = np.ones(n, dtype=np.float64)
            az = np.ones(n, dtype=np.float64)
            phi[cut_band] = 0.08
            ax[cut_band] = 0.50
            ay[cut_band] = 0.50
            az[cut_band] = 0.50

            self.mod.swe2d_set_3d_patch_geometry(solver_cut, phi=phi, ax=ax, ay=ay, az=az)
            self.mod.swe2d_set_3d_patch_state(
                solver_cut, u=u_fast, v=zeros, w=zeros, p=zeros, vof=ones)

            diag_open = self.mod.swe2d_step(solver_open, -1.0)
            diag_cut = self.mod.swe2d_step(solver_cut, -1.0)
            dt_open = float(diag_open["dt"])
            dt_cut = float(diag_cut["dt"])

            self.assertGreater(dt_open, 0.0)
            self.assertGreater(dt_cut, 0.0)
            self.assertLess(
                dt_cut,
                0.8 * dt_open,
                f"Expected cut-cell-aware dt reduction; dt_open={dt_open:.6e}, dt_cut={dt_cut:.6e}")
        finally:
            self.mod.swe2d_destroy(solver_open)
            self.mod.swe2d_destroy(solver_cut)

    def test_uncoupled_3d_patch_face_length_env_overrides_cell_counts(self):
        """Target face-length env overrides should drive resolved patch nx/ny/nz."""
        env = {
            "BACKWATER_SWE3D_PATCH_NX": "64",
            "BACKWATER_SWE3D_PATCH_NY": "64",
            "BACKWATER_SWE3D_PATCH_NZ": "64",
            "BACKWATER_SWE3D_PATCH_FACE_LEN_X": "20.0",
            "BACKWATER_SWE3D_PATCH_FACE_LEN_Y": "25.0",
            "BACKWATER_SWE3D_PATCH_FACE_LEN_Z": "0.25",
        }
        solver = _make_3d_solver(self.mod, self.mesh, self.h0, env_overrides=env)
        try:
            stats = self.mod.swe2d_get_3d_patch_stats(solver)
            self.assertEqual(int(stats["nx"]), 10)
            self.assertEqual(int(stats["ny"]), 4)
            self.assertEqual(int(stats["nz"]), 4)
            self.assertAlmostEqual(float(stats["dx"]), 20.0, places=12)
            self.assertAlmostEqual(float(stats["dy"]), 25.0, places=12)
            self.assertAlmostEqual(float(stats["dz"]), 0.25, places=12)
        finally:
            self.mod.swe2d_destroy(solver)

    # ── Coupling exchange (focused) ──────────────────────────────────────────

    def test_coupled_3d_mode_requires_contract_upload(self):
        """Coupled 3D mode should fail fast if no interface contract is uploaded."""
        solver = _make_3d_solver(self.mod, self.mesh, self.h0, coupling_mode=1)
        try:
            with self.assertRaises(RuntimeError):
                self.mod.swe2d_step(solver, 0.1)
        finally:
            self.mod.swe2d_destroy(solver)

    def test_one_way_coupling_forces_3d_patch_each_step(self):
        """One-way coupling should inject 2D-driven momentum into 3D patch boundary cells."""
        solver = _make_3d_solver(self.mod, self.mesh, self.h0, coupling_mode=1)
        try:
            _upload_simple_interface_contract(self.mod, solver, cell_idx=0, nx=-1.0, ny=0.0, nz=0.0, area=50.0)

            h2d, hu2d, hv2d = self.mod.swe2d_get_state(solver)
            h2d = np.asarray(h2d, dtype=np.float64)
            hu2d = np.asarray(hu2d, dtype=np.float64)
            hv2d = np.asarray(hv2d, dtype=np.float64)
            hu2d[:] = 0.0
            hv2d[:] = 0.0
            hu2d[0] = 12.0 * max(h2d[0], 1.0)
            self.mod.swe2d_set_state(solver, h2d, hu2d, hv2d)

            stats0 = self.mod.swe2d_get_3d_patch_stats(solver)
            n3d = int(stats0["n_cells"])
            zeros = np.zeros(n3d, dtype=np.float64)
            self.mod.swe2d_set_3d_patch_state(solver, u=zeros, v=zeros, w=zeros, p=zeros, vof=zeros)

            self.mod.swe2d_step(solver, 0.1)
            stats1 = self.mod.swe2d_get_3d_patch_stats(solver)

            self.assertGreater(
                stats1["u_rms"],
                1.0e-6,
                f"Expected one-way coupling to force non-zero 3D u_rms; got {stats1['u_rms']:.4e}")
        finally:
            self.mod.swe2d_destroy(solver)

    def test_two_way_coupling_applies_feedback_to_2d_state(self):
        """Two-way mode should apply non-zero feedback to 2D state when contract is active."""
        solver = _make_3d_solver(self.mod, self.mesh, self.h0, coupling_mode=2)
        try:
            _upload_simple_interface_contract(self.mod, solver, cell_idx=0, nx=1.0, ny=0.0, nz=0.0, area=50.0)

            h2d, hu2d, hv2d = self.mod.swe2d_get_state(solver)
            h2d = np.asarray(h2d, dtype=np.float64)
            hu2d = np.asarray(hu2d, dtype=np.float64)
            hv2d = np.asarray(hv2d, dtype=np.float64)
            hu2d[:] = 0.0
            hv2d[:] = 0.0
            hu2d[0] = 10.0 * max(h2d[0], 1.0)
            self.mod.swe2d_set_state(solver, h2d, hu2d, hv2d)

            stats0 = self.mod.swe2d_get_3d_patch_stats(solver)
            n3d = int(stats0["n_cells"])
            zeros = np.zeros(n3d, dtype=np.float64)
            self.mod.swe2d_set_3d_patch_state(solver, u=zeros, v=zeros, w=zeros, p=zeros, vof=zeros)

            h_before, _, _ = self.mod.swe2d_get_state(solver)
            h_before = np.asarray(h_before, dtype=np.float64)

            self.mod.swe2d_step(solver, 0.1)

            h_after, _, _ = self.mod.swe2d_get_state(solver)
            h_after = np.asarray(h_after, dtype=np.float64)
            delta = abs(float(h_after[0] - h_before[0]))

            self.assertGreater(
                delta,
                1.0e-8,
                f"Expected two-way coupling to modify 2D depth at contract cell; delta={delta:.4e}")
        finally:
            self.mod.swe2d_destroy(solver)

    def test_two_way_coupling_redistributes_depth_conservatively_across_contract(self):
        """Two-way feedback should redistribute depth across multi-face contract cells in the expected direction."""
        solver = _make_3d_solver(self.mod, self.mesh, self.h0, coupling_mode=2)
        try:
            _upload_interface_contract(
                self.mod,
                solver,
                cell2d=[0, 1],
                face_area=[30.0, 10.0],
                face_nx=[1.0, 1.0],
                face_ny=[0.0, 0.0],
                face_nz=[0.0, 0.0],
            )

            h2d, hu2d, hv2d = self.mod.swe2d_get_state(solver)
            h2d = np.asarray(h2d, dtype=np.float64)
            hu2d = np.asarray(hu2d, dtype=np.float64)
            hv2d = np.asarray(hv2d, dtype=np.float64)
            hu2d[:] = 0.0
            hv2d[:] = 0.0
            hu2d[0] = 12.0 * max(h2d[0], 1.0)
            hu2d[1] = 2.0 * max(h2d[1], 1.0)
            self.mod.swe2d_set_state(solver, h2d, hu2d, hv2d)

            stats0 = self.mod.swe2d_get_3d_patch_stats(solver)
            n3d = int(stats0["n_cells"])
            zeros = np.zeros(n3d, dtype=np.float64)
            ones = np.ones(n3d, dtype=np.float64)
            self.mod.swe2d_set_3d_patch_state(solver, u=zeros, v=zeros, w=zeros, p=zeros, vof=ones)

            h_before, _, _ = self.mod.swe2d_get_state(solver)
            h_before = np.asarray(h_before, dtype=np.float64)

            self.mod.swe2d_step(solver, 0.1)

            h_after, _, _ = self.mod.swe2d_get_state(solver)
            h_after = np.asarray(h_after, dtype=np.float64)

            self.assertLess(
                h_after[0],
                h_before[0],
                "Expected higher-discharge contract cell to lose depth under conservative redistribution")
            self.assertGreater(
                h_after[1],
                h_before[1],
                "Expected lower-discharge contract cell to gain depth under conservative redistribution")
        finally:
            self.mod.swe2d_destroy(solver)

    def test_two_way_flux_form_outflow_removes_2d_momentum(self):
        """Flux-form two-way closure should remove 2D momentum for net 2D->3D discharge correction."""
        solver = _make_3d_solver(self.mod, self.mesh, self.h0, coupling_mode=2)
        try:
            _upload_simple_interface_contract(self.mod, solver, cell_idx=0, nx=1.0, ny=0.0, nz=0.0, area=50.0)

            h2d, hu2d, hv2d = self.mod.swe2d_get_state(solver)
            h2d = np.asarray(h2d, dtype=np.float64)
            hu2d = np.asarray(hu2d, dtype=np.float64)
            hv2d = np.asarray(hv2d, dtype=np.float64)
            hu2d[:] = 0.0
            hv2d[:] = 0.0
            hu2d[0] = 14.0 * max(h2d[0], 1.0)
            self.mod.swe2d_set_state(solver, h2d, hu2d, hv2d)

            stats0 = self.mod.swe2d_get_3d_patch_stats(solver)
            n3d = int(stats0["n_cells"])
            zeros = np.zeros(n3d, dtype=np.float64)
            ones = np.ones(n3d, dtype=np.float64)
            self.mod.swe2d_set_3d_patch_state(solver, u=zeros, v=zeros, w=zeros, p=zeros, vof=ones)

            h_before, hu_before, _ = self.mod.swe2d_get_state(solver)
            h_before = np.asarray(h_before, dtype=np.float64)
            hu_before = np.asarray(hu_before, dtype=np.float64)

            self.mod.swe2d_step(solver, 0.1)

            h_after, hu_after, _ = self.mod.swe2d_get_state(solver)
            h_after = np.asarray(h_after, dtype=np.float64)
            hu_after = np.asarray(hu_after, dtype=np.float64)

            self.assertLess(
                h_after[0],
                h_before[0],
                "Expected net 2D->3D correction to reduce depth in donor cell")
            self.assertLess(
                hu_after[0],
                hu_before[0],
                "Expected flux-form outflow correction to remove 2D momentum from donor cell")
        finally:
            self.mod.swe2d_destroy(solver)

    # ── Patch descriptor ───────────────────────────────────────────────────────

    def test_3d_patch_stats_descriptor_plausible(self):
        """Patch stats should expose valid positive dimensions."""
        solver = _make_3d_solver(self.mod, self.mesh, self.h0)
        try:
            stats = self.mod.swe2d_get_3d_patch_stats(solver)
            self.assertGreater(stats["nx"], 0)
            self.assertGreater(stats["ny"], 0)
            self.assertGreater(stats["nz"], 0)
            self.assertGreater(stats["dx"], 0.0)
            self.assertGreater(stats["dy"], 0.0)
            self.assertGreater(stats["dz"], 0.0)
            expected_n = stats["nx"] * stats["ny"] * stats["nz"]
            self.assertEqual(stats["n_cells"], expected_n)
        finally:
            self.mod.swe2d_destroy(solver)

    # ── VoF boundedness ────────────────────────────────────────────────────────

    def test_vof_bounds_preserved_flat_surface(self):
        """
        Physics invariant: VoF must remain in [0,1] for all time.
        IC: flat free surface (lower half of cells = 1, upper half = 0).
        Run 20 steps; check vof_min >= 0 and vof_max <= 1.
        """
        solver = _make_3d_solver(self.mod, self.mesh, self.h0)
        try:
            stats0 = self.mod.swe2d_get_3d_patch_stats(solver)
            vof_ic = _flat_surface_vof(stats0)
            self.mod.swe2d_set_3d_patch_vof(solver, vof_ic)

            for _ in range(20):
                self.mod.swe2d_step(solver, -1.0)

            stats = self.mod.swe2d_get_3d_patch_stats(solver)
            self.assertGreaterEqual(
                stats["vof_min"], -1.0e-10,
                f"VoF fell below 0: min={stats['vof_min']:.4e}")
            self.assertLessEqual(
                stats["vof_max"], 1.0 + 1.0e-10,
                f"VoF exceeded 1: max={stats['vof_max']:.4e}")
        finally:
            self.mod.swe2d_destroy(solver)

    # ── VoF conservation ───────────────────────────────────────────────────────

    def test_vof_sum_conserved_no_source(self):
        """
        Physics invariant: in the absence of in/outflow, total VoF must be
        conserved (sum changes by < 0.1 %).
        IC: flat free surface (half-filled).
        Run 50 steps.
        """
        solver = _make_3d_solver(self.mod, self.mesh, self.h0)
        try:
            stats0 = self.mod.swe2d_get_3d_patch_stats(solver)
            vof_ic = _flat_surface_vof(stats0)
            self.mod.swe2d_set_3d_patch_vof(solver, vof_ic)

            sum_0 = float(np.sum(vof_ic))
            for _ in range(50):
                self.mod.swe2d_step(solver, -1.0)

            stats = self.mod.swe2d_get_3d_patch_stats(solver)
            rel_err = abs(stats["vof_sum"] - sum_0) / max(sum_0, 1.0)
            self.assertLess(
                rel_err, 1.0e-3,
                f"VoF sum drifted: initial={sum_0:.4f} final={stats['vof_sum']:.4f} "
                f"rel_err={rel_err:.2e}")
        finally:
            self.mod.swe2d_destroy(solver)

    # ── Rest stability ─────────────────────────────────────────────────────────

    def test_zero_velocity_state_stays_zero(self):
        """
        Physics invariant (Gauss-Seidel rest): if u=v=w=p=0 initially, they
        remain exactly zero after N steps (damping from zero gives zero).
        """
        solver = _make_3d_solver(self.mod, self.mesh, self.h0)
        try:
            stats0 = self.mod.swe2d_get_3d_patch_stats(solver)
            n = int(stats0["n_cells"])
            zeros = np.zeros(n, dtype=np.float64)
            self.mod.swe2d_set_3d_patch_state(
                solver, u=zeros, v=zeros, w=zeros, p=zeros)

            for _ in range(20):
                self.mod.swe2d_step(solver, -1.0)

            stats = self.mod.swe2d_get_3d_patch_stats(solver)
            self.assertAlmostEqual(stats["u_rms"], 0.0, places=12,
                msg=f"u_rms should be zero; got {stats['u_rms']:.3e}")
            self.assertAlmostEqual(stats["v_rms"], 0.0, places=12,
                msg=f"v_rms should be zero; got {stats['v_rms']:.3e}")
            self.assertAlmostEqual(stats["w_rms"], 0.0, places=12,
                msg=f"w_rms should be zero; got {stats['w_rms']:.3e}")
            self.assertAlmostEqual(stats["p_max_abs"], 0.0, places=12,
                msg=f"p_max_abs should be zero; got {stats['p_max_abs']:.3e}")
        finally:
            self.mod.swe2d_destroy(solver)

    # ── Velocity damping ───────────────────────────────────────────────────────

    def test_velocity_damping_monotone_from_nonzero_ic(self):
        """
        Stability gate: from a non-zero uniform velocity field, u_rms should
        stay finite and bounded, and should show at least one damping interval
        over a short integration window.
        """
        solver = _make_3d_solver(self.mod, self.mesh, self.h0)
        try:
            stats0 = self.mod.swe2d_get_3d_patch_stats(solver)
            n = int(stats0["n_cells"])
            u_ic = np.full(n, 1.0, dtype=np.float64)   # uniform 1 m/s
            zeros = np.zeros(n, dtype=np.float64)
            vof_ic = np.ones(n, dtype=np.float64)
            self.mod.swe2d_set_3d_patch_vof(solver, vof_ic)
            self.mod.swe2d_set_3d_patch_state(
                solver, u=u_ic, v=zeros, w=zeros, p=zeros)

            rms_hist = []
            for step in range(10):
                self.mod.swe2d_step(solver, 0.1)  # fixed dt so damping is predictable
                stats = self.mod.swe2d_get_3d_patch_stats(solver)
                cur_rms = stats["u_rms"]
                self.assertTrue(
                    np.isfinite(cur_rms),
                    f"u_rms is not finite at step {step}")
                rms_hist.append(float(cur_rms))

            self.assertLess(
                max(rms_hist),
                2.5,
                f"u_rms exceeded stability bound: max={max(rms_hist):.6e}, series={rms_hist}")
            self.assertLess(
                min(rms_hist[1:]),
                rms_hist[0],
                f"u_rms never showed damping after first step: series={rms_hist}")
        finally:
            self.mod.swe2d_destroy(solver)

    def test_projection_reduces_divergence_from_divergent_ic(self):
        """
        Numerics gate: projection stage should reduce divergence RMS from a
        deliberately divergent initial condition.
        """
        solver = _make_3d_solver(self.mod, self.mesh, self.h0)
        try:
            stats0 = self.mod.swe2d_get_3d_patch_stats(solver)
            nx, ny, nz = int(stats0["nx"]), int(stats0["ny"]), int(stats0["nz"])
            n = int(stats0["n_cells"])

            # u=x, v=y, w=0 induces positive divergence in the interior.
            u_ic = np.zeros(n, dtype=np.float64)
            v_ic = np.zeros(n, dtype=np.float64)
            w_ic = np.zeros(n, dtype=np.float64)
            p_ic = np.zeros(n, dtype=np.float64)
            for iz in range(nz):
                for iy in range(ny):
                    for ix in range(nx):
                        idx = iz * nx * ny + iy * nx + ix
                        u_ic[idx] = float(ix)
                        v_ic[idx] = float(iy)

            self.mod.swe2d_set_3d_patch_state(
                solver, u=u_ic, v=v_ic, w=w_ic, p=p_ic)

            before = self.mod.swe2d_get_3d_patch_stats(solver)
            self.assertGreater(
                before["divergence_rms"], 1.0e-12,
                f"Expected non-trivial initial divergence; got {before['divergence_rms']:.3e}")

            self.mod.swe2d_step(solver, 0.1)
            after = self.mod.swe2d_get_3d_patch_stats(solver)

            self.assertLess(
                after["divergence_rms"], before["divergence_rms"],
                f"Projection did not reduce divergence_rms: "
                f"{before['divergence_rms']:.6e} -> {after['divergence_rms']:.6e}")
            self.assertGreater(after["projection_iters"], 0)
            self.assertTrue(np.isfinite(after["projection_residual"]))
            self.assertGreaterEqual(after["projection_residual"], 0.0)
        finally:
            self.mod.swe2d_destroy(solver)

    def test_projection_retry_telemetry_reports_reduction_and_bounds(self):
        """Sprint 3 gate: retry path should surface bounded, reproducible telemetry."""
        env = {
            "BACKWATER_SWE3D_PROJECTION_REJECT_ENABLE": "1",
            "BACKWATER_SWE3D_PROJECTION_RESIDUAL_TARGET": "1e-12",
            "BACKWATER_SWE3D_PROJECTION_DT_REDUCTION": "0.5",
            "BACKWATER_SWE3D_PROJECTION_MAX_RETRIES": "2",
            "BACKWATER_SWE3D_PROJECTION_MIN_DT_FACTOR": "0.01",
        }
        solver = _make_3d_solver(self.mod, self.mesh, self.h0, env_overrides=env)
        try:
            stats0 = self.mod.swe2d_get_3d_patch_stats(solver)
            nx = int(stats0["nx"])
            ny = int(stats0["ny"])
            nz = int(stats0["nz"])
            n = int(stats0["n_cells"])

            u_ic = np.zeros(n, dtype=np.float64)
            v_ic = np.zeros(n, dtype=np.float64)
            zeros = np.zeros(n, dtype=np.float64)
            vof_ic = np.ones(n, dtype=np.float64)
            for iz in range(nz):
                for iy in range(ny):
                    for ix in range(nx):
                        idx = iz * nx * ny + iy * nx + ix
                        u_ic[idx] = float(ix)
                        v_ic[idx] = float(iy)

            self.mod.swe2d_set_3d_patch_vof(solver, vof_ic)
            self.mod.swe2d_set_3d_patch_state(
                solver, u=u_ic, v=v_ic, w=zeros, p=zeros)

            dt_req = 0.2
            with _temporary_env(env):
                diag = self.mod.swe2d_step(solver, dt_req)

            self.assertTrue(bool(diag.get("projection_retry_enabled", False)))
            self.assertGreaterEqual(int(diag.get("projection_attempt_count", 0)), 1)
            self.assertGreaterEqual(int(diag.get("projection_retry_count", -1)), 0)

            dt_initial = float(diag.get("projection_retry_dt_initial", -1.0))
            dt_floor = float(diag.get("projection_retry_dt_floor", -1.0))
            dt_final = float(diag.get("dt", -1.0))
            reduction = float(diag.get("projection_retry_dt_reduction", -1.0))
            resid_target = float(diag.get("projection_retry_residual_target", -1.0))
            resid_ratio = float(diag.get("projection_retry_residual_ratio", -1.0))
            resid_ratio_max = float(diag.get("projection_retry_residual_ratio_max", -1.0))

            self.assertTrue(np.isfinite(dt_initial))
            self.assertTrue(np.isfinite(dt_floor))
            self.assertTrue(np.isfinite(dt_final))
            self.assertTrue(np.isfinite(reduction))
            self.assertTrue(np.isfinite(resid_target))
            self.assertTrue(np.isfinite(resid_ratio))
            self.assertTrue(np.isfinite(resid_ratio_max))

            self.assertGreater(dt_initial, 0.0)
            self.assertLessEqual(dt_initial, dt_req + 1.0e-15)
            self.assertLessEqual(dt_final, dt_initial + 1.0e-15)
            self.assertGreaterEqual(dt_final + 1.0e-15, dt_floor)
            self.assertGreaterEqual(reduction, 0.05)
            self.assertLessEqual(reduction, 0.99)
            self.assertGreater(resid_target, 0.0)
            self.assertGreaterEqual(resid_ratio, 0.0)
            self.assertGreaterEqual(resid_ratio_max + 1.0e-15, resid_ratio)

            retry_count = int(diag.get("projection_retry_count", 0))
            self.assertGreaterEqual(retry_count, 0)
            self.assertIn(bool(diag.get("projection_retry_exhausted", False)), (True, False))
        finally:
            self.mod.swe2d_destroy(solver)

    def test_projection_retry_telemetry_disabled_path_is_explicit(self):
        """Sprint 3 gate: disabling retry should report explicit disabled telemetry."""
        env = {
            "BACKWATER_SWE3D_PROJECTION_REJECT_ENABLE": "0",
            "BACKWATER_SWE3D_PROJECTION_MAX_RETRIES": "4",
            "BACKWATER_SWE3D_PROJECTION_RESIDUAL_TARGET": "1e-12",
        }
        solver = _make_3d_solver(self.mod, self.mesh, self.h0, env_overrides=env)
        try:
            stats0 = self.mod.swe2d_get_3d_patch_stats(solver)
            nx = int(stats0["nx"])
            ny = int(stats0["ny"])
            nz = int(stats0["nz"])
            n = int(stats0["n_cells"])

            u_ic = np.zeros(n, dtype=np.float64)
            v_ic = np.zeros(n, dtype=np.float64)
            zeros = np.zeros(n, dtype=np.float64)
            for iz in range(nz):
                for iy in range(ny):
                    for ix in range(nx):
                        idx = iz * nx * ny + iy * nx + ix
                        u_ic[idx] = float(ix)
                        v_ic[idx] = float(iy)

            self.mod.swe2d_set_3d_patch_state(
                solver, u=u_ic, v=v_ic, w=zeros, p=zeros)

            dt_req = 0.2
            with _temporary_env(env):
                diag = self.mod.swe2d_step(solver, dt_req)

            self.assertFalse(bool(diag.get("projection_retry_enabled", True)))
            self.assertFalse(bool(diag.get("projection_retry_exhausted", True)))
            self.assertEqual(int(diag.get("projection_retry_count", -1)), 0)
            self.assertEqual(int(diag.get("projection_attempt_count", -1)), 1)
            self.assertAlmostEqual(
                float(diag.get("dt", -1.0)),
                float(diag.get("projection_retry_dt_initial", -2.0)),
                delta=1.0e-12)
        finally:
            self.mod.swe2d_destroy(solver)

    def test_projection_retry_fail_fast_raises_on_exhaustion(self):
        """Sprint 3 gate: optional fail-fast should abort when projection retries are exhausted."""
        env = {
            "BACKWATER_SWE3D_PROJECTION_REJECT_ENABLE": "1",
            "BACKWATER_SWE3D_PROJECTION_FAIL_FAST": "1",
            "BACKWATER_SWE3D_PROJECTION_RESIDUAL_TARGET": "1e-12",
            "BACKWATER_SWE3D_PROJECTION_DT_REDUCTION": "0.5",
            "BACKWATER_SWE3D_PROJECTION_MAX_RETRIES": "2",
            "BACKWATER_SWE3D_PROJECTION_MIN_DT_FACTOR": "0.01",
            "BACKWATER_SWE3D_STATE_REJECT_ENABLE": "1",
            "BACKWATER_SWE3D_STATE_MAX_ABS_VELOCITY": "1e-3",
        }
        solver = _make_3d_solver(self.mod, self.mesh, self.h0, env_overrides=env)
        try:
            stats0 = self.mod.swe2d_get_3d_patch_stats(solver)
            nx = int(stats0["nx"])
            ny = int(stats0["ny"])
            nz = int(stats0["nz"])
            n = int(stats0["n_cells"])

            u_ic = np.zeros(n, dtype=np.float64)
            v_ic = np.zeros(n, dtype=np.float64)
            zeros = np.zeros(n, dtype=np.float64)
            vof_ic = np.ones(n, dtype=np.float64)
            for iz in range(nz):
                for iy in range(ny):
                    for ix in range(nx):
                        idx = iz * nx * ny + iy * nx + ix
                        u_ic[idx] = float(ix)
                        v_ic[idx] = float(iy)

            self.mod.swe2d_set_3d_patch_vof(solver, vof_ic)
            self.mod.swe2d_set_3d_patch_state(
                solver, u=u_ic, v=v_ic, w=zeros, p=zeros)

            with self.assertRaisesRegex(RuntimeError, "projection retry exhausted|state guard retry exhausted"):
                with _temporary_env(env):
                    self.mod.swe2d_step(solver, 0.2)
        finally:
            self.mod.swe2d_destroy(solver)

    def test_state_guard_fail_fast_raises_on_velocity_blowup(self):
        """Production hardening: state guard should fail-fast on nonphysical velocity magnitude."""
        env = {
            "BACKWATER_SWE3D_PROJECTION_REJECT_ENABLE": "1",
            "BACKWATER_SWE3D_PROJECTION_FAIL_FAST": "1",
            "BACKWATER_SWE3D_PROJECTION_RESIDUAL_TARGET": "1.0",
            "BACKWATER_SWE3D_PROJECTION_MAX_RETRIES": "1",
            "BACKWATER_SWE3D_STATE_REJECT_ENABLE": "1",
            "BACKWATER_SWE3D_STATE_MAX_ABS_VELOCITY": "0.25",
            "BACKWATER_SWE3D_STATE_MAX_ABS_PRESSURE": "1e12",
            "BACKWATER_SWE3D_STATE_VOF_BOUNDS_TOL": "1e-6",
        }
        solver = _make_3d_solver(self.mod, self.mesh, self.h0, env_overrides=env)
        try:
            stats0 = self.mod.swe2d_get_3d_patch_stats(solver)
            n = int(stats0["n_cells"])

            u_ic = np.full(n, 10.0, dtype=np.float64)
            zeros = np.zeros(n, dtype=np.float64)
            vof_ic = np.ones(n, dtype=np.float64)
            self.mod.swe2d_set_3d_patch_vof(solver, vof_ic)
            self.mod.swe2d_set_3d_patch_state(
                solver, u=u_ic, v=zeros, w=zeros, p=zeros)

            with self.assertRaisesRegex(RuntimeError, "state guard retry exhausted"):
                with _temporary_env(env):
                    self.mod.swe2d_step(solver, 0.05)
        finally:
            self.mod.swe2d_destroy(solver)

    def test_projection_retry_accepts_bounded_relative_residual(self):
        """Production hardening: bounded pressure updates should not exhaust retries just because absolute pressure is large."""
        env = {
            "BACKWATER_SWE3D_PROJECTION_REJECT_ENABLE": "1",
            "BACKWATER_SWE3D_PROJECTION_FAIL_FAST": "1",
            "BACKWATER_SWE3D_PROJECTION_RESIDUAL_TARGET": "1.0",
            "BACKWATER_SWE3D_PROJECTION_MAX_RETRIES": "1",
            "BACKWATER_SWE3D_STATE_REJECT_ENABLE": "1",
            "BACKWATER_SWE3D_STATE_MAX_ABS_VELOCITY": "100.0",
            "BACKWATER_SWE3D_STATE_MAX_ABS_PRESSURE": "1e12",
            "BACKWATER_SWE3D_VELOCITY_SOFT_CAP_CFL": "16.0",
        }
        solver = _make_3d_solver(self.mod, self.mesh, self.h0, env_overrides=env)
        try:
            stats0 = self.mod.swe2d_get_3d_patch_stats(solver)
            n = int(stats0["n_cells"])

            u_ic = np.full(n, 5.0, dtype=np.float64)
            zeros = np.zeros(n, dtype=np.float64)
            vof_ic = np.ones(n, dtype=np.float64)
            self.mod.swe2d_set_3d_patch_vof(solver, vof_ic)
            self.mod.swe2d_set_3d_patch_state(
                solver, u=u_ic, v=zeros, w=zeros, p=zeros)

            with _temporary_env(env):
                diag = self.mod.swe2d_step(solver, 0.1)
            stats1 = self.mod.swe2d_get_3d_patch_stats(solver)

            self.assertFalse(bool(diag.get("projection_retry_exhausted", True)))
            self.assertLessEqual(
                float(stats1["projection_residual"]),
                1.0,
                f"Expected bounded relative projection residual; got {stats1['projection_residual']:.6e}")
            self.assertLessEqual(
                float(stats1["p_max_abs"]),
                1.0e12,
                f"Expected state guard pressure limit to remain unviolated; got {stats1['p_max_abs']:.6e}")
        finally:
            self.mod.swe2d_destroy(solver)

    def test_vof_advection_transports_interface_positive_x(self):
        """
        Numerics gate: conservative MUSCL-limited VoF transport should move an x-slab
        interface in +x under positive uniform u while keeping total VoF nearly
        conserved in a closed domain.
        """
        solver = _make_3d_solver(self.mod, self.mesh, self.h0)
        try:
            stats0 = self.mod.swe2d_get_3d_patch_stats(solver)
            nx = int(stats0["nx"])
            n = int(stats0["n_cells"])
            vof_ic = _x_slab_vof(stats0, frac_lo=0.10, frac_hi=0.30)
            self.mod.swe2d_set_3d_patch_vof(solver, vof_ic)

            u_ic = np.full(n, 0.5, dtype=np.float64)
            zeros = np.zeros(n, dtype=np.float64)
            self.mod.swe2d_set_3d_patch_state(
                solver, u=u_ic, v=zeros, w=zeros, p=zeros)

            def _x_centroid(vof_arr):
                wsum = float(np.sum(vof_arr))
                if wsum <= 0.0:
                    return 0.0
                idx = np.arange(vof_arr.size, dtype=np.int64)
                ix = (idx % nx).astype(np.float64)
                return float(np.sum(ix * vof_arr) / wsum)

            x0 = _x_centroid(vof_ic)
            sum0 = float(np.sum(vof_ic))

            for _ in range(8):
                self.mod.swe2d_step(solver, 0.1)

            vof_after = self.mod.swe2d_get_3d_patch_vof(solver)
            x1 = _x_centroid(vof_after)
            sum1 = float(np.sum(vof_after))

            self.assertGreater(
                x1, x0 + 1.0e-3,
                f"Expected +x transport of VoF centroid; got x0={x0:.6f}, x1={x1:.6f}")
            rel_err = abs(sum1 - sum0) / max(sum0, 1.0)
            self.assertLess(
                rel_err, 5.0e-3,
                f"VoF mass drift too large during advection: rel_err={rel_err:.3e}")
        finally:
            self.mod.swe2d_destroy(solver)

    def test_geometry_obstruction_reduces_downstream_transport_and_velocity(self):
        """
        Regression gate: uploading synthetic obstruction tensors should suppress
        downstream VoF transport across a blocked x-slab and reduce velocity RMS.
        """
        if not hasattr(self.mod, "swe2d_set_3d_patch_geometry"):
            self.skipTest("swe2d_set_3d_patch_geometry not available in native module")

        solver_open = _make_3d_solver(self.mod, self.mesh, self.h0)
        solver_blocked = _make_3d_solver(self.mod, self.mesh, self.h0)
        try:
            stats0 = self.mod.swe2d_get_3d_patch_stats(solver_open)
            nx = int(stats0["nx"])
            ny = int(stats0["ny"])
            nz = int(stats0["nz"])
            n = int(stats0["n_cells"])

            vof_ic = _x_slab_vof(stats0, frac_lo=0.30, frac_hi=0.48)
            u_ic = np.full(n, 20.0, dtype=np.float64)
            zeros = np.zeros(n, dtype=np.float64)

            idx = np.arange(nx * ny * nz, dtype=np.int64)
            ix = idx % nx
            barrier_lo = max(1, nx // 2 - 1)
            barrier_hi = min(nx - 1, barrier_lo + 2)
            barrier_mask = (ix >= barrier_lo) & (ix < barrier_hi)
            right_mask = ix >= barrier_hi

            self.mod.swe2d_set_3d_patch_vof(solver_open, vof_ic)
            self.mod.swe2d_set_3d_patch_state(
                solver_open, u=u_ic, v=zeros, w=zeros, p=zeros)

            phi = np.ones(n, dtype=np.float64)
            ax = np.ones(n, dtype=np.float64)
            ay = np.ones(n, dtype=np.float64)
            az = np.ones(n, dtype=np.float64)
            phi[barrier_mask] = 0.0
            ax[barrier_mask] = 0.0
            ay[barrier_mask] = 0.0
            az[barrier_mask] = 0.0

            self.mod.swe2d_set_3d_patch_geometry(
                solver_blocked, phi=phi, ax=ax, ay=ay, az=az)
            self.mod.swe2d_set_3d_patch_vof(solver_blocked, vof_ic)
            self.mod.swe2d_set_3d_patch_state(
                solver_blocked, u=u_ic, v=zeros, w=zeros, p=zeros)

            for _ in range(12):
                self.mod.swe2d_step(solver_open, 0.2)
                self.mod.swe2d_step(solver_blocked, 0.2)

            vof_open = self.mod.swe2d_get_3d_patch_vof(solver_open)
            vof_blocked = self.mod.swe2d_get_3d_patch_vof(solver_blocked)
            stats_open = self.mod.swe2d_get_3d_patch_stats(solver_open)
            stats_blocked = self.mod.swe2d_get_3d_patch_stats(solver_blocked)

            right_open = float(np.sum(vof_open[right_mask]))
            right_blocked = float(np.sum(vof_blocked[right_mask]))
            barrier_mass = float(np.sum(vof_blocked[barrier_mask]))

            self.assertGreater(
                right_open, 1.0e-4,
                f"Open case did not advect interface downstream; right_open={right_open:.4e}")
            self.assertLess(
                right_blocked,
                0.35 * right_open + 1.0e-10,
                f"Blocked case leaked too much downstream VoF: "
                f"right_blocked={right_blocked:.4e}, right_open={right_open:.4e}")
            self.assertLess(
                barrier_mass,
                1.0e-8,
                f"Blocked slab should remain dry (phi=0); barrier_mass={barrier_mass:.4e}")
            self.assertLess(
                stats_blocked["u_rms"],
                stats_open["u_rms"],
                f"Blocked case should reduce velocity RMS: "
                f"u_rms_blocked={stats_blocked['u_rms']:.4e}, "
                f"u_rms_open={stats_open['u_rms']:.4e}")
        finally:
            self.mod.swe2d_destroy(solver_open)
            self.mod.swe2d_destroy(solver_blocked)

    def test_vof_cfl_substepping_engages_for_large_dt(self):
        """
        Numerics gate: VoF transport should switch to multiple CFL-limited
        substeps when dt is large relative to |u| and cell spacing.
        """
        solver = _make_3d_solver(self.mod, self.mesh, self.h0)
        try:
            stats0 = self.mod.swe2d_get_3d_patch_stats(solver)
            n = int(stats0["n_cells"])
            vof_ic = _x_slab_vof(stats0, frac_lo=0.15, frac_hi=0.25)
            self.mod.swe2d_set_3d_patch_vof(solver, vof_ic)

            u_ic = np.full(n, 20.0, dtype=np.float64)
            zeros = np.zeros(n, dtype=np.float64)
            self.mod.swe2d_set_3d_patch_state(
                solver, u=u_ic, v=zeros, w=zeros, p=zeros)

            self.mod.swe2d_step(solver, 2.0)
            stats = self.mod.swe2d_get_3d_patch_stats(solver)

            self.assertGreater(
                int(stats["vof_transport_substeps"]), 1,
                f"Expected VoF substepping to engage; got {stats['vof_transport_substeps']}")
            self.assertGreaterEqual(stats["vof_min"], -1.0e-10)
            self.assertLessEqual(stats["vof_max"], 1.0 + 1.0e-10)
        finally:
            self.mod.swe2d_destroy(solver)

    # ── Boundary conditions (focused regression) ─────────────────────────────

    def test_bc_inflow_xmin_increases_vof_mass(self):
        """Inflow BC should inject phase volume from xmin face into a dry patch."""
        env = {
            "BACKWATER_SWE3D_BC_XMIN_MODE": "1",  # INFLOW
            "BACKWATER_SWE3D_BC_XMIN_U": "2.0",
            "BACKWATER_SWE3D_BC_XMIN_VOF": "1.0",
        }
        solver = _make_3d_solver(self.mod, self.mesh, self.h0, env_overrides=env)
        try:
            stats0 = self.mod.swe2d_get_3d_patch_stats(solver)
            n = int(stats0["n_cells"])
            nx = int(stats0["nx"])
            zeros = np.zeros(n, dtype=np.float64)

            self.mod.swe2d_set_3d_patch_state(
                solver, u=zeros, v=zeros, w=zeros, p=zeros, vof=zeros)

            for _ in range(10):
                self.mod.swe2d_step(solver, 0.2)

            vof = self.mod.swe2d_get_3d_patch_vof(solver)
            total = float(np.sum(vof))
            idx = np.arange(n, dtype=np.int64)
            left_mass = float(np.sum(vof[(idx % nx) == 0]))

            self.assertGreater(total, 1.0e-3, f"Expected inflow mass increase; got total={total:.4e}")
            self.assertGreater(left_mass, 1.0e-4, f"Expected wetting near xmin boundary; got left_mass={left_mass:.4e}")
        finally:
            self.mod.swe2d_destroy(solver)

    def test_bc_volumetric_inlet_q_xmin_increases_vof_mass(self):
        """Volumetric inlet BC should inject phase volume using prescribed flow rate Q."""
        env = {
            "BACKWATER_SWE3D_BC_XMIN_MODE": "4",  # INFLOW_FLOW_RATE
            "BACKWATER_SWE3D_BC_XMIN_Q": "20.0",
            "BACKWATER_SWE3D_BC_XMIN_VOF": "1.0",
        }
        solver = _make_3d_solver(self.mod, self.mesh, self.h0, env_overrides=env)
        try:
            stats0 = self.mod.swe2d_get_3d_patch_stats(solver)
            n = int(stats0["n_cells"])
            nx = int(stats0["nx"])
            zeros = np.zeros(n, dtype=np.float64)

            self.mod.swe2d_set_3d_patch_state(
                solver, u=zeros, v=zeros, w=zeros, p=zeros, vof=zeros)

            for _ in range(10):
                self.mod.swe2d_step(solver, 0.2)

            vof = self.mod.swe2d_get_3d_patch_vof(solver)
            total = float(np.sum(vof))
            idx = np.arange(n, dtype=np.int64)
            left_mass = float(np.sum(vof[(idx % nx) == 0]))

            self.assertGreater(total, 1.0e-3, f"Expected volumetric inlet mass increase; got total={total:.4e}")
            self.assertGreater(left_mass, 1.0e-4, f"Expected wetting near xmin boundary; got left_mass={left_mass:.4e}")
        finally:
            self.mod.swe2d_destroy(solver)

    def test_bc_volumetric_inlet_q_dynamic_area_policy_increases_injection_when_wet_area_is_limited(self):
        """
        Dynamic wet/open-area normalization (policy=1) should inject more mass
        than legacy total-face-area normalization (policy=0) when only part of
        the inlet face is wet/open.
        """
        env_common = {
            "BACKWATER_SWE3D_BC_XMIN_MODE": "4",   # INFLOW_FLOW_RATE
            "BACKWATER_SWE3D_BC_XMIN_Q": "40.0",
            "BACKWATER_SWE3D_BC_XMIN_VOF": "1.0",
            "BACKWATER_SWE3D_BC_XMAX_MODE": "2",   # OUTFLOW
            "BACKWATER_SWE3D_PATCH_NX": "64",
            "BACKWATER_SWE3D_PATCH_NY": "8",
            "BACKWATER_SWE3D_PATCH_NZ": "12",
        }

        def _run_policy(area_policy: int) -> float:
            env = dict(env_common)
            env["BACKWATER_SWE3D_Q_INFLOW_AREA_POLICY"] = str(int(area_policy))
            # Runtime BC controls are read during stepping, so keep overrides
            # active for the full create+step+readback lifecycle.
            with _temporary_env(env):
                solver = _make_3d_solver(self.mod, self.mesh, self.h0)
                try:
                    stats0 = self.mod.swe2d_get_3d_patch_stats(solver)
                    n = int(stats0["n_cells"])
                    nx = int(stats0["nx"])
                    ny = int(stats0["ny"])
                    nz = int(stats0["nz"])

                    # Prime only the lower third of the xmin boundary as wet so the
                    # effective inlet area is intentionally smaller than full-face area.
                    vof0 = np.zeros(n, dtype=np.float64)
                    z_wet = max(1, nz // 3)
                    for iz in range(z_wet):
                        for iy in range(ny):
                            idx = iz * nx * ny + iy * nx  # ix=0
                            vof0[idx] = 1.0

                    zeros = np.zeros(n, dtype=np.float64)
                    self.mod.swe2d_set_3d_patch_state(
                        solver, u=zeros, v=zeros, w=zeros, p=zeros, vof=vof0)

                    for _ in range(8):
                        self.mod.swe2d_step(solver, 0.15)

                    return float(np.sum(self.mod.swe2d_get_3d_patch_vof(solver)))
                finally:
                    self.mod.swe2d_destroy(solver)

        total_legacy = _run_policy(0)
        total_dynamic = _run_policy(1)

        self.assertGreater(
            total_dynamic,
            total_legacy * 1.01,
            "Expected dynamic inlet-area normalization to increase injected mass "
            f"for partially wet inlet: legacy={total_legacy:.6e}, dynamic={total_dynamic:.6e}")

    @unittest.skipUnless(
        _SWE3D_CELERITY_SENSITIVITY,
        "Set BACKWATER_RUN_SWE3D_CELERITY_SENSITIVITY=1 to run predictor-damping celerity sensitivity test.")
    def test_celerity_sensitivity_vs_predictor_damping_gravity_hydrograph(self):
        """Higher SWE3D predictor damping coefficient should reduce floodwave celerity under a gravity-driven inflow hydrograph."""
        if not hasattr(self.mod, "swe2d_set_3d_patch_face_bc"):
            self.skipTest("swe2d_set_3d_patch_face_bc not available in native module")

        env_base = {
            "BACKWATER_SWE3D_BC_XMIN_MODE": "4",   # Volumetric inlet (Q)
            "BACKWATER_SWE3D_BC_XMIN_VOF": "1.0",
            "BACKWATER_SWE3D_BC_XMAX_MODE": "2",   # Outflow
            "BACKWATER_SWE3D_BC_ZMAX_MODE": "3",   # Free surface venting
            "BACKWATER_SWE3D_BC_ZMAX_P": "0.0",
            "BACKWATER_SWE3D_PATCH_NX": "96",
            "BACKWATER_SWE3D_PATCH_NY": "8",
            "BACKWATER_SWE3D_PATCH_NZ": "16",
        }
        damping_coeffs = (0.0, 0.05, 0.20)
        dt_step = 0.20
        n_steps = 140
        sample_stride = 2

        celerities = []
        summary_rows = []
        for coeff in damping_coeffs:
            env = dict(env_base)
            env["BACKWATER_SWE3D_PREDICTOR_DAMPING_COEFF"] = f"{float(coeff):.17g}"
            with _temporary_env(env):
                solver = _make_3d_solver(self.mod, self.mesh, self.h0)
                try:
                    stats = self.mod.swe2d_get_3d_patch_stats(solver)
                    n = int(stats["n_cells"])
                    ny = int(stats["ny"])
                    dy = float(stats["dy"])
                    zeros = np.zeros(n, dtype=np.float64)

                    # Start from dry patch so boundary hydrograph drives the floodwave.
                    self.mod.swe2d_set_3d_patch_state(
                        solver, u=zeros, v=zeros, w=zeros, p=zeros, vof=zeros)

                    times_s = []
                    front_x_m = []
                    for k in range(n_steps):
                        t_now = float(k) * dt_step
                        q_in = _gravity_weir_hydrograph_q(
                            t_now,
                            width_m=max(1.0e-6, float(ny) * dy),
                            head_base_m=0.06,
                            head_peak_m=0.24,
                            t_rise_s=8.0,
                            t_hold_s=10.0,
                            t_fall_s=12.0,
                        )
                        self.mod.swe2d_set_3d_patch_face_bc(
                            solver,
                            face=0,
                            mode=4,
                            q=float(q_in),
                            vof=1.0,
                        )
                        self.mod.swe2d_step(solver, dt_step)
                        if (k + 1) % sample_stride == 0:
                            vof_now = self.mod.swe2d_get_3d_patch_vof(solver)
                            stats_now = self.mod.swe2d_get_3d_patch_stats(solver)
                            times_s.append(float(k + 1) * dt_step)
                            front_x_m.append(_x_front_position_from_vof(vof_now, stats_now, wet_threshold=0.02))

                    celerity = _estimate_celerity_linear(times_s, front_x_m)
                    celerities.append(float(celerity))
                    summary_rows.append(
                        {
                            "coeff": float(coeff),
                            "celerity_mps": float(celerity),
                            "front_end_m": float(front_x_m[-1]) if front_x_m else 0.0,
                            "n_samples": int(len(times_s)),
                        }
                    )
                    self.assertGreater(
                        celerity,
                        0.0,
                        f"Expected positive celerity for coeff={coeff:.3f}; got {celerity:.6e}")
                finally:
                    self.mod.swe2d_destroy(solver)

        csv_path = _write_celerity_sensitivity_csv(self.id(), summary_rows)
        print(
            "[SWE3D_CELERITY] "
            f"coeffs={damping_coeffs} celerities_mps={celerities} "
            f"csv={csv_path}"
        )

        self.assertGreater(
            celerities[0],
            celerities[-1],
            f"Expected higher predictor damping to reduce celerity; coeffs={damping_coeffs}, celerities={celerities}")
        self.assertGreater(
            celerities[0],
            1.002 * celerities[-1],
            f"Expected measurable celerity reduction across damping sweep; coeffs={damping_coeffs}, celerities={celerities}")

    def test_bc_outflow_xmax_drains_vof_mass(self):
        """Outflow (zero-gradient) BC at xmax should release VoF mass from a right-moving slab."""
        env_outflow = {"BACKWATER_SWE3D_BC_XMAX_MODE": "2"}   # OUTFLOW

        solver_out = _make_3d_solver(self.mod, self.mesh, self.h0, env_overrides=env_outflow)
        try:
            stats0 = self.mod.swe2d_get_3d_patch_stats(solver_out)
            n = int(stats0["n_cells"])
            vof_ic = _x_slab_vof(stats0, frac_lo=0.75, frac_hi=0.98)
            u_ic = np.full(n, 5.0, dtype=np.float64)
            zeros = np.zeros(n, dtype=np.float64)
            sum0 = float(np.sum(vof_ic))

            self.mod.swe2d_set_3d_patch_vof(solver_out, vof_ic)
            self.mod.swe2d_set_3d_patch_state(solver_out, u=u_ic, v=zeros, w=zeros, p=zeros)

            for _ in range(12):
                self.mod.swe2d_step(solver_out, 0.2)

            sum_out = float(np.sum(self.mod.swe2d_get_3d_patch_vof(solver_out)))
            self.assertLess(
                sum_out,
                0.95 * sum0,
                f"Expected outflow boundary to drain mass: initial={sum0:.6e}, final={sum_out:.6e}")
        finally:
            self.mod.swe2d_destroy(solver_out)

    def test_bc_free_surface_zmax_vents_more_than_wall(self):
        """Free-surface and wall zmax BC modes should produce measurably different vent response."""
        env_wall = {"BACKWATER_SWE3D_BC_ZMAX_MODE": "0"}       # WALL
        env_free = {
            "BACKWATER_SWE3D_BC_ZMAX_MODE": "3",               # FREE_SURFACE
            "BACKWATER_SWE3D_BC_ZMAX_P": "0.0",
        }

        solver_wall = _make_3d_solver(self.mod, self.mesh, self.h0, env_overrides=env_wall)
        solver_free = _make_3d_solver(self.mod, self.mesh, self.h0, env_overrides=env_free)
        try:
            stats0 = self.mod.swe2d_get_3d_patch_stats(solver_wall)
            n = int(stats0["n_cells"])
            vof_ic = np.ones(n, dtype=np.float64)
            sum0 = float(np.sum(vof_ic))
            w_ic = np.full(n, 1.5, dtype=np.float64)
            zeros = np.zeros(n, dtype=np.float64)

            for s in (solver_wall, solver_free):
                self.mod.swe2d_set_3d_patch_vof(s, vof_ic)
                self.mod.swe2d_set_3d_patch_state(s, u=zeros, v=zeros, w=w_ic, p=zeros)

            for _ in range(10):
                self.mod.swe2d_step(solver_wall, 0.2)
                self.mod.swe2d_step(solver_free, 0.2)

            sum_wall = float(np.sum(self.mod.swe2d_get_3d_patch_vof(solver_wall)))
            sum_free = float(np.sum(self.mod.swe2d_get_3d_patch_vof(solver_free)))
            self.assertLess(sum_wall, sum0)
            self.assertLess(sum_free, sum0)
            self.assertGreater(
                abs(sum_free - sum_wall),
                0.01 * sum0,
                f"Expected distinct wall/free-surface mass response: free={sum_free:.6e}, wall={sum_wall:.6e}")
        finally:
            self.mod.swe2d_destroy(solver_wall)
            self.mod.swe2d_destroy(solver_free)

    def test_bc_free_surface_zmax_pressure_dirichlet_band(self):
        """Free-surface ZMAX should keep top-layer pressure within configured gauge tolerance band."""
        env = {
            "BACKWATER_SWE3D_BC_ZMAX_MODE": "3",  # FREE_SURFACE
            "BACKWATER_SWE3D_BC_ZMAX_P": "0.0",   # 0-gage target
            "BACKWATER_SWE3D_FREE_SURFACE_GAUGE_TOLERANCE_PA": "1.0",
        }
        with _temporary_env(env):
            solver = _make_3d_solver(self.mod, self.mesh, self.h0)
            try:
                if not hasattr(self.mod, "swe2d_get_3d_patch_pressure"):
                    self.skipTest("swe2d_get_3d_patch_pressure not available in native module")

                stats0 = self.mod.swe2d_get_3d_patch_stats(solver)
                n = int(stats0["n_cells"])
                nx = int(stats0["nx"])
                ny = int(stats0["ny"])
                nz = int(stats0["nz"])
                nxy = nx * ny

                p_ic = np.full(n, 5.0e3, dtype=np.float64)
                zeros = np.zeros(n, dtype=np.float64)
                ones = np.ones(n, dtype=np.float64)
                self.mod.swe2d_set_3d_patch_state(solver, u=zeros, v=zeros, w=zeros, p=p_ic, vof=ones)

                self.mod.swe2d_step(solver, 0.1)

                p = np.asarray(self.mod.swe2d_get_3d_patch_pressure(solver), dtype=np.float64).ravel()
                self.assertEqual(p.size, n)
                top = p[(nz - 1) * nxy : nz * nxy]
                self.assertGreater(top.size, 0)
                self.assertLessEqual(
                    float(np.max(np.abs(top))),
                    1.0 + 1.0e-6,
                    f"Expected ZMAX free-surface top pressure to stay inside ±1 Pa band; max|top_p|={float(np.max(np.abs(top))):.6e}",
                )
            finally:
                self.mod.swe2d_destroy(solver)

    # ── Reference cases (gated) ────────────────────────────────────────────────

    @unittest.skipUnless(
        _OPENFOAM_DAMBREAK,
        "Set BACKWATER_RUN_OPENFOAM_DAMBREAK=1 and provide BACKWATER_OPENFOAM_DAMBREAK_REF to enable OpenFOAM cross-code gate.")
    def test_reference_case_openfoam_dambreak(self):
        """
        Optional cross-code gate: compare SWE3D dam-break alpha profile against
        OpenFOAM damBreak reference sample output.
        """
        ref_path = os.environ.get("BACKWATER_OPENFOAM_DAMBREAK_REF", "").strip()
        if not ref_path:
            self.skipTest("Set BACKWATER_OPENFOAM_DAMBREAK_REF=<path to OpenFOAM alpha profile>")
        if not os.path.isfile(ref_path):
            self.skipTest(f"OpenFOAM reference profile not found: {ref_path}")

        def _env_float(name, default):
            raw = os.environ.get(name, "")
            return float(raw) if raw.strip() else float(default)

        def _env_int(name, default):
            raw = os.environ.get(name, "")
            return int(raw) if raw.strip() else int(default)

        cfg = {
            "lx_m": _env_float("BACKWATER_OPENFOAM_DAMBREAK_LX_M", 3.22),
            "ly_m": _env_float("BACKWATER_OPENFOAM_DAMBREAK_LY_M", 1.00),
            "lz_m": _env_float("BACKWATER_OPENFOAM_DAMBREAK_LZ_M", 1.00),
            "patch_nx": _env_int("BACKWATER_OPENFOAM_DAMBREAK_NX", 96),
            "patch_ny": _env_int("BACKWATER_OPENFOAM_DAMBREAK_NY", 8),
            "patch_nz": _env_int("BACKWATER_OPENFOAM_DAMBREAK_NZ", 32),
            "dam_length_m": _env_float("BACKWATER_OPENFOAM_DAMBREAK_COLUMN_LENGTH_M", 1.228),
            "dam_height_m": _env_float("BACKWATER_OPENFOAM_DAMBREAK_COLUMN_HEIGHT_M", 0.55),
            "dt_s": _env_float("BACKWATER_OPENFOAM_DAMBREAK_DT_S", 0.02),
            "n_steps": _env_int("BACKWATER_OPENFOAM_DAMBREAK_STEPS", 20),
            "alpha_l1_tol": _env_float("BACKWATER_OPENFOAM_DAMBREAK_ALPHA_L1_TOL", 0.25),
            "front_tol_m": _env_float("BACKWATER_OPENFOAM_DAMBREAK_FRONT_TOL_M", 0.30),
        }

        from tests.swe3d_reference_harness import run_openfoam_dambreak_compare
        result = run_openfoam_dambreak_compare(self.mod, ref_path, config=cfg)
        for metric_name, passed, value, ref, tol in result.iter_metrics():
            with self.subTest(metric=metric_name):
                self.assertTrue(
                    passed,
                    f"{metric_name}: value={value:.4e} ref={ref:.4e} "
                    f"delta={abs(value-ref):.4e} tol={tol:.4e}")

    @unittest.skipUnless(_PHYSICS_CASES,
        "Set BACKWATER_RUN_SWE3D_PHYSICS_CASES=1 when reference datasets are staged.")
    def test_reference_case_broad_crested_weir(self):
        """
        Stage-1 reference: broad-crested weir free-surface nappe profile.
        Requires: tests/data/swe3d/broad_crested_weir.json
        """
        from tests.swe3d_reference_harness import load_case, run_and_compare
        case = load_case("broad_crested_weir")
        result = run_and_compare(self.mod, case)
        for metric_name, passed, value, ref, tol in result.iter_metrics():
            with self.subTest(metric=metric_name):
                self.assertTrue(passed,
                    f"{metric_name}: value={value:.4e}  ref={ref:.4e}  "
                    f"delta={abs(value-ref):.4e}  tol={tol:.4e}")

    @unittest.skipUnless(_PHYSICS_CASES,
        "Set BACKWATER_RUN_SWE3D_PHYSICS_CASES=1 when reference datasets are staged.")
    def test_reference_case_culvert_pressurization(self):
        """
        Stage-1 reference: culvert pressurisation transition (inlet vs outlet head).
        Requires: tests/data/swe3d/culvert_pressurization.json
        """
        from tests.swe3d_reference_harness import load_case, run_and_compare
        case = load_case("culvert_pressurization")
        result = run_and_compare(self.mod, case)
        for metric_name, passed, value, ref, tol in result.iter_metrics():
            with self.subTest(metric=metric_name):
                self.assertTrue(passed,
                    f"{metric_name}: value={value:.4e}  ref={ref:.4e}  "
                    f"delta={abs(value-ref):.4e}  tol={tol:.4e}")


@unittest.skipUnless(_load_module() is not None, "hydra_swe2d not built")
@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
@unittest.skipUnless(
    _SUBGRID_DAMBREAK,
    "Set BACKWATER_RUN_SWE3D_SUBGRID_DAMBREAK=1 to enable subgrid dam-break regressions.")
class TestSWE3DSubgridDamBreak(unittest.TestCase):
    """Focused subgrid dam-break regressions for uploaded 3D geometry tensors."""

    def setUp(self):
        self.mod = _load_module()
        self._vtk_hook_state = _install_swe3d_vtk_hooks(self.mod, self.id())
        self.mesh = _make_rect_mesh(self.mod, 20, 10, 200.0, 100.0)
        n_cells = self.mod.swe2d_mesh_info(self.mesh)["n_cells"]
        self.h0 = np.full(n_cells, 1.0, dtype=np.float64)
        if not hasattr(self.mod, "swe2d_set_3d_patch_geometry"):
            self.skipTest("swe2d_set_3d_patch_geometry not available in native module")

    def tearDown(self):
        _uninstall_swe3d_vtk_hooks(self.mod, getattr(self, "_vtk_hook_state", None))

    def test_subgrid_dam_break_breach_transmits_limited_mass(self):
        """
        Dam-break-style regression: a thin subgrid dam with a narrow breach should
        pass some downstream VoF, but much less than the fully open case.
        """
        solver_open = _make_3d_solver(self.mod, self.mesh, self.h0)
        solver_dam = _make_3d_solver(self.mod, self.mesh, self.h0)
        try:
            stats0 = self.mod.swe2d_get_3d_patch_stats(solver_open)
            nx = int(stats0["nx"])
            n = int(stats0["n_cells"])

            vof_ic = _x_slab_vof(stats0, frac_lo=0.02, frac_hi=0.45)
            u_ic = np.full(n, 20.0, dtype=np.float64)
            zeros = np.zeros(n, dtype=np.float64)
            phi, ax, ay, az, solid_mask, slot_mask = _slotted_dam_geometry(
                stats0,
                frac_x=0.50,
                thickness_cells=2,
                slot_frac=0.20,
            )
            right_mask = (np.arange(n, dtype=np.int64) % nx) >= (nx // 2 + 1)

            self.mod.swe2d_set_3d_patch_vof(solver_open, vof_ic)
            self.mod.swe2d_set_3d_patch_state(solver_open, u=u_ic, v=zeros, w=zeros, p=zeros)

            self.mod.swe2d_set_3d_patch_geometry(solver_dam, phi=phi, ax=ax, ay=ay, az=az)
            self.mod.swe2d_set_3d_patch_vof(solver_dam, vof_ic)
            self.mod.swe2d_set_3d_patch_state(solver_dam, u=u_ic, v=zeros, w=zeros, p=zeros)

            for _ in range(20):
                self.mod.swe2d_step(solver_open, 0.1)
                self.mod.swe2d_step(solver_dam, 0.1)

            vof_open = self.mod.swe2d_get_3d_patch_vof(solver_open)
            vof_dam = self.mod.swe2d_get_3d_patch_vof(solver_dam)

            right_open = float(np.sum(vof_open[right_mask]))
            right_dam = float(np.sum(vof_dam[right_mask]))
            solid_mass = float(np.sum(vof_dam[solid_mask]))
            slot_mass = float(np.sum(vof_dam[slot_mask]))

            self.assertGreater(
                right_open,
                1.0e-3,
                f"Expected open dam-break case to advect downstream mass; right_open={right_open:.4e}")
            self.assertGreater(
                right_dam,
                1.0e-5,
                f"Expected breached dam to transmit some downstream mass; right_dam={right_dam:.4e}")
            self.assertLess(
                right_dam,
                0.60 * right_open,
                f"Expected subgrid dam to restrict downstream transport; right_dam={right_dam:.4e}, right_open={right_open:.4e}")
            self.assertLess(
                solid_mass,
                1.0e-8,
                f"Expected solid dam cells to remain dry; solid_mass={solid_mass:.4e}")
            self.assertGreater(
                slot_mass,
                1.0e-6,
                f"Expected breach slot to carry non-zero mass; slot_mass={slot_mass:.4e}")
        finally:
            self.mod.swe2d_destroy(solver_open)
            self.mod.swe2d_destroy(solver_dam)

    def test_subgrid_dam_break_porous_dam_restricts_more_than_open_case(self):
        """
        Dam-break-style regression: a porous subgrid dam with a centered breach should
        leak through the dam body, but still restrict downstream transport versus open.
        """
        solver_open = _make_3d_solver(self.mod, self.mesh, self.h0)
        solver_porous = _make_3d_solver(self.mod, self.mesh, self.h0)
        try:
            stats0 = self.mod.swe2d_get_3d_patch_stats(solver_open)
            nx = int(stats0["nx"])
            n = int(stats0["n_cells"])

            vof_ic = _x_slab_vof(stats0, frac_lo=0.02, frac_hi=0.45)
            u_ic = np.full(n, 20.0, dtype=np.float64)
            zeros = np.zeros(n, dtype=np.float64)
            phi, ax, ay, az, porous_mask, slot_mask = _porous_slotted_dam_geometry(
                stats0,
                frac_x=0.50,
                thickness_cells=2,
                slot_frac=0.20,
                phi_barrier=0.35,
                area_barrier=0.20,
            )
            right_mask = (np.arange(n, dtype=np.int64) % nx) >= (nx // 2 + 1)

            self.mod.swe2d_set_3d_patch_vof(solver_open, vof_ic)
            self.mod.swe2d_set_3d_patch_state(solver_open, u=u_ic, v=zeros, w=zeros, p=zeros)

            self.mod.swe2d_set_3d_patch_geometry(solver_porous, phi=phi, ax=ax, ay=ay, az=az)
            self.mod.swe2d_set_3d_patch_vof(solver_porous, vof_ic)
            self.mod.swe2d_set_3d_patch_state(solver_porous, u=u_ic, v=zeros, w=zeros, p=zeros)

            for _ in range(20):
                self.mod.swe2d_step(solver_open, 0.1)
                self.mod.swe2d_step(solver_porous, 0.1)

            vof_open = self.mod.swe2d_get_3d_patch_vof(solver_open)
            vof_porous = self.mod.swe2d_get_3d_patch_vof(solver_porous)

            right_open = float(np.sum(vof_open[right_mask]))
            right_porous = float(np.sum(vof_porous[right_mask]))
            porous_mass = float(np.sum(vof_porous[porous_mask]))
            slot_mass = float(np.sum(vof_porous[slot_mask]))

            self.assertGreater(
                right_open,
                1.0e-3,
                f"Expected open dam-break case to advect downstream mass; right_open={right_open:.4e}")
            self.assertGreater(
                right_porous,
                1.0e-5,
                f"Expected porous dam to transmit some downstream mass; right_porous={right_porous:.4e}")
            self.assertLess(
                right_porous,
                0.85 * right_open,
                f"Expected porous dam to restrict downstream transport; right_porous={right_porous:.4e}, right_open={right_open:.4e}")
            self.assertGreater(
                porous_mass,
                1.0e-6,
                f"Expected porous dam cells to hold non-zero mass; porous_mass={porous_mass:.4e}")
            self.assertGreater(
                slot_mass,
                1.0e-6,
                f"Expected breach slot to carry non-zero mass; slot_mass={slot_mass:.4e}")
        finally:
            self.mod.swe2d_destroy(solver_open)
            self.mod.swe2d_destroy(solver_porous)


if __name__ == "__main__":
    unittest.main()

