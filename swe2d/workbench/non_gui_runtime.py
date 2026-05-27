from __future__ import annotations

import os
import time
import urllib.parse
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _env_flag(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return bool(default)
    return raw not in {"0", "false", "no", "off"}


def _env_float(name: str, default: float) -> float:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return float(default)
    try:
        val = float(raw)
    except Exception:
        return float(default)
    return float(val) if np.isfinite(val) else float(default)


def _env_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def build_mesh_snapshot_rows(snapshot_timesteps: Sequence[Tuple[object, object, object, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not snapshot_timesteps:
        return rows

    for snap in snapshot_timesteps:
        try:
            t_s, h, hu, hv = snap
            hh = np.asarray(h, dtype=np.float64).ravel()
            huu = np.asarray(hu, dtype=np.float64).ravel()
            hvv = np.asarray(hv, dtype=np.float64).ravel()
            n = min(hh.size, huu.size, hvv.size)
            ts_val = float(t_s)
            for ci in range(n):
                rows.append(
                    {
                        "t_s": ts_val,
                        "cell_id": int(ci),
                        "h": float(hh[ci]),
                        "hu": float(huu[ci]),
                        "hv": float(hvv[ci]),
                    }
                )
        except Exception:
            continue

    return rows


def parse_obj_scale_value(raw_value: object) -> Tuple[float, float, float]:
    if raw_value is None:
        return (1.0, 1.0, 1.0)

    if isinstance(raw_value, (int, float, np.integer, np.floating)):
        s = float(raw_value)
        if not np.isfinite(s):
            raise ValueError("scale value must be finite")
        return (s, s, s)

    txt = str(raw_value).strip()
    if not txt:
        return (1.0, 1.0, 1.0)

    tokens = [p for p in txt.replace(",", " ").replace(";", " ").split() if p]
    if len(tokens) == 1:
        s = float(tokens[0])
        if not np.isfinite(s):
            raise ValueError("scale value must be finite")
        return (s, s, s)
    if len(tokens) >= 3:
        sx = float(tokens[0])
        sy = float(tokens[1])
        sz = float(tokens[2])
        if not (np.isfinite(sx) and np.isfinite(sy) and np.isfinite(sz)):
            raise ValueError("scale tuple must contain finite values")
        return (sx, sy, sz)

    raise ValueError("scale value must be a scalar or sx,sy,sz tuple")


def resolve_obj_model_path(
    *,
    raw_path: str,
    model_gpkg_path: str,
    project_file_path: str,
    module_dir: str,
    cwd: str,
) -> str:
    path_txt = str(raw_path or "").strip().strip('"').strip("'")
    if not path_txt:
        return ""

    if path_txt.lower().startswith("file://"):
        try:
            parsed = urllib.parse.urlparse(path_txt)
            uri_path = urllib.parse.unquote(str(parsed.path or ""))
            if uri_path:
                path_txt = uri_path
        except Exception:
            pass

    candidates: List[str] = []
    if os.path.isabs(path_txt):
        candidates.append(path_txt)
    else:
        if model_gpkg_path:
            candidates.append(os.path.join(os.path.dirname(model_gpkg_path), path_txt))
        if project_file_path:
            candidates.append(os.path.join(os.path.dirname(project_file_path), path_txt))
        if module_dir:
            candidates.append(os.path.join(module_dir, path_txt))
        if cwd:
            candidates.append(os.path.join(cwd, path_txt))

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return os.path.abspath(candidate)

    if candidates:
        return os.path.abspath(candidates[0])
    return path_txt


def build_patch_spec_from_stats(
    *,
    patch_stats: Dict[str, object],
    swe3d_env_overrides: Dict[str, str],
    patch_grid_spec_cls: object,
) -> Optional[object]:
    if patch_grid_spec_cls is None:
        return None

    try:
        nx = int(patch_stats.get("nx", 0) or 0)
        ny = int(patch_stats.get("ny", 0) or 0)
        nz = int(patch_stats.get("nz", 0) or 0)
        dx = float(patch_stats.get("dx", 0.0) or 0.0)
        dy = float(patch_stats.get("dy", 0.0) or 0.0)
        dz = float(patch_stats.get("dz", 0.0) or 0.0)
        ox = float(swe3d_env_overrides.get("BACKWATER_SWE3D_PATCH_ORIGIN_X", "nan"))
        oy = float(swe3d_env_overrides.get("BACKWATER_SWE3D_PATCH_ORIGIN_Y", "nan"))
        oz = float(swe3d_env_overrides.get("BACKWATER_SWE3D_PATCH_ORIGIN_Z", "nan"))
    except Exception:
        return None

    if nx <= 0 or ny <= 0 or nz <= 0 or dx <= 0.0 or dy <= 0.0 or dz <= 0.0:
        return None
    if not (np.isfinite(ox) and np.isfinite(oy) and np.isfinite(oz)):
        return None

    return patch_grid_spec_cls(
        nx=nx,
        ny=ny,
        nz=nz,
        dx=dx,
        dy=dy,
        dz=dz,
        origin_x=ox,
        origin_y=oy,
        origin_z=oz,
    )


def boundary_edge_owner_cells(
    *,
    mesh_data: Optional[Dict[str, object]],
    edge_n0: np.ndarray,
    edge_n1: np.ndarray,
) -> np.ndarray:
    owners = np.full(int(edge_n0.size), -1, dtype=np.int32)
    if mesh_data is None:
        return owners

    edge_owner: Dict[Tuple[int, int], int] = {}
    if "cell_face_offsets" in mesh_data and "cell_face_nodes" in mesh_data:
        offs = np.asarray(mesh_data["cell_face_offsets"], dtype=np.int32).ravel()
        faces = np.asarray(mesh_data["cell_face_nodes"], dtype=np.int32).ravel()
        for ci in range(max(0, int(offs.size) - 1)):
            s = int(offs[ci])
            e = int(offs[ci + 1])
            poly = faces[s:e]
            if poly.size < 2:
                continue
            for k in range(int(poly.size)):
                a = int(poly[k])
                b = int(poly[(k + 1) % int(poly.size)])
                key = (a, b) if a < b else (b, a)
                edge_owner[key] = ci if key not in edge_owner else -1
    else:
        tris = np.asarray(mesh_data["cell_nodes"], dtype=np.int32).reshape((-1, 3))
        for ci, tri in enumerate(tris):
            for k in range(3):
                a = int(tri[k])
                b = int(tri[(k + 1) % 3])
                key = (a, b) if a < b else (b, a)
                edge_owner[key] = int(ci) if key not in edge_owner else -1

    n = min(int(edge_n0.size), int(edge_n1.size))
    for i in range(n):
        a = int(edge_n0[i])
        b = int(edge_n1[i])
        key = (a, b) if a < b else (b, a)
        owner = int(edge_owner.get(key, -1))
        if owner >= 0:
            owners[i] = owner

    return owners


def build_experimental_3d_interface_contract_arrays(
    *,
    wb: object,
    patch_stats: Dict[str, object],
    bc_n0: np.ndarray,
    bc_n1: np.ndarray,
    bc_tp: np.ndarray,
    bc_inflow_q: int,
    bc_ts_flow: int,
    bc_ts_stage: int,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    if getattr(wb, "_mesh_data", None) is None:
        return None

    edge_n0 = np.asarray(bc_n0, dtype=np.int32).ravel()
    edge_n1 = np.asarray(bc_n1, dtype=np.int32).ravel()
    edge_tp = np.asarray(bc_tp, dtype=np.int32).ravel()
    n = min(int(edge_n0.size), int(edge_n1.size), int(edge_tp.size))
    if n <= 0:
        return None
    edge_n0 = edge_n0[:n]
    edge_n1 = edge_n1[:n]
    edge_tp = edge_tp[:n]

    node_x = np.asarray(wb._mesh_data["node_x"], dtype=np.float64).ravel()
    node_y = np.asarray(wb._mesh_data["node_y"], dtype=np.float64).ravel()
    if node_x.size <= 0 or node_y.size <= 0:
        return None

    valid_nodes = (
        (edge_n0 >= 0)
        & (edge_n1 >= 0)
        & (edge_n0 < int(node_x.size))
        & (edge_n1 < int(node_x.size))
    )
    if not np.any(valid_nodes):
        return None

    edge_len = np.zeros(n, dtype=np.float64)
    mx = np.zeros(n, dtype=np.float64)
    my = np.zeros(n, dtype=np.float64)
    vv = np.where(valid_nodes)[0]
    edge_len[vv] = np.hypot(node_x[edge_n1[vv]] - node_x[edge_n0[vv]], node_y[edge_n1[vv]] - node_y[edge_n0[vv]])
    mx[vv] = 0.5 * (node_x[edge_n0[vv]] + node_x[edge_n1[vv]])
    my[vv] = 0.5 * (node_y[edge_n0[vv]] + node_y[edge_n1[vv]])

    xmin = float(np.min(node_x))
    xmax = float(np.max(node_x))
    ymin = float(np.min(node_y))
    ymax = float(np.max(node_y))
    d = np.vstack(
        [
            np.abs(mx - xmin),
            np.abs(mx - xmax),
            np.abs(my - ymin),
            np.abs(my - ymax),
        ]
    )
    side_idx = np.argmin(d, axis=0)
    face_nx = np.zeros(n, dtype=np.float64)
    face_ny = np.zeros(n, dtype=np.float64)
    face_nz = np.zeros(n, dtype=np.float64)
    face_nx[side_idx == 0] = -1.0
    face_nx[side_idx == 1] = 1.0
    face_ny[side_idx == 2] = -1.0
    face_ny[side_idx == 3] = 1.0

    owners = np.asarray(wb._boundary_edge_owner_cells(edge_n0=edge_n0, edge_n1=edge_n1), dtype=np.int32)
    patch_height = max(
        1.0e-6,
        float(patch_stats.get("dz", 0.0) or 0.0) * float(patch_stats.get("nz", 0.0) or 0.0),
    )
    face_area = edge_len * patch_height

    active_bc_types = np.asarray([bc_inflow_q, 3, 4, 6, 7, bc_ts_flow, bc_ts_stage], dtype=np.int32)
    base_mask = (
        valid_nodes
        & (owners >= 0)
        & np.isfinite(face_area)
        & (face_area > 0.0)
        & np.isfinite(face_nx)
        & np.isfinite(face_ny)
        & np.isfinite(face_nz)
    )
    use_mask = base_mask & np.isin(edge_tp, active_bc_types)
    if not np.any(use_mask):
        use_mask = base_mask
        if np.any(use_mask):
            wb._log(
                "3D coupling contract: no open/forcing BC edges found; "
                "falling back to all boundary edges."
            )

    if not np.any(use_mask):
        return None

    return (
        np.asarray(owners[use_mask], dtype=np.int32),
        np.asarray(face_area[use_mask], dtype=np.float64),
        np.asarray(face_nx[use_mask], dtype=np.float64),
        np.asarray(face_ny[use_mask], dtype=np.float64),
        np.asarray(face_nz[use_mask], dtype=np.float64),
    )


def upload_experimental_3d_interface_contract(
    *,
    wb: object,
    backend: object,
    patch_stats: Dict[str, object],
    bc_n0: np.ndarray,
    bc_n1: np.ndarray,
    bc_tp: np.ndarray,
    coupling_mode: int,
    coupling_mode_enum: object,
) -> None:
    if coupling_mode_enum is None:
        return

    if int(coupling_mode) == int(coupling_mode_enum.OFF):
        try:
            if backend is not None and hasattr(backend, "clear_interface_contract"):
                backend.clear_interface_contract()
        except Exception:
            pass
        wb._log("3D coupling mode: off (no interface contract upload).")
        return

    if backend is None:
        raise RuntimeError("2D-3D coupling requested, but backend is unavailable.")
    for name in (
        "create_interface_contract",
        "is_interface_contract_valid",
        "upload_interface_contract",
        "is_interface_contract_uploaded",
    ):
        if not hasattr(backend, name):
            raise RuntimeError(
                "2D-3D coupling requested, but native contract API is unavailable; "
                "rebuild native module with Phase 7 contract support."
            )

    arrays = wb._build_experimental_3d_interface_contract_arrays(
        patch_stats=patch_stats,
        bc_n0=bc_n0,
        bc_n1=bc_n1,
        bc_tp=bc_tp,
    )
    if arrays is None:
        raise RuntimeError(
            "2D-3D coupling requested, but no valid boundary-edge contract faces were constructed."
        )
    cell2d, face_area, face_nx, face_ny, face_nz = arrays

    contract = backend.create_interface_contract(
        cell2d=cell2d,
        face_area=face_area,
        face_nx=face_nx,
        face_ny=face_ny,
        face_nz=face_nz,
    )
    if contract is None or not bool(backend.is_interface_contract_valid(contract)):
        raise RuntimeError("failed to create valid 2D-3D interface contract")
    if not bool(backend.upload_interface_contract(contract)):
        raise RuntimeError("native upload of 2D-3D interface contract failed")

    mode_label = "unknown"
    if int(coupling_mode) == int(coupling_mode_enum.ONE_WAY_2D_TO_3D):
        mode_label = "one-way (2D -> 3D)"
    elif int(coupling_mode) == int(coupling_mode_enum.TWO_WAY_2D_3D):
        mode_label = "two-way (2D <-> 3D)"

    uploaded = bool(backend.is_interface_contract_uploaded())
    wb._log(
        "3D coupling contract uploaded: "
        f"mode={mode_label}, faces={int(cell2d.size)}, "
        f"area_sum={float(np.sum(face_area)):.6e}, uploaded={uploaded}"
    )


def initialize_experimental_3d_patch_state(
    *,
    wb: object,
    backend: object,
    patch_stats: Dict[str, object],
    swe3d_env_overrides: Dict[str, str],
    bc_n0: np.ndarray,
    bc_n1: np.ndarray,
    bc_tp: np.ndarray,
    bc_vl: np.ndarray,
    log_notes: bool,
    bc_inflow_q: int,
    bc_ts_flow: int,
    bc_ts_stage: int,
    coupling_mode_enum: object,
) -> None:
    if backend is None or not hasattr(backend, "supports_3d_patch_state_upload"):
        if log_notes:
            wb._log("3D patch initial-state seeding skipped: backend state-upload API unavailable.")
        return
    if not bool(backend.supports_3d_patch_state_upload()):
        if log_notes:
            wb._log("3D patch initial-state seeding skipped: native 3D state upload API unavailable.")
        return

    spec = wb._build_patch_spec_from_stats(patch_stats, swe3d_env_overrides)
    if spec is None:
        if log_notes:
            wb._log("3D patch initial-state seeding skipped: invalid patch spec.")
        return

    nx = int(getattr(spec, "nx", 0))
    ny = int(getattr(spec, "ny", 0))
    nz = int(getattr(spec, "nz", 0))
    dz = float(getattr(spec, "dz", 0.0))
    oz = float(getattr(spec, "origin_z", 0.0))
    if nx <= 0 or ny <= 0 or nz <= 0 or dz <= 0.0:
        if log_notes:
            wb._log("3D patch initial-state seeding skipped: invalid patch dimensions.")
        return

    terrain_surface = wb._build_patch_terrain_surface(spec)
    if terrain_surface is not None:
        bed = np.where(np.isfinite(terrain_surface), terrain_surface, float(oz)).astype(np.float64)
    else:
        bed = np.full((ny, nx), float(oz), dtype=np.float64)

    def _manning_normal_depth_from_profile(
        q_abs: float,
        bed_profile: np.ndarray,
        spacing: float,
        slope: float,
        manning_n: float,
        unit_factor: float,
    ) -> Optional[Tuple[float, float]]:
        q_target = abs(float(q_abs))
        if not np.isfinite(q_target) or q_target <= 1.0e-12:
            return None

        z = np.asarray(bed_profile, dtype=np.float64).ravel()
        if z.size <= 0:
            return None
        good = np.isfinite(z)
        if not np.any(good):
            return None
        z = z[good]
        if z.size <= 0:
            return None

        ds = max(float(spacing), 1.0e-9)
        s_eff = max(float(slope), 1.0e-12)
        n_eff = max(float(manning_n), 1.0e-12)
        u_eff = float(unit_factor)
        if not np.isfinite(u_eff) or u_eff <= 0.0:
            u_eff = 1.0

        z_min = float(np.min(z))
        z_rel = z - z_min
        relief = float(np.max(z_rel))

        def _q_for_depth(depth: float) -> float:
            d = max(0.0, float(depth))
            h = np.maximum(0.0, d - z_rel)
            if h.size <= 0:
                return 0.0
            area = float(np.sum(h) * ds)
            if area <= 0.0:
                return 0.0
            wetted_width = float(np.count_nonzero(h > 1.0e-9)) * ds
            wetted_perim = max(1.0e-9, wetted_width + 2.0 * d)
            radius = area / wetted_perim
            if radius <= 0.0:
                return 0.0
            return (u_eff / n_eff) * area * (radius ** (2.0 / 3.0)) * (s_eff ** 0.5)

        d_lo = 0.0
        d_hi = max(1.0, relief + 1.0)
        q_hi = _q_for_depth(d_hi)
        grow_iter = 0
        while (not np.isfinite(q_hi) or q_hi < q_target) and grow_iter < 48:
            d_hi *= 1.6
            q_hi = _q_for_depth(d_hi)
            grow_iter += 1
            if d_hi > 1.0e6:
                break
        if not np.isfinite(q_hi) or q_hi < q_target:
            return None

        for _ in range(80):
            d_mid = 0.5 * (d_lo + d_hi)
            q_mid = _q_for_depth(d_mid)
            if not np.isfinite(q_mid):
                return None
            if q_mid >= q_target:
                d_hi = d_mid
            else:
                d_lo = d_mid

        d_star = max(0.0, d_hi)
        wse_star = z_min + d_star
        return d_star, wse_star

    def _q_boundary_normal_depth_seed() -> Optional[Tuple[int, np.ndarray, float, float, float]]:
        enabled = bool(
            getattr(wb, "experimental_3d_patch_normal_depth_enable_chk", None)
            and wb.experimental_3d_patch_normal_depth_enable_chk.isChecked()
        )
        if not enabled:
            return None

        faces = ("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX")
        face_candidates: List[Tuple[float, int, str, float]] = []
        for face_idx, face_name in enumerate(faces):
            face_key = str(face_name).lower()
            mode_combo = getattr(wb, f"experimental_3d_bc_{face_key}_mode_combo", None)
            q_spin = getattr(wb, f"experimental_3d_bc_{face_key}_q_spin", None)
            if mode_combo is None or q_spin is None:
                continue
            try:
                mode_raw = mode_combo.currentData()
                if mode_raw is None:
                    mode_raw = mode_combo.currentIndex()
                mode_val = int(mode_raw)
            except Exception:
                mode_val = 0
            if mode_val != 4:
                continue
            try:
                q_val = float(q_spin.value())
            except Exception:
                q_val = 0.0
            if not np.isfinite(q_val) or abs(q_val) <= 1.0e-12:
                continue
            face_candidates.append((abs(q_val), face_idx, str(face_name), q_val))

        if not face_candidates:
            return None

        face_candidates.sort(key=lambda rec: rec[0], reverse=True)
        _, face_idx, face_name, q_val = face_candidates[0]

        if face_name == "XMIN":
            profile = bed[:, 0]
            spacing = float(getattr(spec, "dy", 0.0))
            side = 0
        elif face_name == "XMAX":
            profile = bed[:, -1]
            spacing = float(getattr(spec, "dy", 0.0))
            side = 1
        elif face_name == "YMIN":
            profile = bed[0, :]
            spacing = float(getattr(spec, "dx", 0.0))
            side = 2
        elif face_name == "YMAX":
            profile = bed[-1, :]
            spacing = float(getattr(spec, "dx", 0.0))
            side = 3
        else:
            return None

        slope = 0.001
        if hasattr(wb, "experimental_3d_patch_normal_depth_slope_spin"):
            try:
                slope = float(wb.experimental_3d_patch_normal_depth_slope_spin.value())
            except Exception:
                slope = 0.001

        manning_n = 0.02
        if hasattr(wb, "experimental_3d_patch_normal_depth_n_spin"):
            try:
                manning_n = float(wb.experimental_3d_patch_normal_depth_n_spin.value())
            except Exception:
                manning_n = 0.02

        use_us = bool(
            getattr(wb, "experimental_3d_patch_normal_depth_us_units_chk", None)
            and wb.experimental_3d_patch_normal_depth_us_units_chk.isChecked()
        )
        unit_factor = 1.49 if use_us else 1.0

        solved = _manning_normal_depth_from_profile(
            q_abs=abs(float(q_val)),
            bed_profile=np.asarray(profile, dtype=np.float64),
            spacing=max(float(spacing), 1.0e-9),
            slope=slope,
            manning_n=manning_n,
            unit_factor=unit_factor,
        )
        if solved is None:
            if log_notes:
                wb._log(
                    "3D patch Manning normal-depth init: unable to solve depth from Q boundary "
                    f"(face={face_name}, |Q|={abs(float(q_val)):.6g}, S={slope:.6g}, n={manning_n:.6g}, u={unit_factor:.3g})."
                )
            return None

        depth_scalar, wse_scalar = solved
        side_profile_depth = np.maximum(0.0, float(wse_scalar) - np.asarray(profile, dtype=np.float64))
        if log_notes:
            wb._log(
                "3D patch Manning normal-depth init: "
                f"face={face_name}, |Q|={abs(float(q_val)):.6g}, S={slope:.6g}, n={manning_n:.6g}, u={unit_factor:.3g}, "
                f"depth={float(depth_scalar):.6g}, wse={float(wse_scalar):.6g}"
            )
        return side, np.asarray(side_profile_depth, dtype=np.float64), float(wse_scalar), float(depth_scalar), float(q_val)

    mode = str(wb.initial_condition_combo.currentData() if hasattr(wb, "initial_condition_combo") else "dry")
    depth = np.zeros((ny, nx), dtype=np.float64)
    if mode == "uniform_depth":
        depth[:, :] = max(0.0, float(wb.initial_depth_spin.value()))
    elif mode == "uniform_wse":
        wse0 = float(wb.initial_wse_spin.value())
        depth = np.maximum(0.0, wse0 - bed)

    bc_types = np.asarray(bc_tp, dtype=np.int32).ravel() if bc_tp is not None else np.empty(0, dtype=np.int32)
    forced_types = np.asarray([bc_inflow_q, 3, 6, 7, bc_ts_flow, bc_ts_stage], dtype=np.int32)
    stage_types = np.asarray([3, bc_ts_stage], dtype=np.int32)
    forced_bc_present = bool(bc_types.size > 0 and np.any(np.isin(bc_types, forced_types)))
    stage_bc_present = bool(bc_types.size > 0 and np.any(np.isin(bc_types, stage_types)))

    if bc_n0 is not None and bc_n1 is not None and bc_types.size == int(bc_n0.size) and bc_types.size == int(bc_n1.size) and bc_types.size > 0:
        node_x = np.asarray(wb._mesh_data["node_x"], dtype=np.float64).ravel()
        node_y = np.asarray(wb._mesh_data["node_y"], dtype=np.float64).ravel()
        mx = 0.5 * (node_x[bc_n0] + node_x[bc_n1])
        my = 0.5 * (node_y[bc_n0] + node_y[bc_n1])
        d = np.vstack([
            np.abs(mx - float(np.min(node_x))),
            np.abs(mx - float(np.max(node_x))),
            np.abs(my - float(np.min(node_y))),
            np.abs(my - float(np.max(node_y))),
        ])
        side_idx = np.argmin(d, axis=0)

        def _apply_side_depth_max(side: int, side_depth: np.ndarray) -> None:
            if side == 0:
                depth[:, 0] = np.maximum(depth[:, 0], side_depth)
            elif side == 1:
                depth[:, -1] = np.maximum(depth[:, -1], side_depth)
            elif side == 2:
                depth[0, :] = np.maximum(depth[0, :], side_depth)
            elif side == 3:
                depth[-1, :] = np.maximum(depth[-1, :], side_depth)

        if mode == "dry" and forced_bc_present:
            h_min_val = float(wb.h_min_spin.value()) if hasattr(wb, "h_min_spin") else 1.0e-6
            prime_depth = max(h_min_val * 100.0, 1.0e-4)
            forced_mask = np.isin(bc_types, forced_types)
            for side in np.unique(side_idx[forced_mask]).tolist():
                if int(side) in (0, 1):
                    side_depth = np.full((ny,), prime_depth, dtype=np.float64)
                else:
                    side_depth = np.full((nx,), prime_depth, dtype=np.float64)
                _apply_side_depth_max(int(side), side_depth)

        if stage_bc_present and bc_vl is not None:
            bc_vals = np.asarray(bc_vl, dtype=np.float64).ravel()
            if bc_vals.size == bc_types.size:
                stage_mask = np.isin(bc_types, stage_types)
                for side in np.unique(side_idx[stage_mask]).tolist():
                    side = int(side)
                    side_stage_vals = bc_vals[(stage_mask) & (side_idx == side)]
                    if side_stage_vals.size <= 0:
                        continue
                    stage_wse = float(np.nanmax(side_stage_vals))
                    if not np.isfinite(stage_wse):
                        continue
                    if side == 0:
                        side_depth = np.maximum(0.0, stage_wse - bed[:, 0])
                    elif side == 1:
                        side_depth = np.maximum(0.0, stage_wse - bed[:, -1])
                    elif side == 2:
                        side_depth = np.maximum(0.0, stage_wse - bed[0, :])
                    else:
                        side_depth = np.maximum(0.0, stage_wse - bed[-1, :])
                    _apply_side_depth_max(side, np.asarray(side_depth, dtype=np.float64))

        nd_seed = _q_boundary_normal_depth_seed()
        if nd_seed is not None:
            side, side_depth, nd_wse, _nd_depth, _q_raw = nd_seed
            _apply_side_depth_max(int(side), np.asarray(side_depth, dtype=np.float64))
            seed_domain = bool(
                getattr(wb, "experimental_3d_patch_normal_depth_seed_domain_chk", None)
                and wb.experimental_3d_patch_normal_depth_seed_domain_chk.isChecked()
            )
            if seed_domain:
                depth[:, :] = np.maximum(depth, np.maximum(0.0, float(nd_wse) - bed))
                if log_notes:
                    wb._log(
                        "3D patch Manning normal-depth init: applied domain-wide free-surface seed from normal-depth WSE."
                    )

    wse = bed + np.maximum(0.0, depth)
    zc = float(oz) + (np.arange(nz, dtype=np.float64) + 0.5) * float(dz)
    vof3 = (zc[:, None, None] <= wse[None, :, :]).astype(np.float64)
    vof = vof3.ravel(order="C")

    try:
        n_cells = int(vof.size)
        zeros = np.zeros(n_cells, dtype=np.float64)
        backend.set_3d_patch_state(u=zeros, v=zeros, w=zeros, p=zeros, vof=vof)
    except Exception as exc:
        try:
            backend.set_3d_patch_state(vof=vof)
        except Exception:
            if log_notes:
                wb._log(f"3D patch initial-state seeding failed: {exc}")
            return

    wet_columns = int(np.count_nonzero(np.max(vof3, axis=0) > 0.5))
    wet_cells = int(np.count_nonzero(vof > 0.5))
    if log_notes:
        wb._log(
            "3D patch initial VoF seeding: "
            f"mode={mode}, wet_columns={wet_columns}/{nx * ny}, wet_cells={wet_cells}/{vof.size}, "
            f"vof_sum={float(np.sum(vof)):.6e}"
        )
    if forced_bc_present and log_notes:
        coupling_mode = wb._experimental_3d_selected_coupling_mode()
        coupling_off = (
            coupling_mode_enum is None
            or int(coupling_mode) == int(coupling_mode_enum.OFF)
        )
        if coupling_off:
            wb._log(
                "3D BC note: uncoupled 3D patch mode does not yet impose time-varying 2D edge BC forcing each step; "
                "current run uses startup VoF seeding only."
            )
        else:
            wb._log(
                "3D BC note: coupling mode is active; per-step 2D-3D exchange uses uploaded interface contract faces."
            )


def run_experimental_3d_obj_method_probe(
    *,
    wb: object,
    backend_builder: object,
    method_name: str,
    phi: np.ndarray,
    ax: np.ndarray,
    ay: np.ndarray,
    az: np.ndarray,
    swe3d_env_overrides: Dict[str, str],
    bc_n0: np.ndarray,
    bc_n1: np.ndarray,
    bc_tp: np.ndarray,
    bc_vl: np.ndarray,
    probe_steps: int,
    coupling_mode_enum: object,
) -> Dict[str, float]:
    result: Dict[str, float] = {
        "ok": 0.0,
        "steps": float(max(1, int(probe_steps))),
        "vof_sum0": 0.0,
        "vof_sum_last": 0.0,
        "mass_drift_abs": 0.0,
        "mass_drift_rel": 0.0,
        "max_courant": 0.0,
        "p_max_abs": 0.0,
        "u_rms": 0.0,
        "probe_wall_ms": 0.0,
    }
    if backend_builder is None:
        return result

    probe_backend = None
    try:
        probe_backend = backend_builder()
        if probe_backend is None:
            return result
        if not hasattr(probe_backend, "supports_3d_patch_geometry_upload"):
            return result
        if not bool(probe_backend.supports_3d_patch_geometry_upload()):
            return result

        sanitize_kwargs = wb._experimental_3d_geometry_sanitize_options()
        try:
            probe_backend.set_3d_patch_geometry(phi=phi, ax=ax, ay=ay, az=az, **sanitize_kwargs)
        except TypeError:
            probe_backend.set_3d_patch_geometry(phi=phi, ax=ax, ay=ay, az=az)

        probe_stats0 = None
        try:
            if hasattr(probe_backend, "supports_3d_patch_observation") and bool(probe_backend.supports_3d_patch_observation()):
                probe_stats0 = dict(probe_backend.get_3d_patch_stats())
        except Exception:
            probe_stats0 = None
        if probe_stats0 is None:
            return result

        wb._initialize_experimental_3d_patch_state(
            backend=probe_backend,
            patch_stats=probe_stats0,
            swe3d_env_overrides=swe3d_env_overrides,
            bc_n0=bc_n0,
            bc_n1=bc_n1,
            bc_tp=bc_tp,
            bc_vl=bc_vl,
            log_notes=False,
        )

        coupling_mode_val = 0
        try:
            coupling_mode_val = int(wb._experimental_3d_selected_coupling_mode())
        except Exception:
            coupling_mode_val = 0
        coupling_is_active = (
            coupling_mode_enum is not None
            and coupling_mode_val != int(coupling_mode_enum.OFF)
        )
        if coupling_is_active:
            wb._upload_experimental_3d_interface_contract(
                backend=probe_backend,
                patch_stats=probe_stats0,
                bc_n0=bc_n0,
                bc_n1=bc_n1,
                bc_tp=bc_tp,
                coupling_mode=coupling_mode_val,
            )

        vof_sum0 = float(probe_stats0.get("vof_sum", 0.0) or 0.0)
        try:
            vof_seed = np.asarray(probe_backend.get_3d_patch_vof(), dtype=np.float64).ravel()
            if vof_seed.size > 0:
                vof_sum0 = float(np.sum(vof_seed, dtype=np.float64))
        except Exception:
            pass
        result["vof_sum0"] = vof_sum0

        cfl_max = 0.0
        p_abs_max = 0.0
        u_rms_max = 0.0
        vof_sum_last = vof_sum0
        n_steps = max(1, int(probe_steps))
        probe_wall_start = time.perf_counter()

        for _ in range(n_steps):
            try:
                diag = probe_backend.step(-1.0)
            except Exception:
                break
            if isinstance(diag, dict):
                cfl_val = float(diag.get("max_courant", float("nan")))
                if np.isfinite(cfl_val):
                    cfl_max = max(cfl_max, cfl_val)
            try:
                stats_i = dict(probe_backend.get_3d_patch_stats())
            except Exception:
                stats_i = {}
            if stats_i:
                p_i = float(stats_i.get("p_max_abs", float("nan")))
                if np.isfinite(p_i):
                    p_abs_max = max(p_abs_max, p_i)
                u_i = float(stats_i.get("u_rms", float("nan")))
                if np.isfinite(u_i):
                    u_rms_max = max(u_rms_max, u_i)
                vof_i = float(stats_i.get("vof_sum", float("nan")))
                if np.isfinite(vof_i):
                    vof_sum_last = vof_i

        result["probe_wall_ms"] = max(0.0, (time.perf_counter() - probe_wall_start) * 1.0e3)
        result["vof_sum_last"] = vof_sum_last
        result["mass_drift_abs"] = vof_sum_last - vof_sum0
        denom = max(abs(vof_sum0), 1.0e-12)
        result["mass_drift_rel"] = result["mass_drift_abs"] / denom
        result["max_courant"] = cfl_max
        result["p_max_abs"] = p_abs_max
        result["u_rms"] = u_rms_max
        result["ok"] = 1.0
        return result
    except Exception as exc:
        wb._log(f"3D obstacle method probe failed ({method_name}): {exc}")
        return result
    finally:
        if probe_backend is not None and hasattr(probe_backend, "destroy"):
            try:
                probe_backend.destroy()
            except Exception:
                pass


def upload_experimental_3d_obj_geometry(
    *,
    wb: object,
    backend: object,
    patch_stats: Dict[str, object],
    swe3d_env_overrides: Dict[str, str],
    backend_builder: Optional[Callable[[], object]] = None,
    bc_n0: Optional[np.ndarray] = None,
    bc_n1: Optional[np.ndarray] = None,
    bc_tp: Optional[np.ndarray] = None,
    bc_vl: Optional[np.ndarray] = None,
    patch_grid_spec_cls: object,
    load_obj_mesh_fn: object,
    apply_instance_transform_fn: object,
    build_static_geometry_tensors_fn: object,
    write_solid_voxels_obj_fn: object,
    write_fluid_voxels_obj_fn: object = None,
) -> None:
    # Reset last geometry gate diagnostics at each upload attempt so postmortem
    # metadata reflects this run's most recent gate evaluation.
    setattr(wb, "_swe3d_geom_gate_last_config", None)
    setattr(wb, "_swe3d_geom_gate_last_metrics", None)
    setattr(wb, "_swe3d_geom_gate_last_violations", [])

    solids_upload_enabled = bool(
        getattr(wb, "experimental_3d_obj_solids_chk", None)
        and wb.experimental_3d_obj_solids_chk.isChecked()
    )
    export_enabled = bool(
        getattr(wb, "experimental_3d_obj_export_obj_chk", None)
        and wb.experimental_3d_obj_export_obj_chk.isChecked()
    )
    if not solids_upload_enabled and not export_enabled:
        if hasattr(wb, "_log"):
            wb._log(
                "3D solids preprocessing skipped: both '3D sub-grid solids' and '3D export voxel shell OBJ' are disabled."
            )
        return

    if patch_grid_spec_cls is None or load_obj_mesh_fn is None or apply_instance_transform_fn is None or build_static_geometry_tensors_fn is None:
        wb._log("3D sub-grid solids requested, but geometry ingest helpers are unavailable in this runtime.")
        return

    if solids_upload_enabled:
        if backend is None or not hasattr(backend, "supports_3d_patch_geometry_upload"):
            wb._log("3D sub-grid solids requested, but backend geometry upload capability is unavailable.")
            return
        if not bool(backend.supports_3d_patch_geometry_upload()):
            wb._log("3D sub-grid solids requested, but native geometry upload API is not available.")
            return

    spec = wb._build_patch_spec_from_stats(patch_stats, swe3d_env_overrides)
    if spec is None:
        wb._log("3D sub-grid solids skipped: invalid patch descriptor.")
        return
    nx = int(getattr(spec, "nx", 0))
    ny = int(getattr(spec, "ny", 0))
    nz = int(getattr(spec, "nz", 0))

    terrain_surface = None
    terrain_enabled = bool(
        getattr(wb, "experimental_3d_obj_use_terrain_chk", None)
        and wb.experimental_3d_obj_use_terrain_chk.isChecked()
    )
    if terrain_enabled:
        terrain_surface = wb._build_patch_terrain_surface(spec)
        if terrain_surface is not None:
            valid_cols = int(np.count_nonzero(np.isfinite(terrain_surface)))
            wb._log(
                "3D terrain solid preprocessing: "
                f"sampled_columns={valid_cols}/{nx * ny}"
            )
        else:
            wb._log("3D terrain solid preprocessing requested but terrain sampling is unavailable.")

    instances_layer = (
        wb._combo_layer(wb.experimental_3d_obj_layer_combo, "vector")
        if hasattr(wb, "experimental_3d_obj_layer_combo")
        else None
    )

    mesh_items: List[object] = []
    missing_model_paths = 0
    load_errors = 0
    feature_count = 0
    instance_count = 0
    outside_point_assignments = 0

    if instances_layer is not None:
        path_field_req = str(wb.experimental_3d_obj_path_field_edit.text() or "").strip() if hasattr(wb, "experimental_3d_obj_path_field_edit") else ""
        scale_field_req = str(wb.experimental_3d_obj_scale_field_edit.text() or "").strip() if hasattr(wb, "experimental_3d_obj_scale_field_edit") else ""
        yaw_field_req = str(wb.experimental_3d_obj_yaw_field_edit.text() or "").strip() if hasattr(wb, "experimental_3d_obj_yaw_field_edit") else ""
        zoff_field_req = str(wb.experimental_3d_obj_z_offset_field_edit.text() or "").strip() if hasattr(wb, "experimental_3d_obj_z_offset_field_edit") else ""
        instance_id_field_req = str(wb.experimental_3d_obj_instance_id_field_edit.text() or "").strip() if hasattr(wb, "experimental_3d_obj_instance_id_field_edit") else ""
        fallback_path = str(wb.experimental_3d_obj_default_path_edit.text() or "").strip() if hasattr(wb, "experimental_3d_obj_default_path_edit") else ""

        inside_points_layer = (
            wb._combo_layer(wb.experimental_3d_obj_inside_points_layer_combo, "vector")
            if hasattr(wb, "experimental_3d_obj_inside_points_layer_combo")
            else None
        )
        inside_id_field_req = str(wb.experimental_3d_obj_inside_id_field_edit.text() or "").strip() if hasattr(wb, "experimental_3d_obj_inside_id_field_edit") else ""
        inside_z_field_req = str(wb.experimental_3d_obj_inside_z_field_edit.text() or "").strip() if hasattr(wb, "experimental_3d_obj_inside_z_field_edit") else ""

        path_field = wb._resolve_layer_field_name(instances_layer, path_field_req)
        scale_field = wb._resolve_layer_field_name(instances_layer, scale_field_req)
        yaw_field = wb._resolve_layer_field_name(instances_layer, yaw_field_req)
        zoff_field = wb._resolve_layer_field_name(instances_layer, zoff_field_req)
        instance_id_field = wb._resolve_layer_field_name(instances_layer, instance_id_field_req)

        inside_id_field = ""
        inside_z_field = ""
        inside_points_by_id: Dict[str, List[Tuple[float, float, float]]] = {}
        inside_points_all: List[Tuple[float, float, float]] = []
        if inside_points_layer is not None:
            inside_id_field = wb._resolve_layer_field_name(inside_points_layer, inside_id_field_req)
            inside_z_field = wb._resolve_layer_field_name(inside_points_layer, inside_z_field_req)
            for ip_ft in inside_points_layer.getFeatures():
                ip_geom = ip_ft.geometry()
                if ip_geom is None or ip_geom.isEmpty():
                    continue
                try:
                    ip_pt = ip_geom.asPoint()
                    ip_x = float(ip_pt.x())
                    ip_y = float(ip_pt.y())
                except Exception:
                    continue

                ip_z = float("nan")
                try:
                    if hasattr(ip_pt, "is3D") and ip_pt.is3D():
                        ip_z = float(ip_pt.z())
                    elif hasattr(ip_pt, "z"):
                        zv = float(ip_pt.z())
                        if np.isfinite(zv):
                            ip_z = zv
                except Exception:
                    ip_z = float("nan")
                if (not np.isfinite(ip_z)) and inside_z_field:
                    ip_z = wb._parse_feature_float(ip_ft, inside_z_field, float("nan"))

                pt_tuple = (ip_x, ip_y, ip_z)
                inside_points_all.append(pt_tuple)

                ip_id = ""
                if inside_id_field:
                    try:
                        ip_id = str(ip_ft[inside_id_field] or "").strip()
                    except Exception:
                        ip_id = ""
                if ip_id:
                    inside_points_by_id.setdefault(ip_id, []).append(pt_tuple)

            wb._log(
                "3D outside-point scan: "
                f"layer={inside_points_layer.name()}, points={len(inside_points_all)}, "
                f"id_indexed={sum(len(v) for v in inside_points_by_id.values())}"
            )

        if path_field_req and not path_field:
            wb._log(f"3D OBJ path field '{path_field_req}' not found in layer {instances_layer.name()}; using fallback path if provided.")
        if instance_id_field_req and not instance_id_field:
            wb._log(
                f"3D OBJ instance id field '{instance_id_field_req}' not found in layer {instances_layer.name()}; "
                "outside points will be matched by nearest point."
            )

        if not fallback_path:
            fallback_path = wb._infer_obj_path_from_layer_3d_renderer(instances_layer)

        mesh_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for ft in instances_layer.getFeatures():
            feature_count += 1
            geom = ft.geometry()
            if geom is None or geom.isEmpty():
                continue
            try:
                pt = geom.asPoint()
                x = float(pt.x())
                y = float(pt.y())
            except Exception:
                continue

            z_geom = float("nan")
            try:
                if hasattr(pt, "is3D") and pt.is3D():
                    z_geom = float(pt.z())
                elif hasattr(pt, "z"):
                    z_test = float(pt.z())
                    if np.isfinite(z_test):
                        z_geom = z_test
            except Exception:
                z_geom = float("nan")

            model_path = ""
            if path_field:
                try:
                    model_path = str(ft[path_field] or "").strip()
                except Exception:
                    model_path = ""
            if not model_path:
                model_path = fallback_path
            if not model_path:
                missing_model_paths += 1
                continue

            model_abs = wb._resolve_obj_model_path(model_path)
            if not model_abs or not os.path.exists(model_abs):
                missing_model_paths += 1
                continue

            scale = (1.0, 1.0, 1.0)
            if scale_field:
                try:
                    scale = wb._parse_obj_scale_value(ft[scale_field])
                except Exception:
                    scale = (1.0, 1.0, 1.0)

            yaw_deg = wb._parse_feature_float(ft, yaw_field, 0.0)
            z_offset = wb._parse_feature_float(ft, zoff_field, 0.0)

            z_anchor = z_geom
            if not np.isfinite(z_anchor) and terrain_surface is not None:
                ix = int(np.floor((x - spec.origin_x) / max(spec.dx, 1.0e-12)))
                iy = int(np.floor((y - spec.origin_y) / max(spec.dy, 1.0e-12)))
                if 0 <= ix < spec.nx and 0 <= iy < spec.ny:
                    zv = float(terrain_surface[iy, ix])
                    if np.isfinite(zv):
                        z_anchor = zv
            if not np.isfinite(z_anchor):
                z_anchor = float(spec.origin_z)

            outside_point = None
            if inside_points_all:
                instance_id = ""
                if instance_id_field:
                    try:
                        instance_id = str(ft[instance_id_field] or "").strip()
                    except Exception:
                        instance_id = ""

                candidates = inside_points_by_id.get(instance_id, []) if instance_id else []
                if not candidates:
                    candidates = inside_points_all

                if candidates:
                    cand_arr = np.asarray(candidates, dtype=np.float64)
                    d2 = (cand_arr[:, 0] - x) ** 2 + (cand_arr[:, 1] - y) ** 2
                    best = cand_arr[int(np.argmin(d2))]
                    ip_x = float(best[0])
                    ip_y = float(best[1])
                    ip_z = float(best[2])
                    if not np.isfinite(ip_z):
                        if terrain_surface is not None:
                            ix_ip = int(np.floor((ip_x - spec.origin_x) / max(spec.dx, 1.0e-12)))
                            iy_ip = int(np.floor((ip_y - spec.origin_y) / max(spec.dy, 1.0e-12)))
                            if 0 <= ix_ip < spec.nx and 0 <= iy_ip < spec.ny:
                                zv_ip = float(terrain_surface[iy_ip, ix_ip])
                                if np.isfinite(zv_ip):
                                    ip_z = zv_ip
                    if not np.isfinite(ip_z):
                        ip_z = float(z_anchor)
                    outside_point = (ip_x, ip_y, ip_z)

            if model_abs not in mesh_cache:
                try:
                    mesh_cache[model_abs] = load_obj_mesh_fn(model_abs)
                except Exception as exc:
                    load_errors += 1
                    wb._log(f"3D OBJ parse failed ({model_abs}): {exc}")
                    continue

            base_vertices, base_faces = mesh_cache[model_abs]
            try:
                world_vertices = apply_instance_transform_fn(
                    base_vertices,
                    (x, y, z_anchor + z_offset),
                    yaw_deg=yaw_deg,
                    scale_xyz=scale,
                )
            except Exception as exc:
                load_errors += 1
                wb._log(f"3D OBJ transform failed ({model_abs}): {exc}")
                continue

            item_payload = {
                "vertices": world_vertices,
                "faces": base_faces,
            }
            if outside_point is not None:
                item_payload["outside_point"] = outside_point
                outside_point_assignments += 1
            mesh_items.append(item_payload)
            instance_count += 1

        wb._log(
            "3D OBJ instance scan: "
            f"features={feature_count}, instances={instance_count}, "
            f"missing_paths={missing_model_paths}, parse_errors={load_errors}, "
            f"outside_points_assigned={outside_point_assignments}"
        )

    if terrain_surface is None and not mesh_items:
        wb._log("3D sub-grid solids: no terrain or OBJ instances were resolved.")
        if not export_enabled:
            wb._log("3D sub-grid solids enabled but no terrain/OBJ content was found; skipping upload.")
            return
        if solids_upload_enabled:
            wb._log(
                "3D solids upload will be skipped for this run, but fluid OBJ export will continue "
                "using the open-domain tensor field."
            )
            solids_upload_enabled = False

    expected = int(patch_stats.get("n_cells", nx * ny * nz) or (nx * ny * nz))
    obstacle_method = wb._experimental_3d_selected_obstacle_method()
    alt_method = "favor1981_porosity" if obstacle_method == "fractional_cutcell" else "fractional_cutcell"

    ab_compare_enabled = bool(
        getattr(wb, "experimental_3d_obj_ab_compare_chk", None)
        and wb.experimental_3d_obj_ab_compare_chk.isChecked()
    )
    probe_steps = 8
    if hasattr(wb, "experimental_3d_obj_ab_probe_steps_spin"):
        try:
            probe_steps = max(1, int(wb.experimental_3d_obj_ab_probe_steps_spin.value()))
        except Exception:
            probe_steps = 8

    methods_to_build = [obstacle_method]
    if ab_compare_enabled and alt_method not in methods_to_build:
        methods_to_build.append(alt_method)

    tensors_by_method: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]] = {}
    for method_name in methods_to_build:
        try:
            phi_m, ax_m, ay_m, az_m, diag_m = build_static_geometry_tensors_fn(
                spec=spec,
                mesh_items=mesh_items,
                terrain_elevation=terrain_surface,
                obstacle_method=method_name,
            )
        except Exception as exc:
            wb._log(f"3D sub-grid tensor build failed ({method_name}): {exc}")
            return

        if int(phi_m.size) != expected:
            wb._log(
                "3D sub-grid tensor size mismatch; upload aborted: "
                f"method={method_name}, expected={expected}, got={int(phi_m.size)}"
            )
            return
        tensors_by_method[method_name] = (phi_m, ax_m, ay_m, az_m, dict(diag_m))

    phi, ax, ay, az, diag = tensors_by_method[obstacle_method]
    if solids_upload_enabled:
        sanitize_kwargs = wb._experimental_3d_geometry_sanitize_options()
        try:
            backend.set_3d_patch_geometry(phi=phi, ax=ax, ay=ay, az=az, **sanitize_kwargs)
        except TypeError:
            backend.set_3d_patch_geometry(phi=phi, ax=ax, ay=ay, az=az)
        except Exception as exc:
            wb._log(f"3D sub-grid tensor upload failed: {exc}")
            return

        if bool(sanitize_kwargs.get("sanitize", False)):
            wb._log(
                "3D geometry sanitization: enabled "
                f"(phi_snap_min={float(sanitize_kwargs.get('phi_snap_min', 0.0)):.6g}, "
                f"area_snap_min={float(sanitize_kwargs.get('area_snap_min', 0.0)):.6g})"
            )

        wb._log(
            "3D sub-grid tensors uploaded: "
            f"method={obstacle_method} "
            f"solid_cells={int(diag.get('solid_cells', 0.0))}/{expected} "
            f"solid_fraction={float(diag.get('solid_fraction', 0.0)):.3f} "
            f"terrain_cells={int(diag.get('terrain_solid_cells', 0.0))} "
            f"mesh_cells={int(diag.get('mesh_solid_cells', 0.0))} "
            f"mesh_instances_used={int(diag.get('mesh_instances_used', 0.0))} "
            f"seed_requests={int(diag.get('mesh_seed_instances_requested', 0.0))} "
            f"seed_used={int(diag.get('mesh_seed_instances_used', 0.0))} "
            f"seed_leak_fallbacks={int(diag.get('mesh_seed_leak_fallbacks', 0.0))}"
        )
    else:
        wb._log(
            "3D sub-grid solids upload disabled; building tensors for fluid OBJ export only. "
            f"method={obstacle_method}, solid_fraction={float(diag.get('solid_fraction', 0.0)):.3f}"
        )
    solid_fraction = float(diag.get('solid_fraction', 0.0) or 0.0)
    if solid_fraction >= 0.995:
        wb._log(
            "3D sub-grid solids warning: upload is effectively fully solid; "
            "if this is unexpected, widen the ROI/z span or verify terrain/OBJ inputs."
        )
    elif solid_fraction >= 0.90:
        wb._log(
            "3D sub-grid solids warning: upload is mostly solid; "
            "runs can become brittle if the ROI leaves too little fluid space."
        )
    if int(diag.get('mesh_seed_leak_fallbacks', 0.0) or 0) > 0:
        wb._log(
            "3D sub-grid solids note: one or more OBJ instances fell back to leak-seed classification; "
            "this can make cut-cell reconstruction sensitive to geometry gaps."
        )

    geom_gate_strict = _env_flag("BACKWATER_SWE3D_GEOM_STRICT", False)
    geom_gate_max_solid = max(0.0, min(1.0, _env_float("BACKWATER_SWE3D_GEOM_MAX_SOLID_FRACTION", 0.98)))
    geom_gate_max_seed_leak = max(0, _env_int("BACKWATER_SWE3D_GEOM_MAX_SEED_LEAK_FALLBACKS", 0))
    seed_leak_count = int(diag.get("mesh_seed_leak_fallbacks", 0.0) or 0)
    gate_config = {
        "strict": bool(geom_gate_strict),
        "max_solid_fraction": float(geom_gate_max_solid),
        "max_seed_leak_fallbacks": int(geom_gate_max_seed_leak),
    }
    gate_metrics = {
        "solid_fraction": float(solid_fraction),
        "mesh_seed_leak_fallbacks": int(seed_leak_count),
    }
    setattr(wb, "_swe3d_geom_gate_last_config", gate_config)
    setattr(wb, "_swe3d_geom_gate_last_metrics", gate_metrics)

    gate_violations: List[str] = []
    if solid_fraction > geom_gate_max_solid:
        gate_violations.append(
            f"solid_fraction={solid_fraction:.6f} exceeds BACKWATER_SWE3D_GEOM_MAX_SOLID_FRACTION={geom_gate_max_solid:.6f}"
        )
    if seed_leak_count > geom_gate_max_seed_leak:
        gate_violations.append(
            f"mesh_seed_leak_fallbacks={seed_leak_count} exceeds BACKWATER_SWE3D_GEOM_MAX_SEED_LEAK_FALLBACKS={geom_gate_max_seed_leak}"
        )
    if gate_violations:
        wb._log("3D geometry quality gate violation: " + " | ".join(gate_violations))
        setattr(wb, "_swe3d_geom_gate_last_violations", list(gate_violations))
        if geom_gate_strict and solids_upload_enabled:
            raise RuntimeError(
                "3D geometry quality gate failed in strict mode. "
                "Relax geometry, expand ROI/fluid space, or adjust BACKWATER_SWE3D_GEOM_* thresholds."
            )
    else:
        setattr(wb, "_swe3d_geom_gate_last_violations", [])


    if export_enabled:
        # Use fluid domain mesh export instead of solids.
        fluid_export_fn = write_fluid_voxels_obj_fn
        if fluid_export_fn is None:
            try:
                from swe3d_geometry_ingest import write_fluid_voxels_obj as fluid_export_fn
            except ImportError:
                try:
                    from .swe3d_geometry_ingest import write_fluid_voxels_obj as fluid_export_fn
                except ImportError:
                    fluid_export_fn = None
        if fluid_export_fn is None:
            wb._log("3D fluid OBJ export requested, but export helper is unavailable.")
        else:
            export_path = wb._resolve_experimental_3d_obj_export_path(obstacle_method)
            wb._log(f"3D fluid OBJ export target path: {export_path}")
            try:
                export_threshold = 0.001  # Fluid: phi >= 0.001
                export_summary = fluid_export_fn(
                    spec=spec,
                    phi=phi,
                    file_path=export_path,
                    fluid_threshold=export_threshold,
                )
                wb._log(
                    "3D fluid OBJ export complete: "
                    f"threshold={export_threshold:.3f}, "
                    f"path={export_path}, fluid_cells={int(export_summary.get('fluid_cells', 0.0))}, "
                    f"vertices={int(export_summary.get('vertices', 0.0))}, faces={int(export_summary.get('faces', 0.0))}"
                )
            except Exception as exc:
                wb._log(f"3D fluid OBJ export failed ({export_path}): {exc}")

    if ab_compare_enabled and alt_method in tensors_by_method:
        phi_a, ax_a, ay_a, az_a, _diag_a = tensors_by_method[obstacle_method]
        phi_b, ax_b, ay_b, az_b, _diag_b = tensors_by_method[alt_method]
        wb._log(
            "3D obstacle A/B tensor delta: "
            f"{obstacle_method} vs {alt_method}, "
            f"phi_mean_abs={float(np.mean(np.abs(phi_a - phi_b))):.6e}, "
            f"ax_mean_abs={float(np.mean(np.abs(ax_a - ax_b))):.6e}, "
            f"ay_mean_abs={float(np.mean(np.abs(ay_a - ay_b))):.6e}, "
            f"az_mean_abs={float(np.mean(np.abs(az_a - az_b))):.6e}"
        )

        if backend_builder is None:
            wb._log("3D obstacle A/B runtime probe skipped: backend builder unavailable.")
        elif bc_n0 is None or bc_n1 is None or bc_tp is None or bc_vl is None:
            wb._log("3D obstacle A/B runtime probe skipped: boundary arrays unavailable.")
        else:
            probe_primary = wb._run_experimental_3d_obj_method_probe(
                backend_builder=backend_builder,
                method_name=obstacle_method,
                phi=phi_a,
                ax=ax_a,
                ay=ay_a,
                az=az_a,
                swe3d_env_overrides=swe3d_env_overrides,
                bc_n0=np.asarray(bc_n0, dtype=np.int32),
                bc_n1=np.asarray(bc_n1, dtype=np.int32),
                bc_tp=np.asarray(bc_tp, dtype=np.int32),
                bc_vl=np.asarray(bc_vl, dtype=np.float64),
                probe_steps=probe_steps,
            )
            probe_alt = wb._run_experimental_3d_obj_method_probe(
                backend_builder=backend_builder,
                method_name=alt_method,
                phi=phi_b,
                ax=ax_b,
                ay=ay_b,
                az=az_b,
                swe3d_env_overrides=swe3d_env_overrides,
                bc_n0=np.asarray(bc_n0, dtype=np.int32),
                bc_n1=np.asarray(bc_n1, dtype=np.int32),
                bc_tp=np.asarray(bc_tp, dtype=np.int32),
                bc_vl=np.asarray(bc_vl, dtype=np.float64),
                probe_steps=probe_steps,
            )
            if probe_primary.get("ok", 0.0) >= 0.5 and probe_alt.get("ok", 0.0) >= 0.5:
                wb._log(
                    "3D obstacle A/B runtime probe: "
                    f"steps={probe_steps}, "
                    f"{obstacle_method}[drift_rel={float(probe_primary.get('mass_drift_rel', 0.0)):.3e}, "
                    f"cfl_max={float(probe_primary.get('max_courant', 0.0)):.3e}, "
                    f"p_max_abs={float(probe_primary.get('p_max_abs', 0.0)):.3e}, "
                    f"u_rms={float(probe_primary.get('u_rms', 0.0)):.3e}] "
                    f"vs {alt_method}[drift_rel={float(probe_alt.get('mass_drift_rel', 0.0)):.3e}, "
                    f"cfl_max={float(probe_alt.get('max_courant', 0.0)):.3e}, "
                    f"p_max_abs={float(probe_alt.get('p_max_abs', 0.0)):.3e}, "
                    f"u_rms={float(probe_alt.get('u_rms', 0.0)):.3e}]"
                )
                wb._log(
                    "3D obstacle A/B runtime deltas: "
                    f"drift_rel_delta={float(probe_primary.get('mass_drift_rel', 0.0) - probe_alt.get('mass_drift_rel', 0.0)):.3e}, "
                    f"cfl_delta={float(probe_primary.get('max_courant', 0.0) - probe_alt.get('max_courant', 0.0)):.3e}, "
                    f"p_max_abs_delta={float(probe_primary.get('p_max_abs', 0.0) - probe_alt.get('p_max_abs', 0.0)):.3e}, "
                    f"u_rms_delta={float(probe_primary.get('u_rms', 0.0) - probe_alt.get('u_rms', 0.0)):.3e}"
                )
            else:
                wb._log("3D obstacle A/B runtime probe incomplete; one or both method probes failed.")


def execute_run_timestep_loop(
    *,
    wb: object,
    backend: object,
    runtime_step_executor: object,
    runtime_reporter: object,
    run_duration_s: float,
    t_accum: float,
    i: int,
    last_diag: Optional[Dict[str, object]],
    last_valid_cmax: float,
    last_valid_wse_res: float,
    dt_cfg: float,
    dt_request: float,
    stage_coupled_imex_enabled: bool,
    coupling_controller: object,
    dynamic_bc: bool,
    native_bc_forcing: bool,
    bc_n0: np.ndarray,
    bc_n1: np.ndarray,
    bc_tp: np.ndarray,
    bc_vl: np.ndarray,
    side_hydrographs: object,
    edge_hydrographs: object,
    rain_source_for_window_callback: object,
    cell_source_model_at_time_callback: object,
    accumulate_source_volume_model_callback: object,
    native_source_injection_mode: bool,
    accumulate_boundary_flux_volume_model_callback: object,
    sample_map: object,
    cell_min_z: object,
    experimental_3d_runtime: bool,
    timing_totals_ms: Dict[str, float],
    timing_samples: int,
    next_snap_t: float,
    next_line_snap_t: float,
    next_coupling_snap_t: float,
    output_interval_s: float,
    line_output_interval_s: float,
    process_events_interval_s: float,
    last_process_events_wall: float,
    process_events_callback: object,
    get_3d_patch_stats_callback: object,
    get_3d_patch_vof_callback: object,
    get_3d_patch_velocity_callback: Optional[Callable[[], Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]]] = None,
    physics_diag_enabled: bool = False,
    front_flux_damping_value: float = 1.0,
    zmax_bc_mode: Optional[int] = None,
    apply_3d_patch_face_bc_callback: Optional[Callable[..., None]] = None,
) -> Dict[str, object]:
    while float(t_accum) < float(run_duration_s):
        if bool(getattr(wb, "_cancel_requested", False)):
            break

        step_wall_t0 = time.perf_counter()
        step_ms = 0.0
        coupling_ms = 0.0
        source_ms = 0.0
        state_ms = 0.0
        bc_ms = 0.0
        ui_ms = 0.0

        step_result = runtime_step_executor.execute_step(
            backend=backend,
            t_accum=t_accum,
            last_diag=last_diag,
            dt_cfg=dt_cfg,
            dt_request=dt_request,
            stage_coupled_imex_enabled=stage_coupled_imex_enabled,
            coupling_controller=coupling_controller,
            dynamic_bc=dynamic_bc,
            native_bc_forcing=native_bc_forcing,
            bc_n0=bc_n0,
            bc_n1=bc_n1,
            bc_tp=bc_tp,
            bc_vl=bc_vl,
            side_hydrographs=side_hydrographs,
            edge_hydrographs=edge_hydrographs,
            apply_timeseries_bc_values_callback=wb._apply_timeseries_bc_values,
            distribute_total_flow_to_unit_q_callback=wb._distribute_total_flow_to_unit_q,
            rain_source_for_window_callback=rain_source_for_window_callback,
            cell_source_model_at_time_callback=cell_source_model_at_time_callback,
            accumulate_source_volume_model_callback=accumulate_source_volume_model_callback,
            apply_external_sources_callback=wb._apply_external_sources,
            native_source_injection_mode=native_source_injection_mode,
            apply_3d_patch_face_bc_callback=apply_3d_patch_face_bc_callback,
        )
        last_diag = step_result["last_diag"]
        dt_used = float(step_result["dt_used"])
        rain_src = step_result["rain_src"]
        step_ms += float(step_result["step_ms"])
        coupling_ms += float(step_result["coupling_ms"])
        source_ms += float(step_result["source_ms"])
        state_ms += float(step_result["state_ms"])
        bc_ms += float(step_result["bc_ms"])

        if bc_n0.size > 0:
            if dynamic_bc:
                bc_tp_flux, bc_vl_flux = wb._apply_timeseries_bc_values(
                    bc_n0,
                    bc_n1,
                    bc_tp,
                    bc_vl,
                    side_hydrographs,
                    t_accum,
                    edge_hydrographs,
                )
                bc_vl_flux = wb._distribute_total_flow_to_unit_q(
                    bc_n0,
                    bc_n1,
                    bc_tp_flux,
                    bc_vl_flux,
                    bc_tp,
                    side_hydrographs,
                    edge_hydrographs,
                )
            else:
                bc_tp_flux = bc_tp
                # Static flow BC values are stored as total-Q per group; convert
                # to unit-q before boundary flux accounting to avoid edge-length
                # re-scaling of already-totalized values.
                bc_vl_flux = wb._distribute_total_flow_to_unit_q(
                    bc_n0,
                    bc_n1,
                    bc_tp_flux,
                    bc_vl,
                    bc_tp,
                    side_hydrographs,
                    edge_hydrographs,
                )
            accumulate_boundary_flux_volume_model_callback(dt_used, bc_tp_flux, bc_vl_flux)

        report_result = runtime_reporter.process_step(
            backend=backend,
            t_accum=t_accum,
            dt_used=dt_used,
            last_diag=last_diag,
            last_valid_cmax=last_valid_cmax,
            last_valid_wse_res=last_valid_wse_res,
            sample_map=sample_map,
            cell_min_z=cell_min_z,
            coupling_controller=coupling_controller,
            experimental_3d_runtime=experimental_3d_runtime,
            rain_src=rain_src,
            state_ms=state_ms,
            ui_ms=ui_ms,
            step_wall_t0=step_wall_t0,
            step_ms=step_ms,
            coupling_ms=coupling_ms,
            source_ms=source_ms,
            bc_ms=bc_ms,
            timing_totals_ms=timing_totals_ms,
            timing_samples=timing_samples,
            i=i,
            run_duration_s=run_duration_s,
            next_snap_t=next_snap_t,
            next_line_snap_t=next_line_snap_t,
            next_coupling_snap_t=next_coupling_snap_t,
            output_interval_s=output_interval_s,
            line_output_interval_s=line_output_interval_s,
            process_events_interval_s=process_events_interval_s,
            last_process_events_wall=last_process_events_wall,
            h_min=float(wb.h_min_spin.value()),
            length_unit_name=wb._length_unit_name,
            snapshot_timesteps=wb._snapshot_timesteps,
            line_snapshot_rows=wb._line_snapshot_rows,
            line_snapshot_profile_rows=wb._line_snapshot_profile_rows,
            coupling_snapshot_rows=wb._coupling_snapshot_rows,
            get_3d_patch_stats_callback=get_3d_patch_stats_callback,
            get_3d_patch_vof_callback=get_3d_patch_vof_callback,
            get_3d_patch_velocity_callback=get_3d_patch_velocity_callback,
            physics_diag_enabled=physics_diag_enabled,
            front_flux_damping_value=front_flux_damping_value,
            zmax_bc_mode=zmax_bc_mode,
            append_3d_patch_snapshot_callback=wb._append_3d_patch_snapshot,
            sample_line_metrics_callback=wb._sample_line_metrics,
            sample_coupling_object_metrics_callback=wb._sample_coupling_object_metrics,
            process_events_callback=process_events_callback,
            set_progress_callback=wb.progress_bar.setValue,
            log_callback=wb._log,
        )
        t_accum = float(report_result["t_accum"])
        last_valid_cmax = float(report_result["last_valid_cmax"])
        last_valid_wse_res = float(report_result["last_valid_wse_res"])
        next_snap_t = float(report_result["next_snap_t"])
        next_line_snap_t = float(report_result["next_line_snap_t"])
        next_coupling_snap_t = float(report_result["next_coupling_snap_t"])
        last_process_events_wall = float(report_result["last_process_events_wall"])
        timing_samples = int(report_result["timing_samples"])
        i = int(report_result["i"])

    return {
        "t_accum": float(t_accum),
        "i": int(i),
        "last_diag": last_diag,
        "last_valid_cmax": float(last_valid_cmax),
        "last_valid_wse_res": float(last_valid_wse_res),
        "next_snap_t": float(next_snap_t),
        "next_line_snap_t": float(next_line_snap_t),
        "next_coupling_snap_t": float(next_coupling_snap_t),
        "last_process_events_wall": float(last_process_events_wall),
        "timing_samples": int(timing_samples),
    }
