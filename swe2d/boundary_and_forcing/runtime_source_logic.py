from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np


def internal_flow_source_cms_at_time(
    forcing: Optional[Dict[str, object]],
    t_sec: float,
    interp_hydrograph: Callable[[Tuple[np.ndarray, np.ndarray], float], float],
) -> Optional[np.ndarray]:
    if forcing is None:
        return None
    base_q = forcing.get("base_q_cms")
    if base_q is None:
        return None

    cell_q = np.asarray(base_q, dtype=np.float64).copy()
    dynamic_terms = forcing.get("dynamic_terms", [])
    for idx_arr, wt_arr, hg in dynamic_terms:
        q_total = interp_hydrograph(hg, t_sec)
        cell_q[np.asarray(idx_arr, dtype=np.int32)] += q_total * np.asarray(wt_arr, dtype=np.float64)
    return cell_q


def apply_external_sources(
    backend,
    dt_step: float,
    rain_rate_model,
    cell_source_model: Optional[np.ndarray],
    coupled_source_rate: Optional[np.ndarray] = None,
    prefer_native_injection: bool = False,
    mesh_cell_areas: Optional[np.ndarray] = None,
    max_source_rate: float = 0.0,
    h_min: float = 1.0e-4,
    max_rel_depth_increase: float = 0.0,
    max_source_depth_step: float = 0.0,
    shallow_damping_depth: float = 0.0,
    momentum_cap_min_speed: float = 50.0,
    momentum_cap_celerity_mult: float = 20.0,
) -> None:
    if dt_step <= 0.0:
        if prefer_native_injection and hasattr(backend, "set_external_sources_native"):
            try:
                backend.set_external_sources_native(None)
            except Exception:
                pass
        return

    no_external_sources = (
        np.all(np.asarray(rain_rate_model, dtype=np.float64) <= 0.0)
        and cell_source_model is None
        and coupled_source_rate is None
    )
    if no_external_sources:
        # When prefer_native_injection is True, the CUDA coupling kernel has
        # already written structure/drainage source rates directly to the
        # device-resident d_external_source_mps buffer.  Do NOT call
        # set_external_sources_native(None) here — that would memset the
        # buffer to zero, erasing the on-device sources.  Just return.
        if prefer_native_injection and hasattr(backend, "set_external_sources_native"):
            pass  # GPU buffer already populated by coupling kernel
        return

    n_cells_raw = getattr(backend, "n_cells", 0)
    n_cells = int(n_cells_raw() if callable(n_cells_raw) else n_cells_raw)
    rain_arr = np.asarray(rain_rate_model, dtype=np.float64)
    if rain_arr.ndim == 0:
        src = np.full((n_cells,), float(rain_arr), dtype=np.float64)
    else:
        src = np.zeros((n_cells,), dtype=np.float64)
        src[: min(src.shape[0], rain_arr.shape[0])] = rain_arr[: min(src.shape[0], rain_arr.shape[0])]

    if cell_source_model is not None:
        if mesh_cell_areas is None:
            safe_area = np.ones_like(src)
        else:
            safe_area = np.maximum(np.asarray(mesh_cell_areas, dtype=np.float64), 1.0e-8)
        src += (cell_source_model / safe_area)

    if coupled_source_rate is not None:
        csr = np.asarray(coupled_source_rate, dtype=np.float64)
        src[: min(src.shape[0], csr.shape[0])] += csr[: min(src.shape[0], csr.shape[0])]

    src = np.where(np.isfinite(src), src, 0.0)

    if max_source_rate > 0.0:
        src = np.where(src > max_source_rate, max_source_rate, src)

    if prefer_native_injection and hasattr(backend, "set_external_sources_native"):
        try:
            backend.set_external_sources_native(src)
            return
        except Exception:
            pass

    h, hu, hv = backend.get_state()
    h_prev = np.asarray(h, dtype=np.float64)

    dh = dt_step * src

    if max_rel_depth_increase > 0.0:
        dh_pos_cap = np.maximum(h_prev, h_min) * max_rel_depth_increase
        dh = np.where(dh > dh_pos_cap, dh_pos_cap, dh)

    if max_source_depth_step > 0.0:
        dh = np.where(dh > max_source_depth_step, max_source_depth_step, dh)

    h = h_prev + dh
    h = np.where(np.isfinite(h), h, 0.0)
    h = np.maximum(h, 0.0)

    dry = h < h_min
    hu = np.where(dry, 0.0, hu)
    hv = np.where(dry, 0.0, hv)

    newly_wet = (h_prev < h_min) & (~dry)
    hu = np.where(newly_wet, 0.0, hu)
    hv = np.where(newly_wet, 0.0, hv)

    if shallow_damping_depth > h_min:
        damp = np.clip(h / shallow_damping_depth, 0.0, 1.0)
        hu = hu * damp
        hv = hv * damp

    gravity = 9.81
    abs_u = np.abs(hu) / np.maximum(h, 1.0e-12)
    abs_v = np.abs(hv) / np.maximum(h, 1.0e-12)
    abs_speed = np.sqrt(abs_u**2 + abs_v**2)
    wave_speed = np.sqrt(gravity * np.maximum(h, 1.0e-12))
    speed_cap = np.maximum(momentum_cap_min_speed, momentum_cap_celerity_mult * wave_speed)
    clipped = abs_speed > speed_cap
    if np.any(clipped):
        scale = np.where(clipped, speed_cap / np.maximum(abs_speed, 1.0e-12), 1.0)
        hu = hu * scale
        hv = hv * scale

    backend.set_state(h, hu, hv)
