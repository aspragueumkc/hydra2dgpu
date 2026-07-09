from __future__ import annotations

"""Runtime internal-flow source-term computation at a given simulation time."""

from typing import Callable, Dict, Optional, Tuple

import numpy as np


def permute_internal_flow_forcing(forcing: Optional[Dict[str, object]], cell_perm: np.ndarray) -> Optional[Dict[str, object]]:
    """Permute internal-flow forcing from original cell order to solver (RCMK) order.

    The C++ mesh builder reorders cells.  The internal-flow forcing is built
    from the original-order mesh centroids, so its per-cell arrays must be
    permuted before they are consumed by the solver backend.

    Parameters
    ----------
    forcing : dict or None
        Forcing object returned by ``build_internal_flow_forcing_*`` adapters.
    cell_perm : np.ndarray
        Solver permutation where ``cell_perm[c_new] = c_old``.

    Returns
    -------
    dict or None
        New forcing object whose ``base_q`` and dynamic-term indices are in
        solver order.  Returns *forcing* unchanged if it is None or the
        permutation is empty.
    """
    if forcing is None or cell_perm is None or cell_perm.size == 0:
        return forcing
    n_cells = int(cell_perm.size)
    inv_perm = np.zeros(n_cells, dtype=np.int32)
    inv_perm[cell_perm] = np.arange(n_cells, dtype=np.int32)

    out = dict(forcing)
    base_q = out.get("base_q")
    if base_q is not None:
        base_q = np.asarray(base_q, dtype=np.float64)
        if base_q.size == n_cells:
            out["base_q"] = base_q[cell_perm]

    dynamic_terms = out.get("dynamic_terms")
    if dynamic_terms is not None:
        out["dynamic_terms"] = [
            (inv_perm[np.asarray(idx_arr, dtype=np.int32)], wt_arr, hg)
            for idx_arr, wt_arr, hg in dynamic_terms
        ]
    return out


def internal_flow_source_cms_at_time(
    forcing: Optional[Dict[str, object]],
    t_sec: float,
    interp_hydrograph: Callable[[Tuple[np.ndarray, np.ndarray], float], float],
) -> Optional[np.ndarray]:
    """Return per-cell internal flow source rates [m³/s] at simulation time *t_sec*."""
    if forcing is None:
        return None
    base_q = forcing.get("base_q")
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
    cell_source_model: Optional[np.ndarray] = None,
    coupled_source_rate: Optional[np.ndarray] = None,
    mesh_cell_areas: Optional[np.ndarray] = None,
    max_source_rate: float = 0.0,
    h_min: float = 1.0e-4,
    max_rel_depth_increase: float = 0.0,
    max_source_depth_step: float = 0.0,
    shallow_damping_depth: float = 0.0,
    momentum_cap_min_speed: float = 50.0,
    momentum_cap_celerity_mult: float = 20.0,
) -> None:
    """Apply external sources via GPU-native injection."""
    if dt_step <= 0.0:
        backend.set_external_sources_native(None)
        return

    no_external_sources = (
        np.all(np.asarray(rain_rate_model, dtype=np.float64) <= 0.0)
        and cell_source_model is None
        and coupled_source_rate is None
    )
    if no_external_sources:
        return

    if not hasattr(backend, "set_external_sources_native"):
        raise RuntimeError(
            "GPU-native source injection required but "
            "set_external_sources_native is not available on the backend."
        )

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

    backend.set_external_sources_native(src)
