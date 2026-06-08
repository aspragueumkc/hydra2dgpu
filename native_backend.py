"""Optional loader for the native backwater acceleration module."""

from __future__ import annotations

import importlib
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_NATIVE = None
_NATIVE_IMPORT_ERROR: Optional[Exception] = None


def _env_truthy(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in ("1", "true", "yes", "on")


def is_native_enabled() -> bool:
    """Whether native acceleration is explicitly enabled by environment."""
    return _env_truthy("BACKWATER_USE_CPP_SOLVER")


def load_native_module() -> Any:
    """Load and cache the native extension module if available."""
    global _NATIVE, _NATIVE_IMPORT_ERROR
    if _NATIVE is not None:
        return _NATIVE
    if _NATIVE_IMPORT_ERROR is not None:
        raise _NATIVE_IMPORT_ERROR

    try:
        _NATIVE = importlib.import_module("hydra_native")
        return _NATIVE
    except Exception as exc:  # pragma: no cover - depends on local build state
        logger.debug("[BACKEND] failed to import hydra_native: %s", exc)
        _NATIVE_IMPORT_ERROR = exc
        raise


def solve_banded_full(ab, rhs):
    """Solve banded system through native module."""
    mod = load_native_module()
    return mod.solve_banded_full(ab, rhs)


def solve_table_state(
    z,
    q_total,
    z_values,
    a_lob_raw_series,
    t_lob_raw_series,
    k_lob_raw_series,
    a_ch_series,
    t_ch_series,
    k_ch_series,
    a_rob_raw_series,
    t_rob_raw_series,
    k_rob_raw_series,
    left_activation_elev,
    right_activation_elev,
    ramp_depth,
):
    """Compute interpolated hydraulic table state through native module."""
    mod = load_native_module()
    return mod.solve_table_state(
        z,
        q_total,
        z_values,
        a_lob_raw_series,
        t_lob_raw_series,
        k_lob_raw_series,
        a_ch_series,
        t_ch_series,
        k_ch_series,
        a_rob_raw_series,
        t_rob_raw_series,
        k_rob_raw_series,
        left_activation_elev,
        right_activation_elev,
        ramp_depth,
    )


def assemble_system_core(
    reach_lengths,
    z_values,
    q_values,
    area_values,
    conveyance_values,
    top_width_values,
    velocity_values,
    alpha_values,
    dkdz_values,
    dt,
    theta,
    q_upstream_next,
    ds_is_stage,
    ds_bc_value,
    ds_bc_ramp_factor,
):
    """Assemble the unsteady banded matrix and RHS through the native module."""
    mod = load_native_module()
    return mod.assemble_system_core(
        reach_lengths,
        z_values,
        q_values,
        area_values,
        conveyance_values,
        top_width_values,
        velocity_values,
        alpha_values,
        dkdz_values,
        dt,
        theta,
        q_upstream_next,
        ds_is_stage,
        ds_bc_value,
        ds_bc_ramp_factor,
    )


def adaptive_damping_scale(
    bed_elevations,
    z_iter,
    q_iter,
    dz_raw,
    dq_raw,
    wetting_depth,
):
    """Compute the adaptive damping scale through the native module."""
    mod = load_native_module()
    return mod.adaptive_damping_scale(
        bed_elevations,
        z_iter,
        q_iter,
        dz_raw,
        dq_raw,
        wetting_depth,
    )


def compute_node_properties(
    z_n,
    Q_n,
    bed_elevations,
    min_depth: float,
    table_z_2d,
    a_lob_2d,
    t_lob_2d,
    k_lob_2d,
    a_ch_2d,
    t_ch_2d,
    k_ch_2d,
    a_rob_2d,
    t_rob_2d,
    k_rob_2d,
    dk_dz_2d,
    table_lengths,
    left_act_elev,
    right_act_elev,
    ramp_depth: float,
    L_ch,
    L_lob,
    L_rob,
    dx_fallback,
):
    """Batch node property evaluation from pre-packed 2D SoA table layout (HP2).

    Returns (reach_lengths, area, conveyance, top_width, velocity, alpha, dkdz)
    where reach_lengths has shape (N-1,) and the rest have shape (N,).
    """
    mod = load_native_module()
    return mod.compute_node_properties_cpp(
        z_n, Q_n, bed_elevations, float(min_depth),
        table_z_2d, a_lob_2d, t_lob_2d, k_lob_2d,
        a_ch_2d, t_ch_2d, k_ch_2d,
        a_rob_2d, t_rob_2d, k_rob_2d,
        dk_dz_2d, table_lengths,
        left_act_elev, right_act_elev, float(ramp_depth),
        L_ch, L_lob, L_rob, dx_fallback,
    )


def pack_node_property_bundle(
    sections_ordered,
    hydraulic_tables: Dict,
    dx_list: List[float],
) -> Dict[str, Any]:
    """Pre-pack per-section hydraulic table data into 2D SoA numpy arrays.

    All per-section table arrays are stacked row-major into (N x max_len)
    matrices, padded with the last valid value (safe for interp_linear
    out-of-bounds clamping).  Returns a dict that can be passed directly
    to compute_node_properties().

    Parameters
    ----------
    sections_ordered : list of CrossSection, upstream-to-downstream.
    hydraulic_tables : dict mapping id(xs) -> SectionHydraulicTable.
    dx_list : list of float, reach spacings (length N-1), used as fallback
              when L_ch_to_next is missing/zero on a section.

    Returns
    -------
    dict with keys: table_z_2d, a_lob_2d, t_lob_2d, k_lob_2d,
                    a_ch_2d, t_ch_2d, k_ch_2d, a_rob_2d, t_rob_2d, k_rob_2d,
                    dk_dz_2d, table_lengths (int32),
                    left_act_elev, right_act_elev,
                    L_ch, L_lob, L_rob, dx_fallback.
    """
    N = len(sections_ordered)
    tables = [hydraulic_tables[id(xs)] for xs in sections_ordered]
    lengths = np.array([len(t.z_values) for t in tables], dtype=np.int32)
    max_len = int(lengths.max())

    def _pack(attr: str) -> np.ndarray:
        mat = np.empty((N, max_len), dtype=np.float64)
        for i, t in enumerate(tables):
            arr = getattr(t, attr)
            n = len(arr)
            mat[i, :n] = arr
            if n < max_len:
                # Pad with last valid value so interp_linear clamps correctly.
                mat[i, n:] = arr[-1]
        return mat

    table_z_2d = _pack("z_values")
    a_lob_2d   = _pack("A_lob_raw")
    t_lob_2d   = _pack("T_lob_raw")
    k_lob_2d   = _pack("K_lob_raw")
    a_ch_2d    = _pack("A_ch")
    t_ch_2d    = _pack("T_ch")
    k_ch_2d    = _pack("K_ch")
    a_rob_2d   = _pack("A_rob_raw")
    t_rob_2d   = _pack("T_rob_raw")
    k_rob_2d   = _pack("K_rob_raw")
    dk_dz_2d   = _pack("dK_dz_raw")

    left_act_elev  = np.array([t.left_activation_elev  for t in tables], dtype=np.float64)
    right_act_elev = np.array([t.right_activation_elev for t in tables], dtype=np.float64)

    # Reach geometry: for reach r, the downstream section is sections_ordered[r+1],
    # but L_*_to_next on a section points toward the next upstream section in the
    # DS-to-US ordering.  The solver builds dx via the reversed-order traversal:
    # ordered_ds_to_us[N-2-r].L_ch_to_next is the length for reach r.
    ordered_ds_to_us = list(reversed(sections_ordered))
    L_ch_arr  = np.empty(N - 1, dtype=np.float64)
    L_lob_arr = np.empty(N - 1, dtype=np.float64)
    L_rob_arr = np.empty(N - 1, dtype=np.float64)
    for r in range(N - 1):
        ds_sec = ordered_ds_to_us[N - 2 - r]
        L_ch_arr[r]  = float(getattr(ds_sec, "L_ch_to_next",  0.0) or 0.0)
        L_lob_arr[r] = float(getattr(ds_sec, "L_lob_to_next", 0.0) or 0.0)
        L_rob_arr[r] = float(getattr(ds_sec, "L_rob_to_next", 0.0) or 0.0)

    dx_fb = np.array(dx_list, dtype=np.float64)

    return {
        "table_z_2d":    table_z_2d,
        "a_lob_2d":      a_lob_2d,
        "t_lob_2d":      t_lob_2d,
        "k_lob_2d":      k_lob_2d,
        "a_ch_2d":       a_ch_2d,
        "t_ch_2d":       t_ch_2d,
        "k_ch_2d":       k_ch_2d,
        "a_rob_2d":      a_rob_2d,
        "t_rob_2d":      t_rob_2d,
        "k_rob_2d":      k_rob_2d,
        "dk_dz_2d":      dk_dz_2d,
        "table_lengths": lengths,
        "left_act_elev": left_act_elev,
        "right_act_elev": right_act_elev,
        "L_ch":          L_ch_arr,
        "L_lob":         L_lob_arr,
        "L_rob":         L_rob_arr,
        "dx_fallback":   dx_fb,
    }


def run_one_timestep_unsteady_1d_cpp(
    z_n,
    q_n,
    reach_lengths,
    bed_elevations,
    area_values,
    conveyance_values,
    top_width_values,
    velocity_values,
    alpha_values,
    dkdz_values,
    dt,
    theta,
    q_upstream_next,
    ds_is_stage,
    ds_bc_value,
    ds_bc_ramp_factor,
    max_iter,
    tol,
    wetting_depth,
):
    """Run one complete Newton iteration timestep through the native module."""
    mod = load_native_module()
    return mod.run_one_timestep_unsteady_1d_cpp(
        z_n,
        q_n,
        reach_lengths,
        bed_elevations,
        area_values,
        conveyance_values,
        top_width_values,
        velocity_values,
        alpha_values,
        dkdz_values,
        dt,
        theta,
        q_upstream_next,
        ds_is_stage,
        ds_bc_value,
        ds_bc_ramp_factor,
        max_iter,
        tol,
        wetting_depth,
    )


def build_section_hydraulic_table_cpp(
    lob_x,
    lob_z,
    ch_x,
    ch_z,
    rob_x,
    rob_z,
    z_values,
    n_lob,
    n_ch,
    n_rob,
):
    """Build one section hydraulic lookup table through the native module."""
    mod = load_native_module()
    return mod.build_section_hydraulic_table_cpp(
        lob_x,
        lob_z,
        ch_x,
        ch_z,
        rob_x,
        rob_z,
        z_values,
        n_lob,
        n_ch,
        n_rob,
    )


def build_section_hydraulic_table_from_geometry_cpp(
    geom_x,
    geom_z,
    left_bank_station,
    right_bank_station,
    z_values,
    n_lob,
    n_ch,
    n_rob,
):
    """Build one section hydraulic lookup table from full section geometry."""
    mod = load_native_module()
    return mod.build_section_hydraulic_table_from_geometry_cpp(
        geom_x,
        geom_z,
        left_bank_station,
        right_bank_station,
        z_values,
        n_lob,
        n_ch,
        n_rob,
    )


def configure_table_threads_cpp(thread_count):
    """Configure native hydraulic-table thread count (0 means runtime default)."""
    mod = load_native_module()
    return mod.configure_table_threads_cpp(int(thread_count))


def get_table_threads_cpp():
    """Read configured native hydraulic-table thread count."""
    mod = load_native_module()
    return int(mod.get_table_threads_cpp())
