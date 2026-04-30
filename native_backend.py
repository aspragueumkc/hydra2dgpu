"""Optional loader for the native backwater acceleration module."""

from __future__ import annotations

import importlib
import os
from typing import Any, Optional

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
        _NATIVE = importlib.import_module("backwater_native")
        return _NATIVE
    except Exception as exc:  # pragma: no cover - depends on local build state
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
    reach_lengths_input,
    z_values,
    q_values,
    hydraulic_tables_data,
    overbank_ramp_depth,
):
    """Compute node properties (area, conveyance, width, velocity, alpha, dK/dz) through the native module."""
    mod = load_native_module()
    return mod.compute_node_properties(
        reach_lengths_input,
        z_values,
        q_values,
        hydraulic_tables_data,
        overbank_ramp_depth,
    )


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
