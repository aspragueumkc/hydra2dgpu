#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unsteady_model.py
-----------------
1D Unsteady Flow Solver — Preissmann Four-Point Implicit Scheme

Solves the 1D Saint-Venant (dynamic wave) equations:

    Continuity:  ∂A/∂t + ∂Q/∂x = 0
    Momentum:    ∂Q/∂t + ∂(QV)/∂x + gA(∂z_s/∂x + Sf) = 0

where Q is discharge (cfs), A is flow area (ft²), V = Q/A is velocity (ft/s),
z_s is water surface elevation (ft), Sf is friction slope (Manning's equation).

Algorithm reference: HEC-RAS 1D Unsteady Flow Hydrodynamics, Version 6.5,
Section 4 — Implicit Finite Difference Scheme.

The linearized Preissmann box scheme produces a pentadiagonal (bandwidth-5)
linear system solved at every time step using scipy.linalg.solve_banded
(or numpy.linalg.solve as a fallback).

Units: US Customary (ft, cfs, s).  g = 32.174 ft/s².

Notes
-----
- Floodplain/overbank is handled via the existing CrossSection subsection
  hydraulics (LOB/CH/ROB), but all are merged into total A, K, and T here.
- Culvert structures are NOT applied in the unsteady solver in this version.
- Binary results are stored as numpy blobs inside the GeoPackage (SQLite).
"""

from __future__ import annotations

import json
import math
import pickle
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# Try numpy / scipy — required for unsteady solver
try:
    import numpy as np
    _HAVE_NUMPY = True
except ImportError:
    _HAVE_NUMPY = False
    np = None  # type: ignore

try:
    from scipy.linalg import solve_banded as _scipy_solve_banded
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False
    _scipy_solve_banded = None  # type: ignore

# Shared hydraulic helpers from the steady model
try:
    from backwater_model import (
        CrossSection,
        ModelInput,
        submerged_trapezoids_area_perimeter,
        MANNING_CONST,
        G,
        _sorted_sections_by_river_station,
        solve_normal_depth,
        compute_state,
        run_backwater,
    )
except ImportError:
    from .backwater_model import (  # type: ignore
        CrossSection,
        ModelInput,
        submerged_trapezoids_area_perimeter,
        MANNING_CONST,
        G,
        _sorted_sections_by_river_station,
        solve_normal_depth,
        compute_state,
        run_backwater,
    )

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class HydrographBC:
    """Boundary condition time-series (upstream flow or downstream stage).

    Attributes
    ----------
    times : list of float
        Times in seconds from start of simulation.
    values : list of float
        Discharge (cfs) or water-surface elevation (ft), matching *bc_type*.
    bc_type : str
        ``'flow'`` for upstream discharge hydrograph, ``'stage'`` for
        downstream WSE hydrograph.
    label : str
        Human-readable label (for UI display).
    """
    times: List[float]
    values: List[float]
    bc_type: str = 'flow'   # 'flow' or 'stage'
    label: str = ''

    def interpolate(self, t: float) -> float:
        """Linear interpolation at time *t* (seconds)."""
        ts = self.times
        vs = self.values
        if not ts:
            return 0.0
        if t <= ts[0]:
            return vs[0]
        if t >= ts[-1]:
            return vs[-1]
        # binary search
        lo, hi = 0, len(ts) - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if ts[mid] <= t:
                lo = mid
            else:
                hi = mid
        frac = (t - ts[lo]) / (ts[hi] - ts[lo]) if ts[hi] != ts[lo] else 0.0
        return vs[lo] + frac * (vs[hi] - vs[lo])


@dataclass
class UnsteadyParams:
    """Simulation control parameters for the unsteady solver.

    Attributes
    ----------
    dt : float
        Computational time step in seconds.
    t_end : float
        Total simulation duration in seconds.
    theta : float
        Preissmann weighting factor (0.5 ≤ θ ≤ 1.0).  θ=1 is fully
        implicit and most damped; θ=0.6 is a good practical choice.
    output_interval : int
        Store results every *output_interval* time steps (1 = every step).
    max_iter : int
        Max inner iterations per time step for updating nonlinear Sf terms.
    tol : float
        Convergence tolerance on max(|Δz|, |ΔQ|) for inner iterations.
    downstream_bc : str
        ``'normal_depth'`` (Manning's normal depth) or ``'stage'`` (uses
        *downstream_hydrograph*).
    downstream_value : float
        S₀ (slope) for normal-depth BC, or unused if ``'stage'`` BC is
        provided via *downstream_hydrograph*.
    downstream_hydrograph : HydrographBC or None
        Required if *downstream_bc* == ``'stage'``.
    debug_output_path : str
        Optional JSON path for first-step diagnostics of the upstream
        boundary and first reach assembly.
    debug_capture : bool
        Enable detailed timestep debug capture for solver internals.
    debug_frequency : str
        ``'output'`` to capture only output timesteps, or ``'computation'``
        to capture every computational timestep.
    precompute_hydraulic_tables : bool
        Precompute section hydraulic properties on a stage grid at the start
        of the run and use interpolation during the solve.
    hydraulic_table_dz : float
        Stage increment (ft) used for the precomputed hydraulic lookup table.
        Smaller values improve fidelity and increase startup preprocessing time.
    hydraulic_table_padding : float
        Extra elevation range (ft) above the highest surveyed point to include
        in the lookup table before falling back to direct geometry evaluation.
    """
    dt: float = 60.0
    t_end: float = 3600.0
    theta: float = 0.6
    output_interval: int = 1
    max_iter: int = 4
    tol: float = 1e-4
    downstream_bc: str = 'normal_depth'
    downstream_value: float = 0.001
    downstream_hydrograph: Optional[HydrographBC] = None
    debug_output_path: str = ''
    debug_capture: bool = False
    debug_frequency: str = 'output'
    precompute_hydraulic_tables: bool = True
    hydraulic_table_dz: float = 0.01
    hydraulic_table_padding: float = 5.0


@dataclass
class UnsteadyResults:
    """Results from a 1D unsteady simulation.

    Arrays have shape ``(n_output_times, n_sections)``.

    Attributes
    ----------
    times : np.ndarray, shape (n_output_times,)
        Simulation times (seconds) at which results are stored.
    wse : np.ndarray, shape (n_output_times, n_sections)
        Water-surface elevation (ft).  Sections ordered upstream to downstream.
    q : np.ndarray, shape (n_output_times, n_sections)
        Discharge (cfs).
    max_wse : np.ndarray, shape (n_sections,)
        Maximum WSE over all time steps.
    section_ids : list of str
        River-station strings, upstream to downstream, corresponding to
        columns of *wse* and *q*.
    run_id : str
        UTC timestamp string used as primary key in GeoPackage storage.
    run_time : str
        Human-readable run datetime.
    dt : float
        Computational time step used (seconds).
    n_sections : int
    n_output_times : int
    """
    times: Any
    wse: Any
    q: Any
    max_wse: Any
    section_ids: List[str]
    run_id: str = ''
    run_time: str = ''
    dt: float = 0.0
    n_sections: int = 0
    n_output_times: int = 0
    debug_records: Optional[List[dict]] = None


@dataclass
class UnsteadySectionState:
    """Hydraulic state used by the unsteady solver for one cross section."""
    z: float
    alpha: float
    A_lob: float
    A_ch: float
    A_rob: float
    T_lob: float
    T_ch: float
    T_rob: float
    K_lob: float
    K_ch: float
    K_rob: float
    Q_lob: float
    Q_ch: float
    Q_rob: float
    A_t: float
    T_t: float
    K_t: float
    V_t: float
    left_activation_elev: float
    right_activation_elev: float
    left_activation_factor: float
    right_activation_factor: float


@dataclass
class SectionHydraulicTable:
    """Precomputed stage-property lookup table for one cross section."""
    z_values: Any
    A_lob_raw: Any
    T_lob_raw: Any
    K_lob_raw: Any
    A_ch: Any
    T_ch: Any
    K_ch: Any
    A_rob_raw: Any
    T_rob_raw: Any
    K_rob_raw: Any
    K_total_raw: Any
    dK_dz_raw: Any
    left_activation_elev: float
    right_activation_elev: float

    def covers(self, z: float) -> bool:
        return bool(self.z_values[0] <= z <= self.z_values[-1])


# ---------------------------------------------------------------------------
# Section hydraulic helpers (subsection K, T, A)
# ---------------------------------------------------------------------------

WETTING_DEPTH_FT = 0.001


def _station_elevation(xs: CrossSection, station: float) -> float:
    """Linear interpolation of the section elevation at a given station."""
    pts = sorted(xs.geometry, key=lambda p: p[0])
    if not pts:
        return 0.0
    if station <= pts[0][0]:
        return float(pts[0][1])
    if station >= pts[-1][0]:
        return float(pts[-1][1])
    for idx in range(1, len(pts)):
        x0, z0 = pts[idx - 1]
        x1, z1 = pts[idx]
        if x0 <= station <= x1:
            if x1 == x0:
                return float(z0)
            frac = (station - x0) / (x1 - x0)
            return float(z0 + frac * (z1 - z0))
    return float(pts[-1][1])


def _activation_factor(stage: float, activation_elev: float, ramp_depth: float = WETTING_DEPTH_FT) -> float:
    """Smoothly activate an overbank once stage reaches its control elevation."""
    depth = float(stage) - float(activation_elev)
    if depth <= 0.0:
        return 0.0
    if ramp_depth <= 0.0:
        return 1.0
    return max(0.0, min(1.0, depth / float(ramp_depth)))


def _overbank_activation_elevation(xs: CrossSection, side: str) -> float:
    """Return the activation elevation for an overbank.

    Default control is the bank elevation at the subsection cutline. Optional
    ineffective/blocked/levee elevations can raise that threshold.
    """
    if side == 'left':
        bank_station = xs.left_bank_station
        attr_names = (
            'left_ineffective_elev', 'ineffective_left_elev',
            'left_blocked_elev', 'blocked_left_elev',
            'left_levee_elev', 'levee_left_elev',
        )
    else:
        bank_station = xs.right_bank_station
        attr_names = (
            'right_ineffective_elev', 'ineffective_right_elev',
            'right_blocked_elev', 'blocked_right_elev',
            'right_levee_elev', 'levee_right_elev',
        )

    candidates = [_station_elevation(xs, bank_station)]
    for name in attr_names:
        value = getattr(xs, name, None)
        if value is None:
            continue
        try:
            candidates.append(float(value))
        except Exception:
            continue
    return max(candidates)


def _subsection_hydraulics(geom: List[Tuple[float, float]], z: float, n_val: float) -> Tuple[float, float, float, float]:
    """Return subsection (A, P, T, K) at stage z."""
    if not geom:
        return 0.0, 0.0, 0.0, 0.0
    A, P, T = submerged_trapezoids_area_perimeter(geom, z)
    if A <= 0.0 or P <= 0.0:
        return max(0.0, A), max(0.0, P), max(0.0, T), 0.0
    R = A / P
    K = (MANNING_CONST / n_val) * A * (R ** (2.0 / 3.0)) if n_val > 0.0 else 0.0
    return A, P, max(0.0, T), K


def _interp_table_value(z_values: Any, values: Any, z: float) -> float:
    """Linearly interpolate a tabled hydraulic property at stage *z*."""
    return float(np.interp(float(z), z_values, values))


def _interp_table_slope(z_values: Any, values: Any, z: float) -> Optional[float]:
    """Piecewise-linear slope of a tabled property with respect to stage."""
    if z <= float(z_values[0]) or z >= float(z_values[-1]):
        return None
    idx = int(np.searchsorted(z_values, float(z), side='right'))
    idx = max(1, min(idx, len(z_values) - 1))
    z0 = float(z_values[idx - 1])
    z1 = float(z_values[idx])
    if z1 <= z0:
        return 0.0
    return float((values[idx] - values[idx - 1]) / (z1 - z0))


def _build_section_hydraulic_table(
    xs: CrossSection,
    dz: float,
    padding: float,
) -> SectionHydraulicTable:
    """Precompute subsection hydraulic properties on a regular stage grid."""
    if dz <= 0.0:
        raise ValueError("hydraulic_table_dz must be positive.")

    z_min = _hydraulic_bed_elevation(xs) + WETTING_DEPTH_FT
    z_top = max(p[1] for p in xs.geometry)
    z_max = max(z_min + dz, z_top + max(float(padding), dz))
    n_points = max(32, int(math.ceil((z_max - z_min) / dz)) + 1)
    z_values = np.linspace(z_min, z_max, n_points, dtype=np.float64)

    lob_g, ch_g, rob_g = xs._subgeometry()
    A_lob_raw = np.empty(n_points, dtype=np.float64)
    T_lob_raw = np.empty(n_points, dtype=np.float64)
    K_lob_raw = np.empty(n_points, dtype=np.float64)
    A_ch = np.empty(n_points, dtype=np.float64)
    T_ch = np.empty(n_points, dtype=np.float64)
    K_ch = np.empty(n_points, dtype=np.float64)
    A_rob_raw = np.empty(n_points, dtype=np.float64)
    T_rob_raw = np.empty(n_points, dtype=np.float64)
    K_rob_raw = np.empty(n_points, dtype=np.float64)

    for idx, z_val in enumerate(z_values):
        A_lob_raw[idx], _P_lob, T_lob_raw[idx], K_lob_raw[idx] = _subsection_hydraulics(lob_g, float(z_val), xs.n_lob)
        A_ch[idx], _P_ch, T_ch[idx], K_ch[idx] = _subsection_hydraulics(ch_g, float(z_val), xs.n_ch)
        A_rob_raw[idx], _P_rob, T_rob_raw[idx], K_rob_raw[idx] = _subsection_hydraulics(rob_g, float(z_val), xs.n_rob)

    return SectionHydraulicTable(
        z_values=z_values,
        A_lob_raw=A_lob_raw,
        T_lob_raw=T_lob_raw,
        K_lob_raw=K_lob_raw,
        A_ch=A_ch,
        T_ch=T_ch,
        K_ch=K_ch,
        A_rob_raw=A_rob_raw,
        T_rob_raw=T_rob_raw,
        K_rob_raw=K_rob_raw,
        K_total_raw=K_lob_raw + K_ch + K_rob_raw,
        dK_dz_raw=np.gradient(K_lob_raw + K_ch + K_rob_raw, z_values, edge_order=2),
        left_activation_elev=_overbank_activation_elevation(xs, 'left'),
        right_activation_elev=_overbank_activation_elevation(xs, 'right'),
    )


def _build_hydraulic_tables(
    sections_us_to_ds: List[CrossSection],
    dz: float,
    padding: float,
) -> Dict[int, SectionHydraulicTable]:
    """Build lookup tables for all cross sections in the model."""
    return {
        id(xs): _build_section_hydraulic_table(xs, dz=dz, padding=padding)
        for xs in sections_us_to_ds
    }


def _unsteady_section_state_direct(xs: CrossSection, z: float, Q_total: float) -> UnsteadySectionState:
    """Compute subsection-aware hydraulics directly from section geometry."""
    z = _regularized_wse(xs, z)
    lob_g, ch_g, rob_g = xs._subgeometry()

    A_lob_raw, _P_lob, T_lob_raw, K_lob_raw = _subsection_hydraulics(lob_g, z, xs.n_lob)
    A_ch, _P_ch, T_ch, K_ch = _subsection_hydraulics(ch_g, z, xs.n_ch)
    A_rob_raw, _P_rob, T_rob_raw, K_rob_raw = _subsection_hydraulics(rob_g, z, xs.n_rob)

    left_activation_elev = _overbank_activation_elevation(xs, 'left')
    right_activation_elev = _overbank_activation_elevation(xs, 'right')
    left_factor = _activation_factor(z, left_activation_elev)
    right_factor = _activation_factor(z, right_activation_elev)

    A_lob = left_factor * A_lob_raw
    T_lob = left_factor * T_lob_raw
    K_lob = left_factor * K_lob_raw
    A_rob = right_factor * A_rob_raw
    T_rob = right_factor * T_rob_raw
    K_rob = right_factor * K_rob_raw

    A_t = A_lob + A_ch + A_rob
    T_t = T_lob + T_ch + T_rob
    K_t = K_lob + K_ch + K_rob

    if K_t > 0.0:
        Q_lob = Q_total * (K_lob / K_t)
        Q_ch = Q_total * (K_ch / K_t)
        Q_rob = Q_total * (K_rob / K_t)
    else:
        Q_lob = 0.0
        Q_ch = 0.0
        Q_rob = 0.0

    V_t = Q_total / A_t if A_t > 0.0 else 0.0
    alpha_num = 0.0
    for K_i, A_i in ((K_lob, A_lob), (K_ch, A_ch), (K_rob, A_rob)):
        if K_i > 0.0 and A_i > 0.0:
            alpha_num += (K_i ** 3) / (A_i ** 2)
    alpha = ((A_t ** 2) * alpha_num / (K_t ** 3)) if K_t > 0.0 and A_t > 0.0 else 1.0

    if T_t <= 0.0:
        dz = 1e-3
        A_hi, _, _ = submerged_trapezoids_area_perimeter(xs.geometry, z + dz)
        A_lo, _, _ = submerged_trapezoids_area_perimeter(xs.geometry, max(z - dz, min(p[1] for p in xs.geometry) + 1e-6))
        T_t = max(0.01, (A_hi - A_lo) / (2.0 * dz))

    return UnsteadySectionState(
        z=z,
        alpha=alpha,
        A_lob=A_lob,
        A_ch=A_ch,
        A_rob=A_rob,
        T_lob=T_lob,
        T_ch=T_ch,
        T_rob=T_rob,
        K_lob=K_lob,
        K_ch=K_ch,
        K_rob=K_rob,
        Q_lob=Q_lob,
        Q_ch=Q_ch,
        Q_rob=Q_rob,
        A_t=A_t,
        T_t=T_t,
        K_t=K_t,
        V_t=V_t,
        left_activation_elev=left_activation_elev,
        right_activation_elev=right_activation_elev,
        left_activation_factor=left_factor,
        right_activation_factor=right_factor,
    )


def _unsteady_section_state_from_table(
    xs: CrossSection,
    hydraulic_table: SectionHydraulicTable,
    z: float,
    Q_total: float,
) -> UnsteadySectionState:
    """Compute section hydraulics by interpolating a precomputed stage table."""
    z = _regularized_wse(xs, z)
    if not hydraulic_table.covers(z):
        return _unsteady_section_state_direct(xs, z, Q_total)

    A_lob_raw = _interp_table_value(hydraulic_table.z_values, hydraulic_table.A_lob_raw, z)
    T_lob_raw = _interp_table_value(hydraulic_table.z_values, hydraulic_table.T_lob_raw, z)
    K_lob_raw = _interp_table_value(hydraulic_table.z_values, hydraulic_table.K_lob_raw, z)
    A_ch = _interp_table_value(hydraulic_table.z_values, hydraulic_table.A_ch, z)
    T_ch = _interp_table_value(hydraulic_table.z_values, hydraulic_table.T_ch, z)
    K_ch = _interp_table_value(hydraulic_table.z_values, hydraulic_table.K_ch, z)
    A_rob_raw = _interp_table_value(hydraulic_table.z_values, hydraulic_table.A_rob_raw, z)
    T_rob_raw = _interp_table_value(hydraulic_table.z_values, hydraulic_table.T_rob_raw, z)
    K_rob_raw = _interp_table_value(hydraulic_table.z_values, hydraulic_table.K_rob_raw, z)

    left_factor = _activation_factor(z, hydraulic_table.left_activation_elev)
    right_factor = _activation_factor(z, hydraulic_table.right_activation_elev)

    A_lob = left_factor * A_lob_raw
    T_lob = left_factor * T_lob_raw
    K_lob = left_factor * K_lob_raw
    A_rob = right_factor * A_rob_raw
    T_rob = right_factor * T_rob_raw
    K_rob = right_factor * K_rob_raw

    A_t = A_lob + A_ch + A_rob
    T_t = T_lob + T_ch + T_rob
    K_t = K_lob + K_ch + K_rob

    if K_t > 0.0:
        Q_lob = Q_total * (K_lob / K_t)
        Q_ch = Q_total * (K_ch / K_t)
        Q_rob = Q_total * (K_rob / K_t)
    else:
        Q_lob = 0.0
        Q_ch = 0.0
        Q_rob = 0.0

    V_t = Q_total / A_t if A_t > 0.0 else 0.0
    alpha_num = 0.0
    for K_i, A_i in ((K_lob, A_lob), (K_ch, A_ch), (K_rob, A_rob)):
        if K_i > 0.0 and A_i > 0.0:
            alpha_num += (K_i ** 3) / (A_i ** 2)
    alpha = ((A_t ** 2) * alpha_num / (K_t ** 3)) if K_t > 0.0 and A_t > 0.0 else 1.0

    if T_t <= 0.0:
        slope = _interp_table_slope(hydraulic_table.z_values, hydraulic_table.A_ch + hydraulic_table.A_lob_raw + hydraulic_table.A_rob_raw, z)
        T_t = max(0.01, slope if slope is not None else 0.01)

    return UnsteadySectionState(
        z=z,
        alpha=alpha,
        A_lob=A_lob,
        A_ch=A_ch,
        A_rob=A_rob,
        T_lob=T_lob,
        T_ch=T_ch,
        T_rob=T_rob,
        K_lob=K_lob,
        K_ch=K_ch,
        K_rob=K_rob,
        Q_lob=Q_lob,
        Q_ch=Q_ch,
        Q_rob=Q_rob,
        A_t=A_t,
        T_t=T_t,
        K_t=K_t,
        V_t=V_t,
        left_activation_elev=hydraulic_table.left_activation_elev,
        right_activation_elev=hydraulic_table.right_activation_elev,
        left_activation_factor=left_factor,
        right_activation_factor=right_factor,
    )


def _unsteady_section_state(
    xs: CrossSection,
    z: float,
    Q_total: float,
    hydraulic_table: Optional[SectionHydraulicTable] = None,
) -> UnsteadySectionState:
    """Compute subsection-aware hydraulics for the unsteady solver."""
    if hydraulic_table is not None:
        return _unsteady_section_state_from_table(xs, hydraulic_table, z, Q_total)
    return _unsteady_section_state_direct(xs, z, Q_total)


def _effective_reach_length(xs_downstream: CrossSection, state_up: UnsteadySectionState, state_down: UnsteadySectionState, fallback_length: float) -> float:
    """Return HEC-RAS-like discharge-weighted reach length for a reach."""
    L_ch = float(getattr(xs_downstream, 'L_ch_to_next', 0.0) or 0.0)
    L_lob = float(getattr(xs_downstream, 'L_lob_to_next', 0.0) or 0.0)
    L_rob = float(getattr(xs_downstream, 'L_rob_to_next', 0.0) or 0.0)

    if L_ch <= 0.0:
        L_ch = float(fallback_length)
    if L_lob <= 0.0:
        L_lob = L_ch
    if L_rob <= 0.0:
        L_rob = L_ch

    Q_lob_av = 0.5 * (abs(state_up.Q_lob) + abs(state_down.Q_lob))
    Q_ch_av = 0.5 * (abs(state_up.Q_ch) + abs(state_down.Q_ch))
    Q_rob_av = 0.5 * (abs(state_up.Q_rob) + abs(state_down.Q_rob))
    Q_total = Q_lob_av + Q_ch_av + Q_rob_av
    if Q_total <= 0.0:
        return max(1.0, L_ch)
    return max(1.0, (L_lob * Q_lob_av + L_ch * Q_ch_av + L_rob * Q_rob_av) / Q_total)


def _apply_adaptive_damping(
    sections_us_to_ds: List[CrossSection],
    z_iter: Any,
    Q_iter: Any,
    dz_raw: Any,
    dQ_raw: Any,
) -> Tuple[Any, Any, float]:
    """Scale Newton updates to reduce overshoot near wet/dry transitions."""
    scale = 1.0
    for i, xs in enumerate(sections_us_to_ds):
        hbed = _hydraulic_bed_elevation(xs)
        depth = max(0.0, float(z_iter[i]) - hbed)

        # Limit stage updates to a fraction of local depth with sensible floors/caps.
        max_dz = max(0.05, min(0.5, 0.5 * max(depth, WETTING_DEPTH_FT)))
        dz_abs = abs(float(dz_raw[i]))
        if dz_abs > max_dz and dz_abs > 0.0:
            scale = min(scale, max_dz / dz_abs)

        # Limit discharge updates relative to local flow magnitude.
        q_ref = max(20.0, abs(float(Q_iter[i])))
        max_dQ = 0.35 * q_ref + 10.0
        dQ_abs = abs(float(dQ_raw[i]))
        if dQ_abs > max_dQ and dQ_abs > 0.0:
            scale = min(scale, max_dQ / dQ_abs)

    # Keep a minimum step so iterations can still converge in reasonable time.
    scale = max(0.05, min(1.0, scale))
    return dz_raw * scale, dQ_raw * scale, float(scale)


def _linear_system_residual_inf(ab: Any, delta: Any, rhs_vec: Any) -> float:
    """Infinity norm of linear residual for debug diagnostics."""
    n = len(rhs_vec)
    residual = np.zeros(n, dtype=np.float64)
    for j in range(n):
        xj = float(delta[j])
        if xj == 0.0:
            continue
        for band_row in range(5):
            i = j + band_row - 2
            if 0 <= i < n:
                residual[i] += float(ab[band_row, j]) * xj
    residual -= rhs_vec
    return float(np.max(np.abs(residual)))


def _capture_step_debug(
    sections_us_to_ds: List[CrossSection],
    dx: List[float],
    z_state: Any,
    q_state: Any,
    step: int,
    t_new: float,
    output_step: bool,
    inner_stats: List[dict],
    hydraulic_tables: Optional[Dict[int, SectionHydraulicTable]] = None,
) -> dict:
    """Capture a detailed solver snapshot for one timestep."""
    states = []
    for xs, z_val, q_val in zip(sections_us_to_ds, z_state, q_state):
        s = _unsteady_section_state(
            xs,
            float(z_val),
            float(q_val),
            hydraulic_table=hydraulic_tables.get(id(xs)) if hydraulic_tables else None,
        )
        states.append(s)

    reach_lengths = []
    sf_by_node = []
    dkdz_by_node = []
    ds_to_us = list(reversed(sections_us_to_ds))
    for i, (xs, s, q_val) in enumerate(zip(sections_us_to_ds, states, q_state)):
        sf_by_node.append(_Sf(float(q_val), s.K_t))
        dkdz_by_node.append(
            _dK_dz(
                xs,
                float(z_state[i]),
                hydraulic_table=hydraulic_tables.get(id(xs)) if hydraulic_tables else None,
            )
        )
    for r in range(len(states) - 1):
        xs_down = ds_to_us[len(states) - 2 - r]
        reach_lengths.append(
            _effective_reach_length(xs_down, states[r], states[r + 1], dx[r])
        )

    return {
        'step': int(step),
        'time_s': float(t_new),
        'is_output_step': bool(output_step),
        'section_ids': [str(xs.river_station) for xs in sections_us_to_ds],
        'z': [float(v) for v in z_state],
        'q': [float(v) for v in q_state],
        'A_t': [float(s.A_t) for s in states],
        'T_t': [float(s.T_t) for s in states],
        'K_t': [float(s.K_t) for s in states],
        'V_t': [float(s.V_t) for s in states],
        'alpha': [float(s.alpha) for s in states],
        'Q_lob': [float(s.Q_lob) for s in states],
        'Q_ch': [float(s.Q_ch) for s in states],
        'Q_rob': [float(s.Q_rob) for s in states],
        'A_lob': [float(s.A_lob) for s in states],
        'A_ch': [float(s.A_ch) for s in states],
        'A_rob': [float(s.A_rob) for s in states],
        'K_lob': [float(s.K_lob) for s in states],
        'K_ch': [float(s.K_ch) for s in states],
        'K_rob': [float(s.K_rob) for s in states],
        'Sf': [float(v) for v in sf_by_node],
        'dKdz': [float(v) for v in dkdz_by_node],
        'left_activation_factor': [float(s.left_activation_factor) for s in states],
        'right_activation_factor': [float(s.right_activation_factor) for s in states],
        'left_activation_elev': [float(s.left_activation_elev) for s in states],
        'right_activation_elev': [float(s.right_activation_elev) for s in states],
        'effective_reach_length': [float(v) for v in reach_lengths],
        'inner_iterations': inner_stats,
    }


def _hydraulic_bed_elevation(xs: CrossSection) -> float:
    """Return the controlling bed elevation for wetting/drying checks.

    Prefer the main-channel minimum when channel subsection geometry exists;
    otherwise fall back to the overall section minimum.
    """
    z_global = min(p[1] for p in xs.geometry)
    try:
        _lob, ch_g, _rob = xs._subgeometry()
    except Exception:
        ch_g = []
    if ch_g:
        return min(p[1] for p in ch_g)
    return z_global


def _regularized_wse(xs: CrossSection, z: float, min_depth: float = WETTING_DEPTH_FT) -> float:
    """Clamp stage to a minimum wetting depth above the hydraulic bed."""
    return max(float(z), _hydraulic_bed_elevation(xs) + max(1e-6, float(min_depth)))

def _compute_total_K(
    xs: CrossSection,
    z: float,
    hydraulic_table: Optional[SectionHydraulicTable] = None,
) -> float:
    """Total Manning conveyance at WSE *z* using subsection roughness."""
    if hydraulic_table is not None and hydraulic_table.covers(z):
        left_factor = _activation_factor(z, hydraulic_table.left_activation_elev)
        right_factor = _activation_factor(z, hydraulic_table.right_activation_elev)
        return (
            left_factor * _interp_table_value(hydraulic_table.z_values, hydraulic_table.K_lob_raw, z)
            + _interp_table_value(hydraulic_table.z_values, hydraulic_table.K_ch, z)
            + right_factor * _interp_table_value(hydraulic_table.z_values, hydraulic_table.K_rob_raw, z)
        )

    lob_g, ch_g, rob_g = xs._subgeometry()

    def _K_sub(geom, n_val):
        if not geom:
            return 0.0
        A, P, _ = submerged_trapezoids_area_perimeter(geom, z)
        if A <= 0.0 or P <= 0.0:
            return 0.0
        R = A / P
        return (MANNING_CONST / n_val) * A * (R ** (2.0 / 3.0))

    return _K_sub(lob_g, xs.n_lob) + _K_sub(ch_g, xs.n_ch) + _K_sub(rob_g, xs.n_rob)


def _section_vars(
    xs: CrossSection,
    z: float,
    hydraulic_table: Optional[SectionHydraulicTable] = None,
) -> Tuple[float, float, float]:
    """Return (A_total, K_total, T_top_width) at WSE *z*.

    *T* is the free-surface top width (≈ dA/dz), computed numerically.
    """
    state = _unsteady_section_state(xs, z, 0.0, hydraulic_table=hydraulic_table)
    return state.A_t, state.K_t, state.T_t


def _dK_dz(
    xs: CrossSection,
    z: float,
    dz: float = 1e-3,
    hydraulic_table: Optional[SectionHydraulicTable] = None,
) -> float:
    """Numerical dK/dz at WSE *z*."""
    z = _regularized_wse(xs, z)
    z_min = _hydraulic_bed_elevation(xs) + 1e-6
    if hydraulic_table is not None:
        if hydraulic_table.covers(z):
            return _interp_table_value(hydraulic_table.z_values, hydraulic_table.dK_dz_raw, z)
    K_hi = _compute_total_K(xs, z + dz, hydraulic_table=hydraulic_table)
    K_lo = _compute_total_K(xs, max(z - dz, z_min), hydraulic_table=hydraulic_table)
    return (K_hi - K_lo) / (2.0 * dz)


def _Sf(Q: float, K: float) -> float:
    """Friction slope Sf = Q|Q|/K² (signed, handles reverse flow)."""
    if K <= 0.0:
        return 0.0
    return Q * abs(Q) / (K * K)


def _dSf_dQ(Q: float, K: float) -> float:
    """d(Sf)/d(Q) = 2|Q|/K²."""
    if K <= 0.0:
        return 0.0
    return 2.0 * abs(Q) / (K * K)


def _dSf_dz(Q: float, K: float, dKdz: float) -> float:
    """d(Sf)/d(z) = -2*Q|Q|/K³ * dK/dz = -2*Sf * dK_dz / K."""
    if K <= 0.0:
        return 0.0
    return -2.0 * Q * abs(Q) / (K ** 3) * dKdz


def _normal_depth_Q(
    xs: CrossSection,
    S0: float,
    z: float,
    hydraulic_table: Optional[SectionHydraulicTable] = None,
) -> float:
    """Q from Manning normal-depth rating at elevation *z*."""
    z = _regularized_wse(xs, z)
    K = _compute_total_K(xs, z, hydraulic_table=hydraulic_table)
    if S0 <= 0.0 or K <= 0.0:
        return 0.0
    return K * math.sqrt(S0)


def _dQ_dz_normal(
    xs: CrossSection,
    S0: float,
    z: float,
    dz: float = 1e-3,
    hydraulic_table: Optional[SectionHydraulicTable] = None,
) -> float:
    """Numerical dQ/dz for normal-depth BC linearization."""
    return (
        _normal_depth_Q(xs, S0, z + dz, hydraulic_table=hydraulic_table)
        - _normal_depth_Q(xs, S0, z - dz, hydraulic_table=hydraulic_table)
    ) / (2.0 * dz)


# ---------------------------------------------------------------------------
# Preissmann implicit scheme — matrix assembly
# ---------------------------------------------------------------------------

def _assemble_system(
    sections_us_to_ds: List[CrossSection],
    dx: List[float],
    z_n: Any,
    Q_n: Any,
    dt: float,
    theta: float,
    Q_upstream_next: float,
    ds_bc: str,
    ds_bc_value: float,          # S0 for normal depth, or z_ds^{n+1} for stage
    hydraulic_tables: Optional[Dict[int, SectionHydraulicTable]] = None,
) -> Tuple[Any, Any]:
    """Build the pentadiagonal (bandwidth-5) matrix and RHS vector.

    Unknown vector x of length 2N:
        x[2i]   = Δz_i   (change in WSE at node i)
        x[2i+1] = ΔQ_i   (change in discharge at node i)
    Nodes numbered 0 (most upstream) → N-1 (most downstream).

    Returns
    -------
    ab : ndarray, shape (5, 2N)
        Banded storage for scipy.linalg.solve_banded with l=2, u=2.
        ab[2 + i - j, j] = A[i, j]
    rhs : ndarray, shape (2N,)
    """
    N = len(sections_us_to_ds)
    size = 2 * N
    ab = np.zeros((5, size), dtype=np.float64)
    rhs = np.zeros(size, dtype=np.float64)

    # ------------------------------------------------------------------
    # Row 0: Upstream BC — prescribe ΔQ_0 = Q_upstream^{n+1} - Q_0^n
    #   → 1*ΔQ_0 = Q_bc_next - Q_n[0]
    #   Unknown x[1] = ΔQ_0; Row=0, Col=1, i-j=-1 → ab[2+(-1), 1] = ab[1, 1]
    # ------------------------------------------------------------------
    ab[1, 1] = 1.0
    rhs[0] = Q_upstream_next - Q_n[0]

    # ------------------------------------------------------------------
    # Rows 2r+1, 2r+2 for each reach r = 0 … N-2
    # ------------------------------------------------------------------
    for r in range(N - 1):
        xs_r   = sections_us_to_ds[r]
        xs_rp1 = sections_us_to_ds[r + 1]

        z_r   = z_n[r];   Q_r   = Q_n[r]
        z_rp1 = z_n[r+1]; Q_rp1 = Q_n[r+1]
        table_r = hydraulic_tables.get(id(xs_r)) if hydraulic_tables else None
        table_rp1 = hydraulic_tables.get(id(xs_rp1)) if hydraulic_tables else None

        state_r = _unsteady_section_state(xs_r, z_r, Q_r, hydraulic_table=table_r)
        state_rp1 = _unsteady_section_state(xs_rp1, z_rp1, Q_rp1, hydraulic_table=table_rp1)
        L = _effective_reach_length(xs_rp1, state_r, state_rp1, dx[r])

        A_r, K_r, T_r = state_r.A_t, state_r.K_t, state_r.T_t
        A_rp1, K_rp1, T_rp1 = state_rp1.A_t, state_rp1.K_t, state_rp1.T_t

        Sf_r   = _Sf(Q_r,   K_r)
        Sf_rp1 = _Sf(Q_rp1, K_rp1)
        Sf_avg = 0.5 * (Sf_r + Sf_rp1)
        Abar   = 0.5 * (A_r + A_rp1)

        V_r = state_r.V_t
        V_rp1 = state_rp1.V_t
        alpha_r = state_r.alpha
        alpha_rp1 = state_rp1.alpha

        dSf_dQ_r   = _dSf_dQ(Q_r,   K_r)
        dSf_dQ_rp1 = _dSf_dQ(Q_rp1, K_rp1)
        dKdz_r     = _dK_dz(xs_r,   z_r, hydraulic_table=table_r)
        dKdz_rp1   = _dK_dz(xs_rp1, z_rp1, hydraulic_table=table_rp1)
        dSf_dz_r   = _dSf_dz(Q_r,   K_r,   dKdz_r)
        dSf_dz_rp1 = _dSf_dz(Q_rp1, K_rp1, dKdz_rp1)

        # ==============================================================
        # Continuity row (row_c = 2r+1):
        #   CZ_r*Δz_r + CQ_r*ΔQ_r + CZ_{r+1}*Δz_{r+1} + CQ_{r+1}*ΔQ_{r+1} = CB
        # ==============================================================
        row_c = 2 * r + 1
        CZ_r   =  T_r   / (2.0 * dt)
        CQ_r   = -theta  / L
        CZ_rp1 =  T_rp1 / (2.0 * dt)
        CQ_rp1 =  theta  / L
        CB     = -(Q_rp1 - Q_r) / L

        # Columns for row_c: 2r(Δz_r), 2r+1(ΔQ_r), 2r+2(Δz_{r+1}), 2r+3(ΔQ_{r+1})
        # ab[2 + row_c - col, col]
        ab[3, 2*r    ] += CZ_r    # i-j = 1  → ab[3, 2r]
        ab[2, 2*r+1  ] += CQ_r    # i-j = 0  → ab[2, 2r+1]  (diagonal)
        ab[1, 2*r+2  ] += CZ_rp1  # i-j = -1 → ab[1, 2r+2]
        ab[0, 2*r+3  ] += CQ_rp1  # i-j = -2 → ab[0, 2r+3]
        rhs[row_c]     += CB

        # ==============================================================
        # Momentum row (row_m = 2r+2):
        #   MZ_r*Δz_r + MQ_r*ΔQ_r + MZ_{r+1}*Δz_{r+1} + MQ_{r+1}*ΔQ_{r+1} = MB
        # ==============================================================
        row_m = 2 * r + 2
        MQ_r   = (1.0 / (2.0*dt)) - theta*alpha_r*V_r/L   + theta*G*Abar * 0.5*dSf_dQ_r
        MZ_r   = theta*G*Abar * 0.5*dSf_dz_r       - G*Abar*theta/L
        MQ_rp1 = (1.0 / (2.0*dt)) + theta*alpha_rp1*V_rp1/L + theta*G*Abar * 0.5*dSf_dQ_rp1
        MZ_rp1 = theta*G*Abar * 0.5*dSf_dz_rp1    + G*Abar*theta/L

        MB = -(
            (alpha_rp1 * Q_rp1 * V_rp1 - alpha_r * Q_r * V_r) / L
            + G * Abar * (z_rp1 - z_r) / L
            + G * Abar * Sf_avg
        )

        # Columns for row_m: 2r(Δz_r), 2r+1(ΔQ_r), 2r+2(Δz_{r+1}), 2r+3(ΔQ_{r+1})
        ab[4, 2*r    ] += MZ_r    # i-j = 2  → ab[4, 2r]
        ab[3, 2*r+1  ] += MQ_r    # i-j = 1  → ab[3, 2r+1]
        ab[2, 2*r+2  ] += MZ_rp1  # i-j = 0  → ab[2, 2r+2]  (diagonal)
        ab[1, 2*r+3  ] += MQ_rp1  # i-j = -1 → ab[1, 2r+3]
        rhs[row_m]     += MB

    # ------------------------------------------------------------------
    # Row 2N-1: Downstream BC
    # ------------------------------------------------------------------
    row_ds = size - 1
    if ds_bc == 'stage':
        # Prescribe Δz_{N-1} = z_ds^{n+1} - z_n[N-1]
        # x[2(N-1)] = Δz_{N-1}; col = 2N-2; row-col = 1 → ab[3, 2N-2]
        ab[3, size - 2] = 1.0
        rhs[row_ds] = ds_bc_value - z_n[N - 1]
    else:
        # Normal depth: Q^{n+1} = K * sqrt(S0)
        # Linearised: ΔQ = (dQ/dz)*Δz + [Q_nd(z^n) - Q^n]
        # → 1*ΔQ_{N-1} - (dQ/dz)*Δz_{N-1} = Q_nd - Q^n
        # x[2N-1] = ΔQ_{N-1} (diagonal); x[2N-2] = Δz_{N-1}
        S0 = max(ds_bc_value, 1e-8)
        xs_ds = sections_us_to_ds[N - 1]
        table_ds = hydraulic_tables.get(id(xs_ds)) if hydraulic_tables else None
        Q_nd  = _normal_depth_Q(xs_ds, S0, z_n[N - 1], hydraulic_table=table_ds)
        dQdz  = _dQ_dz_normal(xs_ds, S0, z_n[N - 1], hydraulic_table=table_ds)
        ab[2, size - 1]  =  1.0          # diagonal (ΔQ_{N-1})
        ab[3, size - 2] -= dQdz          # below diagonal (Δz_{N-1})
        rhs[row_ds] = Q_nd - Q_n[N - 1]

    return ab, rhs


def _solve_banded(ab: Any, rhs: Any) -> Any:
    """Solve the pentadiagonal system.  Falls back to numpy if scipy absent."""
    if _HAVE_SCIPY:
        return _scipy_solve_banded((2, 2), ab, rhs)
    # Fallback: reconstruct full matrix from banded storage and use numpy
    n = len(rhs)
    A_full = np.zeros((n, n))
    for diag_offset in range(-2, 3):
        k = diag_offset           # offset from diagonal
        row_start = max(0, -k)
        col_start = max(0, k)
        length = n - abs(k)
        ab_row = 2 - k            # ab[2 + i - j, j] where i-j = k → ab[2-k, j]
        for i in range(length):
            A_full[row_start + i, col_start + i] = ab[ab_row, col_start + i]
    return np.linalg.solve(A_full, rhs)


def _capture_first_step_debug(
    sections_us_to_ds: List[CrossSection],
    dx: List[float],
    z_state: Any,
    Q_state: Any,
    dt: float,
    theta: float,
    Q_upstream_next: float,
    ds_bc: str,
    ds_bc_value: float,
    ab: Any,
    rhs_vec: Any,
    delta: Any,
    step: int,
    inner_iter: int,
    t_new: float,
    hydraulic_tables: Optional[Dict[int, SectionHydraulicTable]] = None,
) -> dict:
    """Return a JSON-serializable snapshot for the first-step upstream rows."""
    snapshot = {
        'step': int(step),
        'inner_iteration': int(inner_iter),
        'time_seconds': float(t_new),
        'theta': float(theta),
        'dt': float(dt),
        'section_ids': [str(xs.river_station) for xs in sections_us_to_ds],
        'dx': [float(val) for val in dx],
        'boundary': {
            'upstream_q_target': float(Q_upstream_next),
            'upstream_q_current': float(Q_state[0]),
            'downstream_bc': str(ds_bc),
            'downstream_value': float(ds_bc_value),
        },
        'state_before': {
            'z': [float(val) for val in z_state],
            'Q': [float(val) for val in Q_state],
        },
        'delta': {
            'dz': [float(val) for val in delta[0::2]],
            'dQ': [float(val) for val in delta[1::2]],
        },
        'matrix_rows': {
            'row_0_upstream_bc': {
                'rhs': float(rhs_vec[0]),
                'entries': {
                    'delta_z_0': float(ab[2, 0]),
                    'delta_Q_0': float(ab[1, 1]),
                    'delta_z_1': float(ab[0, 2]) if ab.shape[1] > 2 else 0.0,
                },
            },
        },
    }

    if len(sections_us_to_ds) < 2:
        return snapshot

    xs_0 = sections_us_to_ds[0]
    xs_1 = sections_us_to_ds[1]

    z_0 = float(z_state[0])
    z_1 = float(z_state[1])
    Q_0 = float(Q_state[0])
    Q_1 = float(Q_state[1])

    table_0 = hydraulic_tables.get(id(xs_0)) if hydraulic_tables else None
    table_1 = hydraulic_tables.get(id(xs_1)) if hydraulic_tables else None
    state_0 = _unsteady_section_state(xs_0, z_0, Q_0, hydraulic_table=table_0)
    state_1 = _unsteady_section_state(xs_1, z_1, Q_1, hydraulic_table=table_1)
    L = _effective_reach_length(xs_1, state_0, state_1, float(dx[0]) if dx else 1.0)

    A_0, K_0, T_0 = state_0.A_t, state_0.K_t, state_0.T_t
    A_1, K_1, T_1 = state_1.A_t, state_1.K_t, state_1.T_t
    Sf_0 = _Sf(Q_0, K_0)
    Sf_1 = _Sf(Q_1, K_1)
    Sf_avg = 0.5 * (Sf_0 + Sf_1)
    Abar = 0.5 * (A_0 + A_1)
    V_0 = state_0.V_t
    V_1 = state_1.V_t
    alpha_0 = state_0.alpha
    alpha_1 = state_1.alpha
    dSf_dQ_0 = _dSf_dQ(Q_0, K_0)
    dSf_dQ_1 = _dSf_dQ(Q_1, K_1)
    dKdz_0 = _dK_dz(xs_0, z_0, hydraulic_table=table_0)
    dKdz_1 = _dK_dz(xs_1, z_1, hydraulic_table=table_1)
    dSf_dz_0 = _dSf_dz(Q_0, K_0, dKdz_0)
    dSf_dz_1 = _dSf_dz(Q_1, K_1, dKdz_1)

    snapshot['first_reach'] = {
        'upstream_section_id': str(xs_0.river_station),
        'downstream_section_id': str(xs_1.river_station),
        'length': float(L),
        'section_0': {
            'z': z_0,
            'Q': Q_0,
            'A': float(A_0),
            'K': float(K_0),
            'T': float(T_0),
            'V': float(V_0),
            'alpha': float(alpha_0),
            'Sf': float(Sf_0),
            'dSf_dQ': float(dSf_dQ_0),
            'dSf_dz': float(dSf_dz_0),
            'bed_min': float(min(p[1] for p in xs_0.geometry)),
            'left_activation_elev': float(state_0.left_activation_elev),
            'right_activation_elev': float(state_0.right_activation_elev),
            'left_activation_factor': float(state_0.left_activation_factor),
            'right_activation_factor': float(state_0.right_activation_factor),
            'Q_lob': float(state_0.Q_lob),
            'Q_ch': float(state_0.Q_ch),
            'Q_rob': float(state_0.Q_rob),
        },
        'section_1': {
            'z': z_1,
            'Q': Q_1,
            'A': float(A_1),
            'K': float(K_1),
            'T': float(T_1),
            'V': float(V_1),
            'alpha': float(alpha_1),
            'Sf': float(Sf_1),
            'dSf_dQ': float(dSf_dQ_1),
            'dSf_dz': float(dSf_dz_1),
            'bed_min': float(min(p[1] for p in xs_1.geometry)),
            'left_activation_elev': float(state_1.left_activation_elev),
            'right_activation_elev': float(state_1.right_activation_elev),
            'left_activation_factor': float(state_1.left_activation_factor),
            'right_activation_factor': float(state_1.right_activation_factor),
            'Q_lob': float(state_1.Q_lob),
            'Q_ch': float(state_1.Q_ch),
            'Q_rob': float(state_1.Q_rob),
        },
        'averages': {
            'Abar': float(Abar),
            'Sf_avg': float(Sf_avg),
            'effective_length': float(L),
        },
        'continuity_row': {
            'coefficients': {
                'delta_z_0': float(T_0 / (2.0 * dt)),
                'delta_Q_0': float(-theta / L),
                'delta_z_1': float(T_1 / (2.0 * dt)),
                'delta_Q_1': float(theta / L),
            },
            'rhs': float(-(Q_1 - Q_0) / L),
        },
        'momentum_row': {
            'coefficients': {
                'delta_z_0': float(theta * G * Abar * 0.5 * dSf_dz_0 - G * Abar * theta / L),
                'delta_Q_0': float((1.0 / (2.0 * dt)) - theta * alpha_0 * V_0 / L + theta * G * Abar * 0.5 * dSf_dQ_0),
                'delta_z_1': float(theta * G * Abar * 0.5 * dSf_dz_1 + G * Abar * theta / L),
                'delta_Q_1': float((1.0 / (2.0 * dt)) + theta * alpha_1 * V_1 / L + theta * G * Abar * 0.5 * dSf_dQ_1),
            },
            'rhs': float(-(
                (alpha_1 * Q_1 * V_1 - alpha_0 * Q_0 * V_0) / L
                + G * Abar * (z_1 - z_0) / L
                + G * Abar * Sf_avg
            )),
        },
        'assembled_rows': {
            'row_1_continuity': {
                'rhs': float(rhs_vec[1]) if len(rhs_vec) > 1 else 0.0,
                'entries': {
                    'delta_z_0': float(ab[3, 0]),
                    'delta_Q_0': float(ab[2, 1]),
                    'delta_z_1': float(ab[1, 2]),
                    'delta_Q_1': float(ab[0, 3]),
                },
            },
            'row_2_momentum': {
                'rhs': float(rhs_vec[2]) if len(rhs_vec) > 2 else 0.0,
                'entries': {
                    'delta_z_0': float(ab[4, 0]),
                    'delta_Q_0': float(ab[3, 1]),
                    'delta_z_1': float(ab[2, 2]),
                    'delta_Q_1': float(ab[1, 3]),
                },
            },
        },
    }
    return snapshot


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------

def _initial_conditions(
    model: ModelInput,
    Q_initial: float,
    ds_bc: str,
    ds_bc_value: float,
) -> Tuple[List[CrossSection], Any, Any]:
    """Compute steady-state initial conditions using the existing backwater solver.

    Returns
    -------
    sections_us_to_ds : list
        Sections in upstream→downstream order.
    z0 : np.ndarray
        Initial WSE at each node (upstream→downstream).
    Q0 : np.ndarray
        Initial discharge at each node (upstream→downstream).
    """
    import copy
    steady_model = copy.deepcopy(model)
    steady_model.flow_cfs = Q_initial

    # Map DS BC type to steady-state convention
    if ds_bc == 'stage':
        steady_model.boundary_condition = 'known_wse'
        steady_model.boundary_value = ds_bc_value
    else:
        steady_model.boundary_condition = 'normal_depth'
        steady_model.boundary_value = ds_bc_value

    # run_backwater orders DS=0, US=N-1
    steady_results = run_backwater(steady_model, solver='py')

    # sections are ordered DS→US in model after run_backwater
    ordered = _sorted_sections_by_river_station(model.sections)
    sections_us_to_ds = list(reversed(ordered))      # flip to US→DS
    N = len(sections_us_to_ds)

    z0 = np.zeros(N)
    Q0 = np.full(N, Q_initial)

    # steady_results[0] = DS section, steady_results[N-1] = US section
    # sections_us_to_ds[j] = ordered[N-1-j]
    for j in range(N):
        idx_in_results = N - 1 - j          # result index for this node
        if idx_in_results < len(steady_results):
            z0[j] = steady_results[idx_in_results].wse
        else:
            z0[j] = min(p[1] for p in sections_us_to_ds[j].geometry) + 0.5

    return sections_us_to_ds, z0, Q0


# ---------------------------------------------------------------------------
# Main unsteady solver
# ---------------------------------------------------------------------------

def run_unsteady(
    model: ModelInput,
    upstream_hydrograph: HydrographBC,
    params: UnsteadyParams,
    progress_callback=None,
) -> UnsteadyResults:
    """Run 1D unsteady (dynamic wave) simulation using the Preissmann scheme.

    Parameters
    ----------
    model : ModelInput
        Cross-section geometry and roughness.  Sections may be in any order;
        they are sorted by river station internally.
    upstream_hydrograph : HydrographBC
        Upstream boundary condition (``bc_type='flow'``).
    params : UnsteadyParams
        Solver control parameters (dt, t_end, theta, etc.).
    progress_callback : callable, optional
        Called with ``(current_step, total_steps, message)`` for progress
        reporting from the GUI.  May be ``None``.

    Returns
    -------
    UnsteadyResults

    Raises
    ------
    RuntimeError
        If numpy is not available or the system cannot be solved.
    ValueError
        If input data are insufficient (< 2 sections, zero reach lengths, etc.).
    """
    if not _HAVE_NUMPY:
        raise RuntimeError(
            "NumPy is required for the unsteady solver.  "
            "Install it in the QGIS Python environment."
        )

    dt     = float(params.dt)
    t_end  = float(params.t_end)
    theta  = float(max(0.5, min(1.0, params.theta)))
    n_total_steps = max(1, int(round(t_end / dt)))
    output_interval = max(1, int(params.output_interval))
    max_iter = max(1, int(params.max_iter))
    tol = float(params.tol)
    debug_capture = bool(params.debug_capture)
    debug_frequency = str(params.debug_frequency or 'output').strip().lower()
    if debug_frequency not in ('output', 'computation'):
        debug_frequency = 'output'

    if len(model.sections) < 2:
        raise ValueError("At least two cross sections are required for unsteady routing.")

    # Initial flow (first value of upstream hydrograph)
    Q_init = upstream_hydrograph.interpolate(0.0)
    if Q_init <= 0.0:
        Q_init = max(1.0, upstream_hydrograph.values[0] if upstream_hydrograph.values else 1.0)

    # DS BC
    ds_bc       = params.downstream_bc
    ds_bc_value = params.downstream_value
    ds_hydro    = params.downstream_hydrograph

    # Initial conditions from steady-state run
    sections_us_to_ds, z_n, Q_n = _initial_conditions(model, Q_init, ds_bc, ds_bc_value)
    N = len(sections_us_to_ds)

    # Reach lengths between consecutive nodes (upstream → downstream)
    # dx[r] = distance from node r to node r+1 (going downstream)
    # L_ch_to_next on a section = distance from that section to its upstream neighbor.
    # sections_us_to_ds[r+1] corresponds to sections_ds_to_us[N-2-r],
    # and its L_ch_to_next is the distance from sections_ds_to_us[N-2-r]
    # to sections_ds_to_us[N-1-r] = sections_us_to_ds[r].
    dx = []
    ordered_ds_to_us = list(reversed(sections_us_to_ds))  # DS=0
    for r in range(N - 1):
        # node r = ordered_ds_to_us[N-1-r], node r+1 = ordered_ds_to_us[N-2-r]
        ds_section_of_reach = ordered_ds_to_us[N - 2 - r]
        L = ds_section_of_reach.L_ch_to_next
        if L <= 0.0:
            # fallback: warn and use a small default
            L = 100.0
            import warnings
            warnings.warn(
                f"Reach length L_ch_to_next is 0 for section "
                f"'{ds_section_of_reach.river_station}'.  Using 100 ft as fallback.",
                UserWarning,
                stacklevel=3,
            )
        dx.append(L)

    # Pre-compute section IDs (US→DS)
    section_ids = [str(xs.river_station) for xs in sections_us_to_ds]

    # Enforce a minimum water depth at each node to avoid dry-bed singularities
    MIN_DEPTH = WETTING_DEPTH_FT
    for i, xs in enumerate(sections_us_to_ds):
        z_n[i] = _regularized_wse(xs, z_n[i], MIN_DEPTH)

    hydraulic_tables = None
    if params.precompute_hydraulic_tables:
        hydraulic_tables = _build_hydraulic_tables(
            sections_us_to_ds,
            dz=float(params.hydraulic_table_dz),
            padding=float(params.hydraulic_table_padding),
        )

    # Storage for output
    n_output = (n_total_steps // output_interval) + 1
    wse_out  = np.empty((n_output, N), dtype=np.float64)
    q_out    = np.empty((n_output, N), dtype=np.float64)
    times_out = np.empty(n_output, dtype=np.float64)
    max_wse   = np.array(z_n, dtype=np.float64)

    # Store t=0 initial state
    wse_out[0]  = z_n
    q_out[0]    = Q_n
    times_out[0] = 0.0
    out_idx = 1

    run_dt = datetime.now(timezone.utc)
    run_id = run_dt.strftime('%Y%m%dT%H%M%SZ')
    debug_payload = None
    if params.debug_output_path:
        debug_payload = {
            'run_id': run_id,
            'created_utc': run_dt.strftime('%Y-%m-%d %H:%M:%S UTC'),
            'section_ids': list(section_ids),
            'dx': [float(val) for val in dx],
            'records': [],
        }

    # -----------------------------------------------------------------------
    # Time integration
    # -----------------------------------------------------------------------
    try:
        for step in range(1, n_total_steps + 1):
            t_new = step * dt

            # Boundary values at t_new
            Q_us_next = upstream_hydrograph.interpolate(t_new)
            if ds_bc == 'stage' and ds_hydro is not None:
                ds_value_next = ds_hydro.interpolate(t_new)
            else:
                ds_value_next = ds_bc_value   # S0 (unchanged for normal depth)

            # Working copies (updated in inner iterations)
            z_iter = np.array(z_n)
            Q_iter = np.array(Q_n)
            inner_debug_stats = []

            # Inner Newton-like iterations to handle nonlinear Sf
            for _inner in range(max_iter):
                ab, rhs_vec = _assemble_system(
                    sections_us_to_ds, dx, z_iter, Q_iter, dt, theta,
                    Q_us_next, ds_bc, ds_value_next,
                    hydraulic_tables=hydraulic_tables,
                )
                try:
                    delta = _solve_banded(ab, rhs_vec)
                except Exception as exc:
                    raise RuntimeError(
                        f"Linear system solve failed at t={t_new:.1f} s (step {step}): {exc}"
                    ) from exc

                if debug_payload is not None and step == 1:
                    debug_payload['records'].append(_capture_first_step_debug(
                        sections_us_to_ds=sections_us_to_ds,
                        dx=dx,
                        z_state=z_iter,
                        Q_state=Q_iter,
                        dt=dt,
                        theta=theta,
                        Q_upstream_next=Q_us_next,
                        ds_bc=ds_bc,
                        ds_bc_value=ds_value_next,
                        ab=ab,
                        rhs_vec=rhs_vec,
                        delta=delta,
                        step=step,
                        inner_iter=_inner + 1,
                        t_new=t_new,
                        hydraulic_tables=hydraulic_tables,
                    ))

                dz_raw = delta[0::2]   # Δz at each node
                dQ_raw = delta[1::2]   # ΔQ at each node
                dz, dQ, damping = _apply_adaptive_damping(
                    sections_us_to_ds, z_iter, Q_iter, dz_raw, dQ_raw
                )
                if debug_capture:
                    inner_debug_stats.append({
                        'inner_iter': int(_inner + 1),
                        'max_abs_dz_raw': float(np.max(np.abs(dz_raw))),
                        'max_abs_dQ_raw': float(np.max(np.abs(dQ_raw))),
                        'max_abs_dz_applied': float(np.max(np.abs(dz))),
                        'max_abs_dQ_applied': float(np.max(np.abs(dQ))),
                        'linear_rhs_inf': float(np.max(np.abs(rhs_vec))),
                        'linear_residual_inf': _linear_system_residual_inf(ab, delta, rhs_vec),
                        'damping_factor': float(damping),
                    })

                if debug_payload is not None and step == 1 and debug_payload['records']:
                    debug_payload['records'][-1]['damping_factor'] = damping
                    debug_payload['records'][-1]['delta_raw'] = {
                        'dz': [float(val) for val in dz_raw],
                        'dQ': [float(val) for val in dQ_raw],
                    }

                z_iter = z_iter + dz
                Q_iter = Q_iter + dQ

                # Enforce minimum depth
                for i, xs in enumerate(sections_us_to_ds):
                    z_iter[i] = _regularized_wse(xs, z_iter[i], MIN_DEPTH)

                if np.max(np.abs(dz)) < tol and np.max(np.abs(dQ)) < tol:
                    break

            z_n = z_iter
            Q_n = Q_iter

            # Update max WSE
            np.maximum(max_wse, z_n, out=max_wse)

            # Store output
            if step % output_interval == 0 and out_idx < n_output:
                wse_out[out_idx]   = z_n
                q_out[out_idx]     = Q_n
                times_out[out_idx] = t_new
                out_idx += 1

            if debug_capture and (
                debug_frequency == 'computation' or step % output_interval == 0
            ):
                if debug_payload is None:
                    debug_payload = {
                        'run_id': run_id,
                        'created_utc': run_dt.strftime('%Y-%m-%d %H:%M:%S UTC'),
                        'section_ids': list(section_ids),
                        'dx': [float(val) for val in dx],
                        'records': [],
                    }
                debug_payload['records'].append(_capture_step_debug(
                    sections_us_to_ds=sections_us_to_ds,
                    dx=dx,
                    z_state=z_n,
                    q_state=Q_n,
                    step=step,
                    t_new=t_new,
                    output_step=(step % output_interval == 0),
                        hydraulic_tables=hydraulic_tables,
                    inner_stats=inner_debug_stats,
                ))

            if progress_callback is not None:
                progress_callback(step, n_total_steps, f"t = {t_new:.0f} s")
    finally:
        if debug_payload is not None and params.debug_output_path:
            with open(params.debug_output_path, 'w', encoding='utf-8') as fh:
                json.dump(debug_payload, fh, indent=2)

    # Trim output arrays to actual written size
    wse_out   = wse_out[:out_idx]
    q_out     = q_out[:out_idx]
    times_out = times_out[:out_idx]

    return UnsteadyResults(
        times=times_out,
        wse=wse_out,
        q=q_out,
        max_wse=max_wse,
        section_ids=section_ids,
        run_id=run_id,
        run_time=run_dt.strftime('%Y-%m-%d %H:%M:%S UTC'),
        dt=dt,
        n_sections=N,
        n_output_times=len(times_out),
        debug_records=(debug_payload.get('records', []) if debug_payload is not None else None),
    )


# ---------------------------------------------------------------------------
# Binary GeoPackage I/O for unsteady results
# ---------------------------------------------------------------------------

_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS unsteady_results (
    run_id        TEXT    PRIMARY KEY,
    run_time      TEXT    NOT NULL,
    n_sections    INTEGER NOT NULL,
    n_output_times INTEGER NOT NULL,
    dt_s          REAL    NOT NULL,
    t_end_s       REAL    NOT NULL,
    section_ids   TEXT    NOT NULL,
    times_blob    BLOB    NOT NULL,
    wse_blob      BLOB    NOT NULL,
    q_blob        BLOB    NOT NULL,
    max_wse_blob  BLOB    NOT NULL,
    metadata      TEXT
)
"""

_TABLE_HYDRO_DDL = """
CREATE TABLE IF NOT EXISTS unsteady_hydrographs (
    hydrograph_id TEXT PRIMARY KEY,
    bc_type       TEXT NOT NULL,
    label         TEXT,
    data_json     TEXT NOT NULL
)
"""

_TABLE_DEBUG_DDL = """
CREATE TABLE IF NOT EXISTS unsteady_debug_steps (
    run_id        TEXT    NOT NULL,
    step_idx      INTEGER NOT NULL,
    time_s        REAL    NOT NULL,
    record_kind   TEXT    NOT NULL,
    payload_blob  BLOB    NOT NULL,
    PRIMARY KEY (run_id, step_idx, record_kind)
)
"""


def save_unsteady_results_to_geopackage(
    path: str,
    results: UnsteadyResults,
) -> str:
    """Persist unsteady results as binary blobs in the GeoPackage.

    The arrays ``wse``, ``q``, ``max_wse``, and ``times`` are stored as
    raw float64 byte blobs for efficient storage and fast I/O.

    Parameters
    ----------
    path : str
        Path to the GeoPackage (SQLite) file.
    results : UnsteadyResults

    Returns
    -------
    str
        The ``run_id`` under which the results were stored.
    """
    if not _HAVE_NUMPY:
        raise RuntimeError("NumPy is required to save binary results.")

    conn = sqlite3.connect(path)
    try:
        conn.execute(_TABLE_DDL)
        conn.execute("PRAGMA journal_mode=WAL")

        run_id  = results.run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        t_end   = float(results.times[-1]) if len(results.times) > 0 else 0.0

        metadata = json.dumps({
            'dt_s':  results.dt,
            'n_sections':  results.n_sections,
            'n_output_times': results.n_output_times,
        })

        conn.execute(
            """INSERT OR REPLACE INTO unsteady_results
               (run_id, run_time, n_sections, n_output_times, dt_s, t_end_s,
                section_ids, times_blob, wse_blob, q_blob, max_wse_blob, metadata)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                run_id,
                results.run_time,
                int(results.n_sections),
                int(results.n_output_times),
                float(results.dt),
                t_end,
                json.dumps(results.section_ids),
                sqlite3.Binary(results.times.astype(np.float64).tobytes()),
                sqlite3.Binary(results.wse.astype(np.float64).tobytes()),
                sqlite3.Binary(results.q.astype(np.float64).tobytes()),
                sqlite3.Binary(results.max_wse.astype(np.float64).tobytes()),
                metadata,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    return run_id


def save_unsteady_debug_to_geopackage(
    path: str,
    run_id: str,
    records: List[dict],
    record_kind: str = 'output',
) -> int:
    """Persist detailed unsteady debug records as binary payloads."""
    if not records:
        return 0
    conn = sqlite3.connect(path)
    try:
        conn.execute(_TABLE_DEBUG_DDL)
        conn.execute("PRAGMA journal_mode=WAL")
        rows = []
        kind = str(record_kind or 'output')
        for rec in records:
            step_idx = int(rec.get('step', 0))
            time_s = float(rec.get('time_s', 0.0))
            payload = sqlite3.Binary(pickle.dumps(rec, protocol=pickle.HIGHEST_PROTOCOL))
            rows.append((run_id, step_idx, time_s, kind, payload))
        conn.executemany(
            "INSERT OR REPLACE INTO unsteady_debug_steps "
            "(run_id, step_idx, time_s, record_kind, payload_blob) VALUES (?,?,?,?,?)",
            rows,
        )
        conn.commit()
    finally:
        conn.close()
    return len(records)


def load_unsteady_debug_from_geopackage(
    path: str,
    run_id: str,
    record_kind: Optional[str] = None,
) -> List[dict]:
    """Load detailed unsteady debug records from binary storage."""
    if not os.path.isfile(path):
        return []
    conn = sqlite3.connect(path)
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='unsteady_debug_steps'"
        )
        if cur.fetchone() is None:
            return []
        if record_kind:
            rows = conn.execute(
                "SELECT payload_blob FROM unsteady_debug_steps "
                "WHERE run_id=? AND record_kind=? ORDER BY step_idx",
                (run_id, str(record_kind)),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT payload_blob FROM unsteady_debug_steps "
                "WHERE run_id=? ORDER BY step_idx",
                (run_id,),
            ).fetchall()
    finally:
        conn.close()
    out = []
    for (blob,) in rows:
        try:
            out.append(pickle.loads(bytes(blob)))
        except Exception:
            continue
    return out


def load_unsteady_results_from_geopackage(
    path: str,
    run_id: Optional[str] = None,
) -> Optional[UnsteadyResults]:
    """Load the most recent (or specified) unsteady results from a GeoPackage.

    Parameters
    ----------
    path : str
        Path to the GeoPackage file.
    run_id : str, optional
        Specific run to load; if ``None`` the most recent run is loaded.

    Returns
    -------
    UnsteadyResults or None
    """
    if not _HAVE_NUMPY:
        raise RuntimeError("NumPy is required to load binary results.")
    if not os.path.isfile(path):
        return None

    conn = sqlite3.connect(path)
    try:
        # Check table exists
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='unsteady_results'"
        )
        if cur.fetchone() is None:
            return None

        if run_id is not None:
            row = conn.execute(
                "SELECT * FROM unsteady_results WHERE run_id=?", (run_id,)
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM unsteady_results ORDER BY run_time DESC LIMIT 1"
            ).fetchone()

        if row is None:
            return None

        (run_id_, run_time, n_sections, n_output_times, dt_s, t_end_s,
         section_ids_json, times_blob, wse_blob, q_blob, max_wse_blob, metadata) = row

        section_ids = json.loads(section_ids_json)
        times   = np.frombuffer(bytes(times_blob),   dtype=np.float64).copy()
        wse     = np.frombuffer(bytes(wse_blob),     dtype=np.float64).reshape(n_output_times, n_sections).copy()
        q       = np.frombuffer(bytes(q_blob),       dtype=np.float64).reshape(n_output_times, n_sections).copy()
        max_wse = np.frombuffer(bytes(max_wse_blob), dtype=np.float64).copy()

    finally:
        conn.close()

    return UnsteadyResults(
        times=times,
        wse=wse,
        q=q,
        max_wse=max_wse,
        section_ids=section_ids,
        run_id=run_id_,
        run_time=run_time,
        dt=dt_s,
        n_sections=n_sections,
        n_output_times=n_output_times,
    )


def list_unsteady_runs(path: str) -> List[dict]:
    """Return a list of available unsteady runs in the GeoPackage.

    Returns
    -------
    list of dict with keys ``run_id``, ``run_time``, ``t_end_s``,
    ``n_sections``, ``n_output_times``, ``dt_s``.
    """
    if not os.path.isfile(path):
        return []
    conn = sqlite3.connect(path)
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='unsteady_results'"
        )
        if cur.fetchone() is None:
            return []
        rows = conn.execute(
            "SELECT run_id, run_time, n_sections, n_output_times, dt_s, t_end_s "
            "FROM unsteady_results ORDER BY run_time DESC"
        ).fetchall()
    finally:
        conn.close()

    return [
        {
            'run_id': r[0], 'run_time': r[1],
            'n_sections': r[2], 'n_output_times': r[3],
            'dt_s': r[4], 't_end_s': r[5],
        }
        for r in rows
    ]


# Alias used by backwater_qt.py
list_unsteady_runs_in_geopackage = list_unsteady_runs


# ---------------------------------------------------------------------------
# Hydrograph boundary condition GeoPackage I/O
# ---------------------------------------------------------------------------

def save_hydrograph_to_geopackage(
    path: str,
    hydro: HydrographBC,
    hydrograph_id: Optional[str] = None,
) -> str:
    """Store a ``HydrographBC`` in the GeoPackage as JSON.

    Returns the *hydrograph_id* used.
    """
    conn = sqlite3.connect(path)
    try:
        conn.execute(_TABLE_HYDRO_DDL)
        hid = hydrograph_id or (
            f"{hydro.bc_type}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        )
        data = json.dumps({'times': hydro.times, 'values': hydro.values})
        conn.execute(
            "INSERT OR REPLACE INTO unsteady_hydrographs "
            "(hydrograph_id, bc_type, label, data_json) VALUES (?,?,?,?)",
            (hid, hydro.bc_type, hydro.label, data),
        )
        conn.commit()
    finally:
        conn.close()
    return hid


def load_hydrograph_from_geopackage(
    path: str,
    hydrograph_id: str,
) -> Optional[HydrographBC]:
    """Load a ``HydrographBC`` from the GeoPackage by ID."""
    if not os.path.isfile(path):
        return None
    conn = sqlite3.connect(path)
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='unsteady_hydrographs'"
        )
        if cur.fetchone() is None:
            return None
        row = conn.execute(
            "SELECT bc_type, label, data_json FROM unsteady_hydrographs "
            "WHERE hydrograph_id=?", (hydrograph_id,)
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return None
    bc_type, label, data_json = row
    data = json.loads(data_json)
    return HydrographBC(
        times=data['times'], values=data['values'],
        bc_type=bc_type, label=label or '',
    )


def list_hydrographs_in_geopackage(path: str) -> List[dict]:
    """Return list of stored hydrographs (id, bc_type, label, n_points)."""
    if not os.path.isfile(path):
        return []
    conn = sqlite3.connect(path)
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='unsteady_hydrographs'"
        )
        if cur.fetchone() is None:
            return []
        rows = conn.execute(
            "SELECT hydrograph_id, bc_type, label, data_json "
            "FROM unsteady_hydrographs"
        ).fetchall()
    finally:
        conn.close()

    result = []
    for hid, bct, lbl, dj in rows:
        try:
            n = len(json.loads(dj).get('times', []))
        except Exception:
            n = 0
        result.append({'hydrograph_id': hid, 'bc_type': bct, 'label': lbl or '', 'n_points': n})
    return result
