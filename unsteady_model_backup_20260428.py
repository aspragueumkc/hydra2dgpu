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
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, List, Optional, Tuple

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
    from hydra_1d import (
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
    from .hydra_1d import (  # type: ignore
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


# ---------------------------------------------------------------------------
# Section hydraulic helpers (subsection K, T, A)
# ---------------------------------------------------------------------------

WETTING_DEPTH_FT = 1.0


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
    return max(float(z), _hydraulic_bed_elevation(xs) + max(0.001, float(min_depth)))

def _compute_total_K(xs: CrossSection, z: float) -> float:
    """Total Manning conveyance at WSE *z* using subsection roughness."""
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


def _section_vars(xs: CrossSection, z: float) -> Tuple[float, float, float]:
    """Return (A_total, K_total, T_top_width) at WSE *z*.

    *T* is the free-surface top width (≈ dA/dz), computed numerically.
    """
    z = _regularized_wse(xs, z)
    A, _P, T = submerged_trapezoids_area_perimeter(xs.geometry, z)
    K = _compute_total_K(xs, z)
    # If T came out zero (geometry quirk), estimate numerically
    if T <= 0.0:
        dz = 1e-3
        A_hi, _, _ = submerged_trapezoids_area_perimeter(xs.geometry, z + dz)
        A_lo, _, _ = submerged_trapezoids_area_perimeter(xs.geometry, max(z - dz, min(p[1] for p in xs.geometry) + 1e-6))
        T = max(0.01, (A_hi - A_lo) / (2.0 * dz))
    return A, K, T


def _dK_dz(xs: CrossSection, z: float, dz: float = 1e-3) -> float:
    """Numerical dK/dz at WSE *z*."""
    z = _regularized_wse(xs, z)
    z_min = _hydraulic_bed_elevation(xs) + 1e-6
    K_hi = _compute_total_K(xs, z + dz)
    K_lo = _compute_total_K(xs, max(z - dz, z_min))
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


def _normal_depth_Q(xs: CrossSection, S0: float, z: float) -> float:
    """Q from Manning normal-depth rating at elevation *z*."""
    z = _regularized_wse(xs, z)
    K = _compute_total_K(xs, z)
    if S0 <= 0.0 or K <= 0.0:
        return 0.0
    return K * math.sqrt(S0)


def _dQ_dz_normal(xs: CrossSection, S0: float, z: float, dz: float = 1e-3) -> float:
    """Numerical dQ/dz for normal-depth BC linearization."""
    return (_normal_depth_Q(xs, S0, z + dz) - _normal_depth_Q(xs, S0, z - dz)) / (2.0 * dz)


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
        L = dx[r]
        if L <= 0.0:
            L = 1.0  # avoid division by zero; will warn elsewhere

        z_r   = z_n[r];   Q_r   = Q_n[r]
        z_rp1 = z_n[r+1]; Q_rp1 = Q_n[r+1]

        A_r,   K_r,   T_r   = _section_vars(xs_r,   z_r)
        A_rp1, K_rp1, T_rp1 = _section_vars(xs_rp1, z_rp1)

        Sf_r   = _Sf(Q_r,   K_r)
        Sf_rp1 = _Sf(Q_rp1, K_rp1)
        Sf_avg = 0.5 * (Sf_r + Sf_rp1)
        Abar   = 0.5 * (A_r + A_rp1)

        V_r   = Q_r   / A_r   if A_r   > 0.0 else 0.0
        V_rp1 = Q_rp1 / A_rp1 if A_rp1 > 0.0 else 0.0

        dSf_dQ_r   = _dSf_dQ(Q_r,   K_r)
        dSf_dQ_rp1 = _dSf_dQ(Q_rp1, K_rp1)
        dKdz_r     = _dK_dz(xs_r,   z_r)
        dKdz_rp1   = _dK_dz(xs_rp1, z_rp1)
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
        MQ_r   = (1.0 / (2.0*dt)) - theta*V_r/L   + theta*G*Abar * 0.5*dSf_dQ_r
        MZ_r   = theta*G*Abar * 0.5*dSf_dz_r       - G*Abar*theta/L
        MQ_rp1 = (1.0 / (2.0*dt)) + theta*V_rp1/L + theta*G*Abar * 0.5*dSf_dQ_rp1
        MZ_rp1 = theta*G*Abar * 0.5*dSf_dz_rp1    + G*Abar*theta/L

        MB = -(
            (Q_rp1*V_rp1 - Q_r*V_r) / L
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
        Q_nd  = _normal_depth_Q(xs_ds, S0, z_n[N - 1])
        dQdz  = _dQ_dz_normal(xs_ds, S0, z_n[N - 1])
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
    L = float(dx[0]) if dx else 1.0
    if L <= 0.0:
        L = 1.0

    z_0 = float(z_state[0])
    z_1 = float(z_state[1])
    Q_0 = float(Q_state[0])
    Q_1 = float(Q_state[1])

    A_0, K_0, T_0 = _section_vars(xs_0, z_0)
    A_1, K_1, T_1 = _section_vars(xs_1, z_1)
    Sf_0 = _Sf(Q_0, K_0)
    Sf_1 = _Sf(Q_1, K_1)
    Sf_avg = 0.5 * (Sf_0 + Sf_1)
    Abar = 0.5 * (A_0 + A_1)
    V_0 = Q_0 / A_0 if A_0 > 0.0 else 0.0
    V_1 = Q_1 / A_1 if A_1 > 0.0 else 0.0
    dSf_dQ_0 = _dSf_dQ(Q_0, K_0)
    dSf_dQ_1 = _dSf_dQ(Q_1, K_1)
    dKdz_0 = _dK_dz(xs_0, z_0)
    dKdz_1 = _dK_dz(xs_1, z_1)
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
            'Sf': float(Sf_0),
            'dSf_dQ': float(dSf_dQ_0),
            'dSf_dz': float(dSf_dz_0),
            'bed_min': float(min(p[1] for p in xs_0.geometry)),
        },
        'section_1': {
            'z': z_1,
            'Q': Q_1,
            'A': float(A_1),
            'K': float(K_1),
            'T': float(T_1),
            'V': float(V_1),
            'Sf': float(Sf_1),
            'dSf_dQ': float(dSf_dQ_1),
            'dSf_dz': float(dSf_dz_1),
            'bed_min': float(min(p[1] for p in xs_1.geometry)),
        },
        'averages': {
            'Abar': float(Abar),
            'Sf_avg': float(Sf_avg),
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
                'delta_Q_0': float((1.0 / (2.0 * dt)) - theta * V_0 / L + theta * G * Abar * 0.5 * dSf_dQ_0),
                'delta_z_1': float(theta * G * Abar * 0.5 * dSf_dz_1 + G * Abar * theta / L),
                'delta_Q_1': float((1.0 / (2.0 * dt)) + theta * V_1 / L + theta * G * Abar * 0.5 * dSf_dQ_1),
            },
            'rhs': float(-(
                (Q_1 * V_1 - Q_0 * V_0) / L
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

            # Inner Newton-like iterations to handle nonlinear Sf
            for _inner in range(max_iter):
                ab, rhs_vec = _assemble_system(
                    sections_us_to_ds, dx, z_iter, Q_iter, dt, theta,
                    Q_us_next, ds_bc, ds_value_next,
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
                    ))

                dz = delta[0::2]   # Δz at each node
                dQ = delta[1::2]   # ΔQ at each node

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

            if progress_callback is not None:
                progress_callback(step, n_total_steps, f"t = {t_new:.0f} s")
    finally:
        if debug_payload is not None:
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
