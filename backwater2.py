
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-----------------------------------------------------------------------------
   backwater.py
   Author:   Aaron L. Sprague

   1D Steady Flow Water Surface Profiles (Backwater) using the standard-step
   method consistent with the HEC-RAS Hydraulic Reference Manual approach:

   - Cross-section subdivided into Left Overbank (LOB), Channel (CH), Right Overbank (ROB)
   - Manning conveyance-based flow distribution among subsections
   - Velocity-head coefficient (alpha) per subdivided section
   - Discharge-weighted reach length (DWRL)
   - Representative friction slope (Sf) and minor loss (expansion/contraction)
   - Marching from a downstream boundary (known WSE or normal-depth via S0)

   Units: US Customary (ft, cfs). g = 32.174 ft/s^2.

   Scope limits: single river reach, single steady flow, one optional flow-change
   location (lateral inflow/outflow). No bridges/culverts/junctions/supercritical checks.

   ---------------------------------------------------------------------------
    Usage (CLI):
        python backwater.py --input model.gpkg --ds-bc known_wse --ds-value 502.1

        or normal depth:
        python backwater.py --input model.gpkg --ds-bc normal_depth --ds-value 0.0005

    GeoPackage format with layers: cross_sections, centerline (optional), boundary_conditions.
-----------------------------------------------------------------------------
"""

import argparse
import importlib
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional, Dict

# Optional GUI / plotting
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    HAVE_TK = True
except Exception:
    HAVE_TK = False

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

# Optional numeric accelerators
try:
    import numpy as np
    HAVE_NUMPY = True
except Exception:
    HAVE_NUMPY = False

try:
    from scipy import optimize as _opt
    HAVE_SCIPY = True
except Exception:
    _opt = None
    HAVE_SCIPY = False

try:
    from shapely.geometry import LineString, Polygon
    from shapely.ops import split
    HAVE_SHAPELY = True
except Exception:
    HAVE_SHAPELY = False

# Optional culvert routine for inlet control
try:
    from culvert_routine import (
        CircularXsect, RectangularXsect,
        solve_headwater_depth_for_Q, inlet_controlled_flow,
        critical_depth_in_culvert,
        solve_normal_depth_in_culvert,
        direct_step_culvert_upstream_energy,
    )
    HAVE_CULVERT = True
except Exception:
    try:
        from .culvert_routine import (
            CircularXsect, RectangularXsect,
            solve_headwater_depth_for_Q, inlet_controlled_flow,
            critical_depth_in_culvert,
            solve_normal_depth_in_culvert,
            direct_step_culvert_upstream_energy,
        )
        HAVE_CULVERT = True
    except Exception:
        HAVE_CULVERT = False


def _ensure_culvert_runtime() -> bool:
    """Best-effort lazy import for culvert helpers in host environments."""
    global HAVE_CULVERT
    global CircularXsect, RectangularXsect
    global solve_headwater_depth_for_Q, inlet_controlled_flow
    global critical_depth_in_culvert, solve_normal_depth_in_culvert
    global direct_step_culvert_upstream_energy

    if HAVE_CULVERT:
        return True

    def _bind_from_module(_mod):
        CircularXsect_local = getattr(_mod, 'CircularXsect', None)
        RectangularXsect_local = getattr(_mod, 'RectangularXsect', None)
        solve_headwater_depth_for_Q_local = getattr(_mod, 'solve_headwater_depth_for_Q', None)
        inlet_controlled_flow_local = getattr(_mod, 'inlet_controlled_flow', None)
        critical_depth_in_culvert_local = getattr(_mod, 'critical_depth_in_culvert', None)
        solve_normal_depth_in_culvert_local = getattr(_mod, 'solve_normal_depth_in_culvert', None)
        direct_step_culvert_upstream_energy_local = getattr(_mod, 'direct_step_culvert_upstream_energy', None)
        if not all([
            CircularXsect_local,
            RectangularXsect_local,
            solve_headwater_depth_for_Q_local,
            inlet_controlled_flow_local,
            critical_depth_in_culvert_local,
            solve_normal_depth_in_culvert_local,
            direct_step_culvert_upstream_energy_local,
        ]):
            return False
        globals()['CircularXsect'] = CircularXsect_local
        globals()['RectangularXsect'] = RectangularXsect_local
        globals()['solve_headwater_depth_for_Q'] = solve_headwater_depth_for_Q_local
        globals()['inlet_controlled_flow'] = inlet_controlled_flow_local
        globals()['critical_depth_in_culvert'] = critical_depth_in_culvert_local
        globals()['solve_normal_depth_in_culvert'] = solve_normal_depth_in_culvert_local
        globals()['direct_step_culvert_upstream_energy'] = direct_step_culvert_upstream_energy_local
        globals()['HAVE_CULVERT'] = True
        return True

    try:
        from culvert_routine import (
            CircularXsect as _CircularXsect,
            RectangularXsect as _RectangularXsect,
            solve_headwater_depth_for_Q as _solve_headwater_depth_for_Q,
            inlet_controlled_flow as _inlet_controlled_flow,
            critical_depth_in_culvert as _critical_depth_in_culvert,
            solve_normal_depth_in_culvert as _solve_normal_depth_in_culvert,
            direct_step_culvert_upstream_energy as _direct_step_culvert_upstream_energy,
        )

        CircularXsect = _CircularXsect
        RectangularXsect = _RectangularXsect
        solve_headwater_depth_for_Q = _solve_headwater_depth_for_Q
        inlet_controlled_flow = _inlet_controlled_flow
        critical_depth_in_culvert = _critical_depth_in_culvert
        solve_normal_depth_in_culvert = _solve_normal_depth_in_culvert
        direct_step_culvert_upstream_energy = _direct_step_culvert_upstream_energy
        HAVE_CULVERT = True
        return True
    except Exception:
        pass

    # Package-relative import path (common in QGIS plugin package context)
    try:
        import importlib
        pkg = (__package__ or '').strip()
        if pkg:
            _mod = importlib.import_module(f"{pkg}.culvert_routine")
        else:
            _mod = importlib.import_module('culvert_routine')
        if _bind_from_module(_mod):
            return True
    except Exception:
        pass

    # Last resort: load from this file's directory explicitly.
    try:
        import importlib.util
        _path = os.path.join(os.path.dirname(__file__), 'culvert_routine.py')
        _spec = importlib.util.spec_from_file_location('qgis_backwater_plugin.culvert_routine', _path)
        if _spec is not None and _spec.loader is not None:
            _mod = importlib.util.module_from_spec(_spec)
            sys.modules['qgis_backwater_plugin.culvert_routine'] = _mod
            _spec.loader.exec_module(_mod)
            if _bind_from_module(_mod):
                return True
    except Exception:
        pass

    HAVE_CULVERT = False

    return HAVE_CULVERT

G = 32.174  # gravitational acceleration (ft/s^2)
MANNING_CONST = 1.49  # US customary

# Selection hooks (can be overridden via CLI)
ALPHA_METHOD = 'conveyance'  # options: 'conveyance' (default), 'area'
SF_METHOD = 'combined'       # options: 'combined' (default), 'avg'


# ---------------------------------------------------------------------------
# Geometry & hydraulics primitives
# ---------------------------------------------------------------------------

def lerp(x0: float, y0: float, x1: float, y1: float, x: float) -> float:
    """Linear interpolation of y at x between (x0,y0) and (x1,y1)."""
    if x1 == x0:
        return y0
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def clip_polyline_by_x(poly: List[Tuple[float, float]], x_min: float, x_max: float) -> List[Tuple[float, float]]:
    """
    Clip polyline by vertical lines x=x_min and x=x_max. Inserts boundary points by interpolation.
    Assumes 'poly' is sorted by station (x).
    """
    if not poly:
        return []

    clipped: List[Tuple[float, float]] = []
    for i in range(len(poly) - 1):
        x0, z0 = poly[i]
        x1, z1 = poly[i + 1]
        seg = [(x0, z0)]
        # Insert intersection points with x_min and x_max if segment crosses them
        xs = [x0, x1]
        zs = [z0, z1]

        # Sort segment endpoints (already sorted by input)
        xa, za = xs[0], zs[0]
        xb, zb = xs[1], zs[1]

        # Intersect with x_min
        if (xa < x_min < xb) or (xb < x_min < xa):
            zi = lerp(xa, za, xb, zb, x_min)
            seg.append((x_min, zi))
        # Intersect with x_max
        if (xa < x_max < xb) or (xb < x_max < xa):
            zi = lerp(xa, za, xb, zb, x_max)
            seg.append((x_max, zi))

        # Include far endpoint
        seg.append((x1, z1))
        # Keep only points with x within [x_min, x_max]
        seg = sorted(seg, key=lambda p: p[0])
        seg = [(x, z) for (x, z) in seg if (x_min - 1e-9) <= x <= (x_max + 1e-9)]

        # Skip empty segments (no intersection with clip range)
        if not seg:
            continue

        # Merge into clipped
        if not clipped:
            clipped.extend(seg)
        else:
            # Avoid duplicate join point
            if clipped[-1] == seg[0]:
                clipped.extend(seg[1:])
            else:
                clipped.extend(seg)

    # Deduplicate tiny repeats
    out: List[Tuple[float, float]] = []
    for x, z in clipped:
        if not out or abs(out[-1][0] - x) > 1e-9 or abs(out[-1][1] - z) > 1e-9:
            out.append((x, z))
    return out


def submerged_trapezoids_area_perimeter(poly: List[Tuple[float, float]], wse: float) -> Tuple[float, float, float]:
    """
    Computes flow area A, wetted perimeter P, and top width T for a polyline bed (x,z)
    submerged by a horizontal water surface at elevation 'wse'.
    """
    if len(poly) < 2:
        return 0.0, 0.0, 0.0

    # If Shapely is present, use it for robust clipping and geometry
    if HAVE_SHAPELY:
        try:
            line = LineString(poly)
            xs = [p[0] for p in poly]
            min_x = min(xs) if xs else 0.0
            max_x = max(xs) if xs else 0.0
            # Define a large box below the waterline to clip the line
            box = Polygon([(min_x - 1.0, -1e6), (max_x + 1.0, -1e6), (max_x + 1.0, wse), (min_x - 1.0, wse)])
            submerged = line.intersection(box)

            # wetted perimeter: length of submerged line parts
            P = 0.0
            coords_collections = []
            if submerged.is_empty:
                return 0.0, 0.0, 0.0
            if submerged.geom_type == 'LineString':
                P = submerged.length
                coords_collections = [list(submerged.coords)]
            else:
                # MultiLineString
                for geom in submerged.geoms:
                    if geom.geom_type == 'LineString':
                        P += geom.length
                        coords_collections.append(list(geom.coords))

            # compute submerged polygon area(s) by closing each submerged linestring to the waterline
            A = 0.0
            top_xs = []
            for coords in coords_collections:
                xs_seg = [c[0] for c in coords]
                if not xs_seg:
                    continue
                x0 = min(xs_seg)
                x1 = max(xs_seg)
                top_xs.append((x0, x1))
                # build polygon coords: submerged segment (left->right) + waterline back to left
                poly_coords = [(x, z) for (x, z) in coords]
                # ensure left-to-right order
                poly_coords = sorted(poly_coords, key=lambda p: p[0])
                # append waterline points to close polygon
                poly_coords.append((poly_coords[-1][0], wse))
                poly_coords.append((poly_coords[0][0], wse))
                poly_coords.append(poly_coords[0])
                try:
                    poly_area = Polygon(poly_coords).area
                except Exception:
                    poly_area = 0.0
                A += poly_area

            # compute top width as union length of top_xs spans
            if not top_xs:
                T = 0.0
            else:
                # merge intervals
                top_xs_sorted = sorted(top_xs, key=lambda t: t[0])
                merged = [list(top_xs_sorted[0])]
                for a, b in top_xs_sorted[1:]:
                    if a <= merged[-1][1]:
                        merged[-1][1] = max(merged[-1][1], b)
                    else:
                        merged.append([a, b])
                T = sum(seg[1] - seg[0] for seg in merged)

            return float(A), float(P), float(T)
        except Exception:
            # Fall back to polygon-free method on failure
            pass

    # Fallback: original trapezoidal clipping approach
    A = 0.0
    P = 0.0
    last_wet_x: Optional[float] = None
    top_width = 0.0

    for i in range(len(poly) - 1):
        x0, z0 = poly[i]
        x1, z1 = poly[i + 1]

        y0 = max(0.0, wse - z0)  # depth at x0
        y1 = max(0.0, wse - z1)  # depth at x1
        dx = x1 - x0
        dz = z1 - z0

        # Evaluate intersection with waterline
        both_submerged = (y0 > 0.0 and y1 > 0.0)
        one_submerged = (y0 > 0.0) ^ (y1 > 0.0)

        if both_submerged:
            # Trapezoid area
            A += 0.5 * (y0 + y1) * abs(dx)
            # Wetted perimeter along bed segment
            P += math.hypot(dx, dz)
            # Track top width endpoints
            if last_wet_x is None:
                last_wet_x = x0
            top_width = (x1 - last_wet_x) if last_wet_x is not None else top_width

        elif one_submerged:
            # Find intersection point where z = wse
            if z1 != z0:
                xi = x0 + (x1 - x0) * ((wse - z0) / (z1 - z0))
                zi = wse
            else:
                xi, zi = x0, z0  # horizontal segment at water level, degenerate

            if y0 > 0.0:
                # Submerged from x0 to xi
                y_at_xi = 0.0
                A += 0.5 * (y0 + y_at_xi) * abs(xi - x0)
                P += math.hypot(xi - x0, zi - z0)
                if last_wet_x is None:
                    last_wet_x = x0
                top_width = (xi - last_wet_x) if last_wet_x is not None else top_width
                last_wet_x = None
            else:
                # Submerged from xi to x1
                y_at_xi = 0.0
                A += 0.5 * (y_at_xi + y1) * abs(x1 - xi)
                P += math.hypot(x1 - xi, z1 - zi)
                last_wet_x = xi
                top_width = (x1 - last_wet_x)

        else:
            # Totally dry segment; if we were in wet before, close top width
            continue

    # Top width could be multi-valued for complex shapes; this is a simple estimate
    T = max(0.0, top_width)
    return A, P, T


@dataclass
class SubsectionResult:
    A: float = 0.0  # area
    P: float = 0.0  # wetted perimeter
    R: float = 0.0  # hydraulic radius
    K: float = 0.0  # conveyance
    Q: float = 0.0  # discharge in this subsection
    V: float = 0.0  # velocity


@dataclass
class CrossSection:
    river_station: str
    geometry: List[Tuple[float, float]]  # list of (station, elevation), sorted by station
    left_bank_station: float
    right_bank_station: float
    n_lob: float
    n_ch: float
    n_rob: float
    contraction_coeff: float = 0.1  # Cc (typical)
    expansion_coeff: float = 0.3    # Ce (typical)

    # Reach lengths to next upstream section (for this section index)
    # Reach lengths to next upstream section (for this section index).
    # NOTE: these are the downstream reach lengths used by HEC-RAS —
    # the distance from this (downstream) section to the next upstream section.
    # They are stored on the downstream section and used when marching upstream.
    L_lob_to_next: float = 0.0
    L_ch_to_next: float = 0.0
    L_rob_to_next: float = 0.0

    # Culvert properties (inlet control via FHWA HEC-5 equations)
    # culvert_code = 0 means no culvert; 1-57 are valid FHWA codes
    culvert_code: int = 0
    culvert_shape: Optional[str] = None  # 'circular' or 'rect'; None if no culvert
    culvert_diameter: float = 0.0  # for circular (ft)
    culvert_width: float = 0.0     # for rectangular (ft)
    culvert_height: float = 0.0    # for rectangular (ft)
    culvert_upstream_invert: float = 0.0   # culvert inlet invert elevation (ft)
    culvert_downstream_invert: float = 0.0 # culvert outlet invert elevation (ft)
    culvert_length: float = 0.0            # culvert barrel length (ft)
    culvert_weir_coeff: float = 3.0        # broad-crested weir coefficient (ft^0.5/s)
    culvert_weir_sta_left: float = 0.0   # left station limit for weir integration (ft); 0 = use full geometry
    culvert_weir_sta_right: float = 0.0  # right station limit for weir integration (ft); 0 = use full geometry

    def _subgeometry(self) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], List[Tuple[float, float]]]:
        xs_sorted = sorted(self.geometry, key=lambda p: p[0])
        lob = clip_polyline_by_x(xs_sorted, xs_sorted[0][0], self.left_bank_station)
        ch = clip_polyline_by_x(xs_sorted, self.left_bank_station, self.right_bank_station)
        rob = clip_polyline_by_x(xs_sorted, self.right_bank_station, xs_sorted[-1][0])
        return lob, ch, rob

    def has_culvert(self) -> bool:
        """Check if this section has a culvert defined."""
        return self.culvert_code > 0 and self.culvert_shape is not None

    def culvert_slope(self) -> float:
        """Compute culvert slope from invert elevations and length."""
        if self.culvert_length <= 0.0:
            return 0.0
        return max(0.0, (self.culvert_upstream_invert - self.culvert_downstream_invert) / self.culvert_length)

    def culvert_full_depth(self) -> float:
        """Return culvert full depth based on shape dimensions."""
        shape = (self.culvert_shape or '').strip().lower()
        if shape == 'circular':
            return max(0.0, float(self.culvert_diameter))
        if shape == 'rect':
            return max(0.0, float(self.culvert_height))
        return 0.0

    def culvert_weir_length(self) -> float:
        """Approximate overtopping weir crest length (ft)."""
        shape = (self.culvert_shape or '').strip().lower()
        if shape == 'circular':
            return max(0.0, float(self.culvert_diameter))
        if shape == 'rect':
            return max(0.0, float(self.culvert_width))
        return 0.0


    def hydraulics_at_wse(self, wse: float, Q_total: float) -> Dict[str, SubsectionResult]:
        """
        Compute hydraulics (A,P,R,K,alpha,V) for LOB/CH/ROB and totals at a given WSE and total Q.
        Flow distribution is proportional to conveyance.
        """
        lob_geom, ch_geom, rob_geom = self._subgeometry()

        # Areas and perimeters
        A_lob, P_lob, _ = submerged_trapezoids_area_perimeter(lob_geom, wse)
        A_ch, P_ch, _ = submerged_trapezoids_area_perimeter(ch_geom, wse)
        A_rob, P_rob, _ = submerged_trapezoids_area_perimeter(rob_geom, wse)

        def K_from_AP(A: float, P: float, n: float) -> float:
            if A <= 0.0 or P <= 0.0:
                return 0.0
            R = A / P
            return (MANNING_CONST / n) * A * (R ** (2.0 / 3.0))

        K_lob = K_from_AP(A_lob, P_lob, self.n_lob)
        K_ch = K_from_AP(A_ch, P_ch, self.n_ch)
        K_rob = K_from_AP(A_rob, P_rob, self.n_rob)

        Kt = K_lob + K_ch + K_rob
        At = A_lob + A_ch + A_rob

        # Distribute Q by conveyance
        if Kt <= 0.0 or At <= 0.0:
            # Dry or invalid section at this WSE
            return {
                "lob": SubsectionResult(A_lob, P_lob, 0.0, K_lob, 0.0, 0.0),
                "ch":  SubsectionResult(A_ch, P_ch,  0.0, K_ch,  0.0, 0.0),
                "rob": SubsectionResult(A_rob, P_rob, 0.0, K_rob, 0.0, 0.0),
                "totals": SubsectionResult(At, P_lob + P_ch + P_rob, 0.0, Kt, 0.0, 0.0),
                "alpha": 1.0
            }

        # Q distribution
        Q_lob = Q_total * (K_lob / Kt)
        Q_ch  = Q_total * (K_ch  / Kt)
        Q_rob = Q_total * (K_rob / Kt)

        # Velocities
        V_lob = (Q_lob / A_lob) if A_lob > 0 else 0.0
        V_ch  = (Q_ch  / A_ch)  if A_ch  > 0 else 0.0
        V_rob = (Q_rob / A_rob) if A_rob > 0 else 0.0

        # Hydraulic radii
        R_lob = (A_lob / P_lob) if P_lob > 0 else 0.0
        R_ch  = (A_ch  / P_ch)  if P_ch  > 0 else 0.0
        R_rob = (A_rob / P_rob) if P_rob > 0 else 0.0

        # Velocity-head coefficient alpha: support method selection
        if ALPHA_METHOD == 'area':
            # area-weighted kinetic energy alpha = Σ(A_i*V_i^2) / (At * Vt^2)
            V_t= Q_total / At if At > 0 else 0.0
            try:
                denom = At * (V_t ** 2) if At > 0 and V_t > 0 else 0.0
                if denom > 0.0:
                    num = 0.0
                    for A_i, V_i in [(A_lob, V_lob), (A_ch, V_ch), (A_rob, V_rob)]:
                        num += A_i * (V_i ** 2)
                    alpha = num / denom
                else:
                    alpha = 1.0
            except Exception:
                alpha = 1.0
        else:
            # default: conveyance-based alpha (original formula)
            alpha_num = 0.0
            for K_i, A_i in [(K_lob, A_lob), (K_ch, A_ch), (K_rob, A_rob)]:
                if K_i > 0.0 and A_i > 0.0:
                    alpha_num += (K_i ** 3) / (A_i ** 2)
            alpha = (At ** 2) * alpha_num / (Kt ** 3) if Kt > 0 and At > 0 else 1.0

        return {
            "lob": SubsectionResult(A_lob, P_lob, R_lob, K_lob, Q_lob, V_lob),
            "ch":  SubsectionResult(A_ch,  P_ch,  R_ch,  K_ch,  Q_ch,  V_ch),
            "rob": SubsectionResult(A_rob, P_rob, R_rob, K_rob, Q_rob, V_rob),
            "totals": SubsectionResult(
                A_lob + A_ch + A_rob,
                P_lob + P_ch + P_rob,
                (A_lob + A_ch + A_rob) / (P_lob + P_ch + P_rob) if (P_lob + P_ch + P_rob) > 0 else 0.0,
                Kt,
                Q_total,
                (Q_total / (A_lob + A_ch + A_rob)) if (A_lob + A_ch + A_rob) > 0 else 0.0
            ),
            "alpha": alpha
        }


# ---------------------------------------------------------------------------
# Energy equation & head loss
# ---------------------------------------------------------------------------

@dataclass
class ReachLink:
    """Properties between section i (downstream) and i+1 (upstream).

    The lengths passed to `ReachLink` are the downstream reach lengths
    (distance from the downstream section to the upstream section) as
    stored on the downstream `CrossSection` (fields `L_*_to_next`).
    This matches HEC-RAS convention when marching upstream.
    """
    L_lob: float
    L_ch: float
    L_rob: float

    def discharge_weighted_length(self, Qlob_av: float, Qch_av: float, Qrob_av: float) -> float:
        Qt_av = Qlob_av + Qch_av + Qrob_av
        if Qt_av <= 0.0:
            return 0.0
        return (self.L_lob * Qlob_av + self.L_ch * Qch_av + self.L_rob * Qrob_av) / Qt_av


@dataclass
class SectionState:
    """Computed hydraulic state at a section for a given WSE & total Q."""
    wse: float
    depth_at_min: float
    alpha: float
    A_lob: float; A_ch: float; A_rob: float
    K_lob: float; K_ch: float; K_rob: float
    Q_lob: float; Q_ch: float; Q_rob: float
    V_t: float
    K_t: float
    A_t: float
    Sf_total: float  # (Q_total / K_total)^2
    Froude: float = 0.0


def compute_state(xs: CrossSection, wse: float, Q_total: float) -> SectionState:
    h = xs.hydraulics_at_wse(wse, Q_total)
    res = h
    A_lob = res["lob"].A; A_ch = res["ch"].A; A_rob = res["rob"].A
    K_lob = res["lob"].K; K_ch = res["ch"].K; K_rob = res["rob"].K
    Q_lob = res["lob"].Q; Q_ch = res["ch"].Q; Q_rob = res["rob"].Q
    A_t = res["totals"].A
    K_t = res["totals"].K
    V_t = res["totals"].V
    alpha = res["alpha"]
    zmin = min(z for _, z in xs.geometry)
    depth_at_min = max(0.0, wse - zmin)
    Sf_total = (Q_total / K_t) ** 2 if K_t > 0 else 0.0
    # compute Froude number using hydraulic depth H = A / T when possible
    try:
        # hydraulics_at_wse returns top width as third value when called directly
        # Here res came from hydraulics_at_wse, so attempt to extract T from totals
        # The function returns totals in res['totals'] and top width was computed separately
        # We compute hydraulic depth H = A_t / T if T>0 else fall back to depth_at_min
        # If original hydraulics didn't include T explicitly, attempt to compute roughly as top width
        T = 0.0
        if isinstance(h, dict) and 'totals' in h:
            # we don't have explicit T in totals structure; attempt to recompute via submerged_trapezoids_area_perimeter
            try:
                # reconstruct approximate top width from geometry using current wse
                _, _, T = submerged_trapezoids_area_perimeter(xs.geometry, wse)
            except Exception:
                T = 0.0
        H = (A_t / T) if (T and T > 1e-12) else depth_at_min
        if H > 0.0:
            froude = V_t / math.sqrt(G * H)
        else:
            froude = 0.0
    except Exception:
        froude = 0.0

    return SectionState(
        wse=wse,
        depth_at_min=depth_at_min,
        alpha=alpha,
        A_lob=A_lob, A_ch=A_ch, A_rob=A_rob,
        K_lob=K_lob, K_ch=K_ch, K_rob=K_rob,
        Q_lob=Q_lob, Q_ch=Q_ch, Q_rob=Q_rob,
        V_t=V_t, K_t=K_t, A_t=A_t, Sf_total=Sf_total, Froude=froude
    )


def representative_friction_slope_total(s1: SectionState, s2: SectionState) -> float:
    """Representative (average) friction slope between two sections using total K, Q."""
    # Allow selection of method via SF_METHOD
    if SF_METHOD == 'avg':
        return 0.5 * (s1.Sf_total + s2.Sf_total)

    Ksum = s1.K_t + s2.K_t
    if Ksum <= 0.0:
        return 0.0
    Qsum = s1.Q_lob + s1.Q_ch + s1.Q_rob  # equals total Q
    return ((Qsum + Qsum) / Ksum) ** 2  # = (2*Q / (K1+K2))^2


def minor_loss_coeff(s_dn: SectionState, s_up: SectionState, xs_dn: CrossSection) -> float:
    """
    Select contraction/expansion coefficient for the *flow direction* (US -> DS).

    The marching solver indexes sections as downstream (s_dn) and upstream (s_up),
    but minor-loss coefficients must be selected based on downstream flow direction:
      - expansion if A_ds > A_us
      - contraction if A_ds <= A_us
    """
    if s_dn.A_t > s_up.A_t:  # expanding in downstream flow direction
        return xs_dn.expansion_coeff
    return xs_dn.contraction_coeff


def head_loss(s_dn: SectionState, s_up: SectionState, link: ReachLink, xs_dn: CrossSection) -> float:
    """Compute total head loss: friction + minor (expansion/contraction)."""
    # Average subsection flows for DWRL
    Qlob_av = 0.5 * (s_dn.Q_lob + s_up.Q_lob)
    Qch_av  = 0.5 * (s_dn.Q_ch  + s_up.Q_ch)
    Qrob_av = 0.5 * (s_dn.Q_rob + s_up.Q_rob)

    Ldw = link.discharge_weighted_length(Qlob_av, Qch_av, Qrob_av)

    # Representative friction slope using total conveyance approach
    Sf = representative_friction_slope_total(s_dn, s_up)

    hf = Sf * Ldw

    C = minor_loss_coeff(s_dn, s_up, xs_dn)
    # Minor losses are dissipative and must be non-negative. Use absolute
    # velocity-head difference between the two sections.
    vh_up = (s_up.alpha * s_up.V_t ** 2) / (2.0 * G)
    vh_dn = (s_dn.alpha * s_dn.V_t ** 2) / (2.0 * G)
    hv = C * abs(vh_up - vh_dn)

    return hf + hv


def irregular_weir_flow_from_geometry(
    xs_culvert: CrossSection,
    headwater_wse: float,
    z_crown_inlet: float,
    Cw: float = 3.0,
    samples_per_segment: int = 8,
    sta_left: float = 0.0,
    sta_right: float = 0.0,
) -> float:
    """
    Irregular broad-crested weir discharge using culvert XS geometry.

    Integrates q = Cw * h^(3/2) across the cross section where
    h(x) = headwater_wse - max(z(x), z_crown_inlet), with z(x) linearly
    interpolated between surveyed station/elevation points.

    If sta_left < sta_right, integration is clipped to [sta_left, sta_right].
    This replicates HEC-RAS "Weir Sta Left / Weir Sta Right" which limits
    overtopping to the horizontal extent covered by the upstream channel XS.
    When both are 0 (default), the full geometry extent is used.
    """
    if headwater_wse <= z_crown_inlet:
        return 0.0
    if Cw <= 0.0:
        return 0.0

    try:
        geom = sorted([(float(x), float(z)) for x, z in xs_culvert.geometry], key=lambda p: p[0])
    except Exception:
        geom = []

    if len(geom) < 2:
        # Fallback: simple rectangular weir using the supplied station range or full width.
        if sta_left < sta_right:
            L = sta_right - sta_left
        else:
            L = max(0.0, xs_culvert.culvert_full_depth())  # placeholder; no geometry
        h = max(0.0, headwater_wse - z_crown_inlet)
        return Cw * L * (h ** 1.5)

    # Determine integration bounds.
    x_lo = geom[0][0]
    x_hi = geom[-1][0]
    if sta_left < sta_right:
        x_lo = max(x_lo, sta_left)
        x_hi = min(x_hi, sta_right)
    if x_hi <= x_lo:
        return 0.0

    q_total = 0.0
    n_sub = max(2, int(samples_per_segment))

    for (x0, z0), (x1, z1) in zip(geom[:-1], geom[1:]):
        # Clip segment to integration bounds.
        seg_lo = max(x0, x_lo)
        seg_hi = min(x1, x_hi)
        if seg_hi <= seg_lo:
            continue
        dx_full = x1 - x0
        if abs(dx_full) <= 1.0e-12:
            continue
        # Linearly interpolate z at the clipped endpoints.
        t_lo = (seg_lo - x0) / dx_full
        t_hi = (seg_hi - x0) / dx_full
        dz = z1 - z0
        seg_len = seg_hi - seg_lo

        # Midpoint quadrature over the clipped segment.
        dq = 0.0
        for j in range(n_sub):
            t = t_lo + (t_hi - t_lo) * ((j + 0.5) / n_sub)
            z_t = z0 + dz * t
            crest_t = max(z_t, z_crown_inlet)
            h_t = headwater_wse - crest_t
            if h_t > 0.0:
                dq += h_t ** 1.5

        q_total += Cw * (seg_len / n_sub) * dq

    return max(0.0, q_total)


# ---------------------------------------------------------------------------
# Culvert inlet control (FHWA HEC-5 equations)
# ---------------------------------------------------------------------------

def apply_culvert_control(xs_culvert: CrossSection, z_invert: float, 
                          wse_headwater: float, wse_tailwater: float, 
                          Q_target: float) -> Tuple[float, bool, str]:
    """
    Apply both inlet and outlet control to limit flow (FHWA HEC-5 equations).
    
    Computes:
    - Inlet control: max flow based on headwater depth
    - Outlet control: max flow based on tailwater submergence and friction losses
    - Returns the minimum (most restrictive) control as the culvert-limited flow
    
    Args:
        xs_culvert: CrossSection with culvert properties
        z_invert: Invert elevation at culvert (ft)
        wse_headwater: Water surface elevation at culvert inlet (ft)
        wse_tailwater: Water surface elevation at culvert outlet (ft)
        Q_target: Target flow rate (cfs)
    
    Returns:
        (Q_controlled, culvert_is_restricting, control_type)
        Q_controlled: Culvert-limited flow (cfs)
        culvert_is_restricting: True if culvert is limiting flow below Q_target
        control_type: 'inlet', 'outlet', or 'none'
    """
    if not HAVE_CULVERT or not xs_culvert.has_culvert():
        return Q_target, False, 'none'
    
    try:
        # Culvert geometry
        culvert_slope = xs_culvert.culvert_slope()
        z_inlet = xs_culvert.culvert_upstream_invert if xs_culvert.culvert_upstream_invert != 0.0 else z_invert
        z_outlet = xs_culvert.culvert_downstream_invert if xs_culvert.culvert_downstream_invert != 0.0 else z_inlet
        
        if xs_culvert.culvert_shape == 'circular':
            xsect = CircularXsect(diameter_ft=xs_culvert.culvert_diameter, 
                                  culvert_code=xs_culvert.culvert_code)
        elif xs_culvert.culvert_shape == 'rect':
            xsect = RectangularXsect(width_ft=xs_culvert.culvert_width,
                                     height_ft=xs_culvert.culvert_height,
                                     culvert_code=xs_culvert.culvert_code)
        else:
            return Q_target, False, 'none'
        
        # INLET CONTROL calculation
        h_headwater = max(0.0, wse_headwater - z_inlet)
        q_inlet, _, _, _ = inlet_controlled_flow(xsect, culvert_slope, h_headwater)
        
        # OUTLET CONTROL calculation (simplified version)
        # Outlet control is dominant when tailwater is high (submerged outlet)
        # In this case, flow is limited by normal/critical depth at outlet and friction losses
        h_tailwater = max(0.0, wse_tailwater - z_outlet)
        
        # Simple outlet control: if tailwater is above outlet invert, check submergence
        q_outlet = Q_target  # Default: no outlet control
        is_submerged = h_tailwater > 0.0
        
        if is_submerged:
            # Outlet is submerged; use tailwater as constraint
            # For a submerged outlet, apply a simple friction-based limit
            # Q_outlet = full_flow * (1 - submergence_factor)
            # More accurate version would compute Manning equation with tailwater
            if h_tailwater >= xsect.yFull:
                # Fully submerged outlet; flow significantly reduced
                q_outlet = Q_target * 0.7  # Conservative estimate
            else:
                # Partially submerged; flow reduced proportionally
                submergence_ratio = h_tailwater / xsect.yFull
                q_outlet = Q_target * (1.0 - 0.3 * submergence_ratio)
        
        # Select the more restrictive control
        if q_inlet < q_outlet:
            q_controlled = q_inlet
            control_type = 'inlet'
        elif q_outlet < q_inlet:
            q_controlled = q_outlet
            control_type = 'outlet'
        else:
            q_controlled = q_inlet
            control_type = 'inlet'

        # Overtopping weir flow from irregular culvert cross-section geometry.
        y_full = xs_culvert.culvert_full_depth()
        if y_full > 0.0:
            z_crown = z_inlet + y_full
            Cw = max(0.0, float(getattr(xs_culvert, 'culvert_weir_coeff', 3.0) or 3.0))
            sta_left = float(getattr(xs_culvert, 'culvert_weir_sta_left', 0.0) or 0.0)
            sta_right = float(getattr(xs_culvert, 'culvert_weir_sta_right', 0.0) or 0.0)
            q_weir = irregular_weir_flow_from_geometry(
                xs_culvert=xs_culvert,
                headwater_wse=wse_headwater,
                z_crown_inlet=z_crown,
                Cw=Cw,
                sta_left=sta_left,
                sta_right=sta_right,
            )
            if q_weir > 0.0:
                q_controlled += q_weir
                control_type = f'{control_type}+weir'
        
        culvert_is_restricting = q_controlled < Q_target
        return q_controlled, culvert_is_restricting, control_type
        
    except Exception as e:
        # If culvert computation fails, proceed without culvert control
        print(f"WARNING: Culvert control calculation failed: {e}")
        return Q_target, False, 'none'


def solve_culvert_headwater(
    xs_culvert: 'CrossSection',
    tailwater_wse: float,
    Q_total: float,
    tailwater_state: Optional['SectionState'] = None,
    downstream_boundary_xs: Optional['CrossSection'] = None,
    boundary_reach_length: Optional[float] = None,
) -> Tuple[float, str]:
    """
     Compute culvert headwater using a HEC-RAS-style decision tree.

     Key behaviors implemented here:
     - inlet control from FHWA HEC-5 equations
     - downstream section-to-barrel energy balance at the outlet face using
       the tailwater EGL and expansion loss, giving the correct inside-barrel
       starting depth for the direct-step profile
     - true direct-step water surface profile through the partially full barrel
     - pressurized full-flow outlet control only when the barrel is effectively full
     - supercritical least-error fallback when no converged supercritical branch
       can be found (mirrors HEC-RAS behavior)
     - overtopping weir flow by iterating the culvert/weir flow split

     Parameters
     ----------
     tailwater_state : SectionState, optional
         The computed hydraulic state of the downstream cross section.  When
         provided, the velocity head at the tailwater section is used in the
         downstream section-to-barrel energy balance.  If None the tailwater
         approach velocity is assumed negligible (conservative for wide channels).
     downstream_boundary_xs : CrossSection, optional
         Cross section at the downstream hydraulic boundary for this culvert
         solve.
     boundary_reach_length : float, optional
         Along-channel distance between downstream and upstream boundary cross
         sections (ft). Used to compute approach energy slope for weir offset.

     Returns (headwater_wse, control_type_str).
    """
    if not xs_culvert.has_culvert():
        return tailwater_wse, 'passthrough'

    _ensure_culvert_runtime()

    z_inlet  = xs_culvert.culvert_upstream_invert
    z_outlet = xs_culvert.culvert_downstream_invert
    slope    = xs_culvert.culvert_slope()
    y_full   = xs_culvert.culvert_full_depth()
    L        = xs_culvert.culvert_length

    if y_full <= 0.0:
        return tailwater_wse, 'passthrough'

    # --- Full-barrel geometry ---
    shape = (xs_culvert.culvert_shape or '').strip().lower()
    if shape == 'circular':
        D      = xs_culvert.culvert_diameter
        R_bar  = D / 2.0
        A_full = math.pi * R_bar ** 2
        P_full = math.pi * D
    elif shape == 'rect':
        B      = xs_culvert.culvert_width
        H_rect = xs_culvert.culvert_height
        A_full = B * H_rect
        P_full = 2.0 * (B + H_rect)
    else:
        return tailwater_wse, 'passthrough'

    if A_full <= 0.0:
        return tailwater_wse, 'passthrough'

    R_full = A_full / P_full
    n_barrel = xs_culvert.n_ch

    # -------------------------------------------------------------------
    # Loss coefficients (FHWA HDS-5 / HEC-RAS full-flow formula)
    #   H_L = H_e + H_f + H_x
    #   H_e = K_e * V²/2g             entrance loss
    #   H_f = K_f * V²/2g             Manning friction (full-pipe)
    #   H_x = C_x * V²/2g             exit loss (1.0 = sudden expansion)
    # For a free-outfall outlet (TW < dc): standard FHWA formula uses
    #   HW_OC = TW_eff + (1 + K_e + K_f) * V²/2g
    # For a submerged outlet (TW >= dc): exit loss reduces by TW velocity head
    # but the standard approximation still uses (1 + K_e + K_f); see HEC-RAS §4.
    # -------------------------------------------------------------------
    Ke = 0.5   # entrance loss coefficient (square-edge headwall default)
    if R_full > 0.0 and L > 0.0:
        Kf = (2.0 * G * n_barrel ** 2 * L) / (1.486 ** 2 * R_full ** (4.0 / 3.0))
    else:
        Kf = 0.0

    # -------------------------------------------------------------------
    # Build culvert_routine cross-section for inlet control calculations
    # -------------------------------------------------------------------
    _xsect = None
    if HAVE_CULVERT:
        try:
            if shape == 'circular':
                _xsect = CircularXsect(diameter_ft=xs_culvert.culvert_diameter,
                                       culvert_code=xs_culvert.culvert_code)
            elif shape == 'rect':
                _xsect = RectangularXsect(width_ft=xs_culvert.culvert_width,
                                          height_ft=xs_culvert.culvert_height,
                                          culvert_code=xs_culvert.culvert_code)
        except Exception:
            _xsect = None

    # -------------------------------------------------------------------
    # Culvert crown at inlet (overtopping weir crest)
    # -------------------------------------------------------------------
    # For HEC-RAS style culverts, the cross section geometry represents the
    # embankment profile (road top). The weir crest is the minimum elevation
    # in that geometry — the lowest point where water overtops the embankment.
    # Fallback to inlet+y_full if geometry is not available.
    z_crown_inlet = z_inlet + y_full
    if xs_culvert.geometry and len(xs_culvert.geometry) >= 2:
        try:
            z_geom_min = min(z for x, z in xs_culvert.geometry)
            if z_geom_min > z_inlet:
                z_crown_inlet = z_geom_min
        except Exception:
            pass

    Cw = max(0.0, float(getattr(xs_culvert, 'culvert_weir_coeff', 3.0) or 3.0))
    weir_sta_left = float(getattr(xs_culvert, 'culvert_weir_sta_left', 0.0) or 0.0)
    weir_sta_right = float(getattr(xs_culvert, 'culvert_weir_sta_right', 0.0) or 0.0)

    def inlet_control_hw(Q_trial: float) -> Tuple[float, str]:
        if Q_trial <= 0.0:
            return z_inlet, 'inlet'
        if _xsect is None:
            return z_inlet, 'inlet'
        h_hw, _q, _condition, _yr = solve_headwater_depth_for_Q(
            _xsect,
            slope=slope,
            Q_target=Q_trial,
            h_min=0.0,
            h_max=max(10.0 * y_full, 1.0),
        )
        return z_inlet + h_hw, 'inlet'

    # --- Precompute tailwater EGL and approach velocity head ---
    # When the downstream SectionState is available, the tailwater energy grade
    # line includes the velocity head of the downstream channel section.
    # Otherwise the approach velocity is assumed negligible.
    if tailwater_state is not None:
        try:
            _tw_vh = (tailwater_state.alpha * tailwater_state.V_t ** 2) / (2.0 * G)
        except Exception:
            _tw_vh = 0.0
    else:
        _tw_vh = 0.0
    _egl_tw = tailwater_wse + _tw_vh  # total energy at tailwater cross section

    # Reach length between downstream and upstream boundary cross sections.
    # Prefer explicit caller input; otherwise infer from downstream section data.
    approach_length = 0.0
    if boundary_reach_length is not None:
        try:
            approach_length = max(0.0, float(boundary_reach_length))
        except Exception:
            approach_length = 0.0
    if approach_length <= 0.0 and downstream_boundary_xs is not None:
        try:
            approach_length = max(
                0.0,
                float(getattr(downstream_boundary_xs, 'L_ch_to_next', 0.0) or 0.0),
                float(getattr(downstream_boundary_xs, 'L_lob_to_next', 0.0) or 0.0),
                float(getattr(downstream_boundary_xs, 'L_rob_to_next', 0.0) or 0.0),
            )
        except Exception:
            approach_length = 0.0

    def _approach_energy_slope(hw_wse: float) -> float:
        """Compute approach energy slope from boundary-section energy drop."""
        if approach_length <= 1.0e-6:
            return max(0.0, slope)

        vh_up = 0.0
        try:
            s_up = compute_state(xs_culvert, hw_wse, Q_total)
            vh_up = (s_up.alpha * s_up.V_t ** 2) / (2.0 * G)
        except Exception:
            vh_up = 0.0

        egl_up = hw_wse + vh_up
        return max(0.0, (egl_up - _egl_tw) / approach_length)

    # Expansion coefficient for exit from barrel to downstream channel.
    # HEC-RAS stores this on the culvert (upstream) cross section.
    Ce_exit = max(0.0, min(1.0, float(xs_culvert.expansion_coeff)))

    def _solve_barrel_outlet_depth(Q_trial: float) -> float:
        """
        Downstream section-to-barrel energy balance at the outlet face.

        Solves for the inside-barrel depth at the outlet face by balancing
        the tailwater EGL with the exit expansion loss:

            z_outlet + y_out + V_out²/(2g) = EGL_tw + Ce*(V_out² - V_tw²)/(2g)

        Rearranging to an implicit equation in y_out:
            y_out + V_out²/(2g)*(1 - Ce) = (EGL_tw - z_outlet) - Ce*V_tw²/(2g)

        The result is capped to [dc, y_full] so the profile always starts at or
        above critical depth (subcritical assumption at the outlet).
        """
        if not (HAVE_CULVERT and _xsect is not None):
            return max(0.0, tailwater_wse - z_outlet)

        try:
            dc_local = critical_depth_in_culvert(_xsect, Q_trial)
        except Exception:
            dc_local = 0.5 * y_full
        dc_local = min(dc_local, y_full)

        # Right-hand side of the rearranged energy equation
        rhs = (_egl_tw - z_outlet) - Ce_exit * _tw_vh
        eps = 1.0e-8

        # If barrel is already pressurized at outlet (tailwater above crown),
        # skip the partial solve and signal full flow.
        if rhs >= y_full + eps:
            return y_full  # caller will detect full-flow condition

        # Implicit solve: f(y) = y + (1-Ce)*Q²/(2g*A²) - rhs = 0
        # Only valid on subcritical branch: y in [dc_local, y_full]
        def f_outlet(y: float) -> float:
            if y <= 0.0:
                return -rhs
            area = _xsect.area(y)
            if area <= 0.0:
                return -rhs
            return y + (1.0 - Ce_exit) * Q_trial ** 2 / (2.0 * G * area ** 2) - rhs

        y_lo = max(eps, dc_local)
        y_hi = y_full - eps
        if y_hi <= y_lo:
            return dc_local

        f_lo = f_outlet(y_lo)
        f_hi = f_outlet(y_hi)

        if f_lo * f_hi > 0.0:
            # No sign change on subcritical branch; the energy target is below dc,
            # meaning the outlet is supercritical or free-overfall.  Use dc.
            return dc_local

        for _ in range(60):
            y_m = 0.5 * (y_lo + y_hi)
            f_m = f_outlet(y_m)
            if abs(f_m) < 1.0e-7 or (y_hi - y_lo) < 1.0e-7:
                return max(dc_local, min(y_m, y_full))
            if f_lo * f_m <= 0.0:
                y_hi = y_m
                f_hi = f_m
            else:
                y_lo = y_m
                f_lo = f_m
        return max(dc_local, min(0.5 * (y_lo + y_hi), y_full))

    def outlet_control_hw(Q_trial: float) -> Tuple[float, str]:
        if Q_trial <= 0.0:
            return z_inlet, 'outlet-partial'

        tw_depth_raw = max(0.0, tailwater_wse - z_outlet)

        if HAVE_CULVERT and _xsect is not None:
            try:
                dc = critical_depth_in_culvert(_xsect, Q_trial)
            except Exception:
                dc = 0.5 * y_full
            try:
                yn = solve_normal_depth_in_culvert(_xsect, Q_trial, n_barrel, slope)
            except Exception:
                yn = y_full
        else:
            if shape == 'rect' and B > 0.0:
                dc = ((Q_trial / B) ** 2 / G) ** (1.0 / 3.0)
            else:
                dc = 0.5 * y_full
            yn = y_full

        dc = min(dc, y_full)
        yn = min(yn, y_full)

        tol = 1.0e-6

        # ------------------------------------------------------------------
        # Flowchart branch logic for outlet profile classification:
        # - TW > rise => full-flow equations
        # - slope == 0: direct-step from TW (if TW>dc) else from dc (H2)
        # - slope > 0, yn > dc: M1/M2/uniform family
        # - slope > 0, yn < dc:
        #     * TW > dc: S1 backwater from TW
        #     * TW <= dc: run direct-step from dc (free outfall).
        #       If the barrel pressurizes (partial-full / full-from-outlet),
        #       OC governs — HEC-RAS discards the supercritical IC answer in
        #       that case ("outlet answer will be used").
        #       Only if the barrel stays partially full supercritical does IC
        #       govern (handled in governing_culvert_headwater).
        # ------------------------------------------------------------------
        steep_free_outfall = slope > tol and yn < dc - tol and tw_depth_raw <= dc + tol

        if slope <= tol:
            if tw_depth_raw > dc + tol:
                outlet_start_depth = min(max(tw_depth_raw, dc), y_full)
                profile_family = 'zero-slope-tailwater'
            else:
                outlet_start_depth = dc
                profile_family = 'h2-critical-start'
        elif steep_free_outfall:
            # Steep slope, TW below critical: start direct-step from dc.
            # If the barrel pressurizes, OC governs (HEC-RAS rule).
            outlet_start_depth = dc
            profile_family = 'steep-free-outfall'
        elif yn > dc + tol:
            if tw_depth_raw >= yn - tol:
                outlet_start_depth = min(max(tw_depth_raw, dc), y_full)
                profile_family = 'm1-backwater'
            elif tw_depth_raw > dc + tol:
                outlet_start_depth = min(max(tw_depth_raw, dc), y_full)
                profile_family = 'm2-tailwater-start'
            else:
                outlet_start_depth = dc
                profile_family = 'm2-critical-start'
        else:
            outlet_start_depth = min(max(tw_depth_raw, dc), y_full)
            profile_family = 's1-backwater'

        # ---------------------------------------------------------------
        # Downstream section-to-barrel energy balance: find the correct
        # inside-barrel depth at the outlet face, accounting for the exit
        # expansion loss and the tailwater channel velocity head.
        # This replaces the simple max(tw_depth_raw, dc) surrogate.
        # ---------------------------------------------------------------
        outlet_depth = _solve_barrel_outlet_depth(Q_trial)

        barrel_full = (
            tw_depth_raw >= (y_full - 1.0e-6)
            or outlet_depth >= (y_full - 1.0e-6)
        )

        if barrel_full:
            v_full = Q_trial / max(A_full, 1.0e-6)
            hw_full = max(tailwater_wse, z_outlet + y_full) + (1.0 + Ke + Kf) * (v_full ** 2) / (2.0 * G)
            return hw_full, 'outlet-full'

        if HAVE_CULVERT and _xsect is not None:
            try:
                energy_up_inside, depth_up_inside, profile_mode = direct_step_culvert_upstream_energy(
                    xsect=_xsect,
                    Q=Q_trial,
                    n_value=n_barrel,
                    slope=slope,
                    length=L,
                    tailwater_depth=outlet_start_depth,
                )

                # S1 branch check from the flowchart: when the upstream barrel
                # depth falls to critical in supercritical regime, use inlet control.
                if profile_family == 's1-backwater' and depth_up_inside <= dc + 1.0e-3:
                    hw_inlet_only, _ = inlet_control_hw(Q_trial)
                    return hw_inlet_only, 'inlet-supercritical'

                # Steep free-outfall: if barrel stays partial, inlet control governs.
                # If barrel pressurizes (partial-full / full-from-outlet), OC governs
                # — HEC-RAS discards the supercritical IC answer in that case.
                if profile_family == 'steep-free-outfall' and profile_mode == 'partial':
                    hw_inlet_only, _ = inlet_control_hw(Q_trial)
                    return hw_inlet_only, 'inlet-supercritical'

                area_up = max(_xsect.area(min(max(depth_up_inside, 1.0e-6), y_full)), 1.0e-6)
                v_up = Q_trial / area_up
                hw_partial = z_inlet + energy_up_inside + Ke * (v_up ** 2) / (2.0 * G)
                if profile_mode in ('partial-full', 'full-from-outlet'):
                    return hw_partial, 'outlet-full'
                return hw_partial, f'outlet-partial:{profile_family}'
            except Exception:
                pass

        # Fallback (no culvert_routine): single-depth surrogate
        y_up = max(outlet_depth, yn)
        area_up = max(A_full * max(y_up / y_full, 1.0e-6), 1.0e-6)
        v_up = Q_trial / area_up
        hw_partial = z_inlet + y_up + Ke * (v_up ** 2) / (2.0 * G)
        return hw_partial, f'outlet-partial:{profile_family}'

    def Q_weir_at(hw_wse: float) -> float:
        """Overtopping broad-crested weir flow from irregular XS geometry."""
        return irregular_weir_flow_from_geometry(
            xs_culvert=xs_culvert,
            headwater_wse=hw_wse,
            z_crown_inlet=z_crown_inlet,
            Cw=Cw,
            sta_left=weir_sta_left,
            sta_right=weir_sta_right,
        )

    def governing_culvert_headwater(Q_cul: float) -> Tuple[float, str]:
        hw_ic, _ = inlet_control_hw(Q_cul)
        hw_oc, oc_mode = outlet_control_hw(Q_cul)
        if oc_mode.startswith('inlet-'):
            return hw_oc, 'inlet'
        if hw_ic >= hw_oc:
            return hw_ic, 'inlet'
        return hw_oc, oc_mode

    # -------------------------------------------------------------------
    # Step 1: Solve assuming all flow passes through culvert barrel (§10)
    # -------------------------------------------------------------------
    HW, base_control = governing_culvert_headwater(Q_total)

    # -------------------------------------------------------------------
    # Step 2: Iterative weir flow split (HEC-RAS §10)
    # If the culvert-only HW exceeds the crown, iterate:
    #   Q_weir = f(HW);  Q_culvert = Q_total - Q_weir
    # until energy convergence (same HW from both paths).
    # -------------------------------------------------------------------
    Q_weir_final = 0.0
    if HW > z_crown_inlet:
        # ------------------------------------------------------------------
        # Bisect on HW* to find the self-consistent headwater where:
        #   governing_culvert_headwater(Q_total - Q_weir_at(HW*)) == HW*
        #
        # This is stable against the oscillation that a naive fixed-point
        # iteration produces (fixed-point fires on iteration 0 because the
        # initial seed equals the all-culvert HW, giving HW_new == HW_prev
        # before any flow reduction has occurred).
        #
        # Bracket:
        #   hw_lo = z_crown_inlet  → Q_weir=0, HW_culvert(Q_total) > hw_lo  → f>0
        #   hw_hi = HW             → Q_weir>0, HW_culvert(Q_total-Qw) < hw_hi → f<0
        # ------------------------------------------------------------------
        hw_lo = z_crown_inlet
        hw_hi = HW
        Q_w = 0.0

        def _f_hw(hw_trial: float) -> float:
            q_w_trial = Q_weir_at(hw_trial)
            q_cul_trial = max(0.0, Q_total - q_w_trial)
            hw_cul, _ = governing_culvert_headwater(q_cul_trial)
            return hw_cul - hw_trial

        f_lo = _f_hw(hw_lo)
        f_hi = _f_hw(hw_hi)

        if False:  # DEBUG: set to True for verbose tracing
            print(f'    [weir-bisect] hw_lo={hw_lo:.3f}, f_lo={f_lo:.3f}')
            print(f'    [weir-bisect] hw_hi={hw_hi:.3f}, f_hi={f_hi:.3f}')

        if f_lo <= 0.0:
            # Even at z_crown the culvert alone is below the crown — no weir needed
            HW = hw_lo
            Q_w = 0.0
        elif f_hi >= 0.0:
            # Culvert HW still exceeds hw_hi — use hw_hi (all-culvert answer)
            HW = hw_hi
            Q_w = Q_weir_at(hw_hi)
        else:
            for _ in range(60):
                hw_mid = 0.5 * (hw_lo + hw_hi)
                f_mid = _f_hw(hw_mid)
                if abs(f_mid) < 0.001 or (hw_hi - hw_lo) < 0.001:
                    HW = hw_mid
                    Q_w = Q_weir_at(hw_mid)
                    break
                if f_lo * f_mid <= 0.0:
                    hw_hi = hw_mid
                    f_hi = f_mid
                else:
                    hw_lo = hw_mid
                    f_lo = f_mid
            else:
                HW = 0.5 * (hw_lo + hw_hi)
                Q_w = Q_weir_at(HW)

        Q_weir_final = Q_w

    # -------------------------------------------------------------------
    # Step 3: Convert HW_EGL → HW_WSE (HEC-RAS §1, §4)
    # "The upstream water surface (WSU) is obtained by placing the computed
    # energy into the upstream cross section and computing the water surface
    # that corresponds to that energy for the given flow rate."
    # Solve: WSE + α*(Q/A(WSE))²/(2g) = HW_EGL
    # -------------------------------------------------------------------
    HW_EGL = HW  # our bisection worked in WSE but the inlet eqn gives EGL depth
    # Use the culvert cross-section geometry to subtract velocity head.
    # For the upstream section (wide open channel), approach velocity is usually
    # small; bisect to be exact.
    def _egl_residual(wse: float) -> float:
        """EGL(wse) - HW_EGL."""
        try:
            state = compute_state(xs_culvert, wse, Q_total)
            vh = (state.alpha * state.V_t ** 2) / (2.0 * G)
        except Exception:
            vh = 0.0
        return wse + vh - HW_EGL

    # Only iterate if the EGL correction is potentially meaningful
    try:
        _test_state = compute_state(xs_culvert, HW_EGL, Q_total)
        vh_at_egl = (_test_state.alpha * _test_state.V_t ** 2) / (2.0 * G)
    except Exception:
        vh_at_egl = 0.0

    HW_WSE = HW_EGL
    if vh_at_egl > 0.01:  # correction > 0.01 ft — worth iterating
        wse_lo = max(z_inlet, HW_EGL - 5.0 * vh_at_egl)
        wse_hi = HW_EGL
        try:
            f_lo_egl = _egl_residual(wse_lo)
            f_hi_egl = _egl_residual(wse_hi)
            if f_lo_egl * f_hi_egl < 0.0:
                for _ in range(50):
                    wse_m = 0.5 * (wse_lo + wse_hi)
                    f_m = _egl_residual(wse_m)
                    if abs(f_m) < 5.0e-4:
                        HW_WSE = wse_m
                        break
                    if f_lo_egl * f_m <= 0.0:
                        wse_hi = wse_m
                    else:
                        wse_lo = wse_m
                        f_lo_egl = f_m
                else:
                    HW_WSE = 0.5 * (wse_lo + wse_hi)
        except Exception:
            HW_WSE = HW_EGL

    # -------------------------------------------------------------------
    # Determine control type label
    # -------------------------------------------------------------------
    hw_ic_sol, _ = inlet_control_hw(max(0.0, Q_total - Q_weir_final))
    hw_oc_sol, oc_mode = outlet_control_hw(max(0.0, Q_total - Q_weir_final))
    control = 'inlet' if hw_ic_sol >= hw_oc_sol else oc_mode.replace('-partial', '').replace('-full', '')
    if Q_weir_final > 0.01:
        control = f'{control}+weir'

    return HW_WSE, control


# ---------------------------------------------------------------------------
# Standard-step solver (subcritical, marching upstream)
# ---------------------------------------------------------------------------

def solve_energy_upstream(
    xs_dn: CrossSection, xs_up: CrossSection,
    z_dn: float, z_up: float,
    Q_total_dn: float, Q_total_up: float,
    s_dn: SectionState,
    link: ReachLink,
    wse_up_init: Optional[float] = None,
    tol: float = 0.01,
    max_iter: int = 50
) -> SectionState:
    """
    Solve for upstream water surface elevation given downstream state using the energy equation:
        z_dn + y_dn + alpha_dn V_dn^2/(2g) = z_up + y_up + alpha_up V_up^2/(2g) + h_L
    where h_L = friction + minor losses.

    Uses a secant-like iteration on wse_up.
    """
    if wse_up_init is None:
        # HEC-RAS: first trial projects the previous cross section's depth
        # onto the current (upstream) section: WS_proj = z_up + depth_dn
        try:
            depth_dn = getattr(s_dn, 'depth_at_min', None)
            if depth_dn is None:
                depth_dn = s_dn.wse - min(z for _, z in xs_dn.geometry)
            wse_up_init = max(z_up + depth_dn, z_up + 1e-3)
        except Exception:
            # Fallback to previous WSE or modest headroom
            wse_up_init = max(z_up + 0.5, s_dn.wse)

    # maximum allowed Froude for acceptable (subcritical) solutions
    FR_MAX = 0.9999

    # Secant with robust bracketing/bisection fallback
    w1 = wse_up_init
    s1 = compute_state(xs_up, w1, Q_total_up)

    # Helper: attempt to find a sign-changing bracket by sampling a widened range
    def _find_bracket_by_sampling(w_lo_candidate: float, w_hi_candidate: float, samples: int = 41):
        ws = []
        lo = w_lo_candidate
        hi = w_hi_candidate
        if hi <= lo:
            return None
        step = (hi - lo) / float(samples - 1)
        prev_f = None
        prev_s = None
        for i in range(samples):
            w = lo + i * step
            s_try, f_try, _ = energy_balance_state(w)
            # treat sentinel large residuals as invalid
            if abs(f_try) >= 9e5 or math.isnan(f_try):
                prev_f = None
                prev_s = None
                continue
            if prev_f is not None and prev_f * f_try <= 0:
                # ensure both ends are subcritical
                if getattr(prev_s, 'Froude', 0.0) < 0.9999 and getattr(s_try, 'Froude', 0.0) < 0.9999:
                    return (prev_w, w)
            prev_f = f_try
            prev_s = s_try
            prev_w = w
        return None
    def energy_balance_state(w_try: float):
        s_try = compute_state(xs_up, w_try, Q_total_up)
        y_dn = s_dn.wse - z_dn
        y_up = s_try.wse - z_up
        loss = head_loss(s_dn, s_try, link, xs_dn)
        lhs = z_dn + y_dn + (s_dn.alpha * s_dn.V_t ** 2) / (2.0 * G)
        rhs = z_up + y_up + (s_try.alpha * s_try.V_t ** 2) / (2.0 * G) + loss
        f = lhs - rhs
        # Reject supercritical trial states by returning a large residual (preserves sign)
        FR_MAX = 0.9999
        if getattr(s_try, 'Froude', 0.0) >= FR_MAX or math.isnan(f):
            f = math.copysign(1e6, f if not math.isnan(f) else 1.0)
        return s_try, f, loss

    f1 = energy_balance_state(w1)[1]

    # Choose a second trial per HEC-RAS: compute a one-step "computed" WS
    # via a small finite-difference derivative and move 70% toward it.
    def one_step_computed_ws(w_base: float, f_base: float, eps: float = 0.01) -> Optional[float]:
        try:
            f_plus = energy_balance_state(w_base + eps)[1]
            df = (f_plus - f_base) / eps
            if abs(df) < 1e-12:
                return None
            return w_base - f_base / df
        except Exception:
            return None

    f1 = energy_balance_state(w1)[1]
    w_comp1 = one_step_computed_ws(w1, f1)
    if w_comp1 is None:
        w2 = w1 + 0.5
    else:
        w2 = w1 + 0.70 * (w_comp1 - w1)
    s2, f2, _ = energy_balance_state(w2)

    # Prepare bracket sampling bounds for later use
    zmin_up = min(z for _, z in xs_up.geometry)
    max_search = max(w1 + 50.0, zmin_up + 1.0)

    # If no sign change between w1 and w2, attempt to find a bracket by sampling
    if f1 * f2 > 0:
        found = False
        # search upward values for a sign change
        for factor in [0.5, 1.0, 2.0, 5.0, 10.0]:
            w_try = w1 + factor * (max_search - w1)
            s_try, f_try, _ = energy_balance_state(w_try)
            if f1 * f_try <= 0:
                w2, s2, f2 = w_try, s_try, f_try
                found = True
                break
        if not found:
            # fallback to a modest bracket around initial guess
            w_lo = max(zmin_up + 1e-3, w1 - 10.0)
            w_hi = w1 + 10.0
            s_lo, f_lo, _ = energy_balance_state(w_lo)
            s_hi, f_hi, _ = energy_balance_state(w_hi)
            if f_lo * f_hi <= 0:
                w1, s1, f1 = w_lo, s_lo, f_lo
                w2, s2, f2 = w_hi, s_hi, f_hi

    V_MAX = 50.0
    # HEC-RAS-style conservative under-relaxation: limit change per iteration
    MAX_DELTA_W = 1.0  # ft per iteration (conservative)

    for _ in range(max_iter):
        # Secant step (safe-guarded)
        if abs(f2 - f1) < 1e-12:
            w3 = 0.5 * (w1 + w2)
        else:
            w3 = w2 - f2 * (w2 - w1) / (f2 - f1)

        # Enforce physical lower/upper bounds and apply relaxation to match HEC-RAS behavior
        w3 = max(w3, zmin_up + 1e-3)
        w3 = min(w3, max(w1, w2) + 1e3)
        # Limit change from previous accepted iterate to avoid oscillations
        ref = w2
        if abs(w3 - ref) > MAX_DELTA_W:
            w3 = ref + math.copysign(MAX_DELTA_W, (w3 - ref))

        s3, f3, loss3 = energy_balance_state(w3)

        # If trial state is numerically unsafe (tiny area, tiny K, or huge velocity)
        # reject the secant step and use bisection/midpoint instead.
        if s3.A_t <= 1e-6 or s3.K_t <= 1e-6 or s3.V_t > V_MAX or math.isnan(f3) or abs(f3) > 1e8 or getattr(s3, 'Froude', 0.0) >= 0.9999:
            # try expanding current bracket by sampling between a wider range; fallback to midpoint
            try:
                w_lo_try = max(zmin_up + 1e-3, min(w1, w2) - 50.0)
                w_hi_try = max(w1, w2) + 200.0
                bracket = _find_bracket_by_sampling(w_lo_try, w_hi_try, samples=81)
                if bracket is not None:
                    w1, w2 = bracket
                    s1 = compute_state(xs_up, w1, Q_total_up)
                    s2 = compute_state(xs_up, w2, Q_total_up)
                    f1 = energy_balance_state(w1)[1]
                    f2 = energy_balance_state(w2)[1]
                    # retry secant from new bracket
                    if abs(f2 - f1) < 1e-12:
                        w3 = 0.5 * (w1 + w2)
                    else:
                        w3 = w2 - f2 * (w2 - w1) / (f2 - f1)
                    s3, f3, loss3 = energy_balance_state(w3)
                else:
                    w3 = 0.5 * (w1 + w2)
                    s3, f3, loss3 = energy_balance_state(w3)
            except Exception:
                w3 = 0.5 * (w1 + w2)
                s3, f3, loss3 = energy_balance_state(w3)

        if abs(f3) < tol:
            # Ensure the accepted iterate is subcritical; if not, try midpoint then fail
            if getattr(s3, 'Froude', 0.0) >= FR_MAX:
                w3 = 0.5 * (w1 + w2)
                s3, f3, loss3 = energy_balance_state(w3)
                if getattr(s3, 'Froude', 0.0) >= FR_MAX:
                    # Fallback: compute critical depth and return that state
                    wc = solve_critical_depth(xs_up, Q_total_up, z_guess=w3)
                    return compute_state(xs_up, wc, Q_total_up)
            return s3

        # Maintain a bracket when possible
        if f1 * f3 <= 0:
            # root is between w1 and w3
            w2, s2, f2 = w3, s3, f3
        else:
            # root is between w3 and w2
            w1, s1, f1 = w3, s3, f3

        # If bracket is sufficiently small, perform bisection until convergence
        if abs(w2 - w1) < 1e-6:
            w_mid = 0.5 * (w1 + w2)
            s_mid = compute_state(xs_up, w_mid, Q_total_up)
            if getattr(s_mid, 'Froude', 0.0) >= FR_MAX:
                # Fallback to critical depth
                wc = solve_critical_depth(xs_up, Q_total_up, z_guess=w_mid)
                return compute_state(xs_up, wc, Q_total_up)
            return s_mid

    # If not converged, return the latest safe iterate if subcritical else raise
    if getattr(s2, 'Froude', 0.0) >= FR_MAX:
        wc = solve_critical_depth(xs_up, Q_total_up, z_guess=0.5 * (w1 + w2))
        return compute_state(xs_up, wc, Q_total_up)
    return s2


def solve_energy_upstream_scipy(
    xs_dn: CrossSection, xs_up: CrossSection,
    z_dn: float, z_up: float,
    Q_total_dn: float, Q_total_up: float,
    s_dn: SectionState,
    link: ReachLink,
    wse_up_init: Optional[float] = None,
    tol: float = 1e-6,
    max_iter: int = 60
) -> SectionState:
    """
    SciPy-backed solver for the upstream WSE using a robust bracketing + Brent root-finder.
    Falls back to the pure-Python `solve_energy_upstream` if SciPy is not available or bracketing fails.
    """
    if not HAVE_SCIPY:
        # SciPy not available — fallback to native solver
        return solve_energy_upstream(xs_dn, xs_up, z_dn, z_up, Q_total_dn, Q_total_up, s_dn, link, wse_up_init, tol, max_iter)

    # Residual function for brentq: returns lhs - rhs as in energy_balance_state
    def residual(w_try: float) -> float:
        s_try = compute_state(xs_up, w_try, Q_total_up)
        y_dn = s_dn.wse - z_dn
        y_up = s_try.wse - z_up
        loss = head_loss(s_dn, s_try, link, xs_dn)
        lhs = z_dn + y_dn + (s_dn.alpha * s_dn.V_t ** 2) / (2.0 * G)
        rhs = z_up + y_up + (s_try.alpha * s_try.V_t ** 2) / (2.0 * G) + loss
        f = lhs - rhs
        # Treat supercritical trial states as invalid by returning a large residual
        FR_MAX = 0.9999
        if getattr(s_try, 'Froude', 0.0) >= FR_MAX or math.isnan(f):
            return math.copysign(1e6, f if not math.isnan(f) else 1.0)
        return f

    # initial bracket
    if wse_up_init is None:
        # Prefer normal-depth estimate for a sensible bracket (HEC-RAS-like)
        try:
            Ldw = link.discharge_weighted_length(s_dn.Q_lob, s_dn.Q_ch, s_dn.Q_rob)
            if Ldw > 1e-6:
                S0_est = max(1e-8, (z_dn - z_up) / Ldw)
                w_nd = solve_normal_depth(xs_up, Q_total_up, S0_est)
                wse_up_init = max(w_nd, z_up + 1e-3)
            else:
                wse_up_init = max(z_up + 0.5, s_dn.wse)
        except Exception:
            wse_up_init = max(z_up + 0.5, s_dn.wse)

    zmin_up = min(z for _, z in xs_up.geometry)
    # build a bracket anchored at a conservative lower bound (near normal depth) and an upper bound
    w_lo = max(zmin_up + 1e-3, min(wse_up_init, s_dn.wse) - 20.0)
    w_hi = max(wse_up_init + 20.0, s_dn.wse + 50.0)

    # Ensure monotonic bracket with opposite signs; sample if necessary
    f_lo = residual(w_lo)
    f_hi = residual(w_hi)
    if f_lo * f_hi > 0:
        # try expanding upward; if this fails, sample a wide range to locate a bracket
        found = False
        for factor in [1.0, 2.0, 5.0, 10.0]:
            w_try = w_hi + factor * (w_hi - w_lo + 1.0)
            f_try = residual(w_try)
            if f_lo * f_try <= 0:
                w_hi, f_hi = w_try, f_try
                found = True
                break
        if not found:
            # try moving lower bound down if possible
            for delta in [1.0, 5.0, 10.0]:
                w_try = max(zmin_up + 1e-3, w_lo - delta)
                f_try = residual(w_try)
                if f_try * f_hi <= 0:
                    w_lo, f_lo = w_try, f_try
                    found = True
                    break
        if not found:
            # Aggressive sampling across an expanded window to find any sign change
            def _sample_find_bracket(lo: float, hi: float, samples: int = 81):
                if hi <= lo:
                    return None
                step = (hi - lo) / float(samples - 1)
                prev_f = None
                prev_w = None
                prev_s = None
                for i in range(samples):
                    w = lo + i * step
                    s_try = compute_state(xs_up, w, Q_total_up)
                    f_try = None
                    try:
                        y_dn = s_dn.wse - z_dn
                        y_up = s_try.wse - z_up
                        loss = head_loss(s_dn, s_try, link, xs_dn)
                        lhs = z_dn + y_dn + (s_dn.alpha * s_dn.V_t ** 2) / (2.0 * G)
                        rhs = z_up + y_up + (s_try.alpha * s_try.V_t ** 2) / (2.0 * G) + loss
                        f_try = lhs - rhs
                    except Exception:
                        f_try = None
                    # skip invalid residuals
                    if f_try is None or abs(f_try) >= 9e5 or math.isnan(f_try) or getattr(s_try, 'Froude', 0.0) >= 0.9999:
                        prev_f = None
                        prev_w = None
                        prev_s = None
                        continue
                    if prev_f is not None and prev_f * f_try <= 0:
                        # ensure both ends subcritical
                        if getattr(prev_s, 'Froude', 0.0) < 0.9999 and getattr(s_try, 'Froude', 0.0) < 0.9999:
                            return (prev_w, w)
                    prev_f = f_try
                    prev_s = s_try
                    prev_w = w
                return None

            w_hi2 = max(w_hi + 200.0, s_dn.wse + 200.0)
            bracket = _sample_find_bracket(w_lo, w_hi2, samples=201)
            if bracket is not None:
                w_lo, w_hi = bracket
                f_lo = residual(w_lo)
                f_hi = residual(w_hi)
                found = True
        if not found:
            # Extended attempt: dense sampling and local root refinement before giving up
            try:
                w_hi2 = max(w_hi + 500.0, s_dn.wse + 500.0)
                samples = 401
                step = (w_hi2 - w_lo) / float(samples - 1) if w_hi2 > w_lo else 1.0
                valid_pts = []
                for i in range(samples):
                    w = w_lo + i * step
                    try:
                        s_try = compute_state(xs_up, w, Q_total_up)
                        y_dn = s_dn.wse - z_dn
                        y_up = s_try.wse - z_up
                        loss = head_loss(s_dn, s_try, link, xs_dn)
                        f_try = (z_dn + y_dn + (s_dn.alpha * s_dn.V_t ** 2) / (2.0 * G)) - (z_up + y_up + (s_try.alpha * s_try.V_t ** 2) / (2.0 * G) + loss)
                    except Exception:
                        continue
                    if f_try is None or abs(f_try) >= 9e5 or math.isnan(f_try) or getattr(s_try, 'Froude', 0.0) >= 0.9999:
                        continue
                    valid_pts.append((w, f_try, s_try))

                # Try to find adjacent sign change among valid samples
                bracket_found = None
                for a, b in zip(valid_pts, valid_pts[1:]):
                    wa, fa, sa = a
                    wb, fb, sb = b
                    if fa * fb <= 0 and getattr(sa, 'Froude', 0.0) < 0.9999 and getattr(sb, 'Froude', 0.0) < 0.9999:
                        bracket_found = (wa, wb)
                        break

                if bracket_found is not None:
                    w_lo, w_hi = bracket_found
                    f_lo = residual(w_lo)
                    f_hi = residual(w_hi)
                    found = True
                else:
                    # No sign change found: attempt local refinement around smallest |f|
                    if valid_pts:
                        # pick w with minimum absolute residual
                        w_min, f_min, s_min = min(valid_pts, key=lambda t: abs(t[1]))
                        # Try a Newton/secant refinement starting at w_min
                        try:
                            w_root = _opt.newton(residual, w_min, tol=tol, maxiter=max_iter)
                            s_root = compute_state(xs_up, w_root, Q_total_up)
                            if getattr(s_root, 'Froude', 0.0) < 0.9999 and s_root.K_t > 1e-6 and s_root.A_t > 1e-6:
                                return s_root
                        except Exception:
                            # If newton fails, try small bracketing around w_min
                            delta = max(0.5, step * 2.0)
                            w_lo_try = max(zmin_up + 1e-3, w_min - delta)
                            w_hi_try = w_min + delta
                            try:
                                f_lo_try = residual(w_lo_try)
                                f_hi_try = residual(w_hi_try)
                                if f_lo_try * f_hi_try <= 0:
                                    w_root = _opt.brentq(residual, w_lo_try, w_hi_try, xtol=tol, maxiter=max_iter)
                                    s_root = compute_state(xs_up, w_root, Q_total_up)
                                    if getattr(s_root, 'Froude', 0.0) < 0.9999 and s_root.K_t > 1e-6 and s_root.A_t > 1e-6:
                                        return s_root
                            except Exception:
                                pass

                # If still not found after extended attempts, try a sequence of
                # targeted bracket/newton starts around sensible anchors
                try:
                    anchors = [wse_up_init, s_dn.wse, 0.5 * (wse_up_init + s_dn.wse)]
                    # include w_min if it exists
                    if 'w_min' in locals():
                        anchors.append(w_min)
                    tried = set()
                    for a in anchors:
                        if a in tried:
                            continue
                        tried.add(a)
                        for delta in [0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 200.0]:
                            lo = max(zmin_up + 1e-3, a - delta)
                            hi = a + delta
                            try:
                                f_lo = residual(lo)
                                f_hi = residual(hi)
                            except Exception:
                                continue
                            if math.isnan(f_lo) or math.isnan(f_hi):
                                continue
                            if f_lo * f_hi <= 0:
                                try:
                                    w_root = _opt.brentq(residual, lo, hi, xtol=tol, maxiter=max_iter)
                                    s_root = compute_state(xs_up, w_root, Q_total_up)
                                    if getattr(s_root, 'Froude', 0.0) < 0.9999 and s_root.K_t > 1e-6 and s_root.A_t > 1e-6:
                                        return s_root
                                except Exception:
                                    continue
                        # Try Newton from anchor
                        try:
                            w_root = _opt.newton(residual, a, tol=tol, maxiter=max_iter)
                            s_root = compute_state(xs_up, w_root, Q_total_up)
                            if getattr(s_root, 'Froude', 0.0) < 0.9999 and s_root.K_t > 1e-6 and s_root.A_t > 1e-6:
                                return s_root
                        except Exception:
                            pass
                except Exception:
                    pass
            except Exception:
                pass
            # final fallback to native solver
            return solve_energy_upstream(xs_dn, xs_up, z_dn, z_up, Q_total_dn, Q_total_up, s_dn, link, wse_up_init, tol, max_iter)
        # Additional multi-scale sampling clustered around anchors to try
        # before abandoning SciPy path. This tries linear and denser
        # clustered samplings around sensible anchors and accepts a root
        # only if hydraulics look valid.
        def _try_clustered_samples(anchors):
            scales = [(-200.0, 200.0, 121), (-50.0, 50.0, 121), (-10.0, 10.0, 81), (-1.0, 1.0, 41)]
            for a in anchors:
                for lo_off, hi_off, samples in scales:
                    lo = max(zmin_up + 1e-3, a + lo_off)
                    hi = a + hi_off
                    if hi <= lo:
                        continue
                    step = (hi - lo) / float(samples - 1)
                    prev_f = None
                    prev_w = None
                    prev_s = None
                    for i in range(samples):
                        w = lo + i * step
                        try:
                            s_try = compute_state(xs_up, w, Q_total_up)
                            y_dn = s_dn.wse - z_dn
                            y_up = s_try.wse - z_up
                            loss = head_loss(s_dn, s_try, link, xs_dn)
                            f_try = (z_dn + y_dn + (s_dn.alpha * s_dn.V_t ** 2) / (2.0 * G)) - (z_up + y_up + (s_try.alpha * s_try.V_t ** 2) / (2.0 * G) + loss)
                        except Exception:
                            prev_f = None
                            prev_w = None
                            prev_s = None
                            continue
                        if f_try is None or abs(f_try) >= 9e5 or math.isnan(f_try) or getattr(s_try, 'Froude', 0.0) >= 0.9999 or s_try.K_t <= 1e-6 or s_try.A_t <= 1e-6:
                            prev_f = None
                            prev_w = None
                            prev_s = None
                            continue
                        if prev_f is not None and prev_f * f_try <= 0:
                            # found adjacent sign change
                            try:
                                w_root = _opt.brentq(residual, prev_w, w, xtol=tol, maxiter=max_iter)
                                s_root = compute_state(xs_up, w_root, Q_total_up)
                                if getattr(s_root, 'Froude', 0.0) < 0.9999 and s_root.K_t > 1e-6 and s_root.A_t > 1e-6:
                                    return s_root
                            except Exception:
                                pass
                        prev_f = f_try
                        prev_s = s_try
                        prev_w = w
            return None

        anchors = [wse_up_init, s_dn.wse]
        try:
            # include previously found w_min if present
            if 'w_min' in locals():
                anchors.append(w_min)
        except Exception:
            pass

        s_try_root = _try_clustered_samples(anchors)
        if s_try_root is not None:
            return s_try_root

    # Use Brent's method for robust root finding
    try:
        w_root = _opt.brentq(residual, w_lo, w_hi, xtol=tol, maxiter=max_iter)
    except Exception:
        # fallback to native solver on failure
        return solve_energy_upstream(xs_dn, xs_up, z_dn, z_up, Q_total_dn, Q_total_up, s_dn, link, wse_up_init, tol, max_iter)

    # return computed state
    s_root = compute_state(xs_up, w_root, Q_total_up)
    # ensure we return a subcritical state; if supercritical, try to locate alternate bracket and re-run
    FR_MAX = 0.9999
    if getattr(s_root, 'Froude', 0.0) >= FR_MAX:
        # attempt to find alternative bracket by sampling more densely in an expanded window
        try:
            w_lo2 = max(zmin_up + 1e-3, w_lo - 50.0)
            w_hi2 = max(w_hi + 200.0, s_dn.wse + 200.0)
            def _sample_and_try(lo, hi, samples=201):
                if hi <= lo:
                    return None
                step = (hi - lo) / float(samples - 1)
                prev_f = None
                prev_w = None
                prev_s = None
                for i in range(samples):
                    w = lo + i * step
                    s_try = compute_state(xs_up, w, Q_total_up)
                    # compute residual directly
                    try:
                        y_dn = s_dn.wse - z_dn
                        y_up = s_try.wse - z_up
                        loss = head_loss(s_dn, s_try, link, xs_dn)
                        f_try = (z_dn + y_dn + (s_dn.alpha * s_dn.V_t ** 2) / (2.0 * G)) - (z_up + y_up + (s_try.alpha * s_try.V_t ** 2) / (2.0 * G) + loss)
                    except Exception:
                        f_try = None
                    if f_try is None or abs(f_try) >= 9e5 or math.isnan(f_try) or getattr(s_try, 'Froude', 0.0) >= FR_MAX:
                        prev_f = None
                        prev_w = None
                        prev_s = None
                        continue
                    if prev_f is not None and prev_f * f_try <= 0:
                        if getattr(prev_s, 'Froude', 0.0) < FR_MAX and getattr(s_try, 'Froude', 0.0) < FR_MAX:
                            return (prev_w, w)
                    prev_f = f_try
                    prev_s = s_try
                    prev_w = w
                return None

            bracket = _sample_and_try(w_lo2, w_hi2, samples=201)
            if bracket is not None:
                try:
                    w_new_lo, w_new_hi = bracket
                    w_root = _opt.brentq(residual, w_new_lo, w_new_hi, xtol=tol, maxiter=max_iter)
                    s_root = compute_state(xs_up, w_root, Q_total_up)
                    if getattr(s_root, 'Froude', 0.0) < FR_MAX:
                        return s_root
                except Exception:
                    pass
        except Exception:
            pass
        # Fallback: compute critical depth if no subcritical root found
        wc = solve_critical_depth(xs_up, Q_total_up, z_guess=wse_up_init)
        return compute_state(xs_up, wc, Q_total_up)


def solve_normal_depth(xs: CrossSection, Q_total: float, S0: float, z_guess: Optional[float] = None) -> float:
    """
    Solve for WSE producing Sf_total ≈ S0 at this cross section (uniform depth assumption).
    Brackets WSE between min bed and (max bed + headroom).
    """
    zmin = min(z for _, z in xs.geometry)
    zmax = max(z for _, z in xs.geometry)
    a = zmin + 0.01
    b = zmax + 100.0  # headroom
    fa = compute_state(xs, a, Q_total).Sf_total - S0
    fb = compute_state(xs, b, Q_total).Sf_total - S0
    if fa * fb > 0:
        # Fallback: try a simple incremental search upward
        w = zmin + 1.0
        for _ in range(200):
            Sf = compute_state(xs, w, Q_total).Sf_total
            if Sf <= S0:
                return w
            w += 0.5
        return w
    # Bisection
    for _ in range(60):
        m = 0.5 * (a + b)
        fm = compute_state(xs, m, Q_total).Sf_total - S0
        if abs(fm) < 1e-6:
            return m
        if fa * fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5 * (a + b)


def solve_critical_depth(xs: CrossSection, Q_total: float, z_guess: Optional[float] = None) -> float:
    """
    Find the WSE producing critical flow (Froude == 1) at this cross section.
    Uses bisection on Froude-1 computed via `compute_state`. Returns WSE elevation.
    """
    zmin = min(z for _, z in xs.geometry)
    zmax = max(z for _, z in xs.geometry)
    lo = zmin + 1e-4
    hi = zmax + 50.0 if z_guess is None else max(z_guess + 1.0, zmax + 1.0)

    def froude_minus_one(w: float) -> float:
        return getattr(compute_state(xs, w, Q_total), 'Froude', 0.0) - 1.0

    f_lo = froude_minus_one(lo)
    f_hi = froude_minus_one(hi)
    # expand hi until sign change or limit reached
    attempts = 0
    while f_lo * f_hi > 0 and attempts < 8:
        hi += 50.0 + attempts * 50.0
        f_hi = froude_minus_one(hi)
        attempts += 1

    if f_lo * f_hi > 0:
        # fallback: scan to find w minimizing |Froude-1|
        best_w = lo
        best_err = abs(f_lo)
        for i in range(1, 201):
            w = lo + (hi - lo) * (i / 200.0)
            err = abs(froude_minus_one(w))
            if err < best_err:
                best_err = err
                best_w = w
        return best_w

    # bisection
    a, b = lo, hi
    fa, fb = f_lo, f_hi
    for _ in range(60):
        m = 0.5 * (a + b)
        fm = froude_minus_one(m)
        if abs(fm) < 1e-6:
            return m
        if fa * fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5 * (a + b)


# ---------------------------------------------------------------------------
# Input model & driver
# ---------------------------------------------------------------------------

@dataclass
class ModelInput:
    flow_cfs: float
    flow_change: Optional[Dict]  # {"at_index": int, "delta_cfs": float}
    boundary_condition: str      # "known_wse" or "normal_depth"
    boundary_value: float        # WSE (ft) for known_wse, or S0 for normal_depth
    sections: List[CrossSection] = field(default_factory=list)


def _parse_river_station_value(river_station) -> Optional[float]:
    """Parse numeric river station value from common string formats."""
    if river_station is None:
        return None
    if isinstance(river_station, (int, float)):
        return float(river_station)

    text = str(river_station).strip()
    if not text:
        return None

    try:
        return float(text)
    except Exception:
        pass

    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if match:
        try:
            return float(match.group(0))
        except Exception:
            return None
    return None


def _sorted_sections_by_river_station(sections: List[CrossSection]) -> List[CrossSection]:
    """Return sections sorted DS->US by numeric river station ascending."""
    parsed = [(_parse_river_station_value(xs.river_station), i, xs) for i, xs in enumerate(sections)]
    if not any(val is not None for val, _, _ in parsed):
        return list(sections)

    # Numeric stations sort first ascending; non-numeric keep original relative order at the end.
    return [
        xs for _, _, xs in sorted(
            parsed,
            key=lambda item: (float('inf') if item[0] is None else item[0], item[1])
        )
    ]


def load_input(path: str) -> ModelInput:
    """Load model from a GeoPackage file."""
    return load_from_geopackage(path)


def load_from_geopackage(path: str, cross_layer: str = 'cross_sections', centerline_layer: str = 'centerline', boundary_layer: str = 'boundary_conditions') -> ModelInput:
    """Load model from a GeoPackage (geopandas/shapely required)."""
    try:
        import geopandas as gpd
        from shapely.geometry import LineString
    except Exception as e:
        raise RuntimeError('geopandas/shapely required to load GeoPackage') from e

    cross_gdf = gpd.read_file(path, layer=cross_layer)
    centerline = None
    try:
        centerline_gdf = gpd.read_file(path, layer=centerline_layer)
        if len(centerline_gdf) > 0:
            centerline = centerline_gdf.geometry.iloc[0]
    except Exception:
        centerline = None

    rows = []

    def _safe_float(value, default=0.0):
        try:
            v = float(value)
            if math.isnan(v):
                return float(default)
            return v
        except Exception:
            return float(default)

    def _safe_int(value, default=0):
        try:
            v = float(value)
            if math.isnan(v):
                return int(default)
            return int(v)
        except Exception:
            return int(default)

    def _safe_shape(value):
        if value is None:
            return None
        try:
            if isinstance(value, float) and math.isnan(value):
                return None
        except Exception:
            pass
        text = str(value).strip()
        if not text or text.lower() in ('nan', 'none', 'null'):
            return None
        return text
    for _, feat in cross_gdf.iterrows():
        geom = feat.geometry
        coords = list(geom.coords)
        # Determine dimensionality: if coords are 3D (x,y,z) use x,y for planimetric distances and z for elevation.
        # If coords are 2D, assume they are (x,elevation) as produced by this plugin; compute station as cumulative
        # distance along the polyline using x differences and take elevation from the second component.
        geom_list = []
        if not coords:
            geom_list = []
        else:
            # Check if any coord has length >=3
            is_3d = any(len(c) >= 3 for c in coords)
            stations = [0.0]
            elevations = []
            if is_3d:
                # use planar (x,y) for distances, z for elevation
                prev_xy = (float(coords[0][0]), float(coords[0][1]) if len(coords[0]) >= 2 else 0.0)
                elevations.append(float(coords[0][2]) if len(coords[0]) >= 3 else 0.0)
                for c in coords[1:]:
                    x = float(c[0])
                    y = float(c[1]) if len(c) >= 2 else 0.0
                    z = float(c[2]) if len(c) >= 3 else 0.0
                    dx = x - prev_xy[0]
                    dy = y - prev_xy[1]
                    dist = math.hypot(dx, dy)
                    stations.append(stations[-1] + dist)
                    elevations.append(z)
                    prev_xy = (x, y)
            else:
                # 2D coords: interpret as (x, elevation) where x is along-section coordinate.
                prev_x = float(coords[0][0])
                elevations.append(float(coords[0][1]) if len(coords[0]) >= 2 else 0.0)
                for c in coords[1:]:
                    x = float(c[0])
                    z = float(c[1]) if len(c) >= 2 else 0.0
                    dist = abs(x - prev_x)
                    stations.append(stations[-1] + dist)
                    elevations.append(z)
                    prev_x = x

            geom_list = [(float(s), float(z)) for s, z in zip(stations, elevations)]

        river_station = feat.get('river_station') or str(feat.name if hasattr(feat, 'name') else '')
        lb = feat.get('left_bank_station', 0.0)
        rb = feat.get('right_bank_station', 0.0)
        n_lob = feat.get('n_lob', 0.035)
        n_ch = feat.get('n_ch', 0.035)
        n_rob = feat.get('n_rob', 0.035)
        rows.append({
            'river_station': river_station,
            'geometry': geom_list,
            'left_bank_station': _safe_float(lb, 0.0),
            'right_bank_station': _safe_float(rb, 0.0),
            'n_lob': _safe_float(n_lob, 0.035),
            'n_ch': _safe_float(n_ch, 0.035),
            'n_rob': _safe_float(n_rob, 0.035),
            'contraction_coeff': _safe_float(feat.get('contraction_coeff', 0.1), 0.1),
            'expansion_coeff': _safe_float(feat.get('expansion_coeff', 0.3), 0.3),
            'L_lob_to_next': _safe_float(feat.get('L_lob_to_next', 0.0), 0.0),
            'L_ch_to_next': _safe_float(feat.get('L_ch_to_next', 0.0), 0.0),
            'L_rob_to_next': _safe_float(feat.get('L_rob_to_next', 0.0), 0.0),
            'culvert_code': _safe_int(feat.get('culvert_code', 0), 0),
            'culvert_shape': _safe_shape(feat.get('culvert_shape', None)),
            'culvert_diameter': _safe_float(feat.get('culvert_diameter', 0.0), 0.0),
            'culvert_width': _safe_float(feat.get('culvert_width', 0.0), 0.0),
            'culvert_height': _safe_float(feat.get('culvert_height', 0.0), 0.0),
            'culvert_upstream_invert': _safe_float(feat.get('culvert_upstream_invert', 0.0), 0.0),
            'culvert_downstream_invert': _safe_float(feat.get('culvert_downstream_invert', 0.0), 0.0),
            'culvert_length': _safe_float(feat.get('culvert_length', 0.0), 0.0),
            'culvert_weir_coeff': _safe_float(feat.get('culvert_weir_coeff', 3.0), 3.0),
            'culvert_weir_sta_left': _safe_float(feat.get('culvert_weir_sta_left', 0.0), 0.0),
            'culvert_weir_sta_right': _safe_float(feat.get('culvert_weir_sta_right', 0.0), 0.0),
            'culvert_slope_legacy': _safe_float(feat.get('culvert_slope', 0.0), 0.0),
        })

    if centerline is not None:
        chainages = []
        for r in rows:
            pts = r['geometry']
            if not pts:
                chainages.append(0.0); continue
            try:
                proj = centerline.project(LineString([(p[0], 0.0) for p in pts]).centroid)
            except Exception:
                proj = 0.0
            chainages.append(proj)
        for r,c in zip(rows, chainages):
            r['chainage'] = c
        rows = sorted(rows, key=lambda x: x.get('chainage', 0.0))

    sections = []
    for i, r in enumerate(rows):
        xs = CrossSection(
            river_station=str(r.get('river_station','')),
            geometry=[(float(x), float(z)) for x,z in r['geometry']],
            left_bank_station=float(r.get('left_bank_station',0.0)),
            right_bank_station=float(r.get('right_bank_station',0.0)),
            n_lob=float(r.get('n_lob',0.035)), n_ch=float(r.get('n_ch',0.035)), n_rob=float(r.get('n_rob',0.035)),
            contraction_coeff=float(r.get('contraction_coeff',0.1)), expansion_coeff=float(r.get('expansion_coeff',0.3)),
            L_lob_to_next=float(r.get('L_lob_to_next', 0.0)),
            L_ch_to_next=float(r.get('L_ch_to_next', 0.0)),
            L_rob_to_next=float(r.get('L_rob_to_next', 0.0)),
            culvert_code=int(r.get('culvert_code', 0)),
            culvert_shape=r.get('culvert_shape', None),
            culvert_diameter=float(r.get('culvert_diameter', 0.0)),
            culvert_width=float(r.get('culvert_width', 0.0)),
            culvert_height=float(r.get('culvert_height', 0.0)),
            culvert_upstream_invert=float(r.get('culvert_upstream_invert', 0.0)),
            culvert_downstream_invert=float(r.get('culvert_downstream_invert', 0.0)),
            culvert_length=float(r.get('culvert_length', 0.0)),
            culvert_weir_coeff=float(r.get('culvert_weir_coeff', 3.0)),
            culvert_weir_sta_left=float(r.get('culvert_weir_sta_left', 0.0)),
            culvert_weir_sta_right=float(r.get('culvert_weir_sta_right', 0.0)),
        )
        # Backward compatibility with older GeoPackages that only stored culvert_slope.
        if xs.culvert_length <= 0.0:
            legacy_slope = float(r.get('culvert_slope_legacy', 0.0) or 0.0)
            if legacy_slope > 0.0:
                xs.culvert_length = max(1.0, xs.L_ch_to_next if xs.L_ch_to_next > 0.0 else 1.0)
                xs.culvert_upstream_invert = z_invert = min((z for _, z in xs.geometry), default=0.0)
                xs.culvert_downstream_invert = xs.culvert_upstream_invert - legacy_slope * xs.culvert_length
        sections.append(xs)

    if len(sections) > 1:
        if all('chainage' in r for r in rows):
            for i in range(len(sections)-1):
                # only overwrite L_ch_to_next if it wasn't provided in attributes
                if not sections[i].L_ch_to_next:
                    sections[i].L_ch_to_next = float(rows[i+1]['chainage'] - rows[i]['chainage'])
        else:
            for i in range(len(sections)-1):
                if not sections[i].L_ch_to_next:
                    sections[i].L_ch_to_next = 1.0

    flow_cfs = 0.0
    boundary_condition = 'known_wse'
    boundary_value = 0.0
    try:
        bd = gpd.read_file(path, layer=boundary_layer)
        if len(bd) > 0:
            row = bd.iloc[0]
            flow_cfs = float(row.get('flow_cfs', 0.0))
            boundary_condition = row.get('boundary_type', 'known_wse')
            boundary_value = float(row.get('boundary_value', 0.0))
    except Exception:
        pass

    return ModelInput(flow_cfs=flow_cfs, flow_change=None, boundary_condition=boundary_condition, boundary_value=boundary_value, sections=sections)


def save_to_geopackage(path: str, model: ModelInput, centerline_geom=None, overwrite: bool = True):
    try:
        import geopandas as gpd
        from shapely.geometry import LineString
        import fiona
    except Exception as e:
            raise RuntimeError('geopandas/fiona/shapely required to save GeoPackage') from e

    rows = []
    for xs in model.sections:
        coords = [(float(x), float(z)) for x,z in xs.geometry]
        geom = LineString([(x, z) for x,z in coords])
        rows.append({
            'geometry': geom,
            'river_station': xs.river_station,
            'left_bank_station': xs.left_bank_station,
            'right_bank_station': xs.right_bank_station,
            'n_lob': xs.n_lob, 'n_ch': xs.n_ch, 'n_rob': xs.n_rob,
            'contraction_coeff': xs.contraction_coeff,
            'expansion_coeff': xs.expansion_coeff,
            'L_lob_to_next': xs.L_lob_to_next,
            'L_ch_to_next': xs.L_ch_to_next,
            'L_rob_to_next': xs.L_rob_to_next,
            'culvert_code': xs.culvert_code,
            'culvert_shape': xs.culvert_shape or '',
            'culvert_diameter': xs.culvert_diameter,
            'culvert_width': xs.culvert_width,
            'culvert_height': xs.culvert_height,
            'culvert_upstream_invert': xs.culvert_upstream_invert,
            'culvert_downstream_invert': xs.culvert_downstream_invert,
            'culvert_length': xs.culvert_length,
            'culvert_weir_coeff': xs.culvert_weir_coeff,
            'culvert_weir_sta_left': xs.culvert_weir_sta_left,
            'culvert_weir_sta_right': xs.culvert_weir_sta_right,
            # Preserve legacy slope column for backward compatibility.
            'culvert_slope': xs.culvert_slope(),
        })

    gdf = gpd.GeoDataFrame(rows, geometry='geometry', crs=None)
    mode = 'w' if overwrite else 'a'
    gdf.to_file(path, layer='cross_sections', driver='GPKG', index=False, mode=mode)

    if centerline_geom is not None:
        cgdf = gpd.GeoDataFrame([{'geometry': centerline_geom}], geometry='geometry')
        cgdf.to_file(path, layer='centerline', driver='GPKG', index=False, mode='a')

    try:
        import fiona
        schema = {'geometry': 'None', 'properties': {'boundary_type': 'str', 'boundary_value': 'float', 'flow_cfs': 'float'}}
        with fiona.open(path, mode='a', driver='GPKG', layer='boundary_conditions', schema=schema) as dst:
            props = {'boundary_type': model.boundary_condition, 'boundary_value': float(model.boundary_value), 'flow_cfs': float(model.flow_cfs)}
            dst.write({'properties': props})
    except Exception:
        pass


def run_backwater(model: ModelInput, solver: str = 'py'):
    """
    Execute standard-step solution from downstream to upstream.
    Sections must be ordered from downstream (index 0) to upstream (index N-1).
    """

    if len(model.sections) < 2:
        raise ValueError("At least two cross sections are required.")

    # Enforce canonical order: downstream -> upstream by ascending numeric river station.
    # Convention: lowest river station = most downstream, highest = most upstream.
    original_sections = list(model.sections)
    ordered_sections = _sorted_sections_by_river_station(model.sections)

    def _min_bed(xs: CrossSection) -> float:
        return min(z for _, z in xs.geometry)

    if ordered_sections != original_sections:
        model.sections = ordered_sections
        if model.flow_change is not None:
            try:
                orig_idx = int(model.flow_change["at_index"])
                if 0 <= orig_idx < len(original_sections):
                    target_section = original_sections[orig_idx]
                    new_idx = model.sections.index(target_section)
                    model.flow_change = dict(model.flow_change)
                    model.flow_change["at_index"] = new_idx
            except Exception:
                pass

    # Build per-section total Q considering optional flow change at an index (applied upstream of that index)
    Q_base = model.flow_cfs
    Q_per_section = [Q_base for _ in model.sections]
    if model.flow_change is not None:
        idx = int(model.flow_change["at_index"])
        delta = float(model.flow_change["delta_cfs"])
        for i in range(idx, len(model.sections)):
            Q_per_section[i] = Q_base + delta

    # Downstream boundary
    xs_dn = model.sections[0]
    z_dn = _min_bed(xs_dn)
    if model.boundary_condition == "known_wse":
        wse_dn = model.boundary_value
        if wse_dn < z_dn:
            raise ValueError(
                f"Downstream WSE ({wse_dn:.3f}) is below the minimum bed elevation of the "
                f"downstream section (RS {xs_dn.river_station}, bed={z_dn:.3f}). "
                f"Provide a downstream WSE above the channel bed."
            )
    elif model.boundary_condition == "normal_depth":
        wse_dn = solve_normal_depth(xs_dn, Q_per_section[0], model.boundary_value)
    else:
        raise ValueError("boundary_condition must be 'known_wse' or 'normal_depth'.")

    # For computed/other cases, ensure a modest headroom to avoid exactly-dry numerical states.
    zmax = max(z for _, z in xs_dn.geometry)
    headroom = max(0.01, 0.01 * (zmax - z_dn + 1e-6))
    if wse_dn < z_dn + headroom:
        wse_dn = z_dn + headroom

    # Solve downstream state
    s_dn = compute_state(xs_dn, wse_dn, Q_per_section[0])
    # Debugging: ensure downstream state was computed
    if s_dn is None:
        print(f"DEBUG: compute_state returned None for downstream section {xs_dn.river_station} with wse_dn={wse_dn}")
    else:
        print(f"DEBUG: downstream SectionState type={type(s_dn)}, wse={s_dn.wse}, K_t={s_dn.K_t}")

    # Guard against near-zero downstream conveyance which leads to (Q/K)^2 blow-up.
    if s_dn.K_t <= 1e-6:
        raise ValueError(f"Downstream section conveyance is near zero (K_t={s_dn.K_t}). Adjust downstream WSE or section geometry; solver cannot proceed.")

    results: List[SectionState] = [s_dn]

    # March upstream
    # Boundary WSE produced by culvert control at a culvert section index.
    # For the immediately upstream reach, this boundary is used without
    # reusing culvert conveyance/velocity terms in the standard-step equation.
    culvert_boundary_wse_by_index = {}
    for i in range(len(model.sections) - 1):
        xs_dn = model.sections[i]
        xs_up = model.sections[i + 1]

        # Reach lengths are stored on the DOWNSTREAM section as the distance
        # to the next upstream section. Use xs_dn's lengths for this reach.
        link = ReachLink(xs_dn.L_lob_to_next, xs_dn.L_ch_to_next, xs_dn.L_rob_to_next)

        Q_dn = Q_per_section[i]
        Q_up = Q_per_section[i + 1]

        z_dn = min(z for _, z in xs_dn.geometry)
        z_up = min(z for _, z in xs_up.geometry)

        # ------------------------------------------------------------------
        # Culvert sections act as head-loss boundaries between sub-reaches.
        # When the UPSTREAM section has a culvert, bypass the energy equation
        # entirely: solve for headwater WSE using culvert/weir equations and
        # use that as the WSE for the culvert section.  The reach above the
        # culvert then resumes with the headwater as its downstream boundary.
        # ------------------------------------------------------------------
        if xs_up.has_culvert():
            tailwater_wse = results[-1].wse
            # Pass the downstream SectionState so the outlet energy balance
            # can use the tailwater channel velocity head.
            _tw_state = results[-1] if results else None
            hw_wse = None
            control_type = 'error'
            _last_culvert_err = None
            for _attempt in range(2):
                try:
                    if _attempt == 1:
                        _ensure_culvert_runtime()
                    hw_wse, control_type = solve_culvert_headwater(
                        xs_up,
                        tailwater_wse,
                        Q_up,
                        tailwater_state=_tw_state,
                        downstream_boundary_xs=xs_dn,
                        boundary_reach_length=link.L_ch,
                    )
                    _last_culvert_err = None
                    break
                except Exception as _ce:
                    _last_culvert_err = _ce

            if _last_culvert_err is not None:
                print(
                    f"WARNING: solve_culvert_headwater failed for section {xs_up.river_station}: "
                    f"{_last_culvert_err}; falling back to energy equation"
                )

            if hw_wse is not None:
                # Keep the handoff state physically/numerically valid for the
                # subcritical standard-step solver. Culvert control can return
                # a HW near barrel invert that is below (or effectively at)
                # the section bed in the geometric XS, which yields near-dry
                # states and unstable upstream solves.
                #
                # IMPORTANT: do not always force critical depth here. For
                # culvert/weir-controlled sections, the physically governing
                # headwater can be subcritical and below the section's
                # critical-depth WSE. Forcing to critical would overwrite a
                # valid culvert answer (e.g., RS 89.3 calibration).
                hw_wse_for_section = max(hw_wse, z_up + 1e-3)

                # Only apply critical-depth fallback for near-dry handoffs.
                # This preserves numerical stability without biasing normal
                # culvert control solutions upward.
                if hw_wse_for_section <= z_up + 0.05:
                    try:
                        wse_crit_up = solve_critical_depth(xs_up, Q_up, z_guess=max(hw_wse_for_section, z_up + 0.5))
                        hw_wse_for_section = max(hw_wse_for_section, wse_crit_up)
                    except Exception:
                        pass
                s_up = compute_state(xs_up, hw_wse_for_section, Q_up)
                print(f"  Culvert at section {xs_up.river_station} ({control_type} control): "
                      f"TW={tailwater_wse:.3f} ft, HW={hw_wse:.3f} ft, "
                      f"HW_xs={hw_wse_for_section:.3f} ft, Q={Q_up:.2f} cfs")
                results.append(s_up)
                culvert_boundary_wse_by_index[i + 1] = hw_wse_for_section
                continue   # skip energy-equation block for this reach

        # If this reach starts at a culvert section that was solved as a control
        # boundary, avoid using culvert XS conveyance/velocity in the energy
        # equation handoff to the next upstream cross section.
        use_culvert_boundary_only = (
            xs_dn.has_culvert()
            and i in culvert_boundary_wse_by_index
        )

        s_dn_for_solver = results[-1]
        xs_dn_for_solver = xs_dn
        z_dn_for_solver = z_dn

        if use_culvert_boundary_only:
            wse_boundary = culvert_boundary_wse_by_index[i]
            z_dn_for_solver = wse_boundary
            xs_dn_for_solver = xs_up
            try:
                # Downstream boundary state for the next reach should reflect a
                # water-level boundary, not the culvert-section conveyance.
                s_dn_for_solver = compute_state(xs_up, wse_boundary, Q_dn)
            except Exception:
                s_dn_for_solver = SectionState(
                    wse=wse_boundary,
                    depth_at_min=0.0,
                    alpha=1.0,
                    A_lob=0.0,
                    A_ch=0.0,
                    A_rob=0.0,
                    K_lob=0.0,
                    K_ch=0.0,
                    K_rob=0.0,
                    Q_lob=0.0,
                    Q_ch=Q_dn,
                    Q_rob=0.0,
                    V_t=0.0,
                    K_t=1.0e12,
                    A_t=1.0e12,
                    Sf_total=0.0,
                    Froude=0.0,
                )

        # starting guess upstream WSE
        # Debug: show loop index, boundary condition, and last result repr
        try:
            print(f"DEBUG: loop i={i}, bc={getattr(model,'boundary_condition',None)}, last_result={repr(results[-1])}")
        except Exception:
            print(f"DEBUG: loop i={i}, unable to repr last result")
        # For the first reach (i==0) prefer the downstream boundary condition
        # as the initial guess. If the boundary was a known WSE, use that
        # value; if it was specified as normal depth, compute the downstream
        # cross-section's normal-depth WSE and use it as the initial guess.
        if i == 0:
            bc = getattr(model, 'boundary_condition', '')
            if bc == 'known_wse':
                wse_up_guess = float(model.boundary_value)
            elif bc == 'normal_depth':
                try:
                    wse_up_guess = solve_normal_depth(xs_dn, Q_per_section[0], float(model.boundary_value))
                except Exception:
                    wse_up_guess = results[-1].wse
            else:
                wse_up_guess = results[-1].wse
        else:
            wse_up_guess = s_dn_for_solver.wse  # start near previous

        # The initial guess must be above the upstream section's bed. If the
        # previous WSE is below xs_up's bed (e.g. when the channel rises
        # sharply), seed the guess at bed + a small headroom so the solver
        # starts in a physically valid state.
        wse_up_guess = max(wse_up_guess, z_up + max(0.01, 0.01 * (z_up - z_dn + 1e-6)))

        if solver in ('scipy', 'numpy'):
            s_up = solve_energy_upstream_scipy(
                xs_dn=xs_dn_for_solver, xs_up=xs_up,
                z_dn=z_dn_for_solver, z_up=z_up,
                Q_total_dn=Q_dn, Q_total_up=Q_up,
                s_dn=s_dn_for_solver,
                link=link,
                wse_up_init=wse_up_guess
            )
            # If SciPy-backed solver returned None, fall back to pure-Python solver
            if s_up is None:
                print(f"WARNING: SciPy solver returned None for reach {i} ({xs_dn.river_station} -> {xs_up.river_station}), falling back to native solver")
                s_up = solve_energy_upstream(
                    xs_dn=xs_dn_for_solver, xs_up=xs_up,
                    z_dn=z_dn_for_solver, z_up=z_up,
                    Q_total_dn=Q_dn, Q_total_up=Q_up,
                    s_dn=s_dn_for_solver,
                    link=link,
                    wse_up_init=wse_up_guess
                )
                if s_up is None:
                    raise RuntimeError(f"Upstream solver returned None for reach {i} ({xs_dn.river_station} -> {xs_up.river_station})")
        else:
            s_up = solve_energy_upstream(
                xs_dn=xs_dn_for_solver, xs_up=xs_up,
                z_dn=z_dn_for_solver, z_up=z_up,
                Q_total_dn=Q_dn, Q_total_up=Q_up,
                s_dn=s_dn_for_solver,
                link=link,
                wse_up_init=wse_up_guess
            )

        results.append(s_up)

    return results




# ---------------------------------------------------------------------------
# GUI + plotting helpers
# ---------------------------------------------------------------------------
def _plot_results(model: ModelInput, results: List[SectionState]):
    if not HAVE_MPL:
        raise RuntimeError("Matplotlib required for plotting")

    # Plot each cross-section geometry with waterline
    n = len(model.sections)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(6, 2.5 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, xs, st in zip(axes, model.sections, results):
        geom = sorted(xs.geometry, key=lambda p: p[0])
        xs_x = [p[0] for p in geom]
        xs_z = [p[1] for p in geom]
        ax.plot(xs_x, xs_z, '-k', linewidth=2)
        ax.fill_between(xs_x, xs_z, min(xs_z) - 1.0, color="#f0f0f0")
        ax.axhline(st.wse, color='blue', linestyle='--', linewidth=2, label=f'WSE {st.wse:.3f} ft')
        
        # Add culvert visualization if present
        if xs.has_culvert():
            z_min = min(xs_z)
            # Draw culvert zone in the cross-section
            culvert_label = f"Culvert: {xs.culvert_shape} (code {xs.culvert_code})"
            if xs.culvert_shape == 'circular':
                culvert_label += f" D={xs.culvert_diameter} ft"
            else:
                culvert_label += f" {xs.culvert_width}x{xs.culvert_height} ft"
            
            # Highlight culvert zone with a red rectangle
            ax.axhspan(z_min, z_min + 1.0, alpha=0.2, color='red', label=culvert_label)
            ax.text(xs_x[len(xs_x)//2], z_min + 0.5, "CULVERT", 
                   ha='center', va='center', fontsize=8, fontweight='bold', color='red')
        
        ax.set_ylabel(f"RS {xs.river_station}\n(elev ft)")
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def launch_gui():
    if not HAVE_TK:
        raise RuntimeError("Tkinter not available on this system")

    root = tk.Tk()
    root.title("Backwater — Standard Step GUI")

    frm = ttk.Frame(root, padding=8)
    frm.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    input_path = tk.StringVar()
    flow_var = tk.DoubleVar(value=500.0)

    def choose_file():
        p = filedialog.askopenfilename(filetypes=[("GeoPackage files", "*.gpkg"), ("All files", "*")])
        if p:
            input_path.set(p)

    ttk.Label(frm, text="Input GeoPackage:").grid(column=0, row=0, sticky=tk.W)
    ttk.Entry(frm, textvariable=input_path, width=60).grid(column=1, row=0, sticky=(tk.W, tk.E))
    ttk.Button(frm, text="Browse...", command=choose_file).grid(column=2, row=0, sticky=tk.W)

    ds_bc_var = tk.StringVar(value="known_wse")
    ds_val_var = tk.DoubleVar(value=0.0)

    ttk.Label(frm, text="DS BC:").grid(column=0, row=1, sticky=tk.W)
    ttk.Combobox(frm, textvariable=ds_bc_var, values=("known_wse", "normal_depth"), width=20).grid(column=1, row=1, sticky=tk.W)
    ds_val_label = ttk.Label(frm, text="DS value (WSE ft):")
    ds_val_label.grid(column=0, row=2, sticky=tk.W)
    ttk.Entry(frm, textvariable=ds_val_var, width=20).grid(column=1, row=2, sticky=tk.W)

    def _on_ds_bc_change(*args):
        v = ds_bc_var.get()
        if v == 'normal_depth':
            ds_val_label.config(text='DS value (channel slope S0):')
        else:
            ds_val_label.config(text='DS value (WSE ft):')

    ds_bc_var.trace_add('write', _on_ds_bc_change)
    ttk.Label(frm, text="Flow (cfs):").grid(column=0, row=3, sticky=tk.W)
    ttk.Entry(frm, textvariable=flow_var, width=20).grid(column=1, row=3, sticky=tk.W)

    # Alpha and Sf method selectors (GUI)
    alpha_var = tk.StringVar(value=ALPHA_METHOD)
    sf_var = tk.StringVar(value=SF_METHOD)
    ttk.Label(frm, text="Alpha method:").grid(column=0, row=4, sticky=tk.W)
    ttk.Combobox(frm, textvariable=alpha_var, values=("conveyance","area"), width=20).grid(column=1, row=4, sticky=tk.W)
    ttk.Label(frm, text="Sf method:").grid(column=0, row=5, sticky=tk.W)
    ttk.Combobox(frm, textvariable=sf_var, values=("combined","avg"), width=20).grid(column=1, row=5, sticky=tk.W)

    # Output text area
    text_out = tk.Text(root, height=10)
    text_out.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E))

    # Plot canvas holder (scrollable)
    def make_scrollable_frame(parent, width=600, height=200):
        outer = ttk.Frame(parent)
        canvas = tk.Canvas(outer, width=width, height=height)
        vscroll = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview)
        hscroll = ttk.Scrollbar(outer, orient=tk.HORIZONTAL, command=canvas.xview)
        canvas.configure(yscrollcommand=vscroll.set, xscrollcommand=hscroll.set)
        inner = ttk.Frame(canvas)
        canvas.create_window((0,0), window=inner, anchor='nw')

        def _on_config(event=None):
            canvas.configure(scrollregion=canvas.bbox('all'))

        inner.bind('<Configure>', _on_config)
        canvas.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        vscroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        hscroll.grid(row=1, column=0, sticky=(tk.W, tk.E))
        return outer, inner

    canvas_holder_outer, canvas_holder_inner = make_scrollable_frame(root, width=700, height=300)
    canvas_holder_outer.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))

    fig_canvas = None

    # In-memory model state for editing
    current_model: Optional[ModelInput] = None
    current_file: Optional[str] = None
    current_results: Optional[List[SectionState]] = None

    # Section selector & editable properties
    section_var = tk.StringVar()
    section_idxs: List[int] = []

    prop_frame = ttk.Frame(frm, padding=(0, 8, 0, 8))
    prop_frame.grid(column=0, row=6, columnspan=3, sticky=(tk.W, tk.E))

    ttk.Label(prop_frame, text="Section:").grid(column=0, row=0, sticky=tk.W)
    section_cb = ttk.Combobox(prop_frame, textvariable=section_var, values=[], state='readonly', width=30)
    section_cb.grid(column=1, row=0, sticky=tk.W)

    # Property fields
    props = {}
    prop_names = [
        ("left_bank_station", "Left bank"), ("right_bank_station", "Right bank"),
        ("n_lob", "n_lob"), ("n_ch", "n_ch"), ("n_rob", "n_rob"),
        ("contraction_coeff", "Cc"), ("expansion_coeff", "Ce"),
        ("L_lob_to_next", "L_lob"), ("L_ch_to_next", "L_ch"), ("L_rob_to_next", "L_rob")
    ]

    for i, (key, label) in enumerate(prop_names, start=1):
        ttk.Label(prop_frame, text=f"{label}:").grid(column=0, row=i, sticky=tk.W)
        v = tk.DoubleVar(value=0.0)
        e = ttk.Entry(prop_frame, textvariable=v, width=12)
        e.grid(column=1, row=i, sticky=tk.W)
        props[key] = v

    # Geometry editor: treeview + controls
    geom_frame = ttk.Labelframe(frm, text="Cross-section Geometry (station, elevation)", padding=(6,6))
    geom_frame.grid(column=3, row=0, rowspan=7, sticky=(tk.N, tk.S, tk.E, tk.W), padx=(8,0))

    geom_tv = ttk.Treeview(geom_frame, columns=("station","elevation"), show='headings', height=12)
    geom_tv.heading('station', text='Station')
    geom_tv.heading('elevation', text='Elevation')
    geom_tv.column('station', width=100, anchor='center')
    geom_tv.column('elevation', width=100, anchor='center')
    geom_tv.grid(column=0, row=0, columnspan=4, sticky=(tk.N, tk.S, tk.E, tk.W))

    geom_scroll = ttk.Scrollbar(geom_frame, orient=tk.VERTICAL, command=geom_tv.yview)
    geom_tv.configure(yscroll=geom_scroll.set)
    geom_scroll.grid(column=4, row=0, sticky=(tk.N, tk.S))
    geom_hscroll = ttk.Scrollbar(geom_frame, orient=tk.HORIZONTAL, command=geom_tv.xview)
    geom_tv.configure(xscroll=geom_hscroll.set)
    geom_hscroll.grid(column=0, row=3, columnspan=4, sticky=(tk.W, tk.E))

    def populate_geom_table(idx: int):
        geom_tv.delete(*geom_tv.get_children())
        xs = current_model.sections[idx]
        for i, (st, z) in enumerate(sorted(xs.geometry, key=lambda p: p[0])):
            geom_tv.insert('', 'end', iid=str(i), values=(f"{st:.3f}", f"{z:.3f}"))

    def geom_add_row():
        # add row after selection or at end
        sel = geom_tv.selection()
        if sel:
            idx = int(sel[0]) + 1
        else:
            idx = len(geom_tv.get_children())
        geom_tv.insert('', idx, iid=str(idx), values=("0.0","0.0"))
        # reindex iids
        for i, iid in enumerate(geom_tv.get_children()):
            geom_tv.item(iid, iid=str(i))

    def geom_remove_row():
        sel = geom_tv.selection()
        if not sel:
            return
        geom_tv.delete(sel[0])
        # reindex
        for i, iid in enumerate(geom_tv.get_children()):
            geom_tv.item(iid, iid=str(i))

    def geom_move(up: bool):
        sel = geom_tv.selection()
        if not sel:
            return
        iid = sel[0]
        kids = list(geom_tv.get_children())
        idx = kids.index(iid)
        new_idx = max(0, idx-1) if up else min(len(kids)-1, idx+1)
        if new_idx == idx:
            return
        vals = geom_tv.item(iid)['values']
        geom_tv.delete(iid)
        geom_tv.insert('', new_idx, iid=str(new_idx), values=vals)
        # rebuild all iids in order
        for i, iid2 in enumerate(geom_tv.get_children()):
            geom_tv.item(iid2, iid=str(i))

    # In-place editing for treeview cells (double-click)
    edit_entry = None

    def finish_edit(event=None):
        nonlocal edit_entry
        if edit_entry is None:
            return
        iid = edit_entry._iid
        col = edit_entry._col
        val = edit_entry.get()
        edit_entry.destroy()
        edit_entry = None
        # update treeview
        try:
            # Keep formatting similar
            if col == '#1':
                v = f"{float(val):.6f}"
            else:
                v = f"{float(val):.6f}"
        except Exception:
            messagebox.showerror('Invalid', 'Value must be numeric')
            return
        vals = list(geom_tv.item(iid, 'values'))
        col_index = 0 if col == '#1' else 1
        vals[col_index] = v
        geom_tv.item(iid, values=vals)

    def on_geom_double_click(event):
        nonlocal edit_entry
        # identify row/col
        region = geom_tv.identify_region(event.x, event.y)
        if region != 'cell':
            return
        row = geom_tv.identify_row(event.y)
        col = geom_tv.identify_column(event.x)
        if not row or not col:
            return
        bbox = geom_tv.bbox(row, col)
        if not bbox:
            return
        x, y, w, h = bbox
        val = geom_tv.set(row, column=col)
        # place Entry over cell
        edit_entry = ttk.Entry(geom_tv)
        edit_entry.place(x=x, y=y, width=w, height=h)
        edit_entry.insert(0, val)
        edit_entry._iid = row
        edit_entry._col = col
        edit_entry.focus_set()
        edit_entry.bind('<Return>', finish_edit)
        edit_entry.bind('<FocusOut>', finish_edit)

    geom_tv.bind('<Double-1>', on_geom_double_click)

    def apply_geom_changes(idx: int):
        # read rows and update current_model.sections[idx].geometry
        rows = []
        for iid in geom_tv.get_children():
            st_s, z_s = geom_tv.item(iid)['values']
            try:
                st = float(str(st_s))
                z = float(str(z_s))
            except Exception:
                messagebox.showerror('Invalid', 'Station/elevation must be numeric')
                return
            rows.append((st, z))
        # sort by station
        rows = sorted(rows, key=lambda p: p[0])
        current_model.sections[idx].geometry = [(float(x), float(z)) for x, z in rows]
        messagebox.showinfo('Applied', f'Geometry applied to {current_model.sections[idx].river_station}')
        # update plot
        plot_selected_section(idx)

    def geom_copy_selected():
        sel = geom_tv.selection()
        if not sel:
            messagebox.showinfo('Copy', 'No rows selected')
            return
        lines = []
        for iid in sel:
            st, z = geom_tv.item(iid, 'values')
            lines.append(f"{st}\t{z}")
        txt = '\n'.join(lines)
        try:
            root.clipboard_clear()
            root.clipboard_append(txt)
            messagebox.showinfo('Copied', f'Copied {len(lines)} row(s) to clipboard')
        except Exception as e:
            messagebox.showerror('Copy error', str(e))

    def geom_paste_clipboard(idx: int):
        try:
            txt = root.clipboard_get()
        except Exception:
            messagebox.showerror('Paste', 'Clipboard does not contain text')
            return
        lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        if not lines:
            messagebox.showinfo('Paste', 'No data to paste')
            return
        added = 0
        for ln in lines:
            if '\t' in ln:
                parts = ln.split('\t')
            elif ',' in ln:
                parts = ln.split(',')
            else:
                parts = ln.split()
            try:
                st = float(parts[0])
                z = float(parts[1])
            except Exception:
                continue
            geom_tv.insert('', 'end', values=(f"{st:.6f}", f"{z:.6f}"))
            added += 1
        for i, iid in enumerate(geom_tv.get_children()):
            geom_tv.item(iid, iid=str(i))
        messagebox.showinfo('Pasted', f'Pasted {added} row(s)')

    ttk.Button(geom_frame, text='Add', command=geom_add_row).grid(column=0, row=1, sticky=tk.W, pady=(6,0))
    ttk.Button(geom_frame, text='Remove', command=geom_remove_row).grid(column=1, row=1, sticky=tk.W, pady=(6,0))
    ttk.Button(geom_frame, text='Up', command=lambda: geom_move(True)).grid(column=2, row=1, sticky=tk.W, pady=(6,0))
    ttk.Button(geom_frame, text='Down', command=lambda: geom_move(False)).grid(column=3, row=1, sticky=tk.W, pady=(6,0))
    ttk.Button(geom_frame, text='Copy', command=geom_copy_selected).grid(column=0, row=2, sticky=tk.W, pady=(6,0))
    ttk.Button(geom_frame, text='Paste', command=lambda: geom_paste_clipboard(section_cb.current())).grid(column=1, row=2, sticky=tk.W, pady=(6,0))
    ttk.Button(geom_frame, text='Apply Geometry', command=lambda: apply_geom_changes(section_cb.current())).grid(column=0, row=3, columnspan=4, sticky=(tk.W, tk.E), pady=(6,0))

    # Detail plot for selected section
    detail_frame = ttk.Labelframe(root, text='Section Detail Plot', padding=(6,6))
    detail_frame.grid(column=0, row=4, columnspan=3, sticky=(tk.W, tk.E))
    # scrollable detail plot
    detail_outer, detail_canvas_holder = make_scrollable_frame(detail_frame, width=600, height=220)
    detail_outer.grid(column=0, row=0, sticky=(tk.W, tk.E))

    def plot_selected_section(idx: int):
        if not HAVE_MPL or current_model is None:
            return
        xs = current_model.sections[idx]
        geom = sorted(xs.geometry, key=lambda p: p[0])
        sx = [p[0] for p in geom]
        sz = [p[1] for p in geom]
        fig, ax = plt.subplots(figsize=(6,2.5))
        ax.plot(sx, sz, '-k', marker='o')
        ax.fill_between(sx, sz, min(sz)-1.0, color="#f0f0f0")
        ax.set_xlabel('Station')
        ax.set_ylabel('Elevation (ft)')
        ax.set_title(xs.river_station)
        fig.tight_layout()
        for child in detail_canvas_holder.winfo_children():
            child.destroy()
        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        except Exception:
            return
        canvas = FigureCanvasTkAgg(fig, master=detail_canvas_holder)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)


    def load_model_into_gui(path: str):
        nonlocal current_model, current_file, current_results
        current_model = load_input(path)
        current_file = path
        # populate section combobox
        names = [xs.river_station for xs in current_model.sections]
        section_cb['values'] = names
        if names:
            section_cb.current(0)
            section_var.set(names[0])
            show_section_properties(0)
            populate_geom_table(0)
            plot_selected_section(0)

    def show_section_properties(idx: int):
        xs = current_model.sections[idx]
        props['left_bank_station'].set(xs.left_bank_station)
        props['right_bank_station'].set(xs.right_bank_station)
        props['n_lob'].set(xs.n_lob)
        props['n_ch'].set(xs.n_ch)
        props['n_rob'].set(xs.n_rob)
        props['contraction_coeff'].set(xs.contraction_coeff)
        props['expansion_coeff'].set(xs.expansion_coeff)
        props['L_lob_to_next'].set(xs.L_lob_to_next)
        props['L_ch_to_next'].set(xs.L_ch_to_next)
        props['L_rob_to_next'].set(xs.L_rob_to_next)
        # populate geometry table for this section
        populate_geom_table(idx)
        plot_selected_section(idx)

    def on_section_change(event=None):
        if current_model is None:
            return
        idx = section_cb.current()
        if idx >= 0:
            show_section_properties(idx)

    section_cb.bind('<<ComboboxSelected>>', on_section_change)

    def apply_section_changes():
        if current_model is None:
            messagebox.showerror("No model", "Load a model first")
            return
        idx = section_cb.current()
        if idx < 0:
            return
        xs = current_model.sections[idx]
        xs.left_bank_station = float(props['left_bank_station'].get())
        xs.right_bank_station = float(props['right_bank_station'].get())
        xs.n_lob = float(props['n_lob'].get())
        xs.n_ch = float(props['n_ch'].get())
        xs.n_rob = float(props['n_rob'].get())
        xs.contraction_coeff = float(props['contraction_coeff'].get())
        xs.expansion_coeff = float(props['expansion_coeff'].get())
        xs.L_lob_to_next = float(props['L_lob_to_next'].get())
        xs.L_ch_to_next = float(props['L_ch_to_next'].get())
        xs.L_rob_to_next = float(props['L_rob_to_next'].get())
        messagebox.showinfo("Applied", f"Changes applied to section {xs.river_station}")

    # Controls: Load, Run, Apply, Save
    def on_browse():
        p = filedialog.askopenfilename(filetypes=[("GeoPackage files", "*.gpkg"), ("All files", "*")])
        if p:
            input_path.set(p)

    def on_load():
        p = input_path.get()
        if not p:
            messagebox.showerror("Input required", "Please select an input GeoPackage file.")
            return
        try:
            load_model_into_gui(p)
            text_out.delete(1.0, tk.END)
            text_out.insert(tk.END, f"Loaded model: {p}\n")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def on_run():
        nonlocal fig_canvas, current_results
        if current_model is None:
            # try loading from path
            p = input_path.get()
            if not p:
                messagebox.showerror("Input required", "Please select an input GeoPackage file.")
                return
            try:
                load_model_into_gui(p)
            except Exception as e:
                messagebox.showerror("Error", str(e))
                return

        # override bc and flow
        current_model.boundary_condition = ds_bc_var.get()
        current_model.boundary_value = float(ds_val_var.get())
        try:
            current_model.flow_cfs = float(flow_var.get())
        except Exception:
            messagebox.showerror("Invalid flow", "Flow must be numeric")
            return

        try:
            # apply GUI-selected methods
            global ALPHA_METHOD, SF_METHOD
            ALPHA_METHOD = alpha_var.get()
            SF_METHOD = sf_var.get()
            current_results = run_backwater(current_model)
        except Exception as e:
            messagebox.showerror("Run error", str(e))
            return

        # display textual results
        text_out.delete(1.0, tk.END)
        text_out.insert(tk.END, "Idx  RS            WSE(ft)    Depth(ft)  V(ft/s)  Alpha   K_total     A_total     Sf_total   Froude\n")
        for i, (xs, s) in enumerate(zip(current_model.sections, current_results)):
            text_out.insert(tk.END, f"{i:<4} {xs.river_station:<12} {s.wse:>9.3f} {s.depth_at_min:>11.3f} "
                                f"{s.V_t:>9.3f} {s.alpha:>7.3f} {s.K_t:>10.1f} {s.A_t:>10.2f} {s.Sf_total:>10.6f} {getattr(s, 'Froude', 0.0):>9.3f}\n")

        # plotting
        if HAVE_MPL:
            fig = _plot_results(current_model, current_results)
            for child in canvas_holder_inner.winfo_children():
                child.destroy()
            try:
                from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            except Exception:
                messagebox.showerror('Plot not available', 'matplotlib backend for Tk not available')
                return
            fig_canvas = FigureCanvasTkAgg(fig, master=canvas_holder_inner)
            fig_canvas.draw()
            fig_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)

    def on_save_model():
        if current_model is None:
            messagebox.showerror("No model", "Load and edit a model before saving.")
            return
        p = filedialog.asksaveasfilename(defaultextension='.gpkg', filetypes=[('GeoPackage', '*.gpkg')])
        if not p:
            return
        save_to_geopackage(p, current_model)
        messagebox.showinfo('Saved', f'Model saved to {p}')

    def on_save_plot():
        if not HAVE_MPL:
            messagebox.showerror('Plot not available', 'matplotlib is not installed')
            return
        p = filedialog.asksaveasfilename(defaultextension='.png', filetypes=[('PNG', '*.png')])
        if not p:
            return
        # Last figure is the one in the canvas
        try:
            import matplotlib.pyplot as _plt
            _plt.savefig(p)
            messagebox.showinfo('Saved', f'Plot saved to {p}')
        except Exception as e:
            messagebox.showerror('Save error', str(e))

    def create_new_model():
        nonlocal current_model, current_file, current_results
        # Minimal two-section default model
        try:
            base_flow = float(flow_var.get())
        except Exception:
            base_flow = 500.0

        xs0 = CrossSection(
            river_station='S_down',
            geometry=[(0.0, 100.0), (10.0, 99.5)],
            left_bank_station=2.0,
            right_bank_station=8.0,
            n_lob=0.035, n_ch=0.035, n_rob=0.035,
            contraction_coeff=0.1, expansion_coeff=0.3,
            L_lob_to_next=10.0, L_ch_to_next=10.0, L_rob_to_next=10.0
        )

        xs1 = CrossSection(
            river_station='S_up',
            geometry=[(10.0, 99.5), (20.0, 99.0)],
            left_bank_station=12.0,
            right_bank_station=18.0,
            n_lob=0.035, n_ch=0.035, n_rob=0.035,
            contraction_coeff=0.1, expansion_coeff=0.3,
            L_lob_to_next=10.0, L_ch_to_next=10.0, L_rob_to_next=10.0
        )

        current_model = ModelInput(
            flow_cfs=base_flow,
            flow_change=None,
            boundary_condition='known_wse',
            boundary_value=100.0,
            sections=[xs0, xs1]
        )
        current_file = None
        current_results = None
        # populate GUI
        names = [xs.river_station for xs in current_model.sections]
        section_cb['values'] = names
        if names:
            section_cb.current(0)
            section_var.set(names[0])
            show_section_properties(0)
            populate_geom_table(0)
            plot_selected_section(0)
        input_path.set('')
        messagebox.showinfo('New Model', 'Created new minimal two-section model')

    # Buttons setup
    btn_row = 0
    ttk.Button(frm, text='Browse...', command=on_browse).grid(column=2, row=0, sticky=tk.W)
    ttk.Button(frm, text='Load', command=on_load).grid(column=0, row=1, sticky=tk.W)
    ttk.Button(frm, text='Run', command=on_run).grid(column=1, row=1, sticky=tk.W)
    ttk.Button(frm, text='Apply Section Changes', command=apply_section_changes).grid(column=0, row=6, sticky=tk.W)
    ttk.Button(frm, text='Save Model...', command=on_save_model).grid(column=1, row=6, sticky=tk.W)
    ttk.Button(frm, text='Save Plot...', command=on_save_plot).grid(column=2, row=6, sticky=tk.W)

    root.mainloop()


def main():
    parser = argparse.ArgumentParser(
        description="Backwater (1D steady) standard-step solver for a single river reach / single flow.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--input", required=True, help="Path to GeoPackage input file (.gpkg).")
    parser.add_argument("--ds-bc", choices=["known_wse", "normal_depth"],
                        help="Override downstream boundary condition (optional).")
    parser.add_argument("--ds-value", type=float, help="Override downstream boundary value (WSE or S0).")
    parser.add_argument("--solver", choices=["py", "scipy", "numpy"], default='py', help="Solver backend to use: 'py' for pure Python or 'scipy'/'numpy' for SciPy brentq root-finder (if available).")
    parser.add_argument("--alpha-method", choices=["conveyance", "area"], default='conveyance', help="Alpha computation method: 'conveyance' (default) or 'area'.")
    parser.add_argument("--sf-method", choices=["combined", "avg"], default='combined', help="Representative friction slope method: 'combined' (default) or 'avg'.")
    args = parser.parse_args()

    model = load_input(args.input)
    if args.ds_bc:
        model.boundary_condition = args.ds_bc
    if args.ds_value is not None:
        model.boundary_value = args.ds_value
    # apply CLI method overrides
    global ALPHA_METHOD, SF_METHOD
    ALPHA_METHOD = args.alpha_method
    SF_METHOD = args.sf_method

    results = run_backwater(model, solver=args.solver)

    print("\n--- Results (Downstream → Upstream) ---")
    print("Idx  RS            WSE(ft)    Depth(ft)  V(ft/s)  Alpha   Energy(ft)  K_total     A_total     Sf_total   Froude")
    for i, (xs, s) in enumerate(zip(model.sections, results)):
        energy = getattr(s, 'wse', 0.0) + (getattr(s, 'alpha', 0.0) * getattr(s, 'V_t', 0.0) ** 2) / (2.0 * G)
        print(f"{i:<4} {xs.river_station:<12} {s.wse:>9.3f} {s.depth_at_min:>11.3f} "
              f"{s.V_t:>9.3f} {s.alpha:>7.3f} {energy:>11.3f} {s.K_t:>10.1f} {s.A_t:>10.2f} {s.Sf_total:>10.6f} {getattr(s, 'Froude', 0.0):>9.3f}")

    print("\nNote: Results are based on simplified assumptions as described in the header.\n")


if __name__ == "__main__":
    # If no CLI args supplied and Tkinter available, launch GUI
    if len(sys.argv) == 1 and HAVE_TK:
        try:
            launch_gui()
        except Exception:
            # fallback to CLI
            main()
    else:
        main()
