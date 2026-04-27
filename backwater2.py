
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

    GeoPackage format with layers: cross_sections, centerline, boundary_conditions.
-----------------------------------------------------------------------------
"""

import argparse
from datetime import datetime, timezone
import importlib
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional, Dict

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


def _qgis_profile_from_geometry(geom):
    """Return station/elevation profile from QgsGeometry line feature."""
    try:
        from qgis.core import QgsWkbTypes
    except Exception as exc:
        raise ImportError('PyQGIS not available') from exc

    if geom is None or geom.isEmpty():
        return [], None

    try:
        verts = [v for v in geom.vertices()]
    except Exception:
        verts = []
    if not verts:
        return [], None

    has_z = False
    try:
        has_z = QgsWkbTypes.hasZ(geom.wkbType())
    except Exception:
        has_z = False

    profile = []
    s = 0.0
    prev = verts[0]
    if has_z:
        profile.append((0.0, float(prev.z())))
    else:
        profile.append((0.0, float(prev.y())))

    for v in verts[1:]:
        if has_z:
            s += math.hypot(float(v.x()) - float(prev.x()), float(v.y()) - float(prev.y()))
            z = float(v.z())
        else:
            s += abs(float(v.x()) - float(prev.x()))
            z = float(v.y())
        profile.append((float(s), float(z)))
        prev = v

    try:
        centroid = geom.centroid()
        if centroid is None or centroid.isEmpty():
            centroid = None
    except Exception:
        centroid = None

    return profile, centroid


def _qgis_geometry_from_profile(profile):
    try:
        from qgis.core import QgsGeometry, QgsPoint
    except Exception as exc:
        raise ImportError('PyQGIS not available') from exc
    return QgsGeometry.fromPolyline([QgsPoint(float(st), 0.0, float(z)) for st, z in profile])


def _load_from_geopackage_qgis(path: str, cross_layer: str, centerline_layer: str, boundary_layer: str) -> ModelInput:
    try:
        from qgis.core import QgsVectorLayer
    except Exception as exc:
        raise ImportError('PyQGIS not available') from exc

    def _layer(layer_name: str):
        lyr = QgsVectorLayer(f"{path}|layername={layer_name}", layer_name, 'ogr')
        if not lyr.isValid():
            raise ValueError(f"GeoPackage layer '{layer_name}' could not be loaded from {path}")
        return lyr

    def _value(feat, name: str, default=None):
        try:
            idx = feat.fields().indexOf(name)
            if idx == -1:
                return default
            v = feat[idx]
            return default if v is None else v
        except Exception:
            return default

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
        text = str(value).strip()
        if not text or text.lower() in ('nan', 'none', 'null'):
            return None
        return text

    cross_lyr = _layer(cross_layer)
    center_lyr = _layer(centerline_layer)
    center_feat = next(center_lyr.getFeatures(), None)
    if center_feat is None or center_feat.geometry() is None or center_feat.geometry().isEmpty():
        raise ValueError(
            f"GeoPackage layer '{centerline_layer}' is empty. "
            "At least one centerline feature is required."
        )
    center_geom = center_feat.geometry()

    rows = []
    for feat in cross_lyr.getFeatures():
        profile, centroid = _qgis_profile_from_geometry(feat.geometry())
        chainage = 0.0
        try:
            if centroid is not None:
                chainage = float(center_geom.lineLocatePoint(centroid))
        except Exception:
            chainage = 0.0
        rows.append({
            'river_station': _value(feat, 'river_station', ''),
            'geometry': profile,
            'chainage': chainage,
            'left_bank_station': _safe_float(_value(feat, 'left_bank_station', 0.0), 0.0),
            'right_bank_station': _safe_float(_value(feat, 'right_bank_station', 0.0), 0.0),
            'n_lob': _safe_float(_value(feat, 'n_lob', 0.035), 0.035),
            'n_ch': _safe_float(_value(feat, 'n_ch', 0.035), 0.035),
            'n_rob': _safe_float(_value(feat, 'n_rob', 0.035), 0.035),
            'contraction_coeff': _safe_float(_value(feat, 'contraction_coeff', 0.1), 0.1),
            'expansion_coeff': _safe_float(_value(feat, 'expansion_coeff', 0.3), 0.3),
            'L_lob_to_next': _safe_float(_value(feat, 'L_lob_to_next', 0.0), 0.0),
            'L_ch_to_next': _safe_float(_value(feat, 'L_ch_to_next', 0.0), 0.0),
            'L_rob_to_next': _safe_float(_value(feat, 'L_rob_to_next', 0.0), 0.0),
            'culvert_code': _safe_int(_value(feat, 'culvert_code', 0), 0),
            'culvert_shape': _safe_shape(_value(feat, 'culvert_shape', None)),
            'culvert_diameter': _safe_float(_value(feat, 'culvert_diameter', 0.0), 0.0),
            'culvert_width': _safe_float(_value(feat, 'culvert_width', 0.0), 0.0),
            'culvert_height': _safe_float(_value(feat, 'culvert_height', 0.0), 0.0),
            'culvert_upstream_invert': _safe_float(_value(feat, 'culvert_upstream_invert', 0.0), 0.0),
            'culvert_downstream_invert': _safe_float(_value(feat, 'culvert_downstream_invert', 0.0), 0.0),
            'culvert_length': _safe_float(_value(feat, 'culvert_length', 0.0), 0.0),
            'culvert_weir_coeff': _safe_float(_value(feat, 'culvert_weir_coeff', 3.0), 3.0),
            'culvert_weir_sta_left': _safe_float(_value(feat, 'culvert_weir_sta_left', 0.0), 0.0),
            'culvert_weir_sta_right': _safe_float(_value(feat, 'culvert_weir_sta_right', 0.0), 0.0),
            'culvert_slope_legacy': _safe_float(_value(feat, 'culvert_slope', 0.0), 0.0),
        })

    rows = sorted(rows, key=lambda x: x.get('chainage', 0.0))

    sections = []
    for r in rows:
        xs = CrossSection(
            river_station=str(r.get('river_station', '')),
            geometry=[(float(x), float(z)) for x, z in r['geometry']],
            left_bank_station=float(r.get('left_bank_station', 0.0)),
            right_bank_station=float(r.get('right_bank_station', 0.0)),
            n_lob=float(r.get('n_lob', 0.035)), n_ch=float(r.get('n_ch', 0.035)), n_rob=float(r.get('n_rob', 0.035)),
            contraction_coeff=float(r.get('contraction_coeff', 0.1)), expansion_coeff=float(r.get('expansion_coeff', 0.3)),
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
        if xs.culvert_length <= 0.0:
            legacy_slope = float(r.get('culvert_slope_legacy', 0.0) or 0.0)
            if legacy_slope > 0.0:
                xs.culvert_length = max(1.0, xs.L_ch_to_next if xs.L_ch_to_next > 0.0 else 1.0)
                xs.culvert_upstream_invert = min((z for _, z in xs.geometry), default=0.0)
                xs.culvert_downstream_invert = xs.culvert_upstream_invert - legacy_slope * xs.culvert_length
        sections.append(xs)

    if len(sections) > 1:
        for i in range(len(sections) - 1):
            spacing = max(0.0, float(rows[i + 1]['chainage'] - rows[i]['chainage']))
            if sections[i].L_ch_to_next <= 0.0:
                sections[i].L_ch_to_next = spacing
            if sections[i].L_lob_to_next <= 0.0:
                sections[i].L_lob_to_next = spacing
            if sections[i].L_rob_to_next <= 0.0:
                sections[i].L_rob_to_next = spacing

    flow_cfs = 0.0
    boundary_condition = 'known_wse'
    boundary_value = 0.0
    try:
        b_lyr = _layer(boundary_layer)
        b_feat = next(b_lyr.getFeatures(), None)
        if b_feat is not None:
            flow_cfs = _safe_float(_value(b_feat, 'flow_cfs', 0.0), 0.0)
            boundary_condition = str(_value(b_feat, 'boundary_type', 'known_wse') or 'known_wse')
            boundary_value = _safe_float(_value(b_feat, 'boundary_value', 0.0), 0.0)
    except Exception:
        pass

    return ModelInput(flow_cfs=flow_cfs, flow_change=None, boundary_condition=boundary_condition, boundary_value=boundary_value, sections=sections)


def _save_to_geopackage_qgis(path: str, model: ModelInput, centerline_geom=None, overwrite: bool = True, crs_authid: Optional[str] = None):
    try:
        from qgis.core import QgsFeature, QgsField, QgsGeometry, QgsPointXY, QgsProject, QgsVectorFileWriter, QgsVectorLayer
        from qgis.PyQt.QtCore import QVariant
    except Exception as exc:
        raise ImportError('PyQGIS not available') from exc

    def _to_qgs_geom(geom_like):
        if geom_like is None:
            return None
        if hasattr(geom_like, 'asWkt') and hasattr(geom_like, 'isEmpty'):
            return geom_like
        if hasattr(geom_like, 'wkt'):
            try:
                return QgsGeometry.fromWkt(str(geom_like.wkt))
            except Exception:
                return None
        return None

    def _layer(layer_name: str):
        lyr = QgsVectorLayer(f"{path}|layername={layer_name}", layer_name, 'ogr')
        return lyr if lyr.isValid() else None

    authid = str(crs_authid) if crs_authid else 'EPSG:4326'
    cross_existing = _layer('cross_sections')
    if cross_existing is not None and cross_existing.crs().isValid() and not crs_authid:
        authid = cross_existing.crs().authid()

    center_existing = _layer('centerline')
    centerline_to_write = _to_qgs_geom(centerline_geom)
    if centerline_to_write is None and center_existing is not None:
        ef = next(center_existing.getFeatures(), None)
        if ef is not None:
            centerline_to_write = ef.geometry()
    if centerline_to_write is None or centerline_to_write.isEmpty():
        raise ValueError(
            "A centerline geometry is required when saving a model GeoPackage. "
            "Create/load a centerline layer first."
        )

    transform_ctx = QgsProject.instance().transformContext()
    opts = QgsVectorFileWriter.SaveVectorOptions()
    opts.driverName = 'GPKG'
    opts.fileEncoding = 'UTF-8'

    cross_layer = QgsVectorLayer(f"LineStringZ?crs={authid}", 'cross_sections', 'memory')
    cdp = cross_layer.dataProvider()
    cdp.addAttributes([
        QgsField('centerline_id', QVariant.Int),
        QgsField('river_station', QVariant.String),
        QgsField('left_bank_station', QVariant.Double),
        QgsField('right_bank_station', QVariant.Double),
        QgsField('n_lob', QVariant.Double),
        QgsField('n_ch', QVariant.Double),
        QgsField('n_rob', QVariant.Double),
        QgsField('contraction_coeff', QVariant.Double),
        QgsField('expansion_coeff', QVariant.Double),
        QgsField('L_lob_to_next', QVariant.Double),
        QgsField('L_ch_to_next', QVariant.Double),
        QgsField('L_rob_to_next', QVariant.Double),
        QgsField('culvert_code', QVariant.Int),
        QgsField('culvert_shape', QVariant.String),
        QgsField('culvert_diameter', QVariant.Double),
        QgsField('culvert_width', QVariant.Double),
        QgsField('culvert_height', QVariant.Double),
        QgsField('culvert_upstream_invert', QVariant.Double),
        QgsField('culvert_downstream_invert', QVariant.Double),
        QgsField('culvert_length', QVariant.Double),
        QgsField('culvert_weir_coeff', QVariant.Double),
        QgsField('culvert_weir_sta_left', QVariant.Double),
        QgsField('culvert_weir_sta_right', QVariant.Double),
        QgsField('culvert_slope', QVariant.Double),
    ])
    cross_layer.updateFields()
    feats = []
    for xs in model.sections:
        feat = QgsFeature(cross_layer.fields())
        feat.setGeometry(_qgis_geometry_from_profile([(float(x), float(z)) for x, z in xs.geometry]))
        feat['centerline_id'] = 1
        feat['river_station'] = str(xs.river_station)
        feat['left_bank_station'] = float(xs.left_bank_station)
        feat['right_bank_station'] = float(xs.right_bank_station)
        feat['n_lob'] = float(xs.n_lob)
        feat['n_ch'] = float(xs.n_ch)
        feat['n_rob'] = float(xs.n_rob)
        feat['contraction_coeff'] = float(xs.contraction_coeff)
        feat['expansion_coeff'] = float(xs.expansion_coeff)
        feat['L_lob_to_next'] = float(xs.L_lob_to_next)
        feat['L_ch_to_next'] = float(xs.L_ch_to_next)
        feat['L_rob_to_next'] = float(xs.L_rob_to_next)
        feat['culvert_code'] = int(xs.culvert_code)
        feat['culvert_shape'] = str(xs.culvert_shape or '')
        feat['culvert_diameter'] = float(xs.culvert_diameter)
        feat['culvert_width'] = float(xs.culvert_width)
        feat['culvert_height'] = float(xs.culvert_height)
        feat['culvert_upstream_invert'] = float(xs.culvert_upstream_invert)
        feat['culvert_downstream_invert'] = float(xs.culvert_downstream_invert)
        feat['culvert_length'] = float(xs.culvert_length)
        feat['culvert_weir_coeff'] = float(xs.culvert_weir_coeff)
        feat['culvert_weir_sta_left'] = float(xs.culvert_weir_sta_left)
        feat['culvert_weir_sta_right'] = float(xs.culvert_weir_sta_right)
        feat['culvert_slope'] = float(xs.culvert_slope())
        feats.append(feat)
    cdp.addFeatures(feats)
    opts.layerName = 'cross_sections'
    if hasattr(QgsVectorFileWriter, 'CreateOrOverwriteFile'):
        opts.actionOnExistingFile = (
            QgsVectorFileWriter.CreateOrOverwriteFile
            if overwrite
            else QgsVectorFileWriter.CreateOrOverwriteLayer
        )
    QgsVectorFileWriter.writeAsVectorFormatV2(cross_layer, path, transform_ctx, opts)

    center_layer = QgsVectorLayer(f"LineString?crs={authid}", 'centerline', 'memory')
    center_dp = center_layer.dataProvider()
    center_dp.addAttributes([QgsField('centerline_id', QVariant.Int)])
    center_layer.updateFields()
    cf = QgsFeature(center_layer.fields())
    cf.setGeometry(centerline_to_write)
    cf['centerline_id'] = 1
    center_dp.addFeature(cf)
    opts.layerName = 'centerline'
    if hasattr(QgsVectorFileWriter, 'CreateOrOverwriteLayer'):
        opts.actionOnExistingFile = QgsVectorFileWriter.CreateOrOverwriteLayer
    QgsVectorFileWriter.writeAsVectorFormatV2(center_layer, path, transform_ctx, opts)

    b_layer = QgsVectorLayer(f"Point?crs={authid}", 'boundary_conditions', 'memory')
    bdp = b_layer.dataProvider()
    bdp.addAttributes([
        QgsField('boundary_type', QVariant.String),
        QgsField('boundary_value', QVariant.Double),
        QgsField('flow_cfs', QVariant.Double),
    ])
    b_layer.updateFields()
    bf = QgsFeature(b_layer.fields())
    bf.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(0.0, 0.0)))
    bf['boundary_type'] = str(model.boundary_condition)
    bf['boundary_value'] = float(model.boundary_value)
    bf['flow_cfs'] = float(model.flow_cfs)
    bdp.addFeature(bf)
    opts.layerName = 'boundary_conditions'
    if hasattr(QgsVectorFileWriter, 'CreateOrOverwriteLayer'):
        opts.actionOnExistingFile = QgsVectorFileWriter.CreateOrOverwriteLayer
    QgsVectorFileWriter.writeAsVectorFormatV2(b_layer, path, transform_ctx, opts)


def load_from_geopackage(path: str, cross_layer: str = 'cross_sections', centerline_layer: str = 'centerline', boundary_layer: str = 'boundary_conditions') -> ModelInput:
    """Load model from a GeoPackage (PyQGIS first, geopandas/shapely fallback)."""
    try:
        return _load_from_geopackage_qgis(path, cross_layer, centerline_layer, boundary_layer)
    except ImportError:
        pass

    try:
        import geopandas as gpd
        from shapely.geometry import LineString
    except Exception as e:
        raise RuntimeError('PyQGIS or geopandas/shapely is required to load GeoPackage') from e

    cross_gdf = gpd.read_file(path, layer=cross_layer)
    centerline = None
    try:
        centerline_gdf = gpd.read_file(path, layer=centerline_layer)
        if len(centerline_gdf) > 0:
            centerline = centerline_gdf.geometry.iloc[0]
    except Exception as exc:
        raise ValueError(
            f"GeoPackage is missing required '{centerline_layer}' layer. "
            "Create a model GeoPackage with centerline and try again."
        ) from exc
    if centerline is None:
        raise ValueError(
            f"GeoPackage layer '{centerline_layer}' is empty. "
            "At least one centerline feature is required."
        )

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

    if len(sections) > 1 and all('chainage' in r for r in rows):
        for i in range(len(sections) - 1):
            spacing = max(0.0, float(rows[i + 1]['chainage'] - rows[i]['chainage']))
            # Default reach lengths from centerline spacing when not provided.
            if sections[i].L_ch_to_next <= 0.0:
                sections[i].L_ch_to_next = spacing
            if sections[i].L_lob_to_next <= 0.0:
                sections[i].L_lob_to_next = spacing
            if sections[i].L_rob_to_next <= 0.0:
                sections[i].L_rob_to_next = spacing

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


def save_to_geopackage(path: str, model: ModelInput, centerline_geom=None, overwrite: bool = True, crs_authid: Optional[str] = None):
    try:
        _save_to_geopackage_qgis(path, model, centerline_geom=centerline_geom, overwrite=overwrite, crs_authid=crs_authid)
        return
    except ImportError:
        pass

    try:
        import geopandas as gpd
        from shapely.geometry import LineString, MultiLineString
        import fiona
    except Exception as e:
            raise RuntimeError('PyQGIS or geopandas/fiona/shapely is required to save GeoPackage') from e

    def _interp_profile_elevation(profile: List[Tuple[float, float]], station: float) -> float:
        if not profile:
            return 0.0
        pts = sorted([(float(st), float(z)) for st, z in profile], key=lambda p: p[0])
        if station <= pts[0][0]:
            return float(pts[0][1])
        if station >= pts[-1][0]:
            return float(pts[-1][1])
        for i in range(1, len(pts)):
            s0, z0 = pts[i - 1]
            s1, z1 = pts[i]
            if s1 <= s0:
                continue
            if station <= s1:
                t = (station - s0) / (s1 - s0)
                return float(z0 + (z1 - z0) * t)
        return float(pts[-1][1])

    def _linestring_with_preserved_xy_updated_z(line, profile: List[Tuple[float, float]]):
        coords = list(line.coords)
        if len(coords) < 2:
            return line
        chainage = [0.0]
        for i in range(1, len(coords)):
            x0, y0 = float(coords[i - 1][0]), float(coords[i - 1][1])
            x1, y1 = float(coords[i][0]), float(coords[i][1])
            chainage.append(chainage[-1] + math.hypot(x1 - x0, y1 - y0))
        new_coords = []
        for c, s in zip(coords, chainage):
            x = float(c[0])
            y = float(c[1])
            z = _interp_profile_elevation(profile, float(s))
            new_coords.append((x, y, z))
        return LineString(new_coords)

    def _geometry_with_preserved_xy_updated_z(geom, profile: List[Tuple[float, float]]):
        if geom is None:
            return None
        try:
            if isinstance(geom, LineString):
                return _linestring_with_preserved_xy_updated_z(geom, profile)
            if isinstance(geom, MultiLineString):
                return MultiLineString([
                    _linestring_with_preserved_xy_updated_z(line, profile)
                    for line in geom.geoms
                ])
        except Exception:
            return None
        return None

    existing_by_station = {}
    existing_centerline_geom = None
    existing_crs = None
    try:
        if os.path.exists(path):
            existing_gdf = gpd.read_file(path, layer='cross_sections')
            existing_crs = getattr(existing_gdf, 'crs', None)
            for _, feat in existing_gdf.iterrows():
                rs = feat.get('river_station')
                if rs is None:
                    continue
                existing_by_station[str(rs)] = feat.geometry
            try:
                centerline_gdf = gpd.read_file(path, layer='centerline')
                if len(centerline_gdf) > 0:
                    existing_centerline_geom = centerline_gdf.geometry.iloc[0]
                    if existing_crs is None:
                        existing_crs = getattr(centerline_gdf, 'crs', None)
            except Exception:
                pass
    except Exception:
        existing_by_station = {}

    rows = []
    for xs in model.sections:
        coords = [(float(x), float(z)) for x,z in xs.geometry]
        geom = None
        existing_geom = existing_by_station.get(str(xs.river_station))
        if existing_geom is not None:
            geom = _geometry_with_preserved_xy_updated_z(existing_geom, coords)
        if geom is None:
            # Fallback for new sections with no existing planform geometry.
            geom = LineString([(st, 0.0, z) for st, z in coords])
        rows.append({
            'geometry': geom,
            'centerline_id': 1,
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

    gdf_crs = crs_authid if crs_authid else existing_crs
    gdf = gpd.GeoDataFrame(rows, geometry='geometry', crs=gdf_crs)
    mode = 'w' if overwrite else 'a'
    gdf.to_file(path, layer='cross_sections', driver='GPKG', index=False, mode=mode)

    centerline_to_write = centerline_geom if centerline_geom is not None else existing_centerline_geom
    if centerline_to_write is None:
        raise ValueError(
            "A centerline geometry is required when saving a model GeoPackage. "
            "Create/load a centerline layer first."
        )

    cgdf = gpd.GeoDataFrame([{'geometry': centerline_to_write, 'centerline_id': 1}], geometry='geometry', crs=gdf_crs)
    cgdf.to_file(path, layer='centerline', driver='GPKG', index=False, mode='a')

    try:
        import fiona
        schema = {'geometry': 'None', 'properties': {'boundary_type': 'str', 'boundary_value': 'float', 'flow_cfs': 'float'}}
        with fiona.open(path, mode='a', driver='GPKG', layer='boundary_conditions', schema=schema) as dst:
            props = {'boundary_type': model.boundary_condition, 'boundary_value': float(model.boundary_value), 'flow_cfs': float(model.flow_cfs)}
            dst.write({'properties': props})
    except Exception:
        pass


def _save_results_to_geopackage_qgis(
    path: str,
    model: Optional[ModelInput],
    results: List[SectionState],
    layer_name: str = 'model_results',
    solver: str = 'py',
):
    try:
        from qgis.core import QgsFeature, QgsField, QgsProject, QgsVectorFileWriter, QgsVectorLayer
        from qgis.PyQt.QtCore import QVariant
    except Exception as exc:
        raise ImportError('PyQGIS not available') from exc

    layer = QgsVectorLayer('None', layer_name, 'memory')
    if not layer.isValid():
        raise RuntimeError(f'Could not create in-memory layer for {layer_name}')

    dp = layer.dataProvider()
    dp.addAttributes([
        QgsField('result_index', QVariant.Int),
        QgsField('river_station', QVariant.String),
        QgsField('solver', QVariant.String),
        QgsField('run_time_utc', QVariant.String),
        QgsField('wse', QVariant.Double),
        QgsField('depth_at_min', QVariant.Double),
        QgsField('alpha', QVariant.Double),
        QgsField('A_lob', QVariant.Double),
        QgsField('A_ch', QVariant.Double),
        QgsField('A_rob', QVariant.Double),
        QgsField('K_lob', QVariant.Double),
        QgsField('K_ch', QVariant.Double),
        QgsField('K_rob', QVariant.Double),
        QgsField('Q_lob', QVariant.Double),
        QgsField('Q_ch', QVariant.Double),
        QgsField('Q_rob', QVariant.Double),
        QgsField('V_t', QVariant.Double),
        QgsField('K_t', QVariant.Double),
        QgsField('A_t', QVariant.Double),
        QgsField('Sf_total', QVariant.Double),
        QgsField('Froude', QVariant.Double),
    ])
    layer.updateFields()

    run_time_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')
    features = []
    for idx, state in enumerate(results or []):
        feat = QgsFeature(layer.fields())
        river_station = str(idx)
        if model is not None and idx < len(getattr(model, 'sections', [])):
            river_station = str(model.sections[idx].river_station)
        feat['result_index'] = int(idx)
        feat['river_station'] = river_station
        feat['solver'] = str(solver)
        feat['run_time_utc'] = run_time_utc
        feat['wse'] = float(getattr(state, 'wse', 0.0))
        feat['depth_at_min'] = float(getattr(state, 'depth_at_min', 0.0))
        feat['alpha'] = float(getattr(state, 'alpha', 0.0))
        feat['A_lob'] = float(getattr(state, 'A_lob', 0.0))
        feat['A_ch'] = float(getattr(state, 'A_ch', 0.0))
        feat['A_rob'] = float(getattr(state, 'A_rob', 0.0))
        feat['K_lob'] = float(getattr(state, 'K_lob', 0.0))
        feat['K_ch'] = float(getattr(state, 'K_ch', 0.0))
        feat['K_rob'] = float(getattr(state, 'K_rob', 0.0))
        feat['Q_lob'] = float(getattr(state, 'Q_lob', 0.0))
        feat['Q_ch'] = float(getattr(state, 'Q_ch', 0.0))
        feat['Q_rob'] = float(getattr(state, 'Q_rob', 0.0))
        feat['V_t'] = float(getattr(state, 'V_t', 0.0))
        feat['K_t'] = float(getattr(state, 'K_t', 0.0))
        feat['A_t'] = float(getattr(state, 'A_t', 0.0))
        feat['Sf_total'] = float(getattr(state, 'Sf_total', 0.0))
        feat['Froude'] = float(getattr(state, 'Froude', 0.0))
        features.append(feat)

    if features:
        dp.addFeatures(features)

    transform_ctx = QgsProject.instance().transformContext()
    opts = QgsVectorFileWriter.SaveVectorOptions()
    opts.driverName = 'GPKG'
    opts.fileEncoding = 'UTF-8'
    opts.layerName = layer_name
    if hasattr(QgsVectorFileWriter, 'CreateOrOverwriteLayer'):
        opts.actionOnExistingFile = QgsVectorFileWriter.CreateOrOverwriteLayer
    QgsVectorFileWriter.writeAsVectorFormatV2(layer, path, transform_ctx, opts)


def _load_results_from_geopackage_qgis(path: str, layer_name: str = 'model_results') -> List[SectionState]:
    try:
        from qgis.core import QgsVectorLayer
    except Exception as exc:
        raise ImportError('PyQGIS not available') from exc

    lyr = QgsVectorLayer(f"{path}|layername={layer_name}", layer_name, 'ogr')
    if not lyr.isValid():
        return []

    def _value(feat, name: str, default=None):
        try:
            idx = feat.fields().indexOf(name)
            if idx == -1:
                return default
            v = feat[idx]
            return default if v is None else v
        except Exception:
            return default

    def _safe_float(value, default=0.0):
        try:
            v = float(value)
            return default if math.isnan(v) else v
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

    rows = []
    for feat in lyr.getFeatures():
        idx = _safe_int(_value(feat, 'result_index', len(rows)), len(rows))
        state = SectionState(
            wse=_safe_float(_value(feat, 'wse', 0.0), 0.0),
            depth_at_min=_safe_float(_value(feat, 'depth_at_min', 0.0), 0.0),
            alpha=_safe_float(_value(feat, 'alpha', 0.0), 0.0),
            A_lob=_safe_float(_value(feat, 'A_lob', 0.0), 0.0),
            A_ch=_safe_float(_value(feat, 'A_ch', 0.0), 0.0),
            A_rob=_safe_float(_value(feat, 'A_rob', 0.0), 0.0),
            K_lob=_safe_float(_value(feat, 'K_lob', 0.0), 0.0),
            K_ch=_safe_float(_value(feat, 'K_ch', 0.0), 0.0),
            K_rob=_safe_float(_value(feat, 'K_rob', 0.0), 0.0),
            Q_lob=_safe_float(_value(feat, 'Q_lob', 0.0), 0.0),
            Q_ch=_safe_float(_value(feat, 'Q_ch', 0.0), 0.0),
            Q_rob=_safe_float(_value(feat, 'Q_rob', 0.0), 0.0),
            V_t=_safe_float(_value(feat, 'V_t', 0.0), 0.0),
            K_t=_safe_float(_value(feat, 'K_t', 0.0), 0.0),
            A_t=_safe_float(_value(feat, 'A_t', 0.0), 0.0),
            Sf_total=_safe_float(_value(feat, 'Sf_total', 0.0), 0.0),
            Froude=_safe_float(_value(feat, 'Froude', 0.0), 0.0),
        )
        rows.append((idx, state))

    rows.sort(key=lambda t: t[0])
    return [state for _, state in rows]


def save_results_to_geopackage(
    path: str,
    model: Optional[ModelInput],
    results: List[SectionState],
    layer_name: str = 'model_results',
    solver: str = 'py',
):
    """Persist model run results to a GeoPackage table layer."""
    try:
        _save_results_to_geopackage_qgis(path, model, results, layer_name=layer_name, solver=solver)
        return
    except ImportError:
        raise RuntimeError('PyQGIS is required to save model results to GeoPackage safely.')


def load_results_from_geopackage(path: str, layer_name: str = 'model_results') -> List[SectionState]:
    """Load persisted model run results from a GeoPackage table layer."""
    try:
        return _load_results_from_geopackage_qgis(path, layer_name=layer_name)
    except ImportError:
        pass

    try:
        import fiona
    except Exception:
        return []

    rows = []
    try:
        with fiona.open(path, mode='r', driver='GPKG', layer=layer_name) as src:
            for rec in src:
                props = rec.get('properties') or {}
                idx = int(props.get('result_index', len(rows)))
                rows.append((
                    idx,
                    SectionState(
                        wse=float(props.get('wse', 0.0) or 0.0),
                        depth_at_min=float(props.get('depth_at_min', 0.0) or 0.0),
                        alpha=float(props.get('alpha', 0.0) or 0.0),
                        A_lob=float(props.get('A_lob', 0.0) or 0.0),
                        A_ch=float(props.get('A_ch', 0.0) or 0.0),
                        A_rob=float(props.get('A_rob', 0.0) or 0.0),
                        K_lob=float(props.get('K_lob', 0.0) or 0.0),
                        K_ch=float(props.get('K_ch', 0.0) or 0.0),
                        K_rob=float(props.get('K_rob', 0.0) or 0.0),
                        Q_lob=float(props.get('Q_lob', 0.0) or 0.0),
                        Q_ch=float(props.get('Q_ch', 0.0) or 0.0),
                        Q_rob=float(props.get('Q_rob', 0.0) or 0.0),
                        V_t=float(props.get('V_t', 0.0) or 0.0),
                        K_t=float(props.get('K_t', 0.0) or 0.0),
                        A_t=float(props.get('A_t', 0.0) or 0.0),
                        Sf_total=float(props.get('Sf_total', 0.0) or 0.0),
                        Froude=float(props.get('Froude', 0.0) or 0.0),
                    ),
                ))
    except Exception:
        return []

    rows.sort(key=lambda t: t[0])
    return [state for _, state in rows]


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
# CLI entrypoint
# ---------------------------------------------------------------------------
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
    main()
