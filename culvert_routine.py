#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-----------------------------------------------------------------------------
   culvert.py (Python translation)

   Project:  EPA SWMM5
   Version:  5.1
   Date:     03/20/14   (Build 5.1.001)
   Author:   L. Rossman

   Culvert equations for SWMM5

   Computes flow reduction in a culvert-type conduit due to
   inlet control using equations from the FHWA HEC-5 circular.

   ---------------------------------------------------------------------------
   Python translation & headwater solver:
   - Preserves all data contained in the original culvert.c (incl. comments).
   - Adds:
       * Simple geometry classes for circular & rectangular sections.
       * A Ridder root finder (pure stdlib).
       * A function to solve headwater depth h for a given flow Q.
       * An optional CLI.

   Units: feet, seconds, cfs (consistent with the original code).

   IMPORTANT:
   - Where the original C referenced SWMM objects (Link, Conduit, Node, TXsect)
     this module provides a minimal substitute: a "Xsect" object with required
     attributes/methods.
-----------------------------------------------------------------------------
"""

import math
from dataclasses import dataclass
from typing import Callable, Optional

# -----------------------------------------------------------------------------
#  Constants & indices (as in C)
# -----------------------------------------------------------------------------
# enum CulvertParam {FORM, K, M, C, Y};
FORM, K, M, C, Y = 0, 1, 2, 3, 4
MAX_CULVERT_CODE = 57

# Physical constants
GRAVITY = 32.2  # ft/s^2 (SWMM standard)
BIG = 1.0e20    # large placeholder used in submerged formula when arg <= 0

# -----------------------------------------------------------------------------
#  Culvert parameter table (Params) -- FULLY PRESERVED WITH COMMENTS
# -----------------------------------------------------------------------------
# static const double Params[58][5] = {
Params = [
#   FORM   K       M     C        Y
#------------------------------------
    [0.0, 0.0,    0.0,  0.0,    0.00],

    # Circular concrete
    [1.0, 0.0098, 2.00, 0.0398, 0.67],  # 1 Square edge w/headwall                       
    [1.0, 0.0018, 2.00, 0.0292, 0.74],  # 2 Groove end w/headwall                        
    [1.0, 0.0045, 2.00, 0.0317, 0.69],  # 3 Groove end projecting                        

    # Circular Corrugated Metal Pipe
    [1.0, 0.0078, 2.00, 0.0379, 0.69],  # 4 Headwall                                     
    [1.0, 0.0210, 1.33, 0.0463, 0.75],  # 5 Mitered to slope                             
    [1.0, 0.0340, 1.50, 0.0553, 0.54],  # 6 Projecting                                   

    # Circular Pipe, Beveled Ring Entrance
    [1.0, 0.0018, 2.50, 0.0300, 0.74],  # 7 Beveled ring, 45 deg bevels                  
    [1.0, 0.0018, 2.50, 0.0243, 0.83],  # 8 Beveled ring, 33.7 deg bevels                

    # Rectangular Box with Flared Wingwalls
    [1.0, 0.026, 1.0,   0.0347, 0.81],  # 9 30-75 deg. wingwall flares                   
    [1.0, 0.061, 0.75,  0.0400, 0.80],  # 10 90 or 15 deg. wingwall flares                
    [1.0, 0.061, 0.75,  0.0423, 0.82],  # 11 0 deg. wingwall flares (striaght sides)      

    # Rectanglar Box with Flared Wingwalls & Top Edge Bevel
    [2.0, 0.510, 0.667, 0.0309, 0.80],  # 12 45 deg. flare; 0.43D top edge bevel          
    [2.0, 0.486, 0.667, 0.0249, 0.83],  # 13 18-33.7 deg flare; 0.083D top edge bevel     

    # Rectangular Box; 90-deg Headwall; Chamfered or Beveled Inlet Edges
    [2.0, 0.515, 0.667, 0.0375, 0.79],  # 14 chamfered 3/4-in
    [2.0, 0.495, 0.667, 0.0314, 0.82],  # 15 beveled 1/2-in/ft at 45 deg (1:1)
    [2.0, 0.486, 0.667, 0.0252, 0.865], # 16 beveled 1-in/ft at 33.7 deg (1:1.5)

    # Rectangular Box; Skewed Headwall; Chamfered or Beveled Inlet Edges
    [2.0, 0.545, 0.667, 0.04505,0.73],  # 17 3/4" chamfered edge, 45 deg skewed headwall
    [2.0, 0.533, 0.667, 0.0425, 0.705], # 18 3/4" chamfered edge, 30 deg skewed headwall
    [2.0, 0.522, 0.667, 0.0402, 0.68],  # 19 3/4" chamfered edge, 15 deg skewed headwall
    [2.0, 0.498, 0.667, 0.0327, 0.75],  # 20 45 deg beveled edge, 10-45 deg skewed headwall

    # Rectangular box, Non-offset Flared Wingwalls; 3/4" Chamfer at Top of Inlet
    [2.0, 0.497, 0.667, 0.0339, 0.803], # 21 45 deg (1:1) wingwall flare
    [2.0, 0.493, 0.667, 0.0361, 0.806], # 22 18.4 deg (3:1) wingwall flare
    [2.0, 0.495, 0.667, 0.0386, 0.71],  # 23 18.4 deg (3:1) wingwall flare, 30 deg inlet skew

    # Rectangular box, Offset Flared Wingwalls, Beveled Edge at Inlet Top
    [2.0, 0.497, 0.667, 0.0302, 0.835],  # 24 45 deg (1:1) flare, 0.042D top edge bevel
    [2.0, 0.495, 0.667, 0.0252, 0.881],  # 25 33.7 deg (1.5:1) flare, 0.083D top edge bevel
    [2.0, 0.493, 0.667, 0.0227, 0.887],  # 26 18.4 deg (3:1) flare, 0.083D top edge bevel

    # Corrugated Metal Box
    [1.0, 0.0083, 2.00, 0.0379, 0.69],  # 27 90 deg headwall
    [1.0, 0.0145, 1.75, 0.0419, 0.64],  # 28 Thick wall projecting
    [1.0, 0.0340, 1.50, 0.0496, 0.57],  # 29 Thin wall projecting

    # Horizontal Ellipse Concrete
    [1.0, 0.0100, 2.00, 0.0398, 0.67],  # 30 Square edge w/headwall
    [1.0, 0.0018, 2.50, 0.0292, 0.74],  # 31 Grooved end w/headwall
    [1.0, 0.0045, 2.00, 0.0317, 0.69],  # 32 Grooved end projecting

    # Vertical Ellipse Concrete
    [1.0, 0.0100, 2.00, 0.0398, 0.67],  # 33 Square edge w/headwall
    [1.0, 0.0018, 2.50, 0.0292, 0.74],  # 34 Grooved end w/headwall
    [1.0, 0.0095, 2.00, 0.0317, 0.69],  # 35 Grooved end projecting

    # Pipe Arch, 18" Corner Radius, Corrugated Metal
    [1.0, 0.0083, 2.00, 0.0379, 0.69],  # 36 90 deg headwall
    [1.0, 0.0300, 1.00, 0.0463, 0.75],  # 37 Mitered to slope
    [1.0, 0.0340, 1.50, 0.0496, 0.57],  # 38 Projecting

    # Pipe Arch, 18" Corner Radius, Corrugated Metal
    [1.0, 0.0300, 1.50, 0.0496, 0.57],  # 39 Projecting
    [1.0, 0.0088, 2.00, 0.0368, 0.68],  # 40 No bevels
    [1.0, 0.0030, 2.00, 0.0269, 0.77],  # 41 33.7 deg bevels

    # Pipe Arch, 31" Corner Radius, Corrugated Metal
    [1.0, 0.0300, 1.50, 0.0496, 0.57],  # 42 Projecting
    [1.0, 0.0088, 2.00, 0.0368, 0.68],  # 43 No bevels
    [1.0, 0.0030, 2.00, 0.0269, 0.77],  # 44 33.7 deg. bevels

    # Arch, Corrugated Metal
    [1.0, 0.0083, 2.00, 0.0379, 0.69],  # 45 90 deg headwall
    [1.0, 0.0300, 1.00, 0.0463, 0.75],  # 46 Mitered to slope
    [1.0, 0.0340, 1.50, 0.0496, 0.57],  # 47 Thin wall projecting

    # Circular Culvert
    [2.0, 0.534, 0.555, 0.0196, 0.90],  # 48 Smooth tapered inlet throat
    [2.0, 0.519, 0.640, 0.0210, 0.90],  # 49 Rough tapered inlet throat

    # Elliptical Inlet Face
    [2.0, 0.536, 0.622, 0.0368, 0.83],  # 50 Tapered inlet, beveled edges
    [2.0, 0.5035,0.719, 0.0478, 0.80],  # 51 Tapered inlet, square edges
    [2.0, 0.547, 0.800, 0.0598, 0.75],  # 52 Tapered inlet, thin edge projecting

    # Rectangular
    [2.0, 0.475, 0.667, 0.0179, 0.97],  # 53 Tapered inlet throat

    # Rectangular Concrete
    [2.0, 0.560, 0.667, 0.0446, 0.85],  # 54 Side tapered, less favorable edges
    [2.0, 0.560, 0.667, 0.0378, 0.87],  # 55 Side tapered, more favorable edges

    # Rectangular Concrete
    [2.0, 0.500, 0.667, 0.0446, 0.65], # 56 Slope tapered, less favorable edges
    [2.0, 0.500, 0.667, 0.0378, 0.71]  # 57 Slope tapered, more favorable edges
]
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
#  Culvert data structure (Python equivalent of TCulvert)
# -----------------------------------------------------------------------------
@dataclass
class Culvert:
    yFull: float                 # full depth of culvert (ft)
    scf: float                   # slope correction factor
    dQdH: float                  # derivative of flow w.r.t. head
    qc: float                    # unsubmerged critical flow
    kk: float
    mm: float                    # coefficients for unsubmerged flow
    ad: float
    hPlus: float                 # intermediate term
    xsect: "Xsect"               # reference to cross section geometry


# -----------------------------------------------------------------------------
#  Minimal cross-section geometry interface
#    - Must provide: yFull, aFull, culvertCode, area(y), top_width(y)
# -----------------------------------------------------------------------------
class Xsect:
    def __init__(self, y_full: float, a_full: float, culvert_code: int):
        self.yFull = y_full
        self.aFull = a_full
        self.culvertCode = culvert_code

    def area(self, y: float) -> float:
        raise NotImplementedError

    def top_width(self, y: float) -> float:
        raise NotImplementedError

    def wetted_perimeter(self, y: float) -> float:
        raise NotImplementedError

    def hydraulic_radius(self, y: float) -> float:
        area = self.area(y)
        perimeter = self.wetted_perimeter(y)
        if area <= 0.0 or perimeter <= 0.0:
            return 0.0
        return area / perimeter


class CircularXsect(Xsect):
    """Circular conduit (diameter = yFull)."""
    def __init__(self, diameter_ft: float, culvert_code: int):
        R = 0.5 * diameter_ft
        a_full = math.pi * R * R
        super().__init__(y_full=diameter_ft, a_full=a_full, culvert_code=culvert_code)
        self.R = R
        self.radius_ft = R

    def area(self, y: float) -> float:
        y = max(0.0, min(y, 2.0 * self.R))
        R = self.R
        if y <= 0.0:
            return 0.0
        theta = 2.0 * math.acos(max(-1.0, min(1.0, (R - y) / R)))
        seg_area = 0.5 * (R * R) * (theta - math.sin(theta))
        return seg_area

    def top_width(self, y: float) -> float:
        y = max(0.0, min(y, 2.0 * self.R))
        R = self.R
        if y <= 0.0:
            return 0.0
        return 2.0 * math.sqrt(max(0.0, 2.0 * R * y - y * y))

    def wetted_perimeter(self, y: float) -> float:
        y = max(0.0, min(y, 2.0 * self.R))
        if y <= 0.0:
            return 0.0
        theta = 2.0 * math.acos(max(-1.0, min(1.0, (self.R - y) / self.R)))
        return self.R * theta


class RectangularXsect(Xsect):
    """Rectangular box B x H (yFull = H)."""
    def __init__(self, width_ft: float, height_ft: float, culvert_code: int):
        super().__init__(y_full=height_ft, a_full=width_ft * height_ft, culvert_code=culvert_code)
        self.B = width_ft
        self.H = height_ft

    def area(self, y: float) -> float:
        return self.B * max(0.0, min(y, self.H))

    def top_width(self, y: float) -> float:
        return self.B if y > 0.0 else 0.0

    def wetted_perimeter(self, y: float) -> float:
        y = max(0.0, min(y, self.H))
        if y <= 0.0:
            return 0.0
        return self.B + 2.0 * y


# -----------------------------------------------------------------------------
#  Root finding: Ridder's method (as used in the C code findroot_Ridder)
# -----------------------------------------------------------------------------
def ridder(f: Callable[[float], float], a: float, b: float, tol: float = 1.0e-6, max_iter: int = 100) -> float:
    """Ridder's method: requires a bracket [a,b] with opposite-signed f(a), f(b)."""
    fa = f(a)
    fb = f(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0.0:
        raise ValueError("Ridder requires a bracketing interval with opposite signs.")

    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = f(m)
        # Avoid division by zero
        s_sq = fm * fm - fa * fb
        if s_sq <= 0.0:
            # Fallback to bisection step
            if fa * fm < 0.0:
                b, fb = m, fm
            else:
                a, fa = m, fm
            if abs(b - a) < tol:
                return 0.5 * (a + b)
            continue
        s = math.sqrt(s_sq)
        # Ridder's formula
        sign = -1.0 if (fa - fb) < 0.0 else 1.0
        dx = (m - a) * fm / s * sign
        x = m + dx
        fx = f(x)
        if abs(fx) < tol:
            return x
        # Update bracket
        if fm * fx < 0.0:
            a, fa = m, fm
            b, fb = x, fx
        elif fa * fx < 0.0:
            b, fb = x, fx
        else:
            a, fa = x, fx
        if abs(b - a) < tol:
            return 0.5 * (a + b)
    return 0.5 * (a + b)


# -----------------------------------------------------------------------------
#  Local functions (translated from C)
# -----------------------------------------------------------------------------
def getUnsubmergedFlow(code: int, h: float, culvert: Culvert) -> float:
    """
    //  Input:   code  = culvert type code number
    //           h     = inlet water depth above culvert invert
    //           culvert = pointer to a culvert data structure
    //  Output:  returns flow rate;
    //           computes value of variable Dqdh
    //  Purpose: computes flow rate and its derivative for unsubmerged
    //           culvert inlet.
    """
    culvert.kk = Params[code][K]
    culvert.mm = Params[code][M]
    arg = h / culvert.yFull / culvert.kk

    if Params[code][FORM] == 1.0:
        q = getForm1Flow(h, culvert)
    else:
        # q = ad * (arg)^(1/mm)
        q = culvert.ad * (arg ** (1.0 / culvert.mm))
    # dQdH = q / h / mm
    # Guard against h=0
    culvert.dQdH = (q / max(h, 1e-12)) / culvert.mm
    return q


def getSubmergedFlow(code: int, h: float, culvert: Culvert) -> float:
    """
    //  Input:   code  = culvert type code number
    //           h     = inlet head (ft)
    //           culvert = pointer to a culvert data structure
    //  Output:  returns flow rate;
    //           computes value of Dqdh
    //  Purpose: computes flow rate and its derivative for submerged
    //           culvert inlet.
    """
    cc = Params[code][C]
    yy = Params[code][Y]
    arg = (h / culvert.yFull - yy + culvert.scf) / cc

    if arg <= 0.0:
        culvert.dQdH = 0.0
        return BIG

    q = math.sqrt(arg) * culvert.ad
    culvert.dQdH = 0.5 * q / arg / culvert.yFull / cc
    return q


def getTransitionFlow(code: int, h: float, h1: float, h2: float, culvert: Culvert) -> float:
    """
    //  Input:   code    = culvert type code number
    //           h       = inlet water depth above culvert invert (ft)
    //           h1      = head limit for unsubmerged condition (ft)
    //           h2      = head limit for submerged condition (ft)
    //           culvert = pointer to a culvert data structure
    //  Output:  returns flow rate (cfs);
    //           computes value of Dqdh (cfs/ft)
    //  Purpose: computes flow rate and its derivative for inlet-controlled flow
    //           when inlet water depth lies in the transition range between
    //           submerged and unsubmerged conditions.
    """
    q1 = getUnsubmergedFlow(code, h1, culvert)
    q2 = getSubmergedFlow(code, h2, culvert)
    q = q1 + (q2 - q1) * (h - h1) / (h2 - h1)
    culvert.dQdH = (q2 - q1) / (h2 - h1)
    return q


def form1Eqn(yc: float, culvert: Culvert) -> float:
    """
    //  Input:   yc = critical depth
    //  Output:  returns residual error
    //  Purpose: evaluates the error in satisfying FHWA culvert Equation Form1:
    //
    //  h/yFull + 0.5*s = yc/yFull + yh/2/yFull + K[ac/aFull*sqrt(g*yh/yFull)]^M
    //
    //  for a given value of critical depth yc where:
    //    h = inlet depth above culvert invert
    //    s = culvert slope
    //    yFull = full depth of culvert
    //    yh = hydraulic depth at critical depth
    //    ac = flow area at critical depth
    //    g = accel. of gravity
    //    K and M = coefficients
    """
    ac = culvert.xsect.area(yc)
    wc = culvert.xsect.top_width(yc)
    yh = (ac / wc) if wc > 0.0 else 0.0

    culvert.qc = ac * math.sqrt(GRAVITY * yh)
    return (
        culvert.hPlus
        - yc / culvert.yFull
        - yh / (2.0 * culvert.yFull)
        - culvert.kk * ((culvert.qc / culvert.ad) ** culvert.mm)
    )


def getForm1Flow(h: float, culvert: Culvert) -> float:
    """
    //  Input:   h       = inlet water depth above culvert invert
    //           culvert = pointer to a culvert data structure
    //  Output:  returns inlet controlled flow rate
    //  Purpose: computes inlet-controlled flow rate for unsubmerged culvert
    //           using FHWA Equation Form1.
    //
    //  See pages 195-196 of FHWA HEC-5 (2001) for details.
    """
    # --- save re-used terms in culvert structure
    culvert.hPlus = h / culvert.yFull + culvert.scf

    # --- use Ridder's method to solve Equation Form 1 for critical depth
    #     between a range of 0.01h and h
    a = max(1.0e-6, 0.01 * h)
    b = max(a * 1.01, h)  # ensure b > a even for tiny h

    def F(yc: float) -> float:
        return form1Eqn(yc, culvert)

    # Try to ensure a sign change by adaptive scanning if needed
    def bracket_for_ridder(f, lo, hi, max_sub=40):
        flo = f(lo)
        fhi = f(hi)
        if flo == 0.0:
            return lo, lo
        if fhi == 0.0:
            return hi, hi
        if flo * fhi < 0.0:
            return lo, hi
        # Scan within [lo, hi]
        for k in range(1, max_sub + 1):
            x = lo + (hi - lo) * (k / (max_sub + 1))
            fx = f(x)
            if flo * fx < 0.0:
                return lo, x
            if fx * fhi < 0.0:
                return x, hi
        # As a last resort, expand b
        scale = 2.0
        for _ in range(10):
            hi *= scale
            fhi = f(hi)
            if flo * fhi < 0.0:
                return lo, hi
        # Give up: return original; Ridder will raise
        return lo, hi

    a, b = bracket_for_ridder(F, a, b)
    try:
        yc = ridder(F, a, b, tol=1.0e-3, max_iter=100)
    except Exception:
        # Fallback to midpoint if root bracketing fails
        yc = 0.5 * (a + b)
        # Evaluate once to set qc consistently
        _ = F(yc)

    # --- return the flow value used in evaluating Equation Form 1
    return culvert.qc


# -----------------------------------------------------------------------------
#  Culvert inlet-controlled flow (Python equivalent of culvert_getInflow logic)
#    - This form computes Q(h) given h above invert.
# -----------------------------------------------------------------------------
def inlet_controlled_flow(xsect: Xsect, slope: float, h: float) -> tuple[float, float, int, float]:
    """
    Returns (q, dQdH, condition, yRatio) for a given inlet head h (ft)
    using the same logic as culvert_getInflow() in the original C.

    condition: 0 = transition, 1 = unsubmerged, 2 = submerged
    yRatio = h / yFull
    """
    code = xsect.culvertCode
    if code <= 0 or code > MAX_CULVERT_CODE:
        return (0.0, 0.0, 1, 0.0)

    culvert = Culvert(
        yFull=xsect.yFull,
        scf=0.0,
        dQdH=0.0,
        qc=0.0,
        kk=0.0,
        mm=0.0,
        ad=xsect.aFull * math.sqrt(xsect.yFull),
        hPlus=0.0,
        xsect=xsect
    )

    # --- slope correction factor (-7 for mitered inlets, 0.5 for others)
    if code in (5, 37, 46):
        culvert.scf = -7.0 * slope
    else:
        culvert.scf = 0.5 * slope

    y = h  # head above invert (ft)
    # --- check for submerged flow (based on FHWA criteria of Q/AD > 4)
    y2 = culvert.yFull * (16.0 * Params[code][C] + Params[code][Y] - culvert.scf)
    if y >= y2:
        q = getSubmergedFlow(code, y, culvert)
        condition = 2
    else:
        # --- unsubmerged flow if h <= 0.95 yFull
        y1 = 0.95 * culvert.yFull
        if y <= y1:
            q = getUnsubmergedFlow(code, y, culvert)
            condition = 1
        else:
            q = getTransitionFlow(code, y, y1, y2, culvert)
            condition = 0

    return q, culvert.dQdH, condition, (y / culvert.yFull if culvert.yFull > 0.0 else 0.0)


# -----------------------------------------------------------------------------
#  Headwater solver: find h such that inlet_controlled_flow(h) = Q_target
# -----------------------------------------------------------------------------
def solve_headwater_depth_for_Q(
    xsect: Xsect,
    slope: float,
    Q_target: float,
    h_min: float = 0.0,
    h_max: Optional[float] = None,
    tol_h: float = 1e-5,
    tol_q: float = 1e-6,
    max_iter: int = 120
) -> tuple[float, float, int, float]:
    """
    Solve for headwater depth h (ft above invert) so that inlet-controlled Q(h) = Q_target.

    Returns (h, q, condition, yRatio).
    """
    if Q_target <= 0.0:
        return 0.0, 0.0, 1, 0.0

    # Initial bounds
    if h_max is None:
        h_max = max(xsect.yFull * 10.0, 1.0)  # generous upper bound

    def f(h: float) -> float:
        q, *_ = inlet_controlled_flow(xsect, slope, h)
        return q - Q_target

    # Ensure bracket: f(h_min) <= 0, f(h_max) >= 0
    f_lo = f(h_min)
    f_hi = f(h_max)
    # Expand upper bound if needed
    expand_count = 0
    while f_hi < 0.0 and expand_count < 30:
        h_max *= 2.0
        f_hi = f(h_max)
        expand_count += 1

    if f_lo > 0.0:
        # If even at near-zero head, computed Q >= target (rare), clamp to h_min
        q, cond, yR = inlet_controlled_flow(xsect, slope, h_min)[0], inlet_controlled_flow(xsect, slope, h_min)[2], inlet_controlled_flow(xsect, slope, h_min)[3]
        return h_min, q, cond, yR

    if f_hi < 0.0:
        # Could not bracket; return last upper bound result
        q, cond, yR = inlet_controlled_flow(xsect, slope, h_max)[0], inlet_controlled_flow(xsect, slope, h_max)[2], inlet_controlled_flow(xsect, slope, h_max)[3]
        return h_max, q, cond, yR

    # Bisection
    a, b = h_min, h_max
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = f(m)
        if abs(fm) < tol_q or abs(b - a) < tol_h:
            q, dQdH, condition, yR = inlet_controlled_flow(xsect, slope, m)
            return m, q, condition, yR
        if f_lo * fm <= 0.0:
            b, f_hi = m, fm
        else:
            a, f_lo = m, fm

    # Return best midpoint if not converged
    m = 0.5 * (a + b)
    q, dQdH, condition, yR = inlet_controlled_flow(xsect, slope, m)
    return m, q, condition, yR


# -----------------------------------------------------------------------------
#  Critical depth in culvert barrel  (HEC-RAS Eq. 9 / USGS criterion)
# -----------------------------------------------------------------------------
def critical_depth_in_culvert(xsect: Xsect, Q: float) -> float:
    """
    Compute critical depth dc (ft) in the culvert barrel such that the
    Froude-number criterion is satisfied:

        A(dc)^3 / T(dc) = Q^2 / g          (HEC-RAS Culvert Hydraulics Eq. 9)

    Uses bisection on the culvert section's area/top-width functions.
    For a rectangular barrel, the direct formula dc = (Q^2 / (g * B^2))^(1/3)
    is used (HEC-RAS Eq. 10).

    Returns dc in feet; falls back to yFull if the flow would require full-
    barrel pressure (critical depth >= yFull).
    """
    if Q <= 0.0:
        return 0.0

    # Shortcut: rectangular section has a closed-form solution
    if isinstance(xsect, RectangularXsect):
        q_unit = Q / xsect.B          # discharge per unit width
        return min((q_unit ** 2 / GRAVITY) ** (1.0 / 3.0), xsect.yFull)

    # General iterative bisection: find y s.t. f(y) = A³/T - Q²/g = 0
    target = Q * Q / GRAVITY

    y_lo = 1.0e-4 * xsect.yFull
    y_hi = xsect.yFull

    def _f(y: float) -> float:
        A = xsect.area(y)
        T = xsect.top_width(y)
        return (A ** 3 / T - target) if T > 0.0 else float('inf')

    f_lo = _f(y_lo)
    f_hi = _f(y_hi)

    # If even at full depth the criterion isn't satisfied, critical depth
    # equals yFull (pressure flow conditions).
    if f_hi <= 0.0:
        return xsect.yFull
    # If at near-zero depth the criterion is already satisfied, return y_lo.
    if f_lo >= 0.0:
        return y_lo

    for _ in range(80):
        y_m = 0.5 * (y_lo + y_hi)
        f_m = _f(y_m)
        if abs(f_m) < 1.0e-9 * max(target, 1.0) or (y_hi - y_lo) < 1.0e-7:
            return y_m
        if f_lo * f_m <= 0.0:
            y_hi = y_m
            f_hi = f_m
        else:
            y_lo = y_m
            f_lo = f_m

    return 0.5 * (y_lo + y_hi)


def culvert_manning_discharge(xsect: Xsect, depth: float, n_value: float, slope: float) -> float:
    """Compute culvert discharge from Manning's equation for partially full flow."""
    if depth <= 0.0 or n_value <= 0.0 or slope <= 0.0:
        return 0.0
    area = xsect.area(depth)
    radius = xsect.hydraulic_radius(depth)
    if area <= 0.0 or radius <= 0.0:
        return 0.0
    return (1.49 / n_value) * area * (radius ** (2.0 / 3.0)) * math.sqrt(slope)


def culvert_velocity(xsect: Xsect, Q: float, depth: float) -> float:
    area = xsect.area(depth)
    if area <= 0.0:
        return 0.0
    return Q / area


def culvert_specific_energy(xsect: Xsect, Q: float, depth: float) -> float:
    velocity = culvert_velocity(xsect, Q, depth)
    return depth + velocity * velocity / (2.0 * GRAVITY)


def culvert_friction_slope(xsect: Xsect, Q: float, n_value: float, depth: float) -> float:
    if depth <= 0.0 or n_value <= 0.0:
        return 0.0
    area = xsect.area(depth)
    radius = xsect.hydraulic_radius(depth)
    if area <= 0.0 or radius <= 0.0:
        return 0.0
    conveyance = (1.49 / n_value) * area * (radius ** (2.0 / 3.0))
    if conveyance <= 0.0:
        return 0.0
    return (Q / conveyance) ** 2


def solve_depth_for_specific_energy(
    xsect: Xsect,
    Q: float,
    target_energy: float,
    branch: str = "subcritical",
    tol_depth: float = 1.0e-6,
    samples: int = 240,
) -> tuple[float, float]:
    """
    Solve for a depth on a requested specific-energy branch.

    Returns (depth, residual), where residual is the signed specific-energy error.
    If an exact root on the requested branch cannot be bracketed, the depth with the
    least absolute residual on that branch is returned. This mirrors the HEC-RAS
    least-error behavior for supercritical culvert branch attempts.
    """
    if Q <= 0.0:
        return 0.0, -target_energy

    dc = critical_depth_in_culvert(xsect, Q)
    eps = max(1.0e-6, 1.0e-6 * xsect.yFull)

    if branch == "supercritical":
        lo = eps
        hi = max(eps, min(dc, xsect.yFull - eps))
    else:
        lo = min(max(dc, eps), xsect.yFull - eps)
        hi = xsect.yFull - eps

    if hi <= lo:
        depth = max(eps, min(dc, xsect.yFull - eps))
        return depth, culvert_specific_energy(xsect, Q, depth) - target_energy

    def residual(depth: float) -> float:
        return culvert_specific_energy(xsect, Q, depth) - target_energy

    step = (hi - lo) / float(max(samples - 1, 1))
    best_depth = lo
    best_residual = residual(lo)
    prev_depth = lo
    prev_residual = best_residual
    bracket = None

    for i in range(1, samples):
        depth = lo + i * step
        res = residual(depth)
        if abs(res) < abs(best_residual):
            best_depth = depth
            best_residual = res
        if prev_residual == 0.0:
            return prev_depth, 0.0
        if prev_residual * res < 0.0:
            bracket = (prev_depth, depth)
            break
        prev_depth = depth
        prev_residual = res

    if bracket is None:
        return best_depth, best_residual

    a, b = bracket
    fa = residual(a)
    fb = residual(b)
    for _ in range(80):
        m = 0.5 * (a + b)
        fm = residual(m)
        if abs(fm) < 1.0e-10 or abs(b - a) < tol_depth:
            return m, fm
        if abs(fm) < abs(best_residual):
            best_depth = m
            best_residual = fm
        if fa * fm <= 0.0:
            b = m
            fb = fm
        else:
            a = m
            fa = fm
    return best_depth, best_residual


def solve_normal_depth_in_culvert(
    xsect: Xsect,
    Q: float,
    n_value: float,
    slope: float,
    tol_depth: float = 1.0e-6,
    tol_q: float = 1.0e-6,
    max_iter: int = 100,
) -> float:
    """
    Solve Manning normal depth inside the culvert barrel.

    Following HEC-RAS culvert guidance, if the computed normal depth exceeds the
    culvert rise, return the culvert rise (yFull).

    Note: for circular (and similar) sections the Manning capacity peaks at
    approximately 93.8 % of y_full before decreasing back to the full-pipe value.
    If Q exceeds the full-pipe Manning capacity but is still less than the peak
    capacity, a valid normal depth exists on the rising limb (below ~0.938*y_full).
    Without this correction the bisection would miss that solution and incorrectly
    return y_full.
    """
    if Q <= 0.0:
        return 0.0
    if n_value <= 0.0:
        return xsect.yFull
    if slope <= 0.0:
        return xsect.yFull

    depth_lo = max(1.0e-6, 1.0e-6 * xsect.yFull)
    depth_hi = xsect.yFull

    q_lo = culvert_manning_discharge(xsect, depth_lo, n_value, slope)
    q_hi = culvert_manning_discharge(xsect, depth_hi, n_value, slope)

    if q_hi <= Q:
        # Full-pipe Manning capacity is less than Q.
        # For non-rectangular sections (e.g. circular), the capacity may peak
        # at a depth below y_full (~0.938*y_full for a circle).  Search for
        # the peak by sampling the capacity curve; if Q <= Q_peak, a valid
        # normal depth exists on the rising limb.
        n_scan = 40
        q_peak = q_hi
        d_peak = depth_hi
        for i in range(n_scan, 0, -1):
            d_try = depth_lo + (depth_hi - depth_lo) * (i / n_scan)
            q_try = culvert_manning_discharge(xsect, d_try, n_value, slope)
            if q_try > q_peak:
                q_peak = q_try
                d_peak = d_try
        if q_peak > Q:
            # Bisect on the rising limb [depth_lo, d_peak]
            depth_hi = d_peak
            q_hi = q_peak
        else:
            # Q truly exceeds peak capacity — pressure flow required
            return xsect.yFull

    if q_lo >= Q:
        return depth_lo

    a, b = depth_lo, depth_hi
    fa = q_lo - Q
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = culvert_manning_discharge(xsect, m, n_value, slope) - Q
        if abs(fm) < tol_q or abs(b - a) < tol_depth:
            return m
        if fa * fm <= 0.0:
            b = m
        else:
            a = m
            fa = fm
    return 0.5 * (a + b)


def direct_step_culvert_upstream_energy(
    xsect: Xsect,
    Q: float,
    n_value: float,
    slope: float,
    length: float,
    tailwater_depth: float,
    step_depth: Optional[float] = None,
) -> tuple[float, float, str]:
    """
    Compute the upstream inside-barrel state using a direct-step profile.

    The downstream inside-barrel depth starts at max(tailwater depth, critical depth).
    The profile is marched upstream by increasing depth increments. If the barrel fills
    before the upstream end, the remaining distance is handled as full flow.

    Returns (specific_energy_upstream_inside_barrel, upstream_depth_inside_barrel, mode).
    mode is one of: 'partial', 'partial-full', 'full-from-outlet', 'supercritical-least-error'.
    """
    if Q <= 0.0:
        return 0.0, 0.0, 'partial'

    dc = critical_depth_in_culvert(xsect, Q)
    y_full = xsect.yFull
    eps = max(1.0e-6, 1.0e-6 * y_full)
    y_ds = min(max(tailwater_depth, dc), y_full)

    if step_depth is None:
        step_depth = min(max(0.01, 0.02 * y_full), 0.05)

    if y_ds >= y_full - eps:
        sf_full = culvert_friction_slope(xsect, Q, n_value, y_full - eps)
        e_full = culvert_specific_energy(xsect, Q, y_full - eps)
        return e_full + max(0.0, sf_full - slope) * length, y_full - eps, 'full-from-outlet'

    def dx_to_depth(y_from: float, e_from: float, y_to: float) -> Optional[float]:
        sf_from = culvert_friction_slope(xsect, Q, n_value, y_from)
        sf_to = culvert_friction_slope(xsect, Q, n_value, y_to)
        sf_avg = 0.5 * (sf_from + sf_to)
        denom = slope - sf_avg
        if abs(denom) < 1.0e-12:
            return None
        e_to = culvert_specific_energy(xsect, Q, y_to)
        dx = (e_from - e_to) / denom
        if not math.isfinite(dx) or dx <= 0.0:
            return None
        return dx

    distance = 0.0
    y_cur = max(y_ds, eps)
    e_cur = culvert_specific_energy(xsect, Q, y_cur)

    while distance < length - 1.0e-8:
        if y_cur >= y_full - eps:
            sf_full = culvert_friction_slope(xsect, Q, n_value, y_full - eps)
            remaining = length - distance
            return e_cur + max(0.0, sf_full - slope) * remaining, y_full, 'partial-full'

        dy = min(step_depth, y_full - y_cur)
        dx = None
        y_next = y_cur
        for _ in range(10):
            y_try = min(y_cur + dy, y_full)
            dx_try = dx_to_depth(y_cur, e_cur, y_try)
            if dx_try is not None:
                y_next = y_try
                dx = dx_try
                break
            dy *= 0.5
            if dy <= eps:
                break

        if dx is None:
            # Supercritical branch fallback: choose the least-error depth on the
            # supercritical branch for the current energy target.
            y_super, _residual = solve_depth_for_specific_energy(
                xsect,
                Q,
                e_cur,
                branch='supercritical',
            )
            return culvert_specific_energy(xsect, Q, y_super), y_super, 'supercritical-least-error'

        if distance + dx >= length:
            remaining = length - distance

            def g(y_target: float) -> float:
                dx_target = dx_to_depth(y_cur, e_cur, y_target)
                if dx_target is None:
                    return float('inf')
                return dx_target - remaining

            a = y_cur
            b = y_next
            fa = g(a)
            fb = g(b)
            best_y = a
            best_err = abs(fa)
            for i in range(1, 80):
                y_sample = a + (b - a) * (i / 79.0)
                err = g(y_sample)
                if math.isfinite(err) and abs(err) < best_err:
                    best_y = y_sample
                    best_err = abs(err)
                if math.isfinite(fa) and math.isfinite(err) and fa * err <= 0.0:
                    b = y_sample
                    fb = err
                    break
                fa = err
                a = y_sample

            if math.isfinite(fa) and math.isfinite(fb) and fa * fb <= 0.0:
                lo = a
                hi = b
                flo = g(lo)
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    fmid = g(mid)
                    if not math.isfinite(fmid):
                        break
                    if abs(fmid) < 1.0e-8 or abs(hi - lo) < eps:
                        best_y = mid
                        break
                    if flo * fmid <= 0.0:
                        hi = mid
                    else:
                        lo = mid
                        flo = fmid

            return culvert_specific_energy(xsect, Q, best_y), best_y, 'partial'

        distance += dx
        y_cur = y_next
        e_cur = culvert_specific_energy(xsect, Q, y_cur)

    return e_cur, y_cur, 'partial'


# -----------------------------------------------------------------------------
#  Debug report (translated; optional use)
# -----------------------------------------------------------------------------
def report_CulvertControl(link_id: str, q0: float, q: float, condition: int, yRatio: float) -> None:
    """
    //  Used for debugging only
    //
    //  static   char* conditionTxt[] = {"transition", "unsubmerged", "submerged"};
    //  fprintf(Frpt.file,
    //          "\n  %11s: %8s Culvert %s flow reduced from %.3f to %.3f cfs for %s flow (%.2f).",
    //          theDate, theTime, Link[j].ID, q0, q, conditionTxt[condition], yRatio);
    """
    conditionTxt = ["transition", "unsubmerged", "submerged"]
    print(f"Culvert {link_id}: flow reduced from {q0:.3f} to {q:.3f} cfs for {conditionTxt[condition]} flow ({yRatio:.2f}).")


# -----------------------------------------------------------------------------
#  Simple CLI to solve headwater depth for a given Q
# -----------------------------------------------------------------------------
def compute_headwater(
    code: int,
    slope: float,
    Q: float,
    shape: str,
    diam_ft: Optional[float] = None,
    width_ft: Optional[float] = None,
    height_ft: Optional[float] = None,
    h_max: Optional[float] = None,
    verbose: bool = True,
) -> tuple[float, float, int, float]:
    """
    Programmatic equivalent of the CLI invocation.

    Parameters match the CLI: `code`, `slope`, `Q`, and `shape` ("circular" or "rect").
    For `circular`, provide `diam_ft`; for `rect`, provide `width_ft` and `height_ft`.

    Returns the tuple `(h, q, condition, yRatio)` and prints the same report
    as the CLI when `verbose=True`.
    """
    if not (1 <= code <= MAX_CULVERT_CODE):
        raise ValueError(f"code must be between 1 and {MAX_CULVERT_CODE}")

    if shape == "circular":
        if diam_ft is None:
            raise ValueError("diam_ft is required for circular shape")
        x = CircularXsect(diameter_ft=diam_ft, culvert_code=code)
    elif shape == "rect":
        if width_ft is None or height_ft is None:
            raise ValueError("width_ft and height_ft are required for rect shape")
        x = RectangularXsect(width_ft=width_ft, height_ft=height_ft, culvert_code=code)
    else:
        raise ValueError("shape must be 'circular' or 'rect'")

    h, q, condition, yR = solve_headwater_depth_for_Q(x, slope=slope, Q_target=Q, h_min=0.0, h_max=h_max)

    if verbose:
        conditionTxt = ["transition", "unsubmerged", "submerged"]
        print(f"code={code}  shape={shape}  slope={slope:g}")
        print(f"yFull={x.yFull:.4f} ft,  aFull={x.aFull:.4f} ft^2,  ad={x.aFull*math.sqrt(x.yFull):.4f}")
        print(f"Headwater h = {h:.6f} ft above invert")
        print(f"Computed Q(h) = {q:.6f} cfs  | regime = {conditionTxt[condition]}  | h/yFull = {yR:.4f}")

    return h #, q, condition, yR
"""
def _cli():
    import argparse

    p = argparse.ArgumentParser(
        description="Solve headwater depth for a given culvert code, geometry, slope, and flow (inlet control, FHWA HEC-5)."
    )
    p.add_argument("--code", type=int, required=True, help=f"Culvert code (1..{MAX_CULVERT_CODE}) from FHWA table.")
    p.add_argument("--slope", type=float, required=True, help="Conduit slope (ft/ft).")
    p.add_argument("--Q", type=float, required=True, help="Target flow rate (cfs).")

    sub = p.add_subparsers(dest="shape", required=True)
    c = sub.add_parser("circular", help="Circular culvert")
    c.add_argument("--diam-ft", type=float, required=True, help="Diameter (ft)")

    r = sub.add_parser("rect", help="Rectangular box culvert")
    r.add_argument("--width-ft", type=float, required=True, help="Width B (ft)")
    r.add_argument("--height-ft", type=float, required=True, help="Height H (ft)")

    p.add_argument("--h-max", type=float, default=None, help="Optional upper bound for head (ft).")
    args = p.parse_args()

    if not (1 <= args.code <= MAX_CULVERT_CODE):
        raise SystemExit(f"--code must be between 1 and {MAX_CULVERT_CODE}")

    h, q, condition, yR = compute_headwater(
        code=args.code,
        slope=args.slope,
        Q=args.Q,
        shape=args.shape,
        diam_ft=getattr(args, "diam_ft", None),
        width_ft=getattr(args, "width_ft", None),
        height_ft=getattr(args, "height_ft", None),
        h_max=args.h_max,
        verbose=True,
    )
    return 0


if __name__ == "__main__":
    _cli()
"""