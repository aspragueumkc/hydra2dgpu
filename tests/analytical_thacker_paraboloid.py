"""
Thacker (1981) analytical solution for a paraboloid oscillating basin.

Reference:
    Thacker, W.C. (1981), "Some exact solutions to the nonlinear shallow-water
    wave equations," J. Fluid Mech., 107, 499-508.

This is a port of the ANUGA validation test `analytical_exact/paraboloid_basin/`
to a standalone Python module with no ANUGA dependencies.

Physical setup
--------------
A paraboloid basin with bed:
    z_b(r) = -D0 * (1 - r²/L²)
where r = sqrt(x² + y²).

The bed is deepest (most negative) at the centre (r=0, z_b=-D0) and rises
parabolically to z_b=0 at the rim (r=L).

The water surface w(x,y,t) oscillates with frequency:
    omega = 2/L * sqrt(2*g*D0)
    T = 2π/omega  (period)

At t=0 the water surface is flat (w=0 everywhere), so the initial depth is:
    h(r,0) = max(0, -z_b(r))  inside the initial water radius R0
          = 0                 outside R0

Parameters (from ANUGA paraboloid_basin validation test):
    D0  = 1000.0 m   (characteristic depth scale)
    L   = 2500.0 m   (basin length scale)
    R0  = 2000.0 m   (initial water radius)
    g   = 9.81 m/s²

With these values:
    A     = (L⁴ - R0⁴)/(L⁴ + R0⁴) ≈ 0.6
    omega = 2/L * sqrt(2*g*D0) ≈ 0.2804 rad/s
    T     = 2π/omega ≈ 22.4 s

Usage
------
    w, u, v, h = thacker_paraboloid(x, y, t, D0, L, R0, g)
    # Returns stage w, x-velocity u, y-velocity v, depth h
    # All arrays are the same shape as x, y.
    # Wet-dry is handled: if w < z_b the cell is dry (h=0, u=v=0).
"""

import numpy as np


D0_DEFAULT = 1000.0
L_DEFAULT  = 2500.0
R0_DEFAULT = 2000.0
G_DEFAULT  = 9.81


def bed_elevation(x: np.ndarray, y: np.ndarray,
                  D0: float = D0_DEFAULT, L: float = L_DEFAULT) -> np.ndarray:
    """Bed elevation z_b(x,y) = -D0*(1 - r²/L²) where r=sqrt(x²+y²)."""
    r2 = x*x + y*y
    return -D0 * (1.0 - r2 / (L*L))


def thacker_paraboloid(
    x: np.ndarray,
    y: np.ndarray,
    t: float,
    D0: float = D0_DEFAULT,
    L: float = L_DEFAULT,
    R0: float = R0_DEFAULT,
    g: float = G_DEFAULT,
):
    """
    Compute Thacker analytical solution for paraboloid basin at time t.

    Parameters
    ----------
    x, y : ndarray
        Node or cell-centroid coordinates (same shape).
    t : float
        Time in seconds.
    D0, L, R0, g : float
        Basin parameters (see module docstring).

    Returns
    -------
    w : ndarray
        Water surface elevation (stage) at each point.
    u : ndarray
        x-velocity at each point (m/s).
    v : ndarray
        y-velocity at each point (m/s).
    h : ndarray
        Water depth at each point (max(0, w - z_b)). Dry cells have h=0, u=v=0.
    """
    A = (L**4 - R0**4) / (L**4 + R0**4)
    omega = 2.0 / L * np.sqrt(2.0 * g * D0)

    r = np.sqrt(x*x + y*y)
    cos_omega_t = np.cos(omega * t)
    sin_omega_t = np.sin(omega * t)

    denom = 1.0 - A * cos_omega_t
    denom2 = denom * denom
    sqrt_1mA2 = np.sqrt(1.0 - A*A)

    r2_L2 = r * r / (L * L)

    w = (D0 * sqrt_1mA2 / denom) - D0 - D0 * r2_L2 * ((1.0 - A*A) / denom2 - 1.0)

    u = 0.5 * omega * x * A * sin_omega_t / (1.0 - A * cos_omega_t)
    v = 0.5 * omega * y * A * sin_omega_t / (1.0 - A * cos_omega_t)

    z_b = bed_elevation(x, y, D0, L)

    dry = w < z_b
    w[dry] = z_b[dry]
    u[dry] = 0.0
    v[dry] = 0.0

    h = np.maximum(0.0, w - z_b)

    return w, u, v, h


def oscillation_period(D0: float = D0_DEFAULT, L: float = L_DEFAULT,
                       R0: float = R0_DEFAULT, g: float = G_DEFAULT) -> float:
    """Return the oscillation period T = 2π/omega."""
    omega = 2.0 / L * np.sqrt(2.0 * g * D0)
    return 2.0 * np.pi / omega


def initial_condition(
    x: np.ndarray, y: np.ndarray,
    D0: float = D0_DEFAULT, L: float = L_DEFAULT, R0: float = R0_DEFAULT,
) -> np.ndarray:
    """
    Initial depth h(x,y,0) for the Thacker paraboloid test.

    At t=0 the water surface is flat (w=0 everywhere).
    Inside the initial radius R0: h = D0*(2R0²/L² - r²/L²)
    Outside R0: h = 0 (dry bed).
    """
    r2 = x*x + y*y
    r2_L2 = r2 / (L*L)
    h = D0 * (2.0 * R0*R0 / (L*L) - r2_L2)
    h = np.maximum(0.0, h)
    return h
