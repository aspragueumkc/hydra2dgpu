from qgis.utils import qgsfunction
from qgis.core import QgsGeometry
import math

def _profile_from_geom(geom):
    # Build (station, elevation) profile from 3D vertices.
    verts = [v for v in geom.vertices()]
    if len(verts) < 2:
        return None

    prof = []
    s = 0.0

    z0 = verts[0].z()
    if math.isnan(z0):
        return None
    prof.append((0.0, float(z0)))

    prev = verts[0]
    for v in verts[1:]:
        z = v.z()
        if math.isnan(z):
            continue

        dx = v.x() - prev.x()
        dy = v.y() - prev.y()
        ds = math.hypot(dx, dy)

        # Skip duplicate XY vertices
        if ds <= 0.0:
            prev = v
            continue

        s += ds
        prof.append((s, float(z)))
        prev = v

    return prof if len(prof) >= 2 else None

def _area_perimeter_at_wse(profile, wse):
    # Returns wetted area A and wetted perimeter P for a trial water surface elevation.
    A = 0.0
    P = 0.0

    for i in range(len(profile) - 1):
        x1, z1 = profile[i]
        x2, z2 = profile[i + 1]
        dx = abs(x2 - x1)
        if dx <= 0.0:
            continue

        seg_len = math.hypot(x2 - x1, z2 - z1)
        wet1 = z1 < wse
        wet2 = z2 < wse

        if wet1 and wet2:
            d1 = wse - z1
            d2 = wse - z2
            A += 0.5 * (d1 + d2) * dx
            P += seg_len
        elif (not wet1) and (not wet2):
            continue
        else:
            # Segment crosses WSE once
            dz = (z2 - z1)
            if dz == 0.0:
                continue

            t = (wse - z1) / dz
            if t <= 0.0 or t >= 1.0:
                continue

            x_int = x1 + t * (x2 - x1)

            if wet1:
                dx_wet = abs(x_int - x1)
                d_sub = wse - z1
                wet_frac = t
            else:
                dx_wet = abs(x2 - x_int)
                d_sub = wse - z2
                wet_frac = 1.0 - t

            A += 0.5 * d_sub * dx_wet
            P += seg_len * wet_frac

    return A, P

def _q_manning(profile, wse, n, slope):
    A, P = _area_perimeter_at_wse(profile, wse)
    if A <= 0.0 or P <= 0.0:
        return 0.0
    R = A / P
    return (1.0 / n) * A * (R ** (2.0 / 3.0)) * (slope ** 0.5)

@qgsfunction(args='auto', group='Hydraulics')
def normal_depth_xs(geom, q, n, slope, max_iter=80, tol=1e-5, feature=None, parent=None):
    """
    normal_depth_xs(geometry, Q, n, slope, max_iter=80, tol=1e-5) -> depth

    geometry : MultiLineStringZ cross section
    Q        : flow rate (from field)
    n        : Manning roughness (from field)
    slope    : energy slope S (field or constant)

    Returns normal depth (water surface elevation - minimum bed elevation).
    Returns NULL if no valid solution.
    """
    if geom is None or geom.isEmpty():
        return None

    try:
        Q = float(q)
        n = float(n)
        S = float(slope)
    except Exception:
        return None

    if Q <= 0.0 or n <= 0.0 or S <= 0.0:
        return None

    profile = _profile_from_geom(geom)
    if not profile:
        return None

    z_vals = [z for _, z in profile]
    z_min = min(z_vals)
    z_max = max(z_vals)

    # Bracket solution in WSE space
    low = z_min + 1e-9
    high = z_max + 0.01

    q_high = _q_manning(profile, high, n, S)
    span = max(1.0, z_max - z_min)

    expand_count = 0
    while q_high < Q and expand_count < 40:
        high += 0.5 * span
        q_high = _q_manning(profile, high, n, S)
        expand_count += 1

    if q_high < Q:
        return None

    # Bisection solve Q(wse) = target Q
    for _ in range(int(max_iter)):
        mid = 0.5 * (low + high)
        q_mid = _q_manning(profile, mid, n, S)

        if abs(q_mid - Q) <= max(float(tol), float(tol) * Q):
            return mid - z_min

        if q_mid < Q:
            low = mid
        else:
            high = mid

    return 0.5 * (low + high) - z_min
