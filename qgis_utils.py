"""Utility helpers for QGIS integration: extract cross-section geometry
from 3D vector lines and simple utilities.

These functions use PyQGIS types and expect to run inside QGIS.
"""
from qgis.core import QgsGeometry

def _as_point_list(geom):
    # Return a list of QgsPoint-like objects from a QgsGeometry (polyline)
    try:
        if geom.isMultipart():
            m = geom.asMultiPolyline()
            if m and len(m) > 0:
                return m[0]
            return []
        else:
            return geom.asPolyline()
    except Exception:
        try:
            # fall back to generic asPolyline for some QGIS versions
            return geom.asPolyline()
        except Exception:
            return []

def extract_xs_from_line(geom, samples=50):
    """Extract station/elevation pairs from a QgsGeometry polyline.

    - geom: QgsGeometry (polyline or multiline)
    - samples: number of points to sample along the line (>=2)

    Returns: list of (station, elevation) tuples where station starts at 0.
    """
    pts = _as_point_list(geom)
    if not pts:
        return []
    # compute segment lengths and total length
    seg_lengths = []
    total = 0.0
    for i in range(1, len(pts)):
        a = pts[i-1]
        b = pts[i]
        # use 2D distance
        d = a.distance(b)
        seg_lengths.append(d)
        total += d
    if total <= 0 or samples < 2:
        # return original vertices as station/elev
        out = []
        s = 0.0
        for p in pts:
            z = getattr(p, 'z', None)
            try:
                z = p.z()
            except Exception:
                pass
            out.append((s, float(z) if z is not None else 0.0))
        return out

    # compute sample distances
    sample_ds = [i * total / (samples - 1) for i in range(samples)]
    out = []
    # walk segments and interpolate
    seg_idx = 0
    seg_off = 0.0
    cum = 0.0
    for dtarget in sample_ds:
        # advance to segment containing dtarget
        while seg_idx < len(seg_lengths) and (cum + seg_lengths[seg_idx]) < dtarget:
            cum += seg_lengths[seg_idx]
            seg_idx += 1
        if seg_idx >= len(seg_lengths):
            # use last point
            p = pts[-1]
            z = getattr(p, 'z', None)
            try:
                z = p.z()
            except Exception:
                pass
            out.append((dtarget, float(z) if z is not None else 0.0))
            continue
        a = pts[seg_idx]
        b = pts[seg_idx+1]
        segd = seg_lengths[seg_idx]
        if segd == 0:
            t = 0.0
        else:
            t = (dtarget - cum) / segd
        # interpolate
        x = a.x() + (b.x() - a.x()) * t
        y = a.y() + (b.y() - a.y()) * t
        # interpolate z if available
        za = None
        zb = None
        try:
            za = a.z()
        except Exception:
            try:
                za = getattr(a, 'z', None)
            except Exception:
                za = None
        try:
            zb = b.z()
        except Exception:
            try:
                zb = getattr(b, 'z', None)
            except Exception:
                zb = None
        if za is None and zb is None:
            z = 0.0
        else:
            za = 0.0 if za is None else za
            zb = 0.0 if zb is None else zb
            z = za + (zb - za) * t
        out.append((dtarget, float(z)))
    return out
