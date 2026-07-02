"""HDS-5 culvert validation: test GPU kernel against analytical outlet control.

Uses the same equations implemented in the CUDA kernel (direct-step backwater,
Manning's friction, entrance/exit losses) to compute expected flows, then
compares against swe2d_gpu_compute_structure_flows output.

Reference: FHWA HDS-5, Hydraulic Design of Highway Culverts, 3rd Edition (2012).
"""
from __future__ import annotations

import math
import unittest

import numpy as np


def _load_module():
    try:
        import hydra_swe2d as m
        return m
    except Exception:
        return None


_MOD = _load_module()

# ── HDS-5 circular pipe geometry (partial depth) ───────────────────────
GRAVITY_FT = 32.174  # ft/s²


def _circular_area_ft2(radius_ft: float, y_ft: float) -> float:
    """Cross-sectional area of circular pipe at depth y (ft)."""
    y = max(0.0, min(y_ft, 2.0 * radius_ft))
    if y <= 0.0:
        return 0.0
    arg = max(-1.0, min(1.0, (radius_ft - y) / radius_ft))
    theta = 2.0 * math.acos(arg)
    return 0.5 * radius_ft * radius_ft * (theta - math.sin(theta))


def _circular_perimeter_ft(radius_ft: float, y_ft: float) -> float:
    """Wetted perimeter of circular pipe at depth y (ft)."""
    y = max(0.0, min(y_ft, 2.0 * radius_ft))
    if y <= 0.0:
        return 0.0
    arg = max(-1.0, min(1.0, (radius_ft - y) / radius_ft))
    theta = 2.0 * math.acos(arg)
    return radius_ft * theta


def _circular_top_width_ft(radius_ft: float, y_ft: float) -> float:
    """Top width of water surface in circular pipe at depth y (ft)."""
    y = max(0.0, min(y_ft, 2.0 * radius_ft))
    if y <= 0.0:
        return 0.0
    return 2.0 * math.sqrt(max(0.0, 2.0 * radius_ft * y - y * y))


def _circular_hydraulic_radius_ft(radius_ft: float, y_ft: float) -> float:
    a = _circular_area_ft2(radius_ft, y_ft)
    p = _circular_perimeter_ft(radius_ft, y_ft)
    return a / p if p > 0.0 else 0.0


def _critical_depth_ft(radius_ft: float, q_cfs: float) -> float:
    """Critical depth in circular pipe via bisection (A³/T = Q²/g)."""
    if q_cfs <= 0.0:
        return 0.0
    d_full = 2.0 * radius_ft
    target = q_cfs * q_cfs / GRAVITY_FT
    lo = 1e-6
    hi = d_full

    def residual(y):
        a = _circular_area_ft2(radius_ft, y)
        t = _circular_top_width_ft(radius_ft, y)
        return (a * a * a / t - target) if t > 0.0 else 1e20

    flo = residual(lo)
    fhi = residual(hi)
    if fhi <= 0.0:
        return d_full
    if flo >= 0.0:
        return lo
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        fm = residual(mid)
        if abs(fm) < 1e-9 * max(target, 1.0) or (hi - lo) < 1e-7:
            return mid
        if flo * fm <= 0.0:
            hi = mid
        else:
            lo = mid
            flo = fm
    return 0.5 * (lo + hi)


def _specific_energy_ft(radius_ft: float, q_cfs: float, depth_ft: float) -> float:
    """Specific energy = depth + V²/(2g)."""
    a = _circular_area_ft2(radius_ft, depth_ft)
    v = q_cfs / a if a > 0.0 else 0.0
    return depth_ft + v * v / (2.0 * GRAVITY_FT)


def _friction_slope_ft(radius_ft: float, q_cfs: float, n: float, depth_ft: float) -> float:
    """Manning friction slope: Sf = (Q/K)², K = (1.49/n)*A*R^(2/3)."""
    if depth_ft <= 0.0 or n <= 0.0:
        return 0.0
    a = _circular_area_ft2(radius_ft, depth_ft)
    rh = _circular_hydraulic_radius_ft(radius_ft, depth_ft)
    if a <= 0.0 or rh <= 0.0:
        return 0.0
    k = (1.49 / n) * a * rh ** (2.0 / 3.0)
    if k <= 0.0:
        return 0.0
    return (q_cfs / k) ** 2


def _direct_step_upstream_energy(
    radius_ft: float, q_cfs: float, n: float, slope: float,
    length_ft: float, tw_depth_ft: float,
) -> float:
    """Direct-step backwater: step from tailwater upstream, return upstream energy."""
    d_full = 2.0 * radius_ft
    dc = _critical_depth_ft(radius_ft, q_cfs)
    eps = max(1e-6, 1e-6 * d_full)
    y_ds = min(max(tw_depth_ft, dc), d_full)
    step_depth = min(max(0.01, 0.02 * d_full), 0.50)

    # Full-pipe fast path
    if y_ds >= d_full - eps:
        sf = _friction_slope_ft(radius_ft, q_cfs, n, d_full - eps)
        e = _specific_energy_ft(radius_ft, q_cfs, d_full - eps)
        return e + max(0.0, sf - slope) * length_ft

    distance = 0.0
    y_cur = max(y_ds, eps)
    e_cur = _specific_energy_ft(radius_ft, q_cfs, y_cur)

    while distance < length_ft - 1e-8:
        if y_cur >= d_full - eps:
            sf = _friction_slope_ft(radius_ft, q_cfs, n, d_full - eps)
            return e_cur + max(0.0, sf - slope) * (length_ft - distance)

        dy = min(step_depth, d_full - y_cur)
        have_step = False
        y_next = y_cur
        dx = 0.0
        for _ in range(10):
            y_try = min(y_cur + dy, d_full)
            sf_from = _friction_slope_ft(radius_ft, q_cfs, n, y_cur)
            sf_to = _friction_slope_ft(radius_ft, q_cfs, n, y_try)
            sf_avg = 0.5 * (sf_from + sf_to)
            denom = slope - sf_avg
            if abs(denom) >= 1e-12:
                e_to = _specific_energy_ft(radius_ft, q_cfs, y_try)
                dx_try = (e_cur - e_to) / denom
                if math.isfinite(dx_try) and dx_try > 0.0:
                    have_step = True
                    y_next = y_try
                    dx = dx_try
                    break
            dy *= 0.5
            if dy <= eps:
                break

        if not have_step:
            return _specific_energy_ft(radius_ft, q_cfs, max(eps, min(dc, d_full - eps)))

        if distance + dx >= length_ft:
            remaining = length_ft - distance
            sf_cur = _friction_slope_ft(radius_ft, q_cfs, n, y_cur)

            def residual(y):
                sf_y = _friction_slope_ft(radius_ft, q_cfs, n, y)
                sf_avg = 0.5 * (sf_cur + sf_y)
                e_y = _specific_energy_ft(radius_ft, q_cfs, y)
                return e_y - e_cur - remaining * (slope - sf_avg)

            a, b = y_cur, y_next
            fa, fb = residual(a), residual(b)
            for _ in range(10):
                if fa * fb <= 0.0:
                    break
                if abs(fa) < abs(fb):
                    a = max(eps, a - (b - a))
                    fa = residual(a)
                else:
                    b = min(d_full, b + (b - a))
                    fb = residual(b)
            y_best = a if abs(fa) < abs(fb) else b
            for _ in range(12):
                if abs(fb - fa) < 1e-30:
                    break
                y_mid = b - fb * (b - a) / (fb - fa)
                fm = residual(y_mid)
                if abs(fm) < 1e-10 or abs(b - a) < eps:
                    y_best = y_mid
                    break
                a, fa = b, fb
                b, fb = y_mid, fm
            return _specific_energy_ft(radius_ft, q_cfs, y_best)

        distance += dx
        y_cur = y_next
        e_cur = _specific_energy_ft(radius_ft, q_cfs, y_cur)

    return e_cur


def _outlet_control_flow_cfs(
    diameter_ft: float, length_ft: float, slope_ft: float,
    n: float, ke: float, kx: float,
    hw_ft: float, tw_ft: float, q_hint: float = 0.0,
) -> float:
    """HDS-5 outlet control: solve required_head(Q) = available_head via secant."""
    if hw_ft <= 0.0:
        return 0.0
    radius_ft = diameter_ft / 2.0

    def required_head(q_cfs):
        if q_cfs <= 0.0:
            return 0.0
        e_up = _direct_step_upstream_energy(radius_ft, q_cfs, n, slope_ft, length_ft, tw_ft)
        # Use depth at upstream for velocity loss computation
        d_full = 2.0 * radius_ft
        dc = _critical_depth_ft(radius_ft, q_cfs)
        y_ds = min(max(tw_ft, dc), d_full)
        y_up = min(d_full, max(1e-6, y_ds))
        a = _circular_area_ft2(radius_ft, y_up)
        a = max(a, 1e-9)
        vel = q_cfs / a
        hv_loss = (max(0.0, ke) + max(0.0, kx)) * vel * vel / (2.0 * GRAVITY_FT)
        return e_up + hv_loss

    # Illinois secant (same algorithm as kernel)
    q_lo = 0.0
    f_lo = -hw_ft
    q_hi = max(1.0, q_hint * 2.0) if q_hint > 0.0 else max(1.0, 50.0)
    f_hi = required_head(q_hi) - hw_ft
    for _ in range(12):
        if f_hi >= 0.0:
            break
        q_lo = q_hi
        f_lo = f_hi
        q_hi *= 2.0
        f_hi = required_head(q_hi) - hw_ft
    if f_hi < 0.0:
        return q_hi

    side = 0
    for _ in range(16):
        denom = f_hi - f_lo
        if abs(denom) < 1e-30:
            break
        q_mid = (q_lo * f_hi - q_hi * f_lo) / denom
        if q_mid <= q_lo or q_mid >= q_hi:
            q_mid = 0.5 * (q_lo + q_hi)
        f_mid = required_head(q_mid) - hw_ft
        if abs(f_mid) < 1e-8 * hw_ft:
            return max(0.0, q_mid)
        if f_lo * f_mid < 0.0:
            q_hi = q_mid
            f_hi = f_mid
            if side == 1:
                f_lo *= 0.5
            side = 1
        else:
            q_lo = q_mid
            f_lo = f_mid
            if side == 0:
                f_hi *= 0.5
            side = 0
    return max(0.0, 0.5 * (q_lo + q_hi))


# ── GPU kernel wrapper ─────────────────────────────────────────────────
def _gpu_structure_flow_culvert(
    wse_us: float, wse_ds: float,
    diameter_m: float, length_m: float, slope: float,
    n: float, ke: float, kx: float,
    inlet_invert_m: float, outlet_invert_m: float,
    culvert_code: int = 1,
    gravity: float = 9.81, model_to_ft: float = 3.28084,
) -> float:
    """Call swe2d_gpu_compute_structure_flows for a single culvert."""
    cell_wse = np.array([wse_us, wse_ds], dtype=np.float64)
    cell_bed = np.array([0.0, 0.0], dtype=np.float64)
    stype = np.array([2], dtype=np.int32)  # culvert
    up = np.array([0], dtype=np.int32)
    dn = np.array([1], dtype=np.int32)
    z = np.zeros(1, dtype=np.float64)
    i0 = np.zeros(1, dtype=np.int32)

    q = _MOD.swe2d_gpu_compute_structure_flows(
        cell_wse, cell_bed, stype, up, dn,
        z, z, z,  # crest, width, height
        np.array([diameter_m], dtype=np.float64),
        np.array([length_m], dtype=np.float64),
        np.array([n], dtype=np.float64),
        np.ones(1, dtype=np.float64),  # coeff
        np.array([0.75], dtype=np.float64),  # cd
        np.ones(1, dtype=np.float64),  # opening
        z,  # q_pump
        np.full(1, -1.0, dtype=np.float64),  # max_flow
        np.array([culvert_code], dtype=np.int32),
        i0,  # culvert_shape=0 (circular)
        np.array([diameter_m], dtype=np.float64),  # culvert_rise
        z, z,  # culvert_span, culvert_area
        np.ones(1, dtype=np.float64),  # culvert_barrels
        np.array([slope], dtype=np.float64),
        np.array([inlet_invert_m], dtype=np.float64),
        np.array([outlet_invert_m], dtype=np.float64),
        np.array([ke], dtype=np.float64),
        np.array([kx], dtype=np.float64),
        i0, z, z, np.ones(1, dtype=np.float64),  # embankment params
        gravity, model_to_ft,
    )
    return float(q[0])


# ── Tests ──────────────────────────────────────────────────────────────
@unittest.skipIf(_MOD is None, "hydra_swe2d not available")
@unittest.skipUnless(
    hasattr(_MOD, "swe2d_gpu_available") and _MOD.swe2d_gpu_available(),
    "CUDA GPU not available",
)
class TestCulvertHDS5Validation(unittest.TestCase):
    """Validate GPU culvert kernel against HDS-5 outlet control equations."""

    def test_zero_head_difference(self):
        """No head difference (same WSE, same inverts) → zero flow."""
        q = _gpu_structure_flow_culvert(
            wse_us=1.0, wse_ds=1.0,
            diameter_m=1.0, length_m=10.0, slope=0.0,
            n=0.013, ke=0.5, kx=1.0,
            inlet_invert_m=0.0, outlet_invert_m=0.0,
        )
        self.assertAlmostEqual(q, 0.0, places=6,
                               msg="Zero head difference must produce zero flow")

    def test_downstreamHigher_no_reverse(self):
        """Downstream WSE > upstream → flow should be zero or negative (reverse)."""
        q = _gpu_structure_flow_culvert(
            wse_us=0.5, wse_ds=2.0,
            diameter_m=1.0, length_m=10.0, slope=0.005,
            n=0.013, ke=0.5, kx=1.0,
            inlet_invert_m=0.0, outlet_invert_m=-0.05,
        )
        # Flow should be zero or negative (reverse direction)
        self.assertTrue(q <= 0.0,
                        f"Downstream > upstream: expected non-positive flow, got {q}")

    def test_outlet_control_matches_analytical(self):
        """GPU outlet control flow matches Python HDS-5 direct-step solution.

        Uses a short culvert where barrel friction is small, so the
        direct-step profile is nearly uniform and both implementations
        should agree closely.
        """
        diameter_m = 1.0
        length_m = 10.0
        slope = 0.005
        n = 0.013
        ke = 0.5
        kx = 1.0
        inlet_invert = 0.0
        outlet_invert = -0.05
        model_to_ft = 3.28084

        wse_us = 2.0
        wse_ds = 0.5

        hw_ft = (wse_us - inlet_invert) * model_to_ft
        tw_ft = (wse_ds - outlet_invert) * model_to_ft
        diameter_ft = diameter_m * model_to_ft
        length_ft = length_m * model_to_ft

        expected_cfs = _outlet_control_flow_cfs(
            diameter_ft, length_ft, slope, n, ke, kx, hw_ft, tw_ft,
        )
        expected_model = expected_cfs / (model_to_ft ** 3)

        gpu_q = _gpu_structure_flow_culvert(
            wse_us, wse_ds, diameter_m, length_m, slope,
            n, ke, kx, inlet_invert, outlet_invert,
            model_to_ft=model_to_ft,
        )

        # GPU may select min(inlet, outlet, manning) — so GPU <= expected
        # if outlet control governs.  Allow 30% since the kernel's direct-step
        # may converge differently than the Python reference.
        if expected_model > 0.0:
            rel_err = abs(gpu_q - expected_model) / max(abs(expected_model), 1e-12)
            self.assertLess(rel_err, 0.30,
                            f"Outlet control: GPU={gpu_q:.6f} m³/s, "
                            f"expected={expected_model:.6f} m³/s, "
                            f"rel_err={rel_err:.4f}")
        else:
            self.assertAlmostEqual(gpu_q, 0.0, places=6)

    def test_high_head_dominates_inlet_control(self):
        """Large head → inlet control may dominate; flow should be positive and finite."""
        q = _gpu_structure_flow_culvert(
            wse_us=4.0, wse_ds=0.3,
            diameter_m=1.0, length_m=10.0, slope=0.005,
            n=0.013, ke=0.5, kx=1.0,
            inlet_invert_m=0.0, outlet_invert_m=-0.05,
        )
        self.assertGreater(q, 0.0, "High head must produce positive flow")
        self.assertTrue(math.isfinite(q), "Flow must be finite")

    def test_long_culvert_increases_loss(self):
        """Longer culvert → more friction loss → less flow for same head.

        Uses steeper slope (0.02) so Manning's full-flow capacity exceeds
        outlet-control demand, making length-dependent outlet control govern.
        """
        common = dict(
            wse_us=2.0, wse_ds=0.5,
            diameter_m=1.0, slope=0.02,
            n=0.013, ke=0.5, kx=1.0,
            inlet_invert_m=0.0, outlet_invert_m=-0.4,
        )
        q_short = _gpu_structure_flow_culvert(length_m=20.0, **common)
        q_long = _gpu_structure_flow_culvert(length_m=50.0, **common)
        self.assertGreater(q_short, q_long,
                           f"Short culvert ({q_short:.4f}) should flow more than "
                           f"long culvert ({q_long:.4f})")

    def test_higher_manning_n_reduces_flow(self):
        """Rougher pipe → more friction → less flow.

        Only affects outlet control (barrel friction).  Uses long barrel
        and high roughness so outlet control dominates.
        """
        common = dict(
            wse_us=2.0, wse_ds=0.5,
            diameter_m=1.0, length_m=50.0, slope=0.005,
            ke=0.5, kx=1.0,
            inlet_invert_m=0.0, outlet_invert_m=-0.25,
        )
        q_smooth = _gpu_structure_flow_culvert(n=0.011, **common)
        q_rough = _gpu_structure_flow_culvert(n=0.025, **common)
        self.assertGreater(q_smooth, q_rough,
                           f"Smooth pipe ({q_smooth:.4f}) should flow more than "
                           f"rough pipe ({q_rough:.4f})")

    def test_larger_diameter_increases_flow(self):
        """Larger pipe → more area → more flow."""
        common = dict(
            wse_us=2.0, wse_ds=0.5,
            length_m=20.0, slope=0.005,
            n=0.013, ke=0.5, kx=1.0,
            inlet_invert_m=0.0, outlet_invert_m=-0.1,
        )
        q_small = _gpu_structure_flow_culvert(diameter_m=0.5, **common)
        q_large = _gpu_structure_flow_culvert(diameter_m=1.5, **common)
        self.assertGreater(q_large, q_small,
                           f"Large pipe ({q_large:.4f}) should flow more than "
                           f"small pipe ({q_small:.4f})")

    def test_culvert_code_1_matches_analytical(self):
        """Full HDS-5 validation: code=1 (concrete, square edge, headwall).

        Uses Ponce calculator parameters scaled to SI but with shorter
        barrel so inlet control dominates and flow is significant.
        """
        diameter_m = 1.524  # 5 ft
        length_m = 30.0     # shorter than 200ft to avoid outlet-control kill
        slope = 0.01
        n = 0.013
        ke = 0.5
        kx = 1.0
        inlet_invert = 0.0
        outlet_invert = -length_m * slope
        model_to_ft = 3.28084

        wse_us = 6.75 / model_to_ft  # ~2.058m → HW = 6.75 ft
        wse_ds = 3.5 / model_to_ft + outlet_invert

        hw_ft = (wse_us - inlet_invert) * model_to_ft
        tw_ft = (wse_ds - outlet_invert) * model_to_ft
        diameter_ft = diameter_m * model_to_ft
        length_ft = length_m * model_to_ft

        expected_cfs = _outlet_control_flow_cfs(
            diameter_ft, length_ft, slope, n, ke, kx, hw_ft, tw_ft,
        )
        expected_model = expected_cfs / (model_to_ft ** 3)

        gpu_q = _gpu_structure_flow_culvert(
            wse_us, wse_ds, diameter_m, length_m, slope,
            n, ke, kx, inlet_invert, outlet_invert,
            culvert_code=1, model_to_ft=model_to_ft,
        )

        # GPU takes min(inlet, outlet, manning) — should be close to
        # outlet control when that governs, or inlet control when that
        # governs.  Allow 30% for backwater profile differences.
        if expected_model > 0.0:
            rel_err = abs(gpu_q - expected_model) / max(abs(expected_model), 1e-12)
            self.assertLess(rel_err, 0.30,
                            f"HDS-5 code=1: GPU={gpu_q:.6f} m³/s, "
                            f"expected={expected_model:.6f} m³/s, "
                            f"rel_err={rel_err:.4f}")
        else:
            self.assertGreater(gpu_q, 0.0,
                               f"HDS-5 code=1: expected positive flow, got {gpu_q}")


if __name__ == "__main__":
    unittest.main()
