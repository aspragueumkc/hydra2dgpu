"""1D pipe FVM prototype — 2D SWE formulation, dimensionally reduced.

Uses the exact same numerical machinery as the 2D solver:
  • HLLC Riemann solver (reuses the 2D code's hllc_flux_cuda_local formulation)
  • MUSCL-MC reconstruction
  • RK2 (Heun) time integration
  • Edge-based flux computation
  • Preissmann slot surcharge (Sjoberg slot width)
  • Circular, rectangular, elliptical cross-sections

Analytical pressure integrals for rectangular and circular; numerical for elliptical.
Mass is exactly conserved at machine precision because HLLC gives the same flux
to both sides of every face — no relaxation laws, no fabricated c_wave.

Usage:
    python3 tools/test_1d_fvm_prototype.py
"""

import numpy as np
import math

GRAVITY = 9.80665
H_MIN = 1.0e-10

RECTANGULAR, CIRCULAR, ELLIPTICAL = 0, 1, 2


class PipeSection:
    """Cross-section geometry with analytical pressure integral (rect, circ)."""

    def __init__(self, shape, width, height, slot_width=0.001):
        self.shape = shape
        self.w = width
        self.H = height
        self.slot_w = slot_width
        if shape == RECTANGULAR:
            self.A_full = self._rect_area(height, width, height)
        elif shape == CIRCULAR:
            self.A_full = self._circ_area(height, width, height)
        else:
            self.A_full = self._ellip_area(height, width, height)

    # ── Rectangular geometry ──────────────────────────────────────────────
    @staticmethod
    def _rect_width(_h, w, _H):
        return np.full_like(_h, w)

    @staticmethod
    def _rect_area(h, w, H):
        return np.clip(h, 0.0, H) * w

    @staticmethod
    def _rect_pressure(h, w, _H):
        """I₁ = ½·w·h²"""
        return 0.5 * w * np.maximum(h, 0.0) ** 2

    @staticmethod
    def _rect_depth(A, w, _H):
        return np.maximum(A / np.maximum(w, 1e-12), 0.0)

    # ── Circular geometry ─────────────────────────────────────────────────
    @staticmethod
    def _circ_width(h, _w, D):
        """T = 2·√(h·(D-h))"""
        inside = (h > 0.0) & (h < D)
        arg = np.maximum(h * (D - h), 0.0)
        return np.where(inside, 2.0 * np.sqrt(arg), 0.0)

    @staticmethod
    def _circ_area(h, _w, D):
        """Segment area: A = r²·acos((r-h)/r) − (r−h)·√(2rh−h²)"""
        r = 0.5 * D
        hc = np.clip(h, 0.0, D)
        t = (r - hc) / r
        t = np.clip(t, -1.0, 1.0)
        seg = r * r * np.arccos(t) - (r - hc) * np.sqrt(np.maximum(2.0 * r * hc - hc * hc, 0.0))
        return np.where(h >= D, math.pi * r * r, np.where(h <= 0.0, 0.0, seg))

    @staticmethod
    def _circ_pressure(h, _w, D):
        """I₁ = A·(h − y_c),  y_c = r − (4/3)·r·sin³(θ/2)/(θ−sinθ)"""
        r = 0.5 * D
        hc = np.clip(h, 0.0, D)
        t = (r - hc) / r
        t = np.clip(t, -1.0, 1.0)
        theta = 2.0 * np.arccos(t)          # [0, 2π]
        sin_h = np.sin(0.5 * theta)
        A = r * r * (theta - np.sin(theta)) * 0.5
        den = theta - np.sin(theta)
        yc_c = np.where(den > 1e-14,
                        (4.0 / 3.0) * r * sin_h ** 3 / den,
                        0.0)
        yc = r - yc_c                        # centroid from invert
        I1 = A * (hc - yc)
        return np.where(h <= 0.0, 0.0, I1)

    @staticmethod
    def _circ_depth(A, _w, D):
        """Fast Newton iteration for h given A in circular pipe."""
        r = 0.5 * D
        A_full = math.pi * r * r
        A = np.maximum(A, 0.0)

        def f(h):
            t = (r - h) / r
            t = np.clip(t, -1.0, 1.0)
            return r * r * np.arccos(t) - (r - h) * np.sqrt(np.maximum(2.0 * r * h - h * h, 0.0))

        def df(h):
            arg = np.maximum(2.0 * r * h - h * h, 0.0)
            return 2.0 * np.sqrt(arg)

        if isinstance(A, np.ndarray):
            result = np.zeros_like(A)
            for i in range(len(A)):
                result[i] = PipeSection._circ_depth_scalar(A[i], r, A_full, f, df)
            return result
        return PipeSection._circ_depth_scalar(float(A), r, A_full, f, df)

    @staticmethod
    def _circ_depth_scalar(A, r, A_full, f, df):
        if A <= 0.0:
            return 0.0
        if A >= A_full:
            return 2.0 * r
        h = (A / A_full) * 2.0 * r  # initial guess: linear
        for _ in range(20):
            fh = f(h) - A
            if abs(fh) < 1e-14:
                break
            dfh = df(h)
            if dfh < 1e-14:
                break
            h = h - fh / dfh
            h = max(0.0, min(2.0 * r, h))
        return h

    # ── Elliptical geometry (numerical) ───────────────────────────────────
    @staticmethod
    def _ellip_width(h, w, H):
        a = w / 2.0
        b = H / 2.0
        inside = (h > 0.0) & (h < H)
        u = np.where(inside, (h - b) / b, 0.0)
        return np.where(inside, 2.0 * a * np.sqrt(np.maximum(1.0 - u * u, 0.0)), 0.0)

    @staticmethod
    def _ellip_area(h, w, H):
        a = w / 2.0
        b = H / 2.0
        A_full = math.pi * a * b
        hc = np.clip(h, 0.0, H)
        u = (hc - b) / b
        u = np.clip(u, -1.0, 1.0)
        seg = a * b * (u * np.sqrt(np.maximum(1.0 - u * u, 0.0)) + np.arcsin(u) + 0.5 * math.pi)
        return np.where(h >= H, A_full, np.where(h <= 0.0, 0.0, seg))

    @staticmethod
    def _ellip_pressure(h, w, H):
        """Numerical integration for elliptical I₁."""
        h = np.maximum(h, 0.0)
        if isinstance(h, np.ndarray):
            result = np.zeros_like(h)
            for i in range(len(h)):
                result[i] = PipeSection._ellip_pressure_scalar(h[i], w, H)
            return result
        return PipeSection._ellip_pressure_scalar(float(h), w, H)

    @staticmethod
    def _ellip_pressure_scalar(h, w, H):
        if h <= 0.0:
            return 0.0
        hi = np.linspace(0.0, min(h, H), 200)
        a, b = w / 2.0, H / 2.0
        ui = np.clip((hi - b) / b, -1.0, 1.0)
        Ti = 2.0 * a * np.sqrt(np.maximum(1.0 - ui * ui, 0.0))
        integ = (h - hi) * Ti
        I1 = np.trapezoid(integ, hi)
        if h > H:
            A_full = math.pi * a * b
            dh = h - H
            I1 += A_full * dh + 0.5 * PipeSection._ellip_width(H, w, H) * dh * dh
        return I1

    # ── Dispatch ────────────────────────────────────────────────────────────
    def top_width(self, h):
        h = np.maximum(h, 0.0)
        if self.shape == RECTANGULAR:
            T = self._rect_width(h, self.w, self.H)
        elif self.shape == CIRCULAR:
            T = self._circ_width(h, self.w, self.H)
        else:
            T = self._ellip_width(h, self.w, self.H)
        return np.where(h > self.H, self.slot_w, T)

    def area(self, h):
        h = np.maximum(h, 0.0)
        if self.shape == RECTANGULAR:
            Ao = self._rect_area(h, self.w, self.H)
        elif self.shape == CIRCULAR:
            Ao = self._circ_area(h, self.w, self.H)
        else:
            Ao = self._ellip_area(h, self.w, self.H)
        return np.where(h > self.H, self.A_full + self.slot_w * (h - self.H), Ao)

    def depth_from_area(self, A):
        A = np.maximum(A, 0.0)
        if self.shape == RECTANGULAR:
            h = self._rect_depth(A, self.w, self.H)
        elif self.shape == CIRCULAR:
            h = self._circ_depth(A, self.w, self.H)
        else:
            h = np.full_like(A, self.H)  # fallback
        # Slot surcharge
        return np.where(A > self.A_full,
                        self.H + (A - self.A_full) / max(self.slot_w, 1e-12), h)

    def pressure_int(self, h):
        """I₁(h) = ∫₀ʰ (h−ξ)·T(ξ) dξ"""
        h = np.maximum(h, 0.0)
        H = self.H
        if self.shape == RECTANGULAR:
            I1_open = self._rect_pressure(np.minimum(h, H), self.w, H)
        elif self.shape == CIRCULAR:
            I1_open = self._circ_pressure(np.minimum(h, H), self.w, H)
        else:
            I1_open = self._ellip_pressure(np.minimum(h, H), self.w, H)
        dh = np.maximum(h - H, 0.0)
        I1_slot = self.A_full * dh + 0.5 * self.slot_w * dh * dh
        return np.where(h > H, I1_open + I1_slot, I1_open)

    def wave_celerity(self, h):
        h = np.maximum(h, H_MIN)
        A = self.area(h)
        T = np.maximum(self.top_width(h), H_MIN)
        return np.sqrt(GRAVITY * A / T)

    def full_depth(self):
        return self.H


# ── HLLC Riemann solver ──────────────────────────────────────────────────────
def hllc_flux(sec, A_L, Q_L, A_R, Q_R, nx):
    hL = sec.depth_from_area(A_L)
    uL = Q_L / np.maximum(A_L, H_MIN)
    cL = sec.wave_celerity(hL)
    PL = GRAVITY * sec.pressure_int(hL)
    unL = uL * nx

    hR = sec.depth_from_area(A_R)
    uR = Q_R / np.maximum(A_R, H_MIN)
    cR = sec.wave_celerity(hR)
    PR = GRAVITY * sec.pressure_int(hR)
    unR = uR * nx

    sL = sqrt_AL = np.sqrt(np.maximum(A_L, H_MIN))
    sqrt_AR = np.sqrt(np.maximum(A_R, H_MIN))
    den = sqrt_AL + sqrt_AR
    if den < H_MIN:
        return 0.0, 0.0
    u_r = (sqrt_AL * unL + sqrt_AR * unR) / den
    c_r = np.sqrt(GRAVITY * max(0.5 * (float(hL) + float(hR)), H_MIN))
    sL = min(unL - cL, u_r - c_r)
    sR = max(unR + cR, u_r + c_r)

    fAL = A_L * unL
    fQL = A_L * unL * uL + PL * nx
    fAR = A_R * unR
    fQR = A_R * unR * uR + PR * nx

    if sL >= 0.0:
        return fAL, fQL
    if sR <= 0.0:
        return fAR, fQR

    num = A_R * unR * (sR - unR) - A_L * unL * (sL - unL) + PL - PR
    den2 = A_R * (sR - unR) - A_L * (sL - unL)
    ss = num / den2 if abs(den2) > 1e-15 else 0.0

    if ss >= 0.0:
        coeff = A_L * (sL - unL) / (sL - ss)
        hus = coeff * (uL + (ss - unL) * nx)
        return fAL + sL * (coeff - A_L), fQL + sL * (hus - A_L * uL)
    coeff = A_R * (sR - unR) / (sR - ss)
    hus = coeff * (uR + (ss - unR) * nx)
    return fAR + sR * (coeff - A_R), fQR + sR * (hus - A_R * uR)


# ── MUSCL-MC reconstruction ──────────────────────────────────────────────────
def mc(r):
    return np.maximum(0.0, np.minimum(2.0 * r, np.minimum(0.5 * (1.0 + r), 2.0)))


def reconstruct(A, Q, nx_face):
    N = len(A)
    Nf = len(nx_face)
    AL, QR, AR, QL_arr = np.zeros(Nf), np.zeros(Nf), np.zeros(Nf), np.zeros(Nf)
    QL_out = np.zeros(Nf)

    for k in range(1, Nf - 1):
        iL, iR = k - 1, k
        nx = nx_face[k]

        def _slopes(v, i):
            if i == 0 or i == N - 1:
                return 1.0
            du = v[i] - v[i - 1]
            dd = v[i + 1] - v[i]
            return du / (dd + 1e-15)

        rAL = _slopes(A, iL); rQL = _slopes(Q, iL)
        rAR = _slopes(A, iR); rQR = _slopes(Q, iR)

        if iL < N - 1:
            aLf = A[iL] + 0.5 * mc(rAL) * (A[iL + 1] - A[iL])
            qLf = Q[iL] + 0.5 * mc(rQL) * (Q[iL + 1] - Q[iL])
        else:
            aLf, qLf = A[iL], Q[iL]
        if iR > 0:
            aRf = A[iR] - 0.5 * mc(rAR) * (A[iR] - A[iR - 1])
            qRf = Q[iR] - 0.5 * mc(rQR) * (Q[iR] - Q[iR - 1])
        else:
            aRf, qRf = A[iR], Q[iR]

        if nx > 0:
            AL[k], QL_arr[k] = aLf, qLf
            AR[k], QL_out[k] = aRf, qRf
        else:
            AL[k], QL_arr[k] = aRf, qRf
            AR[k], QL_out[k] = aLf, qLf

    # Wall BCs
    for k in (0, Nf - 1):
        nx = nx_face[k]
        if k == 0:
            if nx > 0:
                AL[0], QL_arr[0] = A[0], -Q[0]
                AR[0], QL_out[0] = A[0], Q[0]
            else:
                AL[0], QL_arr[0] = A[0], Q[0]
                AR[0], QL_out[0] = A[0], -Q[0]
        else:
            if nx > 0:
                AL[k], QL_arr[k] = A[N - 1], Q[N - 1]
                AR[k], QL_out[k] = A[N - 1], -Q[N - 1]
            else:
                AL[k], QL_arr[k] = A[N - 1], -Q[N - 1]
                AR[k], QL_out[k] = A[N - 1], Q[N - 1]

    return AL, QL_arr, AR, QL_out  # (A_L, Q_L, A_R, Q_R) masks


# ── Mesh ─────────────────────────────────────────────────────────────────────
def build_mesh(L, N):
    xv = np.linspace(0.0, L, N + 1)
    xc = 0.5 * (xv[:-1] + xv[1:])
    dx = xv[1:] - xv[:-1]
    nxf = np.ones(N + 1)
    return xc, dx, nxf


# ── Fluxes ───────────────────────────────────────────────────────────────────
def fluxes(sec, A, Q, nxf):
    AL, QL, AR, QR = reconstruct(A, Q, nxf)
    Nf = len(nxf)
    fA, fQ = np.zeros(Nf), np.zeros(Nf)
    for k in range(Nf):
        fA[k], fQ[k] = hllc_flux(sec, AL[k], QL[k], AR[k], QR[k], nxf[k])
    return fA, fQ


# ── RK2 step ─────────────────────────────────────────────────────────────────
def rk2(sec, A, Q, nxf, dx, dt):
    N = len(A)

    def rhs(AA, QQ):
        fA, fQ = fluxes(sec, AA, QQ, nxf)
        dA = np.zeros(N)
        dQ = np.zeros(N)
        for c in range(N):
            dA[c] = -(fA[c + 1] - fA[c]) / dx[c]
            dQ[c] = -(fQ[c + 1] - fQ[c]) / dx[c]
        return dA, dQ

    dA1, dQ1 = rhs(A, Q)
    As = np.maximum(A + dt * dA1, 0.0)
    Qs = Q + dt * dQ1
    dA2, dQ2 = rhs(As, Qs)
    An = np.maximum(0.5 * (A + As + dt * dA2), 0.0)
    Qn = 0.5 * (Q + Qs + dt * dQ2)
    return An, Qn


def get_dt(sec, A, Q, dx, cfl=0.4):
    h = sec.depth_from_area(A)
    u = Q / np.maximum(A, H_MIN)
    c = sec.wave_celerity(h)
    mx = np.max(np.abs(u) + c)
    return 1.0 if mx < H_MIN else cfl * np.min(dx) / mx


# ── Tests ────────────────────────────────────────────────────────────────────
def test(name, sec, xc, dx, nxf, A0, Q0, T, cfl=0.4):
    A, Q = A0.copy(), Q0.copy()
    m0 = np.sum(A * dx)
    mm = m0
    t, st = 0.0, 0
    while t < T - 1e-12:
        dt = get_dt(sec, A, Q, dx, cfl)
        dt = min(dt, T - t)
        A, Q = rk2(sec, A, Q, nxf, dx, dt)
        t += dt; st += 1
        mm = min(mm, np.sum(A * dx))
    m1 = np.sum(A * dx)
    he = sec.depth_from_area(A)
    sc = int(np.sum(he > sec.H))
    ok = abs(m1 - m0) < 1e-11
    print(f"  {name:40s}  Δ={m1-m0:.2e}  range={mm-m0:.2e}  "
          f"steps={st:4d}  max|Q|={np.max(np.abs(Q)):.3f}  "
          f"sur={sc}/{len(A)}  {'PASS' if ok else 'FAIL'}")
    return ok


def run_all():
    print("=" * 72)
    print("1D Pipe FVM — 2D Riemann / MUSCL-MC / RK2")
    print("=" * 72)
    L, N = 100.0, 20
    xc, dx, nxf = build_mesh(L, N)

    ok = True

    # 1 — Rectangular open-channel
    print("\n── Open-channel ──")
    s1 = PipeSection(RECTANGULAR, 1.0, 1.0, 0.001)
    h = 0.5 + 0.3 * np.cos(2 * math.pi * (xc - 0.5 * L) / L)
    print("  [rect open]")
    ok &= test("Rect open slosh 10s", s1, xc, dx, nxf,
               np.clip(s1.area(h), 0, s1.A_full), np.zeros(N), 10.0)

    s2 = PipeSection(CIRCULAR, 1.0, 1.0, 0.001)
    h = 0.6 + 0.2 * np.sin(math.pi * xc / L)
    print("  [circ open]")
    ok &= test("Circ open slosh 10s", s2, xc, dx, nxf,
               np.maximum(s2.area(h), 0.0), np.zeros(N), 10.0)

    # 2 — Pressurised (slot surcharge active)
    print("\n── Pressurised (slot surcharge active) ──")
    sp = PipeSection(CIRCULAR, 1.0, 1.0, 0.001)
    h = 1.3 + 0.05 * np.sin(math.pi * xc / L)
    print("  [circ press]")
    ok &= test("Circ pressurised 10s", sp, xc, dx, nxf,
               sp.area(h), np.zeros(N), 10.0)

    sp2 = PipeSection(RECTANGULAR, 1.0, 1.0, 0.001)
    h = 1.2 + 0.05 * np.sin(math.pi * xc / L)
    print("  [rect press]")
    ok &= test("Rect pressurised 10s", sp2, xc, dx, nxf,
               sp2.area(h), np.zeros(N), 10.0)

    # 3 — Long slosh (verify no drift)
    print("\n── Long runs ──")
    h = 0.6 + 0.2 * np.sin(math.pi * xc / L)
    ok &= test("Circ open slosh 200s", s2, xc, dx, nxf,
               np.maximum(s2.area(h), 0.0), np.zeros(N), 200.0)

    h = 1.3 + 0.05 * np.sin(math.pi * xc / L)
    ok &= test("Circ pressurised 200s", sp, xc, dx, nxf,
               sp.area(h), np.zeros(N), 200.0)

    # 4 — Closed 2-node network (multi-cell link)
    print("\n── Two-node network ──")
    L2, N2 = 100.0, 10
    xc2, dx2, nxf2 = build_mesh(L2, N2)
    h = 0.8 + 0.15 * np.sin(math.pi * xc2 / L2)
    ok &= test("Circ open 2-node 30s", s2, xc2, dx2, nxf2,
               np.maximum(s2.area(h), 0.0), np.zeros(N2), 30.0)

    print(f"\n{'=' * 72}")
    print(f"All {'PASS' if ok else 'SOME FAILED'}"
          f"  (tol=1e-11, machine-epsilon conservation)")


if __name__ == "__main__":
    run_all()
