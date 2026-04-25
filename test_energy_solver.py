import math
import sys
import os

# ensure plugin directory on path
here = os.path.dirname(__file__)
if here not in sys.path:
    sys.path.insert(0, here)

import backwater2 as bw

G = bw.G

# Simple rectangular channel cross-section (flat bed)
geom = [(0.0, 0.0), (10.0, 0.0)]  # stations 0 to 10, bed elevation 0

# downstream section (index 0) with reach length to next = 100 ft
xs_dn = bw.CrossSection(
    river_station='XS_dn', geometry=geom,
    left_bank_station=0.0, right_bank_station=10.0,
    n_lob=0.03, n_ch=0.03, n_rob=0.03,
    L_lob_to_next=0.0, L_ch_to_next=100.0, L_rob_to_next=0.0
)
# upstream section (index 1), identical geometry
xs_up = bw.CrossSection(
    river_station='XS_up', geometry=geom,
    left_bank_station=0.0, right_bank_station=10.0,
    n_lob=0.03, n_ch=0.03, n_rob=0.03
)

Q = 100.0  # cfs
S0 = 0.001  # bed slope (for normal depth)

# Compute downstream WSE as normal depth (uniform flow) for a baseline
wse_dn = bw.solve_normal_depth(xs_dn, Q, S0)
print(f"wse_dn (normal depth estimate) = {wse_dn:.6f}")

z_dn = min(z for _, z in xs_dn.geometry)
z_up = min(z for _, z in xs_up.geometry)

s_dn = bw.compute_state(xs_dn, wse_dn, Q)
print(f"Downstream: A={s_dn.A_t:.6f}, V={s_dn.V_t:.6f}, K={s_dn.K_t:.6f}, Sf={s_dn.Sf_total:.6e}")

# Use the module solver
link = bw.ReachLink(xs_dn.L_lob_to_next, xs_dn.L_ch_to_next, xs_dn.L_rob_to_next)

s_up_module = bw.solve_energy_upstream(
    xs_dn=xs_dn, xs_up=xs_up,
    z_dn=z_dn, z_up=z_up,
    Q_total_dn=Q, Q_total_up=Q,
    s_dn=s_dn,
    link=link,
    wse_up_init=s_dn.wse
)

print(f"Module upstream WSE = {s_up_module.wse:.6f}")

# Independent bisection using same physics functions but different numeric method
tol = 1e-4
def energy_balance_for_wse(wse_up):
    s_up = bw.compute_state(xs_up, wse_up, Q)
    y_dn = s_dn.wse - z_dn
    y_up = s_up.wse - z_up
    loss = bw.head_loss(s_dn, s_up, link, xs_dn)
    lhs = z_dn + y_dn + (s_dn.alpha * s_dn.V_t ** 2) / (2.0 * G)
    rhs = z_up + y_up + (s_up.alpha * s_up.V_t ** 2) / (2.0 * G) + loss
    return lhs - rhs

# Check residual at module solution (primary validation)
f_module = energy_balance_for_wse(s_up_module.wse)
print(f"Energy balance at module solution (lhs-rhs) = {f_module:.6e}")

# Optionally try a safer bracket-searched bisection for independent confirmation
low = z_up + 1e-3
high = s_dn.wse + 50.0
f_low = energy_balance_for_wse(low)
found = False
N = 200
for i in range(1, N + 1):
    w = low + (high - low) * (i / N)
    fw = energy_balance_for_wse(w)
    if fw == 0 or (fw * f_low < 0):
        w_lo = low if fw * f_low < 0 else (w - (high - low) / N)
        w_hi = w
        found = True
        break

if found:
    # bisection
    f_lo = energy_balance_for_wse(w_lo)
    f_hi = energy_balance_for_wse(w_hi)
    for _ in range(80):
        w_mid = 0.5 * (w_lo + w_hi)
        fmid = energy_balance_for_wse(w_mid)
        if abs(fmid) < 1e-6:
            break
        if fmid * f_lo > 0:
            w_lo, f_lo = w_mid, fmid
        else:
            w_hi, f_hi = w_mid, fmid
    s_up_bisect_wse = w_mid
    print(f"Bisection upstream WSE = {s_up_bisect_wse:.6f}")
    diff_module_vs_bisect = abs(s_up_module.wse - s_up_bisect_wse)
    print(f"Difference module vs bisection = {diff_module_vs_bisect:.6e}")
    print(f"Upstream (module): A={s_up_module.A_t:.6f}, V={s_up_module.V_t:.6f}, Sf={s_up_module.Sf_total:.6e}")
    s_up_bisect = bw.compute_state(xs_up, s_up_bisect_wse, Q)
    print(f"Upstream (bisect): A={s_up_bisect.A_t:.6f}, V={s_up_bisect.V_t:.6f}, Sf={s_up_bisect.Sf_total:.6e}")
else:
    print('No sign change found for independent bisection; skipping.')

tol = 1e-4
if abs(f_module) < tol:
    print('PASS: module solver satisfies the energy equation (residual small)')
else:
    print('FAIL: module solver residual is large')
