import math, sys
sys.path.insert(0, '.')
import backwater2 as bw
from culvert_routine import (
    CircularXsect,
    critical_depth_in_culvert, solve_normal_depth_in_culvert,
    direct_step_culvert_upstream_energy, solve_headwater_depth_for_Q,
)
G = bw.G
m = bw.load_input('example_project/example.gpkg')
res = bw.run_backwater(m, solver='py')

xs89 = None
for xs in m.sections:
    try:
        rs = float(xs.river_station)
    except (TypeError, ValueError):
        continue
    if abs(rs - 89.3) < 1.0:
        xs89 = xs
        break

if xs89 is None:
    print('RS 89.3 not found; stations:', [xs.river_station for xs in m.sections])
    sys.exit(1)

D = xs89.culvert_diameter
xsect = CircularXsect(diameter_ft=D, culvert_code=xs89.culvert_code)
slope = xs89.culvert_slope()
n = xs89.n_ch
L = xs89.culvert_length
y_full = xs89.culvert_full_depth()
A_full = math.pi * (D/2)**2
P_full = math.pi * D
R_full = A_full / P_full
Ke = 0.5
Kf = (2.0 * G * n**2 * L) / (1.486**2 * R_full**(4.0/3.0))
z_inlet  = xs89.culvert_upstream_invert
z_outlet = xs89.culvert_downstream_invert
Q = 200.0

# Compute z_crown from geometry min (same as solver)
z_crown = z_inlet + y_full
if xs89.geometry and len(xs89.geometry) >= 2:
    try:
        z_geom_min = min(z for x, z in xs89.geometry)
        if z_geom_min > z_inlet:
            z_crown = z_geom_min
    except Exception:
        pass

dc = critical_depth_in_culvert(xsect, Q)
yn = solve_normal_depth_in_culvert(xsect, Q, n, slope)

# res[0] = downstream BC section; res[1] = culvert XS headwater
tw_wse = res[0].wse
hw_reported = res[1].wse

print(f'=== RS {xs89.river_station} culvert diagnostics ===')
print(f'  shape={xs89.culvert_shape}, D={D}, L={L}, slope={slope:.6f}, n={n}')
print(f'  z_inlet={z_inlet:.3f}, z_outlet={z_outlet:.3f}, z_crown_inlet={z_inlet+y_full:.3f}')
print(f'  y_full={y_full:.4f} ft')
print(f'  dc (critical depth) = {dc:.4f} ft   => WSE at outlet crown = {z_outlet+dc:.3f} ft')
print(f'  yn (normal depth)   = {yn:.4f} ft   => WSE at outlet crown = {z_outlet+yn:.3f} ft')
print(f'  Kf={Kf:.4f}')
print(f'  TW WSE fed in       = {tw_wse:.4f} ft')
print(f'  HW WSE reported     = {hw_reported:.4f} ft')
print()

tw_depth_raw = max(0.0, tw_wse - z_outlet)
print(f'  tw_depth_raw (TW-z_outlet)   = {tw_depth_raw:.4f} ft')

outlet_start = max(min(max(tw_depth_raw, dc), y_full), dc)
print(f'  outlet_start for direct_step = {outlet_start:.4f} ft')

energy_up, depth_up, pmode = direct_step_culvert_upstream_energy(
    xsect=xsect, Q=Q, n_value=n, slope=slope, length=L, tailwater_depth=outlet_start,
)
area_up = xsect.area(min(max(depth_up, 1e-6), y_full))
v_up = Q / max(area_up, 1e-6)
hw_oc_partial = z_inlet + energy_up + Ke * v_up**2 / (2*G)

print(f'  direct_step profile_mode       = {pmode}')
print(f'  depth_up (inside barrel @inlet)= {depth_up:.4f} ft  => WSE inside={z_inlet+depth_up:.3f} ft')
print(f'  energy_up (specific energy)    = {energy_up:.4f} ft')
print(f'  HW_OC (z_inlet+E+Ke*Vhd)      = {hw_oc_partial:.4f} ft WSE')
print()

h_hw_ic, _, _, _ = solve_headwater_depth_for_Q(xsect, slope=slope, Q_target=Q, h_min=0.0, h_max=max(10*y_full, 1.0))
hw_ic = z_inlet + h_hw_ic
print(f'  HW_IC (inlet control)          = {hw_ic:.4f} ft WSE')
governing = "IC" if hw_ic >= hw_oc_partial else "OC"
print(f'  governing                      = {governing}')

# ---------------------------------------------------------------
# Trace the bisection: what Q_barrel does the solver converge to?
# Simulate the bisection to find Q_weir and Q_barrel at convergence
# ---------------------------------------------------------------
print()
print('--- Bisection trace (solving for self-consistent HW) ---')

import backwater2 as bw2
xs89_full = xs89

def governing_hw_at_q(q_cul):
    """Simplified governing HW at a given barrel Q (outlet control, full-pipe)."""
    dc_q = critical_depth_in_culvert(xsect, q_cul)
    yn_q = solve_normal_depth_in_culvert(xsect, q_cul, n, slope)
    dc_q = min(dc_q, y_full)
    yn_q = min(yn_q, y_full)
    outlet_start = max(dc_q, 0.0)
    try:
        eu, du, pm = direct_step_culvert_upstream_energy(
            xsect=xsect, Q=q_cul, n_value=n, slope=slope, length=L, tailwater_depth=outlet_start,
        )
        au = xsect.area(min(max(du, 1e-6), y_full))
        vu = q_cul / max(au, 1e-6)
        hw_oc = z_inlet + eu + Ke * vu**2 / (2*G)
    except Exception:
        v_full_q = q_cul / (math.pi * (D/2)**2)
        Kf_q = (2*G*n**2*L) / (1.486**2 * R_full**(4/3))
        hw_oc = z_inlet + y_full + (1 + Ke + Kf_q) * v_full_q**2 / (2*G)
        pm = 'fallback'
    h_ic, _, _, _ = solve_headwater_depth_for_Q(xsect, slope=slope, Q_target=q_cul, h_min=0.0, h_max=max(10*y_full,1.0))
    hw_ic_q = z_inlet + h_ic
    hw_gov = max(hw_ic_q, hw_oc)
    return hw_gov, hw_ic_q, hw_oc, yn_q, dc_q, pm

# Evaluate at HEC-RAS Q_barrel = 147.05
hw_g, hw_i, hw_o, yn_147, dc_147, pm_147 = governing_hw_at_q(147.05)
print(f'  At Q_barrel=147.05 cfs (HEC-RAS):')
print(f'    dc={dc_147:.4f} ft, yn={yn_147:.4f} ft')
print(f'    HW_IC={hw_i:.4f} ft, HW_OC={hw_o:.4f} ft, governing={hw_g:.4f} ft, mode={pm_147}')

# Evaluate at Q_total = 200 (no weir)
hw_g200, hw_i200, hw_o200, yn_200, dc_200, pm_200 = governing_hw_at_q(200.0)
print(f'  At Q_barrel=200 cfs (no weir):')
print(f'    dc={dc_200:.4f} ft, yn={yn_200:.4f} ft')
print(f'    HW_IC={hw_i200:.4f} ft, HW_OC={hw_o200:.4f} ft, governing={hw_g200:.4f} ft, mode={pm_200}')

print()
print('--- Weir flow trace ---')
print(f'  z_crown_inlet = {z_crown:.3f} ft  (min of embankment geometry)')
print(f'  weir_coeff (Cw) = {xs89.culvert_weir_coeff}')
print(f'  weir_sta_left   = {xs89.culvert_weir_sta_left}')
print(f'  weir_sta_right  = {xs89.culvert_weir_sta_right}')

# Compute Q_weir at several HW values
from backwater2 import irregular_weir_flow_from_geometry
for hw_test in [800.0, 801.0, 803.169, 803.17, 803.5, 804.0, 804.95, 805.0]:
    qw = irregular_weir_flow_from_geometry(
        xs_culvert=xs89,
        headwater_wse=hw_test,
        z_crown_inlet=z_crown,
        Cw=float(xs89.culvert_weir_coeff or 3.0),
        sta_left=float(xs89.culvert_weir_sta_left or 0.0),
        sta_right=float(xs89.culvert_weir_sta_right or 0.0),
    )
    print(f'  HW={hw_test:.3f}: Q_weir={qw:.2f} cfs, Q_barrel={200-qw:.2f} cfs, head_over_weir={hw_test-z_crown:.3f} ft')
q_b = 147.05
v_b = q_b / (math.pi * (D/2)**2)
vh_b = v_b**2 / (2*G)
Kf_full = (2*G*n**2*L) / (1.486**2 * R_full**(4/3))
Sf_full = (q_b * n / (1.486 * math.pi*(D/2)**2 * R_full**(2/3)))**2
fric = Sf_full * L
print(f'  V_full = {v_b:.3f} ft/s, V²/2g = {vh_b:.4f} ft')
print(f'  Sf_full = {Sf_full:.6f} ft/ft, friction = {fric:.4f} ft')
print(f'  Ke={Ke}, entrance loss = {Ke*vh_b:.4f} ft')
print(f'  EG at outlet face (inside, full) = z_outlet+y_full+V²/2g = {z_outlet+y_full+vh_b:.4f} ft')
print(f'  EG at inlet face (inside, full)  = EG_out + friction_loss  = {z_outlet+y_full+vh_b + fric:.4f} ft')
print(f'  EG_US = EG_inlet_inside + entrance_loss = {z_outlet+y_full+vh_b + fric + Ke*vh_b:.4f} ft')
print(f'  Expected HEC-RAS EG_US = 805.00 ft')
