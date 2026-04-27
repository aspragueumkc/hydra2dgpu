import os, csv
import backwater_model as bw

root = os.path.join(os.getcwd(), 'hec_ras_project')
# read sections metadata
rows = []
with open(os.path.join(root, 'sections_metadata.csv'), 'r', newline='') as f:
    rdr = csv.DictReader(f)
    for r in rdr:
        rows.append(r)

# build sections
sections = []
for r in rows:
    rs = r['river_station']
    xs_file = r['file']
    lb = float(r['left_bank_station'])
    rb = float(r['right_bank_station'])
    n_lob = float(r['n_lob'])
    n_ch = float(r['n_ch'])
    n_rob = float(r['n_rob'])
    L_lob = float(r['L_lob_to_next'])
    L_ch = float(r['L_ch_to_next'])
    L_rob = float(r['L_rob_to_next'])
    geom = []
    with open(os.path.join(root, xs_file), 'r', newline='') as g:
        rdr2 = csv.DictReader(g)
        for row in rdr2:
            try:
                off = float(row['Offset'])
                el = float(row['Elevation'])
            except Exception:
                continue
            geom.append((off, el))
    xs = bw.CrossSection(river_station=rs, geometry=geom, left_bank_station=lb, right_bank_station=rb,
                         n_lob=n_lob, n_ch=n_ch, n_rob=n_rob,
                         L_lob_to_next=L_lob, L_ch_to_next=L_ch, L_rob_to_next=L_rob)
    sections.append(xs)

# build baseline results to get s_dn for problematic link (between index 3 and 4)
Q = 500.0
S0 = 0.003
model = bw.ModelInput(flow_cfs=Q, flow_change=None, boundary_condition='normal_depth', boundary_value=S0, sections=sections)
results = bw.run_backwater(model)

# Problematic link: 3 -> 4 (sections index 3 downstream, 4 upstream)
i = 3
xs_dn = sections[i]
xs_up = sections[i+1]
s_dn = results[i]
# We'll run the solver's secant iterations manually and print diagnostics
link = bw.ReachLink(xs_dn.L_lob_to_next, xs_dn.L_ch_to_next, xs_dn.L_rob_to_next)

z_dn = min(z for _, z in xs_dn.geometry)
z_up = min(z for _, z in xs_up.geometry)
Q_dn = Q
Q_up = Q

# initial guesses similar to solve_energy_upstream
w1 = max(z_up + 0.5, s_dn.wse)
s1 = bw.compute_state(xs_up, w1, Q_up)

def energy_balance_state(s_up_state):
    y_dn = s_dn.wse - z_dn
    y_up = s_up_state.wse - z_up
    loss = bw.head_loss(s_dn, s_up_state, link, xs_dn)
    lhs = z_dn + y_dn + (s_dn.alpha * s_dn.V_t ** 2) / (2.0 * bw.G)
    rhs = z_up + y_up + (s_up_state.alpha * s_up_state.V_t ** 2) / (2.0 * bw.G) + loss
    return lhs - rhs, loss

f1, loss1 = energy_balance_state(s1)
print(f'iter 1: w={w1:.6f}, A_up={s1.A_t:.6f}, V_up={s1.V_t:.6f}, alpha_up={s1.alpha:.6f}, loss={loss1:.6f}, f={f1:.6e}')

w2 = w1 + 0.5
s2 = bw.compute_state(xs_up, w2, Q_up)
f2, loss2 = energy_balance_state(s2)
print(f'iter 2: w={w2:.6f}, A_up={s2.A_t:.6f}, V_up={s2.V_t:.6f}, alpha_up={s2.alpha:.6f}, loss={loss2:.6f}, f={f2:.6e}')

for k in range(1,20):
    if abs(f2 - f1) < 1e-12:
        w3 = 0.5 * (w1 + w2)
    else:
        w3 = w2 - f2 * (w2 - w1) / (f2 - f1)
    zmin_up = min(z for _, z in xs_up.geometry)
    w3 = max(w3, zmin_up + 1e-3)
    s3 = bw.compute_state(xs_up, w3, Q_up)
    f3, loss3 = energy_balance_state(s3)
    print(f'iter {k+2}: w={w3:.6f}, A_up={s3.A_t:.6f}, V_up={s3.V_t:.6f}, alpha_up={s3.alpha:.6f}, loss={loss3:.6f}, f={f3:.6e}')
    if abs(f3) < 1e-6:
        break
    w1, f1 = w2, f2
    w2, f2 = w3, f3

print('\nFinal s_dn: A={:.6f}, V={:.6f}, alpha={:.6f}, wse={:.6f}'.format(s_dn.A_t, s_dn.V_t, s_dn.alpha, s_dn.wse))
print('Final s_up: A={:.6f}, V={:.6f}, alpha={:.6f}, wse={:.6f}'.format(s3.A_t, s3.V_t, s3.alpha, s3.wse))
