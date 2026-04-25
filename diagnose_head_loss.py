import os, csv
import backwater2 as bw
import h5py

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

# run model with defaults
Q = 500.0
S0 = 0.003
model = bw.ModelInput(flow_cfs=Q, flow_change=None, boundary_condition='normal_depth', boundary_value=S0, sections=sections)
results = bw.run_backwater(model)

print('Link diagnostics (downstream index -> upstream index):')
for i in range(len(sections)-1):
    xs_dn = sections[i]
    xs_up = sections[i+1]
    s_dn = results[i]
    s_up = results[i+1]
    link = bw.ReachLink(xs_dn.L_lob_to_next, xs_dn.L_ch_to_next, xs_dn.L_rob_to_next)
    Qlob_av = 0.5 * (s_dn.Q_lob + s_up.Q_lob)
    Qch_av  = 0.5 * (s_dn.Q_ch  + s_up.Q_ch)
    Qrob_av = 0.5 * (s_dn.Q_rob + s_up.Q_rob)
    Ldw = link.discharge_weighted_length(Qlob_av, Qch_av, Qrob_av)
    Sf = bw.representative_friction_slope_total(s_dn, s_up)
    hf = Sf * Ldw
    C = bw.minor_loss_coeff(s_dn, s_up, xs_dn)
    hv = C * ((s_up.alpha * s_up.V_t ** 2) - (s_dn.alpha * s_dn.V_t ** 2)) / (2.0 * bw.G)
    total = bw.head_loss(s_dn, s_up, link, xs_dn)
    print(f'Link {i}->{i+1}: Ldw={Ldw:.6f}, Sf={Sf:.6e}, hf={hf:.6e}, C={C:.6f}, hv={hv:.6e}, total={total:.6e}')
    print(f'  s_dn: A={s_dn.A_t:.6f}, V={s_dn.V_t:.6f}, alpha={s_dn.alpha:.6f}')
    print(f'  s_up: A={s_up.A_t:.6f}, V={s_up.V_t:.6f}, alpha={s_up.alpha:.6f}')
