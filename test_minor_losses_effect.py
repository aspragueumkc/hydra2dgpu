import os, csv
import backwater2 as bw

root = os.path.join(os.getcwd(), 'hec_ras_project')
# read sections metadata
rows = []
with open(os.path.join(root, 'sections_metadata.csv'), 'r', newline='') as f:
    rdr = csv.DictReader(f)
    for r in rdr:
        rows.append(r)

def build_sections(contraction=None, expansion=None):
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
        if contraction is not None:
            xs.contraction_coeff = contraction
        if expansion is not None:
            xs.expansion_coeff = expansion
        sections.append(xs)
    return sections

# HEC-RAS reference WSEs
import h5py
fn = os.path.join(root, 'test3.p01.hdf')
with h5py.File(fn, 'r') as f:
    ws = f['Results']['Steady']['Output']['Output Blocks']['Base Output']['Steady Profiles']['Cross Sections']['Water Surface'][0]
    attrs = f['Geometry']['Cross Sections']['Attributes'][()]
    rs_order = [a[2].decode('utf-8') for a in attrs]

Q = 500.0
S0 = 0.003

# Scenario A: default minor losses (use class defaults: contraction=0.1, expansion=0.3)
sections_A = build_sections()
model_A = bw.ModelInput(flow_cfs=Q, flow_change=None, boundary_condition='normal_depth', boundary_value=S0, sections=sections_A)
results_A = bw.run_backwater(model_A)
compA = {sxn.river_station: st.wse for sxn, st in zip(model_A.sections, results_A)}

# Scenario B: remove minor losses and expansion/contraction (set coeffs=0)
sections_B = build_sections(contraction=0.0, expansion=0.0)
model_B = bw.ModelInput(flow_cfs=Q, flow_change=None, boundary_condition='normal_depth', boundary_value=S0, sections=sections_B)
results_B = bw.run_backwater(model_B)
compB = {sxn.river_station: st.wse for sxn, st in zip(model_B.sections, results_B)}

# Compare
print('RS, HEC_RAS, WITH_LOSSES, NO_LOSSES, DIFF(no-loss - with-loss)')
any_higher = False
for i, rs in enumerate(rs_order):
    hec = float(ws[i])
    a = compA.get(rs)
    b = compB.get(rs)
    diff = b - a if (a is not None and b is not None) else None
    print(f'{rs}, {hec:.6f}, {a:.6f}, {b:.6f}, {diff:.6f}')
    if diff is not None and diff > 0:
        any_higher = True

if any_higher:
    print('\nWARNING: some no-loss WSEs are HIGHER than with-losses (unexpected)')
else:
    print('\nOK: no-loss WSEs are not higher (removing losses produced same or lower WSEs)')
