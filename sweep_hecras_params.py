import os, csv, json, math
import h5py
import backwater2 as bw

root = os.path.join(os.getcwd(), 'hec_ras_project')
fn = os.path.join(root, 'test3.p01.hdf')
with h5py.File(fn, 'r') as f:
    ws = f['Results']['Steady']['Output']['Output Blocks']['Base Output']['Steady Profiles']['Cross Sections']['Water Surface'][0]
    attrs = f['Geometry']['Cross Sections']['Attributes'][()]
    rs_order = [a[2].decode('utf-8') for a in attrs]

# read sections metadata
rows = []
with open(os.path.join(root, 'sections_metadata.csv'), 'r', newline='') as f:
    rdr = csv.DictReader(f)
    for r in rdr:
        rows.append(r)

# helper to build sections with optional expansion override
def build_sections(expansion_coeff_override=None):
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
        if expansion_coeff_override is not None:
            xs.expansion_coeff = expansion_coeff_override
        sections.append(xs)
    return sections

# parameter ranges
mannings = [round(1.30 + i*0.02, 3) for i in range(16)]  # 1.30 .. 1.60
exp_coeffs = [0.0, 0.02, 0.05, 0.1]
alpha_methods = ['conveyance', 'area']
sf_methods = ['combined', 'avg']

# fixed model params
Q = 500.0
S0 = 0.003
SOLVER = 'scipy'

out_rows = []
count = 0
best = None

# save originals
rep_orig = bw.representative_friction_slope_total
xs_hyd_orig = bw.CrossSection.hydraulics_at_wse
mann_orig = getattr(bw, 'MANNING_CONST', None)

for mann in mannings:
    for expc in exp_coeffs:
        for alpha_m in alpha_methods:
            for sf_m in sf_methods:
                count += 1
                # build sections
                sections = build_sections(expansion_coeff_override=expc)

                # monkeypatch Sf method
                if sf_m == 'avg':
                    def rep_avg(s1, s2):
                        return 0.5 * (s1.Sf_total + s2.Sf_total)
                    bw.representative_friction_slope_total = rep_avg
                else:
                    bw.representative_friction_slope_total = rep_orig

                # monkeypatch CrossSection.hydraulics_at_wse to adjust alpha if requested
                def make_hyd_with_alpha(orig_func, mode):
                    def wrapped(self, wse, Q_total):
                        res = orig_func(self, wse, Q_total)
                        if mode == 'area':
                            # compute area-weighted alpha: sum(Ai*Vi^2)/(At*Vt^2)
                            lob = res['lob']; ch = res['ch']; rob = res['rob']; tot = res['totals']
                            At = tot.A; Vt = tot.V
                            denom = At * Vt * Vt if At > 0 and Vt > 0 else None
                            if denom:
                                num = 0.0
                                for sub in (lob, ch, rob):
                                    Ai = sub.A; Vi = sub.V
                                    num += Ai * (Vi ** 2)
                                alpha_alt = num / denom if denom else res.get('alpha', 1.0)
                                res['alpha'] = alpha_alt
                        return res
                    return wrapped

                if alpha_m == 'area':
                    bw.CrossSection.hydraulics_at_wse = make_hyd_with_alpha(xs_hyd_orig, 'area')
                else:
                    bw.CrossSection.hydraulics_at_wse = xs_hyd_orig

                # set Manning
                bw.MANNING_CONST = float(mann)

                # run model
                model = bw.ModelInput(flow_cfs=Q, flow_change=None, boundary_condition='normal_depth', boundary_value=S0, sections=sections)
                try:
                    results = bw.run_backwater(model, solver=SOLVER)
                except Exception as e:
                    print('Run failed for', mann, expc, alpha_m, sf_m, '->', e)
                    continue

                # compare to HEC-RAS
                comp_map = {sxn.river_station: st.wse for sxn, st in zip(model.sections, results)}
                diffs = []
                for i, rs in enumerate(rs_order):
                    hec = float(ws[i])
                    backw = comp_map.get(rs)
                    if backw is not None:
                        diffs.append(abs(backw - hec))
                if not diffs:
                    continue
                mean_abs = sum(diffs)/len(diffs)
                rmse = math.sqrt(sum(d*d for d in diffs)/len(diffs))
                max_abs = max(diffs)

                out_rows.append({'mann': mann, 'expansion_coeff': expc, 'alpha_method': alpha_m, 'sf_method': sf_m,
                                 'max_abs': max_abs, 'mean_abs': mean_abs, 'rmse': rmse})

                if best is None or mean_abs < best['mean_abs']:
                    best = out_rows[-1]

# restore
bw.representative_friction_slope_total = rep_orig
bw.CrossSection.hydraulics_at_wse = xs_hyd_orig
if mann_orig is not None:
    bw.MANNING_CONST = mann_orig

# write results
out_csv = 'sweep_hecras_params_summary.csv'
with open(out_csv, 'w', newline='') as f:
    fieldnames = ['mann', 'expansion_coeff', 'alpha_method', 'sf_method', 'max_abs', 'mean_abs', 'rmse']
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in out_rows:
        w.writerow(r)

with open('sweep_hecras_params_best.json', 'w') as f:
    json.dump({'best': best, 'count': count}, f, indent=2)

print('Done sweep. Ran', count, 'cases. Best:', best)
print('Wrote', out_csv, 'and sweep_hecras_params_best.json')
