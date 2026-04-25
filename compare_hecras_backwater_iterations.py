import os, csv, json
import h5py
import math

import backwater2 as bw

root = os.path.join(os.getcwd(), 'hec_ras_project')
# read HDF water-surface and RS ordering
fn = os.path.join(root, 'test3.p01.hdf')
with h5py.File(fn, 'r') as f:
    ws = f['Results']['Steady']['Output']['Output Blocks']['Base Output']['Steady Profiles']['Cross Sections']['Water Surface'][0]
    attrs = f['Geometry']['Cross Sections']['Attributes'][()]
    rs_order = [a[2].decode('utf-8') for a in attrs]  # RS field

# read sections metadata
rows = []
with open(os.path.join(root, 'sections_metadata.csv'), 'r', newline='') as f:
    rdr = csv.DictReader(f)
    for r in rdr:
        rows.append(r)

# helper to build sections from metadata
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
        # read csv geometry
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

# iterations definitions
iterations = [
    {'id': 0, 'name': 'baseline', 'desc': 'Baseline backwater2 implementation', 'params': {}},
    {'id': 1, 'name': 'Sf_avg', 'desc': 'Use average friction slope between sections (0.5*(Sf1+Sf2))', 'params': {'Sf_method': 'avg'}},
    {'id': 2, 'name': 'alpha1_minorloss', 'desc': 'Ignore alpha in minor loss (use alpha=1)', 'params': {'minor_alpha': 1}},
    {'id': 3, 'name': 'no_expansion', 'desc': 'Set expansion coeff to 0 for all cross sections', 'params': {'expansion_coeff': 0.0}}
]

# additional experiments
iterations += [
    {'id': 4, 'name': 'channel_Ldw', 'desc': 'Use channel length (L_ch) instead of DWRL for friction', 'params': {'use_channel_length': True}},
    {'id': 5, 'name': 'no_minor', 'desc': 'Ignore minor losses (C=0)', 'params': {'no_minor_losses': True}},
    {'id': 6, 'name': 'Sf_avg_noexp', 'desc': 'Average Sf and set expansion coeff 0', 'params': {'Sf_method': 'avg', 'expansion_coeff': 0.0}},
    {'id': 7, 'name': 'exp05', 'desc': 'Set expansion coeff to 0.05', 'params': {'expansion_coeff': 0.05}},
]

# Additional solver variants to try to better match HEC-RAS
iterations += [
    {'id': 8, 'name': 'alpha_avg', 'desc': 'Use average alpha in minor loss term', 'params': {'alpha_avg': True}},
    {'id': 9, 'name': 'mann_1486', 'desc': 'Use Manning constant 1.486', 'params': {'mann_const': 1.486}},
    {'id':10, 'name': 'Sf_avg_alpha_avg', 'desc': 'Average Sf and average alpha in minor loss', 'params': {'Sf_method': 'avg', 'alpha_avg': True}},
    {'id':11, 'name': 'Sf_avg_mann1486', 'desc': 'Average Sf and Manning 1.486', 'params': {'Sf_method': 'avg', 'mann_const': 1.486}},
]

# Targeted combos to try to better match HEC-RAS (Sf avg, alpha avg, tuned expansion coeffs, mannings)
iterations += [
    {'id': 12, 'name': 'Sf_alpha_exp005', 'desc': 'Sf avg + alpha avg + expansion coeff 0.05', 'params': {'Sf_method': 'avg', 'alpha_avg': True, 'expansion_coeff': 0.05}},
    {'id': 13, 'name': 'Sf_alpha_exp002', 'desc': 'Sf avg + alpha avg + expansion coeff 0.02', 'params': {'Sf_method': 'avg', 'alpha_avg': True, 'expansion_coeff': 0.02}},
    {'id': 14, 'name': 'Sf_alpha_mann1486_exp005', 'desc': 'Sf avg + alpha avg + Manning 1.486 + expansion 0.05', 'params': {'Sf_method': 'avg', 'alpha_avg': True, 'mann_const': 1.486, 'expansion_coeff': 0.05}},
    {'id': 15, 'name': 'alpha_mann1486_exp005', 'desc': 'alpha avg + Manning 1.486 + expansion 0.05', 'params': {'alpha_avg': True, 'mann_const': 1.486, 'expansion_coeff': 0.05}},
    {'id': 16, 'name': 'Sf_alpha_mann1486_exp002', 'desc': 'Sf avg + alpha avg + Manning 1.486 + expansion 0.02', 'params': {'Sf_method': 'avg', 'alpha_avg': True, 'mann_const': 1.486, 'expansion_coeff': 0.02}},
]

# prepare output files
out_csv = 'hec_vs_backwater_iterations.csv'
summary_csv = 'hec_vs_backwater_summary.csv'
log_json = 'hec_vs_backwater_log.json'

# Solver to use for run_backwater: 'py' or 'scipy'
SOLVER = 'scipy'

rows_out = []
summary_rows = []
log = []

Q = 500.0
S0 = 0.003

for it in iterations:
    it_id = it['id']
    it_name = it['name']
    it_desc = it['desc']
    params = it['params']

    # build sections, applying per-iteration modifications
    if it_name == 'no_expansion':
        sections = build_sections(expansion_coeff_override=0.0)
    else:
        sections = build_sections()

    # monkeypatch functions if needed
    # Save originals
    rep_func_orig = bw.representative_friction_slope_total
    head_loss_orig = bw.head_loss

    if params.get('Sf_method') == 'avg':
        def rep_avg(s1, s2):
            return 0.5 * (s1.Sf_total + s2.Sf_total)
        bw.representative_friction_slope_total = rep_avg

    if params.get('minor_alpha') == 1:
        def head_loss_alpha1(s_dn, s_up, link, xs_dn):
            Qlob_av = 0.5 * (s_dn.Q_lob + s_up.Q_lob)
            Qch_av  = 0.5 * (s_dn.Q_ch  + s_up.Q_ch)
            Qrob_av = 0.5 * (s_dn.Q_rob + s_up.Q_rob)
            Ldw = link.discharge_weighted_length(Qlob_av, Qch_av, Qrob_av)
            Sf = bw.representative_friction_slope_total(s_dn, s_up)
            hf = Sf * Ldw
            C = bw.minor_loss_coeff(s_dn, s_up, xs_dn)
            hv = C * ((1.0 * s_up.V_t ** 2) - (1.0 * s_dn.V_t ** 2)) / (2.0 * bw.G)
            return hf + hv
        bw.head_loss = head_loss_alpha1

    if params.get('use_channel_length'):
        def head_loss_chanL(s_dn, s_up, link, xs_dn):
            Qlob_av = 0.5 * (s_dn.Q_lob + s_up.Q_lob)
            Qch_av  = 0.5 * (s_dn.Q_ch  + s_up.Q_ch)
            Qrob_av = 0.5 * (s_dn.Q_rob + s_up.Q_rob)
            # use channel length only
            Ldw = link.L_ch
            Sf = bw.representative_friction_slope_total(s_dn, s_up)
            hf = Sf * Ldw
            C = bw.minor_loss_coeff(s_dn, s_up, xs_dn)
            hv = C * ((s_up.alpha * s_up.V_t ** 2) - (s_dn.alpha * s_dn.V_t ** 2)) / (2.0 * bw.G)
            return hf + hv
        bw.head_loss = head_loss_chanL

    if params.get('no_minor_losses'):
        def head_loss_no_minor(s_dn, s_up, link, xs_dn):
            Qlob_av = 0.5 * (s_dn.Q_lob + s_up.Q_lob)
            Qch_av  = 0.5 * (s_dn.Q_ch  + s_up.Q_ch)
            Qrob_av = 0.5 * (s_dn.Q_rob + s_up.Q_rob)
            Ldw = link.discharge_weighted_length(Qlob_av, Qch_av, Qrob_av)
            Sf = bw.representative_friction_slope_total(s_dn, s_up)
            hf = Sf * Ldw
            return hf
        bw.head_loss = head_loss_no_minor

    # average-alpha minor loss
    if params.get('alpha_avg'):
        def head_loss_alpha_avg(s_dn, s_up, link, xs_dn):
            Qlob_av = 0.5 * (s_dn.Q_lob + s_up.Q_lob)
            Qch_av  = 0.5 * (s_dn.Q_ch  + s_up.Q_ch)
            Qrob_av = 0.5 * (s_dn.Q_rob + s_up.Q_rob)
            Ldw = link.discharge_weighted_length(Qlob_av, Qch_av, Qrob_av)
            Sf = bw.representative_friction_slope_total(s_dn, s_up)
            hf = Sf * Ldw
            C = bw.minor_loss_coeff(s_dn, s_up, xs_dn)
            avg_alpha = 0.5 * (s_dn.alpha + s_up.alpha)
            hv = C * (avg_alpha * (s_up.V_t ** 2 - s_dn.V_t ** 2)) / (2.0 * bw.G)
            return hf + hv
        bw.head_loss = head_loss_alpha_avg

    # override Manning constant if requested
    mann_orig = None
    if 'mann_const' in params:
        mann_orig = getattr(bw, 'MANNING_CONST', None)
        bw.MANNING_CONST = float(params['mann_const'])

    # run model
    model = bw.ModelInput(flow_cfs=Q, flow_change=None, boundary_condition='normal_depth', boundary_value=S0, sections=sections)
    try:
        results = bw.run_backwater(model, solver=SOLVER)
    except Exception as e:
        # log the error and continue to next iteration
        rows_out.append({'iteration': it_id, 'iter_name': it_name, 'description': it_desc, 'params': json.dumps(params), 'RS': None, 'HEC_WSE': None, 'BACKW_WSE': None, 'DIFF': None, 'error': str(e)})
        summary_rows.append({'iteration': it_id, 'iter_name': it_name, 'description': it_desc, 'params': json.dumps(params), 'max_abs_diff': None, 'mean_abs_diff': None, 'rmse': None})
        log.append({'iteration': it_id, 'iter_name': it_name, 'params': params, 'error': str(e)})
        continue

    # restore originals
    bw.representative_friction_slope_total = rep_func_orig
    bw.head_loss = head_loss_orig

    # compare to HEC-RAS WSEs (rs_order maps index to RS id)
    comp_map = {sxn.river_station: st.wse for sxn, st in zip(model.sections, results)}
    diffs = []
    for i, rs in enumerate(rs_order):
        hec = float(ws[i])
        backw = comp_map.get(rs)
        diff = None
        if backw is not None:
            diff = backw - hec
            diffs.append(abs(diff))
        rows_out.append({'iteration': it_id, 'iter_name': it_name, 'description': it_desc, 'params': json.dumps(params), 'RS': rs, 'HEC_WSE': hec, 'BACKW_WSE': backw, 'DIFF': diff})

    max_abs = max(diffs) if diffs else None
    mean_abs = sum(diffs)/len(diffs) if diffs else None
    rmse = math.sqrt(sum(d*d for d in diffs)/len(diffs)) if diffs else None
    summary_rows.append({'iteration': it_id, 'iter_name': it_name, 'description': it_desc, 'params': json.dumps(params), 'max_abs_diff': max_abs, 'mean_abs_diff': mean_abs, 'rmse': rmse})
    log.append({'iteration': it_id, 'iter_name': it_name, 'params': params, 'summary': summary_rows[-1]})

# write CSV and summary
with open(out_csv, 'w', newline='') as f:
    fieldnames = ['iteration', 'iter_name', 'description', 'params', 'RS', 'HEC_WSE', 'BACKW_WSE', 'DIFF', 'error']
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in rows_out:
        w.writerow(r)

with open(summary_csv, 'w', newline='') as f:
    fieldnames = ['iteration', 'iter_name', 'description', 'params', 'max_abs_diff', 'mean_abs_diff', 'rmse']
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in summary_rows:
        w.writerow(r)

with open(log_json, 'w') as f:
    json.dump(log, f, indent=2)

print('Wrote', out_csv, summary_csv, log_json)
