import math
import os
import sys

# ensure plugin directory on path
here = os.path.dirname(os.path.dirname(__file__))
if here not in sys.path:
    sys.path.insert(0, here)

import backwater2 as bw

G = bw.G


def make_rectangular_xs():
    geom = [(0.0, 0.0), (10.0, 0.0)]
    xs_dn = bw.CrossSection(
        river_station='XS_dn', geometry=geom,
        left_bank_station=0.0, right_bank_station=10.0,
        n_lob=0.03, n_ch=0.03, n_rob=0.03,
        L_lob_to_next=0.0, L_ch_to_next=100.0, L_rob_to_next=0.0
    )
    xs_up = bw.CrossSection(
        river_station='XS_up', geometry=geom,
        left_bank_station=0.0, right_bank_station=10.0,
        n_lob=0.03, n_ch=0.03, n_rob=0.03
    )
    return xs_dn, xs_up


def energy_residual_for_solution():
    xs_dn, xs_up = make_rectangular_xs()
    Q = 100.0
    S0 = 0.001
    wse_dn = bw.solve_normal_depth(xs_dn, Q, S0)
    s_dn = bw.compute_state(xs_dn, wse_dn, Q)
    link = bw.ReachLink(xs_dn.L_lob_to_next, xs_dn.L_ch_to_next, xs_dn.L_rob_to_next)
    s_up = bw.solve_energy_upstream(xs_dn=xs_dn, xs_up=xs_up, z_dn=0.0, z_up=0.0,
                                    Q_total_dn=Q, Q_total_up=Q, s_dn=s_dn, link=link,
                                    wse_up_init=s_dn.wse)

    # residual (lhs - rhs) should be ~0
    y_dn = s_dn.wse - 0.0
    y_up = s_up.wse - 0.0
    loss = bw.head_loss(s_dn, s_up, link, xs_dn)
    lhs = 0.0 + y_dn + (s_dn.alpha * s_dn.V_t ** 2) / (2.0 * G)
    rhs = 0.0 + y_up + (s_up.alpha * s_up.V_t ** 2) / (2.0 * G) + loss
    return lhs - rhs


def test_energy_residual():
    res = energy_residual_for_solution()
    assert abs(res) < 1e-4, f"Energy residual too large: {res}"


if __name__ == '__main__':
    r = energy_residual_for_solution()
    print('energy residual =', r)
    if abs(r) < 1e-4:
        print('PASS')
        sys.exit(0)
    else:
        print('FAIL')
        sys.exit(2)
