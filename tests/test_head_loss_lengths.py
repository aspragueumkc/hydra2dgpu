import unittest
import math
import sys
sys.path.insert(0, '..')
import hydra_1d as bw

class TestHeadLossUsesUpstreamLength(unittest.TestCase):
    def test_head_loss_uses_upstream_lengths(self):
        # Build minimal downstream/upstream SectionState objects with
        # matching velocity/head to eliminate minor loss contribution.
        s_dn = bw.SectionState(
            wse=100.0, depth_at_min=2.0, alpha=1.0,
            A_lob=10.0, A_ch=10.0, A_rob=10.0,
            K_lob=10.0, K_ch=10.0, K_rob=10.0,
            Q_lob=10.0, Q_ch=10.0, Q_rob=10.0,
            V_t=1.0, K_t=30.0, A_t=30.0, Sf_total=0.001, Froude=0.5
        )
        s_up = bw.SectionState(
            wse=101.0, depth_at_min=3.0, alpha=1.0,
            A_lob=12.0, A_ch=12.0, A_rob=12.0,
            K_lob=12.0, K_ch=12.0, K_rob=12.0,
            Q_lob=20.0, Q_ch=20.0, Q_rob=20.0,
            V_t=1.0, K_t=36.0, A_t=36.0, Sf_total=0.004, Froude=0.4
        )
        # Use an upstream CrossSection with distinctive L_* values
        xs_up = bw.CrossSection(
            river_station='up', geometry=[(0,98),(10,97)],
            left_bank_station=0, right_bank_station=10,
            n_lob=0.03, n_ch=0.03, n_rob=0.03,
            contraction_coeff=0.0, expansion_coeff=0.0,
            L_lob_to_next=100.0, L_ch_to_next=200.0, L_rob_to_next=300.0
        )
        # Downstream cross-section (lengths different)
        xs_dn = bw.CrossSection(
            river_station='dn', geometry=[(0,99),(10,98)],
            left_bank_station=0, right_bank_station=10,
            n_lob=0.03, n_ch=0.03, n_rob=0.03,
            contraction_coeff=0.0, expansion_coeff=0.0,
            L_lob_to_next=1.0, L_ch_to_next=1.0, L_rob_to_next=1.0
        )

        # Force SF_METHOD to 'avg' so representative_friction_slope_total is predictable
        old_sf = bw.SF_METHOD
        bw.SF_METHOD = 'avg'
        try:
            # average Sf
            Sf = 0.5 * (s_dn.Sf_total + s_up.Sf_total)
            # build link using upstream lengths (what run_backwater now does)
            link_up = bw.ReachLink(xs_up.L_lob_to_next, xs_up.L_ch_to_next, xs_up.L_rob_to_next)
            # compute head_loss (minor loss zero because coeffs are zero and alphas/velocities equal)
            hl = bw.head_loss(s_dn, s_up, link_up, xs_dn)
            # expected discharge-weighted length
            Qlob_av = 0.5 * (s_dn.Q_lob + s_up.Q_lob)
            Qch_av  = 0.5 * (s_dn.Q_ch  + s_up.Q_ch)
            Qrob_av = 0.5 * (s_dn.Q_rob + s_up.Q_rob)
            Ldw = link_up.discharge_weighted_length(Qlob_av, Qch_av, Qrob_av)
            expected_hf = Sf * Ldw
            # head_loss should equal hf (minor loss zero)
            self.assertAlmostEqual(hl, expected_hf, places=9)
        finally:
            bw.SF_METHOD = old_sf

if __name__ == '__main__':
    unittest.main()
