import math
import unittest

from backwater2 import CrossSection, solve_culvert_headwater, G
from culvert_routine import CircularXsect, solve_normal_depth_in_culvert


class TestCulvertHydraulics(unittest.TestCase):
    def test_culvert_normal_depth_is_capped_at_rise(self):
        xsect = CircularXsect(diameter_ft=3.0, culvert_code=1)
        normal_depth = solve_normal_depth_in_culvert(
            xsect=xsect,
            Q=500.0,
            n_value=0.012,
            slope=0.001,
        )
        self.assertAlmostEqual(normal_depth, xsect.yFull, places=6)

    def test_low_tailwater_avoids_pressurized_full_flow_headwater(self):
        xs = CrossSection(
            river_station='CULV',
            geometry=[(0.0, 100.0), (20.0, 100.0), (40.0, 100.0)],
            left_bank_station=10.0,
            right_bank_station=30.0,
            n_lob=0.04,
            n_ch=0.02,
            n_rob=0.04,
            contraction_coeff=0.1,
            expansion_coeff=0.3,
            L_lob_to_next=50.0,
            L_ch_to_next=50.0,
            L_rob_to_next=50.0,
            culvert_code=1,
            culvert_shape='circular',
            culvert_diameter=3.0,
            culvert_upstream_invert=100.0,
            culvert_downstream_invert=99.0,
            culvert_length=300.0,
        )

        q_total = 20.0
        tailwater_wse = 99.6
        hw_wse, control = solve_culvert_headwater(xs, tailwater_wse, q_total)

        area_full = math.pi * (xs.culvert_diameter / 2.0) ** 2
        perimeter_full = math.pi * xs.culvert_diameter
        radius_full = area_full / perimeter_full
        kf = (2.0 * G * xs.n_ch ** 2 * xs.culvert_length) / (1.486 ** 2 * radius_full ** (4.0 / 3.0))
        v_full = q_total / area_full
        legacy_pressurized_hw = max(tailwater_wse, xs.culvert_downstream_invert + xs.culvert_diameter) + (1.0 + 0.5 + kf) * (v_full ** 2) / (2.0 * G)

        self.assertEqual(control, 'outlet')
        self.assertLess(hw_wse, legacy_pressurized_hw)


if __name__ == '__main__':
    unittest.main()