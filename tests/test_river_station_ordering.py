import os
import sys
import unittest

here = os.path.dirname(os.path.dirname(__file__))
if here not in sys.path:
    sys.path.insert(0, here)

import backwater_model as bw


class TestRiverStationOrdering(unittest.TestCase):
    def _make_xs(self, rs, zmin, zup, lch=100.0):
        return bw.CrossSection(
            river_station=str(rs),
            geometry=[(0.0, float(zmin)), (10.0, float(zup))],
            left_bank_station=0.0,
            right_bank_station=10.0,
            n_lob=0.03,
            n_ch=0.03,
            n_rob=0.03,
            L_lob_to_next=0.0,
            L_ch_to_next=lch,
            L_rob_to_next=0.0,
        )

    def test_run_backwater_sorts_by_numeric_river_station(self):
        # Intentionally out of order: US, middle, DS — solver must sort ascending (low RS = DS)
        xs_10 = self._make_xs("10", 1.0, 1.0)
        xs_3 = self._make_xs("3", 0.5, 0.5)
        xs_0 = self._make_xs("0", 0.0, 0.0)

        model = bw.ModelInput(
            flow_cfs=100.0,
            flow_change=None,
            boundary_condition="known_wse",
            boundary_value=2.0,
            sections=[xs_10, xs_3, xs_0],
        )

        results = bw.run_backwater(model, solver="py")

        # DS->US order: 0 is downstream, 10 is upstream
        self.assertEqual(["0", "3", "10"], [xs.river_station for xs in model.sections])
        self.assertEqual(3, len(results))
        self.assertAlmostEqual(2.0, results[0].wse, places=6)

    def test_run_backwater_known_wse_below_bed_raises(self):
        # known_wse below downstream bed must raise a clear error
        xs_10 = self._make_xs("10", 10.0, 10.0)
        xs_0  = self._make_xs("0",   9.0,  9.0)

        model = bw.ModelInput(
            flow_cfs=100.0,
            flow_change=None,
            boundary_condition="known_wse",
            boundary_value=5.0,   # below downstream (RS 0) bed of 9.0
            sections=[xs_10, xs_0],
        )

        with self.assertRaises(ValueError):
            bw.run_backwater(model, solver="py")


if __name__ == "__main__":
    unittest.main()
