from __future__ import annotations

import math
import unittest

from swe2d_nonorth_gpu_sweep_common import run_gpu_nonorth_vs_orth_sweep
from tests._swe2d_test_helpers import _load_module


@unittest.skipUnless(_load_module() is not None, "hydra_swe2d not built")
class TestGPUChannelOrthVsNonOrth100Cells(unittest.TestCase):
    def test_gpu_sweep_100_cells(self):
        out = run_gpu_nonorth_vs_orth_sweep(
            nx=10,
            ny=5,
            lx=800.0,
            ly=20.0,
            s0=1.0e-3,
            n_mann=0.02,
            q_in=0.8,
            nsteps=240,
            skew_fraction_dx=0.25,
            artifact_tag="swe2d_nonorth_vs_orth_gpu_sweep_100cells",
        )

        self.assertEqual(out["ok_count"], out["total_count"], f"Some combos failed: {out}")
        self.assertTrue(out["all_gpu_active"], "Expected GPU active for all method combos")
        self.assertTrue(math.isfinite(out["max_rel_q_pct"]))
        self.assertLess(out["max_rel_q_pct"], 10.0, f"rel_q delta too high: {out['max_rel_q_pct']:.4f}%")


if __name__ == "__main__":
    unittest.main(verbosity=2)
