"""Master runner for the ANUGA validation suite.

Loads every ANUGA-validated test via explicit import, filtering by
`anuga_reference` class attribute pointing into
reference/anuga_validation_tests/.

Run:
    python -m unittest tests.test_anuga_suite -v
"""

import sys
import unittest


def load_tests(loader, standard_tests, pattern):
    """Unittest discovery hook: load every ANUGA test class by anuga_reference."""
    suite = unittest.TestSuite()

    # Import all ANUGA test modules
    from tests import (
        test_swe2d_gpu_dam_break_wet,
        test_swe2d_gpu_dam_break_dry,
        test_swe2d_gpu_subcritical_over_bump,
        test_swe2d_gpu_supercritical_over_bump,
        test_swe2d_gpu_transcritical_with_shock,
        test_swe2d_gpu_transcritical_without_shock,
        test_swe2d_gpu_lake_at_rest_steep_island,
        test_swe2d_gpu_lake_at_rest_immersed_bump,
        test_swe2d_gpu_subcritical_flat,
        test_swe2d_gpu_subcritical_depth_expansion,
        test_swe2d_gpu_mac_donald_short_channel,
        test_swe2d_gpu_parabolic_basin,
        test_swe2d_gpu_river_at_rest_varying_topo_width,
        test_swe2d_gpu_runup_on_beach,
        test_swe2d_gpu_runup_on_sinusoid_beach,
        test_swe2d_gpu_deep_wave,
        test_swe2d_gpu_rundown_mild_slope,
        test_swe2d_gpu_trapezoidal_channel,
    )

    for mod_name, mod in list(sys.modules.items()):
        if not mod_name.startswith("tests.test_swe2d_gpu_"):
            continue
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name, None)
            if not (isinstance(attr, type) and issubclass(attr, unittest.TestCase)
                    and attr is not unittest.TestCase):
                continue
            ref = getattr(attr, "anuga_reference", "")
            if ref.startswith("reference/anuga_validation_tests/"):
                suite.addTests(loader.loadTestsFromTestCase(attr))

    return suite


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = load_tests(loader, None, None)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
