"""Vulture whitelist for test-only dynamic imports and framework boilerplate.

This file is scanned by vulture as a source of "dummy" usages. Listing a
name here tells vulture the symbol is intentionally referenced even when the
static call graph cannot see it.
"""

# ANUGA validation suite loads these modules dynamically in load_tests().
from tests import (
    test_swe2d_gpu_dam_break_dry,
    test_swe2d_gpu_dam_break_wet,
    test_swe2d_gpu_deep_wave,
    test_swe2d_gpu_lake_at_rest_immersed_bump,
    test_swe2d_gpu_lake_at_rest_steep_island,
    test_swe2d_gpu_mac_donald_short_channel,
    test_swe2d_gpu_parabolic_basin,
    test_swe2d_gpu_river_at_rest_varying_topo_width,
    test_swe2d_gpu_rundown_mild_slope,
    test_swe2d_gpu_runup_on_beach,
    test_swe2d_gpu_runup_on_sinusoid_beach,
    test_swe2d_gpu_subcritical_depth_expansion,
    test_swe2d_gpu_subcritical_flat,
    test_swe2d_gpu_subcritical_over_bump,
    test_swe2d_gpu_supercritical_over_bump,
    test_swe2d_gpu_transcritical_with_shock,
    test_swe2d_gpu_transcritical_without_shock,
    test_swe2d_gpu_trapezoidal_channel,
)

# Keep references so vulture sees them as used.
_ = (
    test_swe2d_gpu_dam_break_dry,
    test_swe2d_gpu_dam_break_wet,
    test_swe2d_gpu_deep_wave,
    test_swe2d_gpu_lake_at_rest_immersed_bump,
    test_swe2d_gpu_lake_at_rest_steep_island,
    test_swe2d_gpu_mac_donald_short_channel,
    test_swe2d_gpu_parabolic_basin,
    test_swe2d_gpu_river_at_rest_varying_topo_width,
    test_swe2d_gpu_rundown_mild_slope,
    test_swe2d_gpu_runup_on_beach,
    test_swe2d_gpu_runup_on_sinusoid_beach,
    test_swe2d_gpu_subcritical_depth_expansion,
    test_swe2d_gpu_subcritical_flat,
    test_swe2d_gpu_subcritical_over_bump,
    test_swe2d_gpu_supercritical_over_bump,
    test_swe2d_gpu_transcritical_with_shock,
    test_swe2d_gpu_transcritical_without_shock,
    test_swe2d_gpu_trapezoidal_channel,
)

# unittest discovery hook parameters that are required by the protocol.
def _load_tests(loader, standard_tests, pattern):
    return standard_tests, pattern
