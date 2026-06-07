# conftest.py — pytest configuration for the default test suite.
#
# Excludes swe3d tests from the default suite.  These tests are only
# meant to run when actively developing the 3D patch solver.  To run
# them explicitly:
#
#     python -m pytest tests/test_swe3d_*.py -v
#
collect_ignore_glob = ["test_swe3d_*"]
