"""
pytest conftest.py — shared fixtures and configuration.

This file provides pytest fixtures that can be used by both
pytest-style test files (test_bridge_stacked_*.py) and unittest-based
tests via ``@pytest.mark.usefixtures``.

Usage:
    # In a pytest test function requiring a real headless QGIS:
    def test_something(real_qgis):
        ...
"""

import os
import sys
import pytest

# Ensure repo root and build dir are on sys.path for all discovery modes.
#
# Build directory selection:
#   - ``HYDRA_BUILD_DIR`` env var, if set, overrides the default.
#     Use this to point a worktree's tests at a sibling checkout's build
#     (e.g. ``HYDRA_BUILD_DIR=/path/to/main-repo/build pytest …``) instead
#     of rebuilding the worktree.
#   - Otherwise the default ``<repo_root>/build`` is used.
#
# When the env var is set, the build dir is inserted at position 0 so it
# takes precedence over per-test-file sys.path manipulations that hardcode
# ``<repo_root>/build``.
_HYDRA_BUILD_DIR = os.environ.get("HYDRA_BUILD_DIR") or ""
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_BUILD_DIR = os.path.join(_REPO_ROOT, "build")

for _p in (_REPO_ROOT, _HYDRA_BUILD_DIR or _DEFAULT_BUILD_DIR):
    if _p and _p not in sys.path and os.path.isdir(_p):
        sys.path.insert(0, _p)
# If HYDRA_BUILD_DIR was set, also prepend it explicitly so per-test
# sys.path.insert(... "build") inserts don't mask it later.
if _HYDRA_BUILD_DIR and os.path.isdir(_HYDRA_BUILD_DIR):
    sys.path.insert(0, _HYDRA_BUILD_DIR)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def real_qgis():
    """Session-scoped real headless QgsApplication.

    Must be the only QGIS-app initializer in the test process (see
    docs/plans/2026-08-02-real-qgis-test-migration.md §Non-negotiables).
    """
    import importlib.util
    if importlib.util.find_spec("qgis.core") is None:
        pytest.skip("real QGIS env required (qgis_stable mamba env)")
    from tests.qgis_real_env import ensure_qgis_app
    yield ensure_qgis_app()


@pytest.fixture
def fallback_tracker():
    """Return a FallbackTracker instance for detecting silent fallbacks."""
    from tests.test_helpers import FallbackTracker
    return FallbackTracker


@pytest.fixture(scope="session")
def unit_config_si():
    """Configure swe2d.units for SI (metric) and return the module."""
    from swe2d import units
    units.configure(1.0)
    return units


@pytest.fixture(scope="session")
def unit_config_usc():
    """Configure swe2d.units for USC (feet) and return the module."""
    from swe2d import units
    units.configure(0.3048)
    return units


# ═══════════════════════════════════════════════════════════════════════════════
# pytest hooks
# ═══════════════════════════════════════════════════════════════════════════════

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "qgis: mark test as requiring QGIS (skipped if QGIS not available)",
    )
    config.addinivalue_line(
        "markers",
        "gpu: mark test as requiring CUDA GPU (skipped if GPU not available)",
    )
    config.addinivalue_line(
        "markers",
        "gmsh: mark test as requiring gmsh Python package",
    )
    config.addinivalue_line(
        "markers",
        "solver: mark test as requiring hydra_swe2d native module",
    )


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests that require unavailable dependencies."""
    import importlib

    _has_qgis = importlib.util.find_spec("qgis.core") is not None
    _has_solver = importlib.util.find_spec("hydra_swe2d") is not None
    _has_gmsh = importlib.util.find_spec("gmsh") is not None
    _has_gpu = False
    if _has_solver:
        try:
            import hydra_swe2d
            _has_gpu = hydra_swe2d.swe2d_gpu_available()
        except Exception:
            pass

    for item in items:
        markers = {m.name for m in item.iter_markers()}
        if "qgis" in markers and not _has_qgis:
            item.add_marker(pytest.mark.skip(reason="real QGIS not importable"))
        if "solver" in markers and not _has_solver:
            item.add_marker(pytest.mark.skip(reason="hydra_swe2d not built"))
        if "gmsh" in markers and not _has_gmsh:
            item.add_marker(pytest.mark.skip(reason="gmsh not installed"))
        if "gpu" in markers and not _has_gpu:
            item.add_marker(pytest.mark.skip(reason="CUDA GPU not available"))
