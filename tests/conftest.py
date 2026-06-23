"""
pytest conftest.py — shared fixtures and configuration.

This file provides pytest fixtures that can be used by both
pytest-style test files (test_bridge_stacked_*.py) and unittest-based
tests via ``@pytest.mark.usefixtures``.

Usage:
    # In a unittest class:
    @pytest.mark.usefixtures("ensure_mock_qgis")
    class TestMyFeature(unittest.TestCase):
        ...

    # In a pytest test function:
    def test_something(ensure_mock_qgis):
        ...
"""

import os
import sys
import pytest

# Ensure repo root and build dir are on sys.path for all discovery modes
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BUILD_DIR = os.path.join(_REPO_ROOT, "build")
for _p in (_REPO_ROOT, _BUILD_DIR):
    if _p not in sys.path and os.path.isdir(_p):
        sys.path.insert(0, _p)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="function")
def ensure_mock_qgis():
    """Install mock QGIS modules before any test that imports swe2d code.

    This fixture ensures that ``from qgis.core import QgsProject`` etc.
    work in a headless environment without a real QGIS installation.
    """
    from tests.mocks.qgis_env import install_qgis_mocks
    install_qgis_mocks()
    yield

    # Optionally reset QgsProject singleton between tests
    from tests.mocks.qgis_env import MockQgsProject
    MockQgsProject._instance = None


@pytest.fixture
def mock_project():
    """Return a fresh mock QgsProject instance."""
    from tests.mocks.qgis_env import MockQgsProject
    MockQgsProject._instance = None
    return MockQgsProject.instance()


@pytest.fixture
def mock_vector_layer():
    """Return a fresh mock QgsVectorLayer."""
    from tests.mocks.qgis_env import MockQgsVectorLayer
    return MockQgsVectorLayer()


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
        if "solver" in markers and not _has_solver:
            item.add_marker(pytest.mark.skip(reason="hydra_swe2d not built"))
        if "gmsh" in markers and not _has_gmsh:
            item.add_marker(pytest.mark.skip(reason="gmsh not installed"))
        if "gpu" in markers and not _has_gpu:
            item.add_marker(pytest.mark.skip(reason="CUDA GPU not available"))
