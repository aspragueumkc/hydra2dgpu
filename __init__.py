"""QGIS plugin entry point for HYDRA

QGIS expects a `classFactory(iface)` function that returns the plugin
instance. This file exposes that function.
"""
import os as _os
import sys as _sys

# Add the compiled C++ extension directory so hydra_tqmesh and
# hydra_swe2d can be imported from anywhere inside the plugin.
_plugin_dir = _os.path.dirname(_os.path.abspath(__file__))
_build_dir = _os.path.join(_plugin_dir, "build")
_release_lib = _os.path.join(_plugin_dir, "lib")
for _d in (_build_dir, _release_lib, _plugin_dir):
    if _d not in _sys.path:
        _sys.path.insert(0, _d)

# ── Lightweight dependency check (logs warnings, never blocks loading) ────
def _check_optional_deps():
    """Warn about missing optional dependencies at plugin load time."""
    import logging
    _log = logging.getLogger("hydra")
    _missing = []
    for _mod, _feat in (
        ("gmsh", "unstructured mesh generation"),
        ("h5py", "HEC-RAS HDF5 export"),
        ("netCDF4", "UGRID NetCDF export"),
        ("matplotlib", "in-plugin plotting"),
    ):
        try:
            __import__(_mod)
        except ImportError:
            _missing.append((_mod, _feat))
    if _missing:
        _names = ", ".join(m for m, _ in _missing)
        _log.info(
            "[HYDRA] Optional packages not installed: %s. "
            "Some features will be unavailable. "
            "Install with: pip install -r requirements.txt",
            _names,
        )


def _check_required_deps():
    """Warn about missing *required* packages at plugin load time.
    Returns True if all required deps are present, False otherwise.
    """
    import logging
    _log = logging.getLogger("hydra")
    _missing = []
    for _mod in ("numpy", "gmsh"):
        try:
            __import__(_mod)
        except ImportError:
            _missing.append(_mod)
    if _missing:
        _names = ", ".join(_missing)
        _log.warning(
            "[HYDRA] REQUIRED packages missing: %s. "
            "The plugin will not function correctly without them. "
            "Open HYDRA2DGPU → Settings → Check & Install Dependencies, "
            "or run from QGIS Python Console:\n"
            "  import sys\n"
            "  subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', '%s'])",
            _names,
            _os.path.join(_plugin_dir, "requirements.txt"),
        )
        return False
    return True

try:
    _check_optional_deps()
    _check_required_deps()
except Exception:
    pass  # never block plugin load


def classFactory(iface):
    from .hydra_plugin import HydraQgisPlugin
    return HydraQgisPlugin(iface)
