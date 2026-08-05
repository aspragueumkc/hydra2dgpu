"""QGIS plugin entry point for HYDRA2DGPU

QGIS expects a `classFactory(iface)` function that returns the plugin
instance. This file exposes that function.
"""
import os as _os
import sys as _sys
import importlib as _importlib
from glob import glob as _glob

# Add the compiled C++ extension directory so hydra_swe2d can be imported
# hydra_swe2d can be imported from anywhere inside the plugin.
# After the Phase 1.1 move, _plugin_dir is repo/qgis_plugin/HYDRA2DGPU/ and
# PLUGIN_ROOT is repo/qgis_plugin/. The repo root (one level above PLUGIN_ROOT)
# is also added so that the `swe2d` package remains importable.
# realpath() resolves the symlink under qgis prefix so paths are stable
# regardless of how the plugin is loaded (symlink vs direct).
_plugin_dir = _os.path.dirname(_os.path.realpath(__file__))
PLUGIN_ROOT = _os.path.dirname(_plugin_dir)
_repo_root = _os.path.dirname(PLUGIN_ROOT)
_build_dir = _os.path.join(PLUGIN_ROOT, "..", "build")
_release_lib = _os.path.join(PLUGIN_ROOT, "..", "lib")
for _d in (_build_dir, _release_lib, _plugin_dir, _repo_root):
    if _d not in _sys.path:
        _sys.path.insert(0, _d)

# Production install: the BackendInstaller (installer.py) writes the wheel
# into ~/.hydra2dgpu/ on first launch. We have to prepend its site-packages
# here so the next QGIS launch can find `swe2d` and `hydra_swe2d`. (The dev
# symlink workflow doesn't need this — the realpath math above already pulls
# in the dev repo's swe2d/. The production install is a separate, parallel
# path that needs its own wiring.)
#
# The venv layout differs by OS:
#   Windows: <env>/Lib/site-packages
#   POSIX:   <env>/lib/pythonX.Y/site-packages
_hydra2dgpu_dir = _os.path.join(_os.path.expanduser("~"), ".hydra2dgpu")
_win_sp = _os.path.join(_hydra2dgpu_dir, "Lib", "site-packages")
if _os.path.isdir(_win_sp) and _win_sp not in _sys.path:
    _sys.path.insert(0, _win_sp)
_hydra2dgpu_lib = _os.path.join(_hydra2dgpu_dir, "lib")
if _os.path.isdir(_hydra2dgpu_lib):
    for _pyver_sp in sorted(_os.listdir(_hydra2dgpu_lib)):
        _sp = _os.path.join(_hydra2dgpu_lib, _pyver_sp, "site-packages")
        if _os.path.isdir(_sp) and _sp not in _sys.path:
            _sys.path.insert(0, _sp)

# ── Windows multiprocessing guard ─────────────────────────────────────────
# On Windows, Python's multiprocessing spawns child processes using
# sys.executable.  In OSGeo4W / QGIS installs, sys.executable points to
# the QGIS launcher binary, so spawned processes start a new QGIS instance
# with --multiprocessing-fork — which QGIS interprets as a data source path.
# We fix this by setting multiprocessing to use the real Python interpreter.
_startup_messages = []  # (tag, msg) list consumed by startup_state.py
if _os.name == 'nt' and '--multiprocessing-fork' not in _sys.argv:
    try:
        import multiprocessing as _mp
        _real_python = None
        _candidates = (
            _os.path.join(_sys.base_exec_prefix, 'python.exe'),
            _os.path.join(_os.path.dirname(_sys.executable), 'python.exe'),
        )
        for _candidate in _candidates:
            if _os.path.exists(_candidate):
                _real_python = _candidate
                break
        if _real_python and _real_python != _sys.executable:
            _mp.set_executable(_real_python)
        elif not _real_python:
            _startup_messages.append((
                "ERROR",
                "[ERROR] Windows multiprocessing guard: cannot locate python.exe. "
                "Gmsh meshing will likely fail (launches a new QGIS instance). "
                "Install Python or add base_exec_prefix/python.exe.",
            ))
        else:
            # _real_python == _sys.executable — same file, so the fallback
            # candidate IS the QGIS launcher.  ProcessPoolExecutor may still
            # work on some setups but gmsh threading may be broken.
            _startup_messages.append((
                "WARNING",
                "[WARNING] Windows multiprocessing guard: base_exec_prefix/python.exe not found; "
                "fallback uses the same binary as sys.executable. "
                "Gmsh meshing may spawn a new QGIS instance instead of meshing inline. "
                "Set General.NumThreads to 1 in meshing options as a workaround.",
            ))
    except Exception as _exc:
        _startup_messages.append((
            "ERROR",
            f"[ERROR] Windows multiprocessing guard raised an exception: {_exc}. "
            "Gmsh meshing will likely fail (launches a new QGIS instance).",
        ))

# ── Lightweight dependency check (logs warnings, never blocks loading) ────
def _check_optional_deps():
    """Warn about missing optional dependencies at plugin load time."""
    import logging
    import warnings
    _log = logging.getLogger("hydra")
    _missing = []
    for _mod, _feat in (
        ("gmsh", "unstructured mesh generation"),
        ("h5py", "HEC-RAS HDF5 export"),
        ("netCDF4", "UGRID NetCDF export"),
        ("matplotlib", "in-plugin plotting"),
    ):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
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
            _os.path.join(PLUGIN_ROOT, "..", "requirements.txt"),
        )
        return False
    return True

try:
    _check_optional_deps()
    _check_required_deps()
except Exception:
    pass  # never block plugin load


# ── Eager internal import check ──────────────────────────────────────────
_ALL_MODULES = [
    "hydra_plugin",
    "swe2d",
    "swe2d.boundary_and_forcing",
    "swe2d.extensions",
    "swe2d.mesh",
    "swe2d.plotting",
    "swe2d.results",
    "swe2d.runtime",
    "swe2d.workbench",
    "swe2d.workbench.controllers",
    "swe2d.workbench.services",
    "swe2d.workbench.views",
]

def _import_all():
    """Eager smoke-test of the plugin's own modules so broken installs are
    loud — but the plugin must still load when the wheel hasn't been
    installed yet, otherwise the first-launch install dialog can never
    fire and we deadlock. Failures are logged at WARNING; the workbench
    call site in hydra_plugin.py will retry the imports after install.
    """
    import logging
    _log = logging.getLogger("hydra")
    _errors = []
    for _mod in _ALL_MODULES:
        try:
            _importlib.import_module(_mod)
        except ImportError as _e:
            _errors.append(f"  {_mod}: {_e}")
    if _errors:
        _log.warning(
            "[HYDRA] %d module(s) failed eager import (likely first launch "
            "before the wheel was installed):\n  %s",
            len(_errors), "\n  ".join(_errors),
        )

_import_all()


def classFactory(iface):
    from .hydra_plugin import HydraQgisPlugin
    return HydraQgisPlugin(iface)
