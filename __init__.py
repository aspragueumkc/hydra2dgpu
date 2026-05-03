"""QGIS plugin entry point for Backwater

QGIS expects a `classFactory(iface)` function that returns the plugin
instance. This file exposes that function.
"""
import os as _os
import sys as _sys

# Add the compiled C++ extension directory so backwater_tqmesh and
# backwater_swe2d can be imported from anywhere inside the plugin.
_plugin_dir = _os.path.dirname(_os.path.abspath(__file__))
_build_dir = _os.path.join(_plugin_dir, "build")
if _build_dir not in _sys.path:
    _sys.path.insert(0, _build_dir)
if _plugin_dir not in _sys.path:
    _sys.path.insert(0, _plugin_dir)


def classFactory(iface):
    from .backwater_plugin import BackwaterQgisPlugin
    return BackwaterQgisPlugin(iface)
