"""QGIS plugin entry point for Backwater

QGIS expects a `classFactory(iface)` function that returns the plugin
instance. This file exposes that function.
"""
def classFactory(iface):
    from .backwater_plugin import BackwaterQgisPlugin
    return BackwaterQgisPlugin(iface)
