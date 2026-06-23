import traceback
from qgis.PyQt.QtCore import QTimer
from qgis.PyQt.QtWidgets import QApplication
import qgis.utils


def _print(msg):
    print(f"VALIDATION: {msg}", flush=True)


def run_validation():
    ok = True

    try:
        iface = qgis.utils.iface
        if iface is None:
            _print("FAIL iface unavailable")
            return

        plugin_key = "qgis-backwater-plugin"

        # Ensure plugin is loaded and started.
        try:
            if plugin_key not in qgis.utils.plugins:
                qgis.utils.loadPlugin(plugin_key)
            if plugin_key not in qgis.utils.active_plugins:
                qgis.utils.startPlugin(plugin_key)
        except Exception:
            _print("FAIL could not load/start plugin")
            _print(traceback.format_exc())
            return

        plugin = qgis.utils.plugins.get(plugin_key)
        if plugin is None:
            _print("FAIL plugin instance missing")
            return

        plugin.run()
        dock = getattr(plugin, "dock", None)
        if dock is None:
            _print("FAIL plugin dock missing after run()")
            return

        widget = dock.widget()
        if widget is None:
            _print("FAIL plugin widget missing")
            return

        has_api = all([
            hasattr(widget, "set_dock_host_window"),
            hasattr(widget, "_detach_tab"),
            hasattr(widget, "_reattach_dock_tab"),
            hasattr(widget, "_detached_docks"),
        ])
        if not has_api:
            _print("FAIL detachable-panel API not found on widget")
            return

        # Baseline tab count.
        left_tabs = widget.left_tabs
        before_count = left_tabs.count()
        if before_count < 1:
            _print("FAIL left_tabs has no tabs to detach")
            return

        # Detach first tab and validate dock creation.
        widget._detach_tab(left_tabs, 0)
        detached_count = len(widget._detached_docks)
        if detached_count < 1:
            _print("FAIL no detached dock created")
            return

        detached_dock = next(iter(widget._detached_docks.keys()))
        if not detached_dock.isFloating():
            ok = False
            _print("FAIL detached dock is not floating")

        after_detach_count = left_tabs.count()
        if after_detach_count != before_count - 1:
            ok = False
            _print(f"FAIL tab count after detach expected {before_count - 1}, got {after_detach_count}")

        # Reattach and verify original tab count restored.
        widget._reattach_dock_tab(detached_dock)
        after_reattach_count = left_tabs.count()
        if after_reattach_count != before_count:
            ok = False
            _print(f"FAIL tab count after reattach expected {before_count}, got {after_reattach_count}")

        if ok:
            _print("PASS detachable tab/dock workflow verified")

    except Exception:
        _print("FAIL unhandled exception during validation")
        _print(traceback.format_exc())

    finally:
        QTimer.singleShot(200, QApplication.instance().quit)


QTimer.singleShot(1500, run_validation)
