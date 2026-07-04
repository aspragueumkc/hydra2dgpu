"""Install/remove the workbench's HYDRA2DGPU main-menu.

This menu is owned by the workbench dialog (a View layer in MVP) and
installed/removed alongside the workbench lifecycle. Plugin-level menu
items (Open Workbench, Settings) stay in hydra_plugin.py.
"""
from __future__ import annotations

import logging
from typing import Optional

from qgis.PyQt import QtCore, QtWidgets
from qgis.PyQt.QtGui import QAction, QKeySequence

logger_wb = logging.getLogger(__name__)

# Tracks the currently-installed workbench menu so we can remove it on close.
_workbench_main_menu: Optional[QtWidgets.QMenu] = None
_workbench_main_menu_actions: list[QAction] = []
_workbench_main_menu_owned: bool = False


def _find_workbench_main_menu(menu_bar: QtWidgets.QMenuBar) -> Optional[QtWidgets.QMenu]:
    """Return an existing HYDRA2DGPU top-level menu on the menu bar, if any."""
    for action in menu_bar.actions():
        menu = action.menu()
        if menu is None:
            continue
        try:
            if menu.objectName() == "HYDRA2DBenchMainMenu":
                return menu
        except Exception:
            pass
        try:
            if str(menu.title()).replace("&", "").strip() == "HYDRA2DGPU":
                return menu
        except Exception:
            pass
    return None


def install_workbench_main_menu(dlg, iface) -> Optional[QtWidgets.QMenu]:
    """Build and install the workbench-scoped HYDRA2DGPU main menu.

    Returns the installed menu, or None if no QGIS main window is available.
    """
    global _workbench_main_menu, _workbench_main_menu_actions, _workbench_main_menu_owned

    menu_bar = _resolve_menu_bar(iface)
    if menu_bar is None:
        logger_wb.warning("[workbench_menu] no menu bar available")
        return None

    # Re-use an existing menu if present (idempotent install).
    menu = _find_workbench_main_menu(menu_bar)
    if menu is None:
        menu = QtWidgets.QMenu("HYDRA2DGPU", menu_bar)
        menu.setObjectName("HYDRA2DBenchMainMenu")
        menu_bar.addMenu(menu)
        _workbench_main_menu_owned = True
    else:
        _workbench_main_menu_owned = False

    _workbench_main_menu = menu

    # Wipe any stale actions left over from a previous install so this
    # function is idempotent.
    for action in list(menu.actions()):
        try:
            action.triggered.disconnect()
        except (TypeError, RuntimeError):
            pass
        menu.removeAction(action)
        try:
            action.deleteLater()
        except (RuntimeError, AttributeError):
            pass
    _workbench_main_menu_actions = []

    # ── Persistent helper ───────────────────────────────────────────
    def add_action(object_name, text, callback, shortcut=None):
        act = QAction(text, menu)
        act.setObjectName(object_name)
        act.triggered.connect(callback)
        if shortcut:
            act.setShortcut(QKeySequence(shortcut))
        menu.addAction(act)
        _workbench_main_menu_actions.append(act)
        return act

    # ── GeoPackage actions ──────────────────────────────────────────
    add_action(
        "HYDRA2DMenuCreateGpkgAction",
        "Create 2D Model GeoPackage…",
        lambda: dlg._mesh_controller.create_2d_model_geopackage(),
    )
    add_action(
        "HYDRA2DMenuLoadGpkgAction",
        "Load 2D Model GeoPackage…",
        lambda: dlg._load_2d_model_geopackage(),
    )
    menu.addSeparator()

    # ── Simulation & I/O actions ────────────────────────────────────
    add_action(
        "HYDRA2DMenuRunLastAction",
        "Run Last Simulation",
        lambda: dlg._controller.on_run(),
        "Ctrl+R",
    )
    add_action(
        "HYDRA2DMenuBatchSimAction",
        "Batch Simulation…",
        lambda: dlg._controller.open_batch_simulation_dialog(),
        "Ctrl+B",
    )
    add_action(
        "HYDRA2DMenuOpenRunLogAction",
        "Open Run Log",
        lambda: dlg._mesh_controller.open_run_log_viewer(),
    )
    add_action(
        "HYDRA2DMenuOpenGpkgExplorerAction",
        "Open GeoPackage Explorer",
        lambda: dlg._topology_controller.open_model_gpkg_explorer(),
    )
    menu.addSeparator()

    # ── Results export ──────────────────────────────────────────────
    add_action(
        "HYDRA2DMenuExportGeoTIFFAction",
        "Export Current Results as GeoTIFF…",
        lambda: dlg._overlay_controller.export_high_perf_overlay_to_geotiff(),
    )
    menu.addSeparator()

    # ── Help ────────────────────────────────────────────────────────
    add_action(
        "HYDRA2DMenuHelpAction",
        "Help → Documentation Hub",
        lambda: dlg._open_documentation_hub(),
    )

    return menu


def remove_workbench_main_menu(iface) -> None:
    """Tear down the workbench main menu and free actions."""
    global _workbench_main_menu, _workbench_main_menu_actions, _workbench_main_menu_owned

    menu = _workbench_main_menu
    if menu is None:
        return

    for action in _workbench_main_menu_actions:
        try:
            action.triggered.disconnect()
        except (TypeError, RuntimeError):
            pass
        try:
            menu.removeAction(action)
        except Exception:
            pass
        try:
            action.deleteLater()
        except (RuntimeError, AttributeError):
            pass
    _workbench_main_menu_actions = []

    if _workbench_main_menu_owned:
        menu_bar = _resolve_menu_bar(iface)
        if menu_bar is not None:
            try:
                menu_bar.removeAction(menu.menuAction())
            except Exception:
                pass
        try:
            menu.deleteLater()
        except (RuntimeError, AttributeError):
            pass

    _workbench_main_menu = None
    _workbench_main_menu_owned = False


def _resolve_menu_bar(iface) -> Optional[QtWidgets.QMenuBar]:
    """Resolve the QGIS menu bar from iface, with defensive error handling."""
    if iface is None:
        return None
    if not hasattr(iface, "mainWindow"):
        return None
    try:
        host_window = iface.mainWindow()
    except Exception as e:
        logger_wb.warning("[workbench_menu] mainWindow failed: %s", e)
        return None
    if host_window is None:
        return None
    try:
        return host_window.menuBar()
    except Exception as e:
        logger_wb.warning("[workbench_menu] menuBar failed: %s", e)
        return None