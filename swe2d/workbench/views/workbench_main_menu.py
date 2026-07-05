"""Install/remove the workbench's HYDRA2DGPU main-menu.

This menu is owned by the workbench dialog (a View layer in MVP) and
installed/removed alongside the workbench lifecycle. Plugin-level menu
items (Open Workbench, Settings) stay in hydra_plugin.py.

Keyboard shortcuts are intentionally NOT set on the actions here. Instead,
each action is registered with QGIS's :class:`QgsShortcutsManager` under
the section "HYDRA2DGPU" with ``defaultShortcut=''``. This means:

* The action shows up in **Settings → Keyboard Shortcuts** under a
  dedicated "HYDRA2DGPU" section.
* The user assigns their own key combo in that dialog — we never
  force a shortcut that could collide with QGIS built-ins
  (Ctrl+S = Save Project, Ctrl+R / Ctrl+B / Ctrl+O / F5 / etc.).
* QGIS's own shortcut conflict detection handles collisions at
  assignment time.

Previously this code installed ``QShortcut`` instances with
``ApplicationShortcut`` context, which bypassed QGIS's manager
entirely. That module-level `KEYBOARD_SHORTCUTS` table was removed.
"""
from __future__ import annotations

import logging
from typing import Optional

from qgis.PyQt import QtCore, QtWidgets
from qgis.PyQt.QtGui import QKeySequence
from qgis.PyQt.QtWidgets import QAction

logger_wb = logging.getLogger(__name__)

# Section name used when registering actions with QGIS's shortcut manager.
_QGIS_SHORTCUTS_SECTION = "HYDRA2DGPU"

# Tracks the currently-installed workbench menu so we can remove it on close.
_workbench_main_menu: Optional[QtWidgets.QMenu] = None
_workbench_main_menu_actions: list[QAction] = []
_workbench_main_menu_owned: bool = False


def _register_with_qgis_shortcut_manager(action: QAction) -> None:
    """Register ``action`` with QGIS's :class:`QgsShortcutsManager`.

    The action is added to the "HYDRA2DGPU" section with no default
    shortcut, so the user can assign a key combo of their choice from
    **Settings → Keyboard Shortcuts**. Failures (e.g. when running
    headless or in tests) are logged and swallowed — shortcut
    registration is best-effort, not load-bearing.
    """
    try:
        from qgis.gui import QgsGui
        manager = QgsGui.shortcutsManager()
        # ``defaultShortcut=''`` means "user assigns later". Passing a
        # real sequence here would risk colliding with QGIS built-ins.
        manager.registerAction(
            action,
            defaultShortcut="",
            section=_QGIS_SHORTCUTS_SECTION,
        )
    except Exception as exc:
        logger_wb.debug(
            "[workbench_menu] registerAction(%s) failed: %s",
            action.objectName(),
            exc,
        )


def _find_workbench_main_menu(menu_bar: QtWidgets.QMenuBar) -> Optional[QtWidgets.QMenu]:
    """Return the workbench-scoped HYDRA2DGPU menu, if already installed.

    Matches only by objectName (``HYDRA2DBenchMainMenu``), NOT by title, so
    the plugin-level ``HYDRA2DGMainMenu`` is never accidentally reused or
    removed.
    """
    for action in menu_bar.actions():
        menu = action.menu()
        if menu is None:
            continue
        try:
            if menu.objectName() == "HYDRA2DBenchMainMenu":
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
    def add_action(object_name, text, callback):
        """Add an action to the menu and register it with the QGIS
        shortcut manager (no default shortcut — user assigns via
        Settings → Keyboard Shortcuts).
        """
        act = QAction(text, menu)
        act.setObjectName(object_name)
        act.triggered.connect(callback)
        menu.addAction(act)
        _workbench_main_menu_actions.append(act)
        # Register so the user can assign a key combo in QGIS's
        # Keyboard Shortcuts settings dialog. No default sequence.
        _register_with_qgis_shortcut_manager(act)
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
    add_action(
        "HYDRA2DMenuCreateTemplateAction",
        "Create Topology Template Layers…",
        lambda: dlg._topology_controller.create_topology_template_layers(),
    )
    menu.addSeparator()

    # ── Simulation & I/O actions ────────────────────────────────────
    add_action(
        "HYDRA2DMenuRunLastAction",
        "Run Last Simulation",
        lambda: dlg._controller.on_run(),
    )
    add_action(
        "HYDRA2DMenuCancelRunAction",
        "Cancel Running Simulation",
        lambda: dlg._controller.on_cancel(),
    )
    add_action(
        "HYDRA2DMenuBatchSimAction",
        "Batch Simulation…",
        lambda: dlg._controller.open_batch_simulation_dialog(),
    )
    add_action(
        "HYDRA2DMenuSaveConfigAction",
        "Save Simulation Config to GeoPackage…",
        lambda: dlg._controller.on_save_simulation_config(),
    )
    add_action(
        "HYDRA2DMenuLoadConfigAction",
        "Load Simulation Config from GeoPackage…",
        lambda: dlg._controller.on_load_simulation_config(),
    )
    add_action(
        "HYDRA2DMenuOpenRunLogAction",
        "Open Run Log",
        lambda: dlg._controller.open_run_log_viewer(),
    )
    add_action(
        "HYDRA2DMenuOpenGpkgExplorerAction",
        "Open GeoPackage Explorer",
        lambda: dlg._topology_controller.open_model_gpkg_explorer(),
    )
    add_action(
        "HYDRA2DMenuRefreshResultsAction",
        "Refresh Results",
        lambda: dlg._on_results_refresh(),
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