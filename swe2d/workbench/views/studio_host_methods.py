"""Dock management, host controls, and launch functions for SWE2D Studio."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from qgis.PyQt import QtCore, QtGui, QtWidgets


logger_wb = logging.getLogger(__name__)

_SWE2D_WORKBENCH_STUDIO_WINDOWS: List[QtWidgets.QDialog] = []
_studio_active_dialog: Optional["SWE2DWorkbenchStudioDialog"] = None


def close_workbench_studio(iface=None) -> None:
    """Close the active workbench studio dock (no-op if already closed).

    This is a public API callable from outside the workbench module
    (e.g. from the plugin menu) to close the workbench without
    disabling the plugin itself.

    Persists dock layout + window geometry to QSettings *before*
    tearing the docks down, so the user's panel arrangement is
    preserved across QGIS sessions.
    """
    global _studio_active_dialog
    if _studio_active_dialog is None:
        # Still mark was_open=False so we don't auto-relaunch on next startup.
        _persist_workbench_was_open(False)
        return
    iface = _resolve_workbench_iface(None, iface)
    try:
        _capture_and_persist_window_state(iface)
    except Exception as e:
        logger_wb.warning("[close] save window state failed: %s", e)
    _remove_workbench_studio_dock(iface, dlg=_studio_active_dialog)
    _studio_active_dialog = None
    _persist_workbench_was_open(False)


def _normalize_workbench_host_mode(host_mode: object) -> str:
    """Normalize host mode to 'dock' or 'window'."""
    mode_txt = str(host_mode or "window").strip().lower()
    return "dock" if mode_txt in {"dock", "docked", "panel"} else "window"


def _resolve_workbench_iface(parent, iface):
    """Resolve the QGIS interface object from parent or qgis.utils."""
    if iface is None and parent is not None:
        if hasattr(parent, "_get_qgis_iface") and callable(getattr(parent, "_get_qgis_iface")):
            try:
                iface = parent._get_qgis_iface()
            except Exception as e:
                logger_wb.warning("[ERROR] resolve iface failed: %s", e)
                iface = None
        if iface is None and hasattr(parent, "iface"):
            try:
                iface = getattr(parent, "iface")
            except Exception as e:
                logger_wb.warning("[ERROR] resolve iface failed: %s", e)
                iface = None
    if iface is None:
        try:
            import qgis.utils as _qutils

            iface = getattr(_qutils, "iface", None)
        except Exception as e:
            logger_wb.warning("[ERROR] resolve iface failed: %s", e)
            iface = None
    return iface


def _close_dialog_windows(window_store: List[QtWidgets.QDialog]) -> None:
    """Close all dialogs in the given window store list."""
    while window_store:
        dlg = window_store.pop()
        try:
            dlg.close()
        except Exception as e:
            logger_wb.warning("[ERROR] close dialog failed: %s", e)


def _close_workbench_studio_windows() -> None:
    """Close all workbench studio window instances."""
    _close_dialog_windows(_SWE2D_WORKBENCH_STUDIO_WINDOWS)


def _remove_workbench_dock_instance(dock, iface_obj):
    """Remove a single dock widget from QGIS and clean up."""
    if dock is None:
        return None
    try:
        widget = dock.widget()
        if widget is not None:
            try:
                widget.blockSignals(True)
            except (RuntimeError, AttributeError):
                logger_wb.exception("[ERROR] widget.blockSignals failed during dock removal")
            try:
                widget.close()
            except (RuntimeError, AttributeError):
                logger_wb.exception("[ERROR] widget.close() failed during dock removal")
    except (RuntimeError, AttributeError):
        logger_wb.exception("[ERROR] dock.widget() failed during dock removal")
    try:
        if iface_obj is not None and hasattr(iface_obj, "removeDockWidget"):
            iface_obj.removeDockWidget(dock)
    except (RuntimeError, AttributeError):
        logger_wb.exception("[ERROR] removeDockWidget failed during dock removal")
    try:
        dock.deleteLater()
    except (RuntimeError, AttributeError):
        logger_wb.exception("[ERROR] deleteLater failed during dock removal")
    return None


def _remove_workbench_studio_dock(iface_obj, dlg=None) -> None:
    """Remove all Studio docks from the QGIS host window and clean up."""
    seen = set()
    if dlg is not None:
        for name, comp in list(dlg._state.studio_components.items()):
            dock = comp.dock
            if dock is None:
                continue
            key = id(dock)
            if key in seen:
                continue
            seen.add(key)
            _remove_workbench_dock_instance(dock, iface_obj)

    if dlg is not None:
        try:
            dlg.blockSignals(True)
        except (RuntimeError, AttributeError):
            logger_wb.exception("[ERROR] studio dialog blockSignals failed")
        try:
            dlg.close()
        except (RuntimeError, AttributeError):
            logger_wb.exception("[ERROR] studio dialog close failed")
        try:
            dlg.deleteLater()
        except (RuntimeError, AttributeError):
            logger_wb.exception("[ERROR] studio dialog deleteLater failed")
    _clear_studio_host_controls(iface_obj)

    # Tear down the workbench-scoped main menu (added in install path).
    try:
        from swe2d.workbench.views.workbench_main_menu import remove_workbench_main_menu
        remove_workbench_main_menu(iface_obj)
    except Exception as e:
        logger_wb.warning("[studio_host_methods] remove_workbench_main_menu failed: %s", e)


def _studio_host_main_window(iface_obj, fallback_parent=None):
    """Get the QGIS main window, falling back to fallback_parent."""
    host_window = None
    if iface_obj is not None and hasattr(iface_obj, "mainWindow"):
        try:
            host_window = iface_obj.mainWindow()
        except Exception as e:
            logger_wb.warning("[ERROR] mainWindow() retrieval failed: %s", e)
            host_window = None
    if host_window is None:
        host_window = fallback_parent
    return host_window


def _clear_studio_host_controls(iface_obj, fallback_parent=None) -> None:
    """Remove the Studio view-mode combo from the QGIS host window.

    The HYDRA2DGPU toolbar and menu are owned by hydra_plugin.py, not the
    workbench. The workbench only manages the view-mode combo corner widget.
    """
    pass


def _install_studio_host_controls(
    iface_obj,
    dlg,
    fallback_parent=None,
    component_docks: Optional[Dict[str, QtWidgets.QDockWidget]] = None,
) -> None:
    """Install the HYDRA2D menu and view-mode combo into the QGIS host window."""
    global _SWE2D_STUDIO_HOST_TOOLBAR, _SWE2D_STUDIO_HOST_MENU
    host_window = _studio_host_main_window(iface_obj, fallback_parent)
    if host_window is None:
        return
    _clear_studio_host_controls(iface_obj, fallback_parent)
    component_docks = dict(component_docks or {})

    host_view_combo = QtWidgets.QComboBox(host_window)
    host_view_combo.addItems(["Mesh", "Depth", "Velocity magnitude",
                               "Time-Series", "Profile", "Structures", "Network"])
    try:
        source_idx = int(getattr(dlg, "view_mode_combo", host_view_combo).currentIndex())
        host_view_combo.setCurrentIndex(max(0, min(source_idx, host_view_combo.count() - 1)))
    except Exception as e:
        logger_wb.warning("[ERROR] host view combo init failed: %s", e)
    host_view_combo.currentIndexChanged.connect(
        lambda idx: dlg.view_mode_combo.setCurrentIndex(idx)
        if hasattr(dlg, "view_mode_combo") and dlg.view_mode_combo is not None
        else None
    )
    try:
        menu_bar = host_window.menuBar()
        if menu_bar is not None:
            menu_bar.setCornerWidget(host_view_combo, QtCore.Qt.TopRightCorner)
    except Exception as e:
        logger_wb.warning("[ERROR] view combo to menuBar corner failed: %s", e)

    # Install the workbench-scoped HYDRA2DGPU main menu.
    try:
        from swe2d.workbench.views.workbench_main_menu import install_workbench_main_menu
        install_workbench_main_menu(dlg=dlg, iface=iface_obj)
    except Exception as e:
        logger_wb.warning("[ERROR] install_workbench_main_menu failed: %s", e)


def launch_swe2d_workbench_studio(parent=None, iface=None, host_mode: str = "dock"):
    """Launch the SWE2D workbench Studio dialog (docked or windowed mode).

    Records ``was_open=True`` in QSettings so the plugin knows to
    re-open the workbench on the next QGIS launch (when the user
    has also enabled open-on-startup).  Per-dock layout restoration
    happens inside ``WorkbenchDialogBuilder._build_component`` once
    every dock has been registered with ``iface.mainWindow()`` —
    see _schedule_restored_dock_layout.
    """
    from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog

    global _studio_active_dialog

    iface = _resolve_workbench_iface(parent, iface)

    mode = "dock"

    if mode == "dock":
        _close_workbench_studio_windows()
        # Clean up previous instance before creating a new one
        if _studio_active_dialog is not None:
            # Capture layout from the outgoing instance before tearing down.
            try:
                _capture_and_persist_window_state(iface)
            except Exception as e:
                logger_wb.warning("[launch] save window state failed: %s", e)
            _remove_workbench_studio_dock(iface, dlg=_studio_active_dialog)
            _studio_active_dialog = None

        host_window = None
        if iface is not None and hasattr(iface, "mainWindow"):
            try:
                host_window = iface.mainWindow()
            except Exception as e:
                logger_wb.warning("[launch] mainWindow failed: %s", e)
                host_window = None
        if host_window is None:
            host_window = parent

        dlg = SWE2DWorkbenchStudioDialog(parent=None, iface=iface)
        dlg._swe2d_workbench_host_mode = mode
        try:
            dlg._restore_project_workbench_state()
            dlg._workbench_state_restored_on_show = True
        except Exception as e:
            logger_wb.warning("[launch] restore state failed: %s", e)

        component_docks = {
            name: comp.dock
            for name, comp in dlg._state.studio_components.items()
            if comp.dock is not None
        }
        _install_studio_host_controls(iface, dlg, host_window, component_docks=component_docks)
        try:
            dlg._studio_update_status()
        except Exception as e:
            logger_wb.warning("[launch] update status failed: %s", e)
            pass

        # Schedule the dock-layout restore on the next tick so every
        # dock has been registered with iface.mainWindow().
        if host_window is not None:
            _schedule_restored_dock_layout(host_window)

        _studio_active_dialog = dlg
        _persist_workbench_was_open(True)
        return dlg

    _remove_workbench_studio_dock(iface)

    for existing in list(_SWE2D_WORKBENCH_STUDIO_WINDOWS):
        try:
            if existing.isVisible():
                existing.show()
                existing.raise_()
                existing.activateWindow()
                return existing
        except Exception as e:
            logger_wb.warning("[launch] existing window failed: %s", e)

    dlg = SWE2DWorkbenchStudioDialog(parent, iface=iface)
    _install_studio_host_controls(iface, dlg, parent)

    def _cleanup():
        """Remove dialog from window store and clear host controls."""
        try:
            _SWE2D_WORKBENCH_STUDIO_WINDOWS.remove(dlg)
        except ValueError:
            logger_wb.warning("Unexpected ValueError silently caught — review this handler", exc_info=True)
        _clear_studio_host_controls(iface, parent)

    _SWE2D_WORKBENCH_STUDIO_WINDOWS.append(dlg)
    dlg.finished.connect(_cleanup)
    dlg.show()
    dlg.raise_()
    dlg.activateWindow()
    return dlg


# ----------------------------------------------------------------------
# Workbench session persistence (QMainWindow.saveState / restoreState)
# ----------------------------------------------------------------------

def _workbench_qsettings():
    """Return the shared QSettings for HYDRA2DGPU.

    Centralized so all four hook points (launch, close, unload, settings UI)
    write to the same file.  Returns None if Qt is not yet importable
    (very early initGui call before ``QApplication`` is up).
    """
    try:
        from qgis.PyQt.QtCore import QSettings
        return QSettings("HYDRA2DGPU", "HYDRA2DGPU")
    except Exception as e:  # pragma: no cover — defensive
        logger_wb.warning("[persistence] QSettings unavailable: %s", e)
        return None


def _capture_and_persist_window_state(iface_obj=None) -> bool:
    """Save the QGIS main window's dock layout to QSettings.

    Called by ``close_workbench_studio`` (user closed via menu) and
    from ``hydra_plugin.unload`` (user closed QGIS without closing
    the workbench first).
    """
    from swe2d.workbench import persistence

    s = _workbench_qsettings()
    if s is None:
        return False
    mw = _studio_host_main_window(iface_obj)
    if mw is None:
        return False
    return persistence.save_window_state(s, mw)


def _persist_workbench_was_open(value: bool) -> None:
    """Mark whether the workbench is currently open."""
    from swe2d.workbench import persistence

    s = _workbench_qsettings()
    if s is None:
        return
    persistence.save_was_open(s, value)


def _schedule_restored_dock_layout(host_window) -> None:
    """Restore the persisted dock layout on a deferred single-shot timer.

    ``QMainWindow.restoreState`` must run *after* every dock it
    references has been added with a stable ``objectName()``.
    Since each dock is added in its own ``_build_component`` call,
    we wait until the next tick of the event loop.
    """
    try:
        QtCore.QTimer.singleShot(0, lambda: _apply_restored_dock_layout(host_window))
    except Exception as e:
        logger_wb.warning("[persistence] schedule layout restore failed: %s", e)


def _apply_restored_dock_layout(host_window) -> None:
    """Perform the actual restore using the persistence helper."""
    from swe2d.workbench import persistence

    s = _workbench_qsettings()
    if s is None or host_window is None:
        return
    try:
        persistence.restore_window_state(s, host_window)
    except Exception as e:
        logger_wb.warning("[persistence] restore_window_state raised: %s", e)
