"""Profile controller — orchestrates the Network Profile Viewer launch.

Reads active GPKG path + run_id + iface from the View through typed
protocol methods. The controller itself does no widget I/O and no numpy
math; it just wires a dialog.

Architecture: MVP controller. The View (SWE2DWorkbenchStudioDialog)
exposes ``get_active_gpkg_path()``, ``get_active_run_id()``,
``get_qgis_iface()`` and ``_log()`` via the ``WorkbenchMainViewProtocol``
in ``swe2d.workbench.views.view_protocols``. This controller depends on
that protocol only — no widget attribute access.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from qgis.PyQt import QtWidgets

logger = logging.getLogger(__name__)


class ProfileController:
    """MVP controller for the SWMM-style Network Profile Viewer.

    The controller is intentionally minimal: it reads the active model
    state from the view (or prompts the user for a GPKG file), then
    launches the standalone :class:`NetworkProfileDialog`. All graph
    traversal, chain assembly, and rendering happen inside the dialog
    (which in turn calls the pure-Python services in
    ``swe2d.workbench.services``).
    """

    def __init__(self, view: Any):
        self._view = view

    def open_network_profile_viewer(self) -> None:
        """Launch the Network Profile Viewer.

        Resolution order for the GeoPackage path:

        1. The active 2D model GPKG (from ``get_active_gpkg_path()``).
        2. A user-chosen file via the standard QGIS file dialog
           (matches the GPKG Explorer and Run Log viewer UX).

        Any unexpected error is surfaced via a modal ``QMessageBox``
        AND logged — never silently swallowed.
        """
        try:
            gpkg_path = self._resolve_gpkg_path()
            if not gpkg_path:
                return  # user cancelled the file picker

            run_id: Optional[str] = self._view.get_active_run_id() or None
            iface = self._view.get_qgis_iface()

            from swe2d.workbench.dialogs.network_profile_dialog import (
                NetworkProfileDialog,
            )

            logger.info(
                "[NetworkProfile] launching dialog gpkg=%s run_id=%s",
                gpkg_path, run_id,
            )
            dlg = NetworkProfileDialog(
                gpkg_path=str(gpkg_path),
                run_id=run_id,
                qgis_iface=iface,
                parent=self._view,
            )
            dlg.exec()
        except Exception as exc:  # noqa: BLE001 - we explicitly want to surface
            logger.exception("[NetworkProfile] failed to open viewer")
            self._show_error(exc)

    def _resolve_gpkg_path(self) -> Optional[str]:
        """Return the active GPKG path, or prompt the user to pick one.

        Returns ``None`` if the user cancels the file picker.
        """
        active = str(self._view.get_active_gpkg_path() or "").strip()
        if active and os.path.exists(active):
            return active

        if active:
            # Active path is set but the file is missing — warn but still prompt.
            self._view._log(
                f"[NetworkProfile] Active model GPKG missing on disk: {active}. "
                "Choose another file."
            )
        else:
            self._view._log(
                "[NetworkProfile] No active 2D model GPKG. "
                "Choose a GeoPackage to inspect."
            )

        # Fall back to the standard file picker (mirrors the GPKG Explorer).
        picked = self._view.get_open_file_name(
            "Select GeoPackage to profile", "",
            "GeoPackage (*.gpkg);;All Files (*)",
        )
        picked = str(picked or "").strip()
        if not picked:
            return None
        if not os.path.exists(picked):
            QtWidgets.QMessageBox.warning(
                self._view,
                "Network Profile Viewer",
                f"Selected GeoPackage not found:\n{picked}",
            )
            return None
        return picked

    def _show_error(self, exc: Exception) -> None:
        """Log + show a modal message box so silent failures are impossible."""
        try:
            self._view._log(f"[NetworkProfile] ERROR: {exc}")
        except Exception:
            pass
        QtWidgets.QMessageBox.critical(
            self._view,
            "Network Profile Viewer — Error",
            f"Failed to open Network Profile Viewer:\n\n{exc}",
        )