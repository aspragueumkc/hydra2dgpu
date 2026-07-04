"""DevTools submenu builder.

The DevTools submenu lives under the main HYDRA2DGPU menu. It contains
three actions:

    - Inspect Next Clicked Widget   (the existing one-shot probe)
    - Show Widget Tree              (open the persistent inspector dock)
    - Enter GUI Edit Mode           (placeholder; full edit-mode ships in sprint 2)
"""

from __future__ import annotations

import os
from typing import Callable, List, Optional

from qgis.PyQt.QtCore import QObject
from qgis.PyQt.QtGui import QKeySequence
from qgis.PyQt.QtWidgets import QAction, QMainWindow, QMenu

from swe2d.workbench.devtools.inspector_dock import InspectorDock
from swe2d.workbench.devtools.property_editor import PropertyEditorDialog
from swe2d.workbench.devtools.widget_walker import WidgetNode


def devtools_enabled() -> bool:
    """Return True if the devtools menu should appear this session.

    DevTools is always available in the workbench menu.
    The env-var gate (SWE2D_DEVTOOLS) was removed so the inspector
    and widget tree are always at your fingertips.
    """
    return True


# Paths to the view files we will scan for rename targets.
DEFAULT_VIEW_FILES = (
    "swe2d/workbench/views/model_tab_view.py",
    "swe2d/workbench/views/map_tab_view.py",
    "swe2d/workbench/views/topology_tab_view.py",
    "swe2d/workbench/views/results_controls.py",
    "swe2d/workbench/views/temporal_dock.py",
)


def _view_files_for(plugin_root: str) -> List[str]:
    return [
        os.path.join(plugin_root, rel) for rel in DEFAULT_VIEW_FILES
    ]


class DevToolsController(QObject):
    """Holds the inspector dock + glue for the DevTools submenu actions.

    Created once per workbench session by ``build_devtools_menu``. The
    controller owns the QDockWidget so QGIS can tear it down with the
    workbench.
    """

    def __init__(self, main_window: QMainWindow, plugin_root: str) -> None:
        super().__init__(main_window)
        self._main_window = main_window
        self._plugin_root = plugin_root
        self._view_files = _view_files_for(plugin_root)

        self._inspector_dock: Optional[InspectorDock] = None

        # Wire callbacks so the dock can request renames without us
        # importing the dialog at module load time.
        self._on_edit_requested: Optional[Callable[[WidgetNode], None]] = None

    # ------------------------------------------------------------------
    # Public slots wired to QActions
    # ------------------------------------------------------------------

    def show_widget_tree(self) -> None:
        """Open (or raise) the inspector dock and populate it."""
        if self._inspector_dock is None:
            self._inspector_dock = InspectorDock(
                self._main_window,
                on_select=self._on_node_selected,
                on_edit_requested=self._on_edit_requested_default,
            )
            self._main_window.addDockWidget(
                0x2,  # Qt.RightDockWidgetArea = 0x2
                self._inspector_dock,
            )
        else:
            self._inspector_dock.show()
            self._inspector_dock.raise_()
        # Populate (or refresh) with the main window's widget tree.
        self._inspector_dock.set_root(self._main_window)

    def _on_node_selected(self, node: WidgetNode) -> None:
        """Flash the live widget when the user picks it in the tree."""
        if self._inspector_dock is None:
            return
        live = self._main_window.find(QObject, node.object_name) if node.object_name else None
        if live is not None:
            self._inspector_dock.flash_widget(live)

    def _on_edit_requested_default(self, node: WidgetNode) -> None:
        """Right-click → "Edit properties…" handler."""
        if self._inspector_dock is None:
            return
        # Try to find the live widget so the dialog can flash it after a
        # successful patch.  The dialog does not actually need it; it's a
        # nice-to-have for the dev's feedback loop.
        dlg = PropertyEditorDialog(node, self._view_files, parent=self._main_window)
        dlg.exec_()


def build_devtools_menu(
    parent_menu: QMenu,
    main_window: QMainWindow,
    plugin_root: str,
) -> Optional[QMenu]:
    """Attach a "DevTools" submenu to *parent_menu* and return it.

    Returns ``None`` when ``SWE2D_DEVTOOLS`` is unset (the menu does not
    appear in production deployments).
    """
    if not devtools_enabled():
        return None

    controller = DevToolsController(main_window, plugin_root)

    submenu = QMenu("DevTools", parent_menu)
    submenu.setObjectName("HYDRA2DDevToolsMenu")

    # 1. Existing widget inspector (moved here from the top-level menu).
    inspect_action = QAction("Inspect Next Clicked Widget", submenu)
    inspect_action.setObjectName("HYDRA2DDevToolsInspectAction")
    inspect_action.setShortcut(QKeySequence("Ctrl+Shift+I"))
    def _arm_inspector():
        from swe2d.workbench.dialogs.widget_inspector import arm
        arm()
    inspect_action.triggered.connect(_arm_inspector)
    submenu.addAction(inspect_action)

    submenu.addSeparator()

    # 2. Persistent widget-tree dock.
    tree_action = QAction("Show Widget Tree", submenu)
    tree_action.setObjectName("HYDRA2DDevToolsTreeAction")
    tree_action.triggered.connect(controller.show_widget_tree)
    submenu.addAction(tree_action)

    submenu.addSeparator()

    # 3. Enter GUI edit mode (sprint 2 placeholder).
    edit_action = QAction("Enter GUI Edit Mode… (sprint 2)", submenu)
    edit_action.setObjectName("HYDRA2DDevToolsEditModeAction")
    edit_action.setEnabled(False)
    submenu.addAction(edit_action)

    parent_menu.addMenu(submenu)
    return submenu


__all__ = [
    "build_devtools_menu",
    "devtools_enabled",
    "DevToolsController",
    "DEVTOOLS_ENV_FLAG",
]