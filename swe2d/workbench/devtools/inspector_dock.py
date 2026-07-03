"""Inspector dock — persistent QTreeWidget view of the workbench widget tree.

Behaviour:
    - On ``set_root(widget)``, walks the widget tree and populates the tree.
    - Click a node -> the property editor receives a selection callback.
    - Right-click -> "Edit properties…" opens the rename dialog (sprint 1).
    - The tree is read-only; everything is done via the property editor.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from qgis.PyQt.QtCore import Qt, QTimer
from qgis.PyQt.QtGui import QColor, QBrush
from qgis.PyQt.QtWidgets import (
    QApplication,
    QDockWidget,
    QMenu,
    QTreeWidget,
    QTreeWidgetItem,
    QWidget,
)

from swe2d.workbench.devtools.widget_walker import (
    WidgetNode,
    build_child_index,
    walk_widget_tree,
)


class InspectorDock(QDockWidget):
    """A read-only widget-tree browser.

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget for Qt ownership.
    on_select : callable, optional
        ``on_select(node: WidgetNode) -> None`` called when the user clicks
        a tree row.
    on_edit_requested : callable, optional
        ``on_edit_requested(node: WidgetNode) -> None`` called when the
        user picks "Edit properties…" from the right-click menu.
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        on_select: Optional[Callable[[WidgetNode], None]] = None,
        on_edit_requested: Optional[Callable[[WidgetNode], None]] = None,
    ) -> None:
        super().__init__("Hydra Designer — Widget Tree", parent)
        self.setObjectName("HydraDesignerInspectorDock")
        self.setFeatures(
            QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
            | QDockWidget.DockWidgetClosable
        )

        self._on_select = on_select
        self._on_edit_requested = on_edit_requested

        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Widget", "Class"])
        self._tree.setColumnWidth(0, 360)
        self._tree.setColumnWidth(1, 160)
        self._tree.setUniformRowHeights(True)
        self._tree.itemSelectionChanged.connect(self._on_item_selected)
        self._tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._show_context_menu)
        self.setWidget(self._tree)

        self._nodes: List[WidgetNode] = []
        self._child_index: Dict[int, List[WidgetNode]] = {}
        self._row_to_node: Dict[QTreeWidgetItem, WidgetNode] = {}

        # Highlight timer for "flash selected widget in the live UI".
        self._flash_timer = QTimer(self)
        self._flash_timer.setSingleShot(True)
        self._flash_timer.timeout.connect(self._clear_flash)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_root(self, root_widget: Optional[QWidget]) -> None:
        """Re-walk the tree starting at *root_widget*."""
        self._tree.clear()
        self._row_to_node.clear()
        if root_widget is None:
            self._nodes = []
            self._child_index = {}
            return
        self._nodes = walk_widget_tree(root_widget)
        self._child_index = build_child_index(self._nodes)
        # Build a tree item per node, top-down.
        items_by_id: Dict[int, QTreeWidgetItem] = {}
        for node in self._nodes:
            item = QTreeWidgetItem([node.display_label(), node.class_name])
            item.setData(0, Qt.UserRole, node.widget_id)
            self._row_to_node[item] = node
            items_by_id[node.widget_id] = item
        # Now wire parents: items whose parent is the root get added as top-level;
        # otherwise they get parented to their parent's item.
        # findChildren returns ALL descendants, so the parent_id of any node
        # (other than the root) is the *immediate* parent in our flat list.
        # To honour that, walk the nodes in tree order and attach each to
        # its parent's tree item.
        for node in self._nodes:
            item = items_by_id[node.widget_id]
            if node.parent_id is None or node.parent_id not in items_by_id:
                self._tree.addTopLevelItem(item)
            else:
                parent_item = items_by_id[node.parent_id]
                parent_item.addChild(item)
        self._tree.expandAll()

    def selected_node(self) -> Optional[WidgetNode]:
        """Return the node currently selected in the tree, or ``None``."""
        items = self._tree.selectedItems()
        if not items:
            return None
        return self._row_to_node.get(items[0])

    def find_node_by_object_name(self, object_name: str) -> Optional[WidgetNode]:
        """Locate the first node in the cache with the given objectName."""
        for node in self._nodes:
            if node.object_name == object_name:
                return node
        return None

    def flash_widget(self, widget: QWidget) -> None:
        """Briefly highlight *widget* in the live UI (best-effort)."""
        if widget is None:
            return
        try:
            widget.setStyleSheet("border: 2px solid #ff5050;")
        except (RuntimeError, TypeError):
            return
        self._flash_timer.start(900)

    # ------------------------------------------------------------------
    # Internal slots
    # ------------------------------------------------------------------

    def _on_item_selected(self) -> None:
        node = self.selected_node()
        if node is None:
            return
        if self._on_select is not None:
            self._on_select(node)

    def _show_context_menu(self, pos) -> None:
        item = self._tree.itemAt(pos)
        if item is None:
            return
        node = self._row_to_node.get(item)
        if node is None:
            return
        menu = QMenu(self._tree)
        edit_action = menu.addAction("Edit properties…")
        chosen = menu.exec_(self._tree.viewport().mapToGlobal(pos))
        if chosen is edit_action and self._on_edit_requested is not None:
            self._on_edit_requested(node)

    def _clear_flash(self) -> None:
        """Best-effort clear of any flash stylesheet set via ``flash_widget``."""
        # We don't know which widget was flashed; iterate all top-level widgets.
        for widget in QApplication.topLevelWidgets():
            try:
                widget.setStyleSheet("")
            except (RuntimeError, TypeError):
                continue


__all__ = ["InspectorDock"]