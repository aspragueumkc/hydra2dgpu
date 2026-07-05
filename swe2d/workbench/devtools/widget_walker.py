"""Widget walker — convert a QWidget tree to a flat list of lightweight nodes.

The walker is intentionally pure-Python + Qt-only. It does not import
``qgis.core`` or any workbench controller/service. It can therefore be
unit-tested with a bare ``QApplication`` and no QGIS instance.

Why a dataclass and not the live QWidget?
    - Selections in the inspector tree need a stable identity even if the
      underlying widget is destroyed (e.g. on tab rebuild). The node stores
      ``objectName`` + ``className`` + ``id()`` so the property editor can
      look the live widget back up when needed.
    - The walker is called from a non-GUI thread by the AST scanner during
      patch validation. Returning plain dataclasses avoids touching the
      Qt object system off the GUI thread.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List, Optional

from qgis.PyQt.QtCore import QObject
from qgis.PyQt.QtWidgets import QWidget


# Widget classes that are too noisy to show in the inspector tree.
# They typically have hundreds of internal children.
_NOISY_TYPES = frozenset({
    "QStackedLayout",
    "QGridLayout",
    "QFormLayout",
    "QBoxLayout",
    "QLayout",
})


@dataclass(frozen=True)
class WidgetNode:
    """A single widget in the inspected tree.

    Attributes
    ----------
    object_name : str
        The ``objectName()`` of the widget. May be empty for anonymous widgets.
    class_name : str
        The Python class name, e.g. ``"QDoubleSpinBox"``.
    widget_id : int
        ``id(widget)`` at the time of capture. Use to look up the live widget
        if it still exists; check ``id()`` against a fresh capture first.
    parent_id : Optional[int]
        ``id()`` of the parent widget, or ``None`` for the root.
    text : str
        Convenience: ``windowTitle()`` for windows/docks, ``text()`` for
        buttons/labels, the first 80 chars of ``toolTip()`` otherwise.
    depth : int
        0 for the root, +1 per nesting level.
    """
    object_name: str
    class_name: str
    widget_id: int
    parent_id: Optional[int]
    text: str
    depth: int

    def display_label(self) -> str:
        """Return a human-friendly label for tree rendering."""
        oname = f' "{self.object_name}"' if self.object_name else ""
        if self.text:
            return f"{self.class_name}{oname}  —  {self.text}"
        return f"{self.class_name}{oname}"


def _summarise_text(widget: QWidget) -> str:
    """Pick the most useful short text description for *widget*."""
    for getter_name in ("windowTitle", "text", "title", "placeholderText"):
        getter = getattr(widget, getter_name, None)
        if getter is None:
            continue
        try:
            value = getter()
        except (TypeError, RuntimeError):
            continue
        if value:
            text = str(value).strip().replace("\n", " ")
            if len(text) > 80:
                text = text[:77] + "..."
            return text
    tip = getattr(widget, "toolTip", lambda: "")() or ""
    if tip:
        tip = tip.strip().replace("\n", " ")
        return tip[:77] + "..." if len(tip) > 80 else tip
    return ""


def walk_widget_tree(root: QWidget) -> List[WidgetNode]:
    """Walk *root* and return a flat list of ``WidgetNode`` (depth-first).

    The first node is always the root itself. Layouts are skipped — they
    are not widgets, and including them clutters the tree.

    The walker catches ``RuntimeError`` per child because some Qt objects
    raise when accessed from a thread they were not created on.
    """
    if root is None:
        return []
    nodes: List[WidgetNode] = []
    _walk_recursive(root, parent_id=None, depth=0, out=nodes)
    return nodes


def _walk_recursive(
    widget: QWidget,
    parent_id: Optional[int],
    depth: int,
    out: List[WidgetNode],
) -> None:
    try:
        object_name = widget.objectName() or ""
    except RuntimeError:
        return
    try:
        class_name = type(widget).__name__
    except RuntimeError:
        class_name = "<destroyed>"
    try:
        widget_id = id(widget)
    except RuntimeError:
        return
    out.append(
        WidgetNode(
            object_name=object_name,
            class_name=class_name,
            widget_id=widget_id,
            parent_id=parent_id,
            text=_summarise_text(widget),
            depth=depth,
        )
    )
    # Skip noisy / non-widget children.
    if class_name in _NOISY_TYPES:
        return
    try:
        children: List[QWidget] = widget.findChildren(QWidget, "", )  # type: ignore[arg-type]
    except (RuntimeError, TypeError):
        children = []
    # findChildren is recursive and unsorted — sort by id for stable display.
    children.sort(key=lambda c: id(c))
    for child in children:
        # Skip widgets that were already captured (e.g. shared popups).
        if any(n.widget_id == id(child) for n in out):
            continue
        _walk_recursive(child, parent_id=widget_id, depth=depth + 1, out=out)


def find_node_by_object_name(
    nodes: List[WidgetNode], object_name: str
) -> Optional[WidgetNode]:
    """Return the first node whose ``objectName`` matches *object_name*."""
    for node in nodes:
        if node.object_name == object_name:
            return node
    return None


def build_child_index(nodes: List[WidgetNode]) -> dict:
    """Return ``{parent_id: [child_node, ...]}`` for fast child lookup.

    Useful for the QTreeWidget renderer — given a node, finding its direct
    children is O(1) instead of O(n).
    """
    index: dict = {}
    for node in nodes:
        if node.parent_id is None:
            continue
        index.setdefault(node.parent_id, []).append(node)
    return index