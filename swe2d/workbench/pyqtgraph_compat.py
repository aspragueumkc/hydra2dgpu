"""QGIS/PyQt5 compatibility for pyqtgraph graphics-item callbacks.

QGIS can pass removals to ``itemChange`` as an unconvertible
``QVariant(PyQt_PyObject)`` instead of ``None``.  Pyqtgraph then forwards that
wrapper to Qt and SIP raises ``TypeError``.  This is pyqtgraph issue #774; its
binding-agnostic patch was rejected in PR #1125 because it broke PySide.
"""
from __future__ import annotations

from qgis.PyQt.QtCore import QVariant

_PATCH_MARKER = "_hydra_qgis_qvariant_item_change_patch"


def _wrap_item_change(original):
    if getattr(original, _PATCH_MARKER, False):
        return original

    def item_change(self, change, value):
        if (
            isinstance(value, QVariant)
            and value.typeName() == "PyQt_PyObject"
            and change
            in (
                self.GraphicsItemChange.ItemParentChange,
                self.GraphicsItemChange.ItemSceneChange,
            )
        ):
            value = None
        return original(self, change, value)

    setattr(item_change, _PATCH_MARKER, True)
    return item_change


def install_qgis_pyqtgraph_item_change_fix() -> None:
    """Install the narrow PyQt5-only fix once for pyqtgraph item removals."""
    from pyqtgraph.Qt import QT_LIB

    if QT_LIB != "PyQt5":
        return

    from pyqtgraph.graphicsItems.GraphicsObject import GraphicsObject
    from pyqtgraph.graphicsItems.ViewBox.ViewBox import ViewBox

    GraphicsObject.itemChange = _wrap_item_change(GraphicsObject.itemChange)
    # ViewBox derives from GraphicsWidget, not GraphicsObject, so its override
    # needs the same normalization independently.
    ViewBox.itemChange = _wrap_item_change(ViewBox.itemChange)
