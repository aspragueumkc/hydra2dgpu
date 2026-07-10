"""Studio UI component registry — single source of truth for dock lifecycle.

Usage
-----
    from swe2d.workbench.views.studio_component_view import StudioComponent

    # Register a dock component:
    self._register_component(StudioComponent(
        name="my_panel",
        dock=self._studio_my_dock,
        area=Qt.RightDockWidgetArea,
        tab_with="inspector",
    ))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import QDockWidget

if TYPE_CHECKING:
    from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog as SWE2DWorkbenchDialog  # noqa: V104


@dataclass
class StudioComponent:
    """A registered dock component.

    Attributes
    ----------
    name:
        Unique key (e.g. ``"setup"``, ``"inspector"``, ``"results"``).
    dock:
        The QDockWidget instance.
    area:
        Qt.DockWidgetArea where the dock should be placed.
        Defaults to ``Qt.RightDockWidgetArea``.
    title:
        Human-readable title shown in the dock title bar.
    object_name:
        Qt objectName for state persistence.
    tab_with:
        Optional name of another registered component to tabify with.
    """

    name: str
    dock: QDockWidget
    area: Qt.DockWidgetArea = Qt.RightDockWidgetArea
    title: str = ""
    object_name: str = ""
    tab_with: Optional[str] = None

    def __post_init__(self) -> None:
        """Fill default title and object_name if not already set."""
        self.title = self.name.title()
        if not self.object_name:
            self.object_name = f"HYDRA2D{self.name.title()}Dock"


__all__ = ["StudioComponent"]
