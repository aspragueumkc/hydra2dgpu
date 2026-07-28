from typing import TYPE_CHECKING

if TYPE_CHECKING:
    try:
        from PyQt5.QtGui import (  # type: ignore  # noqa: F401
            QAction,
            QBrush,
            QCloseEvent,
            QColor,
            QImage,
            QPainter,
            QPaintEvent,
            QPen,
            QPolygonF,
            QWheelEvent,
        )
    except ImportError:
        from PyQt6.QtGui import (  # type: ignore  # noqa: F401
            QAction,
            QBrush,
            QCloseEvent,
            QColor,
            QImage,
            QPainter,
            QPaintEvent,
            QPen,
            QPolygonF,
            QWheelEvent,
        )
