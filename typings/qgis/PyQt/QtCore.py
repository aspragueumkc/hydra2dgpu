from typing import TYPE_CHECKING

if TYPE_CHECKING:
    try:
        from PyQt5.QtCore import (  # type: ignore  # noqa: F401
            QByteArray,
            QDateTime,
            QElapsedTimer,
            QEvent,
            QObject,
            QPoint,
            QPointF,
            QRect,
            QRectF,
            QSettings,
            QSize,
            Qt,
            QTimer,
            QVariant,
            pyqtProperty,
            pyqtSignal,
            pyqtSlot,
        )
    except ImportError:
        from PyQt6.QtCore import (  # type: ignore  # noqa: F401
            QByteArray,
            QDateTime,
            QElapsedTimer,
            QEvent,
            QObject,
            QPoint,
            QPointF,
            QRect,
            QRectF,
            QSettings,
            QSize,
            Qt,
            QTimer,
            QVariant,
            pyqtProperty,
            pyqtSignal,
            pyqtSlot,
        )
