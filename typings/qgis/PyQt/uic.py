from typing import TYPE_CHECKING

if TYPE_CHECKING:
    try:
        from PyQt5.uic import compileUi, loadUi  # type: ignore  # noqa: F401
    except ImportError:
        from PyQt6.uic import compileUi, loadUi  # type: ignore  # noqa: F401
