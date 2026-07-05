"""Persisted workbench state — QSettings + QMainWindow save/restore.

All public functions are wrappers around QSettings keys.  State is stored
under the same ``HYDRA2DGPU/HYDRA2DGPU`` settings root used elsewhere in the
plugin so the user gets one consistent settings file.

Keys stored:

* ``workbench/open_on_startup`` -- bool, default False
* ``workbench/was_open``         -- bool, default False
* ``workbench/dock_state``       -- QByteArray from
  ``QMainWindow.saveState()`` (dock positions, sizes, visibility,
  tab stacks, floating state)
* ``workbench/geometry``         -- QByteArray from
  ``QMainWindow.saveGeometry()`` (window size + position)

Restoring dock layout depends on every dock having a stable
``objectName``.  ``WorkbenchDialogBuilder._build_component`` already
sets ``HYDRA2D{name.title()}Dock`` for every dock it creates, which is
sufficient for ``QMainWindow.restoreState`` to round-trip the layout.

Failures (corrupt bytes, missing docks, etc.) are logged and swallowed
so a bad QSettings blob cannot brick a QGIS session.

The helper module is intentionally QGIS-free (no ``from qgis.*``),
so it can be unit-tested under headless ``QApplication([])``.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


# QSettings key constants (exported so callers can audit / migrate)
K_OPEN_ON_STARTUP = "workbench/open_on_startup"
K_WAS_OPEN = "workbench/was_open"
K_DOCK_STATE = "workbench/dock_state"
K_GEOMETRY = "workbench/geometry"


def _settings(settings: Optional[Any]) -> Optional[Any]:
    """Return the given QSettings or None (defensive — never raise)."""
    if settings is None:
        return None
    return settings


def load_open_on_startup(settings: Any) -> bool:
    """Return the persisted open-on-startup flag (default False)."""
    s = _settings(settings)
    if s is None:
        return False
    try:
        raw = s.value(K_OPEN_ON_STARTUP, False)
    except Exception as exc:
        logger.warning(
            "load_open_on_startup: QSettings read failed: %s", exc
        )
        return False
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return bool(raw)


def save_open_on_startup(settings: Any, value: bool) -> None:
    """Persist the open-on-startup flag."""
    s = _settings(settings)
    if s is None:
        return
    try:
        s.setValue(K_OPEN_ON_STARTUP, bool(value))
    except Exception as exc:
        logger.warning(
            "save_open_on_startup: QSettings write failed: %s", exc
        )


def was_open(settings: Any) -> Optional[bool]:
    """Return the persisted was-open flag (None if never set)."""
    s = _settings(settings)
    if s is None:
        return None
    try:
        if not s.contains(K_WAS_OPEN):
            return None
        raw = s.value(K_WAS_OPEN)
    except Exception as exc:
        logger.warning("was_open: QSettings read failed: %s", exc)
        return None
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return bool(raw)


def save_was_open(settings: Any, value: bool) -> None:
    """Persist the was-open flag (set at launch + close)."""
    s = _settings(settings)
    if s is None:
        return
    try:
        s.setValue(K_WAS_OPEN, bool(value))
    except Exception as exc:
        logger.warning("save_was_open: QSettings write failed: %s", exc)


def save_window_state(settings: Any, main_window: Any) -> bool:
    """Persist dock state + geometry from *main_window*.

    Returns True on success, False if either save failed (logged).
    Silent on a missing or null ``main_window`` — that's expected
    in test setups.
    """
    if settings is None or main_window is None:
        return False
    try:
        dock_state = main_window.saveState()
        geometry = main_window.saveGeometry()
    except Exception as exc:
        logger.warning("save_window_state: saveState/Geometry failed: %s", exc)
        return False
    try:
        settings.setValue(K_DOCK_STATE, dock_state)
        settings.setValue(K_GEOMETRY, geometry)
    except Exception as exc:
        logger.warning("save_window_state: QSettings write failed: %s", exc)
        return False
    return True


def restore_window_state(settings: Any, main_window: Any) -> bool:
    """Restore dock state + geometry on *main_window*.

    Returns ``True`` if *either* a saved dock state or geometry
    blob was applied (regardless of Qt's per-call return value).
    Returns ``False`` if neither saved blob was present, or the
    helpers raised (e.g. corrupt bytes).

    Rationale: a fresh installation has nothing to restore, and
    that's a successful no-op — not a failure.  ``restoreState``
    with a missing blob returns ``True`` (Qt treats that as OK),
    so we can't trivially delegate the success/failure decision
    to Qt.  Instead we distinguish "had data, attempted restore"
    from "nothing to restore".

    Failures are logged but never raised — a corrupt blob must
    never crash a QGIS session.

    Note: ``restoreState`` is silently effective when called after
    every dock has been added with a stable ``objectName``.  See the
    module docstring.
    """
    if settings is None or main_window is None:
        return False
    try:
        dock_state = settings.value(K_DOCK_STATE)
        geometry = settings.value(K_GEOMETRY)
    except Exception as exc:
        logger.warning("restore_window_state: QSettings read failed: %s", exc)
        return False
    attempted = False
    if dock_state is not None:
        attempted = True
        try:
            main_window.restoreState(dock_state)
        except Exception as exc:
            logger.warning("restore_window_state: restoreState raised: %s", exc)
            return False
    if geometry is not None:
        attempted = True
        try:
            main_window.restoreGeometry(geometry)
        except Exception as exc:
            logger.warning("restore_window_state: restoreGeometry raised: %s", exc)
            return False
    return attempted


def clear_window_state(settings: Any) -> None:
    """Drop the dock state + geometry keys (full reset to defaults)."""
    s = _settings(settings)
    if s is None:
        return
    for key in (K_DOCK_STATE, K_GEOMETRY, K_WAS_OPEN):
        try:
            s.remove(key)
        except Exception as exc:
            logger.warning("clear_window_state: remove %s failed: %s", key, exc)
