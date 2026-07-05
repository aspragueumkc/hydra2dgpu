"""Widget persistence service — pure Python, zero Qt imports.

Extracted from studio_dialog.py to enforce MVP service-layer boundaries.
Functions accept explicit callbacks and iterables; they never touch
``self`` or reference widgets by name.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple


def iter_all_persistable_widgets(
    dialog: object,
    tab_views: Sequence[object],
    persistable_types: Tuple[type, ...] = (),
) -> Iterator[Tuple[str, object]]:
    """Yield ``(attr_name, widget)`` pairs across *dialog* and *tab_views*.

    Parameters
    ----------
    dialog : object
        The top-level dialog (or any object whose ``vars()`` contains widget
        attributes).  The dialog itself is always scanned first.
    tab_views : sequence of object
        Additional objects (tab views) whose public attributes are scanned.
        ``None`` entries are silently skipped.
    persistable_types : tuple of type, optional
        Qt widget classes that should be yielded.  When empty, a sensible
        default of common persistable widgets is used.  The caller must
        supply the types from ``QtWidgets`` — this module never imports Qt.
    """
    if not persistable_types:
        raise ValueError(
            "persistable_types must be supplied by the caller "
            "(import QtWidgets and pass the widget classes)"
        )
    seen: set[int] = set()
    sources = [dialog] + [v for v in tab_views if v is not None]
    for source in sources:
        for attr_name in vars(source):
            if attr_name.startswith("_"):
                continue
            widget = getattr(source, attr_name, None)
            if widget is None:
                continue
            try:
                _ = widget.objectName()
            except RuntimeError:
                continue
            if not isinstance(widget, persistable_types):
                continue
            # Skip widgets marked as non-persistable (e.g. transient display controls)
            if getattr(widget, "_no_persist", False):
                continue
            wid = id(widget)
            if wid in seen:
                continue
            seen.add(wid)
            yield (attr_name, widget)


def is_project_workbench_state_persist_blocked(
    state_obj: object,
) -> bool:
    """Return ``True`` if persistence should be suppressed (during restore).

    Parameters
    ----------
    state_obj : object
        The dialog's ``_state`` attribute (or any object carrying
        ``persist_suppressed``).
    """
    try:
        return bool(state_obj.persist_suppressed)
    except Exception:
        return False


def persist_project_workbench_state(
    *,
    have_qgis_core: bool,
    qgs_project_cls: object,
    workbench_state_key: str,
    state_obj: object,
    iter_widgets_fn: Callable[[], Iterator[Tuple[str, object]]],
    write_project_json_fn: Callable[..., bool],
    log_fn: Callable[[str], None],
) -> bool:
    """Persist widget state to the current QGIS project.

    Parameters
    ----------
    have_qgis_core : bool
        ``True`` when ``qgis.core`` is importable.
    qgs_project_cls : object
        ``QgsProject`` class (or ``None``).
    workbench_state_key : str
        Project-settings key under which the JSON payload is stored.
    state_obj : object
        Object carrying ``persist_suppressed`` (set by the dialog).
    iter_widgets_fn : callable
        Zero-argument callable returning an iterator of
        ``(attr_name, widget)`` pairs.
    write_project_json_fn : callable
        ``write_project_json(...)`` from the project settings bridge.
    log_fn : callable
        Logging callback ``f(msg: str) -> None``.

    Returns
    -------
    bool
        ``True`` when the payload was written successfully.
    """
    if not have_qgis_core or qgs_project_cls is None:
        return False
    if is_project_workbench_state_persist_blocked(state_obj):
        return False

    widgets_data: Dict[str, Dict[str, Any]] = {}
    for attr_name, widget in iter_widgets_fn():
        val = None
        qt_mod = _qt_widgets_module(widget)
        if qt_mod is not None:
            if isinstance(widget, (qt_mod.QSpinBox, qt_mod.QDoubleSpinBox)):
                try:
                    widget.interpretText()
                except Exception as _e:
                    log_fn(f"[ERROR] Exception in widget_persistence_service.py: {_e}")
                val = widget.value()
            elif isinstance(widget, qt_mod.QComboBox):
                val = widget.currentData()
                if val is None:
                    val = widget.currentIndex()
            elif isinstance(widget, qt_mod.QCheckBox):
                val = widget.isChecked()
            elif isinstance(widget, qt_mod.QLineEdit):
                val = widget.text()
        if val is not None:
            widgets_data[attr_name] = {"type": type(widget).__name__, "value": val}

    payload = {"version": 1, "widgets": widgets_data}
    try:
        ok = write_project_json_fn(
            have_qgis_core=have_qgis_core,
            qgs_project_cls=qgs_project_cls,
            key=workbench_state_key,
            payload=payload,
            log_callback=log_fn,
        )
    except Exception as _e:
        log_fn(f"[ERROR] Exception in widget_persistence_service.py: {_e}")
        return False
    if ok:
        log_fn(f"[DEBUG] persist: saved {len(widgets_data)} widgets to project")
    return ok


def restore_project_workbench_state(
    *,
    have_qgis_core: bool,
    qgs_project_cls: object,
    workbench_state_key: str,
    state_obj: object,
    iter_widgets_fn: Callable[[], Iterator[Tuple[str, object]]],
    load_project_json_fn: Callable[..., Optional[object]],
    log_fn: Callable[[str], None],
) -> int:
    """Restore widget state from the current QGIS project.

    Parameters
    ----------
    have_qgis_core : bool
        ``True`` when ``qgis.core`` is importable.
    qgs_project_cls : object
        ``QgsProject`` class (or ``None``).
    workbench_state_key : str
        Project-settings key from which the JSON payload is read.
    state_obj : object
        Object carrying ``persist_suppressed`` (set/cleared around restore).
    iter_widgets_fn : callable
        Zero-argument callable returning an iterator of
        ``(attr_name, widget)`` pairs.
    load_project_json_fn : callable
        ``load_project_json(...)`` from the project settings bridge.
    log_fn : callable
        Logging callback ``f(msg: str) -> None``.

    Returns
    -------
    int
        Number of widgets successfully restored.
    """
    if not have_qgis_core or qgs_project_cls is None:
        log_fn("[DEBUG] restore: QGIS core not available")
        return 0

    payload = load_project_json_fn(
        have_qgis_core=have_qgis_core,
        qgs_project_cls=qgs_project_cls,
        key=workbench_state_key,
        default=None,
        log_callback=log_fn,
    )
    if payload is None:
        log_fn("[DEBUG] restore: no saved workbench state found")
        return 0

    widgets_data = payload.get("widgets", {}) if isinstance(payload, dict) else {}
    log_fn(f"[DEBUG] restore: restoring {len(widgets_data)} widget values")

    widget_map: Dict[str, object] = {}
    for attr_name, widget in iter_widgets_fn():
        widget_map[attr_name] = widget

    state_obj.persist_suppressed = True
    restored_count = 0
    try:
        for attr_name, widget_info in widgets_data.items():
            widget = widget_map.get(attr_name)
            if widget is None or not isinstance(widget_info, dict):
                continue
            value = widget_info.get("value")
            if value is None:
                continue
            try:
                _ = widget.objectName()
            except RuntimeError:
                continue
            qt_mod = _qt_widgets_module(widget)
            if qt_mod is None:
                continue
            if isinstance(widget, qt_mod.QSpinBox):
                widget.setValue(int(value))
                restored_count += 1
            elif isinstance(widget, qt_mod.QDoubleSpinBox):
                widget.setValue(float(value))
                restored_count += 1
            elif isinstance(widget, qt_mod.QComboBox):
                found = False
                for index in range(widget.count()):
                    data = widget.itemData(index)
                    if isinstance(value, list) and isinstance(data, tuple):
                        if tuple(value) == data:
                            widget.setCurrentIndex(index)
                            found = True
                            break
                    elif data == value:
                        widget.setCurrentIndex(index)
                        found = True
                        break
                if not found:
                    try:
                        widget.setCurrentIndex(int(value))
                    except (TypeError, ValueError):
                        pass
                restored_count += 1
            elif isinstance(widget, qt_mod.QCheckBox):
                widget.setChecked(bool(value))
                restored_count += 1
            elif isinstance(widget, qt_mod.QLineEdit):
                widget.setText(str(value))
                restored_count += 1
    finally:
        state_obj.persist_suppressed = False

    log_fn(
        f"[DEBUG] restore: successfully restored "
        f"{restored_count} of {len(widgets_data)} widgets"
    )
    return restored_count


def _qt_widgets_module(widget: object) -> Optional[object]:
    """Return the ``QtWidgets`` module that owns *widget*, or ``None``."""
    qt_mod = getattr(type(widget), "__module__", "")
    if not qt_mod:
        return None
    parts = qt_mod.split(".")
    if len(parts) >= 2 and parts[0] == "PyQt5":
        try:
            from PyQt5 import QtWidgets  # noqa: WPS433 — lazy import
            return QtWidgets
        except ImportError:
            return None
    if len(parts) >= 2 and parts[0] == "PySide2":
        try:
            from PySide2 import QtWidgets  # noqa: WPS433 — lazy import
            return QtWidgets
        except ImportError:
            return None
    return None
