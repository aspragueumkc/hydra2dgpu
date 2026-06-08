from __future__ import annotations

import json
import logging
from typing import Dict, Optional, Sequence

logger = logging.getLogger(__name__)

PROJECT_SETTINGS_SCOPE = "Backwater2DWorkbench"
LAYER_SELECTOR_STATE_KEY = "layer_selector_state_json"
WORKBENCH_STATE_KEY = "workbench_state_json"
LAYER_SELECTOR_STATE_VERSION = 1
WORKBENCH_STATE_VERSION = 1


def read_project_entry_text(
    *,
    have_qgis_core: bool,
    qgs_project_cls: object,
    key: str,
    default: str = "",
) -> str:
    if not have_qgis_core or qgs_project_cls is None:
        return str(default)
    try:
        result = qgs_project_cls.instance().readEntry(PROJECT_SETTINGS_SCOPE, key, str(default))
    except Exception as exc:
        logger.debug("[UI] readEntry failed for key %s: %s", key, exc)
        return str(default)
    if isinstance(result, tuple):
        return str(result[0] if result and result[0] not in (None, "") else default)
    return str(result if result not in (None, "") else default)


def load_project_json(
    *,
    have_qgis_core: bool,
    qgs_project_cls: object,
    key: str,
    default: Optional[object] = None,
    log_callback=None,
) -> Optional[object]:
    raw = read_project_entry_text(
        have_qgis_core=have_qgis_core,
        qgs_project_cls=qgs_project_cls,
        key=key,
        default="",
    )
    if not raw:
        return default
    try:
        return json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        if callable(log_callback):
            log_callback(f"[DEBUG] settings: failed to parse {key}: {exc}")
        return default


def write_project_json(
    *,
    have_qgis_core: bool,
    qgs_project_cls: object,
    key: str,
    payload: object,
    log_callback=None,
) -> bool:
    if not have_qgis_core or qgs_project_cls is None:
        return False
    try:
        qgs_project_cls.instance().writeEntry(
            PROJECT_SETTINGS_SCOPE,
            key,
            json.dumps(payload, separators=(",", ":"), default=str),
        )
        return True
    except Exception as exc:
        if callable(log_callback):
            log_callback(f"[DEBUG] settings: writeEntry failed for {key}: {exc}")
        return False


def build_layer_selector_state(specs: Sequence[tuple[str, object]]) -> Dict[str, object]:
    payload = {"version": LAYER_SELECTOR_STATE_VERSION, "selectors": {}}
    for attr_name, combo in specs:
        idx = combo.currentIndex()
        label = str(combo.currentText() or "").strip() if idx >= 0 else ""
        layer_id = combo.currentData()
        payload["selectors"][attr_name] = {
            "layer_id": "" if layer_id in (None, "") else str(layer_id),
            "layer_name": label,
        }
    return payload


def parse_layer_selector_state(payload: object) -> Dict[str, Dict[str, str]]:
    if not isinstance(payload, dict):
        return {}
    selectors = payload.get("selectors", {})
    if not isinstance(selectors, dict):
        return {}
    out: Dict[str, Dict[str, str]] = {}
    for attr_name, saved in selectors.items():
        if not isinstance(saved, dict):
            continue
        out[str(attr_name)] = {
            "layer_id": str(saved.get("layer_id") or ""),
            "layer_name": str(saved.get("layer_name") or ""),
        }
    return out


def collect_workbench_widget_state(*, ui: object, widget_attrs: Sequence[str], qtwidgets_module: object) -> Dict[str, object]:
    payload = {"version": WORKBENCH_STATE_VERSION, "widgets": {}}
    abstract_spin_box_cls = getattr(qtwidgets_module, "QAbstractSpinBox", None)
    spin_box_cls = getattr(qtwidgets_module, "QSpinBox")
    double_spin_box_cls = getattr(qtwidgets_module, "QDoubleSpinBox")
    combo_box_cls = getattr(qtwidgets_module, "QComboBox")
    check_box_cls = getattr(qtwidgets_module, "QCheckBox")
    line_edit_cls = getattr(qtwidgets_module, "QLineEdit")

    for attr_name in widget_attrs:
        widget = getattr(ui, attr_name, None)
        if widget is None:
            continue

        # Commit any in-progress keyboard edits (especially in spin boxes)
        # before reading persisted values.
        if abstract_spin_box_cls is not None and isinstance(widget, abstract_spin_box_cls):
            try:
                widget.interpretText()
            except Exception as exc:
                logger.debug("[UI] interpretText failed for %s: %s", attr_name, exc)

        value = None
        if isinstance(widget, spin_box_cls):
            value = widget.value()
        elif isinstance(widget, double_spin_box_cls):
            value = widget.value()
        elif isinstance(widget, combo_box_cls):
            value = widget.currentData()
            if value is None:
                value = widget.currentIndex()
        elif isinstance(widget, check_box_cls):
            value = widget.isChecked()
        elif isinstance(widget, line_edit_cls):
            value = widget.text()
        else:
            continue

        payload["widgets"][attr_name] = {
            "type": type(widget).__name__,
            "value": value,
        }
    return payload


def restore_workbench_widget_state(
    *,
    ui: object,
    widgets_data: object,
    qtwidgets_module: object,
    log_callback=None,
) -> int:
    if not isinstance(widgets_data, dict):
        if callable(log_callback):
            log_callback("[DEBUG] restore: widgets not a dict")
        return 0

    spin_box_cls = getattr(qtwidgets_module, "QSpinBox")
    double_spin_box_cls = getattr(qtwidgets_module, "QDoubleSpinBox")
    combo_box_cls = getattr(qtwidgets_module, "QComboBox")
    check_box_cls = getattr(qtwidgets_module, "QCheckBox")
    line_edit_cls = getattr(qtwidgets_module, "QLineEdit")

    restored_count = 0
    for attr_name, widget_info in widgets_data.items():
        widget = getattr(ui, attr_name, None)
        if widget is None or not isinstance(widget_info, dict):
            continue

        value = widget_info.get("value")
        if value is None:
            continue

        try:
            if isinstance(widget, spin_box_cls):
                widget.setValue(int(value))
                restored_count += 1
            elif isinstance(widget, double_spin_box_cls):
                widget.setValue(float(value))
                restored_count += 1
            elif isinstance(widget, combo_box_cls):
                found = False
                for index in range(widget.count()):
                    if widget.itemData(index) == value:
                        widget.setCurrentIndex(index)
                        found = True
                        break
                if not found:
                    widget.setCurrentIndex(int(value))
                restored_count += 1
            elif isinstance(widget, check_box_cls):
                widget.setChecked(bool(value))
                restored_count += 1
            elif isinstance(widget, line_edit_cls):
                widget.setText(str(value))
                restored_count += 1
        except (TypeError, ValueError) as exc:
            if callable(log_callback):
                log_callback(f"[DEBUG] restore: failed to restore {attr_name}: {exc}")
    return restored_count