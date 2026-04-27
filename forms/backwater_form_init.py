import importlib.util
import os

from qgis.PyQt.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QStyle,
    QWidget,
)
from qgis.core import (
    QgsApplication,
    QgsExpressionContextUtils,
    QgsFeatureRequest,
    QgsProject,
    QgsRasterLayer,
)

try:
    from qgis.utils import iface
except Exception:
    iface = None


def _notify(message, level='Info', duration=6):
    if iface is None:
        return
    try:
        from qgis.core import Qgis
        lvl = getattr(Qgis, level, Qgis.Info)
        iface.messageBar().pushMessage('Backwater', message, level=lvl, duration=duration)
    except Exception:
        pass


def _style_action_button(button, icon_kind='play'):
    if button is None:
        return
    try:
        style = QApplication.style()
        if icon_kind == 'terrain':
            icon = style.standardIcon(QStyle.SP_DriveNetIcon)
        elif icon_kind == 'z':
            icon = style.standardIcon(QStyle.SP_ArrowUp)
        else:
            icon = style.standardIcon(QStyle.SP_MediaPlay)
        button.setIcon(icon)
    except Exception:
        pass

    button.setMinimumHeight(30)
    button.setStyleSheet(
        'QPushButton {'
        'background-color: #1f3d5a;'
        'color: white;'
        'border: 1px solid #0f2438;'
        'border-radius: 5px;'
        'padding: 4px 10px;'
        'font-weight: 600;'
        '}'
        'QPushButton:hover { background-color: #2e5c84; }'
        'QPushButton:pressed { background-color: #173149; }'
    )


def _find_action_by_name(layer, action_name):
    if layer is None:
        return None
    manager = layer.actions()
    candidate_scopes = ('Form', 'Feature', 'Layer', 'Canvas', '')
    seen = set()
    for scope in candidate_scopes:
        try:
            actions = manager.actions(scope)
        except Exception:
            actions = []
        for action in actions:
            try:
                aid = str(action.id())
                if aid in seen:
                    continue
                seen.add(aid)
                if str(action.name()) == str(action_name):
                    return action
            except Exception:
                continue
    return None


def _current_feature(layer, feature):
    if feature is None or layer is None:
        return feature
    try:
        fid = feature.id()
        if fid is None or int(fid) < 0:
            return feature
        got = next(layer.getFeatures(QgsFeatureRequest(int(fid))), None)
        return got if got is not None else feature
    except Exception:
        return feature


def _run_action(layer, feature, action_name):
    if layer is None:
        _notify('No active layer/form context for action.', level='Warning')
        return
    action = _find_action_by_name(layer, action_name)
    if action is None:
        _notify(f'Action not found: {action_name}', level='Warning')
        return
    try:
        feat = _current_feature(layer, feature)
        layer.actions().doActionFeature(action.id(), feat)
    except Exception as exc:
        _notify(f'Action failed: {exc}', level='Critical', duration=8)


def _selected_or_default_raster(project):
    raster_id = str(QgsExpressionContextUtils.projectScope(project).variable('backwater_terrain_raster_id') or '').strip()
    raster_layer = project.mapLayer(raster_id) if raster_id else None
    if isinstance(raster_layer, QgsRasterLayer) and raster_layer.isValid():
        return raster_layer

    rasters = [lyr for lyr in project.mapLayers().values() if isinstance(lyr, QgsRasterLayer) and lyr.isValid()]
    if not rasters:
        return None
    raster_layer = rasters[0]
    QgsExpressionContextUtils.setProjectVariable(project, 'backwater_terrain_raster_id', raster_layer.id())
    return raster_layer


def _load_set_z_callable():
    candidate_dirs = []

    file_path = globals().get('__file__')
    if file_path:
        try:
            candidate_dirs.append(os.path.dirname(os.path.dirname(os.path.abspath(file_path))))
        except Exception:
            pass

    try:
        candidate_dirs.append(
            os.path.join(QgsApplication.qgisSettingsDirPath(), 'python', 'plugins', 'qgis-backwater-plugin')
        )
    except Exception:
        pass

    expr_path = None
    for plugin_dir in candidate_dirs:
        cand = os.path.join(plugin_dir, 'expressions', 'vertices_z_from_raster.py')
        if os.path.exists(cand):
            expr_path = cand
            break

    if not expr_path:
        raise RuntimeError(
            f'Could not resolve vertices_z_from_raster.py. Tried: {candidate_dirs}'
        )

    spec = importlib.util.spec_from_file_location('backwater_vertices_z_from_raster', expr_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Could not import raster expression module: {expr_path}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    func = getattr(mod, 'set_z_from_raster_expr_py', None)
    if not callable(func):
        func = getattr(mod, '_set_z_from_raster_impl', None)
    if not callable(func):
        func = getattr(mod, 'set_z_from_raster_expr', None)
    if not callable(func):
        raise RuntimeError('set_z_from_raster_expr callable not found')
    return func


def _find_centerline_layer(project, cross_layer):
    cross_source = ''
    try:
        cross_source = str(cross_layer.source()).split('|', 1)[0]
    except Exception:
        cross_source = ''

    best = None
    for lyr in project.mapLayers().values():
        try:
            if getattr(lyr, 'name', lambda: '')() != 'centerline':
                continue
            if best is None:
                best = lyr
            src = str(lyr.source()).split('|', 1)[0]
            if cross_source and src == cross_source:
                return lyr
        except Exception:
            continue
    return best


def _update_z_for_form_feature(layer, feature):
    if layer is None:
        raise RuntimeError('No active layer/form context')

    feat = _current_feature(layer, feature)
    if feat is None:
        raise RuntimeError('Cross section feature not found in form context')

    geom = feat.geometry()
    if geom is None or geom.isEmpty():
        raise RuntimeError('Current feature has no geometry')

    project = QgsProject.instance()
    raster_layer = _selected_or_default_raster(project)
    if raster_layer is None:
        raise RuntimeError('No valid raster layer available')

    func = _load_set_z_callable()
    sampled_geom = func(geom, raster_layer, 1, True)
    if sampled_geom is None or sampled_geom.isEmpty():
        raise RuntimeError('Raster sampling returned empty geometry')

    if not layer.isEditable():
        layer.startEditing()

    try:
        fid = int(feat.id())
    except Exception:
        fid = -1

    if fid > -1:
        if not layer.changeGeometry(fid, sampled_geom):
            raise RuntimeError('Failed to apply sampled geometry to feature')
    else:
        feat.setGeometry(sampled_geom)

    river_idx = layer.fields().indexOf('river_station')
    center_idx = layer.fields().indexOf('centerline_id')

    def _set_attr(field_idx, value):
        if field_idx == -1:
            return
        if fid > -1:
            layer.changeAttributeValue(fid, field_idx, value)
        else:
            feat.setAttribute(field_idx, value)

    if center_idx != -1:
        try:
            cur_center_id = feat[center_idx]
        except Exception:
            cur_center_id = None
        if cur_center_id is None:
            _set_attr(center_idx, 1)

    center_layer = _find_centerline_layer(project, layer)
    if center_layer is not None:
        center_feat = next(center_layer.getFeatures(), None)
        if center_feat is not None:
            center_geom = center_feat.geometry()
            if center_geom is not None and not center_geom.isEmpty():
                locate_geom = None
                try:
                    crossing = sampled_geom.intersection(center_geom)
                except Exception:
                    crossing = None
                if crossing is not None and not crossing.isEmpty():
                    locate_geom = crossing.centroid()
                if locate_geom is None or locate_geom.isEmpty():
                    try:
                        locate_geom = center_geom.nearestPoint(sampled_geom)
                    except Exception:
                        locate_geom = None
                if locate_geom is None or locate_geom.isEmpty():
                    chainage = -1.0
                else:
                    chainage = float(center_geom.lineLocatePoint(locate_geom))
                if chainage >= 0 and river_idx != -1:
                    _set_attr(river_idx, f"{chainage:.3f}")

    if fid <= -1:
        if not layer.updateFeature(feat):
            raise RuntimeError('Could not update geometry for temporary feature')

    _notify(f'Updated geometry Z from raster {raster_layer.name()}.', level='Success', duration=5)


def _read_numeric_widget_value(widget):
    if widget is None:
        return 0.0
    try:
        if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            return float(widget.value())
        if isinstance(widget, QComboBox):
            return float((widget.currentText() or '0').strip() or 0.0)
        if isinstance(widget, QLineEdit):
            return float((widget.text() or '0').strip() or 0.0)
    except Exception:
        return 0.0
    return 0.0


def _set_field_visible(dialog, field_name, visible):
    fld = dialog.findChild(QWidget, field_name)
    if fld is not None:
        fld.setVisible(bool(visible))
    lbl = dialog.findChild(QWidget, f'lbl_{field_name}')
    if lbl is not None:
        lbl.setVisible(bool(visible))


def _apply_culvert_visibility(dialog):
    code_widget = dialog.findChild(QWidget, 'culvert_code')
    culvert_on = _read_numeric_widget_value(code_widget) > 0.0
    culvert_fields = (
        'culvert_shape',
        'culvert_diameter',
        'culvert_width',
        'culvert_height',
        'culvert_upstream_invert',
        'culvert_downstream_invert',
        'culvert_length',
        'culvert_weir_coeff',
        'culvert_weir_sta_left',
        'culvert_weir_sta_right',
    )
    for field_name in culvert_fields:
        _set_field_visible(dialog, field_name, culvert_on)


def _wire_culvert_visibility(dialog):
    code_widget = dialog.findChild(QWidget, 'culvert_code')
    if code_widget is None:
        return

    _apply_culvert_visibility(dialog)

    try:
        if isinstance(code_widget, QLineEdit):
            code_widget.textChanged.connect(lambda *_: _apply_culvert_visibility(dialog))
        elif isinstance(code_widget, QComboBox):
            code_widget.currentTextChanged.connect(lambda *_: _apply_culvert_visibility(dialog))
        elif isinstance(code_widget, (QSpinBox, QDoubleSpinBox)):
            code_widget.valueChanged.connect(lambda *_: _apply_culvert_visibility(dialog))
    except Exception:
        pass


def _bind_button(dialog, layer, feature, object_name, action_name, icon_kind, callback=None):
    btn = dialog.findChild(QPushButton, object_name)
    if btn is None:
        return
    _style_action_button(btn, icon_kind=icon_kind)
    try:
        btn.clicked.disconnect()
    except Exception:
        pass

    def _on_click(*_args):
        if callable(callback):
            try:
                callback(layer, feature)
            except Exception as exc:
                _notify(f'Action failed: {exc}', level='Critical', duration=8)
            return
        _run_action(layer, feature, action_name)

    btn.clicked.connect(_on_click)


def backwater_cross_sections_form_open(dialog, layer, feature):
    _bind_button(dialog, layer, feature, 'btn_select_terrain', 'Backwater: Select Terrain Raster', 'terrain')
    _bind_button(
        dialog,
        layer,
        feature,
        'btn_update_z',
        'Backwater: Update Z From Terrain',
        'z',
        callback=_update_z_for_form_feature,
    )
    _wire_culvert_visibility(dialog)


def backwater_boundary_form_open(dialog, layer, feature):
    _bind_button(dialog, layer, feature, 'btn_run_model', 'Backwater: Run Model', 'play')
