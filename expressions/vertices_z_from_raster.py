from qgis.utils import qgsfunction
from qgis.core import (
    QgsProject,
    QgsRasterLayer,
    QgsGeometry,
    QgsPoint,
    QgsExpression,
    QgsExpressionContext,
    QgsExpressionContextScope,
)


def _resolve_raster_name(raster_ref):
    if isinstance(raster_ref, QgsRasterLayer):
        return raster_ref.name()

    lyr = QgsProject.instance().mapLayer(str(raster_ref))
    if isinstance(lyr, QgsRasterLayer):
        return lyr.name()

    by_name = QgsProject.instance().mapLayersByName(str(raster_ref))
    if by_name and isinstance(by_name[0], QgsRasterLayer):
        return by_name[0].name()

    return None


@qgsfunction(args="auto", group="Custom")
def set_z_from_raster_expr(geom, raster_ref, band=1, keep_existing_if_nodata=True, feature=None, parent=None):
    return _set_z_from_raster_impl(geom, raster_ref, band, keep_existing_if_nodata)


def set_z_from_raster_expr_py(geom, raster_ref, band=1, keep_existing_if_nodata=True):
    """Plain Python callable for plugin code paths.

    QGIS wraps functions decorated with @qgsfunction in a QgsPyExpressionFunction,
    which is not directly callable from plugin Python code in some contexts.
    """
    return _set_z_from_raster_impl(geom, raster_ref, band, keep_existing_if_nodata)


def _set_z_from_raster_impl(geom, raster_ref, band=1, keep_existing_if_nodata=True):
    """
    set_z_from_raster_expr($geometry, 'USGS_1m_DEM_ft_callaway', 1, true)

    Uses field-calculator expressions internally:
      raster_value(<raster>, <band>, point_n(<geometry>, <n>))
    and writes sampled Z back to each vertex.
    """
    if geom is None or geom.isEmpty():
        return geom

    raster_name = _resolve_raster_name(raster_ref)
    if raster_name is None:
        return geom

    out = QgsGeometry(geom)

    # Ensure Z dimension exists.
    #if not out.is3D():
    #    out.addZValue(0.0)

    expr = QgsExpression("raster_value(@ras, @band, point_n(@g, @n))")
    if expr.hasParserError():
        return geom

    scope = QgsExpressionContextScope()
    scope.setVariable("ras", raster_name)
    scope.setVariable("band", int(band))

    ctx = QgsExpressionContext()
    ctx.appendScope(scope)

    changed = 0
    idx = 0  # moveVertex uses 0-based index

    for v in out.vertices():
        scope.setVariable("g", out)
        scope.setVariable("n", idx + 1)  # point_n is 1-based

        z_val = expr.evaluate(ctx)
        if expr.hasEvalError() or z_val is None:
            if not bool(keep_existing_if_nodata):
                # Explicitly set to 0 if requested.
                p = QgsPoint(v.x(), v.y(), 0.0)
                out.moveVertex(p, idx)
            idx += 1
            continue

        try:
            z_num = float(z_val)
        except Exception:
            if not bool(keep_existing_if_nodata):
                p = QgsPoint(v.x(), v.y(), 0.0)
                out.moveVertex(p, idx)
            idx += 1
            continue

        # Use explicit numeric constructor to avoid "Invalid type in constructor arguments".
        p = QgsPoint(v.x(), v.y(), z_num)
        if out.moveVertex(p, idx):
            changed += 1

        idx += 1

    if changed == 0 and bool(keep_existing_if_nodata):
        return geom

    return out
