"""Build HydraulicStructureConfig from a QGIS vector layer."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from qgis.core import QgsVectorLayer


class _ReadVisitor:
    __slots__ = ("_fields", "_ft")

    def __init__(self, fields: set, ft):
        self._fields = fields
        self._ft = ft

    def read_float(self, key: str, metadata: dict) -> None:
        """read float."""
        if key not in self._fields:
            return
        raw = self._ft[key]
        if raw is None:
            return
        try:
            metadata[key] = float(raw)
        except (ValueError, TypeError):
            s = str(raw).strip().lower()
            if s in ("null", "none", "nan", ""):
                return
            metadata[key] = s


def build_hydraulic_structure_config_from_layer(
    *,
    mesh_data: dict,
    have_qgis_core: bool,
    hydraulic_structure_config_cls: Any,
    structure_type_cls: Any,
    hydraulic_structure_cls: Any,
    structures_layer: "QgsVectorLayer",
    nearest_cell_fn: Optional[Callable] = None,
    log_fn: Optional[Callable] = None,
):
    """Build hydraulic structure config from layer."""
    if (
        mesh_data is None
        or not have_qgis_core
        or hydraulic_structure_config_cls is None
        or structure_type_cls is None
    ):
        return None

    type_name_map = {
        "weir": structure_type_cls.WEIR,
        "culvert": structure_type_cls.CULVERT,
        "gate": structure_type_cls.GATE,
        "bridge": structure_type_cls.BRIDGE,
        "pump": structure_type_cls.PUMP,
    }

    from qgis.core import QgsGeometry, QgsPointXY

    fields = set(structures_layer.fields().names())

    if nearest_cell_fn is None:
        from swe2d.mesh.mesh_runtime_logic import nearest_cell_index as _nCI, mesh_cell_centroids as _mCC
        nearest_cell_fn = _nCI
        cell_cx, cell_cy = _mCC(mesh_data)
    else:
        cell_cx = None
        cell_cy = None

    structures = []
    for ft in structures_layer.getFeatures():
        geom = ft.geometry()
        if geom is None or geom.isEmpty():
            continue
        try:
            if "enabled" in fields and int(ft["enabled"]) <= 0:
                continue
        except (KeyError, ValueError, TypeError):
            pass
        try:
            p0 = geom.interpolate(0.0).asPoint()
            p1 = geom.interpolate(max(0.0, float(geom.length()) - 1.0e-9)).asPoint()
        except Exception:
            continue

        raw_type = ft["structure_type"] if "structure_type" in fields else 2
        if isinstance(raw_type, str):
            structure_type = type_name_map.get(
                raw_type.strip().lower(), structure_type_cls.CULVERT
            )
        else:
            try:
                structure_type = structure_type_cls(int(raw_type))
            except (ValueError, TypeError):
                structure_type = structure_type_cls.CULVERT

        metadata = {}
        rv = _ReadVisitor(fields, ft)
        st = int(structure_type)

        for key in ("coeff", "cd", "max_flow", "use_redistribution", "influence_width"):
            rv.read_float(key, metadata)

        metadata["axis_x0"] = float(p0.x())
        metadata["axis_y0"] = float(p0.y())
        metadata["axis_x1"] = float(p1.x())
        metadata["axis_y1"] = float(p1.y())

        if st == int(structure_type_cls.CULVERT):
            for key in (
                "culvert_code", "culvert_shape", "culvert_rise", "culvert_span",
                "culvert_barrels", "culvert_area_m2",
                "diameter", "length", "roughness_n", "culvert_slope",
                "inlet_invert_elev", "outlet_invert_elev",
                "entrance_loss_k", "exit_loss_k",
                "culvert_entrance_loss", "culvert_exit_loss",
            ):
                rv.read_float(key, metadata)
            for key in (
                "embankment_enabled", "embankment_crest_elev",
                "embankment_overflow_width", "embankment_weir_coeff",
                "road_crest_elev", "road_overflow_width", "road_weir_coeff",
            ):
                rv.read_float(key, metadata)

        elif st == int(structure_type_cls.BRIDGE):
            for key in (
                "width", "length", "deck_soffit_elev", "deck_top_elev",
                "model_top_elev", "under_layers", "over_layers",
                "inlet_loss_coeff", "outlet_loss_coeff",
                "face_flux_depth_safety",
            ):
                rv.read_float(key, metadata)

        elif st == int(structure_type_cls.WEIR):
            for key in ("width",):
                rv.read_float(key, metadata)

        elif st == int(structure_type_cls.GATE):
            for key in ("width", "height", "opening"):
                rv.read_float(key, metadata)

        elif st == int(structure_type_cls.PUMP):
            for key in ("q_pump", "min_q", "max_q", "min_head_diff", "max_head_diff"):
                rv.read_float(key, metadata)

        crest_val = 0.0
        if "crest_elev" in fields:
            raw = ft["crest_elev"]
            if raw is not None:
                s = str(raw).strip().lower()
                if s not in ("null", "none", "nan", ""):
                    try:
                        crest_val = float(raw)
                    except (ValueError, TypeError):
                        pass

        upstream_cell = nearest_cell_fn(float(p0.x()), float(p0.y()), cell_cx, cell_cy)
        downstream_cell = nearest_cell_fn(float(p1.x()), float(p1.y()), cell_cx, cell_cy)
        raw_id = str(
            ft["structure_id"] if "structure_id" in fields else ft.id()
        ).strip()

        structures.append(
            hydraulic_structure_cls(
                structure_id=raw_id,
                structure_type=structure_type,
                upstream_cell=upstream_cell,
                downstream_cell=downstream_cell,
                crest_elev=crest_val,
                metadata=metadata or None,
            )
        )

    if not structures:
        return None
    return hydraulic_structure_config_cls(structures=structures, enabled=True)
