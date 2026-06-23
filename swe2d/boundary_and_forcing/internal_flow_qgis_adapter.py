from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np

from swe2d.boundary_and_forcing.internal_flow_logic import (
    build_hydrograph_lookup_from_features,
    build_internal_flow_forcing_from_features,
    first_matching_field,
    resolve_internal_flow_field_name,
    resolve_layer_hydrograph_for_feature,
)
from swe2d.boundary_and_forcing.internal_flow_qgis_geometry import internal_flow_geom_to_indices_weights_qgis


def build_internal_flow_forcing_qgis(
    *,
    mesh_data,
    have_qgis_core: bool,
    internal_flow_layer_combo,
    combo_layer_fn: Callable[[object, str], Optional[object]],
    requested_field_name: str,
    iter_project_layers_fn: Callable[[], Iterable[object]],
    mesh_cell_centroids_fn: Callable[[], Tuple[np.ndarray, np.ndarray]],
    parse_hydrograph_text_fn: Callable[[str], Optional[Tuple[np.ndarray, np.ndarray]]],
    hydrograph_from_layer_fn: Callable[..., Optional[Tuple[np.ndarray, np.ndarray]]],
    qgs_vector_layer_cls,
    qgs_wkb_types,
    qgs_geometry_cls,
    qgs_pointxy_cls,
    log_fn: Callable[[str], None],
) -> Optional[Dict[str, object]]:
    """
    build internal flow forcing qgis.

    Parameters
    ----------
    mesh_data
        Description of mesh_data.
    have_qgis_core : bool
        Description of have_qgis_core.
    internal_flow_layer_combo
        Description of internal_flow_layer_combo.
    combo_layer_fn : Callable[[object, str], Optional[object]]
        Description of combo_layer_fn.
    requested_field_name : str
        Description of requested_field_name.
    iter_project_layers_fn : Callable[[], Iterable[object]]
        Description of iter_project_layers_fn.
    mesh_cell_centroids_fn : Callable[[], Tuple[np.ndarray, np.ndarray]]
        Description of mesh_cell_centroids_fn.
    parse_hydrograph_text_fn : Callable[[str], Optional[Tuple[np.ndarray, np.ndarray]]]
        Description of parse_hydrograph_text_fn.
    hydrograph_from_layer_fn : Callable[..., Optional[Tuple[np.ndarray, np.ndarray]]]
        Description of hydrograph_from_layer_fn.
    qgs_vector_layer_cls
        Description of qgs_vector_layer_cls.
    qgs_wkb_types
        Description of qgs_wkb_types.
    qgs_geometry_cls
        Description of qgs_geometry_cls.
    qgs_pointxy_cls
        Description of qgs_pointxy_cls.
    log_fn : Callable[[str], None]
        Description of log_fn.

    Returns
    -------
    Optional[Dict[str, object]]
    """
    if mesh_data is None or not have_qgis_core:
        return None
    if internal_flow_layer_combo is None:
        return None

    def _is_vector_layer(layer_obj: object) -> bool:
        """is vector layer"""
        return isinstance(layer_obj, qgs_vector_layer_cls)

    def _layer_name(layer_obj: object) -> str:
        """layer name"""
        return str(layer_obj.name())

    def _layer_id(layer_obj: object) -> str:
        """layer id"""
        return str(layer_obj.id())

    def _hydrograph_from_layer_with_id(layer_obj, hid: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """hydrograph from layer with id"""
        return hydrograph_from_layer_fn(layer_obj, hydrograph_id=hid, bc_type=None)

    def _geometry_to_indices_weights(geom, cx_local: np.ndarray, cy_local: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """geometry to indices weights"""
        return internal_flow_geom_to_indices_weights_qgis(
            geom,
            cx_local,
            cy_local,
            qgs_wkb_types=qgs_wkb_types,
            qgs_geometry_cls=qgs_geometry_cls,
            qgs_pointxy_cls=qgs_pointxy_cls,
        )

    lyr = combo_layer_fn(internal_flow_layer_combo, "vector")
    if lyr is None:
        return None

    fields = set(lyr.fields().names())
    field_name = resolve_internal_flow_field_name(requested_field_name, fields)
    requested_field = str(requested_field_name or "q_cms").strip() or "q_cms"
    if field_name is None:
        log_fn(f"Internal flow layer '{_layer_name(lyr)}' missing flow field '{requested_field}'; skipping internal sources.")
        return None

    hydro_field = first_matching_field(fields, ("hydrograph", "hydrograph_text", "hydro", "hg"))
    hgid_field = "hydrograph_id" if "hydrograph_id" in fields else None
    hlyr_field = "hydrograph_layer" if "hydrograph_layer" in fields else None

    hydro_lookup: Dict[str, str] = {}
    if hgid_field is not None:
        hydro_layers = [
            hlyr
            for hlyr in iter_project_layers_fn()
            if _is_vector_layer(hlyr) and _layer_name(hlyr).lower() in ("swe2d_hydrographs",)
        ]
        if hydro_layers:
            hlyr = hydro_layers[0]
            hfields = set(hlyr.fields().names())
            if "hydrograph_id" in hfields and "hydrograph" in hfields:
                hydro_lookup = build_hydrograph_lookup_from_features(
                    hlyr.getFeatures(),
                    id_field="hydrograph_id",
                    text_field="hydrograph",
                )

    cx, cy = mesh_cell_centroids_fn()

    def _resolve_layer_hydrograph(ft, ref_layer: str, hid: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """resolve layer hydrograph"""
        return resolve_layer_hydrograph_for_feature(
            ft=ft,
            ref_layer=ref_layer,
            hid=hid,
            hydro_field=hydro_field,
            iter_layers_fn=iter_project_layers_fn,
            is_vector_layer_fn=_is_vector_layer,
            layer_name_fn=_layer_name,
            layer_id_fn=_layer_id,
            hydrograph_from_layer_fn=_hydrograph_from_layer_with_id,
        )

    forcing_data = build_internal_flow_forcing_from_features(
        features=lyr.getFeatures(),
        field_name=field_name,
        hydro_field=hydro_field,
        hgid_field=hgid_field,
        hlyr_field=hlyr_field,
        hydro_lookup=hydro_lookup,
        cx=cx,
        cy=cy,
        parse_hydrograph_text_fn=parse_hydrograph_text_fn,
        resolve_layer_hydrograph_fn=_resolve_layer_hydrograph,
        geometry_to_indices_weights_fn=_geometry_to_indices_weights,
    )

    if forcing_data is None:
        return None

    base_q, dynamic_terms, assigned, dynamic_assigned = forcing_data

    log_fn(
        f"Internal flow sources mapped from layer '{_layer_name(lyr)}': features={assigned}, "
        f"timeseries_features={dynamic_assigned}, static_total_Q={float(np.sum(base_q)):.6f} cms"
    )
    return {
        "base_q_cms": base_q,
        "dynamic_terms": dynamic_terms,
        "layer_name": _layer_name(lyr),
    }
