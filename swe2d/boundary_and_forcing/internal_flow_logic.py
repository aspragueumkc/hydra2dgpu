from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np


Hydrograph = Tuple[np.ndarray, np.ndarray]
DynamicTerm = Tuple[np.ndarray, np.ndarray, Hydrograph]


def resolve_internal_flow_field_name(requested_field: str, fields: set) -> Optional[str]:
    field_name = str(requested_field or "q_cms").strip() or "q_cms"
    if field_name in fields:
        return field_name
    for cand in ("q_cms", "flow_cms", "q", "flow"):
        if cand in fields:
            return cand
    return None


def first_matching_field(fields: set, candidates: Iterable[str]) -> Optional[str]:
    for cand in candidates:
        if cand in fields:
            return cand
    return None


def build_hydrograph_lookup_from_features(
    features,
    id_field: str = "hydrograph_id",
    text_field: str = "hydrograph",
) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for ft in features:
        try:
            hid = str(ft[id_field] or "").strip()
            htxt = str(ft[text_field] or "").strip()
        except Exception:
            continue
        if hid and htxt:
            lookup[hid] = htxt
    return lookup


def resolve_layer_hydrograph_for_feature(
    ft,
    ref_layer: str,
    hid: str,
    hydro_field: Optional[str],
    iter_layers_fn: Callable[[], Iterable[object]],
    is_vector_layer_fn: Callable[[object], bool],
    layer_name_fn: Callable[[object], str],
    layer_id_fn: Callable[[object], str],
    hydrograph_from_layer_fn: Callable[[object, str], Optional[Hydrograph]],
) -> Optional[Hydrograph]:
    layer_ref = ref_layer or (str(ft[hydro_field] or "").strip() if hydro_field is not None else "")
    if not layer_ref:
        return None

    target_layer = None
    for hlyr in iter_layers_fn():
        if not is_vector_layer_fn(hlyr):
            continue
        if layer_name_fn(hlyr) == layer_ref or layer_id_fn(hlyr) == layer_ref:
            target_layer = hlyr
            break

    if target_layer is None:
        return None
    return hydrograph_from_layer_fn(target_layer, hid)


def build_internal_flow_forcing_from_features(
    features,
    field_name: str,
    hydro_field: Optional[str],
    hgid_field: Optional[str],
    hlyr_field: Optional[str],
    hydro_lookup: Dict[str, str],
    cx: np.ndarray,
    cy: np.ndarray,
    parse_hydrograph_text_fn: Callable[[str], Optional[Hydrograph]],
    resolve_layer_hydrograph_fn: Callable[[object, str, str], Optional[Hydrograph]],
    geometry_to_indices_weights_fn: Callable[[object, np.ndarray, np.ndarray], Optional[Tuple[np.ndarray, np.ndarray]]],
) -> Optional[Tuple[np.ndarray, List[DynamicTerm], int, int]]:
    base_q = np.zeros(cx.shape[0], dtype=np.float64)
    dynamic_terms: List[DynamicTerm] = []
    assigned = 0
    dynamic_assigned = 0

    for ft in features:
        try:
            geom = ft.geometry()
        except Exception:
            continue
        if geom is None or geom.isEmpty():
            continue

        q_cms = 0.0
        try:
            q_cms = float(ft[field_name])
        except Exception:
            q_cms = 0.0
        if not np.isfinite(q_cms):
            q_cms = 0.0

        hg = None
        raw_h = str(ft[hydro_field] or "").strip() if hydro_field is not None else ""
        ref_layer = str(ft[hlyr_field] or "").strip() if hlyr_field is not None else ""
        hid = str(ft[hgid_field] or "").strip() if hgid_field is not None else ""
        if not raw_h and hid and hid in hydro_lookup:
            raw_h = hydro_lookup[hid]

        if raw_h:
            try:
                hg = parse_hydrograph_text_fn(raw_h)
            except Exception:
                hg = None

        if hg is None and (ref_layer or (hgid_field is not None and hid)):
            try:
                hg = resolve_layer_hydrograph_fn(ft, ref_layer, hid)
            except Exception:
                hg = None

        if abs(q_cms) <= 0.0 and hg is None:
            continue

        mapped = geometry_to_indices_weights_fn(geom, cx, cy)
        if mapped is None:
            continue
        idx_arr, wt_arr = mapped

        if abs(q_cms) > 0.0:
            base_q[idx_arr] += q_cms * wt_arr
        if hg is not None:
            dynamic_terms.append((idx_arr, wt_arr, hg))
            dynamic_assigned += 1
        assigned += 1

    if assigned <= 0:
        return None

    return base_q, dynamic_terms, assigned, dynamic_assigned
