from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from swe2d.extensions.extension_models import HydraulicStructureConfig
from swe2d.mesh.bridge_stacked_mesh import (
    BridgeStackedPlan,
    bridge_specs_from_structure_config,
    build_bridge_stacked_plan,
)


def build_bridge_stacked_plans_for_runtime(
    mesh_data: Optional[Dict[str, np.ndarray]],
    hydraulic_structures_cfg: Optional[HydraulicStructureConfig],
    log_fn: Optional[Callable[[str], None]] = None,
) -> List[BridgeStackedPlan]:
    if mesh_data is None or hydraulic_structures_cfg is None:
        return []

    specs = bridge_specs_from_structure_config(hydraulic_structures_cfg)
    if not specs:
        return []

    plans = build_bridge_stacked_plan(mesh_data, specs)
    if log_fn is not None:
        total_cells = int(sum(int(p.selected_cells.size) for p in plans))
        log_fn(
            "Bridge stacked-plan mapping ready: "
            f"bridges={len(plans)}, selected_cells={total_cells}"
        )
    return plans


def bridge_stacked_source_scale(plan: BridgeStackedPlan) -> float:
    if plan.selected_cells.size <= 0:
        return 1.0
    opening_scale = float(np.clip(np.mean(np.asarray(plan.opening_fraction, dtype=np.float64)), 0.0, 1.0))
    under_layers = int(np.sum(np.asarray(plan.layer_role, dtype=np.int32) == 0))
    over_layers = int(np.sum(np.asarray(plan.layer_role, dtype=np.int32) == 2))
    total_layers = max(under_layers + over_layers, 1)
    layer_scale = (float(under_layers) + 0.5 * float(over_layers)) / float(total_layers)
    return float(np.clip(opening_scale * layer_scale, 0.0, 1.0))


def apply_bridge_stacked_source_weight(source_rate: np.ndarray, plan: BridgeStackedPlan) -> np.ndarray:
    weighted = np.asarray(source_rate, dtype=np.float64).copy()
    weighted *= bridge_stacked_source_scale(plan)
    return weighted


def apply_bridge_stacked_phase3_source_weight(
    source_rate: np.ndarray,
    plan: BridgeStackedPlan,
    cell_area: np.ndarray,
) -> np.ndarray:
    """Apply Phase 3 bridge coupling: attenuation + spatial redistribution.

    This path preserves global conservation by redistributing signed bridge
    source volumes over stacked-plan corridor cells while keeping total source
    and sink volumes identical to the attenuated helper output.
    """

    src = np.asarray(source_rate, dtype=np.float64).copy()
    area = np.asarray(cell_area, dtype=np.float64).ravel()
    if src.ndim != 1:
        src = src.ravel()
    if src.size == 0 or area.size != src.size:
        return src

    sel = np.asarray(plan.selected_cells, dtype=np.int32).ravel()
    if sel.size == 0:
        return apply_bridge_stacked_source_weight(src, plan)
    valid = (sel >= 0) & (sel < src.size)
    sel = sel[valid]
    if sel.size == 0:
        return apply_bridge_stacked_source_weight(src, plan)

    # Keep the established bridge attenuation semantics as the first-stage
    # scaling, then apply spatial shaping over the bridge corridor.
    src = apply_bridge_stacked_source_weight(src, plan)
    q = src * np.maximum(area, 1.0e-12)
    pos_total = float(np.sum(np.maximum(q, 0.0)))
    neg_total = float(-np.sum(np.minimum(q, 0.0)))
    if pos_total <= 0.0 and neg_total <= 0.0:
        return src

    s = np.asarray(plan.streamwise_m, dtype=np.float64).ravel()
    if s.size != sel.size:
        s = np.linspace(0.0, 1.0, sel.size, dtype=np.float64)
    s_min = float(np.nanmin(s)) if s.size > 0 else 0.0
    s_max = float(np.nanmax(s)) if s.size > 0 else 1.0
    if not np.isfinite(s_min) or not np.isfinite(s_max) or (s_max - s_min) <= 1.0e-12:
        s_norm = np.zeros(sel.size, dtype=np.float64)
    else:
        s_norm = (s - s_min) / max(s_max - s_min, 1.0e-12)
    up_mask = s_norm <= 0.5
    dn_mask = ~up_mask
    if not np.any(up_mask) or not np.any(dn_mask):
        ord_idx = np.argsort(s_norm)
        cut = max(1, int(sel.size // 2))
        up_mask = np.zeros(sel.size, dtype=bool)
        up_mask[ord_idx[:cut]] = True
        dn_mask = ~up_mask

    opening = np.asarray(plan.opening_fraction, dtype=np.float64).ravel()
    if opening.size != sel.size:
        opening = np.ones(sel.size, dtype=np.float64)
    opening = np.clip(opening, 0.0, 1.0)
    # Small floor keeps redistribution robust when all opening fractions are 0.
    base_w = np.maximum(opening, 1.0e-6)

    q_sel = q[sel]
    neg_up = float(-np.sum(np.minimum(q_sel[up_mask], 0.0))) if np.any(up_mask) else 0.0
    neg_dn = float(-np.sum(np.minimum(q_sel[dn_mask], 0.0))) if np.any(dn_mask) else 0.0
    pos_up = float(np.sum(np.maximum(q_sel[up_mask], 0.0))) if np.any(up_mask) else 0.0
    pos_dn = float(np.sum(np.maximum(q_sel[dn_mask], 0.0))) if np.any(dn_mask) else 0.0

    target_neg_mask = up_mask if neg_up >= neg_dn else dn_mask
    target_pos_mask = dn_mask if pos_dn >= pos_up else up_mask

    q_new = np.zeros_like(q)
    if neg_total > 0.0:
        w_neg = base_w[target_neg_mask]
        wsum = float(np.sum(w_neg))
        if not np.isfinite(wsum) or wsum <= 0.0:
            w_neg = np.ones(int(np.count_nonzero(target_neg_mask)), dtype=np.float64)
            wsum = float(np.sum(w_neg))
        q_new[sel[target_neg_mask]] += -neg_total * (w_neg / max(wsum, 1.0e-12))

    if pos_total > 0.0:
        w_pos = base_w[target_pos_mask]
        wsum = float(np.sum(w_pos))
        if not np.isfinite(wsum) or wsum <= 0.0:
            w_pos = np.ones(int(np.count_nonzero(target_pos_mask)), dtype=np.float64)
            wsum = float(np.sum(w_pos))
        q_new[sel[target_pos_mask]] += pos_total * (w_pos / max(wsum, 1.0e-12))

    out = q_new / np.maximum(area, 1.0e-12)
    return np.asarray(out, dtype=np.float64)


__all__ = [
    "build_bridge_stacked_plans_for_runtime",
    "bridge_stacked_source_scale",
    "apply_bridge_stacked_source_weight",
    "apply_bridge_stacked_phase3_source_weight",
]
