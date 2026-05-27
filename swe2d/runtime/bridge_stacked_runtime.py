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


__all__ = [
    "build_bridge_stacked_plans_for_runtime",
    "bridge_stacked_source_scale",
    "apply_bridge_stacked_source_weight",
]
