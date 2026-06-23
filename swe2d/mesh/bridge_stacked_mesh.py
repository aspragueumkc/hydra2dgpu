from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from swe2d.extensions.extension_models import HydraulicStructureConfig, StructureType
from swe2d.mesh.mesh_runtime_logic import mesh_cell_centroids


@dataclass(frozen=True)
class BridgeStackedGeometrySpec:
    structure_id: str
    p0_xy: Tuple[float, float]
    p1_xy: Tuple[float, float]
    influence_width_m: float
    upstream_buffer_m: float = 0.0
    downstream_buffer_m: float = 0.0
    deck_soffit_elev_m: float = 0.0
    deck_top_elev_m: float = 1.0
    model_top_elev_m: float = 2.0
    under_layers: int = 2
    over_layers: int = 1
    pier_count: int = 0
    pier_width_m: float = 0.0


@dataclass(frozen=True)
class BridgeStackedPlan:
    structure_id: str
    selected_cells: np.ndarray
    streamwise_m: np.ndarray
    transverse_m: np.ndarray
    layer_bottom_m: np.ndarray
    layer_top_m: np.ndarray
    layer_role: np.ndarray
    opening_fraction: np.ndarray
    effective_opening_width_m: float


def _layer_interfaces(spec: BridgeStackedGeometrySpec) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """layer interfaces"""
    soffit = float(spec.deck_soffit_elev_m)
    deck_top = max(soffit + 1.0e-9, float(spec.deck_top_elev_m))
    model_top = max(deck_top + 1.0e-9, float(spec.model_top_elev_m))

    n_under = max(1, int(spec.under_layers))
    n_over = max(1, int(spec.over_layers))

    under_edges = np.linspace(0.0, soffit, n_under + 1, dtype=np.float64)
    over_edges = np.linspace(deck_top, model_top, n_over + 1, dtype=np.float64)

    bottom = np.concatenate([under_edges[:-1], over_edges[:-1]])
    top = np.concatenate([under_edges[1:], over_edges[1:]])
    role = np.concatenate(
        [
            np.zeros(n_under, dtype=np.int32),  # 0=underdeck fluid
            np.full(n_over, 2, dtype=np.int32),  # 2=overdeck fluid
        ]
    )
    return bottom, top, role


def _pier_bands(half_width: float, pier_count: int, pier_width_m: float) -> List[Tuple[float, float]]:
    """pier bands"""
    if pier_count <= 0 or pier_width_m <= 0.0:
        return []

    total_width = 2.0 * half_width
    total_pier = float(pier_count) * float(pier_width_m)
    if total_pier >= total_width - 1.0e-9:
        return [(-half_width, half_width)]

    opening = (total_width - total_pier) / float(pier_count + 1)
    left = -half_width
    out: List[Tuple[float, float]] = []
    for _ in range(pier_count):
        left += opening
        out.append((left, left + pier_width_m))
        left += pier_width_m
    return out


def _opening_fraction(transverse_m: np.ndarray, half_width: float, pier_bands: Sequence[Tuple[float, float]]) -> np.ndarray:
    """opening fraction"""
    frac = np.ones_like(transverse_m, dtype=np.float64)
    frac[np.abs(transverse_m) > half_width] = 0.0
    for a, b in pier_bands:
        in_pier = (transverse_m >= float(a)) & (transverse_m <= float(b))
        frac[in_pier] = 0.0
    return frac


def build_bridge_stacked_plan(
    mesh_data: Dict[str, np.ndarray],
    bridge_specs: Sequence[BridgeStackedGeometrySpec],
) -> List[BridgeStackedPlan]:
    """
    build bridge stacked plan.

    Parameters
    ----------
    mesh_data : Dict[str, np.ndarray]
        Description of mesh_data.
    bridge_specs : Sequence[BridgeStackedGeometrySpec]
        Description of bridge_specs.

    Returns
    -------
    List[BridgeStackedPlan]
    """
    cx, cy = mesh_cell_centroids(mesh_data)
    out: List[BridgeStackedPlan] = []

    for spec in bridge_specs:
        p0 = np.asarray(spec.p0_xy, dtype=np.float64)
        p1 = np.asarray(spec.p1_xy, dtype=np.float64)
        d = p1 - p0
        length = float(np.hypot(d[0], d[1]))
        if length <= 1.0e-12:
            continue

        axis = d / length
        normal = np.asarray([-axis[1], axis[0]], dtype=np.float64)
        pts = np.column_stack([cx, cy]) - p0[None, :]
        s = pts @ axis
        n = pts @ normal

        s_min = -max(0.0, float(spec.upstream_buffer_m))
        s_max = length + max(0.0, float(spec.downstream_buffer_m))
        half_width = 0.5 * max(0.0, float(spec.influence_width_m))
        cell_mask = (s >= s_min) & (s <= s_max) & (np.abs(n) <= half_width)
        selected = np.where(cell_mask)[0].astype(np.int32)

        if selected.size == 0:
            continue

        bottom, top, role = _layer_interfaces(spec)
        bands = _pier_bands(half_width=half_width, pier_count=int(spec.pier_count), pier_width_m=float(spec.pier_width_m))
        opening = _opening_fraction(n[selected], half_width=half_width, pier_bands=bands)
        effective_opening = max(0.0, 2.0 * half_width - max(0.0, float(spec.pier_count)) * max(0.0, float(spec.pier_width_m)))

        out.append(
            BridgeStackedPlan(
                structure_id=str(spec.structure_id),
                selected_cells=selected,
                streamwise_m=s[selected].astype(np.float64),
                transverse_m=n[selected].astype(np.float64),
                layer_bottom_m=bottom,
                layer_top_m=top,
                layer_role=role,
                opening_fraction=opening,
                effective_opening_width_m=float(effective_opening),
            )
        )
    return out


def bridge_specs_from_structure_config(cfg: Optional[HydraulicStructureConfig]) -> List[BridgeStackedGeometrySpec]:
    """
    bridge specs from structure config.

    Parameters
    ----------
    cfg : Optional[HydraulicStructureConfig]
        Description of cfg.

    Returns
    -------
    List[BridgeStackedGeometrySpec]
    """
    if cfg is None or not cfg.enabled or not cfg.structures:
        return []

    specs: List[BridgeStackedGeometrySpec] = []
    for st in cfg.structures:
        if int(st.structure_type) != int(StructureType.BRIDGE):
            continue

        md = dict(st.metadata or {})
        if int(md.get("stacked_enabled", 0)) <= 0:
            continue

        try:
            p0 = (float(md.get("axis_x0")), float(md.get("axis_y0")))
            p1 = (float(md.get("axis_x1")), float(md.get("axis_y1")))
        except Exception:
            continue

        specs.append(
            BridgeStackedGeometrySpec(
                structure_id=str(st.structure_id),
                p0_xy=p0,
                p1_xy=p1,
                influence_width_m=float(md.get("influence_width_m", md.get("width", 0.0))),
                upstream_buffer_m=float(md.get("upstream_buffer_m", 0.0)),
                downstream_buffer_m=float(md.get("downstream_buffer_m", 0.0)),
                deck_soffit_elev_m=float(md.get("deck_soffit_elev", st.crest_elev)),
                deck_top_elev_m=float(md.get("deck_top_elev", st.crest_elev + max(0.1, float(md.get("height", 1.0))))),
                model_top_elev_m=float(md.get("model_top_elev", st.crest_elev + 2.0 * max(0.1, float(md.get("height", 1.0))))),
                under_layers=int(md.get("under_layers", 2)),
                over_layers=int(md.get("over_layers", 1)),
                pier_count=int(md.get("pier_count", 0)),
                pier_width_m=float(md.get("pier_width", 0.0)),
            )
        )
    return specs


__all__ = [
    "BridgeStackedGeometrySpec",
    "BridgeStackedPlan",
    "bridge_specs_from_structure_config",
    "build_bridge_stacked_plan",
]
