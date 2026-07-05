"""Hydraulic structures skeleton module for SWE2D.

All hydraulic computations run on-device via GPU kernels.
This module provides the configuration wrapper only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from swe2d.extensions.extension_models import (
    HydraulicStructure,
    HydraulicStructureConfig,
    HydraulicStructureEngine,
    StructureType,
)


def build_structures_config_from_json(
    structures_data: Optional[Any],
    n_cells: int,
) -> Optional[HydraulicStructureConfig]:
    """Build HydraulicStructureConfig from a JSON object or array.

    Supports two input forms:

    Form A — dict with metadata wrapper (recommended):
    {
        "enabled": true,
        "control_interval_s": 1.0,
        "controller_name": "none",
        "structures": [
            {"id": "s1", "type": "culvert", ...},
            ...
        ]
    }

    Note: ``gravity`` is not read from this dict — the coupling controller
    derives it from the mesh CRS via ``_u.gravity()``.

    Form B — bare list of structure entries:
    [
        {"id": "s1", "type": "culvert", ...},
        ...
    ]

    Each entry:
    {
        "id": "s1",
        "type": "culvert",
        "upstream_cell": 100,
        "downstream_cell": 101,
        "crest_elev": 5.0,
        "metadata": {"diameter": 1.0, "length": 20.0, ...}
    }
    """
    if not structures_data:
        return None

    _from_dict = False
    if isinstance(structures_data, dict):
        struct_list = structures_data.get("structures")
        if struct_list is None:
            return None
        _from_dict = True
        _enabled = bool(structures_data.get("enabled", True))
        _control_interval_s = float(structures_data.get("control_interval_s", 1.0))
        _controller_name = str(structures_data.get("controller_name", "none"))
        structures_data = struct_list

    if not isinstance(structures_data, list):
        raise TypeError(
            f"build_structures_config_from_json: expected a list of structure dicts, "
            f"got {type(structures_data).__name__}. "
            f"If passing a dict, it must contain a 'structures' key with the list."
        )

    type_map = {
        "weir": StructureType.WEIR,
        "culvert": StructureType.CULVERT,
        "gate": StructureType.GATE,
        "bridge": StructureType.BRIDGE,
        "pump": StructureType.PUMP,
    }

    structs = []
    for s in structures_data:
        stype = type_map.get(str(s.get("type", "")).lower(), StructureType.CULVERT)
        meta = dict(s.get("metadata", {}) or {})
        # Lift top-level keys into metadata for the coupling controller
        for k in ("diameter", "length", "width", "height", "roughness_n",
                   "coefficient", "cd", "opening", "max_flow",
                   "culvert_code", "culvert_shape", "culvert_rise", "culvert_span",
                   "culvert_area", "culvert_barrels", "culvert_slope",
                   "inlet_invert_elev", "outlet_invert_elev",
                   "entrance_loss_k", "exit_loss_k",
                   "embankment_enabled", "embankment_crest_elev",
                   "embankment_overflow_width", "embankment_weir_coeff",
                   "q_pump"):
            if k in s:
                meta[k] = s[k]
        structs.append(HydraulicStructure(
            structure_id=str(s.get("id", f"s_{len(structs)}")),
            structure_type=stype,
            upstream_cell=int(s.get("upstream_cell", 0)),
            downstream_cell=int(s.get("downstream_cell", 0)),
            crest_elev=float(s.get("crest_elev", 0.0)),
            metadata=meta,
        ))

    cfg = HydraulicStructureConfig(structures=structs)
    # Always enable when the user supplied a non-empty structure list.  Without
    # this, the bare-list form (Form B) leaves HydraulicStructureConfig.enabled
    # at its default False, and the runtime silently skips structure coupling.
    if _from_dict:
        # Dict form: respect the explicit ``enabled`` key (default True).
        cfg.enabled = _enabled and (len(structs) > 0)
        cfg.control_interval_s = _control_interval_s
        cfg.controller_name = _controller_name
    else:
        # Bare-list form: enable whenever structures were supplied.
        cfg.enabled = len(structs) > 0
    return cfg


class SWE2DStructureModule(HydraulicStructureEngine):
    """Structure dispatcher — config only.  All hydraulic calcs run on-device."""

    def __init__(self, cfg: HydraulicStructureConfig, model_to_ft: float = 1.0):
        super().__init__(cfg)
        _ = model_to_ft






__all__ = [
    "SWE2DStructureModule",
    "build_structures_config_from_json",
]
