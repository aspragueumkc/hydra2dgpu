"""Hydraulic structures skeleton module for SWE2D.

All hydraulic computations run on-device via GPU kernels.
This module provides the configuration wrapper only.
"""

from __future__ import annotations

from swe2d.extensions.extension_models import (
    HydraulicStructureConfig,
    HydraulicStructureEngine,
)


class SWE2DStructureModule(HydraulicStructureEngine):
    """Structure dispatcher — config only.  All hydraulic calcs run on-device."""

    def __init__(self, cfg: HydraulicStructureConfig, model_to_ft: float = 1.0):
        super().__init__(cfg)
        _ = model_to_ft






__all__ = [
    "StructureType",
    "HydraulicStructure",
    "HydraulicStructureConfig",
    "SWE2DStructureModule",
]
