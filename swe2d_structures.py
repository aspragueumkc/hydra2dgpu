"""Hydraulic structures skeleton module for SWE2D."""

from __future__ import annotations

from typing import Dict, Sequence

from swe2d_extensions import (
    HydraulicStructure,
    HydraulicStructureConfig,
    HydraulicStructureEngine,
    StructureType,
)


class SWE2DStructureModule(HydraulicStructureEngine):
    """
    Structure dispatcher for future embedded hydraulic controls.

    Supported skeleton types:
    - weirs
    - culverts
    - gates
    - bridges
    - pumps
    """

    def compute_flux_adjustments(self, dt: float, cell_wse: Sequence[float]) -> Dict[str, float]:
        return self.compute_structure_fluxes(dt=dt, cell_wse=cell_wse)


__all__ = [
    "StructureType",
    "HydraulicStructure",
    "HydraulicStructureConfig",
    "SWE2DStructureModule",
]
