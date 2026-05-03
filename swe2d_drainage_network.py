"""Urban drainage and SWMM-style coupling skeleton for SWE2D."""

from __future__ import annotations

from typing import Dict, List, Sequence

from swe2d_extensions import (
    DrainageCouplingEngine,
    DrainageLink,
    DrainageNode,
    InletExchange,
    PipeNetworkConfig,
)


class SWE2DUrbanDrainageModule(DrainageCouplingEngine):
    """
    Placeholder module for 2D surface <-> 1D network coupling.

    Roadmap targets:
    - CPU reference coupling with SWMM-compatible equations
    - optional CUDA acceleration for conduit/link solves
    - two-way exchange at inlets, outfalls, surcharge nodes
    """

    def solve_network_step(self, dt: float) -> Dict[str, float]:
        # TODO: integrate pipe/link momentum and continuity updates.
        _ = dt
        return {}

    def apply_surface_exchange(self, dt: float, cell_wse: Sequence[float]) -> List[float]:
        sinks, sources = self.exchange_step(dt=dt, cell_wse=cell_wse)
        if not sinks and not sources:
            return [0.0] * len(cell_wse)
        combined = [0.0] * len(cell_wse)
        for i, v in enumerate(sinks):
            if i < len(combined):
                combined[i] -= v
        for i, v in enumerate(sources):
            if i < len(combined):
                combined[i] += v
        return combined


__all__ = [
    "DrainageNode",
    "DrainageLink",
    "InletExchange",
    "PipeNetworkConfig",
    "SWE2DUrbanDrainageModule",
]
