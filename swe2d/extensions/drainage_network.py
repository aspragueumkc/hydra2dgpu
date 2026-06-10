"""Urban drainage integration for SWE2D — GPU-only computational path.

The class SWE2DUrbanDrainageModule serves as a config/state carrier for the
native CUDA drainage solver.  All hydraulic computation (EGL, diffusion-wave,
dynamic-wave, HDS-5 culvert, inlet/outfall exchange) runs on-device.

The Python methods that survive here are only those that the GPU coupling
controller still calls at runtime:
  - _adaptive_substep_count       — determines substep count for GPU iterative mode
  - _use_simplified_link_model    — used by _adaptive_substep_count for DYNAMIC mode
  - _node_area_m2                 — lookup helper for _adaptive_substep_count
  - _adaptive_depth_fraction      — substep heuristic
  - _adaptive_wave_courant        — substep heuristic
  - _max_adaptive_substeps        — substep cap
"""

from __future__ import annotations

import math
from typing import Dict

from swe2d import units as _u
from swe2d.extensions.extension_models import (
    DrainageCouplingEngine,
    DrainageLink,
    DrainageNode,
    DrainageSolverMode,
    InletExchange,
    OutfallExchange,
    PipeEndExchange,
    PipeNetworkConfig,
)


class SWE2DUrbanDrainageModule(DrainageCouplingEngine):
    """
    Urban drainage solver: 2D surface <-> 1D pipe-network coupling.

    All hydraulics run on-device via the native CUDA module.  This Python
    class holds the PipeNetworkConfig and node/link state, and provides
    helper methods used by the GPU coupling controller to determine
    sub-stepping and solver parameters.
    """

    def _node_area_m2(self, node_id: str) -> float:
        return max(1.0, float(self._node_area.get(node_id, 50.0)))

    def _use_simplified_link_model(self, link: DrainageLink) -> bool:
        t = str(link.link_type or "").strip().lower()
        if t in {"lateral_simple", "lateral", "short_lateral"}:
            return True
        md = link.metadata or {}
        return bool(md.get("simplified", False) or md.get("ignore_inertia", False))

    def _adaptive_depth_fraction(self) -> float:
        return min(1.0, max(1.0e-3, float(getattr(self.cfg, "adaptive_depth_fraction", 0.2))))

    def _adaptive_wave_courant(self) -> float:
        return max(1.0e-3, float(getattr(self.cfg, "adaptive_wave_courant", 0.5)))

    def _max_adaptive_substeps(self) -> int:
        return max(1, int(getattr(self.cfg, "max_coupling_substeps", 64)))

    def _adaptive_substep_count(self, dt_s: float, solver_mode: DrainageSolverMode) -> int:
        if dt_s <= 0.0 or not self.cfg.nodes:
            return 1

        node_abs_q: Dict[str, float] = {n.node_id: 0.0 for n in self.cfg.nodes}
        dt_limit = float("inf")
        g = max(1.0e-6, float(getattr(self.cfg, "gravity", _u.gravity())))

        for link in self.cfg.links:
            q_est = abs(float(self.state.link_flow.get(link.link_id, 0.0)))
            node_abs_q[link.from_node_id] = node_abs_q.get(link.from_node_id, 0.0) + q_est
            node_abs_q[link.to_node_id] = node_abs_q.get(link.to_node_id, 0.0) + q_est

            if solver_mode == DrainageSolverMode.DYNAMIC and not self._use_simplified_link_model(link):
                diameter = float(link.diameter or link.metadata.get("diameter", 0.0) or 0.0)
                length = max(1.0, float(link.length or 1.0))
                if diameter > 0.0:
                    wave_celerity = math.sqrt(g * max(1.0e-3, diameter))
                    if wave_celerity > 0.0:
                        dt_limit = min(dt_limit, self._adaptive_wave_courant() * length / wave_celerity)

        for node in self.cfg.nodes:
            q_sum = node_abs_q.get(node.node_id, 0.0)
            if q_sum <= 0.0:
                continue
            area = self._node_area_m2(node.node_id)
            max_depth = max(0.0, float(node.max_depth))
            allowed_depth_change = max(1.0e-2, min(5.0e-2, self._adaptive_depth_fraction() * max(max_depth, 0.1)))
            dt_limit = min(dt_limit, area * allowed_depth_change / q_sum)

        if not math.isfinite(dt_limit) or dt_limit <= 0.0:
            return 1
        return min(self._max_adaptive_substeps(), max(1, int(math.ceil(dt_s / dt_limit))))


__all__ = [
    "DrainageNode",
    "DrainageLink",
    "DrainageSolverMode",
    "InletExchange",
    "OutfallExchange",
    "PipeEndExchange",
    "PipeNetworkConfig",
    "SWE2DUrbanDrainageModule",
]
