"""Rain-on-grid skeleton module for SWE2D."""

from __future__ import annotations

from typing import List

from swe2d.extensions.extension_models import RainFieldConfig, RainfallSourceEngine


class SWE2DRainfallModule(RainfallSourceEngine):
    """Adapter alias for future solver-managed rainfall source integration."""

    def cell_source_term(self, t_seconds: float, n_cells: int) -> List[float]:
        return self.sample_cell_rain(t_seconds=t_seconds, n_cells=n_cells)


__all__ = ["RainFieldConfig", "SWE2DRainfallModule"]
