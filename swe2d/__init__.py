"""SWE2D package namespace (incremental migration surface)."""

from swe2d.core import (
    SWE2DBackend,
    SWE2DCouplingController,
    SWE2DUrbanDrainageModule,
    SWE2DStructureModule,
    swe2d_available,
    swe2d_gpu_available,
)
from swe2d import units

__all__ = [
    "SWE2DBackend",
    "SWE2DCouplingController",
    "SWE2DUrbanDrainageModule",
    "SWE2DStructureModule",
    "swe2d_available",
    "swe2d_gpu_available",
    "units",
]
