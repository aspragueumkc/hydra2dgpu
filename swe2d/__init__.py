"""SWE2D package namespace."""

from swe2d.runtime.backend import SWE2DBackend, swe2d_available, swe2d_gpu_available
from swe2d.runtime.coupling import SWE2DCouplingController
from swe2d.extensions.drainage_network import SWE2DUrbanDrainageModule
from swe2d.extensions.structures import SWE2DStructureModule
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
