"""Core solver/domain exports for incremental package migration."""

from swe2d.runtime.backend import SWE2DBackend, swe2d_available, swe2d_gpu_available
from swe2d.runtime.coupling import SWE2DCouplingController, pack_coupling_soa
from swe2d.extensions.drainage_network import SWE2DUrbanDrainageModule
from swe2d.extensions.extension_models import *  # noqa: F403
from swe2d.extensions.structures import SWE2DStructureModule
