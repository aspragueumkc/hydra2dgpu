"""hydra-swe2d: CUDA-accelerated 2D SWE solver (HYDRA2DGPU backend).

This package is built and shipped as a platform-specific wheel by
cibuildwheel (see .github/workflows/build-wheels.yml). At runtime the
actual entry point is the compiled hydra_swe2d extension module that
scikit-build-core installs next to this __init__.py.
"""
from __future__ import annotations

try:
    from hydra_swe2d.hydra_swe2d import *  # noqa: F401,F403
except ImportError as _e:  # pragma: no cover
    raise ImportError(
        "hydra_swe2d native extension not found. "
        "Reinstall via: pip install --force-reinstall hydra-swe2d"
    ) from _e

__version__ = "1.2.0"
