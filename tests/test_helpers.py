"""
Shared test utilities for HYDRA 2D GPU tests.

Provides:

- ``FallbackTracker`` — detect silent fallback paths by intercepting
  ``logging.warning`` calls and asserting none occurred.
- ``SilentFallbackFailure`` — test failure raised when an unexpected
  fallback is detected.
"""

from __future__ import annotations

import logging
import os
import sys
import traceback
from typing import Callable, Dict, List, Optional
from unittest.mock import patch


class SilentFallbackFailure(AssertionError):
    """Raised when an unexpected silent fallback is detected during a test."""


# ═══════════════════════════════════════════════════════════════════════════════
# FallbackTracker
# ═══════════════════════════════════════════════════════════════════════════════

class FallbackTracker:
    """Context manager + decorator that detects silent fallback paths.

    Intercepts ``logger.warning()`` calls and records those matching
    fallback patterns.  On exit, asserts no unexpected fallbacks occurred.

    Usage as context manager::

        from tests.test_helpers import FallbackTracker

        def test_something(self):
            with FallbackTracker() as ft:
                my_function()
            # ft.fallbacks now contains any warning messages
            self.assertEqual(len(ft.fallbacks), 0,
                f"Unexpected fallbacks: {ft.fallbacks}")

    Usage as decorator::

        @FallbackTracker()
        def test_something():
            my_function()
    """

    # Patterns that indicate a silent fallback rather than a benign warning
    _FALLBACK_PATTERNS = [
        "fallback",
        "failed",
        "could not",
        "unable to",
        "returning default",
        "using best-effort",
        "using best available",
        "no valid",
        "retry",
        "attempt",
        "does not exist",
        "not found",
        "import failed",
        "not available",
        "disabling",
        "skipping",
    ]

    def __init__(
        self,
        logger_name: str = "swe2d",
        fail_on_any_warning: bool = False,
        ignore_patterns: Optional[List[str]] = None,
    ):
        self.logger_name = logger_name
        self.fail_on_any_warning = fail_on_any_warning
        self.ignore_patterns = ignore_patterns or []
        self.fallbacks: List[Dict[str, object]] = []
        self._patches: list = []

    def _is_fallback(self, msg: str) -> bool:
        msg_lower = msg.lower()
        for pat in self._FALLBACK_PATTERNS:
            if pat in msg_lower:
                return True
        return False

    def _is_ignored(self, msg: str) -> bool:
        msg_lower = msg.lower()
        for pat in self.ignore_patterns:
            if pat in msg_lower:
                return True
        return False

    def _intercept_warning(self, logger_instance, original_warning):
        """Return a patched warning method that records fallbacks."""

        def patched_warning(msg, *args, **kwargs):
            formatted = str(msg) % args if args else str(msg)
            is_fallback = self._is_fallback(formatted)
            if is_fallback or self.fail_on_any_warning:
                if not self._is_ignored(formatted):
                    self.fallbacks.append({
                        "message": formatted,
                        "logger": logger_instance.name,
                        "traceback": "".join(
                            traceback.format_stack(limit=6)[:-1]
                        ),
                    })
            return original_warning(msg, *args, **kwargs)

        return patched_warning

    def __enter__(self):
        logger = logging.getLogger(self.logger_name)
        self._patches.append(
            patch.object(logger, "warning",
                         self._intercept_warning(logger, logger.warning))
        )
        self._patches.append(
            patch.object(logger, "error",
                         self._intercept_warning(logger, logger.error))
        )
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, exc_type, _exc_val, _exc_tb):
        for p in self._patches:
            p.stop()
        self._patches.clear()
        if exc_type is None and self.fallbacks:
            raise SilentFallbackFailure(
                f"Detected {len(self.fallbacks)} silent fallback(s):\n"
                + "\n".join(f"  [{f['logger']}] {f['message']}"
                            for f in self.fallbacks)
            )

    def __call__(self, func):
        """Decorator support."""
        import functools

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper


# ═══════════════════════════════════════════════════════════════════════════════
# Environment helpers
# ═══════════════════════════════════════════════════════════════════════════════

def set_test_env() -> None:
    """Set standard test environment variables if not already set."""
    os.environ.setdefault("BACKWATER_SWE2D_DIAG_MODE", "1")
    os.environ.setdefault("BACKWATER_GMSH_VERBOSITY", "0")


def require_native_solver():
    """Skip decorator: require hydra_swe2d native module."""
    import unittest
    try:
        import hydra_swe2d  # noqa: F401
        return unittest.skipIf(False, "")
    except ImportError:
        return unittest.skip("hydra_swe2d not built")


def require_gpu():
    """Skip decorator: require CUDA GPU."""
    import unittest
    try:
        import hydra_swe2d  # noqa: F401
        has_gpu = hydra_swe2d.swe2d_gpu_available()
        return unittest.skipIf(not has_gpu, "CUDA GPU not available")
    except ImportError:
        return unittest.skip("hydra_swe2d not built (GPU check unavailable)")


def require_gmsh():
    """Skip decorator: require gmsh Python package."""
    import unittest
    try:
        import gmsh  # noqa: F401
        return unittest.skipIf(False, "")
    except ImportError:
        return unittest.skip("gmsh not installed")
