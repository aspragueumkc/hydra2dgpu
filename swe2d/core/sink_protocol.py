"""Protocol for simulation execution callbacks."""

import threading
from typing import Any, Dict, List, Optional, Protocol

import numpy as np


class PermutationResult:
    """Thread-safe holder for mesh permutation results.

    The executor creates an instance, passes it to :meth:`Sink.permutation`,
    and then waits on ``event``.  A GUI sink can emit the permutation to the
    main thread, which fills ``sample_map`` and sets ``event`` when done.  If
    ``sample_map`` is left ``None``, the executor falls back to the
    ``RunContext.sample_map_data`` already in the context.
    """

    def __init__(self) -> None:
        self.event = threading.Event()
        self.sample_map: Optional[List[Dict[str, Any]]] = None
        self.error: str = ""


class Sink(Protocol):
    """Protocol for receiving simulation execution events.

    Implementations can be signal-emitting Qt objects, CLI loggers,
    or test mocks. All callbacks are synchronous and must not block
    for extended periods.
    """

    snapshot_request_event: threading.Event

    def log(self, message: str) -> None:
        """Log a message from the executor."""
        ...

    def progress(self, percent: float, diagnostics: Dict[str, Any]) -> None:
        """Report simulation progress.

        Args:
            percent: Progress percentage (0-100)
            diagnostics: Additional diagnostic information (dt, wet_cells, etc.)
        """
        ...

    def snapshot(self, fields: List[Any]) -> None:
        """Receive a snapshot of the current simulation state.

        Args:
            fields: List containing snapshot data (timesteps, line_ts, etc.)
        """
        ...

    def finished(self, result: Dict[str, Any]) -> None:
        """Called when simulation completes successfully.

        Args:
            result: Dictionary containing simulation results
        """
        ...

    def failed(self, error: str) -> None:
        """Called when simulation fails.

        Args:
            error: Error message describing the failure
        """
        ...

    def permutation(self, cell_perm: np.ndarray, result: PermutationResult) -> None:
        """Notify the consumer of the solver's mesh cell permutation.

        Args:
            cell_perm: The solver cell permutation array.
            result: A holder that the consumer should fill and signal.
        """
        ...

    def backend_ready(self, backend: Any) -> None:
        """Called once the active SWE2DBackend has been built and initialized.

        Used by the GPU Direct Viewer to grab the live solver handle
        (``backend._solver_h``) so the GL render path can register the
        device pointer with the GL texture.  Optional — sinks that don't
        need it (CLI loggers, mocks) don't have to implement it.
        """
        ...

    def request_snapshot(self) -> None:
        """Request that the executor read back a snapshot on the next step.

        Implementations should set ``snapshot_request_event`` so the executor
        can detect the request in a thread-safe way.
        """
        ...

