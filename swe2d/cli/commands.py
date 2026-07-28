"""Single source of truth for batch subprocess command construction.

The CLI batch runner, the workbench's ``BatchWorker``, and any future
subprocess-driven run flow construct the same ``python -m swe2d.cli …``
command.  This module produces that command from a single ``spec_path``
(i.e. a JSON replay file) plus the standard optional flags, so the
two call sites don't drift on flag spelling, ordering, or the new
``--status-file-path`` rewrite (the docstring guarantee from
``BatchWorker._launch_sim``).

Why ``replay`` instead of ``run``?
----------------------------------

The batch spec is a ``swe2d-replay/1`` JSON (or an inline legacy form),
so the canonical command is ``python -m swe2d.cli replay
--replay-file <path>``.  ``run`` only accepts ``mesh_gpkg`` as its first
positional arg, which requires two separate JSON / GPKG paths; ``replay``
treats the JSON as the entire input and is what the GUI's
"export replay" path emits.

Backward-compat fallback (``run``)
----------------------------------

If the spec is NOT a ``swe2d-replay/1`` payload (legacy inline form),
we fall back to the ``run`` command.  This matches the prior batch
runner behavior and keeps the regression test
``tests/test_batch_runner.py`` green.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from typing import Any, Dict, List


def is_replay_spec(params: Dict[str, Any]) -> bool:
    """Return True if ``params`` is a ``swe2d-replay/1`` payload.

    The replay payload is the canonical output of the GUI's
    ``_build_replay_payload`` and is what the canonical
    ``build_run_context`` accepts via ``swe2d-replay/1`` schema_version.
    """
    return params.get("schema_version") == "swe2d-replay/1"


def build_run_command(
    spec_path: str,
    *,
    status_file_path: str = "",
    status_interval_s: float = 5.0,
    results_gpkg: str = "",
    extra_args: List[str] = None,
) -> List[str]:
    """Build the ``python -m swe2d.cli`` command for a single batch sim.

    Parameters
    ----------
    spec_path : str
        Path to a JSON spec file (the ``swe2d-replay/1`` payload).  The
        same file is fed to the ``replay`` subcommand.
    status_file_path : str
        When non-empty, pass ``--status-file-path`` so the subprocess
        writes a periodic JSON status (matches the docstring guarantee
        of ``BatchWorker._launch_sim`` "poll its status file").
    status_interval_s : float
        Status file write interval in seconds (default 5.0).
    results_gpkg : str
        Optional path for the results GPKG — only honored for the
        legacy ``run`` command path; the ``replay`` path embeds this
        inside the spec.
    extra_args : list[str]
        Additional positional/flag arguments to append verbatim.  Used
        for advanced / test-only paths.

    Returns
    -------
    cmd : list[str]
        The ``argv`` ready to pass to ``subprocess.Popen`` /
        ``subprocess.run``.
    """
    cmd: List[str] = [sys.executable, "-m", "swe2d.cli", "replay",
                       "--replay-file", spec_path]
    if status_file_path:
        cmd.extend(["--status-file-path", status_file_path])
        cmd.extend(["--status-interval", str(status_interval_s)])
    if extra_args:
        cmd.extend(list(extra_args))
    return cmd


def build_run_command_for_params(
    params: Dict[str, Any],
    *,
    results_gpkg: str = "",
    status_file_path: str = "",
    status_interval_s: float = 5.0,
    extra_args: List[str] = None,
) -> List[str]:
    """Build the subprocess command for a single batch sim.

    Two paths:
    - ``swe2d-replay/1`` payload → write a temp JSON file and use
      ``replay --replay-file``.
    - Legacy inline form → use ``run mesh_path params_json`` directly.

    The replay-file is tracked in a module-level list so the caller can
    ``cleanup_replay_files()`` after the batch completes (mirrors the
    batch_worker's cleanup pattern).
    """
    if is_replay_spec(params):
        # Write the spec to a temp file.  The caller is responsible for
        # cleanup via ``cleanup_temp_specs()``.
        tf = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        )
        json.dump(params, tf)
        tf.close()
        _TEMP_SPECS.append(tf.name)
        return build_run_command(
            tf.name,
            status_file_path=status_file_path,
            status_interval_s=status_interval_s,
            extra_args=extra_args,
        )

    # Legacy inline form — pass the mesh path and the JSON string.
    mesh_path = params.get("mesh_path") or params.get("mesh_gpkg", "")
    params_json = json.dumps(params)
    cmd: List[str] = [sys.executable, "-m", "swe2d.cli", "run",
                       mesh_path, params_json]
    if results_gpkg:
        cmd.extend(["--results", results_gpkg])
    if status_file_path:
        cmd.extend(["--status-file-path", status_file_path])
        cmd.extend(["--status-interval", str(status_interval_s)])
    if extra_args:
        cmd.extend(list(extra_args))
    return cmd


# Module-level list of temp spec files written by
# ``build_run_command_for_params``.  Callers should clear it after
# the batch completes; ``cleanup_temp_specs()`` is the supported API.
_TEMP_SPECS: List[str] = []


def cleanup_temp_specs() -> None:
    """Remove all temp spec files created by ``build_run_command_for_params``."""
    for path in _TEMP_SPECS:
        try:
            os.unlink(path)
        except OSError:
            pass
    _TEMP_SPECS.clear()


__all__ = [
    "build_run_command",
    "build_run_command_for_params",
    "cleanup_temp_specs",
    "is_replay_spec",
]
