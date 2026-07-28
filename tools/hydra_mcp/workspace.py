"""Workspace path containment for the HYDRA MCP server.

The MCP server must not accept filesystem paths that escape the active
workspace.  ``WorkspacePath`` rejects paths that:

* contain ``..`` segments that escape the workspace root after
  ``Path.resolve()``;
* resolve through a symlink whose target is outside the workspace root
  (``Path.resolve(strict=False)`` follows symlinks, so we explicitly
  ``lstat`` each ancestor to detect them).

The workspace root is set when the tool starts and is the canonical
repository root (``_REPO_ROOT``).  Callers may override it via
``HYDRA_MCP_WORKSPACE_ROOT`` for tests / sandboxed deployments.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Union


class WorkspacePathError(ValueError):
    """Raised when a path escapes the workspace root."""


class WorkspacePath:
    """Validate that a user-supplied path stays inside *workspace_root*."""

    def __init__(self, workspace_root: Union[str, Path]):
        self.root = Path(workspace_root).resolve()

    def resolve_under(self, path: Union[str, Path]) -> Path:
        """Resolve *path* relative to the workspace; reject escapes.

        Relative paths are resolved against the workspace root.  Absolute
        paths must already live inside the workspace.  Symlinks whose
        targets point outside the workspace are rejected — even if the
        symlink itself is inside the workspace — because that is the
        common exfiltration / RCE primitive the MCP server must defend
        against.

        Raises:
            WorkspacePathError: when *path* escapes via ``..`` or a
                symlink, or when it does not exist (after resolution).
        """
        if path is None or str(path) == "":
            raise WorkspacePathError("path is empty")

        p_raw = Path(str(path))
        # Treat relative paths as workspace-relative — the workspace is the
        # canonical root and a relative input should never be interpreted
        # against the MCP server's CWD (which may be / or /tmp).
        candidate = (self.root / p_raw) if not p_raw.is_absolute() else p_raw

        # Reject textual ".." escapes BEFORE resolution so the caller sees a
        # precise error rather than "resolved path is outside workspace".
        for part in candidate.parts:
            if part == "..":
                raise WorkspacePathError(
                    f"path {path!r} contains '..' and escapes the workspace "
                    f"root {self.root}"
                )

        # ``Path.resolve(strict=False)`` follows symlinks and produces an
        # absolute path even if the file does not exist.  We require the
        # target to actually exist (the MCP tools operate on real files)
        # so a missing file is also an error.
        try:
            resolved = candidate.resolve(strict=True)
        except (FileNotFoundError, RuntimeError) as exc:
            raise WorkspacePathError(
                f"path not found: {path!r} does not resolve to an existing "
                f"file under workspace root {self.root}: {exc}"
            ) from exc

        # Refuse symlinks pointing outside the workspace: lstat the
        # original (pre-resolve) target, and if it is a symlink, check
        # that the resolved target is still under the workspace root.
        try:
            if candidate.is_symlink():
                target = candidate.resolve(strict=False)
                if not self._is_inside(target):
                    raise WorkspacePathError(
                        f"path {path!r} is a symlink whose target "
                        f"{str(target)!r} is outside the workspace root "
                        f"{self.root}"
                    )
        except OSError as exc:
            raise WorkspacePathError(
                f"path {path!r} cannot be lstat'ed: {exc}"
            ) from exc

        if not self._is_inside(resolved):
            raise WorkspacePathError(
                f"path {path!r} resolves to {str(resolved)!r} which is "
                f"outside the workspace root {self.root}"
            )

        return resolved

    def _is_inside(self, path: Path) -> bool:
        """True if *path* is the workspace root or a descendant of it."""
        try:
            path.relative_to(self.root)
            return True
        except ValueError:
            return False


def default_workspace_root() -> Path:
    """Return the workspace root for the running MCP server.

    Honours the ``HYDRA_MCP_WORKSPACE_ROOT`` env override (used by tests
    and sandboxed deployments); otherwise resolves the canonical repo
    root from ``tools/hydra_mcp/``'s position in the source tree.
    """
    override = os.environ.get("HYDRA_MCP_WORKSPACE_ROOT")
    if override:
        return Path(override).resolve()
    # tools/hydra_mcp/workspace.py -> repo root is three levels up.
    return Path(__file__).resolve().parents[2]


def default_workspace() -> WorkspacePath:
    """Return a ``WorkspacePath`` bound to the default workspace root."""
    return WorkspacePath(default_workspace_root())
