# Git Safety — Destructive File Operations

- **NEVER** run `git checkout -- <file>` (or any command that overwrites working-tree files with committed versions) without first checking for uncommitted changes across the **entire repo**:
  ```bash
  git status --short
  ```
- If ANY file shows `M` (modified), `A` (added), `D` (deleted), or `??` (untracked) that could be relevant, do NOT use destructive git commands. Use manual `edit` tool edits to revert only specific changes instead.
- `git checkout -- <file>` silently discards ALL uncommitted changes in `<file>` — including changes the agent didn't make. There is no undo.
- QGIS holds modules in memory for the session, and stale `.pyc` files cause invisible failures (wrong arity, missing attributes, silent fallback paths).
- When in doubt, purge `__pycache__` before asking the user to restart.
