# Studio UI Architecture & Structural Changes

- **`.ui` files are the source of truth** for widget layout and properties.
  Use Qt Designer to edit them. Only create widgets programmatically when
  they cannot live in a `.ui` file (dynamically populated combos, etc.).
- When making structural changes (new tabs, new forms, new feature toggles,
  widget moves/renames), follow the checklists in
  `docs/STUDIO_UI_ARCHITECTURE.md`.
- After any `.ui` change, run:
  ```bash
  python tools/ui_bind_sync.py forms/swe2d_<name>.ui <py_files> --missing
  ```
  to verify all widgets have bindings and no orphans remain.
- Feature toggles touch 3 files: feature flags dict + keyword function in
  `SWE2DWorkbenchStudioDialog`, and menu/toolbar actions in `studio_host_methods.py`.
  All three must be updated together.
