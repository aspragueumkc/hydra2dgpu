# SWE2D Workbench GUI Audit

## Scope
This audit covers `swe2d_workbench_qt.py` and the GUI-related patterns it introduces for the interactive Studio workbench.

## Summary
`swe2d_workbench_qt.py` is a large monolithic file with mixed responsibilities:
- UI shell / layout construction
- `.ui` file loading and fallback programmatic widget creation
- widget binding and control initialization
- dialog classes and runtime viewers

That mix makes the GUI hard to maintain, difficult to validate, and prone to silent failures when UI structure changes.

## Biggest Issues

### 1. Inconsistent UI binding strategy
- Many `_build_<tab>_page()` methods try to load a `.ui` file and then silently fall back to hand-built widgets.
- Binding methods use `_find_or_create_*()` helpers that create missing widgets on the fly.

Why this is bad:
- It hides broken or mismatched `.ui` files.
- It creates invisible drift between the UI file and the runtime UI.
- It makes the codebase harder to audit and test.

Example:
- `_populate_boundary_tab_controls()` will create missing widgets if they are not found in the `.ui`.
- `_bind_mesh_tab_controls()` raises on missing controls, but other binders do not.

### 2. Silent exception handling
- There are many `except Exception: pass` blocks around UI loading and widget operations.
- This includes tab page loading, binder setup, layout adjustments, and signal disconnects.

Why this is bad:
- Runtime GUI failures are converted into silent degraded behavior.
- It makes regressions hard to detect.
- It increases the probability of stale / invalid `QWidget` references being used later.

### 3. Dirty fallback code and dead code risk
- The code retains programmatic fallback builders for UI tabs that already have `.ui` files.
- Fallback builders and dynamic object creation are effectively dead unless the `.ui` file fails to load.
- This duplicates UI structure in Python and is a maintenance burden.

### 4. Poor documentation and missing docstrings
- Some tab-build and bind methods have good docstrings, but many do not.
- There is inconsistent documentation for the key UI composition chain.
- `_bind_topology_tab_dynamic_controls()` and many `_bind_*()` methods lack explanations of required invariants.

### 5. Monolithic file and cross-module binding imports
- `swe2d_workbench_qt.py` imports binding logic from `swe2d.workbench.monolith_methods` in many places.
- This undermines the module’s own UI ownership and spreads important GUI wiring across files.

Why this is bad:
- It makes it difficult to know where UI behavior is implemented.
- It breaks the expected separation between UI composition and business logic.

### 6. Hard-coded layout tuning in code
- `_compose_left_pane()` contains manual minimum-width and size-adjust policy adjustments for child widgets.
- `_make_left_controls_compact()` adjusts layout margins and spacing in code.

Why this is bad:
- These are presentation concerns that should live in `.ui` or a dedicated styling helper.
- They make runtime GUI layout fragile and harder to reason about.

## Concrete Suggested Fixes

### A. Enforce `.ui` files as the source of truth
- Remove `_find_or_create_*()` factory logic from runtime binders.
- Make any missing UI control condition an immediate error, not an opportunity to invent a fallback widget.
- Keep fallback builder code only for development / very early bootstrap; do not mix fallback creation with binding.
- Prefer `loadUi()` and explicit `assert widget is not None` checks.

### B. Replace broad `except Exception: pass` with narrow handlers and logging
- Catch only expected exceptions, such as `FileNotFoundError` or `AttributeError`.
- Log failures with context and include the affected widget or UI file path.
- Use fail-fast behavior for malformed UI rather than silent degradation.

### C. Move UI binding logic out of `swe2d_workbench_qt.py` where possible
- Group tab-building and binding into smaller modules per domain:
  - `mesh_tab.py`
  - `map_tab.py`
  - `topology_tab.py`
  - `model_tab.py`
  - `run_tab.py`
- Keep `swe2d_workbench_qt.py` focused on shell creation, pane composition, and dialog orchestration.

### D. Add docstrings to all tab builder/binder entry points
- Document expected object names, required `.ui` IDs, and binding invariants.
- Include a clear comment for every `_bind_<...>()` method explaining whether it requires a `.ui` page or can operate against fallback UI.

### E. Clean up layout tuning code
- Extract presentation helpers for sizing into a dedicated utility or move them into `.ui` stylesheet/layout settings.
- Avoid repeated `findChildren()` loops across the entire left host widget tree.

### F. Audit and remove redundant fallback builders
- If the `.ui` files are stable, consider removing the fallback page builders entirely or keeping them in a separate `ui_fallbacks.py` file that is explicitly opt-in.
- Preserve fallback code only where the plugin must work in editor-less environments, but do not let it become the normal runtime path.

## Recommended Next Actions
1. Create a small `docs/GUI_BINDING_CONTRACT.md` describing:
   - `.ui` file naming rules
   - required object names for each tab
   - binding process and failure policy
2. Refactor `swe2d_workbench_qt.py` into smaller GUI modules.
3. Introduce unit tests or `tools/ui_bind_sync.py` checks that verify expected widget names for each `.ui`.
4. Replace the current dynamic binding factories with explicit `findChild` lookups + runtime assertions.

## High-Priority Risks
- Hidden UI regressions when `.ui` object names change.
- Silent failures during UI load or component binding.
- Confusing maintenance because widget setup is distributed between `.ui`, fallbacks, and `monolith_methods`.

## Note
This audit focused on the GUI composition and binding patterns in `swe2d_workbench_qt.py` rather than runtime simulation or QGIS-specific behavior.
If you want, I can follow this with a second pass to audit the companion extracted UI files and the `tools/ui_bind_sync.py` integration.