# AGENTS
a silent fallback/degradation is the biggest failure you can make in this repo. It is even worse than code that crashes or doesn't run at all. NO SILENT FALLBACKS!

## Available Agent Skills

This repo ships the following domain skills (see `.agents/skills/`):

| Skill | File | When to use |
|-------|------|-------------|
| PyQt5 Desktop Patterns | `.agents/skills/pyqt5-desktop-patterns/SKILL.md` | Studio UI, dock lifecycle, widget bindings, signal safety |
| QGIS Plugin Conventions | `.agents/skills/qgis-plugin-conventions/SKILL.md` | Plugin init, iface, layers, menus, seam imports |
| FVM / CFD Solver Patterns | `.agents/skills/fvm-cfd-solver-patterns/SKILL.md` | Mesh structs, GPU kernels, BCs, coupling, unit system |
| GPU Test Diagnostics | `.commandcode/skills/gpu-test-diagnostics/SKILL.md` | Diagnose CUDA/hydra_swe2d test failures, graph caching, NaN debugging |
| Mesh Quality Triage | `.commandcode/skills/mesh-quality-triage/SKILL.md` | Diagnose and fix Gmsh mesh quality failures |

## Command Code Resources (`.commandcode/`)

This repo's `.commandcode/` directory mirrors the legacy `.opencode/` config:

| Resource | Path | What it contains |
|----------|------|------------------|
| Rules | `.commandcode/rules/` | Architecture, planning, env, test priority, cache, git safety, unit system, UI, session docs, agent selection |
| Agents | `.commandcode/agents/` | 49 agent definitions (build-engineer, cpp-pro, debugger, performance-engineer, python-pro, test-automator × model variants) |
| Skills | `.commandcode/skills/` | GPU test diagnostics, mesh quality triage |
| Skills | `.agents/skills/` | PyQt5 patterns, QGIS conventions, FVM/CFD solver patterns, frontend design, skills discovery |

## ALWAYS KEEP THESE PRINCIPLES OF GOOD SOFTWARE ENGINEERING IN MIND 
Below are twelve fundamental software engineer principles that every developer, architect, and team lead should embrace in their software engineering approach.

### 1. Don’t Repeat Yourself (DRY)

The DRY principle stresses the importance of reducing repetition within a codebase. Duplicate code increases the risk of bugs and inconsistencies because any change must be applied in multiple places.

For example, instead of writing the same validation logic in different parts of your application, you should encapsulate it within a single function or module. This not only improves maintainability but also makes future updates more manageable.

This is one of the core software design principles in software engineering and is essential for creating sustainable codebases that scale.

### 2. Keep It Simple, Stupid (KISS)

Complexity is the enemy of good software design. The KISS principle advises developers to build straightforward solutions that fulfill requirements without unnecessary complications.

Overly complex code is hard to understand, test, and maintain. Instead, focus on clear, readable, and concise implementations. Simple designs are easier for teams to collaborate on and less prone to errors, making this a fundamental software engineer approach.

### 3. You Aren’t Gonna Need It (YAGNI)

YAGNI is a principle that encourages developers to avoid implementing features or functionality before they are actually needed. Prematurely adding features can waste time and introduce unnecessary complexity.

By focusing on current requirements and postponing speculative additions, teams can deliver working software faster and keep the codebase clean and focused. This approach aligns well with agile methodologies and is a valuable development principle in practical software engineering.

### 4. SOLID Principles

The SOLID principles are a collection of five software design principles that guide object-oriented programming for building maintainable and flexible software:

    Single Responsibility Principle (SRP): Each class or module should have one, and only one, reason to change.
    Open/Closed Principle (OCP): Software entities should be open for extension but closed for modification.
    Liskov Substitution Principle (LSP): Objects of a superclass should be replaceable with objects of a subclass without affecting correctness.
    Interface Segregation Principle (ISP): Clients should not be forced to depend on interfaces they do not use.
    Dependency Inversion Principle (DIP): Depend upon abstractions, not on concrete implementations.

Together, these principles reduce tight coupling and improve code reusability and testability, forming the backbone of a sound software engineering approach.

### 5. Separation of Concerns

Separation of Concerns is a design principle that recommends splitting a program into distinct sections, each addressing a separate concern or responsibility. For example, separating user interface code from business logic and data access layers.

This separation helps teams work in parallel on different components, reduces complexity, and makes the system easier to maintain and extend. It is a key principle of software engineering that fosters modular and scalable architecture.

### 6. High Cohesion and Low Coupling

High cohesion means that elements within a module or class should be closely related in functionality, whereas low coupling means that modules should have minimal dependencies on each other.

Achieving high cohesion and low coupling leads to code that is easier to understand, test, and modify. This principle supports building software systems that are modular and adaptable to change, which is essential in a professional software engineer approach.

### 7. Fail Fast

The Fail Fast principle encourages systems to detect and report errors as soon as they occur, rather than allowing issues to propagate silently or manifest later in unpredictable ways.

By failing fast, developers can identify problems early in the development software engineering process, making debugging simpler and reducing the overall cost of fixing defects.

### 8. Design for Testability

Testability is a crucial aspect of quality software development. Designing software with testing in mind means writing modular, loosely coupled code that can be easily isolated and tested independently.

Techniques like dependency injection, mocking, and writing small, single-purpose functions contribute to testability, ensuring that automated tests are reliable and efficient. This principle is an integral part of any mature software engineering approach.

### 9. Encapsulation

Encapsulation is the principle of hiding the internal state and implementation details of a component and exposing only what is necessary through a well-defined interface.

This abstraction reduces complexity for users of the component and protects its internal state from unintended interference. Encapsulation is a foundational principle software engineering practice that promotes safer and more maintainable code.

### 10. Continuous Refactoring

Refactoring involves regularly improving the structure and readability of existing code without changing its external behavior. Continuous refactoring helps to reduce technical debt, improve code quality, and keep the codebase adaptable to evolving requirements.

Adopting this principle ensures that software remains clean and easy to work with, even as it grows in size and complexity over time.

### 11. Use Meaningful Names

Choosing clear and descriptive names for variables, functions, classes, and modules significantly improves code readability and maintainability.

Meaningful names act as documentation and reduce the cognitive load for developers reading or modifying the code later, enhancing team collaboration and speeding up development.

### 12. Principle of Least Astonishment (POLA)

The Principle of Least Astonishment states that software should behave in a way that least surprises users and developers.

Predictable, intuitive behavior reduces confusion, minimizes mistakes, and makes the system easier to use and maintain. Applying this principle is critical when designing APIs, user interfaces, and coding standards within your software engineering approach.


## Planning Discipline — Auto-Agent-Selector & Superpowers

- **Before any multi-step implementation, read the spec and write a plan document in `docs/`.**
- Every plan MUST be written against the spec document, not from memory.
- Before dispatching implementation, verify every spec section/requirement has
  a corresponding task. If a spec requirement has no task, add one. This check
  is part of the plan self-review and must pass before execution begins.
- Every plan MUST include:
  1. Steps as selector-consumable dicts with `action`, `type`, and `phase` fields
  2. Pre-computed `agent` and `model` assignments (run through `auto_agent_selector`)
  3. A machine-readable JSON block (§8 pattern) for the plugin hook
  4. A superpowers workflow section listing which skills to use and when
- Steps must contain routing keywords (`python`, `refactor`, `test`, `debug`, etc.)
  so `recommend_agent_from_keywords()` assigns the right agent.
- Do NOT skip the plan format step. If you just start coding without the plan
  dict, the auto_agent_selector cannot rewrite the plan, and subagents won't
  be assigned correctly.

## repo specific notes

### MVP Architecture — Mandatory

This project uses **Model-View-Presenter** architecture. See `.commandcode/rules/MVP_ARCHITECTURE.md` for the full spec.

**The iron rule**: Code in `swe2d/runtime/` and `swe2d/*service*.py` MUST NOT import Qt, touch widgets, or reference `run_btn`, `cancel_btn`, or any UI element. If a service component needs to update UI state (e.g., enabling a button), it calls a View protocol method. If you're tempted to write `self._ui.run_btn.setEnabled(True)` in a runtime module, STOP — that is an architecture violation. Route through the Controller + View protocol instead.

### Validation Priority

- Prefer GPU-focused validation suites first:
  - `tests/test_swe2d_gpu_validation_perf.py`
  - `tests/test_swe2d_gpu_unstructured.py`

## Repository Session Documentation

- Store implementation handoff and recovery notes in repository-tracked docs under `docs/` so they can be pushed to origin.
- Current rolling session log: [docs/AGENT_SESSION_RECOVERY_LOG.md](docs/AGENT_SESSION_RECOVERY_LOG.md).

## Python Cache Discipline

- After any structural change to a Python module (signature changes, new return values, new classes, changed imports), **always purge `__pycache__`** before the user restarts QGIS:
  ```bash
  find . -type d -name __pycache__ -exec rm -rf {} +
  ```

## Git Safety — Destructive File Operations

- **NEVER** run `git checkout -- <file>` (or any command that overwrites working-tree files with committed versions) without first checking for uncommitted changes across the **entire repo**:
  ```bash
  git status --short
  ```
- If ANY file shows `M` (modified), `A` (added), `D` (deleted), or `??` (untracked) that could be relevant, do NOT use destructive git commands. Use manual `replace_string_in_file` edits to revert only specific changes instead.
- `git checkout -- <file>` silently discards ALL uncommitted changes in `<file>` — including changes the agent didn't make. There is no undo.
- QGIS holds modules in memory for the session, and stale `.pyc` files cause invisible failures (wrong arity, missing attributes, silent fallback paths).
- When in doubt, purge before asking the user to restart.

## Unit System Conventions

- **Never assume a specific unit system** (SI or USC). All conversions must be based on the CRS-derived map units via `swe2d.units`.
- **C++ kernel accepts model units** for all geometry. Weir, orifice, bridge, and pump formulas are unit-agnostic — they produce correct results in whatever units the inputs are in, as long as the `gravity` parameter matches. Only the HDS-5 culvert path converts geometry to feet internally, computes in USC, then converts the result back to model units using the caller-supplied `model_to_ft` factor.
- **C++ kernel culvert output** is converted from CFS back to model units (÷ `model_to_ft³`) before returning. Non-culvert types return values directly in model units.
- **Python coupling controller** (`coupling.py`) converts kernel CFS output to model units via `SI_M3_PER_USC_FT3 / si_m3_per_model_volume()`.
- **Python structure module** (`swe2d/extensions/structures.py`) always returns **CMS** because culvert routines adopted from SWMM compute in USC and explicitly convert CFS→CMS.
- **Diagnostics stored in `SWE2DCouplingDiagnostics`** are in **model units** (not SI). The coupling controller converts from kernel/Python output units to model units before storing.
- **Runtime reporter** (`runtime_reporting.py`) displays diagnostics using `length_unit_name` and assumes values are already in model units.
- **Heap gravity bug fix**: Orifice/bridge formulas now use CRS-derived `gravity()` (9.81 m/s² for SI, 32.17 ft/s² for USC) instead of the old hardcoded 9.81 — this was a ~45% underestimation for USC projects.
- **`model_to_ft`**: Passed from Python to the C++ kernel as `units.model_to_ft()`. Needed so culvert code can convert model-unit geometry to feet internally.

## Studio UI Architecture & Structural Changes

- **`.ui` files are the source of truth** for widget layout and properties.
  Use Qt Designer to edit them.  Only create widgets programmatically when
  they cannot live in a `.ui` file (dynamically populated combos, etc.).
- When making structural changes (new tabs, new forms, new feature toggles,
  widget moves/renames), follow the checklists in
  [docs/STUDIO_UI_ARCHITECTURE.md](docs/STUDIO_UI_ARCHITECTURE.md).
- After any `.ui` change, run:
  ```bash
  python tools/ui_bind_sync.py forms/swe2d_<name>.ui <py_files> --missing
  ```
  to verify all widgets have bindings and no orphans remain.
- Feature toggles touch 3 files: feature flags dict + keyword function in
  `SWE2DWorkbenchStudioDialog`, and menu/toolbar actions in `studio_host_methods.py`.
  All three must be updated together.

## Proper Widget Lifecycle — No Dead Shells

- When extracting widgets from a parent (via `detach_controls()`,
  `populate_view_tabs()`, etc.), **delete the empty parent shell**.
  Keeping it alive "just in case" leaves a dead QWidget hull in memory
  that can end up registered as a QGIS dock widget, showing a blank
  panel in View → Panels.  This is confusing and lazy.
- If signals or timers reference the old parent, reparent them first
  or disconnect before deleting.  An `iface.removeDockWidget()` call
  is not a substitute for proper cleanup — destroy the object.
- PyQt5 parent-child ownership means deleting a parent cascades to
  children.  If you extracted the children, set their parent to `None`
  first via `widget.setParent(None)`, then `deleteLater()` the shell.
- Exception: QTimer children with active timeouts need to be stopped
  before the parent can be safely destroyed.  Call `timer.stop()` and
  `timer.deleteLater()` during the extraction step.
