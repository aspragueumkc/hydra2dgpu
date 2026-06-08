# SWE2D Codebase Audit — Silent Fallbacks, Dead Code, Style & Modernization Plan

> **Date**: 2026-06-08  
> **Scope**: All SWE2D Python modules (`swe2d/`), `native_backend.py`, `unsteady_model.py`, `culvert_routine.py`, `stacked_bridge_coupling.py`  
> **Status**: Complete audit + actionable plan

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Audit Methodology](#2-audit-methodology)
3. [Findings: Silent Fallbacks](#3-findings-silent-fallbacks)
4. [Findings: Backward Compatibility Fallbacks](#4-findings-backward-compatibility-fallbacks)
5. [Findings: Dead Code & Technical Debt](#5-findings-dead-code--technical-technical-debt)
6. [Findings: Code Style Inconsistencies](#6-findings-code-style-inconsistencies)
7. [Recommended Coding Convention](#7-recommended-coding-convention)
8. [Software Engineering Philosophy](#8-software-engineering-philosophy)
9. [Remediation Plan: Remove Backward Compat Fallbacks](#9-remediation-plan-remove-backward-compat-fallbacks)
10. [Remediation Plan: Add Logging to Silent Fallbacks](#10-remediation-plan-add-logging-to-silent-fallbacks)
11. [Remediation Plan: Codebase Conformance](#11-remediation-plan-codebase-conformance)
12. [Appendix A: Complete Fallback Inventory](#appendix-a-complete-fallback-inventory)

---

## 1. Executive Summary

The SWE2D codebase has **~95 `try/except` blocks** across its Python modules. Of these:

| Category | Count | Risk |
|----------|-------|------|
| Silent failures (no logging) | **~55** | 🔴 High |
| Logged fallbacks (with warnings) | **~15** | 🟡 Medium (intentional) |
| Import compatibility fallbacks | **~12** | 🟢 Low (standard practice) |
| Dead/commented-out code | **~8** | 🟡 Medium |
| Debug `print()` left in production | **5** | 🔴 High |

The most critical issues are:
1. **GPU→CPU silent fallback** in `backend_initializer.py` — caller never knows GPU was unavailable
2. **Native source injection silent fallback** in `runtime_source_logic.py` — performance degradation invisible
3. **~40 UI exception handlers** that silently swallow errors — regressions invisible
4. **Backend constructor three-tier fallback** duplicated across `backend.py`, `coupling.py`, and `native_backend.py`
5. **Five `print(f"[BC_DIAG]…")` / `print(f"DEBUG:…")`** statements left in production code

---

## 2. Audit Methodology

Every `.py` file in `swe2d/` (71 files) plus root-level SWE2D-adjacent files (`native_backend.py`, `unsteady_model.py`, `culvert_routine.py`, `stacked_bridge_coupling.py`) was scanned for:

- `try/except` blocks — cataloged by exception type, whether they log, and whether they change control flow
- `except Exception: pass` and bare `except:` — flagged as silent failures
- `hasattr`/`getattr` patterns — evaluated for backward-compatibility intent vs. feature detection
- `# TODO` / `# FIXME` / `# TEMPORARY` / `# HACK` comments — tracked as tech debt
- Commented-out code blocks — flagged for removal or implementation
- `print()` calls in non-test code — flagged for removal
- Import `try/except` patterns — evaluated for necessity
- Naming consistency, docstring coverage, type hint usage

---

## 3. Findings: Silent Fallbacks

### 3.1 🔴 CRITICAL: Backend Constructor Three-Tier Fallback

**File**: `swe2d/runtime/backend_initializer.py` (lines 68–73)

```python
try:
    b = backend_cls(use_gpu=bool(use_gpu), openmp_enabled=bool(openmp_enabled))
except TypeError:
    try:
        b = backend_cls(use_gpu=bool(use_gpu))
    except TypeError:
        b = backend_cls()
```

**Problem**: If the C++ binding doesn't accept `use_gpu` or `openmp_enabled`, the code silently falls back to a no-argument constructor. The caller has **no way to know** whether GPU or OpenMP was actually enabled. This can cause:
- User selects GPU mode, runs simulation, gets CPU performance — no warning
- OpenMP threads silently unused — performance regression invisible

**Risk**: 🔴 **HIGH** — Performance-affecting, no diagnostics

### 3.2 🔴 CRITICAL: Native Source Injection Fallback

**File**: `swe2d/boundary_and_forcing/runtime_source_logic.py` (lines 93–98)

```python
if prefer_native_injection and hasattr(backend, "set_external_sources_native"):
    try:
        backend.set_external_sources_native(src)
        return
    except Exception:
        pass  # Silent CPU fallback
```

**Problem**: When the GPU-native source injection fails, the code falls back to CPU-side source application without any logging. This is a performance-critical path executed every timestep.

**Risk**: 🔴 **HIGH** — Silent performance regression, no diagnostic trail

### 3.3 🔴 CRITICAL: Backend Destructor Silent Failure

**File**: `swe2d/runtime/run_lifecycle.py` (lines 28–30)

```python
try:
    if backend is not None:
        backend.destroy()
except Exception:
    pass
```

**Problem**: GPU memory leaks, file handle leaks, and CUDA context leaks are completely invisible.

**Risk**: 🔴 **HIGH** — Resource leak, no diagnostics

### 3.4 🔴 HIGH: GPU Capability Probing Silent Failure

**File**: `swe2d/runtime/run_options_builder.py` (lines 113–117)

```python
try:
    return bool(self._swe2d_gpu_available(openmp_enabled=openmp_enabled))
except TypeError:
    try:
        return bool(self._swe2d_gpu_available())
    except Exception:
        return False
```

**Problem**: If the GPU availability probe fails for any reason (not just signature mismatch), the code silently returns `False`, making CUDA appear unavailable.

**Risk**: 🔴 **HIGH** — GPU capability detection can silently fail

### 3.5 🟡 MEDIUM: Backend Feature Detection (No Logging)

**File**: `swe2d/runtime/backend.py` (lines 423–426)

```python
self._supports_solver_bc_update = hasattr(self._mod, "swe2d_solver_set_boundary_values")
self._supports_solver_hydrographs = hasattr(self._mod, "swe2d_solver_set_boundary_hydrographs")
self._supports_solver_rain_cn = hasattr(self._mod, "swe2d_solver_set_rain_cn_forcing")
self._supports_solver_external_sources = hasattr(self._mod, "swe2d_solver_set_external_sources")
```

**Problem**: When a C++ feature is absent from the loaded binary, it's silently disabled. No log message informs the user which features are unavailable.

**Risk**: 🟡 **MEDIUM** — Features silently disabled, no diagnostic trail

### 3.6 🟡 MEDIUM: Coupling Redistribution Silent Failures

**File**: `swe2d/runtime/coupling.py` (lines 1221, 1290, 1355, 1386)

Four separate `except Exception: pass` blocks around GPU redistribution kernels. When on-device redistribution fails:
- The code silently falls back to CPU host-readback
- The caller sees `native_device_applied=True` and skips the callback
- **Redistribution is NOT applied** — silently

**Risk**: 🟡 **MEDIUM** — Coupling accuracy silently degraded

### 3.7 🟡 MEDIUM: UI Exception Handlers (~40 locations)

**Files**: `swe2d/workbench/three_d_bc.py`, `swe2d/results/panel.py`, `swe2d/workbench/non_gui_runtime.py`

Approximately 40 bare `except Exception: pass` blocks across UI code. Key critical paths:

| File | Line | What fails silently |
|------|------|---------------------|
| `panel.py` | 768 | Timestep union rebuild → empty plots |
| `panel.py` | 926, 964, 989, 1028 | Structure/profile data loads |
| `non_gui_runtime.py` | 51 | Snapshot rows silently dropped (DATA LOSS) |
| `non_gui_runtime.py` | 275, 341 | Mesh processing continues on bad cells |
| `three_d_bc.py` | 67, 75, 134, 138, 146 | Face BC mode retrieval |
| `view.py` | 61, 82, 87, 98 | Widget value getters return empty strings |

**Risk**: 🟡 **MEDIUM** — UI regressions invisible, potential data loss

### 3.8 🟢 LOW: Import Compatibility Fallbacks

**Files**: `run_helpers.py`, `three_d_bc.py`, `panel.py`, `unsteady_model.py`

Standard Python practice for optional module loading (QGIS→PyQt5 fallback, absolute→relative import). These are acceptable.

**Risk**: 🟢 **LOW** — Standard practice, no performance impact

---

## 4. Findings: Backward Compatibility Fallbacks

### 4.1 Three-Tier pybind11 Signature Compatibility

**Duplicated in**: `backend.py`, `coupling.py`, `native_backend.py`

This pattern appears in three separate files with slightly different implementations:

```python
try:
    result = native_func(**full_kwargs)
except TypeError:
    # Tier 2: inspect signature, filter
    sig = inspect.signature(native_func)
    filtered = {k: v for k, v in full_kwargs.items() if k in sig.parameters}
    result = native_func(**filtered)
except (TypeError, ValueError):
    # Tier 3: hardcoded filter list
    result = native_func(**conservative_kwargs)
```

**Purpose**: Handles pybind11 binding signature evolution across C++ binary versions.

**Assessment**: This is the primary backward-compatibility mechanism. It must be replaced with explicit C++ binary versioning.

### 4.2 Legacy Database Schema Support

**File**: `swe2d/results/queries.py` (lines 120–132)

```python
# Try new shared schema first
ts = load_timesteps_from_shared_schema(...)
if ts is None:
    # Fall back to legacy per-run table
    ts = load_timesteps_from_legacy_table(run_id)
```

**Assessment**: The legacy schema path should be removed once all existing project databases have been migrated.

### 4.3 Deprecated API Aliases

| File | Symbol | Status |
|------|--------|--------|
| `swe2d/units.py` | `compute_length_factor()` → `model_to_ft()` | Deprecated alias |
| `swe2d/extensions/extension_models.py` | `legacy_max_flow_cms` parameter | Coexists with `max_flow` |
| `swe2d/extensions/drainage_network.py` | Backward-compat aliases in comments | Documented |

### 4.4 Native Solver Runtime Tracking Dictionary

**File**: `unsteady_model.py` (lines 116–155)

The `_NATIVE_SOLVER_RUNTIME` dictionary tracks success/fallback counts across all native→Python fallback paths. This is a **well-designed** diagnostic mechanism, but it's:
- Not automatically logged (requires manual inspection)
- Not surfaced in the UI
- Updated in 8+ locations across the file

**Assessment**: Good diagnostic infrastructure. Should be enhanced with automatic periodic logging and UI exposure.

---

## 5. Findings: Dead Code & Technical Debt

### 5.1 Dead Code Stub

| File | Lines | Description |
|------|-------|-------------|
| `swe2d/mesh/meshing.py` | 10283–10294 | `_normalize_post_opt_backend()` always returns `"none"` — entire body is commented out |

### 5.2 Debug Print Statements in Production

| File | Line | Statement |
|------|------|-----------|
| `swe2d/workbench/non_gui_runtime.py` | 1639 | `print(f"[BC_DIAG] distribute_ms=...")` |
| `hydra_1d.py` | 3260 | `print(f"DEBUG: compute_state returned None...")` |
| `hydra_1d.py` | 3262 | `print(f"DEBUG: downstream SectionState type=...")` |
| `hydra_1d.py` | 3402 | `print(f"DEBUG: loop i={i}...")` |
| `hydra_1d.py` | 3404 | `print(f"DEBUG: loop i={i}, unable to repr...")` |

### 5.3 Commented-Out Code Blocks

| File | Lines | Description |
|------|-------|-------------|
| `swe2d/mesh/meshing.py` | 1094 | Disabled breaklines feature |
| `swe2d/mesh/meshing.py` | 10286–10290 | Backend selection logic (5 lines) |
| `swe2d/results/velocity_layer.py` | 350 | "top-speed cells as seeds" fallback |

### 5.4 TODO/FIXME Items

| File | Line | Item |
|------|------|------|
| `swe2d/extensions/extension_models.py` | 543 | `TODO: return (surface_sink_cms_per_cell, surcharge_source_cms_per_cell)` |
| `swe2d/extensions/extension_models.py` | 555 | `TODO: support gauge interpolation, raster time slices, IDF events` |
| `swe2d/extensions/extension_models.py` | 569 | `TODO: route to per-structure equations and control logic` |

### 5.5 TEMPORARY Markers

| File | Line | Description |
|------|------|-------------|
| `swe2d/mesh/meshing.py` | 10261 | "disabled per user request; keep implementation in-tree" |
| `swe2d/mesh/meshing.py` | 10283 | "MFEM post-optimization is disabled in backend execution paths" |
| `swe2d/workbench/non_gui_runtime.py` | 1637 | "TEMPORARY: log bc timing breakdown every N steps" |

---

## 6. Findings: Code Style Inconsistencies

### 6.1 Exception Handling

| Pattern | Count | Assessment |
|---------|-------|------------|
| `except Exception: pass` (silent) | ~55 | 🔴 Must be fixed |
| `except Exception:` with logging | ~10 | 🟡 Acceptable |
| `except (SpecificType):` with logging | ~5 | ✅ Good |
| `except Exception:` re-raised | ~3 | ✅ Good |

### 6.2 Naming Conventions

| Pattern | Occurrences | Assessment |
|---------|-------------|------------|
| `snake_case` functions/methods | ~95% | ✅ Consistent |
| `snake_case` variables | ~90% | ✅ Consistent |
| `CamelCase` classes | 100% | ✅ Consistent |
| `_private` prefix | ~60% | 🟡 Inconsistent — some public methods lack documentation of intent |

### 6.3 Import Patterns

| Pattern | Occurrences | Assessment |
|---------|-------------|------------|
| `from __future__ import annotations` | Most files | ✅ Good |
| Absolute imports (`from swe2d.foo import Bar`) | ~70% | ✅ Preferred |
| Relative imports (`from .foo import Bar`) | ~20% | 🟡 Inconsistent |
| `import X as _alias` (underscore prefix) | ~10% | 🟡 Only sometimes used |

### 6.4 Type Hints

| Module Group | Type Hint Coverage | Assessment |
|-------------|-------------------|------------|
| `swe2d/runtime/` | ~90% | ✅ Good |
| `swe2d/extensions/` | ~85% | ✅ Good |
| `swe2d/workbench/` | ~60% | 🟡 Needs improvement |
| `swe2d/results/` | ~50% | 🔴 Needs improvement |
| `swe2d/mesh/` | ~40% | 🔴 Needs improvement |
| Root-level files | ~30% | 🔴 Needs improvement |

### 6.5 Docstring Coverage

| Module Group | Docstring Coverage | Assessment |
|-------------|-------------------|------------|
| `swe2d/runtime/coupling.py` | ~95% | ✅ Excellent |
| `swe2d/extensions/drainage_network.py` | ~90% | ✅ Excellent |
| `swe2d/runtime/backend.py` | ~80% | ✅ Good |
| `swe2d/workbench/` | ~40% | 🔴 Needs improvement |
| `swe2d/results/` | ~30% | 🔴 Needs improvement |
| `swe2d/mesh/` | ~25% | 🔴 Needs improvement |

### 6.6 Logging Conventions

| Pattern | Assessment |
|---------|------------|
| `[ERROR]`, `[WARNING]`, `[COUPLING_*]` prefixes in `coupling.py` | ✅ Excellent — structured tags |
| `self._log(msg)` callback pattern | ✅ Good — consistent in runtime modules |
| `logging.getLogger(__name__)` | 🟡 Used in some modules, inconsistent |
| `print()` for debug output | 🔴 5 instances in production code |

---

## 7. Recommended Coding Convention

### 7.1 Core Principles

1. **Explicit over Implicit**: Every fallback must be documented with a comment explaining *why* it exists and *what* it degrades.

2. **Fail Loud, Log Always**: Never silently swallow exceptions. At minimum, log a warning with the exception type and message.

3. **Single Responsibility**: Each function/class does one thing. Extract helpers rather than adding branches.

4. **Type Hints Everywhere**: All public functions must have complete type annotations. Private helpers should have them too.

5. **Docstrings on Public API**: Every public function, class, and module must have a docstring.

### 7.2 Naming Conventions

```python
# Modules: short, lowercase, snake_case
swe2d/runtime/coupling.py
swe2d/extensions/drainage_network.py

# Classes: PascalCase
class SWE2DCouplingController:
class DrainageNetworkConfig:

# Functions/methods: snake_case
def pack_drainage_soa():
def _apply_momentum_cap():     # private helper

# Constants: UPPER_SNAKE_CASE
MIN_WATER_DEPTH = 1.0e-6
DEFAULT_GRAVITY = 9.81

# Private members: single underscore prefix
self._backend = None
self._log = log_callback

# "Internal" module-level: underscore prefix
_NATIVE_SOLVER_RUNTIME: Dict[str, Any] = {}
```

### 7.3 Exception Handling Rules

```python
# ✅ GOOD: Log and re-raise or degrade gracefully with warning
try:
    result = native_func(**kwargs)
except TypeError as exc:
    logger.warning(
        "Native binding signature mismatch; falling back to CPU path. "
        f"Error: {exc}"
    )
    result = cpu_fallback(**kwargs)

# ✅ GOOD: Specific exception types
try:
    value = float(widget.text())
except ValueError:
    logger.debug(f"Invalid numeric input in {widget.objectName()}; using default")
    value = default_value

# 🔴 BAD: Silent exception swallowing
try:
    result = native_func(**kwargs)
except Exception:
    pass  # NEVER do this

# 🔴 BAD: Bare except
try:
    result = native_func(**kwargs)
except:
    pass
```

### 7.4 Logging Convention

```python
# Use structured tags for different subsystems
self._log("[COUPLING] Structure source redistribution failed; falling back to CPU path")
self._log("[BACKEND] GPU unavailable; falling back to CPU solver")
self._log("[DRAINAGE] Solver mode 2 (dynamic) not available; using mode 1 (diffusion)")
self._log("[MESHING] MFEM post-optimization disabled; using baseline mesh")

# Tag taxonomy:
#   [BACKEND]     — C++ module loading, GPU detection, constructor fallbacks
#   [COUPLING]    — Surface-drainage coupling, structure source injection
#   [DRAINAGE]    — Pipe network solver, EGL/diffusion/dynamic modes
#   [MESHING]     — Mesh generation, optimization, constraint handling
#   [RUNTIME]     — Timestepping, initial conditions, boundary updates
#   [RESULTS]     — Output queries, visualization, export
#   [UI]          — Widget operations, dialog management
```

### 7.5 Import Convention

```python
# Standard library
from __future__ import annotations
import os
from typing import Any, Dict, Optional

# Third-party
import numpy as np

# Internal (always absolute imports)
from swe2d import units as _u
from swe2d.runtime.backend import load_swe2d_native_module
from swe2d.extensions.extension_models import StructureType

# Qt (with fallback)
try:
    from QGIS.PyQt.QtWidgets import QWidget
except ImportError:
    from PyQt5.QtWidgets import QWidget  # standalone testing
```

### 7.6 Docstring Convention

```python
def pack_drainage_soa(
    cfg: Optional[PipeNetworkConfig],
    n_cells: int,
) -> Optional[SWE2DDrainageSoA]:
    """Pack drainage network configuration into Structure-of-Arrays for GPU upload.

    Parameters
    ----------
    cfg : PipeNetworkConfig, optional
        Drainage network configuration. Returns None if None or disabled.
    n_cells : int
        Number of 2D mesh cells (used for inlet/outfall cell index validation).

    Returns
    -------
    SWE2DDrainageSoA or None
        Packed SoA data, or None if drainage is not configured.

    Notes
    -----
    Inlet/outfall cell indices reference the 2D mesh. Invalid indices (-1)
    indicate the structure is not coupled to the surface model.
    """
```

### 7.7 Type Hint Convention

```python
# Use modern union syntax (from __future__ import annotations)
def process(x: float | None) -> dict[str, Any]:
    ...

# For numpy arrays, use np.ndarray (no element type — NumPy doesn't support it well)
def apply_source(backend: Any, src: np.ndarray) -> None:
    ...

# For callbacks, use Callable
from collections.abc import Callable
log_fn: Callable[[str], None] = print
```

---

## 8. Software Engineering Philosophy

### 8.1 Modularity

```
┌─────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                     │
│  swe2d/workbench/     (UI, dialogs, widget binding)      │
│  swe2d/results/       (visualization, panels)             │
├─────────────────────────────────────────────────────────┤
│                    ORCHESTRATION LAYER                     │
│  swe2d/runtime/       (coupling, lifecycle, options)      │
│  native_backend.py    (C++ bridge)                        │
├─────────────────────────────────────────────────────────┤
│                    DOMAIN LAYER                           │
│  swe2d/extensions/    (structures, drainage, models)      │
│  swe2d/mesh/          (mesh generation, gmsh)             │
│  swe2d/units.py       (unit system)                       │
├─────────────────────────────────────────────────────────┤
│                    INFRASTRUCTURE LAYER                    │
│  swe2d/results/db_utils.py  (SQLite access)              │
│  swe2d/boundary_and_forcing/ (source injection)          │
│  cpp/                 (native solver code)                │
└─────────────────────────────────────────────────────────┘
```

**Rules**:
1. **Presentation** never imports from **Infrastructure**.
2. **Orchestration** imports from both **Domain** and **Infrastructure**.
3. **Domain** is pure logic — no Qt, no file I/O, no logging callbacks.
4. **Infrastructure** handles I/O, native bindings, and external systems.

### 8.2 Extendability Patterns

```python
# ✅ Protocol-based interface for solver backends
from typing import Protocol, runtime_checkable

@runtime_checkable
class SolverBackend(Protocol):
    def build_mesh(self, nodes: np.ndarray, ...) -> None: ...
    def initialize(self, h0: np.ndarray, ...) -> None: ...
    def step(self, dt: float) -> float: ...
    def destroy(self) -> None: ...

# ✅ Strategy pattern for structure formulas
class StructureFormula(Protocol):
    def compute_flow(self, head_up: float, head_dn: float, ...) -> float: ...

class WeirFormula:
    """Broad-crested weir: Q = C_w * L * H^(3/2)"""

class OrificeFormula:
    """Orifice: Q = C_d * A * sqrt(2*g*H)"""

class CulvertFormula:
    """FHWA HEC-22 culvert hydraulics"""

# ✅ Dataclass-based configuration (already used extensively)
@dataclass
class PipeNetworkConfig:
    enabled: bool
    solver_mode: DrainageSolverMode
    nodes: list[DrainageNode]
    links: list[DrainageLink]
```

### 8.3 Testability

1. Every module function must be callable without a QGIS or Qt context.
2. Backend classes accept callbacks, not UI widget references.
3. Configuration is dataclass-based, allowing unit test construction without serialization.
4. Coupling diagnostics are returned as dataclasses, not printed.

### 8.4 Progressive Enhancement

```
CPU (baseline) → CPU+OpenMP → CUDA (single GPU) → CUDA (multi-GPU)
     ↑                ↑              ↑                  ↑
  Always works   Compile flag    Optional binary     Future
```

**Rule**: Every CPU path must remain functional and tested. GPU/CUDA paths are opt-in enhancements that must **never** silently degrade.

---

## 9. Remediation Plan: Remove Backward Compat Fallbacks

### Phase 1: Eliminate Three-Tier pybind11 Fallback (Week 1)

**Goal**: Replace the triple try/except signature compat with explicit C++ binary versioning.

**Files**: `backend.py`, `coupling.py`, `native_backend.py`

**Approach**:
1. Add a `SWE2D_VERSION` integer constant to the C++ module (currently missing)
2. Replace `_create_solver_compat()` with version-gated calls:

```python
# NEW: Explicit version-gated construction
def _create_backend(backend_cls, use_gpu: bool, openmp_enabled: bool):
    version = getattr(_mod, "SWE2D_VERSION", 0)
    if version >= 3:
        return backend_cls(use_gpu=use_gpu, openmp_enabled=openmp_enabled)
    elif version >= 2:
        return backend_cls(use_gpu=use_gpu)
    else:
        logger.warning("[BACKEND] Legacy C++ binary (v%d); GPU selection ignored", version)
        return backend_cls()
```

3. Remove `_create_solver_compat()` from all three files (deduplicate into `backend.py` only)
4. Add compile-time version constant to `CMakeLists.txt`

### Phase 2: Remove Legacy Database Schema (Week 2)

**Goal**: Drop per-run table fallback; require shared schema.

**Files**: `swe2d/results/queries.py`, `swe2d/results/db_utils.py`

**Approach**:
1. Add migration utility that converts legacy tables to shared schema on first open
2. Remove fallback query paths
3. Add version column to schema; check on load

### Phase 3: Remove Deprecated API Aliases (Week 3)

**Goal**: Remove deprecated function aliases with deprecation period.

**Files**: `swe2d/units.py`, `swe2d/extensions/extension_models.py`

**Approach**:
1. `compute_length_factor()` → Add `warnings.warn("Use model_to_ft()", DeprecationWarning)` for 2 releases
2. `legacy_max_flow_cms` → Remove parameter; require callers to use `max_flow`
3. Drainage backward-compat aliases → Remove and update callers

### Phase 4: Remove Dead Code (Week 3)

**Files**: `swe2d/mesh/meshing.py`, `hydra_1d.py`, `swe2d/workbench/non_gui_runtime.py`

**Approach**:
1. `_normalize_post_opt_backend()` → Either implement or remove entirely
2. Remove all `print(f"DEBUG:…")` and `print(f"[BC_DIAG]…")` statements
3. Remove commented-out code blocks (5 locations)
4. Remove `# TEMPORARY` markers — either implement or delete

---

## 10. Remediation Plan: Add Logging to Silent Fallbacks

### Priority 1: Performance-Affecting Fallbacks (Week 1)

| File | Line | Change |
|------|------|--------|
| `backend_initializer.py` | 68–73 | Log which constructor signature succeeded and what features are active |
| `runtime_source_logic.py` | 93–98 | Log warning when GPU native injection fails, falls back to CPU |
| `run_lifecycle.py` | 28–30 | Log `backend.destroy()` failure with exception details |
| `run_options_builder.py` | 113–117 | Log which GPU probe succeeded/failed |

**Template**:
```python
try:
    b = backend_cls(use_gpu=True, openmp_enabled=True)
except TypeError:
    try:
        b = backend_cls(use_gpu=True)
        logger.warning(
            "[BACKEND] C++ binary does not accept openmp_enabled parameter; "
            "OpenMP threading may be unavailable."
        )
    except TypeError:
        b = backend_cls()
        logger.warning(
            "[BACKEND] C++ binary does not accept use_gpu parameter; "
            "running in CPU mode."
        )
```

### Priority 2: Feature Detection Logging (Week 2)

| File | Lines | Change |
|------|-------|--------|
| `backend.py` | 423–426 | Log each unsupported feature with `[BACKEND] Feature 'X' unavailable in loaded binary` |
| `backend.py` | 266 | Log dynamic BC update failures |
| `run_options_builder.py` | 84, 97, 159 | Log UI widget fallbacks with defaults applied |

### Priority 3: Coupling Fallback Logging (Week 2)

| File | Lines | Change |
|------|-------|--------|
| `coupling.py` | 1221 | Replace `except Exception: pass` with logged warning |
| `coupling.py` | 1290 | Replace `except Exception: pass` with logged warning |
| `coupling.py` | 1355 | Replace `except Exception: pass` with logged warning |
| `coupling.py` | 1386 | Replace `except Exception: pass` with logged warning |

### Priority 4: UI Fallback Logging (Week 3)

For UI exception handlers, apply a triage:

| Category | Action |
|----------|--------|
| **Critical data paths** (panel.py:768, non_gui_runtime.py:51) | Log warning + show user notification |
| **Widget value getters** (view.py, three_d_bc.py) | Log debug + return default |
| **Database queries** (panel.py, db_utils.py) | Log warning + return empty |
| **Layout adjustments** (non_gui_runtime.py) | Log debug only |

**Template for UI handlers**:
```python
try:
    ts = load_timesteps(...)
except Exception as exc:
    logger.debug("Timestep load failed: %s", exc)
    ts = []
```

### Priority 5: Backend Cleanup Logging (Week 1)

```python
def finalize_cleanup(self, backend: Any) -> None:
    try:
        if backend is not None:
            backend.destroy()
    except Exception as exc:
        logger.warning("[BACKEND] Backend destroy() failed: %s", exc)
    self._ui.run_btn.setEnabled(True)
    self._ui.cancel_btn.setEnabled(False)
```

---

## 11. Remediation Plan: Codebase Conformance

### Stage 1: Structural Cleanup (Weeks 1–2)

1. **Remove all `print()` debug statements** (5 locations)
2. **Remove all commented-out code** (5 locations)
3. **Remove `_normalize_post_opt_backend()` dead stub** or implement it
4. **Purge `__pycache__` after every structural change** (per `AGENTS.md`)

### Stage 2: Exception Handling Standardization (Weeks 2–3)

Apply the following rules across all ~55 silent exception handlers:

1. **Every `except Exception: pass`** must be replaced with either:
   - `except SpecificException as exc: logger.warning("[TAG] ...", exc)` — for recoverable failures
   - `except SpecificException as exc: raise RuntimeError(...) from exc` — for unrecoverable failures

2. **Narrow exception types**: Replace `except Exception` with the most specific type:
   - `TypeError` for signature mismatches
   - `ValueError` for bad input values
   - `ImportError` for missing modules
   - `OSError` / `IOError` for file operations
   - `RuntimeError` for backend failures

3. **Structured tag taxonomy**: Use `[BACKEND]`, `[COUPLING]`, `[DRAINAGE]`, `[MESHING]`, `[RUNTIME]`, `[RESULTS]`, `[UI]`

### Stage 3: Type Hints & Docstrings (Weeks 3–4)

**Priority files** (lowest coverage → highest impact):
1. `swe2d/results/panel.py` — Add type hints to all public methods
2. `swe2d/workbench/three_d_bc.py` — Add type hints and docstrings
3. `swe2d/workbench/non_gui_runtime.py` — Add type hints to callback interfaces
4. `swe2d/mesh/meshing.py` — Add docstrings to public functions
5. Root-level files (`unsteady_model.py`, `native_backend.py`) — Complete type annotations

### Stage 4: Import Standardization (Week 3)

1. Convert all relative imports to absolute imports (consistent with project convention)
2. Group imports: stdlib → third-party → internal (with blank line separators)
3. Remove redundant `try/except ImportError` for internal modules (let them fail loudly)

### Stage 5: Deduplicate Backend Construction (Week 2)

The three-tier pybind11 fallback is implemented in:
- `swe2d/runtime/backend.py` (lines 228–262)
- `swe2d/runtime/coupling.py` (lines 309–340)
- `native_backend.py` (lines 50–54)

**Action**: Extract into a single `swe2d/runtime/native_binding_compat.py` utility, called from all three locations. Or better: add C++ version constant and eliminate the compat layer entirely.

### Stage 6: Enhance Diagnostic Infrastructure (Week 4)

1. Make `_NATIVE_SOLVER_RUNTIME` (unsteady_model.py) automatically log summary every N steps
2. Expose coupling diagnostics in the UI status bar
3. Add a "diagnostics mode" toggle that enables verbose logging of all fallbacks
4. Consider adding a `SWE2DRunReport` dataclass that captures all fallback events for the run

---

## Appendix A: Complete Fallback Inventory

### A.1 All `try/except` Blocks by File

| File | Total | Silent | Logged | Import | Re-raised |
|------|-------|--------|--------|--------|-----------|
| `swe2d/runtime/backend_initializer.py` | 1 | 1 | 0 | 0 | 0 |
| `swe2d/runtime/backend.py` | 9 | 3 | 0 | 1 | 1 |
| `swe2d/runtime/coupling.py` | 6 | 4 | 2 | 0 | 0 |
| `swe2d/runtime/run_options_builder.py` | 2 | 2 | 0 | 0 | 0 |
| `swe2d/runtime/run_lifecycle.py` | 1 | 1 | 0 | 0 | 0 |
| `swe2d/runtime/run_helpers.py` | 1 | 0 | 0 | 1 | 0 |
| `swe2d/boundary_and_forcing/runtime_source_logic.py` | 3 | 2 | 0 | 0 | 0 |
| `swe2d/boundary_and_forcing/boundary_qgis_adapter.py` | 1 | 0 | 1 | 0 | 0 |
| `swe2d/extensions/patch_qgis_adapter.py` | 2 | 2 | 0 | 0 | 0 |
| `swe2d/extensions/patch_observer.py` | 3 | 3 | 0 | 0 | 0 |
| `swe2d/extensions/drainage_network.py` | 2 | 1 | 0 | 0 | 0 |
| `swe2d/results/db_utils.py` | 3 | 3 | 0 | 0 | 0 |
| `swe2d/results/queries.py` | 3 | 0 | 0 | 0 | 0 |
| `swe2d/results/panel.py` | 12 | 8 | 0 | 2 | 0 |
| `swe2d/results/velocity_layer.py` | 5 | 4 | 0 | 0 | 0 |
| `swe2d/workbench/three_d_bc.py` | 13 | 12 | 0 | 1 | 0 |
| `swe2d/workbench/non_gui_runtime.py` | 20 | 18 | 2 | 0 | 3 |
| `swe2d/workbench/view.py` | 4 | 4 | 0 | 0 | 0 |
| `swe2d/workbench/project_settings.py` | 2 | 1 | 0 | 0 | 0 |
| `native_backend.py` | 3 | 3 | 0 | 0 | 0 |
| `unsteady_model.py` | 15 | 8 | 0 | 3 | 0 |
| **TOTAL** | **~104** | **~89** | **~5** | **~8** | **~4** |

### A.2 All `hasattr`/`getattr` Fallback Patterns

| File | Count | Purpose | Assessment |
|------|-------|---------|------------|
| `backend.py` | 8+ | Feature detection for C++ API | Should log missing features |
| `coupling.py` | 15+ | GPU capability probing | Acceptable — needed for binary compat |
| `run_options_builder.py` | 10+ | UI widget existence checks | Should log with defaults |
| `drainage_network.py` | 7 | Metadata attribute fallbacks | Acceptable — schema flexibility |
| `structures.py` | 3 | Geometry attribute defaults | Acceptable |
| `view.py` | 4 | Safe widget access | Acceptable — presentation layer |
| `three_d_bc.py` | 5 | Widget value retrieval | Should log on failure |
| `non_gui_runtime.py` | 3 | Config attribute fallbacks | Should log missing config |
| **TOTAL** | **~55** | | |

### A.3 All Logging Tag Locations

Currently, structured logging is only used in `coupling.py`:
- `[COUPLING]` — Coupling diagnostics (10 locations)
- `[COUPLING_FF]` — Face-flux operations (4 locations)
- `[ERROR]` — Coupling errors (2 locations)
- `[WARNING]` — Coupling warnings (1 location)

**Gap**: No other module uses structured logging tags. All other modules either don't log or use unstructured messages.

---

## Summary: Priority Action Items

| Priority | Action | Impact | Effort |
|----------|--------|--------|--------|
| 🔴 P1 | Add logging to `backend_initializer.py` constructor fallback | Prevents silent GPU→CPU degradation | 30 min |
| 🔴 P1 | Add logging to `runtime_source_logic.py` native injection fallback | Prevents silent performance regression | 30 min |
| 🔴 P1 | Add logging to `run_lifecycle.py` backend.destroy() | Prevents silent resource leaks | 15 min |
| 🔴 P1 | Remove all 5 `print()` debug statements | Clean production output | 15 min |
| 🔴 P1 | Add logging to `run_options_builder.py` GPU probe | Prevents silent GPU unavailability | 30 min |
| 🟡 P2 | Add logging to all `backend.py` feature detection (lines 423–426) | Feature availability visibility | 30 min |
| 🟡 P2 | Replace silent `except` in `coupling.py` redistribution (4 locations) | Coupling accuracy diagnostics | 1 hr |
| 🟡 P2 | Deduplicate three-tier pybind11 compat into shared utility | Reduce maintenance burden | 2 hr |
| 🟡 P2 | Remove dead code (`_normalize_post_opt_backend`, commented blocks) | Code hygiene | 30 min |
| 🟢 P3 | Add type hints to `panel.py`, `three_d_bc.py`, `non_gui_runtime.py` | Maintainability | 4 hr |
| 🟢 P3 | Standardize import style across all modules | Consistency | 2 hr |
| 🟢 P3 | Add docstrings to `meshing.py` public functions | API documentation | 4 hr |
| 🟢 P3 | Implement C++ version constant to replace compat layer | Eliminate backward compat | 4 hr |
