# Qt GUI UX Improvement Recommendations

**Audit Date:** 2026-07-02  
**Scope:** 14 view files, 14 dialogs, workbench host  
**Method:** Static audit of Qt widget hierarchy, layout patterns, and tooltip coverage

---

## Summary

This audit identified **10 easy-win UX improvements** ranked by impact-to-effort ratio. All changes are localized refactors — no architectural changes needed. They add visual hierarchy, discoverability, and muscle-memory ergonomics without breaking existing logic.

---

## Top 10 Improvements

### 🎯 High-Impact (Visual Hierarchy)

#### 1. Add `QGroupBox` sections to Model tab Solver Parameters

**File:** `swe2d/workbench/views/model_tab_view.py`

**Problem:** 22 controls in a single long scrollable `QFormLayout` — no visual sub-grouping. Users get lost scrolling.

**Fix:** Wrap related controls in `QGroupBox` titled sections:
- `Manning & Friction`
- `Time Stepping (CFL)`
- `Tiny Mode & Performance`
- `Reconstruction & Temporal Order`
- `Internal Flow`

**Effort:** Low (refactor existing `addRow` calls into nested layouts)

---

#### 2. Visually disable `QToolBox` pages that are unavailable

**File:** `swe2d/workbench/views/topology_tab_view.py`

**Problem:** 8 QToolBox pages, but 6 (Algorithm, Arcs, Sizing, Threading, Transfinite, Quality) are disabled when backend ≠ Gmsh. Users see them as clickable but nothing happens.

**Fix:** Either:
- **a)** Hide disabled pages entirely (when no Gmsh mesh)
- **b)** Dim the page title text + add `(Gmsh only)` suffix

**Effort:** Low (override `QToolBox` paint or use `setItemText` + `setItemEnabled`)

---

#### 3. Disable child widgets when parent checkbox is unchecked

**File:** `swe2d/workbench/views/results_controls.py` (Overlay page)

**Problem:** Arrow density/length/head and streamline seeds/steps spinboxes remain active even when `arrows_chk` / `streamlines_chk` are off. Users can change them but nothing happens.

**Fix:** Connect `toggled(bool)` signal on each checkbox to a `setEnabled(bool)` slot that enables/disables its child controls.

**Effort:** Low (5 checkbox→children mappings)

---

### 🔍 Discoverability

#### 4. Add a search/filter bar to the Model tab parameters

**File:** `swe2d/workbench/views/model_tab_view.py`

**Problem:** 50+ controls across 5 QToolBox pages — finding a specific setting requires manual scrolling through each page.

**Fix:** Add a `QLineEdit` above the QToolBox with placeholder "🔍 Filter parameters...". On `textChanged`, iterate all labels and hide rows that don't match.

**Effort:** Low (one QLineEdit + filter lambda)

---

#### 5. Add context-snippets to DocHubWidget search results

**File:** `swe2d/workbench/views/doc_viewer.py`

**Problem:** Search shows only doc titles — easy to pick the wrong doc.

**Fix:** Store ~200 chars of preview text per doc; show first matching line + … in the result list.

**Effort:** Low (modify `_SearchHit` dataclass to include `snippet: str`)

---

#### 6. Add "Invert selection" + "Only newest" to Run Selection dialog

**File:** `swe2d/workbench/dialogs/run_selection_dialog.py`

**Problem:** Users with many GPKG runs must manually uncheck 20+ items.

**Fix:** Two small buttons:
- `Invert` — toggles all checkboxes
- `Only newest` — leaves only the run with the most recent `created_utc`

**Effort:** Low (iterate `QListWidget` items)

---

### 📐 Layout Polish

#### 7. Fix the column-cycling grid in Map → Mesh Setup

**File:** `swe2d/workbench/views/map_tab_view.py`

**Problem:** Uses `col = (col + cspan) % 2` cycling where every button spans both columns — the rotation is meaningless; it's effectively a vertical list pretending to be a grid.

**Fix:** Replace with a simple `QFormLayout` (one button per row with a label) or `QGridLayout` with explicit `(row, col)` coordinates.

**Effort:** Low (5-line refactor)

---

#### 8. Standardize button sizing

**Problem:** Mixed sizes (24×22, 20 px, full-width grid) make the UI feel inconsistent.

**Fix:** Pick three canonical sizes:
- **Small icon button:** 24×24 (toolbars, save menu)
- **Action button:** minimum 80×28 (dialog actions)
- **Primary action:** minimum 100×32 with bold text (Run, Apply)

Apply via a `_size_button(btn, role)` helper.

**Effort:** Low–Medium (touches many files, but each change is mechanical)

---

#### 9. Swap coupling dialog splitter defaults

**File:** `swe2d/workbench/dialogs/coupling_results_dialog.py`

**Problem:** Splitter defaults to `[380, 220]` — table larger than plot, inverting the expected "plot is the main view" pattern.

**Fix:** Reverse to `[220, 380]` or use `setSizes([QApplication.primaryScreen().size().height() // 3, ...])`.

**Effort:** Trivial (one-line swap)

---

### ⌨️ Ergonomics

#### 10. Add keyboard shortcuts for common actions

**Problem:** Every action is mouse-only — Run, Refresh, Open Viewer, etc.

**Fix:** Add `QShortcut` or set `setShortcut()` on key actions:
- `Ctrl+R` — Run
- `Ctrl+.` — Cancel (period = stop)
- `F5` — Refresh results
- `Ctrl+S` — Save run config
- `Ctrl+O` — Open GeoPackage

Document in a single `KEYBOARD_SHORTCUTS` constant in `studio_dialog.py`.

**Effort:** Low (one shortcut per action, central doc)

---

## Summary Table

| # | Fix | Effort | Files Touched |
|---|-----|--------|---------------|
| 1 | QGroupBox sections in Solver Parameters | Low | 1 |
| 2 | Visually disable unavailable topology pages | Low | 1 |
| 3 | Disable children when checkbox unchecked | Low | 1 |
| 4 | Filter bar for Model tab parameters | Low | 1 |
| 5 | Doc search snippets | Low | 1 |
| 6 | Run selection "Invert" + "Only newest" | Low | 1 |
| 7 | Fix column-cycling grid in Mesh Setup | Low | 1 |
| 8 | Standardize button sizing | Low–Med | Many |
| 9 | Swap coupling dialog splitter defaults | Trivial | 1 |
| 10 | Keyboard shortcuts | Low | 1–2 |

---

## Additional Findings (Beyond Top 10)

These were noted during the audit but didn't make the top 10:

- **`hydrograph_editor.py`** — No input validation on time/value columns; consider `QDoubleValidator`
- **`topo_attr_table_dialog.py`** — Non-enum fields use raw `QTableWidgetItem`; consider inline spinboxes for int/float
- **`simulation_config_dialog.py`** — No search/filter for saved configs
- **`batch_simulation_dialog.py`** — "Apply to Selected" vs "Apply to All" labels are unclear
- **`gpkg_explorer_dialog.py`** — "Delete by Run ID" is rare; move to "More" menu
- **`studio_viewer.py`** — `setMinimumWidth(0)` on all children disables size hints globally — too broad
- **Cross-tab** — No theme/dark-mode handling; matplotlib canvas hardcoded white
- **Cross-tab** — No global "Save/Load preset" button; config save/load buried in Run page
- **`model_tab_view.py:148`** — Temporal order disable logic only fires on first form open, not on data restore
- **All toolboxes** — No way to collapse the QToolBox itself; only pages can be collapsed
- **`studio_viewer_plot.py`** — Matplotlib retained for Mesh tab; toolbar icon size hardcoded 16×16

---

## Audit Methodology

Reviewed all files under:
- `swe2d/workbench/views/` (14 files)
- `swe2d/workbench/dialogs/` (14 files)
- `swe2d/workbench/forms/` (1 file)
- `swe2d/workbench/services/` (17 files — GUI-allowed, but contain no widget trees)

Mapped:
- Tab widgets and tab names
- Container types (`QGroupBox`, `QSplitter`, `QTabWidget`, `QStackedWidget`)
- Layout patterns (`QVBoxLayout`, `QHBoxLayout`, `QGridLayout`, `QFormLayout`)
- Widget type usage (combos, spins, checks, buttons, tables)
- Tooltip coverage
- Common patterns and clutter sources