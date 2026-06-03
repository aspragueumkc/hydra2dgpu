#!/usr/bin/env python3
"""
ui_bind_sync.py — Sync Python bind code with .ui file widget definitions.

Usage:
    python tools/ui_bind_sync.py forms/swe2d_map_tab.ui swe2d_workbench_qt.py
    python tools/ui_bind_sync.py forms/swe2d_map_tab.ui swe2d/workbench/extracted/results_and_ui_methods.py
    python tools/ui_bind_sync.py forms/swe2d_model_tab.ui swe2d_workbench_qt.py --missing

Modes:
    (default)   Remove Python bind code for widgets DELETED from the .ui file.
    --missing   Report widgets in the .ui that have NO Python _find_or_create_*
                binding AND report form layouts without bind methods (new tabs).
    --dry-run   Print what would be removed/added without modifying files.

What it does (default / delete mode):
    1. Parses the .ui file for all <widget name="..."> entries.
    2. Scans the Python file for `self.xxx = _find_or_create_*("xxx", ...)` and
       `self.xxx = map_tab_page.findChild(..., "xxx")` assignments.
    3. For any widget name that exists in the Python code but NOT in the .ui file,
       removes every line that references it (definitions, addWidget guards,
       signal wiring, setToolTip, setText, combo population, etc.).
    4. Writes the cleaned Python file (or prints a diff with --dry-run).

What --missing does:
    1. Parses the .ui for all widget names AND all form layout names.
    2. Scans Python files for all _find_or_create_* calls and bind method names.
    3. Reports widgets in the .ui that lack Python bindings.
    4. Reports form layouts whose widgets are mostly unbound (suggests new tab).
    5. Reports form layouts that have no matching bind method in any scanned file.

The .ui file is the source of truth — delete a widget there, run default mode,
and all its wiring vanishes from the Python bind methods.
Add a widget/tab there, run --missing, and see what needs Python attention.
"""

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def get_ui_widget_names(ui_path: str) -> set[str]:
    """Return the set of all widget object names in a Qt .ui file."""
    tree = ET.parse(ui_path)
    names: set[str] = set()
    for elem in tree.getroot().iter("widget"):
        name = elem.get("name")
        if name:
            names.add(name)
    return names


def find_python_widget_attrs(py_path: str) -> dict[str, str]:
    """Return {self_attr_name: ui_widget_name} for every find-or-create pattern."""
    pat_def = re.compile(
        r"self\.(\w+)\s*=\s*(?:"
        r"_find_or_create_\w+\(\s*\"(\w+)\""
        r"|"
        r"map_tab_page\.findChild\([^,]+,\s*\"(\w+)\""
        r")"
    )
    result: dict[str, str] = {}
    with open(py_path) as f:
        for line in f:
            for m in pat_def.finditer(line):
                name_in_ui = m.group(2) or m.group(3)
                result[m.group(1)] = name_in_ui
    return result


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def process_file(py_path: str, orphaned_attrs: set[str], dry_run: bool = False) -> int:
    """Remove lines referencing orphaned widget attributes.

    Returns the number of lines removed.
    """
    with open(py_path) as f:
        lines = f.readlines()

    removed: set[int] = set()

    # ------------------------------------------------------------------
    # Collect line ranges to remove
    # ------------------------------------------------------------------

    # 1. Simple single-line removals: any line matching simple patterns
    #    that reference an orphaned attribute.
    pat_simple = re.compile(
        r"self\.(" + "|".join(re.escape(a) for a in orphaned_attrs) + r")\b"
    )

    # Pattern for if-indexOf blocks:
    #   if layout.indexOf(self.xxx) < 0:
    #       layout.addWidget(self.xxx, ...)
    pat_indexOf = re.compile(
        r"if\s+\w+\.indexOf\(self\.("
        + "|".join(re.escape(a) for a in orphaned_attrs)
        + r")\)\s*[<>!]=\s*\d+\s*:"
    )

    # Pattern for signal disconnect blocks:
    #   try:
    #       self.xxx.clicked.disconnect(...)
    #   except Exception:
    #       pass
    #   self.xxx.clicked.connect(...)
    pat_try_disconnect = re.compile(
        r"try\s*:"
    )
    pat_connect = re.compile(
        r"self\.(" + "|".join(re.escape(a) for a in orphaned_attrs) + r")\.clicked\.connect\("
    )

    # Pattern for for-btn-cb tuple entries:
    #   (self.xxx, self._handler),
    pat_tuple_entry = re.compile(
        r"^\s*\(self\.("
        + "|".join(re.escape(a) for a in orphaned_attrs)
        + r"),\s*self\.\w+\)\s*,?\s*$"
    )

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # --- if-indexOf blocks ---
        if (m := pat_indexOf.search(stripped)):
            # Remove the if line and the following indented block lines
            removed.add(i)
            base_indent = _indent(line)
            j = i + 1
            while j < len(lines) and _indent(lines[j]) > base_indent:
                removed.add(j)
                j += 1
            i = j
            continue

        # --- try/disconnect/connect blocks ---
        if pat_try_disconnect.search(stripped) and i + 3 < len(lines):
            # Check if next lines reference orphaned attrs
            next_3 = "".join(lines[i : i + 4])
            if pat_connect.search(next_3):
                for k in range(4):
                    removed.add(i + k)
                i += 4
                continue

        # --- for btn, cb in (...) tuple entries ---
        if pat_tuple_entry.search(stripped):
            removed.add(i)
            i += 1
            continue

        # --- pattern matching for any self.<orphaned> reference ---
        # We handle multi-line _find_or_create_* calls by tracking parentheses
        if pat_simple.search(stripped):
            # Check if this is a definition line or method call
            # Skip _find_or_create_* lines (handled below with paren matching)
            is_def = bool(
                re.search(r"self\.\w+\s*=\s*(_find_or_create_\w+|map_tab_page\.findChild)\(", stripped)
                and any(f"self.{a}" in stripped for a in orphaned_attrs)
            )

            if is_def:
                # Remove the definition line and any continuation lines
                # (multi-line _find_or_create_* calls)
                removed.add(i)
                if "(" in stripped and ")" not in stripped:
                    j = i + 1
                    while j < len(lines) and ")" not in lines[j] and "):" not in lines[j]:
                        removed.add(j)
                        j += 1
                    if j < len(lines):
                        removed.add(j)  # close paren line
                i += 1
                continue

            # Otherwise it's a simple method call on the orphaned widget
            # But be careful: we need to handle multi-line setToolTip etc.
            # Check if this starts a multi-line call
            if "(" in stripped and ")" not in stripped and not stripped.endswith("):"):
                # Multi-line method call — remove until close paren
                removed.add(i)
                j = i + 1
                while j < len(lines) and ")" not in lines[j]:
                    removed.add(j)
                    j += 1
                if j < len(lines):
                    removed.add(j)
                i = j + 1
                continue

            # Single line referencing orphaned widget
            # But skip if this is inside a for-btn-cb tuple block we already handle
            if not stripped.startswith("("):
                removed.add(i)

        i += 1

    # ------------------------------------------------------------------
    # Apply removals (build new file)
    # ------------------------------------------------------------------
    new_lines = [line for idx, line in enumerate(lines) if idx not in removed]

    removed_count = len(lines) - len(new_lines)

    if dry_run:
        print(f"--- {py_path}  (dry-run, {removed_count} lines would be removed)")
        for idx in sorted(removed):
            print(f"- {idx+1:5d}: {lines[idx].rstrip()}")
    else:
        with open(py_path, "w") as f:
            f.writelines(new_lines)
        print(f"--- {py_path}  ({removed_count} lines removed)")

    return removed_count


# ── Missing-widget / new-tab detection ──────────────────────────────────────

def get_ui_form_layouts(ui_path: str) -> dict[str, list[str]]:
    """Return {form_layout_name: [widget_name, ...]} for all QFormLayouts."""
    tree = ET.parse(ui_path)
    forms: dict[str, list[str]] = {}
    for elem in tree.getroot().iter("layout"):
        if elem.get("class") != "QFormLayout":
            continue
        name = elem.get("name", "")
        if not name:
            continue
        widgets: list[str] = []
        for item in elem:
            for child in item.iter("widget"):
                wname = child.get("name", "")
                if wname:
                    widgets.append(wname)
        forms[name] = widgets
    return forms


def get_ui_toolbox_pages(ui_path: str) -> list[dict]:
    """Return [{page_widget_name, label, form_layouts: [...]}] for QToolBox pages."""
    tree = ET.parse(ui_path)
    pages: list[dict] = []
    for toolbox in tree.getroot().iter("widget"):
        if toolbox.get("class") != "QToolBox":
            continue
        for page in toolbox:
            if page.tag != "widget":
                continue
            page_name = page.get("name", "")
            label = ""
            for attr in page:
                if attr.tag == "attribute" and attr.get("name") == "label":
                    for s in attr.iter("string"):
                        label = (s.text or "").strip()
            forms: list[str] = []
            for layout in page.iter("layout"):
                if layout.get("class") == "QFormLayout":
                    fname = layout.get("name", "")
                    if fname:
                        forms.append(fname)
            if page_name:
                pages.append({"name": page_name, "label": label, "forms": forms})
    return pages


def find_all_python_widget_names(py_paths: list[str]) -> dict[str, set[str]]:
    """Return {ui_widget_name: {python_files}} for all _find_or_create_* calls.

    Handles both single-line and multi-line call patterns by joining file
    content and using DOTALL regex.
    """
    # Match _find_or_create_xxx("widget_name", ...) across lines
    pat = re.compile(
        r"_find_or_create_\w+\(\s*\"(\w+)\"",
        re.DOTALL,
    )
    # Match .findChild(..., "widget_name")
    pat_findchild = re.compile(
        r"\.findChild\([^,]+,\s*\"(\w+)\"",
        re.DOTALL,
    )
    result: dict[str, set[str]] = {}
    for py_path in py_paths:
        with open(py_path) as f:
            content = f.read()
        for m in pat.finditer(content):
            name = m.group(1)
            if name:
                result.setdefault(name, set()).add(py_path)
        for m in pat_findchild.finditer(content):
            name = m.group(1)
            if name:
                result.setdefault(name, set()).add(py_path)
    return result


def find_python_bind_methods(py_paths: list[str]) -> dict[str, str]:
    """Return {bind_method_name: file_path} for _bind_*_controls methods."""
    pat = re.compile(r"def (_bind_\w+_controls)\(")
    result: dict[str, str] = {}
    for py_path in py_paths:
        with open(py_path) as f:
            for line in f:
                m = pat.search(line)
                if m:
                    result[m.group(1)] = py_path
    return result


def find_python_tab_builders(py_paths: list[str]) -> dict[str, str]:
    """Return {_build_*_tab_page_method: file_path}."""
    pat = re.compile(r"def (_build_\w+_tab_page)\(")
    result: dict[str, str] = {}
    for py_path in py_paths:
        with open(py_path) as f:
            for line in f:
                m = pat.search(line)
                if m:
                    result[m.group(1)] = py_path
    return result


def report_missing(ui_path: str, py_paths: list[str]) -> None:
    """Print a report of .ui widgets and forms lacking Python bindings."""
    ui_widgets = get_ui_widget_names(ui_path)
    form_layouts = get_ui_form_layouts(ui_path)
    toolbox_pages = get_ui_toolbox_pages(ui_path)
    py_widget_names = find_all_python_widget_names(py_paths)
    py_bind_methods = find_python_bind_methods(py_paths)
    py_tab_builders = find_python_tab_builders(py_paths)

    # ── 1. Widgets in .ui without Python _find_or_create_* calls ──────
    unbound = sorted(w for w in ui_widgets if w not in py_widget_names)
    if unbound:
        print(f"\n{'='*60}")
        print(f" UNBOUND WIDGETS ({len(unbound)}): widgets in .ui with no Python binding")
        print(f"{'='*60}")
        print(f"  These exist in the .ui file but have no _find_or_create_* call.")
        print(f"  They may be static (labels, spacers) or may NEED bindings added.\n")
        for w in unbound:
            # Classify: is it a label (likely static) or an interactive widget?
            is_label = w.endswith("_lbl") or "label" in w.lower()
            tag = " [likely static]" if is_label else " [NEEDS BINDING?]"
            print(f"  {w}{tag}")
    else:
        print("\n✓ All UI widgets have Python bindings.")

    # ── 2. Form layouts with low binding coverage (new tabs?) ─────────
    print(f"\n{'='*60}")
    print(f" FORM LAYOUT COVERAGE")
    print(f"{'='*60}")
    for form_name, widgets in sorted(form_layouts.items()):
        bound = [w for w in widgets if w in py_widget_names]
        unbound_in_form = [w for w in widgets if w not in py_widget_names]
        pct = 100 * len(bound) / len(widgets) if widgets else 0
        status = "✓" if pct >= 80 else "⚠" if pct >= 30 else "✗"
        print(f"  {status} {form_name}: {len(bound)}/{len(widgets)} bound ({pct:.0f}%)")
        if unbound_in_form and pct < 80:
            interactive = [w for w in unbound_in_form
                           if not w.endswith("_lbl") and w != "label" and "label" not in w.lower()]
            if interactive:
                print(f"     Unbound interactive: {', '.join(interactive)}")

    # ── 3. Toolbox pages without bind methods ──────────────────────────
    if toolbox_pages:
        print(f"\n{'='*60}")
        print(f" TOOLBOX PAGES")
        print(f"{'='*60}")
        for page in toolbox_pages:
            pname = page["name"]
            label = page["label"]
            forms = page["forms"]
            # Try to find a matching tab builder
            # Convention: model_tab.ui pages -> _build_model_tab_page()
            ui_stem = Path(ui_path).stem  # e.g. swe2d_model_tab
            expected_stem = ui_stem.replace("swe2d_", "").replace("_tab", "")  # model
            expected_builder = f"_build_{expected_stem}_tab_page"  # _build_model_tab_page

            builder_file = py_tab_builders.get(expected_builder, None)
            if builder_file:
                print(f"  ✓ page '{label}' ({pname}) → {expected_builder}() in {Path(builder_file).name}")
            else:
                # Try partial match
                matches = [m for m in py_tab_builders if expected_stem in m]
                if matches:
                    print(f"  ~ page '{label}' ({pname}) → partial match: {matches}")
                else:
                    print(f"  ✗ page '{label}' ({pname}) → NO tab builder found!")
                    # Check if forms have bind methods
                    for fname in forms:
                        # Look for a bind method that names this form
                        bind_candidates = [m for m in py_bind_methods if fname.replace("model_", "").replace("_form", "") in m]
                        if bind_candidates:
                            print(f"     form '{fname}' → bind methods: {bind_candidates}")
                        else:
                            print(f"     form '{fname}' → NO bind method (new tab?)")

    # ── 4. Summary ────────────────────────────────────────────────────
    total_unbound = len(unbound) if unbound else 0
    if total_unbound:
        print(f"\n→ Run default mode (without --missing) to clean up deleted widgets.")
        print(f"→ For new widgets, add _find_or_create_* calls in the appropriate bind method.")
        print(f"→ For new tabs, add _build_*_tab_page() and _bind_*_controls() methods.")
    else:
        print(f"\n✓ No action needed — all .ui widgets have Python bindings.")


# ── main ───────────────────────────────────────────────────────────────────


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    ui_path = sys.argv[1]
    py_paths = sys.argv[2:]
    dry_run = False
    missing_mode = False
    if "--dry-run" in py_paths:
        dry_run = True
        py_paths = [p for p in py_paths if p != "--dry-run"]
    if "--missing" in py_paths:
        missing_mode = True
        py_paths = [p for p in py_paths if p != "--missing"]

    if missing_mode:
        report_missing(ui_path, py_paths)
        return

    ui_widgets = get_ui_widget_names(ui_path)
    print(f"UI widgets ({len(ui_widgets)}): {sorted(ui_widgets)}")

    total_removed = 0
    for py_path in py_paths:
        py_attrs = find_python_widget_attrs(py_path)
        orphaned = {attr for attr, name in py_attrs.items() if name not in ui_widgets}
        if not orphaned:
            print(f"--- {py_path}  (no orphaned widgets)")
            continue
        print(f"  Orphaned attrs in {Path(py_path).name}: {sorted(orphaned)}")
        total_removed += process_file(py_path, orphaned, dry_run=dry_run)

    print(f"\nDone. {total_removed} line(s) {'would be' if dry_run else ''} removed.")


if __name__ == "__main__":
    main()
