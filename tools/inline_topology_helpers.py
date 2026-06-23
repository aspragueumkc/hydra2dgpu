#!/usr/bin/env python3
"""Inline all helper calls in topology_and_io_methods.py"""

import re
import py_compile

_script_dir = os.path.dirname(os.path.abspath(__file__))
fpath = os.path.join(_script_dir, "..", "swe2d", "workbench", "extracted", "topology_and_io_methods.py")
with open(fpath) as f:
    c = f.read()

# -------------------
# Remove helper defs
# -------------------

# _find_child_robust
c = re.sub(
    r"    def _find_child_robust\(widget_type: type, name: str\):\n"
    r'        """.*?"""\n'
    r"        children = topology_tab_page\.findChildren\(widget_type, name\)\n"
    r"        return children\[0\] if children else None\n",
    "", c, flags=re.DOTALL
)

# Helper def removals - more aggressive: remove from "def _" to next "    def" or blank-line-then-def
# Simpler: remove all remaining def lines for these helpers plus their body (4 spaces indent)
import os
lines = c.split('\n')
result = []
skip = False
for i, line in enumerate(lines):
    # Detect start of a helper def
    if re.match(r'    def _(find_child_robust|find_or_create_combo|find_or_create_double_spin|find_or_create_spin|find_or_create_line_edit|find_or_create_check|find_or_create_form_container|ensure_form_row|reconnect)\(', line):
        skip = True
        continue
    # End of helper body: next line at 4-space indent that's blank or next def
    if skip:
        # Skip until we hit a line that's not indented (ends body)
        if line.strip() == '' or re.match(r'    (def |self\.|#)', line):
            skip = False
            # Fall through to add this line
        else:
            continue
    if not skip:
        result.append(line)
c = '\n'.join(result)

# Remove orphaned blank def bodies
c = re.sub(r'\n{4,}', '\n\n\n', c)

# -------------------
# Replace call sites
# -------------------

# _find_or_create_combo("name", N)
c = re.sub(
    r"self\.(\w+)\s*=\s*_find_or_create_combo\(\"(\w+)\",\s*(\d+)\)",
    r"self.\1 = QtWidgets.QComboBox()\n    self.\1.setObjectName(\"\2\")\n    _ensure(self.\1, \3, 1)",
    c
)

# _find_or_create_double_spin("name") - single line
c = re.sub(
    r"self\.(\w+)\s*=\s*_find_or_create_double_spin\(\"(\w+)\"\)",
    r"self.\1 = QtWidgets.QDoubleSpinBox()\n    self.\1.setObjectName(\"\2\")",
    c
)

# _find_or_create_double_spin("name") - multi line
c = re.sub(
    r"self\.(\w+)\s*=\s*_find_or_create_double_spin\(\n\s+\"(\w+)\"\n\s+\)",
    r"self.\1 = QtWidgets.QDoubleSpinBox()\n    self.\1.setObjectName(\"\2\")",
    c
)

# _find_or_create_spin("name") - single
c = re.sub(
    r"self\.(\w+)\s*=\s*_find_or_create_spin\(\"(\w+)\"\)",
    r"self.\1 = QtWidgets.QSpinBox()\n    self.\1.setObjectName(\"\2\")",
    c
)

# _find_or_create_line_edit("name", "text") - single
c = re.sub(
    r'self\.(\w+)\s*=\s*_find_or_create_line_edit\(\"(\w+)\",\s*\"([^"]+)\"\)',
    r'self.\1 = QtWidgets.QLineEdit("\3")\n    self.\1.setObjectName("\2")\n    if not str(self.\1.text() or "").strip():\n        self.\1.setText("\3")',
    c
)

# _find_or_create_line_edit multi-line
c = re.sub(
    r'self\.(\w+)\s*=\s*_find_or_create_line_edit\(\n\s+\"(\w+)\",\n\s+\"([^"]+)\"\n\s+\)',
    r'self.\1 = QtWidgets.QLineEdit("\3")\n    self.\1.setObjectName("\2")\n    if not str(self.\1.text() or "").strip():\n        self.\1.setText("\3")',
    c
)

# _find_or_create_check("name", "text") - single
c = re.sub(
    r'self\.(\w+)\s*=\s*_find_or_create_check\(\"(\w+)\",\s*\"([^"]+)\"\)',
    r'self.\1 = QtWidgets.QCheckBox("\3")\n    self.\1.setObjectName("\2")\n    if not str(self.\1.text() or "").strip():\n        self.\1.setText("\3")',
    c
)

# _find_or_create_check multi-line (2 lines of args)
c = re.sub(
    r'self\.(\w+)\s*=\s*_find_or_create_check\(\n\s+\"(\w+)\",\n\s+\"([^"]+)\"\n\s+\)',
    r'self.\1 = QtWidgets.QCheckBox("\3")\n    self.\1.setObjectName("\2")\n    if not str(self.\1.text() or "").strip():\n        self.\1.setText("\3")',
    c
)

# _find_or_create_check multi-line (3 lines of args)
c = re.sub(
    r'self\.(\w+)\s*=\s*_find_or_create_check\(\n\s+\"(\w+)\",\n\s+\"([^"]+)\",\n\s+\)',
    r'self.\1 = QtWidgets.QCheckBox("\3")\n    self.\1.setObjectName("\2")\n    if not str(self.\1.text() or "").strip():\n        self.\1.setText("\3")',
    c
)

# _find_or_create_form_container("name", row, col=N)
c = re.sub(
    r'(\w+)\s*=\s*_find_or_create_form_container\(\"(\w+)\",\s*(\d+),\s*col=(\d+)\)',
    r'\1 = QtWidgets.QWidget()\n    \1.setObjectName("\2")\n    _ensure(\1, \3, \4, 1, 1)\n    layout = \1.layout()\n    if not isinstance(layout, QtWidgets.QFormLayout):\n        layout = QtWidgets.QFormLayout(\1)\n    layout.setContentsMargins(0, 0, 0, 0)',
    c
)

# _find_child_robust with if-None fallback
c = re.sub(
    r'self\.(\w+)\s*=\s*_find_child_robust\(QtWidgets\.\w+,\s*\"(\w+)\"\)\n'
    r'\s+if self\.\1 is None:\n'
    r'\s+self\.\1\s*=\s*(\w+)\(([^)]*)\)\n'
    r'\s+self\.\1\.setObjectName\("\2"\)',
    r'self.\1 = \3(\4)\n    self.\1.setObjectName("\2")',
    c
)

# _find_child_robust with if-None fallback (local var)
c = re.sub(
    r'(\w+)\s*=\s*_find_child_robust\(QtWidgets\.\w+,\s*\"(\w+)\"\)\n'
    r'\s+if \1 is None:\n'
    r'\s+\1\s*=\s*(\w+)\(([^)]*)\)\n'
    r'\s+\1\.setObjectName\("\2"\)',
    r'\1 = \3(\4)\n    \1.setObjectName("\2")',
    c
)

# _find_child_robust without if-None (rare, should not happen)
c = re.sub(
    r'self\.(\w+)\s*=\s*_find_child_robust\([^)]+\)',
    r'self.\1 = None  # placeholder',
    c
)

# _ensure_form_row(form, widget, "label")
c = re.sub(
    r'_ensure_form_row\((\w+), (self\.\w+),\s*\"([^"]+)\"\)',
    r'\1.addRow("\3", \2)',
    c
)

# _ensure_form_row(form, widget)
c = re.sub(
    r'_ensure_form_row\((\w+), (self\.\w+)\)',
    r'\1.addRow(\2)',
    c
)

# _reconnect(signal, callback)
c = re.sub(
    r'_reconnect\(([^,]+),\s*([^)]+)\)',
    r'safe_disconnect(\1, \2)\n    \1.connect(\2)',
    c
)

# Clean up
c = re.sub(r'\n{4,}', '\n\n\n', c)

with open(fpath, 'w') as f:
    f.write(c)

# Verify
py_compile.compile(fpath, doraise=True)
r = c.count('_find_child_robust') + c.count('_find_or_create_') + c.count('_ensure_form_row') + c.count('_reconnect(')
print(f'SYNTAX OK — remaining: {r}')
