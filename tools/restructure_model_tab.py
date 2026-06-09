#!/usr/bin/env python3
"""
Restructure swe2d_model_tab.ui into a QToolBox with Solver / Rain / Drainage pages,
and extract the 3D Patch group into its own .ui file.
"""

import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

UI_DIR = Path(__file__).resolve().parent.parent / "forms"
MODEL_UI = UI_DIR / "swe2d_model_tab.ui"
PATCH_UI = UI_DIR / "swe2d_3d_patch_tab.ui"
BACKUP = MODEL_UI.with_suffix(".ui.bak")


# ── Widget name → category mapping ──────────────────────────────
SOLVER_PATTERNS = [
    "n_mann", "cfl_", "h_min", "initial_", "adaptive_cfl", "dt_", "gpu_diag",
    "tiny_mode", "tiny_wet_cell", "enable_cuda", "reconstruction", "equation_set",
    "godunov_mode", "temporal_order", "degen_mode", "depth_cap", "cfl_lambda_cap",
    "front_flux", "shallow_damping", "shallow_front", "max_rel_depth", "max_source",
    "max_inv_area", "ia_ratio", "source_cfl", "source_imex", "source_true",
    "source_stage", "source_max", "momentum_cap", "active_set", "extreme_rain_mode", "unit_system", "gpu_default",
]

RAIN_PATTERNS = [
    "rain_rate", "rainfall", "storm_area", "cn_default", "cn_", "use_spatial",
    "rain_boundary", "hyetograph", "infiltration", "internal_flow",
]

DRAIN_PATTERNS = [
    "drain", "coupling_loop", "structure", "patch_3d",
]


def category_for_widget(name: str) -> str:
    """Return 'solver', 'rain', 'drain', or 'other'."""
    name_lower = name.lower()
    for pat in DRAIN_PATTERNS:
        if pat in name_lower:
            return "drain"
    for pat in RAIN_PATTERNS:
        if pat in name_lower:
            return "rain"
    for pat in SOLVER_PATTERNS:
        if pat in name_lower:
            return "solver"
    return "other"


def extract_patch_ui():
    """Extract patch_3d_group from model ui into its own file."""
    tree = ET.parse(MODEL_UI)
    root = tree.getroot()

    # Find patch_3d_group widget
    patch_group = None
    for w in root.iter("widget"):
        if w.get("name") == "patch_3d_group":
            patch_group = w
            break

    if patch_group is None:
        print("ERROR: patch_3d_group not found in model.ui")
        return False

    # Build new .ui for 3D patch
    patch_root = ET.Element("ui", version="4.0")
    patch_class = ET.SubElement(patch_root, "class")
    patch_class.text = "SWE2D3DPatchTabPage"

    patch_widget = ET.SubElement(patch_root, "widget", {"class": "QWidget", "name": "SWE2D3DPatchTabPage"})
    layout = ET.SubElement(patch_widget, "layout", {"class": "QVBoxLayout", "name": "verticalLayout"})
    margin_props = [("leftMargin", "0"), ("topMargin", "0"), ("rightMargin", "0"), ("bottomMargin", "0")]
    for k, v in margin_props:
        p = ET.SubElement(layout, "property", {"name": k})
        n = ET.SubElement(p, "number")
        n.text = v

    item = ET.SubElement(layout, "item")

    # Deep copy patch_group into item
    import copy
    item.append(copy.deepcopy(patch_group))

    # Remove patch_3d_group from original
    parent = None
    for p in root.iter():
        if patch_group in list(p):
            parent = p
            break
    if parent is not None:
        parent.remove(patch_group)

    # Write patch .ui
    tree2 = ET.ElementTree(patch_root)
    ET.indent(tree2, "  ")
    tree2.write(PATCH_UI, encoding="UTF-8", xml_declaration=True)
    print(f"Created {PATCH_UI}")

    # Save updated model .ui (without patch group)
    ET.indent(tree, "  ")
    tree.write(MODEL_UI, encoding="UTF-8", xml_declaration=True)
    print(f"Updated {MODEL_UI} (patch_3d_group removed)")
    return True


def restructure_model_tab():
    """Wrap model_group in a QToolBox with categorized pages."""
    tree = ET.parse(MODEL_UI)
    root = tree.getroot()

    # Find model_group
    model_group = None
    for w in root.iter("widget"):
        if w.get("name") == "model_group":
            model_group = w
            break

    if model_group is None:
        print("ERROR: model_group not found")
        return False

    # Get the form layout
    form = model_group.find("layout")
    if form is None or form.get("name") != "model_param_form":
        print("ERROR: model_param_form not found")
        return False

    items = list(form)
    categorized = {"solver": [], "rain": [], "drain": [], "other": []}

    for item in items:
        widget_el = item.find("widget")
        if widget_el is not None:
            name = widget_el.get("name", "")
            cat = category_for_widget(name)
        else:
            cat = "other"
        categorized[cat].append(item)
        form.remove(item)

    # Build QToolBox with 3 pages
    toolbox = ET.Element("widget", {"class": "QToolBox", "name": "toolBox"})

    # Size policy
    sp = ET.SubElement(toolbox, "property", {"name": "sizePolicy"})
    sp_inner = ET.SubElement(sp, "sizepolicy", {"hsizetype": "Preferred", "vsizetype": "Expanding"})
    hs = ET.SubElement(sp_inner, "horstretch")
    hs.text = "0"
    vs = ET.SubElement(sp_inner, "verstretch")
    vs.text = "1"

    # Bold tabs
    ss = ET.SubElement(toolbox, "property", {"name": "styleSheet"})
    ss_str = ET.SubElement(ss, "string", {"notr": "true"})
    ss_str.text = "QToolBox::tab { font-weight: bold; }"

    # Current index
    ci = ET.SubElement(toolbox, "property", {"name": "currentIndex"})
    ci_n = ET.SubElement(ci, "number")
    ci_n.text = "0"

    pages = [
        ("solver", "Solver Parameters"),
        ("rain", "Rain / Hydrology"),
        ("drain", "Structures & Drainage"),
    ]

    for cat_key, page_label in pages:
        page = ET.Element("widget", {"class": "QWidget", "name": f"model_{cat_key}_page"})
        label_attr = ET.SubElement(page, "attribute", {"name": "label"})
        label_str = ET.SubElement(label_attr, "string")
        label_str.text = page_label

        page_layout = ET.SubElement(page, "layout", {"class": "QVBoxLayout", "name": f"model_{cat_key}_layout"})
        item_wrapper = ET.SubElement(page_layout, "item")
        page_form = ET.SubElement(item_wrapper, "layout", {"class": "QFormLayout", "name": f"model_{cat_key}_form"})

        for item in categorized[cat_key]:
            page_form.append(item)

        toolbox.append(page)

    # Add any 'other' items to solver page as fallback
    if categorized["other"]:
        solver_page = toolbox.find("widget/[@name='model_solver_page']")
        if solver_page is not None:
            form_el = solver_page.find(".//layout/[@name='model_solver_form']")
            if form_el is not None:
                for item in categorized["other"]:
                    form_el.append(item)

    # Replace model_group with toolbox in the root layout
    root_layout = root.find("layout/[@name='verticalLayout']")
    if root_layout is None:
        # Find the root layout differently
        root_layout = root.find(".//layout/[@class='QVBoxLayout']")

    # Find the item containing model_group and replace its child
    parent_item = None
    for item in root_layout.findall("item"):
        for child in item:
            if child.tag == "widget" and child.get("name") == "model_group":
                parent_item = item
                break

    if parent_item is not None:
        parent_item.clear()
        # The item now wraps the toolbox
        # We need to set up the item to point to our toolbox
        # Actually, let's just replace the widget reference
        child_widget = parent_item.find("widget")
        if child_widget is not None:
            parent_item.remove(child_widget)
        else:
            for child in list(parent_item):
                parent_item.remove(child)
    
    # Simpler approach: just put toolbox in the item
    parent_item.append(toolbox)

    ET.indent(tree, "  ")
    tree.write(MODEL_UI, encoding="UTF-8", xml_declaration=True)
    print(f"Restructured {MODEL_UI}")
    return True


def main():
    # Backup original
    shutil.copy2(MODEL_UI, BACKUP)
    print(f"Backup saved to {BACKUP}")

    # Step 1: Extract 3D patch into its own .ui
    if not extract_patch_ui():
        print("Failed to extract patch UI, aborting")
        return

    # Step 2: Restructure model tab into QToolBox
    if not restructure_model_tab():
        print("Failed to restructure model tab")
        return

    print("\nDone. Run the ui_bind_sync.py script to verify.")


if __name__ == "__main__":
    main()
