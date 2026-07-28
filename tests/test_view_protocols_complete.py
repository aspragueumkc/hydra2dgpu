import unittest
"""Verify RunView/MeshView/TopologyMeshView protocols have dialog methods."""


def test_run_view_has_dialog_methods():
    from swe2d.workbench.controllers.protocols_controller import RunView
    for method in (
        "show_critical_message", "show_information_message",
        "show_warning_message", "get_open_file_name",
        "get_save_file_name", "get_input_text",
        "get_results_gpkg_path", "show_mesh_tab",
    ):
        assert hasattr(RunView, method), f"RunView missing {method}"


def test_mesh_view_has_dialog_methods():
    from swe2d.workbench.controllers.protocols_controller import MeshView
    for method in (
        "get_open_file_name", "get_save_file_name",
        "show_warning_message", "show_information_message",
        "show_critical_message", "show_mesh_tab",
    ):
        assert hasattr(MeshView, method), f"MeshView missing {method}"


def test_topology_view_has_dialog_methods():
    from swe2d.workbench.controllers.protocols_controller import TopologyMeshView
    for method in (
        "show_open_file_name", "show_warning_message",
        "show_information_message", "show_question_message",
        "get_topo_status", "show_mesh_tab",
        "create_timer", "stop_timer",
    ):
        assert hasattr(TopologyMeshView, method), f"TopologyMeshView missing {method}"


def test_overlay_view_has_color_range():
    from swe2d.workbench.controllers.protocols_controller import OverlayView
    assert hasattr(OverlayView, "set_overlay_color_range")

class _PytestStyleWrapper(unittest.TestCase):
    """Auto-generated wrapper for module-level test functions.

    Created by tools/wrap_pytest_style.py so that pytest-style tests
    (def test_* at module level) become visible to `python3 -m unittest`.
    Each module-level test is attached as a staticmethod so it can be
    discovered and run as a unittest TestCase.
    """
__wrapped_funcs = []
for _name, _obj in list(globals().items()):
    if _name.startswith("test_") and callable(_obj) and not isinstance(_obj, type):
        setattr(_PytestStyleWrapper, _name, staticmethod(_obj))
        __wrapped_funcs.append(_name)
for _name in __wrapped_funcs:
    del globals()[_name]
del _name, _obj, __wrapped_funcs
