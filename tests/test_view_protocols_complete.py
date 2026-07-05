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
