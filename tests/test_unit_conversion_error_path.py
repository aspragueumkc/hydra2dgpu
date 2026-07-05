def test_update_unit_system_logs_crs_failures_via_log_fn():
    from swe2d.workbench.services import unit_conversion_service as ucs

    captured = []

    try:
        ucs.update_unit_system_from_crs(
            have_qgis_core=True,
            project=None,
            log_fn=lambda m: captured.append(m),
            force_failure=True,
        )
    except RuntimeError:
        pass
    assert any("ERROR" in m for m in captured)
