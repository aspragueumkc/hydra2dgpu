def test_widget_persistence_logs_failures_via_log_fn():
    """The error path must call log_fn, not swallow via inner NameError."""
    from swe2d.workbench.services import widget_persistence_service as wps

    captured = []

    def log_fn(msg):
        captured.append(msg)

    try:
        wps.persist_project_workbench_state(
            have_qgis_core=True,
            qgs_project_cls=object(),
            workbench_state_key="test_key",
            state_obj=object(),
            iter_widgets_fn=iter,
            write_project_json_fn=lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")),
            log_fn=log_fn,
            force_failure=True,
        )
    except RuntimeError:
        pass
    assert any("ERROR" in m for m in captured), captured


def test_widget_persistence_log_fn_called_for_real_errors():
    """When real work fails, log_fn must receive the message."""
    from swe2d.workbench.services import widget_persistence_service as wps

    captured = []

    def boom(**kw):
        raise RuntimeError("real error")

    try:
        wps.persist_project_workbench_state(
            have_qgis_core=True,
            qgs_project_cls=object(),
            workbench_state_key="test_key",
            state_obj=object(),
            iter_widgets_fn=iter,
            write_project_json_fn=boom,
            log_fn=lambda m: captured.append(m),
            force_failure=True,
        )
    except RuntimeError:
        pass
    assert any("ERROR" in m for m in captured)
