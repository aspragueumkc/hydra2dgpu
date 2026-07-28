# SWE2D Studio UI Component Guide

> **Removed 2026-07-26.** This guide described helper methods that never
> shipped: `register_studio_tab`, `get_studio_tab_builders`,
> `_register_left_tab`, `_build_studio_component_docks`,
> `_extract_registered_docks`, `_save_studio_layout_state`,
> `_destroy_component`, `_compose_left_pane`, `_studio_components`
> registry, `_studio_apply_feature_filters`, `_studio_feature_flags`,
> `_wrap_left_tab_page`, `_register_detachable_tab_widget`, and
> `_install_studio_host_controls`. None of these resolve in the current
> codebase.
>
> For the actual Studio API:
>
> - **[Studio GUI API](STUDIO_GUI_API.md)** — public protocols and types (canonical reference)
> - **Code:** `swe2d/workbench/workbench_dialog_builder.py:248` (`WorkbenchDialogBuilder._build_component`), `swe2d/workbench/views/studio_component_view.py:29` (`StudioComponent`)
> - **Real feature toggles:** `SWE2DWorkbenchStudioDialog._studio_set_feature_enabled` (`studio_dialog.py:1617`) and `_studio_feature_keywords` (`studio_dialog.py:1636`)
> - **Canvas overlay:** `swe2d/results/high_perf_viewer.py` (`SWE2DHighPerfCanvasOverlayItem`) is real and is the only piece of the original guide that still applies
>
> This stub is preserved so existing links from `docs/INDEX.md` and the
> changelog stay resolvable. Refer to `STUDIO_GUI_API.md` and the source.

---

## Related Documentation

- **[Documentation Index](INDEX.md)** — All guides by audience
- **[Studio GUI API](STUDIO_GUI_API.md)** — Public protocols and types (canonical)
- **[Developer Guide](DEVELOPER_GUIDE.md)** — Architecture, MVP layers
- **[User Guide](USER_GUIDE.md)** — Studio UI walkthrough
- **[Repository Knowledge Graph](../graphify-out/wiki/index.md)** — Workbench module connections
