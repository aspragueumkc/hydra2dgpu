from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, Mapping, Optional

import numpy as np

try:
    from qgis.PyQt import QtWidgets
except Exception:
    from PyQt5 import QtWidgets  # type: ignore


_EXPERIMENTAL_3D_MODE_WIDGET_ATTRS = [
    "experimental_3d_coupling_mode_combo",
    "experimental_3d_patch_face_len_x_spin",
    "experimental_3d_patch_face_len_y_spin",
    "experimental_3d_patch_face_len_z_spin",
    "experimental_3d_projection_residual_sample_iters_spin",
    "experimental_3d_projection_divergence_gate_enable_chk",
    "experimental_3d_projection_divergence_ratio_target_spin",
    "experimental_3d_patch_xmin_edit",
    "experimental_3d_patch_xmax_edit",
    "experimental_3d_patch_ymin_edit",
    "experimental_3d_patch_ymax_edit",
    "experimental_3d_patch_zmin_edit",
    "experimental_3d_patch_zmax_edit",
    "experimental_3d_patch_set_roi_btn",
    "experimental_3d_patch_hint_lbl",
    "experimental_3d_patch_bc_widget",
    "experimental_3d_patch_bc_hint_lbl",
    "experimental_3d_obj_solids_chk",
    "experimental_3d_obj_method_combo",
    "experimental_3d_geom_sanitize_chk",
    "experimental_3d_geom_phi_snap_spin",
    "experimental_3d_geom_area_snap_spin",
    "experimental_3d_obj_layer_combo",
    "experimental_3d_obj_path_field_edit",
    "experimental_3d_obj_default_path_edit",
    "experimental_3d_obj_scale_field_edit",
    "experimental_3d_obj_yaw_field_edit",
    "experimental_3d_obj_z_offset_field_edit",
    "experimental_3d_obj_inside_points_layer_combo",
    "experimental_3d_obj_instance_id_field_edit",
    "experimental_3d_obj_inside_id_field_edit",
    "experimental_3d_obj_inside_z_field_edit",
    "experimental_3d_obj_use_terrain_chk",
    "experimental_3d_obj_ab_compare_chk",
    "experimental_3d_obj_ab_probe_steps_spin",
    "experimental_3d_obj_export_obj_chk",
    "experimental_3d_obj_export_obj_path_edit",
]


def collect_3d_patch_face_bc_env_overrides(
    *,
    ui: object,
    faces: Iterable[str],
    field_defaults: Mapping[str, float],
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for face in faces:
        face_key = str(face).lower()
        prefix = f"BACKWATER_SWE3D_BC_{face}"

        mode_val = 0
        mode_combo = getattr(ui, f"experimental_3d_bc_{face_key}_mode_combo", None)
        if isinstance(mode_combo, QtWidgets.QComboBox):
            try:
                raw_mode = mode_combo.currentData()
                if raw_mode is None:
                    raw_mode = mode_combo.currentIndex()
                mode_val = int(raw_mode)
            except Exception:
                mode_val = 0
        mode_val = max(0, min(4, int(mode_val)))
        out[f"{prefix}_MODE"] = f"{mode_val}"

        for field_name, default_value in field_defaults.items():
            value = float(default_value)
            spin = getattr(ui, f"experimental_3d_bc_{face_key}_{field_name}_spin", None)
            if isinstance(spin, QtWidgets.QDoubleSpinBox):
                try:
                    value = float(spin.value())
                except Exception:
                    value = float(default_value)
            if field_name == "vof":
                value = max(0.0, min(1.0, value))
            out[f"{prefix}_{field_name.upper()}"] = f"{value:.17g}"

    return out


def summarize_3d_patch_face_bc_modes(
    *,
    overrides: Mapping[str, str],
    faces: Iterable[str],
    mode_label_callback: Callable[[int], str],
) -> str:
    parts = []
    for face in faces:
        key = f"BACKWATER_SWE3D_BC_{face}_MODE"
        try:
            mode_val = int(float(str(overrides.get(key, "0"))))
        except Exception:
            mode_val = 0
        parts.append(f"{face}={mode_label_callback(mode_val)}")
    return ", ".join(parts)


def apply_3d_patch_face_bc_to_backend(
    *,
    ui: object,
    backend: object,
    faces: Iterable[str],
    field_defaults: Mapping[str, float],
    coupling_mode_off: int,
    get_coupling_mode_callback: Callable[[], int],
    log_callback: Callable[[str], None],
    quiet: bool = False,
) -> None:
    if backend is None or not hasattr(backend, "supports_3d_patch_face_bc_upload"):
        return
    try:
        if not bool(backend.supports_3d_patch_face_bc_upload()):
            return
    except Exception:
        return

    applied_summary = []
    has_nonzero_inflow = False

    for face_idx, face in enumerate(faces):
        face_key = str(face).lower()

        mode_val = 0
        mode_combo = getattr(ui, f"experimental_3d_bc_{face_key}_mode_combo", None)
        if isinstance(mode_combo, QtWidgets.QComboBox):
            try:
                raw_mode = mode_combo.currentData()
                if raw_mode is None:
                    raw_mode = mode_combo.currentIndex()
                mode_val = int(raw_mode)
            except Exception:
                mode_val = 0
        mode_val = max(0, min(4, int(mode_val)))

        values: Dict[str, float] = {}
        for field_name, default_value in field_defaults.items():
            value = float(default_value)
            spin = getattr(ui, f"experimental_3d_bc_{face_key}_{field_name}_spin", None)
            if isinstance(spin, QtWidgets.QDoubleSpinBox):
                try:
                    value = float(spin.value())
                except Exception:
                    value = float(default_value)
            if field_name == "vof":
                value = max(0.0, min(1.0, value))
            values[field_name] = value

        backend.set_3d_patch_face_bc(
            face=int(face_idx),
            mode=int(mode_val),
            u=float(values.get("u", 0.0)),
            v=float(values.get("v", 0.0)),
            w=float(values.get("w", 0.0)),
            q=float(values.get("q", 0.0)),
            vof=float(values.get("vof", 1.0)),
            p=float(values.get("p", 0.0)),
        )

        applied_summary.append(
            f"{face}:mode={mode_val},u={values.get('u', 0.0):.4g},v={values.get('v', 0.0):.4g},"
            f"w={values.get('w', 0.0):.4g},q={values.get('q', 0.0):.4g},"
            f"vof={values.get('vof', 1.0):.4g},p={values.get('p', 0.0):.4g}"
        )

        if int(mode_val) == 1:
            if (
                abs(float(values.get("u", 0.0))) > 1.0e-12
                or abs(float(values.get("v", 0.0))) > 1.0e-12
                or abs(float(values.get("w", 0.0))) > 1.0e-12
            ):
                has_nonzero_inflow = True
        elif int(mode_val) == 4:
            if abs(float(values.get("q", 0.0))) > 1.0e-12:
                has_nonzero_inflow = True

    if applied_summary and not bool(quiet):
        log_callback("3D face BC upload (GUI -> backend): " + " | ".join(applied_summary))

    try:
        coupling_mode = int(get_coupling_mode_callback())
    except Exception:
        coupling_mode = coupling_mode_off

    if not bool(quiet) and coupling_mode == coupling_mode_off and not has_nonzero_inflow:
        log_callback(
            "3D BC warning: uncoupled mode has no non-zero inflow BC forcing from GUI. "
            "Set at least one face to Inflow(U/V/W) with non-zero velocity or Volumetric Inlet(Q) with non-zero Q."
        )


def collect_3d_patch_env_overrides(
    *,
    ui: object,
    mesh_data: Mapping[str, np.ndarray],
    target_len_x: float,
    target_len_y: float,
    target_len_z: float,
    edit_optional_float_callback: Callable[[object], Optional[float]],
    sample_terrain_min_z_for_roi_callback: Callable[..., Optional[float]],
    collect_patch_env_overrides_callback: Callable[..., tuple[Dict[str, str], Dict[str, object]]],
    bed_manning_n: float,
    log_callback: Callable[[str], None],
    set_patch_zmin_text_callback: Callable[[str], None],
    collect_face_bc_env_overrides_callback: Callable[[], Dict[str, str]],
) -> Dict[str, str]:
    node_x = np.asarray(mesh_data.get("node_x", np.empty(0)), dtype=np.float64).ravel()
    node_y = np.asarray(mesh_data.get("node_y", np.empty(0)), dtype=np.float64).ravel()
    if node_x.size <= 0 or node_y.size <= 0:
        raise RuntimeError("Mesh node coordinates are missing for 3D patch setup.")

    xmin = edit_optional_float_callback(getattr(ui, "experimental_3d_patch_xmin_edit", None))
    xmax = edit_optional_float_callback(getattr(ui, "experimental_3d_patch_xmax_edit", None))
    ymin = edit_optional_float_callback(getattr(ui, "experimental_3d_patch_ymin_edit", None))
    ymax = edit_optional_float_callback(getattr(ui, "experimental_3d_patch_ymax_edit", None))
    zmin = edit_optional_float_callback(getattr(ui, "experimental_3d_patch_zmin_edit", None))
    zmax = edit_optional_float_callback(getattr(ui, "experimental_3d_patch_zmax_edit", None))

    xmin_probe = float(np.min(node_x)) if xmin is None else float(xmin)
    xmax_probe = float(np.max(node_x)) if xmax is None else float(xmax)
    ymin_probe = float(np.min(node_y)) if ymin is None else float(ymin)
    ymax_probe = float(np.max(node_y)) if ymax is None else float(ymax)
    span_x_probe = max(xmax_probe - xmin_probe, 1.0e-9)
    span_y_probe = max(ymax_probe - ymin_probe, 1.0e-9)
    nx_probe = max(2, int(math.ceil(span_x_probe / max(float(target_len_x), 1.0e-6))))
    ny_probe = max(2, int(math.ceil(span_y_probe / max(float(target_len_y), 1.0e-6))))

    terrain_zmin = sample_terrain_min_z_for_roi_callback(
        xmin=xmin_probe,
        xmax=xmax_probe,
        ymin=ymin_probe,
        ymax=ymax_probe,
        nx_hint=nx_probe,
        ny_hint=ny_probe,
    )

    overrides, meta = collect_patch_env_overrides_callback(
        mesh_data=dict(mesh_data),
        target_len_x=max(float(target_len_x), 1.0e-6),
        target_len_y=max(float(target_len_y), 1.0e-6),
        target_len_z=max(float(target_len_z), 1.0e-6),
        xmin_override=xmin,
        xmax_override=xmax,
        ymin_override=ymin,
        ymax_override=ymax,
        zmin_override=zmin,
        zmax_override=zmax,
        terrain_zmin=terrain_zmin,
        bed_manning_n=float(bed_manning_n),
    )

    sample_iters = 1
    sample_spin = getattr(ui, "experimental_3d_projection_residual_sample_iters_spin", None)
    if isinstance(sample_spin, QtWidgets.QSpinBox):
        try:
            sample_iters = int(sample_spin.value())
        except Exception:
            sample_iters = 1
    sample_iters = max(1, min(1024, sample_iters))
    overrides["BACKWATER_SWE3D_PROJECTION_RESIDUAL_SAMPLE_ITERS"] = str(sample_iters)

    divergence_gate_enabled = False
    divergence_gate_chk = getattr(ui, "experimental_3d_projection_divergence_gate_enable_chk", None)
    if isinstance(divergence_gate_chk, QtWidgets.QCheckBox):
        try:
            divergence_gate_enabled = bool(divergence_gate_chk.isChecked())
        except Exception:
            divergence_gate_enabled = False
    overrides["BACKWATER_SWE3D_PROJECTION_DIVERGENCE_GATE_ENABLE"] = "1" if divergence_gate_enabled else "0"

    divergence_ratio_target = 1.0
    divergence_ratio_spin = getattr(ui, "experimental_3d_projection_divergence_ratio_target_spin", None)
    if isinstance(divergence_ratio_spin, QtWidgets.QDoubleSpinBox):
        try:
            divergence_ratio_target = float(divergence_ratio_spin.value())
        except Exception:
            divergence_ratio_target = 1.0
    divergence_ratio_target = max(1.0e-6, min(100.0, divergence_ratio_target))
    overrides["BACKWATER_SWE3D_PROJECTION_DIVERGENCE_RATIO_TARGET"] = f"{divergence_ratio_target:.17g}"

    if bool(meta.get("terrain_zmin_used", False)):
        zmin_val = float(meta.get("zmin", 0.0))
        zmin_ui = float(meta.get("zmin_ui", zmin_val))
        if not np.isclose(zmin_val, zmin_ui):
            log_callback(
                "3D patch z-min override: using terrain DEM minimum "
                f"{zmin_val:.6g} instead of UI value {zmin_ui:.6g}."
            )
        try:
            set_patch_zmin_text_callback(f"{zmin_val:.9g}")
        except Exception:
            pass

    overrides.update(collect_face_bc_env_overrides_callback())
    return overrides


def sync_experimental_3d_mode_widgets(*, ui: object, active: bool) -> None:
    for attr_name in _EXPERIMENTAL_3D_MODE_WIDGET_ATTRS:
        widget = getattr(ui, attr_name, None)
        if widget is None:
            continue
        try:
            widget.setEnabled(bool(active))
        except Exception:
            pass

    run_btn = getattr(ui, "run_btn", None)
    if run_btn is not None:
        try:
            run_btn.setText("Run 3D Patch Model" if active else "Run 2D Model")
        except Exception:
            pass
