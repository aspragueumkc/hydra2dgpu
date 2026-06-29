"""Topology meshing controller.

MVP domain controller.  Shell — methods extracted from ``WorkbenchController``
as they are rewritten to use ``TopologyMeshView`` protocol.
"""

from __future__ import annotations

import os
import pickle
import subprocess
import tempfile
import time

import numpy as np

from swe2d.workbench.controllers.protocols_controller import TopologyMeshView


def _opt_float(widget, attr: str, default: float) -> float:
    """Safely read a float from a widget attribute (e.g. 'value' or 'text')."""
    try:
        val = getattr(widget, attr)
        if callable(val):
            val = val()
        return float(val)
    except (TypeError, ValueError, AttributeError):
        return float(default)


def _opt_bool(widget, attr: str, default: bool) -> bool:
    """Safely read a bool from a widget attribute (e.g. 'isChecked')."""
    try:
        val = getattr(widget, attr)
        if callable(val):
            val = val()
        return bool(val)
    except (TypeError, ValueError, AttributeError):
        return bool(default)


class TopologyController:
    """MVP controller for topology-layer management and mesh generation."""

    def __init__(self, view: TopologyMeshView):
        self._view = view

    # ── Topology status and quality controls ──────────────────────────

    def _refresh_topology_status(self) -> None:
        """Refresh the topology status label from current topology state.

        Reads the topology tab view and updates the status display.
        """
        view = self._view
        topo = view._topology_tab_view
        if topo is None:
            return
        view.update_topo_status("Topology status refreshed.")
        topo.update_control_summary()

    def _populate_gmsh_quality_controls(self) -> None:
        """Populate gmsh/quality detail widgets inside the placeholder containers.

        Delegates to the topology tab view's build and wire routines.
        """
        view = self._view
        topo = view._topology_tab_view
        if topo is None:
            return
        topo._populate_gmsh_quality_controls()

    def create_topology_template_layers(self) -> None:
        """Create all 14 topology template layers and add them to the QGIS project."""
        from qgis.core import QgsProject, QgsVectorLayer
        from swe2d.workbench.services.topology_template_service import (
            create_topology_template_layers,
        )

        view = self._view
        try:
            crs = QgsProject.instance().crs()
            crs_auth = crs.authid() if crs is not None and crs.isValid() else "EPSG:4326"
        except Exception:
            crs_auth = "EPSG:4326"

        try:
            import os as _os

            _qml_dir = _os.path.join(
                _os.path.dirname(_os.path.dirname(_os.path.dirname(__file__))),
                "QML",
            )
            layer_list = create_topology_template_layers(crs_auth=crs_auth)
            for name, lyr in layer_list:
                if lyr is not None and lyr.isValid():
                    QgsProject.instance().addMapLayer(lyr)
                    if isinstance(lyr, QgsVectorLayer):
                        # Load editor-widget config from QML (ValueMaps, constraints, etc.)
                        _qml_path = _os.path.join(
                            _qml_dir, f"{lyr.name().lower()}.qml"
                        )
                        if _os.path.exists(_qml_path):
                            lyr.loadNamedStyle(_qml_path)

            self.refresh_layer_combos()
            topo = view._topology_tab_view
            topo.view.update_topo_status(
                "Topology template layers created. Define regions (required); "
                "use interior rings or cell_type='empty' zones for holes; "
                "add optional arcs/constraints and optional quad-edge control lines; "
                "then generate mesh."
            )
            view._log(
                "Created topology template layers: SWE2D_Topo_Nodes/Arcs/Regions/"
                "Constraints/Quad_Edges + SWE2D_Manning_Zones + SWE2D_BC_Lines + "
                "SWE2D_Sample_Lines + SWE2D_Drainage_* + SWE2D_Structures + "
                "SWE2D_Hydrographs"
            )
        except Exception as e:
            view._log(f"[ERROR] Failed to create topology template layers: {e}")


    # ── Topology mesh helpers (moved from extracted/topology_and_io_methods) ──
    def cleanup_topology_mesh_checkpoint(
        self,
        topology_mesh_checkpoint_path,
        topology_mesh_progress_path,
        topology_mesh_progress_last_seq,
        topology_mesh_progress_last_sig,
        topology_mesh_progress,
    ):
        """Clean up checkpoint and progress files from a topology mesh operation.

        Returns the tuple of cleaned-up values for state update.
        Formerly ``_cleanup_topology_mesh_checkpoint`` in extracted module.
        """
        log_fn = self._view._log
        cp_path = str(topology_mesh_checkpoint_path or "").strip()
        if cp_path:
            try:
                os.remove(cp_path)
            except FileNotFoundError:
                pass
            except Exception as e:
                log_fn(f"[ERROR] checkpoint cleanup remove: {e}")
        topology_mesh_checkpoint_path = ""

        progress_path = str(topology_mesh_progress_path or "").strip()
        if progress_path:
            try:
                os.remove(progress_path)
            except FileNotFoundError:
                pass
            except Exception as e:
                log_fn(f"[ERROR] progress cleanup remove: {e}")
        topology_mesh_progress_path = ""
        topology_mesh_progress_last_seq = -1
        topology_mesh_progress_last_sig = ""
        topology_mesh_progress = None
        return (topology_mesh_checkpoint_path, topology_mesh_progress_path,
                topology_mesh_progress_last_seq, topology_mesh_progress_last_sig,
                topology_mesh_progress)


    def recover_topology_mesh_checkpoint(
        self,
        topology_mesh_checkpoint_path,
        mesh_data,
        result_data,
        backend_name: str,
        run_mode: str,
        elapsed: float,
    ):
        """Recover mesh from a checkpoint file after a timeout.

        Returns ``(recovered, mesh_data, result_data)``.
        Formerly ``_recover_topology_mesh_checkpoint`` in extracted module.
        """
        view = self._view
        log_fn = view._log
        if str(backend_name).strip().lower() != "gmsh":
            return False, mesh_data, result_data

        cp_path = str(topology_mesh_checkpoint_path or "").strip()
        if not cp_path or not os.path.exists(cp_path):
            return False, mesh_data, result_data

        try:
            with np.load(cp_path, allow_pickle=False) as cp:
                node_x = np.asarray(cp["node_x"], dtype=np.float64)
                node_y = np.asarray(cp["node_y"], dtype=np.float64)
                node_z = np.asarray(cp["node_z"], dtype=np.float64)
                cell_nodes = np.asarray(cp["cell_nodes"], dtype=np.int32)
                cell_face_offsets = np.asarray(cp["cell_face_offsets"], dtype=np.int32)
                cell_face_nodes = np.asarray(cp["cell_face_nodes"], dtype=np.int32)
                cell_type = np.asarray(cp["cell_type"]).astype(object)
                region_id = np.asarray(cp["region_id"], dtype=np.int32)
                target_size = np.asarray(cp["target_size"], dtype=np.float64)
                quality_summary = None
                if "quality_summary_json" in cp.files:
                    try:
                        import json as _json
                        raw = str(np.asarray(cp["quality_summary_json"]).item())
                        quality_summary = _json.loads(raw) if raw else None
                    except Exception as e:
                        log_fn(f"[ERROR] checkpoint quality_summary json parse: {e}")
                        quality_summary = None
        except Exception as exc:
            log_fn(f"mesh> checkpoint-read-fail path={cp_path} error={exc}")
            return False, mesh_data, result_data

        n_nodes = int(node_x.size)
        n_faces = max(0, int(cell_face_offsets.size) - 1)
        if n_nodes <= 0 or n_faces <= 0:
            return False, mesh_data, result_data

        mesh_data = {
            "nx": np.array(max(2, int(round(np.sqrt(node_x.size))))),
            "ny": np.array(max(2, int(round(np.sqrt(node_x.size))))),
            "lx": np.array(max(float(np.max(node_x) - np.min(node_x)), 1.0)),
            "ly": np.array(max(float(np.max(node_y) - np.min(node_y)), 1.0)),
            "node_x": node_x,
            "node_y": node_y,
            "node_z": node_z,
            "cell_nodes": cell_nodes,
            "cell_face_offsets": cell_face_offsets,
            "cell_face_nodes": cell_face_nodes,
            "cell_type": cell_type,
            "region_id": region_id,
            "target_size": target_size,
        }
        if isinstance(quality_summary, dict):
            mesh_data["quality_summary"] = dict(quality_summary)
        view._reset_runtime_snapshot_overlay_cache("topology mesh checkpoint recovered")

        n_tris = int(cell_nodes.size // 3)
        mtv = getattr(view, "_mesh_tab_view", None)
        if mtv is not None and mtv.mesh_info_lbl is not None:
            view.set_layer_status_text(
                f"Topology mesh: nodes={node_x.size}, faces={n_faces}, plot_triangles={n_tris}"
            )
        view.update_topo_status(
            f"Recovered {n_faces} computational faces from latest Gmsh attempt after timeout "
            f"(elapsed={elapsed:.2f}s, backend='{backend_name}')."
        )
        log_fn(
            "mesh> recovered-checkpoint "
            f"backend={backend_name} mode={run_mode} nodes={node_x.size} faces={n_faces} elapsed={elapsed:.2f}s"
        )
        if isinstance(quality_summary, dict):
            best_stats = quality_summary.get("best_stats", {})
            try:
                log_fn(
                    "mesh> gmsh-quality-summary "
                    f"attempts={int(quality_summary.get('attempts', 0))} "
                    f"strict={bool(quality_summary.get('strict_requested', False))} "
                    f"passed={bool(quality_summary.get('had_passing_candidate', False))} "
                    f"fail_cells(any/angle/aspect/area/non_orth)="
                    f"{int(float(best_stats.get('failed_any_cells', 0.0)))}/"
                    f"{int(float(best_stats.get('failed_min_angle_cells', 0.0)))}/"
                    f"{int(float(best_stats.get('failed_max_aspect_cells', 0.0)))}/"
                    f"{int(float(best_stats.get('failed_min_area_cells', 0.0)))}/"
                    f"{int(float(best_stats.get('failed_max_non_orth_cells', 0.0)))}"
                )
            except Exception as e:
                log_fn(f"[ERROR] checkpoint quality summary log: {e}")

        result_data = None
        try:
            viewer = getattr(view, "_studio_viewer", None)
            if viewer is not None:
                viewer.tab_widget.setCurrentWidget(
                    viewer.plot_widgets.get("Mesh"))
        except RuntimeError:
            pass
        view._refresh_plot()
        return True, mesh_data, result_data


    def poll_mesh_progress(
        self,
        topology_mesh_backend,
        topology_mesh_progress_path,
        topology_mesh_progress_last_seq,
        topology_mesh_progress_last_sig,
        topology_mesh_progress,
    ):
        """Poll progress file for a topology mesh operation.

        Returns ``(progress_path, last_seq, last_sig, progress_dict)``.
        """
        log_fn = self._view._log
        backend_name = str(topology_mesh_backend or "").strip().lower()
        if backend_name not in {"gmsh"}:
            return (topology_mesh_progress_path, topology_mesh_progress_last_seq,
                    topology_mesh_progress_last_sig, topology_mesh_progress)

        progress_path = str(topology_mesh_progress_path or "").strip()
        if not progress_path or not os.path.exists(progress_path):
            return (topology_mesh_progress_path, topology_mesh_progress_last_seq,
                    topology_mesh_progress_last_sig, topology_mesh_progress)

        try:
            import json as _json
            with open(progress_path, "r", encoding="utf-8") as fh:
                payload = _json.load(fh)
        except Exception as e:
            log_fn(f"[ERROR] progress json read: {e}")
            return (topology_mesh_progress_path, topology_mesh_progress_last_seq,
                    topology_mesh_progress_last_sig, topology_mesh_progress)

        if not isinstance(payload, dict):
            return (topology_mesh_progress_path, topology_mesh_progress_last_seq,
                    topology_mesh_progress_last_sig, topology_mesh_progress)

        try:
            seq = int(payload.get("seq", -1))
        except Exception as e:
            log_fn(f"[ERROR] progress seq parse: {e}")
            seq = -1

        last_seq_local = int(topology_mesh_progress_last_seq)
        payload_sig = (
            f"{payload.get('stage', '')}|{payload.get('region_id', '')}|"
            f"{payload.get('attempt', '')}|{payload.get('detail', '')}|{seq}"
        )
        last_sig = str(topology_mesh_progress_last_sig or "")
        if (seq >= 0 and seq == last_seq_local) or (payload_sig == last_sig):
            return (topology_mesh_progress_path, topology_mesh_progress_last_seq,
                    topology_mesh_progress_last_sig, topology_mesh_progress)

        topology_mesh_progress_last_seq = seq
        topology_mesh_progress_last_sig = payload_sig
        topology_mesh_progress = dict(payload)

        stage = str(payload.get("stage", "")).strip() or "update"
        detail = str(payload.get("detail", "")).strip()
        region_id = payload.get("region_id", None)
        attempt = payload.get("attempt", None)
        elapsed_s = payload.get("elapsed_s", None)

        parts = [f"stage={stage}"]
        if region_id is not None:
            parts.append(f"region={region_id}")
        if attempt is not None:
            parts.append(f"attempt={attempt}")
        if elapsed_s is not None:
            try:
                parts.append(f"elapsed={float(elapsed_s):.2f}s")
            except Exception as e:
                log_fn(f"[ERROR] progress elapsed_s format: {e}")
        if detail:
            parts.append(f"detail={detail}")
        log_fn(f"mesh> {backend_name}-progress " + " ".join(parts))
        return (topology_mesh_progress_path, topology_mesh_progress_last_seq,
                topology_mesh_progress_last_sig, topology_mesh_progress)


    def poll_topology_mesh_future(self, state: dict) -> dict:
        """Poll the topology mesh future, handling timeout, progress, and fallback.

        Reads all topology-mesh state from ``self._view`` attributes.
        Calls back to sibling controller methods for checkpoint/progress
        management. Formerly ``_poll_topology_mesh_future`` in extracted module.

        Parameters
        ----------
        state : dict
            The current topology mesh state dictionary (mutated and returned).

        Returns
        -------
        dict
            The updated state dict (or ``None`` if the timer should be kept).
        """
        view = self._view
        log_fn = view._log
        set_topology_mesh_busy_fn = view._set_topology_mesh_busy
        format_elapsed_fn = view._format_elapsed
        reset_cache_fn = view._reset_runtime_snapshot_overlay_cache
        refresh_plot_fn = view._refresh_plot
        start_topology_mesh_async_fn = view._start_topology_mesh_async
        topo_status_lbl = view.topo_status_lbl
        mtv = getattr(view, "_mesh_tab_view", None)
        mesh_info_lbl = mtv.mesh_info_lbl if mtv is not None else None
        _studio_viewer = getattr(view, "_studio_viewer", None)
        topology_mesh_timer = view._topology_mesh_timer
        topology_mesh_active_timeout_sec = view._topology_mesh_active_timeout_sec
        topology_mesh_backend = view._topology_mesh_backend
        topology_mesh_run_mode = view._topology_mesh_run_mode
        topology_mesh_default_cell_type = view._topology_mesh_default_cell_type
        topology_mesh_options = view._topology_mesh_options
        topology_mesh_conceptual = view._topology_mesh_conceptual

        fut = state.get("topology_mesh_future")
        if fut is None:
            topology_mesh_timer.stop()
            set_topology_mesh_busy_fn(False)
            return state

        elapsed = 0.0
        topology_mesh_started_at = state.get("topology_mesh_started_at")
        if topology_mesh_started_at is not None:
            elapsed = max(0.0, time.perf_counter() - topology_mesh_started_at)

        topology_mesh_poll_count = state.get("topology_mesh_poll_count", 0)
        topology_mesh_process_pool = state.get("topology_mesh_process_pool")
        topology_mesh_progress = state.get("topology_mesh_progress")
        mesh_data = state.get("mesh_data")
        result_data = state.get("result_data")
        topology_mesh_auto_fallback_used = state.get("topology_mesh_auto_fallback_used", False)
        topology_mesh_checkpoint_path = state.get("topology_mesh_checkpoint_path", "")
        topology_mesh_progress_path = state.get("topology_mesh_progress_path", "")
        topology_mesh_progress_last_seq = state.get("topology_mesh_progress_last_seq", -1)
        topology_mesh_progress_last_sig = state.get("topology_mesh_progress_last_sig", "")

        if elapsed > topology_mesh_active_timeout_sec and not fut.done():
            backend_name = topology_mesh_backend or "unknown"
            run_mode = topology_mesh_run_mode
            topology_mesh_timer.stop()
            fut = None
            topology_mesh_started_at = None
            topology_mesh_poll_count = 0

            if backend_name in {"gmsh"} and topology_mesh_process_pool is not None:
                try:
                    topology_mesh_process_pool.shutdown(wait=False, cancel_futures=True)
                except Exception as e:
                    log_fn(f"[ERROR] process pool shutdown: {e}")
                topology_mesh_process_pool = None

            recovered, mesh_data, result_data = self.recover_topology_mesh_checkpoint(
                topology_mesh_checkpoint_path=topology_mesh_checkpoint_path,
                mesh_data=mesh_data,
                result_data=result_data,
                backend_name=backend_name,
                run_mode=run_mode,
                elapsed=elapsed,
            )

            if not recovered:
                view.update_topo_status(
                    f"Topology meshing timed out after {topology_mesh_active_timeout_sec:.0f}s "
                    f"(backend '{backend_name}')."
                )
            log_fn(
                "mesh> timeout "
                f"backend={backend_name} mode={run_mode} elapsed={elapsed:.2f}s "
                f"limit={topology_mesh_active_timeout_sec:.0f}s"
            )

            if recovered:
                log_fn(
                    "mesh> timeout-recovery "
                    f"backend={backend_name} mode={run_mode} action=loaded_latest_attempt"
                )

            set_topology_mesh_busy_fn(False)
            (topology_mesh_checkpoint_path, topology_mesh_progress_path,
             topology_mesh_progress_last_seq, topology_mesh_progress_last_sig,
             topology_mesh_progress) = self.cleanup_topology_mesh_checkpoint(
                topology_mesh_checkpoint_path=topology_mesh_checkpoint_path,
                topology_mesh_progress_path=topology_mesh_progress_path,
                topology_mesh_progress_last_seq=topology_mesh_progress_last_seq,
                topology_mesh_progress_last_sig=topology_mesh_progress_last_sig,
                topology_mesh_progress=topology_mesh_progress,
            )
            state.update(
                mesh_data=mesh_data,
                result_data=result_data,
                topology_mesh_future=fut,
                topology_mesh_started_at=topology_mesh_started_at,
                topology_mesh_poll_count=topology_mesh_poll_count,
                topology_mesh_process_pool=topology_mesh_process_pool,
                topology_mesh_progress=topology_mesh_progress,
                topology_mesh_checkpoint_path=topology_mesh_checkpoint_path,
                topology_mesh_progress_path=topology_mesh_progress_path,
                topology_mesh_progress_last_seq=topology_mesh_progress_last_seq,
                topology_mesh_progress_last_sig=topology_mesh_progress_last_sig,
            )
            return state

        if not fut.done():
            topology_mesh_poll_count += 1
            (topology_mesh_progress_path, topology_mesh_progress_last_seq,
             topology_mesh_progress_last_sig, topology_mesh_progress) = self.poll_mesh_progress(
                topology_mesh_backend=topology_mesh_backend,
                topology_mesh_progress_path=topology_mesh_progress_path,
                topology_mesh_progress_last_seq=topology_mesh_progress_last_seq,
                topology_mesh_progress_last_sig=topology_mesh_progress_last_sig,
                topology_mesh_progress=topology_mesh_progress,
            )
            if topology_mesh_poll_count % 8 == 0:
                backend_running = str(topology_mesh_backend or "unknown").strip().lower()
                spinner = "|/-\\"[(topology_mesh_poll_count // 8) % 4]
                if backend_running == "gmsh":
                    try:
                        status_txt = str(topo_status_lbl.text() or "").strip()
                    except Exception as e:
                        log_fn(f"[ERROR] gmsh status text read: {e}")
                        status_txt = ""
                    elapsed_s = 0.0
                    if topology_mesh_started_at is not None:
                        elapsed_s = max(0.0, time.perf_counter() - topology_mesh_started_at)
                    topology_mesh_progress = {
                        "backend": "gmsh",
                        "stage": "running",
                        "spinner": str(spinner),
                        "elapsed_s": float(elapsed_s),
                        "detail": str(status_txt),
                    }
                    parts = [
                        "stage=running",
                        f"status={spinner}",
                        f"elapsed={format_elapsed_fn(topology_mesh_started_at)}",
                    ]
                    if status_txt:
                        parts.append(f"detail={status_txt}")
                    log_fn("mesh> gmsh-progress " + " ".join(parts))
                else:
                    log_fn(
                        "mesh> run "
                        f"status={spinner} backend={topology_mesh_backend or 'unknown'} "
                        f"elapsed={format_elapsed_fn(topology_mesh_started_at)}"
                    )
            state.update(
                topology_mesh_poll_count=topology_mesh_poll_count,
                topology_mesh_progress=topology_mesh_progress,
                topology_mesh_progress_path=topology_mesh_progress_path,
                topology_mesh_progress_last_seq=topology_mesh_progress_last_seq,
                topology_mesh_progress_last_sig=topology_mesh_progress_last_sig,
            )
            return state

        topology_mesh_timer.stop()
        backend_name = topology_mesh_backend or "unknown"
        default_cell_type = topology_mesh_default_cell_type or "triangular"
        run_mode = topology_mesh_run_mode
        elapsed_str = format_elapsed_fn(topology_mesh_started_at)
        fut_completed = fut
        fut = None
        topology_mesh_started_at = None
        topology_mesh_poll_count = 0
        topology_mesh_progress = None
        fallback_restarted = False

        try:
            mesh = fut_completed.result()
            n_nodes = int(np.asarray(mesh.node_x).size)
            n_faces = max(0, int(np.asarray(mesh.cell_face_offsets).size) - 1)
            if n_nodes <= 0 or n_faces <= 0:
                raise RuntimeError(
                    f"Topology backend '{backend_name}' produced an empty mesh "
                    f"(nodes={n_nodes}, faces={n_faces})."
                )
            mesh_data = {
                "nx": np.array(max(2, int(round(np.sqrt(mesh.node_x.size))))),
                "ny": np.array(max(2, int(round(np.sqrt(mesh.node_x.size))))),
                "lx": np.array(max(float(np.max(mesh.node_x) - np.min(mesh.node_x)), 1.0)),
                "ly": np.array(max(float(np.max(mesh.node_y) - np.min(mesh.node_y)), 1.0)),
                "node_x": mesh.node_x,
                "node_y": mesh.node_y,
                "node_z": mesh.node_z,
                "cell_nodes": mesh.cell_nodes,
                "cell_face_offsets": mesh.cell_face_offsets,
                "cell_face_nodes": mesh.cell_face_nodes,
                "cell_type": mesh.cell_type,
                "region_id": mesh.region_id,
                "target_size": mesh.target_size,
            }
            quality_summary = getattr(mesh, "quality_summary", None)
            if isinstance(quality_summary, dict):
                mesh_data["quality_summary"] = dict(quality_summary)
            reset_cache_fn("topology mesh regenerated")
            n_faces = int(mesh.cell_face_offsets.size - 1)
            n_tris = int(mesh.cell_nodes.size // 3)
            if mesh_info_lbl is not None:
                view.set_layer_status_text(
                    f"Topology mesh: nodes={mesh.node_x.size}, faces={n_faces}, plot_triangles={n_tris}"
                )
            if run_mode == "fallback-no-constraints":
                view.update_topo_status(
                    f"Generated {n_faces} computational faces using backend '{backend_name}' "
                    "after automatic fallback with constraints disabled. "
                    "Review/repair constraint polygons and regenerate when ready."
                )
            else:
                view.update_topo_status(
                    f"Generated {n_faces} computational faces using backend '{backend_name}'. "
                    "Cell metadata (type/size/region) stored in mesh state."
                )
            log_fn(
                "mesh> done "
                f"backend={backend_name} default_cell_type={default_cell_type} "
                f"mode={run_mode} "
                f"nodes={mesh.node_x.size} faces={n_faces} elapsed={elapsed_str}"
            )
            quality_summary = getattr(mesh, "quality_summary", None)
            if isinstance(quality_summary, dict):
                best_stats = quality_summary.get("best_stats", {})
                try:
                    log_fn(
                        "mesh> gmsh-quality-summary "
                        f"attempts={int(quality_summary.get('attempts', 0))} "
                        f"strict={bool(quality_summary.get('strict_requested', False))} "
                        f"passed={bool(quality_summary.get('had_passing_candidate', False))} "
                        f"fail_cells(any/angle/aspect/area/non_orth)="
                        f"{int(float(best_stats.get('failed_any_cells', 0.0)))}/"
                        f"{int(float(best_stats.get('failed_min_angle_cells', 0.0)))}/"
                        f"{int(float(best_stats.get('failed_max_aspect_cells', 0.0)))}/"
                        f"{int(float(best_stats.get('failed_min_area_cells', 0.0)))}/"
                        f"{int(float(best_stats.get('failed_max_non_orth_cells', 0.0)))}"
                    )
                except Exception as e:
                    log_fn(f"[ERROR] quality summary log in poll: {e}")
            result_data = None
            try:
                if _studio_viewer is not None:
                    _studio_viewer.tab_widget.setCurrentWidget(
                        _studio_viewer.plot_widgets.get("Mesh"))
            except RuntimeError:
                pass
            refresh_plot_fn()
        except NotImplementedError as exc:
            view.update_topo_status(str(exc))
            log_fn(f"mesh> fail backend={backend_name} mode={run_mode} elapsed={elapsed_str} error={exc}")
        except RuntimeError as exc:
            err_txt = str(exc)
            err_l = err_txt.lower()
            empty_mesh_failure = ("empty mesh" in err_l) or ("non-empty mesh" in err_l)
            can_retry_without_constraints = (
                backend_name == "gmsh"
                and run_mode == "full"
                and not topology_mesh_auto_fallback_used
                and topology_mesh_conceptual is not None
                and bool(getattr(topology_mesh_conceptual, "constraints", []))
            )
            if empty_mesh_failure and can_retry_without_constraints:
                try:
                    from swe2d.mesh.meshing import (
                        _clone_conceptual_without_constraints,
                    )
                    fallback_conceptual = _clone_conceptual_without_constraints(topology_mesh_conceptual)
                    topology_mesh_auto_fallback_used = True
                    log_fn(
                        "mesh> fallback "
                        f"backend={backend_name} action=retry_without_constraints "
                        f"reason=empty-mesh elapsed={elapsed_str}"
                    )
                    start_topology_mesh_async_fn(
                        fallback_conceptual,
                        backend_name,
                        default_cell_type,
                        topology_mesh_options,
                        run_mode="fallback-no-constraints",
                    )
                    fallback_restarted = True
                    state.update(
                        mesh_data=mesh_data,
                        result_data=result_data,
                        topology_mesh_future=fut,
                        topology_mesh_started_at=topology_mesh_started_at,
                        topology_mesh_poll_count=topology_mesh_poll_count,
                        topology_mesh_progress=topology_mesh_progress,
                        topology_mesh_auto_fallback_used=topology_mesh_auto_fallback_used,
                        topology_mesh_process_pool=topology_mesh_process_pool,
                    )
                    (topology_mesh_checkpoint_path, topology_mesh_progress_path,
                     topology_mesh_progress_last_seq, topology_mesh_progress_last_sig,
                     topology_mesh_progress) = self.cleanup_topology_mesh_checkpoint(
                        topology_mesh_checkpoint_path=topology_mesh_checkpoint_path,
                        topology_mesh_progress_path=topology_mesh_progress_path,
                        topology_mesh_progress_last_seq=topology_mesh_progress_last_seq,
                        topology_mesh_progress_last_sig=topology_mesh_progress_last_sig,
                        topology_mesh_progress=topology_mesh_progress,
                    )
                    state.update(
                        topology_mesh_checkpoint_path=topology_mesh_checkpoint_path,
                        topology_mesh_progress_path=topology_mesh_progress_path,
                        topology_mesh_progress_last_seq=topology_mesh_progress_last_seq,
                        topology_mesh_progress_last_sig=topology_mesh_progress_last_sig,
                    )
                    return state
                except Exception as fallback_exc:
                    log_fn(
                        "mesh> fallback-fail "
                        f"backend={backend_name} elapsed={elapsed_str} error={fallback_exc}"
                    )
            view.update_topo_status(err_txt)
            log_fn(f"mesh> fail backend={backend_name} mode={run_mode} elapsed={elapsed_str} error={exc}")
        except Exception as exc:
            view.update_topo_status(f"Topology meshing failed: {exc}")
            log_fn(f"mesh> fail backend={backend_name} mode={run_mode} elapsed={elapsed_str} error={exc}")
        finally:
            if not fallback_restarted:
                set_topology_mesh_busy_fn(False)
            (topology_mesh_checkpoint_path, topology_mesh_progress_path,
             topology_mesh_progress_last_seq, topology_mesh_progress_last_sig,
             topology_mesh_progress) = self.cleanup_topology_mesh_checkpoint(
                topology_mesh_checkpoint_path=topology_mesh_checkpoint_path,
                topology_mesh_progress_path=topology_mesh_progress_path,
                topology_mesh_progress_last_seq=topology_mesh_progress_last_seq,
                topology_mesh_progress_last_sig=topology_mesh_progress_last_sig,
                topology_mesh_progress=topology_mesh_progress,
            )

        if not fallback_restarted:
            state.update(
                mesh_data=mesh_data,
                result_data=result_data,
                topology_mesh_future=fut,
                topology_mesh_started_at=topology_mesh_started_at,
                topology_mesh_poll_count=topology_mesh_poll_count,
                topology_mesh_progress=topology_mesh_progress,
                topology_mesh_auto_fallback_used=topology_mesh_auto_fallback_used,
                topology_mesh_process_pool=topology_mesh_process_pool,
                topology_mesh_checkpoint_path=topology_mesh_checkpoint_path,
                topology_mesh_progress_path=topology_mesh_progress_path,
                topology_mesh_progress_last_seq=topology_mesh_progress_last_seq,
                topology_mesh_progress_last_sig=topology_mesh_progress_last_sig,
            )

        return state

    # ── Model GeoPackage explorer ─────────────────────────────────────


    # ── Model GeoPackage explorer ─────────────────────────────────────
    def open_model_gpkg_explorer(self) -> None:
        """Open file dialog, then launch the model GPKG explorer for the chosen file."""
        import os as _os
        from qgis.PyQt import QtWidgets

        view = self._view
        db_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            view, "Select GeoPackage to explore", "",
            "GeoPackage (*.gpkg);;All Files (*)",
        )
        db_path = str(db_path or "").strip()
        if not db_path or not _os.path.exists(db_path):
            if db_path:
                view._log(f"[ERROR] GeoPackage not found: {db_path}")
            return

        try:
            from swe2d.workbench.dialogs.gpkg_explorer_dialog import (
                SWE2DModelGeoPackageExplorerDialog,
            )
            dlg = SWE2DModelGeoPackageExplorerDialog(
                gpkg_path=db_path,
                open_run_log_viewer=view._open_run_log_viewer,
                open_line_results_viewer=view._open_line_results_viewer,
                logger=view._log,
                parent=view,
            )
            dlg.exec()
        except ImportError:
            view._log("[ERROR] Model GeoPackage explorer dialog not available.")

    # ── Generate mesh from topology layers ────────────────────────────


    # ── Generate mesh from topology layers ────────────────────────────
    def generate_mesh_from_topology_layers(self) -> None:
        """Read topology layers from combos, build conceptual model, launch async meshing."""
        from qgis.PyQt import QtWidgets
        from swe2d.workbench.studio_dialog import _HAVE_QGIS_CORE

        view = self._view
        topo = view._topology_tab_view

        view._log("mesh> generate-from-layers started")
        if not _HAVE_QGIS_CORE:
            view._log("QGIS layer API unavailable; cannot read topology layers.")
            return

        nodes_layer = view._combo_layer(topo.topo_nodes_combo, "vector")
        arcs_layer = view._combo_layer(topo.topo_arcs_combo, "vector")
        regions_layer = view._combo_layer(topo.topo_regions_combo, "vector")
        constraints_layer = view._combo_layer(topo.topo_constraints_combo, "vector")
        quad_edges_layer = view._combo_layer(topo.topo_quad_edges_combo, "vector")

        if view.get_topo_combo_data("topo_nodes_combo") is None:
            nodes_layer = None
        if view.get_topo_combo_data("topo_arcs_combo") is None:
            arcs_layer = None
        if view.get_topo_combo_data("topo_constraints_combo") is None:
            constraints_layer = None
        if view.get_topo_combo_data("topo_quad_edges_combo") is None:
            quad_edges_layer = None

        if regions_layer is None:
            QtWidgets.QMessageBox.warning(
                view, "Topology Meshing",
                "Select a topology regions polygon layer first."
            )
            topo.view.update_topo_status("Missing required topology regions layer.")
            return

        try:
            from swe2d.mesh.meshing import conceptual_from_qgis_layers
        except ImportError:
            view._log("[ERROR] conceptual_from_qgis_layers not available")
            return

        default_size = float(view.get_topo_widget_value("topo_default_size_spin"))
        default_cell_type = str(view.get_topo_combo_data("topo_default_cell_type_combo"))
        backend_name = str(view.get_topo_combo_data("topo_backend_combo") or "gmsh")

        try:
            mesh_options = view._build_topology_meshing_options()
            if backend_name == "gmsh":
                workspace_root = view._infer_workspace_root_for_meshing()
                if workspace_root:
                    mesh_options["workspace_module_root"] = workspace_root
                    view._log(f"mesh> module-path workspace-root={workspace_root}")

            conceptual = conceptual_from_qgis_layers(
                nodes_layer=nodes_layer,
                arcs_layer=arcs_layer,
                regions_layer=regions_layer,
                constraints_layer=constraints_layer,
                quad_edges_layer=quad_edges_layer,
                default_size=default_size,
                default_cell_type=default_cell_type,
            )
            view._start_topology_mesh_async(
                conceptual, backend_name, default_cell_type, mesh_options,
            )
        except ValueError as exc:
            topo.view.update_topo_status(f"Invalid topology mesh options: {exc}")
            view._log(f"Topology mesh option error: {exc}")
        except NotImplementedError as exc:
            topo.view.update_topo_status(str(exc))
            view._log(f"Topology meshing backend not implemented: {exc}")
        except RuntimeError as exc:
            topo.view.update_topo_status(str(exc))
            view._log(f"Topology meshing runtime error: {exc}")
        except Exception as exc:
            topo.view.update_topo_status(f"Topology meshing failed: {exc}")
            view._log(f"Topology meshing error: {exc}")

    # ── Terminate topology mesh ───────────────────────────────────────


    def on_terminate_topology_mesh(self) -> None:
        """Kill the running topology mesh subprocess and reset busy state."""
        view = self._view
        timer = getattr(view, "_topology_mesh_timer", None)
        if timer is not None:
            try:
                timer.stop()
            except Exception as _e:

                try:

                    view._log(f"[ERROR] Exception in topology_controller.py: {_e}")

                except Exception:

                    pass
        proc = getattr(view, "_topology_mesh_subprocess", None)
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()
        for attr in ("_topology_mesh_subprocess", "_topology_mesh_started_at"):
            setattr(view, attr, None)
        view._set_topology_mesh_busy(False)
        view._log("mesh> terminated by user")

    def start_topology_mesh_async(
        self,
        conceptual,
        backend_name: str,
        default_cell_type: str,
        mesh_options: dict | None = None,
        run_mode: str = "full",
    ) -> None:
        """Start an async topology mesh job via subprocess worker."""
        import time as _time_mod
        from qgis.PyQt import QtCore

        view = self._view
        log_fn = view._log
        mesh_options = mesh_options or {}

        proc = getattr(view, "_topology_mesh_subprocess", None)
        if proc is not None and proc.poll() is None:
            view._log("[ERROR] A topology mesh job is already running.")
            return
        if proc is not None:
            view._log("mesh> cleaning up stale subprocess reference")
            view._topology_mesh_subprocess = None
            view._topology_mesh_backend = None

        try:
            view._topology_mesh_backend = backend_name
            view._topology_mesh_default_cell_type = default_cell_type
            view._topology_mesh_run_mode = run_mode
            view._topology_mesh_conceptual = conceptual
            view._topology_mesh_options = mesh_options
            view._topology_mesh_started_at = _time_mod.perf_counter()
            view._topology_mesh_poll_count = 0
            view._topology_mesh_progress_last_seq = -1

            combined_options = dict(mesh_options)
            combined_options["run_mode"] = run_mode
            combined_options["default_cell_type"] = default_cell_type

            view._topology_mesh_checkpoint_path = ""
            view._topology_mesh_progress_path = ""
            if backend_name == "gmsh":
                cp_dir = os.path.join("/tmp", "qgis-live-bridge")
                cp_name = f"topology_mesh_checkpoint_{os.getpid()}_{int(time.time() * 1000)}.npz"
                view._topology_mesh_checkpoint_path = os.path.join(cp_dir, cp_name)
                combined_options["gmsh_quality_checkpoint_path"] = view._topology_mesh_checkpoint_path
                progress_name = f"topology_gmsh_progress_{os.getpid()}_{int(time.time() * 1000)}.json"
                view._topology_mesh_progress_path = os.path.join(cp_dir, progress_name)
                combined_options["gmsh_progress_path"] = view._topology_mesh_progress_path
                combined_options.setdefault("gmsh_progress_emit_interval_s", 0.75)
                try:
                    os.makedirs(cp_dir, exist_ok=True)
                except OSError as e:
                    view._log(f"[ERROR] start topology mesh async failed: {e}")
                try:
                    os.remove(view._topology_mesh_checkpoint_path)
                except FileNotFoundError:
                    pass
                except OSError as e:
                    view._log(f"[ERROR] start topology mesh async failed: {e}")
                try:
                    os.remove(view._topology_mesh_progress_path)
                except FileNotFoundError:
                    pass
                except OSError as e:
                    view._log(f"[ERROR] start topology mesh async failed: {e}")

            payload = {
                "conceptual": conceptual,
                "backend_name": backend_name,
                "options": combined_options,
            }
            in_fd, in_path = tempfile.mkstemp(suffix="_mesh_in.pkl")
            os.close(in_fd)
            out_fd, out_path = tempfile.mkstemp(suffix="_mesh_out.pkl")
            os.close(out_fd)
            err_fd, err_path = tempfile.mkstemp(suffix="_mesh_err.txt")
            os.close(err_fd)
            with open(in_path, "wb") as f:
                pickle.dump(payload, f)
            view._topology_mesh_in_path = in_path
            view._topology_mesh_out_path = out_path
            view._topology_mesh_err_path = err_path

            worker = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                "tools", "gmsh_subprocess_worker.py",
            )
            import sys as _sys
            _repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            _pythonpath = os.pathsep.join([str(p) for p in [_repo_root, os.path.join(_repo_root, "build")] if p])
            env = dict(os.environ)
            _existing = env.get("PYTHONPATH", "")
            if _existing:
                _pythonpath = _existing + os.pathsep + _pythonpath
            env["PYTHONPATH"] = _pythonpath
            env.setdefault("DISPLAY", os.environ.get("DISPLAY", ":0"))
            with open(err_path, "wb") as err_file:
                view._topology_mesh_subprocess = subprocess.Popen(
                    [_sys.executable, worker, in_path, out_path],
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=err_file,
                )

            view._set_topology_mesh_busy(True, f"Topology meshing with '{backend_name}'...")
            self._ensure_timer()
            view._topology_mesh_timer.start(500)
            view._log(f"mesh> start backend={backend_name} mode={run_mode}")
        except Exception as exc:
            view._log(f"[ERROR] start topology mesh async: {exc}")
            view._set_topology_mesh_busy(False)

    def _ensure_timer(self):
        """Ensure the topology mesh poll timer exists and is wired to _poll_topology_mesh."""
        view = self._view
        if not hasattr(view, "_topology_mesh_timer") or view._topology_mesh_timer is None:
            from qgis.PyQt import QtCore
            view._topology_mesh_timer = QtCore.QTimer(view)
        try:
            view._topology_mesh_timer.timeout.disconnect()
        except Exception as _e:

            try:

                view._log(f"[ERROR] Exception in topology_controller.py: {_e}")

            except Exception:

                pass
        view._topology_mesh_timer.timeout.connect(lambda: self._poll_topology_mesh())


    def _poll_topology_mesh(self) -> None:
        """Poll the topology mesh subprocess status and handle completion/failure."""
        view = self._view
        proc = getattr(view, "_topology_mesh_subprocess", None)
        if proc is None:
            timer = getattr(view, "_topology_mesh_timer", None)
            if timer is not None:
                try:
                    timer.stop()
                except Exception as _e:

                    try:

                        view._log(f"[ERROR] Exception in topology_controller.py: {_e}")

                    except Exception:

                        pass
            return
        ret = proc.poll()
        if ret is None:
            # --- Stderr tail: read Gmsh logger output forwarded by the worker ---
            err_path = getattr(view, "_topology_mesh_err_path", None)
            if err_path and os.path.exists(err_path):
                try:
                    size = os.path.getsize(err_path)
                    if size > 0:
                        with open(err_path, "rb") as ef:
                            ef.seek(max(0, size - 512))
                            tail = ef.read().decode(errors="replace").strip()
                        if tail:
                            for ln in tail.splitlines()[-3:]:
                                view._log(f"  gmsh> {ln}")
                except Exception as _e:

                    try:

                        view._log(f"[ERROR] Exception in topology_controller.py: {_e}")

                    except Exception:

                        pass
            # --- Progress JSON poll (written by GmshBackend._emit_progress) ---
            self._poll_topology_mesh_progress(view)
            return
        timer = getattr(view, "_topology_mesh_timer", None)
        if timer is not None:
            try:
                timer.stop()
            except Exception as _e:

                try:

                    view._log(f"[ERROR] Exception in topology_controller.py: {_e}")

                except Exception:

                    pass
        out_path = getattr(view, "_topology_mesh_out_path", None)
        if ret != 0 or out_path is None or not os.path.exists(out_path):
            view._log(f"mesh> fail returncode={ret}")
            err_path = getattr(view, "_topology_mesh_err_path", None)
            if err_path and os.path.exists(err_path):
                try:
                    with open(err_path, "r", errors="replace") as ef:
                        for ln in ef.read().splitlines():
                            view._log(f"  stderr> {ln}")
                except Exception as _e:

                    try:

                        view._log(f"[ERROR] Exception in topology_controller.py: {_e}")

                    except Exception:

                        pass
            view._set_topology_mesh_busy(False)
            view.update_topo_status(f"Gmsh failed (code {ret})")
            self._cleanup_mesh_tempfiles(view)
            return
        try:
            with open(out_path, "rb") as f:
                result = pickle.load(f)
        except Exception as exc:
            view._log(f"mesh> fail result-read: {exc}")
            err_path = getattr(view, "_topology_mesh_err_path", None)
            if err_path and os.path.exists(err_path):
                try:
                    with open(err_path, "r", errors="replace") as ef:
                        for ln in ef.read().splitlines():
                            view._log(f"  stderr> {ln}")
                except Exception as _e:

                    try:

                        view._log(f"[ERROR] Exception in topology_controller.py: {_e}")

                    except Exception:

                        pass
            view._set_topology_mesh_busy(False)
            self._cleanup_mesh_tempfiles(view)
            return
        self._cleanup_mesh_tempfiles(view)
        if not result.get("ok"):
            err = result.get("error", "unknown")
            tb = result.get("traceback", "")
            view._log(f"mesh> fail error={err}")
            if tb:
                for ln in tb.rstrip().splitlines():
                    view._log(f"  {ln}")
            view._set_topology_mesh_busy(False)
            view.update_topo_status(f"Gmsh error: {err[:120]}")
            return
        mesh = result["mesh"]
        backend_name = getattr(view, "_topology_mesh_backend", "gmsh")
        log_fn = view._log
        elapsed = 0.0
        started = getattr(view, "_topology_mesh_started_at", None)
        if started is not None:
            elapsed = max(0.0, time.perf_counter() - started)
        mesh_data = {
            "node_x": np.asarray(mesh.node_x, dtype=np.float64),
            "node_y": np.asarray(mesh.node_y, dtype=np.float64),
            "node_z": np.asarray(mesh.node_z, dtype=np.float64),
            "cell_nodes": np.asarray(mesh.cell_nodes, dtype=np.int32),
            "cell_face_offsets": np.asarray(mesh.cell_face_offsets, dtype=np.int32),
            "cell_face_nodes": np.asarray(mesh.cell_face_nodes, dtype=np.int32),
            "cell_type": np.asarray(mesh.cell_type, dtype=object),
            "region_id": np.asarray(mesh.region_id, dtype=np.int32),
            "target_size": np.asarray(mesh.target_size, dtype=np.float64),
        }
        view._mesh_data = mesh_data
        log_fn(f"mesh> done backend={backend_name} elapsed={view._format_elapsed(started)}")
        n_nodes = int(mesh.node_x.size)
        n_faces = max(0, int(mesh.cell_face_offsets.size) - 1)
        mtv = getattr(view, "_mesh_tab_view", None)
        if mtv is not None:
            try:
                mtv.set_mesh_info_text(f"Nodes: {n_nodes}, Faces: {n_faces}")
            except Exception as _e:

                try:

                    view._log(f"[ERROR] Exception in topology_controller.py: {_e}")

                except Exception:

                    pass
        try:
            view._refresh_plot()
        except Exception as _e:

            try:

                view._log(f"[ERROR] Exception in topology_controller.py: {_e}")

            except Exception:

                pass
        view._set_topology_mesh_busy(False)
        view._topology_mesh_subprocess = None
        view._topology_mesh_started_at = None

    def _poll_topology_mesh_progress(self, view) -> None:
        """Poll the Gmsh progress JSON file (written by GmshBackend._emit_progress)."""
        progress_path = getattr(view, "_topology_mesh_progress_path", None)
        if not progress_path or not os.path.exists(progress_path):
            return
        try:
            import json as _json
            with open(progress_path, "r", encoding="utf-8") as fh:
                payload = _json.load(fh)
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        try:
            seq = int(payload.get("seq", -1))
        except Exception:
            seq = -1
        last_seq = getattr(view, "_topology_mesh_progress_last_seq", -1)
        if seq >= 0 and seq == last_seq:
            return
        view._topology_mesh_progress_last_seq = seq
        stage = str(payload.get("stage", "")).strip() or "update"
        detail = str(payload.get("detail", "")).strip()
        region_id = payload.get("region_id", None)
        attempt = payload.get("attempt", None)
        elapsed_s = payload.get("elapsed_s", None)
        parts = [f"stage={stage}"]
        if region_id is not None:
            parts.append(f"region={region_id}")
        if attempt is not None:
            parts.append(f"attempt={attempt}")
        if elapsed_s is not None:
            try:
                parts.append(f"elapsed={float(elapsed_s):.2f}s")
            except Exception as _e:

                try:

                    view._log(f"[ERROR] Exception in topology_controller.py: {_e}")

                except Exception:

                    pass
        if detail:
            parts.append(detail)
        view._log("mesh> gmsh-progress " + " ".join(parts))

    def _cleanup_mesh_tempfiles(self, view):
        """Remove temp files created for topology mesh subprocess communication."""
        for attr in ("_topology_mesh_in_path", "_topology_mesh_out_path", "_topology_mesh_err_path"):
            p = getattr(view, attr, None)
            if p is not None and os.path.exists(p):
                try:
                    os.unlink(p)
                except Exception as _e:

                    try:

                        view._log(f"[ERROR] Exception in topology_controller.py: {_e}")

                    except Exception:

                        pass
            setattr(view, attr, None)

