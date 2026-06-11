from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import re
import time
import warnings

import numpy as np

from swe2d.mesh.mesh_models import (
    CellConstraint, ConceptualArc, ConceptualModel,
    ConceptualNode, ConceptualRegion, MeshResult,
    MeshingBackend, QuadEdgeControl, _GmshQualityConfig,
)
from swe2d.mesh.mesh_quality import (
    _face_mesh_quality_stats, _gmsh_quality_passes,
    _gmsh_quality_score, _polygon_area_xy,
)
_MESHING_HELPERS = None

def _mh():
    """Lazy-import meshing helpers, avoids circular import with meshing.__getattr__."""
    global _MESHING_HELPERS
    if _MESHING_HELPERS is None:
        import swe2d.mesh.meshing as _mod
        _MESHING_HELPERS = _mod
    return _MESHING_HELPERS
logger = logging.getLogger(__name__)


def _gmsh_available() -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec("gmsh") is not None
    except Exception as e:
        logger.warning("_gmsh_available check failed: %s", e, exc_info=True)
        return False


class GmshBackend(MeshingBackend):
    """Production meshing backend using Gmsh 4.x.

    Geometry mapping:
    - Each ConceptualRegion  -> Gmsh Surface with per-region cell-type flags.
    - Each ConceptualArc     -> Gmsh embedded Curve (breakline/constraint).
    - Each CellConstraint    -> Gmsh Size-field override zone (Threshold field).

    Cell-type controls:
    - "triangular"   : Frontal-Delaunay algorithm (Gmsh algorithm 6).
    - "quadrilateral": Blossom quad recombination on top of Delaunay triangles.
    - "cartesian"    : Transfinite Surface + Recombine (structured grid, fast).
    - "empty"        : Surface excluded from mesh entirely.

    Output:
    - Polygon CSR topology (cell_face_offsets / cell_face_nodes) for the solver.
    - Triangulated cell_nodes (triangles-only fan decomposition) for plotting.
    - cell_type per face reflects the source conceptual type.
    """

    name = "gmsh"

    # Gmsh meshing algorithm codes
    _ALGO_FRONTAL = 6           # Frontal-Delaunay (quality triangles)
    _ALGO_DELAUNAY = 5          # Delaunay (fast fallback)
    _ALGO_PACKING_OF_PARALLELOGRAMS = 9  # good for recombination

    def __init__(self, options: Optional[Dict[str, object]] = None):
        self._options = dict(options or {})
        self._last_flow_align_diagnostics: List[Dict[str, object]] = []
        self._last_build_order_fingerprint: Dict[str, object] = {}
        self._last_build_order_stage_ladder: Dict[str, object] = {}
        self._last_pre_generate_entity_signature: Dict[str, object] = {}

    def _opt_int(self, name: str, default: int) -> int:
        value = self._options.get(name)
        if value is None:
            return int(default)
        try:
            return int(round(float(value)))
        except Exception as e:
            logger.warning("_opt_int conversion failed for %s: %s", name, e, exc_info=True)
            return int(default)

    def _opt_bool(self, name: str, default: bool) -> bool:
        value = self._options.get(name)
        if value is None:
            return bool(default)
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
        return bool(default)

    def _opt_float(self, name: str, default: float) -> float:
        value = self._options.get(name)
        if value is None:
            return float(default)
        try:
            return float(value)
        except Exception as e:
            logger.warning("_opt_float conversion failed for %s: %s", name, e, exc_info=True)
            return float(default)

    def _opt_float_tuple(self, name: str, default: Tuple[float, ...]) -> Tuple[float, ...]:
        value = self._options.get(name)
        if value is None:
            return tuple(float(v) for v in default)
        if isinstance(value, str):
            raw_items = value.replace(";", ",").split(",")
        elif isinstance(value, (list, tuple)):
            raw_items = list(value)
        else:
            return tuple(float(v) for v in default)
        parsed: List[float] = []
        for item in raw_items:
            text = str(item).strip()
            if not text:
                continue
            try:
                parsed.append(float(text))
            except Exception as e:
                logger.warning("_opt_float_tuple item parse failed: %s", e, exc_info=True)
                continue
        if not parsed:
            return tuple(float(v) for v in default)
        return tuple(parsed)

    def _opt_str_tuple(self, name: str, default: Tuple[str, ...]) -> Tuple[str, ...]:
        value = self._options.get(name)
        if value is None:
            return tuple(str(v) for v in default)
        if isinstance(value, str):
            raw_items = value.replace(";", ",").split(",")
        elif isinstance(value, (list, tuple)):
            raw_items = list(value)
        else:
            return tuple(str(v) for v in default)
        parsed: List[str] = []
        for item in raw_items:
            text = str(item).strip()
            if text:
                parsed.append(text)
        if not parsed:
            return tuple(str(v) for v in default)
        return tuple(parsed)

    def _gmsh_quality_config(self) -> _GmshQualityConfig:
        enabled = self._opt_bool(
            "gmsh_quality_enable",
            _mh()._env_bool("BACKWATER_GMSH_QUALITY_ENABLE", False),
        )
        strict = self._opt_bool(
            "gmsh_quality_strict",
            self._opt_bool(
                "tqmesh_quality_strict",
                _mh()._env_bool("BACKWATER_GMSH_QUALITY_STRICT", False),
            ),
        )
        min_angle_deg = self._opt_float(
            "gmsh_min_angle_deg",
            self._opt_float(
                "tqmesh_min_angle_deg",
                _mh()._env_float("BACKWATER_GMSH_MIN_ANGLE_DEG", 18.0),
            ),
        )
        max_aspect_ratio = self._opt_float(
            "gmsh_max_aspect_ratio",
            self._opt_float(
                "tqmesh_max_aspect_ratio",
                _mh()._env_float("BACKWATER_GMSH_MAX_ASPECT_RATIO", 12.0),
            ),
        )
        min_area_rel_bbox = self._opt_float(
            "gmsh_min_area_rel_bbox",
            self._opt_float(
                "tqmesh_min_area_rel_bbox",
                _mh()._env_float("BACKWATER_GMSH_MIN_AREA_REL_BBOX", 1.0e-11),
            ),
        )
        max_non_orth_deg = self._opt_float(
            "gmsh_max_non_orth_deg",
            _mh()._env_float("BACKWATER_GMSH_MAX_NON_ORTH_DEG", 82.0),
        )
        max_iterations = max(
            1,
            self._opt_int(
                "gmsh_quality_max_iterations",
                int(round(_mh()._env_float("BACKWATER_GMSH_QUALITY_MAX_ITERATIONS", 2.0))),
            ),
        )
        time_limit_s = max(
            1.0,
            self._opt_float(
                "gmsh_quality_time_limit_s",
                _mh()._env_float("BACKWATER_GMSH_QUALITY_TIME_LIMIT_S", 55.0),
            ),
        )
        size_scales = tuple(
            max(1.0e-3, float(v))
            for v in self._opt_float_tuple(
                "gmsh_quality_size_scales",
                self._opt_float_tuple("tqmesh_size_scales", (1.0, 0.9, 0.8, 0.7)),
            )
        )
        smooth_increments = tuple(
            max(0, int(round(v)))
            for v in self._opt_float_tuple(
                "gmsh_quality_smooth_increments",
                self._opt_float_tuple("tqmesh_smooth_increments", (0.0, 3.0, 6.0)),
            )
        )
        recombine_topology_passes = tuple(
            max(0, int(round(v)))
            for v in self._opt_float_tuple(
                "gmsh_quality_recombine_topology_passes",
                _mh()._env_csv_floats("BACKWATER_GMSH_QUALITY_RECOMBINE_TOPOLOGY_PASSES", (5.0, 12.0, 20.0)),
            )
        )
        recombine_min_quality = tuple(
            max(0.0, float(v))
            for v in self._opt_float_tuple(
                "gmsh_quality_recombine_minimum_quality",
                _mh()._env_csv_floats("BACKWATER_GMSH_QUALITY_RECOMBINE_MIN_QUALITY", (0.01, 0.03, 0.06)),
            )
        )
        random_factors = tuple(
            max(0.0, float(v))
            for v in self._opt_float_tuple(
                "gmsh_quality_random_factors",
                _mh()._env_csv_floats("BACKWATER_GMSH_QUALITY_RANDOM_FACTORS", (1.0e-9, 1.0e-7, 1.0e-6)),
            )
        )
        optimize_methods = tuple(
            str(v)
            for v in self._opt_str_tuple(
                "gmsh_quality_optimize_methods",
                _mh()._env_csv_strings("BACKWATER_GMSH_QUALITY_OPTIMIZE_METHODS", ("Laplace2D", "Relocate2D")),
            )
            if str(v).strip()
        )
        algorithm_switch_on_failure = self._opt_bool(
            "gmsh_algorithm_switch_on_failure",
            _mh()._env_bool("BACKWATER_GMSH_ALGO_SWITCH_ON_FAILURE", False),
        )
        recombine_node_repositioning = self._opt_bool(
            "gmsh_quality_recombine_node_repositioning",
            _mh()._env_bool("BACKWATER_GMSH_RECOMBINE_NODE_REPOSITIONING", True),
        )
        if not size_scales:
            size_scales = (1.0,)
        if not smooth_increments:
            smooth_increments = (0,)
        if not recombine_topology_passes:
            recombine_topology_passes = (5,)
        if not recombine_min_quality:
            recombine_min_quality = (0.01,)
        if not random_factors:
            random_factors = (1.0e-9,)
        # Guard against no-op retry ladders (e.g. "1.0" and "0").
        # When iterative quality is enabled, ensure attempts explore distinct
        # candidates even if the UI left legacy single-value defaults.
        if enabled and max_iterations > 1:
            if all(abs(float(v) - 1.0) <= 1.0e-12 for v in size_scales):
                size_scales = (1.0, 0.9, 0.8, 0.7)
            if all(int(v) == 0 for v in smooth_increments):
                smooth_increments = (0, 2, 4, 6)
            if len(recombine_topology_passes) == 1:
                recombine_topology_passes = (recombine_topology_passes[0], max(8, recombine_topology_passes[0] * 2))
            if len(recombine_min_quality) == 1:
                recombine_min_quality = (recombine_min_quality[0], max(0.02, recombine_min_quality[0] * 1.5))
            if len(random_factors) == 1:
                random_factors = (random_factors[0], max(1.0e-8, random_factors[0] * 100.0))
        return _GmshQualityConfig(
            enabled=enabled,
            strict=strict,
            min_angle_deg=max(1.0, float(min_angle_deg)),
            max_aspect_ratio=max(1.1, float(max_aspect_ratio)),
            min_area_rel_bbox=max(0.0, float(min_area_rel_bbox)),
            max_non_orth_deg=min(89.9, max(1.0, float(max_non_orth_deg))),
            max_iterations=int(max_iterations),
            time_limit_s=float(time_limit_s),
            size_scales=size_scales,
            smooth_increments=smooth_increments,
            recombine_topology_passes=recombine_topology_passes,
            recombine_min_quality=recombine_min_quality,
            random_factors=random_factors,
            optimize_methods=optimize_methods,
            algorithm_switch_on_failure=bool(algorithm_switch_on_failure),
            recombine_node_repositioning=bool(recombine_node_repositioning),
        )

    def generate(self, model: ConceptualModel) -> MeshResult:
        import gmsh

        if not model.regions:
            raise ValueError("No conceptual regions provided.")

        tri_algo = self._opt_int("gmsh_tri_algorithm", self._ALGO_FRONTAL)
        quad_algo = self._opt_int("gmsh_quad_algorithm", self._ALGO_FRONTAL)
        smoothing_passes = max(0, self._opt_int("gmsh_smoothing", 0))
        optimize_iters = max(0, self._opt_int("gmsh_optimize_iters", 0))
        recomb_algo = self._opt_int("gmsh_recombination_algorithm", 1)
        optimize_netgen = self._opt_bool("gmsh_optimize_netgen", False)
        verbosity = max(0, self._opt_int("gmsh_verbosity", 2))
        quality_cfg = self._gmsh_quality_config()
        checkpoint_path = str(self._options.get("gmsh_quality_checkpoint_path", "") or "").strip()
        progress_path = str(self._options.get("gmsh_progress_path", "") or "").strip()
        progress_emit_interval_s = _mh()._as_float(
            self._options.get("gmsh_progress_emit_interval_s"),
            0.75,
        )
        if (not np.isfinite(progress_emit_interval_s)) or progress_emit_interval_s <= 0.0:
            progress_emit_interval_s = 0.75
        progress_emit_interval_s = max(float(progress_emit_interval_s), 0.2)
        progress_seq = 0
        progress_last_emit = -1.0
        t_start = time.perf_counter()

        def _clip_progress_detail(detail: object, max_len: int = 240) -> str:
            txt = str(detail or "").strip()
            if len(txt) <= max_len:
                return txt
            return txt[: max_len - 3] + "..."

        def _emit_progress(
            stage: str,
            attempt: Optional[int] = None,
            detail: str = "",
            force: bool = False,
        ) -> None:
            nonlocal progress_seq, progress_last_emit
            if not progress_path:
                return
            now = time.perf_counter()
            if (not force) and progress_last_emit >= 0.0:
                if (now - progress_last_emit) < progress_emit_interval_s:
                    return
            progress_seq += 1
            payload: Dict[str, object] = {
                "seq": int(progress_seq),
                "stage": str(stage),
                "timestamp": float(time.time()),
                "elapsed_s": float(max(0.0, now - t_start)),
                "backend": "gmsh",
                "quality_loop_enabled": bool(quality_cfg.enabled),
            }
            if attempt is not None:
                payload["attempt"] = int(attempt)
            clipped = _clip_progress_detail(detail)
            if clipped:
                payload["detail"] = clipped
            try:
                _mh()._write_json_atomic(progress_path, payload)
                progress_last_emit = now
            except Exception as e:
                logger.warning("gmsh progress emit write failed: %s", e, exc_info=True)
                pass

        _emit_progress(
            "start",
            detail=(
                f"quality_loop={bool(quality_cfg.enabled)} max_iters={int(quality_cfg.max_iterations)} "
                f"budget_s={float(quality_cfg.time_limit_s):.2f}"
            ),
            force=True,
        )

        gmsh_logger_started = False
        gmsh_logger_emitted: set = set()

        def _emit_gmsh_logger_warnings() -> None:
            if not gmsh_logger_started:
                return
            try:
                msgs = list(gmsh.logger.get())
            except Exception as e:
                logger.warning("gmsh logger.get failed: %s", e, exc_info=True)
                return
            for raw in msgs:
                msg = str(raw).strip()
                if not msg:
                    continue
                low = msg.lower()
                if ("warning" not in low) and ("error" not in low):
                    continue
                if msg in gmsh_logger_emitted:
                    continue
                gmsh_logger_emitted.add(msg)
                warnings.warn(
                    f"Gmsh logger: {msg}",
                    RuntimeWarning,
                )

        # `interruptible=False` avoids installing a SIGINT handler, which lets
        # the Python API run from the QGIS bridge worker thread.
        gmsh.initialize(interruptible=False)
        gmsh.option.setNumber("General.Verbosity", float(verbosity))
        try:
            gmsh.logger.start()
            gmsh_logger_started = True
        except Exception as e:
            logger.warning("gmsh logger.start failed: %s", e, exc_info=True)
            gmsh_logger_started = False

        try:
            if not quality_cfg.enabled:
                flow_align_requested = self._opt_bool(
                    "gmsh_quad_full_region_flow_align",
                    _mh()._env_bool("BACKWATER_GMSH_QUAD_FULL_REGION_FLOW_ALIGN", False),
                )
                gmsh.model.add("swe2d")
                _emit_progress("build-start", detail="single-pass mode", force=True)
                try:
                    mesh = _mh()._require_nonempty_mesh(
                        self._build(
                            gmsh,
                            model,
                            tri_algo=tri_algo,
                            quad_algo=quad_algo,
                            smoothing_passes=smoothing_passes,
                            optimize_iters=optimize_iters,
                            recomb_algo=recomb_algo,
                            optimize_netgen=optimize_netgen,
                            size_scale=1.0,
                        ),
                        "Gmsh",
                    )
                    _emit_gmsh_logger_warnings()
                    n_nodes = int(np.asarray(mesh.node_x).size)
                    n_faces = max(0, int(np.asarray(mesh.cell_face_offsets).size) - 1)
                    _emit_progress("done", detail=f"nodes={n_nodes} faces={n_faces}", force=True)
                    return mesh
                except Exception as exc:
                    _emit_gmsh_logger_warnings()
                    if not flow_align_requested:
                        _emit_progress("fail", detail=f"single-pass build failed: {exc}", force=True)
                        raise

                    _emit_progress(
                        "flow-align-fallback-start",
                        detail=f"initial flow-align build failed: {exc}",
                        force=True,
                    )
                    diagnostics = copy.deepcopy(self._last_flow_align_diagnostics)
                    warnings.warn(
                        "Gmsh full-region flow-aligned quads failed on initial pass; "
                        "retrying with per-region flow-align disabled. "
                        f"Initial error: {exc}",
                        RuntimeWarning,
                    )

                    prev_flow_align = self._options.get("gmsh_quad_full_region_flow_align", None)
                    self._options["gmsh_quad_full_region_flow_align"] = False
                    try:
                        gmsh.clear()
                        gmsh.model.add("swe2d_fallback_no_flow_align")
                        _emit_progress(
                            "build-retry",
                            detail="retry with per-region flow-align disabled",
                            force=True,
                        )
                        try:
                            fallback_mesh = _mh()._require_nonempty_mesh(
                                self._build(
                                    gmsh,
                                    model,
                                    tri_algo=tri_algo,
                                    quad_algo=quad_algo,
                                    smoothing_passes=smoothing_passes,
                                    optimize_iters=optimize_iters,
                                    recomb_algo=recomb_algo,
                                    optimize_netgen=optimize_netgen,
                                    size_scale=1.0,
                                ),
                                "Gmsh",
                            )
                            _emit_gmsh_logger_warnings()
                            n_nodes = int(np.asarray(fallback_mesh.node_x).size)
                            n_faces = max(0, int(np.asarray(fallback_mesh.cell_face_offsets).size) - 1)
                            _emit_progress(
                                "done",
                                detail=(
                                    f"flow-align fallback success nodes={n_nodes} faces={n_faces} "
                                    "mode=no-flow-align"
                                ),
                                force=True,
                            )
                        except Exception as fallback_exc:
                            _emit_gmsh_logger_warnings()
                            diag_txt = "none"
                            if diagnostics:
                                parts = []
                                for d in diagnostics:
                                    rid_txt = str(d.get("region_id", "?"))
                                    status_txt = str(d.get("status", "unknown"))
                                    reasons_txt = ",".join(str(x) for x in d.get("reasons", []) if str(x))
                                    if not reasons_txt:
                                        reasons_txt = "none"
                                    parts.append(
                                        f"region={rid_txt};status={status_txt};reasons={reasons_txt}"
                                    )
                                diag_txt = " | ".join(parts)
                            _emit_progress(
                                "fail",
                                detail=(
                                    "flow-align fallback failed "
                                    f"initial={_clip_progress_detail(exc)} "
                                    f"fallback={_clip_progress_detail(fallback_exc)}"
                                ),
                                force=True,
                            )
                            raise RuntimeError(
                                "Gmsh flow-align fallback retry failed. "
                                f"initial_error={exc}; fallback_error={fallback_exc}; "
                                f"per_region_diagnostics={diag_txt}"
                            )
                    finally:
                        if prev_flow_align is None:
                            self._options.pop("gmsh_quad_full_region_flow_align", None)
                        else:
                            self._options["gmsh_quad_full_region_flow_align"] = prev_flow_align

                    merged_summary = dict(fallback_mesh.quality_summary or {})
                    merged_summary["gmsh_flow_align_runtime_fallback"] = {
                        "triggered": True,
                        "initial_error": str(exc),
                    }
                    if diagnostics:
                        merged_summary["gmsh_flow_align_diagnostics"] = diagnostics
                    fallback_mesh.quality_summary = merged_summary
                    return fallback_mesh

            start_t = time.perf_counter()
            best_mesh: Optional[MeshResult] = None
            best_stats: Optional[Dict[str, float]] = None
            best_score = -1.0e30
            attempts = 0
            attempt_errors: List[str] = []
            scale_i = 0
            smooth_i = 0
            had_passing_candidate = False
            last_attempt_duration_s: Optional[float] = None
            hit_time_budget = False

            # Alternate between configured and fallback algorithms so retries
            # can escape deterministic local minima with identical topology.
            tri_algo_ladder = [int(tri_algo)]
            tri_alt = self._ALGO_DELAUNAY if int(tri_algo) != self._ALGO_DELAUNAY else self._ALGO_FRONTAL
            if tri_alt not in tri_algo_ladder:
                tri_algo_ladder.append(int(tri_alt))

            quad_algo_ladder = [int(quad_algo)]
            quad_alt = self._ALGO_DELAUNAY if int(quad_algo) != self._ALGO_DELAUNAY else self._ALGO_FRONTAL
            if quad_alt not in quad_algo_ladder:
                quad_algo_ladder.append(int(quad_alt))

            recomb_ladder = [int(recomb_algo)]
            recomb_alt = 0 if int(recomb_algo) != 0 else 1
            if recomb_alt not in recomb_ladder:
                recomb_ladder.append(int(recomb_alt))

            recomb_topology_ladder = [int(v) for v in quality_cfg.recombine_topology_passes if int(v) >= 0]
            if not recomb_topology_ladder:
                recomb_topology_ladder = [5]
            recomb_min_quality_ladder = [max(0.0, float(v)) for v in quality_cfg.recombine_min_quality]
            if not recomb_min_quality_ladder:
                recomb_min_quality_ladder = [0.01]
            random_factor_ladder = [max(0.0, float(v)) for v in quality_cfg.random_factors]
            if not random_factor_ladder:
                random_factor_ladder = [1.0e-9]

            while attempts < quality_cfg.max_iterations:
                elapsed = time.perf_counter() - start_t
                if elapsed >= quality_cfg.time_limit_s:
                    hit_time_budget = True
                    _emit_progress(
                        "budget-stop",
                        attempt=int(attempts),
                        detail=(
                            f"elapsed={elapsed:.2f}s reached budget={float(quality_cfg.time_limit_s):.2f}s"
                        ),
                        force=True,
                    )
                    break

                # Avoid starting a fresh attempt when little time remains. A
                # single Gmsh attempt is non-interruptible, so launching a retry
                # too close to the deadline can overrun and get killed by the
                # outer watchdog before best-candidate export runs.
                remaining_s = max(0.0, quality_cfg.time_limit_s - elapsed)
                if last_attempt_duration_s is not None and attempts > 0:
                    min_retry_window_s = max(2.0, 0.75 * float(last_attempt_duration_s))
                    if remaining_s < min_retry_window_s:
                        hit_time_budget = True
                        _emit_progress(
                            "budget-stop",
                            attempt=int(attempts),
                            detail=(
                                f"remaining={remaining_s:.2f}s too low for new attempt "
                                f"need~{min_retry_window_s:.2f}s"
                            ),
                            force=True,
                        )
                        warnings.warn(
                            "Gmsh quality loop stopping retries early due to low remaining budget "
                            f"(remaining={remaining_s:.2f}s, needed~{min_retry_window_s:.2f}s); "
                            "returning best available candidate.",
                            RuntimeWarning,
                        )
                        break

                gmsh.clear()
                gmsh.model.add(f"swe2d_try_{attempts + 1}")

                size_scale = quality_cfg.size_scales[scale_i % len(quality_cfg.size_scales)]
                smooth_inc = quality_cfg.smooth_increments[smooth_i % len(quality_cfg.smooth_increments)]
                scale_i += 1
                if scale_i % len(quality_cfg.size_scales) == 0:
                    smooth_i += 1
                tri_try = tri_algo_ladder[attempts % len(tri_algo_ladder)]
                quad_try = quad_algo_ladder[(attempts // len(tri_algo_ladder)) % len(quad_algo_ladder)]
                recomb_try = recomb_ladder[(attempts // max(1, len(tri_algo_ladder) * len(quad_algo_ladder))) % len(recomb_ladder)]
                recomb_topology_try = recomb_topology_ladder[attempts % len(recomb_topology_ladder)]
                recomb_min_quality_try = recomb_min_quality_ladder[attempts % len(recomb_min_quality_ladder)]
                random_factor_try = random_factor_ladder[attempts % len(random_factor_ladder)]

                _emit_progress(
                    "attempt-start",
                    attempt=int(attempts + 1),
                    detail=(
                        f"tri={tri_try} quad={quad_try} recomb={recomb_try} "
                        f"topo={int(recomb_topology_try)} minq={float(recomb_min_quality_try):.3f} "
                        f"rand={float(random_factor_try):.2e} size_scale={float(size_scale):.3f} "
                        f"smooth={int(max(0, smoothing_passes + int(smooth_inc)))}"
                    ),
                    force=True,
                )

                attempt_start_t = time.perf_counter()
                try:
                    mesh = _mh()._require_nonempty_mesh(
                        self._build(
                            gmsh,
                            model,
                            tri_algo=tri_try,
                            quad_algo=quad_try,
                            smoothing_passes=max(0, smoothing_passes + int(smooth_inc)),
                            optimize_iters=optimize_iters,
                            recomb_algo=recomb_try,
                            optimize_netgen=optimize_netgen,
                            size_scale=float(size_scale),
                            recombine_optimize_topology=int(recomb_topology_try),
                            recombine_node_repositioning=bool(quality_cfg.recombine_node_repositioning),
                            recombine_minimum_quality=float(recomb_min_quality_try),
                            optimize_methods=tuple(quality_cfg.optimize_methods),
                            random_factor=float(random_factor_try),
                            algorithm_switch_on_failure=bool(quality_cfg.algorithm_switch_on_failure),
                        ),
                        "Gmsh",
                    )
                    _emit_gmsh_logger_warnings()
                    stats = _face_mesh_quality_stats(mesh, quality_cfg)
                    score = _gmsh_quality_score(stats, quality_cfg)

                    attempt_summary = {
                        "attempts": int(attempts + 1),
                        "strict_requested": bool(quality_cfg.strict),
                        "had_passing_candidate": bool(_gmsh_quality_passes(stats, quality_cfg)),
                        "best_stats": dict(stats),
                        "recombine_topology_passes": int(recomb_topology_try),
                        "recombine_minimum_quality": float(recomb_min_quality_try),
                        "random_factor": float(random_factor_try),
                        "optimize_methods": list(quality_cfg.optimize_methods),
                        "checkpoint": True,
                    }
                    if checkpoint_path:
                        try:
                            _mh()._write_mesh_checkpoint_npz(checkpoint_path, mesh, attempt_summary)
                        except Exception as cp_exc:
                            warnings.warn(
                                f"Gmsh quality checkpoint write failed (attempt {attempts + 1}): {cp_exc}",
                                RuntimeWarning,
                            )

                    if score > best_score:
                        best_score = score
                        best_mesh = mesh
                        best_stats = stats

                    if _gmsh_quality_passes(stats, quality_cfg):
                        had_passing_candidate = True
                        if quality_cfg.strict:
                            _emit_progress(
                                "done",
                                attempt=int(attempts + 1),
                                detail="strict mode accepted first passing candidate",
                                force=True,
                            )
                            # Strict mode only needs the first passing candidate.
                            summary = dict(mesh.quality_summary or {})
                            summary.update({
                                "attempts": int(attempts + 1),
                                "strict_requested": bool(quality_cfg.strict),
                                "had_passing_candidate": True,
                                "best_stats": dict(stats),
                            })
                            mesh.quality_summary = summary
                            return mesh
                except Exception as exc:
                    _emit_gmsh_logger_warnings()
                    err_msg = (
                        f"Gmsh quality attempt {attempts + 1} failed for tri={tri_try}, quad={quad_try}, "
                        f"recomb={recomb_try}, topo={int(recomb_topology_try)}, minq={float(recomb_min_quality_try):.3f}, "
                        f"rand={float(random_factor_try):.2e}, size_scale={size_scale:.3f}, "
                        f"smooth={smoothing_passes + int(smooth_inc)}: {exc}"
                    )
                    attempt_errors.append(err_msg)
                    warnings.warn(
                        err_msg,
                        RuntimeWarning,
                    )
                    _emit_progress(
                        "attempt-fail",
                        attempt=int(attempts + 1),
                        detail=err_msg,
                        force=True,
                    )
                else:
                    warnings.warn(
                        "Gmsh quality attempt "
                        f"{attempts + 1}: "
                        f"fail_cells(any/angle/aspect/area/non_orth)="
                        f"{int(stats.get('failed_any_cells', 0.0))}/"
                        f"{int(stats.get('failed_min_angle_cells', 0.0))}/"
                        f"{int(stats.get('failed_max_aspect_cells', 0.0))}/"
                        f"{int(stats.get('failed_min_area_cells', 0.0))}/"
                        f"{int(stats.get('failed_max_non_orth_cells', 0.0))}",
                        RuntimeWarning,
                    )
                    _emit_progress(
                        "attempt-done",
                        attempt=int(attempts + 1),
                        detail=(
                            f"passed={bool(_gmsh_quality_passes(stats, quality_cfg))} "
                            f"fail_any={int(stats.get('failed_any_cells', 0.0))}"
                        ),
                        force=True,
                    )

                last_attempt_duration_s = max(0.0, time.perf_counter() - attempt_start_t)
                attempts += 1

            if best_mesh is None or best_stats is None:
                # Best-effort fallback: regardless of iterative quality failures,
                # run one plain baseline build so downstream export still has a mesh
                # whenever geometry is meshable at all.
                try:
                    _emit_progress("fallback-start", detail="building best-effort baseline candidate", force=True)
                    gmsh.clear()
                    gmsh.model.add("swe2d_best_effort_fallback")
                    fallback_mesh = _mh()._require_nonempty_mesh(
                        self._build(
                            gmsh,
                            model,
                            tri_algo=tri_algo,
                            quad_algo=quad_algo,
                            smoothing_passes=smoothing_passes,
                            optimize_iters=optimize_iters,
                            recomb_algo=recomb_algo,
                            optimize_netgen=optimize_netgen,
                            size_scale=1.0,
                            recombine_optimize_topology=int(recomb_topology_ladder[0]),
                            recombine_node_repositioning=bool(quality_cfg.recombine_node_repositioning),
                            recombine_minimum_quality=float(recomb_min_quality_ladder[0]),
                            optimize_methods=tuple(quality_cfg.optimize_methods),
                            random_factor=float(random_factor_ladder[0]),
                            algorithm_switch_on_failure=bool(quality_cfg.algorithm_switch_on_failure),
                        ),
                        "Gmsh",
                    )
                    _emit_gmsh_logger_warnings()
                    fallback_stats = _face_mesh_quality_stats(fallback_mesh, quality_cfg)
                    fallback_summary = dict(fallback_mesh.quality_summary or {})
                    fallback_summary.update({
                        "attempts": int(attempts + 1),
                        "strict_requested": bool(quality_cfg.strict),
                        "had_passing_candidate": bool(_gmsh_quality_passes(fallback_stats, quality_cfg)),
                        "best_stats": dict(fallback_stats),
                        "best_effort_fallback": True,
                        "time_budget_exhausted": bool(hit_time_budget),
                    })
                    fallback_mesh.quality_summary = fallback_summary
                    if checkpoint_path:
                        try:
                            _mh()._write_mesh_checkpoint_npz(
                                checkpoint_path,
                                fallback_mesh,
                                fallback_mesh.quality_summary,
                            )
                        except Exception as cp_exc:
                            warnings.warn(
                                f"Gmsh fallback checkpoint write failed: {cp_exc}",
                                RuntimeWarning,
                            )
                    warnings.warn(
                        "Gmsh quality loop produced no valid candidate during iterative retries; "
                        "using best-effort fallback mesh for export.",
                        RuntimeWarning,
                    )
                    n_nodes = int(np.asarray(fallback_mesh.node_x).size)
                    n_faces = max(0, int(np.asarray(fallback_mesh.cell_face_offsets).size) - 1)
                    _emit_progress(
                        "done",
                        detail=f"best-effort fallback nodes={n_nodes} faces={n_faces}",
                        force=True,
                    )
                    return fallback_mesh
                except Exception as fallback_exc:
                    tail = "; ".join(attempt_errors[-3:]) if attempt_errors else "no attempt diagnostics"
                    _emit_progress(
                        "fail",
                        detail=(
                            "quality loop had no viable candidate and fallback failed: "
                            f"{_clip_progress_detail(fallback_exc)}"
                        ),
                        force=True,
                    )
                    raise RuntimeError(
                        "Gmsh quality loop produced no valid non-empty mesh candidate, and "
                        f"best-effort fallback also failed: {fallback_exc}. "
                        f"Recent attempt errors: {tail}"
                    )

            if had_passing_candidate:
                summary = dict(best_mesh.quality_summary or {})
                summary.update({
                    "attempts": int(attempts),
                    "strict_requested": bool(quality_cfg.strict),
                    "had_passing_candidate": True,
                    "best_stats": dict(best_stats),
                    "time_budget_exhausted": bool(hit_time_budget),
                })
                best_mesh.quality_summary = summary
                n_nodes = int(np.asarray(best_mesh.node_x).size)
                n_faces = max(0, int(np.asarray(best_mesh.cell_face_offsets).size) - 1)
                _emit_progress(
                    "done",
                    detail=(
                        f"best passing candidate nodes={n_nodes} faces={n_faces} "
                        f"attempts={int(attempts)}"
                    ),
                    force=True,
                )
                return best_mesh

            diag = (
                "min_angle={:.2f} deg, max_aspect={:.2f}, min_area={:.3e}, max_non_orth={:.2f} deg"
                .format(
                    float(best_stats.get("min_angle_deg", 0.0)),
                    float(best_stats.get("max_aspect_ratio", float("inf"))),
                    float(best_stats.get("min_area", 0.0)),
                    float(best_stats.get("max_non_orth_deg", 90.0)),
                )
            )
            summary = dict(best_mesh.quality_summary or {})
            summary.update({
                "attempts": int(attempts),
                "strict_requested": bool(quality_cfg.strict),
                "had_passing_candidate": False,
                "best_stats": dict(best_stats),
                "time_budget_exhausted": bool(hit_time_budget),
            })
            best_mesh.quality_summary = summary
            warnings.warn(
                "Gmsh quality constraints were not met; using best available candidate "
                f"(attempts={attempts}, time_limit_s={quality_cfg.time_limit_s:.1f}). {diag}",
                RuntimeWarning,
            )
            n_nodes = int(np.asarray(best_mesh.node_x).size)
            n_faces = max(0, int(np.asarray(best_mesh.cell_face_offsets).size) - 1)
            _emit_progress(
                "done",
                detail=(
                    f"best nonpassing candidate nodes={n_nodes} faces={n_faces} "
                    f"attempts={int(attempts)}"
                ),
                force=True,
            )
            return best_mesh
        finally:
            _emit_gmsh_logger_warnings()
            if gmsh_logger_started:
                try:
                    gmsh.logger.stop()
                except Exception as e:
                    logger.warning("gmsh logger.stop failed: %s", e, exc_info=True)
                    pass
            gmsh.finalize()

    # ------------------------------------------------------------------
    # Internal construction helpers
    # ------------------------------------------------------------------

    def _build(
        self,
        gmsh,
        model: ConceptualModel,
        tri_algo: int,
        quad_algo: int,
        smoothing_passes: int,
        optimize_iters: int,
        recomb_algo: int,
        optimize_netgen: bool,
        size_scale: float,
        recombine_optimize_topology: int = 5,
        recombine_node_repositioning: bool = True,
        recombine_minimum_quality: float = 0.01,
        optimize_methods: Tuple[str, ...] = (),
        random_factor: float = 1.0e-9,
        algorithm_switch_on_failure: bool = True,
    ) -> MeshResult:
        build_started_at = time.perf_counter()
        gmsh_phase_timings_s: Dict[str, float] = {}

        def _record_phase(phase_name: str, started_at: float) -> None:
            gmsh_phase_timings_s[str(phase_name)] = float(max(0.0, time.perf_counter() - started_at))

        arc_mode = str(self._options.get("gmsh_arc_mode", "hard_embed") or "hard_embed").strip().lower()
        if arc_mode not in {"hard_embed", "soft_size_hint", "disabled"}:
            arc_mode = "hard_embed"
        mesh_size_min = max(0.0, self._opt_float("gmsh_mesh_size_min", 0.0))
        tolerance_edge_length = max(0.0, self._opt_float("gmsh_tolerance_edge_length", 0.0))
        mesh_size_from_points = self._opt_bool("gmsh_mesh_size_from_points", False)
        gmsh_num_threads = max(
            0,
            int(
                round(
                    self._opt_float(
                        "gmsh_num_threads",
                        _mh()._env_float("BACKWATER_GMSH_NUM_THREADS", 1.0),
                    )
                )
            ),
        )
        gmsh_max_num_threads_2d = max(
            0,
            int(
                round(
                    self._opt_float(
                        "gmsh_max_num_threads_2d",
                        _mh()._env_float("BACKWATER_GMSH_MAX_NUM_THREADS_2D", 0.0),
                    )
                )
            ),
        )
        gmsh_global_recombine = self._opt_bool(
            "gmsh_global_recombine",
            _mh()._env_bool("BACKWATER_GMSH_GLOBAL_RECOMBINE", False),
        )
        gmsh_quad_full_region_flow_align = self._opt_bool(
            "gmsh_quad_full_region_flow_align",
            _mh()._env_bool("BACKWATER_GMSH_QUAD_FULL_REGION_FLOW_ALIGN", False),
        )
        gmsh_interface_transition_enable = self._opt_bool(
            "gmsh_interface_transition_enable",
            _mh()._env_bool("BACKWATER_GMSH_INTERFACE_TRANSITION_ENABLE", True),
        )
        gmsh_interface_transition_dist_factor = max(
            0.25,
            self._opt_float(
                "gmsh_interface_transition_dist_factor",
                _mh()._env_float("BACKWATER_GMSH_INTERFACE_TRANSITION_DIST_FACTOR", 2.5),
            ),
        )
        gmsh_interface_transition_min_ratio = max(
            1.0,
            self._opt_float(
                "gmsh_interface_transition_min_ratio",
                _mh()._env_float("BACKWATER_GMSH_INTERFACE_TRANSITION_MIN_RATIO", 1.25),
            ),
        )
        gmsh_transfinite_shared_interface_harmonize = self._opt_bool(
            "gmsh_transfinite_shared_interface_harmonize",
            _mh()._env_bool("BACKWATER_GMSH_TRANSFINITE_SHARED_INTERFACE_HARMONIZE", False),
        )
        gmsh_interface_conformance = self._opt_bool(
            "gmsh_interface_conformance",
            _mh()._env_bool("BACKWATER_GMSH_INTERFACE_CONFORMANCE", False),
        )
        gmsh_transverse_interface_centroid_merge = self._opt_bool(
            "gmsh_transverse_interface_centroid_merge",
            _mh()._env_bool("BACKWATER_GMSH_TRANSVERSE_INTERFACE_CENTROID_MERGE", False),
        )
        gmsh_interface_snap_tol = max(
            1.0e-9,
            self._opt_float(
                "gmsh_interface_snap_tol",
                _mh()._env_float("BACKWATER_GMSH_INTERFACE_SNAP_TOL", 1.0),
            ),
        )
        gmsh_interface_reject_near_unshared = self._opt_bool(
            "gmsh_interface_reject_near_unshared",
            _mh()._env_bool("BACKWATER_GMSH_INTERFACE_REJECT_NEAR_UNSHARED", True),
        )
        gmsh_interface_reject_tol = max(
            1.0e-9,
            self._opt_float(
                "gmsh_interface_reject_tol",
                _mh()._env_float("BACKWATER_GMSH_INTERFACE_REJECT_TOL", 1.0e-3),
            ),
        )
        if gmsh_transverse_interface_centroid_merge:
            gmsh_interface_conformance = True
        gmsh_shared_transverse_edge_count_normalize = True
        gmsh_transfinite_opposite_subset_start = max(
            0.0,
            min(
                1.0,
                self._opt_float(
                    "gmsh_transfinite_opposite_subset_start",
                    _mh()._env_float("BACKWATER_GMSH_TRANSFINITE_OPPOSITE_SUBSET_START", 0.30),
                ),
            ),
        )
        gmsh_transfinite_opposite_subset_end = max(
            0.0,
            min(
                1.0,
                self._opt_float(
                    "gmsh_transfinite_opposite_subset_end",
                    _mh()._env_float("BACKWATER_GMSH_TRANSFINITE_OPPOSITE_SUBSET_END", 0.70),
                ),
            ),
        )
        gmsh_transfinite_opposite_subset_density_scale = max(
            0.05,
            self._opt_float(
                "gmsh_transfinite_opposite_subset_density_scale",
                _mh()._env_float("BACKWATER_GMSH_TRANSFINITE_OPPOSITE_SUBSET_DENSITY_SCALE", 0.50),
            ),
        )
        gmsh_transfinite_interface_debug = self._opt_bool(
            "gmsh_transfinite_interface_debug",
            _mh()._env_bool("BACKWATER_GMSH_TRANSFINITE_INTERFACE_DEBUG", False),
        )
        gmsh_transfinite_subset_containment_enable = self._opt_bool(
            "gmsh_transfinite_subset_containment_enable",
            _mh()._env_bool("BACKWATER_GMSH_TRANSFINITE_SUBSET_CONTAINMENT_ENABLE", True),
        )
        gmsh_transfinite_subset_containment_high_overlap = max(
            0.50,
            min(
                1.0,
                self._opt_float(
                    "gmsh_transfinite_subset_containment_high_overlap",
                    _mh()._env_float("BACKWATER_GMSH_TRANSFINITE_SUBSET_CONTAINMENT_HIGH_OVERLAP", 0.95),
                ),
            ),
        )
        gmsh_transfinite_subset_containment_min_overlap = max(
            0.0,
            min(
                gmsh_transfinite_subset_containment_high_overlap,
                self._opt_float(
                    "gmsh_transfinite_subset_containment_min_overlap",
                    _mh()._env_float("BACKWATER_GMSH_TRANSFINITE_SUBSET_CONTAINMENT_MIN_OVERLAP", 0.02),
                ),
            ),
        )
        gmsh_transfinite_subset_containment_max_length_ratio = max(
            1.0e-6,
            self._opt_float(
                "gmsh_transfinite_subset_containment_max_length_ratio",
                _mh()._env_float("BACKWATER_GMSH_TRANSFINITE_SUBSET_CONTAINMENT_MAX_LENGTH_RATIO", 0.35),
            ),
        )
        arc_soft_size_factor = min(1.0, max(0.05, self._opt_float("gmsh_arc_soft_size_factor", 0.5)))
        arc_soft_dist_factor = max(0.1, self._opt_float("gmsh_arc_soft_dist_factor", 2.0))

        # Tolerance for point deduplication (scaled to typical hydraulic coords).
        tol = 1e-6
        surface_tags: List[int] = []
        surface_meta: List[Tuple[int, str, float]] = []  # (region_id, cell_type, target_size)
        surface_curve_tags: Dict[int, List[int]] = {}
        surface_quad_controls: Dict[int, Optional[List[QuadEdgeControl]]] = {}
        surface_quad_edge_curve_groups: Dict[int, Optional[List[List[int]]]] = {}
        flow_align_diagnostics: List[Dict[str, object]] = []
        self._last_flow_align_diagnostics = []

        # Shared geometry registries for conforming inter-region interfaces.
        # Points and single-segment lines on shared boundaries are reused so
        # Gmsh meshes that interface curve exactly once.  Without this, each
        # region independently creates duplicate points/curves at the same
        # physical location; Gmsh then discretises the shared edge twice with
        # potentially different node counts, producing hanging nodes that
        # immediately destabilise the FVM solver.
        _pt_prec = 6  # rounding digits ≈ 1 µm — sufficient for hydraulic coords
        pt_reg: Dict[Tuple[float, float], int] = {}   # (rx,ry) -> gmsh point tag
        pt_xy_by_tag: Dict[int, Tuple[float, float]] = {}
        seg_reg: Dict[Tuple[int, int], int] = {}       # (p0,p1) -> signed curve tag
        polycurve_reg: Dict[Tuple[int, ...], int] = {}  # polyline point tag chain -> curve tag
        polycurve_chain_by_tag: Dict[int, Tuple[int, ...]] = {}
        quad_curve_chain_by_abs: Dict[int, Tuple[int, ...]] = {}
        quad_curve_candidates_by_endpoint: Dict[int, List[int]] = {}
        build_order_events: List[str] = []
        build_order_stage_marks: List[Tuple[str, int]] = []
        build_order_event_cap = min(
            200000,
            max(2000, self._opt_int("gmsh_build_order_event_cap", 50000)),
        )
        build_order_overflow = False
        global_option_tokens: List[str] = []

        def _fmt_event_part(value: object) -> str:
            if isinstance(value, float):
                if np.isfinite(float(value)):
                    return f"{float(value):.12g}"
                return "nan"
            if isinstance(value, (list, tuple)):
                return "[" + ",".join(_fmt_event_part(v) for v in value) + "]"
            return str(value)

        def _record_build_event(event: str, *parts: object) -> None:
            nonlocal build_order_overflow
            if len(build_order_events) >= int(build_order_event_cap):
                build_order_overflow = True
                return
            if parts:
                build_order_events.append(
                    str(event) + "|" + "|".join(_fmt_event_part(p) for p in parts)
                )
            else:
                build_order_events.append(str(event))

        def _sha256_lines(lines: Sequence[str]) -> str:
            digest = hashlib.sha256()
            for line in lines:
                digest.update(str(line).encode("utf-8", "replace"))
                digest.update(b"\n")
            return digest.hexdigest()

        def _preview_tokens(lines: Sequence[str], n: int = 12) -> Dict[str, List[str]]:
            items = [str(v) for v in list(lines or [])]
            n_use = max(0, int(n))
            if len(items) <= 2 * n_use:
                return {"head": items, "tail": []}
            return {"head": items[:n_use], "tail": items[-n_use:]}

        def _hash_int_sequence(values: Sequence[int], limit: int = 1024) -> str:
            vals = [int(v) for v in list(values or [])[: max(1, int(limit))]]
            return _sha256_lines([",".join(str(v) for v in vals)])

        def _build_order_fingerprint_payload() -> Dict[str, object]:
            return {
                "sha256": _sha256_lines(build_order_events),
                "event_count": int(len(build_order_events)),
                "event_cap": int(build_order_event_cap),
                "overflow": bool(build_order_overflow),
                "preview": _preview_tokens(build_order_events, n=16),
            }

        def _mark_build_stage(label: str) -> None:
            build_order_stage_marks.append((str(label), int(len(build_order_events))))

        def _build_order_stage_ladder_payload() -> Dict[str, object]:
            stages: List[Dict[str, object]] = []
            prev_idx = 0
            marks = list(build_order_stage_marks)
            if not marks:
                marks = [("full", int(len(build_order_events)))]

            for label, end_idx_raw in marks:
                end_idx = max(prev_idx, min(int(end_idx_raw), int(len(build_order_events))))
                stage_lines = list(build_order_events[prev_idx:end_idx])
                stage_sha = _sha256_lines(stage_lines)
                cumulative_sha = _sha256_lines(build_order_events[:end_idx])
                stages.append({
                    "label": str(label),
                    "start_index": int(prev_idx),
                    "end_index": int(end_idx),
                    "event_count": int(max(0, end_idx - prev_idx)),
                    "stage_sha256": str(stage_sha),
                    "cumulative_sha256": str(cumulative_sha),
                })
                prev_idx = int(end_idx)

            if prev_idx < len(build_order_events):
                end_idx = int(len(build_order_events))
                stage_lines = list(build_order_events[prev_idx:end_idx])
                stage_sha = _sha256_lines(stage_lines)
                cumulative_sha = _sha256_lines(build_order_events[:end_idx])
                stages.append({
                    "label": "tail",
                    "start_index": int(prev_idx),
                    "end_index": int(end_idx),
                    "event_count": int(max(0, end_idx - prev_idx)),
                    "stage_sha256": str(stage_sha),
                    "cumulative_sha256": str(cumulative_sha),
                })

            compact_lines = [
                f"{str(s.get('label', ''))}|{int(s.get('event_count', 0))}|{str(s.get('stage_sha256', ''))}"
                for s in stages
            ]
            return {
                "sha256": _sha256_lines(compact_lines),
                "stage_count": int(len(stages)),
                "stages": stages,
            }

        def _build_order_stage_ladder_compact_text(payload: Dict[str, object]) -> str:
            stages = list(payload.get("stages") or [])
            parts: List[str] = []
            for stage in stages:
                label = str(stage.get("label", ""))
                count = int(stage.get("event_count", 0) or 0)
                sha = str(stage.get("stage_sha256", ""))
                parts.append(f"{label}:{count}:{sha[:12]}")
            txt = ",".join(parts)
            if len(txt) > 420:
                return txt[:417] + "..."
            return txt

        def _global_option_value_text(value: object) -> str:
            return _fmt_event_part(value).replace("\n", "\\n")

        def _record_global_option(name: str, value: object) -> None:
            opt_name = str(name)
            opt_value = _global_option_value_text(value)
            global_option_tokens.append(f"{opt_name}={opt_value}")
            _record_build_event("mesh-option", opt_name, opt_value)

        def _global_options_payload() -> Dict[str, object]:
            entries = [str(v) for v in global_option_tokens]
            return {
                "sha256": _sha256_lines(entries),
                "count": int(len(entries)),
                "entries": entries,
                "preview": _preview_tokens(entries, n=20),
            }

        def _global_options_compact_text(payload: Dict[str, object]) -> str:
            entries = [str(v) for v in list(payload.get("entries") or [])]
            txt = ";".join(entries)
            if len(txt) > 420:
                return txt[:417] + "..."
            return txt

        def _fmt_float_token(value: object, digits: int = 9) -> str:
            try:
                fv = float(value)
            except Exception as e:
                logger.warning("_fmt_float_token conversion failed: %s", e, exc_info=True)
                return "nan"
            if not np.isfinite(fv):
                return "nan"
            rounded = round(float(fv), int(digits))
            return f"{rounded:.{int(digits)}f}"

        def _safe_bbox(dim: int, tag: int) -> Tuple[float, float, float, float, float, float]:
            try:
                bbox = gmsh.model.getBoundingBox(int(dim), int(tag))
                if bbox is None or len(bbox) != 6:
                    raise ValueError("invalid bbox")
                return tuple(float(v) for v in bbox)
            except Exception as e:
                logger.warning("_safe_bbox failed for dim=%s tag=%s: %s", dim, tag, e, exc_info=True)
                return (float("nan"),) * 6

        def _entity_tokens_pre_generate() -> Tuple[List[str], List[str], List[str]]:
            point_tokens: List[str] = []
            curve_tokens: List[str] = []
            surface_tokens: List[str] = []

            try:
                point_entities = gmsh.model.getEntities(0)
            except Exception as e:
                logger.warning("gmsh getEntities(0) failed: %s", e, exc_info=True)
                point_entities = []
            point_tags = sorted(
                int(tag)
                for dim, tag in list(point_entities or [])
                if int(dim) == 0
            )
            for ptag in point_tags:
                pxy = pt_xy_by_tag.get(int(ptag))
                if pxy is not None:
                    px = float(pxy[0])
                    py = float(pxy[1])
                else:
                    bb = _safe_bbox(0, int(ptag))
                    px = float(bb[0])
                    py = float(bb[1])
                point_tokens.append(
                    f"{int(ptag)}:{_fmt_float_token(px)}:{_fmt_float_token(py)}"
                )

            try:
                curve_entities = gmsh.model.getEntities(1)
            except Exception as e:
                logger.warning("gmsh getEntities(1) failed: %s", e, exc_info=True)
                curve_entities = []
            curve_tags = sorted(
                int(tag)
                for dim, tag in list(curve_entities or [])
                if int(dim) == 1
            )
            for ctag in curve_tags:
                try:
                    boundary = gmsh.model.getBoundary(
                        [(1, int(ctag))],
                        combined=False,
                        oriented=True,
                        recursive=False,
                    )
                except Exception as e:
                    logger.warning("gmsh curve getBoundary failed for tag=%s: %s", ctag, e, exc_info=True)
                    boundary = []
                btags = [
                    int(t)
                    for d, t in list(boundary or [])
                    if int(d) == 0
                ]
                bb = _safe_bbox(1, int(ctag))
                bb_tok = ",".join(_fmt_float_token(v, digits=6) for v in bb)
                bnd_tok = ",".join(str(int(v)) for v in btags)
                curve_tokens.append(f"{int(ctag)}:{bnd_tok}:{bb_tok}")

            try:
                surface_entities = gmsh.model.getEntities(2)
            except Exception as e:
                logger.warning("gmsh getEntities(2) failed: %s", e, exc_info=True)
                surface_entities = []
            surface_tags_local = sorted(
                int(tag)
                for dim, tag in list(surface_entities or [])
                if int(dim) == 2
            )
            for stag in surface_tags_local:
                try:
                    boundary = gmsh.model.getBoundary(
                        [(2, int(stag))],
                        combined=False,
                        oriented=True,
                        recursive=False,
                    )
                except Exception as e:
                    logger.warning("gmsh surface getBoundary failed for tag=%s: %s", stag, e, exc_info=True)
                    boundary = []
                btags = [
                    int(t)
                    for d, t in list(boundary or [])
                    if int(d) == 1
                ]
                bb = _safe_bbox(2, int(stag))
                bb_tok = ",".join(_fmt_float_token(v, digits=6) for v in bb)
                bnd_tok = ",".join(str(int(v)) for v in btags)
                surface_tokens.append(f"{int(stag)}:{bnd_tok}:{bb_tok}")

            return point_tokens, curve_tokens, surface_tokens

        def _pre_generate_entity_signature_payload() -> Dict[str, object]:
            point_tokens, curve_tokens, surface_tokens = _entity_tokens_pre_generate()
            point_sha = _sha256_lines(point_tokens)
            curve_sha = _sha256_lines(curve_tokens)
            surface_sha = _sha256_lines(surface_tokens)
            all_sha = _sha256_lines([point_sha, curve_sha, surface_sha])
            return {
                "sha256": all_sha,
                "counts": {
                    "points": int(len(point_tokens)),
                    "curves": int(len(curve_tokens)),
                    "surfaces": int(len(surface_tokens)),
                },
                "point_sha256": point_sha,
                "curve_sha256": curve_sha,
                "surface_sha256": surface_sha,
                "point_preview": _preview_tokens(point_tokens, n=10),
                "curve_preview": _preview_tokens(curve_tokens, n=10),
                "surface_preview": _preview_tokens(surface_tokens, n=10),
            }

        def _compact_ptag_chain(ptags: Sequence[int]) -> List[int]:
            tags = [int(t) for t in ptags]
            if not tags:
                return []
            out = [tags[0]]
            for t in tags[1:]:
                if t != out[-1]:
                    out.append(t)
            return out

        def _register_quad_curve_candidate(curve_tag: int, edge_ptags: Sequence[int]) -> None:
            cabs = abs(int(curve_tag))
            if cabs <= 0:
                return
            chain = polycurve_chain_by_tag.get(cabs)
            if not chain:
                compact = _compact_ptag_chain(edge_ptags)
                if len(compact) < 2:
                    return
                chain = tuple(compact if int(curve_tag) > 0 else list(reversed(compact)))
            if len(chain) < 2:
                return
            if cabs in quad_curve_chain_by_abs:
                return
            quad_curve_chain_by_abs[cabs] = tuple(int(t) for t in chain)
            a = int(chain[0])
            b = int(chain[-1])
            quad_curve_candidates_by_endpoint.setdefault(a, []).append(cabs)
            if b != a:
                quad_curve_candidates_by_endpoint.setdefault(b, []).append(cabs)

        def _match_quad_curve_along_ring(
            ptags: Sequence[int],
            start_idx: int,
        ) -> Optional[Tuple[int, int, int, int]]:
            n = len(ptags)
            if n < 2:
                return None
            a = int(ptags[start_idx])
            match_tol = max(1.0e-6, 10.0 * float(tol))
            candidate_abs = list(quad_curve_candidates_by_endpoint.get(a, []))
            if not candidate_abs:
                a_xy = pt_xy_by_tag.get(a)
                if a_xy is not None:
                    for cabs, chain in quad_curve_chain_by_abs.items():
                        if len(chain) < 2:
                            continue
                        p0 = pt_xy_by_tag.get(int(chain[0]))
                        p1 = pt_xy_by_tag.get(int(chain[-1]))
                        if p0 is None or p1 is None:
                            continue
                        d0 = float(np.hypot(float(a_xy[0]) - float(p0[0]), float(a_xy[1]) - float(p0[1])))
                        d1 = float(np.hypot(float(a_xy[0]) - float(p1[0]), float(a_xy[1]) - float(p1[1])))
                        if d0 <= match_tol or d1 <= match_tol:
                            candidate_abs.append(int(cabs))
            if not candidate_abs:
                return None

            best: Optional[Tuple[int, int, float, int, int]] = None
            # (span_edges, signed_curve_tag, proj_err_sum, start_tag, end_tag)
            for cabs in candidate_abs:
                chain = quad_curve_chain_by_abs.get(int(cabs))
                if not chain or len(chain) < 2:
                    continue

                if a == int(chain[0]):
                    target = int(chain[-1])
                    oriented_chain = tuple(int(t) for t in chain)
                    signed = int(cabs)
                elif a == int(chain[-1]):
                    rev = tuple(reversed(chain))
                    oriented_chain = tuple(int(t) for t in rev)
                    target = int(oriented_chain[-1])
                    signed = -int(cabs)
                else:
                    a_xy = pt_xy_by_tag.get(a)
                    p0 = pt_xy_by_tag.get(int(chain[0]))
                    p1 = pt_xy_by_tag.get(int(chain[-1]))
                    if a_xy is None or p0 is None or p1 is None:
                        continue
                    d0 = float(np.hypot(float(a_xy[0]) - float(p0[0]), float(a_xy[1]) - float(p0[1])))
                    d1 = float(np.hypot(float(a_xy[0]) - float(p1[0]), float(a_xy[1]) - float(p1[1])))
                    if d0 <= d1 and d0 <= match_tol:
                        oriented_chain = tuple(int(t) for t in chain)
                        target = int(oriented_chain[-1])
                        signed = int(cabs)
                    elif d1 < d0 and d1 <= match_tol:
                        oriented_chain = tuple(int(t) for t in reversed(chain))
                        target = int(oriented_chain[-1])
                        signed = -int(cabs)
                    else:
                        continue

                idx_map = {int(t): i for i, t in enumerate(oriented_chain)}
                chain_xy: List[Tuple[float, float]] = []
                for t in oriented_chain:
                    xy = pt_xy_by_tag.get(int(t))
                    if xy is None:
                        chain_xy = []
                        break
                    chain_xy.append((float(xy[0]), float(xy[1])))
                if len(chain_xy) < 2:
                    continue
                chain_len = max(_mh()._polyline_length(chain_xy), 1.0e-12)
                target_xy = pt_xy_by_tag.get(int(target))

                prev_pos = 0.0
                span_edges = 0
                ok = True
                proj_err_sum = 0.0
                while span_edges < n:
                    p = int(ptags[(start_idx + span_edges + 1) % n])
                    span_edges += 1
                    ci = idx_map.get(p)
                    p_xy = pt_xy_by_tag.get(int(p))
                    if p_xy is None:
                        ok = False
                        break
                    if ci is not None:
                        pos = float(ci)
                        proj_err = 0.0
                    else:
                        proj_err, s_pos = _mh()._polyline_distance_and_s(chain_xy, float(p_xy[0]), float(p_xy[1]))
                        if (not np.isfinite(float(proj_err))) or float(proj_err) > match_tol:
                            ok = False
                            break
                        pos = (float(s_pos) / float(chain_len)) * float(max(1, len(oriented_chain) - 1))

                    if pos + 1.0e-8 < prev_pos:
                        ok = False
                        break
                    prev_pos = float(pos)
                    proj_err_sum += float(proj_err)

                    if p == target:
                        break
                    if target_xy is not None and np.hypot(float(p_xy[0]) - float(target_xy[0]), float(p_xy[1]) - float(target_xy[1])) <= match_tol:
                        break

                if not ok:
                    continue
                end_tag = int(ptags[(start_idx + span_edges) % n])
                end_ok = (end_tag == target)
                if not end_ok:
                    end_xy = pt_xy_by_tag.get(int(end_tag))
                    if end_xy is not None and target_xy is not None:
                        end_ok = np.hypot(float(end_xy[0]) - float(target_xy[0]), float(end_xy[1]) - float(target_xy[1])) <= match_tol
                if not end_ok:
                    continue
                if span_edges <= 0:
                    continue

                if best is None:
                    best = (
                        int(span_edges),
                        int(signed),
                        float(proj_err_sum),
                        int(oriented_chain[0]),
                        int(oriented_chain[-1]),
                    )
                else:
                    if int(span_edges) > int(best[0]) or (
                        int(span_edges) == int(best[0]) and float(proj_err_sum) < float(best[2])
                    ):
                        best = (
                            int(span_edges),
                            int(signed),
                            float(proj_err_sum),
                            int(oriented_chain[0]),
                            int(oriented_chain[-1]),
                        )

            if best is None:
                return None
            return int(best[0]), int(best[1]), int(best[3]), int(best[4])

        def _nearest_quad_endpoint_tag(x: float, y: float, snap_tol: float) -> Optional[int]:
            if float(snap_tol) <= 0.0:
                return None
            endpoint_tags = list(quad_curve_candidates_by_endpoint.keys())
            if not endpoint_tags:
                return None
            x0 = float(x)
            y0 = float(y)
            best_tag: Optional[int] = None
            best_d = float(snap_tol)
            for ptag in endpoint_tags:
                pxy = pt_xy_by_tag.get(int(ptag))
                if pxy is None:
                    continue
                d = float(np.hypot(x0 - float(pxy[0]), y0 - float(pxy[1])))
                if d <= best_d:
                    best_d = d
                    best_tag = int(ptag)
            return best_tag

        def _geo_pt(x: float, y: float, lc: float, *, endpoint_snap_tol: Optional[float] = None) -> int:
            """Return existing gmsh point tag at (x,y) or create a new one."""
            key = (round(float(x), _pt_prec), round(float(y), _pt_prec))
            if key in pt_reg:
                tag = int(pt_reg[key])
                pt_xy_by_tag.setdefault(tag, (float(x), float(y)))
                _record_build_event("geo-pt-reuse", key[0], key[1], int(tag))
                return tag

            snap_tol = float(endpoint_snap_tol) if endpoint_snap_tol is not None else 0.0
            if snap_tol > 0.0:
                snap_tag = _nearest_quad_endpoint_tag(float(x), float(y), float(snap_tol))
                if snap_tag is not None:
                    pt_reg[key] = int(snap_tag)
                    _record_build_event(
                        "geo-pt-snap",
                        key[0],
                        key[1],
                        int(snap_tag),
                        float(snap_tol),
                    )
                    return int(snap_tag)

            tag = gmsh.model.geo.addPoint(float(x), float(y), 0.0, lc)
            pt_reg[key] = tag
            pt_xy_by_tag[int(tag)] = (float(x), float(y))
            _record_build_event("geo-pt-new", int(tag), key[0], key[1], float(lc))
            return tag

        def _geo_seg(p0: int, p1: int) -> int:
            """Return signed line tag for directed segment p0->p1, sharing if it
            already exists in either direction."""
            if (p0, p1) in seg_reg:
                tag = int(seg_reg[(p0, p1)])
                polycurve_chain_by_tag.setdefault(abs(tag), (int(p0), int(p1)))
                _record_build_event("geo-seg-reuse-fwd", int(tag), int(p0), int(p1))
                return tag
            if (p1, p0) in seg_reg:
                tag = int(seg_reg[(p1, p0)])
                polycurve_chain_by_tag.setdefault(abs(tag), (int(p1), int(p0)))
                _record_build_event("geo-seg-reuse-rev", int(tag), int(p0), int(p1))
                return -tag
            tag = gmsh.model.geo.addLine(p0, p1)
            seg_reg[(p0, p1)] = tag
            polycurve_chain_by_tag[int(tag)] = (int(p0), int(p1))
            _record_build_event("geo-seg-new", int(tag), int(p0), int(p1))
            return int(tag)

        def _geo_polycurve(ptags: Sequence[int]) -> int:
            """Return a shared directed curve for a polyline point-tag sequence.

            Reuses existing spline/line entities in forward or reversed direction
            so neighboring regions can share exact same interface entities.
            """
            compact = _compact_ptag_chain(ptags)
            if len(compact) < 2:
                raise ValueError("polycurve requires at least two points")

            if len(compact) == 2:
                return _geo_seg(compact[0], compact[1])

            fwd = tuple(compact)
            rev = tuple(reversed(compact))
            if fwd in polycurve_reg:
                tag_reuse = int(polycurve_reg[fwd])
                _record_build_event(
                    "geo-polycurve-reuse-fwd",
                    int(tag_reuse),
                    int(len(fwd)),
                    int(fwd[0]),
                    int(fwd[-1]),
                )
                return int(tag_reuse)
            if rev in polycurve_reg:
                tag_reuse = int(polycurve_reg[rev])
                _record_build_event(
                    "geo-polycurve-reuse-rev",
                    int(tag_reuse),
                    int(len(rev)),
                    int(rev[0]),
                    int(rev[-1]),
                )
                return -int(tag_reuse)

            tag = gmsh.model.geo.addSpline(list(compact))
            polycurve_reg[fwd] = int(tag)
            polycurve_chain_by_tag[int(tag)] = tuple(compact)
            _record_build_event(
                "geo-polycurve-new",
                int(tag),
                int(len(compact)),
                int(compact[0]),
                int(compact[-1]),
                _hash_int_sequence(compact),
            )
            return int(tag)

        prebuild_subphase_started_at = time.perf_counter()

        region_cell_types: Dict[int, str] = {
            int(r.region_id): str(r.default_cell_type).strip().lower()
            for r in model.regions
        }
        region_quad_setups: Dict[int, Tuple[List[Tuple[float, float]], List[QuadEdgeControl]]] = {}
        for region in model.regions:
            ctype_local = str(region.default_cell_type).strip().lower()
            if ctype_local not in {"quadrilateral", "cartesian", "channel_generator"}:
                continue
            quad_setup_local = _mh()._quad_controls_for_region(model, region)
            if quad_setup_local is None:
                continue
            region_quad_setups[int(region.region_id)] = quad_setup_local
        _record_phase("prebuild_region_quad_setup", prebuild_subphase_started_at)

        prebuild_subphase_started_at = time.perf_counter()
        region_rings_for_junctions: Dict[int, List[Tuple[float, float]]] = {}
        for region in model.regions:
            rid_local = int(region.region_id)
            setup_local = region_quad_setups.get(rid_local)
            ring_local = list(setup_local[0]) if setup_local is not None else list(region.ring_xy)
            if ring_local and np.hypot(float(ring_local[0][0]) - float(ring_local[-1][0]), float(ring_local[0][1]) - float(ring_local[-1][1])) <= 1.0e-12:
                ring_local = ring_local[:-1]
            if len(ring_local) >= 3:
                region_rings_for_junctions[rid_local] = [(float(x), float(y)) for (x, y) in ring_local]

        transfinite_edge_min_nodes: Dict[Tuple[int, int], int] = {}
        transfinite_harmonize_stats: Dict[str, int] = {
            "shared_groups": 0,
            "canonicalized_edges": 0,
            "opposite_subset_requests": 0,
            "junction_points_inserted": 0,
            "subset_containment_requests": 0,
            "singleton_external_junction_edges": 0,
            "candidate_pair_count_prefilter": 0,
            "candidate_pair_count": 0,
            "pair_bbox_reject_count": 0,
            "nontrans_chain_bbox_reject_count": 0,
            "nontrans_overlap_pair_count": 0,
            "nontrans_point_bbox_reject_count": 0,
        }
        transfinite_harmonize_debug: Dict[str, object] = {}
        if gmsh_transfinite_shared_interface_harmonize and region_quad_setups:
            transfinite_edge_min_nodes, transfinite_harmonize_stats = _mh()._harmonize_transfinite_shared_quad_interfaces(
                region_quad_setups=region_quad_setups,
                region_cell_types=region_cell_types,
                gmsh_quad_full_region_flow_align=bool(gmsh_quad_full_region_flow_align),
                all_region_rings=region_rings_for_junctions,
                opposite_subset_start_frac=float(gmsh_transfinite_opposite_subset_start),
                opposite_subset_end_frac=float(gmsh_transfinite_opposite_subset_end),
                opposite_subset_density_scale=float(gmsh_transfinite_opposite_subset_density_scale),
                subset_containment_enable=bool(gmsh_transfinite_subset_containment_enable),
                subset_containment_high_overlap=float(gmsh_transfinite_subset_containment_high_overlap),
                subset_containment_min_overlap=float(gmsh_transfinite_subset_containment_min_overlap),
                subset_containment_max_length_ratio=float(gmsh_transfinite_subset_containment_max_length_ratio),
                debug_capture=transfinite_harmonize_debug if gmsh_transfinite_interface_debug else None,
            )
        _record_phase("prebuild_transfinite_harmonize", prebuild_subphase_started_at)

        def _is_transfinite_region_local(region_id: int) -> bool:
            ctype_local = str(region_cell_types.get(int(region_id), "")).strip().lower()
            if ctype_local == "cartesian":
                return True
            if gmsh_quad_full_region_flow_align and ctype_local in {"quadrilateral", "channel_generator"}:
                return True
            return False

        # Project/split non-transfinite neighboring rings against transfinite
        # interface chains so mixed interfaces can reuse shared geometry.
        prebuild_subphase_started_at = time.perf_counter()
        nontrans_neighbor_projection_rings = 0
        nontrans_chain_bbox_reject_count = 0
        nontrans_overlap_pair_count = 0
        nontrans_point_bbox_reject_count = 0
        transfinite_interface_chains: List[
            Tuple[
                int,
                int,
                List[Tuple[float, float]],
                float,
                Tuple[float, float, float, float],
            ]
        ] = []
        for rid_tf, (_ring_tf, controls_tf) in region_quad_setups.items():
            if not _is_transfinite_region_local(int(rid_tf)):
                continue
            for edge_tf in list(controls_tf or []):
                pts_tf = [(float(x), float(y)) for (x, y) in list(edge_tf.points_xy or [])]
                if len(pts_tf) < 2:
                    continue
                if edge_tf.target_size is not None and float(edge_tf.target_size) > 0.0:
                    size_ref_tf = float(edge_tf.target_size)
                else:
                    size_ref_tf = max(_mh()._polyline_length(pts_tf) / max(1, len(pts_tf) - 1), 1.0e-6)
                tx = [float(p[0]) for p in pts_tf]
                ty = [float(p[1]) for p in pts_tf]
                chain_bbox = (float(min(tx)), float(min(ty)), float(max(tx)), float(max(ty)))
                transfinite_interface_chains.append(
                    (
                        int(rid_tf),
                        int(edge_tf.edge_id),
                        pts_tf,
                        float(max(size_ref_tf, 1.0e-6)),
                        chain_bbox,
                    )
                )

        if transfinite_interface_chains:
            for region in model.regions:
                rid_nt = int(region.region_id)
                if _is_transfinite_region_local(rid_nt):
                    continue

                ring_nt = [(float(x), float(y)) for (x, y) in list(region.ring_xy or [])]
                if ring_nt and np.hypot(
                    float(ring_nt[0][0]) - float(ring_nt[-1][0]),
                    float(ring_nt[0][1]) - float(ring_nt[-1][1]),
                ) <= 1.0e-12:
                    ring_nt = ring_nt[:-1]
                if len(ring_nt) < 3:
                    continue

                ring_changed = False
                rx = [float(p[0]) for p in ring_nt]
                ry = [float(p[1]) for p in ring_nt]
                ring_bbox = (float(min(rx)), float(min(ry)), float(max(rx)), float(max(ry)))

                for _owner_tf, _eid_tf, chain_tf, size_ref_tf, chain_bbox in transfinite_interface_chains:
                    size_ref = max(min(float(max(region.default_size, 1.0e-6)), float(size_ref_tf)), 1.0e-6)
                    near_tol = max(1.0e-6, min(8.0, max(0.5, 0.25 * float(size_ref))))
                    sample_step = max(float(near_tol), 0.25 * float(size_ref))

                    if (
                        float(chain_bbox[2]) < float(ring_bbox[0]) - float(near_tol)
                        or float(ring_bbox[2]) < float(chain_bbox[0]) - float(near_tol)
                        or float(chain_bbox[3]) < float(ring_bbox[1]) - float(near_tol)
                        or float(ring_bbox[3]) < float(chain_bbox[1]) - float(near_tol)
                    ):
                        nontrans_chain_bbox_reject_count += 1
                        continue

                    ring_open = list(ring_nt)
                    if ring_open:
                        ring_open.append((float(ring_open[0][0]), float(ring_open[0][1])))

                    nontrans_overlap_pair_count += 1
                    overlap_tf_nt, overlap_nt_tf = _mh()._polyline_overlap_fractions_open(
                        chain_tf,
                        ring_open,
                        sample_step=float(sample_step),
                        near_tol=float(near_tol),
                    )
                    if max(float(overlap_tf_nt), float(overlap_nt_tf)) < 0.01:
                        continue

                    ring_split = _mh()._insert_focus_points_on_ring_segments(
                        ring=ring_nt,
                        focus_points=chain_tf,
                        max_dist=float(near_tol),
                    )
                    if len(ring_split) < 3:
                        continue

                    chain_len = _mh()._polyline_length(chain_tf)
                    if chain_len <= 1.0e-12:
                        continue

                    ring_proj: List[Tuple[float, float]] = []
                    moved_any = False
                    native = _mh()._load_hydra_meshing_native()
                    if native is not None and hasattr(native, "project_ring_to_chain"):
                        try:
                            proj_out = native.project_ring_to_chain(
                                ring_split,
                                chain_tf,
                                float(near_tol),
                            )
                            ring_proj = [
                                (float(p[0]), float(p[1]))
                                for p in list(proj_out.get("ring_proj", []))
                                if isinstance(p, (list, tuple)) and len(p) >= 2
                            ]
                            moved_any = bool(proj_out.get("moved_any", False))
                            nontrans_point_bbox_reject_count += int(proj_out.get("point_bbox_reject_count", 0))
                        except Exception as e:
                            logger.warning("nontrans neighbor projection failed: %s", e, exc_info=True)
                            ring_proj = []

                    if not ring_proj:
                        chain_bbox_exp = (
                            float(chain_bbox[0]) - float(near_tol),
                            float(chain_bbox[1]) - float(near_tol),
                            float(chain_bbox[2]) + float(near_tol),
                            float(chain_bbox[3]) + float(near_tol),
                        )
                        for px, py in ring_split:
                            if (
                                float(px) < float(chain_bbox_exp[0])
                                or float(px) > float(chain_bbox_exp[2])
                                or float(py) < float(chain_bbox_exp[1])
                                or float(py) > float(chain_bbox_exp[3])
                            ):
                                nontrans_point_bbox_reject_count += 1
                                ring_proj.append((float(px), float(py)))
                                continue
                            d_loc, s_loc = _mh()._polyline_distance_and_s(chain_tf, float(px), float(py))
                            if np.isfinite(float(d_loc)) and float(d_loc) <= float(near_tol):
                                frac_loc = max(0.0, min(1.0, float(s_loc) / float(chain_len)))
                                qx, qy = _mh()._interp_polyline_fraction(chain_tf, frac_loc)
                                ring_proj.append((float(qx), float(qy)))
                                if float(np.hypot(float(qx) - float(px), float(qy) - float(py))) > 1.0e-10:
                                    moved_any = True
                            else:
                                ring_proj.append((float(px), float(py)))

                    if len(ring_proj) < 3:
                        continue
                    ring_clean: List[Tuple[float, float]] = []
                    for px, py in ring_proj:
                        if ring_clean and np.hypot(float(px) - float(ring_clean[-1][0]), float(py) - float(ring_clean[-1][1])) <= 1.0e-12:
                            continue
                        ring_clean.append((float(px), float(py)))
                    while len(ring_clean) >= 2 and np.hypot(
                        float(ring_clean[0][0]) - float(ring_clean[-1][0]),
                        float(ring_clean[0][1]) - float(ring_clean[-1][1]),
                    ) <= 1.0e-12:
                        ring_clean.pop()
                    if len(ring_clean) < 3:
                        continue

                    if moved_any or len(ring_clean) > len(ring_nt):
                        ring_nt = ring_clean
                        rx = [float(p[0]) for p in ring_nt]
                        ry = [float(p[1]) for p in ring_nt]
                        ring_bbox = (float(min(rx)), float(min(ry)), float(max(rx)), float(max(ry)))
                        ring_changed = True

                if ring_changed:
                    region.ring_xy = [(float(x), float(y)) for (x, y) in ring_nt]
                    region_rings_for_junctions[rid_nt] = [(float(x), float(y)) for (x, y) in ring_nt]
                    nontrans_neighbor_projection_rings += 1
                    transfinite_harmonize_stats["nontrans_neighbor_projection_rings"] = int(
                        transfinite_harmonize_stats.get("nontrans_neighbor_projection_rings", 0)
                    ) + 1
        transfinite_harmonize_stats["nontrans_chain_bbox_reject_count"] = int(
            transfinite_harmonize_stats.get("nontrans_chain_bbox_reject_count", 0)
        ) + int(nontrans_chain_bbox_reject_count)
        transfinite_harmonize_stats["nontrans_overlap_pair_count"] = int(
            transfinite_harmonize_stats.get("nontrans_overlap_pair_count", 0)
        ) + int(nontrans_overlap_pair_count)
        transfinite_harmonize_stats["nontrans_point_bbox_reject_count"] = int(
            transfinite_harmonize_stats.get("nontrans_point_bbox_reject_count", 0)
        ) + int(nontrans_point_bbox_reject_count)
        _record_phase("prebuild_nontrans_projection", prebuild_subphase_started_at)

        prebuild_subphase_started_at = time.perf_counter()
        interface_coincidence_report: List[Dict[str, object]] = []
        interface_coincidence_suspects: List[Dict[str, object]] = []
        try:
            interface_coincidence_report = _mh()._gmsh_interface_coincidence_report(
                model,
                region_quad_setups=region_quad_setups,
            )
            for entry in interface_coincidence_report:
                overlap_delta = float(entry.get("overlap_delta", 0.0))
                endpoint_delta_max = float(entry.get("endpoint_delta_max", float("inf")))
                near_tol = max(float(entry.get("near_tol", 1.0e-6)), 1.0e-9)
                if overlap_delta > 0.20 or endpoint_delta_max > 2.0 * near_tol:
                    interface_coincidence_suspects.append(dict(entry))
            if interface_coincidence_suspects:
                preview = "; ".join(
                    (
                        f"{int(e.get('region_a', -1))}-{int(e.get('region_b', -1))} "
                        f"(overlap_delta={float(e.get('overlap_delta', 0.0)):.3f}, "
                        f"endpoint_delta_max={float(e.get('endpoint_delta_max', float('inf'))):.4g})"
                    )
                    for e in interface_coincidence_suspects[:6]
                )
                warnings.warn(
                    "Gmsh interface coincidence preflight flagged potential geometry mismatches: "
                    + preview,
                    RuntimeWarning,
                )
        except Exception as e:
            logger.warning("gmsh interface coincidence preflight failed: %s", e, exc_info=True)
            interface_coincidence_report = []
            interface_coincidence_suspects = []
        _record_phase("prebuild_interface_coincidence", prebuild_subphase_started_at)

        _record_phase("prebuild_setup", build_started_at)
        _mark_build_stage("after-prebuild-setup")

        # ---- 1. Build one Gmsh surface per region ----------------------
        phase_started_at = time.perf_counter()
        def _region_priority(r: ConceptualRegion) -> int:
            c = str(r.default_cell_type).strip().lower()
            return 0 if c in {"quadrilateral", "cartesian", "channel_generator"} else 1

        for region in sorted(model.regions, key=_region_priority):
            ring = list(region.ring_xy)
            if ring and ring[0] == ring[-1]:
                ring = ring[:-1]
            if len(ring) < 3:
                continue

            ctype = str(region.default_cell_type).strip().lower()
            if ctype == "empty":
                continue
            region_size = max(float(region.default_size) * float(size_scale), 1.0e-9)

            quad_controls = None
            if ctype in ("quadrilateral", "cartesian", "channel_generator"):
                quad_setup = region_quad_setups.get(int(region.region_id))
                if quad_setup is not None:
                    ring, quad_controls = quad_setup

            lines: List[int] = []
            edge_curve_groups: List[List[int]] = []
            if quad_controls is not None:
                first_pt_tag: Optional[int] = None
                first_xy: Optional[Tuple[float, float]] = None
                prev_end_tag: Optional[int] = None
                # Use a slightly looser closure tolerance than point-dedup tol
                # to prevent tiny residual seam segments on assembled quad
                # rings, which can break transfinite opposite-side matching.
                closure_snap_tol = max(float(tol), min(1.0e-3, 0.01 * float(region_size)))
                for ei, edge in enumerate(quad_controls):
                    edge_pts = list(edge.points_xy)
                    if len(edge_pts) < 2:
                        continue
                    edge_lc = float(edge.target_size) * float(size_scale) if (edge.target_size is not None and edge.target_size > 0.0) else float(region_size)
                    edge_tags: List[int] = []
                    for pj, (x, y) in enumerate(edge_pts):
                        if ei > 0 and pj == 0 and prev_end_tag is not None:
                            edge_tags.append(prev_end_tag)
                            continue
                        if ei == len(quad_controls) - 1 and pj == len(edge_pts) - 1 and first_pt_tag is not None and first_xy is not None:
                            if np.hypot(x - first_xy[0], y - first_xy[1]) <= float(closure_snap_tol):
                                edge_tags.append(first_pt_tag)
                                continue
                        ptag = _geo_pt(x, y, edge_lc)
                        edge_tags.append(ptag)
                        if first_pt_tag is None:
                            first_pt_tag = ptag
                            first_xy = (float(x), float(y))
                    if len(edge_tags) < 2:
                        continue
                    try:
                        # Build quad interfaces as shared segment chains so
                        # neighboring non-transfinite regions can reuse
                        # interior subsets of the same interface geometry.
                        edge_curves: List[int] = []
                        for k in range(len(edge_tags) - 1):
                            seg = int(_geo_seg(edge_tags[k], edge_tags[k + 1]))
                            edge_curves.append(int(seg))
                            lines.append(int(seg))
                            _register_quad_curve_candidate(int(seg), [edge_tags[k], edge_tags[k + 1]])
                        if edge_curves:
                            edge_curve_groups.append(edge_curves)
                    except Exception as e:
                        logger.warning("quad edge curve build failed: %s", e, exc_info=True)
                        edge_curve_groups.append([])
                        for k in range(len(edge_tags) - 1):
                            lines.append(_geo_seg(edge_tags[k], edge_tags[k + 1]))
                    prev_end_tag = edge_tags[-1]
                if first_pt_tag is not None and prev_end_tag is not None and prev_end_tag != first_pt_tag:
                    p_prev = pt_xy_by_tag.get(int(prev_end_tag))
                    p_first = pt_xy_by_tag.get(int(first_pt_tag))
                    can_snap_close = False
                    if p_prev is not None and p_first is not None:
                        can_snap_close = bool(
                            np.hypot(
                                float(p_prev[0]) - float(p_first[0]),
                                float(p_prev[1]) - float(p_first[1]),
                            ) <= float(closure_snap_tol)
                        )
                    if can_snap_close:
                        prev_end_tag = first_pt_tag
                    else:
                        closing_seg = _geo_seg(prev_end_tag, first_pt_tag)
                        lines.append(closing_seg)
                        if edge_curve_groups:
                            edge_curve_groups[-1].append(int(closing_seg))
            else:
                # Canonicalize near-coincident interface junction points onto
                # previously built quad endpoints to avoid duplicate corner
                # entities across mixed transfinite/non-transfinite neighbors.
                junction_snap_tol = max(20.0 * float(tol), min(1.0e-3, 0.01 * float(region_size)))
                pts = [
                    _geo_pt(x, y, region_size, endpoint_snap_tol=float(junction_snap_tol))
                    for x, y in ring
                ]

                pts_compact: List[int] = []
                for ptag in pts:
                    if not pts_compact or int(ptag) != int(pts_compact[-1]):
                        pts_compact.append(int(ptag))
                while len(pts_compact) >= 2 and int(pts_compact[0]) == int(pts_compact[-1]):
                    pts_compact.pop()
                if len(pts_compact) >= 3:
                    pts = [int(t) for t in pts_compact]

                n_pts = len(pts)
                i = 0
                consumed_edges = 0
                while consumed_edges < n_pts:
                    match = _match_quad_curve_along_ring(pts, i)
                    if match is not None:
                        span, signed_curve, start_tag, end_tag = match
                        if span > 0:
                            # Force exact endpoint tag reuse at matched curve
                            # boundaries so subsequent ring segments stay
                            # topologically connected to the shared chain.
                            if int(pts[i]) != int(start_tag):
                                pts[i] = int(start_tag)
                            end_idx = (i + int(span)) % n_pts
                            if int(pts[end_idx]) != int(end_tag):
                                pts[end_idx] = int(end_tag)
                            lines.append(int(signed_curve))
                            i = int(end_idx)
                            consumed_edges += int(span)
                            continue
                    lines.append(_geo_seg(pts[i], pts[(i + 1) % n_pts]))
                    i = (i + 1) % n_pts
                    consumed_edges += 1

            if len(lines) < 3:
                continue

            loop = gmsh.model.geo.addCurveLoop(lines)
            _record_build_event(
                "geo-loop-new",
                int(loop),
                int(len(lines)),
                _hash_int_sequence(lines),
            )
            hole_loops: List[int] = []
            exclusion_zones = _mh()._region_exclusion_zones(model, region, ring)
            if exclusion_zones:
                outer_area = _polygon_area_xy(
                    np.asarray([p[0] for p in ring], dtype=np.float64),
                    np.asarray([p[1] for p in ring], dtype=np.float64),
                )
                outer_ccw = bool(outer_area > 0.0)
                for ering, esize in exclusion_zones:
                    hring = list(ering)
                    if hring and hring[0] == hring[-1]:
                        hring = hring[:-1]
                    if len(hring) < 3:
                        continue

                    h_area = _polygon_area_xy(
                        np.asarray([p[0] for p in hring], dtype=np.float64),
                        np.asarray([p[1] for p in hring], dtype=np.float64),
                    )
                    if bool(h_area > 0.0) == outer_ccw:
                        hring = list(reversed(hring))

                    hole_size = max(float(esize) * float(size_scale), 1.0e-9)
                    hole_pts = [_geo_pt(x, y, hole_size) for x, y in hring]
                    if len(hole_pts) < 3:
                        continue
                    hlines: List[int] = []
                    for i in range(len(hole_pts)):
                        hlines.append(_geo_seg(hole_pts[i], hole_pts[(i + 1) % len(hole_pts)]))
                    if len(hlines) < 3:
                        continue
                    try:
                        hole_loop = gmsh.model.geo.addCurveLoop(hlines)
                        hole_loops.append(hole_loop)
                        _record_build_event(
                            "geo-hole-loop-new",
                            int(hole_loop),
                            int(len(hlines)),
                            _hash_int_sequence(hlines),
                        )
                    except Exception as e:
                        logger.warning("gmsh addCurveLoop for hole failed: %s", e, exc_info=True)
                        pass

            surf = gmsh.model.geo.addPlaneSurface([loop] + hole_loops)
            _record_build_event(
                "geo-surface-new",
                int(surf),
                int(region.region_id),
                str(ctype),
                int(loop),
                int(len(hole_loops)),
                int(len(lines)),
            )
            surface_tags.append(surf)
            surface_meta.append((region.region_id, ctype, region_size))
            surface_curve_tags[surf] = lines
            surface_quad_controls[surf] = quad_controls
            surface_quad_edge_curve_groups[surf] = edge_curve_groups if quad_controls is not None else None

        if not surface_tags:
            raise ValueError("GmshBackend: no non-empty regions to mesh.")
        _record_phase("build_surfaces", phase_started_at)
        _mark_build_stage("after-build-surfaces")

        # ---- 2. Embed arc breaklines into surfaces ----------------------
        phase_started_at = time.perf_counter()
        arc_soft_groups: Dict[Tuple[float, float], Dict[str, List[int]]] = {}
        if model.arcs and arc_mode != "disabled":
            arc_hard_curve_tags: List[int] = []
            # Build a quick node-id -> (x,y) lookup
            node_xy = {n.node_id: (n.x, n.y) for n in model.nodes}
            arc_lc = min(
                (
                    max(float(r.default_size) * float(size_scale), 1.0e-9)
                    for r in model.regions
                    if str(r.default_cell_type).strip().lower() != "empty"
                ),
                default=1.0,
            )

            channel_region_ids = {
                int(r.region_id)
                for r in model.regions
                if str(r.default_cell_type).strip().lower() == "channel_generator"
            }

            def _arc_mode_for(arc: ConceptualArc) -> str:
                mode_local = str(getattr(arc, "arc_mode_override", "") or "").strip().lower()
                if mode_local in {"hard_embed", "soft_size_hint", "disabled"}:
                    return mode_local

                role_local = str(getattr(arc, "arc_role", "") or "").strip().lower()
                in_channel_region = int(getattr(arc, "region_id", -1)) in channel_region_ids
                if in_channel_region and role_local in {"left_bank", "right_bank"}:
                    return "hard_embed"
                if in_channel_region and role_local == "centerline":
                    return "soft_size_hint"

                if bool(getattr(arc, "use_global_arc_ctrl", True)):
                    return arc_mode
                return arc_mode

            def _arc_soft_size_factor_for(arc: ConceptualArc) -> float:
                if bool(getattr(arc, "use_global_arc_ctrl", True)):
                    return float(arc_soft_size_factor)
                cand = getattr(arc, "arc_soft_size_override", None)
                if cand is None:
                    return float(arc_soft_size_factor)
                return min(1.0, max(0.05, float(cand)))

            def _arc_soft_dist_factor_for(arc: ConceptualArc) -> float:
                if bool(getattr(arc, "use_global_arc_ctrl", True)):
                    return float(arc_soft_dist_factor)
                cand = getattr(arc, "arc_soft_dist_override", None)
                if cand is None:
                    return float(arc_soft_dist_factor)
                return max(0.1, float(cand))

            for arc in model.arcs:
                pts_xy = list(arc.points_xy or [])
                arc_point_tags_local: List[int] = []
                arc_curve_tags_local: List[int] = []
                if len(pts_xy) >= 2:
                    gp_tags: List[int] = []
                    for x, y in pts_xy:
                        ptag = _geo_pt(float(x), float(y), arc_lc)
                        if not gp_tags or gp_tags[-1] != ptag:
                            gp_tags.append(ptag)
                    arc_point_tags_local.extend(gp_tags)
                    for i in range(len(gp_tags) - 1):
                        seg = _geo_seg(gp_tags[i], gp_tags[i + 1])
                        seg_abs = abs(int(seg))
                        arc_curve_tags_local.append(seg_abs)
                else:
                    # Backward-compatible fallback: endpoint IDs in topo_nodes.
                    p0_xy = node_xy.get(arc.node0)
                    p1_xy = node_xy.get(arc.node1)
                    if p0_xy is None or p1_xy is None:
                        continue
                    gp0 = _geo_pt(p0_xy[0], p0_xy[1], arc_lc)
                    gp1 = _geo_pt(p1_xy[0], p1_xy[1], arc_lc)
                    arc_point_tags_local.extend([gp0, gp1])
                    arc_curve_tags_local.append(abs(int(_geo_seg(gp0, gp1))))

                mode_local = _arc_mode_for(arc)
                if mode_local == "hard_embed":
                    arc_hard_curve_tags.extend(arc_curve_tags_local)
                elif mode_local == "soft_size_hint":
                    size_factor_local = _arc_soft_size_factor_for(arc)
                    dist_factor_local = _arc_soft_dist_factor_for(arc)
                    key = (round(float(size_factor_local), 6), round(float(dist_factor_local), 6))
                    group = arc_soft_groups.setdefault(key, {"curves": [], "points": []})
                    group["curves"].extend(arc_curve_tags_local)
                    group["points"].extend(arc_point_tags_local)

            if arc_hard_curve_tags:
                arc_curve_tags = sorted({int(tag) for tag in arc_hard_curve_tags if int(tag) > 0})
                _record_build_event(
                    "geo-sync",
                    "arc-hard-embed-start",
                    int(len(arc_curve_tags)),
                    _hash_int_sequence(arc_curve_tags),
                )
                gmsh.model.geo.synchronize()
                for surf in surface_tags:
                    try:
                        gmsh.model.mesh.embed(1, arc_curve_tags, 2, surf)
                        _record_build_event(
                            "mesh-embed-curves",
                            int(surf),
                            int(len(arc_curve_tags)),
                            _hash_int_sequence(arc_curve_tags),
                        )
                    except Exception as e:
                        logger.warning("gmsh mesh.embed curves failed for surf=%s: %s", surf, e, exc_info=True)
                        pass  # arc may not intersect this surface; skip

        _record_build_event("geo-sync", "post-arc-and-surfaces")
        gmsh.model.geo.synchronize()

        surface_size_map: Dict[int, float] = {int(s): float(sz) for s, (_, _, sz) in zip(surface_tags, surface_meta)}
        surface_ctype_map: Dict[int, str] = {int(s): str(ct) for s, (_, ct, _) in zip(surface_tags, surface_meta)}
        protected_transfinite_surfaces: set = set()
        for surf in surface_tags:
            s = int(surf)
            ctype = str(surface_ctype_map.get(s, "")).strip().lower()
            if ctype == "cartesian":
                protected_transfinite_surfaces.add(s)
            elif gmsh_quad_full_region_flow_align and ctype in {"quadrilateral", "channel_generator"}:
                protected_transfinite_surfaces.add(s)

        interface_transition_specs: List[Dict[str, object]] = []
        if gmsh_interface_transition_enable:
            curve_to_surfaces: Dict[int, List[int]] = {}
            for surf, lines in surface_curve_tags.items():
                s = int(surf)
                for ltag in lines:
                    cabs = abs(int(ltag))
                    if cabs <= 0:
                        continue
                    curve_to_surfaces.setdefault(cabs, []).append(s)

            for cabs, owners in curve_to_surfaces.items():
                uniq = sorted(set(int(v) for v in owners))
                if len(uniq) < 2:
                    continue

                sizes = [float(surface_size_map.get(s, 0.0)) for s in uniq if float(surface_size_map.get(s, 0.0)) > 0.0]
                if len(sizes) < 2:
                    continue
                smin = max(min(sizes), 1.0e-9)
                smax = max(sizes)
                if smax < float(gmsh_interface_transition_min_ratio) * smin:
                    continue

                target_surfaces = [int(s) for s in uniq if int(s) not in protected_transfinite_surfaces]
                if not target_surfaces:
                    continue

                interface_transition_specs.append(
                    {
                        "curve_tag": int(cabs),
                        "owner_surfaces": [int(s) for s in uniq],
                        "target_surfaces": [int(s) for s in target_surfaces],
                        "size_min": float(smin),
                        "size_max": float(smax),
                    }
                )
        _record_phase("embed_arcs_and_interfaces", phase_started_at)
        _mark_build_stage("after-embed-arcs-and-interfaces")

        # ---- 3. Constraint refinement zones (background field) ----------
        phase_started_at = time.perf_counter()
        # Build a region baseline size field and overlay per-constraint
        # threshold fields derived from polygon-clipped point sampling.
        # This is stronger than pure point embedding and enforces local sizing.
        base_surface_fields: List[int] = []
        for surf, (_, _, sz) in zip(surface_tags, surface_meta):
            f_const = gmsh.model.mesh.field.add("MathEval")
            gmsh.model.mesh.field.setString(f_const, "F", f"{max(float(sz), 1.0e-9):.16g}")
            f_restrict = gmsh.model.mesh.field.add("Restrict")
            gmsh.model.mesh.field.setNumber(f_restrict, "InField", float(f_const))
            gmsh.model.mesh.field.setNumbers(f_restrict, "SurfacesList", [int(surf)])
            base_surface_fields.append(f_restrict)

        constraint_point_lists: List[List[int]] = []
        constraint_target_sizes: List[float] = []
        for cst in model.constraints:
            if len(cst.ring_xy) < 3 or str(cst.cell_type).strip().lower() == "empty":
                continue
            ring = list(cst.ring_xy)
            if ring[0] == ring[-1]:
                ring = ring[:-1]
            if len(ring) < 3:
                continue

            pt_tags: List[int] = []
            cst_size = max(float(cst.target_size) * float(size_scale), 1.0e-9)

            # Boundary samples.
            for x, y in ring:
                try:
                    pt_tags.append(gmsh.model.geo.addPoint(float(x), float(y), 0.0, cst_size))
                except Exception as e:
                    logger.warning("gmsh addPoint failed for constraint boundary pt: %s", e, exc_info=True)
                    pass

            # Interior samples clipped to the polygon footprint.
            #
            # Important: avoid one-sided sampling truncation. The previous
            # implementation stopped after a fixed point cap while scanning
            # ymin->ymax, which could leave only part of a large constraint
            # polygon refined. Here we choose an area-adaptive step so sampling
            # remains approximately bounded while covering the full polygon.
            xs = [p[0] for p in ring]
            ys = [p[1] for p in ring]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)

            base_step = max(cst_size, tol * 10.0)
            target_pts = 6000.0
            poly_area = abs(_polygon_area_xy(
                np.asarray(xs, dtype=np.float64),
                np.asarray(ys, dtype=np.float64),
            ))
            if poly_area > 0.0:
                step = max(base_step, float(np.sqrt(poly_area / target_pts)))
            else:
                step = base_step

            y = ymin + 0.5 * step
            while y < ymax - 0.5 * step:
                x = xmin + 0.5 * step
                while x < xmax - 0.5 * step:
                    if _mh()._point_in_polygon(x, y, ring):
                        try:
                            pt_tags.append(gmsh.model.geo.addPoint(float(x), float(y), 0.0, cst_size))
                        except Exception as e:
                            logger.warning("gmsh addPoint failed for constraint interior pt: %s", e, exc_info=True)
                            pass
                    x += step
                y += step

            dedup_tags = list(dict.fromkeys(pt_tags))
            if dedup_tags:
                constraint_point_lists.append(dedup_tags)
                constraint_target_sizes.append(cst_size)

        gmsh.model.geo.synchronize()

        interface_transition_field_count = 0
        if constraint_point_lists or arc_soft_groups or interface_transition_specs:
            all_fields: List[int] = list(base_surface_fields)
            max_region_size = max(max(float(sz), 1.0e-9) for (_, _, sz) in surface_meta)
            for pt_list, cst_size in zip(constraint_point_lists, constraint_target_sizes):
                f_dist = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(f_dist, "PointsList", [int(t) for t in pt_list])

                f_thresh = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(f_thresh, "InField", float(f_dist))
                gmsh.model.mesh.field.setNumber(f_thresh, "SizeMin", float(cst_size))
                gmsh.model.mesh.field.setNumber(f_thresh, "SizeMax", float(max_region_size))
                gmsh.model.mesh.field.setNumber(f_thresh, "DistMin", 0.0)
                gmsh.model.mesh.field.setNumber(f_thresh, "DistMax", float(1.5 * cst_size))
                gmsh.model.mesh.field.setNumber(f_thresh, "StopAtDistMax", 1.0)

                f_restrict = gmsh.model.mesh.field.add("Restrict")
                gmsh.model.mesh.field.setNumber(f_restrict, "InField", float(f_thresh))
                gmsh.model.mesh.field.setNumbers(f_restrict, "SurfacesList", [int(s) for s in surface_tags])
                all_fields.append(f_restrict)

            if arc_soft_groups:
                min_region_size = max(min(float(sz) for (_, _, sz) in surface_meta), 1.0e-9)
                for (size_factor_local, dist_factor_local), group in arc_soft_groups.items():
                    arc_curves = sorted({int(t) for t in group.get("curves", []) if int(t) > 0})
                    arc_pts = sorted({int(t) for t in group.get("points", []) if int(t) > 0})
                    if not arc_curves and not arc_pts:
                        continue

                    arc_size = max(mesh_size_min, min_region_size * float(size_factor_local))
                    arc_dist = max(arc_size, float(dist_factor_local) * arc_size)

                    f_dist = gmsh.model.mesh.field.add("Distance")
                    if arc_curves:
                        gmsh.model.mesh.field.setNumbers(f_dist, "CurvesList", arc_curves)
                    if arc_pts:
                        gmsh.model.mesh.field.setNumbers(f_dist, "PointsList", arc_pts)

                    f_thresh = gmsh.model.mesh.field.add("Threshold")
                    gmsh.model.mesh.field.setNumber(f_thresh, "InField", float(f_dist))
                    gmsh.model.mesh.field.setNumber(f_thresh, "SizeMin", float(arc_size))
                    gmsh.model.mesh.field.setNumber(f_thresh, "SizeMax", float(max_region_size))
                    gmsh.model.mesh.field.setNumber(f_thresh, "DistMin", 0.0)
                    gmsh.model.mesh.field.setNumber(f_thresh, "DistMax", float(arc_dist))
                    gmsh.model.mesh.field.setNumber(f_thresh, "StopAtDistMax", 1.0)

                    f_restrict = gmsh.model.mesh.field.add("Restrict")
                    gmsh.model.mesh.field.setNumber(f_restrict, "InField", float(f_thresh))
                    gmsh.model.mesh.field.setNumbers(f_restrict, "SurfacesList", [int(s) for s in surface_tags])
                    all_fields.append(f_restrict)

            if interface_transition_specs:
                for spec in interface_transition_specs:
                    curve_tag = int(spec["curve_tag"])
                    target_surfaces = [int(s) for s in spec["target_surfaces"]]
                    size_min_local = max(mesh_size_min, float(spec["size_min"]))
                    size_max_local = max(size_min_local, float(spec["size_max"]))
                    dist_max_local = max(
                        size_min_local,
                        float(gmsh_interface_transition_dist_factor) * size_max_local,
                    )

                    f_dist = gmsh.model.mesh.field.add("Distance")
                    gmsh.model.mesh.field.setNumbers(f_dist, "CurvesList", [int(curve_tag)])

                    f_thresh = gmsh.model.mesh.field.add("Threshold")
                    gmsh.model.mesh.field.setNumber(f_thresh, "InField", float(f_dist))
                    gmsh.model.mesh.field.setNumber(f_thresh, "SizeMin", float(size_min_local))
                    gmsh.model.mesh.field.setNumber(f_thresh, "SizeMax", float(size_max_local))
                    gmsh.model.mesh.field.setNumber(f_thresh, "DistMin", 0.0)
                    gmsh.model.mesh.field.setNumber(f_thresh, "DistMax", float(dist_max_local))
                    gmsh.model.mesh.field.setNumber(f_thresh, "StopAtDistMax", 1.0)

                    f_restrict = gmsh.model.mesh.field.add("Restrict")
                    gmsh.model.mesh.field.setNumber(f_restrict, "InField", float(f_thresh))
                    gmsh.model.mesh.field.setNumbers(f_restrict, "SurfacesList", [int(s) for s in target_surfaces])
                    all_fields.append(f_restrict)
                    interface_transition_field_count += 1

            if len(all_fields) == 1:
                bg_field = all_fields[0]
            else:
                bg_field = gmsh.model.mesh.field.add("Min")
                gmsh.model.mesh.field.setNumbers(bg_field, "FieldsList", [int(fid) for fid in all_fields])

            gmsh.model.mesh.field.setAsBackgroundMesh(int(bg_field))
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0.0)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0.0)
        _record_phase("build_size_fields", phase_started_at)
        _mark_build_stage("after-build-size-fields")

        # ---- 4. Per-surface algorithm and recombination flags ----------
        phase_started_at = time.perf_counter()
        want_recombine = False

        def _oriented_curve_chain(curve_tag_signed: int) -> Optional[Tuple[int, ...]]:
            chain = polycurve_chain_by_tag.get(abs(int(curve_tag_signed)))
            if not chain:
                return None
            if int(curve_tag_signed) < 0:
                return tuple(int(t) for t in reversed(chain))
            return tuple(int(t) for t in chain)

        def _curve_length_from_chain(chain: Sequence[int]) -> float:
            if len(chain) < 2:
                return 0.0
            total = 0.0
            for i in range(len(chain) - 1):
                p0 = pt_xy_by_tag.get(int(chain[i]))
                p1 = pt_xy_by_tag.get(int(chain[i + 1]))
                if p0 is None or p1 is None:
                    continue
                total += float(np.hypot(float(p1[0]) - float(p0[0]), float(p1[1]) - float(p0[1])))
            return float(total)

        def _distribute_divisions(total_div: int, seg_lengths: Sequence[float]) -> List[int]:
            n = int(len(seg_lengths))
            if n <= 0:
                return []
            total_div_local = max(int(total_div), int(n))
            base = [1] * n
            remaining = int(total_div_local - n)
            if remaining <= 0:
                return base

            lengths = [max(float(v), 0.0) for v in seg_lengths]
            lsum = float(sum(lengths))
            if lsum <= 1.0e-12:
                for i in range(remaining):
                    base[i % n] += 1
                return base

            raw = [float(remaining) * (float(v) / lsum) for v in lengths]
            adds = [int(np.floor(v)) for v in raw]
            used = int(sum(adds))
            rem = int(remaining - used)
            if rem > 0:
                frac_rank = sorted(
                    [(i, float(raw[i]) - float(adds[i])) for i in range(n)],
                    key=lambda it: it[1],
                    reverse=True,
                )
                for i in range(rem):
                    adds[int(frac_rank[i % n][0])] += 1
            return [int(base[i] + adds[i]) for i in range(n)]

        def _transfinite_corners_from_edge_groups(edge_curve_groups: Sequence[Sequence[int]]) -> Optional[List[int]]:
            if len(edge_curve_groups) != 4:
                return None
            corners: List[int] = []
            for group in edge_curve_groups:
                if not group:
                    return None
                chain = _oriented_curve_chain(int(group[0]))
                if chain is None or len(chain) < 2:
                    return None
                corners.append(int(chain[0]))
            if len(corners) != 4:
                return None
            return corners

        def _transfinite_corners_from_edge_controls(
            edge_controls: Optional[Sequence[QuadEdgeControl]],
        ) -> Optional[List[int]]:
            if edge_controls is None or len(edge_controls) != 4:
                return None
            corners: List[int] = []
            corner_lc = max(float(tol), 1.0e-9)
            for edge in edge_controls:
                pts = list(edge.points_xy or [])
                if not pts:
                    return None
                x0 = float(pts[0][0])
                y0 = float(pts[0][1])
                tag = _geo_pt(x0, y0, float(corner_lc))
                corners.append(int(tag))
            if len(corners) != 4:
                return None
            if len(set(int(v) for v in corners)) != 4:
                return None
            return corners

        def _apply_flow_aligned_transfinite(
            surf_tag: int,
            curve_tags: Sequence[int],
            edge_controls: Optional[List[QuadEdgeControl]],
            fallback_size: float,
            counts_override: Optional[Sequence[int]] = None,
            min_nodes: Optional[Sequence[int]] = None,
            edge_curve_groups: Optional[Sequence[Sequence[int]]] = None,
        ) -> Tuple[bool, Optional[str]]:
            if edge_controls is None:
                return False, "missing edge controls"
            edge_ids_local = [int(getattr(edge, "edge_id", -1)) for edge in list(edge_controls)]
            counts = list(counts_override) if counts_override is not None else _mh()._gmsh_flow_aligned_curve_counts(
                edge_controls,
                fallback_size=fallback_size,
                min_nodes=min_nodes,
            )
            if counts is None:
                return False, "could not compute transfinite counts"
            counts = [int(max(2, int(v))) for v in list(counts)]
            if len(edge_ids_local) == 4 and set(edge_ids_local) == {1, 2, 3, 4}:
                idx_by_edge_local = {int(eid): int(i) for i, eid in enumerate(edge_ids_local)}
                i1 = idx_by_edge_local.get(1)
                i3 = idx_by_edge_local.get(3)
                if i1 is not None and i3 is not None:
                    paired = max(int(counts[i1]), int(counts[i3]))
                    counts[i1] = int(paired)
                    counts[i3] = int(paired)
                i2 = idx_by_edge_local.get(2)
                i4 = idx_by_edge_local.get(4)
                if i2 is not None and i4 is not None:
                    paired = max(int(counts[i2]), int(counts[i4]))
                    counts[i2] = int(paired)
                    counts[i4] = int(paired)
            else:
                counts[0] = counts[2] = max(int(counts[0]), int(counts[2]))
                counts[1] = counts[3] = max(int(counts[1]), int(counts[3]))
            try:
                _record_build_event(
                    "transfinite-apply-start",
                    int(surf_tag),
                    int(len(curve_tags)),
                    int(len(counts)),
                    _hash_int_sequence([int(v) for v in counts]),
                )
                groups = list(edge_curve_groups) if edge_curve_groups is not None else []
                has_group_data = len(groups) == 4 and all(len(g) > 0 for g in groups)
                if has_group_data:
                    # Segmented interfaces require at least one division per
                    # segment. If a side has many segments, raise counts to a
                    # feasible minimum and then re-equalize opposite edges.
                    effective_counts = [int(max(2, int(v))) for v in counts]
                    for ei in range(4):
                        nseg = int(len(groups[ei]))
                        effective_counts[ei] = max(int(effective_counts[ei]), int(nseg + 1))
                    if len(edge_ids_local) == 4 and set(edge_ids_local) == {1, 2, 3, 4}:
                        idx_by_edge_local = {int(eid): int(i) for i, eid in enumerate(edge_ids_local)}
                        i1 = idx_by_edge_local.get(1)
                        i3 = idx_by_edge_local.get(3)
                        if i1 is not None and i3 is not None:
                            paired = max(int(effective_counts[i1]), int(effective_counts[i3]))
                            effective_counts[i1] = int(paired)
                            effective_counts[i3] = int(paired)
                        i2 = idx_by_edge_local.get(2)
                        i4 = idx_by_edge_local.get(4)
                        if i2 is not None and i4 is not None:
                            paired = max(int(effective_counts[i2]), int(effective_counts[i4]))
                            effective_counts[i2] = int(paired)
                            effective_counts[i4] = int(paired)
                    else:
                        pair0 = max(int(effective_counts[0]), int(effective_counts[2]))
                        pair1 = max(int(effective_counts[1]), int(effective_counts[3]))
                        effective_counts[0] = effective_counts[2] = int(pair0)
                        effective_counts[1] = effective_counts[3] = int(pair1)

                    for ei in range(4):
                        group = [int(v) for v in groups[ei]]
                        n_total = max(2, int(effective_counts[ei]))
                        seg_lengths: List[float] = []
                        for ltag in group:
                            chain = _oriented_curve_chain(int(ltag))
                            seg_lengths.append(_curve_length_from_chain(chain or ()))
                        divs = _distribute_divisions(max(1, int(n_total) - 1), seg_lengths)
                        for ltag, div in zip(group, divs):
                            gmsh.model.mesh.setTransfiniteCurve(abs(int(ltag)), int(max(2, int(div) + 1)))
                            _record_build_event(
                                "transfinite-curve",
                                int(abs(int(ltag))),
                                int(max(2, int(div) + 1)),
                            )

                    corners = _transfinite_corners_from_edge_controls(edge_controls)
                    if corners is None:
                        corners = _transfinite_corners_from_edge_groups(groups)
                    if corners is not None and len(corners) == 4:
                        gmsh.model.mesh.setTransfiniteSurface(int(surf_tag), "Left", [int(v) for v in corners])
                        _record_build_event(
                            "transfinite-surface",
                            int(surf_tag),
                            "Left",
                            _hash_int_sequence([int(v) for v in corners]),
                        )
                    else:
                        gmsh.model.mesh.setTransfiniteSurface(int(surf_tag))
                        _record_build_event("transfinite-surface", int(surf_tag), "auto")
                else:
                    if len(curve_tags) != 4:
                        return False, "missing edge group data for non-4-curve surface"
                    for ltag, npt in zip(curve_tags, counts):
                        gmsh.model.mesh.setTransfiniteCurve(abs(int(ltag)), int(npt))
                        _record_build_event(
                            "transfinite-curve",
                            int(abs(int(ltag))),
                            int(npt),
                        )
                    gmsh.model.mesh.setTransfiniteSurface(int(surf_tag))
                    _record_build_event("transfinite-surface", int(surf_tag), "auto")
                return True, None
            except Exception as exc:
                return False, str(exc)

        shared_transverse_count_normalize_diag: Dict[str, object] = {
            "enabled": bool(gmsh_shared_transverse_edge_count_normalize),
            "shared_group_count": 0,
            "affected_surface_count": 0,
        }
        shared_transverse_count_overrides: Dict[int, List[int]] = {}
        if gmsh_shared_transverse_edge_count_normalize and gmsh_quad_full_region_flow_align:
            def _edge_group_key(curve_group: Sequence[int]) -> Tuple[int, ...]:
                return tuple(sorted({abs(int(t)) for t in list(curve_group or []) if int(t) != 0}))

            base_counts_by_surface: Dict[int, List[int]] = {}
            edge_ids_by_surface: Dict[int, List[int]] = {}
            region_id_by_surface: Dict[int, int] = {}
            entries: List[Tuple[int, int, int, Tuple[int, ...]]] = []

            for surf, (rid, ctype, sz) in zip(surface_tags, surface_meta):
                ctype_local = str(ctype).strip().lower()
                if ctype_local not in {"cartesian", "quadrilateral", "channel_generator"}:
                    continue
                edge_controls_local = surface_quad_controls.get(surf)
                groups_local = surface_quad_edge_curve_groups.get(surf)
                if edge_controls_local is None or len(edge_controls_local) != 4:
                    continue
                if groups_local is None or len(groups_local) != 4:
                    continue

                min_nodes_local = [
                    int(transfinite_edge_min_nodes.get((int(rid), int(edge.edge_id)), 0))
                    for edge in edge_controls_local
                ]
                min_nodes_local_use: Optional[List[int]] = None
                if any(int(v) > 0 for v in min_nodes_local):
                    min_nodes_local_use = [int(v) for v in min_nodes_local]

                counts_local = _mh()._gmsh_flow_aligned_curve_counts(
                    edge_controls_local,
                    fallback_size=float(sz),
                    min_nodes=min_nodes_local_use,
                )
                if counts_local is None:
                    continue

                s = int(surf)
                base_counts_by_surface[s] = [int(v) for v in counts_local]
                edge_ids_local = [int(getattr(edge, "edge_id", -1)) for edge in edge_controls_local]
                edge_ids_by_surface[s] = edge_ids_local
                region_id_by_surface[s] = int(rid)

                for idx, curve_group in enumerate(groups_local):
                    if idx >= len(edge_ids_local):
                        continue
                    edge_id_local = int(edge_ids_local[idx])
                    if edge_id_local not in {1, 2, 3, 4}:
                        continue
                    key = _edge_group_key(curve_group)
                    if not key:
                        continue
                    entries.append((s, int(idx), int(edge_id_local), key))

            owners_by_key: Dict[Tuple[int, ...], set] = {}
            for s, _idx, _eid, key in entries:
                owners_by_key.setdefault(key, set()).add(int(s))

            target_by_key: Dict[Tuple[int, ...], int] = {}
            for key, owners in owners_by_key.items():
                if len(owners) < 2:
                    continue
                vals: List[int] = []
                for s, idx, _eid, key_local in entries:
                    if key_local != key:
                        continue
                    if int(s) not in owners:
                        continue
                    vals.append(int(base_counts_by_surface.get(int(s), [0, 0, 0, 0])[int(idx)]))
                if vals:
                    target_by_key[key] = int(max(vals))

            preview: List[Dict[str, object]] = []
            entries_by_surface: Dict[int, List[Tuple[int, int, int, Tuple[int, ...]]]] = {}
            for rec in entries:
                s, _idx, _eid, key = rec
                if key not in target_by_key:
                    continue
                entries_by_surface.setdefault(int(s), []).append(rec)

            for s, recs in entries_by_surface.items():
                base_counts = list(base_counts_by_surface.get(int(s), []))
                edge_ids_local = list(edge_ids_by_surface.get(int(s), []))
                if len(base_counts) != 4 or len(edge_ids_local) != 4:
                    continue

                counts_new = [int(v) for v in base_counts]
                changed = False
                for _s, idx, edge_id_local, key in recs:
                    target_nodes = int(target_by_key.get(key, 0))
                    if target_nodes <= 0:
                        continue
                    if int(target_nodes) > int(counts_new[idx]):
                        counts_new[idx] = int(target_nodes)
                        changed = True
                    if len(preview) < 12:
                        preview.append(
                            {
                                "surface_tag": int(s),
                                "region_id": int(region_id_by_surface.get(int(s), -1)),
                                "edge_id": int(edge_id_local),
                                "base_nodes": int(base_counts[idx]),
                                "target_nodes": int(target_nodes),
                                "shared_curve_count": int(len(key)),
                            }
                        )

                if not changed:
                    continue

                if len(edge_ids_local) == 4 and set(edge_ids_local) == {1, 2, 3, 4}:
                    idx_by_edge_local = {int(eid): int(i) for i, eid in enumerate(edge_ids_local)}
                    i1 = idx_by_edge_local.get(1)
                    i3 = idx_by_edge_local.get(3)
                    if i1 is not None and i3 is not None:
                        paired = max(int(counts_new[i1]), int(counts_new[i3]))
                        counts_new[i1] = int(paired)
                        counts_new[i3] = int(paired)
                    i2 = idx_by_edge_local.get(2)
                    i4 = idx_by_edge_local.get(4)
                    if i2 is not None and i4 is not None:
                        paired = max(int(counts_new[i2]), int(counts_new[i4]))
                        counts_new[i2] = int(paired)
                        counts_new[i4] = int(paired)
                else:
                    counts_new[0] = counts_new[2] = max(int(counts_new[0]), int(counts_new[2]))
                    counts_new[1] = counts_new[3] = max(int(counts_new[1]), int(counts_new[3]))

                shared_transverse_count_overrides[int(s)] = [int(v) for v in counts_new]

            shared_transverse_count_normalize_diag["shared_group_count"] = int(len(target_by_key))
            shared_transverse_count_normalize_diag["affected_surface_count"] = int(len(shared_transverse_count_overrides))
            if preview:
                shared_transverse_count_normalize_diag["preview"] = list(preview)

        for surf, (rid, ctype, sz) in zip(surface_tags, surface_meta):
            region = next((r for r in model.regions if int(r.region_id) == int(rid)), None)
            lines = surface_curve_tags.get(surf, [])
            quad_controls = surface_quad_controls.get(surf)
            edge_curve_groups = surface_quad_edge_curve_groups.get(surf)
            edge_min_nodes: Optional[List[int]] = None
            if quad_controls is not None and len(quad_controls) == 4:
                min_nodes_local = [
                    int(transfinite_edge_min_nodes.get((int(rid), int(edge.edge_id)), 0))
                    for edge in quad_controls
                ]
                if any(int(v) > 0 for v in min_nodes_local):
                    edge_min_nodes = min_nodes_local
            flow_aligned_applied = False
            flow_align_preflight_fallback = False
            if gmsh_quad_full_region_flow_align and ctype in {"cartesian", "quadrilateral", "channel_generator"}:
                diag = _mh()._gmsh_flow_align_region_preflight(
                    region_id=int(rid),
                    cell_type=str(ctype),
                    curve_tags=lines,
                    edge_controls=quad_controls,
                    fallback_size=float(sz),
                    min_nodes=edge_min_nodes,
                )
                diag["surface_tag"] = int(surf)
                diag["requested"] = True
                normalized_counts_override = shared_transverse_count_overrides.get(int(surf))
                if normalized_counts_override is not None:
                    diag["shared_transverse_count_normalized"] = True
                    diag["transfinite_counts_normalized"] = [int(v) for v in list(normalized_counts_override)]
                else:
                    diag["shared_transverse_count_normalized"] = False
                counts_for_apply = normalized_counts_override
                if counts_for_apply is None:
                    counts_for_apply = diag.get("transfinite_counts")
                if bool(diag.get("eligible", False)):
                    ok, err = _apply_flow_aligned_transfinite(
                        surf_tag=int(surf),
                        curve_tags=lines,
                        edge_controls=quad_controls,
                        fallback_size=float(sz),
                        counts_override=counts_for_apply,
                        min_nodes=edge_min_nodes,
                        edge_curve_groups=edge_curve_groups,
                    )
                    flow_aligned_applied = bool(ok)
                    if ok:
                        diag["status"] = "applied"
                        diag["fallback"] = False
                    else:
                        diag["status"] = "fallback"
                        diag["fallback"] = True
                        flow_align_preflight_fallback = True
                        diag["reasons"] = list(diag.get("reasons", [])) + [
                            "gmsh-transfinite-apply-failed"
                        ]
                        if err:
                            diag["apply_error"] = str(err)
                        warnings.warn(
                            "Gmsh flow-align fallback for region "
                            f"{int(rid)}: transfinite apply failed ({err}).",
                            RuntimeWarning,
                        )
                else:
                    diag["status"] = "fallback"
                    flow_align_preflight_fallback = True
                    reason_txt = ", ".join(str(x) for x in diag.get("reasons", []) if str(x)) or "unknown"
                    warnings.warn(
                        "Gmsh flow-align fallback for region "
                        f"{int(rid)}: {reason_txt}",
                        RuntimeWarning,
                    )
                if flow_align_preflight_fallback:
                    diag["transfinite_skipped_after_fallback"] = True
                flow_align_diagnostics.append(diag)
                self._last_flow_align_diagnostics = list(flow_align_diagnostics)
            if ctype == "cartesian":
                # Transfinite + Recombine: structured, fast, pure quads.
                if (
                    (not flow_align_preflight_fallback)
                    and
                    (not flow_aligned_applied)
                    and region is not None
                    and region.edge_lengths
                    and len(region.edge_lengths) == 4
                    and quad_controls is not None
                    and len(quad_controls) == 4
                ):
                    try:
                        edge_geom_len = []
                        edge_geom_len = [_mh()._polyline_length(edge.points_xy) for edge in quad_controls]
                        counts = []
                        for i in range(4):
                            tlen = max(float(region.edge_lengths[i]), tol)
                            ndiv = max(1, int(round(edge_geom_len[i] / tlen)))
                            counts.append(max(2, ndiv + 1))

                        if edge_min_nodes is not None and len(edge_min_nodes) == 4:
                            counts = [max(int(c), int(mn)) if int(mn) > 0 else int(c) for c, mn in zip(counts, edge_min_nodes)]

                        # Opposite edges must match for transfinite surface.
                        n0 = max(counts[0], counts[2])
                        n1 = max(counts[1], counts[3])
                        counts[0] = counts[2] = n0
                        counts[1] = counts[3] = n1
                        ok_tf, _err_tf = _apply_flow_aligned_transfinite(
                            surf_tag=int(surf),
                            curve_tags=lines,
                            edge_controls=quad_controls,
                            fallback_size=float(sz),
                            counts_override=counts,
                            min_nodes=edge_min_nodes,
                            edge_curve_groups=edge_curve_groups,
                        )
                        if not ok_tf:
                            gmsh.model.mesh.setTransfiniteSurface(surf)
                    except Exception as e:
                        logger.warning("gmsh flow-aligned transfinite setup failed for surf=%s: %s", surf, e, exc_info=True)
                        try:
                            gmsh.model.mesh.setTransfiniteSurface(surf)
                        except Exception as e2:
                            logger.warning("gmsh fallback setTransfiniteSurface failed for surf=%s: %s", surf, e2, exc_info=True)
                            pass
                elif (not flow_align_preflight_fallback) and (not flow_aligned_applied):
                    try:
                        gmsh.model.mesh.setTransfiniteSurface(surf)
                    except Exception as e:
                        logger.warning("gmsh setTransfiniteSurface failed for surf=%s: %s", surf, e, exc_info=True)
                        pass  # Works best for 4-sided surfaces.
                gmsh.model.mesh.setRecombine(2, surf)
                want_recombine = True
                # Packing of Parallelograms requires a scaled cross field and
                # is brittle on real project geometries.  For structured quad
                # surfaces, transfinite constraints plus recombination are the
                # controlling inputs; keep the base 2D algorithm on the safer
                # frontal path.
                try:
                    gmsh.model.mesh.setAlgorithm(2, surf, quad_algo)
                except Exception as e:
                    logger.warning("gmsh setAlgorithm(quad_algo) failed for surf=%s: %s", surf, e, exc_info=True)
                    gmsh.option.setNumber("Mesh.Algorithm", float(quad_algo))
            elif ctype in {"quadrilateral", "channel_generator"}:
                # Unstructured quads via Blossom recombination.
                if (
                    (not flow_align_preflight_fallback)
                    and
                    (not flow_aligned_applied)
                    and region is not None
                    and region.edge_lengths
                    and len(region.edge_lengths) == 4
                    and quad_controls is not None
                    and len(quad_controls) == 4
                ):
                    try:
                        edge_geom_len = []
                        edge_geom_len = [_mh()._polyline_length(edge.points_xy) for edge in quad_controls]
                        counts = []
                        for i in range(4):
                            tlen = max(float(region.edge_lengths[i]), tol)
                            ndiv = max(1, int(round(edge_geom_len[i] / tlen)))
                            counts.append(max(2, ndiv + 1))
                        if edge_min_nodes is not None and len(edge_min_nodes) == 4:
                            counts = [max(int(c), int(mn)) if int(mn) > 0 else int(c) for c, mn in zip(counts, edge_min_nodes)]
                        n0 = max(counts[0], counts[2])
                        n1 = max(counts[1], counts[3])
                        counts[0] = counts[2] = n0
                        counts[1] = counts[3] = n1
                        _apply_flow_aligned_transfinite(
                            surf_tag=int(surf),
                            curve_tags=lines,
                            edge_controls=quad_controls,
                            fallback_size=float(sz),
                            counts_override=counts,
                            min_nodes=edge_min_nodes,
                            edge_curve_groups=edge_curve_groups,
                        )
                    except Exception as e:
                        logger.warning("gmsh unstructured quad flow-align failed for surf=%s: %s", surf, e, exc_info=True)
                        pass
                gmsh.model.mesh.setRecombine(2, surf)
                want_recombine = True
                # For general quad regions, generate triangles with the frontal
                # algorithm and let Blossom handle recombination.  This avoids
                # the scaled-cross-field requirement that triggers terminal
                # errors like: "Packing of Parallelograms require a scaled
                # cross field".
                try:
                    gmsh.model.mesh.setAlgorithm(2, surf, quad_algo)
                except Exception as e:
                    logger.warning("gmsh setAlgorithm(quad_algo) unstructured failed for surf=%s: %s", surf, e, exc_info=True)
                    gmsh.option.setNumber("Mesh.Algorithm", float(quad_algo))
            else:
                # triangular: frontal Delaunay for quality.
                try:
                    gmsh.model.mesh.setAlgorithm(2, surf, tri_algo)
                except Exception as e:
                    logger.warning("gmsh setAlgorithm(tri_algo) failed for surf=%s: %s", surf, e, exc_info=True)
                    gmsh.option.setNumber("Mesh.Algorithm", float(tri_algo))
        _record_phase("configure_per_surface", phase_started_at)
        _mark_build_stage("after-configure-per-surface")

        # ---- 5. Global mesh options ------------------------------------
        phase_started_at = time.perf_counter()
        def _set_global_mesh_option(name: str, value: object) -> None:
            numeric_value = float(value)
            gmsh.option.setNumber(str(name), float(numeric_value))
            _record_global_option(str(name), float(numeric_value))

        _set_global_mesh_option("Mesh.RecombineAll", 0.0)  # per-surface only
        _set_global_mesh_option("Mesh.RecombinationAlgorithm", float(recomb_algo))
        _set_global_mesh_option("Mesh.RecombineOptimizeTopology", float(max(0, int(recombine_optimize_topology))))
        _set_global_mesh_option("Mesh.RecombineNodeRepositioning", 1.0 if recombine_node_repositioning else 0.0)
        _set_global_mesh_option("Mesh.RecombineMinimumQuality", max(0.0, float(recombine_minimum_quality)))
        _set_global_mesh_option("Mesh.Smoothing", float(smoothing_passes))
        _set_global_mesh_option("Mesh.OptimizeNetgen", 1.0 if optimize_netgen else 0.0)
        _set_global_mesh_option("Mesh.AlgorithmSwitchOnFailure", 1.0 if algorithm_switch_on_failure else 0.0)
        _set_global_mesh_option("Mesh.RandomFactor", max(0.0, float(random_factor)))
        _set_global_mesh_option("Mesh.MeshSizeMin", float(mesh_size_min))
        _set_global_mesh_option("Mesh.ToleranceEdgeLength", float(tolerance_edge_length))
        _set_global_mesh_option("Mesh.MeshSizeFromPoints", 1.0 if mesh_size_from_points else 0.0)
        _set_global_mesh_option("General.NumThreads", float(gmsh_num_threads))
        _set_global_mesh_option("Mesh.MaxNumThreads2D", float(gmsh_max_num_threads_2d))
        _record_build_event(
            "mesh-options-summary",
            int(tri_algo),
            int(quad_algo),
            int(recomb_algo),
            int(smoothing_passes),
            int(optimize_iters),
            float(mesh_size_min),
            float(tolerance_edge_length),
            bool(mesh_size_from_points),
            int(gmsh_num_threads),
            int(gmsh_max_num_threads_2d),
        )
        _record_phase("configure_global_options", phase_started_at)
        _mark_build_stage("after-configure-global-options")

        # ---- 6. Generate -----------------------------------------------
        phase_started_at = time.perf_counter()
        gmsh_build_order_fingerprint = _build_order_fingerprint_payload()
        gmsh_build_order_stage_ladder = _build_order_stage_ladder_payload()
        gmsh_global_options = _global_options_payload()
        gmsh_pre_generate_entity_signature = _pre_generate_entity_signature_payload()
        self._last_build_order_fingerprint = dict(gmsh_build_order_fingerprint)
        self._last_build_order_stage_ladder = dict(gmsh_build_order_stage_ladder)
        self._last_pre_generate_entity_signature = dict(gmsh_pre_generate_entity_signature)
        try:
            gmsh.model.mesh.generate(2)
        except Exception as exc:
            raise RuntimeError(
                "Gmsh mesh.generate(2) failed "
                f"(build_order_sha256={gmsh_build_order_fingerprint.get('sha256', '')}, "
                f"build_stage_ladder_sha256={gmsh_build_order_stage_ladder.get('sha256', '')}, "
                f"build_stage_ladder={_build_order_stage_ladder_compact_text(gmsh_build_order_stage_ladder)}, "
                f"global_options_sha256={gmsh_global_options.get('sha256', '')}, "
                f"global_options={_global_options_compact_text(gmsh_global_options)}, "
                f"entity_sha256={gmsh_pre_generate_entity_signature.get('sha256', '')}, "
                f"entity_counts={gmsh_pre_generate_entity_signature.get('counts', {})}): {exc}"
            ) from exc
        if want_recombine and bool(gmsh_global_recombine):
            try:
                gmsh.model.mesh.recombine()
            except Exception as e:
                logger.warning("gmsh recombine failed: %s", e, exc_info=True)
                pass
        if optimize_iters > 0:
            methods = tuple(str(m).strip() for m in (optimize_methods or ()) if str(m).strip())
            if not methods:
                methods = ("Laplace2D",)
            for method in methods:
                try:
                    gmsh.model.mesh.optimize(method, niter=int(optimize_iters))
                except TypeError:
                    gmsh.model.mesh.optimize(method)
        _record_phase("generate_and_optimize", phase_started_at)

        phase_started_at = time.perf_counter()
        duplicate_cleanup_summary: Optional[Dict[str, object]] = None
        duplicate_before_count = 0
        duplicate_after_count = 0
        duplicate_cleanup_ran = False
        try:
            dup_before = gmsh.model.mesh.getDuplicateNodes([])
        except TypeError:
            dup_before = gmsh.model.mesh.getDuplicateNodes()
        except Exception as e:
            logger.warning("gmsh getDuplicateNodes (before) failed: %s", e, exc_info=True)
            dup_before = []
        if dup_before is None:
            dup_before = []
        duplicate_before_count = int(len(dup_before))
        if duplicate_before_count > 0:
            duplicate_cleanup_ran = True
            warnings.warn(
                "Gmsh mesh duplicate-node cleanup triggered "
                f"(duplicates={duplicate_before_count}).",
                RuntimeWarning,
            )
            try:
                gmsh.model.mesh.removeDuplicateNodes([])
            except TypeError:
                gmsh.model.mesh.removeDuplicateNodes()
            except Exception as e:
                logger.warning("gmsh removeDuplicateNodes failed: %s", e, exc_info=True)
                pass
            try:
                gmsh.model.mesh.removeDuplicateElements([])
            except TypeError:
                gmsh.model.mesh.removeDuplicateElements()
            except Exception as e:
                logger.warning("gmsh removeDuplicateElements failed: %s", e, exc_info=True)
                pass
            try:
                dup_after = gmsh.model.mesh.getDuplicateNodes([])
            except TypeError:
                dup_after = gmsh.model.mesh.getDuplicateNodes()
            except Exception as e:
                logger.warning("gmsh getDuplicateNodes (after) failed: %s", e, exc_info=True)
                dup_after = []
            if dup_after is None:
                dup_after = []
            duplicate_after_count = int(len(dup_after))
            if duplicate_after_count > 0:
                warnings.warn(
                    "Gmsh duplicate-node cleanup completed with remaining duplicates "
                    f"(remaining={duplicate_after_count}).",
                    RuntimeWarning,
                )
            duplicate_cleanup_summary = {
                "duplicate_nodes_before": int(duplicate_before_count),
                "duplicate_nodes_after": int(duplicate_after_count),
                "cleanup_ran": bool(duplicate_cleanup_ran),
            }
        _record_phase("duplicate_cleanup", phase_started_at)

        # ---- 7. Extract nodes ------------------------------------------
        phase_started_at = time.perf_counter()
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        # node_coords: flat [x0,y0,z0, x1,y1,z1, ...]
        node_coords = np.array(node_coords, dtype=np.float64).reshape(-1, 3)
        tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}
        node_x = node_coords[:, 0].copy()
        node_y = node_coords[:, 1].copy()
        node_z = np.zeros(node_x.shape[0], dtype=np.float64)
        _record_phase("extract_nodes", phase_started_at)

        # ---- 8. Extract elements per surface with metadata -------------
        phase_started_at = time.perf_counter()
        all_face_offsets: List[int] = [0]
        all_face_nodes: List[int] = []
        all_tris: List[int] = []
        all_cell_type: List[str] = []
        all_region_id: List[int] = []
        all_size: List[float] = []

        # Gmsh element type codes: 2 = 3-node triangle, 3 = 4-node quad
        for surf, (rid, ctype, sz) in zip(surface_tags, surface_meta):
            elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2, surf)
            for etype, _, enodes in zip(elem_types, elem_tags, elem_node_tags):
                enodes = np.array(enodes, dtype=np.int64)
                if etype == 2:  # triangle
                    n_elems = len(enodes) // 3
                    enodes = enodes.reshape(n_elems, 3)
                    for tri in enodes:
                        v = [tag_to_idx[int(t)] for t in tri]
                        all_face_nodes.extend(v)
                        all_face_offsets.append(len(all_face_nodes))
                        all_tris.extend(v)
                        all_cell_type.append(ctype)
                        all_region_id.append(rid)
                        all_size.append(sz)
                elif etype == 3:  # quad
                    n_elems = len(enodes) // 4
                    enodes = enodes.reshape(n_elems, 4)
                    for quad in enodes:
                        v = [tag_to_idx[int(t)] for t in quad]
                        all_face_nodes.extend(v)
                        all_face_offsets.append(len(all_face_nodes))
                        # Fan-triangulate for plotting: 0-1-2, 0-2-3
                        all_tris.extend([v[0], v[1], v[2], v[0], v[2], v[3]])
                        all_cell_type.append(ctype)
                        all_region_id.append(rid)
                        all_size.append(sz)

        if not all_face_offsets or len(all_face_offsets) == 1:
            self._last_flow_align_diagnostics = list(flow_align_diagnostics)
            if flow_align_diagnostics:
                diag_parts: List[str] = []
                for d in flow_align_diagnostics:
                    rid_txt = str(d.get("region_id", "?"))
                    status_txt = str(d.get("status", "unknown"))
                    reasons_txt = ",".join(str(x) for x in d.get("reasons", []) if str(x))
                    if not reasons_txt:
                        reasons_txt = "none"
                    diag_parts.append(
                        f"region={rid_txt};status={status_txt};reasons={reasons_txt}"
                    )
                raise ValueError(
                    "GmshBackend: no elements extracted from mesh. "
                    "Flow-align per-region diagnostics: " + " | ".join(diag_parts)
                )
            raise ValueError("GmshBackend: no elements extracted from mesh.")
        _record_phase("extract_elements", phase_started_at)
        gmsh_phase_timings_s["total_build"] = float(max(0.0, time.perf_counter() - build_started_at))

        out = MeshResult(
            node_x=node_x,
            node_y=node_y,
            node_z=node_z,
            cell_nodes=np.asarray(all_tris, dtype=np.int32),
            cell_face_offsets=np.asarray(all_face_offsets, dtype=np.int32),
            cell_face_nodes=np.asarray(all_face_nodes, dtype=np.int32),
            cell_type=np.asarray(all_cell_type, dtype=object),
            region_id=np.asarray(all_region_id, dtype=np.int32),
            target_size=np.asarray(all_size, dtype=np.float64),
        )
        if bool(gmsh_interface_conformance):
            out = _mh()._enforce_quad_interface_conformance(
                out,
                model,
                snap_tol=float(gmsh_interface_snap_tol),
                centroid_merge=bool(gmsh_transverse_interface_centroid_merge),
            )
        out.quality_summary = dict(out.quality_summary or {})
        if bool(gmsh_interface_reject_near_unshared):
            near_unshared_report = _mh()._mixed_transfinite_tri_near_unshared_report(
                out,
                tol=float(gmsh_interface_reject_tol),
            )
            out.quality_summary["gmsh_interface_near_unshared_check"] = dict(near_unshared_report)
            flagged_pair_count = int(near_unshared_report.get("flagged_pair_count", 0) or 0)
            if flagged_pair_count > 0:
                flagged_pairs = list(near_unshared_report.get("flagged_pairs") or [])
                preview = "; ".join(
                    (
                        f"{int(p.get('region_pair', [-1, -1])[0])}-{int(p.get('region_pair', [-1, -1])[1])} "
                        f"(near_only={int(p.get('near_only_a', 0))}/{int(p.get('near_only_b', 0))}, "
                        f"shared_edges={int(p.get('shared_edge_count', 0))})"
                    )
                    for p in flagged_pairs[:6]
                )
                raise ValueError(
                    "Gmsh mixed transfinite/tri interface check failed: detected "
                    f"{flagged_pair_count} region pair(s) with near-coincident unshared nodes "
                    f"(tol={float(gmsh_interface_reject_tol):.6g}). "
                    + (f"Examples: {preview}" if preview else "")
                )
        out.quality_summary["gmsh_phase_timings_s"] = dict(gmsh_phase_timings_s)
        out.quality_summary["gmsh_build_order_fingerprint"] = dict(gmsh_build_order_fingerprint)
        out.quality_summary["gmsh_build_order_stage_ladder"] = dict(gmsh_build_order_stage_ladder)
        out.quality_summary["gmsh_global_options"] = dict(gmsh_global_options)
        out.quality_summary["gmsh_pre_generate_entity_signature"] = dict(gmsh_pre_generate_entity_signature)
        out.quality_summary["gmsh_phase_counts"] = {
            "surface_count": int(len(surface_tags)),
            "constraint_count": int(len(constraint_point_lists)),
            "arc_count": int(len(model.arcs)),
            "face_count": int(max(0, len(all_face_offsets) - 1)),
        }
        meshing_native = _mh()._load_hydra_meshing_native()
        out.quality_summary["gmsh_cpp_prebuild_native"] = {
            "enabled": bool(_mh()._gmsh_cpp_prebuild_enabled()),
            "module_loaded": bool(meshing_native is not None),
            "has_interface_overlap_metrics_closed": bool(
                meshing_native is not None and hasattr(meshing_native, "interface_overlap_metrics_closed")
            ),
            "has_polyline_overlap_fractions_open": bool(
                meshing_native is not None and hasattr(meshing_native, "polyline_overlap_fractions_open")
            ),
            "has_project_ring_to_chain": bool(
                meshing_native is not None and hasattr(meshing_native, "project_ring_to_chain")
            ),
        }
        has_transfinite_harmonize_diag = any(int(v) > 0 for v in transfinite_harmonize_stats.values())
        has_transfinite_harmonize_debug = bool(transfinite_harmonize_debug)
        has_interface_coincidence_diag = bool(interface_coincidence_report)
        has_shared_transverse_count_normalize_diag = bool(gmsh_shared_transverse_edge_count_normalize)
        if flow_align_diagnostics or duplicate_cleanup_summary is not None or interface_transition_specs or has_transfinite_harmonize_diag or has_transfinite_harmonize_debug or has_interface_coincidence_diag or has_shared_transverse_count_normalize_diag:
            out.quality_summary["gmsh_flow_align_diagnostics"] = list(flow_align_diagnostics)
            if duplicate_cleanup_summary is not None:
                out.quality_summary["gmsh_duplicate_cleanup"] = dict(duplicate_cleanup_summary)
            if interface_transition_specs:
                out.quality_summary["gmsh_interface_transition"] = {
                    "enabled": bool(gmsh_interface_transition_enable),
                    "protected_transfinite_surfaces": sorted(int(s) for s in protected_transfinite_surfaces),
                    "spec_count": int(len(interface_transition_specs)),
                    "field_count": int(interface_transition_field_count),
                }
            if has_shared_transverse_count_normalize_diag:
                out.quality_summary["gmsh_shared_transverse_edge_count_normalize"] = dict(shared_transverse_count_normalize_diag)
            if has_transfinite_harmonize_diag:
                out.quality_summary["gmsh_transfinite_interface_harmonize"] = {
                    "enabled": bool(gmsh_transfinite_shared_interface_harmonize),
                    "subset_start": float(gmsh_transfinite_opposite_subset_start),
                    "subset_end": float(gmsh_transfinite_opposite_subset_end),
                    "subset_density_scale": float(gmsh_transfinite_opposite_subset_density_scale),
                    "subset_containment_enable": bool(gmsh_transfinite_subset_containment_enable),
                    "subset_containment_high_overlap": float(gmsh_transfinite_subset_containment_high_overlap),
                    "subset_containment_min_overlap": float(gmsh_transfinite_subset_containment_min_overlap),
                    "subset_containment_max_length_ratio": float(gmsh_transfinite_subset_containment_max_length_ratio),
                    "shared_groups": int(transfinite_harmonize_stats.get("shared_groups", 0)),
                    "canonicalized_edges": int(transfinite_harmonize_stats.get("canonicalized_edges", 0)),
                    "opposite_subset_densified": int(transfinite_harmonize_stats.get("opposite_subset_requests", 0)),
                    "junction_points_inserted": int(transfinite_harmonize_stats.get("junction_points_inserted", 0)),
                    "subset_containment_densified": int(transfinite_harmonize_stats.get("subset_containment_requests", 0)),
                    "singleton_external_junction_edges": int(transfinite_harmonize_stats.get("singleton_external_junction_edges", 0)),
                    "nontrans_neighbor_projection_rings": int(transfinite_harmonize_stats.get("nontrans_neighbor_projection_rings", 0)),
                    "candidate_pair_count_prefilter": int(transfinite_harmonize_stats.get("candidate_pair_count_prefilter", 0)),
                    "candidate_pair_count": int(transfinite_harmonize_stats.get("candidate_pair_count", 0)),
                    "pair_bbox_reject_count": int(transfinite_harmonize_stats.get("pair_bbox_reject_count", 0)),
                    "nontrans_chain_bbox_reject_count": int(transfinite_harmonize_stats.get("nontrans_chain_bbox_reject_count", 0)),
                    "nontrans_overlap_pair_count": int(transfinite_harmonize_stats.get("nontrans_overlap_pair_count", 0)),
                    "nontrans_point_bbox_reject_count": int(transfinite_harmonize_stats.get("nontrans_point_bbox_reject_count", 0)),
                }
            if has_transfinite_harmonize_debug:
                out.quality_summary["gmsh_transfinite_interface_debug"] = dict(transfinite_harmonize_debug)
            if has_interface_coincidence_diag:
                out.quality_summary["gmsh_interface_coincidence_report"] = {
                    "pair_count": int(len(interface_coincidence_report)),
                    "suspect_pair_count": int(len(interface_coincidence_suspects)),
                    "pairs": list(interface_coincidence_report),
                }
            self._last_flow_align_diagnostics = list(flow_align_diagnostics)
        return _mh()._repair_mesh_result(out)



# Lazy re-export of helper functions from meshing to avoid circular import.
# These are resolved on first access, after all modules finish loading.
_MESHING_HELPERS_CACHE = None

def __getattr__(name):
    global _MESHING_HELPERS_CACHE
    # Names that should be resolved from the meshing module
    _MESHING_HELPER_NAMES = {
        "_as_float", "_apply_optional_post_optimization",
        "_bbox_from_ring", "_breakline_fixed_edges_for_region",
        "_cell_overlaps_ring", "_constraints_for_region",
        "_env_bool", "_env_csv_floats", "_env_csv_strings", "_env_float",
        "_gmsh_cpp_prebuild_enabled",
        "_harmonize_transfinite_shared_quad_interfaces",
        "_interp_polyline_fraction", "_iter_qgis_polygon_parts",
        "_junction_points_on_interface", "_load_hydra_meshing_native",
        "_mixed_transfinite_tri_near_unshared_report",
        "_normalize_cell_type", "_normalize_conceptual_model_to_local_origin",
        "_point_in_polygon", "_point_to_segment_projection",
        "_polyline_length", "_polyline_distance_and_s",
        "_polyline_overlap_fractions_open",
        "_quad_controls_for_region", "_region_exclusion_zones",
        "_region_node_sets_from_mesh", "_region_boundary_node_sets_from_mesh",
        "_relax_fixed_edges_and_hints",
        "_repair_mesh_result", "_require_nonempty_mesh",
        "_sanitize_closed_ring",
        "_gmsh_flow_align_region_preflight",
        "_gmsh_flow_aligned_curve_counts",
        "_gmsh_interface_coincidence_report",
        "_enforce_quad_interface_conformance",
        "_interface_overlap_metrics",
        "_ring_centroid_xy", "_ring_from_quad_controls",
        "_ring_intersection_tolerance", "_ring_key",
        "_sample_closed_polyline", "_sample_polyline",
        "_segment_intersection_point",
        "_snap_and_split_boundary_for_breaklines",
        "_split_polyline_at_focus_points",
        "_stitch_boundary_microchains",
        "_write_json_atomic", "_write_mesh_checkpoint_npz",
        "_orient_quad_edge_chains", "_densify_polyline_subset",
        "_restore_mesh_coordinates",
    }
    if name in _MESHING_HELPER_NAMES:
        if _MESHING_HELPERS_CACHE is None:
            from swe2d.mesh import meshing as _m
            _MESHING_HELPERS_CACHE = _m
        return getattr(_MESHING_HELPERS_CACHE, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
