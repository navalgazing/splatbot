from __future__ import annotations

import json
import math
import os
import re
import shlex
import shutil
import struct
import time
import zlib
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from statistics import median
from typing import Awaitable, Callable

from .commands import CommandRunner, render_argv_template
from .config import ScanMode, ScanPreset, ScanPresetConfig, Settings
from .models import JobStatus, MediaItem, MediaKind


StatusCallback = Callable[[str, JobStatus], Awaitable[None]]
MAX_PLY_HEADER_BYTES = 64 * 1024
EXPORT_RETENTION_RE = re.compile(r"only export\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class PipelineOutputs:
    cleaned_ply: Path
    preview_mp4: Path | None
    metrics_path: Path | None = None
    mesh_path: Path | None = None
    quality_report_path: Path | None = None
    candidate_report_path: Path | None = None


@dataclass(frozen=True)
class FrameQuality:
    path: Path
    index: int
    score: float
    blur: float | None = None
    contrast: float | None = None
    brightness: float | None = None
    overexposed_ratio: float | None = None
    underexposed_ratio: float | None = None
    difference_from_previous: float | None = None
    signature: tuple[float, ...] | None = None
    reject_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class FrameSelectionResult:
    selected_count: int
    metrics: dict


@dataclass(frozen=True)
class ReconstructionArtifacts:
    raw_ply: Path
    cleaned_ply: Path
    cleanup: dict


@dataclass(frozen=True)
class ObjectMaskProfile:
    path: Path
    stem: str
    width: int
    height: int
    foreground_pixels: int
    area_ratio: float
    edge_touch_ratio: float
    bbox_fill_ratio: float | None
    center_x: float | None
    center_y: float | None
    reject_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlyLayout:
    format: str
    header_lines: list[str]
    header_bytes: int
    vertex_count: int
    vertex_line_index: int
    properties: list[tuple[str, str]]
    row_size: int | None = None


@dataclass(frozen=True)
class AlphaMask:
    width: int
    height: int
    alpha: bytes

    def is_foreground(self, u: float, v: float, threshold: int, padding: int) -> bool:
        x = round(u)
        y = round(v)
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return False
        x0 = max(0, x - padding)
        x1 = min(self.width - 1, x + padding)
        y0 = max(0, y - padding)
        y1 = min(self.height - 1, y + padding)
        for sample_y in range(y0, y1 + 1):
            row = sample_y * self.width
            for sample_x in range(x0, x1 + 1):
                if self.alpha[row + sample_x] >= threshold:
                    return True
        return False


@dataclass(frozen=True)
class SilhouetteFrame:
    mask: AlphaMask
    world_to_camera: list[list[float]]
    fl_x: float
    fl_y: float
    cx: float
    cy: float
    depth_path: Path | None = None


@dataclass(frozen=True)
class DepthConsistencyFrame:
    frame: SilhouetteFrame
    depth_map: object
    scale: float
    scale_samples: int


@dataclass(frozen=True)
class PointMaskSupport:
    observed: int
    inside: int
    outside: int

    @property
    def inside_fraction(self) -> float:
        return self.inside / self.observed if self.observed else 0.0

    @property
    def outside_fraction(self) -> float:
        return self.outside / self.observed if self.observed else 0.0


@dataclass
class SilhouetteEvaluator:
    frames: list[SilhouetteFrame]
    alpha_threshold: int
    padding_px: int
    outside_ratio: float
    max_inside_views: int
    max_inside_ratio: float
    min_views: int
    checked_points: int = 0
    removed_points: int = 0
    total_observations: int = 0

    def keep(self, values: tuple[float, ...]) -> bool:
        if len(values) < 3:
            return True
        support = point_mask_support(
            values[:3],
            self.frames,
            alpha_threshold=self.alpha_threshold,
            padding_px=self.padding_px,
        )
        if support.observed:
            self.checked_points += 1
            self.total_observations += support.observed
        if support.observed < self.min_views:
            return True
        inside_is_low = support.inside <= self.max_inside_views or support.inside_fraction <= self.max_inside_ratio
        remove = support.outside_fraction >= self.outside_ratio and inside_is_low
        if remove:
            self.removed_points += 1
        return not remove


class ScanPipeline:
    def __init__(self, settings: Settings, runner: CommandRunner | None = None) -> None:
        self.settings = settings
        self.runner = runner or CommandRunner(
            timeout_seconds=settings.command_timeout_seconds,
            tail_bytes=settings.command_tail_bytes,
        )

    async def run(
        self,
        job_id: str,
        mode: ScanMode,
        media: list[MediaItem],
        on_status: StatusCallback | None = None,
        preset: ScanPreset | None = None,
    ) -> PipelineOutputs:
        preset_config = self.settings.preset_config(preset)
        job_dir = self.settings.job_dir(job_id)
        images_dir = job_dir / "images"
        candidate_dir = job_dir / "candidate_frames"
        processed_dir = job_dir / "processed"
        ns_dir = job_dir / "nerfstudio"
        export_dir = job_dir / "export"
        render_dir = job_dir / "renders"
        metrics_path = job_dir / "metrics.json"
        settings_path = job_dir / "settings.json"
        metrics: dict = {
            "job_id": job_id,
            "mode": mode.value,
            "preset": preset_config.preset.value,
            "stages": {},
            "video": {},
            "colmap": {},
            "ply": {},
            "pipeline_events": [],
        }
        if self.settings.matrix_run_metadata:
            try:
                metrics["matrix_run"] = json.loads(self.settings.matrix_run_metadata)
            except json.JSONDecodeError:
                metrics["matrix_run"] = {"raw": self.settings.matrix_run_metadata}
        write_json(
            settings_path,
            {
                "mode": mode.value,
                "preset": preset_config.preset.value,
                "matrix_run": metrics.get("matrix_run"),
                "max_video_frames": preset_config.max_video_frames,
                "max_video_candidate_fps": self.settings.max_video_candidate_fps,
                "adaptive_frame_selection": preset_config.adaptive_frame_selection,
                "frame_selection_strategy": frame_selection_strategy_for_preset(self.settings, preset_config),
                "frame_quality_reject_threshold": self.settings.frame_quality_reject_threshold,
                "blur_reject_threshold": self.settings.blur_reject_threshold,
                "low_contrast_reject_threshold": self.settings.low_contrast_reject_threshold,
                "overexposed_reject_threshold": self.settings.overexposed_reject_threshold,
                "underexposed_reject_threshold": self.settings.underexposed_reject_threshold,
                "duplicate_frame_threshold": self.settings.duplicate_frame_threshold,
                "colmap_retry_frame_counts": parse_int_list(self.settings.colmap_retry_frame_counts),
                "colmap_retry_matching_methods": parse_csv_list(self.settings.colmap_retry_matching_methods),
                "segmentation_backend": self.settings.segmentation_backend,
                "best_segmentation_backends": self.settings.best_segmentation_backends,
                "experimental_sam3_enabled": self.settings.experimental_sam3_enabled,
                "segmentation_min_output_ratio": self.settings.segmentation_min_output_ratio,
                "object_mask_backend": self.settings.object_mask_backend,
                "object_mask_qa_enabled": self.settings.object_mask_qa_enabled,
                "object_mask_refine_enabled": self.settings.object_mask_refine_enabled,
                "pose_backends": parse_csv_list(self.settings.pose_backends),
                "best_pose_backends": parse_csv_list(self.settings.best_pose_backends),
                "train_method": preset_config.train_method,
                "train_backends": parse_csv_list(self.settings.train_backends),
                "best_train_backends": parse_csv_list(self.settings.best_train_backends),
                "experimental_dn_splatter_enabled": self.settings.experimental_dn_splatter_enabled,
                "depth_backends": parse_csv_list(self.settings.depth_backends),
                "best_depth_backends": parse_csv_list(self.settings.best_depth_backends),
                "train_max_iterations": preset_config.train_max_iterations,
                "train_steps_per_save": preset_config.train_steps_per_save,
                "train_extra_args": list(preset_config.train_extra_args),
                "mesh_export_enabled": self.settings.mesh_export_enabled,
                "mesh_backend": self.settings.mesh_backend,
                "colmap_use_gpu": self.settings.colmap_use_gpu,
                "colmap_global_calibrate": self.settings.colmap_global_calibrate,
                "colmap_bin": self.settings.colmap_bin,
                "silhouette_cleanup": {
                    "max_views": self.settings.silhouette_cleanup_max_views,
                    "outside_ratio": self.settings.silhouette_cleanup_outside_ratio,
                    "max_inside_views": self.settings.silhouette_cleanup_max_inside_views,
                    "max_inside_ratio": self.settings.silhouette_cleanup_max_inside_ratio,
                },
            },
        )
        for path in (images_dir, candidate_dir, processed_dir, ns_dir, export_dir, render_dir):
            path.mkdir(parents=True, exist_ok=True)

        is_video = len(media) == 1 and media[0].kind == MediaKind.VIDEO
        try:
            if on_status:
                await on_status(job_id, JobStatus.PREPROCESSING)
            stage_start = time.perf_counter()
            if is_video:
                await self.extract_video_frames(Path(media[0].local_path), images_dir, candidate_dir, preset_config, metrics)
            else:
                await self.copy_or_link_images(media, images_dir)
            record_stage(metrics, "preprocessing", stage_start)
            metrics["frames"] = {"selected": count_files(images_dir), "source_media": len(media)}
            write_json(metrics_path, metrics)
        except Exception:
            write_json(metrics_path, metrics)
            raise
        log_directory_summary("image frames", images_dir)

        input_images_dir = images_dir
        object_dir: Path | None = None
        if mode == ScanMode.OBJECT:
            object_dir = job_dir / "object_images"
            object_dir.mkdir(parents=True, exist_ok=True)
            stage_start = time.perf_counter()
            ensure_required_segmentation_backends_configured(self.settings, preset_config)
            mask_backend = await self.remove_backgrounds(
                images_dir,
                object_dir,
                job_dir,
                preset_config,
                metrics,
                metrics_path,
            )
            refinement = refine_object_masks(object_dir, self.settings)
            record_stage(metrics, "rembg", stage_start)
            input_images_dir = object_dir
            metrics["frames"]["object"] = count_files(object_dir)
            metrics.setdefault("masks", {})["backend"] = mask_backend
            if refinement:
                metrics.setdefault("masks", {})["refinement"] = refinement
            mask_qa = mask_backend.get("qa") or apply_object_mask_qa(
                images_dir,
                object_dir,
                job_dir,
                self.settings,
                preset_config,
            )
            metrics.setdefault("masks", {})["qa"] = mask_qa
            if mask_qa.get("applied"):
                images_dir = Path(str(mask_qa["original_images_dir"]))
                object_dir = Path(str(mask_qa["object_images_dir"]))
                input_images_dir = object_dir
                metrics["frames"]["pre_mask_qa_selected"] = metrics["frames"]["selected"]
                metrics["frames"]["selected"] = count_files(images_dir)
                metrics["frames"]["object"] = count_files(object_dir)
            write_json(metrics_path, metrics)
            log_directory_summary("object images", object_dir)

        if on_status:
            await on_status(job_id, JobStatus.COLMAP)
        stage_start = time.perf_counter()
        ensure_required_pose_backends_configured(self.settings, preset_config)
        await self.process_data_with_quality_gate(
            input_images_dir=input_images_dir,
            processed_dir=processed_dir,
            matching_method="sequential" if is_video else None,
            metrics=metrics,
            metrics_path=metrics_path,
            preset=preset_config,
            mode=mode,
            original_images_dir=images_dir,
            object_images_dir=object_dir,
        )
        record_stage(metrics, "colmap", stage_start)
        if mode == ScanMode.OBJECT:
            mask_paths = write_processed_training_masks(
                processed_dir,
                alpha_threshold=self.settings.object_mask_training_alpha_threshold,
            )
            metrics.setdefault("masks", {})["training_mask_paths"] = mask_paths
        write_json(metrics_path, metrics)
        log_directory_summary("processed data", processed_dir)
        depth_backends = configured_depth_backends(self.settings, preset_config)
        if depth_backends:
            stage_start = time.perf_counter()
            ensure_required_depth_backends_configured(self.settings, preset_config)
            await self.prepare_depth_priors(
                processed_dir=processed_dir,
                images_dir=input_images_dir,
                preset=preset_config,
                metrics=metrics,
                metrics_path=metrics_path,
            )
            record_stage(metrics, "depth", stage_start)
            write_json(metrics_path, metrics)
        ensure_required_train_backends_configured(self.settings, preset_config)
        if on_status:
            await on_status(job_id, JobStatus.TRAINING)
        reconstruction = await self.train_export_reconstruction(
            processed_dir=processed_dir,
            ns_dir=ns_dir,
            export_dir=export_dir,
            mode=mode,
            preset=preset_config,
            metrics=metrics,
            metrics_path=metrics_path,
            job_id=job_id,
            on_status=on_status,
        )
        log_directory_summary("nerfstudio outputs", ns_dir)
        log_ply_summary("raw splat", reconstruction.raw_ply)
        log_ply_summary("cleaned splat", reconstruction.cleaned_ply)
        mesh_path = None
        mesh_metrics = {"enabled": False, "reason": "disabled"}
        if self.settings.mesh_export_enabled:
            stage_start = time.perf_counter()
            mesh_path, mesh_metrics = await self.export_mesh(
                processed_dir,
                ns_dir,
                export_dir,
                reconstruction.cleaned_ply,
            )
            record_stage(metrics, "mesh", stage_start)
            metrics["mesh"] = mesh_metrics
            if mesh_metrics.get("applied") is False:
                record_pipeline_event(
                    metrics,
                    stage="mesh",
                    backend=str(mesh_metrics.get("backend") or self.settings.mesh_backend),
                    status="fallback",
                    reason=str(mesh_metrics.get("reason") or "mesh_not_exported"),
                    error=mesh_metrics.get("error"),
                    recovered=not self.settings.mesh_export_required,
                )
            write_json(metrics_path, metrics)
        preview_mp4 = None
        if self.settings.render_preview:
            if on_status:
                await on_status(job_id, JobStatus.RENDERING)
            stage_start = time.perf_counter()
            preview_mp4 = await self.render_turntable(ns_dir, render_dir)
            record_stage(metrics, "rendering", stage_start)
            write_json(metrics_path, metrics)
        quality_report_path = None
        candidate_report_path = None
        if self.settings.quality_report_enabled:
            quality_report_path = job_dir / "quality_report.json"
            candidate_report_path = job_dir / "candidate_report.json"
            write_json(quality_report_path, build_quality_report(metrics, self.settings))
            write_json(candidate_report_path, build_candidate_report(metrics, self.settings))
        return PipelineOutputs(
            cleaned_ply=reconstruction.cleaned_ply,
            preview_mp4=preview_mp4,
            metrics_path=metrics_path,
            mesh_path=mesh_path,
            quality_report_path=quality_report_path,
            candidate_report_path=candidate_report_path,
        )

    async def extract_video_frames(
        self,
        video: Path,
        images_dir: Path,
        candidate_dir: Path,
        preset: ScanPresetConfig,
        metrics: dict,
    ) -> None:
        if preset.adaptive_frame_selection:
            extraction_metrics = await self.extract_candidate_video_frames(video, candidate_dir)
            selection = select_video_frames(
                sorted(candidate_dir.glob("frame_*.jpg")),
                images_dir,
                target_count=preset.max_video_frames,
                min_count=min(self.settings.min_selected_video_frames, preset.max_video_frames),
                quality_threshold=self.settings.frame_quality_reject_threshold,
                blur_threshold=self.settings.blur_reject_threshold,
                low_contrast_threshold=self.settings.low_contrast_reject_threshold,
                overexposed_threshold=self.settings.overexposed_reject_threshold,
                underexposed_threshold=self.settings.underexposed_reject_threshold,
                duplicate_threshold=self.settings.duplicate_frame_threshold,
                strategy=frame_selection_strategy_for_preset(self.settings, preset),
            )
            metrics["video"] = {
                **metrics.get("video", {}),
                "adaptive_frame_selection": True,
                **extraction_metrics,
                **selection.metrics,
            }
            if selection.metrics.get("quality_selection_fallback"):
                record_pipeline_event(
                    metrics,
                    stage="preprocessing",
                    status="fallback",
                    reason="too_few_high_quality_frames",
                    recovered=True,
                    details={
                        "accepted_frame_candidates": selection.metrics.get("accepted_frame_candidates"),
                        "selected_frames": selection.metrics.get("selected_frames"),
                        "min_selected_video_frames": min(
                            self.settings.min_selected_video_frames,
                            preset.max_video_frames,
                        ),
                    },
                )
            return
        fps = await self.video_sample_fps(video, preset.max_video_frames)
        await self.runner.run(
            [
                self.settings.ffmpeg_bin,
                "-i",
                str(video),
                "-t",
                str(self.settings.max_video_seconds),
                "-vf",
                f"fps={format_fps(fps)}",
                "-q:v",
                "2",
                str(images_dir / "frame_%05d.jpg"),
            ]
        )
        metrics["video"] = {
            **metrics.get("video", {}),
            "adaptive_frame_selection": False,
            "selected_frames": count_files(images_dir),
        }

    async def extract_candidate_video_frames(
        self,
        video: Path,
        candidate_dir: Path,
    ) -> dict:
        source_fps = await self.video_frame_rate(video)
        candidate_fps: float | None = None
        sampling = "native"
        if source_fps is None:
            sampling = "capped_unknown_source_fps"
            candidate_fps = self.settings.max_video_candidate_fps
        elif source_fps > self.settings.max_video_candidate_fps:
            sampling = "capped"
            candidate_fps = self.settings.max_video_candidate_fps

        argv = [
            self.settings.ffmpeg_bin,
            "-i",
            str(video),
            "-t",
            str(self.settings.max_video_seconds),
        ]
        if candidate_fps is not None:
            argv.extend(["-vf", f"fps={format_fps(candidate_fps)}"])
        argv.extend(
            [
                "-q:v",
                "2",
                str(candidate_dir / "frame_%05d.jpg"),
            ]
        )
        await self.runner.run(argv)
        return {
            "candidate_sampling": sampling,
            "source_fps": round(source_fps, 3) if source_fps is not None else None,
            "candidate_sample_fps": round(candidate_fps or source_fps, 3)
            if (candidate_fps or source_fps) is not None
            else None,
        }

    async def video_sample_fps(self, video: Path, target_frames: int | None = None) -> float:
        result = await self.runner.run(
            [
                self.settings.ffprobe_bin,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video),
            ]
        )
        duration = parse_ffprobe_duration(result.stdout)
        target = target_frames or self.settings.max_video_frames
        if duration is None:
            return target / self.settings.max_video_seconds
        sampled_seconds = max(1.0, min(duration, float(self.settings.max_video_seconds)))
        return min(
            target / sampled_seconds,
            self.settings.max_video_sample_fps,
        )

    async def video_frame_rate(self, video: Path) -> float | None:
        result = await self.runner.run(
            [
                self.settings.ffprobe_bin,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=avg_frame_rate,r_frame_rate",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video),
            ]
        )
        return parse_ffprobe_frame_rate(result.stdout)

    async def copy_or_link_images(self, media: list[MediaItem], images_dir: Path) -> None:
        for idx, item in enumerate(media, start=1):
            src = Path(item.local_path)
            dest = images_dir / f"image_{idx:05d}{src.suffix.lower() or '.jpg'}"
            if dest.exists():
                dest.unlink()
            dest.symlink_to(src)

    async def remove_backgrounds(
        self,
        images_dir: Path,
        object_dir: Path,
        job_dir: Path,
        preset: ScanPresetConfig,
        metrics: dict | None = None,
        metrics_path: Path | None = None,
    ) -> dict:
        attempts: list[dict] = []
        input_count = count_files(images_dir)
        required_outputs = minimum_segmentation_outputs(self.settings, input_count)
        failures_before_success = 0
        required_backends = set(required_segmentation_backends(self.settings, preset))
        for backend in configured_segmentation_backends(self.settings, preset):
            if object_dir.exists():
                shutil.rmtree(object_dir)
            object_dir.mkdir(parents=True, exist_ok=True)
            try:
                command = object_mask_command_for_backend(self.settings, backend, images_dir, object_dir)
                if command is None:
                    attempts.append({"backend": backend, "applied": False, "reason": "missing_command"})
                    failures_before_success += 1
                    recovered = backend not in required_backends
                    if metrics is not None:
                        record_pipeline_event(
                            metrics,
                            stage="segmentation",
                            backend=backend,
                            status="skip" if recovered else "failure",
                            reason="missing_command",
                            recovered=recovered,
                        )
                        if metrics_path is not None:
                            write_json(metrics_path, metrics)
                    if backend in required_backends:
                        raise ValueError(f"required best segmentation backend {backend!r} is missing a command")
                    continue
                await self.runner.run(command)
                output_count = count_files(object_dir)
                accepted = output_count >= required_outputs
                attempt = {
                    "backend": backend,
                    "applied": accepted,
                    "output_files": output_count,
                    "required_output_files": required_outputs,
                    "input_files": input_count,
                }
                if not accepted:
                    attempt["reason"] = "too_few_outputs"
                attempts.append(attempt)
                if accepted:
                    qa = apply_object_mask_qa(images_dir, object_dir, job_dir, self.settings, preset)
                    attempt["qa"] = {
                        "applied": qa.get("applied"),
                        "reason": qa.get("reason"),
                        "accepted_masks": qa.get("accepted_masks"),
                        "total_masks": qa.get("total_masks"),
                    }
                    if qa.get("reason") in {"no_masks", "too_few_accepted_masks"}:
                        attempt["applied"] = False
                        attempt["reason"] = qa.get("reason")
                        failures_before_success += 1
                        recovered = backend not in required_backends
                        if metrics is not None:
                            record_pipeline_event(
                                metrics,
                                stage="segmentation",
                                backend=backend,
                                status="fallback" if recovered else "failure",
                                reason=str(qa.get("reason")),
                                recovered=recovered,
                                details=attempt,
                            )
                            if metrics_path is not None:
                                write_json(metrics_path, metrics)
                        if backend in required_backends:
                            raise ValueError(
                                f"required best segmentation backend {backend!r} failed mask QA: {qa.get('reason')}"
                            )
                        continue
                    if failures_before_success and metrics is not None:
                        record_pipeline_event(
                            metrics,
                            stage="segmentation",
                            backend=backend,
                            status="recovered",
                            reason="selected_after_previous_backend_failure",
                            recovered=True,
                            details={"attempts": len(attempts)},
                        )
                        if metrics_path is not None:
                            write_json(metrics_path, metrics)
                    return {"selected": backend, "attempts": attempts, "qa": qa}
                failures_before_success += 1
                recovered = backend not in required_backends
                if metrics is not None:
                    record_pipeline_event(
                        metrics,
                        stage="segmentation",
                        backend=backend,
                        status="fallback" if recovered else "failure",
                        reason="too_few_outputs",
                        recovered=recovered,
                        details=attempt,
                    )
                    if metrics_path is not None:
                        write_json(metrics_path, metrics)
                if backend in required_backends:
                    raise ValueError(f"required best segmentation backend {backend!r} produced too few outputs")
            except Exception as exc:  # noqa: BLE001
                attempts.append({"backend": backend, "applied": False, "error": str(exc)})
                failures_before_success += 1
                recovered = backend not in required_backends
                if metrics is not None:
                    record_pipeline_event(
                        metrics,
                        stage="segmentation",
                        backend=backend,
                        status="fallback" if recovered else "failure",
                        reason="backend_failed",
                        error=str(exc),
                        recovered=recovered,
                    )
                    if metrics_path is not None:
                        write_json(metrics_path, metrics)
                if backend in required_backends:
                    raise
        detail = "; ".join(
            f"{attempt['backend']}: {attempt.get('error') or attempt.get('reason') or 'no outputs'}"
            for attempt in attempts
        )
        if metrics is not None:
            record_pipeline_event(
                metrics,
                stage="segmentation",
                status="failure",
                reason="all_backends_failed",
                recovered=False,
                details={"attempts": attempts},
            )
            if metrics_path is not None:
                write_json(metrics_path, metrics)
        raise ValueError(f"all object segmentation backends failed ({detail})")

    async def process_data(
        self,
        images_dir: Path,
        processed_dir: Path,
        matching_method: str | None = None,
        mapper: str | None = None,
    ) -> None:
        argv = [
            self.settings.ns_process_data_bin,
            "images",
            "--data",
            str(images_dir),
            "--output-dir",
            str(processed_dir),
        ]
        if matching_method:
            argv.extend(["--matching-method", matching_method])
        if self.settings.colmap_bin != "colmap":
            argv.extend(["--colmap-cmd", self.settings.colmap_bin])
        if not self.settings.colmap_use_gpu:
            argv.append("--no-gpu")
        env = None
        if mapper:
            env = os.environ.copy()
            env["SPLATBOT_COLMAP_MAPPER"] = mapper
        if env is None:
            await self.runner.run(argv)
        else:
            await self.runner.run(argv, env=env)

    async def process_data_external_pose_backend(
        self,
        backend: str,
        input_images_dir: Path,
        processed_dir: Path,
        matching_method: str | None,
    ) -> None:
        command = pose_command_for_backend(self.settings, backend)
        if not command:
            raise ValueError(
                f"SPLATBOT_POSE_BACKEND_COMMAND is required for pose backend {backend!r}"
            )
        argv = render_argv_template(
            command,
            {
                "backend": backend,
                "input_dir": str(input_images_dir),
                "images_dir": str(input_images_dir),
                "output_dir": str(processed_dir),
                "processed_dir": str(processed_dir),
                "matching_method": matching_method or "",
                "colmap_bin": self.settings.colmap_bin,
                "glomap_bin": self.settings.glomap_bin,
            },
        )
        await self.runner.run(argv)

    async def process_data_with_quality_gate(
        self,
        input_images_dir: Path,
        processed_dir: Path,
        matching_method: str | None,
        metrics: dict,
        metrics_path: Path,
        preset: ScanPresetConfig,
        mode: ScanMode,
        original_images_dir: Path,
        object_images_dir: Path | None,
    ) -> None:
        errors: list[str] = []
        required_backends = set(required_pose_backends(self.settings, preset))
        for backend in configured_pose_backends(self.settings, preset):
            metrics.setdefault("pose_backend_attempts", []).append(
                {"backend": backend, "input_dir": str(input_images_dir)}
            )
            try:
                if is_colmap_pose_backend(backend):
                    colmap_matching_method = colmap_matching_method_for_pose_backend(backend, matching_method)
                    colmap_mapper = colmap_mapper_for_pose_backend(backend)
                    await self.process_data_colmap_with_quality_gate(
                        input_images_dir=input_images_dir,
                        processed_dir=processed_dir,
                        matching_method=colmap_matching_method,
                        mapper=colmap_mapper,
                        metrics=metrics,
                        metrics_path=metrics_path,
                        preset=preset,
                        mode=mode,
                        original_images_dir=original_images_dir,
                        object_images_dir=object_images_dir,
                    )
                else:
                    await self.process_data_external_pose_backend(
                        backend=backend,
                        input_images_dir=input_images_dir,
                        processed_dir=processed_dir,
                        matching_method=matching_method,
                    )
                    record_colmap_attempt(
                        metrics,
                        source=f"pose:{backend}",
                        input_dir=input_images_dir,
                        frame_count=count_files(input_images_dir),
                        matching_method=matching_method,
                        result=inspect_processed_dataset(processed_dir),
                    )
                    metrics["colmap"] = metrics["colmap_attempts"][-1]["result"]
                    validate_colmap_quality(metrics, preset.max_video_frames, self.settings)
                metrics["pose_backend"] = backend
                if errors:
                    record_pipeline_event(
                        metrics,
                        stage="pose",
                        backend=backend,
                        status="recovered",
                        reason="selected_after_previous_backend_failure",
                        recovered=True,
                        details={"failed_backends": len(errors)},
                    )
                write_json(metrics_path, metrics)
                return
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{backend}: {exc}")
                metrics.setdefault("pose_backend_failures", []).append({"backend": backend, "error": str(exc)})
                recovered = backend not in required_backends
                record_pipeline_event(
                    metrics,
                    stage="pose",
                    backend=backend,
                    status="fallback" if recovered else "failure",
                    reason="backend_failed",
                    error=str(exc),
                    recovered=recovered,
                )
                write_json(metrics_path, metrics)
                if backend in required_backends:
                    raise
                if processed_dir.exists():
                    shutil.rmtree(processed_dir)
                processed_dir.mkdir(parents=True, exist_ok=True)
        record_pipeline_event(
            metrics,
            stage="pose",
            status="failure",
            reason="all_backends_failed",
            recovered=False,
            details={"errors": errors},
        )
        write_json(metrics_path, metrics)
        raise ValueError("all pose backends failed: " + " | ".join(errors))

    async def process_data_colmap_with_quality_gate(
        self,
        input_images_dir: Path,
        processed_dir: Path,
        matching_method: str | None,
        mapper: str | None,
        metrics: dict,
        metrics_path: Path,
        preset: ScanPresetConfig,
        mode: ScanMode,
        original_images_dir: Path,
        object_images_dir: Path | None,
    ) -> None:
        await self.process_data(input_images_dir, processed_dir, matching_method=matching_method, mapper=mapper)
        selected_frames = metrics.get("frames", {}).get("selected") or preset.max_video_frames
        input_source_label = (
            "object" if mode == ScanMode.OBJECT and input_images_dir == object_images_dir else "original"
        )
        record_colmap_attempt(
            metrics,
            source=input_source_label,
            input_dir=input_images_dir,
            frame_count=count_files(input_images_dir),
            matching_method=matching_method,
            result=inspect_processed_dataset(processed_dir),
        )
        metrics["colmap"] = metrics["colmap_attempts"][-1]["result"]
        write_json(metrics_path, metrics)
        masked_error_message = ""
        try:
            validate_colmap_quality(metrics, preset.max_video_frames, self.settings)
            return
        except ValueError as masked_error:
            masked_error_message = str(masked_error)
            if (
                mode != ScanMode.OBJECT
                or object_images_dir is None
                or input_images_dir != object_images_dir
                or not self.settings.object_colmap_original_pose_fallback
            ):
                if await self.retry_colmap_with_subsets(
                    source_images_dir=input_images_dir,
                    processed_dir=processed_dir,
                    matching_method=matching_method,
                    metrics=metrics,
                    metrics_path=metrics_path,
                    target_frames=preset.max_video_frames,
                    selected_frames=selected_frames,
                    source_label=input_source_label,
                    final_object_images_dir=None,
                    mapper=mapper,
                ):
                    return
                raise

        metrics["colmap_masked"] = metrics["colmap"]
        metrics["colmap_fallback"] = {
            "reason": masked_error_message,
            "pose_images": "original",
            "training_images": "object",
        }
        record_pipeline_event(
            metrics,
            stage="pose",
            backend="object_colmap_original_pose_fallback",
            status="fallback",
            reason=masked_error_message,
            recovered=True,
        )
        write_json(metrics_path, metrics)

        if processed_dir.exists():
            shutil.rmtree(processed_dir)
        processed_dir.mkdir(parents=True, exist_ok=True)
        await self.process_data(original_images_dir, processed_dir, matching_method=matching_method, mapper=mapper)
        record_colmap_attempt(
            metrics,
            source="original",
            input_dir=original_images_dir,
            frame_count=count_files(original_images_dir),
            matching_method=matching_method,
            result=inspect_processed_dataset(processed_dir),
        )
        metrics["colmap"] = metrics["colmap_attempts"][-1]["result"]
        write_json(metrics_path, metrics)
        try:
            validate_colmap_quality(metrics, preset.max_video_frames, self.settings)
        except ValueError as fallback_error:
            if not await self.retry_colmap_with_subsets(
                source_images_dir=original_images_dir,
                processed_dir=processed_dir,
                matching_method=matching_method,
                metrics=metrics,
                metrics_path=metrics_path,
                target_frames=preset.max_video_frames,
                selected_frames=selected_frames,
                source_label="original",
                final_object_images_dir=object_images_dir,
                mapper=mapper,
            ):
                raise ValueError(
                    f"{masked_error_message} Original-frame pose fallback also failed: {fallback_error}"
                ) from fallback_error
            return
        replaced = replace_processed_images_with_object_images(
            processed_dir,
            object_images_dir,
            allowed_stems=processed_frame_stems(processed_dir),
        )
        sparse_cleanup = clear_colmap_sparse_points(processed_dir)
        metrics["colmap_fallback"] = {
            **metrics["colmap_fallback"],
            "applied": True,
            "training_image_paths_rewritten": replaced,
            "sparse_points_removed": sparse_cleanup,
        }
        write_json(metrics_path, metrics)

    async def retry_colmap_with_subsets(
        self,
        source_images_dir: Path,
        processed_dir: Path,
        matching_method: str | None,
        metrics: dict,
        metrics_path: Path,
        target_frames: int,
        selected_frames: int,
        source_label: str,
        final_object_images_dir: Path | None,
        mapper: str | None = None,
    ) -> bool:
        image_files = sorted(
            path
            for path in source_images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not image_files:
            return False
        frame_counts = [
            count
            for count in parse_int_list(self.settings.colmap_retry_frame_counts)
            if 10 <= count < len(image_files)
        ]
        matching_methods = parse_csv_list(self.settings.colmap_retry_matching_methods)
        if not frame_counts or not matching_methods:
            return False

        attempt_root = processed_dir.parent / "colmap_retry_inputs"
        for frame_count in frame_counts:
            subset_dir = attempt_root / f"{source_label}_{frame_count:03d}"
            build_even_subset(image_files, subset_dir, frame_count)
            for retry_matching in matching_methods:
                effective_matching = None if retry_matching == "default" else retry_matching
                if effective_matching == matching_method and frame_count == len(image_files):
                    continue
                if processed_dir.exists():
                    shutil.rmtree(processed_dir)
                processed_dir.mkdir(parents=True, exist_ok=True)
                await self.process_data(
                    subset_dir,
                    processed_dir,
                    matching_method=effective_matching,
                    mapper=mapper,
                )
                result = inspect_processed_dataset(processed_dir)
                record_colmap_attempt(
                    metrics,
                    source=source_label,
                    input_dir=subset_dir,
                    frame_count=frame_count,
                    matching_method=effective_matching,
                    result=result,
                    retry=True,
                )
                metrics["colmap"] = result
                write_json(metrics_path, metrics)
                try:
                    validate_colmap_quality(
                        {**metrics, "frames": {**metrics.get("frames", {}), "selected": frame_count}},
                        target_frames,
                        self.settings,
                    )
                except ValueError:
                    continue
                metrics["frames"]["selected_for_colmap"] = frame_count
                metrics["colmap_recovery"] = {
                    "applied": True,
                    "source": source_label,
                    "frame_count": frame_count,
                    "matching_method": effective_matching or "default",
                }
                record_pipeline_event(
                    metrics,
                    stage="pose",
                    backend="colmap_retry_subset",
                    status="recovered",
                    reason="registered_subset_passed_quality_gate",
                    recovered=True,
                    details={
                        "source": source_label,
                        "frame_count": frame_count,
                        "matching_method": effective_matching or "default",
                    },
                )
                if final_object_images_dir is not None:
                    replaced = replace_processed_images_with_object_images(
                        processed_dir,
                        final_object_images_dir,
                        allowed_stems=processed_frame_stems(processed_dir),
                    )
                    sparse_cleanup = clear_colmap_sparse_points(processed_dir)
                    metrics["colmap_fallback"] = {
                        **metrics.get("colmap_fallback", {}),
                        "applied": True,
                        "training_image_paths_rewritten": replaced,
                        "sparse_points_removed": sparse_cleanup,
                    }
                write_json(metrics_path, metrics)
                return True
        return False

    async def prepare_depth_priors(
        self,
        processed_dir: Path,
        images_dir: Path,
        preset: ScanPresetConfig,
        metrics: dict,
        metrics_path: Path,
    ) -> None:
        errors: list[str] = []
        required_backends = set(required_depth_backends(self.settings, preset))
        for backend in configured_depth_backends(self.settings, preset):
            try:
                if not depth_command_for_backend(self.settings, backend):
                    if backend in required_backends:
                        raise ValueError(f"required best depth backend {backend!r} is missing a command")
                    metrics.setdefault("depth_backend_skips", []).append(
                        {"backend": backend, "reason": "missing_command"}
                    )
                    record_pipeline_event(
                        metrics,
                        stage="depth",
                        backend=backend,
                        status="skip",
                        reason="missing_command",
                        recovered=True,
                    )
                    write_json(metrics_path, metrics)
                    continue
                await self.prepare_external_depth_backend(backend, processed_dir, images_dir)
                metrics["depth_backend"] = backend
                record_pipeline_event(
                    metrics,
                    stage="depth",
                    backend=backend,
                    status="success",
                    reason="depth_priors_created",
                    recovered=True,
                )
                write_json(metrics_path, metrics)
                return
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{backend}: {exc}")
                metrics.setdefault("depth_backend_failures", []).append({"backend": backend, "error": str(exc)})
                recovered = backend not in required_backends
                record_pipeline_event(
                    metrics,
                    stage="depth",
                    backend=backend,
                    status="fallback" if recovered else "failure",
                    reason="backend_failed_or_unavailable",
                    error=str(exc),
                    recovered=recovered,
                )
                write_json(metrics_path, metrics)
                if backend in required_backends:
                    raise
        if errors:
            metrics["depth_backend"] = None
            record_pipeline_event(
                metrics,
                stage="depth",
                status="skip",
                reason="all_depth_backends_unavailable",
                recovered=True,
                details={"errors": errors},
            )
            write_json(metrics_path, metrics)

    async def prepare_external_depth_backend(
        self,
        backend: str,
        processed_dir: Path,
        images_dir: Path,
    ) -> None:
        command = depth_command_for_backend(self.settings, backend)
        if not command:
            raise ValueError(f"depth backend {backend!r} is not configured")
        argv = render_argv_template(
            command,
            {
                "backend": backend,
                "processed_dir": str(processed_dir),
                "data_dir": str(processed_dir),
                "images_dir": str(images_dir),
                "input_dir": str(images_dir),
            },
        )
        await self.runner.run(argv)

    async def train_reconstruction(
        self,
        processed_dir: Path,
        ns_dir: Path,
        preset: ScanPresetConfig,
        metrics: dict,
        metrics_path: Path,
    ) -> None:
        errors: list[str] = []
        required_backends = set(required_train_backends(self.settings, preset))
        for backend in configured_train_backends(self.settings, preset):
            try:
                if backend in {"splatfacto", "splatfacto-big", preset.train_method.lower()}:
                    method = backend if backend.startswith("splatfacto") else preset.train_method
                    await self.train_splatfacto(processed_dir, ns_dir, preset, method=method)
                else:
                    await self.train_external_backend(backend, processed_dir, ns_dir, preset)
                metrics["train_backend"] = backend
                if errors:
                    record_pipeline_event(
                        metrics,
                        stage="training",
                        backend=backend,
                        status="recovered",
                        reason="selected_after_previous_backend_failure",
                        recovered=True,
                        details={"failed_backends": len(errors)},
                    )
                write_json(metrics_path, metrics)
                return
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{backend}: {exc}")
                metrics.setdefault("train_backend_failures", []).append({"backend": backend, "error": str(exc)})
                recovered = backend not in required_backends
                record_pipeline_event(
                    metrics,
                    stage="training",
                    backend=backend,
                    status="fallback" if recovered else "failure",
                    reason="backend_failed",
                    error=str(exc),
                    recovered=recovered,
                )
                write_json(metrics_path, metrics)
                if backend in required_backends:
                    raise
                if ns_dir.exists():
                    shutil.rmtree(ns_dir)
                ns_dir.mkdir(parents=True, exist_ok=True)
        record_pipeline_event(
            metrics,
            stage="training",
            status="failure",
            reason="all_backends_failed",
            recovered=False,
            details={"errors": errors},
        )
        write_json(metrics_path, metrics)
        raise ValueError("all train backends failed: " + " | ".join(errors))

    async def train_export_reconstruction(
        self,
        processed_dir: Path,
        ns_dir: Path,
        export_dir: Path,
        mode: ScanMode,
        preset: ScanPresetConfig,
        metrics: dict,
        metrics_path: Path,
        job_id: str,
        on_status: StatusCallback | None,
    ) -> ReconstructionArtifacts:
        errors: list[str] = []
        required_backends = set(required_train_backends(self.settings, preset))
        total_training_seconds = 0.0
        total_exporting_seconds = 0.0
        for backend in configured_train_backends(self.settings, preset):
            attempt: dict = {"backend": backend}
            metrics.setdefault("train_backend_attempts", []).append(attempt)
            attempt_stage = "training"
            if ns_dir.exists():
                shutil.rmtree(ns_dir)
            if export_dir.exists():
                shutil.rmtree(export_dir)
            ns_dir.mkdir(parents=True, exist_ok=True)
            export_dir.mkdir(parents=True, exist_ok=True)
            try:
                metrics["train_backend"] = backend
                training_started = time.perf_counter()
                if backend in {"splatfacto", "splatfacto-big", preset.train_method.lower()}:
                    method = backend if backend.startswith("splatfacto") else preset.train_method
                    await self.train_splatfacto(processed_dir, ns_dir, preset, method=method)
                else:
                    await self.train_external_backend(backend, processed_dir, ns_dir, preset)
                training_seconds = time.perf_counter() - training_started
                total_training_seconds += training_seconds
                attempt["training_seconds"] = round(training_seconds, 3)
                metrics.setdefault("stage_attempts", {}).setdefault("training", []).append(
                    {"backend": backend, "duration_seconds": round(training_seconds, 3)}
                )
                metrics.setdefault("stages", {})["training"] = {
                    "duration_seconds": round(total_training_seconds, 3)
                }
                write_json(metrics_path, metrics)

                if on_status:
                    await on_status(job_id, JobStatus.EXPORTING)
                attempt_stage = "exporting"
                exporting_started = time.perf_counter()
                raw_ply, export_metrics = await self.export_ply(ns_dir, export_dir)
                cleaned_ply = export_dir / "cleaned_splat.ply"
                attempt_stage = "postprocess"
                ply_cleanup = clean_exported_ply(raw_ply, cleaned_ply, processed_dir, mode, self.settings)
                exporting_seconds = time.perf_counter() - exporting_started
                total_exporting_seconds += exporting_seconds
                attempt["exporting_seconds"] = round(exporting_seconds, 3)
                metrics.setdefault("stage_attempts", {}).setdefault("exporting", []).append(
                    {"backend": backend, "duration_seconds": round(exporting_seconds, 3)}
                )
                metrics.setdefault("stages", {})["exporting"] = {
                    "duration_seconds": round(total_exporting_seconds, 3)
                }
                metrics["ply"] = {
                    "raw": inspect_ply(raw_ply),
                    "cleaned": inspect_ply(cleaned_ply),
                    "cleanup": ply_cleanup,
                }
                if export_metrics:
                    metrics["ply"]["export"] = export_metrics
                attempt["raw_vertices"] = metrics["ply"]["raw"].get("vertices")
                attempt["cleaned_vertices"] = metrics["ply"]["cleaned"].get("vertices")
                attempt_stage = "quality"
                validate_ply_quality(metrics, self.settings)
                attempt["passed_quality"] = True
                if errors:
                    record_pipeline_event(
                        metrics,
                        stage="training",
                        backend=backend,
                        status="recovered",
                        reason="selected_after_previous_backend_failure",
                        recovered=True,
                        details={"failed_backends": len(errors)},
                    )
                write_json(metrics_path, metrics)
                return ReconstructionArtifacts(raw_ply=raw_ply, cleaned_ply=cleaned_ply, cleanup=ply_cleanup)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{backend}: {exc}")
                attempt["passed_quality"] = False
                attempt["failed_stage"] = attempt_stage
                attempt["error"] = truncate_text(str(exc), 1200)
                metrics.setdefault("train_backend_failures", []).append(
                    {"backend": backend, "stage": attempt_stage, "error": str(exc)}
                )
                recovered = backend not in required_backends
                reason = {
                    "training": "backend_failed",
                    "exporting": "export_failed",
                    "postprocess": "postprocess_failed",
                    "quality": "output_quality_failed",
                }.get(attempt_stage, "backend_failed")
                record_pipeline_event(
                    metrics,
                    stage="training" if attempt_stage == "training" else attempt_stage,
                    backend=backend,
                    status="fallback" if recovered else "failure",
                    reason=reason,
                    error=str(exc),
                    recovered=recovered,
                )
                write_json(metrics_path, metrics)
                if backend in required_backends:
                    raise
                if on_status:
                    await on_status(job_id, JobStatus.TRAINING)
        record_pipeline_event(
            metrics,
            stage="training",
            status="failure",
            reason="all_backends_failed",
            recovered=False,
            details={"errors": errors},
        )
        write_json(metrics_path, metrics)
        raise ValueError("all train/export backends failed: " + " | ".join(errors))

    async def train_splatfacto(
        self,
        processed_dir: Path,
        ns_dir: Path,
        preset: ScanPresetConfig,
        method: str | None = None,
    ) -> None:
        argv = [
            self.settings.ns_train_bin,
            method or preset.train_method,
            "--data",
            str(processed_dir),
            "--output-dir",
            str(ns_dir),
            "--max-num-iterations",
            str(preset.train_max_iterations),
            "--steps-per-save",
            str(preset.train_steps_per_save),
            "--viewer.quit-on-train-completion",
            "True",
        ]
        argv.extend(preset.train_extra_args)
        await self.runner.run(argv)

    async def train_external_backend(
        self,
        backend: str,
        processed_dir: Path,
        ns_dir: Path,
        preset: ScanPresetConfig,
    ) -> None:
        command = train_command_for_backend(self.settings, backend)
        if not command:
            raise ValueError(
                f"SPLATBOT_TRAIN_BACKEND_COMMAND is required for train backend {backend!r}"
            )
        argv = render_argv_template(
            command,
            {
                "backend": backend,
                "processed_dir": str(processed_dir),
                "data_dir": str(processed_dir),
                "ns_dir": str(ns_dir),
                "output_dir": str(ns_dir),
                "max_iterations": preset.train_max_iterations,
                "steps_per_save": preset.train_steps_per_save,
                "extra_args": preset.train_extra_args,
            },
        )
        await self.runner.run(argv)

    async def export_ply(self, ns_dir: Path, export_dir: Path) -> tuple[Path, dict]:
        raw_ply = export_dir / "raw_splat.ply"
        result = await self.runner.run(
            [
                self.settings.ns_export_bin,
                "gaussian-splat",
                "--load-config",
                str(latest_nerfstudio_config(ns_dir)),
                "--output-dir",
                str(export_dir),
                "--output-filename",
                raw_ply.name,
            ]
        )
        return raw_ply, parse_export_metrics(result.stdout, result.stderr)

    async def render_turntable(self, ns_dir: Path, render_dir: Path) -> Path:
        preview = render_dir / "turntable.mp4"
        await self.runner.run(
            [
                self.settings.ns_render_bin,
                "spiral",
                "--load-config",
                str(latest_nerfstudio_config(ns_dir)),
                "--output-path",
                str(preview),
                "--seconds",
                "3",
                "--frame-rate",
                "24",
            ]
        )
        return preview

    async def export_mesh(
        self,
        processed_dir: Path,
        ns_dir: Path,
        export_dir: Path,
        splat_ply: Path,
    ) -> tuple[Path | None, dict]:
        backend = self.settings.mesh_backend.strip().lower() or "mesh"
        mesh_path = export_dir / (self.settings.mesh_export_filename.strip() or "mesh.glb")
        command = self.settings.mesh_export_command.strip()
        if not command:
            metrics = {
                "enabled": True,
                "applied": False,
                "backend": backend,
                "reason": "missing_command",
                "expected_path": str(mesh_path),
            }
            if self.settings.mesh_export_required:
                raise ValueError("mesh export is required but SPLATBOT_MESH_EXPORT_COMMAND is empty")
            return None, metrics
        argv = render_argv_template(
            command,
            {
                "backend": backend,
                "processed_dir": str(processed_dir),
                "ns_dir": str(ns_dir),
                "export_dir": str(export_dir),
                "splat_ply": str(splat_ply),
                "mesh_path": str(mesh_path),
            },
        )
        try:
            await self.runner.run(argv)
            if not mesh_path.exists():
                raise ValueError(f"mesh backend {backend!r} did not create {mesh_path}")
        except Exception as exc:  # noqa: BLE001
            if self.settings.mesh_export_required:
                raise
            return None, {
                "enabled": True,
                "applied": False,
                "backend": backend,
                "reason": "backend_failed",
                "error": str(exc),
                "expected_path": str(mesh_path),
            }
        return mesh_path, {
            "enabled": True,
            "applied": True,
            "backend": backend,
            "path": str(mesh_path),
            "size_bytes": mesh_path.stat().st_size,
        }


def latest_nerfstudio_config(ns_dir: Path) -> Path:
    configs = sorted(ns_dir.glob("**/config.yml"), key=lambda path: path.stat().st_mtime)
    if not configs:
        raise FileNotFoundError(f"no Nerfstudio config.yml found under {ns_dir}")
    return configs[-1]


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def shell_join(argv: tuple[str, ...] | list[str]) -> str:
    return " ".join(shlex.quote(arg) for arg in argv)


def parse_export_metrics(stdout: str, stderr: str) -> dict:
    text = "\n".join(part for part in (stdout, stderr) if part)
    match = None
    for match in EXPORT_RETENTION_RE.finditer(text):
        pass
    if match is None:
        return {}
    exported = int(match.group(1))
    total = int(match.group(2))
    retention = exported / total if total else 0.0
    return {
        "exported_gaussians": exported,
        "total_gaussians": total,
        "retention_ratio": round(retention, 6),
    }


def build_quality_report(metrics: dict, settings: Settings) -> dict:
    issues: list[str] = []
    warnings: list[str] = []
    events = metrics.get("pipeline_events", [])
    frames = metrics.get("frames", {})
    video = metrics.get("video", {})
    masks = metrics.get("masks", {})
    colmap = metrics.get("colmap", {})
    cleanup = metrics.get("ply", {}).get("cleanup", {})
    cleaned_ply = metrics.get("ply", {}).get("cleaned", {})
    export_metrics = metrics.get("ply", {}).get("export", {})
    validation = cleanup.get("validation", {})
    mesh = metrics.get("mesh", {})

    if video.get("quality_selection_fallback"):
        warnings.append("frame_quality_fallback_used")
    if frames.get("selected", 0) < settings.min_selected_video_frames:
        warnings.append("low_selected_frame_count")
    if masks.get("qa", {}).get("reason") == "too_few_accepted_masks":
        issues.append("too_few_accepted_object_masks")
    if pipeline_events_by_status(events, {"fallback", "recovered", "skip"}):
        warnings.append("pipeline_fallbacks_or_skips")
    if metrics.get("depth_backend_failures"):
        warnings.append("depth_backend_fallback_used")
    if metrics.get("pose_backend_failures"):
        warnings.append("pose_backend_fallback_used")
    if metrics.get("train_backend_failures"):
        warnings.append("train_backend_fallback_used")
    if metrics.get("colmap_fallback", {}).get("applied"):
        warnings.append("object_pose_used_original_frame_fallback")
    registered = colmap.get("active_registered_images")
    selected = frames.get("selected")
    if selected and registered is None and colmap.get("transforms_frames") is not None:
        issues.append("pose_missing_sparse_model")
    if selected and registered and registered < max(2, int(selected * settings.min_colmap_registered_ratio)):
        issues.append("low_pose_registration")
    if validation.get("applied") and not validation.get("passed", True):
        issues.append("postprocess_validation_failed")
    if (
        validation.get("applied")
        and validation.get("unobserved_fraction", 0.0) > settings.postprocess_validation_max_unobserved_fraction
    ):
        issues.append("high_unobserved_splat_fraction")
    if settings.mesh_export_enabled and not mesh.get("applied"):
        warnings.append("mesh_not_exported")
    vertices = cleaned_ply.get("vertices")
    if isinstance(vertices, int) and vertices < max(10_000, int(settings.min_splat_vertices * 1.5)):
        warnings.append("low_splat_vertex_count")
    retention = export_metrics.get("retention_ratio")
    if (
        isinstance(retention, int | float)
        and retention < settings.min_export_gaussian_retention
    ):
        issues.append("low_exported_gaussian_retention")
    ratio = cleaned_ply.get("flat_axis_ratio")
    if (
        ratio is not None
        and ratio < settings.max_flattened_axis_ratio
        and isinstance(vertices, int)
        and vertices < 20_000
    ):
        issues.append("flattened_splat_geometry")

    return {
        "job_id": metrics.get("job_id"),
        "mode": metrics.get("mode"),
        "preset": metrics.get("preset"),
        "passed": not issues,
        "issues": issues,
        "warnings": warnings,
        "pipeline_events": events,
        "event_summary": summarize_pipeline_events(events),
        "stage_seconds": metrics.get("stages", {}),
        "frames": frames,
        "depth_backend": metrics.get("depth_backend"),
        "pose_backend": metrics.get("pose_backend"),
        "train_backend": metrics.get("train_backend"),
        "segmentation_backend": masks.get("backend", {}).get("selected"),
        "ply": metrics.get("ply", {}),
        "mesh": mesh,
    }


def build_candidate_report(metrics: dict, settings: Settings) -> dict:
    preset = settings.preset_config(metrics.get("preset"))
    return {
        "job_id": metrics.get("job_id"),
        "configured": {
            "segmentation_backends": configured_segmentation_backends(settings, preset),
            "depth_backends": configured_depth_backends(settings, preset),
            "pose_backends": configured_pose_backends(settings, preset),
            "train_backends": configured_train_backends(settings, preset),
            "mesh_backend": settings.mesh_backend,
        },
        "video": metrics.get("video", {}),
        "masks": metrics.get("masks", {}),
        "pipeline_events": metrics.get("pipeline_events", []),
        "depth_backend_failures": metrics.get("depth_backend_failures", []),
        "pose_backend_attempts": metrics.get("pose_backend_attempts", []),
        "pose_backend_failures": metrics.get("pose_backend_failures", []),
        "train_backend_failures": metrics.get("train_backend_failures", []),
        "colmap_attempts": metrics.get("colmap_attempts", []),
        "cleanup": metrics.get("ply", {}).get("cleanup", {}),
        "mesh": metrics.get("mesh", {}),
    }


def record_stage(metrics: dict, name: str, started: float) -> None:
    metrics.setdefault("stages", {})[name] = {
        "duration_seconds": round(time.perf_counter() - started, 3)
    }


def record_pipeline_event(
    metrics: dict,
    stage: str,
    status: str,
    reason: str,
    backend: str | None = None,
    error: str | None = None,
    recovered: bool | None = None,
    details: dict | None = None,
) -> None:
    event = {
        "stage": stage,
        "status": status,
        "reason": reason,
    }
    if backend:
        event["backend"] = backend
    if error:
        event["error"] = truncate_text(error, 1200)
    if recovered is not None:
        event["recovered"] = recovered
    if details:
        event["details"] = details
    metrics.setdefault("pipeline_events", []).append(event)


def pipeline_events_by_status(events: list[dict], statuses: set[str]) -> list[dict]:
    return [
        event
        for event in events
        if isinstance(event, dict) and str(event.get("status") or "") in statuses
    ]


def summarize_pipeline_events(events: list[dict], limit: int = 6) -> list[str]:
    summary: list[str] = []
    visible_statuses = {"fallback", "recovered", "skip", "failure", "warning"}
    for event in pipeline_events_by_status(events, visible_statuses):
        stage = str(event.get("stage") or "pipeline")
        status = str(event.get("status") or "event")
        backend = str(event.get("backend") or "").strip()
        reason = str(event.get("reason") or "no reason").strip()
        label = f"{stage}/{backend}" if backend else stage
        summary.append(f"{label}: {status} ({reason})")
        if len(summary) >= limit:
            break
    remaining = max(0, len(pipeline_events_by_status(events, visible_statuses)) - len(summary))
    if remaining:
        summary.append(f"{remaining} more pipeline event(s)")
    return summary


def truncate_text(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "..."


def count_files(path: Path) -> int:
    return sum(1 for item in path.iterdir() if item.is_file())


def minimum_segmentation_outputs(settings: Settings, input_count: int) -> int:
    ratio = clamp(settings.segmentation_min_output_ratio, 0.0, 1.0)
    ratio_count = math.ceil(max(0, input_count) * ratio)
    required = max(settings.segmentation_min_output_files, ratio_count)
    return min(max(1, required), max(1, input_count))


def configured_segmentation_backends(settings: Settings, preset: ScanPresetConfig | None = None) -> list[str]:
    configured = settings.segmentation_backend.strip() or settings.object_mask_backend.strip() or "rembg"
    backends = parse_csv_list(configured)
    if preset is not None and preset.preset == ScanPreset.BEST and backends == ["rembg"]:
        backends = parse_csv_list(settings.best_segmentation_backends) or ["sam2", "rembg"]
    if not settings.experimental_sam3_enabled:
        backends = [backend for backend in backends if backend not in {"sam3", "sam3-video"}]
    if not backends:
        backends = ["rembg"]
    return list(dict.fromkeys(backends))


def ensure_required_segmentation_backends_configured(settings: Settings, preset: ScanPresetConfig) -> None:
    if preset.preset != ScanPreset.BEST:
        return
    configured = configured_segmentation_backends(settings, preset)
    missing = [backend for backend in required_segmentation_backends(settings, preset) if backend not in configured]
    if missing:
        raise ValueError(f"best preset requires configured segmentation backend(s): {', '.join(missing)}")


def required_segmentation_backends(settings: Settings, preset: ScanPresetConfig) -> list[str]:
    return parse_csv_list(settings.best_segmentation_required_backends) if preset.preset == ScanPreset.BEST else []


def object_mask_command_for_backend(
    settings: Settings,
    backend: str,
    images_dir: Path,
    object_dir: Path,
) -> list[str] | None:
    backend = backend.strip().lower()
    if backend in {"", "rembg"}:
        return [settings.rembg_bin, "p", str(images_dir), str(object_dir)]
    command = {
        "sam3": settings.sam3_mask_command,
        "sam3-video": settings.sam3_mask_command,
        "sam2": settings.sam2_mask_command,
        "sam2-video": settings.sam2_mask_command,
        "matting": settings.matting_command,
        "matanyone": settings.matting_command,
        "external": settings.object_mask_command,
    }.get(backend)
    command = (command or settings.object_mask_command).strip()
    if not command:
        return None
    return render_argv_template(
        command,
        {
            "backend": backend,
            "images_dir": str(images_dir),
            "object_dir": str(object_dir),
            "input_dir": str(images_dir),
            "output_dir": str(object_dir),
            "prompt": settings.object_mask_prompt,
        },
    )


def configured_pose_backends(settings: Settings, preset: ScanPresetConfig | None = None) -> list[str]:
    configured = parse_csv_list(settings.pose_backends) or ["colmap"]
    if preset is not None and preset.preset == ScanPreset.BEST and configured == ["colmap"]:
        configured = parse_csv_list(settings.best_pose_backends) or [
            "colmap-global",
            "colmap-sequential",
            "colmap-exhaustive",
            "colmap",
        ]
    return list(dict.fromkeys(configured))


def ensure_required_pose_backends_configured(settings: Settings, preset: ScanPresetConfig) -> None:
    if preset.preset != ScanPreset.BEST:
        return
    configured = configured_pose_backends(settings, preset)
    missing_from_chain = [
        backend for backend in required_pose_backends(settings, preset) if backend not in configured
    ]
    missing_commands = [
        backend
        for backend in required_pose_backends(settings, preset)
        if not is_colmap_pose_backend(backend) and not pose_command_for_backend(settings, backend)
    ]
    missing = missing_from_chain + missing_commands
    if missing:
        raise ValueError(f"best preset requires configured pose backend(s): {', '.join(dict.fromkeys(missing))}")


def required_pose_backends(settings: Settings, preset: ScanPresetConfig) -> list[str]:
    return parse_csv_list(settings.best_pose_required_backends) if preset.preset == ScanPreset.BEST else []


def is_colmap_pose_backend(backend: str) -> bool:
    return backend in {
        "colmap",
        "nerfstudio-colmap",
        "ns-process-data",
        "colmap-global",
        "colmap-sequential",
        "colmap-exhaustive",
        "colmap-vocab-tree",
        "colmap-spatial",
    }


def colmap_matching_method_for_pose_backend(backend: str, default: str | None) -> str | None:
    return {
        "colmap": default,
        "nerfstudio-colmap": default,
        "ns-process-data": default,
        "colmap-global": default,
        "colmap-sequential": "sequential",
        "colmap-exhaustive": "exhaustive",
        "colmap-vocab-tree": "vocab_tree",
        "colmap-spatial": "spatial",
    }.get(backend, default)


def colmap_mapper_for_pose_backend(backend: str) -> str | None:
    if backend in {"colmap-global", "global", "glomap"}:
        return "global"
    return None


def configured_depth_backends(settings: Settings, preset: ScanPresetConfig) -> list[str]:
    configured = parse_csv_list(settings.depth_backends)
    if not configured and preset.preset == ScanPreset.BEST:
        configured = parse_csv_list(settings.best_depth_backends)
    return list(dict.fromkeys(configured))


def ensure_required_depth_backends_configured(settings: Settings, preset: ScanPresetConfig) -> None:
    if preset.preset != ScanPreset.BEST:
        return
    configured = configured_depth_backends(settings, preset)
    missing_from_chain = [
        backend for backend in required_depth_backends(settings, preset) if backend not in configured
    ]
    missing_commands = [
        backend
        for backend in required_depth_backends(settings, preset)
        if not depth_command_for_backend(settings, backend)
    ]
    missing = missing_from_chain + missing_commands
    if missing:
        raise ValueError(f"best preset requires configured depth backend(s): {', '.join(dict.fromkeys(missing))}")


def required_depth_backends(settings: Settings, preset: ScanPresetConfig) -> list[str]:
    return parse_csv_list(settings.best_depth_required_backends) if preset.preset == ScanPreset.BEST else []


def depth_command_for_backend(settings: Settings, backend: str) -> str:
    backend = backend.strip().lower()
    if backend.startswith("da3"):
        return settings.da3_depth_command.strip()
    if backend.startswith("depth-anything-v2"):
        return settings.depth_anything_v2_command.strip()
    return settings.depth_backend_command.strip()


def pose_command_for_backend(settings: Settings, backend: str) -> str:
    backend = backend.strip().lower()
    if backend == "da3-colmap" and settings.da3_pose_command.strip():
        return settings.da3_pose_command.strip()
    if backend == "vggt-colmap" and settings.vggt_pose_command.strip():
        return settings.vggt_pose_command.strip()
    if backend == "mast3r-sfm" and settings.mast3r_pose_command.strip():
        return settings.mast3r_pose_command.strip()
    return settings.pose_backend_command.strip()


def train_command_for_backend(settings: Settings, backend: str) -> str:
    backend = backend.strip().lower()
    if backend == "3dgs-mcmc" and settings.mcmc_train_command.strip():
        return settings.mcmc_train_command.strip()
    if backend == "mip-splatting" and settings.mip_splatting_train_command.strip():
        return settings.mip_splatting_train_command.strip()
    if backend == "2dgs" and settings.twodgs_train_command.strip():
        return settings.twodgs_train_command.strip()
    return settings.train_backend_command.strip()


def configured_train_backends(settings: Settings, preset: ScanPresetConfig) -> list[str]:
    configured = parse_csv_list(settings.train_backends)
    if not configured:
        if preset.preset == ScanPreset.BEST:
            configured = parse_csv_list(settings.best_train_backends) or [preset.train_method.lower()]
            if settings.experimental_dn_splatter_enabled and "dn-splatter-big" not in configured:
                configured.insert(0, "dn-splatter-big")
        else:
            configured = [preset.train_method.lower()]
    return list(dict.fromkeys(configured))


def ensure_required_train_backends_configured(settings: Settings, preset: ScanPresetConfig) -> None:
    if preset.preset != ScanPreset.BEST:
        return
    configured = configured_train_backends(settings, preset)
    missing_from_chain = [
        backend for backend in required_train_backends(settings, preset) if backend not in configured
    ]
    missing_commands = [
        backend
        for backend in required_train_backends(settings, preset)
        if backend not in {"splatfacto", "splatfacto-big", preset.train_method.lower()}
        and not train_command_for_backend(settings, backend)
    ]
    missing = missing_from_chain + missing_commands
    if missing:
        raise ValueError(f"best preset requires configured training backend(s): {', '.join(dict.fromkeys(missing))}")


def required_train_backends(settings: Settings, preset: ScanPresetConfig) -> list[str]:
    return parse_csv_list(settings.best_train_required_backends) if preset.preset == ScanPreset.BEST else []


def select_video_frames(
    candidates: list[Path],
    images_dir: Path,
    target_count: int,
    min_count: int,
    quality_threshold: float,
    blur_threshold: float,
    low_contrast_threshold: float,
    overexposed_threshold: float,
    underexposed_threshold: float,
    duplicate_threshold: float,
    strategy: str = "quality",
) -> FrameSelectionResult:
    if not candidates:
        raise ValueError("video frame extraction produced no candidate frames")
    profiles = score_video_frames(
        candidates,
        quality_threshold=quality_threshold,
        blur_threshold=blur_threshold,
        low_contrast_threshold=low_contrast_threshold,
        overexposed_threshold=overexposed_threshold,
        underexposed_threshold=underexposed_threshold,
        duplicate_threshold=duplicate_threshold,
    )
    accepted = [profile for profile in profiles if not profile.reject_reasons]
    fallback_used = False
    selection_pool = accepted
    if len(selection_pool) < min_count:
        fallback_used = True
        top_up_count = min(max(min_count, len(selection_pool)), len(profiles))
        top_profiles = sorted(profiles, key=lambda profile: profile.score, reverse=True)[:top_up_count]
        by_index = {profile.index: profile for profile in selection_pool}
        by_index.update({profile.index: profile for profile in top_profiles})
        selection_pool = sorted(by_index.values(), key=lambda profile: profile.index)

    selected_count = min(target_count, len(selection_pool))
    normalized_strategy = normalize_frame_selection_strategy(strategy)
    diversity_fallback_used = normalized_strategy == "quality-diversity" and not any(
        profile.signature for profile in selection_pool
    )
    if normalized_strategy == "quality-diversity":
        selected = quality_diverse_sample(selection_pool, selected_count)
    else:
        selected = quality_aware_sample(selection_pool, selected_count)
    for idx, src in enumerate(selected, start=1):
        shutil.copy2(src.path, images_dir / f"frame_{idx:05d}.jpg")

    return FrameSelectionResult(
        selected_count=len(selected),
        metrics={
            "candidate_frames": len(candidates),
            "selected_frames": len(selected),
            "frame_selection_strategy": normalized_strategy,
            "accepted_frame_candidates": len(accepted),
            "quality_selection_fallback": fallback_used,
            "diversity_selection_fallback": diversity_fallback_used,
            "quality_rejected_frames": len(profiles) - len(accepted),
            "rejected_by_reason": rejected_reason_counts(profiles),
            "quality_score": summarize_profile_values(profiles, "score"),
            "selected_quality_score": summarize_profile_values(selected, "score"),
            "blur_score": summarize_profile_values(profiles, "blur"),
            "contrast": summarize_profile_values(profiles, "contrast"),
            "brightness": summarize_profile_values(profiles, "brightness"),
            "overexposed_ratio": summarize_profile_values(profiles, "overexposed_ratio"),
            "underexposed_ratio": summarize_profile_values(profiles, "underexposed_ratio"),
            "frame_difference": summarize_profile_values(profiles, "difference_from_previous"),
        },
    )


def refine_object_masks(object_dir: Path, settings: Settings) -> dict:
    """Tighten alpha masks after background removal when OpenCV is available."""
    if not settings.object_mask_refine_enabled:
        return {"applied": False, "reason": "disabled"}
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        return {"applied": False, "reason": "opencv_unavailable", "error": str(exc)}

    changed = 0
    skipped = 0
    kernel = np.ones((3, 3), np.uint8)
    for image_path in sorted(object_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() != ".png":
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None or len(image.shape) < 3 or image.shape[2] < 4:
            skipped += 1
            continue
        alpha = image[:, :, 3]
        _, binary = cv2.threshold(alpha, settings.object_mask_training_alpha_threshold, 255, cv2.THRESH_BINARY)
        components, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        if components <= 1:
            skipped += 1
            continue
        largest = 1 + max(range(components - 1), key=lambda idx: stats[idx + 1, cv2.CC_STAT_AREA])
        refined = np.where(labels == largest, alpha, 0).astype(np.uint8)
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel, iterations=1)
        if not np.array_equal(alpha, refined):
            image[:, :, 3] = refined
            cv2.imwrite(str(image_path), image)
            changed += 1
    return {"applied": True, "changed": changed, "skipped": skipped}


def apply_object_mask_qa(
    original_images_dir: Path,
    object_images_dir: Path,
    job_dir: Path,
    settings: Settings,
    preset: ScanPresetConfig,
) -> dict:
    profiles = analyze_object_masks(object_images_dir, settings)
    metrics = summarize_object_mask_profiles(profiles)
    if not settings.object_mask_qa_enabled:
        return {**metrics, "applied": False, "reason": "disabled"}
    accepted = [profile for profile in profiles if not profile.reject_reasons]
    rejected = len(profiles) - len(accepted)
    if not profiles:
        return {**metrics, "applied": False, "reason": "no_masks"}
    if rejected == 0:
        return {**metrics, "applied": False, "reason": "all_masks_accepted"}

    min_keep = min(
        preset.max_video_frames,
        max(1, min(settings.min_selected_video_frames, settings.object_mask_min_keep_frames)),
    )
    keep_ratio = len(accepted) / len(profiles)
    if len(accepted) < min_keep or keep_ratio < settings.object_mask_min_keep_ratio:
        return {
            **metrics,
            "applied": False,
            "reason": "too_few_accepted_masks",
            "min_keep_frames": min_keep,
            "min_keep_ratio": settings.object_mask_min_keep_ratio,
        }

    filtered_original = job_dir / "images_maskqa"
    filtered_object = job_dir / "object_images_maskqa"
    for directory in (filtered_original, filtered_object):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)

    accepted_stems = {profile.stem for profile in accepted}
    copy_matching_stems(original_images_dir, filtered_original, accepted_stems)
    copy_matching_stems(object_images_dir, filtered_object, accepted_stems)
    return {
        **metrics,
        "applied": True,
        "reason": "rejected_low_quality_masks",
        "original_images_dir": str(filtered_original),
        "object_images_dir": str(filtered_object),
        "kept_frames": count_files(filtered_object),
        "dropped_frames": rejected,
    }


def copy_matching_stems(source_dir: Path, dest_dir: Path, stems: set[str]) -> int:
    copied = 0
    for path in sorted(source_dir.iterdir()):
        if path.is_file() and path.stem in stems:
            shutil.copy2(path, dest_dir / path.name)
            copied += 1
    return copied


def analyze_object_masks(object_images_dir: Path, settings: Settings) -> list[ObjectMaskProfile]:
    profiles: list[ObjectMaskProfile] = []
    previous_area: float | None = None
    for path in sorted(object_images_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        profile = profile_object_mask(path, settings)
        reasons = list(profile.reject_reasons)
        if previous_area is not None and previous_area > 0 and profile.foreground_pixels > 0:
            area_jump = abs(profile.area_ratio - previous_area) / max(previous_area, 1e-6)
            if area_jump > settings.object_mask_max_area_jump:
                reasons.append("area_jump")
        if reasons:
            profile = ObjectMaskProfile(
                path=profile.path,
                stem=profile.stem,
                width=profile.width,
                height=profile.height,
                foreground_pixels=profile.foreground_pixels,
                area_ratio=profile.area_ratio,
                edge_touch_ratio=profile.edge_touch_ratio,
                bbox_fill_ratio=profile.bbox_fill_ratio,
                center_x=profile.center_x,
                center_y=profile.center_y,
                reject_reasons=tuple(sorted(set(reasons))),
            )
        profiles.append(profile)
        if not profile.reject_reasons:
            previous_area = profile.area_ratio
    return profiles


def profile_object_mask(path: Path, settings: Settings) -> ObjectMaskProfile:
    mask = load_alpha_mask(path)
    if mask is None:
        return ObjectMaskProfile(path, path.stem, 0, 0, 0, 0.0, 0.0, None, None, None, ("missing_alpha",))
    threshold = settings.object_mask_training_alpha_threshold
    foreground = 0
    edge = 0
    min_x = mask.width
    min_y = mask.height
    max_x = -1
    max_y = -1
    edge_margin = 2
    for y in range(mask.height):
        row = y * mask.width
        for x in range(mask.width):
            if mask.alpha[row + x] < threshold:
                continue
            foreground += 1
            if x < edge_margin or y < edge_margin or x >= mask.width - edge_margin or y >= mask.height - edge_margin:
                edge += 1
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
    pixels = max(1, mask.width * mask.height)
    area_ratio = foreground / pixels
    bbox_fill_ratio = None
    center_x = center_y = None
    if foreground > 0 and max_x >= min_x and max_y >= min_y:
        bbox_area = max(1, (max_x - min_x + 1) * (max_y - min_y + 1))
        bbox_fill_ratio = foreground / bbox_area
        center_x = ((min_x + max_x) / 2.0) / max(1, mask.width)
        center_y = ((min_y + max_y) / 2.0) / max(1, mask.height)

    reasons: list[str] = []
    if foreground == 0:
        reasons.append("empty")
    if area_ratio < settings.object_mask_min_area_ratio:
        reasons.append("too_small")
    if area_ratio > settings.object_mask_max_area_ratio:
        reasons.append("too_large")
    edge_touch_ratio = edge / foreground if foreground else 0.0
    if edge_touch_ratio > settings.object_mask_max_edge_touch_ratio:
        reasons.append("touches_frame_edge")

    return ObjectMaskProfile(
        path=path,
        stem=path.stem,
        width=mask.width,
        height=mask.height,
        foreground_pixels=foreground,
        area_ratio=area_ratio,
        edge_touch_ratio=edge_touch_ratio,
        bbox_fill_ratio=bbox_fill_ratio,
        center_x=center_x,
        center_y=center_y,
        reject_reasons=tuple(reasons),
    )


def summarize_object_mask_profiles(profiles: list[ObjectMaskProfile]) -> dict:
    accepted = [profile for profile in profiles if not profile.reject_reasons]
    return {
        "total_masks": len(profiles),
        "accepted_masks": len(accepted),
        "rejected_masks": len(profiles) - len(accepted),
        "rejected_by_reason": object_mask_rejected_reason_counts(profiles),
        "area_ratio": summarize_numbers([profile.area_ratio for profile in profiles if profile.width and profile.height]),
        "edge_touch_ratio": summarize_numbers([profile.edge_touch_ratio for profile in profiles if profile.width and profile.height]),
        "bbox_fill_ratio": summarize_numbers(
            [profile.bbox_fill_ratio for profile in profiles if profile.bbox_fill_ratio is not None]
        ),
    }


def object_mask_rejected_reason_counts(profiles: list[ObjectMaskProfile]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for profile in profiles:
        for reason in profile.reject_reasons:
            counts[reason] = counts.get(reason, 0) + 1
    return counts


def frame_selection_strategy_for_preset(settings: Settings, preset: ScanPresetConfig) -> str:
    if preset.preset == ScanPreset.BEST:
        return normalize_frame_selection_strategy(
            settings.best_frame_selection_strategy or settings.frame_selection_strategy
        )
    return normalize_frame_selection_strategy(settings.frame_selection_strategy)


def normalize_frame_selection_strategy(strategy: str) -> str:
    normalized = (strategy or "quality").strip().lower().replace("_", "-")
    if normalized in {"diverse", "quality-diverse", "quality-diversity"}:
        return "quality-diversity"
    return "quality"


def quality_aware_sample(profiles: list[FrameQuality], count: int) -> list[FrameQuality]:
    if count >= len(profiles):
        return sorted(profiles, key=lambda profile: profile.index)
    if count <= 0:
        return []
    selected: dict[int, FrameQuality] = {}
    total = len(profiles)
    for slot in range(count):
        start = math.floor(slot * total / count)
        end = math.floor((slot + 1) * total / count)
        bucket = profiles[start:max(end, start + 1)]
        best = max(bucket, key=lambda profile: profile.score)
        selected[best.index] = best
    if len(selected) < count:
        for profile in sorted(profiles, key=lambda item: item.score, reverse=True):
            selected.setdefault(profile.index, profile)
            if len(selected) >= count:
                break
    return sorted(selected.values(), key=lambda profile: profile.index)


def quality_diverse_sample(profiles: list[FrameQuality], count: int) -> list[FrameQuality]:
    if count >= len(profiles):
        return sorted(profiles, key=lambda profile: profile.index)
    if count <= 0:
        return []
    if not any(profile.signature for profile in profiles):
        return quality_aware_sample(profiles, count)

    selected: dict[int, FrameQuality] = {}
    total = len(profiles)
    score_values = [profile.score for profile in profiles]
    min_score = min(score_values)
    max_score = max(score_values)
    for slot in range(count):
        start = math.floor(slot * total / count)
        end = math.floor((slot + 1) * total / count)
        bucket = profiles[start:max(end, start + 1)]
        best = max(
            bucket,
            key=lambda profile: diverse_frame_score(
                profile,
                list(selected.values()),
                min_score=min_score,
                max_score=max_score,
                total=max(1, total - 1),
            ),
        )
        selected[best.index] = best
    if len(selected) < count:
        for profile in sorted(
            profiles,
            key=lambda candidate: diverse_frame_score(
                candidate,
                list(selected.values()),
                min_score=min_score,
                max_score=max_score,
                total=max(1, total - 1),
            ),
            reverse=True,
        ):
            selected.setdefault(profile.index, profile)
            if len(selected) >= count:
                break
    return sorted(selected.values(), key=lambda profile: profile.index)


def diverse_frame_score(
    profile: FrameQuality,
    selected: list[FrameQuality],
    min_score: float,
    max_score: float,
    total: int,
) -> float:
    quality = (profile.score - min_score) / (max_score - min_score) if max_score > min_score else 1.0
    if not selected:
        return quality
    signature_distance = min(frame_signature_distance(profile, other) for other in selected)
    temporal_distance = min(abs(profile.index - other.index) / total for other in selected)
    motion = clamp((profile.difference_from_previous or 0.0) / 25.0, 0.0, 1.0)
    return (0.58 * quality) + (0.27 * signature_distance) + (0.10 * temporal_distance) + (0.05 * motion)


def frame_signature_distance(first: FrameQuality, second: FrameQuality) -> float:
    if not first.signature or not second.signature or len(first.signature) != len(second.signature):
        return 0.5
    distance = math.sqrt(
        sum((left - right) ** 2 for left, right in zip(first.signature, second.signature, strict=True))
        / len(first.signature)
    )
    return clamp(distance / 3.0, 0.0, 1.0)


def score_video_frames(
    candidates: list[Path],
    quality_threshold: float,
    blur_threshold: float,
    low_contrast_threshold: float,
    overexposed_threshold: float,
    underexposed_threshold: float,
    duplicate_threshold: float,
) -> list[FrameQuality]:
    try:
        import cv2  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return score_video_frames_without_cv2(candidates, duplicate_threshold)

    profiles: list[FrameQuality] = []
    previous_small = None
    for index, candidate in enumerate(candidates):
        image = cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)
        if image is None:
            profiles.append(FrameQuality(path=candidate, index=index, score=0.0, reject_reasons=("unreadable",)))
            previous_small = None
            continue
        gray = resize_gray_for_metrics(cv2, image)
        small = cv2.resize(gray, (32, 32))
        signature = frame_signature(cv2, gray)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        contrast = float(gray.std())
        brightness = float(gray.mean())
        overexposed = float((gray >= 245).mean())
        underexposed = float((gray <= 10).mean())
        difference = None
        if previous_small is not None:
            difference = float(abs(previous_small.astype("float32") - small.astype("float32")).mean())
        previous_small = small

        reasons: list[str] = []
        if blur < blur_threshold:
            reasons.append("blur")
        if contrast < low_contrast_threshold:
            reasons.append("low_contrast")
        if overexposed > overexposed_threshold:
            reasons.append("overexposed")
        if underexposed > underexposed_threshold:
            reasons.append("underexposed")
        if difference is not None and difference < duplicate_threshold:
            reasons.append("duplicate")
        score = frame_quality_score(blur, contrast, brightness, overexposed, underexposed)
        if score < quality_threshold:
            reasons.append("low_quality")

        profiles.append(
            FrameQuality(
                path=candidate,
                index=index,
                score=score,
                blur=blur,
                contrast=contrast,
                brightness=brightness,
                overexposed_ratio=overexposed,
                underexposed_ratio=underexposed,
                difference_from_previous=difference,
                signature=signature,
                reject_reasons=tuple(reasons),
            )
        )
    return profiles


def score_video_frames_without_cv2(candidates: list[Path], duplicate_threshold: float) -> list[FrameQuality]:
    profiles: list[FrameQuality] = []
    previous: Path | None = None
    for index, candidate in enumerate(candidates):
        difference = frame_difference(previous, candidate) if previous is not None else None
        reasons = ("duplicate",) if difference is not None and difference < duplicate_threshold else ()
        profiles.append(
            FrameQuality(
                path=candidate,
                index=index,
                score=50.0,
                difference_from_previous=difference,
                reject_reasons=reasons,
            )
        )
        previous = candidate
    return profiles


def resize_gray_for_metrics(cv2, image, max_side: int = 640):
    height, width = image.shape[:2]
    largest = max(height, width)
    if largest <= max_side:
        return image
    scale = max_side / largest
    return cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))))


def frame_signature(cv2, gray) -> tuple[float, ...]:
    small = cv2.resize(gray, (8, 8)).astype("float32")
    mean = float(small.mean())
    std = float(small.std())
    if std > 1e-6:
        small = (small - mean) / std
    else:
        small = small - mean
    return tuple(round(float(value), 4) for value in small.reshape(-1))


def frame_quality_score(
    blur: float,
    contrast: float,
    brightness: float,
    overexposed: float,
    underexposed: float,
) -> float:
    blur_component = clamp(math.log1p(max(0.0, blur)) / math.log1p(500.0), 0.0, 1.0)
    contrast_component = clamp(contrast / 60.0, 0.0, 1.0)
    brightness_component = 1.0 - clamp(abs(brightness - 128.0) / 128.0, 0.0, 1.0)
    exposure_component = 1.0 - clamp(max(overexposed, underexposed) * 2.0, 0.0, 1.0)
    return round(
        100.0
        * (
            0.45 * blur_component
            + 0.25 * contrast_component
            + 0.20 * exposure_component
            + 0.10 * brightness_component
        ),
        3,
    )


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def rejected_reason_counts(profiles: list[FrameQuality]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for profile in profiles:
        for reason in profile.reject_reasons:
            counts[reason] = counts.get(reason, 0) + 1
    return counts


def summarize_numbers(numbers: list[float | None]) -> dict[str, float | None]:
    values = [float(value) for value in numbers if value is not None]
    if not values:
        return {"min": None, "median": None, "max": None}
    return {
        "min": round(min(values), 3),
        "median": round(float(median(values)), 3),
        "max": round(max(values), 3),
    }


def summarize_profile_values(profiles: list[FrameQuality], field: str) -> dict[str, float | None]:
    values = [getattr(profile, field) for profile in profiles]
    return summarize_numbers(values)


def frame_difference(previous: Path | None, current: Path) -> float | None:
    if previous is None:
        return None
    try:
        import cv2  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        try:
            previous_size = previous.stat().st_size
            current_size = current.stat().st_size
        except OSError:
            return None
        largest_size = max(previous_size, current_size)
        if largest_size == 0:
            return 0.0
        return abs(previous_size - current_size) * 100.0 / largest_size
    prev = cv2.imread(str(previous), cv2.IMREAD_GRAYSCALE)
    curr = cv2.imread(str(current), cv2.IMREAD_GRAYSCALE)
    if prev is None or curr is None:
        return None
    prev = cv2.resize(prev, (32, 32))
    curr = cv2.resize(curr, (32, 32))
    return float(abs(prev.astype("float32") - curr.astype("float32")).mean())


def inspect_processed_dataset(processed_dir: Path) -> dict:
    transforms_path = processed_dir / "transforms.json"
    transforms_frames = None
    if transforms_path.exists():
        try:
            transforms_frames = len(json.loads(transforms_path.read_text(encoding="utf-8")).get("frames", []))
        except (OSError, json.JSONDecodeError):
            transforms_frames = None
    models = []
    for images_bin in sorted(processed_dir.rglob("images.bin")):
        model_dir = images_bin.parent
        count = read_colmap_registered_images(images_bin)
        points3d = model_dir / "points3D.bin"
        models.append(
            {
                "path": str(model_dir.relative_to(processed_dir)),
                "registered_images": count,
                "points3d_bytes": points3d.stat().st_size if points3d.exists() else None,
                "points3d_count": read_colmap_points3d_count(points3d),
                "active_sparse_0": model_dir.name == "0",
            }
        )
    active = next((model for model in models if model["active_sparse_0"]), None)
    return {
        "transforms_frames": transforms_frames,
        "models": models,
        "active_registered_images": active["registered_images"] if active else None,
        "active_points3d_count": active["points3d_count"] if active else None,
        "active_points3d_bytes": active["points3d_bytes"] if active else None,
        "best_registered_images": max(
            (model["registered_images"] for model in models if model["registered_images"] is not None),
            default=None,
        ),
    }


def read_colmap_registered_images(images_bin: Path) -> int | None:
    try:
        from nerfstudio.data.utils.colmap_parsing_utils import read_images_binary

        return len(read_images_binary(images_bin))
    except Exception:  # noqa: BLE001
        pass
    if images_bin.suffix == ".bin":
        try:
            with images_bin.open("rb") as handle:
                data = handle.read(8)
        except OSError:
            return None
        if len(data) == 8:
            count = struct.unpack("<Q", data)[0]
            if count < 1_000_000:
                return int(count)
    try:
        text = images_bin.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        return None
    except OSError:
        return None
    if text.startswith("images="):
        try:
            return int(text.split("=", 1)[1])
        except ValueError:
            return None
    return None


def read_colmap_points3d_count(points3d_bin: Path) -> int | None:
    if not points3d_bin.exists():
        return None
    if points3d_bin.suffix == ".bin":
        try:
            with points3d_bin.open("rb") as handle:
                data = handle.read(8)
        except OSError:
            return None
        if len(data) != 8:
            return None
        return struct.unpack("<Q", data)[0]
    try:
        text = points3d_bin.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    return sum(1 for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#"))


def validate_colmap_quality(metrics: dict, target_frames: int, settings: Settings) -> None:
    selected_frames = metrics.get("frames", {}).get("selected") or target_frames
    if selected_frames < 10:
        return
    colmap = metrics.get("colmap", {})
    registered = colmap.get("active_registered_images")
    if registered is None:
        if colmap.get("transforms_frames") is not None:
            raise ValueError(
                f"pose backend produced {colmap.get('transforms_frames')} transform frame(s) but no active "
                "COLMAP sparse model. Transform-only poses are not accepted by the production quality gate "
                "because they can produce diffuse or distorted splats."
            )
        raise ValueError(
            f"COLMAP registration count is unknown for {selected_frames} selected frame(s). "
            "This run cannot be quality-gated safely."
        )
    minimum = max(2, int(selected_frames * settings.min_colmap_registered_ratio))
    if registered < minimum:
        best = colmap.get("best_registered_images")
        hint = f"; best sparse model has {best} registered image(s)" if best and best > registered else ""
        raise ValueError(
            f"COLMAP registered only {registered}/{selected_frames} selected frame(s) in the active sparse model"
            f"{hint}. This run would likely produce a distorted splat."
        )
    points3d_count = colmap.get("active_points3d_count")
    points3d_bytes = colmap.get("active_points3d_bytes")
    if points3d_count is not None and points3d_count < settings.min_colmap_sparse_points:
        raise ValueError(
            f"COLMAP active sparse model has only {points3d_count} 3D point(s); expected at least "
            f"{settings.min_colmap_sparse_points}. This run would likely produce weak or diffuse geometry."
        )
    if points3d_count is None and points3d_bytes is not None and points3d_bytes <= 8:
        raise ValueError("COLMAP active sparse model has an empty points3D.bin.")


def record_colmap_attempt(
    metrics: dict,
    source: str,
    input_dir: Path,
    frame_count: int,
    matching_method: str | None,
    result: dict,
    retry: bool = False,
) -> None:
    metrics.setdefault("colmap_attempts", []).append(
        {
            "source": source,
            "input_dir": str(input_dir),
            "frame_count": frame_count,
            "matching_method": matching_method or "default",
            "retry": retry,
            "result": result,
        }
    )


def build_even_subset(image_files: list[Path], subset_dir: Path, frame_count: int) -> None:
    if subset_dir.exists():
        shutil.rmtree(subset_dir)
    subset_dir.mkdir(parents=True, exist_ok=True)
    selected = quality_blind_even_sample(image_files, min(frame_count, len(image_files)))
    for src in selected:
        shutil.copy2(src, subset_dir / src.name)


def quality_blind_even_sample(items: list, count: int) -> list:
    if count >= len(items):
        return list(items)
    if count <= 0:
        return []
    if count == 1:
        return [items[len(items) // 2]]
    step = (len(items) - 1) / (count - 1)
    return [items[round(idx * step)] for idx in range(count)]


def processed_frame_stems(processed_dir: Path) -> set[str]:
    transforms_path = processed_dir / "transforms.json"
    if not transforms_path.exists():
        return set()
    try:
        frames = json.loads(transforms_path.read_text(encoding="utf-8")).get("frames", [])
    except (OSError, json.JSONDecodeError):
        return set()
    stems: set[str] = set()
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        file_path = frame.get("file_path")
        if isinstance(file_path, str) and file_path:
            stems.add(Path(PurePosixPath(file_path).name).stem)
    return stems


def replace_processed_images_with_object_images(
    processed_dir: Path,
    object_images_dir: Path,
    allowed_stems: set[str] | None = None,
) -> int:
    transforms_path = processed_dir / "transforms.json"
    if not transforms_path.exists():
        raise FileNotFoundError(f"no transforms.json found under {processed_dir}")
    data = json.loads(transforms_path.read_text(encoding="utf-8"))
    frames = data.get("frames")
    if not isinstance(frames, list):
        raise ValueError(f"{transforms_path} does not contain a frames list")

    object_images = {
        image.stem: image
        for image in sorted(object_images_dir.iterdir())
        if image.is_file() and image.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and (allowed_stems is None or image.stem in allowed_stems)
    }
    processed_images_dir = processed_dir / "images"
    processed_images_dir.mkdir(parents=True, exist_ok=True)

    rewritten = 0
    missing: list[str] = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        file_path = frame.get("file_path")
        if not isinstance(file_path, str) or not file_path:
            continue
        source_name = PurePosixPath(file_path).name
        object_image = object_images.get(Path(source_name).stem)
        if object_image is None:
            missing.append(source_name)
            continue
        rewritten_path = PurePosixPath("images") / object_image.name
        shutil.copy2(object_image, processed_dir / str(rewritten_path))
        frame["file_path"] = rewritten_path.as_posix()
        rewritten += 1

    if not rewritten and frames:
        sample = ", ".join(missing[:5])
        raise ValueError(f"could not match object images to COLMAP transforms; missing {sample}")
    write_json(transforms_path, data)
    return rewritten


def clear_colmap_sparse_points(processed_dir: Path) -> dict:
    """Keep COLMAP cameras/poses but remove sparse 3D points that can seed background Gaussians."""
    cleared = 0
    total_bytes = 0
    for points_path in sorted(processed_dir.rglob("points3D.bin")):
        if not points_path.is_file():
            continue
        original_size = points_path.stat().st_size
        backup_path = points_path.with_suffix(points_path.suffix + ".splatbot-original")
        if not backup_path.exists():
            shutil.copy2(points_path, backup_path)
        points_path.write_bytes(struct.pack("<Q", 0))
        cleared += 1
        total_bytes += original_size
    for points_path in sorted(processed_dir.rglob("points3D.txt")):
        if not points_path.is_file():
            continue
        original_size = points_path.stat().st_size
        backup_path = points_path.with_suffix(points_path.suffix + ".splatbot-original")
        if not backup_path.exists():
            shutil.copy2(points_path, backup_path)
        points_path.write_text("# Splatbot removed original-frame sparse points for object training\n", encoding="utf-8")
        cleared += 1
        total_bytes += original_size
    return {"applied": cleared > 0, "files": cleared, "original_bytes": total_bytes}


def write_processed_training_masks(processed_dir: Path, alpha_threshold: int = 16) -> int:
    transforms_path = processed_dir / "transforms.json"
    if not transforms_path.exists():
        return 0
    try:
        data = json.loads(transforms_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    frames = data.get("frames")
    if not isinstance(frames, list):
        return 0

    masks_dir = processed_dir / "masks"
    written = 0
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        file_path = frame.get("file_path")
        if not isinstance(file_path, str) or not file_path:
            continue
        image_path = processed_dir / PurePosixPath(file_path.lstrip("./")).as_posix()
        mask = load_alpha_mask(image_path)
        if mask is None:
            continue
        masks_dir.mkdir(parents=True, exist_ok=True)
        mask_name = f"{Path(PurePosixPath(file_path).name).stem}.png"
        mask_path = masks_dir / mask_name
        pixels = bytes(255 if value >= alpha_threshold else 0 for value in mask.alpha)
        write_grayscale_png(mask_path, mask.width, mask.height, pixels)
        frame["mask_path"] = (PurePosixPath("masks") / mask_name).as_posix()
        written += 1

    if written:
        write_json(transforms_path, data)
    return written


def write_grayscale_png(path: Path, width: int, height: int, pixels: bytes) -> None:
    if len(pixels) != width * height:
        raise ValueError("grayscale PNG pixel buffer has the wrong size")

    def chunk(kind: bytes, data: bytes) -> bytes:
        checksum = zlib.crc32(kind + data) & 0xFFFFFFFF
        return len(data).to_bytes(4, "big") + kind + data + checksum.to_bytes(4, "big")

    rows = bytearray()
    for y in range(height):
        rows.append(0)
        start = y * width
        rows.extend(pixels[start : start + width])
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", width.to_bytes(4, "big") + height.to_bytes(4, "big") + bytes([8, 0, 0, 0, 0]))
        + chunk(b"IDAT", zlib.compress(bytes(rows)))
        + chunk(b"IEND", b"")
    )


def inspect_ply(path: Path) -> dict:
    summary: dict = {"size_bytes": path.stat().st_size if path.exists() else None}
    if not path.exists():
        return summary
    layout = read_ply_layout(path)
    summary["parseable"] = layout is not None
    if layout is None:
        return summary
    names = [name for _, name in layout.properties]
    summary.update(
        {
            "format": layout.format,
            "vertices": layout.vertex_count,
            "properties": names,
            "has_xyz": {"x", "y", "z"}.issubset(names),
        }
    )
    if layout.vertex_count:
        try:
            with path.open("rb") as handle:
                bounds = read_ply_xyz_bounds(
                    handle,
                    layout.header_lines,
                    layout.header_bytes,
                    layout.vertex_count,
                    path,
                )
            if bounds:
                summary.update(bounds)
        except (KeyError, OSError, ValueError, struct.error):
            return summary
    return summary


def read_ply_xyz_bounds(
    handle,
    header_lines: list[str],
    header_bytes: int,
    vertices: int,
    path: Path,
    sample_limit: int = 200_000,
) -> dict | None:
    fmt_line = next((line for line in header_lines if line.startswith("format ")), "")
    properties: list[tuple[str, str]] = []
    in_vertex = False
    for line in header_lines:
        if line.startswith("element vertex "):
            in_vertex = True
            continue
        if line.startswith("element ") and in_vertex:
            break
        if in_vertex and line.startswith("property "):
            parts = line.split()
            if len(parts) == 3:
                properties.append((parts[1], parts[2]))
    names = [name for _, name in properties]
    if not {"x", "y", "z"}.issubset(names):
        return None
    mins = [float("inf"), float("inf"), float("inf")]
    maxs = [float("-inf"), float("-inf"), float("-inf")]
    if fmt_line == "format ascii 1.0":
        handle.seek(header_bytes)
        for idx, raw in enumerate(handle):
            if idx >= min(vertices, sample_limit):
                break
            parts = raw.decode("ascii", errors="ignore").split()
            if len(parts) < len(properties):
                continue
            xyz = [float(parts[names.index(axis)]) for axis in ("x", "y", "z")]
            update_bounds(mins, maxs, xyz)
    elif fmt_line == "format binary_little_endian 1.0":
        struct_format = "<" + "".join(ply_struct_code(kind) for kind, _ in properties)
        row_size = struct.calcsize(struct_format)
        xyz_indices = [names.index(axis) for axis in ("x", "y", "z")]
        handle.seek(header_bytes)
        for _ in range(min(vertices, sample_limit)):
            raw = handle.read(row_size)
            if len(raw) != row_size:
                break
            values = struct.unpack(struct_format, raw)
            xyz = [float(values[index]) for index in xyz_indices]
            update_bounds(mins, maxs, xyz)
    else:
        return None
    if not all(math.isfinite(value) for value in mins + maxs):
        return None
    extents = [maxs[idx] - mins[idx] for idx in range(3)]
    largest = max(extents)
    axis_ratio = min(extents) / largest if largest > 0 else None
    return {
        "bounds_min": mins,
        "bounds_max": maxs,
        "bounds_extent": extents,
        "flat_axis_ratio": axis_ratio,
    }


def ply_struct_code(kind: str) -> str:
    return {
        "char": "b",
        "int8": "b",
        "uchar": "B",
        "uint8": "B",
        "short": "h",
        "int16": "h",
        "ushort": "H",
        "uint16": "H",
        "int": "i",
        "int32": "i",
        "uint": "I",
        "uint32": "I",
        "float": "f",
        "float32": "f",
        "double": "d",
        "float64": "d",
    }[kind]


def ply_type_size(kind: str) -> int:
    return struct.calcsize("<" + ply_struct_code(kind))


def update_bounds(mins: list[float], maxs: list[float], xyz: list[float]) -> None:
    if not all(math.isfinite(value) for value in xyz):
        return
    for idx, value in enumerate(xyz):
        mins[idx] = min(mins[idx], value)
        maxs[idx] = max(maxs[idx], value)


def validate_ply_quality(metrics: dict, settings: Settings) -> None:
    ply = metrics.get("ply", {})
    cleaned = ply.get("cleaned", {})
    export_metrics = ply.get("export", {})
    vertices = cleaned.get("vertices")
    if cleaned.get("parseable") is not True:
        raise ValueError("Exported splat is not a parseable PLY file.")
    if cleaned.get("format") not in {"format ascii 1.0", "format binary_little_endian 1.0"}:
        raise ValueError(f"Exported splat has unsupported PLY format: {cleaned.get('format')}.")
    if vertices is None:
        raise ValueError("Exported splat PLY does not declare a vertex count.")
    if not cleaned.get("has_xyz"):
        raise ValueError("Exported splat PLY does not contain x/y/z vertex properties.")
    retention = export_metrics.get("retention_ratio")
    exported = export_metrics.get("exported_gaussians")
    total = export_metrics.get("total_gaussians")
    if isinstance(retention, int | float) and retention < settings.min_export_gaussian_retention:
        record_pipeline_event(
            metrics,
            stage="exporting",
            status="failure",
            reason="low_exported_gaussian_retention",
            recovered=False,
            details={
                "exported_gaussians": exported,
                "total_gaussians": total,
                "retention_ratio": retention,
                "expected_min_retention_ratio": settings.min_export_gaussian_retention,
            },
        )
        raise ValueError(
            "Export retained too few Gaussians "
            f"({exported}/{total}, retention {retention:.4f}); expected at least "
            f"{settings.min_export_gaussian_retention:.4f}."
        )
    if vertices is not None and vertices < settings.min_splat_vertices:
        record_pipeline_event(
            metrics,
            stage="exporting",
            status="failure",
            reason="low_splat_vertex_count",
            recovered=False,
            details={"vertices": vertices, "expected_min_vertices": settings.min_splat_vertices},
        )
        raise ValueError(
            f"Exported splat has only {vertices} vertices; expected at least {settings.min_splat_vertices}."
        )
    ratio = cleaned.get("flat_axis_ratio")
    if (
        ratio is not None
        and ratio < settings.max_flattened_axis_ratio
        and vertices is not None
        and vertices < 20_000
    ):
        record_pipeline_event(
            metrics,
            stage="exporting",
            status="failure",
            reason="flattened_splat_geometry",
            recovered=False,
            details={
                "flat_axis_ratio": ratio,
                "min_axis_ratio": settings.max_flattened_axis_ratio,
                "vertices": vertices,
            },
        )
        raise ValueError(
            f"Exported splat appears flattened (axis ratio {ratio:.4f}, vertices {vertices})."
        )
    validation = metrics.get("ply", {}).get("cleanup", {}).get("validation", {})
    if validation.get("applied") and validation.get("passed") is False:
        outside = validation.get("outside_candidate_fraction")
        low_support = validation.get("low_support_fraction")
        unobserved = validation.get("unobserved_fraction")
        record_pipeline_event(
            metrics,
            stage="postprocess",
            status="failure",
            reason="object_mask_validation_failed",
            recovered=False,
            details={"outside": outside, "low_support": low_support, "unobserved": unobserved},
        )
        raise ValueError(
            "Exported splat failed object-mask validation "
            f"(outside={outside}, low_support={low_support}, unobserved={unobserved})."
        )


def parse_ffprobe_duration(stdout: str) -> float | None:
    try:
        duration = float(stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        return None
    return duration if duration > 0 else None


def parse_ffprobe_frame_rate(stdout: str) -> float | None:
    for line in stdout.strip().splitlines():
        fps = parse_frame_rate_value(line.strip())
        if fps is not None:
            return fps
    return None


def parse_int_list(value: str) -> list[int]:
    parsed: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            parsed.append(int(item))
        except ValueError:
            continue
    return parsed


def parse_csv_list(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def parse_frame_rate_value(value: str) -> float | None:
    if not value or value == "0/0":
        return None
    try:
        if "/" in value:
            numerator, denominator = value.split("/", 1)
            fps = float(numerator) / float(denominator)
        else:
            fps = float(value)
    except (ValueError, ZeroDivisionError):
        return None
    return fps if fps > 0 else None


def format_fps(fps: float) -> str:
    return f"{fps:.3f}".rstrip("0").rstrip(".")


def log_directory_summary(label: str, path: Path, limit: int = 8) -> None:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    print(f"{label}: {len(files)} file(s) under {path}", flush=True)
    for item in files[:limit]:
        try:
            size = item.stat().st_size
        except OSError:
            size = -1
        print(f"  {item.relative_to(path)} {size} bytes", flush=True)
    if len(files) > limit:
        print(f"  ... {len(files) - limit} more file(s)", flush=True)


def log_ply_summary(label: str, path: Path) -> None:
    try:
        size = path.stat().st_size
        fmt = None
        vertices = None
        with path.open("rb") as handle:
            header_bytes = 0
            for raw_line in handle:
                header_bytes += len(raw_line)
                if header_bytes > MAX_PLY_HEADER_BYTES:
                    raise OSError(f"PLY header exceeds {MAX_PLY_HEADER_BYTES} bytes")
                line = raw_line.decode("ascii", errors="ignore").strip()
                if line.startswith("format "):
                    fmt = line
                elif line.startswith("element vertex "):
                    vertices = line.rsplit(" ", 1)[-1]
                elif line == "end_header":
                    break
        print(f"{label}: {path} size={size} format={fmt} vertices={vertices}", flush=True)
    except OSError as exc:
        print(f"{label}: could not inspect {path}: {exc}", flush=True)


def clean_exported_ply(
    src: Path,
    dest: Path,
    processed_dir: Path,
    mode: ScanMode,
    settings: Settings,
) -> dict:
    if mode != ScanMode.OBJECT or not settings.silhouette_cleanup_enabled:
        cleanup = clean_ply(src, dest)
        cleanup["silhouette"] = {"applied": False, "reason": "disabled_or_not_object_mode"}
        return cleanup

    frames = load_silhouette_frames(processed_dir, settings)
    stage_metrics: dict = {}
    intermediates: list[Path] = []
    if len(frames) < settings.silhouette_cleanup_min_views:
        cleanup = clean_ply(src, dest)
        cleanup["silhouette"] = {
            "applied": False,
            "reason": "not_enough_alpha_masks",
            "mask_views": len(frames),
        }
        current = dest
    else:
        current = src
        silhouette_dest = dest.with_suffix(dest.suffix + ".silhouette.tmp")
        evaluator = SilhouetteEvaluator(
            frames=frames,
            alpha_threshold=settings.silhouette_cleanup_alpha_threshold,
            padding_px=settings.silhouette_cleanup_padding_px,
            outside_ratio=settings.silhouette_cleanup_outside_ratio,
            max_inside_views=settings.silhouette_cleanup_max_inside_views,
            max_inside_ratio=settings.silhouette_cleanup_max_inside_ratio,
            min_views=settings.silhouette_cleanup_min_views,
        )
        cleanup = clean_ply(current, silhouette_dest, row_filter=evaluator.keep)
        input_vertices = cleanup.get("input_vertices") or 0
        removed_fraction = evaluator.removed_points / input_vertices if input_vertices else 0.0
        if removed_fraction > settings.silhouette_cleanup_max_remove_fraction:
            if silhouette_dest.exists():
                silhouette_dest.unlink()
            cleanup = clean_ply(current, dest)
            cleanup["silhouette"] = {
                "applied": False,
                "reason": "max_remove_fraction_exceeded",
                "candidate_removed": evaluator.removed_points,
                "candidate_removed_fraction": round(removed_fraction, 6),
                "max_remove_fraction": settings.silhouette_cleanup_max_remove_fraction,
                "mask_views": len(frames),
            }
            current = dest
        else:
            current = silhouette_dest
            intermediates.append(silhouette_dest)
            cleanup["silhouette"] = {
                "applied": True,
                "mask_views": len(frames),
                "checked_points": evaluator.checked_points,
                "removed_points": evaluator.removed_points,
                "removed_fraction": round(removed_fraction, 6),
                "total_observations": evaluator.total_observations,
                "outside_ratio": settings.silhouette_cleanup_outside_ratio,
                "max_inside_views": settings.silhouette_cleanup_max_inside_views,
                "max_inside_ratio": settings.silhouette_cleanup_max_inside_ratio,
                "padding_px": settings.silhouette_cleanup_padding_px,
                "alpha_threshold": settings.silhouette_cleanup_alpha_threshold,
            }

    mask_support_dest = dest.with_suffix(dest.suffix + ".mask-support.tmp")
    mask_support_cleanup = clean_mask_support_outliers(current, mask_support_dest, frames, settings)
    stage_metrics["mask_support"] = mask_support_cleanup
    if mask_support_cleanup.get("applied"):
        current = mask_support_dest
        intermediates.append(mask_support_dest)
    elif mask_support_dest.exists():
        mask_support_dest.unlink()

    depth_dest = dest.with_suffix(dest.suffix + ".depth.tmp")
    depth_cleanup = clean_depth_consistency_outliers(current, depth_dest, frames, settings)
    stage_metrics["depth_consistency"] = depth_cleanup
    if depth_cleanup.get("applied"):
        current = depth_dest
        intermediates.append(depth_dest)
    elif depth_dest.exists():
        depth_dest.unlink()

    gaussian_dest = dest.with_suffix(dest.suffix + ".gaussian.tmp")
    gaussian_cleanup = clean_gaussian_properties(current, gaussian_dest, settings)
    stage_metrics["gaussian"] = gaussian_cleanup
    if gaussian_cleanup.get("applied"):
        current = gaussian_dest
        intermediates.append(gaussian_dest)
    elif gaussian_dest.exists():
        gaussian_dest.unlink()

    spatial_dest = dest.with_suffix(dest.suffix + ".spatial.tmp")
    spatial_cleanup = clean_spatial_outliers(current, spatial_dest, settings)
    stage_metrics["spatial"] = spatial_cleanup
    if spatial_cleanup.get("applied"):
        current = spatial_dest
        intermediates.append(spatial_dest)
    elif spatial_dest.exists():
        spatial_dest.unlink()

    if current != dest:
        shutil.copy2(current, dest)
    for temp in intermediates:
        if temp.exists() and temp != dest:
            temp.unlink()
    cleanup["stages"] = stage_metrics
    cleanup["validation"] = validate_postprocess_against_masks(dest, frames, settings)
    final_summary = clean_ply(dest, dest.with_suffix(dest.suffix + ".validated.tmp"))
    validated = dest.with_suffix(dest.suffix + ".validated.tmp")
    if not validated.exists():
        raise RuntimeError(f"final PLY validation did not create {validated}")
    validated.replace(dest)
    cleanup["final_validation"] = final_summary
    cleanup["output_vertices"] = final_summary.get("output_vertices", cleanup.get("output_vertices"))
    cleanup["filtered_vertices_removed"] = (
        (cleanup.get("filtered_vertices_removed") or 0)
        + (mask_support_cleanup.get("filtered_vertices_removed") or 0)
        + (depth_cleanup.get("filtered_vertices_removed") or 0)
        + (gaussian_cleanup.get("filtered_vertices_removed") or 0)
        + (spatial_cleanup.get("filtered_vertices_removed") or 0)
    )
    return cleanup


def clean_ply(src: Path, dest: Path, row_filter: Callable[[tuple[float, ...]], bool] | None = None) -> dict:
    """Remove invalid vertex rows while preserving PLY properties and binary layout."""
    layout = read_ply_layout(src)
    if layout is None:
        raise ValueError(f"Invalid or unparseable PLY file: {src}")
    if layout.format == "format ascii 1.0":
        return clean_ascii_ply(src, dest, layout, row_filter)
    if layout.format == "format binary_little_endian 1.0" and layout.row_size:
        return clean_binary_ply(src, dest, layout, row_filter)
    raise ValueError(f"Unsupported PLY format or vertex layout: {layout.format}")


def clean_mask_support_outliers(
    src: Path,
    dest: Path,
    frames: list[SilhouetteFrame],
    settings: Settings,
) -> dict:
    if not settings.mask_support_cleanup_enabled:
        return {"applied": False, "reason": "disabled"}
    if len(frames) < settings.mask_support_cleanup_min_views:
        return {"applied": False, "reason": "not_enough_alpha_masks", "mask_views": len(frames)}
    layout = read_ply_layout(src)
    if layout is None:
        return {"applied": False, "reason": "missing_layout"}

    removed = 0

    def keep(values: tuple[float, ...]) -> bool:
        nonlocal removed
        if len(values) < 3:
            return True
        support = point_mask_support(
            values[:3],
            frames,
            alpha_threshold=settings.silhouette_cleanup_alpha_threshold,
            padding_px=settings.silhouette_cleanup_padding_px,
        )
        if support.observed < settings.mask_support_cleanup_min_views:
            return True
        supported = (
            support.inside >= settings.mask_support_cleanup_min_inside_views
            or support.inside_fraction >= settings.mask_support_cleanup_min_inside_ratio
        )
        if supported:
            return True
        removed += 1
        return False

    cleanup = clean_ply(src, dest, row_filter=keep)
    input_vertices = cleanup.get("input_vertices") or 0
    removed_fraction = removed / input_vertices if input_vertices else 0.0
    if removed_fraction > settings.mask_support_cleanup_max_remove_fraction:
        if dest.exists():
            dest.unlink()
        return {
            "applied": False,
            "reason": "max_remove_fraction_exceeded",
            "candidate_removed": removed,
            "candidate_removed_fraction": round(removed_fraction, 6),
            "max_remove_fraction": settings.mask_support_cleanup_max_remove_fraction,
            "mask_views": len(frames),
        }
    cleanup.update(
        {
            "applied": removed > 0,
            "removed_fraction": round(removed_fraction, 6),
            "mask_views": len(frames),
            "min_views": settings.mask_support_cleanup_min_views,
            "min_inside_views": settings.mask_support_cleanup_min_inside_views,
            "min_inside_ratio": settings.mask_support_cleanup_min_inside_ratio,
        }
    )
    return cleanup


def clean_depth_consistency_outliers(
    src: Path,
    dest: Path,
    frames: list[SilhouetteFrame],
    settings: Settings,
) -> dict:
    if not settings.depth_consistency_cleanup_enabled:
        return {"applied": False, "reason": "disabled"}
    layout = read_ply_layout(src)
    if layout is None:
        return {"applied": False, "reason": "missing_layout"}
    depth_frames = load_depth_consistency_frames(frames, src, layout, settings)
    if not depth_frames:
        return {"applied": False, "reason": "no_aligned_depth_priors"}
    points = [values[:3] for values in iter_ply_vertex_values(src, layout) if len(values) >= 3]
    if not points:
        return {"applied": False, "reason": "no_points"}

    remove_indices: set[int] = set()
    max_ratio = max(settings.depth_consistency_cleanup_max_depth_ratio, 1.001)
    min_views = max(1, settings.depth_consistency_cleanup_min_views)
    min_inconsistent_ratio = settings.depth_consistency_cleanup_min_inconsistent_ratio
    for idx, point in enumerate(points):
        checked = 0
        inconsistent = 0
        for depth_frame in depth_frames:
            projection = project_world_point_with_depth(depth_frame.frame, point)
            if projection is None:
                continue
            u, v, alt_v, point_depth = projection
            if not projection_hits_foreground(
                depth_frame.frame,
                u,
                v,
                alt_v,
                settings.silhouette_cleanup_alpha_threshold,
                settings.silhouette_cleanup_padding_px,
            ):
                continue
            raw_depth = sample_depth_map(depth_frame.depth_map, depth_frame.frame, u, v, alt_v)
            if raw_depth is None:
                continue
            expected_depth = raw_depth * depth_frame.scale
            if expected_depth <= 1e-6:
                continue
            checked += 1
            ratio = max(point_depth / expected_depth, expected_depth / point_depth)
            if ratio > max_ratio:
                inconsistent += 1
        if checked >= min_views and (inconsistent / checked) >= min_inconsistent_ratio:
            remove_indices.add(idx)

    if not remove_indices:
        return {
            "applied": False,
            "reason": "no_depth_inconsistent_points",
            "points": len(points),
            "depth_views": len(depth_frames),
        }
    removed_fraction = len(remove_indices) / len(points)
    if removed_fraction > settings.depth_consistency_cleanup_max_remove_fraction:
        return {
            "applied": False,
            "reason": "max_remove_fraction_exceeded",
            "candidate_removed": len(remove_indices),
            "candidate_removed_fraction": round(removed_fraction, 6),
            "max_remove_fraction": settings.depth_consistency_cleanup_max_remove_fraction,
            "depth_views": len(depth_frames),
        }

    row_index = -1

    def keep_by_index(_: tuple[float, ...]) -> bool:
        nonlocal row_index
        row_index += 1
        return row_index not in remove_indices

    cleanup = clean_ply(src, dest, row_filter=keep_by_index)
    cleanup.update(
        {
            "applied": True,
            "removed_fraction": round(removed_fraction, 6),
            "depth_views": len(depth_frames),
            "max_depth_ratio": settings.depth_consistency_cleanup_max_depth_ratio,
            "min_views": min_views,
            "min_inconsistent_ratio": min_inconsistent_ratio,
            "scale_samples": sum(frame.scale_samples for frame in depth_frames),
        }
    )
    return cleanup


def clean_gaussian_properties(src: Path, dest: Path, settings: Settings) -> dict:
    if not settings.gaussian_cleanup_enabled:
        return {"applied": False, "reason": "disabled"}
    layout = read_ply_layout(src)
    if layout is None:
        return {"applied": False, "reason": "missing_layout"}
    property_index = {name: idx for idx, (_, name) in enumerate(layout.properties)}
    opacity_idx = property_index.get("opacity")
    scale_indices = [property_index[name] for name in ("scale_0", "scale_1", "scale_2") if name in property_index]
    if opacity_idx is None and len(scale_indices) < 3:
        return {"applied": False, "reason": "no_gaussian_properties"}

    scale_max_values = [
        max(values[idx] for idx in scale_indices)
        for values in iter_ply_vertex_values(src, layout)
        if len(scale_indices) == 3 and all(math.isfinite(values[idx]) for idx in scale_indices)
    ]
    median_scale = float(median(scale_max_values)) if scale_max_values else None
    max_scale_delta = math.log(max(settings.gaussian_cleanup_max_scale_ratio, 1.001))
    max_anisotropy_delta = math.log(max(settings.gaussian_cleanup_max_anisotropy, 1.001))

    def keep(values: tuple[float, ...]) -> bool:
        if opacity_idx is not None and values[opacity_idx] < settings.gaussian_cleanup_min_opacity:
            return False
        if len(scale_indices) == 3:
            scales = [values[idx] for idx in scale_indices]
            if not all(math.isfinite(value) for value in scales):
                return False
            if max(scales) - min(scales) > max_anisotropy_delta:
                return False
            if median_scale is not None and max(scales) - median_scale > max_scale_delta:
                return False
        return True

    cleanup = clean_ply(src, dest, row_filter=keep)
    input_vertices = cleanup.get("input_vertices") or 0
    filtered = cleanup.get("filtered_vertices_removed") or 0
    removed_fraction = filtered / input_vertices if input_vertices else 0.0
    if removed_fraction > settings.gaussian_cleanup_max_remove_fraction:
        if dest.exists():
            dest.unlink()
        return {
            "applied": False,
            "reason": "max_remove_fraction_exceeded",
            "candidate_removed": filtered,
            "candidate_removed_fraction": round(removed_fraction, 6),
            "max_remove_fraction": settings.gaussian_cleanup_max_remove_fraction,
        }
    cleanup.update(
        {
            "applied": filtered > 0,
            "removed_fraction": round(removed_fraction, 6),
            "min_opacity": settings.gaussian_cleanup_min_opacity,
            "max_scale_ratio": settings.gaussian_cleanup_max_scale_ratio,
            "max_anisotropy": settings.gaussian_cleanup_max_anisotropy,
            "median_scale": round(median_scale, 6) if median_scale is not None else None,
        }
    )
    return cleanup


def clean_spatial_outliers(src: Path, dest: Path, settings: Settings) -> dict:
    if not settings.spatial_cleanup_enabled:
        return {"applied": False, "reason": "disabled"}
    layout = read_ply_layout(src)
    if layout is None:
        return {"applied": False, "reason": "missing_layout"}
    points = [values[:3] for values in iter_ply_vertex_values(src, layout) if len(values) >= 3]
    if len(points) < 5:
        return {"applied": False, "reason": "too_few_points", "points": len(points)}
    bounds = xyz_bounds(points)
    if bounds is None:
        return {"applied": False, "reason": "invalid_bounds"}
    diagonal = math.sqrt(sum((bounds[1][idx] - bounds[0][idx]) ** 2 for idx in range(3)))
    if diagonal <= 0:
        return {"applied": False, "reason": "zero_extent"}

    remove_indices: set[int] = set()
    outlier_indices = spatial_radius_outlier_indices(
        points,
        radius=max(diagonal * settings.spatial_outlier_radius_fraction, 1e-6),
        min_neighbors=settings.spatial_outlier_min_neighbors,
    )
    remove_indices.update(outlier_indices)
    component_indices = spatial_small_component_indices(
        points,
        voxel_size=max(diagonal * settings.spatial_component_voxel_fraction, 1e-6),
        min_vertices=max(
            settings.spatial_component_min_vertices,
            int(len(points) * settings.spatial_component_min_fraction),
        ),
    )
    remove_indices.update(component_indices)
    if not remove_indices:
        return {
            "applied": False,
            "reason": "no_spatial_outliers",
            "points": len(points),
            "radius_outliers": len(outlier_indices),
            "small_component_points": len(component_indices),
        }

    removed_fraction = len(remove_indices) / len(points)
    if removed_fraction > settings.spatial_cleanup_max_remove_fraction:
        return {
            "applied": False,
            "reason": "max_remove_fraction_exceeded",
            "candidate_removed": len(remove_indices),
            "candidate_removed_fraction": round(removed_fraction, 6),
            "max_remove_fraction": settings.spatial_cleanup_max_remove_fraction,
            "radius_outliers": len(outlier_indices),
            "small_component_points": len(component_indices),
        }

    remove_xyz = {points[idx] for idx in remove_indices}
    cleanup = clean_ply(src, dest, row_filter=lambda values: values[:3] not in remove_xyz)
    cleanup.update(
        {
            "applied": True,
            "removed_fraction": round(removed_fraction, 6),
            "radius_outliers": len(outlier_indices),
            "small_component_points": len(component_indices),
            "radius_fraction": settings.spatial_outlier_radius_fraction,
            "component_voxel_fraction": settings.spatial_component_voxel_fraction,
        }
    )
    return cleanup


def validate_postprocess_against_masks(
    ply_path: Path,
    frames: list[SilhouetteFrame],
    settings: Settings,
) -> dict:
    if not settings.postprocess_validation_enabled:
        return {"applied": False, "reason": "disabled"}
    if len(frames) < settings.silhouette_cleanup_min_views:
        return {"applied": False, "reason": "not_enough_alpha_masks", "mask_views": len(frames)}
    layout = read_ply_layout(ply_path)
    if layout is None:
        return {"applied": False, "reason": "missing_layout"}
    sample_limit = max(1, settings.postprocess_validation_sample_limit)
    stride = max(1, math.ceil(layout.vertex_count / sample_limit))
    sampled_points = 0
    checked_points = 0
    unobserved_points = 0
    outside_candidate_points = 0
    low_support_points = 0
    total_observations = 0
    for idx, values in enumerate(iter_ply_vertex_values(ply_path, layout)):
        if idx % stride != 0:
            continue
        if len(values) < 3:
            continue
        sampled_points += 1
        support = point_mask_support(
            values[:3],
            frames,
            alpha_threshold=settings.silhouette_cleanup_alpha_threshold,
            padding_px=settings.silhouette_cleanup_padding_px,
        )
        total_observations += support.observed
        if support.observed < settings.silhouette_cleanup_min_views:
            unobserved_points += 1
            continue
        checked_points += 1
        inside_is_low = (
            support.inside <= settings.silhouette_cleanup_max_inside_views
            or support.inside_fraction <= settings.silhouette_cleanup_max_inside_ratio
        )
        if support.outside_fraction >= settings.silhouette_cleanup_outside_ratio and inside_is_low:
            outside_candidate_points += 1
        if (
            support.inside < settings.postprocess_validation_min_inside_views
            and support.inside_fraction < settings.postprocess_validation_min_inside_ratio
        ):
            low_support_points += 1

    outside_fraction = outside_candidate_points / checked_points if checked_points else 0.0
    low_support_fraction = low_support_points / checked_points if checked_points else 0.0
    unobserved_fraction = unobserved_points / sampled_points if sampled_points else 0.0
    passed = (
        checked_points >= settings.postprocess_validation_min_checked_points
        and outside_fraction <= settings.postprocess_validation_max_outside_fraction
        and low_support_fraction <= settings.postprocess_validation_max_low_support_fraction
        and unobserved_fraction <= settings.postprocess_validation_max_unobserved_fraction
    )
    return {
        "applied": True,
        "mask_views": len(frames),
        "sampled_points": sampled_points,
        "checked_points": checked_points,
        "unobserved_points": unobserved_points,
        "unobserved_fraction": round(unobserved_fraction, 6),
        "outside_candidate_points": outside_candidate_points,
        "outside_candidate_fraction": round(outside_fraction, 6),
        "low_support_points": low_support_points,
        "low_support_fraction": round(low_support_fraction, 6),
        "total_observations": total_observations,
        "passed": passed,
        "max_outside_fraction": settings.postprocess_validation_max_outside_fraction,
        "max_low_support_fraction": settings.postprocess_validation_max_low_support_fraction,
        "min_checked_points": settings.postprocess_validation_min_checked_points,
        "max_unobserved_fraction": settings.postprocess_validation_max_unobserved_fraction,
        "min_inside_views": settings.postprocess_validation_min_inside_views,
        "min_inside_ratio": settings.postprocess_validation_min_inside_ratio,
    }


def read_ply_layout(path: Path) -> PlyLayout | None:
    header_lines: list[str] = []
    header_bytes = 0
    with path.open("rb") as handle:
        while True:
            raw = handle.readline()
            if not raw:
                return None
            header_bytes += len(raw)
            if header_bytes > MAX_PLY_HEADER_BYTES:
                return None
            line = raw.decode("ascii", errors="ignore").strip()
            header_lines.append(line)
            if line == "end_header":
                break
    fmt = next((line for line in header_lines if line.startswith("format ")), "")
    vertex_count: int | None = None
    vertex_line_index = -1
    properties: list[tuple[str, str]] = []
    in_vertex = False
    for idx, line in enumerate(header_lines):
        if line.startswith("element vertex "):
            try:
                vertex_count = int(line.rsplit(" ", 1)[-1])
            except ValueError:
                return None
            vertex_line_index = idx
            in_vertex = True
            continue
        if line.startswith("element ") and in_vertex:
            in_vertex = False
        if in_vertex and line.startswith("property "):
            parts = line.split()
            if len(parts) != 3:
                return None
            properties.append((parts[1], parts[2]))
    if vertex_count is None or vertex_line_index < 0 or not properties:
        return None
    try:
        row_size = sum(ply_type_size(kind) for kind, _ in properties)
    except KeyError:
        row_size = None
    return PlyLayout(
        format=fmt,
        header_lines=header_lines,
        header_bytes=header_bytes,
        vertex_count=vertex_count,
        vertex_line_index=vertex_line_index,
        properties=properties,
        row_size=row_size,
    )


def iter_ply_vertex_values(path: Path, layout: PlyLayout | None = None):
    layout = layout or read_ply_layout(path)
    if layout is None:
        return
    if layout.format == "format ascii 1.0":
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for _ in range(len(layout.header_lines)):
                if not handle.readline():
                    return
            for _ in range(layout.vertex_count):
                line = handle.readline()
                if not line:
                    return
                try:
                    values = tuple(float(part) for part in line.split())
                except ValueError:
                    continue
                if len(values) >= len(layout.properties) and all(math.isfinite(value) for value in values):
                    yield values
        return
    if layout.format == "format binary_little_endian 1.0" and layout.row_size:
        struct_format = "<" + "".join(ply_struct_code(kind) for kind, _ in layout.properties)
        with path.open("rb") as handle:
            handle.seek(layout.header_bytes)
            for _ in range(layout.vertex_count):
                row = handle.read(layout.row_size)
                if len(row) != layout.row_size:
                    return
                values = tuple(float(value) for value in struct.unpack(struct_format, row))
                if all(math.isfinite(value) for value in values):
                    yield values


def xyz_bounds(points: list[tuple[float, float, float]]) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    if not points:
        return None
    mins = [min(point[idx] for point in points) for idx in range(3)]
    maxs = [max(point[idx] for point in points) for idx in range(3)]
    return (mins[0], mins[1], mins[2]), (maxs[0], maxs[1], maxs[2])


def voxel_key(point: tuple[float, float, float], cell_size: float) -> tuple[int, int, int]:
    return (
        math.floor(point[0] / cell_size),
        math.floor(point[1] / cell_size),
        math.floor(point[2] / cell_size),
    )


def neighbor_voxel_keys(key: tuple[int, int, int]):
    x, y, z = key
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                yield x + dx, y + dy, z + dz


def spatial_radius_outlier_indices(
    points: list[tuple[float, float, float]],
    radius: float,
    min_neighbors: int,
) -> set[int]:
    if min_neighbors <= 0:
        return set()
    cells: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for idx, point in enumerate(points):
        cells[voxel_key(point, radius)].append(idx)
    radius_sq = radius * radius
    outliers: set[int] = set()
    for idx, point in enumerate(points):
        count = 0
        for key in neighbor_voxel_keys(voxel_key(point, radius)):
            for other_idx in cells.get(key, []):
                if other_idx == idx:
                    continue
                other = points[other_idx]
                distance_sq = (
                    (point[0] - other[0]) ** 2
                    + (point[1] - other[1]) ** 2
                    + (point[2] - other[2]) ** 2
                )
                if distance_sq <= radius_sq:
                    count += 1
                    if count >= min_neighbors:
                        break
            if count >= min_neighbors:
                break
        if count < min_neighbors:
            outliers.add(idx)
    return outliers


def spatial_small_component_indices(
    points: list[tuple[float, float, float]],
    voxel_size: float,
    min_vertices: int,
) -> set[int]:
    if min_vertices <= 1:
        return set()
    cells: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for idx, point in enumerate(points):
        cells[voxel_key(point, voxel_size)].append(idx)
    occupied = set(cells)
    visited: set[tuple[int, int, int]] = set()
    remove: set[int] = set()
    for start in occupied:
        if start in visited:
            continue
        queue = deque([start])
        visited.add(start)
        component_cells: list[tuple[int, int, int]] = []
        component_count = 0
        while queue:
            key = queue.popleft()
            component_cells.append(key)
            component_count += len(cells[key])
            for neighbor in neighbor_voxel_keys(key):
                if neighbor in occupied and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        if component_count < min_vertices:
            for key in component_cells:
                remove.update(cells[key])
    return remove


def clean_ascii_ply(
    src: Path,
    dest: Path,
    layout: PlyLayout,
    row_filter: Callable[[tuple[float, ...]], bool] | None,
) -> dict:
    temp_body = dest.with_suffix(dest.suffix + ".body.tmp")
    invalid = 0
    filtered = 0
    kept = 0
    with src.open("r", encoding="utf-8", errors="replace") as source, temp_body.open("w", encoding="utf-8") as body:
        for _ in range(len(layout.header_lines)):
            if not source.readline():
                raise ValueError(f"{src} is missing a PLY header")
        for _ in range(layout.vertex_count):
            line = source.readline()
            if not line:
                invalid += 1
                break
            stripped = line.rstrip("\n")
            parts = stripped.split()
            if not parts:
                invalid += 1
                continue
            try:
                values = tuple(float(part) for part in parts)
            except ValueError:
                invalid += 1
                continue
            if not all(math.isfinite(value) for value in values):
                invalid += 1
                continue
            if row_filter is not None and not row_filter(values):
                filtered += 1
                continue
            body.write(stripped + "\n")
            kept += 1
        tail = source.read()

    updated_header = "\n".join(updated_ply_header(layout.header_lines, kept)) + "\n"
    try:
        with dest.open("w", encoding="utf-8") as output:
            output.write(updated_header)
            with temp_body.open("r", encoding="utf-8") as body:
                shutil.copyfileobj(body, output)
            output.write(tail)
    finally:
        temp_body.unlink(missing_ok=True)
    return {
        "input_vertices": layout.vertex_count,
        "output_vertices": kept,
        "invalid_vertices_removed": invalid,
        "filtered_vertices_removed": filtered,
    }


def clean_binary_ply(
    src: Path,
    dest: Path,
    layout: PlyLayout,
    row_filter: Callable[[tuple[float, ...]], bool] | None,
) -> dict:
    assert layout.row_size is not None
    struct_format = "<" + "".join(ply_struct_code(kind) for kind, _ in layout.properties)
    vertex_data_end = layout.header_bytes + (layout.vertex_count * layout.row_size)
    temp_body = dest.with_suffix(dest.suffix + ".body.tmp")
    invalid = 0
    filtered = 0
    kept = 0
    with src.open("rb") as handle, temp_body.open("wb") as body:
        handle.seek(layout.header_bytes)
        for _ in range(layout.vertex_count):
            row = handle.read(layout.row_size)
            if len(row) != layout.row_size:
                invalid += 1
                break
            values = tuple(float(value) for value in struct.unpack(struct_format, row))
            if not all(math.isfinite(value) for value in values):
                invalid += 1
                continue
            if row_filter is not None and not row_filter(values):
                filtered += 1
                continue
            body.write(row)
            kept += 1
        handle.seek(vertex_data_end)
        tail = handle.read()

    header = "\n".join(updated_ply_header(layout.header_lines, kept)) + "\n"
    try:
        with dest.open("wb") as handle:
            handle.write(header.encode("ascii"))
            with temp_body.open("rb") as body:
                shutil.copyfileobj(body, handle)
            handle.write(tail)
    finally:
        temp_body.unlink(missing_ok=True)
    return {
        "input_vertices": layout.vertex_count,
        "output_vertices": kept,
        "invalid_vertices_removed": invalid,
        "filtered_vertices_removed": filtered,
    }


def updated_ply_header(header_lines: list[str], vertex_count: int) -> list[str]:
    updated: list[str] = []
    for line in header_lines:
        if line.startswith("element vertex "):
            updated.append(f"element vertex {vertex_count}")
        else:
            updated.append(line)
    return updated


def load_silhouette_frames(processed_dir: Path, settings: Settings) -> list[SilhouetteFrame]:
    transforms_path = processed_dir / "transforms.json"
    if not transforms_path.exists():
        return []
    try:
        data = json.loads(transforms_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    raw_frames = [frame for frame in data.get("frames", []) if isinstance(frame, dict)]
    if settings.silhouette_cleanup_max_views > 0:
        raw_frames = quality_blind_even_sample(raw_frames, min(settings.silhouette_cleanup_max_views, len(raw_frames)))

    frames: list[SilhouetteFrame] = []
    for frame in raw_frames:
        file_path = frame.get("file_path")
        transform = frame.get("transform_matrix")
        if not isinstance(file_path, str) or not isinstance(transform, list):
            continue
        mask_path = processed_dir / PurePosixPath(file_path.lstrip("./")).as_posix()
        mask = load_alpha_mask(mask_path)
        if mask is None:
            continue
        fl_x = float(frame.get("fl_x") or data.get("fl_x") or 0.0)
        fl_y = float(frame.get("fl_y") or data.get("fl_y") or fl_x)
        cx = float(frame.get("cx") or data.get("cx") or (mask.width / 2.0))
        cy = float(frame.get("cy") or data.get("cy") or (mask.height / 2.0))
        if fl_x <= 0 or fl_y <= 0:
            continue
        world_to_camera = invert_camera_transform(transform)
        if world_to_camera is None:
            continue
        depth_path = None
        raw_depth_path = frame.get("depth_file_path")
        if isinstance(raw_depth_path, str) and raw_depth_path:
            depth_path = processed_dir / PurePosixPath(raw_depth_path.lstrip("./")).as_posix()
        frames.append(
            SilhouetteFrame(
                mask=mask,
                world_to_camera=world_to_camera,
                fl_x=fl_x,
                fl_y=fl_y,
                cx=cx,
                cy=cy,
                depth_path=depth_path,
            )
        )
    return frames


def invert_camera_transform(transform: list) -> list[list[float]] | None:
    try:
        matrix = [[float(transform[row][col]) for col in range(4)] for row in range(4)]
    except (TypeError, ValueError, IndexError):
        return None
    rotation = [[matrix[row][col] for col in range(3)] for row in range(3)]
    translation = [matrix[row][3] for row in range(3)]
    inverse_rotation = [[rotation[row][col] for row in range(3)] for col in range(3)]
    inverse_translation = [
        -sum(inverse_rotation[row][col] * translation[col] for col in range(3))
        for row in range(3)
    ]
    return [
        inverse_rotation[0] + [inverse_translation[0]],
        inverse_rotation[1] + [inverse_translation[1]],
        inverse_rotation[2] + [inverse_translation[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def project_world_point(
    frame: SilhouetteFrame,
    xyz: tuple[float, float, float],
) -> tuple[float, float, float | None] | None:
    projection = project_world_point_with_depth(frame, xyz)
    if projection is None:
        return None
    u, v, alt_v, _ = projection
    return u, v, alt_v


def project_world_point_with_depth(
    frame: SilhouetteFrame,
    xyz: tuple[float, float, float],
) -> tuple[float, float, float | None, float] | None:
    x, y, z = xyz
    matrix = frame.world_to_camera
    cam_x = matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z + matrix[0][3]
    cam_y = matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z + matrix[1][3]
    cam_z = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z + matrix[2][3]
    depth = -cam_z
    if depth <= 1e-6:
        return None
    u = frame.fl_x * (cam_x / depth) + frame.cx
    v = frame.fl_y * (cam_y / depth) + frame.cy
    alt_v = frame.cy - frame.fl_y * (cam_y / depth)
    return u, v, alt_v, depth


def projection_hits_foreground(
    frame: SilhouetteFrame,
    u: float,
    v: float,
    alt_v: float | None,
    alpha_threshold: int,
    padding_px: int,
) -> bool:
    u_px = round(u)
    if u_px < 0 or u_px >= frame.mask.width:
        return False
    v_px = round(v)
    alt_v_px = round(alt_v) if alt_v is not None else None
    v_in_bounds = 0 <= v_px < frame.mask.height
    alt_v_in_bounds = alt_v_px is not None and 0 <= alt_v_px < frame.mask.height
    if not v_in_bounds and not alt_v_in_bounds:
        return False
    foreground = v_in_bounds and frame.mask.is_foreground(u, v, alpha_threshold, padding_px)
    if not foreground and alt_v_in_bounds:
        foreground = frame.mask.is_foreground(u, alt_v, alpha_threshold, padding_px)
    return foreground


def load_depth_consistency_frames(
    frames: list[SilhouetteFrame],
    ply_path: Path,
    layout: PlyLayout,
    settings: Settings,
) -> list[DepthConsistencyFrame]:
    points = [values[:3] for values in iter_ply_vertex_values(ply_path, layout) if len(values) >= 3]
    if not points:
        return []
    depth_frames: list[DepthConsistencyFrame] = []
    for frame in frames:
        if frame.depth_path is None:
            continue
        depth_map = load_depth_map(frame.depth_path)
        if depth_map is None:
            continue
        scale, samples = estimate_depth_scale(points, frame, depth_map, settings)
        if scale is None or samples < settings.depth_consistency_cleanup_min_scale_samples:
            continue
        depth_frames.append(DepthConsistencyFrame(frame=frame, depth_map=depth_map, scale=scale, scale_samples=samples))
    return depth_frames


def load_depth_map(path: Path):
    try:
        import numpy as np  # type: ignore[import-not-found]
    except Exception:
        return None
    try:
        depth = np.asarray(np.load(path), dtype="float32").squeeze()
    except Exception:
        return None
    if depth.ndim != 2 or depth.size == 0:
        return None
    return depth


def estimate_depth_scale(
    points: list[tuple[float, float, float]],
    frame: SilhouetteFrame,
    depth_map,
    settings: Settings,
) -> tuple[float | None, int]:
    sample_limit = max(1, settings.depth_consistency_cleanup_alignment_sample_limit)
    stride = max(1, math.ceil(len(points) / sample_limit))
    ratios: list[float] = []
    for idx, point in enumerate(points):
        if idx % stride != 0:
            continue
        projection = project_world_point_with_depth(frame, point)
        if projection is None:
            continue
        u, v, alt_v, point_depth = projection
        if not projection_hits_foreground(
            frame,
            u,
            v,
            alt_v,
            settings.silhouette_cleanup_alpha_threshold,
            settings.silhouette_cleanup_padding_px,
        ):
            continue
        raw_depth = sample_depth_map(depth_map, frame, u, v, alt_v)
        if raw_depth is not None and raw_depth > 1e-6:
            ratios.append(point_depth / raw_depth)
    if not ratios:
        return None, 0
    return float(median(ratios)), len(ratios)


def sample_depth_map(depth_map, frame: SilhouetteFrame, u: float, v: float, alt_v: float | None) -> float | None:
    height, width = depth_map.shape[:2]
    scale_x = width / max(frame.mask.width, 1)
    scale_y = height / max(frame.mask.height, 1)

    def sample(y_value: float | None) -> float | None:
        if y_value is None:
            return None
        x_px = round(u * scale_x)
        y_px = round(y_value * scale_y)
        if x_px < 0 or x_px >= width or y_px < 0 or y_px >= height:
            return None
        value = float(depth_map[y_px, x_px])
        return value if math.isfinite(value) and value > 0 else None

    return sample(v) or sample(alt_v)


def point_mask_support(
    xyz: tuple[float, float, float],
    frames: list[SilhouetteFrame],
    alpha_threshold: int,
    padding_px: int,
) -> PointMaskSupport:
    observed = 0
    outside = 0
    inside = 0
    for frame in frames:
        projection = project_world_point(frame, xyz)
        if projection is None:
            continue
        u, v, alt_v = projection
        u_px = round(u)
        if u_px < 0 or u_px >= frame.mask.width:
            continue
        v_px = round(v)
        alt_v_px = round(alt_v) if alt_v is not None else None
        v_in_bounds = 0 <= v_px < frame.mask.height
        alt_v_in_bounds = alt_v_px is not None and 0 <= alt_v_px < frame.mask.height
        if not v_in_bounds and not alt_v_in_bounds:
            continue
        observed += 1
        foreground = v_in_bounds and frame.mask.is_foreground(u, v, alpha_threshold, padding_px)
        if not foreground and alt_v_in_bounds:
            # Be conservative across camera-y conventions: an alternate-y hit
            # means this point may be legitimate, so do not count it outside.
            foreground = frame.mask.is_foreground(u, alt_v, alpha_threshold, padding_px)
        if foreground:
            inside += 1
        else:
            outside += 1
    return PointMaskSupport(observed=observed, inside=inside, outside=outside)


def load_alpha_mask(path: Path) -> AlphaMask | None:
    try:
        import cv2  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return load_png_alpha_mask(path)
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or len(image.shape) < 3 or image.shape[2] < 4:
        return load_png_alpha_mask(path)
    height, width = image.shape[:2]
    return AlphaMask(width=width, height=height, alpha=image[:, :, 3].tobytes())


def load_png_alpha_mask(path: Path) -> AlphaMask | None:
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return None
    pos = 8
    width = height = bit_depth = color_type = None
    idat = bytearray()
    while pos + 8 <= len(raw):
        length = int.from_bytes(raw[pos : pos + 4], "big")
        chunk_type = raw[pos + 4 : pos + 8]
        chunk_data = raw[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if chunk_type == b"IHDR":
            width = int.from_bytes(chunk_data[0:4], "big")
            height = int.from_bytes(chunk_data[4:8], "big")
            bit_depth = chunk_data[8]
            color_type = chunk_data[9]
            interlace = chunk_data[12]
            if bit_depth != 8 or interlace != 0 or color_type not in {4, 6}:
                return None
        elif chunk_type == b"IDAT":
            idat.extend(chunk_data)
        elif chunk_type == b"IEND":
            break
    if width is None or height is None or color_type is None:
        return None
    channels = 4 if color_type == 6 else 2
    stride = width * channels
    try:
        decompressed = zlib.decompress(bytes(idat))
    except zlib.error:
        return None
    rows: list[bytes] = []
    alpha = bytearray(width * height)
    offset = 0
    previous = bytes(stride)
    for row_idx in range(height):
        if offset >= len(decompressed):
            return None
        filter_type = decompressed[offset]
        offset += 1
        row = bytearray(decompressed[offset : offset + stride])
        offset += stride
        if len(row) != stride:
            return None
        unfilter_png_row(row, previous, filter_type, channels)
        rows.append(bytes(row))
        previous = rows[-1]
        alpha_offset = row_idx * width
        for x in range(width):
            alpha[alpha_offset + x] = row[(x * channels) + (channels - 1)]
    return AlphaMask(width=width, height=height, alpha=bytes(alpha))


def unfilter_png_row(row: bytearray, previous: bytes, filter_type: int, bytes_per_pixel: int) -> None:
    if filter_type == 0:
        return
    for idx in range(len(row)):
        left = row[idx - bytes_per_pixel] if idx >= bytes_per_pixel else 0
        up = previous[idx] if idx < len(previous) else 0
        up_left = previous[idx - bytes_per_pixel] if idx >= bytes_per_pixel and idx < len(previous) else 0
        if filter_type == 1:
            row[idx] = (row[idx] + left) & 0xFF
        elif filter_type == 2:
            row[idx] = (row[idx] + up) & 0xFF
        elif filter_type == 3:
            row[idx] = (row[idx] + ((left + up) // 2)) & 0xFF
        elif filter_type == 4:
            row[idx] = (row[idx] + paeth_predictor(left, up, up_left)) & 0xFF


def paeth_predictor(left: int, up: int, up_left: int) -> int:
    estimate = left + up - up_left
    left_distance = abs(estimate - left)
    up_distance = abs(estimate - up)
    up_left_distance = abs(estimate - up_left)
    if left_distance <= up_distance and left_distance <= up_left_distance:
        return left
    if up_distance <= up_left_distance:
        return up
    return up_left
