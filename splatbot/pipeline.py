from __future__ import annotations

import json
import math
import shutil
import struct
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from statistics import median
from typing import Awaitable, Callable

from .commands import CommandRunner
from .config import ScanMode, ScanPreset, ScanPresetConfig, Settings
from .models import JobStatus, MediaItem, MediaKind


StatusCallback = Callable[[str, JobStatus], Awaitable[None]]


@dataclass(frozen=True)
class PipelineOutputs:
    cleaned_ply: Path
    preview_mp4: Path | None
    metrics_path: Path | None = None


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
    reject_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class FrameSelectionResult:
    selected_count: int
    metrics: dict


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
        }
        write_json(
            settings_path,
            {
                "mode": mode.value,
                "preset": preset_config.preset.value,
                "max_video_frames": preset_config.max_video_frames,
                "max_video_candidate_fps": self.settings.max_video_candidate_fps,
                "adaptive_frame_selection": preset_config.adaptive_frame_selection,
                "frame_quality_reject_threshold": self.settings.frame_quality_reject_threshold,
                "blur_reject_threshold": self.settings.blur_reject_threshold,
                "low_contrast_reject_threshold": self.settings.low_contrast_reject_threshold,
                "overexposed_reject_threshold": self.settings.overexposed_reject_threshold,
                "underexposed_reject_threshold": self.settings.underexposed_reject_threshold,
                "duplicate_frame_threshold": self.settings.duplicate_frame_threshold,
                "colmap_retry_frame_counts": parse_int_list(self.settings.colmap_retry_frame_counts),
                "colmap_retry_matching_methods": parse_csv_list(self.settings.colmap_retry_matching_methods),
                "train_method": preset_config.train_method,
                "train_max_iterations": preset_config.train_max_iterations,
                "train_steps_per_save": preset_config.train_steps_per_save,
                "train_extra_args": list(preset_config.train_extra_args),
                "colmap_use_gpu": self.settings.colmap_use_gpu,
                "colmap_bin": self.settings.colmap_bin,
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
            await self.remove_backgrounds(images_dir, object_dir)
            record_stage(metrics, "rembg", stage_start)
            input_images_dir = object_dir
            metrics["frames"]["object"] = count_files(object_dir)
            write_json(metrics_path, metrics)
            log_directory_summary("object images", object_dir)

        if on_status:
            await on_status(job_id, JobStatus.COLMAP)
        stage_start = time.perf_counter()
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
        write_json(metrics_path, metrics)
        log_directory_summary("processed data", processed_dir)
        if on_status:
            await on_status(job_id, JobStatus.TRAINING)
        stage_start = time.perf_counter()
        await self.train_splatfacto(processed_dir, ns_dir, preset_config)
        record_stage(metrics, "training", stage_start)
        write_json(metrics_path, metrics)
        log_directory_summary("nerfstudio outputs", ns_dir)
        if on_status:
            await on_status(job_id, JobStatus.EXPORTING)
        stage_start = time.perf_counter()
        raw_ply = await self.export_ply(ns_dir, export_dir)
        cleaned_ply = export_dir / "cleaned_splat.ply"
        clean_ply(raw_ply, cleaned_ply)
        record_stage(metrics, "exporting", stage_start)
        log_ply_summary("raw splat", raw_ply)
        log_ply_summary("cleaned splat", cleaned_ply)
        metrics["ply"] = {
            "raw": inspect_ply(raw_ply),
            "cleaned": inspect_ply(cleaned_ply),
        }
        write_json(metrics_path, metrics)
        validate_ply_quality(metrics, self.settings)
        preview_mp4 = None
        if self.settings.render_preview:
            if on_status:
                await on_status(job_id, JobStatus.RENDERING)
            stage_start = time.perf_counter()
            preview_mp4 = await self.render_turntable(ns_dir, render_dir)
            record_stage(metrics, "rendering", stage_start)
            write_json(metrics_path, metrics)
        return PipelineOutputs(cleaned_ply=cleaned_ply, preview_mp4=preview_mp4, metrics_path=metrics_path)

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
            )
            metrics["video"] = {
                **metrics.get("video", {}),
                "adaptive_frame_selection": True,
                **extraction_metrics,
                **selection.metrics,
            }
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

    async def remove_backgrounds(self, images_dir: Path, object_dir: Path) -> None:
        await self.runner.run(
            [self.settings.rembg_bin, "p", str(images_dir), str(object_dir)]
        )

    async def process_data(
        self,
        images_dir: Path,
        processed_dir: Path,
        matching_method: str | None = None,
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
        await self.process_data(input_images_dir, processed_dir, matching_method=matching_method)
        selected_frames = metrics.get("frames", {}).get("selected") or preset.max_video_frames
        record_colmap_attempt(
            metrics,
            source="object" if mode == ScanMode.OBJECT and input_images_dir == object_images_dir else "original",
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
            if await self.retry_colmap_with_subsets(
                source_images_dir=input_images_dir,
                processed_dir=processed_dir,
                matching_method=matching_method,
                metrics=metrics,
                metrics_path=metrics_path,
                target_frames=preset.max_video_frames,
                selected_frames=selected_frames,
                source_label="object" if mode == ScanMode.OBJECT and input_images_dir == object_images_dir else "original",
                final_object_images_dir=None,
            ):
                return
            if (
                mode != ScanMode.OBJECT
                or object_images_dir is None
                or input_images_dir != object_images_dir
                or not self.settings.object_colmap_original_pose_fallback
            ):
                raise

        metrics["colmap_masked"] = metrics["colmap"]
        metrics["colmap_fallback"] = {
            "reason": masked_error_message,
            "pose_images": "original",
            "training_images": "object",
        }
        write_json(metrics_path, metrics)

        if processed_dir.exists():
            shutil.rmtree(processed_dir)
        processed_dir.mkdir(parents=True, exist_ok=True)
        await self.process_data(original_images_dir, processed_dir, matching_method=matching_method)
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
        metrics["colmap_fallback"] = {
            **metrics["colmap_fallback"],
            "applied": True,
            "training_image_paths_rewritten": replaced,
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
                await self.process_data(subset_dir, processed_dir, matching_method=effective_matching)
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
                if final_object_images_dir is not None:
                    replaced = replace_processed_images_with_object_images(
                        processed_dir,
                        final_object_images_dir,
                        allowed_stems=processed_frame_stems(processed_dir),
                    )
                    metrics["colmap_fallback"] = {
                        **metrics.get("colmap_fallback", {}),
                        "applied": True,
                        "training_image_paths_rewritten": replaced,
                    }
                write_json(metrics_path, metrics)
                return True
        return False

    async def train_splatfacto(self, processed_dir: Path, ns_dir: Path, preset: ScanPresetConfig) -> None:
        argv = [
            self.settings.ns_train_bin,
            preset.train_method,
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

    async def export_ply(self, ns_dir: Path, export_dir: Path) -> Path:
        raw_ply = export_dir / "raw_splat.ply"
        await self.runner.run(
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
        return raw_ply

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


def record_stage(metrics: dict, name: str, started: float) -> None:
    metrics.setdefault("stages", {})[name] = {
        "duration_seconds": round(time.perf_counter() - started, 3)
    }


def count_files(path: Path) -> int:
    return sum(1 for item in path.iterdir() if item.is_file())


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

    selected = quality_aware_sample(selection_pool, min(target_count, len(selection_pool)))
    for idx, src in enumerate(selected, start=1):
        shutil.copy2(src.path, images_dir / f"frame_{idx:05d}.jpg")

    return FrameSelectionResult(
        selected_count=len(selected),
        metrics={
            "candidate_frames": len(candidates),
            "selected_frames": len(selected),
            "accepted_frame_candidates": len(accepted),
            "quality_selection_fallback": fallback_used,
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


def summarize_profile_values(profiles: list[FrameQuality], field: str) -> dict[str, float | None]:
    values = [getattr(profile, field) for profile in profiles]
    numbers = [float(value) for value in values if value is not None]
    if not numbers:
        return {"min": None, "median": None, "max": None}
    return {
        "min": round(min(numbers), 3),
        "median": round(float(median(numbers)), 3),
        "max": round(max(numbers), 3),
    }


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
        if max(previous_size, current_size) == 0:
            return 0.0
        return abs(previous_size - current_size) * 100.0 / max(previous_size, current_size)
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
                "active_sparse_0": model_dir.name == "0",
            }
        )
    active = next((model for model in models if model["active_sparse_0"]), None)
    return {
        "transforms_frames": transforms_frames,
        "models": models,
        "active_registered_images": active["registered_images"] if active else None,
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


def validate_colmap_quality(metrics: dict, target_frames: int, settings: Settings) -> None:
    selected_frames = metrics.get("frames", {}).get("selected") or target_frames
    if selected_frames < 10:
        return
    colmap = metrics.get("colmap", {})
    registered = colmap.get("active_registered_images")
    if registered is None:
        registered = colmap.get("transforms_frames")
    if registered is None:
        return
    minimum = max(2, int(selected_frames * settings.min_colmap_registered_ratio))
    if registered < minimum:
        best = colmap.get("best_registered_images")
        hint = f"; best sparse model has {best} registered image(s)" if best and best > registered else ""
        raise ValueError(
            f"COLMAP registered only {registered}/{selected_frames} selected frame(s) in the active sparse model"
            f"{hint}. This run would likely produce a distorted splat."
        )


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


def quality_blind_even_sample(items: list[Path], count: int) -> list[Path]:
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


def inspect_ply(path: Path) -> dict:
    summary: dict = {"size_bytes": path.stat().st_size if path.exists() else None}
    if not path.exists():
        return summary
    try:
        with path.open("rb") as handle:
            header_lines: list[str] = []
            header_bytes = 0
            while True:
                raw = handle.readline()
                if not raw:
                    break
                header_bytes += len(raw)
                line = raw.decode("ascii", errors="ignore").strip()
                header_lines.append(line)
                if line == "end_header":
                    break
            fmt = next((line for line in header_lines if line.startswith("format ")), None)
            vertex_line = next((line for line in header_lines if line.startswith("element vertex ")), None)
            vertices = int(vertex_line.rsplit(" ", 1)[-1]) if vertex_line else None
            summary.update({"format": fmt, "vertices": vertices})
            if vertices:
                bounds = read_ply_xyz_bounds(handle, header_lines, header_bytes, vertices, path)
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


def update_bounds(mins: list[float], maxs: list[float], xyz: list[float]) -> None:
    if not all(math.isfinite(value) for value in xyz):
        return
    for idx, value in enumerate(xyz):
        mins[idx] = min(mins[idx], value)
        maxs[idx] = max(maxs[idx], value)


def validate_ply_quality(metrics: dict, settings: Settings) -> None:
    cleaned = metrics.get("ply", {}).get("cleaned", {})
    vertices = cleaned.get("vertices")
    if vertices is not None and vertices < settings.min_splat_vertices:
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
        raise ValueError(
            f"Exported splat appears flattened (axis ratio {ratio:.4f}, vertices {vertices})."
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
            for raw_line in handle:
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


def clean_ply(src: Path, dest: Path) -> None:
    """Conservatively remove invalid ASCII vertex rows while preserving properties."""
    header_bytes = src.read_bytes()[:512]
    if b"format binary_" in header_bytes:
        shutil.copy2(src, dest)
        return
    raw = src.read_text(encoding="utf-8", errors="replace").splitlines()
    try:
        end_header_idx = raw.index("end_header")
    except ValueError as exc:
        raise ValueError(f"{src} is missing a PLY header") from exc

    header = raw[: end_header_idx + 1]
    body = raw[end_header_idx + 1 :]
    cleaned: list[str] = []
    for line in body:
        parts = line.split()
        if not parts:
            continue
        try:
            values = [float(part) for part in parts]
        except ValueError:
            continue
        if all(value == value and value not in (float("inf"), float("-inf")) for value in values):
            cleaned.append(line)

    updated_header: list[str] = []
    for line in header:
        if line.startswith("element vertex "):
            updated_header.append(f"element vertex {len(cleaned)}")
        else:
            updated_header.append(line)
    dest.write_text("\n".join(updated_header + cleaned) + "\n", encoding="utf-8")
