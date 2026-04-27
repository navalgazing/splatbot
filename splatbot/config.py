from __future__ import annotations

import shlex
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ScanMode(StrEnum):
    SCENE = "scene"
    OBJECT = "object"


class ScanPreset(StrEnum):
    FAST = "fast"
    BALANCED = "balanced"
    BEST = "best"


@dataclass(frozen=True)
class ScanPresetConfig:
    preset: ScanPreset
    max_video_frames: int
    train_method: str
    train_max_iterations: int
    train_steps_per_save: int
    train_extra_args: tuple[str, ...]
    adaptive_frame_selection: bool


class TelegramMode(StrEnum):
    POLLING = "polling"
    WEBHOOK = "webhook"


class WorkerBackend(StrEnum):
    LOCAL = "local"
    RUNPOD = "runpod"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SPLATBOT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    telegram_token: str = ""
    allowed_telegram_ids: set[int] = Field(default_factory=set)
    allow_all_telegram_users: bool = False
    telegram_mode: TelegramMode = TelegramMode.POLLING

    data_dir: Path = Path("/var/lib/splatbot")
    database_path: Path = Path("/var/lib/splatbot/splatbot.sqlite3")

    s3_endpoint_url: str = ""
    s3_region: str = "auto"
    s3_bucket: str = ""
    s3_access_key_id: str = ""
    s3_secret_access_key: str = ""
    signed_url_ttl_seconds: int = 7 * 24 * 60 * 60

    public_base_url: str = ""
    public_results_dir: Path = Path("/var/www/splatbot/results")

    min_images: int = 100
    max_images: int = 300
    max_video_frames: int = 140
    max_video_seconds: int = 60
    max_video_sample_fps: float = 12.0
    max_video_candidate_fps: float = 30.0
    max_upload_bytes: int = 1024 * 1024 * 1024
    interrupted_job_grace_seconds: int = 10 * 60
    default_scan_mode: ScanMode = ScanMode.SCENE
    default_scan_preset: ScanPreset = ScanPreset.BALANCED
    job_retention_days: int = 14

    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
    colmap_bin: str = "colmap"
    ns_process_data_bin: str = "ns-process-data"
    ns_train_bin: str = "ns-train"
    ns_export_bin: str = "ns-export"
    ns_render_bin: str = "ns-render"
    rembg_bin: str = "rembg"
    rembg_require_gpu: bool = False
    colmap_use_gpu: bool = False
    command_timeout_seconds: int = 6 * 60 * 60
    command_tail_bytes: int = 64 * 1024
    train_max_iterations: int = 10000
    train_steps_per_save: int = 10000
    train_method: str = "splatfacto"
    train_extra_args: str = ""
    render_preview: bool = False
    adaptive_frame_selection: bool = True
    frame_quality_reject_threshold: float = 35.0
    blur_reject_threshold: float = 20.0
    low_contrast_reject_threshold: float = 6.0
    overexposed_reject_threshold: float = 0.55
    underexposed_reject_threshold: float = 0.55
    duplicate_frame_threshold: float = 3.0
    min_selected_video_frames: int = 60
    min_colmap_registered_ratio: float = 0.35
    object_colmap_original_pose_fallback: bool = True
    colmap_retry_frame_counts: str = "120,80,60"
    colmap_retry_matching_methods: str = "sequential,exhaustive"
    min_splat_vertices: int = 10000
    max_flattened_axis_ratio: float = 0.015

    fast_max_video_frames: int = 90
    fast_train_max_iterations: int = 7000
    fast_train_steps_per_save: int = 7000
    fast_train_method: str = "splatfacto"
    fast_train_extra_args: str = ""

    best_max_video_frames: int = 180
    best_train_max_iterations: int = 14000
    best_train_steps_per_save: int = 14000
    best_train_method: str = "splatfacto-big"
    best_train_extra_args: str = (
        "--pipeline.model.cull_alpha_thresh=0.005 "
        "--pipeline.model.use_scale_regularization=True"
    )

    worker_backend: WorkerBackend = WorkerBackend.LOCAL

    runpod_api_key: str = ""
    runpod_gpu_type_id: str = "NVIDIA GeForce RTX 4090"
    runpod_cloud_type: str = "ALL"
    runpod_image_name: str = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
    runpod_container_disk_gb: int = 80
    runpod_volume_gb: int = 80
    runpod_min_vcpu_count: int = 8
    runpod_min_memory_gb: int = 30
    runpod_ports: str = "22/tcp"
    runpod_volume_mount_path: str = "/workspace"
    runpod_network_volume_id: str = ""
    runpod_data_center_ids: str = ""
    runpod_ssh_user: str = "root"
    runpod_pod_ssh_key: Path | None = None
    runpod_ssh_ready_timeout_seconds: int = 900
    runpod_no_endpoint_timeout_seconds: int = 240
    runpod_launch_attempts: int = 3
    runpod_worker_timeout_seconds: int = 6 * 60 * 60
    runpod_vps_host: str = ""
    runpod_vps_user: str = "root"
    runpod_vps_ssh_key: Path | None = None
    runpod_venv: str = ""
    runpod_runtime_cache_version: str = "splatbot-runtime-2026-04-26-v1"
    runpod_runtime_cache_marker: str = "/workspace/.splatbot-runtime-cache-version"
    runpod_bootstrap_command: str = ""
    runpod_setup_command: str = ""

    @field_validator("allowed_telegram_ids", mode="before")
    @classmethod
    def parse_allowed_ids(cls, value: object) -> set[int]:
        if value is None or value == "":
            return set()
        if isinstance(value, int):
            return {value}
        if isinstance(value, set):
            return {int(item) for item in value}
        if isinstance(value, (list, tuple)):
            return {int(item) for item in value}
        if isinstance(value, str):
            return {int(item.strip()) for item in value.split(",") if item.strip()}
        raise TypeError("allowed_telegram_ids must be a comma-separated string or collection")

    def require_telegram(self) -> None:
        if not self.telegram_token:
            raise ValueError("SPLATBOT_TELEGRAM_TOKEN is required")

    def preset_config(self, preset: ScanPreset | str | None = None) -> ScanPresetConfig:
        selected = ScanPreset(preset or self.default_scan_preset)
        if selected == ScanPreset.FAST:
            return ScanPresetConfig(
                preset=selected,
                max_video_frames=self.fast_max_video_frames,
                train_method=self.fast_train_method,
                train_max_iterations=self.fast_train_max_iterations,
                train_steps_per_save=self.fast_train_steps_per_save,
                train_extra_args=tuple(shlex.split(self.fast_train_extra_args)),
                adaptive_frame_selection=self.adaptive_frame_selection,
            )
        if selected == ScanPreset.BEST:
            return ScanPresetConfig(
                preset=selected,
                max_video_frames=self.best_max_video_frames,
                train_method=self.best_train_method,
                train_max_iterations=self.best_train_max_iterations,
                train_steps_per_save=self.best_train_steps_per_save,
                train_extra_args=tuple(shlex.split(self.best_train_extra_args)),
                adaptive_frame_selection=self.adaptive_frame_selection,
            )
        return ScanPresetConfig(
            preset=selected,
            max_video_frames=self.max_video_frames,
            train_method=self.train_method,
            train_max_iterations=self.train_max_iterations,
            train_steps_per_save=self.train_steps_per_save,
            train_extra_args=tuple(shlex.split(self.train_extra_args)),
            adaptive_frame_selection=self.adaptive_frame_selection,
        )

    def job_dir(self, job_id: str) -> Path:
        return self.data_dir / "jobs" / job_id

    def public_job_url(self, job_id: str) -> str:
        return f"{self.public_base_url.rstrip('/')}/results/{job_id}/" if self.public_base_url else ""
