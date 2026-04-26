from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ScanMode(StrEnum):
    SCENE = "scene"
    OBJECT = "object"


class TelegramMode(StrEnum):
    POLLING = "polling"
    WEBHOOK = "webhook"


class WorkerBackend(StrEnum):
    LOCAL = "local"
    SSH = "ssh"
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
    max_video_sample_fps: float = 10.0
    max_upload_bytes: int = 1024 * 1024 * 1024
    interrupted_job_grace_seconds: int = 10 * 60
    default_scan_mode: ScanMode = ScanMode.SCENE
    job_retention_days: int = 14

    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
    colmap_bin: str = "colmap"
    ns_process_data_bin: str = "ns-process-data"
    ns_train_bin: str = "ns-train"
    ns_export_bin: str = "ns-export"
    ns_render_bin: str = "ns-render"
    rembg_bin: str = "rembg"
    colmap_use_gpu: bool = False
    command_timeout_seconds: int = 6 * 60 * 60
    command_tail_bytes: int = 64 * 1024
    train_max_iterations: int = 10000
    train_steps_per_save: int = 10000
    render_preview: bool = False

    worker_backend: WorkerBackend = WorkerBackend.LOCAL
    gpu_ssh_host: str = ""
    gpu_ssh_key: Path | None = None
    gpu_workdir: Path = Path("/srv/splatbot")

    runpod_api_key: str = ""
    runpod_gpu_type_id: str = "NVIDIA GeForce RTX 4090"
    runpod_cloud_type: str = "ALL"
    runpod_image_name: str = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
    runpod_container_disk_gb: int = 80
    runpod_volume_gb: int = 80
    runpod_min_vcpu_count: int = 8
    runpod_min_memory_gb: int = 30
    runpod_ports: str = "22/tcp"
    runpod_volume_mount_path: str = "/workspace"
    runpod_ssh_user: str = "root"
    runpod_pod_ssh_key: Path | None = None
    runpod_ssh_ready_timeout_seconds: int = 900
    runpod_worker_timeout_seconds: int = 6 * 60 * 60
    runpod_vps_host: str = ""
    runpod_vps_user: str = "root"
    runpod_vps_ssh_key: Path | None = None
    runpod_venv: str = ""
    runpod_bootstrap_command: str = "apt-get update && apt-get install -y openssh-client rsync curl ffmpeg colmap python3 python3-venv python3-pip build-essential"
    runpod_setup_command: str = "/workspace/venv/bin/pip install nerfstudio 'rembg[cpu,cli]'"

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

    def job_dir(self, job_id: str) -> Path:
        return self.data_dir / "jobs" / job_id

    def public_job_url(self, job_id: str) -> str:
        return f"{self.public_base_url.rstrip('/')}/results/{job_id}/" if self.public_base_url else ""
