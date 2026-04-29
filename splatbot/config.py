from __future__ import annotations

import shlex
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
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


DEFAULT_RUNPOD_GPU_TYPE_ID = ",".join(
    (
        "NVIDIA L40S",
        "NVIDIA L40",
        "NVIDIA RTX 6000 Ada Generation",
        "NVIDIA RTX A6000",
        "NVIDIA A40",
    )
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SPLATBOT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    telegram_token: SecretStr = SecretStr("")
    allowed_telegram_ids: set[int] = Field(default_factory=set)
    allow_all_telegram_users: bool = False
    telegram_mode: TelegramMode = TelegramMode.POLLING

    data_dir: Path = Path("/var/lib/splatbot")
    database_path: Path = Path("/var/lib/splatbot/splatbot.sqlite3")

    s3_endpoint_url: str = ""
    s3_region: str = "auto"
    s3_bucket: str = ""
    s3_access_key_id: SecretStr = SecretStr("")
    s3_secret_access_key: SecretStr = SecretStr("")
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
    interrupted_job_grace_seconds: int = 30 * 60
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
    segmentation_backend: str = ""
    best_segmentation_backends: str = "sam2,rembg"
    best_segmentation_required_backends: str = "sam2"
    experimental_sam3_enabled: bool = False
    segmentation_min_output_ratio: float = 0.8
    segmentation_min_output_files: int = 1
    object_mask_backend: str = "rembg"
    object_mask_command: str = ""
    object_mask_prompt: str = "main object"
    sam3_mask_command: str = "splatbot-segment --backend sam3 --input {images_dir} --output {object_dir} --prompt {prompt}"
    sam2_mask_command: str = "splatbot-segment --backend sam2 --input {images_dir} --output {object_dir}"
    sam2_checkpoint: str = "/opt/splatbot/models/sam2.1_hiera_large.pt"
    sam2_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml"
    matting_command: str = ""
    object_mask_refine_enabled: bool = True
    object_mask_qa_enabled: bool = True
    object_mask_min_area_ratio: float = 0.01
    object_mask_max_area_ratio: float = 0.75
    object_mask_max_area_jump: float = 0.45
    object_mask_max_edge_touch_ratio: float = 0.35
    object_mask_min_keep_frames: int = 50
    object_mask_min_keep_ratio: float = 0.55
    object_mask_training_alpha_threshold: int = 16
    pose_backends: str = "colmap"
    best_pose_backends: str = "colmap-global,vggt-colmap,mast3r-sfm"
    best_pose_required_backends: str = ""
    pose_backend_command: str = "splatbot-pose --backend {backend} --input {images_dir} --output {processed_dir} --matching-method {matching_method}"
    da3_model: str = "depth-anything/DA3-LARGE-1.1"
    da3_model_cache_dir: str = "/opt/splatbot/models/da3"
    da3_allow_model_download: bool = True
    da3_model_download_attempts: int = 5
    da3_use_ray_pose: bool = True
    da3_ref_view_strategy: str = "middle"
    da3_pose_command: str = "splatbot-da3 --images {images_dir} --processed {processed_dir}"
    torch_home: str = "/opt/splatbot/models/torch"
    torchvision_allow_weight_download: bool = False
    vggt_pose_command: str = (
        "splatbot-vggt --images {images_dir} --processed {processed_dir} "
        "--matching-method {matching_method}"
    )
    pose_python: str = ""
    vggt_repo: str = "/opt/vggt"
    vggt_demo_colmap: str = ""
    vggt_run_command: str = ""
    vggt_args: str = "--use_ba --max_query_pts 1024 --query_frame_num 3"
    vggt_max_images: int = 64
    vggt_weights: str = "/opt/splatbot/models/torch/hub/checkpoints/model.pt"
    vggsfm_tracker_weights: str = "/opt/splatbot/models/torch/hub/checkpoints/vggsfm_v2_tracker.pt"
    dinov2_hub_repo: str = "/opt/splatbot/models/torch/hub/facebookresearch_dinov2_main"
    dinov2_vitb14_reg_weights: str = "/opt/splatbot/models/torch/hub/checkpoints/dinov2_vitb14_reg4_pretrain.pth"
    aliked_n16_weights: str = "/opt/splatbot/models/torch/hub/checkpoints/aliked-n16.pth"
    superpoint_weights: str = "/opt/splatbot/models/torch/hub/checkpoints/superpoint_v1.pth"
    vggt_allow_weight_download: bool = False
    mast3r_pose_command: str = (
        "splatbot-mast3r --images {images_dir} --processed {processed_dir} "
        "--matching-method {matching_method}"
    )
    mast3r_repo: str = "/opt/mast3r"
    mast3r_run_command: str = ""
    mast3r_weights: str = "/opt/splatbot/models/mast3r/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    mast3r_weights_url: str = (
        "https://download.europe.naverlabs.com/ComputerVision/MASt3R/"
        "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    )
    mast3r_allow_weight_download: bool = True
    mast3r_args: str = ""
    mast3r_max_images: int = 80
    mast3r_pair_window: int = 5
    mast3r_pair_cyclic: bool = True
    mast3r_device: str = "cuda"
    mast3r_shared_camera: bool = True
    mast3r_use_glomap: bool = True
    glomap_bin: str = "splatbot-glomap"
    glomap_mapper_args: str = (
        "--log_to_stderr=1 --ba_iteration_num=1 "
        "--GlobalPositioning.max_num_iterations=60 "
        "--BundleAdjustment.max_num_iterations=80"
    )
    colmap_global_calibrate: bool = True
    colmap_use_gpu: bool = False
    command_timeout_seconds: int = 6 * 60 * 60
    command_tail_bytes: int = 64 * 1024
    train_max_iterations: int = 10000
    train_steps_per_save: int = 10000
    train_method: str = "splatfacto"
    train_extra_args: str = ""
    render_preview: bool = False
    adaptive_frame_selection: bool = True
    frame_selection_strategy: str = "quality"
    best_frame_selection_strategy: str = "quality-diversity"
    frame_quality_reject_threshold: float = 35.0
    blur_reject_threshold: float = 20.0
    low_contrast_reject_threshold: float = 6.0
    overexposed_reject_threshold: float = 0.55
    underexposed_reject_threshold: float = 0.55
    duplicate_frame_threshold: float = 3.0
    min_selected_video_frames: int = 60
    min_colmap_registered_ratio: float = 0.35
    min_colmap_sparse_points: int = 1000
    object_colmap_original_pose_fallback: bool = True
    colmap_retry_frame_counts: str = "120,80,60"
    colmap_retry_matching_methods: str = "sequential,exhaustive"
    silhouette_cleanup_enabled: bool = True
    silhouette_cleanup_min_views: int = 4
    silhouette_cleanup_max_views: int = 96
    silhouette_cleanup_alpha_threshold: int = 16
    silhouette_cleanup_padding_px: int = 0
    silhouette_cleanup_outside_ratio: float = 0.45
    silhouette_cleanup_max_inside_views: int = 8
    silhouette_cleanup_max_inside_ratio: float = 0.35
    silhouette_cleanup_max_remove_fraction: float = 0.6
    gaussian_cleanup_enabled: bool = True
    gaussian_cleanup_min_opacity: float = -7.0
    gaussian_cleanup_max_scale_ratio: float = 10.0
    gaussian_cleanup_max_anisotropy: float = 25.0
    gaussian_cleanup_max_remove_fraction: float = 0.35
    spatial_cleanup_enabled: bool = True
    spatial_outlier_radius_fraction: float = 0.02
    spatial_outlier_min_neighbors: int = 3
    spatial_component_voxel_fraction: float = 0.025
    spatial_component_min_vertices: int = 500
    spatial_component_min_fraction: float = 0.005
    spatial_cleanup_max_remove_fraction: float = 0.35
    mask_support_cleanup_enabled: bool = True
    mask_support_cleanup_min_views: int = 4
    mask_support_cleanup_min_inside_views: int = 1
    mask_support_cleanup_min_inside_ratio: float = 0.02
    mask_support_cleanup_max_remove_fraction: float = 0.35
    depth_consistency_cleanup_enabled: bool = False
    depth_consistency_cleanup_min_views: int = 3
    depth_consistency_cleanup_max_depth_ratio: float = 1.6
    depth_consistency_cleanup_min_inconsistent_ratio: float = 0.75
    depth_consistency_cleanup_alignment_sample_limit: int = 20_000
    depth_consistency_cleanup_min_scale_samples: int = 20
    depth_consistency_cleanup_max_remove_fraction: float = 0.35
    postprocess_validation_enabled: bool = True
    postprocess_validation_max_outside_fraction: float = 0.2
    postprocess_validation_min_inside_views: int = 1
    postprocess_validation_min_inside_ratio: float = 0.02
    postprocess_validation_max_low_support_fraction: float = 0.25
    postprocess_validation_min_checked_points: int = 1
    postprocess_validation_max_unobserved_fraction: float = 0.35
    postprocess_validation_sample_limit: int = 200_000
    render_validation_command: str = ""
    quality_report_enabled: bool = True
    min_splat_vertices: int = 10_000
    max_flattened_axis_ratio: float = 0.015
    min_export_gaussian_retention: float = 0.02
    depth_backends: str = ""
    best_depth_backends: str = "da3,depth-anything-v2-large"
    best_depth_required_backends: str = "da3"
    depth_backend_command: str = "splatbot-depth --backend {backend} --processed {processed_dir} --images {images_dir}"
    da3_depth_command: str = "splatbot-da3 --images {images_dir} --processed {processed_dir} --depth-only"
    depth_anything_v2_command: str = ""
    train_backends: str = ""
    best_train_backends: str = "splatfacto-big,3dgs-mcmc"
    best_train_required_backends: str = ""
    experimental_dn_splatter_enabled: bool = False
    mcmc_train_command: str = (
        "ns-train splatfacto-mcmc --data {processed_dir} --output-dir {ns_dir} "
        "--max-num-iterations {max_iterations} --steps-per-save {steps_per_save} "
        "--viewer.quit-on-train-completion True {extra_args}"
    )
    mip_splatting_train_command: str = ""
    twodgs_train_command: str = ""
    train_backend_command: str = "splatbot-train --backend {backend} --data {processed_dir} --output {ns_dir} --max-iterations {max_iterations} --steps-per-save {steps_per_save} {extra_args}"
    mesh_export_enabled: bool = False
    mesh_backend: str = "o3dtsdf"
    mesh_export_command: str = "splatbot-mesh --backend {backend} --ns-dir {ns_dir} --output {mesh_path}"
    mesh_export_filename: str = "mesh.glb"
    mesh_export_required: bool = False

    fast_max_video_frames: int = 90
    fast_train_max_iterations: int = 7000
    fast_train_steps_per_save: int = 7000
    fast_train_method: str = "splatfacto"
    fast_train_extra_args: str = ""

    best_max_video_frames: int = 180
    best_train_max_iterations: int = 30000
    best_train_steps_per_save: int = 30000
    best_train_method: str = "splatfacto-big"
    best_train_extra_args: str = (
        "--pipeline.model.cull-alpha-thresh=0.005 "
        "--pipeline.model.use-scale-regularization=True"
    )

    worker_backend: WorkerBackend = WorkerBackend.LOCAL

    runpod_api_key: SecretStr = SecretStr("")
    runpod_gpu_type_id: str = DEFAULT_RUNPOD_GPU_TYPE_ID
    runpod_cloud_type: str = "ALL"
    runpod_image_name: str = "ghcr.io/navalgazing/splatbot-runpod:cuda-colmap-sota"
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
    runpod_ssh_ready_timeout_seconds: int = 3600
    runpod_no_endpoint_timeout_seconds: int = 1500
    runpod_launch_attempts: int = 5
    runpod_worker_timeout_seconds: int = 6 * 60 * 60
    runpod_vps_host: str = ""
    runpod_vps_user: str = "root"
    runpod_vps_ssh_key: Path | None = None
    runpod_vps_known_hosts: str = ""
    runpod_pod_known_hosts_path: Path = Path("/tmp/splatbot-runpod-known-hosts")
    runpod_venv: str = ""
    runpod_runtime_cache_version: str = "splatbot-runtime-2026-04-26-v1"
    runpod_runtime_cache_marker: str = "/workspace/.splatbot-runtime-cache-version"
    runpod_bootstrap_command: str = ""
    runpod_setup_command: str = ""
    matrix_run_metadata: str = ""

    @field_validator("allowed_telegram_ids", mode="before")
    @classmethod
    def parse_allowed_ids(cls, value: object) -> set[int]:
        def parse_item(item: object) -> int:
            try:
                return int(str(item).strip())
            except (TypeError, ValueError) as exc:
                raise ValueError("SPLATBOT_ALLOWED_TELEGRAM_IDS must contain only integer chat IDs") from exc

        if value is None or value == "":
            return set()
        if isinstance(value, int):
            return {value}
        if isinstance(value, set):
            return {parse_item(item) for item in value}
        if isinstance(value, (list, tuple)):
            return {parse_item(item) for item in value}
        if isinstance(value, str):
            return {parse_item(item) for item in value.split(",") if item.strip()}
        raise TypeError("allowed_telegram_ids must be a comma-separated string or collection")

    def require_telegram(self) -> None:
        if not self.telegram_token_value:
            raise ValueError("SPLATBOT_TELEGRAM_TOKEN is required")

    @staticmethod
    def _secret_value(value: SecretStr | str) -> str:
        return value.get_secret_value() if isinstance(value, SecretStr) else value

    @property
    def telegram_token_value(self) -> str:
        return self._secret_value(self.telegram_token)

    @property
    def s3_access_key_id_value(self) -> str:
        return self._secret_value(self.s3_access_key_id)

    @property
    def s3_secret_access_key_value(self) -> str:
        return self._secret_value(self.s3_secret_access_key)

    @property
    def runpod_api_key_value(self) -> str:
        return self._secret_value(self.runpod_api_key)

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
