from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import Settings
from .models import ScanJob


LOGGER = logging.getLogger(__name__)


class RunPodError(RuntimeError):
    pass


class RunPodApiError(RunPodError):
    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"RunPod REST API failed: {status_code} {body}")


class RunPodSshUnavailableError(RunPodError):
    pass


@dataclass(frozen=True)
class RunPodPod:
    id: str
    image_name: str


@dataclass(frozen=True)
class RunPodSshTarget:
    host: str
    port: int


class RunPodClient:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def request(self, method: str, path: str, payload: dict | None = None) -> dict | list:
        data = None if payload is None else json.dumps(payload).encode()
        last_error: Exception | None = None
        for attempt in range(1, 4):
            request = urllib.request.Request(
                f"https://rest.runpod.io/v1{path}",
                data=data,
                headers={
                    "accept": "application/json",
                    "authorization": f"Bearer {self.api_key}",
                    "content-type": "application/json",
                    "user-agent": "splatbot/0.1 (+https://github.com/navalgazing/splatbot)",
                },
                method=method,
            )
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    body = response.read().decode()
                    return json.loads(body) if body else {}
            except urllib.error.HTTPError as exc:
                body = exc.read().decode(errors="replace")
                error = RunPodApiError(exc.code, body)
                if exc.code not in {429, 500, 502, 503, 504} or attempt >= 3:
                    raise error from exc
                last_error = error
            except urllib.error.URLError as exc:
                if attempt >= 3:
                    raise RunPodError(f"RunPod REST API request failed: {exc}") from exc
                last_error = exc
            LOGGER.warning("RunPod REST API %s %s failed on attempt %s/3: %s", method, path, attempt, last_error)
            time.sleep(2**attempt)
        raise RunPodError(f"RunPod REST API request failed after retries: {last_error}")

    def create_ssh_pod(self, settings: Settings, job: ScanJob, public_key: str) -> RunPodPod:
        cloud_type = settings.runpod_cloud_type
        if cloud_type == "ALL":
            cloud_type = "SECURE"
        payload = {
            "name": "splatbot-" + job.id[:12],
            "imageName": settings.runpod_image_name,
            "gpuTypeIds": [settings.runpod_gpu_type_id],
            "gpuCount": 1,
            "cloudType": cloud_type,
            "computeType": "GPU",
            "containerDiskInGb": settings.runpod_container_disk_gb,
            "volumeInGb": settings.runpod_volume_gb,
            "volumeMountPath": settings.runpod_volume_mount_path,
            "vcpuCount": settings.runpod_min_vcpu_count,
            "minRAMPerGPU": settings.runpod_min_memory_gb,
            "allowedCudaVersions": ["12.8", "12.9", "13.0"],
            "supportPublicIp": True,
            "ports": [settings.runpod_ports.strip() or "22/tcp"],
            "env": {"PUBLIC_KEY": public_key},
        }
        if settings.runpod_network_volume_id:
            payload["networkVolumeId"] = settings.runpod_network_volume_id
            payload.pop("volumeInGb", None)
        data_center_ids = [item.strip() for item in settings.runpod_data_center_ids.split(",") if item.strip()]
        if data_center_ids:
            payload["dataCenterIds"] = data_center_ids
            payload["dataCenterPriority"] = "availability"
        pod = self.request(
            "POST",
            "/pods",
            payload,
        )
        assert isinstance(pod, dict)
        return RunPodPod(id=pod["id"], image_name=pod["imageName"])

    def get_pod(self, pod_id: str) -> dict:
        pod = self.request("GET", f"/pods/{pod_id}")
        assert isinstance(pod, dict)
        return pod

    def delete_pod(self, pod_id: str) -> None:
        try:
            self.request("DELETE", f"/pods/{pod_id}")
        except RunPodApiError as exc:
            if exc.status_code == 404:
                return
            raise


class RunPodLauncher:
    def __init__(self, settings: Settings, client: RunPodClient | None = None) -> None:
        self.settings = settings
        self.client = client or RunPodClient(settings.runpod_api_key_value)

    def launch(self, job: ScanJob, on_pod_id: Callable[[str | None], None] | None = None) -> RunPodPod:
        self._validate()
        public_key = self.settings.runpod_pod_ssh_key.with_suffix(".pub").read_text().strip()
        attempts = max(1, self.settings.runpod_launch_attempts)
        last_error: RunPodSshUnavailableError | None = None
        for attempt in range(1, attempts + 1):
            pod: RunPodPod | None = None
            try:
                pod = self.client.create_ssh_pod(self.settings, job, public_key)
                LOGGER.info("created RunPod pod %s for job %s attempt %s/%s", pod.id, job.id, attempt, attempts)
                if on_pod_id:
                    try:
                        on_pod_id(pod.id)
                    except Exception as exc:  # noqa: BLE001
                        raise RunPodError(f"failed to record RunPod pod id for job {job.id}") from exc
                target = self.wait_for_ssh(pod.id)
                self.run_worker(job, pod.id, target)
                return pod
            except RunPodSshUnavailableError as exc:
                last_error = exc
                LOGGER.warning("RunPod pod %s did not become SSH-ready: %s", pod.id, exc)
                if attempt >= attempts:
                    raise
            finally:
                if pod is not None:
                    try:
                        self.client.delete_pod(pod.id)
                    except Exception:  # noqa: BLE001
                        LOGGER.critical("failed to delete RunPod pod %s; manual cleanup required", pod.id, exc_info=True)
                    else:
                        if on_pod_id:
                            try:
                                on_pod_id(None)
                            except Exception:  # noqa: BLE001
                                LOGGER.exception("failed to clear RunPod pod id for job %s", job.id)
        assert last_error is not None
        raise last_error

    def _validate(self) -> None:
        if not self.settings.runpod_api_key_value:
            raise RunPodError("SPLATBOT_RUNPOD_API_KEY is required for RunPod backend")
        if not self.settings.runpod_vps_host:
            raise RunPodError("SPLATBOT_RUNPOD_VPS_HOST is required for RunPod backend")
        if self.settings.runpod_vps_ssh_key is None:
            raise RunPodError("SPLATBOT_RUNPOD_VPS_SSH_KEY is required for RunPod backend")
        if self.settings.runpod_pod_ssh_key is None:
            raise RunPodError("SPLATBOT_RUNPOD_POD_SSH_KEY is required for RunPod backend")
        self._validate_readable_file(self.settings.runpod_vps_ssh_key, "SPLATBOT_RUNPOD_VPS_SSH_KEY")
        self._validate_readable_file(self.settings.runpod_pod_ssh_key, "SPLATBOT_RUNPOD_POD_SSH_KEY")
        self._validate_readable_file(self.settings.runpod_pod_ssh_key.with_suffix(".pub"), "pod SSH public key")

    @staticmethod
    def _validate_readable_file(path: Path, label: str) -> None:
        if not path.exists():
            raise RunPodError(f"{label} missing: {path}")
        if not path.is_file():
            raise RunPodError(f"{label} is not a file: {path}")
        if not os.access(path, os.R_OK):
            raise RunPodError(f"{label} is not readable by this process: {path}")

    def wait_for_ssh(self, pod_id: str) -> RunPodSshTarget:
        started = time.monotonic()
        deadline = time.monotonic() + self.settings.runpod_ssh_ready_timeout_seconds
        last_seen = ""
        while time.monotonic() < deadline:
            pod = self.client.get_pod(pod_id)
            host = pod.get("publicIp") or ""
            port = (pod.get("portMappings") or {}).get("22")
            last_seen = f"host={host!r} port={port!r} status={pod.get('desiredStatus')!r}"
            if host and port:
                target = RunPodSshTarget(host=host, port=int(port))
                ready, ssh_error = self._ssh_ready(target)
                if ready:
                    return target
                if ssh_error:
                    last_seen = f"{last_seen} ssh_error={ssh_error!r}"
            elif time.monotonic() - started >= self.settings.runpod_no_endpoint_timeout_seconds:
                raise RunPodSshUnavailableError(
                    f"RunPod pod never received a public SSH endpoint before recycle timeout: {last_seen}"
                )
            time.sleep(10)
        raise RunPodSshUnavailableError(f"RunPod pod SSH was not ready before timeout: {last_seen}")

    def _ssh_ready(self, target: RunPodSshTarget) -> tuple[bool, str]:
        result = subprocess.run(
            [
                "ssh",
                *self._pod_ssh_args(target),
                "echo",
                "ready",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=20,
            check=False,
        )
        error = result.stderr.decode(errors="replace").strip()[-500:]
        return result.returncode == 0, error

    def _pod_ssh_args(self, target: RunPodSshTarget) -> list[str]:
        return [
            "-i",
            str(self.settings.runpod_pod_ssh_key),
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            f"UserKnownHostsFile={self.settings.runpod_pod_known_hosts_path}",
            "-o",
            "LogLevel=ERROR",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=6",
            "-p",
            str(target.port),
            f"{self.settings.runpod_ssh_user}@{target.host}",
        ]

    def _install_vps_ssh_key(self, target: RunPodSshTarget) -> None:
        mkdir_result = subprocess.run(
            ["ssh", *self._pod_ssh_args(target), "install", "-d", "-m", "700", "/root/.ssh"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
        if mkdir_result.returncode != 0:
            raise RunPodError(
                "failed to prepare pod SSH directory: "
                + mkdir_result.stderr.decode(errors="replace").strip()[-500:]
            )
        copy_result = subprocess.run(
            ["ssh", *self._pod_ssh_args(target), "sh", "-c", "umask 077 && cat > /root/.ssh/id_ed25519"],
            input=self.settings.runpod_vps_ssh_key.read_bytes(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            check=False,
        )
        if copy_result.returncode != 0:
            raise RunPodError(
                "failed to copy VPS SSH key to pod: "
                + copy_result.stderr.decode(errors="replace").strip()[-500:]
            )
        chmod_result = subprocess.run(
            ["ssh", *self._pod_ssh_args(target), "chmod", "600", "/root/.ssh/id_ed25519"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
        if chmod_result.returncode != 0:
            raise RunPodError(
                "failed to set pod VPS SSH key permissions: "
                + chmod_result.stderr.decode(errors="replace").strip()[-500:]
            )

    def run_worker(self, job: ScanJob, pod_id: str, target: RunPodSshTarget) -> None:
        log_path = self.settings.job_dir(job.id) / "runpod-worker.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._install_vps_ssh_key(target)
        command = render_remote_worker_command(self.settings, job, pod_id)
        ssh_command = ["ssh", *self._pod_ssh_args(target), "bash", "-s"]
        with log_path.open("ab") as log:
            log.write(f"\n--- RunPod worker {pod_id} on {target.host}:{target.port} ---\n".encode())
            result = subprocess.run(
                ssh_command,
                input=command.encode(),
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=self.settings.runpod_worker_timeout_seconds,
                check=False,
            )
        if result.returncode != 0:
            raise RunPodError(f"RunPod worker exited with {result.returncode}; see {log_path}")


def render_remote_worker_command(settings: Settings, job: ScanJob, pod_id: str) -> str:
    host = shlex.quote(settings.runpod_vps_host)
    user = shlex.quote(settings.runpod_vps_user)
    bootstrap_command = shlex.quote(settings.runpod_bootstrap_command.strip())
    setup_command = shlex.quote(settings.runpod_setup_command.strip())
    runtime_cache_version = shlex.quote(settings.runpod_runtime_cache_version.strip())
    runtime_cache_marker = shlex.quote(settings.runpod_runtime_cache_marker.strip())
    vps_known_hosts = shlex.quote(settings.runpod_vps_known_hosts.strip())
    venv_export = (
        f"export SPLATBOT_RUNPOD_VENV={shlex.quote(settings.runpod_venv.strip())}"
        if settings.runpod_venv.strip()
        else ""
    )
    pipeline_exports = "\n".join(
        f"export {name}={shlex.quote(str(value))}"
        for name, value in {
            "SPLATBOT_MAX_IMAGES": settings.max_images,
            "SPLATBOT_DEFAULT_SCAN_PRESET": job.preset.value,
            "SPLATBOT_SCAN_PRESET": job.preset.value,
            "SPLATBOT_MAX_VIDEO_FRAMES": settings.max_video_frames,
            "SPLATBOT_MAX_VIDEO_SECONDS": settings.max_video_seconds,
            "SPLATBOT_MAX_VIDEO_SAMPLE_FPS": settings.max_video_sample_fps,
            "SPLATBOT_MAX_VIDEO_CANDIDATE_FPS": settings.max_video_candidate_fps,
            "SPLATBOT_ADAPTIVE_FRAME_SELECTION": str(settings.adaptive_frame_selection).lower(),
            "SPLATBOT_FRAME_SELECTION_STRATEGY": settings.frame_selection_strategy,
            "SPLATBOT_BEST_FRAME_SELECTION_STRATEGY": settings.best_frame_selection_strategy,
            "SPLATBOT_FRAME_QUALITY_REJECT_THRESHOLD": settings.frame_quality_reject_threshold,
            "SPLATBOT_BLUR_REJECT_THRESHOLD": settings.blur_reject_threshold,
            "SPLATBOT_LOW_CONTRAST_REJECT_THRESHOLD": settings.low_contrast_reject_threshold,
            "SPLATBOT_OVEREXPOSED_REJECT_THRESHOLD": settings.overexposed_reject_threshold,
            "SPLATBOT_UNDEREXPOSED_REJECT_THRESHOLD": settings.underexposed_reject_threshold,
            "SPLATBOT_DUPLICATE_FRAME_THRESHOLD": settings.duplicate_frame_threshold,
            "SPLATBOT_MIN_SELECTED_VIDEO_FRAMES": settings.min_selected_video_frames,
            "SPLATBOT_MIN_COLMAP_REGISTERED_RATIO": settings.min_colmap_registered_ratio,
            "SPLATBOT_OBJECT_COLMAP_ORIGINAL_POSE_FALLBACK": str(
                settings.object_colmap_original_pose_fallback
            ).lower(),
            "SPLATBOT_COLMAP_RETRY_FRAME_COUNTS": settings.colmap_retry_frame_counts,
            "SPLATBOT_COLMAP_RETRY_MATCHING_METHODS": settings.colmap_retry_matching_methods,
            "SPLATBOT_SILHOUETTE_CLEANUP_ENABLED": str(settings.silhouette_cleanup_enabled).lower(),
            "SPLATBOT_SILHOUETTE_CLEANUP_MIN_VIEWS": settings.silhouette_cleanup_min_views,
            "SPLATBOT_SILHOUETTE_CLEANUP_MAX_VIEWS": settings.silhouette_cleanup_max_views,
            "SPLATBOT_SILHOUETTE_CLEANUP_ALPHA_THRESHOLD": settings.silhouette_cleanup_alpha_threshold,
            "SPLATBOT_SILHOUETTE_CLEANUP_PADDING_PX": settings.silhouette_cleanup_padding_px,
            "SPLATBOT_SILHOUETTE_CLEANUP_OUTSIDE_RATIO": settings.silhouette_cleanup_outside_ratio,
            "SPLATBOT_SILHOUETTE_CLEANUP_MAX_INSIDE_VIEWS": settings.silhouette_cleanup_max_inside_views,
            "SPLATBOT_SILHOUETTE_CLEANUP_MAX_INSIDE_RATIO": settings.silhouette_cleanup_max_inside_ratio,
            "SPLATBOT_SILHOUETTE_CLEANUP_MAX_REMOVE_FRACTION": settings.silhouette_cleanup_max_remove_fraction,
            "SPLATBOT_MIN_SPLAT_VERTICES": settings.min_splat_vertices,
            "SPLATBOT_MAX_FLATTENED_AXIS_RATIO": settings.max_flattened_axis_ratio,
            "SPLATBOT_FFMPEG_BIN": settings.ffmpeg_bin,
            "SPLATBOT_FFPROBE_BIN": settings.ffprobe_bin,
            "SPLATBOT_COLMAP_BIN": settings.colmap_bin,
            "SPLATBOT_NS_PROCESS_DATA_BIN": settings.ns_process_data_bin,
            "SPLATBOT_NS_TRAIN_BIN": settings.ns_train_bin,
            "SPLATBOT_NS_EXPORT_BIN": settings.ns_export_bin,
            "SPLATBOT_NS_RENDER_BIN": settings.ns_render_bin,
            "SPLATBOT_REMBG_BIN": settings.rembg_bin,
            "SPLATBOT_REMBG_REQUIRE_GPU": str(settings.rembg_require_gpu).lower(),
            "SPLATBOT_SEGMENTATION_BACKEND": settings.segmentation_backend,
            "SPLATBOT_BEST_SEGMENTATION_BACKENDS": settings.best_segmentation_backends,
            "SPLATBOT_BEST_SEGMENTATION_REQUIRED_BACKENDS": settings.best_segmentation_required_backends,
            "SPLATBOT_EXPERIMENTAL_SAM3_ENABLED": str(settings.experimental_sam3_enabled).lower(),
            "SPLATBOT_SEGMENTATION_MIN_OUTPUT_RATIO": settings.segmentation_min_output_ratio,
            "SPLATBOT_SEGMENTATION_MIN_OUTPUT_FILES": settings.segmentation_min_output_files,
            "SPLATBOT_OBJECT_MASK_BACKEND": settings.object_mask_backend,
            "SPLATBOT_OBJECT_MASK_COMMAND": settings.object_mask_command,
            "SPLATBOT_OBJECT_MASK_PROMPT": settings.object_mask_prompt,
            "SPLATBOT_SAM3_MASK_COMMAND": settings.sam3_mask_command,
            "SPLATBOT_SAM2_MASK_COMMAND": settings.sam2_mask_command,
            "SPLATBOT_SAM2_CHECKPOINT": settings.sam2_checkpoint,
            "SPLATBOT_SAM2_CONFIG": settings.sam2_config,
            "SPLATBOT_MATTING_COMMAND": settings.matting_command,
            "SPLATBOT_OBJECT_MASK_REFINE_ENABLED": str(settings.object_mask_refine_enabled).lower(),
            "SPLATBOT_OBJECT_MASK_QA_ENABLED": str(settings.object_mask_qa_enabled).lower(),
            "SPLATBOT_OBJECT_MASK_MIN_AREA_RATIO": settings.object_mask_min_area_ratio,
            "SPLATBOT_OBJECT_MASK_MAX_AREA_RATIO": settings.object_mask_max_area_ratio,
            "SPLATBOT_OBJECT_MASK_MAX_AREA_JUMP": settings.object_mask_max_area_jump,
            "SPLATBOT_OBJECT_MASK_MAX_EDGE_TOUCH_RATIO": settings.object_mask_max_edge_touch_ratio,
            "SPLATBOT_OBJECT_MASK_MIN_KEEP_FRAMES": settings.object_mask_min_keep_frames,
            "SPLATBOT_OBJECT_MASK_MIN_KEEP_RATIO": settings.object_mask_min_keep_ratio,
            "SPLATBOT_OBJECT_MASK_TRAINING_ALPHA_THRESHOLD": settings.object_mask_training_alpha_threshold,
            "SPLATBOT_POSE_BACKENDS": settings.pose_backends,
            "SPLATBOT_BEST_POSE_BACKENDS": settings.best_pose_backends,
            "SPLATBOT_BEST_POSE_REQUIRED_BACKENDS": settings.best_pose_required_backends,
            "SPLATBOT_POSE_BACKEND_COMMAND": settings.pose_backend_command,
            "SPLATBOT_DA3_MODEL": settings.da3_model,
            "SPLATBOT_DA3_USE_RAY_POSE": str(settings.da3_use_ray_pose).lower(),
            "SPLATBOT_DA3_REF_VIEW_STRATEGY": settings.da3_ref_view_strategy,
            "SPLATBOT_DA3_POSE_COMMAND": settings.da3_pose_command,
            "SPLATBOT_VGGT_POSE_COMMAND": settings.vggt_pose_command,
            "SPLATBOT_MAST3R_POSE_COMMAND": settings.mast3r_pose_command,
            "SPLATBOT_GLOMAP_BIN": settings.glomap_bin,
            "SPLATBOT_COLMAP_GLOBAL_CALIBRATE": str(settings.colmap_global_calibrate).lower(),
            "SPLATBOT_COLMAP_USE_GPU": str(settings.colmap_use_gpu).lower(),
            "SPLATBOT_GAUSSIAN_CLEANUP_ENABLED": str(settings.gaussian_cleanup_enabled).lower(),
            "SPLATBOT_GAUSSIAN_CLEANUP_MIN_OPACITY": settings.gaussian_cleanup_min_opacity,
            "SPLATBOT_GAUSSIAN_CLEANUP_MAX_SCALE_RATIO": settings.gaussian_cleanup_max_scale_ratio,
            "SPLATBOT_GAUSSIAN_CLEANUP_MAX_ANISOTROPY": settings.gaussian_cleanup_max_anisotropy,
            "SPLATBOT_GAUSSIAN_CLEANUP_MAX_REMOVE_FRACTION": settings.gaussian_cleanup_max_remove_fraction,
            "SPLATBOT_SPATIAL_CLEANUP_ENABLED": str(settings.spatial_cleanup_enabled).lower(),
            "SPLATBOT_SPATIAL_OUTLIER_RADIUS_FRACTION": settings.spatial_outlier_radius_fraction,
            "SPLATBOT_SPATIAL_OUTLIER_MIN_NEIGHBORS": settings.spatial_outlier_min_neighbors,
            "SPLATBOT_SPATIAL_COMPONENT_VOXEL_FRACTION": settings.spatial_component_voxel_fraction,
            "SPLATBOT_SPATIAL_COMPONENT_MIN_VERTICES": settings.spatial_component_min_vertices,
            "SPLATBOT_SPATIAL_COMPONENT_MIN_FRACTION": settings.spatial_component_min_fraction,
            "SPLATBOT_SPATIAL_CLEANUP_MAX_REMOVE_FRACTION": settings.spatial_cleanup_max_remove_fraction,
            "SPLATBOT_MASK_SUPPORT_CLEANUP_ENABLED": str(settings.mask_support_cleanup_enabled).lower(),
            "SPLATBOT_MASK_SUPPORT_CLEANUP_MIN_VIEWS": settings.mask_support_cleanup_min_views,
            "SPLATBOT_MASK_SUPPORT_CLEANUP_MIN_INSIDE_VIEWS": settings.mask_support_cleanup_min_inside_views,
            "SPLATBOT_MASK_SUPPORT_CLEANUP_MIN_INSIDE_RATIO": settings.mask_support_cleanup_min_inside_ratio,
            "SPLATBOT_MASK_SUPPORT_CLEANUP_MAX_REMOVE_FRACTION": settings.mask_support_cleanup_max_remove_fraction,
            "SPLATBOT_POSTPROCESS_VALIDATION_ENABLED": str(settings.postprocess_validation_enabled).lower(),
            "SPLATBOT_POSTPROCESS_VALIDATION_MAX_OUTSIDE_FRACTION": settings.postprocess_validation_max_outside_fraction,
            "SPLATBOT_POSTPROCESS_VALIDATION_MIN_INSIDE_VIEWS": settings.postprocess_validation_min_inside_views,
            "SPLATBOT_POSTPROCESS_VALIDATION_MIN_INSIDE_RATIO": settings.postprocess_validation_min_inside_ratio,
            "SPLATBOT_POSTPROCESS_VALIDATION_MAX_LOW_SUPPORT_FRACTION": settings.postprocess_validation_max_low_support_fraction,
            "SPLATBOT_POSTPROCESS_VALIDATION_MIN_CHECKED_POINTS": settings.postprocess_validation_min_checked_points,
            "SPLATBOT_POSTPROCESS_VALIDATION_MAX_UNOBSERVED_FRACTION": settings.postprocess_validation_max_unobserved_fraction,
            "SPLATBOT_POSTPROCESS_VALIDATION_SAMPLE_LIMIT": settings.postprocess_validation_sample_limit,
            "SPLATBOT_RENDER_VALIDATION_COMMAND": settings.render_validation_command,
            "SPLATBOT_QUALITY_REPORT_ENABLED": str(settings.quality_report_enabled).lower(),
            "SPLATBOT_COMMAND_TIMEOUT_SECONDS": settings.command_timeout_seconds,
            "SPLATBOT_COMMAND_TAIL_BYTES": settings.command_tail_bytes,
            "SPLATBOT_DEPTH_BACKENDS": settings.depth_backends,
            "SPLATBOT_BEST_DEPTH_BACKENDS": settings.best_depth_backends,
            "SPLATBOT_BEST_DEPTH_REQUIRED_BACKENDS": settings.best_depth_required_backends,
            "SPLATBOT_DEPTH_BACKEND_COMMAND": settings.depth_backend_command,
            "SPLATBOT_DA3_DEPTH_COMMAND": settings.da3_depth_command,
            "SPLATBOT_DEPTH_ANYTHING_V2_COMMAND": settings.depth_anything_v2_command,
            "SPLATBOT_TRAIN_METHOD": settings.train_method,
            "SPLATBOT_TRAIN_BACKENDS": settings.train_backends,
            "SPLATBOT_BEST_TRAIN_BACKENDS": settings.best_train_backends,
            "SPLATBOT_BEST_TRAIN_REQUIRED_BACKENDS": settings.best_train_required_backends,
            "SPLATBOT_EXPERIMENTAL_DN_SPLATTER_ENABLED": str(settings.experimental_dn_splatter_enabled).lower(),
            "SPLATBOT_TRAIN_BACKEND_COMMAND": settings.train_backend_command,
            "SPLATBOT_MCMC_TRAIN_COMMAND": settings.mcmc_train_command,
            "SPLATBOT_MIP_SPLATTING_TRAIN_COMMAND": settings.mip_splatting_train_command,
            "SPLATBOT_2DGS_TRAIN_COMMAND": settings.twodgs_train_command,
            "SPLATBOT_TRAIN_EXTRA_ARGS": settings.train_extra_args,
            "SPLATBOT_TRAIN_MAX_ITERATIONS": settings.train_max_iterations,
            "SPLATBOT_TRAIN_STEPS_PER_SAVE": settings.train_steps_per_save,
            "SPLATBOT_FAST_MAX_VIDEO_FRAMES": settings.fast_max_video_frames,
            "SPLATBOT_FAST_TRAIN_METHOD": settings.fast_train_method,
            "SPLATBOT_FAST_TRAIN_EXTRA_ARGS": settings.fast_train_extra_args,
            "SPLATBOT_FAST_TRAIN_MAX_ITERATIONS": settings.fast_train_max_iterations,
            "SPLATBOT_FAST_TRAIN_STEPS_PER_SAVE": settings.fast_train_steps_per_save,
            "SPLATBOT_BEST_MAX_VIDEO_FRAMES": settings.best_max_video_frames,
            "SPLATBOT_BEST_TRAIN_METHOD": settings.best_train_method,
            "SPLATBOT_BEST_TRAIN_EXTRA_ARGS": settings.best_train_extra_args,
            "SPLATBOT_BEST_TRAIN_MAX_ITERATIONS": settings.best_train_max_iterations,
            "SPLATBOT_BEST_TRAIN_STEPS_PER_SAVE": settings.best_train_steps_per_save,
            "SPLATBOT_MESH_EXPORT_ENABLED": str(settings.mesh_export_enabled).lower(),
            "SPLATBOT_MESH_BACKEND": settings.mesh_backend,
            "SPLATBOT_MESH_EXPORT_COMMAND": settings.mesh_export_command,
            "SPLATBOT_MESH_EXPORT_FILENAME": settings.mesh_export_filename,
            "SPLATBOT_MESH_EXPORT_REQUIRED": str(settings.mesh_export_required).lower(),
            "SPLATBOT_RENDER_PREVIEW": str(settings.render_preview).lower(),
            "SPLATBOT_LOG_COMMAND_OUTPUT": "true",
        }.items()
    )
    return f"""
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
mkdir -p /root/.ssh /workspace/splatbot-app
if ! command -v ssh >/dev/null || ! command -v rsync >/dev/null || ! command -v python3 >/dev/null; then
  apt-get update
  apt-get install -y openssh-client rsync python3
fi
export SPLATBOT_VPS_KNOWN_HOSTS={vps_known_hosts}
export SPLATBOT_VPS_KNOWN_HOSTS_FILE=/root/.ssh/known_hosts
if [ -n "$SPLATBOT_VPS_KNOWN_HOSTS" ]; then
  printf '%s\n' "$SPLATBOT_VPS_KNOWN_HOSTS" > "$SPLATBOT_VPS_KNOWN_HOSTS_FILE"
else
  ssh-keyscan -T 15 -H {host} > "$SPLATBOT_VPS_KNOWN_HOSTS_FILE"
fi
chmod 600 "$SPLATBOT_VPS_KNOWN_HOSTS_FILE"
SSH_OPTS="-i /root/.ssh/id_ed25519 -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=$SPLATBOT_VPS_KNOWN_HOSTS_FILE -o LogLevel=ERROR -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=6"
rsync -r --delete --no-perms --no-owner --no-group --omit-dir-times --exclude "__pycache__" --exclude "*.egg-info" -e "ssh $SSH_OPTS" {user}@{host}:/opt/splatbot/app/pyproject.toml /workspace/splatbot-app/
rsync -r --delete --no-perms --no-owner --no-group --omit-dir-times --exclude "__pycache__" --exclude "*.egg-info" -e "ssh $SSH_OPTS" {user}@{host}:/opt/splatbot/app/splatbot/ /workspace/splatbot-app/splatbot/
rsync -r --delete --no-perms --no-owner --no-group --omit-dir-times -e "ssh $SSH_OPTS" {user}@{host}:/opt/splatbot/app/scripts/ /workspace/splatbot-app/scripts/
chmod +x /workspace/splatbot-app/scripts/runpod_worker.sh
export SPLATBOT_JOB_ID={shlex.quote(job.id)}
export SPLATBOT_SESSION_ID={shlex.quote(job.session_id)}
export SPLATBOT_SCAN_MODE={shlex.quote(job.mode.value)}
export SPLATBOT_VPS_HOST={host}
export SPLATBOT_VPS_USER={user}
{venv_export}
{pipeline_exports}
export SPLATBOT_RUNPOD_BOOTSTRAP_COMMAND={bootstrap_command}
export SPLATBOT_RUNPOD_SETUP_COMMAND={setup_command}
export SPLATBOT_RUNPOD_RUNTIME_CACHE_VERSION={runtime_cache_version}
export SPLATBOT_RUNPOD_RUNTIME_CACHE_MARKER={runtime_cache_marker}
exec /workspace/splatbot-app/scripts/runpod_worker.sh
"""
