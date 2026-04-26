from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

from .commands import CommandRunner
from .config import ScanMode, Settings
from .models import JobStatus, MediaItem, MediaKind


StatusCallback = Callable[[str, JobStatus], Awaitable[None]]


@dataclass(frozen=True)
class PipelineOutputs:
    cleaned_ply: Path
    preview_mp4: Path | None


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
    ) -> PipelineOutputs:
        job_dir = self.settings.job_dir(job_id)
        images_dir = job_dir / "images"
        processed_dir = job_dir / "processed"
        ns_dir = job_dir / "nerfstudio"
        export_dir = job_dir / "export"
        render_dir = job_dir / "renders"
        for path in (images_dir, processed_dir, ns_dir, export_dir, render_dir):
            path.mkdir(parents=True, exist_ok=True)

        is_video = len(media) == 1 and media[0].kind == MediaKind.VIDEO
        if is_video:
            await self.extract_video_frames(Path(media[0].local_path), images_dir)
        else:
            await self.copy_or_link_images(media, images_dir)

        input_images_dir = images_dir
        if mode == ScanMode.OBJECT:
            object_dir = job_dir / "object_images"
            object_dir.mkdir(parents=True, exist_ok=True)
            await self.remove_backgrounds(images_dir, object_dir)
            input_images_dir = object_dir

        if on_status:
            await on_status(job_id, JobStatus.COLMAP)
        await self.process_data(
            input_images_dir,
            processed_dir,
            matching_method="sequential" if is_video else None,
            max_dataset_size=self.settings.max_video_frames if is_video else self.settings.max_images,
        )
        if on_status:
            await on_status(job_id, JobStatus.TRAINING)
        await self.train_splatfacto(processed_dir, ns_dir)
        if on_status:
            await on_status(job_id, JobStatus.EXPORTING)
        raw_ply = await self.export_ply(ns_dir, export_dir)
        cleaned_ply = export_dir / "cleaned_splat.ply"
        clean_ply(raw_ply, cleaned_ply)
        preview_mp4 = None
        if self.settings.render_preview:
            if on_status:
                await on_status(job_id, JobStatus.RENDERING)
            preview_mp4 = await self.render_turntable(ns_dir, render_dir)
        return PipelineOutputs(cleaned_ply=cleaned_ply, preview_mp4=preview_mp4)

    async def extract_video_frames(self, video: Path, images_dir: Path) -> None:
        await self.runner.run(
            [
                self.settings.ffmpeg_bin,
                "-i",
                str(video),
                "-t",
                str(self.settings.max_video_seconds),
                "-vf",
                f"fps={self.settings.max_video_frames}/{self.settings.max_video_seconds}",
                "-q:v",
                "2",
                str(images_dir / "frame_%05d.jpg"),
            ]
        )

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
        max_dataset_size: int | None = None,
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
        if max_dataset_size:
            argv.extend(["--max-dataset-size", str(max_dataset_size)])
        if not self.settings.colmap_use_gpu:
            argv.append("--no-gpu")
        await self.runner.run(argv)

    async def train_splatfacto(self, processed_dir: Path, ns_dir: Path) -> None:
        await self.runner.run(
            [
                self.settings.ns_train_bin,
                "splatfacto",
                "--data",
                str(processed_dir),
                "--output-dir",
                str(ns_dir),
                "--max-num-iterations",
                str(self.settings.train_max_iterations),
                "--steps-per-save",
                str(self.settings.train_steps_per_save),
                "--viewer.quit-on-train-completion",
                "True",
            ]
        )

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
