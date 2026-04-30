from __future__ import annotations

import argparse
import asyncio
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

from .artifact_manifest import copy_artifact_bundle, write_artifact_manifest
from .config import ScanMode, ScanPreset, Settings
from .media import classify_path
from .models import JobStatus, MediaItem, MediaKind
from .pipeline import ScanPipeline


def media_items(media_dir: Path) -> list[MediaItem]:
    items: list[MediaItem] = []
    for path in sorted(media_dir.iterdir()):
        if not path.is_file():
            continue
        kind = classify_path(path)
        items.append(
            MediaItem(
                id=path.name,
                session_id="runpod",
                kind=kind,
                local_path=str(path),
                remote_key=None,
                created_at=datetime.now(UTC),
            )
        )
    return items


async def run(job_id: str, mode: ScanMode, preset: ScanPreset, media_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(os.environ.get("SPLATBOT_WORKER_DATA_DIR", "/workspace/splatbot-data"))
    settings = Settings(data_dir=data_dir)
    media = media_items(media_dir)
    images_dir = settings.job_dir(job_id) / "uploaded_images"
    normalized_media = await normalize_photo_media(settings, media, images_dir)
    try:
        outputs = await ScanPipeline(settings).run(job_id, mode, normalized_media, report_status, preset)
    except Exception:
        copy_failure_artifacts(settings.job_dir(job_id), output_dir)
        raise
    shutil.copy2(outputs.cleaned_ply, output_dir / "cleaned_splat.ply")
    export_output = output_dir / "export"
    export_output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(outputs.cleaned_ply, export_output / "cleaned_splat.ply")
    if outputs.raw_ply is not None and outputs.raw_ply.exists():
        shutil.copy2(outputs.raw_ply, output_dir / "raw_splat.ply")
        shutil.copy2(outputs.raw_ply, export_output / "raw_splat.ply")
    if outputs.mesh_path is not None and outputs.mesh_path.exists():
        shutil.copy2(outputs.mesh_path, output_dir / outputs.mesh_path.name)
        shutil.copy2(outputs.mesh_path, export_output / outputs.mesh_path.name)
    if outputs.preview_mp4 is not None and outputs.preview_mp4.exists():
        shutil.copy2(outputs.preview_mp4, output_dir / "turntable.mp4")
        renders_output = output_dir / "renders"
        renders_output.mkdir(parents=True, exist_ok=True)
        shutil.copy2(outputs.preview_mp4, renders_output / "turntable.mp4")
    if outputs.metrics_path is not None and outputs.metrics_path.exists():
        shutil.copy2(outputs.metrics_path, output_dir / "metrics.json")
    settings_path = settings.job_dir(job_id) / "settings.json"
    if settings_path.exists():
        shutil.copy2(settings_path, output_dir / "settings.json")
    if outputs.quality_report_path is not None and outputs.quality_report_path.exists():
        shutil.copy2(outputs.quality_report_path, output_dir / "quality_report.json")
    if outputs.candidate_report_path is not None and outputs.candidate_report_path.exists():
        shutil.copy2(outputs.candidate_report_path, output_dir / "candidate_report.json")
    copy_success_artifacts(settings.job_dir(job_id), output_dir)


def copy_failure_artifacts(job_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("metrics.json", "settings.json", "quality_report.json", "candidate_report.json"):
        src = job_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)
    diagnostics = job_dir / "diagnostics"
    if diagnostics.exists():
        shutil.copytree(diagnostics, output_dir / "diagnostics", dirs_exist_ok=True)
    copy_artifact_bundle(job_dir, output_dir)
    write_artifact_manifest(output_dir, job_id=job_dir.name)


def copy_success_artifacts(job_dir: Path, output_dir: Path) -> None:
    copy_artifact_bundle(job_dir, output_dir)
    write_artifact_manifest(output_dir, job_id=job_dir.name)


async def report_status(job_id: str, status: JobStatus) -> None:
    command = os.environ.get("SPLATBOT_STATUS_COMMAND")
    if not command:
        return
    proc = await asyncio.create_subprocess_exec(command, job_id, status.value)
    returncode = await proc.wait()
    if returncode != 0:
        print(f"status update failed with exit code {returncode}: {status.value}", flush=True)


async def normalize_photo_media(settings: Settings, media: list[MediaItem], images_dir: Path) -> list[MediaItem]:
    if not media or any(item.kind != MediaKind.PHOTO for item in media):
        return media
    images_dir.mkdir(parents=True, exist_ok=True)
    normalized: list[MediaItem] = []
    for idx, item in enumerate(media, start=1):
        src = Path(item.local_path)
        suffix = src.suffix.lower()
        if suffix in {".jpg", ".jpeg", ".png"}:
            dest = images_dir / f"image_{idx:05d}{suffix}"
            shutil.copy2(src, dest)
        else:
            dest = images_dir / f"image_{idx:05d}.png"
            await transcode_photo(settings, src, dest)
        normalized.append(
            MediaItem(
                id=item.id,
                session_id=item.session_id,
                kind=item.kind,
                local_path=str(dest),
                remote_key=item.remote_key,
                created_at=item.created_at,
            )
        )
    return normalized


async def transcode_photo(settings: Settings, src: Path, dest: Path) -> None:
    proc = await asyncio.create_subprocess_exec(
        settings.ffmpeg_bin,
        "-y",
        "-i",
        str(src),
        str(dest),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        detail = stderr.decode(errors="replace")[-2000:] or stdout.decode(errors="replace")[-2000:]
        raise RuntimeError(f"failed to transcode uploaded image {src.name}: {detail}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Splatbot job from a media directory.")
    parser.add_argument("job_id")
    parser.add_argument("mode", choices=[mode.value for mode in ScanMode])
    parser.add_argument("--preset", choices=[preset.value for preset in ScanPreset], default=None)
    parser.add_argument("media_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    settings = Settings()
    asyncio.run(
        run(
            args.job_id,
            ScanMode(args.mode),
            ScanPreset(args.preset or settings.default_scan_preset),
            args.media_dir,
            args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
