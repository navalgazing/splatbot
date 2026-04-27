from __future__ import annotations

import argparse
import asyncio
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

from .config import ScanMode, ScanPreset, Settings
from .media import classify_path
from .models import MediaItem
from .pipeline import ScanPipeline
from .models import JobStatus


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
    settings = Settings(data_dir=Path("/workspace/splatbot-data"))
    outputs = await ScanPipeline(settings).run(job_id, mode, media_items(media_dir), report_status, preset)
    shutil.copy2(outputs.cleaned_ply, output_dir / "cleaned_splat.ply")
    if outputs.mesh_path is not None and outputs.mesh_path.exists():
        shutil.copy2(outputs.mesh_path, output_dir / outputs.mesh_path.name)
    if outputs.preview_mp4 is not None and outputs.preview_mp4.exists():
        shutil.copy2(outputs.preview_mp4, output_dir / "turntable.mp4")
    if outputs.metrics_path is not None and outputs.metrics_path.exists():
        shutil.copy2(outputs.metrics_path, output_dir / "metrics.json")
    if outputs.quality_report_path is not None and outputs.quality_report_path.exists():
        shutil.copy2(outputs.quality_report_path, output_dir / "quality_report.json")
    if outputs.candidate_report_path is not None and outputs.candidate_report_path.exists():
        shutil.copy2(outputs.candidate_report_path, output_dir / "candidate_report.json")


async def report_status(job_id: str, status: JobStatus) -> None:
    command = os.environ.get("SPLATBOT_STATUS_COMMAND")
    if not command:
        return
    proc = await asyncio.create_subprocess_exec(command, job_id, status.value)
    returncode = await proc.wait()
    if returncode != 0:
        print(f"status update failed with exit code {returncode}: {status.value}", flush=True)


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
