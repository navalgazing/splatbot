from __future__ import annotations

import argparse
import asyncio
import shutil
from datetime import UTC, datetime
from pathlib import Path

from .config import ScanMode, Settings
from .media import classify_path
from .models import MediaItem
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


async def run(job_id: str, mode: ScanMode, media_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = Settings(data_dir=Path("/workspace/splatbot-data"))
    outputs = await ScanPipeline(settings).run(job_id, mode, media_items(media_dir))
    shutil.copy2(outputs.cleaned_ply, output_dir / "cleaned_splat.ply")
    if outputs.preview_mp4 is not None and outputs.preview_mp4.exists():
        shutil.copy2(outputs.preview_mp4, output_dir / "turntable.mp4")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Splatbot job from a media directory.")
    parser.add_argument("job_id")
    parser.add_argument("mode", choices=[mode.value for mode in ScanMode])
    parser.add_argument("media_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    asyncio.run(run(args.job_id, ScanMode(args.mode), args.media_dir, args.output_dir))


if __name__ == "__main__":
    main()
