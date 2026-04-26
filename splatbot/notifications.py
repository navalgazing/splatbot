from __future__ import annotations

from pathlib import Path

from telegram import Bot

from .models import ArtifactKind, JobArtifact, ScanJob

TELEGRAM_UPLOAD_LIMIT_BYTES = 45 * 1024 * 1024


def _shorten(text: str, limit: int = 1200) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "..."


class TelegramNotifier:
    def __init__(self, token: str) -> None:
        self.bot = Bot(token)

    async def job_done(self, job: ScanJob, artifacts: list[JobArtifact]) -> None:
        viewer = next((artifact for artifact in artifacts if artifact.kind == ArtifactKind.VIEWER and artifact.url), None)
        if viewer:
            await self.bot.send_message(
                chat_id=job.telegram_user_id,
                text=f"Job {job.id} finished.\nOpen 3D viewer: {viewer.url}",
            )
        else:
            await self.bot.send_message(
                chat_id=job.telegram_user_id,
                text=f"Job {job.id} finished. Sending results now.",
            )
        downloadable = [artifact for artifact in artifacts if artifact.kind != ArtifactKind.VIEWER]
        for artifact in downloadable:
            local_path = Path(artifact.local_path)
            if artifact.url:
                await self.bot.send_message(
                    chat_id=job.telegram_user_id,
                    text=f"{artifact.kind.value}: {artifact.url}",
                )
            elif viewer and artifact.kind == ArtifactKind.PLY:
                continue
            elif not local_path.exists():
                await self.bot.send_message(
                    chat_id=job.telegram_user_id,
                    text=f"{artifact.kind.value}: artifact file was not found on the server.",
                )
            elif local_path.stat().st_size > TELEGRAM_UPLOAD_LIMIT_BYTES:
                await self.bot.send_message(
                    chat_id=job.telegram_user_id,
                    text=f"{artifact.kind.value}: saved on the server; use the viewer download link.",
                )
            elif artifact.kind == ArtifactKind.PREVIEW:
                with local_path.open("rb") as video:
                    await self.bot.send_video(
                        chat_id=job.telegram_user_id,
                        video=video,
                        caption="Preview",
                    )
            else:
                with local_path.open("rb") as document:
                    await self.bot.send_document(
                        chat_id=job.telegram_user_id,
                        document=document,
                        caption=artifact.kind.value,
                    )

    async def job_failed(self, job: ScanJob, error: str) -> None:
        await self.bot.send_message(
            chat_id=job.telegram_user_id,
            text=f"Job {job.id} failed: {_shorten(error)}",
        )
