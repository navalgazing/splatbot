from __future__ import annotations

from pathlib import Path

from telegram import Bot

from .models import ArtifactKind, JobArtifact, ScanJob


class TelegramNotifier:
    def __init__(self, token: str) -> None:
        self.bot = Bot(token)

    async def job_done(self, job: ScanJob, artifacts: list[JobArtifact]) -> None:
        await self.bot.send_message(
            chat_id=job.telegram_user_id,
            text=f"Job {job.id} finished. Sending results now.",
        )
        for artifact in artifacts:
            if artifact.url:
                await self.bot.send_message(
                    chat_id=job.telegram_user_id,
                    text=f"{artifact.kind.value}: {artifact.url}",
                )
            elif artifact.kind == ArtifactKind.PREVIEW:
                with Path(artifact.local_path).open("rb") as video:
                    await self.bot.send_video(
                        chat_id=job.telegram_user_id,
                        video=video,
                        caption="Preview",
                    )
            else:
                with Path(artifact.local_path).open("rb") as document:
                    await self.bot.send_document(
                        chat_id=job.telegram_user_id,
                        document=document,
                        caption=artifact.kind.value,
                    )

    async def job_failed(self, job: ScanJob, error: str) -> None:
        await self.bot.send_message(
            chat_id=job.telegram_user_id,
            text=f"Job {job.id} failed: {error}",
        )
