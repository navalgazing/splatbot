from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from telegram import Bot
from telegram.error import RetryAfter, TelegramError

from .models import ArtifactKind, JobArtifact, ScanJob

TELEGRAM_UPLOAD_LIMIT_BYTES = 45 * 1024 * 1024
TELEGRAM_MESSAGE_LIMIT = 4096
LOGGER = logging.getLogger(__name__)


def _shorten(text: str, limit: int = 1200) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "..."


def _fit_message(text: str) -> str:
    if len(text) <= TELEGRAM_MESSAGE_LIMIT:
        return text
    return text[: TELEGRAM_MESSAGE_LIMIT - 3] + "..."


class TelegramNotifier:
    def __init__(self, token: str) -> None:
        self.bot = Bot(token)

    async def job_done(self, job: ScanJob, artifacts: list[JobArtifact]) -> None:
        viewer = next((artifact for artifact in artifacts if artifact.kind == ArtifactKind.VIEWER and artifact.url), None)
        quality = load_quality_report(artifacts)
        fallback_summary = format_quality_summary(quality)
        suffix = f"\n\n{fallback_summary}" if fallback_summary else "\n\nPipeline completed cleanly."
        if viewer:
            await send_message_with_retry(
                self.bot,
                chat_id=job.telegram_user_id,
                text=f"Job {job.id} finished.\nOpen 3D viewer: {viewer.url}{suffix}",
            )
        else:
            await send_message_with_retry(
                self.bot,
                chat_id=job.telegram_user_id,
                text=f"Job {job.id} finished. Sending results now.{suffix}",
            )
        downloadable = [artifact for artifact in artifacts if artifact.kind != ArtifactKind.VIEWER]
        for artifact in downloadable:
            local_path = Path(artifact.local_path)
            if artifact.url:
                await send_message_with_retry(
                    self.bot,
                    chat_id=job.telegram_user_id,
                    text=f"{artifact.kind.value}: {artifact.url}",
                )
            elif viewer and artifact.kind == ArtifactKind.PLY:
                continue
            try:
                local_size = local_path.stat().st_size
            except OSError:
                await send_message_with_retry(
                    self.bot,
                    chat_id=job.telegram_user_id,
                    text=f"{artifact.kind.value}: artifact file was not found on the server.",
                )
                continue
            if local_size > TELEGRAM_UPLOAD_LIMIT_BYTES:
                await send_message_with_retry(
                    self.bot,
                    chat_id=job.telegram_user_id,
                    text=f"{artifact.kind.value}: saved on the server; use the viewer download link.",
                )
            elif artifact.kind == ArtifactKind.PREVIEW:
                try:
                    with local_path.open("rb") as video:
                        await self.bot.send_video(
                            chat_id=job.telegram_user_id,
                            video=video,
                            caption="Preview",
                        )
                except OSError:
                    await send_message_with_retry(
                        self.bot,
                        chat_id=job.telegram_user_id,
                        text=f"{artifact.kind.value}: artifact file was not found on the server.",
                    )
            else:
                try:
                    with local_path.open("rb") as document:
                        await self.bot.send_document(
                            chat_id=job.telegram_user_id,
                            document=document,
                            caption=artifact.kind.value,
                        )
                except OSError:
                    await send_message_with_retry(
                        self.bot,
                        chat_id=job.telegram_user_id,
                        text=f"{artifact.kind.value}: artifact file was not found on the server.",
                    )

    async def job_failed(self, job: ScanJob, error: str, metrics_path: Path | None = None) -> None:
        summary = format_failure_summary(metrics_path)
        suffix = f"\n\n{summary}" if summary else ""
        await send_message_with_retry(
            self.bot,
            chat_id=job.telegram_user_id,
            text=f"Job {job.id} failed: {_shorten(error)}{suffix}",
        )


async def send_message_with_retry(bot: Bot, *, chat_id: int, text: str) -> None:
    fitted = _fit_message(text)
    try:
        await bot.send_message(chat_id=chat_id, text=fitted)
    except RetryAfter as exc:
        raw_retry_after = getattr(exc, "_retry_after", 1)
        if hasattr(raw_retry_after, "total_seconds"):
            raw_retry_after = raw_retry_after.total_seconds()
        retry_after = max(1, int(raw_retry_after))
        LOGGER.warning("Telegram rate limited message to chat %s; retrying after %ss", chat_id, retry_after)
        await asyncio.sleep(retry_after)
        await bot.send_message(chat_id=chat_id, text=fitted)
    except TelegramError:
        LOGGER.warning("Telegram message to chat %s failed", chat_id, exc_info=True)
        raise


def load_quality_report(artifacts: list[JobArtifact]) -> dict | None:
    report_artifact = next((artifact for artifact in artifacts if artifact.kind == ArtifactKind.QUALITY_REPORT), None)
    if report_artifact is None:
        return None
    try:
        path = Path(report_artifact.local_path)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    return None


def format_quality_summary(report: dict | None) -> str:
    if not report:
        return ""
    events = report.get("event_summary") or []
    warnings = report.get("warnings") or []
    issues = report.get("issues") or []
    if not events and not warnings and not issues:
        return ""
    lines = ["Pipeline completed with fallbacks/warnings:"]
    for item in events[:5]:
        lines.append(f"- {item}")
    for item in warnings[:5]:
        if item == "pipeline_fallbacks_or_skips" and events:
            continue
        lines.append(f"- warning: {item}")
    for item in issues[:3]:
        lines.append(f"- issue: {item}")
    return _shorten("\n".join(lines), 1200)


def format_failure_summary(metrics_path: Path | None) -> str:
    if metrics_path is None or not metrics_path.exists():
        return ""
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return ""
    stages = metrics.get("stages") or {}
    events = metrics.get("pipeline_events") or []
    last_stage = next(reversed(stages), None) if isinstance(stages, dict) and stages else None
    lines = []
    if last_stage:
        lines.append(f"Last completed stage: {last_stage}.")
    recent = [
        event
        for event in events
        if isinstance(event, dict) and event.get("status") in {"fallback", "failure", "skip"}
    ][-4:]
    if recent:
        lines.append("Recent pipeline events:")
        for event in recent:
            stage = event.get("stage") or "pipeline"
            backend = event.get("backend")
            label = f"{stage}/{backend}" if backend else str(stage)
            lines.append(f"- {label}: {event.get('status')} ({event.get('reason')})")
    return _shorten("\n".join(lines), 1200)
