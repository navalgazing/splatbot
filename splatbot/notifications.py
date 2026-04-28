from __future__ import annotations

import json
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
        quality = load_quality_report(artifacts)
        fallback_summary = format_quality_summary(quality)
        suffix = f"\n\n{fallback_summary}" if fallback_summary else "\n\nPipeline completed cleanly."
        if viewer:
            await self.bot.send_message(
                chat_id=job.telegram_user_id,
                text=f"Job {job.id} finished.\nOpen 3D viewer: {viewer.url}{suffix}",
            )
        else:
            await self.bot.send_message(
                chat_id=job.telegram_user_id,
                text=f"Job {job.id} finished. Sending results now.{suffix}",
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

    async def job_failed(self, job: ScanJob, error: str, metrics_path: Path | None = None) -> None:
        summary = format_failure_summary(metrics_path)
        suffix = f"\n\n{summary}" if summary else ""
        await self.bot.send_message(
            chat_id=job.telegram_user_id,
            text=f"Job {job.id} failed: {_shorten(error)}{suffix}",
        )


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
