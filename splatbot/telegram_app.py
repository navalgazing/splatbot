from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .config import ScanMode, Settings
from .media import MediaValidationError, validate_submission
from .models import JobArtifact, JobStatus, MediaKind, ScanJob
from .storage import Store

LOGGER = logging.getLogger(__name__)


def _allowed(settings: Settings, user_id: int | None) -> bool:
    return user_id is not None and (
        not settings.allowed_telegram_ids or user_id in settings.allowed_telegram_ids
    )


def _settings(context: ContextTypes.DEFAULT_TYPE) -> Settings:
    return context.application.bot_data["settings"]


def _store(context: ContextTypes.DEFAULT_TYPE) -> Store:
    return context.application.bot_data["store"]


async def _guard(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    user_id = update.effective_user.id if update.effective_user else None
    if _allowed(_settings(context), user_id):
        return True
    if update.effective_message:
        await update.effective_message.reply_text("This bot is private.")
    return False


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update, context):
        return
    await update.effective_message.reply_text(
        "Send /new, upload 100-300 photos or one short video, then send /submit."
    )


async def new_session(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update, context):
        return
    settings = _settings(context)
    store = _store(context)
    user_id = update.effective_user.id
    existing = await store.get_active_session(user_id)
    if existing:
        await store.set_session_status(existing.id, JobStatus.CANCELLED)
    session = await store.create_session(user_id, settings.default_scan_mode)
    session_dir = settings.data_dir / "sessions" / session.id
    session_dir.mkdir(parents=True, exist_ok=True)
    await update.effective_message.reply_text(
        f"New {session.mode.value} scan started. Upload photos or one video, then /submit."
    )


async def set_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update, context):
        return
    if not context.args or context.args[0] not in {ScanMode.SCENE.value, ScanMode.OBJECT.value}:
        await update.effective_message.reply_text("Use /mode scene or /mode object.")
        return
    store = _store(context)
    session = await store.get_active_session(update.effective_user.id)
    if session is None:
        session = await store.create_session(update.effective_user.id, ScanMode(context.args[0]))
    else:
        await store.set_session_mode(session.id, ScanMode(context.args[0]))
    await update.effective_message.reply_text(f"Mode set to {context.args[0]}.")


async def receive_media(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update, context):
        return
    if update.effective_message is None:
        return
    settings = _settings(context)
    store = _store(context)
    user_id = update.effective_user.id
    session = await store.get_active_session(user_id)
    if session is None:
        session = await store.create_session(user_id, settings.default_scan_mode)
    session_dir = settings.data_dir / "sessions" / session.id
    session_dir.mkdir(parents=True, exist_ok=True)

    await update.effective_chat.send_action(ChatAction.UPLOAD_DOCUMENT)
    if update.effective_message.video:
        tg_file = await update.effective_message.video.get_file()
        suffix = Path(tg_file.file_path or "upload.mp4").suffix or ".mp4"
        kind = MediaKind.VIDEO
    elif update.effective_message.document:
        doc = update.effective_message.document
        tg_file = await doc.get_file()
        suffix = Path(doc.file_name or tg_file.file_path or "upload.bin").suffix or ".bin"
        kind = MediaKind.PHOTO if suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".heic"} else MediaKind.VIDEO
    elif update.effective_message.photo:
        tg_file = await update.effective_message.photo[-1].get_file()
        suffix = ".jpg"
        kind = MediaKind.PHOTO
    else:
        return

    dest = session_dir / f"{tg_file.file_unique_id}{suffix.lower()}"
    await tg_file.download_to_drive(custom_path=dest)
    await store.add_media(session.id, kind, dest)
    count = len(await store.list_media(session.id))
    if count == 1 or count % 25 == 0:
        await update.effective_message.reply_text(f"Received {count} file(s).")


async def submit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update, context):
        return
    settings = _settings(context)
    store = _store(context)
    session = await store.get_active_session(update.effective_user.id)
    if session is None:
        await update.effective_message.reply_text("No active scan. Send /new first.")
        return
    items = await store.list_media(session.id)
    try:
        validate_submission(items, settings.min_images, settings.max_images)
    except MediaValidationError as exc:
        await update.effective_message.reply_text(str(exc))
        return
    job = await store.create_job(session)
    await update.effective_message.reply_text(
        f"Queued job {job.id}. Only one GPU job runs at a time; use /status for updates."
    )


def _artifact_label(artifact: JobArtifact) -> str:
    location = artifact.url or artifact.local_path
    return f"{artifact.kind.value}: {location}"


def _job_status_text(job: ScanJob, artifacts: list[JobArtifact]) -> str:
    lines = [f"Job {job.id}: {job.status.value}, mode={job.mode.value}."]
    if job.error:
        lines.append(f"Error: {job.error}")
    if artifacts:
        lines.append("Artifacts:")
        lines.extend(_artifact_label(artifact) for artifact in artifacts)
    return "\n".join(lines)


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update, context):
        return
    store = _store(context)
    user_id = update.effective_user.id
    session = await store.get_active_session(user_id)
    if session:
        items = await store.list_media(session.id)
        await update.effective_message.reply_text(
            f"Collecting {len(items)} file(s), mode={session.mode.value}."
        )
        return
    job = await store.get_latest_job_for_user(user_id)
    if job:
        artifacts = await store.list_artifacts(job.id)
        await update.effective_message.reply_text(_job_status_text(job, artifacts))
        return
    await update.effective_message.reply_text("No active upload session or recent job.")


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update, context):
        return
    store = _store(context)
    session = await store.get_active_session(update.effective_user.id)
    if session is None:
        await update.effective_message.reply_text("No active scan.")
        return
    await store.set_session_status(session.id, JobStatus.CANCELLED)
    await update.effective_message.reply_text("Cancelled current scan.")


async def amain() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = Settings()
    settings.require_telegram()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    store = Store(settings.database_path)
    await store.init()

    app = Application.builder().token(settings.telegram_token).build()
    app.bot_data["settings"] = settings
    app.bot_data["store"] = store
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("new", new_session))
    app.add_handler(CommandHandler("mode", set_mode))
    app.add_handler(CommandHandler("submit", submit))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("cancel", cancel))
    app.add_handler(MessageHandler(filters.PHOTO | filters.VIDEO | filters.Document.ALL, receive_media))
    await app.initialize()
    await app.start()
    await app.updater.start_polling()
    try:
        while True:
            await asyncio.sleep(3600)
    finally:
        await app.updater.stop()
        await app.stop()
        await app.shutdown()


def main() -> None:
    asyncio.run(amain())
