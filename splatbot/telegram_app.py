from __future__ import annotations

import asyncio
import logging
import shutil
import signal
from pathlib import Path

from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .config import ScanMode, ScanPreset, Settings, TelegramMode
from .logging_config import configure_logging
from .media import MediaValidationError, classify_path, validate_submission
from .models import JobArtifact, JobStatus, MediaKind, ScanJob
from .storage import Store

LOGGER = logging.getLogger(__name__)

PRESET_LABELS = {
    ScanPreset.FAST: "Fast",
    ScanPreset.BALANCED: "Balanced",
    ScanPreset.BEST: "Best",
}


def _allowed(settings: Settings, user_id: int | None) -> bool:
    return user_id is not None and (
        settings.allow_all_telegram_users or user_id in settings.allowed_telegram_ids
    )


def _settings(context: ContextTypes.DEFAULT_TYPE) -> Settings:
    return context.application.bot_data["settings"]


def _store(context: ContextTypes.DEFAULT_TYPE) -> Store:
    return context.application.bot_data["store"]


def _main_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("New object scan", callback_data="new:object")],
            [InlineKeyboardButton("New scene scan", callback_data="new:scene")],
            [
                InlineKeyboardButton("Status", callback_data="status"),
                InlineKeyboardButton("Help", callback_data="help"),
            ],
        ]
    )


def _session_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("Submit scan", callback_data="submit")],
            [InlineKeyboardButton("Change preset", callback_data="preset_menu")],
            [
                InlineKeyboardButton("Status", callback_data="status"),
                InlineKeyboardButton("Cancel", callback_data="cancel"),
            ],
        ]
    )


def _preset_keyboard(mode: ScanMode, prefix: str = "preset") -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("Fast", callback_data=f"{prefix}:{mode.value}:fast")],
            [InlineKeyboardButton("Balanced", callback_data=f"{prefix}:{mode.value}:balanced")],
            [InlineKeyboardButton("Best", callback_data=f"{prefix}:{mode.value}:best")],
        ]
    )


def _preset_summary(settings: Settings, preset: ScanPreset) -> str:
    config = settings.preset_config(preset)
    extra = "adaptive frames" if config.adaptive_frame_selection else "uniform frames"
    return (
        f"{PRESET_LABELS[preset]}: up to {config.max_video_frames} video frames, "
        f"{config.train_method}, {config.train_max_iterations} iterations, {extra}"
    )


def _help_text(settings: Settings) -> str:
    return (
        "Splatbot turns a short object/scene capture into an interactive 3D Gaussian splat.\n\n"
        "Flow:\n"
        "1. Choose object or scene.\n"
        "2. Choose Fast, Balanced, or Best.\n"
        "3. Upload one video, or upload photos.\n"
        "4. Press Submit scan.\n\n"
        f"Photos: {settings.min_images}-{settings.max_images} images.\n"
        f"Video: up to {settings.max_video_seconds}s.\n"
        f"{_preset_summary(settings, ScanPreset.FAST)}\n"
        f"{_preset_summary(settings, ScanPreset.BALANCED)}\n"
        f"{_preset_summary(settings, ScanPreset.BEST)}\n\n"
        "Commands:\n"
        "/start - open the guided menu\n"
        "/help - show this help text\n"
        "/new - choose scan type\n"
        "/mode scene|object - change current scan type\n"
        "/preset fast|balanced|best - change current speed/quality preset\n"
        "/submit - queue uploaded media\n"
        "/status - show upload/job status\n"
        "/cancel - cancel current upload"
    )


def _upload_hint(count: int, mode: ScanMode, preset: ScanPreset, settings: Settings) -> str:
    return (
        f"Received {count} file(s) for a {mode.value} scan using {PRESET_LABELS[preset]}.\n\n"
        "When upload is complete, press Submit scan.\n"
        f"Use one video or {settings.min_images}-{settings.max_images} photos."
    )


def _media_file_size(update: Update) -> int | None:
    message = update.effective_message
    if message is None:
        return None
    if message.video:
        return message.video.file_size
    if message.document:
        return message.document.file_size
    if message.photo:
        return message.photo[-1].file_size
    return None


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
        "What are you scanning?",
        reply_markup=_main_keyboard(),
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update, context):
        return
    await update.effective_message.reply_text(_help_text(_settings(context)), reply_markup=_main_keyboard())


async def new_session(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update, context):
        return
    await update.effective_message.reply_text(
        "What are you scanning?",
        reply_markup=InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("Object", callback_data="new:object")],
                [InlineKeyboardButton("Scene", callback_data="new:scene")],
            ]
        ),
    )


async def create_session(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    mode: ScanMode,
    preset: ScanPreset,
) -> None:
    settings = _settings(context)
    store = _store(context)
    user_id = update.effective_user.id
    existing = await store.get_active_session(user_id)
    if existing:
        await store.set_session_status(existing.id, JobStatus.CANCELLED)
        shutil.rmtree(settings.data_dir / "sessions" / existing.id, ignore_errors=True)
    session = await store.create_session(user_id, mode, preset)
    session_dir = settings.data_dir / "sessions" / session.id
    session_dir.mkdir(parents=True, exist_ok=True)
    await update.effective_message.reply_text(
        f"New {mode.value} scan started.\n"
        f"{_preset_summary(settings, preset)}\n\n"
        "Upload one video, or upload photos. I will remind you to submit after media arrives.",
        reply_markup=_session_keyboard(),
    )


async def ask_preset(update: Update, context: ContextTypes.DEFAULT_TYPE, mode: ScanMode) -> None:
    settings = _settings(context)
    await update.effective_message.reply_text(
        f"Choose speed/quality preset for this {mode.value} scan.\n\n"
        f"{_preset_summary(settings, ScanPreset.FAST)}\n"
        f"{_preset_summary(settings, ScanPreset.BALANCED)}\n"
        f"{_preset_summary(settings, ScanPreset.BEST)}",
        reply_markup=_preset_keyboard(mode),
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
        session = await store.create_session(
            update.effective_user.id,
            ScanMode(context.args[0]),
            _settings(context).default_scan_preset,
        )
    else:
        await store.set_session_mode(session.id, ScanMode(context.args[0]))
    await update.effective_message.reply_text(f"Mode set to {context.args[0]}.", reply_markup=_session_keyboard())


async def set_preset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update, context):
        return
    if not context.args or context.args[0] not in {preset.value for preset in ScanPreset}:
        await update.effective_message.reply_text("Use /preset fast, /preset balanced, or /preset best.")
        return
    store = _store(context)
    session = await store.get_active_session(update.effective_user.id)
    if session is None:
        await update.effective_message.reply_text("No active scan. Send /new first.")
        return
    preset = ScanPreset(context.args[0])
    await store.set_session_preset(session.id, preset)
    await update.effective_message.reply_text(
        f"Preset set to {PRESET_LABELS[preset]}.\n{_preset_summary(_settings(context), preset)}",
        reply_markup=_session_keyboard(),
    )


async def handle_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update, context):
        return
    query = update.callback_query
    if query is None:
        return
    await query.answer()
    data = query.data or ""
    if data == "help":
        await query.message.reply_text(_help_text(_settings(context)), reply_markup=_main_keyboard())
    elif data == "status":
        await status(update, context)
    elif data == "submit":
        await submit(update, context)
    elif data == "cancel":
        await cancel(update, context)
    elif data == "preset_menu":
        session = await _store(context).get_active_session(update.effective_user.id)
        if session is None:
            await query.message.reply_text("No active scan. Send /new first.", reply_markup=_main_keyboard())
            return
        await query.message.reply_text(
            "Choose a speed/quality preset.",
            reply_markup=_preset_keyboard(session.mode, prefix="setpreset"),
        )
    elif data.startswith("new:"):
        _, mode_value = data.split(":", 1)
        if mode_value not in {ScanMode.SCENE.value, ScanMode.OBJECT.value}:
            await query.message.reply_text("Unknown scan type.", reply_markup=_main_keyboard())
            return
        await ask_preset(update, context, ScanMode(mode_value))
    elif data.startswith("preset:"):
        _, mode_value, preset_value = data.split(":", 2)
        if mode_value not in {mode.value for mode in ScanMode} or preset_value not in {preset.value for preset in ScanPreset}:
            await query.message.reply_text("Unknown scan settings.", reply_markup=_main_keyboard())
            return
        await create_session(update, context, ScanMode(mode_value), ScanPreset(preset_value))
    elif data.startswith("setpreset:"):
        _, _mode_value, preset_value = data.split(":", 2)
        if preset_value not in {preset.value for preset in ScanPreset}:
            await query.message.reply_text("Unknown preset.", reply_markup=_session_keyboard())
            return
        session = await _store(context).get_active_session(update.effective_user.id)
        if session is None:
            await query.message.reply_text("No active scan. Send /new first.", reply_markup=_main_keyboard())
            return
        preset = ScanPreset(preset_value)
        await _store(context).set_session_preset(session.id, preset)
        await query.message.reply_text(
            f"Preset set to {PRESET_LABELS[preset]}.\n{_preset_summary(_settings(context), preset)}",
            reply_markup=_session_keyboard(),
        )


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
        await update.effective_message.reply_text(
            "Choose object or scene first, then resend the media.",
            reply_markup=_main_keyboard(),
        )
        return
    session_dir = settings.data_dir / "sessions" / session.id
    session_dir.mkdir(parents=True, exist_ok=True)

    await update.effective_chat.send_action(ChatAction.UPLOAD_DOCUMENT)
    file_size = _media_file_size(update)
    if file_size is not None and file_size > settings.max_upload_bytes:
        await update.effective_message.reply_text(
            f"File is too large. Max upload size is {settings.max_upload_bytes // (1024 * 1024)} MB.",
            reply_markup=_session_keyboard(),
        )
        return
    if update.effective_message.video:
        tg_file = await update.effective_message.video.get_file()
        suffix = Path(tg_file.file_path or "upload.mp4").suffix or ".mp4"
        kind = classify_path(Path(f"upload{suffix.lower()}"))
    elif update.effective_message.document:
        doc = update.effective_message.document
        tg_file = await doc.get_file()
        suffix = Path(doc.file_name or tg_file.file_path or "upload.bin").suffix or ".bin"
        try:
            kind = classify_path(Path(f"upload{suffix.lower()}"))
        except MediaValidationError as exc:
            await update.effective_message.reply_text(str(exc), reply_markup=_session_keyboard())
            return
    elif update.effective_message.photo:
        tg_file = await update.effective_message.photo[-1].get_file()
        suffix = ".jpg"
        kind = MediaKind.PHOTO
    else:
        return

    existing_items = await store.list_media(session.id)
    existing_photos = [item for item in existing_items if item.kind == MediaKind.PHOTO]
    existing_videos = [item for item in existing_items if item.kind == MediaKind.VIDEO]
    if kind == MediaKind.VIDEO and existing_items:
        await update.effective_message.reply_text(
            "Use either one video or photos, not both. Send /cancel to start over.",
            reply_markup=_session_keyboard(),
        )
        return
    if kind == MediaKind.PHOTO and existing_videos:
        await update.effective_message.reply_text(
            "This scan already has a video. Send /cancel to start over with photos.",
            reply_markup=_session_keyboard(),
        )
        return
    if kind == MediaKind.PHOTO and len(existing_photos) >= settings.max_images:
        await update.effective_message.reply_text(
            f"Already received the max {settings.max_images} photos. Press Submit scan.",
            reply_markup=_session_keyboard(),
        )
        return

    dest = session_dir / f"{tg_file.file_unique_id}{suffix.lower()}"
    if any(Path(item.local_path) == dest for item in existing_items):
        await update.effective_message.reply_text("Already received that file.", reply_markup=_session_keyboard())
        return
    await tg_file.download_to_drive(custom_path=dest)
    await store.add_media(session.id, kind, dest)
    items = await store.list_media(session.id)
    count = len(items)
    if count == 1 or count % 10 == 0 or kind == MediaKind.VIDEO:
        await update.effective_message.reply_text(
            _upload_hint(count, session.mode, session.preset, settings),
            reply_markup=_session_keyboard(),
        )


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
        await update.effective_message.reply_text(str(exc), reply_markup=_session_keyboard())
        return
    job = await store.create_job(session)
    preset_config = settings.preset_config(job.preset)
    await update.effective_message.reply_text(
        f"Queued job {job.id}.\n"
        f"Mode: {job.mode.value}. Preset: {PRESET_LABELS[job.preset]} "
        f"({preset_config.max_video_frames} frames, {preset_config.train_max_iterations} iterations).\n\n"
        "I will send the viewer link when it is ready. Use /status for updates."
    )


def _artifact_label(artifact: JobArtifact) -> str:
    location = artifact.url or artifact.local_path
    return f"{artifact.kind.value}: {location}"


def _job_status_text(job: ScanJob, artifacts: list[JobArtifact]) -> str:
    lines = [f"Job {job.id}: {job.status.value}, mode={job.mode.value}, preset={job.preset.value}."]
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
            f"Collecting {len(items)} file(s), mode={session.mode.value}, preset={session.preset.value}.\n\n"
            "When upload is complete, press Submit scan.",
            reply_markup=_session_keyboard(),
        )
        return
    job = await store.get_latest_job_for_user(user_id)
    if job:
        artifacts = await store.list_artifacts(job.id)
        await update.effective_message.reply_text(_job_status_text(job, artifacts))
        return
    await update.effective_message.reply_text("No active upload session or recent job.", reply_markup=_main_keyboard())


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update, context):
        return
    store = _store(context)
    session = await store.get_active_session(update.effective_user.id)
    if session is None:
        await update.effective_message.reply_text("No active scan.")
        return
    await store.set_session_status(session.id, JobStatus.CANCELLED)
    shutil.rmtree(_settings(context).data_dir / "sessions" / session.id, ignore_errors=True)
    await update.effective_message.reply_text("Cancelled current scan.", reply_markup=_main_keyboard())


async def amain() -> None:
    configure_logging()
    settings = Settings()
    settings.require_telegram()
    if settings.telegram_mode != TelegramMode.POLLING:
        raise ValueError("Only SPLATBOT_TELEGRAM_MODE=polling is currently implemented")
    if not settings.allowed_telegram_ids and not settings.allow_all_telegram_users:
        raise ValueError("SPLATBOT_ALLOWED_TELEGRAM_IDS is required unless SPLATBOT_ALLOW_ALL_TELEGRAM_USERS=true")
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    store = Store(settings.database_path)
    await store.init()

    app = Application.builder().token(settings.telegram_token_value).build()
    app.bot_data["settings"] = settings
    app.bot_data["store"] = store
    await app.bot.set_my_commands(
        [
            BotCommand("start", "Open guided menu"),
            BotCommand("help", "Show commands and capture tips"),
            BotCommand("new", "Start a new scan"),
            BotCommand("mode", "Set scan type: scene or object"),
            BotCommand("preset", "Set preset: fast, balanced, or best"),
            BotCommand("submit", "Submit uploaded media"),
            BotCommand("status", "Show current status"),
            BotCommand("cancel", "Cancel current upload"),
        ]
    )
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("new", new_session))
    app.add_handler(CommandHandler("mode", set_mode))
    app.add_handler(CommandHandler("preset", set_preset))
    app.add_handler(CommandHandler("submit", submit))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("cancel", cancel))
    app.add_handler(CallbackQueryHandler(handle_button))
    app.add_handler(MessageHandler(filters.PHOTO | filters.VIDEO | filters.Document.ALL, receive_media))
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            pass
    await app.initialize()
    await app.start()
    await app.updater.start_polling()
    try:
        await stop_event.wait()
    finally:
        await app.updater.stop()
        await app.stop()
        await app.shutdown()


def main() -> None:
    asyncio.run(amain())
