import json
from datetime import UTC, datetime

from telegram.error import RetryAfter

from splatbot.models import ArtifactKind, JobArtifact, JobStatus, ScanJob, ScanMode
from splatbot.notifications import TELEGRAM_MESSAGE_LIMIT, _fit_message, format_failure_summary, format_quality_summary, send_message_with_retry


def test_format_quality_summary_includes_pipeline_events(tmp_path) -> None:
    report = {
        "warnings": ["pipeline_fallbacks_or_skips"],
        "issues": [],
        "event_summary": [
            "pose/vggt-colmap: fallback (backend_failed)",
            "training/3dgs-mcmc: fallback (backend_failed)",
        ],
    }

    summary = format_quality_summary(report)

    assert "Pipeline completed with fallbacks/warnings" in summary
    assert "pose/vggt-colmap" in summary
    assert "training/3dgs-mcmc" in summary


def test_format_failure_summary_reports_last_stage_and_events(tmp_path) -> None:
    metrics = tmp_path / "metrics.json"
    metrics.write_text(
        json.dumps(
            {
                "stages": {"preprocessing": {"duration_seconds": 1.0}, "colmap": {"duration_seconds": 2.0}},
                "pipeline_events": [
                    {
                        "stage": "training",
                        "backend": "3dgs-mcmc",
                        "status": "fallback",
                        "reason": "backend_failed",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = format_failure_summary(metrics)

    assert "Last completed stage: colmap" in summary
    assert "training/3dgs-mcmc" in summary


def test_quality_report_artifact_shape_is_compatible(tmp_path) -> None:
    report_path = tmp_path / "quality_report.json"
    report_path.write_text('{"warnings": [], "issues": [], "event_summary": []}\n', encoding="utf-8")
    job = ScanJob(
        id="job",
        session_id="session",
        telegram_user_id=1,
        mode=ScanMode.SCENE,
        status=JobStatus.DONE,
        error=None,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    artifact = JobArtifact(
        id="artifact",
        job_id=job.id,
        kind=ArtifactKind.QUALITY_REPORT,
        local_path=str(report_path),
        remote_key=None,
        url=None,
        created_at=datetime.now(UTC),
    )

    assert artifact.local_path == str(report_path)


def test_fit_message_caps_to_telegram_limit() -> None:
    message = _fit_message("x" * (TELEGRAM_MESSAGE_LIMIT + 50))

    assert len(message) == TELEGRAM_MESSAGE_LIMIT
    assert message.endswith("...")


async def test_send_message_with_retry_handles_rate_limit(monkeypatch) -> None:
    sleeps: list[int] = []

    async def fake_sleep(seconds: int) -> None:
        sleeps.append(seconds)

    class FakeBot:
        def __init__(self) -> None:
            self.calls: list[tuple[int, str]] = []

        async def send_message(self, *, chat_id: int, text: str) -> None:
            self.calls.append((chat_id, text))
            if len(self.calls) == 1:
                raise RetryAfter(2)

    bot = FakeBot()
    monkeypatch.setattr("splatbot.notifications.asyncio.sleep", fake_sleep)

    await send_message_with_retry(bot, chat_id=123, text="hello")

    assert sleeps == [2]
    assert bot.calls == [(123, "hello"), (123, "hello")]
