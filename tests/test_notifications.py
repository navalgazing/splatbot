import json
from datetime import UTC, datetime

from splatbot.models import ArtifactKind, JobArtifact, JobStatus, ScanJob, ScanMode
from splatbot.notifications import format_failure_summary, format_quality_summary


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
