import json

import pytest

from splatbot.benchmark import build_benchmark_report


def write_job(path, job_id: str, passed: bool, low_support: float) -> None:
    path.mkdir()
    (path / "metrics.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "mode": "object",
                "preset": "best",
                "stages": {"training": {"duration_seconds": 10}},
                "ply": {
                    "cleaned": {"vertices": 1000},
                    "cleanup": {
                        "validation": {
                            "applied": True,
                            "passed": passed,
                            "outside_candidate_fraction": 0.0,
                            "low_support_fraction": low_support,
                        }
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "quality_report.json").write_text(
        json.dumps({"passed": passed, "issues": [] if passed else ["postprocess_validation_failed"]})
        + "\n",
        encoding="utf-8",
    )


def test_benchmark_report_ranks_passing_outputs_first(tmp_path) -> None:
    bad = tmp_path / "bad"
    good = tmp_path / "good"
    write_job(bad, "bad", False, 0.5)
    write_job(good, "good", True, 0.0)

    report = build_benchmark_report([bad, good])

    assert report["best_job_id"] == "good"
    assert report["jobs"][0]["passed"] is True
    assert report["jobs"][1]["issues"] == ["postprocess_validation_failed"]


def test_benchmark_report_rejects_missing_job_dirs(tmp_path) -> None:
    with pytest.raises(ValueError, match="job dir does not exist"):
        build_benchmark_report([tmp_path / "missing"])
