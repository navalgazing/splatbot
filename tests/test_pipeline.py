import os
from datetime import UTC, datetime
from pathlib import Path

from splatbot.commands import CommandResult
from splatbot.config import ScanMode, Settings
from splatbot.models import JobStatus, MediaItem, MediaKind
from splatbot.pipeline import ScanPipeline, clean_ply, latest_nerfstudio_config


class FakeRunner:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def run(self, argv: list[str], cwd: Path | None = None) -> CommandResult:
        self.calls.append(argv)
        if argv[0] == "ns-train":
            output_dir = Path(argv[argv.index("--output-dir") + 1])
            config_dir = output_dir / "processed" / "splatfacto" / "2026-04-25_120000"
            config_dir.mkdir(parents=True, exist_ok=True)
            (config_dir / "config.yml").write_text("fake: true\n", encoding="utf-8")
        if argv[0] == "ns-export":
            output_dir = Path(argv[argv.index("--output-dir") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "raw_splat.ply").write_text(
                "\n".join(
                    [
                        "ply",
                        "format ascii 1.0",
                        "element vertex 2",
                        "property float x",
                        "property float y",
                        "property float opacity",
                        "end_header",
                        "0 1 0.5",
                        "nan 2 0.1",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
        return CommandResult(argv=argv, returncode=0, stdout="", stderr="")


def media(path: Path) -> MediaItem:
    return MediaItem(
        id=path.name,
        session_id="s",
        kind=MediaKind.PHOTO,
        local_path=str(path),
        remote_key=None,
        created_at=datetime.now(UTC),
    )


def test_clean_ply_preserves_header_and_removes_invalid_rows(tmp_path) -> None:
    src = tmp_path / "raw.ply"
    dest = tmp_path / "clean.ply"
    src.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 3",
                "property float x",
                "property float f_dc_0",
                "end_header",
                "1 2",
                "nan 2",
                "3 inf",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    clean_ply(src, dest)

    assert dest.read_text(encoding="utf-8").splitlines() == [
        "ply",
        "format ascii 1.0",
        "element vertex 1",
        "property float x",
        "property float f_dc_0",
        "end_header",
        "1 2",
    ]


async def test_pipeline_builds_expected_commands(tmp_path) -> None:
    image = tmp_path / "input.jpg"
    image.write_text("fake", encoding="utf-8")
    settings = Settings(data_dir=tmp_path, max_images=300)
    runner = FakeRunner()
    pipeline = ScanPipeline(settings, runner=runner)

    statuses: list[JobStatus] = []

    async def record_status(job_id: str, status: JobStatus) -> None:
        assert job_id == "job1"
        statuses.append(status)

    outputs = await pipeline.run("job1", ScanMode.SCENE, [media(image)], record_status)

    assert outputs.cleaned_ply == tmp_path / "jobs" / "job1" / "export" / "cleaned_splat.ply"
    assert outputs.preview_mp4 == tmp_path / "jobs" / "job1" / "renders" / "turntable.mp4"
    assert runner.calls[0] == [
        "ns-process-data",
        "images",
        "--data",
        str(tmp_path / "jobs" / "job1" / "images"),
        "--output-dir",
        str(tmp_path / "jobs" / "job1" / "processed"),
        "--no-gpu",
    ]
    assert runner.calls[1] == [
        "ns-train",
        "splatfacto",
        "--data",
        str(tmp_path / "jobs" / "job1" / "processed"),
        "--output-dir",
        str(tmp_path / "jobs" / "job1" / "nerfstudio"),
        "--max-num-iterations",
        "10000",
        "--steps-per-save",
        "10000",
        "--viewer.quit-on-train-completion",
        "True",
    ]
    assert runner.calls[2] == [
        "ns-export",
        "gaussian-splat",
        "--load-config",
        str(tmp_path / "jobs" / "job1" / "nerfstudio" / "processed" / "splatfacto" / "2026-04-25_120000" / "config.yml"),
        "--output-dir",
        str(tmp_path / "jobs" / "job1" / "export"),
        "--output-filename",
        "raw_splat.ply",
    ]
    assert runner.calls[3] == [
        "ns-render",
        "spiral",
        "--load-config",
        str(tmp_path / "jobs" / "job1" / "nerfstudio" / "processed" / "splatfacto" / "2026-04-25_120000" / "config.yml"),
        "--output-path",
        str(tmp_path / "jobs" / "job1" / "renders" / "turntable.mp4"),
        "--seconds",
        "3",
        "--frame-rate",
        "24",
    ]
    assert [call[0] for call in runner.calls] == ["ns-process-data", "ns-train", "ns-export", "ns-render"]
    assert statuses == [JobStatus.COLMAP, JobStatus.TRAINING, JobStatus.EXPORTING, JobStatus.RENDERING]


def test_latest_nerfstudio_config_selects_newest(tmp_path) -> None:
    old = tmp_path / "old" / "config.yml"
    new = tmp_path / "new" / "config.yml"
    old.parent.mkdir(parents=True)
    new.parent.mkdir(parents=True)
    old.write_text("old\n", encoding="utf-8")
    new.write_text("new\n", encoding="utf-8")
    os.utime(old, (1, 1))
    os.utime(new, (2, 2))

    assert latest_nerfstudio_config(tmp_path) == new
