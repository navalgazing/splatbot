from datetime import UTC, datetime
from pathlib import Path

from splatbot.commands import CommandResult
from splatbot.config import ScanMode, Settings
from splatbot.models import JobStatus, MediaItem, MediaKind
from splatbot.pipeline import ScanPipeline, clean_ply


class FakeRunner:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def run(self, argv: list[str], cwd: Path | None = None) -> CommandResult:
        self.calls.append(argv)
        if argv[0] == "ns-export":
            output_dir = Path(argv[-1])
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
    assert [call[0] for call in runner.calls] == ["ns-process-data", "ns-train", "ns-export", "ns-render"]
    assert statuses == [JobStatus.COLMAP, JobStatus.TRAINING, JobStatus.EXPORTING, JobStatus.RENDERING]
