import os
from datetime import UTC, datetime
from pathlib import Path

from splatbot.commands import CommandResult
from splatbot.config import ScanMode, Settings
from splatbot.models import JobStatus, MediaItem, MediaKind
from splatbot.pipeline import (
    ScanPipeline,
    clean_ply,
    format_fps,
    latest_nerfstudio_config,
    parse_ffprobe_duration,
)


class FakeRunner:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def run(self, argv: list[str], cwd: Path | None = None) -> CommandResult:
        self.calls.append(argv)
        if argv[0] == "ffprobe":
            return CommandResult(argv=argv, returncode=0, stdout="21.0\n", stderr="")
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


def media(path: Path, kind: MediaKind = MediaKind.PHOTO) -> MediaItem:
    return MediaItem(
        id=path.name,
        session_id="s",
        kind=kind,
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


def test_clean_ply_leaves_binary_ply_unchanged(tmp_path) -> None:
    src = tmp_path / "raw.ply"
    dest = tmp_path / "clean.ply"
    data = b"ply\nformat binary_little_endian 1.0\nend_header\n\x00\x01\x02"
    src.write_bytes(data)

    clean_ply(src, dest)

    assert dest.read_bytes() == data


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
    assert outputs.preview_mp4 is None
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
    assert [call[0] for call in runner.calls] == ["ns-process-data", "ns-train", "ns-export"]
    assert statuses == [JobStatus.COLMAP, JobStatus.TRAINING, JobStatus.EXPORTING]


async def test_pipeline_uses_video_speedups(tmp_path) -> None:
    video = tmp_path / "scan.mov"
    video.write_text("fake", encoding="utf-8")
    settings = Settings(data_dir=tmp_path)
    runner = FakeRunner()

    await ScanPipeline(settings, runner=runner).run(
        "job1",
        ScanMode.SCENE,
        [media(video, MediaKind.VIDEO)],
    )

    assert runner.calls[0] == [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video),
    ]
    assert runner.calls[1] == [
        "ffmpeg",
        "-i",
        str(video),
        "-t",
        "60",
        "-vf",
        "fps=6.667",
        "-q:v",
        "2",
        str(tmp_path / "jobs" / "job1" / "images" / "frame_%05d.jpg"),
    ]
    assert runner.calls[2] == [
        "ns-process-data",
        "images",
        "--data",
        str(tmp_path / "jobs" / "job1" / "images"),
        "--output-dir",
        str(tmp_path / "jobs" / "job1" / "processed"),
        "--matching-method",
        "sequential",
        "--no-gpu",
    ]


def test_parse_ffprobe_duration() -> None:
    assert parse_ffprobe_duration("21.25\n") == 21.25
    assert parse_ffprobe_duration("N/A\n") is None
    assert parse_ffprobe_duration("") is None
    assert parse_ffprobe_duration("-1\n") is None


def test_format_fps() -> None:
    assert format_fps(6.6666667) == "6.667"
    assert format_fps(10.0) == "10"


async def test_pipeline_can_render_preview_when_enabled(tmp_path) -> None:
    image = tmp_path / "input.jpg"
    image.write_text("fake", encoding="utf-8")
    settings = Settings(data_dir=tmp_path, render_preview=True)
    runner = FakeRunner()

    outputs = await ScanPipeline(settings, runner=runner).run("job1", ScanMode.SCENE, [media(image)])

    assert outputs.preview_mp4 == tmp_path / "jobs" / "job1" / "renders" / "turntable.mp4"
    assert runner.calls[-1][0:2] == ["ns-render", "spiral"]


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
