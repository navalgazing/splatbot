import json
import os
import struct
import zlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from splatbot.commands import CommandResult
from splatbot.config import ScanMode, ScanPreset, Settings
from splatbot.models import JobStatus, MediaItem, MediaKind
from splatbot.pipeline import (
    AlphaMask,
    FrameQuality,
    ScanPipeline,
    SilhouetteFrame,
    apply_object_mask_qa,
    clean_depth_consistency_outliers,
    clean_gaussian_properties,
    clean_exported_ply,
    clean_ply,
    clean_spatial_outliers,
    clear_colmap_sparse_points,
    format_fps,
    build_quality_report,
    inspect_processed_dataset,
    latest_nerfstudio_config,
    object_mask_command_for_backend,
    parse_export_metrics,
    parse_ffprobe_duration,
    parse_ffprobe_frame_rate,
    parse_int_list,
    quality_diverse_sample,
    quality_aware_sample,
    replace_processed_images_with_object_images,
    select_video_frames,
    configured_depth_backends,
    configured_pose_backends,
    configured_segmentation_backends,
    configured_train_backends,
    validate_colmap_quality,
    validate_postprocess_against_masks,
    validate_ply_quality,
    write_processed_training_masks,
)


class FakeRunner:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def run(self, argv: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> CommandResult:
        self.calls.append(argv)
        if argv[0] == "ffprobe":
            return CommandResult(argv=argv, returncode=0, stdout="21.0\n", stderr="")
        if argv[0] == "ffmpeg":
            pattern = Path(argv[-1])
            pattern.parent.mkdir(parents=True, exist_ok=True)
            for idx in range(1, 4):
                (pattern.parent / f"frame_{idx:05d}.jpg").write_bytes(b"frame")
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
                        "property float z",
                        "property float opacity",
                        "end_header",
                        "0 1 2 0.5",
                        "nan 2 3 0.1",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
        return CommandResult(argv=argv, returncode=0, stdout="", stderr="")


class TrainQualityRetryRunner:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.train_calls = 0
        self.export_calls = 0

    async def run(self, argv: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> CommandResult:
        self.calls.append(argv)
        if argv[0] == "ns-train":
            self.train_calls += 1
            output_dir = Path(argv[argv.index("--output-dir") + 1])
            config_dir = output_dir / f"attempt_{self.train_calls}" / "run"
            config_dir.mkdir(parents=True, exist_ok=True)
            (config_dir / "config.yml").write_text("fake: true\n", encoding="utf-8")
        if argv[0] == "ns-export":
            self.export_calls += 1
            output_dir = Path(argv[argv.index("--output-dir") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            vertices = 5 if self.export_calls == 1 else 20
            rows = [
                f"{idx % 5}.0 {(idx // 5) % 4}.0 {(idx // 10) % 2}.0 0.5"
                for idx in range(vertices)
            ]
            (output_dir / "raw_splat.ply").write_text(
                "\n".join(
                    [
                        "ply",
                        "format ascii 1.0",
                        f"element vertex {vertices}",
                        "property float x",
                        "property float y",
                        "property float z",
                        "property float opacity",
                        "end_header",
                        *rows,
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            if self.export_calls == 1:
                return CommandResult(argv=argv, returncode=0, stdout="", stderr="only export 5/1000\n")
            return CommandResult(argv=argv, returncode=0, stdout="", stderr="only export 20/100\n")
        return CommandResult(argv=argv, returncode=0, stdout="", stderr="")


class ColmapFallbackRunner:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.process_calls = 0

    async def run(self, argv: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> CommandResult:
        self.calls.append(argv)
        if argv[0] == "ns-process-data":
            self.process_calls += 1
            output_dir = Path(argv[argv.index("--output-dir") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "images").mkdir(parents=True, exist_ok=True)
            (output_dir / "transforms.json").write_text(
                '{"frames": [{"file_path": "images/frame_00001.jpg"}]}\n',
                encoding="utf-8",
            )
            sparse = output_dir / "colmap" / "sparse" / "0"
            sparse.mkdir(parents=True, exist_ok=True)
            registered = 2 if self.process_calls == 1 else 100
            (sparse / "images.bin").write_text(f"images={registered}", encoding="utf-8")
            (sparse / "points3D.bin").write_bytes(b"background-points")
        return CommandResult(argv=argv, returncode=0, stdout="", stderr="")


class ColmapRetryRunner:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.process_calls = 0

    async def run(self, argv: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> CommandResult:
        self.calls.append(argv)
        if argv[0] == "ns-process-data":
            self.process_calls += 1
            output_dir = Path(argv[argv.index("--output-dir") + 1])
            data_dir = Path(argv[argv.index("--data") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            frames = [
                {"file_path": f"images/{path.name}"}
                for path in sorted(data_dir.iterdir())
                if path.is_file()
            ]
            (output_dir / "images").mkdir(parents=True, exist_ok=True)
            (output_dir / "transforms.json").write_text(
                json.dumps({"frames": frames}) + "\n",
                encoding="utf-8",
            )
            sparse = output_dir / "colmap" / "sparse" / "0"
            sparse.mkdir(parents=True, exist_ok=True)
            registered = 2 if self.process_calls == 1 else len(frames)
            (sparse / "images.bin").write_text(f"images={registered}", encoding="utf-8")
        return CommandResult(argv=argv, returncode=0, stdout="", stderr="")


class SegmentationFallbackRunner:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def run(self, argv: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> CommandResult:
        self.calls.append(argv)
        output_dir = Path(argv[argv.index("--output") + 1]) if "--output" in argv else Path(argv[-1])
        output_dir.mkdir(parents=True, exist_ok=True)
        backend = argv[argv.index("--backend") + 1] if "--backend" in argv else "rembg"
        count = 1 if backend == "sam2" else 3
        valid_alpha = [0] * 100
        for pixel in (44, 45, 54, 55):
            valid_alpha[pixel] = 255
        for idx in range(count):
            write_rgba_png(output_dir / f"frame_{idx + 1:05d}.png", 10, 10, valid_alpha)
        return CommandResult(argv=argv, returncode=0, stdout="", stderr="")


class SuccessfulColmapRunner:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def run(self, argv: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> CommandResult:
        self.calls.append(argv)
        if argv[0] == "ns-process-data":
            output_dir = Path(argv[argv.index("--output-dir") + 1])
            data_dir = Path(argv[argv.index("--data") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            frames = [
                {"file_path": f"images/{path.name}"}
                for path in sorted(data_dir.iterdir())
                if path.is_file()
            ]
            (output_dir / "images").mkdir(parents=True, exist_ok=True)
            (output_dir / "transforms.json").write_text(
                json.dumps({"frames": frames}) + "\n",
                encoding="utf-8",
            )
            sparse = output_dir / "colmap" / "sparse" / "0"
            sparse.mkdir(parents=True, exist_ok=True)
            (sparse / "images.bin").write_text(f"images={len(frames)}", encoding="utf-8")
        return CommandResult(argv=argv, returncode=0, stdout="", stderr="")


class TransformOnlyThenColmapRunner:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def run(self, argv: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> CommandResult:
        self.calls.append(argv)
        if argv[0] in {"splatbot-pose", "splatbot-da3"}:
            output_flag = "--output" if "--output" in argv else "--processed"
            input_flag = "--input" if "--input" in argv else "--images"
            output_dir = Path(argv[argv.index(output_flag) + 1])
            input_dir = Path(argv[argv.index(input_flag) + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            frames = [
                {"file_path": f"images/{path.name}", "transform_matrix": [[1, 0, 0, 0]] * 4}
                for path in sorted(input_dir.iterdir())
                if path.is_file()
            ]
            (output_dir / "transforms.json").write_text(
                json.dumps({"frames": frames}) + "\n",
                encoding="utf-8",
            )
        if argv[0] == "ns-process-data":
            output_dir = Path(argv[argv.index("--output-dir") + 1])
            data_dir = Path(argv[argv.index("--data") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            frames = [
                {"file_path": f"images/{path.name}"}
                for path in sorted(data_dir.iterdir())
                if path.is_file()
            ]
            (output_dir / "images").mkdir(parents=True, exist_ok=True)
            (output_dir / "transforms.json").write_text(
                json.dumps({"frames": frames}) + "\n",
                encoding="utf-8",
            )
            sparse = output_dir / "colmap" / "sparse" / "0"
            sparse.mkdir(parents=True, exist_ok=True)
            (sparse / "images.bin").write_text(f"images={len(frames)}", encoding="utf-8")
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


def write_rgba_png(path: Path, width: int, height: int, alpha: list[int]) -> None:
    def chunk(kind: bytes, data: bytes) -> bytes:
        checksum = zlib.crc32(kind + data) & 0xFFFFFFFF
        return len(data).to_bytes(4, "big") + kind + data + checksum.to_bytes(4, "big")

    rows = bytearray()
    for y in range(height):
        rows.append(0)
        for x in range(width):
            rows.extend([255, 255, 255, alpha[(y * width) + x]])
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", width.to_bytes(4, "big") + height.to_bytes(4, "big") + bytes([8, 6, 0, 0, 0]))
        + chunk(b"IDAT", zlib.compress(bytes(rows)))
        + chunk(b"IEND", b"")
    )


def write_binary_xyz_ply(path: Path, points: list[tuple[float, float, float]]) -> None:
    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {len(points)}",
            "property float x",
            "property float y",
            "property float z",
            "end_header",
        ]
    ) + "\n"
    path.write_bytes(header.encode("ascii") + b"".join(struct.pack("<fff", *point) for point in points))


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
    data = b"".join(
        [
            b"ply\n",
            b"format binary_little_endian 1.0\n",
            b"element vertex 1\n",
            b"property float x\n",
            b"property float y\n",
            b"property float z\n",
            b"end_header\n",
            struct.pack("<fff", 1.0, 2.0, 3.0),
        ]
    )
    src.write_bytes(data)

    clean_ply(src, dest)

    assert dest.read_bytes() == data


def test_clean_ply_rejects_malformed_ply(tmp_path) -> None:
    src = tmp_path / "raw.ply"
    dest = tmp_path / "clean.ply"
    src.write_text("not a ply\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid or unparseable PLY"):
        clean_ply(src, dest)


def test_object_silhouette_cleanup_culls_points_outside_alpha_mask(tmp_path) -> None:
    processed = tmp_path / "processed"
    images = processed / "images"
    images.mkdir(parents=True)
    alpha = [0] * 16
    alpha[5] = 255
    write_rgba_png(images / "frame_00001.png", 4, 4, alpha)
    (processed / "transforms.json").write_text(
        json.dumps(
            {
                "fl_x": 1.0,
                "fl_y": 1.0,
                "cx": 1.0,
                "cy": 1.0,
                "frames": [
                    {
                        "file_path": "images/frame_00001.png",
                        "transform_matrix": [
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    src = tmp_path / "raw.ply"
    dest = tmp_path / "clean.ply"
    write_binary_xyz_ply(src, [(0.0, 0.0, -1.0), (1.0, 0.0, -1.0)])

    cleanup = clean_exported_ply(
        src,
        dest,
        processed,
        ScanMode.OBJECT,
        Settings(
            data_dir=tmp_path,
            silhouette_cleanup_min_views=1,
            silhouette_cleanup_max_views=1,
            silhouette_cleanup_padding_px=0,
            silhouette_cleanup_outside_ratio=1.0,
            silhouette_cleanup_max_inside_views=0,
            silhouette_cleanup_max_remove_fraction=0.9,
        ),
    )

    assert cleanup["output_vertices"] == 1
    assert cleanup["silhouette"]["applied"] is True
    assert cleanup["silhouette"]["removed_points"] == 1
    assert b"element vertex 1" in dest.read_bytes().split(b"end_header", 1)[0]


def test_silhouette_cleanup_skips_when_remove_fraction_is_too_high(tmp_path) -> None:
    processed = tmp_path / "processed"
    images = processed / "images"
    images.mkdir(parents=True)
    write_rgba_png(images / "frame_00001.png", 4, 4, [0] * 16)
    (processed / "transforms.json").write_text(
        json.dumps(
            {
                "fl_x": 1.0,
                "fl_y": 1.0,
                "cx": 1.0,
                "cy": 1.0,
                "frames": [
                    {
                        "file_path": "images/frame_00001.png",
                        "transform_matrix": [
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    src = tmp_path / "raw.ply"
    dest = tmp_path / "clean.ply"
    write_binary_xyz_ply(src, [(0.0, 0.0, -1.0), (1.0, 0.0, -1.0)])

    cleanup = clean_exported_ply(
        src,
        dest,
        processed,
        ScanMode.OBJECT,
        Settings(
            data_dir=tmp_path,
            silhouette_cleanup_min_views=1,
            silhouette_cleanup_max_views=1,
            silhouette_cleanup_padding_px=0,
            silhouette_cleanup_outside_ratio=1.0,
            silhouette_cleanup_max_inside_views=0,
            silhouette_cleanup_max_remove_fraction=0.25,
        ),
    )

    assert cleanup["output_vertices"] == 2
    assert cleanup["silhouette"]["applied"] is False
    assert cleanup["silhouette"]["reason"] == "max_remove_fraction_exceeded"
    assert b"element vertex 2" in dest.read_bytes().split(b"end_header", 1)[0]


def test_validate_ply_quality_warns_failed_mask_validation(tmp_path) -> None:
    metrics = {
        "ply": {
            "cleaned": {
                "parseable": True,
                "format": "format binary_little_endian 1.0",
                "vertices": 50_000,
                "has_xyz": True,
                "flat_axis_ratio": 0.25,
            },
            "cleanup": {
                "validation": {
                    "applied": True,
                    "passed": False,
                    "outside_candidate_fraction": 0.01,
                    "low_support_fraction": 0.4,
                }
            },
        }
    }

    assert validate_ply_quality(metrics, Settings(data_dir=tmp_path, min_splat_vertices=1)) is False
    report = build_quality_report(metrics, Settings(data_dir=tmp_path, min_splat_vertices=1))
    assert "postprocess_validation_failed" in report["issues"]
    assert "quality_gate_warning" in report["warnings"]
    assert metrics["pipeline_events"][-1]["reason"] == "object_mask_validation_failed"
    assert metrics["pipeline_events"][-1]["status"] == "warning"
    assert metrics["quality"]["published_with_warnings"] is True


def test_validate_ply_quality_rejects_unparseable_summary(tmp_path) -> None:
    metrics = {"ply": {"cleaned": {"parseable": False}}}

    with pytest.raises(ValueError, match="parseable PLY"):
        validate_ply_quality(metrics, Settings(data_dir=tmp_path, min_splat_vertices=1))


def test_validate_ply_quality_warns_borderline_vertex_count(tmp_path) -> None:
    metrics = {
        "ply": {
            "cleaned": {
                "parseable": True,
                "format": "format binary_little_endian 1.0",
                "vertices": 9_752,
                "has_xyz": True,
                "flat_axis_ratio": 0.25,
            },
            "cleanup": {"validation": {"applied": True, "passed": True}},
        }
    }

    assert validate_ply_quality(metrics, Settings(data_dir=tmp_path)) is False
    report = build_quality_report(metrics, Settings(data_dir=tmp_path))
    assert "low_splat_vertex_count" in report["warnings"]


def test_validate_ply_quality_warns_too_few_vertices(tmp_path) -> None:
    metrics = {
        "ply": {
            "cleaned": {
                "parseable": True,
                "format": "format binary_little_endian 1.0",
                "vertices": 5_000,
                "has_xyz": True,
                "flat_axis_ratio": 0.25,
            }
        }
    }

    assert validate_ply_quality(metrics, Settings(data_dir=tmp_path)) is False
    report = build_quality_report(metrics, Settings(data_dir=tmp_path))
    assert "low_splat_vertex_count" in report["warnings"]
    assert metrics["pipeline_events"][-1]["reason"] == "low_splat_vertex_count"
    assert metrics["pipeline_events"][-1]["status"] == "warning"


def test_validate_ply_quality_warns_low_export_retention(tmp_path) -> None:
    metrics = {
        "ply": {
            "cleaned": {
                "parseable": True,
                "format": "format binary_little_endian 1.0",
                "vertices": 20_000,
                "has_xyz": True,
                "flat_axis_ratio": 0.25,
            },
            "export": {
                "exported_gaussians": 11_612,
                "total_gaussians": 1_000_000,
                "retention_ratio": 0.011612,
            },
        }
    }

    assert validate_ply_quality(metrics, Settings(data_dir=tmp_path)) is False
    report = build_quality_report(metrics, Settings(data_dir=tmp_path))
    assert "low_exported_gaussian_retention" in report["issues"]


def test_parse_export_metrics_reads_nerfstudio_retention_line() -> None:
    metrics = parse_export_metrics(
        "",
        "0 Gaussians have NaN/Inf and 988388 have low opacity, only export 11612/1000000",
    )

    assert metrics == {
        "exported_gaussians": 11612,
        "total_gaussians": 1000000,
        "retention_ratio": 0.011612,
    }


def test_best_preset_enables_sota_backend_chain_by_default(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path)
    best = settings.preset_config("best")

    assert configured_segmentation_backends(settings, best) == ["sam2", "rembg"]
    assert configured_depth_backends(settings, best) == ["da3", "depth-anything-v2-large"]
    assert configured_pose_backends(settings, best) == ["colmap-global", "vggt-colmap", "mast3r-sfm"]
    assert configured_train_backends(settings, best) == ["splatfacto-big", "3dgs-mcmc"]


def test_experimental_backends_are_explicit_opt_in(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path,
        experimental_sam3_enabled=True,
        experimental_dn_splatter_enabled=True,
    )
    best = settings.preset_config("best")

    assert configured_segmentation_backends(settings, best) == ["sam2", "rembg"]
    assert configured_train_backends(settings, best) == ["dn-splatter-big", "splatfacto-big", "3dgs-mcmc"]


def test_sota_object_mask_command_renders_adapter_cli(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path)
    argv = object_mask_command_for_backend(
        settings,
        "sam2",
        tmp_path / "images",
        tmp_path / "object",
    )

    assert argv == [
        "splatbot-segment",
        "--backend",
        "sam2",
        "--input",
        str(tmp_path / "images"),
        "--output",
        str(tmp_path / "object"),
    ]


async def test_segmentation_backend_falls_back_when_outputs_are_sparse(tmp_path) -> None:
    images = tmp_path / "images"
    object_dir = tmp_path / "object"
    images.mkdir()
    for idx in range(3):
        (images / f"frame_{idx + 1:05d}.jpg").write_bytes(b"image")
    settings = Settings(data_dir=tmp_path, segmentation_backend="sam2,rembg", segmentation_min_output_ratio=0.8)
    runner = SegmentationFallbackRunner()
    metrics = {"pipeline_events": []}

    result = await ScanPipeline(settings, runner=runner).remove_backgrounds(
        images,
        object_dir,
        tmp_path,
        settings.preset_config("balanced"),
        metrics,
        tmp_path / "metrics.json",
    )

    assert result["selected"] == "rembg"
    assert result["attempts"][0]["reason"] == "too_few_outputs"
    assert result["attempts"][0]["output_files"] == 1
    assert result["attempts"][1]["output_files"] == 3
    assert metrics["pipeline_events"][0]["stage"] == "segmentation"


def test_balanced_preset_keeps_stable_default_backend_chain(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path)
    balanced = settings.preset_config("balanced")

    assert configured_segmentation_backends(settings, balanced) == ["rembg"]
    assert configured_pose_backends(settings, balanced) == ["colmap"]
    assert configured_train_backends(settings, balanced) == ["splatfacto"]


def test_object_mask_qa_filters_bad_masks(tmp_path) -> None:
    original = tmp_path / "images"
    object_images = tmp_path / "object_images"
    original.mkdir()
    object_images.mkdir()
    for idx in range(3):
        (original / f"frame_{idx + 1:05d}.jpg").write_bytes(b"image")

    valid_alpha = [0] * 100
    for idx in (44, 45, 54, 55):
        valid_alpha[idx] = 255
    write_rgba_png(object_images / "frame_00001.png", 10, 10, valid_alpha)
    write_rgba_png(object_images / "frame_00002.png", 10, 10, [255] * 100)
    write_rgba_png(object_images / "frame_00003.png", 10, 10, valid_alpha)

    result = apply_object_mask_qa(
        original,
        object_images,
        tmp_path,
        Settings(
            data_dir=tmp_path,
            object_mask_min_keep_frames=2,
            object_mask_min_keep_ratio=0.5,
            object_mask_max_area_ratio=0.5,
            object_mask_max_edge_touch_ratio=1.0,
        ),
        Settings(data_dir=tmp_path).preset_config("fast"),
    )

    assert result["applied"] is True
    assert result["rejected_by_reason"]["too_large"] == 1
    assert sorted(path.name for path in Path(result["object_images_dir"]).iterdir()) == [
        "frame_00001.png",
        "frame_00003.png",
    ]
    assert sorted(path.name for path in Path(result["original_images_dir"]).iterdir()) == [
        "frame_00001.jpg",
        "frame_00003.jpg",
    ]


def test_write_processed_training_masks_adds_mask_paths(tmp_path) -> None:
    processed = tmp_path / "processed"
    images = processed / "images"
    images.mkdir(parents=True)
    alpha = [0, 255, 32, 8]
    write_rgba_png(images / "frame_00001.png", 2, 2, alpha)
    (processed / "transforms.json").write_text(
        json.dumps({"frames": [{"file_path": "images/frame_00001.png"}]}) + "\n",
        encoding="utf-8",
    )

    written = write_processed_training_masks(processed, alpha_threshold=16)

    data = json.loads((processed / "transforms.json").read_text(encoding="utf-8"))
    assert written == 1
    assert data["frames"][0]["mask_path"] == "masks/frame_00001.png"
    assert (processed / "masks" / "frame_00001.png").read_bytes().startswith(b"\x89PNG")


def test_gaussian_cleanup_removes_low_quality_gaussians(tmp_path) -> None:
    src = tmp_path / "raw.ply"
    dest = tmp_path / "clean.ply"
    src.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 4",
                "property float x",
                "property float y",
                "property float z",
                "property float opacity",
                "property float scale_0",
                "property float scale_1",
                "property float scale_2",
                "end_header",
                "0 0 0 0 0 0 0",
                "0.1 0 0 0 0 0 0",
                "0.2 0 0 0 0 0 0",
                "5 5 5 0 5 0 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cleanup = clean_gaussian_properties(
        src,
        dest,
        Settings(
            data_dir=tmp_path,
            gaussian_cleanup_max_scale_ratio=2.0,
            gaussian_cleanup_max_anisotropy=2.0,
            gaussian_cleanup_max_remove_fraction=0.5,
        ),
    )

    assert cleanup["applied"] is True
    assert cleanup["filtered_vertices_removed"] == 1
    assert "element vertex 3" in dest.read_text(encoding="utf-8")


def test_spatial_cleanup_removes_isolated_points(tmp_path) -> None:
    src = tmp_path / "raw.ply"
    dest = tmp_path / "clean.ply"
    cluster = [(idx * 0.01, 0.0, 0.0) for idx in range(6)]
    isolated = [(10.0, 10.0, 10.0), (10.1, 10.0, 10.0)]
    write_binary_xyz_ply(src, cluster + isolated)

    cleanup = clean_spatial_outliers(
        src,
        dest,
        Settings(
            data_dir=tmp_path,
            spatial_outlier_radius_fraction=0.02,
            spatial_outlier_min_neighbors=2,
            spatial_component_min_vertices=3,
            spatial_component_min_fraction=0.0,
            spatial_cleanup_max_remove_fraction=0.5,
        ),
    )

    assert cleanup["applied"] is True
    assert cleanup["filtered_vertices_removed"] == 2
    assert b"element vertex 6" in dest.read_bytes().split(b"end_header", 1)[0]


def test_depth_consistency_cleanup_removes_depth_outliers(tmp_path) -> None:
    np = pytest.importorskip("numpy")
    depth_path = tmp_path / "depth.npy"
    np.save(depth_path, np.ones((4, 4), dtype="float32"))
    src = tmp_path / "raw.ply"
    dest = tmp_path / "clean.ply"
    write_binary_xyz_ply(
        src,
        [
            (0.0, 0.0, -1.0),
            (0.1, 0.0, -1.0),
            (0.0, 0.1, -1.0),
            (0.1, 0.1, -1.0),
            (0.0, 0.0, -3.0),
        ],
    )
    frames = [
        SilhouetteFrame(
            mask=AlphaMask(width=4, height=4, alpha=bytes([255] * 16)),
            world_to_camera=[
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            fl_x=1.0,
            fl_y=1.0,
            cx=2.0,
            cy=2.0,
            depth_path=depth_path,
        )
    ]

    cleanup = clean_depth_consistency_outliers(
        src,
        dest,
        frames,
        Settings(
            data_dir=tmp_path,
            depth_consistency_cleanup_enabled=True,
            depth_consistency_cleanup_min_views=1,
            depth_consistency_cleanup_max_depth_ratio=1.5,
            depth_consistency_cleanup_min_inconsistent_ratio=1.0,
            depth_consistency_cleanup_min_scale_samples=1,
            depth_consistency_cleanup_max_remove_fraction=0.5,
        ),
    )

    assert cleanup["applied"] is True
    assert cleanup["filtered_vertices_removed"] == 1
    assert cleanup["depth_views"] == 1
    assert b"element vertex 4" in dest.read_bytes().split(b"end_header", 1)[0]


async def test_pipeline_builds_expected_commands(tmp_path) -> None:
    image = tmp_path / "input.jpg"
    image.write_text("fake", encoding="utf-8")
    settings = Settings(data_dir=tmp_path, max_images=300, min_splat_vertices=1)
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
    assert statuses == [
        JobStatus.PREPROCESSING,
        JobStatus.COLMAP,
        JobStatus.TRAINING,
        JobStatus.EXPORTING,
    ]


async def test_pipeline_uses_video_speedups(tmp_path) -> None:
    video = tmp_path / "scan.mov"
    video.write_text("fake", encoding="utf-8")
    settings = Settings(data_dir=tmp_path, adaptive_frame_selection=False, min_splat_vertices=1)
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


async def test_process_data_uses_colmap_gpu_when_enabled(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path, colmap_use_gpu=True)
    runner = FakeRunner()
    pipeline = ScanPipeline(settings, runner=runner)

    await pipeline.process_data(tmp_path / "images", tmp_path / "processed")

    assert runner.calls == [
        [
            "ns-process-data",
            "images",
            "--data",
            str(tmp_path / "images"),
            "--output-dir",
            str(tmp_path / "processed"),
        ]
    ]


async def test_process_data_can_use_custom_colmap_command(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path, colmap_bin="splatbot-colmap-wrapper")
    runner = FakeRunner()
    pipeline = ScanPipeline(settings, runner=runner)

    await pipeline.process_data(tmp_path / "images", tmp_path / "processed")

    assert runner.calls == [
        [
            "ns-process-data",
            "images",
            "--data",
            str(tmp_path / "images"),
            "--output-dir",
            str(tmp_path / "processed"),
            "--colmap-cmd",
            "splatbot-colmap-wrapper",
            "--no-gpu",
        ]
    ]


async def test_colmap_pose_backend_can_force_exhaustive_matching(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path, pose_backends="colmap-exhaustive")
    runner = SuccessfulColmapRunner()
    pipeline = ScanPipeline(settings, runner=runner)
    images = tmp_path / "images"
    processed = tmp_path / "processed"
    images.mkdir()
    for idx in range(20):
        (images / f"frame_{idx + 1:05d}.jpg").write_bytes(b"image")
    metrics = {"frames": {"selected": 20}, "colmap": {}, "pipeline_events": []}

    await pipeline.process_data_with_quality_gate(
        input_images_dir=images,
        processed_dir=processed,
        matching_method="sequential",
        metrics=metrics,
        metrics_path=tmp_path / "metrics.json",
        preset=settings.preset_config("balanced"),
        mode=ScanMode.SCENE,
        original_images_dir=images,
        object_images_dir=None,
    )

    assert runner.calls[0][runner.calls[0].index("--matching-method") + 1] == "exhaustive"
    assert metrics["pose_backend"] == "colmap-exhaustive"


async def test_object_colmap_fallback_uses_original_poses_and_object_images(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path)
    runner = ColmapFallbackRunner()
    pipeline = ScanPipeline(settings, runner=runner)
    original = tmp_path / "images"
    object_images = tmp_path / "object_images"
    processed = tmp_path / "processed"
    original.mkdir()
    object_images.mkdir()
    for idx in range(140):
        (original / f"frame_{idx + 1:05d}.jpg").write_bytes(b"original")
        (object_images / f"frame_{idx + 1:05d}.png").write_bytes(b"object")
    metrics = {"frames": {"selected": 140}, "colmap": {}}
    metrics_path = tmp_path / "metrics.json"

    await pipeline.process_data_with_quality_gate(
        input_images_dir=object_images,
        processed_dir=processed,
        matching_method="sequential",
        metrics=metrics,
        metrics_path=metrics_path,
        preset=settings.preset_config("balanced"),
        mode=ScanMode.OBJECT,
        original_images_dir=original,
        object_images_dir=object_images,
    )

    assert [call[call.index("--data") + 1] for call in runner.calls] == [str(object_images), str(original)]
    assert metrics["colmap_masked"]["active_registered_images"] == 2
    assert metrics["colmap"]["active_registered_images"] == 100
    assert metrics["colmap_fallback"]["applied"] is True
    assert metrics["colmap_fallback"]["sparse_points_removed"]["applied"] is True
    assert (processed / "images" / "frame_00001.png").read_bytes() == b"object"
    assert "frame_00001.png" in (processed / "transforms.json").read_text(encoding="utf-8")
    assert (processed / "colmap" / "sparse" / "0" / "points3D.bin").read_bytes() == struct.pack("<Q", 0)
    assert (processed / "colmap" / "sparse" / "0" / "points3D.bin.splatbot-original").read_bytes() == b"background-points"


def test_clear_colmap_sparse_points_replaces_background_seed_cloud(tmp_path) -> None:
    sparse = tmp_path / "processed" / "colmap" / "sparse" / "0"
    sparse.mkdir(parents=True)
    (sparse / "points3D.bin").write_bytes(b"not-empty")
    (sparse / "points3D.txt").write_text("1 0 0 0\n", encoding="utf-8")

    result = clear_colmap_sparse_points(tmp_path / "processed")

    assert result == {"applied": True, "files": 2, "original_bytes": 17}
    assert (sparse / "points3D.bin").read_bytes() == struct.pack("<Q", 0)
    assert (sparse / "points3D.bin.splatbot-original").read_bytes() == b"not-empty"
    assert "removed original-frame sparse points" in (sparse / "points3D.txt").read_text(encoding="utf-8")


async def test_colmap_retry_uses_smaller_subset_when_full_set_fails(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path,
        colmap_retry_frame_counts="120",
        colmap_retry_matching_methods="sequential",
    )
    runner = ColmapRetryRunner()
    pipeline = ScanPipeline(settings, runner=runner)
    images = tmp_path / "images"
    processed = tmp_path / "processed"
    images.mkdir()
    for idx in range(140):
        (images / f"frame_{idx + 1:05d}.jpg").write_bytes(b"image")
    metrics = {"frames": {"selected": 140}, "colmap": {}}
    metrics_path = tmp_path / "metrics.json"

    await pipeline.process_data_with_quality_gate(
        input_images_dir=images,
        processed_dir=processed,
        matching_method="sequential",
        metrics=metrics,
        metrics_path=metrics_path,
        preset=settings.preset_config("balanced"),
        mode=ScanMode.SCENE,
        original_images_dir=images,
        object_images_dir=None,
    )

    assert len(runner.calls) == 2
    assert runner.calls[0][runner.calls[0].index("--data") + 1] == str(images)
    retry_input = Path(runner.calls[1][runner.calls[1].index("--data") + 1])
    assert retry_input.name == "original_120"
    assert len(list(retry_input.iterdir())) == 120
    assert metrics["frames"]["selected_for_colmap"] == 120
    assert metrics["colmap_recovery"] == {
        "applied": True,
        "source": "original",
        "frame_count": 120,
        "matching_method": "sequential",
    }


def test_parse_ffprobe_duration() -> None:
    assert parse_ffprobe_duration("21.25\n") == 21.25
    assert parse_ffprobe_duration("N/A\n") is None
    assert parse_ffprobe_duration("") is None
    assert parse_ffprobe_duration("-1\n") is None


def test_parse_ffprobe_frame_rate() -> None:
    assert parse_ffprobe_frame_rate("30000/1001\n30/1\n") == 29.97002997002997
    assert parse_ffprobe_frame_rate("0/0\n24/1\n") == 24.0
    assert parse_ffprobe_frame_rate("N/A\n") is None


def test_parse_int_list_ignores_invalid_values() -> None:
    assert parse_int_list("120, 80, nope,60") == [120, 80, 60]


def test_format_fps() -> None:
    assert format_fps(6.6666667) == "6.667"
    assert format_fps(10.0) == "10"


async def test_pipeline_can_render_preview_when_enabled(tmp_path) -> None:
    image = tmp_path / "input.jpg"
    image.write_text("fake", encoding="utf-8")
    settings = Settings(data_dir=tmp_path, render_preview=True, min_splat_vertices=1)
    runner = FakeRunner()

    outputs = await ScanPipeline(settings, runner=runner).run("job1", ScanMode.SCENE, [media(image)])

    assert outputs.preview_mp4 == tmp_path / "jobs" / "job1" / "renders" / "turntable.mp4"
    assert runner.calls[-1][0:2] == ["ns-render", "spiral"]


async def test_external_train_backend_preserves_extra_arg_boundaries(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path,
        best_train_extra_args="--pipeline.model.cull_alpha_thresh=0.005 --note 'two words'",
    )
    runner = FakeRunner()
    pipeline = ScanPipeline(settings, runner=runner)

    await pipeline.train_external_backend(
        "dn-splatter-big",
        tmp_path / "processed",
        tmp_path / "nerfstudio",
        settings.preset_config("best"),
    )

    assert runner.calls == [
        [
            "splatbot-train",
            "--backend",
            "dn-splatter-big",
            "--data",
            str(tmp_path / "processed"),
            "--output",
            str(tmp_path / "nerfstudio"),
            "--max-iterations",
            "30000",
            "--steps-per-save",
            "30000",
            "--pipeline.model.cull_alpha_thresh=0.005",
            "--note",
            "two words",
        ]
    ]


async def test_mcmc_default_train_command_preserves_best_extra_args(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path,
        best_train_extra_args="--pipeline.model.cull-alpha-thresh=0.005 --note 'two words'",
    )
    runner = FakeRunner()
    pipeline = ScanPipeline(settings, runner=runner)

    await pipeline.train_external_backend(
        "3dgs-mcmc",
        tmp_path / "processed",
        tmp_path / "nerfstudio",
        settings.preset_config("best"),
    )

    assert runner.calls == [
        [
            "ns-train",
            "splatfacto-mcmc",
            "--data",
            str(tmp_path / "processed"),
            "--output-dir",
            str(tmp_path / "nerfstudio"),
            "--max-num-iterations",
            "30000",
            "--steps-per-save",
            "30000",
            "--viewer.quit-on-train-completion",
            "True",
            "--pipeline.model.cull-alpha-thresh=0.005",
            "--note",
            "two words",
        ]
    ]


async def test_train_export_publishes_quality_warning_without_retry(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path,
        min_splat_vertices=10,
        best_train_backends="splatfacto-big,3dgs-mcmc",
        best_train_required_backends="",
        best_train_extra_args="",
    )
    runner = TrainQualityRetryRunner()
    pipeline = ScanPipeline(settings, runner=runner)
    processed = tmp_path / "processed"
    processed.mkdir()
    metrics = {"pipeline_events": [], "stages": {}, "ply": {}}

    artifacts = await pipeline.train_export_reconstruction(
        processed_dir=processed,
        ns_dir=tmp_path / "nerfstudio",
        export_dir=tmp_path / "export",
        mode=ScanMode.SCENE,
        preset=settings.preset_config(ScanPreset.BEST),
        metrics=metrics,
        metrics_path=tmp_path / "metrics.json",
        job_id="job1",
        on_status=None,
    )

    assert artifacts.cleaned_ply.exists()
    assert runner.train_calls == 1
    assert runner.export_calls == 1
    assert metrics["train_backend"] == "splatfacto-big"
    assert metrics["train_backend_attempts"][0]["cleaned_vertices"] == 5
    assert metrics["train_backend_attempts"][0]["passed_quality"] is False
    assert metrics["ply"]["cleaned"]["vertices"] == 5
    assert metrics["quality"]["published_with_warnings"] is True
    assert any(event["reason"] == "low_splat_vertex_count" for event in metrics["pipeline_events"])


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


def test_select_video_frames_falls_back_to_even_sampling(tmp_path) -> None:
    candidates = []
    for idx in range(1, 7):
        path = tmp_path / f"candidate_{idx:05d}.jpg"
        path.write_bytes(b"x" * idx)
        candidates.append(path)
    images = tmp_path / "images"
    images.mkdir()

    selected = select_video_frames(
        candidates,
        images,
        target_count=3,
        min_count=3,
        quality_threshold=35.0,
        blur_threshold=20.0,
        low_contrast_threshold=6.0,
        overexposed_threshold=0.55,
        underexposed_threshold=0.55,
        duplicate_threshold=3.0,
    )

    assert selected.selected_count == 3
    assert selected.metrics["candidate_frames"] == 6
    assert selected.metrics["selected_frames"] == 3
    assert [path.name for path in sorted(images.iterdir())] == [
        "frame_00001.jpg",
        "frame_00002.jpg",
        "frame_00003.jpg",
    ]


def test_quality_aware_sample_preserves_coverage_and_picks_best(tmp_path) -> None:
    profiles = [
        FrameQuality(path=tmp_path / f"{idx}.jpg", index=idx, score=score)
        for idx, score in enumerate([10, 90, 20, 80, 30, 70])
    ]

    selected = quality_aware_sample(profiles, 3)

    assert [profile.index for profile in selected] == [1, 3, 5]


def test_quality_diverse_sample_prefers_distinct_frame_signatures(tmp_path) -> None:
    profiles = [
        FrameQuality(path=tmp_path / "0.jpg", index=0, score=95, signature=(0.0, 0.0)),
        FrameQuality(path=tmp_path / "1.jpg", index=1, score=94, signature=(0.1, 0.1)),
        FrameQuality(path=tmp_path / "2.jpg", index=2, score=80, signature=(3.0, 3.0)),
        FrameQuality(path=tmp_path / "3.jpg", index=3, score=79, signature=(3.1, 3.1)),
    ]

    selected = quality_diverse_sample(profiles, 2)

    assert [profile.index for profile in selected] == [0, 2]


def test_replace_processed_images_with_object_images_updates_transforms(tmp_path) -> None:
    processed = tmp_path / "processed"
    object_images = tmp_path / "object_images"
    (processed / "images").mkdir(parents=True)
    object_images.mkdir()
    (processed / "images" / "frame_00001.jpg").write_bytes(b"original")
    (processed / "images" / "frame_00002.jpg").write_bytes(b"original")
    (object_images / "frame_00001.png").write_bytes(b"object1")
    (object_images / "frame_00002.png").write_bytes(b"object2")
    (processed / "transforms.json").write_text(
        """
{
  "frames": [
    {"file_path": "images/frame_00001.jpg"},
    {"file_path": "./images/frame_00002.jpg"}
  ]
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    rewritten = replace_processed_images_with_object_images(processed, object_images)

    assert rewritten == 2
    assert (processed / "images" / "frame_00001.png").read_bytes() == b"object1"
    assert (processed / "images" / "frame_00002.png").read_bytes() == b"object2"
    assert '"images/frame_00001.png"' in (processed / "transforms.json").read_text(encoding="utf-8")
    assert '"images/frame_00002.png"' in (processed / "transforms.json").read_text(encoding="utf-8")


def test_colmap_quality_gate_rejects_low_active_sparse_model(tmp_path) -> None:
    processed = tmp_path / "processed" / "colmap" / "sparse" / "0"
    processed.mkdir(parents=True)
    (processed / "images.bin").write_text("images=2", encoding="utf-8")
    metrics = {
        "frames": {"selected": 140},
        "colmap": inspect_processed_dataset(tmp_path / "processed"),
    }

    try:
        validate_colmap_quality(metrics, 140, Settings())
    except ValueError as exc:
        assert "registered only 2/140" in str(exc)
    else:
        raise AssertionError("expected COLMAP quality gate to reject low registration")


def test_colmap_quality_gate_rejects_unknown_registration(tmp_path) -> None:
    metrics = {"frames": {"selected": 140}, "colmap": {}}

    with pytest.raises(ValueError, match="registration count is unknown"):
        validate_colmap_quality(metrics, 140, Settings())


def test_colmap_quality_gate_rejects_transform_only_pose(tmp_path) -> None:
    metrics = {"frames": {"selected": 140}, "colmap": {"transforms_frames": 140, "models": []}}

    with pytest.raises(ValueError, match="Transform-only poses are not accepted"):
        validate_colmap_quality(metrics, 140, Settings())


def test_colmap_quality_gate_rejects_tiny_sparse_point_cloud(tmp_path) -> None:
    metrics = {
        "frames": {"selected": 100},
        "colmap": {
            "active_registered_images": 100,
            "best_registered_images": 100,
            "active_points3d_count": 12,
            "active_points3d_bytes": 104,
        },
    }

    with pytest.raises(ValueError, match="only 12 3D point"):
        validate_colmap_quality(metrics, 100, Settings(data_dir=tmp_path, min_colmap_sparse_points=1000))


def test_quality_report_flags_transform_only_pose(tmp_path) -> None:
    report = build_quality_report(
        {
            "frames": {"selected": 140},
            "colmap": {"transforms_frames": 140, "models": []},
            "ply": {"cleaned": {"vertices": 20_000}},
        },
        Settings(data_dir=tmp_path),
    )

    assert report["passed"] is False
    assert "pose_missing_sparse_model" in report["issues"]


async def test_transform_only_external_pose_falls_back_to_colmap(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path,
        best_pose_backends="da3-colmap,colmap-global",
        best_pose_required_backends="",
    )
    runner = TransformOnlyThenColmapRunner()
    pipeline = ScanPipeline(settings, runner=runner)
    images = tmp_path / "images"
    processed = tmp_path / "processed"
    images.mkdir()
    for idx in range(20):
        (images / f"frame_{idx + 1:05d}.jpg").write_bytes(b"image")
    metrics = {"frames": {"selected": 20}, "colmap": {}, "pipeline_events": []}

    await pipeline.process_data_with_quality_gate(
        input_images_dir=images,
        processed_dir=processed,
        matching_method="sequential",
        metrics=metrics,
        metrics_path=tmp_path / "metrics.json",
        preset=settings.preset_config("best"),
        mode=ScanMode.SCENE,
        original_images_dir=images,
        object_images_dir=None,
    )

    assert metrics["pose_backend"] == "colmap-global"
    assert metrics["pose_backend_failures"][0]["backend"] == "da3-colmap"
    assert "Transform-only poses are not accepted" in metrics["pose_backend_failures"][0]["error"]
    assert metrics["colmap"]["active_registered_images"] == 20


def test_postprocess_validation_rejects_unobserved_points(tmp_path) -> None:
    ply_path = tmp_path / "splat.ply"
    write_binary_xyz_ply(ply_path, [(0.0, 0.0, 1.0), (1.0, 1.0, 1.0)])
    frames = [
        SilhouetteFrame(
            mask=AlphaMask(width=4, height=4, alpha=bytes([255] * 16)),
            world_to_camera=[
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            fl_x=1.0,
            fl_y=1.0,
            cx=2.0,
            cy=2.0,
        )
    ]

    result = validate_postprocess_against_masks(
        ply_path,
        frames,
        Settings(
            data_dir=tmp_path,
            silhouette_cleanup_min_views=1,
            postprocess_validation_min_checked_points=1,
            postprocess_validation_max_unobserved_fraction=0.5,
        ),
    )

    assert result["checked_points"] == 0
    assert result["unobserved_fraction"] == 1.0
    assert result["passed"] is False


def test_postprocess_validation_rejects_unobserved_points_with_clean_checked_support(tmp_path) -> None:
    ply_path = tmp_path / "splat.ply"
    write_binary_xyz_ply(ply_path, [(0.0, 0.0, -1.0), (0.0, 0.0, 1.0), (1.0, 1.0, 1.0)])
    frames = [
        SilhouetteFrame(
            mask=AlphaMask(width=4, height=4, alpha=bytes([255] * 16)),
            world_to_camera=[
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            fl_x=1.0,
            fl_y=1.0,
            cx=2.0,
            cy=2.0,
        )
    ]
    settings = Settings(
        data_dir=tmp_path,
        silhouette_cleanup_min_views=1,
        postprocess_validation_min_checked_points=1,
        postprocess_validation_max_unobserved_fraction=0.5,
    )

    result = validate_postprocess_against_masks(ply_path, frames, settings)
    report = build_quality_report({"ply": {"cleanup": {"validation": result}}}, settings)

    assert result["checked_points"] == 1
    assert result["unobserved_fraction"] == pytest.approx(0.666667)
    assert result["outside_candidate_fraction"] == 0.0
    assert result["low_support_fraction"] == 0.0
    assert result["passed"] is False
    assert "high_unobserved_splat_fraction" in report["issues"]
