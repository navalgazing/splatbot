import json
from pathlib import Path

from splatbot.artifact_manifest import build_artifact_manifest, copy_artifact_bundle
from splatbot.jobctl import prune_heavy_artifacts
from splatbot.run_job_dir import copy_failure_artifacts, copy_success_artifacts


def write_file(path: Path, data: bytes = b"data") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_copy_artifact_bundle_preserves_private_debug_inputs(tmp_path) -> None:
    job_dir = tmp_path / "job1"
    output_dir = tmp_path / "results"
    write_file(job_dir / "source_media" / "00001_scan.mov", b"video")
    write_file(job_dir / "images" / "frame_00001.jpg", b"rgb")
    write_file(job_dir / "object_images" / "frame_00001.png", b"rgba")
    write_file(job_dir / "mask_artifacts" / "sam2_raw" / "frame_00001.png", b"mask")
    write_file(job_dir / "processed" / "transforms.json", b"{}")
    write_file(job_dir / "processed" / "colmap" / "sparse" / "0" / "images.bin", b"poses")
    write_file(job_dir / "processed" / "depth_priors" / "depth_00001.npy", b"depth")
    write_file(job_dir / "nerfstudio" / "run" / "config.yml", b"config")
    write_file(job_dir / "nerfstudio" / "run" / "nerfstudio_models" / "step-000001.ckpt", b"ckpt")

    manifest_path = copy_artifact_bundle(job_dir, output_dir)

    assert (output_dir / "artifacts" / "source_media" / "00001_scan.mov").exists()
    assert (output_dir / "artifacts" / "frames" / "selected" / "frame_00001.jpg").exists()
    assert (output_dir / "artifacts" / "frames" / "object_images" / "frame_00001.png").exists()
    assert (output_dir / "artifacts" / "masks" / "backend_artifacts" / "sam2_raw" / "frame_00001.png").exists()
    assert (output_dir / "artifacts" / "processed" / "transforms.json").exists()
    assert (output_dir / "artifacts" / "training" / "nerfstudio" / "run" / "config.yml").exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["artifact_count"] >= 8
    assert any(entry["path"] == "artifacts/source_media/00001_scan.mov" for entry in manifest["artifacts"])


def test_success_artifacts_manifest_marks_public_and_private_files(tmp_path) -> None:
    job_dir = tmp_path / "job1"
    output_dir = tmp_path / "results"
    write_file(output_dir / "export" / "cleaned_splat.ply", b"clean")
    write_file(output_dir / "export" / "raw_splat.ply", b"raw")
    write_file(output_dir / "metrics.json", b"{}")
    write_file(job_dir / "source_media" / "00001_scan.mov", b"video")

    copy_success_artifacts(job_dir, output_dir)

    manifest = json.loads((output_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    by_path = {entry["path"]: entry for entry in manifest["artifacts"]}
    assert by_path["export/cleaned_splat.ply"]["public"] is True
    assert by_path["export/raw_splat.ply"]["public"] is False
    assert by_path["metrics.json"]["public"] is True
    assert by_path["artifacts/source_media/00001_scan.mov"]["public"] is False


def test_failure_artifacts_include_partial_debug_bundle(tmp_path) -> None:
    job_dir = tmp_path / "job1"
    output_dir = tmp_path / "results"
    write_file(job_dir / "metrics.json", b'{"stage": "colmap"}')
    write_file(job_dir / "diagnostics" / "pose" / "error.log", b"failed")
    write_file(job_dir / "processed" / "transforms.json", b"{}")

    copy_failure_artifacts(job_dir, output_dir)

    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "diagnostics" / "pose" / "error.log").exists()
    assert (output_dir / "artifacts" / "diagnostics" / "pose" / "error.log").exists()
    assert (output_dir / "artifacts" / "processed" / "transforms.json").exists()
    manifest = build_artifact_manifest(output_dir, job_id="job1")
    assert any(entry["path"] == "artifacts/diagnostics/pose/error.log" for entry in manifest["artifacts"])


def test_prune_heavy_artifacts_keeps_debug_outputs(tmp_path) -> None:
    job_dir = tmp_path / "job1"
    write_file(job_dir / "source_media" / "scan.mov")
    write_file(job_dir / "images" / "frame_00001.jpg")
    write_file(job_dir / "object_images" / "frame_00001.png")
    write_file(job_dir / "mask_artifacts" / "sam2_raw" / "frame_00001.png")
    write_file(job_dir / "nerfstudio" / "run" / "nerfstudio_models" / "step.ckpt")
    write_file(job_dir / "artifacts" / "source_media" / "scan.mov")
    write_file(job_dir / "artifacts" / "training" / "nerfstudio" / "run" / "config.yml")
    write_file(job_dir / "metrics.json", b"{}")
    write_file(job_dir / "export" / "cleaned_splat.ply", b"ply")
    write_file(job_dir / "artifacts" / "processed" / "transforms.json", b"{}")

    assert prune_heavy_artifacts(job_dir) is True

    assert not (job_dir / "source_media").exists()
    assert not (job_dir / "images").exists()
    assert not (job_dir / "object_images").exists()
    assert not (job_dir / "mask_artifacts").exists()
    assert not (job_dir / "nerfstudio").exists()
    assert not (job_dir / "artifacts" / "source_media").exists()
    assert not (job_dir / "artifacts" / "training").exists()
    assert (job_dir / "metrics.json").exists()
    assert (job_dir / "export" / "cleaned_splat.ply").exists()
    assert (job_dir / "artifacts" / "processed" / "transforms.json").exists()
