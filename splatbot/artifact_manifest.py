from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


MANIFEST_NAME = "artifact_manifest.json"
ARTIFACT_DIR_NAME = "artifacts"
HASH_LIMIT_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class ArtifactCopyRule:
    source: str
    destination: str


ARTIFACT_COPY_RULES: tuple[ArtifactCopyRule, ...] = (
    ArtifactCopyRule("source_media", "source_media"),
    ArtifactCopyRule("candidate_frames", "frames/candidates"),
    ArtifactCopyRule("images", "frames/selected"),
    ArtifactCopyRule("object_images", "frames/object_images"),
    ArtifactCopyRule("mask_artifacts", "masks/backend_artifacts"),
    ArtifactCopyRule("processed/transforms.json", "processed/transforms.json"),
    ArtifactCopyRule("processed/sparse_pc.ply", "processed/sparse_pc.ply"),
    ArtifactCopyRule("processed/colmap/database.db", "processed/colmap/database.db"),
    ArtifactCopyRule("processed/colmap/sparse", "processed/colmap/sparse"),
    ArtifactCopyRule("processed/masks", "processed/masks"),
    ArtifactCopyRule("processed/depth_priors", "processed/depth_priors"),
    ArtifactCopyRule("nerfstudio", "training/nerfstudio"),
    ArtifactCopyRule("diagnostics", "diagnostics"),
)


def copy_artifact_bundle(job_dir: Path, output_dir: Path) -> Path:
    """Copy private debug artifacts from a worker job directory into output_dir."""
    artifact_dir = output_dir / ARTIFACT_DIR_NAME
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for rule in ARTIFACT_COPY_RULES:
        src = job_dir / rule.source
        dest = artifact_dir / rule.destination
        copy_artifact_path(src, dest)
    return write_artifact_manifest(output_dir)


def copy_artifact_path(src: Path, dest: Path) -> None:
    if not src.exists():
        return
    if src.is_dir():
        if dest.exists() and not dest.is_dir():
            dest.unlink()
        shutil.copytree(src, dest, dirs_exist_ok=True)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def write_artifact_manifest(root: Path, *, job_id: str | None = None) -> Path:
    path = root / MANIFEST_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_artifact_manifest(root, job_id=job_id)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def build_artifact_manifest(root: Path, *, job_id: str | None = None) -> dict:
    root = root.resolve()
    artifacts = [file_entry(root, path) for path in iter_manifest_files(root)]
    return {
        "version": 1,
        "job_id": job_id,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


def iter_manifest_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return ()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    return (path for path in files if include_in_manifest(root, path))


def include_in_manifest(root: Path, path: Path) -> bool:
    rel = path.relative_to(root).as_posix()
    if rel == MANIFEST_NAME or rel.endswith(".tmp"):
        return False
    if rel in {"cleaned_splat.ply", "raw_splat.ply"}:
        export_copy = root / "export" / rel
        return not export_copy.exists()
    return True


def file_entry(root: Path, path: Path) -> dict:
    rel = path.relative_to(root).as_posix()
    stat = path.stat()
    entry = {
        "path": rel,
        "size_bytes": stat.st_size,
        "category": artifact_category(rel),
        "retention_tier": retention_tier(rel),
        "public": is_public_artifact(rel),
    }
    if stat.st_size <= HASH_LIMIT_BYTES:
        entry["sha256"] = sha256_file(path)
    else:
        entry["sha256"] = None
        entry["hash_skipped_reason"] = f"larger_than_{HASH_LIMIT_BYTES}_bytes"
    return entry


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_category(rel: str) -> str:
    if rel.startswith("source_media/") or rel.startswith(f"{ARTIFACT_DIR_NAME}/source_media/"):
        return "source_media"
    if rel.startswith("frames/") or rel.startswith(f"{ARTIFACT_DIR_NAME}/frames/"):
        return "frames"
    if rel.startswith("masks/") or rel.startswith(f"{ARTIFACT_DIR_NAME}/masks/"):
        return "masks"
    if rel.startswith("processed/") or rel.startswith(f"{ARTIFACT_DIR_NAME}/processed/"):
        return "processed_data"
    if rel.startswith("training/") or rel.startswith(f"{ARTIFACT_DIR_NAME}/training/"):
        return "training"
    if rel.startswith("export/"):
        return "export"
    if rel.startswith("renders/"):
        return "render"
    if rel.startswith("diagnostics/") or rel.startswith(f"{ARTIFACT_DIR_NAME}/diagnostics/"):
        return "diagnostics"
    if rel.endswith("_report.json") or rel in {"metrics.json", "settings.json", "job_overrides.json"}:
        return "metadata"
    if rel.endswith(".log"):
        return "logs"
    return "other"


def retention_tier(rel: str) -> str:
    category = artifact_category(rel)
    if category in {"source_media", "frames", "masks", "training"}:
        return "heavy_14d"
    if category in {"metadata", "logs", "diagnostics", "processed_data"}:
        return "debug_30d"
    if category in {"export", "render"}:
        return "result_retention"
    return "debug_30d"


def is_public_artifact(rel: str) -> bool:
    if rel.startswith("export/") and rel.endswith(("cleaned_splat.ply", "mesh.glb", "mesh.gltf", "mesh.obj")):
        return True
    if rel.startswith("renders/") and rel.endswith("turntable.mp4"):
        return True
    return rel in {"metrics.json", "quality_report.json", "candidate_report.json"}
