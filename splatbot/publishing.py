from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .artifacts import ArtifactRef, ArtifactStore
from .config import Settings
from .models import ArtifactKind, JobArtifact, ScanJob
from .pipeline import PipelineOutputs
from .storage import Store
from .viewer import publish_viewer


class ArtifactUploader(Protocol):
    enabled: bool

    def upload(self, path: Path, key: str) -> ArtifactRef:
        ...


def upload_artifact(
    artifact_store: ArtifactUploader,
    job: ScanJob,
    kind: ArtifactKind,
    path: Path,
) -> ArtifactRef | None:
    if not artifact_store.enabled:
        return None
    suffix = path.suffix or f".{kind.value}"
    return artifact_store.upload(path, f"jobs/{job.id}/{kind.value}{suffix}")


async def publish_job_artifacts(
    settings: Settings,
    store: Store,
    job: ScanJob,
    outputs: PipelineOutputs,
    artifact_store: ArtifactUploader | None = None,
) -> list[JobArtifact]:
    artifact_store = artifact_store or ArtifactStore(settings)
    published: list[JobArtifact] = []
    viewer_path = publish_viewer(settings, job.id, outputs)
    viewer_url = settings.public_job_url(job.id)
    artifacts_to_publish = [(ArtifactKind.PLY, outputs.cleaned_ply)]
    if outputs.mesh_path is not None and outputs.mesh_path.exists():
        artifacts_to_publish.append((ArtifactKind.MESH, outputs.mesh_path))
    if outputs.quality_report_path is not None and outputs.quality_report_path.exists():
        artifacts_to_publish.append((ArtifactKind.QUALITY_REPORT, outputs.quality_report_path))
    if outputs.preview_mp4 is not None and outputs.preview_mp4.exists():
        artifacts_to_publish.append((ArtifactKind.PREVIEW, outputs.preview_mp4))
    for kind, path in artifacts_to_publish:
        ref = upload_artifact(artifact_store, job, kind, path)
        published.append(
            await store.add_artifact(
                job.id,
                kind,
                path,
                remote_key=ref.key if ref else None,
                url=ref.url if ref else None,
            )
        )
    published.append(
        await store.add_artifact(
            job.id,
            ArtifactKind.VIEWER,
            viewer_path,
            url=viewer_url or None,
        )
    )
    return published
