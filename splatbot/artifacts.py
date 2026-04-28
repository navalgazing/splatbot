from __future__ import annotations

import atexit
import mimetypes
from dataclasses import dataclass
from pathlib import Path

import boto3
from botocore.client import Config

from .config import Settings


@dataclass(frozen=True)
class ArtifactRef:
    key: str
    url: str


class ArtifactStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = None

    @property
    def enabled(self) -> bool:
        return bool(
            self.settings.s3_endpoint_url
            and self.settings.s3_bucket
            and self.settings.s3_access_key_id_value
            and self.settings.s3_secret_access_key_value
        )

    @property
    def client(self):
        if self._client is None:
            self._client = boto3.client(
                "s3",
                endpoint_url=self.settings.s3_endpoint_url,
                region_name=self.settings.s3_region,
                aws_access_key_id=self.settings.s3_access_key_id_value,
                aws_secret_access_key=self.settings.s3_secret_access_key_value,
                config=Config(signature_version="s3v4"),
            )
            atexit.register(self._client.close)
        return self._client

    def upload(self, path: Path, key: str) -> ArtifactRef:
        if not self.enabled:
            raise RuntimeError("S3 artifact storage is not configured")
        content_type = _content_type(path)
        self.client.upload_file(
            str(path),
            self.settings.s3_bucket,
            key,
            ExtraArgs={"ContentType": content_type} if content_type else None,
        )
        url = self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.settings.s3_bucket, "Key": key},
            ExpiresIn=self.settings.signed_url_ttl_seconds,
        )
        return ArtifactRef(key=key, url=url)


def _content_type(path: Path) -> str | None:
    if path.suffix.lower() == ".ply":
        return "model/ply"
    if path.suffix.lower() == ".glb":
        return "model/gltf-binary"
    if path.suffix.lower() == ".obj":
        return "model/obj"
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed
