from pathlib import Path

from splatbot.artifacts import _content_type


def test_content_type_handles_common_artifacts() -> None:
    assert _content_type(Path("scan.ply")) == "model/ply"
    assert _content_type(Path("mesh.glb")) == "model/gltf-binary"
    assert _content_type(Path("preview.mp4")) == "video/mp4"

