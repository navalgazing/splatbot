from splatbot.config import Settings
from splatbot.pipeline import PipelineOutputs
import struct

import pytest

from splatbot.viewer import publish_viewer, safe_result_dir, write_viewer_point_cloud


def test_publish_viewer_writes_static_result_page(tmp_path) -> None:
    ply = tmp_path / "cleaned_splat.ply"
    preview = tmp_path / "turntable.mp4"
    ply.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 1",
                "property float x",
                "property float y",
                "property float z",
                "end_header",
                "0 0 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    preview.write_bytes(b"fake")

    path = publish_viewer(
        Settings(public_results_dir=tmp_path / "public"),
        "job1",
        PipelineOutputs(cleaned_ply=ply, preview_mp4=preview),
    )

    assert path == tmp_path / "public" / "job1" / "index.html"
    assert (tmp_path / "public" / "job1" / "cleaned_splat.ply").exists()
    assert (tmp_path / "public" / "job1" / "viewer_points.ply").exists()
    assert (tmp_path / "public" / "job1" / "turntable.mp4").exists()
    assert (tmp_path / "public" / "_viewer_assets" / "three.module.js").exists()
    assert (tmp_path / "public" / "_viewer_assets" / "controls" / "OrbitControls.js").exists()
    assert (tmp_path / "public" / "_viewer_assets" / "loaders" / "PLYLoader.js").exists()
    assert (tmp_path / "public" / "_viewer_assets" / "gaussian-splats-3d.module.js").exists()
    html = path.read_text(encoding="utf-8")
    assert 'type="importmap"' in html
    assert 'rel="modulepreload"' in html
    assert "sha384-" in html
    assert 'from "three"' in html
    assert "viewer_points.ply" in html
    assert "PLYLoader" in html
    assert "Loading point preview" in html
    assert "Use full splat view" in html
    assert "startPointPreview();" in html
    assert 'import * as GaussianSplats3D' not in html
    assert 'await import("../_viewer_assets/gaussian-splats-3d.module.js")' in html
    assert "../_viewer_assets/three.module.js" in html
    assert "https://unpkg.com/" not in html
    assert "https://cdn.jsdelivr.net/" not in html
    assert 'name="robots"' in html


def test_publish_viewer_allows_missing_preview(tmp_path) -> None:
    ply = tmp_path / "cleaned_splat.ply"
    ply.write_text("ply\nformat ascii 1.0\nend_header\n", encoding="utf-8")

    path = publish_viewer(
        Settings(public_results_dir=tmp_path / "public"),
        "job1",
        PipelineOutputs(cleaned_ply=ply, preview_mp4=None),
    )

    html = path.read_text(encoding="utf-8")
    assert (tmp_path / "public" / "job1" / "cleaned_splat.ply").exists()
    assert (tmp_path / "public" / "job1" / "viewer_points.ply").exists()
    assert not (tmp_path / "public" / "job1" / "turntable.mp4").exists()
    assert "Download preview video" not in html


def test_publish_viewer_copies_mesh_and_quality_report(tmp_path) -> None:
    ply = tmp_path / "cleaned_splat.ply"
    mesh = tmp_path / "mesh.glb"
    report = tmp_path / "quality_report.json"
    candidate = tmp_path / "candidate_report.json"
    ply.write_text("ply\nformat ascii 1.0\nend_header\n", encoding="utf-8")
    mesh.write_bytes(b"glb")
    report.write_text('{"passed": true}\n', encoding="utf-8")
    candidate.write_text('{"jobs": []}\n', encoding="utf-8")

    path = publish_viewer(
        Settings(public_results_dir=tmp_path / "public"),
        "job1",
        PipelineOutputs(
            cleaned_ply=ply,
            preview_mp4=None,
            mesh_path=mesh,
            quality_report_path=report,
            candidate_report_path=candidate,
        ),
    )

    result_dir = tmp_path / "public" / "job1"
    html = path.read_text(encoding="utf-8")
    assert (result_dir / "mesh.glb").exists()
    assert (result_dir / "quality_report.json").exists()
    assert (result_dir / "candidate_report.json").exists()
    assert "Use mesh view" in html
    assert "Download mesh" in html
    assert "GLTFLoader" in html
    assert (result_dir.parent / "_viewer_assets" / "loaders" / "GLTFLoader.js").exists()


def test_publish_viewer_rejects_unsupported_mesh_extension(tmp_path) -> None:
    ply = tmp_path / "cleaned_splat.ply"
    mesh = tmp_path / "mesh.html"
    ply.write_text("ply\nformat ascii 1.0\nend_header\n", encoding="utf-8")
    mesh.write_text("<p>no</p>", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported viewer mesh extension"):
        publish_viewer(
            Settings(public_results_dir=tmp_path / "public"),
            "job1",
            PipelineOutputs(cleaned_ply=ply, preview_mp4=None, mesh_path=mesh),
        )


def test_write_viewer_point_cloud_converts_gaussian_dc_color(tmp_path) -> None:
    src = tmp_path / "gaussian.ply"
    dest = tmp_path / "viewer_points.ply"
    src.write_bytes(
        (
            "ply\n"
            "format binary_little_endian 1.0\n"
            "element vertex 1\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property float f_dc_0\n"
            "property float f_dc_1\n"
            "property float f_dc_2\n"
            "property float opacity\n"
            "end_header\n"
        ).encode("ascii")
        + struct.pack("<fffffff", 1.0, 2.0, 3.0, 1.0, 0.0, -1.0, 0.5)
    )

    write_viewer_point_cloud(src, dest)

    data = dest.read_bytes()
    header_end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:header_end].decode("ascii")
    assert "property uchar red" in header
    assert "property uchar green" in header
    assert "property uchar blue" in header
    assert struct.unpack("<fffBBB", data[header_end:]) == (1.0, 2.0, 3.0, 199, 128, 56)


def test_safe_result_dir_rejects_path_traversal(tmp_path) -> None:
    with pytest.raises(ValueError):
        safe_result_dir(tmp_path / "public", "../escape")


def test_write_viewer_point_cloud_copies_truncated_binary_ply(tmp_path) -> None:
    src = tmp_path / "bad.ply"
    dest = tmp_path / "viewer_points.ply"
    data = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "element vertex 2\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float f_dc_0\n"
        "property float f_dc_1\n"
        "property float f_dc_2\n"
        "end_header\n"
    ).encode("ascii")
    src.write_bytes(data)

    write_viewer_point_cloud(src, dest)

    assert dest.read_bytes() == data


def test_write_viewer_point_cloud_accepts_crlf_binary_header(tmp_path) -> None:
    src = tmp_path / "gaussian_crlf.ply"
    dest = tmp_path / "viewer_points.ply"
    header = (
        "ply\r\n"
        "format binary_little_endian 1.0\r\n"
        "element vertex 1\r\n"
        "property float x\r\n"
        "property float y\r\n"
        "property float z\r\n"
        "property float f_dc_0\r\n"
        "property float f_dc_1\r\n"
        "property float f_dc_2\r\n"
        "end_header\r\n"
    ).encode("ascii")
    src.write_bytes(header + struct.pack("<ffffff", 1.0, 2.0, 3.0, 0.0, 0.0, 0.0))

    write_viewer_point_cloud(src, dest)

    assert b"property uchar red" in dest.read_bytes()
