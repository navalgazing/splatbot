from splatbot.config import Settings
from splatbot.pipeline import PipelineOutputs
import struct

from splatbot.viewer import publish_viewer, write_viewer_point_cloud


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
    html = path.read_text(encoding="utf-8")
    assert 'type="importmap"' in html
    assert 'from "three"' in html
    assert "viewer_points.ply" in html
    assert "PLYLoader" in html


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
