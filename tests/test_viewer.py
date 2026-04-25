from splatbot.config import Settings
from splatbot.pipeline import PipelineOutputs
from splatbot.viewer import publish_viewer


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
    assert (tmp_path / "public" / "job1" / "turntable.mp4").exists()
    assert "PLYLoader" in path.read_text(encoding="utf-8")


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
    assert not (tmp_path / "public" / "job1" / "turntable.mp4").exists()
    assert "Download preview video" not in html
