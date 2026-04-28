from __future__ import annotations

import os
import shutil
import subprocess
import sys
import uuid
from collections.abc import Callable
from pathlib import Path


def _real_colmap() -> str:
    configured = os.environ.get("SPLATBOT_REAL_COLMAP_BIN")
    if configured:
        return configured
    for candidate in ("/usr/local/bin/colmap", "/usr/bin/colmap"):
        if Path(candidate).exists():
            return candidate
    return "colmap"


def _option_value(argv: list[str], name: str) -> str | None:
    prefix = f"{name}="
    for idx, item in enumerate(argv):
        if item == name and idx + 1 < len(argv):
            return argv[idx + 1]
        if item.startswith(prefix):
            return item[len(prefix) :]
    return None


def _read_registered_image_count(images_bin: Path) -> int:
    from nerfstudio.data.utils.colmap_parsing_utils import read_images_binary

    return len(read_images_binary(images_bin))


def promote_largest_sparse_model(
    sparse_dir: Path,
    image_count_reader: Callable[[Path], int] = _read_registered_image_count,
) -> Path | None:
    models = [path for path in sparse_dir.iterdir() if path.is_dir() and (path / "images.bin").exists()]
    if len(models) < 2:
        return models[0] if models else None

    def score(model: Path) -> tuple[int, int]:
        try:
            image_count = image_count_reader(model / "images.bin")
        except Exception as exc:  # noqa: BLE001
            print(
                f"splatbot-colmap-wrapper: failed to read registered image count for {model}: {exc}",
                file=sys.stderr,
            )
            image_count = 0
        points_size = (model / "points3D.bin").stat().st_size if (model / "points3D.bin").exists() else 0
        return image_count, points_size

    best = max(models, key=score)
    if best.name == "0":
        return best

    target = sparse_dir / "0"
    if target.exists():
        target.rename(sparse_dir / f"0-splatbot-replaced-{uuid.uuid4().hex[:8]}")
    shutil.copytree(best, target)
    print(f"splatbot-colmap-wrapper: promoted {best} to {target} score={score(best)}", file=sys.stderr)
    return target


def _has_colmap_command(command: str) -> bool:
    result = subprocess.run([_real_colmap(), command, "-h"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    return result.returncode == 0


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _mapped_argv(argv: list[str]) -> list[str]:
    mapper = os.environ.get("SPLATBOT_COLMAP_MAPPER", "").strip().lower()
    if argv[:1] == ["mapper"] and mapper in {"global", "glomap", "global_mapper"}:
        if _has_colmap_command("global_mapper"):
            return ["global_mapper", *argv[1:]]
        print(
            "splatbot-colmap-wrapper: requested global mapper but this COLMAP build has no global_mapper; "
            "falling back to mapper",
            file=sys.stderr,
        )
    return argv


def _maybe_calibrate_global_mapper(argv: list[str]) -> None:
    if argv[:1] != ["global_mapper"] or not _truthy_env("SPLATBOT_COLMAP_GLOBAL_CALIBRATE"):
        return
    database_path = _option_value(argv, "--database_path")
    if not database_path:
        return
    if not _has_colmap_command("view_graph_calibrator"):
        print(
            "splatbot-colmap-wrapper: requested view graph calibration but this COLMAP build "
            "has no view_graph_calibrator; continuing",
            file=sys.stderr,
        )
        return
    result = subprocess.run(
        [_real_colmap(), "view_graph_calibrator", "--database_path", database_path],
        check=False,
    )
    if result.returncode != 0:
        print(
            "splatbot-colmap-wrapper: view_graph_calibrator failed; continuing with global_mapper",
            file=sys.stderr,
        )


def main() -> None:
    argv = sys.argv[1:]
    argv = _mapped_argv(argv)
    _maybe_calibrate_global_mapper(argv)
    real_colmap = _real_colmap()
    result = subprocess.run([real_colmap, *argv], check=False)
    if result.returncode == 0 and argv[:1] in (["mapper"], ["global_mapper"]):
        output_path = _option_value(argv, "--output_path")
        if output_path:
            promote_largest_sparse_model(Path(output_path))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
