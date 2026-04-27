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
        except Exception:  # noqa: BLE001
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


def main() -> None:
    argv = sys.argv[1:]
    real_colmap = _real_colmap()
    result = subprocess.run([real_colmap, *argv], check=False)
    if result.returncode == 0 and argv[:1] == ["mapper"]:
        output_path = _option_value(argv, "--output_path")
        if output_path:
            promote_largest_sparse_model(Path(output_path))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
