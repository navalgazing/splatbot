from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_run_prod_matrix():
    script = Path(__file__).parents[1] / "scripts" / "run_prod_matrix.py"
    spec = importlib.util.spec_from_file_location("run_prod_matrix_for_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_match_reference_owner_recursively_chowns_root_created_files(tmp_path, monkeypatch):
    module = load_run_prod_matrix()
    reference = tmp_path / "data"
    target = tmp_path / "jobs" / "job-id"
    nested = target / "nested"
    reference.mkdir()
    nested.mkdir(parents=True)
    (target / "job_overrides.json").write_text("{}", encoding="utf-8")
    (nested / "worker.log").write_text("", encoding="utf-8")

    calls: list[tuple[Path, int, int]] = []
    expected_uid = reference.stat().st_uid
    expected_gid = reference.stat().st_gid
    monkeypatch.setattr(module.os, "geteuid", lambda: 0)
    monkeypatch.setattr(module.os, "chown", lambda path, uid, gid: calls.append((Path(path), uid, gid)))

    module.match_reference_owner(target, reference)

    chowned = {path for path, _, _ in calls}
    assert {target, nested, target / "job_overrides.json", nested / "worker.log"} <= chowned
    assert all(uid == expected_uid and gid == expected_gid for _, uid, gid in calls)


def test_match_reference_owner_skips_chown_when_not_root(tmp_path, monkeypatch):
    module = load_run_prod_matrix()
    reference = tmp_path / "data"
    target = tmp_path / "jobs" / "job-id"
    reference.mkdir()
    target.mkdir(parents=True)
    calls: list[tuple[Path, int, int]] = []
    monkeypatch.setattr(module.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(module.os, "chown", lambda path, uid, gid: calls.append((Path(path), uid, gid)))

    module.match_reference_owner(target, reference)

    assert calls == []
