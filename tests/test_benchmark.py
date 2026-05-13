from __future__ import annotations

import importlib.util
from pathlib import Path


def _benchmark_module():
    spec = importlib.util.spec_from_file_location("benchmark_script", Path("scripts/benchmark.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_safe_run_id_removes_path_separators():
    benchmark = _benchmark_module()
    assert benchmark.safe_run_id("../../bad/id") == "bad_id"


def test_unique_run_id_avoids_existing_json(tmp_path):
    benchmark = _benchmark_module()
    (tmp_path / "run.json").write_text("{}", encoding="utf-8")
    assert benchmark.unique_run_id("run", tmp_path) != "run"


def test_unique_run_id_handles_long_existing_json(tmp_path):
    benchmark = _benchmark_module()
    run_id = "x" * 240
    existing = benchmark.safe_run_id(run_id)
    (tmp_path / f"{existing}.json").write_text("{}", encoding="utf-8")
    unique = benchmark.unique_run_id(run_id, tmp_path)
    assert unique != existing
    assert len(unique) <= 180
