from __future__ import annotations

import importlib.util
import json
from types import ModuleType


def _load_freeze_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("freeze_artifacts_script", "scripts/freeze_artifacts.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_freeze_detects_benchmark_rows_missing_from_ledger(tmp_path):
    module = _load_freeze_module()
    bench = tmp_path / "outputs" / "benchmarks"
    bench.mkdir(parents=True)
    (bench / "row.json").write_text(json.dumps({"run_id": "run_missing"}), encoding="utf-8")
    ledger = tmp_path / "RUNS.md"
    ledger.write_text("# Run Ledger\n", encoding="utf-8")

    missing = module._missing_ledger_rows(ledger, bench)

    assert missing
    assert "run_missing" in missing[0]


def test_freeze_accepts_benchmark_rows_covered_by_ledger(tmp_path):
    module = _load_freeze_module()
    bench = tmp_path / "outputs" / "benchmarks"
    bench.mkdir(parents=True)
    (bench / "row.json").write_text(json.dumps({"run_id": "run_present"}), encoding="utf-8")
    ledger = tmp_path / "RUNS.md"
    ledger.write_text("| run_present | abc | def | outputs/benchmarks/row.json | segresnet | baseline | clean | ok |\n", encoding="utf-8")

    assert module._missing_ledger_rows(ledger, bench) == []

    ledger.write_text("|run_present|abc|def|outputs/benchmarks/row.json|segresnet|baseline|clean|ok|\n", encoding="utf-8")
    assert module._missing_ledger_rows(ledger, bench) == []

    ledger.write_text(
        "| `run_present` | abc | def | `outputs/benchmarks/row.json` | segresnet | baseline | clean | ok |\n",
        encoding="utf-8",
    )
    assert module._missing_ledger_rows(ledger, bench) == []


def test_freeze_reports_malformed_benchmark_json(tmp_path):
    module = _load_freeze_module()
    bench = tmp_path / "outputs" / "benchmarks"
    bench.mkdir(parents=True)
    (bench / "bad.json").write_text("{bad", encoding="utf-8")
    ledger = tmp_path / "RUNS.md"
    ledger.write_text("# Run Ledger\n", encoding="utf-8")

    missing = module._missing_ledger_rows(ledger, bench)

    assert missing
    assert "invalid benchmark JSON" in missing[0]


def test_freeze_rejects_mismatched_ledger_run_id_and_json_path(tmp_path):
    module = _load_freeze_module()
    bench = tmp_path / "outputs" / "benchmarks"
    bench.mkdir(parents=True)
    (bench / "a.json").write_text(json.dumps({"run_id": "run_a"}), encoding="utf-8")
    ledger = tmp_path / "RUNS.md"
    ledger.write_text(
        "\n".join(
            [
                "| run_id | git_sha | config_hash | JSON path | model | optimization | perturbation | one-line result |",
                "|---|---|---|---|---|---|---|---|",
                "| run_a | abc | def | outputs/benchmarks/b.json | segresnet | baseline | clean | ok |",
                "| run_b | abc | def | outputs/benchmarks/a.json | segresnet | baseline | clean | ok |",
            ]
        ),
        encoding="utf-8",
    )

    missing = module._missing_ledger_rows(ledger, bench)

    assert missing
    assert "run_a" in missing[0]
    assert "outputs/benchmarks/a.json" in missing[0]
