from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys


def test_verify_artifacts_rejects_checksum_path_escape(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    digest = hashlib.sha256(outside.read_bytes()).hexdigest()
    (bundle / "checksums.sha256").write_text(f"{digest}  ../outside.txt\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "scripts/verify_artifacts.py", str(bundle)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1
    assert "checksum path escapes bundle root" in result.stdout


def test_verify_artifacts_infers_smoke_bundle_from_run_log(tmp_path):
    spec = importlib.util.spec_from_file_location("verify_artifacts_script", "scripts/verify_artifacts.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    assert module._infer_smoke_bundle(bundle) is False
    (bundle / "RUN_LOG.md").write_text("# Frozen Run Log\n\nRun mode: smoke.\n", encoding="utf-8")
    assert module._infer_smoke_bundle(bundle) is True

    unreadable_bundle = tmp_path / "unreadable_bundle"
    unreadable_bundle.mkdir()
    (unreadable_bundle / "RUN_LOG.md").mkdir()
    assert module._infer_smoke_bundle(unreadable_bundle) is False


def test_verify_artifacts_extracts_ledger_json_paths():
    spec = importlib.util.spec_from_file_location("verify_artifacts_script", "scripts/verify_artifacts.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    ledger = "\n".join(
        [
            "| run_id | git_sha | config_hash | JSON path | model | optimization | perturbation | one-line result |",
            "|---|---|---|---|---|---|---|---|",
            "| run_a | abc | def | outputs/benchmarks/a.json | segresnet | baseline | clean | dice=0.1 |",
            "|run_b|abc|def|outputs/benchmarks/b.json|segresnet|baseline|clean|dice=0.2|",
        ]
    )
    assert module._ledger_json_paths(ledger) == ["outputs/benchmarks/a.json", "outputs/benchmarks/b.json"]
    assert module._ledger_run_ids(ledger) == {"run_a", "run_b"}
    assert module._ledger_entries(ledger) == {
        ("run_a", "outputs/benchmarks/a.json"),
        ("run_b", "outputs/benchmarks/b.json"),
    }

    backtick_ledger = (
        "| `run_c` | abc | def | `outputs/benchmarks/c.json` | segresnet | baseline | clean | dice=0.3 |"
    )
    assert module._ledger_json_paths(backtick_ledger) == ["outputs/benchmarks/c.json"]
    assert module._ledger_run_ids(backtick_ledger) == {"run_c"}
    assert module._ledger_entries(backtick_ledger) == {("run_c", "outputs/benchmarks/c.json")}


def test_verify_artifacts_ledger_entries_do_not_match_swapped_rows():
    spec = importlib.util.spec_from_file_location("verify_artifacts_script", "scripts/verify_artifacts.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    ledger = "\n".join(
        [
            "| run_id | git_sha | config_hash | JSON path | model | optimization | perturbation | one-line result |",
            "|---|---|---|---|---|---|---|---|",
            "| run_a | abc | def | outputs/benchmarks/b.json | segresnet | baseline | clean | ok |",
            "| run_b | abc | def | outputs/benchmarks/a.json | segresnet | baseline | clean | ok |",
        ]
    )

    entries = module._ledger_entries(ledger)

    assert ("run_a", "outputs/benchmarks/a.json") not in entries
    assert ("run_b", "outputs/benchmarks/b.json") not in entries
