#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from spineseg_perfbench.utils.hashing import file_sha256
from spineseg_perfbench.utils.io import ensure_dir, write_json
from spineseg_perfbench.utils.ledger import ledger_entries as _ledger_entries

OS_METADATA_NAMES = {".DS_Store", "Thumbs.db"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Freeze benchmark artifacts into a local bundle.")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--output-dir", default="artifacts/local_frozen")
    p.add_argument(
        "--overwrite-submission-bundle",
        action="store_true",
        help="Permit writing to artifacts/frozen. Equivalent to OVERWRITE_SUBMISSION_BUNDLE=1.",
    )
    return p.parse_args()


def _copy_file(src: Path, dst: Path, copied: list[dict]) -> None:
    if src.name in OS_METADATA_NAMES or "__MACOSX" in src.parts:
        return
    if src.exists() and src.is_file():
        ensure_dir(dst.parent)
        shutil.copy2(src, dst)
        copied.append({"path": str(dst), "source": str(src), "sha256": file_sha256(dst), "bytes": dst.stat().st_size})


def _copy_tree(src: Path, dst: Path, copied: list[dict], patterns: tuple[str, ...] | None = None) -> None:
    if not src.exists():
        return
    files = []
    if patterns:
        for pat in patterns:
            files.extend(src.glob(pat))
    else:
        files = [p for p in src.rglob("*") if p.is_file()]
    for path in sorted(set(files)):
        if path.name in OS_METADATA_NAMES or "__MACOSX" in path.parts:
            continue
        rel = path.relative_to(src)
        _copy_file(path, dst / rel, copied)


def _missing_ledger_rows(ledger_path: Path, benchmark_dir: Path) -> list[str]:
    if not benchmark_dir.exists():
        return []
    ledger = ledger_path.read_text(encoding="utf-8") if ledger_path.exists() else ""
    ledger_entries = _ledger_entries(ledger)
    missing: list[str] = []
    for path in sorted(benchmark_dir.glob("*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            run_id = row["run_id"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            missing.append(f"{path}: invalid benchmark JSON ({exc})")
            continue
        except (OSError, UnicodeDecodeError) as exc:
            missing.append(f"{path}: unreadable benchmark JSON ({exc})")
            continue
        candidate_paths = {path.as_posix(), f"outputs/benchmarks/{path.name}"}
        if not any((str(run_id), candidate_path) in ledger_entries for candidate_path in candidate_paths):
            expected_paths = " or ".join(sorted(candidate_paths))
            missing.append(f"{path}: missing RUNS.md row for run_id {run_id} and JSON path {expected_paths}")
    return missing


def _write_bundle_spec(path: Path) -> None:
    path.write_text(
        "# SpineSeg-PerfBench Frozen Bundle Specification\n\n"
        "This bundle contains the GitHub-clean reproducibility artifacts for a "
        "SpineSeg-PerfBench run. It includes sanitized manifests, benchmark JSON "
        "rows, robustness CSVs, generated result tables and figures, profiler "
        "summaries, training metrics, configs, a run ledger, and checksums. It "
        "does not include raw medical scans, model checkpoints, local WandB "
        "directories, or oversized profiler traces.\n\n"
        "Raw dataset paths in manifests are sanitized with `${VERSE_ROOT}` or "
        "equivalent user-provided roots. The benchmark JSON rows and CSV outputs "
        "are the source of truth for reported measurements.\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    submission_bundle = Path("artifacts/frozen").resolve()
    if output_dir.resolve() == submission_bundle and not (
        args.overwrite_submission_bundle or os.environ.get("OVERWRITE_SUBMISSION_BUNDLE") == "1"
    ):
        print("ERROR: Refusing to overwrite artifacts/frozen/ without OVERWRITE_SUBMISSION_BUNDLE=1.")
        print("       Set OVERWRITE_SUBMISSION_BUNDLE=1 or pass --overwrite-submission-bundle to intentionally overwrite the submitted bundle.")
        return 1
    # Smoke and real bundles share the same traceability rule: benchmark scripts
    # append to RUNS.md synchronously before freeze_artifacts.py is invoked.
    missing = _missing_ledger_rows(Path("RUNS.md"), Path("outputs/benchmarks"))
    if missing:
        print("ERROR: refusing to freeze benchmark JSON rows without RUNS.md coverage.")
        for item in missing:
            print(f"ERROR: {item}")
        return 1
    frozen = ensure_dir(output_dir)
    if frozen.exists():
        shutil.rmtree(frozen)
    ensure_dir(frozen)
    copied: list[dict] = []
    _write_bundle_spec(frozen / "SPEC.md")
    copied.append({"path": str(frozen / "SPEC.md"), "source": "generated", "sha256": file_sha256(frozen / "SPEC.md"), "bytes": (frozen / "SPEC.md").stat().st_size})
    for name in ["RUNS.md", "README.md", "pyproject.toml", "environment.yml"]:
        _copy_file(Path(name), frozen / name, copied)
    _copy_tree(Path("configs"), frozen / "configs", copied)
    _copy_tree(Path("outputs/manifests"), frozen / "outputs/manifests", copied, ("*.csv",))
    _copy_tree(Path("outputs/benchmarks"), frozen / "outputs/benchmarks", copied, ("*.json",))
    _copy_tree(Path("outputs/profiles"), frozen / "outputs/profiles", copied, ("**/*.json", "**/*.csv"))
    _copy_tree(Path("outputs/runs"), frozen / "outputs/runs", copied, ("*/config.yaml", "*/metrics.json"))
    _copy_tree(Path("outputs"), frozen / "outputs", copied, ("robustness_results*.csv",))
    _copy_tree(Path("outputs"), frozen / "outputs/robustness", copied, ("robustness_results*.csv",))
    _copy_tree(Path("outputs/tables"), frozen / "outputs/tables", copied, ("*.csv",))
    _copy_tree(Path("outputs/figures"), frozen / "outputs/figures", copied, ("*.png",))
    _copy_tree(Path("artifacts/demo"), frozen / "artifacts/demo", copied, ("*.png",))
    write_json(frozen / "ARTIFACT_INDEX.json", {"files": copied})
    with (frozen / "checksums.sha256").open("w", encoding="utf-8") as f:
        for item in sorted(copied, key=lambda x: x["path"]):
            rel = Path(item["path"]).relative_to(frozen)
            f.write(f"{item['sha256']}  {rel}\n")
    (frozen / "RUN_LOG.md").write_text(
        "# Frozen Run Log\n\n"
        "This bundle contains schema-validated benchmark rows, generated tables, generated figures, configs, and run ledger files.\n"
        f"\nRun mode: {'smoke' if args.smoke else 'real'}.\n",
        encoding="utf-8",
    )
    print(frozen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
