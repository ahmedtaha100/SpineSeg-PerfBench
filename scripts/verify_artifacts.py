#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

from spineseg_perfbench.utils.hashing import file_sha256
from spineseg_perfbench.utils.ledger import (
    ledger_entries as _ledger_entries,
    ledger_json_paths as _ledger_json_paths,
    ledger_run_ids as _ledger_run_ids,
)
from spineseg_perfbench.utils.schema import validate_run_row

OS_METADATA_NAMES = {".DS_Store", "Thumbs.db"}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify a frozen SpineSeg-PerfBench artifact bundle.")
    p.add_argument("bundle", nargs="?", default="artifacts/frozen")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def _check_no_raw_scans(root: Path, errors: list[str]) -> None:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        suffix = "".join(path.suffixes).lower()
        if suffix in {".nii", ".nii.gz", ".mha", ".mhd", ".nrrd", ".dcm", ".dicom"}:
            errors.append(f"raw medical/synthetic scan file found in frozen bundle: {path}")


def _check_no_os_metadata(root: Path, errors: list[str]) -> None:
    for path in root.rglob("*"):
        if path.name in OS_METADATA_NAMES or "__MACOSX" in path.parts:
            errors.append(f"OS metadata file should not be in frozen bundle: {path}")


def _csv_semantically_equal(frozen_path: Path, regenerated_path: Path) -> bool:
    try:
        frozen_df = pd.read_csv(frozen_path)
        regenerated_df = pd.read_csv(regenerated_path)
        if list(frozen_df.columns) != list(regenerated_df.columns):
            return False
        if frozen_df.shape != regenerated_df.shape:
            return False
        sort_cols = list(frozen_df.columns)
        frozen_df = frozen_df.sort_values(sort_cols).reset_index(drop=True)
        regenerated_df = regenerated_df.sort_values(sort_cols).reset_index(drop=True)
        pd.testing.assert_frame_equal(
            frozen_df,
            regenerated_df,
            check_dtype=False,
            check_exact=False,
            rtol=1e-9,
            atol=1e-12,
        )
        return True
    except Exception:
        return False


def _compare_regenerated_outputs(frozen_dir: Path, regenerated_dir: Path, pattern: str, label: str, errors: list[str]) -> None:
    frozen_files = sorted(frozen_dir.glob(pattern))
    regenerated_files = sorted(regenerated_dir.glob(pattern))
    frozen_names = {p.name for p in frozen_files}
    regenerated_names = {p.name for p in regenerated_files}
    for name in sorted(frozen_names - regenerated_names):
        errors.append(f"regenerated {label} missing: {name}")
    for name in sorted(regenerated_names - frozen_names):
        errors.append(f"unexpected regenerated {label}: {name}")
    for name in sorted(frozen_names & regenerated_names):
        frozen_path = frozen_dir / name
        regenerated_path = regenerated_dir / name
        if label == "table" and frozen_path.suffix.lower() == ".csv":
            if not _csv_semantically_equal(frozen_path, regenerated_path):
                errors.append(f"regenerated {label} differs from frozen artifact: {name}")
        elif label == "figure":
            if regenerated_path.stat().st_size == 0:
                errors.append(f"regenerated {label} is empty: {name}")
        elif file_sha256(frozen_path) != file_sha256(regenerated_path):
            errors.append(f"regenerated {label} differs from frozen artifact: {name}")


def _infer_smoke_bundle(root: Path) -> bool:
    run_log = root / "RUN_LOG.md"
    if not run_log.exists():
        return False
    try:
        return "Run mode: smoke" in run_log.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False


def _check_dataset_consistency(root: Path, json_rows: list[Path], errors: list[str]) -> None:
    manifest_path = root / "outputs/manifests/data_manifest.csv"
    if not manifest_path.exists() or not json_rows:
        return
    try:
        manifest = pd.read_csv(manifest_path)
    except Exception as exc:
        errors.append(f"could not read manifest for dataset consistency check: {exc}")
        return
    if "dataset_source" not in manifest.columns:
        return
    manifest_sources = {str(x).strip().lower() for x in manifest["dataset_source"].dropna().unique() if str(x).strip()}
    if not manifest_sources:
        return
    for path in json_rows:
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        dataset = str(row.get("dataset", "")).strip().lower()
        if dataset not in manifest_sources:
            rel = path.relative_to(root).as_posix()
            errors.append(
                f"dataset label mismatch in {rel}: row dataset={row.get('dataset')!r}, "
                f"manifest dataset_source={sorted(manifest_sources)}"
            )


def main() -> int:
    args = parse_args()
    root = Path(args.bundle)
    root_resolved = root.resolve()
    scripts_dir = Path(__file__).resolve().parent
    mode_flag = ["--smoke"] if args.smoke or _infer_smoke_bundle(root) else []
    errors: list[str] = []
    warnings: list[str] = []
    required = [
        "SPEC.md",
        "RUNS.md",
        "ARTIFACT_INDEX.json",
        "checksums.sha256",
        "outputs/benchmarks",
        "outputs/figures",
        "outputs/robustness",
        "outputs/tables",
    ]
    for rel in required:
        if not (root / rel).exists():
            errors.append(f"missing required path: {rel}")
    checksum_file = root / "checksums.sha256"
    if checksum_file.exists():
        for line in checksum_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                expected, rel = line.split("  ", 1)
            except ValueError:
                errors.append(f"malformed checksum line in {checksum_file}: {line}")
                continue
            candidates = [(root / rel).resolve(), Path(rel).resolve()]
            path = None
            escaped_existing_path = False
            for candidate in candidates:
                try:
                    candidate.relative_to(root_resolved)
                except ValueError:
                    if candidate.exists():
                        escaped_existing_path = True
                    continue
                if candidate.exists():
                    path = candidate
                    break
            if path is None and escaped_existing_path:
                errors.append(f"checksum path escapes bundle root: {rel}")
            elif path is None:
                errors.append(f"checksum path missing: {rel}")
            elif file_sha256(path) != expected:
                errors.append(f"checksum mismatch: {rel}")
    json_rows = sorted((root / "outputs/benchmarks").glob("*.json"))
    if not json_rows:
        errors.append("no benchmark JSON rows in bundle")
    ledger = (root / "RUNS.md").read_text(encoding="utf-8") if (root / "RUNS.md").exists() else ""
    ledger_entries = _ledger_entries(ledger)
    for rel_json in set(_ledger_json_paths(ledger)):
        path = (root / rel_json).resolve()
        try:
            path.relative_to(root_resolved)
        except ValueError:
            errors.append(f"RUNS.md JSON path escapes bundle root: {rel_json}")
            continue
        if not path.exists():
            errors.append(f"RUNS.md references missing JSON path: {rel_json}")
    for path in json_rows:
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            validate_run_row(row)
            if not row.get("config_hash"):
                errors.append(f"missing config_hash in {path}")
            if not row.get("hardware"):
                errors.append(f"missing hardware in {path}")
            rel_json = path.relative_to(root).as_posix()
            if (str(row["run_id"]), rel_json) not in ledger_entries:
                errors.append(f"RUNS.md missing run_id {row['run_id']}")
        except Exception as exc:
            errors.append(f"invalid JSON row {path}: {exc}")
    _check_dataset_consistency(root, json_rows, errors)
    _check_no_raw_scans(root, errors)
    _check_no_os_metadata(root, errors)
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            shutil.copytree(root / "outputs", tmp / "outputs")
            subprocess.check_call(
                [
                    sys.executable,
                    str(scripts_dir / "make_tables.py"),
                    "--input-dir",
                    str(tmp / "outputs"),
                    "--output-dir",
                    str(tmp / "tables"),
                    *mode_flag,
                ],
                stdout=subprocess.DEVNULL,
            )
            subprocess.check_call(
                [
                    sys.executable,
                    str(scripts_dir / "make_plots.py"),
                    "--input-dir",
                    str(tmp / "outputs"),
                    "--output-dir",
                    str(tmp / "figures"),
                    *mode_flag,
                ],
                stdout=subprocess.DEVNULL,
            )
            _compare_regenerated_outputs(
                root / "outputs/tables",
                tmp / "tables",
                "*.csv",
                "table",
                errors,
            )
            _compare_regenerated_outputs(
                root / "outputs/figures",
                tmp / "figures",
                "*.png",
                "figure",
                errors,
            )
    except Exception as exc:
        errors.append(f"table/plot regeneration failed: {exc}")
    if errors:
        for err in errors:
            print(f"ERROR: {err}")
        return 1
    for warning in warnings:
        print(f"WARNING: {warning}")
    print(f"verified {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
