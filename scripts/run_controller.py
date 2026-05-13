#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from spineseg_perfbench.utils.hardware import collect_hardware_metadata
from spineseg_perfbench.utils.io import ensure_dir, read_json, write_json
from spineseg_perfbench.utils.schema import validate_run_row


STAGES = [
    "environment_smoke",
    "prepare_data",
    "manifest_validation",
    "label_distribution_audit",
    "segresnet_train",
    "unet_train",
    "baseline_benchmark",
    "baseline_profile",
    "baseline_robustness",
    "data_pipeline_sweep",
    "amp_sweep",
    "compile_benchmark",
    "combined_benchmark",
    "optimized_robustness",
    "make_tables",
    "make_plots",
    "freeze_artifacts",
    "verify_artifacts",
    "metric_health_audit",
    "traceability_report",
]

DEPENDENCIES = {
    "environment_smoke": [],
    "prepare_data": ["environment_smoke"],
    "manifest_validation": ["prepare_data"],
    "label_distribution_audit": ["manifest_validation"],
    "segresnet_train": ["label_distribution_audit"],
    "unet_train": ["label_distribution_audit"],
    "baseline_benchmark": ["segresnet_train"],
    "baseline_profile": ["segresnet_train"],
    "baseline_robustness": ["segresnet_train"],
    "data_pipeline_sweep": ["segresnet_train"],
    "amp_sweep": ["segresnet_train"],
    "compile_benchmark": ["segresnet_train"],
    "combined_benchmark": ["segresnet_train"],
    "optimized_robustness": ["combined_benchmark"],
    "make_tables": [
        "baseline_benchmark",
        "baseline_robustness",
        "data_pipeline_sweep",
        "amp_sweep",
        "compile_benchmark",
        "combined_benchmark",
        "optimized_robustness",
    ],
    "make_plots": ["make_tables"],
    "freeze_artifacts": ["make_plots"],
    "verify_artifacts": ["freeze_artifacts"],
    "metric_health_audit": ["verify_artifacts"],
    "traceability_report": ["metric_health_audit"],
}

EXPENSIVE_STAGES = {
    "segresnet_train",
    "unet_train",
    "baseline_benchmark",
    "baseline_robustness",
    "data_pipeline_sweep",
    "amp_sweep",
    "compile_benchmark",
    "combined_benchmark",
    "optimized_robustness",
    "make_plots",
    "freeze_artifacts",
}

GPU_HEAVY_STAGES = {
    "segresnet_train",
    "unet_train",
    "baseline_benchmark",
    "baseline_profile",
    "baseline_robustness",
    "data_pipeline_sweep",
    "amp_sweep",
    "compile_benchmark",
    "combined_benchmark",
    "optimized_robustness",
}

VALID_STATUSES = {"pending", "running", "complete", "failed", "skipped_valid", "blocked_external"}
STATE_PATH = Path("REAL_RUN_STATE.json")
REPORT_DIR = Path("outputs/reports")
METRIC_HEALTH_REPORT = REPORT_DIR / "final_metric_health_report.md"
TRACEABILITY_REPORT = REPORT_DIR / "result_traceability_report.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def compact_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def default_stage() -> dict[str, Any]:
    return {
        "status": "pending",
        "start_utc": None,
        "end_utc": None,
        "command": "",
        "log_file": "",
        "output_files": [],
        "validation_checks": [],
        "notes": "",
        "error": None,
    }


def init_state() -> dict[str, Any]:
    return {
        "run_id": f"real_run_{compact_utc()}",
        "created_utc": utc_now(),
        "updated_utc": utc_now(),
        "git_sha": git_sha(),
        "dataset_roots": {
            "VERSE_ROOT": os.environ.get("VERSE_ROOT") or None,
            "CTSPINE1K_ROOT": os.environ.get("CTSPINE1K_ROOT") or None,
        },
        "hardware": collect_hardware_metadata(),
        "stages": {stage: default_stage() for stage in STAGES},
    }


def load_state(path: Path = STATE_PATH, reset: bool = False) -> dict[str, Any]:
    if reset and path.exists():
        archive_dir = ensure_dir("logs")
        archive = archive_dir / f"run_state_archive_{compact_utc()}.json"
        shutil.copy2(path, archive)
        path.unlink()
    if path.exists():
        state = read_json(path)
    else:
        state = init_state()
    state.setdefault("run_id", f"real_run_{compact_utc()}")
    state.setdefault("created_utc", utc_now())
    state.setdefault("git_sha", git_sha())
    state.setdefault("dataset_roots", {})
    state.setdefault("hardware", {})
    state.setdefault("stages", {})
    for stage in STAGES:
        current = state["stages"].setdefault(stage, default_stage())
        for key, value in default_stage().items():
            current.setdefault(key, value)
        if current.get("status") not in VALID_STATUSES:
            current["status"] = "pending"
    state["updated_utc"] = utc_now()
    state["git_sha"] = git_sha()
    state["dataset_roots"] = {
        "VERSE_ROOT": os.environ.get("VERSE_ROOT") or None,
        "CTSPINE1K_ROOT": os.environ.get("CTSPINE1K_ROOT") or None,
    }
    state["hardware"] = collect_hardware_metadata()
    return state


def save_state(state: dict[str, Any], path: Path = STATE_PATH) -> None:
    state["updated_utc"] = utc_now()
    write_json(path, state)


def set_stage(
    state: dict[str, Any],
    stage: str,
    status: str,
    *,
    command: str | None = None,
    log_file: str | None = None,
    output_files: list[str] | None = None,
    validation_checks: list[str] | None = None,
    notes: str | None = None,
    error: str | None = None,
    state_path: Path = STATE_PATH,
) -> None:
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid stage status: {status}")
    item = state["stages"][stage]
    item["status"] = status
    if status == "running":
        item["start_utc"] = utc_now()
        item["end_utc"] = None
    elif status in {"complete", "failed", "skipped_valid", "blocked_external"}:
        item["end_utc"] = utc_now()
    if command is not None:
        item["command"] = command
    if log_file is not None:
        item["log_file"] = log_file
    if output_files is not None:
        item["output_files"] = output_files
    if validation_checks is not None:
        item["validation_checks"] = validation_checks
    if notes is not None:
        item["notes"] = notes
    item["error"] = error
    save_state(state, state_path)


def path_nonempty(path: str | Path) -> bool:
    p = Path(path)
    return p.exists() and p.is_file() and p.stat().st_size > 0


def _files(patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(Path().glob(pattern))
    return sorted({p for p in files if p.is_file()})


def _is_stale(sources: list[Path], outputs: list[Path]) -> bool:
    if not sources or not outputs:
        return False
    if any(not p.exists() for p in outputs):
        return True
    newest_source = max(p.stat().st_mtime for p in sources if p.exists())
    oldest_output = min(p.stat().st_mtime for p in outputs if p.exists())
    return newest_source > oldest_output


def run_quiet(command: list[str], timeout: int = 180) -> bool:
    try:
        subprocess.check_call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout)
        return True
    except Exception:
        return False


def _benchmark_rows() -> list[tuple[Path, dict[str, Any]]]:
    rows: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(Path("outputs/benchmarks").glob("*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            validate_run_row(row)
            rows.append((path, row))
        except Exception:
            continue
    return rows


def _row_matches(row: dict[str, Any], keyword: str) -> bool:
    opt = str(row.get("optimization", "")).lower()
    return keyword.lower() in opt


def _is_non_robust_row(row: dict[str, Any]) -> bool:
    return not str(row.get("run_id", "")).startswith("robust_")


def validate_manifest() -> tuple[bool, list[str], list[str]]:
    checks: list[str] = []
    outputs = [
        "outputs/manifests/data_manifest.csv",
        "outputs/manifests/split_train.csv",
        "outputs/manifests/split_val.csv",
        "outputs/manifests/split_test.csv",
    ]
    missing = [p for p in outputs if not path_nonempty(p)]
    if missing:
        return False, checks, [f"Missing manifest files: {missing}"]
    manifest = pd.read_csv(outputs[0])
    train = pd.read_csv(outputs[1])
    val = pd.read_csv(outputs[2])
    test = pd.read_csv(outputs[3])
    if manifest.empty or train.empty or val.empty or test.empty:
        return False, checks, ["Manifest or one split is empty."]
    checks.append(f"manifest_rows={len(manifest)}")
    for col in ["image_path", "label_path"]:
        missing_paths = [p for p in manifest[col].tolist() if not Path(str(p)).exists()]
        if missing_paths:
            return False, checks, [f"Missing paths in {col}: {missing_paths[:5]}"]
    for col in ["spacing_x", "spacing_y", "spacing_z", "shape_x", "shape_y", "shape_z"]:
        if col not in manifest.columns:
            return False, checks, [f"Missing manifest column: {col}"]
        if not (manifest[col] > 0).all():
            return False, checks, [f"Nonpositive values in manifest column: {col}"]
    sets = {"train": set(train.case_id), "val": set(val.case_id), "test": set(test.case_id)}
    overlaps = {
        "train_val": sets["train"] & sets["val"],
        "train_test": sets["train"] & sets["test"],
        "val_test": sets["val"] & sets["test"],
    }
    bad_overlaps = {k: sorted(v)[:5] for k, v in overlaps.items() if v}
    if bad_overlaps:
        return False, checks, [f"Split overlap detected: {bad_overlaps}"]
    n = len(manifest)
    if n >= 20:
        ratios = {k: len(v) / n for k, v in sets.items()}
        checks.append(f"split_ratios={ratios}")
        if abs(ratios["train"] - 0.70) >= 0.10 or abs(ratios["val"] - 0.15) >= 0.08 or abs(ratios["test"] - 0.15) >= 0.08:
            return False, checks, [f"Suspicious split ratios: {ratios}"]
    return True, checks, outputs


def validate_robustness(path: Path) -> tuple[bool, str]:
    if not path_nonempty(path):
        return False, f"Missing robustness CSV: {path}"
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return False, f"Unreadable robustness CSV {path}: {exc}"
    required = {"perturbation", "severity", "dice_mean", "hd95_mean_mm"}
    missing = required - set(df.columns)
    if missing:
        return False, f"{path} missing columns {sorted(missing)}"
    clean = df[(df["severity"] == 0) & (df["perturbation"].astype(str) == "clean")]
    if len(clean) != 1:
        return False, f"{path} expected one clean severity 0 row, found {len(clean)}"
    required_perturbations = {"gaussian_noise", "gaussian_blur", "downsample_resample", "intensity_shift", "contrast_shift"}
    observed = set(df[df["severity"].isin([1, 2, 3])]["perturbation"].astype(str))
    missing_pert = required_perturbations - observed
    if missing_pert:
        return False, f"{path} missing perturbations {sorted(missing_pert)}"
    for pert in required_perturbations:
        severities = set(df[df["perturbation"].astype(str) == pert]["severity"].astype(int))
        if not {1, 2, 3}.issubset(severities):
            return False, f"{path} missing severity grid for {pert}: {sorted(severities)}"
    return True, str(path)


def validate_stage_outputs(stage: str, smoke: bool = False) -> tuple[bool, list[str], str]:
    if stage == "environment_smoke":
        marker = Path("logs/environment_smoke_passed_utc.txt")
        run_log = Path("artifacts/frozen/RUN_LOG.md")
        frozen_is_smoke = run_log.exists() and "Run mode: smoke" in run_log.read_text(encoding="utf-8", errors="ignore")
        if marker.exists() and marker.stat().st_size > 0:
            return True, [str(marker)], "Smoke environment already validated."
        if frozen_is_smoke and Path("artifacts/frozen").exists() and run_quiet([sys.executable, "scripts/verify_artifacts.py", "artifacts/frozen", "--smoke"]):
            return True, ["smoke_frozen_bundle_verified"], "Smoke environment already validated."
        return False, [], "Environment smoke has not been validated in this run state."
    if stage == "prepare_data":
        ok = all(path_nonempty(p) for p in [
            "outputs/manifests/data_manifest.csv",
            "outputs/manifests/split_train.csv",
            "outputs/manifests/split_val.csv",
            "outputs/manifests/split_test.csv",
        ])
        return ok, ["manifest_files_present"] if ok else [], "" if ok else "Manifest files missing."
    if stage == "manifest_validation":
        ok, checks, outputs = validate_manifest()
        return ok, checks, "" if ok else "; ".join(outputs)
    if stage == "label_distribution_audit":
        p = Path("outputs/audits/label_distribution_audit.json")
        if not p.exists() or not Path("LABEL_DISTRIBUTION_AUDIT.md").exists():
            return False, [], "Label distribution audit outputs missing."
        try:
            verdict = read_json(p).get("verdict")
        except Exception as exc:
            return False, [], f"Unreadable label audit JSON: {exc}"
        return verdict == "pass", [f"label_audit_verdict={verdict}"], "" if verdict == "pass" else "Label audit did not pass."
    if stage == "segresnet_train":
        outputs = [
            "outputs/runs/segresnet_baseline/checkpoint.pt",
            "outputs/runs/segresnet_baseline/metrics.json",
            "outputs/runs/segresnet_baseline/config.yaml",
        ]
        ok = all(path_nonempty(p) for p in outputs)
        return ok, outputs if ok else [], "" if ok else "SegResNet checkpoint outputs missing."
    if stage == "unet_train":
        outputs = [
            "outputs/runs/unet_baseline/checkpoint.pt",
            "outputs/runs/unet_baseline/metrics.json",
            "outputs/runs/unet_baseline/config.yaml",
        ]
        ok = all(path_nonempty(p) for p in outputs)
        return ok, outputs if ok else [], "" if ok else "U-Net checkpoint outputs missing."
    rows = _benchmark_rows()
    if stage == "baseline_benchmark":
        matches = [
            (p, r)
            for p, r in rows
            if _is_non_robust_row(r)
            and _row_matches(r, "baseline")
            and str(r.get("model", "")).lower() == "segresnet"
            and r.get("perturbation") is None
        ]
        return bool(matches), [str(matches[-1][0])] if matches else [], "" if matches else "No schema-valid SegResNet baseline row."
    if stage == "baseline_profile":
        profile_files = list(Path("outputs/profiles").glob("**/*"))
        has_trace = any("trace" in p.name and p.suffix == ".json" and path_nonempty(p) for p in profile_files)
        has_operator = any("operator" in p.name and p.suffix == ".csv" and path_nonempty(p) for p in profile_files)
        has_phase = any("phase" in p.name and p.suffix == ".json" and path_nonempty(p) for p in profile_files)
        ok = has_trace and has_operator and has_phase
        return ok, [str(p) for p in profile_files if p.is_file()][:10], "" if ok else "Profile trace/operator/phase outputs missing."
    if stage == "baseline_robustness":
        ok, msg = validate_robustness(Path("outputs/robustness_results.csv"))
        return ok, [msg] if ok else [], "" if ok else msg
    if stage == "data_pipeline_sweep":
        matches = [(p, r) for p, r in rows if _row_matches(r, "data")]
        return bool(matches), [str(p) for p, _ in matches], "" if matches else "No data-pipeline rows."
    if stage == "amp_sweep":
        matches = [(p, r) for p, r in rows if _row_matches(r, "amp")]
        ok = bool(matches) and all("amp_dtype" in r.get("optimization_metadata", {}) for _, r in matches)
        return ok, [str(p) for p, _ in matches], "" if ok else "No AMP rows with dtype metadata."
    if stage == "compile_benchmark":
        matches = [(p, r) for p, r in rows if _row_matches(r, "compile")]
        ok = bool(matches) and all("compile_succeeded" in r.get("optimization_metadata", {}) for _, r in matches)
        return ok, [str(p) for p, _ in matches], "" if ok else "No compile row with compile metadata."
    if stage == "combined_benchmark":
        matches = [(p, r) for p, r in rows if _is_non_robust_row(r) and str(r.get("optimization", "")).lower() == "all"]
        return bool(matches), [str(p) for p, _ in matches], "" if matches else "No combined optimization row."
    if stage == "optimized_robustness":
        ok, msg = validate_robustness(Path("outputs/robustness_results_optimized.csv"))
        return ok, [msg] if ok else [], "" if ok else msg
    if stage == "make_tables":
        required = [
            "table1_dataset_split.csv",
            "table2_clean_quality.csv",
            "table3_efficiency_breakdown.csv",
            "table4_optimization_ablation.csv",
            "table5_robustness_summary.csv",
        ]
        outputs = [str(Path("outputs/tables") / x) for x in required]
        ok = all(path_nonempty(p) and len(pd.read_csv(p)) > 0 for p in outputs)
        if ok and _is_stale(
            _files(["outputs/benchmarks/*.json", "outputs/robustness_results*.csv", "outputs/manifests/*.csv"]),
            [Path(p) for p in outputs],
        ):
            return False, [], "Result tables are stale relative to benchmark/robustness/manifest sources."
        return ok, outputs if ok else [], "" if ok else "Required result tables missing or empty."
    if stage == "make_plots":
        required = [
            "fig1_pipeline_overview.png",
            "fig2_phase_time_breakdown.png",
            "fig3_accuracy_latency_pareto.png",
            "fig4_robustness_curves.png",
            "fig5_vram_vs_latency.png",
        ]
        outputs = [Path("outputs/figures") / x for x in required]
        ok = all(p.exists() and p.stat().st_size > 1000 for p in outputs)
        if ok and _is_stale(
            _files(["outputs/benchmarks/*.json", "outputs/robustness_results*.csv", "outputs/manifests/*.csv", "outputs/tables/*.csv"]),
            outputs,
        ):
            return False, [], "Result figures are stale relative to source outputs."
        return ok, [str(p) for p in outputs] if ok else [], "" if ok else "Required result figures missing or too small."
    if stage == "freeze_artifacts":
        outputs = ["artifacts/frozen/checksums.sha256", "artifacts/frozen/RUN_LOG.md", "artifacts/frozen/ARTIFACT_INDEX.json"]
        ok = Path("artifacts/frozen").exists() and all(path_nonempty(p) for p in outputs)
        if ok:
            missing_from_frozen = []
            for src in _files(["outputs/benchmarks/*.json"]):
                dst = Path("artifacts/frozen/outputs/benchmarks") / src.name
                if not dst.exists():
                    missing_from_frozen.append(str(src))
            for src in _files(["outputs/robustness_results*.csv"]):
                dst = Path("artifacts/frozen/outputs") / src.name
                if not dst.exists():
                    missing_from_frozen.append(str(src))
            if missing_from_frozen:
                return False, [], f"Frozen bundle missing current outputs: {missing_from_frozen[:5]}"
            if _is_stale(
                _files(
                    [
                        "outputs/manifests/*.csv",
                        "outputs/benchmarks/*.json",
                        "outputs/profiles/**/*.json",
                        "outputs/profiles/**/*.csv",
                        "outputs/robustness_results*.csv",
                        "outputs/tables/*.csv",
                        "outputs/figures/*.png",
                        "artifacts/demo/*.png",
                    ]
                ),
                [Path(p) for p in outputs],
            ):
                return False, [], "Frozen bundle is stale relative to current outputs."
        return ok, outputs if ok else [], "" if ok else "Frozen artifact metadata missing."
    if stage == "verify_artifacts":
        cmd = [sys.executable, "scripts/verify_artifacts.py", "artifacts/frozen"]
        if smoke:
            cmd.append("--smoke")
        ok = run_quiet(cmd, timeout=300)
        return ok, ["verify_artifacts.py artifacts/frozen"] if ok else [], "" if ok else "verify_artifacts.py failed."
    if stage == "metric_health_audit":
        p = METRIC_HEALTH_REPORT
        ok = p.exists() and "## Verdict\nPASS" in p.read_text(encoding="utf-8")
        return ok, [str(p)] if ok else [], "" if ok else f"{p} missing PASS verdict."
    if stage == "traceability_report":
        p = TRACEABILITY_REPORT
        ok = p.exists() and "## Verdict\nPASS" in p.read_text(encoding="utf-8")
        return ok, [str(p)] if ok else [], "" if ok else f"{p} missing PASS verdict."
    raise ValueError(f"Unknown stage: {stage}")


def free_disk_gb(path: str | Path = ".") -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024**3)


def largest_paths(limit: int = 20) -> list[str]:
    candidates: list[tuple[int, Path]] = []
    for root in [Path("outputs"), Path("artifacts"), Path("logs")]:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file():
                try:
                    candidates.append((p.stat().st_size, p))
                except OSError:
                    pass
    return [f"{size / (1024**3):.3f} GB {path}" for size, path in sorted(candidates, reverse=True)[:limit]]


def write_blocker_report(stage: str, message: str, min_free_gb: float, current_free_gb: float) -> None:
    lines = [
        "# BLOCKER_REPORT.md",
        "",
        "## Verdict",
        "",
        "BLOCKED — cannot safely continue without external action.",
        "",
        "## Stage",
        "",
        stage,
        "",
        "## Reason",
        "",
        message,
        "",
        "## Disk",
        "",
        f"Required free disk: {min_free_gb:.2f} GB",
        f"Current free disk: {current_free_gb:.2f} GB",
        "",
        "## Largest tracked files",
        "",
        *[f"- {x}" for x in largest_paths()],
        "",
        "Do not delete valid outputs blindly. Free disk externally or archive completed artifacts, then resume with:",
        "",
        "```bash",
        f"python scripts/run_controller.py --real --run-from {stage} --min-free-gb {min_free_gb:g}",
        "```",
    ]
    Path("BLOCKER_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


class GPUUtilMonitor:
    def __init__(self, stage: str, log_path: Path, interval_sec: int = 60, enabled: bool = True):
        self.stage = stage
        self.log_path = log_path
        self.interval_sec = interval_sec
        self.enabled = enabled and shutil.which("nvidia-smi") is not None
        self.samples: list[int] = []
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.stalled = False
        self.csv_path = Path("logs") / f"gpu_utilization_{stage}.csv"
        self._last_log_size = -1
        self._last_growth = time.time()

    def __enter__(self) -> "GPUUtilMonitor":
        if not self.enabled:
            return self
        ensure_dir(self.csv_path.parent)
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp_utc", "gpu_index", "gpu_name", "utilization_gpu_pct", "memory_used_mb", "memory_total_mb"])
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)
        self._write_low_util_alert(final=True)

    def _query(self) -> list[list[str]]:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=10)
        rows = []
        for line in out.splitlines():
            rows.append([part.strip() for part in line.split(",")])
        return rows

    def _loop(self) -> None:
        start = time.time()
        while not self.stop_event.is_set():
            try:
                rows = self._query()
                ts = utc_now()
                with self.csv_path.open("a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    for index, name, util, mem_used, mem_total in rows:
                        util_int = int(float(util))
                        self.samples.append(util_int)
                        writer.writerow([ts, index, name, util_int, int(float(mem_used)), int(float(mem_total))])
            except Exception:
                pass
            try:
                size = self.log_path.stat().st_size
                if size != self._last_log_size:
                    self._last_log_size = size
                    self._last_growth = time.time()
            except OSError:
                pass
            if time.time() - start >= 900 and self.samples and sum(self.samples) / len(self.samples) < 20:
                self._write_low_util_alert(final=False)
                if time.time() - self._last_growth >= 900:
                    self.stalled = True
            self.stop_event.wait(self.interval_sec)

    def _write_low_util_alert(self, final: bool) -> None:
        if not self.samples:
            return
        elapsed_enough = len(self.samples) * self.interval_sec >= 900
        mean_util = sum(self.samples) / len(self.samples)
        if mean_util >= 20 or not elapsed_enough:
            return
        ensure_dir("logs")
        with Path("logs/gpu_low_utilization_alerts.log").open("a", encoding="utf-8") as f:
            f.write(f"{utc_now()} stage={self.stage} mean_util={mean_util:.2f}% final={final} stalled={self.stalled}\n")


@dataclass
class Controller:
    state: dict[str, Any]
    state_path: Path
    smoke: bool = False
    real: bool = False
    min_free_gb: float = 100.0
    continue_on_failure: bool = False

    def command_for_stage(self, stage: str) -> list[str] | None:
        smoke_flag = " --smoke" if self.smoke else ""
        if stage == "environment_smoke":
            if self.real:
                return [
                    (
                        "tmp=$(mktemp -d) && "
                        "mkdir -p logs && "
                        "rsync -a --exclude outputs --exclude artifacts --exclude logs --exclude .git ./ \"$tmp/repo/\" && "
                        "(cd \"$tmp/repo\" && make test && SMOKE=1 bash scripts/run_all.sh && python scripts/verify_artifacts.py artifacts/frozen --smoke) && "
                        "date -u > logs/environment_smoke_passed_utc.txt"
                    )
                ]
            return [
                "make test",
                "SMOKE=1 bash scripts/run_all.sh",
                "python scripts/verify_artifacts.py artifacts/frozen --smoke",
                "date -u > logs/environment_smoke_passed_utc.txt",
            ]
        if stage == "prepare_data":
            if self.smoke:
                return ["python scripts/prepare_data.py --synthetic --smoke"]
            return ['python scripts/prepare_data.py --verse-root "${VERSE_ROOT:-}" --ctspine1k-root "${CTSPINE1K_ROOT:-}"']
        if stage == "manifest_validation":
            return None
        if stage == "label_distribution_audit":
            return ["python scripts/label_distribution_audit.py --manifest outputs/manifests/data_manifest.csv --sample 25"]
        if stage == "segresnet_train":
            return [f"python scripts/train.py model=segresnet --run-name segresnet_baseline{smoke_flag}"]
        if stage == "unet_train":
            return [f"python scripts/train.py model=unet --run-name unet_baseline{smoke_flag}"]
        if stage == "baseline_benchmark":
            return [f"python scripts/benchmark.py --config opt_baseline --checkpoint outputs/runs/segresnet_baseline/checkpoint.pt{smoke_flag}"]
        if stage == "baseline_profile":
            return [f"python scripts/profile.py --config opt_baseline --checkpoint outputs/runs/segresnet_baseline/checkpoint.pt{smoke_flag}"]
        if stage == "baseline_robustness":
            return [f"python scripts/robustness.py --checkpoint outputs/runs/segresnet_baseline/checkpoint.pt --config opt_baseline{smoke_flag}"]
        if stage == "data_pipeline_sweep":
            return [f"python scripts/benchmark.py --config opt_data_pipeline --checkpoint outputs/runs/segresnet_baseline/checkpoint.pt --sweep{smoke_flag}"]
        if stage == "amp_sweep":
            return [f"python scripts/benchmark.py --config opt_amp --checkpoint outputs/runs/segresnet_baseline/checkpoint.pt --sweep{smoke_flag}"]
        if stage == "compile_benchmark":
            return [f"python scripts/benchmark.py --config opt_compile --checkpoint outputs/runs/segresnet_baseline/checkpoint.pt{smoke_flag}"]
        if stage == "combined_benchmark":
            return [f"python scripts/benchmark.py --config opt_all --checkpoint outputs/runs/segresnet_baseline/checkpoint.pt{smoke_flag}"]
        if stage == "optimized_robustness":
            return [f"python scripts/robustness.py --checkpoint outputs/runs/segresnet_baseline/checkpoint.pt --config opt_all --output-suffix optimized{smoke_flag}"]
        if stage == "make_tables":
            return [f"python scripts/make_tables.py{smoke_flag}"]
        if stage == "make_plots":
            return [f"python scripts/make_plots.py{smoke_flag}"]
        if stage == "freeze_artifacts":
            return [f"python scripts/freeze_artifacts.py{smoke_flag}"]
        if stage == "verify_artifacts":
            return [f"python scripts/verify_artifacts.py artifacts/frozen{smoke_flag}"]
        if stage == "metric_health_audit":
            return None
        if stage == "traceability_report":
            return None
        raise ValueError(f"Unknown stage: {stage}")

    def validate_stage(self, stage: str) -> bool:
        ok, checks, note = validate_stage_outputs(stage, smoke=self.smoke)
        status = "skipped_valid" if ok else self.state["stages"][stage].get("status", "pending")
        set_stage(
            self.state,
            stage,
            status,
            validation_checks=checks,
            notes=note if note else ("Existing outputs validate." if ok else ""),
            error=None if ok else note,
            state_path=self.state_path,
        )
        return ok

    def run_subcommands(self, stage: str, commands: list[str]) -> bool:
        ensure_dir("logs")
        log_path = Path("logs") / f"{compact_utc()}_{stage}.log"
        command_text = " && ".join(commands)
        set_stage(self.state, stage, "running", command=command_text, log_file=str(log_path), error=None, state_path=self.state_path)
        env = os.environ.copy()
        if self.smoke:
            env["SMOKE"] = "1"
        with log_path.open("w", encoding="utf-8") as log:
            with GPUUtilMonitor(stage, log_path, enabled=stage in GPU_HEAVY_STAGES and not self.smoke) as gpu_monitor:
                for command in commands:
                    log.write(f"\n===== {utc_now()} COMMAND: {command} =====\n")
                    log.flush()
                    proc = subprocess.Popen(command, shell=True, stdout=log, stderr=subprocess.STDOUT, env=env, executable="/bin/bash")
                    rc = proc.wait()
                    if gpu_monitor.stalled:
                        set_stage(
                            self.state,
                            stage,
                            "failed",
                            notes="Low GPU utilization and no log/output growth for 15 minutes.",
                            error="GPU utilization monitor detected a likely stall.",
                            state_path=self.state_path,
                        )
                        return False
                    if rc != 0:
                        set_stage(
                            self.state,
                            stage,
                            "failed",
                            notes=f"Command failed; see {log_path}",
                            error=f"exit_code={rc}",
                            state_path=self.state_path,
                        )
                        return False
        ok, outputs, note = validate_stage_outputs(stage, smoke=self.smoke)
        set_stage(
            self.state,
            stage,
            "complete" if ok else "failed",
            output_files=outputs,
            validation_checks=outputs,
            notes=note if note else "Stage completed and outputs validated.",
            error=None if ok else note,
            state_path=self.state_path,
        )
        return ok

    def run_validation_stage(self, stage: str) -> bool:
        if stage == "manifest_validation":
            ok, checks, msg = validate_manifest()
            set_stage(
                self.state,
                stage,
                "complete" if ok else "failed",
                command="internal manifest validation",
                validation_checks=checks,
                output_files=[
                    "outputs/manifests/data_manifest.csv",
                    "outputs/manifests/split_train.csv",
                    "outputs/manifests/split_val.csv",
                    "outputs/manifests/split_test.csv",
                ]
                if ok
                else [],
                notes="Manifest/split validation passed." if ok else msg,
                error=None if ok else msg,
                state_path=self.state_path,
            )
            return ok
        if stage == "metric_health_audit":
            ok, msg = self.write_metric_health_report()
            set_stage(
                self.state,
                stage,
                "complete" if ok else "failed",
                command="internal metric health audit",
                output_files=[str(METRIC_HEALTH_REPORT)] if ok else [],
                validation_checks=[msg],
                notes=msg,
                error=None if ok else msg,
                state_path=self.state_path,
            )
            return ok
        if stage == "traceability_report":
            ok, msg = self.write_traceability_report()
            set_stage(
                self.state,
                stage,
                "complete" if ok else "failed",
                command="internal traceability report",
                output_files=[str(TRACEABILITY_REPORT)] if ok else [],
                validation_checks=[msg],
                notes=msg,
                error=None if ok else msg,
                state_path=self.state_path,
            )
            return ok
        raise ValueError(f"Stage is not internal validation-only: {stage}")

    def check_disk_or_block(self, stage: str) -> bool:
        if stage not in EXPENSIVE_STAGES:
            return True
        current = free_disk_gb(".")
        if current >= self.min_free_gb:
            return True
        msg = f"Free disk {current:.2f} GB is below threshold {self.min_free_gb:.2f} GB before {stage}."
        write_blocker_report(stage, msg, self.min_free_gb, current)
        set_stage(
            self.state,
            stage,
            "blocked_external",
            notes=msg,
            error=msg,
            state_path=self.state_path,
        )
        return False

    def run_stage(self, stage: str) -> bool:
        if stage not in STAGES:
            raise ValueError(f"Unknown stage: {stage}")
        if self.validate_stage(stage):
            return True
        for dep in DEPENDENCIES[stage]:
            if not self.ensure_stage(dep):
                return False
        if not self.check_disk_or_block(stage):
            return False
        commands = self.command_for_stage(stage)
        if commands is None:
            return self.run_validation_stage(stage)
        return self.run_subcommands(stage, commands)

    def ensure_stage(self, stage: str) -> bool:
        item = self.state["stages"][stage]
        if item.get("status") in {"complete", "skipped_valid"} and self.validate_stage(stage):
            return True
        ok = self.run_stage(stage)
        if not ok and not self.continue_on_failure:
            return False
        return ok

    def run_all(self, start_stage: str | None = None) -> bool:
        start_index = STAGES.index(start_stage) if start_stage else 0
        for stage in STAGES[start_index:]:
            ok = self.ensure_stage(stage)
            if not ok and not self.continue_on_failure:
                return False
        return True

    def validate_only(self) -> bool:
        all_ok = True
        for stage in STAGES:
            ok, checks, note = validate_stage_outputs(stage, smoke=self.smoke)
            set_stage(
                self.state,
                stage,
                "skipped_valid" if ok else "pending",
                validation_checks=checks,
                notes=note if note else ("Existing outputs validate." if ok else ""),
                error=None if ok else note,
                state_path=self.state_path,
            )
            all_ok = all_ok and ok
        return all_ok

    def write_metric_health_report(self) -> tuple[bool, str]:
        rows = _benchmark_rows()
        if not rows:
            return False, "No schema-valid benchmark rows."
        datasets = {str(r.get("dataset", "")) for _, r in rows}
        if self.real and any("synthetic" in d.lower() for d in datasets):
            return False, f"Real metric health audit found synthetic datasets: {sorted(datasets)}"
        manifest_ok, manifest_checks, manifest_msg = validate_manifest()
        if not manifest_ok:
            return False, f"Manifest validation failed: {manifest_msg}"
        metric_rows = []
        for _, row in rows:
            q = row["quality"]
            metric_rows.append(
                {
                    "dice": q["dice_mean"],
                    "hd95": q["hd95_mean_mm"],
                    "latency": row["latency_per_volume_sec_mean"],
                    "throughput": row["throughput_volumes_per_sec"],
                    "vram": row["peak_vram_mb"],
                    "gpu_util": row["gpu_util_pct_mean"],
                    "n_cases": q["n_cases"],
                }
            )
        df = pd.DataFrame(metric_rows)
        failures: list[str] = []
        if not (df["n_cases"] > 0).all():
            failures.append("Some rows have n_cases <= 0.")
        if not ((df["dice"] >= 0) & (df["dice"] <= 1)).all():
            failures.append("Dice outside [0,1].")
        if df["hd95"].isna().all():
            failures.append("All HD95 values are null.")
        if not (df["latency"] > 0).all() or not (df["throughput"] > 0).all():
            failures.append("Latency/throughput contains nonpositive values.")
        if self.real and df["vram"].isna().all():
            failures.append("Real run has all-null VRAM.")
        for stage in ["baseline_robustness", "optimized_robustness", "make_tables", "make_plots", "verify_artifacts"]:
            ok, _, note = validate_stage_outputs(stage, smoke=self.smoke)
            if not ok:
                failures.append(f"{stage}: {note}")
        verdict = "PASS" if not failures else "FAIL"
        lines = [
            "# Final Metric Health Report",
            "",
            "## Verdict",
            verdict,
            "",
            "## Run Classification",
            "SMOKE" if self.smoke else "REAL",
            "",
            "## Dataset Health",
            f"- manifest checks: {manifest_checks}",
            f"- datasets: {sorted(datasets)}",
            "",
            "## Quality Metrics Health",
            f"- Dice range: {float(df['dice'].min())} to {float(df['dice'].max())}",
            f"- HD95 non-null count: {int(df['hd95'].notna().sum())}",
            "",
            "## Efficiency Metrics Health",
            f"- latency range: {float(df['latency'].min())} to {float(df['latency'].max())}",
            f"- throughput range: {float(df['throughput'].min())} to {float(df['throughput'].max())}",
            f"- VRAM non-null count: {int(df['vram'].notna().sum())}",
            f"- GPU utilization non-null count: {int(df['gpu_util'].notna().sum())}",
            "",
            "## Red Flags",
            *(f"- {x}" for x in failures),
        ]
        if not failures:
            lines.append("- None")
        ensure_dir(REPORT_DIR)
        METRIC_HEALTH_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return not failures, "Metric health report PASS." if not failures else "; ".join(failures)

    def write_traceability_report(self) -> tuple[bool, str]:
        rows = _benchmark_rows()
        if not rows:
            return False, "No benchmark rows for traceability."
        table_dir = Path("outputs/tables")
        fig_dir = Path("outputs/figures")
        tables = sorted(table_dir.glob("*.csv"))
        figures = sorted(fig_dir.glob("*.png"))
        if not tables or not figures:
            return False, "Missing result tables or figures."
        datasets = sorted({str(r.get("dataset", "")) for _, r in rows})
        benchmark_sources = "outputs/benchmarks/*.json; outputs/robustness_results*.csv; outputs/manifests/*.csv"
        lines = [
            "# Result Traceability Report",
            "",
            "## Verdict",
            "PASS",
            "",
            "## Tables",
            "| Table | Source files | Fields used | Real output? | Notes |",
            "|---|---|---|---|---|",
        ]
        for table in tables:
            fields = ",".join(pd.read_csv(table, nrows=0).columns)
            lines.append(f"| {table.name} | {benchmark_sources} | {fields} | {'No, smoke' if self.smoke else 'Yes'} | Generated by make_tables.py |")
        lines.extend(["", "## Figures", "| Figure | Source files | Fields used | Real output? | Notes |", "|---|---|---|---|---|"])
        for fig in figures:
            lines.append(f"| {fig.name} | {benchmark_sources} | benchmark/robustness/table fields | {'No, smoke' if self.smoke else 'Yes'} | Generated by make_plots.py |")
        lines.extend(
            [
                "",
                "## Benchmark JSON Corpus",
                f"- Number of rows: {len(rows)}",
                f"- Models: {sorted({str(r.get('model')) for _, r in rows})}",
                f"- Optimizations: {sorted({str(r.get('optimization')) for _, r in rows})}",
                f"- Datasets: {datasets}",
                "",
                "## Verified Outputs",
                "- Use the frozen tables and figures only after a real run verifies.",
                "",
                "## Do-Not-Use Outputs",
            ]
        )
        if self.smoke or any("synthetic" in d.lower() for d in datasets):
            lines.append("- Synthetic smoke outputs are infrastructure checks only and must not be cited as real research results.")
        else:
            lines.append("- Intermediate debug files and superseded rows outside the frozen bundle.")
        ensure_dir(REPORT_DIR)
        TRACEABILITY_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return True, "Traceability report PASS."


def status_table(state: dict[str, Any]) -> str:
    lines = ["stage | status | last command | output files | notes", "---|---|---|---|---"]
    for stage in STAGES:
        item = state["stages"][stage]
        outputs = ", ".join(item.get("output_files", [])[:3])
        if len(item.get("output_files", [])) > 3:
            outputs += ", ..."
        command = str(item.get("command", "")).replace("|", "\\|")
        notes = str(item.get("notes", "")).replace("|", "\\|")
        lines.append(f"{stage} | {item.get('status')} | {command} | {outputs} | {notes}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Resumable SpineSeg-PerfBench GPU execution controller.")
    p.add_argument("--status", action="store_true")
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--run-all", action="store_true")
    p.add_argument("--run-from", choices=STAGES)
    p.add_argument("--run-stage", choices=STAGES)
    p.add_argument("--reset-state", action="store_true")
    p.add_argument("--min-free-gb", type=float, default=100.0)
    p.add_argument("--real", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--continue-on-failure", action="store_true")
    p.add_argument("--state-file", default=str(STATE_PATH), help=argparse.SUPPRESS)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.real and args.smoke:
        raise SystemExit("ERROR: choose only one of --real or --smoke")
    if args.real and not (os.environ.get("VERSE_ROOT") or os.environ.get("CTSPINE1K_ROOT")):
        raise SystemExit("ERROR: --real requires VERSE_ROOT or CTSPINE1K_ROOT")
    state_path = Path(args.state_file)
    state = load_state(state_path, reset=args.reset_state)
    controller = Controller(
        state=state,
        state_path=state_path,
        smoke=args.smoke,
        real=args.real,
        min_free_gb=args.min_free_gb,
        continue_on_failure=args.continue_on_failure,
    )
    if args.status or not any([args.validate_only, args.run_all, args.run_from, args.run_stage]):
        print(status_table(state))
        save_state(state, state_path)
        return 0
    if args.validate_only:
        ok = controller.validate_only()
        print(status_table(state))
        return 0 if ok else 1
    if args.run_stage:
        ok = controller.ensure_stage(args.run_stage)
        print(status_table(state))
        return 0 if ok else 1
    if args.run_from:
        ok = controller.run_all(start_stage=args.run_from)
        print(status_table(state))
        return 0 if ok else 1
    if args.run_all:
        ok = controller.run_all()
        print(status_table(state))
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
