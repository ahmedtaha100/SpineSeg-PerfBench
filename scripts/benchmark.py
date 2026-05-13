#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import itertools
import re
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer import run_inference  # noqa: E402

from spineseg_perfbench.config import config_hash, load_config
from spineseg_perfbench.profiling.vram import GPUUtilizationSampler, peak_vram_mb, reset_peak_vram, sample_gpu_utilization
from spineseg_perfbench.utils.hardware import collect_hardware_metadata
from spineseg_perfbench.utils.hashing import git_sha, stable_hash
from spineseg_perfbench.utils.io import append_run_ledger, ensure_dir, read_json, write_json
from spineseg_perfbench.utils.schema import validate_run_row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a SpineSeg-PerfBench benchmark and emit schema-valid JSON rows.")
    p.add_argument("--config", default="opt_baseline")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--run-id", default=None)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def _sweep_cfgs(cfg: dict, smoke: bool) -> list[dict]:
    settings = cfg.get("optimization_settings", {})
    if not settings or "sweep" not in settings:
        return [cfg]
    sweep = settings["sweep"]
    keys = list(sweep)
    values = [sweep[k] for k in keys]
    combos = list(itertools.product(*values))
    if smoke and cfg.get("optimization") == "data_pipeline":
        combos = combos[:2]
    cfgs = []
    for combo in combos:
        c = deepcopy(cfg)
        c["optimization_settings"] = deepcopy(settings)
        for k, v in zip(keys, combo, strict=True):
            c["optimization_settings"][k] = v
        cfgs.append(c)
    return cfgs


def _quality_deltas(row: dict, reference_quality: dict | None) -> tuple[float | None, float | None, str | None]:
    if reference_quality is None:
        return None, None, None
    dice_delta = row["quality"]["dice_mean"] - reference_quality["dice_mean"]
    hd_ref = reference_quality.get("hd95_mean_mm")
    hd_cur = row["quality"].get("hd95_mean_mm")
    hd_delta = None if hd_ref is None or hd_cur is None else hd_cur - hd_ref
    note = "WARNING: absolute Dice drift > 0.005" if abs(dice_delta) > 0.005 else None
    return float(dice_delta), None if hd_delta is None else float(hd_delta), note


def safe_run_id(value: str, max_length: int = 180) -> str:
    run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._-")
    if not run_id:
        raise ValueError(f"run_id is empty after sanitization: {value!r}")
    return run_id[:max_length]


def unique_run_id(run_id: str, out_dir: Path) -> str:
    base = safe_run_id(run_id)
    candidate = base
    for attempt in range(100):
        if not (out_dir / f"{candidate}.json").exists():
            return candidate
        suffix = stable_hash({"base": base, "attempt": attempt, "time_ns": time.time_ns()})
        prefix = base[: max(1, 180 - len(suffix) - 1)]
        candidate = safe_run_id(f"{prefix}_{suffix}")
    raise RuntimeError(f"Could not allocate unique run_id for {base}")


def _dataset_label(cfg: dict, result: dict) -> str:
    configured = str(cfg.get("dataset", "unknown"))
    manifest = cfg.get("manifest", {})
    split = str(result.get("split", ""))
    manifest_path = manifest.get(split) or manifest.get("all")
    if configured.lower() != "synthetic" or not manifest_path:
        return configured
    try:
        with Path(manifest_path).open(newline="", encoding="utf-8") as f:
            sources = {
                str(row.get("dataset_source", "")).strip()
                for row in csv.DictReader(f)
                if str(row.get("dataset_source", "")).strip()
            }
    except (OSError, UnicodeDecodeError):
        return configured
    if len(sources) == 1:
        return next(iter(sources))
    if len(sources) > 1:
        return "+".join(sorted(sources))
    return configured


def make_row(
    cfg: dict,
    result: dict,
    run_id: str,
    perturbation: dict | None = None,
    gpu_util_pct_mean: float | None = None,
) -> dict:
    phase_times = result["phase_times_sec"]
    row = {
        "run_id": run_id,
        "git_sha": git_sha(),
        "config_hash": config_hash(cfg),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": result["model_name"],
        "optimization": str(cfg.get("optimization", "baseline")),
        "dataset": _dataset_label(cfg, result),
        "split": result["split"],
        "perturbation": perturbation,
        "seed": int(cfg.get("seed", 42)),
        "hardware": collect_hardware_metadata(),
        "phase_times_sec": phase_times,
        "latency_per_volume_sec_mean": result["latency_mean"],
        "latency_per_volume_sec_p50": result["latency_p50"],
        "latency_per_volume_sec_p95": result["latency_p95"],
        "throughput_volumes_per_sec": result["throughput"],
        "peak_vram_mb": peak_vram_mb(),
        "gpu_util_pct_mean": gpu_util_pct_mean if gpu_util_pct_mean is not None else sample_gpu_utilization(),
        "compile_overhead_sec": result["compile_overhead_sec"],
        "steady_state_latency_sec": result["steady_state_latency_sec"],
        "quality": result["quality"],
        "optimization_metadata": {
            "amp_dtype": result["amp_dtype"],
            "compile_succeeded": result["compile_succeeded"],
            "quality_delta_dice_vs_fp32": None,
            "quality_delta_hd95_vs_fp32": None,
            "notes": result["notes"],
        },
        "artifacts": {
            "predictions_dir": result["predictions_dir"],
            "profiler_trace": None,
        },
    }
    return row


def run_one(cfg: dict, checkpoint: str, split: str, smoke: bool, run_id: str | None = None, reference_quality: dict | None = None) -> Path:
    reset_peak_vram()
    settings = cfg.get("optimization_settings", {})
    split_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in split)
    raw_rid = run_id or f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{split_id}_{stable_hash({'cfg': cfg, 'checkpoint': checkpoint, 'split': split})}"
    out_dir = ensure_dir("outputs/benchmarks")
    rid = unique_run_id(raw_rid, out_dir)
    with GPUUtilizationSampler() as gpu_sampler:
        result = run_inference(
            checkpoint,
            cfg,
            split=split,
            save_predictions=False,
            amp_dtype=settings.get("amp_dtype", "fp32"),
            compile_enabled=bool(settings.get("compile", False)),
            smoke=smoke,
        )
    row = make_row(cfg, result, rid, gpu_util_pct_mean=gpu_sampler.mean())
    dd, hd, drift_note = _quality_deltas(row, reference_quality)
    row["optimization_metadata"]["quality_delta_dice_vs_fp32"] = dd
    row["optimization_metadata"]["quality_delta_hd95_vs_fp32"] = hd
    if drift_note:
        existing = row["optimization_metadata"]["notes"]
        row["optimization_metadata"]["notes"] = "; ".join(x for x in [existing, drift_note] if x)
    validate_run_row(row)
    out = out_dir / f"{rid}.json"
    write_json(out, row)
    pert = "clean" if row["perturbation"] is None else f"{row['perturbation']['name']}:{row['perturbation']['severity']}"
    append_run_ledger(
        row["run_id"],
        row["git_sha"],
        row["config_hash"],
        str(out),
        row["model"],
        row["optimization"],
        pert,
        f"dice={row['quality']['dice_mean']:.4f}, latency={row['latency_per_volume_sec_mean']:.4f}s",
    )
    print(out)
    return out


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config, smoke=args.smoke)
    cfgs = _sweep_cfgs(cfg, args.smoke) if args.sweep else [cfg]
    cfgs = sorted(cfgs, key=lambda c: str(c.get("optimization_settings", {}).get("amp_dtype", "fp32")).lower() != "fp32")
    reference_quality = None
    for c in cfgs:
        rid = args.run_id if args.run_id and len(cfgs) == 1 else None
        path = run_one(c, args.checkpoint, args.split, args.smoke, run_id=rid, reference_quality=reference_quality)
        row = read_json(path)
        if reference_quality is None and str(row["optimization_metadata"].get("amp_dtype", "fp32")).lower() == "fp32":
            reference_quality = row["quality"]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
