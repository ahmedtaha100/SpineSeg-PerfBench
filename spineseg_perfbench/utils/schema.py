from __future__ import annotations

import math
from typing import Any


TOP_KEYS = {
    "run_id",
    "git_sha",
    "config_hash",
    "timestamp_utc",
    "model",
    "optimization",
    "dataset",
    "split",
    "perturbation",
    "seed",
    "hardware",
    "phase_times_sec",
    "latency_per_volume_sec_mean",
    "latency_per_volume_sec_p50",
    "latency_per_volume_sec_p95",
    "throughput_volumes_per_sec",
    "peak_vram_mb",
    "gpu_util_pct_mean",
    "compile_overhead_sec",
    "steady_state_latency_sec",
    "quality",
    "optimization_metadata",
    "artifacts",
}


def _err(path: str, msg: str) -> ValueError:
    return ValueError(f"Invalid benchmark row at {path}: {msg}")


def _is_float(value: Any, nullable: bool = False) -> bool:
    if value is None:
        return nullable
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return not math.isnan(float(value)) and not math.isinf(float(value))


def _require_keys(d: dict, keys: set[str], path: str) -> None:
    if not isinstance(d, dict):
        raise _err(path, "expected object")
    got = set(d)
    if got != keys:
        raise _err(path, f"expected keys {sorted(keys)}, got {sorted(got)}")


def _is_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int)


def validate_run_row(d: dict) -> None:
    _require_keys(d, TOP_KEYS, "$")
    for k in ["run_id", "git_sha", "config_hash", "timestamp_utc", "model", "optimization", "dataset", "split"]:
        if not isinstance(d[k], str):
            raise _err(k, "expected string")
    if not _is_int(d["seed"]):
        raise _err("seed", "expected int")
    pert = d["perturbation"]
    if pert is not None:
        _require_keys(pert, {"name", "severity"}, "perturbation")
        if not isinstance(pert["name"], str) or not _is_int(pert["severity"]):
            raise _err("perturbation", "expected name string and severity int")
    hw_keys = {
        "gpu_name",
        "cuda_version",
        "driver_version",
        "torch_version",
        "monai_version",
        "platform",
        "cpu",
        "ram_total_gb",
    }
    _require_keys(d["hardware"], hw_keys, "hardware")
    for k in ["gpu_name", "cuda_version", "driver_version"]:
        if d["hardware"][k] is not None and not isinstance(d["hardware"][k], str):
            raise _err(f"hardware.{k}", "expected string or null")
    for k in ["torch_version", "monai_version", "platform", "cpu"]:
        if not isinstance(d["hardware"][k], str):
            raise _err(f"hardware.{k}", "expected string")
    if not _is_float(d["hardware"]["ram_total_gb"], nullable=True):
        raise _err("hardware.ram_total_gb", "expected float or null")
    _require_keys(d["phase_times_sec"], {"preprocess", "dataload", "infer", "total"}, "phase_times_sec")
    for k, v in d["phase_times_sec"].items():
        if not _is_float(v):
            raise _err(f"phase_times_sec.{k}", "expected finite float")
    for k in [
        "latency_per_volume_sec_mean",
        "latency_per_volume_sec_p50",
        "latency_per_volume_sec_p95",
        "throughput_volumes_per_sec",
    ]:
        if not _is_float(d[k]):
            raise _err(k, "expected finite float")
    for k in ["peak_vram_mb", "gpu_util_pct_mean", "compile_overhead_sec", "steady_state_latency_sec"]:
        if not _is_float(d[k], nullable=True):
            raise _err(k, "expected finite float or null")
    _require_keys(d["quality"], {"dice_mean", "dice_std", "hd95_mean_mm", "hd95_std_mm", "n_cases"}, "quality")
    for k in ["dice_mean", "dice_std"]:
        if not _is_float(d["quality"][k]):
            raise _err(f"quality.{k}", "expected finite float")
    for k in ["hd95_mean_mm", "hd95_std_mm"]:
        if not _is_float(d["quality"][k], nullable=True):
            raise _err(f"quality.{k}", "expected finite float or null")
    if not _is_int(d["quality"]["n_cases"]):
        raise _err("quality.n_cases", "expected int")
    _require_keys(
        d["optimization_metadata"],
        {"amp_dtype", "compile_succeeded", "quality_delta_dice_vs_fp32", "quality_delta_hd95_vs_fp32", "notes"},
        "optimization_metadata",
    )
    if d["optimization_metadata"]["amp_dtype"] is not None and not isinstance(d["optimization_metadata"]["amp_dtype"], str):
        raise _err("optimization_metadata.amp_dtype", "expected string or null")
    if d["optimization_metadata"]["compile_succeeded"] is not None and not isinstance(
        d["optimization_metadata"]["compile_succeeded"], bool
    ):
        raise _err("optimization_metadata.compile_succeeded", "expected bool or null")
    for k in ["quality_delta_dice_vs_fp32", "quality_delta_hd95_vs_fp32"]:
        if not _is_float(d["optimization_metadata"][k], nullable=True):
            raise _err(f"optimization_metadata.{k}", "expected float or null")
    if d["optimization_metadata"]["notes"] is not None and not isinstance(d["optimization_metadata"]["notes"], str):
        raise _err("optimization_metadata.notes", "expected string or null")
    _require_keys(d["artifacts"], {"predictions_dir", "profiler_trace"}, "artifacts")
    for k in ["predictions_dir", "profiler_trace"]:
        if d["artifacts"][k] is not None and not isinstance(d["artifacts"][k], str):
            raise _err(f"artifacts.{k}", "expected string or null")
