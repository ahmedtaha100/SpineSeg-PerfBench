from __future__ import annotations

from spineseg_perfbench.utils.schema import validate_run_row


def valid_row():
    return {
        "run_id": "r",
        "git_sha": "abc",
        "config_hash": "hash",
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "model": "segresnet",
        "optimization": "baseline",
        "dataset": "synthetic",
        "split": "test",
        "perturbation": None,
        "seed": 42,
        "hardware": {
            "gpu_name": None,
            "cuda_version": None,
            "driver_version": None,
            "torch_version": "x",
            "monai_version": "x",
            "platform": "x",
            "cpu": "x",
            "ram_total_gb": 1.0,
        },
        "phase_times_sec": {"preprocess": 0.1, "dataload": 0.1, "infer": 0.1, "total": 0.3},
        "latency_per_volume_sec_mean": 0.1,
        "latency_per_volume_sec_p50": 0.1,
        "latency_per_volume_sec_p95": 0.1,
        "throughput_volumes_per_sec": 10.0,
        "peak_vram_mb": None,
        "gpu_util_pct_mean": None,
        "compile_overhead_sec": None,
        "steady_state_latency_sec": 0.1,
        "quality": {"dice_mean": 0.5, "dice_std": 0.0, "hd95_mean_mm": None, "hd95_std_mm": None, "n_cases": 1},
        "optimization_metadata": {
            "amp_dtype": "fp32",
            "compile_succeeded": None,
            "quality_delta_dice_vs_fp32": None,
            "quality_delta_hd95_vs_fp32": None,
            "notes": None,
        },
        "artifacts": {"predictions_dir": None, "profiler_trace": None},
    }


def test_valid_schema_passes():
    validate_run_row(valid_row())


def test_missing_key_fails():
    row = valid_row()
    del row["run_id"]
    try:
        validate_run_row(row)
    except ValueError as exc:
        assert "expected keys" in str(exc)
    else:
        raise AssertionError("schema accepted missing run_id")


def test_schema_rejects_bool_int_fields():
    row = valid_row()
    row["seed"] = True
    try:
        validate_run_row(row)
    except ValueError as exc:
        assert "seed" in str(exc)
    else:
        raise AssertionError("schema accepted bool seed")


def test_schema_rejects_non_object_nested_fields():
    row = valid_row()
    row["hardware"] = None
    try:
        validate_run_row(row)
    except ValueError as exc:
        assert "hardware" in str(exc)
    else:
        raise AssertionError("schema accepted non-object hardware")
