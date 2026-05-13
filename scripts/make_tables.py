#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from spineseg_perfbench.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Regenerate result tables from JSON/CSV outputs.")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--input-dir", default="outputs")
    p.add_argument("--output-dir", default="outputs/tables")
    return p.parse_args()


def _load_rows(input_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((input_dir / "benchmarks").glob("*.json")):
        with path.open() as f:
            row = json.load(f)
        flat = {
            "run_id": row["run_id"],
            "model": row["model"],
            "optimization": row["optimization"],
            "dataset": row["dataset"],
            "split": row["split"],
            "perturbation": "clean" if row["perturbation"] is None else row["perturbation"]["name"],
            "severity": 0 if row["perturbation"] is None else row["perturbation"]["severity"],
            "n_cases": row["quality"]["n_cases"],
            "dice_mean": row["quality"]["dice_mean"],
            "dice_std": row["quality"]["dice_std"],
            "hd95_mean_mm": row["quality"]["hd95_mean_mm"],
            "hd95_std_mm": row["quality"]["hd95_std_mm"],
            "latency_per_volume_sec_mean": row["latency_per_volume_sec_mean"],
            "latency_per_volume_sec_p50": row["latency_per_volume_sec_p50"],
            "latency_per_volume_sec_p95": row["latency_per_volume_sec_p95"],
            "throughput_volumes_per_sec": row["throughput_volumes_per_sec"],
            "peak_vram_mb": row["peak_vram_mb"],
            "gpu_util_pct_mean": row["gpu_util_pct_mean"],
            "preprocess_sec": row["phase_times_sec"]["preprocess"],
            "dataload_sec": row["phase_times_sec"]["dataload"],
            "infer_sec": row["phase_times_sec"]["infer"],
            "total_sec": row["phase_times_sec"]["total"],
            "amp_dtype": row["optimization_metadata"]["amp_dtype"],
            "compile_succeeded": row["optimization_metadata"]["compile_succeeded"],
        }
        rows.append(flat)
    if not rows:
        raise SystemExit("No benchmark JSON rows found")
    return pd.DataFrame(rows)


def _clean_rows(bench: pd.DataFrame) -> pd.DataFrame:
    clean = bench[bench.perturbation.eq("clean")].copy()
    # Robustness scripts emit clean reference rows with run_id=robust_...; keep
    # them out of compact clean/optimization summaries.
    if "run_id" in clean:
        non_robust = clean[~clean.run_id.astype(str).str.startswith("robust_")]
        if len(non_robust):
            clean = non_robust
    return clean


def _select_row(clean: pd.DataFrame, model: str, optimization: str, amp_dtype: str | None = None) -> pd.Series | None:
    rows = clean[(clean.model == model) & (clean.optimization == optimization)]
    if amp_dtype is not None:
        rows = rows[rows.amp_dtype.astype(str).str.lower() == amp_dtype.lower()]
    if rows.empty:
        return None
    return rows.sort_values("latency_per_volume_sec_mean").iloc[0]


def _fmt(value: float, digits: int) -> float:
    return round(float(value), digits)


def _fmt_str(value: float, digits: int) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def _optimization_display(value: str) -> str:
    return "combined" if str(value) == "all" else str(value)


def _compact_rows(clean: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    specs = [
        ("SegResNet baseline", "segresnet", "baseline", None),
        ("Data pipeline best", "segresnet", "data_pipeline", None),
        ("AMP fp16 best", "segresnet", "amp", "fp16"),
        ("AMP bf16", "segresnet", "amp", "bf16"),
        ("torch.compile", "segresnet", "compile", None),
        ("Combined", "segresnet", "all", None),
        ("U-Net baseline", "unet", "baseline", None),
    ]
    selected: list[tuple[str, pd.Series]] = []
    for label, model, opt, dtype in specs:
        row = _select_row(clean, model, opt, dtype)
        if row is not None:
            selected.append((label, row))
    return selected


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    out = ensure_dir(args.output_dir)
    bench = _load_rows(input_dir)
    manifest = pd.read_csv(input_dir / "manifests" / "data_manifest.csv")
    split_rows = []
    for split in ["train", "val", "test"]:
        p = input_dir / "manifests" / f"split_{split}.csv"
        sdf = pd.read_csv(p) if p.exists() else manifest.iloc[0:0]
        split_rows.append({"split": split, "n_cases": len(sdf), "dataset_sources": ",".join(sorted(sdf.dataset_source.unique()))})
    pd.DataFrame(split_rows).to_csv(out / "table1_dataset_split.csv", index=False, lineterminator="\n")

    clean = _clean_rows(bench)
    compact = _compact_rows(clean)
    clean_summary = []
    for label, row in compact:
        if label in {"SegResNet baseline", "Combined", "U-Net baseline"}:
            clean_summary.append(
                {
                    "row": label.replace("Combined", "SegResNet combined"),
                    "n_cases": int(row.n_cases),
                    "dice_mean": _fmt_str(row.dice_mean, 6),
                    "dice_std": _fmt_str(row.dice_std, 6),
                    "hd95_mm": _fmt_str(row.hd95_mean_mm, 3),
                    "latency_s": _fmt_str(row.latency_per_volume_sec_mean, 3),
                    "throughput": _fmt_str(row.throughput_volumes_per_sec, 3),
                }
            )
    pd.DataFrame(clean_summary).to_csv(out / "table2_clean_quality.csv", index=False, lineterminator="\n")

    efficiency = []
    for label, row in compact:
        efficiency.append(
            {
                "row": label,
                "n_cases": int(row.n_cases),
                "lat_mean_s": _fmt_str(row.latency_per_volume_sec_mean, 3),
                "lat_p50_s": _fmt_str(row.latency_per_volume_sec_p50, 3),
                "lat_p95_s": _fmt_str(row.latency_per_volume_sec_p95, 3),
                "throughput": _fmt_str(row.throughput_volumes_per_sec, 3),
                "vram_mb": _fmt_str(row.peak_vram_mb, 1),
                "gpu_util_pct": _fmt_str(row.gpu_util_pct_mean, 1),
            }
        )
    pd.DataFrame(efficiency).to_csv(out / "table3_efficiency_breakdown.csv", index=False, lineterminator="\n")

    baseline = _select_row(clean, "segresnet", "baseline")
    opt_summary = []
    if baseline is not None:
        for label, row in compact:
            if row.model != "segresnet":
                continue
            compile_value = "--"
            if row.optimization in {"compile", "all"}:
                compile_value = "yes" if bool(row.compile_succeeded) else "no"
            opt_summary.append(
                {
                    "row": label,
                    "n_cases": int(row.n_cases),
                    "amp": row.amp_dtype,
                    "compile": compile_value,
                    "lat_s": _fmt_str(row.latency_per_volume_sec_mean, 3),
                    "speedup": f"{float(baseline.latency_per_volume_sec_mean) / float(row.latency_per_volume_sec_mean):.2f}x",
                    "dice_delta": _fmt_str(float(row.dice_mean) - float(baseline.dice_mean), 6),
                    "hd95_delta": _fmt_str(float(row.hd95_mean_mm) - float(baseline.hd95_mean_mm), 3),
                    "vram_mb": _fmt_str(row.peak_vram_mb, 1),
                }
            )
    pd.DataFrame(opt_summary).to_csv(out / "table4_optimization_ablation.csv", index=False, lineterminator="\n")

    robustness_files = sorted(input_dir.glob("robustness_results*.csv"))
    if robustness_files:
        rob = pd.concat([pd.read_csv(p) for p in robustness_files], ignore_index=True)
    else:
        rob = bench[["model", "optimization", "perturbation", "severity", "n_cases", "dice_mean", "hd95_mean_mm", "latency_per_volume_sec_mean"]]
    if "n_cases" not in rob.columns and "run_id" in rob.columns:
        n_cases_by_run = bench.set_index("run_id")["n_cases"].to_dict()
        rob["n_cases"] = rob["run_id"].map(n_cases_by_run)
    if "latency_per_volume_sec_mean" in rob.columns and "latency_s" not in rob.columns:
        rob = rob.rename(columns={"latency_per_volume_sec_mean": "latency_s"})
    rob = rob[rob.perturbation.ne("clean")]
    rob = rob[rob.severity.astype(int).eq(3)]
    rob = rob[["optimization", "perturbation", "severity", "n_cases", "dice_mean", "hd95_mean_mm", "latency_s"]].copy()
    rob["optimization"] = rob["optimization"].map(_optimization_display)
    rob["perturbation"] = rob["perturbation"].astype(str).str.replace("_", r"\_", regex=False)
    rob["severity"] = rob["severity"].astype(int)
    rob["n_cases"] = rob["n_cases"].astype(int)
    rob["dice_mean"] = rob["dice_mean"].map(lambda x: _fmt_str(x, 6))
    rob["hd95_mm"] = rob["hd95_mean_mm"].map(lambda x: _fmt_str(x, 3))
    rob["latency_s"] = rob["latency_s"].map(lambda x: _fmt_str(x, 3))
    rob = rob.drop(columns=["hd95_mean_mm"]).sort_values(["optimization", "perturbation"]).reset_index(drop=True)
    rob = rob[["optimization", "perturbation", "severity", "n_cases", "dice_mean", "hd95_mm", "latency_s"]]
    rob.to_csv(out / "table5_robustness_summary.csv", index=False, lineterminator="\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
