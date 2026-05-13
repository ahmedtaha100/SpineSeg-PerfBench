#!/usr/bin/env python3
"""Build a static PDF summary from artifacts/frozen/wandb_export."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from textwrap import wrap
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


METRICS = [
    "latency_mean_s",
    "latency_p50_s",
    "latency_p95_s",
    "throughput_vol_per_s",
    "gpu_util_pct",
    "vram_peak_mb",
    "dice_mean",
    "hd95_mm",
    "preprocess_s",
    "dataload_s",
    "infer_s",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, default=Path("artifacts/frozen/wandb_export"))
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def load_index(export_dir: Path) -> dict[str, Any]:
    return json.loads((export_dir / "index.json").read_text(encoding="utf-8"))


def load_rows(export_dir: Path, index: dict[str, Any]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for item in index["runs"]:
        run_dir = export_dir / item["path"]
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        run_id = item["run_id"]
        record = {
            "run_id": run_id,
            "logical_run_id": logical_run_id(run_id),
            "run_path": item["path"],
            "name": item["name"],
            "category": item["category"],
            "excluded_from_final_table": bool(item.get("excluded_from_final_table", False)),
            "exclusion_reason": item.get("exclusion_reason"),
            "model": config.get("model"),
            "optimization": config.get("optimization"),
            "amp_dtype": (config.get("optimization_metadata") or {}).get("amp_dtype"),
            "perturbation": config.get("perturbation_name") or config.get("perturbation") or "clean",
            "severity": config.get("severity"),
            "dataset": config.get("dataset"),
            "split": config.get("split"),
            "created_at": metadata.get("created_at"),
            "updated_at": metadata.get("updated_at"),
        }
        for key in METRICS:
            record[key] = summary.get(key)
        records.append(record)
    return pd.DataFrame.from_records(records)


def logical_run_id(run_id: str) -> str:
    """Use the benchmark row token after the timestamp prefix for report de-duplication."""
    if "_test_" in run_id:
        return run_id.rsplit("_test_", 1)[1]
    if run_id.startswith("robust_"):
        return run_id.rsplit("_", 1)[1]
    return run_id


def timestamp_value(row: pd.Series) -> datetime:
    raw = row.get("updated_at") or row.get("created_at") or ""
    if not isinstance(raw, str) or not raw:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)


def deduplicate_clean_rows(clean: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if clean.empty:
        return clean.copy(), {"before": 0, "after": 0, "duplicates": []}
    deduped_rows: list[pd.Series] = []
    duplicates: list[dict[str, Any]] = []
    for logical_id, group in clean.groupby("logical_run_id", sort=False):
        # Match scripts/make_tables.py: compact final tables keep the lowest
        # mean-latency representative for duplicate logical clean rows.
        ordered = group.assign(_timestamp=group.apply(timestamp_value, axis=1)).sort_values(
            ["latency_mean_s", "_timestamp", "run_id"],
            ascending=[True, False, False],
            na_position="last",
        )
        kept = ordered.iloc[0].drop(labels=["_timestamp"])
        deduped_rows.append(kept)
        if len(ordered) > 1:
            dropped = ordered.iloc[1:]["run_id"].tolist()
            duplicates.append(
                {
                    "logical_run_id": logical_id,
                    "kept": kept["run_id"],
                    "dropped": dropped,
                }
            )
    result = pd.DataFrame(deduped_rows).reset_index(drop=True)
    return result, {"before": len(clean), "after": len(result), "duplicates": duplicates}


def display_optimization_value(opt: Any, amp_dtype: Any = "") -> str:
    opt = str(opt or "assets")
    if opt == "all":
        return "combined"
    if opt == "amp":
        dtype = str(amp_dtype or "").lower()
        if dtype in {"fp16", "bf16", "fp32"}:
            return f"amp_{dtype}"
    return opt


def display_optimization(row: pd.Series) -> str:
    return display_optimization_value(row.get("optimization"), row.get("amp_dtype"))


def label_for_row(row: pd.Series) -> str:
    model = str(row.get("model") or row.get("name") or "").replace("_", "-")
    if model == "segresnet":
        model = "SegResNet"
    elif model == "unet":
        model = "3D U-Net"
    opt = display_optimization(row).replace("_", " ")
    run_id = str(row.get("logical_run_id") or row.get("run_id") or row.get("name") or "")
    suffix = run_id[-6:] if len(run_id) > 6 else run_id
    return f"{model} | {opt} | {suffix}"


def add_footer(fig: plt.Figure, text: str) -> None:
    fig.text(0.5, 0.025, text, ha="center", va="bottom", fontsize=8, color="#555555")


def verified_raw_data_pointers(export_dir: Path) -> list[str]:
    bundle_root = export_dir.parent
    repo_root = bundle_root.parent.parent
    pointers: list[str] = []
    if list((bundle_root / "outputs/benchmarks").glob("*.json")):
        ledger = " The run ledger at artifacts/frozen/RUNS.md maps run IDs to JSON files." if (bundle_root / "RUNS.md").exists() else ""
        pointers.append(
            "Headline efficiency, quality, and ablation numbers: "
            f"artifacts/frozen/outputs/benchmarks/*.json.{ledger}"
        )
    if list((bundle_root / "outputs/robustness").glob("*.csv")):
        pointers.append("Robustness sweep numbers: artifacts/frozen/outputs/robustness/*.csv.")
    if (bundle_root / "flops_summary.json").exists():
        pointers.append("FLOPs/MACs and arithmetic-intensity inputs: artifacts/frozen/flops_summary.json.")
    if (bundle_root / "outputs/profiles").exists():
        pointers.append("Profiler operator and phase summaries: artifacts/frozen/outputs/profiles/.")
    if list((bundle_root / "outputs/runs").glob("*/metrics.json")):
        pointers.append("Training metrics: artifacts/frozen/outputs/runs/*/metrics.json.")
    if (export_dir / "runs").exists():
        pointers.append(
            "WandB-side export: artifacts/frozen/wandb_export/runs/<run_id>/ "
            "with config.json, summary.json, history.csv, system_metrics.csv, and metadata.json."
        )
    if (export_dir / "index.json").exists():
        pointers.append("Run-level WandB export index: artifacts/frozen/wandb_export/index.json.")
    verify_artifacts = repo_root / "scripts/verify_artifacts.py"
    verify_github = repo_root / "scripts/verify_github_submission.py"
    if verify_artifacts.exists() and verify_github.exists() and (bundle_root / "checksums.sha256").exists():
        pointers.append(
            "Verification: scripts/verify_artifacts.py artifacts/frozen, "
            "scripts/verify_github_submission.py artifacts/frozen, and "
            "artifacts/frozen/checksums.sha256."
        )
    return pointers


def training_model_label(path: Path) -> str:
    name = path.parent.name.replace("_baseline", "")
    if name == "segresnet":
        return "SegResNet"
    if name == "unet":
        return "3D U-Net"
    return name.replace("_", " ")


def load_training_metrics(export_dir: Path) -> pd.DataFrame:
    bundle_root = export_dir.parent
    records: list[dict[str, Any]] = []
    for path in sorted((bundle_root / "outputs/runs").glob("*/metrics.json")):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        phases = metrics.get("phase_times_sec") or {}
        total_s = phases.get("total")
        dataload_s = phases.get("dataload")
        train_s = phases.get("train")
        dataload_pct = None
        if isinstance(total_s, (int, float)) and total_s:
            dataload_pct = 100.0 * float(dataload_s or 0.0) / float(total_s)
        records.append(
            {
                "model": training_model_label(path),
                "source": path.relative_to(bundle_root).as_posix(),
                "steps": metrics.get("steps"),
                "total_s": total_s,
                "dataload_s": dataload_s,
                "train_s": train_s,
                "dataload_pct": dataload_pct,
                "steps_per_s": metrics.get("throughput_steps_per_sec"),
                "train_loss": metrics.get("train_loss"),
                "val_dice": metrics.get("val_dice"),
                "train_cases_used": metrics.get("train_cases_used"),
                "train_case_limit": metrics.get("train_case_limit"),
            }
        )
    return pd.DataFrame.from_records(records)


def page_title(pdf: PdfPages, index: dict[str, Any], rows: pd.DataFrame, export_dir: Path) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    counts = index.get("category_counts", {})
    title = "SpineSeg-PerfBench WandB Export Snapshot"
    project_url = index.get("project_url") or f"https://wandb.ai/{index.get('entity')}/{index.get('project')}"
    lines = [
        "Static PDF generated from the checked-in WandB API export.",
        f"Project: {index.get('entity')}/{index.get('project')}",
        f"Runs exported: {index.get('run_count')} total; "
        f"{counts.get('clean_inference', 0)} clean inference, "
        f"{counts.get('robustness', 0)} robustness, "
        f"{counts.get('training', 0)} training, "
        f"{counts.get('other', 0)} other.",
        "",
        f"The source WandB project URL is {project_url}; it is hosted under a team entity "
        "that requires authentication. The reviewer-accessible static export is "
        "artifacts/frozen/wandb_export/ (this PDF plus adjacent per-run CSV/JSON files). "
        "The frozen benchmark JSON/CSV files remain the source of truth for reported values, "
        "and this static export is the review artifact for the WandB visualization layer.",
    ]
    ax.text(0.06, 0.90, title, fontsize=23, fontweight="bold", va="top")
    y = 0.79
    for line in lines:
        wrapped = wrap(line, 110) or [""]
        for part in wrapped:
            ax.text(0.06, y, part, fontsize=10.5, va="top")
            y -= 0.033
        y -= 0.006
    ax.text(0.06, 0.50, "Export contents", fontsize=13.5, fontweight="bold", va="top")
    contents = [
        "index.json: run-level export index and summary metrics",
        "runs/<run_id>/config.json: run configuration",
        "runs/<run_id>/summary.json: final scalar summary",
        "runs/<run_id>/history.csv: scalar history rows",
        "runs/<run_id>/system_metrics.csv: system metrics stream when available",
        "runs/<run_id>/metadata.json: run identity, tags, state, timestamps, and original WandB URL",
    ]
    y = 0.455
    for item in contents:
        for part in wrap(f"- {item}", 58):
            ax.text(0.075, y, part, fontsize=7.8, va="top")
            y -= 0.020
        y -= 0.004

    ax.text(0.52, 0.50, "Where to find the raw data", fontsize=13.5, fontweight="bold", va="top")
    raw_intro = "All manuscript numbers can be cross-checked against these frozen artifacts:"
    ax.text(0.52, 0.462, raw_intro, fontsize=7.8, va="top")
    y = 0.432
    for item in verified_raw_data_pointers(export_dir):
        for part in wrap(f"- {item}", 72):
            ax.text(0.535, y, part, fontsize=7.1, va="top")
            y -= 0.017
        y -= 0.004
    add_footer(fig, "Generated from artifacts/frozen/wandb_export/")
    pdf.savefig(fig)
    plt.close(fig)


def page_training_metrics(pdf: PdfPages, training: pd.DataFrame) -> None:
    training = training.sort_values("model").reset_index(drop=True)
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("Frozen Training Metrics", fontsize=18, fontweight="bold", y=0.94)
    ax_note = fig.add_axes([0.07, 0.78, 0.86, 0.11])
    ax_note.axis("off")
    note = (
        "Frozen training metrics - not live WandB runs. Live replay requires authenticated access. "
        "These rows come from artifacts/frozen/outputs/runs/*/metrics.json and are included here "
        "so reviewers can inspect the retained training wall-clock and quality signals without "
        "WandB authentication."
    )
    y = 1.0
    for part in wrap(note, 130):
        ax_note.text(0, y, part, fontsize=9.5, va="top")
        y -= 0.28

    ax_table = fig.add_axes([0.06, 0.49, 0.88, 0.24])
    ax_table.axis("off")
    display = training[
        [
            "model",
            "steps",
            "total_s",
            "dataload_s",
            "dataload_pct",
            "train_s",
            "steps_per_s",
            "train_loss",
            "val_dice",
        ]
    ].copy()
    display = display.rename(
        columns={
            "model": "Model",
            "steps": "Steps",
            "total_s": "Total s",
            "dataload_s": "Data load s",
            "dataload_pct": "Data load %",
            "train_s": "Train s",
            "steps_per_s": "Steps/s",
            "train_loss": "Train loss",
            "val_dice": "Val Dice",
        }
    )
    for col in display.columns:
        if col == "Model":
            continue
        if col == "Steps":
            display[col] = pd.to_numeric(display[col], errors="coerce").map(
                lambda x: "" if pd.isna(x) else f"{int(x)}"
            )
        elif col == "Val Dice":
            display[col] = pd.to_numeric(display[col], errors="coerce").map(
                lambda x: "" if pd.isna(x) else f"{x:.2e}"
            )
        else:
            display[col] = pd.to_numeric(display[col], errors="coerce").map(
                lambda x: "" if pd.isna(x) else f"{x:.3f}"
            )
    table = ax_table.table(cellText=display.values, colLabels=display.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 1.35)

    ax_bar = fig.add_axes([0.16, 0.15, 0.76, 0.27])
    dataload = pd.to_numeric(training["dataload_s"], errors="coerce").fillna(0.0)
    total = pd.to_numeric(training["total_s"], errors="coerce").fillna(0.0)
    remaining = (total - dataload).clip(lower=0.0)
    y_pos = range(len(training))
    ax_bar.barh(y_pos, dataload, color="#d08c60", label="Data loading")
    ax_bar.barh(y_pos, remaining, left=dataload, color="#5b8cc0", label="Remaining train time")
    ax_bar.set_yticks(list(y_pos))
    ax_bar.set_yticklabels(training["model"], fontsize=9)
    ax_bar.set_xlabel("Wall-clock seconds")
    ax_bar.set_title("Training wall-clock decomposition from frozen metrics", fontsize=11)
    ax_bar.grid(axis="x", alpha=0.25)
    ax_bar.legend(loc="lower right", fontsize=8)
    add_footer(fig, "Training metrics are static frozen artifacts, not live WandB replay rows.")
    pdf.savefig(fig)
    plt.close(fig)


def page_clean_table(pdf: PdfPages, clean: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.set_title("Clean Inference Runs", fontsize=18, fontweight="bold", pad=18)
    cols = [
        "model",
        "optimization",
        "logical_run_id",
        "latency_mean_s",
        "latency_p95_s",
        "throughput_vol_per_s",
        "gpu_util_pct",
        "vram_peak_mb",
        "dice_mean",
        "hd95_mm",
    ]
    table_df = clean[cols + ["amp_dtype"]].copy().sort_values(["model", "latency_mean_s"], na_position="last")
    display = table_df[cols].rename(
        columns={
            "model": "Model",
            "optimization": "Optimization",
            "logical_run_id": "Run key",
            "latency_mean_s": "Mean s/vol",
            "latency_p95_s": "P95 s/vol",
            "throughput_vol_per_s": "Vol/s",
            "gpu_util_pct": "GPU %",
            "vram_peak_mb": "VRAM MB",
            "dice_mean": "Dice",
            "hd95_mm": "HD95 mm",
        }
    )
    for col in display.columns:
        if col in {"Model", "Optimization", "Run key"}:
            display[col] = display[col].fillna("")
            if col == "Optimization":
                display[col] = table_df.apply(display_optimization, axis=1)
            if col == "Run key":
                display[col] = display[col].map(lambda x: x[-12:] if len(str(x)) > 12 else x)
        elif col == "Dice":
            display[col] = pd.to_numeric(display[col], errors="coerce").map(
                lambda x: "" if pd.isna(x) else f"{x:.2e}"
            )
        else:
            display[col] = pd.to_numeric(display[col], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    table = ax.table(cellText=display.values, colLabels=display.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.28)
    add_footer(fig, "Clean rows exported from WandB replay of frozen benchmark artifacts.")
    pdf.savefig(fig)
    plt.close(fig)


def page_latency(pdf: PdfPages, clean: pd.DataFrame) -> None:
    plot_df = clean.dropna(subset=["latency_mean_s"]).copy().sort_values("latency_mean_s")
    fig, ax = plt.subplots(figsize=(11, 8.5))
    labels = [label_for_row(row) for _, row in plot_df.iterrows()]
    colors = ["#376795" if model == "segresnet" else "#d95f02" for model in plot_df["model"].fillna("")]
    y_pos = range(len(plot_df))
    ax.barh(y_pos, plot_df["latency_mean_s"], color=colors)
    ax.set_title("Clean Inference Latency by Configuration", fontsize=18, fontweight="bold")
    ax.set_xlabel("Mean latency (s/volume)")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    for idx, value in enumerate(plot_df["latency_mean_s"]):
        ax.text(value + 0.04, idx, f"{value:.2f}", ha="left", va="center", fontsize=8)
    fig.subplots_adjust(left=0.25, right=0.94, top=0.88, bottom=0.12)
    add_footer(fig, "Lower is better. Values are WandB summaries replayed from frozen JSON rows.")
    pdf.savefig(fig)
    plt.close(fig)


def page_efficiency(pdf: PdfPages, clean: pd.DataFrame) -> None:
    plot_df = clean.dropna(subset=["latency_mean_s"]).copy().sort_values("latency_mean_s")
    labels = [label_for_row(row) for _, row in plot_df.iterrows()]
    y_pos = range(len(plot_df))
    fig, axes = plt.subplots(1, 2, figsize=(11, 8.5), sharey=True)
    fig.suptitle("GPU Utilization and Peak VRAM", fontsize=18, fontweight="bold", y=0.94)
    axes[0].barh(y_pos, plot_df["gpu_util_pct"], color="#4c956c")
    axes[0].set_xlabel("GPU util (%)")
    axes[0].set_yticks(list(y_pos))
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].grid(axis="x", alpha=0.25)
    axes[1].barh(y_pos, plot_df["vram_peak_mb"] / 1024.0, color="#8e6c8a")
    axes[1].set_xlabel("Peak VRAM (GiB)")
    axes[1].grid(axis="x", alpha=0.25)
    fig.subplots_adjust(left=0.25, right=0.95, top=0.87, bottom=0.12, wspace=0.18)
    add_footer(fig, "System metrics are exported from WandB summaries when present.")
    pdf.savefig(fig)
    plt.close(fig)


def page_phase_times(pdf: PdfPages, clean: pd.DataFrame) -> None:
    plot_df = clean.dropna(subset=["preprocess_s", "dataload_s", "infer_s"], how="all").copy()
    plot_df = plot_df.sort_values("latency_mean_s")
    labels = [label_for_row(row) for _, row in plot_df.iterrows()]
    fig, ax = plt.subplots(figsize=(11, 8.5))
    bottom = pd.Series([0.0] * len(plot_df))
    colors = {"preprocess_s": "#5b8cc0", "dataload_s": "#d08c60", "infer_s": "#6aa84f"}
    names = {"preprocess_s": "Preprocess", "dataload_s": "Data load", "infer_s": "Inference"}
    y_pos = range(len(plot_df))
    for key in ["preprocess_s", "dataload_s", "infer_s"]:
        vals = pd.to_numeric(plot_df[key], errors="coerce").fillna(0)
        ax.barh(y_pos, vals, left=bottom, label=names[key], color=colors[key])
        bottom += vals.reset_index(drop=True)
    ax.set_title("Phase-Time Breakdown", fontsize=18, fontweight="bold")
    ax.set_xlabel("Seconds")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.subplots_adjust(left=0.25, right=0.95, top=0.88, bottom=0.12)
    add_footer(fig, "Phase timings are aggregate run summaries, not per-volume traces.")
    pdf.savefig(fig)
    plt.close(fig)


def page_robustness(pdf: PdfPages, robust: pd.DataFrame) -> None:
    robust = robust.copy()
    robust["severity"] = pd.to_numeric(robust["severity"], errors="coerce")
    robust = robust.dropna(subset=["severity"])
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)
    for (perturbation, optimization), group in robust.groupby(["perturbation", "optimization"], dropna=False):
        group = group.sort_values("severity")
        label = f"{perturbation} / {display_optimization_value(optimization)}"
        axes[0].plot(group["severity"], group["dice_mean"], marker="o", linewidth=1.5, label=label)
        axes[1].plot(group["severity"], group["latency_mean_s"], marker="o", linewidth=1.5, label=label)
    axes[0].set_title("Robustness Sweeps", fontsize=18, fontweight="bold")
    axes[0].set_ylabel("Dice mean")
    axes[1].set_ylabel("Mean latency (s/volume)")
    axes[1].set_xlabel("Severity")
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7, ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    fig.subplots_adjust(left=0.09, right=0.72, top=0.90, bottom=0.12, hspace=0.30)
    add_footer(
        fig,
        "Severity-0 points are robustness-sweep clean-reference rows on the robustness subset; "
        "do not compare them directly to manuscript 8-case clean latencies.",
    )
    pdf.savefig(fig)
    plt.close(fig)


def page_inventory(
    pdf: PdfPages,
    index: dict[str, Any],
    rows: pd.DataFrame,
    export_dir: Path,
    dedupe_stats: dict[str, Any],
) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.07, 0.90, "Export Inventory", fontsize=20, fontweight="bold", va="top")
    file_count = sum(1 for _ in export_dir.rglob("*") if _.is_file())
    excluded_clean = rows[(rows["category"] == "clean_inference") & (rows["excluded_from_final_table"])]
    final_clean_count = dedupe_stats["after"] - excluded_clean["logical_run_id"].nunique()
    lines = [
        f"Run directories: {len(index.get('runs', []))}",
        f"Export files per run: config.json, summary.json, history.csv, system_metrics.csv, metadata.json",
        f"Total exported run rows in index: {index.get('run_count')}",
        f"Clean inference exports: {dedupe_stats['before']} raw -> "
        f"{dedupe_stats['after']} unique logical IDs after duplicate collapse -> "
        f"{final_clean_count} "
        f"rows in final rendered clean table after AMP-fp32 exclusion.",
        f"Failures recorded in index: {index.get('failed_count')}",
        f"Current export file count: {file_count}",
        f"Project URL recorded for traceability: {index.get('project_url')}",
        "",
        "This PDF is generated from the same static export files bundled for review. "
        "It is intentionally non-interactive; reviewers who want exact values should inspect "
        "index.json and the per-run CSV/JSON files.",
    ]
    y = 0.82
    for line in lines:
        for part in wrap(line, 110) or [""]:
            ax.text(0.07, y, part, fontsize=10.2, va="top")
            y -= 0.031
        y -= 0.004
    if dedupe_stats.get("duplicates"):
        ax.text(0.07, y, "Duplicate clean logical run IDs collapsed", fontsize=12.5, fontweight="bold", va="top")
        y -= 0.035
        for item in dedupe_stats["duplicates"]:
            text = (
                f"{item['logical_run_id']}: kept {item['kept']}; "
                f"dropped {', '.join(item['dropped'])}"
            )
            for part in wrap(text, 112):
                ax.text(0.09, y, part, fontsize=6.8, va="top")
                y -= 0.018
            y -= 0.003
    excluded = excluded_clean
    if not excluded.empty:
        ax.text(0.07, y, "Clean rows excluded from final rendered table", fontsize=12.5, fontweight="bold", va="top")
        y -= 0.035
        ids = ", ".join(excluded["run_id"].astype(str).tolist())
        text = f"{len(excluded)} rows are retained in the raw export but excluded from the final clean table: {ids}."
        for part in wrap(text, 112):
            ax.text(0.09, y, part, fontsize=6.8, va="top")
            y -= 0.018
        y -= 0.003
        ax.text(
            0.09,
            y,
            "Reason: AMP-labeled metadata has effective amp_dtype = fp32.",
            fontsize=7.8,
            va="top",
            fontfamily="DejaVu Sans Mono",
        )
        y -= 0.026
    add_footer(fig, "Generated static PDF snapshot; frozen artifacts remain canonical.")
    pdf.savefig(fig)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    export_dir = args.export_dir
    output = args.output or export_dir / "wandb_report.pdf"
    index = load_index(export_dir)
    rows = load_rows(export_dir, index)
    clean = rows[rows["category"] == "clean_inference"].copy()
    clean_deduped, dedupe_stats = deduplicate_clean_rows(clean)
    clean_rendered = clean_deduped[~clean_deduped["excluded_from_final_table"]].copy()
    robust = rows[rows["category"] == "robustness"].copy()
    training = load_training_metrics(export_dir)
    with PdfPages(output) as pdf:
        page_title(pdf, index, rows, export_dir)
        if not training.empty:
            page_training_metrics(pdf, training)
        if not clean_rendered.empty:
            page_clean_table(pdf, clean_rendered)
            page_latency(pdf, clean_rendered)
            page_efficiency(pdf, clean_rendered)
            page_phase_times(pdf, clean_rendered)
        if not robust.empty:
            page_robustness(pdf, robust)
        page_inventory(pdf, index, rows, export_dir, dedupe_stats)
    print(f"wrote {output} ({output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
