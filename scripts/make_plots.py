#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from spineseg_perfbench.utils.io import ensure_dir


def save_figure(fig, path: Path) -> None:
    fig.savefig(path, dpi=160, metadata={})


def _set_axis_padding(ax, x_values, y_values, x_frac: float = 0.08, y_frac: float = 0.18) -> None:
    x_min, x_max = float(min(x_values)), float(max(x_values))
    y_min, y_max = float(min(y_values)), float(max(y_values))
    x_pad = max((x_max - x_min) * x_frac, 0.1)
    y_pad = max((y_max - y_min) * y_frac, 1e-5)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)


def _annotate_points(ax, rows: pd.DataFrame, x_col: str, y_col: str, offsets: dict[str, tuple[int, int, str, str]]) -> None:
    for _, row in rows.iterrows():
        label = row.get("plot_label", row["optimization"])
        dx, dy, ha, va = offsets.get(label, (6, 6, "left", "bottom"))
        ax.annotate(
            label,
            (row[x_col], row[y_col]),
            textcoords="offset points",
            xytext=(dx, dy),
            ha=ha,
            va=va,
            fontsize=6.5,
            arrowprops={"arrowstyle": "-", "color": "0.45", "lw": 0.45, "shrinkA": 0, "shrinkB": 3},
            bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.78},
        )


def _format_dice_axis(ax) -> None:
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(formatter)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Regenerate result plots from JSON/CSV outputs.")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--input-dir", default="outputs")
    p.add_argument("--output-dir", default="outputs/figures")
    return p.parse_args()


def _bench(input_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((input_dir / "benchmarks").glob("*.json")):
        row = json.loads(path.read_text())
        rows.append(
            {
                "run_id": row["run_id"],
                "model": row["model"],
                "optimization": row["optimization"],
                "n_cases": row["quality"]["n_cases"],
                "amp_dtype": row["optimization_metadata"]["amp_dtype"],
                "perturbation": "clean" if row["perturbation"] is None else row["perturbation"]["name"],
                "severity": 0 if row["perturbation"] is None else row["perturbation"]["severity"],
                "dice_mean": row["quality"]["dice_mean"],
                "latency": row["latency_per_volume_sec_mean"],
                "peak_vram_mb": row["peak_vram_mb"] or 0.0,
                "preprocess": row["phase_times_sec"]["preprocess"],
                "dataload": row["phase_times_sec"]["dataload"],
                "infer": row["phase_times_sec"]["infer"],
            }
        )
    if not rows:
        raise SystemExit("No benchmark JSON rows found")
    return pd.DataFrame(rows)


def _clean_rows(df: pd.DataFrame) -> pd.DataFrame:
    clean = df[df.perturbation.eq("clean")].copy()
    non_robust = clean[~clean.run_id.astype(str).str.startswith("robust_")]
    return non_robust if len(non_robust) else clean


def _select_row(clean: pd.DataFrame, model: str, optimization: str, amp_dtype: str | None = None) -> pd.Series | None:
    rows = clean[(clean.model == model) & (clean.optimization == optimization)]
    if amp_dtype is not None:
        rows = rows[rows.amp_dtype.astype(str).str.lower() == amp_dtype.lower()]
    if rows.empty:
        return None
    return rows.sort_values("latency").iloc[0]


def _compact_clean_rows(clean: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("SegResNet baseline", "segresnet", "baseline", None),
        ("Data pipeline best", "segresnet", "data_pipeline", None),
        ("AMP fp16 best", "segresnet", "amp", "fp16"),
        ("AMP bf16", "segresnet", "amp", "bf16"),
        ("torch.compile", "segresnet", "compile", None),
        ("Combined", "segresnet", "all", None),
        ("U-Net baseline", "unet", "baseline", None),
    ]
    rows = []
    for label, model, optimization, amp_dtype in specs:
        row = _select_row(clean, model, optimization, amp_dtype)
        if row is not None:
            item = row.copy()
            item["plot_label"] = label
            rows.append(item)
    return pd.DataFrame(rows) if rows else clean


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    out = ensure_dir(args.output_dir)
    df = _bench(input_dir)
    clean = _clean_rows(df)
    compact_clean = _compact_clean_rows(clean)

    fig, ax = plt.subplots(figsize=(8, 4))
    steps = ["Prepare data", "Train", "Benchmark", "Robustness", "Tables/plots", "Freeze/verify"]
    ax.plot(range(len(steps)), [1] * len(steps), marker="o")
    ax.set_xticks(range(len(steps)), steps, rotation=25, ha="right")
    ax.set_yticks([])
    ax.set_title("SpineSeg-PerfBench pipeline" + (" (synthetic smoke)" if args.smoke else ""))
    fig.tight_layout()
    save_figure(fig, out / "fig1_pipeline_overview.png")
    plt.close(fig)

    phases = clean[["preprocess", "dataload", "infer"]].sum()
    labels = ["Preprocess", "Dataload", "Infer"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, phases.values, color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_ylabel("Total time across subset rows (s)", fontsize=11)
    ax.set_title("Phase time breakdown (subset execution)", fontsize=12)
    ax.tick_params(axis="both", labelsize=10)
    ymax = max(float(phases.max()), 1.0)
    ax.set_ylim(0, ymax * 1.18)
    for bar, value in zip(bars, phases.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * 0.025,
            f"{value:.1f}s",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.text(
        0.03,
        0.95,
        "Training phase: 97--98% data-loading-bound\n(Section VIII.B)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.65", "alpha": 0.88},
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, out / "fig2_phase_time_breakdown.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for model, grp in compact_clean.groupby("model"):
        ax.scatter(grp["latency"], grp["dice_mean"], label=model, alpha=0.8)
    _set_axis_padding(ax, compact_clean["latency"], compact_clean["dice_mean"], x_frac=0.08, y_frac=0.22)
    _annotate_points(
        ax,
        compact_clean,
        "latency",
        "dice_mean",
        {
            "SegResNet baseline": (-18, 22, "right", "bottom"),
            "Data pipeline best": (-20, -18, "right", "top"),
            "AMP fp16 best": (16, 22, "left", "bottom"),
            "AMP bf16": (-8, -22, "right", "top"),
            "torch.compile": (8, 18, "left", "bottom"),
            "Combined": (-8, -20, "right", "top"),
            "U-Net baseline": (6, 8, "left", "bottom"),
        },
    )
    ax.set_xlabel("Latency per volume (s)")
    ax.set_ylabel("Dice")
    ax.set_ylim(1.8e-3, 3.5e-3)
    ax.set_yticks([2.0e-3, 2.5e-3, 3.0e-3, 3.5e-3])
    _format_dice_axis(ax)
    ax.set_title("Latency by configuration (Dice constant within noise floor)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    save_figure(fig, out / "fig3_accuracy_latency_pareto.png")
    plt.close(fig)

    rob_files = sorted(input_dir.glob("robustness_results*.csv"))
    rob = pd.concat([pd.read_csv(p) for p in rob_files], ignore_index=True) if rob_files else df
    fig, ax = plt.subplots(figsize=(7, 4))
    clean_baseline_rows = rob[(rob["perturbation"] == "clean") & (rob["optimization"] == "baseline")]
    if clean_baseline_rows.empty:
        clean_baseline_rows = rob[rob["perturbation"] == "clean"]
    clean_baseline = float(clean_baseline_rows["dice_mean"].mean())
    ax.axhline(clean_baseline, color="0.45", linestyle="--", linewidth=1.2, label="2-case clean baseline (severity 0)")

    rob_perturbed = rob[rob["perturbation"] != "clean"].copy()
    robustness_summary = rob_perturbed.groupby(["perturbation", "severity"], as_index=False)["dice_mean"].mean()
    for name, grp in robustness_summary.groupby("perturbation"):
        grp = grp.sort_values("severity")
        is_intensity = name == "intensity_shift"
        ax.plot(
            grp["severity"],
            grp["dice_mean"],
            marker="D" if is_intensity else "o",
            linewidth=2.8 if is_intensity else 1.7,
            markersize=6 if is_intensity else 5,
            label=name,
            zorder=3 if is_intensity else 2,
        )
    ax.set_xlabel("Severity")
    ax.set_ylabel("Dice")
    ax.set_xticks([1, 2, 3])
    _format_dice_axis(ax)
    ax.set_title("Robustness curves")
    ax.legend(fontsize=7)
    fig.tight_layout()
    save_figure(fig, out / "fig4_robustness_curves.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(compact_clean["peak_vram_mb"], compact_clean["latency"])
    _set_axis_padding(ax, compact_clean["peak_vram_mb"], compact_clean["latency"], x_frac=0.08, y_frac=0.14)
    _annotate_points(
        ax,
        compact_clean,
        "peak_vram_mb",
        "latency",
        {
            "SegResNet baseline": (-10, 20, "right", "bottom"),
            "Data pipeline best": (-10, -20, "right", "top"),
            "AMP fp16 best": (8, -16, "left", "top"),
            "AMP bf16": (8, 10, "left", "bottom"),
            "torch.compile": (8, 8, "left", "bottom"),
            "Combined": (8, 8, "left", "bottom"),
            "U-Net baseline": (-8, -16, "right", "top"),
        },
    )
    ax.set_xlabel("Peak VRAM (MB)")
    ax.set_ylabel("Latency per volume (s)")
    ax.set_title("VRAM vs latency")
    fig.tight_layout()
    save_figure(fig, out / "fig5_vram_vs_latency.png")
    plt.close(fig)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
