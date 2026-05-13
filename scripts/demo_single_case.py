#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from monai.inferers import sliding_window_inference

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer import load_model_from_checkpoint  # noqa: E402

from spineseg_perfbench.config import load_config
from spineseg_perfbench.data.transforms import normalize_ct
from spineseg_perfbench.metrics.dice import compute_multiclass_dice
from spineseg_perfbench.metrics.hd95 import compute_multiclass_hd95
from spineseg_perfbench.optimization.amp import autocast_context, effective_amp_dtype
from spineseg_perfbench.optimization.compile import try_compile
from spineseg_perfbench.profiling.vram import GPUUtilizationSampler, peak_vram_mb, reset_peak_vram
from spineseg_perfbench.utils.io import ensure_dir, load_nifti


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a single-case demo and write overlay visualization.")
    p.add_argument("--checkpoint", default="outputs/runs/segresnet_baseline/checkpoint.pt")
    p.add_argument("--optimized-checkpoint", default=None)
    p.add_argument("--config", default="opt_baseline")
    p.add_argument("--optimized-config", default="opt_all")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def _ensure_checkpoint(args: argparse.Namespace) -> None:
    if args.smoke and not Path(args.checkpoint).exists():
        scripts_dir = Path(__file__).resolve().parent
        subprocess.check_call([sys.executable, str(scripts_dir / "prepare_data.py"), "--synthetic", "--smoke"])
        subprocess.check_call(
            [sys.executable, str(scripts_dir / "train.py"), "model=segresnet", "--run-name", "segresnet_baseline", "--smoke"]
        )


def _predict(checkpoint: str, cfg: dict, image, label, spacing, device: torch.device, smoke: bool, optimize: bool):
    model, model_name = load_model_from_checkpoint(checkpoint, cfg, device, smoke)
    settings = cfg.get("optimization_settings", {})
    model, compile_succeeded, compile_overhead_sec, compile_notes = try_compile(
        model,
        mode=str(settings.get("compile_mode", "reduce-overhead")),
        enabled=bool(settings.get("compile", False)) and optimize,
        force_fail=bool(smoke and optimize),
    )
    requested_amp = str(settings.get("amp_dtype", "fp32")) if optimize else "fp32"
    effective_amp, amp_note = effective_amp_dtype(requested_amp, device)
    tensor = torch.from_numpy(normalize_ct(image)[None, None]).float().to(device)
    roi_size = tuple(int(x) for x in cfg["preprocess"]["patch_size"])
    reset_peak_vram()
    start = time.perf_counter()
    with GPUUtilizationSampler() as gpu_sampler:
        with torch.no_grad(), autocast_context(effective_amp, device):
            logits = sliding_window_inference(
                tensor,
                roi_size=roi_size,
                sw_batch_size=1,
                predictor=model,
                overlap=float(cfg["inference"]["overlap"]),
                mode=str(cfg["inference"]["mode"]),
            )
    if device.type == "cuda":
        torch.cuda.synchronize()
    latency = time.perf_counter() - start
    pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
    dice = compute_multiclass_dice(pred, label.astype("int16"), labels=list(range(1, 26)))["case_mean"]
    hd95 = compute_multiclass_hd95(pred, label.astype("int16"), spacing=spacing, labels=list(range(1, 26)))["case_mean"]
    notes = "; ".join(x for x in [amp_note, compile_notes] if x) or None
    return {
        "model": model_name,
        "prediction": pred,
        "latency_sec": float(latency),
        "throughput_volumes_per_sec": float(1.0 / max(latency, 1e-9)),
        "dice": float(dice),
        "hd95_mm": None if pd.isna(hd95) else float(hd95),
        "peak_vram_mb": peak_vram_mb(),
        "gpu_util_pct_mean": gpu_sampler.mean(),
        "compile_succeeded": compile_succeeded,
        "compile_overhead_sec": compile_overhead_sec,
        "amp_dtype": effective_amp,
        "requested_amp_dtype": requested_amp,
        "notes": notes,
    }


def main() -> int:
    args = parse_args()
    _ensure_checkpoint(args)
    baseline_cfg = load_config(args.config, smoke=args.smoke)
    optimized_cfg = load_config(args.optimized_config, smoke=args.smoke)
    df = pd.read_csv(baseline_cfg["manifest"]["test"])
    if args.smoke:
        df = df.head(1)
    row = df.iloc[0]
    image, spacing = load_nifti(row.image_path)
    label, _ = load_nifti(row.label_path)
    device = torch.device("cpu" if args.smoke or not torch.cuda.is_available() else "cuda")
    optimized_checkpoint = args.optimized_checkpoint or args.checkpoint

    baseline = _predict(args.checkpoint, baseline_cfg, image, label, spacing, device, args.smoke, optimize=False)
    optimized = _predict(optimized_checkpoint, optimized_cfg, image, label, spacing, device, args.smoke, optimize=True)

    out = ensure_dir("artifacts/demo")
    z = image.shape[0] // 2
    x = image.shape[2] // 2
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    axes[0, 0].imshow(image[z], cmap="gray")
    axes[0, 0].imshow(label[z], alpha=0.35)
    axes[0, 0].set_title("Axial GT")
    axes[0, 1].imshow(image[z], cmap="gray")
    axes[0, 1].imshow(baseline["prediction"][z], alpha=0.35)
    axes[0, 1].set_title("Baseline")
    axes[0, 2].imshow(image[z], cmap="gray")
    axes[0, 2].imshow(optimized["prediction"][z], alpha=0.35)
    axes[0, 2].set_title("Optimized")
    axes[1, 0].imshow(image[:, :, x], cmap="gray")
    axes[1, 0].imshow(label[:, :, x], alpha=0.35)
    axes[1, 0].set_title("Sagittal GT")
    axes[1, 1].imshow(image[:, :, x], cmap="gray")
    axes[1, 1].imshow(baseline["prediction"][:, :, x], alpha=0.35)
    axes[1, 1].set_title("Baseline")
    axes[1, 2].imshow(image[:, :, x], cmap="gray")
    axes[1, 2].imshow(optimized["prediction"][:, :, x], alpha=0.35)
    axes[1, 2].set_title("Optimized")
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "single_case_overlay.png", dpi=160)
    plt.close(fig)

    summary = pd.DataFrame(
        [
            {k: v for k, v in baseline.items() if k != "prediction"} | {"mode": "baseline"},
            {k: v for k, v in optimized.items() if k != "prediction"} | {"mode": "optimized"},
        ]
    )
    summary.to_csv(out / "single_case_summary.csv", index=False, lineterminator="\n")
    print("runtime_report")
    print(summary.to_string(index=False))
    print("preprocess: CT normalization to [0, 1]")
    print(f"dataload: {row.image_path}")
    print(f"spacing: {spacing}")
    print(out / "single_case_overlay.png")
    print(out / "single_case_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
