#!/usr/bin/env python
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from spineseg_perfbench.config import load_config
from spineseg_perfbench.models.registry import build_model
from spineseg_perfbench.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute CPU-only bare-model FLOPs for model-complexity reporting.")
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--manifest-dir", default="artifacts/frozen/outputs/manifests")
    p.add_argument("--output", default="artifacts/frozen/flops_summary.json")
    p.add_argument("--out-channels", type=int, default=26)
    p.add_argument("--in-channels", type=int, default=1)
    return p.parse_args()


def _resampled_shape(row: pd.Series, target_spacing: list[float]) -> list[int]:
    axes = ["x", "y", "z"]
    return [
        int(round(float(row[f"shape_{axis}"]) * float(row[f"spacing_{axis}"]) / float(target_spacing[i])))
        for i, axis in enumerate(axes)
    ]


def _patch_count(shape: list[int], patch_size: list[int], overlap: float) -> int:
    intervals = [max(1, int(patch * (1.0 - overlap))) for patch in patch_size]
    counts = [
        max(1, math.ceil(max(size - patch, 0) / interval) + 1)
        for size, patch, interval in zip(shape, patch_size, intervals)
    ]
    return int(math.prod(counts))


def _typical_volume(manifest_dir: Path, patch_size: list[int], target_spacing: list[float], overlap: float) -> dict[str, Any]:
    test_path = manifest_dir / "split_test.csv"
    manifest_path = manifest_dir / "data_manifest.csv"
    path = test_path if test_path.exists() else manifest_path
    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        shape = _resampled_shape(row, target_spacing)
        rows.append({"shape": shape, "patches": _patch_count(shape, patch_size, overlap)})
    shape_df = pd.DataFrame([r["shape"] + [r["patches"]] for r in rows], columns=["d", "h", "w", "patches"])
    return {
        "source": path.as_posix(),
        "n_cases": int(len(shape_df)),
        "median_resampled_shape": [int(round(shape_df[c].median())) for c in ["d", "h", "w"]],
        "median_patches_per_volume": int(round(shape_df["patches"].median())),
        "mean_patches_per_volume": float(shape_df["patches"].mean()),
        "min_patches_per_volume": int(shape_df["patches"].min()),
        "max_patches_per_volume": int(shape_df["patches"].max()),
    }


def _leaf_activation_bytes(model: torch.nn.Module, dummy: torch.Tensor) -> int:
    total = 0
    handles = []

    def tensor_bytes(value: Any) -> int:
        if isinstance(value, torch.Tensor):
            return int(value.numel() * value.element_size())
        if isinstance(value, (list, tuple)):
            return sum(tensor_bytes(v) for v in value)
        if isinstance(value, dict):
            return sum(tensor_bytes(v) for v in value.values())
        return 0

    def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        nonlocal total
        total += tensor_bytes(output)

    for module in model.modules():
        if module is model:
            continue
        if not any(module.children()):
            handles.append(module.register_forward_hook(hook))
    try:
        with torch.no_grad():
            model(dummy)
    finally:
        for handle in handles:
            handle.remove()
    return total


def _fvcore_flops(model: torch.nn.Module, dummy: torch.Tensor) -> tuple[float, list[str], str]:
    from fvcore.nn import FlopCountAnalysis

    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        analysis = FlopCountAnalysis(model, dummy)
        analysis.tracer_warnings("all")
        flops = float(analysis.total())
        unsupported = analysis.unsupported_ops()
    unsupported_ops = [f"{name}: {count}" for name, count in sorted(Counter(unsupported).items())]
    return flops, unsupported_ops, stderr.getvalue().strip()


def _torch_flops(model: torch.nn.Module, dummy: torch.Tensor) -> tuple[float, list[str], str]:
    from torch.utils.flop_counter import FlopCounterMode

    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        with torch.no_grad(), FlopCounterMode(display=False) as flop_counter:
            model(dummy)
        total = float(flop_counter.get_total_flops())
    return total, ["fallback=torch.utils.flop_counter"], stderr.getvalue().strip()


def _model_summary(name: str, patch_size: list[int], in_channels: int, out_channels: int, patches: int) -> dict[str, Any]:
    model = build_model(name, in_channels=in_channels, out_channels=out_channels, smoke=False).cpu().eval()
    dummy = torch.zeros((1, in_channels, *patch_size), dtype=torch.float32)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    weight_bytes = params * 4
    activation_bytes = _leaf_activation_bytes(model, dummy)
    tool = "fvcore.nn.FlopCountAnalysis"
    try:
        flops, unsupported_ops, stderr = _fvcore_flops(model, dummy)
    except Exception as exc:
        tool = "torch.utils.flop_counter.FlopCounterMode"
        flops, unsupported_ops, stderr = _torch_flops(model, dummy)
        unsupported_ops.append(f"fvcore_error={type(exc).__name__}: {exc}")
    flops_g = flops / 1e9
    denominator_bytes = weight_bytes + activation_bytes
    return {
        "params_M": round(params / 1e6, 3),
        "flops_per_patch_G": round(flops_g, 3),
        "macs_per_patch_G": round(flops_g / 2.0, 3),
        "patches_per_volume_estimate": int(patches),
        "flops_per_volume_G": round(flops_g * patches, 1),
        "unsupported_ops": unsupported_ops,
        "tool": tool,
        "weight_bytes_fp32": int(weight_bytes),
        "leaf_activation_bytes_fp32": int(activation_bytes),
        "arithmetic_intensity_flops_per_byte": round(flops / denominator_bytes, 2),
        "tool_warnings": stderr,
    }


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    patch_size = [int(x) for x in cfg["preprocess"]["patch_size"]]
    target_spacing = [float(x) for x in cfg["preprocess"]["spacing"]]
    overlap = float(cfg["inference"]["overlap"])
    typical = _typical_volume(Path(args.manifest_dir), patch_size, target_spacing, overlap)
    patches = int(typical["median_patches_per_volume"])
    models = {
        "segresnet": _model_summary("segresnet", patch_size, args.in_channels, args.out_channels, patches),
        "unet3d": _model_summary("unet", patch_size, args.in_channels, args.out_channels, patches),
    }
    result = {
        "patch_size": patch_size,
        "in_channels": int(args.in_channels),
        "models": models,
        "notes": (
            "FLOPs are computed on the bare MONAI model with a single fp32 patch-shaped input tensor, "
            "not through the sliding-window inferer. MACs are reported as FLOPs/2. Per-volume estimates "
            f"use the median patch count from {typical['n_cases']} frozen test cases after resampling "
            f"manifest shapes to {target_spacing} mm spacing with overlap={overlap}. "
            "Arithmetic intensity uses FLOPs divided by fp32 trainable-parameter bytes plus leaf-module "
            "output activation bytes for one forward pass; this is a proxy, not a hardware roofline measurement."
        ),
        "volume_patch_estimate": typical,
    }
    out = Path(args.output)
    ensure_dir(out.parent)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
