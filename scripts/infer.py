#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from monai.inferers import sliding_window_inference

from spineseg_perfbench.config import load_config
from spineseg_perfbench.data.transforms import normalize_ct
from spineseg_perfbench.metrics.dice import compute_multiclass_dice
from spineseg_perfbench.metrics.hd95 import compute_multiclass_hd95
from spineseg_perfbench.models.registry import build_model
from spineseg_perfbench.optimization.amp import autocast_context, effective_amp_dtype
from spineseg_perfbench.optimization.compile import try_compile
from spineseg_perfbench.optimization.dataloader import make_dataloader
from spineseg_perfbench.profiling.timer import PhaseTimer
from spineseg_perfbench.robustness.perturbations import apply_perturbation
from spineseg_perfbench.utils.hashing import stable_hash
from spineseg_perfbench.utils.io import ensure_dir, load_nifti, save_nifti, write_json
from spineseg_perfbench.utils.seed import set_seed


class InferenceNiftiDataset(torch.utils.data.Dataset):
    def __init__(self, rows: pd.DataFrame, cache: str = "none"):
        self.rows = rows.reset_index(drop=True)
        self.cache = cache
        self.memory_cache: dict[int, tuple[np.ndarray, np.ndarray, tuple[float, float, float], str]] = {}
        self.disk_cache_dir = ensure_dir("outputs/cache/inference") if cache == "persistent_disk" else None

    def __len__(self) -> int:
        return len(self.rows)

    def _load(self, idx: int) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float], str]:
        row = self.rows.iloc[idx]
        image, spacing = load_nifti(row.image_path)
        label, _ = load_nifti(row.label_path)
        return image.astype(np.float32), label.astype(np.int16), spacing, str(row.case_id)

    def __getitem__(self, idx: int) -> dict:
        if self.cache == "monai_cache" and idx in self.memory_cache:
            image, label, spacing, case_id = self.memory_cache[idx]
        elif self.cache == "persistent_disk" and self.disk_cache_dir is not None:
            row = self.rows.iloc[idx]
            cache_key = stable_hash(
                {
                    "case_id": str(row.case_id),
                    "dataset_source": str(row.get("dataset_source", "")),
                    "image_path": str(row.image_path),
                    "label_path": str(row.label_path),
                }
            )
            cache_path = self.disk_cache_dir / f"{row.case_id}_{cache_key}.npz"
            if cache_path.exists():
                data = np.load(cache_path)
                image = data["image"]
                label = data["label"]
                spacing = tuple(float(x) for x in data["spacing"])
                case_id = str(row.case_id)
            else:
                image, label, spacing, case_id = self._load(idx)
                np.savez_compressed(cache_path, image=image, label=label, spacing=np.asarray(spacing, dtype=np.float32))
        else:
            image, label, spacing, case_id = self._load(idx)
            if self.cache == "monai_cache":
                self.memory_cache[idx] = (image, label, spacing, case_id)
        return {
            "image": image,
            "label": label,
            "spacing": np.asarray(spacing, dtype=np.float32),
            "case_id": case_id,
            "index": int(idx),
        }


def _batch_to_numpy(batch: dict) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float], str, int]:
    image = batch["image"]
    label = batch["label"]
    spacing = batch["spacing"]
    if isinstance(image, torch.Tensor):
        image = image[0].cpu().numpy()
    else:
        image = np.asarray(image[0])
    if isinstance(label, torch.Tensor):
        label = label[0].cpu().numpy()
    else:
        label = np.asarray(label[0])
    if isinstance(spacing, torch.Tensor):
        spacing_values = spacing[0].cpu().numpy()
    else:
        spacing_values = np.asarray(spacing[0])
    case_id = batch["case_id"][0] if isinstance(batch["case_id"], (list, tuple)) else str(batch["case_id"])
    index = int(batch["index"][0].item() if isinstance(batch["index"], torch.Tensor) else batch["index"][0])
    return image, label, tuple(float(x) for x in spacing_values), str(case_id), index


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run sliding-window inference and compute quality metrics.")
    p.add_argument("--config", default="opt_baseline")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--save-predictions", action="store_true")
    p.add_argument("--output-dir", default="outputs/inference")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def load_model_from_checkpoint(checkpoint: str | Path, cfg: dict, device: torch.device, smoke: bool):
    try:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(checkpoint, map_location=device)
    model_name = ckpt.get("model", cfg.get("model", "segresnet"))
    model = build_model(model_name, out_channels=26, smoke=bool(ckpt.get("smoke", smoke))).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, model_name


def _split_path(cfg: dict, split: str) -> str:
    return cfg["manifest"].get(split, cfg["manifest"]["test"])


def _compute_case_metrics(pred: np.ndarray, label: np.ndarray, spacing: tuple[float, float, float]) -> tuple[float, float]:
    label = label.astype(np.int16, copy=False)
    dice = compute_multiclass_dice(pred, label, labels=list(range(1, 26)))["case_mean"]
    hd95 = compute_multiclass_hd95(pred, label, spacing=spacing, labels=list(range(1, 26)))["case_mean"]
    return float(dice), float(hd95)


def run_inference(
    checkpoint: str | Path,
    cfg: dict,
    split: str = "test",
    perturbation: dict | None = None,
    save_predictions: bool = False,
    predictions_dir: str | Path | None = None,
    amp_dtype: str | None = None,
    compile_enabled: bool = False,
    smoke: bool = False,
    profile_step=None,
) -> dict[str, Any]:
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device("cpu" if smoke or cfg.get("device") == "cpu" else ("cuda" if torch.cuda.is_available() else "cpu"))
    timer = PhaseTimer()
    rows = pd.read_csv(_split_path(cfg, split))
    infer_case_limit = int(os.environ.get("SPINESEGBENCH_INFER_CASE_LIMIT", "0") or "0")
    if infer_case_limit > 0 and not smoke and len(rows) > infer_case_limit:
        rows = rows.head(infer_case_limit).reset_index(drop=True)
    if smoke:
        if split == "test" and len(rows) < 2 and Path(cfg["manifest"].get("val", "")).exists():
            val_rows = pd.read_csv(cfg["manifest"]["val"])
            rows = pd.concat([rows, val_rows], ignore_index=True).drop_duplicates("case_id")
        rows = rows.head(2)
    model, model_name = load_model_from_checkpoint(checkpoint, cfg, device, smoke)
    compile_requested = bool(compile_enabled)
    force_compile_fallback = bool(smoke and device.type == "cpu" and compile_requested)
    model, compile_succeeded, compile_overhead_sec, compile_notes = try_compile(
        model,
        mode=str(cfg.get("optimization_settings", {}).get("compile_mode", "reduce-overhead")),
        enabled=compile_requested,
        force_fail=force_compile_fallback,
    )
    requested_amp = (amp_dtype or cfg.get("optimization_settings", {}).get("amp_dtype") or "fp32").lower()
    effective_amp, amp_note = effective_amp_dtype(requested_amp, device)
    roi_size = tuple(int(x) for x in cfg["preprocess"]["patch_size"])
    warmup = int(cfg["inference"]["warmup_volumes"])
    overlap = float(cfg["inference"]["overlap"])
    mode = str(cfg["inference"]["mode"])
    sw_batch_size = int(os.environ.get("SPINESEGBENCH_SW_BATCH_SIZE", "1") or "1")
    metric_workers = int(os.environ.get("SPINESEGBENCH_METRIC_WORKERS", "0") or "0")
    if smoke:
        metric_workers = 0
    pred_dir = ensure_dir(predictions_dir or Path("outputs/predictions")) if save_predictions else None
    settings = cfg.get("optimization_settings", {})
    dataset = InferenceNiftiDataset(rows, cache=str(settings.get("cache", "none")))
    loader = make_dataloader(dataset, batch_size=1, shuffle=False, settings=settings, smoke=smoke)
    iterator = iter(loader)

    dice_values: list[float] = []
    hd95_values: list[float] = []
    all_latencies: list[float] = []
    latencies: list[float] = []
    metric_futures: list[Future[tuple[float, float]]] = []
    steady_state_latency_sec = None
    latency_note = None
    total_start = time.perf_counter()
    executor: ThreadPoolExecutor | None = ThreadPoolExecutor(max_workers=metric_workers) if metric_workers > 0 else None
    try:
        for _ in range(len(dataset)):
            with timer.phase("dataload"):
                batch = next(iterator)
                image, label, spacing, case_id, i = _batch_to_numpy(batch)
            with timer.phase("preprocess"):
                image = normalize_ct(image, cfg["preprocess"]["clip_min"], cfg["preprocess"]["clip_max"])
                if perturbation is not None:
                    image, label = apply_perturbation(
                        image,
                        label,
                        name=perturbation["name"],
                        severity=int(perturbation["severity"]),
                        seed=int(cfg.get("seed", 42)) + int(i),
                    )
                input_tensor = torch.from_numpy(image[None, None]).float().to(device)
            with torch.no_grad():
                start = time.perf_counter()
                with timer.phase("infer"):
                    with autocast_context(effective_amp, device):
                        logits = sliding_window_inference(
                            input_tensor,
                            roi_size=roi_size,
                            sw_batch_size=sw_batch_size,
                            predictor=model,
                            overlap=overlap,
                            mode=mode,
                        )
                    if profile_step is not None:
                        profile_step()
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                latency = time.perf_counter() - start
            all_latencies.append(float(latency))
            if i >= warmup:
                latencies.append(float(latency))
            pred = torch.argmax(logits, dim=1).cpu().numpy()[0].astype(np.int16)
            label = label.astype(np.int16, copy=False)
            if pred_dir is not None:
                save_nifti(pred_dir / f"{case_id}_pred.nii.gz", pred, spacing)
            if executor is not None:
                metric_futures.append(executor.submit(_compute_case_metrics, pred, label, spacing))
            else:
                dice, hd95 = _compute_case_metrics(pred, label, spacing)
                dice_values.append(dice)
                hd95_values.append(hd95)
            if not smoke:
                print(
                    {
                        "inference_progress": {
                            "case": int(len(all_latencies)),
                            "n_cases": int(len(dataset)),
                            "case_id": case_id,
                            "latency_sec": float(latency),
                            "metric_workers": metric_workers,
                        }
                    },
                    flush=True,
                )
        if executor is not None:
            for fut in metric_futures:
                dice, hd95 = fut.result()
                dice_values.append(dice)
                hd95_values.append(hd95)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    timer.times["total"] += time.perf_counter() - total_start
    if not latencies:
        if not all_latencies:
            raise ValueError("no cases available for inference")
        latencies = all_latencies
        latency_note = "warmup consumed all cases; latency computed over all evaluated volumes"
    steady_state_latency_sec = float(np.mean(latencies))
    hd_arr = np.asarray(hd95_values, dtype=float)
    hd_mean = float(np.nanmean(hd_arr)) if hd_arr.size and not np.all(np.isnan(hd_arr)) else None
    hd_std = float(np.nanstd(hd_arr)) if hd_arr.size and not np.all(np.isnan(hd_arr)) else None
    notes = "; ".join(x for x in [amp_note, compile_notes, latency_note] if x) or None
    return {
        "model_name": model_name,
        "split": split,
        "phase_times_sec": timer.benchmark_phases(),
        "latencies": latencies,
        "latency_mean": float(np.mean(latencies)),
        "latency_p50": float(np.percentile(latencies, 50)),
        "latency_p95": float(np.percentile(latencies, 95)),
        "throughput": float(len(latencies) / max(sum(latencies), 1e-9)),
        "quality": {
            "dice_mean": float(np.mean(dice_values)) if dice_values else 0.0,
            "dice_std": float(np.std(dice_values)) if dice_values else 0.0,
            "hd95_mean_mm": hd_mean,
            "hd95_std_mm": hd_std,
            "n_cases": int(len(dice_values)),
        },
        "predictions_dir": str(pred_dir) if pred_dir is not None else None,
        "compile_succeeded": compile_succeeded,
        "compile_overhead_sec": compile_overhead_sec,
        "steady_state_latency_sec": steady_state_latency_sec,
        "amp_dtype": effective_amp,
        "notes": notes,
    }


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config, smoke=args.smoke)
    result = run_inference(
        args.checkpoint,
        cfg,
        split=args.split,
        save_predictions=args.save_predictions,
        predictions_dir=args.output_dir,
        smoke=args.smoke,
    )
    write_json(Path(args.output_dir) / "summary.json", result)
    print(Path(args.output_dir) / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
