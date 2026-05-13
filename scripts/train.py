#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference

from spineseg_perfbench.config import load_config, write_config
from spineseg_perfbench.data.transforms import normalize_ct
from spineseg_perfbench.metrics.dice import compute_multiclass_dice
from spineseg_perfbench.models.registry import build_model
from spineseg_perfbench.optimization.dataloader import make_dataloader
from spineseg_perfbench.utils.io import ensure_dir, load_nifti, write_json
from spineseg_perfbench.utils.seed import set_seed


class NiftiSegDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        rows: pd.DataFrame,
        smoke: bool = False,
        patch_size: tuple[int, int, int] | None = None,
        seed: int = 42,
    ):
        self.rows = rows.reset_index(drop=True)
        self.smoke = smoke
        self.patch_size = patch_size
        self.seed = int(seed)
        self._calls: dict[int, int] = {}
        self._volume_cache: list[tuple[np.ndarray, np.ndarray]] | None = None
        if not smoke:
            self._volume_cache = []
            print(f"preloading_training_volumes n={len(self.rows)}", flush=True)
            for i, row in self.rows.iterrows():
                self._volume_cache.append(self._load_arrays(row))
                if (i + 1) % 5 == 0 or i + 1 == len(self.rows):
                    print(f"preloaded_training_volumes {i + 1}/{len(self.rows)}", flush=True)

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _load_arrays(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        image, _ = load_nifti(row.image_path)
        label, _ = load_nifti(row.label_path)
        image = normalize_ct(image)
        label = label.astype(np.int16, copy=False)
        label[(label < 0) | (label > 25)] = 0
        return image, label

    @staticmethod
    def _crop_bounds(center: int, size: int, limit: int) -> tuple[int, int]:
        if limit <= size:
            return 0, limit
        start = min(max(center - size // 2, 0), limit - size)
        return start, start + size

    @staticmethod
    def _pad_to_shape(array: np.ndarray, shape: tuple[int, int, int], value: float | int) -> np.ndarray:
        pad_width = []
        for current, target in zip(array.shape, shape, strict=True):
            total = max(target - current, 0)
            before = total // 2
            pad_width.append((before, total - before))
        if not any(before or after for before, after in pad_width):
            return array
        return np.pad(array, pad_width, mode="constant", constant_values=value)

    def _sample_patch(self, image: np.ndarray, label: np.ndarray, idx: int) -> tuple[np.ndarray, np.ndarray]:
        if self.patch_size is None:
            return image, label
        call = self._calls.get(idx, 0)
        self._calls[idx] = call + 1
        rng = np.random.default_rng(self.seed + idx * 100_003 + call)
        patch = tuple(int(x) for x in self.patch_size)
        if image.shape != label.shape:
            raise ValueError(f"Image/label shape mismatch for row {idx}: {image.shape} vs {label.shape}")

        foreground = np.argwhere(label > 0)
        if foreground.size and rng.random() < 0.5:
            center = foreground[int(rng.integers(0, len(foreground)))]
        else:
            center = np.asarray([int(rng.integers(0, max(dim, 1))) for dim in image.shape])

        slices = tuple(slice(*self._crop_bounds(int(c), p, dim)) for c, p, dim in zip(center, patch, image.shape, strict=True))
        image_patch = image[slices]
        label_patch = label[slices]
        return self._pad_to_shape(image_patch, patch, 0.0), self._pad_to_shape(label_patch, patch, 0)

    def __getitem__(self, idx: int) -> dict:
        if self._volume_cache is not None:
            image, label = self._volume_cache[idx]
        else:
            image, label = self._load_arrays(self.rows.iloc[idx])
        if not self.smoke:
            image, label = self._sample_patch(image, label, idx)
        return {
            "image": torch.from_numpy(image[None]).float(),
            "label": torch.from_numpy(label[None]).long(),
            "case_id": str(self.rows.iloc[idx].case_id),
        }


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(description="Train a MONAI model for SpineSeg-PerfBench sanity runs.")
    p.add_argument("--config", default=None)
    p.add_argument("--run-name", default="segresnet_baseline")
    p.add_argument("--smoke", action="store_true")
    return p.parse_known_args()


def _ensure_smoke_data(smoke: bool) -> None:
    if smoke and not Path("outputs/manifests/split_train.csv").exists():
        subprocess.check_call([sys.executable, "scripts/prepare_data.py", "--synthetic", "--smoke"])


def _device(cfg: dict, smoke: bool) -> torch.device:
    if smoke or cfg.get("device") == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate(
    model,
    rows: pd.DataFrame,
    device: torch.device,
    smoke: bool,
    patch_size: tuple[int, int, int],
) -> float:
    model.eval()
    vals = []
    with torch.no_grad():
        for _, row in rows.head(2 if smoke else len(rows)).iterrows():
            image, _ = load_nifti(row.image_path)
            label, _ = load_nifti(row.label_path)
            tensor = torch.from_numpy(normalize_ct(image)[None, None]).float().to(device)
            if smoke:
                logits = model(tensor)
            else:
                logits = sliding_window_inference(
                    tensor,
                    roi_size=patch_size,
                    sw_batch_size=1,
                    predictor=model,
                    overlap=0.5,
                    mode="gaussian",
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
            pred = torch.argmax(logits, dim=1).cpu().numpy()[0].astype(np.int16)
            vals.append(compute_multiclass_dice(pred, label.astype(np.int16), labels=list(range(1, 26)))["case_mean"])
    return float(np.mean(vals)) if vals else 0.0


def main() -> int:
    args, overrides = parse_args()
    cfg = load_config(args.config, overrides, smoke=args.smoke)
    _ensure_smoke_data(args.smoke)
    set_seed(int(cfg["seed"]))
    model_name = str(cfg.get("model", "segresnet"))
    run_dir = ensure_dir(Path(cfg["output_dir"]) / "runs" / args.run_name)
    write_config(run_dir / "config.yaml", cfg)

    train_df = pd.read_csv(cfg["manifest"]["train"])
    train_case_limit = int(os.environ.get("SPINESEGBENCH_TRAIN_CASE_LIMIT", "0") or "0")
    if train_case_limit > 0 and not args.smoke and len(train_df) > train_case_limit:
        train_df = train_df.sort_values(["dataset_source", "case_id"]).head(train_case_limit).reset_index(drop=True)
        print(f"train_case_limit_applied {train_case_limit}", flush=True)
    val_path = Path(cfg["manifest"]["val"])
    if not val_path.exists():
        raise FileNotFoundError(f"Validation split not found: {val_path}. Run scripts/prepare_data.py first.")
    val_df = pd.read_csv(val_path)
    if len(val_df) == 0:
        raise ValueError(f"Validation split is empty: {val_path}. Regenerate splits with validation cases before training.")

    device = _device(cfg, args.smoke)
    model = build_model(model_name, out_channels=26, smoke=args.smoke).to(device)
    model.train()
    loss_fn = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    dataset = NiftiSegDataset(
        train_df,
        smoke=args.smoke,
        patch_size=tuple(int(x) for x in cfg["preprocess"]["patch_size"]),
        seed=int(cfg["seed"]),
    )
    loader = make_dataloader(
        dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        settings={},
        smoke=args.smoke,
    )
    max_steps = int(cfg["smoke"]["max_train_steps"]) if args.smoke else len(loader) * int(cfg["training"]["epochs"])
    losses = []
    steps = 0
    dataload_time = 0.0
    train_time = 0.0
    total_start = time.perf_counter()
    for _epoch in range(int(cfg["training"]["epochs"])):
        iterator = iter(loader)
        while True:
            start = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            dataload_time += time.perf_counter() - start
            image = batch["image"].to(device)
            label = batch["label"].to(device)
            start = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(image), label)
            loss.backward()
            opt.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            train_time += time.perf_counter() - start
            losses.append(float(loss.detach().cpu()))
            steps += 1
            if steps >= max_steps:
                break
        if steps >= max_steps:
            break
    total_time = time.perf_counter() - total_start

    val_dice = validate(
        model,
        val_df,
        device,
        args.smoke,
        tuple(int(x) for x in cfg["preprocess"]["patch_size"]),
    )
    checkpoint = {
        "model": model_name,
        "state_dict": model.state_dict(),
        "seed": int(cfg["seed"]),
        "smoke": bool(args.smoke),
    }
    torch.save(checkpoint, run_dir / "checkpoint.pt")
    write_json(
        run_dir / "metrics.json",
        {
            "train_loss": float(np.mean(losses)) if losses else None,
            "val_dice": val_dice,
            "steps": steps,
            "train_cases_used": int(len(train_df)),
            "train_case_limit": train_case_limit if train_case_limit > 0 else None,
            "phase_times_sec": {
                "dataload": float(dataload_time),
                "train": float(train_time),
                "total": float(total_time),
            },
            "throughput_steps_per_sec": float(steps / max(total_time, 1e-9)),
        },
    )
    print(json.dumps({"checkpoint": str(run_dir / "checkpoint.pt"), "val_dice": val_dice, "steps": steps}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
