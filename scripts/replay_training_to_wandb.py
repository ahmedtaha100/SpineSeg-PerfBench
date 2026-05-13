#!/usr/bin/env python3
"""Replay frozen training metrics into WandB.

The source metrics are the immutable JSON files under
artifacts/frozen/outputs/runs/*/metrics.json. This script reconstructs WandB
runs from those frozen files; it does not train models or run inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import wandb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="spineseg-perfbench")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--metrics-root", type=Path, default=Path("artifacts/frozen/outputs/runs"))
    return parser.parse_args()


def model_name(metrics_path: Path, metrics: dict[str, Any]) -> str:
    raw = str(metrics.get("model") or metrics_path.parent.name)
    raw = raw.replace("_baseline", "").replace("-", "_").lower()
    if raw in {"unet", "3d_unet", "3d-u-net"}:
        return "3d_unet"
    if raw == "segresnet":
        return "segresnet"
    return raw


def metric_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("history", "steps_history", "metrics", "rows"):
        value = metrics.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
    row: dict[str, Any] = {}
    for key, value in metrics.items():
        if key == "phase_times_sec" and isinstance(value, dict):
            row["dataload_seconds"] = value.get("dataload")
            row["train_seconds"] = value.get("train")
            row["total_seconds"] = value.get("total")
        elif isinstance(value, (int, float, str, bool)) or value is None:
            row[key] = value
    if "throughput_steps_per_sec" in row:
        row["steps_per_second"] = row["throughput_steps_per_sec"]
    return [row]


def step_for(row: dict[str, Any], fallback: int) -> int:
    for key in ("step", "steps", "global_step"):
        value = row.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
    return fallback


def replay_file(metrics_path: Path, args: argparse.Namespace) -> None:
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: could not read {metrics_path}: {exc}")
        return
    if not isinstance(metrics, dict):
        print(f"ERROR: {metrics_path} does not contain a JSON object")
        return

    model = model_name(metrics_path, metrics)
    source = metrics_path.as_posix()
    rows = metric_rows(metrics)
    config = {
        "model": model,
        "source_metrics_path": source,
        "replay_note": f"Replayed from frozen {source}; not a live training run.",
    }

    if args.dry_run:
        print(f"DRY RUN: training_replay_{model} -> {len(rows)} metric row(s) from {source}")
        for idx, row in enumerate(rows):
            print(json.dumps({"step": step_for(row, idx), "metrics": row}, sort_keys=True))
        return

    init_kwargs: dict[str, Any] = {
        "project": args.project,
        "name": f"training_replay_{model}",
        "tags": ["training", "replay", "frozen-source"],
        "config": config,
        "reinit": True,
    }
    if args.entity:
        init_kwargs["entity"] = args.entity

    run = wandb.init(**init_kwargs)
    try:
        for idx, row in enumerate(rows):
            wandb.log(row, step=step_for(row, idx))
    finally:
        wandb.finish()


def main() -> int:
    args = parse_args()
    paths = sorted(args.metrics_root.glob("*/metrics.json"))
    if not paths:
        print(f"ERROR: no metrics.json files found under {args.metrics_root}")
        return 1
    for path in paths:
        replay_file(path, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
