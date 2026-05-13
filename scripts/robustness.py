#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark import make_row, unique_run_id  # noqa: E402
from infer import run_inference  # noqa: E402

from spineseg_perfbench.config import config_hash, load_config
from spineseg_perfbench.profiling.vram import GPUUtilizationSampler, reset_peak_vram
from spineseg_perfbench.robustness.perturbations import PERTURBATION_NAMES
from spineseg_perfbench.utils.hashing import git_sha, stable_hash
from spineseg_perfbench.utils.io import append_run_ledger, ensure_dir, write_json
from spineseg_perfbench.utils.schema import validate_run_row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run robustness perturbation grid.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="opt_baseline")
    p.add_argument("--output-suffix", default="")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def grid() -> list[dict | None]:
    cells: list[dict | None] = [None]
    for name in PERTURBATION_NAMES:
        for severity in (1, 2, 3):
            cells.append({"name": name, "severity": severity})
    names_filter = os.environ.get("SPINESEGBENCH_ROBUSTNESS_NAMES")
    if names_filter:
        wanted = {x.strip() for x in names_filter.split(",") if x.strip()}
        cells = [
            cell
            for cell in cells
            if ("clean" in wanted and cell is None) or (cell is not None and cell["name"] in wanted)
        ]
    severities_filter = os.environ.get("SPINESEGBENCH_ROBUSTNESS_SEVERITIES")
    if severities_filter:
        wanted_severities = {int(x.strip()) for x in severities_filter.split(",") if x.strip()}
        cells = [
            cell
            for cell in cells
            if (cell is None and 0 in wanted_severities) or (cell is not None and int(cell["severity"]) in wanted_severities)
        ]
    return cells


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config, smoke=args.smoke)
    rows = []
    out_dir = ensure_dir("outputs/benchmarks")
    for pert in grid():
        reset_peak_vram()
        with GPUUtilizationSampler() as gpu_sampler:
            result = run_inference(
                args.checkpoint,
                cfg,
                perturbation=pert,
                amp_dtype=cfg.get("optimization_settings", {}).get("amp_dtype", "fp32"),
                compile_enabled=bool(cfg.get("optimization_settings", {}).get("compile", False)),
                smoke=args.smoke,
            )
        raw_rid = f"robust_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{stable_hash({'cfg': cfg, 'pert': pert, 'ckpt': args.checkpoint})}"
        rid = unique_run_id(raw_rid, out_dir)
        row = make_row(cfg, result, rid, perturbation=pert, gpu_util_pct_mean=gpu_sampler.mean())
        validate_run_row(row)
        json_path = out_dir / f"{rid}.json"
        write_json(json_path, row)
        pert_label = "clean" if pert is None else f"{pert['name']}:{pert['severity']}"
        append_run_ledger(
            row["run_id"],
            git_sha(),
            config_hash(cfg),
            str(json_path),
            row["model"],
            row["optimization"],
            pert_label,
            f"dice={row['quality']['dice_mean']:.4f}, latency={row['latency_per_volume_sec_mean']:.4f}s",
        )
        rows.append(
            {
                "run_id": row["run_id"],
                "model": row["model"],
                "optimization": row["optimization"],
                "perturbation": "clean" if pert is None else pert["name"],
                "severity": 0 if pert is None else pert["severity"],
                "dice_mean": row["quality"]["dice_mean"],
                "hd95_mean_mm": row["quality"]["hd95_mean_mm"],
                "latency_per_volume_sec_mean": row["latency_per_volume_sec_mean"],
                "json_path": str(json_path),
            }
        )
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    csv_path = Path(f"outputs/robustness_results{suffix}.csv")
    pd.DataFrame(rows).sort_values(["perturbation", "severity"]).to_csv(csv_path, index=False, lineterminator="\n")
    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
