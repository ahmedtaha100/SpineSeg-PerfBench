#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the complete CPU synthetic smoke pipeline.")
    p.add_argument("--all", action="store_true")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def reset_generated_outputs() -> None:
    for path in [
        "outputs/manifests",
        "outputs/benchmarks",
        "outputs/profiles",
        "outputs/predictions",
        "outputs/inference",
        "outputs/cache",
        "outputs/runs",
        "outputs/tables",
        "outputs/figures",
        "artifacts/smoke_frozen",
        "artifacts/demo",
        "tests/fixtures/synthetic",
    ]:
        p = Path(path)
        if p.exists():
            shutil.rmtree(p)
    for path in Path("outputs").glob("robustness_results*.csv"):
        path.unlink()
    Path("RUNS.md").write_text(
        "# Run Ledger\n\n"
        "| run_id | git_sha | config_hash | JSON path | model | optimization | perturbation | one-line result |\n"
        "|---|---|---|---|---|---|---|---|\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    if not args.all:
        print("Running the full smoke pipeline. Use --all to make this explicit.", flush=True)
    if os.environ.get("PRESERVE_OUTPUTS") != "1":
        reset_generated_outputs()
    smoke = ["--smoke"] if args.smoke else []
    py = sys.executable
    run([py, "scripts/prepare_data.py", "--synthetic", *smoke])
    run([py, "scripts/train.py", "model=segresnet", "--run-name", "segresnet_smoke", *smoke])
    run([py, "scripts/train.py", "model=unet", "--run-name", "unet_smoke", *smoke])
    ckpt = "outputs/runs/segresnet_smoke/checkpoint.pt"
    run([py, "scripts/benchmark.py", "--config", "opt_baseline", "--checkpoint", ckpt, *smoke])
    run([py, "scripts/robustness.py", "--checkpoint", ckpt, "--config", "opt_baseline", "--output-suffix", "smoke_one", *smoke])
    run([py, "scripts/make_tables.py", *smoke])
    run([py, "scripts/make_plots.py", *smoke])
    run([py, "scripts/freeze_artifacts.py", "--output-dir", "artifacts/smoke_frozen", *smoke])
    run([py, "scripts/verify_artifacts.py", "artifacts/smoke_frozen", *smoke])
    if not list(Path("outputs/benchmarks").glob("*.json")):
        raise SystemExit("No benchmark JSON rows produced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
