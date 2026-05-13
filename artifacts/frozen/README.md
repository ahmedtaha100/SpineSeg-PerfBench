# SpineSeg-PerfBench Frozen Bundle

This directory contains the GitHub-clean frozen reproducibility bundle for the
checked-in SpineSeg-PerfBench subset run. The bundle contains derived metrics,
sanitized manifests, generated tables/figures, profiler summaries, configs, run
ledgers, and checksums. It does not contain raw medical scans, model
checkpoints, local WandB run folders, or oversized profiler traces.

## Layout

- `outputs/benchmarks/`: benchmark JSON rows.
- `outputs/manifests/`: sanitized data manifests and train/val/test splits.
- `outputs/profiles/`: profiler operator and phase summaries.
- `outputs/robustness/`: robustness CSVs.
- `outputs/tables/`: generated result table CSVs.
- `outputs/figures/`: generated result figure PNGs.
- `outputs/runs/`: training metrics and run configs; checkpoints are omitted.
- `configs/`: frozen configuration files used by the run.
- `flops_summary.json`: CPU-only bare-model FLOPs/MACs analysis.
- `RUNS.md`, `RUN_LOG.md`, `SPEC.md`, `ARTIFACT_INDEX.json`, and
  `checksums.sha256`: bundle provenance and verification metadata.

## Verification

From the repository root:

```bash
python scripts/verify_artifacts.py artifacts/frozen
python scripts/verify_github_submission.py artifacts/frozen
shasum -a 256 -c artifacts/frozen/checksums.sha256
python -m json.tool artifacts/frozen/ARTIFACT_INDEX.json >/tmp/artifact_index_check.json
```

The benchmark JSON/CSV rows and checksum file are the source of truth for the
checked-in measurements.
