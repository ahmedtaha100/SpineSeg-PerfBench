# SpineSeg-PerfBench

SpineSeg-PerfBench is a reproducible benchmark and profiling harness for 3D
vertebra segmentation from spine CT. It runs MONAI SegResNet and 3D U-Net
baselines, records segmentation and efficiency metrics, applies robustness
perturbations, and verifies frozen artifacts with checksums.

This is the **code-only repository** for the HPML final submission. The final
report PDF and PowerPoint are submitted separately and are not included here.

For the checked-in frozen subset run, the combined SegResNet inference
configuration reduced mean latency from **5.460 s/volume** to
**3.346 s/volume**, a **1.63x speedup**. These numbers are infrastructure and
profiling measurements from frozen JSON rows, not segmentation-quality claims.

Submission state is pinned by the pushed `submission-final` git tag.

## What is included

- `spineseg_perfbench/` - package code.
- `scripts/` - data preparation, training, benchmarking, plotting, freezing,
  WandB export, and verification entry points.
- `configs/` - model, optimization, and perturbation configs.
- `tests/` - CPU-oriented tests and smoke checks.
- `artifacts/frozen/` - the reviewed frozen artifact bundle, including JSON/CSV
  rows, generated tables/figures, profiler summaries, checksums, run ledgers,
  and the static WandB export.

The repo does **not** include raw medical scans, model checkpoints, local WandB
folders, oversized profiler traces, the final report PDF, or the PowerPoint.

## Install

```bash
conda env create -f environment.yml
conda activate spineseg-perfbench
python -m pip install -e .
```

Pip-only install:

```bash
python -m pip install -e .
```

For the recorded H200/CUDA run environment, see `requirements-frozen.txt`.

## CPU smoke test

```bash
SMOKE=1 bash scripts/run_all.sh
```

The smoke test uses tiny synthetic data, does not require a GPU or medical
images, and writes to `artifacts/smoke_frozen/`.

## Verify the frozen artifacts

```bash
python scripts/verify_artifacts.py artifacts/frozen
python scripts/verify_github_submission.py artifacts/frozen
shasum -a 256 -c artifacts/frozen/checksums.sha256
python -m json.tool artifacts/frozen/ARTIFACT_INDEX.json >/tmp/artifact_index_check.json
```

The benchmark JSON/CSV files under `artifacts/frozen/outputs/` are the canonical
source of truth for reported values.

## Real-data rerun

This repo does not redistribute VerSe, CTSpine1K, NIfTI, DICOM, or other raw
medical imaging data. Download data from the official source, keep it outside
the repository, then run:

```bash
export VERSE_ROOT=/path/to/verse
bash scripts/run_all.sh
```

Real-data reruns write to `artifacts/local_frozen/` by default and leave the
submitted `artifacts/frozen/` bundle untouched. To intentionally overwrite the
submitted bundle:

```bash
OVERWRITE_SUBMISSION_BUNDLE=1 bash scripts/run_all.sh
```

## WandB export

We had difficulty making the team-owned WandB project publicly viewable without
a sign-in prompt on the current WandB plan. For review, we exported the WandB
data into this repository and generated a static PDF view instead.

- Static PDF snapshot:
  `artifacts/frozen/wandb_export/wandb_report.pdf`
- Raw WandB export:
  `artifacts/frozen/wandb_export/runs/<run_id>/`
- WandB export index:
  `artifacts/frozen/wandb_export/index.json`

Each run export contains `config.json`, `summary.json`, `history.csv`,
`system_metrics.csv`, and `metadata.json`. The PDF is only a readable dashboard
snapshot; the raw JSON/CSV files and `artifacts/frozen/outputs/` remain the
auditable data.

## License

Apache License 2.0. See `LICENSE`.
