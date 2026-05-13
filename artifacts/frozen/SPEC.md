# SpineSeg-PerfBench Specification

## Project

SpineSeg-PerfBench is a reproducible accuracy, robustness, and efficiency benchmark for 3D vertebra segmentation from spine CT volumes.

## Models

- MONAI 3D U-Net
  - channels=(16,32,64,128,256)
  - strides=(2,2,2,2)
  - num_res_units=2
- MONAI SegResNet
  - init_filters=16
  - blocks_down=(1,2,2,4)
  - blocks_up=(1,1,1)
No other models are in scope.

## Datasets

Primary:
- VerSe 2019: https://osf.io/nqjyw/ and https://verse2019.grand-challenge.org/
- VerSe 2020: https://osf.io/t98fz/ and https://verse2020.grand-challenge.org/
Fallback:
- CTSpine1K: https://github.com/MIRACLE-Center/CTSpine1K
- CTSpine1K XNAT: https://xnat.bmia.nl/data/archive/projects/africai_miccai2024_ctspine1k
- CTSpine1K Hugging Face mirror: https://huggingface.co/datasets/alexanderdann/CTSpine1K
No raw scans are committed or redistributed. The repo provides manifests, scripts, docs, and synthetic fixtures only.

## Dataset discovery

The default loader supports recursive glob discovery. Users may override patterns with:
- --image-glob
- --label-glob
Expected VerSe-like defaults:
- image candidates: `**/*ct*.nii.gz`, `**/*img*.nii.gz`, `**/*_ct.nii.gz`
- label candidates: `**/*seg*.nii.gz`, `**/*msk*.nii.gz`, `**/*_seg-vert_msk.nii.gz`
The manifest generator must validate image-label pairs and fail loudly on ambiguous or missing labels.

## Preprocessing

Frozen preprocessing for real runs:
- Reorient to RAS
- Resample to spacing (1.0, 1.0, 1.0) mm
- Clip HU to [-1024, 3071]
- Normalize to [0, 1]
- Patch size: (96, 96, 96)
- Foreground/background sampling ratio: 1:1 where applicable
Smoke preprocessing may use smaller synthetic volumes and smaller patch sizes.

## Splits

- Deterministic seed=42
- Train/val/test split: 70/15/15
- Stratify by dataset_source where possible
- Rerunning split generation with the same inputs must produce byte-identical CSVs.

## Training

Training is sanity-only for infrastructure validation.
Development/smoke:
- 2 iterations or 1 tiny epoch
- CPU-compatible
- Synthetic fixtures
Full GPU config:
- configurable epochs, default 50
- batch size default 2
- optimizer AdamW
- lr=1e-4
- weight_decay=1e-5
- loss DiceCELoss(include_background=False)
- seeds tracked: 42, 1337, 2024
- default single-seed run uses seed 42

## Inference

- Sliding-window inference
- Full GPU patch: (96,96,96)
- Smoke patch: smaller if needed
- overlap 0.5
- mode "gaussian"
- number of warmup volumes: 2 for real benchmark, 0 or 1 for smoke as needed
- warmup volumes are excluded from timing

## Metrics

Labels:
- vertebra classes 1-25
- background 0
Dice:
- per-vertebra-class
- mean over present labels per case
- then mean over cases
- if GT non-empty and prediction empty, Dice = 0
- if GT empty and prediction empty, Dice = 1
- missing absent classes are not included in the per-case mean
HD95:
- physical millimeters using image spacing
- per-vertebra-class
- computed with scipy distance transform using spacing/sampling
- mean over present labels per case
- nanmean over cases
- empty prediction when GT is non-empty gives HD95 = nan
- never silently replace nan HD95 with zero

## Robustness perturbations

Inference-time only.
Severity grid:
- 0 = clean
- 1 = mild
- 2 = moderate
- 3 = severe
Perturbations:
- gaussian_noise: std multipliers [0, 0.01, 0.025, 0.05]
- gaussian_blur: sigma [0, 0.5, 1.0, 2.0]
- downsample_resample: factor [1.0, 1.25, 1.5, 2.0]
- intensity_shift: offset [0, 0.05, 0.10, 0.20]
- contrast_shift: gamma [1.0, 1.1, 1.25, 1.5]
Severity 0 is clean baseline and should not be recomputed redundantly for every perturbation if avoidable.

## Optimizations

Data pipeline:
- num_workers in {0,2,4,8}
- pin_memory in {true,false}
- persistent_workers in {true,false}
- prefetch_factor in {2,4}
- cache in {none, monai_cache, persistent_disk}
AMP:
- dtype in {fp32, fp16, bf16}
- compare Dice and HD95 to fp32
- flag absolute Dice drift > 0.005
torch.compile:
- inference only by default
- torch.compile(model, mode="reduce-overhead")
- if compile fails, gracefully fall back to eager
- record compile_succeeded
- record compile_overhead_sec separately from steady_state_latency_sec
- do not compile training unless an explicit non-default flag is used
Combined:
- best data pipeline + best AMP dtype + compile if stable
- combined optimization reuses the same trained checkpoint unless explicitly retraining
- generated checkpoints are local run outputs and are not redistributed in the GitHub-clean frozen bundle

## Output JSON schema

Each benchmark run writes exactly one JSON row with this structure:
{
  "run_id": "str",
  "git_sha": "str",
  "config_hash": "str",
  "timestamp_utc": "str",
  "model": "str",
  "optimization": "str",
  "dataset": "str",
  "split": "str",
  "perturbation": {"name": "str", "severity": "int"} | null,
  "seed": "int",
  "hardware": {
    "gpu_name": "str|null",
    "cuda_version": "str|null",
    "driver_version": "str|null",
    "torch_version": "str",
    "monai_version": "str",
    "platform": "str",
    "cpu": "str",
    "ram_total_gb": "float|null"
  },
  "phase_times_sec": {
    "preprocess": "float",
    "dataload": "float",
    "infer": "float",
    "total": "float"
  },
  "latency_per_volume_sec_mean": "float",
  "latency_per_volume_sec_p50": "float",
  "latency_per_volume_sec_p95": "float",
  "throughput_volumes_per_sec": "float",
  "peak_vram_mb": "float|null",
  "gpu_util_pct_mean": "float|null",
  "compile_overhead_sec": "float|null",
  "steady_state_latency_sec": "float|null",
  "quality": {
    "dice_mean": "float",
    "dice_std": "float",
    "hd95_mean_mm": "float|null",
    "hd95_std_mm": "float|null",
    "n_cases": "int"
  },
  "optimization_metadata": {
    "amp_dtype": "str|null",
    "compile_succeeded": "bool|null",
    "quality_delta_dice_vs_fp32": "float|null",
    "quality_delta_hd95_vs_fp32": "float|null",
    "notes": "str|null"
  },
  "artifacts": {
    "predictions_dir": "str|null",
    "profiler_trace": "str|null"
  }
}

## Hardware logging

Every run logs:
- gpu_name
- cuda_version
- driver_version
- torch_version
- monai_version
- platform
- cpu
- ram_total_gb
On CPU-only systems, GPU fields are null.

## Run ledger

`RUNS.md` has one row per benchmark run. During a fresh run it is written at
the repository root; inside a frozen bundle it is copied to the bundle root.
- run_id
- git_sha
- config_hash
- JSON path
- model
- optimization
- perturbation
- one-line result
