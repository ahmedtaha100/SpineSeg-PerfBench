#!/usr/bin/env bash
set -euo pipefail
SMOKE_FLAG=""
if [ "${SMOKE:-0}" = "1" ]; then
    FREEZE_DIR="${FREEZE_DIR:-artifacts/smoke_frozen}"
elif [ "${OVERWRITE_SUBMISSION_BUNDLE:-0}" = "1" ]; then
    FREEZE_DIR="${FREEZE_DIR:-artifacts/frozen}"
else
    FREEZE_DIR="${FREEZE_DIR:-artifacts/local_frozen}"
fi
if [ "${SMOKE:-0}" = "1" ]; then
  SMOKE_FLAG="--smoke"
fi
if [[ "${PRESERVE_OUTPUTS:-0}" != "1" ]]; then
  CLEANUP_DIRS=(
    "outputs/manifests"
    "outputs/benchmarks"
    "outputs/profiles"
    "outputs/predictions"
    "outputs/inference"
    "outputs/cache"
    "outputs/runs"
    "outputs/tables"
    "outputs/figures"
    "${FREEZE_DIR}"
    "artifacts/demo"
    "tests/fixtures/synthetic"
  )
  if [ "${FREEZE_DIR}" = "artifacts/frozen" ] && [ "${OVERWRITE_SUBMISSION_BUNDLE:-0}" != "1" ]; then
    echo "ERROR: Refusing to delete artifacts/frozen/ without OVERWRITE_SUBMISSION_BUNDLE=1." >&2
    echo "       Set OVERWRITE_SUBMISSION_BUNDLE=1 to intentionally overwrite the submitted bundle." >&2
    exit 1
  fi
  rm -rf "${CLEANUP_DIRS[@]}"
  rm -f outputs/robustness_results*.csv
  python - <<'PY'
from pathlib import Path

Path("RUNS.md").write_text(
    "# Run Ledger\n\n"
    "| run_id | git_sha | config_hash | JSON path | model | optimization | perturbation | one-line result |\n"
    "|---|---|---|---|---|---|---|---|\n",
    encoding="utf-8",
)
PY
fi
if [[ "${SMOKE:-0}" == "1" ]]; then
  python scripts/prepare_data.py --synthetic ${SMOKE_FLAG}
else
  if [[ -z "${VERSE_ROOT:-}" && -z "${CTSPINE1K_ROOT:-}" ]]; then
    echo "ERROR: Set VERSE_ROOT or CTSPINE1K_ROOT for real runs."
    exit 1
  fi
  if [[ -n "${VERSE_ROOT:-}" ]]; then
    export SPINESEGBENCH_DATASET="${SPINESEGBENCH_DATASET:-verse}"
  else
    export SPINESEGBENCH_DATASET="${SPINESEGBENCH_DATASET:-ctspine1k}"
  fi
  PREPARE_ARGS=(--verse-root "${VERSE_ROOT:-}" --ctspine1k-root "${CTSPINE1K_ROOT:-}")
  if [[ -n "${IMAGE_GLOB:-}" ]]; then
    PREPARE_ARGS+=(--image-glob "${IMAGE_GLOB}")
  fi
  if [[ -n "${LABEL_GLOB:-}" ]]; then
    PREPARE_ARGS+=(--label-glob "${LABEL_GLOB}")
  fi
  python scripts/prepare_data.py "${PREPARE_ARGS[@]}"
fi
python scripts/train.py model=segresnet --run-name segresnet_baseline ${SMOKE_FLAG}
python scripts/train.py model=unet --run-name unet_baseline ${SMOKE_FLAG}
SEG_CKPT="outputs/runs/segresnet_baseline/checkpoint.pt"
UNET_CKPT="outputs/runs/unet_baseline/checkpoint.pt"
python scripts/benchmark.py --config opt_baseline --checkpoint "${SEG_CKPT}" ${SMOKE_FLAG}
python scripts/benchmark.py --config opt_baseline --checkpoint "${UNET_CKPT}" --run-id unet_baseline ${SMOKE_FLAG}
python scripts/profile.py --config opt_baseline --checkpoint "${SEG_CKPT}" ${SMOKE_FLAG}
python scripts/robustness.py --checkpoint "${SEG_CKPT}" --config opt_baseline ${SMOKE_FLAG}
python scripts/benchmark.py --config opt_data_pipeline --checkpoint "${SEG_CKPT}" --sweep ${SMOKE_FLAG}
python scripts/benchmark.py --config opt_amp --checkpoint "${SEG_CKPT}" --sweep ${SMOKE_FLAG}
python scripts/benchmark.py --config opt_compile --checkpoint "${SEG_CKPT}" ${SMOKE_FLAG}
python scripts/benchmark.py --config opt_all --checkpoint "${SEG_CKPT}" ${SMOKE_FLAG}
python scripts/robustness.py --checkpoint "${SEG_CKPT}" --config opt_all --output-suffix optimized ${SMOKE_FLAG}
python scripts/make_tables.py ${SMOKE_FLAG}
python scripts/make_plots.py ${SMOKE_FLAG}
python scripts/demo_single_case.py ${SMOKE_FLAG}
python scripts/freeze_artifacts.py --output-dir "${FREEZE_DIR}" ${SMOKE_FLAG}
python scripts/verify_artifacts.py "${FREEZE_DIR}" ${SMOKE_FLAG}
echo "DONE. Frozen bundle ready at ${FREEZE_DIR}/"
