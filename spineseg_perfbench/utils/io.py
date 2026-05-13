from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return sanitize_for_json(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(data), f, indent=2, sort_keys=True)
        f.write("\n")


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_nifti(path: str | Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    img = nib.load(str(path))
    data = np.asarray(img.get_fdata(dtype=np.float32))
    zooms = img.header.get_zooms()[:3]
    return data, tuple(float(z) for z in zooms)


def save_nifti(path: str | Path, data: np.ndarray, spacing: tuple[float, float, float]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
    nib.save(nib.Nifti1Image(data, affine), str(p))


def append_run_ledger(
    run_id: str,
    git_sha: str,
    config_hash: str,
    json_path: str,
    model: str,
    optimization: str,
    perturbation: str,
    result: str,
    ledger_path: str | Path = "RUNS.md",
) -> None:
    p = Path(ledger_path)
    if not p.exists():
        p.write_text(
            "# Run Ledger\n\n"
            "| run_id | git_sha | config_hash | JSON path | model | optimization | perturbation | one-line result |\n"
            "|---|---|---|---|---|---|---|---|\n",
            encoding="utf-8",
        )
    existing = p.read_text(encoding="utf-8")
    row = f"| {run_id} | {git_sha} | {config_hash} | {json_path} | {model} | {optimization} | {perturbation} | {result} |\n"
    if f"| {run_id} |" not in existing:
        with p.open("a", encoding="utf-8") as f:
            f.write(row)
