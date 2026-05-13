#!/usr/bin/env python
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd

from spineseg_perfbench.utils.io import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit NIfTI label distributions before real GPU training.")
    p.add_argument("--manifest", default="outputs/manifests/data_manifest.csv")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--sample", type=int, default=25, help="Deterministic sample size to inspect.")
    group.add_argument("--all", action="store_true", help="Inspect every label file.")
    p.add_argument("--out-channels", type=int, default=26, help="Model output channels; labels must be < this value.")
    p.add_argument(
        "--allow-labels-above-out-channels-with-remap-note",
        action="store_true",
        help=(
            "Allow labels >= out_channels only when the run explicitly documents that those labels are outside the "
            "configured 1-25 vertebra label space and are remapped away by the training/inference pipeline."
        ),
    )
    p.add_argument("--output-json", default="outputs/audits/label_distribution_audit.json")
    p.add_argument("--output-md", default="LABEL_DISTRIBUTION_AUDIT.md")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def select_manifest_rows(manifest: pd.DataFrame, sample: int | None, seed: int = 42) -> pd.DataFrame:
    ordered = manifest.sort_values(["dataset_source", "case_id", "label_path"]).reset_index(drop=True)
    if sample is None or sample >= len(ordered):
        return ordered
    if sample <= 0:
        raise ValueError("--sample must be positive")
    sampled = ordered.sample(n=sample, random_state=seed)
    return sampled.sort_values(["dataset_source", "case_id", "label_path"]).reset_index(drop=True)


def _is_integer_valued(values: np.ndarray) -> bool:
    if values.size == 0:
        return True
    return bool(np.all(np.isfinite(values)) and np.allclose(values, np.round(values), rtol=0.0, atol=1e-6))


def inspect_label(row: pd.Series, out_channels: int) -> dict[str, Any]:
    label_path = Path(str(row["label_path"]))
    item: dict[str, Any] = {
        "case_id": str(row["case_id"]),
        "label_path": str(label_path),
        "readable": False,
        "error": None,
    }
    try:
        nii = nib.load(str(label_path))
        arr = np.asanyarray(nii.dataobj)
        finite = arr[np.isfinite(arr)]
        unique = np.unique(finite) if finite.size else np.asarray([], dtype=np.float32)
        integer_valued = _is_integer_valued(unique)
        rounded = np.round(unique).astype(np.int64) if integer_valued and unique.size else np.asarray([], dtype=np.int64)
        nonzero = unique[np.abs(unique) > 1e-6]
        item.update(
            {
                "readable": True,
                "min_label": None if unique.size == 0 else float(unique.min()),
                "max_label": None if unique.size == 0 else float(unique.max()),
                "n_unique": int(unique.size),
                "first_50_unique_labels": unique[:50].tolist(),
                "integer_valued": integer_valued,
                "appears_empty": bool(nonzero.size == 0),
                "shape": list(arr.shape[:3]),
                "spacing": [float(x) for x in nii.header.get_zooms()[:3]],
                "has_negative_labels": bool(unique.size > 0 and unique.min() < 0),
                "exceeds_out_channels": bool(rounded.size > 0 and rounded.max() >= out_channels),
            }
        )
    except Exception as exc:
        item["error"] = repr(exc)
    return item


def build_audit(
    manifest_path: str | Path,
    sample: int | None = 25,
    out_channels: int = 26,
    seed: int = 42,
    allow_exceeding_with_remap_note: bool = False,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest = pd.read_csv(manifest_path)
    if manifest.empty:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    required = {"case_id", "label_path"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    selected = select_manifest_rows(manifest, sample=sample, seed=seed)
    cases = [inspect_label(row, out_channels=out_channels) for _, row in selected.iterrows()]
    readable = [c for c in cases if c.get("readable")]
    errors = [f"{c['case_id']}: {c['error']}" for c in cases if not c.get("readable")]
    empty_count = sum(1 for c in readable if c["appears_empty"])
    non_integer_count = sum(1 for c in readable if not c["integer_valued"])
    exceeding_count = sum(1 for c in readable if c["exceeds_out_channels"])
    negative_count = sum(1 for c in readable if c["has_negative_labels"])

    observed_values: list[float] = []
    for c in readable:
        observed_values.extend(float(v) for v in c["first_50_unique_labels"])
    observed = np.asarray(observed_values, dtype=np.float64)
    union_labels = sorted({int(round(v)) for v in observed_values if np.isfinite(v) and abs(v - round(v)) <= 1e-6})
    foreground = [v for v in union_labels if v != 0]

    failures = list(errors)
    if not readable:
        failures.append("No label files could be read.")
    if readable and empty_count == len(readable):
        failures.append("All inspected label masks are empty.")
    if negative_count:
        failures.append(f"{negative_count} inspected label file(s) contain negative labels.")
    if non_integer_count:
        failures.append(f"{non_integer_count} inspected label file(s) contain non-integer-valued labels.")
    notes: list[str] = []
    if exceeding_count and allow_exceeding_with_remap_note:
        notes.append(
            f"{exceeding_count} inspected label file(s) contain labels >= out_channels ({out_channels}). "
            "This run explicitly treats those labels as outside the configured vertebra label space 1-25; "
            "the training pipeline clamps labels outside [0,25] to background."
        )
    elif exceeding_count:
        failures.append(
            f"{exceeding_count} inspected label file(s) contain labels >= out_channels ({out_channels}); "
            "add an explicit documented remapping before training."
        )
    if readable and not foreground:
        failures.append("No foreground labels were observed in inspected masks.")

    return {
        "verdict": "pass" if not failures else "fail",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path),
        "sample_size_requested": sample,
        "sample_size_inspected": len(cases),
        "out_channels": out_channels,
        "expected_labels": {"background": 0, "vertebrae": list(range(1, out_channels))},
        "aggregate": {
            "global_min_label": None if observed.size == 0 else float(observed.min()),
            "global_max_label": None if observed.size == 0 else float(observed.max()),
            "observed_label_union": union_labels,
            "empty_mask_count": empty_count,
            "non_integer_label_count": non_integer_count,
            "exceeds_out_channels_count": exceeding_count,
            "negative_label_count": negative_count,
            "unreadable_count": len(errors),
        },
        "failures": failures,
        "notes": notes,
        "cases": cases,
    }


def write_markdown(audit: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    lines = [
        "# Label Distribution Audit",
        "",
        f"Verdict: {audit['verdict'].upper()}",
        "",
        f"Created UTC: {audit['created_utc']}",
        f"Manifest: `{audit['manifest']}`",
        f"Inspected labels: {audit['sample_size_inspected']}",
        f"Out channels: {audit['out_channels']}",
        "",
        "## Aggregate",
        "",
    ]
    for k, v in audit["aggregate"].items():
        lines.append(f"- {k}: {v}")
    lines.extend(["", "## Failures", ""])
    if audit["failures"]:
        lines.extend(f"- {x}" for x in audit["failures"])
    else:
        lines.append("- None")
    lines.extend(["", "## Notes", ""])
    if audit.get("notes"):
        lines.extend(f"- {x}" for x in audit["notes"])
    else:
        lines.append("- None")
    lines.extend(["", "## Cases", ""])
    lines.append("| case_id | min | max | n_unique | integer | empty | shape | spacing | first values |")
    lines.append("|---|---:|---:|---:|---:|---:|---|---|---|")
    for c in audit["cases"]:
        if not c.get("readable"):
            lines.append(f"| {c['case_id']} | | | | | | | | ERROR: {c.get('error')} |")
            continue
        lines.append(
            "| {case_id} | {min_label} | {max_label} | {n_unique} | {integer_valued} | {appears_empty} | "
            "{shape} | {spacing} | {first_50_unique_labels} |".format(**c)
        )
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    sample = None if args.all else args.sample
    audit = build_audit(
        args.manifest,
        sample=sample,
        out_channels=args.out_channels,
        seed=args.seed,
        allow_exceeding_with_remap_note=args.allow_labels_above_out_channels_with_remap_note,
    )
    write_json(args.output_json, audit)
    write_markdown(audit, args.output_md)
    print(args.output_md)
    print(args.output_json)
    if audit["verdict"] != "pass":
        for failure in audit["failures"]:
            print(f"ERROR: {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
