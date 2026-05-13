from __future__ import annotations

import re
from pathlib import Path

import nibabel as nib
import pandas as pd

from spineseg_perfbench.data.synthetic import write_synthetic_dataset
from spineseg_perfbench.utils.io import ensure_dir

DEFAULT_IMAGE_GLOBS = ["**/*ct*.nii.gz", "**/*img*.nii.gz", "**/*_ct.nii.gz"]
DEFAULT_LABEL_GLOBS = ["**/*seg*.nii.gz", "**/*msk*.nii.gz", "**/*_seg-vert_msk.nii.gz"]


def _strip_nii(name: str) -> str:
    return re.sub(r"\.nii(\.gz)?$", "", name, flags=re.IGNORECASE)


def normalize_case_id(path: str | Path) -> str:
    stem = _strip_nii(Path(path).name).lower()
    replacements = [
        "_seg-vert_msk",
        "_labels",
        "_label",
        "_seg",
        "_msk",
        "_mask",
        "_ct",
        "_img",
        "_image",
    ]
    for marker in replacements:
        stem = stem.replace(marker, "")
    stem = re.sub(r"(^|[_-])(seg|msk|mask|label|labels|ct|img|image)([_-]|$)", "_", stem)
    return re.sub(r"[_-]+", "_", stem).strip("_")


def _collect(root: Path, patterns: list[str]) -> list[Path]:
    files: set[Path] = set()
    for pattern in patterns:
        files.update(root.glob(pattern))
    return sorted(p.resolve() for p in files if p.is_file())


def discover_pairs(
    roots: list[tuple[Path, str]],
    image_glob: str | None = None,
    label_glob: str | None = None,
) -> pd.DataFrame:
    rows = []
    img_patterns = [image_glob] if image_glob else DEFAULT_IMAGE_GLOBS
    lbl_patterns = [label_glob] if label_glob else DEFAULT_LABEL_GLOBS
    for root, source in roots:
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {root}")
        images = _collect(root, img_patterns)
        labels = _collect(root, lbl_patterns)
        image_by_case: dict[str, list[Path]] = {}
        for img in images:
            image_by_case.setdefault(normalize_case_id(img), []).append(img)
        duplicate_image_cases = {case: paths for case, paths in image_by_case.items() if len(paths) != 1}
        if duplicate_image_cases:
            raise ValueError(f"Found multiple images for normalized case IDs: {duplicate_image_cases}")
        label_by_case: dict[str, list[Path]] = {}
        for lbl in labels:
            label_by_case.setdefault(normalize_case_id(lbl), []).append(lbl)
        used_label_cases: set[str] = set()
        for img in images:
            case_id = normalize_case_id(img)
            matches = label_by_case.get(case_id, [])
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one label for image {img}, found {len(matches)}: {matches}")
            used_label_cases.add(case_id)
            rows.append(_manifest_row(case_id, img, matches[0], source))
        orphan_label_cases = sorted(set(label_by_case) - used_label_cases)
        if orphan_label_cases:
            raise ValueError(f"Found labels without matching images for cases: {orphan_label_cases}")
    if not rows:
        raise ValueError("No image-label pairs discovered")
    df = pd.DataFrame(rows)
    duplicates = df[df.duplicated(["dataset_source", "case_id"], keep=False)]
    if not duplicates.empty:
        duplicate_keys = sorted({(str(row.dataset_source), str(row.case_id)) for row in duplicates.itertuples()})
        raise ValueError(f"Found duplicate dataset_source/case_id pairs across roots: {duplicate_keys}")
    return df.sort_values(["dataset_source", "case_id", "image_path"]).reset_index(drop=True)


def _manifest_row(case_id: str, image: Path, label: Path, source: str) -> dict:
    nii = nib.load(str(image))
    shape = nii.shape[:3]
    spacing = nii.header.get_zooms()[:3]
    return {
        "case_id": case_id,
        "image_path": str(image.resolve()),
        "label_path": str(label.resolve()),
        "dataset_source": source,
        "spacing_x": float(spacing[0]),
        "spacing_y": float(spacing[1]),
        "spacing_z": float(spacing[2]),
        "shape_x": int(shape[0]),
        "shape_y": int(shape[1]),
        "shape_z": int(shape[2]),
    }


def synthetic_manifest(out_dir: str | Path = "tests/fixtures/synthetic", n_cases: int = 4, shape=(32, 32, 32)) -> pd.DataFrame:
    pairs = write_synthetic_dataset(out_dir, n_cases=n_cases, shape=tuple(shape))
    rows = [_manifest_row(normalize_case_id(img), img, lbl, "synthetic") for img, lbl in pairs]
    return pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)


def deterministic_split(df: pd.DataFrame, seed: int = 42) -> dict[str, pd.DataFrame]:
    train_parts = []
    val_parts = []
    test_parts = []
    for _, group in df.sort_values(["dataset_source", "case_id"]).groupby("dataset_source", sort=True):
        shuffled = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = len(shuffled)
        if n == 0:
            n_train, n_val = 0, 0
        elif n == 1:
            n_train, n_val = 1, 0
        elif n == 2:
            n_train, n_val = 1, 0
        elif n == 3:
            n_train, n_val = 1, 1
        else:
            n_train = max(1, int(round(n * 0.70)))
            n_val = max(1, int(round(n * 0.15)))
            if n_train + n_val >= n:
                n_train = n - n_val - 1
        train_parts.append(shuffled.iloc[:n_train])
        val_parts.append(shuffled.iloc[n_train : n_train + n_val])
        test_parts.append(shuffled.iloc[n_train + n_val :])
    splits = {
        "train": pd.concat(train_parts, ignore_index=True) if train_parts else df.iloc[0:0],
        "val": pd.concat(val_parts, ignore_index=True) if val_parts else df.iloc[0:0],
        "test": pd.concat(test_parts, ignore_index=True) if test_parts else df.iloc[0:0],
    }
    if len(splits["test"]) == 0 and len(df) > 0:
        if len(splits["val"]) > 0:
            splits["test"] = splits["val"].tail(1).copy()
            splits["val"] = splits["val"].iloc[:-1].copy()
        elif len(splits["train"]) > 1:
            splits["test"] = splits["train"].tail(1).copy()
            splits["train"] = splits["train"].iloc[:-1].copy()
    return {k: v.sort_values(["dataset_source", "case_id"]).reset_index(drop=True) for k, v in splits.items()}


def write_manifest_and_splits(df: pd.DataFrame, out_dir: str | Path = "outputs/manifests") -> dict[str, Path]:
    out = ensure_dir(out_dir)
    columns = [
        "case_id",
        "image_path",
        "label_path",
        "dataset_source",
        "spacing_x",
        "spacing_y",
        "spacing_z",
        "shape_x",
        "shape_y",
        "shape_z",
    ]
    df = df[columns].sort_values(["dataset_source", "case_id"]).reset_index(drop=True)
    paths = {"all": out / "data_manifest.csv"}
    df.to_csv(paths["all"], index=False, float_format="%.6f", lineterminator="\n")
    for name, split_df in deterministic_split(df).items():
        p = out / f"split_{name}.csv"
        split_df[columns].to_csv(p, index=False, float_format="%.6f", lineterminator="\n")
        paths[name] = p
    return paths
