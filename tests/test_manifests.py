from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from spineseg_perfbench.data.manifests import deterministic_split, discover_pairs, normalize_case_id


def _df(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": [f"case_{i:04d}" for i in range(n)],
            "image_path": [str(Path(f"image_{i:04d}.nii.gz")) for i in range(n)],
            "label_path": [str(Path(f"label_{i:04d}.nii.gz")) for i in range(n)],
            "dataset_source": ["synthetic"] * n,
        }
    )


def test_deterministic_split_uses_spec_ratios_for_real_sized_groups():
    splits = deterministic_split(_df(100), seed=42)
    assert {k: len(v) for k, v in splits.items()} == {"train": 70, "val": 15, "test": 15}


def test_deterministic_split_seed_changes_membership():
    first = deterministic_split(_df(20), seed=42)
    second = deterministic_split(_df(20), seed=1337)
    assert set(first["train"]["case_id"]) != set(second["train"]["case_id"])


def test_deterministic_split_has_no_overlap():
    splits = deterministic_split(_df(4), seed=42)
    sets = {k: set(v["case_id"]) for k, v in splits.items()}
    assert sets["train"].isdisjoint(sets["val"])
    assert sets["train"].isdisjoint(sets["test"])
    assert sets["val"].isdisjoint(sets["test"])


def _write_nifti(path: Path) -> None:
    data = np.zeros((4, 4, 4), dtype=np.float32)
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))


def test_discover_pairs_rejects_orphan_labels(tmp_path):
    _write_nifti(tmp_path / "case_001_ct.nii.gz")
    _write_nifti(tmp_path / "case_001_seg.nii.gz")
    _write_nifti(tmp_path / "case_002_seg.nii.gz")
    with pytest.raises(ValueError, match="labels without matching images"):
        discover_pairs([(tmp_path, "verse")])


def test_discover_pairs_rejects_duplicate_normalized_image_ids(tmp_path):
    _write_nifti(tmp_path / "case_001_ct.nii.gz")
    _write_nifti(tmp_path / "case_001_img.nii.gz")
    _write_nifti(tmp_path / "case_001_seg.nii.gz")
    with pytest.raises(ValueError, match="multiple images"):
        discover_pairs([(tmp_path, "verse")])


def test_discover_pairs_rejects_duplicate_case_ids_across_roots(tmp_path):
    first = tmp_path / "root_a"
    second = tmp_path / "root_b"
    first.mkdir()
    second.mkdir()
    for root in [first, second]:
        _write_nifti(root / "case_001_ct.nii.gz")
        _write_nifti(root / "case_001_seg.nii.gz")
    with pytest.raises(ValueError, match="duplicate dataset_source/case_id"):
        discover_pairs([(first, "verse"), (second, "verse")])


def test_normalize_case_id_strips_labels_suffix_before_label_suffix():
    assert normalize_case_id("case_001_labels.nii.gz") == "case_001"
