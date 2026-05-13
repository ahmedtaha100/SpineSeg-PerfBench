from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from scripts.label_distribution_audit import build_audit


def _write_label(path: Path, data: np.ndarray) -> None:
    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    img.set_qform(affine, code=1)
    img.set_sform(affine, code=1)
    nib.save(img, str(path))


def _manifest(tmp_path: Path, arrays: list[np.ndarray]) -> Path:
    rows = []
    for i, arr in enumerate(arrays):
        label = tmp_path / f"case_{i:03d}_seg.nii.gz"
        _write_label(label, arr)
        rows.append(
            {
                "case_id": f"case_{i:03d}",
                "image_path": str(label),
                "label_path": str(label),
                "dataset_source": "synthetic_test",
                "spacing_x": 1.0,
                "spacing_y": 1.0,
                "spacing_z": 1.0,
                "shape_x": arr.shape[0],
                "shape_y": arr.shape[1],
                "shape_z": arr.shape[2],
            }
        )
    path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_valid_labels_pass(tmp_path: Path) -> None:
    arr = np.zeros((4, 4, 4), dtype=np.int16)
    arr[1:3, 1:3, 1:3] = 5
    audit = build_audit(_manifest(tmp_path, [arr]), sample=None, out_channels=26)
    assert audit["verdict"] == "pass"


def test_negative_label_fails(tmp_path: Path) -> None:
    arr = np.zeros((4, 4, 4), dtype=np.int16)
    arr[0, 0, 0] = -1
    audit = build_audit(_manifest(tmp_path, [arr]), sample=None, out_channels=26)
    assert audit["verdict"] == "fail"
    assert any("negative" in x for x in audit["failures"])


def test_non_integer_label_fails(tmp_path: Path) -> None:
    arr = np.zeros((4, 4, 4), dtype=np.float32)
    arr[0, 0, 0] = 1.5
    audit = build_audit(_manifest(tmp_path, [arr]), sample=None, out_channels=26)
    assert audit["verdict"] == "fail"
    assert any("non-integer" in x for x in audit["failures"])


def test_label_exceeding_out_channels_fails(tmp_path: Path) -> None:
    arr = np.zeros((4, 4, 4), dtype=np.int16)
    arr[0, 0, 0] = 26
    audit = build_audit(_manifest(tmp_path, [arr]), sample=None, out_channels=26)
    assert audit["verdict"] == "fail"
    assert any("out_channels" in x for x in audit["failures"])


def test_label_exceeding_out_channels_can_pass_with_explicit_remap_note(tmp_path: Path) -> None:
    arr = np.zeros((4, 4, 4), dtype=np.int16)
    arr[0, 0, 0] = 26
    arr[1, 1, 1] = 1
    audit = build_audit(
        _manifest(tmp_path, [arr]),
        sample=None,
        out_channels=26,
        allow_exceeding_with_remap_note=True,
    )
    assert audit["verdict"] == "pass"
    assert audit["notes"]


def test_all_empty_labels_fail(tmp_path: Path) -> None:
    arr = np.zeros((4, 4, 4), dtype=np.int16)
    audit = build_audit(_manifest(tmp_path, [arr, arr.copy()]), sample=None, out_channels=26)
    assert audit["verdict"] == "fail"
    assert any("empty" in x for x in audit["failures"])


def test_deterministic_sampling_is_stable(tmp_path: Path) -> None:
    arrays = []
    for i in range(8):
        arr = np.zeros((4, 4, 4), dtype=np.int16)
        arr[1, 1, 1] = i % 5 + 1
        arrays.append(arr)
    manifest = _manifest(tmp_path, arrays)
    first = build_audit(manifest, sample=3, out_channels=26, seed=123)
    second = build_audit(manifest, sample=3, out_channels=26, seed=123)
    assert [x["case_id"] for x in first["cases"]] == [x["case_id"] for x in second["cases"]]
