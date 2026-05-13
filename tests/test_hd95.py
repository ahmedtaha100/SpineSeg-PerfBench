from __future__ import annotations

import numpy as np

from spineseg_perfbench.metrics.hd95 import compute_multiclass_hd95


def cuboid(offset=(0, 0, 0)):
    arr = np.zeros((8, 8, 8), dtype=int)
    z, y, x = offset
    arr[2 + z : 5 + z, 2 + y : 5 + y, 2 + x : 5 + x] = 1
    return arr


def test_identical_hd95_zero():
    gt = cuboid()
    assert compute_multiclass_hd95(gt, gt, (1, 1, 1), labels=[1])["case_mean"] == 0.0


def test_translated_cuboid_spacing_one():
    gt = cuboid()
    pred = cuboid((1, 0, 0))
    val = compute_multiclass_hd95(pred, gt, (1, 1, 1), labels=[1])["case_mean"]
    assert val > 0
    assert val < 2


def test_translated_cuboid_spacing_changes_physical_mm():
    gt = cuboid()
    pred = cuboid((1, 0, 0))
    one = compute_multiclass_hd95(pred, gt, (1, 1, 1), labels=[1])["case_mean"]
    two = compute_multiclass_hd95(pred, gt, (2, 1, 1), labels=[1])["case_mean"]
    assert two > one


def test_empty_prediction_nonempty_gt_nan():
    gt = cuboid()
    pred = np.zeros_like(gt)
    assert np.isnan(compute_multiclass_hd95(pred, gt, (1, 1, 1), labels=[1])["case_mean"])


def test_nanmean_aggregation():
    gt = np.zeros((8, 8, 8), dtype=int)
    pred = np.zeros_like(gt)
    gt[1:3, 1:3, 1:3] = 1
    pred[1:3, 1:3, 1:3] = 1
    gt[5:7, 5:7, 5:7] = 2
    result = compute_multiclass_hd95(pred, gt, (1, 1, 1), labels=[1, 2])
    assert result["case_mean"] == 0.0
    assert np.isnan(result["per_label"][2])
