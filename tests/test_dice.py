from __future__ import annotations

import numpy as np

from spineseg_perfbench.metrics.dice import compute_multiclass_dice


def test_identical_binary_masks_dice_one():
    gt = np.zeros((4, 4, 4), dtype=int)
    gt[1:3, 1:3, 1:3] = 1
    assert compute_multiclass_dice(gt, gt, labels=[1])["case_mean"] == 1.0


def test_disjoint_masks_dice_zero():
    gt = np.zeros((4, 4, 4), dtype=int)
    pred = np.zeros_like(gt)
    gt[:2] = 1
    pred[2:] = 1
    assert compute_multiclass_dice(pred, gt, labels=[1])["case_mean"] == 0.0


def test_partial_overlap_hand_computed():
    gt = np.zeros((5,), dtype=int)
    pred = np.zeros((5,), dtype=int)
    gt[:4] = 1
    pred[2:] = 1
    # intersection=2, pred=3, gt=4 -> 4/7
    assert np.isclose(compute_multiclass_dice(pred, gt, labels=[1])["case_mean"], 4 / 7)


def test_multiclass_present_labels():
    gt = np.array([0, 1, 1, 2, 2, 3])
    pred = np.array([0, 1, 0, 2, 0, 3])
    result = compute_multiclass_dice(pred, gt, labels=[1, 2, 3])
    assert result["present_labels"] == [1, 2, 3]
    assert result["per_label"][3] == 1.0


def test_absent_class_policy_and_empty_cases():
    gt = np.zeros((4,), dtype=int)
    pred = np.zeros((4,), dtype=int)
    result = compute_multiclass_dice(pred, gt, labels=[1])
    assert result["per_label"][1] == 1.0
    assert result["case_mean"] == 1.0
    assert result["present_labels"] == []


def test_gt_empty_prediction_nonempty_is_zero():
    gt = np.zeros((4,), dtype=int)
    pred = np.array([0, 1, 0, 0])
    result = compute_multiclass_dice(pred, gt, labels=[1])
    assert result["per_label"][1] == 0.0
    assert result["case_mean"] == 0.0


def test_gt_nonempty_pred_empty_is_zero():
    gt = np.array([0, 1, 1])
    pred = np.array([0, 0, 0])
    assert compute_multiclass_dice(pred, gt, labels=[1])["case_mean"] == 0.0


def test_duplicate_requested_labels_do_not_bias_case_mean():
    gt = np.array([0, 1, 2, 2])
    pred = np.array([0, 1, 0, 0])
    result = compute_multiclass_dice(pred, gt, labels=[1, 1, 2])
    assert result["present_labels"] == [1, 2]
    assert result["case_mean"] == 0.5
