from __future__ import annotations

import numpy as np


def compute_multiclass_dice(pred: np.ndarray, gt: np.ndarray, labels: list[int] | None = None) -> dict:
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    if pred.shape != gt.shape:
        raise ValueError(f"pred and gt shape mismatch: {pred.shape} vs {gt.shape}")
    if labels is None:
        labels = sorted(int(x) for x in np.union1d(pred, gt) if int(x) != 0)
    else:
        labels = list(dict.fromkeys(int(x) for x in labels if int(x) != 0))
    per_label: dict[int, float] = {}
    present_labels: list[int] = []
    for label in labels:
        label_int = int(label)
        pred_mask = pred == label_int
        gt_mask = gt == label_int
        if not gt_mask.any():
            per_label[label_int] = 1.0 if not pred_mask.any() else 0.0
            if pred_mask.any():
                present_labels.append(label_int)
            continue
        present_labels.append(label_int)
        if not pred_mask.any():
            per_label[label_int] = 0.0
            continue
        denom = float(pred_mask.sum() + gt_mask.sum())
        per_label[label_int] = 1.0 if denom == 0 else float(2.0 * np.logical_and(pred_mask, gt_mask).sum() / denom)
    case_values = [per_label[label] for label in present_labels if label in per_label]
    if case_values:
        case_mean = float(np.mean(case_values))
    else:
        false_positive_labels = [label for label in labels if per_label.get(label) == 0.0]
        case_mean = 0.0 if false_positive_labels else 1.0
    return {"per_label": per_label, "case_mean": case_mean, "present_labels": present_labels}
