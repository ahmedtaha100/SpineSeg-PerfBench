from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.spatial import cKDTree


def _surface(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask.astype(bool)
    eroded = binary_erosion(mask, structure=np.ones((3, 3, 3), dtype=bool), border_value=0)
    surface = np.logical_xor(mask, eroded)
    return surface if surface.any() else mask.astype(bool)


def _crop_to_union(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(np.logical_or(pred, gt))
    if coords.size == 0:
        return pred, gt
    lo = np.maximum(coords.min(axis=0) - 1, 0)
    hi = np.minimum(coords.max(axis=0) + 2, pred.shape)
    slices = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi, strict=True))
    return pred[slices], gt[slices]


def _hd95_binary(pred: np.ndarray, gt: np.ndarray, spacing: tuple[float, float, float]) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if not gt.any() and not pred.any():
        return 0.0
    if gt.any() and not pred.any():
        return float("nan")
    if pred.any() and not gt.any():
        return float("nan")
    if np.array_equal(pred, gt):
        return 0.0
    pred, gt = _crop_to_union(pred, gt)
    pred_surface = _surface(pred)
    gt_surface = _surface(gt)
    # For large 3D CT crops, full-volume EDT can dominate runtime. A KD-tree over
    # surface voxels gives the same physical nearest-surface distance definition
    # without allocating distance volumes over empty space.
    pred_coords = np.argwhere(pred_surface)
    gt_coords = np.argwhere(gt_surface)
    if pred_coords.size and gt_coords.size and pred.size > 2_000_000:
        scale = np.asarray(spacing, dtype=float)
        pred_mm = pred_coords.astype(float) * scale
        gt_mm = gt_coords.astype(float) * scale
        dt_to_gt, _ = cKDTree(gt_mm).query(pred_mm, k=1, workers=-1)
        dt_to_pred, _ = cKDTree(pred_mm).query(gt_mm, k=1, workers=-1)
        distances = np.concatenate([dt_to_gt, dt_to_pred])
        return float(np.percentile(distances, 95)) if distances.size else 0.0
    dt_to_gt = distance_transform_edt(~gt_surface, sampling=spacing)
    dt_to_pred = distance_transform_edt(~pred_surface, sampling=spacing)
    distances = np.concatenate([dt_to_gt[pred_surface], dt_to_pred[gt_surface]])
    return float(np.percentile(distances, 95)) if distances.size else 0.0


def compute_multiclass_hd95(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: tuple[float, float, float],
    labels: list[int] | None = None,
) -> dict:
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    if pred.shape != gt.shape:
        raise ValueError(f"pred and gt shape mismatch: {pred.shape} vs {gt.shape}")
    if labels is None:
        labels = sorted(int(x) for x in np.union1d(pred, gt) if int(x) != 0)
    per_label: dict[int, float] = {}
    present_labels: list[int] = []
    for label in labels:
        gt_mask = gt == label
        pred_mask = pred == label
        if not gt_mask.any():
            continue
        present_labels.append(int(label))
        per_label[int(label)] = _hd95_binary(pred_mask, gt_mask, spacing)
    values = np.array([per_label[label] for label in present_labels], dtype=float)
    case_mean = float(np.nanmean(values)) if values.size and not np.all(np.isnan(values)) else float("nan")
    return {"per_label": per_label, "case_mean": case_mean, "present_labels": present_labels}
