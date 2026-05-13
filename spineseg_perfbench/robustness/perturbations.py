from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

NOISE_STD = [0.0, 0.01, 0.025, 0.05]
BLUR_SIGMA = [0.0, 0.5, 1.0, 2.0]
RESAMPLE_FACTOR = [1.0, 1.25, 1.5, 2.0]
INTENSITY_OFFSET = [0.0, 0.05, 0.10, 0.20]
CONTRAST_GAMMA = [1.0, 1.1, 1.25, 1.5]
PERTURBATION_NAMES = ["gaussian_noise", "gaussian_blur", "downsample_resample", "intensity_shift", "contrast_shift"]


def apply_perturbation(
    image: np.ndarray,
    label: np.ndarray | None = None,
    name: str = "gaussian_noise",
    severity: int = 0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray | None]:
    if severity < 0 or severity > 3:
        raise ValueError("severity must be in {0,1,2,3}")
    image = np.asarray(image, dtype=np.float32)
    out = image.copy()
    if severity == 0:
        return out, None if label is None else np.asarray(label).copy()
    if name == "gaussian_noise":
        rng = np.random.default_rng(seed)
        scale = float(np.nanstd(out) or 1.0) * NOISE_STD[severity]
        out = out + rng.normal(0.0, scale, size=out.shape).astype(np.float32)
    elif name == "gaussian_blur":
        out = gaussian_filter(out, sigma=BLUR_SIGMA[severity]).astype(np.float32)
    elif name == "downsample_resample":
        factor = RESAMPLE_FACTOR[severity]
        down = zoom(out, zoom=1.0 / factor, order=1)
        out = zoom(down, zoom=np.array(out.shape) / np.array(down.shape), order=1)
        padded = np.zeros_like(image)
        crop = tuple(slice(0, min(src, dst)) for src, dst in zip(out.shape, image.shape, strict=True))
        padded[crop] = out[crop]
        out = padded
    elif name == "intensity_shift":
        out = out + INTENSITY_OFFSET[severity]
    elif name == "contrast_shift":
        min_v = float(np.min(out))
        max_v = float(np.max(out))
        scaled = (out - min_v) / max(max_v - min_v, 1e-6)
        out = np.power(np.clip(scaled, 0.0, 1.0), CONTRAST_GAMMA[severity]) * (max_v - min_v) + min_v
    else:
        raise ValueError(f"Unknown perturbation: {name}")
    out = np.nan_to_num(out.astype(np.float32))
    return out, None if label is None else np.asarray(label).copy()
