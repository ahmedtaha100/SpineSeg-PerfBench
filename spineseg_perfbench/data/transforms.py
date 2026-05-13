from __future__ import annotations

import numpy as np


def normalize_ct(image: np.ndarray, clip_min: float = -1024.0, clip_max: float = 3071.0) -> np.ndarray:
    if clip_max <= clip_min:
        raise ValueError("clip_max must be greater than clip_min")
    image = np.clip(image.astype(np.float32), clip_min, clip_max)
    return ((image - clip_min) / (clip_max - clip_min)).astype(np.float32)
