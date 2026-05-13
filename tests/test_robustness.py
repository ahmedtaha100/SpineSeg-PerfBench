from __future__ import annotations

import numpy as np

from spineseg_perfbench.robustness.perturbations import PERTURBATION_NAMES, apply_perturbation


def test_perturbations_preserve_shape_and_labels():
    image = np.linspace(0, 1, 16**3, dtype=np.float32).reshape(16, 16, 16)
    label = np.zeros((16, 16, 16), dtype=np.int16)
    label[4:8, 4:8, 4:8] = 1
    for name in PERTURBATION_NAMES:
        clean, lbl = apply_perturbation(image, label, name=name, severity=0)
        assert np.array_equal(clean, image)
        assert np.array_equal(lbl, label)
        perturbed, lbl = apply_perturbation(image, label, name=name, severity=2)
        assert perturbed.shape == image.shape
        assert np.array_equal(lbl, label)
        assert not np.isnan(perturbed).any()
        assert not np.array_equal(perturbed, image)
