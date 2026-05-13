from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter

from spineseg_perfbench.utils.io import ensure_dir


def make_synthetic_volume(
    shape: tuple[int, int, int] = (64, 64, 64),
    n_vertebrae: int = 5,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    rng = np.random.default_rng(seed)
    z, y, x = np.indices(shape)
    image = rng.normal(loc=-700.0, scale=35.0, size=shape).astype(np.float32)
    image += np.linspace(0, 120, shape[0], dtype=np.float32)[:, None, None]
    label = np.zeros(shape, dtype=np.int16)

    z_centers = np.linspace(shape[0] * 0.18, shape[0] * 0.82, n_vertebrae)
    y_center = shape[1] / 2.0
    x_center = shape[2] / 2.0
    for idx, cz in enumerate(z_centers, start=1):
        ry = max(3.0, shape[1] / 9.5 + rng.uniform(-0.8, 0.8))
        rx = max(3.0, shape[2] / 10.5 + rng.uniform(-0.8, 0.8))
        rz = max(2.5, shape[0] / (n_vertebrae * 4.5))
        cy = y_center + rng.uniform(-1.5, 1.5)
        cx = x_center + rng.uniform(-1.5, 1.5)
        blob = ((z - cz) / rz) ** 2 + ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2 <= 1.0
        label[blob] = idx
        image[blob] = rng.normal(loc=850.0 + idx * 10.0, scale=45.0, size=int(blob.sum()))

        posterior = (
            ((z - cz) / (rz * 0.8)) ** 2
            + ((y - (cy + ry * 0.95)) / (ry * 0.35)) ** 2
            + ((x - cx) / (rx * 0.45)) ** 2
            <= 1.0
        )
        label[posterior] = idx
        image[posterior] = rng.normal(loc=950.0, scale=35.0, size=int(posterior.sum()))

    image = gaussian_filter(image, sigma=0.8).astype(np.float32)
    image = np.clip(image, -1024, 3071)
    return image, label, spacing


def write_synthetic_pair(
    image_path: str | Path,
    label_path: str | Path,
    shape: tuple[int, int, int] = (64, 64, 64),
    n_vertebrae: int = 5,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    seed: int = 42,
) -> tuple[Path, Path]:
    image, label, spacing = make_synthetic_volume(shape, n_vertebrae, spacing, seed)
    image_path = Path(image_path)
    label_path = Path(label_path)
    ensure_dir(image_path.parent)
    ensure_dir(label_path.parent)
    affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
    img_nii = nib.Nifti1Image(image.astype(np.float32), affine)
    lbl_nii = nib.Nifti1Image(label.astype(np.int16), affine)
    img_nii.set_qform(affine, code=1)
    img_nii.set_sform(affine, code=1)
    lbl_nii.set_qform(affine, code=1)
    lbl_nii.set_sform(affine, code=1)
    nib.save(img_nii, str(image_path))
    nib.save(lbl_nii, str(label_path))
    return image_path, label_path


def write_synthetic_dataset(
    out_dir: str | Path,
    n_cases: int = 4,
    shape: tuple[int, int, int] = (32, 32, 32),
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    seed: int = 42,
) -> list[tuple[Path, Path]]:
    out = ensure_dir(out_dir)
    pairs: list[tuple[Path, Path]] = []
    for i in range(n_cases):
        case = f"synthetic_{i:03d}"
        pairs.append(
            write_synthetic_pair(
                out / f"{case}_ct.nii.gz",
                out / f"{case}_seg.nii.gz",
                shape=shape,
                spacing=spacing,
                seed=seed + i,
            )
        )
    return pairs
