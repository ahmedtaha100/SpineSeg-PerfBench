from __future__ import annotations

from pathlib import Path

from monai.data import CacheDataset, Dataset, PersistentDataset
from torch.utils.data import DataLoader

from spineseg_perfbench.utils.io import ensure_dir


def make_dataset(items, transform=None, cache: str = "none", smoke: bool = False, cache_root: str | Path = "outputs/cache"):
    cache = "none" if smoke and cache == "persistent_disk" else cache
    if cache == "monai_cache":
        return CacheDataset(items, transform=transform, cache_rate=1.0 if smoke else 0.25, num_workers=0)
    if cache == "persistent_disk":
        cache_dir = ensure_dir(Path(cache_root) / "persistent_dataset")
        return PersistentDataset(items, transform=transform, cache_dir=str(cache_dir))
    if cache == "none":
        return Dataset(items, transform=transform)
    raise ValueError(f"Unsupported cache mode: {cache}. Expected one of: none, monai_cache, persistent_disk")


def make_dataloader(dataset, batch_size: int = 1, shuffle: bool = False, settings: dict | None = None, smoke: bool = False):
    settings = settings or {}
    num_workers = int(settings.get("num_workers", 0))
    if smoke:
        num_workers = 0
    pin_memory = bool(settings.get("pin_memory", False)) and not smoke
    persistent_workers = bool(settings.get("persistent_workers", False)) and num_workers > 0
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = int(settings.get("prefetch_factor", 2))
    return DataLoader(dataset, **kwargs)
