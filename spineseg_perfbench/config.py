from __future__ import annotations

from pathlib import Path
from typing import Any
import os

import yaml
from omegaconf import OmegaConf

from spineseg_perfbench.utils.hashing import stable_hash


def _config_path(name: str | None) -> Path | None:
    if not name:
        return None
    p = Path(name)
    if p.exists():
        return p
    if not p.suffix:
        p = Path("configs") / f"{name}.yaml"
    else:
        p = Path("configs") / name
    return p


def load_config(config: str | None = None, overrides: list[str] | None = None, smoke: bool = False) -> dict[str, Any]:
    base = OmegaConf.load("configs/base.yaml")
    parts = [base]
    cp = _config_path(config)
    if cp is not None:
        if not cp.exists():
            raise FileNotFoundError(f"Config not found: {cp}")
        parts.append(OmegaConf.load(cp))
    if overrides:
        normalized = []
        for item in overrides:
            if item.startswith("--"):
                continue
            if "=" in item:
                normalized.append(item)
        if normalized:
            parts.append(OmegaConf.from_dotlist(normalized))
    cfg = OmegaConf.merge(*parts)
    if smoke:
        cfg.smoke.enabled = True
        cfg.preprocess.patch_size = cfg.smoke.patch_size
        cfg.training.epochs = cfg.training.smoke_epochs
        cfg.training.batch_size = cfg.training.smoke_batch_size
        cfg.inference.warmup_volumes = cfg.inference.smoke_warmup_volumes
        cfg.device = "cpu"
    dataset_name = os.environ.get("SPINESEGBENCH_DATASET")
    if dataset_name:
        cfg.dataset = dataset_name
    return OmegaConf.to_container(cfg, resolve=True)


def config_hash(cfg: dict[str, Any]) -> str:
    return stable_hash(cfg)


def write_config(path: str | Path, cfg: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=True)
