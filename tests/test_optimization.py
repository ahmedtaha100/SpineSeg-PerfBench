from __future__ import annotations

import pytest
import torch

from spineseg_perfbench.optimization.amp import autocast_context
from spineseg_perfbench.optimization.dataloader import make_dataset


def test_unknown_amp_dtype_uses_noop_context():
    with autocast_context("invalid_dtype", torch.device("cpu")) as value:
        assert value is None


def test_unknown_cache_mode_fails_fast():
    with pytest.raises(ValueError, match="Unsupported cache mode"):
        make_dataset([], cache="persistant_disk")
