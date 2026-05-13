from __future__ import annotations

import os
import time

import torch


def try_compile(model, mode: str = "reduce-overhead", enabled: bool = True, force_fail: bool = False):
    if not enabled:
        return model, None, None, None
    start = time.perf_counter()
    try:
        if force_fail or os.environ.get("SPINESEG_FORCE_COMPILE_FAIL") == "1":
            raise RuntimeError("forced compile failure")
        compiled = torch.compile(model, mode=mode)
        return compiled, True, float(time.perf_counter() - start), None
    except Exception as exc:
        return model, False, float(time.perf_counter() - start), f"torch.compile fallback: {exc}"
