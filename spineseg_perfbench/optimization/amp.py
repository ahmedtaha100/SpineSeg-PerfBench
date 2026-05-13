from __future__ import annotations

from contextlib import nullcontext

import torch


def autocast_context(dtype: str | None, device: torch.device):
    dtype = (dtype or "fp32").lower()
    if dtype == "fp32":
        return nullcontext()
    if dtype == "fp16":
        torch_dtype = torch.float16
    elif dtype == "bf16":
        torch_dtype = torch.bfloat16
    else:
        return nullcontext()
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch_dtype)
    if device.type == "cpu" and dtype == "bf16":
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    return nullcontext()


def effective_amp_dtype(dtype: str | None, device: torch.device) -> tuple[str, str | None]:
    dtype = (dtype or "fp32").lower()
    if dtype == "fp16" and device.type != "cuda":
        return "fp32", "fp16 unsupported on CPU; used fp32"
    if dtype == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        return "fp32", "bf16 unsupported on this CUDA device; used fp32"
    if dtype == "bf16" and device.type == "cpu":
        return "bf16", None
    if dtype in {"fp32", "fp16", "bf16"}:
        return dtype, None
    return "fp32", f"unknown AMP dtype {dtype}; used fp32"
