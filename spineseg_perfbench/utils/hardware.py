from __future__ import annotations

import platform
import shutil
import subprocess

import psutil
import torch

try:
    import monai
except ImportError:  # pragma: no cover
    monai = None


def _nvidia_smi_value(query: str) -> str | None:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return None
    try:
        out = subprocess.check_output(
            [nvidia_smi, f"--query-gpu={query}", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired):
        return None
    return out.splitlines()[0].strip() if out else None


def collect_hardware_metadata() -> dict:
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    driver = _nvidia_smi_value("driver_version") if gpu_name else None
    try:
        ram_total_gb = round(psutil.virtual_memory().total / (1024**3), 3)
    except (OSError, RuntimeError):
        ram_total_gb = None
    return {
        "gpu_name": gpu_name,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "driver_version": driver,
        "torch_version": torch.__version__,
        "monai_version": getattr(monai, "__version__", "unknown"),
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "ram_total_gb": ram_total_gb,
    }
