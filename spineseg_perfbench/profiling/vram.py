from __future__ import annotations

import subprocess
import shutil
import threading
import time

import numpy as np
import torch


def reset_peak_vram() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_vram_mb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated() / (1024**2))


def sample_gpu_utilization() -> float | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except Exception:
        return None
    vals = []
    for line in out.splitlines():
        try:
            vals.append(float(line.strip()))
        except ValueError:
            continue
    return float(np.mean(vals)) if vals else None


class GPUUtilizationSampler:
    """Sample GPU utilization while an operation is running."""

    def __init__(self, interval_sec: float = 1.0) -> None:
        self.interval_sec = interval_sec
        self._samples: list[float] = []
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "GPUUtilizationSampler":
        if shutil.which("nvidia-smi") is None:
            return self
        self._thread = threading.Thread(target=self._run, name="gpu-utilization-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_sec + 1.0))
        self._record_sample()

    def _record_sample(self) -> None:
        value = sample_gpu_utilization()
        if value is not None:
            with self._lock:
                self._samples.append(float(value))

    def _run(self) -> None:
        while not self._stop.is_set():
            self._record_sample()
            self._stop.wait(self.interval_sec)

    def mean(self) -> float | None:
        with self._lock:
            if not self._samples:
                return None
            return float(np.mean(self._samples))
