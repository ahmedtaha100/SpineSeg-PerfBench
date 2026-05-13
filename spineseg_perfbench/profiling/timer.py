from __future__ import annotations

import time
from contextlib import contextmanager
from collections import defaultdict


class PhaseTimer:
    def __init__(self) -> None:
        self.times = defaultdict(float)

    @contextmanager
    def phase(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.times[name] += time.perf_counter() - start

    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.times.items()}

    def benchmark_phases(self) -> dict[str, float]:
        d = self.as_dict()
        total = d.get("total", sum(v for k, v in d.items() if k != "total"))
        return {
            "preprocess": float(d.get("preprocess", 0.0)),
            "dataload": float(d.get("dataload", 0.0)),
            "infer": float(d.get("infer", 0.0)),
            "total": float(total),
        }
