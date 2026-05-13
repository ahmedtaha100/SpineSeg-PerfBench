from __future__ import annotations

from pathlib import Path

import torch

from spineseg_perfbench.utils.io import ensure_dir


class TorchProfiler:
    def __init__(self, out_dir: str | Path):
        self.out_dir = ensure_dir(out_dir)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self.profiler = torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        )

    def __enter__(self):
        self.profiler.__enter__()
        return self

    def step(self) -> None:
        self.profiler.step()

    def __exit__(self, exc_type, exc, tb):
        self.profiler.__exit__(exc_type, exc, tb)
        if exc_type is not None:
            return None
        trace_path = self.out_dir / "trace.json"
        csv_path = self.out_dir / "operator_summary.csv"
        self.profiler.export_chrome_trace(str(trace_path))
        summary = self.profiler.key_averages().table(sort_by="cpu_time_total", row_limit=200)
        rows = ["operator_summary"]
        rows.extend(line.replace(",", ";") for line in summary.splitlines())
        csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        self.trace_path = trace_path
        self.operator_summary_path = csv_path
