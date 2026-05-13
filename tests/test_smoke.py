from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_smoke_pipeline(clean_smoke_outputs):
    result = subprocess.run(
        [sys.executable, "scripts/smoke_test.py", "--all", "--smoke"],
        text=True,
        capture_output=True,
        timeout=900,
        check=False,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert list(Path("outputs/benchmarks").glob("*.json"))
