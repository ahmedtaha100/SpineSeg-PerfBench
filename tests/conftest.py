from __future__ import annotations

from pathlib import Path
import shutil

import pytest


@pytest.fixture
def clean_smoke_outputs(tmp_path):
    backups: dict[str, Path] = {}
    for path in ["outputs", "artifacts", "tests/fixtures/synthetic", "RUNS.md"]:
        p = Path(path)
        if p.exists():
            backup = tmp_path / path.replace("/", "__")
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), backup)
            backups[path] = backup
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/.gitkeep").touch()
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/.gitkeep").touch()
    try:
        yield
    finally:
        for path in ["outputs", "artifacts", "tests/fixtures/synthetic", "RUNS.md"]:
            p = Path(path)
            if p.exists():
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()
            backup = backups.get(path)
            if backup is not None and backup.exists():
                p.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(backup), p)
        Path("outputs").mkdir(exist_ok=True)
        Path("outputs/.gitkeep").touch()
        Path("artifacts").mkdir(exist_ok=True)
        Path("artifacts/.gitkeep").touch()
