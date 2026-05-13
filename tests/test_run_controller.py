from __future__ import annotations

import os
from pathlib import Path

import pytest

from scripts import run_controller as rc


def test_state_initializes_with_all_required_stages(tmp_path: Path) -> None:
    state_path = tmp_path / "REAL_RUN_STATE.json"
    state = rc.load_state(state_path)
    assert set(rc.STAGES) == set(state["stages"])
    assert state["stages"]["segresnet_train"]["status"] == "pending"


def test_status_transition_persists(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state = rc.load_state(state_path)
    rc.set_stage(state, "prepare_data", "running", command="cmd", state_path=state_path)
    rc.set_stage(state, "prepare_data", "complete", output_files=["x"], state_path=state_path)
    loaded = rc.load_state(state_path)
    assert loaded["stages"]["prepare_data"]["status"] == "complete"
    assert loaded["stages"]["prepare_data"]["output_files"] == ["x"]


def test_completed_valid_stage_is_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    run_dir = Path("outputs/runs/segresnet_baseline")
    run_dir.mkdir(parents=True)
    for name in ["checkpoint.pt", "metrics.json", "config.yaml"]:
        (run_dir / name).write_text("ok", encoding="utf-8")
    state_path = Path("state.json")
    state = rc.load_state(state_path)
    controller = rc.Controller(state=state, state_path=state_path, smoke=True, min_free_gb=0)
    assert controller.ensure_stage("segresnet_train")
    assert state["stages"]["segresnet_train"]["status"] == "skipped_valid"


def test_failed_stage_stops_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    state_path = Path("state.json")
    state = rc.load_state(state_path)
    controller = rc.Controller(state=state, state_path=state_path, smoke=True, min_free_gb=0)
    monkeypatch.setattr(controller, "command_for_stage", lambda stage: ["false"])
    assert not controller.run_stage("environment_smoke")
    assert state["stages"]["environment_smoke"]["status"] == "failed"


def test_reset_archives_old_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    path = Path("REAL_RUN_STATE.json")
    rc.save_state(rc.init_state(), path)
    assert path.exists()
    rc.load_state(path, reset=True)
    archives = list(Path("logs").glob("run_state_archive_*.json"))
    assert archives


def test_dependency_ordering_is_declared() -> None:
    assert "segresnet_train" in rc.DEPENDENCIES["baseline_benchmark"]
    assert "combined_benchmark" in rc.DEPENDENCIES["optimized_robustness"]
    assert "verify_artifacts" in rc.DEPENDENCIES["metric_health_audit"]


def test_disk_threshold_failure_marks_blocked_external(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    state_path = Path("state.json")
    state = rc.load_state(state_path)
    controller = rc.Controller(state=state, state_path=state_path, min_free_gb=10**9)
    assert not controller.check_disk_or_block("segresnet_train")
    assert state["stages"]["segresnet_train"]["status"] == "blocked_external"
    assert Path("BLOCKER_REPORT.md").exists()


def test_smoke_status_mode_does_not_require_gpu(tmp_path: Path) -> None:
    state = rc.load_state(tmp_path / "state.json")
    controller = rc.Controller(state=state, state_path=tmp_path / "state.json", smoke=True)
    assert controller.smoke is True


def test_real_mode_requires_dataset_root(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.pop("VERSE_ROOT", None)
    env.pop("CTSPINE1K_ROOT", None)
    result = rc.subprocess.run(
        [
            rc.sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts/run_controller.py"),
            "--real",
            "--validate-only",
            "--state-file",
            str(tmp_path / "state.json"),
        ],
        env=env,
        stdout=rc.subprocess.PIPE,
        stderr=rc.subprocess.PIPE,
        text=True,
    )
    assert result.returncode != 0
    assert "--real requires VERSE_ROOT or CTSPINE1K_ROOT" in result.stderr
