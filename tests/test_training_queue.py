from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import train_multiseed


@pytest.mark.parametrize("failure_index", [None, 0, 1])
def test_queue_runs_in_order_and_stops_on_first_failure(tmp_path, monkeypatch, failure_index):
    monkeypatch.setattr(train_multiseed, "PROJECT_ROOT", tmp_path)
    stages = [
        train_multiseed.Stage(f"stage{i}", ["fake-python", f"config{i}"],
                             tmp_path / "output" / "runs" / f"stage{i}")
        for i in range(4)
    ]
    called = []

    def fake_run(command):
        called.append(command)
        return 7 if len(called) - 1 == failure_index else 0

    monkeypatch.setattr(train_multiseed, "run_command", fake_run)
    result = train_multiseed.run_queue(stages)
    expected_count = 4 if failure_index is None else failure_index + 1
    assert called == [stage.command for stage in stages[:expected_count]]
    assert result == (0 if failure_index is None else 7)
    assert not (tmp_path / "output").exists()


def test_queue_order_matches_seed_pairs(monkeypatch):
    monkeypatch.setattr(train_multiseed, "check_run_dir", lambda path: None)
    monkeypatch.setattr(Path, "is_file", lambda path: True)
    stages = train_multiseed.build_stages()
    assert [stage.name for stage in stages] == [
        "baseline_seed1", "sdcl_seed1", "baseline_seed2", "sdcl_seed2",
    ]
    assert [Path(stage.command[-1]).name for stage in stages] == [
        "baseline_100_seed1.yaml", "sdcl_100_seed1.yaml",
        "baseline_100_seed2.yaml", "sdcl_100_seed2.yaml",
    ]


def test_empty_interrupted_run_is_removed_without_backup(tmp_path, monkeypatch):
    monkeypatch.setattr(train_multiseed, "PROJECT_ROOT", tmp_path)
    run_dir = tmp_path / "output" / "runs" / "baseline_yolo11s_100ep_seed1"
    (run_dir / "weights").mkdir(parents=True)
    for name in ("args.yaml", "experiment.json"):
        (run_dir / name).write_text("metadata", encoding="utf-8")
    train_multiseed.clear_unstarted_run(run_dir)
    assert not run_dir.exists()
    assert list(run_dir.parent.iterdir()) == []


@pytest.mark.parametrize("saved_file", ["weights/last.pt", "results.csv", "notes.txt"])
def test_saved_results_and_unrecognized_files_are_preserved(tmp_path, monkeypatch, saved_file):
    monkeypatch.setattr(train_multiseed, "PROJECT_ROOT", tmp_path)
    run_dir = tmp_path / "output" / "runs" / "baseline_yolo11s_100ep_seed1"
    file = run_dir / saved_file
    file.parent.mkdir(parents=True)
    file.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Stopped to preserve"):
        train_multiseed.clear_unstarted_run(run_dir)
    assert file.read_text(encoding="utf-8") == "keep"


def test_cleanup_refuses_path_outside_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(train_multiseed, "PROJECT_ROOT", tmp_path)
    outside = tmp_path / "data"
    outside.mkdir()
    with pytest.raises(ValueError, match="directly inside"):
        train_multiseed.clear_unstarted_run(outside)
    assert outside.exists()


def test_dry_run_does_not_remove_interrupted_setup(tmp_path, monkeypatch):
    run_dir = tmp_path / "output" / "runs" / "baseline_yolo11s_100ep_seed1"
    run_dir.mkdir(parents=True)
    file = run_dir / "args.yaml"
    file.write_text("metadata", encoding="utf-8")
    stage = train_multiseed.Stage("baseline_seed1", ["fake-python"], run_dir)
    monkeypatch.setattr(train_multiseed, "build_stages", lambda: [stage])
    assert train_multiseed.main(["--dry-run"]) == 0
    assert file.exists()


@pytest.mark.parametrize("exit_code", [0, 7])
def test_child_inherits_terminal_and_preserves_exit_code(monkeypatch, exit_code):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=exit_code)

    monkeypatch.setattr(train_multiseed.subprocess, "run", fake_run)
    command = ["fake-python", "train.py"]
    assert train_multiseed.run_command(command) == exit_code
    assert calls[0][0] == command
    assert "stdout" not in calls[0][1]
    assert "stderr" not in calls[0][1]
    assert "capture_output" not in calls[0][1]
