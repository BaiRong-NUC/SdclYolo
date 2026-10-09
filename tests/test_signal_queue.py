from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import train_signal_ablation


@pytest.mark.parametrize("failure_index", [None, 0, 1])
def test_signal_queue_inherits_terminal_and_stops_on_failure(tmp_path, monkeypatch, failure_index):
    monkeypatch.setattr(train_signal_ablation, "PROJECT_ROOT", tmp_path)
    commands = [("scale", ["fake-python", "scale"]), ("difficulty", ["fake-python", "difficulty"])]
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        assert kwargs == {"cwd": tmp_path}
        return SimpleNamespace(returncode=7 if len(calls) - 1 == failure_index else 0)

    monkeypatch.setattr(train_signal_ablation.subprocess, "run", fake_run)
    result = train_signal_ablation.run_queue(commands)
    expected_count = 2 if failure_index is None else failure_index + 1
    assert calls == [command for _, command in commands[:expected_count]]
    assert result == (0 if failure_index is None else 7)
    assert list(tmp_path.iterdir()) == []


def mock_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(train_signal_ablation, "PROJECT_ROOT", tmp_path)
    (tmp_path / "yolo11s.pt").touch()
    data = tmp_path / "data" / "visdrone" / "dataset.yaml"
    data.parent.mkdir(parents=True)
    data.touch()

    def fake_load(config):
        return {"project": str(tmp_path / "output" / "runs"), "name": config.stem}, None, True

    monkeypatch.setattr(train_signal_ablation, "load_experiment", fake_load)


def test_signal_queue_uses_two_new_configs_in_order(tmp_path, monkeypatch):
    mock_inputs(tmp_path, monkeypatch)
    commands = train_signal_ablation.build_commands()
    assert [Path(command[-1]).name for _, command in commands] == [
        "sdcl_scale_100.yaml", "sdcl_difficulty_100.yaml",
    ]
    assert all("--resume" not in command for _, command in commands)


def test_signal_queue_does_not_overwrite_existing_run(tmp_path, monkeypatch):
    mock_inputs(tmp_path, monkeypatch)
    output = tmp_path / "output" / "runs" / "sdcl_scale_100"
    output.mkdir(parents=True)
    file = output / "results.csv"
    file.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Output already exists"):
        train_signal_ablation.build_commands()
    assert file.read_text(encoding="utf-8") == "keep"


def test_signal_queue_dry_run_does_not_start_or_write(tmp_path, monkeypatch):
    mock_inputs(tmp_path, monkeypatch)
    before = set(tmp_path.rglob("*"))
    monkeypatch.setattr(train_signal_ablation, "run_queue",
                        lambda commands: pytest.fail("Dry run started training"))
    assert train_signal_ablation.main(["--dry-run"]) == 0
    assert set(tmp_path.rglob("*")) == before
