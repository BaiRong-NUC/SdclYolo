from pathlib import Path

import pytest
import yaml

from sdcl.config import PROJECT_ROOT, SDCLConfig, load_experiment, project_path
from sdcl.trainer import SDCLTrainer


def test_project_paths_use_repository_root(monkeypatch, tmp_path):
    expected = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)
    assert PROJECT_ROOT == expected
    assert project_path("data/visdrone/dataset.yaml") == expected / "data/visdrone/dataset.yaml"
    assert project_path("configs/experiments/baseline.yaml").is_file()


def test_main_recipes_have_same_training_arguments():
    baseline, base_settings, base_ignore = load_experiment("configs/experiments/baseline.yaml")
    method, method_settings, method_ignore = load_experiment("configs/experiments/sdcl.yaml")
    assert baseline.pop("name") != method.pop("name")
    assert baseline == method
    assert base_ignore and method_ignore
    assert not base_settings.enabled
    assert method_settings.enabled


def test_config_unknown_section(tmp_path):
    file = tmp_path / "bad.yaml"
    file.write_text(yaml.safe_dump({"train": {"data": "data/visdrone/dataset.yaml"}, "typo": {}}))
    with pytest.raises(ValueError, match="Unknown experiment sections"):
        load_experiment(file)


def test_multi_device_rejected_before_training():
    with pytest.raises(ValueError, match="single-device"):
        SDCLTrainer(overrides={"device": "0,1"}, sdcl=SDCLConfig())
