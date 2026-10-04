import pytest
import yaml

from sdcl.config import SDCLConfig, load_experiment
from sdcl.trainer import SDCLTrainer


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
