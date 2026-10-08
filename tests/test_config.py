from pathlib import Path
from dataclasses import replace
import json

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


def test_100_epoch_pair_differs_only_in_weighting_enablement():
    baseline, base_settings, base_ignore = load_experiment("configs/experiments/baseline_100.yaml")
    method, method_settings, method_ignore = load_experiment("configs/experiments/sdcl_100.yaml")
    assert baseline.pop("name") != method.pop("name")
    assert baseline == method
    assert method_settings == replace(base_settings, enabled=True)
    assert not base_settings.enabled
    assert base_ignore and method_ignore
    assert method_settings.apply_to == "both"
    assert method_settings.classification_weighting == "positive_term"
    assert not baseline.get("resume")
    assert not baseline.get("exist_ok")


@pytest.mark.parametrize("recipe", ["baseline", "sdcl"])
def test_100_epoch_recipes_preserve_original_method_and_training_recipe(recipe):
    original, original_settings, original_ignore = load_experiment(
        f"configs/experiments/{recipe}.yaml"
    )
    longer, longer_settings, longer_ignore = load_experiment(
        f"configs/experiments/{recipe}_100.yaml"
    )
    assert original.pop("name") != longer.pop("name")
    assert longer == {**original, "epochs": 100, "batch": 8}
    assert longer["model"] == "yolo11s.pt"
    assert longer_settings == original_settings
    assert longer_ignore == original_ignore


@pytest.mark.parametrize(
    ("file", "scope"),
    [("sdcl_cls.yaml", "classification"), ("sdcl_reg.yaml", "regression")],
)
def test_branch_recipes_preserve_30_epoch_comparison(file, scope):
    method, method_settings, method_ignore = load_experiment("configs/experiments/sdcl.yaml")
    branch, branch_settings, branch_ignore = load_experiment(f"configs/experiments/{file}")
    assert branch.pop("name") != method.pop("name")
    assert branch == {**method, "epochs": 30, "batch": 8}
    assert branch_settings == replace(method_settings, apply_to=scope)
    assert branch_ignore == method_ignore


def test_scope_default_validation_and_round_trip():
    assert SDCLConfig.from_dict({"enabled": True}).apply_to == "both"
    for scope in ("both", "classification", "regression"):
        settings = SDCLConfig(apply_to=scope)
        assert SDCLConfig.from_dict(settings.to_dict()) == settings
    with pytest.raises(ValueError, match="Unknown apply_to"):
        SDCLConfig(apply_to="classfication")


def test_full_bce_recipe_changes_only_classification_weighting():
    original, original_settings, original_ignore = load_experiment("configs/experiments/sdcl_cls.yaml")
    modified, modified_settings, modified_ignore = load_experiment(
        "configs/experiments/sdcl_cls_full_bce.yaml"
    )
    assert original.pop("name") != modified.pop("name")
    assert original == modified
    assert modified_settings == replace(original_settings, classification_weighting="full_bce")
    assert original_ignore == modified_ignore


def test_classification_weighting_default_validation_and_round_trip():
    assert SDCLConfig.from_dict({"apply_to": "classification"}).classification_weighting == "positive_term"
    for mode in ("positive_term", "full_bce"):
        settings = SDCLConfig(classification_weighting=mode)
        assert SDCLConfig.from_dict(settings.to_dict()) == settings
    with pytest.raises(ValueError, match="Unknown classification_weighting"):
        SDCLConfig(classification_weighting="typo")


@pytest.mark.parametrize("apply_to", ["both", "classification", "regression"])
@pytest.mark.parametrize("mode", ["positive_term", "full_bce"])
def test_legacy_resume_manifest_defaults_to_both(tmp_path, monkeypatch, apply_to, mode):
    from sdcl.cli import train_experiment

    settings = SDCLConfig(apply_to=apply_to, classification_weighting=mode)
    config = tmp_path / "experiment.yaml"
    config.write_text(yaml.safe_dump({
        "train": {"data": "data/visdrone/dataset.yaml"},
        "sdcl": settings.to_dict(),
    }), encoding="utf-8")
    weights = tmp_path / "run" / "weights"
    weights.mkdir(parents=True)
    saved_settings = SDCLConfig().to_dict()
    saved_settings.pop("apply_to")
    saved_settings.pop("classification_weighting")
    (weights.parent / "experiment.json").write_text(json.dumps({
        "sdcl": saved_settings, "ignore_regions": True,
    }), encoding="utf-8")
    started = []

    class FakeTrainer:
        save_dir = weights.parent

        def __init__(self, **kwargs):
            started.append(kwargs)

        def train(self):
            pass

    monkeypatch.setattr("sdcl.trainer.SDCLTrainer", FakeTrainer)
    if apply_to == "both" and mode == "positive_term":
        train_experiment(config, resume=str(weights / "last.pt"))
        assert started[0]["sdcl"].apply_to == "both"
        assert started[0]["sdcl"].classification_weighting == "positive_term"
    else:
        with pytest.raises(ValueError, match="same SDCL settings"):
            train_experiment(config, resume=str(weights / "last.pt"))
        assert not started


def test_config_unknown_section(tmp_path):
    file = tmp_path / "bad.yaml"
    file.write_text(yaml.safe_dump({"train": {"data": "data/visdrone/dataset.yaml"}, "typo": {}}))
    with pytest.raises(ValueError, match="Unknown experiment sections"):
        load_experiment(file)


def test_multi_device_rejected_before_training():
    with pytest.raises(ValueError, match="single-device"):
        SDCLTrainer(overrides={"device": "0,1"}, sdcl=SDCLConfig())
