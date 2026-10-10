from copy import deepcopy
import json

import numpy as np
from PIL import Image
import pytest
import yaml

from sdcl.config import SDCLConfig, load_experiment
from sdcl.resolution import check_training_pair, decide_resolution, run_resolution_screen
from sdcl.size_analysis import run_size_analysis


def gate_report():
    return {
        "protocol": "COCO-style diagnostics, NOT official VisDrone",
        "reference_size": 640, "inference_sizes": {"baseline": 960, "sdcl": 960},
        "captures": {
            name: {"settings": {"checkpoint_sha256": name, "imgsz": 960,
                                "records_sha256": "same", "max_det": 500}}
            for name in ("baseline", "sdcl")
        },
        "coco_reference": {
            "baseline": {group: {"ap50_95": 0.2} for group in ("tiny", "small_all", "all")},
            "sdcl": {group: {"ap50_95": 0.21} for group in ("tiny", "small_all", "all")},
        },
        "matched_fp_budgets": [{
            "label": "fppi_10", "delta_recall_pp": {"small_all": 0.6},
            "paired_image_bootstrap_delta_recall_95ci_pp": {"small_all": [0.1, 0.9]},
        }],
    }


def test_decision_requires_ap_and_equal_fp_recall_not_fixed_confidence():
    report = gate_report()
    assert decide_resolution(report)["decision"] == "CONTINUE"
    report["matched_fp_budgets"][0]["delta_recall_pp"]["small_all"] = 0.49
    report["misses"] = {"small_all": {"delta_recall_pp": 10}}
    assert decide_resolution(report)["decision"] == "STOP"
    report = gate_report()
    report["coco_reference"]["sdcl"]["tiny"]["ap50_95"] = 0.20009
    assert decide_resolution(report)["decision"] == "STOP"
    report = gate_report()
    report["matched_fp_budgets"][0]["paired_image_bootstrap_delta_recall_95ci_pp"]["small_all"][0] = 0
    assert decide_resolution(report)["decision"] == "STOP"


@pytest.mark.parametrize("mutation", ["inference", "reference", "data", "nan"])
def test_invalid_comparisons_are_not_classified_as_negative_results(mutation):
    report = gate_report()
    if mutation == "inference":
        report["inference_sizes"]["baseline"] = 640
    elif mutation == "reference":
        report["reference_size"] = 960
    elif mutation == "data":
        report["captures"]["sdcl"]["settings"]["records_sha256"] = "different"
    else:
        report["coco_reference"]["sdcl"]["tiny"]["ap50_95"] = float("nan")
    with pytest.raises(ValueError):
        decide_resolution(report)


def make_pair(tmp_path):
    train, settings, ignore = load_experiment("configs/experiments/baseline_100.yaml")
    train.update(resume=False, save_dir="old")
    manifest = {"train": train, "sdcl": settings.to_dict(),
                "ignore_regions": ignore, "dataset_hashes": {"data": "same"}}
    checkpoints = []
    for size in (640, 960):
        run = tmp_path / f"run{size}"
        (run / "weights").mkdir(parents=True)
        checkpoint = run / "weights/best.pt"
        checkpoint.write_bytes(str(size).encode())
        saved = deepcopy(manifest)
        saved["train"].update(imgsz=size, name=f"run{size}", save_dir=str(run))
        if size == 640:
            saved["sdcl"].pop("object_scope")
            saved["sdcl"].pop("small_side_threshold")
        (run / "experiment.json").write_text(json.dumps(saved), encoding="utf-8")
        (run / "results.csv").write_text("epoch\n100\n", encoding="utf-8")
        checkpoints.append(checkpoint)
    return checkpoints


def test_training_recipe_preserves_batch_and_initialization_and_accepts_legacy_manifest(tmp_path):
    original, original_settings, original_ignore = load_experiment("configs/experiments/baseline_100.yaml")
    new, new_settings, new_ignore = load_experiment("configs/experiments/baseline_960_100.yaml")
    original.pop("name")
    new.pop("name")
    assert new == {**original, "imgsz": 960}
    assert original_settings == new_settings == SDCLConfig(enabled=False)
    assert original_ignore and new_ignore
    baseline, candidate = make_pair(tmp_path)
    check_training_pair(baseline, candidate)
    path = candidate.parent.parent / "experiment.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved["train"]["batch"] = 4
    path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(ValueError, match="batch-8"):
        check_training_pair(baseline, candidate)


def tiny_dataset(tmp_path):
    dataset = tmp_path / "data"
    for directory in ("images/val", "labels/val", "metadata/val"):
        (dataset / directory).mkdir(parents=True)
    Image.new("RGB", (640, 640), "white").save(dataset / "images/val/toy.jpg")
    (dataset / "labels/val/toy.txt").write_text("", encoding="utf-8")
    objects = [
        {"object_id": i, "class_id": 0, "box_xyxy": box, "occlusion": 0, "truncation": 0}
        for i, box in enumerate(([0, 0, 12, 12], [100, 100, 124, 124]))
    ]
    (dataset / "metadata/val/toy.json").write_text(
        json.dumps({"original_size": [640, 640], "valid_objects": objects}), encoding="utf-8")
    (dataset / "dataset.yaml").write_text(yaml.safe_dump({"names": {0: "object"}}), encoding="utf-8")
    return dataset


def fake_capture(calls):
    def capture(checkpoint, data_yaml, records, output, device, imgsz, batch, cache=None):
        calls.append((checkpoint, imgsz, cache))
        detections = np.array([[0, 0, 12, 12, 0.9, 0], [100, 100, 124, 124, 0.8, 0]])
        output.mkdir(parents=True)
        manifest = {
            "settings": {"checkpoint_sha256": str(checkpoint), "imgsz": imgsz,
                         "records_sha256": "same", "batch": batch, "max_det": 500},
            "performance": {"speed_ms_per_image": {"inference": 1}},
        }
        return {"toy": detections}, manifest
    return capture


def test_resolution_screen_keeps_gt_groups_stable_and_reuses_correct_capture(tmp_path, monkeypatch):
    baseline, candidate = make_pair(tmp_path)
    dataset = tiny_dataset(tmp_path)
    calls = []
    monkeypatch.setattr("sdcl.size_analysis.capture_predictions", fake_capture(calls))
    result = run_resolution_screen(
        str(baseline), str(candidate), dataset=str(dataset), output=str(tmp_path / "screen"))
    assert result["decision"] == "STOP"
    assert result["training_increment_deltas_pp"]["tiny_ap50_95_pp"] == 0
    for directory in ("inference_only", "training_increment"):
        report = json.loads((tmp_path / "screen" / directory / "report.json").read_text())
        assert report["reference_size"] == 640
        assert report["coco_reference"]["baseline"]["tiny"]["gt"] == 1
        assert report["coco_reference"]["sdcl"]["small"]["gt"] == 1
        assert report["misses"]["tiny"]["gt"] == 1
        assert report["misses"]["small"]["gt"] == 1
    assert [size for _, size, _ in calls] == [640, 960, 960, 960]
    assert calls[2][2] == tmp_path / "screen/inference_only/predictions/sdcl"
    assert (tmp_path / "screen/decision.md").is_file()
    assert (tmp_path / "screen/plan.json").is_file()
    with pytest.raises(FileExistsError):
        run_resolution_screen(
            str(baseline), str(candidate), dataset=str(dataset), output=str(tmp_path / "screen"))


def test_legacy_analysis_uses_imgsz_for_groups_when_reference_is_unspecified(tmp_path, monkeypatch):
    dataset = tiny_dataset(tmp_path)
    monkeypatch.setattr("sdcl.size_analysis.capture_predictions", fake_capture([]))
    output = tmp_path / "legacy"
    run_size_analysis("base.pt", "candidate.pt", dataset=str(dataset), output=str(output), imgsz=960)
    report = json.loads((output / "report.json").read_text())
    assert report["reference_size"] == 960
    assert report["misses"]["tiny"]["gt"] == 0
    assert report["misses"]["small"]["gt"] == 1
    assert report["misses"]["medium"]["gt"] == 1


@pytest.mark.parametrize("kwargs", [
    {"baseline_imgsz": 961}, {"sdcl_imgsz": 0},
    {"reference_size": 0}, {"reference_size": float("nan")},
    {"prediction_cache": "paired", "candidate_prediction_cache": "single"},
])
def test_invalid_size_and_cache_inputs_leave_no_output(tmp_path, kwargs):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        run_size_analysis("base.pt", "candidate.pt", output=str(output), **kwargs)
    assert not output.exists()
