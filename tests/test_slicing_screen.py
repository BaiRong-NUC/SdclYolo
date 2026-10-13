from dataclasses import asdict
import json
from types import SimpleNamespace

import numpy as np
import pytest

from sdcl.evaluation.slicing_screen import (
    BASELINE_SHA256, DATA_YAML_SHA256, RECORDS_SHA256, SlicePolicy,
    capture_slicing, decide_slicing, map_tile_predictions, merge_predictions, tile_windows,
)


def test_fixed_tiles_cover_borders_and_overlap_without_gt():
    windows = tile_windows(1920, 1080)
    assert windows == [(0, 0, 1067, 600), (853, 0, 1920, 600),
                       (0, 480, 1067, 1080), (853, 480, 1920, 1080)]
    mask = np.zeros((1080, 1920), dtype=bool)
    for x1, y1, x2, y2 in windows:
        mask[y1:y2, x1:x2] = True
    assert mask.all()
    assert tile_windows(2, 2)[-1] == (0, 0, 2, 2)
    with pytest.raises(ValueError):
        tile_windows(1, 10)


def test_tile_coordinate_mapping_clips_crop_before_offset():
    local = np.array([[-3, 5, 100, 70, 0.9, 2]])
    mapped = map_tile_predictions(local, (40, 20, 120, 80))
    np.testing.assert_allclose(mapped, [[40, 25, 120, 80, 0.9, 2]])
    assert local[0, 0] == -3
    assert map_tile_predictions(np.empty((0, 6)), (40, 20, 120, 80)).shape == (0, 6)


def test_merge_suppresses_global_tile_duplicates_per_class():
    global_boxes = [[0, 0, 10, 10, 0.8, 0], [0, 0, 10, 10, 0.7, 1]]
    tile_boxes = [[0, 0, 10, 10, 0.9, 0], [20, 20, 25, 25, 0.6, 0]]
    merged, ignored = merge_predictions([global_boxes, tile_boxes], (100, 100), [])
    assert len(merged) == 3
    assert ignored == 0
    np.testing.assert_allclose(merged[:, 4:], [[0.9, 0], [0.7, 1], [0.6, 0]])


def test_merge_filters_ignore_invalid_boxes_and_caps_after_merge():
    boxes = [[20 * i, 0, 20 * i + 10, 10, 0.8, 0] for i in range(600)]
    boxes.extend([[np.nan, 0, 5, 5, 0.9, 0], [0, 0, 0, 10, 0.9, 0],
                  [0, 0, 10, 10, 0.9, 0.5], [0, 0, 10, 10, 2, 0]])
    merged, ignored = merge_predictions([boxes], (13000, 20), [[0, 0, 10, 10]])
    assert ignored == 1
    assert len(merged) == 500
    assert np.isfinite(merged).all()
    assert (merged[:, 0] > 0).all()
    empty, ignored = merge_predictions([[]], (100, 100), [])
    assert empty.shape == (0, 6) and ignored == 0


def decision_report():
    settings = {"policy": asdict(SlicePolicy()), "checkpoint_sha256": BASELINE_SHA256,
                "records_sha256": RECORDS_SHA256, "dataset_yaml_sha256": DATA_YAML_SHA256}
    return {
        "protocol": "diagnostic", "reference_size": 640, "images": 548,
        "inference_sizes": {"baseline": 960, "sdcl": 960},
        "captures": {"baseline": {"settings": settings, "views_per_image": 1},
                     "sdcl": {"settings": settings.copy(), "views_per_image": 5}},
        "coco_reference": {
            "baseline": {group: {"ap50_95": 0.20} for group in ("all", "tiny", "small_all")},
            "sdcl": {"all": {"ap50_95": 0.21}, "tiny": {"ap50_95": 0.22},
                     "small_all": {"ap50_95": 0.21}},
        },
        "matched_fp_budgets": [{
            "label": "fppi_10", "delta_recall_pp": {"small_all": 1.5},
            "paired_image_bootstrap_delta_recall_95ci_pp": {"small_all": [0.4, 2.5]},
        }],
    }


def test_decision_requires_all_gates_and_full_frozen_validation():
    report = decision_report()
    assert decide_slicing(report)["decision"] == "CONTINUE"
    report["matched_fp_budgets"][0]["delta_recall_pp"]["small_all"] = 0.9
    assert decide_slicing(report)["decision"] == "STOP"
    report["images"] = 2
    assert decide_slicing(report)["decision"] == "SMOKE_ONLY"
    assert decide_slicing(decision_report(), full_validation=False)["decision"] == "SMOKE_ONLY"


def test_ci_must_be_strictly_positive_and_protocol_must_match():
    report = decision_report()
    report["matched_fp_budgets"][0]["paired_image_bootstrap_delta_recall_95ci_pp"]["small_all"][0] = 0
    assert decide_slicing(report)["decision"] == "STOP"
    report["captures"]["sdcl"]["settings"] = {"policy": {"rect": True}}
    with pytest.raises(ValueError, match="identical"):
        decide_slicing(report)


def test_decision_rejects_changed_checkpoint_or_view_count():
    report = decision_report()
    report["captures"]["baseline"]["settings"]["checkpoint_sha256"] = "changed"
    report["captures"]["sdcl"]["settings"]["checkpoint_sha256"] = "changed"
    with pytest.raises(ValueError, match="checkpoint"):
        decide_slicing(report)
    report = decision_report()
    report["captures"]["sdcl"]["views_per_image"] = 4
    with pytest.raises(ValueError, match="four"):
        decide_slicing(report)


def test_capture_uses_same_square_predictor_and_maps_all_four_tiles(tmp_path, monkeypatch):
    import cv2
    import torch

    calls = []

    class PredictorModel:
        def __init__(self, checkpoint):
            self.predictor = SimpleNamespace(model=SimpleNamespace(fp16=True))

        def predict(self, source, **kwargs):
            calls.append((source.shape, kwargs))
            boxes = SimpleNamespace(xyxy=torch.tensor([[0, 0, 5, 5]], dtype=torch.float32),
                                    conf=torch.tensor([0.9]), cls=torch.tensor([0]))
            return [SimpleNamespace(boxes=boxes)]

    monkeypatch.setattr("ultralytics.YOLO", PredictorModel)
    image_path = tmp_path / "toy.jpg"
    cv2.imwrite(str(image_path), np.zeros((60, 100, 3), dtype=np.uint8))
    metadata_path = tmp_path / "metadata/val"
    metadata_path.mkdir(parents=True)
    (metadata_path / "toy.json").write_text(
        json.dumps({"ignore_boxes_xyxy": [], "valid_objects": []}), encoding="utf-8")
    output = tmp_path / "capture"
    output.mkdir()
    predictions, captures = capture_slicing(
        tmp_path / "unused.pt", tmp_path,
        [{"stem": "toy", "image": str(image_path), "size": [100, 60]}],
        output, "cpu", {},
    )
    assert len(calls) == 3 + 1 + 4
    assert calls[3][0] == (60, 100, 3)
    assert all(shape == (34, 56, 3) for shape, _ in calls[4:])
    for _, kwargs in calls:
        assert kwargs["rect"] is False
        assert kwargs["imgsz"] == 960 and kwargs["batch"] == 1
        assert kwargs["quantize"] == 16 and kwargs["max_det"] == 500
    assert len(predictions["baseline"]["toy"]) == 1
    mapped = predictions["sdcl"]["toy"][:, :4].tolist()
    assert sorted(mapped) == [[0, 0, 5, 5], [0, 26, 5, 31],
                             [44, 0, 49, 5], [44, 26, 49, 31]]
    assert captures["sdcl"]["views_per_image"] == 5
    assert (output / "predictions/sdcl/predictions.jsonl").is_file()
