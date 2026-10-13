from dataclasses import asdict
import json
from types import SimpleNamespace

import numpy as np
import pytest

from sdcl.evaluation.fusion_screen import (
    FusionPolicy, capture_fusion, decide_fusion, gaussian_soft_nms, soft_merge_predictions,
)
from tests.test_slicing_screen import decision_report


def test_gaussian_decay_reranks_duplicate_without_changing_coordinates_or_input():
    values = np.array([[0, 0, 10, 10, 0.9, 0], [0, 0, 10, 10, 0.8, 0],
                       [20, 20, 30, 30, 0.7, 0], [0, 0, 10, 10, 0.6, 1]])
    original = values.copy()
    result = gaussian_soft_nms(values)
    np.testing.assert_allclose(result[:, 4], [0.9, 0.7, 0.6, 0.8 * np.exp(-2)])
    assert result[:, 5].tolist() == [0, 0, 1, 0]
    np.testing.assert_array_equal(values, original)
    np.testing.assert_array_equal(result[1, :4], [20, 20, 30, 30])


def test_partial_iou_decay_uses_continuous_coordinates_and_current_scores():
    values = np.array([[0, 0, 10, 10, 0.9, 0], [5, 0, 15, 10, 0.8, 0]])
    result = gaussian_soft_nms(values)
    assert result[1, 4] == pytest.approx(0.8 * np.exp(-(1 / 3)**2 / 0.5))


def test_floor_empty_and_cap_after_reranking():
    assert gaussian_soft_nms([]).shape == (0, 6)
    assert len(gaussian_soft_nms([[0, 0, 10, 10, 0.9, 0],
                                [0, 0, 10, 10, 0.002, 0]])) == 1
    boxes = [[i * 20, 0, i * 20 + 10, 10, 0.8, 0] for i in range(600)]
    assert len(gaussian_soft_nms(boxes)) == 500
    reranked = gaussian_soft_nms([[0, 0, 10, 10, 0.9, 0],
                                 [0, 0, 10, 10, 0.8, 0],
                                 [20, 20, 30, 30, 0.7, 0]], max_det=2)
    np.testing.assert_allclose(reranked[:, 4], [0.9, 0.7])


@pytest.mark.parametrize("values", [
    [[0, 0, 0, 10, 0.9, 0]], [[0, 0, 10, 10, np.nan, 0]],
    [[0, 0, 10, 10, 1.1, 0]], [[0, 0, 10, 10, 0.9, 0.5]],
])
def test_invalid_inputs_rejected(values):
    with pytest.raises(ValueError):
        gaussian_soft_nms(values)


def test_ignore_filtering_precedes_decay_and_handles_empty_inputs():
    # The ignored higher-scoring box must not depress the valid box's score.
    values = [[0, 0, 10, 10, 0.9, 0], [4, 0, 14, 10, 0.8, 0]]
    result, ignored = soft_merge_predictions([values], (100, 100), [[0, 0, 6, 10]])
    assert ignored == 1 and len(result) == 1
    assert result[0, 4] == pytest.approx(0.8)
    empty, ignored = soft_merge_predictions([[]], (100, 100), [])
    assert empty.shape == (0, 6) and ignored == 0


def fusion_report():
    report = decision_report()
    report["captures"]["baseline"]["fusion"] = {"method": "hard_nms", "iou": 0.7}
    report["captures"]["sdcl"]["fusion"] = asdict(FusionPolicy())
    return report


def test_three_fixed_gates_and_smoke_status():
    report = fusion_report()
    decision = decide_fusion(report)
    assert decision["decision"] == "CONTINUE"
    assert len(decision["checks"]) == 3
    report["coco_reference"]["sdcl"]["tiny"]["ap50_95"] = 0.1
    assert decide_fusion(report)["decision"] == "CONTINUE"  # Secondary, not a new gate.
    report["matched_fp_budgets"][0]["delta_recall_pp"]["small_all"] = 0.99
    assert decide_fusion(report)["decision"] == "STOP"
    assert decide_fusion(fusion_report(), full_validation=False)["decision"] == "SMOKE_ONLY"
    report = fusion_report()
    report["images"] = 2
    assert decide_fusion(report)["decision"] == "SMOKE_ONLY"


def test_ci_overall_gate_and_policy_are_enforced():
    report = fusion_report()
    report["matched_fp_budgets"][0]["paired_image_bootstrap_delta_recall_95ci_pp"]["small_all"][0] = 0
    assert decide_fusion(report)["decision"] == "STOP"
    report = fusion_report()
    report["coco_reference"]["sdcl"]["all"]["ap50_95"] = 0.198
    assert decide_fusion(report)["decision"] == "STOP"
    report = fusion_report()
    report["captures"]["sdcl"]["fusion"]["sigma"] = 0.6
    with pytest.raises(ValueError, match="fusion"):
        decide_fusion(report)


def test_capture_shares_exact_raw_views_and_writes_all_three_arms(tmp_path, monkeypatch):
    import cv2
    import torch

    calls = []

    class Model:
        def __init__(self, checkpoint):
            self.predictor = SimpleNamespace(model=SimpleNamespace(fp16=True))

        def predict(self, source, **kwargs):
            calls.append((source.shape, kwargs))
            boxes = SimpleNamespace(
                xyxy=torch.tensor([[0, 0, 5, 5]], dtype=torch.float32),
                conf=torch.tensor([0.9]), cls=torch.tensor([0]))
            return [SimpleNamespace(boxes=boxes)]

    monkeypatch.setattr("ultralytics.YOLO", Model)
    image = tmp_path / "toy.jpg"
    cv2.imwrite(str(image), np.zeros((60, 100, 3), dtype=np.uint8))
    metadata = tmp_path / "metadata/val"
    metadata.mkdir(parents=True)
    (metadata / "toy.json").write_text(json.dumps({"ignore_boxes_xyxy": []}), encoding="utf-8")
    output = tmp_path / "capture"
    output.mkdir()
    predictions, captures = capture_fusion(
        tmp_path / "unused.pt", tmp_path,
        [{"stem": "toy", "image": str(image), "size": [100, 60]}], output, "cpu", {})
    assert len(calls) == 8  # Three warmups, one global and four tiles; no repeated arms.
    assert all(kwargs["rect"] is False and kwargs["quantize"] == 16 for _, kwargs in calls)
    assert len(predictions["baseline"]["toy"]) == 1
    assert len(predictions["hard_nms"]["toy"]) == 4
    assert len(predictions["soft_nms"]["toy"]) == 5
    assert predictions["soft_nms"]["toy"][-1, 4] == pytest.approx(0.9 * np.exp(-2))
    raw = json.loads((output / "raw_views.jsonl").read_text())
    assert len(raw["views"]) == 5
    assert raw["windows"][0] == [0, 0, 100, 60]
    assert len({capture["raw_views_sha256"] for capture in captures.values()}) == 1
    assert captures["hard_nms"]["fusion"] != captures["soft_nms"]["fusion"]
