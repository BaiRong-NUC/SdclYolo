import numpy as np
import pytest

from sdcl.size_analysis import (
    area_group, box_iou_matrix, coco_size_metrics, match_at_threshold,
    miss_reasons, paired_misses, recall_at_fp_budgets, summarize_misses,
)


def test_half_open_area_boundaries():
    assert area_group(255.99) == "tiny"
    assert area_group(256) == "small"
    assert area_group(1024) == "medium"
    assert area_group(9216) == "large"


def test_matching_is_class_aware_one_to_one_and_score_ordered():
    boxes = [[0, 0, 10, 10], [0, 0, 10, 10]]
    detections = np.array([[0, 0, 10, 10, 0.9, 0], [0, 0, 10, 10, 0.8, 0],
                           [0, 0, 10, 10, 0.7, 1]])
    matched, used = match_at_threshold(box_iou_matrix(boxes, detections[:, :4]), [0, 1], detections)
    assert matched.tolist() == [0, 2]
    assert used.tolist() == [True, False, True]


def test_miss_causes_include_confidence_confusion_and_localization():
    detections = np.array([[0, 0, 10, 10, 0.1, 0], [20, 0, 30, 10, 0.9, 1],
                           [40, 0, 50, 10, 0.9, 0]])
    gt = [[0, 0, 10, 10], [20, 0, 30, 10], [45, 0, 55, 10], [100, 0, 110, 10]]
    ious = box_iou_matrix(gt, detections[:, :4])
    matched, _ = match_at_threshold(ious, [0] * 4, detections)
    assert miss_reasons(ious, [0] * 4, detections, matched) == [
        "below_confidence", "class_confusion", "localization_error", "no_nearby_detection"]


def make_record(objects, size=(640, 640)):
    return {"id": 1, "stem": "toy", "size": list(size), "objects": [
        {"object_id": index, "class_id": 0, "box_xyxy": box, "occlusion": 0, "truncation": 0}
        for index, box in enumerate(objects)]}


def test_coco_other_scale_true_positive_does_not_become_false_positive():
    record = make_record([[0, 0, 10, 10], [100, 100, 220, 220]])
    # Higher-scoring large TP must be ignored, rather than counted as FP, for tiny AP.
    predictions = {"toy": np.array([[100, 100, 220, 220, 0.99, 0],
                                    [0, 0, 10, 10, 0.5, 0]])}
    result = coco_size_metrics([record], predictions, {0: "object"}, 640)
    assert result["tiny"]["ap50_95"] == pytest.approx(1)
    assert result["large"]["ap50_95"] == pytest.approx(1)
    assert result["medium"]["ap50_95"] is None


def test_empty_predictions_zero_ap_and_empty_categories_are_excluded():
    result = coco_size_metrics([make_record([[0, 0, 10, 10]])],
                               {"toy": np.empty((0, 6))}, {0: "object", 1: "absent"}, 640)
    assert result["all"]["ap50_95"] == 0
    assert result["all"]["per_class_ap50_95"]["absent"] is None


def test_reference_area_is_independent_of_absolute_box_position():
    record = make_record([[0, 0, 48, 48], [1820, 833, 1868, 881]], size=(1920, 1080))
    result = coco_size_metrics([record], {"toy": np.empty((0, 6))}, {0: "object"}, 640)
    assert result["tiny"]["gt"] == 0
    assert result["small"]["gt"] == 2
    predictions = {name: {"toy": np.empty((0, 6))} for name in ("baseline", "sdcl")}
    rows, per_image, _ = paired_misses([record], predictions, {0: "object"})
    summary = summarize_misses(rows, per_image, bootstrap_samples=0)
    assert summary["small"]["gt"] == result["small"]["gt"]


def test_paired_recovery_lost_and_reference_size():
    record = make_record([[0, 0, 20, 20], [100, 100, 300, 300]], size=(1280, 1280))
    predictions = {"baseline": {"toy": np.array([[100, 100, 300, 300, 0.9, 0]])},
                   "sdcl": {"toy": np.array([[0, 0, 20, 20, 0.9, 0]])}}
    rows, per_image, _ = paired_misses([record], predictions, {0: "object"})
    assert [row["scale_group"] for row in rows] == ["tiny", "large"]
    result = summarize_misses(rows, per_image, bootstrap_samples=20)
    assert result["tiny"]["recovered"] == 1
    assert result["large"]["lost"] == 1
    assert result["all"]["delta_recall_pp"] == 0
    assert result["small_all"]["delta_recall_pp"] == 100
    assert result["tiny"]["paired_image_bootstrap_delta_recall_95ci_pp"] == [100, 100]


def test_same_fp_budget_does_not_split_confidence_ties():
    record = make_record([[0, 0, 10, 10]])
    tied = np.array([[0, 0, 10, 10, 0.9, 0], [20, 20, 30, 30, 0.9, 0],
                     [40, 40, 50, 50, 0.9, 0]])
    predictions = {"baseline": {"toy": tied}, "sdcl": {"toy": tied}}
    comparisons, _ = recall_at_fp_budgets([record], predictions, budgets=(1, 2), bootstrap_samples=10)
    assert comparisons[0]["models"]["baseline"]["actual_fp"] == 0
    assert comparisons[0]["models"]["baseline"]["recall"]["tiny"] == 0
    assert comparisons[1]["models"]["baseline"]["actual_fp"] == 2
    assert comparisons[1]["models"]["baseline"]["recall"]["tiny"] == 1
    assert comparisons[1]["delta_recall_pp"]["tiny"] == 0


def test_same_fp_budget_distinguishes_confidence_ranking():
    record = make_record([[0, 0, 10, 10]])
    predictions = {
        "baseline": {"toy": np.array([[20, 20, 30, 30, 0.9, 0], [0, 0, 10, 10, 0.7, 0]])},
        "sdcl": {"toy": np.array([[0, 0, 10, 10, 0.95, 0], [20, 20, 30, 30, 0.9, 0]])},
    }
    comparisons, curves = recall_at_fp_budgets([record], predictions, budgets=(0,), bootstrap_samples=10)
    assert comparisons[0]["delta_recall_pp"]["tiny"] == 100
    assert comparisons[0]["paired_image_bootstrap_delta_recall_95ci_pp"]["tiny"] == [100, 100]
    assert curves["baseline"][-1]["tiny_recall"] == 1
