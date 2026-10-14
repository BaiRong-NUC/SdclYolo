import numpy as np
import torch

from sdcl.evaluation.native_deficit import CATEGORIES, exclusive_categories, native_support


def test_nonargmax_true_class_and_very_weak_score_geometry_are_not_discarded():
    boxes = torch.tensor([[0., 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10]])
    probability = torch.zeros(3, 10)
    probability[0, 1], probability[0, 0] = .1, .8
    probability[1, 1] = .00001
    probability[2, 1] = .2
    gt = np.array([[0, 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10], [70, 0, 80, 10]])
    support, _, _ = native_support(boxes, probability, gt, np.ones(4, int))
    actual = exclusive_categories(support["0.5"])
    assert [CATEGORIES[k] for k in actual] == ["nonargmax_class_suppression",
        "class_evidence_almost_absent", "native_support_not_realized", "geometry_absent"]


def test_iou_thresholds_are_separate_and_low_true_confidence_has_own_category():
    boxes = torch.tensor([[0., 0, 10, 10]])
    probability = torch.zeros(1, 10)
    probability[0, 1] = .01
    gt = np.array([[0, 0, 10, 15]])
    support, max_iou, _ = native_support(boxes, probability, gt, np.array([1]))
    assert np.isclose(max_iou[0], 2/3)
    assert CATEGORIES[exclusive_categories(support["0.5"])[0]] == "low_true_class_confidence"
    assert CATEGORIES[exclusive_categories(support["0.75"])[0]] == "geometry_absent"
