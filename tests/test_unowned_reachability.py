from itertools import product

import numpy as np

from sdcl.evaluation.size_analysis import box_iou_matrix
from sdcl.evaluation.unowned_reachability import cardinality, coordinate_bounds, projected_iou, score_frame


def test_projection_is_maximal_over_all_interval_endpoints_and_GT_edges():
    generator = np.random.default_rng(0)
    for _ in range(30):
        low = generator.uniform(1, 50, 2)
        high = low+generator.uniform(2, 25, 2)
        reference = np.r_[low, high, .8, 1][None].astype(np.float32)
        scale = np.tile(high-low, 2)[None]
        lower, upper = coordinate_bounds(reference, scale, (100, 100))
        gt_low = low+generator.uniform(-10, 10, 2)
        gt = np.r_[gt_low, gt_low+generator.uniform(1, 30, 2)][None]
        clipped = np.clip(gt[0], lower[0], upper[0])
        choices = [np.unique([lower[0, j], upper[0, j], clipped[j]]) for j in range(4)]
        boxes = np.array(list(product(*choices)))
        np.testing.assert_allclose(projected_iou(gt, lower, upper)[0, 0], box_iou_matrix(gt, boxes).max(), atol=1e-12)


def test_bipartite_capacity_prevents_counting_one_prediction_twice():
    assert cardinality(np.array([[True], [True], [True]])) == 1
    assert cardinality(np.array([[True, False], [True, True], [False, True]])) == 2
    assert cardinality(np.zeros((2, 0), bool)) == 0


def test_unique_pool_sees_already_detected_GT_and_cannot_reuse_a_matched_row():
    gt = np.array([[0., 0, 10, 10], [3., 0, 13, 10]])
    prediction = np.array([[0., 0, 10, 10, .9, 1], [6.5, 0, 16.5, 10, .8, 1]])
    scale = np.tile([10., 10, 10, 10], (2, 1))
    result, trace = score_frame(gt, np.array([1, 1]), prediction, prediction, scale,
                                np.ones(2, bool), np.array([0, -1]), (640, 640))
    assert result["tiny"]["thresholds"][0]["baseline_tp"] == 1
    assert result["tiny"]["thresholds"][0]["unowned"] == 1
    assert not trace["unused_0"][0]
    # Even an already matched GT remains part of the uniqueness calculation.
    assert not trace["unique_at050"][0]
    assert trace["unique_at050"][1]


def test_wrong_class_cannot_create_oracle_recovery_and_budget_is_not_cumulative():
    gt = np.array([[4., 0, 14, 10]])
    regret = np.array([[0., 0, 10, 10, .9, 2]], np.float32)
    prediction = regret.copy()
    prediction[0, [0, 2]] += 1
    scale = np.full((1, 4), 10.)
    result, trace = score_frame(gt, np.array([1]), prediction, regret, scale, np.array([True]), np.array([-1]), (640, 640))
    assert result["tiny"]["thresholds"][0]["all_unused"] == 0
    np.testing.assert_allclose(trace["upper"][0], [1, 1, 11, 11])
    np.testing.assert_array_equal(trace["unique_at050"], False)
