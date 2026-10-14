import numpy as np

from sdcl.evaluation.action_order_sensitivity import aligned_ious, perturbed_boxes, summarize
from sdcl.evaluation.size_analysis import box_iou_matrix


def test_corner_probe_keeps_nominal_then_all16_corners_and_clips_edges():
    gt = np.array([[0, 1, 10, 11], [2, 2, 3, 3]])
    variants, complete = perturbed_boxes(gt, [20, 20])
    assert variants.shape == (2, 17, 4)
    np.testing.assert_array_equal(variants[:, 0], gt)
    assert np.all((variants >= 0) & (variants <= 20))
    np.testing.assert_array_equal(complete, [True, False])


def test_aligned_variant_iou_matches_direct_matrix_without_owner_rematching():
    gt = np.array([[10, 10, 20, 20], [40, 40, 55, 55]])
    variants, _ = perturbed_boxes(gt, [100, 100])
    actions = np.stack([gt, gt + .7, gt - .5], axis=1)
    measured = aligned_ious(actions, variants)
    for row in range(2):
        np.testing.assert_allclose(measured[row], box_iou_matrix(variants[row], actions[row]))


def test_stable_oracle_rejects_reversal_and_recovers_consistent_improvement():
    ious = np.full((2, 17, 6), .6)
    ious[0, :, 1] = .65
    ious[0, 2, 1] = .55
    ious[1, :, 1] = .7
    valid = np.zeros((2, 6), bool)
    valid[:, :2] = True
    metrics = summarize(ious, valid, np.zeros(2, int), np.full(2, 100), np.ones(2, bool))
    tiny = metrics["tiny"]
    assert tiny["nominal_beneficial_actions"] == 2
    assert tiny["hard_reversed_beneficial_actions"] == 1
    assert tiny["hard_reversal_percent"] == 50
    assert tiny["stable_improvement_anchors"] == 1
    assert tiny["stable_oracle_threshold_gain_pp"] > 0
