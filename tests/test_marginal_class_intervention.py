import numpy as np

from sdcl.evaluation.size_analysis import match_at_threshold
from sdcl.postprocess.marginal_class_intervention import choose, features, marginal_targets


def test_marginal_labels_account_for_duplicate_true_class_detection():
    baseline = np.array([[0, 0, 10, 10, .9, 1], [0, 0, 10, 10, .8, 0]], np.float32)
    gt_cls = np.array([0])
    iou = np.ones((1, 2))
    valid = np.zeros((2, 10), bool)
    valid[:, :2] = True
    result = marginal_targets(iou, gt_cls, baseline, valid)
    assert result[0, 0] == 0
    assert result[1, 1] == -1
    no_duplicate = marginal_targets(iou[:, :1], gt_cls, baseline[:1], valid[:1])
    assert no_duplicate[0, 0] == 1


def test_two_class_rematching_equals_full_rematching():
    rng = np.random.default_rng(11)
    baseline = np.zeros((8, 6))
    baseline[:, 4] = np.linspace(.9, .2, 8)
    baseline[:, 5] = rng.integers(0, 3, 8)
    gt_cls, iou = rng.integers(0, 3, 5), rng.random((5, 8))
    valid = np.zeros((8, 10), bool)
    valid[:, :3] = True
    result = marginal_targets(iou, gt_cls, baseline, valid)
    before = (match_at_threshold(iou, gt_cls, baseline, .25)[0] >= 0).sum()
    for row in range(8):
        for cls in range(3):
            after = baseline.copy()
            after[row, 5] = cls
            expected = (match_at_threshold(iou, gt_cls, after, .25)[0] >= 0).sum() - before
            assert result[row, cls] == expected


def test_competition_features_and_negative_gain_keep_original():
    baseline = np.array([[0, 0, 10, 10, .9, 1], [0, 0, 10, 10, .8, 0]], np.float32)
    x = features(baseline, np.zeros((2, 10, 5)), np.zeros((2, 10, 5)), [100, 100])
    assert x.shape == (2, 10, 61)
    assert x[1, 1, -4] > 0
    assert x[0, 0, -4] == 0
    valid = np.zeros((2, 10), bool)
    valid[:, :2] = True
    outputs = np.full((2, 10), -1.)
    outputs[0, 1] = 0
    outputs[1, 0] = 0
    np.testing.assert_array_equal(choose(outputs, baseline, valid, "marginal_local"), [1, 0])
