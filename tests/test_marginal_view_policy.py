import numpy as np

from sdcl.postprocess.view_value import global_window_features
from sdcl.training.marginal_view_policy import labels_and_losses


def test_standalone_coverage_and_marginal_value_differ_for_redundant_crop():
    gt = np.array([[0, 0, 10, 10], [20, 0, 30, 10]], float)
    pool = np.array([[0, 0, 10, 10, .9, 0], [0, 0, 10, 10, .9, 0],
                     [20, 0, 30, 10, .9, 0]], float)
    target, loss = labels_and_losses(gt, np.array([0, 0]), np.array([100, 100]),
                                     pool, np.array([0, 1, 2]))
    assert target[0].tolist() == [2, 0]
    assert target[1].tolist() == [2, 2]
    assert loss[0]["0.75"]["lost_supported_gt"]["tiny"] == 0
    assert loss[1]["0.75"]["lost_supported_gt"]["tiny"] == 1


def test_policy_features_need_only_global_predictions_and_support_empty_global():
    windows = [[0, 0, 100, 100], [0, 0, 60, 60], [40, 0, 100, 60],
               [0, 40, 60, 100], [40, 40, 100, 100]]
    empty = global_window_features(np.empty((0, 6)), windows, [100, 100])
    ordinary = global_window_features(np.array([[0, 0, 10, 10, .8, 0]]), windows, [100, 100])
    assert empty.shape == ordinary.shape
    assert empty.shape[0] == 4
    assert np.isfinite(empty).all()
    assert not np.array_equal(empty, ordinary)
