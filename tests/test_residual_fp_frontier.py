import numpy as np

from sdcl.evaluation.residual_fp_frontier import transitions


def test_higher_allowed_cutoff_causes_loss_without_any_coordinate_degradation():
    gt = np.array([[0., 0, 10, 10]])
    detections = np.array([[0., 0, 10, 10, .6, 1]])
    trace = transitions(gt, np.array([1]), detections, detections.copy(), .5, .7)
    assert trace["category"][0] == 2
    assert trace["common"][0] == 0
    assert trace["new"][0] == -1


def test_same_cutoff_coordinate_failure_is_distinct_from_matching_competition():
    gt = np.array([[0., 0, 10, 10], [5, 0, 15, 10]])
    baseline = np.array([[0., 0, 10, 10, .9, 1], [5, 0, 15, 10, .8, 1]])
    changed = baseline.copy()
    changed[0, :4] = [20, 20, 30, 30]
    trace = transitions(gt, np.array([1, 1]), baseline, changed, .5, .5)
    assert trace["category"][0] == 0
    changed[0, :4] = [3, 0, 13, 10]
    changed[1, :4] = [20, 20, 30, 30]
    trace = transitions(gt, np.array([1, 1]), baseline, changed, .5, .5)
    assert trace["category"][0] == 1
