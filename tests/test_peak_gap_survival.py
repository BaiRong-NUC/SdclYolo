import numpy as np

from sdcl.evaluation.peak_gap_survival import frame_statistics, partition, support_scores


def test_support_requires_correct_class_quality_and_returns_score_before_thresholding():
    gt = np.array([[0., 0., 10., 10.], [20., 0., 30., 10.]])
    predictions = np.array([[0., 0., 10., 10., .9, 1], [0., 0., 10., 10., .1, 0],
                            [20., 0., 25., 10., .8, 0]])
    scores, _ = support_scores(gt, np.array([0, 0]), predictions)
    np.testing.assert_array_equal(scores, [.1, 0.])
    np.testing.assert_array_equal(support_scores(gt, np.array([0, 0]), np.empty((0, 6)))[0], [0., 0.])


def test_partition_distinguishes_existing_support_threshold_and_two_censoring_stages():
    result = partition(np.array([1, 1, 1, 1, 0], bool),
                       np.array([1, 1, 1, 0, 1], bool),
                       np.array([1, 1, 0, 0, 1], bool),
                       np.array([1, 0, 0, 0, 0], bool))
    np.testing.assert_array_equal(result, [0, 1, 2, 3, -1])


def test_primary_counts_missing_support_even_if_other_greedy_owners_exist():
    frame = dict(areas=np.array([100., 120., 200., 300.]),
                 native_high=np.ones(4, bool), peak_gap=np.ones(4, bool),
                 remaining_gap=np.array([0, 1, 1, 1], bool),
                 partition=np.array([0, 1, 3, 2]),
                 raw001=np.array([1, 1, 0, 1], bool), raw005=np.zeros(4, bool),
                 calibrated001=np.array([1, 1, 0, 0], bool),
                 calibrated025=np.array([1, 0, 0, 0], bool),
                 greedy_owner=np.array([0, -1, -1, -1]))
    tiny = frame_statistics(frame)["tiny"]
    assert tiny["gt"] == 3 and tiny["remaining_gap"] == 2
    assert tiny["partition"] == [1, 1, 0, 1] and tiny["calibrated025_greedy_matched"] == 1
