import numpy as np

from sdcl.evaluation.crop_teacher import center_indices, choose_teachers, frame_metrics, teacher_arms


def test_center_mapping_uses_padding_offsets_and_fixed_ties():
    gt = np.array([[0., 0, 16, 16], [480, 240, 496, 256]])
    np.testing.assert_array_equal(center_indices(gt, [0, 0, 960, 480]), [31*120+1, 61*120+61])
    np.testing.assert_array_equal(center_indices(np.array([[4., 4, 12, 12]]), [0, 0, 960, 960]), [121])


def test_teacher_selection_uses_full_box_not_center_or_class_and_stable_margin():
    windows = [[0, 0, 100, 100], [0, 0, 60, 60], [40, 0, 100, 60], [0, 40, 60, 100], [40, 40, 100, 100]]
    gt = np.array([[45., 10, 55, 20], [35, 10, 55, 20], [45, 45, 55, 55]])
    available, _, teachers, active = choose_teachers(gt, windows)
    assert available[0].sum() == 2 and available[1].sum() == 1 and available[2].sum() == 4
    np.testing.assert_array_equal(active, [True, False, True])
    np.testing.assert_array_equal(teachers[[0, 2]], [[1, 2], [1, 2]])


def test_disagreeing_teachers_keep_student_while_confidence_can_switch():
    sampled = np.zeros((3, 5, 10))
    sampled[:, 0, 0] = 2
    sampled[:, 1, 1] = 5
    sampled[0, 2, 2], sampled[1, 2, 1], sampled[2, 2, 1] = 5, 4, 4
    active = np.array([True, True, False])
    arms, masks = teacher_arms(sampled, np.tile([1, 2], (3, 1)), active)
    np.testing.assert_array_equal(masks["consensus"], [False, True, False])
    np.testing.assert_array_equal(masks["confidence"], [True, True, False])
    np.testing.assert_array_equal(arms["consensus"][[0, 2]], sampled[[0, 2], 0])


def test_conditional_metric_keeps_correct_and_incorrect_rows_without_quality_filter():
    sampled = np.zeros((2, 5, 10))
    sampled[:, 0, 0] = 2
    sampled[:, 1, 1], sampled[:, 2, 1] = 3, 3
    active = np.ones(2, bool)
    arms, masks = teacher_arms(sampled, np.tile([1, 2], (2, 1)), active)
    values = frame_metrics(arms, masks, active, np.array([1, 0]), np.array([100., 100.]))["tiny"]
    assert values["rows"] == 2 and values["teacher1_recovered"] == 1 and values["teacher1_lost"] == 1
    assert values["correct"]["consensus"] == 1
