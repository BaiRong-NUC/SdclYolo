import numpy as np

from sdcl.evaluation.assignment_quality_conflict import choose_pairs, pair_descent


def fixture():
    return dict(gt=np.array([[0., 0, 10, 10]]), classes=np.array([0]),
        anchors=np.array([[5., 5], [5, 5]]),
        probabilities=np.array([[.1, .01], [.2, .01]]),
        decoded_boxes=np.array([[0., 0, 10, 10], [0, 0, 6, 10]]),
        foreground=np.array([False, True]), owner=np.array([0, 0]), target_mass=np.array([0., .7]))


def test_better_active_background_and_worse_positive_create_opposed_descent():
    f = fixture()
    ignored = np.zeros(2, bool)
    p = choose_pairs(f, ignored)
    r = pair_descent(f, p, ignored, f["foreground"], f["owner"], f["target_mass"])
    np.testing.assert_array_equal(p["best"], [0])
    np.testing.assert_array_equal(p["peak"], [1])
    assert p["eligible"][0] and r["harmful"][0]
    np.testing.assert_allclose(r["relative_descent"], [-.6])


def test_unassigned_GT_and_ignored_good_position_cannot_count_as_this_route():
    f = fixture()
    assert not choose_pairs(f, np.array([True, False]))["eligible"][0]
    f["foreground"][:] = False
    assert not choose_pairs(f, np.zeros(2, bool))["eligible"][0]


def test_optimistic_score_ties_remove_apparent_peak_gap():
    f = fixture()
    f["probabilities"][0, 0] = .2
    p = choose_pairs(f, np.zeros(2, bool))
    assert p["peak"][0] == 0 and not p["peak_gap"][0]


def test_zero_target_negative_direction_alone_is_not_harmful_opposition():
    f = fixture()
    f["target_mass"][1] = .1
    p = choose_pairs(f, np.zeros(2, bool))
    r = pair_descent(f, p, np.zeros(2, bool), f["foreground"], f["owner"], f["target_mass"])
    assert not r["harmful"][0]


def test_targets_use_native_assigned_class_and_ignore_only_nonforeground():
    f = fixture()
    f["classes"] = np.array([0, 1])
    f["gt"] = np.array([[0., 0, 10, 10], [0., 0, 10, 10]])
    f["owner"][1] = 1
    p = choose_pairs(f, np.zeros(2, bool))
    r = pair_descent(f, p, np.array([True, True]), f["foreground"], f["owner"], f["target_mass"])
    assert r["best_descent"][0] == 0.
    assert r["peak_target"][0] == 0.
    np.testing.assert_allclose(r["peak_descent"][0], -.2)
