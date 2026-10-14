import numpy as np

from sdcl.evaluation.positive_rank_gradient import pair_and_targets, gap_directions


def frame():
    return dict(gt=np.array([[0., 0, 10, 10]]), classes=np.array([0]),
        decoded_boxes=np.array([[0., 0, 10, 10], [0., 0, 6, 10], [0., 0, 5, 10]]),
        probabilities=np.array([[.2, .01], [.7, .01], [.05, .01]], np.float32),
        foreground=np.array([True, True, False]), owner=np.array([0, 0, 0]),
        target_mass=np.array([.3, .8, 0], np.float32))


def test_within_positive_pair_and_histogram_transport_preserve_background():
    f = frame()
    values = pair_and_targets(f, np.zeros(3, bool))
    np.testing.assert_array_equal(values["pairs"], [[0, 1]])
    assert values["eligible"][0]
    np.testing.assert_array_equal(values["mass_transport"], f["target_mass"][[1, 0, 2]])
    np.testing.assert_allclose(values["mass_ordinary"].sum(), f["target_mass"].sum(), rtol=1e-6)
    assert values["mass_ordinary"][0] > values["mass_ordinary"][1] > 0


def test_good_background_does_not_enter_new_positive_pair_pool():
    f = frame()
    f["foreground"][0] = False
    values = pair_and_targets(f, np.zeros(3, bool))
    assert not values["eligible"][0]
    np.testing.assert_array_equal(values["pairs"], [[1, 1]])


def test_ignored_pair_and_optimistic_score_ties_are_ineligible():
    f = frame()
    assert not pair_and_targets(f, np.array([True, False, False]))["eligible"][0]
    f["probabilities"][0, 0] = .7
    values = pair_and_targets(f, np.zeros(3, bool))
    np.testing.assert_array_equal(values["pairs"], [[0, 0]])
    assert not values["eligible"][0]


def test_same_projection_shared_features_have_zero_gap_jacobian():
    features = np.ones((2, 128))
    gradients = {a: (np.ones((3, 10, 128)), np.ones((3, 10))) for a in ("original", "transport", "ordinary")}
    values, norms = gap_directions(features, np.array([0, 0]), np.array([[0, 1]]), np.array([0]), gradients)
    assert norms[0] == 0 and all(v[0] == 0 for v in values.values())


def test_cross_level_bias_difference_enters_actual_rank_direction():
    features = np.zeros((2, 128))
    weight, bias = np.zeros((3, 10, 128)), np.zeros((3, 10))
    bias[0, 0], bias[1, 0] = 1., -1.
    gradients = {a: (weight, bias) for a in ("original", "transport", "ordinary")}
    values, norms = gap_directions(features, np.array([0, 1]), np.array([[0, 1]]), np.array([0]), gradients)
    np.testing.assert_allclose(norms, [np.sqrt(2)])
    np.testing.assert_allclose(values["original"], [-1.])
