import numpy as np

from sdcl.evaluation.classifier_nullspace import fit, logistic_value, projectors, shuffle_null


def test_null_projection_preserves_logits_and_completes_feature_reconstruction():
    rng = np.random.default_rng(0)
    weight, features = rng.normal(size=(10, 128)), rng.normal(size=(20, 128))
    row, null, rank, error = projectors(weight)
    assert rank == 10 and error < 1e-12
    np.testing.assert_allclose(features@null@weight.T, 0, atol=1e-12)
    np.testing.assert_allclose(features@row+features@null, features, atol=1e-12)
    np.testing.assert_allclose(row@null, 0, atol=1e-12)


def test_shuffle_preserves_class_source_level_and_inactive_rows():
    values = np.arange(9*128).reshape(9, 128)
    classes = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1])
    chosen = np.array([[0, 10], [0, 11], [0, 14400], [1, 10], [1, 11],
                       [0, 10], [0, 11], [0, 18000], [-1, -1]])
    active = np.arange(9) < 8
    shuffled, permutation = shuffle_null(values, classes, chosen, active, "fixed")
    np.testing.assert_array_equal(shuffled, values[permutation])
    assert permutation[-1] == 8
    np.testing.assert_array_equal(classes[permutation], classes)
    np.testing.assert_array_equal(chosen[permutation, 0], chosen[:, 0])
    assert permutation[2] == 2 and permutation[7] == 7


def test_logistic_gradient_matches_finite_difference_including_intercept():
    rng = np.random.default_rng(2)
    inputs, labels, offset = rng.normal(size=(15, 4)), rng.integers(0, 2, 15), rng.normal(size=15)
    theta = rng.normal(size=5)
    _, gradient = logistic_value(theta, inputs, offset, labels)
    numeric = []
    for i in range(5):
        direction = np.zeros(5)
        direction[i] = 1e-5
        numeric.append((logistic_value(theta+direction, inputs, offset, labels)[0]-
                        logistic_value(theta-direction, inputs, offset, labels)[0])/2e-5)
    np.testing.assert_allclose(gradient, numeric, atol=1e-9)


def test_convex_information_fit_converges_without_evaluation_selection():
    rng = np.random.default_rng(3)
    inputs = rng.normal(size=(150, 4))
    labels = (inputs[:, 0] > .2).astype(float)
    coefficients, report = fit(inputs, np.zeros(150), labels)
    assert report["success"] and report["gradient_inf"] < 1e-5
    assert coefficients[0] > 0
