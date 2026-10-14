import numpy as np

from sdcl.evaluation.shared_residual_bias import class_means, donor_prediction


def test_query_class_is_excluded_even_with_an_extreme_error_and_multiple_owners():
    classes = np.array([1, 1, 2, 2, 2, 2, 2])
    error = np.full((7, 4), .03)
    error[:2] = -.1
    means = np.zeros((10, 4))
    prediction, available, count = donor_prediction(classes, classes, error, means)
    np.testing.assert_allclose(prediction[:2], .03)
    np.testing.assert_array_equal(count[:2], [5, 5])
    assert available[:2].all()
    assert not available[2:].any()


def test_training_class_means_and_centering_do_not_invent_an_image_bias():
    frames = [{"classes": np.array([1, 2]), "error": np.array([[.02]*4, [-.01]*4])}]
    means, count = class_means(frames)
    np.testing.assert_allclose(means[1], .02)
    np.testing.assert_allclose(means[2], -.01)
    np.testing.assert_array_equal(count[[1, 2]], [1, 1])
    donor_cls = np.full(5, 2)
    predicted, available, _ = donor_prediction(np.array([1]), donor_cls, means[donor_cls], means)
    np.testing.assert_allclose(predicted, .02)
    assert available[0]
