import numpy as np

from sdcl.evaluation.label_gradient_sensitivity import amplitudes_and_guard, coordinate_gradient, decide, direction_flip


def test_noise_transport_and_guard_is_common_to_both_arms_without_clipping():
    gt = np.array([[100., 100, 116, 116], [0, 0, 16, 16], [200, 200, 200.5, 201]])
    absolute, relative, guard = amplitudes_and_guard(gt, [960, 960])
    np.testing.assert_allclose(absolute, .5)
    np.testing.assert_allclose(relative[0], .5)
    np.testing.assert_array_equal(guard, [True, False, False])
    absolute, _, _ = amplitudes_and_guard(gt, [1920, 1080])
    np.testing.assert_allclose(absolute, .25)


def test_pixel_coordinate_gradient_is_finite_and_respects_stride_chain():
    prediction = np.array([[0., 0, 8, 8], [5, 3, 12, 10]])
    target = np.array([[1., 1, 9, 9], [4, 2, 11, 9]])
    gradient = coordinate_gradient(prediction, target, np.array([[8.], [16.]]))
    assert np.isfinite(gradient).all()
    unit = coordinate_gradient(prediction/np.array([[8.], [16.]]), target/np.array([[8.], [16.]]), np.ones((2, 1)))
    np.testing.assert_allclose(gradient, unit/np.array([[8.], [16.]]), atol=1e-12)


def test_zero_and_perpendicular_gradients_are_not_strict_reversals():
    nominal = np.array([[1., 0, 0, 0], [0, 1, 0, 0]])
    perturbed = np.array([[[-1., 0, 0, 0], [1, 0, 0, 0]], [[0, 0, 0, 0], [0, -1, 0, 0]]])
    np.testing.assert_array_equal(direction_flip(nominal, perturbed), [[True, False], [False, True]])


def test_absolute_sensitivity_without_excess_over_relative_control_stops():
    metrics = {"tiny": dict(productive_covered_gt=300, productive_images=40, absolute_flip_rate=.2,
                            absolute_minus_relative_flip_rate=-.05, absolute_minus_relative_image_bootstrap95ci=[-.1, -.01])}
    policy = dict(tiny_productive_covered_gt_min=100, tiny_productive_images_min=24, tiny_absolute_flip_rate_min=.03,
                  tiny_absolute_minus_relative_flip_rate_min=.005, tiny_difference_bootstrap_lower_min=0.)
    assert decide(metrics, policy)["decision"] == "STOP_THIS_HALF_PIXEL_SPECIFIC_CIOU_ROUTE"
