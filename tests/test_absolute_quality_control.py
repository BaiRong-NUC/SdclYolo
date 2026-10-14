import numpy as np
import torch

from sdcl.postprocess.absolute_quality_control import QualityControl, quality_scores


def test_zero_initial_quality_model_exactly_preserves_every_score():
    values = np.array([[0, 0, 10, 10, .25, 0], [0, 0, 20, 20, .9999999, 2]], float)
    model = QualityControl()
    residual = model(torch.randn(2, 68)).detach().numpy()
    result, correction = quality_scores(values, residual, np.ones(2, bool))
    np.testing.assert_array_equal(result, values)
    np.testing.assert_array_equal(correction, np.zeros(2))


def test_extreme_quality_estimates_are_bounded_and_inactive_rows_unchanged():
    values = np.array([[0, 0, 10, 10, .25, 0], [0, 0, 20, 20, .4, 2],
                       [0, 0, 5, 5, .01, 1]], float)
    result, correction = quality_scores(values, np.array([10000., -10000., 10000.]),
                                         np.array([True, True, False]))
    assert np.isfinite(result).all()
    assert (abs(correction) <= 1).all()
    assert result[0, 4] > values[0, 4] and result[1, 4] < values[1, 4]
    np.testing.assert_array_equal(result[2], values[2])
    np.testing.assert_array_equal(result[:, :4], values[:, :4])
    np.testing.assert_array_equal(result[:, 5:], values[:, 5:])
