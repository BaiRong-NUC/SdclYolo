import numpy as np
import pytest

from sdcl.postprocess.coordinate_score_debt import correct_scores, threshold_quality


def test_zero_quality_delta_is_exact_and_other_columns_never_change():
    values = np.array([[0, 0, 10, 10, .25, 0], [0, 0, 20, 20, .4, 2]], float)
    np.testing.assert_array_equal(correct_scores(values, np.zeros(2)), values)
    updated = correct_scores(values, np.array([.3, -.2]))
    assert updated[0, 4] > values[0, 4] and updated[1, 4] < values[1, 4]
    np.testing.assert_array_equal(updated[:, :4], values[:, :4])
    np.testing.assert_array_equal(updated[:, 5:], values[:, 5:])
    with pytest.raises(ValueError, match="residuals"):
        correct_scores(values, np.array([1.1, 0]))


def test_quality_uses_ten_thresholds_and_never_invents_an_unowned_label():
    actual = threshold_quality(np.array([[.5, .75, .95], [.9, .9, .9]]), np.array([True, False]))
    np.testing.assert_allclose(actual[0], [.1, .6, 1.])
    np.testing.assert_array_equal(actual[1], np.zeros(3))
