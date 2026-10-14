import numpy as np
import pytest

from sdcl.postprocess.observed_view_policy import derange_images, observed_features


def test_observed_disagreement_is_prediction_only_and_requires_fixed_views():
    windows = [[0, 0, 100, 100], [0, 0, 60, 60], [40, 0, 100, 60],
               [0, 40, 60, 100], [40, 40, 100, 100]]
    global_prediction = np.array([[10, 10, 20, 20, .8, 0]], float)
    local = np.array([[10, 10, 20, 20, .8, 0], [42, 10, 48, 18, .7, 1]], float)
    empty = np.empty((0, 6))
    actual = observed_features(global_prediction, {1: local, 4: empty}, windows, (100, 100))
    matched = observed_features(global_prediction, {1: local[:1], 4: empty}, windows, (100, 100))
    np.testing.assert_array_equal(actual[:, :61], matched[:, :61])
    assert np.any(actual[:, 61:] != matched[:, 61:])
    with pytest.raises(ValueError, match="observed"):
        observed_features(global_prediction, {1: local, 2: empty, 4: empty}, windows, (100, 100))


def test_shuffle_keeps_candidate_pairs_and_global_input_without_fixed_images():
    values = np.arange(5*2*133).reshape(5, 2, 133)
    shuffled, partner = derange_images(values, 1041)
    assert np.all(partner != np.arange(5))
    assert set(partner) == set(range(5))
    np.testing.assert_array_equal(shuffled[:, :, :61], values[:, :, :61])
    np.testing.assert_array_equal(shuffled[:, :, 61:], values[partner, :, 61:])
