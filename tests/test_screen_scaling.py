import json

import numpy as np
import pytest

from sdcl.evaluation.screen_scaling import FitEnvelope


def test_near_constant_fit_cannot_amplify_new_scene_and_never_updates_state():
    fit = np.array([[1.-2e-6, 1., -2.], [1., 1., 2.]])
    transform = FitEnvelope.fit(fit)
    state = json.dumps(transform.to_dict(), sort_keys=True)
    holdout = np.array([[.97, 0., -100.], [1.01, 10., 100.]])
    np.testing.assert_array_equal(transform.transform(holdout), [[-1., 0., -1.], [1., 0., 1.]])
    assert state == json.dumps(transform.to_dict(), sort_keys=True)
    np.testing.assert_array_equal(
        transform.transform(holdout),
        FitEnvelope.from_dict(json.loads(state)).transform(holdout))
    assert transform.transform(np.empty((0, 3))).shape == (0, 3)


def test_constant_and_roundoff_only_columns_are_zero_even_when_unseen_values_arrive():
    values = np.array([[0., 1.], [0., np.nextafter(1., 2.)]])
    transform = FitEnvelope.fit(values)
    assert not transform.active.any()
    np.testing.assert_array_equal(transform.transform([[1e100, -1e100]]), [[0., 0.]])
    with pytest.raises(ValueError):
        transform.lower[0] = -1.


def test_extreme_finite_bounds_do_not_overflow_and_interior_order_is_preserved():
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        transform = FitEnvelope.fit([[-1e308, 9e307], [1e308, 1e308]])
        actual = transform.transform([[-1e308, 9e307], [0., 9.5e307], [1e308, 1e308]])
    assert np.isfinite(actual).all() and abs(actual).max() <= 1.
    assert (np.diff(actual, axis=0) > 0).all()
    np.testing.assert_allclose(actual, [[-1., -1.], [0., 0.], [1., 1.]], atol=1e-14)


@pytest.mark.parametrize("values", [[], [[np.nan]], [[np.inf]], [[-np.inf]], np.empty((0, 2)), np.empty((2, 0))])
def test_invalid_fit_population_rejected(values):
    with pytest.raises(ValueError):
        FitEnvelope.fit(values)


def test_invalid_apply_and_corrupted_serialization_rejected():
    transform = FitEnvelope.fit([[0., 0.], [1., 1.]])
    for values in ([[1.]], [[1., np.nan]], [1., 2.]):
        with pytest.raises(ValueError):
            transform.transform(values)
    for state in ({"kind": "other", "lower": [0.], "upper": [1.]},
                  {"kind": "fit_envelope_v1", "lower": [2.], "upper": [1.]},
                  {"kind": "fit_envelope_v1", "lower": [0.], "upper": [np.inf]}):
        with pytest.raises(ValueError):
            FitEnvelope.from_dict(state)
