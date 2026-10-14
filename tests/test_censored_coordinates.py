import numpy as np

from sdcl.postprocess.censored_coordinates import assign_anchors, exchange


def test_one_native_candidate_has_only_one_prediction_anchor():
    baseline = np.array([[0, 0, 10, 10, .8, 0], [1, 0, 11, 10, .8, 0]], float)
    candidates = np.array([[.5, 0, 10.5, 10, .2, 0]], float)
    assert assign_anchors(candidates, baseline).tolist() == [0]


def test_coordinate_exchange_preserves_identity_and_does_not_promote_scores():
    baseline = np.array([[1, 0, 11, 10, .8, 0]], float)
    dense = np.array([[1, 0, 11, 10, .8, 0], [0, 0, 10, 10, .2, 0],
                      [0, 0, 10, 10, .5, 0]], float)
    origin = np.array([0, 0, 1])
    kept, views = dense[[0, 2]], origin[[0, 2]]
    arms, inventory = exchange(baseline, dense, origin, kept, views)
    assert inventory["selected_anchor_rows"] == [0]
    np.testing.assert_array_equal(arms["contrast_coordinates"][0, :4], dense[1, :4])
    for values in arms.values():
        assert values.shape == baseline.shape
        np.testing.assert_array_equal(values[:, 4:], baseline[:, 4:])
