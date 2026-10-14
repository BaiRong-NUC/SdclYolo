import numpy as np

from sdcl.postprocess.correlated_error_fusion import fuse, simplex_minimum


def test_dependence_can_remove_a_redundant_view_and_beat_diagonal_weighting():
    matrix = np.array([[1., .95, 0], [.95, 1.1, 0], [0, 0, 1.]])
    full = simplex_minimum(matrix)
    diagonal = simplex_minimum(np.diag(np.diag(matrix)))
    assert full @ matrix @ full < diagonal @ matrix @ diagonal
    assert np.all(full >= 0)
    np.testing.assert_allclose(full.sum(), 1)
    assert full[2] > diagonal[2]


def test_same_weights_for_four_coordinates_preserve_validity_and_other_columns():
    candidates = np.array([[0, 0, 10, 10, .8, 2], [1, 2, 11, 12, .01, 3]], float)
    native = np.zeros((2, 5, 4))
    native[0, 0] = [0, 0, 10, 10]
    native[0, 1] = [1, 2, 11, 12]
    available = np.array([[True, True, False, False, False], [False]*5])
    table = np.zeros((3, 32, 5))
    table[0, 3, :2] = [.25, .75]
    actual = fuse(candidates, native, available, np.array([0, 0]), table, candidates)
    np.testing.assert_allclose(actual[0, :4], [.75, 1.5, 10.75, 11.5])
    np.testing.assert_array_equal(actual[:, 4:], candidates[:, 4:])
    np.testing.assert_array_equal(actual[1], candidates[1])
