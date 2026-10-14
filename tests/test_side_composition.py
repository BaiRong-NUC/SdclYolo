import numpy as np

from sdcl.postprocess.side_composition import choose_coordinates


def test_sides_can_form_new_valid_box_while_preserving_score_and_class():
    candidates = np.array([[0, 0, 10, 10, .8, 0]], float)
    boxes = np.array([[[0, 0, 10, 10], [1, 0, 11, 10], [0, 1, 10, 11],
                       [0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 10, 10]]], float)
    utility = np.zeros((1, 6, 4))
    utility[0, 1, 0], utility[0, 2, 3] = 1, 1
    result, counts = choose_coordinates(candidates, boxes, np.ones((1, 6), bool),
                                        np.array([0]), utility, True)
    np.testing.assert_array_equal(result[0, :4], [1, 0, 10, 11])
    np.testing.assert_array_equal(result[:, 4:], candidates[:, 4:])
    assert counts["new_compositions_at_score025"] == 1


def test_crossed_side_choices_fall_back_to_uniform_whole_box():
    candidates = np.array([[0, 0, 10, 10, .8, 0]], float)
    boxes = np.tile(candidates[:, None, :4], (1, 6, 1))
    boxes[0, 1] = [9, 0, 20, 10]
    boxes[0, 2] = [-10, 0, 1, 10]
    utility = np.zeros((1, 6, 4))
    utility[0, 1, 0], utility[0, 2, 2] = 1, 1
    result, counts = choose_coordinates(candidates, boxes, np.ones((1, 6), bool),
                                        np.array([0]), utility, True)
    np.testing.assert_array_equal(result, candidates)
    assert counts["invalid_fallback"] == 1
