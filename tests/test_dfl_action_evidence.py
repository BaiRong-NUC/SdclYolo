import numpy as np

from sdcl.postprocess.dfl_action_evidence import action_distances, evidence


def test_dfl_shape_can_distinguish_actions_with_identical_mean_and_variance():
    # Both distributions have mean2 and variance1; their modal masses differ.
    probability = np.zeros((2, 4, 16))
    probability[0, :, [1, 3]] = .5
    probability[1, :, [0, 4]] = .125
    probability[1, :, 2] = .75
    distances = np.tile([[[1., 1., 1., 1.], [2., 2., 2., 2.]]], (2, 1, 1))
    scores = evidence(probability, distances)
    np.testing.assert_allclose(scores["mean_distance"][0], scores["mean_distance"][1])
    np.testing.assert_allclose(scores["gaussian"][0], scores["gaussian"][1])
    assert scores["distribution"][0, 0] > scores["distribution"][0, 1]
    assert scores["distribution"][1, 0] < scores["distribution"][1, 1]


def test_letterbox_and_crop_mapping_keeps_native_left_top_right_bottom_order():
    # Crop400x200 -> square960 has scale2.4 and vertical pad240.
    boxes = np.array([[[110, 60, 130, 80]]], float)
    distances = action_distances(boxes, np.array([[3., 18.]]), np.array([16.]),
                                 [100, 50, 500, 250], [960, 960])
    np.testing.assert_allclose(distances, [[[1.5, 1.5, 1.5, 1.5]]])
