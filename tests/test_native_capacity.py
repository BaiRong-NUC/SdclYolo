import numpy as np

from sdcl.evaluation.native_capacity import certified_capacity, graph_edges, map_boxes


def test_independent_support_can_exceed_simultaneous_capacity():
    result = certified_capacity(3, np.array([[0, 1], [7, 7]]))
    assert result["statistics"] == dict(gt=3, independently_supported_gt=2, matching_capacity=1,
                                        capacity_deficit=1, deficit_images=1)
    assert len(result["cover_left"])+len(result["cover_right"]) == 1


def test_augmenting_path_recovers_two_objects_and_empty_is_supported():
    result = certified_capacity(2, np.array([[0, 0, 1], [4, 8, 4]]))
    assert result["statistics"]["matching_capacity"] == 2
    assert len(set(result["matched"])) == 2
    empty = certified_capacity(0, np.empty((2, 0), int))
    assert empty["statistics"]["capacity_deficit"] == 0


def test_geometry_support_does_not_imply_true_class_support():
    gt = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], float)
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], float)
    logits = np.array([[1, -10], [-10, 1]], float)
    edges = graph_edges(gt, np.array([0, 0]), boxes, logits)
    assert edges["geometry"].shape[1] == 4
    assert edges["argmax005"].shape[1] == 2
    assert certified_capacity(2, edges["argmax005"])["statistics"]["capacity_deficit"] == 1


def test_local_fp32_clipping_precedes_offset_mapping():
    boxes = np.array([[[-1, 0, 15, 20]]], np.float32)
    mapped = map_boxes(boxes, [[100, 200, 110, 212]])
    np.testing.assert_array_equal(mapped, [[100, 200, 110, 212]])
