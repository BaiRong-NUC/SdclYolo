import numpy as np

from sdcl.evaluation.second_observation import (
    correspondences, measurements, mirror_boxes, project_boxes,
)


def test_projected_boxes_and_mirrors_preserve_coordinates():
    boxes = np.array([[10., 20., 30., 40.], [2., 4., 8., 10.]])
    h = np.array([[2, 0, 5], [0, 2, -3], [0, 0, 1.]])
    np.testing.assert_allclose(project_boxes(boxes, h), boxes*2+[5, -3, 5, -3])
    np.testing.assert_array_equal(mirror_boxes(mirror_boxes(boxes, 100), 100), boxes)
    assert project_boxes(np.zeros((0, 4)), h).shape == (0, 4)


def test_correspondence_rejects_ambiguous_identical_objects_and_wrong_class():
    meta = dict(original_size=[960, 960], valid_objects=[
        dict(box_xyxy=[10, 10, 20, 20], class_id=0),
        dict(box_xyxy=[50, 50, 60, 60], class_id=1)])
    matches, _ = correspondences(meta, meta, np.eye(3))
    np.testing.assert_array_equal(matches, [[0, 0], [1, 1]])
    source = dict(meta, valid_objects=[meta["valid_objects"][0], meta["valid_objects"][0],
                                       dict(box_xyxy=[50, 50, 60, 60], class_id=2)])
    matches, _ = correspondences(meta, source, np.eye(3))
    assert len(matches) == 0


def test_confidence_selector_does_not_use_gt_iou_or_correct_class():
    gt = np.array([[0., 0., 10., 10.]])
    boxes = np.tile(gt[:, None], (1, 3, 1))
    logits = np.full((1, 3, 10), -10.)
    logits[0, 0, 0] = 1.
    logits[0, 1, 0] = .5
    logits[0, 2, 1] = 3.
    measured = measurements(boxes, logits, gt, np.array([0]))
    assert measured["changed"].tolist() == [[False, False, True]]
    assert measured["Q"][0, 2] == 0
    assert measured["Q"][0, 0] == measured["Q"][0, 1]


def test_empty_alignment_population_is_not_a_positive_signal():
    measured = measurements(np.zeros((0, 3, 4)), np.zeros((0, 3, 10)),
                            np.zeros((0, 4)), np.zeros(0, int))
    assert all(value.shape == (0, 3) for value in measured.values())
