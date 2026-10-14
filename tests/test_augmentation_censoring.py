from copy import deepcopy

import numpy as np
from ultralytics.data.augment import RandomPerspective
from ultralytics.utils.instance import Instances

from sdcl.evaluation.augmentation_censoring import TrackedPerspective, contains, geometry_checks


def test_full_visibility_does_not_count_a_clipped_sliver():
    boxes = np.array([[10., 10, 12, 15], [-1, 10, 1, 15], [10, 10, 10, 15]])
    np.testing.assert_array_equal(contains(boxes, [0, 0, 960, 960]), [True, False, False])


def test_tracking_preserves_actual_affine_label_filter_including_ignore():
    boxes = np.array([[100., 100, 101, 110], [200, 200, 220, 220], [300, 300, 320, 320]], dtype=np.float32)
    source = dict(instances=Instances(boxes, np.zeros((0, 1000, 2), np.float32),
                                    bbox_format="xyxy", normalized=False),
                  cls=np.array([[2.], [3.], [10.]], dtype=np.float32))
    params = dict(M=np.eye(3, dtype=np.float32), scale=1., orig_shape=(960, 960), size=(960, 960))
    plain = RandomPerspective().apply_instances(deepcopy(source), params)
    tracked = TrackedPerspective()
    observed = tracked.apply_instances(deepcopy(source), params)
    np.testing.assert_array_equal(observed["cls"], plain["cls"])
    np.testing.assert_array_equal(observed["instances"].bboxes, plain["instances"].bboxes)
    np.testing.assert_array_equal(tracked.keep, [False, True, True])
    np.testing.assert_array_equal(tracked.unclipped, boxes)


def test_rejection_at_two_pixel_boundary_is_strict():
    boxes = np.array([[0., 0, 2, 10], [0, 0, 2.01, 10]], dtype=np.float32).T
    np.testing.assert_array_equal(RandomPerspective.box_candidates(boxes, boxes), [False, True])


def test_screen_success_cannot_replace_failed_confirmation():
    tiny = dict(fully_visible_gt=200, rejected_gt=12, affected_mosaics=8, rejected_fraction=.06)
    metrics = dict(screen=dict(tiny=tiny), confirmation=dict(tiny={**tiny, "rejected_gt": 0, "rejected_fraction": 0.}))
    policy = dict(fully_visible_tiny_gt_per_half_min=100, fully_visible_discarded_tiny_per_half_min=8,
                  affected_mosaics_per_half_min=4, discarded_tiny_fraction_per_half_min=.005)
    checks = geometry_checks(metrics, policy)
    assert all(checks["screen"].values())
    assert not all(checks["confirmation"].values())
