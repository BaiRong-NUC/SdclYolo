import numpy as np
from scipy.optimize import differential_evolution

from sdcl.evaluation.assignment_reachability import decide, enclosure_upper_bound, group_statistics


def test_enclosure_bound_is_one_inside_and_strictly_lower_outside():
    boxes = np.tile([1., 2., 5., 6.], (4, 1))
    points = np.array([[3., 4.], [1., 2.], [9., 4.], [9., 10.]])
    np.testing.assert_allclose(enclosure_upper_bound(boxes, points), [1, 1, .5, .25])


def test_nonnegative_distance_box_search_cannot_exceed_enclosure_bound():
    gt = np.array([1., 2., 5., 6.])
    for point in (np.array([3., 4.]), np.array([8., 1.]), np.array([3., 9.])):
        def objective(distances):
            box = np.r_[point-distances[:2], point+distances[2:]]
            inter = np.maximum(np.minimum(gt[2:], box[2:])-np.maximum(gt[:2], box[:2]), 0).prod()
            return -inter/(16+np.prod(box[2:]-box[:2])-inter)
        found = -differential_evolution(objective, [(0, 10)]*4, seed=0, tol=1e-8).fun
        bound = enclosure_upper_bound(gt, point)
        np.testing.assert_allclose(found, bound, atol=1e-7)


def test_statistics_keep_unassigned_gt_and_distinguish_alternative_support():
    frame = {
        "gt": np.array([[0., 0, 2, 2], [20, 20, 22, 22], [40, 40, 42, 42]]),
        "anchors": np.array([[1., 1], [3, 1], [21, 21], [30, 30]]),
        "owner": np.array([0, 0, 1, 1]), "foreground": np.array([True, True, False, True]),
        "target_mass": np.array([.5, .5, 0., .5]),
    }
    ceiling = enclosure_upper_bound(frame["gt"][frame["owner"]], frame["anchors"])
    values, assigned, grid = group_statistics(frame, ceiling)
    assert values["tiny"]["gt"] == 3 and values["tiny"]["covered_gt"] == 2
    assert values["tiny"]["low_ceiling_affected_gt"] == 2
    assert values["tiny"]["affected_with_inside_assigned_alternative"] == 1
    assert values["tiny"]["affected_with_inside_grid_but_no_inside_assigned"] == 1
    assert assigned[2] == 0 and grid[2] < 1


def test_high_count_cannot_replace_insufficient_supervision_mass():
    metrics = {"tiny": dict(low_ceiling_affected_gt=100, low_ceiling_affected_images=40,
                            low_ceiling_mass_fraction=.001, weighted_unattainable_iou_gap=.0001)}
    policy = dict(tiny_low_ceiling_affected_gt_min=30, tiny_low_ceiling_affected_images_min=12,
                  tiny_low_ceiling_mass_fraction_min=.03, tiny_weighted_unattainable_iou_gap_min=.005)
    assert decide(metrics, policy)["decision"] == "STOP_THIS_NONNEGATIVE_SUPPORT_ROUTE_ON_FIXED_PROXY"
