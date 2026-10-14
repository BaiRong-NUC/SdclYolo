import numpy as np

from sdcl.evaluation.crop_censored_sides import censor_flags, fuse, mechanism_decision, one_side


def test_spatial_flags_require_exact_internal_edge_not_image_boundary_or_missing_view():
    windows = np.array([[0, 0, 100, 80], [0, 0, 60, 45], [40, 0, 100, 45],
                        [0, 35, 60, 80], [40, 35, 100, 80]])
    boxes = windows[None].astype(float)
    available = np.ones((1, 5), bool)
    flags = censor_flags(boxes, available, windows, [100, 80])
    np.testing.assert_array_equal(flags[0], [[0, 0, 0, 0], [0, 0, 1, 1], [1, 0, 0, 1],
                                          [0, 1, 1, 0], [1, 1, 0, 0]])
    boxes[0, 1, 2] -= .00001
    assert not censor_flags(boxes, available, windows, [100, 80])[0, 1, 2]
    available[0, 2] = False
    assert not censor_flags(boxes, available, windows, [100, 80])[0, 2].any()


def test_one_sided_estimator_respects_direction_and_retains_unclipped_evidence():
    assert one_side([10, 20], [False, True], 15, True) == 10
    assert one_side([30, 20], [False, True], 15, True) == 25
    assert one_side([30, 20], [False, True], 15, False) == 30
    assert one_side([10, 20], [False, True], 15, False) == 15
    assert one_side([20, 25], [True, True], 30, True) == 20
    assert one_side([20, 25], [True, True], 10, False) == 25


def test_fusion_preserves_all_scores_classes_inactive_rows_and_invalid_fallback():
    reference = np.array([[0., 0, 10, 10, .8, 1], [0, 0, 10, 10, .7, 0]])
    native = np.tile(reference[:, None, :4], (1, 5, 1))
    native[0, 0, 0], native[0, 1, 0] = 9, 2
    native[0, 0, 2], native[0, 1, 2] = 10, 3
    available = np.zeros((2, 5), bool)
    available[:, :2] = True
    flags = np.zeros_like(native, bool)
    flags[0, 1, 0], flags[0, 0, 2] = True, True
    outputs, fallback = fuse(reference, native, available, flags, np.array([True, False]))
    for output in outputs.values():
        np.testing.assert_array_equal(output[:, 4:], reference[:, 4:])
        np.testing.assert_array_equal(output[1], reference[1])
    assert fallback["drop_side"] == 1
    np.testing.assert_array_equal(outputs["drop_side"], reference)


def test_sparse_mechanism_stops_even_with_valid_inequalities():
    tiny = {"affected_owned_rows": 1, "affected_owned_images": 1,
            "censored_side_observations": 2, "inequality_valid_fraction": 1., "strict_extension_fraction": 1.}
    policy = {"tiny_affected_owned_rows_min": 20, "tiny_affected_owned_images_min": 8,
              "tiny_censored_side_observations_min": 20, "tiny_inequality_valid_fraction_min": .75,
              "tiny_strict_extension_fraction_min": .5}
    assert mechanism_decision({"tiny": tiny}, policy)["decision"].startswith("STOP")
