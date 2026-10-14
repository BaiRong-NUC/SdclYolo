import numpy as np

from sdcl.evaluation.context_intervention import anchor_choice, class_choice, crop_variants, selected_rows


def test_background_edits_preserve_object_pixels_and_crop_scale_at_image_edge():
    image = np.arange(50 * 60 * 3, dtype=np.uint8).reshape(50, 60, 3)
    variants, box, keep = crop_variants(image, np.array([1, 2, 13, 16]))
    assert all(v.shape == (192, 192, 3) for v in variants)
    np.testing.assert_allclose(box[2:] - box[:2], [12, 14])
    np.testing.assert_array_equal(variants[1][keep], variants[0][keep])
    np.testing.assert_array_equal(variants[2][keep], variants[0][keep])
    assert not np.array_equal(variants[1][~keep], variants[0][~keep])


def test_fixed_anchor_uses_geometry_and_missing_anchor_keeps_label():
    decoded = np.zeros((2, 14), np.float32)
    decoded[:, :4] = [[480, 480, 100, 100], [20, 20, 10, 10]]
    decoded[:, 4] = [.6, .9]
    assert anchor_choice(decoded, [86, 86, 106, 106]) == 0
    assert anchor_choice(decoded, [0, 0, 1, 1]) == -1
    np.testing.assert_array_equal(class_choice(np.zeros((1, 10)), np.array([3])), [3])


def test_selection_uses_only_predictions_and_caps_both_cohorts():
    baseline = np.tile([0, 0, 20, 20, .7, 0], (12, 1)).astype(np.float32)
    score = np.zeros((12, 10, 5), np.float32)
    score[:, 0, 0] = .8
    score[:, 1, 0] = np.arange(12) / 20
    rows = selected_rows(baseline, score, [640, 640])
    assert len(rows) == len(set(rows)) == 8
    assert rows[:4].tolist() == [11, 10, 9, 8]
