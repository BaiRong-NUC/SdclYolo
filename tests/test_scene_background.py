import numpy as np

from sdcl.evaluation.scene_background import decide, scene_permutation, view_statistics


def test_padding_and_predicted_regions_excluded_without_labels():
    logits = np.zeros((18900, 10), np.float16)
    boxes = np.array([[0., 0, 480, 240, .25, 3]])
    whole, masked, fractions, masks, fallback = view_statistics(logits, boxes, 960, 480)
    assert whole.shape == masked.shape == (3, 10, 4)
    assert not masks[0, 0]
    assert masks[:14400, 0].sum() == 7200
    np.testing.assert_allclose(fractions, [.75, .75, 1-8/30])
    assert not fallback.any()
    np.testing.assert_array_equal(whole, masked)


def test_full_prediction_mask_has_explicit_whole_content_fallback():
    logits = np.ones((18900, 10), np.float16)
    whole, masked, fractions, _, fallback = view_statistics(logits, [[0, 0, 960, 960, .9, 1]], 960, 960)
    assert fallback.all()
    np.testing.assert_array_equal(fractions, [0, 0, 0])
    np.testing.assert_array_equal(whole, masked)


def test_scene_permutation_is_deterministic_and_has_no_fixed_points():
    stems = ["b", "d", "a", "c"]
    order = scene_permutation(stems)
    np.testing.assert_array_equal(np.sort(order), np.arange(4))
    assert np.all(order != np.arange(4))
    np.testing.assert_array_equal(order, scene_permutation(stems))


def test_masking_must_beat_whole_scene_not_just_row_control():
    metrics = dict(tiny=dict(tp=200, fp=200, images=40, NLL=dict(row=.4, whole=.3, masked=.31, shuffled=.32),
                            AUROC=dict(whole=.8, masked=.79), whole_minus_masked_NLL_bootstrap95ci=[-.02, -.001]),
                   all=dict(NLL=dict(whole=.3, masked=.31)))
    policy = dict(tiny_tp_min=100, tiny_nonduplicate_fp_min=100, tiny_images_min=20,
                  tiny_NLL_gain_vs_row_min=.002, tiny_NLL_gain_vs_whole_min=.001, tiny_NLL_gain_vs_shuffle_min=.001,
                  tiny_AUROC_gain_vs_whole_min=.003, tiny_NLL_gain_vs_whole_bootstrap_lower_min_exclusive=0.,
                  all_NLL_gain_vs_whole_min=0.)
    assert decide(metrics, policy, True)["decision"] == "STOP_THIS_SCENE_BACKGROUND_QUANTILE_LINEAR_RECIPE"
