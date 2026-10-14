import numpy as np

from sdcl.evaluation.native_peak_quality import FIELDS, frame_statistics, pool_statistics


def test_equal_logit_ties_get_optimistic_quality_and_preserve_ambiguity():
    logits = np.array([[2., -3.], [2., -3.], [1., -3.]])
    result = pool_statistics([[.6, .8, .95]], logits, [0], np.ones(3, bool))
    assert result["best_iou"][0] == .95
    assert result["peak_iou"][0] == .8
    assert result["peak_min_iou"][0] == .6
    assert result["peak_count"][0] == 2
    assert result["uniform_high_iou_fraction"][0] == 2/3


def test_pool_rejects_wrong_class_low_score_low_overlap_and_invalid_rows():
    logits = np.array([[2., 3.], [-4., -5.], [3., 2.], [2., 1.], [1., -1.]])
    result = pool_statistics([[.9, .9, .49, .9, .75]], logits, [0],
                             np.array([True, True, True, False, True]))
    assert result["count"][0] == 1 and result["peak_iou"][0] == .75
    empty = pool_statistics([[.1]*5], logits, [0], np.ones(5, bool))
    assert all(value[0] == 0 for value in empty.values())


def test_any_view_may_rescue_peak_and_group_denominator_includes_unsupported_gt():
    values = {key: np.zeros((3, 5)) for key in FIELDS}
    values["best_iou"][:2, 0] = .8
    values["peak_iou"][:2, 0] = .6
    values["best_iou"][1, 4] = .9
    values["peak_iou"][1, 4] = .8
    result = frame_statistics(np.array([100., 120., 200.]), values)["tiny"]
    assert result == dict(gt=3, high_quality_supported=2, peak_high_quality_selected=1,
                          peak_loss=1, peak_loss_images=1)
