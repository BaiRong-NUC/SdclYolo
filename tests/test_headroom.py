"""No-training diagnostic identity, matching, partition and gate tests."""

import numpy as np
import pytest

from sdcl.evaluation.headroom_screen import (
    CAUSES, GATES, decide_headroom, max_pool_matches, miss_partition,
    paired_owned_summary, select_owned_oracle, uncluttered_raw_pool,
)
from sdcl.postprocess.localization import action_targets, apply_actions


def test_oracle_preserves_identity_masks_fp_and_prefers_keep_on_tie():
    gt1, gt2 = [10, 10, 20, 20], [14, 10, 24, 20]
    record = {"objects": [{"box_xyxy": gt1, "class_id": 0}, {"box_xyxy": gt2, "class_id": 0}]}
    final = np.array([gt1 + [.9, 0], gt1 + [.8, 0]])
    boxes = np.repeat(final[:, None, :4], 6, axis=1)
    boxes[:, 1] = gt2
    targets, owned = action_targets(record, final, boxes)
    valid = np.ones((2, 6), bool)
    selected = select_owned_oracle(targets, owned, valid)
    assert selected.tolist() == [0, 0]
    assert np.array_equal(apply_actions(final, boxes, selected), final)
    targets[0] = [.6, .7, .9, .6, .6, .6]
    valid[0, 2] = False
    assert select_owned_oracle(targets, owned, valid).tolist() == [1, 0]


@pytest.mark.parametrize("name,final,raw,before,threshold", [
    ("detected", [[0, 0, 10, 10, .9, 0]], [], [], .5),
    ("final_below_operating_score", [[0, 0, 10, 10, .2, 0]], [], [], .5),
    ("fusion_or_cap_removal", [], [[0, 0, 10, 10, .2, 0]], [], .5),
    ("ignore_filter_removal", [], [], [[0, 0, 10, 10, .2, 0]], .5),
    ("raw_class_mismatch", [], [[0, 0, 10, 10, .2, 1]], [], .5),
    ("raw_localization_error", [], [[4, 0, 14, 10, .2, 0]], [], .5),
    ("no_sameclass_nearby_in_cached_pool", [], [], [], .5),
])
def test_miss_partition(name, final, raw, before, threshold):
    record = {"objects": [{"box_xyxy": [0, 0, 10, 10], "class_id": 0}]}
    arrays = [np.asarray(v, dtype=float).reshape(-1, 6) for v in (final, raw, before)]
    causes, matched, _ = miss_partition(record, *arrays, threshold)
    assert causes == [name] and name in CAUSES
    assert (matched[0] >= 0) == (name == "detected")


def test_match_competition_distinct_from_independent_coverage():
    record = {"objects": [{"box_xyxy": [0, 0, 10, 10], "class_id": 0}] * 2}
    final = np.array([[0, 0, 10, 10, .9, 0]])
    causes, _, info = miss_partition(record, final, final, final, .5)
    assert causes == ["detected", "final_matching_competition"]
    assert (info["raw_same_iou"] >= .5).sum() == 2
    assert max_pool_matches(info["raw_edges"]) == 1


def test_bipartite_matching_beats_a_greedy_score_assignment():
    edges = np.array([[True, True], [True, False]])
    assert max_pool_matches(edges) == 2
    assert max_pool_matches(np.empty((0, 2), bool)) == 0
    assert max_pool_matches(np.empty((2, 0), bool)) == 0


def test_before_ignore_pool_clips_and_filters_but_retains_ignore_boxes():
    views = [[[-1, -1, 10, 10, .5, 0], [20, 20, 20, 30, .5, 0],
              [20, 20, 30, 30, .0001, 0], [20, 20, 30, 30, .5, .5],
              [20, 20, 30, 30, float("nan"), 0]], [], [], [], []]
    actual = uncluttered_raw_pool(views, [100, 100])
    assert actual.tolist() == [[0, 0, 10, 10, .5, 0]]


def test_empty_gt_and_predictions():
    empty = np.empty((0, 6))
    causes, matched, info = miss_partition({"objects": []}, empty, empty, empty, .5)
    assert not causes and len(matched) == 0
    assert info["raw_edges"].shape == (0, 0)
    assert select_owned_oracle(np.empty((0, 6)), np.empty(0, bool), np.empty((0, 6), bool)).shape == (0,)


def test_owned_bootstrap_uses_paired_image_sums():
    rows = [dict(owned_small_all=2, oracle_iou_sum=1.6, calibration_iou_sum=1.4, gain_iou_sum=1.5),
            dict(owned_small_all=1, oracle_iou_sum=.8, calibration_iou_sum=.7, gain_iou_sum=.75)]
    summary = paired_owned_summary(rows)
    assert summary["oracle_vs_calibration_pp"] == pytest.approx(10)
    assert summary["oracle_vs_calibration_paired_image_95ci_pp"] == pytest.approx([10, 10])
    assert summary["learned_gain_fraction_of_owned_iou_reference"] == pytest.approx(.5)


def test_headroom_never_restarts_stopped_training_even_if_passed():
    values = {name: limit + (.01 if comparison == ">" else 0)
              for name, (comparison, limit) in GATES.items()}
    result = decide_headroom(values, True)
    assert result["decision"] == "HEADROOM_ONLY"
    assert not result["training_authorized_by_diagnostic"]
    assert decide_headroom(values, False)["decision"] == "SMOKE_ONLY"
    for name in values:
        other = values.copy()
        other[name] = GATES[name][1] - .01
        assert decide_headroom(other, True)["decision"] == "STOP_CURRENT_ACTION_SPACE"
    with pytest.raises(ValueError):
        decide_headroom({**values, "tiny_oracle_ap_vs_calibration_pp": np.nan}, True)
