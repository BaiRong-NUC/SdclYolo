"""Mechanism, parity, geometry and decision checks for the bounded screen."""

import numpy as np
import pytest
import torch

from sdcl.evaluation.localization_screen import GATES, decide_localization, owned_diagnostic, summarize_owned
from sdcl.evaluation.size_analysis import box_iou_matrix
from sdcl.postprocess.localization import (
    ActionScorer, LocalizationRecipe, action_feature_names, action_menu, action_targets,
    apply_actions, calibrated_candidates, load_recipe, normalized_actions, sample_anchors,
    select_actions,
)
from sdcl.postprocess.rescue import (
    RescueScorer, candidate_features, normalized_features, score_correction, soft_nms_with_origins,
)
from sdcl.training.localization import fitting_statistics, regression_loss


def menu(views, candidate=None):
    candidate = np.asarray(candidate if candidate is not None else views[0], dtype=float).reshape(-1, 6)
    return action_menu(candidate, np.zeros(len(candidate), dtype=int), candidate[:, 4],
                       np.zeros((len(candidate), 54)), views, [100, 100], [])


def test_recipe_and_shapes():
    assert load_recipe() == LocalizationRecipe()
    assert len(action_feature_names()) == 68
    model = ActionScorer()
    x = torch.randn(3, 6, 68)
    assert model(x).shape == (3, 6)


def test_peer_highest_iou_then_score_then_original_order():
    anchor = [[10, 10, 20, 20, .9, 0]]
    other = [[10, 10, 20, 20, .8, 1], [11, 10, 21, 20, .99, 0],
             [10, 10, 20, 20, .4, 0], [10, 10, 20, 20, .8, 0]]
    boxes, valid, features = menu([anchor, other, [], [], []])
    assert valid[0].tolist() == [True, True, False, False, False, True]
    assert boxes[0, 1].tolist() == anchor[0][:4]
    assert features[0, 1, 59] == pytest.approx(.8)
    assert features[0, 1, 61] == 1  # View1 one-hot.


def test_vote_is_anchor_conditioned_and_coordinates_only():
    anchor = np.array([[10, 10, 20, 20, .9, 0]])
    views = [anchor, [[11, 10, 21, 20, .3, 0]], [], [], []]
    boxes, valid, _ = menu(views)
    selected = select_actions(np.zeros((1, 6)), valid, "vote")
    result = apply_actions(anchor, boxes, selected)
    assert result[0, 0] == pytest.approx(10.25, abs=1e-6)
    assert np.array_equal(result[:, 4:], anchor[:, 4:])


def test_slots_follow_other_views_for_every_anchor_origin():
    views = [np.array([[10 + i * .1, 10, 20 + i * .1, 20, .8, 0]]) for i in range(5)]
    candidates = np.concatenate(views)
    boxes, valid, features = action_menu(candidates, np.arange(5), candidates[:, 4],
                                        np.zeros((5, 54)), views, [100, 100], [])
    assert valid.all()
    for own in range(5):
        for slot, peer in enumerate([v for v in range(5) if v != own], 1):
            assert np.allclose(boxes[own, slot], views[peer][0, :4])
            assert features[own, slot, 60 + peer] == 1


@pytest.mark.parametrize("peers", [
    [], [[50, 50, 60, 60, .9, 0]], [[10, 10, 20, 20, .04, 0]],
    [[10, 10, 20, 20, .8, 1]],
])
def test_missing_or_incompatible_peers_keep(peers):
    views = [[[10, 10, 20, 20, .9, 0]], peers, [], [], []]
    boxes, valid, _ = menu(views)
    assert valid[0].tolist() == [True, False, False, False, False, False]
    for name in ("vote", "gain", "gain_always", "absolute"):
        assert select_actions(np.ones((1, 6)), valid, name).tolist() == [0]
    assert np.array_equal(boxes[0, 0], boxes[0, 5])


def test_gain_abstention_and_absolute_reference():
    valid = np.array([[True, True, True, False, False, True]])
    outputs = np.array([[.8, -.1, .01, 99., 99., -.03]])
    assert select_actions(outputs, valid, "gain").tolist() == [0]
    assert select_actions(outputs, valid, "gain_always").tolist() == [2]
    absolute = np.array([[.8, .81, .83, 99., 99., .7]])
    assert select_actions(absolute, valid, "absolute").tolist() == [2]
    # A high predicted gain for keep is ignored; keep is the zero-gain reference.
    outputs[0, 1] = .02
    assert select_actions(outputs, valid, "gain").tolist() == [1]


def test_targets_never_switch_to_adjacent_gt_and_duplicates_are_zero():
    gt1, gt2 = [10, 10, 20, 20], [14, 10, 24, 20]
    record = {"objects": [{"box_xyxy": gt1, "class_id": 0}, {"box_xyxy": gt2, "class_id": 0}]}
    candidates = np.array([gt1 + [.9, 0], gt1 + [.8, 0]], dtype=float)
    boxes = np.repeat(candidates[:, None, :4], 6, axis=1)
    boxes[:, 1] = gt2
    targets, owned = action_targets(record, candidates, boxes)
    assert owned.tolist() == [True, False]
    assert targets[0, 0] == 1
    assert targets[0, 1] == pytest.approx(box_iou_matrix([gt1], [gt2])[0, 0])
    assert targets[0, 1] < 1
    assert not targets[1].any()


def test_empty_actions_and_calibration():
    empty = np.empty((0, 6))
    boxes, valid, features = menu([empty, [], [], [], []])
    assert boxes.shape == (0, 6, 4)
    assert features.shape == (0, 6, 68)
    result = apply_actions(empty, boxes, select_actions(np.empty((0, 6)), valid, "gain"))
    assert result.shape == (0, 6)
    model = RescueScorer().eval()
    actual, *_ = calibrated_candidates([empty, [], [], [], []], [[0, 0, 100, 100]] * 5,
                                       [100, 100], [], (model, np.zeros(54, np.float32), np.ones(54, np.float32)))
    assert actual.shape == (0, 6)


def test_calibration_arithmetic_and_index_alignment_exact():
    torch.manual_seed(0)
    model = RescueScorer().eval()
    views = [[[10, 10, 20, 20, .9, 0], [40, 40, 50, 50, .4, 1]],
             [[11, 10, 21, 20, .8, 0]], [], [], []]
    windows = [[0, 0, 100, 100]] * 5
    mean, std = np.zeros(54, np.float32), np.ones(54, np.float32)
    candidates, origins, raw_scores = soft_nms_with_origins(views, [100, 100], [])
    features = candidate_features(candidates, origins, raw_scores, views, windows, [100, 100], [])
    with torch.inference_mode():
        p = model(torch.from_numpy(normalized_features(features, mean, std, "calibration"))).softmax(1).numpy()
    reference = score_correction(candidates, p, "calibration")
    actual, source, scores, base = calibrated_candidates(views, windows, [100, 100], [], (model, mean, std))
    assert np.array_equal(actual, reference)
    for index, row in enumerate(actual):
        original = np.flatnonzero((candidates[:, :4] == row[:4]).all(1))[0]
        assert source[index] == origins[original]
        assert scores[index] == raw_scores[original]
        assert np.array_equal(base[index], features[original])


def test_deterministic_stratified_sampling_and_importance_weights():
    owned = np.r_[np.zeros(150, bool), np.ones(200, bool)]
    a, weights = sample_anchors(owned, "example_001")
    b, other = sample_anchors(owned, "example_001")
    assert np.array_equal(a, b) and np.array_equal(weights, other)
    assert len(a) == 128
    assert (~owned[a]).sum() == owned[a].sum() == 64
    assert weights.sum() == pytest.approx(350)


def test_invalid_actions_cannot_change_loss_and_gradients_finite():
    model = ActionScorer()
    features = torch.randn(4, 6, 68)
    valid = torch.tensor([[True, True, False, False, False, True]] * 4)
    targets = torch.rand(4, 6)
    weights = torch.tensor([1., 2., 3., 4.])
    for variant in ("absolute", "gain"):
        model.zero_grad()
        outputs = model(features)
        loss = regression_loss(outputs, targets, valid, weights, variant)
        changed = outputs.detach().clone()
        changed[~valid] += 100
        assert loss.item() == pytest.approx(regression_loss(changed, targets, valid, weights, variant).item())
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters())


def test_stats_ignore_invalid_actions_and_normalization_bounded():
    features = np.arange(2 * 6 * 68, dtype=np.float32).reshape(2, 6, 68)
    valid = np.zeros((2, 6), bool)
    valid[:, :2] = True
    mean, std = fitting_statistics(features, valid)
    assert np.allclose(mean, features[valid].mean(0))
    assert np.allclose(std, np.maximum(features[valid].std(0), .05))
    features[~valid] = 1e10
    other = fitting_statistics(features, valid)
    assert np.array_equal(mean, other[0]) and np.array_equal(std, other[1])
    assert np.abs(normalized_actions(features, mean, std)).max() <= 6


def test_owned_diagnostic_uses_reference_identity():
    record = {"stem": "test", "size": [100, 100], "objects": [
        {"box_xyxy": [10, 10, 12, 12], "class_id": 0},
        {"box_xyxy": [11, 10, 13, 12], "class_id": 0}]}
    anchor = np.array([[10, 10, 12, 12, .9, 0]])
    moved = anchor.copy()
    moved[0, :4] = [11, 10, 13, 12]
    maps = dict.fromkeys(("calibration", "vote", "absolute", "gain_always", "gain"), moved)
    maps["calibration"] = anchor
    row = owned_diagnostic(record, maps)
    assert row["count"] == 1
    assert row["gain_iou_sum"] == pytest.approx(1 / 3)
    summary = summarize_owned([row])
    assert summary["gain_vs_absolute_pp"] == 0


def test_gate_boundaries_and_smoke_never_formal():
    values = {name: threshold + (.01 if comparison == ">" else 0)
              for name, (comparison, threshold) in GATES.items()}
    assert len(GATES) == 11
    assert decide_localization(values, True)["decision"] == "CONTINUE"
    assert decide_localization(values, False)["decision"] == "SMOKE_ONLY"
    for name, (comparison, threshold) in GATES.items():
        changed = values.copy()
        changed[name] = threshold + (.001 if comparison == "<=" else -.001) if comparison != ">" else threshold
        assert decide_localization(changed, True)["decision"] == "STOP"
    with pytest.raises(ValueError):
        decide_localization({**values, "added_cached_mean_ms": float("nan")}, True)
