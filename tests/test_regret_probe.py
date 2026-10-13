"""Matched objective algebra and frozen one-run investment gates."""

import numpy as np
import pytest
import torch

from sdcl.evaluation.regret_probe import GATES, decide_probe, summarize_mechanism
from sdcl.postprocess.localization import ActionScorer
from sdcl.training.regret_probe import ProbeRecipe, load_recipe, ranking_loss


def test_recipe_and_rank_inversion_cost():
    assert load_recipe() == ProbeRecipe()
    valid = torch.ones(1, 3, dtype=torch.bool)
    target = torch.tensor([[.9, .7, .5]])
    importance = torch.ones(1)
    good = torch.tensor([[.3, .2, .1]])
    bad = -good
    for arm in ("uniform", "regret"):
        assert ranking_loss(good, target, valid, importance, arm) < ranking_loss(bad, target, valid, importance, arm)
        assert ranking_loss(good + 100, target, valid, importance, arm).item() == pytest.approx(
            ranking_loss(good, target, valid, importance, arm).item(), abs=1e-4)


def test_regret_emphasizes_larger_same_gt_iou_misorder():
    target = torch.tensor([[.9, .89, .5]])
    predicted = torch.tensor([[.2, .21, -.1]])
    valid = torch.ones(1, 3, dtype=torch.bool)
    # The only wrong preference has tiny IoU cost, so regret weighs it less than uniform.
    assert ranking_loss(predicted, target, valid, torch.ones(1), "regret") < ranking_loss(
        predicted, target, valid, torch.ones(1), "uniform")


@pytest.mark.parametrize("arm", ["uniform", "regret"])
def test_invalid_actions_do_not_influence_loss_and_backward_finite(arm):
    torch.manual_seed(0)
    model = ActionScorer()
    x = torch.randn(4, 6, 68)
    target = torch.tensor([[.8, .7, .75, .8, .8, .9], [0., 0., 0., 0., 0., 0.]] * 2)
    valid = torch.tensor([[True, True, False, False, False, True]] * 4)
    predicted = model(x)
    w = torch.tensor([1., 3., 1., 3.])
    loss = ranking_loss(predicted, target, valid, w, arm)
    other = predicted.detach().clone()
    other[~valid] += 1000
    assert loss.item() == ranking_loss(other, target, valid, w, arm).item()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


def test_fp_anchors_learn_keep_without_fake_iou_preference():
    y = torch.zeros(1, 6)
    valid = torch.ones(1, 6, dtype=torch.bool)
    keep = torch.tensor([[.5, 0, 0, 0, 0, 0]])
    wrong = -keep
    for arm in ("uniform", "regret"):
        assert ranking_loss(keep, y, valid, torch.ones(1), arm) < ranking_loss(wrong, y, valid, torch.ones(1), arm)


def test_no_peer_or_all_tied_owned_targets_have_zero_preference_loss():
    x = torch.randn(2, 6, requires_grad=True)
    valid = torch.tensor([[True, False, False, False, False, False]] * 2)
    y = torch.tensor([[.8] * 6, [0.] * 6])
    for arm in ("uniform", "regret"):
        loss = ranking_loss(x, y, valid, torch.ones(2), arm)
        assert loss.item() == 0
        loss.backward()
    assert torch.isfinite(x.grad).all()


def test_paired_owned_mechanism():
    row = dict(count=2, regret_iou_sum=1.6, uniform_iou_sum=1.4, calibration_iou_sum=1.3, old_gain_iou_sum=1.35)
    summary = summarize_mechanism([row])
    assert summary["regret_vs_uniform_pp"] == pytest.approx(10)
    assert summary["paired_image_95ci_pp"] == pytest.approx([10, 10])


def test_all_ten_checks_required_and_smoke_never_formal():
    values = {name: threshold + (.01 if op == ">" else 0) for name, (op, threshold) in GATES.items()}
    assert len(values) == 10
    assert decide_probe(values, True)["decision"] == "CONTINUE"
    assert decide_probe(values, False)["decision"] == "SMOKE_ONLY"
    for name, (op, threshold) in GATES.items():
        failed = values.copy()
        failed[name] = threshold + (.01 if op == "<=" else -.01)
        assert decide_probe(failed, True)["decision"] == "STOP"
    with pytest.raises(ValueError):
        decide_probe({**values, "tiny_ap_vs_calibration_pp": np.nan}, True)
