import copy

import numpy as np
import pytest
import torch

from sdcl.evaluation.official_visdrone import METRICS
from sdcl.evaluation.ranking_seed_confirmation import seed_decision
from sdcl.training.ranking_seed_confirmation import matched_models, objective, recipe
from sdcl.training.localization import regression_loss
from sdcl.training.regret_probe import ranking_loss


def metrics():
    results = {}
    for seed in range(3):
        results[str(seed)] = {
            a: {m: 30. for m in METRICS} for a in ("calibration", "absolute", "uniform", "regret")
        }
        results[str(seed)]["uniform"]["AP"] += .2
        results[str(seed)]["regret"]["AP"] += .22
    return results


def test_no_best_seed_selection_or_missing_controls():
    with pytest.raises(ValueError):
        seed_decision({"0": metrics()["0"]})
    damaged = metrics()
    damaged["1"]["uniform"]["AP"] = np.nan
    with pytest.raises(ValueError):
        seed_decision(damaged)


def test_mean_gate_and_paired_direction_are_both_required():
    baseline = metrics()
    assert seed_decision(baseline)["ranking_task"] == "PROMISING_HEAD_SEED_REPEAT"
    damaged = copy.deepcopy(baseline)
    for s, delta in zip(range(3), [1., -.01, -.01]):
        damaged[str(s)]["uniform"]["AP"] = 30 + delta
    assert seed_decision(damaged)["ranking_task"] == "NO_REPEATABLE_MATCHED_ADVANTAGE"


def test_small_positive_is_not_reported_as_no_effect():
    values = metrics()
    for seed in range(3):
        values[str(seed)]["uniform"]["AP"] = 30.05
    assert seed_decision(values)["ranking_task"] == "SMALL_POSITIVE_SIGNAL"


def test_weight_has_separate_matched_test():
    values = metrics()
    values["1"]["regret"]["AP"] = 30.19
    values["2"]["regret"]["AP"] = 30.19
    verdict = seed_decision(values)
    assert verdict["ranking_task"] == "PROMISING_HEAD_SEED_REPEAT"
    assert verdict["regret_weight"] == "NO_REPEATABLE_EXTRA_ADVANTAGE_WITHIN_BUDGET"


def test_matched_initialization_and_actual_seed_variation():
    assert recipe().seeds == (1, 2)
    a, b = matched_models(1, "cpu"), matched_models(2, "cpu")
    for key in a["absolute"].state_dict():
        assert torch.equal(a["absolute"].state_dict()[key], a["uniform"].state_dict()[key])
        assert torch.equal(a["absolute"].state_dict()[key], a["regret"].state_dict()[key])
    assert any(not torch.equal(a["absolute"].state_dict()[k], b["absolute"].state_dict()[k])
               for k in a["absolute"].state_dict())


def test_objectives_exactly_reuse_registered_losses():
    outputs = torch.tensor([[.2, .1, .05], [.3, .1, .2]], requires_grad=True)
    targets = torch.tensor([[.8, .6, .3], [0., 0., 0.]])
    valid = torch.tensor([[True, True, False], [True, True, True]])
    weights = torch.tensor([1., 2.])
    for arm in ("absolute", "uniform", "regret"):
        expected = (regression_loss(outputs, targets, valid, weights, arm)
                    if arm == "absolute" else ranking_loss(outputs, targets, valid, weights, arm))
        loss = objective(outputs, targets, valid, weights, arm)
        assert torch.equal(loss, expected)
        loss.backward()
        assert torch.isfinite(outputs.grad).all()
