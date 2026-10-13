import numpy as np
import pytest
import torch

from sdcl.evaluation.official_visdrone import METRICS
from sdcl.evaluation.prior_gap_confirmation import decide
from sdcl.training.prior_gap_confirmation import prior_gap_loss, recipe
from sdcl.training.regret_probe import ranking_loss


def test_positive_loss_matches_existing_exponential_pair_shape():
    output = torch.tensor([[.2, .1]])
    target = torch.tensor([[.8, .6]])
    valid = torch.ones(1, 2, dtype=torch.bool)
    expected = torch.exp(-(target[0, 0] - target[0, 1]) * (output[0, 0] - output[0, 1]))
    assert prior_gap_loss(output, target, valid, torch.ones(1)) == expected
    assert prior_gap_loss(-output, target, valid, torch.ones(1)) > expected
    assert recipe()["alpha"] == 1


def test_invalid_exponential_pair_does_not_overflow_or_change_gradient():
    output = torch.tensor([[.2, .1, -1e30]], requires_grad=True)
    target = torch.tensor([[.8, .6, .9]])
    valid = torch.tensor([[True, True, False]])
    loss = prior_gap_loss(output, target, valid, torch.ones(1))
    reference = prior_gap_loss(output[:, :2], target[:, :2], valid[:, :2], torch.ones(1))
    assert loss == reference
    loss.backward()
    assert torch.isfinite(output.grad).all()
    assert output.grad[0, 2] == 0


def test_fp_keep_penalty_and_no_pair_behavior_are_preserved():
    output = torch.tensor([[.5, .1, .2]], requires_grad=True)
    target = torch.zeros(1, 3)
    valid = torch.ones(1, 3, dtype=torch.bool)
    weights = torch.ones(1)
    assert prior_gap_loss(output, target, valid, weights) == ranking_loss(
        output, target, valid, weights, "uniform")
    valid[:, 1:] = False
    assert prior_gap_loss(output, target + .8, valid, weights) == 0


def test_three_seeds_and_matched_advantage_required_for_ordinary_control():
    controls = {str(s): {a: {k: 30. for k in METRICS}
                        for a in ("calibration", "absolute", "uniform", "regret")}
                for s in range(3)}
    prior = {str(s): {k: 30. for k in METRICS} for s in range(3)}
    for values in prior.values():
        values["AP"] += .2
    assert decide(prior, controls)["decision"] == "POSITIVE_ORDINARY_PRIOR_WORK_CONTROL"
    prior["1"]["AP"] = prior["2"]["AP"] = 29.99
    assert decide(prior, controls)["decision"] == "NO_REPEATABLE_ADVANTAGE_OVER_UNIFORM"
    prior["0"]["AP"] = np.nan
    with pytest.raises(ValueError):
        decide(prior, controls)
