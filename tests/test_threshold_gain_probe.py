import torch

from sdcl.training.threshold_gain_probe import ThresholdScorer, threshold_loss, threshold_targets


def test_relative_targets_keep_identity_and_zero_keep():
    ious = torch.tensor([[.6, .9, .4]])
    quality = threshold_targets(ious, "threshold_absolute")
    gain = threshold_targets(ious, "threshold_gain")
    assert torch.equal(gain, quality - quality[:, :1])
    assert not gain[:, 0].any()
    assert gain[0, 1].max() == 1
    assert gain[0, 2].min() == -1


def test_unowned_anchors_have_zero_targets_in_both_arms():
    for arm in ("threshold_absolute", "threshold_gain"):
        assert not threshold_targets(torch.zeros(2, 6), arm).any()


def test_invalid_actions_do_not_contribute_to_loss():
    ious = torch.tensor([[.8, .3]])
    valid = torch.tensor([[True, False]])
    outputs = threshold_targets(ious, "threshold_gain")
    outputs[:, 1] = 1000
    assert threshold_loss(outputs, ious, valid, torch.ones(1), "threshold_gain") == 0


def test_matched_capacity_and_output_shape():
    model = ThresholdScorer()
    assert model(torch.zeros(2, 6, 68)).shape == (2, 6, 10)
    assert sum(p.numel() for p in model.parameters()) == 9226
