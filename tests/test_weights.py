import pytest
import torch

from sdcl.config import SDCLConfig
from sdcl.weights import object_weights


def inputs():
    boxes = torch.tensor([[[0, 0, 8, 8], [0, 0, 8, 8], [0, 0, 80, 80], [0, 0, 0, 0]]]).float()
    scores = torch.tensor([[[0.8, 0], [0.4, 0], [0, 0.6], [0, 0]]])
    ids = torch.tensor([[0, 0, 1, 0]])
    foreground = torch.tensor([[True, True, True, False]])
    probabilities = torch.tensor([[[0.1, 0.2], [0.3, 0.1], [0.1, 0.8], [0.2, 0.1]]], requires_grad=True)
    ious = torch.tensor([[0.2, 0.4, 0.9, 0]])
    return boxes, scores, ids, foreground, probabilities, ious, (640, 640)


@pytest.mark.parametrize("signal", ["joint", "scale", "difficulty", "additive"])
def test_bounds_mass_and_object_shared(signal):
    result = object_weights(SDCLConfig(signal=signal, warmup_epochs=0, ramp_epochs=0), 0, *inputs())
    w = result.weights
    q = inputs()[1].sum(-1)
    assert (w >= 0.5).all() and (w <= 1.5).all()
    torch.testing.assert_close((q * w).sum(), q.sum())
    assert w[0, 0] == w[0, 1]
    assert w[0, 3] == 1
    assert not w.requires_grad
    assert result.diagnostics["matched_objects"] == 2


def test_zero_strength_and_empty_foreground():
    result = object_weights(SDCLConfig(lambda_max=0), 40, *inputs())
    assert (result.weights == 1).all()
    args = list(inputs())
    args[3] = torch.zeros_like(args[3])
    args[1] = torch.zeros_like(args[1])
    result = object_weights(SDCLConfig(), 40, *args)
    assert (result.weights == 1).all()
    assert result.diagnostics["matched_objects"] == 0


def test_batch_gt_ids_do_not_collide():
    args = list(inputs())
    args[:-1] = [value.repeat(2, *([1] * (value.ndim - 1))) for value in args[:-1]]
    result = object_weights(SDCLConfig(), 40, *args)
    assert result.diagnostics["matched_objects"] == 4


def test_schedule_and_validation():
    settings = SDCLConfig()
    assert settings.strength(0) == 0
    assert settings.strength(3) == 0
    assert settings.strength(8) == 0.25
    assert settings.strength(13) == 0.5
    with pytest.raises(ValueError):
        SDCLConfig(lambda_max=1)
    with pytest.raises(ValueError):
        SDCLConfig.from_dict({"typo": 1})


def test_scale_signal_does_not_depend_on_difficulty_inputs():
    settings = SDCLConfig(signal="scale", warmup_epochs=0, ramp_epochs=0)
    original = object_weights(settings, 0, *inputs())
    changed = list(inputs())
    changed[4] = torch.ones_like(changed[4]) * 0.95
    changed[5] = torch.ones_like(changed[5]) * 0.95
    modified = object_weights(settings, 0, *changed)
    torch.testing.assert_close(original.weights, modified.weights)
    assert original.weights[0, 0] > original.weights[0, 2]


def test_difficulty_signal_does_not_depend_on_object_scale():
    settings = SDCLConfig(signal="difficulty", warmup_epochs=0, ramp_epochs=0)
    original = object_weights(settings, 0, *inputs())
    changed = list(inputs())
    changed[0] = changed[0].clone()
    changed[0][0, :2, 2:] = 160
    changed[0][0, 2, 2:] = 4
    modified = object_weights(settings, 0, *changed)
    torch.testing.assert_close(original.weights, modified.weights)

