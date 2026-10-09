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


@pytest.mark.parametrize("object_scope", ["all", "small"])
def test_zero_strength_and_empty_foreground(object_scope):
    result = object_weights(SDCLConfig(lambda_max=0, object_scope=object_scope), 40, *inputs())
    assert (result.weights == 1).all()
    args = list(inputs())
    args[3] = torch.zeros_like(args[3])
    args[1] = torch.zeros_like(args[1])
    result = object_weights(SDCLConfig(object_scope=object_scope), 40, *args)
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


def grouped_inputs():
    boxes = torch.tensor([[[0, 0, 8, 8], [0, 0, 8, 8], [0, 0, 24, 24],
                           [0, 0, 80, 80], [0, 0, 0, 0]]]).float()
    scores = torch.tensor([[[0.8, 0], [0.4, 0], [0, 0.6], [0.9, 0], [0, 0]]])
    ids = torch.tensor([[0, 0, 1, 2, 0]])
    foreground = torch.tensor([[True, True, True, True, False]])
    probabilities = torch.tensor([[[0.1, 0.2], [0.3, 0.1], [0.1, 0.8],
                                   [0.2, 0.1], [0.2, 0.1]]], requires_grad=True)
    ious = torch.tensor([[0.2, 0.4, 0.9, 0.2, 0]])
    return boxes, scores, ids, foreground, probabilities, ious, (640, 640)


@pytest.mark.parametrize("signal", ["joint", "scale", "difficulty", "additive"])
def test_small_group_conserves_its_own_mass_and_keeps_other_weights_one(signal):
    args = grouped_inputs()
    settings = SDCLConfig(object_scope="small", signal=signal, warmup_epochs=0, ramp_epochs=0)
    result = object_weights(settings, 0, *args)
    q = args[1].sum(-1)
    w = result.weights
    assert w[0, 3] == 1 and w[0, 4] == 1
    assert w[0, 0] == w[0, 1]
    assert w[0, 0] != w[0, 2]
    assert (w >= 0.5).all() and (w <= 1.5).all()
    torch.testing.assert_close((q[:, :3] * w[:, :3]).sum(), q[:, :3].sum())
    torch.testing.assert_close((q * w).sum(), q.sum())
    assert not w.requires_grad
    assert result.diagnostics["eligible_objects"] == 2
    assert result.diagnostics["ineligible_weight_max_deviation"] == 0
    assert result.diagnostics["eligible_mass_relative_error"] < 1e-6


def test_small_group_uses_quality_mass_not_positive_location_count():
    args = grouped_inputs()
    settings = SDCLConfig(object_scope="small", warmup_epochs=0, ramp_epochs=0)
    result = object_weights(settings, 0, *args)
    signal0 = (1 / (1 + (8 / 32) ** 2)) * (0.5 * 0.4 + 0.5 * 0.7)
    signal1 = (1 / (1 + (24 / 32) ** 2)) * (0.5 * 0.2 + 0.5 * 0.1)
    center = (1.2 * signal0 + 0.6 * signal1) / 1.8
    expected = torch.tensor([[1 + 0.5 * (signal0 - center)] * 2
                             + [1 + 0.5 * (signal1 - center), 1, 1]])
    torch.testing.assert_close(result.weights, expected)


@pytest.mark.parametrize("case", ["none", "one", "boundary", "zero_strength"])
def test_small_group_degenerate_cases_keep_weights_one(case):
    args = list(grouped_inputs())
    settings = SDCLConfig(object_scope="small", warmup_epochs=0, ramp_epochs=0,
                          lambda_max=0 if case == "zero_strength" else 0.5)
    if case == "none":
        args[0][0, :4, 2:] = 80
    elif case == "one":
        args[0][0, :2, 2:] = 80
    elif case == "boundary":
        args[0][0, 2, 2:] = 32
    result = object_weights(settings, 0, *args)
    torch.testing.assert_close(result.weights, torch.ones_like(result.weights))
    if case == "boundary":
        assert result.diagnostics["eligible_objects"] == 1


def test_small_membership_uses_reference_scale_after_augmentation():
    args = list(grouped_inputs())
    args[0] = args[0] * 2
    args[-1] = (1280, 1280)
    settings = SDCLConfig(object_scope="small", warmup_epochs=0, ramp_epochs=0)
    original = object_weights(settings, 0, *grouped_inputs())
    resized = object_weights(settings, 0, *args)
    torch.testing.assert_close(original.weights, resized.weights)
    assert resized.diagnostics["eligible_objects"] == 2


@pytest.mark.parametrize("signal", ["joint", "scale", "difficulty", "additive"])
def test_default_all_scope_preserves_original_weight_formula(signal):
    scales = 1 / (1 + (torch.tensor([8.0, 24.0, 80.0]) / 32).square())
    difficulty = torch.tensor([0.55, 0.15, 0.75])
    values = {"joint": scales * difficulty, "scale": scales,
              "difficulty": difficulty, "additive": (scales + difficulty) / 2}[signal]
    mass = torch.tensor([1.2, 0.6, 0.9])
    expected_objects = 1 + 0.5 * (values - (mass * values).sum() / mass.sum())
    expected = torch.cat((expected_objects[[0, 0, 1, 2]], torch.ones(1)))[None]
    result = object_weights(SDCLConfig(signal=signal, warmup_epochs=0, ramp_epochs=0),
                            0, *grouped_inputs())
    torch.testing.assert_close(result.weights, expected)

