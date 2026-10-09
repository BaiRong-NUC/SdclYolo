from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss

from sdcl.config import SDCLConfig
from sdcl.criterion import SDCLLoss, weighted_classification
from sdcl.weights import WeightResult


@pytest.mark.parametrize("mode", ["positive_term", "full_bce"])
def test_classification_zero_weight_loss_and_gradient(mode):
    logits = torch.randn(2, 8, 3, requires_grad=True)
    targets = torch.rand_like(logits)
    actual = weighted_classification(logits, targets, torch.ones(2, 8), mode=mode).sum()
    expected = F.binary_cross_entropy_with_logits(logits, targets, reduction="sum")
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, logits, retain_graph=True)[0],
        torch.autograd.grad(expected, logits)[0],
    )


def test_positive_weighting_matches_soft_label_formula_and_gradient():
    logits = torch.tensor([[[-1.0, 0.5], [1.0, -0.5]]], requires_grad=True)
    targets = torch.tensor([[[0.8, 0.0], [0.0, 0.4]]])
    weights = torch.tensor([[1.4, 0.6]])
    actual = weighted_classification(logits, targets, weights).sum()
    expected = (
        -weights[..., None] * targets * F.logsigmoid(logits)
        -(1 - targets) * F.logsigmoid(-logits)
    ).sum()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, logits, retain_graph=True)[0],
        torch.autograd.grad(expected, logits)[0],
    )


def test_full_bce_weights_only_true_class_and_preserves_negative_gradients():
    logits = torch.tensor([[[-1.0, 0.5], [1.0, -0.5], [0.2, -0.2]]], requires_grad=True)
    targets = torch.tensor([[[0.8, 0.0], [0.0, 0.4], [0.0, 0.0]]])
    weights = torch.tensor([[1.4, 0.6, 1.3]])
    actual = weighted_classification(logits, targets, weights, mode="full_bce")
    unweighted = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    expected = unweighted.clone()
    expected[0, 0, 0] *= 1.4
    expected[0, 1, 1] *= 0.6
    torch.testing.assert_close(actual, expected)
    actual_gradient = torch.autograd.grad(actual.sum(), logits, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected.sum(), logits, retain_graph=True)[0]
    unweighted_gradient = torch.autograd.grad(unweighted.sum(), logits)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)
    torch.testing.assert_close(actual[targets == 0], unweighted[targets == 0])
    torch.testing.assert_close(actual_gradient[targets == 0], unweighted_gradient[targets == 0])


def test_full_bce_has_stationary_probability_at_soft_target_for_fixed_weights():
    targets = torch.tensor([[[0.2], [0.8]]], dtype=torch.float64)
    weights = torch.tensor([[1.4, 0.6]], dtype=torch.float64)
    logits = torch.logit(targets).requires_grad_()
    loss = weighted_classification(logits, targets, weights, mode="full_bce").sum()
    gradient = torch.autograd.grad(loss, logits, create_graph=True)[0]
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-12, rtol=0)
    curvature = torch.autograd.grad(gradient.sum(), logits)[0]
    assert (curvature > 0).all()
    legacy_loss = weighted_classification(logits, targets, weights).sum()
    legacy_gradient = torch.autograd.grad(legacy_loss, logits)[0]
    assert (legacy_gradient.abs() > 1e-3).all()


def test_unknown_classification_mode_rejected():
    with pytest.raises(ValueError, match="Unknown classification weighting mode"):
        weighted_classification(torch.zeros(1, 1, 1), torch.zeros(1, 1, 1),
                                torch.ones(1, 1), mode="typo")


@pytest.mark.parametrize("mode", ["positive_term", "full_bce"])
def test_ignore_preserves_matched_foreground(mode):
    logits = torch.zeros(1, 3, 2, requires_grad=True)
    targets = torch.zeros_like(logits)
    targets[0, 0, 0] = 0.8
    ignored = torch.tensor([[True, True, False]])
    foreground = torch.tensor([[True, False, False]])
    loss = weighted_classification(logits, targets, torch.ones(1, 3), ignored, foreground, mode=mode)
    assert loss[0, 0].sum() > 0
    assert loss[0, 1].sum() == 0
    assert loss[0, 2].sum() > 0


@pytest.fixture(scope="module")
def model():
    torch.set_num_threads(2)
    model = DetectionModel("yolo11n.yaml", nc=2, verbose=False)
    model.args = SimpleNamespace(**DEFAULT_CFG_DICT)
    model.sdcl_settings = SDCLConfig(lambda_max=0).to_dict()
    model.sdcl_ignore_regions = True
    return model


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("apply_to", ["both", "classification", "regression"])
@pytest.mark.parametrize("mode", ["positive_term", "full_bce"])
@pytest.mark.parametrize("object_scope", ["all", "small"])
def test_full_loss_zero_strength_matches_upstream(model, empty, apply_to, mode, object_scope, monkeypatch):
    monkeypatch.setattr(model, "sdcl_settings", SDCLConfig(
        lambda_max=0, apply_to=apply_to, classification_weighting=mode,
        object_scope=object_scope,
    ).to_dict())
    model.train()
    images = torch.rand(2, 3, 64, 64)
    predictions = model(images)
    batch = {
        "img": images,
        "batch_idx": torch.tensor([]) if empty else torch.tensor([0.0, 1.0]),
        "cls": torch.empty(0, 1) if empty else torch.tensor([[0.0], [1.0]]),
        "bboxes": torch.empty(0, 4) if empty else torch.tensor([[0.3, 0.3, 0.2, 0.2], [0.7, 0.7, 0.2, 0.2]]),
        "ignore_bboxes": torch.empty(0, 4),
        "ignore_batch_idx": torch.empty(0),
    }
    upstream = v8DetectionLoss(model)
    ours = SDCLLoss(model)
    expected = upstream(predictions, batch)[0]
    actual = ours(predictions, batch)[0]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    last_parameter = next(model.model[-1].parameters())
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), last_parameter, retain_graph=True)[0],
        torch.autograd.grad(expected.sum(), last_parameter)[0],
        atol=1e-6, rtol=1e-5,
    )


@pytest.mark.parametrize("apply_to", ["both", "classification", "regression"])
@pytest.mark.parametrize("mode", ["positive_term", "full_bce"])
def test_nonzero_weighting_isolates_loss_branches_and_gradients(model, apply_to, mode, monkeypatch):
    # Fixed non-unit weights isolate branch selection from the weighting algorithm.
    def fixed_weights(config, epoch, boxes, scores, ids, foreground, probabilities, ious, size):
        weights = torch.ones_like(scores[..., 0])
        assert foreground.any()
        if config.strength(epoch) > 0:
            weights[foreground] = 1.4
        return WeightResult(weights, {})

    monkeypatch.setattr("sdcl.criterion.object_weights", fixed_weights)
    model.train()
    torch.manual_seed(5)
    batch = {
        "img": torch.rand(2, 3, 64, 64),
        "batch_idx": torch.tensor([0.0, 1.0]),
        "cls": torch.tensor([[0.0], [1.0]]),
        "bboxes": torch.tensor([[0.3, 0.3, 0.2, 0.2], [0.7, 0.7, 0.4, 0.4]]),
        "ignore_bboxes": torch.empty(0, 4),
        "ignore_batch_idx": torch.empty(0),
    }
    raw = model(batch["img"])
    predictions = {
        **raw,
        "boxes": raw["boxes"].detach().requires_grad_(),
        "scores": raw["scores"].detach().requires_grad_(),
    }
    settings = SDCLConfig(warmup_epochs=0, ramp_epochs=0, classification_weighting=mode)
    monkeypatch.setattr(model, "sdcl_settings", settings.to_dict())
    both = SDCLLoss(model)(predictions, batch)[0]
    monkeypatch.setattr(model, "sdcl_settings", {**settings.to_dict(), "enabled": False})
    baseline = SDCLLoss(model)(predictions, batch)[0]
    monkeypatch.setattr(model, "sdcl_settings", {**settings.to_dict(), "apply_to": apply_to})
    actual = SDCLLoss(model)(predictions, batch)[0]
    if mode == "full_bce":
        legacy_settings = {**settings.to_dict(), "classification_weighting": "positive_term"}
        monkeypatch.setattr(model, "sdcl_settings", legacy_settings)
        legacy = SDCLLoss(model)(predictions, batch)[0]
        assert not torch.allclose(both[1], legacy[1])
        torch.testing.assert_close(both[[0, 2]], legacy[[0, 2]])
    expected = torch.stack([
        (baseline if apply_to == "classification" else both)[0],
        (baseline if apply_to == "regression" else both)[1],
        (baseline if apply_to == "classification" else both)[2],
    ])
    assert not torch.allclose(both[1], baseline[1])
    torch.testing.assert_close(both[[0, 2]], baseline[[0, 2]] * 1.4)
    torch.testing.assert_close(actual, expected)
    inputs = (predictions["scores"], predictions["boxes"])
    for actual_grad, expected_grad in zip(
        torch.autograd.grad(actual.sum(), inputs, retain_graph=True),
        torch.autograd.grad(expected.sum(), inputs),
    ):
        torch.testing.assert_close(actual_grad, expected_grad)

