from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss

from sdcl.config import SDCLConfig
from sdcl.criterion import SDCLLoss, weighted_classification


def test_classification_zero_weight_loss_and_gradient():
    logits = torch.randn(2, 8, 3, requires_grad=True)
    targets = torch.rand_like(logits)
    actual = weighted_classification(logits, targets, torch.ones(2, 8)).sum()
    expected = F.binary_cross_entropy_with_logits(logits, targets, reduction="sum")
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, logits, retain_graph=True)[0],
        torch.autograd.grad(expected, logits)[0],
    )


def test_ignore_preserves_matched_foreground():
    logits = torch.zeros(1, 3, 2, requires_grad=True)
    targets = torch.zeros_like(logits)
    targets[0, 0, 0] = 0.8
    ignored = torch.tensor([[True, True, False]])
    foreground = torch.tensor([[True, False, False]])
    loss = weighted_classification(logits, targets, torch.ones(1, 3), ignored, foreground)
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
def test_full_loss_zero_strength_matches_upstream(model, empty):
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

