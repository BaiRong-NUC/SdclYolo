from copy import deepcopy
import io

import pytest
import torch
from ultralytics.utils.tal import TaskAlignedAssigner
from ultralytics.utils.torch_utils import ModelEMA

from sdcl.config import load_coverage_config, load_experiment
from sdcl.coverage import CoverageAssigner, CoverageConfig, repair_ownership
from test_phase import batch_example, make_model


def test_recipe_matches_baseline_and_settings_are_validated():
    baseline = load_experiment("configs/experiments/baseline_960_100.yaml")
    candidate = load_experiment("configs/experiments/coverage_960_100.yaml")
    baseline[0].pop("name")
    candidate[0].pop("name")
    assert baseline == candidate
    assert load_coverage_config("configs/experiments/coverage_960_100.yaml") == CoverageConfig(enabled=True)
    for values in [{"enabled": 1}, {"candidates": 0}, {"max_side": float("nan")},
                   {"minimum_ciou": 0}, {"typo": 1}]:
        with pytest.raises(ValueError):
            CoverageConfig.from_dict(values)


def test_sparse_repair_prioritizes_scarcity_free_points_and_never_starves_donor():
    owners = [0, 0, 0, -1]
    counts = [3, 0, 0, 0]
    rows = [(1, [(0, .9, .8), (3, .3, 0)]),
            (2, [(0, .5, .8)]), (3, [(1, .8, .5)])]
    changes = repair_ownership(owners, counts, rows)
    assert owners == [2, 3, 0, 1]
    assert counts == [1, 1, 1, 1]
    assert len(changes) == 3
    counts.append(0)
    assert repair_ownership(owners, counts, [(4, [(2, 1, 1)])]) == []


def test_disabled_and_warmup_assignment_are_bitwise_identical():
    torch.manual_seed(3)
    anchors = torch.rand(40, 2) * 40
    gt = torch.tensor([[[0., 0., 18., 18.], [9., 9., 30., 30.]]])
    boxes = torch.cat([anchors - 5, anchors + 5], -1)[None]
    scores = torch.rand(1, 40, 2)
    inputs = (scores, boxes, anchors, torch.tensor([[[0.], [1.]]]), gt, torch.ones(1, 2, 1))
    original = TaskAlignedAssigner(topk=5, num_classes=2, alpha=.5, beta=6)
    expected = original(*inputs)
    for config in [CoverageConfig(), CoverageConfig(enabled=True),
                   CoverageConfig(enabled=True, warmup_epochs=0, max_side=.01)]:
        other = CoverageAssigner(topk=5, num_classes=2, alpha=.5, beta=6, coverage=config)
        for first, second in zip(expected, other(*inputs)):
            assert torch.equal(first, second)


def test_repair_uses_real_box_positive_quality_and_protects_donor():
    assigner = CoverageAssigner(coverage=CoverageConfig(enabled=True, warmup_epochs=0))
    assigner.bs = 1
    assigner.image_hw = (128, 128)
    mask = torch.tensor([[[1, 1, 0], [0, 0, 0], [0, 0, 0]]], dtype=torch.int8)
    alignment = torch.tensor([[[1., .8, 0.], [.5, .9, .4], [.8, .9, .7]]])
    overlap = torch.ones_like(alignment)
    anchors = torch.tensor([[6., 6.], [14., 6.], [30., 6.]])
    gt = torch.tensor([[[0., 0., 20., 12.], [3., 3., 9., 9.], [60., 60., 70., 70.]]])
    statistics = assigner._repair(mask, alignment, overlap, anchors, gt, torch.ones(1, 3, 1))
    assert mask[0, 1, 0] == 1 and mask[0, 0, 1] == 1
    assert not mask[0, 2].any()
    assert statistics["rescued_gt"] == 1
    assert statistics["donor_starvation"] == 0
    assert (mask.sum(1) <= 1).all()


def test_enabled_loss_gradients_ema_checkpoint_and_inference_remain_compatible():
    model = make_model()
    model.coverage_settings = CoverageConfig(enabled=True, warmup_epochs=0).to_dict()
    batch = batch_example()
    loss, items = model(batch)
    assert len(loss) == 3 and "phase_loss" not in items
    loss.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert isinstance(model.criterion.assigner, CoverageAssigner)
    assert model.criterion.assigner.statistics["batches"] == 1
    ema = ModelEMA(model)
    ema.update(model)
    stream = io.BytesIO()
    torch.save({"model": ema.ema}, stream)
    stream.seek(0)
    loaded = torch.load(stream, weights_only=False)["model"].float().eval()
    with torch.no_grad():
        output = loaded(batch["img"])
    assert output[0].shape[1] == 7
    assert loaded.coverage_settings == model.coverage_settings
    assert loaded.model[-1].reg_max == 16
    assert torch.isfinite(output[0]).all()


def test_legacy_disabled_loss_is_bitwise_identical():
    first = make_model()
    second = deepcopy(first)
    first.coverage_settings = CoverageConfig().to_dict()
    batch = batch_example()
    a, ai = first(batch)
    b, bi = second(batch)
    assert torch.equal(a, b) and ai.keys() == bi.keys()
