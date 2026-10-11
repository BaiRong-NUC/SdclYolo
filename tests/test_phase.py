from copy import deepcopy
import io

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel

from sdcl.config import SDCLConfig, load_experiment, load_phase_config
from sdcl.phase import (
    PhaseConfig, auxiliary_batch_stats, object_samples, phase_distance,
    sample_features, translate_images,
)
from sdcl.trainer import SDCLDetectionModel


def batch_example():
    return {
        "img": torch.rand(2, 3, 128, 128),
        "batch_idx": torch.tensor([0., 0., 0., 1.]),
        "cls": torch.tensor([[0.], [1.], [2.], [0.]]),
        "bboxes": torch.tensor([[.40, .45, .16, .16], [.49, .45, .16, .16],
                                [.02, .1, .03, .03], [.5, .5, .2, .2]]),
        "ignore_bboxes": torch.tensor([[.34, .45, .03, .3]]),
        "ignore_batch_idx": torch.tensor([0.]),
    }


def make_model(config=PhaseConfig(), device="cpu"):
    model = SDCLDetectionModel("yolo11n.yaml", nc=3, verbose=False).to(device)
    model.args = get_cfg()
    model.sdcl_settings = SDCLConfig(enabled=False).to_dict()
    model.sdcl_ignore_regions = True
    model.phase_settings = config.to_dict()
    model.sdcl_epoch = 20
    return model.train()


def test_config_preserves_baseline_recipe_and_independent_schedule():
    base, sdcl, ignore = load_experiment("configs/experiments/baseline_960_100.yaml")
    new, new_sdcl, new_ignore = load_experiment("configs/experiments/phase_960_100.yaml")
    base.pop("name")
    new.pop("name")
    assert base == new and sdcl == new_sdcl and ignore == new_ignore
    config = load_phase_config("configs/experiments/phase_960_100.yaml")
    assert config.strength(5) == 0 and config.strength(10) == .5 and config.strength(15) == 1
    assert PhaseConfig().strength(100) == 0
    with pytest.raises(ValueError):
        PhaseConfig.from_dict({"typo": True})
    with pytest.raises(ValueError):
        PhaseConfig(weight=float("nan"))


def test_integer_shift_uses_padding_and_never_wraps():
    original = torch.arange(20.).reshape(1, 1, 4, 5)
    shifted = translate_images(original, (1, -1))
    torch.testing.assert_close(shifted[..., :3, 1:], original[..., 1:, :4])
    assert (shifted[..., :, 0] == 114 / 255).all()
    assert (shifted[..., 3, :] == 114 / 255).all()


def test_ownership_masks_neighbors_ignores_and_borders_without_rng_changes():
    batch = batch_example()
    config = PhaseConfig(reference_size=128, border_margin=8)
    rng = torch.get_rng_state()
    groups, stats = object_samples(batch, (3, 1), config)
    assert torch.equal(rng, torch.get_rng_state())
    assert stats["selected_objects"] == 3
    assert stats["valid_objects"] == 3
    assert stats["valid_points"] < 27
    assert not groups[0][2][0, :,].all()
    batch["ignore_bboxes"] = torch.tensor([[.5, .5, 1., 1.]])
    batch["ignore_batch_idx"] = torch.tensor([0.])
    groups, stats = object_samples(batch, (3, 1), config)
    assert stats["valid_objects"] == 1
    assert not groups[0][2].any()


def test_sampling_matches_grid_sample_including_zero_padding():
    feature = torch.randn(1, 7, 5, 8, requires_grad=True)
    points = torch.tensor([[[-3., 1.], [20., 15.], [65., 40.]]])
    actual = sample_features(feature, 0, points, (40, 64))
    grid = points * torch.tensor([2/64, 2/40]) - 1
    expected = F.grid_sample(feature, grid[None], align_corners=False)[0].permute(1, 2, 0)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert torch.isfinite(feature.grad).all()


def test_distance_aligned_translation_and_detached_target():
    original = torch.randn(1, 4, 32, 32, requires_grad=True)
    target = translate_images(original.detach(), (1, 0)).requires_grad_()
    points = torch.tensor([[[80., 80.], [81., 81.], [88., 87.]]])
    groups = [(0, points, torch.ones(1, 3, dtype=torch.bool), torch.tensor([8.]))]
    loss, _, _ = phase_distance(original, target, groups, (8, 0), (256, 256))
    assert abs(float(loss.detach())) < 1e-6
    loss.backward()
    assert original.grad is not None and target.grad is None


def test_auxiliary_bn_stats_restored_even_on_error():
    model = nn.Sequential(nn.BatchNorm2d(3)).train()
    buffers = {k: v.clone() for k, v in model.named_buffers()}
    with pytest.raises(RuntimeError):
        with auxiliary_batch_stats(model):
            model(torch.randn(2, 3, 8, 8))
            raise RuntimeError("probe")
    assert model[0].track_running_stats
    for key, value in model.named_buffers():
        torch.testing.assert_close(value, buffers[key])


def test_disabled_loss_exact_and_legacy_checkpoint_compatible():
    first = make_model()
    second = deepcopy(first)
    del second.phase_settings
    batch = batch_example()
    first_loss, first_items = first(batch)
    second_loss, second_items = DetectionModel.loss(second, batch)
    assert torch.equal(first_loss, second_loss)
    assert first_items.keys() == second_items.keys()
    for key in first_items:
        assert torch.equal(first_items[key], second_items[key])
    assert len(first_loss) == 3


def test_enabled_loss_gradients_validation_ema_and_serialization():
    config = PhaseConfig(enabled=True, warmup_epochs=0, ramp_epochs=0,
                         reference_size=128, border_margin=8, interval=1)
    model = make_model(config)
    control = deepcopy(model)
    control.phase_settings = PhaseConfig().to_dict()
    batch = batch_example()
    loss, items = model(batch)
    control(batch)
    assert len(loss) == 4 and items["phase_loss"] > 0
    assert model.phase_statistics["active_steps"] == 1
    for (key, value), (other_key, other) in zip(model.named_buffers(), control.named_buffers()):
        assert key == other_key
        torch.testing.assert_close(value, other)
    loss.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    from ultralytics.utils.torch_utils import ModelEMA
    ema = ModelEMA(model)
    ema.update(model)
    stream = io.BytesIO()
    torch.save({"model": ema.ema}, stream)
    stream.seek(0)
    loaded = torch.load(stream, weights_only=False)["model"].float().eval()
    val_loss, val_items = loaded(batch)
    assert len(val_loss) == 4 and val_items["phase_loss"] == 0
    assert loaded.phase_settings == config.to_dict()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_deterministic_cuda_sampling_and_enabled_amp_backward():
    before = torch.are_deterministic_algorithms_enabled()
    before_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        config = PhaseConfig(enabled=True, warmup_epochs=0, ramp_epochs=0,
                             reference_size=128, border_margin=8, interval=1)
        model = make_model(config, "cuda")
        batch = {k: v.cuda() for k, v in batch_example().items()}
        with torch.autocast("cuda", dtype=torch.float16):
            loss, items = model(batch)
        loss.sum().backward()
        assert items["phase_loss"] > 0
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    finally:
        torch.use_deterministic_algorithms(before, warn_only=before_warn)
